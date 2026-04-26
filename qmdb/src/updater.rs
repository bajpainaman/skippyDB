use crate::{
    compactor::CompactJob,
    def::{
        is_compactible, DEFAULT_ENTRY_SIZE, IN_BLOCK_IDX_BITS, IN_BLOCK_IDX_MASK, OP_CREATE,
        OP_DELETE, OP_READ, OP_WRITE, SHARD_DIV,
    },
    entryfile::{
        entry::entry_equal, Entry, EntryBuffer, EntryBufferWriter, EntryBz, EntryCache, EntryFile,
    },
    indexer::Indexer,
    tasks::TaskHub,
    utils::ringchannel::Consumer,
    utils::OpRecord,
};
use std::collections::HashMap;
use std::sync::Arc;
use std::thread;

/// Phase 2.4-v2: a read request dispatched to a worker for parallel indexer
/// traversal + HPFile/cache read. Only WRITE ops go through this path;
/// CREATE/DELETE stay on the sequencer (they mutate two entries with
/// interdependent sn_end assignments).
struct ReadRequest {
    orig_idx: usize,
    key_hash: [u8; 32],
    key: Vec<u8>,
    value: Vec<u8>,
    height: i64,
    rec: Option<Box<OpRecord>>,
}

/// Phase 2.4-v2: a worker's response carrying the data needed to build the
/// new entry in the sequencer. `needs_seq_read=true` means the old entry
/// lives in the sequencer's curr_buf, which workers cannot access; the
/// sequencer re-reads it in Phase C.
struct ReadResponse {
    orig_idx: usize,
    key_hash: [u8; 32],
    key: Vec<u8>,
    value: Vec<u8>,
    old_pos: i64,
    old_sn: u64,
    next_key_hash: [u8; 32],
    needs_seq_read: bool,
    rec: Option<Box<OpRecord>>,
}

/// One entry in the sequencer's changeset-ordered queue. Either a parallel
/// read (dispatched to a worker) or a serial op handled entirely by the
/// sequencer (CREATE/DELETE/READ).
enum SeqOp {
    Write { orig_idx: usize },
    Create { key_hash: [u8; 32], key: Vec<u8>, value: Vec<u8>, rec: Option<Box<OpRecord>> },
    Delete { key_hash: [u8; 32], key: Vec<u8>, rec: Option<Box<OpRecord>> },
}

/// Handle to a worker thread. Drops cleanly by closing the request channel
/// and joining the thread.
struct WorkerHandle {
    req_tx: crossbeam_channel::Sender<Vec<ReadRequest>>,
    resp_rx: crossbeam_channel::Receiver<Vec<ReadResponse>>,
    thread: Option<thread::JoinHandle<()>>,
}

impl Drop for WorkerHandle {
    fn drop(&mut self) {
        // Drop sender -> worker's recv returns Err -> worker exits.
        // Replace with a dummy closed channel to force drop of self.req_tx.
        let (dummy, _) = crossbeam_channel::bounded(0);
        let _ = std::mem::replace(&mut self.req_tx, dummy);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

/// Shared read-side context sent to each worker for parallel entry reads.
struct WorkerCtx {
    shard_id: usize,
    indexer: Arc<Indexer>,
    entry_file: Arc<EntryFile>,
    entry_buffer: Arc<EntryBuffer>,
}

pub struct Updater {
    shard_id: usize,
    task_hub: Arc<dyn TaskHub>,
    update_buffer: EntryBufferWriter,
    cache: Arc<EntryCache>,
    entry_file: Arc<EntryFile>,
    indexer: Arc<Indexer>,
    read_entry_buf: Vec<u8>, // its content is only accessed by Updater's functions
    prev_entry_buf: Vec<u8>, // its content is only accessed by Updater's functions
    curr_version: i64,       // will be contained by the new entries
    sn_start: u64,           // increased after compacting old entries
    sn_end: u64,             // increased after appending new entries
    compact_consumer: Consumer<CompactJob>,
    compact_done_pos: i64,
    utilization_div: i64,
    utilization_ratio: i64,
    compact_thres: i64,
    next_task_id_map: HashMap<i64, i64>,
    next_task_id: i64,
    /// Phase 2.4-v2: number of worker threads this shard fans out to for
    /// parallel indexer reads. `1` means the sequencer does everything
    /// inline (the pre-Phase-2.4 behavior). Must be >= 1.
    worker_count: usize,
    /// Spawned worker threads. Empty when `worker_count == 1`.
    workers: Vec<WorkerHandle>,
    /// Scratch buffers reused across parallel dispatches to avoid allocation
    /// in the hot path. Indexed by worker_id.
    req_bufs: Vec<Vec<ReadRequest>>,
    /// Ordered queue of ops for the current task, used by the parallel path
    /// to re-sequence results into changeset order.
    seq_ops: Vec<SeqOp>,
    /// Collected responses from workers, re-sorted to changeset order.
    resp_scratch: Vec<Option<ReadResponse>>,
}

impl Updater {
    pub fn new(
        shard_id: usize,
        task_hub: Arc<dyn TaskHub>,
        update_buffer: EntryBufferWriter,
        entry_file: Arc<EntryFile>,
        indexer: Arc<Indexer>,
        curr_version: i64,
        sn_start: u64,
        sn_end: u64,
        compact_consumer: Consumer<CompactJob>,
        compact_done_pos: i64,
        utilization_div: i64,
        utilization_ratio: i64,
        compact_thres: i64,
        next_task_id: i64,
        workers_per_shard: usize,
    ) -> Self {
        assert!(workers_per_shard >= 1, "workers_per_shard must be >= 1");
        let worker_count = workers_per_shard;
        let entry_buffer_arc = update_buffer.entry_buffer.clone();
        let mut workers = Vec::with_capacity(if worker_count > 1 { worker_count } else { 0 });
        if worker_count > 1 {
            for worker_id in 0..worker_count {
                let ctx = Arc::new(WorkerCtx {
                    shard_id,
                    indexer: indexer.clone(),
                    entry_file: entry_file.clone(),
                    entry_buffer: entry_buffer_arc.clone(),
                });
                let (req_tx, req_rx) = crossbeam_channel::bounded::<Vec<ReadRequest>>(2);
                let (resp_tx, resp_rx) = crossbeam_channel::bounded::<Vec<ReadResponse>>(2);
                let thread = thread::Builder::new()
                    .name(format!("updater-worker-s{}-w{}", shard_id, worker_id))
                    .spawn(move || run_worker(ctx, req_rx, resp_tx))
                    .expect("spawn updater worker");
                workers.push(WorkerHandle {
                    req_tx,
                    resp_rx,
                    thread: Some(thread),
                });
            }
        }
        let req_bufs = (0..worker_count).map(|_| Vec::new()).collect();
        Self {
            shard_id,
            task_hub,
            update_buffer,
            cache: Arc::new(EntryCache::new_uninit()),
            entry_file,
            indexer,
            read_entry_buf: Vec::with_capacity(DEFAULT_ENTRY_SIZE),
            prev_entry_buf: Vec::with_capacity(DEFAULT_ENTRY_SIZE),
            curr_version,
            sn_start,
            sn_end,
            compact_consumer,
            compact_done_pos,
            utilization_div,
            utilization_ratio,
            compact_thres,
            next_task_id_map: HashMap::new(),
            next_task_id,
            worker_count,
            workers,
            req_bufs,
            seq_ops: Vec::new(),
            resp_scratch: Vec::new(),
        }
    }

    fn read_entry(&mut self, shard_id: usize, file_pos: i64) {
        let cache_hit = self.cache.lookup(shard_id, file_pos, |entry_bz| {
            self.read_entry_buf.resize(0, 0);
            self.read_entry_buf.extend_from_slice(entry_bz.bz);
        });
        if cache_hit {
            return;
        }
        let (in_disk, accessed) = self.update_buffer.get_entry_bz_at(file_pos, |entry_bz| {
            self.read_entry_buf.resize(0, 0);
            self.read_entry_buf.extend_from_slice(entry_bz.bz);
        });
        //println!("BB in_disk={} accessed={} file_pos={}", in_disk, accessed, file_pos);
        if accessed {
            let entry_bz = EntryBz {
                bz: &self.read_entry_buf[..],
            };
            let _e = Entry::from_bz(&entry_bz);
            return;
        }
        self.read_entry_buf.resize(DEFAULT_ENTRY_SIZE, 0);
        let ef = &self.entry_file;
        if in_disk {
            let size = ef.read_entry(file_pos, &mut self.read_entry_buf[..]);
            self.read_entry_buf.resize(size, 0);
            if self.read_entry_buf.len() < size {
                ef.read_entry(file_pos, &mut self.read_entry_buf[..]);
            }
        } else {
            panic!("Cannot read the entry");
        }
    }

    // handle out-of-order id
    pub fn run_task_with_ooo_id(&mut self, task_id: i64, next_task_id: i64) {
        // insert them so they are viewed as "ready to run"
        self.next_task_id_map.insert(task_id, next_task_id);
        let mut next_task_id = self.next_task_id;
        // try to step forward in the task_id linked list
        loop {
            if let Some(next) = self.next_task_id_map.remove(&next_task_id) {
                self.run_task(next_task_id);
                next_task_id = next; //follow the linked list
            } else {
                break; // not "ready to run"
            }
        }
        self.next_task_id = next_task_id;
    }

    pub fn run_task(&mut self, task_id: i64) {
        // Phase 2.4-v2 fast path: W=1 must be bitwise-identical to the
        // pre-Phase-2.4 behavior to avoid the branch-overhead regression
        // observed on rewrite/phase2.4-workerpool.
        if self.worker_count == 1 {
            let (cache_for_new_block, end_block) = self.task_hub.check_begin_end(task_id);
            if let Some(cache) = cache_for_new_block {
                self.cache = cache;
            }
            let task_hub = self.task_hub.clone();
            if (task_id & IN_BLOCK_IDX_MASK) == 0 {
                //task_index ==0 and new block start
                self.curr_version = task_id;
            }
            for change_set in &*task_hub.get_change_sets(task_id) {
                change_set.run_in_shard(self.shard_id, |op, key_hash, k, v, r| {
                    self.compare_active_info(r);
                    match op {
                        OP_WRITE => self.write_kv(key_hash, k, v, r),
                        OP_CREATE => self.create_kv(key_hash, k, v, r),
                        OP_DELETE => self.delete_kv(key_hash, k, r),
                        OP_READ => (), //used for debug
                        _ => {
                            panic!("Updater: unsupported operation");
                        }
                    }
                });
                self.curr_version += 1;
            }
            if end_block {
                self.update_buffer
                    .end_block(self.compact_done_pos, self.sn_start, self.sn_end);
            }
            return;
        }
        self.run_task_parallel(task_id);
    }

    /// Phase 2.4-v2 parallel path: only OP_WRITE reads are dispatched to
    /// workers. CREATE/DELETE stay serial because they mutate two entries
    /// with interdependent sn_end values. All `append` calls and indexer
    /// mutations happen on the sequencer thread in changeset order, so
    /// SN assignment / buffer positions / root-hash are identical to W=1.
    fn run_task_parallel(&mut self, task_id: i64) {
        let (cache_for_new_block, end_block) = self.task_hub.check_begin_end(task_id);
        if let Some(cache) = cache_for_new_block {
            self.cache = cache;
        }
        let task_hub = self.task_hub.clone();
        if (task_id & IN_BLOCK_IDX_MASK) == 0 {
            self.curr_version = task_id;
        }
        let shard_id = self.shard_id;
        for change_set in &*task_hub.get_change_sets(task_id) {
            self.seq_ops.clear();
            for buf in &mut self.req_bufs {
                buf.clear();
            }
            let height = self.curr_version >> IN_BLOCK_IDX_BITS;
            // Phase A: walk the changeset; dispatch WRITE reads to workers,
            // queue CREATE/DELETE as serial ops.
            change_set.run_in_shard(shard_id, |op, key_hash, k, v, r| {
                match op {
                    OP_WRITE => {
                        let orig_idx = self.seq_ops.len();
                        self.seq_ops.push(SeqOp::Write { orig_idx });
                        // Partition by a byte that provides entropy but is not
                        // the shard byte. `key_hash[1]` is untouched by shard
                        // routing.
                        let w_id = (key_hash[1] as usize) % self.worker_count;
                        self.req_bufs[w_id].push(ReadRequest {
                            orig_idx,
                            key_hash: *key_hash,
                            key: k.to_vec(),
                            value: v.to_vec(),
                            height,
                            rec: r.map(|b| b.clone()),
                        });
                    }
                    OP_CREATE => {
                        self.seq_ops.push(SeqOp::Create {
                            key_hash: *key_hash,
                            key: k.to_vec(),
                            value: v.to_vec(),
                            rec: r.map(|b| b.clone()),
                        });
                    }
                    OP_DELETE => {
                        self.seq_ops.push(SeqOp::Delete {
                            key_hash: *key_hash,
                            key: k.to_vec(),
                            rec: r.map(|b| b.clone()),
                        });
                    }
                    OP_READ => (),
                    _ => panic!("Updater: unsupported operation"),
                }
            });
            // Phase B: send request batches to workers (swap out to avoid
            // hanging on to the buffer's memory).
            for (w_id, worker) in self.workers.iter().enumerate() {
                let batch = std::mem::take(&mut self.req_bufs[w_id]);
                worker.req_tx.send(batch).expect("worker req_tx");
            }
            // Phase B-collect: gather responses and place by orig_idx.
            self.resp_scratch.clear();
            self.resp_scratch
                .resize_with(self.seq_ops.len(), || None);
            for worker in self.workers.iter() {
                let resps = worker.resp_rx.recv().expect("worker resp_rx");
                for r in resps {
                    let i = r.orig_idx;
                    self.resp_scratch[i] = Some(r);
                }
            }
            // Phase C: build new entries + append in original order.
            for i in 0..self.seq_ops.len() {
                let op = std::mem::replace(&mut self.seq_ops[i], SeqOp::Write { orig_idx: 0 });
                match op {
                    SeqOp::Write { orig_idx } => {
                        let resp = self.resp_scratch[orig_idx]
                            .take()
                            .expect("missing worker response");
                        let rec_ref = resp.rec.as_ref();
                        self.compare_active_info(rec_ref);
                        self.finish_write(resp);
                    }
                    SeqOp::Create { key_hash, key, value, rec } => {
                        let rec_ref = rec.as_ref();
                        self.compare_active_info(rec_ref);
                        self.create_kv(&key_hash, &key, &value, rec_ref);
                    }
                    SeqOp::Delete { key_hash, key, rec } => {
                        let rec_ref = rec.as_ref();
                        self.compare_active_info(rec_ref);
                        self.delete_kv(&key_hash, &key, rec_ref);
                    }
                }
            }
            self.curr_version += 1;
        }
        if end_block {
            self.update_buffer
                .end_block(self.compact_done_pos, self.sn_start, self.sn_end);
        }
    }

    /// Phase C worker for the parallel path. The worker has already done the
    /// indexer walk + entry read in parallel; we just fill in the fields
    /// that must come from the sequencer's monotonic state (sn_end,
    /// curr_version), append to the buffer, and mutate the indexer.
    fn finish_write(&mut self, resp: ReadResponse) {
        let ReadResponse {
            orig_idx: _,
            key_hash,
            key,
            value,
            mut old_pos,
            mut old_sn,
            mut next_key_hash,
            needs_seq_read,
            rec,
        } = resp;
        let rec_ref = rec.as_ref();
        if needs_seq_read {
            // Worker couldn't read because old_pos is in the sequencer's
            // curr_buf (written earlier in this same block). Re-do the
            // indexer walk inline so we get the curr_buf-aware path.
            let height = self.curr_version >> IN_BLOCK_IDX_BITS;
            let mut found_pos: i64 = -1;
            let indexer = self.indexer.clone();
            indexer.for_each_value(height, &key_hash[..], |file_pos| -> bool {
                self.read_entry(self.shard_id, file_pos);
                let old_entry = EntryBz {
                    bz: &self.read_entry_buf[..],
                };
                if old_entry.key() == &key[..] {
                    found_pos = file_pos;
                }
                found_pos >= 0
            });
            if found_pos < 0 {
                panic!(
                    "Write to non-exist key shard_id={} key={:?} key_hash={:?}",
                    self.shard_id, &key, key_hash
                );
            }
            let old_entry = EntryBz {
                bz: &self.read_entry_buf[..],
            };
            old_pos = found_pos;
            old_sn = old_entry.serial_number();
            next_key_hash.copy_from_slice(old_entry.next_key_hash());
        }
        let new_entry = Entry {
            key: &key[..],
            value: &value[..],
            next_key_hash: &next_key_hash[..],
            version: self.curr_version,
            serial_number: self.sn_end,
        };
        let dsn_list: [u64; 1] = [old_sn];
        let new_pos = self.update_buffer.append(&new_entry, &dsn_list[..]);
        self.indexer
            .change_kv(&key_hash[..], old_pos, new_pos, dsn_list[0], self.sn_end);
        self.sn_end += 1;
        if self.is_compactible() {
            self.compact(rec_ref, 0);
        }
    }

    fn write_kv(
        &mut self,
        key_hash: &[u8; 32],
        key: &[u8],
        value: &[u8],
        r: Option<&Box<OpRecord>>,
    ) {
        let height = self.curr_version >> IN_BLOCK_IDX_BITS;
        let mut old_pos = -1;
        let indexer = self.indexer.clone();
        indexer.for_each_value(height, &key_hash[..], |file_pos| -> bool {
            self.read_entry(self.shard_id, file_pos);
            let old_entry = EntryBz {
                bz: &self.read_entry_buf[..],
            };
            if old_entry.key() == key {
                old_pos = file_pos;
            }
            old_pos >= 0 // break if old_pos was assigned
        });
        if old_pos < 0 {
            panic!(
                "Write to non-exist key shard_id={} key={:?} key_hash={:?}",
                self.shard_id, key, key_hash
            );
        }
        let old_entry = EntryBz {
            bz: &self.read_entry_buf[..],
        };
        let new_entry = Entry {
            key,
            value,
            next_key_hash: old_entry.next_key_hash(),
            version: self.curr_version,
            serial_number: self.sn_end,
        };
        let dsn_list: [u64; 1] = [old_entry.serial_number()];
        let new_pos = self.update_buffer.append(&new_entry, &dsn_list[..]);
        self.indexer
            .change_kv(&key_hash[..], old_pos, new_pos, dsn_list[0], self.sn_end);
        self.sn_end += 1;
        if self.is_compactible() {
            // println!("compact when write kv");
            self.compact(r, 0);
        }
    }

    fn delete_kv(&mut self, key_hash: &[u8; 32], key: &[u8], r: Option<&Box<OpRecord>>) {
        let height = self.curr_version >> IN_BLOCK_IDX_BITS;
        let mut del_entry_pos = -1;
        let mut del_entry_sn = 0;
        let mut old_next_key_hash = [0u8; 32];
        let mut prev_k80 = [0u8; 10];
        let mut old_pos = -1;
        let indexer = self.indexer.clone();
        indexer.for_each_adjacent_value(height, &key_hash[..], |k_buf, file_pos| -> bool {
            self.read_entry(self.shard_id, file_pos);
            let entry_bz = EntryBz {
                bz: &self.read_entry_buf[..],
            };
            if del_entry_pos < 0 && entry_bz.key() == key {
                compare_old_entry(r, &entry_bz);
                del_entry_pos = file_pos;
                del_entry_sn = entry_bz.serial_number();
                old_next_key_hash.copy_from_slice(entry_bz.next_key_hash());
            } else if old_pos < 0 && entry_bz.next_key_hash() == key_hash {
                self.prev_entry_buf.clear();
                self.prev_entry_buf
                    .extend_from_slice(&self.read_entry_buf[..]);
                compare_prev_entry(r, &entry_bz);
                prev_k80.copy_from_slice(k_buf);
                old_pos = file_pos;
            }
            // exit loop if del_entry_pos and old_pos were assigned
            del_entry_pos >= 0 && old_pos >= 0
        });
        if del_entry_pos < 0 {
            panic!("Delete non-exist key at id={} key={:?}", del_entry_pos, key);
        }
        if old_pos < 0 {
            panic!("Cannot find prevEntry");
        }
        let prev_entry = EntryBz {
            bz: &self.prev_entry_buf[..],
        };
        let prev_changed = Entry {
            key: prev_entry.key(),
            value: prev_entry.value(),
            next_key_hash: &old_next_key_hash[..],
            version: self.curr_version,
            serial_number: self.sn_end,
        };
        let dsn_list: [u64; 2] = [del_entry_sn, prev_entry.serial_number()];
        compare_prev_changed(r, &prev_changed, &dsn_list[..]);
        let new_pos = self.update_buffer.append(&prev_changed, &dsn_list[..]);

        self.indexer
            .erase_kv(&key_hash[..], del_entry_pos, del_entry_sn);
        self.indexer
            .change_kv(&prev_k80[..], old_pos, new_pos, dsn_list[1], self.sn_end);
        self.sn_end += 1;
    }

    fn create_kv(
        &mut self,
        key_hash: &[u8; 32],
        key: &[u8],
        value: &[u8],
        r: Option<&Box<OpRecord>>,
    ) {
        let height = self.curr_version >> IN_BLOCK_IDX_BITS;
        let mut old_pos = -1;
        let mut prev_k80 = [0u8; 10];
        let indexer = self.indexer.clone();
        indexer.for_each_adjacent_value(height, &key_hash[..], |k_buf, file_pos| -> bool {
            self.read_entry(self.shard_id, file_pos);
            let prev_entry = EntryBz {
                bz: &self.read_entry_buf[..],
            };
            if prev_entry.key_hash() < *key_hash && &key_hash[..] < prev_entry.next_key_hash() {
                compare_prev_entry(r, &prev_entry);
                prev_k80.copy_from_slice(k_buf);
                old_pos = file_pos;
            }
            old_pos >= 0
        });
        if old_pos < 0 {
            indexer.for_each_adjacent_value(height, &key_hash[..], |key, file_pos| -> bool {
                println!("FF key = {:?} file_pos = {}", key, file_pos);
                false
            });
            panic!(
                "Cannot find prevKey when creating shard_id={} key={:?}",
                self.shard_id, key
            );
        }
        let prev_entry = EntryBz {
            bz: &self.read_entry_buf[..],
        };
        let new_entry = Entry {
            key,
            value,
            next_key_hash: prev_entry.next_key_hash(),
            version: self.curr_version,
            serial_number: self.sn_end,
        };
        compare_new_entry(r, &new_entry, &[]);
        let create_pos = self.update_buffer.append(&new_entry, &[]);
        //println!("create_pos:{:?}", create_pos);
        let prev_changed = Entry {
            key: prev_entry.key(),
            value: prev_entry.value(),
            next_key_hash: &key_hash[..],
            version: self.curr_version,
            serial_number: self.sn_end + 1,
        };
        //println!(
        //    "prev_changed:{:?}, {:?}",
        //    prev_changed.version, prev_changed.serial_number
        //);
        let dsn_list: [u64; 1] = [prev_entry.serial_number()];
        compare_prev_changed(r, &prev_changed, &dsn_list[..]);
        let new_pos = self.update_buffer.append(&prev_changed, &dsn_list[..]);
        //println!("new_pos:{:?}", new_pos);
        self.indexer.add_kv(&key_hash[..], create_pos, self.sn_end);
        self.indexer.change_kv(
            &prev_k80[..],
            old_pos,
            new_pos,
            dsn_list[0],
            self.sn_end + 1,
        );
        self.sn_end += 2;
        if self.is_compactible() {
            //println!("compact when create kv");
            self.compact(r, 0);
            self.compact(r, 1);
        }
    }

    pub fn compact(&mut self, r: Option<&Box<OpRecord>>, comp_idx: usize) {
        let (job, kh) = loop {
            //println!("before updater what something from consumer channel");
            let job = self.compact_consumer.consume();
            let e = EntryBz { bz: &job.entry_bz };
            let kh = e.key_hash();

            if self.indexer.key_exists(&kh, job.old_pos, e.serial_number()) {
                break (job, kh);
            }
            self.compact_consumer.send_returned(job);
        };

        let entry_bz = EntryBz { bz: &job.entry_bz };

        compare_dig_entry(r, &entry_bz, comp_idx);

        let new_entry = Entry {
            key: entry_bz.key(),
            value: entry_bz.value(),
            next_key_hash: entry_bz.next_key_hash(),
            version: entry_bz.version(),
            serial_number: self.sn_end,
        };

        let dsn_list = [entry_bz.serial_number()];
        compare_put_entry(r, &new_entry, &dsn_list, comp_idx);

        let new_pos = self.update_buffer.append(&new_entry, &dsn_list);
        self.indexer
            .change_kv(&kh, job.old_pos, new_pos, dsn_list[0], self.sn_end);

        self.sn_end += 1;
        self.sn_start = entry_bz.serial_number() + 1;
        self.compact_done_pos = job.old_pos + entry_bz.len() as i64;

        let job_clone = job.clone();
        //println!("before updater what send something from consumer channel position B");
        self.compact_consumer.send_returned(job_clone);
    }

    fn is_compactible(&self) -> bool {
        is_compactible(
            self.utilization_div,
            self.utilization_ratio,
            self.compact_thres,
            self.indexer.len(self.shard_id),
            self.sn_start,
            self.sn_end,
        )
    }
    fn compare_active_info(&self, rec: Option<&Box<OpRecord>>) {
        if cfg!(feature = "check_rec") {
            _compare_active_info(self, rec);
        }
    }
}

/// Worker thread body. Consumes batches of `ReadRequest` from `req_rx`,
/// performs indexer + entry reads in parallel, and sends `ReadResponse`
/// batches back on `resp_tx`. Exits cleanly when `req_rx` disconnects.
///
/// Concurrency model: this runs simultaneously with other workers on the
/// same shard and (critically) does NOT touch the sequencer's
/// `EntryBufferWriter`, `sn_end`, `sn_start`, or any mutable state. It only
/// reads via shared `Arc` handles to immutable-or-internally-synchronized
/// structures (Indexer has per-unit RwLock, EntryFile reads are disk,
/// EntryBuffer::get_entry_bz takes `&self` and its DashMap is thread-safe).
fn run_worker(
    ctx: Arc<WorkerCtx>,
    req_rx: crossbeam_channel::Receiver<Vec<ReadRequest>>,
    resp_tx: crossbeam_channel::Sender<Vec<ReadResponse>>,
) {
    let mut read_buf: Vec<u8> = Vec::with_capacity(DEFAULT_ENTRY_SIZE);
    while let Ok(batch) = req_rx.recv() {
        let mut out: Vec<ReadResponse> = Vec::with_capacity(batch.len());
        for req in batch {
            let (old_pos, old_sn, next_key_hash, needs_seq_read) =
                worker_read_old_entry(&ctx, &req.key_hash, &req.key, req.height, &mut read_buf);
            out.push(ReadResponse {
                orig_idx: req.orig_idx,
                key_hash: req.key_hash,
                key: req.key,
                value: req.value,
                old_pos,
                old_sn,
                next_key_hash,
                needs_seq_read,
                rec: req.rec,
            });
        }
        if resp_tx.send(out).is_err() {
            return;
        }
    }
}

/// Worker-side equivalent of `Updater::write_kv`'s read-path. Returns
/// `(old_pos, old_sn, next_key_hash, needs_seq_read)`. If the indexer hit
/// points into the sequencer's curr_buf (writer-owned), the worker signals
/// `needs_seq_read=true` and the sequencer will redo the read inline.
fn worker_read_old_entry(
    ctx: &WorkerCtx,
    key_hash: &[u8; 32],
    key: &[u8],
    height: i64,
    read_buf: &mut Vec<u8>,
) -> (i64, u64, [u8; 32], bool) {
    let mut old_pos: i64 = -1;
    let mut old_sn: u64 = 0;
    let mut next_key_hash = [0u8; 32];
    let mut needs_seq_read = false;
    ctx.indexer
        .for_each_value(height, &key_hash[..], |file_pos| -> bool {
            // Worker-side read: no writer curr_buf, no cache (the sequencer
            // owns its per-block cache; passing it to workers would require
            // per-dispatch Arc cloning and is left for a follow-up once the
            // baseline path is proven). The sequencer's curr_buf-aware
            // fallback catches any entries the worker can't see.
            let mut got = false;
            let (in_disk, accessed) = ctx.entry_buffer.get_entry_bz(file_pos, |entry_bz| {
                read_buf.clear();
                read_buf.extend_from_slice(entry_bz.bz);
                got = true;
            });
            if !got {
                if !in_disk && !accessed {
                    // The entry is in the writer's curr_buf region but
                    // curr_buf wasn't handed in. Signal the sequencer.
                    needs_seq_read = true;
                    return true;
                }
                if in_disk {
                    read_buf.resize(DEFAULT_ENTRY_SIZE, 0);
                    let size = ctx.entry_file.read_entry(file_pos, &mut read_buf[..]);
                    if read_buf.len() < size {
                        read_buf.resize(size, 0);
                        ctx.entry_file.read_entry(file_pos, &mut read_buf[..]);
                    } else {
                        read_buf.resize(size, 0);
                    }
                } else {
                    // should not happen per get_entry_bz contract
                    needs_seq_read = true;
                    return true;
                }
            }
            let entry_bz = EntryBz { bz: &read_buf[..] };
            if entry_bz.key() == key {
                old_pos = file_pos;
                old_sn = entry_bz.serial_number();
                next_key_hash.copy_from_slice(entry_bz.next_key_hash());
                return true;
            }
            false
        });
    if old_pos < 0 && !needs_seq_read {
        // No match and no curr_buf hit — fall through to sequencer which
        // will panic with the normal "write to non-exist key" message. We
        // still flag needs_seq_read so the panic location is unchanged.
        needs_seq_read = true;
    }
    // Suppress unused-var for shard_id (kept for future per-shard metrics).
    let _ = ctx.shard_id;
    (old_pos, old_sn, next_key_hash, needs_seq_read)
}

fn _compare_active_info(updater: &Updater, rec: Option<&Box<OpRecord>>) {
    if let Some(rec) = rec {
        let num_active = updater.indexer.len(updater.shard_id);
        assert_eq!(rec.num_active, num_active, "incorrect num_active");
        assert_eq!(rec.oldest_active_sn, updater.sn_start, "incorrect sn_start");
    }
}

fn _compare_old_entry(rec: Option<&Box<OpRecord>>, entry_bz: &EntryBz) {
    if let Some(rec) = rec {
        let v = rec.rd_list.last().unwrap();
        assert_eq!(&v[..], entry_bz.bz, "compare_old_entry failed");
    }
}

fn _compare_prev_entry(rec: Option<&Box<OpRecord>>, entry_bz: &EntryBz) {
    if let Some(rec) = rec {
        let v = rec.rd_list.first().unwrap();
        //if &v[..] != entry_bz.bz {
        //    let e = EntryBz { bz: &v[..] };
        //    println!("ref k={} v={}", hex::encode(e.key()), hex::encode(e.value()));
        //    println!("ref sn={:#016x} ver={:#016x}", e.serial_number(), e.version());
        //    println!("imp k={} v={}", hex::encode(entry_bz.key()), hex::encode(entry_bz.value()));
        //    println!("imp sn={:#016x} ver={:#016x}", entry_bz.serial_number(), entry_bz.version());
        //}
        assert_eq!(&v[..], entry_bz.bz, "compare_prev_entry failed");
    }
}

fn _compare_prev_changed(rec: Option<&Box<OpRecord>>, entry: &Entry, dsn_list: &[u64]) {
    if let Some(rec) = rec {
        let v = rec.wr_list.first().unwrap();
        let equal = entry_equal(&v[..], entry, dsn_list);
        if !equal {
            let tmp = EntryBz { bz: &v[..] };
            let r = Entry::from_bz(&tmp);
            let key_hash = tmp.key_hash();
            let shard_id = key_hash[0] as usize * 256 / SHARD_DIV;
            println!(
                "AA cmpr prev_C shard_id={}\nref={:?}\nimp={:?}\ndsn_list={:?}",
                shard_id, r, entry, dsn_list
            );
            for (_, sn) in tmp.dsn_iter() {
                println!("--{}", sn);
            }
        }
        assert!(equal, "compare_prev_changed failed");
    }
}

fn _compare_new_entry(rec: Option<&Box<OpRecord>>, entry: &Entry, dsn_list: &[u64]) {
    if let Some(rec) = rec {
        let v = rec.wr_list.last().unwrap();
        let equal = entry_equal(&v[..], entry, dsn_list);
        assert!(equal, "compare_new_entry failed");
    }
}

fn _compare_dig_entry(rec: Option<&Box<OpRecord>>, entry_bz: &EntryBz, comp_idx: usize) {
    if let Some(rec) = rec {
        let v = rec.dig_list.get(comp_idx).unwrap();
        if &v[..] != entry_bz.bz {
            let tmp = EntryBz { bz: &v[..] };
            let r = Entry::from_bz(&tmp);
            let i = Entry::from_bz(entry_bz);
            let key_hash = entry_bz.key_hash();
            let shard_id = key_hash[0] >> 4;
            println!(
                "AA cmpr dig_E shard_id={}\nref={:?}\nimp={:?}\nref={:?}\nimp={:?}",
                shard_id,
                r,
                i,
                &v[..],
                entry_bz.bz
            );
        }
        assert_eq!(&v[..], entry_bz.bz, "compare_dig_entry failed");
    }
}

fn _compare_put_entry(
    rec: Option<&Box<OpRecord>>,
    entry: &Entry,
    dsn_list: &[u64],
    comp_idx: usize,
) {
    if let Some(rec) = rec {
        let v = rec.put_list.get(comp_idx).unwrap();
        assert!(
            entry_equal(&v[..], entry, dsn_list),
            "compare_put_entry failed"
        );
    }
}

fn compare_old_entry(rec: Option<&Box<OpRecord>>, entry_bz: &EntryBz) {
    if cfg!(feature = "check_rec") {
        _compare_old_entry(rec, entry_bz)
    }
}

fn compare_prev_entry(rec: Option<&Box<OpRecord>>, entry_bz: &EntryBz) {
    if cfg!(feature = "check_rec") {
        _compare_prev_entry(rec, entry_bz)
    }
}

fn compare_prev_changed(rec: Option<&Box<OpRecord>>, entry: &Entry, dsn_list: &[u64]) {
    if cfg!(feature = "check_rec") {
        _compare_prev_changed(rec, entry, dsn_list);
    }
}

fn compare_new_entry(rec: Option<&Box<OpRecord>>, entry: &Entry, dsn_list: &[u64]) {
    if cfg!(feature = "check_rec") {
        _compare_new_entry(rec, entry, dsn_list);
    }
}

fn compare_dig_entry(rec: Option<&Box<OpRecord>>, entry_bz: &EntryBz, comp_idx: usize) {
    if cfg!(feature = "check_rec") {
        _compare_dig_entry(rec, entry_bz, comp_idx);
    }
}

fn compare_put_entry(
    rec: Option<&Box<OpRecord>>,
    entry: &Entry,
    dsn_list: &[u64],
    comp_idx: usize,
) {
    if cfg!(feature = "check_rec") {
        _compare_put_entry(rec, entry, dsn_list, comp_idx);
    }
}

#[cfg(test)]
mod updater_tests {
    use std::vec;

    use crate::{
        entryfile::{entry::entry_to_bytes, entrybuffer, EntryFileWriter},
        tasks::BlockPairTaskHub,
        test_helper::{to_k80, SimpleTask, TempDir},
        utils::ringchannel::{self, Producer},
    };

    use super::*;

    fn new_updater(dir: &str) -> (TempDir, Updater, Producer<CompactJob>) {
        let temp_dir = TempDir::new(dir);
        let (entry_buffer_w, _entry_buffer_r) = entrybuffer::new(8, 1024);
        let cache_arc = Arc::new(EntryCache::new());
        let entry_file_arc = Arc::new(EntryFile::new(
            512,
            2048,
            dir.to_string(),
            cfg!(feature = "directio"),
            None,
        ));
        let btree_arc = Arc::new(Indexer::new(16));
        let job = CompactJob {
            old_pos: 0,
            entry_bz: Vec::new(),
        };
        let (producer, consumer) = ringchannel::new(100, &job);
        let updater = Updater {
            shard_id: 0,
            task_hub: Arc::new(BlockPairTaskHub::<SimpleTask>::new()),
            update_buffer: entry_buffer_w,
            cache: cache_arc,
            entry_file: entry_file_arc,
            indexer: btree_arc,
            read_entry_buf: vec![0u8; 1024],
            prev_entry_buf: vec![0u8; 1024],
            curr_version: 0,
            sn_start: 0,
            sn_end: 0,
            compact_done_pos: 0,
            utilization_div: 10,
            utilization_ratio: 7,
            compact_thres: 8,
            next_task_id_map: HashMap::new(),
            next_task_id: 0,
            compact_consumer: consumer,
            worker_count: 1,
            workers: Vec::new(),
            req_bufs: Vec::new(),
            seq_ops: Vec::new(),
            resp_scratch: Vec::new(),
        };
        (temp_dir, updater, producer)
    }

    fn new_test_entry<'a>() -> Entry<'a> {
        Entry {
            key: "key".as_bytes(),
            value: "value".as_bytes(),
            next_key_hash: &[0xab; 32],
            version: 12345,
            serial_number: 99999,
        }
    }

    fn append_and_flush_entry_to_file(
        entry_file: Arc<EntryFile>,
        entry: &Entry,
        dsn_list: &[u64],
    ) -> i64 {
        let mut w = EntryFileWriter::new(entry_file.clone(), 512);
        let mut entry_bz = [0u8; 512];
        let _entry_size = entry.dump(&mut entry_bz, dsn_list);
        let pos = w.append(&EntryBz { bz: &entry_bz[..] }).unwrap();
        let _ = w.flush();
        pos
    }

    fn put_entry_in_cache(updater: &Updater, file_pos: i64, entry: &Entry, dsn_list: &[u64]) {
        let mut entry_buf = [0u8; 1024];
        let entry_size = entry.dump(&mut entry_buf[..], dsn_list);
        let entry_bz = EntryBz {
            bz: &entry_buf[..entry_size],
        };
        updater.cache.insert(updater.shard_id, file_pos, &entry_bz);
    }

    #[test]
    fn test_read_entry_cache_hit() {
        let (_dir, mut updater, _producer) = new_updater("test_read_entry_cache_hit");

        let entry = new_test_entry();
        let dsn_list = [1, 2, 3, 4];
        put_entry_in_cache(&updater, 123, &entry, &dsn_list);

        updater.read_entry(updater.shard_id, 123);
        assert_eq!(
            "03050000046b657976616c7565ababababababab",
            hex::encode(&updater.read_entry_buf[0..20])
        );
    }

    #[test]
    fn test_read_entry_from_buffer() {
        let (_dir, mut updater, _producer) = new_updater("test_read_entry_from_buffer");
        let entry = new_test_entry();
        let dsn_list = [1, 2, 3, 4];
        let pos = updater.update_buffer.append(&entry, &dsn_list);

        updater.read_entry(7, pos);
        assert_eq!(
            "03050000046b657976616c7565ababababababab",
            hex::encode(&updater.read_entry_buf[0..20])
        );
    }

    #[test]
    fn test_read_entry_from_file() {
        let (_dir, mut updater, _producer) = new_updater("test_read_entry_from_file");
        let entry = new_test_entry();
        let dsn_list = [1, 2, 3, 4];
        let pos = append_and_flush_entry_to_file(updater.entry_file.clone(), &entry, &dsn_list);

        updater.read_entry(7, pos);
        assert_eq!(
            "03050000046b657976616c7565ababababababab",
            hex::encode(&updater.read_entry_buf[0..20])
        );
    }

    #[test]
    #[should_panic(expected = "incorrect num_active")]
    fn test_compare_active_info1() {
        let (_dir, updater, _producer) = new_updater("test_compare_active_info1");
        let mut op = Box::new(OpRecord::new(0));
        op.num_active = 123;
        let rec = Option::Some(&op);
        _compare_active_info(&updater, rec);
    }

    #[test]
    #[should_panic(expected = "incorrect sn_start")]
    fn test_compare_active_info2() {
        let (_dir, updater, _producer) = new_updater("test_compare_active_info2");
        let mut op = Box::new(OpRecord::new(0));
        op.oldest_active_sn = 123;
        let rec = Option::Some(&op);
        _compare_active_info(&updater, rec);
    }

    #[test]
    #[should_panic(expected = "Cannot find prevKey when creating shard_id=0 key=[107, 101, 121]")]
    fn test_create_kv_non_exist_key() {
        let (_dir, mut updater, _producer) = new_updater("test_create_kv_non_exist_key");
        updater.create_kv(
            &[5u8; 32],
            "key".as_bytes(),
            "value".as_bytes(),
            Option::None,
        );
    }

    #[test]
    fn test_create_kv() {
        let (_dir, mut updater, _producer) = new_updater("test_create_kv");

        let entry = new_test_entry();
        let dsn_list = [];
        let pos = append_and_flush_entry_to_file(updater.entry_file.clone(), &entry, &dsn_list);

        updater
            .indexer
            .add_kv(&to_k80(0x7777_0000_0000_0000), pos, 0);
        assert_eq!(1, updater.indexer.len(0));
        assert_eq!(0, updater.sn_end);

        updater.create_kv(
            &[0x77u8; 32],
            "key".as_bytes(),
            "value".as_bytes(),
            Option::None,
        );

        assert_eq!(2, updater.indexer.len(0));
        assert_eq!(2, updater.sn_end);
        // TODO: check more
    }

    #[test]
    #[should_panic(expected = "Cannot find prevKey when creating shard_id=0 key=[107, 101, 121]")]
    fn test_write_kv_non_exist_key() {
        let (_dir, mut updater, _producer) = new_updater("test_write_kv_non_exist_key");
        updater.create_kv(
            &[5u8; 32],
            "key".as_bytes(),
            "value".as_bytes(),
            Option::None,
        );
    }

    #[test]
    fn test_write_kv() {
        let (_dir, mut updater, _producer) = new_updater("test_write_kv");
        let entry = new_test_entry();
        let dsn_list = [];
        let pos = append_and_flush_entry_to_file(updater.entry_file.clone(), &entry, &dsn_list);

        updater
            .indexer
            .add_kv(&to_k80(0x7777_0000_0000_0000), pos, 0);
        updater.create_kv(
            &[0x77u8; 32],
            "key".as_bytes(),
            "value".as_bytes(),
            Option::None,
        );
        assert_eq!(2, updater.indexer.len(0));
        assert_eq!(2, updater.sn_end);

        updater.write_kv(
            &[0x77u8; 32],
            "key".as_bytes(),
            "val2".as_bytes(),
            Option::None,
        );
        assert_eq!(2, updater.indexer.len(0));
        assert_eq!(3, updater.sn_end);
        // TODO: check more
    }

    #[test]
    #[should_panic(expected = "Delete non-exist key")]
    fn test_delete_kv_non_exist_key() {
        let (_dir, mut updater, _producer) = new_updater("test_delete_kv_non_exist_key");
        updater.delete_kv(&[3u8; 32], "key".as_bytes(), Option::None);
    }

    #[test]
    #[should_panic(expected = "Cannot find prevEntry")]
    fn test_delete_kv_no_prev_entry() {
        let (_dir, mut updater, _producer) = new_updater("test_delete_kv_no_prev_entry");

        let entry = new_test_entry();
        let dsn_list = [];
        let pos = append_and_flush_entry_to_file(updater.entry_file.clone(), &entry, &dsn_list);
        updater
            .indexer
            .add_kv(&to_k80(0x7777_7777_7777_7777), pos, 0);

        updater.delete_kv(&[0x77u8; 32], "key".as_bytes(), Option::None);
    }

    #[test]
    fn test_delete_kv() {
        let (_dir, mut updater, _producer) = new_updater("test_delete_kv");
        let entry = new_test_entry();
        let dsn_list = [];
        let pos1 = append_and_flush_entry_to_file(updater.entry_file.clone(), &entry, &dsn_list);
        updater
            .indexer
            .add_kv(&to_k80(0x7777_2000_0000_0000), pos1, 0);
        updater.create_kv(
            &[0x77u8; 32],
            "key".as_bytes(),
            "value".as_bytes(),
            Option::None,
        );
        assert_eq!(2, updater.indexer.len(0));
        assert_eq!(2, updater.sn_end);

        let entry2 = Entry {
            key: "key2".as_bytes(),
            value: "val2".as_bytes(),
            next_key_hash: &[0x77u8; 32],
            version: 12345,
            serial_number: 100000,
        };
        let pos2: i64 =
            append_and_flush_entry_to_file(updater.entry_file.clone(), &entry2, &dsn_list);
        put_entry_in_cache(&updater, pos2, &entry2, &dsn_list);
        updater
            .indexer
            .add_kv(&to_k80(0x7777_3000_0000_0000), pos2, 0);
        assert_eq!(3, updater.indexer.len(0));
        assert_eq!(2, updater.sn_end);

        updater.delete_kv(&[0x77u8; 32], "key".as_bytes(), Option::None);
        assert_eq!(2, updater.indexer.len(0));
        assert_eq!(3, updater.sn_end);
        // TODO: check more
    }

    #[test]
    fn test_is_compactible() {
        // utilization: 60%
        let (_dir, mut updater, _producer) = new_updater("test_is_compactible");
        updater.sn_start = 0;
        updater.sn_end = 20;
        updater.compact_thres = 10;

        for i in 0..20 {
            updater.indexer.add_kv(&to_k80(i), (i * 8) as i64, 0);
            assert_eq!(8 < i && i < 14, updater.is_compactible());
        }

        updater.sn_end = 40;
        assert!(updater.is_compactible());

        updater.compact_thres = 41;
        assert!(!updater.is_compactible());
    }

    #[test]
    fn test_try_compact() {
        let (_dir, mut updater, mut producer) = new_updater("test_try_compact");
        let entry = new_test_entry();
        let dsn_list = [0u64; 0];
        let mut entry_buf = [0u8; 500];
        let entry_bz = entry_to_bytes(&entry, &dsn_list, &mut entry_buf);
        let pos = append_and_flush_entry_to_file(updater.entry_file.clone(), &entry, &dsn_list);
        let kh = entry_bz.key_hash();
        updater.indexer.add_kv(&kh[..], pos, 0);
        updater.sn_end = 10;
        updater.compact_thres = 0;
        updater.utilization_ratio = 1;
        assert!(updater.is_compactible());
        assert_eq!(1, updater.indexer.len(0));
        assert_eq!(10, updater.sn_end);

        producer
            .produce(CompactJob {
                old_pos: 0,
                entry_bz: entry_buf.to_vec(),
            })
            .unwrap();
        producer.receive_returned().unwrap();

        updater.compact(Option::None, 0);
        assert_eq!(1, updater.indexer.len(0));
        assert_eq!(11, updater.sn_end);
        // TODO: check mores
    }

    #[test]
    fn test_run_task() {
        // todo
    }
}

#[cfg(test)]
mod compare_tests {
    use super::*;
    use crate::{
        entryfile::{Entry, EntryBz},
        test_helper::EntryBuilder,
        utils::OpRecord,
    };

    fn new_test_entry<'a>() -> Entry<'a> {
        Entry {
            key: "key".as_bytes(),
            value: "value".as_bytes(),
            next_key_hash: &[0xab; 32],
            version: 12345,
            serial_number: 99999,
        }
    }

    #[test]
    #[should_panic(expected = "compare_old_entry failed")]
    fn test_compare_old_entry() {
        let mut op = Box::new(OpRecord::new(0));
        op.rd_list.push(vec![4, 5, 6]);
        op.rd_list.push(vec![1, 2, 3]);
        let rec = Option::Some(&op);
        let bz: [u8; 3] = [4, 5, 6];
        _compare_old_entry(rec, &EntryBz { bz: &bz[..] });
    }

    #[test]
    #[should_panic(expected = "compare_prev_entry failed")]
    fn test_compare_prev_entry() {
        let mut op = Box::new(OpRecord::new(0));
        op.rd_list.push(vec![1, 2, 3]);
        op.rd_list.push(vec![4, 5, 6]);
        let rec = Option::Some(&op);
        let bz: [u8; 3] = [4, 5, 6];
        _compare_prev_entry(rec, &EntryBz { bz: &bz[..] });
    }

    #[test]
    #[should_panic(expected = "compare_prev_changed failed")]
    fn test_compare_prev_changed() {
        let entry = new_test_entry();
        let dsn_list: [u64; 4] = [1, 2, 3, 4];

        let mut op = Box::new(OpRecord::new(0));
        op.wr_list
            .push(EntryBuilder::kv("abc", "def").build_and_dump(&[]));
        op.wr_list.push(vec![4, 5, 6]);
        let rec = Option::Some(&op);
        _compare_prev_changed(rec, &entry, &dsn_list);
    }

    #[test]
    #[should_panic(expected = "compare_new_entry failed")]
    fn test_compare_new_entry() {
        let entry = new_test_entry();
        let dsn_list: [u64; 4] = [1, 2, 3, 4];

        let mut op = Box::new(OpRecord::new(0));
        op.wr_list.push(vec![1, 2, 3]);
        op.wr_list.push(vec![4, 5, 6]);
        let rec = Option::Some(&op);
        _compare_new_entry(rec, &entry, &dsn_list);
    }

    #[test]
    #[should_panic(expected = "compare_dig_entry failed")]
    fn test_compare_dig_entry() {
        let entry1 = EntryBuilder::kv("abc", "def").build_and_dump(&[]);
        let entry2 = EntryBuilder::kv("hhh", "www").build_and_dump(&[]);
        let entry3 = EntryBuilder::kv("123", "456").build_and_dump(&[]);

        let mut op = Box::new(OpRecord::new(0));
        op.dig_list.push(entry1);
        op.rd_list.push(entry2);
        let rec = Option::Some(&op);
        _compare_dig_entry(rec, &EntryBz { bz: &entry3 }, 0);
    }

    #[test]
    #[should_panic(expected = "compare_put_entry failed")]
    fn test_compare_put_entry() {
        let entry = new_test_entry();
        let dsn_list: [u64; 4] = [1, 2, 3, 4];

        let mut op = Box::new(OpRecord::new(0));
        op.put_list.push(vec![1, 2, 3]);
        op.put_list.push(vec![4, 5, 6]);
        let rec = Option::Some(&op);
        _compare_put_entry(rec, &entry, &dsn_list, 1);
    }
}
