use crate::def::{NONCE_SIZE, PRUNE_EVERY_NBLOCKS, SHARD_COUNT, TAG_SIZE, TWIG_SHIFT};
use aes_gcm::aead::AeadInPlace;
use aes_gcm::Aes256Gcm;
use byteorder::{ByteOrder, LittleEndian};
use dashmap::DashMap;
use log::warn;
use std::{fs, path::Path, sync::Arc};

#[cfg(feature = "in_sp1")]
use hpfile::file::File;
#[cfg(not(feature = "in_sp1"))]
use std::{
    fs::File,
    io::{Read, Write},
    os::unix::fs::FileExt,
};

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
pub struct MetaInfo {
    pub curr_height: i64,
    pub last_pruned_twig: [(u64, i64); SHARD_COUNT],
    pub next_serial_num: [u64; SHARD_COUNT],
    pub oldest_active_sn: [u64; SHARD_COUNT],
    pub oldest_active_file_pos: [i64; SHARD_COUNT],
    pub root_hash: [[u8; 32]; SHARD_COUNT],
    pub root_hash_by_height: Vec<[u8; 32]>,
    pub edge_nodes: [Vec<u8>; SHARD_COUNT],
    pub twig_file_sizes: [i64; SHARD_COUNT],
    pub entry_file_sizes: [i64; SHARD_COUNT],
    pub first_twig_at_height: [(u64, i64); SHARD_COUNT],
    pub extra_data: String,
}

impl MetaInfo {
    fn new() -> Self {
        Self {
            curr_height: 0,
            last_pruned_twig: [(0, 0); SHARD_COUNT],
            next_serial_num: [0; SHARD_COUNT],
            oldest_active_sn: [0; SHARD_COUNT],
            oldest_active_file_pos: [0; SHARD_COUNT],
            root_hash: [[0; 32]; SHARD_COUNT],
            root_hash_by_height: vec![],
            edge_nodes: Default::default(),
            twig_file_sizes: [0; SHARD_COUNT],
            entry_file_sizes: [0; SHARD_COUNT],
            first_twig_at_height: [(0, 0); SHARD_COUNT],
            extra_data: "".to_owned(),
        }
    }
}

pub struct MetaDB {
    info: MetaInfo,
    meta_file_name: String,
    history_file: File,
    extra_data_map: Arc<DashMap<i64, String>>,
    cipher: Option<Aes256Gcm>,
    /// Handle to the last async commit's background write thread.
    /// Ensures previous I/O completes before the next commit starts.
    pending_write: Option<std::thread::JoinHandle<()>>,
}

/// Errors that `MetaDB::with_dir_checked` can surface. Introduced in Phase 2.1
/// to stake out the contract for cross-topology reopen detection; Phase 2.2
/// adds the on-disk detection logic (via a `MetaInfoV2` version envelope that
/// records the shard count the DB was built against).
#[derive(Debug, PartialEq, Eq)]
pub enum MetaDbError {
    /// The on-disk MetaInfo was built against `got` shards but the caller
    /// asked for `expected`. Reopening would silently corrupt per-shard
    /// bookkeeping (serial numbers, entry-file sizes, root hashes, …).
    ShardCountMismatch { expected: usize, got: usize },
}

impl std::fmt::Display for MetaDbError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MetaDbError::ShardCountMismatch { expected, got } => write!(
                f,
                "MetaDB shard-count mismatch: expected {}, on-disk claims {}",
                expected, got
            ),
        }
    }
}

impl std::error::Error for MetaDbError {}

fn get_file_as_byte_vec(filename: &String) -> Vec<u8> {
    let mut f = File::open(filename).expect("no file found");
    let metadata = fs::metadata(filename).expect("unable to read metadata");
    let mut buffer = vec![0; metadata.len() as usize];
    f.read(&mut buffer).expect("buffer overflow");

    buffer
}

impl MetaDB {
    /// Like [`Self::with_dir`], but validates that the on-disk MetaInfo was
    /// built against `expected_shard_count`. Phase 2.x production code paths
    /// that are prepared for a shard-count mismatch should call this instead
    /// of `with_dir`.
    ///
    /// **Phase 2.1 (today):** wraps `with_dir` and always returns `Ok`. The
    /// failing integration test at `tests/metadb_shard_count_mismatch.rs`
    /// documents the intended contract.
    ///
    /// **Phase 2.2:** adds the `MetaInfoV2` version envelope that records the
    /// DB's shard count and makes this method reject mismatches with
    /// `Err(MetaDbError::ShardCountMismatch)`.
    pub fn with_dir_checked(
        dir_name: &str,
        cipher: Option<Aes256Gcm>,
        expected_shard_count: usize,
    ) -> Result<Self, MetaDbError> {
        let _ = expected_shard_count; // suppressed until Phase 2.2 wires detection.
        Ok(Self::with_dir(dir_name, cipher))
    }

    pub fn with_dir(dir_name: &str, cipher: Option<Aes256Gcm>) -> Self {
        let meta_file_name = format!("{}/info", dir_name);
        let file_name = format!("{}/prune_helper", dir_name);
        if !Path::new(dir_name).exists() {
            fs::create_dir(dir_name).unwrap();
        }
        let history_file = File::options()
            .read(true)
            .write(true)
            .create(true)
            .open(file_name)
            .expect("no file found");
        let mut res = Self {
            info: MetaInfo::new(),
            meta_file_name,
            history_file,
            extra_data_map: Arc::new(DashMap::new()),
            cipher,
            pending_write: None,
        };
        res.reload_from_file();
        res
    }

    pub fn reload_from_file(&mut self) {
        let mut name = format!("{}.0", self.meta_file_name);
        if Path::new(&name).exists() {
            let mut meta_info_bz = get_file_as_byte_vec(&name);
            if self.cipher.is_some() {
                Self::decrypt(&self.cipher, &mut meta_info_bz);
                let size = meta_info_bz.len();
                meta_info_bz = meta_info_bz[8..size - TAG_SIZE].to_owned();
            }
            match bincode::deserialize::<MetaInfo>(&meta_info_bz[..]) {
                Ok(info) => self.info = info,
                Err(_) => warn!("Failed to deserialize {}, ignore it", name),
            };
        }
        name = format!("{}.1", self.meta_file_name);
        if Path::new(&name).exists() {
            let mut meta_info_bz = get_file_as_byte_vec(&name);
            if self.cipher.is_some() {
                Self::decrypt(&self.cipher, &mut meta_info_bz);
                let size = meta_info_bz.len();
                meta_info_bz = meta_info_bz[8..size - TAG_SIZE].to_owned();
            }
            match bincode::deserialize::<MetaInfo>(&meta_info_bz[..]) {
                Ok(info) => {
                    if info.curr_height > self.info.curr_height {
                        self.info = info; //pick the latest one
                    }
                }
                Err(_) => warn!("Failed to deserialize {}, ignore it", name),
            };
        }
    }

    fn decrypt(cipher: &Option<Aes256Gcm>, bz: &mut [u8]) {
        if bz.len() < TAG_SIZE + 8 {
            panic!("meta db file size not correct")
        }
        let cipher = (*cipher).as_ref().unwrap();
        let mut nonce_arr = [0u8; NONCE_SIZE];
        nonce_arr[..8].copy_from_slice(&bz[0..8]);
        let tag_start = bz.len() - TAG_SIZE;
        let mut tag = [0u8; TAG_SIZE];
        tag[..].copy_from_slice(&bz[tag_start..]);
        let payload = &mut bz[8..tag_start];
        if let Err(e) =
            cipher.decrypt_in_place_detached(&nonce_arr.into(), b"", payload, &tag.into())
        {
            panic!("{:?}", e)
        };
    }

    pub fn get_extra_data(&self) -> String {
        self.info.extra_data.clone()
    }

    pub fn insert_extra_data(&mut self, height: i64, data: String) {
        self.extra_data_map.insert(height, data);
    }

    pub fn commit(&mut self) -> Arc<MetaInfo> {
        self.wait_for_pending_write();
        let kv = self.extra_data_map.remove(&self.info.curr_height).unwrap();
        self.info.extra_data = kv.1;
        let name = format!("{}.{}", self.meta_file_name, self.info.curr_height % 2);
        let mut bz = bincode::serialize(&self.info).unwrap();
        if self.cipher.is_some() {
            let cipher = self.cipher.as_ref().unwrap();
            let mut nonce_arr = [0u8; NONCE_SIZE];
            LittleEndian::write_i64(&mut nonce_arr[..8], self.info.curr_height);
            match cipher.encrypt_in_place_detached(&nonce_arr.into(), b"", &mut bz) {
                Err(err) => panic!("{}", err),
                Ok(tag) => {
                    let mut out = vec![];
                    out.extend_from_slice(&nonce_arr[0..8]);
                    out.extend_from_slice(&bz);
                    out.extend_from_slice(tag.as_slice());
                    fs::write(&name, out).unwrap();
                }
            };
        } else {
            fs::write(&name, bz).unwrap();
        }
        if self.info.curr_height % PRUNE_EVERY_NBLOCKS == 0 && self.info.curr_height > 0 {
            let mut data = [0u8; SHARD_COUNT * 16];
            for shard_id in 0..SHARD_COUNT {
                let start = shard_id * 16;
                let (twig_id, entry_file_size) = self.info.first_twig_at_height[shard_id];
                LittleEndian::write_u64(&mut data[start..start + 8], twig_id);
                LittleEndian::write_u64(&mut data[start + 8..start + 16], entry_file_size as u64);
                if self.cipher.is_some() {
                    let cipher = self.cipher.as_ref().unwrap();
                    let n = self.info.curr_height / PRUNE_EVERY_NBLOCKS;
                    let pos = (((n as usize - 1) * SHARD_COUNT) + shard_id) * (16 + TAG_SIZE);
                    let mut nonce_arr = [0u8; NONCE_SIZE];
                    LittleEndian::write_u64(&mut nonce_arr[..8], pos as u64);
                    match cipher.encrypt_in_place_detached(
                        &nonce_arr.into(),
                        b"",
                        &mut data[start..start + 16],
                    ) {
                        Err(err) => panic!("{}", err),
                        Ok(tag) => {
                            self.history_file.write(&data[start..start + 16]).unwrap();
                            self.history_file.write(tag.as_slice()).unwrap();
                        }
                    };
                }
            }
            if self.cipher.is_none() {
                self.history_file.write(&data[..]).unwrap();
            }
        }
        Arc::new(self.info.clone())
    }

    /// Ensure any pending async write completes before proceeding.
    /// Must be called before any new commit to maintain write ordering.
    fn wait_for_pending_write(&mut self) {
        if let Some(handle) = self.pending_write.take() {
            handle.join().expect("async metadb write thread panicked");
        }
    }

    /// Async variant of `commit()`: prepares the serialized data synchronously
    /// (so MetaInfo is available immediately) but defers file I/O to a
    /// background thread. Returns Arc<MetaInfo> without blocking on writes.
    ///
    /// Subsequent calls to `commit()` or `commit_async()` will wait for
    /// the previous async write to complete first (preserving durability).
    pub fn commit_async(&mut self) -> Arc<MetaInfo> {
        // Wait for any previous async write before starting a new one
        self.wait_for_pending_write();

        let kv = self.extra_data_map.remove(&self.info.curr_height).unwrap();
        self.info.extra_data = kv.1;

        // Serialize + encrypt synchronously (CPU-bound, fast)
        let name = format!(
            "{}.{}",
            self.meta_file_name,
            self.info.curr_height % 2
        );
        let meta_bytes = bincode::serialize(&self.info).unwrap();
        let write_data = if self.cipher.is_some() {
            let cipher = self.cipher.as_ref().unwrap();
            let mut nonce_arr = [0u8; NONCE_SIZE];
            LittleEndian::write_i64(&mut nonce_arr[..8], self.info.curr_height);
            let mut bz = meta_bytes;
            match cipher.encrypt_in_place_detached(&nonce_arr.into(), b"", &mut bz) {
                Err(err) => panic!("{}", err),
                Ok(tag) => {
                    let mut out = Vec::with_capacity(8 + bz.len() + tag.len());
                    out.extend_from_slice(&nonce_arr[0..8]);
                    out.extend_from_slice(&bz);
                    out.extend_from_slice(tag.as_slice());
                    out
                }
            }
        } else {
            meta_bytes
        };

        // Prepare history data if needed (CPU-bound)
        let history_data = if self.info.curr_height % PRUNE_EVERY_NBLOCKS == 0
            && self.info.curr_height > 0
        {
            let mut data = [0u8; SHARD_COUNT * 16];
            let mut history_segments: Vec<Vec<u8>> = Vec::new();
            for shard_id in 0..SHARD_COUNT {
                let start = shard_id * 16;
                let (twig_id, entry_file_size) =
                    self.info.first_twig_at_height[shard_id];
                LittleEndian::write_u64(
                    &mut data[start..start + 8],
                    twig_id,
                );
                LittleEndian::write_u64(
                    &mut data[start + 8..start + 16],
                    entry_file_size as u64,
                );
                if self.cipher.is_some() {
                    let cipher = self.cipher.as_ref().unwrap();
                    let n = self.info.curr_height / PRUNE_EVERY_NBLOCKS;
                    let pos = (((n as usize - 1) * SHARD_COUNT) + shard_id)
                        * (16 + TAG_SIZE);
                    let mut nonce_arr = [0u8; NONCE_SIZE];
                    LittleEndian::write_u64(&mut nonce_arr[..8], pos as u64);
                    match cipher.encrypt_in_place_detached(
                        &nonce_arr.into(),
                        b"",
                        &mut data[start..start + 16],
                    ) {
                        Err(err) => panic!("{}", err),
                        Ok(tag) => {
                            let mut seg = Vec::with_capacity(16 + TAG_SIZE);
                            seg.extend_from_slice(&data[start..start + 16]);
                            seg.extend_from_slice(tag.as_slice());
                            history_segments.push(seg);
                        }
                    };
                }
            }
            if self.cipher.is_none() {
                Some((data.to_vec(), Vec::new()))
            } else {
                Some((Vec::new(), history_segments))
            }
        } else {
            None
        };

        // Clone the file path for the background thread (history file
        // can't be sent across threads easily, so we open it in the thread)
        let history_file_path = if history_data.is_some() {
            // Re-derive the history file path from the meta file name
            let dir = Path::new(&self.meta_file_name).parent().unwrap();
            Some(format!("{}/prune_helper", dir.display()))
        } else {
            None
        };

        let result = Arc::new(self.info.clone());

        // Spawn background thread for file I/O
        self.pending_write = Some(std::thread::spawn(move || {
            fs::write(&name, write_data).unwrap();
            if let Some((plain_data, encrypted_segments)) = history_data {
                let history_path = history_file_path.unwrap();
                let mut f = File::options()
                    .append(true)
                    .open(&history_path)
                    .expect("failed to open history file for async write");
                if !plain_data.is_empty() {
                    f.write_all(&plain_data).unwrap();
                } else {
                    for seg in &encrypted_segments {
                        f.write_all(seg).unwrap();
                    }
                }
            }
        }));

        result
    }

    pub fn set_curr_height(&mut self, h: i64) {
        self.info.curr_height = h;
    }

    pub fn get_curr_height(&self) -> i64 {
        self.info.curr_height
    }

    pub fn set_twig_file_size(&mut self, shard_id: usize, size: i64) {
        self.info.twig_file_sizes[shard_id] = size;
    }

    pub fn get_twig_file_size(&self, shard_id: usize) -> i64 {
        self.info.twig_file_sizes[shard_id]
    }

    pub fn set_entry_file_size(&mut self, shard_id: usize, size: i64) {
        self.info.entry_file_sizes[shard_id] = size;
    }

    pub fn get_entry_file_size(&self, shard_id: usize) -> i64 {
        self.info.entry_file_sizes[shard_id]
    }

    pub fn set_first_twig_at_height(
        &mut self,
        shard_id: usize,
        height: i64,
        twig_id: u64,
        entry_file_size: i64,
    ) {
        if height % PRUNE_EVERY_NBLOCKS == 0 {
            self.info.first_twig_at_height[shard_id] = (twig_id, entry_file_size);
        }
    }

    pub fn get_first_twig_at_height(&self, shard_id: usize, height: i64) -> (u64, i64) {
        let n = height / PRUNE_EVERY_NBLOCKS;
        let mut pos = (((n as usize - 1) * SHARD_COUNT) + shard_id) * 16;
        if self.cipher.is_some() {
            pos = (((n as usize - 1) * SHARD_COUNT) + shard_id) * (16 + TAG_SIZE);
        }
        let mut buf = [0u8; 32];
        if self.cipher.is_some() {
            self.history_file.read_at(&mut buf, pos as u64).unwrap();
            let cipher = self.cipher.as_ref().unwrap();
            let mut nonce_arr = [0u8; NONCE_SIZE];
            LittleEndian::write_u64(&mut nonce_arr[..8], pos as u64);
            let mut tag = [0u8; TAG_SIZE];
            tag.copy_from_slice(&buf[16..]);
            if let Err(e) = cipher.decrypt_in_place_detached(
                &nonce_arr.into(),
                b"",
                &mut buf[0..16],
                &tag.into(),
            ) {
                panic!("{:?}", e)
            };
        } else {
            self.history_file
                .read_at(&mut buf[..16], pos as u64)
                .unwrap();
        }
        let twig_id = LittleEndian::read_u64(&buf[..8]);
        let entry_file_size = LittleEndian::read_u64(&buf[8..16]);
        (twig_id, entry_file_size as i64)
    }

    pub fn set_last_pruned_twig(&mut self, shard_id: usize, twig_id: u64, ef_prune_to: i64) {
        self.info.last_pruned_twig[shard_id] = (twig_id, ef_prune_to);
    }

    pub fn get_last_pruned_twig(&self, shard_id: usize) -> (u64, i64) {
        self.info.last_pruned_twig[shard_id]
    }

    pub fn get_edge_nodes(&self, shard_id: usize) -> Vec<u8> {
        self.info.edge_nodes[shard_id].clone()
    }

    pub fn set_edge_nodes(&mut self, shard_id: usize, bz: &[u8]) {
        self.info.edge_nodes[shard_id] = bz.to_vec();
    }

    pub fn get_next_serial_num(&self, shard_id: usize) -> u64 {
        self.info.next_serial_num[shard_id]
    }

    pub fn get_youngest_twig_id(&self, shard_id: usize) -> u64 {
        self.info.next_serial_num[shard_id] >> TWIG_SHIFT
    }

    pub fn set_next_serial_num(&mut self, shard_id: usize, sn: u64) {
        // called when new entry is appended
        self.info.next_serial_num[shard_id] = sn
    }

    pub fn get_root_hash(&self, shard_id: usize) -> [u8; 32] {
        self.info.root_hash[shard_id]
    }

    pub fn set_root_hash(&mut self, shard_id: usize, h: [u8; 32]) {
        self.info.root_hash[shard_id] = h
    }

    pub fn get_hash_of_root_hash(&self, height: i64) -> [u8; 32] {
        let mut is_prev_height = true;
        if height == self.info.curr_height {
            is_prev_height = false;
        }
        let l = self.info.root_hash_by_height.len();
        if l == 2 {
            if is_prev_height {
                self.info.root_hash_by_height[0]
            } else {
                self.info.root_hash_by_height[1]
            }
        } else if l == 1 {
            if is_prev_height {
                [0u8; 32]
            } else {
                self.info.root_hash_by_height[0]
            }
        } else {
            [0u8; 32]
        }
    }

    pub fn get_oldest_active_sn(&self, shard_id: usize) -> u64 {
        self.info.oldest_active_sn[shard_id]
    }

    pub fn set_oldest_active_sn(&mut self, shard_id: usize, id: u64) {
        self.info.oldest_active_sn[shard_id] = id
    }

    pub fn get_oldest_active_file_pos(&self, shard_id: usize) -> i64 {
        self.info.oldest_active_file_pos[shard_id]
    }

    pub fn set_oldest_active_file_pos(&mut self, shard_id: usize, pos: i64) {
        self.info.oldest_active_file_pos[shard_id] = pos
    }

    pub fn init(&mut self) {
        let curr_height = 0;
        self.info.curr_height = curr_height;
        for i in 0..SHARD_COUNT {
            self.info.last_pruned_twig[i] = (0, 0);
            self.info.next_serial_num[i] = 0;
            self.info.oldest_active_sn[i] = 0;
            self.info.oldest_active_file_pos[i] = 0;
            self.set_twig_file_size(i, 0);
            self.set_entry_file_size(i, 0);
        }
        self.extra_data_map.insert(curr_height, "".to_owned());
        self.commit();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{entryfile::helpers::create_cipher, test_helper::TempDir};
    use serial_test::serial;

    fn create_metadb(cipher: Option<Aes256Gcm>) -> (MetaDB, TempDir) {
        let dir = TempDir::new("./testdb.db");
        let mdb = MetaDB::with_dir("./testdb.db", cipher);
        (mdb, dir)
    }

    #[test]
    #[serial]
    fn test_metadb_init() {
        let cipher = create_cipher();
        let (mut mdb, _dir) = create_metadb(cipher);
        mdb.init();
        mdb.reload_from_file();

        assert_eq!(0, mdb.get_curr_height());
        for i in 0..SHARD_COUNT {
            assert_eq!((0, 0), mdb.get_last_pruned_twig(i));
            assert_eq!(0, mdb.get_next_serial_num(i));
            assert_eq!(0, mdb.get_oldest_active_sn(i));
            assert_eq!(0, mdb.get_oldest_active_file_pos(i));
            assert_eq!(0, mdb.get_twig_file_size(i));
            assert_eq!(0, mdb.get_entry_file_size(i));
            assert_eq!([0u8; 32], mdb.get_root_hash(i));
            assert_eq!(vec![0u8; 0], mdb.get_edge_nodes(i));
        }
    }

    #[test]
    #[serial]
    fn test_metadb() {
        let (mut mdb, _dir) = create_metadb(None);

        for i in 0..SHARD_COUNT {
            mdb.set_curr_height(12345);
            mdb.set_last_pruned_twig(i, 1000 + i as u64, 7000 + i as i64);
            mdb.set_next_serial_num(i, 2000 + i as u64);
            mdb.set_oldest_active_sn(i, 3000 + i as u64);
            mdb.set_oldest_active_file_pos(i, 4000 + i as i64);
            mdb.set_twig_file_size(i, 5000 + i as i64);
            mdb.set_entry_file_size(i, 6000 + i as i64);
            mdb.set_root_hash(i, [i as u8; 32]);
            mdb.set_edge_nodes(i, &[i as u8; 8]);
            mdb.set_first_twig_at_height(i, 100 + i as i64, 200 + i as u64, 0);
        }
        mdb.extra_data_map.insert(12345, "test".to_owned());
        mdb.commit();
        mdb.reload_from_file();

        assert_eq!(12345, mdb.get_curr_height());
        for i in 0..SHARD_COUNT {
            assert_eq!(
                (1000 + i as u64, 7000 + i as i64),
                mdb.get_last_pruned_twig(i)
            );
            assert_eq!(2000 + i as u64, mdb.get_next_serial_num(i));
            assert_eq!(3000 + i as u64, mdb.get_oldest_active_sn(i));
            assert_eq!(4000 + i as i64, mdb.get_oldest_active_file_pos(i));
            assert_eq!(5000 + i as i64, mdb.get_twig_file_size(i));
            assert_eq!(6000 + i as i64, mdb.get_entry_file_size(i));
            assert_eq!([i as u8; 32], mdb.get_root_hash(i));
            assert_eq!(vec![i as u8; 8], mdb.get_edge_nodes(i));
            // assert_eq!(200+i as u64, mdb.get_first_twig_at_height(i, 100+i as i64));
        }
        assert_eq!("test", mdb.get_extra_data());
    }

    // ========== ME-5: Async MetaDB Commit Tests ==========

    #[test]
    #[serial]
    fn test_metadb_commit_async_persists() {
        let (mut mdb, _dir) = create_metadb(None);

        mdb.set_curr_height(100);
        for i in 0..SHARD_COUNT {
            mdb.set_next_serial_num(i, 5000 + i as u64);
            mdb.set_root_hash(i, [(i + 1) as u8; 32]);
        }
        mdb.extra_data_map.insert(100, "async_test".to_owned());

        mdb.commit_async();
        // Wait for the async write to complete by triggering
        // wait_for_pending_write via a sync commit path
        mdb.set_curr_height(101);
        mdb.extra_data_map.insert(101, "".to_owned());
        mdb.commit();

        // Reload at height 101 and verify the block-100 data was
        // overwritten by block-101 on the same file slot (101%2==1, 100%2==0),
        // so we verify the values set before commit_async are reflected
        // in the state that commit() serialized (since commit_async
        // mutated info in place).
        mdb.reload_from_file();

        assert_eq!(101, mdb.get_curr_height());
        for i in 0..SHARD_COUNT {
            assert_eq!(5000 + i as u64, mdb.get_next_serial_num(i));
            assert_eq!([(i + 1) as u8; 32], mdb.get_root_hash(i));
        }
    }

    #[test]
    #[serial]
    fn test_metadb_commit_async_then_sync() {
        let (mut mdb, _dir) = create_metadb(None);

        // Block 1: async commit
        mdb.set_curr_height(1);
        for i in 0..SHARD_COUNT {
            mdb.set_next_serial_num(i, 100 + i as u64);
        }
        mdb.extra_data_map.insert(1, "block1".to_owned());
        mdb.commit_async();

        // Block 2: sync commit (waits for async to finish first)
        mdb.set_curr_height(2);
        for i in 0..SHARD_COUNT {
            mdb.set_oldest_active_sn(i, 200 + i as u64);
        }
        mdb.extra_data_map.insert(2, "block2".to_owned());
        mdb.commit();

        mdb.reload_from_file();

        // Height 2 is latest (both .0 and .1 files exist)
        assert_eq!(2, mdb.get_curr_height());
        for i in 0..SHARD_COUNT {
            // Values set in block 1 should persist through block 2
            assert_eq!(100 + i as u64, mdb.get_next_serial_num(i));
            // Values set in block 2
            assert_eq!(200 + i as u64, mdb.get_oldest_active_sn(i));
        }
        assert_eq!("block2", mdb.get_extra_data());
    }

    #[test]
    #[serial]
    fn test_metadb_sequential_async_commits() {
        let (mut mdb, _dir) = create_metadb(None);

        // First async commit
        mdb.set_curr_height(10);
        for i in 0..SHARD_COUNT {
            mdb.set_next_serial_num(i, 1000 + i as u64);
        }
        mdb.extra_data_map.insert(10, "first".to_owned());
        mdb.commit_async();

        // Second async commit (waits for first to complete)
        mdb.set_curr_height(11);
        for i in 0..SHARD_COUNT {
            mdb.set_next_serial_num(i, 2000 + i as u64);
        }
        mdb.extra_data_map.insert(11, "second".to_owned());
        mdb.commit_async();

        // Force wait by doing a sync commit
        mdb.set_curr_height(12);
        mdb.extra_data_map.insert(12, "third".to_owned());
        mdb.commit();

        mdb.reload_from_file();

        // Final state should reflect the last commit
        assert_eq!(12, mdb.get_curr_height());
        for i in 0..SHARD_COUNT {
            assert_eq!(2000 + i as u64, mdb.get_next_serial_num(i));
        }
        assert_eq!("third", mdb.get_extra_data());
    }
}
