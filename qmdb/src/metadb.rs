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

/// On-disk version envelope magic. Written at the head of every plaintext
/// MetaInfo payload (inside the AES-GCM ciphertext when `tee_cipher` is on).
///
/// - Phase 2.2 format `b"SKIPV2\x00\x00"`. Anything else is rejected by
///   `reload_from_file` and `with_dir_checked` so old DBs fail loudly instead
///   of being silently reinitialized to defaults (the pre-2.2 behavior was
///   a data-loss hazard).
///
/// When we bump the format again (Phase 3+), bump the last two bytes, add the
/// new magic to the `recognize_magic` dispatch below, and leave the V2 path
/// green for the duration of the deprecation window.
pub const META_MAGIC_V2: &[u8; 8] = b"SKIPV2\x00\x00";

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
pub struct MetaInfo {
    /// Number of shards the DB was built against. Recorded so a binary with a
    /// different compile-time `SHARD_COUNT` refuses to reopen rather than
    /// silently corrupt per-shard bookkeeping. Stamped by `MetaInfo::new` at
    /// build time; validated by `MetaDB::with_dir_checked`.
    ///
    /// `shard_count` also sizes the `Box<[T]>` per-shard fields below — they
    /// are length-prefixed by bincode, so the invariant
    /// `self.shard_count as usize == self.<field>.len()` holds across every
    /// serialize/deserialize round-trip.
    pub shard_count: u32,
    pub curr_height: i64,
    pub last_pruned_twig: Box<[(u64, i64)]>,
    pub next_serial_num: Box<[u64]>,
    pub oldest_active_sn: Box<[u64]>,
    pub oldest_active_file_pos: Box<[i64]>,
    pub root_hash: Box<[[u8; 32]]>,
    pub root_hash_by_height: Vec<[u8; 32]>,
    pub edge_nodes: Box<[Vec<u8>]>,
    pub twig_file_sizes: Box<[i64]>,
    pub entry_file_sizes: Box<[i64]>,
    pub first_twig_at_height: Box<[(u64, i64)]>,
    pub extra_data: String,
}

impl MetaInfo {
    /// Build a fresh `MetaInfo` sized for `shard_count` shards. Every per-
    /// shard boxed slice is allocated with the right length so downstream
    /// setters can index `[shard_id]` without bounds juggling.
    fn new(shard_count: usize) -> Self {
        Self {
            shard_count: shard_count as u32,
            curr_height: 0,
            last_pruned_twig: vec![(0u64, 0i64); shard_count].into_boxed_slice(),
            next_serial_num: vec![0u64; shard_count].into_boxed_slice(),
            oldest_active_sn: vec![0u64; shard_count].into_boxed_slice(),
            oldest_active_file_pos: vec![0i64; shard_count].into_boxed_slice(),
            root_hash: vec![[0u8; 32]; shard_count].into_boxed_slice(),
            root_hash_by_height: vec![],
            edge_nodes: vec![Vec::<u8>::new(); shard_count].into_boxed_slice(),
            twig_file_sizes: vec![0i64; shard_count].into_boxed_slice(),
            entry_file_sizes: vec![0i64; shard_count].into_boxed_slice(),
            first_twig_at_height: vec![(0u64, 0i64); shard_count].into_boxed_slice(),
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
    /// Found a MetaInfo file whose version envelope we don't recognize.
    /// Covers both pre-V2 files (no magic) and future formats. Callers
    /// should surface a "rebuild the DB" message rather than retry.
    UnsupportedFormat { head: [u8; 8] },
    /// Bincode couldn't parse the plaintext payload after the V2 magic.
    /// Usually means the file was truncated or partially written during a
    /// crash.
    CorruptPayload(String),
}

impl std::fmt::Display for MetaDbError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MetaDbError::ShardCountMismatch { expected, got } => write!(
                f,
                "MetaDB shard-count mismatch: expected {}, on-disk claims {}",
                expected, got
            ),
            MetaDbError::UnsupportedFormat { head } => write!(
                f,
                "MetaDB format not recognized (head={:02x?}); expected V2 magic {:02x?}",
                head, META_MAGIC_V2
            ),
            MetaDbError::CorruptPayload(reason) => write!(
                f,
                "MetaDB payload could not be deserialized: {}",
                reason
            ),
        }
    }
}

impl std::error::Error for MetaDbError {}

/// Strip the V2 magic header from a plaintext MetaInfo blob and deserialize
/// the rest. Returns `UnsupportedFormat` for anything that doesn't start with
/// `META_MAGIC_V2` (which notably includes pre-2.2 files).
fn parse_metainfo_v2(plaintext: &[u8]) -> Result<MetaInfo, MetaDbError> {
    if plaintext.len() < META_MAGIC_V2.len() {
        let mut head = [0u8; 8];
        head[..plaintext.len()].copy_from_slice(plaintext);
        return Err(MetaDbError::UnsupportedFormat { head });
    }
    let (magic, rest) = plaintext.split_at(META_MAGIC_V2.len());
    if magic != META_MAGIC_V2 {
        let mut head = [0u8; 8];
        head.copy_from_slice(magic);
        return Err(MetaDbError::UnsupportedFormat { head });
    }
    bincode::deserialize::<MetaInfo>(rest)
        .map_err(|e| MetaDbError::CorruptPayload(e.to_string()))
}

/// Prepend `META_MAGIC_V2` to `bincode(info)` to produce the plaintext written
/// to disk (or fed into AES-GCM when `tee_cipher` is on). Every commit path
/// must route through this so on-disk files always carry the envelope.
fn serialize_metainfo_v2(info: &MetaInfo) -> Vec<u8> {
    let body = bincode::serialize(info).expect("MetaInfo serialization is infallible");
    let mut out = Vec::with_capacity(META_MAGIC_V2.len() + body.len());
    out.extend_from_slice(META_MAGIC_V2);
    out.extend_from_slice(&body);
    out
}

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
            info: MetaInfo::new(SHARD_COUNT),
            meta_file_name,
            history_file,
            extra_data_map: Arc::new(DashMap::new()),
            cipher,
            pending_write: None,
        };
        // A fresh `info.0` doesn't exist yet — both slots are absent. In that
        // case `reload_from_file_checked` is a no-op and `shard_count` stays
        // at whatever the caller's `SHARD_COUNT` was stamped by
        // `MetaInfo::new`. Validate the compare against that too so e.g.
        // `with_dir_checked(fresh_dir, expected=99)` still fails fast.
        res.reload_from_file_checked()?;
        let got = res.info.shard_count as usize;
        if got != expected_shard_count {
            return Err(MetaDbError::ShardCountMismatch {
                expected: expected_shard_count,
                got,
            });
        }
        Ok(res)
    }

    pub fn with_dir(dir_name: &str, cipher: Option<Aes256Gcm>) -> Self {
        Self::with_dir_and_shard_count(dir_name, cipher, SHARD_COUNT)
    }

    /// Build a `MetaDB` sized for a specific runtime shard count. Matches
    /// `with_dir` when `shard_count == SHARD_COUNT`.
    ///
    /// Use this from Phase 2.3+ callers that carry a `Topology`; `with_dir`
    /// is retained for backward compatibility and pins to the compile-time
    /// constant. On reload, the on-disk `shard_count` wins over the caller's
    /// claim (so a 32-shard DB reopened via `with_dir_and_shard_count(.., 64)`
    /// still loads at 32 — use `with_dir_checked` when you need the typed
    /// mismatch error instead).
    pub fn with_dir_and_shard_count(
        dir_name: &str,
        cipher: Option<Aes256Gcm>,
        shard_count: usize,
    ) -> Self {
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
            info: MetaInfo::new(shard_count),
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
        // Old lossy callers (everything except `with_dir_checked`) get the
        // loud-panic behavior: V1 or corrupt files panic on reload instead
        // of silently reinitializing to defaults — the pre-2.2 behavior was
        // a data-loss hazard. Missing files are still fine (fresh DB).
        if let Err(e) = self.reload_from_file_checked() {
            panic!(
                "MetaDB::reload_from_file: {} — old pre-V2 DBs must be rebuilt, \
                 see TODO.md 'Phase 2.2 on-disk format bump'",
                e
            );
        }
    }

    /// Fallible reload path. Reads `{meta_file_name}.0` and `.1`, strips the
    /// V2 magic envelope, and keeps the higher-`curr_height` winner. Used by
    /// `with_dir_checked`; `reload_from_file` is a panicking wrapper for the
    /// legacy `with_dir` callers.
    pub fn reload_from_file_checked(&mut self) -> Result<(), MetaDbError> {
        for slot in [0, 1] {
            let name = format!("{}.{}", self.meta_file_name, slot);
            if !Path::new(&name).exists() {
                continue;
            }
            let mut bz = get_file_as_byte_vec(&name);
            if self.cipher.is_some() {
                Self::decrypt(&self.cipher, &mut bz);
                let size = bz.len();
                bz = bz[8..size - TAG_SIZE].to_owned();
            }
            let candidate = parse_metainfo_v2(&bz)?;
            if slot == 0 || candidate.curr_height > self.info.curr_height {
                self.info = candidate;
            }
        }
        Ok(())
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
        let mut bz = serialize_metainfo_v2(&self.info);
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
            let shard_count = self.info.shard_count as usize;
            let mut data = vec![0u8; shard_count * 16];
            for shard_id in 0..shard_count {
                let start = shard_id * 16;
                let (twig_id, entry_file_size) = self.info.first_twig_at_height[shard_id];
                LittleEndian::write_u64(&mut data[start..start + 8], twig_id);
                LittleEndian::write_u64(&mut data[start + 8..start + 16], entry_file_size as u64);
                if self.cipher.is_some() {
                    let cipher = self.cipher.as_ref().unwrap();
                    let n = self.info.curr_height / PRUNE_EVERY_NBLOCKS;
                    let pos = (((n as usize - 1) * shard_count) + shard_id) * (16 + TAG_SIZE);
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
        let meta_bytes = serialize_metainfo_v2(&self.info);
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
            let shard_count = self.info.shard_count as usize;
            let mut data = vec![0u8; shard_count * 16];
            let mut history_segments: Vec<Vec<u8>> = Vec::new();
            for shard_id in 0..shard_count {
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
                    let pos = (((n as usize - 1) * shard_count) + shard_id)
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
        let shard_count = self.info.shard_count as usize;
        let n = height / PRUNE_EVERY_NBLOCKS;
        let mut pos = (((n as usize - 1) * shard_count) + shard_id) * 16;
        if self.cipher.is_some() {
            pos = (((n as usize - 1) * shard_count) + shard_id) * (16 + TAG_SIZE);
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
        let shard_count = self.info.shard_count as usize;
        for i in 0..shard_count {
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
