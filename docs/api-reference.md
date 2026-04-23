# SkippyDB API Reference

This document covers the public API surface of SkippyDB. For architecture details, see [architecture.md](architecture.md). For GPU-specific APIs, see [gpu-acceleration.md](gpu-acceleration.md).

---

## Table of Contents

- [ADS Trait](#ads-trait)
- [AdsWrap (Database Handle)](#adswrap-database-handle)
- [AdsCore (Internal Engine)](#adscore-internal-engine)
- [Config](#config)
- [ChangeSet (Mutations)](#changeset-mutations)
- [Entry and EntryBz](#entry-and-entrybz)
- [EntryFile](#entryfile)
- [Tree and Proofs](#tree-and-proofs)
- [Task Traits](#task-traits)
- [TasksManager](#tasksmanager)
- [HPFile](#hpfile)
- [Hasher Utilities](#hasher-utilities)
- [Constants](#constants)

---

## ADS Trait

**File**: `qmdb/src/lib.rs`

The primary interface for interacting with SkippyDB. Implemented by `SharedAdsWrap`.

```rust
pub trait ADS: Send + Sync + 'static {
    /// Read an entry by key hash and optional key.
    ///
    /// - `height`: Block height to query (-1 for latest)
    /// - `key_hash`: SHA256 hash of the key (first 10 bytes used for index lookup)
    /// - `key`: Actual key bytes (empty slice skips key verification)
    /// - `buf`: Output buffer (must be >= entry size, typically 300 bytes)
    ///
    /// Returns `(entry_size, found)`.
    fn read_entry(
        &self,
        height: i64,
        key_hash: &[u8],
        key: &[u8],
        buf: &mut [u8],
    ) -> (usize, bool);

    /// Pre-warm the entry cache for upcoming reads.
    /// Called by the prefetcher before update operations.
    fn warmup<F>(&self, height: i64, k80: &[u8], access: F)
    where
        F: FnMut(EntryBz);

    /// Submit a task for processing.
    /// `task_id` = `(block_height << IN_BLOCK_IDX_BITS) | task_index`
    fn add_task(&self, task_id: i64);

    /// Attach extra data (e.g., block hash) to a block height.
    fn insert_extra_data(&self, height: i64, data: String);

    /// Get the Merkle root hash at a specific block height.
    fn get_root_hash_of_height(&self, height: i64) -> [u8; 32];
}
```

### Example: Reading an Entry

```rust
use qmdb::utils::hasher;
use qmdb::def::DEFAULT_ENTRY_SIZE;

let shared = ads.get_shared();
let key = b"user:alice";
let key_hash = hasher::hash(key);
let mut buf = [0u8; DEFAULT_ENTRY_SIZE];

let (size, found) = shared.read_entry(-1, &key_hash, key, &mut buf);
if found {
    let entry = EntryBz { bz: &buf[..size] };
    println!("Value: {:?}", entry.value());
}
```

---

## AdsWrap (Database Handle)

**File**: `qmdb/src/lib.rs`

The user-facing database wrapper. Owns the pipeline and manages block lifecycle.

```rust
pub struct AdsWrap<T: Task> { /* ... */ }

impl<T: Task + 'static> AdsWrap<T> {
    /// Create a new database instance.
    /// The directory must be initialized with `AdsCore::init_dir()` first.
    pub fn new(config: &Config) -> Self;

    /// Start a new block at the given height.
    ///
    /// Returns `(success, previous_block_meta)`:
    /// - `success`: true if block was started
    /// - `previous_block_meta`: MetaInfo from 2 blocks ago (if available)
    pub fn start_block(
        &mut self,
        height: i64,
        tasks_manager: Arc<TasksManager<T>>,
    ) -> (bool, Option<Arc<MetaInfo>>);

    /// Get a thread-safe shared handle for concurrent reads/writes.
    pub fn get_shared(&self) -> SharedAdsWrap;

    /// Flush the pipeline — blocks until the current block's data
    /// is safely on SSD. Returns accumulated MetaInfo from completed blocks.
    pub fn flush(&mut self) -> Vec<Arc<MetaInfo>>;

    /// Generate a Merkle proof for the entry at `sn` in `shard_id`.
    pub fn get_proof(
        &self,
        shard_id: usize,
        sn: u64,
    ) -> anyhow::Result<ProofPath>;

    /// Get the MetaDB handle for direct metadata access.
    pub fn get_metadb(&self) -> Arc<RwLock<MetaDB>>;

    /// Get all entry files (one per shard).
    pub fn get_entry_files(&self) -> Vec<Arc<EntryFile>>;
}
```

### Example: Full Block Lifecycle

```rust
use qmdb::config::Config;
use qmdb::def::IN_BLOCK_IDX_BITS;
use qmdb::tasks::TasksManager;
use qmdb::{AdsCore, AdsWrap, ADS};
use std::sync::Arc;
use parking_lot::RwLock;

// Initialize
let config = Config::from_dir("my_db");
AdsCore::init_dir(&config);
let mut ads = AdsWrap::new(&config);

// Block height 1
let height = 1i64;
let task_id = height << IN_BLOCK_IDX_BITS;

// Create tasks (see ChangeSet section below)
let tasks = vec![RwLock::new(Some(my_task))];
let tm = Arc::new(TasksManager::new(tasks, task_id));

// Start block
let (ok, _prev_meta) = ads.start_block(height, tm);
assert!(ok);

// Submit tasks
let shared = ads.get_shared();
shared.insert_extra_data(height, String::new());
shared.add_task(task_id);

// Flush and get metadata
let metas = ads.flush();
for meta in &metas {
    println!("Block {} committed, root: {:?}", meta.height, meta.root_hash);
}
```

---

## AdsCore (Internal Engine)

**File**: `qmdb/src/lib.rs`

Low-level engine. Most users should use `AdsWrap` instead.

```rust
pub struct AdsCore { /* ... */ }

impl AdsCore {
    /// Initialize the data directory structure.
    /// Creates subdirectories for entries, twigs, metadata, and index.
    /// Inserts 4096 sentry entries per shard as boundary markers.
    pub fn init_dir(config: &Config);

    /// Initialize with custom sentry key-value data.
    pub fn init_dir_with_kv(
        config: &Config,
        kv: &[(Vec<u8>, Vec<u8>, Vec<u8>)],
    );

    /// Create the core engine. Returns the engine, a metadata receiver,
    /// and the flusher (which must be started on its own thread).
    pub fn _new(
        task_hub: Arc<dyn TaskHub>,
        config: &Config,
    ) -> (Self, Receiver<Arc<MetaInfo>>, Flusher);

    /// Start all pipeline threads (prefetcher, updater, compactor, flusher).
    pub fn start_threads(
        self: Arc<Self>,
        config: &Config,
        flusher: Flusher,
    );
}
```

---

## Config

**File**: `qmdb/src/config.rs`

```rust
pub struct Config {
    /// Root directory for all data files.
    pub dir: String,

    /// Write buffer size for HPFile (bytes). Default: 8 MB.
    pub wrbuf_size: usize,

    /// Segment size for HPFile (bytes). Default: 1 GB.
    pub file_segment_size: usize,

    /// Enable twig file for proof generation. Default: false.
    pub with_twig_file: bool,

    /// Compaction threshold (minimum entries before compaction). Default: 20,000,000.
    pub compact_thres: i64,

    /// Utilization ratio numerator. Compact when active/total < ratio/div.
    pub utilization_ratio: i64,

    /// Utilization ratio denominator.
    pub utilization_div: i64,

    /// Channel buffer size for task pipeline. Default: 200,000.
    pub task_chan_size: usize,

    /// Number of prefetcher threads. Default: 512.
    pub prefetcher_thread_count: usize,

    /// AES-256-GCM keys for encryption (3 keys × 32 bytes = 96 bytes).
    /// None = no encryption.
    pub aes_keys: Option<[u8; 96]>,

    /// io_uring instance count (Linux only). Default: 32.
    pub uring_count: usize,

    /// io_uring submission queue depth. Default: 1024.
    pub uring_size: u32,

    /// Channel size for subscription IDs. Default: 20,000.
    pub sub_id_chan_size: usize,
}

impl Config {
    /// Create a config with the given directory, using all defaults.
    pub fn from_dir(dir: &str) -> Self;
}
```

---

## ChangeSet (Mutations)

**File**: `qmdb/src/utils/changeset.rs`

Groups CRUD operations for a single transaction.

```rust
pub struct ChangeSet {
    pub data: Vec<u8>,              // Concatenated keys, values, key_hashes
    pub op_list: Vec<ChangeOp>,     // List of operations
}

impl ChangeSet {
    /// Create a new empty ChangeSet.
    pub fn new() -> Self;

    /// Add an operation.
    ///
    /// - `op_type`: OP_CREATE, OP_WRITE, OP_DELETE, or OP_READ
    /// - `shard_id`: byte0_to_shard_id(key_hash[0])
    /// - `key_hash`: SHA256(key)
    /// - `k`: key bytes
    /// - `v`: value bytes
    /// - `rec`: optional OpRecord for tracking
    pub fn add_op(
        &mut self,
        op_type: u8,
        shard_id: u8,
        key_hash: &[u8; 32],
        k: &[u8],
        v: &[u8],
        rec: Option<Box<OpRecord>>,
    );

    /// Add an operation with old value tracking.
    pub fn add_op_with_old_value(
        &mut self,
        op_type: u8,
        shard_id: u8,
        key_hash: &[u8; 32],
        k: &[u8],
        v: &[u8],
        old_v: &[u8],
        rec: Option<Box<OpRecord>>,
    );

    /// Add an operation from an OpRecord.
    pub fn add_op_rec(&mut self, rec: OpRecord);

    /// Sort operations by (shard_id, key_hash, op_type).
    /// MUST be called before submitting to the pipeline.
    pub fn sort(&mut self);

    /// Iterate all operations.
    pub fn run_all<F>(&self, access: F)
    where
        F: FnMut(u8, &[u8; 32], &[u8], &[u8], Option<&Box<OpRecord>>);

    /// Iterate operations for a specific shard.
    pub fn run_in_shard<F>(&self, shard_id: usize, access: F)
    where
        F: FnMut(u8, &[u8; 32], &[u8], &[u8], Option<&Box<OpRecord>>);
}
```

### Example: Building a ChangeSet

```rust
use qmdb::def::{OP_CREATE, OP_WRITE, OP_DELETE};
use qmdb::utils::{changeset::ChangeSet, hasher, byte0_to_shard_id};

let mut cs = ChangeSet::new();

// Create a new key
let key = b"account:0x1234";
let value = b"balance:1000";
let kh = hasher::hash(key);
let sid = byte0_to_shard_id(kh[0]) as u8;
cs.add_op(OP_CREATE, sid, &kh, key, value, None);

// Update an existing key
let new_val = b"balance:2000";
cs.add_op(OP_WRITE, sid, &kh, key, new_val, None);

// Delete a key
cs.add_op(OP_DELETE, sid, &kh, key, &[], None);

// IMPORTANT: sort before use
cs.sort();
```

---

## Entry and EntryBz

**File**: `qmdb/src/entryfile/entry.rs`

### Entry (Write Path)

```rust
pub struct Entry<'a> {
    pub key: &'a [u8],
    pub value: &'a [u8],
    pub next_key_hash: &'a [u8],
    pub version: i64,
    pub serial_number: u64,
}

impl Entry {
    /// Serialize the entry into `buf`. Returns bytes written.
    /// `dsn_list` is the list of deactivated serial numbers.
    pub fn dump(&self, buf: &mut [u8], dsn_list: &[u64]) -> usize;

    /// Compute SHA256 hash of the serialized entry.
    pub fn hash(&self, dsn_list: &[u64]) -> [u8; 32];
}
```

### EntryBz (Read Path)

Zero-copy view over serialized entry bytes.

```rust
pub struct EntryBz<'a> {
    pub bz: &'a [u8],
}

impl<'a> EntryBz<'a> {
    /// Key bytes.
    pub fn key(&self) -> &[u8];

    /// Value bytes.
    pub fn value(&self) -> &[u8];

    /// Total serialized length.
    pub fn len(&self) -> usize;

    /// Block height when this entry was created.
    pub fn version(&self) -> i64;

    /// Global sequence number.
    pub fn serial_number(&self) -> u64;

    /// 32-byte hash of the next key in sorted order.
    pub fn next_key_hash(&self) -> &[u8];

    /// SHA256 hash of this entry's serialized bytes.
    pub fn hash(&self) -> [u8; 32];

    /// SHA256 of the key.
    pub fn key_hash(&self) -> [u8; 32];

    /// Number of deactivated serial numbers.
    pub fn dsn_count(&self) -> usize;

    /// Get the i-th deactivated serial number.
    pub fn dsn_at(&self, i: usize) -> u64;
}
```

### Example: Parsing an Entry

```rust
let mut buf = [0u8; 300];
let (size, found) = ads.read_entry(-1, &key_hash, &[], &mut buf);

if found {
    let entry = EntryBz { bz: &buf[..size] };

    println!("Key:       {:?}", String::from_utf8_lossy(entry.key()));
    println!("Value:     {:?}", String::from_utf8_lossy(entry.value()));
    println!("Version:   {}", entry.version());
    println!("Serial#:   {}", entry.serial_number());
    println!("Next KH:   {}", hex::encode(entry.next_key_hash()));
    println!("Hash:      {}", hex::encode(entry.hash()));

    // Iterate deactivated serial numbers
    for i in 0..entry.dsn_count() {
        println!("Deactivated SN: {}", entry.dsn_at(i));
    }
}
```

---

## EntryFile

**File**: `qmdb/src/entryfile/entryfile.rs`

```rust
pub struct EntryFile { /* ... */ }

impl EntryFile {
    /// Read an entry from disk at the given file position.
    /// Returns the number of bytes read into `buf`.
    pub fn read_entry(&self, pos: i64, buf: &mut [u8]) -> usize;
}
```

---

## Tree and Proofs

### ProofPath

**File**: `qmdb/src/merkletree/proof.rs`

```rust
pub struct ProofPath {
    /// 11 sibling hashes in the entry Merkle tree (levels 0-10).
    pub left_of_twig: [ProofNode; 11],

    /// 3 sibling hashes in the ActiveBits tree (levels 8-10).
    pub right_of_twig: [ProofNode; 3],

    /// Sibling hashes from twig root to global root.
    pub upper_path: Vec<ProofNode>,

    /// The entry's serial number.
    pub serial_num: u64,

    /// The expected Merkle root.
    pub root: [u8; 32],
}

impl ProofPath {
    /// Verify the proof. If `complete` is true, fills in computed hashes.
    pub fn check(&mut self, complete: bool) -> anyhow::Result<()>;

    /// Serialize the proof to bytes for transmission.
    pub fn to_bytes(&self) -> Vec<u8>;
}

/// Deserialize a proof from bytes.
pub fn bytes_to_proof_path(bz: &Vec<u8>) -> anyhow::Result<ProofPath>;

/// Verify a proof and return its serialized form.
pub fn check_proof(path: &mut ProofPath) -> anyhow::Result<Vec<u8>>;
```

### ProofNode

```rust
pub struct ProofNode {
    pub self_hash: [u8; 32],   // This node's hash
    pub peer_hash: [u8; 32],   // Sibling node's hash
    pub peer_at_left: bool,    // Is the sibling on the left?
}
```

### Example: Generating and Verifying a Proof

```rust
// Generate
let proof = ads.get_proof(shard_id, serial_number)?;

// Verify
let mut proof_copy = proof;
proof_copy.check(false)?;

// Serialize for network transmission
let bytes = proof_copy.to_bytes();
println!("Proof: {} bytes", bytes.len());

// Deserialize and re-verify
let mut reconstructed = bytes_to_proof_path(&bytes)?;
reconstructed.check(true)?;

assert_eq!(reconstructed.root, proof_copy.root);
```

---

## Task Traits

**File**: `qmdb/src/tasks/task.rs`

```rust
/// A single task (transaction) containing a list of ChangeSets.
pub trait Task: Send + Sync {
    fn get_change_sets(&self) -> Arc<Vec<ChangeSet>>;
}

/// Hub for managing tasks across block boundaries.
pub trait TaskHub: Send + Sync {
    /// Check if a task ID is the beginning or end of a block.
    /// Returns (optional_new_cache, is_end_of_block).
    fn check_begin_end(&self, task_id: i64) -> (Option<Arc<EntryCache>>, bool);

    /// Get the ChangeSets for a task.
    fn get_change_sets(&self, task_id: i64) -> Arc<Vec<ChangeSet>>;
}
```

### Implementing the Task Trait

```rust
use qmdb::tasks::task::Task;
use qmdb::utils::changeset::ChangeSet;
use std::sync::Arc;

struct MyTask {
    change_sets: Arc<Vec<ChangeSet>>,
}

impl MyTask {
    fn new(sets: Vec<ChangeSet>) -> Self {
        Self {
            change_sets: Arc::new(sets),
        }
    }
}

impl Task for MyTask {
    fn get_change_sets(&self) -> Arc<Vec<ChangeSet>> {
        self.change_sets.clone()
    }
}
```

---

## TasksManager

**File**: `qmdb/src/tasks/tasksmanager.rs`

```rust
pub struct TasksManager<T: Task> {
    /* ... */
}

impl<T: Task> TasksManager<T> {
    /// Create a new TasksManager.
    ///
    /// - `tasks`: List of tasks wrapped in `RwLock<Option<T>>`
    /// - `last_task_id`: The task ID of the last task in this block
    pub fn new(
        tasks: Vec<RwLock<Option<T>>>,
        last_task_id: i64,
    ) -> Self;
}
```

---

## HPFile

**File**: `hpfile/src/lib.rs`

Head-prunable file — the storage foundation.

```rust
pub struct HPFile { /* ... */ }

impl HPFile {
    /// Create/open an HPFile.
    ///
    /// - `wr_buf_size`: Write buffer capacity
    /// - `segment_size`: Size of each segment file (must be divisible by wr_buf_size)
    /// - `dir_name`: Directory path for segment files
    /// - `directio`: Enable O_DIRECT for readonly segments (Linux only)
    pub fn new(
        wr_buf_size: i64,
        segment_size: i64,
        dir_name: String,
        directio: bool,
    ) -> Result<HPFile>;

    /// Current logical file size (including buffered, unflushed data).
    pub fn size(&self) -> i64;

    /// Flushed file size on disk.
    pub fn size_on_disk(&self) -> i64;

    /// Append bytes. Returns the start position of the written data.
    pub fn append(&self, bz: &[u8], buffer: &mut Vec<u8>) -> io::Result<i64>;

    /// Read bytes at a logical offset.
    pub fn read_at(&self, bz: &mut [u8], offset: i64) -> io::Result<usize>;

    /// Read with spatial locality caching.
    pub fn read_at_with_pre_reader(
        &self,
        buf: &mut Vec<u8>,
        num_bytes: usize,
        offset: i64,
        pre_reader: &mut PreReader,
    ) -> io::Result<usize>;

    /// Flush write buffer to disk.
    pub fn flush(&self, buffer: &mut Vec<u8>, eof: bool) -> io::Result<()>;

    /// Truncate from the end (discard data from `size` to current end).
    pub fn truncate(&self, size: i64) -> io::Result<()>;

    /// Prune from the head (delete segments before `offset`).
    pub fn prune_head(&self, offset: i64) -> io::Result<()>;

    /// Close all file handles.
    pub fn close(&self);
}
```

---

## Hasher Utilities

**File**: `qmdb/src/utils/hasher.rs`

```rust
pub type Hash32 = [u8; 32];

/// SHA256 hash of arbitrary bytes.
pub fn hash(data: &[u8]) -> Hash32;

/// SHA256(level_byte || left_32B || right_32B).
/// Used for Merkle tree node hashing.
pub fn hash2(level: u8, a: &[u8; 32], b: &[u8; 32]) -> Hash32;

/// Like hash2 but swaps inputs based on `peer_at_left`.
pub fn hash2x(
    level: u8,
    self_hash: &[u8; 32],
    peer_hash: &[u8; 32],
    peer_at_left: bool,
) -> Hash32;

/// CPU batch hash of node inputs.
pub fn batch_node_hash_cpu(
    jobs: &[(u8, [u8; 32], [u8; 32])],
) -> Vec<Hash32>;
```

---

## Constants

**File**: `qmdb/src/def.rs`

| Constant | Value | Description |
|---|---|---|
| `SHARD_COUNT` | 16 | Number of data shards |
| `SENTRY_COUNT` | 256 | Boundary entries per shard at init |
| `LEAF_COUNT_IN_TWIG` | 2048 | Entries per twig |
| `TWIG_SHIFT` | 11 | `twig_id = serial_number >> 11` |
| `TWIG_MASK` | 2047 | `entry_offset = sn & 0x7FF` |
| `TWIG_ROOT_LEVEL` | 12 | Tree level of twig roots |
| `FIRST_LEVEL_ABOVE_TWIG` | 13 | First upper tree level |
| `MAX_TREE_LEVEL` | 64 | Maximum supported tree height |
| `MAX_UPPER_LEVEL` | 51 | Levels in upper tree (64 - 13) |
| `NODE_SHARD_COUNT` | 4 | Sub-shards for upper tree nodes |
| `TWIG_SHARD_COUNT` | 4 | Sub-shards for active twigs |
| `IN_BLOCK_IDX_BITS` | 24 | `task_id = (height << 24) \| idx` |
| `DEFAULT_ENTRY_SIZE` | 300 | Recommended buffer size for reads |
| `ENTRY_FIXED_LENGTH` | 54 | Minimum entry size (no key/value) |
| `PRUNE_EVERY_NBLOCKS` | 500 | Twig pruning frequency |
| `MIN_PRUNE_COUNT` | 64 | Minimum twigs to prune at once |
| `OP_READ` | 1 | Read operation type |
| `OP_WRITE` | 2 | Update operation type |
| `OP_CREATE` | 3 | Create operation type |
| `OP_DELETE` | 4 | Delete operation type |
