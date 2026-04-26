# SkippyDB Internal Architecture

This document covers the internal architecture of SkippyDB — the data structures, pipeline stages, file formats, threading model, and recovery process.

For the high-level concepts and usage guide, see the [README](../README.md).
For GPU acceleration details, see [gpu-acceleration.md](gpu-acceleration.md).

---

## Table of Contents

- [System Overview](#system-overview)
- [Data Structures](#data-structures)
  - [Entry](#entry)
  - [Twig](#twig)
  - [Tree](#tree)
  - [UpperTree](#uppertree)
  - [EdgeNodes](#edgenodes)
  - [ActiveBits](#activebits)
- [File Layer](#file-layer)
  - [HPFile (Head-Prunable File)](#hpfile-head-prunable-file)
  - [EntryFile](#entryfile)
  - [TwigFile](#twigfile)
  - [MetaDB](#metadb)
- [Indexer](#indexer)
  - [InMemIndexer](#inmemindexer)
  - [HybridIndexer](#hybridindexer)
- [Pipeline Architecture](#pipeline-architecture)
  - [Prefetcher](#prefetcher)
  - [Updater](#updater)
  - [Flusher](#flusher)
  - [Compactor](#compactor)
- [TaskHub and Block Processing](#taskhub-and-block-processing)
- [Proof Generation](#proof-generation)
- [Recovery](#recovery)
- [Threading Model](#threading-model)
- [Sharding](#sharding)

---

## System Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                          Client (e.g., EVM)                          │
│                                                                      │
│  Creates ChangeSets ──▶ Wraps in Tasks ──▶ Submits via start_block  │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │       TaskHub           │
                    │  (double-buffered:      │
                    │   height H, height H+1) │
                    └────────────┬────────────┘
                                 │
        ┌────────────────────────▼────────────────────────┐
        │                Pipeline (per shard × 16)         │
        │                                                  │
        │  ┌──────────┐   ┌──────────┐   ┌──────────────┐ │
        │  │Prefetcher│──▶│ Updater  │──▶│   Flusher    │ │
        │  │          │   │          │   │              │ │
        │  │Fills the │   │Updates   │   │Jobs 1-8:    │ │
        │  │EntryCache│   │B-tree    │   │Append, hash,│ │
        │  │for D/C   │   │index,    │   │evict, flush,│ │
        │  │operations│   │sends new │   │prune, meta  │ │
        │  │          │   │entries   │   │              │ │
        │  └──────────┘   │via       │   │Optional GPU │ │
        │                 │EntryBuf  │   │acceleration │ │
        │                 └──────────┘   └──────────────┘ │
        │                                                  │
        │  ┌─────────┐  ┌──────────┐  ┌─────────────────┐ │
        │  │Compactor │  │EntryBuf  │  │  EntryCache     │ │
        │  │(bg GC)   │  │(FIFO)    │  │  (read cache)   │ │
        │  └─────────┘  └──────────┘  └─────────────────┘ │
        └──────────────────────────────────────────────────┘
                                 │
        ┌────────────────────────▼────────────────────────┐
        │              Storage Layer (per shard × 16)      │
        │                                                  │
        │  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
        │  │EntryFile │  │ TwigFile │  │   Indexer     │   │
        │  │(HPFile)  │  │(HPFile)  │  │  (B-tree)     │   │
        │  └──────────┘  └──────────┘  └──────────────┘   │
        │                                                  │
        │              ┌──────────────────┐                │
        │              │     MetaDB       │                │
        │              │   (RocksDB)      │                │
        │              └──────────────────┘                │
        └──────────────────────────────────────────────────┘
```

---

## Data Structures

### Entry

**File**: [`qmdb/src/entryfile/entry.rs`](../qmdb/src/entryfile/entry.rs)

An Entry is the primitive data unit — one key-value pair at a specific block height.

```rust
pub struct Entry<'a> {
    pub key: &'a [u8],           // Variable-length key (max 255 bytes)
    pub value: &'a [u8],         // Variable-length value (max 16MB)
    pub next_key_hash: &'a [u8], // 32B: hash of the next key in sorted order
    pub version: i64,            // Block height when created
    pub serial_number: u64,      // Global monotonic sequence number
}
```

**Serialized binary layout**:

```
Offset  Size    Field
──────  ────    ─────
0       1B      key_length (u8)
1       3B      value_length (u24, little-endian)
4       1B      deactivated_sn_count (u8)
5       var     key (key_length bytes)
5+K     var     value (value_length bytes)
5+K+V   32B     next_key_hash
37+K+V  8B      version (i64, little-endian)
45+K+V  8B      serial_number (u64, little-endian)
53+K+V  var     deactivated_sn_list (N × 8B, little-endian)
...     0-7B    padding (to 8-byte alignment)
...     16B     AES-GCM tag (only with tee_cipher feature)
```

**EntryBz** wraps a borrowed byte slice and provides zero-copy accessors:

```rust
pub struct EntryBz<'a> {
    pub bz: &'a [u8],
}

impl EntryBz {
    pub fn key(&self) -> &[u8];
    pub fn value(&self) -> &[u8];
    pub fn key_hash(&self) -> Hash32;
    pub fn next_key_hash(&self) -> &[u8];
    pub fn version(&self) -> i64;
    pub fn serial_number(&self) -> u64;
    pub fn hash(&self) -> Hash32;     // SHA256 of payload bytes
    pub fn dsn_count(&self) -> usize;
    pub fn dsn_iter(&self) -> DSNIter;
}
```

**Special entry types**:

- **Sentry entries**: Boundary markers created during `init_dir()`. 4096 per shard (65536 total). Their key hashes are deterministic 2-byte prefixes, not computed from actual keys. They partition the keyspace into non-overlapping ranges to prevent cross-shard iteration bugs.

- **Null entries**: Placeholder entries with `version = -2`, `serial_number = u64::MAX`. Used to initialize empty twig leaf slots.

**EntryVec**: A batched container holding entries across all shards in contiguous big buffers (64KB chunks). Used for bulk operations like stateless validation.

### Twig

**File**: [`qmdb/src/merkletree/twig.rs`](../qmdb/src/merkletree/twig.rs)

A Twig is a compact sub-tree containing 2048 entries.

```
              TwigRoot (Level 12)
             /                    \
        LeftRoot (L11)        ActiveBitsMTL3 (L10)
       /                      /                \
   11-level tree         ABits_MTL2[0]     ABits_MTL2[1]
   2048 leaves           /       \           /        \
   (entry hashes)   ABits_L1[0] ABits_L1[1] ABits_L1[2] ABits_L1[3]
                    ────────────────────────────────────────────────
                    8 pages × 32B = 256B ActiveBits (2048 bits)
```

```rust
pub struct Twig {
    pub active_bits_mtl1: [Hash32; 4],    // Level 8 hashes
    pub active_bits_mtl2: [Hash32; 2],    // Level 9 hashes
    pub active_bits_mtl3: Hash32,         // Level 10 hash
    pub left_root: Hash32,                // Root of 11-level entry tree
    pub twig_root: Hash32,                // hash2(11, left_root, active_bits_mtl3)
}
```

**TwigMT** (`[Hash32; 4096]`): The 11-level binary tree inside a twig. Nodes are numbered 1-4095:
- Node 1: root (= `left_root`)
- Nodes 2, 3: children of root
- Node N has children 2N and 2N+1
- Leaves at positions 2048-4095 contain entry hashes

**Global singletons** (computed once at startup):
- `NULL_MT_FOR_TWIG`: TwigMT with all null entry hashes
- `NULL_TWIG`: Twig with all-zero active bits
- `NULL_NODE_IN_HIGHER_TREE[64]`: Null hash at each tree level
- `NULL_ACTIVE_BITS`: All-zero 256-byte active bits

### Tree

**File**: [`qmdb/src/merkletree/tree.rs`](../qmdb/src/merkletree/tree.rs)

The Tree struct holds the complete Merkle state for one shard.

```rust
pub struct Tree {
    pub shard_id: usize,
    pub entry_file_wr: EntryFileWriter,        // Append-only entry storage
    pub twig_file_wr: TwigFileWriter,          // Append-only twig storage
    pub upper_tree: UpperTree,                 // In-DRAM upper levels
    pub youngest_twig_id: u64,                 // Current accumulation twig
    pub active_bits_shards: Vec<HashMap<u64, ActiveBits>>,
    pub mtree_for_youngest_twig: Box<TwigMT>,  // 131KB in-DRAM buffer
    // ... scratchpad fields for block execution
}
```

Key operations:
- `append_entry()`: Add entry to youngest twig, increment serial number
- `flush_files()`: Write twig data to SSD
- `deactivate()`: Clear an entry's active bit, mark for re-hashing
- `sync_mtree()`: Bottom-up hash synchronization of TwigMT
- `get_root()`: Compute global Merkle root from upper tree

### UpperTree

The nodes above twig level, stored entirely in DRAM.

```rust
pub struct UpperTree {
    pub my_shard_id: usize,
    // nodes[level - FIRST_LEVEL_ABOVE_TWIG][node_shard] = HashMap<NodePos, Hash32>
    pub nodes: Vec<Vec<HashMap<NodePos, [u8; 32]>>>,
    // active_twig_shards[twig_shard] = HashMap<twig_id, Box<Twig>>
    pub active_twig_shards: Vec<HashMap<u64, Box<twig::Twig>>>,
}
```

**NodePos** encodes a node's position as `(level << 56) | nth`:
```rust
pub struct NodePos(u64);
// Level: high 8 bits
// Nth: low 56 bits
// Children of (L, N): (L-1, 2N) and (L-1, 2N+1)
```

Nodes are sharded internally:
- `NODE_SHARD_COUNT = 4`: Upper tree nodes divided into 4 submaps per level
- `TWIG_SHARD_COUNT = 4`: Active twigs divided into 4 submaps

### EdgeNodes

When twigs are pruned, "edge nodes" preserve the ability to generate proofs for remaining entries.

```rust
pub struct EdgeNode {
    pub pos: NodePos,
    pub value: [u8; 32],
}
```

A node is an edge node if:
1. It's the twig root of the just-pruned twig (largest pruned twig ID), OR
2. It has both pruned descendants and non-pruned descendants

Edge nodes are persisted in MetaDB and restored during recovery.

```
 Pruned    Pruned    Active    Active    Active    (youngest)
┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐
│  ░░  │  │  ░░  │  │      │  │      │  │      │  │  ..  │
└──┬───┘  └──┬───┘  └──┬───┘  └──┬───┘  └──┬───┘  └──┬───┘
   A         B         │         │         │         │
   └────┬────┘         │         │         │         │
        C              │         │         │         │
        └──────┬───────┘         │         │         │
               D                 │         │         │
               └────────┬────────┘         │         │
                        │                  │         │
                        └──────┬───────────┘         │
                               │                     │
                               └──────────┬──────────┘
                                          │
                                        Root

Edge nodes: A, B, C, D (preserve proof path for active entries)
```

### ActiveBits

**File**: [`qmdb/src/merkletree/twig.rs`](../qmdb/src/merkletree/twig.rs)

256 bytes (2048 bits) per twig, one bit per entry:

```rust
pub struct ActiveBits([u8; 256]);

impl ActiveBits {
    pub fn set_bit(&mut self, offset: u32);   // Mark entry as active
    pub fn clear_bit(&mut self, offset: u32); // Mark entry as deactivated
    pub fn get_bit(&self, offset: u32) -> bool;
    pub fn get_bits(&self, page_num: usize, page_size: usize) -> &[u8];
}
```

ActiveBits are divided into 8 pages of 32 bytes for the right sub-tree of the twig:
- Pages 0-7 → hashed pairwise → `active_bits_mtl1[0..3]`
- `active_bits_mtl1[0..3]` → hashed pairwise → `active_bits_mtl2[0..1]`
- `active_bits_mtl2[0..1]` → hashed → `active_bits_mtl3`

---

## File Layer

### HPFile (Head-Prunable File)

**File**: [`hpfile/src/lib.rs`](../hpfile/src/lib.rs)

Normal files can't be truncated from the beginning. HPFile simulates a single large file using a sequence of fixed-size segments.

```
Segment 0          Segment 1          Segment 2
(pruned/deleted)   (current head)     (latest)
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  XXXXXXXXXX  │   │ data data    │   │ data data    │
│  XXXXXXXXXX  │   │ data data    │   │ data ...     │
└──────────────┘   └──────────────┘   └──────────────┘
 0-{seg_size}      {seg_size}-{2*seg}  {2*seg}-...
```

Key properties:
- **Append-only**: Data written is immutable
- **Head-prunable**: Delete earliest segments for garbage collection
- **Tail-truncatable**: Discard partial writes after crash recovery
- **Configurable segment size**: Default 1GB
- **Write-buffered**: Collects small writes into buffer-sized flushes
- **Optional DirectIO**: Bypass OS page cache on Linux (via `io_uring`)

### EntryFile

**File**: [`qmdb/src/entryfile/entryfile.rs`](../qmdb/src/entryfile/entryfile.rs)

Stores serialized entries using HPFile. One EntryFile per shard.

```rust
pub struct EntryFile {
    hp: Arc<HPFile>,
    cipher: Option<Aes256Gcm>,
}

impl EntryFile {
    pub fn read_entry(&self, pos: i64, buf: &mut [u8]) -> usize;
}
```

The `pos` (file position) returned by `append` is stored in the indexer for later retrieval.

### TwigFile

**File**: [`qmdb/src/merkletree/twigfile.rs`](../qmdb/src/merkletree/twigfile.rs)

Stores the internal Merkle nodes of finalized twigs. One TwigFile per shard. Each twig occupies a fixed-size region:
- 4095 nodes × 32 bytes = 131,040 bytes for the binary tree
- 12 bytes for the first entry's position (8B position + 4B checksum)
- Total: ~128 KB per twig

### MetaDB

**File**: [`qmdb/src/metadb.rs`](../qmdb/src/metadb.rs)

RocksDB-backed storage for metadata that must survive restarts:

| Key | Per-Shard | Description |
|---|---|---|
| `CurrHeight` | No | Latest committed block height |
| `EntryFileSize[i]` | Yes | Entry file size at commit |
| `TwigFileSize[i]` | Yes | Twig file size at commit |
| `NextSerialNum[i]` | Yes | Next available serial number |
| `OldestActiveSN[i]` | Yes | Oldest active entry's serial number |
| `OldestActiveFilePos[i]` | Yes | Oldest active entry's file position |
| `YoungestTwigID[i]` | Yes | Latest twig ID |
| `LastPrunedTwig[i]` | Yes | Most recently pruned twig ID |
| `EdgeNodes[i]` | Yes | Serialized edge node list |
| `RootHash[i]` | Yes | Merkle root hash |
| `ExtraData[height]` | No | User-provided block data (JSON) |

---

## Indexer

### InMemIndexer

**File**: [`qmdb/src/indexer/inmem.rs`](../qmdb/src/indexer/inmem.rs)

Pure in-memory B-tree. Maps 10-byte key hash prefix → file position.

**Design choices**:
- Uses 64-bit short key hash (first 8 bytes of SHA256) for the B-tree key
- Uses 48-bit file offset (divided by 8 since entries are 8-byte aligned), supporting up to 2048 TB
- 65536 shards for parallel access
- Hash collisions handled by iterating all matching positions
- Background compaction thread merges and cleans old entries

### HybridIndexer

**File**: [`qmdb/src/indexer/hybrid/`](../qmdb/src/indexer/hybrid/)

SSD-optimized indexer for very large datasets. Four components:

1. **Readonly disk files**: Sorted 14-byte records (8B key + 6B offset)
2. **Cache in DRAM**: Results from recent file lookups (per-block lifetime)
3. **Overlay in DRAM**: Latest add/delete/update operations
4. **ActiveBits in DRAM**: Tracks which disk records are still valid

Records are grouped into 32-record pages. A first-key list enables binary search to locate the correct page.

---

## Pipeline Architecture

### Prefetcher

**File**: [`qmdb/src/uniprefetcher.rs`](../qmdb/src/uniprefetcher.rs) (standard) / [`qmdb/src/dioprefetcher.rs`](../qmdb/src/dioprefetcher.rs) (direct I/O)

Receives tasks from the client. For Delete and Create operations, pre-reads the affected entries into EntryCache so the Updater can access them from memory instead of SSD.

- Uses a thread pool (default 512 threads) for parallel I/O
- Each shard has its own work queue
- Forwards processed tasks to the Updater via a channel

### Updater

**File**: [`qmdb/src/updater.rs`](../qmdb/src/updater.rs)

Processes each task's operations:

| Operation | Steps |
|---|---|
| **Update** | 1. Read old entry from cache/disk, deactivate it. 2. Create new entry with updated value/height. |
| **Delete** | 1. Deactivate old entry. 2. Read previous neighbor, deactivate it. 3. Create new neighbor entry with updated NextKeyHash. |
| **Create** | 1. Create new entry. 2. Read previous neighbor, deactivate it. 3. Create new neighbor with updated NextKeyHash. |

The Updater:
- Updates the B-tree index (add new position, remove old position)
- Sends new entries to the Flusher via EntryBuffer
- Handles out-of-order task IDs (tasks within a block may arrive in any order)

### Flusher

**File**: [`qmdb/src/flusher.rs`](../qmdb/src/flusher.rs)

The most complex pipeline stage. Performs 8 jobs per block:

| Job | Description | I/O |
|---|---|---|
| **Job 1** | Append new entries to EntryFile | SSD write |
| **Job 2** | Clear ActiveBits for deactivated entries | Memory |
| **Job 3** | Sync twig roots (hash entry trees + active bit trees) | CPU/GPU |
| **Job 4** | Evict inactive twigs from DRAM | Memory |
| **Job 5** | Flush EntryFile and TwigFile to SSD | SSD write |
| **Job 6** | Sync upper tree (propagate twig root changes upward) | CPU |
| **Job 7** | Prune old twigs (every 500 blocks) | SSD delete |
| **Job 8** | Update MetaDB (root hash, edge nodes, sizes) | SSD write |

**Pipelining**: Jobs 1-5 run in the repeating flusher thread. Jobs 6-8 are forked as one-time threads, allowing Job 1-2 of the next block to overlap with Job 6-8 of the current block.

**Interlock requirements**:
1. Job 6-8 threads of different blocks cannot overlap
2. Job 6 of block N+1 must wait for Job 3 of block N (twig roots shared)
3. Client cannot start block N+2 before Job 8 of block N completes

### Compactor

**File**: [`qmdb/src/compactor.rs`](../qmdb/src/compactor.rs)

Background garbage collection. When utilization drops below the configured ratio:

1. Scan the oldest active entries
2. For each active entry: re-create it (append as new entry, deactivate old)
3. This moves the "oldest active" boundary forward, allowing old twigs to be evicted and pruned

Compaction runs continuously on a dedicated thread per shard, throttled by the ring channel capacity.

---

## TaskHub and Block Processing

**File**: [`qmdb/src/tasks/bptaskhub.rs`](../qmdb/src/tasks/bptaskhub.rs)

The `BlockPairTaskHub` implements double-buffering: at most 2 blocks in flight.

```
Time ──────────────────────────────────────────────────▶

Block N:   ┌─Prefetch─┐ ┌──Update──┐ ┌──Flush J1-5──┐
                                            ┌──Flush J6-8──┐

Block N+1:              ┌─Prefetch─┐ ┌──Update──┐ ┌──Flush J1-5──┐
                                                        ┌──Flush J6-8──┐

Block N+2:                          [blocked until Job 8 of N completes]
                                    ┌─Prefetch─┐ ...
```

**ChangeSet** groups operations for a single transaction:

```rust
pub struct ChangeSet {
    pub data: Vec<u8>,       // Concatenated keys, values, key_hashes
    pub op_list: Vec<ChangeOp>,
    shard_starts: [u32; 16], // Per-shard index into op_list
    shard_op_count: [u32; 16],
}
```

Operations within a ChangeSet must be sorted by `(shard_id, key_hash, op_type)` before submission.

---

## Proof Generation

**File**: [`qmdb/src/merkletree/proof.rs`](../qmdb/src/merkletree/proof.rs)

A Merkle proof for entry at serial number `sn`:

```rust
pub struct ProofPath {
    pub left_of_twig: [ProofNode; 11],  // Entry tree path
    pub right_of_twig: [ProofNode; 3],  // ActiveBits tree path
    pub upper_path: Vec<ProofNode>,     // Twig root → global root
    pub serial_num: u64,
    pub root: [u8; 32],
}
```

**Proof verification** (`ProofPath::check()`):

1. Verify left path: hash entry leaf upward 11 levels to get `left_root`
2. Verify right path: hash active bits page upward 3 levels to get `active_bits_mtl3`
3. Compute `twig_root = hash2(11, left_root, active_bits_mtl3)`
4. Verify upper path: hash `twig_root` upward through sibling hashes to reach `root`

**Inclusion proof**: Entry exists with specific key/value at this serial number, and its ActiveBit = 1.

**Exclusion proof**: Two adjacent entries with `key_hash(A) < target < NextKeyHash(A)` prove no entry exists for `target`.

---

## Recovery

**File**: [`qmdb/src/merkletree/recover.rs`](../qmdb/src/merkletree/recover.rs)

On restart, SkippyDB rebuilds volatile state from MetaDB + SSD files:

1. **Read MetaDB**: Get heights, file sizes, edge nodes, serial numbers
2. **Truncate files**: If files are larger than MetaDB records (partial block), truncate to consistent state
3. **Rebuild trees** (parallel, one thread per shard):
   - Reconstruct twigs from entry file
   - Restore active bits by scanning entries
   - Restore upper tree from edge nodes + twig roots
4. **Rebuild index**: Scan active entries, populate B-tree

Recovery runs in parallel across all 16 shards using `thread::spawn`.

---

## Threading Model

| Component | Thread Count | Lifetime | Purpose |
|---|---|---|---|
| Prefetcher | 512 (pool) | Process lifetime | Pre-read entries for D/C ops |
| Prefetcher dispatch | 1 per shard | Process lifetime | Route tasks to updater |
| Updater | 16 (1/shard) | Process lifetime | Apply operations |
| Flusher (main) | 1 | Process lifetime | Jobs 1-5, spawns Job 6-8 |
| Flusher (upper tree) | 1 per block | One-time | Jobs 6-8 |
| Compactor | 16 (1/shard) | Process lifetime | Background GC |
| Indexer compaction | 1 | Process lifetime | B-tree maintenance |
| Rayon pool | auto | Process lifetime | Parallel Merkle hashing |
| Recovery | 16 (1/shard) | Startup only | Parallel tree rebuild |

**Synchronization primitives**:
- `SyncChannel`: Task queues between pipeline stages
- `Arc<RwLock<>>`: MetaDB access
- `DashMap`: Concurrent indexer maps
- `AtomPtr`: Lock-free BlockPairTaskHub switching
- `parking_lot::Mutex`: GpuHasher serialization
- `Condvar`: Proof request signaling

---

## Sharding

Data is partitioned into 16 shards by the first byte of `SHA256(key)`:

```rust
const SHARD_COUNT: usize = 16;
const SHARD_DIV: usize = (1 << 16) / SHARD_COUNT;  // 4096

fn byte0_to_shard_id(byte0: u8) -> usize {
    byte0 as usize * 256 / SHARD_DIV
}
```

Each shard owns:
- 1 EntryFile (HPFile on SSD)
- 1 TwigFile (HPFile on SSD, if enabled)
- 1 Tree (in-DRAM upper tree + youngest twig)
- Portion of B-tree index
- 1 Updater thread
- 1 Compactor thread

Shards are fully independent — no cross-shard locking during normal operation. The only synchronization point is the flusher barrier between Job 5 and Job 6, where all shards must complete before upper tree sync begins.

---

## Further Reading

- **Design rationale**: [docs/design.md](design.md) — Detailed explanation of why each data structure was chosen
- **GPU acceleration**: [docs/gpu-acceleration.md](gpu-acceleration.md) — CUDA kernel details, benchmarks, tuning
- **Paper**: [QMDB: Quick Merkle Database](https://arxiv.org/pdf/2501.05262) — Full academic paper with proofs
