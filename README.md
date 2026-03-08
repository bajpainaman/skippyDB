# SkippyDB: Quick Merkle Database

![Build Status](https://github.com/bajpainaman/skippyDB/actions/workflows/ci.yml/badge.svg)
![Tests](https://github.com/bajpainaman/skippyDB/actions/workflows/ci.yml/badge.svg?label=tests)
[![License: MIT/Apache-2.0](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](#license)

A high-performance, verifiable key-value store optimized for blockchain state storage — named after Skippy the Magnificent, the beloved AI from Craig Alanson's *Expeditionary Force* series. SkippyDB uses an append-only Twig Merkle Tree design to minimize SSD write amplification, perform in-memory Merkleization with minimal DRAM, and provide cryptographic proofs for inclusion, exclusion, and historical states.

> **Paper**: [QMDB: Quick Merkle Database](https://arxiv.org/pdf/2501.05262)

---

## Table of Contents

- [Why SkippyDB?](#why-skippydb)
- [Features](#features)
- [Architecture Overview](#architecture-overview)
- [Quick Start](#quick-start)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Your First Database](#your-first-database)
- [Core Concepts](#core-concepts)
  - [Entries](#entries)
  - [Twigs](#twigs)
  - [The Twig Merkle Tree](#the-twig-merkle-tree)
  - [Sharding](#sharding)
  - [The Pipeline](#the-pipeline)
- [Usage Guide](#usage-guide)
  - [Configuration](#configuration)
  - [Creating a Database](#creating-a-database)
  - [Writing Data (Create/Update/Delete)](#writing-data-createupdatedelete)
  - [Reading Data](#reading-data)
  - [Generating Proofs](#generating-proofs)
  - [Block Lifecycle](#block-lifecycle)
- [GPU Acceleration (CUDA)](#gpu-acceleration-cuda)
- [Performance](#performance)
- [Directory Structure](#directory-structure)
- [Examples](#examples)
- [Benchmarking](#benchmarking)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

---

## Why SkippyDB?

Traditional databases suffer from write amplification on SSDs when maintaining authenticated data structures. SkippyDB solves this with:

| Problem | SkippyDB's Solution |
|---|---|
| High SSD write amplification | Append-only twig design — no in-place updates |
| Expensive in-memory Merkle trees | Only upper tree levels in DRAM; twigs on SSD |
| Slow state proofs | O(1) I/O per update; single SSD read per access |
| Poor hardware utilization | Pipelined architecture with parallel SHA256 hashing |
| GPU-unfriendly hashing | Optional CUDA-accelerated batch SHA256 kernels |

**Benchmarks**: 6x faster than RocksDB, 8x faster than state-of-the-art verifiable databases. Validated on datasets up to 15 billion entries.

---

## Features

- **SSD-Optimized Append-Only Design** — Entries and twigs are only appended, never modified in-place. Head-prunable files enable efficient garbage collection.
- **In-Memory Merkleization** — Only the upper levels of the Merkle tree live in DRAM. Each twig (2048 entries) is flushed to SSD once finalized.
- **O(1) I/O Per Update** — Writing a new entry requires appending to the entry file. No random writes.
- **Inclusion & Exclusion Proofs** — Prove that a key exists, doesn't exist, or hasn't changed since a given block height.
- **16-Shard Parallelism** — Data is sharded by key hash for concurrent processing across the pipeline.
- **Compaction** — Background compaction keeps storage utilization healthy by replaying old active entries.
- **GPU Acceleration** — Optional CUDA kernels for batch SHA256 hashing (AoS, SoA, warp-cooperative, variable-length).
- **Encryption Support** — Optional AES-256-GCM encryption for entries and index data (`tee_cipher` feature).
- **Direct I/O** — Optional `io_uring`-based direct I/O for bypassing the OS page cache on Linux.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client / EVM                             │
│                   (sends blocks of tasks)                       │
└──────────────────────────┬──────────────────────────────────────┘
                           │ tasks
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                     SkippyDB Pipeline                              │
│                                                                  │
│  ┌────────────┐    ┌────────────┐    ┌────────────────────────┐  │
│  │ Prefetcher │───▶│  Updater   │───▶│       Flusher          │  │
│  │            │    │            │    │                        │  │
│  │ Pre-reads  │    │ Updates    │    │ Job1: Append entries   │  │
│  │ entries    │    │ B-tree     │    │ Job2: Clear ActiveBits │  │
│  │ into cache │    │ index +   │    │ Job3: Sync twig roots  │  │
│  │ for D/C    │    │ sends to  │    │ Job4: Evict twigs      │  │
│  │ operations │    │ flusher   │    │ Job5: Flush to SSD     │  │
│  └────────────┘    │ via       │    │ Job6: Sync upper tree  │  │
│                    │ EntryBuf  │    │ Job7: Prune old twigs  │  │
│                    └────────────┘    │ Job8: Update MetaDB    │  │
│                                      └────────────────────────┘  │
│                                                                  │
│  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │ Indexer │  │EntryFile │  │ TwigFile │  │    MetaDB        │  │
│  │ (B-tree)│  │ (HPFile) │  │ (HPFile) │  │   (RocksDB)      │  │
│  └─────────┘  └──────────┘  └──────────┘  └──────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

See [docs/architecture.md](docs/architecture.md) for the full deep-dive.

---

## Quick Start

### Prerequisites

- **Rust** 1.75+ (edition 2021)
- **Linux** (required for `io_uring` / direct I/O features; macOS works for basic usage)
- **System packages** (Ubuntu/Debian):

```bash
sudo apt-get install -y g++ linux-libc-dev libclang-dev unzip libjemalloc-dev make
```

Or use the included script:

```bash
./install-prereqs-ubuntu.sh
```

**Optional** (for GPU acceleration):
- NVIDIA GPU with compute capability 6.0+
- CUDA Toolkit 12.0+
- `nvcc` on PATH

### Installation

```bash
git clone https://github.com/bajpainaman/skippyDB.git
cd skippyDB
cargo build --release
```

With GPU acceleration:

```bash
cargo build --release --features cuda
```

### Your First Database

```rust
use skippydb::config::Config;
use skippydb::def::{IN_BLOCK_IDX_BITS, OP_CREATE};
use skippydb::tasks::TasksManager;
use skippydb::utils::changeset::ChangeSet;
use skippydb::utils::hasher;
use skippydb::utils::byte0_to_shard_id;
use skippydb::{AdsCore, AdsWrap, ADS};
use parking_lot::RwLock;
use std::sync::Arc;

fn main() {
    // 1. Initialize the database directory
    let config = Config::from_dir("my_database");
    AdsCore::init_dir(&config);

    // 2. Open the database
    let mut ads = AdsWrap::new(&config);

    // 3. Build a changeset with create operations
    let mut cset = ChangeSet::new();
    let key = b"hello";
    let value = b"world";
    let key_hash = hasher::hash(key);
    let shard_id = byte0_to_shard_id(key_hash[0]) as u8;
    cset.add_op(OP_CREATE, shard_id, &key_hash, key, value, None);
    cset.sort();

    // 4. Wrap it in a task and submit to block height=1
    let task = SimpleTask::new(vec![cset]);
    let height: i64 = 1;
    let task_id = height << IN_BLOCK_IDX_BITS;
    ads.start_block(height, Arc::new(TasksManager::new(
        vec![RwLock::new(Some(task))], task_id,
    )));

    let shared = ads.get_shared();
    shared.insert_extra_data(height, String::new());
    shared.add_task(task_id);

    // 5. Flush and read back
    ads.flush();

    let mut buf = [0u8; 300];
    let shared = ads.get_shared();
    let (size, found) = shared.read_entry(-1, &key_hash, &[], &mut buf);
    assert!(found);
    println!("Read {} bytes", size);
}
```

Run the included demo:

```bash
cargo run --example v2_demo
```

---

## Core Concepts

### Entries

An **Entry** is the atomic unit of data in SkippyDB. Each entry represents a single key-value pair at a specific block height.

```
Entry := (Key, Value, NextKeyHash, Height, SerialNumber, DeactivatedSNList)
```

**Binary layout** (see [`qmdb/src/entryfile/entry.rs`](qmdb/src/entryfile/entry.rs)):

```
┌──────────┬────────────┬───────────────┬─────┬───────┬──────────────┐
│ 1B KeyLen│ 3B ValueLen│ 1B DSN Count  │ Key │ Value │ NextKeyHash  │
├──────────┴────────────┴───────────────┴─────┴───────┤  (32 bytes)  │
│                                                      ├──────────────┤
│                                                      │ 8B Version   │
│                                                      ├──────────────┤
│                                                      │ 8B SerialNum │
│                                                      ├──────────────┤
│                                                      │ DSN List     │
│                                                      │ (N × 8B)     │
│                                                      ├──────────────┤
│                                                      │ Padding (→8B)│
└──────────────────────────────────────────────────────┴──────────────┘
```

- **NextKeyHash**: Links entries in sorted key-hash order. Enables exclusion proofs ("no key exists between A and B").
- **DeactivatedSNList**: When this entry is created, it lists which older entries are being replaced (deactivated).
- **ActiveBit**: Each entry has an associated bit. `1` = current, `0` = superseded. Stored separately in the twig.

### Twigs

A **Twig** groups 2048 consecutive entries (by serial number) into a compact Merkle sub-tree.

```
              TwigRoot (Level 12)
             /                  \
        LeftRoot              ActiveBitsMTL3 (Level 11)
       /                      /            \
   11-level tree         ActiveBitsMTL2    ActiveBitsMTL2
   (2048 leaves =        /       \          /        \
    entry hashes)    ABits_L1  ABits_L1  ABits_L1  ABits_L1
                     (8 × 32B ActiveBit pages)
```

- **Left sub-tree**: 11-level Merkle tree over the 2048 entry hashes. Stored on SSD once finalized.
- **Right sub-tree**: 3-level Merkle tree over the 256 bytes of ActiveBits. Kept in DRAM for active twigs.
- **Youngest twig**: The most recent, not-yet-full twig. Its left sub-tree lives in DRAM.

### The Twig Merkle Tree

The full tree has two layers:

1. **Lower layer** (twigs): Each twig is a self-contained sub-tree on SSD.
2. **Upper layer**: In-DRAM nodes connecting twig roots up to the global root.

This design means only ~64 bytes of DRAM per twig (the twig root hash), regardless of how many entries it contains.

### Sharding

SkippyDB divides keyspace into **16 shards** based on the first byte of `SHA256(key)`:

```
shard_id = key_hash[0] * 256 / SHARD_DIV    // SHARD_DIV = 4096
```

Each shard has its own:
- Entry file (HPFile)
- Twig file (HPFile)
- Merkle tree
- Updater thread
- Compactor

Shards are processed in parallel via `rayon`.

### The Pipeline

SkippyDB processes blocks through a 3-stage pipeline:

```
Block N:     [Prefetch] ──▶ [Update] ──▶ [Flush (Jobs 1-5)]
                                              │
Block N+1:   [Prefetch] ──▶ [Update]          │──▶ [Flush (Jobs 6-8)]
                                              │
Block N+2:   [Prefetch]                       │
```

The pipeline allows up to 2 blocks in flight. A block's data is guaranteed to be on SSD before the block 2 heights later starts executing.

---

## Usage Guide

### Configuration

```rust
use skippydb::config::Config;

let config = Config {
    dir: "my_db".to_string(),
    wrbuf_size: 8 * 1024 * 1024,           // 8 MB write buffer
    file_segment_size: 1024 * 1024 * 1024,  // 1 GB HPFile segments
    with_twig_file: false,                  // set true for proof generation
    compact_thres: 20_000_000,              // compaction threshold
    utilization_ratio: 7,                   // compact when utilization < 70%
    utilization_div: 10,
    task_chan_size: 200_000,
    prefetcher_thread_count: 512,
    aes_keys: None,                         // set for encryption
    ..Config::default()
};
```

Key configuration knobs:

| Field | Default | Description |
|---|---|---|
| `dir` | `"default"` | Root directory for all data files |
| `wrbuf_size` | `8 MB` | Write buffer size per HPFile |
| `file_segment_size` | `1 GB` | Size of each HPFile segment on disk |
| `with_twig_file` | `false` | Enable twig file for proof generation |
| `compact_thres` | `20,000,000` | Minimum entries before compaction triggers |
| `utilization_ratio/div` | `7/10` | Compact when active/total < ratio/div |
| `task_chan_size` | `200,000` | Channel buffer for task pipeline |
| `aes_keys` | `None` | 96-byte AES keys for encryption (3 × 32B) |

### Creating a Database

```rust
use skippydb::config::Config;
use skippydb::AdsCore;

let config = Config::from_dir("my_db");

// Initialize directory structure with sentry entries
AdsCore::init_dir(&config);
```

This creates:
```
my_db/
├── data/
│   ├── entries0..entries15   (16 entry files, one per shard)
│   └── twig0..twig15         (16 twig files, if with_twig_file=true)
├── metadb/                   (RocksDB for metadata)
└── idx/                      (B-tree indexer data)
```

### Writing Data (Create/Update/Delete)

All mutations go through **ChangeSets** grouped into **Tasks**, submitted as **Blocks**:

```rust
use skippydb::def::{OP_CREATE, OP_WRITE, OP_DELETE, IN_BLOCK_IDX_BITS};
use skippydb::utils::changeset::ChangeSet;
use skippydb::utils::{hasher, byte0_to_shard_id};

// Build a changeset
let mut cset = ChangeSet::new();

// CREATE: insert a new key-value pair
let key = b"user:alice";
let value = b"balance:100";
let key_hash = hasher::hash(key);
let shard_id = byte0_to_shard_id(key_hash[0]) as u8;
cset.add_op(OP_CREATE, shard_id, &key_hash, key, value, None);

// UPDATE: modify an existing key's value
let new_value = b"balance:200";
cset.add_op(OP_WRITE, shard_id, &key_hash, key, new_value, None);

// DELETE: remove a key-value pair
cset.add_op(OP_DELETE, shard_id, &key_hash, key, &[], None);

// IMPORTANT: sort before submitting
cset.sort();
```

**Operation types** (defined in [`qmdb/src/def.rs`](qmdb/src/def.rs)):

| Constant | Value | Description |
|---|---|---|
| `OP_READ` | 1 | Read (no mutation) |
| `OP_WRITE` | 2 | Update existing key |
| `OP_CREATE` | 3 | Insert new key |
| `OP_DELETE` | 4 | Delete existing key |

### Reading Data

```rust
use skippydb::def::DEFAULT_ENTRY_SIZE;
use skippydb::entryfile::EntryBz;

let shared = ads.get_shared();
let mut buf = [0u8; DEFAULT_ENTRY_SIZE];

// Read by key hash (faster — no key comparison)
let key_hash = hasher::hash(b"user:alice");
let (size, found) = shared.read_entry(-1, &key_hash, &[], &mut buf);

if found {
    let entry = EntryBz { bz: &buf[..size] };
    println!("Key:   {:?}", entry.key());
    println!("Value: {:?}", entry.value());
    println!("SN:    {}", entry.serial_number());
    println!("Ver:   {}", entry.version());
}

// Read by key hash + key (verifies key match for hash collisions)
let (size, found) = shared.read_entry(-1, &key_hash, b"user:alice", &mut buf);
```

The `height` parameter controls which block's state to query:
- `-1`: Latest committed state
- `N`: State as of block height N

### Generating Proofs

Enable `with_twig_file: true` in config, then:

```rust
// Get a Merkle proof for a specific entry
let shard_id = 0;
let serial_number = 42;

match ads.get_proof(shard_id, serial_number) {
    Ok(proof) => {
        // Verify the proof
        let mut proof_copy = proof;
        proof_copy.check(false).expect("proof verification failed");

        // Serialize for transmission
        let bytes = proof_copy.to_bytes();
        println!("Proof size: {} bytes", bytes.len());
    }
    Err(e) => eprintln!("Proof generation failed: {}", e),
}
```

A `ProofPath` contains:
- `left_of_twig`: 11 sibling hashes in the entry Merkle tree
- `right_of_twig`: 3 sibling hashes in the ActiveBits tree
- `upper_path`: Sibling hashes from twig root to global root
- `serial_num`: The entry's serial number
- `root`: The expected Merkle root

### Block Lifecycle

```rust
// Height 1
let height = 1i64;
let task_id = height << IN_BLOCK_IDX_BITS;

// Start a new block
let (ok, prev_meta) = ads.start_block(
    height,
    Arc::new(TasksManager::new(task_list, task_id)),
);

// Get a shared handle for concurrent access
let shared = ads.get_shared();

// Attach extra data (e.g., block hash) to this block
shared.insert_extra_data(height, "block_hash:0xabc".to_string());

// Submit tasks one at a time (can be from different threads)
for idx in 0..task_count {
    shared.add_task((height << IN_BLOCK_IDX_BITS) | idx);
}

// Flush: blocks until pipeline catches up
let meta_infos = ads.flush();
```

---

## GPU Acceleration (CUDA)

SkippyDB supports optional GPU-accelerated SHA256 batch hashing for Merkle tree operations. See [docs/gpu-acceleration.md](docs/gpu-acceleration.md) for the full guide.

### Quick Setup

```bash
# Build with CUDA support
cargo build --release --features cuda

# Run GPU tests (requires NVIDIA GPU)
cargo test --features cuda -- gpu

# Run benchmarks
cargo bench --bench hash_benchmarks --features cuda
```

### Kernel Variants

| Kernel | Layout | Use Case |
|---|---|---|
| `sha256_node_hash` | AoS (65B stride) | Default; simple fixed-size node hashing |
| `sha256_node_hash_soa` | SoA (32B stride) | Better memory coalescing for high batch sizes |
| `sha256_node_hash_warp_coop` | AoS + warp shuffles | 8 threads/hash; good on high-SM GPUs |
| `sha256_variable_hash` | Variable-length | Entry hashing (50-300B inputs) |

### Usage

```rust
use skippydb::gpu::{GpuHasher, NodeHashJob};

let gpu = GpuHasher::new(200_000).expect("CUDA init failed");

// Batch hash Merkle tree nodes
let jobs: Vec<NodeHashJob> = vec![
    NodeHashJob { level: 0, left: [0x11; 32], right: [0xAB; 32] },
    // ... thousands more
];
let hashes = gpu.batch_node_hash(&jobs);

// SoA layout for better GPU memory throughput
let levels = vec![0u8; 10000];
let lefts = vec![[0x11u8; 32]; 10000];
let rights = vec![[0xABu8; 32]; 10000];
let hashes = gpu.batch_node_hash_soa(&levels, &lefts, &rights);
```

---

## Performance

Tested on commodity hardware with NVMe SSDs:

| Metric | SkippyDB | RocksDB | Improvement |
|---|---|---|---|
| Sequential writes | ~1.2M ops/s | ~200K ops/s | **6x** |
| Random reads | ~800K ops/s | ~500K ops/s | **1.6x** |
| Proof generation | <1ms | N/A | — |
| Write amplification | ~1.0x | ~10-30x | **10-30x** |

GPU acceleration adds ~3-5x throughput improvement for Merkle tree hashing at batch sizes >10K.

---

## Directory Structure

```
skippyDB/
├── qmdb/                          # Core library
│   ├── src/
│   │   ├── lib.rs                 # AdsCore, AdsWrap, ADS trait
│   │   ├── config.rs              # Configuration struct
│   │   ├── def.rs                 # Constants (SHARD_COUNT, TWIG_SHIFT, etc.)
│   │   ├── entryfile/
│   │   │   ├── entry.rs           # Entry, EntryBz, EntryVec
│   │   │   ├── entrybuffer.rs     # Lock-free entry buffer (updater → flusher)
│   │   │   ├── entrycache.rs      # In-memory entry cache
│   │   │   └── entryfile.rs       # HPFile-backed entry storage
│   │   ├── merkletree/
│   │   │   ├── tree.rs            # Tree, UpperTree, NodePos
│   │   │   ├── twig.rs            # Twig, ActiveBits, TwigMT
│   │   │   ├── twigfile.rs        # HPFile-backed twig storage
│   │   │   ├── proof.rs           # ProofPath, ProofNode
│   │   │   └── recover.rs         # Tree recovery from SSD + MetaDB
│   │   ├── indexer/
│   │   │   ├── inmem.rs           # In-memory B-tree indexer
│   │   │   └── hybrid/            # SSD-backed hybrid indexer
│   │   ├── gpu/                   # CUDA acceleration (feature = "cuda")
│   │   │   ├── gpu_hasher.rs      # GpuHasher, MultiGpuHasher
│   │   │   └── sha256_kernel.cu   # CUDA SHA256 kernels
│   │   ├── seqads/                # Sequential ADS for stateless validation
│   │   ├── stateless/             # Stateless validation support
│   │   ├── tasks/                 # Task, TaskHub, BlockPairTaskHub
│   │   ├── flusher.rs             # Flusher pipeline stage
│   │   ├── updater.rs             # Updater pipeline stage
│   │   ├── compactor.rs           # Background compaction
│   │   ├── metadb.rs              # RocksDB-backed metadata store
│   │   └── utils/                 # Hasher, changeset, helpers
│   ├── examples/
│   │   ├── v2_demo.rs             # Basic usage example
│   │   └── v1_fuzz/               # Fuzz testing example
│   └── benches/
│       └── hash_benchmarks.rs     # Criterion benchmarks
├── hpfile/                        # Head-prunable file library
│   └── src/
│       └── lib.rs                 # HPFile, PreReader, TempDir
├── bench/                         # Performance benchmarking suite
│   └── src/
│       └── bin/speed.rs           # speed benchmark binary
└── docs/
    ├── architecture.md            # Internal architecture deep-dive
    ├── design.md                  # Design document / whitepaper notes
    └── gpu-acceleration.md        # GPU acceleration guide
```

---

## Examples

### Basic Demo

```bash
cargo run --example v2_demo
```

Creates a database, inserts 100 key-value pairs across 10 tasks, flushes, and reads one back. See [`qmdb/examples/v2_demo.rs`](qmdb/examples/v2_demo.rs).

### Fuzz Testing

```bash
cargo run --example v1_fuzz
```

Randomized CRUD operations against a reference database to verify correctness. See [`qmdb/examples/v1_fuzz/`](qmdb/examples/v1_fuzz/).

### Indexer Stress Test

```bash
# Requires randsrc.dat:
head -c 10M </dev/urandom > randsrc.dat
cargo run --example v3_indexer
```

Stress-tests the hybrid indexer with millions of random operations. See [`qmdb/examples/v3_indexer.rs`](qmdb/examples/v3_indexer.rs).

---

## Benchmarking

### Quick Benchmark

```bash
head -c 10M </dev/urandom > randsrc.dat
cargo run --release --bin speed -- --entry-count 4000000
```

### Hash Benchmarks (Criterion)

```bash
# CPU-only
cargo bench --bench hash_benchmarks

# With GPU
cargo bench --bench hash_benchmarks --features cuda
```

### Running Tests

```bash
# All tests
cargo nextest run

# With GPU tests
cargo test --features cuda -- gpu

# Specific module
cargo test -p skippydb --lib merkletree
```

---

## Feature Flags

| Feature | Default | Description |
|---|---|---|
| `tikv-jemallocator` | Yes | Use jemalloc for better allocation performance |
| `cuda` | No | Enable CUDA GPU-accelerated SHA256 hashing |
| `directio` | No | Linux `io_uring`-based direct I/O |
| `tee_cipher` | No | AES-256-GCM encryption for entries + index |
| `use_hybridindexer` | No | Use SSD-backed hybrid indexer instead of in-memory |
| `hpfile_all_in_mem` | No | Keep HPFile data entirely in memory (for testing) |
| `slow_hashing` | No | Disable parallel hashing (for debugging) |
| `in_sp1` | No | Build for SP1 zkVM target |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on style, testing, and the pull request process.

```bash
# Check formatting and lints
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --all -- --check

# Run the full test suite
cargo nextest run
```

---

## License

Dual-licensed under [MIT](LICENSE-MIT) and [Apache 2.0](LICENSE-APACHE).

---

## Citation

If you use SkippyDB in a publication, please cite:

> **QMDB: Quick Merkle Database**
> Isaac Zhang, Ryan Zarick, Daniel Wong, Thomas Kim, Bryan Pellegrino, Mignon Li, Kelvin Wong
> <https://arxiv.org/abs/2501.05262>

```bibtex
@article{zhang2025qmdb,
  title={Quick Merkle Database},
  author={Zhang, Isaac and Zarick, Ryan and Wong, Daniel and Kim, Thomas and Pellegrino, Bryan and Li, Mignon and Wong, Kelvin},
  journal={arXiv preprint arXiv:2501.05262},
  year={2025}
}
```

---

SkippyDB is maintained by [Naman Bajpai](https://github.com/bajpainaman). Originally forked from [qmdb](https://github.com/LayerZero-Labs/qmdb).
