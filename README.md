# SkippyDB

<p align="center">
  <img src="docs/figures/skippy.svg" alt="Skippy the Magnificent — a tiny shiny beer can with two glowing cyan eyes and a tiny gold crown floating above" width="220" />
</p>
<p align="center"><sub><i>Skippy. Magnificent. (Self-described.)</i></sub></p>

> *"I'm Skippy the Magnificent. An ancient, godlike intelligence stuck inside a tiny shiny beer can. Do try to keep up."*
>
> — Skippy, on himself, frequently, unsolicited

A GPU-accelerated, append-only Merkle key-value store for blockchain state.
Built around a tiny shiny coordinator (the upper-tree root) that tells a fleet
of much-less-magnificent shards what to do. Targets multi-million-ops/s
sustained throughput on a single host with verifiable read/write proofs and
~1× SSD write amplification.

> *"Yes, monkey, the throughput is good. You may now go fetch me a benchmark."*

![Build Status](https://github.com/bajpainaman/SkippyDB/actions/workflows/build.yml/badge.svg)
![Tests](https://github.com/bajpainaman/SkippyDB/actions/workflows/tests.yml/badge.svg)
[![License: MIT/Apache-2.0](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](#license)

> Forked from QMDB ([paper](https://arxiv.org/pdf/2501.05262)). Named after
> Skippy the Magnificent from Craig Alanson's *Expeditionary Force* — an
> arrogant ancient AI who keeps a fleet of squishy hairless apes alive
> despite their many faults. Mostly metaphorical. The `skippydb` crate name
> is the only public-API rename from upstream; on-disk format and most
> internal module structure remain compatible with QMDB's design. Phase 2
> introduced a `MetaInfoV2` envelope — see
> [On-Disk Format](#on-disk-format) before pointing this at a pre-fork DB.
> *Skippy notes that this disclaimer was added by a monkey.*


---

## Headline numbers

Measured on `skippy-dev` (AMD Ryzen 9 5900X · NVIDIA RTX 4080 SUPER · 46 GB
RAM · ext4 on a single NVMe Gen4 SSD), 40-million-entry cuda bench, no env
flags, raw JSON in `bench/results/perlevel-default-40m.json`:

| metric | value |
|---|---:|
| Sequential updates | **1.35M ops/s** |
| Random reads | **1.60M ops/s** |
| Inserts | **1.12M ops/s** |
| End-to-end transactions | **47.5K txns/s** |
| Block population | **11.1 blocks/s** (100K ops/block) |
| Wall-clock for the whole 40M bench | **49.5 s** |
| Proof generation | <1 ms |
| Write amplification | ~1.0× |

That's **~4.3× the throughput of `main`** and **~3.6× the throughput of the
prior moonshot baseline** at the same workload. See
[`bench/results/`](bench/results/) for every interim run and the
[A/B comparisons](#performance-history) below for the path that got there.

Reproduce:

```bash
head -c 10M </dev/urandom > randsrc.dat
cargo run --release --features cuda --bin speed -- --entry-count 40000000
```

---

## Why SkippyDB?

| Problem | SkippyDB |
|---|---|
| SSD write amplification on authenticated stores | Append-only twig design — entries and twigs are never modified in place |
| Expensive in-memory Merkle trees | Only the upper-tree levels live in DRAM; each twig (2048 entries) ships to SSD once finalized |
| Slow state proofs | O(1) I/O per update; one SSD read per access; <1ms inclusion / exclusion / historical proofs |
| Single-threaded hashing | 16-shard parallel pipeline + GPU-batched SHA256 (AoS, SoA, warp-cooperative variants) |
| Thread-pool backpressure | Lock-free `EntryBuffer` between updater and flusher; per-block `AtomicUsize` countdown commit |

Validated correctness via deterministic byte-level parity tests across the
CPU and GPU code paths
([`qmdb/tests/sync_upper_nodes_parity.rs`](qmdb/tests/sync_upper_nodes_parity.rs),
[`active_bits_sync_parity.rs`](qmdb/tests/active_bits_sync_parity.rs),
[`twig_sync_parity.rs`](qmdb/tests/twig_sync_parity.rs),
[`cpu_gpu_sha256_parity.rs`](qmdb/tests/cpu_gpu_sha256_parity.rs)). Every
sync-upper-nodes change has to clear those gates before merging.

---

## Quick start

### Prerequisites

```bash
# Ubuntu / Debian
sudo apt-get install -y g++ linux-libc-dev libclang-dev unzip libjemalloc-dev make
# Or:
./install-prereqs-ubuntu.sh
```

- Rust 1.75+ (edition 2021)
- Linux for `io_uring` / direct I/O features (macOS works for the basic API)
- For CUDA: NVIDIA GPU compute capability 6.0+, CUDA Toolkit 12.0+, `nvcc` on PATH

### Build

```bash
git clone https://github.com/bajpainaman/SkippyDB.git
cd SkippyDB
cargo build --release                  # CPU-only
cargo build --release --features cuda  # with GPU SHA256 kernels
```

### Hello, SkippyDB

```rust
use skippydb::config::Config;
use skippydb::def::{IN_BLOCK_IDX_BITS, OP_CREATE};
use skippydb::tasks::TasksManager;
use skippydb::test_helper::SimpleTask;
use skippydb::utils::byte0_to_shard_id;
use skippydb::utils::changeset::ChangeSet;
use skippydb::utils::hasher;
use skippydb::{AdsCore, AdsWrap};
use parking_lot::RwLock;
use std::sync::Arc;

fn main() {
    let config = Config::from_dir("my_database");
    AdsCore::init_dir(&config);
    let mut ads = AdsWrap::<SimpleTask>::new(&config);

    // Build a CREATE op for ("hello", "world")
    let key = b"hello";
    let value = b"world";
    let key_hash = hasher::hash(key);
    let shard_id = byte0_to_shard_id(key_hash[0]) as u8;

    let mut cset = ChangeSet::new();
    cset.add_op(OP_CREATE, shard_id, &key_hash, key, value, None);
    cset.sort();

    // Submit as a single-task block at height 1
    let height: i64 = 1;
    let task_id = height << IN_BLOCK_IDX_BITS;
    let task = SimpleTask::new(vec![cset]);
    ads.start_block(
        height,
        Arc::new(TasksManager::new(vec![RwLock::new(Some(task))], task_id)),
    );

    let shared = ads.get_shared();
    shared.insert_extra_data(height, String::new());
    shared.add_task(task_id);
    ads.flush();

    // Read it back
    let mut buf = [0u8; 300];
    let (size, found) = shared.read_entry(-1, &key_hash, &[], &mut buf);
    assert!(found);
    println!("Read {} bytes back", size);
}
```

Or just run the bundled example:

```bash
cargo run --example v2_demo
```

---

## Architecture (one diagram)

```
                 client / EVM submits blocks of tasks
                                 │
                                 ▼
┌────────────────────────────────────────────────────────────────────┐
│                         SkippyDB pipeline                          │
│                                                                    │
│    Prefetcher  ──▶  Updater  ──▶  Flusher                          │
│    (warms        (B-tree         (8 jobs per block: append +       │
│     cache         indexer +       deactivate + sync twig roots +   │
│     for D/C)      EntryBuf)       evict + flush + sync upper +     │
│                                   prune + commit MetaDB)           │
│                                                                    │
│    16-shard parallelism on every stage; per-shard updater + tree.  │
│                                                                    │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────────┐   │
│  │ Indexer  │ │EntryFile │ │ TwigFile │ │       MetaDB         │   │
│  │ (B-tree, │ │ (HPFile, │ │ (HPFile) │ │ (custom 2-file       │   │
│  │  in-mem  │ │  append- │ │          │ │  ping-pong, V2       │   │
│  │  or SSD) │ │  only)   │ │          │ │  envelope)           │   │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────────────┘   │
└────────────────────────────────────────────────────────────────────┘
```

### Key pieces

- **Twig**: 2048-entry Merkle sub-tree. Lower 11 levels = entry hashes; upper 3 levels = ActiveBits. Once a twig fills, it's serialized to SSD and only the twig-root hash stays in DRAM. ~64 bytes DRAM per twig regardless of entry size.
- **Upper tree**: in-DRAM Merkle nodes connecting twig roots to the global root. Computed on GPU by default (`sync_upper_nodes_gpu`, per-level kernel launches). Resident-store path (`sync_upper_nodes_gpu_resident`) is opt-in (see [Env toggles](#env-toggles)).
- **EntryBuffer**: lock-free ring buffer feeding entries from updater to flusher. Per-block `AtomicUsize` countdown picks the last shard to commit MetaDB — no shard-0 hardcode, no double-barrier dance.
- **Sharding**: `shard_id = first_byte_of(SHA256(key)) >> 4` → 16 shards. Each gets its own entry file, twig file, indexer slice, updater thread, compactor.
- **Pipeline depth = 2**: block N+2 is allowed to start prefetching while block N is mid-flush. Empirically depth>2 regresses on this hardware tier (see `TODO.md`).

The deep dive lives at [`docs/architecture.md`](docs/architecture.md).

---

## Core concepts

### Entry layout (`qmdb/src/entryfile/entry.rs`)

```
┌──────────┬────────────┬──────────────┬──────┬───────┬──────────────┐
│ 1B KeyLen│ 3B ValueLen│ 1B DSN Count │ Key  │ Value │ NextKeyHash  │
└──────────┴────────────┴──────────────┴──────┴───────┤  (32 bytes)  │
                                                     ├──────────────┤
                                                     │ 8B Version   │
                                                     ├──────────────┤
                                                     │ 8B SerialNum │
                                                     ├──────────────┤
                                                     │ DSN List     │
                                                     │ (N × 8B)     │
                                                     ├──────────────┤
                                                     │ Padding (→8B)│
                                                     └──────────────┘
```

- `NextKeyHash` links entries in sorted key-hash order — enables exclusion proofs ("no key exists between A and B").
- `DeactivatedSNList` lists older entries this entry replaces.
- `ActiveBit` (one bit per SN, stored separately in the twig) tracks live/superseded.

### Operations

| op | constant | semantics |
|---|---:|---|
| `OP_READ` | 1 | Read, no mutation. Used for cache warming + debug. |
| `OP_WRITE` | 2 | Update an existing key (deactivate old SN, append new entry). |
| `OP_CREATE` | 3 | Insert a brand-new key. |
| `OP_DELETE` | 4 | Delete an existing key. |

Group ops into a `ChangeSet`, sort it, wrap in a `Task`, submit at a block height. See [`Hello, SkippyDB`](#hello-skippydb) above.

### Reading

```rust
let mut buf = [0u8; skippydb::def::DEFAULT_ENTRY_SIZE];
let (size, found) = shared.read_entry(
    -1,         // height: -1 = latest committed; N = state at height N
    &key_hash,
    &[],        // optional key bytes for collision check; pass &[] to skip
    &mut buf,
);
if found {
    let entry = skippydb::entryfile::EntryBz { bz: &buf[..size] };
    let value = entry.value();
    // ...
}
```

### Proofs

Set `with_twig_file: true` in the config first.

```rust
let proof = ads.get_proof(shard_id, serial_number)?;
proof.check(false)?;          // verify
let bytes = proof.to_bytes(); // serialize for transmission
```

A `ProofPath` carries: 11 sibling hashes for the twig's left sub-tree, 3 for the active-bits sub-tree, and the upper-tree path from the twig root to the global root.

---

## Configuration

```rust
use skippydb::config::Config;

let config = Config {
    dir: "my_db".into(),
    wrbuf_size: 8 * 1024 * 1024,            // 8 MB write buffer per HPFile
    file_segment_size: 1024 * 1024 * 1024,  // 1 GB segments
    with_twig_file: false,                  // true → enables proof generation
    compact_thres: 20_000_000,              // entries before compaction triggers
    utilization_ratio: 7,                   // compact when active/total < 7/10
    utilization_div: 10,
    task_chan_size: 200_000,
    prefetcher_thread_count: 512,
    aes_keys: None,                         // 96-byte AES-256-GCM keys (3 × 32B)
    ..Config::default()
};
```

| Field | Default | Notes |
|---|---|---|
| `dir` | `"default"` | Data root (entries, twigs, MetaDB, idx all live here). |
| `wrbuf_size` | `8 MB` | Per-HPFile write buffer. |
| `file_segment_size` | `1 GB` | HPFile segment rotation. |
| `with_twig_file` | `false` | Required for `get_proof`. |
| `compact_thres` | `20,000,000` | Compaction starts above this entry count. |
| `utilization_ratio / div` | `7 / 10` | Compact when liveness drops below the ratio. |
| `task_chan_size` | `200,000` | Pipeline buffer. |
| `aes_keys` | `None` | Set for `tee_cipher` encryption. |
| `topology` | `Topology::default()` | Runtime shard count + workers-per-shard (Phase 2.3a). |

---

## GPU acceleration

CUDA SHA256 kernels handle the upper-tree Merkle sync, twig sync, and active-bits sync. Path selection happens at runtime:

| Path | When | Notes |
|---|---|---|
| `sync_upper_nodes_gpu` (per-level) | **default** | Sends a small `n_list` per level to GPU; produces byte-identical roots to CPU. ~4.5× faster than the resident path at 40M cuda. |
| `sync_upper_nodes_gpu_resident` | `SKIPPY_USE_GPU_RESIDENT=1` | Keeps the entire upper tree in GPU-resident memory via `flash-map`. After the [`NULL_TWIG.twig_root` populate fix](#performance-history) it produces parity-equal roots, but the every-block bulk-populate of the active twig set dominates and slows the path down. Kept for benchmarking only. |
| CPU SHA256 (SHA-NI) | small batches | `auto_batch_node_hash` falls back to CPU when n < 256 — Zen 3's hardware SHA-NI ties Blake3 at 65B inputs, so the round-trip cost beats the kernel launch at small sizes. |

Kernel variants:

| kernel | layout | use |
|---|---|---|
| `sha256_node_hash` | AoS (65B stride) | Default fixed-size node hashing |
| `sha256_node_hash_soa` | SoA (32B stride) | Better coalesced reads at high batch sizes |
| `sha256_node_hash_warp_coop` | AoS + warp shuffles | 8 threads/hash on high-SM GPUs |
| `sha256_variable_hash` | Variable-length | Entry-content hashing (50–300 B inputs) |

Direct API:

```rust
use skippydb::gpu::{GpuHasher, NodeHashJob};

let gpu = GpuHasher::new(200_000)?;

let jobs = vec![
    NodeHashJob { level: 0, left: [0x11; 32], right: [0xAB; 32] },
    // ... batch as many as you like
];
let parents = gpu.batch_node_hash(&jobs); // auto-dispatch CPU / AoS / SoA

// Or pin to SoA explicitly:
let levels = vec![0u8; 10_000];
let lefts = vec![[0x11u8; 32]; 10_000];
let rights = vec![[0xABu8; 32]; 10_000];
let parents = gpu.batch_node_hash_soa(&levels, &lefts, &rights);
```

Full guide: [`docs/gpu-acceleration.md`](docs/gpu-acceleration.md).

---

## Env toggles

All env vars read once at startup via `OnceLock`; setting them mid-run does nothing. Production deployments should leave these unset.

| var | default | what |
|---|---|---|
| `SKIPPY_TRACE` | unset | Emit `TRACE shard=S height=H phase=P us=US` to stderr per flusher phase. Adds noticeable wall-clock from `eprintln!`; use the trace to read phase **shares**, not absolute times. |
| `SKIPPY_USE_GPU_RESIDENT` | unset | Opt into the legacy GPU-resident upper-tree sync path. Off by default since 2026-04-26 — see [Performance history](#performance-history). |
| `SKIPPY_WORKERS_PER_SHARD` | `1` | Runtime `Topology.workers_per_shard`. Values >1 enable the experimental Phase 2.4-v2 parallel indexer-read path; `W=1` is byte-identical to the prior production code. |
| `SKIPPY_ROOT_DUMP` | unset | Stderr-dump every committed `(shard, height) → root_hash` for debugging parity issues. |
| `SKIPPY_NO_GPU_RESIDENT` | unset | Legacy gate (pre-default-flip). Equivalent to leaving `SKIPPY_USE_GPU_RESIDENT` unset; kept so older scripts still work. |

---

## On-disk format

The `MetaInfo` plaintext is prefixed with the 8-byte magic
`META_MAGIC_V2 = b"SKIPV2\x00\x00"`. Pre-V2 databases (anything from
QMDB or pre-Phase-2 SkippyDB) **cannot be loaded** — `MetaDB::reload_from_file`
panics loudly with a pointer to TODO.md. `MetaDB::with_dir_checked`
returns `MetaDbError::UnsupportedFormat` if a caller wants to surface the
error cleanly instead.

To bump the format again: change the last two bytes of `META_MAGIC_V2`,
add a dispatch in `parse_metainfo_v2`, leave V2 green for a deprecation
window. See `qmdb/src/metadb.rs`.

The entry-file, twig-file, and indexer formats are unchanged from QMDB
upstream.

---

## Performance history

> *"I have to explain everything around here. Honestly, you'd think after a
> billion years a species would learn to read a graph."* — Skippy

Every optimization and every regression on the way to the current baseline
is tracked in `TODO.md` and `bench/results/`. Highlights:

| phase | branch / commit | 40M cuda elapsed | block_pop | updates/s | note |
|---|---|---:|---:|---:|---|
| `main` baseline | upstream | 214.4 s | 2.27/s | 206K | Original QMDB-derived. |
| Phase 1 (async commit + countdown) | merged → moonshot | ~199 s | 2.43/s | 224K | +6.9% block_pop, +9.2% updates. |
| Phase 2 (`MetaInfoV2` + Box<[T]> per-shard) | merged → moonshot | 178.5 s | 2.81/s | 247K | +20% across the board, reproducible. |
| Phase 2.4-v2 W=1 fast-path | `9d077de` | (parity-only) | — | — | W>1 has a known channel-cascade panic, opt-in only. |
| Phase 0 second-capture | `55b9e1c` | — | — | — | Sub-trace `entry_append`; surfaced that `sync_upper_nodes` was 62% of block time, not `entry_append`. |
| `NULL_TWIG.twig_root` parity fix | `eb4659e` | — | — | — | Resident-path GPU `bulk_get_device` was returning uninitialized memory for missing twig positions. Fixed; locked behind unit-level parity test. |
| **Per-level GPU is the default** | `1916aaa` | **49.5 s** | **11.10/s** | **1.35M** | ~4.3× over `main`, ~3.6× over Phase 2. |

The full A/B numbers (resident vs per-level, 5M and 40M, pre-fix and
post-fix) are in `bench/results/`.

### Failed probes (kept for the record)

> *"This was Joe's idea. I told him it would not work. He did not listen. It
> did not work."* — Skippy, on the failed probes below

- **Phase 4 — Blake3 swap (CPU only)**: -13.9% at 40M. Zen 3 SHA-NI ties Blake3 at 65B inputs; force-CPU disabled the SoA GPU kernel that was carrying upper-tree sync.
- **Phase 4 — Blake3 swap (CUDA kernel)**: -15.8% at 40M. Kernel is parity-verified (`qmdb/tests/blake3_kernel_parity.rs`), but had to drop the fused `batch_active_bits_fused` SHA256 kernel to keep tree algorithmically consistent. Net regression. Kernel is parked at `qmdb/src/gpu/blake3_kernel.cu` on the `rewrite/phase4-blake3-cuda` branch.
- **Phase 3.1 — `sync_all` → `sync_data`**: 5M ran clean, 40M crashed. Compactor reads via O_DIRECT immediately after flush; `fdatasync` skips metadata updates that O_DIRECT read-after-write needs. Reverted; see TODO.md for the full crash log.
- **Phase 2.4-workerpool (Commit 1)**: -12.7% at W=1 alone. The `reserve(size) + fill_at(pos, ...)` API is correct but the split adds enough branch overhead at W=1 to cost ~13% by itself. Reverted; led directly to the Phase 2.4-v2 indexer-only design.

---

## Repo layout

```
SkippyDB/
├── qmdb/                          # core library (crate name: skippydb)
│   ├── src/
│   │   ├── lib.rs                 # AdsCore, AdsWrap, ADS trait
│   │   ├── config.rs              # Configuration
│   │   ├── topology.rs            # Runtime Topology { shard_count, workers_per_shard }
│   │   ├── def.rs                 # Constants (TWIG_SHIFT, FIRST_LEVEL_ABOVE_TWIG, ...)
│   │   ├── entryfile/             # Entry, EntryBz, EntryBuffer, EntryFile
│   │   ├── merkletree/
│   │   │   ├── tree.rs            # Tree, UpperTree, NodePos, sync_upper_nodes_gpu*
│   │   │   ├── twig.rs            # Twig, ActiveBits, TwigMT, sync_l1/l2/l3/top
│   │   │   ├── twigfile.rs
│   │   │   ├── proof.rs
│   │   │   ├── recover.rs
│   │   │   └── check.rs           # check_twig, check_upper_nodes, check_hash_consistency
│   │   ├── indexer/               # in-mem B-tree + SSD-backed hybrid
│   │   ├── gpu/                   # CUDA kernels (feature = "cuda")
│   │   │   ├── gpu_hasher.rs      # GpuHasher, MultiGpuHasher, dispatch threshold
│   │   │   ├── gpu_node_store.rs  # GPU-resident node store (flash-map)
│   │   │   └── *.cu               # SHA256 / Blake3 kernels
│   │   ├── seqads/                # Sequential ADS for stateless validation
│   │   ├── stateless/
│   │   ├── tasks/                 # Task, TaskHub, BlockPairTaskHub
│   │   ├── flusher.rs             # 8-job flusher pipeline + Phase 0 trace + env gates
│   │   ├── updater.rs             # Updater + Phase 2.4-v2 worker pool
│   │   ├── compactor.rs
│   │   └── metadb.rs              # Custom 2-file ping-pong, MetaInfoV2 envelope
│   └── tests/
│       ├── sync_upper_nodes_parity.rs   # resident-vs-per-level Merkle root parity
│       ├── active_bits_sync_parity.rs   # CPU-vs-GPU phase1+phase2 active-bits parity
│       ├── twig_sync_parity.rs          # CPU-vs-GPU twig MT sync parity
│       ├── cpu_gpu_sha256_parity.rs     # CPU-vs-GPU SoA SHA256 parity
│       ├── metadb_topology_roundtrip.rs # parametric shard-count roundtrip
│       └── blake3_kernel_parity.rs      # parked Blake3 CUDA kernel parity
├── hpfile/                        # Head-prunable file library
├── flash-map/                     # GPU-resident hash map (vendored)
├── bench/                         # speed binary + bench/results/
└── docs/                          # architecture / gpu / design notes
```

---

## Benchmarking

```bash
# Smoke / dev (5M is the smallest --entry-count that satisfies the bench's
# blocks_for_db_population >= tps_blocks precondition with release defaults).
cargo run --release --features cuda --bin speed -- --entry-count 5000000

# Production-ish workload on a single host.
cargo run --release --features cuda --bin speed -- --entry-count 40000000

# Multi-billion. Adjust ops_per_block / tps_blocks for your hardware.
cargo run --release --features cuda --bin speed -- \
    --db-dir /mnt/nvme/QMDB \
    --entry-count 7000000000 \
    --ops-per-block 1000000 \
    --hover-recreate-block 100 \
    --hover-write-block 100 \
    --hover-interval 1000 \
    --tps-blocks 500
```

Bench harness docs: [`bench/README.md`](bench/README.md).

For Criterion-style hash microbenches:

```bash
cargo bench --bench hash_benchmarks                  # CPU only
cargo bench --bench hash_benchmarks --features cuda  # + GPU
```

---

## Testing

```bash
cargo test --release --features cuda                 # full suite (4 parity gates included)
cargo test --release --features cuda --test sync_upper_nodes_parity -- --nocapture
cargo test --release --features cuda --lib gpu::     # GPU subset
```

The four parity tests
([`sync_upper_nodes`, `cpu_gpu_sha256`, `twig_sync`,
`active_bits_sync`](qmdb/tests/)) are gates: any future change to the
upper-tree-sync, twig-sync, or active-bits-sync paths has to keep them
green or the bug it would have introduced shows up immediately on the
build.

Two tests are intentionally `#[ignore]`'d
(`test_full_pipeline_large_cpu_vs_gpu`,
`test_twig_eviction_cpu_vs_gpu`) — they need the full
`flush_files`/`flush_files_gpu` orchestration to migrate `new_twig_map`
into `active_twig_shards`, which the partial-pipeline harness can't do.
The active-bits parity they were trying to verify is covered properly
by `qmdb/tests/active_bits_sync_parity.rs`.

---

## Feature flags

| flag | default | what |
|---|---|---|
| `tikv-jemallocator` | yes | jemalloc allocator |
| `cuda` | no | CUDA SHA256 kernels |
| `directio` | no | Linux `io_uring` direct I/O |
| `tee_cipher` | no | AES-256-GCM encryption for entries + index |
| `use_hybridindexer` | no | SSD-backed hybrid indexer |
| `hpfile_all_in_mem` | no | Keep HPFile data entirely in memory (testing) |
| `slow_hashing` | no | Disable parallel hashing (debug) |
| `in_sp1` | no | Build for the SP1 zkVM |

---

## Contributing

```bash
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --all -- --check
cargo test --release --features cuda
```

PR checklist:

- New code passes the parity gates above.
- New on-disk format / consensus-affecting changes bump `META_MAGIC_V2`.
- New env toggles documented in [Env toggles](#env-toggles).
- Bench numbers backed by a JSON file in `bench/results/`.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the longer style + review notes.

---

## License

Dual-licensed under [MIT](LICENSE-MIT) and [Apache-2.0](LICENSE-APACHE).

> *"You may use my magnificent database under either license. I would have
> preferred a third option named the Skippy License, but the monkeys
> wouldn't let me."*

## Citation

If you use SkippyDB in research, please cite the upstream QMDB paper:

```bibtex
@misc{qmdb2025,
  title  = {QMDB: Quick Merkle Database},
  author = {Layr Labs},
  year   = {2025},
  eprint = {2501.05262},
  url    = {https://arxiv.org/pdf/2501.05262},
}
```
