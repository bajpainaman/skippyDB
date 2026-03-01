# QMDB GPU Acceleration & Performance Optimization Plan

> Comprehensive analysis of QMDB's hashing, parallelism, memory layout, and GPU integration —
> with prioritized, actionable optimization proposals backed by source-level evidence.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Current GPU Integration](#3-current-gpu-integration)
4. [CPU-Side Hashing Analysis](#4-cpu-side-hashing-analysis)
5. [Parallelism & Threading Bottlenecks](#5-parallelism--threading-bottlenecks)
6. [Atomic Ordering Audit](#6-atomic-ordering-audit)
7. [Memory Layout & Cache Efficiency](#7-memory-layout--cache-efficiency)
8. [CUDA Kernel Analysis](#8-cuda-kernel-analysis)
9. [GPU Transfer Pipeline](#9-gpu-transfer-pipeline)
10. [Feature-Gated Code Duplication](#10-feature-gated-code-duplication)
11. [Optimization Proposals (Prioritized)](#11-optimization-proposals-prioritized)
12. [Risk Assessment & Migration Strategy](#12-risk-assessment--migration-strategy)
13. [Appendix: File Reference](#13-appendix-file-reference)

---

## 1. Executive Summary

QMDB is a high-performance key-value store with a Merkle tree commitment layer. Its flush pipeline — the hot path — hashes tens of thousands of SHA256 digests per block to maintain a cryptographic proof of state. The codebase already features:

- **CUDA-accelerated batch hashing** (`cuda` feature) for fixed 65-byte node hashes and variable-length entry hashes
- **Rayon-based parallelism** for shard-level Merkle tree synchronization
- **SHA-NI hardware acceleration** via the `sha2` crate with `sha2-asm` feature
- **16-shard data architecture** with 4-way node/twig sub-sharding

However, several high-impact optimization opportunities remain untapped. The three highest-ROI changes are:

| # | Change | Estimated Impact | Complexity |
|---|--------|-----------------|------------|
| 1 | **Unlock rayon parallelism** (currently gated behind `slow_hashing`) | 3-4x CPU merkle throughput | Trivial (~4 lines) |
| 2 | **Pre-partition `n_list` by shard** before dispatch | Eliminate 75% wasted iteration | Low (~30 lines) |
| 3 | **Relax 12 SeqCst atomics** on the entry-append hot path | Measurable on 16+ cores | Low (~12 lines) |

Additional medium- and long-term improvements are detailed in [Section 11](#11-optimization-proposals-prioritized).

---

## 2. Architecture Overview

### 2.1 Data Flow

```
 Entries ──┬──▶ EntryBuffer (append, atomic) ──▶ EntryFile (disk)
           │
           └──▶ MerkleTree Shard (in-memory)
                  │
                  ├── TwigMT[4096]           ← 2048-leaf subtrees, 131 KB each
                  ├── Twig.active_bits_mtl*   ← 4-level hash rollup per twig
                  └── UpperTree.nodes         ← 64-level × 4-shard HashMaps
                        │
                        └──▶ Root Hash (per shard)
```

### 2.2 Sharding Model

| Parameter | Value | Source |
|-----------|-------|--------|
| `SHARD_COUNT` | 16 | `def.rs:14` |
| `NODE_SHARD_COUNT` | 4 | `def.rs:43` |
| `TWIG_SHARD_COUNT` | 4 | `def.rs:42` |
| `BYTES_CACHE_SHARD_COUNT` | 32 | `def.rs:1` |
| Effective cache shards | 512 (16 × 32) | `entrycache.rs:36-39` |

### 2.3 Merkle Tree Geometry

| Constant | Value | Source |
|----------|-------|--------|
| `MAX_TREE_LEVEL` | 64 | `def.rs:44` |
| `FIRST_LEVEL_ABOVE_TWIG` | 13 | `def.rs:36` |
| `TWIG_ROOT_LEVEL` | 12 | `def.rs:37` |
| `LEAF_COUNT_IN_TWIG` | 2048 | `def.rs:48` |
| TwigMT size | 4096 × 32B = 131,072B | `twig.rs:15` |
| Twig struct | 288B (4×32 + 2×32 + 32 + 32 + 32) | `twig.rs:17-24` |

### 2.4 Key Configuration

| Parameter | Value | Source |
|-----------|-------|--------|
| Write buffer | 8 MB | `config.rs:31` |
| File segment | 1 GB | `config.rs:32` |
| Task channel | 200,000 | `config.rs:5` |
| Prefetcher threads | 512 | `config.rs:6` |
| io_uring count | 32 | `config.rs:8` |
| io_uring depth | 1024 | `config.rs:7` |
| GPU max batch | 200,000 | `lib.rs:191` |

---

## 3. Current GPU Integration

### 3.1 GpuHasher Architecture

**File:** `gpu/gpu_hasher.rs`

```
GpuHasher
├── device: Arc<CudaDevice>           ← GPU device 0
├── node_hash_fn: CudaFunction        ← sha256_node_hash kernel
├── var_hash_fn: CudaFunction         ← sha256_variable_hash kernel
└── max_batch_size: usize             ← 200,000
```

- **Initialization** (lines 34-59): NVRTC runtime compilation of embedded CUDA source, PTX loading, kernel function extraction.
- **Thread safety**: `Arc<GpuHasher>` shared across all shard threads. CUDA serializes device operations internally.

### 3.2 Batch Node Hashing

**Method:** `batch_node_hash()` (lines 64-125)

```
Input:  Vec<NodeHashJob>  ─┐
        each 65 bytes:     │  [level(1B) | left(32B) | right(32B)]
                           │
   ┌── Flatten to Vec<u8>  │  n × 65 bytes        (line 77-83)
   ├── htod_copy           │  Host → Device        (line 88)
   ├── alloc_zeros         │  n × 32 output        (line 91)
   ├── Kernel launch       │  grid=(n+255)/256     (line 96-108)
   └── dtoh_sync_copy      │  Device → Host        (line 114)
                           │
Output: Vec<[u8; 32]>    ──┘
```

**Grid configuration:** 256 threads/block, `ceil(n/256)` blocks, 0 shared memory.

### 3.3 Variable-Length Hashing

**Method:** `batch_hash_variable()` (lines 142-216)

Three separate uploads: flat data buffer, offset array (`u32`), length array (`u32`). Same grid/block configuration. Used for entry payloads (typically 50-300 bytes).

### 3.4 GPU Call Sites in Flush Pipeline

| Call Site | File:Line | Function | When Called |
|-----------|-----------|----------|-------------|
| Youngest twig merkle sync | `tree.rs:1183` | `sync_mt_for_youngest_twig_gpu()` | Every flush |
| Active bits phase 1 (L1) | `tree.rs:1265` | `gpu.batch_node_hash(&jobs)` | Every flush |
| Active bits phase 2a (L2) | `tree.rs:506` | `gpu.batch_node_hash(&jobs)` | Every flush |
| Active bits phase 2b (L3) | `tree.rs:539` | `gpu.batch_node_hash(&l3_jobs)` | Every flush |
| Active bits phase 2c (top) | `tree.rs:560` | `gpu.batch_node_hash(&top_jobs)` | Every flush |
| Upper tree level sync | `tree.rs:418` | `gpu.batch_node_hash(&jobs)` | Per level (13-64) |
| Multi-twig batch sync | `twig.rs:170` | `gpu.batch_node_hash(&jobs)` | Per twig level |

### 3.5 NodeHashJob Memory Layout

```rust
#[repr(C)]            // gpu_hasher.rs:5
pub struct NodeHashJob {
    pub level: u8,    // 1 byte    offset 0
    pub left: [u8; 32],  // 32 bytes  offset 1
    pub right: [u8; 32], // 32 bytes  offset 33
}                     // Total: 65 bytes, no padding
```

The `repr(C)` annotation ensures deterministic layout for GPU transfer. `unsafe impl DeviceRepr` (line 14) permits cudarc transfers.

---

## 4. CPU-Side Hashing Analysis

### 4.1 Hash Function Inventory

**File:** `utils/hasher.rs`

| Function | Lines | Input Size | Usage |
|----------|-------|-----------|-------|
| `hash(a)` | 7-11 | Variable | Entry payload hashing |
| `hash1(level, a)` | 13-18 | 1 + variable | Single-child node |
| `hash2(level, a, b)` | 20-26 | 1 + 32 + 32 = 65B | **Primary hot path** (merkle nodes) |
| `hash2x(level, a, b, swap)` | 28-34 | 65B | Conditional left/right swap |
| `node_hash_inplace(level, target, a, b)` | 36-47 | 65B | In-place output variant of hash2 |

All functions instantiate a new `Sha256::new()` per call — no state reuse, no batching.

### 4.2 SHA2 Crate Configuration

**File:** `Cargo.toml:18`
```toml
sha2 = { version = "0.10.8", features = ["asm", "asm-aarch64", "sha2-asm"] }
```

- **SHA-NI**: Enabled via `sha2-asm` → inline assembly on x86_64 with SHA extensions
- **aarch64**: Enabled via `asm-aarch64` → hardware SHA256 on ARM
- **CPU detection**: Via `cpufeatures` crate (transitive dependency)

**Lock file confirms:** sha2 v0.10.9, sha2-asm v0.6.4

### 4.3 CPU Hash Call Sites (Hot Paths)

#### Twig Merkle Tree Sync (`twig.rs`)

| Function | Lines | Hash Calls | Notes |
|----------|-------|-----------|-------|
| `sync_mtree()` | 88-111 | `node_hash_inplace()` per pair per level | **Main CPU hot path** |
| `sync_l1()` | 205-232 | 4× `node_hash_inplace()` | Active bits level 1 |
| `sync_l2()` | 237-248 | 2× `node_hash_inplace()` | Active bits level 2 |
| `sync_l3()` | 254-259 | 1× `node_hash_inplace()` | Active bits level 3 |
| `sync_top()` | 263-268 | 1× `node_hash_inplace()` | Twig root |

#### Upper Tree Node Sync (`tree.rs`)

| Function | Lines | Hash Calls | Notes |
|----------|-------|-----------|-------|
| `do_sync_job()` | 580-650 | `node_hash_inplace()` per node | Per-level shard dispatch |
| `sync_nodes_by_level()` | 196-249 | Delegates to `do_sync_job()` via rayon | 4 shards |

#### Entry Hashing (`entry.rs`)

| Function | Lines | Hash Calls | Notes |
|----------|-------|-----------|-------|
| `EntryBz::hash()` | 165-167 | `hasher::hash()` | Variable-length payload |
| `EntryBz::key_hash()` | 176-187 | `hasher::hash()` or 2-byte shortcut | Deleted entries skip SHA256 |

#### Twig File Recovery (`twigfile.rs`)

| Function | Lines | Hash Calls | Notes |
|----------|-------|-----------|-------|
| `recover_cone()` | ~141-160 | 3-5× `hash2()` | Sequential, during recovery only |

### 4.4 Instruction-Level Parallelism (ILP) Gap

**Current pattern** in `sync_mtree()` (lines 94-110):

```rust
while j <= end_round && j + 1 < base {
    let i = (base + j) as usize;
    hasher::node_hash_inplace(level, &mut node[..], &mtree[i], &mtree[i + 1]);  // BLOCKING
    mtree[i / 2].copy_from_slice(&node[..]);  // output dependency
    j += 2;
}
```

Each SHA256 hash completes before the next one starts. On a CPU with SHA-NI, the compression function can overlap with independent work, but the current serial loop prevents this. SHA-NI throughput on Intel is ~4 cycles/byte (single stream) vs ~1.5 cycles/byte with 2-4 interleaved streams.

---

## 5. Parallelism & Threading Bottlenecks

### 5.1 The `slow_hashing` Gate Problem

**This is the single highest-impact finding.**

QMDB uses `rayon::scope()` to set up shard-parallel Merkle tree hashing. However, the actual `s.spawn()` call is gated behind `cfg!(feature = "slow_hashing")`. When `slow_hashing` is disabled (the default and intended production configuration), **all shards run sequentially on the main thread**:

```rust
// tree.rs:226-237
rayon::scope(|s| {
    for (shard_id, nodes) in written_nodes.iter_mut().enumerate() {
        if cfg!(feature = "slow_hashing") {
            s.spawn(move |_| do_sync_job(...));  // ← ONLY runs in parallel with slow_hashing
        } else {
            do_sync_job(...);  // ← DEFAULT: sequential execution
        }
    }
});
```

This pattern repeats at **4 locations**:

| Site | File:Lines | Affected Operation |
|------|------------|-------------------|
| `sync_nodes_by_level()` | `tree.rs:226-237` | Upper tree node hashing |
| `sync_mt_for_active_bits_phase2()` (L2) | `tree.rs:288-313` | Active bits L2 sync |
| `sync_mt_for_active_bits_phase2()` (L3+top) | `tree.rs:324-349` | Active bits L3 + twig root |
| `sync_mt_for_active_bits_phase1()` | `tree.rs:913-947` | Active bits L1 sync |

**Impact:** On the CPU path (non-GPU), enabling rayon parallelism across 4 node shards would yield a **~3-4x throughput improvement** for merkle tree synchronization with zero algorithmic changes.

### 5.2 Shard Iteration Waste

**File:** `tree.rs:580-591` (`do_sync_job()`)

```rust
fn do_sync_job(upper_tree: &UpperTree, nodes: &mut HashMap<...>, level: i64, shard_id: usize, n_list: &[u64]) {
    for &i in n_list {
        if i as usize % NODE_SHARD_COUNT != shard_id {
            continue;  // ← 75% of iterations are wasted (skipped)
        }
        // ... actual work
    }
}
```

Each of the 4 shards iterates the **entire** `n_list` and skips ~75% of entries. This wastes CPU time on branch mispredictions and cache pollution.

**Same pattern in:**
- `tree.rs:920-925` (active bits L1 phase)
- `tree.rs:294-298` (active bits L2 phase)
- `tree.rs:330-333` (active bits L3+top phase)

### 5.3 GPU Path: Same Pattern, Same Problem

The GPU equivalents (`sync_nodes_by_level_gpu`, `sync_mt_for_active_bits_phase2_gpu`) build job vectors by iterating `n_list` linearly. They don't share the shard-skip problem (they build flat job arrays), but they do iterate `n_list` once per GPU call level.

### 5.4 Flusher Thread Architecture

```
Flusher::flush()  or  Flusher::flush_gpu()
    │
    └── thread::scope(|s|)
        ├── shard[0].flush(...)  ──┐
        ├── shard[1].flush(...)    │  16 shard threads
        ├── ...                    │
        └── shard[15].flush(...) ──┘
                │
                ├── flush_bar.wait()      ← all shards synchronize
                │
                ├── [shard 0 waits for metadb_bar]
                │   [shards 1-15 write metadata then signal metadb_bar]
                │
                └── metadb_bar.wait()     ← final synchronization
```

**Key observations:**
- Each flush creates **new** barriers (`flusher.rs:73, 104`) — barrier objects are not reused.
- Shard 0 is the **serialization point** for MetaDB commits.
- In `slow_hashing` mode, the upper tree computation is spawned to a **detached `thread::spawn()`** (`flusher.rs:246-306`), which runs concurrently with the next flush iteration. The upper tree is passed back via a `sync_channel(2)`.

### 5.5 Thread Spawn Summary

| Component | Thread Count | Mechanism | Source |
|-----------|-------------|-----------|--------|
| Flusher shards | 16 | `thread::scope()` | `flusher.rs:74-86` |
| Entry/twig file flush | 2 | `thread::scope()` | `tree.rs:876-881` |
| Upper tree (slow_hashing) | 1 (detached) | `thread::spawn()` | `flusher.rs:246` |
| Prefetcher | 512 | (config) | `config.rs:6` |
| io_uring | 32 | (config) | `config.rs:8` |

---

## 6. Atomic Ordering Audit

### 6.1 EntryBuffer Atomics

**File:** `entryfile/entrybuffer.rs`

The `EntryBuffer` struct uses two `AtomicI64` fields (`start` and `end`) to manage a ring buffer. **All 12 atomic operations use `Ordering::SeqCst`**, which is the strongest (and most expensive) ordering.

| Line | Operation | Current | Recommended | Rationale |
|------|-----------|---------|-------------|-----------|
| 104 | `self.end.load()` | SeqCst | **Acquire** | Reads end position to insert new buf_map entry |
| 131 | `self.end.load()` | SeqCst | **Acquire** | Reads file_pos for append offset calculation |
| 137 | `self.end.fetch_add(size)` | SeqCst | **AcqRel** | Atomically reserves space; must be visible to concurrent readers |
| 146 | `self.end.fetch_add(size)` | SeqCst | **AcqRel** | Same as above (overflow/split-buffer case) |
| 155 | `self.start.load()` | SeqCst | **Acquire** | Check lower bound during buffer removal |
| 157 | `self.start.store(new_start)` | SeqCst | **Release** | Advance start pointer after removing buffer |
| 184 | `self.start.load()` | SeqCst | **Acquire** | Bounds check on entry lookup |
| 185 | `self.end.load()` | SeqCst | **Acquire** | Bounds check on entry lookup |
| 214 | `self.start.load()` | SeqCst | **Acquire** | Re-check for concurrent start advancement |

**SeqCst operations in tests (lines 383, 390):** These can remain SeqCst since they're not performance-critical.

### 6.2 Why SeqCst is Excessive Here

`SeqCst` guarantees a single total order across all threads for all `SeqCst` operations. This requires a full memory fence (MFENCE on x86_64) after every store.

The `EntryBuffer` pattern is a classic producer-consumer:
- **Producer** (appender): `fetch_add` on `end` to reserve space, then write data
- **Consumer** (flusher): `load` on `end` to read available data, `store` on `start` to release

This only requires **Acquire/Release** semantics:
- `fetch_add` on `end` → `AcqRel` (producer acquires current position, releases new position)
- `load` on `start`/`end` → `Acquire` (consumer reads latest position)
- `store` on `start` → `Release` (consumer publishes new lower bound)

### 6.3 Expected Impact

On x86_64, `SeqCst` loads compile to plain `MOV` instructions (same as `Acquire`), but `SeqCst` stores compile to `MOV` + `MFENCE` or `XCHG`. The `fetch_add` compiles to `LOCK XADD` regardless of ordering.

**Net effect:** The `store` at line 157 and any non-`fetch_add` store paths would lose their MFENCE. On systems with 16+ cores, this reduces cache-line bouncing on the `start` variable. Estimated improvement: **5-15% on the entry-append hot path** on high-core-count systems.

---

## 7. Memory Layout & Cache Efficiency

### 7.1 Upper Tree Node Storage

```rust
// tree.rs:88
pub nodes: Vec<Vec<HashMap<NodePos, [u8; 32]>>>
//          ^^^  ^^^  ^^^^^^^^^^^^^^^^^^^^^^^
//        64 levels  4 shards  hash per node
```

**Concern:** `HashMap` stores entries in a hash table with open addressing. Iterating over entries produces **non-sequential memory access patterns**. When `do_sync_job()` reads child nodes:

```rust
// tree.rs:615-616
let sl = node_pos_l.nth() as usize % NODE_SHARD_COUNT;  // could be different shard
let sr = node_pos_r.nth() as usize % NODE_SHARD_COUNT;  // from the write shard
let node_l = child_nodes[sl].get(&node_pos_l);
let node_r = child_nodes[sr].get(&node_pos_r);
```

The left and right child may reside in **different HashMap shards**, causing 2 random cache-line loads per hash computation.

### 7.2 Twig Active Bits Structure

```rust
// twig.rs:17-24
pub struct Twig {
    pub active_bits_mtl1: [Hash32; 4],   // 128B ← sync_l1 writes here
    pub active_bits_mtl2: [Hash32; 2],   //  64B ← sync_l2 writes here
    pub active_bits_mtl3: Hash32,        //  32B ← sync_l3 writes here
    pub left_root: Hash32,               //  32B ← from mtree[1]
    pub twig_root: Hash32,               //  32B ← sync_top writes here
}                                        // Total: 288B
```

288B fits in **5 cache lines** (64B each). The sync_l1 → sync_l2 → sync_l3 → sync_top sequence reads/writes within the same struct, so cache locality is good once the Twig is loaded.

**However:** Twigs are stored in `HashMap<u64, Box<Twig>>`. The `Box` adds an indirection, and HashMap iteration is non-sequential. Twigs accessed in `n_list` order may be scattered in memory.

### 7.3 TwigMT Array

```rust
type TwigMT = [Hash32];  // twig.rs:15 — 4096 × 32B = 131,072B per twig
```

At 128 KB per TwigMT, a single twig's merkle tree spans ~2048 cache lines. The bottom-up `sync_mtree()` accesses pairs at stride 2, then stride 4, then stride 8, etc. The first few levels (stride 2-8) have good spatial locality. Higher levels (stride 1024+) touch only a few cache lines.

### 7.4 GPU Transfer Memory

The `batch_node_hash()` method allocates a flat `Vec<u8>` of `n × 65` bytes, copies `NodeHashJob` structs into it, and uploads via `htod_copy`. This is a single contiguous allocation — good for DMA transfer but requires the CPU-side copy loop:

```rust
// gpu_hasher.rs:78-83
for (i, job) in jobs.iter().enumerate() {
    let off = i * 65;
    flat_input[off] = job.level;
    flat_input[off + 1..off + 33].copy_from_slice(&job.left);
    flat_input[off + 33..off + 65].copy_from_slice(&job.right);
}
```

For 200,000 jobs, this copies 13 MB of data. The copy itself is sequential and cache-friendly, but it doubles memory usage (jobs Vec + flat buffer).

---

## 8. CUDA Kernel Analysis

### 8.1 Fixed 65-Byte Node Hash Kernel

**File:** `gpu/sha256_kernel.cu:114-154`

```
Thread i reads:  jobs[i×65 .. i×65+65]   (65 bytes from global memory)
Thread i writes: out[i×32 .. i×32+32]    (32 bytes to global memory)
```

**Compression blocks:** Exactly 2 per hash (64B + 1B + padding + length).

**Optimizations applied:**
- `#pragma unroll` on message schedule (lines 87, 97)
- `#pragma unroll` on state initialization (line 130)
- `#pragma unroll` on output store (line 150)
- `__forceinline__` on all utility functions (lines 34-60)
- `__constant__` memory for K[64] and H_INIT[8] (lines 9-32)
- `__restrict__` pointers (line 115-116)

**Not applied:**
- **No shared memory usage** (`shared_mem_bytes: 0` at `gpu_hasher.rs:101`)
- **No coalesced reads** — 65-byte stride is not aligned to 32/128-byte boundaries
- **No warp-level cooperation** — each thread independently hashes one job

### 8.2 Variable-Length Hash Kernel

**File:** `gpu/sha256_kernel.cu:163-238`

Handles arbitrary-length inputs with proper SHA256 padding:
- Full 64-byte blocks processed in a loop (lines 182-185)
- Final block with `remaining < 56` (single padding block, lines 199-212)
- Final block with `remaining >= 56` (two padding blocks, lines 213-231)

**Performance note:** The loop on line 183 (`for (uint32_t b = 0; b < full_blocks; b++)`) is not unrolled. For typical entry sizes of 50-300 bytes, this means 0-4 loop iterations — branch prediction should handle this well.

### 8.3 Kernel Launch Configuration

```
Block size:  256 threads (BLOCK_SIZE constant, gpu_hasher.rs:16)
Grid size:   ceil(n / 256) blocks
Shared mem:  0 bytes
```

For a typical batch of 10,000-200,000 jobs:
- 10K jobs → 40 blocks → low GPU occupancy
- 200K jobs → 782 blocks → good occupancy on most GPUs

### 8.4 GPU Kernel Optimization Opportunities

1. **Memory coalescing:** The 65-byte job stride means adjacent threads read from addresses 65 apart. A Structure-of-Arrays (SoA) layout — separate arrays for levels, lefts, and rights — would enable 32-byte coalesced reads for left/right arrays.

2. **Shared memory staging:** Load jobs into shared memory first, then process. This would amortize global memory latency for the 2-block compression.

3. **Warp-level SHA256:** Distribute the 64 rounds across 8 threads per hash (8 × 8 state words). This is a well-known optimization for batch SHA256 on GPUs but adds significant complexity.

---

## 9. GPU Transfer Pipeline

### 9.1 Current Pattern (Synchronous)

```
CPU: [Build jobs] ──▶ [Flatten] ──▶ [htod_copy] ──▶ [launch] ──▶ [dtoh_sync] ──▶ [Scatter results]
GPU:                                               ──▶ [compute] ──▶
     ^^^^^^^^^^^^^^^^                                                 ^^^^^^^^^^^^^^^^^^
     CPU idle during compute                                          GPU idle during scatter
```

**Blocking call:** `dtoh_sync_copy` at `gpu_hasher.rs:114` synchronously waits for the kernel to complete and transfers results.

### 9.2 Pipeline Opportunity

The flush pipeline calls `batch_node_hash()` multiple times per flush:
1. Youngest twig sync
2. Active bits L1
3. Active bits L2
4. Active bits L3
5. Active bits top
6. Upper tree (per level, 13 to max_level)

Each call blocks on `dtoh_sync_copy`. With async CUDA streams, the pattern could be:

```
                GPU Stream 1           GPU Stream 2
CPU:  Build L1 ──▶ Launch L1 ──▶ Build L2 ──▶ Launch L2 ──▶ ...
GPU:                [compute L1] ──▶          [compute L2] ──▶ ...
                               ↑ fetch results L1
```

**Limitation:** cudarc 0.12 exposes async launch (`LaunchAsync`) but `dtoh_sync_copy` is the only download method. Async download would require cudarc enhancements or raw CUDA API calls.

### 9.3 Transfer Overhead Estimate

For a flush with 50,000 node hash jobs:
- Upload: 50K × 65B = 3.17 MB
- Download: 50K × 32B = 1.56 MB
- Total: 4.73 MB per flush

At PCIe 3.0 x16 bandwidth (~12 GB/s): **~0.4 ms** transfer time.
At PCIe 4.0 x16 bandwidth (~25 GB/s): **~0.2 ms** transfer time.

The GPU compute time for 50K SHA256-65B hashes on a modern GPU: **~0.3-1.0 ms**.

**Conclusion:** Transfer overhead is comparable to compute time — pipelining transfers with computation would yield meaningful gains.

---

## 10. Feature-Gated Code Duplication

### 10.1 `slow_hashing` Duplication in Flusher

**File:** `flusher.rs`

The `flush()` method has two massive conditional blocks:

```rust
#[cfg(feature = "slow_hashing")]     // lines 202-307 (~105 lines)
{
    // Spawns detached thread for upper tree computation
    // Uses sync_channel for upper tree passback
    // Identical metadata write logic
}

#[cfg(not(feature = "slow_hashing"))] // lines 309-405 (~96 lines)
{
    // Runs upper tree computation inline
    // No channel overhead
    // Identical metadata write logic
}
```

**~200 lines of near-identical code** with only the threading strategy differing.

### 10.2 `cuda` Feature Code Paths

Three parallel implementations exist:
1. CPU path (`flush()` → `flush_files()` → `evict_twigs()` → `sync_upper_nodes()`)
2. GPU path (`flush_gpu()` → `flush_files_gpu()` → `evict_twigs_gpu()` → `sync_upper_nodes_gpu()`)
3. `slow_hashing` path (CPU + detached thread for upper tree)

### 10.3 Rayon Gates in `tree.rs`

4 rayon scopes with identical `if cfg!(feature = "slow_hashing") { s.spawn() } else { inline }` patterns:
- `tree.rs:232-236`
- `tree.rs:290-312`
- `tree.rs:326-348`
- `tree.rs:915-946`

---

## 11. Optimization Proposals (Prioritized)

### Tier 1: High Impact, Low Risk (Days)

#### P1: Unlock Rayon Parallelism

**Impact:** ~3-4x CPU merkle throughput
**Complexity:** ~4 line changes
**Risk:** Low (rayon scopes already set up correctly)
**Files:** `tree.rs:232-236, 290-312, 326-348, 913-946`

**Problem:** `s.spawn()` is only called when `slow_hashing` is enabled. The default (non-slow_hashing) path runs all 4 shard jobs sequentially within `rayon::scope()`.

**Fix:** Remove the `cfg!(feature = "slow_hashing")` guard and always call `s.spawn()`. The rayon scope already handles lifetime/borrow semantics correctly.

```rust
// Before (tree.rs:232-236):
if cfg!(feature = "slow_hashing") {
    s.spawn(move |_| do_sync_job(upper_tree, nodes, level, id, n_list));
} else {
    do_sync_job(upper_tree, nodes, level, id, n_list);
}

// After:
s.spawn(move |_| do_sync_job(upper_tree, nodes, level, id, n_list));
```

**Note:** On the GPU path, this has no effect (GPU functions don't use rayon). On the CPU path (non-CUDA builds or `slow_hashing` fallback), this immediately parallelizes across 4 node shards.

---

#### P2: Pre-Partition `n_list` by Shard

**Impact:** Eliminate 75% wasted iteration per shard
**Complexity:** ~30 lines
**Risk:** Low
**Files:** `tree.rs:580-591` (`do_sync_job`), and 3 similar patterns in `sync_mt_for_active_bits_phase2` and `sync_mt_for_active_bits_phase1`

**Problem:** Each shard iterates the full `n_list` and skips entries not belonging to it:

```rust
for &i in n_list {
    if i as usize % NODE_SHARD_COUNT != shard_id {
        continue;  // 75% of iterations wasted
    }
}
```

**Fix:** Pre-partition `n_list` into per-shard buckets before dispatching:

```rust
let mut shard_lists: [Vec<u64>; NODE_SHARD_COUNT] = Default::default();
for &i in &n_list {
    shard_lists[i as usize % NODE_SHARD_COUNT].push(i);
}
// Then pass shard_lists[shard_id] to each shard's do_sync_job()
```

---

#### P3: Relax Atomic Orderings

**Impact:** 5-15% on entry-append path (16+ cores)
**Complexity:** ~12 line changes
**Risk:** Low (Acquire/Release is sufficient for producer-consumer)
**Files:** `entryfile/entrybuffer.rs:104, 131, 137, 146, 155, 157, 184, 185, 214`

**Change:** Replace `SeqCst` with `Acquire` (loads), `Release` (stores), or `AcqRel` (fetch_add). See [Section 6](#6-atomic-ordering-audit) for the full mapping.

---

### Tier 2: Medium Impact, Moderate Complexity (Weeks)

#### P4: CPU Batch Hashing with ILP

**Impact:** ~1.5-2x improvement on CPU hash throughput
**Complexity:** ~100 lines
**Risk:** Medium (requires careful SHA-NI interleaving)
**Files:** New function in `utils/hasher.rs`, callers in `twig.rs` and `tree.rs`

**Problem:** Each `node_hash_inplace()` call runs a complete SHA256 before starting the next. Modern CPUs can overlap independent SHA256 operations when interleaved at the instruction level.

**Fix:** Implement a `batch_hash2_cpu()` function that processes 2-4 independent hash jobs simultaneously, interleaving their SHA256 compression rounds. Libraries like `sha2` don't expose this directly, but a custom implementation using `core::arch::x86_64` SHA-NI intrinsics could achieve 2-4x throughput per core.

**Alternative:** Use the [`sha2-asm`](https://crates.io/crates/sha2-asm) multi-buffer API if available, or consider the `ring` crate which has multi-buffer SHA256 support.

---

#### P5: GPU Memory Coalescing (SoA Layout)

**Impact:** ~1.3-2x GPU kernel throughput
**Complexity:** ~60 lines (kernel + host changes)
**Risk:** Medium
**Files:** `gpu/sha256_kernel.cu`, `gpu/gpu_hasher.rs`

**Problem:** The Array-of-Structures layout (`[level|left|right][level|left|right]...`) means adjacent threads read from addresses 65 bytes apart, causing non-coalesced global memory reads.

**Fix:** Restructure to Structure-of-Arrays:
```
levels[N]:  [l0, l1, l2, ...]         (N bytes, tight)
lefts[N]:   [left0, left1, ...]        (N × 32 bytes, coalesced 32B reads)
rights[N]:  [right0, right1, ...]      (N × 32 bytes, coalesced 32B reads)
```

This allows the GPU to issue coalesced 32-byte reads for left/right arrays.

---

#### P6: GPU Transfer Pipelining

**Impact:** ~30-50% reduction in GPU wall time per flush
**Complexity:** ~80 lines
**Risk:** Medium (requires async CUDA stream management)
**Files:** `gpu/gpu_hasher.rs`

**Problem:** Each `batch_node_hash()` call blocks on `dtoh_sync_copy`. Multiple GPU dispatches per flush are serialized.

**Fix:** Use CUDA streams to overlap:
- Upload of batch N+1 with compute of batch N
- Compute of batch N with download of batch N-1

**Limitation:** May require changes to cudarc or raw CUDA API calls for async memcpy.

---

#### P7: Persistent GPU Memory Pools

**Impact:** Eliminate per-call allocation overhead
**Complexity:** ~40 lines
**Risk:** Low
**Files:** `gpu/gpu_hasher.rs:77-92`

**Problem:** Each `batch_node_hash()` call allocates a new flat input buffer on the host and new device buffers for input/output.

**Fix:** Pre-allocate host and device buffers at `GpuHasher::new()` time using `max_batch_size`:
```rust
pub struct GpuHasher {
    // ...existing fields...
    d_input: CudaSlice<u8>,   // max_batch_size × 65 pre-allocated
    d_output: CudaSlice<u8>,  // max_batch_size × 32 pre-allocated
    h_input: Vec<u8>,         // max_batch_size × 65 pre-allocated
}
```

For 200K batch size: 13MB input + 6.4MB output = ~19.4MB persistent GPU memory.

---

### Tier 3: Long-Term / Experimental

#### P8: Multi-GPU Support

**Current:** Single GPU (`CudaDevice::new(0)` at `gpu_hasher.rs:35`).
**Opportunity:** For systems with multiple GPUs, distribute shard batches across devices. Each of the 16 flush shards could target a different GPU.

---

#### P9: Twig Recovery Batching

**Impact:** 30-50% faster recovery
**Complexity:** ~50 lines
**Files:** `merkletree/twigfile.rs:141-160`

**Problem:** Twig file recovery uses sequential `hash2()` calls.
**Fix:** Collect recovery hash jobs and batch them to GPU or CPU batch hasher.

---

#### P10: Lock-Free Entry Buffer Free List

**Impact:** Eliminates mutex contention on buffer reuse
**Complexity:** ~50 lines
**Files:** `entryfile/entrybuffer.rs:61, 114`

**Problem:** `free_list: Mutex<Vec<Box<BigBuf>>>` uses parking_lot mutex. Under high contention (16 shards releasing buffers), this could become a bottleneck.

**Fix:** Replace with a lock-free stack (e.g., `crossbeam::queue::SegQueue`).

---

#### P11: NUMA-Aware Twig Allocation

**Impact:** Reduced cross-socket memory latency
**Complexity:** High
**Files:** Twig allocation in `tree.rs`

**Problem:** `Box<Twig>` and `Box<TwigMT>` allocations may land on any NUMA node. A shard's twigs could be physically remote from the CPU core processing that shard.

**Fix:** Use `libnuma` or `memmap2` with NUMA policy hints to pin shard memory to the local NUMA node.

---

#### P12: Warp-Cooperative SHA256 Kernel

**Impact:** Potentially 2-3x GPU kernel throughput
**Complexity:** Very high
**Files:** `gpu/sha256_kernel.cu`

**Problem:** Each thread independently computes a full SHA256. GPU threads within a warp are underutilized (only ALU operations, no cooperation).

**Fix:** Implement a warp-level SHA256 where 8 threads cooperate on one hash (each managing one state word). This is a known technique in GPU cryptography literature but adds significant code complexity.

---

## 12. Risk Assessment & Migration Strategy

### 12.1 Risk Matrix

| Proposal | Correctness Risk | Performance Risk | Compatibility Risk |
|----------|-----------------|-----------------|-------------------|
| P1 (rayon unlock) | Low — rayon scope is already set up | None — strictly additive | None |
| P2 (pre-partition) | Low — same computation, different iteration order | None | None |
| P3 (atomic relax) | **Medium** — requires careful reasoning about memory ordering | None — strictly faster | None |
| P4 (CPU batch hash) | Medium — new hash implementation must match SHA256 spec | Low | May need new deps |
| P5 (SoA layout) | Medium — kernel change, must verify correctness | Low | GPU kernel change |
| P6 (GPU pipeline) | Medium — async CUDA is error-prone | Low | cudarc API limits |
| P7 (GPU mem pool) | Low — same operations, pre-allocated | None | None |

### 12.2 Suggested Rollout Order

```
Phase 1 (Immediate): ✅ COMPLETE
  P1 → P2 → P3
  P1: Unlocked rayon parallelism (removed slow_hashing gate)
  P2: Pre-partition n_list by shard (eliminated 75% wasted iteration)
  P3: Relaxed atomic orderings (SeqCst → Acquire/Release)

Phase 2 (Short-term): ✅ COMPLETE
  P7 → P5 → P6
  P7: Persistent GPU memory pools (pre-allocated CudaSlice buffers)
  P5: SoA (Structure-of-Arrays) GPU kernel + batch_node_hash_soa API
      - New CUDA kernel: sha256_node_hash_soa (separate levels/lefts/rights arrays)
      - Coalesced 32-byte reads for lefts/rights (32B stride vs 65B AoS stride)
      - Pre-allocated SoA device+host buffers alongside AoS buffers
      - MultiGpuHasher SoA dispatch support
  P6: Async CUDA stream pipeline (new_with_stream + memcpy_async)

Phase 3 (Medium-term): ✅ COMPLETE
  P4 → P9 → P10
  P4: batch_node_hash_cpu with Sha256::finalize_reset()
  P9: Twig recovery batching (level-0 cone via batch_node_hash_cpu)
  P10: Lock-free Treiber stack for EntryBuffer free_list

Phase 4 (Long-term / Experimental): ✅ COMPLETE
  P8 → P11 → P12
  P8: MultiGpuHasher with per-device GpuHasher and round-robin shard dispatch
  P11: NUMA topology detection + shard-to-node mapping module
  P12: Warp-cooperative SHA256 kernel (8 threads/hash via __shfl_sync)
```

### 12.3 Benchmarking Strategy ✅ COMPLETE

Criterion benchmarks have been added (`qmdb/benches/hash_benchmarks.rs`):

1. **Microbenchmarks:**
   - `single_hash`: SHA256 throughput at 32/65/128/256/512B input sizes
   - `node_hash_inplace`: In-place 65B node hash latency
   - `variable_hash`: Entry hash at 50/100/200/300B sizes

2. **Batch CPU benchmarks:**
   - `batch_cpu_hash/batch_node_hash_cpu`: P4 batch path (finalize_reset) at 10/100/1K/10K/50K
   - `batch_cpu_hash/individual_hash2`: Sequential baseline for comparison

3. **GPU benchmarks** (with `--features cuda`):
   - `gpu_node_hash/aos`: AoS kernel at 1K/10K/50K/100K/200K batch sizes
   - `gpu_node_hash/soa`: SoA kernel (P5) at same batch sizes
   - `gpu_node_hash/warp_coop`: Warp-cooperative kernel (P12) at same sizes
   - `gpu_variable_hash`: Variable-length entry hashing
   - `gpu_vs_cpu`: Direct CPU batch vs GPU AoS vs GPU SoA comparison

Run with:
```bash
cargo bench -p qmdb --bench hash_benchmarks
cargo bench -p qmdb --bench hash_benchmarks --features cuda  # GPU benchmarks
```

---

## 13. Appendix: File Reference

### Source Files

| File | Lines | Primary Role |
|------|-------|-------------|
| `qmdb/src/utils/hasher.rs` | 69 | All CPU hash functions + GPU batch wrappers |
| `qmdb/src/gpu/gpu_hasher.rs` | ~390 | GpuHasher struct, batch_node_hash, batch_hash_variable |
| `qmdb/src/gpu/sha256_kernel.cu` | 239 | CUDA SHA256 kernels (node + variable) |
| `qmdb/src/merkletree/tree.rs` | ~1300 | Upper tree sync, rayon dispatch, GPU coordination |
| `qmdb/src/merkletree/twig.rs` | ~320 | Twig sync functions, TwigMT, ActiveBits |
| `qmdb/src/merkletree/twigfile.rs` | ~200 | Twig file I/O and recovery |
| `qmdb/src/entryfile/entry.rs` | ~500 | Entry serialization and hashing |
| `qmdb/src/entryfile/entrybuffer.rs` | ~400 | Atomic ring buffer for entries |
| `qmdb/src/flusher.rs` | 544 | Flush coordination, thread spawning, barriers |
| `qmdb/src/merkletree/check.rs` | ~140 | Hash consistency validation (test/recovery) |
| `qmdb/src/def.rs` | 82 | Constants and type definitions |
| `qmdb/src/config.rs` | 107 | Runtime configuration defaults |
| `qmdb/src/lib.rs` | ~200 | Top-level init, GpuHasher creation |

### Dependencies

| Crate | Version | Role |
|-------|---------|------|
| `sha2` | 0.10.9 | CPU SHA256 with SHA-NI assembly |
| `sha2-asm` | 0.6.4 | x86_64/aarch64 assembly backends |
| `rayon` | 1.10.0 | Work-stealing thread pool |
| `parking_lot` | 0.12.1 | Fast mutexes (with `arc_lock`) |
| `dashmap` | (latest) | Concurrent HashMap for buf_map |
| `cudarc` | 0.12 | CUDA driver API bindings (optional) |
| `aes-gcm` | (latest) | AES encryption (tee_cipher feature) |

### Feature Flags

| Flag | Effect |
|------|--------|
| `cuda` | Enable GPU-accelerated hashing via cudarc |
| `slow_hashing` | Enable rayon parallelism + detached upper tree thread (ironically, this is the "faster" CPU path) |
| `check_rec` | Enable recursive hash verification |
| `tee_cipher` | Enable AES-GCM encryption on entries |
| `directio` | Enable io_uring direct I/O |
| `hpfile_all_in_mem` | Keep all data in memory |
| `in_sp1` | SP1 zkVM environment |
