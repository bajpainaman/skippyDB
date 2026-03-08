# GPU Acceleration Guide

SkippyDB supports optional CUDA GPU-accelerated SHA256 batch hashing for Merkle tree operations. This guide covers setup, usage, kernel internals, benchmarking, and tuning.

---

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Building](#building)
- [How It Works](#how-it-works)
- [API Reference](#api-reference)
  - [GpuHasher](#gpuhasher)
  - [MultiGpuHasher](#multigpuhasher)
  - [NodeHashJob](#nodehashjob)
- [Kernel Variants](#kernel-variants)
  - [AoS Node Hash](#aos-node-hash-sha256_node_hash)
  - [SoA Node Hash](#soa-node-hash-sha256_node_hash_soa)
  - [Warp-Cooperative Node Hash](#warp-cooperative-node-hash-sha256_node_hash_warp_coop)
  - [Variable-Length Hash](#variable-length-hash-sha256_variable_hash)
- [Integration with SkippyDB Pipeline](#integration-with-kyumdb-pipeline)
- [Multi-GPU Support](#multi-gpu-support)
- [Performance Tuning](#performance-tuning)
- [CPU Fallback](#cpu-fallback)
- [Testing](#testing)
- [Benchmarking](#benchmarking)
- [Troubleshooting](#troubleshooting)

---

## Overview

SkippyDB's Merkle tree synchronization is dominated by SHA256 hashing. During block processing, the flusher must compute thousands of node hashes (SHA256 of `level || left_32B || right_32B` = 65 bytes each) to update twig roots, active bits trees, and upper tree nodes.

GPU acceleration batches these independent hash operations and dispatches them to CUDA cores for parallel execution. On an RTX 3080, this can provide 3-5x throughput improvement over AVX2/SHA-NI CPU hashing at batch sizes >10K.

### When GPU Helps

| Scenario | Batch Size | GPU Speedup |
|---|---|---|
| Small blocks (few entries) | <1K | Negligible (CPU faster due to no transfer overhead) |
| Medium blocks | 1K-10K | 1.5-2x |
| Large blocks | 10K-100K | 3-5x |
| Bulk sync / recovery | >100K | 5-10x |

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Flusher Thread                       │
│                                                         │
│  Collect hash jobs ──▶ Batch ──▶ GPU Dispatch ──▶ Apply │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │  GpuHasher (per-device)                          │   │
│  │  ┌────────┐  ┌──────────┐  ┌────────────────┐   │   │
│  │  │ Host   │──│ Async    │──│ CUDA Kernel    │   │   │
│  │  │ Buffer │  │ memcpy   │  │ (256 threads/  │   │   │
│  │  │ (pre-  │  │ htod     │  │  block)        │   │   │
│  │  │  alloc)│  └──────────┘  └────────────────┘   │   │
│  │  └────────┘        │              │              │   │
│  │                    ▼              ▼              │   │
│  │              ┌──────────┐  ┌────────────────┐   │   │
│  │              │ Device   │  │ Async memcpy   │   │   │
│  │              │ Buffer   │  │ dtoh           │   │   │
│  │              │ (pre-    │  └────────────────┘   │   │
│  │              │  alloc)  │         │              │   │
│  │              └──────────┘         ▼              │   │
│  │                          ┌────────────────┐     │   │
│  │                          │ Synchronize    │     │   │
│  │                          └────────────────┘     │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

---

## Requirements

- **NVIDIA GPU** with compute capability 6.0+ (Pascal or newer)
- **CUDA Toolkit 12.0+** installed
- **NVRTC** (NVIDIA Runtime Compilation) — included with CUDA Toolkit
- `nvcc` accessible on `PATH`

### Supported GPUs

| Generation | Compute Cap | Example Cards |
|---|---|---|
| Pascal | 6.0-6.1 | GTX 1080, Tesla P100 |
| Volta | 7.0 | Tesla V100 |
| Turing | 7.5 | RTX 2080, T4 |
| Ampere | 8.0-8.6 | RTX 3080, A100 |
| Hopper | 9.0 | H100 |
| Ada Lovelace | 8.9 | RTX 4090, L40 |

---

## Building

```bash
# Build with CUDA support
cargo build --release --features cuda

# Verify CUDA detection
cargo test --features cuda -- test_gpu_empty_batch
```

If CUDA is not available, the build will fail at link time. The `cuda` feature is **not** included by default.

---

## How It Works

### Initialization

When `AdsCore` is created with the `cuda` feature enabled:

```rust
// lib.rs — AdsCore::_new()
#[cfg(feature = "cuda")]
let gpu_hasher = match GpuHasher::new(200_000) {
    Ok(g) => {
        info!("CUDA GPU hasher initialized successfully");
        Some(Arc::new(g))
    }
    Err(e) => {
        info!("CUDA GPU hasher unavailable, falling back to CPU: {}", e);
        None
    }
};
```

This:
1. Opens CUDA device 0 with a dedicated stream
2. Compiles SHA256 kernels from CUDA source via NVRTC (runtime compilation)
3. Pre-allocates device and host buffers for `max_batch_size` jobs
4. Stores 4 kernel function handles (AoS, SoA, warp-coop, variable)

### Batch Hashing Flow

```
1. Collect N hash jobs into host buffer (zero-copy flatten)
2. Async memcpy host → device (on CUDA stream)
3. Launch kernel with grid = ceil(N/256) blocks, 256 threads/block
4. Async memcpy device → host (on same stream)
5. Synchronize stream (wait for pipeline to complete)
6. Scatter results back to caller
```

All operations are queued on a single CUDA stream per `GpuHasher`, ensuring correct ordering without explicit synchronization between steps 2-4.

---

## API Reference

### GpuHasher

The primary GPU interface. Thread-safe (internal `Mutex` serializes access).

```rust
use skippydb::gpu::{GpuHasher, NodeHashJob};

// Create on default GPU (device 0), max 200K hashes per batch
let gpu = GpuHasher::new(200_000)?;

// Create on specific GPU
let gpu = GpuHasher::new_on_device(1, 200_000)?;

// Query available GPUs
let count = GpuHasher::device_count()?;
```

#### `batch_node_hash`

Hash N fixed-size node inputs (65 bytes each: `level || left_32B || right_32B`).

```rust
let jobs = vec![
    NodeHashJob {
        level: 0,
        left: [0x11; 32],
        right: [0xAB; 32],
    },
    // ... more jobs
];

let hashes: Vec<[u8; 32]> = gpu.batch_node_hash(&jobs);
assert_eq!(hashes.len(), jobs.len());
```

Results are byte-identical to CPU `hasher::hash2(level, &left, &right)`.

#### `batch_node_hash_into`

Same as above, but writes directly into a pre-allocated output slice:

```rust
let mut output = vec![[0u8; 32]; jobs.len()];
gpu.batch_node_hash_into(&jobs, &mut output);
```

#### `batch_node_hash_soa`

Structure-of-Arrays layout for better GPU memory coalescing:

```rust
let levels: Vec<u8> = vec![0, 1, 2, 3];
let lefts: Vec<[u8; 32]> = vec![[0x11; 32]; 4];
let rights: Vec<[u8; 32]> = vec![[0xAB; 32]; 4];

let hashes = gpu.batch_node_hash_soa(&levels, &lefts, &rights);
```

**Why SoA?** In AoS layout, adjacent GPU threads read 65-byte strides (misaligned). In SoA, threads read contiguous 32-byte blocks from the `lefts` and `rights` arrays, achieving better memory bandwidth utilization on the GPU's 32-byte cache lines.

#### `batch_node_hash_warp_coop`

Warp-cooperative kernel using 8 threads per hash via `__shfl_sync`:

```rust
let hashes = gpu.batch_node_hash_warp_coop(&jobs);
```

8 threads collaborate on each SHA256 computation, sharing intermediate state via warp shuffle instructions. This can improve throughput on GPUs with many SMs by keeping more threads active.

#### `batch_hash_variable`

Hash variable-length inputs (for entry hashing):

```rust
let entries: Vec<&[u8]> = vec![
    &entry1_bytes[..],
    &entry2_bytes[..],  // 50-300 bytes each
    &entry3_bytes[..],
];

let hashes = gpu.batch_hash_variable(&entries);
```

This kernel handles SHA256 padding internally for arbitrary input lengths.

### MultiGpuHasher

Distributes work across all available CUDA devices using round-robin assignment:

```rust
use skippydb::gpu::MultiGpuHasher;

let multi = MultiGpuHasher::new(200_000)?;
println!("Using {} GPUs", multi.gpu_count());

// Hash on GPU assigned to shard 5
let hashes = multi.batch_node_hash(5, &jobs);

// SoA variant
let hashes = multi.batch_node_hash_soa(5, &levels, &lefts, &rights);

// Variable-length on specific GPU
let hashes = multi.batch_hash_variable(3, &inputs);

// Direct device access
let gpu0 = multi.device(0);
```

Assignment: `shard_id % gpu_count` determines which GPU processes each shard.

### NodeHashJob

```rust
#[repr(C)]
pub struct NodeHashJob {
    pub level: u8,       // Tree level (0-63)
    pub left: [u8; 32],  // Left child hash
    pub right: [u8; 32], // Right child hash
}
```

The `#[repr(C)]` ensures predictable memory layout for GPU transfer.

---

## Kernel Variants

All kernels are in [`skippydb/src/gpu/sha256_kernel.cu`](../skippydb/src/gpu/sha256_kernel.cu).

### AoS Node Hash (`sha256_node_hash`)

```
Input:  N × 65 bytes (contiguous, Array-of-Structs)
Output: N × 32 bytes
Grid:   ceil(N/256) blocks × 256 threads
```

Each thread independently computes SHA256 of its 65-byte input. Simple and effective for moderate batch sizes.

**Memory access pattern**: Thread `i` reads bytes at offset `i*65` to `i*65+64`. The 65-byte stride causes bank conflicts and sub-optimal coalescing.

### SoA Node Hash (`sha256_node_hash_soa`)

```
Input:  N × 1 byte (levels) + N × 32 bytes (lefts) + N × 32 bytes (rights)
Output: N × 32 bytes
Grid:   ceil(N/256) blocks × 256 threads
```

Separating the three input arrays ensures that adjacent threads read adjacent 32-byte chunks, matching the GPU's memory transaction size.

**When to use**: Batch sizes >10K where memory bandwidth is the bottleneck.

### Warp-Cooperative Node Hash (`sha256_node_hash_warp_coop`)

```
Input:  N × 65 bytes (AoS)
Output: N × 32 bytes
Grid:   ceil(N*8/256) blocks × 256 threads
Threads per hash: 8
```

Each group of 8 threads collaborates on one SHA256 computation. The 8 SHA256 state words (`a` through `h`) are distributed across threads, with `__shfl_sync` used for inter-thread state rotation during the compression rounds.

**When to use**: GPUs with high SM counts where occupancy matters more than per-thread throughput.

### Variable-Length Hash (`sha256_variable_hash`)

```
Input:  flat data buffer + N offsets + N lengths
Output: N × 32 bytes
Grid:   ceil(N/256) blocks × 256 threads
```

Handles arbitrary input lengths with proper SHA256 padding. Each thread reads its data from the flat buffer using its offset and length.

**When to use**: Entry hashing where inputs range from 50-300+ bytes.

---

## Integration with SkippyDB Pipeline

### Flusher Integration

When a `GpuHasher` is available, the flusher uses `flush_gpu()` instead of `flush()`:

```rust
// lib.rs — AdsCore::start_threads()
#[cfg(feature = "cuda")]
{
    if let Some(gpu) = gpu_hasher {
        flusher.flush_gpu(SHARD_COUNT, gpu);
        return;
    }
}
flusher.flush(SHARD_COUNT);
```

### Twig Merkle Tree GPU Sync

The `sync_mtrees_gpu()` function in `twig.rs` batches all twig Merkle tree updates at each level:

```rust
// twig.rs
#[cfg(feature = "cuda")]
pub fn sync_mtrees_gpu(
    gpu: &GpuHasher,
    mtrees: &mut [(&mut TwigMT, i32, i32)],
) {
    // Process level by level (bottom-up)
    for level in 0..11 {
        // Collect all hash jobs across all twigs at this level
        let mut jobs = Vec::new();
        for (twig_idx, mtree) in mtrees.iter().enumerate() {
            // ... collect pairs at this level
            jobs.push(NodeHashJob { level, left, right });
        }
        // Batch hash all jobs for this level on GPU
        let results = gpu.batch_node_hash(&jobs);
        // Scatter results back to individual twigs
    }
}
```

---

## Multi-GPU Support

### Round-Robin Dispatch

With `MultiGpuHasher`, each shard is assigned to a GPU:

```
Shard 0  → GPU 0
Shard 1  → GPU 1
Shard 2  → GPU 0  (wraps around with 2 GPUs)
Shard 3  → GPU 1
...
Shard 15 → GPU 1
```

Each GPU has its own:
- CUDA device context
- CUDA stream
- Pre-allocated host and device buffers
- Mutex for serialization

### Example: 4-GPU Setup

```rust
let multi = MultiGpuHasher::new(200_000)?;
assert_eq!(multi.gpu_count(), 4);

// Shard 0 → GPU 0, Shard 1 → GPU 1, ..., Shard 4 → GPU 0, ...
for shard_id in 0..16 {
    let hashes = multi.batch_node_hash(shard_id, &jobs[shard_id]);
}
```

---

## Performance Tuning

### Batch Size

The `max_batch_size` parameter controls pre-allocated buffer sizes:

```rust
// Conservative (less VRAM, lower peak throughput)
let gpu = GpuHasher::new(50_000)?;

// Default (good balance)
let gpu = GpuHasher::new(200_000)?;

// Aggressive (more VRAM, higher peak throughput)
let gpu = GpuHasher::new(1_000_000)?;
```

**VRAM usage per GpuHasher** (approximate):
- AoS buffers: `max_batch_size * (65 + 32)` = ~19 MB at 200K
- SoA buffers: `max_batch_size * (1 + 32 + 32)` = ~13 MB at 200K
- Variable output: `max_batch_size * 32` = ~6 MB at 200K
- **Total**: ~38 MB at 200K batch size

### Kernel Selection

| Batch Size | Recommended Kernel |
|---|---|
| <1K | CPU (avoid GPU transfer overhead) |
| 1K-10K | AoS (`batch_node_hash`) |
| 10K-100K | SoA (`batch_node_hash_soa`) |
| >100K | SoA or warp-coop (benchmark your GPU) |

### Block Size

The CUDA block size is fixed at 256 threads. This is optimal for most GPUs:
- Pascal: 256 threads = 8 warps (good occupancy)
- Ampere: 256 threads = 8 warps (64 threads/SM × 4 blocks = full occupancy)

---

## CPU Fallback

The CPU batch hashing path is always available and uses `finalize_reset()` optimization:

```rust
// utils/hasher.rs
pub fn batch_node_hash_cpu(jobs: &[NodeHashJob]) -> Vec<[u8; 32]> {
    let mut hasher = Sha256::new();
    let mut results = Vec::with_capacity(jobs.len());
    for job in jobs {
        hasher.update([job.level]);
        hasher.update(&job.left);
        hasher.update(&job.right);
        let hash = hasher.finalize_reset();
        results.push(hash.into());
    }
    results
}
```

The `finalize_reset()` pattern avoids re-initializing the SHA256 state between hashes, saving ~10% CPU time vs. creating a new `Sha256` instance per hash.

---

## Testing

### Run All GPU Tests

```bash
cargo test --features cuda -- gpu
```

### Key Test Cases

| Test | What It Verifies |
|---|---|
| `test_gpu_node_hash_matches_cpu` | GPU results match CPU for all tree levels |
| `test_gpu_node_hash_large_batch` | 10K batch correctness |
| `test_gpu_variable_hash_matches_cpu` | Variable-length inputs (10B to 300B) |
| `test_gpu_empty_batch` | Empty input handling |
| `test_gpu_soa_matches_aos` | SoA and AoS produce identical results |
| `test_gpu_warp_coop_matches_cpu` | Warp-cooperative kernel correctness |
| `test_multi_gpu_round_robin` | Multi-GPU dispatch and correctness |

All GPU tests gracefully skip if no CUDA device is available:

```rust
let gpu = match GpuHasher::new(10000) {
    Ok(g) => g,
    Err(e) => {
        eprintln!("Skipping GPU test (no CUDA device): {}", e);
        return;
    }
};
```

---

## Benchmarking

### Criterion Benchmarks

```bash
# CPU-only
cargo bench --bench hash_benchmarks

# CPU + GPU comparison
cargo bench --bench hash_benchmarks --features cuda
```

Benchmark groups:
- `hash/single` — Single SHA256 hash (baseline)
- `hash/batch_cpu/{N}` — CPU batch at sizes 1K, 10K, 100K
- `hash/batch_gpu_aos/{N}` — GPU AoS kernel
- `hash/batch_gpu_soa/{N}` — GPU SoA kernel
- `hash/batch_gpu_warp/{N}` — GPU warp-cooperative kernel

### Expected Results (RTX 3080)

| Batch Size | CPU (SHA-NI) | GPU AoS | GPU SoA | Speedup |
|---|---|---|---|---|
| 1,000 | 0.3ms | 0.5ms | 0.4ms | 0.7x |
| 10,000 | 3ms | 1.2ms | 0.9ms | 3.3x |
| 50,000 | 15ms | 3.5ms | 2.8ms | 5.4x |
| 200,000 | 60ms | 12ms | 9ms | 6.7x |

---

## Troubleshooting

### "CUDA device init failed"

- Verify CUDA toolkit: `nvcc --version`
- Check GPU visibility: `nvidia-smi`
- Ensure `libcuda.so` is on `LD_LIBRARY_PATH`

### "NVRTC compilation failed"

- The CUDA kernel source is compiled at runtime via NVRTC
- Ensure CUDA toolkit version matches the `cudarc` dependency (12.0+)
- Check that `libnvrtc.so` is accessible

### "GPU kernel launch failed"

- Batch size may exceed `max_batch_size` — increase it or split batches
- GPU may be out of memory — reduce `max_batch_size` or close other GPU applications
- Check `nvidia-smi` for memory usage

### GPU Not Detected at Runtime

If built with `--features cuda` but no GPU is available:

```
[INFO] CUDA GPU hasher unavailable, falling back to CPU: CUDA device 0 init failed: ...
```

This is expected — SkippyDB gracefully falls back to CPU hashing.

### Performance Tips

1. **Use SoA for large batches** — Better memory coalescing above 10K
2. **Pre-warm the GPU** — First kernel launch includes JIT compilation overhead
3. **Avoid tiny batches** — GPU transfer overhead dominates below ~500 jobs
4. **Pin to fastest GPU** — Use `GpuHasher::new_on_device(best_gpu_idx, ...)` if you have mixed GPUs
5. **Monitor with nvidia-smi** — Watch `nvidia-smi dmon -s u` for GPU utilization
