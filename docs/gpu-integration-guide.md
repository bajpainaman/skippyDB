# GPU Integration Guide

Prescriptive guide for using KyumDB's GPU-accelerated hashing. This covers
**when** and **how** to use each GPU entry point. For kernel internals and
architecture details, see `gpu-acceleration.md`.

---

## 1. Quick Decision Tree

```
                  How many hash jobs?
                         |
            +------------+------------+
            |            |            |
         < 256       256-1024      > 1024
            |            |            |
         CPU only     AoS kernel   SoA kernel
```

| Batch Size | Recommended Path | Why |
|------------|-----------------|-----|
| < 256 | CPU (`batch_node_hash_cpu`) | GPU launch overhead dominates; CPU is faster |
| 256 -- 1024 | AoS kernel (`batch_node_hash`) | Good GPU utilization, simple 65-byte stride |
| > 1024 | SoA kernel (`batch_node_hash_soa`) | Coalesced 32-byte reads saturate memory bandwidth |

**In practice, use `auto_batch_node_hash`.** It encodes these thresholds
internally and picks the right kernel for you. The thresholds above are
provided so you understand what it does, not so you reimplement the logic.

---

## 2. The Recommended Path: `auto_batch_node_hash`

This is the default entry point for all Merkle node hashing. It selects
CPU, AoS GPU, or SoA GPU based on the job count.

```rust
use skippydb::gpu::{GpuHasher, NodeHashJob};

let gpu = GpuHasher::new(200_000)?;
let jobs: Vec<NodeHashJob> = /* collect your hash jobs */;
let hashes = gpu.auto_batch_node_hash(&jobs);
// hashes[i] = SHA256(jobs[i].level || jobs[i].left || jobs[i].right)
```

Each `NodeHashJob` is a fixed-size 65-byte input:

```rust
pub struct NodeHashJob {
    pub level: u8,        // 1 byte: tree level
    pub left:  [u8; 32],  // 32 bytes: left child hash
    pub right: [u8; 32],  // 32 bytes: right child hash
}
```

### Zero-allocation variant

When you already have an output buffer, use the `_into` variant to skip
the `Vec` allocation on the return path:

```rust
let mut out = vec![[0u8; 32]; jobs.len()];
gpu.auto_batch_node_hash_into(&jobs, &mut out);
```

### Constructor note

`GpuHasher::new(max_batch_size)` pre-allocates all host and device
buffers at construction time. Choose `max_batch_size` to be the largest
batch you will ever submit. Passing a batch larger than this value panics.

---

## 3. Fused Active Bits: `batch_active_bits_fused`

A single GPU dispatch that computes the entire twig right-side hash chain
(L2, L3, and twig root) in one kernel launch. Replaces four separate
`batch_node_hash` calls.

### When to use it

Use this whenever you have a batch of twigs whose active-bits (L1 hashes)
and left-root hashes are ready. It eliminates three round-trip
upload-compute-download cycles compared to hashing each level separately.

### Signature

```rust
pub fn batch_active_bits_fused(
    &self,
    l1_values: &[[u8; 32]],  // 4 * N L1 hashes (4 per twig, contiguous)
    left_roots: &[[u8; 32]], // N left-root hashes
) -> (Vec<[u8; 32]>, Vec<[u8; 32]>, Vec<[u8; 32]>)
// Returns (twig_roots, l2_values, l3_values)
```

### What it computes per twig

```
L2[0] = SHA256(9  || L1[0] || L1[1])
L2[1] = SHA256(9  || L1[2] || L1[3])
L3    = SHA256(10 || L2[0] || L2[1])
root  = SHA256(11 || left_root || L3)
```

### Example

```rust
use skippydb::gpu::{GpuHasher};

let gpu = GpuHasher::new(100_000)?;

// 3 twigs: 12 L1 hashes + 3 left roots
let l1_values: Vec<[u8; 32]> = collect_l1_hashes(); // length = 12
let left_roots: Vec<[u8; 32]> = collect_left_roots(); // length = 3

let (twig_roots, l2_values, l3_values) =
    gpu.batch_active_bits_fused(&l1_values, &left_roots);

assert_eq!(twig_roots.len(), 3);
assert_eq!(l2_values.len(), 6);  // 2 per twig
assert_eq!(l3_values.len(), 3);
```

The returned `l2_values` and `l3_values` are needed for Merkle proof
generation. Do not discard them unless you are only computing the root.

---

## 4. Async Pipelining: `batch_node_hash_async`

Dispatches a batch on a secondary CUDA stream and returns immediately.
The caller gets a `GpuPending` handle and can do CPU work (building the
next batch, I/O, etc.) while the GPU computes.

### Pattern

```rust
let mut gpu = GpuHasher::new(200_000)?;

// Dispatch level N
let pending = gpu.batch_node_hash_async(&level_n_jobs);

// CPU work while GPU computes
let level_n_plus_1_jobs = build_next_level_jobs(/* ... */);

// Collect results (blocks until GPU is done)
let level_n_hashes = pending.wait();
```

### Rules

1. **One in flight at a time.** You must call `wait()` (or `wait_into()`)
   on the current `GpuPending` before dispatching another async batch.
   The borrow checker enforces this: `batch_node_hash_async` takes
   `&mut self`, and `GpuPending` borrows the hasher.

2. **`wait()` vs `wait_into()`.** Use `wait()` to get a fresh `Vec`.
   Use `wait_into(&mut buf)` to write directly into a pre-allocated
   buffer.

3. **Drop safety.** Dropping a `GpuPending` without calling `wait()`
   will synchronize the stream implicitly. This is safe but wastes the
   pipelining benefit.

### When to use it

Async dispatch is valuable when the CPU has meaningful work to do between
dispatch and collection. If you would just call `wait()` immediately
after dispatch, use the synchronous `batch_node_hash` instead.

---

## 5. GPU-Resident Upper Tree: `GpuNodeStore`

For the upper levels of the Merkle tree (above the twig layer), the node
hashes can be kept entirely on the GPU using `GpuNodeStore`. This avoids
host-device transfers for intermediate tree levels.

### Architecture

`GpuNodeStore` wraps a `FlashMap<u64, [u8; 32]>` (GPU-resident hash map).
Keys are `NodePos` values encoded as `(level << 56) | nth`. Values are
32-byte SHA256 hashes.

### Key methods

| Method | Direction | Purpose |
|--------|-----------|---------|
| `insert_from_host` | H -> D | Populate twig roots from CPU |
| `sync_level_on_device` | D -> D | Hash one tree level entirely on GPU |
| `sync_upper_nodes_on_device` | D -> D, then D -> H (root only) | Hash all upper levels, return root |
| `get_to_host` | D -> H | Read specific nodes back to CPU |
| `get_single` | D -> H | Read one node (convenience wrapper) |

### When to use `sync_upper_nodes_on_device` vs CPU

Use GPU-resident sync when:
- The upper tree has many levels (e.g., 10+ levels above twigs)
- The n_list (dirty node indices) per level is large (hundreds+)
- You want to avoid N round-trip H<->D transfers per level

Stay on CPU when:
- Only a few nodes changed (small n_list)
- The tree is shallow (2-3 upper levels)

### Example

```rust
use skippydb::gpu::{GpuHasher, GpuNodeStore};

let gpu = GpuHasher::new(200_000)?;
let mut store = GpuNodeStore::new()?;

// Populate twig roots from CPU
let twig_pairs: Vec<(u64, [u8; 32])> = collect_twig_root_pairs();
store.insert_from_host(&twig_pairs)?;

// Sync all upper levels on GPU (no intermediate H<->D transfers)
let n_list: Vec<u64> = dirty_node_indices();
let first_level: i64 = 13;  // first level above twigs
let max_level: i64 = 20;    // tree root level

let (_final_n_list, root_hash) =
    store.sync_upper_nodes_on_device(&gpu, n_list, first_level, max_level)?;

// Only root_hash (32 bytes) was transferred D->H
```

### Capacity

Default capacity is 1M slots (2^20). With the 50% load factor used by
FlashMap, this supports up to ~512K nodes. Use `GpuNodeStore::with_capacity`
for larger trees.

---

## 6. Multi-GPU: `MultiGpuHasher`

Distributes work across all available CUDA devices using round-robin
shard assignment.

### Constructor

```rust
use skippydb::gpu::MultiGpuHasher;

let multi = MultiGpuHasher::new(200_000)?;
println!("Using {} GPUs", multi.gpu_count());
```

Each GPU gets its own `GpuHasher` with independent pre-allocated buffers
and CUDA streams.

### Shard dispatch with `for_shard`

The `for_shard(shard_id)` method returns the `GpuHasher` assigned to a
given shard via round-robin: `shard_id % gpu_count()`.

```rust
let multi = MultiGpuHasher::new(200_000)?;

// In a sharded tree, each shard dispatches to its assigned GPU
for shard_id in 0..num_shards {
    let jobs = collect_jobs_for_shard(shard_id);
    let gpu = multi.for_shard(shard_id);
    let hashes = gpu.auto_batch_node_hash(&jobs);
    apply_hashes(shard_id, &hashes);
}
```

### Convenience methods

`MultiGpuHasher` also exposes direct batch methods that take a `shard_id`:

```rust
let hashes = multi.batch_node_hash(shard_id, &jobs);
let var_hashes = multi.batch_hash_variable(shard_id, &inputs);
let soa_hashes = multi.batch_node_hash_soa(shard_id, &levels, &lefts, &rights);
```

### Fallback

On a single-GPU system, `MultiGpuHasher` works identically to a plain
`GpuHasher` -- all shards map to GPU 0.

---

## 7. Anti-Patterns

### Single-job GPU calls

```rust
// WRONG: GPU launch overhead (~5us) dominates a single 65-byte hash
let one_hash = gpu.batch_node_hash(&[job]);

// RIGHT: use CPU for small batches, or let auto_batch decide
let one_hash = gpu.auto_batch_node_hash(&[job]); // falls through to CPU
```

### Manually picking AoS/SoA

```rust
// WRONG: hard-coding kernel selection
if some_condition {
    gpu.batch_node_hash(&jobs)
} else {
    gpu.batch_node_hash_soa(&levels, &lefts, &rights)
}

// RIGHT: let the adaptive dispatcher handle it
gpu.auto_batch_node_hash(&jobs)
```

The only reason to call `batch_node_hash` or `batch_node_hash_soa`
directly is benchmarking or when you already have data in SoA layout
and want to avoid the conversion overhead inside `auto_batch_node_hash`.

### Ignoring async lifetime rules

```rust
// WRONG: dispatching twice without waiting
let pending_a = gpu.batch_node_hash_async(&jobs_a);
let pending_b = gpu.batch_node_hash_async(&jobs_b); // compile error: &mut borrow conflict

// RIGHT: wait before dispatching the next batch
let result_a = gpu.batch_node_hash_async(&jobs_a).wait();
let pending_b = gpu.batch_node_hash_async(&jobs_b);
// ... do CPU work ...
let result_b = pending_b.wait();
```

The borrow checker prevents this at compile time. If you find yourself
fighting it, you are using the API incorrectly.

### Forgetting to call `gpu.sync()` after device operations

```rust
// WRONG: reading device results before sync completes
let d_results = gpu.batch_node_hash_device_soa(&d_levels, &d_lefts, &d_rights, n);
store.bulk_insert_device(&d_parent_keys, &d_results, n)?; // kernel may not be done

// RIGHT: sync before consuming device output
let d_results = gpu.batch_node_hash_device_soa(&d_levels, &d_lefts, &d_rights, n);
gpu.sync();
store.bulk_insert_device(&d_parent_keys, &d_results, n)?;
```

`batch_node_hash` and `batch_node_hash_soa` synchronize internally before
returning host results. The `_device_` variants do **not** -- the caller
must call `gpu.sync()` explicitly.

### Using GPU for variable-length hashing on small inputs

```rust
// WRONG: GPU variable-length hash for 10 short strings
let inputs: Vec<&[u8]> = small_strings.iter().map(|s| s.as_bytes()).collect();
let hashes = gpu.batch_hash_variable(&inputs);

// RIGHT: use CPU sha2 crate directly for small/few inputs
use sha2::{Digest, Sha256};
let hashes: Vec<[u8; 32]> = small_strings
    .iter()
    .map(|s| Sha256::digest(s.as_bytes()).into())
    .collect();
```

`batch_hash_variable` requires per-call device allocation (the total
input size varies). It only wins over CPU when the batch is large
(thousands of entries) and the inputs are non-trivial in size.
