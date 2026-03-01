use sha2::{Digest, Sha256};

pub type Hash32 = [u8; 32];

pub const ZERO_HASH32: Hash32 = [0u8; 32];

pub fn hash<T: AsRef<[u8]>>(a: T) -> Hash32 {
    let mut hasher = Sha256::new();
    hasher.update(a);
    hasher.finalize().into()
}

pub fn hash1<T: AsRef<[u8]>>(level: u8, a: T) -> Hash32 {
    let mut hasher = Sha256::new();
    hasher.update([level]);
    hasher.update(a);
    hasher.finalize().into()
}

pub fn hash2<T: AsRef<[u8]>>(children_level: u8, a: T, b: T) -> Hash32 {
    let mut hasher = Sha256::new();
    hasher.update([children_level]);
    hasher.update(a);
    hasher.update(b);
    hasher.finalize().into()
}

pub fn hash2x<T: AsRef<[u8]>>(children_level: u8, a: T, b: T, exchange_ab: bool) -> Hash32 {
    if exchange_ab {
        hash2(children_level, b, a)
    } else {
        hash2(children_level, a, b)
    }
}

pub fn node_hash_inplace<T: AsRef<[u8]>>(
    children_level: u8,
    target: &mut [u8],
    src_a: T,
    src_b: T,
) {
    let mut hasher = Sha256::new();
    hasher.update([children_level]);
    hasher.update(src_a);
    hasher.update(src_b);
    target.copy_from_slice(&hasher.finalize());
}

/// Batch CPU node hashing: hash N independent (level, left, right) tuples.
/// Processes hashes in a tight loop with a reused Sha256 instance to minimize
/// overhead and improve cache locality over individual hash2() calls.
pub fn batch_node_hash_cpu(
    levels: &[u8],
    lefts: &[[u8; 32]],
    rights: &[[u8; 32]],
    out: &mut [[u8; 32]],
) {
    debug_assert_eq!(levels.len(), lefts.len());
    debug_assert_eq!(levels.len(), rights.len());
    debug_assert_eq!(levels.len(), out.len());

    let mut hasher = Sha256::new();
    for i in 0..levels.len() {
        hasher.update([levels[i]]);
        hasher.update(&lefts[i]);
        hasher.update(&rights[i]);
        out[i].copy_from_slice(&hasher.finalize_reset());
    }
}

/// Batch node hash using GPU. Hashes N jobs of (level, left, right) → N hashes.
/// Falls back to CPU if `gpu` is None.
#[cfg(feature = "cuda")]
pub fn batch_node_hash_gpu(
    gpu: &crate::gpu::GpuHasher,
    jobs: &[crate::gpu::NodeHashJob],
    out: &mut [[u8; 32]],
) {
    gpu.batch_node_hash_into(jobs, out);
}

/// Batch hash variable-length entries using GPU.
/// Falls back to CPU if `gpu` is None.
#[cfg(feature = "cuda")]
pub fn batch_hash_variable_gpu(
    gpu: &crate::gpu::GpuHasher,
    inputs: &[&[u8]],
) -> Vec<Hash32> {
    gpu.batch_hash_variable(inputs)
}

/// Batch node hash using GPU with SoA (Structure-of-Arrays) layout.
/// SoA enables coalesced GPU memory reads for improved bandwidth utilization.
#[cfg(feature = "cuda")]
pub fn batch_node_hash_soa_gpu(
    gpu: &crate::gpu::GpuHasher,
    levels: &[u8],
    lefts: &[[u8; 32]],
    rights: &[[u8; 32]],
    out: &mut [[u8; 32]],
) {
    gpu.batch_node_hash_soa_into(levels, lefts, rights, out);
}
