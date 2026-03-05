//! GPU-accelerated SHA256 batch hashing for Merkle tree operations.
//!
//! This module provides CUDA-based parallel SHA256 hashing that can deliver
//! 3-5x throughput improvement over CPU hashing at batch sizes above 10K.
//! It includes multiple kernel variants (AoS, SoA, warp-cooperative, fused),
//! async pipelining, GPU-resident node storage, and multi-GPU support.
//!
//! Requires the `cuda` feature flag: `cargo add kyumdb --features cuda`
//!
//! # Quick Start
//!
//! ```no_run
//! use kyumdb::gpu::{GpuHasher, NodeHashJob};
//!
//! let gpu = GpuHasher::new(200_000).expect("CUDA init failed");
//! let jobs = vec![NodeHashJob {
//!     level: 1,
//!     left: [0u8; 32],
//!     right: [1u8; 32],
//! }];
//! let hashes = gpu.auto_batch_node_hash(&jobs);
//! ```
//!
//! For detailed usage patterns, see the
//! [GPU Integration Guide](https://github.com/bajpainaman/kyumdb/blob/main/docs/gpu-integration-guide.md).

mod gpu_hasher;
mod gpu_node_store;
#[cfg(test)]
mod gpu_tests;
#[cfg(test)]
mod integration_tests;

pub use gpu_hasher::{GpuHasher, MultiGpuHasher, NodeHashJob};
pub use gpu_node_store::GpuNodeStore;
