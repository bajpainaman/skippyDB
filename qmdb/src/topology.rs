//! Runtime topology: how many shards the DB operates with and how many
//! updater workers each shard fans out to.
//!
//! # Phased rollout (Phase 2.3)
//!
//! - **2.3a (this commit)**: introduce the `Topology` type and thread it
//!   through `AdsCore::with_topology` / `AdsWrap::with_topology`. Fixed
//!   `[T; SHARD_COUNT]` arrays are unchanged; `Topology::new` asserts that
//!   the runtime `shard_count` equals the compile-time `SHARD_COUNT` so
//!   existing call sites stay correct. No behavior change.
//! - **2.3b**: swap `[T; SHARD_COUNT]` in `MetaInfo` / `Tree` / `Flusher` /
//!   `AdsCore` to `Box<[T]>` sized at construction from `topology.shard_count`.
//! - **2.3c**: replace `byte0_to_shard_id` with a bit-range hash that takes
//!   `ceil(log2(shard_count))` bits from `key_hash[0..2]`.
//! - **2.3d**: parametric root-hash-determinism tests at `shard_count`
//!   ∈ {16, 32, 64}.

use crate::def::SHARD_COUNT;

/// The shape of a SkippyDB deployment. Passed at open time; immutable for
/// the lifetime of the `AdsCore`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Topology {
    /// Number of data shards. Must match `SHARD_COUNT` during Phase 2.3a;
    /// Phase 2.3b lifts that constraint.
    pub shard_count: usize,
    /// How many updater worker threads each shard fans out to. `1` preserves
    /// the pre-Phase-2.4 single-worker-per-shard behavior. Must be >= 1.
    pub workers_per_shard: usize,
}

impl Topology {
    /// Build the classic 16-shard / single-worker topology. Matches behavior
    /// on every branch before Phase 2.
    #[inline]
    pub const fn compile_time() -> Self {
        Self {
            shard_count: SHARD_COUNT,
            workers_per_shard: 1,
        }
    }

    /// Construct a Topology with explicit knobs. Phase 2.3a asserts
    /// `shard_count == SHARD_COUNT` — callers that want a different
    /// shard count must wait for Phase 2.3b. `workers_per_shard >= 1`.
    pub fn new(shard_count: usize, workers_per_shard: usize) -> Self {
        assert!(
            shard_count == SHARD_COUNT,
            "Phase 2.3a pins runtime shard_count ({}) to compile-time \
             SHARD_COUNT ({}); Phase 2.3b lifts this once fixed arrays are \
             swapped to Box<[T]>.",
            shard_count,
            SHARD_COUNT,
        );
        assert!(
            workers_per_shard >= 1,
            "workers_per_shard must be >= 1 (got {})",
            workers_per_shard,
        );
        Self {
            shard_count,
            workers_per_shard,
        }
    }
}

impl Default for Topology {
    fn default() -> Self {
        Self::compile_time()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_compile_time_shard_count() {
        let t = Topology::default();
        assert_eq!(t.shard_count, SHARD_COUNT);
        assert_eq!(t.workers_per_shard, 1);
    }

    #[test]
    fn new_rejects_mismatched_shard_count() {
        let bad = SHARD_COUNT + 1;
        let err = std::panic::catch_unwind(|| Topology::new(bad, 1));
        assert!(err.is_err(), "expected panic for shard_count={}", bad);
    }

    #[test]
    fn new_rejects_zero_workers() {
        let err = std::panic::catch_unwind(|| Topology::new(SHARD_COUNT, 0));
        assert!(err.is_err(), "expected panic for workers_per_shard=0");
    }
}
