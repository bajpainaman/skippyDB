//! Parity test for `UpperTree::sync_upper_nodes_gpu_resident` vs
//! `UpperTree::sync_upper_nodes_gpu`.
//!
//! Phase 0 second-capture A/B benching showed the per-level GPU path is ~3.4×
//! faster than the resident path at 40M cuda. Before flipping the production
//! default, both paths must produce byte-identical Merkle roots from the same
//! deterministic input — otherwise we ship a silent consensus break.
//!
//! The bench harness is non-deterministic (resident-vs-resident only matched
//! 1.5% of (shard, height) tuples across two runs), so this test drives both
//! paths from a fixed in-memory entry sequence built via `build_test_tree`.
//!
//! ## History
//!
//! 2026-04-26: introduced as `#[ignore]` known-failing — the resident path
//! diverged from per-level on this fixed input. Root cause: the resident
//! path's populate phase (tree.rs:614-619) only uploaded twig roots from
//! `active_twig_shards`, but `sync_level_on_device`'s `bulk_get_device`
//! discards the "found" flag, so missing twig positions returned garbage.
//! The per-level path falls back to `NULL_TWIG.twig_root` for missing
//! twigs (tree.rs:521-527).
//!
//! Fix: pre-fill `NULL_TWIG.twig_root` at every (2*i, 2*i+1) twig position
//! in `n_list` before uploading active twig roots, so missing twigs return
//! the correct sentinel. Both tests pass after the fix; kept enabled so a
//! future regression in either path is caught immediately.

#![cfg(feature = "cuda")]

use serial_test::serial;
use skippydb::gpu::{GpuHasher, GpuNodeStore};
use skippydb::merkletree::helpers::build_test_tree;
use skippydb::test_helper::TempDir;

const COUNT_BEFORE: i32 = 2000;
const COUNT_AFTER: i32 = 100;

fn run_parity(deact_sn_list: Vec<u64>, dir_a: &str, dir_b: &str, label: &str) {
    let _t_a = TempDir::new(dir_a);
    let _t_b = TempDir::new(dir_b);

    let (mut tree_a, _, _, _) =
        build_test_tree(dir_a, &deact_sn_list, COUNT_BEFORE, COUNT_AFTER);
    let (mut tree_b, _, _, _) =
        build_test_tree(dir_b, &deact_sn_list, COUNT_BEFORE, COUNT_AFTER);

    let gpu = GpuHasher::new(200_000).expect("GpuHasher::new");

    let n_list_a = tree_a.flush_files_gpu(&gpu, 0, 0);
    let n_list_b = tree_b.flush_files_gpu(&gpu, 0, 0);
    assert_eq!(n_list_a, n_list_b, "[{label}] flush_files_gpu n_list diverged");

    let ytwig_a = tree_a.youngest_twig_id;
    let ytwig_b = tree_b.youngest_twig_id;
    assert_eq!(ytwig_a, ytwig_b, "[{label}] youngest_twig_id diverged");

    let mut store =
        GpuNodeStore::new().expect("GpuNodeStore::new (cuda must be available)");

    let (_, root_resident) = tree_a.upper_tree.sync_upper_nodes_gpu_resident(
        &gpu, &mut store, n_list_a, ytwig_a,
    );
    let (_, root_perlevel) =
        tree_b.upper_tree.sync_upper_nodes_gpu(&gpu, n_list_b, ytwig_b);

    eprintln!(
        "[{label}] per-level = {}\n[{label}] resident  = {}",
        hex(&root_perlevel),
        hex(&root_resident),
    );

    assert_eq!(
        root_resident, root_perlevel,
        "[{label}] ROOT MISMATCH:\n  resident   = {}\n  per-level  = {}",
        hex(&root_resident),
        hex(&root_perlevel),
    );
}

fn hex(b: &[u8; 32]) -> String {
    let mut s = String::with_capacity(64);
    for byte in b {
        use std::fmt::Write;
        let _ = write!(s, "{:02x}", byte);
    }
    s
}

#[test]
#[serial]
fn sync_upper_nodes_parity_no_deactivations() {
    run_parity(
        vec![],
        "./DataTree-parity-A-nodeact",
        "./DataTree-parity-B-nodeact",
        "no_deactivations",
    );
}

#[test]
#[serial]
fn sync_upper_nodes_parity_with_deactivations() {
    run_parity(
        vec![5, 10, 15, 20, 100, 500, 1000],
        "./DataTree-parity-A-deact",
        "./DataTree-parity-B-deact",
        "with_deactivations",
    );
}
