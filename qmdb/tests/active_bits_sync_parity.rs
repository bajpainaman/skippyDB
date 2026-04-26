//! Parity test for the active-bits sync pipeline (`sync_mt_for_active_bits_phase1`
//! + `sync_mt_for_active_bits_phase2`) — CPU vs GPU.
//!
//! The 10 pre-existing `*_cpu_vs_gpu` tests in `gpu::integration_tests` panic
//! at `check.rs:31` "L1-0/L2-0 Not Equal" — but inspection showed those tests
//! are sloppy: they never run phase2 (`sync_l2`/`sync_l3`/`sync_top`) before
//! `check_hash_consistency` which validates `twig.active_bits_mtl2[0]` against
//! `hash2(9, mtl1[0], mtl1[1])`. They produce CPU-vs-GPU divergence reports
//! that just reflect both paths having un-synced state.
//!
//! This test runs the FULL pipeline (phase1 then phase2) on both CPU and GPU
//! over identical input, then asserts byte-equal twig fields:
//! `active_bits_mtl1[0..4]`, `active_bits_mtl2[0..2]`, `active_bits_mtl3`,
//! `twig_root`. If any field diverges, that's a real CPU-vs-GPU bug in the
//! active-bits Merkle path. If all match, the 10 pre-existing failures are
//! confirmed test-bugs and the GPU active-bits path is correct.

#![cfg(feature = "cuda")]

use serial_test::serial;
use skippydb::gpu::GpuHasher;
use skippydb::merkletree::helpers::build_test_tree;
use skippydb::test_helper::TempDir;

const COUNT_BEFORE: i32 = 50;
const COUNT_AFTER: i32 = 50;

fn run_pipeline_parity(
    deact_sn_list: Vec<u64>,
    dir_cpu: &str,
    dir_gpu: &str,
    label: &str,
) {
    let _t_cpu = TempDir::new(dir_cpu);
    let _t_gpu = TempDir::new(dir_gpu);

    let (mut tree_cpu, _, _, _) =
        build_test_tree(dir_cpu, &deact_sn_list, COUNT_BEFORE, COUNT_AFTER);
    let (mut tree_gpu, _, _, _) =
        build_test_tree(dir_gpu, &deact_sn_list, COUNT_BEFORE, COUNT_AFTER);

    let gpu = GpuHasher::new(200_000).expect("GpuHasher::new");

    // Phase 1 (sync L1)
    let n_list_cpu = tree_cpu.sync_mt_for_active_bits_phase1();
    let n_list_gpu = tree_gpu.sync_mt_for_active_bits_phase1_gpu(&gpu);
    assert_eq!(
        n_list_cpu, n_list_gpu,
        "[{label}] phase1 n_list diverged"
    );

    // Phase 2 (sync L2, L3, top)
    let _ = tree_cpu.upper_tree.sync_mt_for_active_bits_phase2(n_list_cpu);
    let _ = tree_gpu.upper_tree.sync_mt_for_active_bits_phase2_gpu(&gpu, n_list_gpu);

    // Compare every active twig's full state, byte-for-byte.
    for shard_idx in 0..tree_cpu.upper_tree.active_twig_shards.len() {
        let twigs_cpu = &tree_cpu.upper_tree.active_twig_shards[shard_idx];
        let twigs_gpu = &tree_gpu.upper_tree.active_twig_shards[shard_idx];
        assert_eq!(
            twigs_cpu.len(),
            twigs_gpu.len(),
            "[{label}] active_twig_shards[{shard_idx}] count diverged"
        );
        for (twig_id, twig_cpu) in twigs_cpu {
            let twig_gpu = twigs_gpu
                .get(twig_id)
                .unwrap_or_else(|| panic!("[{label}] twig {twig_id} missing from GPU side"));

            for i in 0..4 {
                assert_eq!(
                    twig_cpu.active_bits_mtl1[i], twig_gpu.active_bits_mtl1[i],
                    "[{label}] twig {twig_id} mtl1[{i}] diverged: cpu={} gpu={}",
                    hex(&twig_cpu.active_bits_mtl1[i]),
                    hex(&twig_gpu.active_bits_mtl1[i]),
                );
            }
            for i in 0..2 {
                assert_eq!(
                    twig_cpu.active_bits_mtl2[i], twig_gpu.active_bits_mtl2[i],
                    "[{label}] twig {twig_id} mtl2[{i}] diverged: cpu={} gpu={}",
                    hex(&twig_cpu.active_bits_mtl2[i]),
                    hex(&twig_gpu.active_bits_mtl2[i]),
                );
            }
            assert_eq!(
                twig_cpu.active_bits_mtl3, twig_gpu.active_bits_mtl3,
                "[{label}] twig {twig_id} mtl3 diverged: cpu={} gpu={}",
                hex(&twig_cpu.active_bits_mtl3),
                hex(&twig_gpu.active_bits_mtl3),
            );
            assert_eq!(
                twig_cpu.twig_root, twig_gpu.twig_root,
                "[{label}] twig {twig_id} twig_root diverged: cpu={} gpu={}",
                hex(&twig_cpu.twig_root),
                hex(&twig_gpu.twig_root),
            );
        }
    }
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
fn active_bits_pipeline_no_deactivations() {
    run_pipeline_parity(
        vec![],
        "./DataTree-actbits-A-nodeact",
        "./DataTree-actbits-B-nodeact",
        "no_deactivations",
    );
}

#[test]
#[serial]
fn active_bits_pipeline_with_deactivations() {
    run_pipeline_parity(
        vec![5, 10, 15, 20],
        "./DataTree-actbits-A-deact",
        "./DataTree-actbits-B-deact",
        "with_deactivations",
    );
}
