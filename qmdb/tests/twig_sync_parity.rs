//! Minimal parity check for `sync_mtree` (CPU) vs `sync_mtrees_gpu`.
//!
//! Tests with TINY input (2 leaves, single twig) to isolate the divergence
//! observed in `gpu::integration_tests::test_single_entry_cpu_vs_gpu`. At
//! these batch sizes `auto_batch_node_hash` falls back to CPU, so if both
//! paths still diverge, the bug is structural (e.g. mtree initialization,
//! null-leaf handling, scatter ordering).

#![cfg(feature = "cuda")]

use serial_test::serial;
use skippydb::def::LEAF_COUNT_IN_TWIG;
use skippydb::gpu::GpuHasher;
use skippydb::merkletree::twig::{self, NULL_MT_FOR_TWIG};
use skippydb::utils::hasher::Hash32;

type TwigMT = [Hash32; LEAF_COUNT_IN_TWIG as usize * 2];

fn fresh_mtree() -> Box<TwigMT> {
    let mut mt = Box::new([[0u8; 32]; LEAF_COUNT_IN_TWIG as usize * 2]);
    mt.copy_from_slice(NULL_MT_FOR_TWIG.as_ref());
    mt
}

fn write_leaf(mt: &mut TwigMT, position: usize, content: u8) {
    let mut h = [0u8; 32];
    for k in 0..32 {
        h[k] = content.wrapping_add(k as u8);
    }
    mt[LEAF_COUNT_IN_TWIG as usize + position].copy_from_slice(&h);
}

fn first_diff(a: &TwigMT, b: &TwigMT) -> Option<usize> {
    for i in 0..a.len() {
        if a[i] != b[i] {
            return Some(i);
        }
    }
    None
}

#[test]
#[serial]
fn sync_mtree_two_leaves_cpu_vs_gpu() {
    let mut mt_cpu = fresh_mtree();
    let mut mt_gpu = fresh_mtree();

    write_leaf(&mut mt_cpu, 0, 0x11);
    write_leaf(&mut mt_cpu, 1, 0x22);
    write_leaf(&mut mt_gpu, 0, 0x11);
    write_leaf(&mut mt_gpu, 1, 0x22);

    // Pre-sync sanity: both initial mtrees must be byte-identical.
    assert_eq!(
        first_diff(&mt_cpu, &mt_gpu),
        None,
        "pre-sync mtrees diverged"
    );

    // Run CPU
    twig::sync_mtree(&mut mt_cpu[..], 0, 1);

    // Run GPU
    let gpu = GpuHasher::new(200_000).expect("GpuHasher::new");
    twig::sync_mtrees_gpu(&gpu, &mut [(&mut mt_gpu[..], 0, 1)]);

    if let Some(idx) = first_diff(&mt_cpu, &mt_gpu) {
        let mut hex_cpu = String::new();
        let mut hex_gpu = String::new();
        for b in &mt_cpu[idx] {
            use std::fmt::Write;
            let _ = write!(hex_cpu, "{:02x}", b);
        }
        for b in &mt_gpu[idx] {
            use std::fmt::Write;
            let _ = write!(hex_gpu, "{:02x}", b);
        }
        panic!(
            "post-sync DIVERGE at mtree[{idx}]:\n  cpu = {hex_cpu}\n  gpu = {hex_gpu}"
        );
    }
}

#[test]
#[serial]
fn sync_mtree_one_leaf_cpu_vs_gpu() {
    let mut mt_cpu = fresh_mtree();
    let mut mt_gpu = fresh_mtree();

    write_leaf(&mut mt_cpu, 0, 0xAA);
    write_leaf(&mut mt_gpu, 0, 0xAA);

    twig::sync_mtree(&mut mt_cpu[..], 0, 0);

    let gpu = GpuHasher::new(200_000).expect("GpuHasher::new");
    twig::sync_mtrees_gpu(&gpu, &mut [(&mut mt_gpu[..], 0, 0)]);

    if let Some(idx) = first_diff(&mt_cpu, &mt_gpu) {
        let mut hex_cpu = String::new();
        let mut hex_gpu = String::new();
        for b in &mt_cpu[idx] {
            use std::fmt::Write;
            let _ = write!(hex_cpu, "{:02x}", b);
        }
        for b in &mt_gpu[idx] {
            use std::fmt::Write;
            let _ = write!(hex_gpu, "{:02x}", b);
        }
        panic!(
            "post-sync DIVERGE at mtree[{idx}]:\n  cpu = {hex_cpu}\n  gpu = {hex_gpu}"
        );
    }
}
