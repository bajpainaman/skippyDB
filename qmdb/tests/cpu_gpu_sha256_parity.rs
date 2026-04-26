//! Direct parity check: do `batch_node_hash_cpu` and the GPU SoA kernel
//! produce identical SHA256(level || left || right) for the same inputs?
//!
//! If this test passes for all batch sizes, the resident-vs-per-level
//! divergence is upstream in the populate/boundary logic. If it fails,
//! the GPU SHA256 kernel itself is wrong, which would also explain the
//! ~10 pre-existing `*_cpu_vs_gpu` test failures.

#![cfg(feature = "cuda")]

use serial_test::serial;
use skippydb::gpu::GpuHasher;
use skippydb::utils::hasher::batch_node_hash_cpu;

fn deterministic_inputs(n: usize, seed: u64) -> (Vec<u8>, Vec<[u8; 32]>, Vec<[u8; 32]>) {
    let mut levels = Vec::with_capacity(n);
    let mut lefts = Vec::with_capacity(n);
    let mut rights = Vec::with_capacity(n);
    let mut s = seed;
    for i in 0..n {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        levels.push((i % 50) as u8);
        let mut l = [0u8; 32];
        let mut r = [0u8; 32];
        for k in 0..32 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            l[k] = (s >> (k as u64 * 8 % 56)) as u8;
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            r[k] = (s >> (k as u64 * 8 % 56)) as u8;
        }
        lefts.push(l);
        rights.push(r);
    }
    (levels, lefts, rights)
}

fn run_for_size(n: usize) {
    let (levels, lefts, rights) = deterministic_inputs(n, 0xdeadbeef);

    let mut cpu_out = vec![[0u8; 32]; n];
    batch_node_hash_cpu(&levels, &lefts, &rights, &mut cpu_out);

    let gpu = GpuHasher::new(200_000).expect("GpuHasher::new");
    let gpu_out = gpu.batch_node_hash_soa(&levels, &lefts, &rights);

    assert_eq!(cpu_out.len(), gpu_out.len(), "[n={n}] length mismatch");
    let mut mismatches = 0usize;
    for i in 0..n {
        if cpu_out[i] != gpu_out[i] {
            if mismatches < 3 {
                eprintln!(
                    "[n={n}] mismatch at i={i} level={} cpu={} gpu={}",
                    levels[i],
                    hex(&cpu_out[i]),
                    hex(&gpu_out[i]),
                );
            }
            mismatches += 1;
        }
    }
    assert_eq!(mismatches, 0, "[n={n}] {mismatches}/{n} entries diverged between CPU and GPU SoA SHA256");
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
fn cpu_vs_gpu_soa_n_2() {
    run_for_size(2);
}

#[test]
#[serial]
fn cpu_vs_gpu_soa_n_64() {
    run_for_size(64);
}

#[test]
#[serial]
fn cpu_vs_gpu_soa_n_1024() {
    run_for_size(1024);
}

#[test]
#[serial]
fn cpu_vs_gpu_soa_n_4096() {
    run_for_size(4096);
}
