//! Criterion benchmarks for QMDB hash operations.
//!
//! Covers:
//!   - Single SHA256 hash throughput (variable-length, 65-byte node)
//!   - Batch CPU node hashing at various batch sizes
//!   - GPU AoS vs SoA kernel comparison (when cuda feature enabled)
//!
//! Run with:
//!   cargo bench -p qmdb --bench hash_benchmarks
//!   cargo bench -p qmdb --bench hash_benchmarks --features cuda  # includes GPU benchmarks

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use sha2::{Digest, Sha256};

// ============================================================
// Helpers
// ============================================================

fn make_levels(n: usize) -> Vec<u8> {
    (0..n).map(|i| (i % 64) as u8).collect()
}

fn make_hashes(n: usize, seed: u8) -> Vec<[u8; 32]> {
    (0..n)
        .map(|i| {
            let mut h = [0u8; 32];
            for j in 0..32 {
                h[j] = ((i * 7 + j * 13) as u8).wrapping_add(seed);
            }
            h
        })
        .collect()
}

// ============================================================
// CPU Hash Benchmarks
// ============================================================

fn bench_single_hash(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_hash");

    // Variable-length hash at different input sizes
    for size in [32, 65, 128, 256, 512] {
        let data = vec![0xABu8; size];
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::new("sha256", size), &data, |b, data| {
            b.iter(|| {
                let mut hasher = Sha256::new();
                hasher.update(black_box(data));
                let _: [u8; 32] = hasher.finalize().into();
            })
        });
    }

    // Node hash (65-byte: level + left + right)
    let level = 5u8;
    let left = [0xAA; 32];
    let right = [0xBB; 32];
    group.throughput(Throughput::Bytes(65));
    group.bench_function("node_hash_65B", |b| {
        b.iter(|| {
            qmdb::utils::hasher::hash2(
                black_box(level),
                black_box(&left),
                black_box(&right),
            )
        })
    });

    group.finish();
}

fn bench_batch_cpu_hash(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_cpu_hash");

    for &n in &[10, 100, 1_000, 10_000, 50_000] {
        let levels = make_levels(n);
        let lefts = make_hashes(n, 0x11);
        let rights = make_hashes(n, 0xAB);

        group.throughput(Throughput::Elements(n as u64));

        // batch_node_hash_cpu (P4 optimized path with finalize_reset)
        group.bench_with_input(
            BenchmarkId::new("batch_node_hash_cpu", n),
            &n,
            |b, _| {
                let mut out = vec![[0u8; 32]; n];
                b.iter(|| {
                    qmdb::utils::hasher::batch_node_hash_cpu(
                        black_box(&levels),
                        black_box(&lefts),
                        black_box(&rights),
                        black_box(&mut out),
                    );
                })
            },
        );

        // Individual hash2 calls (baseline for comparison)
        group.bench_with_input(
            BenchmarkId::new("individual_hash2", n),
            &n,
            |b, _| {
                b.iter(|| {
                    for i in 0..n {
                        black_box(qmdb::utils::hasher::hash2(
                            levels[i],
                            &lefts[i],
                            &rights[i],
                        ));
                    }
                })
            },
        );
    }

    group.finish();
}

fn bench_node_hash_inplace(c: &mut Criterion) {
    let mut group = c.benchmark_group("node_hash_inplace");

    let level = 5u8;
    let left = [0xAA; 32];
    let right = [0xBB; 32];
    let mut target = [0u8; 32];

    group.throughput(Throughput::Bytes(65));
    group.bench_function("inplace_65B", |b| {
        b.iter(|| {
            qmdb::utils::hasher::node_hash_inplace(
                black_box(level),
                black_box(&mut target),
                black_box(&left),
                black_box(&right),
            );
        })
    });

    group.finish();
}

fn bench_variable_hash(c: &mut Criterion) {
    let mut group = c.benchmark_group("variable_hash");

    for &size in &[50, 100, 200, 300] {
        let data = vec![0x42u8; size];
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(
            BenchmarkId::new("entry_hash", size),
            &data,
            |b, data| {
                b.iter(|| qmdb::utils::hasher::hash(black_box(data)))
            },
        );
    }

    group.finish();
}

// ============================================================
// GPU Hash Benchmarks (cuda feature only)
// ============================================================

#[cfg(feature = "cuda")]
fn bench_gpu_node_hash(c: &mut Criterion) {
    use qmdb::gpu::{GpuHasher, NodeHashJob};

    let gpu = match GpuHasher::new(200_000) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Skipping GPU benchmarks (no CUDA): {}", e);
            return;
        }
    };

    let mut group = c.benchmark_group("gpu_node_hash");

    for &n in &[1_000, 10_000, 50_000, 100_000, 200_000] {
        let levels = make_levels(n);
        let lefts = make_hashes(n, 0x11);
        let rights = make_hashes(n, 0xAB);

        let jobs: Vec<NodeHashJob> = (0..n)
            .map(|i| NodeHashJob {
                level: levels[i],
                left: lefts[i],
                right: rights[i],
            })
            .collect();

        group.throughput(Throughput::Elements(n as u64));

        // AoS kernel
        group.bench_with_input(BenchmarkId::new("aos", n), &n, |b, _| {
            b.iter(|| gpu.batch_node_hash(black_box(&jobs)))
        });

        // SoA kernel
        group.bench_with_input(BenchmarkId::new("soa", n), &n, |b, _| {
            b.iter(|| {
                gpu.batch_node_hash_soa(
                    black_box(&levels),
                    black_box(&lefts),
                    black_box(&rights),
                )
            })
        });

        // Warp-cooperative kernel
        group.bench_with_input(BenchmarkId::new("warp_coop", n), &n, |b, _| {
            b.iter(|| gpu.batch_node_hash_warp_coop(black_box(&jobs)))
        });
    }

    group.finish();
}

#[cfg(feature = "cuda")]
fn bench_gpu_variable_hash(c: &mut Criterion) {
    use qmdb::gpu::GpuHasher;

    let gpu = match GpuHasher::new(200_000) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Skipping GPU variable hash benchmarks: {}", e);
            return;
        }
    };

    let mut group = c.benchmark_group("gpu_variable_hash");

    for &n in &[1_000, 10_000, 50_000] {
        let data: Vec<Vec<u8>> = (0..n)
            .map(|i| vec![(i & 0xFF) as u8; 100 + (i % 200)])
            .collect();
        let inputs: Vec<&[u8]> = data.iter().map(|v| v.as_slice()).collect();

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("variable_100_300B", n), &n, |b, _| {
            b.iter(|| gpu.batch_hash_variable(black_box(&inputs)))
        });
    }

    group.finish();
}

#[cfg(feature = "cuda")]
fn bench_gpu_vs_cpu(c: &mut Criterion) {
    use qmdb::gpu::{GpuHasher, NodeHashJob};

    let gpu = match GpuHasher::new(200_000) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Skipping GPU vs CPU benchmarks: {}", e);
            return;
        }
    };

    let mut group = c.benchmark_group("gpu_vs_cpu");

    for &n in &[1_000, 10_000, 50_000] {
        let levels = make_levels(n);
        let lefts = make_hashes(n, 0x11);
        let rights = make_hashes(n, 0xAB);

        let jobs: Vec<NodeHashJob> = (0..n)
            .map(|i| NodeHashJob {
                level: levels[i],
                left: lefts[i],
                right: rights[i],
            })
            .collect();

        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(BenchmarkId::new("cpu_batch", n), &n, |b, _| {
            let mut out = vec![[0u8; 32]; n];
            b.iter(|| {
                qmdb::utils::hasher::batch_node_hash_cpu(
                    black_box(&levels),
                    black_box(&lefts),
                    black_box(&rights),
                    black_box(&mut out),
                );
            })
        });

        group.bench_with_input(BenchmarkId::new("gpu_aos", n), &n, |b, _| {
            b.iter(|| gpu.batch_node_hash(black_box(&jobs)))
        });

        group.bench_with_input(BenchmarkId::new("gpu_soa", n), &n, |b, _| {
            b.iter(|| {
                gpu.batch_node_hash_soa(
                    black_box(&levels),
                    black_box(&lefts),
                    black_box(&rights),
                )
            })
        });
    }

    group.finish();
}

// ============================================================
// Criterion Groups
// ============================================================

#[cfg(not(feature = "cuda"))]
criterion_group!(
    benches,
    bench_single_hash,
    bench_batch_cpu_hash,
    bench_node_hash_inplace,
    bench_variable_hash,
);

#[cfg(feature = "cuda")]
criterion_group!(
    benches,
    bench_single_hash,
    bench_batch_cpu_hash,
    bench_node_hash_inplace,
    bench_variable_hash,
    bench_gpu_node_hash,
    bench_gpu_variable_hash,
    bench_gpu_vs_cpu,
);

criterion_main!(benches);
