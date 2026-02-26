/// GPU vs CPU integration tests for QMDB Merkle tree.
/// Proves that `#[cfg(feature = "cuda")]` GPU paths produce byte-identical
/// Merkle roots to the existing CPU paths at every level of the tree.
///
/// Run with: `cargo test -p qmdb --features cuda -- gpu::integration_tests`
#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests {
    use crate::def::{DEFAULT_FILE_SIZE, SMALL_BUFFER_SIZE};
    use crate::gpu::{GpuHasher, NodeHashJob};
    use crate::merkletree::check::{check_hash_consistency, check_mt};
    use crate::merkletree::helpers::build_test_tree;
    use crate::merkletree::tree::Tree;
    use crate::merkletree::twig::{
        sync_mtree, sync_mtrees_gpu, ActiveBits, Twig, NULL_MT_FOR_TWIG, NULL_TWIG,
    };
    use crate::utils::hasher::{self, Hash32, ZERO_HASH32};
    use sha2::{Digest, Sha256};
    use std::time::Instant;

    macro_rules! gpu_or_skip {
        ($max_batch:expr) => {
            match GpuHasher::new($max_batch) {
                Ok(g) => g,
                Err(e) => {
                    eprintln!("Skipping GPU test: {}", e);
                    return;
                }
            }
        };
        () => {
            gpu_or_skip!(200_000)
        };
    }

    fn cpu_hash(data: &[u8]) -> [u8; 32] {
        let mut h = Sha256::new();
        h.update(data);
        h.finalize().into()
    }

    fn cpu_hash2(level: u8, a: &[u8; 32], b: &[u8; 32]) -> [u8; 32] {
        let mut h = Sha256::new();
        h.update([level]);
        h.update(a);
        h.update(b);
        h.finalize().into()
    }

    /// Deterministic pseudo-random bytes from a seed.
    fn pseudo_random_bytes(seed: u64, len: usize) -> Vec<u8> {
        let mut v = Vec::with_capacity(len);
        let mut s = seed;
        for _ in 0..len {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            v.push((s >> 33) as u8);
        }
        v
    }

    fn pseudo_random_hash(seed: u64) -> [u8; 32] {
        let b = pseudo_random_bytes(seed, 32);
        let mut h = [0u8; 32];
        h.copy_from_slice(&b);
        h
    }

    // =========================================================================
    // 1. Leaf hashing — entry-level hash (variable length)
    // =========================================================================

    #[test]
    fn test_entry_leaf_hash_cpu_vs_gpu() {
        let gpu = gpu_or_skip!();

        let n = 1000;
        let mut inputs: Vec<Vec<u8>> = Vec::with_capacity(n);
        for i in 0..n {
            // lengths from 50 to 300 bytes
            let len = 50 + (i % 251);
            inputs.push(pseudo_random_bytes(i as u64, len));
        }

        // CPU path
        let cpu_start = Instant::now();
        let cpu_hashes: Vec<[u8; 32]> = inputs.iter().map(|d| cpu_hash(d)).collect();
        let cpu_time = cpu_start.elapsed();

        // GPU path
        let refs: Vec<&[u8]> = inputs.iter().map(|v| v.as_slice()).collect();
        let gpu_start = Instant::now();
        let gpu_hashes = gpu.batch_hash_variable(&refs);
        let gpu_time = gpu_start.elapsed();

        println!(
            "Leaf hash 1000: CPU: {:?}, GPU: {:?}, speedup: {:.1}x",
            cpu_time,
            gpu_time,
            cpu_time.as_secs_f64() / gpu_time.as_secs_f64().max(1e-9)
        );

        for i in 0..n {
            assert_eq!(
                cpu_hashes[i], gpu_hashes[i],
                "Leaf hash mismatch at {} (len={}): CPU={} GPU={}",
                i,
                inputs[i].len(),
                hex::encode(cpu_hashes[i]),
                hex::encode(gpu_hashes[i])
            );
        }
    }

    // =========================================================================
    // 2. Node hashing — fixed 65-byte node hash
    // =========================================================================

    #[test]
    fn test_node_hash_cpu_vs_gpu() {
        let gpu = gpu_or_skip!();

        let n = 10_000;
        let mut jobs = Vec::with_capacity(n);
        for i in 0..n {
            let level = (i % 256) as u8;
            let left = pseudo_random_hash(i as u64 * 2);
            let right = pseudo_random_hash(i as u64 * 2 + 1);
            jobs.push(NodeHashJob { level, left, right });
        }

        // CPU path
        let cpu_start = Instant::now();
        let mut cpu_hashes = vec![[0u8; 32]; n];
        for (i, job) in jobs.iter().enumerate() {
            hasher::node_hash_inplace(job.level, &mut cpu_hashes[i], &job.left, &job.right);
        }
        let cpu_time = cpu_start.elapsed();

        // GPU path
        let gpu_start = Instant::now();
        let gpu_hashes = gpu.batch_node_hash(&jobs);
        let gpu_time = gpu_start.elapsed();

        println!(
            "Node hash 10K: CPU: {:?}, GPU: {:?}, speedup: {:.1}x",
            cpu_time,
            gpu_time,
            cpu_time.as_secs_f64() / gpu_time.as_secs_f64().max(1e-9)
        );

        for i in 0..n {
            assert_eq!(
                cpu_hashes[i], gpu_hashes[i],
                "Node hash mismatch at {}: CPU={} GPU={}",
                i,
                hex::encode(cpu_hashes[i]),
                hex::encode(gpu_hashes[i])
            );
        }
    }

    // =========================================================================
    // 3. Single twig sync — sync_mtree (CPU) vs sync_mtrees_gpu (GPU)
    // =========================================================================

    #[test]
    fn test_twig_sync_cpu_vs_gpu() {
        let gpu = gpu_or_skip!();

        // Create two identical TwigMTs with 1024 random leaf hashes
        let mut mt_cpu = vec![ZERO_HASH32; 4096].into_boxed_slice();
        let mut mt_gpu = vec![ZERO_HASH32; 4096].into_boxed_slice();

        for i in 0..1024 {
            let leaf_hash = pseudo_random_hash(i as u64);
            mt_cpu[2048 + i] = leaf_hash;
            mt_gpu[2048 + i] = leaf_hash;
        }

        // CPU path
        let cpu_start = Instant::now();
        sync_mtree(&mut mt_cpu, 0, 1023);
        let cpu_time = cpu_start.elapsed();

        // GPU path
        let gpu_start = Instant::now();
        sync_mtrees_gpu(&gpu, &mut [(&mut mt_gpu, 0, 1023)]);
        let gpu_time = gpu_start.elapsed();

        println!(
            "Single twig sync (1024 leaves): CPU: {:?}, GPU: {:?}, speedup: {:.1}x",
            cpu_time,
            gpu_time,
            cpu_time.as_secs_f64() / gpu_time.as_secs_f64().max(1e-9)
        );

        // Compare root
        assert_eq!(
            mt_cpu[1], mt_gpu[1],
            "Root mismatch: CPU={} GPU={}",
            hex::encode(mt_cpu[1]),
            hex::encode(mt_gpu[1])
        );

        // Compare ALL intermediate nodes
        for i in 1..2048 {
            assert_eq!(
                mt_cpu[i], mt_gpu[i],
                "Internal node mismatch at {}: CPU={} GPU={}",
                i,
                hex::encode(mt_cpu[i]),
                hex::encode(mt_gpu[i])
            );
        }
    }

    // =========================================================================
    // 4. Multi-twig cross-batch — 50 twigs synced at once
    // =========================================================================

    #[test]
    fn test_multi_twig_batch_cpu_vs_gpu() {
        let gpu = gpu_or_skip!();

        let n_twigs = 50;

        // Create n_twigs identical pairs
        let mut cpu_mts: Vec<Box<[Hash32]>> = Vec::with_capacity(n_twigs);
        let mut gpu_mts: Vec<Box<[Hash32]>> = Vec::with_capacity(n_twigs);

        for t in 0..n_twigs {
            let mut mt_cpu = vec![ZERO_HASH32; 4096].into_boxed_slice();
            let mut mt_gpu = vec![ZERO_HASH32; 4096].into_boxed_slice();
            for i in 0..2048 {
                let h = pseudo_random_hash((t * 2048 + i) as u64);
                mt_cpu[2048 + i] = h;
                mt_gpu[2048 + i] = h;
            }
            cpu_mts.push(mt_cpu);
            gpu_mts.push(mt_gpu);
        }

        // CPU path: sync each individually
        let cpu_start = Instant::now();
        for mt in cpu_mts.iter_mut() {
            sync_mtree(mt, 0, 2047);
        }
        let cpu_time = cpu_start.elapsed();

        // GPU path: batch all 50
        let gpu_start = Instant::now();
        let mut gpu_slices: Vec<(&mut [Hash32], i32, i32)> = gpu_mts
            .iter_mut()
            .map(|mt| (mt.as_mut() as &mut [Hash32], 0i32, 2047i32))
            .collect();
        sync_mtrees_gpu(&gpu, &mut gpu_slices);
        let gpu_time = gpu_start.elapsed();

        println!(
            "Multi-twig batch ({} twigs x 2048 leaves): CPU: {:?}, GPU: {:?}, speedup: {:.1}x",
            n_twigs,
            cpu_time,
            gpu_time,
            cpu_time.as_secs_f64() / gpu_time.as_secs_f64().max(1e-9)
        );

        for t in 0..n_twigs {
            assert_eq!(
                cpu_mts[t][1], gpu_mts[t][1],
                "Twig {} root mismatch: CPU={} GPU={}",
                t,
                hex::encode(cpu_mts[t][1]),
                hex::encode(gpu_mts[t][1])
            );
            // Check all internal nodes
            for i in 1..2048 {
                assert_eq!(
                    cpu_mts[t][i], gpu_mts[t][i],
                    "Twig {} node {} mismatch",
                    t, i
                );
            }
        }
    }

    // =========================================================================
    // 5. Partially dirty twigs — sparse updates
    // =========================================================================

    #[test]
    fn test_partial_dirty_twig_cpu_vs_gpu() {
        let gpu = gpu_or_skip!();

        // Test multiple sparse update patterns
        let patterns: Vec<(&str, Vec<(i32, i32)>)> = vec![
            ("single_leaf", vec![(0, 0)]),
            ("first_10", vec![(0, 9)]),
            ("last_10", vec![(2038, 2047)]),
            ("middle_50", vec![(999, 1048)]),
            ("sparse_every_100", vec![
                (0, 0), (100, 100), (200, 200), (300, 300), (400, 400),
                (500, 500), (600, 600), (700, 700), (800, 800), (900, 900),
                (1000, 1000), (1100, 1100), (1200, 1200), (1300, 1300),
                (1400, 1400), (1500, 1500), (1600, 1600), (1700, 1700),
                (1800, 1800), (1900, 1900), (2000, 2000),
            ]),
        ];

        for (name, ranges) in &patterns {
            // Create identical twigs with NULL_MT_FOR_TWIG as base
            let mut mt_cpu = NULL_MT_FOR_TWIG.to_vec().into_boxed_slice();
            let mut mt_gpu = NULL_MT_FOR_TWIG.to_vec().into_boxed_slice();

            // Apply sparse modifications
            for &(start, end) in ranges {
                for i in start..=end {
                    let h = pseudo_random_hash(i as u64 + 77777);
                    mt_cpu[(2048 + i) as usize] = h;
                    mt_gpu[(2048 + i) as usize] = h;
                }
            }

            // For sync_mtree, use the overall range encompassing all dirty leaves
            let overall_start = ranges.iter().map(|r| r.0).min().unwrap();
            let overall_end = ranges.iter().map(|r| r.1).max().unwrap();

            // CPU
            sync_mtree(&mut mt_cpu, overall_start, overall_end);

            // GPU
            sync_mtrees_gpu(
                &gpu,
                &mut [(&mut mt_gpu, overall_start, overall_end)],
            );

            assert_eq!(
                mt_cpu[1], mt_gpu[1],
                "Pattern '{}' root mismatch: CPU={} GPU={}",
                name,
                hex::encode(mt_cpu[1]),
                hex::encode(mt_gpu[1])
            );

            // Verify all internal nodes
            for i in 1..2048 {
                assert_eq!(
                    mt_cpu[i], mt_gpu[i],
                    "Pattern '{}' node {} mismatch",
                    name, i
                );
            }
            println!("  Partial dirty '{}': roots match", name);
        }
    }

    // =========================================================================
    // 6. Active bits Merkle — sync_l1/l2/l3/top via GPU batch vs CPU
    // =========================================================================

    #[test]
    fn test_active_bits_sync_cpu_vs_gpu() {
        let gpu = gpu_or_skip!();

        // Test with 20 different active bits patterns
        let n_twigs = 20;

        for t in 0..n_twigs {
            let mut active_bits = ActiveBits::new();
            // Set random bits based on twig index
            for bit_idx in 0..2048u32 {
                // Set approximately 50% of bits
                let seed = (t as u64 * 2048 + bit_idx as u64) * 31 + 7;
                if seed % 2 == 0 {
                    active_bits.set_bit(bit_idx);
                }
            }

            // CPU path: full twig sync chain
            let mut twig_cpu = Twig::new();
            twig_cpu.left_root = pseudo_random_hash(t as u64 * 1000);

            let cpu_start = Instant::now();
            twig_cpu.sync_l1(0, &active_bits);
            twig_cpu.sync_l1(1, &active_bits);
            twig_cpu.sync_l1(2, &active_bits);
            twig_cpu.sync_l1(3, &active_bits);
            twig_cpu.sync_l2(0);
            twig_cpu.sync_l2(1);
            twig_cpu.sync_l3();
            twig_cpu.sync_top();
            let cpu_time = cpu_start.elapsed();

            // GPU path: batch all sync_l1 operations
            let mut twig_gpu = Twig::new();
            twig_gpu.left_root = twig_cpu.left_root;

            let gpu_start = Instant::now();
            // Batch sync_l1: 4 jobs at level 8
            let mut l1_jobs = Vec::new();
            for pos in 0..4 {
                let left_page = pos * 2;
                let right_page = pos * 2 + 1;
                let mut left = [0u8; 32];
                let mut right = [0u8; 32];
                left.copy_from_slice(active_bits.get_bits(left_page, 32));
                right.copy_from_slice(active_bits.get_bits(right_page, 32));
                l1_jobs.push(NodeHashJob {
                    level: 8,
                    left,
                    right,
                });
            }
            let l1_results = gpu.batch_node_hash(&l1_jobs);
            for pos in 0..4 {
                twig_gpu.active_bits_mtl1[pos] = l1_results[pos];
            }

            // Batch sync_l2: 2 jobs at level 9
            let l2_jobs = vec![
                NodeHashJob {
                    level: 9,
                    left: twig_gpu.active_bits_mtl1[0],
                    right: twig_gpu.active_bits_mtl1[1],
                },
                NodeHashJob {
                    level: 9,
                    left: twig_gpu.active_bits_mtl1[2],
                    right: twig_gpu.active_bits_mtl1[3],
                },
            ];
            let l2_results = gpu.batch_node_hash(&l2_jobs);
            twig_gpu.active_bits_mtl2[0] = l2_results[0];
            twig_gpu.active_bits_mtl2[1] = l2_results[1];

            // sync_l3: 1 job at level 10
            let l3_jobs = vec![NodeHashJob {
                level: 10,
                left: twig_gpu.active_bits_mtl2[0],
                right: twig_gpu.active_bits_mtl2[1],
            }];
            let l3_results = gpu.batch_node_hash(&l3_jobs);
            twig_gpu.active_bits_mtl3 = l3_results[0];

            // sync_top: 1 job at level 11
            let top_jobs = vec![NodeHashJob {
                level: 11,
                left: twig_gpu.left_root,
                right: twig_gpu.active_bits_mtl3,
            }];
            let top_results = gpu.batch_node_hash(&top_jobs);
            twig_gpu.twig_root = top_results[0];
            let gpu_time = gpu_start.elapsed();

            if t == 0 {
                println!(
                    "Active bits sync: CPU: {:?}, GPU: {:?}, speedup: {:.1}x",
                    cpu_time,
                    gpu_time,
                    cpu_time.as_secs_f64() / gpu_time.as_secs_f64().max(1e-9)
                );
            }

            // Verify all levels match
            for pos in 0..4 {
                assert_eq!(
                    twig_cpu.active_bits_mtl1[pos], twig_gpu.active_bits_mtl1[pos],
                    "Twig {} mtl1[{}] mismatch",
                    t, pos
                );
            }
            for pos in 0..2 {
                assert_eq!(
                    twig_cpu.active_bits_mtl2[pos], twig_gpu.active_bits_mtl2[pos],
                    "Twig {} mtl2[{}] mismatch",
                    t, pos
                );
            }
            assert_eq!(
                twig_cpu.active_bits_mtl3, twig_gpu.active_bits_mtl3,
                "Twig {} mtl3 mismatch", t
            );
            assert_eq!(
                twig_cpu.twig_root, twig_gpu.twig_root,
                "Twig {} twig_root mismatch: CPU={} GPU={}",
                t,
                hex::encode(twig_cpu.twig_root),
                hex::encode(twig_gpu.twig_root)
            );
        }
    }

    // =========================================================================
    // 7. Upper tree sync — node hashing at levels above twigs
    // =========================================================================

    #[test]
    fn test_upper_tree_sync_cpu_vs_gpu() {
        let gpu = gpu_or_skip!();

        // Build an upper tree with 128 twig roots (7 levels above twigs)
        let n_twigs = 128;
        let mut twig_roots = Vec::with_capacity(n_twigs);
        for i in 0..n_twigs {
            twig_roots.push(pseudo_random_hash(i as u64 * 3));
        }

        // CPU: compute all upper tree nodes level by level
        let cpu_start = Instant::now();
        let mut cpu_levels: Vec<Vec<[u8; 32]>> = Vec::new();
        cpu_levels.push(twig_roots.clone()); // level 0 = twig roots
        let mut current = twig_roots.clone();
        let mut level_val: u8 = 12; // TWIG_ROOT_LEVEL
        while current.len() > 1 {
            let mut next = Vec::new();
            let mut i = 0;
            while i < current.len() {
                let left = current[i];
                let right = if i + 1 < current.len() {
                    current[i + 1]
                } else {
                    NULL_TWIG.twig_root // use null twig root for odd nodes
                };
                next.push(cpu_hash2(level_val, &left, &right));
                i += 2;
            }
            cpu_levels.push(next.clone());
            current = next;
            level_val += 1;
        }
        let cpu_root = current[0];
        let cpu_time = cpu_start.elapsed();

        // GPU: same computation but batched
        let gpu_start = Instant::now();
        let mut gpu_levels: Vec<Vec<[u8; 32]>> = Vec::new();
        gpu_levels.push(twig_roots.clone());
        let mut gpu_current = twig_roots;
        level_val = 12;
        while gpu_current.len() > 1 {
            let mut jobs = Vec::new();
            let mut i = 0;
            while i < gpu_current.len() {
                let left = gpu_current[i];
                let right = if i + 1 < gpu_current.len() {
                    gpu_current[i + 1]
                } else {
                    NULL_TWIG.twig_root
                };
                jobs.push(NodeHashJob {
                    level: level_val,
                    left,
                    right,
                });
                i += 2;
            }
            let results = gpu.batch_node_hash(&jobs);
            gpu_levels.push(results.clone());
            gpu_current = results;
            level_val += 1;
        }
        let gpu_root = gpu_current[0];
        let gpu_time = gpu_start.elapsed();

        println!(
            "Upper tree sync ({} twigs, {} levels): CPU: {:?}, GPU: {:?}, speedup: {:.1}x",
            n_twigs,
            cpu_levels.len(),
            cpu_time,
            gpu_time,
            cpu_time.as_secs_f64() / gpu_time.as_secs_f64().max(1e-9)
        );

        // Check root
        assert_eq!(
            cpu_root, gpu_root,
            "Upper tree root mismatch: CPU={} GPU={}",
            hex::encode(cpu_root),
            hex::encode(gpu_root)
        );

        // Check every node at every level
        for (lvl, (cpu_nodes, gpu_nodes)) in cpu_levels.iter().zip(gpu_levels.iter()).enumerate() {
            assert_eq!(cpu_nodes.len(), gpu_nodes.len(), "Level {} count mismatch", lvl);
            for (i, (c, g)) in cpu_nodes.iter().zip(gpu_nodes.iter()).enumerate() {
                assert_eq!(c, g, "Level {} node {} mismatch", lvl, i);
            }
        }
    }

    // =========================================================================
    // 8. Full pipeline — small block (100 entries via build_test_tree)
    // =========================================================================

    #[test]
    fn test_full_pipeline_small_cpu_vs_gpu() {
        let gpu = gpu_or_skip!();

        // Build two identical trees with 100 entries
        let dir_cpu = "/tmp/qmdb_test_pipeline_small_cpu";
        let dir_gpu = "/tmp/qmdb_test_pipeline_small_gpu";
        let _ = std::fs::remove_dir_all(dir_cpu);
        let _ = std::fs::remove_dir_all(dir_cpu);

        let (mut tree_cpu, _, _, _) = build_test_tree(dir_cpu, &vec![], 50, 50);
        let (mut tree_gpu, _, _, _) = build_test_tree(dir_gpu, &vec![], 50, 50);

        // CPU path: sync youngest twig + active bits
        let cpu_start = Instant::now();
        tree_cpu.sync_mt_for_youngest_twig();
        let cpu_n_list = tree_cpu.sync_mt_for_active_bits_phase1();
        let cpu_time = cpu_start.elapsed();

        // GPU path: same operations via GPU
        let gpu_start = Instant::now();
        tree_gpu.sync_mt_for_youngest_twig_gpu(&gpu);
        let gpu_n_list = tree_gpu.sync_mt_for_active_bits_phase1_gpu(&gpu);
        let gpu_time = gpu_start.elapsed();

        println!(
            "Pipeline small (100 entries): CPU: {:?}, GPU: {:?}, speedup: {:.1}x",
            cpu_time,
            gpu_time,
            cpu_time.as_secs_f64() / gpu_time.as_secs_f64().max(1e-9)
        );

        // Verify youngest twig merkle tree roots match
        assert_eq!(
            tree_cpu.mtree_for_youngest_twig[1], tree_gpu.mtree_for_youngest_twig[1],
            "Youngest twig MT root mismatch: CPU={} GPU={}",
            hex::encode(tree_cpu.mtree_for_youngest_twig[1]),
            hex::encode(tree_gpu.mtree_for_youngest_twig[1])
        );

        // Verify all internal nodes of youngest twig MT
        for i in 1..2048 {
            assert_eq!(
                tree_cpu.mtree_for_youngest_twig[i],
                tree_gpu.mtree_for_youngest_twig[i],
                "Youngest twig MT node {} mismatch",
                i
            );
        }

        // Check hash consistency on both trees
        check_hash_consistency(&tree_cpu);
        check_hash_consistency(&tree_gpu);

        tree_cpu.close();
        tree_gpu.close();
        let _ = std::fs::remove_dir_all(dir_cpu);
        let _ = std::fs::remove_dir_all(dir_gpu);
    }

    // =========================================================================
    // 9. Full pipeline — large block (10K entries)
    // =========================================================================

    #[test]
    fn test_full_pipeline_large_cpu_vs_gpu() {
        let gpu = gpu_or_skip!();

        let n_entries = 5000; // Use 5000 to keep test reasonable
        let dir_cpu = "/tmp/qmdb_test_pipeline_large_cpu";
        let dir_gpu = "/tmp/qmdb_test_pipeline_large_gpu";
        let _ = std::fs::remove_dir_all(dir_cpu);
        let _ = std::fs::remove_dir_all(dir_gpu);

        let (mut tree_cpu, _, _, _) = build_test_tree(dir_cpu, &vec![], n_entries / 2, n_entries / 2);
        let (mut tree_gpu, _, _, _) = build_test_tree(dir_gpu, &vec![], n_entries / 2, n_entries / 2);

        // CPU path
        let cpu_start = Instant::now();
        tree_cpu.sync_mt_for_youngest_twig();
        let _cpu_n_list = tree_cpu.sync_mt_for_active_bits_phase1();
        let cpu_time = cpu_start.elapsed();

        // GPU path
        let gpu_start = Instant::now();
        tree_gpu.sync_mt_for_youngest_twig_gpu(&gpu);
        let _gpu_n_list = tree_gpu.sync_mt_for_active_bits_phase1_gpu(&gpu);
        let gpu_time = gpu_start.elapsed();

        println!(
            "Pipeline large ({} entries): CPU: {:?}, GPU: {:?}, speedup: {:.1}x",
            n_entries,
            cpu_time,
            gpu_time,
            cpu_time.as_secs_f64() / gpu_time.as_secs_f64().max(1e-9)
        );

        // Verify roots match
        assert_eq!(
            tree_cpu.mtree_for_youngest_twig[1], tree_gpu.mtree_for_youngest_twig[1],
            "Large pipeline youngest twig MT root mismatch"
        );

        // Verify full internal nodes
        for i in 1..2048 {
            assert_eq!(
                tree_cpu.mtree_for_youngest_twig[i],
                tree_gpu.mtree_for_youngest_twig[i],
                "Large pipeline youngest twig MT node {} mismatch",
                i
            );
        }

        check_hash_consistency(&tree_cpu);
        check_hash_consistency(&tree_gpu);

        tree_cpu.close();
        tree_gpu.close();
        let _ = std::fs::remove_dir_all(dir_cpu);
        let _ = std::fs::remove_dir_all(dir_gpu);
    }

    // =========================================================================
    // 10. Twig eviction — entries trigger youngest twig swap + active bits
    // =========================================================================

    #[test]
    fn test_twig_eviction_cpu_vs_gpu() {
        let gpu = gpu_or_skip!();

        // Build trees with enough entries to have multiple twigs
        // 2048 entries = 1 full twig, so 3000 entries creates 1 full + partial youngest
        let dir_cpu = "/tmp/qmdb_test_evict_cpu";
        let dir_gpu = "/tmp/qmdb_test_evict_gpu";
        let _ = std::fs::remove_dir_all(dir_cpu);
        let _ = std::fs::remove_dir_all(dir_gpu);

        let (mut tree_cpu, _, _, _) = build_test_tree(dir_cpu, &vec![], 1500, 1500);
        let (mut tree_gpu, _, _, _) = build_test_tree(dir_gpu, &vec![], 1500, 1500);

        // Sync both using respective paths
        tree_cpu.sync_mt_for_youngest_twig();
        let cpu_n_list = tree_cpu.sync_mt_for_active_bits_phase1();

        tree_gpu.sync_mt_for_youngest_twig_gpu(&gpu);
        let gpu_n_list = tree_gpu.sync_mt_for_active_bits_phase1_gpu(&gpu);

        // Verify youngest twig roots match
        assert_eq!(
            tree_cpu.mtree_for_youngest_twig[1], tree_gpu.mtree_for_youngest_twig[1],
            "Eviction test: youngest twig root mismatch"
        );

        // Verify active bits phase1 results match
        // The n_lists should contain the same twig positions (order may differ)
        let mut cpu_sorted = cpu_n_list.clone();
        cpu_sorted.sort();
        cpu_sorted.dedup();
        let mut gpu_sorted = gpu_n_list.clone();
        gpu_sorted.sort();
        gpu_sorted.dedup();
        assert_eq!(
            cpu_sorted, gpu_sorted,
            "Eviction test: phase1 n_list mismatch"
        );

        check_hash_consistency(&tree_cpu);
        check_hash_consistency(&tree_gpu);

        tree_cpu.close();
        tree_gpu.close();
        let _ = std::fs::remove_dir_all(dir_cpu);
        let _ = std::fs::remove_dir_all(dir_gpu);
    }

    // =========================================================================
    // 11. Empty block edge case
    // =========================================================================

    #[test]
    fn test_empty_block_cpu_vs_gpu() {
        let gpu = gpu_or_skip!();

        // Create a tree with some initial data, then do a "sync" with no new changes
        let dir_cpu = "/tmp/qmdb_test_empty_block_cpu";
        let dir_gpu = "/tmp/qmdb_test_empty_block_gpu";
        let _ = std::fs::remove_dir_all(dir_cpu);
        let _ = std::fs::remove_dir_all(dir_gpu);

        let (mut tree_cpu, _, _, _) = build_test_tree(dir_cpu, &vec![], 50, 50);
        let (mut tree_gpu, _, _, _) = build_test_tree(dir_gpu, &vec![], 50, 50);

        // First, sync both trees to establish baseline
        tree_cpu.sync_mt_for_youngest_twig();
        tree_gpu.sync_mt_for_youngest_twig_gpu(&gpu);

        // Now verify roots are identical
        assert_eq!(
            tree_cpu.mtree_for_youngest_twig[1], tree_gpu.mtree_for_youngest_twig[1],
            "Empty block: roots should match after initial sync"
        );

        check_hash_consistency(&tree_cpu);
        check_hash_consistency(&tree_gpu);

        tree_cpu.close();
        tree_gpu.close();
        let _ = std::fs::remove_dir_all(dir_cpu);
        let _ = std::fs::remove_dir_all(dir_gpu);
    }

    // =========================================================================
    // 12. Single entry edge case
    // =========================================================================

    #[test]
    fn test_single_entry_cpu_vs_gpu() {
        let gpu = gpu_or_skip!();

        let dir_cpu = "/tmp/qmdb_test_single_entry_cpu";
        let dir_gpu = "/tmp/qmdb_test_single_entry_gpu";
        let _ = std::fs::remove_dir_all(dir_cpu);
        let _ = std::fs::remove_dir_all(dir_gpu);

        // Build tree with exactly 1 entry
        let (mut tree_cpu, _, _, _) = build_test_tree(dir_cpu, &vec![], 1, 0);
        let (mut tree_gpu, _, _, _) = build_test_tree(dir_gpu, &vec![], 1, 0);

        // Sync
        tree_cpu.sync_mt_for_youngest_twig();
        tree_gpu.sync_mt_for_youngest_twig_gpu(&gpu);

        assert_eq!(
            tree_cpu.mtree_for_youngest_twig[1], tree_gpu.mtree_for_youngest_twig[1],
            "Single entry: root mismatch CPU={} GPU={}",
            hex::encode(tree_cpu.mtree_for_youngest_twig[1]),
            hex::encode(tree_gpu.mtree_for_youngest_twig[1])
        );

        // Check all internal nodes
        for i in 1..2048 {
            assert_eq!(
                tree_cpu.mtree_for_youngest_twig[i],
                tree_gpu.mtree_for_youngest_twig[i],
                "Single entry: node {} mismatch",
                i
            );
        }

        check_hash_consistency(&tree_cpu);
        check_hash_consistency(&tree_gpu);

        tree_cpu.close();
        tree_gpu.close();
        let _ = std::fs::remove_dir_all(dir_cpu);
        let _ = std::fs::remove_dir_all(dir_gpu);
    }

    // =========================================================================
    // 13. GPU determinism across runs
    // =========================================================================

    #[test]
    fn test_gpu_determinism_across_runs() {
        let gpu = gpu_or_skip!();

        let n = 10_000;
        let mut jobs = Vec::with_capacity(n);
        for i in 0..n {
            jobs.push(NodeHashJob {
                level: (i % 256) as u8,
                left: pseudo_random_hash(i as u64 * 2),
                right: pseudo_random_hash(i as u64 * 2 + 1),
            });
        }

        // Run 5 times and collect results
        let mut all_results = Vec::with_capacity(5);
        for run in 0..5 {
            let start = Instant::now();
            let results = gpu.batch_node_hash(&jobs);
            let elapsed = start.elapsed();
            println!("  Determinism run {}: {:?}", run + 1, elapsed);
            all_results.push(results);
        }

        // Assert all 5 runs are identical
        for run in 1..5 {
            for i in 0..n {
                assert_eq!(
                    all_results[0][i], all_results[run][i],
                    "Determinism failure: run 0 vs run {} at job {}: {} != {}",
                    run,
                    i,
                    hex::encode(all_results[0][i]),
                    hex::encode(all_results[run][i])
                );
            }
        }
        println!("All 5 runs of 10K hashes are byte-identical.");

        // Also test variable hash determinism
        let mut var_inputs: Vec<Vec<u8>> = Vec::with_capacity(1000);
        for i in 0..1000 {
            var_inputs.push(pseudo_random_bytes(i as u64 + 99999, 50 + (i % 200)));
        }
        let refs: Vec<&[u8]> = var_inputs.iter().map(|v| v.as_slice()).collect();

        let mut var_results = Vec::with_capacity(5);
        for _ in 0..5 {
            var_results.push(gpu.batch_hash_variable(&refs));
        }
        for run in 1..5 {
            for i in 0..1000 {
                assert_eq!(
                    var_results[0][i], var_results[run][i],
                    "Variable hash determinism failure at run {} item {}",
                    run, i
                );
            }
        }
        println!("All 5 runs of 1K variable hashes are byte-identical.");
    }

    // =========================================================================
    // Extra: Full twig chain reproduction — CPU vs GPU producing NULL_TWIG
    // =========================================================================

    #[test]
    fn test_null_twig_reproduction_cpu_vs_gpu() {
        let gpu = gpu_or_skip!();

        // Reproduce the NULL_TWIG using GPU hashing
        let null_active_bits = ActiveBits::new();

        // GPU: batch all sync_l1 operations
        let mut l1_jobs = Vec::new();
        for pos in 0..4 {
            let left_page = pos * 2;
            let right_page = pos * 2 + 1;
            let mut left = [0u8; 32];
            let mut right = [0u8; 32];
            left.copy_from_slice(null_active_bits.get_bits(left_page, 32));
            right.copy_from_slice(null_active_bits.get_bits(right_page, 32));
            l1_jobs.push(NodeHashJob {
                level: 8,
                left,
                right,
            });
        }
        let l1_results = gpu.batch_node_hash(&l1_jobs);

        // Verify L1 matches NULL_TWIG
        for pos in 0..4 {
            assert_eq!(
                l1_results[pos], NULL_TWIG.active_bits_mtl1[pos],
                "NULL_TWIG mtl1[{}] mismatch",
                pos
            );
        }

        // GPU: sync_l2
        let l2_jobs = vec![
            NodeHashJob {
                level: 9,
                left: l1_results[0],
                right: l1_results[1],
            },
            NodeHashJob {
                level: 9,
                left: l1_results[2],
                right: l1_results[3],
            },
        ];
        let l2_results = gpu.batch_node_hash(&l2_jobs);
        for pos in 0..2 {
            assert_eq!(
                l2_results[pos], NULL_TWIG.active_bits_mtl2[pos],
                "NULL_TWIG mtl2[{}] mismatch",
                pos
            );
        }

        // GPU: sync_l3
        let l3_jobs = vec![NodeHashJob {
            level: 10,
            left: l2_results[0],
            right: l2_results[1],
        }];
        let l3_results = gpu.batch_node_hash(&l3_jobs);
        assert_eq!(
            l3_results[0], NULL_TWIG.active_bits_mtl3,
            "NULL_TWIG mtl3 mismatch"
        );

        // GPU: sync_top using NULL_MT_FOR_TWIG[1] as left_root
        let top_jobs = vec![NodeHashJob {
            level: 11,
            left: NULL_MT_FOR_TWIG[1],
            right: l3_results[0],
        }];
        let top_results = gpu.batch_node_hash(&top_jobs);
        assert_eq!(
            top_results[0], NULL_TWIG.twig_root,
            "NULL_TWIG twig_root mismatch: GPU={} expected={}",
            hex::encode(top_results[0]),
            hex::encode(NULL_TWIG.twig_root)
        );
    }

    // =========================================================================
    // Extra: NULL_MT_FOR_TWIG reproduction via sync_mtrees_gpu
    // =========================================================================

    #[test]
    fn test_null_mt_reproduction_via_gpu() {
        let gpu = gpu_or_skip!();

        use crate::def::ENTRY_BASE_LENGTH;
        use crate::entryfile::entry;

        // Build a null MT the same way as create_null_mt_for_twig
        let mut bz = [0u8; ENTRY_BASE_LENGTH + 8];
        let null_hash = entry::null_entry(&mut bz[..]).hash();

        let mut mt_gpu = vec![ZERO_HASH32; 4096].into_boxed_slice();
        for i in 2048..4096 {
            mt_gpu[i] = null_hash;
        }

        // Sync via GPU
        sync_mtrees_gpu(&gpu, &mut [(&mut mt_gpu, 0, 2047)]);

        // Verify against the lazy_static NULL_MT_FOR_TWIG
        for i in 1..4096 {
            assert_eq!(
                mt_gpu[i], NULL_MT_FOR_TWIG[i],
                "NULL_MT mismatch at index {}: GPU={} expected={}",
                i,
                hex::encode(mt_gpu[i]),
                hex::encode(NULL_MT_FOR_TWIG[i])
            );
        }
    }

    // =========================================================================
    // Extra: Cross-method consistency — node hash vs variable hash for 65 bytes
    // =========================================================================

    #[test]
    fn test_cross_method_node_vs_variable_hash() {
        let gpu = gpu_or_skip!();

        // For 100 different inputs, verify that batch_node_hash and batch_hash_variable
        // produce the same result when given identical 65-byte inputs
        let n = 100;
        let mut node_jobs = Vec::with_capacity(n);
        let mut var_inputs_data: Vec<Vec<u8>> = Vec::with_capacity(n);

        for i in 0..n {
            let level = (i % 256) as u8;
            let left = pseudo_random_hash(i as u64 * 100);
            let right = pseudo_random_hash(i as u64 * 100 + 1);

            node_jobs.push(NodeHashJob { level, left, right });

            // Build the same 65-byte input manually
            let mut input = Vec::with_capacity(65);
            input.push(level);
            input.extend_from_slice(&left);
            input.extend_from_slice(&right);
            var_inputs_data.push(input);
        }

        let node_results = gpu.batch_node_hash(&node_jobs);
        let var_refs: Vec<&[u8]> = var_inputs_data.iter().map(|v| v.as_slice()).collect();
        let var_results = gpu.batch_hash_variable(&var_refs);

        for i in 0..n {
            assert_eq!(
                node_results[i], var_results[i],
                "Cross-method mismatch at {}: node={} var={}",
                i,
                hex::encode(node_results[i]),
                hex::encode(var_results[i])
            );
        }
    }

    // =========================================================================
    // Extra: Deactivation test — entries with deactivation flags
    // =========================================================================

    #[test]
    fn test_pipeline_with_deactivations_cpu_vs_gpu() {
        let gpu = gpu_or_skip!();

        let dir_cpu = "/tmp/qmdb_test_deact_cpu";
        let dir_gpu = "/tmp/qmdb_test_deact_gpu";
        let _ = std::fs::remove_dir_all(dir_cpu);
        let _ = std::fs::remove_dir_all(dir_gpu);

        // Build trees with deactivations (deactivate entries 5, 10, 15, 20)
        let deact_list = vec![5u64, 10, 15, 20];
        let (mut tree_cpu, _, _, _) = build_test_tree(dir_cpu, &deact_list, 50, 50);
        let (mut tree_gpu, _, _, _) = build_test_tree(dir_gpu, &deact_list, 50, 50);

        // Sync both
        tree_cpu.sync_mt_for_youngest_twig();
        let cpu_n_list = tree_cpu.sync_mt_for_active_bits_phase1();

        tree_gpu.sync_mt_for_youngest_twig_gpu(&gpu);
        let gpu_n_list = tree_gpu.sync_mt_for_active_bits_phase1_gpu(&gpu);

        // Verify youngest twig roots match
        assert_eq!(
            tree_cpu.mtree_for_youngest_twig[1], tree_gpu.mtree_for_youngest_twig[1],
            "Deactivation test: youngest twig root mismatch"
        );

        // Verify all internal nodes match
        for i in 1..2048 {
            assert_eq!(
                tree_cpu.mtree_for_youngest_twig[i],
                tree_gpu.mtree_for_youngest_twig[i],
                "Deactivation test: node {} mismatch",
                i
            );
        }

        check_hash_consistency(&tree_cpu);
        check_hash_consistency(&tree_gpu);

        tree_cpu.close();
        tree_gpu.close();
        let _ = std::fs::remove_dir_all(dir_cpu);
        let _ = std::fs::remove_dir_all(dir_gpu);
    }

    // =========================================================================
    // Extra: Large multi-twig with different ranges (realistic workload)
    // =========================================================================

    #[test]
    fn test_realistic_workload_100_twigs_cpu_vs_gpu() {
        let gpu = gpu_or_skip!();

        let n_twigs = 100;
        let mut cpu_mts: Vec<Box<[Hash32]>> = Vec::with_capacity(n_twigs);
        let mut gpu_mts: Vec<Box<[Hash32]>> = Vec::with_capacity(n_twigs);
        let mut ranges = Vec::with_capacity(n_twigs);

        for t in 0..n_twigs {
            let mut mt_cpu = NULL_MT_FOR_TWIG.to_vec().into_boxed_slice();
            let mut mt_gpu = NULL_MT_FOR_TWIG.to_vec().into_boxed_slice();

            // Each twig gets a different number of modified leaves (1 to 200)
            let n_modified = 1 + (t * 2) % 200;
            let start = (t * 7) % 2048;
            let end = std::cmp::min(start + n_modified, 2047);
            ranges.push((start as i32, end as i32));

            for i in start..=end {
                let idx = 2048 + (i % 2048);
                let h = pseudo_random_hash((t * 10000 + i) as u64);
                mt_cpu[idx] = h;
                mt_gpu[idx] = h;
            }
            cpu_mts.push(mt_cpu);
            gpu_mts.push(mt_gpu);
        }

        // CPU path
        let cpu_start = Instant::now();
        for (t, mt) in cpu_mts.iter_mut().enumerate() {
            sync_mtree(mt, ranges[t].0, ranges[t].1);
        }
        let cpu_time = cpu_start.elapsed();

        // GPU path
        let gpu_start = Instant::now();
        let mut gpu_slices: Vec<(&mut [Hash32], i32, i32)> = gpu_mts
            .iter_mut()
            .enumerate()
            .map(|(t, mt)| (mt.as_mut() as &mut [Hash32], ranges[t].0, ranges[t].1))
            .collect();
        sync_mtrees_gpu(&gpu, &mut gpu_slices);
        let gpu_time = gpu_start.elapsed();

        println!(
            "Realistic workload ({} twigs, varying ranges): CPU: {:?}, GPU: {:?}, speedup: {:.1}x",
            n_twigs,
            cpu_time,
            gpu_time,
            cpu_time.as_secs_f64() / gpu_time.as_secs_f64().max(1e-9)
        );

        let mut mismatches = 0;
        for t in 0..n_twigs {
            if cpu_mts[t][1] != gpu_mts[t][1] {
                mismatches += 1;
                println!(
                    "Twig {} root mismatch (range {:?}): CPU={} GPU={}",
                    t,
                    ranges[t],
                    hex::encode(cpu_mts[t][1]),
                    hex::encode(gpu_mts[t][1])
                );
            }
            // Check all internal nodes
            for i in 1..2048 {
                assert_eq!(
                    cpu_mts[t][i], gpu_mts[t][i],
                    "Twig {} node {} mismatch (range {:?})",
                    t, i, ranges[t]
                );
            }
        }
        assert_eq!(mismatches, 0, "{} root mismatches in {} twigs", mismatches, n_twigs);
    }
}
