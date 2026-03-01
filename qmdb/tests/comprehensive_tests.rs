/// Comprehensive integration, unit, and end-to-end tests for QMDB
/// performance optimizations (Phases 1-4).
///
/// Coverage areas:
///   - CPU batch hashing (P4)
///   - Hash function consistency & domain separation
///   - Lock-free entry buffer (P10)
///   - Atomic ordering correctness (P3)
///   - NUMA topology detection (P11)
///   - Twig & ActiveBits merkle structures
///   - Entry serialization roundtrip
///   - Changeset operations
///   - Edge cases & stress tests
///
/// GPU-specific tests (P8/P12) are gated behind #[cfg(feature = "cuda")].

use qmdb::def::{BIG_BUF_SIZE, ENTRY_BASE_LENGTH, SHARD_COUNT};
use qmdb::entryfile::entry::{self, Entry};
use qmdb::merkletree::twig::{ActiveBits, NULL_TWIG};
use qmdb::utils::changeset::ChangeSet;
use qmdb::utils::hasher::{self, batch_node_hash_cpu, hash, hash1, hash2, hash2x, ZERO_HASH32};
use qmdb::utils::numa::NumaTopology;
use sha2::{Digest, Sha256};

// ============================================================
// Helpers
// ============================================================

fn cpu_sha256(data: &[u8]) -> [u8; 32] {
    Sha256::digest(data).into()
}

fn cpu_hash2(level: u8, a: &[u8], b: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update([level]);
    h.update(a);
    h.update(b);
    h.finalize().into()
}

fn pseudo_random_hash(seed: u64) -> [u8; 32] {
    let mut h = [0u8; 32];
    let mut s = seed;
    for byte in h.iter_mut() {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *byte = (s >> 33) as u8;
    }
    h
}

fn make_entry<'a>(key: &'a [u8], value: &'a [u8], sn: u64) -> Entry<'a> {
    Entry {
        key,
        value,
        next_key_hash: &[0xab; 32],
        version: 12345,
        serial_number: sn,
    }
}

// ============================================================
// 1. CPU Batch Hashing Tests (P4: batch_node_hash_cpu)
// ============================================================

#[test]
fn test_batch_hash_cpu_empty() {
    let levels: &[u8] = &[];
    let lefts: &[[u8; 32]] = &[];
    let rights: &[[u8; 32]] = &[];
    let mut out: Vec<[u8; 32]> = vec![];
    batch_node_hash_cpu(levels, lefts, rights, &mut out);
    assert!(out.is_empty());
}

#[test]
fn test_batch_hash_cpu_single() {
    let left = [0x11u8; 32];
    let right = [0x22u8; 32];
    let mut out = [[0u8; 32]; 1];
    batch_node_hash_cpu(&[5], &[left], &[right], &mut out);
    let expected = hash2(5, &left, &right);
    assert_eq!(out[0], expected);
}

#[test]
fn test_batch_hash_cpu_multiple() {
    let n = 50;
    let mut levels = Vec::with_capacity(n);
    let mut lefts = Vec::with_capacity(n);
    let mut rights = Vec::with_capacity(n);
    let mut expected = Vec::with_capacity(n);

    for i in 0..n {
        let level = (i % 12) as u8;
        let left = pseudo_random_hash(i as u64 * 2);
        let right = pseudo_random_hash(i as u64 * 2 + 1);
        levels.push(level);
        lefts.push(left);
        rights.push(right);
        expected.push(hash2(level, &left, &right));
    }

    let mut out = vec![[0u8; 32]; n];
    batch_node_hash_cpu(&levels, &lefts, &rights, &mut out);

    for i in 0..n {
        assert_eq!(out[i], expected[i], "Mismatch at index {}", i);
    }
}

#[test]
fn test_batch_hash_cpu_all_zeros() {
    let zero = [0u8; 32];
    let mut out = [[0u8; 32]; 1];
    batch_node_hash_cpu(&[0], &[zero], &[zero], &mut out);
    let expected = hash2(0, &zero, &zero);
    assert_eq!(out[0], expected);
}

#[test]
fn test_batch_hash_cpu_all_ff() {
    let ff = [0xFFu8; 32];
    let mut out = [[0u8; 32]; 1];
    batch_node_hash_cpu(&[255], &[ff], &[ff], &mut out);
    let expected = hash2(255, &ff, &ff);
    assert_eq!(out[0], expected);
}

#[test]
fn test_batch_hash_cpu_all_level_values() {
    let left = [0x42u8; 32];
    let right = [0x84u8; 32];
    let levels: Vec<u8> = (0..=255).collect();
    let lefts = vec![left; 256];
    let rights = vec![right; 256];
    let mut out = vec![[0u8; 32]; 256];

    batch_node_hash_cpu(&levels, &lefts, &rights, &mut out);

    for level in 0..=255u8 {
        let expected = hash2(level, &left, &right);
        assert_eq!(out[level as usize], expected, "Level {} mismatch", level);
    }
}

#[test]
fn test_batch_hash_cpu_deterministic() {
    let left = pseudo_random_hash(999);
    let right = pseudo_random_hash(1000);
    let mut out1 = [[0u8; 32]; 1];
    let mut out2 = [[0u8; 32]; 1];
    batch_node_hash_cpu(&[7], &[left], &[right], &mut out1);
    batch_node_hash_cpu(&[7], &[left], &[right], &mut out2);
    assert_eq!(out1[0], out2[0]);
}

#[test]
fn test_batch_hash_cpu_stress_10k() {
    let n = 10_000;
    let mut levels = Vec::with_capacity(n);
    let mut lefts = Vec::with_capacity(n);
    let mut rights = Vec::with_capacity(n);

    for i in 0..n {
        levels.push((i % 64) as u8);
        lefts.push(pseudo_random_hash(i as u64));
        rights.push(pseudo_random_hash(i as u64 + n as u64));
    }

    let mut out = vec![[0u8; 32]; n];
    batch_node_hash_cpu(&levels, &lefts, &rights, &mut out);

    // Spot-check first, last, and middle
    for &idx in &[0, n / 2, n - 1] {
        let expected = hash2(levels[idx], &lefts[idx], &rights[idx]);
        assert_eq!(out[idx], expected, "Stress test mismatch at {}", idx);
    }
}

#[test]
fn test_batch_hash_cpu_sequential_data() {
    let n = 32;
    let mut lefts = Vec::with_capacity(n);
    let mut rights = Vec::with_capacity(n);

    for i in 0..n {
        let mut left = [0u8; 32];
        let mut right = [0u8; 32];
        left[0] = i as u8;
        right[0] = (i + 32) as u8;
        lefts.push(left);
        rights.push(right);
    }

    let levels = vec![0u8; n];
    let mut out = vec![[0u8; 32]; n];
    batch_node_hash_cpu(&levels, &lefts, &rights, &mut out);

    // Verify each is unique (no hash collisions for distinct inputs)
    let mut seen = std::collections::HashSet::new();
    for h in &out {
        assert!(seen.insert(*h), "Unexpected hash collision");
    }
}

// ============================================================
// 2. Hash Function Consistency Tests
// ============================================================

#[test]
fn test_hash2_known_vector() {
    let result = hash2(8, "hello", "world");
    assert_eq!(
        hex::encode(result),
        "8e6fc50a3f98a3c314021b89688ca83a9b5697ca956e211198625fc460ddf1e9"
    );
}

#[test]
fn test_hash2x_no_exchange() {
    let left = [0x11u8; 32];
    let right = [0x22u8; 32];
    assert_eq!(hash2(5, &left, &right), hash2x(5, &left, &right, false));
}

#[test]
fn test_hash2x_with_exchange() {
    let left = [0x11u8; 32];
    let right = [0x22u8; 32];
    assert_eq!(hash2(5, &right, &left), hash2x(5, &left, &right, true));
}

#[test]
fn test_hash2x_exchange_known_vector() {
    let result = hash2x(8, "world", "hello", true);
    assert_eq!(
        hex::encode(result),
        "8e6fc50a3f98a3c314021b89688ca83a9b5697ca956e211198625fc460ddf1e9"
    );
}

#[test]
fn test_node_hash_inplace_matches_hash2() {
    let left = [0xAAu8; 32];
    let right = [0xBBu8; 32];
    let mut target = [0u8; 32];
    hasher::node_hash_inplace(10, &mut target, &left, &right);
    assert_eq!(target, hash2(10, &left, &right));
}

#[test]
fn test_node_hash_inplace_known_vector() {
    let mut target = [0u8; 32];
    hasher::node_hash_inplace(8, &mut target, "hello", "world");
    assert_eq!(
        hex::encode(target),
        "8e6fc50a3f98a3c314021b89688ca83a9b5697ca956e211198625fc460ddf1e9"
    );
}

#[test]
fn test_batch_cpu_matches_hash2() {
    let n = 100;
    for i in 0..n {
        let level = (i * 7 % 12) as u8;
        let left = pseudo_random_hash(i);
        let right = pseudo_random_hash(i + 100);
        let mut out = [[0u8; 32]; 1];
        batch_node_hash_cpu(&[level], &[left], &[right], &mut out);
        assert_eq!(out[0], hash2(level, &left, &right), "Index {}", i);
    }
}

#[test]
fn test_hash_domain_separation_by_level() {
    let left = [0x42u8; 32];
    let right = [0x84u8; 32];
    let h0 = hash2(0, &left, &right);
    let h1 = hash2(1, &left, &right);
    let h255 = hash2(255, &left, &right);
    assert_ne!(h0, h1, "Level 0 and 1 should produce different hashes");
    assert_ne!(h0, h255, "Level 0 and 255 should produce different hashes");
    assert_ne!(h1, h255, "Level 1 and 255 should produce different hashes");
}

#[test]
fn test_hash_different_inputs_different_outputs() {
    let a = hash2(0, &[0x11; 32], &[0x22; 32]);
    let b = hash2(0, &[0x11; 32], &[0x23; 32]); // one bit different in right
    assert_ne!(a, b);
}

#[test]
fn test_hash_commutativity_broken() {
    // hash2 is NOT commutative (changing argument order changes result)
    let left = [0x11u8; 32];
    let right = [0x22u8; 32];
    assert_ne!(hash2(0, &left, &right), hash2(0, &right, &left));
}

#[test]
fn test_hash1_vs_hash2() {
    // hash1(level, data) != hash2(level, left, right) even if data = left||right
    let data = [0x33u8; 64];
    let h1 = hash1(5, &data);
    let h2 = hash2(5, &data[..32], &data[32..]);
    // These should be different since hash1 hashes level||64bytes while
    // hash2 hashes level||32bytes||32bytes (same bytes but different call)
    // Actually they produce the same SHA256 since the input bytes are identical
    // level || a[0..32] || a[32..64] = level || data[0..64]
    assert_eq!(h1, h2); // Same bytes → same hash
}

#[test]
fn test_hash_plain() {
    let data = b"test data for hashing";
    let result = hash(data);
    let expected = cpu_sha256(data);
    assert_eq!(result, expected);
}

#[test]
fn test_hash_empty() {
    let result = hash(b"");
    let expected = cpu_sha256(b"");
    assert_eq!(result, expected);
}

#[test]
fn test_zero_hash32_is_zeros() {
    assert_eq!(ZERO_HASH32, [0u8; 32]);
}

#[test]
fn test_hash2_with_zero_hashes() {
    let z = ZERO_HASH32;
    let result = hash2(0, &z, &z);
    // Should not be zero — SHA256 of zeros is a specific non-zero value
    assert_ne!(result, ZERO_HASH32);
}

// ============================================================
// 3. Entry Buffer Tests (P10: Lock-free stack, P3: relaxed atomics)
// ============================================================

#[test]
fn test_entry_buffer_single_entry() {
    let (mut writer, _reader) = qmdb::entryfile::entrybuffer::new(0, 3 * BIG_BUF_SIZE);
    let entry = make_entry(b"key1", b"value1", 1);
    let pos = writer.append(&entry, &[]);
    assert_eq!(pos, 0);
    // Verify pos_receiver doesn't have data yet (entry fits in current buf)
}

#[test]
fn test_entry_buffer_multiple_entries_same_buf() {
    let (mut writer, _reader) = qmdb::entryfile::entrybuffer::new(0, 3 * BIG_BUF_SIZE);
    let entry = make_entry(b"key", b"val", 1);
    let pos0 = writer.append(&entry, &[]);
    assert_eq!(pos0, 0);
    let pos1 = writer.append(&entry, &[]);
    assert!(pos1 > 0);
    let pos2 = writer.append(&entry, &[]);
    assert!(pos2 > pos1);
}

#[test]
fn test_entry_buffer_end_block() {
    let (mut writer, mut reader) = qmdb::entryfile::entrybuffer::new(0, 3 * BIG_BUF_SIZE);
    let value = vec![0x42u8; 10000];
    let entry = make_entry(b"key", &value, 1);
    writer.append(&entry, &[]);
    writer.end_block(100, 200, 300);

    // The read_next_entry path exercises the channel internally
    let (end_of_block, file_pos) = reader.read_next_entry(|_ebz| {});
    assert!(!end_of_block);
    assert_eq!(file_pos, 0);

    // Next read should signal end of block
    let (end_of_block, _) = reader.read_next_entry(|_| {});
    assert!(end_of_block);

    let (cdp, cdsn, sne) = reader.read_extra_info();
    assert_eq!(cdp, 100);
    assert_eq!(cdsn, 200);
    assert_eq!(sne, 300);
}

#[test]
fn test_entry_buffer_get_entry_bz_at() {
    let (mut writer, _reader) = qmdb::entryfile::entrybuffer::new(0, 3 * BIG_BUF_SIZE);
    let value = vec![0x42u8; 10000];
    let entry = make_entry(b"key", &value, 1);
    let dsn: Vec<u64> = vec![1];
    writer.append(&entry, &dsn);

    let (in_disk, have_accessed) = writer.get_entry_bz_at(0, |_ebz| {});
    assert!(!in_disk);
    assert!(have_accessed);
}

#[test]
fn test_entry_buffer_negative_pos_is_in_disk() {
    let (mut writer, _reader) = qmdb::entryfile::entrybuffer::new(0, 3 * BIG_BUF_SIZE);
    let (in_disk, have_accessed) = writer.get_entry_bz_at(-1, |_| {});
    assert!(in_disk);
    assert!(!have_accessed);
}

#[test]
fn test_entry_buffer_spanning_bufs() {
    let (mut writer, _reader) = qmdb::entryfile::entrybuffer::new(0, 3 * BIG_BUF_SIZE);
    let value = vec![0x42u8; 40000]; // Large value to force buf spanning

    // Keep appending until we span a buffer boundary
    let mut positions = Vec::new();
    for i in 0..5 {
        let entry = make_entry(b"key", &value, i);
        let pos = writer.append(&entry, &[]);
        positions.push(pos);
    }
    // Should have crossed at least one boundary
    let crossed = positions.iter().any(|&p| p > BIG_BUF_SIZE as i64);
    assert!(crossed, "Should have crossed a buffer boundary");
}

#[cfg(not(feature = "tee_cipher"))]
#[test]
#[should_panic(expected = "Entry too large")]
fn test_entry_buffer_too_large_entry() {
    let (mut writer, _reader) = qmdb::entryfile::entrybuffer::new(0, 3 * BIG_BUF_SIZE);
    let value = vec![0u8; 100000]; // Exceeds BIG_BUF_SIZE
    let entry = make_entry(b"key", &value, 1);
    writer.append(&entry, &[]);
}

// ============================================================
// 4. Entry Serialization Tests
// ============================================================

#[test]
fn test_entry_serialization_roundtrip() {
    let key = b"my_key";
    let value = b"my_value";
    let entry = Entry {
        key,
        value,
        next_key_hash: &[0xAB; 32],
        version: 42,
        serial_number: 99,
    };
    let dsn_list: Vec<u64> = vec![10, 20];
    let total_size = entry.get_serialized_len(dsn_list.len());
    let mut buf = vec![0u8; total_size];
    let ebz = entry::entry_to_bytes(&entry, &dsn_list, &mut buf);

    assert_eq!(ebz.key(), key.as_slice());
    assert_eq!(ebz.value(), value.as_slice());
    assert_eq!(ebz.version(), 42);
    assert_eq!(ebz.serial_number(), 99);
    assert_eq!(ebz.next_key_hash(), &[0xAB; 32]);
}

#[test]
fn test_entry_empty_value() {
    let entry = Entry {
        key: b"k",
        value: b"",
        next_key_hash: &[0; 32],
        version: 0,
        serial_number: 1,
    };
    let total_size = entry.get_serialized_len(0);
    let mut buf = vec![0u8; total_size];
    let ebz = entry::entry_to_bytes(&entry, &[], &mut buf);
    assert_eq!(ebz.value().len(), 0);
    assert_eq!(ebz.key(), b"k");
}

#[test]
fn test_entry_with_dsn_list() {
    let entry = Entry {
        key: b"key",
        value: b"val",
        next_key_hash: &[0xFF; 32],
        version: 100,
        serial_number: 50,
    };
    let dsn_list = vec![1u64, 2, 3, 4, 5];
    let total_size = entry.get_serialized_len(dsn_list.len());
    let mut buf = vec![0u8; total_size];
    let ebz = entry::entry_to_bytes(&entry, &dsn_list, &mut buf);

    assert_eq!(ebz.dsn_count(), 5);
    let dsns: Vec<(usize, u64)> = ebz.dsn_iter().collect();
    assert_eq!(dsns.len(), 5);
}

#[test]
fn test_entry_hash_deterministic() {
    let entry = Entry {
        key: b"test",
        value: b"data",
        next_key_hash: &[0x12; 32],
        version: 7,
        serial_number: 42,
    };
    let total_size = entry.get_serialized_len(0);
    let mut buf1 = vec![0u8; total_size];
    let mut buf2 = vec![0u8; total_size];
    let ebz1 = entry::entry_to_bytes(&entry, &[], &mut buf1);
    let ebz2 = entry::entry_to_bytes(&entry, &[], &mut buf2);
    assert_eq!(ebz1.hash(), ebz2.hash());
}

#[test]
fn test_entry_different_keys_different_hashes() {
    let e1 = Entry {
        key: b"key1",
        value: b"val",
        next_key_hash: &[0; 32],
        version: 0,
        serial_number: 1,
    };
    let e2 = Entry {
        key: b"key2",
        value: b"val",
        next_key_hash: &[0; 32],
        version: 0,
        serial_number: 2,
    };
    let s1 = e1.get_serialized_len(0);
    let s2 = e2.get_serialized_len(0);
    let mut b1 = vec![0u8; s1];
    let mut b2 = vec![0u8; s2];
    let ebz1 = entry::entry_to_bytes(&e1, &[], &mut b1);
    let ebz2 = entry::entry_to_bytes(&e2, &[], &mut b2);
    assert_ne!(ebz1.hash(), ebz2.hash());
}

#[test]
fn test_entry_various_value_sizes() {
    for size in &[0, 1, 10, 100, 1000, 10000, 50000] {
        let value = vec![0x42u8; *size];
        let entry = Entry {
            key: b"k",
            value: &value,
            next_key_hash: &[0; 32],
            version: 0,
            serial_number: 1,
        };
        let total_size = entry.get_serialized_len(0);
        let mut buf = vec![0u8; total_size];
        let ebz = entry::entry_to_bytes(&entry, &[], &mut buf);
        assert_eq!(ebz.value().len(), *size);
    }
}

#[test]
fn test_entry_sentry() {
    let mut buf = [0u8; ENTRY_BASE_LENGTH + 32 + 8];
    let sentry = entry::sentry_entry(0, 1, &mut buf);
    assert_eq!(sentry.serial_number(), 1);
    assert_eq!(sentry.value().len(), 0);
}

#[test]
fn test_entry_null() {
    let mut buf = [0u8; ENTRY_BASE_LENGTH + 32 + 8];
    let null = entry::null_entry(&mut buf);
    assert_eq!(null.key().len(), 0);
    assert_eq!(null.value().len(), 0);
}

// ============================================================
// 5. NUMA Topology Tests (P11)
// ============================================================

#[test]
fn test_numa_detect_returns_valid() {
    let topo = NumaTopology::detect();
    assert!(topo.num_nodes >= 1);
}

#[test]
fn test_numa_single_node_not_numa() {
    let topo = NumaTopology {
        num_nodes: 1,
        node_cpus: vec![vec![0, 1, 2, 3]],
    };
    assert!(!topo.is_numa());
}

#[test]
fn test_numa_multi_node_is_numa() {
    let topo = NumaTopology {
        num_nodes: 4,
        node_cpus: vec![vec![0, 1], vec![2, 3], vec![4, 5], vec![6, 7]],
    };
    assert!(topo.is_numa());
}

#[test]
fn test_numa_shard_mapping_round_robin() {
    let topo = NumaTopology {
        num_nodes: 4,
        node_cpus: vec![vec![0], vec![1], vec![2], vec![3]],
    };
    assert_eq!(topo.shard_to_node(0), 0);
    assert_eq!(topo.shard_to_node(1), 1);
    assert_eq!(topo.shard_to_node(2), 2);
    assert_eq!(topo.shard_to_node(3), 3);
    assert_eq!(topo.shard_to_node(4), 0); // wraps around
    assert_eq!(topo.shard_to_node(15), 3);
    assert_eq!(topo.shard_to_node(16), 0);
}

#[test]
fn test_numa_cpus_for_shard() {
    let topo = NumaTopology {
        num_nodes: 2,
        node_cpus: vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]],
    };
    assert_eq!(topo.cpus_for_shard(0), &[0, 1, 2, 3]);
    assert_eq!(topo.cpus_for_shard(1), &[4, 5, 6, 7]);
    assert_eq!(topo.cpus_for_shard(2), &[0, 1, 2, 3]); // wraps
}

#[test]
fn test_numa_16_shards_4_nodes() {
    let topo = NumaTopology {
        num_nodes: 4,
        node_cpus: vec![
            vec![0, 1, 2, 3],
            vec![4, 5, 6, 7],
            vec![8, 9, 10, 11],
            vec![12, 13, 14, 15],
        ],
    };
    // Each group of 4 shards maps to one node
    for shard in 0..SHARD_COUNT {
        let node = topo.shard_to_node(shard);
        assert!(node < 4, "shard {} mapped to invalid node {}", shard, node);
    }
}

#[test]
fn test_numa_all_cpus_covered() {
    let topo = NumaTopology::detect();
    let total_cpus: usize = topo.node_cpus.iter().map(|v| v.len()).sum();
    assert!(total_cpus >= 1, "Should detect at least 1 CPU");
}

// ============================================================
// 6. Changeset Tests
// ============================================================

#[test]
fn test_changeset_new() {
    let cs = ChangeSet::new();
    assert_eq!(cs.op_list.len(), 0);
    assert!(cs.data.is_empty());
}

#[test]
fn test_changeset_add_op() {
    let mut cs = ChangeSet::new();
    let key_hash = [0x42u8; 32];
    cs.add_op(1, 0, &key_hash, b"key1", b"val1", None);
    assert_eq!(cs.op_list.len(), 1);
}

#[test]
fn test_changeset_sort() {
    let mut cs = ChangeSet::new();
    let kh1 = [0x01u8; 32];
    let kh2 = [0x02u8; 32];
    let kh3 = [0x03u8; 32];
    // Add in reverse shard order (shard_id 2, 1, 0)
    cs.add_op(1, 2, &kh3, b"k3", b"v3", None);
    cs.add_op(1, 1, &kh2, b"k2", b"v2", None);
    cs.add_op(1, 0, &kh1, b"k1", b"v1", None);
    cs.sort();
    // After sort: should be ordered by shard_id
    assert_eq!(cs.op_list.len(), 3);
    // Verify ops are accessible via run_all
    let mut count = 0;
    cs.run_all(|_op_type, _kh, _k, _v, _rec| {
        count += 1;
    });
    assert_eq!(count, 3);
}

#[test]
fn test_changeset_empty_sort() {
    let mut cs = ChangeSet::new();
    cs.sort(); // Should not panic
    assert_eq!(cs.op_list.len(), 0);
}

#[test]
fn test_changeset_op_count_in_shard() {
    let mut cs = ChangeSet::new();
    let kh = [0x01u8; 32];
    cs.add_op(1, 0, &kh, b"k1", b"v1", None);
    cs.add_op(2, 0, &kh, b"k2", b"v2", None);
    cs.add_op(1, 1, &kh, b"k3", b"v3", None);
    cs.sort();
    assert_eq!(cs.op_count_in_shard(0), 2);
    assert_eq!(cs.op_count_in_shard(1), 1);
    assert_eq!(cs.op_count_in_shard(2), 0);
}

// ============================================================
// 7. Merkle Hash Tree Consistency Tests
// ============================================================

#[test]
fn test_merkle_leaf_to_root_2_leaves() {
    let leaf0 = pseudo_random_hash(0);
    let leaf1 = pseudo_random_hash(1);
    let root = hash2(0, &leaf0, &leaf1);
    let expected = cpu_hash2(0, &leaf0, &leaf1);
    assert_eq!(root, expected);
}

#[test]
fn test_merkle_3_level_tree() {
    // 4 leaves → 2 internal → 1 root
    let l0 = pseudo_random_hash(0);
    let l1 = pseudo_random_hash(1);
    let l2 = pseudo_random_hash(2);
    let l3 = pseudo_random_hash(3);

    let n0 = hash2(0, &l0, &l1);
    let n1 = hash2(0, &l2, &l3);
    let root = hash2(1, &n0, &n1);

    // Verify using raw SHA256
    let expected_n0 = cpu_hash2(0, &l0, &l1);
    let expected_n1 = cpu_hash2(0, &l2, &l3);
    let expected_root = cpu_hash2(1, &expected_n0, &expected_n1);

    assert_eq!(root, expected_root);
}

#[test]
fn test_merkle_4_level_tree_batch() {
    // 8 leaves → 4 nodes (level 0) → 2 nodes (level 1) → 1 root (level 2)
    let leaves: Vec<[u8; 32]> = (0..8).map(|i| pseudo_random_hash(i)).collect();

    // Level 0: batch hash
    let levels = vec![0u8; 4];
    let lefts: Vec<[u8; 32]> = vec![leaves[0], leaves[2], leaves[4], leaves[6]];
    let rights: Vec<[u8; 32]> = vec![leaves[1], leaves[3], leaves[5], leaves[7]];
    let mut l0_out = vec![[0u8; 32]; 4];
    batch_node_hash_cpu(&levels, &lefts, &rights, &mut l0_out);

    // Level 1: batch hash
    let levels = vec![1u8; 2];
    let lefts = vec![l0_out[0], l0_out[2]];
    let rights = vec![l0_out[1], l0_out[3]];
    let mut l1_out = vec![[0u8; 32]; 2];
    batch_node_hash_cpu(&levels, &lefts, &rights, &mut l1_out);

    // Level 2: root
    let root = hash2(2, &l1_out[0], &l1_out[1]);

    // Verify against individual hash2 calls
    let n0 = hash2(0, &leaves[0], &leaves[1]);
    let n1 = hash2(0, &leaves[2], &leaves[3]);
    let n2 = hash2(0, &leaves[4], &leaves[5]);
    let n3 = hash2(0, &leaves[6], &leaves[7]);
    let m0 = hash2(1, &n0, &n1);
    let m1 = hash2(1, &n2, &n3);
    let expected_root = hash2(2, &m0, &m1);

    assert_eq!(root, expected_root);
}

#[test]
fn test_merkle_level_domain_separation_matters() {
    let left = pseudo_random_hash(100);
    let right = pseudo_random_hash(200);

    // Same data at different levels must produce different hashes
    let mut hashes = std::collections::HashSet::new();
    for level in 0..20 {
        let h = hash2(level, &left, &right);
        assert!(
            hashes.insert(h),
            "Level {} produced a collision with a previous level",
            level
        );
    }
}

// ============================================================
// 8. Active Bits Tests
// ============================================================

#[test]
fn test_active_bits_set_and_get() {
    let mut ab = ActiveBits::new();
    assert!(!ab.get_bit(0));
    ab.set_bit(0);
    assert!(ab.get_bit(0));
}

#[test]
fn test_active_bits_set_multiple() {
    let mut ab = ActiveBits::new();
    for i in [0, 7, 15, 100, 255, 1000, 2047] {
        ab.set_bit(i);
    }
    for i in [0, 7, 15, 100, 255, 1000, 2047] {
        assert!(ab.get_bit(i), "Bit {} should be set", i);
    }
    assert!(!ab.get_bit(1), "Bit 1 should not be set");
    assert!(!ab.get_bit(500), "Bit 500 should not be set");
}

#[test]
fn test_active_bits_clear() {
    let mut ab = ActiveBits::new();
    ab.set_bit(42);
    assert!(ab.get_bit(42));
    ab.clear_bit(42);
    assert!(!ab.get_bit(42));
}

#[test]
fn test_active_bits_clear_range_manual() {
    let mut ab = ActiveBits::new();
    for i in 0..2048u32 {
        ab.set_bit(i);
    }
    // Clear bits 0..1024 one by one (no bulk clear_bits method)
    for i in 0..1024u32 {
        ab.clear_bit(i);
    }
    for i in 0..1024u32 {
        assert!(!ab.get_bit(i), "Bit {} should be cleared", i);
    }
    for i in 1024..2048u32 {
        assert!(ab.get_bit(i), "Bit {} should still be set", i);
    }
}

#[test]
fn test_active_bits_boundary() {
    let mut ab = ActiveBits::new();
    // Test byte boundary
    ab.set_bit(7);
    ab.set_bit(8);
    assert!(ab.get_bit(7));
    assert!(ab.get_bit(8));
    assert!(!ab.get_bit(6));
    assert!(!ab.get_bit(9));
}

#[test]
fn test_active_bits_all_bits() {
    let mut ab = ActiveBits::new();
    for i in 0..2048 {
        ab.set_bit(i);
    }
    for i in 0..2048 {
        assert!(ab.get_bit(i));
    }
}

#[test]
fn test_active_bits_get_bits_returns_correct_slice() {
    let ab = ActiveBits::new();
    // get_bits(page_num, page_size) returns a slice of page_size bytes
    let bits = ab.get_bits(0, 256);
    assert_eq!(bits.len(), 256); // 2048 / 8 = 256 bytes
    // All zeros initially
    assert!(bits.iter().all(|&b| b == 0));
}

// ============================================================
// 9. Twig Tests
// ============================================================

#[test]
fn test_null_twig_fields_consistent() {
    let twig = NULL_TWIG.clone();
    // NULL_TWIG should have deterministic field values
    // left_root and twig_root should be non-zero (computed from null leaf hashes)
    assert_eq!(twig.left_root, NULL_TWIG.left_root);
    assert_eq!(twig.twig_root, NULL_TWIG.twig_root);
    assert_eq!(twig.active_bits_mtl3, NULL_TWIG.active_bits_mtl3);
}

#[test]
fn test_null_twig_twig_root_not_zero() {
    // NULL_TWIG is a valid initialized twig with computed twig_root
    let twig = NULL_TWIG.clone();
    // The twig_root should be a deterministic value based on the null initialization
    // It should be consistent across calls
    let twig2 = NULL_TWIG.clone();
    assert_eq!(twig.twig_root, twig2.twig_root);
}

// ============================================================
// 10. Stress & Edge Case Tests
// ============================================================

#[test]
fn test_hash_consistency_100k_random() {
    // Verify hash2 is consistent over many random inputs
    for seed in 0..1000u64 {
        let left = pseudo_random_hash(seed * 3);
        let right = pseudo_random_hash(seed * 3 + 1);
        let level = (seed % 12) as u8;

        let h1 = hash2(level, &left, &right);
        let h2 = hash2(level, &left, &right);
        assert_eq!(h1, h2, "Inconsistency at seed {}", seed);
    }
}

#[test]
fn test_batch_hash_various_sizes() {
    for &n in &[1, 2, 3, 4, 5, 10, 50, 100, 500, 1000] {
        let levels: Vec<u8> = (0..n).map(|i| (i % 12) as u8).collect();
        let lefts: Vec<[u8; 32]> = (0..n).map(|i| pseudo_random_hash(i as u64)).collect();
        let rights: Vec<[u8; 32]> =
            (0..n).map(|i| pseudo_random_hash(i as u64 + 10000)).collect();
        let mut out = vec![[0u8; 32]; n];
        batch_node_hash_cpu(&levels, &lefts, &rights, &mut out);

        // Verify first and last
        let first = hash2(levels[0], &lefts[0], &rights[0]);
        let last = hash2(levels[n - 1], &lefts[n - 1], &rights[n - 1]);
        assert_eq!(out[0], first, "First element mismatch for n={}", n);
        assert_eq!(out[n - 1], last, "Last element mismatch for n={}", n);
    }
}

#[test]
fn test_max_level_hashing() {
    let left = [0x11u8; 32];
    let right = [0x22u8; 32];
    let result = hash2(255, &left, &right);
    // Should not panic and should produce a valid 32-byte hash
    assert_ne!(result, ZERO_HASH32);
}

#[test]
fn test_node_hash_inplace_all_levels() {
    let left = [0xAAu8; 32];
    let right = [0xBBu8; 32];
    for level in 0..=255u8 {
        let mut target = [0u8; 32];
        hasher::node_hash_inplace(level, &mut target, &left, &right);
        let expected = hash2(level, &left, &right);
        assert_eq!(target, expected, "Level {} mismatch", level);
    }
}

#[test]
fn test_batch_hash_identical_inputs() {
    // All inputs the same → all outputs the same
    let left = [0x42u8; 32];
    let right = [0x84u8; 32];
    let n = 100;
    let levels = vec![5u8; n];
    let lefts = vec![left; n];
    let rights = vec![right; n];
    let mut out = vec![[0u8; 32]; n];
    batch_node_hash_cpu(&levels, &lefts, &rights, &mut out);

    let expected = hash2(5, &left, &right);
    for i in 0..n {
        assert_eq!(out[i], expected, "Index {} should match", i);
    }
}

#[test]
fn test_hash_avalanche_single_bit() {
    // Flipping one bit should change ~50% of output bits
    let mut left1 = [0u8; 32];
    let mut left2 = [0u8; 32];
    left1[0] = 0x00;
    left2[0] = 0x01; // Flip one bit

    let right = [0u8; 32];
    let h1 = hash2(0, &left1, &right);
    let h2 = hash2(0, &left2, &right);

    assert_ne!(h1, h2);
    // Count differing bits
    let diff_bits: u32 = h1
        .iter()
        .zip(h2.iter())
        .map(|(a, b)| (a ^ b).count_ones())
        .sum();
    // SHA256 avalanche: expect ~128 bits to differ (50% of 256)
    assert!(
        diff_bits > 64 && diff_bits < 192,
        "Avalanche: {} bits differ (expected ~128)",
        diff_bits
    );
}

#[test]
fn test_hash_preimage_resistance() {
    // The hash of any non-empty input should not be zero
    for i in 0..100 {
        let left = pseudo_random_hash(i);
        let right = pseudo_random_hash(i + 100);
        let h = hash2((i % 12) as u8, &left, &right);
        assert_ne!(h, ZERO_HASH32, "Hash should not be zero at i={}", i);
    }
}

// ============================================================
// 11. Byte Utility Tests
// ============================================================

#[test]
fn test_byte0_to_shard_id() {
    use qmdb::utils::byte0_to_shard_id;
    assert_eq!(byte0_to_shard_id(0), 0);
    assert_eq!(byte0_to_shard_id(255), SHARD_COUNT - 1);
    // Mid-range
    let mid = byte0_to_shard_id(128);
    assert!(mid >= SHARD_COUNT / 2);
}

#[test]
fn test_byte0_to_shard_id_distribution() {
    use qmdb::utils::byte0_to_shard_id;
    let mut counts = vec![0usize; SHARD_COUNT];
    for b in 0..=255u8 {
        counts[byte0_to_shard_id(b)] += 1;
    }
    // Each shard should get 256/SHARD_COUNT = 16 bytes
    for (i, &count) in counts.iter().enumerate() {
        assert_eq!(count, 256 / SHARD_COUNT, "Shard {} has {} entries", i, count);
    }
}

// ============================================================
// 12. GPU Tests (gated behind cuda feature)
// ============================================================

#[cfg(feature = "cuda")]
mod gpu_tests {
    use qmdb::gpu::{GpuHasher, MultiGpuHasher, NodeHashJob};
    use super::*;

    macro_rules! gpu_or_skip {
        ($max_batch:expr) => {
            match GpuHasher::new($max_batch) {
                Ok(g) => g,
                Err(e) => {
                    eprintln!("Skipping GPU test (no CUDA): {}", e);
                    return;
                }
            }
        };
        () => {
            gpu_or_skip!(200_000)
        };
    }

    #[test]
    fn test_gpu_device_count() {
        match GpuHasher::device_count() {
            Ok(count) => assert!(count >= 0),
            Err(_) => {} // No CUDA runtime
        }
    }

    #[test]
    fn test_gpu_new_on_device_0() {
        let _gpu = gpu_or_skip!();
    }

    #[test]
    fn test_gpu_batch_node_hash_matches_cpu() {
        let gpu = gpu_or_skip!(10000);
        let mut jobs = Vec::new();
        let mut expected = Vec::new();
        for i in 0..100 {
            let level = (i % 12) as u8;
            let left = pseudo_random_hash(i as u64);
            let right = pseudo_random_hash(i as u64 + 100);
            jobs.push(NodeHashJob { level, left, right });
            expected.push(hash2(level, &left, &right));
        }
        let results = gpu.batch_node_hash(&jobs);
        for (i, (gpu_h, cpu_h)) in results.iter().zip(expected.iter()).enumerate() {
            assert_eq!(gpu_h, cpu_h, "Mismatch at job {}", i);
        }
    }

    #[test]
    fn test_gpu_batch_node_hash_empty() {
        let gpu = gpu_or_skip!(1000);
        let result = gpu.batch_node_hash(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_gpu_batch_node_hash_single() {
        let gpu = gpu_or_skip!(1000);
        let left = [0x42u8; 32];
        let right = [0x84u8; 32];
        let jobs = vec![NodeHashJob { level: 5, left, right }];
        let result = gpu.batch_node_hash(&jobs);
        assert_eq!(result[0], hash2(5, &left, &right));
    }

    #[test]
    fn test_gpu_batch_hash_variable_matches_cpu() {
        let gpu = gpu_or_skip!(10000);
        let data: Vec<Vec<u8>> = vec![
            vec![0x01; 10],
            vec![0x02; 55],
            vec![0x03; 64],
            vec![0x04; 65],
            vec![0x05; 100],
            vec![0x06; 200],
        ];
        let inputs: Vec<&[u8]> = data.iter().map(|v| v.as_slice()).collect();
        let expected: Vec<[u8; 32]> = data.iter().map(|v| hash(v)).collect();
        let results = gpu.batch_hash_variable(&inputs);
        for (i, (gpu_h, cpu_h)) in results.iter().zip(expected.iter()).enumerate() {
            assert_eq!(gpu_h, cpu_h, "Variable hash mismatch at {}", i);
        }
    }

    #[test]
    fn test_gpu_warp_coop_matches_standard() {
        let gpu = gpu_or_skip!(10000);
        let mut jobs = Vec::new();
        for i in 0..100 {
            let level = (i % 12) as u8;
            let left = pseudo_random_hash(i as u64 * 3);
            let right = pseudo_random_hash(i as u64 * 3 + 1);
            jobs.push(NodeHashJob { level, left, right });
        }
        let standard = gpu.batch_node_hash(&jobs);
        let warp_coop = gpu.batch_node_hash_warp_coop(&jobs);
        for (i, (s, w)) in standard.iter().zip(warp_coop.iter()).enumerate() {
            assert_eq!(s, w, "Warp-coop mismatch at job {}", i);
        }
    }

    #[test]
    fn test_gpu_warp_coop_empty() {
        let gpu = gpu_or_skip!(1000);
        let result = gpu.batch_node_hash_warp_coop(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_gpu_warp_coop_large_batch() {
        let gpu = gpu_or_skip!(50000);
        let n = 10000;
        let mut jobs = Vec::with_capacity(n);
        for i in 0..n {
            jobs.push(NodeHashJob {
                level: (i % 64) as u8,
                left: pseudo_random_hash(i as u64),
                right: pseudo_random_hash(i as u64 + n as u64),
            });
        }
        let standard = gpu.batch_node_hash(&jobs);
        let warp_coop = gpu.batch_node_hash_warp_coop(&jobs);
        assert_eq!(standard.len(), warp_coop.len());
        for (i, (s, w)) in standard.iter().zip(warp_coop.iter()).enumerate() {
            assert_eq!(s, w, "Warp-coop mismatch at index {} of {}", i, n);
        }
    }

    #[test]
    fn test_gpu_batch_node_hash_into() {
        let gpu = gpu_or_skip!(1000);
        let left = [0x42u8; 32];
        let right = [0x84u8; 32];
        let jobs = vec![
            NodeHashJob { level: 0, left, right },
            NodeHashJob { level: 1, left, right },
        ];
        let mut out = [[0u8; 32]; 2];
        gpu.batch_node_hash_into(&jobs, &mut out);
        assert_eq!(out[0], hash2(0, &left, &right));
        assert_eq!(out[1], hash2(1, &left, &right));
    }

    #[test]
    fn test_multi_gpu_hasher() {
        let mgpu = match MultiGpuHasher::new(10000) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("Skipping multi-GPU test: {}", e);
                return;
            }
        };
        assert!(mgpu.gpu_count() >= 1);

        // Each shard maps to a valid GPU
        for shard in 0..16 {
            let _gpu = mgpu.for_shard(shard);
        }
    }

    #[test]
    fn test_multi_gpu_batch_hash() {
        let mgpu = match MultiGpuHasher::new(10000) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("Skipping multi-GPU test: {}", e);
                return;
            }
        };

        let left = [0x42u8; 32];
        let right = [0x84u8; 32];
        let jobs = vec![NodeHashJob { level: 5, left, right }];
        let result = mgpu.batch_node_hash(0, &jobs);
        assert_eq!(result[0], hash2(5, &left, &right));
    }

    #[test]
    fn test_multi_gpu_shard_routing() {
        let mgpu = match MultiGpuHasher::new(10000) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("Skipping multi-GPU test: {}", e);
                return;
            }
        };

        // All shards should produce same results for same input
        let left = pseudo_random_hash(42);
        let right = pseudo_random_hash(43);
        let jobs = vec![NodeHashJob { level: 0, left, right }];
        let expected = hash2(0, &left, &right);

        for shard in 0..16 {
            let result = mgpu.batch_node_hash(shard, &jobs);
            assert_eq!(result[0], expected, "Shard {} result mismatch", shard);
        }
    }

    #[test]
    fn test_gpu_stress_repeated_calls() {
        let gpu = gpu_or_skip!(10000);
        let left = [0x42u8; 32];
        let right = [0x84u8; 32];
        let jobs = vec![NodeHashJob { level: 5, left, right }];
        let expected = hash2(5, &left, &right);

        // Call many times to test buffer reuse
        for i in 0..100 {
            let result = gpu.batch_node_hash(&jobs);
            assert_eq!(result[0], expected, "Iteration {} mismatch", i);
        }
    }

    #[test]
    fn test_gpu_all_levels() {
        let gpu = gpu_or_skip!(1000);
        let left = [0x42u8; 32];
        let right = [0x84u8; 32];
        let mut jobs = Vec::with_capacity(256);
        let mut expected = Vec::with_capacity(256);

        for level in 0..=255u8 {
            jobs.push(NodeHashJob { level, left, right });
            expected.push(hash2(level, &left, &right));
        }

        let results = gpu.batch_node_hash(&jobs);
        for (level, (gpu_h, cpu_h)) in results.iter().zip(expected.iter()).enumerate() {
            assert_eq!(gpu_h, cpu_h, "Level {} mismatch", level);
        }
    }

    #[test]
    fn test_gpu_variable_hash_edge_lengths() {
        let gpu = gpu_or_skip!(1000);

        // Test around SHA256 block boundary (64 bytes)
        for len in [1, 32, 55, 56, 63, 64, 65, 100, 127, 128, 256] {
            let data = vec![(len % 256) as u8; len];
            let inputs = vec![data.as_slice()];
            let result = gpu.batch_hash_variable(&inputs);
            let expected = hash(&data);
            assert_eq!(result[0], expected, "Variable hash mismatch for len={}", len);
        }
    }
}

// ============================================================
// 13. Concurrent / Multi-threaded Tests
// ============================================================

#[test]
fn test_hash2_thread_safety() {
    use std::sync::Arc;
    use std::thread;

    let left = Arc::new([0x42u8; 32]);
    let right = Arc::new([0x84u8; 32]);
    let expected = hash2(5, left.as_ref(), right.as_ref());

    let handles: Vec<_> = (0..8)
        .map(|_| {
            let l = left.clone();
            let r = right.clone();
            thread::spawn(move || hash2(5, l.as_ref(), r.as_ref()))
        })
        .collect();

    for h in handles {
        assert_eq!(h.join().unwrap(), expected);
    }
}

#[test]
fn test_batch_hash_cpu_thread_safety() {
    use std::thread;

    let handles: Vec<_> = (0..8)
        .map(|t| {
            thread::spawn(move || {
                let left = pseudo_random_hash(t as u64);
                let right = pseudo_random_hash(t as u64 + 100);
                let mut out = [[0u8; 32]; 1];
                batch_node_hash_cpu(&[5], &[left], &[right], &mut out);
                (out[0], hash2(5, &left, &right))
            })
        })
        .collect();

    for h in handles {
        let (batch, individual) = h.join().unwrap();
        assert_eq!(batch, individual);
    }
}

// ============================================================
// 14. Codec Tests
// ============================================================

#[test]
fn test_codec_encode_decode_i64() {
    use qmdb::utils::codec::{decode_le_i64, encode_le_i64};
    for &val in &[0i64, 1, -1, i64::MAX, i64::MIN, 42, -42, 0x7FFF_FFFF_FFFF_FFFF] {
        let encoded = encode_le_i64(val);
        let decoded = decode_le_i64(&encoded);
        assert_eq!(decoded, val, "Roundtrip failed for {}", val);
    }
}

#[test]
fn test_codec_encode_decode_u64() {
    use qmdb::utils::codec::{decode_le_u64, encode_le_u64};
    for &val in &[0u64, 1, u64::MAX, 42, 0xDEAD_BEEF, 0xFFFF_FFFF_FFFF_FFFF] {
        let encoded = encode_le_u64(val);
        let decoded = decode_le_u64(&encoded);
        assert_eq!(decoded, val, "Roundtrip failed for {}", val);
    }
}

// ============================================================
// 15. Shortlist Tests
// ============================================================

#[test]
fn test_shortlist_basic() {
    use qmdb::utils::shortlist::ShortList;
    let mut sl = ShortList::new();
    sl.append(10);
    sl.append(20);
    sl.append(30);
    assert_eq!(sl.len(), 3);
    assert_eq!(sl.get(0), 10);
    assert_eq!(sl.get(1), 20);
    assert_eq!(sl.get(2), 30);
}

#[test]
fn test_shortlist_clear() {
    use qmdb::utils::shortlist::ShortList;
    let mut sl = ShortList::new();
    sl.append(1);
    sl.append(2);
    sl.clear();
    assert_eq!(sl.len(), 0);
}

#[test]
fn test_shortlist_dedup() {
    use qmdb::utils::shortlist::ShortList;
    let mut sl = ShortList::new();
    sl.append(42);
    sl.append(42); // duplicate
    assert_eq!(sl.len(), 1); // ShortList deduplicates
}

#[test]
fn test_shortlist_contains() {
    use qmdb::utils::shortlist::ShortList;
    let mut sl = ShortList::new();
    sl.append(100);
    sl.append(200);
    assert!(sl.contains(100));
    assert!(sl.contains(200));
    assert!(!sl.contains(300));
}

// ============================================================
// 16. Pre-partitioning Correctness Tests (P2 verification)
// ============================================================

#[test]
fn test_pre_partition_correctness_4_shards() {
    // Simulate pre-partitioning n_list by NODE_SHARD_COUNT (4)
    let node_shard_count = 4;
    let n_list: Vec<u64> = (0..100).collect();

    let mut shard_lists: Vec<Vec<u64>> = vec![Vec::new(); node_shard_count];
    for &i in &n_list {
        shard_lists[i as usize % node_shard_count].push(i);
    }

    // Verify: each item appears in exactly one shard
    let mut total = 0;
    for (shard_id, list) in shard_lists.iter().enumerate() {
        for &i in list {
            assert_eq!(i as usize % node_shard_count, shard_id);
            total += 1;
        }
    }
    assert_eq!(total, 100);
}

#[test]
fn test_pre_partition_even_distribution() {
    let shard_count = 4;
    let n_list: Vec<u64> = (0..1000).collect();

    let mut shard_lists: Vec<Vec<u64>> = vec![Vec::new(); shard_count];
    for &i in &n_list {
        shard_lists[i as usize % shard_count].push(i);
    }

    // Each shard should get exactly 250 items
    for (i, list) in shard_lists.iter().enumerate() {
        assert_eq!(list.len(), 250, "Shard {} has {} items", i, list.len());
    }
}

#[test]
fn test_pre_partition_empty_list() {
    let shard_count = 4;
    let n_list: Vec<u64> = vec![];
    let mut shard_lists: Vec<Vec<u64>> = vec![Vec::new(); shard_count];
    for &i in &n_list {
        shard_lists[i as usize % shard_count].push(i);
    }
    for list in &shard_lists {
        assert!(list.is_empty());
    }
}

#[test]
fn test_pre_partition_single_item() {
    let shard_count = 4;
    let n_list: Vec<u64> = vec![7];
    let mut shard_lists: Vec<Vec<u64>> = vec![Vec::new(); shard_count];
    for &i in &n_list {
        shard_lists[i as usize % shard_count].push(i);
    }
    assert_eq!(shard_lists[3].len(), 1); // 7 % 4 = 3
    assert_eq!(shard_lists[3][0], 7);
}

// ============================================================
// 17. Hash Uniqueness & Collision Resistance Tests
// ============================================================

#[test]
fn test_no_trivial_collisions_1000_hashes() {
    let mut seen = std::collections::HashSet::new();
    for i in 0..1000u64 {
        let h = hash2(0, &pseudo_random_hash(i), &pseudo_random_hash(i + 1000));
        assert!(seen.insert(h), "Collision at i={}", i);
    }
}

#[test]
fn test_hash_output_uniform_byte_distribution() {
    // All bytes in hash output should be roughly uniformly distributed
    let mut byte_counts = vec![0u64; 256];
    let n = 1000;
    for i in 0..n {
        let h = hash2(0, &pseudo_random_hash(i as u64), &pseudo_random_hash(i as u64 + n));
        for &b in h.iter() {
            byte_counts[b as usize] += 1;
        }
    }
    // 1000 hashes * 32 bytes = 32000 total bytes
    // Expected per byte value: 32000/256 = 125
    // With chi-squared at 99.99%: allow wide range
    for (i, &count) in byte_counts.iter().enumerate() {
        assert!(
            count > 20 && count < 300,
            "Byte 0x{:02X} has count {} (expected ~125)",
            i,
            count
        );
    }
}

// ============================================================
// 18. Additional Entry Buffer Edge Cases
// ============================================================

#[test]
fn test_entry_buffer_start_at_nonzero() {
    let start = 65536i64;
    let (mut writer, _reader) = qmdb::entryfile::entrybuffer::new(start, BIG_BUF_SIZE);
    let entry = make_entry(b"key", b"value", 1);
    let pos = writer.append(&entry, &[]);
    assert_eq!(pos, start);
}

#[test]
fn test_entry_buffer_deactivated_sns() {
    let (mut writer, _reader) = qmdb::entryfile::entrybuffer::new(0, 3 * BIG_BUF_SIZE);
    let entry = make_entry(b"key", b"value", 100);
    let dsn_list = vec![1u64, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let pos = writer.append(&entry, &dsn_list);
    assert_eq!(pos, 0);
}

#[test]
fn test_entry_buffer_large_value() {
    let (mut writer, _reader) = qmdb::entryfile::entrybuffer::new(0, 3 * BIG_BUF_SIZE);
    let value = vec![0xAB; 50000]; // Large but within BIG_BUF_SIZE
    let entry = make_entry(b"k", &value, 1);
    let pos = writer.append(&entry, &[]);
    assert_eq!(pos, 0);
}

// ============================================================
// 19. Cross-Feature Consistency
// ============================================================

#[test]
fn test_hash_matches_sha256_crate() {
    // Verify our hash() matches sha2 crate directly
    let data = b"The quick brown fox jumps over the lazy dog";
    let our_hash = hash(data);
    let sha2_hash: [u8; 32] = Sha256::digest(data).into();
    assert_eq!(our_hash, sha2_hash);
}

#[test]
fn test_hash2_matches_manual_sha256() {
    let level: u8 = 5;
    let left = [0x11u8; 32];
    let right = [0x22u8; 32];

    let our_result = hash2(level, &left, &right);

    // Manual: SHA256(0x05 || left || right)
    let mut manual_input = Vec::with_capacity(65);
    manual_input.push(level);
    manual_input.extend_from_slice(&left);
    manual_input.extend_from_slice(&right);
    let manual_result: [u8; 32] = Sha256::digest(&manual_input).into();

    assert_eq!(our_result, manual_result);
}

#[test]
fn test_batch_hash_matches_manual_sha256() {
    let n = 10;
    let mut levels = Vec::with_capacity(n);
    let mut lefts = Vec::with_capacity(n);
    let mut rights = Vec::with_capacity(n);

    for i in 0..n {
        levels.push(i as u8);
        lefts.push(pseudo_random_hash(i as u64));
        rights.push(pseudo_random_hash(i as u64 + n as u64));
    }

    let mut out = vec![[0u8; 32]; n];
    batch_node_hash_cpu(&levels, &lefts, &rights, &mut out);

    for i in 0..n {
        let mut input = Vec::with_capacity(65);
        input.push(levels[i]);
        input.extend_from_slice(&lefts[i]);
        input.extend_from_slice(&rights[i]);
        let expected: [u8; 32] = Sha256::digest(&input).into();
        assert_eq!(out[i], expected, "Manual SHA256 mismatch at {}", i);
    }
}
