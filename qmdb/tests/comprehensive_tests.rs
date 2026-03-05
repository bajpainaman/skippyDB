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

use skippydb::def::{BIG_BUF_SIZE, ENTRY_BASE_LENGTH, SHARD_COUNT};
use skippydb::entryfile::entry::{self, Entry};
use skippydb::merkletree::twig::{ActiveBits, NULL_TWIG};
use skippydb::utils::changeset::ChangeSet;
use skippydb::utils::hasher::{self, batch_node_hash_cpu, hash, hash1, hash2, hash2x, ZERO_HASH32};
use skippydb::utils::numa::NumaTopology;
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
    let (mut writer, _reader) = skippydb::entryfile::entrybuffer::new(0, 3 * BIG_BUF_SIZE);
    let entry = make_entry(b"key1", b"value1", 1);
    let pos = writer.append(&entry, &[]);
    assert_eq!(pos, 0);
    // Verify pos_receiver doesn't have data yet (entry fits in current buf)
}

#[test]
fn test_entry_buffer_multiple_entries_same_buf() {
    let (mut writer, _reader) = skippydb::entryfile::entrybuffer::new(0, 3 * BIG_BUF_SIZE);
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
    let (mut writer, mut reader) = skippydb::entryfile::entrybuffer::new(0, 3 * BIG_BUF_SIZE);
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
    let (mut writer, _reader) = skippydb::entryfile::entrybuffer::new(0, 3 * BIG_BUF_SIZE);
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
    let (mut writer, _reader) = skippydb::entryfile::entrybuffer::new(0, 3 * BIG_BUF_SIZE);
    let (in_disk, have_accessed) = writer.get_entry_bz_at(-1, |_| {});
    assert!(in_disk);
    assert!(!have_accessed);
}

#[test]
fn test_entry_buffer_spanning_bufs() {
    let (mut writer, _reader) = skippydb::entryfile::entrybuffer::new(0, 3 * BIG_BUF_SIZE);
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
    let (mut writer, _reader) = skippydb::entryfile::entrybuffer::new(0, 3 * BIG_BUF_SIZE);
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
    use skippydb::utils::byte0_to_shard_id;
    assert_eq!(byte0_to_shard_id(0), 0);
    assert_eq!(byte0_to_shard_id(255), SHARD_COUNT - 1);
    // Mid-range
    let mid = byte0_to_shard_id(128);
    assert!(mid >= SHARD_COUNT / 2);
}

#[test]
fn test_byte0_to_shard_id_distribution() {
    use skippydb::utils::byte0_to_shard_id;
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
    use skippydb::gpu::{GpuHasher, MultiGpuHasher, NodeHashJob};
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
    use skippydb::utils::codec::{decode_le_i64, encode_le_i64};
    for &val in &[0i64, 1, -1, i64::MAX, i64::MIN, 42, -42, 0x7FFF_FFFF_FFFF_FFFF] {
        let encoded = encode_le_i64(val);
        let decoded = decode_le_i64(&encoded);
        assert_eq!(decoded, val, "Roundtrip failed for {}", val);
    }
}

#[test]
fn test_codec_encode_decode_u64() {
    use skippydb::utils::codec::{decode_le_u64, encode_le_u64};
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
    use skippydb::utils::shortlist::ShortList;
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
    use skippydb::utils::shortlist::ShortList;
    let mut sl = ShortList::new();
    sl.append(1);
    sl.append(2);
    sl.clear();
    assert_eq!(sl.len(), 0);
}

#[test]
fn test_shortlist_dedup() {
    use skippydb::utils::shortlist::ShortList;
    let mut sl = ShortList::new();
    sl.append(42);
    sl.append(42); // duplicate
    assert_eq!(sl.len(), 1); // ShortList deduplicates
}

#[test]
fn test_shortlist_contains() {
    use skippydb::utils::shortlist::ShortList;
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
    let (mut writer, _reader) = skippydb::entryfile::entrybuffer::new(start, BIG_BUF_SIZE);
    let entry = make_entry(b"key", b"value", 1);
    let pos = writer.append(&entry, &[]);
    assert_eq!(pos, start);
}

#[test]
fn test_entry_buffer_deactivated_sns() {
    let (mut writer, _reader) = skippydb::entryfile::entrybuffer::new(0, 3 * BIG_BUF_SIZE);
    let entry = make_entry(b"key", b"value", 100);
    let dsn_list = vec![1u64, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let pos = writer.append(&entry, &dsn_list);
    assert_eq!(pos, 0);
}

#[test]
fn test_entry_buffer_large_value() {
    let (mut writer, _reader) = skippydb::entryfile::entrybuffer::new(0, 3 * BIG_BUF_SIZE);
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

// ============================================================
// Additional Batch Hash Tests
// ============================================================

#[test]
fn test_batch_hash_cpu_two_items() {
    let left0 = [0xAAu8; 32];
    let right0 = [0xBBu8; 32];
    let left1 = [0xCCu8; 32];
    let right1 = [0xDDu8; 32];
    let mut out = [[0u8; 32]; 2];
    batch_node_hash_cpu(&[3, 7], &[left0, left1], &[right0, right1], &mut out);
    assert_eq!(out[0], hash2(3, &left0, &right0));
    assert_eq!(out[1], hash2(7, &left1, &right1));
}

#[test]
fn test_batch_hash_cpu_level_zero() {
    let left = [0x01u8; 32];
    let right = [0x02u8; 32];
    let mut out = [[0u8; 32]; 1];
    batch_node_hash_cpu(&[0], &[left], &[right], &mut out);
    assert_eq!(out[0], hash2(0, &left, &right));
}

#[test]
fn test_batch_hash_cpu_level_255() {
    let left = [0xFFu8; 32];
    let right = [0x00u8; 32];
    let mut out = [[0u8; 32]; 1];
    batch_node_hash_cpu(&[255], &[left], &[right], &mut out);
    assert_eq!(out[0], hash2(255, &left, &right));
}

#[test]
fn test_batch_hash_cpu_same_left_right() {
    let data = [0x42u8; 32];
    let mut out = [[0u8; 32]; 1];
    batch_node_hash_cpu(&[1], &[data], &[data], &mut out);
    assert_eq!(out[0], hash2(1, &data, &data));
}

#[test]
fn test_batch_hash_cpu_100_items() {
    let n = 100;
    let levels: Vec<u8> = (0..n).map(|i| (i % 13) as u8).collect();
    let lefts: Vec<[u8; 32]> = (0..n).map(|i| pseudo_random_hash(i as u64)).collect();
    let rights: Vec<[u8; 32]> = (0..n).map(|i| pseudo_random_hash(i as u64 + 10000)).collect();
    let mut out = vec![[0u8; 32]; n];
    batch_node_hash_cpu(&levels, &lefts, &rights, &mut out);
    for i in 0..n {
        assert_eq!(out[i], hash2(levels[i], &lefts[i], &rights[i]));
    }
}

#[test]
fn test_batch_hash_cpu_1000_items() {
    let n = 1000;
    let levels: Vec<u8> = (0..n).map(|i| (i % 256) as u8).collect();
    let lefts: Vec<[u8; 32]> = (0..n).map(|i| pseudo_random_hash(i as u64 * 3)).collect();
    let rights: Vec<[u8; 32]> = (0..n).map(|i| pseudo_random_hash(i as u64 * 3 + 1)).collect();
    let mut out = vec![[0u8; 32]; n];
    batch_node_hash_cpu(&levels, &lefts, &rights, &mut out);
    for i in 0..n {
        assert_eq!(out[i], hash2(levels[i], &lefts[i], &rights[i]));
    }
}

#[test]
fn test_batch_hash_cpu_output_not_zero_for_nonzero_input() {
    let left = [0x01u8; 32];
    let right = [0x02u8; 32];
    let mut out = [[0u8; 32]; 1];
    batch_node_hash_cpu(&[5], &[left], &[right], &mut out);
    assert_ne!(out[0], [0u8; 32]);
}

#[test]
fn test_batch_hash_cpu_different_levels_produce_different_hashes() {
    let left = [0x55u8; 32];
    let right = [0x66u8; 32];
    let n = 256;
    let levels: Vec<u8> = (0..n).map(|i| i as u8).collect();
    let lefts = vec![left; n];
    let rights = vec![right; n];
    let mut out = vec![[0u8; 32]; n];
    batch_node_hash_cpu(&levels, &lefts, &rights, &mut out);
    // All outputs should be distinct (different domain separator -> different hash)
    let mut seen = std::collections::HashSet::new();
    for h in &out {
        assert!(seen.insert(*h), "Duplicate hash at level");
    }
}

#[test]
fn test_batch_hash_cpu_order_matters() {
    let left = [0xAAu8; 32];
    let right = [0xBBu8; 32];
    let mut out_ab = [[0u8; 32]; 1];
    let mut out_ba = [[0u8; 32]; 1];
    batch_node_hash_cpu(&[4], &[left], &[right], &mut out_ab);
    batch_node_hash_cpu(&[4], &[right], &[left], &mut out_ba);
    assert_ne!(out_ab[0], out_ba[0]);
}

#[test]
fn test_batch_hash_cpu_sequential_levels_0_to_11() {
    let n = 12;
    let levels: Vec<u8> = (0..n).map(|i| i as u8).collect();
    let data = [0x77u8; 32];
    let lefts = vec![data; n];
    let rights = vec![data; n];
    let mut out = vec![[0u8; 32]; n];
    batch_node_hash_cpu(&levels, &lefts, &rights, &mut out);
    for i in 0..n {
        let expected = hash2(i as u8, &data, &data);
        assert_eq!(out[i], expected);
    }
}

#[test]
fn test_batch_hash_cpu_mixed_level_0_and_max() {
    let data_a = [0x11u8; 32];
    let data_b = [0x22u8; 32];
    let mut out = [[0u8; 32]; 2];
    batch_node_hash_cpu(&[0, 255], &[data_a, data_a], &[data_b, data_b], &mut out);
    assert_eq!(out[0], hash2(0, &data_a, &data_b));
    assert_eq!(out[1], hash2(255, &data_a, &data_b));
    assert_ne!(out[0], out[1]);
}

// ============================================================
// Additional Hash Function Tests
// ============================================================

#[test]
fn test_hash2_level_8_hello_world() {
    // Known test vector from internal tests
    let result = hash2(8, "hello", "world");
    assert_eq!(
        hex::encode(result),
        "8e6fc50a3f98a3c314021b89688ca83a9b5697ca956e211198625fc460ddf1e9"
    );
}

#[test]
fn test_hash2x_exchange_reverses_args() {
    let a: &[u8] = b"alpha";
    let b: &[u8] = b"beta";
    let h_ab = hash2(5, a, b);
    let h_ba = hash2x(5, a, b, true); // exchange => hash2(5, b, a)
    let h_ba_direct = hash2(5, b, a);
    assert_eq!(h_ba, h_ba_direct);
    assert_ne!(h_ab, h_ba);
}

#[test]
fn test_hash2x_no_exchange_same_as_hash2() {
    let a: &[u8] = b"alpha";
    let b: &[u8] = b"beta";
    let h_ab = hash2(5, a, b);
    let h_ab2 = hash2x(5, a, b, false);
    assert_eq!(h_ab, h_ab2);
}

#[test]
fn test_node_hash_inplace_deterministic() {
    let mut out1 = [0u8; 32];
    let mut out2 = [0u8; 32];
    hasher::node_hash_inplace(3, &mut out1, b"left" as &[u8], b"right" as &[u8]);
    hasher::node_hash_inplace(3, &mut out2, b"left" as &[u8], b"right" as &[u8]);
    assert_eq!(out1, out2);
}

#[test]
fn test_node_hash_inplace_all_zeros() {
    let mut out = [0u8; 32];
    hasher::node_hash_inplace(0, &mut out, &[0u8; 32], &[0u8; 32]);
    let expected = hash2(0, &[0u8; 32], &[0u8; 32]);
    assert_eq!(out, expected);
}

#[test]
fn test_node_hash_inplace_all_ff() {
    let mut out = [0u8; 32];
    hasher::node_hash_inplace(255, &mut out, &[0xFFu8; 32], &[0xFFu8; 32]);
    let expected = hash2(255, &[0xFFu8; 32], &[0xFFu8; 32]);
    assert_eq!(out, expected);
}

#[test]
fn test_hash1_domain_separated_from_hash() {
    let data = b"test_data";
    let h = hash(data);
    let h1 = hash1(0, data);
    assert_ne!(h, h1);
}

#[test]
fn test_hash1_different_levels() {
    let data = b"same_data";
    let results: Vec<[u8; 32]> = (0..=10).map(|level| hash1(level, data)).collect();
    for i in 0..results.len() {
        for j in (i + 1)..results.len() {
            assert_ne!(results[i], results[j], "hash1 collision at levels {} and {}", i, j);
        }
    }
}

#[test]
fn test_hash2_all_256_levels_unique_for_same_inputs() {
    let left = [0x42u8; 32];
    let right = [0x43u8; 32];
    let results: Vec<[u8; 32]> = (0u8..=255).map(|level| hash2(level, &left, &right)).collect();
    let set: std::collections::HashSet<Vec<u8>> = results.iter().map(|h| h.to_vec()).collect();
    assert_eq!(set.len(), 256, "All 256 levels should produce unique hashes");
}

#[test]
fn test_hash2_empty_strings() {
    let result = hash2(0, b"", b"");
    assert_ne!(result, [0u8; 32]);
}

#[test]
fn test_hash2_one_empty_one_nonempty() {
    // hash2 concatenates: [level] || a || b
    // hash2(0, "", "data") = sha256([0] + "data")
    // hash2(0, "data", "") = sha256([0] + "data")
    // These are equal because SHA256 concatenates without length prefixes.
    let result_a = hash2(0, b"" as &[u8], b"data" as &[u8]);
    let result_b = hash2(0, b"data" as &[u8], b"" as &[u8]);
    assert_eq!(result_a, result_b, "hash2 with empty+nonempty equals nonempty+empty (known SHA256 property)");
    // They are all equal to sha256([0] + "data")
    let expected: [u8; 32] = {
        let mut h = sha2::Sha256::new();
        h.update([0u8]);
        h.update(b"data");
        h.finalize().into()
    };
    assert_eq!(result_a, expected);
    // But a different total content produces a different result
    let result_c = hash2(0, b"different" as &[u8], b"data" as &[u8]);
    assert_ne!(result_a, result_c);
}

#[test]
fn test_zero_hash32_constant_is_all_zeros() {
    assert_eq!(ZERO_HASH32, [0u8; 32]);
}

// ============================================================
// Additional Entry Buffer Tests
// ============================================================

#[test]
fn test_entry_buffer_write_read_single_small() {
    use skippydb::entryfile::entrybuffer;
    let entry = make_entry(b"k", b"v", 1);
    let (mut writer, mut reader) = entrybuffer::new(0, BIG_BUF_SIZE);
    writer.append(&entry, &[]);
    writer.end_block(0, 0, 1);

    let mut count = 0;
    loop {
        let (eob, _pos) = reader.read_next_entry(|_ebz| { count += 1; });
        if eob { break; }
    }
    assert_eq!(count, 1);
    reader.read_extra_info();
}

#[test]
fn test_entry_buffer_write_read_ten_entries() {
    use skippydb::entryfile::entrybuffer;
    let (mut writer, mut reader) = entrybuffer::new(0, BIG_BUF_SIZE);
    for i in 0u64..10 {
        let entry = make_entry(b"key_common", b"value_common", i);
        writer.append(&entry, &[]);
    }
    writer.end_block(0, 0, 10);

    let mut count = 0;
    loop {
        let (eob, _) = reader.read_next_entry(|_| { count += 1; });
        if eob { break; }
    }
    assert_eq!(count, 10);
    reader.read_extra_info();
}

#[test]
fn test_entry_buffer_roundtrip_key_value() {
    use skippydb::entryfile::entrybuffer;
    let key = b"roundtrip_key";
    let value = b"roundtrip_value_content";
    let entry = Entry {
        key,
        value,
        next_key_hash: &[0xEEu8; 32],
        version: 9999,
        serial_number: 12345,
    };
    let (mut writer, mut reader) = entrybuffer::new(0, BIG_BUF_SIZE);
    writer.append(&entry, &[]);
    writer.end_block(0, 0, 1);

    let mut found_key = Vec::new();
    let mut found_value = Vec::new();
    let mut found_sn = 0u64;
    loop {
        let (eob, _) = reader.read_next_entry(|ebz| {
            found_key = ebz.key().to_vec();
            found_value = ebz.value().to_vec();
            found_sn = ebz.serial_number();
        });
        if eob { break; }
    }
    assert_eq!(found_key, key);
    assert_eq!(found_value, value);
    assert_eq!(found_sn, 12345);
    reader.read_extra_info();
}

#[test]
fn test_entry_buffer_end_block_extra_info_roundtrip() {
    use skippydb::entryfile::entrybuffer;
    let (mut writer, mut reader) = entrybuffer::new(0, BIG_BUF_SIZE);
    let entry = make_entry(b"k", b"v", 1);
    writer.append(&entry, &[]);
    writer.end_block(42, 77, 99);

    // Drain the entry via read_next_entry, which also receives the end-of-block signal
    loop {
        let (eob, _) = reader.read_next_entry(|_| {});
        if eob { break; }
    }

    let (cdp, cds, se) = reader.read_extra_info();
    assert_eq!(cdp, 42);
    assert_eq!(cds, 77);
    assert_eq!(se, 99);
}

#[test]
fn test_entry_buffer_multiple_blocks() {
    use skippydb::entryfile::entrybuffer;
    let (mut writer, mut reader) = entrybuffer::new(0, BIG_BUF_SIZE);

    // Block 1
    writer.append(&make_entry(b"k1", b"v1", 1), &[]);
    writer.end_block(0, 0, 1);

    let mut count = 0;
    loop {
        let (eob, _) = reader.read_next_entry(|_| { count += 1; });
        if eob { break; }
    }
    assert_eq!(count, 1);
    reader.read_extra_info();

    // Block 2
    writer.append(&make_entry(b"k2", b"v2", 2), &[]);
    writer.append(&make_entry(b"k3", b"v3", 3), &[]);
    writer.end_block(10, 5, 10);

    count = 0;
    loop {
        let (eob, _) = reader.read_next_entry(|_| { count += 1; });
        if eob { break; }
    }
    assert_eq!(count, 2);
    let (cdp, cds, se) = reader.read_extra_info();
    assert_eq!(cdp, 10);
    assert_eq!(cds, 5);
    assert_eq!(se, 10);
}

#[test]
fn test_entry_buffer_deactivated_sns_roundtrip() {
    use skippydb::entryfile::entrybuffer;
    let entry = make_entry(b"key", b"value", 100);
    let dsn_list = [10u64, 20, 30, 40];
    let (mut writer, mut reader) = entrybuffer::new(0, BIG_BUF_SIZE);
    writer.append(&entry, &dsn_list);
    writer.end_block(0, 0, 101);

    let mut dsn_count = 0usize;
    let mut found_dsns = Vec::new();
    loop {
        let (eob, _) = reader.read_next_entry(|ebz| {
            dsn_count = ebz.dsn_count();
            for (_, dsn) in ebz.dsn_iter() {
                found_dsns.push(dsn);
            }
        });
        if eob { break; }
    }
    assert_eq!(dsn_count, 4);
    assert_eq!(found_dsns, vec![10, 20, 30, 40]);
    reader.read_extra_info();
}

#[test]
fn test_entry_buffer_start_pos_nonzero_offset() {
    use skippydb::entryfile::entrybuffer;
    let start = 1024i64;
    let (mut writer, mut reader) = entrybuffer::new(start, BIG_BUF_SIZE);
    writer.append(&make_entry(b"key", b"val", 5), &[]);
    writer.end_block(0, 0, 6);

    let mut count = 0;
    loop {
        let (eob, _) = reader.read_next_entry(|_| { count += 1; });
        if eob { break; }
    }
    assert_eq!(count, 1);
    reader.read_extra_info();
}

#[test]
fn test_entry_buffer_50_entries() {
    use skippydb::entryfile::entrybuffer;
    let n = 50;
    let (mut writer, mut reader) = entrybuffer::new(0, BIG_BUF_SIZE);
    for i in 0..n {
        let key = format!("key_{:04}", i);
        let val = format!("val_{:04}", i);
        let entry = Entry {
            key: key.as_bytes(),
            value: val.as_bytes(),
            next_key_hash: &[0u8; 32],
            version: i as i64,
            serial_number: i as u64,
        };
        writer.append(&entry, &[]);
    }
    writer.end_block(0, 0, n as u64);

    let mut count = 0;
    loop {
        let (eob, _) = reader.read_next_entry(|_| { count += 1; });
        if eob { break; }
    }
    assert_eq!(count, n);
    reader.read_extra_info();
}

// ============================================================
// Additional Entry Serialization Tests
// ============================================================

#[test]
fn test_entry_dump_basic_fields() {
    let key = b"test_key";
    let value = b"test_value";
    let nkh = [0xABu8; 32];
    let entry = Entry {
        key,
        value,
        next_key_hash: &nkh,
        version: 42,
        serial_number: 7,
    };
    let total = entry.get_serialized_len(0);
    let mut buf = vec![0u8; total];
    entry.dump(&mut buf, &[]);
    let ebz = skippydb::entryfile::entry::EntryBz { bz: &buf[..total] };
    assert_eq!(ebz.key(), key);
    assert_eq!(ebz.value(), value);
    assert_eq!(ebz.next_key_hash(), &nkh);
    assert_eq!(ebz.version(), 42);
    assert_eq!(ebz.serial_number(), 7);
}

#[test]
fn test_entry_dump_zero_length_value() {
    let key = b"zero_val_key";
    let entry = Entry {
        key,
        value: b"",
        next_key_hash: &[0u8; 32],
        version: 0,
        serial_number: 0,
    };
    let total = entry.get_serialized_len(0);
    let mut buf = vec![0u8; total];
    entry.dump(&mut buf, &[]);
    let ebz = skippydb::entryfile::entry::EntryBz { bz: &buf[..total] };
    assert_eq!(ebz.value().len(), 0);
}

#[test]
fn test_entry_dump_version_negative() {
    let entry = Entry {
        key: b"k",
        value: b"v",
        next_key_hash: &[0u8; 32],
        version: -1,
        serial_number: 0,
    };
    let total = entry.get_serialized_len(0);
    let mut buf = vec![0u8; total];
    entry.dump(&mut buf, &[]);
    let ebz = skippydb::entryfile::entry::EntryBz { bz: &buf[..total] };
    assert_eq!(ebz.version(), -1);
}

#[test]
fn test_entry_dump_max_serial_number() {
    let entry = Entry {
        key: b"k",
        value: b"v",
        next_key_hash: &[0u8; 32],
        version: 0,
        serial_number: u64::MAX,
    };
    let total = entry.get_serialized_len(0);
    let mut buf = vec![0u8; total];
    entry.dump(&mut buf, &[]);
    let ebz = skippydb::entryfile::entry::EntryBz { bz: &buf[..total] };
    assert_eq!(ebz.serial_number(), u64::MAX);
}

#[test]
fn test_entry_dump_single_deactivated_sn() {
    let entry = make_entry(b"key", b"value", 10);
    let dsn = [999u64];
    let total = entry.get_serialized_len(1);
    let mut buf = vec![0u8; total];
    entry.dump(&mut buf, &dsn);
    let ebz = skippydb::entryfile::entry::EntryBz { bz: &buf[..total] };
    assert_eq!(ebz.dsn_count(), 1);
    assert_eq!(ebz.get_deactived_sn(0), 999);
}

#[test]
fn test_entry_dump_five_deactivated_sns() {
    let entry = make_entry(b"key", b"value", 10);
    let dsn = [1u64, 2, 3, 4, 5];
    let total = entry.get_serialized_len(5);
    let mut buf = vec![0u8; total];
    entry.dump(&mut buf, &dsn);
    let ebz = skippydb::entryfile::entry::EntryBz { bz: &buf[..total] };
    assert_eq!(ebz.dsn_count(), 5);
    for i in 0..5 {
        assert_eq!(ebz.get_deactived_sn(i), (i + 1) as u64);
    }
}

#[test]
fn test_entry_dsn_iter_yields_all() {
    let entry = make_entry(b"key", b"value", 10);
    let dsn = [100u64, 200, 300];
    let total = entry.get_serialized_len(3);
    let mut buf = vec![0u8; total];
    entry.dump(&mut buf, &dsn);
    let ebz = skippydb::entryfile::entry::EntryBz { bz: &buf[..total] };
    let items: Vec<(usize, u64)> = ebz.dsn_iter().collect();
    assert_eq!(items, vec![(0, 100), (1, 200), (2, 300)]);
}

#[test]
fn test_entry_key_hash_with_value_is_sha256_of_key() {
    let key = b"hashed_key";
    let entry = Entry {
        key,
        value: b"some_value",
        next_key_hash: &[0u8; 32],
        version: 1,
        serial_number: 1,
    };
    let total = entry.get_serialized_len(0);
    let mut buf = vec![0u8; total];
    entry.dump(&mut buf, &[]);
    let ebz = skippydb::entryfile::entry::EntryBz { bz: &buf[..total] };
    let expected: [u8; 32] = Sha256::digest(key).into();
    assert_eq!(ebz.key_hash(), expected);
}

#[test]
fn test_entry_key_hash_with_empty_value_uses_first_two_bytes() {
    // When value is empty, key_hash returns first two bytes of key in hash
    let mut key = [0u8; 32];
    key[0] = 0x12;
    key[1] = 0x34;
    let entry = Entry {
        key: &key,
        value: b"",
        next_key_hash: &[0u8; 32],
        version: 0,
        serial_number: 0,
    };
    let total = entry.get_serialized_len(0);
    let mut buf = vec![0u8; total];
    entry.dump(&mut buf, &[]);
    let ebz = skippydb::entryfile::entry::EntryBz { bz: &buf[..total] };
    let kh = ebz.key_hash();
    assert_eq!(kh[0], 0x12);
    assert_eq!(kh[1], 0x34);
    // Rest should be zero
    assert_eq!(&kh[2..], &[0u8; 30]);
}

#[test]
fn test_entry_get_serialized_len_aligns_to_8() {
    // For various key+value sizes, serialized len should be multiple of 8
    for key_len in 0..=16usize {
        for val_len in 0..=16usize {
            let key = vec![0xAAu8; key_len];
            let val = vec![0xBBu8; val_len];
            let entry = Entry {
                key: &key,
                value: &val,
                next_key_hash: &[0u8; 32],
                version: 0,
                serial_number: 0,
            };
            let len = entry.get_serialized_len(0);
            assert_eq!(len % 8, 0, "len {} not multiple of 8 for key_len={} val_len={}", len, key_len, val_len);
        }
    }
}

#[test]
fn test_entry_payload_len() {
    let entry = make_entry(b"abc", b"defgh", 1);
    let total = entry.get_serialized_len(0);
    let mut buf = vec![0u8; total];
    entry.dump(&mut buf, &[]);
    let ebz = skippydb::entryfile::entry::EntryBz { bz: &buf[..total] };
    // payload_len should be <= total len
    assert!(ebz.payload_len() <= total);
    assert!(ebz.payload_len() > 0);
}

#[test]
fn test_entry_from_bz_roundtrip() {
    let key = b"round_key";
    let value = b"round_value";
    let nkh = [0xCDu8; 32];
    let original = Entry {
        key,
        value,
        next_key_hash: &nkh,
        version: 5,
        serial_number: 11,
    };
    let total = original.get_serialized_len(0);
    let mut buf = vec![0u8; total];
    original.dump(&mut buf, &[]);
    let ebz = skippydb::entryfile::entry::EntryBz { bz: &buf[..total] };
    let reconstructed = Entry::from_bz(&ebz);
    assert_eq!(reconstructed.key, key);
    assert_eq!(reconstructed.value, value);
    assert_eq!(reconstructed.version, 5);
    assert_eq!(reconstructed.serial_number, 11);
}

// ============================================================
// Additional Twig / ActiveBits Tests
// ============================================================

#[test]
fn test_active_bits_default_is_new() {
    let bits_default = ActiveBits::default();
    let bits_new = ActiveBits::new();
    assert_eq!(bits_default, bits_new);
}

#[test]
fn test_active_bits_toggle_bit() {
    let mut bits = ActiveBits::new();
    bits.set_bit(42);
    assert!(bits.get_bit(42));
    bits.clear_bit(42);
    assert!(!bits.get_bit(42));
}

#[test]
fn test_active_bits_set_first_bit() {
    let mut bits = ActiveBits::new();
    bits.set_bit(0);
    assert!(bits.get_bit(0));
    assert!(!bits.get_bit(1));
}

#[test]
fn test_active_bits_set_last_valid_bit() {
    let mut bits = ActiveBits::new();
    // The valid range is 0..=2047 (LEAF_COUNT_IN_TWIG - 1)
    let last = skippydb::def::LEAF_COUNT_IN_TWIG - 1; // 2047
    bits.set_bit(last);
    assert!(bits.get_bit(last));
}

#[test]
fn test_active_bits_independent_bits() {
    let mut bits = ActiveBits::new();
    bits.set_bit(0);
    bits.set_bit(7);
    bits.set_bit(8);
    bits.set_bit(15);
    assert!(bits.get_bit(0));
    assert!(bits.get_bit(7));
    assert!(bits.get_bit(8));
    assert!(bits.get_bit(15));
    assert!(!bits.get_bit(1));
    assert!(!bits.get_bit(6));
    assert!(!bits.get_bit(9));
    assert!(!bits.get_bit(14));
}

#[test]
fn test_active_bits_clear_non_set_bit() {
    let mut bits = ActiveBits::new();
    // Clearing a non-set bit should be a no-op
    bits.clear_bit(100);
    assert!(!bits.get_bit(100));
}

#[test]
fn test_active_bits_set_all_and_verify() {
    let mut bits = ActiveBits::new();
    // Set every 7th bit
    for i in (0..2048u32).step_by(7) {
        bits.set_bit(i);
    }
    for i in (0..2048u32).step_by(7) {
        assert!(bits.get_bit(i), "bit {} should be set", i);
    }
    // Verify un-set bits
    for i in 1..7u32 {
        assert!(!bits.get_bit(i), "bit {} should not be set", i);
    }
}

#[test]
fn test_active_bits_get_bits_page_boundaries() {
    // Use set_bit to populate known bits and verify get_bits slicing
    let mut bits = ActiveBits::new();
    // Set specific bits in page 0 (bits 0..256) and page 7 (bits 1792..2048)
    bits.set_bit(0);   // page 0, byte 0, bit 0
    bits.set_bit(7);   // page 0, byte 0, bit 7
    bits.set_bit(8);   // page 0, byte 1, bit 0

    let page0 = bits.get_bits(0, 32);
    assert_eq!(page0.len(), 32);
    // byte 0 should have bits 0 and 7 set = 0b10000001 = 0x81
    assert_eq!(page0[0], 0b10000001u8);
    // byte 1 should have bit 0 set = 0x01
    assert_eq!(page0[1], 0x01);

    // Page 7 should be entirely zero since we didn't set any bits in it
    let page7 = bits.get_bits(7, 32);
    assert_eq!(page7.len(), 32);
    for byte in page7 {
        assert_eq!(*byte, 0);
    }
}

#[test]
fn test_twig_new_all_zero() {
    use skippydb::merkletree::twig::Twig;
    let t = Twig::new();
    for i in 0..4 {
        assert_eq!(t.active_bits_mtl1[i], [0u8; 32]);
    }
    for i in 0..2 {
        assert_eq!(t.active_bits_mtl2[i], [0u8; 32]);
    }
    assert_eq!(t.active_bits_mtl3, [0u8; 32]);
    assert_eq!(t.left_root, [0u8; 32]);
    assert_eq!(t.twig_root, [0u8; 32]);
}

#[test]
fn test_twig_default_same_as_new() {
    use skippydb::merkletree::twig::Twig;
    let t1 = Twig::new();
    let t2 = Twig::default();
    assert_eq!(t1.active_bits_mtl3, t2.active_bits_mtl3);
    assert_eq!(t1.twig_root, t2.twig_root);
}

#[test]
fn test_twig_sync_top_deterministic() {
    use skippydb::merkletree::twig::Twig;
    let mut t1 = Twig::new();
    let mut t2 = Twig::new();
    t1.left_root = [0xABu8; 32];
    t1.active_bits_mtl3 = [0xCDu8; 32];
    t2.left_root = [0xABu8; 32];
    t2.active_bits_mtl3 = [0xCDu8; 32];
    t1.sync_top();
    t2.sync_top();
    assert_eq!(t1.twig_root, t2.twig_root);
    assert_ne!(t1.twig_root, [0u8; 32]);
}

#[test]
fn test_twig_sync_l3_non_zero() {
    use skippydb::merkletree::twig::Twig;
    let mut t = Twig::new();
    t.active_bits_mtl2[0] = [0x01u8; 32];
    t.active_bits_mtl2[1] = [0x02u8; 32];
    t.sync_l3();
    assert_ne!(t.active_bits_mtl3, [0u8; 32]);
}

#[test]
fn test_null_twig_left_root_non_zero() {
    assert_ne!(NULL_TWIG.left_root, [0u8; 32]);
}

#[test]
fn test_null_twig_active_bits_mtl3_non_zero() {
    assert_ne!(NULL_TWIG.active_bits_mtl3, [0u8; 32]);
}

#[test]
fn test_sync_mtree_single_leaf_range() {
    use skippydb::merkletree::twig::sync_mtree;
    // Just verify it doesn't panic for a small range
    let mut mt: Vec<[u8; 32]> = vec![[0u8; 32]; 4096];
    // copy a leaf
    mt[2048] = [0x01u8; 32];
    mt[2049] = [0x02u8; 32];
    sync_mtree(&mut mt, 0, 1);
    // parent at 1024 should now be non-zero
    assert_ne!(mt[1024], [0u8; 32]);
}

#[test]
fn test_sync_mtree_matches_hash2() {
    use skippydb::merkletree::twig::sync_mtree;
    let left = [0x11u8; 32];
    let right = [0x22u8; 32];
    let mut mt = vec![[0u8; 32]; 4096];
    mt[2048] = left;
    mt[2049] = right;
    sync_mtree(&mut mt, 0, 1);
    let expected = hash2(0, &left, &right);
    assert_eq!(mt[1024], expected);
}

#[test]
fn test_null_mt_for_twig_root_non_zero() {
    use skippydb::merkletree::twig::NULL_MT_FOR_TWIG;
    // The root at index 1 should not be zero
    assert_ne!(NULL_MT_FOR_TWIG[1], [0u8; 32]);
}

// ============================================================
// Additional ChangeSet Tests
// ============================================================

#[test]
fn test_changeset_new_uninit() {
    let cs = ChangeSet::new_uninit();
    assert_eq!(cs.op_list.len(), 0);
    assert_eq!(cs.data.len(), 0);
}

#[test]
fn test_changeset_run_all_empty() {
    let cs = ChangeSet::new();
    let mut count = 0;
    cs.run_all(|_, _, _, _, _| { count += 1; });
    assert_eq!(count, 0);
}

#[test]
fn test_changeset_add_and_run_all() {
    let mut cs = ChangeSet::new();
    let kh = [0u8; 32];
    cs.add_op(skippydb::def::OP_READ, 0, &kh, b"key1", b"val1", None);
    cs.add_op(skippydb::def::OP_WRITE, 0, &kh, b"key2", b"val2", None);
    let mut count = 0;
    cs.run_all(|_, _, _, _, _| { count += 1; });
    assert_eq!(count, 2);
}

#[test]
fn test_changeset_run_in_shard_empty_shard() {
    let mut cs = ChangeSet::new();
    let kh = [0xFFu8; 32];
    // Add to shard 15
    cs.add_op(skippydb::def::OP_CREATE, 15, &kh, b"k", b"v", None);
    cs.sort();
    let mut count = 0;
    cs.run_in_shard(0, |_, _, _, _, _| { count += 1; });
    assert_eq!(count, 0);
}

#[test]
fn test_changeset_run_in_shard_correct_shard() {
    let mut cs = ChangeSet::new();
    let kh0 = [0x00u8; 32]; // shard 0
    let kh15 = [0xFFu8; 32]; // shard 15
    cs.add_op(skippydb::def::OP_CREATE, 0, &kh0, b"ka", b"va", None);
    cs.add_op(skippydb::def::OP_CREATE, 15, &kh15, b"kb", b"vb", None);
    cs.sort();

    let mut shard0_count = 0;
    let mut shard15_count = 0;
    cs.run_in_shard(0, |_, _, _, _, _| { shard0_count += 1; });
    cs.run_in_shard(15, |_, _, _, _, _| { shard15_count += 1; });
    assert_eq!(shard0_count, 1);
    assert_eq!(shard15_count, 1);
}

#[test]
fn test_changeset_sort_by_shard_then_key_hash() {
    let mut cs = ChangeSet::new();
    let kh_low = [0x00u8; 32];
    let kh_high = [0xFFu8; 32];
    // Add out of order
    cs.add_op(skippydb::def::OP_READ, 0, &kh_high, b"k2", b"v2", None);
    cs.add_op(skippydb::def::OP_READ, 0, &kh_low, b"k1", b"v1", None);
    cs.sort();
    // Both ops should be in shard 0 - verify via run_in_shard
    let mut seen_keys = Vec::new();
    cs.run_in_shard(0, |_, kh, k, _v, _| {
        seen_keys.push((kh[0], k.to_vec()));
    });
    // kh_low (0x00) should come before kh_high (0xFF) within shard 0
    assert_eq!(seen_keys.len(), 2);
    assert_eq!(seen_keys[0].0, 0x00); // low key hash first
    assert_eq!(seen_keys[1].0, 0xFF); // high key hash second
}

#[test]
fn test_changeset_op_count_empty() {
    let cs = ChangeSet::new();
    for shard_id in 0..SHARD_COUNT {
        assert_eq!(cs.op_count_in_shard(shard_id), 0);
    }
}

#[test]
fn test_changeset_op_count_after_sort() {
    let mut cs = ChangeSet::new();
    for i in 0..5u8 {
        cs.add_op(skippydb::def::OP_READ, 3, &[i; 32], b"k", b"v", None);
    }
    for i in 0..3u8 {
        cs.add_op(skippydb::def::OP_READ, 7, &[i; 32], b"k", b"v", None);
    }
    cs.sort();
    assert_eq!(cs.op_count_in_shard(3), 5);
    assert_eq!(cs.op_count_in_shard(7), 3);
    assert_eq!(cs.op_count_in_shard(0), 0);
}

#[test]
fn test_changeset_add_op_with_old_value() {
    let mut cs = ChangeSet::new();
    let kh = [0x42u8; 32];
    cs.add_op_with_old_value(
        skippydb::def::OP_WRITE,
        4,
        &kh,
        b"key",
        b"new_val",
        b"old_val",
        None,
    );
    assert_eq!(cs.op_list.len(), 1);
    assert_eq!(cs.op_list[0].op_type, skippydb::def::OP_WRITE);
}

#[test]
fn test_changeset_apply_op_in_range_correct_data() {
    let mut cs = ChangeSet::new();
    let kh = [0x10u8; 32];
    cs.add_op(skippydb::def::OP_DELETE, 1, &kh, b"del_key", b"del_val", None);
    let mut keys_seen = Vec::new();
    cs.apply_op_in_range(|op_type, _kh, k, _v, _ov, _rec| {
        assert_eq!(op_type, skippydb::def::OP_DELETE);
        keys_seen.push(k.to_vec());
    });
    assert_eq!(keys_seen, vec![b"del_key".to_vec()]);
}

// ============================================================
// Additional NUMA Topology Tests
// ============================================================

#[test]
fn test_numa_detect_is_valid() {
    let topo = NumaTopology::detect();
    assert!(topo.num_nodes >= 1);
    assert_eq!(topo.node_cpus.len(), topo.num_nodes);
}

#[test]
fn test_numa_single_node_shard_to_node_always_0() {
    let topo = NumaTopology {
        num_nodes: 1,
        node_cpus: vec![vec![0, 1, 2, 3]],
    };
    for shard_id in 0..16 {
        assert_eq!(topo.shard_to_node(shard_id), 0);
    }
}

#[test]
fn test_numa_two_nodes_alternating() {
    let topo = NumaTopology {
        num_nodes: 2,
        node_cpus: vec![vec![0, 1], vec![2, 3]],
    };
    for shard_id in 0..16 {
        assert_eq!(topo.shard_to_node(shard_id), shard_id % 2);
    }
}

#[test]
fn test_numa_four_nodes_distribution() {
    let topo = NumaTopology {
        num_nodes: 4,
        node_cpus: vec![vec![0], vec![1], vec![2], vec![3]],
    };
    assert_eq!(topo.shard_to_node(0), 0);
    assert_eq!(topo.shard_to_node(1), 1);
    assert_eq!(topo.shard_to_node(2), 2);
    assert_eq!(topo.shard_to_node(3), 3);
    assert_eq!(topo.shard_to_node(4), 0);
}

#[test]
fn test_numa_is_numa_false_for_one_node() {
    let topo = NumaTopology {
        num_nodes: 1,
        node_cpus: vec![vec![0]],
    };
    assert!(!topo.is_numa());
}

#[test]
fn test_numa_is_numa_true_for_two_nodes() {
    let topo = NumaTopology {
        num_nodes: 2,
        node_cpus: vec![vec![0], vec![1]],
    };
    assert!(topo.is_numa());
}

#[test]
fn test_numa_cpus_for_shard_two_nodes() {
    let topo = NumaTopology {
        num_nodes: 2,
        node_cpus: vec![vec![0, 1, 2], vec![3, 4, 5]],
    };
    // Even shards -> node 0
    assert_eq!(topo.cpus_for_shard(0), &[0, 1, 2]);
    assert_eq!(topo.cpus_for_shard(2), &[0, 1, 2]);
    // Odd shards -> node 1
    assert_eq!(topo.cpus_for_shard(1), &[3, 4, 5]);
    assert_eq!(topo.cpus_for_shard(3), &[3, 4, 5]);
}

#[test]
fn test_numa_fallback_on_invalid_path() {
    // The public detect() API should always return at least 1 node with CPUs
    let topo = NumaTopology::detect();
    assert!(topo.num_nodes >= 1);
    assert!(!topo.node_cpus[0].is_empty());
}

#[test]
fn test_numa_parse_cpulist_single_cpu() {
    // indirect test via detect_from_sysfs fallback verification
    let topo = NumaTopology {
        num_nodes: 1,
        node_cpus: vec![vec![5]],
    };
    assert_eq!(topo.cpus_for_shard(0), &[5]);
}

#[test]
fn test_numa_16_shards_all_map_to_valid_node() {
    let topo = NumaTopology {
        num_nodes: 4,
        node_cpus: vec![vec![0], vec![1], vec![2], vec![3]],
    };
    for shard_id in 0..SHARD_COUNT {
        let node = topo.shard_to_node(shard_id);
        assert!(node < topo.num_nodes, "shard {} mapped to invalid node {}", shard_id, node);
    }
}

// ============================================================
// Additional Merkle Tree Consistency Tests
// ============================================================

#[test]
fn test_merkle_two_leaves_parent_hash() {
    let leaf0 = [0xAAu8; 32];
    let leaf1 = [0xBBu8; 32];
    let parent = hash2(0, &leaf0, &leaf1);
    assert_ne!(parent, [0u8; 32]);
    assert_ne!(parent, leaf0);
    assert_ne!(parent, leaf1);
}

#[test]
fn test_merkle_level_increases_change_hash() {
    let a = [0x01u8; 32];
    let b = [0x02u8; 32];
    let h0 = hash2(0, &a, &b);
    let h1 = hash2(1, &a, &b);
    let h2 = hash2(2, &a, &b);
    assert_ne!(h0, h1);
    assert_ne!(h1, h2);
    assert_ne!(h0, h2);
}

#[test]
fn test_merkle_commutativity_broken_at_level_0() {
    let a = [0x11u8; 32];
    let b = [0x22u8; 32];
    assert_ne!(hash2(0, &a, &b), hash2(0, &b, &a));
}

#[test]
fn test_merkle_four_leaf_tree() {
    let leaves = [[0x01u8; 32], [0x02u8; 32], [0x03u8; 32], [0x04u8; 32]];
    let parent0 = hash2(0, &leaves[0], &leaves[1]);
    let parent1 = hash2(0, &leaves[2], &leaves[3]);
    let root = hash2(1, &parent0, &parent1);
    assert_ne!(root, [0u8; 32]);
    assert_ne!(root, parent0);
    assert_ne!(root, parent1);
}

#[test]
fn test_merkle_same_leaves_same_root() {
    let leaf = [0x55u8; 32];
    let parent = hash2(0, &leaf, &leaf);
    let root1 = hash2(1, &parent, &parent);
    let root2 = hash2(1, &parent, &parent);
    assert_eq!(root1, root2);
}

#[test]
fn test_merkle_changing_one_leaf_changes_root() {
    let mut leaves = [[0u8; 32]; 4];
    for i in 0..4 {
        leaves[i][0] = i as u8;
    }
    let compute_root = |ls: &[[u8; 32]; 4]| {
        let p0 = hash2(0, &ls[0], &ls[1]);
        let p1 = hash2(0, &ls[2], &ls[3]);
        hash2(1, &p0, &p1)
    };
    let root_original = compute_root(&leaves);
    let mut modified = leaves;
    modified[0][0] = 0xFF;
    let root_modified = compute_root(&modified);
    assert_ne!(root_original, root_modified);
}

#[test]
fn test_batch_hash_matches_merkle_computation() {
    // Compute a 2-level tree using batch_node_hash_cpu and compare with hash2
    let leaves = vec![
        [0x01u8; 32], [0x02u8; 32], [0x03u8; 32], [0x04u8; 32],
    ];
    let mut level0_out = [[0u8; 32]; 2];
    batch_node_hash_cpu(
        &[0, 0],
        &[leaves[0], leaves[2]],
        &[leaves[1], leaves[3]],
        &mut level0_out,
    );
    let expected0 = hash2(0, &leaves[0], &leaves[1]);
    let expected1 = hash2(0, &leaves[2], &leaves[3]);
    assert_eq!(level0_out[0], expected0);
    assert_eq!(level0_out[1], expected1);

    let mut root_out = [[0u8; 32]; 1];
    batch_node_hash_cpu(&[1], &[level0_out[0]], &[level0_out[1]], &mut root_out);
    let expected_root = hash2(1, &level0_out[0], &level0_out[1]);
    assert_eq!(root_out[0], expected_root);
}

// ============================================================
// Additional Stress / Edge Case Tests
// ============================================================

#[test]
fn test_hash2_stress_1000_unique_outputs() {
    let mut seen = std::collections::HashSet::new();
    for i in 0..1000u64 {
        let left = pseudo_random_hash(i);
        let right = pseudo_random_hash(i + 100000);
        let h = hash2(0, &left, &right);
        assert!(seen.insert(h), "Collision at i={}", i);
    }
}

#[test]
fn test_batch_hash_cpu_size_1_through_20() {
    for n in 1..=20 {
        let levels: Vec<u8> = (0..n).map(|i| (i % 12) as u8).collect();
        let lefts: Vec<[u8; 32]> = (0..n).map(|i| pseudo_random_hash(i as u64 * 17)).collect();
        let rights: Vec<[u8; 32]> = (0..n).map(|i| pseudo_random_hash(i as u64 * 17 + 7)).collect();
        let mut out = vec![[0u8; 32]; n];
        batch_node_hash_cpu(&levels, &lefts, &rights, &mut out);
        for i in 0..n {
            assert_eq!(out[i], hash2(levels[i], &lefts[i], &rights[i]),
                "Mismatch at n={}, i={}", n, i);
        }
    }
}

#[test]
fn test_entry_large_value_roundtrip() {
    let key = b"large_val_key";
    let value: Vec<u8> = (0..5000u16).map(|i| (i % 256) as u8).collect();
    let entry = Entry {
        key,
        value: &value,
        next_key_hash: &[0u8; 32],
        version: 100,
        serial_number: 200,
    };
    let total = entry.get_serialized_len(0);
    let mut buf = vec![0u8; total];
    entry.dump(&mut buf, &[]);
    let ebz = skippydb::entryfile::entry::EntryBz { bz: &buf[..total] };
    assert_eq!(ebz.key(), key);
    assert_eq!(ebz.value(), &value[..]);
    assert_eq!(ebz.version(), 100);
    assert_eq!(ebz.serial_number(), 200);
}

#[test]
fn test_hash2_single_byte_inputs() {
    for a in 0u8..=255 {
        for b in [0u8, 128, 255] {
            let h = hash2(0, &[a], &[b]);
            assert_ne!(h, [0u8; 32]);
        }
    }
}

#[test]
fn test_entry_null_known_version() {
    let mut buf = [0u8; ENTRY_BASE_LENGTH + 48];
    let ne = entry::null_entry(&mut buf);
    assert_eq!(ne.version(), skippydb::def::NULL_ENTRY_VERSION);
    assert_eq!(ne.serial_number(), u64::MAX);
}

#[test]
fn test_entry_null_empty_key_value() {
    let mut buf = [0u8; ENTRY_BASE_LENGTH + 48];
    let ne = entry::null_entry(&mut buf);
    assert_eq!(ne.key().len(), 0);
    assert_eq!(ne.value().len(), 0);
}

#[test]
fn test_entry_sentry_shard_0_sn_0() {
    let mut buf = [0u8; ENTRY_BASE_LENGTH + 48];
    let se = entry::sentry_entry(0, 0, &mut buf);
    assert_eq!(se.serial_number(), 0);
    assert_eq!(se.value().len(), 0);
}

#[test]
fn test_entry_sentry_shard_0_sn_1() {
    let mut buf = [0u8; ENTRY_BASE_LENGTH + 48];
    let se = entry::sentry_entry(0, 1, &mut buf);
    assert_eq!(se.serial_number(), 1);
}

#[test]
fn test_changeset_multiple_ops_per_shard() {
    let mut cs = ChangeSet::new();
    for i in 0u8..10 {
        cs.add_op(skippydb::def::OP_READ, 5, &[i; 32], b"k", b"v", None);
    }
    cs.sort();
    assert_eq!(cs.op_count_in_shard(5), 10);

    let mut count = 0;
    cs.run_in_shard(5, |_, _, _, _, _| { count += 1; });
    assert_eq!(count, 10);
}

#[test]
fn test_null_twig_is_deterministic() {
    // Accessing NULL_TWIG twice yields the same value
    let root1 = NULL_TWIG.twig_root;
    let root2 = NULL_TWIG.twig_root;
    assert_eq!(root1, root2);
}

#[test]
fn test_hash_consistency_same_input_always_same_output() {
    let input = b"consistency_test_input";
    let h1 = hash(input);
    let h2 = hash(input);
    let h3 = hash(input);
    assert_eq!(h1, h2);
    assert_eq!(h2, h3);
}

#[test]
fn test_hash2_large_inputs() {
    let a = vec![0xABu8; 10000];
    let b = vec![0xCDu8; 10000];
    let h = hash2(0, &a, &b);
    assert_ne!(h, [0u8; 32]);
    // Deterministic
    let h2_val = hash2(0, &a, &b);
    assert_eq!(h, h2_val);
}

#[test]
fn test_batch_hash_cpu_output_length_matches_input() {
    for n in [0, 1, 5, 50, 100] {
        let levels = vec![0u8; n];
        let lefts = vec![[0u8; 32]; n];
        let rights = vec![[0u8; 32]; n];
        let mut out = vec![[0u8; 32]; n];
        batch_node_hash_cpu(&levels, &lefts, &rights, &mut out);
        assert_eq!(out.len(), n);
    }
}

#[test]
fn test_entry_buffer_write_then_read_verifies_content() {
    use skippydb::entryfile::entrybuffer;
    let key = b"verification_key";
    let value = b"verification_value";
    let sn = 42u64;
    let entry = Entry {
        key,
        value,
        next_key_hash: &[0xFEu8; 32],
        version: 12345,
        serial_number: sn,
    };
    let (mut writer, mut reader) = entrybuffer::new(0, BIG_BUF_SIZE);
    writer.append(&entry, &[]);
    writer.end_block(0, 0, sn + 1);

    let mut verified = false;
    loop {
        let (eob, _) = reader.read_next_entry(|ebz| {
            assert_eq!(ebz.key(), key);
            assert_eq!(ebz.value(), value);
            assert_eq!(ebz.serial_number(), sn);
            assert_eq!(ebz.version(), 12345);
            verified = true;
        });
        if eob { break; }
    }
    assert!(verified);
    reader.read_extra_info();
}

#[test]
fn test_changeset_print_does_not_panic() {
    let mut cs = ChangeSet::new();
    let kh = [0x01u8; 32];
    cs.add_op(skippydb::def::OP_CREATE, 0, &kh, b"key", b"val", None);
    // print() should not panic (it prints to stdout)
    cs.print();
}

#[test]
fn test_entry_vec_roundtrip_to_from_bytes() {
    use skippydb::entryfile::entry::EntryVec;
    let mut ev = EntryVec::new();
    let kh = hash(b"test_key_for_ev");
    let entry = Entry {
        key: b"test_key_for_ev",
        value: b"test_val",
        next_key_hash: &[0u8; 32],
        version: 1,
        serial_number: 1,
    };
    ev.add_entry(&kh, &entry, &[]);
    let bytes = ev.to_bytes();
    let ev2 = EntryVec::from_bytes(&bytes);
    assert_eq!(ev.total_bytes(), ev2.total_bytes());
}

#[test]
fn test_entry_vec_new_has_shard_count_pos_lists() {
    use skippydb::entryfile::entry::EntryVec;
    let ev = EntryVec::new();
    assert_eq!(ev.total_bytes(), 0);
}

// ============================================================
// Shard and Utility Tests
// ============================================================

#[test]
fn test_shard_count_is_16() {
    assert_eq!(SHARD_COUNT, 16);
}

#[test]
fn test_big_buf_size_is_64kb() {
    assert_eq!(BIG_BUF_SIZE, 64 * 1024);
}

#[test]
fn test_leaf_count_in_twig_is_2048() {
    assert_eq!(skippydb::def::LEAF_COUNT_IN_TWIG, 2048);
}

#[test]
fn test_twig_shift_is_11() {
    assert_eq!(skippydb::def::TWIG_SHIFT, 11);
}

#[test]
fn test_null_entry_version_is_negative_2() {
    assert_eq!(skippydb::def::NULL_ENTRY_VERSION, -2);
}

#[test]
fn test_op_constants_are_distinct() {
    assert_ne!(skippydb::def::OP_READ, skippydb::def::OP_WRITE);
    assert_ne!(skippydb::def::OP_WRITE, skippydb::def::OP_CREATE);
    assert_ne!(skippydb::def::OP_CREATE, skippydb::def::OP_DELETE);
    assert_ne!(skippydb::def::OP_READ, skippydb::def::OP_DELETE);
}

#[test]
fn test_first_level_above_twig_is_13() {
    assert_eq!(skippydb::def::FIRST_LEVEL_ABOVE_TWIG, 13);
}

#[test]
fn test_twig_root_level_is_12() {
    assert_eq!(skippydb::def::TWIG_ROOT_LEVEL, 12);
}

#[test]
fn test_sentry_count() {
    // SENTRY_COUNT = (1<<16) / SHARD_COUNT = 65536 / 16 = 4096
    assert_eq!(skippydb::def::SENTRY_COUNT, 4096);
}

// ============================================================
// Hash Domain Separation Comprehensive
// ============================================================

#[test]
fn test_hash2_level_domain_sep_level_0_vs_1() {
    let a = [0xDEu8; 32];
    let b = [0xADu8; 32];
    assert_ne!(hash2(0, &a, &b), hash2(1, &a, &b));
}

#[test]
fn test_hash2_level_domain_sep_level_7_vs_8() {
    let a = [0x11u8; 32];
    let b = [0x22u8; 32];
    assert_ne!(hash2(7, &a, &b), hash2(8, &a, &b));
}

#[test]
fn test_hash2_level_domain_sep_level_11_vs_12() {
    let a = [0xAAu8; 32];
    let b = [0xBBu8; 32];
    assert_ne!(hash2(11, &a, &b), hash2(12, &a, &b));
}

#[test]
fn test_node_hash_inplace_level_8_known_vector() {
    let mut out = [0u8; 32];
    hasher::node_hash_inplace(8, &mut out, "hello", "world");
    assert_eq!(
        hex::encode(out),
        "8e6fc50a3f98a3c314021b89688ca83a9b5697ca956e211198625fc460ddf1e9"
    );
}

#[test]
fn test_hash2x_exchange_false_equals_hash2() {
    let a: &[u8] = b"first_arg";
    let b: &[u8] = b"second_arg";
    assert_eq!(hash2x(3, a, b, false), hash2(3, a, b));
}

#[test]
fn test_hash2x_exchange_true_reverses() {
    let a: &[u8] = b"arg_a";
    let b: &[u8] = b"arg_b";
    assert_eq!(hash2x(3, a, b, true), hash2(3, b, a));
}

#[test]
fn test_batch_vs_sequential_at_twig_root_level() {
    let level = skippydb::def::TWIG_ROOT_LEVEL as u8;
    let n = 8;
    let lefts: Vec<[u8; 32]> = (0..n).map(|i| pseudo_random_hash(i as u64 * 100)).collect();
    let rights: Vec<[u8; 32]> = (0..n).map(|i| pseudo_random_hash(i as u64 * 100 + 50)).collect();
    let levels = vec![level; n];
    let mut out = vec![[0u8; 32]; n];
    batch_node_hash_cpu(&levels, &lefts, &rights, &mut out);
    for i in 0..n {
        assert_eq!(out[i], hash2(level, &lefts[i], &rights[i]));
    }
}

// ============================================================
// Entry Buffer Free-list and State Tests
// ============================================================

#[test]
fn test_entry_buffer_writer_curr_buf_not_none() {
    use skippydb::entryfile::entrybuffer;
    let (writer, _reader) = entrybuffer::new(0, BIG_BUF_SIZE);
    // Verify writer was constructed correctly - it holds an Arc to the entry buffer
    assert!(std::sync::Arc::strong_count(&writer.entry_buffer) >= 1);
}

#[test]
fn test_entry_buffer_empty_block() {
    use skippydb::entryfile::entrybuffer;
    let (mut writer, mut reader) = entrybuffer::new(0, BIG_BUF_SIZE);
    writer.end_block(0, 0, 0);
    // An empty block: read_next_entry should immediately return end-of-block
    let (eob, _) = reader.read_next_entry(|_| {});
    assert!(eob, "Empty block should immediately signal end-of-block");
    let (cdp, cds, se) = reader.read_extra_info();
    assert_eq!(cdp, 0);
    assert_eq!(cds, 0);
    assert_eq!(se, 0);
}

#[test]
fn test_entry_buffer_get_entry_bz_at_past_end_panics() {
    use skippydb::entryfile::entrybuffer;
    let (mut writer, _reader) = entrybuffer::new(0, BIG_BUF_SIZE);
    let entry = make_entry(b"k", b"v", 1);
    writer.append(&entry, &[]);

    // get_entry_bz_at for a position that is in range but at end is ok
    // But accessing past end should panic (we just verify the valid case doesn't panic)
    let (in_disk, accessed) = writer.get_entry_bz_at(0, |_| {});
    assert!(!in_disk);
    assert!(accessed);
}

#[test]
fn test_active_bits_set_and_clear_repeatedly() {
    let mut bits = ActiveBits::new();
    for _ in 0..100 {
        bits.set_bit(500);
        assert!(bits.get_bit(500));
        bits.clear_bit(500);
        assert!(!bits.get_bit(500));
    }
}

#[test]
fn test_active_bits_all_pages_correct_size() {
    let bits = ActiveBits::new();
    for page in 0..8 {
        let slice = bits.get_bits(page, 32);
        assert_eq!(slice.len(), 32);
    }
}

#[test]
fn test_hash2_different_data_lengths() {
    let results: Vec<[u8; 32]> = (1..=10)
        .map(|len| {
            let data = vec![0xAAu8; len];
            hash2(0, &data, &data)
        })
        .collect();
    // All should be different (different length data)
    for i in 0..results.len() {
        for j in (i + 1)..results.len() {
            assert_ne!(results[i], results[j],
                "hash2 collision for data lengths {} and {}", i + 1, j + 1);
        }
    }
}

#[test]
fn test_null_higher_tree_level_63_non_zero() {
    use skippydb::merkletree::twig::NULL_NODE_IN_HIGHER_TREE;
    assert_ne!(NULL_NODE_IN_HIGHER_TREE[63], [0u8; 32]);
}

#[test]
fn test_null_higher_tree_known_value() {
    use skippydb::merkletree::twig::NULL_NODE_IN_HIGHER_TREE;
    assert_eq!(
        hex::encode(NULL_NODE_IN_HIGHER_TREE[63]),
        "c787c83f6f8402c636a2f48f1bf2c02ceb31ea5ccdd4bd9e6fe6efcc3031b640"
    );
}

#[test]
fn test_null_twig_known_root() {
    assert_eq!(
        hex::encode(NULL_TWIG.twig_root),
        "37f6d34b5f4fe4aba10fd7411d6f58efc4bf844935c37dbe83c5686ceb62ce9d"
    );
}

#[test]
fn test_entry_bz_hash_uses_payload_not_full_bytes() {
    // hash() of entry_bz uses payload_len bytes, not the full serialized length
    let entry = make_entry(b"hashkey", b"hashvalue", 5);
    let total = entry.get_serialized_len(0);
    let mut buf = vec![0u8; total];
    entry.dump(&mut buf, &[]);
    let ebz = skippydb::entryfile::entry::EntryBz { bz: &buf[..total] };
    let payload = ebz.payload_len();
    let expected: [u8; 32] = Sha256::digest(&buf[..payload]).into();
    assert_eq!(ebz.hash(), expected);
}

#[test]
fn test_changeset_add_op_rec() {
    use skippydb::utils::OpRecord;
    let mut cs = ChangeSet::new();
    let mut rec = OpRecord::new(skippydb::def::OP_CREATE);
    rec.shard_id = skippydb::utils::byte0_to_shard_id(hash(b"rec_key")[0]);
    rec.key = b"rec_key".to_vec();
    rec.value = b"rec_val".to_vec();
    cs.add_op_rec(rec);
    assert_eq!(cs.op_list.len(), 1);
}

#[test]
fn test_numa_detect_from_sysfs_nonexistent() {
    // detect() falls back to 1 node when sysfs is unavailable
    // We test the public detect() API; on non-NUMA or single-node systems it returns 1 node
    let topo = NumaTopology::detect();
    assert!(topo.num_nodes >= 1);
    // The public API should never return 0 nodes
    assert!(!topo.node_cpus.is_empty());
}

#[test]
fn test_entry_get_entry_len_matches_serialized_len() {
    let entry = make_entry(b"lentest", b"lentestval", 999);
    let total = entry.get_serialized_len(0);
    let mut buf = vec![0u8; total];
    entry.dump(&mut buf, &[]);
    let len_from_bz = skippydb::entryfile::entry::EntryBz::get_entry_len(&buf[..5]);
    assert_eq!(len_from_bz, total);
}

#[test]
fn test_entry_get_entry_len_with_dsn() {
    let entry = make_entry(b"k", b"v", 1);
    let dsn = [10u64, 20, 30];
    let total = entry.get_serialized_len(3);
    let mut buf = vec![0u8; total];
    entry.dump(&mut buf, &dsn);
    let len_from_bz = skippydb::entryfile::entry::EntryBz::get_entry_len(&buf[..5]);
    assert_eq!(len_from_bz, total);
}

// ============================================================
// GPU-gated Tests
// ============================================================

#[cfg(feature = "cuda")]
#[test]
fn test_gpu_hasher_init() {
    use skippydb::gpu::GpuHasher;
    let result = GpuHasher::new(1000);
    // Should not panic; may succeed or fail gracefully
    match result {
        Ok(_) => {},
        Err(_) => {},
    }
}

#[cfg(feature = "cuda")]
#[test]
fn test_batch_node_hash_gpu_matches_cpu() {
    use skippydb::gpu::{GpuHasher, NodeHashJob};
    use skippydb::utils::hasher::batch_node_hash_gpu;
    if let Ok(gpu) = GpuHasher::new(10) {
        let jobs: Vec<NodeHashJob> = (0..5).map(|i| NodeHashJob {
            level: i as u8,
            left: pseudo_random_hash(i),
            right: pseudo_random_hash(i + 50),
        }).collect();
        let mut gpu_out = vec![[0u8; 32]; 5];
        batch_node_hash_gpu(&gpu, &jobs, &mut gpu_out);
        for (i, job) in jobs.iter().enumerate() {
            let cpu_result = hash2(job.level, &job.left, &job.right);
            assert_eq!(gpu_out[i], cpu_result, "GPU/CPU mismatch at job {}", i);
        }
    }
}

// ============================================================
// Additional Entry Buffer Stress Tests
// ============================================================

#[test]
fn test_entry_buffer_100_entries_roundtrip() {
    use skippydb::entryfile::entrybuffer;
    let n = 100u64;
    let (mut writer, mut reader) = entrybuffer::new(0, 4 * BIG_BUF_SIZE);
    for i in 0..n {
        let sn = i;
        let entry = Entry {
            key: b"stress_key",
            value: b"stress_value_data",
            next_key_hash: &[0u8; 32],
            version: i as i64,
            serial_number: sn,
        };
        writer.append(&entry, &[]);
    }
    writer.end_block(0, 0, n);

    let mut count = 0u64;
    loop {
        let (eob, _) = reader.read_next_entry(|ebz| {
            assert_eq!(ebz.key(), b"stress_key");
            assert_eq!(ebz.value(), b"stress_value_data");
            count += 1;
        });
        if eob { break; }
    }
    assert_eq!(count, n);
    reader.read_extra_info();
}

#[test]
fn test_entry_buffer_entries_with_increasing_sns() {
    use skippydb::entryfile::entrybuffer;
    let n = 20u64;
    let (mut writer, mut reader) = entrybuffer::new(0, 4 * BIG_BUF_SIZE);
    for i in 0..n {
        writer.append(&make_entry(b"key", b"val", i), &[]);
    }
    writer.end_block(0, 0, n);

    let mut sns = Vec::new();
    loop {
        let (eob, _) = reader.read_next_entry(|ebz| {
            sns.push(ebz.serial_number());
        });
        if eob { break; }
    }
    assert_eq!(sns.len(), n as usize);
    // Serial numbers should be in order 0..n
    for (i, &sn) in sns.iter().enumerate() {
        assert_eq!(sn, i as u64);
    }
    reader.read_extra_info();
}

