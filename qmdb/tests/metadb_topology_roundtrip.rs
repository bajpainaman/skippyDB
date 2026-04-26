//! Phase 2.3d parametric test. Proves that `MetaDB` actually round-trips
//! through commit + reload at shard counts other than the compile-time
//! `SHARD_COUNT`.
//!
//! This is the direct evidence that 2.3b-v's lifted `Topology::new`
//! assertion is load-bearing: the boxed `MetaInfo` fields allocate at the
//! requested size, `commit()` persists that many per-shard records into
//! the V2 envelope, and a fresh `with_dir_and_shard_count` re-opens and
//! reads each of those records back unchanged.
//!
//! Does NOT cover the Tree / Flusher / indexer pipelines — those still
//! build against `SHARD_COUNT` via call sites that Phase 2.3e (not yet
//! landed) will thread the Topology through. Once those land, a parallel
//! integration test should open an `AdsCore` at shard_count ∈ {32, 64}
//! and drive a block through it.

use skippydb::metadb::{MetaDB, MetaDbError};
use skippydb::test_helper::TempDir;

/// Round-trip helper: build a MetaDB at `shard_count`, populate every per-
/// shard field with a deterministic value keyed off the shard id, commit,
/// drop, reopen, verify every field round-tripped.
fn roundtrip_at(shard_count: usize, tag: &str) {
    let dir_path = format!("./test_metadb_topology_{}_{}", shard_count, tag);
    let _guard = TempDir::new(&dir_path);

    // Populate + commit.
    {
        let mut mdb = MetaDB::with_dir_and_shard_count(&dir_path, None, shard_count);
        mdb.set_curr_height(7);
        for i in 0..shard_count {
            mdb.set_next_serial_num(i, 10_000 + i as u64);
            mdb.set_oldest_active_sn(i, 20_000 + i as u64);
            mdb.set_oldest_active_file_pos(i, 30_000 + i as i64);
            mdb.set_twig_file_size(i, 40_000 + i as i64);
            mdb.set_entry_file_size(i, 50_000 + i as i64);
            mdb.set_root_hash(i, [(i as u8).wrapping_mul(17); 32]);
            mdb.set_last_pruned_twig(i, 60_000 + i as u64, 70_000 + i as i64);
            mdb.set_edge_nodes(i, &[(i as u8).wrapping_mul(3); 8]);
        }
        mdb.insert_extra_data(7, format!("height:7-sc:{}", shard_count));
        mdb.commit();
    }

    // Reopen and verify every field round-tripped.
    {
        let mdb = MetaDB::with_dir_and_shard_count(&dir_path, None, shard_count);
        assert_eq!(
            mdb.get_curr_height(),
            7,
            "curr_height lost at shard_count={}",
            shard_count
        );
        for i in 0..shard_count {
            assert_eq!(mdb.get_next_serial_num(i), 10_000 + i as u64);
            assert_eq!(mdb.get_oldest_active_sn(i), 20_000 + i as u64);
            assert_eq!(mdb.get_oldest_active_file_pos(i), 30_000 + i as i64);
            assert_eq!(mdb.get_twig_file_size(i), 40_000 + i as i64);
            assert_eq!(mdb.get_entry_file_size(i), 50_000 + i as i64);
            assert_eq!(mdb.get_root_hash(i), [(i as u8).wrapping_mul(17); 32]);
            assert_eq!(
                mdb.get_last_pruned_twig(i),
                (60_000 + i as u64, 70_000 + i as i64)
            );
            assert_eq!(
                mdb.get_edge_nodes(i),
                vec![(i as u8).wrapping_mul(3); 8],
                "edge_nodes diverged at shard_count={} shard={}",
                shard_count,
                i
            );
        }
        assert_eq!(
            mdb.get_extra_data(),
            format!("height:7-sc:{}", shard_count)
        );
    }
}

#[test]
fn roundtrip_at_16_shards() {
    roundtrip_at(16, "a");
}

#[test]
fn roundtrip_at_32_shards() {
    roundtrip_at(32, "a");
}

#[test]
fn roundtrip_at_64_shards() {
    roundtrip_at(64, "a");
}

#[test]
fn reopen_with_wrong_expected_shard_count_errors() {
    // Build a 32-shard DB, try to reopen via `with_dir_checked(expected=64)`.
    // The envelope records shard_count=32, so the checked path must surface
    // ShardCountMismatch regardless of what the binary's SHARD_COUNT is.
    let dir_path = "./test_metadb_topology_mismatch";
    let _guard = TempDir::new(dir_path);

    {
        let mut mdb = MetaDB::with_dir_and_shard_count(dir_path, None, 32);
        mdb.set_curr_height(1);
        mdb.insert_extra_data(1, String::new());
        mdb.commit();
    }

    let result = MetaDB::with_dir_checked(dir_path, None, 64);
    match result {
        Err(MetaDbError::ShardCountMismatch { expected, got }) => {
            assert_eq!(expected, 64);
            assert_eq!(got, 32);
        }
        Err(other) => panic!("expected ShardCountMismatch, got {:?}", other),
        Ok(_) => panic!(
            "with_dir_checked silently accepted 64 when on-disk was 32"
        ),
    }
}
