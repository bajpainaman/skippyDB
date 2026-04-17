//! End-to-end test for `AdsCore::for_each_active_entry` /
//! `AdsWrap::for_each_active_entry`.
//!
//! This is the critical roundtrip test: it populates a real sharded ADS
//! across multiple blocks with a mix of creates, updates, and deletes,
//! then uses `for_each_active_entry` to iterate every shard and asserts
//! that the yielded `(key, value)` set exactly matches the final live
//! state. If this passes, downstream consumers (e.g. the ETO UVM) can
//! rebuild in-memory state from disk on restart.

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::RwLock;

use kyumdb::config::Config;
use kyumdb::def::{IN_BLOCK_IDX_BITS, SHARD_COUNT};
use kyumdb::seqads::task::{SingleCsTask, TaskBuilder};
use kyumdb::tasks::TasksManager;
use kyumdb::test_helper::TempDir;
use kyumdb::utils::{byte0_to_shard_id, hasher};
use kyumdb::{AdsCore, AdsWrap, ADS};

fn key_for(i: u32) -> Vec<u8> {
    let mut k = b"acct-".to_vec();
    k.extend_from_slice(&i.to_be_bytes());
    k
}

fn val_for(i: u32, tag: u8) -> Vec<u8> {
    // Keep values small but deterministic, tag lets us discriminate
    // "first version" vs "updated version".
    let mut v = vec![tag];
    v.extend_from_slice(&i.to_be_bytes());
    v.extend_from_slice(b"-payload-xxxxxxxxxx"); // fixed padding
    v
}

fn shard_of(k: &[u8]) -> usize {
    let kh = hasher::hash(k);
    byte0_to_shard_id(kh[0])
}

/// Drive one task at `height` carrying a single change set. Mirrors the
/// pattern in `tests/test_consistency_with_seqads.rs` and the in-tree
/// `test_start_block`.
fn run_block(
    ads: &mut AdsWrap<SingleCsTask>,
    height: i64,
    task: SingleCsTask,
) {
    let last_task_id = height << IN_BLOCK_IDX_BITS;
    let tasks = vec![RwLock::new(Some(task))];
    let tm = Arc::new(TasksManager::new(tasks, last_task_id));
    let (started, _prev) = ads.start_block(height, tm);
    assert!(started, "start_block at height {} refused", height);

    let shared = ads.get_shared();
    // Tag the block so MetaDB can commit it.
    shared.insert_extra_data(height, String::new());
    // Submit the single task in this block.
    let task_id = height << IN_BLOCK_IDX_BITS;
    shared.add_task(task_id);
}

#[cfg(not(feature = "tee_cipher"))]
#[test]
fn test_for_each_active_entry_roundtrip() {
    // 10,000 entries spread across 5 blocks (2,000 per block),
    // then: delete 2,000, update 3,000, leaving 8,000 live entries total
    // (of which 3,000 carry updated values).
    const TOTAL: u32 = 10_000;
    const BLOCKS: u32 = 5;
    const PER_BLOCK: u32 = TOTAL / BLOCKS;
    const DELETED: u32 = 2_000;
    const UPDATED: u32 = 3_000;

    let ads_dir = "./test_for_each_active_entry_roundtrip";
    let _tmp_dir = TempDir::new(ads_dir);

    // Tight compaction so flushed entries are visible via the entry file.
    let config = Config::from_dir_and_compact_opt(ads_dir, 1, 1, 1);
    AdsCore::init_dir(&config);
    let mut ads: AdsWrap<SingleCsTask> = AdsWrap::new(&config);

    // ----- Phase 1: creates, spread across BLOCKS blocks -----
    for b in 0..BLOCKS {
        let mut tb = TaskBuilder::new();
        for local in 0..PER_BLOCK {
            let i = b * PER_BLOCK + local;
            let k = key_for(i);
            let v = val_for(i, 0);
            tb.create(&k, &v);
        }
        run_block(&mut ads, (b + 1) as i64, tb.build());
    }

    // Drain the pipeline so all 5 blocks' writes hit the entry files.
    let _ = ads.flush();

    // ----- Phase 2: delete 2,000 (indices 0..DELETED) -----
    let mut tb = TaskBuilder::new();
    for i in 0..DELETED {
        let k = key_for(i);
        let v = val_for(i, 0);
        tb.delete(&k, &v);
    }
    run_block(&mut ads, (BLOCKS + 1) as i64, tb.build());
    let _ = ads.flush();

    // ----- Phase 3: update 3,000 (indices DELETED..DELETED+UPDATED) -----
    let mut tb = TaskBuilder::new();
    for i in DELETED..DELETED + UPDATED {
        let k = key_for(i);
        let new_v = val_for(i, 1); // tag=1 to distinguish from initial
        tb.write(&k, &new_v);
    }
    run_block(&mut ads, (BLOCKS + 2) as i64, tb.build());
    let _ = ads.flush();

    // Nudge the pipeline by submitting an empty-ish block, so any in-flight
    // writes from phase 3 are guaranteed to flush before we start iterating
    // (the pipeline maintains a 2-block depth on task_hub).
    let mut tb_noop = TaskBuilder::new();
    // A no-op ChangeSet is illegal, so create+delete a sentinel key.
    tb_noop.create(b"__sentinel__", b"1");
    run_block(&mut ads, (BLOCKS + 3) as i64, tb_noop.build());
    let mut tb_noop2 = TaskBuilder::new();
    tb_noop2.delete(b"__sentinel__", b"1");
    run_block(&mut ads, (BLOCKS + 4) as i64, tb_noop2.build());
    let _ = ads.flush();

    // Build the expected (key -> value) map for the final live state.
    let mut expected: HashMap<Vec<u8>, Vec<u8>> = HashMap::new();
    for i in 0..TOTAL {
        let tag = if i >= DELETED && i < DELETED + UPDATED {
            1u8
        } else {
            0u8
        };
        if i < DELETED {
            continue; // deleted
        }
        expected.insert(key_for(i), val_for(i, tag));
    }

    // Dump file sizes for debugging
    {
        let meta = ads.get_metadb();
        let meta = meta.read();
        for shard_id in 0..SHARD_COUNT {
            eprintln!(
                "shard {}: entry_file_size={} oldest_active_file_pos={} oldest_active_sn={}",
                shard_id,
                meta.get_entry_file_size(shard_id),
                meta.get_oldest_active_file_pos(shard_id),
                meta.get_oldest_active_sn(shard_id),
            );
        }
        let efs = ads.get_entry_files();
        for (shard_id, ef) in efs.iter().enumerate() {
            eprintln!(
                "shard {}: ef.size()={} ef.size_on_disk()={}",
                shard_id,
                ef.size(),
                ef.size_on_disk()
            );
        }
    }

    // ----- Phase 4: scan every shard and collect -----
    let mut seen: HashMap<Vec<u8>, Vec<u8>> = HashMap::new();
    for shard_id in 0..SHARD_COUNT {
        ads.for_each_active_entry(shard_id, |k, v| {
            // Assert per-shard routing: every key yielded by shard `shard_id`
            // must actually hash to that shard.
            assert_eq!(
                shard_of(k),
                shard_id,
                "key yielded by wrong shard: {:?}", k
            );
            let prev = seen.insert(k.to_vec(), v.to_vec());
            assert!(prev.is_none(), "duplicate key yielded: {:?}", k);
            true
        });
    }

    // ----- Phase 5: assert exact match -----
    assert_eq!(
        seen.len(),
        expected.len(),
        "yielded {} entries, expected {}",
        seen.len(),
        expected.len()
    );
    for (k, v) in &expected {
        let got = seen.get(k).unwrap_or_else(|| {
            panic!("missing key {:?} in yielded set", k)
        });
        assert_eq!(got, v, "value mismatch for key {:?}", k);
    }

    // ----- Phase 6: verify early-stop semantics -----
    let mut counted = 0usize;
    let mut stop_at = 5usize;
    ads.for_each_active_entry(0, |_k, _v| {
        counted += 1;
        stop_at -= 1;
        stop_at > 0
    });
    assert!(
        counted <= 5,
        "early stop was not honored: counted {} entries",
        counted
    );
}
