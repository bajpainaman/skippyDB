//! Phase 2.1 failing-load test (TDD-red).
//!
//! Stakes out the contract that Phase 2.2's `MetaInfoV2` version envelope will
//! implement: reopening a MetaDB dir with a different shard count than it was
//! built against must surface a typed error, not silently succeed.
//!
//! Today this test **fails** because `MetaDB::with_dir_checked` is just a
//! no-op wrapper around `with_dir`. Phase 2.2 is expected to flip it green.

use skippydb::def::SHARD_COUNT;
use skippydb::metadb::{MetaDB, MetaDbError};
use skippydb::test_helper::TempDir;

const TEST_DIR: &str = "./test_metadb_sc_mismatch";

#[test]
fn metadb_with_dir_checked_detects_shard_count_mismatch() {
    // TempDir handles cleanup on drop; keep it alive for the whole test.
    let _guard = TempDir::new(TEST_DIR);

    // Build an initial MetaDB at the host binary's SHARD_COUNT and commit once
    // so an `info.0` file exists on disk.
    {
        let mut mdb = MetaDB::with_dir(TEST_DIR, None);
        mdb.init();
    }

    // Reopen, claiming a different shard count than what's on disk.
    let expected = SHARD_COUNT * 2;
    let got = SHARD_COUNT;
    let result = MetaDB::with_dir_checked(TEST_DIR, None, expected);

    // Contract (Phase 2.2 target): typed error whose variant carries both
    // sides of the mismatch so callers can report a useful message.
    // MetaDbError only has ShardCountMismatch today, so this match is
    // exhaustive; when Phase 2.2 adds more variants, the compiler will
    // force us to handle them here.
    match result {
        Err(MetaDbError::ShardCountMismatch {
            expected: e,
            got: g,
        }) => {
            assert_eq!(e, expected, "error.expected must round-trip the caller's claim");
            assert_eq!(g, got, "error.got must reflect the on-disk shard count");
        }
        Ok(_) => panic!(
            "BUG: MetaDB::with_dir_checked silently accepted a shard-count mismatch \
             (expected={}, on-disk={}). Phase 2.2 must detect this before any downstream \
             per-shard indexing happens.",
            expected, got
        ),
    }
}
