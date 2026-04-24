# TODO — parking lot for the moonshot rewrite

Unrelated bugs, smells, and follow-ups found while doing the moonshot work.
These do NOT go into the phase diffs. Clear once they're fixed, triaged, or
have owners.

## Latent / dead code

- `MetaInfo::root_hash_by_height: Vec<[u8;32]>` (qmdb/src/metadb.rs:26) is
  never pushed anywhere. Read path at lines 453–464 has logic for len 0/1/2,
  always hits the len==0 branch in production. Either wire up the writer or
  delete the field in the V2 envelope.

- `commit_async` exists (metadb.rs:227) but every production caller uses
  sync `commit()` (flusher.rs:305,406,545,693 + lib.rs:529). Phase 1.1 flips
  these; this note exists so future readers know the asymmetry was
  intentional before Phase 1.1 landed.

## README / docs drift

- README.md:104 architecture diagram says "MetaDB (RocksDB)". MetaDB is a
  custom two-file ping-pong (metadb.rs), not RocksDB. Fix after Phase 1 so
  the diagram reflects reality.

- README.md performance table (line ~543) cites "~1.2M ops/s sequential
  writes". The only recorded benchmark on disk
  (bench/benchmark-data.json: 161K updates/s) was produced on a GitHub
  Actions `ubuntu-latest` VM with no GPU, 4-vCPU shared, `--entry-count
  4000000`, all hover flags pinned to 1. Numbers don't match. Either rerun
  on bench host and update, or qualify the claim.

## Bench

- `bench/benchmark-data.json` is stale (CI smoke-test artifact). Move or
  delete once rewrite/moonshot has credible baseline numbers.

- `bench/README.md` is a one-line stub. Fill in once bench/results/ has real
  data.

- **Ultraplan drift: `--entry-count 4000000` fails precondition** at
  `bench/src/bin/speed.rs:74` (`blocks_for_db_population=40 < tps_blocks=50`
  with release defaults). Minimum that works with default flags is
  `--entry-count 5000000`. Using 5M for the "4M bench" slot and calling it
  `main-5m.json` / `phaseN-5m.json`. 40M bench unaffected. Every phase uses
  the same flags, so ratios stay honest.

## Phase 0 trace — how to read it

The `SKIPPY_TRACE=1` env gate wraps each phase of
`FlusherShard::flush_gpu_resident` with an `Instant` timer that prints
`TRACE shard=S height=H phase=P us=U` to stderr. Run + post-process:

```
SKIPPY_TRACE=1 cargo run --release --features cuda --bin speed \
  -- --entry-count 5000000 2>/tmp/t.log
grep '^TRACE' /tmp/t.log | awk '
{ for (i=1;i<=NF;i++) { split($i,kv,"="); if (kv[1]=="phase") p=kv[2]; else if (kv[1]=="us") u=kv[2]; }
  s[p]+=u; n[p]++; if (u>m[p]) m[p]=u }
END { for (x in s) printf "%-20s n=%5d total_ms=%8.1f avg_us=%7.0f max_us=%7.0f\n",
      x, n[x], s[x]/1000, s[x]/n[x], m[x] }'
```

eprintln! I/O inflates wall-clock ~3.5× while tracing is on — use the
trace to read phase **shares**, not absolute times. First capture:
`entry_append` = 64% of per-shard block time (the Tree-append + deactive
loop). `sync_upper_nodes` = 26%. `metadb_commit`, `flush_files_gpu` tiny.

## Phase 4 Blake3 probe — FAILED (-13.9% at 40M cuda)

- Branch: `rewrite/phase4-blake3` (commit a2045a4). Evidence JSON
  copied to moonshot as `bench/results/phase4-blake3-40m.json`.
- SHA-NI on Zen 3 makes CPU SHA256 ~as fast as Blake3 at 65B inputs;
  force-CPU routing (no Blake3 CUDA kernel yet) disabled the existing
  SoA GPU kernel that was carrying upper-tree sync. Net regression.
- Reopen path: write a Blake3 CUDA kernel first, THEN redo the swap.
  That's a weeks-long kernel port (the Blake3 tree structure is simple
  but getting GPU warp-lane layout right takes care). Not cheap.
- Don't re-attempt on CPU-only path expecting a win.

## Phase 3.1 fdatasync probe — FAILED, experimental branch retained

- Tried `sync_all` → `sync_data` at `hpfile/src/lib.rs:314` on a
  throwaway `rewrite/phase3-fdatasync` branch (commit 7c0b9f3).
- 5M cuda smoke: ran clean, `success: true`, numbers in range
  (block_pop 10.9/s, updates 1.31M/s — within noise of main-5m's
  10–12/s spread).
- 40M cuda: **crashed mid-run.** Panic sites: `gpu_hasher.rs:977`
  (GPU sync), `gpu_hasher.rs:967` (another GPU op), and
  `flusher.rs:144` (entry append path). Failure cascades through a
  scoped thread.
- Hypothesis: compactor (trips at `compact_thres=20M`, so 40M exercises
  it) reads old entries from the entry file immediately after a flush.
  `fdatasync` skips some metadata updates that matter for O_DIRECT
  read-after-write across the segmented HPFile; the compactor's
  re-append then feeds bad bytes into GPU hashing and the panic chain
  lights up. 5M never triggers compaction so the bug stays invisible.
- Correct fix needs either (a) targeted `fdatasync` where safe + keep
  `fsync` on segment rotation / reads-after-write, or (b) Phase 0
  instrumentation to see which calls the compactor actually makes
  after flush. Not attempted this session — reverted to `sync_all` on
  rewrite/moonshot. The experimental branch stays around for the
  crash log.

## Phase 2.2 on-disk format bump (BREAKING)

- `MetaInfo` plaintext is now prefixed with the 8-byte magic
  `META_MAGIC_V2 = b"SKIPV2\x00\x00"`. Written at the head of every
  `info.0` / `info.1` file (inside the AES-GCM payload when
  `tee_cipher` is on).
- `MetaInfo` gained a `shard_count: u32` field (stamped from
  `SHARD_COUNT` at build time) so Phase 2.3's runtime Topology can
  refuse to reopen a DB built against a different compile-time
  shard count.
- Pre-V2 DBs (anything from `rewrite/phase1` and earlier) are **not
  migrated** — `MetaDB::reload_from_file` panics loudly with a
  pointer back to this TODO entry. `MetaDB::with_dir_checked` returns
  `MetaDbError::UnsupportedFormat` so callers that care can surface
  a clean error instead.
- When bumping the format again, bump the last two bytes of
  `META_MAGIC_V2`, add a dispatch in `parse_metainfo_v2`, and leave
  the V2 path green for a deprecation window.

## Phase 1.2 empirical finding (DO NOT reintroduce depth>2 blindly)

- Bumping `BLOCK_PIPELINE_DEPTH` from 2 → 4 regressed at 40M cuda on
  skippy-dev (5900X + RTX 4080S + ext4 NVMe): **reads -10%**, updates
  -4.2%, transactions -4.4% vs main. block_population stayed positive
  but weaker than depth 2 (+4.5% vs +6.9%).
- Root cause hypothesis: this workload isn't backpressure-bound on
  free pipeline slots — the flusher is not stalling waiting for the
  updater to hand off, so extra in-flight slots add no overlap
  benefit. Meanwhile each extra slot holds a live `Arc<EntryCache>`
  which spreads hot entries across more caches and hurts read hit
  rate (`reads -10%` is the clearest signal).
- Reverted commit: `perf(taskhub): pipeline depth 2 → 4 (Phase 1.2)`
  on rewrite/phase1 (b674501) via 2861e21. The `Slot`-array refactor
  is also reverted — re-add it in Phase 3.4 when
  `BlockRingTaskHub<N>` becomes runtime-configurable and can be tuned
  per target-validator profile (enterprise PCIe Gen5 NVMe + H100/H200
  is where >2 might pay off, because the fsync tail shrinks and the
  cache-pollution penalty per slot scales differently).

## Test flakes

- `compactor::compactor_tests::test_compact` is flaky under `cargo test`
  parallelism. Passes 3/3 in isolation on both rewrite/moonshot (pre Phase
  1.1) and rewrite/phase1 (post Phase 1.1). Failure mode: hpfile panics at
  `cannot read data just fetched in test_compactor/entries.test fileID 0`,
  cascading through ringchannel. Pre-existing test-dir collision, not a
  Phase 1.1 regression. Fix: give the compactor test a unique TempDir
  suffix or mark it `#[serial]` like the metadb tests.

## Known `slow_hashing` wart

- `flusher.rs:261-321` detached-thread variant; Phase 1.1 routes it through
  the same `AtomicUsize` gate inside the detached thread. Verify in nsys
  trace that there isn't a second barrier hiding in the detached path.

## `use_hybridindexer` interaction

- Hybrid indexer's `RefUnit` per-unit locking may not partition cleanly at
  `workers_per_shard > 1` granularity. Phase 2 either proves it does or
  pins `workers_per_shard = 1` when the feature is on.
