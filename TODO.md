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

## Known `slow_hashing` wart

- `flusher.rs:261-321` detached-thread variant; Phase 1.1 routes it through
  the same `AtomicUsize` gate inside the detached thread. Verify in nsys
  trace that there isn't a second barrier hiding in the detached path.

## `use_hybridindexer` interaction

- Hybrid indexer's `RefUnit` per-unit locking may not partition cleanly at
  `workers_per_shard > 1` granularity. Phase 2 either proves it does or
  pins `workers_per_shard = 1` when the feature is on.
