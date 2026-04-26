# Benchmarking QMDB

The benchmark utility `speed` is used to benchmark QMDB. It is a command line tool that runs the benchmark and outputs the results to a JSON file.

**Quick start**: run the benchmark with `./run.sh`, which contains default parameters and assumes your SSD is mounted at `/mnt/nvme`.

## Initial setup

QMDB requires a source of entropy to generate the workload. We recommend generating this once for ease of debugging and reproduction. You can use any large file, or generate it from `/dev/urandom`:

```bash
head -c 10M </dev/urandom > randsrc.dat
```

## Running the benchmark (speed)

Assuming your SSD is mounted at `/mnt/nvme`:

```bash
ulimit -n 65535
# Smoke / dev bench (5M is the smallest size that satisfies the
# bench's blocks_for_db_population >= tps_blocks precondition with
# release defaults).
cargo run --release --features cuda --bin speed -- --entry-count 5000000

# Production-ish bench: 40M entries on this NVMe.
cargo run --release --features cuda --bin speed -- --entry-count 40000000

# Runs the benchmark with 7 billion entries
cargo run --bin speed --release -- \
    --db-dir /mnt/nvme/QMDB \
    --entry-count 7000000000 \
    --ops-per-block 1000000 \
    --hover-recreate-block 100 \
    --hover-write-block 100 \
    --hover-interval 1000 \
    --tps-blocks 500
```

## Environment toggles

- `SKIPPY_TRACE=1` — emit `TRACE shard=S height=H phase=P us=US` lines per
  per-shard flusher phase. Adds noticeable wall-clock overhead (eprintln IO);
  use only to read phase **shares**, not absolute timings.
- `SKIPPY_USE_GPU_RESIDENT=1` — opt into the legacy GPU-resident upper-tree
  sync path. Off by default; the per-level path is faster and produces
  byte-identical roots (see `qmdb/tests/sync_upper_nodes_parity.rs`).
- `SKIPPY_WORKERS_PER_SHARD=N` — set runtime `Topology.workers_per_shard`.
  Default `1`. `>1` enables the experimental Phase 2.4-v2 parallel
  indexer-read path; `W=1` is byte-identical to the prior production code.

## Reference baselines (skippy-dev: 5900X · RTX 4080 SUPER · NVMe Gen4)

| Bench size | elapsed | block_pop | updates | reads | transactions |
|---|---:|---:|---:|---:|---:|
| 5M cuda  | 10.5s | 14.1/s  | 1.68M/s | 1.55M/s | — |
| 40M cuda | 49.5s | 11.1/s  | 1.35M/s | 1.60M/s | 47.5K/s |

Per-block JSON files in `bench/results/`. These are the moonshot baselines
post the 2026-04-26 per-level-default flip + NULL_TWIG.twig_root parity fix.

## Arguments for speed

We document key arguments to the benchmarking utility `speed` here. Refer to `src/cli.rs` for all arguments.

### Essential arguments to provide values for

- **`entry_count`**: Number of entries to populate the database with. Default is 500 million.
- **`db_dir`**: Directory to store the database's persistent files. You should set this to a directory on your SSD, e.g., `/mnt/nvme/QMDB`. Default is `/tmp/QMDB`.
- **`output_filename`**: Output file for benchmark results. Default is `results.json`.
- **`randsrc_filename`**: Source of randomness for workload generation. Default is `./randsrc.dat`.

### Benchmarking workload

These argments control the workload and can be adjusted accordingly or left as the default. QMDB supports pack multiple transactions into one task, although its clients usually only have one transaction in a task. "Hover tasks" in the code refers to benchmarking tasks run periodically during the population of the database.

- **`ops_per_block`**: Target number of operations per block. Default is 100,000.
- **`hover_interval`**: Interval (in blocks) between benchmarking tasks. Default is 10,000 blocks in release mode.
- **`hover_recreate_block`**: Number of blocks to insert during benchmarking. Default is 50 blocks in release mode.
- **`hover_write_block`**: Number of blocks to write during benchmarking. Default is 50 blocks in release mode.
- **`tps_blocks`**: When benchmarking transactions per second (TPS), this sets the number of blocks to run. Default is 50 in release mode.
- **`changesets_per_task`**: Number of changesets per task. Each changeset corresponds to a transaction. Default is 2.

## Troubleshooting

- Ensure `ulimit` is set high enough to handle file descriptors.
- Verify the SSD is correctly mounted at `/mnt/nvme`.
- Ensure the `randsrc.dat` file is present
