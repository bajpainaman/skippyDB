# SkippyDB bench results

Every JSON file in this directory references the bench host recorded below.
A result captured on different hardware is a different host and lives in a
subdirectory named for that host.

## Bench host: `skippy-dev` (primary, 2026-04-23)

| Component | Value |
|---|---|
| CPU | AMD Ryzen 9 5900X — 12 cores / 24 threads, single socket, single NUMA node |
| GPU | NVIDIA GeForce RTX 4080 SUPER, 16376 MiB VRAM |
| GPU driver | 590.48.01 |
| CUDA toolchain | `nvcc` 13.1.115 (cuda_13.1.r13.1/compiler.37061995_0) |
| RAM | 46 GiB total |
| Storage | `/dev/nvme0n1p2` — NVMe, ext4, mounted `rw,relatime` |
| OS / kernel | Linux 6.17.0-22-generic |
| rustc | 1.93.0 (254b59607 2026-01-19) |
| cargo | 1.93.0 (083ac5135 2025-12-15) |

Relative to the ultraplan's target-validator spec (1× H100/H200 +
enterprise PCIe Gen5 NVMe), this bench host is **tier-2 class**: consumer
GPU, consumer NVMe, consumer CPU. Numbers here are directional for tier-1,
not ceilings. Any extrapolation must explicitly cite PCIe bandwidth
utilization from the nsys trace, not wall-clock throughput.

## File naming

`{branch-or-phase}-{entry-count}.json` — e.g. `main-4m.json`,
`phase1a-4m.json`, `phase3-40m.json`.

## Acceptance numbers

- `main-4m.json` + `main-40m.json` — baseline. Every phase is measured as
  a ratio against these.
- Phase 3 must hit `blocks_per_sec >= 5 × main-4m.blocks_per_sec` on
  `skippy-dev`. If reference-box access is secured by Phase 4 start, a
  second set of `{phase}-ref-{size}.json` files go in `ref-box/`.

## Non-goals for this directory

- No flamegraphs (keep SVGs in `bench/flamegraphs/`).
- No nsys traces (keep .nsys-rep in `bench/nsys/`).
- No microbenchmarks (criterion output lives in `target/criterion/`).

Only ops/s, latency percentiles, bytes/sec, GPU utilization JSON files go
here.
