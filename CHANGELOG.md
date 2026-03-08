# Changelog

All notable changes to SkippyDB will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.2] - 2026-03-08

### Fixed
- `GpuHasher` now implements `Send + Sync` for thread-safe usage behind `Arc` ([etofdn/etovm#2](https://github.com/etofdn/etovm/issues/2))
- Wrapped `result::stream::synchronize` calls in `unsafe {}` blocks (required by cudarc 0.12)

### Changed
- Replaced `println!` with structured `log` crate macros in all library code
- Replaced bare `.unwrap()` with `.expect()` messages in critical paths (lib.rs, flusher.rs, tree.rs)

## [0.2.1] - 2026-03-05

### Changed
- Bumped `flash-map` dependency from 0.1 to 0.5 (removed `cpu-fallback`, default is now `rayon`)

## [0.2.0] - 2026-03-05

### Added
- GPU acceleration: SoA kernels, fused active bits, async pipelining, auto-adaptive kernel selection
- Multi-GPU support via `MultiGpuHasher` with round-robin shard dispatch
- GPU-resident upper tree sync via `GpuNodeStore` (backed by `flash-map`)
- GPU integration guide (`docs/gpu-integration-guide.md`)
- Crate-level and module-level `//!` doc comments for docs.rs
- Crate README for crates.io
- Published to crates.io as `skippydb`

### Changed
- Renamed crate from `qmdb` to `skippydb`
- Removed LayerZero branding, updated attribution
- Decoupled `hpfile_all_in_mem` feature for crates.io compatibility

## [0.1.0] - 2025-01-01

### Added
- Initial release as `qmdb`
- SHA256 Merkle tree with twig-based structure
- Sharded append-only entry files
- Incremental tree synchronization
- Key-to-position indexing
- Block-level flush pipeline
- Background compaction
- AES-GCM encryption at rest (optional)
- Linux io_uring direct I/O (optional)
- Stateless proof generation and verification
