# skippydb

A high-performance verifiable key-value store optimized for blockchain state storage, featuring SHA256 Merkle trees with optional CUDA GPU acceleration.

## Installation

```bash
cargo add skippydb
# With GPU acceleration:
cargo add skippydb --features cuda
```

## Quick Start

```rust
use skippydb::config::Config;
use skippydb::AdsCore;

let config = Config::from_dir("/path/to/data");
AdsCore::init_dir(&config);
```

## Feature Flags

| Feature | Description | Default |
|---------|-------------|---------|
| `default` | Includes jemalloc allocator | Yes |
| `cuda` | CUDA GPU-accelerated SHA256 batch hashing | No |
| `directio` | Linux io_uring direct I/O for reads | No |
| `hpfile_all_in_mem` | Keep heap files entirely in memory | No |
| `tee_cipher` | AES-GCM encryption for data at rest | No |
| `slow_hashing` | Slow hashing mode for testing | No |

## GPU Acceleration

With `--features cuda`, SkippyDB batches Merkle tree SHA256 operations to CUDA cores:
- 3-5x throughput improvement at batch sizes >10K
- Automatic kernel selection (CPU/AoS/SoA) via `auto_batch_node_hash`
- Async pipelining, fused active bits, GPU-resident upper tree
- Multi-GPU support via `MultiGpuHasher`

See the [GPU Integration Guide](https://github.com/bajpainaman/SkippyDB/blob/main/docs/gpu-integration-guide.md) for usage patterns.

## Documentation

- [API Docs (docs.rs)](https://docs.rs/skippydb)
- [GPU Acceleration Details](https://github.com/bajpainaman/SkippyDB/blob/main/docs/gpu-acceleration.md)
- [GPU Integration Guide](https://github.com/bajpainaman/SkippyDB/blob/main/docs/gpu-integration-guide.md)

## License

Licensed under either of:
- Apache License, Version 2.0
- MIT License

at your option.
