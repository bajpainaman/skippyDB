# Contributing to KyumDB

Thank you for your interest in contributing to KyumDB! This guide covers everything you need to get started.

## Table of Contents

- [Setting Expectations](#setting-expectations)
- [Development Setup](#development-setup)
- [Project Layout](#project-layout)
- [Code Style](#code-style)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Architecture Quick Reference](#architecture-quick-reference)
- [Common Tasks](#common-tasks)
- [Licensing and Copyright](#licensing-and-copyright)

---

## Setting Expectations

External contributors are encouraged to submit issues and pull requests. That being said, not all issues will be addressed nor pull requests merged (even if well-intentioned).

KyumDB provides reliable and high-performance verifiable database storage. Contributions that do not advance these goals may not be accepted. This could include (but is not limited to) replacing code with external dependencies, implementing optional functionality, and/or introducing algorithms that substantially increase complexity for marginal performance improvements.

**Almost always welcome:**
- Bug fixes with regression tests
- New tests and benchmarks
- Documentation improvements
- Performance improvements with benchmark evidence

---

## Development Setup

### Prerequisites

```bash
# Ubuntu / Debian
sudo apt-get install -y g++ linux-libc-dev libclang-dev unzip libjemalloc-dev make

# Or use the included script
./install-prereqs-ubuntu.sh
```

### Building

```bash
# Standard build
cargo build

# Release build (enables SHA-NI, jemalloc)
cargo build --release

# With GPU support (requires CUDA 12.0+)
cargo build --features cuda

# With all features
cargo build --all-features
```

### Running Tests

```bash
# Run all tests (recommended: nextest for parallel execution)
cargo nextest run

# Standard cargo test
cargo test

# GPU tests (requires NVIDIA GPU)
cargo test --features cuda -- gpu

# Specific crate
cargo test -p qmdb
cargo test -p hpfile

# Specific module
cargo test -p qmdb --lib merkletree
cargo test -p qmdb --lib entryfile
```

### Running Benchmarks

```bash
# Criterion hash benchmarks
cargo bench --bench hash_benchmarks

# Speed benchmark
head -c 10M </dev/urandom > randsrc.dat
cargo run --release --bin speed -- --entry-count 4000000
```

---

## Project Layout

```
kyumdb/
├── qmdb/               # Core library (the main crate)
│   ├── src/
│   │   ├── lib.rs       # AdsCore, AdsWrap, ADS trait — entry point
│   │   ├── config.rs    # Configuration
│   │   ├── def.rs       # Constants
│   │   ├── entryfile/   # Entry storage layer
│   │   ├── merkletree/  # Merkle commitment layer
│   │   ├── indexer/     # Key → position mapping
│   │   ├── gpu/         # CUDA acceleration
│   │   ├── tasks/       # Task/block pipeline
│   │   ├── seqads/      # Sequential validation mode
│   │   ├── stateless/   # Stateless validation
│   │   └── utils/       # Hashing, changeset, helpers
│   ├── examples/        # Runnable examples
│   ├── benches/         # Criterion benchmarks
│   └── tests/           # Integration tests
├── hpfile/              # Head-prunable file crate (standalone)
├── bench/               # Performance benchmark binary
└── docs/                # Extended documentation
```

### Key files to understand first

1. **`qmdb/src/def.rs`** — All constants (shard counts, tree geometry, operation codes)
2. **`qmdb/src/entryfile/entry.rs`** — Entry binary format and serialization
3. **`qmdb/src/merkletree/twig.rs`** — Twig structure, ActiveBits
4. **`qmdb/src/merkletree/tree.rs`** — Tree, UpperTree, the core Merkle logic
5. **`qmdb/src/lib.rs`** — AdsCore, AdsWrap, the public API surface

---

## Code Style

### Formatting

This repository uses default `cargo fmt` and `cargo clippy` rules, treating warnings as errors.

```bash
# Check formatting and lints
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --all -- --check

# Auto-fix formatting
cargo fmt --all
```

### Guidelines

- Follow Rust idioms. Prefer `&[u8]` over `Vec<u8>` for input parameters.
- Use `#[cfg(feature = "...")]` guards for optional functionality.
- Keep `unsafe` blocks minimal and document safety invariants.
- Performance-critical code should have benchmark coverage.
- Avoid introducing new dependencies without discussion.

### Naming Conventions

| Item | Convention | Example |
|---|---|---|
| Structs | PascalCase | `EntryFile`, `GpuHasher` |
| Functions | snake_case | `batch_node_hash`, `sync_mtree` |
| Constants | SCREAMING_SNAKE | `SHARD_COUNT`, `TWIG_SHIFT` |
| Feature flags | snake_case | `cuda`, `tee_cipher`, `directio` |
| Test functions | `test_` prefix | `test_gpu_node_hash_matches_cpu` |

---

## Testing

### Test Categories

| Category | Location | Run Command |
|---|---|---|
| Unit tests | `#[cfg(test)]` in source files | `cargo test -p qmdb --lib` |
| Integration tests | `qmdb/tests/` | `cargo test -p qmdb --test '*'` |
| GPU tests | `qmdb/src/gpu/gpu_tests.rs` | `cargo test --features cuda -- gpu` |
| HPFile tests | `hpfile/src/lib.rs` | `cargo test -p hpfile` |
| Benchmarks | `qmdb/benches/` | `cargo bench` |

### Writing Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entry_roundtrip() {
        let entry = Entry {
            key: b"test_key",
            value: b"test_value",
            next_key_hash: &[0xAB; 32],
            version: 42,
            serial_number: 100,
        };

        let mut buf = [0u8; 1024];
        let size = entry.dump(&mut buf, &[]);

        let bz = EntryBz { bz: &buf[..size] };
        assert_eq!(bz.key(), b"test_key");
        assert_eq!(bz.value(), b"test_value");
        assert_eq!(bz.version(), 42);
        assert_eq!(bz.serial_number(), 100);
    }
}
```

For tests that create temporary directories, use `TempDir`:

```rust
use qmdb::test_helper::TempDir;

#[test]
fn test_with_temp_dir() {
    let dir = "my_test_dir";
    let _tmp = TempDir::new(dir);  // cleaned up on drop

    // ... test code using dir ...
}
```

For GPU tests, always handle missing CUDA gracefully:

```rust
#[test]
fn test_gpu_feature() {
    let gpu = match GpuHasher::new(10000) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Skipping GPU test (no CUDA device): {}", e);
            return;
        }
    };
    // ... test with gpu ...
}
```

---

## Pull Request Process

1. **Fork and branch**: Create a feature branch from `main`.
2. **Make changes**: Keep commits focused and well-described.
3. **Test**: Run `cargo nextest run` and `cargo clippy --all-targets --all-features -- -D warnings`.
4. **Format**: Run `cargo fmt --all`.
5. **Submit PR**: Describe what changed and why. Include benchmark results for performance changes.

### PR Checklist

- [ ] All existing tests pass (`cargo nextest run`)
- [ ] New tests added for new functionality
- [ ] `cargo clippy` passes with no warnings
- [ ] `cargo fmt` applied
- [ ] Documentation updated if public API changed
- [ ] No new `unsafe` blocks without safety comments
- [ ] Benchmark results included for performance PRs

---

## Architecture Quick Reference

### Data Flow

```
Client → Prefetcher → Updater → EntryBuffer → Flusher → SSD
                                      │
                              Indexer (B-tree)
```

### Key Constants

| Constant | Value | Purpose |
|---|---|---|
| `SHARD_COUNT` | 16 | Parallel shard partitioning |
| `LEAF_COUNT_IN_TWIG` | 2048 | Entries per Merkle sub-tree |
| `TWIG_SHIFT` | 11 | `twig_id = serial_number >> 11` |
| `MAX_TREE_LEVEL` | 64 | Maximum Merkle tree height |
| `IN_BLOCK_IDX_BITS` | 24 | `task_id = (height << 24) \| idx` |

### Threading Model

- 16 Updater threads (one per shard)
- 16 Flusher threads (one per shard)
- 16 Compactor threads (one per shard)
- 512 Prefetcher threads (configurable)
- `rayon` thread pool for parallel Merkle hashing

---

## Common Tasks

### Adding a new feature flag

1. Add to `qmdb/Cargo.toml` under `[features]`
2. Guard code with `#[cfg(feature = "your_feature")]`
3. Add to CI matrix if it needs special testing
4. Document in the README feature flags table

### Adding a new CUDA kernel

1. Add the kernel to `qmdb/src/gpu/sha256_kernel.cu`
2. Register it in `GpuHasher::new_on_device()` via `device.load_ptx()`
3. Add a Rust wrapper method in `gpu_hasher.rs`
4. Add tests in `gpu_tests.rs` that compare GPU vs CPU results
5. Add benchmarks in `benches/hash_benchmarks.rs`

### Modifying the entry format

1. Update the binary layout in `entryfile/entry.rs`
2. Update `ENTRY_FIXED_LENGTH` and `ENTRY_BASE_LENGTH` in `def.rs`
3. Update serialization in `Entry::dump()` and deserialization in `EntryBz` methods
4. Run the full test suite — many tests depend on exact byte layouts

---

## Releases

Releases are automatically published to `cargo` by [GitHub Actions](.github/workflows/publish.yml) whenever a version update is merged into the `main` branch.

To increment the patch version:

```bash
./scripts/bump_versions.sh
```

---

## Licensing and Copyright

You agree that any work submitted to this repository shall be dual-licensed under the included [Apache 2.0](./LICENSE-APACHE) and [MIT](./LICENSE-MIT) licenses, without any additional terms or conditions. Additionally, you agree to release your copyright interest in said work to the public domain, such that anyone is free to use, modify, and distribute your contributions without restriction.

---

## Support

- **Bug reports**: [GitHub Issues](https://github.com/bajpainaman/kyumdb/issues)
- **Discussions**: [GitHub Discussions](https://github.com/bajpainaman/kyumdb/discussions)
