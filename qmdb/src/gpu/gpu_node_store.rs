//! GPU-resident node store for the upper Merkle tree.
//!
//! Uses flash-map's FlashMap<u64, [u8; 32]> to keep tree node hashes on the GPU.
//! During `sync_upper_nodes`, child hashes are fetched from the GPU-resident store,
//! hashed via the SoA SHA256 kernel (device-to-device), and results are written
//! back — all without H↔D transfers. Only the final root hash is transferred to CPU.
//!
//! Key type: u64 (NodePos encoded as (level << 56) | nth)
//! Value type: [u8; 32] (SHA256 hash)

use flash_map::{FlashMap, FlashMapError};

use super::gpu_hasher::GpuHasher;

/// Initial capacity for the GPU node store (power of 2).
/// With 50% max load factor, supports up to 512K nodes before needing resize.
const DEFAULT_NODE_STORE_CAPACITY: usize = 1 << 20; // 1M slots

/// GPU-resident node store wrapping FlashMap<u64, [u8; 32]>.
///
/// Stores upper tree nodes on the GPU. Provides methods to:
/// - Populate nodes from CPU (initial load / twig root eviction)
/// - Fetch child pairs on GPU, hash them, and store parent results (all device-side)
/// - Retrieve the final root hash to CPU
pub struct GpuNodeStore {
    map: FlashMap<u64, [u8; 32]>,
}

impl GpuNodeStore {
    /// Create a new GPU-resident node store.
    ///
    /// Uses Identity hash (first 8 bytes of key as u64), which is ideal since
    /// NodePos keys are already well-distributed u64 values.
    pub fn new() -> Result<Self, FlashMapError> {
        Self::with_capacity(DEFAULT_NODE_STORE_CAPACITY)
    }

    /// Create with a specific capacity (rounded up to power of 2).
    pub fn with_capacity(capacity: usize) -> Result<Self, FlashMapError> {
        let map = FlashMap::<u64, [u8; 32]>::with_capacity(capacity)?;
        Ok(Self { map })
    }

    /// Bulk-insert nodes from CPU host data.
    /// Used for initial population and twig root eviction.
    pub fn insert_from_host(&mut self, pairs: &[(u64, [u8; 32])]) -> Result<usize, FlashMapError> {
        self.map.bulk_insert(pairs)
    }

    /// Bulk-get nodes to CPU host data.
    /// Used for reading specific nodes back (e.g., root hash, edge nodes).
    pub fn get_to_host(&self, keys: &[u64]) -> Result<Vec<Option<[u8; 32]>>, FlashMapError> {
        self.map.bulk_get(keys)
    }

    /// Get a single node value on the host. Convenience wrapper.
    pub fn get_single(&self, key: u64) -> Result<Option<[u8; 32]>, FlashMapError> {
        let results = self.map.bulk_get(&[key])?;
        Ok(results.into_iter().next().unwrap())
    }

    /// Perform one level of upper tree synchronization entirely on the GPU.
    ///
    /// For each index `i` in `n_list`, computes:
    ///   parent_hash = SHA256(level_byte || left_child_hash || right_child_hash)
    ///
    /// Where:
    ///   left_child  = node_store[NodePos(child_level, 2*i)]
    ///   right_child = node_store[NodePos(child_level, 2*i+1)]
    ///   result stored at node_store[NodePos(level, i)]
    ///
    /// All data stays on GPU. Only n_list positions are transferred H→D (tiny).
    ///
    /// Returns the next-level n_list (deduplicated i/2 values).
    pub fn sync_level_on_device(
        &mut self,
        gpu: &GpuHasher,
        level: i64,
        n_list: &[u64],
    ) -> Result<Vec<u64>, String> {
        if n_list.is_empty() {
            return Ok(Vec::new());
        }

        let n = n_list.len();
        let child_level = level - 1;

        // Build left/right child position keys on CPU (tiny: n * 8 bytes)
        let mut left_keys: Vec<u64> = Vec::with_capacity(n);
        let mut right_keys: Vec<u64> = Vec::with_capacity(n);
        let mut parent_keys: Vec<u64> = Vec::with_capacity(n);

        for &i in n_list {
            left_keys.push(encode_node_pos(child_level as u64, 2 * i));
            right_keys.push(encode_node_pos(child_level as u64, 2 * i + 1));
            parent_keys.push(encode_node_pos(level as u64, i));
        }

        // Upload position keys to GPU (small transfer: 3 * n * 8 bytes)
        let d_left_keys = self
            .map
            .upload_keys(&left_keys)
            .map_err(|e| format!("upload left keys: {e}"))?;
        let d_right_keys = self
            .map
            .upload_keys(&right_keys)
            .map_err(|e| format!("upload right keys: {e}"))?;
        let d_parent_keys = self
            .map
            .upload_keys(&parent_keys)
            .map_err(|e| format!("upload parent keys: {e}"))?;

        // Fetch left children from GPU-resident store (device → device)
        let (d_left_vals, _d_left_found) = self
            .map
            .bulk_get_device(&d_left_keys, n)
            .map_err(|e| format!("bulk_get left: {e}"))?;

        // Fetch right children from GPU-resident store (device → device)
        let (d_right_vals, _d_right_found) = self
            .map
            .bulk_get_device(&d_right_keys, n)
            .map_err(|e| format!("bulk_get right: {e}"))?;

        // Create level bytes on GPU (tiny: n bytes, all same value)
        let d_levels = gpu.fill_device_bytes((level - 1) as u8, n);

        // SHA256 on device: hash(level || left || right) → parent hashes
        // d_left_vals and d_right_vals are each n*32 bytes (contiguous [u8;32] values)
        let d_results = gpu.batch_node_hash_device_soa(&d_levels, &d_left_vals, &d_right_vals, n);

        // Sync GPU before inserting results (kernel must complete)
        gpu.sync();

        // Store parent hashes back into GPU-resident store (device → device)
        self.map
            .bulk_insert_device(&d_parent_keys, &d_results, n)
            .map_err(|e| format!("bulk_insert results: {e}"))?;

        // Build next level's n_list on CPU (cheap)
        let mut new_list = Vec::with_capacity(n);
        for &i in n_list {
            if new_list.is_empty() || *new_list.last().unwrap() != i / 2 {
                new_list.push(i / 2);
            }
        }
        Ok(new_list)
    }

    /// Sync all upper tree levels from `first_level` to `max_level` on GPU.
    ///
    /// The entire computation stays on GPU. Only transfers:
    /// - H→D: n_list positions per level (tiny, ~64-128 u64s = 512-1024 bytes)
    /// - D→H: final root hash (32 bytes)
    ///
    /// Returns (final_n_list, root_hash).
    pub fn sync_upper_nodes_on_device(
        &mut self,
        gpu: &GpuHasher,
        mut n_list: Vec<u64>,
        first_level: i64,
        max_level: i64,
    ) -> Result<(Vec<u64>, [u8; 32]), String> {
        if !n_list.is_empty() {
            for level in first_level..=max_level {
                n_list = self.sync_level_on_device(gpu, level, &n_list)?;
            }
        }

        // Fetch root hash to CPU (single 32-byte D→H transfer)
        let root_pos = encode_node_pos(max_level as u64, 0);
        let root = self
            .get_single(root_pos)
            .map_err(|e| format!("get root: {e}"))?
            .unwrap_or([0u8; 32]);

        Ok((n_list, root))
    }

    /// Number of nodes in the store.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Clear all nodes from the GPU store.
    pub fn clear(&mut self) -> Result<(), FlashMapError> {
        self.map.clear()
    }
}

/// Encode a (level, nth) pair into a u64 NodePos value.
/// Matches the encoding in `tree.rs`: `(level << 56) | nth`
#[inline]
fn encode_node_pos(level: u64, nth: u64) -> u64 {
    (level << 56) | nth
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_node_pos() {
        let pos = encode_node_pos(13, 42);
        assert_eq!(pos >> 56, 13);
        assert_eq!((pos << 8) >> 8, 42);
    }

    #[test]
    fn test_gpu_node_store_basic() {
        let mut store = match GpuNodeStore::new() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Skipping GPU node store test (no CUDA): {e}");
                return;
            }
        };

        // Insert some nodes
        let pairs: Vec<(u64, [u8; 32])> = (0..100)
            .map(|i| {
                let key = encode_node_pos(13, i);
                let mut val = [0u8; 32];
                val[0] = i as u8;
                (key, val)
            })
            .collect();

        store.insert_from_host(&pairs).unwrap();
        assert_eq!(store.len(), 100);

        // Read back
        let keys: Vec<u64> = pairs.iter().map(|(k, _)| *k).collect();
        let results = store.get_to_host(&keys).unwrap();
        for (i, result) in results.iter().enumerate() {
            let val = result.unwrap();
            assert_eq!(val[0], i as u8);
        }
    }
}
