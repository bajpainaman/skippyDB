use aes_gcm::Aes256Gcm;
use rayon;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::{fmt, fs, mem, thread};

use super::twig::{sync_mtree, ActiveBits, TwigMT, NULL_NODE_IN_HIGHER_TREE};
use super::twigfile::{TwigFile, TwigFileWriter};
use super::{proof, twigfile};
use super::{recover, twig};
use crate::def::{
    calc_max_level, ENTRIES_PATH, FIRST_LEVEL_ABOVE_TWIG, LEAF_COUNT_IN_TWIG, MAX_TREE_LEVEL,
    MIN_PRUNE_COUNT, NODE_SHARD_COUNT, TWIG_MASK, TWIG_PATH, TWIG_ROOT_LEVEL, TWIG_SHARD_COUNT,
    TWIG_SHIFT,
};
use crate::entryfile::{Entry, EntryBz};
use crate::entryfile::{EntryFile, EntryFileWriter};
use crate::utils::hasher::{self, Hash32};

/*
             ____TwigRoot___                   Level_12
            /               \
           /                 \
1       leftRoot              activeBitsMTL3   Level_11
2       Level_10        2     activeBitsMTL2
4       Level_9         4     activeBitsMTL1
8       Level_8    8*32bytes  activeBits
16      Level_7
32      Level_6
64      Level_5
128     Level_4
256     Level_3
512     Level_2
1024    Level_1
2048    Level_0
*/

/*         1
     2             3
  4     5       6     7
 8 9   a b     c d   e f
*/

#[derive(Copy, Clone, Eq, Hash, PartialEq)]
pub struct NodePos(u64);

impl NodePos {
    pub fn new(pos: u64) -> NodePos {
        NodePos(pos)
    }
    pub fn pos(level: u64, n: u64) -> NodePos {
        NodePos((level << 56) | n)
    }
    pub fn level(&self) -> u64 {
        self.0 >> 56 // extract the high 8 bits
    }
    pub fn nth(&self) -> u64 {
        (self.0 << 8) >> 8 // extract the low 56 bits
    }
    pub fn as_u64(&self) -> u64 {
        self.0
    }
}

impl fmt::Debug for NodePos {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "NodePos: {:?} {{ level: {}, nth: {} }}",
            self.as_u64(),
            self.level(),
            self.nth()
        )
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct EdgeNode {
    pub pos: NodePos,
    pub value: [u8; 32],
}

/// Maximum level (exclusive) that uses dense Vec-based storage.
/// Levels TWIG_ROOT_LEVEL..DENSE_LEVEL_MAX use Vec<Option<[u8;32]>> for O(1)
/// lookups. Levels DENSE_LEVEL_MAX..MAX_TREE_LEVEL use HashMap for sparse storage.
const DENSE_LEVEL_MAX: usize = 21;

/// Per-shard node storage: dense (Vec-indexed) or sparse (HashMap).
/// Dense shards map `nth / NODE_SHARD_COUNT` → hash via direct indexing.
/// Sparse shards use HashMap<NodePos, hash> for levels with few nodes.
#[derive(Clone)]
pub enum NodeShard {
    Dense(Vec<Option<[u8; 32]>>),
    Sparse(HashMap<NodePos, [u8; 32]>),
}

impl NodeShard {
    fn new_dense() -> Self {
        NodeShard::Dense(Vec::new())
    }

    fn new_sparse() -> Self {
        NodeShard::Sparse(HashMap::new())
    }

    pub fn get(&self, pos: &NodePos) -> Option<&[u8; 32]> {
        match self {
            NodeShard::Dense(v) => {
                let idx = pos.nth() as usize / NODE_SHARD_COUNT;
                v.get(idx).and_then(|opt| opt.as_ref())
            }
            NodeShard::Sparse(m) => m.get(pos),
        }
    }

    pub fn insert(&mut self, pos: NodePos, hash: [u8; 32]) {
        match self {
            NodeShard::Dense(v) => {
                let idx = pos.nth() as usize / NODE_SHARD_COUNT;
                if idx >= v.len() {
                    v.resize(idx + 1, None);
                }
                v[idx] = Some(hash);
            }
            NodeShard::Sparse(m) => {
                m.insert(pos, hash);
            }
        }
    }

    pub fn remove(&mut self, pos: &NodePos) {
        match self {
            NodeShard::Dense(v) => {
                let idx = pos.nth() as usize / NODE_SHARD_COUNT;
                if idx < v.len() {
                    v[idx] = None;
                }
            }
            NodeShard::Sparse(m) => {
                m.remove(pos);
            }
        }
    }

    /// Iterate over all (NodePos, hash) entries in this shard.
    pub fn iter(&self) -> NodeShardIter<'_> {
        match self {
            NodeShard::Dense(v) => NodeShardIter::Dense {
                vec: v,
                idx: 0,
                shard_id: 0, // set by caller via iter_with_shard
                level: 0,    // set by caller via iter_with_shard
            },
            NodeShard::Sparse(m) => NodeShardIter::Sparse(m.iter()),
        }
    }

    /// Iterate with known shard_id and level (needed to reconstruct NodePos
    /// for dense entries).
    pub fn iter_with_context(
        &self,
        shard_id: usize,
        level: usize,
    ) -> NodeShardIter<'_> {
        match self {
            NodeShard::Dense(v) => NodeShardIter::Dense {
                vec: v,
                idx: 0,
                shard_id,
                level,
            },
            NodeShard::Sparse(m) => NodeShardIter::Sparse(m.iter()),
        }
    }
}

/// Iterator over entries in a NodeShard.
pub enum NodeShardIter<'a> {
    Dense {
        vec: &'a Vec<Option<[u8; 32]>>,
        idx: usize,
        shard_id: usize,
        level: usize,
    },
    Sparse(std::collections::hash_map::Iter<'a, NodePos, [u8; 32]>),
}

impl<'a> Iterator for NodeShardIter<'a> {
    type Item = (NodePos, &'a [u8; 32]);

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            NodeShardIter::Dense {
                vec,
                idx,
                shard_id,
                level,
            } => {
                while *idx < vec.len() {
                    let i = *idx;
                    *idx += 1;
                    if let Some(ref hash) = vec[i] {
                        let nth = i * NODE_SHARD_COUNT + *shard_id;
                        let pos = NodePos::pos(*level as u64, nth as u64);
                        return Some((pos, hash));
                    }
                }
                None
            }
            NodeShardIter::Sparse(iter) => {
                iter.next().map(|(pos, hash)| (*pos, hash))
            }
        }
    }
}

#[derive(Clone)]
pub struct UpperTree {
    pub my_shard_id: usize,
    // the nodes in high level tree (higher than twigs)
    // this variable can be recovered from saved edge nodes and activeTwigs
    pub nodes: Vec<Vec<NodeShard>>, //MaxUpperLevel*NodeShardCount maps
    // this variable can be recovered from entry file
    pub active_twig_shards: Vec<HashMap<u64, Box<twig::Twig>>>, //TwigShardCount maps
}

impl UpperTree {
    pub fn empty() -> Self {
        Self {
            my_shard_id: 0,
            nodes: Vec::with_capacity(0),
            active_twig_shards: Vec::with_capacity(0),
        }
    }

    pub fn new(my_shard_id: usize) -> Self {
        let nodes: Vec<Vec<NodeShard>> = (0..MAX_TREE_LEVEL)
            .map(|level| {
                if level < DENSE_LEVEL_MAX {
                    vec![NodeShard::new_dense(); NODE_SHARD_COUNT]
                } else {
                    vec![NodeShard::new_sparse(); NODE_SHARD_COUNT]
                }
            })
            .collect();
        let active_twig_shards = vec![HashMap::<u64, Box<twig::Twig>>::new(); TWIG_SHARD_COUNT];

        Self {
            my_shard_id,
            nodes,
            active_twig_shards,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.len() == 0
    }

    pub fn add_twigs(&mut self, twig_map: HashMap<u64, Box<twig::Twig>>) {
        for (twig_id, twig) in twig_map {
            let (shard_idx, key) = get_shard_idx_and_key(twig_id);
            self.active_twig_shards[shard_idx].insert(key, twig);
        }
    }

    pub fn get_twig(&mut self, twig_id: u64) -> Option<&mut Box<twig::Twig>> {
        let (shard_idx, key) = get_shard_idx_and_key(twig_id);
        self.active_twig_shards[shard_idx].get_mut(&key)
    }

    pub fn get_twig_root(&self, n: u64) -> Option<&[u8; 32]> {
        let (shard_idx, key) = get_shard_idx_and_key(n);
        let twig_option = self.active_twig_shards[shard_idx].get(&key);
        match twig_option {
            Some(v) => Some(&v.twig_root),
            None => {
                // the twig has been evicted
                let pos = NodePos::pos(TWIG_ROOT_LEVEL as u64, n);
                self.get_node(pos)
            }
        }
    }

    pub fn set_node_copy(&mut self, pos: NodePos, node: &[u8; 32]) {
        let mut n = [0; 32];
        n.copy_from_slice(node);
        // self.nodes[pos.level() as usize][pos.nth() as usize % NODE_SHARD_COUNT].insert(pos, n);
        self.set_node(pos, n);
    }

    pub fn set_node(&mut self, pos: NodePos, node: [u8; 32]) {
        self.nodes[pos.level() as usize][pos.nth() as usize % NODE_SHARD_COUNT]
            .insert(pos, node);
    }

    pub fn get_node(&self, pos: NodePos) -> Option<&[u8; 32]> {
        self.nodes[pos.level() as usize][pos.nth() as usize % NODE_SHARD_COUNT]
            .get(&pos)
    }

    fn delete_node(&mut self, pos: NodePos) {
        self.nodes[pos.level() as usize][pos.nth() as usize % NODE_SHARD_COUNT]
            .remove(&pos);
    }

    pub fn prune_nodes(&mut self, start: u64, end: u64, youngest_twig_id: u64) -> Vec<u8> {
        let max_level = calc_max_level(youngest_twig_id);
        self.remove_useless_nodes(start, end, max_level);
        recover::edge_nodes_to_bytes(&self.get_edge_nodes(end, max_level))
    }

    fn remove_useless_nodes(&mut self, start: u64, end: u64, max_level: i64) {
        let mut cur_start = start;
        let mut cur_end = end;
        for level in TWIG_ROOT_LEVEL..=max_level {
            let mut end_back = cur_end;
            if cur_end % 2 != 0 && level != TWIG_ROOT_LEVEL {
                end_back -= 1;
            }

            let mut start_back = cur_start;
            start_back = start_back.saturating_sub(1);
            for i in start_back..end_back {
                let pos = NodePos::pos(level as u64, i);
                self.delete_node(pos);
            }
            cur_start >>= 1;
            cur_end >>= 1;
        }
    }

    fn get_edge_nodes(&self, end: u64, max_level: i64) -> Vec<EdgeNode> {
        let mut cur_end = end;
        let mut new_edge_nodes = Vec::new();
        for level in TWIG_ROOT_LEVEL..=max_level {
            let mut end_back = cur_end;
            if cur_end % 2 != 0 && level != TWIG_ROOT_LEVEL {
                end_back -= 1;
            }
            let pos = NodePos::pos(level as u64, end_back);
            if let Some(v) = self.get_node(pos) {
                new_edge_nodes.push(EdgeNode { pos, value: *v });
            } else {
                panic!(
                    "What? can not find shard_id={} max_level={} level={} end={} cur_end={}",
                    self.my_shard_id, max_level, level, end, cur_end
                );
            }
            cur_end >>= 1;
        }
        new_edge_nodes
    }

    pub fn sync_nodes_by_level(
        &mut self,
        level: i64,
        n_list: Vec<u64>,
        youngest_twig_id: u64,
    ) -> Vec<u64> {
        let max_n = max_n_at_level(youngest_twig_id, level);
        let pos = NodePos::pos(level as u64, max_n);
        self.set_node_copy(pos, &NULL_NODE_IN_HIGHER_TREE[level as usize]);
        let pos = NodePos::pos(level as u64, max_n + 1);
        self.set_node_copy(pos, &NULL_NODE_IN_HIGHER_TREE[level as usize]);
        // take written_nodes out from self.nodes
        self.nodes.push(Vec::new()); // push a placeholder that will be removed
        let mut written_nodes = self.nodes.swap_remove(level as usize);

        // Pre-partition n_list by node shard to avoid 75% wasted iteration
        let mut shard_lists: [Vec<u64>; NODE_SHARD_COUNT] = Default::default();
        for &i in &n_list {
            shard_lists[i as usize % NODE_SHARD_COUNT].push(i);
        }

        let mut new_list = Vec::with_capacity(n_list.len());
        rayon::scope(|s| {
            // run hashing in parallel across node shards
            for (shard_id, (nodes, shard_list)) in written_nodes
                .iter_mut()
                .zip(shard_lists.iter())
                .enumerate()
            {
                let upper_tree = &*self; // change a mutable borrow to an immutable borrow
                s.spawn(move |_| do_sync_job(upper_tree, nodes, level, shard_id, shard_list));
            }
            for &i in n_list.iter() {
                if new_list.is_empty() || *new_list.last().unwrap() != i / 2 {
                    new_list.push(i / 2);
                }
            }
        });

        // return written_nodes back to self.nodes
        self.nodes.push(written_nodes);
        self.nodes.swap_remove(level as usize); // the placeholder is removed
        new_list
    }

    pub fn sync_upper_nodes(
        &mut self,
        mut n_list: Vec<u64>,
        youngest_twig_id: u64,
    ) -> (Vec<u64>, [u8; 32]) {
        let max_level = calc_max_level(youngest_twig_id);
        if !n_list.is_empty() {
            for level in FIRST_LEVEL_ABOVE_TWIG..=max_level {
                n_list = self.sync_nodes_by_level(level, n_list, youngest_twig_id);
            }
        }
        let root = *self.get_node(NodePos::pos(max_level as u64, 0)).unwrap();
        (n_list, root)
    }

    pub fn evict_twigs(
        &mut self,
        n_list: Vec<u64>,
        twig_evict_start: u64,
        twig_evict_end: u64,
    ) -> Vec<u64> {
        let new_list = self.sync_mt_for_active_bits_phase2(n_list);
        // run the pending twig-eviction jobs
        // they were not evicted earlier because sync_mt_for_active_bits_phase2 needs their content
        for twig_id in twig_evict_start..twig_evict_end {
            // evict the twig and store its twigRoot in nodes
            let pos = NodePos::pos(TWIG_ROOT_LEVEL as u64, twig_id);
            let twig_root = self.get_twig(twig_id).unwrap().twig_root;
            self.set_node_copy(pos, &twig_root);
            let (shard_idx, key) = get_shard_idx_and_key(twig_id);
            self.active_twig_shards[shard_idx].remove(&key);
        }
        new_list
    }

    pub fn sync_mt_for_active_bits_phase2(&mut self, mut n_list: Vec<u64>) -> Vec<u64> {
        // Pre-partition L2 items by twig shard
        let mut l2_by_shard: [Vec<u64>; TWIG_SHARD_COUNT] = Default::default();
        for &i in &n_list {
            let twig_id = i >> 1;
            let (s, _) = get_shard_idx_and_key(twig_id);
            l2_by_shard[s].push(i);
        }

        let mut new_list = Vec::with_capacity(n_list.len());
        rayon::scope(|s| {
            for (sid, twig_shard) in self.active_twig_shards.iter_mut().enumerate() {
                let shard_items = &l2_by_shard[sid];
                s.spawn(move |_| {
                    for &i in shard_items {
                        let twig_id = i >> 1;
                        let (_, k) = get_shard_idx_and_key(twig_id);
                        twig_shard.get_mut(&k).unwrap().sync_l2((i & 1) as i32);
                    }
                });
            }

            for i in &n_list {
                if new_list.is_empty() || *new_list.last().unwrap() != i / 2 {
                    new_list.push(i / 2);
                }
            }
        });

        mem::swap(&mut new_list, &mut n_list);
        new_list.clear();

        // Pre-partition L3+top items by twig shard
        let mut l3_by_shard: [Vec<u64>; TWIG_SHARD_COUNT] = Default::default();
        for &twig_id in &n_list {
            let (s, _) = get_shard_idx_and_key(twig_id);
            l3_by_shard[s].push(twig_id);
        }

        rayon::scope(|s| {
            for (sid, twig_shard) in self.active_twig_shards.iter_mut().enumerate() {
                let shard_items = &l3_by_shard[sid];
                s.spawn(move |_| {
                    for &twig_id in shard_items {
                        let (_, k) = get_shard_idx_and_key(twig_id);
                        twig_shard.get_mut(&k).unwrap().sync_l3();
                        twig_shard.get_mut(&k).unwrap().sync_top();
                    }
                });
            }

            for i in &n_list {
                if new_list.is_empty() || *new_list.last().unwrap() != i / 2 {
                    new_list.push(i / 2);
                }
            }
        });

        new_list
    }

    /// GPU-accelerated sync_nodes_by_level.
    /// Batches all node hashes at a given level into a single GPU dispatch.
    #[cfg(feature = "cuda")]
    pub fn sync_nodes_by_level_gpu(
        &mut self,
        gpu: &crate::gpu::GpuHasher,
        level: i64,
        n_list: Vec<u64>,
        youngest_twig_id: u64,
    ) -> Vec<u64> {
        use crate::gpu::NodeHashJob;

        let max_n = max_n_at_level(youngest_twig_id, level);
        let pos = NodePos::pos(level as u64, max_n);
        self.set_node_copy(pos, &NULL_NODE_IN_HIGHER_TREE[level as usize]);
        let pos = NodePos::pos(level as u64, max_n + 1);
        self.set_node_copy(pos, &NULL_NODE_IN_HIGHER_TREE[level as usize]);

        // Collect all hash jobs for this level
        let mut jobs = Vec::with_capacity(n_list.len());
        let mut job_positions: Vec<NodePos> = Vec::with_capacity(n_list.len());

        for &i in &n_list {
            let pos = NodePos::pos(level as u64, i);
            if level == FIRST_LEVEL_ABOVE_TWIG {
                let left = self
                    .get_twig_root(2 * i)
                    .copied()
                    .unwrap_or(twig::NULL_TWIG.twig_root);
                let right = self
                    .get_twig_root(2 * i + 1)
                    .copied()
                    .unwrap_or(twig::NULL_TWIG.twig_root);
                jobs.push(NodeHashJob {
                    level: level as u8 - 1,
                    left,
                    right,
                });
            } else {
                let child_nodes = self.nodes.get((level - 1) as usize).unwrap();
                let node_pos_l = NodePos::pos((level - 1) as u64, 2 * i);
                let node_pos_r = NodePos::pos((level - 1) as u64, 2 * i + 1);
                let sl = node_pos_l.nth() as usize % NODE_SHARD_COUNT;
                let sr = node_pos_r.nth() as usize % NODE_SHARD_COUNT;
                let left = *child_nodes[sl].get(&node_pos_l).unwrap();
                let right = *child_nodes[sr].get(&node_pos_r).unwrap();
                jobs.push(NodeHashJob {
                    level: level as u8 - 1,
                    left,
                    right,
                });
            }
            job_positions.push(pos);
        }

        // GPU batch hash
        if !jobs.is_empty() {
            let results = gpu.auto_batch_node_hash(&jobs);
            for (idx, pos) in job_positions.iter().enumerate() {
                self.set_node(*pos, results[idx]);
            }
        }

        // Build next level's n_list
        let mut new_list = Vec::with_capacity(n_list.len());
        for &i in &n_list {
            if new_list.is_empty() || *new_list.last().unwrap() != i / 2 {
                new_list.push(i / 2);
            }
        }
        new_list
    }

    /// GPU-accelerated sync_upper_nodes.
    #[cfg(feature = "cuda")]
    pub fn sync_upper_nodes_gpu(
        &mut self,
        gpu: &crate::gpu::GpuHasher,
        mut n_list: Vec<u64>,
        youngest_twig_id: u64,
    ) -> (Vec<u64>, [u8; 32]) {
        let max_level = calc_max_level(youngest_twig_id);
        if !n_list.is_empty() {
            for level in FIRST_LEVEL_ABOVE_TWIG..=max_level {
                n_list = self.sync_nodes_by_level_gpu(gpu, level, n_list, youngest_twig_id);
            }
        }
        let root = *self.get_node(NodePos::pos(max_level as u64, 0)).unwrap();
        (n_list, root)
    }

    /// GPU-resident sync_upper_nodes: entire upper tree computation stays on GPU.
    ///
    /// Instead of per-level H↔D round-trips (69 sync calls), this:
    /// 1. Populates a GPU-resident FlashMap with all relevant nodes + twig roots
    /// 2. Runs all level hashing on GPU (device-to-device via flash-map)
    /// 3. Transfers only the root hash back to CPU (32 bytes)
    /// 4. Updates CPU-side HashMaps with the results for edge node/prune operations
    ///
    /// Reduces sync calls from ~54 to ~4 and eliminates ~368KB of PCIe transfers.
    #[cfg(feature = "cuda")]
    pub fn sync_upper_nodes_gpu_resident(
        &mut self,
        gpu: &crate::gpu::GpuHasher,
        gpu_store: &mut crate::gpu::GpuNodeStore,
        n_list: Vec<u64>,
        youngest_twig_id: u64,
    ) -> (Vec<u64>, [u8; 32]) {
        let max_level = calc_max_level(youngest_twig_id);

        if n_list.is_empty() {
            let root = *self.get_node(NodePos::pos(max_level as u64, 0)).unwrap();
            return (n_list, root);
        }

        // Phase 1: Populate GPU node store with all nodes needed for this sync.
        // Collect twig roots and existing nodes at levels below what we need.
        let mut populate_pairs: Vec<(u64, [u8; 32])> = Vec::new();

        // Pre-fill NULL_TWIG.twig_root at every twig position that might be
        // read at FIRST_LEVEL_ABOVE_TWIG (i.e. 2*i and 2*i+1 for each i in
        // n_list). The per-level path (sync_nodes_by_level_gpu, line 521-527)
        // uses `unwrap_or(NULL_TWIG.twig_root)` to fall back when a twig is
        // not in `active_twig_shards`. Without this pre-fill, the resident
        // path's `bulk_get_device` returns uninitialized memory for missing
        // twigs and produces a different parent hash. This is the
        // resident-vs-per-level parity divergence root cause.
        for &i in &n_list {
            let pos_l = (TWIG_ROOT_LEVEL as u64) << 56 | (2 * i);
            let pos_r = (TWIG_ROOT_LEVEL as u64) << 56 | (2 * i + 1);
            populate_pairs.push((pos_l, twig::NULL_TWIG.twig_root));
            populate_pairs.push((pos_r, twig::NULL_TWIG.twig_root));
        }

        // Add all active twig roots to the GPU store at TWIG_ROOT_LEVEL.
        // These overwrite the NULL_TWIG.twig_root pre-fill above for any
        // twig that actually exists.
        for twig_shard in &self.active_twig_shards {
            for (&key, twig) in twig_shard {
                let pos_val = (TWIG_ROOT_LEVEL as u64) << 56 | key;
                populate_pairs.push((pos_val, twig.twig_root));
            }
        }

        // Add all existing nodes from CPU node storage to GPU store
        for level_idx in 0..MAX_TREE_LEVEL {
            for (shard_id, shard) in self.nodes[level_idx].iter().enumerate() {
                for (pos, hash) in shard.iter_with_context(shard_id, level_idx) {
                    populate_pairs.push((pos.as_u64(), *hash));
                }
            }
        }

        // Batch upload all nodes to GPU store
        if !populate_pairs.is_empty() {
            if let Err(e) = gpu_store.insert_from_host(&populate_pairs) {
                eprintln!("[gpu-resident] Failed to populate store: {e}, falling back to per-level GPU");
                return self.sync_upper_nodes_gpu(gpu, n_list, youngest_twig_id);
            }
        }

        // Also set NULL_NODE sentinel values at max_n boundaries for each level
        {
            let mut boundary_pairs: Vec<(u64, [u8; 32])> = Vec::new();
            let mut current_n_list = n_list.clone();
            for level in FIRST_LEVEL_ABOVE_TWIG..=max_level {
                let max_n = max_n_at_level(youngest_twig_id, level);
                let pos0 = NodePos::pos(level as u64, max_n).as_u64();
                let pos1 = NodePos::pos(level as u64, max_n + 1).as_u64();
                boundary_pairs.push((pos0, NULL_NODE_IN_HIGHER_TREE[level as usize]));
                boundary_pairs.push((pos1, NULL_NODE_IN_HIGHER_TREE[level as usize]));

                // Build next level n_list for boundary calculation
                let mut new_list = Vec::with_capacity(current_n_list.len());
                for &i in &current_n_list {
                    if new_list.is_empty() || *new_list.last().unwrap() != i / 2 {
                        new_list.push(i / 2);
                    }
                }
                current_n_list = new_list;
            }
            if !boundary_pairs.is_empty() {
                let _ = gpu_store.insert_from_host(&boundary_pairs);
            }
        }

        // Phase 2: Run upper tree sync entirely on GPU
        let result = gpu_store.sync_upper_nodes_on_device(
            gpu,
            n_list.clone(),
            FIRST_LEVEL_ABOVE_TWIG,
            max_level,
        );

        match result {
            Ok((final_n_list, root_hash)) => {
                // Phase 3: Write back results to CPU HashMaps.
                // We need the CPU-side nodes up-to-date for edge node/prune operations.
                // Fetch the computed nodes back from GPU for each level.
                let mut current_list = n_list;
                for level in FIRST_LEVEL_ABOVE_TWIG..=max_level {
                    // Collect positions we computed at this level
                    let positions: Vec<u64> = current_list
                        .iter()
                        .map(|&i| NodePos::pos(level as u64, i).as_u64())
                        .collect();

                    if let Ok(results) = gpu_store.get_to_host(&positions) {
                        for (idx, &i) in current_list.iter().enumerate() {
                            if let Some(hash) = results[idx] {
                                let pos = NodePos::pos(level as u64, i);
                                self.set_node(pos, hash);
                            }
                        }
                    }

                    // Build next level
                    let mut new_list = Vec::with_capacity(current_list.len());
                    for &i in &current_list {
                        if new_list.is_empty() || *new_list.last().unwrap() != i / 2 {
                            new_list.push(i / 2);
                        }
                    }
                    current_list = new_list;
                }

                (final_n_list, root_hash)
            }
            Err(e) => {
                eprintln!(
                    "[gpu-resident] sync failed: {e}, falling back to per-level GPU"
                );
                self.sync_upper_nodes_gpu(gpu, n_list, youngest_twig_id)
            }
        }
    }

    /// GPU-accelerated evict_twigs.
    #[cfg(feature = "cuda")]
    pub fn evict_twigs_gpu(
        &mut self,
        gpu: &crate::gpu::GpuHasher,
        n_list: Vec<u64>,
        twig_evict_start: u64,
        twig_evict_end: u64,
    ) -> Vec<u64> {
        let new_list = self.sync_mt_for_active_bits_phase2_gpu(gpu, n_list);
        for twig_id in twig_evict_start..twig_evict_end {
            let pos = NodePos::pos(TWIG_ROOT_LEVEL as u64, twig_id);
            let twig_root = self.get_twig(twig_id).unwrap().twig_root;
            self.set_node_copy(pos, &twig_root);
            let (shard_idx, key) = get_shard_idx_and_key(twig_id);
            self.active_twig_shards[shard_idx].remove(&key);
        }
        new_list
    }

    /// GPU-accelerated active bits phase2 sync.
    /// Batches sync_l2, sync_l3, and sync_top across all touched twigs.
    #[cfg(feature = "cuda")]
    pub fn sync_mt_for_active_bits_phase2_gpu(
        &mut self,
        gpu: &crate::gpu::GpuHasher,
        n_list: Vec<u64>,
    ) -> Vec<u64> {
        // Build deduplicated twig list from L2-level n_list
        let mut twig_list: Vec<u64> = Vec::with_capacity(n_list.len());
        for &i in &n_list {
            let twig_id = i / 2;
            if twig_list.is_empty() || *twig_list.last().unwrap() != twig_id {
                twig_list.push(twig_id);
            }
        }

        if !twig_list.is_empty() {
            // Gather all 4 L1 values + left_root per twig for the fused kernel
            let n = twig_list.len();
            let mut l1_values: Vec<[u8; 32]> = Vec::with_capacity(n * 4);
            let mut left_roots: Vec<[u8; 32]> = Vec::with_capacity(n);

            for &twig_id in &twig_list {
                let (s, k) = get_shard_idx_and_key(twig_id);
                let twig = self.active_twig_shards[s].get(&k).unwrap();
                l1_values.push(twig.active_bits_mtl1[0]);
                l1_values.push(twig.active_bits_mtl1[1]);
                l1_values.push(twig.active_bits_mtl1[2]);
                l1_values.push(twig.active_bits_mtl1[3]);
                left_roots.push(twig.left_root);
            }

            // Single GPU dispatch: L2 + L3 + top fused
            let (twig_roots, l2_out, l3_out) =
                gpu.batch_active_bits_fused(&l1_values, &left_roots);

            // Write back all results
            for (idx, &twig_id) in twig_list.iter().enumerate() {
                let (s, k) = get_shard_idx_and_key(twig_id);
                let twig = self.active_twig_shards[s].get_mut(&k).unwrap();
                twig.active_bits_mtl2[0] = l2_out[idx * 2];
                twig.active_bits_mtl2[1] = l2_out[idx * 2 + 1];
                twig.active_bits_mtl3 = l3_out[idx];
                twig.twig_root = twig_roots[idx];
            }
        }

        // Return next-level n_list
        let mut new_list: Vec<u64> = Vec::with_capacity(twig_list.len());
        for &i in &twig_list {
            if new_list.is_empty() || *new_list.last().unwrap() != i / 2 {
                new_list.push(i / 2);
            }
        }
        new_list
    }
}

fn do_sync_job(
    upper_tree: &UpperTree,
    nodes: &mut NodeShard,
    level: i64,
    _shard_id: usize,
    n_list: &[u64],
) {
    let child_nodes = upper_tree.nodes.get((level - 1) as usize).unwrap();
    for &i in n_list {
        let pos = NodePos::pos(level as u64, i);
        if level == FIRST_LEVEL_ABOVE_TWIG {
            let left_option = upper_tree.get_twig_root(2 * i);
            let left = match left_option {
                Some(v) => v,
                None => panic!("Cannot find left twig root {}", 2 * i),
            };
            let right_option = upper_tree.get_twig_root(2 * i + 1);
            let mut right = [0u8; 32];
            match right_option {
                Some(v) => {
                    right.copy_from_slice(v);
                }
                None => {
                    right.copy_from_slice(&twig::NULL_TWIG.twig_root[..]);
                }
            };
            let mut hash = [0u8; 32];
            hasher::node_hash_inplace(level as u8 - 1, &mut hash, left, &right);
            nodes.insert(pos, hash);
        } else {
            let node_pos_l = NodePos::pos((level - 1) as u64, 2 * i);
            let node_pos_r = NodePos::pos((level - 1) as u64, 2 * i + 1);
            let sl = node_pos_l.nth() as usize % NODE_SHARD_COUNT;
            let sr = node_pos_r.nth() as usize % NODE_SHARD_COUNT;
            let node_l = match child_nodes[sl].get(&node_pos_l) {
                Some(v) => v,
                None => {
                    panic!(
                        "Cannot find left child {}-{} {}-{} {} {:?}",
                        level,
                        i,
                        level - 1,
                        2 * i,
                        2 * i + 1,
                        node_pos_l
                    );
                }
            };

            let node_r = match child_nodes[sr].get(&node_pos_r) {
                Some(v) => v,
                None => {
                    panic!(
                        "Cannot find right child {}-{} {}-{} {} {:?}",
                        level,
                        i,
                        level - 1,
                        2 * i,
                        2 * i + 1,
                        node_pos_r
                    )
                }
            };

            let mut hash = [0u8; 32];
            hasher::node_hash_inplace(level as u8 - 1, &mut hash, node_l, node_r);
            nodes.insert(pos, hash);
        }
    }
}

pub struct Tree {
    pub my_shard_id: usize,

    pub upper_tree: UpperTree,
    pub new_twig_map: HashMap<u64, Box<twig::Twig>>,

    pub entry_file_wr: EntryFileWriter,
    pub twig_file_wr: TwigFileWriter,
    pub dir_name: String,

    // these variables can be recovered from entry file
    pub youngest_twig_id: u64,
    // pub active_bit_shards: [HashMap<u64, [u8; 256]>; TWIG_SHARD_COUNT],
    pub active_bit_shards: Vec<HashMap<u64, ActiveBits>>,
    pub mtree_for_youngest_twig: Box<TwigMT>,

    // The following variables are only used during the execution of one block
    pub mtree_for_yt_change_start: i32,
    pub mtree_for_yt_change_end: i32,
    touched_pos_of_512b: HashSet<u64>,
}

impl Tree {
    pub fn new_blank(
        shard_id: usize,
        buffer_size: usize,
        segment_size: i64,
        dir_name: String,
        suffix: String,
        with_twig_file: bool,
        cipher: Option<Aes256Gcm>,
    ) -> Self {
        let dir_entry = format!("{}/{}{}", dir_name, ENTRIES_PATH, suffix);
        let _ = fs::create_dir_all(&dir_entry);
        let twig_file = if with_twig_file {
            let dir_twig = format!("{}/{}{}", dir_name, TWIG_PATH, suffix);
            let _ = fs::create_dir_all(&dir_twig);
            TwigFile::new(buffer_size, segment_size, dir_twig)
        } else {
            TwigFile::empty()
        };
        let twig_arc = Arc::new(twig_file);
        let directio = cfg!(feature = "directio");
        let ef = EntryFile::new(buffer_size, segment_size, dir_entry, directio, cipher);

        Self {
            my_shard_id: shard_id,
            upper_tree: UpperTree::new(shard_id),
            new_twig_map: HashMap::new(),
            entry_file_wr: EntryFileWriter::new(Arc::new(ef), buffer_size),
            twig_file_wr: TwigFileWriter::new(twig_arc, buffer_size),
            dir_name,
            youngest_twig_id: 0,
            active_bit_shards: vec![HashMap::new(); TWIG_SHARD_COUNT],
            mtree_for_youngest_twig: twig::NULL_MT_FOR_TWIG.clone(),
            mtree_for_yt_change_start: -1,
            mtree_for_yt_change_end: -1,
            touched_pos_of_512b: HashSet::new(),
        }
    }

    pub fn new(
        shard_id: usize,
        buffer_size: usize,
        segment_size: i64,
        dir_name: String,
        suffix: String,
        with_twig_file: bool,
        cipher: Option<Aes256Gcm>,
    ) -> Self {
        let mut tree = Self::new_blank(
            shard_id,
            buffer_size,
            segment_size,
            dir_name,
            suffix,
            with_twig_file,
            cipher,
        );

        tree.new_twig_map.insert(0, twig::NULL_TWIG.clone());
        tree.upper_tree
            .set_node(NodePos::pos(FIRST_LEVEL_ABOVE_TWIG as u64, 0), [0; 32]);
        tree.upper_tree.active_twig_shards[0].insert(0, twig::NULL_TWIG.clone());
        tree.active_bit_shards[0].insert(0, twig::NULL_ACTIVE_BITS.clone());

        tree
    }

    pub fn close(&mut self) {
        // Close files
        self.entry_file_wr.entry_file.close();
        self.twig_file_wr.twig_file.close();
    }

    pub fn get_file_sizes(&self) -> (i64, i64) {
        (
            self.entry_file_wr.entry_file.size(),
            self.twig_file_wr.twig_file.hp_file.size(),
        )
    }

    pub fn truncate_files(&self, entry_file_size: i64, twig_file_size: i64) {
        self.entry_file_wr
            .entry_file
            .truncate(entry_file_size)
            .unwrap();
        self.twig_file_wr.twig_file.truncate(twig_file_size);
    }

    pub fn get_active_bits(&self, twig_id: u64) -> &ActiveBits {
        let (shard_idx, key) = get_shard_idx_and_key(twig_id);
        match self.active_bit_shards[shard_idx].get(&key) {
            Some(v) => v,
            None => panic!("cannot find twig {}", twig_id),
        }
    }

    fn get_active_bits_mut(&mut self, twig_id: u64) -> &mut ActiveBits {
        let (shard_idx, key) = get_shard_idx_and_key(twig_id);
        self.active_bit_shards[shard_idx].get_mut(&key).unwrap()
    }

    pub fn get_active_bit(&self, sn: u64) -> bool {
        let twig_id = sn >> TWIG_SHIFT;
        let pos = sn as u32 & TWIG_MASK;
        self.get_active_bits(twig_id).get_bit(pos)
    }

    pub fn set_entry_activiation(&mut self, sn: u64, active: bool) {
        let twig_id = sn >> TWIG_SHIFT;
        let pos = sn as u32 & TWIG_MASK;
        let active_bits = self.get_active_bits_mut(twig_id);
        if active {
            active_bits.set_bit(pos);
        } else {
            active_bits.clear_bit(pos);
        }
        self.touch_pos(sn);
    }

    pub fn touch_pos(&mut self, sn: u64) {
        self.touched_pos_of_512b.insert(sn / 512);
    }

    pub fn clear_touched_pos(&mut self) {
        self.touched_pos_of_512b.clear();
    }

    pub fn active_entry(&mut self, sn: u64) {
        self.set_entry_activiation(sn, true);
    }

    pub fn deactive_entry(&mut self, sn: u64) {
        self.set_entry_activiation(sn, false);
    }

    pub fn append_entry(&mut self, entry_bz: &EntryBz) -> Result<i64, std::io::Error> {
        let sn = entry_bz.serial_number();
        self.active_entry(sn);

        let twig_id = sn >> TWIG_SHIFT;
        self.youngest_twig_id = twig_id;
        // record change_start/change_end for endblock sync
        let position = sn as u32 & TWIG_MASK;
        if self.mtree_for_yt_change_start == -1 {
            self.mtree_for_yt_change_start = position as i32;
        } else if self.mtree_for_yt_change_end + 1 != position as i32 {
            panic!("non-increasing position!");
        }
        self.mtree_for_yt_change_end = position as i32;

        let pos = self.entry_file_wr.append(entry_bz)?;
        self.mtree_for_youngest_twig[(LEAF_COUNT_IN_TWIG + position) as usize]
            .copy_from_slice(entry_bz.hash().as_slice());

        if position == TWIG_MASK {
            // when this is the last entry of current twig
            // write the merkle tree of youngest twig to twig_file
            self.sync_mt_for_youngest_twig(false);
            self.twig_file_wr.append_twig(
                &self.mtree_for_youngest_twig[..],
                pos + entry_bz.len() as i64,
            );
            // allocate new twig as youngest twig
            self.youngest_twig_id += 1;
            let (s, i) = get_shard_idx_and_key(self.youngest_twig_id);
            self.new_twig_map
                .insert(self.youngest_twig_id, twig::NULL_TWIG.clone());
            self.active_bit_shards[s].insert(i, twig::NULL_ACTIVE_BITS.clone());

            self.mtree_for_youngest_twig
                .copy_from_slice(&twig::NULL_MT_FOR_TWIG[..]);
            self.touch_pos(sn + 1)
        }
        Ok(pos)
    }

    pub fn prune_twigs(&mut self, start_id: u64, end_id: u64, entry_file_size: i64) {
        if end_id - start_id < MIN_PRUNE_COUNT {
            panic!(
                "The count of pruned twigs is too small: {}",
                end_id - start_id
            );
        }

        self.entry_file_wr
            .entry_file
            .prune_head(entry_file_size)
            .unwrap();
        self.twig_file_wr
            .twig_file
            .prune_head((end_id * twigfile::TWIG_SIZE) as i64);
    }

    pub fn flush_files(&mut self, twig_delete_start: u64, twig_delete_end: u64) -> Vec<u64> {
        let mut entry_file_tmp = self.entry_file_wr.temp_clone();
        let mut twig_file_tmp = self.twig_file_wr.temp_clone();
        mem::swap(&mut entry_file_tmp, &mut self.entry_file_wr);
        mem::swap(&mut twig_file_tmp, &mut self.twig_file_wr);
        let n_list = thread::scope(|s| {
            // run flushing in a threads such that sync_* won't be blocked
            s.spawn(|| {
                entry_file_tmp.flush().unwrap();
            });
            s.spawn(|| {
                twig_file_tmp.flush();
            });
            self.sync_mt_for_youngest_twig(false);
            let youngest_twig = self.new_twig_map.get(&self.youngest_twig_id).unwrap();
            let mut twig_map = HashMap::new();
            twig_map.insert(self.youngest_twig_id, youngest_twig.clone());
            mem::swap(&mut self.new_twig_map, &mut twig_map);
            //add new_twig_map's old content to upper_tree
            self.upper_tree.add_twigs(twig_map);
            //now, new_twig_map only contains one member: youngest_twig.clone()

            let n_list = self.sync_mt_for_active_bits_phase1();
            for twig_id in twig_delete_start..twig_delete_end {
                let (shard_idx, key) = get_shard_idx_and_key(twig_id);
                self.active_bit_shards[shard_idx].remove(&key);
            }
            self.touched_pos_of_512b.clear();
            n_list
        });
        mem::swap(&mut entry_file_tmp, &mut self.entry_file_wr);
        mem::swap(&mut twig_file_tmp, &mut self.twig_file_wr);
        n_list
    }

    pub fn sync_mt_for_active_bits_phase1(&mut self) -> Vec<u64> {
        let mut n_list = self
            .touched_pos_of_512b
            .iter()
            .cloned()
            .collect::<Vec<u64>>();
        n_list.sort();

        // Pre-partition by twig shard to avoid 75% wasted iteration
        let mut by_shard: [Vec<u64>; TWIG_SHARD_COUNT] = Default::default();
        for &i in &n_list {
            let twig_id = i >> 2;
            let (s, _) = get_shard_idx_and_key(twig_id);
            by_shard[s].push(i);
        }

        let mut new_list = Vec::with_capacity(n_list.len());
        rayon::scope(|s| {
            for (sid, twig_shard) in self.upper_tree.active_twig_shards.iter_mut().enumerate() {
                let active_bit_shards = &self.active_bit_shards;
                let shard_items = &by_shard[sid];
                s.spawn(move |_| {
                    for &i in shard_items {
                        let twig_id = i >> 2;
                        let (s, k) = get_shard_idx_and_key(twig_id);
                        let active_bits = active_bit_shards[s].get(&k).unwrap();
                        twig_shard
                            .get_mut(&k)
                            .unwrap()
                            .sync_l1((i & 3) as i32, active_bits);
                    }
                });
            }
            for i in &n_list {
                if new_list.is_empty() || *new_list.last().unwrap() != i / 2 {
                    new_list.push(i / 2);
                }
            }
            new_list
        })
    }

    pub fn sync_mt_for_youngest_twig(&mut self, recover_mode: bool) {
        if self.mtree_for_yt_change_start == -1 {
            return;
        }
        sync_mtree(
            &mut self.mtree_for_youngest_twig,
            self.mtree_for_yt_change_start,
            self.mtree_for_yt_change_end,
        );
        self.mtree_for_yt_change_start = -1;
        self.mtree_for_yt_change_end = 0;
        let youngest_twig;
        if recover_mode {
            youngest_twig = self.upper_tree.get_twig(self.youngest_twig_id).unwrap();
        } else {
            youngest_twig = self.new_twig_map.get_mut(&self.youngest_twig_id).unwrap();
        }
        youngest_twig
            .left_root
            .copy_from_slice(&self.mtree_for_youngest_twig[1]);
    }

    pub fn load_mt_for_non_youngest_twig(&mut self, twig_id: u64) {
        if self.mtree_for_yt_change_start == -1 {
            return;
        }
        self.mtree_for_yt_change_start = -1;
        self.mtree_for_yt_change_end = 0;
        let active_twig = self.upper_tree.get_twig(self.youngest_twig_id).unwrap();
        self.twig_file_wr
            .twig_file
            .get_hash_root(twig_id, &mut active_twig.left_root);
    }

    fn get_upper_path_and_root(&self, twig_id: u64) -> (Vec<proof::ProofNode>, [u8; 32]) {
        let max_level = calc_max_level(self.youngest_twig_id);

        let mut peer_hash = [0u8; 32];
        // use '^ 1' to flip the lowest bit to get sibling
        if let Some(v) = self.upper_tree.get_twig_root(twig_id ^ 1) {
            peer_hash.copy_from_slice(v);
        } else {
            peer_hash.copy_from_slice(&twig::NULL_TWIG.twig_root[..]);
        }

        let mut self_hash = [0u8; 32];
        if let Some(v) = self.upper_tree.get_twig_root(twig_id) {
            self_hash.copy_from_slice(v);
        } else {
            return (Vec::new(), [0; 32]);
        }

        let mut upper_path = Vec::with_capacity((max_level - FIRST_LEVEL_ABOVE_TWIG + 1) as usize);
        upper_path.push(proof::ProofNode {
            self_hash,
            peer_hash,
            peer_at_left: (twig_id & 1) != 0, //twig_id's lowest bit == 1 so the peer is at left
        });

        let mut n = twig_id >> 1;
        for level in FIRST_LEVEL_ABOVE_TWIG..max_level {
            let peer_at_left = (n & 1) != 0;

            let snode = match self.upper_tree.get_node(NodePos::pos(level as u64, n)) {
                Some(v) => *v,
                None => panic!("Cannot find node"),
            };
            let pnode = match self.upper_tree.get_node(NodePos::pos(level as u64, n ^ 1)) {
                Some(v) => *v,
                None => panic!("Cannot find node"),
            };
            upper_path.push(proof::ProofNode {
                self_hash: snode,
                peer_hash: pnode,
                peer_at_left,
            });
            n >>= 1;
        }

        let root_option = self.upper_tree.get_node(NodePos::pos(max_level as u64, 0));
        let root = match root_option {
            Some(v) => *v,
            None => panic!("cannot find node {}-{}", max_level, 0),
        };

        (upper_path, root)
    }

    pub fn get_proof(&self, sn: u64) -> Result<proof::ProofPath, String> {
        let twig_id = sn >> TWIG_SHIFT;
        let mut path = proof::ProofPath::new();
        path.serial_num = sn;

        if twig_id > self.youngest_twig_id {
            return Err("twig_id > self.youngest_twig_id".to_string());
        }

        (path.upper_path, path.root) = self.get_upper_path_and_root(twig_id);
        if path.upper_path.is_empty() {
            return Err("Cannot find upper path".to_string());
        }

        if twig_id == self.youngest_twig_id {
            path.left_of_twig = proof::get_left_path_in_mem(&self.mtree_for_youngest_twig, sn);
        } else {
            let twig_file = &self.twig_file_wr.twig_file;
            if twig_file.is_empty() {
                return Err("twig_file is empty".to_string());
            }
            path.left_of_twig = proof::get_left_path_on_disk(twig_file, twig_id, sn);
        }
        let (s, k) = get_shard_idx_and_key(twig_id);
        let twig = self.upper_tree.active_twig_shards[s]
            .get(&k)
            .unwrap_or(&twig::NULL_TWIG);
        let active_bits = self.active_bit_shards[s]
            .get(&k)
            .unwrap_or(&twig::NULL_ACTIVE_BITS);
        path.right_of_twig = proof::get_right_path(twig, active_bits, sn);

        Ok(path)
    }

    pub fn get_hashes_by_pos_list(&self, pos_list: &Vec<(u8, u64)>) -> Vec<[u8; 32]> {
        let mut hashes = Vec::with_capacity(pos_list.len());
        for (_, hash) in self.hash_iter(pos_list) {
            hashes.push(hash);
        }
        hashes
    }

    pub fn hash_iter<'a>(&'a self, pos_list: &'a Vec<(u8, u64)>) -> HashIterForPosList<'a> {
        HashIterForPosList {
            cache: HashMap::with_capacity(20),
            tree: self,
            pos_list,
            idx: 0,
        }
    }

    fn get_hash_by_node(
        &self,
        level: u8,
        nth: u64,
        cache: &mut HashMap<i64, [u8; 32]>,
    ) -> [u8; 32] {
        let mut twig_id: u64 = 0;
        let mut level_stride: u64 = 0;
        if level <= 12 {
            level_stride = 4096 >> level;
            twig_id = nth / level_stride;
        }

        // left tree of twig
        if level <= 11 && (nth % level_stride) < level_stride / 2 {
            let is_youngest_twig_id = twig_id == self.youngest_twig_id;
            let self_id: u64 = nth % level_stride;
            let idx = level_stride / 2 + self_id;
            if is_youngest_twig_id {
                return self.mtree_for_youngest_twig[idx as usize];
            } else {
                let mut hash = [0u8; 32];
                self.twig_file_wr
                    .twig_file
                    .get_hash_node(twig_id, idx as i64, cache, &mut hash);
                return hash;
            }
        }

        // right tree of twig
        if (8..=11).contains(&level) {
            let (s, k) = get_shard_idx_and_key(twig_id);
            let active_bits = self.active_bit_shards[s]
                .get(&k)
                .unwrap_or(&twig::NULL_ACTIVE_BITS);
            let self_id: u64 = (nth % level_stride) - level_stride / 2;
            if level == 8 {
                let hash = active_bits.get_bits(self_id as usize, 32);
                return hash.try_into().unwrap();
            }
            let twig = self.upper_tree.active_twig_shards[s]
                .get(&k)
                .unwrap_or(&twig::NULL_TWIG);
            if level == 9 {
                return twig.active_bits_mtl1[self_id as usize];
            }
            if level == 10 {
                return twig.active_bits_mtl2[self_id as usize];
            }
            if level == 11 {
                return twig.active_bits_mtl3;
            }
        }

        // upper tree
        if level == 12 {
            return *self
                .upper_tree
                .get_twig_root(twig_id)
                .unwrap_or(&twig::NULL_TWIG.twig_root);
        }
        *self
            .upper_tree
            .get_node(NodePos::pos(level as u64, nth))
            .unwrap_or(&NULL_NODE_IN_HIGHER_TREE[level as usize])
    }

    /// GPU-accelerated flush_files. Same as flush_files but uses GPU batch hashing.
    #[cfg(feature = "cuda")]
    pub fn flush_files_gpu(
        &mut self,
        gpu: &crate::gpu::GpuHasher,
        twig_delete_start: u64,
        twig_delete_end: u64,
    ) -> Vec<u64> {
        let mut entry_file_tmp = self.entry_file_wr.temp_clone();
        let mut twig_file_tmp = self.twig_file_wr.temp_clone();
        mem::swap(&mut entry_file_tmp, &mut self.entry_file_wr);
        mem::swap(&mut twig_file_tmp, &mut self.twig_file_wr);
        let n_list = thread::scope(|s| {
            s.spawn(|| {
                entry_file_tmp.flush().unwrap();
            });
            s.spawn(|| {
                twig_file_tmp.flush();
            });
            self.sync_mt_for_youngest_twig_gpu(gpu);
            let youngest_twig = self.new_twig_map.get(&self.youngest_twig_id).unwrap();
            let mut twig_map = HashMap::new();
            twig_map.insert(self.youngest_twig_id, youngest_twig.clone());
            mem::swap(&mut self.new_twig_map, &mut twig_map);
            self.upper_tree.add_twigs(twig_map);

            let n_list = self.sync_mt_for_active_bits_phase1_gpu(gpu);
            for twig_id in twig_delete_start..twig_delete_end {
                let (shard_idx, key) = get_shard_idx_and_key(twig_id);
                self.active_bit_shards[shard_idx].remove(&key);
            }
            self.touched_pos_of_512b.clear();
            n_list
        });
        mem::swap(&mut entry_file_tmp, &mut self.entry_file_wr);
        mem::swap(&mut twig_file_tmp, &mut self.twig_file_wr);
        n_list
    }

    /// GPU-accelerated sync for youngest twig Merkle tree.
    #[cfg(feature = "cuda")]
    pub fn sync_mt_for_youngest_twig_gpu(&mut self, gpu: &crate::gpu::GpuHasher) {
        if self.mtree_for_yt_change_start == -1 {
            return;
        }
        let start = self.mtree_for_yt_change_start;
        let end = self.mtree_for_yt_change_end;
        twig::sync_mtrees_gpu(
            gpu,
            &mut [(&mut self.mtree_for_youngest_twig, start, end)],
        );
        self.mtree_for_yt_change_start = -1;
        self.mtree_for_yt_change_end = 0;
        let youngest_twig = self.new_twig_map.get_mut(&self.youngest_twig_id).unwrap();
        youngest_twig
            .left_root
            .copy_from_slice(&self.mtree_for_youngest_twig[1]);
    }

    /// GPU-accelerated active bits phase1 sync.
    /// Batches all sync_l1 operations into a single GPU dispatch.
    #[cfg(feature = "cuda")]
    pub fn sync_mt_for_active_bits_phase1_gpu(
        &mut self,
        gpu: &crate::gpu::GpuHasher,
    ) -> Vec<u64> {
        use crate::gpu::NodeHashJob;

        let mut n_list = self
            .touched_pos_of_512b
            .iter()
            .cloned()
            .collect::<Vec<u64>>();
        n_list.sort();

        // Collect all sync_l1 jobs
        let mut jobs = Vec::with_capacity(n_list.len());
        let mut targets: Vec<(u64, i32)> = Vec::with_capacity(n_list.len()); // (twig_id, pos)

        for &i in &n_list {
            let twig_id = i >> 2;
            let pos = (i & 3) as i32;
            let (s, k) = get_shard_idx_and_key(twig_id);
            let active_bits = self.active_bit_shards[s].get(&k).unwrap();
            let start = pos as usize * 512;
            // sync_l1 hashes active_bits pages into mtl1
            let left_bits = active_bits.get_bits(start / 256, 32);
            let right_bits = active_bits.get_bits(start / 256 + 1, 32);
            let mut left = [0u8; 32];
            let mut right = [0u8; 32];
            left.copy_from_slice(left_bits);
            right.copy_from_slice(right_bits);
            jobs.push(NodeHashJob {
                level: 8,
                left,
                right,
            });
            targets.push((twig_id, pos));
        }

        if !jobs.is_empty() {
            let results = gpu.auto_batch_node_hash(&jobs);
            for (idx, (twig_id, pos)) in targets.iter().enumerate() {
                let (s, k) = get_shard_idx_and_key(*twig_id);
                let twig = self.upper_tree.active_twig_shards[s]
                    .get_mut(&k)
                    .unwrap();
                twig.active_bits_mtl1[*pos as usize] = results[idx];
            }
        }

        let mut new_list = Vec::with_capacity(n_list.len());
        for &i in &n_list {
            if new_list.is_empty() || *new_list.last().unwrap() != i / 2 {
                new_list.push(i / 2);
            }
        }
        new_list
    }
}

pub struct HashIterForPosList<'a> {
    cache: HashMap<i64, [u8; 32]>,
    tree: &'a Tree,
    pos_list: &'a Vec<(u8, u64)>,
    idx: usize,
}

impl Iterator for HashIterForPosList<'_> {
    type Item = (usize, [u8; 32]);

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.pos_list.len() {
            return None;
        }
        let (level, nth) = self.pos_list[self.idx];
        let hash = self.tree.get_hash_by_node(level, nth, &mut self.cache);
        let idx = self.idx;
        self.idx += 1;
        Some((idx, hash))
    }
}

pub fn max_n_at_level(youngest_twig_id: u64, level: i64) -> u64 {
    if level < FIRST_LEVEL_ABOVE_TWIG {
        panic!("level is too small");
    }
    let shift = level - FIRST_LEVEL_ABOVE_TWIG + 1;
    youngest_twig_id >> shift
}

pub fn get_shard_idx_and_key(twig_id: u64) -> (usize, u64) {
    let idx = twig_id as usize % TWIG_SHARD_COUNT;
    let key = twig_id / TWIG_SHARD_COUNT as u64;
    (idx, key)
}

// debug

impl Tree {
    pub fn print(&self) {
        let mut offset: i64 = 0;
        let mut buf = vec![0u8; 2048];
        for twig_id in 0..self.youngest_twig_id {
            for _sn in twig_id * 2048..(twig_id + 1) * 2048 {
                let n = self.entry_file_wr.entry_file.read_entry(offset, &mut buf);
                if n > buf.len() {
                    buf.resize(n, 0);
                    self.entry_file_wr.entry_file.read_entry(offset, &mut buf);
                }
                offset += n as i64;

                let entry_bz = EntryBz { bz: &buf[0..n] };
                let entry = Entry::from_bz(&entry_bz);

                println!(
                    "[entry] twig: {}, sn: {}, k: {}, v: {}",
                    twig_id,
                    entry.serial_number,
                    hex::encode(entry.key),
                    hex::encode(entry.value)
                );
            }
        }

        let mut cache: HashMap<i64, Hash32> = HashMap::new();
        for twig_id in 0..self.youngest_twig_id {
            for _sn in twig_id * 2048..(twig_id + 1) * 2048 {
                let _h = self.get_hash_by_node(0, _sn, &mut cache);
                println!(
                    "[hash] level: {}, nth: {}, hash: {}",
                    0,
                    _sn,
                    hex::encode(_h)
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========== ME-4: NodeShard Dense/Sparse Tests ==========

    #[test]
    fn test_node_shard_dense_insert_get_remove() {
        let mut shard = NodeShard::new_dense();
        let hash_a = [0xAA; 32];
        let hash_b = [0xBB; 32];

        // nth values must be multiples of NODE_SHARD_COUNT for shard 0
        let pos_a = NodePos::pos(5, 0);
        let pos_b = NodePos::pos(5, NODE_SHARD_COUNT as u64 * 3);

        shard.insert(pos_a, hash_a);
        shard.insert(pos_b, hash_b);

        assert_eq!(shard.get(&pos_a), Some(&hash_a));
        assert_eq!(shard.get(&pos_b), Some(&hash_b));

        // Remove pos_a
        shard.remove(&pos_a);
        assert_eq!(shard.get(&pos_a), None);
        // pos_b still present
        assert_eq!(shard.get(&pos_b), Some(&hash_b));

        // Remove pos_b
        shard.remove(&pos_b);
        assert_eq!(shard.get(&pos_b), None);
    }

    #[test]
    fn test_node_shard_sparse_insert_get_remove() {
        let mut shard = NodeShard::new_sparse();
        let hash_a = [0x11; 32];
        let hash_b = [0x22; 32];
        let hash_c = [0x33; 32];

        let pos_a = NodePos::pos(30, 100);
        let pos_b = NodePos::pos(30, 200);
        let pos_c = NodePos::pos(31, 50);

        shard.insert(pos_a, hash_a);
        shard.insert(pos_b, hash_b);
        shard.insert(pos_c, hash_c);

        assert_eq!(shard.get(&pos_a), Some(&hash_a));
        assert_eq!(shard.get(&pos_b), Some(&hash_b));
        assert_eq!(shard.get(&pos_c), Some(&hash_c));

        shard.remove(&pos_b);
        assert_eq!(shard.get(&pos_b), None);
        assert_eq!(shard.get(&pos_a), Some(&hash_a));
        assert_eq!(shard.get(&pos_c), Some(&hash_c));
    }

    #[test]
    fn test_node_shard_dense_sparse_equivalence() {
        let mut dense = NodeShard::new_dense();
        let mut sparse = NodeShard::new_sparse();

        let test_data: Vec<(NodePos, [u8; 32])> = (0..20)
            .map(|i| {
                let nth = i as u64 * NODE_SHARD_COUNT as u64;
                let pos = NodePos::pos(5, nth);
                let mut hash = [0u8; 32];
                for k in 0..32 {
                    hash[k] = ((i * 7 + k) & 0xFF) as u8;
                }
                (pos, hash)
            })
            .collect();

        for (pos, hash) in &test_data {
            dense.insert(*pos, *hash);
            sparse.insert(*pos, *hash);
        }

        for (pos, _) in &test_data {
            let d = dense.get(pos);
            let s = sparse.get(pos);
            assert_eq!(
                d, s,
                "dense vs sparse mismatch at level={} nth={}",
                pos.level(),
                pos.nth()
            );
        }

        // Non-existent position returns None in both
        let missing = NodePos::pos(5, 99999 * NODE_SHARD_COUNT as u64);
        assert_eq!(dense.get(&missing), None);
        assert_eq!(sparse.get(&missing), None);
    }

    #[test]
    fn test_node_shard_iter_with_context() {
        let mut shard = NodeShard::new_dense();
        let shard_id = 2;
        let level = 7;

        let mut inserted: HashMap<u64, [u8; 32]> = HashMap::new();
        for i in 0..10 {
            let nth = i * NODE_SHARD_COUNT + shard_id;
            let pos = NodePos::pos(level as u64, nth as u64);
            let mut hash = [0u8; 32];
            hash[0] = i as u8;
            shard.insert(pos, hash);
            inserted.insert(nth as u64, hash);
        }

        let mut recovered: HashMap<u64, [u8; 32]> = HashMap::new();
        for (pos, hash) in shard.iter_with_context(shard_id, level) {
            assert_eq!(pos.level(), level as u64);
            recovered.insert(pos.nth(), *hash);
        }

        assert_eq!(
            inserted.len(),
            recovered.len(),
            "iter count mismatch: {} vs {}",
            inserted.len(),
            recovered.len()
        );
        for (nth, expected_hash) in &inserted {
            let got = recovered.get(nth);
            assert_eq!(
                got,
                Some(expected_hash),
                "missing or wrong hash at nth={}", nth
            );
        }
    }

    #[test]
    fn test_node_shard_dense_auto_resize() {
        let mut shard = NodeShard::new_dense();

        let pos_0 = NodePos::pos(5, 0);
        let pos_far = NodePos::pos(5, 1000 * NODE_SHARD_COUNT as u64);

        let hash_0 = [0x01; 32];
        let hash_far = [0x02; 32];

        shard.insert(pos_0, hash_0);
        assert_eq!(shard.get(&pos_0), Some(&hash_0));

        // Inserting at a much higher index auto-resizes
        shard.insert(pos_far, hash_far);
        assert_eq!(shard.get(&pos_far), Some(&hash_far));

        // Original still accessible
        assert_eq!(shard.get(&pos_0), Some(&hash_0));

        // In-between indices are None
        let pos_mid = NodePos::pos(5, 500 * NODE_SHARD_COUNT as u64);
        assert_eq!(shard.get(&pos_mid), None);
    }

    #[test]
    fn test_node_shard_remove_nonexistent() {
        let mut dense = NodeShard::new_dense();
        let mut sparse = NodeShard::new_sparse();

        let pos = NodePos::pos(5, 0);

        // Removing from empty shards should not panic
        dense.remove(&pos);
        sparse.remove(&pos);

        assert_eq!(dense.get(&pos), None);
        assert_eq!(sparse.get(&pos), None);

        // Insert then remove a different position
        dense.insert(NodePos::pos(5, 0), [0xAA; 32]);
        dense.remove(&NodePos::pos(5, NODE_SHARD_COUNT as u64 * 99));
        assert_eq!(dense.get(&NodePos::pos(5, 0)), Some(&[0xAA; 32]));
    }
}
