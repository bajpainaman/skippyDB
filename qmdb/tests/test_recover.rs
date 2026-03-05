use std::collections::HashMap;

use kyumdb::def::{DEFAULT_FILE_SIZE, SMALL_BUFFER_SIZE};
use kyumdb::test_helper::TempDir;
use kyumdb::{
    def::{MAX_UPPER_LEVEL, NODE_SHARD_COUNT, TWIG_MASK, TWIG_SHARD_COUNT},
    merkletree::{
        helpers::build_test_tree,
        recover,
        tree::{NodePos, NodeShard},
        twig::{ActiveBits, Twig},
    },
};

fn compare_nodes(
    nodes_a: &Vec<Vec<NodeShard>>,
    nodes_b: &Vec<Vec<NodeShard>>,
) {
    for level in 0..MAX_UPPER_LEVEL {
        for shard_id in 0..NODE_SHARD_COUNT {
            for (pos, val_a) in nodes_a[level][shard_id].iter_with_context(shard_id, level) {
                let val_b = nodes_b[level][shard_id]
                    .get(&pos)
                    .expect("node not found in nodes_b");
                assert_eq!(val_a, val_b);
            }
        }
    }
}

fn compare_twigs(
    twig_map_a: &Vec<HashMap<u64, Box<Twig>>>,
    twig_map_b: &Vec<HashMap<u64, Box<Twig>>>,
    active_bits_a: &Vec<HashMap<u64, ActiveBits>>,
    active_bits_b: &Vec<HashMap<u64, ActiveBits>>,
) {
    for shard_id in 0..TWIG_SHARD_COUNT {
        assert_eq!(twig_map_a[shard_id].len(), twig_map_b[shard_id].len());
        for (twig_id, twig_a) in &twig_map_a[shard_id] {
            let twig_b = twig_map_b[shard_id].get(twig_id).unwrap();
            assert_eq!(
                active_bits_a[shard_id][twig_id],
                active_bits_b[shard_id][twig_id]
            );
            assert_eq!(twig_a.active_bits_mtl1, twig_b.active_bits_mtl1);
            assert_eq!(twig_a.active_bits_mtl2, twig_b.active_bits_mtl2);
            assert_eq!(twig_a.active_bits_mtl3, twig_b.active_bits_mtl3);
            assert_eq!(twig_a.left_root, twig_b.left_root);
            assert_eq!(twig_a.twig_root, twig_b.twig_root);
        }
    }
}

#[test]
fn test_load_tree() {
    let dir_name = "./DataTree-loadtree";
    let _tmp_dir = TempDir::new(dir_name);

    let deact_sn_list = vec![101, 999, 1002];
    let (mut tree0, _, _, _) =
        build_test_tree(dir_name, &deact_sn_list, TWIG_MASK as i32 * 4, 1600);

    let mut n_list = tree0.flush_files(0, 0);
    n_list = tree0.upper_tree.evict_twigs(n_list, 0, 0);
    let (_, hash1) = tree0
        .upper_tree
        .sync_upper_nodes(n_list, tree0.youngest_twig_id);
    let nodes0 = tree0.upper_tree.nodes.clone();
    let active_twigs0 = tree0.upper_tree.active_twig_shards.clone();
    let active_bits0 = tree0.active_bit_shards.clone();
    let mtree4_youngest_twig0 = tree0.mtree_for_youngest_twig.clone();
    tree0.close();

    let (mut tree1, hash2) = recover::recover_tree(
        0,
        SMALL_BUFFER_SIZE as usize,
        DEFAULT_FILE_SIZE as usize,
        true,
        dir_name.to_string(),
        "".to_string(),
        &Vec::new(),
        0,
        0,
        0,
        1,
        &Vec::new(),
        None,
    );
    assert_eq!(hash1, hash2);
    println!("Recover finished");
    assert_eq!(4, tree1.youngest_twig_id);
    assert_eq!(tree1.mtree_for_youngest_twig, mtree4_youngest_twig0);
    compare_twigs(
        &tree1.upper_tree.active_twig_shards,
        &active_twigs0,
        &tree1.active_bit_shards,
        &active_bits0,
    );
    compare_nodes(&tree1.upper_tree.nodes, &nodes0);
    tree1.close();
}
