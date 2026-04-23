use super::task::{Task, TaskHub};
use super::tasksmanager::TasksManager;
use crate::def::{IN_BLOCK_IDX_BITS, IN_BLOCK_IDX_MASK};
use crate::entryfile::EntryCache;
use crate::utils::changeset::ChangeSet;
use atomptr::{AtomPtr, Ref};
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;

/// Number of block commits that can be in-flight through the flusher pipeline
/// at once. Phase 1.2 raised this from 2 to 4.
///
/// Each additional slot costs: one `Arc<TasksManager<T>>`, one
/// `Arc<EntryCache>`, and one `AtomicI64` height. At steady state it lets the
/// updater/flusher overlap more of the block-commit tail (async MetaDB write,
/// background commit thread, next block's prefetch) before blocking on
/// `end_block_chan.recv()`.
pub const BLOCK_PIPELINE_DEPTH: usize = 4;

/// One in-flight block's state: its task list, the height it's on (or `-1` if
/// the slot is idle), and the entry cache it filled during updater work.
struct Slot<T: Task> {
    tasks: AtomPtr<Arc<TasksManager<T>>>,
    height: AtomicI64,
    cache: AtomPtr<Arc<EntryCache>>,
}

impl<T: Task> Slot<T> {
    fn new() -> Self {
        Self {
            tasks: AtomPtr::new(Arc::new(TasksManager::<T>::default())),
            height: AtomicI64::new(-1),
            cache: AtomPtr::new(Arc::new(EntryCache::new_uninit())),
        }
    }
}

/// Ring of `BLOCK_PIPELINE_DEPTH` block-in-flight slots. The struct name is
/// kept for API stability through Phase 1; it will be renamed to
/// `BlockRingTaskHub<N>` in Phase 3.4 when the depth is made configurable.
pub struct BlockPairTaskHub<T: Task> {
    slots: [Slot<T>; BLOCK_PIPELINE_DEPTH],
}

impl<T: Task> Default for BlockPairTaskHub<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Task> BlockPairTaskHub<T> {
    pub fn new() -> Self {
        Self {
            slots: [Slot::new(), Slot::new(), Slot::new(), Slot::new()],
        }
    }

    pub fn free_slot_count(&self) -> usize {
        self.slots
            .iter()
            .filter(|s| s.height.load(Ordering::SeqCst) < 0)
            .count()
    }

    pub fn end_block(&self, height: i64) {
        for slot in &self.slots {
            if slot.height.load(Ordering::SeqCst) == height {
                slot.height.store(-1, Ordering::SeqCst);
                return;
            }
        }
        panic!("no data found for height");
    }

    pub fn start_block(
        &self,
        height: i64,
        tasks_in_blk: Arc<TasksManager<T>>,
        cache: Arc<EntryCache>,
    ) {
        for slot in &self.slots {
            if slot.height.load(Ordering::SeqCst) < 0 {
                let old = slot.tasks.swap(tasks_in_blk);
                drop(old);
                let old = slot.cache.swap(cache);
                drop(old);
                slot.height.store(height, Ordering::SeqCst);
                return;
            }
        }
        panic!("no data found for height");
    }
}

impl<T: Task> TaskHub for BlockPairTaskHub<T> {
    // updater in ads check this to know if a block is end.
    fn check_begin_end(&self, task_id: i64) -> (Option<Arc<EntryCache>>, bool) {
        let target_height = task_id >> IN_BLOCK_IDX_BITS;
        for slot in &self.slots {
            if slot.height.load(Ordering::SeqCst) == target_height {
                let last_task_id = slot.tasks.get_ref().as_ref().get_last_task_id();
                if (task_id & IN_BLOCK_IDX_MASK) != 0 {
                    return (None, last_task_id == task_id); // not first task in block
                }
                // first task in block → return cache
                let arc: Ref<Arc<EntryCache>> = slot.cache.get_ref();
                let cache: Arc<EntryCache> = Arc::clone(&arc);
                return (Some(cache), last_task_id == task_id);
            }
        }
        panic!("no data found for height");
    }

    fn get_change_sets(&self, task_id: i64) -> Arc<Vec<ChangeSet>> {
        let target_height = task_id >> IN_BLOCK_IDX_BITS;
        for slot in &self.slots {
            if slot.height.load(Ordering::SeqCst) == target_height {
                let idx = (task_id & IN_BLOCK_IDX_MASK) as usize;
                return slot.tasks.get_ref().as_ref().get_tasks_change_sets(idx);
            }
        }
        panic!("no data found for height");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entryfile::EntryCache;
    use crate::test_helper::SimpleTask;
    use crate::utils::changeset::ChangeSet;
    use parking_lot::RwLock;
    use std::sync::Arc;

    #[test]
    fn test_initialize() {
        let hub: BlockPairTaskHub<SimpleTask> = BlockPairTaskHub::new();
        assert_eq!(hub.free_slot_count(), BLOCK_PIPELINE_DEPTH);
    }

    #[test]
    fn test_start_end_block() {
        let hub: BlockPairTaskHub<SimpleTask> = BlockPairTaskHub::new();
        let tasks_in_blk = Arc::new(TasksManager::default());
        let cache = Arc::new(EntryCache::new_uninit());

        hub.start_block(1, tasks_in_blk.clone(), cache.clone());
        assert_eq!(hub.free_slot_count(), BLOCK_PIPELINE_DEPTH - 1);
        // first free slot takes the new block, rest are still idle
        assert_eq!(hub.slots[0].height.load(Ordering::SeqCst), 1);
        for slot in &hub.slots[1..] {
            assert_eq!(slot.height.load(Ordering::SeqCst), -1);
        }

        hub.start_block(2, tasks_in_blk.clone(), cache.clone());
        hub.start_block(3, tasks_in_blk.clone(), cache.clone());
        hub.start_block(4, tasks_in_blk.clone(), cache.clone());
        assert_eq!(hub.free_slot_count(), 0);

        hub.end_block(1);
        assert_eq!(hub.free_slot_count(), 1);

        assert!(std::panic::catch_unwind(move || {
            hub.end_block(1);
        })
        .is_err());
    }

    #[test]
    fn test_fills_beyond_depth_panics() {
        use std::panic::AssertUnwindSafe;
        let hub: BlockPairTaskHub<SimpleTask> = BlockPairTaskHub::new();
        let tasks_in_blk = Arc::new(TasksManager::default());
        let cache = Arc::new(EntryCache::new_uninit());

        for h in 1..=BLOCK_PIPELINE_DEPTH as i64 {
            hub.start_block(h, tasks_in_blk.clone(), cache.clone());
        }
        assert_eq!(hub.free_slot_count(), 0);

        // One more start_block when all slots are full must panic.
        // AssertUnwindSafe: the captured Arcs wrap interior-mutability types
        // (UnsafeCell inside TasksManager / EntryCache); we don't mutate them
        // across the panic boundary, so asserting unwind-safety is sound.
        assert!(std::panic::catch_unwind(AssertUnwindSafe(move || {
            hub.start_block(
                BLOCK_PIPELINE_DEPTH as i64 + 1,
                tasks_in_blk,
                cache,
            );
        }))
        .is_err());
    }

    #[test]
    fn test_check_begin_end() {
        let hub: BlockPairTaskHub<SimpleTask> = BlockPairTaskHub::new();
        let changeset = ChangeSet::new();
        let tasks_in_blk = vec![RwLock::new(Some(SimpleTask::new(vec![changeset])))];
        let last_task_id_in_blk = 1 << IN_BLOCK_IDX_BITS;
        let tasks_manager = Arc::new(TasksManager::new(tasks_in_blk, last_task_id_in_blk));
        let cache = Arc::new(EntryCache::new_uninit());

        hub.start_block(1, tasks_manager, cache.clone());

        let (cache_opt, is_end) = hub.check_begin_end((1 << IN_BLOCK_IDX_BITS) + 1);
        assert!(cache_opt.is_none());
        assert!(!is_end);

        let (cache_opt, is_end) = hub.check_begin_end(1 << IN_BLOCK_IDX_BITS);
        assert!(cache_opt.is_some());
        assert!(is_end);

        hub.end_block(1);

        assert!(std::panic::catch_unwind(move || {
            hub.check_begin_end(0);
        })
        .is_err());
    }

    #[test]
    fn test_get_change_sets() {
        let hub: BlockPairTaskHub<SimpleTask> = BlockPairTaskHub::new();
        let changeset = ChangeSet::new();
        let tasks_in_blk = vec![RwLock::new(Some(SimpleTask::new(vec![changeset])))];
        let tasks_manager = Arc::new(TasksManager::new(tasks_in_blk, 0));
        let cache = Arc::new(EntryCache::new_uninit());

        hub.start_block(1, tasks_manager, cache.clone());

        let change_sets = hub.get_change_sets(1 << IN_BLOCK_IDX_BITS);
        assert_eq!(change_sets.len(), 1);

        hub.end_block(1);

        assert!(std::panic::catch_unwind(move || {
            hub.get_change_sets(0);
        })
        .is_err());
    }
}
