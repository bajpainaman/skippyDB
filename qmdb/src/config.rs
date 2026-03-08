pub const COMPACT_THRES: i64 = 20000000;
pub const UTILIZATION_RATIO: i64 = 7;
pub const UTILIZATION_DIV: i64 = 10;

const TASK_CHAN_SIZE: usize = 200000;
const PREFETCHER_THREAD_COUNT: usize = 512;
const URING_SIZE: u32 = 1024;
const URING_COUNT: usize = 32;
const SUB_ID_CHAN_SIZE: usize = 20000;

/// Database configuration controlling storage layout, compaction, I/O, and encryption.
pub struct Config {
    /// Root directory for all database files.
    pub dir: String,
    /// Size of the in-memory write buffer in bytes.
    pub wrbuf_size: usize,
    /// Maximum size of each file segment on disk in bytes.
    pub file_segment_size: usize,
    /// Whether to persist twig data to a separate file.
    pub with_twig_file: bool,
    /// Optional 96-byte AES key material for at-rest encryption.
    pub aes_keys: Option<[u8; 96]>,
    /// Minimum entry count before compaction is triggered.
    pub compact_thres: i64,
    /// Numerator of the utilization ratio that governs compaction eligibility.
    pub utilization_ratio: i64,
    /// Denominator of the utilization ratio that governs compaction eligibility.
    pub utilization_div: i64,
    /// Capacity of the task channel between producers and the updater.
    pub task_chan_size: usize,
    /// Number of prefetcher threads for parallel I/O.
    pub prefetcher_thread_count: usize,
    /// Number of io_uring instances to create.
    pub uring_count: usize,
    /// Queue depth for each io_uring instance.
    pub uring_size: u32,
    /// Channel capacity for subscription ID notifications.
    pub sub_id_chan_size: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            dir: "default".to_string(),
            wrbuf_size: 8 * 1024 * 1024,           //8MB
            file_segment_size: 1024 * 1024 * 1024, // 1GB
            with_twig_file: false,
            aes_keys: None,
            compact_thres: COMPACT_THRES,
            utilization_ratio: UTILIZATION_RATIO,
            utilization_div: UTILIZATION_DIV,
            task_chan_size: TASK_CHAN_SIZE,
            prefetcher_thread_count: PREFETCHER_THREAD_COUNT,
            uring_count: URING_COUNT,
            uring_size: URING_SIZE,
            sub_id_chan_size: SUB_ID_CHAN_SIZE,
        }
    }
}

impl Config {
    /// Create a fully-specified configuration.
    pub fn new(
        dir: &str,
        wrbuf_size: usize,
        file_segment_size: usize,
        with_twig_file: bool,
        aes_keys: Option<[u8; 96]>,
        compact_thres: i64,
        utilization_ratio: i64,
        utilization_div: i64,
        task_chan_size: usize,
        prefetcher_thread_count: usize,
        uring_count: usize,
        uring_size: u32,
        sub_id_chan_size: usize,
    ) -> Self {
        Self {
            dir: dir.to_string(),
            wrbuf_size,
            file_segment_size,
            with_twig_file,
            aes_keys,
            compact_thres,
            utilization_ratio,
            utilization_div,
            task_chan_size,
            prefetcher_thread_count,
            uring_count,
            uring_size,
            sub_id_chan_size,
        }
    }

    /// Create a default configuration rooted at the given directory.
    pub fn from_dir(dir: &str) -> Self {
        let mut config = Config::default();
        config.dir = dir.to_string();
        config
    }

    /// Create a configuration with custom compaction parameters and defaults for everything else.
    pub fn from_dir_and_compact_opt(
        dir: &str,
        compact_thres: i64,
        utilization_ratio: i64,
        utilization_div: i64,
    ) -> Self {
        let mut config = Config::default();
        config.dir = dir.to_string();
        config.compact_thres = compact_thres;
        config.utilization_ratio = utilization_ratio;
        config.utilization_div = utilization_div;
        config
    }

    /// Enable at-rest encryption with the given 96-byte key material.
    pub fn set_aes_keys(&mut self, keys: [u8; 96]) {
        self.aes_keys = Some(keys);
    }

    /// Enable or disable writing twig data to a separate file.
    pub fn set_with_twig_file(&mut self, with_twig_file: bool) {
        self.with_twig_file = with_twig_file;
    }
}
