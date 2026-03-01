//! NUMA-aware allocation utilities for shard-local memory placement.
//!
//! On NUMA systems, this module provides helpers to:
//! - Detect the NUMA topology (number of nodes, CPUs per node)
//! - Map shard IDs to NUMA nodes for locality-aware allocation
//! - Hint the OS to prefer local memory for shard-specific allocations
//!
//! Falls back gracefully to standard allocation on non-NUMA or single-node systems.

use std::fs;

/// Detected NUMA topology.
#[derive(Debug, Clone)]
pub struct NumaTopology {
    /// Number of NUMA nodes on this system.
    pub num_nodes: usize,
    /// CPU IDs belonging to each NUMA node (indexed by node ID).
    pub node_cpus: Vec<Vec<usize>>,
}

impl NumaTopology {
    /// Detect NUMA topology by reading /sys/devices/system/node/.
    /// Returns a single-node topology if detection fails or not on Linux.
    pub fn detect() -> Self {
        Self::detect_from_sysfs("/sys/devices/system/node")
    }

    fn detect_from_sysfs(base_path: &str) -> Self {
        let fallback = Self {
            num_nodes: 1,
            node_cpus: vec![Self::all_cpus()],
        };

        let entries = match fs::read_dir(base_path) {
            Ok(e) => e,
            Err(_) => return fallback,
        };

        let mut node_cpus: Vec<(usize, Vec<usize>)> = Vec::new();

        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if !name.starts_with("node") {
                continue;
            }
            let node_id: usize = match name[4..].parse() {
                Ok(id) => id,
                Err(_) => continue,
            };

            let cpulist_path = format!("{}/{}/cpulist", base_path, name);
            let cpus = match fs::read_to_string(&cpulist_path) {
                Ok(s) => parse_cpulist(&s),
                Err(_) => Vec::new(),
            };

            node_cpus.push((node_id, cpus));
        }

        if node_cpus.is_empty() {
            return fallback;
        }

        node_cpus.sort_by_key(|(id, _)| *id);
        let max_node = node_cpus.last().unwrap().0;
        let mut result = vec![Vec::new(); max_node + 1];
        for (id, cpus) in node_cpus {
            result[id] = cpus;
        }

        Self {
            num_nodes: result.len(),
            node_cpus: result,
        }
    }

    fn all_cpus() -> Vec<usize> {
        match fs::read_to_string("/sys/devices/system/cpu/online") {
            Ok(s) => parse_cpulist(&s),
            Err(_) => (0..num_cpus()).collect(),
        }
    }

    /// Returns true if this is a multi-node NUMA system.
    pub fn is_numa(&self) -> bool {
        self.num_nodes > 1
    }

    /// Map a shard ID to a NUMA node (round-robin distribution).
    pub fn shard_to_node(&self, shard_id: usize) -> usize {
        shard_id % self.num_nodes
    }

    /// Get the preferred CPUs for a given shard ID.
    pub fn cpus_for_shard(&self, shard_id: usize) -> &[usize] {
        &self.node_cpus[self.shard_to_node(shard_id)]
    }
}

/// Parse a Linux cpulist string like "0-3,6,8-11" into a Vec of CPU IDs.
fn parse_cpulist(s: &str) -> Vec<usize> {
    let mut cpus = Vec::new();
    for part in s.trim().split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let Some((start, end)) = part.split_once('-') {
            if let (Ok(s), Ok(e)) = (start.parse::<usize>(), end.parse::<usize>()) {
                cpus.extend(s..=e);
            }
        } else if let Ok(cpu) = part.parse::<usize>() {
            cpus.push(cpu);
        }
    }
    cpus
}

/// Best-effort CPU count detection.
fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_cpulist_single() {
        assert_eq!(parse_cpulist("0"), vec![0]);
    }

    #[test]
    fn test_parse_cpulist_range() {
        assert_eq!(parse_cpulist("0-3"), vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_parse_cpulist_mixed() {
        assert_eq!(parse_cpulist("0-2,5,8-10"), vec![0, 1, 2, 5, 8, 9, 10]);
    }

    #[test]
    fn test_parse_cpulist_with_newline() {
        assert_eq!(parse_cpulist("0-3\n"), vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_parse_cpulist_empty() {
        assert_eq!(parse_cpulist(""), Vec::<usize>::new());
    }

    #[test]
    fn test_topology_detect() {
        let topo = NumaTopology::detect();
        assert!(topo.num_nodes >= 1);
        // At least one node should have CPUs
        assert!(topo.node_cpus.iter().any(|cpus| !cpus.is_empty()));
    }

    #[test]
    fn test_shard_to_node_single_node() {
        let topo = NumaTopology {
            num_nodes: 1,
            node_cpus: vec![vec![0, 1, 2, 3]],
        };
        assert_eq!(topo.shard_to_node(0), 0);
        assert_eq!(topo.shard_to_node(15), 0);
        assert!(!topo.is_numa());
    }

    #[test]
    fn test_shard_to_node_multi_node() {
        let topo = NumaTopology {
            num_nodes: 2,
            node_cpus: vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]],
        };
        assert_eq!(topo.shard_to_node(0), 0);
        assert_eq!(topo.shard_to_node(1), 1);
        assert_eq!(topo.shard_to_node(2), 0);
        assert_eq!(topo.shard_to_node(15), 1);
        assert!(topo.is_numa());
    }

    #[test]
    fn test_cpus_for_shard() {
        let topo = NumaTopology {
            num_nodes: 2,
            node_cpus: vec![vec![0, 1], vec![2, 3]],
        };
        assert_eq!(topo.cpus_for_shard(0), &[0, 1]);
        assert_eq!(topo.cpus_for_shard(1), &[2, 3]);
        assert_eq!(topo.cpus_for_shard(2), &[0, 1]);
    }

    #[test]
    fn test_fallback_non_numa() {
        // Non-existent path triggers fallback
        let topo = NumaTopology::detect_from_sysfs("/nonexistent/path");
        assert_eq!(topo.num_nodes, 1);
        assert!(!topo.node_cpus[0].is_empty());
    }
}
