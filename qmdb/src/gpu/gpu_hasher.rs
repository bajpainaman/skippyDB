use cudarc::driver::{
    result, CudaDevice, CudaFunction, CudaSlice, DevicePtr, DeviceRepr, LaunchAsync, LaunchConfig,
};
use parking_lot::Mutex;
use std::sync::Arc;

/// A single node-hash job: SHA256(level || left || right) = 65 bytes input
#[repr(C)]
#[derive(Clone, Copy)]
pub struct NodeHashJob {
    pub level: u8,
    pub left: [u8; 32],
    pub right: [u8; 32],
}

// Safety: NodeHashJob is a plain C struct with no padding concerns for GPU transfer
unsafe impl DeviceRepr for NodeHashJob {}

const BLOCK_SIZE: u32 = 256;
const PTX_SRC: &str = include_str!("sha256_kernel.cu");

/// Pre-allocated host and device buffers for GPU batch operations.
/// Avoids per-call heap allocation and CUDA device memory allocation.
struct GpuBuffers {
    // Node hash: pre-allocated device buffers
    d_node_input: CudaSlice<u8>,  // max_batch_size * 65
    d_node_output: CudaSlice<u8>, // max_batch_size * 32
    // Node hash: pre-allocated host buffers
    h_node_input: Vec<u8>,  // max_batch_size * 65
    h_node_output: Vec<u8>, // max_batch_size * 32
    // Variable hash: pre-allocated device output + host output
    d_var_output: CudaSlice<u8>, // max_batch_size * 32
    h_var_output: Vec<u8>,       // max_batch_size * 32
}

/// GPU-accelerated batch SHA256 hasher for QMDB Merkle operations.
///
/// Uses a CUDA stream for async pipeline (upload → compute → download).
/// Pre-allocates device and host memory at init for `max_batch_size` jobs,
/// eliminating per-call allocation overhead.
///
/// All methods are safe to call from any thread (operations are serialized
/// by the internal Mutex and CUDA stream).
pub struct GpuHasher {
    device: Arc<CudaDevice>,
    node_hash_fn: CudaFunction,
    var_hash_fn: CudaFunction,
    max_batch_size: usize,
    bufs: Mutex<GpuBuffers>,
}

impl GpuHasher {
    /// Create a new GpuHasher on GPU device 0.
    /// `max_batch_size`: maximum number of hashes per batch (e.g. 200_000).
    /// Pre-compiles the CUDA kernels, creates a stream, and pre-allocates buffers.
    pub fn new(max_batch_size: usize) -> Result<Self, String> {
        // Use new_with_stream for async pipeline (upload → compute → download on one stream)
        let device = CudaDevice::new_with_stream(0)
            .map_err(|e| format!("CUDA device init failed: {}", e))?;

        // Compile PTX from CUDA source at runtime via NVRTC
        let ptx = cudarc::nvrtc::compile_ptx(PTX_SRC)
            .map_err(|e| format!("NVRTC compilation failed: {}", e))?;

        device
            .load_ptx(ptx, "sha256", &["sha256_node_hash", "sha256_variable_hash"])
            .map_err(|e| format!("PTX load failed: {}", e))?;

        let node_hash_fn = device
            .get_func("sha256", "sha256_node_hash")
            .ok_or_else(|| "sha256_node_hash function not found".to_string())?;

        let var_hash_fn = device
            .get_func("sha256", "sha256_variable_hash")
            .ok_or_else(|| "sha256_variable_hash function not found".to_string())?;

        // Pre-allocate persistent device buffers
        let d_node_input: CudaSlice<u8> = device
            .alloc_zeros(max_batch_size * 65)
            .map_err(|e| format!("GPU node input alloc failed: {}", e))?;
        let d_node_output: CudaSlice<u8> = device
            .alloc_zeros(max_batch_size * 32)
            .map_err(|e| format!("GPU node output alloc failed: {}", e))?;
        let d_var_output: CudaSlice<u8> = device
            .alloc_zeros(max_batch_size * 32)
            .map_err(|e| format!("GPU var output alloc failed: {}", e))?;

        let bufs = Mutex::new(GpuBuffers {
            d_node_input,
            d_node_output,
            h_node_input: vec![0u8; max_batch_size * 65],
            h_node_output: vec![0u8; max_batch_size * 32],
            d_var_output,
            h_var_output: vec![0u8; max_batch_size * 32],
        });

        Ok(Self {
            device,
            node_hash_fn,
            var_hash_fn,
            max_batch_size,
            bufs,
        })
    }

    /// Batch-hash N fixed 65-byte node inputs on the GPU.
    /// Each job = SHA256(level_byte || left_32B || right_32B).
    /// Returns N x [u8; 32] hashes, byte-identical to CPU `hasher::hash2`.
    ///
    /// Uses pre-allocated buffers and async CUDA stream for pipelined execution.
    pub fn batch_node_hash(&self, jobs: &[NodeHashJob]) -> Vec<[u8; 32]> {
        let n = jobs.len();
        if n == 0 {
            return Vec::new();
        }
        assert!(
            n <= self.max_batch_size,
            "batch size {} exceeds max {}",
            n,
            self.max_batch_size
        );

        let mut bufs = self.bufs.lock();

        // Flatten jobs into pre-allocated host buffer
        for (i, job) in jobs.iter().enumerate() {
            let off = i * 65;
            bufs.h_node_input[off] = job.level;
            bufs.h_node_input[off + 1..off + 33].copy_from_slice(&job.left);
            bufs.h_node_input[off + 33..off + 65].copy_from_slice(&job.right);
        }

        // Async upload to pre-allocated device buffer (partial copy)
        let stream = *self.device.cu_stream();
        unsafe {
            result::memcpy_htod_async(
                *bufs.d_node_input.device_ptr(),
                &bufs.h_node_input[..n * 65],
                stream,
            )
            .expect("GPU input upload failed");
        }

        // Launch kernel (queued on same stream, executes after upload completes)
        let grid = ((n as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);
        let cfg = LaunchConfig {
            grid_dim: grid,
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.node_hash_fn
                .clone()
                .launch(cfg, (&bufs.d_node_input, &bufs.d_node_output, n as u32))
                .expect("GPU kernel launch failed");
        }

        // Async download to pre-allocated host buffer (partial copy)
        let d_out_ptr = *bufs.d_node_output.device_ptr();
        unsafe {
            result::memcpy_dtoh_async(
                &mut bufs.h_node_output[..n * 32],
                d_out_ptr,
                stream,
            )
            .expect("GPU output download failed");
        }

        // Synchronize: wait for upload → compute → download pipeline to complete
        self.device.synchronize().expect("GPU sync failed");

        // Convert flat bytes to array of [u8; 32]
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            let mut hash = [0u8; 32];
            hash.copy_from_slice(&bufs.h_node_output[i * 32..(i + 1) * 32]);
            result.push(hash);
        }
        result
    }

    /// Batch-hash N fixed 65-byte node inputs, writing directly into an output slice.
    /// More efficient than `batch_node_hash` when you already have the output buffer.
    pub fn batch_node_hash_into(&self, jobs: &[NodeHashJob], out: &mut [[u8; 32]]) {
        let n = jobs.len();
        assert_eq!(n, out.len(), "jobs and output length mismatch");
        if n == 0 {
            return;
        }
        let hashes = self.batch_node_hash(jobs);
        out.copy_from_slice(&hashes);
    }

    /// Batch-hash N variable-length inputs on the GPU.
    /// Each input can be any length (typically 50-300 bytes for entries).
    /// Returns N x [u8; 32] hashes, byte-identical to CPU `hasher::hash`.
    ///
    /// Variable-length inputs require per-call allocation for the data buffer
    /// (since total size varies), but output buffers are pre-allocated.
    pub fn batch_hash_variable(&self, inputs: &[&[u8]]) -> Vec<[u8; 32]> {
        let n = inputs.len();
        if n == 0 {
            return Vec::new();
        }
        assert!(
            n <= self.max_batch_size,
            "batch size {} exceeds max {}",
            n,
            self.max_batch_size
        );

        // Build flat data buffer + offset/length arrays
        let total_bytes: usize = inputs.iter().map(|x| x.len()).sum();
        let mut flat_data = Vec::with_capacity(total_bytes);
        let mut offsets = Vec::with_capacity(n);
        let mut lengths = Vec::with_capacity(n);

        for input in inputs {
            offsets.push(flat_data.len() as u32);
            lengths.push(input.len() as u32);
            flat_data.extend_from_slice(input);
        }

        // Variable-length data buffer must be allocated per call (size varies)
        let d_data = self
            .device
            .htod_copy(flat_data)
            .expect("GPU data upload failed");
        let d_offsets = self
            .device
            .htod_copy(offsets)
            .expect("GPU offsets upload failed");
        let d_lengths = self
            .device
            .htod_copy(lengths)
            .expect("GPU lengths upload failed");

        let mut bufs = self.bufs.lock();

        // Launch kernel
        let grid = ((n as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);
        let cfg = LaunchConfig {
            grid_dim: grid,
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.var_hash_fn
                .clone()
                .launch(
                    cfg,
                    (&d_data, &d_offsets, &d_lengths, &bufs.d_var_output, n as u32),
                )
                .expect("GPU variable hash kernel launch failed");
        }

        // Download to pre-allocated host buffer
        let stream = *self.device.cu_stream();
        let d_out_ptr = *bufs.d_var_output.device_ptr();
        unsafe {
            result::memcpy_dtoh_async(
                &mut bufs.h_var_output[..n * 32],
                d_out_ptr,
                stream,
            )
            .expect("GPU output download failed");
        }

        self.device.synchronize().expect("GPU sync failed");

        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            let mut hash = [0u8; 32];
            hash.copy_from_slice(&bufs.h_var_output[i * 32..(i + 1) * 32]);
            result.push(hash);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    fn cpu_hash2(level: u8, a: &[u8; 32], b: &[u8; 32]) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update([level]);
        hasher.update(a);
        hasher.update(b);
        hasher.finalize().into()
    }

    fn cpu_hash(data: &[u8]) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(data);
        hasher.finalize().into()
    }

    #[test]
    fn test_gpu_node_hash_matches_cpu() {
        let gpu = match GpuHasher::new(10000) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("Skipping GPU test (no CUDA device): {}", e);
                return;
            }
        };

        // Test with known inputs
        let mut jobs = Vec::new();
        let mut expected = Vec::new();

        for level in 0u8..12 {
            let left = [level.wrapping_mul(7).wrapping_add(0x11); 32];
            let right = [level.wrapping_mul(13).wrapping_add(0xAB); 32];
            jobs.push(NodeHashJob { level, left, right });
            expected.push(cpu_hash2(level, &left, &right));
        }

        // Test with zero hashes
        jobs.push(NodeHashJob {
            level: 0,
            left: [0; 32],
            right: [0; 32],
        });
        expected.push(cpu_hash2(0, &[0; 32], &[0; 32]));

        // Test with max values
        jobs.push(NodeHashJob {
            level: 255,
            left: [0xFF; 32],
            right: [0xFF; 32],
        });
        expected.push(cpu_hash2(255, &[0xFF; 32], &[0xFF; 32]));

        let gpu_results = gpu.batch_node_hash(&jobs);

        for (i, (gpu_hash, cpu_hash)) in gpu_results.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                gpu_hash, cpu_hash,
                "Mismatch at job {}: GPU={} CPU={}",
                i,
                hex::encode(gpu_hash),
                hex::encode(cpu_hash)
            );
        }
    }

    #[test]
    fn test_gpu_node_hash_large_batch() {
        let gpu = match GpuHasher::new(100000) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("Skipping GPU test (no CUDA device): {}", e);
                return;
            }
        };

        let n = 10000;
        let mut jobs = Vec::with_capacity(n);
        let mut expected = Vec::with_capacity(n);

        for i in 0..n {
            let level = (i % 12) as u8;
            let mut left = [0u8; 32];
            let mut right = [0u8; 32];
            // Deterministic pseudo-random fill
            for j in 0..32 {
                left[j] = ((i * 7 + j * 13) & 0xFF) as u8;
                right[j] = ((i * 11 + j * 17) & 0xFF) as u8;
            }
            jobs.push(NodeHashJob { level, left, right });
            expected.push(cpu_hash2(level, &left, &right));
        }

        let gpu_results = gpu.batch_node_hash(&jobs);
        assert_eq!(gpu_results.len(), n);

        for (i, (gpu_hash, cpu_hash)) in gpu_results.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                gpu_hash, cpu_hash,
                "Mismatch at job {}: GPU={} CPU={}",
                i,
                hex::encode(gpu_hash),
                hex::encode(cpu_hash)
            );
        }
    }

    #[test]
    fn test_gpu_variable_hash_matches_cpu() {
        let gpu = match GpuHasher::new(10000) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("Skipping GPU test (no CUDA device): {}", e);
                return;
            }
        };

        // Test various lengths
        let test_data: Vec<Vec<u8>> = vec![
            vec![0x01; 10],             // short
            vec![0x02; 53],             // just under one block
            vec![0x03; 55],             // one block boundary
            vec![0x04; 56],             // triggers two-block padding
            vec![0x05; 64],             // exactly one block
            vec![0x06; 65],             // just over one block (our node hash size)
            vec![0x07; 100],            // typical entry size
            vec![0x08; 200],            // larger entry
            vec![0x09; 300],            // max typical entry
            (0..256).map(|i| i as u8).collect(), // sequential bytes
        ];

        let inputs: Vec<&[u8]> = test_data.iter().map(|v| v.as_slice()).collect();
        let expected: Vec<[u8; 32]> = test_data.iter().map(|v| cpu_hash(v)).collect();

        let gpu_results = gpu.batch_hash_variable(&inputs);

        for (i, (gpu_hash, cpu_hash)) in gpu_results.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                gpu_hash, cpu_hash,
                "Variable hash mismatch at {} (len={}): GPU={} CPU={}",
                i,
                test_data[i].len(),
                hex::encode(gpu_hash),
                hex::encode(cpu_hash)
            );
        }
    }

    #[test]
    fn test_gpu_empty_batch() {
        let gpu = match GpuHasher::new(1000) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("Skipping GPU test (no CUDA device): {}", e);
                return;
            }
        };

        let empty_jobs: Vec<NodeHashJob> = vec![];
        let result = gpu.batch_node_hash(&empty_jobs);
        assert!(result.is_empty());

        let empty_inputs: Vec<&[u8]> = vec![];
        let result = gpu.batch_hash_variable(&empty_inputs);
        assert!(result.is_empty());
    }
}
