use cudarc::driver::{
    result, CudaDevice, CudaFunction, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, DeviceRepr,
    LaunchAsync, LaunchConfig,
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
    // Node hash (AoS): pre-allocated device buffers
    d_node_input: CudaSlice<u8>,  // max_batch_size * 65
    d_node_output: CudaSlice<u8>, // max_batch_size * 32
    // Node hash (AoS): pre-allocated host buffers
    h_node_input: Vec<u8>,  // max_batch_size * 65
    h_node_output: Vec<u8>, // max_batch_size * 32
    // Node hash (SoA): pre-allocated device buffers
    d_soa_levels: CudaSlice<u8>,  // max_batch_size * 1
    d_soa_lefts: CudaSlice<u8>,   // max_batch_size * 32
    d_soa_rights: CudaSlice<u8>,  // max_batch_size * 32
    // Node hash (SoA): pre-allocated host buffers
    h_soa_levels: Vec<u8>,        // max_batch_size * 1
    h_soa_lefts: Vec<u8>,         // max_batch_size * 32
    h_soa_rights: Vec<u8>,        // max_batch_size * 32
    // Variable hash: pre-allocated device output + host output
    d_var_output: CudaSlice<u8>, // max_batch_size * 32
    h_var_output: Vec<u8>,       // max_batch_size * 32
}

/// Secondary buffers for the async stream (ping-pong pipelining).
/// Separate from primary buffers so async dispatch doesn't conflict
/// with synchronous operations on the default stream.
struct AsyncGpuBuffers {
    d_node_input: CudaSlice<u8>,
    d_node_output: CudaSlice<u8>,
    h_node_input: Vec<u8>,
    h_node_output: Vec<u8>,
}

/// Handle to an in-flight GPU computation on the async stream.
/// Callers can do CPU work between dispatching and collecting results.
/// Dropping without calling `wait()` will synchronize implicitly.
pub struct GpuPending<'a> {
    device: &'a Arc<CudaDevice>,
    stream: &'a CudaStream,
    async_bufs: &'a AsyncGpuBuffers,
    count: usize,
}

impl<'a> GpuPending<'a> {
    /// Block until the async computation completes and return results.
    pub fn wait(self) -> Vec<[u8; 32]> {
        // Safety: synchronize is a blocking FFI call on a valid CUDA stream handle.
        // The stream is owned by GpuHasher and only accessed under serialization.
        unsafe { result::stream::synchronize(self.stream.stream) }
            .expect("GPU async stream sync failed");
        let hashes: &[[u8; 32]] =
            bytemuck::cast_slice(&self.async_bufs.h_node_output[..self.count * 32]);
        hashes.to_vec()
    }

    /// Block until complete, writing results directly into caller's buffer.
    pub fn wait_into(self, out: &mut [[u8; 32]]) {
        assert_eq!(self.count, out.len(), "pending count and output mismatch");
        // Safety: synchronize is a blocking FFI call on a valid CUDA stream handle.
        unsafe { result::stream::synchronize(self.stream.stream) }
            .expect("GPU async stream sync failed");
        let hashes: &[[u8; 32]] =
            bytemuck::cast_slice(&self.async_bufs.h_node_output[..self.count * 32]);
        out.copy_from_slice(hashes);
    }
}

/// GPU-accelerated batch SHA256 hasher for QMDB Merkle operations.
///
/// Uses a CUDA stream for async pipeline (upload → compute → download).
/// Pre-allocates device and host memory at init for `max_batch_size` jobs,
/// eliminating per-call allocation overhead.
///
/// Supports dual-stream pipelining: the default stream handles synchronous
/// operations while a secondary stream allows overlapping H↔D transfers
/// with computation via `batch_node_hash_async()`.
///
/// All methods are safe to call from any thread (operations are serialized
/// by the internal Mutex and CUDA stream).
pub struct GpuHasher {
    device: Arc<CudaDevice>,
    node_hash_fn: CudaFunction,
    var_hash_fn: CudaFunction,
    warp_coop_fn: CudaFunction,
    soa_hash_fn: CudaFunction,
    fused_active_bits_fn: CudaFunction,
    max_batch_size: usize,
    bufs: Mutex<GpuBuffers>,
    async_stream: CudaStream,
    async_bufs: AsyncGpuBuffers,
}

// Safety: GpuHasher operations are serialized by the internal Mutex on `bufs`.
// The CudaStream raw pointer is only accessed through cudarc's safe API and
// the async_stream is only used via GpuPending which borrows &self.
unsafe impl Send for GpuHasher {}
unsafe impl Sync for GpuHasher {}

impl GpuHasher {
    /// Returns the number of available CUDA devices.
    pub fn device_count() -> Result<i32, String> {
        CudaDevice::count().map_err(|e| format!("CUDA device count failed: {}", e))
    }

    /// Create a new GpuHasher on GPU device 0.
    /// `max_batch_size`: maximum number of hashes per batch (e.g. 200_000).
    /// Pre-compiles the CUDA kernels, creates a stream, and pre-allocates buffers.
    pub fn new(max_batch_size: usize) -> Result<Self, String> {
        Self::new_on_device(0, max_batch_size)
    }

    /// Create a new GpuHasher on a specific GPU device.
    /// `ordinal`: CUDA device index (0, 1, 2, ...).
    /// `max_batch_size`: maximum number of hashes per batch.
    pub fn new_on_device(ordinal: usize, max_batch_size: usize) -> Result<Self, String> {
        let device = CudaDevice::new_with_stream(ordinal)
            .map_err(|e| format!("CUDA device {} init failed: {}", ordinal, e))?;

        // Compile PTX from CUDA source at runtime via NVRTC.
        // -I/usr/include so NVRTC can resolve stdint.h on Linux.
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(
            PTX_SRC,
            cudarc::nvrtc::CompileOptions {
                options: vec!["-I/usr/include".to_string()],
                ..Default::default()
            },
        ).map_err(|e| format!("NVRTC compilation failed: {}", e))?;

        device
            .load_ptx(
                ptx,
                "sha256",
                &[
                    "sha256_node_hash",
                    "sha256_variable_hash",
                    "sha256_node_hash_warp_coop",
                    "sha256_node_hash_soa",
                    "sha256_active_bits_fused",
                ],
            )
            .map_err(|e| format!("PTX load failed: {}", e))?;

        let node_hash_fn = device
            .get_func("sha256", "sha256_node_hash")
            .ok_or_else(|| "sha256_node_hash function not found".to_string())?;

        let var_hash_fn = device
            .get_func("sha256", "sha256_variable_hash")
            .ok_or_else(|| "sha256_variable_hash function not found".to_string())?;

        let warp_coop_fn = device
            .get_func("sha256", "sha256_node_hash_warp_coop")
            .ok_or_else(|| "sha256_node_hash_warp_coop function not found".to_string())?;

        let soa_hash_fn = device
            .get_func("sha256", "sha256_node_hash_soa")
            .ok_or_else(|| "sha256_node_hash_soa function not found".to_string())?;

        let fused_active_bits_fn = device
            .get_func("sha256", "sha256_active_bits_fused")
            .ok_or_else(|| "sha256_active_bits_fused function not found".to_string())?;

        // Pre-allocate persistent device buffers (AoS)
        let d_node_input: CudaSlice<u8> = device
            .alloc_zeros(max_batch_size * 65)
            .map_err(|e| format!("GPU node input alloc failed: {}", e))?;
        let d_node_output: CudaSlice<u8> = device
            .alloc_zeros(max_batch_size * 32)
            .map_err(|e| format!("GPU node output alloc failed: {}", e))?;

        // Pre-allocate persistent device buffers (SoA)
        let d_soa_levels: CudaSlice<u8> = device
            .alloc_zeros(max_batch_size)
            .map_err(|e| format!("GPU SoA levels alloc failed: {}", e))?;
        let d_soa_lefts: CudaSlice<u8> = device
            .alloc_zeros(max_batch_size * 32)
            .map_err(|e| format!("GPU SoA lefts alloc failed: {}", e))?;
        let d_soa_rights: CudaSlice<u8> = device
            .alloc_zeros(max_batch_size * 32)
            .map_err(|e| format!("GPU SoA rights alloc failed: {}", e))?;

        let d_var_output: CudaSlice<u8> = device
            .alloc_zeros(max_batch_size * 32)
            .map_err(|e| format!("GPU var output alloc failed: {}", e))?;

        let bufs = Mutex::new(GpuBuffers {
            d_node_input,
            d_node_output,
            h_node_input: vec![0u8; max_batch_size * 65],
            h_node_output: vec![0u8; max_batch_size * 32],
            d_soa_levels,
            d_soa_lefts,
            d_soa_rights,
            h_soa_levels: vec![0u8; max_batch_size],
            h_soa_lefts: vec![0u8; max_batch_size * 32],
            h_soa_rights: vec![0u8; max_batch_size * 32],
            d_var_output,
            h_var_output: vec![0u8; max_batch_size * 32],
        });

        // Secondary stream + buffers for async pipelining
        let async_stream = device
            .fork_default_stream()
            .map_err(|e| format!("Async stream creation failed: {}", e))?;
        let async_d_input: CudaSlice<u8> = device
            .alloc_zeros(max_batch_size * 65)
            .map_err(|e| format!("GPU async input alloc failed: {}", e))?;
        let async_d_output: CudaSlice<u8> = device
            .alloc_zeros(max_batch_size * 32)
            .map_err(|e| format!("GPU async output alloc failed: {}", e))?;
        let async_bufs = AsyncGpuBuffers {
            d_node_input: async_d_input,
            d_node_output: async_d_output,
            h_node_input: vec![0u8; max_batch_size * 65],
            h_node_output: vec![0u8; max_batch_size * 32],
        };

        Ok(Self {
            device,
            node_hash_fn,
            var_hash_fn,
            warp_coop_fn,
            soa_hash_fn,
            fused_active_bits_fn,
            max_batch_size,
            bufs,
            async_stream,
            async_bufs,
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

        // Zero-copy reinterpret flat u8 buffer as [[u8; 32]]
        let hashes: &[[u8; 32]] =
            bytemuck::cast_slice(&bufs.h_node_output[..n * 32]);
        hashes.to_vec()
    }

    /// Batch-hash N fixed 65-byte node inputs, writing directly into an output slice.
    /// Writes results directly from GPU host buffer — no intermediate Vec allocation.
    pub fn batch_node_hash_into(&self, jobs: &[NodeHashJob], out: &mut [[u8; 32]]) {
        let n = jobs.len();
        assert_eq!(n, out.len(), "jobs and output length mismatch");
        if n == 0 {
            return;
        }
        assert!(
            n <= self.max_batch_size,
            "batch size {} exceeds max {}",
            n,
            self.max_batch_size
        );

        let mut bufs = self.bufs.lock();

        for (i, job) in jobs.iter().enumerate() {
            let off = i * 65;
            bufs.h_node_input[off] = job.level;
            bufs.h_node_input[off + 1..off + 33].copy_from_slice(&job.left);
            bufs.h_node_input[off + 33..off + 65].copy_from_slice(&job.right);
        }

        let stream = *self.device.cu_stream();
        unsafe {
            result::memcpy_htod_async(
                *bufs.d_node_input.device_ptr(),
                &bufs.h_node_input[..n * 65],
                stream,
            )
            .expect("GPU input upload failed");
        }

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

        let d_out_ptr = *bufs.d_node_output.device_ptr();
        unsafe {
            result::memcpy_dtoh_async(
                &mut bufs.h_node_output[..n * 32],
                d_out_ptr,
                stream,
            )
            .expect("GPU output download failed");
        }

        self.device.synchronize().expect("GPU sync failed");

        // Copy directly from host buffer into caller's output slice
        let hashes: &[[u8; 32]] =
            bytemuck::cast_slice(&bufs.h_node_output[..n * 32]);
        out.copy_from_slice(hashes);
    }

    /// Batch-hash using the warp-cooperative kernel (8 threads per hash).
    /// Functionally identical to `batch_node_hash` but uses warp shuffles
    /// for inter-thread state rotation, potentially improving throughput
    /// on GPUs with high SM counts.
    pub fn batch_node_hash_warp_coop(&self, jobs: &[NodeHashJob]) -> Vec<[u8; 32]> {
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

        let stream = *self.device.cu_stream();
        unsafe {
            result::memcpy_htod_async(
                *bufs.d_node_input.device_ptr(),
                &bufs.h_node_input[..n * 65],
                stream,
            )
            .expect("GPU input upload failed");
        }

        // 8 threads per job
        let total_threads = n as u32 * 8;
        let grid = ((total_threads + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);
        let cfg = LaunchConfig {
            grid_dim: grid,
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.warp_coop_fn
                .clone()
                .launch(cfg, (&bufs.d_node_input, &bufs.d_node_output, n as u32))
                .expect("GPU warp-coop kernel launch failed");
        }

        let d_out_ptr = *bufs.d_node_output.device_ptr();
        unsafe {
            result::memcpy_dtoh_async(
                &mut bufs.h_node_output[..n * 32],
                d_out_ptr,
                stream,
            )
            .expect("GPU output download failed");
        }

        self.device.synchronize().expect("GPU sync failed");

        let hashes: &[[u8; 32]] =
            bytemuck::cast_slice(&bufs.h_node_output[..n * 32]);
        hashes.to_vec()
    }

    /// Batch-hash N fixed 65-byte node inputs using Structure-of-Arrays layout.
    /// Input is provided as three separate slices: levels, lefts, rights.
    ///
    /// SoA layout enables coalesced GPU memory reads: adjacent threads read
    /// adjacent 32-byte blocks from the lefts/rights arrays (32B stride)
    /// instead of the AoS 65-byte stride, improving memory bandwidth utilization.
    ///
    /// Functionally identical to `batch_node_hash` — produces byte-identical results.
    pub fn batch_node_hash_soa(
        &self,
        levels: &[u8],
        lefts: &[[u8; 32]],
        rights: &[[u8; 32]],
    ) -> Vec<[u8; 32]> {
        let n = levels.len();
        assert_eq!(n, lefts.len(), "levels and lefts length mismatch");
        assert_eq!(n, rights.len(), "levels and rights length mismatch");
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

        // Copy levels into host buffer (contiguous)
        bufs.h_soa_levels[..n].copy_from_slice(levels);

        // Copy lefts into host buffer (flatten [u8; 32] → contiguous bytes)
        for (i, left) in lefts.iter().enumerate() {
            bufs.h_soa_lefts[i * 32..(i + 1) * 32].copy_from_slice(left);
        }

        // Copy rights into host buffer
        for (i, right) in rights.iter().enumerate() {
            bufs.h_soa_rights[i * 32..(i + 1) * 32].copy_from_slice(right);
        }

        // Async upload three separate arrays
        let stream = *self.device.cu_stream();
        unsafe {
            result::memcpy_htod_async(
                *bufs.d_soa_levels.device_ptr(),
                &bufs.h_soa_levels[..n],
                stream,
            )
            .expect("GPU SoA levels upload failed");

            result::memcpy_htod_async(
                *bufs.d_soa_lefts.device_ptr(),
                &bufs.h_soa_lefts[..n * 32],
                stream,
            )
            .expect("GPU SoA lefts upload failed");

            result::memcpy_htod_async(
                *bufs.d_soa_rights.device_ptr(),
                &bufs.h_soa_rights[..n * 32],
                stream,
            )
            .expect("GPU SoA rights upload failed");
        }

        // Launch SoA kernel
        let grid = ((n as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);
        let cfg = LaunchConfig {
            grid_dim: grid,
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.soa_hash_fn
                .clone()
                .launch(
                    cfg,
                    (
                        &bufs.d_soa_levels,
                        &bufs.d_soa_lefts,
                        &bufs.d_soa_rights,
                        &bufs.d_node_output,
                        n as u32,
                    ),
                )
                .expect("GPU SoA kernel launch failed");
        }

        // Async download
        let d_out_ptr = *bufs.d_node_output.device_ptr();
        unsafe {
            result::memcpy_dtoh_async(
                &mut bufs.h_node_output[..n * 32],
                d_out_ptr,
                stream,
            )
            .expect("GPU SoA output download failed");
        }

        self.device.synchronize().expect("GPU sync failed");

        let hashes: &[[u8; 32]] =
            bytemuck::cast_slice(&bufs.h_node_output[..n * 32]);
        hashes.to_vec()
    }

    /// Batch-hash N fixed 65-byte node inputs using SoA layout, writing into an output slice.
    /// Writes directly from GPU host buffer — no intermediate Vec allocation.
    pub fn batch_node_hash_soa_into(
        &self,
        levels: &[u8],
        lefts: &[[u8; 32]],
        rights: &[[u8; 32]],
        out: &mut [[u8; 32]],
    ) {
        let n = levels.len();
        assert_eq!(n, out.len(), "levels and output length mismatch");
        if n == 0 {
            return;
        }
        assert_eq!(n, lefts.len(), "levels and lefts length mismatch");
        assert_eq!(n, rights.len(), "levels and rights length mismatch");
        assert!(
            n <= self.max_batch_size,
            "batch size {} exceeds max {}",
            n,
            self.max_batch_size
        );

        let mut bufs = self.bufs.lock();

        bufs.h_soa_levels[..n].copy_from_slice(levels);
        for (i, left) in lefts.iter().enumerate() {
            bufs.h_soa_lefts[i * 32..(i + 1) * 32].copy_from_slice(left);
        }
        for (i, right) in rights.iter().enumerate() {
            bufs.h_soa_rights[i * 32..(i + 1) * 32].copy_from_slice(right);
        }

        let stream = *self.device.cu_stream();
        unsafe {
            result::memcpy_htod_async(
                *bufs.d_soa_levels.device_ptr(),
                &bufs.h_soa_levels[..n],
                stream,
            )
            .expect("GPU SoA levels upload failed");
            result::memcpy_htod_async(
                *bufs.d_soa_lefts.device_ptr(),
                &bufs.h_soa_lefts[..n * 32],
                stream,
            )
            .expect("GPU SoA lefts upload failed");
            result::memcpy_htod_async(
                *bufs.d_soa_rights.device_ptr(),
                &bufs.h_soa_rights[..n * 32],
                stream,
            )
            .expect("GPU SoA rights upload failed");
        }

        let grid = ((n as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);
        let cfg = LaunchConfig {
            grid_dim: grid,
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.soa_hash_fn
                .clone()
                .launch(
                    cfg,
                    (
                        &bufs.d_soa_levels,
                        &bufs.d_soa_lefts,
                        &bufs.d_soa_rights,
                        &bufs.d_node_output,
                        n as u32,
                    ),
                )
                .expect("GPU SoA kernel launch failed");
        }

        let d_out_ptr = *bufs.d_node_output.device_ptr();
        unsafe {
            result::memcpy_dtoh_async(
                &mut bufs.h_node_output[..n * 32],
                d_out_ptr,
                stream,
            )
            .expect("GPU SoA output download failed");
        }

        self.device.synchronize().expect("GPU sync failed");

        let hashes: &[[u8; 32]] =
            bytemuck::cast_slice(&bufs.h_node_output[..n * 32]);
        out.copy_from_slice(hashes);
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

        let hashes: &[[u8; 32]] =
            bytemuck::cast_slice(&bufs.h_var_output[..n * 32]);
        hashes.to_vec()
    }

    /// Get a reference to the underlying CUDA device.
    /// Allows sharing the device context with flash-map or other CUDA code.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Device-to-device SHA256 SoA hash: inputs and output stay on GPU.
    ///
    /// Takes device-side SoA buffers (levels, lefts, rights) and produces
    /// hashes directly into a device output buffer. No H↔D transfers.
    ///
    /// - `d_levels`: N bytes on device (one level byte per job)
    /// - `d_lefts`: N*32 bytes on device (contiguous left hashes)
    /// - `d_rights`: N*32 bytes on device (contiguous right hashes)
    /// - `n`: number of hash jobs
    ///
    /// Returns `CudaSlice<u8>` of N*32 bytes (hash results on device).
    pub fn batch_node_hash_device_soa(
        &self,
        d_levels: &CudaSlice<u8>,
        d_lefts: &CudaSlice<u8>,
        d_rights: &CudaSlice<u8>,
        n: usize,
    ) -> CudaSlice<u8> {
        assert!(
            n <= self.max_batch_size,
            "batch size {} exceeds max {}",
            n,
            self.max_batch_size
        );

        if n == 0 {
            return self
                .device
                .alloc_zeros::<u8>(0)
                .expect("GPU empty alloc failed");
        }

        let bufs = self.bufs.lock();

        let grid = ((n as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);
        let cfg = LaunchConfig {
            grid_dim: grid,
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.soa_hash_fn
                .clone()
                .launch(
                    cfg,
                    (
                        d_levels,
                        d_lefts,
                        d_rights,
                        &bufs.d_node_output,
                        n as u32,
                    ),
                )
                .expect("GPU SoA device kernel launch failed");
        }

        // Copy results to a new device buffer so we can release the pre-allocated buffer
        let mut d_result: CudaSlice<u8> = self
            .device
            .alloc_zeros(n * 32)
            .expect("GPU result alloc failed");

        let stream = *self.device.cu_stream();
        unsafe {
            result::memcpy_dtod_async(
                *d_result.device_ptr_mut(),
                *bufs.d_node_output.device_ptr(),
                n * 32,
                stream,
            )
            .expect("GPU dtod copy failed");
        }

        drop(bufs);
        d_result
    }

    /// Create a device buffer filled with a repeated byte value.
    /// Useful for creating level byte arrays on the GPU.
    pub fn fill_device_bytes(&self, value: u8, count: usize) -> CudaSlice<u8> {
        let host = vec![value; count];
        self.device
            .htod_copy(host)
            .expect("GPU fill upload failed")
    }

    /// Synchronize the CUDA stream, blocking until all queued operations complete.
    pub fn sync(&self) {
        self.device.synchronize().expect("GPU sync failed");
    }

    /// Dispatch a batch node hash on the async (secondary) stream.
    ///
    /// Returns a `GpuPending` handle. The caller can perform CPU work
    /// (e.g., building the next level's job list) while the GPU computes,
    /// then call `handle.wait()` to collect results.
    ///
    /// This enables pipelining: level N's upload overlaps with level N-1's
    /// compute on the default stream.
    ///
    /// **Note:** Only one async dispatch can be in flight at a time.
    /// The caller must `wait()` on the previous handle before dispatching again.
    pub fn batch_node_hash_async(&mut self, jobs: &[NodeHashJob]) -> GpuPending<'_> {
        let n = jobs.len();
        assert!(n > 0, "async dispatch requires at least 1 job");
        assert!(
            n <= self.max_batch_size,
            "batch size {} exceeds max {}",
            n,
            self.max_batch_size
        );

        // Fill async host buffer
        for (i, job) in jobs.iter().enumerate() {
            let off = i * 65;
            self.async_bufs.h_node_input[off] = job.level;
            self.async_bufs.h_node_input[off + 1..off + 33]
                .copy_from_slice(&job.left);
            self.async_bufs.h_node_input[off + 33..off + 65]
                .copy_from_slice(&job.right);
        }

        let stream = &self.async_stream;

        // Ensure async stream waits for any prior default-stream work
        stream.wait_for_default()
            .expect("async stream wait_for_default failed");

        // Upload on async stream
        unsafe {
            result::memcpy_htod_async(
                *self.async_bufs.d_node_input.device_ptr(),
                &self.async_bufs.h_node_input[..n * 65],
                stream.stream,
            )
            .expect("GPU async input upload failed");
        }

        // Launch kernel on async stream
        let grid = ((n as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);
        let cfg = LaunchConfig {
            grid_dim: grid,
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.node_hash_fn
                .clone()
                .launch_on_stream(
                    stream,
                    cfg,
                    (
                        &self.async_bufs.d_node_input,
                        &self.async_bufs.d_node_output,
                        n as u32,
                    ),
                )
                .expect("GPU async kernel launch failed");
        }

        // Download on async stream
        let d_out_ptr = *self.async_bufs.d_node_output.device_ptr();
        unsafe {
            result::memcpy_dtoh_async(
                &mut self.async_bufs.h_node_output[..n * 32],
                d_out_ptr,
                stream.stream,
            )
            .expect("GPU async output download failed");
        }

        GpuPending {
            device: &self.device,
            stream: &self.async_stream,
            async_bufs: &self.async_bufs,
            count: n,
        }
    }

    /// Fused active-bits hash: computes L2, L3, and twig_root in a single GPU dispatch.
    ///
    /// Per twig, performs 4 SHA256 ops in one kernel launch:
    ///   L2[0] = SHA256(9 || L1[0] || L1[1])
    ///   L2[1] = SHA256(9 || L1[2] || L1[3])
    ///   L3    = SHA256(10 || L2[0] || L2[1])
    ///   top   = SHA256(11 || left_root || L3)
    ///
    /// Returns `(twig_roots, l2_values, l3_values)` — all N-element Vecs of [u8; 32].
    /// `l2_values` is 2*N elements (l2[0], l2[1] interleaved per twig).
    pub fn batch_active_bits_fused(
        &self,
        l1_values: &[[u8; 32]],
        left_roots: &[[u8; 32]],
    ) -> (Vec<[u8; 32]>, Vec<[u8; 32]>, Vec<[u8; 32]>) {
        let n = left_roots.len();
        assert_eq!(
            l1_values.len(),
            n * 4,
            "l1_values must have 4 entries per twig"
        );
        if n == 0 {
            return (Vec::new(), Vec::new(), Vec::new());
        }
        assert!(
            n <= self.max_batch_size,
            "batch size {} exceeds max {}",
            n,
            self.max_batch_size
        );

        // Flatten inputs into contiguous byte buffers
        let h_l1: Vec<u8> = bytemuck::cast_slice(l1_values).to_vec();
        let h_lr: Vec<u8> = bytemuck::cast_slice(left_roots).to_vec();

        // Upload inputs to GPU
        let d_l1 = self
            .device
            .htod_copy(h_l1)
            .expect("GPU fused l1 upload failed");
        let d_lr = self
            .device
            .htod_copy(h_lr)
            .expect("GPU fused left_roots upload failed");

        // Allocate output device buffers
        let d_twig_roots: CudaSlice<u8> = self
            .device
            .alloc_zeros(n * 32)
            .expect("GPU fused twig_roots alloc failed");
        let d_l2: CudaSlice<u8> = self
            .device
            .alloc_zeros(n * 64)
            .expect("GPU fused l2 alloc failed");
        let d_l3: CudaSlice<u8> = self
            .device
            .alloc_zeros(n * 32)
            .expect("GPU fused l3 alloc failed");

        let grid = ((n as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);
        let cfg = LaunchConfig {
            grid_dim: grid,
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.fused_active_bits_fn
                .clone()
                .launch(
                    cfg,
                    (&d_l1, &d_lr, &d_twig_roots, &d_l2, &d_l3, n as u32),
                )
                .expect("GPU fused active bits kernel launch failed");
        }

        // Download results
        let mut h_twig_roots = vec![0u8; n * 32];
        let mut h_l2 = vec![0u8; n * 64];
        let mut h_l3 = vec![0u8; n * 32];

        let stream = *self.device.cu_stream();
        unsafe {
            result::memcpy_dtoh_async(
                &mut h_twig_roots,
                *d_twig_roots.device_ptr(),
                stream,
            )
            .expect("GPU fused twig_roots download failed");
            result::memcpy_dtoh_async(&mut h_l2, *d_l2.device_ptr(), stream)
                .expect("GPU fused l2 download failed");
            result::memcpy_dtoh_async(&mut h_l3, *d_l3.device_ptr(), stream)
                .expect("GPU fused l3 download failed");
        }

        self.device.synchronize().expect("GPU sync failed");

        let twig_roots: Vec<[u8; 32]> =
            bytemuck::cast_slice(&h_twig_roots).to_vec();
        let l2_out: Vec<[u8; 32]> = bytemuck::cast_slice(&h_l2).to_vec();
        let l3_out: Vec<[u8; 32]> = bytemuck::cast_slice(&h_l3).to_vec();

        (twig_roots, l2_out, l3_out)
    }

    /// Adaptive batch node hash: selects the optimal kernel based on batch size.
    ///
    /// - <256 jobs: CPU batch (GPU launch overhead dominates)
    /// - 256..=1024 jobs: AoS GPU kernel (good for moderate batches)
    /// - >1024 jobs: SoA GPU kernel (coalesced reads win at scale)
    ///
    /// Returns the same results regardless of which path is taken.
    pub fn auto_batch_node_hash(
        &self,
        jobs: &[NodeHashJob],
    ) -> Vec<[u8; 32]> {
        let n = jobs.len();
        if n == 0 {
            return Vec::new();
        }

        const CPU_THRESHOLD: usize = 256;
        const SOA_THRESHOLD: usize = 1024;

        if n < CPU_THRESHOLD {
            let mut levels = Vec::with_capacity(n);
            let mut lefts = Vec::with_capacity(n);
            let mut rights = Vec::with_capacity(n);
            for job in jobs {
                levels.push(job.level);
                lefts.push(job.left);
                rights.push(job.right);
            }
            let mut out = vec![[0u8; 32]; n];
            crate::utils::hasher::batch_node_hash_cpu(
                &levels, &lefts, &rights, &mut out,
            );
            out
        } else if n <= SOA_THRESHOLD {
            self.batch_node_hash(jobs)
        } else {
            let mut levels = Vec::with_capacity(n);
            let mut lefts = Vec::with_capacity(n);
            let mut rights = Vec::with_capacity(n);
            for job in jobs {
                levels.push(job.level);
                lefts.push(job.left);
                rights.push(job.right);
            }
            self.batch_node_hash_soa(&levels, &lefts, &rights)
        }
    }

    /// Adaptive batch node hash writing directly into caller's buffer.
    ///
    /// Same kernel selection as `auto_batch_node_hash` but avoids the
    /// intermediate Vec allocation when the caller already has an output buffer.
    pub fn auto_batch_node_hash_into(
        &self,
        jobs: &[NodeHashJob],
        out: &mut [[u8; 32]],
    ) {
        let n = jobs.len();
        assert_eq!(n, out.len(), "jobs and output length mismatch");
        if n == 0 {
            return;
        }

        const CPU_THRESHOLD: usize = 256;
        const SOA_THRESHOLD: usize = 1024;

        if n < CPU_THRESHOLD {
            let mut levels = Vec::with_capacity(n);
            let mut lefts = Vec::with_capacity(n);
            let mut rights = Vec::with_capacity(n);
            for job in jobs {
                levels.push(job.level);
                lefts.push(job.left);
                rights.push(job.right);
            }
            crate::utils::hasher::batch_node_hash_cpu(
                &levels, &lefts, &rights, out,
            );
        } else if n <= SOA_THRESHOLD {
            self.batch_node_hash_into(jobs, out);
        } else {
            let mut levels = Vec::with_capacity(n);
            let mut lefts = Vec::with_capacity(n);
            let mut rights = Vec::with_capacity(n);
            for job in jobs {
                levels.push(job.level);
                lefts.push(job.left);
                rights.push(job.right);
            }
            self.batch_node_hash_soa_into(
                &levels, &lefts, &rights, out,
            );
        }
    }
}

/// Multi-GPU hasher that distributes work across all available CUDA devices.
/// Uses round-robin assignment: shard N goes to GPU (N % num_gpus).
/// Falls back to single-GPU if only one device is available.
pub struct MultiGpuHasher {
    hashers: Vec<GpuHasher>,
}

impl MultiGpuHasher {
    /// Initialize a MultiGpuHasher across all available CUDA devices.
    /// Each GPU gets its own pre-allocated buffers and CUDA stream.
    pub fn new(max_batch_size: usize) -> Result<Self, String> {
        let count = GpuHasher::device_count()?;
        if count <= 0 {
            return Err("No CUDA devices available".to_string());
        }
        let mut hashers = Vec::with_capacity(count as usize);
        for i in 0..count as usize {
            hashers.push(GpuHasher::new_on_device(i, max_batch_size)?);
        }
        Ok(Self { hashers })
    }

    /// Number of GPUs being used.
    pub fn gpu_count(&self) -> usize {
        self.hashers.len()
    }

    /// Get the GpuHasher for a given shard (round-robin assignment).
    pub fn for_shard(&self, shard_id: usize) -> &GpuHasher {
        &self.hashers[shard_id % self.hashers.len()]
    }

    /// Get the GpuHasher for GPU device index directly.
    pub fn device(&self, ordinal: usize) -> &GpuHasher {
        &self.hashers[ordinal]
    }

    /// Batch-hash on a specific GPU (selected by shard_id).
    pub fn batch_node_hash(&self, shard_id: usize, jobs: &[NodeHashJob]) -> Vec<[u8; 32]> {
        self.for_shard(shard_id).batch_node_hash(jobs)
    }

    /// Batch-hash variable-length inputs on a specific GPU (selected by shard_id).
    pub fn batch_hash_variable(&self, shard_id: usize, inputs: &[&[u8]]) -> Vec<[u8; 32]> {
        self.for_shard(shard_id).batch_hash_variable(inputs)
    }

    /// Batch-hash using SoA layout on a specific GPU (selected by shard_id).
    pub fn batch_node_hash_soa(
        &self,
        shard_id: usize,
        levels: &[u8],
        lefts: &[[u8; 32]],
        rights: &[[u8; 32]],
    ) -> Vec<[u8; 32]> {
        self.for_shard(shard_id)
            .batch_node_hash_soa(levels, lefts, rights)
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

    // ========== ME-2: Fused Active Bits Kernel Tests ==========

    #[test]
    fn test_fused_active_bits_single_twig() {
        let gpu = match GpuHasher::new(1000) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("Skipping GPU test (no CUDA device): {}", e);
                return;
            }
        };

        let l1: [[u8; 32]; 4] = [
            [0x11; 32],
            [0x22; 32],
            [0x33; 32],
            [0x44; 32],
        ];
        let left_root = [0xAA; 32];

        let (twig_roots, l2_values, l3_values) =
            gpu.batch_active_bits_fused(&l1, &[left_root]);

        // CPU reference chain
        let exp_l2_0 = cpu_hash2(9, &l1[0], &l1[1]);
        let exp_l2_1 = cpu_hash2(9, &l1[2], &l1[3]);
        let exp_l3 = cpu_hash2(10, &exp_l2_0, &exp_l2_1);
        let exp_top = cpu_hash2(11, &left_root, &exp_l3);

        assert_eq!(l2_values.len(), 2);
        assert_eq!(l3_values.len(), 1);
        assert_eq!(twig_roots.len(), 1);

        assert_eq!(
            l2_values[0], exp_l2_0,
            "L2[0] mismatch: GPU={} CPU={}",
            hex::encode(l2_values[0]),
            hex::encode(exp_l2_0)
        );
        assert_eq!(
            l2_values[1], exp_l2_1,
            "L2[1] mismatch: GPU={} CPU={}",
            hex::encode(l2_values[1]),
            hex::encode(exp_l2_1)
        );
        assert_eq!(
            l3_values[0], exp_l3,
            "L3 mismatch: GPU={} CPU={}",
            hex::encode(l3_values[0]),
            hex::encode(exp_l3)
        );
        assert_eq!(
            twig_roots[0], exp_top,
            "twig_root mismatch: GPU={} CPU={}",
            hex::encode(twig_roots[0]),
            hex::encode(exp_top)
        );
    }

    #[test]
    fn test_fused_active_bits_multi_twig() {
        let gpu = match GpuHasher::new(10000) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("Skipping GPU test (no CUDA device): {}", e);
                return;
            }
        };

        let n = 100;
        let mut l1_all = Vec::with_capacity(n * 4);
        let mut left_roots = Vec::with_capacity(n);

        for i in 0..n {
            for j in 0..4 {
                let mut h = [0u8; 32];
                for k in 0..32 {
                    h[k] = ((i * 4 + j) * 7 + k * 13) as u8;
                }
                l1_all.push(h);
            }
            let mut lr = [0u8; 32];
            for k in 0..32 {
                lr[k] = (i * 11 + k * 3) as u8;
            }
            left_roots.push(lr);
        }

        let (twig_roots, l2_values, l3_values) =
            gpu.batch_active_bits_fused(&l1_all, &left_roots);

        assert_eq!(twig_roots.len(), n);
        assert_eq!(l2_values.len(), n * 2);
        assert_eq!(l3_values.len(), n);

        for i in 0..n {
            let l1_0 = &l1_all[i * 4];
            let l1_1 = &l1_all[i * 4 + 1];
            let l1_2 = &l1_all[i * 4 + 2];
            let l1_3 = &l1_all[i * 4 + 3];
            let lr = &left_roots[i];

            let exp_l2_0 = cpu_hash2(9, l1_0, l1_1);
            let exp_l2_1 = cpu_hash2(9, l1_2, l1_3);
            let exp_l3 = cpu_hash2(10, &exp_l2_0, &exp_l2_1);
            let exp_top = cpu_hash2(11, lr, &exp_l3);

            assert_eq!(l2_values[i * 2], exp_l2_0, "L2[0] mismatch at twig {}", i);
            assert_eq!(l2_values[i * 2 + 1], exp_l2_1, "L2[1] mismatch at twig {}", i);
            assert_eq!(l3_values[i], exp_l3, "L3 mismatch at twig {}", i);
            assert_eq!(twig_roots[i], exp_top, "twig_root mismatch at twig {}", i);
        }
    }

    #[test]
    fn test_fused_active_bits_edge_values() {
        let gpu = match GpuHasher::new(1000) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("Skipping GPU test (no CUDA device): {}", e);
                return;
            }
        };

        // All-zero inputs
        let l1_zero = [[0u8; 32]; 4];
        let lr_zero = [0u8; 32];
        let (roots_z, l2_z, l3_z) =
            gpu.batch_active_bits_fused(&l1_zero, &[lr_zero]);

        let exp_l2_0 = cpu_hash2(9, &[0u8; 32], &[0u8; 32]);
        let exp_l2_1 = cpu_hash2(9, &[0u8; 32], &[0u8; 32]);
        let exp_l3 = cpu_hash2(10, &exp_l2_0, &exp_l2_1);
        let exp_top = cpu_hash2(11, &lr_zero, &exp_l3);

        assert_eq!(l2_z[0], exp_l2_0);
        assert_eq!(l2_z[1], exp_l2_1);
        assert_eq!(l3_z[0], exp_l3);
        assert_eq!(roots_z[0], exp_top);

        // All-0xFF inputs
        let l1_ff = [[0xFF_u8; 32]; 4];
        let lr_ff = [0xFF_u8; 32];
        let (roots_f, l2_f, l3_f) =
            gpu.batch_active_bits_fused(&l1_ff, &[lr_ff]);

        let exp_l2_0 = cpu_hash2(9, &[0xFF; 32], &[0xFF; 32]);
        let exp_l2_1 = cpu_hash2(9, &[0xFF; 32], &[0xFF; 32]);
        let exp_l3 = cpu_hash2(10, &exp_l2_0, &exp_l2_1);
        let exp_top = cpu_hash2(11, &lr_ff, &exp_l3);

        assert_eq!(l2_f[0], exp_l2_0);
        assert_eq!(l2_f[1], exp_l2_1);
        assert_eq!(l3_f[0], exp_l3);
        assert_eq!(roots_f[0], exp_top);
    }

    #[test]
    fn test_fused_active_bits_empty() {
        let gpu = match GpuHasher::new(1000) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("Skipping GPU test (no CUDA device): {}", e);
                return;
            }
        };

        let (roots, l2, l3) =
            gpu.batch_active_bits_fused(&[], &[]);

        assert!(roots.is_empty());
        assert!(l2.is_empty());
        assert!(l3.is_empty());
    }

    #[test]
    fn test_fused_active_bits_determinism() {
        let gpu = match GpuHasher::new(1000) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("Skipping GPU test (no CUDA device): {}", e);
                return;
            }
        };

        let l1: Vec<[u8; 32]> = (0..4)
            .map(|i| {
                let mut h = [0u8; 32];
                for k in 0..32 {
                    h[k] = (i * 37 + k as u8 * 11) as u8;
                }
                h
            })
            .collect();
        let left_root = [0xBB; 32];

        let first = gpu.batch_active_bits_fused(&l1, &[left_root]);

        for run in 1..5 {
            let result = gpu.batch_active_bits_fused(&l1, &[left_root]);
            assert_eq!(first.0, result.0, "twig_roots differ on run {}", run);
            assert_eq!(first.1, result.1, "l2_values differ on run {}", run);
            assert_eq!(first.2, result.2, "l3_values differ on run {}", run);
        }
    }

    // ========== ME-3: Adaptive Kernel Selection Tests ==========

    #[test]
    fn test_auto_batch_all_paths_equivalent() {
        let gpu = match GpuHasher::new(100000) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("Skipping GPU test (no CUDA device): {}", e);
                return;
            }
        };

        let n = 2000;
        let mut jobs = Vec::with_capacity(n);
        for i in 0..n {
            let level = (i % 12) as u8;
            let mut left = [0u8; 32];
            let mut right = [0u8; 32];
            for j in 0..32 {
                left[j] = ((i * 7 + j * 13) & 0xFF) as u8;
                right[j] = ((i * 11 + j * 17) & 0xFF) as u8;
            }
            jobs.push(NodeHashJob { level, left, right });
        }

        // CPU reference
        let mut cpu_out = vec![[0u8; 32]; n];
        let levels: Vec<u8> = jobs.iter().map(|j| j.level).collect();
        let lefts: Vec<[u8; 32]> = jobs.iter().map(|j| j.left).collect();
        let rights: Vec<[u8; 32]> = jobs.iter().map(|j| j.right).collect();
        crate::utils::hasher::batch_node_hash_cpu(
            &levels, &lefts, &rights, &mut cpu_out,
        );

        // auto_batch (picks SoA for n=2000 > 1024)
        let auto_out = gpu.auto_batch_node_hash(&jobs);
        assert_eq!(auto_out.len(), n);

        // AoS GPU
        let aos_out = gpu.batch_node_hash(&jobs);
        assert_eq!(aos_out.len(), n);

        // SoA GPU
        let soa_out = gpu.batch_node_hash_soa(&levels, &lefts, &rights);
        assert_eq!(soa_out.len(), n);

        for i in 0..n {
            assert_eq!(
                auto_out[i], cpu_out[i],
                "auto vs CPU mismatch at {}: auto={} cpu={}",
                i,
                hex::encode(auto_out[i]),
                hex::encode(cpu_out[i])
            );
            assert_eq!(
                aos_out[i], cpu_out[i],
                "AoS vs CPU mismatch at {}", i
            );
            assert_eq!(
                soa_out[i], cpu_out[i],
                "SoA vs CPU mismatch at {}", i
            );
        }
    }

    #[test]
    fn test_auto_batch_threshold_boundaries() {
        let gpu = match GpuHasher::new(10000) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("Skipping GPU test (no CUDA device): {}", e);
                return;
            }
        };

        let boundary_sizes = [1, 255, 256, 1024, 1025];

        for &size in &boundary_sizes {
            let mut jobs = Vec::with_capacity(size);
            for i in 0..size {
                let level = (i % 12) as u8;
                let mut left = [0u8; 32];
                let mut right = [0u8; 32];
                for j in 0..32 {
                    left[j] = ((i * 7 + j * 13) & 0xFF) as u8;
                    right[j] = ((i * 11 + j * 17) & 0xFF) as u8;
                }
                jobs.push(NodeHashJob { level, left, right });
            }

            let auto_out = gpu.auto_batch_node_hash(&jobs);
            assert_eq!(auto_out.len(), size, "wrong output len for size={}", size);

            // CPU reference
            for (i, job) in jobs.iter().enumerate() {
                let expected = cpu_hash2(job.level, &job.left, &job.right);
                assert_eq!(
                    auto_out[i], expected,
                    "mismatch at i={} for batch size={}", i, size
                );
            }
        }
    }

    #[test]
    fn test_auto_batch_into_matches_alloc() {
        let gpu = match GpuHasher::new(10000) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("Skipping GPU test (no CUDA device): {}", e);
                return;
            }
        };

        let n = 500;
        let mut jobs = Vec::with_capacity(n);
        for i in 0..n {
            let level = (i % 12) as u8;
            let mut left = [0u8; 32];
            let mut right = [0u8; 32];
            for j in 0..32 {
                left[j] = ((i * 3 + j * 19) & 0xFF) as u8;
                right[j] = ((i * 5 + j * 23) & 0xFF) as u8;
            }
            jobs.push(NodeHashJob { level, left, right });
        }

        let alloc_out = gpu.auto_batch_node_hash(&jobs);
        let mut into_out = vec![[0u8; 32]; n];
        gpu.auto_batch_node_hash_into(&jobs, &mut into_out);

        assert_eq!(alloc_out, into_out, "auto_batch_node_hash vs into mismatch");
    }

    // ========== ME-1: Multi-Stream Async Dispatch Tests ==========

    #[test]
    fn test_async_matches_sync() {
        let mut gpu = match GpuHasher::new(10000) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("Skipping GPU test (no CUDA device): {}", e);
                return;
            }
        };

        let n = 1000;
        let mut jobs = Vec::with_capacity(n);
        for i in 0..n {
            let level = (i % 12) as u8;
            let mut left = [0u8; 32];
            let mut right = [0u8; 32];
            for j in 0..32 {
                left[j] = ((i * 7 + j * 13) & 0xFF) as u8;
                right[j] = ((i * 11 + j * 17) & 0xFF) as u8;
            }
            jobs.push(NodeHashJob { level, left, right });
        }

        let sync_out = gpu.batch_node_hash(&jobs);
        let async_out = gpu.batch_node_hash_async(&jobs).wait();

        assert_eq!(sync_out.len(), async_out.len());
        for i in 0..n {
            assert_eq!(
                sync_out[i], async_out[i],
                "sync vs async mismatch at {}", i
            );
        }
    }

    #[test]
    fn test_async_wait_into() {
        let mut gpu = match GpuHasher::new(10000) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("Skipping GPU test (no CUDA device): {}", e);
                return;
            }
        };

        let n = 500;
        let mut jobs = Vec::with_capacity(n);
        for i in 0..n {
            let level = (i % 12) as u8;
            let mut left = [0u8; 32];
            let mut right = [0u8; 32];
            for j in 0..32 {
                left[j] = ((i * 5 + j * 11) & 0xFF) as u8;
                right[j] = ((i * 13 + j * 7) & 0xFF) as u8;
            }
            jobs.push(NodeHashJob { level, left, right });
        }

        let expected = gpu.batch_node_hash(&jobs);

        let mut buf = vec![[0u8; 32]; n];
        gpu.batch_node_hash_async(&jobs).wait_into(&mut buf);

        assert_eq!(buf, expected, "wait_into results differ from sync");
    }

    #[test]
    fn test_async_sequential() {
        let mut gpu = match GpuHasher::new(10000) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("Skipping GPU test (no CUDA device): {}", e);
                return;
            }
        };

        let mut jobs_a = Vec::with_capacity(300);
        let mut jobs_b = Vec::with_capacity(400);

        for i in 0..300 {
            let level = (i % 8) as u8;
            let left = [i as u8; 32];
            let right = [(i + 1) as u8; 32];
            jobs_a.push(NodeHashJob { level, left, right });
        }
        for i in 0..400 {
            let level = ((i + 3) % 10) as u8;
            let left = [(i * 2) as u8; 32];
            let right = [(i * 3) as u8; 32];
            jobs_b.push(NodeHashJob { level, left, right });
        }

        // Sync references
        let expected_a = gpu.batch_node_hash(&jobs_a);
        let expected_b = gpu.batch_node_hash(&jobs_b);

        // Sequential async: dispatch A, wait, dispatch B, wait
        let result_a = gpu.batch_node_hash_async(&jobs_a).wait();
        let result_b = gpu.batch_node_hash_async(&jobs_b).wait();

        assert_eq!(result_a, expected_a, "async sequential A mismatch");
        assert_eq!(result_b, expected_b, "async sequential B mismatch");
    }
}
