// CUDA SHA256 batch kernel for QMDB Merkle tree hashing
// Optimized for two workloads:
//   1. Fixed 65-byte node hashing: SHA256(level_byte || left_32B || right_32B)
//   2. Variable-length entry hashing: SHA256(entry_payload)

// Inline typedefs instead of #include <stdint.h> to avoid NVRTC
// needing glibc system headers (bits/, gnu/stubs-32.h, etc.)
typedef unsigned char      uint8_t;
typedef unsigned int        uint32_t;
typedef unsigned long long  uint64_t;

// SHA256 round constants
__constant__ uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

// SHA256 initial hash values
__constant__ uint32_t H_INIT[8] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};

__device__ __forceinline__ uint32_t rotr(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

__device__ __forceinline__ uint32_t ch(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (~x & z);
}

__device__ __forceinline__ uint32_t maj(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

__device__ __forceinline__ uint32_t sigma0(uint32_t x) {
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
}

__device__ __forceinline__ uint32_t sigma1(uint32_t x) {
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
}

__device__ __forceinline__ uint32_t gamma0(uint32_t x) {
    return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
}

__device__ __forceinline__ uint32_t gamma1(uint32_t x) {
    return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10);
}

// Load a big-endian uint32 from a byte pointer
__device__ __forceinline__ uint32_t load_be32(const uint8_t* p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] << 8)  | ((uint32_t)p[3]);
}

// Store a big-endian uint32 to a byte pointer
__device__ __forceinline__ void store_be32(uint8_t* p, uint32_t v) {
    p[0] = (uint8_t)(v >> 24);
    p[1] = (uint8_t)(v >> 16);
    p[2] = (uint8_t)(v >> 8);
    p[3] = (uint8_t)(v);
}

// SHA256 compression function for a single 64-byte block
__device__ void sha256_compress(uint32_t state[8], const uint8_t block[64]) {
    uint32_t W[64];

    // Load message schedule from block (big-endian)
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        W[i] = load_be32(&block[i * 4]);
    }

    // Extend message schedule
    #pragma unroll
    for (int i = 16; i < 64; i++) {
        W[i] = gamma1(W[i-2]) + W[i-7] + gamma0(W[i-15]) + W[i-16];
    }

    // Working variables
    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];

    // 64 rounds
    #pragma unroll
    for (int i = 0; i < 64; i++) {
        uint32_t T1 = h + sigma1(e) + ch(e, f, g) + K[i] + W[i];
        uint32_t T2 = sigma0(a) + maj(a, b, c);
        h = g; g = f; f = e; e = d + T1;
        d = c; c = b; b = a; a = T1 + T2;
    }

    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

// Device function: SHA256 of exactly 65 bytes (level || left || right).
// Writes 32-byte hash to `output`.
__device__ void sha256_65b(uint8_t level, const uint8_t* left, const uint8_t* right, uint8_t* output) {
    uint32_t state[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) state[i] = H_INIT[i];

    // Block 1: [level(1) | left(32) | right[0..31](31)] = 64 bytes
    uint8_t block1[64];
    block1[0] = level;
    #pragma unroll
    for (int i = 0; i < 32; i++) block1[1 + i] = left[i];
    #pragma unroll
    for (int i = 0; i < 31; i++) block1[33 + i] = right[i];
    sha256_compress(state, block1);

    // Block 2: right[31] + padding + length(520 bits = 0x208)
    uint8_t block2[64];
    block2[0] = right[31];
    block2[1] = 0x80;
    #pragma unroll
    for (int i = 2; i < 56; i++) block2[i] = 0;
    block2[56] = 0; block2[57] = 0; block2[58] = 0; block2[59] = 0;
    block2[60] = 0; block2[61] = 0; block2[62] = 0x02; block2[63] = 0x08;
    sha256_compress(state, block2);

    #pragma unroll
    for (int i = 0; i < 8; i++) store_be32(&output[i * 4], state[i]);
}

// ============================================================
// Kernel 1: Fixed 65-byte node hash
// Input:  jobs[N * 65] = N entries of [level(1B) | left(32B) | right(32B)]
// Output: out[N * 32]  = N SHA256 hashes
// ============================================================
extern "C" __global__ void sha256_node_hash(
    const uint8_t* __restrict__ jobs,
    uint8_t* __restrict__ out,
    uint32_t count
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    const uint8_t* input = jobs + idx * 65;
    uint8_t* output = out + idx * 32;

    // SHA256 of 65 bytes requires 2 compression blocks:
    //   Block 1: bytes 0-63 (first 64 bytes of input)
    //   Block 2: byte 64, then 0x80 padding, then length

    uint32_t state[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) state[i] = H_INIT[i];

    // Block 1: first 64 bytes of the 65-byte input
    sha256_compress(state, input);

    // Block 2: last 1 byte + padding + length
    uint8_t block2[64];
    block2[0] = input[64];           // the 65th byte
    block2[1] = 0x80;                // padding start
    // Zero fill bytes 2..55
    #pragma unroll
    for (int i = 2; i < 56; i++) block2[i] = 0;
    // Length in bits = 65 * 8 = 520 = 0x208, stored as big-endian 64-bit at end
    block2[56] = 0; block2[57] = 0; block2[58] = 0; block2[59] = 0;
    block2[60] = 0; block2[61] = 0; block2[62] = 0x02; block2[63] = 0x08;

    sha256_compress(state, block2);

    // Write output (big-endian)
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        store_be32(&output[i * 4], state[i]);
    }
}

// ============================================================
// Kernel 2: Variable-length hash
// Input:  data[]        = flat byte buffer with all entries concatenated
//         offsets[N]    = start offset of each entry in data[]
//         lengths[N]    = byte length of each entry
// Output: out[N * 32]   = N SHA256 hashes
// ============================================================
extern "C" __global__ void sha256_variable_hash(
    const uint8_t* __restrict__ data,
    const uint32_t* __restrict__ offsets,
    const uint32_t* __restrict__ lengths,
    uint8_t* __restrict__ out,
    uint32_t count
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    const uint8_t* input = data + offsets[idx];
    uint32_t len = lengths[idx];
    uint8_t* output = out + idx * 32;

    uint32_t state[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) state[i] = H_INIT[i];

    // Process full 64-byte blocks
    uint32_t full_blocks = len / 64;
    for (uint32_t b = 0; b < full_blocks; b++) {
        sha256_compress(state, input + b * 64);
    }

    // Final block(s) with padding
    uint32_t remaining = len - full_blocks * 64;
    uint8_t final_block[64];

    // Copy remaining bytes
    for (uint32_t i = 0; i < remaining; i++) {
        final_block[i] = input[full_blocks * 64 + i];
    }

    // Add padding bit
    final_block[remaining] = 0x80;

    if (remaining < 56) {
        // Padding and length fit in one block
        for (uint32_t i = remaining + 1; i < 56; i++) final_block[i] = 0;
        // Length in bits (big-endian 64-bit)
        uint64_t bit_len = (uint64_t)len * 8;
        final_block[56] = (uint8_t)(bit_len >> 56);
        final_block[57] = (uint8_t)(bit_len >> 48);
        final_block[58] = (uint8_t)(bit_len >> 40);
        final_block[59] = (uint8_t)(bit_len >> 32);
        final_block[60] = (uint8_t)(bit_len >> 24);
        final_block[61] = (uint8_t)(bit_len >> 16);
        final_block[62] = (uint8_t)(bit_len >> 8);
        final_block[63] = (uint8_t)(bit_len);
        sha256_compress(state, final_block);
    } else {
        // Need two blocks for padding
        for (uint32_t i = remaining + 1; i < 64; i++) final_block[i] = 0;
        sha256_compress(state, final_block);

        // Second padding block: zeros + length
        uint8_t len_block[64];
        for (int i = 0; i < 56; i++) len_block[i] = 0;
        uint64_t bit_len = (uint64_t)len * 8;
        len_block[56] = (uint8_t)(bit_len >> 56);
        len_block[57] = (uint8_t)(bit_len >> 48);
        len_block[58] = (uint8_t)(bit_len >> 40);
        len_block[59] = (uint8_t)(bit_len >> 32);
        len_block[60] = (uint8_t)(bit_len >> 24);
        len_block[61] = (uint8_t)(bit_len >> 16);
        len_block[62] = (uint8_t)(bit_len >> 8);
        len_block[63] = (uint8_t)(bit_len);
        sha256_compress(state, len_block);
    }

    // Write output (big-endian)
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        store_be32(&output[i * 4], state[i]);
    }
}

// ============================================================
// Kernel 3: Warp-cooperative fixed 65-byte node hash
//
// 8 threads cooperate on one SHA256 hash. Each thread "owns" one
// of the 8 state words (a-h). Warp shuffles rotate state between
// lanes without shared memory, improving occupancy.
//
// Launch with: blockDim.x must be a multiple of 8.
// Each group of 8 consecutive threads processes one hash job.
// Grid: (ceil(count * 8 / blockDim.x), 1, 1)
// ============================================================
extern "C" __global__ void sha256_node_hash_warp_coop(
    const uint8_t* __restrict__ jobs,
    uint8_t* __restrict__ out,
    uint32_t count
) {
    uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t job_idx = global_tid / 8;
    uint32_t lane = global_tid % 8; // 0=a, 1=b, ..., 7=h

    if (job_idx >= count) return;

    const uint8_t* input = jobs + job_idx * 65;
    uint8_t* output = out + job_idx * 32;

    // Mask for our 8-thread team within the warp
    uint32_t team_base = (threadIdx.x / 8) * 8;
    uint32_t team_mask = 0xFFu << (team_base % 32);

    // Each thread holds one state word
    uint32_t my_state = H_INIT[lane];

    // --- Block 1: first 64 bytes ---
    uint32_t W[64];
    for (int i = 0; i < 16; i++) {
        W[i] = load_be32(&input[i * 4]);
    }
    for (int i = 16; i < 64; i++) {
        W[i] = gamma1(W[i-2]) + W[i-7] + gamma0(W[i-15]) + W[i-16];
    }

    for (int i = 0; i < 64; i++) {
        uint32_t val_a = __shfl_sync(team_mask, my_state, team_base + 0);
        uint32_t val_b = __shfl_sync(team_mask, my_state, team_base + 1);
        uint32_t val_c = __shfl_sync(team_mask, my_state, team_base + 2);
        uint32_t val_d = __shfl_sync(team_mask, my_state, team_base + 3);
        uint32_t val_e = __shfl_sync(team_mask, my_state, team_base + 4);
        uint32_t val_f = __shfl_sync(team_mask, my_state, team_base + 5);
        uint32_t val_g = __shfl_sync(team_mask, my_state, team_base + 6);
        uint32_t val_h = __shfl_sync(team_mask, my_state, team_base + 7);

        uint32_t T1 = val_h + sigma1(val_e) + ch(val_e, val_f, val_g) + K[i] + W[i];
        uint32_t T2 = sigma0(val_a) + maj(val_a, val_b, val_c);

        switch (lane) {
            case 0: my_state = T1 + T2; break;
            case 1: my_state = val_a; break;
            case 2: my_state = val_b; break;
            case 3: my_state = val_c; break;
            case 4: my_state = val_d + T1; break;
            case 5: my_state = val_e; break;
            case 6: my_state = val_f; break;
            case 7: my_state = val_g; break;
        }
    }

    my_state += H_INIT[lane];

    // --- Block 2: byte 64 + padding + length ---
    uint8_t block2[64];
    block2[0] = input[64];
    block2[1] = 0x80;
    #pragma unroll
    for (int i = 2; i < 56; i++) block2[i] = 0;
    block2[56] = 0; block2[57] = 0; block2[58] = 0; block2[59] = 0;
    block2[60] = 0; block2[61] = 0; block2[62] = 0x02; block2[63] = 0x08;

    for (int i = 0; i < 16; i++) {
        W[i] = load_be32(&block2[i * 4]);
    }
    for (int i = 16; i < 64; i++) {
        W[i] = gamma1(W[i-2]) + W[i-7] + gamma0(W[i-15]) + W[i-16];
    }

    uint32_t pre_state = my_state;

    for (int i = 0; i < 64; i++) {
        uint32_t val_a = __shfl_sync(team_mask, my_state, team_base + 0);
        uint32_t val_b = __shfl_sync(team_mask, my_state, team_base + 1);
        uint32_t val_c = __shfl_sync(team_mask, my_state, team_base + 2);
        uint32_t val_d = __shfl_sync(team_mask, my_state, team_base + 3);
        uint32_t val_e = __shfl_sync(team_mask, my_state, team_base + 4);
        uint32_t val_f = __shfl_sync(team_mask, my_state, team_base + 5);
        uint32_t val_g = __shfl_sync(team_mask, my_state, team_base + 6);
        uint32_t val_h = __shfl_sync(team_mask, my_state, team_base + 7);

        uint32_t T1 = val_h + sigma1(val_e) + ch(val_e, val_f, val_g) + K[i] + W[i];
        uint32_t T2 = sigma0(val_a) + maj(val_a, val_b, val_c);

        switch (lane) {
            case 0: my_state = T1 + T2; break;
            case 1: my_state = val_a; break;
            case 2: my_state = val_b; break;
            case 3: my_state = val_c; break;
            case 4: my_state = val_d + T1; break;
            case 5: my_state = val_e; break;
            case 6: my_state = val_f; break;
            case 7: my_state = val_g; break;
        }
    }

    my_state += pre_state;

    // Each thread writes its 4-byte state word
    store_be32(&output[lane * 4], my_state);
}

// ============================================================
// Kernel 4: Fixed 65-byte node hash with Structure-of-Arrays layout
//
// Instead of AoS: [level0|left0|right0|level1|left1|right1|...]
// Uses SoA:       levels[N], lefts[N*32], rights[N*32]
//
// Adjacent threads read adjacent 32-byte blocks from lefts/rights,
// enabling coalesced global memory reads (32B stride vs 65B stride).
// ============================================================
extern "C" __global__ void sha256_node_hash_soa(
    const uint8_t* __restrict__ levels,
    const uint8_t* __restrict__ lefts,
    const uint8_t* __restrict__ rights,
    uint8_t* __restrict__ out,
    uint32_t count
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    uint8_t* output = out + idx * 32;

    // Assemble the 65-byte message in registers from SoA arrays:
    //   byte 0:     levels[idx]
    //   bytes 1-32: lefts[idx*32 .. idx*32+32]
    //   bytes 33-64: rights[idx*32 .. idx*32+32]
    uint8_t block1[64];
    block1[0] = levels[idx];

    // Coalesced 32-byte read from lefts array (adjacent threads read adjacent 32B blocks)
    const uint8_t* left_ptr = lefts + idx * 32;
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        block1[1 + i] = left_ptr[i];
    }

    // Coalesced 32-byte read from rights array
    const uint8_t* right_ptr = rights + idx * 32;
    #pragma unroll
    for (int i = 0; i < 31; i++) {
        block1[33 + i] = right_ptr[i];
    }

    // SHA256 init
    uint32_t state[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) state[i] = H_INIT[i];

    // Block 1: first 64 bytes = [level(1) | left(32) | right[0..31](31)]
    sha256_compress(state, block1);

    // Block 2: last 1 byte of right + padding + length
    uint8_t block2[64];
    block2[0] = right_ptr[31];          // the 65th byte (right[31])
    block2[1] = 0x80;                   // padding start
    #pragma unroll
    for (int i = 2; i < 56; i++) block2[i] = 0;
    // Length in bits = 65 * 8 = 520 = 0x208
    block2[56] = 0; block2[57] = 0; block2[58] = 0; block2[59] = 0;
    block2[60] = 0; block2[61] = 0; block2[62] = 0x02; block2[63] = 0x08;

    sha256_compress(state, block2);

    // Write output (big-endian)
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        store_be32(&output[i * 4], state[i]);
    }
}

// ============================================================
// Device helper: SHA256 of exactly 65 bytes (level || left || right)
// Computes the hash in registers, writing 32 bytes to `out`.
// Reusable building block for the fused active-bits kernel.
// ============================================================
__device__ void sha256_65b(
    uint8_t level_byte,
    const uint8_t* __restrict__ left,
    const uint8_t* __restrict__ right,
    uint8_t* __restrict__ out
) {
    uint32_t state[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) state[i] = H_INIT[i];

    // Block 1: [level(1) | left(32) | right[0..31](31)] = 64 bytes
    uint8_t block1[64];
    block1[0] = level_byte;
    #pragma unroll
    for (int i = 0; i < 32; i++) block1[1 + i] = left[i];
    #pragma unroll
    for (int i = 0; i < 31; i++) block1[33 + i] = right[i];

    sha256_compress(state, block1);

    // Block 2: right[31] + padding + 65*8=520 bit length
    uint8_t block2_h[64];
    block2_h[0] = right[31];
    block2_h[1] = 0x80;
    #pragma unroll
    for (int i = 2; i < 56; i++) block2_h[i] = 0;
    block2_h[56] = 0; block2_h[57] = 0; block2_h[58] = 0; block2_h[59] = 0;
    block2_h[60] = 0; block2_h[61] = 0; block2_h[62] = 0x02; block2_h[63] = 0x08;

    sha256_compress(state, block2_h);

    #pragma unroll
    for (int i = 0; i < 8; i++) store_be32(&out[i * 4], state[i]);
}

// ============================================================
// Kernel 5: Fused active-bits hashing (L2 -> L3 -> top) per twig
//
// One thread per twig. Computes 4 SHA256 ops:
//   L2[0] = SHA256(9 || L1[0] || L1[1])
//   L2[1] = SHA256(9 || L1[2] || L1[3])
//   L3    = SHA256(10 || L2[0] || L2[1])
//   top   = SHA256(11 || left_root || L3)
//
// Inputs (SoA, N twigs):
//   l1_values:  N * 4 * 32 bytes (4 L1 hashes per twig, contiguous)
//   left_roots: N * 32 bytes
//
// Outputs (SoA, N twigs):
//   twig_roots: N * 32 bytes
//   l2_values:  N * 2 * 32 bytes (for proof generation)
//   l3_values:  N * 32 bytes     (for proof generation)
// ============================================================
extern "C" __global__ void sha256_active_bits_fused(
    const uint8_t* __restrict__ l1_values,
    const uint8_t* __restrict__ left_roots,
    uint8_t* __restrict__ twig_roots,
    uint8_t* __restrict__ l2_values,
    uint8_t* __restrict__ l3_values,
    uint32_t count
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    const uint8_t* l1 = l1_values + idx * 128;  // 4 * 32 bytes
    uint8_t* l2 = l2_values + idx * 64;          // 2 * 32 bytes
    uint8_t* l3 = l3_values + idx * 32;

    // L2[0] = SHA256(9 || L1[0] || L1[1])
    sha256_65b(9, l1, l1 + 32, l2);

    // L2[1] = SHA256(9 || L1[2] || L1[3])
    sha256_65b(9, l1 + 64, l1 + 96, l2 + 32);

    // L3 = SHA256(10 || L2[0] || L2[1])
    sha256_65b(10, l2, l2 + 32, l3);

    // top = SHA256(11 || left_root || L3)
    sha256_65b(11, left_roots + idx * 32, l3, twig_roots + idx * 32);
}
