pub mod activebits;
pub mod bytescache;
pub mod changeset;
pub mod codec;
pub mod hasher;
pub mod numa;
pub mod ringchannel;
pub mod shortlist;
pub mod slice;

use crate::def::{BIG_BUF_SIZE, SHARD_COUNT};

pub type BigBuf = [u8]; // size is BIG_BUF_SIZE

pub fn new_big_buf_boxed() -> Box<[u8]> {
    vec![0u8; BIG_BUF_SIZE].into_boxed_slice()
}

pub fn byte0_to_shard_id(byte0: u8) -> usize {
    (byte0 as usize) * SHARD_COUNT / 256
}

/// Pick a shard id from the first two bytes of a key hash given a runtime
/// shard count. Takes `ceil(log2(shard_count))` high bits from the `u16` big-
/// endian head of `key_hash_head`; non-power-of-2 shard counts fold back via
/// modulo.
///
/// Phase 2.3c contract:
/// - `shard_count >= 1`. `0` panics.
/// - `key_hash_head.len() >= 2`. Shorter slices panic.
/// - For `shard_count == SHARD_COUNT` (i.e., 16) the mapping matches
///   `byte0_to_shard_id(key_hash_head[0])` for every 8-bit prefix — so a
///   caller can migrate from `byte0_to_shard_id` to `byte_range_to_shard_id`
///   without a root-hash change. Proven by `matches_byte0_at_16_shards`.
///
/// # Examples
/// ```
/// # use skippydb::utils::byte_range_to_shard_id;
/// // 32 shards — top 5 bits of byte 0.
/// assert_eq!(byte_range_to_shard_id(32, &[0b00001000, 0]), 1);
/// assert_eq!(byte_range_to_shard_id(32, &[0b11111000, 0]), 31);
/// ```
pub fn byte_range_to_shard_id(shard_count: usize, key_hash_head: &[u8]) -> usize {
    assert!(shard_count >= 1, "shard_count must be >= 1");
    assert!(
        key_hash_head.len() >= 2,
        "byte_range_to_shard_id needs >= 2 bytes of hash head (got {})",
        key_hash_head.len()
    );
    let raw = u16::from_be_bytes([key_hash_head[0], key_hash_head[1]]);
    // Round up to the next power of two so e.g. 17 shards still uses 5 bits.
    // `trailing_zeros` on a power of two gives log2 directly.
    let next_pow2 = shard_count.next_power_of_two();
    let bits = next_pow2.trailing_zeros() as usize;
    let shift = 16usize.saturating_sub(bits);
    let idx = (raw >> shift) as usize;
    if shard_count.is_power_of_two() {
        idx
    } else {
        idx % shard_count
    }
}

#[derive(Debug, PartialEq, Clone, serde::Serialize, serde::Deserialize)]
pub struct OpRecord {
    pub op_type: u8,
    pub num_active: usize,
    pub oldest_active_sn: u64,
    pub shard_id: usize,
    pub next_sn: u64,
    pub key: Vec<u8>,
    pub value: Vec<u8>,
    pub rd_list: Vec<Vec<u8>>,
    pub wr_list: Vec<Vec<u8>>,
    pub dig_list: Vec<Vec<u8>>, //old entries in compaction
    pub put_list: Vec<Vec<u8>>, //new entries in compaction
}

impl OpRecord {
    pub fn new(op_type: u8) -> OpRecord {
        OpRecord {
            op_type,
            num_active: 0,
            oldest_active_sn: 0,
            shard_id: 0,
            next_sn: 0,
            key: Vec::with_capacity(0),
            value: Vec::with_capacity(0),
            rd_list: Vec::with_capacity(2),
            wr_list: Vec::with_capacity(2),
            dig_list: Vec::with_capacity(2),
            put_list: Vec::with_capacity(2),
        }
    }
}

#[cfg(test)]
mod tests {

    use super::codec::*;
    use super::hasher::*;

    #[test]
    fn test_hash2() {
        assert_eq!(
            hex::encode(hash2(8, "hello", "world")),
            "8e6fc50a3f98a3c314021b89688ca83a9b5697ca956e211198625fc460ddf1e9"
        );

        assert_eq!(
            hex::encode(hash2x(8, "world", "hello", true)),
            "8e6fc50a3f98a3c314021b89688ca83a9b5697ca956e211198625fc460ddf1e9"
        );
    }

    #[test]
    fn test_node_hash_inplace() {
        let mut target: [u8; 32] = [0; 32];
        node_hash_inplace(8, &mut target, "hello", "world");
        assert_eq!(
            hex::encode(target),
            "8e6fc50a3f98a3c314021b89688ca83a9b5697ca956e211198625fc460ddf1e9"
        );
    }

    #[test]
    fn test_encode_decode_n64() {
        let v = vec![0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
        assert_eq!(decode_le_i64(&v), -8613303245920329199);
        assert_eq!(decode_le_u64(&v), 0x8877665544332211);
        assert_eq!(encode_le_i64(-8613303245920329199), v);
        assert_eq!(encode_le_u64(0x8877665544332211), v);
    }

    // --- Phase 2.3c byte_range_to_shard_id boundary tests ---

    use super::{byte0_to_shard_id, byte_range_to_shard_id};
    use crate::def::SHARD_COUNT;

    #[test]
    fn byte_range_extracts_top_bits_for_power_of_two_shard_counts() {
        // 16 shards → 4 top bits of byte 0. Exercise each nibble.
        for nibble in 0u8..16 {
            let b0 = nibble << 4;
            assert_eq!(
                byte_range_to_shard_id(16, &[b0, 0]),
                nibble as usize,
                "nibble {:x} under 16-shard split",
                nibble
            );
        }
        // 32 shards → 5 top bits.
        assert_eq!(byte_range_to_shard_id(32, &[0b00000000, 0]), 0);
        assert_eq!(byte_range_to_shard_id(32, &[0b00001000, 0]), 1);
        assert_eq!(byte_range_to_shard_id(32, &[0b11111000, 0]), 31);
        assert_eq!(byte_range_to_shard_id(32, &[0b11111111, 0xff]), 31);
        // 64 shards → 6 top bits.
        assert_eq!(byte_range_to_shard_id(64, &[0b00000100, 0]), 1);
        assert_eq!(byte_range_to_shard_id(64, &[0b11111100, 0]), 63);
    }

    #[test]
    fn matches_byte0_at_16_shards() {
        // Phase 2.3c migration invariant: every 8-bit prefix produces the
        // same shard id under the old and new hashes at SHARD_COUNT=16. If
        // this breaks, a switchover would invalidate on-disk root hashes.
        assert_eq!(SHARD_COUNT, 16, "this migration invariant targets the 16-shard default");
        for b0 in 0u8..=255 {
            let old = byte0_to_shard_id(b0);
            let new_ = byte_range_to_shard_id(16, &[b0, 0]);
            assert_eq!(old, new_, "divergence at byte0 = 0x{:02x}", b0);
        }
    }

    #[test]
    fn non_power_of_two_shard_counts_fold_via_modulo() {
        // 17 shards: 5-bit index space = 0..32, folded mod 17 into 0..17.
        for hi in 0u16..32 {
            let raw = (hi << 11) as u16; // hi 5 bits high-justified
            let bytes = raw.to_be_bytes();
            let id = byte_range_to_shard_id(17, &bytes);
            assert!(id < 17, "shard_id {} out of range for 17 shards", id);
            assert_eq!(id, (hi as usize) % 17);
        }
    }

    #[test]
    #[should_panic(expected = "shard_count must be >= 1")]
    fn panics_on_zero_shard_count() {
        byte_range_to_shard_id(0, &[0, 0]);
    }

    #[test]
    #[should_panic(expected = "needs >= 2 bytes of hash head")]
    fn panics_on_short_hash_slice() {
        byte_range_to_shard_id(16, &[0]);
    }
}
