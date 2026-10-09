//! A fast, non-cryptographic hasher for the model's own indexes.
//!
//! The standard map's SipHash defends against keys chosen to collide. The keys
//! these indexes hold — object-list references, backing ids, channel and
//! ingress numbers — are small integers the guest names, and the only party a
//! colliding guest could slow down is its own session. Against that, SipHash
//! was ~3 % of the drain thread's busy time on a 5K Weather workload (b078:
//! `hash_one` over `ObjectListRef`, `BackingId` and `PipelineId`, plus the
//! hasher's `write`), spent hashing a handful of words per admission and per
//! name lookup.
//!
//! This is the multiply-rotate hash rustc uses for its own tables: one rotate,
//! one xor and one multiply per word. Lookups stay exact — a hash only picks
//! the bucket — so a weaker hash can cost probe length, never an answer.

use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};

const SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;

/// See the module doc.
#[derive(Clone, Copy, Debug, Default)]
pub struct FxHasher {
    hash: u64,
}

impl FxHasher {
    #[inline]
    fn add(&mut self, word: u64) {
        self.hash = (self.hash.rotate_left(5) ^ word).wrapping_mul(SEED);
    }
}

impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut chunks = bytes.chunks_exact(8);
        for chunk in &mut chunks {
            let mut word = [0u8; 8];
            word.copy_from_slice(chunk);
            self.add(u64::from_le_bytes(word));
        }
        let rest = chunks.remainder();
        if !rest.is_empty() {
            let mut word = [0u8; 8];
            word[..rest.len()].copy_from_slice(rest);
            self.add(u64::from_le_bytes(word));
        }
    }

    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.add(u64::from(i));
    }

    #[inline]
    fn write_u16(&mut self, i: u16) {
        self.add(u64::from(i));
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.add(u64::from(i));
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.add(i);
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.add(i as u64);
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }
}

/// The builder the indexes are parameterised on.
pub type FxBuildHasher = BuildHasherDefault<FxHasher>;
/// A [`HashMap`] keyed through [`FxHasher`].
pub type FxHashMap<K, V> = HashMap<K, V, FxBuildHasher>;
/// A [`HashSet`] keyed through [`FxHasher`].
pub type FxHashSet<K> = HashSet<K, FxBuildHasher>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::hash::{BuildHasher, Hash};

    #[test]
    fn equal_keys_hash_equal_and_a_map_finds_what_it_holds() {
        let b = FxBuildHasher::default();
        assert_eq!(b.hash_one((7u32, 9u64)), b.hash_one((7u32, 9u64)));
        let mut m: FxHashMap<(u32, u64), usize> = FxHashMap::default();
        for i in 0..10_000u32 {
            m.insert((i, u64::from(i) * 3), i as usize);
        }
        for i in 0..10_000u32 {
            assert_eq!(m.get(&(i, u64::from(i) * 3)), Some(&(i as usize)));
        }
        assert_eq!(m.get(&(1, 4)), None);
    }

    #[test]
    fn bytes_shorter_than_a_word_still_reach_the_hash() {
        let mut a = FxHasher::default();
        let mut b = FxHasher::default();
        [1u8, 2, 3].hash(&mut a);
        [1u8, 2, 4].hash(&mut b);
        assert_ne!(a.finish(), b.finish());
    }
}
