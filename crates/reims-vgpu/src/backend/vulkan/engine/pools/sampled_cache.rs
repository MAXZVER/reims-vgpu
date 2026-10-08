//! The retained sampled-image cache: LRU order plus the two names an entry is
//! found by.
//!
//! This used to be a `Vec` in LRU order, oldest first, and every lookup was a
//! `position` scan from the oldest end. Only the byte cap bounds it, so a
//! workload of small textures keeps hundreds of entries live, and a hit — which
//! is by construction near the newest end — walked nearly all of them, then
//! moved the entry with a `remove` + `push`. The Weather app on a 5K guest
//! spent ~20 % of the recorder's busy time in that scan (b058:
//! `find_sampled_by_identity` 5.5 % of all recorder samples, 72 % of them idle),
//! and the recorder is what a faster drain waits on next (b064: the drain's
//! `sampled` phase -1.45 us a draw, its `engine` phase +0.68).
//!
//! Same answers, found by index:
//!
//! * **identity** — `(key, identity)` names at most one entry here. Two entries
//!   can carry the same pair (an identity adopted by a content match, or by a
//!   duplicate admission), and the producer's contract makes them the same
//!   bytes, so the index keeps the one that took the name last; the other stays
//!   reachable by content when it has bytes, and leaves by the caps like any
//!   entry.
//! * **content** — `(key, digest)` proposes candidates; the caller's byte
//!   compare decides, exactly as the scan did. A `Gathered` entry has no digest
//!   and is never a content candidate.
//!
//! Eviction stays oldest-first, and removal goes through
//! [`SampledCache::pop_oldest`] only, so the pool's victim ledger remains the
//! single account of how entries leave.

use super::{ResidentSampledSlot, SampledFingerprint, SampledKey};
use crate::backend::vulkan::engine::SampledContentIdentity;
use std::collections::HashMap;

const NIL: u32 = u32::MAX;

struct Node {
    entry: ResidentSampledSlot,
    /// Towards the oldest entry.
    prev: u32,
    /// Towards the newest entry.
    next: u32,
}

/// A handle to one live entry; valid until that entry is popped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct EntryId(u32);

#[derive(Default)]
pub(super) struct SampledCache {
    nodes: Vec<Option<Node>>,
    free: Vec<u32>,
    oldest: u32,
    newest: u32,
    len: usize,
    by_identity: HashMap<(SampledKey, SampledContentIdentity), u32>,
    by_content: HashMap<(SampledKey, u128), Vec<u32>>,
}

impl SampledCache {
    pub(super) fn new() -> Self {
        SampledCache {
            nodes: Vec::new(),
            free: Vec::new(),
            oldest: NIL,
            newest: NIL,
            len: 0,
            by_identity: HashMap::new(),
            by_content: HashMap::new(),
        }
    }

    pub(super) fn len(&self) -> usize {
        self.len
    }

    pub(super) fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn node(&self, i: u32) -> &Node {
        self.nodes[i as usize].as_ref().expect("live sampled-cache node")
    }

    fn node_mut(&mut self, i: u32) -> &mut Node {
        self.nodes[i as usize].as_mut().expect("live sampled-cache node")
    }

    fn link_newest(&mut self, i: u32) {
        let prev = self.newest;
        {
            let n = self.node_mut(i);
            n.prev = prev;
            n.next = NIL;
        }
        if prev == NIL {
            self.oldest = i;
        } else {
            self.node_mut(prev).next = i;
        }
        self.newest = i;
    }

    fn unlink(&mut self, i: u32) {
        let (prev, next) = {
            let n = self.node(i);
            (n.prev, n.next)
        };
        if prev == NIL {
            self.oldest = next;
        } else {
            self.node_mut(prev).next = next;
        }
        if next == NIL {
            self.newest = prev;
        } else {
            self.node_mut(next).prev = prev;
        }
    }

    /// Admit `entry` as the newest. It takes over the identity name if another
    /// entry held it.
    pub(super) fn push(&mut self, entry: ResidentSampledSlot) -> EntryId {
        let key = entry.slot.key();
        let identity = entry.identity;
        let fingerprint = entry.fingerprint;
        let node = Node {
            entry,
            prev: NIL,
            next: NIL,
        };
        let i = match self.free.pop() {
            Some(i) => {
                self.nodes[i as usize] = Some(node);
                i
            }
            None => {
                self.nodes.push(Some(node));
                (self.nodes.len() - 1) as u32
            }
        };
        self.link_newest(i);
        self.len += 1;
        if let Some(id) = identity {
            self.by_identity.insert((key, id), i);
        }
        if let SampledFingerprint::Content(digest) = fingerprint {
            self.by_content.entry((key, digest)).or_default().push(i);
        }
        EntryId(i)
    }

    /// Remove and return the oldest entry.
    pub(super) fn pop_oldest(&mut self) -> Option<ResidentSampledSlot> {
        let i = self.oldest;
        if i == NIL {
            return None;
        }
        self.unlink(i);
        let node = self.nodes[i as usize].take().expect("live sampled-cache node");
        self.free.push(i);
        self.len -= 1;
        let key = node.entry.slot.key();
        if let Some(id) = node.entry.identity {
            if self.by_identity.get(&(key, id)) == Some(&i) {
                self.by_identity.remove(&(key, id));
            }
        }
        if let SampledFingerprint::Content(digest) = node.entry.fingerprint {
            if let Some(list) = self.by_content.get_mut(&(key, digest)) {
                list.retain(|&j| j != i);
                if list.is_empty() {
                    self.by_content.remove(&(key, digest));
                }
            }
        }
        Some(node.entry)
    }

    /// The entry named `(key, identity)`, if any.
    pub(super) fn find_identity(
        &self,
        key: SampledKey,
        identity: SampledContentIdentity,
    ) -> Option<EntryId> {
        self.by_identity.get(&(key, identity)).map(|&i| EntryId(i))
    }

    /// The first entry under `(key, digest)` that `accept` takes. The digest
    /// only proposes; `accept` is the caller's byte compare.
    pub(super) fn find_content(
        &self,
        key: SampledKey,
        digest: u128,
        mut accept: impl FnMut(&ResidentSampledSlot) -> bool,
    ) -> Option<EntryId> {
        self.by_content
            .get(&(key, digest))?
            .iter()
            .copied()
            .find(|&i| accept(&self.node(i).entry))
            .map(EntryId)
    }

    /// Make `id` the newest entry.
    pub(super) fn touch(&mut self, id: EntryId) {
        if self.newest != id.0 {
            self.unlink(id.0);
            self.link_newest(id.0);
        }
    }

    pub(super) fn get(&self, id: EntryId) -> &ResidentSampledSlot {
        &self.node(id.0).entry
    }

    pub(super) fn get_mut_last_touch(&mut self, id: EntryId) -> &mut u64 {
        &mut self.node_mut(id.0).entry.last_touch_ms
    }

    /// Give entry `id` the identity name, taking it over from any other entry.
    pub(super) fn adopt_identity(&mut self, id: EntryId, identity: SampledContentIdentity) {
        let key = self.node(id.0).entry.slot.key();
        if let Some(old) = self.node(id.0).entry.identity {
            if old != identity && self.by_identity.get(&(key, old)) == Some(&id.0) {
                self.by_identity.remove(&(key, old));
            }
        }
        self.node_mut(id.0).entry.identity = Some(identity);
        self.by_identity.insert((key, identity), id.0);
    }

    /// Every entry, oldest first.
    #[cfg(test)]
    pub(super) fn iter(&self) -> impl Iterator<Item = &ResidentSampledSlot> {
        let mut i = self.oldest;
        std::iter::from_fn(move || {
            if i == NIL {
                return None;
            }
            let n = self.node(i);
            i = n.next;
            Some(&n.entry)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::SampledSlot;
    use super::*;
    use ash::vk;

    fn null_slot(w: u32, h: u32) -> SampledSlot {
        SampledSlot {
            image: vk::Image::null(),
            memory: vk::DeviceMemory::null(),
            view: vk::ImageView::null(),
            width: w,
            height: h,
            layers: 1,
            kind: reims_vgpu_core::texture_shape::TextureKind::D2,
            format: crate::backend::vulkan::translate::pixel::vk_texel_layout(
                crate::protocol::pixel_format::TexelLayout::Bgra8,
            ),
            swizzle: Default::default(),
            levels: 1,
        }
    }

    fn named(k: u64) -> SampledContentIdentity {
        SampledContentIdentity {
            key: k,
            generation: 1,
        }
    }

    fn gathered(w: u32, id: u64) -> ResidentSampledSlot {
        ResidentSampledSlot {
            slot: null_slot(w, 8),
            fingerprint: SampledFingerprint::Gathered,
            content: None,
            content_len: 64,
            identity: Some(named(id)),
            last_touch_ms: 0,
        }
    }

    #[test]
    fn eviction_is_oldest_first_and_a_touch_moves_an_entry_to_the_newest_end() {
        let mut c = SampledCache::new();
        let a = c.push(gathered(8, 1));
        c.push(gathered(9, 2));
        c.push(gathered(10, 3));
        c.touch(a);
        let order: Vec<u32> = c.iter().map(|e| e.slot.width).collect();
        assert_eq!(order, vec![9, 10, 8]);
        assert_eq!(c.pop_oldest().map(|e| e.slot.width), Some(9));
        assert_eq!(c.pop_oldest().map(|e| e.slot.width), Some(10));
        assert_eq!(c.pop_oldest().map(|e| e.slot.width), Some(8));
        assert!(c.pop_oldest().is_none());
        assert!(c.is_empty());
    }

    #[test]
    fn a_popped_entry_is_found_by_neither_name_and_its_slot_is_reused() {
        let mut c = SampledCache::new();
        let bytes = std::sync::Arc::new(vec![7u8; 64]);
        let e = ResidentSampledSlot {
            slot: null_slot(8, 8),
            fingerprint: SampledFingerprint::Content(42),
            content: Some(bytes.clone()),
            content_len: 64,
            identity: Some(named(5)),
            last_touch_ms: 0,
        };
        let key = e.slot.key();
        c.push(e);
        assert!(c.find_identity(key, named(5)).is_some());
        assert!(c.find_content(key, 42, |_| true).is_some());
        c.pop_oldest();
        assert!(c.find_identity(key, named(5)).is_none());
        assert!(c.find_content(key, 42, |_| true).is_none());
        c.push(gathered(8, 6));
        assert_eq!(c.nodes.len(), 1, "the freed node is reused");
    }

    #[test]
    fn an_adopted_identity_moves_to_the_adopter_and_survives_the_old_holder() {
        let mut c = SampledCache::new();
        let first = c.push(gathered(8, 1));
        let second = c.push(gathered(8, 2));
        let key = c.get(first).slot.key();
        c.adopt_identity(second, named(1));
        assert_eq!(c.find_identity(key, named(1)), Some(second));
        assert!(c.find_identity(key, named(2)).is_none(), "the adopter's old name is gone");
        // The old holder leaves; the name stays with the adopter.
        assert_eq!(c.pop_oldest().map(|e| e.identity), Some(Some(named(1))));
        assert_eq!(c.find_identity(key, named(1)), Some(second));
    }

    #[test]
    fn a_content_candidate_is_decided_by_the_callers_compare() {
        let mut c = SampledCache::new();
        let e = ResidentSampledSlot {
            slot: null_slot(8, 8),
            fingerprint: SampledFingerprint::Content(9),
            content: Some(std::sync::Arc::new(vec![1u8; 4])),
            content_len: 4,
            identity: None,
            last_touch_ms: 0,
        };
        let key = e.slot.key();
        c.push(e);
        assert!(c.find_content(key, 9, |e| e.content.as_deref() == Some(&vec![2u8; 4])).is_none());
        assert!(c.find_content(key, 9, |e| e.content.as_deref() == Some(&vec![1u8; 4])).is_some());
        assert!(c.find_content(key, 8, |_| true).is_none(), "another digest proposes nothing");
    }
}
