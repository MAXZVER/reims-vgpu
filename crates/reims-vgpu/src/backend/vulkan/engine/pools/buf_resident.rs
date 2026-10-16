//! TEMP (lab, `REIMS_VGPU_LAB_BUF_RESIDENT=on`): staging copies of draw-time
//! buffer windows kept past the command buffer that made them.
//!
//! # Why
//!
//! On a driven Maps leg 89-91 % of the bytes the recorder lane copies out of
//! guest RAM into staging are the same bytes the same window was copied with
//! before (b653, `REIMS_VGPU_LAB_COPY_REPEAT`: `copy_repeat_mb=778/s` of
//! `lane_copy_mb=879/s`). Each of those copies is a fresh slot that dies with
//! its command buffer, so the next frame copies the window again.
//!
//! # The rule
//!
//! The runtime's buffer witness (`gather_witness::note_buffer_window`) names a
//! window `(key, generation)` and keeps the generation while neither the guest
//! nor this device has written its pages. A slot filled under a generation is
//! therefore the window's bytes for as long as later binds carry that same
//! generation, and binding it again is the copy the CPU would have made.
//!
//! The CPU never writes a kept slot again: a new generation gets a new slot
//! from the staging pool, and the old one goes back to the pool through the
//! *current* entry's live list — freed after this command buffer's fence, which
//! on the one queue is after every earlier command buffer that could have bound
//! it.

use std::collections::HashMap;

use super::BufferSlot;
use super::ResourcePools;
use crate::runtime::gather_witness::GatheredIdentity;

struct Kept {
    generation: u64,
    slot: BufferSlot,
    last: u64,
}

#[derive(Default)]
pub(crate) struct BufResident {
    kept: HashMap<u64, Kept>,
    bytes: u64,
    tick: u64,
}

/// Bytes of kept slots before the least recently bound is let go
/// (`REIMS_VGPU_LAB_BUF_RESIDENT_MB`, default 768).
fn byte_cap() -> u64 {
    static CAP: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *CAP.get_or_init(|| {
        std::env::var("REIMS_VGPU_LAB_BUF_RESIDENT_MB")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(768)
            << 20
    })
}

impl ResourcePools {
    /// The slot kept for `id`, when it was filled under this very generation.
    pub(crate) fn buf_resident_hit(&mut self, id: GatheredIdentity, len: u64) -> Option<BufferSlot> {
        let res = &mut self.buf_resident;
        res.tick += 1;
        let kept = res.kept.get_mut(&id.key)?;
        if kept.generation != id.generation || kept.slot.size < len {
            return None;
        }
        kept.last = res.tick;
        crate::runtime::drain::note_store_route("brs_hit");
        crate::runtime::drain::note_store_route_n("brs_hit_kb", len / 1024);
        Some(kept.slot)
    }

    /// Keep `slot` — just taken from the pool and filled for `id` — past this
    /// command buffer. Whatever `id`'s key kept before, and whatever the byte
    /// cap pushes out, returns to the pool after this entry's fence.
    pub(crate) fn buf_resident_adopt(&mut self, id: GatheredIdentity, slot: BufferSlot) {
        let live = &mut self.entry.staging_live;
        let Some(at) = live.iter().rposition(|s| s.buffer == slot.buffer) else {
            return;
        };
        live.swap_remove(at);
        let res = &mut self.buf_resident;
        res.tick += 1;
        res.bytes += slot.size;
        crate::runtime::drain::note_store_route("brs_adopt");
        let old = res.kept.insert(
            id.key,
            Kept {
                generation: id.generation,
                slot,
                last: res.tick,
            },
        );
        if let Some(old) = old {
            res.bytes -= old.slot.size;
            live.push(old.slot);
        }
        while res.bytes > byte_cap() {
            let Some((&victim, _)) = res
                .kept
                .iter()
                .filter(|(&k, _)| k != id.key)
                .min_by_key(|(_, k)| k.last)
            else {
                break;
            };
            if let Some(gone) = res.kept.remove(&victim) {
                res.bytes -= gone.slot.size;
                live.push(gone.slot);
                crate::runtime::drain::note_store_route("brs_evict");
            }
        }
    }

    /// Every kept slot, for teardown.
    pub(crate) fn buf_resident_take_all(&mut self) -> Vec<BufferSlot> {
        let res = &mut self.buf_resident;
        res.bytes = 0;
        res.kept.drain().map(|(_, k)| k.slot).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::super::{BufferBacking, BufferSlot, ResourcePools};
    use crate::runtime::gather_witness::GatheredIdentity;
    use ash::vk::{self, Handle};

    fn slot(raw: u64, size: u64) -> BufferSlot {
        BufferSlot {
            buffer: vk::Buffer::from_raw(raw),
            memory: vk::DeviceMemory::null(),
            size,
            mapped: 0,
            backing: BufferBacking::Dedicated,
            coherent: true,
            cached: false,
        }
    }

    fn id(key: u64, generation: u64) -> GatheredIdentity {
        GatheredIdentity { key, generation }
    }

    /// A kept slot answers only its own generation, leaves the entry's live list
    /// when kept, and goes back to it — to be freed after this entry's fence —
    /// when a new generation replaces it.
    #[test]
    fn a_kept_slot_serves_its_generation_and_returns_through_the_live_list() {
        let mut pools = ResourcePools::new();
        let a = slot(1, 4096);
        pools.entry.staging_live.push(a);
        pools.buf_resident_adopt(id(7, 1), a);
        assert!(pools.entry.staging_live.is_empty());
        assert_eq!(pools.buf_resident_hit(id(7, 1), 4096).map(|s| s.buffer), Some(a.buffer));
        assert!(pools.buf_resident_hit(id(7, 2), 4096).is_none());
        assert!(pools.buf_resident_hit(id(7, 1), 8192).is_none());

        let b = slot(2, 4096);
        pools.entry.staging_live.push(b);
        pools.buf_resident_adopt(id(7, 2), b);
        assert_eq!(pools.entry.staging_live.len(), 1);
        assert_eq!(pools.entry.staging_live[0].buffer, a.buffer);
        assert_eq!(pools.buf_resident_hit(id(7, 2), 4096).map(|s| s.buffer), Some(b.buffer));

        let all = pools.buf_resident_take_all();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].buffer, b.buffer);
    }

    /// A slot that is not in the entry's live list was not just taken for this
    /// bind, so it is not adopted.
    #[test]
    fn a_slot_not_live_in_this_entry_is_not_kept() {
        let mut pools = ResourcePools::new();
        pools.buf_resident_adopt(id(3, 1), slot(9, 64));
        assert!(pools.buf_resident_hit(id(3, 1), 64).is_none());
    }
}
