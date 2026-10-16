//! Guest resources the GPU will read from guest RAM when already-recorded work
//! executes, so that a host-side write into them can be ordered after it.
//!
//! Metal orders a blit after every command encoded before it on the queue. This
//! device executes blits on the CPU, straight into guest pages, while it walks
//! the stream - but the draws of earlier passes read their guest-backed sources
//! (`guest_runs` textures, imported buffers) only when the GPU executes them,
//! which can be later. A blit that rewrites such a source therefore reached the
//! earlier passes' draws before they ran: Skia Graphite re-uploading atlas rows
//! between passes during a window resize made glyphs and masks vanish from the
//! tiles of the earlier passes (Claude.app, 2026-10-16, b594-b596).
//!
//! The guest-read debt the completion stamp keeps cannot answer this: the
//! GPU-ordered stamp path takes it the moment the stamp is queued behind the
//! batch, without waiting, so it says "a stamp still has to be ordered behind a
//! read", not "a read is still in flight".
//!
//! So draw preparation names every `(task, reference)` whose bytes a recorded
//! draw will read at execute time, and a host-side writer asks
//! [`take_if_pending`] about its destination before it writes. A hit waits for
//! all recorded GPU work and forgets every name, since everything recorded
//! before the wait has then executed. Names recorded after are new reads.
//!
//! Keyed by reference, not by pages: a view and the texture it views are two
//! names for one storage, and a write through one is not seen as a hit on the
//! other. The draws that hit the hazard measured so far read and are written
//! under one name.

use std::collections::HashSet;
use std::sync::Mutex;

static PENDING: Mutex<Option<HashSet<(u32, u32)>>> = Mutex::new(None);

/// Whether the ordering is on. `REIMS_VGPU_BLIT_READ_ORDER=off` restores the
/// unordered writes (A/B only).
pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        crate::config::switch(crate::config::BLIT_READ_ORDER) != crate::config::Switch::Off
    })
}

/// A recorded draw will read `(task_id, reference)` from guest RAM when it runs.
pub fn note(task_id: u32, reference: u32) {
    if reference == 0 || !enabled() {
        return;
    }
    let mut g = PENDING.lock().unwrap_or_else(|p| p.into_inner());
    g.get_or_insert_with(HashSet::new)
        .insert((task_id, reference));
}

/// Whether a host-side write into `(task_id, reference)` must first wait for
/// recorded GPU work; on `true` every name is forgotten, because the caller
/// is about to wait for all of it.
pub fn take_if_pending(task_id: u32, reference: u32) -> bool {
    if reference == 0 || !enabled() {
        return false;
    }
    let mut g = PENDING.lock().unwrap_or_else(|p| p.into_inner());
    let Some(set) = g.as_mut() else {
        return false;
    };
    if !set.contains(&(task_id, reference)) {
        return false;
    }
    set.clear();
    true
}

/// Everything recorded has executed (a full wait elsewhere): no name is pending.
pub fn forget_all() {
    let mut g = PENDING.lock().unwrap_or_else(|p| p.into_inner());
    if let Some(set) = g.as_mut() {
        set.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_write_into_a_pending_read_waits_once_and_forgets_everything() {
        forget_all();
        note(7, 11);
        note(7, 12);
        assert!(!take_if_pending(7, 13), "an unread reference never waits");
        assert!(take_if_pending(7, 11), "a write into a pending read waits");
        assert!(
            !take_if_pending(7, 12),
            "the wait covered every recorded read"
        );
        note(8, 11);
        assert!(
            !take_if_pending(7, 11),
            "another task's reference is another name"
        );
        assert!(take_if_pending(8, 11));
    }
}
