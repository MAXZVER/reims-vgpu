//! TEMP (lab, FRAME_GATE + AHEAD): what work prepared ahead of its stream's
//! execution read, and what the drain wrote since - the write-note log of the
//! frame-region design (`lab/journal/2026-10-10-frame-region-design.md` 5.4).
//!
//! A frame helper may prepare a stream's draws while the drain still executes
//! the streams before it. The plans those draws carry validate surface and
//! buffer state at the commit, but not guest bytes read through the CPU rails
//! (linear textures, buffer contents): an earlier stream's Store into the same
//! guest pages arms a debt the prepared draw never saw. So an ahead job logs
//! the guest windows it reads ([`ReadScope`]), the drain notes every guest
//! window it is about to write and every write it cannot name ([`note_gva`],
//! [`note_all`]), and the commit takes a prepared draw only when no note since
//! its preparation intersects its reads ([`clean_since`]).
//!
//! Address spaces are named by the task's page-directory pfn, not its id: two
//! ids over one address space read and write the same pages.

use std::cell::RefCell;
use std::collections::VecDeque;

/// One guest window: `(address space, first byte, end)`.
pub(crate) type Window = (u32, u64, u64);

thread_local! {
    static READS: RefCell<Option<Vec<Window>>> = const { RefCell::new(None) };
}

/// Log this thread's guest reads until [`ReadScope::finish`]. Nested scopes are
/// not supported (the inner one wins until it finishes).
pub(crate) struct ReadScope {
    was: Option<Vec<Window>>,
}

impl ReadScope {
    pub(crate) fn begin() -> ReadScope {
        let was = READS.with(|r| r.borrow_mut().replace(Vec::new()));
        ReadScope { was }
    }

    /// The windows read since [`Self::begin`], merged where they touch.
    pub(crate) fn finish(mut self) -> Vec<Window> {
        let mut got = READS.with(|r| std::mem::replace(&mut *r.borrow_mut(), self.was.take()))
            .unwrap_or_default();
        got.sort_unstable();
        let mut out: Vec<Window> = Vec::with_capacity(got.len());
        for w in got {
            match out.last_mut() {
                Some(last) if last.0 == w.0 && w.1 <= last.2 => last.2 = last.2.max(w.2),
                _ => out.push(w),
            }
        }
        out
    }
}

impl Drop for ReadScope {
    fn drop(&mut self) {
        if let Some(was) = self.was.take() {
            READS.with(|r| *r.borrow_mut() = Some(was));
        }
    }
}

/// Whether this thread is logging (an ahead job is running here).
#[inline]
pub(crate) fn logging() -> bool {
    READS.with(|r| r.borrow().is_some())
}

/// A guest read of `len` bytes at `gva` in address space `space`.
#[inline]
pub(crate) fn note_read(space: u32, gva: u64, len: u64) {
    if len == 0 {
        return;
    }
    READS.with(|r| {
        if let Some(v) = r.borrow_mut().as_mut() {
            v.push((space, gva, gva.saturating_add(len)));
        }
    });
}

#[derive(Clone, Copy, Debug)]
enum Note {
    Gva(Window),
    All,
}

struct Notes {
    seq: u64,
    log: VecDeque<(u64, Note)>,
}

/// Notes kept; a preparation older than the oldest kept is not trusted.
const KEEP: usize = 8192;

static NOTES: parking_lot::Mutex<Notes> = parking_lot::Mutex::new(Notes {
    seq: 0,
    log: VecDeque::new(),
});

fn push(note: Note) {
    let mut n = NOTES.lock();
    n.seq += 1;
    let seq = n.seq;
    n.log.push_back((seq, note));
    if n.log.len() > KEEP {
        n.log.pop_front();
    }
}

/// The drain is about to write `len` guest bytes at `gva` in `space`.
pub(crate) fn note_gva(space: u32, gva: u64, len: u64) {
    if len != 0 {
        push(Note::Gva((space, gva, gva.saturating_add(len))));
    }
}

/// The drain wrote (or may have written) guest memory it does not name.
pub(crate) fn note_all() {
    push(Note::All);
}

/// The note sequence now: a preparation records it before it starts.
pub(crate) fn seq() -> u64 {
    NOTES.lock().seq
}

/// Whether nothing noted after `since` intersects `reads`.
pub(crate) fn clean_since(since: u64, reads: &[Window]) -> bool {
    let n = NOTES.lock();
    if n.seq == since {
        return true;
    }
    if n.log.front().is_some_and(|(s, _)| *s > since + 1) {
        return false;
    }
    for (s, note) in n.log.iter().rev() {
        if *s <= since {
            break;
        }
        match note {
            Note::All => return false,
            Note::Gva((space, lo, hi)) => {
                if reads
                    .iter()
                    .any(|(rs, rlo, rhi)| rs == space && rlo < hi && lo < rhi)
                {
                    return false;
                }
            }
        }
    }
    true
}
