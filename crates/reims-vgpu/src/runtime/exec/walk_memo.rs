//! TEMP (lab, `REIMS_VGPU_LAB_WALK_MEMO`, default on): what one stream's walk
//! has already resolved, so a bind does not ask the object namespace for it
//! again.
//!
//! # The claim, and why it holds
//!
//! A bind record resolves its object through [`crate::runtime::objects`]:
//! the namespace's name lookup, the retained-resource registry, and for a
//! texture the mapper and backing probes. A stream binds the same few objects
//! over and over — a driven 5K desktop made 40 000 such memo-hit resolutions a
//! second for 800 streams — and each repeat costs two mutexes and a tree walk to
//! reach the answer it gave a few microseconds earlier.
//!
//! Within **one** `walk_stream` nothing can change that answer. A retained
//! resource leaves the registry only through `DeviceState::delete_object` and
//! `delete_task`, which the drain calls from the packet handlers of the delete
//! commands and from nowhere in the record walk; a resolution that succeeded
//! once is therefore the resolution every later bind of the same reference in the
//! same stream would get. The memo lives in the stream's accumulator, so it is
//! born and dropped with the stream and cannot outlive a delete between two
//! streams.
//!
//! **Only successes are kept.** A failed construction is retried on the next
//! bind, as the registry's own contract says ("a descriptor that is still being
//! published can succeed on retry"), so a failure takes the original path every
//! time and counts what it always counted.
//!
//! **The counters stay the same.** A registry hit counts
//! `task_resource_reexamine_deferred` (its interval had not elapsed); the memo
//! counts the same route for the hit it answers, once per skipped resolution,
//! and posts them at the end of the stream. The interval is 100 ms and a stream
//! is microseconds, so the second resolution of a reference in a stream is
//! always one the interval defers.
//!
//! `=off` restores the per-bind resolution, and the unchanged table writes of
//! `apply_binds`, exactly as they were.

use crate::model::TaskResource;
use std::sync::Arc;

/// Whether the memo is on. Read once.
#[inline]
pub(crate) fn on() -> bool {
    #[cfg(test)]
    if let Some(on) = TEST.with(std::cell::Cell::get) {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("REIMS_VGPU_LAB_WALK_MEMO").map_or(true, |v| v != "off"))
}

#[cfg(test)]
thread_local! {
    static TEST: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Run `f` with [`on`] answering `state` on this thread.
#[cfg(test)]
pub(crate) fn with<R>(state: bool, f: impl FnOnce() -> R) -> R {
    let before = TEST.with(|c| c.replace(Some(state)));
    let out = f();
    TEST.with(|c| c.set(before));
    out
}

/// How many distinct references one stream memoises. A stream binding more
/// than this resolves the rest the way it always did; the bound is the one
/// linear scan's length and not a cache policy.
const MAX_REFS: usize = 48;

/// One stream's resolved references.
#[derive(Clone, Debug, Default)]
pub(super) struct StreamMemo {
    resources: Vec<(u32, Arc<TaskResource>)>,
    /// References whose backing probe succeeded in this stream.
    backed: Vec<u32>,
    /// Resolutions this memo answered, not yet counted.
    replayed: u64,
}

impl StreamMemo {
    /// The retained resource this stream already resolved for `obj_ref`, counting
    /// `resolutions` skipped registry lookups against it.
    #[inline]
    pub(super) fn resource(&mut self, obj_ref: u32, resolutions: u64) -> Option<Arc<TaskResource>> {
        let hit = self
            .resources
            .iter()
            .find(|(r, _)| *r == obj_ref)
            .map(|(_, resource)| Arc::clone(resource))?;
        self.replayed += resolutions;
        Some(hit)
    }

    /// Keep a resolution that succeeded.
    #[inline]
    pub(super) fn remember(&mut self, obj_ref: u32, resource: &Arc<TaskResource>) {
        if self.resources.len() < MAX_REFS && !self.resources.iter().any(|(r, _)| *r == obj_ref) {
            self.resources.push((obj_ref, Arc::clone(resource)));
        }
    }

    /// Whether this stream's backing probe for `obj_ref` already succeeded.
    #[inline]
    pub(super) fn is_backed(&self, obj_ref: u32) -> bool {
        self.backed.contains(&obj_ref)
    }

    /// Keep a backing probe that succeeded.
    #[inline]
    pub(super) fn remember_backed(&mut self, obj_ref: u32) {
        if self.backed.len() < MAX_REFS && !self.backed.contains(&obj_ref) {
            self.backed.push(obj_ref);
        }
    }

    /// Post the resolutions answered here, as the route a registry hit counts.
    pub(super) fn flush(&mut self) {
        if self.replayed != 0 {
            crate::runtime::drain::note_store_route_n(
                "task_resource_reexamine_deferred",
                self.replayed,
            );
            self.replayed = 0;
        }
    }
}
