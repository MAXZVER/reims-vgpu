//! TEMP (lab): values the drain is done with, dropped on another thread.
//!
//! A prepared draw is built on a pool helper: its vectors come from that
//! helper's allocator cache and its `Arc<TaskResource>` clones bumped counts on
//! that helper's core. Dropping it on the drain after the commit pays a cache
//! miss per count and a cross-thread free per vector - b163 (Weather, PGO
//! build, drain sampler) put `Arc` decrements at the end of `commit_shared`
//! alone at ~2.7 % of the drain's busy samples, and the region's shared request
//! vector, `DrawEncodeRequest` and bind-vector drops at ~4 % more. Nothing in
//! those drops is ordered against guest-visible work (the device's own tables
//! hold the owning references), so a reaper thread may run them whenever.
//!
//! [`later`] hands a value over (a channel send, batched by the callers per
//! region); without the thread (tests, or it failed to start) the value drops
//! where it is, as before. `REIMS_VGPU_LAB_REAP=off` keeps every drop on the
//! drain.

use std::sync::mpsc::{channel, Sender};
use std::sync::OnceLock;

type Garbage = Box<dyn Send>;

fn reaper() -> Option<&'static Sender<Garbage>> {
    static TX: OnceLock<Option<Sender<Garbage>>> = OnceLock::new();
    TX.get_or_init(|| {
        if cfg!(test) || std::env::var("REIMS_VGPU_LAB_REAP").as_deref() == Ok("off") {
            return None;
        }
        let (tx, rx) = channel::<Garbage>();
        std::thread::Builder::new()
            .name("reims-reap".into())
            .spawn(move || {
                while let Ok(garbage) = rx.recv() {
                    drop(garbage);
                }
            })
            .ok()?;
        Some(tx)
    })
    .as_ref()
}

/// Drop `value` on the reaper thread (here when there is none).
pub(crate) fn later<T: Send + 'static>(value: T) {
    if let Some(tx) = reaper() {
        // A send fails only once the reaper is gone; the value comes back.
        if let Err(back) = tx.send(Box::new(value)) {
            drop(back.0);
        }
    } else {
        drop(value);
    }
}

/// A value that goes to the reaper when it is dropped or replaced.
pub(crate) struct Reaped<T: Send + 'static>(Option<T>);

impl<T: Send + 'static> Reaped<T> {
    pub(crate) fn new(value: T) -> Self {
        Reaped(Some(value))
    }
}

impl<T: Send + 'static> std::ops::Deref for Reaped<T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.0.as_ref().expect("a reaped value is present until it drops")
    }
}

impl<T: Send + 'static> Drop for Reaped<T> {
    fn drop(&mut self) {
        if let Some(value) = self.0.take() {
            later(value);
        }
    }
}
