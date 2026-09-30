//! What the resident registry says about an identity's content, readable
//! without the engine lock.
//!
//! # The question, and why it cannot cost the engine lock
//!
//! "Does a ready resident hold this identity's pixels, and is that image the
//! guest's own allocation?" is asked before a draw decides where a sampled
//! source comes from, and the registry that knows lives behind `ENGINE`. The
//! drain holds that lock for most of every draw; the window thread takes it;
//! and the parallel preparation workers of [`crate::runtime::par`] would queue
//! behind it for a reading that is a few words. So the registry *publishes* the
//! answer here, at the moment it changes, and readers ask this table instead.
//!
//! # What is published
//!
//! Per identity: whether the resident is content-ready, whether its image is a
//! guest import, the mapping content epoch it was last vouched for at, and a
//! **version**. The version is what a reader records beside its answer and
//! re-reads to learn that the answer is still current: it moves on every
//! publish — a colour or depth store, an epoch stamp, a birth — and it is issued
//! from one counter for the whole table, so an identity that dies and is born
//! again carries a version no earlier incarnation ever had. Absence is not a
//! version; a reader that saw none and still sees none saw nothing change.
//!
//! It moves on a store even when no field does, because a store is the event
//! that matters: the pixels are different. A reader that recorded `ready = true`
//! at version 7 and finds `ready = true` at version 9 has been told the image was
//! drawn into since, which is exactly what invalidates a plan that read it.
//!
//! # Who writes, and why there is one writer per cell
//!
//! Every publish happens under `ENGINE`, from [`super::ResidentTargetSlot`]'s
//! own mutators, through a [`ReadinessHandle`] the slot owns. The handle's
//! `publish` takes `&mut self` and the slot is reached through `&mut
//! ResourcePools`, so two writers to one cell cannot exist — which is what lets
//! the cell be a single-writer seqlock and not a lock. Birth is the slot being
//! registered; death is the slot being dropped, so a path that removes a
//! resident cannot forget to take it off this table.
//!
//! # What "lock-free" means here, exactly
//!
//! A read takes no engine lock, waits for no writer of a cell's fields (a
//! seqlock retries, it does not block), and allocates nothing. It does take the
//! table's map lock for *shared* access to find the cell, and that lock is
//! written only when a resident is born or dies — a per-surface event, tens of
//! nanoseconds, never held across any other lock or any call. So the hot
//! question, "what is the state of a resident that already exists", is an atomic
//! read, and the only thing a reader can ever wait for is a neighbouring
//! resident being registered.
//!
//! # One table, several pools
//!
//! The table is process-wide, like the engine it mirrors, but it is written by
//! whichever `ResourcePools` owns the slot. A pools that is replaced or dropped
//! takes its residents off the table as its slots drop, and a handle removes its
//! entry only if the entry is still *its own* cell, so a second pools that has
//! registered the same identity meanwhile — which unit tests do, and the live
//! engine does not — is left alone.

use super::super::types::TargetIdentity;
use once_cell::sync::Lazy;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::atomic::{fence, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

/// One resident's published state, read as a whole.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidentReadiness {
    /// The image holds rendered contents (a draw stored into it, or it *is* the
    /// guest's pages and so always does).
    pub ready: bool,
    /// The image is the guest's own allocation, imported, rather than a device
    /// allocation of ours.
    pub guest_imported: bool,
    /// The mapping content epoch the pixels were last vouched for at, or `None`
    /// when nothing has vouched for them since the last draw.
    pub content_epoch: Option<u32>,
    /// Changes whenever the resident's content can have changed; see the module
    /// doc. Never 0, and never repeats for an identity.
    pub version: u64,
}

const READY: u32 = 1;
const GUEST_IMPORTED: u32 = 1 << 1;
const HAS_EPOCH: u32 = 1 << 2;

/// Source of every version, for every identity.
static NEXT_VERSION: AtomicU64 = AtomicU64::new(1);

/// One resident's cell: a seqlock over four words, with one writer.
///
/// The writer marks `seq` odd, stores the words, and marks it even again; a
/// reader retries until it reads the same even `seq` before and after the
/// words. The words are atomics accessed `Relaxed` and ordered by the fences, so
/// a torn read is a retry and never a data race.
struct Cell {
    seq: AtomicU32,
    flags: AtomicU32,
    epoch: AtomicU32,
    version: AtomicU64,
}

impl Cell {
    fn new() -> Self {
        Self {
            seq: AtomicU32::new(0),
            flags: AtomicU32::new(0),
            epoch: AtomicU32::new(0),
            version: AtomicU64::new(0),
        }
    }

    /// Single writer: the caller holds `&mut` to the handle that owns this cell
    /// for writing.
    fn write(&self, ready: bool, guest_imported: bool, epoch: Option<u32>) -> u64 {
        let version = NEXT_VERSION.fetch_add(1, Ordering::Relaxed);
        let flags = u32::from(ready) * READY
            | u32::from(guest_imported) * GUEST_IMPORTED
            | u32::from(epoch.is_some()) * HAS_EPOCH;
        let open = self.seq.load(Ordering::Relaxed).wrapping_add(1);
        self.seq.store(open, Ordering::Relaxed);
        fence(Ordering::Release);
        self.flags.store(flags, Ordering::Relaxed);
        self.epoch.store(epoch.unwrap_or(0), Ordering::Relaxed);
        self.version.store(version, Ordering::Relaxed);
        self.seq.store(open.wrapping_add(1), Ordering::Release);
        version
    }

    fn read(&self) -> ResidentReadiness {
        loop {
            let before = self.seq.load(Ordering::Acquire);
            if before & 1 == 1 {
                std::hint::spin_loop();
                continue;
            }
            let flags = self.flags.load(Ordering::Relaxed);
            let epoch = self.epoch.load(Ordering::Relaxed);
            let version = self.version.load(Ordering::Relaxed);
            fence(Ordering::Acquire);
            if self.seq.load(Ordering::Relaxed) == before {
                return ResidentReadiness {
                    ready: flags & READY != 0,
                    guest_imported: flags & GUEST_IMPORTED != 0,
                    content_epoch: (flags & HAS_EPOCH != 0).then_some(epoch),
                    version,
                };
            }
        }
    }
}

static TABLE: Lazy<RwLock<HashMap<TargetIdentity, Arc<Cell>>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

/// The published state of `identity`, or `None` when no resident is registered
/// under it.
///
/// Takes no engine lock. The answer is the state as of the last publish: a
/// reader that needs it to stay true records the `version` and compares it
/// later.
pub(crate) fn lookup(identity: &TargetIdentity) -> Option<ResidentReadiness> {
    TABLE.read().get(identity).map(|cell| cell.read())
}

/// A slot's hold on its table entry: how it publishes, and what removes the
/// entry when the slot goes.
pub(crate) struct ReadinessHandle {
    cell: Arc<Cell>,
    /// The key the cell is registered under, until it is retired. `None` for a
    /// handle that was never on the table.
    key: Option<TargetIdentity>,
}

impl ReadinessHandle {
    /// Register a new resident under `identity` with its birth state.
    ///
    /// Takes the map's write lock for the insert, and publishes the first
    /// version before the entry is visible, so a reader can never find a cell
    /// that has not been written.
    pub(crate) fn register(identity: &TargetIdentity, ready: bool, guest_imported: bool) -> Self {
        let cell = Arc::new(Cell::new());
        cell.write(ready, guest_imported, None);
        TABLE.write().insert(identity.clone(), Arc::clone(&cell));
        Self {
            cell,
            key: Some(identity.clone()),
        }
    }

    /// A handle that is not on the table: what a slot built by a test's fixture
    /// carries. Publishing to it is harmless and nothing reads it.
    #[cfg(test)]
    pub(crate) fn detached() -> Self {
        Self {
            cell: Arc::new(Cell::new()),
            key: None,
        }
    }

    /// Publish the slot's state, moving its version. Returns the new version.
    pub(crate) fn publish(&mut self, ready: bool, guest_imported: bool, epoch: Option<u32>) -> u64 {
        self.cell.write(ready, guest_imported, epoch)
    }

    /// Take the entry off the table now. Idempotent; also what dropping does.
    ///
    /// Removes the entry only if it is still this handle's own cell, so a
    /// resident re-registered under the same identity — the old slot is
    /// dropped *after* the new one is inserted — is not removed with it.
    pub(crate) fn retire(&mut self) {
        let Some(key) = self.key.take() else {
            return;
        };
        let mut table = TABLE.write();
        if table
            .get(&key)
            .is_some_and(|cell| Arc::ptr_eq(cell, &self.cell))
        {
            table.remove(&key);
        }
    }
}

impl Drop for ReadinessHandle {
    fn drop(&mut self) {
        self.retire();
    }
}

impl std::fmt::Debug for ReadinessHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReadinessHandle")
            .field("on_table", &self.key.is_some())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;

    fn surface(id: u32) -> TargetIdentity {
        // Ids well away from anything another test registers, so the shared
        // process-wide table cannot hand this module somebody else's entry.
        TargetIdentity::Surface {
            id: 0x7e57_0000 + id,
            width: 64,
            height: 64,
            generation: 1,
            format: ash::vk::Format::B8G8R8A8_UNORM,
        }
    }

    #[test]
    fn a_registered_resident_is_found_with_its_birth_state_and_leaves_with_its_handle() {
        let id = surface(1);
        assert_eq!(lookup(&id), None);
        let mut handle = ReadinessHandle::register(&id, false, false);
        let born = lookup(&id).expect("registered");
        assert!(!born.ready && !born.guest_imported);
        assert_eq!(born.content_epoch, None);
        assert_ne!(born.version, 0);

        let drawn = handle.publish(true, false, None);
        let after = lookup(&id).unwrap();
        assert!(after.ready);
        assert_eq!(after.version, drawn);
        assert!(drawn > born.version, "a store moves the version");

        let stamped = handle.publish(true, false, Some(5));
        let after = lookup(&id).unwrap();
        assert_eq!(after.content_epoch, Some(5));
        assert!(stamped > drawn);

        // A store with nothing else changed still moves it: the pixels did.
        let again = handle.publish(true, false, None);
        assert!(again > stamped);
        assert_eq!(lookup(&id).unwrap().content_epoch, None);

        drop(handle);
        assert_eq!(lookup(&id), None, "the slot going takes its entry with it");
    }

    #[test]
    fn a_guest_imported_resident_born_ready_says_so() {
        let id = surface(2);
        let _handle = ReadinessHandle::register(&id, true, true);
        let r = lookup(&id).unwrap();
        assert!(r.ready && r.guest_imported);
    }

    /// An identity that dies and is born again never repeats a version, which is
    /// what lets a reader that recorded one incarnation's version tell it from
    /// the next's.
    #[test]
    fn a_reborn_identity_never_repeats_a_version() {
        let id = surface(3);
        let first = ReadinessHandle::register(&id, false, false);
        let v1 = lookup(&id).unwrap().version;
        drop(first);
        let _second = ReadinessHandle::register(&id, false, false);
        let v2 = lookup(&id).unwrap().version;
        assert!(v2 > v1);
    }

    /// The registry inserts the new slot before the replaced one is dropped, so
    /// the old handle's retirement must not remove the new one's entry.
    #[test]
    fn retiring_a_replaced_handle_leaves_the_replacement_on_the_table() {
        let id = surface(4);
        let old = ReadinessHandle::register(&id, false, false);
        let new = ReadinessHandle::register(&id, true, false);
        drop(old);
        let r = lookup(&id).expect("the replacement is still registered");
        assert!(r.ready, "and it is the replacement's state that is served");
        drop(new);
        assert_eq!(lookup(&id), None);
    }

    #[test]
    fn a_detached_handle_publishes_nowhere_anyone_can_see() {
        let id = surface(5);
        let mut detached = ReadinessHandle::detached();
        detached.publish(true, true, Some(1));
        assert_eq!(lookup(&id), None);
    }

    /// Readers never see a torn tuple and never see time go backwards, while a
    /// writer flips between two states that differ in every field.
    #[test]
    fn a_reader_never_sees_a_torn_state_or_a_version_go_backwards() {
        let id = surface(6);
        let mut handle = ReadinessHandle::register(&id, false, false);
        let stop = AtomicBool::new(false);
        std::thread::scope(|s| {
            let readers: Vec<_> = (0..3)
                .map(|_| {
                    s.spawn(|| {
                        let mut last = 0u64;
                        let mut reads = 0u64;
                        while !stop.load(Ordering::Relaxed) {
                            let r = lookup(&id).expect("registered for the whole test");
                            // One of exactly the two states the writer flips
                            // between, and nothing in between.
                            let quiet = !r.ready && !r.guest_imported && r.content_epoch.is_none();
                            let loud =
                                r.ready && r.guest_imported && r.content_epoch == Some(0xfeed);
                            assert!(quiet || loud, "torn read: {r:?}");
                            assert!(r.version >= last, "version went backwards");
                            last = r.version;
                            reads += 1;
                        }
                        reads
                    })
                })
                .collect();
            for i in 0..200_000u32 {
                if i % 2 == 0 {
                    handle.publish(true, true, Some(0xfeed));
                } else {
                    handle.publish(false, false, None);
                }
            }
            stop.store(true, Ordering::Relaxed);
            for r in readers {
                assert!(r.join().expect("reader does not panic") > 0);
            }
        });
    }
}
