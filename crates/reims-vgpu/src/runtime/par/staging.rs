//! TEMP (lab, `REIMS_VGPU_LAB_PASS_STAGE=on|off`): the staging arena one pool
//! thread memcpys draw inputs into, so the recorder thread that executes the
//! draw records only the Vulkan commands that read them.
//!
//! Design: `lab/journal/2026-10-03-intra-packet-parallel-design.md` §8.4 (C1).
//! The engine's staging cost — `acquire_staging` + `write_staging` for vertex /
//! storage / seed binds, the sampled gather's `write_staging_from_runs`, the
//! seed's R/B exchange — is ~9.4 µs of the recorder's ~36 µs per draw. This
//! module is the bookkeeping half of moving it: one arena per pool thread (a
//! helper or the coordinator), a bump pointer inside each chunk, and the
//! chunk returned only when the submissions that read it have retired.
//!
//! # The model is `reims_vgpu_vulkan::staging`, with the timeline replaced
//!
//! That arena frees a whole chunk when a timeline point passes it. The engine
//! here has no timeline semaphore; its retirement unit is the ring slot's
//! fence, and one window can be read by a submission that did not exist when
//! the window was carved (a deferred draw joins an open batch and records from
//! a window an earlier plan staged). So the free unit is the *reference*, not
//! the submission: [`StagingRing::alloc`] takes one (the fact's own, released
//! when the plan that carries it dies), [`StagingRing::retain`] takes one per
//! submission that records a read, and [`StagingRing::release`] drops one. A
//! chunk is reusable when its references reach zero — the same exactness the
//! timeline gave, expressed against this engine's own lifetime.
//!
//! # Bookkeeping without handles
//!
//! [`StagingRing`] holds no Vulkan object, so every rule here is tested on a
//! machine with no GPU. The engine side that owns the buffers is
//! `crate::backend::vulkan::engine::pass_stage`; this module must stay usable
//! from the runtime layer, which is why it lives under `par` beside the pool
//! that runs the memcpy.
//!
//! # Exhaustion is a refusal, never a wait
//!
//! When every chunk holds references, [`StagingRing::alloc`] refuses and the
//! caller falls back to the serial staging path — the recorder memcpy this
//! module exists to remove, paid once more. Blocking inside a planning job is
//! how one worker's depth becomes every worker's latency, and resetting the
//! oldest chunk is the use-after-submit with extra steps
//! (`reims_vgpu_vulkan::staging` states the same rule of its own ring).
//!
//! # Switches (TEMP, lab)
//!
//! * `REIMS_VGPU_LAB_PASS_STAGE=on|off` — the whole rail. Off is byte-for-byte
//!   the old path: no arena is created, no fact is staged, no census fires.
//! * `REIMS_VGPU_LAB_PASS_STAGE_CHUNK_KB=N` — chunk size in KiB (default 8192;
//!   a 1080p RGBA8 seed is 8.1 MB and needs the next power of two).
//! * `REIMS_VGPU_LAB_PASS_STAGE_CHUNKS=N` — chunks per arena (default 2).
//! * `REIMS_VGPU_LAB_STAGED_PROVEN=off` — restores the eager sampled attach
//!   (every plan's window copied at plan time) for A/B inside the rail; the
//!   default copies a gather window only once a commit has proven the plan
//!   (the repeat that survived a bind), because the eager copies died with
//!   99.4 % of the plans they served.

use std::sync::Mutex;

/// The pool population the arenas are sized for; see [`population`].
static POPULATION: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(1);

/// TEMP (lab): `REIMS_VGPU_LAB_PASS_STAGE`, read once per process.
pub(crate) fn enabled() -> bool {
    // Unit tests drive the staging arms through their own doors; the process
    // default stays off so an unrelated test never stages.
    #[cfg(test)]
    {
        return TEST_ENABLED.load(std::sync::atomic::Ordering::Relaxed);
    }
    #[cfg(not(test))]
    {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        // On by default again (`=off` restores the plain rail). n223's loss
        // (windowed glass 88.6 / 158.9 Hz on against 196.0 / 193.4 off) was
        // the eager attach copying every fresh plan's window (143 GB in 32 s);
        // with the STAGED_PROVEN gate (n231, Intel, cssscrollglassv) the copies
        // fall to 55-181 MB and Hz holds parity (176.4 / 192.1 against
        // 187.3 / 191.6). Flipped for the merged candidate (Mish 2026-10-05:
        // "merge everything into the final version"); p009 re-checks it.
        *ON.get_or_init(|| {
            !matches!(
                std::env::var("REIMS_VGPU_LAB_PASS_STAGE").as_deref(),
                Ok("off")
            )
        })
    }
}

/// TEMP (lab): `REIMS_VGPU_LAB_STAGED_PROVEN`, on by default — the sampled
/// staging rule: a gather window is copied only for a *proven* plan, the
/// repeat a commit handed back after serving one bind (see
/// [`crate::runtime::draw::vulkan::sample_plan::attach_staged`]). The eager
/// attach copied every fresh plan's window at plan time, and 99.4 % of them
/// died before any bind read them (cssscrollglassv, 32 s:
/// `pass_stage_sampled_planned=9582` against `hit=54`, `pass_stage_bytes=121 GB`
/// of copies no engine arm ever read). `=off` restores the eager attach for
/// A/B inside the staging rail; the rail's own gate stays
/// `REIMS_VGPU_LAB_PASS_STAGE`. Live (phases inside one boot) through
/// `REIMS_VGPU_LAB_LIVE_FILE` — the rule is a per-plan decision, so a phase
/// boundary costs nothing.
pub(crate) fn staged_proven() -> bool {
    // Unit tests pin their arm through their own door (the rule-off tests
    // must keep the eager rail the old tests were written against).
    #[cfg(test)]
    {
        return TEST_STAGED_PROVEN.load(std::sync::atomic::Ordering::Relaxed);
    }
    #[cfg(not(test))]
    {
        if let Some(on) = crate::runtime::lab_live::on("STAGED_PROVEN") {
            return on;
        }
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| {
            !matches!(
                std::env::var("REIMS_VGPU_LAB_STAGED_PROVEN").as_deref(),
                Ok("off")
            )
        })
    }
}

#[cfg(test)]
static TEST_STAGED_PROVEN: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Force the rule for the lifetime of the returned guard (tests only).
#[cfg(test)]
pub(crate) fn test_staged_proven(on: bool) -> StagedProvenGuard {
    TEST_STAGED_PROVEN.store(on, std::sync::atomic::Ordering::Relaxed);
    StagedProvenGuard
}

/// Restores the test rule. Never outlives its guard's frame.
#[cfg(test)]
pub(crate) struct StagedProvenGuard;

#[cfg(test)]
impl Drop for StagedProvenGuard {
    fn drop(&mut self) {
        TEST_STAGED_PROVEN.store(false, std::sync::atomic::Ordering::Relaxed);
    }
}

#[cfg(test)]
static TEST_ENABLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Force the flag for the lifetime of the returned guard (tests only).
#[cfg(test)]
pub(crate) fn test_enable(on: bool) -> TestEnableGuard {
    TEST_ENABLED.store(on, std::sync::atomic::Ordering::Relaxed);
    TestEnableGuard
}

/// Restores the test flag. Never outlives its guard's frame.
#[cfg(test)]
pub(crate) struct TestEnableGuard;

#[cfg(test)]
impl Drop for TestEnableGuard {
    fn drop(&mut self) {
        TEST_ENABLED.store(false, std::sync::atomic::Ordering::Relaxed);
    }
}

/// TEMP (lab): `REIMS_VGPU_LAB_PASS_STAGE_CHUNK_KB`, clamped to at least 64 —
/// a chunk smaller than a page cannot stage a seed row and refuses
/// everything, which reads as a broken rail rather than a small one.
pub(crate) fn chunk_bytes() -> u64 {
    static CHUNK: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *CHUNK.get_or_init(|| {
        let kb = std::env::var("REIMS_VGPU_LAB_PASS_STAGE_CHUNK_KB")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            // 48 MiB: the 5K colour seed is 44 MB and refused to fit 8 MiB
            // (n207: arena_full 1.1 GB a boot bypassing the rail).
            .unwrap_or(49152)
            .max(64);
        kb.saturating_mul(1024)
    })
}

/// TEMP (lab): `REIMS_VGPU_LAB_PASS_STAGE_CHUNKS`, clamped 1..=8.
pub(crate) fn chunks_per_arena() -> usize {
    static CHUNKS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CHUNKS.get_or_init(|| {
        std::env::var("REIMS_VGPU_LAB_PASS_STAGE_CHUNKS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(2)
            .clamp(1, 8)
    })
}

/// How many arenas the engine must create: every pool member needs one, and
/// the population is helpers plus the coordinator. Set when the device
/// installs its pool, read when the engine first needs the arenas; a device
/// with no pool still gets one arena, because a serial scope's coordinator
/// stages on the drain thread's own arena (correct, just not parallel).
pub(crate) fn population() -> usize {
    POPULATION.load(std::sync::atomic::Ordering::Relaxed)
}

/// See [`population`]. Called from `DeviceState::install_par_pool`.
pub(crate) fn set_population(members: usize) {
    POPULATION.store(members.max(1), std::sync::atomic::Ordering::Relaxed);
}

/// One carved window of one chunk: where the staged bytes sit, and what must
/// be released when they die.
///
/// The chunk index is small by construction (a chunk is MiBs and an arena is
/// a handful of them) and the pair travels inside facts across the
/// plan/commit boundary, so it is `Copy` and carries nothing but position.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StagingWindow {
    pub chunk: u32,
    pub offset: u64,
    pub len: u64,
}

impl StagingWindow {
    /// The chunk and offset as one `u64` for content keys: the chunk rides the
    /// high bits, clear of the offset because a chunk is MiBs and an arena a
    /// handful of chunks (offsets stay under 2^48 by the size clamps).
    pub fn packed_position(&self) -> u64 {
        debug_assert!(self.offset < (1u64 << 48), "a chunk larger than 2^48");
        (u64::from(self.chunk) << 48) | self.offset
    }
}

/// Why an allocation refused. Every arm is served by the caller falling back
/// to the serial staging path, so the refusal names the condition for the
/// census rather than for a handler.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StagingRefusal {
    /// Larger than a chunk; no arrangement of chunks fits it. The serial path
    /// stages it in a dedicated slot, exactly as it always did.
    TooLarge { requested: u64, chunk: u64 },
    /// Every chunk still holds references. The timeline (the ring slots'
    /// fences) is the only thing that frees them.
    Exhausted,
    /// An alignment that is not a power of two. Every Vulkan alignment is.
    BadAlignment { alignment: u64 },
    /// A request for no bytes. The serial path's staging slot zeroes a 4-byte
    /// minimum; a staged window of nothing would copy nothing, so the caller
    /// keeps the serial shape rather than a second one.
    Empty,
}

impl std::fmt::Debug for StagingRing {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The census is the readable state; the chunks are lock-guarded and
        // nothing that prints a ring wants their addresses.
        f.debug_struct("StagingRing")
            .field("census", &self.census())
            .field("in_flight", &self.in_flight())
            .finish()
    }
}

/// How the ring has been used, for the `pass_stage_*` census.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StagingCensus {
    pub allocated: u64,
    pub allocated_bytes: u64,
    pub refusals: u64,
    pub refused_bytes: u64,
    pub released: u64,
    /// Chunks reset for reuse after their last reference dropped.
    pub recycled: u64,
}

#[derive(Clone, Copy)]
struct ChunkState {
    size: u64,
    used: u64,
    /// Outstanding references: one from the alloc (the fact's own) plus one
    /// per submission that recorded a read. Zero is the only state in which
    /// the chunk may be reset.
    outstanding: u64,
}

struct Inner {
    chunks: Vec<ChunkState>,
    /// The chunk the last allocation bumped, so a frame's windows stay in one
    /// chunk while it has room.
    cursor: usize,
    census: StagingCensus,
}

/// One thread's staging arena: chunks sub-allocated by bump, freed by
/// reference count. See the module doc for the lifetime model.
pub struct StagingRing {
    inner: Mutex<Inner>,
}

impl StagingRing {
    /// A ring of `chunks` chunks of `chunk` bytes each. Panics on a zero
    /// chunk or no chunks — a ring that can hold nothing is a caller error,
    /// and every later answer it gave would be a lie.
    pub fn new(chunk: u64, chunks: usize) -> Self {
        assert!(chunk > 0, "a chunk of no bytes stages nothing");
        assert!(chunks > 0, "an arena with no chunks stages nothing");
        Self {
            inner: Mutex::new(Inner {
                chunks: vec![
                    ChunkState {
                        size: chunk,
                        used: 0,
                        outstanding: 0,
                    };
                    chunks
                ],
                cursor: 0,
                census: StagingCensus::default(),
            }),
        }
    }

    /// Carve `len` bytes aligned to `alignment`, taking the fact's one
    /// reference. Exhaustion refuses; nothing is taken on any refusal.
    pub fn alloc(&self, len: u64, alignment: u64) -> Result<StagingWindow, StagingRefusal> {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let chunk_size = inner.chunks[0].size;
        if !alignment.is_power_of_two() {
            inner.census.refusals += 1;
            return Err(StagingRefusal::BadAlignment { alignment });
        }
        if len == 0 {
            inner.census.refusals += 1;
            return Err(StagingRefusal::Empty);
        }
        if len > chunk_size {
            inner.census.refusals += 1;
            inner.census.refused_bytes += len;
            return Err(StagingRefusal::TooLarge {
                requested: len,
                chunk: chunk_size,
            });
        }
        // The open chunk first — a pass's windows stay together while it has
        // room, and appending beside that chunk's still-in-flight windows is
        // sound because they are distinct bytes. A chunk with no room left or
        // one whose tail cannot fit the ask rolls to a chunk every reference
        // has left, resetting it; a chunk still holding references is not
        // considered, however long it has held them: the ring slots' fences
        // are the only thing that frees it.
        let cursor = inner.cursor;
        let mut target = bump(&mut inner, cursor, len, alignment);
        if target.is_none() {
            // The cursor chunk had no room at its tail. Any chunk every
            // reference has left may be reset — the cursor included, which is
            // what keeps a fully-bumped chunk whose windows all retired from
            // stranding the arena behind two less-used chunks.
            for index in 0..inner.chunks.len() {
                if inner.chunks[index].outstanding != 0 {
                    continue;
                }
                inner.chunks[index].used = 0;
                if let Some(window) = bump(&mut inner, index, len, alignment) {
                    inner.cursor = index;
                    inner.census.recycled += 1;
                    target = Some(window);
                    break;
                }
            }
        }
        match target {
            Some(window) => {
                inner.census.allocated += 1;
                inner.census.allocated_bytes += len;
                Ok(window)
            }
            None => {
                inner.census.refusals += 1;
                inner.census.refused_bytes += len;
                Err(StagingRefusal::Exhausted)
            }
        }
    }

    /// Add one reference: a submission is about to record a read of `window`.
    ///
    /// Returns whether it did. A `false` is a window whose chunk index left
    /// the ring (the ring was rebuilt behind a device loss); the caller drops
    /// the read and takes the serial path rather than arithmeticking over a
    /// chunk that no longer exists.
    pub fn retain(&self, window: StagingWindow) -> bool {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let Some(chunk) = inner.chunks.get_mut(window.chunk as usize) else {
            return false;
        };
        chunk.outstanding += 1;
        true
    }

    /// Drop one reference: the fact died, or the submission that read the
    /// window retired. The last reference out leaves the chunk reset and
    /// reusable.
    pub fn release(&self, window: StagingWindow) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let Some(chunk) = inner.chunks.get_mut(window.chunk as usize) else {
            return;
        };
        chunk.outstanding = chunk.outstanding.saturating_sub(1);
        if chunk.outstanding == 0 {
            chunk.used = 0;
        }
        inner.census.released += 1;
    }

    /// Sum of outstanding references over every chunk. Zero is "nothing the
    /// GPU may still read"; a teardown that asks it destroys nothing before
    /// it is true.
    pub fn in_flight(&self) -> u64 {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .chunks
            .iter()
            .map(|c| c.outstanding)
            .sum()
    }

    /// The census snapshot.
    pub fn census(&self) -> StagingCensus {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).census
    }
}

/// Bump `index`'s chunk for `len` bytes at `alignment`, taking the fact's
/// reference. `None` when the chunk cannot fit the ask.
///
/// A chunk holding references may still be bumped at its tail: the new window
/// is bytes no live window covers, and it carries its own reference, so the
/// invariants hold without freezing a chunk until its first windows retire.
fn bump(inner: &mut Inner, index: usize, len: u64, alignment: u64) -> Option<StagingWindow> {
    let chunk = inner.chunks.get_mut(index)?;
    let start = chunk.used.next_multiple_of(alignment);
    let end = start.checked_add(len)?;
    if end > chunk.size {
        return None;
    }
    chunk.used = end;
    chunk.outstanding += 1;
    Some(StagingWindow {
        chunk: index as u32,
        offset: start,
        len,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const CHUNK: u64 = 1024;

    /// The invariant the module exists for: a chunk whose references are
    /// outstanding is not handed out again, and the last reference out resets
    /// it whole.
    #[test]
    fn a_referenced_chunk_is_not_reused_until_the_last_release() {
        let ring = StagingRing::new(CHUNK, 2);
        let first = ring.alloc(64, 4).expect("room");
        // Fill the second chunk with references held, so a third allocation
        // has nowhere to go while both are live.
        let second = ring.alloc(CHUNK, 4).expect("the other chunk");
        assert_eq!(
            ring.alloc(4, 4),
            Err(StagingRefusal::Exhausted),
            "every chunk holds references; the timeline is the only release"
        );
        ring.release(first);
        let recycled = ring.alloc(CHUNK, 4).expect("the first chunk came back");
        assert_eq!(recycled.chunk, first.chunk);
        assert_eq!(recycled.offset, 0, "the whole chunk, reset");
        assert_eq!(ring.in_flight(), 2);
        ring.release(second);
        ring.release(recycled);
        assert_eq!(ring.in_flight(), 0);
    }

    /// The recorder-side half: one window, read by two submissions, survives
    /// the first submission's retire.
    #[test]
    fn a_window_read_by_two_submissions_outlives_the_first_retire() {
        let ring = StagingRing::new(CHUNK, 1);
        let window = ring.alloc(64, 4).expect("room");
        assert!(ring.retain(window), "the first submission records a read");
        assert!(
            ring.retain(window),
            "and the batch it joined records one too"
        );
        ring.release(window); // the fact's own reference dies at the frame pop
        ring.release(window); // the first submission retires
        assert_eq!(ring.in_flight(), 1, "the second submission still reads it");
        assert!(
            ring.alloc(CHUNK, 4).is_err(),
            "so the chunk is not reusable yet"
        );
        ring.release(window);
        assert_eq!(ring.in_flight(), 0);
        assert!(ring.alloc(CHUNK, 4).is_ok());
    }

    /// A window nobody recorded a read of is freed by its own reference
    /// alone — the invalid-plan path, where the fact dies unused.
    #[test]
    fn an_unused_window_is_freed_by_its_own_reference() {
        let ring = StagingRing::new(CHUNK, 1);
        let unused = ring.alloc(CHUNK, 4).expect("room");
        ring.release(unused);
        assert!(ring.alloc(CHUNK, 4).is_ok(), "nothing else held it");
    }

    /// Exhaustion refuses rather than resetting the oldest referenced chunk,
    /// and the refusal is counted with its bytes.
    #[test]
    fn exhaustion_refuses_rather_than_recycling_referenced_bytes() {
        let ring = StagingRing::new(CHUNK, 1);
        let live = ring.alloc(16, 4).expect("room");
        assert_eq!(ring.alloc(CHUNK, 4), Err(StagingRefusal::Exhausted));
        let census = ring.census();
        assert_eq!(census.refusals, 1);
        assert_eq!(census.refused_bytes, CHUNK);
        // And the live window was not disturbed.
        assert_eq!(ring.in_flight(), 1);
        ring.release(live);
    }

    /// Bumps stay inside one chunk while it has room, respect alignment, and
    /// a tail append beside in-flight windows of the same chunk is allowed
    /// (distinct bytes, own reference).
    #[test]
    fn bumps_stay_in_one_chunk_and_align() {
        let ring = StagingRing::new(CHUNK, 2);
        let a = ring.alloc(10, 4).expect("room");
        let b = ring.alloc(10, 32).expect("room");
        assert_eq!(a.chunk, b.chunk);
        assert_eq!(a.offset, 0);
        assert_eq!(b.offset, 32);
        // The same chunk may take more while `a` is in flight.
        ring.retain(a);
        let c = ring.alloc(8, 8).expect("tail room");
        assert_eq!(c.chunk, a.chunk);
        assert_eq!(c.offset, 48);
    }

    /// A full cursor chunk rolls to the next free one and stays there.
    #[test]
    fn a_full_chunk_rolls_and_stays() {
        let ring = StagingRing::new(CHUNK, 2);
        let _fill = ring.alloc(CHUNK, 4).expect("room");
        let rolled = ring.alloc(4, 4).expect("the other chunk");
        assert_eq!(rolled.chunk, 1);
        assert_eq!(rolled.offset, 0);
        assert_eq!(ring.alloc(4, 4).expect("stays").chunk, 1);
    }

    /// A request larger than a chunk names itself, and a zero or badly
    /// aligned one is refused without touching the ring.
    #[test]
    fn refusals_name_themselves() {
        let ring = StagingRing::new(CHUNK, 2);
        assert_eq!(
            ring.alloc(CHUNK + 1, 4),
            Err(StagingRefusal::TooLarge {
                requested: CHUNK + 1,
                chunk: CHUNK
            })
        );
        assert_eq!(
            ring.alloc(4, 3),
            Err(StagingRefusal::BadAlignment { alignment: 3 })
        );
        assert_eq!(ring.alloc(0, 4), Err(StagingRefusal::Empty));
        assert_eq!(ring.census().allocated, 0);
        assert_eq!(ring.census().refusals, 3);
    }

    /// A window naming a chunk past the ring (a rebuilt device) is refused
    /// its retain and ignored by its release, rather than indexing another
    /// chunk's bytes.
    #[test]
    fn a_window_past_the_ring_is_refused_rather_than_miscounted() {
        let ring = StagingRing::new(CHUNK, 1);
        let alien = StagingWindow {
            chunk: 9,
            offset: 0,
            len: 4,
        };
        assert!(!ring.retain(alien));
        ring.release(alien);
        assert_eq!(ring.in_flight(), 0);
    }

    /// A driven history: allocations, retains, releases and resets never hand
    /// out overlapping live bytes and never reuse referenced chunks — the two
    /// invariants whose violation is a wrong frame with no error anywhere.
    #[test]
    fn a_driven_history_never_overlaps_or_reuses_referenced_chunks() {
        struct Rng(u64);
        impl Rng {
            const fn new(seed: u64) -> Self {
                Self(seed ^ 0x9E37_79B9_7F4A_7C15)
            }
            fn next(&mut self) -> u64 {
                self.0 ^= self.0 << 13;
                self.0 ^= self.0 >> 7;
                self.0 ^= self.0 << 17;
                self.0
            }
            fn below(&mut self, bound: u64) -> u64 {
                if bound == 0 {
                    0
                } else {
                    self.next() % bound
                }
            }
        }
        const CHUNKS: usize = 3;
        let mut rng = Rng::new(0x1234_5678);
        let (mut allocs, mut exhaustions) = (0u64, 0u64);
        for _case in 0..200 {
            let ring = StagingRing::new(CHUNK, CHUNKS);
            // The shadow: per chunk, the live windows and the outstanding
            // reference count, derived from the calls and nothing else.
            let mut live: Vec<Vec<(u64, u64)>> = vec![Vec::new(); CHUNKS];
            let mut outstanding = vec![0u64; CHUNKS];
            // (window, references it still holds). A handle dies with its
            // last reference -- the same bytes may be handed out again later,
            // as a fresh handle with a fresh count.
            let mut handles: Vec<(StagingWindow, u64)> = Vec::new();
            for _step in 0..120 {
                match rng.below(4) {
                    0 | 1 => {
                        let len = 1 + rng.below(CHUNK / 3);
                        let align = 1u64 << rng.below(4);
                        match ring.alloc(len, align) {
                            Ok(window) => {
                                assert_eq!(window.len, len);
                                assert_eq!(window.offset % align, 0);
                                assert!(window.offset + window.len <= CHUNK);
                                let chunk = &mut live[window.chunk as usize];
                                for &(start, end) in chunk.iter() {
                                    assert!(
                                        window.offset + window.len <= start || window.offset >= end,
                                        "new {}..{} overlaps live {start}..{end}",
                                        window.offset,
                                        window.offset + window.len
                                    );
                                }
                                chunk.push((window.offset, window.offset + window.len));
                                outstanding[window.chunk as usize] += 1;
                                handles.push((window, 1));
                                allocs += 1;
                            }
                            Err(StagingRefusal::Exhausted) => {
                                // Only legal when every chunk holds references
                                // or cannot fit the ask — with lengths a third
                                // of the chunk, a released chunk always fits.
                                assert!(
                                    (0..CHUNKS).all(|i| outstanding[i] > 0),
                                    "refused while a chunk was fully released"
                                );
                                exhaustions += 1;
                            }
                            Err(other) => panic!("unexpected refusal {other:?}"),
                        }
                    }
                    2 => {
                        // A submission records a read of a random live window.
                        if !handles.is_empty() {
                            let at = rng.below(handles.len() as u64) as usize;
                            let (window, refs) = handles[at];
                            if ring.retain(window) {
                                handles[at].1 = refs + 1;
                                outstanding[window.chunk as usize] += 1;
                            }
                        }
                    }
                    _ => {
                        // A reference drops: the fact's or a submission's.
                        // Only handles that still hold one may drop it.
                        let held: Vec<usize> =
                            (0..handles.len()).filter(|&i| handles[i].1 > 0).collect();
                        if !held.is_empty() {
                            let at = held[rng.below(held.len() as u64) as usize];
                            let (window, refs) = handles[at];
                            ring.release(window);
                            handles[at].1 = refs - 1;
                            let chunk = window.chunk as usize;
                            outstanding[chunk] -= 1;
                            if outstanding[chunk] == 0 {
                                live[chunk].clear();
                            }
                        }
                    }
                }
                assert_eq!(
                    ring.in_flight(),
                    outstanding.iter().sum::<u64>(),
                    "the ring's flight matches the calls"
                );
            }
        }
        assert!(allocs > 4_000, "{allocs} allocations were driven");
        assert!(exhaustions > 400, "{exhaustions} exhaustions were driven");
    }
}
