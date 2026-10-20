//! TEMP (lab, REC_LANE, design journal/2026-10-10-parallel-recorder-design.md):
//! the recorder's command recording moved to a second thread.
//!
//! The recorder thread runs `execute_draw_inner` as before - every decision,
//! every registry read and write, every allocation, in FIFO order - but while a
//! draw is being executed ([`activate`]) the `vkCmd*` calls it makes are not
//! made: each is copied, with everything it points at, into a [`List`], and the
//! draw's list is handed to the *lane*, which replays it into the same command
//! buffer on its own thread. Recording is write-only, so nothing the recorder
//! decides afterwards depends on a command having been recorded; what does
//! depend on it is the command buffer's lifecycle, and [`sync`] - the lane has
//! replayed everything handed to it - stands in front of each of those:
//! `vkEndCommandBuffer` (`batch_flush`, the immediate submits), the reset of the
//! next slot (`begin_slot_recording`, the other use of the ring's command pool,
//! which Vulkan requires externally synchronized), the recycle of staging the
//! lane may still copy into, and every `vkCmd*` made outside a draw (each one
//! syncs first, so order and exclusive use of the pool both hold).
//!
//! TEMP (lab, REC_LANES / REC_LANE_END, 2026-10-12): there are N lanes
//! (`REIMS_VGPU_LAB_REC_LANES`, default 1), each fixed for a share of the ring's
//! command buffers, each of which comes from its own pool, so lanes record
//! different buffers at once. Inside a draw (and in the drain's batch flush,
//! [`scope`]) the buffer's reset/begin and end are lane commands too, and the
//! submission carries a [`Gate`] - the list holding the end - which the queue
//! owner waits for before `vkQueueSubmit`. Submit order stays the order the
//! submissions were handed to the owner, and the recorder no longer waits for a
//! lane at a batch boundary while it holds the engine lock (b270/b274: ~18 % of
//! its busy time on heavy Maps, and the drain blocked on that lock ~160 ms/s).
//!
//! A list is an arena: the commands name ranges of typed side arrays, and lists
//! come back from the lane cleared, so a steady stream allocates nothing. The
//! lane spins briefly before it parks, and the recorder wakes it only from a
//! park, so a burst of draws costs no system call per draw.
//!
//! Off (`REIMS_VGPU_LAB_REC_LANE=off`, live `REC_LANE`): every function here is
//! the `ash` call it wraps, made on the caller's thread.

use ash::vk;
use parking_lot::{Condvar, Mutex};
use std::cell::{Cell, RefCell};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

/// How long the lane polls for the next list before it parks.
const LANE_SPIN: Duration = Duration::from_micros(80);
/// How long a [`sync`] polls before it sleeps.
const SYNC_SPIN: Duration = Duration::from_micros(300);

/// A range of one of a [`List`]'s side arrays.
#[derive(Clone, Copy)]
struct R(u32, u32);

impl R {
    fn put<T: Copy>(v: &mut Vec<T>, s: &[T]) -> R {
        let start = v.len() as u32;
        v.extend_from_slice(s);
        R(start, s.len() as u32)
    }

    fn of<T>(self, v: &[T]) -> &[T] {
        &v[self.0 as usize..(self.0 + self.1) as usize]
    }
}

/// One recorded command; what its `ash` call borrowed lives in the list.
enum Cmd {
    Barrier(
        vk::CommandBuffer,
        vk::PipelineStageFlags,
        vk::PipelineStageFlags,
        vk::DependencyFlags,
        R,
        R,
        R,
    ),
    BeginPass(
        vk::CommandBuffer,
        vk::RenderPass,
        vk::Framebuffer,
        vk::Rect2D,
        R,
        vk::SubpassContents,
    ),
    EndPass(vk::CommandBuffer),
    BindPipeline(vk::CommandBuffer, vk::PipelineBindPoint, vk::Pipeline),
    BindSets(
        vk::CommandBuffer,
        vk::PipelineBindPoint,
        vk::PipelineLayout,
        u32,
        R,
        R,
    ),
    PushSet(
        vk::CommandBuffer,
        vk::PipelineBindPoint,
        vk::PipelineLayout,
        u32,
        R,
    ),
    BindVertex(vk::CommandBuffer, u32, R, R),
    BindIndex(vk::CommandBuffer, vk::Buffer, vk::DeviceSize, vk::IndexType),
    Draw(vk::CommandBuffer, u32, u32, u32, u32),
    DrawIndexed(vk::CommandBuffer, u32, u32, u32, i32, u32),
    CopyBuffer(vk::CommandBuffer, vk::Buffer, vk::Buffer, R),
    CopyBufferToImage(vk::CommandBuffer, vk::Buffer, vk::Image, vk::ImageLayout, R),
    CopyImage(
        vk::CommandBuffer,
        vk::Image,
        vk::ImageLayout,
        vk::Image,
        vk::ImageLayout,
        R,
    ),
    CopyImageToBuffer(vk::CommandBuffer, vk::Image, vk::ImageLayout, vk::Buffer, R),
    ClearColor(
        vk::CommandBuffer,
        vk::Image,
        vk::ImageLayout,
        vk::ClearColorValue,
        R,
    ),
    BeginQuery(vk::CommandBuffer, vk::QueryPool, u32, vk::QueryControlFlags),
    EndQuery(vk::CommandBuffer, vk::QueryPool, u32),
    ResetQuery(vk::CommandBuffer, vk::QueryPool, u32, u32),
    Timestamp(
        vk::CommandBuffer,
        vk::PipelineStageFlags,
        vk::QueryPool,
        u32,
    ),
    Viewport(vk::CommandBuffer, u32, R),
    Scissor(vk::CommandBuffer, u32, R),
    BlendConstants(vk::CommandBuffer, [f32; 4]),
    StencilReference(vk::CommandBuffer, vk::StencilFaceFlags, u32),
    DepthBias(vk::CommandBuffer, f32, f32, f32),
    LineWidth(vk::CommandBuffer, f32),
    CullMode(vk::CommandBuffer, vk::CullModeFlags),
    FrontFace(vk::CommandBuffer, vk::FrontFace),
    Topology(vk::CommandBuffer, vk::PrimitiveTopology),
    PolygonMode(vk::CommandBuffer, vk::PolygonMode),
    DepthClampEnable(vk::CommandBuffer, bool),
    DepthTest(vk::CommandBuffer, bool),
    DepthWrite(vk::CommandBuffer, bool),
    DepthCompare(vk::CommandBuffer, vk::CompareOp),
    DepthBoundsTest(vk::CommandBuffer, bool),
    StencilTest(vk::CommandBuffer, bool),
    StencilOp(
        vk::CommandBuffer,
        vk::StencilFaceFlags,
        vk::StencilOp,
        vk::StencilOp,
        vk::StencilOp,
        vk::CompareOp,
    ),
    StencilCompareMask(vk::CommandBuffer, vk::StencilFaceFlags, u32),
    StencilWriteMask(vk::CommandBuffer, vk::StencilFaceFlags, u32),
    PushConstants(
        vk::CommandBuffer,
        vk::PipelineLayout,
        vk::ShaderStageFlags,
        u32,
        R,
    ),
    Dispatch(vk::CommandBuffer, u32, u32, u32),
    DispatchBase(vk::CommandBuffer, u32, u32, u32, u32, u32, u32),
    FillBuffer(
        vk::CommandBuffer,
        vk::Buffer,
        vk::DeviceSize,
        vk::DeviceSize,
        u32,
    ),
    /// Host copies into a mapped staging span (`write_staging_from_runs`).
    HostCopy(R),
    /// TEMP (lab, REC_LANE_END): `vkResetCommandBuffer` + `vkBeginCommandBuffer`
    /// (one-time submit) of a ring slot's command buffer.
    Begin(vk::CommandBuffer),
    /// TEMP (lab, REC_LANE_END): `vkEndCommandBuffer`; the submission that
    /// carries the buffer waits for it at the queue owner ([`Gate`]).
    End(vk::CommandBuffer),
}

/// A push-descriptor write; its infos are a range of `iinfos` or `binfos`.
#[derive(Clone, Copy)]
struct Write {
    binding: u32,
    array_element: u32,
    ty: vk::DescriptorType,
    count: u32,
    images: R,
    buffers: R,
}

/// One host copy: a guest-RAM source (a stable RAMBlock alias) and a disjoint
/// destination in a staging slot the command buffer holds until it completes.
#[derive(Clone, Copy)]
pub(crate) struct HostSeg {
    pub(crate) src: *const u8,
    pub(crate) dst: *mut u8,
    pub(crate) n: usize,
}

#[derive(Default)]
struct List {
    cmds: Vec<Cmd>,
    mem: Vec<vk::MemoryBarrier<'static>>,
    buf: Vec<vk::BufferMemoryBarrier<'static>>,
    img: Vec<vk::ImageMemoryBarrier<'static>>,
    clears: Vec<vk::ClearValue>,
    sets: Vec<vk::DescriptorSet>,
    u32s: Vec<u32>,
    buffers: Vec<vk::Buffer>,
    sizes: Vec<vk::DeviceSize>,
    bcopies: Vec<vk::BufferCopy>,
    bicopies: Vec<vk::BufferImageCopy>,
    icopies: Vec<vk::ImageCopy>,
    ranges: Vec<vk::ImageSubresourceRange>,
    viewports: Vec<vk::Viewport>,
    rects: Vec<vk::Rect2D>,
    writes: Vec<Write>,
    iinfos: Vec<vk::DescriptorImageInfo>,
    binfos: Vec<vk::DescriptorBufferInfo>,
    bytes: Vec<u8>,
    segs: Vec<HostSeg>,
    /// Scratch for building a replayed push's `VkWriteDescriptorSet`s.
    vk_writes: Vec<vk::WriteDescriptorSet<'static>>,
}

// SAFETY: the handles name objects of one device, used by one thread at a time
// (the lane replays a list only after the recorder handed it over and before
// the recorder's next `sync` returns); the host-copy pointers obey
// [`host_copy`]'s contract; `vk_writes` is empty between replays.
unsafe impl Send for List {}

impl List {
    fn clear(&mut self) {
        self.cmds.clear();
        self.mem.clear();
        self.buf.clear();
        self.img.clear();
        self.clears.clear();
        self.sets.clear();
        self.u32s.clear();
        self.buffers.clear();
        self.sizes.clear();
        self.bcopies.clear();
        self.bicopies.clear();
        self.icopies.clear();
        self.ranges.clear();
        self.viewports.clear();
        self.rects.clear();
        self.writes.clear();
        self.iinfos.clear();
        self.binfos.clear();
        self.bytes.clear();
        self.segs.clear();
        self.vk_writes.clear();
    }
}

struct Loaders {
    device: ash::Device,
    push: Option<ash::khr::push_descriptor::Device>,
    eds: Option<ash::ext::extended_dynamic_state::Device>,
    eds3: Option<ash::ext::extended_dynamic_state3::Device>,
}

struct LoadersCell(Loaders);
// SAFETY: `ash` loader tables are plain function pointers and handles.
unsafe impl Send for LoadersCell {}
unsafe impl Sync for LoadersCell {}

#[derive(Default)]
struct Queue {
    ready: VecDeque<List>,
    free: Vec<List>,
    parked: bool,
}

struct Lane {
    queue: Mutex<Option<Queue>>,
    work: Condvar,
    idle: Condvar,
    pushed: AtomicU64,
    done: AtomicU64,
    waiters: AtomicU32,
    busy_ns: AtomicU64,
    lists: AtomicU64,
}

impl Lane {
    const fn new() -> Self {
        Self {
            queue: Mutex::new(None),
            work: Condvar::new(),
            idle: Condvar::new(),
            pushed: AtomicU64::new(0),
            done: AtomicU64::new(0),
            waiters: AtomicU32::new(0),
            busy_ns: AtomicU64::new(0),
            lists: AtomicU64::new(0),
        }
    }
}

/// TEMP (lab, REC_LANES): the most lanes [`lane_count`] may ask for.
const MAX_LANES: usize = 4;

/// TEMP (lab, REC_LANES): one lane per group of ring slots. A command buffer
/// always replays on the same lane ([`register_cb`]), so a lane is the only
/// thread recording into it; the ring's command buffers each come from their
/// own pool, so two lanes recording two of them at once touch no pool in
/// common. Lists for different command buffers have no order between them but
/// the submit order, which the queue owner keeps ([`Gate`]).
static LANES: [Lane; MAX_LANES] = [const { Lane::new() }; MAX_LANES];
static LOADERS: OnceLock<LoadersCell> = OnceLock::new();
/// The loaders the lanes replay with now: the first context's, until a
/// device recreate binds the new device's (see [`rebind`]). Read once per list.
static CURRENT: parking_lot::RwLock<Option<std::sync::Arc<LoadersCell>>> =
    parking_lot::RwLock::new(None);
/// Lanes running (0 until the first [`activate`]).
static RUNNING: AtomicUsize = AtomicUsize::new(0);

/// TEMP (lab, REC_LANES): the command buffers a lane is fixed for, as raw
/// handles, and its index. Read only when a thread's commands move to a
/// different command buffer than its previous command named (~once a batch).
static CB_LANES: Mutex<Vec<(u64, u8)>> = Mutex::new(Vec::new());
/// Bumped by every [`register_cb`], so a thread's cached routing (`CUR_CB`)
/// cannot outlive a re-registration of the same handle.
static CB_GEN: AtomicU64 = AtomicU64::new(1);

/// TEMP (lab, REC_LANES, `REIMS_VGPU_LAB_REC_LANES=N`, 1..=4, default 1): how many
/// lanes replay. Read once; the ring's command buffers are dealt to them in
/// slot order. b284: with the lifecycle on the lanes one lane is ~20 % busy on
/// heavy Maps and two were no faster, so one is the default (a lane spins
/// between lists, a cost on a host the 24 vCPUs share).
pub(crate) fn lane_count() -> usize {
    if cfg!(test) {
        return 1;
    }
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("REIMS_VGPU_LAB_REC_LANES")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(1)
            .clamp(1, MAX_LANES)
    })
}

/// TEMP (lab, REC_LANES): fix the lane `cb` replays on - `slot % lane_count()`.
/// Called when the ring's command buffers are allocated (a handle reused by a
/// later ring simply takes its new slot's lane: the old ring was drained).
pub(crate) fn register_cb(cb: vk::CommandBuffer, slot: usize) {
    use ash::vk::Handle;
    let lane = (slot % lane_count()) as u8;
    let raw = cb.as_raw();
    let mut map = CB_LANES.lock();
    match map.iter_mut().find(|(h, _)| *h == raw) {
        Some(e) => e.1 = lane,
        None => map.push((raw, lane)),
    }
    CB_GEN.fetch_add(1, Ordering::AcqRel);
}

fn lane_of(cb: vk::CommandBuffer) -> u8 {
    use ash::vk::Handle;
    let raw = cb.as_raw();
    CB_LANES
        .lock()
        .iter()
        .find(|(h, _)| *h == raw)
        .map_or(0, |(_, lane)| *lane)
        .min((RUNNING.load(Ordering::Acquire).max(1) - 1) as u8)
}

/// TEMP (lab, REC_LANE_END): the first `vkReset/Begin/EndCommandBuffer` a lane
/// saw fail. Every submission waits its gate and then reads this: a buffer
/// that did not record must not reach the queue, and the failure poisons the
/// context like an asynchronous submit failure does.
static LANE_FAILURE: Mutex<Option<vk::Result>> = Mutex::new(None);

fn note_lane_failure(what: &str, e: vk::Result) {
    let mut f = LANE_FAILURE.lock();
    if f.is_none() {
        *f = Some(e);
        crate::observe::fail(format!(
            "vk_rec_lane_failed reason=vk_rec_lane_failed op={what} result={e:?}"
        ));
        super::device_lost::note_device_lost_seen();
    }
}

/// TEMP (lab, REC_LANE_END): the point a submission waits for - lane `lane`
/// has replayed its first `seq` lists, the last of which ends the command
/// buffer being submitted.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Gate {
    lane: u8,
    seq: u64,
}

/// TEMP (lab, REC_LANE_END): wait at the queue owner until `gate`'s list has
/// been replayed, then refuse the submission if any lane failed to record.
pub(crate) fn wait_gate(gate: Gate) -> Result<(), vk::Result> {
    let lane = &LANES[gate.lane as usize];
    if lane.done.load(Ordering::Acquire) < gate.seq {
        let started = Instant::now();
        wait_done(lane, gate.seq);
        STATS
            .gate_wait_ns
            .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        STATS.gate_waits.fetch_add(1, Ordering::Relaxed);
    }
    match *LANE_FAILURE.lock() {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// Spin, then sleep, until `lane` has replayed `target` lists.
fn wait_done(lane: &Lane, target: u64) {
    let started = Instant::now();
    let mut polls = 0u32;
    while lane.done.load(Ordering::Acquire) < target {
        std::hint::spin_loop();
        polls += 1;
        if polls % 64 == 0 && started.elapsed() >= SYNC_SPIN {
            break;
        }
    }
    if lane.done.load(Ordering::SeqCst) < target {
        lane.waiters.fetch_add(1, Ordering::SeqCst);
        let mut q = lane.queue.lock();
        while lane.done.load(Ordering::SeqCst) < target {
            lane.idle.wait_for(&mut q, Duration::from_micros(200));
        }
        drop(q);
        lane.waiters.fetch_sub(1, Ordering::SeqCst);
    }
}

/// TEMP (lab, REC_LANES): what the lanes did, for the recorder census line.
struct Stats {
    sync_ns: AtomicU64,
    syncs: AtomicU64,
    gate_wait_ns: AtomicU64,
    gate_waits: AtomicU64,
    deferred_ends: AtomicU64,
    copy_bytes: AtomicU64,
}

static STATS: Stats = Stats {
    sync_ns: AtomicU64::new(0),
    syncs: AtomicU64::new(0),
    gate_wait_ns: AtomicU64::new(0),
    gate_waits: AtomicU64::new(0),
    deferred_ends: AtomicU64::new(0),
    copy_bytes: AtomicU64::new(0),
};

/// TEMP (lab, REC_LANES): the lanes' window, as the recorder census's tail
/// (`lanes=N lane_busy_us=a/b.. lane_lists=.. sync_n= sync_us= gate_n=
/// gate_us= ends= copy_mb=`), or empty before the lanes run.
pub(crate) fn take_window_text() -> String {
    let n = RUNNING.load(Ordering::Acquire);
    if n == 0 {
        return String::new();
    }
    let busy: Vec<String> = LANES[..n]
        .iter()
        .map(|l| (l.busy_ns.swap(0, Ordering::Relaxed) / 1000).to_string())
        .collect();
    let lists: u64 = LANES[..n]
        .iter()
        .map(|l| l.lists.swap(0, Ordering::Relaxed))
        .sum();
    format!(
        " lanes={n} lane_busy_us={} lane_lists={lists} lane_sync_n={} lane_sync_us={} \
         lane_gate_n={} lane_gate_us={} lane_ends={} lane_copy_mb={}",
        busy.join("/"),
        STATS.syncs.swap(0, Ordering::Relaxed),
        STATS.sync_ns.swap(0, Ordering::Relaxed) / 1000,
        STATS.gate_waits.swap(0, Ordering::Relaxed),
        STATS.gate_wait_ns.swap(0, Ordering::Relaxed) / 1000,
        STATS.deferred_ends.swap(0, Ordering::Relaxed),
        STATS.copy_bytes.swap(0, Ordering::Relaxed) >> 20,
    ) + &copy_repeat::take_window_text()
}

/// TEMP (lab, `REIMS_VGPU_LAB_COPY_REPEAT=on`): how much of the lane's guest
/// RAM -> staging copying repeats bytes it already copied. Maps streams ~2.7
/// GB/s through these copies (b651, ~68 MB a frame); the question is how much
/// of it is the same source span with the same content as its previous copy
/// (a resident GPU copy would skip it) against new or changed bytes. Each
/// segment's source is hashed on the lane, keyed by (address, length); the
/// census tail adds `copy_repeat_mb= copy_changed_mb= copy_new_mb=`.
/// Measure only: the copies themselves are untouched.
mod copy_repeat {
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Mutex, OnceLock};

    static REPEAT: AtomicU64 = AtomicU64::new(0);
    static CHANGED: AtomicU64 = AtomicU64::new(0);
    static NEW: AtomicU64 = AtomicU64::new(0);

    pub(super) fn on() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| std::env::var("REIMS_VGPU_LAB_COPY_REPEAT").as_deref() == Ok("on"))
    }

    fn seen() -> &'static Mutex<HashMap<(usize, usize), u64>> {
        static SEEN: OnceLock<Mutex<HashMap<(usize, usize), u64>>> = OnceLock::new();
        SEEN.get_or_init(|| Mutex::new(HashMap::new()))
    }

    /// A multiply-xor digest over 8-byte words: ~10 GB/s, enough to tell a
    /// repeat from a change for a census (a collision only miscounts).
    ///
    /// # Safety
    /// `src` is readable for `n` bytes.
    unsafe fn digest(src: *const u8, n: usize) -> u64 {
        let mut h = 0x9e37_79b9_7f4a_7c15u64 ^ n as u64;
        let words = n / 8;
        for i in 0..words {
            // SAFETY: i * 8 + 8 <= n.
            let w = unsafe { (src.add(i * 8) as *const u64).read_unaligned() };
            h = (h ^ w).wrapping_mul(0x0100_0000_01b3).rotate_left(29);
        }
        for i in words * 8..n {
            // SAFETY: i < n.
            h = (h ^ u64::from(unsafe { *src.add(i) })).wrapping_mul(0x0100_0000_01b3);
        }
        h
    }

    /// # Safety
    /// `src` is readable for `n` bytes.
    pub(super) unsafe fn note(src: *const u8, n: usize) {
        // SAFETY: forwarded.
        let h = unsafe { digest(src, n) };
        let mut seen = seen().lock().unwrap_or_else(|e| e.into_inner());
        if seen.len() >= 1 << 20 {
            seen.clear();
        }
        let counter = match seen.insert((src as usize, n), h) {
            Some(old) if old == h => &REPEAT,
            Some(_) => &CHANGED,
            None => &NEW,
        };
        counter.fetch_add(n as u64, Ordering::Relaxed);
    }

    pub(super) fn take_window_text() -> String {
        if !on() {
            return String::new();
        }
        format!(
            " copy_repeat_mb={} copy_changed_mb={} copy_new_mb={}",
            REPEAT.swap(0, Ordering::Relaxed) >> 20,
            CHANGED.swap(0, Ordering::Relaxed) >> 20,
            NEW.swap(0, Ordering::Relaxed) >> 20,
        )
    }
}

thread_local! {
    static ACTIVE: Cell<bool> = const { Cell::new(false) };
    static LOCAL: RefCell<List> = RefCell::new(List::default());
    /// The lane this thread's pending list goes to.
    static LOCAL_LANE: Cell<u8> = const { Cell::new(0) };
    /// The command buffer this thread's last command named (raw handle), and
    /// the [`CB_GEN`] its lane was looked up at.
    static CUR_CB: Cell<(u64, u64)> = const { Cell::new((0, 0)) };
}

/// Whether the lane is on (`REIMS_VGPU_LAB_REC_LANE=off` / live `REC_LANE=off`).
pub(crate) fn enabled() -> bool {
    if cfg!(test) {
        return false;
    }
    if let Some(on) = crate::runtime::lab_live::on("REC_LANE") {
        return on;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("REIMS_VGPU_LAB_REC_LANE").as_deref() != Ok("off"))
}

/// TEMP (lab, REC_LANE_END, live `REC_LANE_END`, else
/// `REIMS_VGPU_LAB_REC_LANE_END=off`): whether a command buffer's reset, begin
/// and end go to its lane too, so the recorder never waits for a lane: the
/// submission waits instead, on the queue owner's thread ([`Gate`]).
fn lifecycle_on() -> bool {
    if let Some(on) = crate::runtime::lab_live::on("REC_LANE_END") {
        return on;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("REIMS_VGPU_LAB_REC_LANE_END").as_deref() != Ok("off"))
}

/// TEMP (lab, REC_LANE_END): whether this thread defers a command buffer's
/// reset/begin ([`begin`]) and end ([`end`]) to its lane.
pub(crate) fn defer_lifecycle() -> bool {
    active() && lifecycle_on()
}

/// Defer this thread's `vkCmd*` calls to the lane until the guard drops (which
/// hands the list over). A no-op when the lane is off. `ctx` provides the
/// loaders the lane replays with, cloned once.
pub(crate) fn activate(ctx: &super::context::DeviceContext) -> ActiveGuard {
    if !enabled() {
        return ActiveGuard(false);
    }
    LOADERS.get_or_init(|| {
        let n = lane_count();
        for lane in &LANES[..n] {
            *lane.queue.lock() = Some(Queue::default());
        }
        let cell = LoadersCell(Loaders {
            device: ctx.device.clone(),
            push: ctx.push_descriptor.clone(),
            eds: ctx.extended_dynamic_state.clone(),
            eds3: ctx.extended_dynamic_state3.clone(),
        });
        for i in 0..n {
            let name = if i == 0 {
                "reims-vgpu-rec-lane".to_string()
            } else {
                format!("reims-vgpu-rec-lane{i}")
            };
            std::thread::Builder::new()
                .name(name)
                .spawn(move || lane_main(i))
                .expect("spawn a recording lane");
        }
        RUNNING.store(n, Ordering::Release);
        cell
    });
    rebind(ctx);
    if ACTIVE.with(Cell::get) {
        return ActiveGuard(false);
    }
    ACTIVE.with(|a| a.set(true));
    ActiveGuard(true)
}

/// Bind the lanes to `ctx`'s device when it is not the one they replay with.
///
/// Both lane statics outlived the device they were set for. The loaders were
/// cloned once, so after a device recreate the lanes replayed the new device's
/// command buffers through the old device's dispatch (invalid, and wrong under
/// a layer that keys its tables by device). And the failure latch was never
/// cleared, so one failed reset/begin/end refused every gated submission
/// forever: the new device's work too, each refusal a device loss, each
/// recreate undone by the next submission - no guest GPU work until QEMU
/// restarted. A new device has new command buffers; what failed on the old one
/// says nothing about them.
fn rebind(ctx: &super::context::DeviceContext) {
    use ash::vk::Handle;
    let want = ctx.device.handle().as_raw();
    let bound = CURRENT.read().as_ref().map(|c| c.0.device.handle().as_raw());
    if bound == Some(want) {
        return;
    }
    // Everything handed over for the old device replays through its loaders.
    sync();
    *CURRENT.write() = Some(std::sync::Arc::new(LoadersCell(Loaders {
        device: ctx.device.clone(),
        push: ctx.push_descriptor.clone(),
        eds: ctx.extended_dynamic_state.clone(),
        eds3: ctx.extended_dynamic_state3.clone(),
    })));
    if bound.is_some() {
        let cleared = LANE_FAILURE.lock().take();
        crate::observe::off(format!(
            "vk_rec_lane_rebound device={want:#x} cleared_failure={cleared:?}"
        ));
    }
}

/// TEMP (lab, REC_LANE_END): [`activate`] for a caller that ends a command
/// buffer outside a draw (the drain's batch flush) - a no-op unless the lanes
/// already run and the lifecycle goes to them, or when this thread is active.
pub(crate) fn scope() -> ActiveGuard {
    if LOADERS.get().is_none() || !enabled() || !lifecycle_on() || ACTIVE.with(Cell::get) {
        return ActiveGuard(false);
    }
    ACTIVE.with(|a| a.set(true));
    ActiveGuard(true)
}

pub(crate) struct ActiveGuard(bool);

impl Drop for ActiveGuard {
    fn drop(&mut self) {
        if self.0 {
            hand_over();
            ACTIVE.with(|a| a.set(false));
        }
    }
}

/// Whether a staging copy made now may be handed to the lane instead (live
/// `REC_LANE_COPY`, on with the lane).
pub(crate) fn defer_host_copies() -> bool {
    if !active() {
        return false;
    }
    if let Some(on) = crate::runtime::lab_live::on("REC_LANE_COPY") {
        return on;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("REIMS_VGPU_LAB_REC_LANE_COPY").as_deref() != Ok("off"))
}

/// Hand the lane a host copy to make before the command buffer ends. It goes
/// to the lane of the command buffer this thread last recorded into - the one
/// whose staging it fills.
///
/// # Safety
///
/// The source stays readable and the destination writable and unaliased
/// until the next [`sync`] returns, or until the submission of the command
/// buffer it is recorded with has passed its gate.
pub(crate) unsafe fn host_copy(seg: HostSeg) {
    debug_assert!(active());
    LOCAL.with(|l| {
        let mut l = l.borrow_mut();
        let r = R::put(&mut l.segs, &[seg]);
        l.cmds.push(Cmd::HostCopy(r));
    });
}

/// TEMP (lab, REC_LANE_END): reset and begin `cb` on its lane.
pub(crate) fn begin(cb: vk::CommandBuffer) {
    debug_assert!(defer_lifecycle());
    rec(cb, |l| l.cmds.push(Cmd::Begin(cb)));
}

/// TEMP (lab, REC_LANE_END): end `cb` on its lane and hand the list over now;
/// the returned gate is what its submission waits for.
pub(crate) fn end(cb: vk::CommandBuffer) -> Gate {
    debug_assert!(defer_lifecycle());
    rec(cb, |l| l.cmds.push(Cmd::End(cb)));
    STATS.deferred_ends.fetch_add(1, Ordering::Relaxed);
    hand_over().expect("a list holding the end")
}

/// A command about to be recorded on the caller's thread: the lane must be
/// done with everything handed to it first (it may be replaying into this very
/// command buffer, and Vulkan wants the pool externally synchronized).
fn direct() {
    sync();
}

fn active() -> bool {
    ACTIVE.with(Cell::get)
}

/// Append to this thread's list for `cb`'s lane; a list for another lane is
/// handed over first.
fn rec(cb: vk::CommandBuffer, f: impl FnOnce(&mut List)) {
    use ash::vk::Handle;
    let key = (cb.as_raw(), CB_GEN.load(Ordering::Acquire));
    if CUR_CB.with(Cell::get) != key {
        CUR_CB.with(|c| c.set(key));
        let lane = lane_of(cb);
        if LOCAL_LANE.with(Cell::get) != lane {
            hand_over();
            LOCAL_LANE.with(|c| c.set(lane));
        }
    }
    LOCAL.with(|l| f(&mut l.borrow_mut()));
}

/// Give this thread's pending list to its lane, taking a cleared one back:
/// the lane and the list's sequence number there, or `None` when it was empty.
fn hand_over() -> Option<Gate> {
    LOCAL.with(|l| {
        let mut local = l.borrow_mut();
        if local.cmds.is_empty() {
            return None;
        }
        let index = LOCAL_LANE.with(Cell::get);
        let lane = &LANES[index as usize];
        let mut guard = lane.queue.lock();
        let q = guard.as_mut().expect("lane queue before a hand-over");
        let fresh = q.free.pop().unwrap_or_default();
        let full = std::mem::replace(&mut *local, fresh);
        q.ready.push_back(full);
        let seq = lane.pushed.fetch_add(1, Ordering::SeqCst) + 1;
        if q.parked {
            lane.work.notify_one();
        }
        Some(Gate { lane: index, seq })
    })
}

/// Everything recorded so far is in its command buffer: hand over this
/// thread's list and wait for every lane to replay all it holds. Cheap when
/// the lanes are idle; a no-op when they never started.
pub(crate) fn sync() {
    if LOADERS.get().is_none() {
        return;
    }
    hand_over();
    let n = RUNNING.load(Ordering::Acquire);
    let mut started: Option<Instant> = None;
    for lane in &LANES[..n] {
        let target = lane.pushed.load(Ordering::SeqCst);
        if lane.done.load(Ordering::SeqCst) >= target {
            continue;
        }
        started.get_or_insert_with(Instant::now);
        wait_done(lane, target);
    }
    if let Some(started) = started {
        let ns = started.elapsed().as_nanos() as u64;
        STATS.sync_ns.fetch_add(ns, Ordering::Relaxed);
        STATS.syncs.fetch_add(1, Ordering::Relaxed);
        crate::runtime::drain::note_store_route_us("rec_lane_sync_wait_us", ns / 1000);
        crate::runtime::drain::note_store_route("rec_lane_sync_waits");
    }
}

fn lane_main(index: usize) {
    super::recorder::apply_thread_priority(super::recorder::current_thread_priority());
    let lane = &LANES[index];
    let mut taken = 0u64;
    let mut spent: Option<List> = None;
    let mut busy = Duration::ZERO;
    let mut lists = 0u64;
    loop {
        // Poll a little before taking the lock: a burst of draws hands over a
        // list every few microseconds.
        if lane.pushed.load(Ordering::Acquire) == taken {
            let started = Instant::now();
            let mut polls = 0u32;
            while lane.pushed.load(Ordering::Acquire) == taken {
                std::hint::spin_loop();
                polls += 1;
                if polls % 64 == 0 && started.elapsed() >= LANE_SPIN {
                    break;
                }
            }
        }
        let mut list = {
            let mut guard = lane.queue.lock();
            if let Some(mut done) = spent.take() {
                done.clear();
                guard.as_mut().expect("lane queue").free.push(done);
            }
            loop {
                let q = guard.as_mut().expect("lane queue");
                if let Some(list) = q.ready.pop_front() {
                    break list;
                }
                q.parked = true;
                if lists > 0 {
                    crate::runtime::drain::note_store_route_us(
                        "rec_lane_busy_us",
                        busy.as_micros() as u64,
                    );
                    crate::runtime::drain::note_store_route_n("rec_lane_lists", lists);
                    lane.busy_ns.fetch_add(busy.as_nanos() as u64, Ordering::Relaxed);
                    lane.lists.fetch_add(lists, Ordering::Relaxed);
                    busy = Duration::ZERO;
                    lists = 0;
                }
                lane.work.wait(&mut guard);
                guard.as_mut().expect("lane queue").parked = false;
                crate::runtime::drain::note_store_route("rec_lane_wakes");
            }
        };
        taken += 1;
        let started = Instant::now();
        let loaders = CURRENT.read().clone().expect("loaders before the lane");
        // SAFETY: each command is the `ash` call the recorder would have made,
        // with the same handles, on the command buffer it named.
        unsafe { replay(&loaders.0, &mut list) };
        busy += started.elapsed();
        lists += 1;
        if busy >= Duration::from_millis(2) {
            // A lane that never parks (a long burst) still reports.
            lane.busy_ns.fetch_add(busy.as_nanos() as u64, Ordering::Relaxed);
            lane.lists.fetch_add(lists, Ordering::Relaxed);
            crate::runtime::drain::note_store_route_us("rec_lane_busy_us", busy.as_micros() as u64);
            crate::runtime::drain::note_store_route_n("rec_lane_lists", lists);
            busy = Duration::ZERO;
            lists = 0;
        }
        spent = Some(list);
        lane.done.fetch_add(1, Ordering::SeqCst);
        if lane.waiters.load(Ordering::SeqCst) > 0 {
            let _q = lane.queue.lock();
            lane.idle.notify_all();
        }
    }
}

unsafe fn replay(lo: &Loaders, l: &mut List) {
    let d = &lo.device;
    let cmds = std::mem::take(&mut l.cmds);
    unsafe {
        for cmd in &cmds {
            match *cmd {
                Cmd::Barrier(cb, src, dst, deps, mem, buf, img) => d.cmd_pipeline_barrier(
                    cb,
                    src,
                    dst,
                    deps,
                    mem.of(&l.mem),
                    buf.of(&l.buf),
                    img.of(&l.img),
                ),
                Cmd::BeginPass(cb, render_pass, framebuffer, area, clears, contents) => {
                    let info = vk::RenderPassBeginInfo::default()
                        .render_pass(render_pass)
                        .framebuffer(framebuffer)
                        .render_area(area)
                        .clear_values(clears.of(&l.clears));
                    d.cmd_begin_render_pass(cb, &info, contents)
                }
                Cmd::EndPass(cb) => d.cmd_end_render_pass(cb),
                Cmd::BindPipeline(cb, bp, p) => d.cmd_bind_pipeline(cb, bp, p),
                Cmd::BindSets(cb, bp, layout, first, sets, offsets) => d.cmd_bind_descriptor_sets(
                    cb,
                    bp,
                    layout,
                    first,
                    sets.of(&l.sets),
                    offsets.of(&l.u32s),
                ),
                Cmd::PushSet(cb, bp, layout, set, writes) => {
                    let mut vk_writes: Vec<vk::WriteDescriptorSet<'_>> =
                        std::mem::take(&mut l.vk_writes);
                    for w in writes.of(&l.writes) {
                        let mut v = vk::WriteDescriptorSet::default()
                            .dst_binding(w.binding)
                            .dst_array_element(w.array_element)
                            .descriptor_type(w.ty);
                        if w.images.1 > 0 {
                            v = v.image_info(w.images.of(&l.iinfos));
                        } else if w.buffers.1 > 0 {
                            v = v.buffer_info(w.buffers.of(&l.binfos));
                        }
                        v.descriptor_count = w.count;
                        vk_writes.push(v);
                    }
                    if let Some(push) = lo.push.as_ref() {
                        push.cmd_push_descriptor_set(cb, bp, layout, set, &vk_writes)
                    }
                    vk_writes.clear();
                    // SAFETY: empty, so it borrows nothing; only its capacity is kept.
                    l.vk_writes = std::mem::transmute::<
                        Vec<vk::WriteDescriptorSet<'_>>,
                        Vec<vk::WriteDescriptorSet<'static>>,
                    >(vk_writes);
                }
                Cmd::BindVertex(cb, first, buffers, offsets) => d.cmd_bind_vertex_buffers(
                    cb,
                    first,
                    buffers.of(&l.buffers),
                    offsets.of(&l.sizes),
                ),
                Cmd::BindIndex(cb, b, o, t) => d.cmd_bind_index_buffer(cb, b, o, t),
                Cmd::Draw(cb, v, i, fv, fi) => d.cmd_draw(cb, v, i, fv, fi),
                Cmd::DrawIndexed(cb, ic, inst, fi, vo, finst) => {
                    d.cmd_draw_indexed(cb, ic, inst, fi, vo, finst)
                }
                Cmd::CopyBuffer(cb, s, t, r) => d.cmd_copy_buffer(cb, s, t, r.of(&l.bcopies)),
                Cmd::CopyBufferToImage(cb, s, i, il, r) => {
                    d.cmd_copy_buffer_to_image(cb, s, i, il, r.of(&l.bicopies))
                }
                Cmd::CopyImage(cb, s, sl, t, tl, r) => {
                    d.cmd_copy_image(cb, s, sl, t, tl, r.of(&l.icopies))
                }
                Cmd::CopyImageToBuffer(cb, i, il, b, r) => {
                    d.cmd_copy_image_to_buffer(cb, i, il, b, r.of(&l.bicopies))
                }
                Cmd::ClearColor(cb, i, il, c, r) => {
                    d.cmd_clear_color_image(cb, i, il, &c, r.of(&l.ranges))
                }
                Cmd::BeginQuery(cb, p, q, f) => d.cmd_begin_query(cb, p, q, f),
                Cmd::EndQuery(cb, p, q) => d.cmd_end_query(cb, p, q),
                Cmd::ResetQuery(cb, p, f, c) => d.cmd_reset_query_pool(cb, p, f, c),
                Cmd::Timestamp(cb, s, p, q) => d.cmd_write_timestamp(cb, s, p, q),
                Cmd::Viewport(cb, f, v) => d.cmd_set_viewport(cb, f, v.of(&l.viewports)),
                Cmd::Scissor(cb, f, r) => d.cmd_set_scissor(cb, f, r.of(&l.rects)),
                Cmd::BlendConstants(cb, c) => d.cmd_set_blend_constants(cb, &c),
                Cmd::StencilReference(cb, f, v) => d.cmd_set_stencil_reference(cb, f, v),
                Cmd::DepthBias(cb, a, b, c) => d.cmd_set_depth_bias(cb, a, b, c),
                Cmd::LineWidth(cb, w) => d.cmd_set_line_width(cb, w),
                Cmd::CullMode(cb, m) => {
                    if let Some(e) = lo.eds.as_ref() {
                        e.cmd_set_cull_mode(cb, m)
                    }
                }
                Cmd::FrontFace(cb, f) => {
                    if let Some(e) = lo.eds.as_ref() {
                        e.cmd_set_front_face(cb, f)
                    }
                }
                Cmd::Topology(cb, t) => {
                    if let Some(e) = lo.eds.as_ref() {
                        e.cmd_set_primitive_topology(cb, t)
                    }
                }
                Cmd::PolygonMode(cb, m) => {
                    if let Some(e) = lo.eds3.as_ref() {
                        e.cmd_set_polygon_mode(cb, m)
                    }
                }
                Cmd::DepthClampEnable(cb, b) => {
                    if let Some(e) = lo.eds3.as_ref() {
                        e.cmd_set_depth_clamp_enable(cb, b)
                    }
                }
                Cmd::DepthTest(cb, b) => {
                    if let Some(e) = lo.eds.as_ref() {
                        e.cmd_set_depth_test_enable(cb, b)
                    }
                }
                Cmd::DepthWrite(cb, b) => {
                    if let Some(e) = lo.eds.as_ref() {
                        e.cmd_set_depth_write_enable(cb, b)
                    }
                }
                Cmd::DepthCompare(cb, op) => {
                    if let Some(e) = lo.eds.as_ref() {
                        e.cmd_set_depth_compare_op(cb, op)
                    }
                }
                Cmd::DepthBoundsTest(cb, b) => {
                    if let Some(e) = lo.eds.as_ref() {
                        e.cmd_set_depth_bounds_test_enable(cb, b)
                    }
                }
                Cmd::StencilTest(cb, b) => {
                    if let Some(e) = lo.eds.as_ref() {
                        e.cmd_set_stencil_test_enable(cb, b)
                    }
                }
                Cmd::StencilOp(cb, f, fail, pass, depth_fail, cmp) => {
                    if let Some(e) = lo.eds.as_ref() {
                        e.cmd_set_stencil_op(cb, f, fail, pass, depth_fail, cmp)
                    }
                }
                Cmd::StencilCompareMask(cb, f, m) => d.cmd_set_stencil_compare_mask(cb, f, m),
                Cmd::StencilWriteMask(cb, f, m) => d.cmd_set_stencil_write_mask(cb, f, m),
                Cmd::PushConstants(cb, layout, stages, offset, bytes) => {
                    d.cmd_push_constants(cb, layout, stages, offset, bytes.of(&l.bytes))
                }
                Cmd::Dispatch(cb, x, y, z) => d.cmd_dispatch(cb, x, y, z),
                Cmd::DispatchBase(cb, bx, by, bz, x, y, z) => {
                    d.cmd_dispatch_base(cb, bx, by, bz, x, y, z)
                }
                Cmd::FillBuffer(cb, b, o, size, data) => d.cmd_fill_buffer(cb, b, o, size, data),
                Cmd::HostCopy(r) => {
                    let bytes: usize = r.of(&l.segs).iter().map(|s| s.n).sum();
                    STATS.copy_bytes.fetch_add(bytes as u64, Ordering::Relaxed);
                    let repeat = copy_repeat::on();
                    for seg in r.of(&l.segs) {
                        if repeat {
                            copy_repeat::note(seg.src, seg.n);
                        }
                        std::ptr::copy_nonoverlapping(seg.src, seg.dst, seg.n)
                    }
                }
                Cmd::Begin(cb) => {
                    if let Err(e) = d.reset_command_buffer(cb, vk::CommandBufferResetFlags::empty()) {
                        note_lane_failure("reset", e);
                    } else if let Err(e) = d.begin_command_buffer(
                        cb,
                        &vk::CommandBufferBeginInfo::default()
                            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                    ) {
                        note_lane_failure("begin", e);
                    }
                }
                Cmd::End(cb) => {
                    if let Err(e) = d.end_command_buffer(cb) {
                        note_lane_failure("end", e);
                    }
                }
            }
        }
    }
    l.cmds = cmds;
}

fn own_mem(b: &vk::MemoryBarrier<'_>) -> vk::MemoryBarrier<'static> {
    debug_assert!(b.p_next.is_null());
    vk::MemoryBarrier::default()
        .src_access_mask(b.src_access_mask)
        .dst_access_mask(b.dst_access_mask)
}

fn own_buf(b: &vk::BufferMemoryBarrier<'_>) -> vk::BufferMemoryBarrier<'static> {
    debug_assert!(b.p_next.is_null());
    vk::BufferMemoryBarrier::default()
        .src_access_mask(b.src_access_mask)
        .dst_access_mask(b.dst_access_mask)
        .src_queue_family_index(b.src_queue_family_index)
        .dst_queue_family_index(b.dst_queue_family_index)
        .buffer(b.buffer)
        .offset(b.offset)
        .size(b.size)
}

fn own_img(b: &vk::ImageMemoryBarrier<'_>) -> vk::ImageMemoryBarrier<'static> {
    debug_assert!(b.p_next.is_null());
    vk::ImageMemoryBarrier::default()
        .src_access_mask(b.src_access_mask)
        .dst_access_mask(b.dst_access_mask)
        .old_layout(b.old_layout)
        .new_layout(b.new_layout)
        .src_queue_family_index(b.src_queue_family_index)
        .dst_queue_family_index(b.dst_queue_family_index)
        .image(b.image)
        .subresource_range(b.subresource_range)
}

fn put_mapped<T, U>(v: &mut Vec<U>, s: &[T], f: impl Fn(&T) -> U) -> R {
    let start = v.len() as u32;
    v.extend(s.iter().map(f));
    R(start, s.len() as u32)
}

// ---------------------------------------------------------------------------
// The call sites' functions: the `ash` call, or its deferral.
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn pipeline_barrier(
    d: &ash::Device,
    cb: vk::CommandBuffer,
    src: vk::PipelineStageFlags,
    dst: vk::PipelineStageFlags,
    deps: vk::DependencyFlags,
    mem: &[vk::MemoryBarrier<'_>],
    buf: &[vk::BufferMemoryBarrier<'_>],
    img: &[vk::ImageMemoryBarrier<'_>],
) {
    if active() {
        rec(cb, |l| {
            let m = put_mapped(&mut l.mem, mem, own_mem);
            let b = put_mapped(&mut l.buf, buf, own_buf);
            let i = put_mapped(&mut l.img, img, own_img);
            l.cmds.push(Cmd::Barrier(cb, src, dst, deps, m, b, i));
        });
    } else {
        direct();
        unsafe { d.cmd_pipeline_barrier(cb, src, dst, deps, mem, buf, img) }
    }
}

pub(crate) unsafe fn begin_render_pass(
    d: &ash::Device,
    cb: vk::CommandBuffer,
    info: &vk::RenderPassBeginInfo<'_>,
    contents: vk::SubpassContents,
) {
    if active() {
        debug_assert!(info.p_next.is_null());
        let clears: &[vk::ClearValue] =
            if info.clear_value_count == 0 || info.p_clear_values.is_null() {
                &[]
            } else {
                // SAFETY: the info names `clear_value_count` values at the pointer.
                unsafe {
                    std::slice::from_raw_parts(info.p_clear_values, info.clear_value_count as usize)
                }
            };
        rec(cb, |l| {
            let c = R::put(&mut l.clears, clears);
            l.cmds.push(Cmd::BeginPass(
                cb,
                info.render_pass,
                info.framebuffer,
                info.render_area,
                c,
                contents,
            ));
        });
    } else {
        direct();
        unsafe { d.cmd_begin_render_pass(cb, info, contents) }
    }
}

macro_rules! simple {
    ($name:ident, $call:ident, ($($arg:ident : $ty:ty),*), $variant:expr) => {
        pub(crate) unsafe fn $name(d: &ash::Device, cb: vk::CommandBuffer $(, $arg: $ty)*) {
            if active() {
                rec(cb, |l| l.cmds.push($variant(cb $(, $arg)*)));
            } else {
                direct();
                unsafe { d.$call(cb $(, $arg)*) }
            }
        }
    };
}

simple!(end_render_pass, cmd_end_render_pass, (), Cmd::EndPass);
simple!(bind_pipeline, cmd_bind_pipeline, (bp: vk::PipelineBindPoint, p: vk::Pipeline), Cmd::BindPipeline);
simple!(bind_index_buffer, cmd_bind_index_buffer, (b: vk::Buffer, o: vk::DeviceSize, t: vk::IndexType), Cmd::BindIndex);
simple!(draw, cmd_draw, (v: u32, i: u32, fv: u32, fi: u32), Cmd::Draw);
simple!(draw_indexed, cmd_draw_indexed, (ic: u32, inst: u32, fi: u32, vo: i32, finst: u32), Cmd::DrawIndexed);
simple!(begin_query, cmd_begin_query, (p: vk::QueryPool, q: u32, f: vk::QueryControlFlags), Cmd::BeginQuery);
simple!(end_query, cmd_end_query, (p: vk::QueryPool, q: u32), Cmd::EndQuery);
simple!(reset_query_pool, cmd_reset_query_pool, (p: vk::QueryPool, f: u32, c: u32), Cmd::ResetQuery);
simple!(write_timestamp, cmd_write_timestamp, (s: vk::PipelineStageFlags, p: vk::QueryPool, q: u32), Cmd::Timestamp);
simple!(set_stencil_reference, cmd_set_stencil_reference, (f: vk::StencilFaceFlags, v: u32), Cmd::StencilReference);
simple!(set_depth_bias, cmd_set_depth_bias, (a: f32, b: f32, c: f32), Cmd::DepthBias);
simple!(set_line_width, cmd_set_line_width, (w: f32), Cmd::LineWidth);
simple!(set_stencil_compare_mask, cmd_set_stencil_compare_mask, (f: vk::StencilFaceFlags, m: u32), Cmd::StencilCompareMask);
simple!(dispatch, cmd_dispatch, (x: u32, y: u32, z: u32), Cmd::Dispatch);
simple!(dispatch_base, cmd_dispatch_base, (bx: u32, by: u32, bz: u32, x: u32, y: u32, z: u32), Cmd::DispatchBase);
simple!(fill_buffer, cmd_fill_buffer, (b: vk::Buffer, o: vk::DeviceSize, size: vk::DeviceSize, data: u32), Cmd::FillBuffer);
simple!(set_stencil_write_mask, cmd_set_stencil_write_mask, (f: vk::StencilFaceFlags, m: u32), Cmd::StencilWriteMask);

pub(crate) unsafe fn bind_descriptor_sets(
    d: &ash::Device,
    cb: vk::CommandBuffer,
    bind_point: vk::PipelineBindPoint,
    layout: vk::PipelineLayout,
    first: u32,
    sets: &[vk::DescriptorSet],
    offsets: &[u32],
) {
    if active() {
        rec(cb, |l| {
            let s = R::put(&mut l.sets, sets);
            let o = R::put(&mut l.u32s, offsets);
            l.cmds
                .push(Cmd::BindSets(cb, bind_point, layout, first, s, o));
        });
    } else {
        direct();
        unsafe { d.cmd_bind_descriptor_sets(cb, bind_point, layout, first, sets, offsets) }
    }
}

pub(crate) unsafe fn push_descriptor_set(
    p: &ash::khr::push_descriptor::Device,
    cb: vk::CommandBuffer,
    bind_point: vk::PipelineBindPoint,
    layout: vk::PipelineLayout,
    set: u32,
    writes: &[vk::WriteDescriptorSet<'_>],
) {
    if active() {
        rec(cb, |l| {
            let start = l.writes.len() as u32;
            for w in writes {
                debug_assert!(w.p_next.is_null() && w.p_texel_buffer_view.is_null());
                let n = w.descriptor_count as usize;
                // SAFETY: a write names `descriptor_count` infos at whichever
                // pointer its type uses.
                let images = if w.p_image_info.is_null() {
                    R(0, 0)
                } else {
                    R::put(&mut l.iinfos, unsafe {
                        std::slice::from_raw_parts(w.p_image_info, n)
                    })
                };
                let buffers = if w.p_buffer_info.is_null() {
                    R(0, 0)
                } else {
                    R::put(&mut l.binfos, unsafe {
                        std::slice::from_raw_parts(w.p_buffer_info, n)
                    })
                };
                l.writes.push(Write {
                    binding: w.dst_binding,
                    array_element: w.dst_array_element,
                    ty: w.descriptor_type,
                    count: w.descriptor_count,
                    images,
                    buffers,
                });
            }
            let r = R(start, writes.len() as u32);
            l.cmds.push(Cmd::PushSet(cb, bind_point, layout, set, r));
        });
    } else {
        direct();
        unsafe { p.cmd_push_descriptor_set(cb, bind_point, layout, set, writes) }
    }
}

pub(crate) unsafe fn push_constants(
    d: &ash::Device,
    cb: vk::CommandBuffer,
    layout: vk::PipelineLayout,
    stages: vk::ShaderStageFlags,
    offset: u32,
    bytes: &[u8],
) {
    if active() {
        rec(cb, |l| {
            let r = R::put(&mut l.bytes, bytes);
            l.cmds
                .push(Cmd::PushConstants(cb, layout, stages, offset, r));
        });
    } else {
        direct();
        unsafe { d.cmd_push_constants(cb, layout, stages, offset, bytes) }
    }
}

pub(crate) unsafe fn bind_vertex_buffers(
    d: &ash::Device,
    cb: vk::CommandBuffer,
    first: u32,
    buffers: &[vk::Buffer],
    offsets: &[vk::DeviceSize],
) {
    if active() {
        rec(cb, |l| {
            let b = R::put(&mut l.buffers, buffers);
            let o = R::put(&mut l.sizes, offsets);
            l.cmds.push(Cmd::BindVertex(cb, first, b, o));
        });
    } else {
        direct();
        unsafe { d.cmd_bind_vertex_buffers(cb, first, buffers, offsets) }
    }
}

pub(crate) unsafe fn copy_buffer(
    d: &ash::Device,
    cb: vk::CommandBuffer,
    s: vk::Buffer,
    t: vk::Buffer,
    r: &[vk::BufferCopy],
) {
    if active() {
        rec(cb, |l| {
            let x = R::put(&mut l.bcopies, r);
            l.cmds.push(Cmd::CopyBuffer(cb, s, t, x));
        });
    } else {
        direct();
        unsafe { d.cmd_copy_buffer(cb, s, t, r) }
    }
}

pub(crate) unsafe fn copy_buffer_to_image(
    d: &ash::Device,
    cb: vk::CommandBuffer,
    s: vk::Buffer,
    i: vk::Image,
    il: vk::ImageLayout,
    r: &[vk::BufferImageCopy],
) {
    if active() {
        rec(cb, |l| {
            let x = R::put(&mut l.bicopies, r);
            l.cmds.push(Cmd::CopyBufferToImage(cb, s, i, il, x));
        });
    } else {
        direct();
        unsafe { d.cmd_copy_buffer_to_image(cb, s, i, il, r) }
    }
}

pub(crate) unsafe fn copy_image(
    d: &ash::Device,
    cb: vk::CommandBuffer,
    s: vk::Image,
    sl: vk::ImageLayout,
    t: vk::Image,
    tl: vk::ImageLayout,
    r: &[vk::ImageCopy],
) {
    if active() {
        rec(cb, |l| {
            let x = R::put(&mut l.icopies, r);
            l.cmds.push(Cmd::CopyImage(cb, s, sl, t, tl, x));
        });
    } else {
        direct();
        unsafe { d.cmd_copy_image(cb, s, sl, t, tl, r) }
    }
}

pub(crate) unsafe fn copy_image_to_buffer(
    d: &ash::Device,
    cb: vk::CommandBuffer,
    i: vk::Image,
    il: vk::ImageLayout,
    b: vk::Buffer,
    r: &[vk::BufferImageCopy],
) {
    if active() {
        rec(cb, |l| {
            let x = R::put(&mut l.bicopies, r);
            l.cmds.push(Cmd::CopyImageToBuffer(cb, i, il, b, x));
        });
    } else {
        direct();
        unsafe { d.cmd_copy_image_to_buffer(cb, i, il, b, r) }
    }
}

pub(crate) unsafe fn clear_color_image(
    d: &ash::Device,
    cb: vk::CommandBuffer,
    i: vk::Image,
    il: vk::ImageLayout,
    c: &vk::ClearColorValue,
    r: &[vk::ImageSubresourceRange],
) {
    if active() {
        rec(cb, |l| {
            let x = R::put(&mut l.ranges, r);
            l.cmds.push(Cmd::ClearColor(cb, i, il, *c, x));
        });
    } else {
        direct();
        unsafe { d.cmd_clear_color_image(cb, i, il, c, r) }
    }
}

pub(crate) unsafe fn set_viewport(
    d: &ash::Device,
    cb: vk::CommandBuffer,
    first: u32,
    v: &[vk::Viewport],
) {
    if active() {
        rec(cb, |l| {
            let x = R::put(&mut l.viewports, v);
            l.cmds.push(Cmd::Viewport(cb, first, x));
        });
    } else {
        direct();
        unsafe { d.cmd_set_viewport(cb, first, v) }
    }
}

pub(crate) unsafe fn set_scissor(
    d: &ash::Device,
    cb: vk::CommandBuffer,
    first: u32,
    r: &[vk::Rect2D],
) {
    if active() {
        rec(cb, |l| {
            let x = R::put(&mut l.rects, r);
            l.cmds.push(Cmd::Scissor(cb, first, x));
        });
    } else {
        direct();
        unsafe { d.cmd_set_scissor(cb, first, r) }
    }
}

pub(crate) unsafe fn set_blend_constants(d: &ash::Device, cb: vk::CommandBuffer, c: &[f32; 4]) {
    if active() {
        rec(cb, |l| l.cmds.push(Cmd::BlendConstants(cb, *c)));
    } else {
        direct();
        unsafe { d.cmd_set_blend_constants(cb, c) }
    }
}

macro_rules! eds {
    ($name:ident, $call:ident, $loader:ty, ($($arg:ident : $ty:ty),*), $variant:expr) => {
        pub(crate) unsafe fn $name(e: &$loader, cb: vk::CommandBuffer $(, $arg: $ty)*) {
            if active() {
                rec(cb, |l| l.cmds.push($variant(cb $(, $arg)*)));
            } else {
                direct();
                unsafe { e.$call(cb $(, $arg)*) }
            }
        }
    };
}

eds!(set_cull_mode, cmd_set_cull_mode, ash::ext::extended_dynamic_state::Device, (m: vk::CullModeFlags), Cmd::CullMode);
eds!(set_front_face, cmd_set_front_face, ash::ext::extended_dynamic_state::Device, (f: vk::FrontFace), Cmd::FrontFace);
eds!(set_primitive_topology, cmd_set_primitive_topology, ash::ext::extended_dynamic_state::Device, (t: vk::PrimitiveTopology), Cmd::Topology);
eds!(set_depth_test_enable, cmd_set_depth_test_enable, ash::ext::extended_dynamic_state::Device, (b: bool), Cmd::DepthTest);
eds!(set_depth_write_enable, cmd_set_depth_write_enable, ash::ext::extended_dynamic_state::Device, (b: bool), Cmd::DepthWrite);
eds!(set_depth_compare_op, cmd_set_depth_compare_op, ash::ext::extended_dynamic_state::Device, (op: vk::CompareOp), Cmd::DepthCompare);
eds!(set_depth_bounds_test_enable, cmd_set_depth_bounds_test_enable, ash::ext::extended_dynamic_state::Device, (b: bool), Cmd::DepthBoundsTest);
eds!(set_stencil_test_enable, cmd_set_stencil_test_enable, ash::ext::extended_dynamic_state::Device, (b: bool), Cmd::StencilTest);
eds!(set_stencil_op, cmd_set_stencil_op, ash::ext::extended_dynamic_state::Device, (f: vk::StencilFaceFlags, fail: vk::StencilOp, pass: vk::StencilOp, depth_fail: vk::StencilOp, cmp: vk::CompareOp), Cmd::StencilOp);
eds!(set_polygon_mode, cmd_set_polygon_mode, ash::ext::extended_dynamic_state3::Device, (m: vk::PolygonMode), Cmd::PolygonMode);
eds!(set_depth_clamp_enable, cmd_set_depth_clamp_enable, ash::ext::extended_dynamic_state3::Device, (b: bool), Cmd::DepthClampEnable);
