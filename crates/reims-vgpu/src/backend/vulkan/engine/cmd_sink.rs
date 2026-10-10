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
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
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
    loaders: OnceLock<LoadersCell>,
}

static LANE: Lane = Lane {
    queue: Mutex::new(None),
    work: Condvar::new(),
    idle: Condvar::new(),
    pushed: AtomicU64::new(0),
    done: AtomicU64::new(0),
    waiters: AtomicU32::new(0),
    loaders: OnceLock::new(),
};

thread_local! {
    static ACTIVE: Cell<bool> = const { Cell::new(false) };
    static LOCAL: RefCell<List> = RefCell::new(List::default());
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

/// Defer this thread's `vkCmd*` calls to the lane until the guard drops (which
/// hands the list over). A no-op when the lane is off. `ctx` provides the
/// loaders the lane replays with, cloned once.
pub(crate) fn activate(ctx: &super::context::DeviceContext) -> ActiveGuard {
    if !enabled() {
        return ActiveGuard(false);
    }
    LANE.loaders.get_or_init(|| {
        *LANE.queue.lock() = Some(Queue::default());
        let cell = LoadersCell(Loaders {
            device: ctx.device.clone(),
            push: ctx.push_descriptor.clone(),
            eds: ctx.extended_dynamic_state.clone(),
            eds3: ctx.extended_dynamic_state3.clone(),
        });
        std::thread::Builder::new()
            .name("reims-vgpu-rec-lane".into())
            .spawn(lane_main)
            .expect("spawn the recording lane");
        cell
    });
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

/// Hand the lane a host copy to make before the command buffer ends.
///
/// # Safety
///
/// The source stays readable and the destination writable and unaliased
/// until the next [`sync`] returns.
pub(crate) unsafe fn host_copy(seg: HostSeg) {
    debug_assert!(active());
    rec(|l| {
        let r = R::put(&mut l.segs, &[seg]);
        l.cmds.push(Cmd::HostCopy(r));
    });
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

fn rec(f: impl FnOnce(&mut List)) {
    LOCAL.with(|l| f(&mut l.borrow_mut()));
}

/// Give the lane this thread's pending list, taking a cleared one back.
fn hand_over() {
    LOCAL.with(|l| {
        let mut local = l.borrow_mut();
        if local.cmds.is_empty() {
            return;
        }
        let mut guard = LANE.queue.lock();
        let q = guard.as_mut().expect("lane queue before a hand-over");
        let fresh = q.free.pop().unwrap_or_default();
        let full = std::mem::replace(&mut *local, fresh);
        q.ready.push_back(full);
        LANE.pushed.fetch_add(1, Ordering::SeqCst);
        if q.parked {
            LANE.work.notify_one();
        }
    });
}

/// Everything recorded so far is in its command buffer: hand over this
/// thread's list and wait for the lane to replay all it holds. Cheap when the
/// lane is idle; a no-op when it never started.
pub(crate) fn sync() {
    if LANE.loaders.get().is_none() {
        return;
    }
    hand_over();
    let target = LANE.pushed.load(Ordering::SeqCst);
    if LANE.done.load(Ordering::SeqCst) >= target {
        return;
    }
    let started = Instant::now();
    let mut polls = 0u32;
    while LANE.done.load(Ordering::Acquire) < target {
        std::hint::spin_loop();
        polls += 1;
        if polls % 64 == 0 && started.elapsed() >= SYNC_SPIN {
            break;
        }
    }
    if LANE.done.load(Ordering::SeqCst) < target {
        LANE.waiters.fetch_add(1, Ordering::SeqCst);
        let mut q = LANE.queue.lock();
        while LANE.done.load(Ordering::SeqCst) < target {
            LANE.idle.wait_for(&mut q, Duration::from_micros(200));
        }
        drop(q);
        LANE.waiters.fetch_sub(1, Ordering::SeqCst);
    }
    crate::runtime::drain::note_store_route_us(
        "rec_lane_sync_wait_us",
        started.elapsed().as_micros() as u64,
    );
    crate::runtime::drain::note_store_route("rec_lane_sync_waits");
}

fn lane_main() {
    super::recorder::apply_thread_priority(super::recorder::current_thread_priority());
    let loaders = &LANE.loaders.get().expect("loaders before the lane").0;
    let mut taken = 0u64;
    let mut spent: Option<List> = None;
    let mut busy = Duration::ZERO;
    let mut lists = 0u64;
    loop {
        // Poll a little before taking the lock: a burst of draws hands over a
        // list every few microseconds.
        if LANE.pushed.load(Ordering::Acquire) == taken {
            let started = Instant::now();
            let mut polls = 0u32;
            while LANE.pushed.load(Ordering::Acquire) == taken {
                std::hint::spin_loop();
                polls += 1;
                if polls % 64 == 0 && started.elapsed() >= LANE_SPIN {
                    break;
                }
            }
        }
        let mut list = {
            let mut guard = LANE.queue.lock();
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
                    busy = Duration::ZERO;
                    lists = 0;
                }
                LANE.work.wait(&mut guard);
                guard.as_mut().expect("lane queue").parked = false;
                crate::runtime::drain::note_store_route("rec_lane_wakes");
            }
        };
        taken += 1;
        let started = Instant::now();
        // SAFETY: each command is the `ash` call the recorder would have made,
        // with the same handles, on the command buffer it named.
        unsafe { replay(loaders, &mut list) };
        busy += started.elapsed();
        lists += 1;
        spent = Some(list);
        LANE.done.fetch_add(1, Ordering::SeqCst);
        if LANE.waiters.load(Ordering::SeqCst) > 0 {
            let _q = LANE.queue.lock();
            LANE.idle.notify_all();
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
                    for seg in r.of(&l.segs) {
                        std::ptr::copy_nonoverlapping(seg.src, seg.dst, seg.n)
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
        rec(|l| {
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
        rec(|l| {
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
                rec(|l| l.cmds.push($variant(cb $(, $arg)*)));
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
        rec(|l| {
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
        rec(|l| {
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
        rec(|l| {
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
        rec(|l| {
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
        rec(|l| {
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
        rec(|l| {
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
        rec(|l| {
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
        rec(|l| {
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
        rec(|l| {
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
        rec(|l| {
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
        rec(|l| {
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
        rec(|l| l.cmds.push(Cmd::BlendConstants(cb, *c)));
    } else {
        direct();
        unsafe { d.cmd_set_blend_constants(cb, c) }
    }
}

macro_rules! eds {
    ($name:ident, $call:ident, $loader:ty, ($($arg:ident : $ty:ty),*), $variant:expr) => {
        pub(crate) unsafe fn $name(e: &$loader, cb: vk::CommandBuffer $(, $arg: $ty)*) {
            if active() {
                rec(|l| l.cmds.push($variant(cb $(, $arg)*)));
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
