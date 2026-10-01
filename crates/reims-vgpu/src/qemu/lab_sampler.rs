//! TEMP (lab, draw): an in-process stack sampler of the drain thread.
//!
//! `REIMS_VGPU_LAB_SAMPLER=<out file>` turns it on at the first drain entry;
//! `REIMS_VGPU_LAB_SAMPLER_GATE=<flag file>` limits sampling to the time the flag
//! file exists (the light-scene measurement window), and
//! `REIMS_VGPU_LAB_SAMPLER_HZ` sets the rate (default 1000).
//!
//! Why this exists: attaching gdb once a second to a VM gives sixty samples a
//! minute and freezes the guest each time. This thread suspends the drain
//! thread, reads its context, walks the stack with the OS's own unwind tables
//! (`RtlLookupFunctionEntry` / `RtlVirtualUnwind`, which is what gdb reads too)
//! and resumes it, a few hundred microseconds each. One line per sample,
//! `S <rva> <rva> ...` (leaf first, return addresses for the callers), with the
//! module base on the first line; `addr2line -i -e <exe>` turns the addresses
//! into functions and source lines offline. A frame in another module (the
//! Vulkan driver, the C runtime, ntdll) is written `@<absolute address>`, and the
//! first such frame in a module adds a line `M <base> <size> <path>` (sampsym.py
//! names it by the module's nearest export).
//!
//! Nothing is allocated or locked between `SuspendThread` and `ResumeThread`: the
//! suspended thread may hold the heap lock.

#![cfg(windows)]

use std::io::Write;
use std::sync::Once;

type Handle = *mut core::ffi::c_void;

#[repr(C, align(16))]
struct Context([u8; 1232]);

const CONTEXT_CONTROL_INTEGER: u32 = 0x0010_0003;
const OFF_FLAGS: usize = 0x30;
const OFF_RSP: usize = 0x98;
const OFF_RIP: usize = 0xF8;
const THREAD_SUSPEND_RESUME: u32 = 0x0002;
const THREAD_GET_CONTEXT: u32 = 0x0008;
const THREAD_QUERY_INFORMATION: u32 = 0x0040;
const MAX_FRAMES: usize = 48;

extern "system" {
    fn GetCurrentThreadId() -> u32;
    fn OpenThread(access: u32, inherit: i32, tid: u32) -> Handle;
    fn SuspendThread(h: Handle) -> u32;
    fn ResumeThread(h: Handle) -> u32;
    fn GetThreadContext(h: Handle, ctx: *mut Context) -> i32;
    fn GetModuleHandleW(name: *const u16) -> Handle;
    fn GetModuleHandleExW(flags: u32, name: *const u16, module: *mut Handle) -> i32;
    fn GetModuleFileNameW(module: Handle, name: *mut u16, size: u32) -> u32;
    fn K32GetModuleInformation(process: Handle, module: Handle, info: *mut ModuleInfo, size: u32) -> i32;
    fn GetCurrentProcess() -> Handle;
    fn RtlLookupFunctionEntry(
        pc: u64,
        base: *mut u64,
        history: *mut core::ffi::c_void,
    ) -> *mut core::ffi::c_void;
    fn RtlVirtualUnwind(
        handler_type: u32,
        image_base: u64,
        control_pc: u64,
        function_entry: *mut core::ffi::c_void,
        context: *mut Context,
        handler_data: *mut *mut core::ffi::c_void,
        establisher: *mut u64,
        pointers: *mut core::ffi::c_void,
    ) -> *mut core::ffi::c_void;
}

#[repr(C)]
struct ModuleInfo {
    base: *mut core::ffi::c_void,
    size: u32,
    entry: *mut core::ffi::c_void,
}

const GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS: u32 = 0x4;
const GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT: u32 = 0x2;

/// The module holding `pc` as (base, end), printing its `M` line the first time.
/// Called only with the drain thread running (it takes the loader lock).
fn module_of(pc: u64, known: &mut Vec<(u64, u64)>, w: &mut impl Write) -> bool {
    if known.iter().any(|&(b, e)| pc >= b && pc < e) {
        return true;
    }
    let mut h: Handle = core::ptr::null_mut();
    // SAFETY: plain Win32 queries about a loaded module; the buffers are ours.
    unsafe {
        if GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            pc as *const u16,
            &mut h,
        ) == 0
        {
            return false;
        }
        let mut info = ModuleInfo { base: core::ptr::null_mut(), size: 0, entry: core::ptr::null_mut() };
        if K32GetModuleInformation(GetCurrentProcess(), h, &mut info, core::mem::size_of::<ModuleInfo>() as u32) == 0 {
            return false;
        }
        let mut name = [0u16; 520];
        let len = GetModuleFileNameW(h, name.as_mut_ptr(), name.len() as u32) as usize;
        let path = String::from_utf16_lossy(&name[..len.min(name.len())]);
        let b = info.base as u64;
        known.push((b, b + info.size as u64));
        let _ = writeln!(w, "M {b:x} {:x} {path}", info.size);
    }
    true
}

struct SendHandle(Handle);
// SAFETY: a thread handle is usable from any thread.
unsafe impl Send for SendHandle {}

fn rd(ctx: &Context, off: usize) -> u64 {
    u64::from_le_bytes(ctx.0[off..off + 8].try_into().unwrap())
}

fn wr(ctx: &mut Context, off: usize, v: u64) {
    ctx.0[off..off + 8].copy_from_slice(&v.to_le_bytes());
}

/// Called on the drain thread; the first call starts the sampler.
pub(crate) fn start_once() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let Ok(out) = std::env::var("REIMS_VGPU_LAB_SAMPLER") else {
            return;
        };
        let gate = std::env::var("REIMS_VGPU_LAB_SAMPLER_GATE").ok();
        let hz: u64 = std::env::var("REIMS_VGPU_LAB_SAMPLER_HZ")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1000u64)
            .clamp(10, 20000);
        // SAFETY: plain Win32 calls on the current thread id.
        let target = unsafe {
            OpenThread(
                THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION,
                0,
                GetCurrentThreadId(),
            )
        };
        if target.is_null() {
            return;
        }
        let target = SendHandle(target);
        let _ = std::thread::Builder::new()
            .name("lab-sampler".into())
            .spawn(move || {
                let target = target;
                run(target.0, &out, gate.as_deref(), hz)
            });
    });
}

fn run(target: Handle, out: &str, gate: Option<&str>, hz: u64) {
    // SAFETY: a null name asks for the executable's own module.
    let base = unsafe { GetModuleHandleW(core::ptr::null()) } as u64;
    let Ok(file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(out)
    else {
        return;
    };
    let mut w = std::io::BufWriter::new(file);
    let _ = writeln!(w, "B {base:#x}");
    let period = std::time::Duration::from_micros(1_000_000 / hz);
    let mut rng: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut gate_open = false;
    let mut last_gate = std::time::Instant::now() - std::time::Duration::from_secs(1);
    let mut last_flush = std::time::Instant::now();
    let mut ctx = Context([0u8; 1232]);
    let mut frames = [0u64; MAX_FRAMES];
    let mut modules: Vec<(u64, u64)> = Vec::new();
    let exe_end = {
        let mut info = ModuleInfo { base: core::ptr::null_mut(), size: 0, entry: core::ptr::null_mut() };
        // SAFETY: a query about our own executable module.
        if unsafe {
            K32GetModuleInformation(GetCurrentProcess(), base as Handle, &mut info, core::mem::size_of::<ModuleInfo>() as u32)
        } != 0
        {
            base + info.size as u64
        } else {
            u64::MAX
        }
    };
    loop {
        if let Some(g) = gate {
            if last_gate.elapsed() >= std::time::Duration::from_millis(100) {
                last_gate = std::time::Instant::now();
                gate_open = std::path::Path::new(g).exists();
            }
            if !gate_open {
                let _ = w.flush();
                std::thread::sleep(std::time::Duration::from_millis(20));
                continue;
            }
        }
        // Jitter, so a fixed period cannot lock to a periodic workload.
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        let jitter = rng % 1000;
        std::thread::sleep(period + std::time::Duration::from_micros(jitter / 4));
        let mut n = 0usize;
        // SAFETY: the handle stays valid for the process lifetime; nothing below
        // allocates or locks until `ResumeThread`.
        unsafe {
            if SuspendThread(target) == u32::MAX {
                continue;
            }
            ctx.0[OFF_FLAGS..OFF_FLAGS + 4].copy_from_slice(&CONTEXT_CONTROL_INTEGER.to_le_bytes());
            if GetThreadContext(target, &mut ctx) != 0 {
                let mut last_rsp = 0u64;
                while n < MAX_FRAMES {
                    let rip = rd(&ctx, OFF_RIP);
                    let rsp = rd(&ctx, OFF_RSP);
                    if rip == 0 || (n > 0 && rsp <= last_rsp) {
                        break;
                    }
                    last_rsp = rsp;
                    frames[n] = rip;
                    n += 1;
                    let mut image_base = 0u64;
                    let entry = RtlLookupFunctionEntry(rip, &mut image_base, core::ptr::null_mut());
                    if entry.is_null() {
                        // A leaf without unwind data: the return address is on top.
                        let ret = *(rsp as *const u64);
                        wr(&mut ctx, OFF_RIP, ret);
                        wr(&mut ctx, OFF_RSP, rsp + 8);
                    } else {
                        let mut handler_data = core::ptr::null_mut();
                        let mut establisher = 0u64;
                        RtlVirtualUnwind(
                            0,
                            image_base,
                            rip,
                            entry,
                            &mut ctx,
                            &mut handler_data,
                            &mut establisher,
                            core::ptr::null_mut(),
                        );
                    }
                }
            }
            ResumeThread(target);
        }
        if n == 0 {
            continue;
        }
        let mut line = String::with_capacity(16 + n * 9);
        line.push('S');
        for (i, f) in frames[..n].iter().enumerate() {
            // A caller's address is a return address; step back into the call.
            let pc = if i == 0 { *f } else { f.wrapping_sub(1) };
            line.push(' ');
            if pc >= base && pc < exe_end {
                line.push_str(&format!("{:x}", pc - base));
            } else {
                module_of(pc, &mut modules, &mut w);
                line.push_str(&format!("@{pc:x}"));
            }
        }
        let _ = writeln!(w, "{line}");
        if last_flush.elapsed() >= std::time::Duration::from_secs(1) {
            last_flush = std::time::Instant::now();
            // The device clock the census lines carry as `t=`, so a window of the
            // fail log can be cut out of the samples without a gate file.
            let _ = writeln!(w, "T {}", crate::observe::elapsed_us() / 1000);
            let _ = w.flush();
        }
    }
}
