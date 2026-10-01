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
//! into functions and source lines offline.
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
            line.push_str(&format!("{:x}", pc.wrapping_sub(base)));
        }
        let _ = writeln!(w, "{line}");
        if last_flush.elapsed() >= std::time::Duration::from_secs(2) {
            last_flush = std::time::Instant::now();
            let _ = w.flush();
        }
    }
}
