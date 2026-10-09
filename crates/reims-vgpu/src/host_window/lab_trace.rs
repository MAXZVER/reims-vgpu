//! TEMP lab: a timeline of the guest's cursor on the host, for the passed-through
//! pointer's stutter (2026-10-07). `REIMS_VGPU_LAB_CURSOR_TRACE=<file>` appends one
//! line per event: `<qpc_ns> <kind> <x> <y>`, where kind is `G` (the guest's
//! cursor doorbell reported a new position) or `W` (the window placed the host
//! pointer, in window pixels). The clock is the host's performance counter in
//! nanoseconds, the same one QEMU's `get_clock()` and Python's
//! `time.perf_counter_ns()` read on Windows, so a run's QEMU and driver traces
//! line up with this one.

use std::io::Write as _;
use std::sync::Mutex;
use std::sync::OnceLock;

struct Trace {
    file: std::fs::File,
    buf: Vec<u8>,
}

fn trace() -> Option<&'static Mutex<Trace>> {
    static T: OnceLock<Option<Mutex<Trace>>> = OnceLock::new();
    T.get_or_init(|| {
        let path = std::env::var_os("REIMS_VGPU_LAB_CURSOR_TRACE")?;
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .ok()?;
        Some(Mutex::new(Trace {
            file,
            buf: Vec::with_capacity(1 << 16),
        }))
    })
    .as_ref()
}

/// Whether a trace file was asked for.
pub fn on() -> bool {
    trace().is_some()
}

#[cfg(windows)]
pub fn now_ns() -> u64 {
    #[link(name = "kernel32")]
    extern "system" {
        fn QueryPerformanceCounter(count: *mut i64) -> i32;
        fn QueryPerformanceFrequency(freq: *mut i64) -> i32;
    }
    static FREQ: OnceLock<i64> = OnceLock::new();
    let freq = *FREQ.get_or_init(|| {
        let mut f = 0i64;
        // SAFETY: writes one i64 through a valid pointer.
        unsafe { QueryPerformanceFrequency(&mut f) };
        f.max(1)
    });
    let mut c = 0i64;
    // SAFETY: writes one i64 through a valid pointer.
    unsafe { QueryPerformanceCounter(&mut c) };
    (c as i128 * 1_000_000_000 / freq as i128) as u64
}

#[cfg(not(windows))]
pub fn now_ns() -> u64 {
    static T0: OnceLock<std::time::Instant> = OnceLock::new();
    T0.get_or_init(std::time::Instant::now).elapsed().as_nanos() as u64
}

/// Record one event (a no-op without a trace file).
pub fn event(kind: char, x: f64, y: f64) {
    let Some(t) = trace() else { return };
    let ns = now_ns();
    let Ok(mut t) = t.lock() else { return };
    let _ = writeln!(t.buf, "{ns} {kind} {x:.1} {y:.1}");
    if t.buf.len() > 60_000 {
        flush_locked(&mut t);
    }
}

/// Write out what is buffered (the window calls this once a second).
pub fn flush() {
    let Some(t) = trace() else { return };
    if let Ok(mut t) = t.lock() {
        flush_locked(&mut t);
    }
}

fn flush_locked(t: &mut Trace) {
    if t.buf.is_empty() {
        return;
    }
    let Trace { file, buf } = t;
    let _ = file.write_all(buf);
    buf.clear();
}
