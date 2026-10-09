//! TEMP (lab): which kind of core the calling thread is on, for the census.
//!
//! The host is a hybrid i7-14700K (8 performance cores with two threads each,
//! 12 efficiency cores). A drain tranche that runs on an efficiency core runs
//! ~40 % slower, and nothing in the census says where the scheduler put it.
//! [`on_efficiency_core`] answers from the CPU-set table the OS publishes
//! (`EfficiencyClass` 0 is the least performant class; on a machine with one
//! class every core reads as performance).

use std::sync::OnceLock;

extern "system" {
    fn GetSystemCpuSetInformation(
        info: *mut u8,
        len: u32,
        returned: *mut u32,
        process: *mut core::ffi::c_void,
        flags: u32,
    ) -> i32;
    fn GetCurrentProcessorNumber() -> u32;
    fn GetCurrentThread() -> *mut core::ffi::c_void;
    fn SetThreadSelectedCpuSets(thread: *mut core::ffi::c_void, ids: *const u32, n: u32) -> i32;
}

/// Lab `REIMS_VGPU_LAB_PCORES=on`: the calling thread prefers the performance
/// cores (a CPU-set selection, which the scheduler keeps as a preference). For
/// the draw recorder; QEMU does the same for the vCPU and drain threads.
pub(crate) fn pin_to_performance_cores(who: &str) {
    if !std::env::var("REIMS_VGPU_LAB_PCORES").is_ok_and(|v| v == "on") {
        return;
    }
    pin_to_performance_cores_now(who);
}

/// TEMP (lab, DRAW_PAR, `REIMS_VGPU_LAB_PAR_PCORES=on`): the pool's helpers
/// alone on the performance cores. b120: a helper's job took 8.1 us against
/// the drain's ~3.4 for the same work, and the fork-join waits for the slowest.
pub(crate) fn pin_pool_to_performance_cores(who: &str) {
    if !std::env::var("REIMS_VGPU_LAB_PAR_PCORES").is_ok_and(|v| v == "on") {
        return;
    }
    pin_to_performance_cores_now(who);
}

fn pin_to_performance_cores_now(who: &str) {
    let ids = performance_cpu_set_ids();
    if ids.is_empty() {
        return;
    }
    // SAFETY: the pseudo handle of this thread and a slice of `ids.len()` ids.
    if unsafe { SetThreadSelectedCpuSets(GetCurrentThread(), ids.as_ptr(), ids.len() as u32) } != 0 {
        crate::observe::off(format!("lab_pcores {who} sets={}", ids.len()));
    }
}

/// TEMP (lab, `REIMS_VGPU_LAB_DEVICE_CCD=N`): the calling device thread prefers
/// the logical processors that share last-level cache `N` - one CCD of the home
/// rig's Ryzen 9 7950X (two CCDs of 8 cores, an L3 each). There a draw's encode
/// cost 88 us against 36 on Intel's single-die i7 over a whole run (hq057
/// finish_phase), and the draw sub-phases account for ~25 of the 88: the drain,
/// its recorder and the parallel pool hand work to each other for every draw,
/// and a hand-off between CCDs crosses the die-to-die link. The drain and the
/// other QEMU-side device threads take the same CCD (reims_vgpu_lab_device_prio).
/// A CPU-set selection is a preference the scheduler keeps. Latched.
pub(crate) fn pin_device_ccd(who: &str) {
    static CCD: OnceLock<Option<u8>> = OnceLock::new();
    let Some(ccd) = *CCD.get_or_init(|| {
        std::env::var("REIMS_VGPU_LAB_DEVICE_CCD").ok().and_then(|v| v.trim().parse::<u8>().ok())
    }) else {
        return;
    };
    let ids: Vec<u32> = cpu_set_records().iter().filter(|r| r.3 == ccd).map(|r| r.0).collect();
    if ids.is_empty() {
        return;
    }
    // SAFETY: the pseudo handle of this thread and a slice of `ids.len()` ids.
    if unsafe { SetThreadSelectedCpuSets(GetCurrentThread(), ids.as_ptr(), ids.len() as u32) } != 0
        && crate::observe::first_sight(
            "lab_device_ccd",
            who.bytes().fold(0u64, |h, b| h.wrapping_mul(31).wrapping_add(u64::from(b))),
        )
    {
        crate::observe::off(format!("lab_device_ccd {who} ccd={ccd} sets={}", ids.len()));
    }
}

/// CPU-set ids of the most performant efficiency class (group 0).
fn performance_cpu_set_ids() -> Vec<u32> {
    let recs = cpu_set_records();
    let max_class = recs.iter().map(|r| r.2).max().unwrap_or(0);
    recs.iter().filter(|r| r.2 == max_class).map(|r| r.0).collect()
}

/// (cpu-set id, logical processor, efficiency class, last-level cache index) per
/// group-0 CPU set.
fn cpu_set_records() -> Vec<(u32, usize, u8, u8)> {
    let mut need = 0u32;
    // SAFETY: a size query (null buffer, zero length).
    unsafe { GetSystemCpuSetInformation(core::ptr::null_mut(), 0, &mut need, core::ptr::null_mut(), 0) };
    if need == 0 {
        return Vec::new();
    }
    let mut buf = vec![0u8; need as usize];
    // SAFETY: the buffer is `need` bytes, as asked.
    if unsafe { GetSystemCpuSetInformation(buf.as_mut_ptr(), need, &mut need, core::ptr::null_mut(), 0) } == 0 {
        return Vec::new();
    }
    let mut recs = Vec::new();
    let mut off = 0usize;
    while off + 19 <= buf.len() {
        let size = u32::from_le_bytes(buf[off..off + 4].try_into().unwrap()) as usize;
        if size == 0 {
            break;
        }
        let id = u32::from_le_bytes(buf[off + 8..off + 12].try_into().unwrap());
        let group = u16::from_le_bytes(buf[off + 12..off + 14].try_into().unwrap());
        if group == 0 {
            recs.push((id, buf[off + 14] as usize, buf[off + 18], buf[off + 16]));
        }
        off += size;
    }
    recs
}

/// Per logical processor (group 0): true for an efficiency core.
fn table() -> &'static [bool] {
    static T: OnceLock<Vec<bool>> = OnceLock::new();
    T.get_or_init(|| {
        let mut need = 0u32;
        // SAFETY: a size query (null buffer, zero length).
        unsafe { GetSystemCpuSetInformation(core::ptr::null_mut(), 0, &mut need, core::ptr::null_mut(), 0) };
        if need == 0 {
            return Vec::new();
        }
        let mut buf = vec![0u8; need as usize];
        // SAFETY: the buffer is `need` bytes, as asked.
        if unsafe {
            GetSystemCpuSetInformation(buf.as_mut_ptr(), need, &mut need, core::ptr::null_mut(), 0)
        } == 0
        {
            return Vec::new();
        }
        // SYSTEM_CPU_SET_INFORMATION: Size u32 @0, Type u32 @4, then CpuSet:
        // Id u32 @8, Group u16 @12, LogicalProcessorIndex u8 @14, CoreIndex u8 @15,
        // LastLevelCacheIndex u8 @16, NumaNodeIndex u8 @17, EfficiencyClass u8 @18.
        let mut recs = Vec::new();
        let mut off = 0usize;
        while off + 19 <= buf.len() {
            let size = u32::from_le_bytes(buf[off..off + 4].try_into().unwrap()) as usize;
            if size == 0 {
                break;
            }
            let group = u16::from_le_bytes(buf[off + 12..off + 14].try_into().unwrap());
            if group == 0 {
                recs.push((buf[off + 14] as usize, buf[off + 18]));
            }
            off += size;
        }
        let max_class = recs.iter().map(|r| r.1).max().unwrap_or(0);
        let n = recs.iter().map(|r| r.0 + 1).max().unwrap_or(0);
        let mut t = vec![false; n];
        for (lp, class) in recs {
            t[lp] = class < max_class;
        }
        crate::observe::off(format!(
            "lab_cores logical={n} efficiency={} (EfficiencyClass below {max_class})",
            t.iter().filter(|e| **e).count()
        ));
        t
    })
}

/// Whether the calling thread is on an efficiency core right now.
pub(crate) fn on_efficiency_core() -> bool {
    // SAFETY: no arguments, no preconditions.
    let cpu = unsafe { GetCurrentProcessorNumber() } as usize;
    table().get(cpu).copied().unwrap_or(false)
}

/// TEMP (lab, `REIMS_VGPU_LAB_DEVICE_PRIO=on`): the calling device thread runs at above-normal priority, ahead of
/// the vCPU threads. With 24 vCPUs on a 28-thread host the device's helpers (the parallel pool, the recorder, the
/// submit and stamp threads) compete with the guest for cores, and n217's heavy-glass profile had the drain
/// waiting on them for a quarter of its time (pool joins 8.8 %, dirty-sync workers 6 %, deferred-draw settles
/// 9 %). The drain alone at above normal (HIQOS) was neutral at 8 vCPUs (2026-10-01/02).
pub(crate) fn device_thread_priority(who: &str) {
    pin_device_ccd(who);
    static ON: OnceLock<bool> = OnceLock::new();
    if !*ON.get_or_init(|| std::env::var("REIMS_VGPU_LAB_DEVICE_PRIO").is_ok_and(|v| v == "on")) {
        return;
    }
    extern "system" {
        fn SetThreadPriority(h: *mut core::ffi::c_void, p: i32) -> i32;
    }
    /// THREAD_PRIORITY_ABOVE_NORMAL.
    const ABOVE_NORMAL: i32 = 1;
    // SAFETY: the pseudo handle of the calling thread.
    if unsafe { SetThreadPriority(GetCurrentThread(), ABOVE_NORMAL) } != 0
        && crate::observe::first_sight(
            "lab_device_prio",
            who.bytes().fold(0u64, |h, b| h.wrapping_mul(31).wrapping_add(u64::from(b))),
        )
    {
        crate::observe::off(format!("lab_device_prio {who}"));
    }
}
