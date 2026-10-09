//! TEMP (lab, `REIMS_VGPU_LAB_LIVE_FILE=<path>`): lab switches that change while
//! the device runs, so an A/B runs as phases inside one boot.
//!
//! A lab A/B used to cost a boot an arm: on the home rig an arm reached its
//! first full-screen toggle ~200 s after QEMU started (macOS to its Dock, the
//! login, the probes) and measured for ~115 s, so more than half of a
//! four-arm queue was boot - and every boot re-rolled the windowed-glass
//! scene's heavy/light mode, which made arms of one queue less comparable than
//! two phases of one boot. Here the host writes `NAME=value` lines into one
//! file and the switches that consult it follow, re-read at most every 200 ms
//! and only when the file's modification time moved.
//!
//! A switch asks [`value`] (or [`on`]) first and falls back to its own latched
//! environment reading when the file is unset, missing, or does not name it, so
//! with the variable unset nothing changes. Reads are a mutex over a handful of
//! pairs; the switches that use this are consulted per tranche end or per
//! oversized carve, not per draw. Every change is reported once on the failure
//! channel (`lab_live gen=… set=…`) so a log says which phase a line belongs to.
//! Off under test.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Mutex, OnceLock};
use std::time::SystemTime;

struct Live {
    path: std::path::PathBuf,
    /// When the file was last looked at (process microseconds).
    checked_us: AtomicU64,
    /// Bumped on every change of the parsed contents.
    generation: AtomicU64,
    state: Mutex<(Option<SystemTime>, Vec<(String, String)>)>,
}

fn live() -> Option<&'static Live> {
    static LIVE: OnceLock<Option<Live>> = OnceLock::new();
    LIVE.get_or_init(|| {
        if cfg!(test) {
            return None;
        }
        let path = std::env::var_os("REIMS_VGPU_LAB_LIVE_FILE")?;
        crate::observe::off(format!("lab_live file={}", std::path::Path::new(&path).display()));
        Some(Live {
            path: path.into(),
            checked_us: AtomicU64::new(0),
            generation: AtomicU64::new(0),
            state: Mutex::new((None, Vec::new())),
        })
    })
    .as_ref()
}

fn refresh(l: &Live) {
    let now = crate::observe::elapsed_us();
    let last = l.checked_us.load(Relaxed);
    if last != 0 && now.saturating_sub(last) < 200_000 {
        return;
    }
    if l.checked_us.compare_exchange(last, now.max(1), Relaxed, Relaxed).is_err() {
        return; // another reader is looking at the file this round
    }
    let mtime = std::fs::metadata(&l.path).and_then(|m| m.modified()).ok();
    let mut st = l.state.lock().unwrap_or_else(|e| e.into_inner());
    if mtime == st.0 {
        return;
    }
    let pairs: Vec<(String, String)> = std::fs::read_to_string(&l.path)
        .unwrap_or_default()
        .lines()
        .filter_map(|ln| {
            let (k, v) = ln.trim().split_once('=')?;
            let k = k.trim().trim_start_matches("REIMS_VGPU_LAB_");
            (!k.is_empty()).then(|| (k.to_string(), v.trim().to_string()))
        })
        .collect();
    if pairs != st.1 {
        let generation = l.generation.fetch_add(1, Relaxed) + 1;
        let set: Vec<String> = pairs.iter().map(|(k, v)| format!("{k}={v}")).collect();
        crate::observe::off(format!("lab_live gen={generation} set={}", set.join(",")));
        st.1 = pairs;
    }
    st.0 = mtime;
}

/// The live value of lab switch `name` (without the `REIMS_VGPU_LAB_` prefix),
/// or `None` when the live file is unset, missing, or does not name it.
pub(crate) fn value(name: &str) -> Option<String> {
    with_value(name, |v| v.to_string())
}

/// Whether a live file was configured at all - a production boot has none, and a
/// caller that would do work to ask [`value`] can skip it.
pub(crate) fn present() -> bool {
    live().is_some()
}

/// [`value`] read as a switch: `on` is true, anything else false.
pub(crate) fn on(name: &str) -> Option<bool> {
    with_value(name, |v| v == "on")
}

/// Look `name` up in this thread's copy of the parsed pairs, refreshed from
/// the shared state only when its generation moved. Some switches are asked
/// per draw or per bind (b069: `value` ~1.5 % of the drain's busy samples when
/// it took the state lock and cloned a string each time).
fn with_value<T>(name: &str, f: impl FnOnce(&str) -> T) -> Option<T> {
    let l = live()?;
    refresh(l);
    thread_local! {
        static COPY: std::cell::RefCell<(u64, Vec<(String, String)>)> =
            const { std::cell::RefCell::new((u64::MAX, Vec::new())) };
    }
    let generation = l.generation.load(Relaxed);
    COPY.with(|copy| {
        let mut copy = copy.borrow_mut();
        if copy.0 != generation {
            let st = l.state.lock().unwrap_or_else(|e| e.into_inner());
            // Read the generation again under the lock: `refresh` bumps it
            // while holding the lock, so this pair is consistent.
            copy.0 = l.generation.load(Relaxed);
            copy.1 = st.1.clone();
        }
        copy.1.iter().find(|(k, _)| k == name).map(|(_, v)| f(v))
    })
}
