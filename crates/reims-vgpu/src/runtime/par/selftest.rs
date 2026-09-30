//! A fork-join per drain tranche that does no device work, to measure the pool.
//!
//! TEMP (lab): `REIMS_VGPU_LAB_PAR_SELFTEST=noop|spin|read|gen`. Off (and free:
//! one `OnceLock` load per tranche) when the variable is unset.
//!
//! The pool is a mechanism, and a mechanism that nothing uses yet can only be
//! judged by running it. This runs one [`Pool::scope`] at the end of every drain
//! tranche of a live boot — on the real drain thread, under the real device
//! lock, with the vCPUs busy — so the three questions that decide whether the
//! pool is worth building on can be read from `store_routes` before any consumer
//! exists:
//!
//! * What does a fork-join cost when its items do nothing (`noop`):
//!   `par_fork_us`, `par_wait_us`, and the scope's own wall clock,
//!   `par_selftest_scope_us`?
//! * Does it actually run in parallel (`spin`): items that busy-wait a fixed
//!   time, where `par_selftest_scope_us` per tick should be near one item and
//!   not the sum, and `par_jobs_helped` says who ran them?
//! * Are the workers really RCU-safe readers of guest memory (`read`): every
//!   item reads a page of RAM through the worker host while the vCPUs run, and
//!   `par_selftest_read_ok` / `_fail` / `_skip` say what came back?
//! * What does the dirty tracker do when several threads ask it at once (`gen`):
//!   each item also asks `guest_write_gen` for a live mapping's token, timed, so
//!   `par_selftest_gen_us` over `par_selftest_gen_calls` is the FFI cost under
//!   concurrency. That last one is the open question the design names
//!   (section 5, "scaling of the dirty tracker over threads").
//!
//! It selects no behaviour: no result of any item is used by the device.

use super::{Pool, WorkerEnv};
use crate::model::DeviceState;
use crate::runtime::drain::{note_store_route, note_store_route_n};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

/// What each item of the self-test does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mode {
    Off,
    /// Nothing: the pure cost of the fork and the join.
    Noop,
    /// Busy-wait [`SPIN_ITEM`]: proves items overlap.
    Spin,
    /// Read a page of guest RAM through the worker host.
    Read,
    /// `Read`, plus one `guest_write_gen` for a live mapping token.
    Gen,
}

/// How long a `spin` item occupies its thread.
const SPIN_ITEM: Duration = Duration::from_micros(50);

/// Where a `read` item looks: 2 MiB plus a megabyte per item. Low guest RAM on
/// every machine the lab boots, and read-only here, so nothing depends on what
/// the guest keeps there. An address the shim says is not RAM is skipped and
/// counted rather than read.
const READ_BASE: u64 = 0x20_0000;
const READ_STRIDE: u64 = 0x10_0000;
const READ_LEN: usize = 4096;

/// The mode the environment asks for, parsed once per process.
pub(crate) fn mode() -> Mode {
    static MODE: OnceLock<Mode> = OnceLock::new();
    *MODE.get_or_init(
        || match std::env::var("REIMS_VGPU_LAB_PAR_SELFTEST").as_deref() {
            Ok("noop") => Mode::Noop,
            Ok("spin") => Mode::Spin,
            Ok("read") => Mode::Read,
            Ok("gen") => Mode::Gen,
            _ => Mode::Off,
        },
    )
}

/// What one item saw.
#[derive(Clone, Copy, Debug, Default)]
struct Probe {
    read_ok: u32,
    read_fail: u32,
    read_skip: u32,
    gen_some: u32,
    gen_none: u32,
    gen_calls: u32,
    gen_ns: u64,
}

/// One item of the self-test, run on whichever thread claimed it.
fn item(state: &DeviceState, env: &WorkerEnv<'_>, index: &u32, mode: Mode) -> Probe {
    let mut probe = Probe::default();
    match mode {
        Mode::Off | Mode::Noop => {}
        Mode::Spin => {
            let until = Instant::now() + SPIN_ITEM;
            while Instant::now() < until {
                std::hint::spin_loop();
            }
        }
        Mode::Read | Mode::Gen => {
            use crate::runtime::host::{HostMemory as _, HostOps as _};
            let host = env.host();
            let gpa = READ_BASE + u64::from(*index) * READ_STRIDE;
            if host.is_ram_gpa(gpa) {
                let mut page = [0u8; READ_LEN];
                match host.read_gpa(gpa, &mut page) {
                    Ok(()) => probe.read_ok = 1,
                    Err(_) => probe.read_fail = 1,
                }
            } else {
                probe.read_skip = 1;
            }
            if mode == Mode::Gen {
                let token = state
                    .mappings
                    .values()
                    .map(|m| m.guest_write_token)
                    .filter(|&t| t != 0)
                    .nth(*index as usize);
                if let Some(token) = token {
                    let started = Instant::now();
                    let gen = host.guest_write_gen(token);
                    probe.gen_ns = started.elapsed().as_nanos() as u64;
                    probe.gen_calls = 1;
                    if gen.is_some() {
                        probe.gen_some = 1;
                    } else {
                        probe.gen_none = 1;
                    }
                }
            }
        }
    }
    probe
}

/// Run the configured self-test fork-join once, if one is configured.
///
/// Called by `device_drain` after the tranche's guest work, with the device
/// state shared: from here until it returns the drain can do nothing else, which
/// is the condition a real consumer will be under.
pub(crate) fn tick(state: &DeviceState) {
    let mode = mode();
    if mode == Mode::Off {
        return;
    }
    let Some(pool) = state.par() else {
        // A shim with no thread registration, or a device made without host
        // ops: nothing to test, and saying so once per tranche is the point.
        note_store_route("par_selftest_no_pool");
        return;
    };
    run(pool, state, mode);
}

fn run(pool: &Pool, state: &DeviceState, mode: Mode) {
    // One item per member of the pool, and never fewer than two: a single item
    // would not fork, and the self-test is about the fork.
    let items: Vec<u32> = (0..(pool.workers() as u32 + 1).max(2)).collect();
    let started = Instant::now();
    let scoped = pool.scope(state, &items, |st, env, i| item(st, env, i, mode));
    let scope_us = started.elapsed().as_micros() as u64;

    let mut total = Probe::default();
    let mut panicked = 0u64;
    for outcome in &scoped.results {
        match outcome {
            Ok(p) => {
                total.read_ok += p.read_ok;
                total.read_fail += p.read_fail;
                total.read_skip += p.read_skip;
                total.gen_some += p.gen_some;
                total.gen_none += p.gen_none;
                total.gen_calls += p.gen_calls;
                total.gen_ns += p.gen_ns;
            }
            Err(_) => panicked += 1,
        }
    }
    note_store_route("par_selftest_ticks");
    note_store_route_n("par_selftest_scope_us", scope_us);
    note_store_route_n("par_selftest_read_ok", u64::from(total.read_ok));
    note_store_route_n("par_selftest_read_fail", u64::from(total.read_fail));
    note_store_route_n("par_selftest_read_skip", u64::from(total.read_skip));
    note_store_route_n("par_selftest_gen_calls", u64::from(total.gen_calls));
    note_store_route_n("par_selftest_gen_some", u64::from(total.gen_some));
    note_store_route_n("par_selftest_gen_none", u64::from(total.gen_none));
    note_store_route_n("par_selftest_gen_us", total.gen_ns / 1000);
    note_store_route_n("par_selftest_panicked", panicked);
}
