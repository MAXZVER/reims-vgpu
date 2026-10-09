//! The pool's contract, tested at its boundary: what a scope returns, what a
//! panic does, who registers with RCU and when, and what teardown leaves behind.
//!
//! The shim is a table of counting callbacks whose `ctx` names per-test state, so
//! a test reads exactly its own pool's registrations and never another test's.

use super::{default_workers, planning, Pool, Settings, DEFAULT_WORKER_CAP};
use crate::model::{DeviceId, DeviceState, PAGE_SHIFT_X86};
use crate::qemu::host_ops::ReimsVgpuHostOps;
use std::os::raw::c_void;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// What the fake shim saw: one line per attach and per detach, from the calling
/// thread's own name.
#[derive(Default)]
struct Registrations {
    attached: AtomicUsize,
    detached: AtomicUsize,
    names: Mutex<Vec<String>>,
}

unsafe extern "C" fn attach(ctx: *mut c_void) {
    // SAFETY: `ctx` is the `Registrations` the test that built the table keeps
    // alive until its pool is shut down.
    let seen = unsafe { &*(ctx as *const Registrations) };
    seen.attached.fetch_add(1, Ordering::SeqCst);
    let name = std::thread::current().name().unwrap_or("").to_string();
    seen.names.lock().unwrap().push(name);
}

unsafe extern "C" fn detach(ctx: *mut c_void) {
    // SAFETY: as in `attach`.
    let seen = unsafe { &*(ctx as *const Registrations) };
    seen.detached.fetch_add(1, Ordering::SeqCst);
}

/// A table that offers the registration pair, reporting into `seen`.
fn ops(seen: &Registrations) -> ReimsVgpuHostOps {
    let mut table = ReimsVgpuHostOps::null();
    table.ctx = seen as *const Registrations as *mut c_void;
    table.thread_attach = Some(attach);
    table.thread_detach = Some(detach);
    table
}

fn settings(workers: usize) -> Settings {
    Settings {
        workers,
        worker_spin: Duration::ZERO,
        join_spin: Duration::from_micros(50),
    }
}

fn pool(seen: &Registrations, workers: usize) -> Pool {
    Pool::new(&ops(seen), &settings(workers)).expect("the table offers the registration pair")
}

fn wait_until(what: &str, mut done: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !done() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// The structural gate. A shim that does not say how to register a thread with
/// RCU gets no helper at all, whatever the settings ask for.
#[test]
fn a_table_without_the_registration_pair_gets_no_pool() {
    let none = ReimsVgpuHostOps::null();
    assert!(Pool::new(&none, &settings(4)).is_none());

    let seen = Registrations::default();
    let mut half = ops(&seen);
    half.thread_detach = None;
    assert!(
        Pool::new(&half, &settings(4)).is_none(),
        "attach without detach would strand a registered reader"
    );
}

/// Nothing is owned until something is wanted: a device whose switches are off
/// never pays for a thread, which is what keeps the default behaviour as it was.
#[test]
fn the_pool_is_cold_until_a_scope_wants_a_helper() {
    let seen = Registrations::default();
    let pool = pool(&seen, 3);
    assert_eq!(pool.stats().accepted.len(), 4, "three helpers + the caller");
    assert_eq!(seen.attached.load(Ordering::SeqCst), 0);

    // One item is not worth a fork, however many helpers exist.
    let one = pool.scope(&(), &[7u32], |_, _, i| *i + 1);
    assert!(one.serial);
    assert_eq!(one.results, vec![Ok(8)]);
    assert_eq!(seen.attached.load(Ordering::SeqCst), 0, "still cold");

    let items: Vec<u32> = (0..16).collect();
    let many = pool.scope(&(), &items, |_, _, i| *i);
    assert!(!many.serial);
    wait_until("the helpers to register", || {
        seen.attached.load(Ordering::SeqCst) == 3
    });
    pool.shutdown();
}

/// The zero-helper pool is the serial pool: `REIMS_VGPU_LAB_PAR_WORKERS=0`.
#[test]
fn the_serial_pool_never_spawns() {
    let seen = Registrations::default();
    let pool = pool(&seen, 0);
    let items: Vec<u32> = (0..64).collect();
    for _ in 0..5 {
        let out = pool.scope(&(), &items, |_, env, i| {
            assert!(!env.helper);
            *i * 3
        });
        assert!(out.serial);
        assert_eq!(out.helped, 0);
        assert_eq!(out.results[10], Ok(30));
    }
    let stats = pool.stats();
    assert_eq!(stats.scopes, 5);
    assert_eq!(stats.serial_scopes, 5);
    assert_eq!(seen.attached.load(Ordering::SeqCst), 0);
}

/// A scope is a parallel map: every item once, results in item order, whatever
/// the helper count and whichever thread ran what.
#[test]
fn every_item_runs_exactly_once_and_results_keep_item_order() {
    for workers in [0usize, 1, 4] {
        let seen = Registrations::default();
        let pool = pool(&seen, workers);
        let items: Vec<usize> = (0..300).collect();
        let runs: Vec<AtomicUsize> = items.iter().map(|_| AtomicUsize::new(0)).collect();
        for round in 0..20 {
            for r in &runs {
                r.store(0, Ordering::SeqCst);
            }
            let out = pool.scope(&runs, &items, |runs, _, &i| {
                runs[i].fetch_add(1, Ordering::SeqCst);
                i * 2 + round
            });
            assert_eq!(out.results.len(), items.len());
            for (i, result) in out.results.iter().enumerate() {
                assert_eq!(result, &Ok(i * 2 + round), "W={workers} item {i}");
                assert_eq!(runs[i].load(Ordering::SeqCst), 1, "W={workers} item {i}");
            }
        }
        pool.shutdown();
    }
}

/// Jobs borrow the caller's stack, which is the point of a scope: the shared
/// state, the items and what the closure captures need not be `'static`.
#[test]
fn jobs_borrow_the_callers_stack() {
    let seen = Registrations::default();
    let pool = pool(&seen, 3);
    let table: Vec<u64> = (0..100).map(|n| n * n).collect();
    let words: Vec<String> = (0..40).map(|n| "x".repeat(n)).collect();
    let bias = 1000usize;
    let out = pool.scope(&table, &words, |table, _, w| table[w.len()] as usize + bias);
    for (n, r) in out.results.iter().enumerate() {
        assert_eq!(r, &Ok(n * n + bias));
    }
    pool.shutdown();
}

/// What a job holds is the device state, shared, which is the claim
/// `DeviceState: Sync` exists for.
#[test]
fn jobs_read_the_device_state_shared() {
    let seen = Registrations::default();
    let pool = pool(&seen, 2);
    let state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let items = [0u8; 8];
    let out = pool.scope(&state, &items, |st, _, _| st.page_shift);
    assert!(out.results.iter().all(|r| r == &Ok(PAGE_SHIFT_X86)));
    // The borrow is over: the state is ours again.
    drop(state);
    pool.shutdown();
}

/// A panic is data in its own slot. The helper survives it, the other items are
/// unaffected, the next scope works, and the count says how many there were.
#[test]
fn a_panicking_job_is_data_and_the_pool_survives() {
    let seen = Registrations::default();
    let pool = pool(&seen, 3);
    let items: Vec<u32> = (0..64).collect();
    let before = crate::runtime::drain::census::store_route_count("par_job_panicked");
    let out = pool.scope(&(), &items, |_, _, &i| {
        if i % 7 == 3 {
            panic!("boom {i}");
        }
        i
    });
    let mut panicked = 0;
    for (i, r) in out.results.iter().enumerate() {
        if i % 7 == 3 {
            let failure = r.as_ref().expect_err("a panicking job has no result");
            assert!(
                failure.message.contains(&format!("boom_{i}")),
                "the message names the job: {failure}"
            );
            panicked += 1;
        } else {
            assert_eq!(r, &Ok(i as u32));
        }
    }
    assert_eq!(pool.stats().job_panics, panicked);
    assert_eq!(
        crate::runtime::drain::census::store_route_count("par_job_panicked") - before,
        panicked
    );

    // The helpers are still there and still serve.
    let again = pool.scope(&(), &items, |_, _, &i| i + 1);
    assert!(again
        .results
        .iter()
        .enumerate()
        .all(|(i, r)| r == &Ok(i as u32 + 1)));
    pool.shutdown();
    assert_eq!(seen.detached.load(Ordering::SeqCst), 3, "nobody died early");
}

/// The fork forks: items overlap in time, on threads named for the pool, and the
/// pool's threads did some of the work.
#[test]
fn helpers_run_items_in_parallel_on_named_threads() {
    let seen = Registrations::default();
    let pool = pool(&seen, 3);
    let inflight = AtomicUsize::new(0);
    let overlapped = AtomicBool::new(false);
    let names = Mutex::new(Vec::<String>::new());
    let items: Vec<u32> = (0..8).collect();
    let out = pool.scope(&(), &items, |_, env, _| {
        inflight.fetch_add(1, Ordering::SeqCst);
        // Hold until another thread is in an item too, bounded so a pool that
        // cannot fork fails the assertion instead of the suite hanging.
        let until = Instant::now() + Duration::from_secs(2);
        while inflight.load(Ordering::SeqCst) < 2 && Instant::now() < until {
            std::hint::spin_loop();
        }
        if inflight.load(Ordering::SeqCst) >= 2 {
            overlapped.store(true, Ordering::SeqCst);
        }
        inflight.fetch_sub(1, Ordering::SeqCst);
        names.lock().unwrap().push(format!(
            "{}:{}",
            env.helper,
            std::thread::current().name().unwrap_or("")
        ));
    });
    assert!(out.results.iter().all(Result::is_ok));
    assert!(
        overlapped.load(Ordering::SeqCst),
        "two items were never in flight together"
    );
    assert!(out.helped >= 1, "a pool thread took part");
    for name in names.lock().unwrap().iter() {
        if let Some(thread) = name.strip_prefix("true:") {
            assert!(
                thread.starts_with("reims-vgpu-par-"),
                "helper thread name {thread}"
            );
        }
    }
    pool.shutdown();
    let registered = seen.names.lock().unwrap().clone();
    assert_eq!(registered.len(), 3);
    for name in registered {
        assert!(
            name.starts_with("reims-vgpu-par-"),
            "registered under its own name: {name}"
        );
    }
}

/// Teardown: every helper that attached has detached and exited when `shutdown`
/// returns, it is idempotent, and a stopped pool still answers (serially).
#[test]
fn shutdown_stops_joins_and_detaches_and_later_scopes_run_serially() {
    let seen = Registrations::default();
    let pool = pool(&seen, 4);
    let items: Vec<u32> = (0..32).collect();
    assert!(!pool.scope(&(), &items, |_, _, &i| i).serial);
    wait_until("the helpers to register", || {
        seen.attached.load(Ordering::SeqCst) == 4
    });

    pool.shutdown();
    assert_eq!(seen.attached.load(Ordering::SeqCst), 4);
    assert_eq!(
        seen.detached.load(Ordering::SeqCst),
        4,
        "every helper detached"
    );
    let stats = pool.stats();
    assert_eq!(stats.attached, 0);
    assert_eq!(stats.detached, 4);

    pool.shutdown();
    assert_eq!(seen.detached.load(Ordering::SeqCst), 4, "idempotent");

    let after = pool.scope(&(), &items, |_, _, &i| i * 2);
    assert!(after.serial, "a stopped pool runs on its caller");
    assert_eq!(after.results[5], Ok(10));
}

/// Dropping a pool is shutting it down: no helper outlives it.
#[test]
fn dropping_a_pool_joins_its_helpers() {
    let seen = Registrations::default();
    {
        let pool = pool(&seen, 2);
        let items: Vec<u32> = (0..16).collect();
        let _ = pool.scope(&(), &items, |_, _, &i| i);
        wait_until("the helpers to register", || {
            seen.attached.load(Ordering::SeqCst) == 2
        });
    }
    assert_eq!(seen.detached.load(Ordering::SeqCst), 2);
}

/// One scope at a time forks. A scope that arrives while another is in flight —
/// from another thread, or from inside a job — runs serially and answers, and
/// nothing deadlocks.
#[test]
fn a_scope_that_finds_the_pool_busy_runs_serially() {
    let seen = Registrations::default();
    let pool = pool(&seen, 2);
    let release = AtomicBool::new(false);
    let entered = AtomicUsize::new(0);
    let items: Vec<u32> = (0..4).collect();

    std::thread::scope(|s| {
        let outer = s.spawn(|| {
            pool.scope(&(), &items, |_, _, &i| {
                entered.fetch_add(1, Ordering::SeqCst);
                let until = Instant::now() + Duration::from_secs(10);
                while !release.load(Ordering::SeqCst) && Instant::now() < until {
                    std::thread::yield_now();
                }
                i
            })
        });
        wait_until("the outer scope to be running items", || {
            entered.load(Ordering::SeqCst) >= 1
        });

        // From another thread, while the outer scope holds the fork.
        let other: Vec<u32> = (0..6).collect();
        let concurrent = pool.scope(&(), &other, |_, _, &i| i + 100);
        assert!(concurrent.serial, "the pool was busy");
        assert_eq!(concurrent.results[3], Ok(103));

        release.store(true, Ordering::SeqCst);
        let outer = outer.join().unwrap();
        assert!(!outer.serial);
        assert!(outer.results.iter().all(Result::is_ok));
    });

    // Nested: a job that scopes again gets the serial path, not a deadlock.
    let items: Vec<u32> = (0..4).collect();
    let out = pool.scope(&pool, &items, |pool, _, &i| {
        let inner: Vec<u32> = (0..3).collect();
        let nested = pool.scope(&(), &inner, |_, _, &j| j + i);
        assert!(nested.serial);
        nested
            .results
            .iter()
            .map(|r| *r.as_ref().unwrap())
            .sum::<u32>()
    });
    assert_eq!(out.results[2], Ok(3 + 6));
    pool.shutdown();
}

/// The executor's population accounting sees the coordinator and the helpers,
/// and records the batches each took part in.
#[test]
fn accounting_counts_every_member_that_took_part() {
    let seen = Registrations::default();
    let pool = pool(&seen, 2);
    let items: Vec<u32> = (0..64).collect();
    for _ in 0..10 {
        let _ = pool.scope(&(), &items, |_, _, &i| {
            // Enough work that a helper that woke has something to claim.
            std::hint::black_box((0..200).fold(i, |a, b| a.wrapping_mul(31).wrapping_add(b)))
        });
    }
    let stats = pool.stats();
    assert_eq!(stats.accepted.len(), 3);
    assert!(
        stats.accepted[2] >= 1,
        "the coordinator takes part in the batches it runs"
    );
    assert_eq!(stats.scopes, 10);
    assert_eq!(stats.jobs, 640);
    pool.shutdown();
}

#[test]
fn the_default_helper_count_is_bounded() {
    assert!(default_workers() <= DEFAULT_WORKER_CAP);
    assert!(Settings::MAX_WORKERS >= DEFAULT_WORKER_CAP);
}

/// TEMP (lab, PASS_PREP): the planning flag is up for the whole of a scope on
/// the coordinator and on every helper that serves it, down after, and down
/// after a job panicked too — the recorder-settle bypass must never outlive the
/// scope it was raised for, or the drain would stop settling for good.
#[test]
fn planning_flag_is_scoped() {
    assert!(!planning(), "an ordinary thread does not plan");
    // The serial pool first: a scope that cannot fork still plans, because its
    // items run the same read-only job code on the caller.
    {
        let seen = Registrations::default();
        let pool = pool(&seen, 0);
        let items = [0u32, 1];
        let out = pool.scope(&(), &items, |_, _, &i| {
            assert!(planning(), "the coordinator plans inside its own scope");
            i
        });
        assert_eq!(out.results[1], Ok(1));
        assert!(!planning(), "the scope is over");
        pool.shutdown();
    }
    // A forked scope: the flag is up on the coordinator and on the helpers,
    // and comes down on both sides. The items spin briefly, or the coordinator
    // would claim all of them before a helper wakes and the helper half of the
    // claim would be untestable.
    {
        let seen = Registrations::default();
        let pool = pool(&seen, 2);
        let helper_saw_planning = AtomicUsize::new(0);
        let coordinator_saw_planning = AtomicBool::new(false);
        let items: Vec<u32> = (0..64).collect();
        let out = pool.scope(&(), &items, |_, env, _| {
            let until = Instant::now() + Duration::from_micros(100);
            while Instant::now() < until {
                std::hint::spin_loop();
            }
            if planning() {
                if env.helper {
                    helper_saw_planning.fetch_add(1, Ordering::SeqCst);
                } else {
                    coordinator_saw_planning.store(true, Ordering::SeqCst);
                }
            }
        });
        assert!(out.results.iter().all(Result::is_ok));
        assert!(!planning(), "the scope is over on the coordinator");
        assert!(
            coordinator_saw_planning.load(Ordering::SeqCst),
            "the coordinator planned"
        );
        wait_until("the helpers to register", || {
            seen.attached.load(Ordering::SeqCst) == 2
        });
        assert!(
            helper_saw_planning.load(Ordering::SeqCst) > 0,
            "a helper planned"
        );
        // An ordinary thread outside the pool does not plan: the negative
        // control for the flag being thread-local rather than global.
        let helper = std::thread::spawn(|| {
            assert!(!planning());
        });
        helper.join().unwrap();
        pool.shutdown();
    }
    // A panicking job unwinds out of the scope's items only: the token's Drop
    // still runs on the coordinator, so the flag is down afterwards.
    {
        let seen = Registrations::default();
        let pool = pool(&seen, 2);
        let items: Vec<u32> = (0..8).collect();
        let _ = pool.scope(&(), &items, |_, _, &i| {
            assert!(planning());
            if i == 3 {
                panic!("boom");
            }
        });
        assert!(!planning(), "a panicked job leaves the flag down");
        pool.shutdown();
    }
    // And a scope nested inside a job leaves the outer flag standing: the
    // token restores what it found rather than clearing unconditionally.
    {
        let seen = Registrations::default();
        let pool = pool(&seen, 0);
        let items = [0u32, 1];
        let _ = pool.scope(&(), &items, |_, _, _| {
            assert!(planning());
            let inner = [0u32, 1];
            let _ = pool.scope(&(), &inner, |_, _, _| {
                assert!(planning(), "the nested scope keeps the outer flag");
            });
            assert!(planning(), "the outer flag survived the nested scope");
        });
        assert!(!planning());
        pool.shutdown();
    }
}

#[test]
fn an_overlap_hands_every_item_in_order_while_helpers_run_the_rest() {
    let seen = Registrations::default();
    let pool = pool(&seen, 4);
    let items: Vec<u32> = (0..64).collect();
    let runs = AtomicUsize::new(0);
    let got = pool.overlap(
        &(),
        &items,
        |_, _, &i| {
            runs.fetch_add(1, Ordering::SeqCst);
            std::thread::sleep(Duration::from_micros(50));
            i * 3
        },
        |ov| {
            assert!(ov.forked());
            (0..items.len())
                .map(|i| ov.wait(i).expect("not cancelled").expect("no panic"))
                .collect::<Vec<u32>>()
        },
    );
    assert_eq!(got, items.iter().map(|i| i * 3).collect::<Vec<_>>());
    assert_eq!(runs.load(Ordering::SeqCst), items.len());
    pool.shutdown();
}

#[test]
fn an_overlap_stopped_early_cancels_the_unclaimed_and_keeps_the_finished() {
    let seen = Registrations::default();
    let pool = pool(&seen, 2);
    let items: Vec<u32> = (0..200).collect();
    let runs = AtomicUsize::new(0);
    let (first, rest) = pool.overlap(
        &(),
        &items,
        |_, _, &i| {
            runs.fetch_add(1, Ordering::SeqCst);
            std::thread::sleep(Duration::from_micros(200));
            i
        },
        |ov| {
            let first = ov.wait(0).unwrap().unwrap();
            let rest = ov.stop(1);
            // Every later wait is over.
            assert!(ov.wait(5).is_none());
            (first, rest)
        },
    );
    assert_eq!(first, 0);
    assert_eq!(rest.len(), items.len() - 1);
    // Finished items form a prefix and carry their own value; the cancelled
    // suffix never ran.
    let done = rest.iter().take_while(|r| r.is_some()).count();
    assert!(rest[done..].iter().all(Option::is_none));
    for (k, r) in rest.iter().take(done).enumerate() {
        assert_eq!(r.as_ref().unwrap().as_ref().unwrap(), &(k as u32 + 1));
    }
    assert_eq!(runs.load(Ordering::SeqCst), 1 + done);
    assert!(done < items.len() - 1, "the stop cancelled something");
    pool.shutdown();
}

#[test]
fn an_overlap_without_helpers_runs_each_item_on_the_caller() {
    let seen = Registrations::default();
    let pool = pool(&seen, 0);
    let items: Vec<u32> = (0..8).collect();
    let caller = std::thread::current().id();
    let got = pool.overlap(
        &(),
        &items,
        |_, _, &i| {
            assert_eq!(std::thread::current().id(), caller);
            assert!(planning(), "an item runs as planning");
            i + 1
        },
        |ov| {
            assert!(!ov.forked());
            assert!(!planning(), "the body is not planning");
            (0..items.len()).map(|i| ov.wait(i).unwrap().unwrap()).collect::<Vec<_>>()
        },
    );
    assert_eq!(got, (1..=8).collect::<Vec<u32>>());
}
