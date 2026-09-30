//! TEMP (lab, S0): how much of the work WindowServer sends repeats from one
//! frame to the next. `REIMS_VGPU_LAB_REPEAT=1`; measurement only.
//!
//! A frame is what ends at a present. Each draw carries three *struct* keys and a
//! *bytes* key (the first 4 KiB of every vertex and fragment buffer it binds):
//!
//! * A - everything: pipeline, targets (ref, mapping, gva), bound texture refs and
//!   resource identity, samplers, viewport/scissor, counts;
//! * B - A without the render-target identity (the framebuffers rotate, so A can
//!   only repeat every third frame);
//! * C - B without the bound-texture identity (pure shape).
//!
//! Counters (store-route windows): `rep_a_full`, `rep_b_full`, `rep_c_full` -
//! key and bytes seen in the previous frame; `rep_a_full_w4`, `rep_b_full_w4` - in any
//! of the last four; `rep_b_struct` - B seen in the previous frame but the bytes
//! differ (a drag: same draw, new positions); `rep_new` - B not seen; `rep_partial` -
//! draws with a buffer over 4 KiB (only its prefix is keyed).
//!
//! A stream is a run of draws continuing one render pass; `rep_streams`,
//! `rep_streams_repeat_b` (ordered B+bytes list seen in the previous frame),
//! `rep_streams_repeat_b_w4`.

use std::collections::HashSet;
use std::sync::Mutex;

type Key = (u64, u64);

#[derive(Default)]
struct Frames {
    a: [HashSet<Key>; 4],
    b: [HashSet<Key>; 4],
    c_prev: HashSet<Key>,
    b_struct_prev: HashSet<u64>,
    a_cur: HashSet<Key>,
    b_cur: HashSet<Key>,
    c_cur: HashSet<Key>,
    b_struct_cur: HashSet<u64>,
    streams: [HashSet<u64>; 4],
    streams_cur: HashSet<u64>,
    v_prev: [HashSet<Key>; 6],
    v_cur: [HashSet<Key>; 6],
    p: [[HashSet<Key>; 4]; 3],
    p_cur: [HashSet<Key>; 3],
    stream_hash: u64,
    stream_draws: u32,
}

static STATE: Mutex<Option<Frames>> = Mutex::new(None);

pub fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("REIMS_VGPU_LAB_REPEAT").is_ok_and(|v| v == "1"))
}

fn close_stream(f: &mut Frames) {
    if f.stream_draws > 0 {
        crate::runtime::drain::note_store_route("rep_streams");
        if f.streams[0].contains(&f.stream_hash) {
            crate::runtime::drain::note_store_route("rep_streams_repeat_b");
        }
        if f.streams.iter().any(|s| s.contains(&f.stream_hash)) {
            crate::runtime::drain::note_store_route("rep_streams_repeat_b_w4");
        }
        f.streams_cur.insert(f.stream_hash);
    }
    f.stream_hash = 0;
    f.stream_draws = 0;
}

/// A present ended the frame.
pub fn frame() {
    if !on() {
        return;
    }
    let mut guard = STATE.lock().unwrap_or_else(|e| e.into_inner());
    let f = guard.get_or_insert_with(Frames::default);
    close_stream(f);
    crate::runtime::drain::note_store_route("rep_frames");
    for i in (1..4).rev() {
        f.a[i] = std::mem::take(&mut f.a[i - 1]);
        f.b[i] = std::mem::take(&mut f.b[i - 1]);
        f.streams[i] = std::mem::take(&mut f.streams[i - 1]);
    }
    f.a[0] = std::mem::take(&mut f.a_cur);
    f.b[0] = std::mem::take(&mut f.b_cur);
    f.streams[0] = std::mem::take(&mut f.streams_cur);
    f.c_prev = std::mem::take(&mut f.c_cur);
    f.b_struct_prev = std::mem::take(&mut f.b_struct_cur);
    for i in 0..6 {
        f.v_prev[i] = std::mem::take(&mut f.v_cur[i]);
    }
    for k in 0..3 {
        for i in (1..4).rev() {
            f.p[k][i] = std::mem::take(&mut f.p[k][i - 1]);
        }
        f.p[k][0] = std::mem::take(&mut f.p_cur[k]);
    }
}

/// One draw: `new_stream` when it does not continue the previous draw's pass.
pub fn draw(
    a: u64,
    b: u64,
    c: u64,
    bytes: u64,
    partial: bool,
    new_stream: bool,
    variants: [u64; 6],
    prefix: [u64; 3],
) {
    let mut guard = STATE.lock().unwrap_or_else(|e| e.into_inner());
    let f = guard.get_or_insert_with(Frames::default);
    if new_stream {
        close_stream(f);
    }
    let (ka, kb, kc) = ((a, bytes), (b, bytes), (c, bytes));
    let note = crate::runtime::drain::note_store_route;
    note("rep_draws");
    if partial {
        note("rep_partial");
    }
    if f.a[0].contains(&ka) {
        note("rep_a_full");
    }
    if f.b[0].contains(&kb) {
        note("rep_b_full");
    } else if f.b_struct_prev.contains(&b) {
        note("rep_b_struct");
    } else {
        note("rep_new");
    }
    if f.c_prev.contains(&kc) {
        note("rep_c_full");
    }
    if f.a.iter().any(|s| s.contains(&ka)) {
        note("rep_a_full_w4");
    }
    if f.b.iter().any(|s| s.contains(&kb)) {
        note("rep_b_full_w4");
    }
    const NAMES: [&str; 6] = [
        "rep_v_no_first_vertex",
        "rep_v_no_index_ref",
        "rep_v_no_viewport_scissor",
        "rep_v_no_sampler",
        "rep_v_no_pipeline",
        "rep_v_no_counts",
    ];
    for (i, v) in variants.iter().enumerate() {
        let k = (*v, bytes);
        if f.v_prev[i].contains(&k) {
            note(NAMES[i]);
        }
        f.v_cur[i].insert(k);
    }
    // The index-buffer-normalised shape with only the first 64 / 256 / 1024 bytes of
    // every buffer keyed: the window a draw is handed runs on into its neighbours'
    // data, so a long prefix repeats only when the whole neighbourhood does.
    const PN: [(&str, &str); 3] = [
        ("rep_p64", "rep_p64_w4"),
        ("rep_p256", "rep_p256_w4"),
        ("rep_p1024", "rep_p1024_w4"),
    ];
    for k in 0..3 {
        let key = (variants[1], prefix[k]);
        if f.p[k][0].contains(&key) {
            note(PN[k].0);
        }
        if f.p[k].iter().any(|s| s.contains(&key)) {
            note(PN[k].1);
        }
        f.p_cur[k].insert(key);
    }
    f.a_cur.insert(ka);
    f.b_cur.insert(kb);
    f.c_cur.insert(kc);
    f.b_struct_cur.insert(b);
    f.stream_hash = f.stream_hash.rotate_left(5) ^ b ^ bytes.rotate_left(17);
    f.stream_draws += 1;
}
