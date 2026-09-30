//! TEMP (lab, S0): how much of the work WindowServer sends repeats from one
//! frame to the next. `REIMS_VGPU_LAB_REPEAT=1`; measurement only.
//!
//! A frame is what ends at a present. Each draw is keyed twice: a *struct* key
//! (pipeline, targets, bound resources, samplers, viewport/scissor, counts) and a
//! *bytes* key (the vertex and fragment buffer bytes the draw carries), and is
//! counted against the previous frame and the last four:
//!
//! * `rep_full` - struct and bytes both seen in the previous frame;
//! * `rep_struct_only` - struct seen, bytes differ (a drag: same draw, new positions);
//! * `rep_new` - neither; `rep_full_w4` - full key seen in any of the last four
//!   frames; `rep_partial` - draws that carried a buffer too large to hash whole.
//!
//! A stream is a run of draws continuing one render pass. `rep_streams` counts
//! them and `rep_streams_repeat` those whose ordered list of draw keys appeared in
//! the previous frame. The counts land in the store-route second windows.

use std::collections::HashSet;
use std::sync::Mutex;

#[derive(Default)]
struct Frames {
    prev: [HashSet<(u64, u64)>; 4],
    prev_struct: HashSet<u64>,
    cur: HashSet<(u64, u64)>,
    cur_struct: HashSet<u64>,
    prev_streams: HashSet<u64>,
    cur_streams: HashSet<u64>,
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
        if f.prev_streams.contains(&f.stream_hash) {
            crate::runtime::drain::note_store_route("rep_streams_repeat");
        }
        f.cur_streams.insert(f.stream_hash);
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
        f.prev[i] = std::mem::take(&mut f.prev[i - 1]);
    }
    f.prev[0] = std::mem::take(&mut f.cur);
    f.prev_struct = std::mem::take(&mut f.cur_struct);
    f.prev_streams = std::mem::take(&mut f.cur_streams);
}

/// One draw: `new_stream` when it does not continue the previous draw's pass.
pub fn draw(struct_key: u64, bytes_key: u64, partial: bool, new_stream: bool) {
    let mut guard = STATE.lock().unwrap_or_else(|e| e.into_inner());
    let f = guard.get_or_insert_with(Frames::default);
    if new_stream {
        close_stream(f);
    }
    let full = (struct_key, bytes_key);
    crate::runtime::drain::note_store_route("rep_draws");
    if partial {
        crate::runtime::drain::note_store_route("rep_partial");
    }
    if f.prev[0].contains(&full) {
        crate::runtime::drain::note_store_route("rep_full");
    } else if f.prev_struct.contains(&struct_key) {
        crate::runtime::drain::note_store_route("rep_struct_only");
    } else {
        crate::runtime::drain::note_store_route("rep_new");
    }
    if f.prev.iter().any(|s| s.contains(&full)) {
        crate::runtime::drain::note_store_route("rep_full_w4");
    }
    f.cur.insert(full);
    f.cur_struct.insert(struct_key);
    f.stream_hash = f.stream_hash.rotate_left(5) ^ struct_key ^ bytes_key.rotate_left(17);
    f.stream_draws += 1;
}
