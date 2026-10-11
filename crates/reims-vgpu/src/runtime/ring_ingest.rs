//! TEMP (lab, `REIMS_VGPU_LAB_RING_INGEST=on`): take whole packets out of each
//! child ring into a device-side queue as soon as they are published, so the
//! guest's consumer pointer moves whether or not the drain is free to run them.
//!
//! **Why.** The drain advanced a child ring's head only when it read the next
//! packet, and it reads the next packet only after running the one before. One
//! EXEC can hold the drain 40-130 ms (Maps: compute readbacks, blits, 5K
//! consumers), and while it runs the ring does not move. Maps fills ch1 to the
//! byte (b207: 65480 of 65536) and the guest's paravirt driver then gives up on
//! the channel: `AppleParavirtVirtualChannel::resetHardwareAndReplay`, which is
//! an assertion for a virtual channel - the guest panics (b196). A device that
//! consumes its ring independently of execution keeps the guest's backpressure
//! where it belongs, on completion stamps.
//!
//! **What is safe to move.** Only packets the producer has finished publishing:
//! a header whose `total_size` is in range and fully covered by the published
//! bytes. A partial or malformed packet stays in the guest ring, and the drain's
//! own reader meets it there exactly as before (it reports faults, it waits for
//! partial packets). The drain reads the queue first, so ring order is kept.
//!
//! The head advancing before a packet runs is the drain's own rule already
//! (see the head write in `drain_child_fifo_g`): everything a packet needs is
//! taken at arrival, and a packet that cannot run holds an ordering position,
//! not a consumer pointer.
//!
//! Keyed by device handle and channel like [`crate::runtime::ring_tail`]: the
//! heartbeat ingests without the device lock, and the drain ingests on its way
//! into a channel. Both go through one mutex per channel.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use parking_lot::Mutex;

use crate::model::{
    published_byte_count, CHILD_REG_HEAD, CHILD_REG_TAIL, PACKET_HEADER_LEN, PACKET_TOTAL_SIZE,
};
use crate::runtime::host::{HostMemory, HostOps};

/// The most a channel's queue holds. A guest whose command buffers outrun the
/// device by more than this meets a full ring again, as before.
pub const QUEUE_CAP: usize = 8 << 20;

/// The modulus queue positions are taken under for [`crate::runtime::drain`]'s
/// reader: wide enough that `published_byte_count` accepts a full queue.
pub const QUEUE_SPAN: u32 = 1 << 30;

/// Whether the switch is on.
pub fn enabled() -> bool {
    if cfg!(test) {
        return false;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("REIMS_VGPU_LAB_RING_INGEST").is_ok_and(|v| v == "on");
        crate::observe::off(format!("lab_ring_ingest enabled={on}"));
        on
    })
}

/// Where a child ring lives, as the drain set it up.
#[derive(Clone, PartialEq, Eq)]
pub struct GuestRing {
    pub regs_gpa: u64,
    pub page_gpas: Vec<u64>,
    pub page_shift: u32,
    pub length: u32,
}

/// One channel's queue: the packets taken out of the guest ring, in order.
#[derive(Default)]
pub struct Queue {
    ring: Option<GuestRing>,
    /// Queue position of `bytes[0]`; the drain's `head` in queue mode.
    pub head: u32,
    bytes: VecDeque<u8>,
    /// The guest tail the last ingest read, for the dirty-tracking epoch the
    /// drain opens over the packets it then reads.
    pub guest_tail: u32,
}

impl Queue {
    /// Queue position one past the last byte held.
    #[must_use]
    pub fn tail(&self) -> u32 {
        self.head.wrapping_add(self.bytes.len() as u32)
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// A copy of everything held, from `head`.
    #[must_use]
    pub fn snapshot(&self) -> Vec<u8> {
        let (a, b) = self.bytes.as_slices();
        let mut v = Vec::with_capacity(a.len() + b.len());
        v.extend_from_slice(a);
        v.extend_from_slice(b);
        v
    }

    /// The drain read up to `next` (a queue position at or past `head`).
    pub fn consume_to(&mut self, next: u32) {
        let n = (next.wrapping_sub(self.head) as usize).min(self.bytes.len());
        self.bytes.drain(..n);
        self.head = self.head.wrapping_add(n as u32);
    }

    /// Point the queue at `ring`. A ring that moved or was rebuilt starts an
    /// empty queue: what was held belonged to the old ring's generation.
    pub fn arm(&mut self, ring: GuestRing) {
        if self.ring.as_ref() != Some(&ring) {
            if !self.bytes.is_empty() {
                crate::runtime::drain::note_store_route("lab_ring_ingest_rearm_dropped");
            }
            self.bytes.clear();
            self.ring = Some(ring);
        }
    }

    pub fn disarm(&mut self) {
        self.ring = None;
        self.bytes.clear();
    }

    /// Move every whole published packet from the guest ring into the queue and
    /// write the guest's head past them. Returns the bytes moved.
    pub fn ingest<H: HostMemory + HostOps>(&mut self, host: &mut H, page_size: usize) -> usize {
        let Some(ring) = self.ring.clone() else {
            return 0;
        };
        if ring.length == 0 {
            return 0;
        }
        let (Ok(mut head), Ok(tail)) = (
            crate::runtime::host::read_u32(host, ring.regs_gpa + CHILD_REG_HEAD),
            crate::runtime::host::read_u32(host, ring.regs_gpa + CHILD_REG_TAIL),
        ) else {
            return 0;
        };
        self.guest_tail = tail;
        let start = head;
        let mut moved = 0usize;
        loop {
            let Some(available) = published_byte_count(head, tail, ring.length) else {
                break; // desynced: the drain's own reader reports it
            };
            if available < PACKET_HEADER_LEN {
                break;
            }
            let mut hdr = [0u8; PACKET_HEADER_LEN as usize];
            if read_ring(host, &ring, head, &mut hdr).is_err() {
                break;
            }
            let total = u32::from_le_bytes(
                hdr[PACKET_TOTAL_SIZE..PACKET_TOTAL_SIZE + 4].try_into().unwrap_or([0; 4]),
            );
            if total < PACKET_HEADER_LEN || total > ring.length || available < total {
                break; // malformed or partial: left for the drain's reader
            }
            if self.bytes.len() + total as usize > QUEUE_CAP {
                crate::runtime::drain::note_store_route("lab_ring_ingest_queue_full");
                break;
            }
            let mut pkt = vec![0u8; total as usize];
            if read_ring(host, &ring, head, &mut pkt).is_err() {
                break;
            }
            self.bytes.extend(pkt);
            head = head.wrapping_add(total);
            moved += total as usize;
        }
        census::QMAX.fetch_max(self.bytes.len() as u64, std::sync::atomic::Ordering::Relaxed);
        if let Some(left) = published_byte_count(head, tail, ring.length) {
            census::LEFT_MAX.fetch_max(u64::from(left), std::sync::atomic::Ordering::Relaxed);
        }
        if moved > 0 {
            let _ = crate::runtime::gpa_map::write_u32(
                host,
                ring.regs_gpa + CHILD_REG_HEAD,
                head,
                page_size,
            );
            let _ = start;
        }
        moved
    }
}

/// Read `out.len()` bytes of the guest ring at ring position `at`, a page run
/// at a time.
fn read_ring<M: HostMemory>(
    mem: &M,
    ring: &GuestRing,
    at: u32,
    out: &mut [u8],
) -> Result<(), crate::runtime::host::MemError> {
    let page_size = 1u32 << ring.page_shift;
    let mut done = 0usize;
    while done < out.len() {
        let off = at.wrapping_add(done as u32) % ring.length;
        let page = (off >> ring.page_shift) as usize;
        let in_page = off & (page_size - 1);
        let Some(&gpa) = ring.page_gpas.get(page) else {
            return Err(crate::runtime::host::MemError::Unmapped);
        };
        let to_ring_end = ring.length - off;
        let n = ((page_size - in_page).min(to_ring_end) as usize).min(out.len() - done);
        mem.read_gpa(gpa + u64::from(in_page), &mut out[done..done + n])?;
        done += n;
    }
    Ok(())
}

/// TEMP (lab): one `lab_ring_ingest_census` line a second - bytes the heartbeat
/// and the drain moved, the deepest queue, and the most the guest ring still
/// held right after an ingest (what the guest sees as used).
mod census {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    pub static HB_BYTES: AtomicU64 = AtomicU64::new(0);
    pub static DRAIN_BYTES: AtomicU64 = AtomicU64::new(0);
    pub static HB_CALLS: AtomicU64 = AtomicU64::new(0);
    pub static QMAX: AtomicU64 = AtomicU64::new(0);
    pub static LEFT_MAX: AtomicU64 = AtomicU64::new(0);
    static LAST_MS: AtomicU64 = AtomicU64::new(0);

    pub fn maybe_emit() {
        let now = crate::observe::elapsed_ms() as u64;
        let last = LAST_MS.load(Relaxed);
        if now.saturating_sub(last) < 1000
            || LAST_MS.compare_exchange(last, now, Relaxed, Relaxed).is_err()
        {
            return;
        }
        crate::observe::off(format!(
            "lab_ring_ingest_census hb_calls={} hb_bytes={} drain_bytes={} queue_max={} guest_left_max={}",
            HB_CALLS.swap(0, Relaxed),
            HB_BYTES.swap(0, Relaxed),
            DRAIN_BYTES.swap(0, Relaxed),
            QMAX.swap(0, Relaxed),
            LEFT_MAX.swap(0, Relaxed),
        ));
    }
}

/// The drain's own ingest, counted apart from the heartbeat's.
pub fn note_drain_moved(n: usize) {
    census::DRAIN_BYTES.fetch_add(n as u64, std::sync::atomic::Ordering::Relaxed);
}

type Registry = HashMap<(u64, u32), Arc<Mutex<Queue>>>;
static QUEUES: Mutex<Option<Registry>> = Mutex::new(None);

/// `device`'s queue for child channel `channel`.
pub fn queue(device: u64, channel: u32) -> Arc<Mutex<Queue>> {
    let mut q = QUEUES.lock();
    Arc::clone(
        q.get_or_insert_with(HashMap::new)
            .entry((device, channel))
            .or_default(),
    )
}

/// Every armed queue of `device`.
fn queues_of(device: u64) -> Vec<(u32, Arc<Mutex<Queue>>)> {
    let q = QUEUES.lock();
    q.as_ref()
        .map(|m| {
            m.iter()
                .filter(|((d, _), _)| *d == device)
                .map(|((_, c), v)| (*c, Arc::clone(v)))
                .collect()
        })
        .unwrap_or_default()
}

/// Whether `device`'s channel `channel` holds packets the drain has not read.
pub fn pending(device: u64, channel: u32) -> bool {
    if !enabled() {
        return false;
    }
    let q = QUEUES.lock();
    q.as_ref()
        .and_then(|m| m.get(&(device, channel)))
        .is_some_and(|v| v.try_lock().is_none_or(|g| !g.is_empty()))
}

/// The heartbeat's half: ingest every armed channel whose queue is not busy.
/// Returns the channels that moved anything (the drain is owed a wake).
pub fn heartbeat<H: HostMemory + HostOps>(device: u64, host: &mut H, page_size: usize) -> u32 {
    if !enabled() {
        return 0;
    }
    let mut moved_mask = 0u32;
    census::HB_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    census::maybe_emit();
    for (ch, q) in queues_of(device) {
        let Some(mut g) = q.try_lock() else {
            continue;
        };
        let n = g.ingest(host, page_size);
        census::HB_BYTES.fetch_add(n as u64, std::sync::atomic::Ordering::Relaxed);
        if n > 0 {
            moved_mask |= 1 << (ch & 31);
            crate::runtime::drain::note_store_route("lab_ring_ingest_heartbeat_moved");
        }
    }
    moved_mask
}

/// Drop `device`'s queue for `channel` (its ring was torn down).
pub fn reset(device: u64, channel: u32) {
    if let Some(v) = QUEUES.lock().as_ref().and_then(|m| m.get(&(device, channel))) {
        v.lock().disarm();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn consume_moves_the_head_and_drops_the_bytes() {
        let mut q = Queue::default();
        q.head = 100;
        q.bytes.extend([1u8, 2, 3, 4, 5]);
        assert_eq!(q.tail(), 105);
        q.consume_to(103);
        assert_eq!(q.head, 103);
        assert_eq!(q.snapshot(), vec![4, 5]);
        q.consume_to(200);
        assert!(q.is_empty());
        assert_eq!(q.head, 105);
    }
}
