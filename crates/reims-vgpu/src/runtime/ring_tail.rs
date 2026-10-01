//! Where each child ring's tail register is in guest memory, for the QEMU
//! shim's doorbell.
//!
//! A child ring's tail is a word in guest memory, not a value the doorbell
//! carries, so the drain can see packets whose doorbell it has not been told
//! about yet. The dirty tracker owes a packet a sync begun after the packet was
//! published; a doorbell's epoch gives that only for the packets published
//! before the doorbell. The shim therefore reads the tail *before* the
//! doorbell opens its epoch, and needs to know where the tail is without
//! taking the device lock a drain tranche may hold for milliseconds. The drain
//! publishes the address here whenever it sets a ring up; the shim reads it.
//!
//! Keyed by device handle and channel: a process may bind more than one device.

use std::collections::HashMap;
use std::sync::Mutex;

static TAILS: Mutex<Option<HashMap<(u64, u32), u64>>> = Mutex::new(None);

/// Record that `device`'s child channel `channel` keeps its tail at `gpa`.
pub fn publish(device: u64, channel: u32, gpa: u64) {
    let mut tails = TAILS.lock().unwrap_or_else(|e| e.into_inner());
    tails
        .get_or_insert_with(HashMap::new)
        .insert((device, channel), gpa);
}

/// Where `device`'s child channel `channel` keeps its tail, or 0 before the
/// drain has set that ring up.
pub fn lookup(device: u64, channel: u32) -> u64 {
    let tails = TAILS.lock().unwrap_or_else(|e| e.into_inner());
    tails
        .as_ref()
        .and_then(|t| t.get(&(device, channel)).copied())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_ring_is_unknown_until_published_and_devices_do_not_share_one() {
        assert_eq!(lookup(901, 2), 0);
        publish(901, 2, 0x1234_0000);
        assert_eq!(lookup(901, 2), 0x1234_0000);
        assert_eq!(lookup(902, 2), 0, "another device's ring is its own");
        assert_eq!(lookup(901, 3), 0, "another channel's ring is its own");
        publish(901, 2, 0x5678_0000);
        assert_eq!(lookup(901, 2), 0x5678_0000, "a re-set ring moves its tail");
    }
}
