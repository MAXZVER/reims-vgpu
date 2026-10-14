"""matrix.py [SECONDS] [K=V ...]: run stress.exe over the gate matrix of reims-vgpu-dirty.c.

Each row is one configuration of the REIMS_VGPU_LAB_* gates (and of the model: STRESS_*), with what the oracle must say:
`0` violations, or `>0` for the controls that show the oracle is looking (a drain that reads through the lagged token bit;
packets the drain sees before their doorbell, i.e. the pre-existing "hole", which only DIRTY_HOLE=close closes).
Prints one block per row and exits non-zero if any row does not give what it must. ONLY=<substring> selects rows.
"""
import subprocess, sys, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
secs = sys.argv[1] if len(sys.argv) > 1 else "12"
extra = dict(a.split('=', 1) for a in sys.argv[2:])
G = 'REIMS_VGPU_LAB_'
NAMED = {'STRESS_NAMED_ONLY': '1', 'STRESS_NONDB_HZ': '250'}
HOLE = {'STRESS_HOLE': '25', 'STRESS_HOLE_US': '40'}
RING = {'STRESS_RING': '1', 'STRESS_RING_SKIP': '30', 'STRESS_HOLE_US': '40'}
# Pinned to the queried path: a hashed set (DIRTY_HASH, the default since n178) is never protected, so these rows
# would stop exercising WP. The mixed path is the `hash:` rows and `DEFAULTS + wp`.
# DIRTY_WP_AGE_MS=0: the calm age floor (1 s + 2 ms a run) would leave the churned sets of a 10-s row unprotected.
WP = {G + 'DIRTY_WP': 'on', G + 'DIRTY_WP_MIN': '8', G + 'DIRTY_WP_STREAK': '2', G + 'DIRTY_HASH': 'off',
      G + 'DIRTY_WP_AGE_MS': '0'}
WP0 = {k: v for k, v in WP.items() if k != G + 'DIRTY_HASH'}   # WP with the hash switch left to the row
EXACT3 = {G + 'DIRTY_STEAL': 'on', G + 'GEN_MEMO': 'on', G + 'GEN_LOCKFREE': 'on'}
cfgs = [
    ('off', {}, '0'),
    ('steal=on', {G + 'DIRTY_STEAL': 'on'}, '0'),
    ('steal=verify', {G + 'DIRTY_STEAL': 'verify'}, '0'),
    ('memo=on', {G + 'GEN_MEMO': 'on'}, '0'),
    ('memo=verify', {G + 'GEN_MEMO': 'verify'}, '0'),
    ('lockfree=on', {G + 'GEN_LOCKFREE': 'on'}, '0'),
    ('lockfree=verify', {G + 'GEN_LOCKFREE': 'verify'}, '0'),
    ('steal+memo+lockfree on', dict(EXACT3), '0'),
    ('steal+memo+lockfree verify', {G + 'DIRTY_STEAL': 'verify', G + 'GEN_MEMO': 'verify', G + 'GEN_LOCKFREE': 'verify'}, '0'),
    ('named+nondb: off', dict(NAMED), '0'),
    ('named+nondb: epoch_named', dict(NAMED, **{G + 'EPOCH_NAMED': 'on'}), '0'),
    ('named+nondb: epoch_named + exact3', dict(NAMED, **EXACT3, **{G + 'EPOCH_NAMED': 'on'}), '0'),
    ('hole 25%: off (the pre-existing hole)', dict(NAMED, **HOLE), '>0'),
    ('hole 25%: epoch_named', dict(NAMED, **HOLE, **{G + 'EPOCH_NAMED': 'on'}), '>0'),
    ('hole 25%: epoch_named + hole=census', dict(NAMED, **HOLE, **{G + 'EPOCH_NAMED': 'on', G + 'DIRTY_HOLE': 'census'}), '>0'),
    ('hole 25%: epoch_named + hole=close', dict(NAMED, **HOLE, **{G + 'EPOCH_NAMED': 'on', G + 'DIRTY_HOLE': 'close'}), '0'),
    ('hole 25%: epoch_named + hole=close + exact3', dict(NAMED, **HOLE, **EXACT3, **{G + 'EPOCH_NAMED': 'on', G + 'DIRTY_HOLE': 'close'}), '0'),
    ('ring (real protocol, 30% unrung): off', dict(NAMED, **RING), '>0'),
    ('ring (real protocol, 30% unrung): hole=close', dict(NAMED, **RING, **{G + 'DIRTY_HOLE': 'close'}), '?'),
    ('ring (real protocol, 30% unrung): hole=tail', dict(NAMED, **RING, **{G + 'DIRTY_HOLE': 'tail'}), '0'),
    ('ring: hole=tail + epoch_named + exact3', dict(NAMED, **RING, **EXACT3, **{G + 'DIRTY_HOLE': 'tail', G + 'EPOCH_NAMED': 'on'}), '0'),
    ('period=on (every set gated)', {G + 'DIRTY_PERIOD': 'on', G + 'DIRTY_PERIOD_BIG': '1'}, '0'),
    ('period=on + exact3, named+nondb', dict(NAMED, **EXACT3, **{G + 'DIRTY_PERIOD': 'on', G + 'DIRTY_PERIOD_BIG': '1'}), '0'),
    ('bigsplit+period on (every set big)', dict(NAMED, **EXACT3, **{G + 'DIRTY_PERIOD': 'on', G + 'DIRTY_BIGSPLIT': 'on', G + 'DIRTY_PERIOD_BIG': '1'}), '0'),
    ('bigsplit on, ring + hole=tail', dict(NAMED, **RING, **EXACT3, **{G + 'DIRTY_BIGSPLIT': 'on', G + 'DIRTY_PERIOD_BIG': '2', G + 'DIRTY_HOLE': 'tail'}), '0'),
    ('rsettle on', {G + 'DIRTY_RSETTLE': 'on'}, '0'),
    ('rsettle + bigsplit + period + exact3, named', dict(NAMED, **EXACT3, **{G + 'DIRTY_RSETTLE': 'on', G + 'DIRTY_BIGSPLIT': 'on', G + 'DIRTY_PERIOD': 'on', G + 'DIRTY_PERIOD_BIG': '2'}), '0'),
    ('rsettle + bigsplit, ring + hole=tail', dict(NAMED, **RING, **EXACT3, **{G + 'DIRTY_RSETTLE': 'on', G + 'DIRTY_BIGSPLIT': 'on', G + 'DIRTY_PERIOD_BIG': '2', G + 'DIRTY_HOLE': 'tail'}), '0'),
    ('bigphase + period (every set big)', dict(NAMED, **EXACT3, **{G + 'DIRTY_BIGPHASE': 'on', G + 'DIRTY_PERIOD': 'on', G + 'DIRTY_PERIOD_BIG': '1'}), '0'),
    ('bigphase (sets >= 2 pages big), ring + hole=tail', dict(NAMED, **RING, **EXACT3, **{G + 'DIRTY_BIGPHASE': 'on', G + 'DIRTY_PERIOD_BIG': '2', G + 'DIRTY_HOLE': 'tail'}), '0'),
    ('frontier, ring + hole=tail', dict(NAMED, **RING, **{G + 'DIRTY_HOLE': 'tail', G + 'DIRTY_FRONTIER': 'on'}), '0'),
    ('frontier + named + tailwake + exact3, ring + hole=tail', dict(NAMED, **RING, **EXACT3, **{G + 'DIRTY_HOLE': 'tail', G + 'DIRTY_FRONTIER': 'on', G + 'EPOCH_NAMED': 'on', G + 'DIRTY_TAILWAKE': 'on'}), '0'),
    ('frontier + named + bigphase, ring + hole=tail', dict(NAMED, **RING, **EXACT3, **{G + 'DIRTY_HOLE': 'tail', G + 'DIRTY_FRONTIER': 'on', G + 'EPOCH_NAMED': 'on', G + 'DIRTY_BIGPHASE_BIG': '2'}), '0'),
    ('CONTROL: drain reads lagged', {'STRESS_DRAIN_LAGGED': '1'}, '>0'),
    ('DEFAULTS (no gate env): named model', dict(NAMED), '0'),
    ('DEFAULTS (no gate env): real ring protocol', dict(NAMED, **RING), '0'),
    ('wp (sets >= 8 pages, 2 clean syncs)', dict(WP), '0'),
    ('wp + exact3, named model', dict(NAMED, **WP, **EXACT3), '0'),
    ('wp + frontier + named, ring + hole=tail', dict(NAMED, **RING, **EXACT3, **WP, **{G + 'DIRTY_HOLE': 'tail', G + 'DIRTY_FRONTIER': 'on', G + 'EPOCH_NAMED': 'on'}), '0'),
    ('CONTROL: wp protects without pausing the writers', dict(WP, STRESS_WP_NOPAUSE='1'), '>0'),
    ('DEFAULTS + wp: named model', dict(NAMED, **WP), '0'),
    ('DEFAULTS + wp, hash by the cost model: named model', dict(NAMED, **WP0), '0'),
    ('DEFAULTS + wp: real ring protocol + hole=tail', dict(NAMED, **RING, **WP, **{G + 'DIRTY_HOLE': 'tail'}), '0'),
    ('wp nopause (writers never paused)', dict(WP, STRESS_WP_NOPAUSE='1', **{G + 'DIRTY_WP_NOPAUSE': 'on'}), '0'),
    ('wp nopause, every set (min 1), 1 clean sync', dict(WP, STRESS_WP_NOPAUSE='1', **{G + 'DIRTY_WP_NOPAUSE': 'on', G + 'DIRTY_WP_MIN': '1', G + 'DIRTY_WP_STREAK': '1'}), '0'),
    ('wp nopause + exact3 + rsettle, named', dict(NAMED, **WP, **EXACT3, STRESS_WP_NOPAUSE='1', **{G + 'DIRTY_WP_NOPAUSE': 'on', G + 'DIRTY_RSETTLE': 'on'}), '0'),
    ('DEFAULTS + wp nopause: ring + hole=tail', dict(NAMED, **RING, **WP, STRESS_WP_NOPAUSE='1', **{G + 'DIRTY_HOLE': 'tail', G + 'DIRTY_WP_NOPAUSE': 'on'}), '0'),
    ('ticket_batch 2 (named model)', dict(NAMED, **{G + 'DIRTY_TICKET_BATCH': '2'}), '0'),
    ('DEFAULTS + ticket_batch 2: real ring protocol + hole=tail', dict(NAMED, **RING, **{G + 'DIRTY_HOLE': 'tail', G + 'DIRTY_TICKET_BATCH': '2'}), '0'),
    ('DEFAULTS + wp nopause + wp_hole + ticket_batch 2: ring + hole=tail', dict(NAMED, **RING, **WP, STRESS_WP_NOPAUSE='1', **{G + 'DIRTY_HOLE': 'tail', G + 'DIRTY_WP_NOPAUSE': 'on', G + 'DIRTY_WP_HOLE': 'on', G + 'DIRTY_TICKET_BATCH': '2'}), '0'),
    ('pstamp (named model)', dict(NAMED, **{G + 'DIRTY_PSTAMP': 'on'}), '0'),
    ('pstamp + exact3, named', dict(NAMED, **EXACT3, **{G + 'DIRTY_PSTAMP': 'on'}), '0'),
    ('DEFAULTS + pstamp: real ring protocol + hole=tail', dict(NAMED, **RING, **{G + 'DIRTY_HOLE': 'tail', G + 'DIRTY_PSTAMP': 'on'}), '0'),
    ('DEFAULTS + pstamp + ticket_batch 2 + wp nopause + wp_hole: ring + hole=tail', dict(NAMED, **RING, **WP, STRESS_WP_NOPAUSE='1', **{G + 'DIRTY_HOLE': 'tail', G + 'DIRTY_WP_NOPAUSE': 'on', G + 'DIRTY_WP_HOLE': 'on', G + 'DIRTY_TICKET_BATCH': '2', G + 'DIRTY_PSTAMP': 'on'}), '0'),
    ('fg split min 4 (named model)', dict(NAMED, **{G + 'DIRTY_FG_SPLIT_MIN': '4'}), '0'),
    ('DEFAULTS + fg split min 4: real ring protocol + hole=tail', dict(NAMED, **RING, **{G + 'DIRTY_HOLE': 'tail', G + 'DIRTY_FG_SPLIT_MIN': '4'}), '0'),
    ('packet scope, ring + hole=tail', dict(NAMED, **RING, STRESS_PACKET_SCOPE='1', **{G + 'DIRTY_HOLE': 'tail'}), '0'),
    ('packet scope + chan prefetch, ring + hole=tail', dict(NAMED, **RING, STRESS_PACKET_SCOPE='1', **{G + 'DIRTY_HOLE': 'tail', G + 'DIRTY_CHAN_PREFETCH': 'on'}), '0'),
    ('CONTROL: packet scope without hole=tail, ring', dict(NAMED, **RING, STRESS_PACKET_SCOPE='1'), '>0'),
    ('DEFAULTS + packet scope + chan prefetch + wp nopause: ring + hole=tail', dict(NAMED, **RING, **WP, STRESS_WP_NOPAUSE='1', STRESS_PACKET_SCOPE='1', **{G + 'DIRTY_HOLE': 'tail', G + 'DIRTY_WP_NOPAUSE': 'on', G + 'DIRTY_CHAN_PREFETCH': 'on'}), '0'),
    ('wp nopause + wp_hole (fault opens a page, heals)', dict(WP, STRESS_WP_NOPAUSE='1', **{G + 'DIRTY_WP_NOPAUSE': 'on', G + 'DIRTY_WP_HOLE': 'on'}), '0'),
    ('wp nopause + wp_hole, every set (min 1), 1 clean sync', dict(WP, STRESS_WP_NOPAUSE='1', **{G + 'DIRTY_WP_NOPAUSE': 'on', G + 'DIRTY_WP_HOLE': 'on', G + 'DIRTY_WP_MIN': '1', G + 'DIRTY_WP_STREAK': '1'}), '0'),
    ('DEFAULTS + wp nopause + wp_hole: ring + hole=tail', dict(NAMED, **RING, **WP, STRESS_WP_NOPAUSE='1', **{G + 'DIRTY_HOLE': 'tail', G + 'DIRTY_WP_NOPAUSE': 'on', G + 'DIRTY_WP_HOLE': 'on'}), '0'),
    ('DEFAULTS + wp nopause + wp_hole, settle_skip=off: ring + hole=tail', dict(NAMED, **RING, **WP, STRESS_WP_NOPAUSE='1', **{G + 'DIRTY_HOLE': 'tail', G + 'DIRTY_WP_NOPAUSE': 'on', G + 'DIRTY_WP_HOLE': 'on', G + 'DIRTY_SETTLE_SKIP': 'off'}), '0'),
    ('hash: every set hashed', {G + 'DIRTY_HASH': 'on', G + 'DIRTY_HASH_ALL': 'on'}, '0'),
    ('hash: cost model, named + exact3', dict(NAMED, **EXACT3, **{G + 'DIRTY_HASH': 'on'}), '0'),
    ('hash: every set + wp nopause, ring + hole=tail', dict(NAMED, **RING, **WP0, STRESS_WP_NOPAUSE='1', **{G + 'DIRTY_HOLE': 'tail', G + 'DIRTY_WP_NOPAUSE': 'on', G + 'DIRTY_HASH': 'on', G + 'DIRTY_HASH_ALL': 'on'}), '0'),
    ('hash: every set, small max (mixed), ring + hole=tail', dict(NAMED, **RING, **{G + 'DIRTY_HOLE': 'tail', G + 'DIRTY_HASH': 'on', G + 'DIRTY_HASH_ALL': 'on', G + 'DIRTY_HASH_MAX': '64'}), '0'),
    ('hash: due on (background by read gaps), named + exact3', dict(NAMED, **EXACT3, **{G + 'DIRTY_HASH': 'on', G + 'DIRTY_HASH_DUE': 'on'}), '0'),
    ('hash: due on, every set, ring + hole=tail', dict(NAMED, **RING, **{G + 'DIRTY_HOLE': 'tail', G + 'DIRTY_HASH': 'on', G + 'DIRTY_HASH_ALL': 'on', G + 'DIRTY_HASH_DUE': 'on'}), '0'),
    ('hash=off: the queried path, named + exact3', dict(NAMED, **EXACT3, **{G + 'DIRTY_HASH': 'off'}), '0'),
    ('hash=off + wp nopause: ring + hole=tail', dict(NAMED, **RING, **WP0, STRESS_WP_NOPAUSE='1', **{G + 'DIRTY_HOLE': 'tail', G + 'DIRTY_WP_NOPAUSE': 'on', G + 'DIRTY_HASH': 'off'}), '0'),
    # DIRTY_WP_HASHED (the Intel default since 2026-10-09, floor 64): clean hashed sets write-protected and read as
    # synced with no hash. The floor is lowered to the model's set sizes.
    ('hash+wp_hashed: every set hashed, wp nopause', dict(WP0, STRESS_WP_NOPAUSE='1', **{G + 'DIRTY_WP_NOPAUSE': 'on', G + 'DIRTY_HASH': 'on', G + 'DIRTY_HASH_ALL': 'on', G + 'DIRTY_WP_HASHED': 'on', G + 'DIRTY_WP_HASHED_MIN': '8'}), '0'),
    ('hash+wp_hashed: every set, named + ring + hole=tail', dict(NAMED, **RING, **WP0, STRESS_WP_NOPAUSE='1', **{G + 'DIRTY_HOLE': 'tail', G + 'DIRTY_WP_NOPAUSE': 'on', G + 'DIRTY_HASH': 'on', G + 'DIRTY_HASH_ALL': 'on', G + 'DIRTY_WP_HASHED': 'on', G + 'DIRTY_WP_HASHED_MIN': '8'}), '0'),
    ('hash+wp_hashed: cost model (mixed), named + exact3', dict(NAMED, **EXACT3, **WP0, STRESS_WP_NOPAUSE='1', **{G + 'DIRTY_WP_NOPAUSE': 'on', G + 'DIRTY_HASH': 'on', G + 'DIRTY_WP_HASHED': 'on', G + 'DIRTY_WP_HASHED_MIN': '8'}), '0'),
    ('hash+wp_hashed: every set, paused protect', dict(WP0, **{G + 'DIRTY_HASH': 'on', G + 'DIRTY_HASH_ALL': 'on', G + 'DIRTY_WP_HASHED': 'on', G + 'DIRTY_WP_HASHED_MIN': '8'}), '0'),
    # Informational, not a control: unlike the queried path, a hashed set is hashed again once wholly read-only, so a
    # store that raced the unpaused protect is caught by that hash (2026-10-09: 0 violations, ~480 faults a row).
    ('hash+wp_hashed: protects without pausing the writers', dict(WP0, STRESS_WP_NOPAUSE='1', **{G + 'DIRTY_HASH': 'on', G + 'DIRTY_HASH_ALL': 'on', G + 'DIRTY_WP_HASHED': 'on', G + 'DIRTY_WP_HASHED_MIN': '8'}), '?'),
    # DIRTY_WP_OVERLAP (opt-in, 2026-10-09): sets protected over pages another protected set holds; a page keeps
    # read-only while any protected set counts it. Every set eligible (min 1, 1 clean sync) maximises overlaps.
    ('wp overlap: nopause, every set (min 1), 1 clean sync', dict(WP, STRESS_WP_NOPAUSE='1', **{G + 'DIRTY_WP_NOPAUSE': 'on', G + 'DIRTY_WP_MIN': '1', G + 'DIRTY_WP_STREAK': '1', G + 'DIRTY_WP_OVERLAP': 'on'}), '0'),
    ('wp overlap: paused protect, every set', dict(WP, **{G + 'DIRTY_WP_MIN': '1', G + 'DIRTY_WP_STREAK': '1', G + 'DIRTY_WP_OVERLAP': 'on'}), '0'),
    ('wp overlap: nopause + merge off, named + ring + hole=tail', {**NAMED, **RING, **WP, 'STRESS_WP_NOPAUSE': '1', G + 'DIRTY_HOLE': 'tail', G + 'DIRTY_WP_NOPAUSE': 'on', G + 'DIRTY_WP_MIN': '1', G + 'DIRTY_WP_STREAK': '1', G + 'DIRTY_WP_OVERLAP': 'on', G + 'DIRTY_WP_MERGE': 'off'}, '0'),
    ('wp overlap + hash+wp_hashed: every set, nopause', {**WP0, 'STRESS_WP_NOPAUSE': '1', G + 'DIRTY_WP_NOPAUSE': 'on', G + 'DIRTY_HASH': 'on', G + 'DIRTY_WP_HASHED': 'on', G + 'DIRTY_WP_HASHED_MIN': '1', G + 'DIRTY_WP_MIN': '1', G + 'DIRTY_WP_STREAK': '1', G + 'DIRTY_WP_OVERLAP': 'on'}, '0'),
    # TEMP lab DIRTY_WP_BRIDGE (b451, opt-in): a protect remaps over gaps that are read-only or held only by hashed
    # sets; a store into a bridged page faults back like an orphan's. Every set hashed (all gaps bridgeable), the cost
    # model (logged sets stop the bridge), and the ring/hole rows.
    ('wp bridge: every set hashed, nopause, 1 clean sync', {**WP0, 'STRESS_WP_NOPAUSE': '1', G + 'DIRTY_WP_NOPAUSE': 'on', G + 'DIRTY_HASH': 'on', G + 'DIRTY_HASH_ALL': 'on', G + 'DIRTY_WP_HASHED': 'on', G + 'DIRTY_WP_HASHED_MIN': '1', G + 'DIRTY_WP_MIN': '1', G + 'DIRTY_WP_STREAK': '1', G + 'DIRTY_WP_OVERLAP': 'on', G + 'DIRTY_WP_BRIDGE': 'on'}, '0'),
    ('wp bridge: cost model (mixed), named + exact3', dict(NAMED, **EXACT3, **WP0, STRESS_WP_NOPAUSE='1', **{G + 'DIRTY_WP_NOPAUSE': 'on', G + 'DIRTY_HASH': 'on', G + 'DIRTY_WP_HASHED': 'on', G + 'DIRTY_WP_HASHED_MIN': '1', G + 'DIRTY_WP_MIN': '1', G + 'DIRTY_WP_OVERLAP': 'on', G + 'DIRTY_WP_BRIDGE': 'on'}), '0'),
    ('wp bridge: every set hashed, named + ring + hole=tail', dict(NAMED, **RING, **WP0, STRESS_WP_NOPAUSE='1', **{G + 'DIRTY_HOLE': 'tail', G + 'DIRTY_WP_NOPAUSE': 'on', G + 'DIRTY_HASH': 'on', G + 'DIRTY_HASH_ALL': 'on', G + 'DIRTY_WP_HASHED': 'on', G + 'DIRTY_WP_HASHED_MIN': '1', G + 'DIRTY_WP_MIN': '1', G + 'DIRTY_WP_OVERLAP': 'on', G + 'DIRTY_WP_BRIDGE': 'on'}), '0'),
    ('DEFAULTS + wp bridge: hashed, ring + hole=tail', dict(NAMED, **RING, **WP0, STRESS_WP_NOPAUSE='1', **{G + 'DIRTY_HOLE': 'tail', G + 'DIRTY_WP_NOPAUSE': 'on', G + 'DIRTY_HASH': 'on', G + 'DIRTY_WP_HASHED': 'on', G + 'DIRTY_WP_HASHED_MIN': '1', G + 'DIRTY_WP_BRIDGE': 'on'}), '0'),
    ('CONTROL wp overlap: protects without pausing the writers', dict(WP, STRESS_WP_NOPAUSE='1', **{G + 'DIRTY_WP_MIN': '1', G + 'DIRTY_WP_STREAK': '1', G + 'DIRTY_WP_OVERLAP': 'on'}), '>0'),
    ('CONTROL hash: drain reads lagged', {'STRESS_DRAIN_LAGGED': '1', G + 'DIRTY_HASH': 'on', G + 'DIRTY_HASH_ALL': 'on'}, '>0'),
    ('wp calm=off: wp nopause, ring + hole=tail', dict(NAMED, **RING, **WP, STRESS_WP_NOPAUSE='1', **{G + 'DIRTY_HOLE': 'tail', G + 'DIRTY_WP_NOPAUSE': 'on', G + 'DIRTY_WP_CALM': 'off'}), '0'),
    ('wp calm=off: wp nopause + wp_hole, ring + hole=tail', dict(NAMED, **RING, **WP, STRESS_WP_NOPAUSE='1', **{G + 'DIRTY_HOLE': 'tail', G + 'DIRTY_WP_NOPAUSE': 'on', G + 'DIRTY_WP_HOLE': 'on', G + 'DIRTY_WP_CALM': 'off'}), '0'),
    ('wp calm, real age floor: wp nopause, ring + hole=tail', dict(NAMED, **RING, **{k: v for k, v in WP.items() if k != G + 'DIRTY_WP_AGE_MS'}, STRESS_WP_NOPAUSE='1', **{G + 'DIRTY_HOLE': 'tail', G + 'DIRTY_WP_NOPAUSE': 'on'}), '0'),
    # TEMP lab DIRTY_WP_MERGE (on by default): merged give-backs and whole-set fault give-backs; the merge marks only the
    # runs, which holds under the measured WHPX remap (RW->RW keeps its bits, the stress default). The CONTROL drops them.
    ('wp merge (hv keeps RW->RW): wp nopause', dict(WP, STRESS_WP_NOPAUSE='1', STRESS_HV_KEEP_RW='1', **{G + 'DIRTY_WP_NOPAUSE': 'on', G + 'DIRTY_WP_MERGE': 'on'}), '0'),
    ('wp merge gap 1 (hv keeps RW->RW): every set, 1 clean sync', dict(WP, STRESS_WP_NOPAUSE='1', STRESS_HV_KEEP_RW='1', **{G + 'DIRTY_WP_NOPAUSE': 'on', G + 'DIRTY_WP_MERGE': 'on', G + 'DIRTY_WP_MERGE_GAP': '1', G + 'DIRTY_WP_MIN': '1', G + 'DIRTY_WP_STREAK': '1'}), '0'),
    ('DEFAULTS + wp merge (hv keeps RW->RW): ring + hole=tail', dict(NAMED, **RING, **WP, STRESS_WP_NOPAUSE='1', STRESS_HV_KEEP_RW='1', **{G + 'DIRTY_HOLE': 'tail', G + 'DIRTY_WP_NOPAUSE': 'on', G + 'DIRTY_WP_MERGE': 'on'}), '0'),
    ('CONTROL: wp merge with the worst-case hv (RW->RW drops)', dict(WP, STRESS_WP_NOPAUSE='1', STRESS_HV_KEEP_RW='0', **{G + 'DIRTY_WP_NOPAUSE': 'on', G + 'DIRTY_WP_MERGE': 'on', G + 'DIRTY_WP_MIN': '1', G + 'DIRTY_WP_STREAK': '1'}), '?'),
]
ORIGINAL = {G + 'DIRTY_STEAL': 'off', G + 'GEN_LOCKFREE': 'off', G + 'EPOCH_NAMED': 'off', G + 'DIRTY_HOLE': 'off'}
KEYS = ('steals', 'steal_same', 'steal_late', 'memo_hit', 'memo_fresh', 'memo_diff', 'memo_bad', 'unnamed_skipped',
        'lf_hit', 'lf_hitdiff', 'lf_gendiff', 'lf_race', 'lfw_hit', 'lfw_miss', 'lfw_diff', 'lfw_race', 'closed')
only = os.environ.get('ONLY')
env0 = dict(os.environ)
env0['PATH'] = r'C:\msys64\ucrt64\bin;' + env0['PATH']
bad = 0
for name, e, expect in cfgs:
    if only and only not in name:
        continue
    env = dict(env0)
    # Every row states the original path for the gates it does not name (they
    # default on since 2026-10-01), except the rows that test the defaults.
    if not name.startswith('DEFAULTS'):
        env.update(ORIGINAL)
    env.update(extra)
    env.update(e)
    r = subprocess.run([os.path.join(HERE, 'stress.exe'), secs], env=env, capture_output=True, text=True)
    res = re.search(r'RESULT (.*)', r.stdout)
    viol = re.search(r'violations=(\d+)', res.group(1)) if res else None
    n = int(viol.group(1)) if viol else -1
    ok = True if expect == '?' else (n == 0) if expect == '0' else (n > 0)
    bad += not ok
    lab = {}
    for l in r.stderr.splitlines():
        if l.startswith(('reims-dirty-lab', 'reims-dirty-hole')):
            for k, v in re.findall(r'(\w+)=(\d+)', l):
                if k in KEYS:
                    lab[k] = lab.get(k, 0) + int(v)
    print('%-5s %-48s expect %-2s %s' % ('ok' if ok else 'FAIL', name, expect, res.group(1) if res else 'NO RESULT'))
    print('      lab %s' % {k: v for k, v in lab.items() if v})
    for l in [l for l in r.stderr.splitlines() if 'VIOLATION' in l or 'READER' in l][:1]:
        print('      ', l)
print('rows that did not give what they must:', bad)
sys.exit(1 if bad else 0)
