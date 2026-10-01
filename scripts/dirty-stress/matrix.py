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
    ('CONTROL: drain reads lagged', {'STRESS_DRAIN_LAGGED': '1'}, '>0'),
    ('DEFAULTS (no gate env): named model', dict(NAMED), '0'),
    ('DEFAULTS (no gate env): real ring protocol', dict(NAMED, **RING), '0'),
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
