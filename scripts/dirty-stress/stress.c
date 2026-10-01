/*
 * Host-less stress test of reims-vgpu-dirty.c: the real tracker, linked against
 * a fake hypervisor, a fake guest and a fake drain.
 *
 * Contract under test (the dirty-sync exactness contract). The guest writes pages, then opens a
 * doorbell epoch (reims_vgpu_dirty_note_doorbell), then hands the drain a packet. The drain announces
 * a work scope, then for each set the packet reads it asks reims_vgpu_dirty_gen(); like the device it
 * keeps per set (generation, content as of its last gather) and re-gathers the pages that
 * reims_vgpu_dirty_written_since() names when the generation moved. After every read:
 *     for every page p of the set:  content[p] >= snapshot_at_doorbell[p]
 * i.e. every write ordered before the doorbell is reflected. Writes after the snapshot (guest races
 * from the free-running writer thread) may or may not be seen. A violation is a stale page.
 *
 * Fake hypervisor: dirty bits set by guest writes; reims_vgpu_dirty's memory_region_sync_dirty_ranges()
 * takes (test-and-clear) the bits of the ranges, spins, and only then deposits them in the VGA bitmap, so
 * two overlapping queries can interleave exactly the way the ticket protocol exists for.
 *
 * Controls: DRAIN_LAGGED=1 makes the drain read through the lagged token bit (a read that does not sync):
 * known unsafe, must produce violations, or the oracle is not looking.
 */
#include "qemu/osdep.h"
#include "qemu/thread.h"
#include "qemu/rcu.h"
#include "qemu/timer.h"
#include "qemu/main-loop.h"
#include "qemu/bitmap.h"
#include "qemu/int128.h"
#include "exec/target_page.h"
#include "system/memory.h"
#include "system/ramlist.h"
#include "reims-vgpu-dirty.h"
#include "system/whpx-wp.h"
#include "system/cpus.h"

#define NPAGES 16384
#define PAGE 4096ULL
#define MAXSETS 24

/* ---- the world ------------------------------------------------------------------------------ */
const TargetPageBits target_page = { .decided = true, .bits = 12, .mask = -4096 };
RAMList ram_list;
AddressSpace address_space_memory;
void (*memory_dirty_log_flush_hook)(void);
static MemoryRegion fake_mr;
static uint8_t hv_dirty[NPAGES];
static uint64_t wseq[NPAGES];          /* last write's sequence number per page */
static uint64_t gseq;
static QemuMutex glock;                /* the guest's big lock: a write and a snapshot are atomic */
static volatile int stop_flag;
static int g_hv_delay_us = 30;         /* the max random delay inside a hypervisor query */

static void spin_us(int us)
{
    int64_t t = get_clock() + (int64_t)us * 1000;

    while (get_clock() < t) {
        cpu_relax();
    }
}

static uint32_t rnd(uint32_t *st)
{
    *st = *st * 1664525u + 1013904223u;
    return *st >> 8;
}

/*
 * Lab DIRTY_WP: the fake hypervisor's read-only pages. A remap drops the dirty bits its range had
 * collected (the worst case WHPX may do), so only a protect taken with every writer paused right after a
 * query is exact. Writers are the "vCPUs": a store into a read-only page first goes to the fault hook,
 * which must make it writable; pause_all_vcpus waits for every store in flight and holds new ones.
 * STRESS_WP_NOPAUSE=1 is the control: pausing does nothing, so protects race the writers.
 */
bool (*whpx_wp_fault_hook)(uint64_t gpa);
void (*whpx_wp_invalidate_hook)(void);
static uint8_t hv_ro[NPAGES];
static uint64_t n_wp_faults, n_wp_remaps;
static int cfg_wp_nopause;
static QemuMutex vg_lock;
static QemuCond vg_cond;
static int vg_paused, vg_active;

int whpx_wp_remap(uint64_t gpa, uint64_t size, bool writable)
{
    uint64_t pg;

    for (pg = gpa / PAGE; pg < (gpa + size) / PAGE && pg < NPAGES; pg++) {
        qatomic_set(&hv_dirty[pg], 0);
        qatomic_set(&hv_ro[pg], !writable);
    }
    qatomic_inc(&n_wp_remaps);
    if (g_hv_delay_us) {
        spin_us(g_hv_delay_us / 2);
    }
    return 0;
}

void pause_all_vcpus(void)
{
    if (cfg_wp_nopause) {
        return;
    }
    qemu_mutex_lock(&vg_lock);
    vg_paused++;
    while (vg_active) {
        qemu_cond_wait(&vg_cond, &vg_lock);
    }
    qemu_mutex_unlock(&vg_lock);
}

void resume_all_vcpus(void)
{
    if (cfg_wp_nopause) {
        return;
    }
    qemu_mutex_lock(&vg_lock);
    vg_paused--;
    qemu_cond_broadcast(&vg_cond);
    qemu_mutex_unlock(&vg_lock);
}

static void vcpu_enter(void)
{
    qemu_mutex_lock(&vg_lock);
    while (vg_paused) {
        qemu_cond_wait(&vg_cond, &vg_lock);
    }
    vg_active++;
    qemu_mutex_unlock(&vg_lock);
}

static void vcpu_leave(void)
{
    qemu_mutex_lock(&vg_lock);
    if (--vg_active == 0) {
        qemu_cond_broadcast(&vg_cond);
    }
    qemu_mutex_unlock(&vg_lock);
}

struct QEMUBH {
    QEMUBHFunc *cb;
    void *opaque;
    int sched;
};
QEMUBH *reims_stress_bh_new(QEMUBHFunc *cb, void *opaque);
void reims_stress_bh_schedule(QEMUBH *bh);
static QEMUBH *stress_bh;

QEMUBH *reims_stress_bh_new(QEMUBHFunc *cb, void *opaque)
{
    QEMUBH *b = g_new0(QEMUBH, 1);

    b->cb = cb;
    b->opaque = opaque;
    stress_bh = b;
    return b;
}

void reims_stress_bh_schedule(QEMUBH *bh)
{
    qatomic_set(&bh->sched, 1);
}

static void *mainloop_thread(void *arg)
{
    (void)arg;
    rcu_register_thread();
    while (!stop_flag) {
        if (stress_bh && qatomic_xchg(&stress_bh->sched, 0)) {
            stress_bh->cb(stress_bh->opaque);
        }
        g_usleep(100);
    }
    rcu_unregister_thread();
    return NULL;
}

ram_addr_t memory_region_get_ram_addr(const MemoryRegion *mr) { (void)mr; return 0; }
void memory_region_ref(MemoryRegion *mr) { (void)mr; }
void memory_region_unref(MemoryRegion *mr) { (void)mr; }
void memory_region_set_log(MemoryRegion *mr, bool log, unsigned client) { (void)mr; (void)log; (void)client; }
void memory_region_reset_dirty(MemoryRegion *mr, hwaddr addr, hwaddr size, unsigned client)
{
    (void)mr; (void)addr; (void)size; (void)client;
}

void flatview_for_each_range(FlatView *fv, flatview_cb cb, void *opaque)
{
    (void)fv;
    cb(int128_make64(0), int128_make64(NPAGES * PAGE), &fake_mr, 0, opaque);
}

/* The hypervisor query: take the bits of the range, take a while, deposit them in the VGA bitmap. */
void memory_region_sync_dirty_ranges(MemoryRegion *mr, const MemoryRegionRange *ranges, size_t n)
{
    static __thread uint32_t st = 12345;
    DirtyMemoryBlocks *blocks;
    size_t i;
    uint64_t pg;

    (void)mr;
    if (st == 12345) {
        st ^= (uint32_t)(uintptr_t)&st;
    }
    for (i = 0; i < n; i++) {
        uint64_t first = ranges[i].start / PAGE, last = (ranges[i].start + ranges[i].len + PAGE - 1) / PAGE;
        uint64_t taken[256], nt = 0;
        uint64_t *bigtaken = NULL, bigcap = 0, bign = 0;

        if (last > NPAGES) {
            last = NPAGES;
        }
        for (pg = first; pg < last; pg++) {
            if (qatomic_read(&hv_dirty[pg]) && qatomic_xchg(&hv_dirty[pg], 0)) {
                if (nt < 256) {
                    taken[nt++] = pg;
                } else {
                    if (bign == bigcap) {
                        bigcap = bigcap ? bigcap * 2 : 1024;
                        bigtaken = g_realloc(bigtaken, bigcap * sizeof(uint64_t));
                    }
                    bigtaken[bign++] = pg;
                }
            }
        }
        if (g_hv_delay_us) {
            spin_us(rnd(&st) % (g_hv_delay_us + 1));   /* taken, not yet deposited */
        }
        rcu_read_lock();
        blocks = qatomic_rcu_read(&ram_list.dirty_memory[DIRTY_MEMORY_VGA]);
        for (pg = 0; pg < nt; pg++) {
            qatomic_or(&blocks->blocks[0][BIT_WORD(taken[pg])], BIT_MASK(taken[pg]));
        }
        for (pg = 0; pg < bign; pg++) {
            qatomic_or(&blocks->blocks[0][BIT_WORD(bigtaken[pg])], BIT_MASK(bigtaken[pg]));
        }
        rcu_read_unlock();
        g_free(bigtaken);
    }
}

/* ---- the guest ---------------------------------------------------------------------------------- */
typedef struct Set {
    uint64_t token;
    size_t count;
    uint64_t *gpas;                    /* sorted */
    int64_t *pos;                      /* page frame -> index in this set, or -1 */
    uint64_t gen_prev;
    uint64_t *content;                 /* wseq the reader last gathered, per page of the set */
} Set;

static Set sets[MAXSETS];
static int nsets;
static QemuMutex set_lock;             /* the drain's use of a set vs the churn thread */
static ReimsVgpuDirty *D;

typedef struct Packet {
    uint64_t *snap;                    /* wseq[] at the doorbell */
    uint32_t sets_mask;                /* which sets the packet reads, in index order */
    int channel;
    struct Packet *next;
} Packet;

static QemuMutex pk_lock;
static QemuCond pk_cond;
static Packet *pk_head, *pk_tail;

static QemuMutex hv_lock;              /* the harvest thread's doorbell */
static QemuCond hv_cond;
static uint64_t hv_asked, hv_done;

#define UNIVERSE 12000
#define NHOT 512
static uint64_t hot_pages[NHOT];

static int cmp64(const void *a, const void *b)
{
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;

    return x < y ? -1 : x > y;
}

static void set_build(Set *s, uint32_t *st, int kind)
{
    size_t cap = kind == 0 ? 1 + rnd(st) % 4 : kind == 1 ? 8 + rnd(st) % 64 : 400 + rnd(st) % 2500;
    uint64_t base = rnd(st) % (UNIVERSE - 3500), p;
    size_t n = 0, i, u = 0;
    uint64_t *tmp = g_new(uint64_t, cap + 4);

    if (kind == 0) {
        for (i = 0; i < cap; i++) {
            tmp[n++] = hot_pages[rnd(st) % NHOT] * PAGE;
        }
    } else {
        for (p = base; n < cap && p < UNIVERSE; p += 1 + (rnd(st) % 8 == 0 ? rnd(st) % 300 : 0)) {
            tmp[n++] = p * PAGE;
        }
        tmp[n++] = hot_pages[rnd(st) % NHOT] * PAGE;
        tmp[n++] = hot_pages[rnd(st) % NHOT] * PAGE;
    }
    qsort(tmp, n, sizeof(*tmp), cmp64);
    for (i = 0; i < n; i++) {
        if (u == 0 || tmp[u - 1] != tmp[i]) {
            tmp[u++] = tmp[i];
        }
    }
    n = u;
    s->count = n;
    s->gpas = tmp;
    s->pos = g_new(int64_t, NPAGES);
    for (i = 0; i < NPAGES; i++) {
        s->pos[i] = -1;
    }
    for (i = 0; i < n; i++) {
        s->pos[tmp[i] / PAGE] = (int64_t)i;
    }
    s->content = g_new0(uint64_t, n);
    s->gen_prev = 0;
    s->token = reims_vgpu_dirty_track(D, tmp, n, PAGE);
}

static void set_drop(Set *s)
{
    reims_vgpu_dirty_untrack(D, s->token);
    g_free(s->gpas);
    g_free(s->pos);
    g_free(s->content);
    s->token = 0;
}

static void do_write(uint32_t *st, int hot)
{
    uint64_t pg = hot && (rnd(st) & 1) ? hot_pages[rnd(st) % NHOT] : rnd(st) % UNIVERSE;

    while (qatomic_read(&hv_ro[pg])) {   /* lab DIRTY_WP: the store exits first */
        if (!whpx_wp_fault_hook || !whpx_wp_fault_hook(pg * PAGE)) {
            fprintf(stderr, "wp: a store into a read-only page was not taken\n");
            abort();
        }
        qatomic_inc(&n_wp_faults);
    }
    wseq[pg] = ++gseq;
    qatomic_set(&hv_dirty[pg], 1);
}

static int cfg_threads_readers = 3;
static int cfg_drain_lagged;
static uint64_t n_packets, n_violations, n_reads, n_regathers, n_reader_calls;
static int cfg_reader_spin = 30, cfg_reader_pure;
static uint64_t fixed_tokens[MAXSETS];
static int cfg_churn = 1;
static int cfg_named_only, cfg_nondb_hz, cfg_hole_pct, cfg_hole_us = 40;
static int cfg_ring_gap_us = 150;
/*
 * STRESS_RING=1: the real child-ring protocol. The guest stores the packet and then the ring tail
 * (publication), and rings the doorbell after that; the drain reads the tail from memory, so it can run a
 * packet before its doorbell is noted, or one appended behind the tail a doorbell covered. The doorbell reads
 * the tail before it opens its epoch (reims_vgpu_dirty_note_child_tail) and the drain reports every tail it
 * reads (reims_vgpu_dirty_note_ring_tail). STRESS_RING_SKIP=pct: the guest rings for only some packets (a
 * later doorbell covers the batch). STRESS_HOLE_US is the publish-to-doorbell delay.
 */
static int cfg_ring, cfg_ring_skip;
static uint32_t ring_tail_pub[8];      /* per channel: packets published, under pk_lock */

static void *guest_thread(void *arg)
{
    uint32_t st = 777;
    (void)arg;

    rcu_register_thread();
    while (!stop_flag) {
        Packet *p = g_new0(Packet, 1);
        int nw = 1 + rnd(&st) % 12, i;

        vcpu_enter();
        qemu_mutex_lock(&glock);
        for (i = 0; i < nw; i++) {
            do_write(&st, 1);
        }
        p->snap = g_memdup2(wseq, sizeof(wseq));
        qemu_mutex_unlock(&glock);
        vcpu_leave();
        /* -1..4: some doorbells name no channel (unless STRESS_NAMED_ONLY: then 1..4, like Tahoe) */
        p->channel = cfg_named_only ? 1 + (int)(rnd(&st) % 4) : (int)(rnd(&st) % 6) - 1;
        p->sets_mask = 0;
        for (i = 0; i < nsets; i++) {
            if (rnd(&st) % 3) {
                p->sets_mask |= 1u << i;
            }
        }
        if (cfg_ring) {
            uint32_t snap_tail;
            int ch = p->channel < 1 ? 1 : p->channel;

            p->channel = ch;
            qemu_mutex_lock(&pk_lock);
            if (pk_tail) {
                pk_tail->next = p;
            } else {
                pk_head = p;
            }
            pk_tail = p;
            ring_tail_pub[ch]++;
            qemu_cond_signal(&pk_cond);
            qemu_mutex_unlock(&pk_lock);
            if (!(cfg_ring_skip && (int)(rnd(&st) % 100) < cfg_ring_skip)) {
                spin_us(rnd(&st) % (cfg_hole_us + 1));
                qemu_mutex_lock(&pk_lock);
                snap_tail = ring_tail_pub[ch];
                qemu_mutex_unlock(&pk_lock);
                if (reims_vgpu_dirty_note_doorbell(D, ch)) {
                    qemu_mutex_lock(&hv_lock);
                    hv_asked++;
                    qemu_cond_signal(&hv_cond);
                    qemu_mutex_unlock(&hv_lock);
                }
                reims_vgpu_dirty_note_child_tail(D, ch, snap_tail, true);
            }
            spin_us(rnd(&st) % (cfg_ring_gap_us * 2 + 1));
            continue;
        }
        /* The doorbell (epoch), then the packet; or, in a "hole" packet, the packet first (the drain reads the
         * ring tail from guest memory, so it can see the packet before the doorbell write lands). */
        {
            bool hole = cfg_hole_pct && (int)(rnd(&st) % 100) < cfg_hole_pct;

            if (hole) {
                qemu_mutex_lock(&pk_lock);
                if (pk_tail) {
                    pk_tail->next = p;
                } else {
                    pk_head = p;
                }
                pk_tail = p;
                qemu_cond_signal(&pk_cond);
                qemu_mutex_unlock(&pk_lock);
                spin_us(rnd(&st) % (cfg_hole_us + 1));
            }
            if (reims_vgpu_dirty_note_doorbell(D, p->channel)) {
                qemu_mutex_lock(&hv_lock);
                hv_asked++;
                qemu_cond_signal(&hv_cond);
                qemu_mutex_unlock(&hv_lock);
            }
            if (!hole) {
                qemu_mutex_lock(&pk_lock);
                if (pk_tail) {
                    pk_tail->next = p;
                } else {
                    pk_head = p;
                }
                pk_tail = p;
                qemu_cond_signal(&pk_cond);
                qemu_mutex_unlock(&pk_lock);
            }
        }
        spin_us(rnd(&st) % (cfg_ring_gap_us * 2 + 1));
    }
    rcu_unregister_thread();
    return NULL;
}

/* Register writes that hand the device nothing (the display-interrupt enable): they only call
 * note_doorbell(-1), never publish a packet. */
static void *nondb_thread(void *arg)
{
    uint32_t st = 5150;
    (void)arg;

    rcu_register_thread();
    while (!stop_flag) {
        if (reims_vgpu_dirty_note_doorbell(D, -1)) {
            qemu_mutex_lock(&hv_lock);
            hv_asked++;
            qemu_cond_signal(&hv_cond);
            qemu_mutex_unlock(&hv_lock);
        }
        spin_us(1000000 / cfg_nondb_hz + rnd(&st) % 200);
    }
    rcu_unregister_thread();
    return NULL;
}

static void *racer_thread(void *arg)
{
    uint32_t st = 4242;
    (void)arg;

    rcu_register_thread();
    while (!stop_flag) {
        vcpu_enter();
        qemu_mutex_lock(&glock);
        do_write(&st, 1);
        qemu_mutex_unlock(&glock);
        vcpu_leave();
        spin_us(rnd(&st) % 40);
    }
    rcu_unregister_thread();
    return NULL;
}

static void *harvest_thread(void *arg)
{
    uint64_t target;
    (void)arg;

    rcu_register_thread();
    for (;;) {
        qemu_mutex_lock(&hv_lock);
        while (hv_done >= hv_asked && !stop_flag) {
            qemu_cond_wait(&hv_cond, &hv_lock);
        }
        if (stop_flag) {
            qemu_mutex_unlock(&hv_lock);
            break;
        }
        target = hv_asked;
        qemu_mutex_unlock(&hv_lock);
        reims_vgpu_dirty_background(D);
        qemu_mutex_lock(&hv_lock);
        hv_done = target;
        qemu_mutex_unlock(&hv_lock);
    }
    rcu_unregister_thread();
    return NULL;
}

/* The drain: one packet at a time, the way the device reads a set. */
static void read_set(Set *s, const uint64_t *snap, uint64_t token_bits)
{
    uint64_t g = reims_vgpu_dirty_gen(D, s->token | token_bits);
    size_t i;
    bool all = false;

    n_reads++;
    if (g == 0) {
        all = true;                                    /* "cannot tell": gather everything */
    } else if (g != s->gen_prev) {
        uint64_t *pg = g_new(uint64_t, s->count + 1);
        int64_t n = s->gen_prev ? reims_vgpu_dirty_written_since(D, s->token, s->gen_prev, pg, s->count + 1) : -1;

        if (n < 0) {
            all = true;
        } else {
            for (i = 0; i < (size_t)n; i++) {
                int64_t ix = s->pos[pg[i] / PAGE];

                if (ix >= 0) {
                    s->content[ix] = qatomic_read(&wseq[pg[i] / PAGE]);
                }
            }
            n_regathers++;
        }
        g_free(pg);
        s->gen_prev = g;
    } else {
        s->gen_prev = g;
    }
    if (all) {
        for (i = 0; i < s->count; i++) {
            s->content[i] = qatomic_read(&wseq[s->gpas[i] / PAGE]);
        }
        if (g) {
            s->gen_prev = g;
        }
        n_regathers++;
    }
    for (i = 0; i < s->count; i++) {
        if (s->content[i] < snap[s->gpas[i] / PAGE]) {
            if (qatomic_fetch_inc(&n_violations) < 5) {
                fprintf(stderr, "VIOLATION set token=%" PRIu64 " page=%" PRIu64 " content=%" PRIu64
                        " snap=%" PRIu64 " gen=%" PRIu64 " prev=%" PRIu64 "\n",
                        s->token, s->gpas[i] / PAGE, s->content[i], snap[s->gpas[i] / PAGE], g, s->gen_prev);
            }
            /* count once per page; carry on with the content brought up to date so one bug is one line */
            s->content[i] = qatomic_read(&wseq[s->gpas[i] / PAGE]);
        }
    }
}

static void *drain_thread(void *arg)
{
    (void)arg;
    rcu_register_thread();
    for (;;) {
        Packet *p;
        int i;

        qemu_mutex_lock(&pk_lock);
        if (!pk_head && !stop_flag) {
            int c;

            /* The drain visits every channel's ring and finds it empty. */
            qemu_mutex_unlock(&pk_lock);
            for (c = 1; c <= 4; c++) {
                reims_vgpu_dirty_note_work_scope(D, 0x20000000u | c);
            }
            qemu_mutex_lock(&pk_lock);
        }
        while (!pk_head && !stop_flag) {
            qemu_cond_wait(&pk_cond, &pk_lock);
        }
        if (!pk_head) {
            qemu_mutex_unlock(&pk_lock);
            break;
        }
        p = pk_head;
        pk_head = p->next;
        if (!pk_head) {
            pk_tail = NULL;
        }
        {
            /* The tail as the drain reads it from memory: at least this packet, maybe more behind it. */
            uint32_t seen_tail = (cfg_ring && p->channel > 0 && p->channel < 8)
                                 ? ring_tail_pub[p->channel] : 0;

            qemu_mutex_unlock(&pk_lock);
            if (cfg_ring) {
                reims_vgpu_dirty_note_ring_tail(D, (uint32_t)p->channel, seen_tail);
            }
        }

        reims_vgpu_dirty_note_work_scope(D, 0xffffffffu);
        reims_vgpu_dirty_note_work_scope(D, 0x80000000u | (uint32_t)(p->channel < 0 ? 0 : p->channel));
        qemu_mutex_lock(&set_lock);
        for (i = 0; i < nsets; i++) {
            if (p->sets_mask & (1u << i)) {
                read_set(&sets[i], p->snap, cfg_drain_lagged ? (1ULL << 63) : 0);
            }
        }
        qemu_mutex_unlock(&set_lock);
        reims_vgpu_dirty_note_work_scope(D, 0x40000000u | (uint32_t)(p->channel < 0 ? 0 : p->channel));
        n_packets++;
        g_free(p->snap);
        g_free(p);
    }
    rcu_unregister_thread();
    return NULL;
}

/* Extra readers: concurrent generation reads from other threads (a worker pool), checked only for
 * a generation that never shrinks per token. */
static void *reader_thread(void *arg)
{
    uint32_t st = 99 + (uint32_t)(uintptr_t)arg;
    uint64_t *last = g_new0(uint64_t, 1 << 16);
    uint64_t calls = 0;

    rcu_register_thread();
    while (!stop_flag) {
        uint64_t tok;
        uint64_t g;

        if (cfg_reader_pure) {
            tok = fixed_tokens[rnd(&st) % nsets];     /* no harness lock: measures the tracker's own */
        } else {
            qemu_mutex_lock(&set_lock);
            tok = sets[rnd(&st) % nsets].token;
            qemu_mutex_unlock(&set_lock);
        }
        g = reims_vgpu_dirty_gen(D, tok);
        calls++;
        if (tok < (1 << 16) && g != 0) {
            if (g < last[tok]) {
                fprintf(stderr, "READER: generation of token %" PRIu64 " went %" PRIu64 " -> %" PRIu64 "\n",
                        tok, last[tok], g);
                qatomic_inc(&n_violations);
            }
            last[tok] = g;
        }
        if (cfg_reader_spin) {
            spin_us(rnd(&st) % cfg_reader_spin);
        }
    }
    qatomic_add(&n_reader_calls, calls);
    rcu_unregister_thread();
    return NULL;
}

static void *churn_thread(void *arg)
{
    uint32_t st = 31337;
    (void)arg;

    rcu_register_thread();
    while (!stop_flag) {
        int i = rnd(&st) % nsets;

        spin_us(2000 + rnd(&st) % 3000);
        qemu_mutex_lock(&set_lock);
        set_drop(&sets[i]);
        set_build(&sets[i], &st, rnd(&st) % 3);
        qemu_mutex_unlock(&set_lock);
    }
    rcu_unregister_thread();
    return NULL;
}

int main(int argc, char **argv)
{
    int seconds = argc > 1 ? atoi(argv[1]) : 10;
    uint32_t st = 1;
    QemuThread th_guest, th_racer, th_harvest, th_drain, th_churn, th_nondb, th_readers[8], th_main;
    int i;
    const char *e;

    rcu_register_thread();
    qemu_mutex_init(&glock);
    qemu_mutex_init(&set_lock);
    qemu_mutex_init(&pk_lock);
    qemu_cond_init(&pk_cond);
    qemu_mutex_init(&hv_lock);
    qemu_cond_init(&hv_cond);
    fake_mr.ram = true;
    address_space_memory.current_map = (FlatView *)(uintptr_t)8;
    {
        DirtyMemoryBlocks *b = g_malloc0(sizeof(DirtyMemoryBlocks) + sizeof(unsigned long *));

        b->blocks[0] = g_new0(unsigned long, DIRTY_MEMORY_BLOCK_SIZE / BITS_PER_LONG);
        ram_list.dirty_memory[DIRTY_MEMORY_VGA] = b;
    }
    cfg_drain_lagged = getenv("STRESS_DRAIN_LAGGED") && atoi(getenv("STRESS_DRAIN_LAGGED"));
    cfg_threads_readers = (e = getenv("STRESS_READERS")) ? atoi(e) : 3;
    cfg_churn = (e = getenv("STRESS_CHURN")) ? atoi(e) : 1;
    cfg_ring_gap_us = (e = getenv("STRESS_GAP_US")) ? atoi(e) : 150;
    g_hv_delay_us = (e = getenv("STRESS_HV_US")) ? atoi(e) : 30;
    cfg_named_only = (e = getenv("STRESS_NAMED_ONLY")) ? atoi(e) : 0;
    cfg_nondb_hz = (e = getenv("STRESS_NONDB_HZ")) ? atoi(e) : 0;
    cfg_hole_pct = (e = getenv("STRESS_HOLE")) ? atoi(e) : 0;
    cfg_ring = (e = getenv("STRESS_RING")) ? atoi(e) : 0;
    cfg_ring_skip = (e = getenv("STRESS_RING_SKIP")) ? atoi(e) : 0;
    cfg_reader_spin = (e = getenv("STRESS_READER_SPIN_US")) ? atoi(e) : 30;
    cfg_reader_pure = (e = getenv("STRESS_READER_PURE")) ? atoi(e) : 0;
    cfg_hole_us = (e = getenv("STRESS_HOLE_US")) ? atoi(e) : 40;
    nsets = (e = getenv("STRESS_SETS")) ? MIN(atoi(e), MAXSETS) : 16;
    cfg_wp_nopause = (e = getenv("STRESS_WP_NOPAUSE")) ? atoi(e) : 0;
    qemu_mutex_init(&vg_lock);
    qemu_cond_init(&vg_cond);

    for (i = 0; i < NHOT; i++) {
        hot_pages[i] = rnd(&st) % UNIVERSE;
    }
    D = reims_vgpu_dirty_new();
    reims_vgpu_dirty_set_ondemand(D, true);
    for (i = 0; i < nsets; i++) {
        set_build(&sets[i], &st, i % 3);
    }

    for (i = 0; i < nsets; i++) {
        fixed_tokens[i] = sets[i].token;
    }
    qemu_thread_create(&th_harvest, "harvest", harvest_thread, NULL, QEMU_THREAD_JOINABLE);
    qemu_thread_create(&th_main, "mainloop", mainloop_thread, NULL, QEMU_THREAD_JOINABLE);
    qemu_thread_create(&th_drain, "drain", drain_thread, NULL, QEMU_THREAD_JOINABLE);
    qemu_thread_create(&th_guest, "guest", guest_thread, NULL, QEMU_THREAD_JOINABLE);
    qemu_thread_create(&th_racer, "racer", racer_thread, NULL, QEMU_THREAD_JOINABLE);
    if (cfg_churn) {
        qemu_thread_create(&th_churn, "churn", churn_thread, NULL, QEMU_THREAD_JOINABLE);
    }
    if (cfg_nondb_hz) {
        qemu_thread_create(&th_nondb, "nondb", nondb_thread, NULL, QEMU_THREAD_JOINABLE);
    }
    for (i = 0; i < cfg_threads_readers && i < 8; i++) {
        qemu_thread_create(&th_readers[i], "reader", reader_thread, (void *)(uintptr_t)i, QEMU_THREAD_JOINABLE);
    }
    {
        int64_t t_end = get_clock() + (int64_t)seconds * 1000000000LL;

        while (get_clock() < t_end) {
            g_usleep(100000);
        }
    }
    stop_flag = 1;
    qemu_mutex_lock(&hv_lock);
    qemu_cond_signal(&hv_cond);
    qemu_mutex_unlock(&hv_lock);
    qemu_mutex_lock(&pk_lock);
    qemu_cond_broadcast(&pk_cond);
    qemu_mutex_unlock(&pk_lock);
    qemu_thread_join(&th_guest);
    qemu_thread_join(&th_racer);
    if (cfg_nondb_hz) {
        qemu_thread_join(&th_nondb);
    }
    if (cfg_churn) {
        qemu_thread_join(&th_churn);
    }
    for (i = 0; i < cfg_threads_readers && i < 8; i++) {
        qemu_thread_join(&th_readers[i]);
    }
    qemu_thread_join(&th_drain);
    qemu_thread_join(&th_harvest);
    qemu_thread_join(&th_main);
    printf("RESULT packets=%" PRIu64 " reads=%" PRIu64 " regathers=%" PRIu64 " violations=%" PRIu64
           " reader_calls=%" PRIu64 " wp_faults=%" PRIu64 " wp_remaps=%" PRIu64 "\n",
           n_packets, n_reads, n_regathers, n_violations, n_reader_calls, n_wp_faults, n_wp_remaps);
    fflush(stdout);
    return n_violations ? 1 : 0;
}
