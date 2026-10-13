//! Recent owner events, retained in bounded memory and dumped on a pipeline failure.
//! This journal observes existing identities; it does not assign lifetimes.
use reims_vgpu_core::identity::{IngressOrdinal, ResourceId};
use std::collections::VecDeque;
#[derive(Clone, Copy, Debug)]
pub(crate) enum Phase {
    Define,
    DeleteTask,
    BindList,
    Create,
    DeleteObject,
    RetainPipeline,
    Admit,
    Execute,
    Missing,
}
#[derive(Clone, Copy, Debug)]
pub(crate) struct Event {
    pub phase: Phase,
    pub task: u32,
    pub incarnation: u64,
    pub directory: u32,
    pub list: u32,
    pub list_count: u32,
    pub resource: Option<ResourceId>,
    pub ordinal: Option<IngressOrdinal>,
    pub host_object: Option<u64>,
}
#[derive(Debug, Default)]
pub(crate) struct Trace {
    events: VecDeque<(u64, Event)>,
    next: u64,
    dumps: u32,
}
impl Trace {
    pub(crate) fn record(&mut self, event: Event) {
        if self.events.len() == 256 {
            self.events.pop_front();
        }
        self.next = self.next.saturating_add(1);
        self.events.push_back((self.next, event));
    }
    pub(crate) fn dump(&mut self, task: u32) {
        if self.dumps >= 8 {
            return;
        }
        self.dumps += 1;
        crate::observe::fail(format!(
            "DEV_LIFETIME_BEGIN task={task} dump={} retained={}",
            self.dumps,
            self.events.len()
        ));
        for (sequence, e) in &self.events {
            if e.task == task {
                crate::observe::fail(format!("DEV_LIFETIME seq={sequence} phase={:?} task={} incarnation={} dir={:#x} list={:#x} count={} resource={:?} ordinal={:?} host_object={:?}", e.phase, e.task, e.incarnation, e.directory, e.list, e.list_count, e.resource, e.ordinal, e.host_object));
            }
        }
        crate::observe::fail(format!("DEV_LIFETIME_END task={task}"));
    }
}
