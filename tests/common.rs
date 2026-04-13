#![allow(unused)]

pub use atomic_slotmap::*;
pub use slotmap::*;

#[cfg(loom)]
pub use loom::sync::Arc;
#[cfg(loom)]
pub use loom::thread;
#[cfg(loom)]
pub fn model<F: Fn() + 'static + Send + Sync>(bound: Option<usize>, f: F) {
    let mut builder = loom::model::Builder::new();
    builder.preemption_bound = bound;
    builder.check(f);
}

use std::ops::{Deref, DerefMut};
#[cfg(not(loom))]
pub use std::sync::Arc;
#[cfg(not(loom))]
pub use std::thread;

#[cfg(not(loom))]
pub fn model<F: Fn() + 'static + Send + Sync>(_bound: Option<usize>, mut f: F) {
    f();
}

/// Tracks the amount of drops that happen on this value
#[derive(Clone, Default)]
pub struct DropCounter<T = ()> {
    val: T,
    // no loom, this isnt a part of the test suite
    counter: Arc<core::sync::atomic::AtomicUsize>,
}

impl<T> DropCounter<T> {
    pub fn get(&self) -> usize {
        self.counter.load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl<T> Drop for DropCounter<T> {
    fn drop(&mut self) {
        self.counter
            .fetch_add(1, core::sync::atomic::Ordering::Relaxed);
    }
}

impl<T> Deref for DropCounter<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.val
    }
}

impl<T> DerefMut for DropCounter<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.val
    }
}
