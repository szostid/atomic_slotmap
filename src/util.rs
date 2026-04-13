//! Utilities for manipulating the [`slotmap::KeyData`] which is `pub(in slotmap)`.
use core::num::NonZeroU32;
use slotmap::KeyData;

#[cfg(loom)]
use loom::sync::atomic::Ordering;

/// Extension trait for reading [`slotmap::KeyData`]
pub trait KeyDataRead {
    #[must_use]
    fn version(&self) -> NonZeroU32;
    #[must_use]
    fn idx(&self) -> u32;
    #[must_use]
    fn new(idx: u32, version: u32) -> Self;
}

impl KeyDataRead for KeyData {
    #[inline]
    fn idx(&self) -> u32 {
        let value = self.as_ffi();
        let idx = value & 0xffff_ffff;
        idx as u32
    }

    #[inline]
    fn version(&self) -> NonZeroU32 {
        let value = self.as_ffi();
        let version = (value >> 32) | 1; // Ensure version is odd.

        // SAFETY: we or'ed the version with 1 so it'll never be zero
        unsafe { NonZeroU32::new_unchecked(version as u32) }
    }

    #[inline]
    fn new(idx: u32, version: u32) -> Self {
        let ffi = (u64::from(version) << 32) | u64::from(idx);
        Self::from_ffi(ffi)
    }
}

pub trait AtomicGetExclusive {
    type Output;

    /// Performs a `get_mut` + `Deref` (but works with loom
    /// too, where the method is not available and an atomic
    /// load has to be performed instead)
    fn get(&mut self) -> Self::Output;
}

#[cfg(not(loom))]
impl AtomicGetExclusive for core::sync::atomic::AtomicU32 {
    type Output = u32;

    fn get(&mut self) -> u32 {
        *self.get_mut()
    }
}

#[cfg(loom)]
impl AtomicGetExclusive for loom::sync::atomic::AtomicU32 {
    type Output = u32;

    fn get(&mut self) -> u32 {
        self.load(Ordering::SeqCst)
    }
}

#[cfg(not(loom))]
impl<T> AtomicGetExclusive for core::sync::atomic::AtomicPtr<T> {
    type Output = *mut T;

    fn get(&mut self) -> *mut T {
        *self.get_mut()
    }
}

#[cfg(loom)]
impl<T> AtomicGetExclusive for loom::sync::atomic::AtomicPtr<T> {
    type Output = *mut T;

    fn get(&mut self) -> *mut T {
        self.load(Ordering::SeqCst)
    }
}
