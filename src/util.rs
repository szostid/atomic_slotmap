//! Utilities for manipulating the [`slotmap::KeyData`] which is `pub(in slotmap)`.
use core::num::NonZeroU32;
use slotmap::KeyData;

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
