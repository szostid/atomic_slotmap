use crate::atomic::{fence, AtomicU32, Ordering};
use crate::util::AtomicGetExclusive;
use crate::UnsafeCell;
use core::mem::MaybeUninit;

/// Similar to [`std::sync::Arc`], the Slot has a maximum reference
/// count cap to prevent leaked references from overflowing the
/// reference count
const MAX_REFCOUNT: u32 = 0xFF_FF_FF_00;

/// A single slot in the atomic slotmap.
///
/// # Ownership
/// The refererence count of the slot strictly determines the ownership.
///
/// If the slot has no references, it is safe to write, assuming that it cannot
/// be reached through other means (mainly the free list of the slot map. i.e.
/// if you pop a slot off the free list then it isn't there and it will have
/// a zeroed refcount so its safe to write).
pub struct Slot<T> {
    pub(crate) data: UnsafeCell<MaybeUninit<T>>,
    /// The index of the next free slot in the linked list of free
    /// slots. The value of this is unspecified if the slot is
    /// not a free slot (a slot in the free list)
    pub(crate) next_free: AtomicU32,
    /// The reference count. Includes the slotmap itself too (i.e.
    /// if the slot is not removed / is reachable by the slotmap
    /// then the reference count will never drop to zerop)
    pub(crate) ref_count: AtomicU32,
    /// If even, the slot is free and unoccupied.
    pub(crate) version: AtomicU32,
}

unsafe impl<T: Send> Send for Slot<T> {}
unsafe impl<T: Sync> Sync for Slot<T> {}

impl<T> Slot<T> {
    /// Returns the pointer to the data contained within the slot.
    ///
    /// The data is safe to read as init if the slot is occupied
    /// (i.e. odd version)
    ///
    /// This data is safe to write to if the slot is exclusively owned
    /// (the exact requirements are explained in the [`Slot`]s
    /// documentation).
    pub(crate) fn data_ptr(&self) -> *mut MaybeUninit<T> {
        #[cfg(not(loom))]
        return self.data.get();
        #[cfg(loom)]
        return self.data.with(|ptr| ptr as *mut _);
    }

    /// Tried to acquire a guard for this slot, expecting it to have the version
    /// of `expected_version`.
    ///
    /// # Errors
    /// Returns an error if the guard failed to acquire because the slot didnt
    /// match the version. Returns `Err(true)` if the slot has to be dropped
    /// and marked for removal
    #[inline]
    pub(crate) fn acquire_guard(&self, expected_version: u32) -> Result<(), bool> {
        // note: we could add a fail fast and check the expected_version right away,
        // but that would speed up the failure path and slow down the fast path
        // (valid handles would have to fetch the version unnecessarily), the version
        // is checked after the CAS loop anyways

        let mut refcount = self.ref_count.load(Ordering::Relaxed);

        // we cannot fetch_add, because once the reference count drops to zero,
        // it MUST stay there - otherwise, when dropping, two locks could see
        // a reference count of 1 thinking that they are supposed to drop the value.
        // we need to check if the reference count was possibly zero, compute the
        // new value, try to exchange it, and if it did changed, then we have to try
        // again. otherwise we're fine
        loop {
            assert!(refcount <= MAX_REFCOUNT, "max refcount reached");

            if refcount == 0 {
                return Err(false);
            }

            match self.ref_count.compare_exchange_weak(
                refcount,
                refcount + 1,
                Ordering::Acquire,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(cur_refcount) => refcount = cur_refcount,
            }
        }

        if self.version.load(Ordering::Acquire) == expected_version {
            return Ok(());
        }

        // either the slot had an outdated version from the get-go or it got marked for removal
        // dropped reallocated and reused. either way, we've gotten a lock onto the slot but its
        // not what we want it to be, so we drop the reference
        let needs_drop = self.dec_ref_count();

        Err(needs_drop)
    }

    /// Increments the reference count for a clone operation.
    ///
    /// This assumes that the slot already has at least one referent
    #[inline]
    pub(crate) fn inc_ref_count_for_clone(&self) {
        // this is more or less equivalent to the clone impl for Arc:
        // https://doc.rust-lang.org/src/alloc/sync.rs.html#2376
        let old_count = self.ref_count.fetch_add(1, Ordering::Relaxed);

        assert!(old_count <= MAX_REFCOUNT, "max refcount reached");
    }

    /// Decrements `ref_count` for dropping a reference to the slot.
    ///
    /// Returns whether the slot has no existing references and therefore
    /// should be dropped. When this function returns `true` it is guaranteed
    /// to be safe to deallocate the slot and mark it as free.
    #[inline]
    pub(crate) fn dec_ref_count(&self) -> bool {
        // this is more or less equivalent to the drop impl for Arc:
        // https://doc.rust-lang.org/src/alloc/sync.rs.html#2807

        if self.ref_count.fetch_sub(1, Ordering::Release) != 1 {
            return false;
        }

        fence(Ordering::Acquire);

        true
    }

    /// Drops the inner value contained within the slot.
    ///
    /// # Safety
    /// The slot should be exclusively owned (the exact requirements
    /// are explained in the [`Slot`]s documentation), and there must
    /// be a valid value in the slot (i.e. not dropped yet)
    #[inline]
    #[track_caller]
    pub(crate) unsafe fn drop_inner_value(&self) {
        // SAFETY: the safety clause requires exclusive access to the slot,
        // and it requires the value to be present.
        unsafe {
            let data = &mut *self.data_ptr();
            core::ptr::drop_in_place(data.as_mut_ptr());
        }
    }

    #[inline]
    #[track_caller]
    pub(crate) fn debug_assert_exclusively_owned(&self) {
        debug_assert_eq!(
            self.ref_count.load(Ordering::Acquire),
            0,
            "Slot::drop_inner_value called on a referred-to slot"
        );

        debug_assert_eq!(
            self.version.load(Ordering::Acquire) % 2,
            0,
            "Slot::drop_inner_value called on an occupied slot"
        );
    }
}

/// If running on loom, AtomicU32 and UnsafeCell are not
/// primitives that can be zeroed anymore, and we need a
/// proper default impl to use with DefaultIfLoom on the
/// AtomicVec
#[cfg(loom)]
impl<T> Default for Slot<T> {
    fn default() -> Self {
        Self {
            data: UnsafeCell::new(MaybeUninit::uninit()),
            next_free: AtomicU32::new(0),
            ref_count: AtomicU32::new(0),
            version: AtomicU32::new(0),
        }
    }
}

impl<T> crate::vec::ZeroedOrDefault for Slot<T> {}

impl<T> Drop for Slot<T> {
    fn drop(&mut self) {
        // we have exclusive ownership of the slot, we know the slotmap
        // is being dropped so there are no guards pointing to it
        if core::mem::needs_drop::<T>() && (self.version.get() % 2 == 1) {
            // SAFETY: we have a mutable reference to the slot, so we have
            // a guarantee of exclusive ownership, and the version is odd so
            // there's a value contained within the slot
            unsafe { self.drop_inner_value() };
        }
    }
}
