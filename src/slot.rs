use core::cell::UnsafeCell;
use core::mem::MaybeUninit;
use core::sync::atomic::{AtomicU32, Ordering};

pub struct Slot<T> {
    /// The data contained within the slot.
    ///
    /// The contents of [`SlotData`] are generally unknown, but it
    /// can be assumed that is the slot is occupied (that is, the
    /// slot is either occupied or marked as unoccupied but still
    /// has a non-zero ref_count) then it can be assumed that the
    /// data contains [`SlotData::value`].
    ///
    /// Otherwise, the `data` will contain [`SlotData::next_free`]
    /// only if is is reachable through [`AtomicSlotMap::free_head`]
    pub(crate) data: UnsafeCell<MaybeUninit<T>>,
    pub(crate) next_free: AtomicU32,
    /// Reference count. Includes the slotmap and all SlotGuards.
    pub(crate) ref_count: AtomicU32,
    pub(crate) version: AtomicU32,
}

impl<T> Slot<T> {
    /// Returns the pointer to the data contained within the slot.
    ///
    /// The data is free to read if either:
    /// - the slot is referred to as one of the free nodes. Then the
    ///   slot is free to be read as `next_free`
    /// - the slot is occupied. This means that it contains a value.
    /// - the slot is not occupied, but has a non-zero refcount. This
    ///   means that there are [`SlotGuard`]s pointing to it
    pub(crate) fn data_ptr(&self) -> *mut MaybeUninit<T> {
        self.data.get()
    }

    /// Drops the inner value contained within the slot.
    ///
    /// # Safety
    /// The slot should be occupied. Nothing should refer to it, that
    /// is, the reference to it should be unique, and it should actually
    /// store the data inside (it cannot store `next_free` instead)
    #[inline]
    #[track_caller]
    pub(crate) unsafe fn drop_inner_value(&self) {
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

        // SAFETY: the safety clause requires exclusive access to the slot,
        // and it requires the value to be present.
        unsafe {
            let data = &mut *self.data_ptr();
            core::ptr::drop_in_place(data.as_mut_ptr());
        }
    }
}

impl<T> Drop for Slot<T> {
    fn drop(&mut self) {
        // we have exclusive ownership of the slot, we know the slotmap
        // is being dropped so there are no guards pointing to it
        if core::mem::needs_drop::<T>() && (*self.version.get_mut() % 2 == 1) {
            let data = self.data.get_mut();

            // SAFETY: the slot is occupied, we have a mutable reference to the slot
            // so we own the value and it exists. We can read the data as
            // value because we know the slot is occupied.
            unsafe {
                core::ptr::drop_in_place(data.as_mut_ptr());
            }
        }
    }
}
