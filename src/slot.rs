use core::cell::UnsafeCell;
use core::mem::MaybeUninit;
use core::sync::atomic::{fence, AtomicU32, Ordering};

/// Similar to [`std::sync::Arc`], the Slot has a maximum reference
/// count cap to prevent leaked references from overflowing the
/// reference count
const MAX_REFCOUNT: u32 = 0xFFFFFF00;

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

    /// Tried to acquire a guard for this slot, expecting it to have the version
    /// of `expected_version`.
    ///
    /// # Errors
    /// Returns an error if the guard failed to acquire because the slot didnt
    /// match the version. Returns `Err(true)` if the slot has to be dropped
    /// and marked for removal
    #[inline]
    pub(crate) fn acquire_guard(&self, expected_version: u32) -> Result<(), bool> {
        // fail-fast if the slot is outdated
        if self.version.load(Ordering::Acquire) != expected_version {
            return Err(false);
        }

        let mut refcount = self.ref_count.load(Ordering::Relaxed);

        // we cannot fetch_add, because once the reference count drops to zero,
        // it **must** stay there - otherwise, when dropping, two locks could see
        // a reference count of 1 thinking that they are supposed to drop the value.
        // we need to check if the reference count was possibly zero, compute the
        // new value, try to exchange it, and if it did changed, then we have to try
        // again. otherwise we're fine
        loop {
            if refcount > MAX_REFCOUNT {
                panic!("max refcount reached");
            }

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
                Err(c) => refcount = c,
            }
        }

        // we need to validate that the slot version is still what we expect it
        // to be. while it wouldn't be that big of a deal if the slot got marked
        // for removal in the meanwhile, it would be a HUGE deal if after the version
        // check but before the reference count check the slot was marked for removal,
        // deallocated and reused (which is extremely unlikely, but possible). the
        // reference count would never be read as zero, but the slot would
        // contain a totally different version then expected. if that happens, then
        // we need to drop the value that was inside of the slot.
        if self.version.load(Ordering::Acquire) == expected_version {
            return Ok(());
        }

        // we failed to acquire the lock because it got marked for removal, deallocated and
        // reused. we don't want this lock - we decrement the ref count, but its possible
        // that we're actually the last referrant now, and in that case we need to deallocate
        // the contents of the slot
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

        if old_count > MAX_REFCOUNT {
            panic!("max refcount reached");
        }
    }

    /// Decrements ref_count for dropping a reference to the slot.
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
    /// The slot should have no referents (that is, its
    /// referent count should be 0) BUT it shouldn't be
    /// placed within the free list of the slotmap which
    /// guarantees that the slot is owned exclusively
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
