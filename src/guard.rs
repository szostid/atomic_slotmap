use crate::{util::KeyDataRead as _, AtomicSlotMap, Slot};
use alloc::fmt;
use core::ops::Deref;
use core::sync::atomic::Ordering;
use slotmap::Key;

/// A guard into a slot of a [`AtomicSlotMap`].
pub struct SlotGuard<'a, K: Key, V> {
    /// We cannot keep &V or &Slot<V> to satisfy the borrowing rules
    /// when dropping the slot - calling drop_in_place on the inner
    /// value of `Slot<V>` is unsafe if we have a reference to either
    /// `&V` or `&Slot<V>`, so that would trigger MIRI. We can, however,
    /// keep a raw pointer to *mut V.
    value: *const V,
    key: K,
    map: &'a AtomicSlotMap<K, V>,
}

impl<'a, K: Key, V> SlotGuard<'a, K, V> {
    /// Constructs a new slotguard.
    ///
    /// The key needs to have an odd version index (occupied slot)
    pub(crate) fn new(key: K, map: &'a AtomicSlotMap<K, V>) -> Option<Self> {
        let kd = key.data();
        let slot = map.slots.get(kd.idx())?;

        let v_start = slot.version.load(Ordering::Acquire);

        if v_start != kd.version().get() {
            return None;
        }

        // we cannot fetch_add, because once the reference count drops to zero,
        // it **must** stay there - otherwise, when dropping, two locks could see
        // a reference count of 1 thinking that they are supposed to drop the value.
        // we need to check if the reference count was possibly zero, compute the
        // new value, try to exchange it, and if it did changed, then we have to try
        // again. otherwise we're fine
        let mut count = slot.ref_count.load(Ordering::Relaxed);
        loop {
            if count == 0 {
                return None;
            }

            match slot.ref_count.compare_exchange_weak(
                count,
                count + 1,
                Ordering::Acquire,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(c) => count = c,
            }
        }

        // we need to validate that the slot version is still what we expect it
        // to be. while it wouldn't be that big of a deal if the slot got marked
        // for removal in the meanwhile, it would be a HUGE deal if after the version
        // check but before the reference count check the slot was marked for removal,
        // deallocated and reused (which is extremely unlikely, but whatever). the
        // reference count would never be read as zero, but the slot would
        // contain a totally different version then expected. if that happens, then
        // we need to drop the value that was inside of the slot.
        if slot.version.load(Ordering::Acquire) != kd.version().get() {
            let ref_count = slot.ref_count.fetch_sub(1, Ordering::AcqRel);

            // we're the last user of this slot value that we don't care about anyways
            if ref_count == 1 {
                unsafe {
                    slot.drop_inner_value();
                    map.push_free_index(kd.idx());
                }
            }

            return None;
        }

        // SAFETY: we know the reference count is non-zero and we know that
        // the slot is occupied, so its safe to read the value of the slot
        let value = unsafe { (*slot.data_ptr()).as_ptr() };

        Some(Self { key, value, map })
    }

    /// Returns the key of the value that this slot guard guards.
    #[inline]
    #[must_use]
    pub fn key(&self) -> K {
        self.key
    }

    /// Returns the reference to the slot that this [`SlotGuard`] points to.
    fn slot(&self) -> &Slot<V> {
        // SAFETY: an AtomicVec (self.map.slots) cannot pop elements so once a
        // valid index into it exists, it will keep on existing. additionally, we
        // know that this `SlotGuard` indeed exists, so there's even a guarantee
        // that tells us that the slot won't be modified while this guard exists
        unsafe { self.map.slots.get_unchecked(self.key.data().idx()) }
    }

    /// Returns the value that the slot points to.
    ///
    /// # Safety
    /// When calling this method, you have to ensure that the returned reference
    /// will not overlap with any mutable usages. This means that the refcount
    /// of the `SlotGuard` must be non-zero in order for this to exists. This will
    /// be true everywhere except in `drop`
    unsafe fn value(&self) -> &V {
        // SAFETY: the value is safe to read as long as the slotguard exists and is not being dropped
        unsafe { &*self.value }
    }
}

impl<K: Key, V> fmt::Display for SlotGuard<'_, K, V>
where
    V: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // SAFETY: self exists so refcount != 0
        let value = unsafe { self.value() };
        V::fmt(value, f)
    }
}

impl<K: Key, V> fmt::Debug for SlotGuard<'_, K, V>
where
    V: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // SAFETY: self exists so refcount != 0
        let value = unsafe { self.value() };
        V::fmt(value, f)
    }
}

impl<K: Key, V> Deref for SlotGuard<'_, K, V> {
    type Target = V;

    fn deref(&self) -> &Self::Target {
        // SAFETY: self exists so refcount != 0
        unsafe { self.value() }
    }
}

impl<K: Key, V> AsRef<V> for SlotGuard<'_, K, V> {
    fn as_ref(&self) -> &V {
        // SAFETY: self exists so refcount != 0
        unsafe { self.value() }
    }
}

impl<K: Key, V> Drop for SlotGuard<'_, K, V> {
    fn drop(&mut self) {
        let ref_count = self.slot().ref_count.fetch_sub(1, Ordering::AcqRel);

        // we're the last user, we have to drop the value
        if ref_count == 1 {
            // SAFETY: we know the refcount == 0, so nothing refers and
            // tries to read the value of the slotguard at this point
            unsafe {
                self.slot().drop_inner_value();
            }

            // SAFETY: we know this is a valid index into the slotmap,
            // because it was valid when we created the SlotGuard.
            // we're the last referant to this slot because ref_count == 1.
            unsafe {
                self.map.push_free_index(self.key.data().idx());
            }
        }
    }
}

// SAFETY: SlotGuard behaves exactly like Arc<T>. It's send whenever the elements
// within are Send + Sync. It doesn't derive the trait automatically only because
// of the `*const V` that's within, but that doesn't change anything
unsafe impl<K: Send + Sync + Key, V: Send + Sync> Send for SlotGuard<'_, K, V> {}

// SAFETY: SlotGuard behaves exactly like Arc<T>. It's sync whenever the elements
// within are Send + Sync. It doesn't derive the trait automatically only because
// of the `*const V` that's within, but that doesn't change anything
unsafe impl<K: Send + Sync + Key, V: Send + Sync> Sync for SlotGuard<'_, K, V> {}
