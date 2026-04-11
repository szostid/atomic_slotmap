use crate::{util::KeyDataRead as _, AtomicSlotMap, Slot};
use alloc::fmt;
use alloc::sync::Arc;
use core::ops::Deref;
use slotmap::Key;

/// A guard into a slot of a [`AtomicSlotMap`] which has an `Arc` into it
/// (meaning that it isn't tied to the slotmap by lifetime)
pub struct OwningSlotGuard<K: Key, V> {
    /// We cannot keep &V or &Slot<V> to satisfy the borrowing rules
    /// when dropping the slot - calling drop_in_place on the inner
    /// value of `Slot<V>` is unsafe if we have a reference to either
    /// `&V` or `&Slot<V>`, so that would trigger MIRI. We can, however,
    /// keep a raw pointer to *mut V.
    value: *const V,
    key: K,
    map: Arc<AtomicSlotMap<K, V>>,
}

impl<K: Key, V> OwningSlotGuard<K, V> {
    /// Constructs a new slotguard.
    ///
    /// The key needs to have an odd version index (occupied slot)
    pub(crate) fn new(key: K, map: Arc<AtomicSlotMap<K, V>>) -> Option<Self> {
        let kd = key.data();
        let slot = map.slots.get(kd.idx())?;

        if let Err(needs_drop) = slot.acquire_guard(kd.version().get()) {
            if needs_drop {
                // SAFETY: dec_ref_count assures this is safe, because the
                // value needs drop.
                unsafe {
                    slot.drop_inner_value();
                }

                // SAFETY: dec_ref_count assures this is safe, because the
                // value needs drop. we know this is a valid index into the
                // slotmap, because it was valid when we created the SlotGuard.
                unsafe {
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
    ///
    /// # Safety
    /// The slotguard must exist and must not be actively being dropped. This
    /// ensures that the slot is kept alive by this guard.
    fn slot(&self) -> &Slot<V> {
        // SAFETY: the value is safe to read as long as the slotguard exists and is not being dropped
        unsafe { self.map.slots.get_unchecked(self.key.data().idx()) }
    }

    /// Returns the value that the slot points to.
    ///
    /// # Safety
    /// The slotguard must exist and must not be actively being dropped. This
    /// ensures that the slot is kept alive by this guard.
    unsafe fn value(&self) -> &V {
        // SAFETY: the value is safe to read as long as the slotguard exists and is not being dropped
        unsafe { &*self.value }
    }
}

impl<K: Key, V> fmt::Display for OwningSlotGuard<K, V>
where
    V: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // SAFETY: self exists so refcount != 0
        let value = unsafe { self.value() };
        V::fmt(value, f)
    }
}

impl<K: Key, V> fmt::Debug for OwningSlotGuard<K, V>
where
    V: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // SAFETY: self exists so refcount != 0
        let value = unsafe { self.value() };
        V::fmt(value, f)
    }
}

impl<K: Key, V> Deref for OwningSlotGuard<K, V> {
    type Target = V;

    fn deref(&self) -> &Self::Target {
        // SAFETY: self exists so refcount != 0
        unsafe { self.value() }
    }
}

impl<K: Key, V> AsRef<V> for OwningSlotGuard<K, V> {
    fn as_ref(&self) -> &V {
        // SAFETY: self exists so refcount != 0
        unsafe { self.value() }
    }
}

impl<K: Key, V> Drop for OwningSlotGuard<K, V> {
    fn drop(&mut self) {
        let needs_drop = self.slot().dec_ref_count();

        // we're the last user, we have to drop the value
        if needs_drop {
            // SAFETY: dec_ref_count assures this is safe, because the
            // value needs drop.
            unsafe {
                self.slot().drop_inner_value();
            }

            // SAFETY: dec_ref_count assures this is safe, because the
            // value needs drop. we know this is a valid index into the
            // slotmap, because it was valid when we created the SlotGuard.
            unsafe {
                self.map.push_free_index(self.key.data().idx());
            }
        }
    }
}

impl<K: Key, V> Clone for OwningSlotGuard<K, V> {
    fn clone(&self) -> Self {
        self.slot().inc_ref_count_for_clone();

        Self {
            value: self.value,
            key: self.key,
            map: self.map.clone(),
        }
    }
}

// SAFETY: OwningSlotGuard behaves exactly like Arc<T>. It's send whenever
// the elements within are Send + Sync. It doesn't derive the trait automatically
// only because of the `*const V` that's within, but that doesn't change anything
unsafe impl<K: Send + Sync + Key, V: Send + Sync> Send for OwningSlotGuard<K, V> {}

// SAFETY: OwningSlotGuard behaves exactly like Arc<T>. It's sync whenever
// the elements within are Send + Sync. It doesn't derive the trait automatically
// only because of the `*const V` that's within, but that doesn't change anything
unsafe impl<K: Send + Sync + Key, V: Send + Sync> Sync for OwningSlotGuard<K, V> {}
