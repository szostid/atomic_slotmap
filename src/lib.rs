//! # atomic_slotmap
//!
//! This library provies an extension to the [slotmap] crate, adding an atomic
//! slotmap which can be modified without a mutable reference.
//!
//! Unlike the slotmaps from [slotmap], this slotmap is impossible to iterate,
//! or generally to modify in bulk because of TOCTOU issues.
#![crate_name = "atomic_slotmap"]
#![cfg_attr(not(test), no_std)]
#![warn(
    missing_debug_implementations,
    trivial_casts,
    trivial_numeric_casts,
    unused_lifetimes,
    unused_import_braces
)]
#![deny(missing_docs)]
#![deny(clippy::all)]
#![allow(
    clippy::while_let_on_iterator, // Style differences.
    clippy::unnecessary_map_or // Too high MSRV.
)]

extern crate alloc;

use alloc::sync::Arc;
use core::cell::UnsafeCell;
use core::convert::Infallible;
use core::fmt;
use core::marker::PhantomData;
use core::mem::MaybeUninit;
use core::ops::Deref;
use core::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use slotmap::{DefaultKey, Key, KeyData};

mod util;
use util::KeyDataRead as _;

mod vec;
use vec::AtomicVec;

mod guard;
pub use guard::SlotGuard;

mod owning_guard;
pub use owning_guard::OwningSlotGuard;

mod slot;
use slot::Slot;

const SENTINEL: u32 = u32::MAX;

/// Creates a packed `free_head` from a tag and index
#[inline]
#[must_use]
fn pack_free_head(tag: u32, index: u32) -> u64 {
    ((tag as u64) << 32) | (index as u64)
}

/// Unpacks a packed `free_head` element into its tag and index
#[inline]
#[must_use]
fn unpack_free_head(packed: u64) -> (u32, u32) {
    ((packed >> 32) as u32, packed as u32)
}

/// Atomic slot map, lockfree storage with stable unique keys.
///
/// See [crate documentation](crate) for more details.
///
/// It is impossible to bulk view/modify the elements of an
/// atomic slotmae because of the nature of lockfree structures,
/// which are subject to TOCTOU. This means that it is impossible
/// to [`Debug`], [`Iterate`](`Iterator`), [`Clone`] or even clear
/// the slotmap.
#[allow(missing_debug_implementations)]
pub struct AtomicSlotMap<K: Key, V> {
    slots: AtomicVec<Slot<V>>,
    /// `free_head` is packed using [`pack`] and contains
    /// an operation index and free head index. Operation
    /// index is incremented during any operation which
    /// modified `free_head`, and it is used to prevent ABA
    free_head: AtomicU64,
    num_elems: AtomicU32,
    _k: PhantomData<fn(K) -> K>,
}

impl<V> AtomicSlotMap<DefaultKey, V> {
    /// Constructs a new, empty [`SlotMap`].
    ///
    /// # Examples
    ///
    /// ```
    /// # use atomic_slotmap::*;
    /// let mut sm: AtomicSlotMap<_, i32> = AtomicSlotMap::new();
    /// ```
    pub fn new() -> Self {
        Self::with_capacity_and_key(0)
    }

    /// Creates an empty [`AtomicSlotMap`] with the given capacity.
    ///
    /// The slot map will never allocate if it contains less keys than `capacity`.
    ///
    /// The capacity of the hashmap is not guaranteed to be exactly `capacity`. For
    /// more information, look at [`AtomicSlotMap::capacity`]
    ///
    /// # Examples
    ///
    /// ```
    /// # use atomic_slotmap::*;
    /// let mut sm: AtomicSlotMap<_, i32> = AtomicSlotMap::with_capacity(10);
    /// ```
    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_and_key(capacity)
    }
}

impl<K: Key, V> AtomicSlotMap<K, V> {
    /// Constructs a new, empty [`AtomicSlotMap`] with a custom key type.
    ///
    /// # Examples
    ///
    /// ```
    /// # use atomic_slotmap::*;
    /// # use slotmap::*;
    ///
    /// new_key_type! {
    ///     struct PositionKey;
    /// }
    /// let mut positions: AtomicSlotMap<PositionKey, i32> = AtomicSlotMap::with_key();
    /// ```
    pub fn with_key() -> Self {
        Self::with_capacity_and_key(0)
    }

    /// Creates an empty [`AtomicSlotMap`] with the given capacity and a custom key
    /// type.
    ///
    /// The slot map will never allocate if it contains less keys than `capacity`.
    ///
    /// The capacity of the hashmap is not guaranteed to be exactly `capacity`. For
    /// more information, look at [`AtomicSlotMap::capacity`]
    ///
    /// # Examples
    ///
    /// ```
    /// # use atomic_slotmap::*;
    /// # use slotmap::*;
    /// new_key_type! {
    ///     struct MessageKey;
    /// }
    /// let mut messages = AtomicSlotMap::with_capacity_and_key(4);
    /// let welcome: MessageKey = messages.insert("Welcome");
    /// let good_day = messages.insert("Good day");
    /// let hello = messages.insert("Hello");
    /// let bye = messages.insert("Bye");
    ///
    /// // AtomicHashMap allocates chunks starting from `32` and then scales by 4
    /// assert_eq!(messages.capacity(), 32);
    /// ```
    pub fn with_capacity_and_key(capacity: usize) -> Self {
        Self {
            slots: AtomicVec::with_capacity(capacity),
            free_head: AtomicU64::new(pack_free_head(0, SENTINEL)),
            num_elems: AtomicU32::new(0),
            _k: PhantomData,
        }
    }

    /// Returns the number of elements in the slot map.
    ///
    /// Because of the nature of lock-free structures, you cannot
    /// trust the result to not change unpredictably.
    ///
    /// # Examples
    ///
    /// ```
    /// # use atomic_slotmap::*;
    /// let mut sm = AtomicSlotMap::with_capacity(10);
    /// sm.insert("len() counts actual elements, not capacity");
    /// let key = sm.insert("removed elements don't count either");
    /// sm.remove(key);
    /// assert_eq!(sm.len(), 1);
    /// ```
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.num_elems.load(Ordering::Acquire) as usize
    }

    /// Returns if the slot map is empty.
    ///
    /// Because of the nature of lock-free structures, you cannot
    /// trust the result to not change unpredictably.
    ///
    /// # Examples
    ///
    /// ```
    /// # use atomic_slotmap::*;
    /// let mut sm = AtomicSlotMap::new();
    /// let key = sm.insert("dummy");
    /// assert_eq!(sm.is_empty(), false);
    /// sm.remove(key);
    /// assert_eq!(sm.is_empty(), true);
    /// ```
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the number of elements the [`AtomicSlotMap`] can hold without
    /// reallocating.
    ///
    /// Because of the nature of lock-free structures, you cannot
    /// trust the result to not change unpredictably.
    ///
    /// Note that an atomic slot map is based on an atomic vector-like solution,
    /// which cannot allocate exact capacities most of the time. This means that
    /// only `AtomicSlotMap::with_capacity(n).capacity() >= n` is guaranteed. The
    /// structure will allocate chunks of powers of two. It will allocate as many
    /// chunks as needed to fit the contained amount of elements.
    ///
    /// # Examples
    ///
    /// ```
    /// # use atomic_slotmap::*;
    /// let sm: AtomicSlotMap<_, f64> = AtomicSlotMap::with_capacity(50);
    ///
    /// assert_eq!(sm.capacity(), 32 + 128);
    /// ```
    pub fn capacity(&self) -> usize {
        self.slots.capacity()
    }

    /// Reserves capacity for at least `additional` more elements to be inserted
    /// in the [`AtomicSlotMap`]. The collection may reserve more space to avoid
    /// frequent reallocations.
    ///
    /// Because of the nature of lock-free structures, you cannot
    /// trust the result to not change unpredictably.
    ///
    /// # Panics
    ///
    /// Panics if the new allocation size overflows [`usize`].
    ///
    /// # Examples
    ///
    /// ```
    /// # use atomic_slotmap::*;
    /// let mut sm = AtomicSlotMap::new();
    /// sm.insert("foo");
    /// sm.reserve(32);
    /// assert!(sm.capacity() >= 33);
    /// ```
    pub fn reserve(&self, additional: usize) {
        let needed = (self.len() + additional).saturating_sub(self.slots.len() as usize);
        self.slots.reserve(needed);
    }

    // /// Tries to reserve capacity for at least `additional` more elements to be
    // /// inserted in the [`SlotMap`]. The collection may reserve more space to
    // /// avoid frequent reallocations.
    // ///
    // /// # Examples
    // ///
    // /// ```
    // /// # use atomic_slotmap::*;
    // /// let mut sm = SlotMap::new();
    // /// sm.insert("foo");
    // /// sm.try_reserve(32).unwrap();
    // /// assert!(sm.capacity() >= 33);
    // /// ```
    // pub fn try_reserve(&self, additional: usize) -> Result<(), TryReserveError> {
    //     // One slot is reserved for the sentinel.
    //     let needed = (self.len() + additional).saturating_sub(self.slots.len() - 1);
    //     self.slots.try_reserve(needed)
    // }

    /// Returns `true` if the slot map contains `key`.
    ///
    /// This will return `false` if the `key` is still stored within one of the
    /// slots of the hashmap, but wasn't dropped h
    ///
    /// # Examples
    ///
    /// ```
    /// # use atomic_slotmap::*;
    /// let mut sm = AtomicSlotMap::new();
    /// let key = sm.insert(42);
    /// assert_eq!(sm.contains_key(key), true);
    /// sm.remove(key);
    /// assert_eq!(sm.contains_key(key), false);
    /// ```
    pub fn contains_key(&self, key: K) -> bool {
        let kd = key.data();
        self.slots.get(kd.idx()).map_or(false, |slot| {
            slot.version.load(Ordering::Acquire) == kd.version().get()
        })
    }

    /// Inserts a value into the slot map. Returns a unique key that can be used
    /// to access this value.
    ///
    /// # Panics
    ///
    /// Panics if the slot map is full.
    ///
    /// # Examples
    ///
    /// ```
    /// # use atomic_slotmap::*;
    /// let mut sm = AtomicSlotMap::new();
    /// let key = sm.insert(42);
    ///
    /// let guard = sm.get(key);
    /// assert_eq!(guard.as_deref(), Some(&42));
    /// ```
    #[inline(always)]
    pub fn insert(&self, value: V) -> K {
        let Ok(key) = self.try_insert_with_key::<_, Infallible>(move |_| Ok(value));

        key
    }

    /// Inserts a value given by `f` into the slot map. The key where the
    /// value will be stored is passed into `f`. This is useful to store values
    /// that contain their own key.
    ///
    /// # Panics
    ///
    /// Panics if the slot map is full.
    ///
    /// # Examples
    ///
    /// ```
    /// # use atomic_slotmap::*;
    /// let mut sm = AtomicSlotMap::new();
    /// let key = sm.insert_with_key(|k| (k, 20));
    ///
    /// let guard = sm.get(key);
    /// assert_eq!(guard.as_deref(), Some(&(key, 20)));
    /// ```
    #[inline(always)]
    pub fn insert_with_key<F>(&self, f: F) -> K
    where
        F: FnOnce(K) -> V,
    {
        let Ok(key) = self.try_insert_with_key::<_, Infallible>(move |k| Ok(f(k)));

        key
    }

    /// Inserts a value given by `f` into the slot map. The key where the
    /// value will be stored is passed into `f`. This is useful to store values
    /// that contain their own key.
    ///
    /// If `f` returns `Err`, this method returns the error. The slotmap is untouched.
    ///
    /// # Panics
    ///
    /// Panics if the slot map is full.
    ///
    /// # Examples
    ///
    /// ```
    /// # use atomic_slotmap::*;
    /// let mut sm = AtomicSlotMap::new();
    /// let key = sm.try_insert_with_key::<_, ()>(|k| Ok((k, 20))).unwrap();
    ///
    /// let guard = sm.get(key);
    /// assert_eq!(guard.as_deref(), Some(&(key, 20)));
    /// ```
    pub fn try_insert_with_key<F, E>(&self, f: F) -> Result<K, E>
    where
        F: FnOnce(K) -> Result<V, E>,
    {
        if let Some(slot_idx) = self.pop_free_index() {
            let slot = unsafe { self.slots.get_unchecked(slot_idx) };

            let occupied_version = slot.version.load(Ordering::Acquire) | 1;

            // only referenced by the slotmap
            slot.ref_count.store(1, Ordering::Release);

            let kd = KeyData::new(slot_idx, occupied_version);

            // Get value first in case f panics or returns an error.
            let value = f(kd.into())?;

            // SAFETY: we have exclusive access to all the free slots because
            // it was just popped out of free node indices (so its not in the
            // list of free nodes) and also it has an unoccupied version index
            // so forged keys won't be able to read its value
            unsafe { *slot.data_ptr() = MaybeUninit::new(value) };

            // we have exclusive access into slot so we can just write the value back
            slot.version.store(occupied_version, Ordering::Release);

            self.num_elems.fetch_add(1, Ordering::Release);

            return Ok(kd.into());
        }

        let version = 1;

        // SAFETY: the zeroed representation of `Slot<T>` is perfectly fine to read.
        // `value` is `UnsafeCell::new(MaybeUninit::uninit())` which is perfectly okay,
        // and both `ref_count` and `version` are initialized to zero, which is exactly
        // what we want. Such slot, when read, will have a version-count of zero, which
        // means that it is unoccupied, and therefore inaccessible to any calls which
        // try to use a forged key to access it.
        let pushed_index = unsafe { self.slots.push_zeroed() };

        let kd = KeyData::new(pushed_index, version);

        let slot = unsafe { self.slots.get_unchecked(pushed_index) };

        // SAFETY: we have exclusive access to all the free slots because
        // its not referred to the list of free nodes and also it has an
        // unoccupied version index so forged keys won't be able to read
        // its value
        unsafe { *slot.data_ptr() = MaybeUninit::new(f(kd.into())?) }

        slot.ref_count.store(1, Ordering::Release);

        // we only increment the version (mark the slot occupied) after
        // intializing it with a value
        slot.version.fetch_add(1, Ordering::Release);

        self.num_elems.fetch_add(1, Ordering::Release);

        Ok(kd.into())
    }

    /// Removes a key from the slot map, returning whether the key
    /// was previously contained within the slotmap.
    ///
    /// The key will no longer be retrievable. Any [`SlotGuard`]s
    /// pointing to the key will still be valid. The last alive
    /// guard will drop the value. If no guards are active, the
    /// value will be dropped inside this function.
    ///
    /// # Examples
    ///
    /// ```
    /// # use atomic_slotmap::*;
    /// let mut sm = AtomicSlotMap::new();
    /// let key = sm.insert(42);
    ///
    /// let guard = sm.get(key);
    /// assert_eq!(guard.as_deref(), Some(&42));
    ///
    /// assert_eq!(sm.remove(key), true);
    ///
    /// assert_eq!(guard.as_deref(), Some(&42));
    ///
    /// assert_eq!(sm.remove(key), false);
    /// ```
    pub fn remove(&self, key: K) -> bool {
        let kd = key.data();

        let Some(slot) = self.slots.get(kd.idx()) else {
            return false;
        };

        let mut current_version = slot.version.load(Ordering::Acquire);

        // we enter a CAS loop to ensure that nothing marks this slot down
        // to be dropped (by decrementing its reference count) concurrently.
        // we also ensure the version matches whatever it is supposed to be.
        loop {
            // this means that the key is outdated / got outdated (perhaps
            // because of another .remove in parallel?)
            if current_version != kd.version().get() {
                return false;
            }

            let unoccupied_version = current_version.wrapping_add(1);

            match slot.version.compare_exchange(
                current_version,
                unoccupied_version,
                Ordering::Release,
                Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(v) => current_version = v,
            }
        }

        let ref_count = slot.ref_count.fetch_sub(1, Ordering::AcqRel);

        // we're the last reference to the slot. the slot had no locks acquired.
        // we're responsible for dropping its inner value
        if ref_count == 1 {
            unsafe {
                slot.drop_inner_value();
                self.push_free_index(kd.idx());
            }
        }

        self.num_elems.fetch_sub(1, Ordering::Release);

        // we've successfully removed the value. this does not mean that it was dropped,
        // but it will be eventually dropped when the last reference to it drops.
        true
    }

    /// Returns a guard into the value corresponding to the key.
    ///
    /// Multiple guards can coexist at once. [`AtomicSlotMap::remove`]
    /// calls won't invalidate such guards. In case the key gets removed
    /// from the slotmap, the last existing [`SlotGuard`] will drop the
    /// value.
    ///
    /// # Examples
    ///
    /// ```
    /// # use atomic_slotmap::*;
    /// let mut sm = AtomicSlotMap::new();
    /// let key = sm.insert("bar");
    ///
    /// let guard = sm.get(key);
    /// assert_eq!(guard.as_deref(), Some(&"bar"));
    ///
    /// sm.remove(key);
    ///
    /// // guard is still valid. the value won't actually
    /// // be dropped until this guard is dropped.
    /// assert_eq!(guard.as_deref(), Some(&"bar"));
    ///
    /// // no new guards can be created now that the value is
    /// // removed
    /// let guard = sm.get(key);
    /// assert_eq!(guard.as_deref(), None);
    ///
    /// ```
    pub fn get(&self, key: K) -> Option<SlotGuard<'_, K, V>> {
        // if the key points to an unoccupied slot then
        // it won't ever point to an occupied slot
        if key.data().version().get() % 2 == 0 {
            return None;
        }

        SlotGuard::new(key, self)
    }

    /// Returns a guard into the value corresponding to the key.
    /// The guard has an `Arc` into the `AtomicSlotMap`, so it is
    /// not tied to the slotmap by lifetime.
    ///
    /// Multiple guards can coexist at once. [`AtomicSlotMap::remove`]
    /// calls won't invalidate such guards. In case the key gets removed
    /// from the slotmap, the last existing [`SlotGuard`] will drop the
    /// value.
    ///
    /// # Examples
    ///
    /// ```
    /// # use atomic_slotmap::*;
    /// # use std::sync::Arc;
    ///    
    /// let guard = {
    ///     let sm = Arc::new(AtomicSlotMap::new());
    ///     let key = sm.insert("bar");
    ///     let guard = sm.get_owning(key);
    ///
    ///     assert_eq!(guard.as_deref(), Some(&"bar"));
    ///
    ///     guard
    /// };
    ///
    /// // the slotmap is dropped, but the guard keeps an explicit
    /// // reference to it, and so it is still accessible
    /// assert_eq!(guard.as_deref(), Some(&"bar"));
    /// ```
    pub fn get_owning(self: &Arc<Self>, key: K) -> Option<OwningSlotGuard<K, V>> {
        // if the key points to an unoccupied slot then
        // it won't ever point to an occupied slot
        if key.data().version().get() % 2 == 0 {
            return None;
        }

        OwningSlotGuard::new(key, Arc::clone(self))
    }

    /// Pops an index from the free stack. The returned index is guaranteed
    /// to be a valid index into `self.slots`, into a slot that is unoccupied
    fn pop_free_index(&self) -> Option<u32> {
        let mut old_free_head = self.free_head.load(Ordering::Acquire);

        loop {
            let (tag, idx) = unpack_free_head(old_free_head);

            if idx == SENTINEL {
                return None;
            }

            // SAFETY: If the index is in the free list, the slot must exist.
            let slot = unsafe { self.slots.get_unchecked(idx) };

            let next_free = slot.next_free.load(Ordering::Acquire);

            // we construct a new free head which will point one index further.
            // the tag is incremented to prevent an ABA issue (look below)
            let new_free_head = pack_free_head(tag.wrapping_add(1), next_free);

            // it is entirely possible that in the meanwhile `next_free` has changed.
            // assume such stack:
            //
            // 1 -> 2 -> 3
            // ^    ^
            // |  next head
            // free head
            //
            // in the meanwhile, another thread could have popped `1`, `2`, and pushed `1` again
            // the stack is as follows:
            //
            // 1 -> 3
            //
            // next head should be 3, but it is 2. we completely alleviate the ABA by
            // keeping a tag that is incremented whenever an operation happens. this way the
            // head pointer would change from (1, 1) -> (3, 1) so it wouldn't get exchanged.
            match self.free_head.compare_exchange_weak(
                old_free_head,
                new_free_head,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Some(idx),
                Err(current) => old_free_head = current,
            }
        }
    }

    /// Pushes an index onto the free stack.
    ///
    /// The provided index will mark the slot as reusable for future
    /// insertions of keys into the slotmap.
    ///
    /// # Safety
    /// The index must be a valid index into a slot. The slot at
    /// the index must be unoccupied and have no referents.
    unsafe fn push_free_index(&self, idx: u32) {
        let slot = unsafe { self.slots.get_unchecked(idx) };
        let mut old_free_head = self.free_head.load(Ordering::Relaxed);

        loop {
            let (tag, old_next_free) = unpack_free_head(old_free_head);

            slot.next_free.store(old_next_free, Ordering::Release);

            // we increment the tag to prevent any potential ABA issues.
            // otherwise, `pop_free_index` calls would be unsound.
            let new_free_head = pack_free_head(tag.wrapping_add(1), idx);

            match self.free_head.compare_exchange_weak(
                old_free_head,
                new_free_head,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => return,
                Err(current) => old_free_head = current,
            }
        }
    }
}

impl<K: Key, V> Default for AtomicSlotMap<K, V> {
    fn default() -> Self {
        Self::with_key()
    }
}

#[cfg(test)]
mod tests {
    use quickcheck::quickcheck;
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::thread::spawn;
    use std::time::Duration;

    use super::*;

    const _: () = {
        const fn f<T: Send + Sync>() {}

        f::<AtomicSlotMap<DefaultKey, u32>>();
        f::<SlotGuard<DefaultKey, u32>>();
        f::<OwningSlotGuard<DefaultKey, u32>>();
    };

    #[derive(Clone)]
    struct CountDrop<'a>(&'a std::cell::RefCell<usize>);

    impl<'a> Drop for CountDrop<'a> {
        fn drop(&mut self) {
            *self.0.borrow_mut() += 1;
        }
    }

    #[test]
    fn check_drops() {
        let drops = std::cell::RefCell::new(0usize);

        {
            // Insert 1000 items.
            let sm = AtomicSlotMap::new();
            let mut sm_keys = Vec::new();
            for _ in 0..1000 {
                sm_keys.push(sm.insert(CountDrop(&drops)));
            }

            // Remove even keys.
            for i in (0..1000).filter(|i| i % 2 == 0) {
                sm.remove(sm_keys[i]);
            }

            // Should only have dropped 500 so far.
            assert_eq!(*drops.borrow(), 500);
        };

        // Now all original items should have been dropped exactly once.
        assert_eq!(*drops.borrow(), 1000);
    }

    #[test]
    fn check_drops_with_multiple_guards() {
        let drops = std::cell::RefCell::new(0usize);
        let sm = AtomicSlotMap::new();

        let key = sm.insert(CountDrop(&drops));

        let guard1 = sm.get(key).unwrap();
        let guard2 = sm.get(key).unwrap();

        assert_eq!(*drops.borrow(), 0);

        drop(guard1);
        drop(guard2);

        assert_eq!(*drops.borrow(), 0);

        let guard1 = sm.get(key).unwrap();
        let guard2 = sm.get(key).unwrap();

        assert!(sm.remove(key));

        assert_eq!(*drops.borrow(), 0);

        drop(guard1);

        assert_eq!(*drops.borrow(), 0);

        drop(guard2);

        assert_eq!(*drops.borrow(), 1);
    }

    quickcheck! {
        fn qc_slotmap_equiv_hashmap(operations: Vec<(u8, u32)>) -> bool {
            let mut hm = HashMap::new();
            let mut hm_keys = Vec::new();
            let mut unique_key = 0u32;
            let sm = AtomicSlotMap::new();
            let mut sm_keys = Vec::new();

            let num_ops = 3;

            for (op, val) in operations {
                match op % num_ops {
                    // Insert.
                    0 => {
                        hm.insert(unique_key, val);
                        hm_keys.push(unique_key);
                        unique_key += 1;

                        sm_keys.push(sm.insert(val));
                    }

                    // Delete.
                    1 => {
                        if hm_keys.is_empty() { continue; }

                        let idx = val as usize % hm_keys.len();

                        if hm.remove(&hm_keys[idx]).is_some() != sm.remove(sm_keys[idx]) {
                            return false;
                        }
                    }

                    // Access.
                    2 => {
                        if hm_keys.is_empty() { continue; }
                        let idx = val as usize % hm_keys.len();
                        let (hm_key, sm_key) = (&hm_keys[idx], sm_keys[idx]);

                        if hm.contains_key(hm_key) != sm.contains_key(sm_key) ||
                           hm.get(hm_key) != sm.get(sm_key).as_deref() {
                            return false;
                        }
                    }

                    _ => unreachable!(),
                }
            }

            true
        }
    }

    #[test]
    fn test_multithreaded() {
        // tests multiple threads adding and removing elements into the slotmap and verifying that
        // they have correct values. this test does not modify correct dropping of elements.
        let sm = Arc::new(AtomicSlotMap::<_, u32>::new());

        let mut threads = Vec::with_capacity(10);

        #[allow(clippy::needless_range_loop)]
        for _ in 0..10 {
            let sm = Arc::clone(&sm);

            threads.push(spawn(move || {
                let mut keys = [DefaultKey::null(); 100];

                for i in 0..100 {
                    keys[i] = sm.insert(i as u32);

                    // verify that all previous keys still have their expected values
                    for k in 0..i {
                        assert_eq!(sm.get(keys[k]).as_deref().copied(), Some(k as u32));
                    }
                }

                // now we deallocate 10 keys so that we can rest reclamation
                for i in 0..10 {
                    assert!(sm.remove(keys[i]));

                    // verify that all removed keys still have their expected values
                    for k in (i + 1)..100 {
                        assert_eq!(sm.get(keys[k]).as_deref().copied(), Some(k as u32));
                    }
                }

                // we allocate 10 keys again
                for i in 0..10 {
                    keys[i] = sm.insert(i as u32);

                    // verify that all previous keys still have their expected values
                    for k in 0..i {
                        assert_eq!(sm.get(keys[k]).as_deref().copied(), Some(k as u32));
                    }
                }

                // we deallocate all keys now
                for i in 0..100 {
                    assert!(sm.remove(keys[i]));

                    // verify that all removed keys still have their expected values
                    for k in (i + 1)..100 {
                        assert_eq!(sm.get(keys[k]).as_deref().copied(), Some(k as u32));
                    }
                }
            }));
        }

        for thread in threads {
            thread.join().unwrap()
        }
    }

    #[test]
    fn test_multithreaded_closure_insertion() {
        // this additionally stress-tests the slotmap by using a very slow (sleeping) closure
        // for the insertion of elements. this makes everything more prone to possible collisions
        let sm = Arc::new(AtomicSlotMap::<_, u32>::new());

        let mut threads = Vec::with_capacity(10);

        #[allow(clippy::needless_range_loop)]
        for _ in 0..10 {
            let sm = Arc::clone(&sm);

            threads.push(spawn(move || {
                let mut keys = [DefaultKey::null(); 100];

                for i in 0..100 {
                    keys[i] = sm.insert_with_key(|_| {
                        std::thread::sleep(Duration::from_millis(1));
                        i as u32
                    });

                    // verify that all previous keys still have their expected values
                    for k in 0..i {
                        assert_eq!(sm.get(keys[k]).as_deref().copied(), Some(k as u32));
                    }
                }

                // now we deallocate 10 keys so that we can rest reclamation
                for i in 0..10 {
                    assert!(sm.remove(keys[i]));

                    // verify that all removed keys still have their expected values
                    for k in (i + 1)..100 {
                        assert_eq!(sm.get(keys[k]).as_deref().copied(), Some(k as u32));
                    }
                }

                // we allocate 10 keys again
                for i in 0..10 {
                    keys[i] = sm.insert_with_key(|_| {
                        std::thread::sleep(Duration::from_millis(1));
                        i as u32
                    });

                    // verify that all previous keys still have their expected values
                    for k in 0..i {
                        assert_eq!(sm.get(keys[k]).as_deref().copied(), Some(k as u32));
                    }
                }

                // we deallocate all keys now
                for i in 0..100 {
                    assert!(sm.remove(keys[i]));

                    // verify that all removed keys still have their expected values
                    for k in (i + 1)..100 {
                        assert_eq!(sm.get(keys[k]).as_deref().copied(), Some(k as u32));
                    }
                }
            }));
        }

        for thread in threads {
            thread.join().unwrap();
        }
    }
}
