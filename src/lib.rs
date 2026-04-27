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

#[cfg(not(loom))]
use alloc::sync::Arc;
#[cfg(loom)]
use loom::sync::Arc;

#[cfg(not(loom))]
use core::cell::UnsafeCell;
#[cfg(loom)]
use loom::cell::UnsafeCell;

#[cfg(not(loom))]
use core::sync::atomic;
#[cfg(loom)]
use loom::sync::atomic;

use crate::atomic::{AtomicU32, AtomicU64, Ordering};
use core::convert::Infallible;
use core::marker::PhantomData;
use core::mem::MaybeUninit;
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

mod iter;
pub use iter::LossyIter;

const SENTINEL: u32 = u32::MAX;

/// Creates a packed `free_head` from a tag and index
#[inline]
#[must_use]
const fn pack_free_head(tag: u32, index: u32) -> u64 {
    ((tag as u64) << 32) | (index as u64)
}

/// Unpacks a packed `free_head` element into its tag and index
#[inline]
#[must_use]
const fn unpack_free_head(packed: u64) -> (u32, u32) {
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
    /// `free_head` is packed using [`pack_free_head`] and contains
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
    pub const fn new() -> Self {
        Self::with_key()
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
    pub const fn with_key() -> Self {
        Self {
            slots: AtomicVec::new(),
            free_head: AtomicU64::new(pack_free_head(0, SENTINEL)),
            num_elems: AtomicU32::new(0),
            _k: PhantomData,
        }
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
    /// If `f` returns `Err`, no element is inserted.
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
        let (slot_idx, cur_version) = if let Some(slot_idx) = self.pop_free_index() {
            let slot = unsafe { self.slots.get_unchecked(slot_idx) };
            let version = slot.version.load(Ordering::Acquire);
            (slot_idx, version)
        } else {
            // SAFETY: the zeroed representation of `Slot<T>` is perfectly fine to read.
            // `value` is `UnsafeCell::new(MaybeUninit::uninit())` which is perfectly okay,
            // and both `ref_count` and `version` are initialized to zero, which is exactly
            // what we want. Such slot, when read, will have a version-count of zero, which
            // means that it is unoccupied, and therefore inaccessible to any calls which
            // try to use a forged key to access it.
            let pushed_index = unsafe { self.slots.push_zeroed() };
            (pushed_index, 0)
        };

        let occupied_version = cur_version | 1;

        let kd = KeyData::new(slot_idx, occupied_version);

        // Get value first in case f panics or returns an error.
        let value = match f(kd.into()) {
            Ok(value) => value,
            Err(err) => {
                // SAFETY: the slot was just popped from the free list or is a fresh slot,
                // and is still unoccupied with no referents.
                unsafe { self.push_free_index(slot_idx) };
                return Err(err);
            }
        };

        let slot = unsafe { self.slots.get_unchecked(slot_idx) };

        slot.debug_assert_exclusively_owned();

        // SAFETY: we have exclusive access to the slot because we've just
        // popped it off the free list. the slot is exclusively owned, we
        // can write the data into it
        unsafe { *slot.data_ptr() = MaybeUninit::new(value) };

        // only referenced by the slotmap
        slot.ref_count.store(1, Ordering::Release);

        // we have exclusive access into slot so we can just write the value back
        slot.version.store(occupied_version, Ordering::Release);

        self.num_elems.fetch_add(1, Ordering::Relaxed);

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

        let needs_drop = slot.dec_ref_count();

        // we're the last reference to the slot. the slot had no locks acquired.
        // we're responsible for dropping its inner value
        if needs_drop {
            unsafe {
                slot.drop_inner_value();
                self.push_free_index(kd.idx());
            }
        }

        self.num_elems.fetch_sub(1, Ordering::Relaxed);

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
    #[cfg(not(loom))]
    pub fn get_owning(self: &Arc<Self>, key: K) -> Option<OwningSlotGuard<K, V>> {
        // if the key points to an unoccupied slot then
        // it won't ever point to an occupied slot
        if key.data().version().get() % 2 == 0 {
            return None;
        }

        OwningSlotGuard::new(key, Arc::clone(self))
    }

    /// Returns an owning slot guard (loom version, not associated)
    #[cfg(loom)]
    pub fn get_owning(this: &Arc<Self>, key: K) -> Option<OwningSlotGuard<K, V>> {
        // if the key points to an unoccupied slot then
        // it won't ever point to an occupied slot
        if key.data().version().get() % 2 == 0 {
            return None;
        }

        OwningSlotGuard::new(key, Arc::clone(this))
    }

    /// Pops an index from the free slot linked list. The returned index is guaranteed
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
                Ok(_) => {
                    slot.debug_assert_exclusively_owned();
                    return Some(idx);
                }
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

        slot.debug_assert_exclusively_owned();

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

/// Implementation of lossy methods
#[cfg(feature = "lossy")]
impl<K: Key, V> AtomicSlotMap<K, V> {
    /// Reserves capacity for at least `additional` more elements to be inserted
    /// in the [`AtomicSlotMap`]. The collection may reserve more space to avoid
    /// frequent reallocations.
    ///
    /// Because of the nature of lock-free structures, you cannot fully trust the
    /// reservation to be successful (e.g. another thread could add an element in
    /// the meanwhile and now there isn't enough elements leftover), but this should
    /// rarely be an issue because the [`AtomicVec`] that backs the slotmap allocates
    /// very rarely.
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
        let needed = (self.lossy_len() + additional).saturating_sub(self.slots.len() as usize);
        self.slots.reserve(needed);
    }

    /// Returns the number of elements in the slot map.
    ///
    /// If the map is being used on a single thread, the
    /// length will likely be correct. On concurrent accesses
    /// however, this can change dynamically and cannot
    /// be depended on.
    ///
    /// # Examples
    ///
    /// ```
    /// # use atomic_slotmap::*;
    /// let mut sm = AtomicSlotMap::with_capacity(10);
    /// sm.insert("len() counts actual elements, not capacity");
    /// let key = sm.insert("removed elements don't count either");
    /// sm.remove(key);
    /// assert_eq!(sm.lossy_len(), 1);
    /// ```
    #[inline]
    #[must_use]
    pub fn lossy_len(&self) -> usize {
        self.num_elems.load(Ordering::Relaxed) as usize
    }

    /// Returns if the slot map is empty.
    ///
    /// If the map is being used on a single thread, the
    /// result will likely be correct. On concurrent accesses
    /// however, this can change dynamically and cannot
    /// be depended on.
    ///
    /// # Examples
    ///
    /// ```
    /// # use atomic_slotmap::*;
    /// let mut sm = AtomicSlotMap::new();
    /// let key = sm.insert("dummy");
    /// assert_eq!(sm.lossy_is_empty(), false);
    /// sm.remove(key);
    /// assert_eq!(sm.lossy_is_empty(), true);
    /// ```
    #[inline]
    #[must_use]
    pub fn lossy_is_empty(&self) -> bool {
        self.lossy_len() == 0
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

    /// Returns a lossy iterator over the currently occupied slots.
    ///
    /// This iterator does not guarantee it will visit every element if concurrent
    /// insertions or removals are happening.
    pub fn lossy_iter(&self) -> LossyIter<'_, K, V> {
        LossyIter::new(self)
    }
}

impl<K: Key, V> Default for AtomicSlotMap<K, V> {
    fn default() -> Self {
        Self::with_key()
    }
}
