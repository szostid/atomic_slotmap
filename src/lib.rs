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

struct Slot<T> {
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
    data: UnsafeCell<MaybeUninit<T>>,
    next_free: AtomicU32,
    /// Reference count. Includes the slotmap and all SlotGuards.
    ref_count: AtomicU32,
    version: AtomicU32,
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
    fn data_ptr(&self) -> *mut MaybeUninit<T> {
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
    unsafe fn drop_inner_value(&self) {
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
    fn new(key: K, map: &'a AtomicSlotMap<K, V>) -> Option<Self> {
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
    /// // AtomicHashMap allocates chunks of powers of two. It will
    /// // allocate as many chunks as needed to fit the specified
    /// // amount of elements.
    /// assert_eq!(messages.capacity(), 1 + 2 + 4);
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
    /// let sm: AtomicSlotMap<_, f64> = AtomicSlotMap::with_capacity(10);
    ///
    /// assert_eq!(sm.capacity(), 1 + 2 + 4 + 8);
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

        if let Some(slot) = self.slots.get(kd.idx()) {
            let mut current_version = slot.version.load(Ordering::Acquire);

            // we enter a CAS loop to ensure that nothing marks this slot down
            // to be dropped (by decrementing its reference count) concurrently.
            // we also ensure the version matches whatever it is supposed to be.
            loop {
                // this means that the key is outdated
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
