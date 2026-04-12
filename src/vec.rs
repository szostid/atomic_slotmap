use crate::atomic::{AtomicPtr, AtomicU32, Ordering};
use alloc::alloc;
use core::marker::PhantomData;

/// If running on loom, most sync types (atomics, cells)
/// aren't primitives that are safe to be zeroed anymore.
///
/// This trait is used to enforce T: Default to propely
/// initialize those elements if running on loom
#[cfg(not(loom))]
pub trait DefaultIfLoom {}
#[cfg(not(loom))]
impl<T> DefaultIfLoom for T {}

/// If running on loom, most sync types (atomics, cells)
/// aren't primitives that are safe to be zeroed anymore.
///
/// This trait is used to enforce T: Default to propely
/// initialize those elements if running on loom
#[cfg(loom)]
pub trait DefaultIfLoom: Default {}
#[cfg(loom)]
impl<T: Default> DefaultIfLoom for T {}

/// An atomic vector.
///
/// - You can only push the zeroed version of `T`, but its
///   done completely lock-free.
/// - `get` / `get_unchecked` are completely lockfree
/// - It's impossible to clear the contents of the vector
///   or remove its elements. Something like
///   a clear wouldn't be able to do something like
///   `len = 0 -> drop all elements` because in the
///   meanwhile another thread could possibly push a new
///   elements and write a valud at index 0. Similarly,
///   something like a `.pop` wouldn't be able to read
///   the value because another thread could claim it
///   and start writing to it.
pub struct AtomicVec<T: DefaultIfLoom> {
    chunks: [AtomicPtr<T>; 15],
    len: AtomicU32,
    _marker: PhantomData<T>,
}

impl<T: DefaultIfLoom> AtomicVec<T> {
    /// Creates a new atomic vector that is able to store zero elements.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        let chunks = core::array::from_fn(|_| AtomicPtr::new(core::ptr::null_mut()));

        Self {
            len: AtomicU32::new(0),
            chunks,
            _marker: PhantomData,
        }
    }

    /// Creates a new atomic vector that is able to store `cap` elements
    /// without performing additional allocations.
    #[must_use]
    pub fn with_capacity(cap: usize) -> Self {
        let this = Self::new();
        this.reserve(cap);
        this
    }

    /// Pushes another element to the vector. The element is initialized
    /// to its zero-value. Any `get_*` method can potentially read this
    /// value.
    ///
    /// # Safety
    /// You must ensure that the zeroed version of `T` is fine to read,
    /// because it will be returned by `get` / `get_unchecked`.
    pub unsafe fn push_zeroed(&self) -> u32 {
        // all methods which retrieve data from the vector correctly
        // handle the short time in which the `chunk_idx` is (potentially)
        // unallocated but index is incremented. this is unavoidable.
        let idx = self.len.fetch_add(1, Ordering::Relaxed);

        let (chunk_idx, _) = self.get_location(idx);

        self.ensure_chunk_exists(chunk_idx);

        idx
    }

    /// Retrieves the element at `idx`.
    ///
    /// The returned value might be zeroed, but that's an invariant
    /// that should be handled when calling [`Self::push_zeroed`]
    #[must_use]
    pub fn get(&self, idx: u32) -> Option<&T> {
        if idx >= self.len.load(Ordering::Acquire) {
            return None;
        }

        let (chunk_idx, offset) = self.get_location(idx);

        // we need to wait for the chunk. a push call that needs to allocate
        // a new chunk will, for a short time, increment the length but not
        // allocate the chunk to house the elements
        let chunk_ptr = self.wait_for_chunk(chunk_idx);

        // SAFETY: no overflow should ever occur within the length of the vector
        let elem_ptr = unsafe { chunk_ptr.add(offset) };

        // SAFETY: this is within the bounds of the length, so the pointer definitely
        // points to readable memory. the data should be either zeroed (through the
        // push_zeroed call) or written to by the user (which should be done safely)
        unsafe { Some(&*elem_ptr) }
    }

    /// Retrieves the element at `idx` without checking whether it
    /// is within the bounds of the vector.
    ///
    /// The returned value might be zeroed, but that's an invariant
    /// that should be handled when calling [`Self::push_zeroed`]
    ///
    /// # Safety
    /// Only safe to call if `idx` is contained within the vector.
    #[inline]
    #[must_use]
    pub unsafe fn get_unchecked(&self, idx: u32) -> &T {
        // SAFETY: the user ensures (through the safety clause) that this is
        // within the length of the vector
        unsafe { &*self.get_unchecked_ptr(idx) }
    }

    /// Retrieves the pointer to the element at `idx` without checking
    /// whether it is within the bounds of the vector.
    ///
    /// The returned value might be zeroed, but that's an invariant
    /// that should be handled when calling [`Self::push_zeroed`]
    ///
    /// # Safety
    /// Only safe to call if `idx` is contained within the vector.
    pub unsafe fn get_unchecked_ptr(&self, idx: u32) -> *mut T {
        let (chunk_idx, offset) = self.get_location(idx);

        // we need to wait for the chunk. a push call that needs to allocate
        // a new chunk will, for a short time, increment the length but not
        // allocate the chunk to house the elements
        let chunk_ptr = self.wait_for_chunk(chunk_idx);

        // SAFETY: no overflow should ever occur within the length of the vector.
        // the user ensures (through the safety clause) that this is within the
        // length of the vectore
        unsafe { chunk_ptr.add(offset) }
    }

    /// Reserves the spaced for `additional` additional elements in the vector.
    pub fn reserve(&self, additional: usize) {
        let len = self.len.load(Ordering::Relaxed);
        let target_cap = len as usize + additional;

        if target_cap == 0 {
            return;
        }

        let (max_chunk, _) = self.get_location((target_cap - 1) as u32);

        // preallocate all required chunks
        for i in 0..=max_chunk {
            self.ensure_chunk_exists(i);
        }
    }

    /// Returns the approximate capacity of the vector.
    ///
    /// The returned value can't fully be trusted to perform any
    /// operations because, whatever the function returns, will
    /// always be subject to TOCTOU.
    pub fn capacity(&self) -> usize {
        let mut total_cap = 0;

        for (i, chunk) in self.chunks.iter().enumerate() {
            // relaxed is fine, not much will happen if we don't count a
            // newly initialized chunk
            let ptr = chunk.load(Ordering::Relaxed);
            if !ptr.is_null() {
                total_cap += 32_usize << (i * 2);
            } else {
                // chunks are allocated in order, so if we see the first
                // null value then all consecutive chunks are null
                break;
            }
        }
        total_cap
    }

    /// Returns the approximate length of the vector.
    ///
    /// The returned value can't fully be trusted to perform any
    /// operations because, whatever the function returns, will
    /// always be subject to TOCTOU.
    #[inline]
    #[must_use]
    pub fn len(&self) -> u32 {
        self.len.load(Ordering::Acquire)
    }

    /// Returns the `(chunk, offset)` of the given index.
    #[inline]
    fn get_location(&self, idx: u32) -> (usize, usize) {
        // We want to map `idx` to a chunk `k` where sizes grow as 32 * 4^k.
        // The cumulative capacity before chunk k is: Sum(32 * 4^i) = 32 * (4^k - 1) / 3
        // We solve for k:
        // 32 * (4^k - 1) / 3 <= idx
        // 4^k <= (3 * idx / 32) + 1
        // 2k <= log2((3 * idx / 32) + 1)

        let val = (3 * (idx as u64) / 32) + 1;
        let log2 = 63 - val.leading_zeros();
        let chunk_idx = (log2 / 2) as usize;

        // These are the start indices of subsequent chunks
        const STARTS: [u32; 15] = [
            0,          // size: 32,         starts at 0
            32,         // size: 128,        starts at 32         + 0         = 32
            160,        // size: 512,        starts at 128        + 32        = 160
            672,        // size: 2048,       starts at 512        + 160       = 672
            2720,       // size: 8192,       starts at 2048       + 672       = 2720
            10912,      // size: 32768,      starts at 8192       + 2720      = 10912
            43680,      // size: 131072,     starts at 32768      + 10912     = 43680
            174752,     // size: 524288,     starts at 131072     + 43680     = 174752
            699040,     // size: 2097152,    starts at 524288     + 174752    = 699040
            2796192,    // size: 8388608,    starts at 2097152    + 699040    = 2796192
            11184800,   // size: 33554432,   starts at 8388608    + 2796192   = 11184800
            44739232,   // size: 134217728,  starts at 33554432   + 11184800  = 44739232
            178956960,  // size: 536870912,  starts at 134217728  + 44739232  = 178956960
            715827872,  // size: 2147483648, starts at 536870912  + 178956960 = 715827872
            2863311520, // size: 8589934592, starts at 2147483648 + 715827872 = 2863311520
        ];

        let start = unsafe { *STARTS.get_unchecked(chunk_idx) };
        let offset = (idx - start) as usize;

        (chunk_idx, offset)
    }

    /// Ensures that a chunk with the provided index exists and returns a
    /// pointer to its first element
    fn ensure_chunk_exists(&self, chunk_idx: usize) -> *mut T {
        // try to first fetch the pointer. we will allocate if there's nothing else to do
        let ptr = self.chunks[chunk_idx].load(Ordering::Acquire);
        if !ptr.is_null() {
            return ptr;
        }

        let cap = 32_usize << (chunk_idx * 2);
        let layout = alloc::Layout::array::<T>(cap).unwrap();

        unsafe {
            // we alloc_zeroed because this is mainly used by the `push_zeroed` call. we
            // want to ensure that whatever could possibly be visible to readers is always
            // zeroed (or written to, in a safe manner, by the user)
            let new_ptr = alloc::alloc_zeroed(layout) as *mut T;

            #[cfg(loom)]
            {
                // in loom, we cannot depend on the zeroed version of T (slot)
                // to be a valid object (i.e. a zeroed AtomicU32 is correctly
                // a 0, but in loom its a more complex object that needs proper
                // initialization)
                for i in 0..cap {
                    core::ptr::write(new_ptr.add(i), T::default());
                }
            }

            match self.chunks[chunk_idx].compare_exchange(
                core::ptr::null_mut(),
                new_ptr,
                Ordering::Release,
                Ordering::Acquire,
            ) {
                Ok(_) => new_ptr,
                Err(existing) => {
                    // the chunk was already allocated somewhere else. we need to
                    // deallocate the current pointer and return the existing chunk
                    alloc::dealloc(new_ptr as *mut u8, layout);
                    existing
                }
            }
        }
    }

    /// Waits for a chunk to get loaded in a spinloop. `push_zeroed` calls will,
    /// for a short while, increment the length but not allocate the chunk yet,
    /// so this should wait for those calls to write the chunk pointers
    fn wait_for_chunk(&self, chunk_idx: usize) -> *mut T {
        let mut ptr = self.chunks[chunk_idx].load(Ordering::Acquire);

        while ptr.is_null() {
            #[cfg(not(loom))]
            core::hint::spin_loop();
            #[cfg(loom)]
            loom::thread::yield_now();
            ptr = self.chunks[chunk_idx].load(Ordering::Acquire);
        }

        ptr
    }
}

impl<T: DefaultIfLoom> Drop for AtomicVec<T> {
    fn drop(&mut self) {
        // drop all elements, then deallocate all chunks

        if core::mem::needs_drop::<T>() {
            for i in 0..self.len.load(Ordering::Relaxed) {
                let (chunk_idx, offset) = self.get_location(i);

                let chunk_ptr = self.chunks[chunk_idx].load(Ordering::Relaxed);

                // its possible for chunk_ptr to be null if a thread
                // has panicked when allocating the chunk.
                if !chunk_ptr.is_null() {
                    unsafe {
                        let elem_ptr = chunk_ptr.add(offset);
                        core::ptr::drop_in_place(elem_ptr);
                    }
                }
            }
        }

        for (i, chunk) in self.chunks.iter().enumerate() {
            let ptr = chunk.load(Ordering::Relaxed);

            if !ptr.is_null() {
                let cap = 32_usize << (i * 2);
                let layout = alloc::Layout::array::<T>(cap).unwrap();

                unsafe {
                    alloc::dealloc(ptr as *mut u8, layout);
                }
            }
        }
    }
}
