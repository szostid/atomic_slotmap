use alloc::alloc;
use core::sync::atomic::{AtomicPtr, AtomicU32, Ordering};

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
pub struct AtomicVec<T> {
    chunks: [AtomicPtr<T>; 32],
    len: AtomicU32,
}

impl<T> AtomicVec<T> {
    /// Creates a new atomic vector that is able to store zero elements.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        // equivalent to [AtomicPtr::new(ptr::null_mut()); 32] but AtomicPtr does not implement copy
        let chunks = unsafe { core::mem::MaybeUninit::zeroed().assume_init() };

        Self {
            len: AtomicU32::new(0),
            chunks,
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
                total_cap += 1 << i;
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
        let pos = (idx as usize) + 1;
        let chunk_idx = (usize::BITS - 1) as usize - pos.leading_zeros() as usize;
        let base = 1 << chunk_idx;
        let offset = pos - base;
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

        let cap = 1 << chunk_idx;
        let layout = alloc::Layout::array::<T>(cap).unwrap();

        unsafe {
            // we alloc_zeroed because this is mainly used by the `push_zeroed` call. we
            // want to ensure that whatever could possibly be visible to readers is always
            // zeroed (or written to, in a safe manner, by the user)
            let new_ptr = alloc::alloc_zeroed(layout) as *mut T;

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
            core::hint::spin_loop();
            ptr = self.chunks[chunk_idx].load(Ordering::Acquire);
        }

        ptr
    }
}

impl<T> Drop for AtomicVec<T> {
    fn drop(&mut self) {
        // drop all elements, then deallocate all chunks

        if core::mem::needs_drop::<T>() {
            for i in 0..*self.len.get_mut() {
                let (chunk_idx, offset) = self.get_location(i);

                let chunk_ptr = *self.chunks[chunk_idx].get_mut();

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

        for (i, chunk) in self.chunks.iter_mut().enumerate() {
            let ptr = *chunk.get_mut();

            if !ptr.is_null() {
                let cap = 1 << i;
                let layout = alloc::Layout::array::<T>(cap).unwrap();

                unsafe {
                    alloc::dealloc(ptr as *mut u8, layout);
                }
            }
        }
    }
}
