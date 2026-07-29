# `atomic_slotmap`

This library is an extension to the [slotmap](https://crates.io/crates/slotmap) crate, adding an atomic slotmap which can be modified without a mutable reference, in a lockfree[^1] way.

The API of this crate is mostly similar to the slotmaps from the slotmap crate. A `get_owning` operation which isn't tied to the lifetime of the slotmap is added (requiring `Arc<Self>`).

## Limitations

The lock-free-ness of this slotmap makes it impossible to bulk read / modify its contents, so the potentially TOCTOU-prone methods are marked as `lossy_...` and require the `lossy` feature to be enabled (enabled by default). It is impossible to clone or clear this structure.

## Performance

The structure is perfect for minimizing latency, as it allows for multiple readers at the same time, while still allowing for insertions to happen (they don't block each other at all). Uncontended insertions are relatively performant, with singlethreaded insertion tests showing about the same performance as a regular `Mutex<SlotMap<...>>`. The worst-case scenario in terms of usage is pure insertions with high contention. Note that if ~10% of the operations involve insertions and the rest is reads, which is a more real-life workload, the `AtomicSlotMap` seems to outperform the `SlotMap` in all multithreaded tests.

Note that there are two versions of the benchmark: one for a situation where you need a mutable access to the elements within (and hence the `AtomicSlotMap<RwLock<...>>`), and one for the situation where mutable access is not needed (which actually aligns better with the purpose of this structure). Mixed workload is 90% read, 10% insertion.

<img src="perf.png" alt="example"/>
This test was performed on an M2 macbook air with 8 cores.
<img src="perf_i5.png" alt="example"/>
This test was performed on an Intel i5-12400KF

[^1]: Threads might spinloop for very short amounts of time when an atomic vector allocated a new chunk of memory (this code path is hard to hit even after a long time of fuzzing)