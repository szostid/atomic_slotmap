use atomic_slotmap::AtomicSlotMap;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use slotmap::{DefaultKey, SlotMap};
use std::sync::{Arc, Barrier};
use std::time::{Duration, Instant};

const OPS_PER_THREAD: usize = 1000;

pub trait Benchmark<S: ConcurrentMap> {
    type SharedState: Send + Sync;
    fn name() -> &'static str;
    fn setup(map: &S) -> Self::SharedState;
    fn run(map: Arc<S>, state: Arc<Self::SharedState>, thread_id: usize);
}

pub trait ConcurrentMap: Send + Sync + 'static {
    type Key: Send + Sync + Copy;
    fn name() -> &'static str;
    fn create() -> Self;
    fn insert(&self, val: usize) -> Self::Key;
    fn get(&self, key: Self::Key) -> Option<usize>;
}

impl ConcurrentMap for AtomicSlotMap<DefaultKey, usize> {
    type Key = DefaultKey;

    #[inline(always)]
    fn name() -> &'static str {
        "AtomicSlotMap"
    }

    #[inline(always)]
    fn create() -> Self {
        AtomicSlotMap::new()
    }

    #[inline(always)]
    fn insert(&self, val: usize) -> Self::Key {
        self.insert(val)
    }

    #[inline(always)]
    fn get(&self, key: Self::Key) -> Option<usize> {
        Some(*self.get(key)?)
    }
}

pub struct StdMutexMap(std::sync::Mutex<SlotMap<DefaultKey, usize>>);

impl ConcurrentMap for StdMutexMap {
    type Key = DefaultKey;

    #[inline(always)]
    fn name() -> &'static str {
        "std::Mutex"
    }

    #[inline(always)]
    fn create() -> Self {
        Self(std::sync::Mutex::new(SlotMap::with_key()))
    }

    #[inline(always)]
    fn insert(&self, val: usize) -> Self::Key {
        self.0.lock().unwrap().insert(val)
    }

    #[inline(always)]
    fn get(&self, key: Self::Key) -> Option<usize> {
        self.0.lock().unwrap().get(key).copied()
    }
}

pub struct PlMutexMap(parking_lot::Mutex<SlotMap<DefaultKey, usize>>);

impl ConcurrentMap for PlMutexMap {
    type Key = DefaultKey;

    #[inline(always)]
    fn name() -> &'static str {
        "parking_lot::Mutex"
    }

    #[inline(always)]
    fn create() -> Self {
        Self(parking_lot::Mutex::new(SlotMap::with_key()))
    }

    #[inline(always)]
    fn insert(&self, val: usize) -> Self::Key {
        self.0.lock().insert(val)
    }

    #[inline(always)]
    fn get(&self, key: Self::Key) -> Option<usize> {
        self.0.lock().get(key).copied()
    }
}

pub struct StdRwLockMap(std::sync::RwLock<SlotMap<DefaultKey, usize>>);

impl ConcurrentMap for StdRwLockMap {
    type Key = DefaultKey;

    #[inline(always)]
    fn name() -> &'static str {
        "std::RwLock"
    }

    #[inline(always)]
    fn create() -> Self {
        Self(std::sync::RwLock::new(SlotMap::with_key()))
    }

    #[inline(always)]
    fn insert(&self, val: usize) -> Self::Key {
        self.0.write().unwrap().insert(val)
    }

    #[inline(always)]
    fn get(&self, key: Self::Key) -> Option<usize> {
        self.0.read().unwrap().get(key).copied()
    }
}

pub struct PlRwLockMap(parking_lot::RwLock<SlotMap<DefaultKey, usize>>);

impl ConcurrentMap for PlRwLockMap {
    type Key = DefaultKey;

    #[inline(always)]
    fn name() -> &'static str {
        "parking_lot::RwLock"
    }

    #[inline(always)]
    fn create() -> Self {
        Self(parking_lot::RwLock::new(SlotMap::with_key()))
    }

    #[inline(always)]
    fn insert(&self, val: usize) -> Self::Key {
        self.0.write().insert(val)
    }

    #[inline(always)]
    fn get(&self, key: Self::Key) -> Option<usize> {
        self.0.read().get(key).copied()
    }
}

fn run_bench<S: ConcurrentMap, B: Benchmark<S>>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    thread_count: usize,
) {
    group.bench_function(S::name(), |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;

            for _ in 0..iters {
                let map = Arc::new(S::create());
                let state = Arc::new(B::setup(&map));

                // We need two barriers to box in the actual workload
                let start_barrier = Arc::new(Barrier::new(thread_count + 1));
                let end_barrier = Arc::new(Barrier::new(thread_count + 1));

                std::thread::scope(|s| {
                    for thread_id in 0..thread_count {
                        let map = Arc::clone(&map);
                        let state = Arc::clone(&state);
                        let start_b = Arc::clone(&start_barrier);
                        let end_b = Arc::clone(&end_barrier);

                        s.spawn(move || {
                            start_b.wait();

                            B::run(map, state, thread_id);

                            end_b.wait();
                        });
                    }

                    start_barrier.wait();
                    let start = Instant::now();

                    end_barrier.wait();
                    total += start.elapsed();
                });
            }
            total
        });
    });
}

macro_rules! run_bench_for_all {
    ($c:expr, $bench_type:ident, $threads:expr) => {
        let name = format!(
            "{} (tc: {})",
            <$bench_type as Benchmark<StdMutexMap>>::name(),
            $threads
        );

        let mut group = $c.benchmark_group(name);
        run_bench::<StdMutexMap, $bench_type>(&mut group, $threads);
        run_bench::<PlMutexMap, $bench_type>(&mut group, $threads);
        run_bench::<StdRwLockMap, $bench_type>(&mut group, $threads);
        run_bench::<PlRwLockMap, $bench_type>(&mut group, $threads);
        run_bench::<AtomicSlotMap<DefaultKey, usize>, $bench_type>(&mut group, $threads);
        group.finish();
    };
}

struct ConcurrentInsertBenchmark;

impl<S: ConcurrentMap> Benchmark<S> for ConcurrentInsertBenchmark {
    type SharedState = ();

    fn name() -> &'static str {
        "Concurrent inserts"
    }

    fn setup(_map: &S) -> Self::SharedState {}

    fn run(map: Arc<S>, _state: Arc<Self::SharedState>, _thread_id: usize) {
        for i in 0..OPS_PER_THREAD {
            map.insert(i);
        }
    }
}

fn bench_concurrent_inserts(c: &mut Criterion) {
    let threads = std::thread::available_parallelism().unwrap().get();
    run_bench_for_all!(c, ConcurrentInsertBenchmark, 1);
    run_bench_for_all!(c, ConcurrentInsertBenchmark, threads);
    run_bench_for_all!(c, ConcurrentInsertBenchmark, threads * 2);
    run_bench_for_all!(c, ConcurrentInsertBenchmark, threads * 3);
}

struct ConcurrentReadsBenchmark;

impl<S: ConcurrentMap> Benchmark<S> for ConcurrentReadsBenchmark {
    type SharedState = Vec<S::Key>;

    fn name() -> &'static str {
        "Concurrent reads"
    }

    fn setup(map: &S) -> Self::SharedState {
        let mut keys = Vec::with_capacity(OPS_PER_THREAD);
        for i in 0..OPS_PER_THREAD {
            keys.push(map.insert(i));
        }
        keys
    }

    fn run(map: Arc<S>, keys: Arc<Self::SharedState>, _thread_id: usize) {
        for &k in keys.iter() {
            black_box(map.get(k));
        }
    }
}

fn bench_concurrent_reads(c: &mut Criterion) {
    let threads = std::thread::available_parallelism().unwrap().get();
    run_bench_for_all!(c, ConcurrentReadsBenchmark, 1);
    run_bench_for_all!(c, ConcurrentReadsBenchmark, threads);
    run_bench_for_all!(c, ConcurrentReadsBenchmark, threads * 2);
    run_bench_for_all!(c, ConcurrentReadsBenchmark, threads * 3);
}

struct MixedWorkloadBenchmark;

impl<S: ConcurrentMap> Benchmark<S> for MixedWorkloadBenchmark {
    type SharedState = Vec<S::Key>;

    fn name() -> &'static str {
        "Mixed workload"
    }

    fn setup(map: &S) -> Self::SharedState {
        let mut keys = Vec::with_capacity(OPS_PER_THREAD);
        for i in 0..OPS_PER_THREAD {
            keys.push(map.insert(i));
        }
        keys
    }

    fn run(map: Arc<S>, keys: Arc<Self::SharedState>, thread_id: usize) {
        for i in 0..OPS_PER_THREAD {
            if i % 10 == 0 {
                map.insert(thread_id * 1000 + i);
            } else {
                black_box(map.get(keys[i]));
            }
        }
    }
}

fn bench_concurrent_mixed(c: &mut Criterion) {
    let threads = std::thread::available_parallelism().unwrap().get();
    run_bench_for_all!(c, MixedWorkloadBenchmark, 1);
    run_bench_for_all!(c, MixedWorkloadBenchmark, threads);
    run_bench_for_all!(c, MixedWorkloadBenchmark, threads * 2);
    run_bench_for_all!(c, MixedWorkloadBenchmark, threads * 3);
}

criterion_group!(
    benches,
    bench_concurrent_inserts,
    bench_concurrent_reads,
    bench_concurrent_mixed
);
criterion_main!(benches);
