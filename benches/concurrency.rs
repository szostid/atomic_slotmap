use atomic_slotmap::AtomicSlotMap;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use slotmap::SlotMap;
use std::sync::{Arc, Mutex, RwLock};
use std::thread;

const THREADS: usize = 8;
const OPS_PER_THREAD: usize = 1000;

fn bench_concurrent_inserts(c: &mut Criterion) {
    let mut group = c.benchmark_group("Concurrent Inserts");

    group.bench_function("AtomicSlotMap", |b| {
        b.iter(|| {
            let map = Arc::new(AtomicSlotMap::new());
            thread::scope(|s| {
                for _ in 0..THREADS {
                    let map = Arc::clone(&map);
                    s.spawn(move || {
                        for i in 0..OPS_PER_THREAD {
                            map.insert(i);
                        }
                    });
                }
            });
        });
    });

    group.bench_function("Mutex<SlotMap>", |b| {
        b.iter(|| {
            let map = Arc::new(Mutex::new(SlotMap::new()));
            thread::scope(|s| {
                for _ in 0..THREADS {
                    let map = Arc::clone(&map);
                    s.spawn(move || {
                        for i in 0..OPS_PER_THREAD {
                            map.lock().unwrap().insert(i);
                        }
                    });
                }
            });
        });
    });

    group.bench_function("RwLock<SlotMap>", |b| {
        b.iter(|| {
            let map = Arc::new(RwLock::new(SlotMap::new()));
            thread::scope(|s| {
                for _ in 0..THREADS {
                    let map = Arc::clone(&map);
                    s.spawn(move || {
                        for i in 0..OPS_PER_THREAD {
                            map.write().unwrap().insert(i);
                        }
                    });
                }
            });
        });
    });

    group.finish();
}

fn bench_uncontended_reads(c: &mut Criterion) {
    let mut group = c.benchmark_group("Uncontended Reads");

    // Pre-populate maps
    let atomic_map = AtomicSlotMap::new();
    let mut atomic_keys = Vec::new();
    for i in 0..OPS_PER_THREAD {
        atomic_keys.push(atomic_map.insert(i));
    }

    // Pre-populate maps
    let mut std_map = SlotMap::new();
    let mut std_keys = Vec::new();
    for i in 0..OPS_PER_THREAD {
        std_keys.push(std_map.insert(i));
    }

    group.bench_function("AtomicSlotMap", |b| {
        b.iter(|| {
            for &k in &atomic_keys {
                black_box(atomic_map.get(k));
            }
        });
    });

    let mutex_map = Mutex::new(std_map.clone());
    group.bench_function("Mutex<SlotMap>", |b| {
        b.iter(|| {
            let guard = mutex_map.lock().unwrap();
            for &k in &std_keys {
                black_box(guard.get(k));
            }
        });
    });

    let rwlock_map = RwLock::new(std_map);
    group.bench_function("RwLock<SlotMap>", |b| {
        b.iter(|| {
            let guard = rwlock_map.read().unwrap();
            for &k in &std_keys {
                black_box(guard.get(k));
            }
        });
    });

    group.finish();
}

fn bench_concurrent_reads(c: &mut Criterion) {
    let mut group = c.benchmark_group("Concurrent Reads");

    // Pre-populate maps
    let atomic_map = Arc::new(AtomicSlotMap::new());
    let mut atomic_keys = Vec::new();
    for i in 0..OPS_PER_THREAD {
        atomic_keys.push(atomic_map.insert(i));
    }
    let atomic_keys = Arc::new(atomic_keys);

    let mut std_map = SlotMap::new();
    let mut std_keys = Vec::new();
    for i in 0..OPS_PER_THREAD {
        std_keys.push(std_map.insert(i));
    }
    let std_keys = Arc::new(std_keys);

    group.bench_function("AtomicSlotMap", |b| {
        b.iter(|| {
            thread::scope(|s| {
                for _ in 0..THREADS {
                    let map = Arc::clone(&atomic_map);
                    let keys = Arc::clone(&atomic_keys);
                    s.spawn(move || {
                        for &k in keys.iter() {
                            black_box(map.get(k));
                        }
                    });
                }
            });
        });
    });

    let mutex_map = Arc::new(Mutex::new(std_map.clone()));
    group.bench_function("Mutex<SlotMap>", |b| {
        b.iter(|| {
            thread::scope(|s| {
                for _ in 0..THREADS {
                    let map = Arc::clone(&mutex_map);
                    let keys = Arc::clone(&std_keys);
                    s.spawn(move || {
                        for &k in keys.iter() {
                            black_box(map.lock().unwrap().get(k));
                        }
                    });
                }
            });
        });
    });

    let rwlock_map = Arc::new(RwLock::new(std_map));
    group.bench_function("RwLock<SlotMap>", |b| {
        b.iter(|| {
            thread::scope(|s| {
                for _ in 0..THREADS {
                    let map = Arc::clone(&rwlock_map);
                    let keys = Arc::clone(&std_keys);
                    s.spawn(move || {
                        for &k in keys.iter() {
                            black_box(map.read().unwrap().get(k));
                        }
                    });
                }
            });
        });
    });

    group.finish();
}

fn bench_concurrent_mixed(c: &mut Criterion) {
    let mut group = c.benchmark_group("Mixed Workload (90% Read, 10% Write)");

    // Pre-populate maps
    let atomic_map = Arc::new(AtomicSlotMap::new());
    let mut atomic_keys = Vec::new();
    for i in 0..OPS_PER_THREAD {
        atomic_keys.push(atomic_map.insert(i));
    }
    let atomic_keys = Arc::new(atomic_keys);

    let mut std_map = SlotMap::new();
    let mut std_keys = Vec::new();
    for i in 0..OPS_PER_THREAD {
        std_keys.push(std_map.insert(i));
    }
    let std_keys = Arc::new(std_keys);

    group.bench_function("AtomicSlotMap", |b| {
        b.iter(|| {
            thread::scope(|s| {
                for t in 0..THREADS {
                    let map = Arc::clone(&atomic_map);
                    let keys = Arc::clone(&atomic_keys);
                    s.spawn(move || {
                        for i in 0..OPS_PER_THREAD {
                            if i % 10 == 0 {
                                map.insert(t * 1000 + i);
                            } else {
                                black_box(map.get(keys[i]));
                            }
                        }
                    });
                }
            });
        });
    });

    let rwlock_map = Arc::new(RwLock::new(std_map));
    group.bench_function("RwLock<SlotMap>", |b| {
        b.iter(|| {
            thread::scope(|s| {
                for t in 0..THREADS {
                    let map = Arc::clone(&rwlock_map);
                    let keys = Arc::clone(&std_keys);
                    s.spawn(move || {
                        for i in 0..OPS_PER_THREAD {
                            if i % 10 == 0 {
                                map.write().unwrap().insert(t * 1000 + i);
                            } else {
                                black_box(map.read().unwrap().get(keys[i]));
                            }
                        }
                    });
                }
            });
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_uncontended_reads,
    bench_concurrent_inserts,
    bench_concurrent_reads,
    bench_concurrent_mixed
);
criterion_main!(benches);
