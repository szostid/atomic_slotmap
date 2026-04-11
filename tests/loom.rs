#![cfg(loom)]

use atomic_slotmap::AtomicSlotMap;
use loom::thread;

#[test]
fn test_concurrent_insert() {
    loom::model(|| {
        let sm = loom::sync::Arc::new(AtomicSlotMap::new());

        let sm_thread1 = sm.clone();
        let t1 = thread::spawn(move || {
            sm_thread1.insert(42);
        });

        let sm_thread2 = sm.clone();
        let t2 = thread::spawn(move || {
            sm_thread2.insert(24);
        });

        t1.join().unwrap();
        t2.join().unwrap();

        assert_eq!(sm.len(), 2);
    });
}

#[test]
fn test_concurrent_remove() {
    loom::model(|| {
        let sm = loom::sync::Arc::new(AtomicSlotMap::new());
        let key = sm.insert(100);

        let sm_thread1 = sm.clone();
        let t1 = thread::spawn(move || sm_thread1.remove(key));

        let sm_thread2 = sm.clone();
        let t2 = thread::spawn(move || sm_thread2.remove(key));

        let r1 = t1.join().unwrap();
        let r2 = t2.join().unwrap();

        assert!(r1 ^ r2); // Exactly one removal should succeed.
        assert_eq!(sm.len(), 0);
    });
}

#[test]
fn test_concurrent_get_and_remove() {
    loom::model(|| {
        let sm = loom::sync::Arc::new(AtomicSlotMap::new());
        let key = sm.insert(100);

        let sm_thread1 = sm.clone();
        let t1 = thread::spawn(move || {
            let guard = sm_thread1.get(key);
            if let Some(val) = guard {
                assert_eq!(*val, 100);
            }
        });

        let sm_thread2 = sm.clone();
        let t2 = thread::spawn(move || {
            sm_thread2.remove(key);
        });

        t1.join().unwrap();
        t2.join().unwrap();

        assert_eq!(sm.len(), 0);
    });
}

#[test]
fn test_aba_free_list() {
    loom::model(|| {
        let sm = loom::sync::Arc::new(AtomicSlotMap::new());
        let key1 = sm.insert(1);
        let key2 = sm.insert(2);

        sm.remove(key1);
        sm.remove(key2);

        let sm_thread1 = sm.clone();
        let t1 = thread::spawn(move || {
            sm_thread1.insert(3);
        });

        let sm_thread2 = sm.clone();
        let t2 = thread::spawn(move || {
            sm_thread2.insert(4);
        });

        t1.join().unwrap();
        t2.join().unwrap();

        assert_eq!(sm.len(), 2);
    });
}

#[test]
fn test_hard_guard_drop_interference() {
    let mut builder = loom::model::Builder::new();
    builder.preemption_bound = Some(3);
    builder.check(|| {
        let sm = loom::sync::Arc::new(AtomicSlotMap::new());
        let key = sm.insert(77);

        let sm_thread1 = sm.clone();
        let t1 = loom::thread::spawn(move || {
            // Thread 1 acquires a guard and immediately drops it
            let _guard = AtomicSlotMap::get_owning(&sm_thread1, key);
        });

        let sm_thread2 = sm.clone();
        let t2 = loom::thread::spawn(move || {
            // Thread 2 attempts to remove the key
            sm_thread2.remove(key);
        });

        let sm_thread3 = sm.clone();
        let t3 = loom::thread::spawn(move || {
            // Thread 3 attempts to read while drop/remove fight
            let _guard = sm_thread3.get(key);
        });

        t1.join().unwrap();
        t2.join().unwrap();
        t3.join().unwrap();

        // Eventually, the element must be fully freed.
        assert_eq!(sm.len(), 0);
    });
}
