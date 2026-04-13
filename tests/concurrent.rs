//! Basic concurrency tests, for both loom, miri and normal testing
mod common;
use common::*;

#[test]
fn test_concurrent_insert() {
    model(None, || {
        let counter = DropCounter::<u32>::default();

        let sm = Arc::new(AtomicSlotMap::new());

        let sm_thread1 = sm.clone();
        let mut v1 = counter.clone();
        let t1 = thread::spawn(move || {
            *v1 = 42;
            sm_thread1.insert(v1)
        });

        let sm_thread2 = sm.clone();
        let mut v2 = counter.clone();
        let t2 = thread::spawn(move || {
            *v2 = 24;
            sm_thread2.insert(v2)
        });

        let k1 = t1.join().unwrap();
        let k2 = t2.join().unwrap();

        assert_eq!(sm.lossy_len(), 2);
        assert_eq!(sm.get(k1).map(|v| **v), Some(42));
        assert_eq!(AtomicSlotMap::get_owning(&sm, k2).map(|v| **v), Some(24));

        drop(sm);

        assert_eq!(counter.get(), 2)
    });
}

#[test]
fn test_concurrent_remove() {
    model(None, || {
        let sm = Arc::new(AtomicSlotMap::new());
        let key = sm.insert(100);

        let sm_thread1 = sm.clone();
        let t1 = thread::spawn(move || sm_thread1.remove(key));

        let sm_thread2 = sm.clone();
        let t2 = thread::spawn(move || sm_thread2.remove(key));

        let r1 = t1.join().unwrap();
        let r2 = t2.join().unwrap();

        assert!(r1 ^ r2); // Exactly one removal should succeed.
        assert_eq!(sm.lossy_len(), 0);
    });
}

#[test]
fn test_concurrent_get_and_remove() {
    model(None, || {
        let sm = Arc::new(AtomicSlotMap::new());
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

        assert_eq!(sm.lossy_len(), 0);
    });
}

#[test]
fn test_aba_free_list() {
    model(None, || {
        let sm = Arc::new(AtomicSlotMap::new());
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

        assert_eq!(sm.lossy_len(), 2);
    });
}

#[test]
fn test_hard_guard_drop_interference() {
    model(Some(2), || {
        let sm = Arc::new(AtomicSlotMap::new());
        let key = sm.insert(77);

        // Thread 1 acquires a guard and immediately drops it
        // Thread 2 attempts to remove the key
        // Thread 3 attempts to read owning while the other threads do their stuff

        let sm_thread1 = sm.clone();
        let t1 = thread::spawn(move || {
            let _guard = AtomicSlotMap::get_owning(&sm_thread1, key);
        });

        let sm_thread2 = sm.clone();
        let t2 = thread::spawn(move || {
            sm_thread2.remove(key);
        });

        let sm_thread3 = sm.clone();
        let t3 = thread::spawn(move || {
            let guard = sm_thread3.get(key);

            // this will fail if thread 2 happens to remove the key before
            // we read it here, but that path should be explored by loom anyways
            if let Some(guard) = guard {
                assert_eq!(*guard, 77);
                assert_eq!(guard.as_ref(), &77);
                assert_eq!(guard.key(), key);
            }
        });

        t1.join().unwrap();
        t2.join().unwrap();
        t3.join().unwrap();

        // Eventually, the element must be fully freed.
        assert_eq!(sm.lossy_len(), 0);
    });
}

#[test]
fn test_guard_send() {
    model(Some(2), || {
        let sm = Arc::new(AtomicSlotMap::new());
        let key = sm.insert(77);

        // Thread 1 acquires a guard and immediately drops it
        // Thread 2 attempts to read owning, clone the key, and then read it from another thread

        let sm_thread1 = sm.clone();
        let t1 = thread::spawn(move || {
            assert!(sm_thread1.contains_key(key));
            let _guard = AtomicSlotMap::get_owning(&sm_thread1, key);
        });

        let sm_thread2 = sm.clone();
        let t2 = thread::spawn(move || AtomicSlotMap::get_owning(&sm_thread2, key).clone());

        t1.join().unwrap();
        let guard = t2.join().unwrap().unwrap();

        assert_eq!(*guard, 77);
        assert_eq!(guard.key(), key);
        assert_eq!(guard.as_ref(), &77);

        assert_eq!(sm.lossy_len(), 1);
        assert!(sm.contains_key(key));

        sm.remove(key);

        assert!(sm.get(key).is_none());
        assert!(!sm.remove(key));
        assert!(!sm.contains_key(key));
        assert_eq!(sm.lossy_len(), 0);
    });
}
