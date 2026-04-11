#![cfg(loom)]

use atomic_slotmap::AtomicSlotMap;
use loom::thread;

#[test]
fn test_concurrent_insert_and_remove() {
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
    });
}
