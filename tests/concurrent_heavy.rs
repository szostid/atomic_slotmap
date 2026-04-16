//! Heavy concurrency tests, for miri and normal testing.
//!
//! Too heavy for loom
#![cfg(not(loom))]
mod common;
use common::*;

use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::sleep;
use std::time::Duration;

#[test]
fn test_multithreaded() {
    // tests multiple threads adding and removing elements into the slotmap and verifying that
    // they have correct values. this test does not modify correct dropping of elements.
    let sm = Arc::new(AtomicSlotMap::<_, u32>::new());

    let mut threads = Vec::with_capacity(10);

    #[allow(clippy::needless_range_loop)]
    for _ in 0..10 {
        let sm = Arc::clone(&sm);

        threads.push(thread::spawn(move || {
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

        threads.push(thread::spawn(move || {
            let mut keys = [DefaultKey::null(); 100];

            for i in 0..100 {
                keys[i] = sm.insert_with_key(|_| {
                    sleep(Duration::from_millis(1));
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
                    sleep(Duration::from_millis(1));
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

#[test]
fn test_multithreaded_closure_insertion_with_interference() {
    // this adds interference to the insertion test by adding another thread that
    // constantly inserts and removes keys
    let sm = Arc::new(AtomicSlotMap::<_, u32>::new());

    let mut threads = Vec::with_capacity(10);

    let interference_running = Arc::new(AtomicBool::new(true));
    let sm_interference = Arc::clone(&sm);
    let interference_running_c = Arc::clone(&interference_running);
    let interference_thread = thread::spawn(move || {
        let mut i = 0;

        while interference_running.load(Ordering::Relaxed) {
            let key = sm_interference.insert(i);
            assert!(sm_interference.contains_key(key));
            assert_eq!(*sm_interference.get(key).unwrap(), i);
            sm_interference.remove(key);
            i += 1;
        }
    });

    #[allow(clippy::needless_range_loop)]
    for _ in 0..10 {
        let sm = Arc::clone(&sm);

        threads.push(thread::spawn(move || {
            let mut keys = [DefaultKey::null(); 100];

            for i in 0..100 {
                keys[i] = sm.insert_with_key(|_| {
                    sleep(Duration::from_millis(1));
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
                    sleep(Duration::from_millis(1));
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

    interference_running_c.store(false, Ordering::Relaxed);
    interference_thread.join().unwrap();
}
