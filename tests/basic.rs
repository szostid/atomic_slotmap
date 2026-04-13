#![cfg(not(loom))]

use quickcheck::quickcheck;
use std::collections::HashMap;
use std::sync::Arc;
use std::thread::spawn;
use std::time::Duration;

use atomic_slotmap::*;
use slotmap::*;

const _: () = {
    const fn f<T: Send + Sync>() {}

    f::<AtomicSlotMap<DefaultKey, u32>>();
    f::<SlotGuard<DefaultKey, u32>>();
    f::<OwningSlotGuard<DefaultKey, u32>>();
};

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

#[test]
fn try_insert_err_keeps_fresh_slot_reusable() {
    use slotmap::Key as _;

    let sm = AtomicSlotMap::new();

    assert!(sm.try_insert_with_key::<_, ()>(|_| Err(())).is_err());
    assert_eq!(sm.lossy_len(), 0);

    let key = sm.insert(123_u32);

    // The freshly allocated slot should be recycled after the failed insert.
    let idx = key.data().as_ffi() as u32;
    assert_eq!(idx, 0);
    assert_eq!(sm.get(key).as_deref(), Some(&123));
}

#[test]
fn try_insert_err_keeps_reused_slot_reusable() {
    use slotmap::Key as _;

    let sm = AtomicSlotMap::new();
    let key = sm.insert(1_u32);
    let idx = key.data().as_ffi() as u32;

    assert!(sm.remove(key));
    assert!(sm.try_insert_with_key::<_, ()>(|_| Err(())).is_err());
    assert_eq!(sm.lossy_len(), 0);

    let key2 = sm.insert(2_u32);
    let idx2 = key2.data().as_ffi() as u32;

    assert_eq!(idx2, idx);
    assert_eq!(sm.get(key2).as_deref(), Some(&2));
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
