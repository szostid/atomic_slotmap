//! Basic slotmap tests.
//!
//! Singlethreaded, not for loom
#![cfg(not(loom))]
use quickcheck::quickcheck;
use std::collections::HashMap;

mod common;
use common::*;

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}

    assert_send_sync::<AtomicSlotMap<DefaultKey, u32>>();
    assert_send_sync::<SlotGuard<DefaultKey, u32>>();
    assert_send_sync::<OwningSlotGuard<DefaultKey, u32>>();
};

#[test]
fn check_drops() {
    let drops = DropCounter::<()>::default();

    {
        let sm = AtomicSlotMap::new();
        let mut sm_keys = Vec::new();
        for _ in 0..1000 {
            sm_keys.push(sm.insert(drops.clone()));
        }

        // Remove even keys (i.e. half of the keys)
        for i in (0..1000).filter(|i| i % 2 == 0) {
            sm.remove(sm_keys[i]);
        }

        // Should only have dropped 500 so far
        assert_eq!(drops.get(), 500);
    };

    // Now all original items should have been dropped exactly once.
    assert_eq!(drops.get(), 1000);
}

#[test]
fn check_drops_with_multiple_guards() {
    let drops = DropCounter::<()>::default();
    let sm = AtomicSlotMap::new();

    let key = sm.insert(drops.clone());

    let guard1 = sm.get(key).unwrap();
    let guard2 = sm.get(key).unwrap();

    // held alive by the slotmap itself
    assert_eq!(drops.get(), 0);

    drop(guard1);
    drop(guard2);

    // still held alive by the slotmap itself
    assert_eq!(drops.get(), 0);

    let guard1 = sm.get(key).unwrap();
    let guard2 = sm.get(key).unwrap();

    assert!(sm.remove(key));

    // held alive by guard 1 2
    assert_eq!(drops.get(), 0);

    drop(guard1);

    // held alive by guard 2
    assert_eq!(drops.get(), 0);

    drop(guard2);

    assert_eq!(drops.get(), 1);
}

#[test]
fn check_slot_reusability() {
    use slotmap::Key as _;

    // just to cover Default impl, should be equivalent to AtomicSlotMap::new
    let sm = AtomicSlotMap::default();

    let mut first_key = DefaultKey::null();

    assert_eq!(
        sm.try_insert_with_key(|key| {
            first_key = key;
            Err(())
        }),
        Err(())
    );

    assert!(sm.lossy_is_empty());

    // the closure shouldn't have provided a null key
    assert!(!first_key.is_null());

    // the slotmap must have resized to somehow acquire the key to provide for the closure
    let cap = sm.capacity();
    assert_ne!(cap, 0);

    let key = sm.insert(123);

    // the value should reuse the first key, and the slotmap shouldnt resize
    assert_eq!(key, first_key);
    assert_eq!(sm.get(key).as_deref(), Some(&123));
    assert_eq!(cap, sm.capacity());
    assert_eq!(sm.lossy_len(), 1);

    sm.remove(key);

    assert!(sm.lossy_is_empty());
    assert!(sm.get(key).is_none());

    // the value should reuse the first key, and the slotmap shouldnt resize
    let key = sm.insert(456);
    assert_eq!(sm.get(key).as_deref(), Some(&456));
    assert_eq!(cap, sm.capacity());
    assert_eq!(sm.lossy_len(), 1);
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
fn check_lossy_methods() {
    // on singlethreaded (i.e. exclusive access), the lossy iter should be predictable
    let sm = AtomicSlotMap::with_capacity(4);
    let cap1 = sm.capacity();
    assert!(cap1 >= 4);

    let _k1 = sm.insert(10);
    let _k2 = sm.insert(20);
    let _k3 = sm.insert(30);

    assert_eq!(sm.lossy_iter().count(), 3);
    assert!(!sm.lossy_is_empty());

    // shouldn't have resized
    assert_eq!(sm.capacity(), cap1);

    sm.reserve(50);
    let cap2 = sm.capacity();
    assert_ne!(cap2, cap1);
    assert!(cap2 >= 53);

    for i in 0..50 {
        let _ = sm.insert(i);
    }

    assert_eq!(sm.capacity(), cap2);
}

#[test]
fn test_guards_and_contains() {
    // just to cover with_key, it shouldnt make a difference
    let sm = Arc::new(AtomicSlotMap::with_key());
    let k1 = sm.insert(42);

    assert!(sm.contains_key(k1));
    assert!(!sm.contains_key(DefaultKey::null()));

    {
        let owning_guard = AtomicSlotMap::get_owning(&sm, k1).unwrap();
        assert_eq!(*owning_guard, 42);
        assert_eq!(owning_guard.as_ref(), &42);
        assert_eq!(owning_guard.key(), k1);

        let owning_clone = owning_guard.clone();
        assert_eq!(*owning_clone, 42);
        assert_eq!(owning_clone.as_ref(), &42);
        assert_eq!(owning_clone.key(), k1);

        let normal_guard = sm.get(k1).unwrap();
        assert_eq!(*normal_guard, 42);
        assert_eq!(normal_guard.as_ref(), &42);
        assert_eq!(normal_guard.key(), k1);
    }

    sm.remove(k1);

    assert!(!sm.contains_key(k1));
    assert!(sm.get(k1).is_none());
    assert!(AtomicSlotMap::get_owning(&sm, k1).is_none());
}

#[test]
fn test_fmt() {
    // this is kinda pointless and more for the sake of coverage and MIRI checking than
    // actually checking the correctness of the method, because this uses the same exact
    // approach as .as_ref() / deref and then just the formatting impl of T
    //
    // note that the slotmap itself doesnt have a format impl because of TOCTOU (a method
    // like .lossy_fmt could exist but then it wouldnt integrate into the formatting
    // macros so i feel like its pointless)
    let sm = Arc::new(AtomicSlotMap::new());

    let key = sm.insert("hi");

    let sm_thread1 = Arc::clone(&sm);
    let t1 = thread::spawn(move || {
        let guard = sm_thread1.get(key).unwrap();
        assert_eq!(format!("{guard}"), "hi");
        assert_eq!(format!("{guard:?}"), "\"hi\"");
    });

    let sm_thread2 = Arc::clone(&sm);
    let t2 = thread::spawn(move || {
        let guard = sm_thread2.get_owning(key).unwrap();
        assert_eq!(format!("{guard}"), "hi");
        assert_eq!(format!("{guard:?}"), "\"hi\"");
    });

    t1.join().unwrap();
    t2.join().unwrap();
}
