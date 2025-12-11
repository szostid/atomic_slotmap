#![no_main]
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use atomic_slotmap::{AtomicSlotMap, OwningSlotGuard, SlotGuard};
use libfuzzer_sys::arbitrary::{self, Arbitrary};
use libfuzzer_sys::fuzz_target;
use slotmap::{DefaultKey, Key, KeyData};

#[derive(Arbitrary, Debug)]
pub struct Target {
    pub ctor: Constructor,
    pub ops: [Vec<Op>; 4],
    pub dtor: Destructor,
}

#[derive(Arbitrary, Debug)]
pub enum Constructor {
    New,
    WithCapacity(u8),
}

#[derive(Arbitrary, Debug)]
pub enum Index {
    Shared(usize),
    Private(usize),
}

#[derive(Arbitrary, Debug)]
pub enum Op {
    Reserve(u8),

    /// Inserts a new value (tracked by private keys)
    Insert(u32),
    /// Inserts a new value (tracked by private keys)
    InsertWithKey(u32),

    /// Publishes a private key into shared keys
    Publish(usize),

    /// Removes a key from the slotmap.
    Remove(Index),
    /// Removes a key from the slotmap but keeps it in the graveyard.
    ///
    /// There's a flag that signals whether the generation of the
    /// key should be mutated. A
    RemoveToGraveyard(Index, bool),
    /// Retrieves a key from the slotmap and compares it to
    /// the known value.
    Retrieve(Index),

    /// Retains a key that exists at the provided index. A
    /// guard will be held for that key keeping it alive
    /// until a [`Op::RemoveRetained`] removes it.
    AddRetained(Index),
    /// Reads a retained key and compares its value.
    ReadRetained(usize),
    /// Removes a retained key.
    RemoveRetained(usize),

    /// Retains a key that exists at the provided index. A
    /// guard will be held for that key keeping it alive
    /// until a [`Op::RemoveRetained`] removes it.
    AddRetainedArc(Index),
    /// Reads a retained key and compares its value.
    ReadRetainedArc(usize),
    /// Removes a retained key.
    RemoveRetainedArc(usize),

    /// Checks that a key that came from the graveyard is still
    /// invalid.
    CheckGraveyard(usize),

    /// Checks a forged key, larger than 0x7000_0000, it should
    /// be physically impossible for this many keys to exist in
    /// the atomicslotmap
    RetrieveForged(usize),
}

macro_rules! constrain_idx {
    ($array:expr, $idx:expr) => {{
        if $array.is_empty() {
            continue;
        }

        $idx % $array.len()
    }};
}

macro_rules! access_idx {
    ($index:expr, $private:expr, $shared:expr, |$idx:ident, $keys:ident| $body:block) => {
        match $index {
            Index::Private(i) => {
                if !$private.is_empty() {
                    let $idx = i % $private.len();
                    let $keys = &mut $private;
                    $body
                }
            }
            Index::Shared(i) => {
                let mut lock = $shared.lock().unwrap();
                if !lock.is_empty() {
                    let $idx = i % lock.len();
                    let $keys = &mut *lock;
                    $body
                }
            }
        }
    };
}

#[derive(Clone, Copy)]
struct KeyInfo(DefaultKey, u32);

struct RetainedKey<'a> {
    info: KeyInfo,
    guard: SlotGuard<'a, DefaultKey, Value>,
}

struct ArcRetainedKey {
    info: KeyInfo,
    guard: OwningSlotGuard<DefaultKey, Value>,
}

static EXISTING_VALUE_COUNTER: AtomicU32 = AtomicU32::new(0);

struct Value(u32);

impl Value {
    pub fn new(v: u32) -> Self {
        EXISTING_VALUE_COUNTER.fetch_add(1, Ordering::Release);
        Self(v)
    }
}

impl Drop for Value {
    fn drop(&mut self) {
        EXISTING_VALUE_COUNTER.fetch_sub(1, Ordering::Release);
    }
}

#[derive(Arbitrary, Debug)]
pub enum Destructor {
    LetDrop,
}

fn fuzz(data: Target) {
    let map = match data.ctor {
        Constructor::New => AtomicSlotMap::new(),
        Constructor::WithCapacity(n) => AtomicSlotMap::with_capacity(n as usize),
    };
    let map = Arc::new(map);
    let shared_keys = Arc::new(Mutex::new(Vec::new()));

    let mut handles = Vec::new();

    for thread_ops in data.ops {
        let shared_keys = Arc::clone(&shared_keys);
        let map = Arc::clone(&map);

        handles.push(thread::spawn(move || {
            let mut private_keys = Vec::new();
            let mut retained_keys = Vec::new();
            let mut atomic_retained_keys = Vec::new();
            let mut graveyard = Vec::new();

            for op in thread_ops {
                match op {
                    Op::Reserve(additional) => {
                        map.reserve(additional as usize);
                    }
                    Op::Insert(value) => {
                        let key = map.insert(Value::new(value));
                        private_keys.push(KeyInfo(key, value));
                    }
                    Op::InsertWithKey(value) => {
                        let key = map.insert_with_key(|_| Value::new(value));
                        private_keys.push(KeyInfo(key, value));
                    }

                    Op::Publish(idx) => {
                        let idx = constrain_idx!(private_keys, idx);
                        let info = private_keys.remove(idx);
                        shared_keys.lock().unwrap().push(info);
                    }

                    Op::AddRetained(idx) => {
                        access_idx!(idx, private_keys, shared_keys, |idx, keys| {
                            let info = keys[idx];

                            let guard = map.get(info.0).unwrap();

                            assert_eq!(guard.0, info.1);

                            retained_keys.push(RetainedKey { info, guard });
                        });
                    }
                    Op::ReadRetained(idx) => {
                        let idx = constrain_idx!(retained_keys, idx);
                        let info = &retained_keys[idx];

                        assert_eq!(info.guard.0, info.info.1);
                    }
                    Op::RemoveRetained(idx) => {
                        let idx = constrain_idx!(retained_keys, idx);
                        let info = retained_keys.remove(idx);

                        assert_eq!(info.guard.0, info.info.1);
                    }

                    Op::AddRetainedArc(idx) => {
                        access_idx!(idx, private_keys, shared_keys, |idx, keys| {
                            let info = keys[idx];

                            let guard = map.get_owning(info.0).unwrap();

                            assert_eq!(guard.0, info.1);

                            atomic_retained_keys.push(ArcRetainedKey { info, guard });
                        });
                    }

                    Op::ReadRetainedArc(idx) => {
                        let idx = constrain_idx!(atomic_retained_keys, idx);
                        let info = &atomic_retained_keys[idx];

                        assert_eq!(info.guard.0, info.info.1);
                    }
                    Op::RemoveRetainedArc(idx) => {
                        let idx = constrain_idx!(atomic_retained_keys, idx);
                        let info = atomic_retained_keys.remove(idx);

                        assert_eq!(info.guard.0, info.info.1);
                    }

                    Op::RemoveToGraveyard(idx, remove_gen) => {
                        access_idx!(idx, private_keys, shared_keys, |idx, keys| {
                            let mut info = keys.remove(idx);
                            assert!(map.remove(info.0));

                            let key = info.0;
                            // the data is packed into LSB 32 bytes of idx and then MSB 32 bytes of gen
                            //
                            // the handle has to have a non-zero index to not cause unsafety so we do need
                            // to check that (otherwise idx=1 would become idx=0 and cause unsafety)
                            let key_ffi = key.data().as_ffi();
                            let is_key_gen_zero = key_ffi & (0xFFFF_FFFF_0000_0000) == 0;

                            let key_ffi = if !remove_gen || is_key_gen_zero {
                                key_ffi
                            } else {
                                key_ffi & 0xFFFF_FFFE_FFFF_FFFF
                            };

                            let key = DefaultKey::from(KeyData::from_ffi(key_ffi));

                            info.0 = key;

                            graveyard.push(info);
                        });
                    }

                    Op::Remove(idx) => {
                        access_idx!(idx, private_keys, shared_keys, |idx, keys| {
                            let info = keys.remove(idx);
                            assert!(map.remove(info.0));
                        });
                    }

                    Op::Retrieve(idx) => {
                        access_idx!(idx, private_keys, shared_keys, |idx, keys| {
                            let info = &keys[idx];

                            let guard = map.get(info.0).unwrap();

                            assert_eq!(guard.0, info.1);
                        });
                    }

                    Op::CheckGraveyard(idx) => {
                        let idx = constrain_idx!(graveyard, idx);

                        assert!(map.get(graveyard[idx].0).is_none());
                        assert!(!map.contains_key(graveyard[idx].0));
                        assert!(!map.remove(graveyard[idx].0));
                    }

                    Op::RetrieveForged(raw_offset) => {
                        let key_idx = (0x7000_0000 + raw_offset % 10_000) as u64;
                        let key_gen = raw_offset as u64;

                        let key_ffi = (key_gen << 32) | key_idx;

                        let key = DefaultKey::from(KeyData::from_ffi(key_ffi));

                        assert!(map.get(key).is_none());
                        assert!(!map.contains_key(key));
                        assert!(!map.remove(key));
                    }
                }
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }

    match data.dtor {
        Destructor::LetDrop => {
            drop(map);
        }
    }

    assert_eq!(EXISTING_VALUE_COUNTER.load(Ordering::Acquire), 0);
}

fuzz_target!(|data: Target| fuzz(data));
