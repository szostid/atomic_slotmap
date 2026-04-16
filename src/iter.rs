use crate::atomic::Ordering;
use crate::util::KeyDataRead as _;
use crate::{AtomicSlotMap, SlotGuard};
use slotmap::{Key, KeyData};

/// An iterator over the contents of the slotmap
///
/// Due to the lockfree nature of the atomic slot map,
/// the iteration is lossy. That is, it's not guaranteed
/// to display the real time state of items of the map.
///
/// This is just equivalent to generating every valid
/// key for any slot and trying to lock it. This means
/// that during the iteration, free slots that were
/// already checked might become used up and therefore
/// they will be skipped during iteration.
#[allow(missing_debug_implementations)]
pub struct LossyIter<'a, K: Key, V> {
    map: &'a AtomicSlotMap<K, V>,
    current_idx: u32,
}

impl<'a, K: Key, V> LossyIter<'a, K, V> {
    pub(crate) fn new(map: &'a AtomicSlotMap<K, V>) -> Self {
        Self {
            map,
            current_idx: 0,
        }
    }
}

impl<'a, K: Key, V> Iterator for LossyIter<'a, K, V>
where
    K: From<KeyData>,
{
    type Item = (K, SlotGuard<'a, K, V>);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let idx = self.current_idx;

            if idx >= self.map.slots.len() {
                return None;
            }

            self.current_idx += 1;

            let slot = match self.map.slots.get(idx) {
                Some(s) => s,
                None => continue,
            };

            let version = slot.version.load(Ordering::Acquire);

            // if the version is even then the slot is unoccupied and
            // there's no point in checking
            if version % 2 != 0 {
                let key: K = KeyData::new(idx, version).into();

                if let Some(guard) = self.map.get(key) {
                    return Some((key, guard));
                }
            }
        }
    }
}
