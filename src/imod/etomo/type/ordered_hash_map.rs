//! `IMOD/Etomo/src/etomo/type/OrderedHashMap.java`.
//!
//! Map which also contains an optional array which may be used to order the values
//! with a separate ordinal that functions as an index.  A value with an identical
//! ordinal will replace the existing element in the array list.
//!
//! The Java values are object references; here they are `Clone` handles (`Arc`s), and
//! the map is a `BTreeMap` (Java's `HashMap` iteration order is not observable through
//! any caller).

use std::collections::BTreeMap;

/// Java `public final class OrderedHashMap<K, V>`.
pub struct OrderedHashMap<K: Ord, V: Clone> {
    /// Java private final `map`.
    map: BTreeMap<K, V>,
    /// Java private `array`, initially null.
    array: Option<Array<V>>,
}

impl<K: Ord, V: Clone> Default for OrderedHashMap<K, V> {
    fn default() -> Self {
        OrderedHashMap {
            map: BTreeMap::new(),
            array: None,
        }
    }
}

impl<K: Ord, V: Clone> OrderedHashMap<K, V> {
    /// Java's implicit constructor.
    pub fn new() -> Self {
        Self::default()
    }

    /// Java `clear()`.  Clears the map and the array.
    pub fn clear(&mut self) {
        self.map.clear();
        if let Some(array) = &mut self.array {
            array.clear();
        }
    }

    /// Java `put(K, V)`.  Add to map without adding to the array.  The array should be
    /// considered invalid after this function has been called, and before clear() has
    /// been called.
    pub fn put(&mut self, key: K, value: V) {
        self.map.insert(key, value);
    }

    /// Java `put(int, K, V)`.  Adds an element and also placed the element in the array
    /// to create an ordered list.
    pub fn put_ordinal(&mut self, ordinal: i32, key: K, value: V) {
        self.map.insert(key, value.clone());
        if self.array.is_none() {
            self.array = Some(Array::new());
        }
        self.array.as_mut().unwrap().add(ordinal, value);
    }

    /// Java `get(String)`.
    pub fn get<Q: Ord + ?Sized>(&self, key: &Q) -> Option<V>
    where
        K: std::borrow::Borrow<Q>,
    {
        self.map.get(key).cloned()
    }

    /// Java `values()`.  Retrieves all values in an unspecified order.
    pub fn values(&self) -> Vec<V> {
        self.map.values().cloned().collect()
    }

    /// Java `orderedValues()`.  Retrieves all values in order by ordinal.  Will return
    /// null if the ordinal was never used.
    pub fn ordered_values(&self) -> Option<&dyn ReadOnlyArray<V>> {
        self.array.as_ref().map(|array| array as &dyn ReadOnlyArray<V>)
    }
}

/// Java private static final nested `Array<V> implements ReadOnlyArray`.
struct Array<V> {
    /// Java private final `array`.
    array: Vec<Option<V>>,
}

impl<V> Array<V> {
    fn new() -> Array<V> {
        Array { array: Vec::new() }
    }

    /// Java private `clear()`.
    fn clear(&mut self) {
        self.array.clear();
    }

    /// Java private `add(int, V)`.
    fn add(&mut self, ordinal: i32, value: V) {
        // Java `ArrayList.add(int, V)` and `remove(int)` throw for a negative ordinal;
        // a negative row number is never stored.
        let Ok(ordinal) = usize::try_from(ordinal) else {
            return;
        };
        if self.array.len() <= ordinal {
            for _ in self.array.len()..ordinal + 1 {
                self.array.push(None);
            }
        }
        // Add(int,V) doesn't work like a primative array, it shoves existing elements
        // forwards to make a new space. Remove whatever is in the location of the ordinal
        // so as to create an ordered list.
        self.array.remove(ordinal);
        self.array.insert(ordinal, Some(value));
    }
}

/// Java `public static interface ReadOnlyArray<V>`.
pub trait ReadOnlyArray<V> {
    /// Java `size()`.
    fn size(&self) -> i32;
    /// Java `get(int)`: null where no element was put.
    fn get(&self, index: i32) -> Option<V>;
}

impl<V: Clone> ReadOnlyArray<V> for Array<V> {
    fn size(&self) -> i32 {
        self.array.len() as i32
    }

    fn get(&self, index: i32) -> Option<V> {
        self.array.get(index as usize).cloned().flatten()
    }
}
