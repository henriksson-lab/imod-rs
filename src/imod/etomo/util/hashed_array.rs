//! `IMOD/Etomo/src/etomo/util/HashedArray.java`.
//!
//! A list of name, value pairs that can be accessed by keys or indexes.  Java keeps a
//! `Hashtable` (key -> value) and a `Vector` of the keys in insertion order.
//!
//! **Shape.**  The Java object is shared between threads (`LoadMonitor.programs`,
//! `IntermittentBackgroundProcess.monitors`) and its mutators are `synchronized`, so
//! both collections sit behind one `Mutex` and every method takes `&self`.  The
//! unsynchronized Java readers (`get(int)`, `size()`) went through the internally
//! synchronized `Vector`/`Hashtable`, so they lock too.  Values are handed out by
//! clone (an `Arc` in every caller), as Java hands out the shared reference.
//!
//! Java keys are `Object`s compared by `equals`/`hashCode`; here `K: Eq + Hash`.  A
//! Java `null` key is ignored by `add` and `remove`; a Rust key cannot be null, so
//! those guards have no counterpart (a caller whose key is an `Option` stores `None`
//! as an ordinary key).  A `Hashtable` rejects a null value, which a Rust `V` cannot
//! be either.
//!
//! Overloads carry a suffix naming their parameter types: `add(Object, Object)` is
//! `add_object_object`, `add(Object)` is `add_object`, `get(Object)` is `get_object`,
//! `get(int)` is `get_int`.

use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::Mutex;

use crate::imod::etomo::util::utilities;

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `HashedArray`.
pub struct HashedArray<K, V> {
    /// Java private final `valueMap` and `keyArray`, under the object's monitor.
    inner: Mutex<HashedArrayFields<K, V>>,
}

/// The two Java fields of `HashedArray` (not a source type: the contents of the lock).
struct HashedArrayFields<K, V> {
    /// Java private final `valueMap`, `new Hashtable()`.
    value_map: HashMap<K, V>,
    /// Java private final `keyArray`, `new Vector()`.
    key_array: Vec<K>,
}

impl<K, V> Default for HashedArray<K, V>
where
    K: Eq + Hash + Clone + Debug,
    V: Clone,
{
    fn default() -> Self {
        HashedArray::new()
    }
}

impl<K, V> HashedArray<K, V>
where
    K: Eq + Hash + Clone + Debug,
    V: Clone,
{
    /// Java implicit constructor `HashedArray()`.
    pub fn new() -> HashedArray<K, V> {
        HashedArray {
            inner: Mutex::new(HashedArrayFields {
                value_map: HashMap::new(),
                key_array: Vec::new(),
            }),
        }
    }

    /// Java `printToErr()`.  Java prints each key's `toString()`; the key's `Debug`
    /// form stands in for it.
    pub fn print_to_err(&self) {
        let inner = self.inner.lock().unwrap();
        for key in inner.key_array.iter() {
            eprintln!("{:?}", key);
        }
    }

    /// Java package-private `selfTestInvariants()`.  Java throws
    /// `IllegalStateException`; that is a panic here.
    fn self_test_invariants(&self, inner: &HashedArrayFields<K, V>) {
        if !utilities::is_self_test() {
            return;
        }
        // collections should be the same size
        if inner.value_map.len() != inner.key_array.len() {
            panic!(
                "{}",
                "sizes are different:".to_string()
                    + "valueMap="
                    + &inner.value_map.len().to_string()
                    + ",keyArray="
                    + &inner.key_array.len().to_string()
            );
        }
        // a key in keyArray must exist in valueMap
        for i in 0..inner.key_array.len() {
            if !inner.value_map.contains_key(&inner.key_array[i]) {
                panic!(
                    "{}",
                    "a key in keyArray is not in valueMap:".to_string()
                        + "key="
                        + &format!("{:?}", inner.key_array[i])
                );
            }
        }
    }

    /// Java package-private `selfTestAdd(Object, Object)`.  Java throws
    /// `IllegalStateException`; that is a panic here.
    ///
    /// Java's third test, `valueMap.get(key) != value`, compares object identity with
    /// the value just put.  The Rust value was moved into the map by `add_object_object`
    /// in the same critical section, so it is always the stored one and the test has no
    /// counterpart; the value is also not printed (it need not be `Debug`).
    fn self_test_add(&self, inner: &HashedArrayFields<K, V>, key: &K) {
        if !utilities::is_self_test() {
            return;
        }
        if !inner.value_map.contains_key(key) {
            panic!(
                "{}",
                "The added key is not in valueMap:".to_string() + "key=" + &format!("{:?}", key)
            );
        }
        if !inner.key_array.contains(key) {
            panic!(
                "{}",
                "The added key is not in keyArray:".to_string() + "key=" + &format!("{:?}", key)
            );
        }
    }

    /// Java synchronized `add(Object key, Object value)`.  Add a new value with key.
    pub fn add_object_object(&self, key: K, value: V) {
        let mut inner = self.inner.lock().unwrap();
        if inner.value_map.insert(key.clone(), value).is_none() {
            inner.key_array.push(key.clone());
        }
        self.self_test_invariants(&inner);
        self.self_test_add(&inner, &key);
    }

    /// Java synchronized `remove(Object key)`.
    ///
    /// Upstream bug fixed (HashedArray.java:141-145): Java removes from `keyArray` every
    /// key whose `hashCode()` equals the removed key's, so a hash collision also drops
    /// an unrelated key from the ordered list (leaving it in `valueMap`, which breaks
    /// the class's own invariant that the two are the same size), and the loop skips
    /// the element after each removal.  The keys in `keyArray` are unique (`add` appends
    /// only a key that was not already mapped), so the evident intent is to remove the
    /// one equal key; that is what is done.
    pub fn remove(&self, key: &K) {
        let mut inner = self.inner.lock().unwrap();
        if !inner.value_map.contains_key(key) {
            return;
        }
        inner.value_map.remove(key);
        let mut i = 0;
        while i < inner.key_array.len() {
            if inner.key_array[i] == *key {
                inner.key_array.remove(i);
                break;
            }
            i += 1;
        }
        self.self_test_invariants(&inner);
    }

    /// Java synchronized `get(Object key)`.
    pub fn get_object(&self, key: &K) -> Option<V> {
        let inner = self.inner.lock().unwrap();
        inner.value_map.get(key).cloned()
    }

    /// Java `get(int index)`.
    ///
    /// Upstream bug fixed (HashedArray.java:163): `keyArray.get(index)` throws
    /// `ArrayIndexOutOfBoundsException` for an index at or past the end, which a caller
    /// looping to `size()` reaches when another thread removes an element meanwhile
    /// (the method is not synchronized).  An index past the end returns null, as a
    /// negative index already does.
    pub fn get_int(&self, index: i32) -> Option<V> {
        if index < 0 {
            return None;
        }
        let inner = self.inner.lock().unwrap();
        let key = inner.key_array.get(index as usize)?;
        inner.value_map.get(key).cloned()
    }

    /// Java `size()`.
    pub fn size(&self) -> i32 {
        self.inner.lock().unwrap().value_map.len() as i32
    }

    /// Java synchronized `containsKey(Object key)`.
    pub fn contains_key(&self, key: &K) -> bool {
        self.inner.lock().unwrap().value_map.contains_key(key)
    }
}

impl<K> HashedArray<K, K>
where
    K: Eq + Hash + Clone + Debug,
{
    /// Java synchronized `add(Object key)`: `add(key, key)`.  Only a list whose values
    /// are its keys can take it.
    pub fn add_object(&self, key: K) {
        self.add_object_object(key.clone(), key);
    }
}
