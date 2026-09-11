//! `IMOD/Etomo/src/etomo/util/UniqueHashedArray.java`.

use std::collections::HashMap;

use super::unique_key::UniqueKey;

/// Java `UniqueHashedArray<V>`.
///
/// `key_array` deliberately remains separate from `map`: Java uses the map
/// for lookup while preserving insertion order independently for UI indexing.
#[derive(Debug)]
pub struct UniqueHashedArray<V> {
    map: HashMap<UniqueKey, V>,
    key_array: Vec<UniqueKey>,
}

impl<V> UniqueHashedArray<V> {
    /// Java `UniqueHashedArray()`.
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
            key_array: Vec::new(),
        }
    }

    /// Java package-private `UniqueHashedArray(List<UniqueKey>)`.
    fn with_key_array(key_array: Vec<UniqueKey>) -> Self {
        Self {
            map: HashMap::new(),
            key_array,
        }
    }

    /// Java `add(String, V)`.
    pub fn add_with_name(&mut self, key_name: String, value: V) -> UniqueKey {
        let key = UniqueKey::new(key_name, self);
        self.key_array.push(key.clone());
        self.map.insert(key.clone(), value);
        key
    }

    /// Java `add(UniqueKey, V)`.
    pub fn add(&mut self, key: UniqueKey, value: V) -> Result<UniqueKey, String> {
        if self.get(&key).is_some() {
            return Err(format!("Key, {key}, is not unique."));
        }
        self.key_array.push(key.clone());
        self.map.insert(key.clone(), value);
        Ok(key)
    }

    /// Java `set(int, V)`.
    pub fn set(&mut self, key_index: usize, value: V) -> Option<UniqueKey> {
        let key = self.key_array.get(key_index)?.clone();
        self.map.remove(&key);
        self.map.insert(key.clone(), value);
        Some(key)
    }

    /// Java `remove(UniqueKey)`.
    pub fn remove(&mut self, key: &UniqueKey) -> Option<V> {
        self.key_array
            .retain(|stored_key| !stored_key.equals(Some(key)));
        self.map.remove(key)
    }

    /// Java `rekey(UniqueKey, String)`.
    pub fn rekey_with_name(
        &mut self,
        old_key: &UniqueKey,
        new_key_name: String,
    ) -> Option<UniqueKey> {
        self.rekey(old_key, UniqueKey::new(new_key_name, self))
    }

    /// Java `rekey(UniqueKey, UniqueKey)`.
    pub fn rekey(&mut self, old_key: &UniqueKey, new_key: UniqueKey) -> Option<UniqueKey> {
        let index = self.get_index(old_key)?;
        let value = self.map.remove(old_key)?;
        self.map.insert(new_key.clone(), value);
        self.key_array[index] = new_key.clone();
        Some(new_key)
    }

    /// Java `contains(UniqueKey)`.
    pub fn contains(&self, key: Option<&UniqueKey>) -> bool {
        key.is_some_and(|key| self.map.contains_key(key))
    }

    /// Java `get(UniqueKey)`.
    pub fn get(&self, key: &UniqueKey) -> Option<&V> {
        self.map.get(key)
    }

    /// Java `get(int)`.
    pub fn get_at(&self, index: usize) -> Option<&V> {
        self.key_array.get(index).and_then(|key| self.map.get(key))
    }

    /// Rust mutable equivalent of Java `get(UniqueKey)` for the translated
    /// Swing component state.
    pub fn get_mut(&mut self, key: &UniqueKey) -> Option<&mut V> {
        self.map.get_mut(key)
    }

    /// Rust mutable equivalent of Java `get(int)` for the translated Swing
    /// component state.
    pub fn get_at_mut(&mut self, index: usize) -> Option<&mut V> {
        let key = self.key_array.get(index)?.clone();
        self.map.get_mut(&key)
    }

    /// Java `getKey(int)`.
    pub fn get_key(&self, index: usize) -> Option<&UniqueKey> {
        self.key_array.get(index)
    }

    /// Java `getIndex(UniqueKey)`.
    pub fn get_index(&self, key: &UniqueKey) -> Option<usize> {
        self.key_array
            .iter()
            .position(|stored_key| key.equals(Some(stored_key)))
    }

    /// Java `size()`.
    pub fn size(&self) -> usize {
        self.key_array.len()
    }

    /// Java `getEmptyUniqueHashedArray()`.
    pub fn get_empty_unique_hashed_array(&self) -> Self {
        Self::with_key_array(self.key_array.clone())
    }

    /// Java `toString()` / package-private `paramString()`.
    pub fn param_string(&self) -> String {
        let mut buffer = String::from(",map=");
        for key in &self.key_array {
            buffer.push_str(&format!("\nkey={key}"));
            if let Some(value) = self.map.get(key) {
                buffer.push_str(&format!(",value={:?}", std::any::type_name_of_val(value)));
            }
        }
        buffer
    }
}

impl<V> Default for UniqueHashedArray<V> {
    fn default() -> Self {
        Self::new()
    }
}
