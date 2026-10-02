//! `IMOD/Etomo/src/etomo/storage/DirectiveMap.java`.
//!
//! A sorted map (Java `TreeMap`, here `BTreeMap`) from directive key to the shared
//! `Directive`.  `getDirective` is overloaded: `getDirective(String)` is
//! `get_directive_string` and `getDirective(DirectiveDef)` is
//! `get_directive_directive_def`.
//!
//! `KeySet` and `Iterator` take a snapshot of the key set: Java's `TreeMap.keySet()` is a
//! live view whose iterator throws `ConcurrentModificationException` if the map is
//! modified during iteration, so a snapshot answers identically for every iteration the
//! source can complete.

use std::collections::BTreeMap;
// This module defines a struct named `Iterator` (Java `DirectiveMap.Iterator`), which
// shadows the prelude trait; import the trait anonymously for its methods.
use std::iter::Iterator as _;
use std::sync::Arc;

use crate::imod::etomo::storage::directive::Directive;
use crate::imod::etomo::storage::directive_def::DirectiveDef;
use crate::imod::etomo::storage::directive_name::DirectiveName;
use crate::imod::etomo::storage::directive_type::DirectiveType;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::directive_interface::DirectiveInterface;
use crate::imod::etomo::r#type::directive_map_interface::DirectiveMapInterface;

/// Java `rcsid`.
pub const RCSID: &str = "$Id:$";

/// Java final `DirectiveMap implements DirectiveMapInterface`.
#[derive(Default)]
pub struct DirectiveMap {
    /// Java private final field `map = new TreeMap<String, Directive>()`.
    map: BTreeMap<String, Arc<Directive>>,
}

impl DirectiveMap {
    /// Java `DirectiveMap()`.
    pub fn new() -> DirectiveMap {
        DirectiveMap {
            map: BTreeMap::new(),
        }
    }

    /// Java `put(String, Directive)`.
    pub fn put(&mut self, key: &str, value: Arc<Directive>) {
        self.map.insert(key.to_string(), value);
    }

    /// Java `clear()`.
    pub fn clear(&mut self) {
        self.map.clear();
    }

    /// Java `getDirective(String)`.
    pub fn get_directive_string(&self, key: Option<&str>) -> Option<Arc<Directive>> {
        let key = key?;
        self.map.get(key).cloned()
    }

    /// Java `getDirective(DirectiveDef)`.
    pub fn get_directive_directive_def(
        &self,
        directive_def: Option<DirectiveDef>,
    ) -> Option<Arc<Directive>> {
        let directive_def = directive_def?;
        // `map.get(null)` on a TreeMap throws NullPointerException; `getStandardKey`
        // returns null only for a def with no name, which no DirectiveDef constant has.
        self.map.get(&directive_def.get_standard_key()?).cloned()
    }

    /// Java `getDirectiveFromPair(DirectiveDef, AxisID)`.
    pub fn get_directive_from_pair(
        &self,
        directive_def: Option<DirectiveDef>,
        pair_axis_id: Option<AxisID>,
    ) -> Option<Arc<Directive>> {
        let directive_def = directive_def?;
        self.map
            .get(&directive_def.switch_standard_key(pair_axis_id)?)
            .cloned()
    }

    /// Java `keySet(DirectiveType)`.
    pub fn key_set(&self, r#type: Option<DirectiveType>) -> KeySet {
        KeySet::new(self.map.keys().cloned().collect(), r#type)
    }
}

impl DirectiveMapInterface for DirectiveMap {
    /// Java `@Override getDirectiveFromPair(DirectiveDef, AxisID)`.
    fn get_directive_from_pair(
        &self,
        directive_def: Option<DirectiveDef>,
        pair_axis_id: Option<AxisID>,
    ) -> Option<Arc<dyn DirectiveInterface>> {
        DirectiveMap::get_directive_from_pair(self, directive_def, pair_axis_id)
            .map(|directive| directive as Arc<dyn DirectiveInterface>)
    }

    /// Java `@Override getDirective(DirectiveDef)`.
    fn get_directive_directive_def(
        &self,
        directive_def: Option<DirectiveDef>,
    ) -> Option<Arc<dyn DirectiveInterface>> {
        DirectiveMap::get_directive_directive_def(self, directive_def)
            .map(|directive| directive as Arc<dyn DirectiveInterface>)
    }
}

/// Java `toString()`: `map.toString()`, `AbstractMap`'s `{key=value, ...}` form.
impl std::fmt::Display for DirectiveMap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("{")?;
        let mut first = true;
        for (key, value) in self.map.iter() {
            if !first {
                f.write_str(", ")?;
            }
            first = false;
            write!(f, "{}={}", key, value)?;
        }
        f.write_str("}")
    }
}

/// Java public static final nested class `DirectiveMap.KeySet`.
pub struct KeySet {
    /// Java private final field `keySet`.
    key_set: Vec<String>,
    /// Java private final field `type`.
    r#type: Option<DirectiveType>,
}

impl KeySet {
    /// Java private `KeySet(Set<String>, DirectiveType)`.
    fn new(key_set: Vec<String>, r#type: Option<DirectiveType>) -> KeySet {
        KeySet { key_set, r#type }
    }

    /// Java `iterator()`.
    pub fn iterator(&self) -> Iterator {
        Iterator::new(&self.key_set, self.r#type)
    }
}

/// Java public static final nested class `DirectiveMap.Iterator`.
pub struct Iterator {
    /// Java private final field `iterator` (over the key set, in sorted order).
    iterator: std::vec::IntoIter<String>,
    /// Java private final field `type`.
    r#type: Option<DirectiveType>,
    /// Java private field `saveKey`, initialised to null.
    save_key: Option<String>,
}

impl Iterator {
    /// Java private `Iterator(Set<String>, DirectiveType)`.
    fn new(key_set: &[String], r#type: Option<DirectiveType>) -> Iterator {
        Iterator {
            iterator: key_set.to_vec().into_iter(),
            r#type,
            save_key: None,
        }
    }

    /// Java `hasNext()`.  Can be run multiple times without incrementing the iterator.
    /// Returns true if there is at least one key left that matches type.
    pub fn has_next(&mut self) -> bool {
        if self.r#type.is_none() {
            return self.iterator.len() > 0;
        }
        // Use next() to check if any keys match type. If saveKey is already set, then
        // hasNext() was already run successfully.
        if self.save_key.is_some() {
            return true;
        }
        // `next()` throwing NoSuchElementException is `None` here, and hasNext answers
        // false for it as the source's catch does.
        self.save_key = self.next();
        self.save_key.is_some()
    }

    /// Java `next()`.  Increments the iterator.  Returns the next element in the
    /// iteration that matches type.  Java throws NoSuchElementException when the
    /// iteration has no more elements; that is `None` here.
    pub fn next(&mut self) -> Option<String> {
        if self.r#type.is_none() {
            return self.iterator.next();
        }
        // If saveKey is set, then hasNext() was run, so use the key saved by hasNext.
        if self.save_key.is_some() {
            // Set saveKey to null so that iterator will be incremented next time next()
            // is called.
            return self.save_key.take();
        }
        // Call next() until a key is returned that matches type.
        while let Some(key) = self.iterator.next() {
            if DirectiveName::equals_string_directive_type(Some(&key), self.r#type) {
                return Some(key);
            }
        }
        None
    }
}
