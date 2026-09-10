//! `IMOD/Etomo/src/etomo/storage/Storable.java`.

use std::collections::BTreeMap;

/// Rust trait equivalent to Java `Storable`: its four source methods operate
/// on a Java-`Properties`-equivalent deterministic string map.
pub trait Storable {
    fn store(&self, properties: &mut BTreeMap<String, String>);
    fn store_with_prepend(&self, properties: &mut BTreeMap<String, String>, prepend: &str);
    fn load(&mut self, properties: &BTreeMap<String, String>);
    fn load_with_prepend(&mut self, properties: &BTreeMap<String, String>, prepend: &str);
}
