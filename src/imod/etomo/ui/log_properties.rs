//! `IMOD/Etomo/src/etomo/ui/LogProperties.java`.
//!
//! A Java interface with two methods and one constant.  Java `Properties` is modelled
//! throughout this translation by a deterministic `BTreeMap<String, String>`, as
//! `etomo/storage/storable.rs` does.
#![allow(dead_code)]

use std::collections::BTreeMap;

/// Java `rcsid`.
pub const RCSID: &str = "$Id:$";

/// Java `LogProperties`.
pub trait LogProperties {
    /// Java `store(Properties, String)`.
    fn store(&self, props: &mut BTreeMap<String, String>, prepend: Option<&str>);

    /// Java `load(Properties, String)`.
    fn load(&mut self, props: &BTreeMap<String, String>, prepend: Option<&str>);
}
