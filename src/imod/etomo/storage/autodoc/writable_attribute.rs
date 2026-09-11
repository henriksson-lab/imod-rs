//! `IMOD/Etomo/src/etomo/storage/autodoc/WritableAttribute.java`.
#![allow(dead_code)]

use super::read_only_attribute::ReadOnlyAttribute;

/// Source `WritableAttribute extends ReadOnlyAttribute`.
pub trait WritableAttribute: ReadOnlyAttribute {
    /// Java `setValue(String)`.
    fn set_value(&mut self, new_value: Option<&str>);
}
