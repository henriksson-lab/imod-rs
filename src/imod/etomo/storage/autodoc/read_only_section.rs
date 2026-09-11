//! `IMOD/Etomo/src/etomo/storage/autodoc/ReadOnlySection.java`.
#![allow(dead_code)]

use super::attribute::Attribute;
use super::read_only_section_list::ReadOnlySectionList;
use super::read_only_statement_list::ReadOnlyStatementList;

/// Source `ReadOnlySection extends ReadOnlyStatementList, ReadOnlySectionList`.  Java's
/// `getString`/`getName` are declared by both supertypes; Rust inherits one copy.
pub trait ReadOnlySection: ReadOnlyStatementList + ReadOnlySectionList {
    /// Java `getAttribute(String)`, declared as `ReadOnlyAttribute`, whose only
    /// implementation in the package is `Attribute`.
    ///
    /// # Safety
    /// The section's attribute list must be live.
    unsafe fn get_attribute(&self, name: Option<&str>) -> *mut Attribute;
    /// Java `getType()`.
    fn get_type(&self) -> String;
}
