//! `IMOD/Etomo/src/etomo/storage/autodoc/ReadOnlySectionList.java`.
#![allow(dead_code)]

use super::section::Section;
use super::section_location::SectionLocation;

/// Source `ReadOnlySectionList` interface.  `getSection` and `nextSection` are declared
/// as `ReadOnlySection`, whose only implementation in the package is `Section`.
pub trait ReadOnlySectionList {
    /// Java `getSection(String, String)`.
    ///
    /// # Safety
    /// Every section in the list must be live.
    unsafe fn get_section(&self, r#type: Option<&str>, name: Option<&str>) -> *mut Section;
    /// Java `getSectionLocation(String)`.
    ///
    /// # Safety
    /// See `get_section`.
    unsafe fn get_section_location_by_type(&self, r#type: Option<&str>) -> Option<SectionLocation>;
    /// Java `getSectionLocation()`.
    fn get_section_location(&self) -> Option<SectionLocation>;
    /// Java `nextSection(SectionLocation)`.  Java's parameter may be null.
    ///
    /// # Safety
    /// See `get_section`.
    unsafe fn next_section(&self, location: Option<&mut SectionLocation>) -> *mut Section;
    /// Java `getString()`.
    fn get_string(&self) -> String;
    /// Java `setDebug()`.
    fn set_debug(&mut self);
    /// Java `getName()`.
    fn get_name(&self) -> Option<String>;
}
