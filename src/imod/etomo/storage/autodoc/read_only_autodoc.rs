//! `IMOD/Etomo/src/etomo/storage/autodoc/ReadOnlyAutodoc.java`.
#![allow(dead_code)]

use super::attribute::Attribute;
use super::attribute_list::AttributeList;
use super::autodoc::InternalTestType;
use super::read_only_section_list::ReadOnlySectionList;
use super::read_only_statement_list::ReadOnlyStatementList;
use std::collections::HashMap;

/// Source `ReadOnlyAutodoc extends ReadOnlyStatementList, ReadOnlySectionList`.
///
/// The interface redeclares `getSection`, `getSectionLocation(String)`,
/// `getSectionLocation()` and `nextSection` with `@Override`; those are the
/// supertrait's methods and are not repeated here.  `setDebug(boolean)` is
/// `set_debug_to` because `ReadOnlySectionList` already declares the no-argument
/// `setDebug()`.
pub trait ReadOnlyAutodoc: ReadOnlyStatementList + ReadOnlySectionList {
    /// Java `getAttributeValues(String, String)`.
    ///
    /// # Safety
    /// Every section and attribute in the autodoc must be live.
    unsafe fn get_attribute_values(
        &self,
        section_type: Option<&str>,
        attribute_name: Option<&str>,
    ) -> Option<HashMap<String, Option<String>>>;
    /// Java `getAttributeMultiLineValues(String, String)`.
    ///
    /// # Safety
    /// See `get_attribute_values`.
    unsafe fn get_attribute_multi_line_values(
        &self,
        section_type: Option<&str>,
        attribute_name: Option<&str>,
    ) -> Option<HashMap<String, Option<String>>>;
    /// Java `isError()`.
    fn is_error(&self) -> bool;
    /// Java `printStoredData()`.
    ///
    /// # Safety
    /// See `get_attribute_values`.
    unsafe fn print_stored_data(&self);
    /// Java `sectionExists(String)`.
    ///
    /// # Safety
    /// See `get_attribute_values`.
    unsafe fn section_exists(&self, r#type: Option<&str>) -> bool;
    /// Java `getAttribute(String)`, declared as `ReadOnlyAttribute`.
    ///
    /// # Safety
    /// See `get_attribute_values`.
    unsafe fn get_attribute(&self, name: Option<&str>) -> *mut Attribute;
    /// Java `runInternalTest(InternalTestType, boolean, boolean)`.
    fn run_internal_test(
        &mut self,
        r#type: InternalTestType,
        show_tokens: bool,
        show_details: bool,
    );
    /// Java `isDebug()`.
    fn is_debug(&self) -> bool;
    /// Java `setDebug(boolean)`.
    fn set_debug_to(&mut self, input: bool);
    /// Java `getAutodocName()`.
    fn get_autodoc_name(&self) -> String;
    /// Java `exists()`.
    fn exists(&self) -> bool;
    /// Java `getChildren()`, declared as `ReadOnlyAttributeList`.
    fn get_children(&self) -> *mut AttributeList;
}
