//! `IMOD/Etomo/src/etomo/storage/autodoc/WritableAutodoc.java`.
#![allow(dead_code)]

use super::attribute::Attribute;
use super::read_only_autodoc::ReadOnlyAutodoc;
use super::statement::Statement;
use crate::imod::etomo::ui::swing::token::Token;

/// Source `WritableAutodoc extends ReadOnlyAutodoc`.  It redeclares
/// `setDebug(boolean)` with `@Override`; that is `ReadOnlyAutodoc`'s `set_debug_to`
/// and is not repeated here.  `getWritableAttribute` is
/// declared as `WritableAttribute` and `removeNameValuePair`/`removeStatement` as
/// `WritableStatement`; `Attribute` and `Statement` are the package's only
/// implementations of those.
pub trait WritableAutodoc: ReadOnlyAutodoc {
    /// Java `addNameValuePairAttribute(String, String)`.
    ///
    /// # Safety
    /// The autodoc's attribute and statement lists must be live.
    unsafe fn add_name_value_pair_attribute(&mut self, name: Option<&str>, value: Option<&str>);
    /// Java `addNameValuePairAttribute(String, String, int)`.
    ///
    /// # Safety
    /// See `add_name_value_pair_attribute`.
    unsafe fn add_name_value_pair_attribute_with_line_num(
        &mut self,
        name: Option<&str>,
        value: Option<&str>,
        line_num: i32,
    );
    /// Java `getWritableAttribute(String)`.
    ///
    /// # Safety
    /// See `add_name_value_pair_attribute`.
    unsafe fn get_writable_attribute(&self, name: Option<&str>) -> *mut Attribute;
    /// Java `addComment(Token, int)`.
    ///
    /// # Safety
    /// `comment` must be null or point to a live `Token` link list.
    unsafe fn add_comment(&mut self, comment: *mut Token, line_num: i32);
    /// Java `addEmptyLine(int)`.
    ///
    /// # Safety
    /// See `add_name_value_pair_attribute`.
    unsafe fn add_empty_line(&mut self, line_num: i32);
    /// Java `addComment(String, int)`.
    ///
    /// # Safety
    /// See `add_name_value_pair_attribute`.
    unsafe fn add_comment_string(&mut self, comment: Option<&str>, line_num: i32);
    /// Java `removeNameValuePair(String)`.
    ///
    /// # Safety
    /// See `add_name_value_pair_attribute`.
    unsafe fn remove_name_value_pair(&mut self, name: Option<&str>) -> *mut dyn Statement;
    /// Java `removeStatement(WritableStatement)`.
    ///
    /// # Safety
    /// `statement` must point to a live statement in this autodoc's list.
    unsafe fn remove_statement(&mut self, statement: *mut dyn Statement) -> *mut dyn Statement;
    // TODO(unit): needs etomo/process/EmergencyMonitor.java - Java `write()`
    // (WritableAutodoc.java:29) writes the autodoc through a `LogFile.Handle`, whose
    // constructor requires an `EmergencyMonitor`; `Autodoc.write()` implements it.

    /// Java `printStatementList()`.
    fn print_statement_list(&self);
    /// Java `wrapAttributeValues(String, String, String, String, int, int)`.
    ///
    /// # Safety
    /// See `add_name_value_pair_attribute`.
    unsafe fn wrap_attribute_values(
        &mut self,
        no_wrap_prefix: Option<&str>,
        wrap_prefix: Option<&str>,
        divider: Option<&str>,
        default_divider: Option<&str>,
        min_length: i32,
        wrap_length: i32,
    );
}
