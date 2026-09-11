//! `IMOD/Etomo/src/etomo/storage/autodoc/WriteOnlyStatementList.java`.
#![allow(dead_code)]

use super::name_value_pair::NameValuePair;
use super::section::Section;
use super::write_only_attribute_list::WriteOnlyAttributeList;
use crate::imod::etomo::ui::swing::token::Token;

/// Java's package-private abstract `WriteOnlyStatementList extends
/// WriteOnlyAttributeList`.
pub trait WriteOnlyStatementList: WriteOnlyAttributeList {
    /// Java `addNameValuePair(int)`.
    ///
    /// # Safety
    /// The statement list this appends to must be live.
    unsafe fn add_name_value_pair(&mut self, line_num: i32) -> *mut NameValuePair;
    /// Java `addSection(Token, Token, int)`.
    ///
    /// # Safety
    /// `type` and `name` must be null or point to live `Token` link lists.
    unsafe fn add_section(
        &mut self,
        r#type: *mut Token,
        name: *mut Token,
        line_num: i32,
    ) -> *mut Section;
    /// Java `addEmptyLine(int)`.
    ///
    /// # Safety
    /// See `add_name_value_pair`.
    unsafe fn add_empty_line(&mut self, line_num: i32);
    /// Java `addComment(Token, int)`.
    ///
    /// # Safety
    /// `comment` must be null or point to a live `Token` link list.
    unsafe fn add_comment(&mut self, comment: *mut Token, line_num: i32);
    /// Java `setCurrentDelimiter(Token)`.
    ///
    /// # Safety
    /// `new_delimiter` must point to a live `Token` link list.
    unsafe fn set_current_delimiter(&mut self, new_delimiter: *mut Token);
    /// Java `getCurrentDelimiter()`.
    fn get_current_delimiter(&self) -> String;
}
