//! `IMOD/Etomo/src/etomo/storage/autodoc/WriteOnlyAttributeList.java`.
#![allow(dead_code)]

use super::attribute::Attribute;
use crate::imod::etomo::ui::swing::token::Token;

/// Source abstract base for Autodoc/Section/Attribute attribute owners.
///
/// `addAttribute` is declared as returning `WriteOnlyAttributeList`, but every
/// implementation returns the `Attribute` that `AttributeList.addAttribute` produced,
/// and `AutodocParser.buildAttribute` casts the result back to `Attribute`; the
/// concrete pointer is therefore the return type here.
pub trait WriteOnlyAttributeList {
    /// Java `addAttribute(Token, int)`.
    ///
    /// # Safety
    /// `name` must be null or point to a live `Token` link list.
    unsafe fn add_attribute(&mut self, name: *mut Token, line_num: i32) -> *mut Attribute;
    /// Java `isGlobal()`.
    fn is_global(&self) -> bool;
    /// Java `isAttribute()`.
    fn is_attribute(&self) -> bool;
}

#[cfg(test)]
mod tests {
    use super::WriteOnlyAttributeList;
    use crate::imod::etomo::storage::autodoc::autodoc::Autodoc;
    use crate::imod::etomo::storage::autodoc::write_only_statement_list::WriteOnlyStatementList;
    use crate::imod::etomo::ui::swing::token::{self, Token};

    /// Java's three implementations disagree on `isGlobal`/`isAttribute`: the autodoc is
    /// global and not an attribute, a section is neither, and an attribute takes
    /// `isGlobal` from its parent and is always an attribute.
    #[test]
    fn ownership_flags_follow_the_three_source_implementations() {
        unsafe {
            let autodoc = Autodoc::new(Some("flags"), std::ptr::null_mut());
            assert!((*autodoc).is_global());
            assert!(!(*autodoc).is_attribute());
            let name = Box::into_raw(Box::new(Token::new()));
            (*name).set_type_and_string(token::Type::Anything, "global");
            let attribute = (*autodoc).add_attribute(name, 1);
            assert!((*attribute).is_global());
            assert!((*attribute).is_attribute());
            let r#type = Box::into_raw(Box::new(Token::new()));
            (*r#type).set_type_and_string(token::Type::Anything, "Field");
            let section_name = Box::into_raw(Box::new(Token::new()));
            (*section_name).set_type_and_string(token::Type::Anything, "One");
            let section = (*autodoc).add_section(r#type, section_name, 2);
            assert!(!(*section).is_global());
            assert!(!(*section).is_attribute());
            let in_section = Box::into_raw(Box::new(Token::new()));
            (*in_section).set_type_and_string(token::Type::Anything, "local");
            let local = (*section).add_attribute(in_section, 2);
            assert!(!(*local).is_global());
            assert!((*local).is_attribute());
        }
    }
}
