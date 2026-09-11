//! `IMOD/Etomo/src/etomo/storage/autodoc/ReadOnlyStatement.java`.
#![allow(dead_code)]

use super::section::Section;
use super::statement::Type;

/// Complete `ReadOnlyStatement` source interface.  `getSubsection` is declared as
/// `ReadOnlySection`, whose only implementation in the package is `Section`, so it is
/// the concrete pointer here; `Statement`'s `getType` returns the nested
/// `Statement.Type`, translated in `statement.rs`.
pub trait ReadOnlyStatement {
    /// Java `getType()`.
    fn get_type(&self) -> Type;
    /// Java `getString()`.
    fn get_string(&self) -> String;
    /// Java `sizeLeftSide()`.
    fn size_left_side(&self) -> i32;
    /// Java `getLeftSide()`.
    fn get_left_side(&self) -> Option<String>;
    /// Java `getLeftSide(int)`.
    fn get_left_side_at(&self, index: i32) -> Option<String>;
    /// Java `getRightSide()`.
    fn get_right_side(&self) -> Option<String>;
    /// Java `getSubsection()`.
    fn get_subsection(&self) -> *mut Section;
    /// Java `getLineNum()`.
    fn get_line_num(&self) -> i32;
}

#[cfg(test)]
mod tests {
    use super::super::statement::Type;
    #[test]
    fn statement_type_renders_the_source_strings() {
        assert_eq!(Type::NameValuePair.to_string(), "NAME_VALUE_PAIR");
        assert_eq!(Type::Subsection.to_string(), "SUBSECTION");
        assert_eq!(Type::Comment.to_string(), "COMMENT");
        assert_eq!(Type::EmptyLine.to_string(), "EMPTY_LINE");
    }
}
