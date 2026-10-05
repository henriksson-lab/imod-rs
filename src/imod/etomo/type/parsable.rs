//! `IMOD/Etomo/src/etomo/type/Parsable.java`.
//!
//! Any kind of parsable element or list.

use crate::imod::etomo::storage::autodoc::read_only_attribute::ReadOnlyAttribute;

/// Java `public interface Parsable`.
pub trait Parsable {
    /// Java `clear()`.
    fn clear_parsable(&mut self);

    /// Java `parse(String)`.
    fn parse_string(&mut self, parsable_string: Option<&str>);

    /// Java `validate()`.
    fn validate_parsable(&self) -> Option<String>;

    /// Java `getParsableString()`.
    fn get_parsable_string_parsable(&self) -> Option<String>;

    /// Java `parse(ReadOnlyAttribute)`.
    fn parse_attribute(&mut self, attribute: Option<&dyn ReadOnlyAttribute>);

    /// Java `isEmpty()`.
    fn is_empty_parsable(&self) -> bool;

    /// Java `size()`: number of elements.
    fn size_parsable(&self) -> i32;
}
