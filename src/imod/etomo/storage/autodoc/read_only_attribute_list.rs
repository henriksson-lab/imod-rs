//! `IMOD/Etomo/src/etomo/storage/autodoc/ReadOnlyAttributeList.java`.
#![allow(dead_code)]

use super::attribute::Attribute;
use super::read_only_attribute_iterator::ReadOnlyAttributeIterator;

/// Source `ReadOnlyAttributeList` interface.  The Rust list owns stable boxed
/// attributes, so its iterator borrows those boxes rather than exposing an owning raw
/// pointer collection.
pub trait ReadOnlyAttributeList {
    /// Java `iterator()`.
    fn iterator(&self) -> ReadOnlyAttributeIterator<'_>;
}

#[cfg(test)]
mod tests {
    use super::ReadOnlyAttributeList;
    use crate::imod::etomo::storage::autodoc::attribute_list::AttributeList;
    use crate::imod::etomo::storage::autodoc::autodoc::Autodoc;
    use crate::imod::etomo::storage::autodoc::read_only_attribute::ReadOnlyAttribute;
    use crate::imod::etomo::storage::autodoc::read_only_autodoc::ReadOnlyAutodoc;
    use crate::imod::etomo::storage::autodoc::writable_autodoc::WritableAutodoc;

    /// `iterator()` walks `list`, which carries the source's insertion order, and a
    /// repeated attribute is one entry with a second occurrence rather than a second
    /// entry.
    #[test]
    fn iterator_keeps_the_source_insertion_order() {
        unsafe {
            let autodoc = Autodoc::new(Some("iter"), std::ptr::null_mut());
            (*autodoc).add_name_value_pair_attribute(Some("first"), Some("1"));
            (*autodoc).add_name_value_pair_attribute(Some("second"), Some("2"));
            (*autodoc).add_name_value_pair_attribute(Some("first"), Some("3"));
            let children: *mut AttributeList = (*autodoc).get_children();
            let mut iterator = (*children).iterator();
            let mut names = Vec::new();
            while iterator.has_next() {
                names.push(ReadOnlyAttribute::get_name(&**iterator.next().unwrap()));
            }
            assert_eq!(names, ["first", "second"]);
        }
    }
}
