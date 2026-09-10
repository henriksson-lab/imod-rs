//! `IMOD/Etomo/src/etomo/storage/autodoc/ReadOnlyAttributeList.java`.

use super::read_only_attribute_iterator::ReadOnlyAttributeIterator;

/// Source `ReadOnlyAttributeList` interface.  `T` is the eventual Rust
/// translation of Java's package-private `Attribute`; the generic is directly
/// equivalent to Java's `Iterator<Attribute>` return contract.
pub trait ReadOnlyAttributeList<T> {
    /// Java `iterator()`.
    fn iterator(&self) -> ReadOnlyAttributeIterator<'_, T>;
}

#[cfg(test)]
mod tests {
    use super::ReadOnlyAttributeList;
    use crate::imod::etomo::storage::autodoc::read_only_attribute_iterator::ReadOnlyAttributeIterator;

    struct AttributeList {
        values: Vec<String>,
    }
    impl ReadOnlyAttributeList<String> for AttributeList {
        fn iterator(&self) -> ReadOnlyAttributeIterator<'_, String> {
            ReadOnlyAttributeIterator::new(&self.values)
        }
    }
    #[test]
    fn source_interface_returns_read_only_attribute_iterator() {
        let list = AttributeList {
            values: vec!["Name".into()],
        };
        let mut iterator = list.iterator();
        assert_eq!(iterator.next().map(String::as_str), Some("Name"));
    }
}
