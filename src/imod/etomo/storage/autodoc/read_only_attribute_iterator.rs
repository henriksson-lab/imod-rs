//! `IMOD/Etomo/src/etomo/storage/autodoc/ReadOnlyAttributeIterator.java`.

/// Java's typed `Iterator<Attribute>` represented with a Rust generic, so the
/// read-only Autodoc interfaces can use their eventual `Attribute` type
/// without a second compatibility iterator.
pub struct ReadOnlyAttributeIterator<'a, T> {
    iterator: std::slice::Iter<'a, T>,
}

impl<'a, T> ReadOnlyAttributeIterator<'a, T> {
    /// Java package-private `ReadOnlyAttributeIterator(List<Attribute>)`.
    pub fn new(list: &'a [T]) -> Self {
        Self {
            iterator: list.iter(),
        }
    }
    /// Java `next()`; `None` is Rust's non-panicking representation of Java's
    /// exhausted-iterator exception boundary.
    pub fn next(&mut self) -> Option<&'a T> {
        self.iterator.next()
    }
    /// Java `hasNext()`.
    pub fn has_next(&self) -> bool {
        !self.iterator.as_slice().is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::ReadOnlyAttributeIterator;
    #[test]
    fn source_iterator_order_and_exhaustion_are_preserved() {
        let values = ["first", "second"];
        let mut iterator = ReadOnlyAttributeIterator::new(&values);
        assert!(iterator.has_next());
        assert_eq!(iterator.next(), Some(&"first"));
        assert_eq!(iterator.next(), Some(&"second"));
        assert!(!iterator.has_next());
        assert_eq!(iterator.next(), None);
    }
}
