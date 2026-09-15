//! `IMOD/Etomo/src/etomo/storage/autodoc/ReadOnlyAttributeIterator.java`.

use super::attribute::Attribute;

/// Java's typed `Iterator<Attribute>`.  The source's list contains references;
/// the Rust list owns `Box<Attribute>` values and this iterator exposes a borrowed
/// stable address for compatibility with the remaining read-only interface.
pub struct ReadOnlyAttributeIterator<'a> {
    iterator: std::slice::Iter<'a, Box<Attribute>>,
    current: *mut Attribute,
}

impl<'a> ReadOnlyAttributeIterator<'a> {
    /// Java package-private `ReadOnlyAttributeIterator(List<Attribute>)`.
    pub fn new(list: &'a [Box<Attribute>]) -> Self {
        Self {
            iterator: list.iter(),
            current: std::ptr::null_mut(),
        }
    }
    /// Java `next()`; `None` is Rust's non-panicking representation of Java's
    /// exhausted-iterator exception boundary.
    pub fn next(&mut self) -> Option<&*mut Attribute> {
        let attribute = self.iterator.next()?;
        self.current = attribute.as_ref() as *const Attribute as *mut Attribute;
        Some(&self.current)
    }
    /// Java `hasNext()`.
    pub fn has_next(&self) -> bool {
        !self.iterator.as_slice().is_empty()
    }
}
