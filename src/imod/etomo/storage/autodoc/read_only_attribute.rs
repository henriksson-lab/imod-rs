//! `IMOD/Etomo/src/etomo/storage/autodoc/ReadOnlyAttribute.java`.
#![allow(dead_code)]

use super::attribute::Attribute;
use super::attribute_list::AttributeList;

/// Source read-only Autodoc attribute interface.  `getAttribute(String)` and
/// `getChildren()` are declared as `ReadOnlyAttribute`/`ReadOnlyAttributeList`, whose
/// only implementations in the package are `Attribute` and `AttributeList`, so those
/// are the concrete pointers here; `getAttribute(int)` already returns `Attribute`.
pub trait ReadOnlyAttribute {
    /// Java `getValue()`.
    fn get_value(&self) -> Option<String>;
    /// Java `getMultiLineValue()`.
    fn get_multi_line_value(&self) -> Option<String>;
    /// Java `getAttribute(String)`.
    ///
    /// # Safety
    /// The attribute's children must be live.
    unsafe fn get_attribute_by_name(&self, name: Option<&str>) -> *mut Attribute;
    /// Java `getAttribute(int)`.
    ///
    /// # Safety
    /// See `get_attribute_by_name`.
    unsafe fn get_attribute_by_index(&self, name: i32) -> *mut Attribute;
    /// Java `getName()`.
    fn get_name(&self) -> String;
    /// Java `toString()`.
    fn to_string(&self) -> String;
    /// Java `getChildren()`.
    fn get_children(&self) -> *mut AttributeList;
    /// Java `getLineNum()`.
    fn get_line_num(&self) -> i32;
}

#[cfg(test)]
mod tests {
    use super::ReadOnlyAttribute;
    use crate::imod::etomo::storage::autodoc::autodoc::Autodoc;
    use crate::imod::etomo::storage::autodoc::read_only_autodoc::ReadOnlyAutodoc;
    use crate::imod::etomo::storage::autodoc::writable_attribute::WritableAttribute;
    use crate::imod::etomo::storage::autodoc::writable_autodoc::WritableAutodoc;

    /// A duplicate attribute reports the *last* value, as `Autodoc`'s version 1.3 note
    /// describes, and a non-leaf attribute has no value of its own.
    #[test]
    fn attribute_views_follow_the_source_value_rules() {
        unsafe {
            let autodoc = Autodoc::new(Some("views"), std::ptr::null_mut());
            (*autodoc).add_name_value_pair_attribute_with_line_num(Some("Version"), Some("1.2"), 3);
            (*autodoc).add_name_value_pair_attribute_with_line_num(Some("Version"), Some("1.3"), 8);
            (*autodoc).add_name_value_pair_attribute_with_line_num(Some("a.b"), Some("deep"), 9);
            let version = (*autodoc).get_attribute(Some("VERSION"));
            assert_eq!(
                ReadOnlyAttribute::get_value(&*version).as_deref(),
                Some("1.3")
            );
            assert_eq!(ReadOnlyAttribute::get_line_num(&*version), 3);
            assert!(ReadOnlyAttribute::get_children(&*version).is_null());
            WritableAttribute::set_value(&mut *version, Some("1.4"));
            assert_eq!(
                ReadOnlyAttribute::get_value(&*version).as_deref(),
                Some("1.4")
            );
            let a = (*autodoc).get_attribute(Some("a"));
            assert_eq!(ReadOnlyAttribute::get_value(&*a), None);
            let b = ReadOnlyAttribute::get_attribute_by_name(&*a, Some("b"));
            assert_eq!(ReadOnlyAttribute::get_value(&*b).as_deref(), Some("deep"));
            assert!(ReadOnlyAttribute::get_attribute_by_index(&*a, 0).is_null());
        }
    }
}
