//! `IMOD/Etomo/src/etomo/storage/autodoc/ReadOnlyAttribute.java`.

use super::read_only_attribute_list::ReadOnlyAttributeList;

/// Source read-only Autodoc attribute interface.  The associated mutable type
/// directly models Java's package-private `Attribute`; it is deliberately not
/// replaced with a parser-specific compatibility object.
pub trait ReadOnlyAttribute: Sized {
    type MutableAttribute;
    type Children: ReadOnlyAttributeList<Self>;

    /// Java `getValue()`.
    fn get_value(&self) -> String;
    /// Java `getMultiLineValue()`.
    fn get_multi_line_value(&self) -> String;
    /// Java `getAttribute(String)`.
    fn get_attribute_by_name(&self, name: &str) -> Option<&Self>;
    /// Java `getAttribute(int)`.
    fn get_attribute_by_index(&self, name: i32) -> Option<&Self::MutableAttribute>;
    /// Java `getName()`.
    fn get_name(&self) -> String;
    /// Java `toString()`.
    fn to_string(&self) -> String;
    /// Java `getChildren()`.
    fn get_children(&self) -> Option<&Self::Children>;
    /// Java `getLineNum()`.
    fn get_line_num(&self) -> i32;
}

#[cfg(test)]
mod tests {
    use super::ReadOnlyAttribute;
    use crate::imod::etomo::storage::autodoc::read_only_attribute_iterator::ReadOnlyAttributeIterator;
    use crate::imod::etomo::storage::autodoc::read_only_attribute_list::ReadOnlyAttributeList;

    struct Attribute {
        name: String,
        value: String,
        line_num: i32,
        children: Vec<Attribute>,
    }
    impl ReadOnlyAttributeList<Attribute> for Vec<Attribute> {
        fn iterator(&self) -> ReadOnlyAttributeIterator<'_, Attribute> {
            ReadOnlyAttributeIterator::new(self)
        }
    }
    impl ReadOnlyAttribute for Attribute {
        type MutableAttribute = Attribute;
        type Children = Vec<Attribute>;
        fn get_value(&self) -> String {
            self.value.clone()
        }
        fn get_multi_line_value(&self) -> String {
            self.value.clone()
        }
        fn get_attribute_by_name(&self, name: &str) -> Option<&Self> {
            self.children.iter().find(|child| child.name == name)
        }
        fn get_attribute_by_index(&self, index: i32) -> Option<&Self::MutableAttribute> {
            self.children.get(index as usize)
        }
        fn get_name(&self) -> String {
            self.name.clone()
        }
        fn to_string(&self) -> String {
            format!("{}={}", self.name, self.value)
        }
        fn get_children(&self) -> Option<&Self::Children> {
            Some(&self.children)
        }
        fn get_line_num(&self) -> i32 {
            self.line_num
        }
    }
    #[test]
    fn exposes_all_source_attribute_views() {
        let attribute = Attribute {
            name: "Root".into(),
            value: "value".into(),
            line_num: 7,
            children: vec![Attribute {
                name: "Child".into(),
                value: "child-value".into(),
                line_num: 8,
                children: vec![],
            }],
        };
        assert_eq!(
            attribute
                .get_attribute_by_name("Child")
                .unwrap()
                .get_value(),
            "child-value"
        );
        assert_eq!(
            attribute.get_attribute_by_index(0).unwrap().get_line_num(),
            8
        );
        assert_eq!(attribute.to_string(), "Root=value");
    }
}
