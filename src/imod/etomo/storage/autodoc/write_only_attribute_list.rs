//! `IMOD/Etomo/src/etomo/storage/autodoc/WriteOnlyAttributeList.java`.

/// Source abstract base for Autodoc/Section/Attribute attribute owners.  `T`
/// corresponds directly to Java's lexical `Token` without importing Swing UI;
/// Token is a non-GUI lexer unit despite its historical package name.
pub trait WriteOnlyAttributeList<T>: Sized {
    /// Java `addAttribute(Token, int)`.
    fn add_attribute(&mut self, name: T, line_num: i32) -> &mut Self;
    /// Java `isGlobal()`.
    fn is_global(&self) -> bool;
    /// Java `isAttribute()`.
    fn is_attribute(&self) -> bool;
}

#[cfg(test)]
mod tests {
    use super::WriteOnlyAttributeList;
    struct Owner {
        names: Vec<String>,
    }
    impl WriteOnlyAttributeList<String> for Owner {
        fn add_attribute(&mut self, name: String, _: i32) -> &mut Self {
            self.names.push(name);
            self
        }
        fn is_global(&self) -> bool {
            true
        }
        fn is_attribute(&self) -> bool {
            false
        }
    }
    #[test]
    fn abstract_source_contract_is_implementable() {
        let mut owner = Owner { names: vec![] };
        owner.add_attribute("A".into(), 9);
        assert_eq!(owner.names, ["A"]);
        assert!(owner.is_global());
        assert!(!owner.is_attribute());
    }
}
