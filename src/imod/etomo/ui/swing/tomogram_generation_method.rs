//! `IMOD/Etomo/src/etomo/ui/swing/TomogramGenerationMethod.java`.

/// Java package-private `TomogramGenerationMethod` interface.
pub trait TomogramGenerationMethod {
    /// Java `isMultifilt()`.
    fn is_multifilt(&self) -> bool;
}

#[cfg(test)]
mod tests {
    use super::TomogramGenerationMethod;
    struct Method(bool);
    impl TomogramGenerationMethod for Method {
        fn is_multifilt(&self) -> bool {
            self.0
        }
    }
    #[test]
    fn interface_preserves_multifilt_predicate() {
        assert!(Method(true).is_multifilt());
        assert!(!Method(false).is_multifilt());
    }
}
