//! `IMOD/Etomo/src/etomo/ui/swing/AbstractRadioButtonModel.java`.
//!
//! `JToggleButton.ToggleButtonModel` is a Swing-owned model and remains at the GUI
//! boundary.  Java's abstract superclass is represented by a Rust trait: an associated
//! type keeps each concrete radio-button model tied to the particular
//! `EnumeratedType` implementation it returns, while `Option` preserves Java's
//! nullable reference result.
#![allow(dead_code)]

/// Boundary for Java `JToggleButton.ToggleButtonModel`.
///
/// Selection, event notification, and ButtonGroup coordination are Swing operations;
/// concrete GUI adapters implement this marker when they wrap that Java model.
pub trait ToggleButtonModelBoundary {}

/// Java `AbstractRadioButtonModel`.
///
/// The Java superclass extends `JToggleButton.ToggleButtonModel` and declares only
/// `getEnumeratedType`.  The associated type is the Rust equivalent of Java's
/// `EnumeratedType` interface reference; it avoids erasing a concrete translated enum
/// behind an ad-hoc conversion boundary.
pub trait AbstractRadioButtonModel: ToggleButtonModelBoundary {
    type EnumeratedType;

    /// Java `getEnumeratedType`.
    fn get_enumerated_type(&self) -> Option<&Self::EnumeratedType>;
}

#[cfg(test)]
mod tests {
    use super::{AbstractRadioButtonModel, ToggleButtonModelBoundary};

    #[derive(Debug, Eq, PartialEq)]
    struct TestEnumeratedType(&'static str);

    struct TestRadioButtonModel {
        enumerated_type: Option<TestEnumeratedType>,
    }

    impl ToggleButtonModelBoundary for TestRadioButtonModel {}

    impl AbstractRadioButtonModel for TestRadioButtonModel {
        type EnumeratedType = TestEnumeratedType;

        fn get_enumerated_type(&self) -> Option<&Self::EnumeratedType> {
            self.enumerated_type.as_ref()
        }
    }

    #[test]
    fn get_enumerated_type_preserves_a_non_null_java_reference() {
        let model = TestRadioButtonModel {
            enumerated_type: Some(TestEnumeratedType("selected")),
        };

        assert_eq!(
            model.get_enumerated_type(),
            Some(&TestEnumeratedType("selected"))
        );
    }

    #[test]
    fn get_enumerated_type_preserves_a_null_java_reference() {
        let model = TestRadioButtonModel {
            enumerated_type: None,
        };

        assert_eq!(model.get_enumerated_type(), None);
    }
}
