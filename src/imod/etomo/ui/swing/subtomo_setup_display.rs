//! `IMOD/Etomo/src/etomo/ui/swing/SubtomoSetupDisplay.java`.
#![allow(dead_code)]

use super::subtomograms_panel::SubtomoSetupParam;

/// Java `SubtomoSetupDisplay`.
///
/// `SubtomogramsPanel` keeps the Java-derived rootname and processor-number
/// inputs explicit.  It is intentionally not adapted into hidden panel state.
pub trait SubtomoSetupDisplay<P: SubtomoSetupParam> {
    fn get_parameters(&self, param: &mut P, do_validation: bool) -> bool;
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct Param;
    impl SubtomoSetupParam for Param {
        fn set(&mut self, _: &str, _: String) {}
        fn reset(&mut self, _: &str) {}
        fn is_set(&self, _: &str) -> bool {
            false
        }
        fn get(&self, _: &str) -> Option<String> {
            None
        }
    }
    struct Display;
    impl SubtomoSetupDisplay<Param> for Display {
        fn get_parameters(&self, _: &mut Param, validation: bool) -> bool {
            validation
        }
    }
    #[test]
    fn parameter_and_validation_match_java_contract() {
        assert!(Display.get_parameters(&mut Param, true));
    }
}
