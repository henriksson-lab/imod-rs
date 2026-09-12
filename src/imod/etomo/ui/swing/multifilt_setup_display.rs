//! `IMOD/Etomo/src/etomo/ui/swing/MultifiltSetupDisplay.java`.
#![allow(dead_code)]

use super::multifilt_panel::{MultifiltPanel, MultifiltSetupParam};

/// Java `MultifiltSetupDisplay`.
pub trait MultifiltSetupDisplay<P: MultifiltSetupParam> {
    fn get_parameters(&self, param: &mut P, do_validation: bool) -> bool;
}

impl<P: MultifiltSetupParam> MultifiltSetupDisplay<P> for MultifiltPanel {
    fn get_parameters(&self, param: &mut P, do_validation: bool) -> bool {
        MultifiltPanel::get_parameters(self, param, do_validation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Display;
    struct Parameter;
    impl MultifiltSetupParam for Parameter {
        fn is_fake_sirt_iterations(&self) -> bool {
            false
        }
        fn fake_sirt_iterations(&self) -> String {
            String::new()
        }
        fn is_exact_object_sizes(&self) -> bool {
            false
        }
        fn exact_object_sizes(&self) -> String {
            String::new()
        }
        fn is_gaussian_cutoffs(&self) -> bool {
            false
        }
        fn gaussian_cutoffs(&self) -> String {
            String::new()
        }
        fn is_gaussian_falloffs(&self) -> bool {
            false
        }
        fn gaussian_falloffs(&self) -> String {
            String::new()
        }
        fn is_hamming_like_starts(&self) -> bool {
            false
        }
        fn hamming_like_starts(&self) -> String {
            String::new()
        }
        fn width_in_x(&self) -> String {
            String::new()
        }
        fn shift_in_x(&self) -> String {
            String::new()
        }
        fn size_in_y(&self) -> String {
            String::new()
        }
        fn shift_in_y(&self) -> String {
            String::new()
        }
        fn thickness_in_z(&self) -> String {
            String::new()
        }
        fn shift_in_depth(&self) -> String {
            String::new()
        }
        fn set_fake_sirt_iterations(&mut self, _: String) {}
        fn reset_fake_sirt_iterations(&mut self) {}
        fn set_exact_object_sizes(&mut self, _: String) {}
        fn reset_exact_object_sizes(&mut self) {}
        fn set_gaussian_cutoffs(&mut self, _: String) {}
        fn reset_gaussian_cutoffs(&mut self) {}
        fn set_gaussian_falloffs(&mut self, _: String) {}
        fn reset_gaussian_falloffs(&mut self) {}
        fn set_hamming_like_starts(&mut self, _: String) {}
        fn reset_hamming_like_starts(&mut self) {}
        fn set_width_in_x(&mut self, _: String) {}
        fn set_shift_in_x(&mut self, _: String) {}
        fn set_size_in_y(&mut self, _: String) {}
        fn set_shift_in_y(&mut self, _: String) {}
        fn set_thickness_in_z(&mut self, _: String) {}
        fn set_shift_in_depth(&mut self, _: String) {}
    }
    impl MultifiltSetupDisplay<Parameter> for Display {
        fn get_parameters(&self, _: &mut Parameter, validation: bool) -> bool {
            validation
        }
    }
    #[test]
    fn validation_argument_is_not_lost() {
        assert!(Display.get_parameters(&mut Parameter, true));
    }
}
