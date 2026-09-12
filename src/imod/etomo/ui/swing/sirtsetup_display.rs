//! `IMOD/Etomo/src/etomo/ui/swing/SirtsetupDisplay.java`.
#![allow(dead_code)]

use super::{
    radial_panel::{RadialPanel, SirtsetupParam},
    radial_parent::RadialParent,
};

/// Java `SirtsetupDisplay`.
pub trait SirtsetupDisplay<P: SirtsetupParam> {
    fn get_parameters(&self, param: &mut P, do_validation: bool) -> bool;
}

impl<R: RadialParent, P: SirtsetupParam> SirtsetupDisplay<P> for RadialPanel<R> {
    fn get_parameters(&self, param: &mut P, do_validation: bool) -> bool {
        RadialPanel::get_parameters_sirtsetup(self, param, do_validation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Param {
        value: String,
    }
    impl SirtsetupParam for Param {
        fn set_radius_and_sigma(&mut self, _: usize, value: String) {
            self.value = value;
        }
        fn radius_and_sigma(&self, _: usize) -> String {
            self.value.clone()
        }
        fn falloff_is_true_sigma(&self) -> bool {
            false
        }
    }
    struct Display;
    impl SirtsetupDisplay<Param> for Display {
        fn get_parameters(&self, p: &mut Param, validation: bool) -> bool {
            p.value = validation.to_string();
            validation
        }
    }
    #[test]
    fn validation_parameter_is_retained() {
        assert!(Display.get_parameters(
            &mut Param {
                value: String::new()
            },
            true
        ));
    }
}
