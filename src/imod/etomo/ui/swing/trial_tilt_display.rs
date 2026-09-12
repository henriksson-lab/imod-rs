//! `IMOD/Etomo/src/etomo/ui/swing/TrialTiltDisplay.java`.
#![allow(dead_code)]

use super::{
    process_display::ProcessDisplay,
    tilt_display::{TiltDisplay, TiltDisplayException},
    trial_tilt_panel::{SplittiltParam, TrialTiltPanel, TrialTiltParam},
    trial_tilt_parent::TrialTiltParent,
};

/// Java `TrialTiltDisplay`, including the inherited `TiltDisplay` members.
pub trait TrialTiltDisplay: TiltDisplay {
    fn get_trial_tomogram_name(&self) -> String;
    fn contains_trial_tomogram_name(&self, trial_tomogram_name: &str) -> bool;
    fn add_trial_tomogram_name(&mut self, trial_tomogram_name: String);
}

impl<P: TrialTiltParent> ProcessDisplay for TrialTiltPanel<P> {}
impl<P: TrialTiltParent> TiltDisplay for TrialTiltPanel<P> {
    type TiltParam = TrialTiltParam;
    type SplittiltParam = SplittiltParam;
    fn get_parameters(
        &self,
        param: &mut Self::TiltParam,
        do_validation: bool,
    ) -> Result<bool, TiltDisplayException> {
        Ok(TrialTiltPanel::get_parameters_tilt(
            self,
            param,
            do_validation,
        ))
    }
    fn get_splittilt_parameters(
        &self,
        param: &mut Self::SplittiltParam,
        do_validation: bool,
    ) -> bool {
        TrialTiltPanel::get_parameters_splittilt(self, param, do_validation)
    }
    fn allow_tilt_com_save(&self) -> bool {
        TrialTiltPanel::allow_tilt_com_save(self)
    }
    fn set_debug(&mut self, debug: bool) {
        TrialTiltPanel::set_debug(self, debug);
    }
}
impl<P: TrialTiltParent> TrialTiltDisplay for TrialTiltPanel<P> {
    fn get_trial_tomogram_name(&self) -> String {
        TrialTiltPanel::get_trial_tomogram_name(self)
    }
    fn contains_trial_tomogram_name(&self, name: &str) -> bool {
        TrialTiltPanel::contains_trial_tomogram_name(self, name)
    }
    fn add_trial_tomogram_name(&mut self, name: String) {
        TrialTiltPanel::add_trial_tomogram_name(self, name);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Display {
        values: Vec<String>,
    }
    impl ProcessDisplay for Display {}
    impl TiltDisplay for Display {
        type TiltParam = String;
        type SplittiltParam = String;
        fn get_parameters(&self, _: &mut String, _: bool) -> Result<bool, TiltDisplayException> {
            Ok(true)
        }
        fn get_splittilt_parameters(&self, _: &mut String, _: bool) -> bool {
            true
        }
        fn allow_tilt_com_save(&self) -> bool {
            true
        }
        fn set_debug(&mut self, _: bool) {}
    }
    impl TrialTiltDisplay for Display {
        fn get_trial_tomogram_name(&self) -> String {
            self.values.first().cloned().unwrap_or_default()
        }
        fn contains_trial_tomogram_name(&self, name: &str) -> bool {
            self.values.iter().any(|value| value == name)
        }
        fn add_trial_tomogram_name(&mut self, name: String) {
            self.values.push(name)
        }
    }
    #[test]
    fn trial_name_operations_are_preserved() {
        let mut display = Display { values: vec![] };
        display.add_trial_tomogram_name("trial".into());
        assert!(display.contains_trial_tomogram_name("trial"));
        assert_eq!(display.get_trial_tomogram_name(), "trial");
    }
}
