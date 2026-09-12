//! `IMOD/Etomo/src/etomo/ui/swing/AltStackDisplay.java`.
//!
//! This is the source's small interface between the alternative-stack process
//! path and its panel.  `TiltParam` and `AltTomoSetupParam` remain explicit
//! translated-boundary values; the overloads are named by their parameter type
//! because Rust does not support Java method overloading.
#![allow(dead_code)]

use super::alt_stack_panel::{AltStackPanel, AltStackTiltParamBoundary, AltTomoSetupParamBoundary};
use super::process_display::ProcessDisplay;
use crate::imod::etomo::r#type::axis_id::AxisID;

/// Java `AltStackDisplay`.
pub trait AltStackDisplay: ProcessDisplay {
    /// Java `getParameters(TiltParam)`.
    fn get_parameters_tilt(&mut self, tilt_param: &mut AltStackTiltParamBoundary) -> bool;

    /// Java `getAxisID`; `None` preserves Java's null result for Both axes.
    fn get_axis_id(&self) -> Option<AxisID>;

    /// Java `getParameters(AltTomoSetupParam, boolean)`.
    fn get_parameters_alt_tomo_setup(
        &self,
        param: &mut AltTomoSetupParamBoundary,
        do_validation: bool,
    ) -> bool;
}

impl ProcessDisplay for AltStackPanel {}

impl AltStackDisplay for AltStackPanel {
    fn get_parameters_tilt(&mut self, tilt_param: &mut AltStackTiltParamBoundary) -> bool {
        AltStackPanel::get_parameters_tilt(self, tilt_param)
    }

    fn get_axis_id(&self) -> Option<AxisID> {
        AltStackPanel::get_axis_id(self)
    }

    fn get_parameters_alt_tomo_setup(
        &self,
        param: &mut AltTomoSetupParamBoundary,
        do_validation: bool,
    ) -> bool {
        AltStackPanel::get_parameters(self, param, do_validation)
    }
}

#[cfg(test)]
mod tests {
    use super::AltStackDisplay;
    use crate::imod::etomo::r#type::axis_id::AxisID;
    use crate::imod::etomo::r#type::dialog_type::DialogType;
    use crate::imod::etomo::ui::swing::alt_stack_panel::{
        AltStackPanel, AltStackTiltParamBoundary, AltStackTomogramState, AltTomoSetupParamBoundary,
    };

    #[test]
    fn panel_implements_the_alt_stack_display_contract() {
        let mut panel = AltStackPanel::get_instance(
            AxisID::Only,
            DialogType::TomogramGeneration,
            false,
            false,
            false,
            AltStackTomogramState::default(),
        );
        panel.ltf_rootname_of_alt_stack.set_text("alternate");
        let display: &mut dyn AltStackDisplay = &mut panel;
        let mut tilt_param = AltStackTiltParamBoundary::default();
        let mut setup_param = AltTomoSetupParamBoundary::default();

        assert!(!display.get_parameters_tilt(&mut tilt_param));
        assert_eq!(display.get_axis_id(), Some(AxisID::Only));
        assert!(display.get_parameters_alt_tomo_setup(&mut setup_param, true));
        assert_eq!(
            setup_param.rootname_to_process.as_deref(),
            Some("alternate")
        );
    }
}
