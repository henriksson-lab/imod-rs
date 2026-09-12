//! `IMOD/Etomo/src/etomo/ui/swing/BeadTrackDisplay.java`.
//!
//! Java stores the `ApplicationManager` in each concrete Swing display.  The
//! translated `BeadtrackPanel` deliberately keeps that GUI/process owner at an
//! explicit call boundary, so the equivalent Rust contract carries it as a
//! generic method argument.  `BeadtrackParam` stays the source-visible
//! parameter object and no widget policy is introduced here.

#![allow(dead_code)]

use crate::imod::etomo::comscript::fortran_input_syntax_exception::FortranInputSyntaxException;
use crate::imod::etomo::r#type::invalid_etomo_number_exception::InvalidEtomoNumberException;

use super::beadtrack_panel::{BeadtrackPanel, BeadtrackPanelApplicationManager, BeadtrackParam};

/// The two checked exceptions on Java `BeadTrackDisplay.getParameters`.
#[derive(Debug)]
pub enum BeadTrackDisplayException {
    /// Java `FortranInputSyntaxException`.
    FortranInputSyntaxException(FortranInputSyntaxException),
    /// Java `InvalidEtomoNumberException`.
    InvalidEtomoNumberException(InvalidEtomoNumberException),
}

impl std::fmt::Display for BeadTrackDisplayException {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FortranInputSyntaxException(exception) => exception.fmt(formatter),
            Self::InvalidEtomoNumberException(exception) => exception.fmt(formatter),
        }
    }
}

impl std::error::Error for BeadTrackDisplayException {}

/// Java `BeadTrackDisplay`.
pub trait BeadTrackDisplay<P: BeadtrackParam, M: BeadtrackPanelApplicationManager<P>> {
    /// Java `getParameters(BeadtrackParam, boolean)`.
    fn get_parameters(
        &self,
        manager: &mut M,
        beadtrack_params: &mut P,
        do_validation: bool,
    ) -> Result<bool, BeadTrackDisplayException>;
}

impl<P: BeadtrackParam, M: BeadtrackPanelApplicationManager<P>> BeadTrackDisplay<P, M>
    for BeadtrackPanel
{
    fn get_parameters(
        &self,
        manager: &mut M,
        beadtrack_params: &mut P,
        do_validation: bool,
    ) -> Result<bool, BeadTrackDisplayException> {
        BeadtrackPanel::get_parameters(self, manager, beadtrack_params, do_validation).map_err(
            |message| {
                BeadTrackDisplayException::InvalidEtomoNumberException(
                    InvalidEtomoNumberException::new(&message),
                )
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
    use crate::imod::etomo::r#type::axis_id::AxisID;
    use crate::imod::etomo::r#type::dialog_type::DialogType;
    use crate::imod::etomo::ui::swing::beadtrack_panel::BeadtrackField;
    use crate::imod::etomo::ui::swing::multi_line_button::MultiLineButton;

    #[derive(Default)]
    struct Param(BTreeMap<BeadtrackField, String>);

    impl BeadtrackParam for Param {
        fn get(&self, field: BeadtrackField) -> Option<String> {
            self.0.get(&field).cloned()
        }

        fn set(&mut self, field: BeadtrackField, value: String) -> Result<(), String> {
            self.0.insert(field, value);
            Ok(())
        }
    }

    struct Manager;

    impl BeadtrackPanelApplicationManager<Param> for Manager {
        fn stack_binning(&self, _axis_id: AxisID) -> String {
            "2".into()
        }

        fn fiducial_model_track(
            &mut self,
            _axis_id: AxisID,
            _button: &MultiLineButton,
            _dialog_type: DialogType,
        ) {
        }

        fn make_fiducial_model_seed_model(&mut self, _axis_id: AxisID) -> bool {
            false
        }

        fn imod_fix_fiducials(
            &mut self,
            _axis_id: AxisID,
            _options: Option<Run3dmodMenuOptions>,
            _button: &MultiLineButton,
            _skip_list: Option<String>,
        ) {
        }

        fn open_message_dialog(&mut self, _message: String, _title: &str, _axis_id: AxisID) {}

        fn pack(&mut self, _axis_id: AxisID) {}
    }

    #[test]
    fn beadtrack_panel_implements_the_display_contract() {
        let panel = BeadtrackPanel::get_instance(AxisID::Only, DialogType::FiducialModel);
        let mut manager = Manager;
        let mut parameters = Param::default();

        assert!(
            BeadTrackDisplay::get_parameters(&panel, &mut manager, &mut parameters, false).unwrap()
        );
        assert_eq!(
            parameters.0.get(&BeadtrackField::ImagesAreBinned),
            Some(&"2".to_string())
        );
    }

    #[test]
    fn exception_sum_preserves_both_java_checked_exception_kinds() {
        let exception = BeadTrackDisplayException::FortranInputSyntaxException(
            FortranInputSyntaxException::new("bad input"),
        );
        assert_eq!(exception.to_string(), "bad input");
        let exception = BeadTrackDisplayException::InvalidEtomoNumberException(
            InvalidEtomoNumberException::new("bad number"),
        );
        assert_eq!(exception.to_string(), "bad number");
    }
}
