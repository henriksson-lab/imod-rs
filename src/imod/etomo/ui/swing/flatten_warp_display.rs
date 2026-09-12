//! `IMOD/Etomo/src/etomo/ui/swing/FlattenWarpDisplay.java`.
//!
//! The Java panel holds its manager field.  The translated panel keeps it at
//! the action/validation boundary, so the same direct dependency is explicit
//! in this generic Rust interface.
#![allow(dead_code)]

use super::flatten_volume_panel::{
    FlattenVolumePanel, FlattenVolumePanelManager, FlattenWarpParamBoundary,
};

/// Java `FlattenWarpDisplay`.
pub trait FlattenWarpDisplay<M: FlattenVolumePanelManager> {
    /// Java `getParameters(FlattenWarpParam, boolean)`.
    fn get_parameters(
        &self,
        param: &mut FlattenWarpParamBoundary,
        do_validation: bool,
        manager: &mut M,
    ) -> bool;
}

impl<M: FlattenVolumePanelManager> FlattenWarpDisplay<M> for FlattenVolumePanel {
    fn get_parameters(
        &self,
        param: &mut FlattenWarpParamBoundary,
        do_validation: bool,
        manager: &mut M,
    ) -> bool {
        FlattenVolumePanel::get_parameters_flatten_warp(self, param, do_validation, manager)
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::imod::etomo::r#type::axis_id::AxisID;
    use crate::imod::etomo::r#type::dialog_type::DialogType;

    struct Manager;
    impl FlattenVolumePanelManager for Manager {
        fn property_user_dir(&self) -> &Path {
            Path::new(".")
        }
        fn trim_vol_output_file(&self, _: AxisID) -> Option<std::path::PathBuf> {
            None
        }
        fn flatten_output_file_name(&self) -> String {
            "flatten.rec".into()
        }
        fn flatten_tool_output_file_name(&self) -> String {
            "flatten.rec".into()
        }
        fn reduce_filt_vol_files(&self) -> Vec<(std::path::PathBuf, u64)> {
            Vec::new()
        }
    }

    #[test]
    fn panel_implements_canonical_flatten_warp_display() {
        let mut panel = FlattenVolumePanel::get_tools_instance(AxisID::Only, DialogType::Tools);
        panel.ltf_lambda_for_smoothing.set_text("1.5");
        panel.ltf_warp_spacing_x.set_text("2");
        panel.ltf_warp_spacing_y.set_text("3");
        let mut param = FlattenWarpParamBoundary::default();
        assert!(FlattenWarpDisplay::get_parameters(
            &panel,
            &mut param,
            true,
            &mut Manager,
        ));
        assert_eq!(param.lambda_for_smoothing, "1.5");
    }
}
