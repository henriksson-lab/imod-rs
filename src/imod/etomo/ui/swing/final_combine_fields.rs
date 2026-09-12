//! `IMOD/Etomo/src/etomo/ui/swing/FinalCombineFields.java`.
//!
//! The two tab panels implement this package-local synchronization contract.
//! Java `String` parameters are borrowed here because each source setter copies
//! its value into a text widget; getters retain the source-owned widget value.

#![allow(dead_code)]

/// Java package-local `FinalCombineFields` interface.
pub trait FinalCombineFields {
    fn set_use_patch_region_model(&mut self, use_patch_region_model: bool);
    fn is_use_patch_region_model(&self) -> bool;
    fn set_x_min(&mut self, x_min: &str);
    fn get_x_min(&self) -> String;
    fn set_x_max(&mut self, x_max: &str);
    fn get_x_max(&self) -> String;
    fn set_y_min(&mut self, y_min: &str);
    fn get_y_min(&self) -> String;
    fn set_y_max(&mut self, y_max: &str);
    fn get_y_max(&self) -> String;
    fn set_z_min(&mut self, z_min: &str);
    fn get_z_min(&self) -> String;
    fn set_z_max(&mut self, z_max: &str);
    fn get_z_max(&self) -> String;
    fn set_parallel(&mut self, parallel: bool);
    fn is_parallel(&self) -> bool;
    fn set_parallel_enabled(&mut self, parallel_enabled: bool);
    fn is_parallel_enabled(&self) -> bool;
    fn set_no_volcombine(&mut self, no_volcombine: bool);
    fn is_no_volcombine(&self) -> bool;
    fn is_enabled(&self) -> bool;
}

#[cfg(test)]
mod tests {
    use super::FinalCombineFields;
    use crate::imod::etomo::{
        r#type::dialog_type::DialogType,
        ui::swing::{
            final_combine_panel::FinalCombinePanel, setup_combine_panel::SetupCombinePanel,
        },
    };

    #[test]
    fn final_and_setup_panels_share_every_synchronized_field() {
        let mut setup =
            SetupCombinePanel::get_instance(DialogType::TomogramCombination, "parallel");
        let mut final_panel = FinalCombinePanel::new(DialogType::TomogramCombination, "parallel");
        setup.set_use_patch_region_model(true);
        setup.set_x_min("1");
        setup.set_x_max("2");
        setup.set_y_min("3");
        setup.set_y_max("4");
        setup.set_z_min("5");
        setup.set_z_max("6");
        setup.set_parallel(true);
        setup.set_parallel_enabled(false);
        setup.set_no_volcombine(true);

        if setup.is_enabled() && final_panel.is_enabled() {
            final_panel.set_use_patch_region_model(setup.is_use_patch_region_model());
            final_panel.set_x_min(&setup.get_x_min());
            final_panel.set_x_max(&setup.get_x_max());
            final_panel.set_y_min(&setup.get_y_min());
            final_panel.set_y_max(&setup.get_y_max());
            final_panel.set_z_min(&setup.get_z_min());
            final_panel.set_z_max(&setup.get_z_max());
            final_panel.set_parallel(setup.is_parallel());
            final_panel.set_parallel_enabled(setup.is_parallel_enabled());
            final_panel.set_no_volcombine(setup.is_no_volcombine());
        }

        assert!(final_panel.is_use_patch_region_model());
        assert_eq!(final_panel.get_x_min(), "1");
        assert_eq!(final_panel.get_x_max(), "2");
        assert_eq!(final_panel.get_y_min(), "3");
        assert_eq!(final_panel.get_y_max(), "4");
        assert_eq!(final_panel.get_z_min(), "5");
        assert_eq!(final_panel.get_z_max(), "6");
        assert!(final_panel.is_parallel());
        assert!(!final_panel.is_parallel_enabled());
        assert!(final_panel.is_no_volcombine());
    }

    #[test]
    fn final_enabled_reflects_the_dialog_tab_boundary() {
        let mut final_panel = FinalCombinePanel::new(DialogType::TomogramCombination, "parallel");
        final_panel.pnl_root.final_tab_enabled = false;

        assert!(!final_panel.is_enabled());
    }
}
