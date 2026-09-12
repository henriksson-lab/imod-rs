//! `IMOD/Etomo/src/etomo/ui/swing/FindBeads3dDisplay.java`.
//!
//! Java stores the application manager and the 3d-find parent in each
//! `FindBeads3dPanel`.  The panel translation deliberately exposes those two
//! source fields at its Rust call boundary, so this interface carries them as
//! type parameters instead of inventing a second controller.
#![allow(dead_code)]

use super::find_beads3d_panel::{
    FindBeads3dPanel, FindBeads3dPanelApplicationManager, FindBeads3dParam,
};
use super::newstack_or_blendmont_3d_find_parent::NewstackOrBlendmont3dFindParent;

/// Java `FindBeads3dDisplay`.
pub trait FindBeads3dDisplay<P, M, T>
where
    P: FindBeads3dParam,
    M: FindBeads3dPanelApplicationManager,
    T: NewstackOrBlendmont3dFindParent,
{
    /// Java `getParameters(FindBeads3dParam, boolean)`.
    fn get_parameters(&self, manager: &M, param: &mut P, do_validation: bool) -> bool;

    /// Java `isFiducialess`.
    fn is_fiducialess(&self, parent: &T) -> bool;
}

impl<P, M, T> FindBeads3dDisplay<P, M, T> for FindBeads3dPanel
where
    P: FindBeads3dParam,
    M: FindBeads3dPanelApplicationManager,
    T: NewstackOrBlendmont3dFindParent,
{
    fn get_parameters(&self, manager: &M, param: &mut P, do_validation: bool) -> bool {
        FindBeads3dPanel::get_parameters(self, manager, param, do_validation)
    }

    fn is_fiducialess(&self, parent: &T) -> bool {
        FindBeads3dPanel::is_fiducialess(self, parent)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
    use crate::imod::etomo::r#type::axis_id::AxisID;
    use crate::imod::etomo::r#type::dialog_type::DialogType;
    use crate::imod::etomo::ui::swing::find_beads3d_panel::FindBeads3dField;

    #[derive(Default)]
    struct Param(BTreeMap<FindBeads3dField, String>);

    impl super::super::find_beads3d_panel::ConstFindBeads3dParam for Param {
        fn value(&self, field: FindBeads3dField) -> Option<String> {
            self.0.get(&field).cloned()
        }
        fn storage_threshold(&self) -> Option<i32> {
            None
        }
    }
    impl FindBeads3dParam for Param {
        fn set_value(&mut self, field: FindBeads3dField, value: String) -> Result<(), String> {
            self.0.insert(field, value);
            Ok(())
        }
        fn set_input_file(&mut self, _value: &str) {}
        fn set_output_file(&mut self, _value: String) {}
        fn set_storage_threshold_number(&mut self, _value: i32) {}
    }

    struct Manager;
    impl FindBeads3dPanelApplicationManager for Manager {
        fn calc_unbinned_bead_diameter_pixels(&self) -> String {
            "1".into()
        }
        fn find_beads3d_output_model_file_name(&self, _axis_id: AxisID) -> String {
            "beads.mod".into()
        }
        fn find_beads3d(
            &mut self,
            _: &super::super::find_beads3d_panel::FindBeads3dProcessButton,
            _: Option<&super::super::find_beads3d_panel::FindBeads3dViewButton>,
            _: AxisID,
            _: Option<Run3dmodMenuOptions>,
            _: DialogType,
        ) {
        }
        fn imod_find_beads3d(
            &mut self,
            _: AxisID,
            _: Option<Run3dmodMenuOptions>,
            _: &str,
            _: String,
            _: DialogType,
        ) {
        }
        fn pack(&mut self, _: AxisID) {}
    }
    struct Parent;
    impl NewstackOrBlendmont3dFindParent for Parent {
        fn get_bead_size(&self) -> String {
            "10".into()
        }
        fn is_fiducialess(&self) -> bool {
            true
        }
    }

    #[test]
    fn panel_implements_canonical_display_contract() {
        let panel = FindBeads3dPanel::get_instance(AxisID::Only, DialogType::FinalAlignedStack);
        assert!(FindBeads3dDisplay::<Param, Manager, Parent>::is_fiducialess(&panel, &Parent));
        let mut param = Param::default();
        assert!(
            FindBeads3dDisplay::<Param, Manager, Parent>::get_parameters(
                &panel, &Manager, &mut param, false,
            )
        );
    }
}
