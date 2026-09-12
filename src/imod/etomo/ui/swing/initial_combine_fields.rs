//! `IMOD/Etomo/src/etomo/ui/swing/InitialCombineFields.java`.
//!
//! The Initial and Setup combination tabs use this package-local contract for
//! the dialog's source-order field synchronization.

#![allow(dead_code)]

use crate::imod::etomo::r#type::{fiducial_match::FiducialMatch, match_mode::MatchMode};

use super::labeled_text_field::FieldValidationFailedException;

/// Java package-local `InitialCombineFields` interface.
pub trait InitialCombineFields {
    fn set_surfaces_or_models(&mut self, use_matching_models: FiducialMatch);
    fn get_surfaces_or_models(&self) -> FiducialMatch;
    fn set_bin_by_2(&mut self, bin_by_2: bool);
    fn is_bin_by_2(&self) -> bool;
    fn set_fiducial_match_list_a(&mut self, fiducial_match_list_a: &str);
    fn get_fiducial_match_list_a(
        &self,
        do_validation: bool,
    ) -> Result<String, FieldValidationFailedException>;
    fn get_fiducial_match_list_a_unvalidated(&self) -> String;
    fn set_fiducial_match_list_b(&mut self, fiducial_match_list_b: &str);
    fn get_fiducial_match_list_b(
        &self,
        do_validation: bool,
    ) -> Result<String, FieldValidationFailedException>;
    fn get_fiducial_match_list_b_unvalidated(&self) -> String;
    fn is_enabled(&self) -> bool;
    fn is_use_corresponding_points(&self) -> bool;
    fn set_use_corresponding_points(&mut self, use_points: bool);
    fn set_use_list(&mut self, use_list: &str);
    fn get_use_list(&self, do_validation: bool) -> Result<String, FieldValidationFailedException>;
    fn get_use_list_unvalidated(&self) -> String;
    fn get_match_mode(&self) -> Option<MatchMode>;
    fn set_match_mode(&mut self, match_mode: Option<MatchMode>);
    fn set_initial_volume_matching(&mut self, input: bool);
    fn is_initial_volume_matching(&self) -> bool;
}

#[cfg(test)]
mod tests {
    use super::InitialCombineFields;
    use crate::imod::etomo::{
        r#type::{dialog_type::DialogType, fiducial_match::FiducialMatch, match_mode::MatchMode},
        ui::swing::{
            initial_combine_panel::InitialCombinePanel, setup_combine_panel::SetupCombinePanel,
        },
    };

    #[test]
    fn source_synchronization_order_copies_fields_and_leaves_setup_direction_unchanged() {
        let mut initial = InitialCombinePanel::get_instance(DialogType::TomogramCombination);
        let mut setup =
            SetupCombinePanel::get_instance(DialogType::TomogramCombination, "parallel");
        initial.set_surfaces_or_models(FiducialMatch::UseModelOnly);
        initial.set_bin_by_2(true);
        initial.set_fiducial_match_list_a("1,2");
        initial.set_fiducial_match_list_b("3,4");
        initial.set_use_corresponding_points(true);
        initial.set_use_list("5,6");
        initial.set_match_mode(Some(MatchMode::BToA));
        initial.set_initial_volume_matching(true);

        if initial.is_enabled() && setup.is_enabled() {
            setup.set_surfaces_or_models(initial.get_surfaces_or_models());
            setup.set_bin_by_2(initial.is_bin_by_2());
            setup.set_fiducial_match_list_a(&initial.get_fiducial_match_list_a_unvalidated());
            setup.set_fiducial_match_list_b(&initial.get_fiducial_match_list_b_unvalidated());
            setup.set_use_corresponding_points(initial.is_use_corresponding_points());
            setup.set_use_list(&initial.get_use_list_unvalidated());
            setup.set_match_mode(initial.get_match_mode());
            setup.set_initial_volume_matching(initial.is_initial_volume_matching());
        }

        assert_eq!(setup.get_surfaces_or_models(), FiducialMatch::UseModelOnly);
        assert!(setup.is_bin_by_2());
        assert_eq!(setup.get_fiducial_match_list_a_unvalidated(), "1,2");
        assert_eq!(setup.get_fiducial_match_list_b_unvalidated(), "3,4");
        assert!(setup.is_use_corresponding_points());
        assert_eq!(setup.get_use_list_unvalidated(), "5,6");
        // Java `InitialCombinePanel.getMatchMode()` always returns null; its
        // `setMatchMode` only updates the initial-tab size-information label.
        assert_eq!(
            InitialCombineFields::get_match_mode(&setup),
            Some(MatchMode::AToB)
        );
        assert!(setup.is_initial_volume_matching());
    }
}
