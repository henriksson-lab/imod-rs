//! `IMOD/Etomo/src/etomo/ui/swing/InitialCombineFields.java`.
//!
//! Package-local interface to the initial combine screen fields, implemented
//! by the Setup tab (`SetupCombinePanel`) and the Initial Match tab
//! (`InitialCombinePanel`); `TomogramCombinationDialog.synchronize` copies
//! these fields from one to the other.  The implementers are EDT objects, so
//! every method takes `&self`.
//!
//! Java `String` values are `Option<String>` (Java null); the overloaded
//! getters carry the `ui.md` parameter-type suffixes.

use crate::imod::etomo::r#type::fiducial_match::FiducialMatch;
use crate::imod::etomo::r#type::match_mode::MatchMode;
use crate::imod::etomo::ui::field_validation_failed_exception::FieldValidationFailedException;

/// Java package-local `interface InitialCombineFields`.
pub trait InitialCombineFields {
    /// Java `setSurfacesOrModels(FiducialMatch)`.
    fn set_surfaces_or_models(&self, use_matching_models: FiducialMatch);

    /// Java `getSurfacesOrModels()`.
    fn get_surfaces_or_models(&self) -> FiducialMatch;

    /// Java `setBinBy2(boolean)`.
    fn set_bin_by2(&self, bin_by2: bool);

    /// Java `isBinBy2()`.
    fn is_bin_by2(&self) -> bool;

    /// Java `setFiducialMatchListA(String)`.
    fn set_fiducial_match_list_a(&self, fiducial_match_list_a: Option<&str>);

    /// Java `getFiducialMatchListA(boolean) throws FieldValidationFailedException`.
    fn get_fiducial_match_list_a_boolean(
        &self,
        do_validation: bool,
    ) -> Result<Option<String>, FieldValidationFailedException>;

    /// Java `getFiducialMatchListA()`.
    fn get_fiducial_match_list_a_void(&self) -> Option<String>;

    /// Java `setFiducialMatchListB(String)`.
    fn set_fiducial_match_list_b(&self, fiducial_match_list_b: Option<&str>);

    /// Java `getFiducialMatchListB(boolean) throws FieldValidationFailedException`.
    fn get_fiducial_match_list_b_boolean(
        &self,
        do_validation: bool,
    ) -> Result<Option<String>, FieldValidationFailedException>;

    /// Java `getFiducialMatchListB()`.
    fn get_fiducial_match_list_b_void(&self) -> Option<String>;

    /// Java `isEnabled()`.
    fn is_enabled(&self) -> bool;

    /// Java `isUseCorrespondingPoints()`.
    fn is_use_corresponding_points(&self) -> bool;

    /// Java `setUseCorrespondingPoints(boolean)`.
    fn set_use_corresponding_points(&self, use_: bool);

    /// Java `setUseList(String)`.
    fn set_use_list(&self, use_list: Option<&str>);

    /// Java `getUseList(boolean) throws FieldValidationFailedException`.
    fn get_use_list_boolean(
        &self,
        do_validation: bool,
    ) -> Result<Option<String>, FieldValidationFailedException>;

    /// Java `getUseList()`.
    fn get_use_list_void(&self) -> Option<String>;

    /// Java `getMatchMode()`; `None` is Java null.
    fn get_match_mode(&self) -> Option<MatchMode>;

    /// Java `setMatchMode(MatchMode)`; `None` is Java null.
    fn set_match_mode(&self, match_mode: Option<MatchMode>);

    /// Java `setInitialVolumeMatching(boolean)`.
    fn set_initial_volume_matching(&self, input: bool);

    /// Java `isInitialVolumeMatching()`.
    fn is_initial_volume_matching(&self) -> bool;
}
