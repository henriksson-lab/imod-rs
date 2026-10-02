//! `IMOD/Etomo/src/etomo/comscript/ConstCombineParams.java`.
//!
//! setupcombine script.

use crate::imod::etomo::r#type::combine_patch_size::CombinePatchSize;
use crate::imod::etomo::r#type::const_etomo_number::ConstEtomoNumber;
use crate::imod::etomo::r#type::fiducial_match::FiducialMatch;
use crate::imod::etomo::r#type::match_mode::MatchMode;

/// Java `ConstCombineParams`.
pub trait ConstCombineParams {
    /// Java `isPatchSizeSet`.
    fn is_patch_size_set(&self, auto_final: bool) -> bool;
    /// Java `isExtraResidualTargetsSet`.
    fn is_extra_residual_targets_set(&self) -> bool;
    /// Java `isPatchBoundarySet`.  Returns true if the patch boundary values have
    /// been modified.
    fn is_patch_boundary_set(&self) -> bool;
    /// Java `isValid`.  Checks the validity of the attribute values.  Returns true
    /// if all entries are valid, otherwise the reasons are available through the
    /// method getInvalidReasons.
    fn is_valid(&self, y_and_zflipped: bool) -> bool;
    /// Java `getInvalidReasons`.  Returns the reasons the attribute values are
    /// invalid as a string array.
    fn get_invalid_reasons(&self) -> Vec<String>;
    /// Java `getMatchMode`.
    fn get_match_mode(&self) -> Option<MatchMode>;
    /// Java `isTransfer`.
    fn is_transfer(&self) -> bool;
    /// Java `getFiducialMatch`.
    fn get_fiducial_match(&self) -> Option<FiducialMatch>;
    /// Java `getUseList`.
    fn get_use_list(&self) -> String;
    /// Java `getFiducialMatchListA`.
    fn get_fiducial_match_list_a(&self) -> String;
    /// Java `getFiducialMatchListB`.
    fn get_fiducial_match_list_b(&self) -> String;
    /// Java `getPatchSize`.
    fn get_patch_size(&self, auto_final: bool) -> Option<CombinePatchSize>;
    /// Java `getPatchSizeXYZArray`.
    fn get_patch_size_xyz_array(&self, auto_final: bool) -> Option<Vec<String>>;
    /// Java `getExtraResidualTargets`.
    fn get_extra_residual_targets(&self) -> Option<String>;
    /// Java `getTempDirectory`.
    fn get_temp_directory(&self) -> String;
    /// Java `getManualCleanup`.
    fn get_manual_cleanup(&self) -> bool;
    /// Java `getPatchXMax`.
    fn get_patch_x_max(&self) -> i32;
    /// Java `getPatchXMin`.
    fn get_patch_x_min(&self) -> i32;
    /// Java `getPatchYMax`.
    fn get_patch_y_max(&self) -> i32;
    /// Java `getPatchYMin`.
    fn get_patch_y_min(&self) -> i32;
    /// Java `getPatchZMax`.
    fn get_patch_z_max(&self) -> &ConstEtomoNumber;
    /// Java `getPatchZMin`.
    fn get_patch_z_min(&self) -> &ConstEtomoNumber;
    /// Java `getMaxPatchZMax`.
    fn get_max_patch_z_max(&self) -> i32;
    /// Java `usePatchRegionModel`.  Returns true if a patch region model has been
    /// specified.
    fn use_patch_region_model(&self) -> bool;
    /// Java `getWedgeReductionFraction`.
    fn get_wedge_reduction_fraction(&self) -> Option<String>;
    /// Java `getLowFromBothRadius`.
    fn get_low_from_both_radius(&self) -> Option<String>;
    /// Java `isInitialVolumeMatching`.
    fn is_initial_volume_matching(&self) -> bool;
}
