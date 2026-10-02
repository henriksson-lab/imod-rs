//! `IMOD/Etomo/src/etomo/comscript/ConstTiltxcorrParam.java`.

use super::command::Command;

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `ConstTiltxcorrParam extends Command`.
pub trait ConstTiltxcorrParam: Command {
    /// Java `getAngleOffset`.
    fn get_angle_offset(&self) -> String;
    /// Java `getBordersInXandY`.
    fn get_borders_in_x_and_y(&self) -> String;
    /// Java `getExcludeCentralPeak`.
    fn get_exclude_central_peak(&self) -> bool;
    /// Java `getFilterRadius2String`.
    fn get_filter_radius2_string(&self) -> String;
    /// Java `getFilterSigma1String`.
    fn get_filter_sigma1_string(&self) -> String;
    /// Java `getFilterSigma2String`.
    fn get_filter_sigma2_string(&self) -> String;
    /// Java `getPadsInXandYString`.
    fn get_pads_in_x_and_y_string(&self) -> String;
    /// Java `getStartingEndingViews`.
    fn get_starting_ending_views(&self) -> String;
    /// Java `getTaperPercentString`.
    fn get_taper_percent_string(&self) -> String;
    /// Java `getTestOutput`.
    fn get_test_output(&self) -> Option<String>;
    /// Java `getXMaxString`.
    fn get_x_max_string(&self) -> String;
    /// Java `getXMinString`.
    fn get_x_min_string(&self) -> String;
    /// Java `getYMaxString`.
    fn get_y_max_string(&self) -> String;
    /// Java `getYMinString`.
    fn get_y_min_string(&self) -> String;
    /// Java `isAbsoluteCosineStretch`.
    fn is_absolute_cosine_stretch(&self) -> bool;
    /// Java `isCumulativeCorrelation`.
    fn is_cumulative_correlation(&self) -> bool;
    /// Java `isNoCosineStretch`.
    fn is_no_cosine_stretch(&self) -> bool;
    /// Java `getSizeOfPatchesXandY`.
    fn get_size_of_patches_x_and_y(&self) -> String;
    /// Java `getOverlapOfPatchesXandY`.
    fn get_overlap_of_patches_x_and_y(&self) -> String;
    /// Java `isOverlapOfPatchesXandYSet`.
    fn is_overlap_of_patches_x_and_y_set(&self) -> bool;
    /// Java `isSearchMagChanges`.
    fn is_search_mag_changes(&self) -> bool;
    /// Java `isNumberOfPatchesXandYSet`.
    fn is_number_of_patches_x_and_y_set(&self) -> bool;
    /// Java `getNumberOfPatchesXandY`.
    fn get_number_of_patches_x_and_y(&self) -> String;
    /// Java `getIterateCorrelations`.
    fn get_iterate_correlations(&self) -> i32;
    /// Java `getShiftLimitsXandY`.
    fn get_shift_limits_x_and_y(&self) -> String;
    /// Java `isBoundaryModelSet`.
    fn is_boundary_model_set(&self) -> bool;
    /// Java `isBordersInXandYSet`.
    fn is_borders_in_x_and_y_set(&self) -> bool;
    /// Java `isFilterRadius2Set`.
    fn is_filter_radius2_set(&self) -> bool;
    /// Java `isFilterSigma1Set`.
    fn is_filter_sigma1_set(&self) -> bool;
    /// Java `isFilterSigma2Set`.
    fn is_filter_sigma2_set(&self) -> bool;
    /// Java `getSkipViews`.
    fn get_skip_views(&self) -> String;
    /// Java `getViewsWithMagChanges`.
    fn get_views_with_mag_changes(&self) -> String;
}
