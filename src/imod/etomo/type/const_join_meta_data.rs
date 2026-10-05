//! `IMOD/Etomo/src/etomo/type/ConstJoinMetaData.java`.
//!
//! The read-only view of `JoinMetaData` (`join_meta_data.rs`), which implements it by
//! forwarding to its inherent methods.  A returned field object is a copy of the field
//! taken under its lock (the meta data is shared with process threads), and the
//! section table is the list of shared rows.

use std::sync::Arc;

use super::const_etomo_number::ConstEtomoNumber;
use super::etomo_boolean2::EtomoBoolean2;
use super::image_filename_style::ImageFilenameStyle;
use super::int_key_list::IntKeyList;
use super::join_state::JoinState;
use super::null_required_number_exception::NullRequiredNumberException;
use super::script_parameter::ScriptParameter;
use super::section_table_row_data::SectionTableRowData;
use super::transform::Transform;
use crate::imod::etomo::r#type::auto_alignment_meta_data::AutoAlignmentMetaData;
use std::sync::Mutex;

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `public interface ConstJoinMetaData`.
///
/// `getBoundaryRowEndListWalker` and `getBoundaryRowStartListWalker` return a
/// `IntKeyList.Walker` over the list; the list lives behind the meta data's lock, so
/// these return a copy of the list, over which the caller takes the walker.
pub trait ConstJoinMetaData: Send + Sync {
    /// Java `getImageFilenameStyle()`.
    fn get_image_filename_style(&self) -> ImageFilenameStyle;

    /// Java `getAlignmentRefSection()`.
    fn get_alignment_ref_section(&self) -> ConstEtomoNumber;

    /// Java `getBoundariesToAnalyze()`.
    fn get_boundaries_to_analyze(&self) -> Option<String>;

    /// Java `getCoordinate(ConstEtomoNumber, JoinState) throws
    /// NullRequiredNumberException`.
    fn get_coordinate(
        &self,
        coordinate: &ConstEtomoNumber,
        state: &JoinState,
    ) -> Result<i32, NullRequiredNumberException>;

    /// Java `getDatasetName()`.
    fn get_dataset_name(&self) -> String;

    /// Java `getDensityRefSection()`.
    fn get_density_ref_section(&self) -> ConstEtomoNumber;

    /// Java `isUseAlignmentRefSection()`.
    fn is_use_alignment_ref_section(&self) -> bool;

    /// Java `getShiftInX()`.
    fn get_shift_in_x(&self) -> ConstEtomoNumber;

    /// Java `getSizeInX()`.
    fn get_size_in_x(&self) -> ConstEtomoNumber;

    /// Java `getShiftInY()`.
    fn get_shift_in_y(&self) -> ConstEtomoNumber;

    /// Java `getSizeInY()`.
    fn get_size_in_y(&self) -> ConstEtomoNumber;

    /// Java `isLocalFits()`.
    fn is_local_fits(&self) -> bool;

    /// Java `getUseEveryNSlices()`.
    fn get_use_every_n_slices(&self) -> ConstEtomoNumber;

    /// Java `getRejoinUseEveryNSlices()`.
    fn get_rejoin_use_every_n_slices(&self) -> ConstEtomoNumber;

    /// Java `getTrialBinning()`.
    fn get_trial_binning(&self) -> ConstEtomoNumber;

    /// Java `getModelTransform()`.
    fn get_model_transform(&self) -> Transform;

    /// Java `getMidasLimit()`.
    fn get_midas_limit(&self) -> ConstEtomoNumber;

    /// Java `getObjectsToInclude()`.
    fn get_objects_to_include(&self) -> Option<String>;

    /// Java `getGap()`.
    /// Java declares `ConstEtomoNumber`; the object is the `EtomoBoolean2`, whose
    /// `isNull`/`is` overrides (the display value) the caller reaches.
    fn get_gap(&self) -> EtomoBoolean2;

    /// Java `getGapStart()`.
    fn get_gap_start(&self) -> ConstEtomoNumber;

    /// Java `getGapEnd()`.
    fn get_gap_end(&self) -> ConstEtomoNumber;

    /// Java `getGapInc()`.
    fn get_gap_inc(&self) -> ConstEtomoNumber;

    /// Java `getPointsToFitMin()`.
    fn get_points_to_fit_min(&self) -> ConstEtomoNumber;

    /// Java `getPointsToFitMax()`.
    fn get_points_to_fit_max(&self) -> ConstEtomoNumber;

    /// Java `getRejoinTrialBinning()`.
    fn get_rejoin_trial_binning(&self) -> ConstEtomoNumber;

    /// Java `getBoundaryRowEnd(int)`; null is `None`.
    fn get_boundary_row_end(&self, key: i32) -> Option<ConstEtomoNumber>;

    /// Java `isBoundaryRowEndListEmpty()`.
    fn is_boundary_row_end_list_empty(&self) -> bool;

    /// Java `getSectionTableData()` (an `ArrayList` of `SectionTableRowData`, or null).
    fn get_section_table_data(&self) -> Option<Vec<Arc<SectionTableRowData>>>;

    /// Java `getBoundaryRowEndListWalker()`: the list the walker walks (see the trait
    /// documentation).
    fn get_boundary_row_end_list(&self) -> IntKeyList;

    /// Java `getBoundaryRowStartListWalker()`: the list the walker walks (see the trait
    /// documentation).
    fn get_boundary_row_start_list(&self) -> IntKeyList;

    /// Java `getSizeInXParameter()`.
    fn get_size_in_x_parameter(&self) -> ScriptParameter;

    /// Java `getSizeInYParameter()`.
    fn get_size_in_y_parameter(&self) -> ScriptParameter;

    /// Java `getShiftInXParameter()`.
    fn get_shift_in_x_parameter(&self) -> ScriptParameter;

    /// Java `getShiftInYParameter()`.
    fn get_shift_in_y_parameter(&self) -> ScriptParameter;

    /// Java `getRejoinTrialBinningParameter()`.
    fn get_rejoin_trial_binning_parameter(&self) -> ScriptParameter;

    /// Java `getTrialBinningParameter()`.
    fn get_trial_binning_parameter(&self) -> ScriptParameter;

    /// Java `getAutoAlignmentMetaData()`: the shared object.
    fn get_auto_alignment_meta_data(&self) -> &Mutex<AutoAlignmentMetaData>;

    /// Java `getName()`.
    fn get_name(&self) -> String;

    /// Java `getDensityRefSectionParameter()`.
    fn get_density_ref_section_parameter(&self) -> ScriptParameter;
}
