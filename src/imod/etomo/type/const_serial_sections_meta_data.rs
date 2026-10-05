//! `IMOD/Etomo/src/etomo/type/ConstSerialSectionsMetaData.java`.
//!
//! The read-only view of `SerialSectionsMetaData` (`serial_sections_meta_data.rs`),
//! its only implementor.  A returned field object is a copy taken under the field's
//! lock (the meta data is shared with process threads).

use std::sync::Mutex;

use super::auto_alignment_meta_data::AutoAlignmentMetaData;
use super::const_etomo_number::ConstEtomoNumber;
use super::view_type::ViewType;

/// Java `rcsid`.
pub const RCSID: &str = "$Id:$";

/// Java `public interface ConstSerialSectionsMetaData`.
pub trait ConstSerialSectionsMetaData: Send + Sync {
    /// Java `getStack()`.
    fn get_stack(&self) -> String;

    /// Java `getViewType()`.
    fn get_view_type(&self) -> Option<ViewType>;

    /// Java `getAutoAlignmentMetaData()`.
    fn get_auto_alignment_meta_data(&self) -> &Mutex<AutoAlignmentMetaData>;

    /// Java `getMidasBinning()`.
    fn get_midas_binning(&self) -> ConstEtomoNumber;

    /// Java `getReferenceSection()`.
    fn get_reference_section(&self) -> ConstEtomoNumber;

    /// Java `getRobustFitCriterion()`.
    fn get_robust_fit_criterion(&self) -> String;

    /// Java `getShiftX()`.
    fn get_shift_x(&self) -> String;

    /// Java `getShiftY()`.
    fn get_shift_y(&self) -> String;

    /// Java `getSizeX()`.
    fn get_size_x(&self) -> String;

    /// Java `getSizeY()`.
    fn get_size_y(&self) -> String;

    /// Java `isNullHybridFitsTranslations()`.
    fn is_null_hybrid_fits_translations(&self) -> bool;

    /// Java `isHybridFitsTranslations()`.
    fn is_hybrid_fits_translations(&self) -> bool;

    /// Java `isNullHybridFitsTranslationsRotations()`.
    fn is_null_hybrid_fits_translations_rotations(&self) -> bool;

    /// Java `isHybridFitsTranslationsRotations()`.
    fn is_hybrid_fits_translations_rotations(&self) -> bool;

    /// Java `isNullNoOptions()`.
    fn is_null_no_options(&self) -> bool;

    /// Java `isNoOptions()`.
    fn is_no_options(&self) -> bool;

    /// Java `isNumberToFitGlobalAlignment()`.
    fn is_number_to_fit_global_alignment(&self) -> bool;

    /// Java `getTab()`.
    fn get_tab(&self) -> i32;

    /// Java `isTabEmpty()`.
    fn is_tab_empty(&self) -> bool;

    /// Java `isUseReferenceSection()`.
    fn is_use_reference_section(&self) -> bool;

    /// Java `isPreblendVerySloppyMontage()`.
    fn is_preblend_very_sloppy_montage(&self) -> bool;

    /// Java `isBoolPreblendWeightForExpectedShifts()`.
    fn is_bool_preblend_weight_for_expected_shifts(&self) -> bool;

    /// Java `isStrPreblendWeightForExpectedShifts()`.
    fn is_str_preblend_weight_for_expected_shifts(&self) -> bool;

    /// Java `getStrPreblendWeightForExpectedShifts()`.
    fn get_str_preblend_weight_for_expected_shifts(&self) -> String;

    /// Java `isPreblendEMGridMapFilter()`.
    fn is_preblend_e_m_grid_map_filter(&self) -> bool;

    /// Java `isPreblendHighFrequencyFilterCutoff()`.
    fn is_preblend_high_frequency_filter_cutoff(&self) -> bool;

    /// Java `getPreblendHighFrequencyFilterCutoff()`.
    fn get_preblend_high_frequency_filter_cutoff(&self) -> String;

    /// Java `getOtherSumGradientFile()`.
    fn get_other_sum_gradient_file(&self) -> String;
}
