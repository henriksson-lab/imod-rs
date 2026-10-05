//! `IMOD/Etomo/src/etomo/storage/MatlabParam.java`.
//!
//! The PEET parameter file (`.prm`), a Matlab-syntax autodoc.  prmParser doesn't
//! like `dPhi = {:}`, `szVol = [, , ]`, `outsideMaskRadius =`.
//!
//! **Representation.**  The parsed values are the translated `Parsed*` family (owned
//! values; where Java hands the same element object to a list, the list receives a
//! copy, see `ParsedElement::clone_element`).  `userCommands` is either a
//! `ParsedQuotedString` or a `ParsedList` behind the `Parsable` interface.  The class
//! is `synchronized` on `read`/`write`; the manager keeps it behind a lock, which
//! is the same exclusion.  The inner classes `InitMotlCode`, `MaskType`,
//! `SampleSphere` and `YAxisType` are Rust enums implementing `EnumeratedType`;
//! `Volume`, `SearchAngleArea` and `Iteration` are structs.
//!
//! **Autodoc boundary.**  The autodoc layer hands out raw pointers (`*mut Autodoc`,
//! `*mut Attribute`), owned by `AutodocFactory`; they are dereferenced here only
//! while the factory keeps them.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::storage::autodoc::attribute::Attribute;
use crate::imod::etomo::storage::autodoc::autodoc::Autodoc;
use crate::imod::etomo::storage::autodoc::autodoc_factory;
use crate::imod::etomo::storage::autodoc::read_only_attribute::ReadOnlyAttribute;
use crate::imod::etomo::storage::autodoc::read_only_autodoc::ReadOnlyAutodoc;
use crate::imod::etomo::storage::autodoc::read_only_section_list::ReadOnlySectionList;
use crate::imod::etomo::storage::autodoc::statement::{Statement, Type as StatementType};
use crate::imod::etomo::storage::autodoc::writable_attribute::WritableAttribute;
use crate::imod::etomo::storage::autodoc::writable_autodoc::WritableAutodoc;
use crate::imod::etomo::storage::log_file::{LogFile, LogFileError};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::{
    ConstEtomoNumber, Number, Type, java_lang_string_matches_whitespace,
};
use crate::imod::etomo::r#type::enumerated_type::EnumeratedType;
use crate::imod::etomo::r#type::etomo_autodoc;
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::r#type::parsable::Parsable;
use crate::imod::etomo::r#type::parsed_array::ParsedArray;
use crate::imod::etomo::r#type::parsed_array_descriptor::ParsedArrayDescriptor;
use crate::imod::etomo::r#type::parsed_descriptor::ParsedDescriptor;
use crate::imod::etomo::r#type::parsed_element::ParsedElement;
use crate::imod::etomo::r#type::parsed_list::{self, ParsedList};
use crate::imod::etomo::r#type::parsed_number::ParsedNumber;
use crate::imod::etomo::r#type::parsed_quoted_string::{self, ParsedQuotedString};
use crate::imod::etomo::ui::shared_strings;
use crate::imod::etomo::ui::swing::ui_harness;
use crate::imod::etomo::ui::ui_component::UIComponent;
use crate::imod::etomo::util::dataset_files;
use crate::imod::etomo::util::utilities;

/// Java `REFERENCE_KEY`.
pub const REFERENCE_KEY: &str = "reference";
/// Java `REFERENCE_FLG_FAIR_REFERENCE_GROUPS_DEFAULT`.
pub const REFERENCE_FLG_FAIR_REFERENCE_GROUPS_DEFAULT: i32 = 10;
/// Java `REFERENCE_FLG_FAIR_REFERENCE_PARTICLES_DEFAULT`.
pub const REFERENCE_FLG_FAIR_REFERENCE_PARTICLES_DEFAULT: i32 = 10;
/// Java `FN_VOLUME_KEY`.
pub const FN_VOLUME_KEY: &str = "fnVolume";
/// Java `FN_MOD_PARTICLE_KEY`.
pub const FN_MOD_PARTICLE_KEY: &str = "fnModParticle";
/// Java `TILT_RANGE_KEY`.
pub const TILT_RANGE_KEY: &str = "tiltRange";
/// Java private static final `RELATIVE_ORIENT_KEY` (deprecated).
const RELATIVE_ORIENT_KEY: &str = "relativeOrient";
/// Java `SZ_VOL_KEY`.
pub const SZ_VOL_KEY: &str = "szVol";
/// Java `X_INDEX`.
pub const X_INDEX: i32 = 0;
/// Java `Y_INDEX`.
pub const Y_INDEX: i32 = 1;
/// Java `Z_INDEX`.
pub const Z_INDEX: i32 = 2;
/// Java `FN_OUTPUT_KEY`.
pub const FN_OUTPUT_KEY: &str = "fnOutput";
/// Java `D_PHI_KEY`.
pub const D_PHI_KEY: &str = "dPhi";
/// Java `D_THETA_KEY`.
pub const D_THETA_KEY: &str = "dTheta";
/// Java `D_PSI_KEY`.
pub const D_PSI_KEY: &str = "dPsi";
/// Java `SEARCH_RADIUS_KEY`.
pub const SEARCH_RADIUS_KEY: &str = "searchRadius";
/// Java `LOW_CUTOFF_KEY`.
pub const LOW_CUTOFF_KEY: &str = "lowCutoff";
/// Java `LOW_CUTOFF_DEFAULT`.
pub const LOW_CUTOFF_DEFAULT: &str = "0";
/// Java `LOW_CUTOFF_SIGMA_DEFAULT`.
pub const LOW_CUTOFF_SIGMA_DEFAULT: &str = "0.05";
/// Java `DOUBLE_LOW_CUTOFF_DEFAULT`.
pub const DOUBLE_LOW_CUTOFF_DEFAULT: f64 = 0.0;
/// Java `DOUBLE_LOW_CUTOFF_SIGMA_DEFAULT`.
pub const DOUBLE_LOW_CUTOFF_SIGMA_DEFAULT: f64 = 0.05;
/// Java `HI_CUTOFF_KEY`.
pub const HI_CUTOFF_KEY: &str = "hiCutoff";
/// Java `CC_MODE_KEY` (deprecated).
pub const CC_MODE_KEY: &str = "CCMode";
/// Java `REF_THRESHOLD_KEY`.
pub const REF_THRESHOLD_KEY: &str = "refThreshold";
/// Java `REF_FLAG_ALL_TOM_KEY`.
pub const REF_FLAG_ALL_TOM_KEY: &str = "refFlagAllTom";
/// Java `EDGE_SHIFT_KEY`.
pub const EDGE_SHIFT_KEY: &str = "edgeShift";
/// Java `EDGE_SHIFT_DEFAULT`.
pub const EDGE_SHIFT_DEFAULT: i32 = 1;
/// Java `EDGE_SHIFT_MIN`.
pub const EDGE_SHIFT_MIN: i32 = 0;
/// Java `EDGE_SHIFT_MAX`.
pub const EDGE_SHIFT_MAX: i32 = 3;
/// Java `LST_THRESHOLDS_KEY`.
pub const LST_THRESHOLDS_KEY: &str = "lstThresholds";
/// Java `LST_FLAG_ALL_TOM_KEY`.
pub const LST_FLAG_ALL_TOM_KEY: &str = "lstFlagAllTom";
/// Java `ALIGNED_BASE_NAME_KEY`.
pub const ALIGNED_BASE_NAME_KEY: &str = "alignedBaseName";
/// Java `DEBUG_LEVEL_KEY`.
pub const DEBUG_LEVEL_KEY: &str = "debugLevel";
/// Java `DEBUG_LEVEL_MIN`.
pub const DEBUG_LEVEL_MIN: i32 = 0;
/// Java `DEBUG_LEVEL_MAX`.
pub const DEBUG_LEVEL_MAX: i32 = 3;
/// Java `DEBUG_LEVEL_DEFAULT`.
pub const DEBUG_LEVEL_DEFAULT: i32 = 3;
/// Java `PARTICLE_PER_CPU_KEY`.
pub const PARTICLE_PER_CPU_KEY: &str = "particlePerCPU";
/// Java `PARTICLE_PER_CPU_MIN`.
pub const PARTICLE_PER_CPU_MIN: i32 = 1;
/// Java `PARTICLE_PER_CPU_MAX`.
pub const PARTICLE_PER_CPU_MAX: i32 = 9999;
/// Java `PARTICLE_PER_CPU_DEFAULT`.
pub const PARTICLE_PER_CPU_DEFAULT: i32 = 20;
/// Java `YAXIS_CONTOUR_KEY` (deprecated; replaced by yaxisObject and yaxisContour).
pub const YAXIS_CONTOUR_KEY: &str = "yaxisContour";
/// Java `YAXIS_OBJECT_NUM_KEY`.
pub const YAXIS_OBJECT_NUM_KEY: &str = "yaxisObjectNum";
/// Java `YAXIS_CONTOUR_NUM_KEY`.
pub const YAXIS_CONTOUR_NUM_KEY: &str = "yaxisContourNum";
/// Java `FLG_WEDGE_WEIGHT_KEY`.
pub const FLG_WEDGE_WEIGHT_KEY: &str = "flgWedgeWeight";
/// Java `FLG_WEDGE_WEIGHT_DEFAULT`.
pub const FLG_WEDGE_WEIGHT_DEFAULT: bool = false;
/// Java `SAMPLE_INTERVAL_KEY`.
pub const SAMPLE_INTERVAL_KEY: &str = "sampleInterval";
/// Java `MASK_TYPE_KEY`.
pub const MASK_TYPE_KEY: &str = "maskType";
/// Java `MASK_MODEL_PTS_KEY`.
pub const MASK_MODEL_PTS_KEY: &str = "maskModelPts";
/// Java `INSIDE_MASK_RADIUS_KEY`.
pub const INSIDE_MASK_RADIUS_KEY: &str = "insideMaskRadius";
/// Java `OUTSIDE_MASK_RADIUS_KEY`.
pub const OUTSIDE_MASK_RADIUS_KEY: &str = "outsideMaskRadius";
/// Java `N_WEIGHT_GROUP_KEY`.
pub const N_WEIGHT_GROUP_KEY: &str = "nWeightGroup";
/// Java `N_WEIGHT_GROUP_DEFAULT`.
pub const N_WEIGHT_GROUP_DEFAULT: i32 = 8;
/// Java `N_WEIGHT_GROUP_OFF`.
pub const N_WEIGHT_GROUP_OFF: i32 = 0;
/// Java `N_WEIGHT_GROUP_MIN`.
pub const N_WEIGHT_GROUP_MIN: i32 = 0;
/// Java `N_WEIGHT_GROUP_MAX`.
pub const N_WEIGHT_GROUP_MAX: i32 = 32;
/// Java `FLG_REMOVE_DUPLICATES_KEY`.
pub const FLG_REMOVE_DUPLICATES_KEY: &str = "flgRemoveDuplicates";
/// Java `DUPLICATE_SHIFT_TOLERANCE_KEY`.
pub const DUPLICATE_SHIFT_TOLERANCE_KEY: &str = "duplicateShiftTolerance";
/// Java `DUPLICATE_ANGULAR_TOLERANCE_KEY`.
pub const DUPLICATE_ANGULAR_TOLERANCE_KEY: &str = "duplicateAngularTolerance";
/// Java `FLG_ALIGN_AVERAGES_KEY`.
pub const FLG_ALIGN_AVERAGES_KEY: &str = "flgAlignAverages";
/// Java `FLG_FAIR_REFERENCE_KEY`.
pub const FLG_FAIR_REFERENCE_KEY: &str = "flgFairReference";
/// Java `FLG_ABS_VALUE_KEY`.
pub const FLG_ABS_VALUE_KEY: &str = "flgAbsValue";
/// Java `FLG_ABS_VALUE_DEFAULT`.
pub const FLG_ABS_VALUE_DEFAULT: bool = true;
/// Java `FLG_STRICT_SEARCH_LIMITS_KEY`.
pub const FLG_STRICT_SEARCH_LIMITS_KEY: &str = "flgStrictSearchLimits";
/// Java `FLG_STRICT_SEARCH_LIMITS_DEFAULT`.
pub const FLG_STRICT_SEARCH_LIMITS_DEFAULT: bool = false;
/// Java `SELECT_CLASS_ID_KEY`.
pub const SELECT_CLASS_ID_KEY: &str = "selectClassID";
/// Java `FLG_NO_REFERENCE_REFINEMENT_KEY`.
pub const FLG_NO_REFERENCE_REFINEMENT_KEY: &str = "flgNoReferenceRefinement";
/// Java `FLG_RANDOMIZE_KEY`.
pub const FLG_RANDOMIZE_KEY: &str = "flgRandomize";
/// Java `CYLINDER_HEIGHT_KEY`.
pub const CYLINDER_HEIGHT_KEY: &str = "cylinderHeight";
/// Java `MASK_BLUR_STD_DEV_KEY`.
pub const MASK_BLUR_STD_DEV_KEY: &str = "maskBlurStdDev";
/// Java `FLG_VOL_NAMES_ARE_TEMPLATES_KEY`.
pub const FLG_VOL_NAMES_ARE_TEMPLATES_KEY: &str = "flgVolNamesAreTemplates";
/// Java `EXCLUDE_LIST_KEY`.
pub const EXCLUDE_LIST_KEY: &str = "excludeList";
/// Java `INCLUDE_LIST_KEY`.
pub const INCLUDE_LIST_KEY: &str = "includeList";
/// Java `FLG_ELEVATION_COMPENSATION_KEY`.
pub const FLG_ELEVATION_COMPENSATION_KEY: &str = "flgElevationCompensation";
/// Java `FLG_FRM_KEY`.
pub const FLG_FRM_KEY: &str = "flgFRM";
/// Java `FLG_ALLOW_MASKED_CORRELATION_KEY`.
pub const FLG_ALLOW_MASKED_CORRELATION_KEY: &str = "flgAllowMaskedCorrelation";
/// Java `FLG_FILTER_REF_ONLY_KEY`.
pub const FLG_FILTER_REF_ONLY_KEY: &str = "flgFilterRefOnly";
/// Java `FLG_SEARCH_ALONG_PARTICLE_AXES_KEY`.
pub const FLG_SEARCH_ALONG_PARTICLE_AXES_KEY: &str = "flgSearchAlongParticleAxes";
/// Java `FLG_FP_WEDGE_MASK_KEY`.
pub const FLG_FP_WEDGE_MASK_KEY: &str = "flgFPWedgeMask";
/// Java `Y_AXIS_SYMMETRY_KEY`.
pub const Y_AXIS_SYMMETRY_KEY: &str = "yAxisSymmetry";
/// Java `FLG_USE_EXTRACTED_PARTICLES_KEY`.
pub const FLG_USE_EXTRACTED_PARTICLES_KEY: &str = "flgUseExtractedParticles";
/// Java `CN_SYMMETRIC_AVERAGING_KEY`.
pub const CN_SYMMETRIC_AVERAGING_KEY: &str = "cNSymmetricAveraging";
/// Java `CN_SYMMETRIC_AVERAGING_DEFAULT`.
pub const CN_SYMMETRIC_AVERAGING_DEFAULT: i32 = 2;
/// Java `CN_SYMMETRIC_AVERAGING_MIN`.
pub const CN_SYMMETRIC_AVERAGING_MIN: i32 = 2;
/// Java `CN_SYMMETRIC_AVERAGING_MAX`.
pub const CN_SYMMETRIC_AVERAGING_MAX: i32 = 32;
/// Java `CN_SYMMETRIC_AVERAGING_STEP`.
pub const CN_SYMMETRIC_AVERAGING_STEP: i32 = 1;
/// Java `FLG_CN_MASKING_KEY`.
pub const FLG_CN_MASKING_KEY: &str = "flgCNMasking";
/// Java `USER_COMMANDS_KEY`.
pub const USER_COMMANDS_KEY: &str = "userCommands";

/// Java private static final `VOLUME_INDEX`.
const VOLUME_INDEX: i32 = 0;
/// Java private static final `PARTICLE_INDEX`.
const PARTICLE_INDEX: i32 = 1;
/// Java private static final `LEVEL_INDEX`.
const LEVEL_INDEX: i32 = 0;
/// Java private static final `Z_ROTATION_INDEX`.
const Z_ROTATION_INDEX: i32 = 0;
/// Java private static final `Y_ROTATION_INDEX`.
const Y_ROTATION_INDEX: i32 = 1;
/// Java private static final `WRAP_MIN_LENGTH`.
const WRAP_MIN_LENGTH: i32 = 72;
/// Java private static final `WRAP_LENGTH`.
const WRAP_LENGTH: i32 = 80;

/// Java private static final `SQUIGGLY_BRACKET = ParsedList.OPEN_SYMBOL.toString()`.
fn squiggly_bracket() -> String {
    parsed_list::OPEN_SYMBOL.to_string()
}

/// Java private static final `QUOTE = ParsedQuotedString.DELIMITER_SYMBOL.toString()`.
fn quote() -> String {
    parsed_quoted_string::DELIMITER_SYMBOL.to_string()
}

/// Java private static final `DIVIDER = ParsedList.DIVIDER_SYMBOL.toString()`.
fn divider() -> String {
    parsed_list::DIVIDER_SYMBOL.to_string()
}

/// Java private static final `STRING_DIVIDER = QUOTE + ParsedList.DIVIDER_SYMBOL`.
fn string_divider() -> String {
    format!("{}{}", quote(), parsed_list::DIVIDER_SYMBOL)
}

/// The attribute `name` of a read-only autodoc, as the `ReadOnlyAttribute` Java's
/// `autodoc.getAttribute(name)` returns (null when absent).
///
/// # Safety
/// `autodoc` must be a live autodoc from `AutodocFactory`.
unsafe fn get_attribute<'a>(autodoc: *mut Autodoc, name: &str) -> Option<&'a dyn ReadOnlyAttribute> {
    unsafe {
        let attribute: *mut Attribute = (*autodoc).get_attribute(Some(name));
        attribute.as_ref().map(|attribute| attribute as &dyn ReadOnlyAttribute)
    }
}

/// Java `public final class MatlabParam`.
pub struct MatlabParam {
    /// Java private final `particlePerCpu`.
    particle_per_cpu: ParsedNumber,
    /// Java private final `szVol`.
    sz_vol: ParsedArray,
    /// Java private final `fnOutput`.
    fn_output: ParsedQuotedString,
    /// Java private final `refFlagAllTom`.
    ref_flag_all_tom: ParsedNumber,
    /// Java private final `edgeShift`.
    edge_shift: ParsedNumber,
    /// Java private final `lstThresholds`.
    lst_thresholds: ParsedArray,
    /// Java private final `lstFlagAllTom`.
    lst_flag_all_tom: ParsedNumber,
    /// Java private final `alignedBaseName`.
    aligned_base_name: ParsedQuotedString,
    /// Java private final `debugLevel`.
    debug_level: ParsedNumber,
    /// Java private final `volumeList`.
    volume_list: Vec<Volume>,
    /// Java private final `iterationList`.
    iteration_list: Vec<Iteration>,
    /// Java private final `referenceFile`.
    reference_file: ParsedQuotedString,
    /// Java private final `reference`.
    reference: ParsedArray,
    /// Java private final `yaxisObjectNum`.
    yaxis_object_num: ParsedNumber,
    /// Java private final `yaxisContourNum`.
    yaxis_contour_num: ParsedNumber,
    /// Java private final `flgWedgeWeight`.
    flg_wedge_weight: ParsedNumber,
    /// Java private final `sampleSphere`.
    sample_sphere: ParsedQuotedString,
    /// Java private final `sampleInterval`.
    sample_interval: ParsedNumber,
    /// Java private final `maskType`.
    mask_type: ParsedQuotedString,
    /// Java private final `maskModelPts`.
    mask_model_pts: ParsedArray,
    /// Java private final `insideMaskRadius`.
    inside_mask_radius: ParsedNumber,
    /// Java private final `outsideMaskRadius`.
    outside_mask_radius: ParsedNumber,
    /// Java private final `nWeightGroup`.
    n_weight_group: ParsedNumber,
    /// Java private final `flgRemoveDuplicates`.
    flg_remove_duplicates: ParsedNumber,
    /// Java private final `flgAlignAverages`.
    flg_align_averages: ParsedNumber,
    /// Java private final `flgFairReference`.
    flg_fair_reference: ParsedNumber,
    /// Java private final `flgAbsValue`.
    flg_abs_value: ParsedNumber,
    /// Java private final `flgStrictSearchLimits`.
    flg_strict_search_limits: ParsedNumber,
    /// Java private final `flgRandomize`.
    flg_randomize: ParsedNumber,
    /// Java private final `cylinderHeight`.
    cylinder_height: ParsedNumber,
    /// Java private final `maskBlurStdDev`.
    mask_blur_std_dev: ParsedNumber,
    /// Java private final `flgVolNamesAreTemplates`.
    flg_vol_names_are_templates: ParsedNumber,
    /// Java private final `bcSelectClassID` (deprecated: backwards compatibility for
    /// selectClassID, which used to be a number).
    bc_select_class_id: ParsedNumber,
    /// Java private final `selectClassID`.
    select_class_id: ParsedArray,
    /// Java private final `flgNoReferenceRefinement`.
    flg_no_reference_refinement: ParsedNumber,
    /// Java private final `excludeList`.
    exclude_list: ParsedArray,
    /// Java private final `includeList`.
    include_list: ParsedArray,
    /// Java private final `flgElevationCompensation`.
    flg_elevation_compensation: ParsedNumber,
    /// Java private final `flgFRM`.
    flg_frm: ParsedNumber,
    /// Java private final `flgAllowMaskedCorrelation`.
    flg_allow_masked_correlation: ParsedNumber,
    /// Java private final `flgFilterRefOnly`.
    flg_filter_ref_only: ParsedNumber,
    /// Java private final `flgSearchAlongParticleAxes`.
    flg_search_along_particle_axes: ParsedNumber,
    /// Java private final `flgFPWedgeMask`.
    flg_fp_wedge_mask: ParsedNumber,
    /// Java private final `yAxisSymmetry`.
    y_axis_symmetry: ParsedArray,
    /// Java private final `flgUseExtractedParticles`.
    flg_use_extracted_particles: ParsedNumber,
    /// Java private final `cNSymmetricAveraging`.
    c_n_symmetric_averaging: ParsedNumber,
    /// Java private final `flgCNMasking`.
    flg_cn_masking: ParsedNumber,

    /// Java private final `manager`.
    manager: Option<&'static dyn BaseManager>,
    /// Java private final `axisID`.
    axis_id: AxisID,

    /// Java private `lowCutoff`.
    low_cutoff: String,
    /// Java private `lowCutoffSigma`.
    low_cutoff_sigma: String,
    /// Java private `initMotlCode` (null when initial motive list files are used).
    init_motl_code: Option<InitMotlCode>,
    /// Java private `useReferenceFile`.
    use_reference_file: bool,
    /// Java private `yAxisType`.
    y_axis_type: YAxisType,
    /// Java private `tiltRangeEmpty`.
    tilt_range_empty: bool,
    /// Java private `isTiltRangeMultiAxes`.
    is_tilt_range_multi_axes: bool,
    /// Java private `userCommands`, initially null.
    user_commands: Option<Box<dyn Parsable + Send + Sync>>,

    /// Java private `newFile`.
    new_file: bool,
    /// Java private `file`.
    file: PathBuf,
}

// `ParsedList` and `ParsedQuotedString` are the two `Parsable`s; both are `Send +
// Sync` (their elements are `ParsedElement`s).
impl Parsable for ParsedQuotedStringParsable {
    fn clear_parsable(&mut self) {
        self.0.clear_parsable()
    }
    fn parse_string(&mut self, parsable_string: Option<&str>) {
        self.0.parse_string(parsable_string)
    }
    fn validate_parsable(&self) -> Option<String> {
        self.0.validate_parsable()
    }
    fn get_parsable_string_parsable(&self) -> Option<String> {
        self.0.get_parsable_string_parsable()
    }
    fn parse_attribute(&mut self, attribute: Option<&dyn ReadOnlyAttribute>) {
        self.0.parse_attribute(attribute)
    }
    fn is_empty_parsable(&self) -> bool {
        self.0.is_empty_parsable()
    }
    fn size_parsable(&self) -> i32 {
        self.0.size_parsable()
    }
}

/// `userCommands` holding a `ParsedQuotedString` (the trait object needs a sized
/// owner; this is a transparent wrapper, not a source class).
struct ParsedQuotedStringParsable(ParsedQuotedString);

impl MatlabParam {
    /// Java `MatlabParam(BaseManager, AxisID, File, boolean)`.
    pub fn new(
        manager: Option<&'static dyn BaseManager>,
        axis_id: AxisID,
        file: &Path,
        new_file: bool,
    ) -> MatlabParam {
        let mut instance = MatlabParam {
            particle_per_cpu: ParsedNumber::get_matlab_instance(Some(PARTICLE_PER_CPU_KEY)),
            sz_vol: ParsedArray::get_matlab_instance(Some(SZ_VOL_KEY)),
            fn_output: ParsedQuotedString::get_instance(Some(FN_OUTPUT_KEY)),
            ref_flag_all_tom: ParsedNumber::get_matlab_instance(Some(REF_FLAG_ALL_TOM_KEY)),
            edge_shift: ParsedNumber::get_matlab_instance(Some(EDGE_SHIFT_KEY)),
            lst_thresholds: ParsedArray::get_matlab_instance(Some(LST_THRESHOLDS_KEY)),
            lst_flag_all_tom: ParsedNumber::get_matlab_instance(Some(LST_FLAG_ALL_TOM_KEY)),
            aligned_base_name: ParsedQuotedString::get_instance(Some(ALIGNED_BASE_NAME_KEY)),
            debug_level: ParsedNumber::get_matlab_instance(Some(DEBUG_LEVEL_KEY)),
            volume_list: Vec::new(),
            iteration_list: Vec::new(),
            reference_file: ParsedQuotedString::get_instance(Some(REFERENCE_KEY)),
            reference: ParsedArray::get_matlab_instance(Some(REFERENCE_KEY)),
            yaxis_object_num: ParsedNumber::get_matlab_instance(Some(YAXIS_OBJECT_NUM_KEY)),
            yaxis_contour_num: ParsedNumber::get_matlab_instance(Some(YAXIS_CONTOUR_NUM_KEY)),
            flg_wedge_weight: ParsedNumber::get_matlab_instance(Some(FLG_WEDGE_WEIGHT_KEY)),
            sample_sphere: ParsedQuotedString::get_instance(Some(SampleSphere::KEY)),
            sample_interval: ParsedNumber::get_matlab_instance_type(
                Some(Type::Double),
                Some(SAMPLE_INTERVAL_KEY),
            ),
            mask_type: ParsedQuotedString::get_instance(Some(MASK_TYPE_KEY)),
            mask_model_pts: ParsedArray::get_matlab_instance_type(
                Some(Type::Double),
                Some(MASK_MODEL_PTS_KEY),
            ),
            inside_mask_radius: ParsedNumber::get_matlab_instance_type(
                Some(Type::Double),
                Some(INSIDE_MASK_RADIUS_KEY),
            ),
            outside_mask_radius: ParsedNumber::get_matlab_instance_type(
                Some(Type::Double),
                Some(OUTSIDE_MASK_RADIUS_KEY),
            ),
            n_weight_group: ParsedNumber::get_matlab_instance(Some(N_WEIGHT_GROUP_KEY)),
            flg_remove_duplicates: ParsedNumber::get_matlab_instance(Some(
                FLG_REMOVE_DUPLICATES_KEY,
            )),
            flg_align_averages: ParsedNumber::get_matlab_instance(Some(FLG_ALIGN_AVERAGES_KEY)),
            flg_fair_reference: ParsedNumber::get_matlab_instance(Some(FLG_FAIR_REFERENCE_KEY)),
            flg_abs_value: ParsedNumber::get_matlab_instance(Some(FLG_ABS_VALUE_KEY)),
            flg_strict_search_limits: ParsedNumber::get_matlab_instance(Some(
                FLG_STRICT_SEARCH_LIMITS_KEY,
            )),
            flg_randomize: ParsedNumber::get_matlab_instance(Some(FLG_RANDOMIZE_KEY)),
            cylinder_height: ParsedNumber::get_matlab_instance_type(
                Some(Type::Double),
                Some(CYLINDER_HEIGHT_KEY),
            ),
            mask_blur_std_dev: ParsedNumber::get_matlab_instance_type(
                Some(Type::Double),
                Some(MASK_BLUR_STD_DEV_KEY),
            ),
            flg_vol_names_are_templates: ParsedNumber::get_matlab_instance(Some(
                FLG_VOL_NAMES_ARE_TEMPLATES_KEY,
            )),
            bc_select_class_id: ParsedNumber::get_matlab_instance(Some(SELECT_CLASS_ID_KEY)),
            select_class_id: ParsedArray::get_matlab_instance(Some(SELECT_CLASS_ID_KEY)),
            flg_no_reference_refinement: ParsedNumber::get_matlab_instance(Some(
                FLG_NO_REFERENCE_REFINEMENT_KEY,
            )),
            exclude_list: ParsedArray::get_matlab_instance(Some(EXCLUDE_LIST_KEY)),
            include_list: ParsedArray::get_matlab_instance(Some(INCLUDE_LIST_KEY)),
            flg_elevation_compensation: ParsedNumber::get_matlab_instance(Some(
                FLG_ELEVATION_COMPENSATION_KEY,
            )),
            flg_frm: ParsedNumber::get_matlab_instance(Some(FLG_FRM_KEY)),
            flg_allow_masked_correlation: ParsedNumber::get_matlab_instance(Some(
                FLG_ALLOW_MASKED_CORRELATION_KEY,
            )),
            flg_filter_ref_only: ParsedNumber::get_matlab_instance(Some(FLG_FILTER_REF_ONLY_KEY)),
            flg_search_along_particle_axes: ParsedNumber::get_matlab_instance(Some(
                FLG_SEARCH_ALONG_PARTICLE_AXES_KEY,
            )),
            flg_fp_wedge_mask: ParsedNumber::get_matlab_instance(Some(FLG_FP_WEDGE_MASK_KEY)),
            y_axis_symmetry: ParsedArray::get_matlab_instance(Some(Y_AXIS_SYMMETRY_KEY)),
            flg_use_extracted_particles: ParsedNumber::get_matlab_instance(Some(
                FLG_USE_EXTRACTED_PARTICLES_KEY,
            )),
            c_n_symmetric_averaging: ParsedNumber::get_matlab_instance(Some(
                CN_SYMMETRIC_AVERAGING_KEY,
            )),
            flg_cn_masking: ParsedNumber::get_matlab_instance(Some(FLG_CN_MASKING_KEY)),
            manager,
            axis_id,
            low_cutoff: LOW_CUTOFF_DEFAULT.to_owned(),
            low_cutoff_sigma: LOW_CUTOFF_SIGMA_DEFAULT.to_owned(),
            init_motl_code: Some(InitMotlCode::DEFAULT),
            use_reference_file: false,
            y_axis_type: YAxisType::DEFAULT,
            tilt_range_empty: false,
            is_tilt_range_multi_axes: false,
            user_commands: None,
            new_file,
            file: file.to_path_buf(),
        };
        instance.n_weight_group.set_default_int(N_WEIGHT_GROUP_DEFAULT);
        instance.flg_fair_reference.set_default_boolean(false);
        instance.flg_abs_value.set_default_boolean(FLG_ABS_VALUE_DEFAULT);
        instance
            .flg_strict_search_limits
            .set_default_boolean(FLG_STRICT_SEARCH_LIMITS_DEFAULT);
        instance.edge_shift.set_default_int(EDGE_SHIFT_DEFAULT);
        instance.flg_no_reference_refinement.set_default_boolean(false);
        instance.particle_per_cpu.set_default_int(PARTICLE_PER_CPU_DEFAULT);
        instance.ref_flag_all_tom.set_default_int(1);
        instance.lst_flag_all_tom.set_default_int(1);
        instance.debug_level.set_default_int(DEBUG_LEVEL_DEFAULT);
        instance.flg_wedge_weight.set_default_boolean(FLG_WEDGE_WEIGHT_DEFAULT);
        instance.inside_mask_radius.set_default_int(0);
        instance.flg_remove_duplicates.set_default_boolean(false);
        instance.flg_align_averages.set_default_boolean(false);
        instance.flg_randomize.set_default_boolean(false);
        instance.flg_vol_names_are_templates.set_default_boolean(false);
        instance.flg_elevation_compensation.set_default_boolean(false);
        instance.flg_frm.set_default_boolean(true);
        instance.flg_allow_masked_correlation.set_default_boolean(false);
        instance.flg_filter_ref_only.set_default_boolean(false);
        instance.flg_search_along_particle_axes.set_default_boolean(false);
        instance.flg_fp_wedge_mask.set_default_boolean(false);
        instance.flg_use_extracted_particles.set_default_boolean(false);
        instance.flg_cn_masking.set_default_boolean(true);
        if let Some(user_commands) = instance.user_commands.as_mut() {
            user_commands.clear_parsable();
        }
        instance
    }

    /// Java `setFile(String)`.  Change file to newDir + fnOutput + .prm.  Also sets
    /// newFile to true.  This allows MatlabParam to read from one file and then write
    /// to another.
    pub fn set_file(&mut self, new_dir: &str) {
        self.new_file = true;
        self.file = Path::new(new_dir).join(format!(
            "{}{}",
            self.fn_output
                .get_raw_string_void()
                .unwrap_or_else(|| "null".to_owned()),
            dataset_files::MATLAB_PARAM_FILE_EXT
        ));
    }

    /// Java synchronized `read(BaseManager, List, UIComponent)`.  Reads data from the
    /// .prm autodoc.
    pub fn read(
        &mut self,
        manager: Option<&'static dyn BaseManager>,
        error_list: &mut Vec<String>,
        component: Option<&dyn UIComponent>,
    ) -> bool {
        self.clear();
        // if newFile is on, either there is no file, or the user doesn't want to read it
        if self.new_file {
            return true;
        }
        let absolute_path = utilities::java_io_file_get_absolute_path(&self.file.to_string_lossy());
        let autodoc = unsafe { autodoc_factory::get_matlab_instance(manager, Some(&self.file), false) };
        match autodoc {
            Ok(autodoc) => {
                if autodoc.is_null() {
                    ui_harness::with(|harness| {
                        harness.open_message_dialog_base_manager_string_string(
                            manager,
                            &format!("Unable to read {absolute_path}."),
                            "File Error",
                        )
                    });
                    return false;
                }
                self.parse_data(autodoc, Some(error_list), component);
                if !error_list.is_empty() {
                    return false;
                }
            }
            Err(LogFileError::Lock(_)) => return false,
            Err(LogFileError::Io(e)) => {
                ui_harness::with(|harness| {
                    harness.open_message_dialog_base_manager_string_string(
                        manager,
                        &format!("Unable to load {absolute_path}.  IOException:  {e}"),
                        "File Error",
                    )
                });
                return false;
            }
            Err(e) => {
                eprintln!("{e}");
                ui_harness::with(|harness| {
                    harness.open_message_dialog_base_manager_string_string(
                        manager,
                        &format!(
                            "Unable to read {absolute_path}.  LogFile.ReadException:  {}",
                            e.get_message()
                        ),
                        "File Error",
                    )
                });
                return false;
            }
        }
        true
    }

    /// Java synchronized `write(BaseManager)`.  Write stored data to the .prm
    /// autodoc.
    pub fn write(&mut self, manager: Option<&'static dyn BaseManager>) {
        // Place the string representation of each value in a map.
        // This allows the values to be passed to updateOrBuildAutodoc().
        // When building a new .prm autodoc, this also allows the values to be
        // accessed in the same order as the FieldInterface sections in peetprm.adoc.
        let mut value_map: HashMap<String, Option<String>> = HashMap::new();
        self.build_parsable_values(&mut value_map);
        // try to get the peetprm.adoc, which contains the comments for the .prm file
        // in its FieldInterface sections.
        let mut comment_autodoc: *mut Autodoc = std::ptr::null_mut();
        match unsafe {
            autodoc_factory::get_instance(
                manager,
                Some(autodoc_factory::PEET_PRM),
                AxisID::Only,
                false,
            )
        } {
            Ok(autodoc) => comment_autodoc = autodoc,
            Err(LogFileError::Lock(_)) => {}
            Err(LogFileError::Io(e)) => {
                eprintln!(
                    "Problem with {}.adoc.\nIOException:  {e}",
                    autodoc_factory::PEET_PRM
                );
            }
            Err(e) => {
                eprintln!("{e}");
                eprintln!(
                    "Problem with {}.adoc.\nLogFile.ReadException:  {}",
                    autodoc_factory::PEET_PRM,
                    e.get_message()
                );
            }
        }
        let result = (|| -> Result<(), LogFileError> {
            let mut autodoc =
                unsafe { autodoc_factory::get_matlab_instance(manager, Some(&self.file), true)? };
            if autodoc.is_null() {
                // get an empty .prm autodoc if the file doesn't exist
                autodoc = unsafe { autodoc_factory::get_empty_matlab_instance(manager, Some(&self.file)) };
            } else {
                let log_file = LogFile::get_instance_file(
                    Some(&self.file),
                    manager.map(|manager| manager.get_emergency_monitor(Some(AxisID::Only))),
                )?;
                if !log_file.is_backedup() {
                    log_file.double_backup_once()?;
                } else {
                    log_file.backup()?;
                }
            }
            let autodoc_ref: &mut Autodoc = unsafe { &mut *autodoc };
            if comment_autodoc.is_null() {
                // The peetprm.adoc is not available.
                // Build a new .prm autodoc with no comments
                self.update_or_build_autodoc(manager, &value_map, autodoc_ref, None);
            } else {
                // Get the FieldInterface sections from the peetprm.adoc
                let sec_loc = unsafe {
                    ReadOnlySectionList::get_section_location_by_type(
                        &*comment_autodoc,
                        Some(etomo_autodoc::FIELD_SECTION_NAME),
                    )
                };
                if sec_loc.is_none() {
                    // There are no FieldInterface sections in the peetprm.adoc.
                    // Build a new .prm autodoc with no comments
                    self.update_or_build_autodoc(manager, &value_map, autodoc_ref, None);
                } else {
                    // Build a new .prm autodoc. Use the FieldInterface sections from the
                    // peetprm.adoc to dictate the order of the name/value pairs.
                    // Also use the comments from the peetprm.adoc FieldInterface sections.
                    // This makes MatlabParam dependent on peetprm.adoc so peetprm.adoc
                    // must be the responsibility of the Etomo developer.
                    self.update_or_build_autodoc(
                        manager,
                        &value_map,
                        autodoc_ref,
                        Some(unsafe { &*comment_autodoc }),
                    );
                }
            }
            // write the autodoc file (the backup is done by autodoc)
            autodoc_ref.wrap_attribute_values(
                Some(&quote()),
                Some(&format!("{}{}", squiggly_bracket(), quote())),
                Some(&string_divider()),
                Some(&divider()),
                WRAP_MIN_LENGTH,
                WRAP_LENGTH,
            );
            autodoc_ref.write()?;
            // the file is written, so it is no longer new
            self.new_file = false;
            Ok(())
        })();
        match result {
            Ok(()) | Err(LogFileError::Lock(_)) => {}
            Err(e) => {
                let name = utilities::java_io_file_get_name(&self.file.to_string_lossy());
                ui_harness::with(|harness| {
                    harness.open_message_dialog_base_manager_string_string(
                        manager,
                        &format!("Unable to read {name}.  FileException:  {}", e.get_message()),
                        "File Error",
                    )
                });
            }
        }
    }

    /// Java `getVolume(int)`.
    pub fn get_volume(&mut self, index: i32) -> &mut Volume {
        if index as usize == self.volume_list.len() {
            self.volume_list.push(Volume::new(self.manager, self.axis_id));
        }
        &mut self.volume_list[index as usize]
    }

    /// Java `getIteration(int)`.
    pub fn get_iteration(&mut self, index: i32) -> &mut Iteration {
        if index as usize == self.iteration_list.len() {
            self.iteration_list.push(Iteration::new());
        }
        &mut self.iteration_list[index as usize]
    }

    /// Java `getFnOutput()`.
    pub fn get_fn_output(&self) -> Option<String> {
        self.fn_output.get_raw_string_void()
    }

    /// Java `getFnVolume(int)`.
    pub fn get_fn_volume(&self, index: i32) -> Option<String> {
        self.volume_list[index as usize].get_fn_volume_string()
    }

    /// Java `getFnModParticle(int)`.
    pub fn get_fn_mod_particle(&self, index: i32) -> Option<String> {
        self.volume_list[index as usize].get_fn_mod_particle_string()
    }

    /// Java `getTiltRangeMultiAxes(int)`.
    pub fn get_tilt_range_multi_axes(&self, index: i32) -> Option<String> {
        self.volume_list[index as usize].get_tilt_range_multi_axes_string()
    }

    /// Java `getVolume(int)` read-only, for the getters the source calls through
    /// `getVolume(i)` without adding a volume.
    pub fn volume(&self, index: i32) -> &Volume {
        &self.volume_list[index as usize]
    }

    /// Java `getIteration(int)` read-only.
    pub fn iteration(&self, index: i32) -> &Iteration {
        &self.iteration_list[index as usize]
    }

    /// Java `getInitMotlCode()`.
    pub fn get_init_motl_code(&self) -> Option<InitMotlCode> {
        self.init_motl_code
    }

    /// Java `getYAxisType()`.
    pub fn get_y_axis_type(&self) -> YAxisType {
        self.y_axis_type
    }

    /// Java `setInitMotlCode(EnumeratedType)`.  A null (or other) type is a null code.
    pub fn set_init_motl_code(&mut self, enumerated_type: Option<InitMotlCode>) {
        self.init_motl_code = enumerated_type;
    }

    /// Java `setYaxisType(EnumeratedType)`.
    ///
    /// Fixed in translation: a null type (no radio button selected) is a
    /// NullPointerException later in `buildParsableValues`; it keeps the current type.
    pub fn set_yaxis_type(&mut self, enumerated_type: Option<YAxisType>) {
        if let Some(enumerated_type) = enumerated_type {
            self.y_axis_type = enumerated_type;
        }
    }

    /// Java `setSampleSphere(EnumeratedType)`.
    pub fn set_sample_sphere(&mut self, enumerated_type: SampleSphere) {
        self.sample_sphere
            .set_raw_string_string(Some(&enumerated_type.to_string()));
    }

    /// Java `setMaskType(EnumeratedType)`.
    pub fn set_mask_type_enumerated_type(&mut self, enumerated_type: MaskType) {
        self.mask_type
            .set_raw_string_string(Some(&enumerated_type.to_string()));
    }

    /// Java `setFnOutput(String)`.
    pub fn set_fn_output(&mut self, fn_output: Option<&str>) {
        self.fn_output.set_raw_string_string(fn_output);
    }

    /// Java `setUserCommands(ReadOnlyAttribute)`.
    pub fn set_user_commands_attribute(&mut self, attribute: Option<&dyn ReadOnlyAttribute>) {
        if let Some(attribute) = attribute {
            self.set_user_commands(attribute.get_value().as_deref());
        } else if let Some(user_commands) = self.user_commands.as_mut() {
            user_commands.clear_parsable();
        }
    }

    /// Java `validateUserCommands()`.  Returns an error message if validation failed.
    pub fn validate_user_commands(&self) -> Option<String> {
        let Some(user_commands) = &self.user_commands else {
            return None;
        };
        if user_commands.is_empty_parsable() || user_commands.size_parsable() == 1 {
            return None;
        }
        if (user_commands.size_parsable() as usize) < self.iteration_list.len() {
            return Some("add one command for each iteration".to_owned());
        }
        if (user_commands.size_parsable() as usize) > self.iteration_list.len() {
            return Some("use only one command per iteration".to_owned());
        }
        None
    }

    /// Java `validateYAxisSymmetry()`.  Returns an error message if validation failed.
    pub fn validate_y_axis_symmetry(&self) -> Option<String> {
        if self.y_axis_symmetry.is_empty() {
            return None;
        }
        if self.y_axis_symmetry.size() as usize != self.iteration_list.len() {
            return Some("enter one value for each iteration".to_owned());
        }
        None
    }

    /// Java `setUserCommands(String)`.  Parses either a cell array or a string
    /// (optional brackets; quotes required when necessary).
    pub fn set_user_commands(&mut self, parsable_string: Option<&str>) {
        if parsable_string.is_none() && self.user_commands.is_some() {
            self.user_commands.as_mut().unwrap().clear_parsable();
            return;
        }
        let mut list = ParsedList::get_optional_bracket_string_instance(Some(USER_COMMANDS_KEY));
        list.parse_string(parsable_string);
        if list.validate_parsable().is_some()
            && !list.is_bracket_found()
            && !list.is_quote_found()
            && !ParsedList::contains_divider(parsable_string.unwrap_or(""))
        {
            // This might be a single, unquoted command, which is OK if it doesn't
            // contain any characters that can be interpretated as array delimiters
            // (commas and whitespace).
            let mut quoted_string = ParsedQuotedString::get_instance(Some(USER_COMMANDS_KEY));
            quoted_string.set_raw_string_string(parsable_string);
            self.user_commands = Some(Box::new(ParsedQuotedStringParsable(quoted_string)));
        } else if list.size_parsable() == 1 {
            // UserCommands should be treated as a string when there is only one element.
            let element = list
                .get_element(0)
                .and_then(|element| element.as_any().downcast_ref::<ParsedQuotedString>())
                .cloned();
            self.user_commands = element
                .map(|element| Box::new(ParsedQuotedStringParsable(element)) as Box<dyn Parsable + Send + Sync>);
        } else {
            self.user_commands = Some(Box::new(list));
        }
    }

    /// Java `setFlgRemoveDuplicates(boolean)`.
    pub fn set_flg_remove_duplicates(&mut self, input: bool) {
        self.flg_remove_duplicates.set_raw_string_boolean(input);
    }

    /// Java `useReferenceFile()`.
    pub fn use_reference_file(&self) -> bool {
        self.use_reference_file
    }

    /// Java `setRefFlagAllTom(boolean)`.
    pub fn set_ref_flag_all_tom(&mut self, input: bool) {
        self.ref_flag_all_tom.set_raw_string_boolean(input);
    }

    /// Java `setLstFlagAllTom(boolean)`.
    pub fn set_lst_flag_all_tom(&mut self, input: bool) {
        self.lst_flag_all_tom.set_raw_string_boolean(input);
    }

    /// Java `setFlgWedgeWeight(boolean)`.
    pub fn set_flg_wedge_weight(&mut self, input: bool) {
        self.flg_wedge_weight.set_raw_string_boolean(input);
    }

    /// Java `isMaskModelPtsEmpty()`.
    pub fn is_mask_model_pts_empty(&self) -> bool {
        self.mask_model_pts.is_empty()
    }

    /// Java `setTiltRangeEmpty()`.  If the tilt range check box is uncheck, then tilt
    /// range should be {}.
    pub fn set_tilt_range_empty(&mut self) {
        self.tilt_range_empty = true;
    }

    /// Java `setTiltRangeMultiAxes(boolean)`.
    pub fn set_tilt_range_multi_axes(&mut self, input: bool) {
        self.is_tilt_range_multi_axes = input;
    }

    /// Java `isTiltRangeMultiAxes()`.
    pub fn is_tilt_range_multi_axes(&self) -> bool {
        self.is_tilt_range_multi_axes
    }

    /// Java `isTiltRangeEmpty()`.  This is just for backwards compatibility in setting
    /// the tilt range check box.
    pub fn is_tilt_range_empty(&self) -> bool {
        if self.volume_list.is_empty() {
            return true;
        }
        for volume in &self.volume_list {
            if !volume.is_tilt_range_empty() {
                return false;
            }
        }
        true
    }

    /// Java `isFlgRemoveDuplicates()`.
    pub fn is_flg_remove_duplicates(&self) -> bool {
        self.flg_remove_duplicates.get_raw_boolean()
    }

    /// Java `isRefFlagAllTom()`.
    pub fn is_ref_flag_all_tom(&self) -> bool {
        self.ref_flag_all_tom.get_raw_boolean()
    }

    /// Java `isFlgAlignAverages()`.
    pub fn is_flg_align_averages(&self) -> bool {
        self.flg_align_averages.get_raw_boolean()
    }

    /// Java `isEmptyFlgAlignAverages()`.
    pub fn is_empty_flg_align_averages(&self) -> bool {
        self.flg_align_averages.is_empty()
    }

    /// Java `isFlgFairReference()`.
    pub fn is_flg_fair_reference(&self) -> bool {
        self.flg_fair_reference.get_raw_boolean()
    }

    /// Java `isFlgRandomize()`.
    pub fn is_flg_randomize(&self) -> bool {
        self.flg_randomize.get_raw_boolean()
    }

    /// Java `isFlgVolNamesAreTemplates()`.
    pub fn is_flg_vol_names_are_templates(&self) -> bool {
        self.flg_vol_names_are_templates.get_raw_boolean()
    }

    /// Java `isFlgNoReferenceRefinement()`.
    pub fn is_flg_no_reference_refinement(&self) -> bool {
        self.flg_no_reference_refinement.get_raw_boolean()
    }

    /// Java `isFlgAbsValue()`.
    pub fn is_flg_abs_value(&self) -> bool {
        self.flg_abs_value.get_raw_boolean()
    }

    /// Java `isFlgStrictSearchLimits()`.
    pub fn is_flg_strict_search_limits(&self) -> bool {
        self.flg_strict_search_limits.get_raw_boolean()
    }

    /// Java `resetFlgAlignAverages()`.
    pub fn reset_flg_align_averages(&mut self) {
        self.flg_align_averages.clear();
    }

    /// Java `setFlgAlignAverages(boolean)`.
    pub fn set_flg_align_averages(&mut self, input: bool) {
        self.flg_align_averages.set_raw_string_boolean(input);
    }

    /// Java `setFlgFairReference(boolean)`.
    pub fn set_flg_fair_reference(&mut self, input: bool) {
        self.flg_fair_reference.set_raw_string_boolean(input);
    }

    /// Java `setFlgRandomize(boolean)`.
    pub fn set_flg_randomize(&mut self, input: bool) {
        self.flg_randomize.set_raw_string_boolean(input);
    }

    /// Java `setFlgVolNamesAreTemplates(boolean)`.
    pub fn set_flg_vol_names_are_templates(&mut self, input: bool) {
        self.flg_vol_names_are_templates.set_raw_string_boolean(input);
    }

    /// Java `setFlgNoReferenceRefinement(boolean)`.
    pub fn set_flg_no_reference_refinement(&mut self, input: bool) {
        self.flg_no_reference_refinement.set_raw_string_boolean(input);
    }

    /// Java `setFlgAbsValue(boolean)`.
    pub fn set_flg_abs_value(&mut self, input: bool) {
        self.flg_abs_value.set_raw_string_boolean(input);
    }

    /// Java `setFlgStrictSearchLimits(boolean)`.
    pub fn set_flg_strict_search_limits(&mut self, input: bool) {
        self.flg_strict_search_limits.set_raw_string_boolean(input);
    }

    /// Java `getLstFlagAllTom()`.
    pub fn get_lst_flag_all_tom(&self) -> Option<String> {
        self.lst_flag_all_tom.get_raw_string_void()
    }

    /// Java `isLstFlagAllTom()`.
    pub fn is_lst_flag_all_tom(&self) -> bool {
        self.lst_flag_all_tom.get_raw_boolean()
    }

    /// Java `isFlgWedgeWeight()`.
    pub fn is_flg_wedge_weight(&self) -> bool {
        self.flg_wedge_weight.get_raw_boolean()
    }

    /// Java `isAlignedBaseNameEmpty()`.
    pub fn is_aligned_base_name_empty(&self) -> bool {
        self.aligned_base_name.is_empty()
    }

    /// Java `setAlignedBaseName(String)`.
    pub fn set_aligned_base_name(&mut self, aligned_base_name: Option<&str>) {
        self.aligned_base_name.set_raw_string_string(aligned_base_name);
    }

    /// Java `getAlignedBaseName()`.
    pub fn get_aligned_base_name(&self) -> Option<String> {
        self.aligned_base_name.get_raw_string_void()
    }

    /// Java `resetAlignedBaseName()`.
    pub fn reset_aligned_base_name(&mut self) {
        self.aligned_base_name.clear();
    }

    /// Java `setReferenceVolume(Number)`.
    pub fn set_reference_volume_number(&mut self, input: Number) {
        self.set_reference_volume_string(Some(&input.to_string()));
    }

    /// Java `setReferenceVolume(String)`.
    pub fn set_reference_volume_string(&mut self, input: Option<&str>) {
        self.use_reference_file = false;
        self.reference.set_raw_string_int_string(VOLUME_INDEX, input);
    }

    /// Java `setNWeightGroup(Number)`.
    pub fn set_n_weight_group(&mut self, input: Option<Number>) {
        self.n_weight_group.set_raw_string_number(input);
    }

    /// Java `setMaskModelPts(String, String)`.  Set mastModelPts.  If only one of the
    /// rotations is set, the other one should be zero.
    pub fn set_mask_model_pts(&mut self, z_rotation: Option<&str>, y_rotation: Option<&str>) {
        let z_empty = z_rotation.is_none_or(java_lang_string_matches_whitespace);
        let y_empty = y_rotation.is_none_or(java_lang_string_matches_whitespace);
        if z_empty && y_empty {
            self.mask_model_pts.clear();
            return;
        }
        let mut z_rotation = z_rotation;
        let mut y_rotation = y_rotation;
        if z_empty || y_empty {
            if z_empty {
                z_rotation = Some("0");
            } else {
                y_rotation = Some("0");
            }
        }
        self.mask_model_pts
            .set_raw_string_int_string(Z_ROTATION_INDEX, z_rotation);
        self.mask_model_pts
            .set_raw_string_int_string(Y_ROTATION_INDEX, y_rotation);
    }

    /// Java `getMaskModelPtsYRotation()`.
    pub fn get_mask_model_pts_y_rotation(&self) -> Option<String> {
        self.mask_model_pts.get_raw_string_int(Y_ROTATION_INDEX)
    }

    /// Java `getMaskModelPtsZRotation()`.
    pub fn get_mask_model_pts_z_rotation(&self) -> Option<String> {
        self.mask_model_pts.get_raw_string_int(Z_ROTATION_INDEX)
    }

    /// Java `getReferenceParticle()`.
    pub fn get_reference_particle(&self) -> Option<String> {
        self.reference.get_raw_string_int(PARTICLE_INDEX)
    }

    /// Java `getReferenceLevel()`.
    pub fn get_reference_level(&self) -> Option<String> {
        self.reference.get_raw_string_int(LEVEL_INDEX)
    }

    /// Java `getReferenceVolume()`.
    pub fn get_reference_volume(&self) -> Option<&dyn ParsedElement> {
        self.reference.get_element(VOLUME_INDEX)
    }

    /// Java `getReferenceVolumeString()`.
    pub fn get_reference_volume_string(&self) -> Option<String> {
        self.reference.get_raw_string_int(VOLUME_INDEX)
    }

    /// Java `getYaxisObjectNum()`.
    pub fn get_yaxis_object_num(&self) -> Option<String> {
        self.yaxis_object_num.get_raw_string_void()
    }

    /// Java `getYaxisContourNum()`.
    pub fn get_yaxis_contour_num(&self) -> Option<String> {
        self.yaxis_contour_num.get_raw_string_void()
    }

    /// Java `getSampleSphere(UIComponent)`.
    pub fn get_sample_sphere(&self, component: Option<&dyn UIComponent>) -> SampleSphere {
        SampleSphere::get_instance(&self.sample_sphere, component)
    }

    /// Java `getMaskType()`.
    pub fn get_mask_type(&self) -> Option<String> {
        self.mask_type.get_raw_string_void()
    }

    /// Java `setEdgeShift(Number)`.
    pub fn set_edge_shift(&mut self, edge_shift: Option<Number>) {
        self.edge_shift.set_raw_string_number(edge_shift);
    }

    /// Java `clear()`.
    pub fn clear(&mut self) {
        self.particle_per_cpu
            .set_raw_string_number(Some(Number::Integer(PARTICLE_PER_CPU_DEFAULT)));
        self.sz_vol.clear();
        self.fn_output.clear();
        self.ref_flag_all_tom
            .set_raw_string_number(Some(Number::Integer(1)));
        self.edge_shift.set_raw_string_number(Some(Number::Integer(1)));
        self.lst_thresholds.clear();
        self.lst_flag_all_tom
            .set_raw_string_number(Some(Number::Integer(1)));
        self.aligned_base_name.clear();
        self.debug_level
            .set_raw_string_number(Some(Number::Integer(DEBUG_LEVEL_DEFAULT)));
        self.volume_list.clear();
        self.iteration_list.clear();
        self.reference_file.clear();
        self.reference.clear();
        self.low_cutoff = LOW_CUTOFF_DEFAULT.to_owned();
        self.low_cutoff_sigma = LOW_CUTOFF_SIGMA_DEFAULT.to_owned();
        self.init_motl_code = Some(InitMotlCode::DEFAULT);
        self.use_reference_file = false;
        self.y_axis_type = YAxisType::DEFAULT;
        self.yaxis_object_num.clear();
        self.yaxis_contour_num.clear();
        self.flg_wedge_weight
            .set_raw_string_boolean(FLG_WEDGE_WEIGHT_DEFAULT);
        self.sample_interval.clear();
        self.sample_sphere.clear();
        self.mask_type
            .set_raw_string_string(Some(&MaskType::DEFAULT.to_string()));
        self.mask_model_pts.clear();
        self.inside_mask_radius
            .set_raw_string_number(Some(Number::Integer(0)));
        self.outside_mask_radius.clear();
        self.n_weight_group
            .set_raw_string_number(Some(Number::Integer(N_WEIGHT_GROUP_DEFAULT)));
        self.tilt_range_empty = false;
        self.flg_remove_duplicates.set_raw_string_boolean(false);
        self.flg_align_averages.set_raw_string_boolean(false);
        self.flg_fair_reference.set_raw_string_boolean(false);
        self.flg_abs_value.set_raw_string_boolean(FLG_ABS_VALUE_DEFAULT);
        self.flg_strict_search_limits
            .set_raw_string_boolean(FLG_STRICT_SEARCH_LIMITS_DEFAULT);
        self.bc_select_class_id.clear();
        self.select_class_id.clear();
        self.flg_no_reference_refinement.set_raw_string_boolean(false);
        self.cylinder_height.clear();
        self.mask_blur_std_dev.clear();
        self.exclude_list.clear();
        self.include_list.clear();
        self.flg_elevation_compensation.set_raw_string_boolean(false);
        self.flg_frm.set_raw_string_boolean(false);
        self.flg_allow_masked_correlation.set_raw_string_boolean(false);
        self.flg_filter_ref_only.set_raw_string_boolean(false);
        self.flg_search_along_particle_axes
            .set_raw_string_boolean(false);
        self.flg_fp_wedge_mask.set_raw_string_boolean(false);
        self.y_axis_symmetry.clear();
        self.flg_use_extracted_particles.set_raw_string_boolean(false);
        self.c_n_symmetric_averaging.clear();
        self.flg_cn_masking.set_raw_string_boolean(false);
        if let Some(user_commands) = self.user_commands.as_mut() {
            user_commands.clear_parsable();
        }
    }

    /// Java `clearEdgeShift()`.
    pub fn clear_edge_shift(&mut self) {
        self.edge_shift.clear();
    }

    /// Java `getEdgeShift()`.
    pub fn get_edge_shift(&self) -> &dyn ParsedElement {
        &self.edge_shift
    }

    /// Java `getSampleInterval()`.
    pub fn get_sample_interval(&self) -> Option<String> {
        self.sample_interval.get_raw_string_void()
    }

    /// Java `getCylinderHeight()`.
    pub fn get_cylinder_height(&self) -> Option<String> {
        self.cylinder_height.get_raw_string_void()
    }

    /// Java `getMaskBlurStdDev()`.
    pub fn get_mask_blur_std_dev(&self) -> Option<String> {
        self.mask_blur_std_dev.get_raw_string_void()
    }

    /// Java `getInsideMaskRadius()`.
    pub fn get_inside_mask_radius(&self) -> Option<String> {
        self.inside_mask_radius.get_raw_string_void()
    }

    /// Java `getNWeightGroup()`.
    pub fn get_n_weight_group(&self) -> &dyn ParsedElement {
        &self.n_weight_group
    }

    /// Java `isNWeightGroupEmpty()`.
    pub fn is_n_weight_group_empty(&self) -> bool {
        self.n_weight_group.is_empty()
    }

    /// Java `getOutsideMaskRadius()`.
    pub fn get_outside_mask_radius(&self) -> Option<String> {
        self.outside_mask_radius.get_raw_string_void()
    }

    /// Java `getFile()`.
    pub fn get_file(&self) -> &Path {
        &self.file
    }

    /// Java `setSampleInterval(String)`.
    pub fn set_sample_interval(&mut self, input: Option<&str>) {
        self.sample_interval.set_raw_string_string(input);
    }

    /// Java `setCylinderHeight(String)`.
    pub fn set_cylinder_height(&mut self, input: Option<&str>) {
        self.cylinder_height.set_raw_string_string(input);
    }

    /// Java `setMaskBlurStdDev(String)`.
    pub fn set_mask_blur_std_dev(&mut self, input: Option<&str>) {
        self.mask_blur_std_dev.set_raw_string_string(input);
    }

    /// Java `setInsideMaskRadius(String)`.
    pub fn set_inside_mask_radius(&mut self, input: Option<&str>) {
        self.inside_mask_radius.set_raw_string_string(input);
    }

    /// Java `setOutsideMaskRadius(String)`.
    pub fn set_outside_mask_radius(&mut self, input: Option<&str>) {
        self.outside_mask_radius.set_raw_string_string(input);
    }

    /// Java `setSzVolX(String)`.
    pub fn set_sz_vol_x(&mut self, sz_vol_x: Option<&str>) {
        self.sz_vol.set_raw_string_int_string(X_INDEX, sz_vol_x);
    }

    /// Java `setSzVolY(String)`.
    pub fn set_sz_vol_y(&mut self, sz_vol_y: Option<&str>) {
        self.sz_vol.set_raw_string_int_string(Y_INDEX, sz_vol_y);
    }

    /// Java `setSzVolZ(String)`.
    pub fn set_sz_vol_z(&mut self, sz_vol_z: Option<&str>) {
        self.sz_vol.set_raw_string_int_string(Z_INDEX, sz_vol_z);
    }

    /// Java `getLowCutoffCutoff()`.  LowCutoff is an iteration value, but it is only
    /// set once, so get the value at the first index.
    pub fn get_low_cutoff_cutoff(&self) -> Option<String> {
        if self.iteration_list.is_empty() {
            return Some(self.low_cutoff.clone());
        }
        self.iteration_list[0].get_low_cutoff_cutoff_string()
    }

    /// Java `getLowCutoffSigma()`.  As `getLowCutoffCutoff`.
    pub fn get_low_cutoff_sigma(&self) -> Option<String> {
        if self.iteration_list.is_empty() {
            return Some(self.low_cutoff_sigma.clone());
        }
        self.iteration_list[0].get_low_cutoff_sigma_string()
    }

    /// Java `setDebugLevel(Number)`.
    pub fn set_debug_level(&mut self, input: Number) {
        self.debug_level
            .set_raw_string_string(Some(&input.to_string()));
    }

    /// Java `setParticlePerCPU(Number)`.
    pub fn set_particle_per_cpu(&mut self, input: Number) {
        self.particle_per_cpu
            .set_raw_string_string(Some(&input.to_string()));
    }

    /// Java `setSelectClassID(String)`.
    pub fn set_select_class_id(&mut self, input: Option<&str>) {
        self.select_class_id.set_raw_string_string(input);
    }

    /// Java `getDebugLevel()`.
    pub fn get_debug_level(&self) -> ConstEtomoNumber {
        self.debug_level.get_etomo_number().clone()
    }

    /// Java `getParticlePerCPU()`.
    pub fn get_particle_per_cpu(&self) -> ConstEtomoNumber {
        self.particle_per_cpu.get_etomo_number().clone()
    }

    /// Java `getSelectClassID()`.
    pub fn get_select_class_id(&self) -> Option<String> {
        self.select_class_id.get_raw_string_void()
    }

    /// Java `getSzVol()`.
    pub fn get_sz_vol(&self) -> Option<String> {
        self.sz_vol.get_raw_string_void()
    }

    /// Java `getSzVolX()`.
    pub fn get_sz_vol_x(&self) -> Option<String> {
        self.sz_vol.get_raw_string_int(X_INDEX)
    }

    /// Java `getSzVolY()`.
    pub fn get_sz_vol_y(&self) -> Option<String> {
        self.sz_vol.get_raw_string_int(Y_INDEX)
    }

    /// Java `getSzVolZ()`.
    pub fn get_sz_vol_z(&self) -> Option<String> {
        self.sz_vol.get_raw_string_int(Z_INDEX)
    }

    /// Java `getReferenceFile()`.
    pub fn get_reference_file(&self) -> Option<String> {
        self.reference_file.get_raw_string_void()
    }

    /// Java `setReferenceParticle(String)`.
    pub fn set_reference_particle_string(&mut self, reference_particle: Option<&str>) {
        self.use_reference_file = false;
        self.reference
            .set_raw_string_int_string(PARTICLE_INDEX, reference_particle);
    }

    /// Java `setReferenceLevel(String)`.
    pub fn set_reference_level(&mut self, input: Option<&str>) {
        self.use_reference_file = false;
        self.reference.set_raw_string_int_string(LEVEL_INDEX, input);
    }

    /// Java `setReferenceParticle(Number)`.
    pub fn set_reference_particle_number(&mut self, input: Number) {
        self.use_reference_file = false;
        self.reference
            .set_raw_string_int_string(PARTICLE_INDEX, Some(&input.to_string()));
    }

    /// Java `clearMaskModelPts()`.
    pub fn clear_mask_model_pts(&mut self) {
        self.mask_model_pts.clear();
    }

    /// Java `setYaxisObjectNum(String)`.
    pub fn set_yaxis_object_num(&mut self, input: Option<&str>) {
        self.yaxis_object_num.set_raw_string_string(input);
    }

    /// Java `setYaxisContourNum(String)`.
    pub fn set_yaxis_contour_num(&mut self, input: Option<&str>) {
        self.yaxis_contour_num.set_raw_string_string(input);
    }

    /// Java `setReferenceFile(String)`.
    pub fn set_reference_file(&mut self, reference_file: Option<&str>) {
        self.use_reference_file = true;
        self.reference_file.set_raw_string_string(reference_file);
    }

    /// Java `setMaskType(String)`.
    pub fn set_mask_type_string(&mut self, input: Option<&str>) {
        self.mask_type.set_raw_string_string(input);
    }

    /// Java `getVolumeListSize()`.
    pub fn get_volume_list_size(&self) -> i32 {
        self.volume_list.len() as i32
    }

    /// Java `getIterationListSize()`.
    pub fn get_iteration_list_size(&self) -> i32 {
        self.iteration_list.len() as i32
    }

    /// Java `setVolumeListSize(int)`.
    ///
    /// The source's shrinking loop (`for (i = size; i < volumeList.size(); i++)
    /// volumeList.remove(i)`) removes every other element; the callers only grow the
    /// list from empty (`getParameters` after `clear()`), where it does nothing.  It
    /// is translated as written.
    pub fn set_volume_list_size(&mut self, size: i32) {
        // if volume list is too small, add new Volumes
        let mut i = self.volume_list.len() as i32;
        while i < size {
            self.volume_list.push(Volume::new(self.manager, self.axis_id));
            i += 1;
        }
        // if volume list is too big, remove Volumes from the end
        let mut i = size;
        while (i as usize) < self.volume_list.len() {
            self.volume_list.remove(i as usize);
            i += 1;
        }
    }

    /// Java `setIterationListSize(int)`.  (Same loops as `setVolumeListSize`.)
    pub fn set_iteration_list_size(&mut self, size: i32) {
        // if iteration list is too small, add new Iterations
        let mut i = self.iteration_list.len() as i32;
        while i < size {
            self.iteration_list.push(Iteration::new());
            i += 1;
        }
        // if iteration list is too big, remove Iterations from the end
        let mut i = size;
        while (i as usize) < self.iteration_list.len() {
            self.iteration_list.remove(i as usize);
            i += 1;
        }
    }

    /// Java `setLstThresholdsStart(String)`.
    pub fn set_lst_thresholds_start(&mut self, input: Option<&str>) {
        self.lst_thresholds.set_raw_string_start(input);
    }

    /// Java `setLstThresholdsIncrement(String)`.
    pub fn set_lst_thresholds_increment(&mut self, input: Option<&str>) {
        self.lst_thresholds.set_raw_string_increment(input);
    }

    /// Java `setLstThresholdsEnd(String)`.
    pub fn set_lst_thresholds_end(&mut self, input: Option<&str>) {
        self.lst_thresholds.set_raw_string_end(input);
    }

    /// Java `setLstThresholdsAdditional(String)`.
    pub fn set_lst_thresholds_additional(&mut self, input: Option<&str>) {
        self.lst_thresholds.set_raw_strings(input);
    }

    /// Java `getLstThresholdsStart()`.
    pub fn get_lst_thresholds_start(&self) -> Option<String> {
        self.lst_thresholds.get_raw_string_start()
    }

    /// Java `getLstThresholdsEnd()`.
    pub fn get_lst_thresholds_end(&self) -> Option<String> {
        self.lst_thresholds.get_raw_string_end()
    }

    /// Java `getLstThresholdsIncrement()`.
    pub fn get_lst_thresholds_increment(&self) -> Option<String> {
        self.lst_thresholds.get_raw_string_increment()
    }

    /// Java `getLstThresholdsExpandedArray()`.
    pub fn get_lst_thresholds_expanded_array(&self) -> Vec<String> {
        self.lst_thresholds.get_padded_string_expanded_array()
    }

    /// Java `getLstThresholdsAdditional()`.  Find all the numbers after the
    /// descriptor and return their values.
    pub fn get_lst_thresholds_additional(&self) -> String {
        self.lst_thresholds
            .get_raw_strings_except_first_array_descriptor()
    }

    /// Java `setExcludeList(String)`.
    pub fn set_exclude_list(&mut self, input: Option<&str>) {
        self.exclude_list.set_raw_string_string(input);
    }

    /// Java `setIncludeList(String)`.
    pub fn set_include_list(&mut self, input: Option<&str>) {
        self.include_list.set_raw_string_string(input);
    }

    /// Java `setFlgElevationCompensation(boolean)`.
    pub fn set_flg_elevation_compensation(&mut self, input: bool) {
        self.flg_elevation_compensation.set_raw_string_boolean(input);
    }

    /// Java `setFlgFRM(boolean)`.
    pub fn set_flg_frm(&mut self, input: bool) {
        self.flg_frm.set_raw_string_boolean(input);
    }

    /// Java `setFlgAllowMaskedCorrelation(boolean)`.
    pub fn set_flg_allow_masked_correlation(&mut self, input: bool) {
        self.flg_allow_masked_correlation.set_raw_string_boolean(input);
    }

    /// Java `setFlgFilterRefOnly(boolean)`.
    pub fn set_flg_filter_ref_only(&mut self, input: bool) {
        self.flg_filter_ref_only.set_raw_string_boolean(input);
    }

    /// Java `setFlgSearchAlongParticleAxes(boolean)`.
    pub fn set_flg_search_along_particle_axes(&mut self, input: bool) {
        self.flg_search_along_particle_axes
            .set_raw_string_boolean(input);
    }

    /// Java `setFlgFPWedgeMask(boolean)`.
    pub fn set_flg_fp_wedge_mask(&mut self, input: bool) {
        self.flg_fp_wedge_mask.set_raw_string_boolean(input);
    }

    /// Java `setYAxisSymmetry(String)`.
    pub fn set_y_axis_symmetry(&mut self, input: Option<&str>) {
        self.y_axis_symmetry.set_raw_string_string(input);
    }

    /// Java `setFlgUseExtractedParticles(boolean)`.
    pub fn set_flg_use_extracted_particles(&mut self, input: bool) {
        self.flg_use_extracted_particles.set_raw_string_boolean(input);
    }

    /// Java `resetCNSymmetricAveraging()`.
    pub fn reset_cn_symmetric_averaging(&mut self) {
        self.c_n_symmetric_averaging.clear();
    }

    /// Java `setCNSymmetricAveraging(Number)`.
    pub fn set_cn_symmetric_averaging(&mut self, input: Number) {
        self.c_n_symmetric_averaging
            .set_raw_string_string(Some(&input.to_string()));
    }

    /// Java `setFlgCNMasking(boolean)`.
    pub fn set_flg_cn_masking(&mut self, input: bool) {
        self.flg_cn_masking.set_raw_string_boolean(input);
    }

    /// Java `getExcludeList()`.
    pub fn get_exclude_list(&self) -> Option<String> {
        self.exclude_list.get_raw_string_void()
    }

    /// Java `getIncludeList()`.
    pub fn get_include_list(&self) -> Option<String> {
        self.include_list.get_raw_string_void()
    }

    /// Java `isFlgElevationCompensation()`.
    pub fn is_flg_elevation_compensation(&self) -> bool {
        self.flg_elevation_compensation.get_raw_boolean()
    }

    /// Java `isFlgFRM()`.
    pub fn is_flg_frm(&self) -> bool {
        self.flg_frm.get_raw_boolean()
    }

    /// Java `isFlgAllowMaskedCorrelation()`.
    pub fn is_flg_allow_masked_correlation(&self) -> bool {
        self.flg_allow_masked_correlation.get_raw_boolean()
    }

    /// Java `isFlgFilterRefOnly()`.
    pub fn is_flg_filter_ref_only(&self) -> bool {
        self.flg_filter_ref_only.get_raw_boolean()
    }

    /// Java `isFlgSearchAlongParticleAxes()`.
    pub fn is_flg_search_along_particle_axes(&self) -> bool {
        self.flg_search_along_particle_axes.get_raw_boolean()
    }

    /// Java `isFlgFPWedgeMask()`.
    pub fn is_flg_fp_wedge_mask(&self) -> bool {
        self.flg_fp_wedge_mask.get_raw_boolean()
    }

    /// Java `getYAxisSymmetry()`.
    pub fn get_y_axis_symmetry(&self) -> Option<String> {
        self.y_axis_symmetry.get_raw_string_void()
    }

    /// Java `isFlgUseExtractedParticles()`.
    pub fn is_flg_use_extracted_particles(&self) -> bool {
        self.flg_use_extracted_particles.get_raw_boolean()
    }

    /// Java `isCNSymmetricAveraging()`.
    pub fn is_cn_symmetric_averaging(&self) -> bool {
        !self.c_n_symmetric_averaging.is_empty()
    }

    /// Java `getCNSymmetricAveraging()`.
    pub fn get_cn_symmetric_averaging(&self) -> Option<String> {
        self.c_n_symmetric_averaging.get_raw_string_void()
    }

    /// Java `isFlgCNMasking()`.
    pub fn is_flg_cn_masking(&self) -> bool {
        self.flg_cn_masking.get_raw_boolean()
    }

    /// Java `getUserCommands()`.
    pub fn get_user_commands(&self) -> String {
        if let Some(user_commands) = &self.user_commands {
            return user_commands
                .get_parsable_string_parsable()
                .unwrap_or_else(|| "null".to_owned());
        }
        String::new()
    }

    /// Java private `addError(ParsedElement, List)` (and its `ParsedQuotedString`
    /// overload, whose body is the same).  Returns true if an error was found.
    fn add_error(element: &dyn ParsedElement, error_list: Option<&mut Vec<String>>) -> bool {
        let mut retval = false;
        if let Some(error_list) = error_list {
            let error = element.validate();
            if let Some(error) = error {
                error_list.push(error);
                retval = true;
            }
        }
        retval
    }

    /// Java private `addError(Parsable, List)`.  Returns true if an error was found.
    fn add_error_parsable(
        parsable: Option<&dyn Parsable>,
        error_list: Option<&mut Vec<String>>,
    ) -> bool {
        let Some(parsable) = parsable else {
            return false;
        };
        let mut retval = false;
        if let Some(error_list) = error_list {
            let error = parsable.validate_parsable();
            if let Some(error) = error {
                error_list.push(error);
                retval = true;
            }
        }
        retval
    }
}

impl MatlabParam {
    /// Java private `parseData(ReadOnlyAutodoc, List, UIComponent)`.  Called by
    /// read().  Parses data from the the file.
    fn parse_data(
        &mut self,
        autodoc: *mut Autodoc,
        mut error_list: Option<&mut Vec<String>>,
        component: Option<&dyn UIComponent>,
    ) {
        self.parse_volume_data(autodoc, error_list.as_deref_mut(), component);
        self.parse_iteration_data(autodoc, error_list.as_deref_mut(), component);
        // reference
        let attribute = unsafe { get_attribute(autodoc, REFERENCE_KEY) };
        if ParsedQuotedString::is_quoted_string_attribute(attribute) {
            self.use_reference_file = true;
            self.reference_file.parse_attribute(attribute);
            Self::add_error(&self.reference_file, error_list.as_deref_mut());
        } else {
            self.use_reference_file = false;
            self.reference.parse_attribute(attribute);
            if !self.reference.is_valid() {
                // Reference may be a single number
                let mut n_reference = ParsedNumber::get_matlab_instance(Some(REFERENCE_KEY));
                n_reference.parse_attribute(attribute);
                if n_reference.is_valid() {
                    self.reference.add_element(Box::new(n_reference));
                } else {
                    Self::add_error(&self.reference, error_list.as_deref_mut());
                }
            }
        }
        // particlePerCPU
        self.particle_per_cpu
            .parse_attribute(unsafe { get_attribute(autodoc, PARTICLE_PER_CPU_KEY) });
        Self::add_error(&self.particle_per_cpu, error_list.as_deref_mut());
        // szVol
        self.sz_vol
            .parse_attribute(unsafe { get_attribute(autodoc, SZ_VOL_KEY) });
        Self::add_error(&self.sz_vol, error_list.as_deref_mut());
        // fnOutput
        self.fn_output
            .parse_attribute(unsafe { get_attribute(autodoc, FN_OUTPUT_KEY) });
        Self::add_error(&self.fn_output, error_list.as_deref_mut());
        // refFlagAllTom
        self.ref_flag_all_tom
            .parse_attribute(unsafe { get_attribute(autodoc, REF_FLAG_ALL_TOM_KEY) });
        if !Self::add_error(&self.ref_flag_all_tom, error_list.as_deref_mut()) {
            self.check_value_expected(
                FieldRef::RefFlagAllTom,
                &[0, 1],
                component,
                REF_FLAG_ALL_TOM_KEY,
                None,
                1,
            );
        }
        // edgeShift
        self.edge_shift
            .parse_attribute(unsafe { get_attribute(autodoc, EDGE_SHIFT_KEY) });
        if !Self::add_error(&self.edge_shift, error_list.as_deref_mut()) {
            self.check_value_range(
                FieldRef::EdgeShift,
                EDGE_SHIFT_MIN,
                EDGE_SHIFT_MAX,
                1,
                component,
                EDGE_SHIFT_KEY,
                Some(shared_strings::EDGE_SHIFT_LABEL),
                Some(&EDGE_SHIFT_DEFAULT.to_string()),
            );
        }
        // lstThresholds
        self.lst_thresholds
            .parse_attribute(unsafe { get_attribute(autodoc, LST_THRESHOLDS_KEY) });
        Self::add_error(&self.lst_thresholds, error_list.as_deref_mut());
        // lstFlagAllTom
        self.lst_flag_all_tom
            .parse_attribute(unsafe { get_attribute(autodoc, LST_FLAG_ALL_TOM_KEY) });
        Self::add_error(&self.lst_flag_all_tom, error_list.as_deref_mut());
        if !Self::add_error(&self.lst_flag_all_tom, error_list.as_deref_mut()) {
            self.check_value_expected(
                FieldRef::LstFlagAllTom,
                &[0, 1],
                component,
                LST_FLAG_ALL_TOM_KEY,
                None,
                1,
            );
        }
        // alignedBaseName
        self.aligned_base_name
            .parse_attribute(unsafe { get_attribute(autodoc, ALIGNED_BASE_NAME_KEY) });
        Self::add_error(&self.aligned_base_name, error_list.as_deref_mut());
        // debugLevel
        self.debug_level
            .parse_attribute(unsafe { get_attribute(autodoc, DEBUG_LEVEL_KEY) });
        Self::add_error(&self.debug_level, error_list.as_deref_mut());
        if !Self::add_error(&self.debug_level, error_list.as_deref_mut()) {
            self.check_value_range(
                FieldRef::DebugLevel,
                DEBUG_LEVEL_MIN,
                DEBUG_LEVEL_MAX,
                1,
                component,
                DEBUG_LEVEL_KEY,
                Some(shared_strings::DEBUG_LEVEL_LABEL),
                Some(&DEBUG_LEVEL_DEFAULT.to_string()),
            );
        }
        // YaxisType
        self.y_axis_type =
            YAxisType::get_instance(unsafe { get_attribute(autodoc, YAxisType::KEY) }, component);
        // YaxisObjectNum
        self.yaxis_object_num
            .parse_attribute(unsafe { get_attribute(autodoc, YAXIS_OBJECT_NUM_KEY) });
        Self::add_error(&self.yaxis_object_num, error_list.as_deref_mut());
        // YaxisContourNum
        self.yaxis_contour_num
            .parse_attribute(unsafe { get_attribute(autodoc, YAXIS_CONTOUR_NUM_KEY) });
        Self::add_error(&self.yaxis_contour_num, error_list.as_deref_mut());
        // flgWedgeWeight
        self.flg_wedge_weight
            .parse_attribute(unsafe { get_attribute(autodoc, FLG_WEDGE_WEIGHT_KEY) });
        if !Self::add_error(&self.flg_wedge_weight, error_list.as_deref_mut()) {
            self.check_value_expected(
                FieldRef::FlgWedgeWeight,
                &[0, 1],
                component,
                FLG_WEDGE_WEIGHT_KEY,
                None,
                1,
            );
        }
        // sampleSphere
        self.sample_sphere
            .parse_attribute(unsafe { get_attribute(autodoc, SampleSphere::KEY) });
        Self::add_error(&self.sample_sphere, error_list.as_deref_mut());
        // sampleInterval
        self.sample_interval
            .parse_attribute(unsafe { get_attribute(autodoc, SAMPLE_INTERVAL_KEY) });
        Self::add_error(&self.sample_interval, error_list.as_deref_mut());
        // maskType
        self.mask_type
            .parse_attribute(unsafe { get_attribute(autodoc, MASK_TYPE_KEY) });
        Self::add_error(&self.mask_type, error_list.as_deref_mut());
        // maskModelPts
        self.mask_model_pts
            .parse_attribute(unsafe { get_attribute(autodoc, MASK_MODEL_PTS_KEY) });
        Self::add_error(&self.mask_model_pts, error_list.as_deref_mut());
        // insideMaskRadius
        self.inside_mask_radius
            .parse_attribute(unsafe { get_attribute(autodoc, INSIDE_MASK_RADIUS_KEY) });
        Self::add_error(&self.inside_mask_radius, error_list.as_deref_mut());
        // outsideMaskRadius
        self.outside_mask_radius
            .parse_attribute(unsafe { get_attribute(autodoc, OUTSIDE_MASK_RADIUS_KEY) });
        Self::add_error(&self.outside_mask_radius, error_list.as_deref_mut());
        // nWeightGroup
        self.n_weight_group
            .parse_attribute(unsafe { get_attribute(autodoc, N_WEIGHT_GROUP_KEY) });
        if !Self::add_error(&self.n_weight_group, error_list.as_deref_mut()) {
            self.check_value_range(
                FieldRef::NWeightGroup,
                N_WEIGHT_GROUP_MIN,
                N_WEIGHT_GROUP_MAX,
                1,
                component,
                N_WEIGHT_GROUP_KEY,
                Some(shared_strings::N_WEIGHT_GROUP_LABEL),
                Some(&N_WEIGHT_GROUP_DEFAULT.to_string()),
            );
        }
        // flgRemoveDuplicates
        self.flg_remove_duplicates
            .parse_attribute(unsafe { get_attribute(autodoc, FLG_REMOVE_DUPLICATES_KEY) });
        Self::add_error(&self.flg_remove_duplicates, error_list.as_deref_mut());
        if !Self::add_error(&self.flg_remove_duplicates, error_list.as_deref_mut()) {
            self.check_value_expected(
                FieldRef::FlgRemoveDuplicates,
                &[0, 1],
                component,
                FLG_REMOVE_DUPLICATES_KEY,
                Some(shared_strings::FLG_REMOVE_DUPLICATES_LABEL),
                1,
            );
        }
        // flgAlignAverages
        self.flg_align_averages
            .parse_attribute(unsafe { get_attribute(autodoc, FLG_ALIGN_AVERAGES_KEY) });
        Self::add_error(&self.flg_align_averages, error_list.as_deref_mut());
        if !Self::add_error(&self.flg_align_averages, error_list.as_deref_mut()) {
            self.check_value_expected(
                FieldRef::FlgAlignAverages,
                &[0, 1],
                component,
                FLG_ALIGN_AVERAGES_KEY,
                Some(shared_strings::FLG_ALIGN_AVERAGES_LABEL),
                1,
            );
        }
        // flgFairReference
        self.flg_fair_reference
            .parse_attribute(unsafe { get_attribute(autodoc, FLG_FAIR_REFERENCE_KEY) });
        if !Self::add_error(&self.flg_fair_reference, error_list.as_deref_mut()) {
            self.check_value_expected(
                FieldRef::FlgFairReference,
                &[0, 1],
                component,
                FLG_FAIR_REFERENCE_KEY,
                Some(shared_strings::FLG_FAIR_REFERENCE_LABEL),
                -1,
            );
        }
        // flgAbsValue
        self.flg_abs_value
            .parse_attribute(unsafe { get_attribute(autodoc, FLG_ABS_VALUE_KEY) });
        if !Self::add_error(&self.flg_abs_value, error_list.as_deref_mut()) {
            self.check_value_expected(
                FieldRef::FlgAbsValue,
                &[0, 1],
                component,
                FLG_ABS_VALUE_KEY,
                Some(shared_strings::FLG_ABS_VALUE_LABEL),
                1,
            );
        }
        // flgStrictSearchLimits
        self.flg_strict_search_limits
            .parse_attribute(unsafe { get_attribute(autodoc, FLG_STRICT_SEARCH_LIMITS_KEY) });
        Self::add_error(&self.flg_strict_search_limits, error_list.as_deref_mut());
        if !Self::add_error(&self.flg_strict_search_limits, error_list.as_deref_mut()) {
            self.check_value_expected(
                FieldRef::FlgStrictSearchLimits,
                &[0, 1],
                component,
                FLG_STRICT_SEARCH_LIMITS_KEY,
                Some(shared_strings::FLG_STRICT_SEARCH_LIMITS_LABEL),
                1,
            );
        }
        // selectClassID
        let attribute = unsafe { get_attribute(autodoc, SELECT_CLASS_ID_KEY) };
        self.select_class_id.parse_attribute(attribute);
        // Backwards compatibility - read it in if its a number
        if self.select_class_id.validate().is_some() {
            self.bc_select_class_id.parse_attribute(attribute);
            if self.bc_select_class_id.validate().is_some() {
                Self::add_error(&self.select_class_id, error_list.as_deref_mut());
            } else {
                let raw_string = self.bc_select_class_id.get_raw_string_void();
                self.select_class_id
                    .set_raw_string_string(raw_string.as_deref());
            }
        }
        // FlgNoReferenceRefinement
        self.flg_no_reference_refinement
            .parse_attribute(unsafe { get_attribute(autodoc, FLG_NO_REFERENCE_REFINEMENT_KEY) });
        if !Self::add_error(&self.flg_no_reference_refinement, error_list.as_deref_mut()) {
            self.check_value_expected(
                FieldRef::FlgNoReferenceRefinement,
                &[0, 1],
                component,
                FLG_NO_REFERENCE_REFINEMENT_KEY,
                Some(shared_strings::FLG_NO_REFERENCE_REFINEMENT_LABEL),
                -1,
            );
        }
        // flgRandomize
        self.flg_randomize
            .parse_attribute(unsafe { get_attribute(autodoc, FLG_RANDOMIZE_KEY) });
        if !Self::add_error(&self.flg_randomize, error_list.as_deref_mut()) {
            self.check_value_expected(
                FieldRef::FlgRandomize,
                &[0, 1],
                component,
                FLG_RANDOMIZE_KEY,
                Some(shared_strings::FLG_RANDOMIZE_LABEL),
                -1,
            );
        }
        // cylinderHeight
        self.cylinder_height
            .parse_attribute(unsafe { get_attribute(autodoc, CYLINDER_HEIGHT_KEY) });
        Self::add_error(&self.cylinder_height, error_list.as_deref_mut());
        // maskBlurStdDev
        self.mask_blur_std_dev
            .parse_attribute(unsafe { get_attribute(autodoc, MASK_BLUR_STD_DEV_KEY) });
        Self::add_error(&self.mask_blur_std_dev, error_list.as_deref_mut());
        // flgVolNamesAreTemplates
        self.flg_vol_names_are_templates
            .parse_attribute(unsafe { get_attribute(autodoc, FLG_VOL_NAMES_ARE_TEMPLATES_KEY) });
        if !Self::add_error(&self.flg_vol_names_are_templates, error_list.as_deref_mut()) {
            self.check_value_expected(
                FieldRef::FlgVolNamesAreTemplates,
                &[0, 1],
                component,
                FLG_VOL_NAMES_ARE_TEMPLATES_KEY,
                Some(shared_strings::FLG_VOL_NAMES_ARE_TEMPLATES_LABEL),
                -1,
            );
        }
        // excludeList
        self.exclude_list
            .parse_attribute(unsafe { get_attribute(autodoc, EXCLUDE_LIST_KEY) });
        Self::add_error(&self.exclude_list, error_list.as_deref_mut());
        // includeList
        self.include_list
            .parse_attribute(unsafe { get_attribute(autodoc, INCLUDE_LIST_KEY) });
        Self::add_error(&self.include_list, error_list.as_deref_mut());
        // flgEelevationCompensation
        self.flg_elevation_compensation
            .parse_attribute(unsafe { get_attribute(autodoc, FLG_ELEVATION_COMPENSATION_KEY) });
        if !Self::add_error(&self.flg_elevation_compensation, error_list.as_deref_mut()) {
            self.check_value_expected(
                FieldRef::FlgElevationCompensation,
                &[0, 1],
                component,
                FLG_ELEVATION_COMPENSATION_KEY,
                Some(shared_strings::FLG_ELEVATION_COMPENSATION_LABEL),
                -1,
            );
        }
        // flgFRM
        self.flg_frm
            .parse_attribute(unsafe { get_attribute(autodoc, FLG_FRM_KEY) });
        if !Self::add_error(&self.flg_frm, error_list.as_deref_mut()) {
            self.check_value_expected(
                FieldRef::FlgFrm,
                &[0, 1],
                component,
                FLG_FRM_KEY,
                Some(shared_strings::FLG_FRM_LABEL),
                -1,
            );
        }
        // flgAllowMaskedCorrelation
        self.flg_allow_masked_correlation
            .parse_attribute(unsafe { get_attribute(autodoc, FLG_ALLOW_MASKED_CORRELATION_KEY) });
        if !Self::add_error(&self.flg_allow_masked_correlation, error_list.as_deref_mut()) {
            self.check_value_expected(
                FieldRef::FlgAllowMaskedCorrelation,
                &[0, 1],
                component,
                FLG_ALLOW_MASKED_CORRELATION_KEY,
                Some(shared_strings::FLG_ALLOW_MASKED_CORRELATION_LABEL),
                -1,
            );
        }
        // flgFilterRefOnly
        self.flg_filter_ref_only
            .parse_attribute(unsafe { get_attribute(autodoc, FLG_FILTER_REF_ONLY_KEY) });
        if !Self::add_error(&self.flg_filter_ref_only, error_list.as_deref_mut()) {
            self.check_value_expected(
                FieldRef::FlgFilterRefOnly,
                &[0, 1],
                component,
                FLG_FILTER_REF_ONLY_KEY,
                Some(shared_strings::FLG_FILTER_REF_ONLY_LABEL),
                -1,
            );
        }
        // flgSearchAlongParticleAxes
        self.flg_search_along_particle_axes.parse_attribute(unsafe {
            get_attribute(autodoc, FLG_SEARCH_ALONG_PARTICLE_AXES_KEY)
        });
        if !Self::add_error(&self.flg_search_along_particle_axes, error_list.as_deref_mut()) {
            self.check_value_expected(
                FieldRef::FlgSearchAlongParticleAxes,
                &[0, 1],
                component,
                FLG_SEARCH_ALONG_PARTICLE_AXES_KEY,
                Some(shared_strings::FLG_SEARCH_ALONG_PARTICLE_AXES_LABEL),
                -1,
            );
        }
        // flgFPWedgeMask
        self.flg_fp_wedge_mask
            .parse_attribute(unsafe { get_attribute(autodoc, FLG_FP_WEDGE_MASK_KEY) });
        if !Self::add_error(&self.flg_fp_wedge_mask, error_list.as_deref_mut()) {
            self.check_value_expected(
                FieldRef::FlgFpWedgeMask,
                &[0, 1],
                component,
                FLG_FP_WEDGE_MASK_KEY,
                Some(shared_strings::FLG_FP_WEDGE_MASK_LABEL),
                -1,
            );
        }
        // yAxisSymmetry
        self.y_axis_symmetry
            .parse_attribute(unsafe { get_attribute(autodoc, Y_AXIS_SYMMETRY_KEY) });
        Self::add_error(&self.y_axis_symmetry, error_list.as_deref_mut());
        // flgUseExtractedParticles
        self.flg_use_extracted_particles
            .parse_attribute(unsafe { get_attribute(autodoc, FLG_USE_EXTRACTED_PARTICLES_KEY) });
        if !Self::add_error(&self.flg_use_extracted_particles, error_list.as_deref_mut()) {
            self.check_value_expected(
                FieldRef::FlgUseExtractedParticles,
                &[0, 1],
                component,
                FLG_USE_EXTRACTED_PARTICLES_KEY,
                Some(shared_strings::FLG_USE_EXTRACTED_PARTICLES_LABEL),
                -1,
            );
        }
        // cNSymmetricAveraging
        self.c_n_symmetric_averaging
            .parse_attribute(unsafe { get_attribute(autodoc, CN_SYMMETRIC_AVERAGING_KEY) });
        Self::add_error(&self.c_n_symmetric_averaging, error_list.as_deref_mut());
        if !Self::add_error(&self.c_n_symmetric_averaging, error_list.as_deref_mut()) {
            self.check_value_range(
                FieldRef::CNSymmetricAveraging,
                CN_SYMMETRIC_AVERAGING_MIN,
                CN_SYMMETRIC_AVERAGING_MAX,
                1,
                component,
                CN_SYMMETRIC_AVERAGING_KEY,
                Some(shared_strings::CN_SYMMETRIC_AVERAGING_LABEL),
                Some(&CN_SYMMETRIC_AVERAGING_DEFAULT.to_string()),
            );
        }
        // flgCNMasking
        self.flg_cn_masking
            .parse_attribute(unsafe { get_attribute(autodoc, FLG_CN_MASKING_KEY) });
        if !Self::add_error(&self.flg_cn_masking, error_list.as_deref_mut()) {
            self.check_value_expected(
                FieldRef::FlgCnMasking,
                &[0, 1],
                component,
                FLG_CN_MASKING_KEY,
                Some(shared_strings::FLG_CN_MASKING_LABEL),
                -1,
            );
        }
        // userCommands
        self.set_user_commands_attribute(unsafe { get_attribute(autodoc, USER_COMMANDS_KEY) });
        let user_commands = self
            .user_commands
            .as_deref()
            .map(|user_commands| user_commands as &dyn Parsable);
        Self::add_error_parsable(user_commands, error_list.as_deref_mut());
    }

    /// The `ParsedNumber` field a `checkValue` call names.
    fn field_mut(&mut self, field: FieldRef) -> &mut ParsedNumber {
        match field {
            FieldRef::RefFlagAllTom => &mut self.ref_flag_all_tom,
            FieldRef::EdgeShift => &mut self.edge_shift,
            FieldRef::LstFlagAllTom => &mut self.lst_flag_all_tom,
            FieldRef::DebugLevel => &mut self.debug_level,
            FieldRef::FlgWedgeWeight => &mut self.flg_wedge_weight,
            FieldRef::NWeightGroup => &mut self.n_weight_group,
            FieldRef::FlgRemoveDuplicates => &mut self.flg_remove_duplicates,
            FieldRef::FlgAlignAverages => &mut self.flg_align_averages,
            FieldRef::FlgFairReference => &mut self.flg_fair_reference,
            FieldRef::FlgAbsValue => &mut self.flg_abs_value,
            FieldRef::FlgStrictSearchLimits => &mut self.flg_strict_search_limits,
            FieldRef::FlgNoReferenceRefinement => &mut self.flg_no_reference_refinement,
            FieldRef::FlgRandomize => &mut self.flg_randomize,
            FieldRef::FlgVolNamesAreTemplates => &mut self.flg_vol_names_are_templates,
            FieldRef::FlgElevationCompensation => &mut self.flg_elevation_compensation,
            FieldRef::FlgFrm => &mut self.flg_frm,
            FieldRef::FlgAllowMaskedCorrelation => &mut self.flg_allow_masked_correlation,
            FieldRef::FlgFilterRefOnly => &mut self.flg_filter_ref_only,
            FieldRef::FlgSearchAlongParticleAxes => &mut self.flg_search_along_particle_axes,
            FieldRef::FlgFpWedgeMask => &mut self.flg_fp_wedge_mask,
            FieldRef::FlgUseExtractedParticles => &mut self.flg_use_extracted_particles,
            FieldRef::CNSymmetricAveraging => &mut self.c_n_symmetric_averaging,
            FieldRef::FlgCnMasking => &mut self.flg_cn_masking,
        }
    }

    /// Java package-private `checkValue(ParsedNumber, int[], UIComponent, String,
    /// String, int)`.
    fn check_value_expected(
        &mut self,
        field: FieldRef,
        expected_values: &[i32],
        component: Option<&dyn UIComponent>,
        param_name: &str,
        field_label: Option<&str>,
        replacement_value_index: i32,
    ) {
        let manager = self.manager;
        let number = self.field_mut(field);
        if number.is_missing_attribute() {
            return;
        }
        let mut ok = false;
        for expected_value in expected_values {
            if number.equals(*expected_value) {
                ok = true;
            }
        }
        if !ok {
            let replacement = if replacement_value_index != -1
                && (replacement_value_index as usize) < expected_values.len()
            {
                Some(expected_values[replacement_value_index as usize].to_string())
            } else {
                None
            };
            let raw_string = number.get_raw_string_void();
            ui_harness::with(|harness| {
                harness.open_problem_value_message_dialog(
                    manager,
                    component,
                    "Unknown",
                    Some(param_name),
                    None,
                    field_label,
                    raw_string.as_deref(),
                    replacement.as_deref(),
                    None,
                )
            });
        }
    }

    /// Java package-private `checkValue(ParsedNumber, int, int, int, UIComponent,
    /// String, String, String)`.  Checks for an out of range value.  If it is, pops up
    /// a warning and changes number to to replacement value (if replacement value is
    /// not null).
    #[allow(clippy::too_many_arguments)]
    fn check_value_range(
        &mut self,
        field: FieldRef,
        min: i32,
        max: i32,
        step: i32,
        component: Option<&dyn UIComponent>,
        param_name: &str,
        field_label: Option<&str>,
        replacement_value: Option<&str>,
    ) {
        let manager = self.manager;
        let number = self.field_mut(field);
        if number.is_missing_attribute() {
            return;
        }
        let Some(n_number) = number.get_raw_number() else {
            return;
        };
        let num = n_number.int_value();
        let mut ok = false;
        if step == 1 {
            if num >= min && num <= max {
                ok = true;
            }
        } else {
            let mut i = min;
            while i <= max {
                if num == i {
                    ok = true;
                    break;
                }
                i += step;
            }
        }
        if !ok {
            let raw_string = number.get_raw_string_void();
            ui_harness::with(|harness| {
                harness.open_problem_value_message_dialog(
                    manager,
                    component,
                    "Out of range",
                    Some(param_name),
                    None,
                    field_label,
                    raw_string.as_deref(),
                    replacement_value,
                    None,
                )
            });
            if let Some(replacement_value) = replacement_value {
                number.set_raw_string_string(Some(replacement_value));
            }
        }
    }

    /// Java `validate(boolean)`.
    pub fn validate(&self, for_run: bool) -> bool {
        for volume in &self.volume_list {
            if !volume.validate(for_run) {
                return false;
            }
        }
        true
    }

    /// Java private `parseVolumeData(ReadOnlyAutodoc, List, UIComponent)`.
    fn parse_volume_data(
        &mut self,
        autodoc: *mut Autodoc,
        mut error_list: Option<&mut Vec<String>>,
        component: Option<&dyn UIComponent>,
    ) {
        self.volume_list.clear();
        let mut size = 0;
        // relativeOrient
        let mut relative_orient =
            ParsedList::get_matlab_instance_type(Some(Type::Double), Some(RELATIVE_ORIENT_KEY));
        relative_orient.parse_attribute(unsafe { get_attribute(autodoc, RELATIVE_ORIENT_KEY) });
        Self::add_error_parsable(Some(&relative_orient), error_list.as_deref_mut());
        size = size.max(relative_orient.size_parsable());
        // fnVolume
        let mut fn_volume = ParsedList::get_string_instance(Some(FN_VOLUME_KEY));
        fn_volume.parse_attribute(unsafe { get_attribute(autodoc, FN_VOLUME_KEY) });
        Self::add_error_parsable(Some(&fn_volume), error_list.as_deref_mut());
        size = size.max(fn_volume.size_parsable());
        // fnModParticle
        let mut fn_mod_particle = ParsedList::get_string_instance(Some(FN_MOD_PARTICLE_KEY));
        fn_mod_particle.parse_attribute(unsafe { get_attribute(autodoc, FN_MOD_PARTICLE_KEY) });
        Self::add_error_parsable(Some(&fn_mod_particle), error_list.as_deref_mut());
        size = size.max(fn_mod_particle.size_parsable());
        // initMOTL
        let mut init_motl_file = None;
        let attribute = unsafe { get_attribute(autodoc, InitMotlCode::KEY) };
        if ParsedList::is_list(attribute) {
            self.init_motl_code = None;
            let mut list = ParsedList::get_string_instance(Some(InitMotlCode::KEY));
            list.parse_attribute(attribute);
            Self::add_error_parsable(Some(&list), error_list.as_deref_mut());
            size = size.max(list.size_parsable());
            init_motl_file = Some(list);
        } else {
            self.init_motl_code = Some(InitMotlCode::get_instance(attribute, component));
        }
        // tiltRange
        let attribute = unsafe { get_attribute(autodoc, TILT_RANGE_KEY) };
        let mut tilt_range = if !ParsedList::is_string_list(attribute) {
            self.is_tilt_range_multi_axes = false;
            ParsedList::get_matlab_instance_type(Some(Type::Double), Some(TILT_RANGE_KEY))
        } else {
            self.is_tilt_range_multi_axes = true;
            ParsedList::get_string_instance(Some(TILT_RANGE_KEY))
        };
        tilt_range.parse_attribute(attribute);
        Self::add_error_parsable(Some(&tilt_range), error_list.as_deref_mut());
        size = size.max(tilt_range.size_parsable());
        // Add elements to volumeList
        for i in 0..size {
            let mut volume = Volume::new(self.manager, self.axis_id);
            volume.set_relative_orient(relative_orient.get_element(i));
            volume.set_fn_volume_element(fn_volume.get_element(i));
            volume.set_fn_mod_particle_element(fn_mod_particle.get_element(i));
            if self.init_motl_code.is_none() {
                volume.set_init_motl_element(
                    init_motl_file.as_ref().and_then(|list| list.get_element(i)),
                );
            }
            volume.set_tilt_range(tilt_range.get_element(i));
            self.volume_list.push(volume);
        }
    }

    /// Java `resetVolumeList()`.
    pub fn reset_volume_list(&mut self) {
        let mut i = 0;
        while i < self.volume_list.len() {
            self.volume_list.clear();
            i += 1;
        }
    }

    /// Java private `parseIterationData(ReadOnlyAutodoc, List, UIComponent)`.
    fn parse_iteration_data(
        &mut self,
        autodoc: *mut Autodoc,
        mut error_list: Option<&mut Vec<String>>,
        component: Option<&dyn UIComponent>,
    ) {
        self.iteration_list.clear();
        let mut size = 0;
        // dPhi
        let mut d_phi = ParsedList::get_matlab_instance_type(Some(Type::Double), Some(D_PHI_KEY));
        d_phi.parse_attribute(unsafe { get_attribute(autodoc, D_PHI_KEY) });
        Self::add_error_parsable(Some(&d_phi), error_list.as_deref_mut());
        size = size.max(d_phi.size_parsable());
        // dTheta
        let mut d_theta =
            ParsedList::get_matlab_instance_type(Some(Type::Double), Some(D_THETA_KEY));
        d_theta.parse_attribute(unsafe { get_attribute(autodoc, D_THETA_KEY) });
        Self::add_error_parsable(Some(&d_theta), error_list.as_deref_mut());
        size = size.max(d_theta.size_parsable());
        // dPsi
        let mut d_psi = ParsedList::get_matlab_instance_type(Some(Type::Double), Some(D_PSI_KEY));
        d_psi.parse_attribute(unsafe { get_attribute(autodoc, D_PSI_KEY) });
        Self::add_error_parsable(Some(&d_psi), error_list.as_deref_mut());
        size = size.max(d_psi.size_parsable());
        // searchRadius
        let mut search_radius = ParsedList::get_matlab_instance(Some(SEARCH_RADIUS_KEY));
        search_radius.parse_attribute(unsafe { get_attribute(autodoc, SEARCH_RADIUS_KEY) });
        Self::add_error_parsable(Some(&search_radius), error_list.as_deref_mut());
        size = size.max(search_radius.size_parsable());
        // lowCutoff
        let mut low_cutoff =
            ParsedList::get_matlab_instance_type(Some(Type::Double), Some(LOW_CUTOFF_KEY));
        low_cutoff.parse_attribute(unsafe { get_attribute(autodoc, LOW_CUTOFF_KEY) });
        Self::add_error_parsable(Some(&low_cutoff), error_list.as_deref_mut());
        // hiCutoff
        let mut hi_cutoff =
            ParsedList::get_matlab_instance_type(Some(Type::Double), Some(HI_CUTOFF_KEY));
        hi_cutoff.parse_attribute(unsafe { get_attribute(autodoc, HI_CUTOFF_KEY) });
        Self::add_error_parsable(Some(&hi_cutoff), error_list.as_deref_mut());
        size = size.max(hi_cutoff.size_parsable());
        // refThreshold
        let mut ref_threshold =
            ParsedList::get_matlab_instance_type(Some(Type::Double), Some(REF_THRESHOLD_KEY));
        ref_threshold.parse_attribute(unsafe { get_attribute(autodoc, REF_THRESHOLD_KEY) });
        Self::add_error_parsable(Some(&ref_threshold), error_list.as_deref_mut());
        // duplicateShiftTolerance
        let mut duplicate_shift_tolerance =
            ParsedArray::get_matlab_instance(Some(DUPLICATE_SHIFT_TOLERANCE_KEY));
        duplicate_shift_tolerance
            .parse_attribute(unsafe { get_attribute(autodoc, DUPLICATE_SHIFT_TOLERANCE_KEY) });
        Self::add_error(&duplicate_shift_tolerance, error_list.as_deref_mut());
        // duplicateAngularTolerance
        let mut duplicate_angular_tolerance =
            ParsedArray::get_matlab_instance(Some(DUPLICATE_ANGULAR_TOLERANCE_KEY));
        duplicate_angular_tolerance
            .parse_attribute(unsafe { get_attribute(autodoc, DUPLICATE_ANGULAR_TOLERANCE_KEY) });
        Self::add_error(&duplicate_angular_tolerance, error_list.as_deref_mut());
        size = size.max(ref_threshold.size_parsable());
        // add elements to iterationList
        for i in 0..size {
            let mut iteration = Iteration::new();
            iteration.set_d_phi(d_phi.get_element(i), component);
            iteration.set_d_theta(d_theta.get_element(i), component);
            iteration.set_d_psi(d_psi.get_element(i), component);
            iteration.set_search_radius_element(search_radius.get_element(i));
            iteration.set_low_cutoff(low_cutoff.get_element(i));
            iteration.set_hi_cutoff(hi_cutoff.get_element(i));
            iteration.set_ref_threshold_element(ref_threshold.get_element(i));
            iteration
                .set_duplicate_shift_tolerance_element(duplicate_shift_tolerance.get_element(i));
            iteration.set_duplicate_angular_tolerance_element(
                duplicate_angular_tolerance.get_element(i),
            );
            self.iteration_list.push(iteration);
        }
    }

    /// Java private `buildParsableValues(Map)`.  Called by write().  Places parsable
    /// strings (string that will be written to the file) into a Map in preparation for
    /// writing.
    fn build_parsable_values(&mut self, value_map: &mut HashMap<String, Option<String>>) {
        self.build_parsable_volume_values(value_map);
        self.build_parsable_iteration_values(value_map);
        if self.use_reference_file {
            value_map.insert(
                REFERENCE_KEY.to_owned(),
                self.reference_file.get_parsable_string(),
            );
        } else {
            value_map.insert(REFERENCE_KEY.to_owned(), self.reference.get_parsable_string());
        }
        value_map.insert(FN_OUTPUT_KEY.to_owned(), self.fn_output.get_parsable_string());
        // copy szVol X value to Y and Z when Y and/or Z is empty
        let sz_vol_x = self
            .sz_vol
            .get_element(X_INDEX)
            .filter(|sz_vol_x| !sz_vol_x.is_empty())
            .and_then(|sz_vol_x| sz_vol_x.get_raw_string_void().map(Some))
            .unwrap_or(None);
        let sz_vol_x_present = self
            .sz_vol
            .get_element(X_INDEX)
            .is_some_and(|sz_vol_x| !sz_vol_x.is_empty());
        if sz_vol_x_present {
            if self.sz_vol.is_empty_int(Y_INDEX) {
                self.sz_vol
                    .set_raw_string_int_string(Y_INDEX, sz_vol_x.as_deref());
            }
            if self.sz_vol.is_empty_int(Z_INDEX) {
                self.sz_vol
                    .set_raw_string_int_string(Z_INDEX, sz_vol_x.as_deref());
            }
        }
        value_map.insert(SZ_VOL_KEY.to_owned(), self.sz_vol.get_parsable_string());
        if !self.is_tilt_range_empty() && !self.edge_shift.is_empty() {
            value_map.insert(
                EDGE_SHIFT_KEY.to_owned(),
                self.edge_shift.get_parsable_string(),
            );
        }
        if let Some(init_motl_code) = self.init_motl_code {
            value_map.insert(
                InitMotlCode::KEY.to_owned(),
                Some(init_motl_code.to_string()),
            );
        }
        value_map.insert(
            ALIGNED_BASE_NAME_KEY.to_owned(),
            self.aligned_base_name.get_parsable_string(),
        );
        value_map.insert(
            DEBUG_LEVEL_KEY.to_owned(),
            self.debug_level.get_parsable_string(),
        );
        value_map.insert(
            LST_THRESHOLDS_KEY.to_owned(),
            self.lst_thresholds.get_parsable_string(),
        );
        value_map.insert(
            REF_FLAG_ALL_TOM_KEY.to_owned(),
            self.ref_flag_all_tom.get_parsable_string(),
        );
        value_map.insert(
            LST_FLAG_ALL_TOM_KEY.to_owned(),
            self.lst_flag_all_tom.get_parsable_string(),
        );
        value_map.insert(
            PARTICLE_PER_CPU_KEY.to_owned(),
            self.particle_per_cpu.get_parsable_string(),
        );
        value_map.insert(YAxisType::KEY.to_owned(), Some(self.y_axis_type.to_string()));
        value_map.insert(
            YAXIS_OBJECT_NUM_KEY.to_owned(),
            self.yaxis_object_num.get_parsable_string(),
        );
        value_map.insert(
            YAXIS_CONTOUR_NUM_KEY.to_owned(),
            self.yaxis_contour_num.get_parsable_string(),
        );
        value_map.insert(
            FLG_WEDGE_WEIGHT_KEY.to_owned(),
            self.flg_wedge_weight.get_parsable_string(),
        );
        value_map.insert(
            SampleSphere::KEY.to_owned(),
            self.sample_sphere.get_parsable_string(),
        );
        value_map.insert(
            SAMPLE_INTERVAL_KEY.to_owned(),
            self.sample_interval.get_parsable_string(),
        );
        value_map.insert(MASK_TYPE_KEY.to_owned(), self.mask_type.get_parsable_string());
        value_map.insert(
            MASK_MODEL_PTS_KEY.to_owned(),
            self.mask_model_pts.get_parsable_string(),
        );
        value_map.insert(
            INSIDE_MASK_RADIUS_KEY.to_owned(),
            self.inside_mask_radius.get_parsable_string(),
        );
        value_map.insert(
            OUTSIDE_MASK_RADIUS_KEY.to_owned(),
            self.outside_mask_radius.get_parsable_string(),
        );
        value_map.insert(
            N_WEIGHT_GROUP_KEY.to_owned(),
            self.n_weight_group.get_parsable_string(),
        );
        value_map.insert(
            FLG_REMOVE_DUPLICATES_KEY.to_owned(),
            self.flg_remove_duplicates.get_parsable_string(),
        );
        value_map.insert(
            FLG_ALIGN_AVERAGES_KEY.to_owned(),
            self.flg_align_averages.get_parsable_string(),
        );
        value_map.insert(
            FLG_FAIR_REFERENCE_KEY.to_owned(),
            self.flg_fair_reference.get_parsable_string(),
        );
        value_map.insert(
            FLG_ABS_VALUE_KEY.to_owned(),
            self.flg_abs_value.get_parsable_string(),
        );
        value_map.insert(
            FLG_STRICT_SEARCH_LIMITS_KEY.to_owned(),
            self.flg_strict_search_limits.get_parsable_string(),
        );
        if !self.select_class_id.is_empty() {
            value_map.insert(
                SELECT_CLASS_ID_KEY.to_owned(),
                self.select_class_id.get_parsable_string(),
            );
        } else {
            value_map.remove(SELECT_CLASS_ID_KEY);
        }
        value_map.insert(
            FLG_NO_REFERENCE_REFINEMENT_KEY.to_owned(),
            self.flg_no_reference_refinement.get_parsable_string(),
        );
        value_map.insert(
            FLG_RANDOMIZE_KEY.to_owned(),
            self.flg_randomize.get_parsable_string(),
        );
        value_map.insert(
            CYLINDER_HEIGHT_KEY.to_owned(),
            self.cylinder_height.get_parsable_string(),
        );
        value_map.insert(
            MASK_BLUR_STD_DEV_KEY.to_owned(),
            self.mask_blur_std_dev.get_parsable_string(),
        );
        value_map.insert(
            FLG_VOL_NAMES_ARE_TEMPLATES_KEY.to_owned(),
            self.flg_vol_names_are_templates.get_parsable_string(),
        );
        value_map.insert(
            EXCLUDE_LIST_KEY.to_owned(),
            self.exclude_list.get_parsable_string(),
        );
        value_map.insert(
            INCLUDE_LIST_KEY.to_owned(),
            self.include_list.get_parsable_string(),
        );
        value_map.insert(
            FLG_ELEVATION_COMPENSATION_KEY.to_owned(),
            self.flg_elevation_compensation.get_parsable_string(),
        );
        value_map.insert(FLG_FRM_KEY.to_owned(), self.flg_frm.get_parsable_string());
        value_map.insert(
            FLG_ALLOW_MASKED_CORRELATION_KEY.to_owned(),
            self.flg_allow_masked_correlation.get_parsable_string(),
        );
        value_map.insert(
            FLG_FILTER_REF_ONLY_KEY.to_owned(),
            self.flg_filter_ref_only.get_parsable_string(),
        );
        value_map.insert(
            FLG_SEARCH_ALONG_PARTICLE_AXES_KEY.to_owned(),
            self.flg_search_along_particle_axes.get_parsable_string(),
        );
        value_map.insert(
            FLG_FP_WEDGE_MASK_KEY.to_owned(),
            self.flg_fp_wedge_mask.get_parsable_string(),
        );
        value_map.insert(
            Y_AXIS_SYMMETRY_KEY.to_owned(),
            self.y_axis_symmetry.get_parsable_string(),
        );
        value_map.insert(
            FLG_USE_EXTRACTED_PARTICLES_KEY.to_owned(),
            self.flg_use_extracted_particles.get_parsable_string(),
        );
        if !self.c_n_symmetric_averaging.is_empty() {
            value_map.insert(
                CN_SYMMETRIC_AVERAGING_KEY.to_owned(),
                self.c_n_symmetric_averaging.get_parsable_string(),
            );
        } else {
            value_map.remove(CN_SYMMETRIC_AVERAGING_KEY);
        }
        value_map.insert(
            FLG_CN_MASKING_KEY.to_owned(),
            self.flg_cn_masking.get_parsable_string(),
        );
        // Fixed in translation (MatlabParam.java:1849): with no user commands
        // (`userCommands` still null, e.g. a .prm without the key, copied by "Copy
        // project" and written before the dialog sets it) Java throws
        // NullPointerException and the .prm is not written; the key is left out
        // instead (a null value is skipped by setNameValuePairValue).  (BUGS.md)
        value_map.insert(
            USER_COMMANDS_KEY.to_owned(),
            self.user_commands
                .as_ref()
                .and_then(|user_commands| user_commands.get_parsable_string_parsable()),
        );
    }

    /// Java private `buildParsableVolumeValues(Map)`.
    fn build_parsable_volume_values(&self, value_map: &mut HashMap<String, Option<String>>) {
        let mut fn_volume = ParsedList::get_string_instance(Some(FN_VOLUME_KEY));
        let mut fn_mod_particle = ParsedList::get_string_instance(Some(FN_MOD_PARTICLE_KEY));
        let mut init_motl_file = None;
        if self.init_motl_code.is_none() {
            init_motl_file = Some(ParsedList::get_string_instance(Some(InitMotlCode::KEY)));
        }
        let mut tilt_range = if !self.is_tilt_range_multi_axes {
            ParsedList::get_matlab_instance_type(Some(Type::Double), Some(TILT_RANGE_KEY))
        } else {
            ParsedList::get_string_instance(Some(TILT_RANGE_KEY))
        };
        // build the lists
        for volume in &self.volume_list {
            fn_volume.add_element(volume.get_fn_volume().clone_element());
            fn_mod_particle.add_element(volume.get_fn_mod_particle().clone_element());
            if self.init_motl_code.is_none() {
                init_motl_file
                    .as_mut()
                    .unwrap()
                    .add_element(volume.get_init_motl().clone_element());
            }
            tilt_range.add_element(
                volume
                    .get_tilt_range(self.is_tilt_range_multi_axes)
                    .clone_element(),
            );
        }
        value_map.insert(
            FN_VOLUME_KEY.to_owned(),
            fn_volume.get_parsable_string_parsable(),
        );
        value_map.insert(
            FN_MOD_PARTICLE_KEY.to_owned(),
            fn_mod_particle.get_parsable_string_parsable(),
        );
        if self.init_motl_code.is_none() {
            value_map.insert(
                InitMotlCode::KEY.to_owned(),
                init_motl_file.unwrap().get_parsable_string_parsable(),
            );
        }
        if self.tilt_range_empty {
            tilt_range.clear_parsable();
        }
        value_map.insert(
            TILT_RANGE_KEY.to_owned(),
            tilt_range.get_parsable_string_parsable(),
        );
    }

    /// Java private `buildParsableIterationValues(Map)`.
    fn build_parsable_iteration_values(&self, value_map: &mut HashMap<String, Option<String>>) {
        let mut d_phi = ParsedList::get_matlab_instance_type(Some(Type::Double), Some(D_PHI_KEY));
        let mut d_theta =
            ParsedList::get_matlab_instance_type(Some(Type::Double), Some(D_THETA_KEY));
        let mut d_psi = ParsedList::get_matlab_instance_type(Some(Type::Double), Some(D_PSI_KEY));
        let mut search_radius = ParsedList::get_matlab_instance(Some(SEARCH_RADIUS_KEY));
        let mut low_cutoff =
            ParsedList::get_matlab_instance_type(Some(Type::Double), Some(LOW_CUTOFF_KEY));
        let mut hi_cutoff =
            ParsedList::get_matlab_instance_type(Some(Type::Double), Some(HI_CUTOFF_KEY));
        let mut ref_threshold =
            ParsedList::get_matlab_instance_type(Some(Type::Double), Some(REF_THRESHOLD_KEY));
        let mut duplicate_shift_tolerance =
            ParsedArray::get_matlab_instance(Some(DUPLICATE_SHIFT_TOLERANCE_KEY));
        let mut duplicate_angular_tolerance =
            ParsedArray::get_matlab_instance(Some(DUPLICATE_ANGULAR_TOLERANCE_KEY));
        // build the lists
        for iteration in &self.iteration_list {
            d_phi.add_element(iteration.get_d_phi().clone_element());
            d_theta.add_element(iteration.get_d_theta().clone_element());
            d_psi.add_element(iteration.get_d_psi().clone_element());
            search_radius.add_element(iteration.get_search_radius().clone_element());
            low_cutoff.add_element(iteration.get_low_cutoff().clone_element());
            hi_cutoff.add_element(iteration.get_hi_cutoff().clone_element());
            ref_threshold.add_element(iteration.get_ref_threshold().clone_element());
            duplicate_shift_tolerance
                .add_element(iteration.get_duplicate_shift_tolerance().clone_element());
            duplicate_angular_tolerance
                .add_element(iteration.get_duplicate_angular_tolerance().clone_element());
        }
        value_map.insert(D_PHI_KEY.to_owned(), d_phi.get_parsable_string_parsable());
        value_map.insert(
            D_THETA_KEY.to_owned(),
            d_theta.get_parsable_string_parsable(),
        );
        value_map.insert(D_PSI_KEY.to_owned(), d_psi.get_parsable_string_parsable());
        value_map.insert(
            SEARCH_RADIUS_KEY.to_owned(),
            search_radius.get_parsable_string_parsable(),
        );
        value_map.insert(
            LOW_CUTOFF_KEY.to_owned(),
            low_cutoff.get_parsable_string_parsable(),
        );
        value_map.insert(
            HI_CUTOFF_KEY.to_owned(),
            hi_cutoff.get_parsable_string_parsable(),
        );
        value_map.insert(
            REF_THRESHOLD_KEY.to_owned(),
            ref_threshold.get_parsable_string_parsable(),
        );
        value_map.insert(
            DUPLICATE_SHIFT_TOLERANCE_KEY.to_owned(),
            duplicate_shift_tolerance.get_parsable_string(),
        );
        value_map.insert(
            DUPLICATE_ANGULAR_TOLERANCE_KEY.to_owned(),
            duplicate_angular_tolerance.get_parsable_string(),
        );
    }

    /// Java private `updateOrBuildAutodoc(BaseManager, Map, WritableAutodoc,
    /// ReadOnlyAutodoc)`.  Called by write().  Updates or adds all the name/value pair
    /// to autodoc.  Will attempt to add comments when adding a new name/value pair.
    fn update_or_build_autodoc(
        &self,
        manager: Option<&'static dyn BaseManager>,
        value_map: &HashMap<String, Option<String>>,
        autodoc: &mut Autodoc,
        comment_autodoc: Option<&Autodoc>,
    ) {
        let mut comment_map = None;
        if let Some(comment_autodoc) = comment_autodoc {
            comment_map = comment_autodoc.get_attribute_multi_line_values(
                Some(etomo_autodoc::FIELD_SECTION_NAME),
                Some(etomo_autodoc::COMMENT_KEY),
            );
        }
        // write to a autodoc, name/value pairs as necessary
        // the order doesn't matter, because this is either an existing autodoc
        // (so new entries will end up at the bottom), or the comment autodoc (which
        // provides the order) is not usable.
        self.set_name_value_pair_values(manager, value_map, autodoc, comment_map.as_ref());
    }

    /// Java private `setNameValuePairValues(BaseManager, Map, WritableAutodoc, Map)`.
    /// Adds or changes the value of an name/value pair in the file.
    fn set_name_value_pair_values(
        &self,
        manager: Option<&'static dyn BaseManager>,
        value_map: &HashMap<String, Option<String>>,
        autodoc: &mut Autodoc,
        comment_map: Option<&HashMap<String, Option<String>>>,
    ) {
        let value = |key: &str| value_map.get(key).cloned().flatten();
        self.set_volume_name_value_pair_values(manager, value_map, autodoc, comment_map);
        self.set_iteration_name_value_pair_values(manager, value_map, autodoc, comment_map);
        Self::set_name_value_pair_value(autodoc, REFERENCE_KEY, value(REFERENCE_KEY), comment_map);
        Self::set_name_value_pair_value(autodoc, FN_OUTPUT_KEY, value(FN_OUTPUT_KEY), comment_map);
        Self::set_name_value_pair_value(autodoc, SZ_VOL_KEY, value(SZ_VOL_KEY), comment_map);
        if self.is_tilt_range_empty() {
            Self::remove_name_value_pair(autodoc, EDGE_SHIFT_KEY);
        } else {
            Self::set_name_value_pair_value(
                autodoc,
                EDGE_SHIFT_KEY,
                value(EDGE_SHIFT_KEY),
                comment_map,
            );
        }
        for key in [
            ALIGNED_BASE_NAME_KEY,
            DEBUG_LEVEL_KEY,
            LST_THRESHOLDS_KEY,
            REF_FLAG_ALL_TOM_KEY,
            LST_FLAG_ALL_TOM_KEY,
            PARTICLE_PER_CPU_KEY,
            YAxisType::KEY,
        ] {
            Self::set_name_value_pair_value(autodoc, key, value(key), comment_map);
        }
        Self::remove_name_value_pair(autodoc, YAXIS_CONTOUR_KEY);
        for key in [
            YAXIS_OBJECT_NUM_KEY,
            YAXIS_CONTOUR_NUM_KEY,
            FLG_WEDGE_WEIGHT_KEY,
            SampleSphere::KEY,
            SAMPLE_INTERVAL_KEY,
            MASK_TYPE_KEY,
            MASK_MODEL_PTS_KEY,
            INSIDE_MASK_RADIUS_KEY,
            OUTSIDE_MASK_RADIUS_KEY,
            N_WEIGHT_GROUP_KEY,
            FLG_REMOVE_DUPLICATES_KEY,
        ] {
            Self::set_name_value_pair_value(autodoc, key, value(key), comment_map);
        }
        if self.flg_align_averages.is_empty() {
            Self::remove_name_value_pair(autodoc, FLG_ALIGN_AVERAGES_KEY);
        } else {
            Self::set_name_value_pair_value(
                autodoc,
                FLG_ALIGN_AVERAGES_KEY,
                value(FLG_ALIGN_AVERAGES_KEY),
                comment_map,
            );
        }
        for key in [
            FLG_FAIR_REFERENCE_KEY,
            FLG_ABS_VALUE_KEY,
            FLG_STRICT_SEARCH_LIMITS_KEY,
        ] {
            Self::set_name_value_pair_value(autodoc, key, value(key), comment_map);
        }
        let select_class_id = value_map.get(SELECT_CLASS_ID_KEY);
        if select_class_id.is_some_and(Option::is_some) {
            Self::set_name_value_pair_value(
                autodoc,
                SELECT_CLASS_ID_KEY,
                value(SELECT_CLASS_ID_KEY),
                comment_map,
            );
        } else {
            Self::remove_name_value_pair(autodoc, SELECT_CLASS_ID_KEY);
        }
        for key in [
            FLG_NO_REFERENCE_REFINEMENT_KEY,
            FLG_RANDOMIZE_KEY,
            CYLINDER_HEIGHT_KEY,
            MASK_BLUR_STD_DEV_KEY,
            FLG_VOL_NAMES_ARE_TEMPLATES_KEY,
            EXCLUDE_LIST_KEY,
            INCLUDE_LIST_KEY,
            FLG_ELEVATION_COMPENSATION_KEY,
            FLG_FRM_KEY,
            FLG_ALLOW_MASKED_CORRELATION_KEY,
            FLG_FILTER_REF_ONLY_KEY,
            FLG_SEARCH_ALONG_PARTICLE_AXES_KEY,
            FLG_FP_WEDGE_MASK_KEY,
            Y_AXIS_SYMMETRY_KEY,
            FLG_USE_EXTRACTED_PARTICLES_KEY,
        ] {
            Self::set_name_value_pair_value(autodoc, key, value(key), comment_map);
        }
        if self.c_n_symmetric_averaging.is_empty() {
            Self::remove_name_value_pair(autodoc, CN_SYMMETRIC_AVERAGING_KEY);
        } else {
            Self::set_name_value_pair_value(
                autodoc,
                CN_SYMMETRIC_AVERAGING_KEY,
                value(CN_SYMMETRIC_AVERAGING_KEY),
                comment_map,
            );
        }
        Self::set_name_value_pair_value(
            autodoc,
            FLG_CN_MASKING_KEY,
            value(FLG_CN_MASKING_KEY),
            comment_map,
        );
        Self::set_name_value_pair_value(
            autodoc,
            USER_COMMANDS_KEY,
            value(USER_COMMANDS_KEY),
            comment_map,
        );
    }

    /// Java private `setVolumeNameValuePairValues(BaseManager, Map, WritableAutodoc,
    /// Map)`.
    fn set_volume_name_value_pair_values(
        &self,
        _manager: Option<&'static dyn BaseManager>,
        value_map: &HashMap<String, Option<String>>,
        autodoc: &mut Autodoc,
        comment_map: Option<&HashMap<String, Option<String>>>,
    ) {
        for key in [
            FN_VOLUME_KEY,
            FN_MOD_PARTICLE_KEY,
            InitMotlCode::KEY,
            TILT_RANGE_KEY,
        ] {
            Self::set_name_value_pair_value(
                autodoc,
                key,
                value_map.get(key).cloned().flatten(),
                comment_map,
            );
        }
    }

    /// Java private `setIterationNameValuePairValues(BaseManager, Map,
    /// WritableAutodoc, Map)`.
    fn set_iteration_name_value_pair_values(
        &self,
        _manager: Option<&'static dyn BaseManager>,
        value_map: &HashMap<String, Option<String>>,
        autodoc: &mut Autodoc,
        comment_map: Option<&HashMap<String, Option<String>>>,
    ) {
        for key in [
            D_PHI_KEY,
            D_THETA_KEY,
            D_PSI_KEY,
            SEARCH_RADIUS_KEY,
            LOW_CUTOFF_KEY,
            HI_CUTOFF_KEY,
            REF_THRESHOLD_KEY,
            DUPLICATE_SHIFT_TOLERANCE_KEY,
            DUPLICATE_ANGULAR_TOLERANCE_KEY,
        ] {
            Self::set_name_value_pair_value(
                autodoc,
                key,
                value_map.get(key).cloned().flatten(),
                comment_map,
            );
        }
    }

    /// Java private `setNameValuePairValue(BaseManager, WritableAutodoc, String,
    /// String, Map)`.  Gets the attribute.  If the attribute doesn't exist, it adds the
    /// attribute.  Adds or changes the value of the attribute.
    fn set_name_value_pair_value(
        autodoc: &mut Autodoc,
        name: &str,
        value: Option<String>,
        comment_map: Option<&HashMap<String, Option<String>>>,
    ) {
        let Some(value) = value else {
            return;
        };
        let attribute = unsafe { autodoc.get_writable_attribute(Some(name)) };
        if attribute.is_null() {
            match comment_map {
                // new attribute, so add attribute and name/value pair
                None => Self::set_name_value_pair(autodoc, name, &value, None),
                // new attribute, so add comment, attribute, and name/value pair
                Some(comment_map) => Self::set_name_value_pair(
                    autodoc,
                    name,
                    &value,
                    comment_map.get(name).cloned().flatten().as_deref(),
                ),
            }
        } else {
            unsafe { (*attribute).set_value(Some(&value)) };
        }
    }

    /// Java private `removeNameValuePair(WritableAutodoc, String)`.
    fn remove_name_value_pair(autodoc: &mut Autodoc, name: &str) {
        let mut previous_statement = unsafe { autodoc.remove_name_value_pair(Some(name)) };
        // remove the associated comments
        while !previous_statement.is_null()
            && unsafe { (*previous_statement).get_type() } == StatementType::Comment
        {
            previous_statement = unsafe { autodoc.remove_statement(previous_statement) };
        }
        // remove the associated empty line
        if !previous_statement.is_null()
            && unsafe { (*previous_statement).get_type() } == StatementType::EmptyLine
        {
            unsafe { autodoc.remove_statement(previous_statement) };
        }
    }

    /// Java private `setNameValuePair(BaseManager, WritableAutodoc, String, String,
    /// String)`.  Adds or updates a name/value pair.  If adding, also trys to add a
    /// new-line and a comment.
    fn set_name_value_pair(
        autodoc: &mut Autodoc,
        attribute_name: &str,
        attribute_value: &str,
        comment: Option<&str>,
    ) {
        let attribute = unsafe { autodoc.get_writable_attribute(Some(attribute_name)) };
        if attribute.is_null() {
            // If the attribute doesn't exist try to add a comment and add the attribute
            if let Some(comment) = comment {
                // there's a comment, so add an empty line first
                autodoc.add_empty_line(-1);
                // Format and add the comment
                let comment_array =
                    etomo_autodoc::format(Some(&format!("{attribute_name}:\n{comment}")))
                        .unwrap_or_default();
                for line in comment_array {
                    autodoc.add_comment_string(Some(&format!(" {line}")), -1);
                }
            }
            // Add the attribute and name/value pair
            autodoc.add_name_value_pair_attribute(Some(attribute_name), Some(attribute_value));
        } else {
            // If atttribute does exist, change its value
            unsafe { (*attribute).set_value(Some(attribute_value)) };
        }
    }
}

/// The `ParsedNumber` fields `checkValue` is called on (Java passes the field
/// itself; the translation names it so the check can borrow it mutably).
#[derive(Clone, Copy)]
enum FieldRef {
    RefFlagAllTom,
    EdgeShift,
    LstFlagAllTom,
    DebugLevel,
    FlgWedgeWeight,
    NWeightGroup,
    FlgRemoveDuplicates,
    FlgAlignAverages,
    FlgFairReference,
    FlgAbsValue,
    FlgStrictSearchLimits,
    FlgNoReferenceRefinement,
    FlgRandomize,
    FlgVolNamesAreTemplates,
    FlgElevationCompensation,
    FlgFrm,
    FlgAllowMaskedCorrelation,
    FlgFilterRefOnly,
    FlgSearchAlongParticleAxes,
    FlgFpWedgeMask,
    FlgUseExtractedParticles,
    CNSymmetricAveraging,
    FlgCnMasking,
}

/// Java `public static final class InitMotlCode implements EnumeratedType`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InitMotlCode {
    /// Java `ZERO`, value 0, "Set all angles to 0".
    Zero,
    /// Java `Z_AXIS`, value 1, no label (deprecated: convert 1 to 2).
    ZAxis,
    /// Java `X_AND_Z_AXIS`, value 2.
    XAndZAxis,
    /// Java `RANDOM_ROTATIONS`, value 3.
    RandomRotations,
    /// Java `RANDOM_AXIAL_ROTATIONS`, value 4.
    RandomAxialRotations,
}

impl InitMotlCode {
    /// Java `DEFAULT = ZERO`.
    pub const DEFAULT: InitMotlCode = InitMotlCode::Zero;
    /// Java `KEY`.
    pub const KEY: &'static str = "initMOTL";

    /// The constructor's `value`.
    fn value(self) -> i32 {
        match self {
            InitMotlCode::Zero => 0,
            InitMotlCode::ZAxis => 1,
            InitMotlCode::XAndZAxis => 2,
            InitMotlCode::RandomRotations => 3,
            InitMotlCode::RandomAxialRotations => 4,
        }
    }

    /// Java private static `getInstance(ReadOnlyAttribute, UIComponent)`.
    fn get_instance(
        attribute: Option<&dyn ReadOnlyAttribute>,
        component: Option<&dyn UIComponent>,
    ) -> InitMotlCode {
        let Some(attribute) = attribute else {
            return InitMotlCode::DEFAULT;
        };
        let Some(value) = attribute.get_value() else {
            return InitMotlCode::DEFAULT;
        };
        let mut number = EtomoNumber::new();
        for (code, result) in [
            (InitMotlCode::Zero, InitMotlCode::Zero),
            (InitMotlCode::ZAxis, InitMotlCode::XAndZAxis),
            (InitMotlCode::XAndZAxis, InitMotlCode::XAndZAxis),
            (InitMotlCode::RandomRotations, InitMotlCode::RandomRotations),
            (
                InitMotlCode::RandomAxialRotations,
                InitMotlCode::RandomAxialRotations,
            ),
        ] {
            // `value.equals(String)` on the code's EtomoNumber.
            number.set_int(code.value());
            if number.equals_string(Some(&value)) {
                return result;
            }
        }
        ui_harness::with(|harness| {
            harness.open_problem_value_message_dialog(
                None,
                component,
                "Unknown",
                Some(InitMotlCode::KEY),
                None,
                Some(shared_strings::INIT_MOTL_LABEL),
                Some(&value),
                Some(&InitMotlCode::DEFAULT.value().to_string()),
                InitMotlCode::DEFAULT.get_label().as_deref(),
            )
        });
        InitMotlCode::DEFAULT
    }
}

/// Java `toString()`: the value.
impl std::fmt::Display for InitMotlCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.value())
    }
}

impl EnumeratedType for InitMotlCode {
    fn is_default(&self) -> bool {
        *self == InitMotlCode::DEFAULT
    }

    /// Java `getValue()`: Z_AXIS reports X_AND_Z_AXIS's value.
    fn get_value(&self) -> ConstEtomoNumber {
        let code = if *self == InitMotlCode::ZAxis {
            InitMotlCode::XAndZAxis
        } else {
            *self
        };
        let mut value = EtomoNumber::new();
        value.set_int(code.value());
        (*value).clone()
    }

    fn get_label(&self) -> Option<String> {
        match self {
            InitMotlCode::Zero => Some("Set all angles to 0".to_owned()),
            InitMotlCode::ZAxis => None,
            InitMotlCode::XAndZAxis => Some(shared_strings::INIT_MOTL_X_AND_Z_AXIS_LABEL.to_owned()),
            InitMotlCode::RandomRotations => {
                Some(shared_strings::INIT_MOTL_RANDOM_ROTATIONS.to_owned())
            }
            InitMotlCode::RandomAxialRotations => {
                Some(shared_strings::INIT_MOTL_RANDOM_AXIAL_ROTATIONS.to_owned())
            }
        }
    }
}

/// Java `public static final class MaskType implements EnumeratedType`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaskType {
    /// Java `NONE`, "none".
    None,
    /// Java `VOLUME`, "DUMMY_VOLUME_VALUE".
    Volume,
    /// Java `SPHERE`, "sphere".
    Sphere,
    /// Java `CYLINDER`, "cylinder".
    Cylinder,
}

impl MaskType {
    /// Java private static final `DEFAULT = NONE`.
    const DEFAULT: MaskType = MaskType::None;

    /// The constructor's `value`.
    fn value(self) -> &'static str {
        match self {
            MaskType::None => "none",
            MaskType::Volume => "DUMMY_VOLUME_VALUE",
            MaskType::Sphere => "sphere",
            MaskType::Cylinder => "cylinder",
        }
    }

    /// Java static `getInstance(String)`.  An empty string returns the default
    /// instance.  An unrecognized string returns a volume instance because the volume
    /// string is actually an absolute file path.
    pub fn get_instance(value: Option<&str>) -> MaskType {
        let Some(value) = value else {
            return MaskType::DEFAULT;
        };
        if java_lang_string_matches_whitespace(value) {
            return MaskType::DEFAULT;
        }
        for mask_type in [
            MaskType::None,
            MaskType::Volume,
            MaskType::Sphere,
            MaskType::Cylinder,
        ] {
            if mask_type.value() == value {
                return mask_type;
            }
        }
        MaskType::Volume
    }
}

/// Java `toString()`: the value.
impl std::fmt::Display for MaskType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.value())
    }
}

impl EnumeratedType for MaskType {
    fn is_default(&self) -> bool {
        *self == MaskType::DEFAULT
    }

    /// Java `getValue()`: null.  (No caller reads it; an empty number stands in.)
    fn get_value(&self) -> ConstEtomoNumber {
        (*EtomoNumber::new()).clone()
    }

    fn get_label(&self) -> Option<String> {
        None
    }
}

/// Java `public static final class SampleSphere implements EnumeratedType`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SampleSphere {
    /// Java `NONE`, "none".
    None,
    /// Java `FULL`, "full".
    Full,
    /// Java `HALF`, "half".
    Half,
}

impl SampleSphere {
    /// Java private static final `DEFAULT = NONE`.
    const DEFAULT: SampleSphere = SampleSphere::None;
    /// Java private static final `KEY`.
    const KEY: &'static str = "sampleSphere";

    /// The constructor's `value`.
    fn value(self) -> &'static str {
        match self {
            SampleSphere::None => "none",
            SampleSphere::Full => "full",
            SampleSphere::Half => "half",
        }
    }

    /// Java private static `getInstance(ParsedElement, UIComponent)`.
    fn get_instance(
        parsed_element: &dyn ParsedElement,
        component: Option<&dyn UIComponent>,
    ) -> SampleSphere {
        if parsed_element.is_missing_attribute() {
            return SampleSphere::DEFAULT;
        }
        let Some(value) = parsed_element.get_raw_string_void() else {
            return SampleSphere::DEFAULT;
        };
        for sample_sphere in [SampleSphere::None, SampleSphere::Full, SampleSphere::Half] {
            if sample_sphere.value() == value {
                return sample_sphere;
            }
        }
        ui_harness::with(|harness| {
            harness.open_problem_value_message_dialog(
                None,
                component,
                "Unknown",
                Some(SampleSphere::KEY),
                None,
                Some(shared_strings::SAMPLE_SPHERE_LABEL),
                Some(&value),
                Some(SampleSphere::DEFAULT.value()),
                None,
            )
        });
        SampleSphere::DEFAULT
    }
}

/// Java `toString()`: the value.
impl std::fmt::Display for SampleSphere {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.value())
    }
}

impl EnumeratedType for SampleSphere {
    fn is_default(&self) -> bool {
        *self == SampleSphere::DEFAULT
    }

    /// Java `getValue()`: null.  (No caller reads it.)
    fn get_value(&self) -> ConstEtomoNumber {
        (*EtomoNumber::new()).clone()
    }

    fn get_label(&self) -> Option<String> {
        None
    }
}

/// Java `public static final class YAxisType implements EnumeratedType`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum YAxisType {
    /// Java `Y_AXIS`, value 0.
    YAxis,
    /// Java `PARTICLE_MODEL`, value 1.
    ParticleModel,
    /// Java `CONTOUR`, value 2.
    Contour,
    /// Java `CSV_FILES`, value 3.
    CsvFiles,
}

impl YAxisType {
    /// Java `DEFAULT = Y_AXIS`.
    pub const DEFAULT: YAxisType = YAxisType::YAxis;
    /// Java `KEY`.
    pub const KEY: &'static str = "yaxisType";

    /// The constructor's `value`.
    fn value(self) -> i32 {
        match self {
            YAxisType::YAxis => 0,
            YAxisType::ParticleModel => 1,
            YAxisType::Contour => 2,
            YAxisType::CsvFiles => 3,
        }
    }

    /// Java private static `getInstance(ReadOnlyAttribute, UIComponent)`.
    fn get_instance(
        attribute: Option<&dyn ReadOnlyAttribute>,
        component: Option<&dyn UIComponent>,
    ) -> YAxisType {
        let Some(attribute) = attribute else {
            return YAxisType::DEFAULT;
        };
        let Some(value) = attribute.get_value() else {
            return YAxisType::DEFAULT;
        };
        let mut number = EtomoNumber::new();
        for y_axis_type in [
            YAxisType::YAxis,
            YAxisType::ParticleModel,
            YAxisType::Contour,
            YAxisType::CsvFiles,
        ] {
            number.set_int(y_axis_type.value());
            if number.equals_string(Some(&value)) {
                return y_axis_type;
            }
        }
        ui_harness::with(|harness| {
            harness.open_problem_value_message_dialog(
                None,
                component,
                "Unknown",
                Some(YAxisType::KEY),
                None,
                Some(shared_strings::YAXIS_TYPE_LABEL),
                Some(&value),
                Some(&YAxisType::DEFAULT.value().to_string()),
                YAxisType::DEFAULT.get_label().as_deref(),
            )
        });
        YAxisType::DEFAULT
    }
}

/// Java `toString()`: the value.
impl std::fmt::Display for YAxisType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.value())
    }
}

impl EnumeratedType for YAxisType {
    fn is_default(&self) -> bool {
        *self == YAxisType::DEFAULT
    }

    /// Java `getValue()`: null.  (No caller reads it.)
    fn get_value(&self) -> ConstEtomoNumber {
        (*EtomoNumber::new()).clone()
    }

    fn get_label(&self) -> Option<String> {
        Some(
            match self {
                YAxisType::YAxis => shared_strings::YAXIS_TYPE_Y_AXIS_LABEL,
                YAxisType::ParticleModel => shared_strings::YAXIS_TYPE_PARTICLE_MODEL_LABEL,
                YAxisType::Contour => shared_strings::YAXIS_TYPE_CONTOUR_LABEL,
                YAxisType::CsvFiles => shared_strings::CSV_FILES_LABEL,
            }
            .to_owned(),
        )
    }
}

/// Java `Volume.START_INDEX`.
const START_INDEX: i32 = 0;
/// Java `Volume.END_INDEX`.
const END_INDEX: i32 = 1;

/// Java `public static final class Volume`.
pub struct Volume {
    /// Java private final `tiltRange`.
    tilt_range: ParsedArray,
    /// Java private final `relativeOrient` (deprecated - only for validation).
    relative_orient: ParsedArray,
    /// Java private final `fnVolume`.
    fn_volume: ParsedQuotedString,
    /// Java private final `fnModParticle`.
    fn_mod_particle: ParsedQuotedString,
    /// Java private final `initMotl`.
    init_motl: ParsedQuotedString,
    /// Java private final `tiltRangeMultiAxes`.
    tilt_range_multi_axes: ParsedQuotedString,
    /// Java private final `manager`.
    manager: Option<&'static dyn BaseManager>,
    /// Java private final `axisID`.
    axis_id: AxisID,
}

impl Volume {
    /// Java private `Volume(BaseManager, AxisID)`.
    fn new(manager: Option<&'static dyn BaseManager>, axis_id: AxisID) -> Volume {
        let mut relative_orient =
            ParsedArray::get_instance(Some(Type::Double), Some(RELATIVE_ORIENT_KEY), None);
        relative_orient.set_backward_compatible_null_key();
        Volume {
            tilt_range: ParsedArray::get_matlab_instance_type(
                Some(Type::Double),
                Some(TILT_RANGE_KEY),
            ),
            relative_orient,
            fn_volume: ParsedQuotedString::get_instance(Some(FN_VOLUME_KEY)),
            fn_mod_particle: ParsedQuotedString::get_instance(Some(FN_MOD_PARTICLE_KEY)),
            init_motl: ParsedQuotedString::get_instance(Some(InitMotlCode::KEY)),
            tilt_range_multi_axes: ParsedQuotedString::get_instance(Some(TILT_RANGE_KEY)),
            manager,
            axis_id,
        }
    }

    /// Java private `validate(boolean)`.
    fn validate(&self, for_run: bool) -> bool {
        if !for_run || self.relative_orient.is_empty() {
            return true;
        }
        for i in 0..self.relative_orient.size() {
            if !self.relative_orient.is_empty_int(i)
                && !self
                    .relative_orient
                    .get_element(i)
                    .is_some_and(|element| element.equals(0))
            {
                ui_harness::with(|harness| {
                    harness.open_message_dialog_base_manager_string_string_axis_id(
                        self.manager,
                        "Relative orientations are no longer supported in PEET. This functionality is now handled via initial motive lists. If you are already using initial motive list(s), use the PEET program modifyMotiveList to generate new one(s) incorporating the relative orientation rotation(s). If you do not yet have initial motive list(s), use PEET program slicer2MOTL to generate them.",
                        "Incompatible .prm File",
                        Some(self.axis_id),
                    )
                });
                return false;
            }
        }
        true
    }

    /// Java `setFnVolume(ParsedElement)`.
    pub fn set_fn_volume_element(&mut self, fn_volume: Option<&dyn ParsedElement>) {
        self.fn_volume.set_element(fn_volume);
    }

    /// Java `setFnVolume(String)`.
    pub fn set_fn_volume(&mut self, fn_volume: Option<&str>) {
        self.fn_volume.set_raw_string_string(fn_volume);
    }

    /// Java `getFnVolumeString()`.
    pub fn get_fn_volume_string(&self) -> Option<String> {
        self.fn_volume.get_raw_string_void()
    }

    /// Java `getFnModParticleString()`.
    pub fn get_fn_mod_particle_string(&self) -> Option<String> {
        self.fn_mod_particle.get_raw_string_void()
    }

    /// Java `getTiltRangeMultiAxesString()`.
    pub fn get_tilt_range_multi_axes_string(&self) -> Option<String> {
        self.tilt_range_multi_axes.get_raw_string_void()
    }

    /// Java `getInitMotlString()`.
    pub fn get_init_motl_string(&self) -> Option<String> {
        self.init_motl.get_raw_string_void()
    }

    /// Java `setFnModParticle(ParsedElement)`.
    pub fn set_fn_mod_particle_element(&mut self, fn_mod_particle: Option<&dyn ParsedElement>) {
        self.fn_mod_particle.set_element(fn_mod_particle);
    }

    /// Java `setFnModParticle(String)`.
    pub fn set_fn_mod_particle(&mut self, fn_mod_particle: Option<&str>) {
        self.fn_mod_particle.set_raw_string_string(fn_mod_particle);
    }

    /// Java `setInitMotl(ParsedElement)`.
    pub fn set_init_motl_element(&mut self, init_motl: Option<&dyn ParsedElement>) {
        self.init_motl.set_element(init_motl);
    }

    /// Java `setInitMotl(String)`.
    pub fn set_init_motl(&mut self, init_motl: Option<&str>) {
        self.init_motl.set_raw_string_string(init_motl);
    }

    /// Java `setTiltRangeMultiAxes(String)`.
    pub fn set_tilt_range_multi_axes(&mut self, input: Option<&str>) {
        self.tilt_range_multi_axes.set_raw_string_string(input);
    }

    /// Java `getTiltRangeStart()`.
    pub fn get_tilt_range_start(&self) -> Option<String> {
        self.tilt_range.get_raw_string_int(START_INDEX)
    }

    /// Java `setTiltRangeStart(String)`.
    pub fn set_tilt_range_start(&mut self, tilt_range_start: Option<&str>) {
        self.tilt_range
            .set_raw_string_int_string(START_INDEX, tilt_range_start);
    }

    /// Java `getTiltRangeEnd()`.
    pub fn get_tilt_range_end(&self) -> Option<String> {
        self.tilt_range.get_raw_string_int(END_INDEX)
    }

    /// Java `setTiltRangeEnd(String)`.
    pub fn set_tilt_range_end(&mut self, tilt_range_end: Option<&str>) {
        self.tilt_range
            .set_raw_string_int_string(END_INDEX, tilt_range_end);
    }

    /// Java private `getFnVolume()`.
    fn get_fn_volume(&self) -> &ParsedQuotedString {
        &self.fn_volume
    }

    /// Java private `getFnModParticle()`.
    fn get_fn_mod_particle(&self) -> &ParsedQuotedString {
        &self.fn_mod_particle
    }

    /// Java private `getInitMotl()`.
    fn get_init_motl(&self) -> &ParsedQuotedString {
        &self.init_motl
    }

    /// Java private `isTiltRangeEmpty()`.
    fn is_tilt_range_empty(&self) -> bool {
        self.tilt_range.is_empty()
    }

    /// Java private `getTiltRange(boolean)`.
    fn get_tilt_range(&self, is_tilt_range_multi_axes: bool) -> &dyn ParsedElement {
        if !is_tilt_range_multi_axes {
            &self.tilt_range
        } else {
            &self.tilt_range_multi_axes
        }
    }

    /// Java private `setRelativeOrient(ParsedElement)`.
    fn set_relative_orient(&mut self, relative_orient: Option<&dyn ParsedElement>) {
        self.relative_orient.set(relative_orient);
    }

    /// Java private `setTiltRange(ParsedElement)`.
    fn set_tilt_range(&mut self, tilt_range: Option<&dyn ParsedElement>) {
        self.tilt_range.set(tilt_range);
    }
}

/// Java private static final class `SearchAngleArea`.  Class to handle Phi, Theta,
/// Psi.  Sets End to 0 if it is null.  Sets Start to the negation of End.  Sets
/// Increment to 1 if it is 0.
struct SearchAngleArea {
    /// Java private final `descriptor`.
    descriptor: ParsedArrayDescriptor,
}

impl SearchAngleArea {
    /// Java implicit constructor.
    fn new() -> SearchAngleArea {
        SearchAngleArea {
            descriptor: ParsedArrayDescriptor::get_instance(Some(Type::Double), None),
        }
    }

    /// Java private `clear()`.
    fn clear(&mut self) {
        self.descriptor.clear();
    }

    /// Java private `setEnd(String)`.  Sets both the End and Start values.  The Start
    /// is set to the negation of the End value.  If End is empty, set it and Start to
    /// 0.
    fn set_end(&mut self, input: Option<&str>) {
        let mut end = EtomoNumber::new_with_type(Some(Type::Double));
        end.set_string(input);
        if end.is_null() {
            end.set_int(0);
        }
        self.descriptor.set_raw_string_end(Some(&end.to_string()));
        if !end.equals_int(0) {
            end.multiply_int(-1);
        }
        self.descriptor.set_raw_string_start(Some(&end.to_string()));
    }

    /// Java private `setIncrement(String)`.  Sets the increment value.  If increment
    /// is 0, set it to 1.
    fn set_increment(&mut self, input: Option<&str>) {
        let mut increment = EtomoNumber::new_with_type(Some(Type::Double));
        increment.set_string(input);
        if increment.equals_int(0) {
            increment.set_int(1);
        }
        self.descriptor
            .set_raw_string_increment(Some(&increment.to_string()));
    }

    /// Java private `getEnd()`.
    fn get_end(&self) -> Option<String> {
        self.descriptor.get_raw_string_end()
    }

    /// Java private `set(ParsedElement, String, String, UIComponent)`.
    fn set(
        &mut self,
        input: Option<&dyn ParsedElement>,
        key: &str,
        label: &str,
        component: Option<&dyn UIComponent>,
    ) {
        self.descriptor.set(input);
        if self.descriptor.validate().is_none() {
            Self::check_start(
                self.descriptor.get_start(),
                self.descriptor.get_end(),
                key,
                label,
                component,
            );
        }
    }

    /// Java private `checkStart(ParsedElement, ParsedElement, String, String,
    /// UIComponent)`.  Popup a warning if start*-1 != end.
    ///
    /// Java compares the two values as `BigDecimal`s; the parsed values are plain
    /// decimal numbers, which compare the same as doubles, and `BigDecimal.toString()`
    /// of a plain decimal string is the string itself (the negation flips its sign).
    fn check_start(
        start: Option<&dyn ParsedElement>,
        end: Option<&dyn ParsedElement>,
        key: &str,
        label: &str,
        component: Option<&dyn UIComponent>,
    ) {
        let (Some(start), Some(end)) = (start, end) else {
            // ignore validation errors
            return;
        };
        if start.is_missing_attribute() || start.is_empty() || end.is_missing_attribute() || end.is_empty() {
            // ignore validation errors
            return;
        }
        let start_string = start.get_raw_string_void().unwrap_or_default();
        let end_string = end.get_raw_string_void().unwrap_or_default();
        let bd_start: f64 = start_string.trim().parse().unwrap_or(f64::NAN);
        let bd_end: f64 = end_string.trim().parse().unwrap_or(f64::NAN);
        if bd_start * -1.0 != bd_end {
            let negated_end = if let Some(stripped) = end_string.strip_prefix('-') {
                stripped.to_owned()
            } else if bd_end == 0.0 {
                end_string.clone()
            } else {
                format!("-{end_string}")
            };
            ui_harness::with(|harness| {
                harness.open_problem_value_message_dialog(
                    None,
                    component,
                    "Incorrect",
                    Some(key),
                    Some("start"),
                    Some(label),
                    Some(&start_string),
                    Some(&negated_end),
                    Some("-end"),
                )
            });
        }
    }

    /// Java private `getParsedElement()`.
    fn get_parsed_element(&self) -> &dyn ParsedElement {
        &self.descriptor
    }

    /// Java private `getIncrement()`.
    fn get_increment(&self) -> Option<String> {
        self.descriptor.get_raw_string_increment()
    }
}

/// Java `Iteration.CUTOFF_INDEX`.
const CUTOFF_INDEX: i32 = 0;
/// Java `Iteration.SIGMA_INDEX`.
const SIGMA_INDEX: i32 = 1;

/// Java `public static final class Iteration`.
pub struct Iteration {
    /// Java private final `searchRadius`.
    search_radius: ParsedArray,
    /// Java private final `lowCutoff`.
    low_cutoff: ParsedArray,
    /// Java private final `hiCutoff`.
    hi_cutoff: ParsedArray,
    /// Java private final `refThreshold`.
    ref_threshold: ParsedNumber,
    /// Java private final `duplicateShiftTolerance`.
    duplicate_shift_tolerance: ParsedNumber,
    /// Java private final `duplicateAngularTolerance`.
    duplicate_angular_tolerance: ParsedNumber,
    // search spaces
    /// Java private final `dPhi`.
    d_phi: SearchAngleArea,
    /// Java private final `dTheta`.
    d_theta: SearchAngleArea,
    /// Java private final `dPsi`.
    d_psi: SearchAngleArea,
    /// Java private `userCommand`, initially null.
    user_command: Option<Box<dyn ParsedElement>>,
}

impl Iteration {
    /// Java private `Iteration()`.
    fn new() -> Iteration {
        Iteration {
            search_radius: ParsedArray::get_matlab_instance(Some(SEARCH_RADIUS_KEY)),
            low_cutoff: ParsedArray::get_matlab_instance_type(
                Some(Type::Double),
                Some(LOW_CUTOFF_KEY),
            ),
            hi_cutoff: ParsedArray::get_matlab_instance_type(
                Some(Type::Double),
                Some(HI_CUTOFF_KEY),
            ),
            ref_threshold: ParsedNumber::get_matlab_instance_type(
                Some(Type::Double),
                Some(REF_THRESHOLD_KEY),
            ),
            duplicate_shift_tolerance: ParsedNumber::get_matlab_instance(Some(
                DUPLICATE_SHIFT_TOLERANCE_KEY,
            )),
            duplicate_angular_tolerance: ParsedNumber::get_matlab_instance(Some(
                DUPLICATE_ANGULAR_TOLERANCE_KEY,
            )),
            d_phi: SearchAngleArea::new(),
            d_theta: SearchAngleArea::new(),
            d_psi: SearchAngleArea::new(),
            user_command: None,
        }
    }

    /// Java `getUserCommands()`.
    pub fn get_user_commands(&self) -> Option<String> {
        if let Some(user_command) = &self.user_command {
            return user_command.get_raw_string_void();
        }
        None
    }

    /// Java `clearDPhi()`.
    pub fn clear_d_phi(&mut self) {
        self.d_phi.clear();
    }

    /// Java `clearDTheta()`.
    pub fn clear_d_theta(&mut self) {
        self.d_theta.clear();
    }

    /// Java `clearDPsi()`.
    pub fn clear_d_psi(&mut self) {
        self.d_psi.clear();
    }

    /// Java `setDPhiEnd(String)`.
    pub fn set_d_phi_end(&mut self, input: Option<&str>) {
        self.d_phi.set_end(input);
    }

    /// Java `setDThetaEnd(String)`.
    pub fn set_d_theta_end(&mut self, input: Option<&str>) {
        self.d_theta.set_end(input);
    }

    /// Java `setDPsiEnd(String)`.
    pub fn set_d_psi_end(&mut self, input: Option<&str>) {
        self.d_psi.set_end(input);
    }

    /// Java `setDPhiIncrement(String)`.
    pub fn set_d_phi_increment(&mut self, input: Option<&str>) {
        self.d_phi.set_increment(input);
    }

    /// Java `setDThetaIncrement(String)`.
    pub fn set_d_theta_increment(&mut self, input: Option<&str>) {
        self.d_theta.set_increment(input);
    }

    /// Java `setDPsiIncrement(String)`.
    pub fn set_d_psi_increment(&mut self, input: Option<&str>) {
        self.d_psi.set_increment(input);
    }

    /// Java `setSearchRadius(String)`.
    pub fn set_search_radius(&mut self, input: Option<&str>) {
        self.search_radius.set_raw_string_string(input);
    }

    /// Java `setUserCommand(ParsedElement)`.
    pub fn set_user_command(&mut self, input: Option<Box<dyn ParsedElement>>) {
        self.user_command = input;
    }

    /// Java `getDPhiEnd()`.
    pub fn get_d_phi_end(&self) -> Option<String> {
        self.d_phi.get_end()
    }

    /// Java `getDThetaEnd()`.
    pub fn get_d_theta_end(&self) -> Option<String> {
        self.d_theta.get_end()
    }

    /// Java `getDPsiEnd()`.
    pub fn get_d_psi_end(&self) -> Option<String> {
        self.d_psi.get_end()
    }

    /// Java `setHiCutoffCutoff(String)`.
    pub fn set_hi_cutoff_cutoff(&mut self, input: Option<&str>) {
        self.hi_cutoff.set_raw_string_int_string(CUTOFF_INDEX, input);
    }

    /// Java `setHiCutoffSigma(String)`.
    pub fn set_hi_cutoff_sigma(&mut self, input: Option<&str>) {
        self.hi_cutoff.set_raw_string_int_string(SIGMA_INDEX, input);
    }

    /// Java `setLowCutoffCutoff(String)`.
    pub fn set_low_cutoff_cutoff(&mut self, input: Option<&str>) {
        self.low_cutoff.set_raw_string_int_string(CUTOFF_INDEX, input);
    }

    /// Java `setLowCutoffSigma(String)`.
    pub fn set_low_cutoff_sigma(&mut self, input: Option<&str>) {
        self.low_cutoff.set_raw_string_int_string(SIGMA_INDEX, input);
    }

    /// Java `setRefThreshold(String)`.
    pub fn set_ref_threshold(&mut self, input: Option<&str>) {
        self.ref_threshold.set_raw_string_string(input);
    }

    /// Java `setDuplicateShiftTolerance(String)`.
    pub fn set_duplicate_shift_tolerance(&mut self, input: Option<&str>) {
        self.duplicate_shift_tolerance.set_raw_string_string(input);
    }

    /// Java `setDuplicateAngularTolerance(String)`.
    pub fn set_duplicate_angular_tolerance(&mut self, input: Option<&str>) {
        self.duplicate_angular_tolerance.set_raw_string_string(input);
    }

    /// Java `getDPhiIncrement()`.  Assume that the current format is start:inc:end
    /// even if inc is empty.
    pub fn get_d_phi_increment(&self) -> Option<String> {
        self.d_phi.get_increment()
    }

    /// Java `getDThetaIncrement()`.
    pub fn get_d_theta_increment(&self) -> Option<String> {
        self.d_theta.get_increment()
    }

    /// Java `getDPsiIncrement()`.
    pub fn get_d_psi_increment(&self) -> Option<String> {
        self.d_psi.get_increment()
    }

    /// Java `getSearchRadiusString()`.
    pub fn get_search_radius_string(&self) -> Option<String> {
        self.search_radius.get_raw_string_void()
    }

    /// Java `getHiCutoffCutoff()`.
    pub fn get_hi_cutoff_cutoff(&self) -> Option<String> {
        self.hi_cutoff.get_raw_string_int(CUTOFF_INDEX)
    }

    /// Java `getHiCutoffSigma()`.
    pub fn get_hi_cutoff_sigma(&self) -> Option<String> {
        self.hi_cutoff.get_raw_string_int(SIGMA_INDEX)
    }

    /// Java `getLowCutoffCutoff()`.
    pub fn get_low_cutoff_cutoff(&self) -> Option<String> {
        self.low_cutoff.get_raw_string_int(CUTOFF_INDEX)
    }

    /// Java `isLowCutoffCutoff()`.
    pub fn is_low_cutoff_cutoff(&self) -> bool {
        !self.low_cutoff.is_empty_int(CUTOFF_INDEX)
    }

    /// Java `getLowCutoffSigma()`.
    pub fn get_low_cutoff_sigma(&self) -> Option<String> {
        self.low_cutoff.get_raw_string_int(SIGMA_INDEX)
    }

    /// Java `isLowCutoffSigma()`.
    pub fn is_low_cutoff_sigma(&self) -> bool {
        !self.low_cutoff.is_empty_int(SIGMA_INDEX)
    }

    /// Java `getRefThresholdString()`.
    pub fn get_ref_threshold_string(&self) -> Option<String> {
        self.ref_threshold.get_raw_string_void()
    }

    /// Java `getDuplicateShiftToleranceString()`.
    pub fn get_duplicate_shift_tolerance_string(&self) -> Option<String> {
        self.duplicate_shift_tolerance.get_raw_string_void()
    }

    /// Java `getDuplicateAngularToleranceString()`.
    pub fn get_duplicate_angular_tolerance_string(&self) -> Option<String> {
        self.duplicate_angular_tolerance.get_raw_string_void()
    }

    /// Java private `setDPhi(ParsedElement, UIComponent)`.
    fn set_d_phi(&mut self, input: Option<&dyn ParsedElement>, component: Option<&dyn UIComponent>) {
        self.d_phi
            .set(input, D_PHI_KEY, shared_strings::D_PHI_LABEL, component);
    }

    /// Java private `setDTheta(ParsedElement, UIComponent)`.
    fn set_d_theta(&mut self, input: Option<&dyn ParsedElement>, component: Option<&dyn UIComponent>) {
        self.d_theta
            .set(input, D_THETA_KEY, shared_strings::D_THETA_LABEL, component);
    }

    /// Java private `setDPsi(ParsedElement, UIComponent)`.
    fn set_d_psi(&mut self, input: Option<&dyn ParsedElement>, component: Option<&dyn UIComponent>) {
        self.d_psi
            .set(input, D_PSI_KEY, shared_strings::D_PSI_LABEL, component);
    }

    /// Java private `getDPhi()`.
    fn get_d_phi(&self) -> &dyn ParsedElement {
        self.d_phi.get_parsed_element()
    }

    /// Java private `getDTheta()`.
    fn get_d_theta(&self) -> &dyn ParsedElement {
        self.d_theta.get_parsed_element()
    }

    /// Java private `getDPsi()`.
    fn get_d_psi(&self) -> &dyn ParsedElement {
        self.d_psi.get_parsed_element()
    }

    /// Java private `setSearchRadius(ParsedElement)`.
    fn set_search_radius_element(&mut self, search_radius: Option<&dyn ParsedElement>) {
        self.search_radius.set(search_radius);
    }

    /// Java private `getSearchRadius()`.
    fn get_search_radius(&self) -> &dyn ParsedElement {
        &self.search_radius
    }

    /// Java private `setLowCutoff(ParsedElement)`.
    fn set_low_cutoff(&mut self, input: Option<&dyn ParsedElement>) {
        self.low_cutoff.set(input);
    }

    /// Java private `setHiCutoff(ParsedElement)`.
    fn set_hi_cutoff(&mut self, input: Option<&dyn ParsedElement>) {
        self.hi_cutoff.set(input);
    }

    /// Java private `getLowCutoff()`.
    fn get_low_cutoff(&self) -> &dyn ParsedElement {
        &self.low_cutoff
    }

    /// Java private `getHiCutoff()`.
    fn get_hi_cutoff(&self) -> &dyn ParsedElement {
        &self.hi_cutoff
    }

    /// Java private `getLowCutoffCutoffString()`.
    fn get_low_cutoff_cutoff_string(&self) -> Option<String> {
        if self.low_cutoff.is_empty() {
            return Some(LOW_CUTOFF_DEFAULT.to_owned());
        }
        self.low_cutoff.get_raw_string_int(CUTOFF_INDEX)
    }

    /// Java private `getLowCutoffSigmaString()`.
    fn get_low_cutoff_sigma_string(&self) -> Option<String> {
        if self.low_cutoff.is_empty() || self.low_cutoff.is_empty_int(SIGMA_INDEX) {
            return Some(LOW_CUTOFF_SIGMA_DEFAULT.to_owned());
        }
        self.low_cutoff.get_raw_string_int(SIGMA_INDEX)
    }

    /// Java private `getRefThreshold()`.
    fn get_ref_threshold(&self) -> &dyn ParsedElement {
        &self.ref_threshold
    }

    /// Java `getDuplicateShiftTolerance()`.
    pub fn get_duplicate_shift_tolerance(&self) -> &dyn ParsedElement {
        &self.duplicate_shift_tolerance
    }

    /// Java `getDuplicateAngularTolerance()`.
    pub fn get_duplicate_angular_tolerance(&self) -> &dyn ParsedElement {
        &self.duplicate_angular_tolerance
    }

    /// Java private `setRefThreshold(ParsedElement)`.
    fn set_ref_threshold_element(&mut self, ref_threshold: Option<&dyn ParsedElement>) {
        self.ref_threshold.set_element(ref_threshold);
    }

    /// Java private `setDuplicateShiftTolerance(ParsedElement)`.
    fn set_duplicate_shift_tolerance_element(
        &mut self,
        duplicate_shift_tolerance: Option<&dyn ParsedElement>,
    ) {
        self.duplicate_shift_tolerance
            .set_element(duplicate_shift_tolerance);
    }

    /// Java private `setDuplicateAngularTolerance(ParsedElement)`.
    fn set_duplicate_angular_tolerance_element(
        &mut self,
        duplicate_angular_tolerance: Option<&dyn ParsedElement>,
    ) {
        self.duplicate_angular_tolerance
            .set_element(duplicate_angular_tolerance);
    }
}
