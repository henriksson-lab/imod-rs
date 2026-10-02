//! `IMOD/Etomo/src/etomo/comscript/BeadtrackParam.java`.
//!
//! The PIP (keyword/value) beadtrack command in `track.com`.
//!
//! Java's `BeadtrackParam extends OldBeadtrackParam`, which extends
//! `OldConstBeadtrackParam`.  Rust has no inheritance, so - as
//! `etomo/type/script_parameter.rs` does for its own superclass - each
//! superclass's state is held in a `base` field and reached through
//! `Deref`/`DerefMut`; every inherited field and method is therefore reachable
//! on a `BeadtrackParam` exactly as in Java.  The three superclass methods
//! `BeadtrackParam` overrides (`getMagnificationGroupSize`,
//! `setTiltAngleGroups`, `setMagnificationGroups`) are inherent methods here,
//! which Rust method resolution prefers over the `Deref` target's.
//!
//! Java leaves every `ScriptParameter`/`StringList`/`EtomoBoolean2` field null
//! until `initialize()` runs (from `initializeDefaults` or the first
//! `parseComScriptCommand`), so a getter or setter called before that throws a
//! NullPointerException.  Here the fields are constructed in `new` with the
//! same types, names and key strings `initialize` gives them (without the
//! autodoc's required map, which `initialize` then applies), so those calls
//! see empty values instead of crashing.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use super::bad_com_script_exception::BadComScriptException;
use super::com_script_command::ComScriptCommand;
use super::command::Command;
use super::command_details::CommandDetails;
use super::command_mode::CommandMode;
use super::command_param::{CommandParam, ParseComScriptError};
use super::field_interface::{self, FieldInterface};
use super::fortran_input_string::FortranInputString;
use super::fortran_input_syntax_exception::FortranInputSyntaxException;
use super::old_beadtrack_param::OldBeadtrackParam;
use super::old_const_beadtrack_param::{
    NONDEFAULT_GROUP_INTEGER_TYPE, NONDEFAULT_GROUP_SIZE, OldConstBeadtrackParam,
};
use super::param_utilities;
use super::process_details::ProcessDetails;
use super::string_list::StringList;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::storage::autodoc::autodoc_factory;
use crate::imod::etomo::storage::autodoc::read_only_autodoc::ReadOnlyAutodoc;
use crate::imod::etomo::storage::log_file::LogFileError;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::{
    ConstEtomoNumber, Type, java_lang_string_matches_whitespace,
};
use crate::imod::etomo::r#type::etomo_autodoc;
use crate::imod::etomo::r#type::etomo_boolean2::EtomoBoolean2;
use crate::imod::etomo::r#type::file_key::FileKey;
use crate::imod::etomo::r#type::file_type::{self, FileType};
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::script_parameter::ScriptParameter;
use crate::imod::etomo::ui::swing::log_interface::{Loggable, LoggableException};
use crate::imod::etomo::util::utilities;

/// Java `PROCESS_NAME`.
pub const PROCESS_NAME: ProcessName = ProcessName::TRACK;

/// Java `LOW_PASS_CUTOFF_INVERSE_NM`.
pub const LOW_PASS_CUTOFF_INVERSE_NM: &str = "LowPassCutoffInverseNm";
/// Java `INPUT_FILE_KEY`.
pub const INPUT_FILE_KEY: &str = "ImageFile";
/// Java private `PIECE_LIST_FILE_KEY`.
const PIECE_LIST_FILE_KEY: &str = "PieceListFile";
/// Java `SEED_MODEL_FILE_KEY`.
pub const SEED_MODEL_FILE_KEY: &str = "InputSeedModel";
/// Java `OUTPUT_MODEL_FILE_KEY`.
pub const OUTPUT_MODEL_FILE_KEY: &str = "OutputModel";
/// Java `SKIP_VIEW_LIST_KEY`.
pub const SKIP_VIEW_LIST_KEY: &str = "SkipViews";
/// Java private `IMAGE_ROTATION_KEY`.
const IMAGE_ROTATION_KEY: &str = "RotationAngle";
/// Java `ADDITIONAL_VIEW_GROUPS_KEY`.
pub const ADDITIONAL_VIEW_GROUPS_KEY: &str = "SeparateGroup";
/// Java `TILT_ANGLE_GROUP_PARAMS_KEY`.
pub const TILT_ANGLE_GROUP_PARAMS_KEY: &str = "TiltDefaultGrouping";
/// Java `TILT_ANGLE_GROUPS_KEY`.
pub const TILT_ANGLE_GROUPS_KEY: &str = "TiltNondefaultGroup";
/// Java `MAGNIFICATION_GROUP_PARAMS_KEY`.
pub const MAGNIFICATION_GROUP_PARAMS_KEY: &str = "MagDefaultGrouping";
/// Java `MAGNIFICATION_GROUPS_KEY`.
pub const MAGNIFICATION_GROUPS_KEY: &str = "MagNondefaultGroup";
/// Java `N_MIN_VIEWS_KEY`.
pub const N_MIN_VIEWS_KEY: &str = "MinViewsForTiltalign";
/// Java `FILL_GAPS_KEY`.
pub const FILL_GAPS_KEY: &str = "FillGaps";
/// Java `MAX_GAP_KEY`.
pub const MAX_GAP_KEY: &str = "MaxGapSize";
/// Java `SEARCH_BOX_PIXELS_KEY`.
pub const SEARCH_BOX_PIXELS_KEY: &str = "BoxSizeXandY";
/// Java `MAX_FIDUCIALS_AVG_KEY`.
pub const MAX_FIDUCIALS_AVG_KEY: &str = "MaxBeadsToAverage";
/// Java `FIDUCIAL_EXTRAPOLATION_PARAMS_KEY`.
pub const FIDUCIAL_EXTRAPOLATION_PARAMS_KEY: &str = "PointsToFitMaxAndMin";
/// Java `RESCUE_ATTEMPT_PARAMS_KEY`.
pub const RESCUE_ATTEMPT_PARAMS_KEY: &str = "DensityRescueFractionAndSD";
/// Java `MIN_RESCUE_DISTANCE_KEY`.
pub const MIN_RESCUE_DISTANCE_KEY: &str = "DistanceRescueCriterion";
/// Java `RESCUE_RELAXATION_PARAMS_KEY`.
pub const RESCUE_RELAXATION_PARAMS_KEY: &str = "RescueRelaxationDensityAndDistance";
/// Java `RESIDUAL_DISTANCE_LIMIT_KEY`.
pub const RESIDUAL_DISTANCE_LIMIT_KEY: &str = "PostFitRescueResidual";
/// Java `MEAN_RESID_CHANGE_LIMITS_KEY`.
pub const MEAN_RESID_CHANGE_LIMITS_KEY: &str = "ResidualsToAnalyzeMaxAndMin";
/// Java `DELETION_PARAMS_KEY`.
pub const DELETION_PARAMS_KEY: &str = "DeletionCriterionMinAndSD";
/// Java `DENSITY_RELAXATION_POST_FIT_KEY`.
pub const DENSITY_RELAXATION_POST_FIT_KEY: &str = "DensityRelaxationPostFit";
/// Java `MAX_RESCUE_DISTANCE_KEY`.
pub const MAX_RESCUE_DISTANCE_KEY: &str = "MaxRescueDistance";
/// Java `MIN_TILT_RANGE_TO_FIND_AXIS_KEY`.
pub const MIN_TILT_RANGE_TO_FIND_AXIS_KEY: &str = "MinTiltRangeToFindAxis";
/// Java `MIN_TILT_RANGE_TO_FIND_ANGLES_KEY`.
pub const MIN_TILT_RANGE_TO_FIND_ANGLES_KEY: &str = "MinTiltRangeToFindAngles";
/// Java `LIGHT_BEADS_KEY`.
pub const LIGHT_BEADS_KEY: &str = "LightBeads";
/// Java `BEAD_DIAMETER_KEY`.
pub const BEAD_DIAMETER_KEY: &str = "BeadDiameter";

/// Java `LOCAL_AREA_TRACKING_KEY`.
pub const LOCAL_AREA_TRACKING_KEY: &str = "LocalAreaTracking";
/// Java `LOCAL_AREA_TARGET_SIZE_KEY`.
pub const LOCAL_AREA_TARGET_SIZE_KEY: &str = "LocalAreaTargetSize";
/// Java `MIN_BEADS_IN_AREA_KEY`.
pub const MIN_BEADS_IN_AREA_KEY: &str = "MinBeadsInArea";
/// Java `MIN_OVERLAP_BEADS_KEY`.
pub const MIN_OVERLAP_BEADS_KEY: &str = "MinOverlapBeads";
/// Java `MAX_VIEWS_IN_ALIGN_KEY`.
pub const MAX_VIEWS_IN_ALIGN_KEY: &str = "MaxViewsInAlign";
/// Java `ROUNDS_OF_TRACKING_KEY`.
pub const ROUNDS_OF_TRACKING_KEY: &str = "RoundsOfTracking";
/// Java `SOBEL_FILTER_CENTERING_KEY`.
pub const SOBEL_FILTER_CENTERING_KEY: &str = "SobelFilterCentering";
/// Java `KERNEL_SIGMA_FOR_SOBEL_KEY`.
pub const KERNEL_SIGMA_FOR_SOBEL_KEY: &str = "KernelSigmaForSobel";
/// Java `LOW_PASS_CUTOFF_INVERSE_NM_KEY`.
pub const LOW_PASS_CUTOFF_INVERSE_NM_KEY: &str = "LowPassCutoffInverseNm";
/// Java `SCALABLE_SIGMA_FOR_SOBEL_KEY`.  ScalableSigmaForSobel replaces
/// KernelSigmaForSobel.
pub const SCALABLE_SIGMA_FOR_SOBEL_KEY: &str = "ScalableSigmaForSobel";

/// Java nested `BeadtrackParam.Field` (a typesafe-enum `FieldInterface`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Field {
    /// Java `LIGHT_BEADS`.
    LightBeads,
}

impl FieldInterface for Field {}

/// Java final `BeadtrackParam`.
pub struct BeadtrackParam {
    /// Java superclass `OldBeadtrackParam` state.
    pub base: OldBeadtrackParam,
    manager: &'static dyn BaseManager,
    initialized: bool,
    /// was viewSkipList
    skip_views: StringList,
    /// was imageRotation
    rotation_angle: ScriptParameter,
    /// was tiltAngleGroupParams
    tilt_default_grouping: ScriptParameter,
    /// was magnificationGroupParams
    mag_default_grouping: ScriptParameter,
    /// was nMinViews
    min_views_for_tiltalign: ScriptParameter,
    /// was fiducialParams(0)
    centroid_radius: ScriptParameter,
    /// was fiducialParams(1)
    light_beads: EtomoBoolean2,
    /// was maxGap
    max_gap_size: ScriptParameter,
    /// was tiltAngleMinRange(0)
    min_tilt_range_to_find_axis: ScriptParameter,
    /// was tiltAngleMinRange(1)
    min_tilt_range_to_find_angles: ScriptParameter,
    /// was maxFiducialsAvg
    max_beads_to_average: ScriptParameter,
    /// was minRescueDistance
    distance_rescue_criterion: ScriptParameter,
    /// was residualDistanceLimit
    post_fit_rescue_residual: ScriptParameter,
    /// was secondPassParams(0)
    density_relaxation_post_fit: ScriptParameter,
    /// was secondPassParams(1)
    max_rescue_distance: ScriptParameter,

    local_area_tracking: EtomoBoolean2,
    local_area_target_size: ScriptParameter,
    min_beads_in_area: ScriptParameter,
    min_overlap_beads: ScriptParameter,
    max_views_in_align: ScriptParameter,
    rounds_of_tracking: ScriptParameter,
    images_are_binned: ScriptParameter,
    bead_diameter: ScriptParameter,
    sobel_filter_centering: EtomoBoolean2,
    kernel_sigma_for_sobel: ScriptParameter,
    low_pass_cutoff_inverse_nm: ScriptParameter,
    /// ScalableSigmaForSobel replaces KernelSigmaForSobel.
    scalable_sigma_for_sobel: ScriptParameter,

    axis_id: AxisID,
}

/// Java inheritance: every `OldBeadtrackParam` (and so `OldConstBeadtrackParam`)
/// member is reachable on a `BeadtrackParam`.
impl std::ops::Deref for BeadtrackParam {
    type Target = OldBeadtrackParam;

    fn deref(&self) -> &OldBeadtrackParam {
        &self.base
    }
}

impl std::ops::DerefMut for BeadtrackParam {
    fn deref_mut(&mut self) -> &mut OldBeadtrackParam {
        &mut self.base
    }
}

impl BeadtrackParam {
    /// Java `BeadtrackParam(AxisID, BaseManager)`.
    pub fn new(axis_id: AxisID, manager: &'static dyn BaseManager) -> BeadtrackParam {
        // The fields Java leaves null until `initialize()`; see the module
        // comment.
        let mut local_area_tracking = EtomoBoolean2::new_with_name(LOCAL_AREA_TRACKING_KEY);
        local_area_tracking.set_display_as_integer(true);
        BeadtrackParam {
            base: OldBeadtrackParam::new(),
            manager,
            initialized: false,
            skip_views: StringList::new_with_key(Some(SKIP_VIEW_LIST_KEY)),
            rotation_angle: ScriptParameter::new_with_type_and_name(
                Type::Double,
                IMAGE_ROTATION_KEY,
            ),
            tilt_default_grouping: ScriptParameter::new_with_type_and_name(
                Type::Integer,
                TILT_ANGLE_GROUP_PARAMS_KEY,
            ),
            mag_default_grouping: ScriptParameter::new_with_type_and_name(
                Type::Integer,
                MAGNIFICATION_GROUP_PARAMS_KEY,
            ),
            min_views_for_tiltalign: ScriptParameter::new_with_type_and_name(
                Type::Integer,
                N_MIN_VIEWS_KEY,
            ),
            centroid_radius: ScriptParameter::new_with_type_and_name(
                Type::Double,
                "CentroidRadius",
            ),
            light_beads: EtomoBoolean2::new_with_name(LIGHT_BEADS_KEY),
            max_gap_size: ScriptParameter::new_with_type_and_name(Type::Integer, MAX_GAP_KEY),
            min_tilt_range_to_find_axis: ScriptParameter::new_with_type_and_name(
                Type::Double,
                MIN_TILT_RANGE_TO_FIND_AXIS_KEY,
            ),
            min_tilt_range_to_find_angles: ScriptParameter::new_with_type_and_name(
                Type::Double,
                MIN_TILT_RANGE_TO_FIND_ANGLES_KEY,
            ),
            max_beads_to_average: ScriptParameter::new_with_type_and_name(
                Type::Integer,
                MAX_FIDUCIALS_AVG_KEY,
            ),
            distance_rescue_criterion: ScriptParameter::new_with_type_and_name(
                Type::Double,
                MIN_RESCUE_DISTANCE_KEY,
            ),
            post_fit_rescue_residual: ScriptParameter::new_with_type_and_name(
                Type::Double,
                RESIDUAL_DISTANCE_LIMIT_KEY,
            ),
            density_relaxation_post_fit: ScriptParameter::new_with_type_and_name(
                Type::Double,
                DENSITY_RELAXATION_POST_FIT_KEY,
            ),
            max_rescue_distance: ScriptParameter::new_with_type_and_name(
                Type::Double,
                MAX_RESCUE_DISTANCE_KEY,
            ),
            local_area_tracking,
            local_area_target_size: ScriptParameter::new_with_type_and_name(
                Type::Integer,
                LOCAL_AREA_TARGET_SIZE_KEY,
            ),
            min_beads_in_area: ScriptParameter::new_with_type_and_name(
                Type::Integer,
                MIN_BEADS_IN_AREA_KEY,
            ),
            min_overlap_beads: ScriptParameter::new_with_type_and_name(
                Type::Integer,
                MIN_OVERLAP_BEADS_KEY,
            ),
            max_views_in_align: ScriptParameter::new_with_type_and_name(
                Type::Integer,
                MAX_VIEWS_IN_ALIGN_KEY,
            ),
            rounds_of_tracking: ScriptParameter::new_with_type_and_name(
                Type::Integer,
                ROUNDS_OF_TRACKING_KEY,
            ),
            images_are_binned: ScriptParameter::new_with_name("ImagesAreBinned"),
            bead_diameter: ScriptParameter::new_with_type_and_name(Type::Double, "BeadDiameter"),
            sobel_filter_centering: EtomoBoolean2::new_with_name(SOBEL_FILTER_CENTERING_KEY),
            kernel_sigma_for_sobel: ScriptParameter::new_with_type_and_name(
                Type::Double,
                KERNEL_SIGMA_FOR_SOBEL_KEY,
            ),
            low_pass_cutoff_inverse_nm: ScriptParameter::new_with_type_and_name(
                Type::Double,
                LOW_PASS_CUTOFF_INVERSE_NM_KEY,
            ),
            scalable_sigma_for_sobel: ScriptParameter::new_with_type_and_name(
                Type::Double,
                SCALABLE_SIGMA_FOR_SOBEL_KEY,
            ),
            axis_id,
        }
    }

    // <p>Updates done</p>

    /// Java private `initialize`.
    fn initialize(&mut self) {
        if self.initialized {
            self.reset();
        }
        self.initialized = true;

        let required_map = self.get_required_map();
        let required_map = required_map.as_ref();
        self.skip_views = StringList::new_with_key(Some(SKIP_VIEW_LIST_KEY));
        self.rotation_angle =
            ScriptParameter::new_with_required_map(Type::Double, IMAGE_ROTATION_KEY, required_map);
        self.additional_view_groups
            .set_key(Some(ADDITIONAL_VIEW_GROUPS_KEY));
        self.additional_view_groups
            .set_successive_entries_accumulate();

        self.tilt_angle_spec
            .set_range_min_key(Some("FirstTiltAngle"), Some("first"));
        self.tilt_angle_spec
            .set_range_step_key(Some("TiltIncrement"), Some("increment"));
        self.tilt_angle_spec
            .set_tilt_angle_filename_key(Some("TiltFile"), Some("tiltfile"));
        self.tilt_angle_spec
            .set_tilt_angles_key(Some("TiltAngles"), Some("angles"));

        self.tilt_default_grouping = ScriptParameter::new_with_required_map(
            Type::Integer,
            TILT_ANGLE_GROUP_PARAMS_KEY,
            required_map,
        );
        self.mag_default_grouping = ScriptParameter::new_with_required_map(
            Type::Integer,
            MAGNIFICATION_GROUP_PARAMS_KEY,
            required_map,
        );
        self.min_views_for_tiltalign =
            ScriptParameter::new_with_required_map(Type::Integer, N_MIN_VIEWS_KEY, required_map);
        self.centroid_radius =
            ScriptParameter::new_with_type_and_name(Type::Double, "CentroidRadius");
        self.light_beads = EtomoBoolean2::new_with_required_map(LIGHT_BEADS_KEY, required_map);
        self.max_gap_size =
            ScriptParameter::new_with_required_map(Type::Integer, MAX_GAP_KEY, required_map);
        self.min_tilt_range_to_find_axis = ScriptParameter::new_with_required_map(
            Type::Double,
            MIN_TILT_RANGE_TO_FIND_AXIS_KEY,
            required_map,
        );
        self.min_tilt_range_to_find_angles = ScriptParameter::new_with_required_map(
            Type::Double,
            MIN_TILT_RANGE_TO_FIND_ANGLES_KEY,
            required_map,
        );
        self.max_beads_to_average = ScriptParameter::new_with_required_map(
            Type::Integer,
            MAX_FIDUCIALS_AVG_KEY,
            required_map,
        );
        self.rescue_attempt_params.set_integer_type_index(1, false);
        self.distance_rescue_criterion = ScriptParameter::new_with_required_map(
            Type::Double,
            MIN_RESCUE_DISTANCE_KEY,
            required_map,
        );
        self.post_fit_rescue_residual = ScriptParameter::new_with_required_map(
            Type::Double,
            RESIDUAL_DISTANCE_LIMIT_KEY,
            required_map,
        );
        self.density_relaxation_post_fit = ScriptParameter::new_with_required_map(
            Type::Double,
            DENSITY_RELAXATION_POST_FIT_KEY,
            required_map,
        );
        self.max_rescue_distance = ScriptParameter::new_with_required_map(
            Type::Double,
            MAX_RESCUE_DISTANCE_KEY,
            required_map,
        );
        self.deletion_params.set_integer_type_index(1, false);

        self.local_area_tracking =
            EtomoBoolean2::new_with_required_map(LOCAL_AREA_TRACKING_KEY, required_map);
        self.local_area_tracking.set_display_as_integer(true);

        self.local_area_target_size = ScriptParameter::new_with_required_map(
            Type::Integer,
            LOCAL_AREA_TARGET_SIZE_KEY,
            required_map,
        );
        self.min_beads_in_area = ScriptParameter::new_with_required_map(
            Type::Integer,
            MIN_BEADS_IN_AREA_KEY,
            required_map,
        );
        self.min_overlap_beads = ScriptParameter::new_with_required_map(
            Type::Integer,
            MIN_OVERLAP_BEADS_KEY,
            required_map,
        );
        self.max_views_in_align = ScriptParameter::new_with_required_map(
            Type::Integer,
            MAX_VIEWS_IN_ALIGN_KEY,
            required_map,
        );
        self.rounds_of_tracking = ScriptParameter::new_with_required_map(
            Type::Integer,
            ROUNDS_OF_TRACKING_KEY,
            required_map,
        );
        self.images_are_binned = ScriptParameter::new_with_name("ImagesAreBinned");
        self.bead_diameter = ScriptParameter::new_with_type_and_name(Type::Double, "BeadDiameter");
        self.sobel_filter_centering = EtomoBoolean2::new_with_name(SOBEL_FILTER_CENTERING_KEY);
        self.kernel_sigma_for_sobel =
            ScriptParameter::new_with_type_and_name(Type::Double, KERNEL_SIGMA_FOR_SOBEL_KEY);
        self.low_pass_cutoff_inverse_nm =
            ScriptParameter::new_with_type_and_name(Type::Double, LOW_PASS_CUTOFF_INVERSE_NM_KEY);
        // ScalableSigmaForSobel replaces KernelSigmaForSobel.
        self.scalable_sigma_for_sobel =
            ScriptParameter::new_with_type_and_name(Type::Double, SCALABLE_SIGMA_FOR_SOBEL_KEY);
    }

    // <p>Updates done</p>

    /// Java private `reset`.
    fn reset(&mut self) {
        if !self.initialized {
            self.initialize();
        }

        self.input_file = None;
        self.piece_list_file = None;
        self.seed_model_file = None;
        self.output_model_file = None;
        self.skip_views.reset();
        self.rotation_angle.reset();
        self.additional_view_groups.reset();
        self.tilt_angle_spec.reset();

        self.tilt_default_grouping.reset();
        self.tilt_angle_groups = None;
        self.mag_default_grouping.reset();
        self.magnification_groups = None;
        self.min_views_for_tiltalign.reset();
        self.centroid_radius.reset();
        self.light_beads.reset();
        self.fill_gaps = true;
        self.max_gap_size.reset();
        self.min_tilt_range_to_find_axis.reset();
        self.min_tilt_range_to_find_angles.reset();
        self.search_box_pixels.reset();
        self.max_beads_to_average.reset();

        self.fiducial_extrapolation_params.set_index_double(0, 7.0);
        self.fiducial_extrapolation_params.set_index_double(1, 3.0);

        self.rescue_attempt_params.reset();
        self.distance_rescue_criterion.reset();
        self.rescue_relaxation_params.reset();
        self.post_fit_rescue_residual.reset();
        self.density_relaxation_post_fit.reset();
        self.max_rescue_distance.reset();

        self.mean_resid_change_limits.set_index_double(0, 9.0);
        self.mean_resid_change_limits.set_index_double(1, 5.0);
        self.deletion_params.reset();

        self.local_area_tracking.reset();
        self.local_area_target_size.reset();
        self.min_beads_in_area.reset();
        self.min_overlap_beads.reset();
        self.max_views_in_align.reset();
        self.rounds_of_tracking.reset();
        self.images_are_binned.reset();
        self.bead_diameter.reset();
        self.sobel_filter_centering.reset();
        self.kernel_sigma_for_sobel.reset();
        self.low_pass_cutoff_inverse_nm.reset();
        self.scalable_sigma_for_sobel.reset();
    }

    /// Java private `getRequiredMap`.  The autodoc's map may hold null values;
    /// a null value and an absent key read the same through `HashMap.get`, so
    /// the null values are dropped.
    fn get_required_map(&self) -> Option<HashMap<String, String>> {
        let autodoc = match unsafe {
            autodoc_factory::get_instance(
                Some(self.manager),
                Some(autodoc_factory::BEADTRACK),
                self.axis_id,
                false,
            )
        } {
            Ok(autodoc) => autodoc,
            // `catch (final LockException e) {}`
            Err(LogFileError::Lock(_)) => std::ptr::null_mut(),
            // `catch (final LogFileException | IOException except)` ->
            // `except.printStackTrace()`
            Err(except) => {
                eprintln!("{except}");
                std::ptr::null_mut()
            }
        };
        if autodoc.is_null() {
            return None;
        }
        let values = unsafe {
            (*autodoc).get_attribute_values(
                Some(etomo_autodoc::FIELD_SECTION_NAME),
                Some(etomo_autodoc::REQUIRED_ATTRIBUTE_NAME),
            )
        };
        values.map(|values| {
            values
                .into_iter()
                .filter_map(|(key, value)| value.map(|value| (key, value)))
                .collect()
        })
    }

    /// Java `getSkipViews`.
    pub fn get_skip_views(&self) -> String {
        self.skip_views.to_string()
    }

    /// Java `getRotationAngle`.
    pub fn get_rotation_angle(&self) -> &ConstEtomoNumber {
        &self.rotation_angle
    }

    /// Java `getTiltDefaultGrouping`.
    pub fn get_tilt_default_grouping(&self) -> &ConstEtomoNumber {
        &self.tilt_default_grouping
    }

    /// Java `getMagnificationGroupSize`, overriding the deprecated
    /// `OldConstBeadtrackParam` version.
    pub fn get_magnification_group_size(&self) -> i32 {
        self.mag_default_grouping.get_int()
    }

    /// Java `getMinViewsForTiltalign`.
    pub fn get_min_views_for_tiltalign(&self) -> &ConstEtomoNumber {
        &self.min_views_for_tiltalign
    }

    /// Java `getLightBeads`.
    pub fn get_light_beads(&self) -> &ConstEtomoNumber {
        &self.light_beads
    }

    /// Java `getMaxGapSize`.
    pub fn get_max_gap_size(&self) -> &ConstEtomoNumber {
        &self.max_gap_size
    }

    /// Java `getMinTiltRangeToFindAxis`.
    pub fn get_min_tilt_range_to_find_axis(&self) -> &ConstEtomoNumber {
        &self.min_tilt_range_to_find_axis
    }

    /// Java `getMinTiltRangeToFindAngles`.
    pub fn get_min_tilt_range_to_find_angles(&self) -> &ConstEtomoNumber {
        &self.min_tilt_range_to_find_angles
    }

    /// Java `getMaxBeadsToAverage`.
    pub fn get_max_beads_to_average(&self) -> &ConstEtomoNumber {
        &self.max_beads_to_average
    }

    /// Java `getDistanceRescueCriterion`.
    pub fn get_distance_rescue_criterion(&self) -> &ConstEtomoNumber {
        &self.distance_rescue_criterion
    }

    /// Java `getPostFitRescueResidual`.
    pub fn get_post_fit_rescue_residual(&self) -> &ConstEtomoNumber {
        &self.post_fit_rescue_residual
    }

    /// Java `isBeadDiameterSet`.
    pub fn is_bead_diameter_set(&self) -> bool {
        !self.bead_diameter.is_null()
    }

    /// Java `getBeadDiameter`.
    pub fn get_bead_diameter(&self) -> &ConstEtomoNumber {
        &self.bead_diameter
    }

    /// Java `getDensityRelaxationPostFit`.
    pub fn get_density_relaxation_post_fit(&self) -> &ConstEtomoNumber {
        &self.density_relaxation_post_fit
    }

    /// Java `getMaxRescueDistance`.
    pub fn get_max_rescue_distance(&self) -> &ConstEtomoNumber {
        &self.max_rescue_distance
    }

    /// Java `getLocalAreaTracking`.
    pub fn get_local_area_tracking(&self) -> &ConstEtomoNumber {
        &self.local_area_tracking
    }

    /// Java `isSobelFilterCentering`.
    pub fn is_sobel_filter_centering(&self) -> bool {
        self.sobel_filter_centering.is()
    }

    /// Java `convertKernelSigmaForSobelToScalableSigmaForSobel`.  Converts
    /// kernelSigmaForSobel and sets ScalableSigmaForSobel to the result.
    /// Nothing is does if kernelSigmaForSobel is empty or stackFileType is null.
    pub fn convert_kernel_sigma_for_sobel_to_scalable_sigma_for_sobel(
        &mut self,
        fiducial_diameter: f64,
        stack_file_type: Option<&Arc<FileType>>,
    ) {
        if !self.kernel_sigma_for_sobel.is_null()
            && let Some(stack_file_type) = stack_file_type
        {
            let binning = utilities::get_stack_binning_for_file_type(
                self.manager,
                self.axis_id,
                stack_file_type,
            );
            self.scalable_sigma_for_sobel.set_double(
                self.kernel_sigma_for_sobel.get_double() / (fiducial_diameter / binning as f64),
            );
            self.kernel_sigma_for_sobel.reset();
        }
    }

    /// Java `getKernelSigmaForSobel`.
    pub fn get_kernel_sigma_for_sobel(&self) -> String {
        self.kernel_sigma_for_sobel.to_string()
    }

    /// Java `getScalableSigmaForSobel`.
    pub fn get_scalable_sigma_for_sobel(&self) -> String {
        self.scalable_sigma_for_sobel.to_string()
    }

    /// Java `getLowPassCutoffInverseNm`.
    pub fn get_low_pass_cutoff_inverse_nm(&self) -> String {
        self.low_pass_cutoff_inverse_nm.to_string()
    }

    /// Java `getLocalAreaTargetSize`.
    pub fn get_local_area_target_size(&self) -> &ConstEtomoNumber {
        &self.local_area_target_size
    }

    /// Java `getMinBeadsInArea`.
    pub fn get_min_beads_in_area(&self) -> &ConstEtomoNumber {
        &self.min_beads_in_area
    }

    /// Java `getMinOverlapBeads`.
    pub fn get_min_overlap_beads(&self) -> &ConstEtomoNumber {
        &self.min_overlap_beads
    }

    /// Java `getMaxViewsInAlign`.
    pub fn get_max_views_in_align(&self) -> &ConstEtomoNumber {
        &self.max_views_in_align
    }

    /// Java `getRoundsOfTracking`.
    pub fn get_rounds_of_tracking(&self) -> &ConstEtomoNumber {
        &self.rounds_of_tracking
    }

    // Non-const code

    /// Java private `convertToPIP`.
    fn convert_to_pip(
        &mut self,
        script_command: &ComScriptCommand,
    ) -> Result<(), ParseComScriptError> {
        let mut old_param = OldBeadtrackParam::new();
        old_param.parse_old_com_script_command(script_command)?;
        self.set(Some(&old_param.base));
        Ok(())
    }

    /// Java private `set(OldConstBeadtrackParam)`.
    fn set(&mut self, param: Option<&OldConstBeadtrackParam>) {
        let param = match param {
            None => panic!("java.lang.IllegalStateException: param is null"),
            Some(param) => param,
        };
        self.input_file = param.input_file.clone();
        self.piece_list_file = param.piece_list_file.clone();
        self.seed_model_file = param.seed_model_file.clone();
        self.output_model_file = param.output_model_file.clone();
        self.view_skip_list = param.view_skip_list.clone();
        self.rotation_angle.set_double(param.image_rotation);
        self.additional_view_groups
            .parse_string_string_list(Some(&param.additional_view_groups));
        self.tilt_angle_spec.set(Some(&param.tilt_angle_spec));
        // `param.tiltAngleGroupParams != null`: always constructed.
        self.tilt_default_grouping
            .set_fortran_input_string(&param.tilt_angle_group_params, 0);
        if let Some(tilt_angle_groups) = &param.tilt_angle_groups {
            let mut groups = Vec::with_capacity(tilt_angle_groups.len());
            for group in tilt_angle_groups {
                groups.push(FortranInputString::new_from_instance(group));
            }
            self.tilt_angle_groups = Some(groups);
        }
        // `param.magnificationGroupParams != null`: always constructed.
        self.mag_default_grouping
            .set_fortran_input_string(&param.magnification_group_params, 0);

        if let Some(magnification_groups) = &param.magnification_groups {
            let mut groups = Vec::with_capacity(magnification_groups.len());
            for group in magnification_groups {
                groups.push(FortranInputString::new_from_instance(group));
            }
            self.magnification_groups = Some(groups);
        }
        self.min_views_for_tiltalign.set_int(param.n_min_views);
        // Java tests this object's own `fiducialParams != null` (and below its
        // own `tiltAngleMinRange`) rather than param's; every one of these
        // FortranInputStrings is constructed in `OldConstBeadtrackParam()`, so
        // each test is always true.
        self.centroid_radius
            .set_fortran_input_string(&param.fiducial_params, 0);
        self.light_beads
            .set_fortran_input_string(&param.fiducial_params, 1);
        self.fill_gaps = param.fill_gaps;
        self.max_gap_size.set_int(param.max_gap);
        self.min_tilt_range_to_find_axis
            .set_fortran_input_string(&param.tilt_angle_min_range, 0);
        self.min_tilt_range_to_find_angles
            .set_fortran_input_string(&param.tilt_angle_min_range, 1);
        self.search_box_pixels
            .set_fortran_input_string(&param.search_box_pixels);
        self.max_beads_to_average.set_int(param.max_fiducials_avg);
        self.fiducial_extrapolation_params
            .set_fortran_input_string(&param.fiducial_extrapolation_params);
        self.rescue_attempt_params
            .set_fortran_input_string(&param.rescue_attempt_params);
        self.distance_rescue_criterion
            .set_int(param.min_rescue_distance);
        self.rescue_relaxation_params
            .set_fortran_input_string(&param.rescue_relaxation_params);
        self.post_fit_rescue_residual
            .set_double(param.residual_distance_limit);
        self.density_relaxation_post_fit
            .set_fortran_input_string(&param.second_pass_params, 0);
        self.max_rescue_distance
            .set_fortran_input_string(&param.second_pass_params, 1);
        self.mean_resid_change_limits
            .set_fortran_input_string(&param.mean_resid_change_limits);
        self.deletion_params
            .set_fortran_input_string(&param.deletion_params);

        self.local_area_target_size.set_int(1000);
        self.min_beads_in_area.set_int(8);
        self.min_overlap_beads.set_int(5);
        self.rounds_of_tracking.set_int(2);
    }

    /// Java `setSkipViews`.
    pub fn set_skip_views(&mut self, skip_views: Option<&str>) {
        self.skip_views.parse_string(skip_views);
    }

    /// Java `setRotationAngle(int)`.
    pub fn set_rotation_angle(&mut self, rotation_angle: i32) -> &ConstEtomoNumber {
        self.rotation_angle.set_int(rotation_angle);
        &self.rotation_angle
    }

    /// Java `setTiltDefaultGrouping`.
    pub fn set_tilt_default_grouping(
        &mut self,
        tilt_default_grouping: Option<&str>,
    ) -> &ConstEtomoNumber {
        self.tilt_default_grouping.set_string(tilt_default_grouping);
        &self.tilt_default_grouping
    }

    /// Java `setTiltAngleGroups(String)`, overriding the deprecated
    /// `OldBeadtrackParam` version.
    pub fn set_tilt_angle_groups(
        &mut self,
        new_tilt_angle_groups: Option<&str>,
    ) -> Result<(), FortranInputSyntaxException> {
        self.tilt_angle_groups =
            param_utilities::parse_string(new_tilt_angle_groups, true, NONDEFAULT_GROUP_SIZE)?;
        Ok(())
    }

    /// Java `setMagDefaultGrouping`.
    pub fn set_mag_default_grouping(
        &mut self,
        magnification_group_params: Option<&str>,
    ) -> &ConstEtomoNumber {
        self.mag_default_grouping
            .set_string(magnification_group_params);
        &self.mag_default_grouping
    }

    /// Java `setMagnificationGroups(String)`, overriding the deprecated
    /// `OldBeadtrackParam` version.
    pub fn set_magnification_groups(
        &mut self,
        new_magnification_groups: Option<&str>,
    ) -> Result<(), FortranInputSyntaxException> {
        self.magnification_groups =
            param_utilities::parse_string(new_magnification_groups, true, NONDEFAULT_GROUP_SIZE)?;
        Ok(())
    }

    /// Java `setMinViewsForTiltalign`.
    pub fn set_min_views_for_tiltalign(
        &mut self,
        min_views_for_tiltalign: Option<&str>,
    ) -> &ConstEtomoNumber {
        self.min_views_for_tiltalign
            .set_string(min_views_for_tiltalign);
        &self.min_views_for_tiltalign
    }

    /// Java `setLightBeads`.
    pub fn set_light_beads(&mut self, light_beads: bool) -> &ConstEtomoNumber {
        self.light_beads.set_boolean(light_beads)
    }

    /// Java `setMaxGapSize`.
    pub fn set_max_gap_size(&mut self, max_gap_size: Option<&str>) -> &ConstEtomoNumber {
        self.max_gap_size.set_string(max_gap_size);
        &self.max_gap_size
    }

    /// Java `setMinTiltRangeToFindAxis`.
    pub fn set_min_tilt_range_to_find_axis(
        &mut self,
        min_tilt_range_to_find_axis: Option<&str>,
    ) -> &ConstEtomoNumber {
        self.min_tilt_range_to_find_axis
            .set_string(min_tilt_range_to_find_axis);
        &self.min_tilt_range_to_find_axis
    }

    /// Java `setMinTiltRangeToFindAngles`.
    pub fn set_min_tilt_range_to_find_angles(
        &mut self,
        min_tilt_range_to_find_angles: Option<&str>,
    ) -> &ConstEtomoNumber {
        self.min_tilt_range_to_find_angles
            .set_string(min_tilt_range_to_find_angles);
        &self.min_tilt_range_to_find_angles
    }

    /// Java `setMaxBeadsToAverage`.
    pub fn set_max_beads_to_average(
        &mut self,
        max_beads_to_average: Option<&str>,
    ) -> &ConstEtomoNumber {
        self.max_beads_to_average.set_string(max_beads_to_average);
        &self.max_beads_to_average
    }

    /// Java `setDistanceRescueCriterion`.
    pub fn set_distance_rescue_criterion(
        &mut self,
        distance_rescue_criterion: Option<&str>,
    ) -> &ConstEtomoNumber {
        self.distance_rescue_criterion
            .set_string(distance_rescue_criterion);
        &self.distance_rescue_criterion
    }

    /// Java `setPostFitRescueResidual`.
    pub fn set_post_fit_rescue_residual(
        &mut self,
        post_fit_rescue_residual: Option<&str>,
    ) -> &ConstEtomoNumber {
        self.post_fit_rescue_residual
            .set_string(post_fit_rescue_residual);
        &self.post_fit_rescue_residual
    }

    /// Java `setBeadDiameter`.
    pub fn set_bead_diameter(&mut self, input: Option<&str>) -> &ConstEtomoNumber {
        self.bead_diameter.set_string(input);
        &self.bead_diameter
    }

    /// Java `setDensityRelaxationPostFit`.
    pub fn set_density_relaxation_post_fit(
        &mut self,
        density_relaxation_post_fit: Option<&str>,
    ) -> &ConstEtomoNumber {
        self.density_relaxation_post_fit
            .set_string(density_relaxation_post_fit);
        &self.density_relaxation_post_fit
    }

    /// Java `setMaxRescueDistance`.
    pub fn set_max_rescue_distance(
        &mut self,
        max_rescue_distance: Option<&str>,
    ) -> &ConstEtomoNumber {
        self.max_rescue_distance.set_string(max_rescue_distance);
        &self.max_rescue_distance
    }

    /// Java `setLocalAreaTracking`.
    pub fn set_local_area_tracking(&mut self, local_area_tracking: bool) -> &ConstEtomoNumber {
        self.local_area_tracking.set_boolean(local_area_tracking)
    }

    /// Java `setSobelFilterCentering`.
    pub fn set_sobel_filter_centering(&mut self, input: bool) {
        self.sobel_filter_centering.set_boolean(input);
    }

    /// Java `setScalableSigmaForSobel`.
    pub fn set_scalable_sigma_for_sobel(&mut self, input: Option<&str>) {
        self.scalable_sigma_for_sobel.set_string(input);
    }

    /// Java `setLowPassCutoffInverseNm`.
    pub fn set_low_pass_cutoff_inverse_nm(&mut self, input: Option<&str>) {
        self.low_pass_cutoff_inverse_nm.set_string(input);
    }

    /// Java `setLocalAreaTargetSize`.
    pub fn set_local_area_target_size(
        &mut self,
        local_area_target_size: Option<&str>,
    ) -> &ConstEtomoNumber {
        self.local_area_target_size
            .set_string(local_area_target_size);
        &self.local_area_target_size
    }

    /// Java `setMinBeadsInArea`.
    pub fn set_min_beads_in_area(&mut self, min_beads_in_area: Option<&str>) -> &ConstEtomoNumber {
        self.min_beads_in_area.set_string(min_beads_in_area);
        &self.min_beads_in_area
    }

    /// Java `setMinOverlapBeads`.
    pub fn set_min_overlap_beads(&mut self, min_overlap_beads: Option<&str>) -> &ConstEtomoNumber {
        self.min_overlap_beads.set_string(min_overlap_beads);
        &self.min_overlap_beads
    }

    /// Java `setImagesAreBinned`.
    pub fn set_images_are_binned(&mut self, input: i32) {
        self.images_are_binned.set_int(input);
    }

    /// Java `setMaxViewsInAlign`.
    pub fn set_max_views_in_align(
        &mut self,
        max_views_in_align: Option<&str>,
    ) -> &ConstEtomoNumber {
        self.max_views_in_align.set_string(max_views_in_align);
        &self.max_views_in_align
    }

    /// Java `setRoundsOfTracking`.
    pub fn set_rounds_of_tracking(
        &mut self,
        rounds_of_tracking: Option<&str>,
    ) -> &ConstEtomoNumber {
        self.rounds_of_tracking.set_string(rounds_of_tracking);
        &self.rounds_of_tracking
    }
}

impl CommandParam for BeadtrackParam {
    /// Java `parseComScriptCommand`.  Get the parameters from the
    /// ComScriptCommand.
    fn parse_com_script_command(
        &mut self,
        script_command: &ComScriptCommand,
    ) -> Result<(), ParseComScriptError> {
        self.reset();
        if !script_command.is_keyword_value_pairs() {
            self.convert_to_pip(script_command)?;
        } else {
            self.input_file = script_command.get_value(Some(INPUT_FILE_KEY))?;
            self.piece_list_file = script_command.get_value(Some(PIECE_LIST_FILE_KEY))?;
            self.seed_model_file = script_command.get_value(Some(SEED_MODEL_FILE_KEY))?;
            self.output_model_file = script_command.get_value(Some(OUTPUT_MODEL_FILE_KEY))?;
            self.skip_views.parse(script_command)?;
            self.rotation_angle.parse(script_command)?;
            self.additional_view_groups.parse(script_command)?;
            self.tilt_angle_spec.parse(script_command)?;
            self.tilt_default_grouping.parse(script_command)?;
            self.tilt_angle_groups =
                param_utilities::set_param_if_present_fortran_input_string_array(
                    script_command,
                    TILT_ANGLE_GROUPS_KEY,
                    NONDEFAULT_GROUP_SIZE,
                    &NONDEFAULT_GROUP_INTEGER_TYPE,
                )?;
            self.mag_default_grouping.parse(script_command)?;
            self.magnification_groups =
                param_utilities::set_param_if_present_fortran_input_string_array(
                    script_command,
                    MAGNIFICATION_GROUPS_KEY,
                    NONDEFAULT_GROUP_SIZE,
                    &NONDEFAULT_GROUP_INTEGER_TYPE,
                )?;
            self.min_views_for_tiltalign.parse(script_command)?;
            self.centroid_radius.parse(script_command)?;
            self.light_beads.parse(script_command)?;
            self.fill_gaps = script_command.has_keyword(Some(FILL_GAPS_KEY))?;
            self.max_gap_size.parse(script_command)?;
            self.min_tilt_range_to_find_axis.parse(script_command)?;
            self.min_tilt_range_to_find_angles.parse(script_command)?;
            let value = script_command.get_value(Some(SEARCH_BOX_PIXELS_KEY))?;
            self.search_box_pixels.validate_and_set(value.as_deref())?;
            self.max_beads_to_average.parse(script_command)?;
            let value = script_command.get_value(Some(FIDUCIAL_EXTRAPOLATION_PARAMS_KEY))?;
            self.fiducial_extrapolation_params
                .validate_and_set(value.as_deref())?;
            let value = script_command.get_value(Some(RESCUE_ATTEMPT_PARAMS_KEY))?;
            self.rescue_attempt_params
                .validate_and_set(value.as_deref())?;
            self.distance_rescue_criterion.parse(script_command)?;
            let value = script_command.get_value(Some(RESCUE_RELAXATION_PARAMS_KEY))?;
            self.rescue_relaxation_params
                .validate_and_set(value.as_deref())?;
            self.post_fit_rescue_residual.parse(script_command)?;
            self.density_relaxation_post_fit.parse(script_command)?;
            self.max_rescue_distance.parse(script_command)?;
            let value = script_command.get_value(Some(MEAN_RESID_CHANGE_LIMITS_KEY))?;
            self.mean_resid_change_limits
                .validate_and_set(value.as_deref())?;
            let value = script_command.get_value(Some(DELETION_PARAMS_KEY))?;
            self.deletion_params.validate_and_set(value.as_deref())?;

            self.local_area_tracking.parse(script_command)?;
            self.local_area_target_size.parse(script_command)?;
            self.min_beads_in_area.parse(script_command)?;
            self.min_overlap_beads.parse(script_command)?;
            self.max_views_in_align.parse(script_command)?;
            self.rounds_of_tracking.parse(script_command)?;
            self.bead_diameter.parse(script_command)?;
            self.sobel_filter_centering.parse(script_command)?;
            self.kernel_sigma_for_sobel.parse(script_command)?;
            self.low_pass_cutoff_inverse_nm.parse(script_command)?;
            self.scalable_sigma_for_sobel.parse(script_command)?;
        }
        // backward compatibility bug# 1160
        if !self.centroid_radius.is_null() {
            if self.bead_diameter.is_null() {
                let bead_diameter = 2.0 * self.centroid_radius.get_double() - 3.0;
                self.bead_diameter.set_double(bead_diameter);
            }
            self.centroid_radius.reset();
        }
        Ok(())
    }

    /// Java `initializeDefaults`.
    fn initialize_defaults(&mut self) {
        self.initialize();
    }

    /// Java `updateComScriptCommand`.  Update the supplied ComScriptCommand with
    /// the parameters of this object.
    fn update_com_script_command(
        &self,
        script_command: &mut ComScriptCommand,
    ) -> Result<(), BadComScriptException> {
        let invalid_reason = self.validate();
        if let Some(invalid_reason) = &invalid_reason
            && !java_lang_string_matches_whitespace(invalid_reason)
        {
            return Err(BadComScriptException::new(invalid_reason));
        }
        // Switch to keyword/value pairs
        script_command.use_keyword_value();

        param_utilities::update_script_parameter_string(
            script_command,
            Some(INPUT_FILE_KEY),
            self.input_file.as_deref(),
        )?;
        param_utilities::update_script_parameter_string(
            script_command,
            Some(PIECE_LIST_FILE_KEY),
            self.piece_list_file.as_deref(),
        )?;
        param_utilities::update_script_parameter_string(
            script_command,
            Some(SEED_MODEL_FILE_KEY),
            self.seed_model_file.as_deref(),
        )?;
        param_utilities::update_script_parameter_string(
            script_command,
            Some(OUTPUT_MODEL_FILE_KEY),
            self.output_model_file.as_deref(),
        )?;
        param_utilities::update_script_parameter_string(
            script_command,
            Some(OUTPUT_MODEL_FILE_KEY),
            self.output_model_file.as_deref(),
        )?;
        param_utilities::update_script_parameter_string_list(
            script_command,
            self.skip_views.get_key(),
            Some(&self.skip_views),
        )?;
        self.rotation_angle.update_com_script(script_command);
        // ParamUtilities.updateScriptParameter(scriptCommand, additionalViewGroups
        // .getKey(), additionalViewGroups);
        self.additional_view_groups
            .update_com_script(script_command)?;
        self.tilt_angle_spec.update_com_script(script_command)?;
        self.tilt_default_grouping.update_com_script(script_command);
        param_utilities::update_script_parameter_fortran_input_string_array(
            script_command,
            Some(TILT_ANGLE_GROUPS_KEY),
            self.tilt_angle_groups.as_deref(),
        );
        self.mag_default_grouping.update_com_script(script_command);
        param_utilities::update_script_parameter_fortran_input_string_array(
            script_command,
            Some(MAGNIFICATION_GROUPS_KEY),
            self.magnification_groups.as_deref(),
        );
        self.min_views_for_tiltalign
            .update_com_script(script_command);
        self.centroid_radius.update_com_script(script_command);
        self.light_beads.update_com_script(script_command);
        param_utilities::update_script_parameter_boolean(
            script_command,
            Some(FILL_GAPS_KEY),
            self.fill_gaps,
        );
        self.max_gap_size.update_com_script(script_command);

        self.local_area_tracking.update_com_script(script_command);
        self.local_area_target_size
            .update_com_script(script_command);
        self.min_beads_in_area.update_com_script(script_command);
        self.min_overlap_beads.update_com_script(script_command);
        self.max_views_in_align.update_com_script(script_command);
        self.rounds_of_tracking.update_com_script(script_command);
        self.sobel_filter_centering
            .update_com_script(script_command);
        // kernelSigmaForSobel.updateComScript(scriptCommand);
        self.low_pass_cutoff_inverse_nm
            .update_com_script(script_command);
        self.scalable_sigma_for_sobel
            .update_com_script(script_command);

        self.min_tilt_range_to_find_axis
            .update_com_script(script_command);
        self.min_tilt_range_to_find_angles
            .update_com_script(script_command);
        param_utilities::update_script_parameter_fortran_input_string(
            script_command,
            Some(SEARCH_BOX_PIXELS_KEY),
            &self.search_box_pixels,
        );
        self.max_beads_to_average.update_com_script(script_command);
        param_utilities::update_script_parameter_fortran_input_string(
            script_command,
            Some(FIDUCIAL_EXTRAPOLATION_PARAMS_KEY),
            &self.fiducial_extrapolation_params,
        );
        param_utilities::update_script_parameter_fortran_input_string(
            script_command,
            Some(RESCUE_ATTEMPT_PARAMS_KEY),
            &self.rescue_attempt_params,
        );
        self.distance_rescue_criterion
            .update_com_script(script_command);
        param_utilities::update_script_parameter_fortran_input_string(
            script_command,
            Some(RESCUE_RELAXATION_PARAMS_KEY),
            &self.rescue_relaxation_params,
        );
        self.post_fit_rescue_residual
            .update_com_script(script_command);
        self.density_relaxation_post_fit
            .update_com_script(script_command);
        self.max_rescue_distance.update_com_script(script_command);
        param_utilities::update_script_parameter_fortran_input_string(
            script_command,
            Some(MEAN_RESID_CHANGE_LIMITS_KEY),
            &self.mean_resid_change_limits,
        );
        param_utilities::update_script_parameter_fortran_input_string(
            script_command,
            Some(DELETION_PARAMS_KEY),
            &self.deletion_params,
        );
        self.images_are_binned.update_com_script(script_command);
        self.bead_diameter.update_com_script(script_command);
        Ok(())
    }
}

impl Command for BeadtrackParam {
    /// Java `getAxisID`.
    fn get_axis_id(&self) -> AxisID {
        self.axis_id
    }

    /// Java `getCommandMode`.
    fn get_command_mode(&self) -> Option<&dyn CommandMode> {
        None
    }

    /// Java `getProcessName`.
    fn get_process_name(&self) -> Option<ProcessName> {
        Some(PROCESS_NAME)
    }

    /// Java `getCommand`.
    fn get_command(&self) -> Option<String> {
        Some(format!("track{}.com", self.axis_id.get_extension()))
    }

    /// Java `getCommandName`.
    fn get_command_name(&self) -> Option<String> {
        Some(PROCESS_NAME.to_string())
    }

    /// Java `getCommandLine`.
    fn get_command_line(&self) -> Option<String> {
        file_type::CLASS
            .track_comscript
            .get_file_name(Some(self.manager), Some(self.axis_id))
    }

    /// Java `getCommandArray`: `{ getCommandLine() }`.  A null command line
    /// would be a one-element array holding null; that is no array here.
    fn get_command_array(&self) -> Option<Vec<String>> {
        self.get_command_line()
            .map(|command_line| vec![command_line])
    }

    /// Java `getCommandInputFile`.
    fn get_command_input_file(&self) -> Option<PathBuf> {
        None
    }

    /// Java `getCommandOutputFile`.
    fn get_command_output_file(&self) -> Option<PathBuf> {
        None
    }

    /// Java `getOutputImageFileType` (deprecated 3/15/2019).
    fn get_output_image_file_type(&self) -> Option<Arc<FileType>> {
        None
    }

    /// Java `getOutputImageFileKey`.
    fn get_output_image_file_key(&self) -> Option<FileKey> {
        None
    }

    /// Java `getOutputImageFileType2` (deprecated 3/15/2019).
    fn get_output_image_file_type2(&self) -> Option<Arc<FileType>> {
        None
    }

    /// Java `getOutputImageFileKey2`.
    fn get_output_image_file_key2(&self) -> Option<FileKey> {
        None
    }

    /// Java `isMessageReporter`.
    fn is_message_reporter(&self) -> bool {
        false
    }

    /// Java `getSubcommandProcessName`.
    fn get_subcommand_process_name(&self) -> Option<String> {
        None
    }

    /// Java `getSubcommandDetails`.
    fn get_subcommand_details(&self) -> Option<&dyn CommandDetails> {
        None
    }

    /// Java `CommandDetails` cast: this param is its own `ProcessDetails`.
    fn get_process_details(&self) -> Option<&dyn ProcessDetails> {
        Some(self)
    }
}

/// Java `ProcessDetails` half of `CommandDetails`.  For a field it does not
/// handle, each Java getter throws `IllegalArgumentException("field=" +
/// field)`; here it returns `None`.
impl ProcessDetails for BeadtrackParam {
    fn get_int_value(&self, _field: &dyn FieldInterface) -> Option<i32> {
        None
    }

    fn get_boolean_value(&self, field: &dyn FieldInterface) -> Option<bool> {
        if field_interface::as_field::<Field>(field) == Some(&Field::LightBeads) {
            return Some(self.light_beads.is());
        }
        None
    }

    fn get_double_value(&self, _field: &dyn FieldInterface) -> Option<f64> {
        None
    }

    fn get_hashtable(&self, _field: &dyn FieldInterface) -> Option<Vec<(String, String)>> {
        None
    }

    fn get_etomo_number(&self, _field: &dyn FieldInterface) -> Option<ConstEtomoNumber> {
        None
    }

    fn get_int_key_list(&self, _field: &dyn FieldInterface) -> Option<Vec<(i32, String)>> {
        None
    }

    fn get_string(&self, _field: &dyn FieldInterface) -> Option<String> {
        None
    }

    fn get_string_array(&self, _field: &dyn FieldInterface) -> Option<Vec<String>> {
        None
    }

    fn get_iterator_element_list(&self, _field: &dyn FieldInterface) -> Option<Vec<i32>> {
        None
    }
}

/// Java `Loggable` (via `ProcessDetails`).
impl Loggable for BeadtrackParam {
    /// Java `getName`.
    fn get_name(&self) -> String {
        PROCESS_NAME.to_string()
    }

    /// Java `getLogMessage`, which returns null: nothing to log.
    fn get_log_message(&self) -> Result<Vec<Option<String>>, LoggableException> {
        Ok(Vec::new())
    }
}
