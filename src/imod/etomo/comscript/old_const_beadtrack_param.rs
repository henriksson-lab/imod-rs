//! `IMOD/Etomo/src/etomo/comscript/OldConstBeadtrackParam.java`.
//!
//! Was ConstBeadtrack.  The pre-PIP beadtrack parameters, kept so that an old
//! (sequential standard input) `track.com` can be read and converted by
//! `BeadtrackParam.convertToPIP`.  Java's package-private fields are
//! `pub(crate)` here because the subclasses `OldBeadtrackParam` and
//! `BeadtrackParam` (which hold this struct as `base`, see
//! `old_beadtrack_param.rs`) read and write them directly.

use super::fortran_input_string::FortranInputString;
use super::param_utilities;
use super::string_list::StringList;
use crate::imod::etomo::r#type::tilt_angle_spec::TiltAngleSpec;

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";
/// Java package-private static `nondefaultGroupSize`.
pub(crate) const NONDEFAULT_GROUP_SIZE: i32 = 3;
/// Java package-private static `nondefaultGroupIntegerType`.
pub(crate) const NONDEFAULT_GROUP_INTEGER_TYPE: [bool; 3] = [true, true, true];

/// Java package-private class `OldConstBeadtrackParam`.
#[derive(Clone, Debug)]
pub struct OldConstBeadtrackParam {
    /// corresponds to ImageFile
    pub(crate) input_file: Option<String>,
    /// corresponds to PieceListFile
    pub(crate) piece_list_file: Option<String>,
    /// corresponds to InputSeedModel
    pub(crate) seed_model_file: Option<String>,
    /// corresponds to OutputModel
    pub(crate) output_model_file: Option<String>,
    /// changed to SkipViews
    pub(crate) view_skip_list: Option<String>,
    /// changed to RotationAngle
    pub(crate) image_rotation: f64,
    /// not in use
    pub(crate) n_additional_view_sets: i32,
    /// corresponds to SeparateGroup
    pub(crate) additional_view_groups: StringList,
    /// corresponds to FirstTiltAngle,TiltIncrement,TiltFile,TiltAngles
    pub(crate) tilt_angle_spec: TiltAngleSpec,
    /// changed to TiltDefaultGrouping
    pub(crate) tilt_angle_group_params: FortranInputString,
    /// corresponds to TiltNondefaultGroup
    pub(crate) tilt_angle_groups: Option<Vec<FortranInputString>>,
    /// changed to MagDefaultGrouping
    pub(crate) magnification_group_params: FortranInputString,
    /// corresponds to MagNondefaultGroup
    pub(crate) magnification_groups: Option<Vec<FortranInputString>>,
    /// changed to MinViewsForTiltalign
    pub(crate) n_min_views: i32,
    /// changed to CentroidRadius,LightBeads
    pub(crate) fiducial_params: FortranInputString,
    /// corresponds to FillGaps
    pub(crate) fill_gaps: bool,
    /// changed to MaxGapSize
    pub(crate) max_gap: i32,
    /// changed to MinTiltRangeToFindAxis,MinTiltRangeToFindAngles
    pub(crate) tilt_angle_min_range: FortranInputString,
    /// corresponds to BoxSizeXandY
    pub(crate) search_box_pixels: FortranInputString,
    /// changed to MaxBeadsToAverage
    pub(crate) max_fiducials_avg: i32,
    /// corresponds to PointsToFitMaxAndMin
    pub(crate) fiducial_extrapolation_params: FortranInputString,
    /// corresponds to DensityRescueFractionAndSD
    pub(crate) rescue_attempt_params: FortranInputString,
    /// changed to DistanceRescueCriterion
    pub(crate) min_rescue_distance: i32,
    /// corresponds to RescueRelaxationDensityAndDistance
    pub(crate) rescue_relaxation_params: FortranInputString,
    /// changed to PostFitRescueResidual
    pub(crate) residual_distance_limit: f64,
    /// changed to DensityRelaxationPostFit,MaxRescueDistance
    pub(crate) second_pass_params: FortranInputString,
    /// corresponds to ResidualsToAnalyzeMaxAndMin
    pub(crate) mean_resid_change_limits: FortranInputString,
    /// corresponds to DeletionCriterionMinAndSD
    pub(crate) deletion_params: FortranInputString,
}

impl OldConstBeadtrackParam {
    /// Java package-private `OldConstBeadtrackParam()`.  The attributes of this
    /// class are initialized in the constructor because many of them required
    /// additional calls besides construction for appropriate initialization.
    pub(crate) fn new() -> OldConstBeadtrackParam {
        let additional_view_groups = StringList::new_with_n_elements(0);

        let tilt_angle_spec = TiltAngleSpec::new();

        let mut tilt_angle_group_params = FortranInputString::new(2);
        tilt_angle_group_params.set_integer_type_index(0, true);
        tilt_angle_group_params.set_integer_type_index(1, true);

        let mut magnification_group_params = FortranInputString::new(2);
        magnification_group_params.set_integer_type_index(0, true);
        magnification_group_params.set_integer_type_index(1, true);

        let mut fiducial_params = FortranInputString::new(2);
        fiducial_params.set_integer_type_index(1, true);
        let tilt_angle_min_range = FortranInputString::new(2);

        let mut search_box_pixels = FortranInputString::new(2);
        search_box_pixels.set_integer_type_index(0, true);
        search_box_pixels.set_integer_type_index(1, true);

        let mut fiducial_extrapolation_params = FortranInputString::new(2);
        fiducial_extrapolation_params.set_integer_type_index(0, true);
        fiducial_extrapolation_params.set_integer_type_index(1, true);

        let mut rescue_attempt_params = FortranInputString::new(2);
        rescue_attempt_params.set_integer_type_index(1, true);

        let rescue_relaxation_params = FortranInputString::new(2);

        let second_pass_params = FortranInputString::new(2);

        let mut mean_resid_change_limits = FortranInputString::new(2);
        mean_resid_change_limits.set_integer_type_index(0, true);
        mean_resid_change_limits.set_integer_type_index(1, true);

        let mut deletion_params = FortranInputString::new(2);
        deletion_params.set_integer_type_index(0, false);
        deletion_params.set_integer_type_index(1, true);

        OldConstBeadtrackParam {
            input_file: None,
            piece_list_file: None,
            seed_model_file: None,
            output_model_file: None,
            view_skip_list: None,
            image_rotation: 0.0,
            n_additional_view_sets: 0,
            additional_view_groups,
            tilt_angle_spec,
            tilt_angle_group_params,
            tilt_angle_groups: None,
            magnification_group_params,
            magnification_groups: None,
            n_min_views: 0,
            fiducial_params,
            fill_gaps: false,
            max_gap: 0,
            tilt_angle_min_range,
            search_box_pixels,
            max_fiducials_avg: 0,
            fiducial_extrapolation_params,
            rescue_attempt_params,
            min_rescue_distance: 0,
            rescue_relaxation_params,
            residual_distance_limit: 0.0,
            second_pass_params,
            mean_resid_change_limits,
            deletion_params,
        }
    }

    /// Java package-private `validate()`.  Validate the parameters stored in the
    /// BeadtrackObject.
    pub(crate) fn validate(&self) -> Option<String> {
        let mut errors: Option<String> = None;
        // Compare the number of additional view sets and the number of entries in
        // in additionalViewGroups
        if self.n_additional_view_sets != self.additional_view_groups.get_n_elements() {
            let mut buffer = String::from(
                "The number of additional view groups does not equal the number specified",
            );
            buffer.push_str(&format!(
                "\nnumber of additional views sets: {}",
                self.n_additional_view_sets
            ));
            buffer.push_str(&format!(
                "\nnumber of list entries: {}",
                self.additional_view_groups.get_n_elements()
            ));
            errors = Some(buffer);
        }
        errors
    }

    /// Java `getInputFile`.
    pub(crate) fn get_input_file(&self) -> Option<&str> {
        self.input_file.as_deref()
    }

    /// Java `getPieceListFile`.
    pub(crate) fn get_piece_list_file(&self) -> Option<&str> {
        self.piece_list_file.as_deref()
    }

    /// Java `getSeedModelFile`.
    pub(crate) fn get_seed_model_file(&self) -> Option<&str> {
        self.seed_model_file.as_deref()
    }

    /// Java `getOutputModelFile`.
    pub(crate) fn get_output_model_file(&self) -> Option<&str> {
        self.output_model_file.as_deref()
    }

    /// Java `getViewSkipList` (deprecated).
    pub(crate) fn get_view_skip_list(&self) -> Option<&str> {
        self.view_skip_list.as_deref()
    }

    /// Java `getImageRotation` (deprecated).
    pub(crate) fn get_image_rotation(&self) -> f64 {
        self.image_rotation
    }

    /// Java `getNAdditionalViewSets` (deprecated).
    pub(crate) fn get_n_additional_view_sets(&self) -> i32 {
        self.n_additional_view_sets
    }

    /// Java public `getAdditionalViewGroups`.
    pub fn get_additional_view_groups(&self) -> String {
        self.additional_view_groups.to_string()
    }

    /// Java `getTiltAngleSpec`.  Returns a copy.
    pub(crate) fn get_tilt_angle_spec(&self) -> TiltAngleSpec {
        TiltAngleSpec::new_from_instance(&self.tilt_angle_spec)
    }

    /// Java `getTiltAngleGroupParams` (deprecated).
    pub(crate) fn get_tilt_angle_group_params(&self) -> String {
        self.tilt_angle_group_params.to_string()
    }

    /// Java `getTiltAngleGroupSize` (deprecated).
    pub(crate) fn get_tilt_angle_group_size(&self) -> i32 {
        self.tilt_angle_group_params.get_int(0)
    }

    /// Java public `getTiltAngleGroups`.
    pub fn get_tilt_angle_groups(&self) -> String {
        param_utilities::value_of_fortran_input_string_array(self.tilt_angle_groups.as_deref())
    }

    /// Java `getMagnificationGroupParams` (deprecated).
    pub(crate) fn get_magnification_group_params(&self) -> String {
        self.magnification_group_params.to_string()
    }

    /// Java `getMagnificationGroupSize` (deprecated).  `BeadtrackParam` overrides
    /// it; see `BeadtrackParam::get_magnification_group_size`.
    pub(crate) fn get_magnification_group_size(&self) -> i32 {
        self.magnification_group_params.get_int(0)
    }

    /// Java public `getMagnificationGroups`.
    pub fn get_magnification_groups(&self) -> String {
        param_utilities::value_of_fortran_input_string_array(self.magnification_groups.as_deref())
    }

    /// Java `getNMinViews` (deprecated).
    pub(crate) fn get_n_min_views(&self) -> i32 {
        self.n_min_views
    }

    /// Java `getFiducialParams` (deprecated).
    pub(crate) fn get_fiducial_params(&self) -> String {
        self.fiducial_params.to_string()
    }

    /// Java public `getFillGaps`.
    pub fn get_fill_gaps(&self) -> bool {
        self.fill_gaps
    }

    /// Java `getMaxGap` (deprecated).
    pub(crate) fn get_max_gap(&self) -> i32 {
        self.max_gap
    }

    /// Java `getTiltAngleMinRange` (deprecated).
    pub(crate) fn get_tilt_angle_min_range(&self) -> String {
        self.tilt_angle_min_range.to_string()
    }

    /// Java public `getSearchBoxPixels`.
    pub fn get_search_box_pixels(&self) -> String {
        self.search_box_pixels.to_string()
    }

    /// Java `getMaxFiducialsAvg` (deprecated).
    pub(crate) fn get_max_fiducials_avg(&self) -> i32 {
        self.max_fiducials_avg
    }

    /// Java public `getFiducialExtrapolationParams`.
    pub fn get_fiducial_extrapolation_params(&self) -> String {
        self.fiducial_extrapolation_params.to_string()
    }

    /// Java public `getRescueAttemptParams`.
    pub fn get_rescue_attempt_params(&self) -> String {
        self.rescue_attempt_params.to_string()
    }

    /// Java `getMinRescueDistance` (deprecated).
    pub(crate) fn get_min_rescue_distance(&self) -> i32 {
        self.min_rescue_distance
    }

    /// Java public `getRescueRelaxationParams`.
    pub fn get_rescue_relaxation_params(&self) -> String {
        self.rescue_relaxation_params.to_string()
    }

    /// Java `getResidualDistanceLimit` (deprecated).
    pub(crate) fn get_residual_distance_limit(&self) -> f64 {
        self.residual_distance_limit
    }

    /// Java `getSecondPassParams` (deprecated).
    pub(crate) fn get_second_pass_params(&self) -> String {
        self.second_pass_params.to_string()
    }

    /// Java public `getMeanResidChangeLimits`.
    pub fn get_mean_resid_change_limits(&self) -> String {
        self.mean_resid_change_limits.to_string()
    }

    /// Java public `getDeletionParams`.
    pub fn get_deletion_params(&self) -> String {
        self.deletion_params.to_string()
    }
}
