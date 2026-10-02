//! `IMOD/Etomo/src/etomo/comscript/TiltxcorrParam.java`.
//!
//! Parameters of the tiltxcorr command in xcorr.com / xcorr_pt.com, read from and
//! written to the com script both as PIP keyword/value pairs and (read only) as the
//! old sequential standard input.
//!
// TODO(unit): needs etomo/comscript/ParamUtilities.java - `param_utilities` value,
// parse and updateScriptParameter overloads.
// TODO(unit): needs etomo/type/StringParameter.java - boundaryModel, skipViews,
// prealignmentTransformFile, viewsWithMagChanges.
// TODO(unit): needs etomo/type/TiltAngleSpec.java and etomo/type/TiltAngleType.java -
// the sequential tilt angle specification.
// TODO(unit): needs FortranInputString.validateAndSet(ComScriptCommand) and
// FortranInputString.updateScriptParameter(ComScriptCommand) (both marked TODO in
// comscript/fortran_input_string.rs).

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use super::bad_com_script_exception::BadComScriptException;
use super::com_script_command::ComScriptCommand;
use super::com_script_input_arg::ComScriptInputArg;
use super::command::Command;
use super::command_details::CommandDetails;
use super::command_mode::CommandMode;
use super::command_param::{CommandParam, ParseComScriptError};
use super::const_command_param::ConstCommandParam;
use super::const_tiltxcorr_param::ConstTiltxcorrParam;
use super::fortran_input_string::FortranInputString;
use super::fortran_input_syntax_exception::FortranInputSyntaxException;
use super::param_utilities;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::{
    Number, Type, java_lang_double_value_of, java_lang_integer_parse_int,
};
use crate::imod::etomo::r#type::etomo_boolean2::EtomoBoolean2;
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::r#type::file_key::FileKey;
use crate::imod::etomo::r#type::file_type::{self, FileType};
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::script_parameter::ScriptParameter;
use crate::imod::etomo::r#type::string_parameter::StringParameter;
use crate::imod::etomo::r#type::tilt_angle_spec::TiltAngleSpec;
use crate::imod::etomo::r#type::tilt_angle_type::TiltAngleType;
use crate::imod::etomo::ui::swing::ui_harness;
use crate::imod::etomo::util::mrc_header::MRCHeader;
use crate::imod::etomo::util::utilities::{java_io_file_new, java_lang_math_round};

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `GOTO_LABEL`.
pub const GOTO_LABEL: &str = "doxcorr";
/// Java private `COMMAND`.
const COMMAND: &str = "tiltxcorr";
/// Java `SIZE_OF_PATCHES_X_AND_Y_KEY`.
pub const SIZE_OF_PATCHES_X_AND_Y_KEY: &str = "SizeOfPatchesXandY";
/// Java `OVERLAP_OF_PATCHES_X_AND_Y_KEY`.
pub const OVERLAP_OF_PATCHES_X_AND_Y_KEY: &str = "OverlapOfPatchesXandY";
/// Java `OVERLAP_OF_PATCHES_X_AND_Y_DEFAULT`.
pub const OVERLAP_OF_PATCHES_X_AND_Y_DEFAULT: &str = ".33,.33";
/// Java `NUMBER_OF_PATCHES_X_AND_Y_KEY`.
pub const NUMBER_OF_PATCHES_X_AND_Y_KEY: &str = "NumberOfPatchesXandY";
/// Java `ITERATE_CORRELATIONS_KEY`.
pub const ITERATE_CORRELATIONS_KEY: &str = "IterateCorrelations";
/// Java `ITERATE_CORRELATIONS_DEFAULT`.
pub const ITERATE_CORRELATIONS_DEFAULT: i32 = 1;
/// Java `ITERATE_CORRELATIONS_MIN`.
pub const ITERATE_CORRELATIONS_MIN: i32 = 1;
/// Java `ITERATE_CORRELATIONS_MAX`.
pub const ITERATE_CORRELATIONS_MAX: i32 = 4;
/// Java `SHIFT_LIMITS_X_AND_Y_KEY`.
pub const SHIFT_LIMITS_X_AND_Y_KEY: &str = "ShiftLimitsXandY";
/// Java `LENGTH_AND_OVERLAP_KEY`.
pub const LENGTH_AND_OVERLAP_KEY: &str = "LengthAndOverlap";
/// Java `BOUNDARY_MODEL_KEY`.
pub const BOUNDARY_MODEL_KEY: &str = "BoundaryModel";
/// Java `FILTER_SIGMA_1_DEFAULT`.
pub const FILTER_SIGMA_1_DEFAULT: &str = "0.03";
/// Java `FILTER_RADIUS_2_DEFAULT`.
pub const FILTER_RADIUS_2_DEFAULT: &str = "0.25";
/// Java `FILTER_SIGMA_2_DEFAULT`.
pub const FILTER_SIGMA_2_DEFAULT: &str = "0.05";
/// Java `SKIP_VIEWS_KEY`.
pub const SKIP_VIEWS_KEY: &str = "SkipViews";
/// Java `FILTER_RADIUS_2_KEY`.
pub const FILTER_RADIUS_2_KEY: &str = "FilterRadius2";
/// Java `FILTER_SIGMA_2_KEY`.
pub const FILTER_SIGMA_2_KEY: &str = "FilterSigma2";
/// Java `SEARCH_MAG_CHANGES_KEY`.
pub const SEARCH_MAG_CHANGES_KEY: &str = "SearchMagChanges";
/// Java `VIEWS_WITH_MAG_CHANGES_KEY`.
pub const VIEWS_WITH_MAG_CHANGES_KEY: &str = "ViewsWithMagChanges";
/// Java `FILTER_SIGMA1_KEY`.
pub const FILTER_SIGMA1_KEY: &str = "FilterSigma1";

/// Java final `TiltxcorrParam implements ConstTiltxcorrParam, CommandParam,
/// ConstCommandParam`.
pub struct TiltxcorrParam {
    // PIP and sequential input
    input_file: Option<String>,
    piece_list_file: Option<String>,
    output_file: Option<String>,
    exclude_central_peak: bool,
    /// was imageRotation
    rotation_angle: f64,
    /// was trim
    borders_in_x_and_y: FortranInputString,
    x_min_and_max: FortranInputString,
    y_min_and_max: FortranInputString,
    /// was padPercent
    pads_in_x_and_y: FortranInputString,
    /// was taperPercent
    tapers_in_x_and_y: FortranInputString,

    cumulative_correlation: bool,
    absolute_cosine_stretch: bool,
    no_cosine_stretch: bool,
    test_output: Option<String>,
    /// was viewRange
    starting_ending_views: FortranInputString,

    axis_id: AxisID,
    manager: &'static dyn BaseManager,

    // PIP only
    // was tiltAngleSpec
    first_tilt_angle: f64,
    tilt_increment: f64,
    tilt_file: Option<String>,
    tilt_angles: Option<Vec<f64>>,

    // was filterParams
    filter_radius1: f64,
    filter_radius2: ScriptParameter,
    filter_sigma1: ScriptParameter,
    filter_sigma2: ScriptParameter,
    angle_offset: ScriptParameter,

    // sequential input only
    tilt_angle_spec: TiltAngleSpec,
    filter_params: FortranInputString,

    // Patch tracking
    size_of_patches_x_and_y: FortranInputString,
    overlap_of_patches_x_and_y: FortranInputString,
    number_of_patches_x_and_y: FortranInputString,
    iterate_correlations: ScriptParameter,
    shift_limits_x_and_y: FortranInputString,
    /// Deprecated: read lengthAndOverlap, but do not write it out.
    length_and_overlap: FortranInputString,
    boundary_model: StringParameter,
    process_name: ProcessName,
    skip_views: StringParameter,
    prealignment_transform_file: StringParameter,
    images_are_binned: ScriptParameter,
    search_mag_changes: EtomoBoolean2,
    views_with_mag_changes: StringParameter,

    partial_save: bool,
    validate: bool,
}

impl TiltxcorrParam {
    /// Java `TiltxcorrParam(BaseManager, AxisID, ProcessName)`.
    pub fn new(
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        process_name: ProcessName,
    ) -> TiltxcorrParam {
        let tilt_angle_spec = TiltAngleSpec::new();
        let filter_params = FortranInputString::new(4);
        let mut borders_in_x_and_y = FortranInputString::new(2);
        borders_in_x_and_y.set_integer_type_index(0, true);
        borders_in_x_and_y.set_integer_type_index(1, true);
        let mut x_min_and_max = FortranInputString::new(2);
        x_min_and_max.set_integer_type_index(0, true);
        x_min_and_max.set_integer_type_index(1, true);
        let mut y_min_and_max = FortranInputString::new(2);
        y_min_and_max.set_integer_type_index(0, true);
        y_min_and_max.set_integer_type_index(1, true);
        let mut pads_in_x_and_y = FortranInputString::new(2);
        pads_in_x_and_y.set_integer_type_index(0, true);
        pads_in_x_and_y.set_integer_type_index(1, true);
        let mut tapers_in_x_and_y = FortranInputString::new(2);
        tapers_in_x_and_y.set_integer_type_index(0, true);
        tapers_in_x_and_y.set_integer_type_index(1, true);
        let mut starting_ending_views = FortranInputString::new(2);
        starting_ending_views.set_integer_type_index(0, true);
        starting_ending_views.set_integer_type_index(1, true);
        let mut size_of_patches_x_and_y =
            FortranInputString::new_with_key(Some(SIZE_OF_PATCHES_X_AND_Y_KEY), 2);
        size_of_patches_x_and_y.set_integer_type_index(0, true);
        size_of_patches_x_and_y.set_integer_type_index(1, true);
        let overlap_of_patches_x_and_y =
            FortranInputString::new_with_key(Some(OVERLAP_OF_PATCHES_X_AND_Y_KEY), 2);
        let mut number_of_patches_x_and_y =
            FortranInputString::new_with_key(Some(NUMBER_OF_PATCHES_X_AND_Y_KEY), 2);
        number_of_patches_x_and_y.set_integer_type_index(0, true);
        number_of_patches_x_and_y.set_integer_type_index(1, true);
        let mut iterate_correlations = ScriptParameter::new_with_name(ITERATE_CORRELATIONS_KEY);
        iterate_correlations.set_floor(ITERATE_CORRELATIONS_MIN);
        iterate_correlations.set_ceiling(ITERATE_CORRELATIONS_MAX);
        let mut shift_limits_x_and_y =
            FortranInputString::new_with_key(Some(SHIFT_LIMITS_X_AND_Y_KEY), 2);
        shift_limits_x_and_y.set_integer_type_index(0, true);
        shift_limits_x_and_y.set_integer_type_index(1, true);
        let mut length_and_overlap =
            FortranInputString::new_with_key(Some(LENGTH_AND_OVERLAP_KEY), 2);
        length_and_overlap.set_integer_type_index(0, true);
        length_and_overlap.set_integer_type_index(1, true);
        let mut param = TiltxcorrParam {
            input_file: None,
            piece_list_file: None,
            output_file: None,
            exclude_central_peak: false,
            rotation_angle: 0.0,
            borders_in_x_and_y,
            x_min_and_max,
            y_min_and_max,
            pads_in_x_and_y,
            tapers_in_x_and_y,
            cumulative_correlation: false,
            absolute_cosine_stretch: false,
            no_cosine_stretch: false,
            test_output: None,
            starting_ending_views,
            axis_id,
            manager,
            first_tilt_angle: 0.0,
            tilt_increment: 0.0,
            tilt_file: None,
            tilt_angles: None,
            filter_radius1: 0.0,
            filter_radius2: ScriptParameter::new_with_type_and_name(
                Type::Double,
                FILTER_RADIUS_2_KEY,
            ),
            filter_sigma1: ScriptParameter::new_with_type_and_name(Type::Double, FILTER_SIGMA1_KEY),
            filter_sigma2: ScriptParameter::new_with_type_and_name(
                Type::Double,
                FILTER_SIGMA_2_KEY,
            ),
            angle_offset: ScriptParameter::new_with_type_and_name(Type::Double, "AngleOffset"),
            tilt_angle_spec,
            filter_params,
            size_of_patches_x_and_y,
            overlap_of_patches_x_and_y,
            number_of_patches_x_and_y,
            iterate_correlations,
            shift_limits_x_and_y,
            length_and_overlap,
            boundary_model: StringParameter::new(BOUNDARY_MODEL_KEY),
            process_name,
            skip_views: StringParameter::new(SKIP_VIEWS_KEY),
            prealignment_transform_file: StringParameter::new("PrealignmentTransformFile"),
            images_are_binned: ScriptParameter::new_with_name("ImagesAreBinned"),
            search_mag_changes: EtomoBoolean2::new_with_name(SEARCH_MAG_CHANGES_KEY),
            views_with_mag_changes: StringParameter::new("ViewsWithMagChanges"),
            partial_save: false,
            validate: false,
        };
        param.reset();
        param
    }

    /// Java `setValidate`.  Set validate to true to cause validations to happen.
    pub fn set_validate(&mut self, validate: bool) {
        self.validate = validate;
    }

    /// Java `setViewsWithMagChanges`.
    pub fn set_views_with_mag_changes(&mut self, input: Option<&str>) {
        self.views_with_mag_changes.set(input);
    }

    /// Java static `getBordersInXandYDefault(BaseManager, AxisID, FileType)`.
    pub fn get_borders_in_x_and_y_default(
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        file_type: &Arc<FileType>,
    ) -> String {
        let mut borders_in_x = EtomoNumber::new();
        let mut borders_in_y = EtomoNumber::new();
        let header = MRCHeader::get_instance_in_dir(
            manager.get_property_user_dir().as_deref(),
            file_type
                .get_file_name(Some(manager), Some(axis_id))
                .as_deref(),
            Some(axis_id),
        );
        let header = match header {
            None => return String::new(),
            Some(header) => header,
        };
        // `catch (IOException | InvalidParameterException e)`: print and return "".
        if let Err(e) = header.borrow_mut().read_with_manager(manager) {
            eprintln!("{}", e);
            return String::new();
        }
        let x = header.borrow().get_n_columns();
        if x == -1 {
            return String::new();
        }
        borders_in_x.set_long(java_lang_math_round(x as f64 * 0.05));
        let y = header.borrow().get_n_rows();
        if y == -1 {
            return String::new();
        }
        borders_in_y.set_long(java_lang_math_round(y as f64 * 0.05));
        format!("{},{}", borders_in_x, borders_in_y)
    }

    /// Java private `reset`.
    fn reset(&mut self) {
        self.validate = false;
        self.input_file = Some(String::new());
        self.piece_list_file = Some(String::new());
        self.output_file = Some(String::new());
        self.exclude_central_peak = false;
        self.rotation_angle = f64::NAN;
        self.borders_in_x_and_y.set_default();
        self.x_min_and_max.set_default();
        self.y_min_and_max.set_default();
        self.pads_in_x_and_y.set_default();
        self.tapers_in_x_and_y.set_default();
        self.cumulative_correlation = false;
        self.absolute_cosine_stretch = false;
        self.no_cosine_stretch = false;
        self.test_output = Some(String::new());
        self.starting_ending_views.set_default();
        self.first_tilt_angle = f64::NAN;
        self.tilt_increment = f64::NAN;
        self.tilt_file = Some(String::new());
        self.tilt_angles = None;
        self.filter_radius1 = f64::NAN;
        self.filter_radius2.reset();
        self.filter_sigma1.reset();
        self.filter_sigma2.reset();
        // TiltxcorrParam.java:337 constructs a local `TiltAngleSpec` that shadows the
        // field and is discarded, so the field is not reset.  Kept as written.
        let _tilt_angle_spec = TiltAngleSpec::new();
        self.filter_params.set_default();
        self.angle_offset.reset();
        self.size_of_patches_x_and_y.set_default();
        self.overlap_of_patches_x_and_y.set_default();
        self.number_of_patches_x_and_y.set_default();
        self.iterate_correlations.reset();
        self.shift_limits_x_and_y.set_default();
        self.length_and_overlap.set_default();
        self.skip_views.reset();
        self.prealignment_transform_file.reset();
        self.images_are_binned.reset();
        self.search_mag_changes.reset();
    }

    /// Java `resetBoundaryModel`.
    pub fn reset_boundary_model(&mut self) {
        self.boundary_model.reset();
    }

    /// Java `resetNumberOfPatchesXandY`.
    pub fn reset_number_of_patches_x_and_y(&mut self) {
        self.number_of_patches_x_and_y.set_default();
    }

    /// Java `resetOverlapOfPatchesXandY`.
    pub fn reset_overlap_of_patches_x_and_y(&mut self) {
        self.overlap_of_patches_x_and_y.set_default();
    }

    /// Java `isViewsWithMagChangesNull`.
    pub fn is_views_with_mag_changes_null(&self) -> bool {
        self.views_with_mag_changes.is_empty()
    }

    /// Java `getProcessName`.
    pub fn get_process_name(&self) -> ProcessName {
        self.process_name
    }

    /// Java `setPartialSave`.
    pub fn set_partial_save(&mut self, input: bool) {
        self.partial_save = input;
    }

    /// Java `setTiltAngleSpec`.
    pub fn set_tilt_angle_spec(&mut self, input: &TiltAngleSpec) {
        self.tilt_angle_spec.set(Some(input));
        if self.tilt_angle_spec.get_type() == TiltAngleType::Extract {
            self.tilt_file = file_type::CLASS
                .raw_tilt_angles
                .get_file_name(Some(self.manager), Some(self.axis_id));
        }
    }

    /// Java `getTiltAngleSpec`.
    pub fn get_tilt_angle_spec(&self) -> &TiltAngleSpec {
        &self.tilt_angle_spec
    }

    /// Java private `sequentialInputToPip`.
    fn sequential_input_to_pip(&mut self) {
        // LIST is not implemented and EXTRACT isn't used with tiltxcorr
        if self.tilt_angle_spec.get_type() == TiltAngleType::File {
            self.tilt_file = Some(self.tilt_angle_spec.get_tilt_angle_filename());
        } else if self.tilt_angle_spec.get_type() == TiltAngleType::Range {
            self.first_tilt_angle = self.tilt_angle_spec.get_range_min();
            self.tilt_increment = self.tilt_angle_spec.get_range_step();
        }
        self.filter_sigma1
            .set_double(self.filter_params.get_double_index(0));
        self.filter_sigma2
            .set_double(self.filter_params.get_double_index(1));
        self.filter_radius1 = self.filter_params.get_double_index(2);
        self.filter_radius2
            .set_double(self.filter_params.get_double_index(3));
    }

    /// Java `setInputFile`.  Set the input file name.
    pub fn set_input_file(&mut self, input_file: Option<&str>) {
        self.input_file = input_file.map(|s| s.to_string());
    }

    /// Java `setIterateCorrelations(Number)`.  Returns error message if invalid.
    pub fn set_iterate_correlations(&mut self, input: Option<Number>) -> Option<String> {
        self.iterate_correlations.set_number(input);
        if !self.iterate_correlations.is_valid() {
            return Some(self.iterate_correlations.get_invalid_reason());
        }
        None
    }

    /// Java `setPieceListFile`.
    pub fn set_piece_list_file(&mut self, piece_list_file: Option<&str>) {
        self.piece_list_file = piece_list_file.map(|s| s.to_string());
    }

    /// Java `setSizeOfPatchesXandY(String, String)`.
    pub fn set_size_of_patches_x_and_y(
        &mut self,
        input: Option<&str>,
        description: &str,
    ) -> Result<bool, FortranInputSyntaxException> {
        self.size_of_patches_x_and_y.validate_and_set(input)?;
        if self.validate {
            if !self.size_of_patches_x_and_y.is_null_index(0)
                && self.size_of_patches_x_and_y.is_null_index(1)
            {
                ui_harness::post_message_dialog(
                    Some(self.manager),
                    format!("Two values are required for {}.", description),
                    "Entry Error".to_string(),
                    Some(self.axis_id),
                );
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Java `setOutputFile`.
    pub fn set_output_file(&mut self, output_file: Option<&str>) {
        self.output_file = output_file.map(|s| s.to_string());
    }

    /// Java `setOverlapOfPatchesXandY`.
    pub fn set_overlap_of_patches_x_and_y(
        &mut self,
        input: Option<&str>,
    ) -> Result<(), FortranInputSyntaxException> {
        self.overlap_of_patches_x_and_y.validate_and_set(input)
    }

    /// Java `setFilterRadius1`.  `ParamUtilities.parseDouble` throws an unchecked
    /// NumberFormatException for a non-numeric entry; it comes back as `Err` (the
    /// field is left unchanged, as the throw leaves it).
    pub fn set_filter_radius1(&mut self, filter_radius1: Option<&str>) -> Result<(), String> {
        self.filter_radius1 = param_utilities::parse_double(filter_radius1)?;
        Ok(())
    }

    /// Java `setFilterRadius2`.
    pub fn set_filter_radius2(&mut self, filter_radius2: Option<&str>) {
        self.filter_radius2.set_string(filter_radius2);
    }

    /// Java `setFilterSigma1`.
    pub fn set_filter_sigma1(&mut self, filter_sigma1: Option<&str>) {
        self.filter_sigma1.set_string(filter_sigma1);
    }

    /// Java `setFilterSigma2`.
    pub fn set_filter_sigma2(&mut self, filter_sigma2: Option<&str>) {
        self.filter_sigma2.set_string(filter_sigma2);
    }

    /// Java `setBordersInXandY`.  Set the borders in x and y.
    pub fn set_borders_in_x_and_y(
        &mut self,
        borders_in_x_and_y: Option<&str>,
    ) -> Result<(), FortranInputSyntaxException> {
        param_utilities::set_fortran_input_string(borders_in_x_and_y, &mut self.borders_in_x_and_y)
    }

    /// Java `setBoundaryModel`.
    pub fn set_boundary_model(&mut self, input: Option<&str>) {
        self.boundary_model.set(input);
    }

    /// Java `setXMin`.  `Err` carries an unchecked NumberFormatException.
    pub fn set_x_min(&mut self, x_min: Option<&str>) -> Result<(), String> {
        param_utilities::set_fortran_input_string_index(x_min, &mut self.x_min_and_max, 0)
    }

    /// Java `setXMax`.
    pub fn set_x_max(&mut self, x_max: Option<&str>) -> Result<(), String> {
        param_utilities::set_fortran_input_string_index(x_max, &mut self.x_min_and_max, 1)
    }

    /// Java `setYMin`.
    pub fn set_y_min(&mut self, y_min: Option<&str>) -> Result<(), String> {
        param_utilities::set_fortran_input_string_index(y_min, &mut self.y_min_and_max, 0)
    }

    /// Java `setYMax`.
    pub fn set_y_max(&mut self, y_max: Option<&str>) -> Result<(), String> {
        param_utilities::set_fortran_input_string_index(y_max, &mut self.y_min_and_max, 1)
    }

    /// Java `setPadsInXandY`.  Set the pads in x and y.
    pub fn set_pads_in_x_and_y(
        &mut self,
        pads_in_x_and_y: Option<&str>,
    ) -> Result<(), FortranInputSyntaxException> {
        param_utilities::set_fortran_input_string(pads_in_x_and_y, &mut self.pads_in_x_and_y)
    }

    /// Java `setShiftLimitsXandY`.
    pub fn set_shift_limits_x_and_y(
        &mut self,
        input: Option<&str>,
    ) -> Result<(), FortranInputSyntaxException> {
        self.shift_limits_x_and_y.validate_and_set(input)
    }

    /// Java `setTapersInXandY`.  Set the taper percentage.
    pub fn set_tapers_in_x_and_y(
        &mut self,
        tapers_in_x_and_y: Option<&str>,
    ) -> Result<(), FortranInputSyntaxException> {
        param_utilities::set_fortran_input_string(tapers_in_x_and_y, &mut self.tapers_in_x_and_y)
    }

    /// Java `setCumulativeCorrelation`.
    pub fn set_cumulative_correlation(&mut self, cumulative_correlation: bool) {
        self.cumulative_correlation = cumulative_correlation;
    }

    /// Java `setAbsoluteCosineStretch`.
    pub fn set_absolute_cosine_stretch(&mut self, absolute_cosine_stretch: bool) {
        self.absolute_cosine_stretch = absolute_cosine_stretch;
    }

    /// Java `setNoCosineStretch`.
    pub fn set_no_cosine_stretch(&mut self, no_cosine_stretch: bool) {
        self.no_cosine_stretch = no_cosine_stretch;
    }

    /// Java `setNumberOfPatchesXandY`.
    pub fn set_number_of_patches_x_and_y(
        &mut self,
        input: Option<&str>,
    ) -> Result<(), FortranInputSyntaxException> {
        self.number_of_patches_x_and_y.validate_and_set(input)
    }

    /// Java `setStartingEndingViews`.  Set the range of view to process.
    pub fn set_starting_ending_views(
        &mut self,
        starting_ending_views: Option<&str>,
    ) -> Result<(), FortranInputSyntaxException> {
        param_utilities::set_fortran_input_string(
            starting_ending_views,
            &mut self.starting_ending_views,
        )
    }

    /// Java `setExcludeCentralPeak`.  Set/unset the exclude central peak flag.
    pub fn set_exclude_central_peak(&mut self, exclude_central_peak: bool) {
        self.exclude_central_peak = exclude_central_peak;
    }

    /// Java `setTestOutput`.
    pub fn set_test_output(&mut self, test_output: &str) {
        self.test_output = Some(test_output.to_string());
    }

    /// Java `setAngleOffset`.
    pub fn set_angle_offset(&mut self, input: Option<&str>) {
        self.angle_offset.set_string(input);
    }

    /// Java private `getInputArguments`.  Get the standand input arguments from the
    /// ComScriptCommand validating the name of the command.
    fn get_input_arguments(
        &self,
        script_command: &ComScriptCommand,
    ) -> Result<Vec<Rc<RefCell<ComScriptInputArg>>>, BadComScriptException> {
        // Check to be sure that it is a tiltxcorr xommand.  (A null command is a
        // NullPointerException at TiltxcorrParam.java:760; it is "not a tiltxcorr
        // command" here.)
        if script_command.get_command() != Some("tiltxcorr") {
            return Err(BadComScriptException::new("Not a tiltxcorr command"));
        }
        // Get the input arguments parameters to preserve the comments
        let input_args = script_command.get_input_arguments();
        Ok(input_args)
    }

    /// Java `getInputFile`.
    pub fn get_input_file(&self) -> Option<String> {
        self.input_file.clone()
    }

    /// Java `getLengthFromLengthAndOverlap` (deprecated).
    pub fn get_length_from_length_and_overlap(&self) -> String {
        self.length_and_overlap
            .to_string_index_default_is_blank(0, true)
    }

    /// Java `getOverlapFromLengthAndOverlap` (deprecated).
    pub fn get_overlap_from_length_and_overlap(&self) -> String {
        self.length_and_overlap
            .to_string_index_default_is_blank(1, true)
    }

    /// Java `getPieceListFile`.
    pub fn get_piece_list_file(&self) -> Option<String> {
        self.piece_list_file.clone()
    }

    /// Java `getOutputFile`.
    pub fn get_output_file(&self) -> Option<String> {
        self.output_file.clone()
    }

    /// Java `getFirstTiltAngleString`.
    pub fn get_first_tilt_angle_string(&self) -> String {
        param_utilities::value_of_double(self.first_tilt_angle)
    }

    /// Java `getTiltIncrementString`.
    pub fn get_tilt_increment_string(&self) -> String {
        param_utilities::value_of_double(self.tilt_increment)
    }

    /// Java `getTiltFile`.
    pub fn get_tilt_file(&self) -> Option<String> {
        self.tilt_file.clone()
    }

    /// Java `getTiltAnglesString`.
    ///
    /// TiltxcorrParam.java:918 passes `tiltAngles` - null after every `reset` - to
    /// `ParamUtilities.valueOf(double[])`, which reads `values.length` and throws
    /// NullPointerException.  A null array is read here as an empty one, which that
    /// routine answers with a single empty string.
    pub fn get_tilt_angles_string(&self) -> Vec<String> {
        param_utilities::value_of_double_array(self.tilt_angles.as_deref().unwrap_or(&[]))
    }

    /// Java `getRotationAngleString`.
    pub fn get_rotation_angle_string(&self) -> String {
        param_utilities::value_of_double(self.rotation_angle)
    }

    /// Java `setRotationAngle`.
    pub fn set_rotation_angle(&mut self, input: f64) {
        self.rotation_angle = input;
    }

    /// Java `setSearchMagChanges`.
    pub fn set_search_mag_changes(&mut self, input: bool) {
        self.search_mag_changes.set_boolean(input);
    }

    /// Java `getFilterRadius1String`.
    pub fn get_filter_radius1_string(&self) -> String {
        param_utilities::value_of_double(self.filter_radius1)
    }

    /// Java `setSkipViews`.
    pub fn set_skip_views(&mut self, input: Option<&str>) {
        self.skip_views.set(input);
    }

    /// Java `setPrealignmentTransformFileDefault`.
    pub fn set_prealignment_transform_file_default(&mut self) {
        let file_name = file_type::CLASS
            .pre_xg
            .get_file_name(Some(self.manager), Some(self.axis_id));
        self.prealignment_transform_file.set(file_name.as_deref());
    }

    /// Java `setImagesAreBinned(int)`.
    pub fn set_images_are_binned(&mut self, input: i32) {
        self.images_are_binned.set_int(input);
    }
}

/// Java `toString`.  Return a string describing the class attributes.
impl std::fmt::Display for TiltxcorrParam {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[tiltFile:{},iterateCorrelations:{}]",
            self.tilt_file.as_deref().unwrap_or("null"),
            self.iterate_correlations
        )
    }
}

impl CommandParam for TiltxcorrParam {
    /// Java `parseComScriptCommand`.  Get the parameters from the ComScriptCommand.
    ///
    /// Runtime exceptions the source lets escape (NumberFormatException, and the
    /// NullPointerException / ArrayIndexOutOfBoundsException of a short or valueless
    /// sequential input) reach `ComScriptUtil.initialize`'s `catch (Exception)`; they
    /// are `ParseComScriptError::NumberFormat` here.
    fn parse_com_script_command(
        &mut self,
        script_command: &ComScriptCommand,
    ) -> Result<(), ParseComScriptError> {
        // get the input arguments from the command
        let input_args = self.get_input_arguments(script_command)?;
        let _cmd_line_args = script_command.get_command_line_args();
        if script_command.is_keyword_value_pairs() {
            self.input_file = script_command.get_value(Some("InputFile"))?;
            self.piece_list_file = script_command.get_value(Some("PieceListFile"))?;
            self.output_file = script_command.get_value(Some("OutputFile"))?;
            if script_command.has_keyword(Some("FirstTiltAngle"))? {
                self.first_tilt_angle = java_lang_double_value_of(
                    script_command
                        .get_value(Some("FirstTiltAngle"))?
                        .as_deref()
                        .unwrap_or("null"),
                )
                .map_err(ParseComScriptError::NumberFormat)?;
            }
            if script_command.has_keyword(Some("TiltIncrement"))? {
                self.tilt_increment = java_lang_double_value_of(
                    script_command
                        .get_value(Some("TiltIncrement"))?
                        .as_deref()
                        .unwrap_or("null"),
                )
                .map_err(ParseComScriptError::NumberFormat)?;
            }
            self.tilt_file = script_command.get_value(Some("TiltFile"))?;
            // TiltxcorrParam.java:450-455 stores each comma-separated token into
            // `tiltAngles[index++]`, but `tiltAngles` is null after `reset`, so any
            // TiltAngles value throws NullPointerException (and a value with more
            // entries than an earlier array throws ArrayIndexOutOfBoundsException).
            // Here the tokens become the array; with no tokens the array is left as
            // it was, as in the source.  A keyword with no value (null) has no
            // tokens here, where `new StringTokenizer(null)` would throw.
            let tilt_angles_value = script_command.get_value(Some("TiltAngles"))?;
            let mut tilt_angles: Vec<f64> = Vec::new();
            if let Some(tilt_angles_value) = &tilt_angles_value {
                for token in tilt_angles_value.split(',') {
                    if token.is_empty() {
                        continue;
                    }
                    tilt_angles.push(
                        java_lang_double_value_of(token)
                            .map_err(ParseComScriptError::NumberFormat)?,
                    );
                }
            }
            if !tilt_angles.is_empty() {
                self.tilt_angles = Some(tilt_angles);
            }
            if script_command.has_keyword(Some("RotationAngle"))? {
                self.rotation_angle = java_lang_double_value_of(
                    script_command
                        .get_value(Some("RotationAngle"))?
                        .as_deref()
                        .unwrap_or("null"),
                )
                .map_err(ParseComScriptError::NumberFormat)?;
            }
            if script_command.has_keyword(Some("FilterRadius1"))? {
                self.filter_radius1 = java_lang_double_value_of(
                    script_command
                        .get_value(Some("FilterRadius1"))?
                        .as_deref()
                        .unwrap_or("null"),
                )
                .map_err(ParseComScriptError::NumberFormat)?;
            }
            self.filter_radius2.parse(script_command)?;
            self.filter_sigma1.parse(script_command)?;
            self.filter_sigma2.parse(script_command)?;
            self.exclude_central_peak = script_command.has_keyword(Some("ExcludeCentralPeak"))?;

            if script_command.has_keyword(Some("BordersInXandY"))? {
                self.borders_in_x_and_y.validate_and_set(
                    script_command.get_value(Some("BordersInXandY"))?.as_deref(),
                )?;
            }
            if script_command.has_keyword(Some("XMinAndMax"))? {
                self.x_min_and_max
                    .validate_and_set(script_command.get_value(Some("XMinAndMax"))?.as_deref())?;
            }
            if script_command.has_keyword(Some("YMinAndMax"))? {
                self.y_min_and_max
                    .validate_and_set(script_command.get_value(Some("YMinAndMax"))?.as_deref())?;
            }
            if script_command.has_keyword(Some("PadsInXandY"))? {
                self.pads_in_x_and_y
                    .validate_and_set(script_command.get_value(Some("PadsInXandY"))?.as_deref())?;
            }
            if script_command.has_keyword(Some("TapersInXandY"))? {
                self.tapers_in_x_and_y.validate_and_set(
                    script_command.get_value(Some("TapersInXandY"))?.as_deref(),
                )?;
            }
            self.cumulative_correlation =
                script_command.has_keyword(Some("CumulativeCorrelation"))?;
            self.absolute_cosine_stretch =
                script_command.has_keyword(Some("AbsoluteCosineStretch"))?;
            self.no_cosine_stretch = script_command.has_keyword(Some("NoCosineStretch"))?;
            if script_command.has_keyword(Some("TestOutput"))? {
                self.test_output = script_command.get_value(Some("TestOutput"))?;
            }
            if script_command.has_keyword(Some("StartingEndingViews"))? {
                self.starting_ending_views.validate_and_set(
                    script_command
                        .get_value(Some("StartingEndingViews"))?
                        .as_deref(),
                )?;
            }
            self.angle_offset.parse(script_command)?;
            self.size_of_patches_x_and_y
                .validate_and_set_com_script(script_command)?;
            self.overlap_of_patches_x_and_y
                .validate_and_set_com_script(script_command)?;
            self.number_of_patches_x_and_y
                .validate_and_set_com_script(script_command)?;
            self.iterate_correlations.parse(script_command)?;
            self.shift_limits_x_and_y
                .validate_and_set_com_script(script_command)?;
            self.length_and_overlap
                .validate_and_set_com_script(script_command)?;
            self.boundary_model.parse(script_command)?;
            self.skip_views.parse(script_command)?;
            self.prealignment_transform_file.parse(script_command)?;
            self.images_are_binned.parse(script_command)?;
            self.search_mag_changes.parse(script_command)?;
            self.views_with_mag_changes.parse(script_command)?;
            return Ok(());
        }

        // `inputArgs[inputLine++].getArgument()`: an index past the end is an
        // ArrayIndexOutOfBoundsException in the source.
        let mut input_line: usize = 0;
        let mut next_argument =
            |input_line: &mut usize| -> Result<Option<String>, ParseComScriptError> {
                if *input_line >= input_args.len() {
                    return Err(ParseComScriptError::NumberFormat(format!(
                        "java.lang.ArrayIndexOutOfBoundsException: {}",
                        *input_line
                    )));
                }
                let argument = input_args[*input_line]
                    .borrow()
                    .get_argument()
                    .map(|argument| argument.to_string());
                *input_line += 1;
                Ok(argument)
            };
        self.input_file = next_argument(&mut input_line)?;
        self.piece_list_file = next_argument(&mut input_line)?;
        self.output_file = next_argument(&mut input_line)?;

        let type_spec = java_lang_integer_parse_int(
            next_argument(&mut input_line)?.as_deref().unwrap_or("null"),
        )
        .map_err(ParseComScriptError::NumberFormat)?;
        // Java `tiltAngleSpec.setType(TiltAngleType.parseInt(typeSpec))` stores null for
        // an unknown type, which then falls to the error branch below.  The Rust spec
        // holds no null type, so an unknown type is not stored; the branch is taken from
        // the parsed value.
        let tilt_angle_type = TiltAngleType::parse_int(type_spec);
        if let Some(tilt_angle_type) = tilt_angle_type {
            self.tilt_angle_spec.set_type(tilt_angle_type);
        }
        if tilt_angle_type == Some(TiltAngleType::File) {
            let filename = next_argument(&mut input_line)?;
            self.tilt_angle_spec
                .set_tilt_angle_filename(filename.as_deref().unwrap_or(""));
            self.tilt_file = Some(self.tilt_angle_spec.get_tilt_angle_filename());
        } else if tilt_angle_type == Some(TiltAngleType::Range) {
            // `pair.split(",")` on a null argument is a NullPointerException in the
            // source.
            let pair = next_argument(&mut input_line)?.ok_or_else(|| {
                ParseComScriptError::NumberFormat("java.lang.NullPointerException".to_string())
            })?;
            // `String.split(",")` drops trailing empty strings.
            let mut values: Vec<&str> = pair.split(',').collect();
            while values.last() == Some(&"") {
                values.pop();
            }
            if values.len() != 2 {
                return Err(ParseComScriptError::BadComScript(
                    BadComScriptException::new("Incorrect tilt angle specification type"),
                ));
            }
            self.tilt_angle_spec.set_range_min_double(
                java_lang_double_value_of(values[0]).map_err(ParseComScriptError::NumberFormat)?,
            );
            self.tilt_angle_spec.set_range_step_double(
                java_lang_double_value_of(values[1]).map_err(ParseComScriptError::NumberFormat)?,
            );
            self.first_tilt_angle = self.tilt_angle_spec.get_range_min();
            self.tilt_increment = self.tilt_angle_spec.get_range_step();
        } else if self.tilt_angle_spec.get_type() == TiltAngleType::List {
            return Err(ParseComScriptError::BadComScript(
                BadComScriptException::new("Unimplemented tilt angle specification type"),
            ));
        } else {
            return Err(ParseComScriptError::BadComScript(
                BadComScriptException::new("Incorrect tilt angle specification type"),
            ));
        }

        self.rotation_angle =
            java_lang_double_value_of(next_argument(&mut input_line)?.as_deref().unwrap_or("null"))
                .map_err(ParseComScriptError::NumberFormat)?;
        // The source's `try` block: only a FortranInputSyntaxException is caught and
        // rethrown with the argument number; anything else propagates unchanged.
        let result: Result<(), ParseComScriptError> = (|| {
            self.filter_params
                .validate_and_set(next_argument(&mut input_line)?.as_deref())?;
            self.filter_sigma1
                .set_double(self.filter_params.get_double_index(0));
            self.filter_sigma2
                .set_double(self.filter_params.get_double_index(1));
            self.filter_radius1 = self.filter_params.get_double_index(2);
            self.filter_radius2
                .set_double(self.filter_params.get_double_index(3));
            // `getArgument().matches("\\s*1\\s*")`; a null argument (a
            // NullPointerException in the source) does not match here.
            self.exclude_central_peak = match next_argument(&mut input_line)? {
                None => false,
                Some(argument) => {
                    argument.trim_matches(|c: char| {
                        matches!(c, ' ' | '\t' | '\n' | '\u{0B}' | '\u{0C}' | '\r')
                    }) == "1"
                }
            };
            self.borders_in_x_and_y
                .validate_and_set(next_argument(&mut input_line)?.as_deref())?;
            self.pads_in_x_and_y
                .validate_and_set(next_argument(&mut input_line)?.as_deref())?;
            self.tapers_in_x_and_y
                .validate_and_set(next_argument(&mut input_line)?.as_deref())?;
            self.starting_ending_views
                .validate_and_set(next_argument(&mut input_line)?.as_deref())?;
            Ok(())
        })();
        match result {
            Ok(()) => {}
            Err(ParseComScriptError::FortranInputSyntax(except)) => {
                let message = format!(
                    "Parse error in tiltxcorr command, standard input argument: {}\n{}",
                    input_line,
                    except.get_message().unwrap_or("null")
                );
                return Err(ParseComScriptError::FortranInputSyntax(
                    FortranInputSyntaxException::new_with_new_values(
                        &message,
                        except.get_new_string(),
                    ),
                ));
            }
            Err(other) => return Err(other),
        }
        self.sequential_input_to_pip();
        Ok(())
    }

    /// Java `updateComScriptCommand`.  Update the script command with the current
    /// values of this TiltxcorrParam object.
    fn update_com_script_command(
        &self,
        script_command: &mut ComScriptCommand,
    ) -> Result<(), BadComScriptException> {
        // get the input arguments from the command
        let _input_args = self.get_input_arguments(script_command)?;

        // When partialSave is on, don't require any fields.
        let required = true && !self.partial_save;

        // Switch to keyword/value pairs
        script_command.use_keyword_value();
        param_utilities::update_script_parameter_string_required(
            script_command,
            Some("InputFile"),
            self.input_file.as_deref(),
            required,
        )?;
        param_utilities::update_script_parameter_string(
            script_command,
            Some("PieceListFile"),
            self.piece_list_file.as_deref(),
        )?;
        param_utilities::update_script_parameter_string(
            script_command,
            Some("OutputFile"),
            self.output_file.as_deref(),
        )?;
        param_utilities::update_script_parameter_double(
            script_command,
            Some("FirstTiltAngle"),
            self.first_tilt_angle,
        );
        param_utilities::update_script_parameter_double(
            script_command,
            Some("TiltIncrement"),
            self.tilt_increment,
        );
        param_utilities::update_script_parameter_string(
            script_command,
            Some("TiltFile"),
            self.tilt_file.as_deref(),
        )?;
        param_utilities::update_script_parameter_double_array(
            script_command,
            Some("TiltAngles"),
            self.tilt_angles.as_deref(),
        )?;
        param_utilities::update_script_parameter_double(
            script_command,
            Some("RotationAngle"),
            self.rotation_angle,
        );
        param_utilities::update_script_parameter_double(
            script_command,
            Some("FilterRadius1"),
            self.filter_radius1,
        );
        self.filter_radius2.update_com_script(script_command);
        self.filter_sigma1.update_com_script(script_command);
        self.filter_sigma2.update_com_script(script_command);
        param_utilities::update_script_parameter_boolean(
            script_command,
            Some("ExcludeCentralPeak"),
            self.exclude_central_peak,
        );
        param_utilities::update_script_parameter_fortran_input_string(
            script_command,
            Some("BordersInXandY"),
            &self.borders_in_x_and_y,
        );
        param_utilities::update_script_parameter_fortran_input_string(
            script_command,
            Some("XMinAndMax"),
            &self.x_min_and_max,
        );
        param_utilities::update_script_parameter_fortran_input_string(
            script_command,
            Some("YMinAndMax"),
            &self.y_min_and_max,
        );
        param_utilities::update_script_parameter_fortran_input_string(
            script_command,
            Some("PadsInXandY"),
            &self.pads_in_x_and_y,
        );
        param_utilities::update_script_parameter_fortran_input_string(
            script_command,
            Some("TapersInXandY"),
            &self.tapers_in_x_and_y,
        );
        param_utilities::update_script_parameter_boolean(
            script_command,
            Some("CumulativeCorrelation"),
            self.cumulative_correlation,
        );
        param_utilities::update_script_parameter_boolean(
            script_command,
            Some("AbsoluteCosineStretch"),
            self.absolute_cosine_stretch,
        );
        param_utilities::update_script_parameter_boolean(
            script_command,
            Some("NoCosineStretch"),
            self.no_cosine_stretch,
        );
        param_utilities::update_script_parameter_string(
            script_command,
            Some("TestOutput"),
            self.test_output.as_deref(),
        )?;
        param_utilities::update_script_parameter_fortran_input_string(
            script_command,
            Some("StartingEndingViews"),
            &self.starting_ending_views,
        );
        self.angle_offset.update_com_script(script_command);
        self.size_of_patches_x_and_y
            .update_script_parameter(script_command);
        self.overlap_of_patches_x_and_y
            .update_script_parameter(script_command);
        self.number_of_patches_x_and_y
            .update_script_parameter(script_command);
        self.iterate_correlations.update_com_script(script_command);
        self.shift_limits_x_and_y
            .update_script_parameter(script_command);
        self.boundary_model.update_com_script(script_command);
        self.skip_views.update_com_script(script_command);
        self.prealignment_transform_file
            .update_com_script(script_command);
        self.images_are_binned.update_com_script(script_command);
        self.search_mag_changes.update_com_script(script_command);
        self.views_with_mag_changes
            .update_com_script(script_command);
        Ok(())
    }

    /// Java `initializeDefaults`.
    fn initialize_defaults(&mut self) {}
}

impl ConstCommandParam for TiltxcorrParam {
    /// Java `isParseComments`.
    fn is_parse_comments(&self) -> bool {
        true
    }

    /// Java `getProcessNameString`.
    fn get_process_name_string(&self) -> Option<String> {
        Some(self.process_name.to_string())
    }

    /// Java `getCommand`.
    fn get_command(&self) -> Option<String> {
        Some(self.process_name.get_comscript(self.axis_id))
    }
}

impl ConstTiltxcorrParam for TiltxcorrParam {
    fn is_search_mag_changes(&self) -> bool {
        self.search_mag_changes.is()
    }

    fn is_filter_radius2_set(&self) -> bool {
        !self.filter_radius2.is_null()
    }

    fn is_filter_sigma1_set(&self) -> bool {
        !self.filter_sigma1.is_null()
    }

    fn is_filter_sigma2_set(&self) -> bool {
        !self.filter_sigma2.is_null()
    }

    fn get_iterate_correlations(&self) -> i32 {
        if !self.iterate_correlations.is_null() && self.iterate_correlations.is_valid() {
            return self.iterate_correlations.get_int();
        }
        ITERATE_CORRELATIONS_DEFAULT
    }

    fn get_number_of_patches_x_and_y(&self) -> String {
        self.number_of_patches_x_and_y
            .to_string_default_is_blank(true)
    }

    fn get_overlap_of_patches_x_and_y(&self) -> String {
        self.overlap_of_patches_x_and_y
            .to_string_default_is_blank(true)
    }

    fn get_views_with_mag_changes(&self) -> String {
        self.views_with_mag_changes.to_string()
    }

    fn get_shift_limits_x_and_y(&self) -> String {
        self.shift_limits_x_and_y.to_string_default_is_blank(true)
    }

    fn get_size_of_patches_x_and_y(&self) -> String {
        self.size_of_patches_x_and_y
            .to_string_default_is_blank(true)
    }

    fn get_filter_radius2_string(&self) -> String {
        self.filter_radius2.to_string()
    }

    fn get_filter_sigma1_string(&self) -> String {
        self.filter_sigma1.to_string()
    }

    fn get_filter_sigma2_string(&self) -> String {
        self.filter_sigma2.to_string()
    }

    fn get_borders_in_x_and_y(&self) -> String {
        self.borders_in_x_and_y.to_string_default_is_blank(true)
    }

    fn get_x_min_string(&self) -> String {
        self.x_min_and_max.to_string_index(0)
    }

    fn get_x_max_string(&self) -> String {
        self.x_min_and_max.to_string_index(1)
    }

    fn get_y_min_string(&self) -> String {
        self.y_min_and_max.to_string_index(0)
    }

    fn get_y_max_string(&self) -> String {
        self.y_min_and_max.to_string_index(1)
    }

    fn get_pads_in_x_and_y_string(&self) -> String {
        self.pads_in_x_and_y.to_string_default_is_blank(true)
    }

    fn get_taper_percent_string(&self) -> String {
        self.tapers_in_x_and_y.to_string_default_is_blank(true)
    }

    fn get_skip_views(&self) -> String {
        self.skip_views.to_string()
    }

    fn is_cumulative_correlation(&self) -> bool {
        self.cumulative_correlation
    }

    fn is_absolute_cosine_stretch(&self) -> bool {
        self.absolute_cosine_stretch
    }

    fn is_borders_in_x_and_y_set(&self) -> bool {
        !self.borders_in_x_and_y.is_default()
    }

    fn is_boundary_model_set(&self) -> bool {
        !self.boundary_model.is_empty()
    }

    fn is_no_cosine_stretch(&self) -> bool {
        self.no_cosine_stretch
    }

    fn is_number_of_patches_x_and_y_set(&self) -> bool {
        !self.number_of_patches_x_and_y.is_default()
    }

    fn is_overlap_of_patches_x_and_y_set(&self) -> bool {
        !self.overlap_of_patches_x_and_y.is_default()
    }

    fn get_starting_ending_views(&self) -> String {
        self.starting_ending_views.to_string_default_is_blank(true)
    }

    fn get_exclude_central_peak(&self) -> bool {
        self.exclude_central_peak
    }

    fn get_test_output(&self) -> Option<String> {
        self.test_output.clone()
    }

    fn get_angle_offset(&self) -> String {
        self.angle_offset.to_string()
    }
}

impl Command for TiltxcorrParam {
    /// Java `getCommandName`.
    fn get_command_name(&self) -> Option<String> {
        Some(COMMAND.to_string())
    }

    /// Java `getCommand`.
    fn get_command(&self) -> Option<String> {
        Some(self.process_name.get_comscript(self.axis_id))
    }

    /// Java `getCommandLine`.
    fn get_command_line(&self) -> Option<String> {
        Command::get_command(self)
    }

    /// Java `getCommandArray`.
    fn get_command_array(&self) -> Option<Vec<String>> {
        Some(self.process_name.get_comscript_array(self.axis_id))
    }

    /// Java `getCommandMode`.
    fn get_command_mode(&self) -> Option<&dyn CommandMode> {
        None
    }

    /// Java `isMessageReporter`.
    fn is_message_reporter(&self) -> bool {
        false
    }

    /// Java `getSubcommandDetails`.
    fn get_subcommand_details(&self) -> Option<&dyn CommandDetails> {
        None
    }

    /// Java `getSubcommandProcessName`.
    fn get_subcommand_process_name(&self) -> Option<String> {
        None
    }

    /// Java `getProcessName`.
    fn get_process_name(&self) -> Option<ProcessName> {
        Some(self.process_name)
    }

    /// Java `getCommandOutputFile`.  `new File(propertyUserDir, outputFile)`; a null
    /// `outputFile` (a NullPointerException in the source) has no file here.
    fn get_command_output_file(&self) -> Option<std::path::PathBuf> {
        let output_file = self.output_file.as_deref()?;
        Some(std::path::PathBuf::from(
            match self.manager.get_property_user_dir() {
                None => output_file.to_string(),
                Some(dir) => java_io_file_new(&dir, output_file),
            },
        ))
    }

    /// Java `getCommandInputFile`.
    fn get_command_input_file(&self) -> Option<std::path::PathBuf> {
        None
    }

    /// Java `getAxisID`.
    fn get_axis_id(&self) -> AxisID {
        self.axis_id
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
}
