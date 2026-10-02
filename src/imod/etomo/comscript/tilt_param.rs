//! `IMOD/Etomo/src/etomo/comscript/TiltParam.java`.
//!
//! Description: Tilt command model.
//!
//! Copyright: Copyright 2002 - 2025 by the Regents of the University of Colorado
//!
//! Organization: Dept. of MCD Biology, University of Colorado
//!
//! **Unchecked exceptions in `parseComScriptCommand`.**  The source parses numbers
//! with `Integer.parseInt`/`Double.parseDouble` and indexes `split` results without a
//! length check, so a malformed com file throws `NumberFormatException` or
//! `ArrayIndexOutOfBoundsException`.  Its caller (`ComScriptUtil.initialize`) catches
//! `Exception` and reports a parse error, so here each of those is returned as
//! `ParseComScriptError::NumberFormat` carrying the exception's message.
//!
//! **Java `IllegalArgumentException` in the `ProcessDetails` getters** (an unknown
//! field) is `None` here: the trait's return is `Option`, and no caller passes a
//! field this class does not handle.
//!
//! TODO(unit): needs etomo/type/StringParameter.java - `StringParameter` self.
//! TODO(unit): needs etomo/comscript/StringList.java - `excludeList`/`excludeList2`.
//! TODO(unit): needs etomo/comscript/SharedConstants.java -
//! `ACTION_IF_GPU_FAILS_DEFAULT`, `USE_GPU_BEST`.
//! TODO(unit): needs etomo/comscript/Utilities.java - `is90DegreeImageRotation`,
//! `getGoodframeFromMontageSize`.
//! TODO(unit): needs etomo/util/Goodframe.java - `getOutput`.
//! TODO(unit): needs etomo/comscript/ConstTiltalignParam.java -
//! static `getOutputZFactorFileName`.
//! TODO(unit): needs etomo/type/MetaData.java - `ApplicationManager.getConstMetaData()
//! .getImageRotation(AxisID)`.
//! TODO(unit): needs etomo/ui/swing/UIExpertUtilities.java - the manager form of
//! `getStackBinningFromFileName(BaseManager, AxisID, String, boolean)`.

use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use regex::Regex;

use super::bad_com_script_exception::BadComScriptException;
use super::com_script_command::ComScriptCommand;
use super::com_script_input_arg::ComScriptInputArg;
use super::command::Command;
use super::command_details::CommandDetails;
use super::command_mode::CommandMode;
use super::command_param::{CommandParam, ParseComScriptError};
use super::const_tilt_param::ConstTiltParam;
use super::const_tiltalign_param;
use super::field_interface::{self, FieldInterface};
use super::process_details::ProcessDetails;
use super::shared_constants;
use super::string_list::StringList;
use super::utilities;
use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::{
    ConstEtomoNumber, Type, java_lang_double_to_string, java_lang_double_value_of,
    java_lang_integer_parse_int,
};
use crate::imod::etomo::r#type::etomo_boolean2::EtomoBoolean2;
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::r#type::file_key::{self, FileKey};
use crate::imod::etomo::r#type::file_type::{self, FileType};
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::script_parameter::ScriptParameter;
use crate::imod::etomo::r#type::string_parameter::StringParameter;
use crate::imod::etomo::ui::swing::log_interface::{Loggable, LoggableException};
use crate::imod::etomo::ui::swing::ui_expert_utilities::UIExpertUtilities;
use crate::imod::etomo::ui::swing::ui_harness;
use crate::imod::etomo::util::dataset_files;
use crate::imod::etomo::util::mrc_header::MRCHeader;
use crate::imod::etomo::util::utilities::{java_io_file_new, java_lang_string_split_limit};

/// Java `LOG_KEY`.
pub const LOG_KEY: &str = "LOG";
/// Java `SUBSETSTART_KEY`.
pub const SUBSETSTART_KEY: &str = "SUBSETSTART";
/// Java `COMMAND_NAME`.
pub const COMMAND_NAME: &str = "tilt";
/// Java `LINEAR_SCALE_FACTOR_DEFAULT`.
pub const LINEAR_SCALE_FACTOR_DEFAULT: &str = "1.0";
/// Java `LINEAR_SCALE_OFFSET_DEFAULT`.
pub const LINEAR_SCALE_OFFSET_DEFAULT: &str = "0.0";
/// Java `SUPERSAMPLEFACTOR_KEY`.
pub const SUPERSAMPLEFACTOR_KEY: &str = "SuperSampleFactor";
/// Java `EXPANDLINESINPUT_KEY`.
pub const EXPANDLINESINPUT_KEY: &str = "ExpandInputLines";

/// Java nested `TiltParam.Field`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Field {
    /// Java `X_AXIS_TILT`.
    XAxisTilt,
    /// Java `FIDUCIALESS`.
    Fiducialess,
    /// Java `Z_SHIFT`.
    ZShift,
    /// Java `TILT_ANGLE_OFFSET`.
    TiltAngleOffset,
    /// Java `ADJUST_ORIGIN`.
    AdjustOrigin,
}

impl FieldInterface for Field {}

/// Java nested `TiltParam.Mode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Java `SAMPLE`.
    Sample,
    /// Java `WHOLE`.
    Whole,
    /// Java `TILT`.
    Tilt,
    /// Java `TILT_3D_FIND`.
    Tilt3dFind,
    /// Java `TRIAL_TILT`.
    TrialTilt,
}

impl Mode {
    /// Java private `DEFAULT`.
    const DEFAULT: Mode = Mode::Tilt;
}

/// Java `Mode.toString`: the `string` field.
impl std::fmt::Display for Mode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Mode::Sample => "SAMPLE",
            Mode::Whole => "WHOLE",
            Mode::Tilt => "TILT",
            Mode::Tilt3dFind => "TILT_3D_FIND",
            Mode::TrialTilt => "TRIAL_TILT",
        })
    }
}

impl CommandMode for Mode {}

/// Java final `TiltParam implements ConstTiltParam, CommandParam`.
pub struct TiltParam {
    input_file: StringParameter,
    output_file: StringParameter,
    /// tempFullImage: utility variable that is not kept up to date contains fullImageX
    /// and fullImageY
    temp_full_image: Mutex<StringParameter>,
    full_image_x: i32,
    full_image_y: i32,
    local_align_file: Mutex<StringParameter>,
    /// TODO localScale not used and doesn't go into the .com file - what is it for?
    local_scale: f64,
    log_offset: ScriptParameter,
    mode: ScriptParameter,
    /// tempOffset is not kept up to date
    temp_offset: Mutex<StringParameter>,
    tilt_angle_offset: EtomoNumber,
    tilt_axis_offset: f64,
    parallel: EtomoBoolean2,
    perpendicular: EtomoBoolean2,
    temp_radial: Mutex<StringParameter>,
    radial_bandwidth: EtomoNumber,
    radial_falloff: EtomoNumber,
    temp_scale: Mutex<StringParameter>,
    scale_f_level: f64,
    scale_coeff: f64,
    temp_shift: Mutex<StringParameter>,
    x_shift: f64,
    z_shift: EtomoNumber,
    temp_slice: Mutex<StringParameter>,
    idx_slice_start: i32,
    idx_slice_stop: i32,
    temp_subset_start: Mutex<StringParameter>,
    idx_x_subset_start: i32,
    idx_y_subset_start: i32,
    thickness: ScriptParameter,
    tilt_file: StringParameter,
    width: ScriptParameter,
    x_axis_tilt: ScriptParameter,
    x_tilt_file: Mutex<StringParameter>,
    use_z_factors: bool,
    z_factor_file_name: Mutex<StringParameter>,
    loaded_from_file: bool,
    command_mode: Mode,
    process_name: ProcessName,
    exclude_list2: StringList,
    exclude_list: StringList,
    image_binned: ScriptParameter,
    fiducialess: EtomoBoolean2,
    /// @version 3.10 Script is from an earlier version if false.
    adjust_origin: EtomoBoolean2,
    project_model: StringParameter,
    use_gpu: ScriptParameter,
    action_if_gpu_fails: StringParameter,
    done: EtomoBoolean2,
    hamming_like_filter: ScriptParameter,
    fake_sirt_iterations: ScriptParameter,
    exact_filter_size: ScriptParameter,
    falloff_is_true_sigma: EtomoBoolean2,
    super_sample_factor: ScriptParameter,
    expand_input_lines: EtomoBoolean2,
    dataset_name: Option<String>,
    manager: &'static ApplicationManager,
    axis_id: AxisID,
}

impl TiltParam {
    /// Java `TiltParam(ApplicationManager, String, AxisID)`.
    pub fn new(
        manager: &'static ApplicationManager,
        dataset_name: Option<&str>,
        axis_id: AxisID,
    ) -> TiltParam {
        let mut param = TiltParam {
            input_file: StringParameter::new("InputProjections"),
            output_file: StringParameter::new("OutputFile"),
            temp_full_image: Mutex::new(StringParameter::new("FULLIMAGE")),
            full_image_x: i32::MIN,
            full_image_y: i32::MIN,
            local_align_file: Mutex::new(StringParameter::new("LOCALFILE")),
            local_scale: f64::NAN,
            log_offset: ScriptParameter::new_with_type_and_name(Type::Double, LOG_KEY),
            mode: ScriptParameter::new_with_name("MODE"),
            temp_offset: Mutex::new(StringParameter::new("OFFSET")),
            tilt_angle_offset: EtomoNumber::new_with_type(Some(Type::Double)),
            tilt_axis_offset: f64::NAN,
            parallel: EtomoBoolean2::new_with_name("PARALLEL"),
            perpendicular: EtomoBoolean2::new_with_name("PERPENDICULAR"),
            temp_radial: Mutex::new(StringParameter::new("RADIAL")),
            radial_bandwidth: EtomoNumber::new_with_type(Some(Type::Double)),
            radial_falloff: EtomoNumber::new_with_type(Some(Type::Double)),
            temp_scale: Mutex::new(StringParameter::new("SCALE")),
            scale_f_level: f64::NAN,
            scale_coeff: f64::NAN,
            temp_shift: Mutex::new(StringParameter::new("SHIFT")),
            x_shift: f64::NAN,
            z_shift: EtomoNumber::new_with_type(Some(Type::Double)),
            temp_slice: Mutex::new(StringParameter::new("SLICE")),
            idx_slice_start: i32::MIN,
            idx_slice_stop: i32::MIN,
            temp_subset_start: Mutex::new(StringParameter::new(SUBSETSTART_KEY)),
            idx_x_subset_start: i32::MIN,
            idx_y_subset_start: i32::MIN,
            thickness: ScriptParameter::new_with_name("THICKNESS"),
            tilt_file: StringParameter::new("TILTFILE"),
            width: ScriptParameter::new_with_name("WIDTH"),
            x_axis_tilt: ScriptParameter::new_with_type_and_name(Type::Double, "XAXISTILT"),
            x_tilt_file: Mutex::new(StringParameter::new("XTILTFILE")),
            use_z_factors: false,
            z_factor_file_name: Mutex::new(StringParameter::new("ZFACTORFILE")),
            loaded_from_file: false,
            command_mode: Mode::DEFAULT,
            process_name: ProcessName::TILT,
            exclude_list2: StringList::new_with_n_elements(0),
            exclude_list: StringList::new_with_n_elements(0),
            image_binned: ScriptParameter::new_with_name("IMAGEBINNED"),
            fiducialess: EtomoBoolean2::new_with_name("Fiducialess"),
            adjust_origin: EtomoBoolean2::new_with_name("AdjustOrigin"),
            project_model: StringParameter::new("ProjectModel"),
            use_gpu: ScriptParameter::new_with_name("UseGPU"),
            action_if_gpu_fails: StringParameter::new("ActionIfGPUFails"),
            done: EtomoBoolean2::new_with_name("DONE"),
            hamming_like_filter: ScriptParameter::new_with_type_and_name(
                Type::Double,
                "HammingLikeFilter",
            ),
            fake_sirt_iterations: ScriptParameter::new_with_name("FakeSIRTiterations"),
            exact_filter_size: ScriptParameter::new_with_name("ExactFilterSize"),
            falloff_is_true_sigma: EtomoBoolean2::new_with_name("FalloffIsTrueSigma"),
            super_sample_factor: ScriptParameter::new_with_type_and_name(
                Type::Integer,
                SUPERSAMPLEFACTOR_KEY,
            ),
            expand_input_lines: EtomoBoolean2::new_with_name(EXPANDLINESINPUT_KEY),
            dataset_name: dataset_name.map(|name| name.to_string()),
            manager,
            axis_id,
        };
        // do not default imageBinned
        param.image_binned.set_floor(1);
        param.input_file.set(Some(""));
        param.output_file.set(Some(""));
        param.exclude_list.set_key(Some("EXCLUDELIST"));
        param.exclude_list.set_successive_entries_accumulate();
        param.exclude_list.set_convert_to_single_entry();
        param.exclude_list2.set_key(Some("EXCLUDELIST2"));
        param.exclude_list2.set_successive_entries_accumulate();
        param.exclude_list2.set_convert_to_single_entry();
        param.local_align_file.lock().unwrap().set(Some(""));
        param.tilt_file.set(Some(""));
        param.x_tilt_file.lock().unwrap().set(Some(""));
        param.project_model.set(Some(""));
        param
            .action_if_gpu_fails
            .set(Some(shared_constants::ACTION_IF_GPU_FAILS_DEFAULT));
        param
    }

    /// Java `setAdjustOrigin`.
    pub fn set_adjust_origin(&mut self, input: bool) {
        self.adjust_origin.set_boolean(input);
    }

    /// Java `setCommandMode(Mode)`.
    pub fn set_command_mode(&mut self, input: Mode) {
        self.command_mode = input;
    }

    /// Java `getLocalAlignFile`.
    pub fn get_local_align_file(&self) -> String {
        self.local_align_file.lock().unwrap().to_string()
    }

    /// Java `isParallel`.
    pub fn is_parallel(&self) -> bool {
        self.parallel.is()
    }

    /// Java `setUseGpu`.
    pub fn set_use_gpu(&mut self, input: bool) {
        if input {
            self.use_gpu.set_int(shared_constants::USE_GPU_BEST);
        } else {
            self.use_gpu.reset();
        }
    }

    /// Java `isPerpendicular`.
    pub fn is_perpendicular(&self) -> bool {
        self.perpendicular.is()
    }

    /// Java `getTiltFile`.
    pub fn get_tilt_file(&self) -> String {
        self.tilt_file.to_string()
    }

    /// Java `getExcludeList`.  Gets the excludeList.
    pub fn get_exclude_list(&self) -> String {
        self.exclude_list.to_string()
    }

    /// Java `isUseZFactors`.
    pub fn is_use_z_factors(&self) -> bool {
        self.use_z_factors
    }

    /// Java `getTiltAxisOffset`.
    pub fn get_tilt_axis_offset(&self) -> f64 {
        self.tilt_axis_offset
    }

    /// Java `hasTiltAxisOffset`.
    pub fn has_tilt_axis_offset(&self) -> bool {
        if self.tilt_axis_offset.is_nan() {
            return false;
        }
        true
    }

    /// Java `getFullImageY`.
    pub fn get_full_image_y(&self) -> i32 {
        self.full_image_y
    }

    /// Java `isOldVersion`.  identifies an old version
    pub fn is_old_version(&self) -> bool {
        self.loaded_from_file && self.image_binned.is_null()
    }

    /// Java `backwardCompatibleParseComScriptCommand`.  Get the parameters from the
    /// ComScriptCommand.
    ///
    /// The source declares only `BadComScriptException`; the unchecked
    /// `NumberFormatException`, `ArrayIndexOutOfBoundsException` (a keyword line with
    /// no value) and `NullPointerException` (an argument with no text) it can also throw
    /// come back as `ParseComScriptError::NumberFormat`, which is what its caller's
    /// `catch (Exception)` reports.
    pub fn backward_compatible_parse_com_script_command(
        &mut self,
        script_command: &ComScriptCommand,
    ) -> Result<(), ParseComScriptError> {
        let number_format = |message: String| ParseComScriptError::NumberFormat(message);
        let out_of_bounds = |index: usize, length: usize| {
            ParseComScriptError::NumberFormat(format!(
                "Index {index} out of bounds for length {length}"
            ))
        };
        // get the input arguments from the command
        let input_args: Vec<Rc<RefCell<ComScriptInputArg>>> =
            self.get_input_arguments(script_command)?;

        // Get the input and output file names from the input arguments
        let n_input_args = input_args.len();
        let mut arg_index = 0;
        self.input_file
            .set(input_args[arg_index].borrow().get_argument());
        arg_index += 1;
        self.output_file
            .set(input_args[arg_index].borrow().get_argument());
        arg_index += 1;
        let mut _found_done = false;
        let whitespace = Regex::new(r"\s+").unwrap();
        for i in arg_index..n_input_args {
            // split the line into the parameter name and the rest of the line
            let argument = match input_args[i].borrow().get_argument() {
                None => {
                    return Err(ParseComScriptError::NumberFormat(
                        "java.lang.NullPointerException".to_string(),
                    ));
                }
                Some(argument) => argument.to_string(),
            };
            let tokens = java_lang_string_split_limit(&argument, &whitespace, 2);
            let token1 = || tokens.get(1).ok_or_else(|| out_of_bounds(1, tokens.len()));
            if tokens[0].eq_ignore_ascii_case("IMAGEBINNED") {
                self.image_binned.set_string(Some(token1()?.as_str()));
            }
            if tokens[0].eq_ignore_ascii_case("DONE") {
                _found_done = true;
            }
            if tokens[0].eq_ignore_ascii_case("EXCLUDELIST") {
                self.exclude_list.parse_string(Some(token1()?.as_str()));
            }
            if tokens[0].eq_ignore_ascii_case("EXCLUDELIST2") {
                self.exclude_list2.parse_string(Some(token1()?.as_str()));
            }
            if tokens[0].eq_ignore_ascii_case("FULLIMAGE") {
                let params = java_lang_string_split_limit(token1()?, &whitespace, 2);
                self.full_image_x =
                    java_lang_integer_parse_int(&params[0]).map_err(number_format)?;
                self.full_image_y = java_lang_integer_parse_int(
                    params
                        .get(1)
                        .ok_or_else(|| out_of_bounds(1, params.len()))?,
                )
                .map_err(number_format)?;
            }
            if tokens[0].eq_ignore_ascii_case("LOCALFILE") {
                self.local_align_file
                    .lock()
                    .unwrap()
                    .set(Some(token1()?.as_str()));
            }
            if tokens[0].eq_ignore_ascii_case(LOG_KEY) {
                self.log_offset.set_string(Some(token1()?.as_str()));
            }
            if tokens[0].eq_ignore_ascii_case("MODE") {
                self.mode.set_string(Some(token1()?.as_str()));
            }
            if tokens[0].eq_ignore_ascii_case("OFFSET") {
                let params = java_lang_string_split_limit(token1()?, &whitespace, 2);
                self.tilt_angle_offset.set_string(Some(params[0].as_str()));
                if params.len() > 1 {
                    self.tilt_axis_offset =
                        java_lang_double_value_of(&params[1]).map_err(number_format)?;
                }
            }
            if tokens[0].eq_ignore_ascii_case("PARALLEL") {
                self.perpendicular.set_boolean(false);
                self.parallel.set_boolean(true);
            }
            if tokens[0].eq_ignore_ascii_case("PERPENDICULAR") {
                self.perpendicular.set_boolean(true);
                self.parallel.set_boolean(false);
            }
            if tokens[0].eq_ignore_ascii_case("RADIAL") {
                let params = java_lang_string_split_limit(token1()?, &whitespace, 2);
                self.radial_bandwidth.set_string(Some(params[0].as_str()));
                self.radial_falloff.set_string(Some(
                    params
                        .get(1)
                        .ok_or_else(|| out_of_bounds(1, params.len()))?
                        .as_str(),
                ));
            }
            if tokens[0].eq_ignore_ascii_case("SCALE") {
                let params = java_lang_string_split_limit(token1()?, &whitespace, 2);
                self.scale_f_level =
                    java_lang_double_value_of(&params[0]).map_err(number_format)?;
                self.scale_coeff = java_lang_double_value_of(
                    params
                        .get(1)
                        .ok_or_else(|| out_of_bounds(1, params.len()))?,
                )
                .map_err(number_format)?;
            }
            if tokens[0].eq_ignore_ascii_case("SHIFT") {
                let params = java_lang_string_split_limit(token1()?, &whitespace, 2);
                self.x_shift = java_lang_double_value_of(&params[0]).map_err(number_format)?;
                if params.len() > 1 {
                    self.z_shift.set_string(Some(params[1].as_str()));
                }
            }
            if tokens[0].eq_ignore_ascii_case("SLICE") {
                let params = java_lang_string_split_limit(token1()?, &whitespace, 3);
                self.idx_slice_start =
                    java_lang_integer_parse_int(&params[0]).map_err(number_format)?;
                self.idx_slice_stop = java_lang_integer_parse_int(
                    params
                        .get(1)
                        .ok_or_else(|| out_of_bounds(1, params.len()))?,
                )
                .map_err(number_format)?;
                // Increment is being ignored
            }
            if tokens[0].eq_ignore_ascii_case("SUBSETSTART") {
                let params = java_lang_string_split_limit(token1()?, &whitespace, 2);
                self.idx_x_subset_start =
                    java_lang_integer_parse_int(&params[0]).map_err(number_format)?;
                self.idx_y_subset_start = java_lang_integer_parse_int(
                    params
                        .get(1)
                        .ok_or_else(|| out_of_bounds(1, params.len()))?,
                )
                .map_err(number_format)?;
            }
            if tokens[0].eq_ignore_ascii_case("THICKNESS") {
                self.thickness.set_string(Some(token1()?.as_str()));
            }
            if tokens[0].eq_ignore_ascii_case("TILTFILE") {
                self.tilt_file.set(Some(token1()?.as_str()));
            }
            if tokens[0].eq_ignore_ascii_case("WIDTH") {
                self.width.set_string(Some(token1()?.as_str()));
            }
            if tokens[0].eq_ignore_ascii_case("XAXISTILT") {
                self.x_axis_tilt.set_string(Some(token1()?.as_str()));
            }
            if tokens[0].eq_ignore_ascii_case("XTILTFILE") {
                self.x_tilt_file
                    .lock()
                    .unwrap()
                    .set(Some(token1()?.as_str()));
            }
            if tokens[0].eq_ignore_ascii_case("ZFACTORFILE") {
                self.use_z_factors = true;
                self.z_factor_file_name
                    .lock()
                    .unwrap()
                    .set(Some(token1()?.as_str()));
            }
            if tokens[0].eq_ignore_ascii_case("AdjustOrigin") {
                self.adjust_origin.set_boolean(true);
            }
            if tokens[0].eq_ignore_ascii_case("ProjectModel") {
                self.project_model.set(Some(token1()?.as_str()));
            }
        }
        self.loaded_from_file = true;
        Ok(())
    }

    /// Java `setImageBinned(int)`.
    pub fn set_image_binned_int(&mut self, image_binned: i32) -> &ConstEtomoNumber {
        self.image_binned.set_int(image_binned);
        &self.image_binned
    }

    /// Java `setImageBinned()`.  If the current binning can be retrieved, set
    /// imageBinned to current binning.  If not, and imageBinned is null then set
    /// imageBinned to 1.
    pub fn set_image_binned(&mut self) -> &ConstEtomoNumber {
        let mut current_binning = EtomoNumber::new();
        let input_file = self.input_file.to_string();
        current_binning.set_int(
            UIExpertUtilities::INSTANCE.get_stack_binning_from_file_name(
                self.manager,
                self.axis_id,
                Some(input_file.as_str()),
                true,
            ),
        );
        if !current_binning.is_null() {
            self.image_binned
                .set_const_etomo_number(Some(&*current_binning));
        } else if self.image_binned.is_null() {
            self.image_binned.set_int(1);
        }
        &self.image_binned
    }

    /// Java `setFiducialess`.
    pub fn set_fiducialess(&mut self, input: bool) {
        self.fiducialess.set_boolean(input);
    }

    /// Java `setExcludeList`.  Sets the excludeList.
    pub fn set_exclude_list(&mut self, list: Option<&str>) {
        self.exclude_list.parse_string(list);
    }

    /// Java `setExcludeList2`.  Sets the excludeList2.
    pub fn set_exclude_list2(&mut self, list: Option<&str>) {
        self.exclude_list2.parse_string(list);
    }

    /// Java `resetExcludeList`.
    pub fn reset_exclude_list(&mut self) {
        self.exclude_list.set_n_elements(0);
    }

    /// Java `setMontageSubsetStart`.  If the tilt axis angle is closer to 90 degree, x
    /// and y need to be transposed.  The .ali or _3dfind.ali file will already be
    /// transposed.  So just transpose the goodframe outputs.
    ///
    /// The source catches the header read's `IOException` ("ok if tilt is being
    /// updated before .ali exists") and lets its `InvalidParameterException`
    /// propagate.  `MRCHeader::read_with_manager` reports both as one `Err(String)`,
    /// so both propagate here.
    pub fn set_montage_subset_start(&mut self) -> Result<(), String> {
        self.reset_subset_start();
        let goodframe = utilities::get_goodframe_from_montage_size(self.axis_id, self.manager);
        if let Some(goodframe) = goodframe {
            let input_file = self.input_file.to_string();
            let header = match MRCHeader::get_instance_from_file_name(
                self.manager,
                Some(self.axis_id),
                Some(input_file.as_str()),
            ) {
                None => return Ok(()),
                Some(header) => header,
            };
            if !header.borrow_mut().read_with_manager(self.manager)? {
                // ok if tilt is being updated before .ali exists
                return Ok(());
            }
            let goodframe_x;
            let goodframe_y;
            if utilities::is_90_degree_image_rotation(
                self.manager
                    .get_const_meta_data()
                    .get_image_rotation(self.axis_id)
                    .get_double(),
            ) {
                // transpose x and y
                goodframe_x = goodframe.get_output(1).get_int();
                goodframe_y = goodframe.get_output(0).get_int();
            } else {
                goodframe_x = goodframe.get_output(0).get_int();
                goodframe_y = goodframe.get_output(1).get_int();
            }
            // Multiply header output by binning to work with the goodframe output,
            // which is unbinned.
            let n_columns = header.borrow().get_n_columns();
            let binned = self.set_image_binned().get_int();
            self.idx_x_subset_start = goodframe_x.wrapping_sub(n_columns.wrapping_mul(binned)) / 2;
            let n_rows = header.borrow().get_n_rows();
            let binned = self.set_image_binned().get_int();
            self.idx_y_subset_start = goodframe_y.wrapping_sub(n_rows.wrapping_mul(binned)) / 2;
        }
        Ok(())
    }

    /// Java `resetSubsetStart`.
    pub fn reset_subset_start(&mut self) {
        self.idx_x_subset_start = 0;
        self.idx_y_subset_start = 0;
    }

    /// Java `setSubsetStart`.  If the tilt axis angle is closer to 90 degree, x and y
    /// need to be transposed.  The .ali file will already be transposed.  So just
    /// transpose the stackHeader columns (x) and rows (y).
    ///
    /// The source returns true after an `IOException` and opens a message dialog and
    /// returns false after an `InvalidParameterException`.  `read_with_manager`
    /// reports both as one `Err(String)`; it takes the `InvalidParameterException`
    /// path, so a failed read is always reported.
    pub fn set_subset_start(&mut self) -> bool {
        self.reset_subset_start();
        let result: Result<bool, String> = (|| -> Result<bool, String> {
            let stack_header = match MRCHeader::get_instance_from_file_type(
                self.manager,
                Some(self.axis_id),
                &file_type::CLASS.raw_stack,
            ) {
                None => return Ok(true),
                Some(header) => header,
            };
            if !stack_header.borrow_mut().read_with_manager(self.manager)? {
                return Ok(true);
            }
            let input_file = self.input_file.to_string();
            let ali_header = match MRCHeader::get_instance_from_file_name(
                self.manager,
                Some(self.axis_id),
                Some(input_file.as_str()),
            ) {
                None => return Ok(true),
                Some(header) => header,
            };
            if !ali_header.borrow_mut().read_with_manager(self.manager)? {
                return Ok(true);
            }
            let stack_x;
            let stack_y;
            if utilities::is_90_degree_image_rotation(
                self.manager
                    .get_const_meta_data()
                    .get_image_rotation(self.axis_id)
                    .get_double(),
            ) {
                stack_x = stack_header.borrow().get_n_rows();
                stack_y = stack_header.borrow().get_n_columns();
            } else {
                stack_x = stack_header.borrow().get_n_columns();
                stack_y = stack_header.borrow().get_n_rows();
            }
            let n_columns = ali_header.borrow().get_n_columns();
            let binned = self.set_image_binned().get_int();
            self.idx_x_subset_start = stack_x.wrapping_sub(n_columns.wrapping_mul(binned)) / 2;
            let n_rows = ali_header.borrow().get_n_rows();
            let binned = self.set_image_binned().get_int();
            self.idx_y_subset_start = stack_y.wrapping_sub(n_rows.wrapping_mul(binned)) / 2;
            Ok(true)
        })();
        match result {
            Ok(value) => value,
            Err(message) => {
                // `e.printStackTrace()`
                eprintln!("{message}");
                ui_harness::post_message_dialog(
                    Some(self.manager as &'static dyn BaseManager),
                    format!("Unable to set subset start in tilt.com.\n{message}"),
                    "Setting Comscript Failed".to_string(),
                    Some(self.axis_id),
                );
                false
            }
        }
    }

    /// Java `setMontageFullImage`.  If the tilt angle axis is closer to 90 degree,
    /// transpose x and y.
    pub fn set_montage_full_image(&mut self) {
        let goodframe = utilities::get_goodframe_from_montage_size(self.axis_id, self.manager);
        if let Some(goodframe) = goodframe {
            if utilities::is_90_degree_image_rotation(
                self.manager
                    .get_const_meta_data()
                    .get_image_rotation(self.axis_id)
                    .get_double(),
            ) {
                self.full_image_x = goodframe.get_output(1).get_int();
                self.full_image_y = goodframe.get_output(0).get_int();
            } else {
                self.full_image_x = goodframe.get_output(0).get_int();
                self.full_image_y = goodframe.get_output(1).get_int();
            }
        }
    }

    /// Java `setFullImage(File)`.  If the tilt angle axis is closer to 90 degree,
    /// transpose x and y.  Both of the read's exceptions are swallowed, as in the
    /// source (which prints the `InvalidParameterException`'s stack trace).
    pub fn set_full_image(&mut self, stack: &std::path::Path) {
        let stack_name = stack
            .file_name()
            .map(|name| name.to_string_lossy().to_string())
            .unwrap_or_default();
        let header = match MRCHeader::get_instance_in_dir(
            self.manager.get_property_user_dir().as_deref(),
            Some(stack_name.as_str()),
            Some(self.axis_id),
        ) {
            None => return,
            Some(header) => header,
        };
        match header.borrow_mut().read_with_manager(self.manager) {
            Ok(false) => return,
            Ok(true) => {}
            Err(message) => {
                eprintln!("{message}");
                return;
            }
        }
        if utilities::is_90_degree_image_rotation(
            self.manager
                .get_const_meta_data()
                .get_image_rotation(self.axis_id)
                .get_double(),
        ) {
            self.full_image_x = header.borrow().get_n_rows();
            self.full_image_y = header.borrow().get_n_columns();
        } else {
            self.full_image_x = header.borrow().get_n_columns();
            self.full_image_y = header.borrow().get_n_rows();
        }
    }

    /// Java `setFullImageX`.
    pub fn set_full_image_x(&mut self, input: i32) {
        self.full_image_x = input;
    }

    /// Java `setFullImageY`.
    pub fn set_full_image_y(&mut self, input: i32) {
        self.full_image_y = input;
    }

    /// Java `setHammingLikeFilter`.
    pub fn set_hamming_like_filter(&mut self, input: Option<&str>) {
        self.hamming_like_filter.set_string(input);
    }

    /// Java `resetHammingLikeFilter`.
    pub fn reset_hamming_like_filter(&mut self) {
        self.hamming_like_filter.reset();
    }

    /// Java `setFakeSIRTiterations`.
    pub fn set_fake_sirt_iterations(&mut self, input: Option<&str>) {
        self.fake_sirt_iterations.set_string(input);
    }

    /// Java `resetFakeSIRTiterations`.
    pub fn reset_fake_sirt_iterations(&mut self) {
        self.fake_sirt_iterations.reset();
    }

    /// Java `setExactFilterSize`.
    pub fn set_exact_filter_size(&mut self, input: Option<&str>) {
        self.exact_filter_size.set_string(input);
    }

    /// Java `resetExactFilterSize`.
    pub fn reset_exact_filter_size(&mut self) {
        self.exact_filter_size.reset();
    }

    /// Java `setIdxSliceStart`.
    pub fn set_idx_slice_start(&mut self, i: i32) {
        self.idx_slice_start = i;
    }

    /// Java `setIdxSliceStop`.
    pub fn set_idx_slice_stop(&mut self, i: i32) {
        self.idx_slice_stop = i;
    }

    /// Java `setIdxXSubsetStart`.
    pub fn set_idx_x_subset_start(&mut self, input: i32) {
        self.idx_x_subset_start = input;
    }

    /// Java `setIdxYSubsetStart`.
    pub fn set_idx_y_subset_start(&mut self, input: i32) {
        self.idx_y_subset_start = input;
    }

    /// Java `resetIdxSlice`.
    pub fn reset_idx_slice(&mut self) {
        self.idx_slice_start = i32::MIN;
        self.idx_slice_stop = i32::MIN;
    }

    /// Java `setInputFile`.
    pub fn set_input_file(&mut self, file: Option<&str>) {
        self.input_file.set(file);
    }

    /// Java `setLoadedFromFile`.
    pub fn set_loaded_from_file(&mut self, input: bool) {
        self.loaded_from_file = input;
    }

    /// Java `resetInputFile`.
    pub fn reset_input_file(&mut self) {
        self.input_file.set(Some(""));
    }

    /// Java `setLocalAlignFile`.
    pub fn set_local_align_file(&mut self, filename: Option<&str>) {
        self.local_align_file.lock().unwrap().set(filename);
    }

    /// Java `setLocalScale`.
    pub fn set_local_scale(&mut self, input: f64) {
        self.local_scale = input;
    }

    /// Java `setLogOffset`.
    pub fn set_log_offset(&mut self, input: f64) {
        self.log_offset.set_double(input);
    }

    /// Java `setProjectModel(FileType)`.
    pub fn set_project_model(&mut self, input: &FileType) {
        let file_name = input.get_file_name(
            Some(self.manager as &'static dyn BaseManager),
            Some(self.axis_id),
        );
        self.project_model.set(file_name.as_deref());
    }

    /// Java `resetLocalAlignFile`.
    pub fn reset_local_align_file(&mut self) {
        self.local_align_file.lock().unwrap().set(Some(""));
    }

    /// Java `setLogShift`.
    pub fn set_log_shift(&mut self, shift: f64) {
        self.log_offset.set_double(shift);
    }

    /// Java `resetLogShift`.
    pub fn reset_log_shift(&mut self) {
        self.log_offset.reset();
    }

    /// Java `setMode`.
    pub fn set_mode(&mut self, new_mode: i32) {
        self.mode.set_int(new_mode);
    }

    /// Java `resetMode`.
    pub fn reset_mode(&mut self) {
        self.mode.reset();
    }

    /// Java `setOutputFile`.
    pub fn set_output_file(&mut self, file: Option<&str>) {
        self.output_file.set(file);
    }

    /// Java `resetOutputFile`.
    pub fn reset_output_file(&mut self) {
        self.output_file.set(Some(""));
    }

    /// Java `setParallel`.
    pub fn set_parallel(&mut self) {
        self.parallel.set_boolean(true);
        self.perpendicular.set_boolean(false);
    }

    /// Java `setPerpendicular`.
    pub fn set_perpendicular(&mut self) {
        self.parallel.set_boolean(false);
        self.perpendicular.set_boolean(true);
    }

    /// Java `setProcessName(ProcessName)`.
    pub fn set_process_name(&mut self, input: ProcessName) {
        self.process_name = input;
    }

    /// Java `setProcessName(String)`.
    pub fn set_process_name_string(&mut self, input: Option<&str>) {
        self.process_name = match ProcessName::get_instance(input) {
            None => ProcessName::TILT,
            Some(process_name) => process_name,
        };
    }

    /// Java `resetAxisOrder`.
    pub fn reset_axis_order(&mut self) {
        self.parallel.reset();
        self.perpendicular.reset();
    }

    /// Java `setRadialBandwidth`.
    pub fn set_radial_bandwidth(&mut self, value: Option<&str>) {
        self.radial_bandwidth.set_string(value);
    }

    /// Java `setRadialFalloff`.
    pub fn set_radial_falloff(&mut self, value: Option<&str>) {
        self.radial_falloff.set_string(value);
    }

    /// Java `resetRadialFilter`.
    pub fn reset_radial_filter(&mut self) {
        self.radial_bandwidth.reset();
        self.radial_falloff.reset();
    }

    /// Java `setScale`.
    pub fn set_scale(&mut self, f_level: f64, coef: f64) {
        self.scale_coeff = coef;
        self.scale_f_level = f_level;
    }

    /// Java `resetScale`.
    pub fn reset_scale(&mut self) {
        self.scale_coeff = f64::NAN;
        self.scale_f_level = f64::NAN;
    }

    /// Java `setScaleCoeff`.
    pub fn set_scale_coeff(&mut self, scale_coeff: f64) {
        self.scale_coeff = scale_coeff;
    }

    /// Java `setScaleFLevel`.
    pub fn set_scale_f_level(&mut self, scale_f_level: f64) {
        self.scale_f_level = scale_f_level;
    }

    /// Java `setThickness(int)`.
    pub fn set_thickness_int(&mut self, input: i32) {
        self.thickness.set_int(input);
    }

    /// Java `setThickness(String)`.
    pub fn set_thickness(&mut self, new_thickness: Option<&str>) {
        self.thickness.set_string(new_thickness);
    }

    /// Java `resetThickness`.
    pub fn reset_thickness(&mut self) {
        self.thickness.reset();
    }

    /// Java `setTiltAngleOffset`.
    pub fn set_tilt_angle_offset(&mut self, input: Option<&str>) {
        self.tilt_angle_offset.set_string(input);
    }

    /// Java `resetTiltAngleOffset`.
    pub fn reset_tilt_angle_offset(&mut self) {
        self.tilt_angle_offset.reset();
    }

    /// Java `setTiltAxisOffset`.
    pub fn set_tilt_axis_offset(&mut self, d: f64) {
        self.tilt_axis_offset = d;
    }

    /// Java `resetTiltAxisOffset`.
    pub fn reset_tilt_axis_offset(&mut self) {
        self.tilt_axis_offset = f64::NAN;
    }

    /// Java `setTiltFile`.
    pub fn set_tilt_file(&mut self, filename: Option<&str>) {
        self.tilt_file.set(filename);
    }

    /// Java `setWidth(int)`, deprecated 7/28/2018.
    #[deprecated]
    pub fn set_width_int(&mut self, i: i32) {
        self.width.set_int(i);
    }

    /// Java `setWidth(String)`.
    pub fn set_width(&mut self, input: Option<&str>) {
        self.width.set_string(input);
    }

    /// Java `resetWidth`.
    pub fn reset_width(&mut self) {
        self.width.reset();
    }

    /// Java `setXAxisTilt(double)`.
    pub fn set_x_axis_tilt_double(&mut self, input: f64) {
        self.x_axis_tilt.set_double(input);
    }

    /// Java `setXAxisTilt(String)`.
    pub fn set_x_axis_tilt(&mut self, angle: Option<&str>) {
        self.x_axis_tilt.set_string(angle);
    }

    /// Java `resetXAxisTilt`.
    pub fn reset_x_axis_tilt(&mut self) {
        self.x_axis_tilt.reset();
    }

    /// Java `setXShift`.
    pub fn set_x_shift(&mut self, d: f64) {
        self.x_shift = d;
    }

    /// Java `setXTiltFile`.
    pub fn set_x_tilt_file(&mut self, input: Option<&str>) {
        self.x_tilt_file.lock().unwrap().set(input);
    }

    /// Java `setZFactorFileName`.
    pub fn set_z_factor_file_name(&mut self, input: Option<&str>) {
        self.z_factor_file_name.lock().unwrap().set(input);
    }

    /// Java `resetXShift`.
    pub fn reset_x_shift(&mut self) {
        self.x_shift = f64::NAN;
    }

    /// Java `setZShift(double)`.
    pub fn set_z_shift_double(&mut self, d: f64) {
        self.z_shift.set_double(d);
    }

    /// Java `setZShift(String)`.
    pub fn set_z_shift(&mut self, z_shift: Option<&str>) {
        self.z_shift.set_string(z_shift);
    }

    /// Java `resetZShift`.
    pub fn reset_z_shift(&mut self) {
        self.z_shift.reset();
    }

    /// Java `setUseZFactors`.
    pub fn set_use_z_factors(&mut self, use_z_factors: bool) {
        self.use_z_factors = use_z_factors;
    }

    /// Java `setSuperSampleFactor`.
    pub fn set_super_sample_factor(&mut self, input: i32) {
        self.super_sample_factor.set_int(input);
    }

    /// Java `resetSuperSampleFactor`.
    pub fn reset_super_sample_factor(&mut self) {
        self.super_sample_factor.reset();
    }

    /// Java `setExpandInputLines`.
    pub fn set_expand_input_lines(&mut self, input: bool) {
        self.expand_input_lines.set_boolean(input);
    }

    /// Java private `getInputArguments`.  Get the standand input arguments from the
    /// ComScriptCommand validating the name of the command and the appropriate number
    /// of input arguments.
    ///
    /// A null command (`getCommand().equals` would throw NullPointerException) is
    /// treated as not a tilt command.
    fn get_input_arguments(
        &self,
        script_command: &ComScriptCommand,
    ) -> Result<Vec<Rc<RefCell<ComScriptInputArg>>>, BadComScriptException> {
        // Check to be sure that it is a tiltxcorr xommand
        if script_command.get_command() != Some("tilt") {
            return Err(BadComScriptException::new("Not a tiltalign command"));
        }

        // Get the input arguments parameters to preserve the comments
        let input_args = script_command.get_input_arguments();
        if input_args.len() < 3 {
            return Err(BadComScriptException::new(&format!(
                "Incorrect number of input arguments to tiltalign command\nGot {} expected at least 3.",
                input_args.len()
            )));
        }
        Ok(input_args)
    }

    /// Java `upgradeOldVersion`.  Backward compatibility fix.  Unbinned all the
    /// parameters which where binned in the old version.  Ignore parameters with reset
    /// values.  The param should be loaded from a com file before running this
    /// function.  Returns true if changes where made.
    pub fn upgrade_old_version(&mut self, correction_binning: i32, current_binning: i32) -> bool {
        if !self.is_old_version() {
            return false;
        }
        self.image_binned.set_int(current_binning);
        // Currently this function only multiplies by binning, so there is nothing to
        // do if binning is 1.
        if correction_binning != 1 {
            if self.full_image_x != i32::MIN && self.full_image_x != 0 {
                self.full_image_x = self.full_image_x.wrapping_mul(correction_binning);
            }
            if self.full_image_y != i32::MIN && self.full_image_y != 0 {
                self.full_image_y = self.full_image_y.wrapping_mul(correction_binning);
            }
            if !self.width.is_null() && !self.width.equals_int(0) {
                self.width.multiply_int(correction_binning);
            }
            if !self.z_shift.is_null() {
                let mut f_z_shift = self.z_shift.get_double();
                if f_z_shift != 0.0 {
                    f_z_shift *= correction_binning as f64;
                    self.z_shift.set_double(f_z_shift);
                }
            }
            if !self.x_shift.is_nan() && self.x_shift != 0.0 {
                self.x_shift *= correction_binning as f64;
            }
            if self.idx_slice_start != i32::MIN && self.idx_slice_start != 0 {
                self.idx_slice_start = self.idx_slice_start.wrapping_mul(correction_binning);
            }
            if self.idx_slice_stop != i32::MIN && self.idx_slice_stop != 0 {
                self.idx_slice_stop = self.idx_slice_stop.wrapping_mul(correction_binning);
            }
            if !self.thickness.is_null() && !self.thickness.equals_int(0) {
                self.thickness.multiply_int(correction_binning);
            }
        }
        let mut buffer = format!("\nUpgraded tilt{}.com:\n", self.axis_id.get_extension());
        if correction_binning > 1 {
            buffer.push_str(&format!(
                "Multiplied binned FullImage, Width, Offset, IdxSliceStart, and/or Thickness by {correction_binning}.\n"
            ));
        }
        buffer.push_str(&format!(
            "Added {} {}.\n",
            self.image_binned.get_name(),
            current_binning
        ));
        eprintln!("{buffer}");
        true
    }
}

impl ConstTiltParam for TiltParam {
    fn get_image_binned(&self) -> &ConstEtomoNumber {
        &self.image_binned
    }

    fn get_input_file(&self) -> String {
        self.input_file.to_string()
    }

    fn get_log_shift(&self) -> String {
        self.log_offset.to_string()
    }

    fn has_log_offset(&self) -> bool {
        !self.log_offset.is_null()
    }

    fn get_mode(&self) -> i32 {
        self.mode.get_int()
    }

    fn has_mode(&self) -> bool {
        !self.mode.is_null()
    }

    fn has_local_align_file(&self) -> bool {
        if self.local_align_file.lock().unwrap().equals(Some("")) {
            return false;
        }
        true
    }

    fn get_output_file(&self) -> String {
        self.output_file.to_string()
    }

    fn is_use_gpu(&self) -> bool {
        !self.use_gpu.is_null()
    }

    fn is_fiducialess(&self) -> bool {
        self.fiducialess.is()
    }

    fn is_falloff_is_true_sigma(&self) -> bool {
        self.falloff_is_true_sigma.is()
    }

    fn get_radial_bandwidth(&self) -> String {
        self.radial_bandwidth.to_string()
    }

    fn has_radial_weighting_function(&self) -> bool {
        !self.radial_bandwidth.is_null()
    }

    fn get_thickness(&self) -> i32 {
        self.thickness.get_int()
    }

    fn has_thickness(&self) -> bool {
        !self.thickness.is_null()
    }

    fn get_x_axis_tilt(&self) -> f64 {
        self.x_axis_tilt.get_double()
    }

    fn get_x_axis_tilt_string(&self) -> String {
        self.x_axis_tilt.to_string()
    }

    fn has_x_axis_tilt(&self) -> bool {
        !self.x_axis_tilt.is_null()
    }

    fn get_exclude_list2(&self) -> String {
        self.exclude_list2.to_string()
    }

    fn get_radial_falloff(&self) -> String {
        self.radial_falloff.to_string()
    }

    fn get_width(&self) -> i32 {
        self.width.get_int()
    }

    fn has_width(&self) -> bool {
        !self.width.is_null()
    }

    fn get_x_shift(&self) -> f64 {
        self.x_shift
    }

    fn has_x_shift(&self) -> bool {
        if self.x_shift.is_nan() {
            return false;
        }
        true
    }

    fn get_z_shift(&self) -> &ConstEtomoNumber {
        &self.z_shift
    }

    fn has_z_shift(&self) -> bool {
        !self.z_shift.is_null()
    }

    fn is_super_sample_factor_set(&self) -> bool {
        !self.super_sample_factor.is_null()
    }

    fn get_super_sample_factor(&self) -> String {
        self.super_sample_factor.to_string()
    }

    fn is_expand_input_lines_set(&self) -> bool {
        self.expand_input_lines.is()
    }

    fn get_idx_slice_start(&self) -> i32 {
        self.idx_slice_start
    }

    fn get_idx_slice_stop(&self) -> i32 {
        self.idx_slice_stop
    }

    fn has_slice(&self) -> bool {
        if self.idx_slice_stop == i32::MIN {
            return false;
        }
        true
    }

    fn get_tilt_angle_offset(&self) -> &ConstEtomoNumber {
        &self.tilt_angle_offset
    }

    fn has_tilt_angle_offset(&self) -> bool {
        !self.tilt_angle_offset.is_null()
    }

    fn get_scale_coeff(&self) -> f64 {
        self.scale_coeff
    }

    fn get_scale_f_level(&self) -> f64 {
        self.scale_f_level
    }

    fn has_scale(&self) -> bool {
        if self.scale_f_level.is_nan() {
            return false;
        }
        true
    }

    fn get_full_image_x(&self) -> i32 {
        self.full_image_x
    }

    fn get_hamming_like_filter(&self) -> String {
        self.hamming_like_filter.to_string()
    }

    fn is_hamming_like_filter(&self) -> bool {
        !self.hamming_like_filter.is_null()
    }

    fn get_fake_sirt_iterations(&self) -> String {
        self.fake_sirt_iterations.to_string()
    }

    fn is_fake_sirt_iterations(&self) -> bool {
        !self.fake_sirt_iterations.is_null()
    }

    fn get_exact_filter_size(&self) -> String {
        self.exact_filter_size.to_string()
    }

    fn is_exact_filter_size(&self) -> bool {
        !self.exact_filter_size.is_null()
    }

    fn has_z_factor_file_name(&self) -> bool {
        if self.z_factor_file_name.lock().unwrap().equals(Some("")) {
            return false;
        }
        true
    }
}

impl Command for TiltParam {
    fn get_axis_id(&self) -> AxisID {
        self.axis_id
    }

    fn get_command_name(&self) -> Option<String> {
        Some(self.process_name.to_string())
    }

    fn get_process_name(&self) -> Option<ProcessName> {
        Some(self.process_name)
    }

    fn get_command(&self) -> Option<String> {
        Some(self.process_name.get_comscript(self.axis_id))
    }

    fn get_command_line(&self) -> Option<String> {
        self.get_command()
    }

    fn get_command_array(&self) -> Option<Vec<String>> {
        Some(self.process_name.get_comscript_array(self.axis_id))
    }

    fn get_command_mode(&self) -> Option<&dyn CommandMode> {
        Some(&self.command_mode)
    }

    fn is_message_reporter(&self) -> bool {
        self.is_use_gpu()
    }

    fn get_subcommand_details(&self) -> Option<&dyn CommandDetails> {
        None
    }

    fn get_subcommand_process_name(&self) -> Option<String> {
        None
    }

    /// Java `new File(manager.getPropertyUserDir(), outputFile.toString())`.
    fn get_command_output_file(&self) -> Option<PathBuf> {
        let output_file = self.output_file.to_string();
        Some(PathBuf::from(match self.manager.get_property_user_dir() {
            None => output_file,
            Some(dir) => java_io_file_new(&dir, &output_file),
        }))
    }

    fn get_command_input_file(&self) -> Option<PathBuf> {
        None
    }

    /// Java `getOutputImageFileType`, deprecated 3/15/2019.  The source reads the axis
    /// type into a local it never uses; the `getBaseMetaData()` call has no effect.
    fn get_output_image_file_type(&self) -> Option<Arc<FileType>> {
        let _axis_type = self
            .manager
            .get_base_meta_data()
            .map(|meta_data| meta_data.base().get_axis_type());
        if self.command_mode == Mode::Sample {
            return None;
        }
        if self.command_mode == Mode::Tilt {
            return Some(file_type::CLASS.tilt_output.clone());
        }
        if self.command_mode == Mode::Tilt3dFind {
            return Some(file_type::CLASS.tilt_3d_find_output.clone());
        }
        if self.command_mode == Mode::TrialTilt {
            return None;
        }
        if self.command_mode == Mode::Whole {
            // Handled by NewstParam and BlendmontParam
            return None;
        }
        None
    }

    fn get_output_image_file_key(&self) -> Option<FileKey> {
        if self.command_mode == Mode::Sample {
            return Some((*file_key::POSITIONING_SAMPLE).clone());
        }
        if self.command_mode == Mode::Tilt {
            let key: &FileKey = &file_type::CLASS.tilt_output;
            return Some(key.clone());
        }
        if self.command_mode == Mode::Tilt3dFind {
            let key: &FileKey = &file_type::CLASS.tilt_3d_find_output;
            return Some(key.clone());
        }
        if self.command_mode == Mode::TrialTilt {
            return None;
        }
        if self.command_mode == Mode::Whole {
            // Handled by NewstParam and BlendmontParam
            return None;
        }
        None
    }

    /// Java `getOutputImageFileType2`, deprecated 3/15/2019.
    fn get_output_image_file_type2(&self) -> Option<Arc<FileType>> {
        None
    }

    fn get_output_image_file_key2(&self) -> Option<FileKey> {
        None
    }

    /// Java `command instanceof ProcessDetails`: this is a `CommandDetails`.
    fn get_process_details(&self) -> Option<&dyn ProcessDetails> {
        Some(self)
    }
}

impl Loggable for TiltParam {
    /// Java `getName`.
    fn get_name(&self) -> String {
        self.process_name.to_string()
    }

    /// Java `getLogMessage`, which returns null; there is no message.
    fn get_log_message(&self) -> Result<Vec<Option<String>>, LoggableException> {
        Ok(Vec::new())
    }
}

/// Java `ProcessDetails`.  A field the source does not handle throws
/// `IllegalArgumentException("field=" + field)`, an uncaught crash; here it is
/// `None` (fixed in translation).
impl ProcessDetails for TiltParam {
    fn get_boolean_value(&self, field: &dyn FieldInterface) -> Option<bool> {
        match field_interface::as_field::<Field>(field) {
            Some(Field::Fiducialess) => Some(self.fiducialess.is()),
            Some(Field::AdjustOrigin) => Some(self.adjust_origin.is()),
            // throw new IllegalArgumentException("field=" + field)
            _ => None,
        }
    }

    fn get_string_array(&self, _field: &dyn FieldInterface) -> Option<Vec<String>> {
        None
    }

    fn get_string(&self, _field: &dyn FieldInterface) -> Option<String> {
        None
    }

    fn get_double_value(&self, field: &dyn FieldInterface) -> Option<f64> {
        match field_interface::as_field::<Field>(field) {
            Some(Field::XAxisTilt) => Some(self.x_axis_tilt.get_double()),
            Some(Field::ZShift) => Some(self.z_shift.get_double()),
            Some(Field::TiltAngleOffset) => Some(self.tilt_angle_offset.get_double()),
            // throw new IllegalArgumentException("field=" + field)
            _ => None,
        }
    }

    fn get_etomo_number(&self, _field: &dyn FieldInterface) -> Option<ConstEtomoNumber> {
        None
    }

    fn get_int_key_list(&self, _field: &dyn FieldInterface) -> Option<Vec<(i32, String)>> {
        None
    }

    fn get_int_value(&self, _field: &dyn FieldInterface) -> Option<i32> {
        None
    }

    fn get_iterator_element_list(&self, _field: &dyn FieldInterface) -> Option<Vec<i32>> {
        None
    }

    fn get_hashtable(&self, _field: &dyn FieldInterface) -> Option<Vec<(String, String)>> {
        None
    }
}

impl CommandParam for TiltParam {
    /// Java `parseComScriptCommand`.  Get the parameters from the ComScriptCommand.
    fn parse_com_script_command(
        &mut self,
        script_command: &ComScriptCommand,
    ) -> Result<(), ParseComScriptError> {
        if !script_command.is_keyword_value_pairs() {
            // tilt.com doesn't contain -StandardInput - use old parse method
            return self.backward_compatible_parse_com_script_command(script_command);
        }
        let number_format = |message: String| ParseComScriptError::NumberFormat(message);
        let out_of_bounds = |index: usize, length: usize| {
            ParseComScriptError::NumberFormat(format!(
                "Index {index} out of bounds for length {length}"
            ))
        };
        let separator = Regex::new(r"\s*,\s*|\s+").unwrap();
        self.input_file.parse(script_command)?;
        self.output_file.parse(script_command)?;
        self.image_binned.parse(script_command)?;
        self.exclude_list.parse(script_command)?;
        self.exclude_list2.parse(script_command)?;
        self.temp_full_image.lock().unwrap().parse(script_command)?;
        if !self.temp_full_image.lock().unwrap().is_empty() {
            let params = java_lang_string_split_limit(
                &self.temp_full_image.lock().unwrap().to_string(),
                &separator,
                2,
            );
            self.full_image_x = java_lang_integer_parse_int(&params[0]).map_err(number_format)?;
            self.full_image_y = java_lang_integer_parse_int(
                params
                    .get(1)
                    .ok_or_else(|| out_of_bounds(1, params.len()))?,
            )
            .map_err(number_format)?;
        }
        self.local_align_file
            .lock()
            .unwrap()
            .parse(script_command)?;
        self.log_offset.parse(script_command)?;
        self.mode.parse(script_command)?;
        self.temp_offset.lock().unwrap().parse(script_command)?;
        if !self.temp_offset.lock().unwrap().is_empty() {
            let params = java_lang_string_split_limit(
                &self.temp_offset.lock().unwrap().to_string(),
                &separator,
                2,
            );
            self.tilt_angle_offset.set_string(Some(params[0].as_str()));
            if params.len() > 1 {
                self.tilt_axis_offset =
                    java_lang_double_value_of(&params[1]).map_err(number_format)?;
            }
        }
        self.parallel.parse(script_command)?;
        self.perpendicular.parse(script_command)?;
        self.temp_radial.lock().unwrap().parse(script_command)?;
        if !self.temp_radial.lock().unwrap().is_empty() {
            let params = java_lang_string_split_limit(
                &self.temp_radial.lock().unwrap().to_string(),
                &separator,
                2,
            );
            self.radial_bandwidth.set_string(Some(params[0].as_str()));
            self.radial_falloff.set_string(Some(
                params
                    .get(1)
                    .ok_or_else(|| out_of_bounds(1, params.len()))?
                    .as_str(),
            ));
        }
        self.temp_scale.lock().unwrap().parse(script_command)?;
        if !self.temp_scale.lock().unwrap().is_empty() {
            let params = java_lang_string_split_limit(
                &self.temp_scale.lock().unwrap().to_string(),
                &separator,
                2,
            );
            self.scale_f_level = java_lang_double_value_of(&params[0]).map_err(number_format)?;
            self.scale_coeff = java_lang_double_value_of(
                params
                    .get(1)
                    .ok_or_else(|| out_of_bounds(1, params.len()))?,
            )
            .map_err(number_format)?;
        }
        self.temp_shift.lock().unwrap().parse(script_command)?;
        if !self.temp_shift.lock().unwrap().is_empty() {
            let params = java_lang_string_split_limit(
                &self.temp_shift.lock().unwrap().to_string(),
                &separator,
                2,
            );
            self.x_shift = java_lang_double_value_of(&params[0]).map_err(number_format)?;
            if params.len() > 1 {
                self.z_shift.set_string(Some(params[1].as_str()));
            }
        }
        self.temp_slice.lock().unwrap().parse(script_command)?;
        if !self.temp_slice.lock().unwrap().is_empty() {
            let params = java_lang_string_split_limit(
                &self.temp_slice.lock().unwrap().to_string(),
                &separator,
                3,
            );
            self.idx_slice_start =
                java_lang_integer_parse_int(&params[0]).map_err(number_format)?;
            self.idx_slice_stop = java_lang_integer_parse_int(
                params
                    .get(1)
                    .ok_or_else(|| out_of_bounds(1, params.len()))?,
            )
            .map_err(number_format)?;
            // increment is being ignored
        }
        self.temp_subset_start
            .lock()
            .unwrap()
            .parse(script_command)?;
        if !self.temp_subset_start.lock().unwrap().is_empty() {
            let params = java_lang_string_split_limit(
                &self.temp_subset_start.lock().unwrap().to_string(),
                &separator,
                2,
            );
            self.idx_x_subset_start =
                java_lang_integer_parse_int(&params[0]).map_err(number_format)?;
            self.idx_y_subset_start = java_lang_integer_parse_int(
                params
                    .get(1)
                    .ok_or_else(|| out_of_bounds(1, params.len()))?,
            )
            .map_err(number_format)?;
        }
        self.thickness.parse(script_command)?;
        self.tilt_file.parse(script_command)?;
        self.width.parse(script_command)?;
        self.x_axis_tilt.parse(script_command)?;
        self.z_factor_file_name
            .lock()
            .unwrap()
            .parse(script_command)?;
        if !self.z_factor_file_name.lock().unwrap().is_empty() {
            self.use_z_factors = true;
        }
        self.x_tilt_file.lock().unwrap().parse(script_command)?;
        self.adjust_origin.parse(script_command)?;
        self.project_model.parse(script_command)?;
        self.use_gpu.parse(script_command)?;
        self.action_if_gpu_fails.parse(script_command)?;
        if self.action_if_gpu_fails.is_empty() {
            self.action_if_gpu_fails
                .set(Some(shared_constants::ACTION_IF_GPU_FAILS_DEFAULT));
        }
        self.hamming_like_filter.parse(script_command)?;
        self.fake_sirt_iterations.parse(script_command)?;
        self.exact_filter_size.parse(script_command)?;
        self.falloff_is_true_sigma.parse(script_command)?;
        self.super_sample_factor.parse(script_command)?;
        self.expand_input_lines.parse(script_command)?;
        self.loaded_from_file = true;
        Ok(())
    }

    /// Java `updateComScriptCommand`.  Update the script command with the current
    /// values of this TiltParam object.
    ///
    /// The fields the source assigns here (the temporaries, `localAlignFile`,
    /// `zFactorFileName` and `xTiltFile`) are behind a `Mutex`, so the assignments
    /// persist exactly as in Java although the trait method takes `&self`.
    fn update_com_script_command(
        &self,
        script_command: &mut ComScriptCommand,
    ) -> Result<(), BadComScriptException> {
        // Switch to keyword/value pairs
        script_command.use_keyword_value();
        // get rid of the DONE parameter from the old tilt.com
        self.done.update_com_script(script_command);
        self.input_file.update_com_script(script_command);
        self.output_file.update_com_script(script_command);
        self.image_binned.update_com_script(script_command);
        self.exclude_list.update_com_script(script_command)?;
        self.exclude_list2.update_com_script(script_command)?;
        if self.full_image_x > i32::MIN {
            self.temp_full_image.lock().unwrap().set(Some(&*format!(
                "{} {}",
                self.full_image_x, self.full_image_y
            )));
        } else {
            self.temp_full_image.lock().unwrap().reset();
        }
        self.temp_full_image
            .lock()
            .unwrap()
            .update_com_script(script_command);
        if self.fiducialess.is() {
            self.local_align_file.lock().unwrap().reset();
        }
        self.local_align_file
            .lock()
            .unwrap()
            .update_com_script(script_command);
        self.log_offset.update_com_script(script_command);
        self.mode.update_com_script(script_command);
        if !self.tilt_angle_offset.is_null() {
            let mut arg = self.tilt_angle_offset.to_string();
            if !self.tilt_axis_offset.is_nan() {
                arg += &format!(" {}", java_lang_double_to_string(self.tilt_axis_offset));
            }
            self.temp_offset.lock().unwrap().set(Some(arg.as_str()));
        } else {
            self.temp_offset.lock().unwrap().reset();
        }
        self.temp_offset
            .lock()
            .unwrap()
            .update_com_script(script_command);
        self.parallel.update_com_script(script_command);
        self.perpendicular.update_com_script(script_command);
        if !self.radial_bandwidth.is_null() {
            self.temp_radial.lock().unwrap().set(Some(&*format!(
                "{} {}",
                self.radial_bandwidth, self.radial_falloff
            )));
        } else {
            self.temp_radial.lock().unwrap().reset();
        }
        self.temp_radial
            .lock()
            .unwrap()
            .update_com_script(script_command);
        if !self.scale_f_level.is_nan() {
            self.temp_scale.lock().unwrap().set(Some(&*format!(
                "{} {}",
                java_lang_double_to_string(self.scale_f_level),
                java_lang_double_to_string(self.scale_coeff)
            )));
        } else {
            self.temp_scale.lock().unwrap().reset();
        }
        self.temp_scale
            .lock()
            .unwrap()
            .update_com_script(script_command);
        // The source also builds an unused `StringBuffer shiftBuffer`.
        if !self.x_shift.is_nan() || !self.z_shift.is_null() {
            let mut arg = String::new();
            if self.x_shift.is_nan() {
                arg.push_str("0 ");
            } else {
                arg.push_str(&java_lang_double_to_string(self.x_shift));
            }
            if !self.z_shift.is_null() {
                arg.push_str(&format!(" {}", self.z_shift));
            }
            self.temp_shift.lock().unwrap().set(Some(arg.as_str()));
        } else {
            self.temp_shift.lock().unwrap().reset();
        }
        self.temp_shift
            .lock()
            .unwrap()
            .update_com_script(script_command);
        if self.idx_slice_start > i32::MIN {
            let arg = format!("{} {}", self.idx_slice_start, self.idx_slice_stop);
            self.temp_slice.lock().unwrap().set(Some(arg.as_str()));
        } else {
            self.temp_slice.lock().unwrap().reset();
        }
        self.temp_slice
            .lock()
            .unwrap()
            .update_com_script(script_command);
        if self.idx_x_subset_start > i32::MIN {
            self.temp_subset_start.lock().unwrap().set(Some(&*format!(
                "{} {}",
                self.idx_x_subset_start, self.idx_y_subset_start
            )));
        } else {
            self.temp_subset_start.lock().unwrap().reset();
        }
        self.temp_subset_start
            .lock()
            .unwrap()
            .update_com_script(script_command);
        self.thickness.update_com_script(script_command);
        self.tilt_file.update_com_script(script_command);
        self.width.update_com_script(script_command);
        self.x_axis_tilt.update_com_script(script_command);
        if self.use_z_factors && !self.fiducialess.is() {
            if self.z_factor_file_name.lock().unwrap().is_empty() {
                self.z_factor_file_name.lock().unwrap().set(Some(
                    &*const_tiltalign_param::ConstTiltalignParam::get_output_z_factor_file_name(
                        self.dataset_name.as_deref(),
                        self.axis_id,
                    ),
                ));
            }
        } else {
            self.z_factor_file_name.lock().unwrap().reset();
        }
        self.z_factor_file_name
            .lock()
            .unwrap()
            .update_com_script(script_command);
        // A fiducialess align means that tilt should not use the xtilt file.
        if !self.fiducialess.is() {
            // backwards compatibility: if xTiltFile is empty, set the default xtilt file
            // name
            if self.x_tilt_file.lock().unwrap().is_empty() {
                self.x_tilt_file
                    .lock()
                    .unwrap()
                    .set(Some(&*dataset_files::get_x_tilt_file_name(
                        self.manager,
                        Some(self.axis_id),
                    )));
            }
            // use xtilt file if the file exists
            // This is backwards compatibility issue since the only good reason for the
            // file not to exist is that the state comes from an earlier version.
            let x_tilt_file = self.x_tilt_file.lock().unwrap().to_string();
            let path = match self.manager.get_property_user_dir() {
                None => x_tilt_file,
                Some(dir) => java_io_file_new(&dir, &x_tilt_file),
            };
            if !std::path::Path::new(&path).exists() {
                self.x_tilt_file.lock().unwrap().reset();
            }
        } else {
            self.x_tilt_file.lock().unwrap().reset();
        }
        self.x_tilt_file
            .lock()
            .unwrap()
            .update_com_script(script_command);
        self.adjust_origin.update_com_script(script_command);
        self.project_model.update_com_script(script_command);
        self.use_gpu.update_com_script(script_command);
        self.action_if_gpu_fails.update_com_script(script_command);
        self.hamming_like_filter.update_com_script(script_command);
        self.fake_sirt_iterations.update_com_script(script_command);
        self.exact_filter_size.update_com_script(script_command);
        self.super_sample_factor.update_com_script(script_command);
        self.expand_input_lines.update_com_script(script_command);
        Ok(())
    }

    /// Java `initializeDefaults`.
    fn initialize_defaults(&mut self) {}
}
