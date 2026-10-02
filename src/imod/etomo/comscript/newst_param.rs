//! `IMOD/Etomo/src/etomo/comscript/NewstParam.java`.
//!
//! Parameters for newstack (and colornewst) in prenewst.com, newst.com and their
//! variants.
//!
// TODO(unit): needs etomo/type/StringParameter.java - `outputFile`, `fileOfInputs`,
// `fileOfOutputs`, `transformFile`, `useTransformLines`, `distortionField`,
// `gradientFile` (r#type::string_parameter::StringParameter).
// TODO(unit): needs etomo/comscript/Utilities.java - `is90DegreeImageRotation`
// (comscript::utilities::is_90_degree_image_rotation(f64) -> bool).
// TODO(unit): needs FortranInputString `validateAndSet(ComScriptCommand)` and
// `updateScriptParameter(ComScriptCommand)` (TODOs in fortran_input_string.rs) -
// assumed as `validate_and_set_com_script(&mut self, &ComScriptCommand) ->
// Result<(), ParseComScriptError>` and `update_script_parameter(&self, &mut
// ComScriptCommand)`.
//
// Kept as in the source: `updateComScriptCommand` writes each `InputFile`,
// `SectionsToRead` and `NumberToOutput` entry with `setValue`, which replaces the
// existing keyword, so only the last entry of each list reaches the com file.

use std::path::PathBuf;
use std::sync::Arc;

use super::bad_com_script_exception::BadComScriptException;
use super::com_script_command::ComScriptCommand;
use super::command::Command;
use super::command_details::CommandDetails;
use super::command_mode::CommandMode;
use super::command_param::{CommandParam, ParseComScriptError};
use super::const_newst_param::ConstNewstParam;
use super::field_interface::{self, FieldInterface};
use super::fortran_input_string::{self, FortranInputString};
use super::fortran_input_syntax_exception::FortranInputSyntaxException;
use super::invalid_parameter_exception::InvalidParameterException;
use super::process_details::ProcessDetails;
use super::utilities;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::const_etomo_number::{ConstEtomoNumber, Number, Type};
use crate::imod::etomo::r#type::etomo_boolean2::EtomoBoolean2;
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::r#type::file_key::FileKey;
use crate::imod::etomo::r#type::file_type::{self, FileType};
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::script_parameter::ScriptParameter;
use crate::imod::etomo::r#type::string_parameter::StringParameter;
use crate::imod::etomo::ui::swing::log_interface::{Loggable, LoggableException};
use crate::imod::etomo::ui::swing::ui_harness;
use crate::imod::etomo::util::mrc_header::MRCHeader;

const INPUT_FILE_KEY: &str = "InputFile";
const SECTIONS_TO_READ_KEY: &str = "SectionsToRead";
const NUMBER_TO_OUTPUT_KEY: &str = "NumberToOutput";
pub const SIZE_TO_OUTPUT_IN_X_AND_Y: &str = "SizeToOutputInXandY";
pub const IMAGES_ARE_BINNED_KEY: &str = "ImagesAreBinned";
pub const DISTORTION_FIELD_KEY: &str = "DistortionField";
pub const OFFSETS_IN_X_AND_Y_KEY: &str = "OffsetsInXandY";
pub const BIN_BY_FACTOR_KEY: &str = "BinByFactor";
pub const FILL_VALUE_KEY: &str = "FillValue";
pub const MODE_TO_OUTPUT_KEY: &str = "ModeToOutput";
// data mode
pub const DATA_MODE_OPTION: &str = "-mo";
pub const DATA_MODE_DEFAULT: i32 = i32::MIN;
pub const DATA_MODE_BYTE: i32 = 0;
// float densities
pub const FLOAT_DENSITIES_OPTION: &str = "-fl";
pub const FLOAT_DENSITIES_DEFAULT: i32 = i32::MIN;
pub const FLOAT_DENSITIES_MEAN: i32 = 2;
const COMMAND_FILE_EXTENSION: &str = ".com";
const DEFAULT_ANTIALIAS_FILTER: i32 = -1;

/// Java nested `NewstParam.Field implements FieldInterface`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Field {
    /// Java `FIDUCIALESS_ALIGNMENT`.
    FiducialessAlignment,
    /// Java `BINNING`.
    Binning,
    /// Java `USE_LINEAR_INTERPOLATION`.
    UseLinearInterpolation,
    /// Java `USER_SIZE_TO_OUTPUT_IN_X_AND_Y`.
    UserSizeToOutputInXAndY,
    /// Java `IMAGE_ROTATION`.
    ImageRotation,
}

impl FieldInterface for Field {}

/// Java nested `NewstParam.Mode implements CommandMode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Java `PREALIGNED`, "Prealigned".
    Prealigned,
    /// Java `WHOLE_TOMOGRAM_SAMPLE`, "WholeTomogramSample".
    WholeTomogramSample,
    /// Java `FULL_ALIGNED_STACK`, "FullAlignedStack".
    FullAlignedStack,
}

impl std::fmt::Display for Mode {
    /// Java `toString()`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Mode::Prealigned => "Prealigned",
            Mode::WholeTomogramSample => "WholeTomogramSample",
            Mode::FullAlignedStack => "FullAlignedStack",
        })
    }
}

impl CommandMode for Mode {}

/// The three exceptions Java's `setSizeToOutputInXandY` declares:
/// `FortranInputSyntaxException`, and `etomo.util.InvalidParameterException` /
/// `IOException` from `MRCHeader.read`, which `MRCHeader::read_with_manager` reports as
/// one `Err(String)`.
#[derive(Debug)]
pub enum SetSizeToOutputInXandYError {
    FortranInputSyntax(FortranInputSyntaxException),
    HeaderRead(String),
}

impl From<FortranInputSyntaxException> for SetSizeToOutputInXandYError {
    fn from(e: FortranInputSyntaxException) -> SetSizeToOutputInXandYError {
        SetSizeToOutputInXandYError::FortranInputSyntax(e)
    }
}

impl std::fmt::Display for SetSizeToOutputInXandYError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SetSizeToOutputInXandYError::FortranInputSyntax(e) => write!(f, "{e}"),
            SetSizeToOutputInXandYError::HeaderRead(e) => write!(f, "{e}"),
        }
    }
}

/// Java final `NewstParam implements ConstNewstParam, CommandParam`.
pub struct NewstParam {
    input_file: Vec<String>,
    output_file: StringParameter,
    file_of_inputs: StringParameter,
    file_of_outputs: StringParameter,
    sections_to_read: Vec<String>,
    number_to_output: Vec<String>,
    size_to_output_in_xand_y: FortranInputString,
    mode_to_output: ScriptParameter,
    user_size_to_output_in_xand_y: FortranInputString,
    offsets_in_xand_y: FortranInputString,
    offsets_in_xand_y_extra_entries: Vec<FortranInputString>,
    apply_offsets_first: EtomoBoolean2,
    transform_file: StringParameter,
    use_transform_lines: StringParameter,
    rotate_by_angle: ScriptParameter,
    expand_by_factor: ScriptParameter,
    bin_by_factor: ScriptParameter,
    linear_interpolation: EtomoBoolean2,
    float_densities: ScriptParameter,
    contrast_black_white: FortranInputString,
    scale_min_and_max: FortranInputString,
    distortion_field: StringParameter,
    images_are_binned: ScriptParameter,
    test_limits: FortranInputString,
    gradient_file: StringParameter,
    /// @version 3.10.  Script is from an earlier version if false.
    adjust_origin: EtomoBoolean2,
    taper_at_fill: FortranInputString,
    fill_value: ScriptParameter,
    image_rotation: EtomoNumber,
    antialias_filter: ScriptParameter,

    // colornewst only parameters
    cntiff: EtomoBoolean2,
    cntempdir: ScriptParameter,
    cnmaxtemp: EtomoNumber,
    cnverbose: EtomoBoolean2,

    axis_id: AxisID,
    manager: &'static dyn BaseManager,
    use_color_newst: bool,

    /// Set when outputFile is set from the dialog, otherwise set to null when
    /// outputFile changed.
    output_file_type: Option<Arc<FileType>>,
    process_name: ProcessName,
    validate: bool,
    output_image_file_key: Option<FileKey>,

    fiducialess_alignment: bool,
    mode: Option<Mode>,
}

impl NewstParam {
    /// Java private `NewstParam(BaseManager, AxisID, boolean)`.
    fn new(
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        use_color_newst: bool,
    ) -> NewstParam {
        let mut param = NewstParam {
            input_file: Vec::new(),
            output_file: StringParameter::new("OutputFile"),
            file_of_inputs: StringParameter::new("FileOfInputs"),
            file_of_outputs: StringParameter::new("FileOfOutputs"),
            sections_to_read: Vec::new(),
            number_to_output: Vec::new(),
            size_to_output_in_xand_y: FortranInputString::new_with_key(
                Some("SizeToOutputInXandY"),
                2,
            ),
            mode_to_output: ScriptParameter::new_with_name(MODE_TO_OUTPUT_KEY),
            user_size_to_output_in_xand_y: FortranInputString::new(2),
            offsets_in_xand_y: FortranInputString::new_with_key(Some(OFFSETS_IN_X_AND_Y_KEY), 2),
            offsets_in_xand_y_extra_entries: Vec::new(),
            apply_offsets_first: EtomoBoolean2::new_with_name("ApplyOffsetsFirst"),
            transform_file: StringParameter::new("TransformFile"),
            use_transform_lines: StringParameter::new("UseTransformLines"),
            rotate_by_angle: ScriptParameter::new_with_type_and_name(Type::Double, "RotateByAngle"),
            expand_by_factor: ScriptParameter::new_with_type_and_name(
                Type::Double,
                "ExpandByFactor",
            ),
            bin_by_factor: ScriptParameter::new_with_name("BinByFactor"),
            linear_interpolation: EtomoBoolean2::new_with_name("LinearInterpolation"),
            float_densities: ScriptParameter::new_with_name("FloatDensities"),
            contrast_black_white: FortranInputString::new_with_key(Some("ContrastBlackWhite"), 2),
            scale_min_and_max: FortranInputString::new_with_key(Some("ScaleMinAndMax"), 2),
            distortion_field: StringParameter::new("DistortionField"),
            images_are_binned: ScriptParameter::new_with_type_and_name(
                Type::Double,
                "ImagesAreBinned",
            ),
            test_limits: FortranInputString::new_with_key(Some("TestLimits"), 2),
            gradient_file: StringParameter::new("GradientFile"),
            adjust_origin: EtomoBoolean2::new_with_name("AdjustOrigin"),
            taper_at_fill: FortranInputString::new_with_key(Some("TaperAtFill"), 2),
            fill_value: ScriptParameter::new_with_type_and_name(Type::Double, "FillValue"),
            image_rotation: EtomoNumber::new_with_type(Some(Type::Double)),
            antialias_filter: ScriptParameter::new_with_name("AntialiasFilter"),
            cntiff: EtomoBoolean2::new_with_name("-cntiff"),
            cntempdir: ScriptParameter::new_with_name("-cntempdir"),
            cnmaxtemp: EtomoNumber::new_with_type_and_name(Type::Double, "-cnmaxtemp"),
            cnverbose: EtomoBoolean2::new_with_name("-cnverbose"),
            axis_id,
            manager,
            use_color_newst,
            output_file_type: None,
            process_name: ProcessName::NEWST,
            validate: false,
            output_image_file_key: None,
            fiducialess_alignment: false,
            mode: None,
        };
        param.size_to_output_in_xand_y.set_integer_type(true);
        param.user_size_to_output_in_xand_y.set_integer_type(true);
        param.bin_by_factor.set_display_value_int(1);
        param.contrast_black_white.set_integer_type(true);
        param.test_limits.set_integer_type(true);
        param.taper_at_fill.set_integer_type(true);
        param.reset();
        param
    }

    /// Java `getInstance(BaseManager, AxisID)`.
    pub fn get_instance(manager: &'static dyn BaseManager, axis_id: AxisID) -> NewstParam {
        NewstParam::new(manager, axis_id, false)
    }

    /// Java `getColorInstance(BaseManager, AxisID)`.
    pub fn get_color_instance(manager: &'static dyn BaseManager, axis_id: AxisID) -> NewstParam {
        NewstParam::new(manager, axis_id, true)
    }

    /// Java `setValidate(boolean)`.
    pub fn set_validate(&mut self, validate: bool) {
        self.validate = validate;
    }

    /// Java private `reset()`.
    fn reset(&mut self) {
        self.validate = false;
        self.input_file.clear();
        self.output_file.reset();
        self.output_file_type = None;
        self.file_of_inputs.reset();
        self.file_of_outputs.reset();
        self.sections_to_read.clear();
        self.number_to_output.clear();
        self.size_to_output_in_xand_y.reset();
        self.user_size_to_output_in_xand_y.reset();
        self.mode_to_output.reset();
        self.offsets_in_xand_y.reset();
        self.offsets_in_xand_y_extra_entries.clear();
        self.apply_offsets_first.reset();
        self.transform_file.reset();
        self.use_transform_lines.reset();
        self.rotate_by_angle.reset();
        self.expand_by_factor.reset();
        self.bin_by_factor.reset();
        self.linear_interpolation.reset();
        self.float_densities.reset();
        self.contrast_black_white.reset();
        self.scale_min_and_max.reset();
        self.distortion_field.reset();
        self.images_are_binned.reset();
        self.test_limits.reset();
        self.fiducialess_alignment = false;
        self.gradient_file.reset();
        self.adjust_origin.reset();
        self.taper_at_fill.reset();
        self.image_rotation.reset();
        self.fill_value.reset();
        self.antialias_filter.reset();
    }

    /// Java `setBinByFactor(Number)`.
    pub fn set_bin_by_factor(&mut self, input: Option<Number>) {
        match input {
            None => {
                self.bin_by_factor.set_int(1);
            }
            Some(input) => {
                self.bin_by_factor.set_number(Some(input));
            }
        }
    }

    /// Java `setCnverbose(boolean)`.
    pub fn set_cnverbose(&mut self, input: bool) {
        self.cnverbose.set_boolean(input);
    }

    /// Java `resetDistortionField()`.
    pub fn reset_distortion_field(&mut self) {
        self.distortion_field.reset();
    }

    /// Java `setDistortionField(String)`.
    pub fn set_distortion_field(&mut self, distortion_field: Option<&str>) {
        self.distortion_field.set(distortion_field);
    }

    /// Java `setFloatDensities(int)`.
    pub fn set_float_densities(&mut self, float_densities: i32) {
        self.float_densities.set_int(float_densities);
    }

    /// Java `setFiducialessAlignment(boolean)`.
    pub fn set_fiducialess_alignment(&mut self, fiducialess_alignment: bool) {
        self.fiducialess_alignment = fiducialess_alignment;
    }

    /// Java `setFillValue(int)`.
    pub fn set_fill_value(&mut self, input: i32) {
        self.fill_value.set_int(input);
    }

    /// Java `setImagesAreBinned(Number)`.
    pub fn set_images_are_binned(&mut self, input: Option<Number>) {
        match input {
            None => {
                self.images_are_binned.reset();
            }
            Some(input) => {
                self.images_are_binned.set_int(input.int_value());
            }
        }
    }

    /// Java `resetInputFile()`.
    pub fn reset_input_file(&mut self) {
        self.input_file.clear();
    }

    /// Java `setInputFile(String)`.
    pub fn set_input_file(&mut self, input: Option<&str>) {
        self.input_file.clear();
        let input = match input {
            None => return,
            Some(input) => input,
        };
        self.input_file.push(input.to_string());
    }

    /// Java `setLinearInterpolation(boolean)`.
    pub fn set_linear_interpolation(&mut self, linear_interpolation: bool) {
        self.linear_interpolation.set_boolean(linear_interpolation);
    }

    /// Java `setModeToOutput(int)`.
    pub fn set_mode_to_output(&mut self, mode_to_output: i32) {
        self.mode_to_output.set_int(mode_to_output);
    }

    /// Java `setOffsetsInXandY(ConstEtomoNumber[])`.
    pub fn set_offsets_in_xand_y(&mut self, pair: Option<&[ConstEtomoNumber]>) {
        match pair {
            None => {
                self.offsets_in_xand_y.reset();
            }
            Some(pair) => {
                for i in 0..pair.len() {
                    self.offsets_in_xand_y
                        .set_index_const_etomo_number(i as i32, &pair[i]);
                }
            }
        }
    }

    /// Java `setOutputFile(FileType, String, AxisType)`.  Set the output file before
    /// the manager is set up.
    ///
    /// NewstParam.java:714-716 dereferences `manager.getBaseMetaData()` without a null
    /// check.  Fixed in translation: with no metadata the style and extension are
    /// passed as null, which `deriveFileName` accepts.
    pub fn set_output_file_derived(
        &mut self,
        file_type: &Arc<FileType>,
        root_name: Option<&str>,
        axis_type: Option<AxisType>,
    ) {
        let meta_data = self.manager.get_base_meta_data();
        let image_filename_style =
            meta_data.map(|meta_data| meta_data.base().get_image_filename_style());
        let raw_image_stack_extension =
            meta_data.and_then(|meta_data| meta_data.get_raw_image_stack_extension());
        self.output_file.set(
            file_type
                .derive_file_name(
                    root_name,
                    axis_type,
                    Some(self.axis_id),
                    image_filename_style,
                    raw_image_stack_extension,
                )
                .as_deref(),
        );
        self.output_file_type = Some(Arc::clone(file_type));
    }

    /// Java `setOutputFile(FileType)`.
    pub fn set_output_file(&mut self, file_type: &Arc<FileType>) {
        self.output_file.set(
            file_type
                .get_file_name(Some(self.manager), Some(self.axis_id))
                .as_deref(),
        );
        self.output_file_type = Some(Arc::clone(file_type));
    }

    /// Java `setAdjustOrigin(boolean)`.
    pub fn set_adjust_origin(&mut self, input: bool) {
        self.adjust_origin.set_boolean(input);
    }

    /// Java `setAntialiasFilter(boolean)`.
    pub fn set_antialias_filter(&mut self, input: bool) {
        if input {
            self.antialias_filter.set_int(DEFAULT_ANTIALIAS_FILTER);
        } else {
            self.antialias_filter.reset();
        }
    }

    /// Java `setAntialiasFilterValue(ConstEtomoNumber)`.
    pub fn set_antialias_filter_value(&mut self, input: &ConstEtomoNumber) {
        if !input.is_null() {
            self.antialias_filter.set_const_etomo_number(Some(input));
        }
    }

    /// Java `resetSizeToOutputInXandY()`.  The declared
    /// `FortranInputSyntaxException` is never thrown.
    pub fn reset_size_to_output_in_xand_y(&mut self) -> Result<(), FortranInputSyntaxException> {
        self.size_to_output_in_xand_y.reset();
        self.user_size_to_output_in_xand_y.reset();
        self.image_rotation.reset();
        Ok(())
    }

    /// Java `setSizeToOutputInXandY(String, String, int, double, String)`.  Calls
    /// setSizeToOutputInXandY.
    pub fn set_size_to_output_in_xand_y_xy(
        &mut self,
        user_size_x: &str,
        user_size_y: &str,
        binning: i32,
        image_rotation: f64,
        description: Option<&str>,
    ) -> Result<bool, SetSizeToOutputInXandYError> {
        let mut user_size = String::new();
        // make sure an empty string really causes sizeToOutputInXandY to be empty.
        // (`matches("\\s*")`)
        let is_white = |c: char| matches!(c, ' ' | '\t' | '\n' | '\x0B' | '\x0C' | '\r');
        if !user_size_x.chars().all(is_white) || !user_size_y.chars().all(is_white) {
            user_size = format!(
                "{}{}{}",
                user_size_x,
                fortran_input_string::DEFAULT_DIVIDER,
                user_size_y
            );
        }
        self.set_size_to_output_in_xand_y(&user_size, binning, image_rotation, description)
    }

    /// Java `setSizeToOutputInXandY(String, int, double, String)`.
    ///
    /// If the user size is set, then this works even if the tilt axis angle is closer
    /// to 90 degress.  If the user size is not set and the tilt axis angle is closer to
    /// 90 degrees, then use x and y from the raw stack and transpose them.  In either
    /// case, when sizeToOutputInXandY is set always apply binning.
    ///
    /// Will be called with an empty string by Positioning - whole tomogram.
    ///
    /// Save the userSize so it can be stored as a state value.  It may be reused to
    /// run newst_3dfind.com, which may have a different binning.  Also save image
    /// rotation so that fiducialess parameters, which usually come from the dialog,
    /// can be set for newst_3dfind.com.
    pub fn set_size_to_output_in_xand_y(
        &mut self,
        user_size: &str,
        binning: i32,
        image_rotation: f64,
        description: Option<&str>,
    ) -> Result<bool, SetSizeToOutputInXandYError> {
        let mut user_size = user_size.to_string();
        // make sure an empty string really causes sizeToOutputInXandY to be empty.
        if user_size == "" {
            user_size = "/".to_string();
        }
        self.size_to_output_in_xand_y
            .validate_and_set(Some(&user_size))?;
        self.user_size_to_output_in_xand_y
            .validate_and_set(Some(&user_size))?;
        self.image_rotation.set_double(image_rotation);
        // UserSize is empty, check for an angle close to 90 degrees.
        if (self.size_to_output_in_xand_y.is_default() || self.size_to_output_in_xand_y.is_empty())
            && utilities::is_90_degree_image_rotation(image_rotation)
        {
            let header = MRCHeader::get_instance_from_file_type(
                self.manager,
                Some(self.axis_id),
                &file_type::CLASS.raw_stack,
            );
            if let Some(header) = header {
                header
                    .borrow_mut()
                    .read_with_manager(self.manager)
                    .map_err(SetSizeToOutputInXandYError::HeaderRead)?;
                let header = header.borrow();
                // Set y from columns (x)
                self.size_to_output_in_xand_y
                    .set_index_double(1, header.get_n_columns() as f64);
                // Set x from rows (y)
                self.size_to_output_in_xand_y
                    .set_index_double(0, header.get_n_rows() as f64);
            }
        }
        // NewstParam.java:846-847 divides by `binning` unguarded; a binning of 0 is an
        // ArithmeticException.  Fixed in translation: a zero binning is not applied.
        if binning != 1
            && binning != 0
            && !self.size_to_output_in_xand_y.is_default()
            && !self.size_to_output_in_xand_y.is_empty()
        {
            for ixy in 0..2 {
                let i_size = self.size_to_output_in_xand_y.get_int(ixy);
                let mut i_bin_size = i_size.wrapping_div(binning);

                // If the odd/evenness is preserved, newstack increases by 2 when
                // remainder > 1.  Otherwise it increases size by 1 to preserve
                // odd/evenness
                if i_size.wrapping_rem(2) == i_bin_size.wrapping_rem(2) {
                    if i_size.wrapping_rem(binning) > 1 {
                        i_bin_size = i_bin_size.wrapping_add(2);
                    }
                } else {
                    i_bin_size = i_bin_size.wrapping_add(1);
                }
                self.size_to_output_in_xand_y
                    .set_index_double(ixy, i_bin_size as f64);
            }
        }
        if self.validate {
            if let Some(description) = description {
                if !self.user_size_to_output_in_xand_y.is_null_index(0)
                    && self.user_size_to_output_in_xand_y.is_null_index(1)
                {
                    ui_harness::post_message_dialog(
                        Some(self.manager),
                        format!("Two values are required for {description}."),
                        "Entry Error".to_string(),
                        Some(self.axis_id),
                    );
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    /// Java `setTransformFile(String)`.
    pub fn set_transform_file(&mut self, transform_file: Option<&str>) {
        self.transform_file.set(transform_file);
    }

    /// Java `setProcessName(ProcessName)`.
    pub fn set_process_name(&mut self, input: ProcessName) {
        self.process_name = input;
    }

    /// Java `setCommandMode(Mode)`.
    pub fn set_command_mode(&mut self, input: Option<Mode>) {
        self.mode = input;
        if self.mode == Some(Mode::Prealigned) {
            self.output_image_file_key = Some(FileKey::clone(&file_type::CLASS.prealigned_stack));
        }
    }

    /// Java `setOutputImageFileKey(FileKey)`.
    pub fn set_output_image_file_key(&mut self, output_image_file_key: Option<FileKey>) {
        self.output_image_file_key = output_image_file_key;
    }

    /// Java `getCommandFileName(AxisID)`.
    pub fn get_command_file_name(&self, axis_id: AxisID) -> String {
        format!(
            "{}{}{}",
            self.process_name,
            axis_id.get_extension(),
            COMMAND_FILE_EXTENSION
        )
    }
}

impl CommandParam for NewstParam {
    /// Java `parseComScriptCommand(ComScriptCommand)`.  Get the parameters from the
    /// ComScriptCommand containing the newst command and parameters.
    fn parse_com_script_command(
        &mut self,
        script_command: &ComScriptCommand,
    ) -> Result<(), ParseComScriptError> {
        self.reset();
        if script_command.is_keyword_value_pairs() {
            // Read in everything that the backwards compatibility code is reading in,
            // even if it is not used or changed.
            // (`getValues` never returns null elements.)
            let array = script_command.get_values(Some(INPUT_FILE_KEY));
            for i in 0..array.len() {
                self.input_file
                    .push(array[i].clone().unwrap_or_else(|| "null".to_string()));
            }
            self.output_file.parse(script_command)?;
            self.file_of_inputs.parse(script_command)?;
            self.file_of_outputs.parse(script_command)?;
            let array = script_command.get_values(Some(SECTIONS_TO_READ_KEY));
            for i in 0..array.len() {
                self.sections_to_read
                    .push(array[i].clone().unwrap_or_else(|| "null".to_string()));
            }
            let array = script_command.get_values(Some(NUMBER_TO_OUTPUT_KEY));
            for i in 0..array.len() {
                self.number_to_output
                    .push(array[i].clone().unwrap_or_else(|| "null".to_string()));
            }
            self.size_to_output_in_xand_y
                .validate_and_set_com_script(script_command)?;
            self.mode_to_output.parse(script_command)?;
            let array = script_command.get_values(Some(OFFSETS_IN_X_AND_Y_KEY));
            for i in 0..array.len() {
                if i == 0 {
                    self.offsets_in_xand_y
                        .validate_and_set(array[i].as_deref())?;
                } else {
                    let mut fis = FortranInputString::new_with_key(Some(OFFSETS_IN_X_AND_Y_KEY), 2);
                    fis.validate_and_set(array[i].as_deref())?;
                    self.offsets_in_xand_y_extra_entries.push(fis);
                }
            }
            self.apply_offsets_first.parse(script_command)?;
            self.transform_file.parse(script_command)?;
            self.use_transform_lines.parse(script_command)?;
            self.rotate_by_angle.parse(script_command)?;
            self.expand_by_factor.parse(script_command)?;
            self.bin_by_factor.parse(script_command)?;
            self.linear_interpolation.parse(script_command)?;
            self.float_densities.parse(script_command)?;
            self.contrast_black_white
                .validate_and_set_com_script(script_command)?;
            self.scale_min_and_max
                .validate_and_set_com_script(script_command)?;
            self.distortion_field.parse(script_command)?;
            self.images_are_binned.parse(script_command)?;
            self.test_limits
                .validate_and_set_com_script(script_command)?;
            self.gradient_file.parse(script_command)?;
            self.adjust_origin.parse(script_command)?;
            self.taper_at_fill
                .validate_and_set_com_script(script_command)?;
            self.fill_value.parse(script_command)?;
            self.antialias_filter.parse(script_command)?;
        } else {
            // Backwards compatibility
            // A null argument array or element, and an option missing its value
            // (`cmdLineArgs[++i]` past the end), are a NullPointerException /
            // ArrayIndexOutOfBoundsException in the source, which its caller
            // (ComScriptUtil.initialize) catches as a parse failure; they are returned
            // as errors here.
            let raw_args = script_command.get_command_line_args().unwrap_or_default();
            let mut cmd_line_args: Vec<String> = Vec::with_capacity(raw_args.len());
            for arg in raw_args.iter() {
                match arg {
                    Some(arg) => cmd_line_args.push(arg.clone()),
                    None => {
                        return Err(ParseComScriptError::NumberFormat(
                            "java.lang.NullPointerException".to_string(),
                        ));
                    }
                }
            }
            let length = cmd_line_args.len();
            let out_of_bounds = |index: usize| {
                ParseComScriptError::NumberFormat(format!(
                    "Index {index} out of bounds for length {length}"
                ))
            };
            self.reset();
            let mut i: usize = 0;
            while i < length {
                let arg = cmd_line_args[i].clone();
                let lower = arg.to_lowercase();
                // Is it an argument or filename
                if arg.starts_with('-') {
                    // Handle all the colornewst parameters
                    if self.use_color_newst && (arg.starts_with("-cn") || arg.starts_with("--cn")) {
                        if lower.ends_with(self.cntiff.get_name()) {
                            self.cntiff.set_boolean(true);
                        } else if arg.ends_with(self.cntempdir.get_name()) {
                            i += 1;
                            let value = cmd_line_args.get(i).ok_or_else(|| out_of_bounds(i))?;
                            self.cntempdir.set_string(Some(value.as_str()));
                        } else if arg.ends_with(self.cnmaxtemp.get_name()) {
                            i += 1;
                            let value = cmd_line_args.get(i).ok_or_else(|| out_of_bounds(i))?;
                            self.cnmaxtemp.set_string(Some(value.as_str()));
                        } else if lower.ends_with(self.cnverbose.get_name()) {
                            self.cnverbose.set_boolean(true);
                        } else {
                            let message = format!("Unknown argument: {arg}");
                            return Err(InvalidParameterException::new(Some(&message)).into());
                        }
                    } else if lower.starts_with("-inp") {
                        i += 1;
                        let value = cmd_line_args.get(i).ok_or_else(|| out_of_bounds(i))?;
                        self.input_file.push(value.clone());
                    } else if lower.starts_with("-out") {
                        i += 1;
                        let value = cmd_line_args.get(i).ok_or_else(|| out_of_bounds(i))?;
                        self.output_file.set(Some(value.as_str()));
                        self.output_file_type = None;
                    } else if arg.starts_with("-filei") || arg.starts_with("-FileOfI") {
                        i += 1;
                        let value = cmd_line_args.get(i).ok_or_else(|| out_of_bounds(i))?;
                        self.file_of_inputs.set(Some(value.as_str()));
                    } else if arg.starts_with("-fileo") || arg.starts_with("-FileOfO") {
                        i += 1;
                        let value = cmd_line_args.get(i).ok_or_else(|| out_of_bounds(i))?;
                        self.file_of_outputs.set(Some(value.as_str()));
                    } else if lower.starts_with("-sec") {
                        i += 1;
                        let value = cmd_line_args.get(i).ok_or_else(|| out_of_bounds(i))?;
                        self.sections_to_read.push(value.clone());
                    } else if lower.starts_with("-num") {
                        i += 1;
                        let value = cmd_line_args.get(i).ok_or_else(|| out_of_bounds(i))?;
                        self.number_to_output.push(value.clone());
                    } else if lower.starts_with("-siz") {
                        i += 1;
                        let value = cmd_line_args.get(i).ok_or_else(|| out_of_bounds(i))?;
                        self.size_to_output_in_xand_y
                            .validate_and_set(Some(value.as_str()))?;
                    } else if lower.starts_with(DATA_MODE_OPTION) {
                        i += 1;
                        let value = cmd_line_args.get(i).ok_or_else(|| out_of_bounds(i))?;
                        self.mode_to_output.set_string(Some(value.as_str()));
                    } else if lower.starts_with("-off") {
                        i += 1;
                        let value = cmd_line_args.get(i).ok_or_else(|| out_of_bounds(i))?;
                        // The first entry goes into the single variable, the next
                        // entries go into the vector.
                        if self.offsets_in_xand_y.is_null() {
                            self.offsets_in_xand_y
                                .validate_and_set(Some(value.as_str()))?;
                        } else {
                            let mut input = FortranInputString::new(2);
                            input.validate_and_set(Some(value.as_str()))?;
                            self.offsets_in_xand_y_extra_entries.push(input);
                        }
                    } else if lower.starts_with("-appl") {
                        self.apply_offsets_first.set_boolean(true);
                    } else if arg.starts_with("-xf") || arg.starts_with("-Tra") {
                        i += 1;
                        let value = cmd_line_args.get(i).ok_or_else(|| out_of_bounds(i))?;
                        self.transform_file.set(Some(value.as_str()));
                    } else if lower.starts_with("-use") {
                        i += 1;
                        let value = cmd_line_args.get(i).ok_or_else(|| out_of_bounds(i))?;
                        self.use_transform_lines.set(Some(value.as_str()));
                    } else if lower.starts_with("-rot") {
                        i += 1;
                        let value = cmd_line_args.get(i).ok_or_else(|| out_of_bounds(i))?;
                        self.rotate_by_angle.set_string(Some(value.as_str()));
                    } else if lower.starts_with("-exp") {
                        i += 1;
                        let value = cmd_line_args.get(i).ok_or_else(|| out_of_bounds(i))?;
                        self.expand_by_factor.set_string(Some(value.as_str()));
                    } else if lower.starts_with("-bin") {
                        i += 1;
                        let value = cmd_line_args.get(i).ok_or_else(|| out_of_bounds(i))?;
                        self.bin_by_factor.set_string(Some(value.as_str()));
                    } else if lower.starts_with("-lin") {
                        self.linear_interpolation.set_boolean(true);
                    } else if lower.starts_with(FLOAT_DENSITIES_OPTION) {
                        i += 1;
                        let value = cmd_line_args.get(i).ok_or_else(|| out_of_bounds(i))?;
                        self.float_densities.set_string(Some(value.as_str()));
                    } else if lower.starts_with("-con") {
                        i += 1;
                        let value = cmd_line_args.get(i).ok_or_else(|| out_of_bounds(i))?;
                        self.contrast_black_white
                            .validate_and_set(Some(value.as_str()))?;
                    } else if lower.starts_with("-sca") {
                        i += 1;
                        let value = cmd_line_args.get(i).ok_or_else(|| out_of_bounds(i))?;
                        self.scale_min_and_max
                            .validate_and_set(Some(value.as_str()))?;
                    } else if lower.starts_with("-dis") {
                        i += 1;
                        let value = cmd_line_args.get(i).ok_or_else(|| out_of_bounds(i))?;
                        self.distortion_field.set(Some(value.as_str()));
                    } else if lower.starts_with("-ima") {
                        i += 1;
                        let value = cmd_line_args.get(i).ok_or_else(|| out_of_bounds(i))?;
                        self.images_are_binned.set_string(Some(value.as_str()));
                    } else if lower.starts_with("-tes") {
                        i += 1;
                        let value = cmd_line_args.get(i).ok_or_else(|| out_of_bounds(i))?;
                        self.test_limits.validate_and_set(Some(value.as_str()))?;
                    } else if lower.starts_with("-grad") {
                        i += 1;
                        let value = cmd_line_args.get(i).ok_or_else(|| out_of_bounds(i))?;
                        self.gradient_file.set(Some(value.as_str()));
                    } else if lower.starts_with("-ori") || arg.starts_with("-Adj") {
                        self.adjust_origin.set_boolean(true);
                    } else if lower.starts_with("-taper") {
                        i += 1;
                        let value = cmd_line_args.get(i).ok_or_else(|| out_of_bounds(i))?;
                        self.taper_at_fill.validate_and_set(Some(value.as_str()))?;
                    } else if lower.starts_with("-fill") {
                        i += 1;
                        let value = cmd_line_args.get(i).ok_or_else(|| out_of_bounds(i))?;
                        self.fill_value.set_string(Some(value.as_str()));
                    } else {
                        let message = format!("Unknown argument: {arg}");
                        return Err(InvalidParameterException::new(Some(&message)).into());
                    }
                }
                // input and output filename arguments
                else if i == length - 1 {
                    self.output_file.set(Some(arg.as_str()));
                    self.output_file_type = None;
                } else {
                    self.input_file.push(arg.clone());
                }
                i += 1;
            }
        }
        Ok(())
    }

    /// Java `updateComScriptCommand(ComScriptCommand)`.  Update the script command
    /// with the current values of this NewstParam object.
    fn update_com_script_command(
        &self,
        script_command: &mut ComScriptCommand,
    ) -> Result<(), BadComScriptException> {
        if !self.use_color_newst {
            script_command.use_keyword_value();
            for value in self.input_file.iter() {
                script_command.set_value(Some(INPUT_FILE_KEY), Some(value.as_str()));
            }
            self.output_file.update_com_script(script_command);
            self.file_of_inputs.update_com_script(script_command);
            self.file_of_outputs.update_com_script(script_command);
            for value in self.sections_to_read.iter() {
                script_command.set_value(Some(SECTIONS_TO_READ_KEY), Some(value.as_str()));
            }
            for value in self.number_to_output.iter() {
                script_command.set_value(Some(NUMBER_TO_OUTPUT_KEY), Some(value.as_str()));
            }
            self.size_to_output_in_xand_y
                .update_script_parameter(script_command);
            self.mode_to_output.update_com_script(script_command);
            self.offsets_in_xand_y
                .update_script_parameter(script_command);
            for fis in self.offsets_in_xand_y_extra_entries.iter() {
                fis.update_script_parameter(script_command);
            }
            self.apply_offsets_first.update_com_script(script_command);
            self.transform_file.update_com_script(script_command);
            self.use_transform_lines.update_com_script(script_command);
            self.rotate_by_angle.update_com_script(script_command);
            self.expand_by_factor.update_com_script(script_command);
            self.bin_by_factor.update_com_script(script_command);
            self.linear_interpolation.update_com_script(script_command);
            self.float_densities.update_com_script(script_command);
            self.contrast_black_white
                .update_script_parameter(script_command);
            self.scale_min_and_max
                .update_script_parameter(script_command);
            self.distortion_field.update_com_script(script_command);
            self.images_are_binned.update_com_script(script_command);
            self.test_limits.update_script_parameter(script_command);
            self.gradient_file.update_com_script(script_command);
            self.adjust_origin.update_com_script(script_command);
            self.taper_at_fill.update_script_parameter(script_command);
            self.fill_value.update_com_script(script_command);
            self.antialias_filter.update_com_script(script_command);
        } else {
            // Create a new command line argument array
            let mut cmd_line_args: Vec<String> = Vec::new();
            // colornewst
            if self.cntiff.is() {
                cmd_line_args.push(self.cntiff.get_name().to_string());
            }
            if !self.cntempdir.is_null() {
                cmd_line_args.push(self.cntempdir.get_name().to_string());
                cmd_line_args.push(self.cntempdir.to_string());
            }
            if !self.cnmaxtemp.is_null() {
                cmd_line_args.push(self.cnmaxtemp.get_name().to_string());
                cmd_line_args.push(self.cnmaxtemp.to_string());
            }
            if self.cnverbose.is() {
                cmd_line_args.push(self.cnverbose.get_name().to_string());
            }
            if !self.file_of_inputs.is_empty() {
                cmd_line_args.push("-fileinlist".to_string());
                cmd_line_args.push(self.file_of_inputs.to_string());
            }
            if !self.file_of_outputs.is_empty() {
                cmd_line_args.push("-fileoutlist".to_string());
                cmd_line_args.push(self.file_of_outputs.to_string());
            }
            for value in self.sections_to_read.iter() {
                cmd_line_args.push("-secs".to_string());
                cmd_line_args.push(value.clone());
            }
            for value in self.number_to_output.iter() {
                cmd_line_args.push("-numout".to_string());
                cmd_line_args.push(value.clone());
            }
            if self.size_to_output_in_xand_y.values_set()
                && (!self.size_to_output_in_xand_y.is_default())
            {
                cmd_line_args.push("-size".to_string());
                cmd_line_args.push(self.size_to_output_in_xand_y.to_string());
            }
            if !self.mode_to_output.is_null() {
                cmd_line_args.push(DATA_MODE_OPTION.to_string());
                cmd_line_args.push(self.mode_to_output.to_string());
            }
            if !self.offsets_in_xand_y.is_null() {
                cmd_line_args.push("-offset".to_string());
                cmd_line_args.push(self.offsets_in_xand_y.to_string_default_is_blank(true));
            }
            for fis in self.offsets_in_xand_y_extra_entries.iter() {
                if !fis.is_null() {
                    cmd_line_args.push("-offset".to_string());
                    cmd_line_args.push(fis.to_string_default_is_blank(true));
                }
            }
            if self.apply_offsets_first.is() {
                cmd_line_args.push("-applyfirst".to_string());
            }
            if !self.transform_file.is_empty() {
                cmd_line_args.push("-xform".to_string());
                cmd_line_args.push(self.transform_file.to_string());
            }
            if !self.use_transform_lines.is_empty() {
                cmd_line_args.push("-uselines".to_string());
                cmd_line_args.push(self.use_transform_lines.to_string());
            }
            if !self.rotate_by_angle.is_null() {
                cmd_line_args.push("-rotate".to_string());
                cmd_line_args.push(self.rotate_by_angle.to_string());
            }
            if !self.expand_by_factor.is_null() {
                cmd_line_args.push("-expand".to_string());
                cmd_line_args.push(self.expand_by_factor.to_string());
            }
            if !self.bin_by_factor.is_null() {
                cmd_line_args.push("-bin".to_string());
                cmd_line_args.push(self.bin_by_factor.to_string());
            }
            if self.linear_interpolation.is() {
                cmd_line_args.push("-linear".to_string());
            }
            if !self.float_densities.is_null() {
                cmd_line_args.push(FLOAT_DENSITIES_OPTION.to_string());
                cmd_line_args.push(self.float_densities.to_string());
            }
            if self.contrast_black_white.values_set() && (!self.contrast_black_white.is_default()) {
                cmd_line_args.push("-contrast".to_string());
                cmd_line_args.push(self.contrast_black_white.to_string());
            }
            if self.scale_min_and_max.values_set() && (!self.scale_min_and_max.is_default()) {
                cmd_line_args.push("-scale".to_string());
                cmd_line_args.push(self.scale_min_and_max.to_string());
            }
            if !self.distortion_field.is_empty() {
                cmd_line_args.push("-distort".to_string());
                cmd_line_args.push(self.distortion_field.to_string());
            }
            if !self.images_are_binned.is_null() {
                cmd_line_args.push("-imagebinned".to_string());
                cmd_line_args.push(self.images_are_binned.to_string());
            }
            if self.test_limits.values_set() && (!self.test_limits.is_default()) {
                cmd_line_args.push("-test".to_string());
                cmd_line_args.push(self.test_limits.to_string());
            }
            if !self.gradient_file.is_empty() {
                cmd_line_args.push("-grad".to_string());
                cmd_line_args.push(self.gradient_file.to_string());
            }
            if self.adjust_origin.is() {
                cmd_line_args.push("-origin".to_string());
            }
            if self.taper_at_fill.values_set() && (!self.taper_at_fill.is_default()) {
                cmd_line_args.push("-taper".to_string());
                cmd_line_args.push(self.taper_at_fill.to_string());
            }
            if !self.fill_value.is_null() {
                cmd_line_args.push("-fill".to_string());
                cmd_line_args.push(self.fill_value.to_string());
            }
            // Add input file(s) and output file last and without a parameter tag.
            for value in self.input_file.iter() {
                // cmdLineArgs.add("-input");
                cmd_line_args.push(value.clone());
            }
            // cmdLineArgs.add("-output");
            cmd_line_args.push(self.output_file.to_string());
            let args: Vec<Option<String>> = cmd_line_args.into_iter().map(Some).collect();
            script_command.set_command_line_args(&args);

            // If the command is currently newst change it to newstack
            let command_name = self.get_command_name();
            script_command.set_command(command_name.as_deref());
            if etomo_director::INSTANCE.get_arguments().is_debug() {
                eprintln!("{}", script_command.get_command().unwrap_or("null"));
                let command_array = script_command.get_command_line_args();
                if let Some(command_array) = command_array {
                    for i in 0..command_array.len() {
                        eprint!("{} ", command_array[i].as_deref().unwrap_or("null"));
                    }
                    if !command_array.is_empty() {
                        eprintln!();
                    }
                }
            }
        }
        Ok(())
    }

    /// Java `initializeDefaults()`.
    fn initialize_defaults(&mut self) {}
}

impl ConstNewstParam for NewstParam {
    /// Java `fillValueEquals(int)`.
    fn fill_value_equals(&self, value: i32) -> bool {
        self.fill_value.equals_int(value)
    }

    /// Java `getBinByFactor()`.
    fn get_bin_by_factor(&self) -> i32 {
        self.bin_by_factor.get_int()
    }

    /// Java `getFloatDensities()`.
    fn get_float_densities(&self) -> i32 {
        self.float_densities.get_int()
    }

    /// Java `getInputFile()`.  Backward compatibility with pre PIP structure, just
    /// return the first input file.
    ///
    /// NewstParam.java:915 is `inputFile.get(0)`, an uncaught
    /// ArrayIndexOutOfBoundsException when there is no input file.  Fixed in
    /// translation: `None`.
    fn get_input_file(&self) -> Option<String> {
        self.input_file.first().cloned()
    }

    /// Java `isAntialiasFilterNull()`.
    fn is_antialias_filter_null(&self) -> bool {
        self.antialias_filter.is_null()
    }

    /// Java `isLinearInterpolation()`.
    fn is_linear_interpolation(&self) -> bool {
        self.linear_interpolation.is()
    }

    /// Java `getModeToOutput()`.
    fn get_mode_to_output(&self) -> i32 {
        self.mode_to_output.get_int()
    }

    /// Java `getAntialiasFilter()`.
    fn get_antialias_filter(&self) -> String {
        self.antialias_filter.to_string()
    }

    /// Java `getOffsetInX()`.  Returns the X value of offsetsInXandY.
    fn get_offset_in_x(&self) -> String {
        self.offsets_in_xand_y.to_string_index(0)
    }

    /// Java `getOffsetInY()`.  Returns the Y value of offsetsInXandY.
    fn get_offset_in_y(&self) -> String {
        self.offsets_in_xand_y.to_string_index(1)
    }

    /// Java `getOutputFile()`.
    fn get_output_file(&self) -> String {
        if self.output_file.is_empty() {
            return String::new();
        }
        self.output_file.to_string()
    }

    /// Java `getSizeToOutputInX()`.
    fn get_size_to_output_in_x(&self) -> i32 {
        self.size_to_output_in_xand_y.get_int(0)
    }

    /// Java `getSizeToOutputInY()`.
    fn get_size_to_output_in_y(&self) -> i32 {
        self.size_to_output_in_xand_y.get_int(1)
    }

    /// Java `isSizeToOutputInXandYSet()`.
    fn is_size_to_output_in_x_and_y_set(&self) -> bool {
        !self.size_to_output_in_xand_y.is_null()
    }
}

impl Command for NewstParam {
    /// Java `getAxisID()`.
    fn get_axis_id(&self) -> AxisID {
        self.axis_id
    }

    /// Java `getSubcommandDetails()`.
    fn get_subcommand_details(&self) -> Option<&dyn CommandDetails> {
        None
    }

    /// Java `getSubcommandProcessName()`.
    fn get_subcommand_process_name(&self) -> Option<String> {
        None
    }

    /// Java `getOutputImageFileType()` (deprecated 3/15/2019).
    fn get_output_image_file_type(&self) -> Option<Arc<FileType>> {
        None
    }

    /// Java `getOutputImageFileKey()`.
    fn get_output_image_file_key(&self) -> Option<FileKey> {
        if let Some(output_image_file_key) = &self.output_image_file_key {
            return Some(output_image_file_key.clone());
        }
        if let Some(output_file_type) = &self.output_file_type {
            return Some(FileKey::clone(output_file_type));
        }
        None
    }

    /// Java `getOutputImageFileType2()` (deprecated 3/15/2019).
    fn get_output_image_file_type2(&self) -> Option<Arc<FileType>> {
        if self.mode == Some(Mode::WholeTomogramSample) {
            // Handle tiltParam here so the user doesn't have to wait.
            // (Java reads the axis type and never uses it.)
            let _axis_type = self
                .manager
                .get_base_meta_data()
                .map(|meta_data| meta_data.base().get_axis_type());
            return Some(Arc::clone(&file_type::CLASS.tilt_output));
        }
        None
    }

    /// Java `getOutputImageFileKey2()`.
    fn get_output_image_file_key2(&self) -> Option<FileKey> {
        if self.mode == Some(Mode::WholeTomogramSample) {
            // Handle tiltParam here so the user doesn't have to wait.
            let _axis_type = self
                .manager
                .get_base_meta_data()
                .map(|meta_data| meta_data.base().get_axis_type());
            return Some(FileKey::clone(&file_type::CLASS.tilt_output));
        }
        None
    }

    /// Java `getCommand()`.
    fn get_command(&self) -> Option<String> {
        Some(self.get_command_file_name(self.axis_id))
    }

    /// Java `getCommandLine()`.
    fn get_command_line(&self) -> Option<String> {
        Some(self.get_command_file_name(self.axis_id))
    }

    /// Java `getCommandName()`.
    fn get_command_name(&self) -> Option<String> {
        if self.use_color_newst {
            return Some("colornewst".to_string());
        }
        Some("newstack".to_string())
    }

    /// Java `getProcessName()`.
    fn get_process_name(&self) -> Option<ProcessName> {
        Some(self.process_name)
    }

    /// Java `getCommandArray()`.
    fn get_command_array(&self) -> Option<Vec<String>> {
        let array = vec![self.get_command_line().unwrap_or_default()];
        Some(array)
    }

    /// Java `getCommandMode()`.
    fn get_command_mode(&self) -> Option<&dyn CommandMode> {
        self.mode.as_ref().map(|mode| mode as &dyn CommandMode)
    }

    /// Java `isMessageReporter()`.
    fn is_message_reporter(&self) -> bool {
        false
    }

    /// Java `getCommandOutputFile()`.
    fn get_command_output_file(&self) -> Option<PathBuf> {
        None
    }

    /// Java `getCommandInputFile()`.
    fn get_command_input_file(&self) -> Option<PathBuf> {
        None
    }

    /// Java `command instanceof ProcessDetails`: this is a `CommandDetails`.
    fn get_process_details(&self) -> Option<&dyn ProcessDetails> {
        Some(self)
    }
}

impl Loggable for NewstParam {
    /// Java `getName()`.
    fn get_name(&self) -> String {
        self.process_name.to_string()
    }

    /// Java `getLogMessage()`, which returns null; there is no message.
    fn get_log_message(&self) -> Result<Vec<Option<String>>, LoggableException> {
        Ok(Vec::new())
    }
}

/// The ProcessDetails getters throw `IllegalArgumentException("field=" + field)` for
/// an unrecognised field (NewstParam.java:1090-1144), which no caller catches.  Fixed
/// in translation: the value is unavailable (`None`).
impl ProcessDetails for NewstParam {
    /// Java `getIntValue(FieldInterface)`.
    fn get_int_value(&self, field: &dyn FieldInterface) -> Option<i32> {
        match field_interface::as_field::<Field>(field) {
            Some(Field::Binning) => Some(self.get_bin_by_factor()),
            _ => None,
        }
    }

    /// Java `getIteratorElementList(FieldInterface)`.
    fn get_iterator_element_list(&self, _field: &dyn FieldInterface) -> Option<Vec<i32>> {
        None
    }

    /// Java `getBooleanValue(FieldInterface)`.
    fn get_boolean_value(&self, field: &dyn FieldInterface) -> Option<bool> {
        match field_interface::as_field::<Field>(field) {
            Some(Field::FiducialessAlignment) => Some(self.fiducialess_alignment),
            Some(Field::UseLinearInterpolation) => Some(self.linear_interpolation.is()),
            _ => None,
        }
    }

    /// Java `getStringArray(FieldInterface)`.
    fn get_string_array(&self, _field: &dyn FieldInterface) -> Option<Vec<String>> {
        None
    }

    /// Java `getString(FieldInterface)`.
    fn get_string(&self, field: &dyn FieldInterface) -> Option<String> {
        match field_interface::as_field::<Field>(field) {
            Some(Field::UserSizeToOutputInXAndY) => Some(
                self.user_size_to_output_in_xand_y
                    .to_string_default_is_blank(true),
            ),
            _ => None,
        }
    }

    /// Java `getDoubleValue(FieldInterface)`.
    fn get_double_value(&self, _field: &dyn FieldInterface) -> Option<f64> {
        None
    }

    /// Java `getEtomoNumber(FieldInterface)`.
    fn get_etomo_number(&self, field: &dyn FieldInterface) -> Option<ConstEtomoNumber> {
        match field_interface::as_field::<Field>(field) {
            Some(Field::ImageRotation) => Some((*self.image_rotation).clone()),
            _ => None,
        }
    }

    /// Java `getIntKeyList(FieldInterface)`.
    fn get_int_key_list(&self, _field: &dyn FieldInterface) -> Option<Vec<(i32, String)>> {
        None
    }

    /// Java `getHashtable(FieldInterface)`.
    fn get_hashtable(&self, _field: &dyn FieldInterface) -> Option<Vec<(String, String)>> {
        None
    }
}
