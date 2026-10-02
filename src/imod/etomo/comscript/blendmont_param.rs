//! `IMOD/Etomo/src/etomo/comscript/BlendmontParam.java`.
//!
//! Parameters for the blendmont command in xcorr, preblend, blend, undistort and the
//! 3dfind/whole-tomogram/serial-section variants.
//!
//! TODO(unit): needs etomo/type/StringParameter.java - `distortionField`,
//! `imageInputFile`, `pieceListInput`, `rootNameForEdges`, `transformFile`,
//! `otherSumGradientFile` (r#type::string_parameter::StringParameter).
//! TODO(unit): needs etomo/comscript/Utilities.java - `is90DegreeImageRotation` and
//! `getGoodframeFromMontageSize` (comscript::utilities).
//! TODO(unit): needs etomo/util/Goodframe.java - `getOutput`.
//! TODO(unit): needs etomo/util/Montagesize.java - `getInstance`, `read`, `getX`, `getY`.
//! TODO(unit): needs etomo/type/EtomoState.java - `TRUE_VALUE`.
//! TODO(unit): needs etomo/logic/TomogramTool.java - `PairXAndY`.
//! TODO(unit): needs FortranInputString.validateAndSet(ComScriptCommand) and
//! updateScriptParameter(ComScriptCommand) (TODOs in fortran_input_string.rs).

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
use super::process_details::ProcessDetails;
use super::utilities;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::logic::tomogram_tool::PairXAndY;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::const_etomo_number::{ConstEtomoNumber, Number, Type};
use crate::imod::etomo::r#type::etomo_boolean2::EtomoBoolean2;
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::r#type::etomo_state;
use crate::imod::etomo::r#type::file_key::FileKey;
use crate::imod::etomo::r#type::file_type::{self, FileType};
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::script_parameter::ScriptParameter;
use crate::imod::etomo::r#type::string_parameter::StringParameter;
use crate::imod::etomo::ui::swing::log_interface::{Loggable, LoggableException};
use crate::imod::etomo::ui::swing::ui_harness;
use crate::imod::etomo::util::montagesize::Montagesize;
use crate::imod::etomo::util::utilities::java_io_file_last_modified;

pub const GOTO_LABEL: &str = "doblend";
pub const COMMAND_NAME: &str = "blendmont";
pub const LINEAR_INTERPOLATION_ORDER: i32 = 1;
pub const PIECE_TO_PIECE_DIFFERENCES_ONLY: i32 = 1;
pub const GRADIENT_WITHIN_PIECES_ALSO: i32 = 2;
pub const SUM_PIECES_FOR_GRADIENT: i32 = 1;
pub const WEIGHT_FOR_EXPECTED_SHIFTS_DEFAULT: f64 = 1.0;
/// Deprecated 4/12/2019.
pub const OUTPUT_FILE_EXTENSION: &str = ".ali";
/// Deprecated 6/17/2019.
pub const DISTORTION_CORRECTED_STACK_EXTENSION: &str = ".dcst";
/// Deprecated 6/17/2019.
pub const BLENDMONT_STACK_EXTENSION: &str = ".bl";
pub const IMAGE_OUTPUT_FILE_KEY: &str = "ImageOutputFile";
pub const IMAGES_ARE_BINNED_KEY: &str = "ImagesAreBinned";
pub const DISTORTION_FIELD_KEY: &str = "DistortionField";
pub const VERY_SLOPPY_MONTAGE_KEY: &str = "VerySloppyMontage";
pub const WEIGHT_FOR_EXPECTED_SHIFTS_KEY: &str = "WeightForExpectedShifts";
pub const EM_GRID_MAP_FILTER_KEY: &str = "EMGridMapFilter";
pub const ROBUST_FIT_CRITERION_KEY: &str = "RobustFitCriterion";
pub const STARTING_AND_ENDING_X_KEY: &str = "StartingAndEndingX";
pub const STARTING_AND_ENDING_Y_KEY: &str = "StartingAndEndingY";
pub const BIN_BY_FACTOR_KEY: &str = "BinByFactor";
pub const FILL_VALUE_KEY: &str = "FillValue";

/// The three checked exceptions `convertToStartingAndEndingXandY` declares.
/// `Montagesize.read` reports its `etomo.util.InvalidParameterException` and
/// `IOException` through one message, carried by `MontagesizeRead`.
#[derive(Debug)]
pub enum ConvertError {
    FortranInputSyntax(FortranInputSyntaxException),
    MontagesizeRead(String),
}

impl std::fmt::Display for ConvertError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConvertError::FortranInputSyntax(e) => write!(f, "{e}"),
            ConvertError::MontagesizeRead(e) => write!(f, "{e}"),
        }
    }
}

impl From<FortranInputSyntaxException> for ConvertError {
    fn from(e: FortranInputSyntaxException) -> ConvertError {
        ConvertError::FortranInputSyntax(e)
    }
}

/// Java nested `BlendmontParam.Field implements FieldInterface`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Field {
    OldEdgeFunctions,
    Fiducialess,
    ImageRotation,
    LinearInterpolation,
    UserSizeToOutputInXAndY,
    RobustFitting,
    FixIntensityFromEdges,
    SumPiecesForGradient,
    OtherSumGradientFile,
}

impl FieldInterface for Field {}

/// Java nested `BlendmontParam.Mode implements CommandMode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// "XCorr".
    Xcorr,
    /// "Preblend".
    Preblend,
    /// "Blend".
    Blend,
    /// "Blend_3dfind".
    Blend3dFind,
    /// "Undistort".
    Undistort,
    /// "WholeTomogramSample".
    WholeTomogramSample,
    /// "SerialSections_Preblend".
    SerialSectionPreblend,
    /// "SerialSections_Blend".
    SerialSectionBlend,
}

impl std::fmt::Display for Mode {
    /// Java `toString()`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Mode::Xcorr => "XCorr",
            Mode::Preblend => "Preblend",
            Mode::Blend => "Blend",
            Mode::Blend3dFind => "Blend_3dfind",
            Mode::Undistort => "Undistort",
            Mode::WholeTomogramSample => "WholeTomogramSample",
            Mode::SerialSectionPreblend => "SerialSections_Preblend",
            Mode::SerialSectionBlend => "SerialSections_Blend",
        })
    }
}

impl CommandMode for Mode {}

/// Java final `BlendmontParam implements CommandParam, CommandDetails`.
pub struct BlendmontParam {
    distortion_field: StringParameter,
    image_input_file: StringParameter,
    piece_list_input: StringParameter,
    root_name_for_edges: StringParameter,
    images_are_binned: ScriptParameter,
    sloppy_montage: EtomoBoolean2,
    very_sloppy_montage: EtomoBoolean2,
    weight_for_expected_shifts: ScriptParameter,
    e_m_grid_map_filter: ScriptParameter,
    robust_fit_criterion: ScriptParameter,
    fill_value: ScriptParameter,
    transform_file: StringParameter,
    fix_intensity_from_edges: ScriptParameter,
    sum_pieces_for_gradient: ScriptParameter,
    other_sum_gradient_file: StringParameter,
    /// @version 3.10.  Script is from an earlier version if false.
    adjust_origin: EtomoBoolean2,
    starting_and_ending_x: FortranInputString,
    starting_and_ending_y: FortranInputString,
    image_rotation: EtomoNumber,
    unaligned_starting_xand_y: FortranInputString,

    axis_id: AxisID,
    dataset_name: Option<String>,
    read_in_xcorrs: EtomoBoolean2,
    old_edge_functions: EtomoBoolean2,
    interpolation_order: ScriptParameter,
    just_undistort: EtomoBoolean2,
    bin_by_factor: ScriptParameter,
    manager: &'static dyn BaseManager,

    user_size_to_output_in_xand_y: String,
    image_output_file_type: Option<Arc<FileType>>,
    image_output_file: Option<String>,
    fiducialess: bool,
    mode: Mode,
    process_name: ProcessName,
    image_output_file_for_3d_find: bool,
    validate: bool,
    from_scratch: bool,
}

impl BlendmontParam {
    /// Java `BlendmontParam(BaseManager, String, AxisID)`.
    pub fn new(
        manager: &'static dyn BaseManager,
        dataset_name: Option<&str>,
        axis_id: AxisID,
    ) -> BlendmontParam {
        BlendmontParam::new_with_mode(manager, dataset_name, axis_id, Mode::Xcorr)
    }

    /// Java `BlendmontParam(BaseManager, String, AxisID, Mode)`.
    pub fn new_with_mode(
        manager: &'static dyn BaseManager,
        dataset_name: Option<&str>,
        axis_id: AxisID,
        mode: Mode,
    ) -> BlendmontParam {
        let mut param = BlendmontParam {
            distortion_field: StringParameter::new("DistortionField"),
            image_input_file: StringParameter::new("ImageInputFile"),
            piece_list_input: StringParameter::new("PieceListInput"),
            root_name_for_edges: StringParameter::new("RootNameForEdges"),
            images_are_binned: ScriptParameter::new_with_type_and_name(
                Type::Double,
                "ImagesAreBinned",
            ),
            sloppy_montage: EtomoBoolean2::new_with_name("SloppyMontage"),
            very_sloppy_montage: EtomoBoolean2::new_with_name(VERY_SLOPPY_MONTAGE_KEY),
            weight_for_expected_shifts: ScriptParameter::new_with_type_and_name(
                Type::Double,
                WEIGHT_FOR_EXPECTED_SHIFTS_KEY,
            ),
            e_m_grid_map_filter: ScriptParameter::new_with_type_and_name(
                Type::Double,
                EM_GRID_MAP_FILTER_KEY,
            ),
            robust_fit_criterion: ScriptParameter::new_with_type_and_name(
                Type::Double,
                ROBUST_FIT_CRITERION_KEY,
            ),
            fill_value: ScriptParameter::new_with_type_and_name(Type::Double, FILL_VALUE_KEY),
            transform_file: StringParameter::new("TransformFile"),
            fix_intensity_from_edges: ScriptParameter::new_with_type_and_name(
                Type::Integer,
                "FixIntensityFromEdges",
            ),
            sum_pieces_for_gradient: ScriptParameter::new_with_type_and_name(
                Type::Integer,
                "SumPiecesForGradient",
            ),
            other_sum_gradient_file: StringParameter::new("OtherSumGradientFile"),
            adjust_origin: EtomoBoolean2::new_with_name("AdjustOrigin"),
            starting_and_ending_x: FortranInputString::new_with_key(
                Some(STARTING_AND_ENDING_X_KEY),
                2,
            ),
            starting_and_ending_y: FortranInputString::new_with_key(
                Some(STARTING_AND_ENDING_Y_KEY),
                2,
            ),
            image_rotation: EtomoNumber::new_with_type(Some(Type::Double)),
            unaligned_starting_xand_y: FortranInputString::new_with_key(
                Some("UnalignedStartingXandY"),
                2,
            ),
            axis_id,
            dataset_name: dataset_name.map(|s| s.to_string()),
            read_in_xcorrs: EtomoBoolean2::new_with_name("ReadInXcorrs"),
            old_edge_functions: EtomoBoolean2::new_with_name("OldEdgeFunctions"),
            interpolation_order: ScriptParameter::new_with_type_and_name(
                Type::Integer,
                "InterpolationOrder",
            ),
            just_undistort: EtomoBoolean2::new_with_name("JustUndistort"),
            bin_by_factor: ScriptParameter::new_with_type_and_name(
                Type::Integer,
                BIN_BY_FACTOR_KEY,
            ),
            manager,
            user_size_to_output_in_xand_y: String::new(),
            image_output_file_type: None,
            image_output_file: None,
            fiducialess: false,
            mode,
            process_name: ProcessName::XCORR,
            image_output_file_for_3d_find: false,
            validate: false,
            from_scratch: false,
        };
        param.read_in_xcorrs.set_display_as_integer(true);
        param.old_edge_functions.set_display_as_integer(true);
        param.image_output_file = None;
        param.image_output_file_type = None;
        // Only explcitly write out the binning if its value is something other than
        // the default of 1 to keep from cluttering up the com script
        param.bin_by_factor.set_default_int(1);
        param
            .starting_and_ending_x
            .set_integer_type_array(&[true, true]);
        param.starting_and_ending_x.set_divider(' ');
        param
            .starting_and_ending_y
            .set_integer_type_array(&[true, true]);
        param.starting_and_ending_y.set_divider(' ');
        param
            .unaligned_starting_xand_y
            .set_integer_type_array(&[true, true]);
        param.set_process_name();
        param
    }

    /// Java `setValidate(boolean)`.
    pub fn set_validate(&mut self, validate: bool) {
        self.validate = validate;
    }

    /// Java private `reset()`.
    fn reset(&mut self) {
        self.validate = false;
        self.read_in_xcorrs.reset();
        self.old_edge_functions.reset();
        self.interpolation_order.reset();
        self.just_undistort.reset();
        self.image_output_file = None;
        self.image_output_file_type = None;
        self.image_output_file_for_3d_find = false;
        self.bin_by_factor.reset();
        self.starting_and_ending_x.reset();
        self.starting_and_ending_y.reset();
        self.adjust_origin.reset();
        self.fiducialess = false;
        self.user_size_to_output_in_xand_y = String::new();
        self.image_rotation.reset();
        self.distortion_field.reset();
        self.image_input_file.reset();
        self.piece_list_input.reset();
        self.root_name_for_edges.reset();
        self.images_are_binned.reset();
        self.sloppy_montage.reset();
        self.very_sloppy_montage.reset();
        self.weight_for_expected_shifts.reset();
        self.e_m_grid_map_filter.reset();
        self.robust_fit_criterion.reset();
        self.fill_value.reset();
        self.transform_file.reset();
        self.unaligned_starting_xand_y.reset();
        self.fix_intensity_from_edges.reset();
        self.sum_pieces_for_gradient.reset();
        self.other_sum_gradient_file.reset();
    }

    /// Java `convertToStartingAndEndingXandY(String, double, String)`.
    ///
    /// If nx is the size of the raw montage in X, and the user requests a size mx, then
    /// the starting coordinate to give blendmont is: int(nx/2) - int((mx+1)/2).  The
    /// ending coordinate is the starting coordinate + mx - 1.  Blendmont expects unbinned
    /// numbers here.
    ///
    /// If the user size is set, then this works even if the tilt axis angle is closer
    /// to 90 degress.  If the user size is not set and the tilt axis angle is closer to
    /// 90 degrees, then use x and y from goodframe (see
    /// TiltParam.setMontageFullImage()).  Mx = y from goodframe.  My = x from
    /// goodframe.  Then apply the formula above.
    ///
    /// Wlll be called with an empty string by Positioning - whole tomogram.
    pub fn convert_to_starting_and_ending_xand_y(
        &mut self,
        size_to_output_in_xand_y: &str,
        image_rotation: f64,
        description: Option<&str>,
    ) -> Result<bool, ConvertError> {
        let mut size_to_output_in_xand_y = size_to_output_in_xand_y.to_string();
        let mut retval = true;
        // make sure an empty string really causes sizeToOutputInXandY to be empty.
        if size_to_output_in_xand_y == "" {
            size_to_output_in_xand_y = "/".to_string();
        }
        self.starting_and_ending_x.reset();
        self.starting_and_ending_y.reset();
        self.user_size_to_output_in_xand_y = size_to_output_in_xand_y.clone();
        self.image_rotation.set_double(image_rotation);
        let mut fis_size_to_output_in_xand_y = FortranInputString::new(2);
        fis_size_to_output_in_xand_y.set_integer_type_array(&[true, true]);
        fis_size_to_output_in_xand_y.validate_and_set(Some(&size_to_output_in_xand_y))?;
        if (fis_size_to_output_in_xand_y.is_default() || fis_size_to_output_in_xand_y.is_empty())
            && utilities::is_90_degree_image_rotation(image_rotation)
        {
            let goodframe = utilities::get_goodframe_from_montage_size(self.axis_id, self.manager);
            if let Some(goodframe) = goodframe {
                // transposing x and y
                fis_size_to_output_in_xand_y
                    .set_index_const_etomo_number(1, &goodframe.get_output(0));
                fis_size_to_output_in_xand_y
                    .set_index_const_etomo_number(0, &goodframe.get_output(1));
            }
        }
        if self.validate && description.is_some() {
            if !fis_size_to_output_in_xand_y.is_null_index(0)
                && fis_size_to_output_in_xand_y.is_null_index(1)
            {
                ui_harness::post_message_dialog(
                    Some(self.manager),
                    format!("Two values are required for {}.", description.unwrap()),
                    "Entry Error".to_string(),
                    Some(self.axis_id),
                );
                retval = false;
            }
        }
        if fis_size_to_output_in_xand_y.is_default() || fis_size_to_output_in_xand_y.is_empty() {
            return Ok(retval);
        }
        let montagesize = Montagesize::get_instance(
            self.manager,
            self.axis_id,
            &file_type::CLASS.raw_stack,
            false,
        );
        // BlendmontParam.java:318-320 dereferences the instance without a null check.
        // Fixed in translation: with no instance nothing is converted.
        let montagesize = match montagesize {
            Some(montagesize) => montagesize,
            None => return Ok(retval),
        };
        montagesize
            .read(self.manager)
            .map_err(ConvertError::MontagesizeRead)?;
        let montage_x = montagesize.get_x().get_int();
        let size_x = fis_size_to_output_in_xand_y.get_int(0);
        BlendmontParam::convert_to_starting_and_ending(
            &mut self.starting_and_ending_x,
            montage_x,
            size_x,
        );
        let montage_y = montagesize.get_y().get_int();
        let size_y = fis_size_to_output_in_xand_y.get_int(1);
        BlendmontParam::convert_to_starting_and_ending(
            &mut self.starting_and_ending_y,
            montage_y,
            size_y,
        );
        Ok(retval)
    }

    /// Java private `convertToStartingAndEnding(FortranInputString, int, int)`.
    fn convert_to_starting_and_ending(
        starting_and_ending: &mut FortranInputString,
        montage_size: i32,
        size: i32,
    ) {
        let starting = (montage_size / 2).wrapping_sub(size.wrapping_add(1) / 2);
        starting_and_ending.set_index_double(0, starting as f64);
        if size == 0 {
            starting_and_ending.set_index_double(1, starting as f64);
        } else {
            starting_and_ending
                .set_index_double(1, starting.wrapping_add(size).wrapping_sub(1) as f64);
        }
    }

    /// Java `resetStartingAndEndingXandY()`.
    pub fn reset_starting_and_ending_xand_y(&mut self) {
        self.starting_and_ending_x.reset();
        self.starting_and_ending_y.reset();
        self.user_size_to_output_in_xand_y = String::new();
        self.image_rotation.reset();
    }

    /// Java `setMode(Mode)`.
    pub fn set_mode(&mut self, mode: Mode) {
        self.mode = mode;
        self.set_process_name();
    }

    /// Java `getMode()`.
    pub fn get_mode(&self) -> Mode {
        self.mode
    }

    /// Java `fillValueEquals(int)`.
    pub fn fill_value_equals(&self, value: i32) -> bool {
        self.fill_value.equals_int(value)
    }

    /// Java `setTransformFile(String)`.
    pub fn set_transform_file(&mut self, input: Option<&str>) {
        self.transform_file.set(input);
    }

    /// Java `setUnalignedStartingXandY(String[])`.
    pub fn set_unaligned_starting_xand_y(&mut self, input: Option<&[Option<String>]>) {
        self.unaligned_starting_xand_y.reset();
        let input = match input {
            None => return,
            Some(input) => input,
        };
        let mut i = 0;
        if input.len() > i {
            self.unaligned_starting_xand_y
                .set_index_string(i as i32, input[i].as_deref());
        }
        i += 1;
        if input.len() > i {
            self.unaligned_starting_xand_y
                .set_index_string(i as i32, input[i].as_deref());
        }
    }

    /// Java `setFillValue(int)`.
    pub fn set_fill_value(&mut self, input: i32) {
        self.fill_value.set_int(input);
    }

    /// Java `setPieceListInput(String)`.
    pub fn set_piece_list_input(&mut self, input: Option<&str>) {
        self.piece_list_input.set(input);
    }

    /// Java `setFromScratch(boolean)`.
    pub fn set_from_scratch(&mut self, input: bool) {
        self.from_scratch = input;
    }

    /// Java `setRootNameForEdges(String)`.
    pub fn set_root_name_for_edges(&mut self, input: Option<&str>) {
        self.root_name_for_edges.set(input);
    }

    /// Java `setStartingAndEndingXAndY(TomogramTool.PairXAndY)`.
    pub fn set_starting_and_ending_xand_y(&mut self, pair_xand_y: Option<&PairXAndY>) {
        self.reset_starting_and_ending_xand_y();
        let pair_xand_y = match pair_xand_y {
            None => return,
            Some(pair_xand_y) => pair_xand_y,
        };
        if !pair_xand_y.is_x_null() {
            self.starting_and_ending_x
                .set_index_double(0, pair_xand_y.get_first_x() as f64);
            self.starting_and_ending_x
                .set_index_double(1, pair_xand_y.get_second_x() as f64);
        }
        if !pair_xand_y.is_y_null() {
            self.starting_and_ending_y
                .set_index_double(0, pair_xand_y.get_first_y() as f64);
            self.starting_and_ending_y
                .set_index_double(1, pair_xand_y.get_second_y() as f64);
        }
    }

    /// Java `setVerySloppyMontage(boolean)`.
    pub fn set_very_sloppy_montage(&mut self, input: bool) {
        if input {
            self.sloppy_montage.set_boolean(false);
        } else {
            self.sloppy_montage.set_boolean(true);
        }
        self.very_sloppy_montage.set_boolean(input);
    }

    /// Java `setWeightForExpectedShifts(String)`.
    pub fn set_weight_for_expected_shifts(&mut self, input: Option<&str>) {
        self.weight_for_expected_shifts.set_string(input);
    }

    /// Java `resetWeightForExpectedShifts()`.
    pub fn reset_weight_for_expected_shifts(&mut self) {
        self.weight_for_expected_shifts.reset();
    }

    /// Java `setEMGridMapFilter(String)`.
    pub fn set_e_m_grid_map_filter(&mut self, input: Option<&str>) {
        self.e_m_grid_map_filter.set_string(input);
    }

    /// Java `resetEMGridMapFilter()`.
    pub fn reset_e_m_grid_map_filter(&mut self) {
        self.e_m_grid_map_filter.reset();
    }

    /// Java `setImagesAreBinned(Number)`.
    pub fn set_images_are_binned(&mut self, input: Option<Number>) {
        self.images_are_binned.set_number(input);
    }

    /// Java `setImageInputFile(File)`.
    ///
    /// BlendmontParam.java:494-497 resets on a null file and then dereferences it
    /// anyway (NullPointerException).  Fixed in translation: a null file only resets.
    pub fn set_image_input_file_file(&mut self, input: Option<&std::path::Path>) {
        let input = match input {
            None => {
                self.image_input_file.reset();
                return;
            }
            Some(input) => input,
        };
        let name = input
            .file_name()
            .map(|name| name.to_string_lossy().to_string())
            .unwrap_or_default();
        self.image_input_file.set(Some(&name));
    }

    /// Java `setImageInputFile(String)`.
    pub fn set_image_input_file(&mut self, input: Option<&str>) {
        self.image_input_file.set(input);
    }

    /// Java `setImageOutputFile(FileType, String, AxisType)`.  For setting the output
    /// file before the manager is set up.  Not for 3d find.
    pub fn set_image_output_file(
        &mut self,
        file_type: &Arc<FileType>,
        root_name: Option<&str>,
        axis_type: Option<AxisType>,
    ) {
        self.image_output_file_for_3d_find = false;
        // BlendmontParam.java:513-515 dereferences `manager.getBaseMetaData()` without a
        // null check.  Fixed in translation: with no metadata the style and extension
        // are passed as null, which `deriveFileName` accepts.
        let meta_data = self.manager.get_base_meta_data();
        let image_filename_style =
            meta_data.map(|meta_data| meta_data.base().get_image_filename_style());
        let raw_image_stack_extension =
            meta_data.and_then(|meta_data| meta_data.get_raw_image_stack_extension());
        self.image_output_file = file_type.derive_file_name(
            root_name,
            axis_type,
            Some(self.axis_id),
            image_filename_style,
            raw_image_stack_extension,
        );
        self.image_output_file_type = Some(Arc::clone(file_type));
    }

    /// Java `setImageOutputFileFor3dFind(FileType)`.  Set the output for for 3d find.
    pub fn set_image_output_file_for_3d_find(&mut self, file_type: &Arc<FileType>) {
        self.image_output_file_for_3d_find = true;
        self.image_output_file = file_type.get_file_name(Some(self.manager), Some(self.axis_id));
        self.image_output_file_type = Some(Arc::clone(file_type));
        self.set_process_name();
    }

    /// Java `setBlendmontState(ConstEtomoNumber)`.
    pub fn set_blendmont_state(&mut self, invalid_edge_functions: &ConstEtomoNumber) -> bool {
        self.set_blendmont_state_recreated(invalid_edge_functions, false)
    }

    /// Java `setBlendmontState(ConstEtomoNumber, boolean)`.  Sets the state of
    /// blendmont parameters based on the .edc and .xef files.  Returns true if blendmont
    /// needs to be run, false if blendmont does not need to be run.
    pub fn set_blendmont_state_recreated(
        &mut self,
        invalid_edge_functions: &ConstEtomoNumber,
        edge_functions_recreated: bool,
    ) -> bool {
        let manager = Some(self.manager);
        let axis_id = Some(self.axis_id);
        if self.mode == Mode::Undistort {
            if !self.image_output_file_for_3d_find {
                let file_type = Arc::clone(&file_type::CLASS.distortion_corrected_stack);
                self.image_output_file = file_type.get_file_name(manager, axis_id);
                self.image_output_file_type = Some(file_type);
                // imageOutputFile = datasetName + axisID.getExtension() +
                // DISTORTION_CORRECTED_STACK_EXTENSION;
            }
            self.just_undistort.set_boolean(true);
            return true;
        } else {
            self.just_undistort.set_boolean(false);
            if !self.image_output_file_for_3d_find {
                if self.mode == Mode::Xcorr {
                    // imageOutputFile = datasetName + axisID.getExtension() +
                    // BLENDMONT_STACK_EXTENSION;
                    let file_type = Arc::clone(&file_type::CLASS.xcorr_blend_output);
                    self.image_output_file = file_type.get_file_name(manager, axis_id);
                    self.image_output_file_type = Some(file_type);
                } else if self.mode == Mode::Preblend {
                    // imageOutputFile = datasetName + axisID.getExtension() + ".preali";
                    let file_type = Arc::clone(&file_type::CLASS.prealigned_stack);
                    self.image_output_file = file_type.get_file_name(manager, axis_id);
                    self.image_output_file_type = Some(file_type);
                } else if self.mode == Mode::Blend || self.mode == Mode::WholeTomogramSample {
                    // datasetName + axisID.getExtension() + ".ali";
                    let file_type = Arc::clone(&file_type::CLASS.aligned_stack);
                    self.image_output_file = file_type.get_file_name(manager, axis_id);
                    self.image_output_file_type = Some(file_type);
                }
                if self.mode == Mode::SerialSectionBlend {
                    self.sloppy_montage.set_boolean(true);
                    self.image_output_file_type =
                        Some(Arc::clone(&file_type::CLASS.preblend_output_mrc));
                }
            }
        }
        // `new File(manager.getPropertyUserDir(), name)`
        let dir = self.manager.get_property_user_dir();
        let in_dir = |name: String| -> PathBuf {
            match &dir {
                Some(dir) => PathBuf::from(dir).join(name),
                None => PathBuf::from(name),
            }
        };
        let dataset_name = self.dataset_name.as_deref().unwrap_or("null");
        let extension = self.axis_id.get_extension();
        let ecd_file = in_dir(format!("{dataset_name}{extension}.ecd"));
        let xef_file = in_dir(format!("{dataset_name}{extension}.xef"));
        let yef_file = in_dir(format!("{dataset_name}{extension}.yef"));
        // ReadInXcorrs is a checkbox in serial sections. It should have been already set.
        // Do not change it here.
        if self.mode != Mode::SerialSectionPreblend {
            // Read in xcorr output if it exists. Turn on for preblend and blend.
            // In serial sections since blend is based on preblend, then .ecd file file
            // must be there and we must use the same edge functions.
            self.read_in_xcorrs.set_boolean(
                self.mode == Mode::Preblend
                    || self.mode == Mode::Blend
                    || self.mode == Mode::SerialSectionBlend
                    || self.mode == Mode::WholeTomogramSample
                    || ecd_file.exists(),
            );
        }
        // Use existing edge functions, if they are up to date and valid. Turn on for
        // blend.  Robust fitting makes new edge functions, so don't turn this on if
        // robust fitting has changed.
        // If readInXcorrs is off, the ecdfile will change - so edge functions must be
        // updated.
        let last_modified = |file: &PathBuf| java_io_file_last_modified(&file.to_string_lossy());
        let old_edge_functions = self.read_in_xcorrs.is()
            && !edge_functions_recreated
            && (self.mode == Mode::Blend
                || self.mode == Mode::SerialSectionBlend
                || self.mode == Mode::WholeTomogramSample
                || self.mode == Mode::SerialSectionPreblend
                || (invalid_edge_functions.get_int() != etomo_state::TRUE_VALUE
                    && xef_file.exists()
                    && yef_file.exists()
                    && last_modified(&ecd_file) <= last_modified(&xef_file)
                    && last_modified(&ecd_file) <= last_modified(&yef_file)));
        self.old_edge_functions.set_boolean(old_edge_functions);
        if self.mode == Mode::Xcorr {
            // If xcorr output exists and the edge functions are up to date, then don't
            // run blendmont, as long as the blendmont output is more recent then the
            // stack.
            let stack_file = file_type::CLASS.raw_stack.get_file(manager, axis_id);
            let blend_file = in_dir(
                file_type::CLASS
                    .xcorr_blend_output
                    .get_file_name(manager, axis_id)
                    .unwrap_or_else(|| "null".to_string()),
                /*was: datasetName + axisID.getExtension() + BLENDMONT_STACK_EXTENSION*/
            );
            // `stackFile.lastModified()`: a null file is a NullPointerException in Java;
            // it counts as 0 (the value `File.lastModified` gives a missing file) here.
            let stack_last_modified = stack_file.as_ref().map(last_modified).unwrap_or(0);
            if self.read_in_xcorrs.is()
                && self.old_edge_functions.is()
                && blend_file.exists()
                && stack_last_modified < last_modified(&blend_file)
            {
                return false;
            }
        }
        true
    }

    /// Java `isVerySloppyMontage()`.
    pub fn is_very_sloppy_montage(&self) -> bool {
        self.very_sloppy_montage.is()
    }

    /// Java `isWeightForExpectedShifts()`.
    pub fn is_weight_for_expected_shifts(&self) -> bool {
        self.weight_for_expected_shifts.is()
    }

    /// Java `getWeightForExpectedShifts()`.
    pub fn get_weight_for_expected_shifts(&self) -> String {
        self.weight_for_expected_shifts.to_string()
    }

    /// Java `isEMGridMapFilter()`.
    pub fn is_e_m_grid_map_filter(&self) -> bool {
        self.e_m_grid_map_filter.is()
    }

    /// Java `getEMGridMapFilter()`.
    pub fn get_e_m_grid_map_filter(&self) -> String {
        self.e_m_grid_map_filter.to_string()
    }

    /// Java `getImageOutputFile()`.
    pub fn get_image_output_file(&self) -> Option<String> {
        self.image_output_file.clone()
    }

    /// Java `resetDistortionField()`.
    pub fn reset_distortion_field(&mut self) {
        self.distortion_field.reset();
    }

    /// Java `setDistortionField(String)`.
    pub fn set_distortion_field(&mut self, input: Option<&str>) {
        self.distortion_field.set(input);
    }

    /// Java private `setProcessName()`.  Call every time mode or
    /// overrideModeForImageOutputFile changes.
    fn set_process_name(&mut self) {
        self.process_name = BlendmontParam::get_process_name_for3d_find(
            self.mode,
            self.image_output_file_for_3d_find,
        );
    }

    /// Java static `getProcessName(Mode)`.
    pub fn get_process_name_for_mode(mode: Mode) -> ProcessName {
        BlendmontParam::get_process_name_for3d_find(mode, false)
    }

    /// Java private static `getProcessName(Mode, boolean)`.  The source's final
    /// `throw new IllegalArgumentException("mode=" + mode)` cannot be reached: `Mode`
    /// has only the values tested here.
    fn get_process_name_for3d_find(mode: Mode, for3d_find: bool) -> ProcessName {
        if mode == Mode::Preblend {
            return ProcessName::PREBLEND;
        }
        if mode == Mode::Blend {
            if for3d_find {
                return ProcessName::BLEND_3D_FIND;
            }
            return ProcessName::BLEND;
        }
        if mode == Mode::Blend3dFind {
            return ProcessName::BLEND_3D_FIND;
        }
        if mode == Mode::Undistort {
            return ProcessName::UNDISTORT;
        }
        if mode == Mode::Xcorr {
            return ProcessName::XCORR;
        }
        if mode == Mode::WholeTomogramSample {
            return ProcessName::BLEND;
        }
        if mode == Mode::SerialSectionPreblend {
            return ProcessName::PREBLEND;
        }
        // Mode::SerialSectionBlend
        ProcessName::BLEND
    }

    /// Java static `getDistortionCorrectedFile(BaseManager, String, AxisID)`.
    pub fn get_distortion_corrected_file(
        manager: &'static dyn BaseManager,
        working_dir: Option<&str>,
        axis_id: AxisID,
    ) -> PathBuf {
        let name = file_type::CLASS
            .distortion_corrected_stack
            .get_file_name(Some(manager), Some(axis_id))
            /* was: datasetName + axisID.getExtension() + DISTORTION_CORRECTED_STACK_EXTENSION*/
            .unwrap_or_else(|| "null".to_string());
        match working_dir {
            Some(working_dir) => PathBuf::from(working_dir).join(name),
            None => PathBuf::from(name),
        }
    }

    /// Java `isLinearInterpolation()`.
    pub fn is_linear_interpolation(&self) -> bool {
        self.interpolation_order.get_int() == LINEAR_INTERPOLATION_ORDER
    }

    /// Java `setLinearInterpolation(boolean)`.
    pub fn set_linear_interpolation(&mut self, linear_interpolation: bool) {
        if linear_interpolation {
            self.interpolation_order.set_int(LINEAR_INTERPOLATION_ORDER);
        } else {
            self.interpolation_order.reset();
        }
    }

    /// Java `setFiducialess(boolean)`.
    pub fn set_fiducialess(&mut self, input: bool) {
        self.fiducialess = input;
    }

    /// Java `isFiducialess()`.
    pub fn is_fiducialess(&self) -> bool {
        self.fiducialess
    }

    /// Java `setAdjustOrigin(boolean)`.
    pub fn set_adjust_origin(&mut self, input: bool) {
        self.adjust_origin.set_boolean(input);
    }

    /// Java `setBinByFactor(int)`.
    pub fn set_bin_by_factor_int(&mut self, bin_by_factor: i32) {
        self.bin_by_factor.set_int(bin_by_factor);
    }

    /// Java `setBinByFactor(Number)`.
    pub fn set_bin_by_factor(&mut self, input: Option<Number>) {
        self.bin_by_factor.set_number(input);
    }

    /// Java `getBinByFactor()`.
    pub fn get_bin_by_factor(&self) -> &ConstEtomoNumber {
        &self.bin_by_factor
    }

    /// Java `isReadInXcorrs()`.
    pub fn is_read_in_xcorrs(&self) -> bool {
        self.read_in_xcorrs.is()
    }

    /// Java `setReadInXcorrs(boolean)`.
    pub fn set_read_in_xcorrs(&mut self, input: bool) {
        self.read_in_xcorrs.set_boolean(input);
    }

    /// Java `isRobustFitCriterion()`.
    pub fn is_robust_fit_criterion(&self) -> bool {
        !self.robust_fit_criterion.is_null()
    }

    /// Java `getRobustFitCriterion()`.
    pub fn get_robust_fit_criterion(&self) -> String {
        self.robust_fit_criterion.to_string()
    }

    /// Java `setRobustFitCriterion(String)`.
    pub fn set_robust_fit_criterion(&mut self, input: Option<&str>) {
        self.robust_fit_criterion.set_string(input);
    }

    /// Java `resetRobustFitCriterion()`.
    pub fn reset_robust_fit_criterion(&mut self) {
        self.robust_fit_criterion.reset();
    }

    /// Java `setFixIntensityFromEdges(Integer)`.
    pub fn set_fix_intensity_from_edges(&mut self, input: Option<i32>) {
        self.fix_intensity_from_edges
            .set_number(input.map(Number::Integer));
    }

    /// Java `setSumPiecesForGradient(Integer)`.
    pub fn set_sum_pieces_for_gradient(&mut self, input: Option<i32>) {
        self.sum_pieces_for_gradient
            .set_number(input.map(Number::Integer));
    }

    /// Java `getOtherSumGradientFile()`.
    pub fn get_other_sum_gradient_file(&self) -> String {
        self.other_sum_gradient_file.to_string()
    }

    /// Java `setOtherSumGradientFile(String)`.
    pub fn set_other_sum_gradient_file(&mut self, input: Option<&str>) {
        self.other_sum_gradient_file.set(input);
    }

    /// Java `resetOtherSumGradientFile()`.
    pub fn reset_other_sum_gradient_file(&mut self) {
        self.other_sum_gradient_file.reset();
    }
}

impl CommandParam for BlendmontParam {
    /// Java `parseComScriptCommand(ComScriptCommand)`.
    fn parse_com_script_command(
        &mut self,
        script_command: &ComScriptCommand,
    ) -> Result<(), ParseComScriptError> {
        self.reset();
        self.read_in_xcorrs.parse(script_command)?;
        self.old_edge_functions.parse(script_command)?;
        self.interpolation_order.parse(script_command)?;
        self.just_undistort.parse(script_command)?;
        self.image_output_file = script_command.get_value(Some(IMAGE_OUTPUT_FILE_KEY))?;
        self.image_output_file_type = None;
        self.bin_by_factor.parse(script_command)?;
        self.starting_and_ending_x
            .validate_and_set_com_script(script_command)?;
        self.starting_and_ending_y
            .validate_and_set_com_script(script_command)?;
        self.adjust_origin.parse(script_command)?;
        self.distortion_field.parse(script_command)?;
        self.image_input_file.parse(script_command)?;
        self.piece_list_input.parse(script_command)?;
        self.root_name_for_edges.parse(script_command)?;
        self.images_are_binned.parse(script_command)?;
        self.sloppy_montage.parse(script_command)?;
        self.very_sloppy_montage.parse(script_command)?;
        self.weight_for_expected_shifts.parse(script_command)?;
        self.e_m_grid_map_filter.parse(script_command)?;
        self.robust_fit_criterion.parse(script_command)?;
        self.fill_value.parse(script_command)?;
        self.transform_file.parse(script_command)?;
        self.unaligned_starting_xand_y
            .validate_and_set_com_script(script_command)?;
        self.fix_intensity_from_edges.parse(script_command)?;
        self.sum_pieces_for_gradient.parse(script_command)?;
        self.other_sum_gradient_file.parse(script_command)?;
        Ok(())
    }

    /// Java `updateComScriptCommand(ComScriptCommand)`.
    fn update_com_script_command(
        &self,
        script_command: &mut ComScriptCommand,
    ) -> Result<(), BadComScriptException> {
        if self.from_scratch {
            script_command.use_keyword_value();
        }
        self.read_in_xcorrs.update_com_script(script_command);
        self.old_edge_functions.update_com_script(script_command);
        self.interpolation_order.update_com_script(script_command);
        self.just_undistort.update_com_script(script_command);
        script_command.set_value(
            Some(IMAGE_OUTPUT_FILE_KEY),
            self.image_output_file.as_deref(),
        );
        self.bin_by_factor.update_com_script(script_command);
        self.starting_and_ending_x
            .update_script_parameter(script_command);
        self.starting_and_ending_y
            .update_script_parameter(script_command);
        self.adjust_origin.update_com_script(script_command);
        self.distortion_field.update_com_script(script_command);
        self.image_input_file.update_com_script(script_command);
        self.piece_list_input.update_com_script(script_command);
        self.root_name_for_edges.update_com_script(script_command);
        self.images_are_binned.update_com_script(script_command);
        self.sloppy_montage.update_com_script(script_command);
        self.very_sloppy_montage.update_com_script(script_command);
        self.weight_for_expected_shifts
            .update_com_script(script_command);
        self.e_m_grid_map_filter.update_com_script(script_command);
        self.robust_fit_criterion.update_com_script(script_command);
        self.fill_value.update_com_script(script_command);
        self.transform_file.update_com_script(script_command);
        self.unaligned_starting_xand_y
            .update_script_parameter(script_command);
        self.fix_intensity_from_edges
            .update_com_script(script_command);
        self.sum_pieces_for_gradient
            .update_com_script(script_command);
        self.other_sum_gradient_file
            .update_com_script(script_command);
        Ok(())
    }

    /// Java `initializeDefaults()`.
    fn initialize_defaults(&mut self) {}
}

impl Command for BlendmontParam {
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
        if let Some(image_output_file_type) = &self.image_output_file_type {
            return Some(Arc::clone(image_output_file_type));
        }
        FileType::get_instance_from_manager(
            Some(self.manager),
            self.axis_id,
            true,
            true,
            self.image_output_file.as_deref(),
        )
    }

    /// Java `getOutputImageFileKey()`: the `FileType` is its own `FileKey`.
    fn get_output_image_file_key(&self) -> Option<FileKey> {
        self.image_output_file_type
            .as_ref()
            .map(|file_type| (**file_type).clone())
    }

    /// Java `getOutputImageFileType2()` (deprecated 3/15/2019).
    fn get_output_image_file_type2(&self) -> Option<Arc<FileType>> {
        if self.mode == Mode::WholeTomogramSample {
            // Handle tiltParam here so the user doesn't have to wait.
            // (Java reads `axisType` and never uses it.)
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
        if self.mode == Mode::WholeTomogramSample {
            // Handle tiltParam here so the user doesn't have to wait.
            let _axis_type = self
                .manager
                .get_base_meta_data()
                .map(|meta_data| meta_data.base().get_axis_type());
            return Some((*file_type::CLASS.tilt_output).clone());
        }
        None
    }

    /// Java `getCommandName()`.
    fn get_command_name(&self) -> Option<String> {
        Some(COMMAND_NAME.to_string())
    }

    /// Java `getCommandLine()`.
    fn get_command_line(&self) -> Option<String> {
        self.get_command()
    }

    /// Java `getCommand()`.
    fn get_command(&self) -> Option<String> {
        Some(self.process_name.get_comscript(self.axis_id))
    }

    /// Java `getCommandArray()`.
    fn get_command_array(&self) -> Option<Vec<String>> {
        Some(self.process_name.get_comscript_array(self.axis_id))
    }

    /// Java `getCommandMode()`.
    fn get_command_mode(&self) -> Option<&dyn CommandMode> {
        Some(&self.mode)
    }

    /// Java `isMessageReporter()`.
    fn is_message_reporter(&self) -> bool {
        false
    }

    /// Java `getCommandOutputFile()`.  `new File(dir, imageOutputFile)`; a null name is
    /// a NullPointerException in Java and `None` here.
    fn get_command_output_file(&self) -> Option<PathBuf> {
        let name = self.image_output_file.as_deref()?;
        match self.manager.get_property_user_dir() {
            Some(dir) => Some(PathBuf::from(dir).join(name)),
            None => Some(PathBuf::from(name)),
        }
    }

    /// Java `getCommandInputFile()`.
    fn get_command_input_file(&self) -> Option<PathBuf> {
        None
    }

    /// Java `getAxisID()`.
    fn get_axis_id(&self) -> AxisID {
        self.axis_id
    }

    /// Java `getProcessName()`.
    fn get_process_name(&self) -> Option<ProcessName> {
        Some(self.process_name)
    }

    /// Java `command instanceof ProcessDetails`: this is a `CommandDetails`.
    fn get_process_details(&self) -> Option<&dyn ProcessDetails> {
        Some(self)
    }
}

impl Loggable for BlendmontParam {
    /// Java `getName()`.
    fn get_name(&self) -> String {
        self.process_name.to_string()
    }

    /// Java `getLogMessage()`, which returns null; there is no message.
    fn get_log_message(&self) -> Result<Vec<Option<String>>, LoggableException> {
        Ok(Vec::new())
    }
}

/// For a field the source does not recognise, every getter throws
/// `IllegalArgumentException("field=" + field)` (BlendmontParam.java:692-753), which no
/// caller catches.  Fixed in translation: the value is unavailable (`None`).
impl ProcessDetails for BlendmontParam {
    /// Java `getIntValue(FieldInterface)`.
    fn get_int_value(&self, field: &dyn FieldInterface) -> Option<i32> {
        let field = field_interface::as_field::<Field>(field)?;
        if *field == Field::FixIntensityFromEdges {
            return Some(self.fix_intensity_from_edges.get_int());
        }
        if *field == Field::SumPiecesForGradient {
            return Some(self.sum_pieces_for_gradient.get_int());
        }
        None
    }

    /// Java `getIteratorElementList(FieldInterface)`.
    fn get_iterator_element_list(&self, _field: &dyn FieldInterface) -> Option<Vec<i32>> {
        None
    }

    /// Java `getBooleanValue(FieldInterface)`.
    fn get_boolean_value(&self, field: &dyn FieldInterface) -> Option<bool> {
        let field = field_interface::as_field::<Field>(field)?;
        if *field == Field::OldEdgeFunctions {
            return Some(self.old_edge_functions.is());
        }
        if *field == Field::Fiducialess {
            return Some(self.fiducialess);
        }
        if *field == Field::LinearInterpolation {
            return Some(self.is_linear_interpolation());
        }
        None
    }

    /// Java `getString(FieldInterface)`.
    fn get_string(&self, field: &dyn FieldInterface) -> Option<String> {
        let field = field_interface::as_field::<Field>(field)?;
        if *field == Field::UserSizeToOutputInXAndY {
            return Some(self.user_size_to_output_in_xand_y.clone());
        }
        if *field == Field::OtherSumGradientFile {
            return Some(self.other_sum_gradient_file.to_string());
        }
        None
    }

    /// Java `getStringArray(FieldInterface)`.
    fn get_string_array(&self, _field: &dyn FieldInterface) -> Option<Vec<String>> {
        None
    }

    /// Java `getHashtable(FieldInterface)`.
    fn get_hashtable(&self, _field: &dyn FieldInterface) -> Option<Vec<(String, String)>> {
        None
    }

    /// Java `getDoubleValue(FieldInterface)`.
    fn get_double_value(&self, field: &dyn FieldInterface) -> Option<f64> {
        let field = field_interface::as_field::<Field>(field)?;
        if *field == Field::RobustFitting {
            return Some(self.robust_fit_criterion.get_double());
        }
        None
    }

    /// Java `getEtomoNumber(FieldInterface)`.
    fn get_etomo_number(&self, field: &dyn FieldInterface) -> Option<ConstEtomoNumber> {
        let field = field_interface::as_field::<Field>(field)?;
        if *field == Field::ImageRotation {
            return Some((*self.image_rotation).clone());
        }
        None
    }

    /// Java `getIntKeyList(FieldInterface)`.
    fn get_int_key_list(&self, _field: &dyn FieldInterface) -> Option<Vec<(i32, String)>> {
        None
    }
}
