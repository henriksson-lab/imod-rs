//! `IMOD/Etomo/src/etomo/comscript/TrimvolParam.java`.
//!
//! Parameter list for trimvol.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use super::command::Command;
use super::command_details::CommandDetails;
use super::command_mode::CommandMode;
use super::field_interface::{self, FieldInterface};
use super::process_details::ProcessDetails;
use super::xy_param::XYParam;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::logic::trimvol_input_file_state::TrimvolInputFileState;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::const_etomo_number::{ConstEtomoNumber, INTEGER_NULL_VALUE};
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::r#type::file_key::FileKey;
use crate::imod::etomo::r#type::file_type::{self, FileType};
use crate::imod::etomo::r#type::image_output_format::ImageOutputFormat;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::string_parameter::StringParameter;
use crate::imod::etomo::r#type::tomogram_state::TomogramState;
use crate::imod::etomo::ui::swing::log_interface::{Loggable, LoggableException};
use crate::imod::etomo::util::mrc_header::MRCHeader;

/// Java private `VERSION`.
#[allow(dead_code)]
const VERSION: &str = "1.1";
/// Java private `VERSION_KEY`.
#[allow(dead_code)]
const VERSION_KEY: &str = "Version";

/// Java `PARAM_ID`.
pub const PARAM_ID: &str = "Trimvol";
/// Java `CONVERT_TO_BYTES`.
pub const CONVERT_TO_BYTES: &str = "ConvertToBytes";
/// Java `FIXED_SCALING`.
pub const FIXED_SCALING: &str = "FixedScaling";
/// Java `FLIPPED_VOLUME`.
pub const FLIPPED_VOLUME: &str = "FlippedVolume";
/// Java private `swapYZString`.
#[allow(dead_code)]
const SWAP_YZ_STRING: &str = "SwapYZ";
/// Java private `ROTATE_X_KEY`.
#[allow(dead_code)]
const ROTATE_X_KEY: &str = "RotateX";
/// Java `INPUT_FILE`.
pub const INPUT_FILE: &str = "InputFile";
/// Java `OUTPUT_FILE`.
pub const OUTPUT_FILE: &str = "OutputFile";

/// Java private `commandSize`.
const COMMAND_SIZE: usize = 4;
/// Java `commandName`.
pub const COMMAND_NAME: &str = "trimvol";

/// Java nested `TrimvolParam.Mode implements CommandMode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Java `Mode.POST_PROCESSING`.
    PostProcessing,
    /// Java `Mode.NAD`.
    Nad,
}

/// The Java class does not override `toString`, so `Object.toString` (class name plus
/// identity hash) is what the source prints; the constant's name stands in for the
/// hash here.
impl std::fmt::Display for Mode {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Mode::PostProcessing => formatter.write_str("POST_PROCESSING"),
            Mode::Nad => formatter.write_str("NAD"),
        }
    }
}

impl CommandMode for Mode {}

/// Java nested `TrimvolParam.Fields implements FieldInterface`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fields {
    /// Java `Fields.SWAP_YZ`.
    SwapYz,
    /// Java `Fields.ROTATE_X`.
    RotateX,
}

impl FieldInterface for Fields {}

/// Java final `TrimvolParam implements CommandDetails`.
pub struct TrimvolParam {
    x_min: EtomoNumber,

    // Updates done
    x_max: EtomoNumber,
    y_min: EtomoNumber,
    y_max: EtomoNumber,
    z_min: EtomoNumber,
    z_max: EtomoNumber,
    scale_xy_param: XYParam,
    // The default convertToBytes comes from from metadata.
    convert_to_bytes: bool,
    fixed_scaling: bool,
    flipped_volume: bool,
    section_scale_min: EtomoNumber,
    section_scale_max: EtomoNumber,
    fixed_scale_min: EtomoNumber,
    fixed_scale_max: EtomoNumber,
    format_of_output_file: StringParameter,
    swap_yz: bool,
    rotate_x: bool,
    input_file: String,
    output_file: String,
    /// Java `commandArray`, built by `getCommandArray`, which the `Command` trait
    /// calls through `&self`.
    command_array: Mutex<Option<Vec<String>>>,
    /// Java `axisID`: declared and never assigned, so always null.
    axis_id: Option<AxisID>,
    old_version: bool,
    keep_same_origin: bool,
    n_columns_changed: bool,
    n_rows_changed: bool,
    n_sections_changed: bool,
    old_flipped_coordinates: EtomoNumber,

    manager: &'static dyn BaseManager,
    /// Java `mode` (a `CommandMode`; only this class's `Mode` constants are compared).
    mode: Option<Mode>,
}

impl TrimvolParam {
    /// Java `TrimvolParam(BaseManager, CommandMode)`.
    pub fn new(manager: &'static dyn BaseManager, mode: Option<Mode>) -> TrimvolParam {
        TrimvolParam {
            x_min: EtomoNumber::new_with_name("XMin"),
            x_max: EtomoNumber::new_with_name("XMax"),
            y_min: EtomoNumber::new_with_name("YMin"),
            y_max: EtomoNumber::new_with_name("YMax"),
            z_min: EtomoNumber::new_with_name("ZMin"),
            z_max: EtomoNumber::new_with_name("ZMax"),
            scale_xy_param: XYParam::new("Scale"),
            convert_to_bytes: false,
            fixed_scaling: false,
            flipped_volume: false,
            section_scale_min: EtomoNumber::new_with_name("SectionScaleMin"),
            section_scale_max: EtomoNumber::new_with_name("SectionScaleMax"),
            fixed_scale_min: EtomoNumber::new_with_name("FixedScaleMin"),
            fixed_scale_max: EtomoNumber::new_with_name("FixedScaleMax"),
            format_of_output_file: StringParameter::new("-FormatOfOutputFile"),
            swap_yz: false,
            rotate_x: true,
            input_file: String::new(),
            output_file: String::new(),
            command_array: Mutex::new(None),
            axis_id: None,
            old_version: false,
            keep_same_origin: false,
            n_columns_changed: false,
            n_rows_changed: false,
            n_sections_changed: false,
            old_flipped_coordinates: EtomoNumber::new_with_name("-old"),
            manager,
            mode,
        }
    }

    /// Java static `convertIndexCoordsToImodCoords`.
    pub fn convert_index_coords_to_imod_coords(
        x_min: &mut EtomoNumber,
        x_max: &mut EtomoNumber,
        y_min: &mut EtomoNumber,
        y_max: &mut EtomoNumber,
    ) {
        // In the old version, scale x and y min and max had been converted to index
        // coords. In the current version, they should be imod coords.
        // shift X
        let mut min;
        let mut max;
        let mut shift;
        if !x_min.is_null() {
            min = x_min.get_int();
            max = x_max.get_int();
            shift = TrimvolParam::get_scale_shift(min, max);
            x_min.set_int(min.wrapping_add(shift));
            x_max.set_int(max.wrapping_add(shift));
        }
        // shift Y
        if !y_min.is_null() {
            min = y_min.get_int();
            max = y_max.get_int();
            shift = TrimvolParam::get_scale_shift(min, max);
            y_min.set_int(min.wrapping_add(shift));
            y_max.set_int(max.wrapping_add(shift));
        }
    }

    /// Java package-private static `getScaleShift`.  Attempt to convert scale min and
    /// max from index coords to imod coords and correct bug# 915.
    pub(crate) fn get_scale_shift(min: i32, _max: i32) -> i32 {
        // Possibilities:
        // min and max may be reduced by 1 or more
        // min and max may not be reduced at all
        // Assume min and max are reduced by 1, since that is the most likely
        // situation.
        let mut shift = 1;
        if min < 0 {
            // min and max where reduce by more then 1
            shift = (0i32).wrapping_sub(min).wrapping_add(1);
        }
        shift
    }

    /// Java `isConvertToBytes` (deprecated since 2/13/2019).
    pub fn is_convert_to_bytes(&self) -> bool {
        self.convert_to_bytes
    }

    /// Java `setConvertToBytes`.
    pub fn set_convert_to_bytes(&mut self, convert_to_bytes: bool) {
        self.convert_to_bytes = convert_to_bytes;
    }

    /// Java private `createCommand`.
    fn create_command(&self) {
        let options = self.gen_options();
        let mut command_array = vec![String::new(); options.len() + COMMAND_SIZE];
        // Do not use the -e flag for tcsh since David's scripts handle the failure
        // of commands and then report appropriately. The exception to this is the
        // com scripts which require the -e flag. RJG: 2003-11-06
        command_array[0] = "python".to_owned();
        command_array[1] = "-u".to_owned();
        // `EtomoDirector.INSTANCE.getPythonScriptPath() + commandName`: Java string
        // concatenation writes a null path as "null".
        let script_path = etomo_director::INSTANCE
            .get_python_script_path()
            .as_deref()
            .unwrap_or("null")
            .to_owned();
        command_array[2] = format!("{script_path}{COMMAND_NAME}");
        command_array[3] = "-PID".to_owned();
        for i in 0..options.len() {
            command_array[i + COMMAND_SIZE] = options[i].clone();
        }
        *self.command_array.lock().unwrap() = Some(command_array);
    }

    /// Java `genOptions`.  Get the command string specified by the current state.
    pub fn gen_options(&self) -> Vec<String> {
        let mut options: Vec<String> = Vec::new();
        // options.add("-P");

        // TODO add error checking and throw an exception if the parameters have not
        // been set
        if self.x_min.get_int() >= 0 && self.x_max.get_int() >= 0 {
            options.push("-x".to_owned());
            options.push(format!("{},{}", self.x_min, self.x_max));
        }
        if self.y_min.get_int() >= 0 && self.y_max.get_int() >= 0 {
            options.push("-y".to_owned());
            options.push(format!("{},{}", self.y_min, self.y_max));
        }
        if self.z_min.get_int() >= 0 && self.z_max.get_int() >= 0 {
            options.push("-z".to_owned());
            options.push(format!("{},{}", self.z_min, self.z_max));
        }
        if self.convert_to_bytes {
            if self.fixed_scaling {
                options.push("-c".to_owned());
                options.push(format!("{},{}", self.fixed_scale_min, self.fixed_scale_max));
            } else {
                options.push("-sz".to_owned());
                options.push(format!(
                    "{},{}",
                    self.section_scale_min, self.section_scale_max
                ));
                if !self.scale_xy_param.get_x_min().is_null()
                    && !self.scale_xy_param.get_x_max().is_null()
                {
                    options.push("-sx".to_owned());
                    options.push(format!(
                        "{},{}",
                        self.scale_xy_param.get_x_min(),
                        self.scale_xy_param.get_x_max()
                    ));
                }
                if !self.scale_xy_param.get_y_min().is_null()
                    && !self.scale_xy_param.get_y_max().is_null()
                {
                    options.push("-sy".to_owned());
                    options.push(format!(
                        "{},{}",
                        self.scale_xy_param.get_y_min(),
                        self.scale_xy_param.get_y_max()
                    ));
                }
            }
        }

        if self.flipped_volume {
            options.push("-f".to_owned());
        }

        if self.swap_yz {
            options.push("-yz".to_owned());
        }

        if self.rotate_x {
            options.push("-rx".to_owned());
        }
        if self.keep_same_origin {
            options.push("-k".to_owned());
        }
        if !self.old_flipped_coordinates.is_null() {
            options.push(self.old_flipped_coordinates.get_name().to_owned());
            options.push(self.old_flipped_coordinates.to_string());
        }
        if !self.format_of_output_file.is_empty() {
            options.push(self.format_of_output_file.get_name().to_owned());
            options.push(self.format_of_output_file.to_string());
        }
        // TODO check to see that filenames are apropriate
        options.push(self.input_file.clone());
        options.push(self.output_file.clone());

        options
    }

    /// Java `getFixedScaleMax`.
    pub fn get_fixed_scale_max(&self) -> &ConstEtomoNumber {
        &self.fixed_scale_max
    }

    /// Java `getFixedScaleMin`.
    pub fn get_fixed_scale_min(&self) -> &ConstEtomoNumber {
        &self.fixed_scale_min
    }

    /// Java `isFixedScaling`.
    pub fn is_fixed_scaling(&self) -> bool {
        self.fixed_scaling
    }

    /// Java `getSectionScaleMax`.
    pub fn get_section_scale_max(&self) -> &ConstEtomoNumber {
        &self.section_scale_max
    }

    /// Java `getSectionScaleMin`.
    pub fn get_section_scale_min(&self) -> &ConstEtomoNumber {
        &self.section_scale_min
    }

    /// Java `isSwapYZ`.
    pub fn is_swap_yz(&self) -> bool {
        self.swap_yz
    }

    /// Java `isRotateX`.
    pub fn is_rotate_x(&self) -> bool {
        self.rotate_x
    }

    /// Java `getXMax`.
    pub fn get_x_max(&self) -> i32 {
        self.x_max.get_int()
    }

    /// Java `getXMin`.
    pub fn get_x_min(&self) -> i32 {
        self.x_min.get_int()
    }

    /// Java `getYMax`.
    pub fn get_y_max(&self) -> i32 {
        self.y_max.get_int()
    }

    /// Java `getYMin`.
    pub fn get_y_min(&self) -> i32 {
        self.y_min.get_int()
    }

    /// Java `getScaleXYParam`.  The source hands out the mutable instance.
    pub fn get_scale_xy_param(&mut self) -> &mut XYParam {
        &mut self.scale_xy_param
    }

    /// Java `getZMax`.
    pub fn get_z_max(&self) -> i32 {
        self.z_max.get_int()
    }

    /// Java `getZMin`.
    pub fn get_z_min(&self) -> i32 {
        self.z_min.get_int()
    }

    /// Java `setFixedScaleMax`.
    pub fn set_fixed_scale_max(&mut self, fixed_scale_max: Option<&str>) -> &ConstEtomoNumber {
        self.fixed_scale_max.set_string(fixed_scale_max)
    }

    /// Java `setFixedScaleMin`.
    pub fn set_fixed_scale_min(&mut self, fixed_scale_min: Option<&str>) -> &ConstEtomoNumber {
        self.fixed_scale_min.set_string(fixed_scale_min)
    }

    /// Java `setKeepSameOrigin`.
    pub fn set_keep_same_origin(&mut self, input: bool) {
        self.keep_same_origin = input;
    }

    /// Java `setFixedScaling`.
    pub fn set_fixed_scaling(&mut self, fixed_scaling: bool) {
        self.fixed_scale_min.set_null_is_valid(!fixed_scaling);
        self.fixed_scale_max.set_null_is_valid(!fixed_scaling);
        self.section_scale_min.set_null_is_valid(fixed_scaling);
        self.section_scale_max.set_null_is_valid(fixed_scaling);
        self.fixed_scaling = fixed_scaling;
    }

    /// Java `setSectionScaleMax`.
    pub fn set_section_scale_max(&mut self, scale_section_max: Option<&str>) -> &ConstEtomoNumber {
        self.section_scale_max.set_string(scale_section_max)
    }

    /// Java `setSectionScaleMin`.
    pub fn set_section_scale_min(&mut self, scale_section_min: Option<&str>) -> &ConstEtomoNumber {
        self.section_scale_min.set_string(scale_section_min)
    }

    /// Java `setFlippedVolume`.
    pub fn set_flipped_volume(&mut self, input: bool) {
        self.flipped_volume = input;
    }

    /// Java `setSwapYZ`.
    pub fn set_swap_yz(&mut self, swap_yz: bool) {
        self.swap_yz = swap_yz;
    }

    /// Java `setRotateX`.
    pub fn set_rotate_x(&mut self, rotate_x: bool) {
        self.rotate_x = rotate_x;
    }

    /// Java `setXMax`.
    pub fn set_x_max(&mut self, x_max: Option<&str>) {
        self.x_max.set_string(x_max);
    }

    /// Java `setXMin`.
    pub fn set_x_min(&mut self, x_min: Option<&str>) {
        self.x_min.set_string(x_min);
    }

    /// Java `setYMax`.
    pub fn set_y_max(&mut self, y_max: Option<&str>) {
        self.y_max.set_string(y_max);
    }

    /// Java `setYMin`.
    pub fn set_y_min(&mut self, y_min: Option<&str>) {
        self.y_min.set_string(y_min);
    }

    /// Java `setZMax`.
    pub fn set_z_max(&mut self, z_max: Option<&str>) {
        self.z_max.set_string(z_max);
    }

    /// Java `setZMin`.
    pub fn set_z_min(&mut self, z_min: Option<&str>) {
        self.z_min.set_string(z_min);
    }

    /// Java `isNColumnsChanged`.
    pub fn is_n_columns_changed(&self) -> bool {
        self.n_columns_changed
    }

    /// Java `isNRowsChanged`.
    pub fn is_n_rows_changed(&self) -> bool {
        self.n_rows_changed
    }

    /// Java `isNSectionsChanged`.
    pub fn is_n_sections_changed(&self) -> bool {
        self.n_sections_changed
    }

    /// Java private `hasInputFileSizeChanged` (no caller in the source).
    #[allow(dead_code)]
    fn has_input_file_size_changed(
        &mut self,
        mrc_header: &MRCHeader,
        state: &TomogramState,
    ) -> bool {
        let mut changed = false;
        if !state.is_post_proc_trim_vol_input_n_columns_null()
            && mrc_header.get_n_columns() != state.get_post_proc_trim_vol_input_n_columns()
        {
            changed = true;
            self.n_columns_changed = true;
        } else {
            self.n_columns_changed = false;
        }
        if !state.is_post_proc_trim_vol_input_n_rows_null()
            && mrc_header.get_n_rows() != state.get_post_proc_trim_vol_input_n_rows()
        {
            changed = true;
            self.n_rows_changed = true;
        } else {
            self.n_rows_changed = false;
        }
        if !state.is_post_proc_trim_vol_input_n_sections_null()
            && mrc_header.get_n_sections() != state.get_post_proc_trim_vol_input_n_sections()
        {
            changed = true;
            self.n_sections_changed = true;
        } else {
            self.n_sections_changed = false;
        }
        changed
    }

    /// Java `setDefaultRange`.  Set the default range if the dialog is new
    /// (!dialogExists) or partially set the default range if the input tomogram size
    /// has changed.
    pub fn set_default_range(
        &mut self,
        input_file_state: &TrimvolInputFileState,
        dialog_exists: bool,
    ) {
        // Don't override existing values unless the size of the trimvol input file
        // has changed since the last time trimvol was run.
        if dialog_exists
            && self.x_min.get_int() != INTEGER_NULL_VALUE
            && !input_file_state.is_changed()
        {
            return;
        }
        // Refresh X and Y together. Refresh Z separately.
        // Make sure that the dialog is refreshed the first time the dialog is
        // displayed. Also fix any null values that may have appeared. This is
        // done because there was a bug which caused null values.
        if !dialog_exists
            || input_file_state.is_n_columns_changed()
            || input_file_state.is_n_rows_changed()
        {
            self.x_min.set_int(1);
            self.x_max.set_int(input_file_state.get_n_columns());
        } else {
            if self.x_min.is_null() {
                self.x_min.set_int(1);
            }
            if self.x_max.is_null() {
                self.x_max.set_int(input_file_state.get_n_columns());
            }
        }
        if !dialog_exists || input_file_state.is_n_rows_changed() {
            self.y_min.set_int(1);
            self.y_max.set_int(input_file_state.get_n_rows());
        } else {
            if self.y_min.is_null() {
                self.y_min.set_int(1);
            }
            if self.y_max.is_null() {
                self.y_max.set_int(input_file_state.get_n_rows());
            }
        }
        if !dialog_exists || input_file_state.is_n_sections_changed() {
            self.z_min.set_int(1);
            self.z_max.set_int(input_file_state.get_n_sections());
        } else {
            if self.z_min.is_null() {
                self.z_min.set_int(1);
            }
            if self.z_max.is_null() {
                self.z_max.set_int(input_file_state.get_n_sections());
            }
        }
        // zMax always contains the number of sections.
        let z_max = self.z_max.get_int();
        self.section_scale_min.set_int(z_max / 3);
        self.section_scale_max.set_int(z_max.wrapping_mul(2) / 3);
    }

    /// Java `getInputFileName()`.
    pub fn get_input_file_name(&self) -> &str {
        &self.input_file
    }

    /// Java static `getInputFileName(BaseManager, AxisType, String)`.
    pub fn get_input_file_name_for(
        manager: &'static dyn BaseManager,
        axis_type: AxisType,
        dataset_name: Option<&str>,
    ) -> Option<String> {
        if axis_type == AxisType::SingleAxis {
            return file_type::CLASS.tilt_output.get_file_name_with_axis_type(
                Some(manager),
                dataset_name,
                Some(axis_type),
                Some(AxisID::Only),
            );
        }
        file_type::CLASS
            .combined_volume
            .get_file_name_with_axis_type(
                Some(manager),
                dataset_name,
                Some(axis_type),
                Some(AxisID::Only),
            )
    }

    /// Java `setInputFileName(AxisType, String)`.  A null name from `FileType` would
    /// make the Java field null; it becomes an empty string here, which is what the
    /// field starts as.
    pub fn set_input_file_name_for(&mut self, axis_type: AxisType, dataset_name: Option<&str>) {
        self.input_file =
            TrimvolParam::get_input_file_name_for(self.manager, axis_type, dataset_name)
                .unwrap_or_default();
    }

    /// Java `setInputFileName(String)`.
    pub fn set_input_file_name(&mut self, file_name: &str) {
        self.input_file = file_name.to_owned();
    }

    /// Java `getOutputFileName()`.
    pub fn get_output_file_name(&self) -> &str {
        &self.output_file
    }

    /// Java static `getOutputFileName(BaseManager, AxisID)`.
    pub fn get_output_file_name_for(
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
    ) -> Option<String> {
        file_type::CLASS
            .trim_vol_output
            .get_file_name(Some(manager), Some(axis_id))
        // return datasetName + ".rec";
    }

    /// Java `setFormatOfOutputFile`.
    pub fn set_format_of_output_file(&mut self, image_output_format: Option<ImageOutputFormat>) {
        match image_output_format {
            None => self.format_of_output_file.reset(),
            Some(image_output_format) => self
                .format_of_output_file
                .set(Some(&image_output_format.to_string())),
        }
    }

    /// Java `setOldFlippedCoordinates(boolean, boolean)`.
    pub fn set_old_flipped_coordinates_scaling(
        &mut self,
        new_style_z: bool,
        scaling_new_style_z: bool,
    ) {
        if new_style_z && scaling_new_style_z {
            self.old_flipped_coordinates.reset();
        } else if !new_style_z && scaling_new_style_z {
            self.old_flipped_coordinates.set_int(1);
        } else if new_style_z && !scaling_new_style_z {
            self.old_flipped_coordinates.set_int(2);
        } else {
            self.old_flipped_coordinates.set_int(3);
        }
    }

    /// Java `setOldFlippedCoordinates(boolean)`.
    pub fn set_old_flipped_coordinates(&mut self, new_style_z: bool) {
        if new_style_z {
            self.old_flipped_coordinates.reset();
        } else {
            self.old_flipped_coordinates.set_int(1);
        }
    }

    /// Java `setOutputFileName`.
    pub fn set_output_file_name(&mut self, file: &str) {
        self.output_file = file.to_owned();
    }

    /// Java `equals(TrimvolParam)` (deprecated since 2/13/2019: not used and most
    /// likely not kept up to date).
    pub fn equals_trimvol_param(&self, trim: &TrimvolParam) -> bool {
        if !self.x_min.equals_int(trim.get_x_min()) {
            return false;
        }
        if !self.x_max.equals_int(trim.get_x_max()) {
            return false;
        }
        if !self.y_min.equals_int(trim.get_y_min()) {
            return false;
        }
        if !self.y_max.equals_int(trim.get_y_max()) {
            return false;
        }
        if !self.z_min.equals_int(trim.get_z_min()) {
            return false;
        }
        if !self.z_max.equals_int(trim.get_z_max()) {
            return false;
        }
        if self.convert_to_bytes != trim.is_convert_to_bytes() {
            return false;
        }
        if self.fixed_scaling != trim.is_fixed_scaling() {
            return false;
        }
        if !self
            .section_scale_min
            .equals_const_etomo_number(Some(&*trim.section_scale_min))
        {
            return false;
        }
        if !self
            .section_scale_max
            .equals_const_etomo_number(Some(&*trim.section_scale_max))
        {
            return false;
        }
        if !self
            .fixed_scale_min
            .equals_const_etomo_number(Some(&*trim.fixed_scale_min))
        {
            return false;
        }
        if !self
            .fixed_scale_max
            .equals_const_etomo_number(Some(&*trim.fixed_scale_max))
        {
            return false;
        }
        if self.swap_yz != trim.is_swap_yz() {
            return false;
        }
        if self.rotate_x != trim.is_rotate_x() {
            return false;
        }
        if self.input_file != trim.get_input_file_name()
            && (self.input_file == "\\S+" || trim.get_input_file_name() == "\\S+")
        {
            return false;
        }
        // Fixed in translation: TrimvolParam.java:854-855 compares the String
        // `outputFile` with `trim.getCommandOutputFile()`, a File, which is never equal,
        // so the source's first operand is always true.  The evident intent is the
        // other instance's output file name, compared here.
        if self.output_file != trim.get_output_file_name()
            && (self.output_file == "\\S+" || trim.get_output_file_name() == "\\S+")
        {
            return false;
        }
        if !self.scale_xy_param.equals(&trim.scale_xy_param) {
            return false;
        }
        true
    }
}

impl Command for TrimvolParam {
    /// Java `command instanceof ProcessDetails`: this class is a `CommandDetails`.
    fn get_process_details(&self) -> Option<&dyn ProcessDetails> {
        Some(self)
    }

    /// Java `getAxisID`.  The source's `axisID` field is never assigned, so it returns
    /// null; the Rust trait has no null, so `AxisID::Only` - the single-axis value a
    /// null axis ID stands for elsewhere in eTomo - is returned instead.
    fn get_axis_id(&self) -> AxisID {
        self.axis_id.unwrap_or(AxisID::Only)
    }

    fn get_command_mode(&self) -> Option<&dyn CommandMode> {
        None
    }

    fn get_process_name(&self) -> Option<ProcessName> {
        Some(ProcessName::TRIMVOL)
    }

    fn get_command(&self) -> Option<String> {
        Some(COMMAND_NAME.to_owned())
    }

    fn get_command_name(&self) -> Option<String> {
        Some(COMMAND_NAME.to_owned())
    }

    fn get_command_line(&self) -> Option<String> {
        let command_array = self.command_array.lock().unwrap();
        let command_array = match command_array.as_ref() {
            None => return Some(String::new()),
            Some(command_array) => command_array,
        };
        let mut buffer = String::new();
        for element in command_array {
            buffer.push_str(element);
            buffer.push(' ');
        }
        Some(buffer)
    }

    fn get_command_array(&self) -> Option<Vec<String>> {
        self.create_command();
        self.command_array.lock().unwrap().clone()
    }

    fn get_command_input_file(&self) -> Option<PathBuf> {
        None
    }

    fn get_command_output_file(&self) -> Option<PathBuf> {
        Some(PathBuf::from(&self.output_file))
    }

    /// Java `getOutputImageFileType` (deprecated 3/15/2019).
    fn get_output_image_file_type(&self) -> Option<Arc<FileType>> {
        if self.mode == Some(Mode::PostProcessing) {
            return Some(file_type::CLASS.trim_vol_output.clone());
        }
        if self.mode == Some(Mode::Nad) {
            return Some(file_type::CLASS.nad_test_input.clone());
        }
        None
    }

    fn get_output_image_file_key(&self) -> Option<FileKey> {
        if self.mode == Some(Mode::PostProcessing) {
            return Some(FileKey::clone(&file_type::CLASS.trim_vol_output));
        }
        if self.mode == Some(Mode::Nad) {
            return Some(FileKey::clone(&file_type::CLASS.nad_test_input));
        }
        None
    }

    /// Java `getOutputImageFileType2` (deprecated 3/15/2019).
    fn get_output_image_file_type2(&self) -> Option<Arc<FileType>> {
        None
    }

    fn get_output_image_file_key2(&self) -> Option<FileKey> {
        None
    }

    fn is_message_reporter(&self) -> bool {
        false
    }

    fn get_subcommand_process_name(&self) -> Option<String> {
        None
    }

    fn get_subcommand_details(&self) -> Option<&dyn CommandDetails> {
        None
    }
}

/// Every getter but `getBooleanValue` throws `IllegalArgumentException("field=" +
/// field)` in the source, as does `getBooleanValue` for a field it does not know; those
/// return `None` here.
impl ProcessDetails for TrimvolParam {
    fn get_int_value(&self, _field: &dyn FieldInterface) -> Option<i32> {
        None
    }

    fn get_boolean_value(&self, field: &dyn FieldInterface) -> Option<bool> {
        match field_interface::as_field::<Fields>(field) {
            Some(Fields::SwapYz) => Some(self.swap_yz),
            Some(Fields::RotateX) => Some(self.rotate_x),
            None => None,
        }
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

impl Loggable for TrimvolParam {
    /// Java `getName`.
    fn get_name(&self) -> String {
        COMMAND_NAME.to_owned()
    }

    /// Java `getLogMessage`, which returns null.
    fn get_log_message(&self) -> Result<Vec<Option<String>>, LoggableException> {
        Ok(Vec::new())
    }
}
