//! `IMOD/Etomo/src/etomo/comscript/Patchcrawl3DParam.java`.
//!
//! `Patchcrawl3DParam extends ConstPatchcrawl3DParam`: the superclass state is the
//! `base` field, reached through `Deref`/`DerefMut`.

use std::cell::RefCell;
use std::rc::Rc;

use super::bad_com_script_exception::BadComScriptException;
use super::com_script_command::ComScriptCommand;
use super::com_script_input_arg::ComScriptInputArg;
use super::command_param::{CommandParam, ParseComScriptError};
use super::const_patchcrawl3d_param::*;
use super::fortran_input_syntax_exception::FortranInputSyntaxException;
use super::matchorwarp_param::MatchorwarpParam;
use super::param_utilities;
use super::patchcrawl3d_pre_pip_param::Patchcrawl3DPrePIPParam;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::logic::combine_tool::CombineTool;
use crate::imod::etomo::r#type::const_etomo_number::java_lang_string_matches_whitespace;

/// Java `rcsid`.
pub const RCSID: &str = "$Id:";
/// Java `COMMAND`.
pub const COMMAND: &str = "corrsearch3d";

/// Java `Patchcrawl3DParam`.
pub struct Patchcrawl3DParam {
    /// Java superclass `ConstPatchcrawl3DParam` state.
    pub base: ConstPatchcrawl3DParam,
    /// Java `convertToPIP`.
    convert_to_pip: bool,
    /// Java `manager`.
    manager: &'static dyn BaseManager,
}

impl std::ops::Deref for Patchcrawl3DParam {
    type Target = ConstPatchcrawl3DParam;

    fn deref(&self) -> &ConstPatchcrawl3DParam {
        &self.base
    }
}

impl std::ops::DerefMut for Patchcrawl3DParam {
    fn deref_mut(&mut self) -> &mut ConstPatchcrawl3DParam {
        &mut self.base
    }
}

impl Patchcrawl3DParam {
    /// Java package-private `Patchcrawl3DParam(BaseManager)`.
    pub fn new(manager: &'static dyn BaseManager) -> Patchcrawl3DParam {
        Patchcrawl3DParam {
            base: ConstPatchcrawl3DParam::new(),
            convert_to_pip: false,
            manager,
        }
    }

    /// Java `load(Patchcrawl3DPrePIPParam)`.
    ///
    /// Java throws IllegalStateException when `convertToPIP` is false; the only
    /// caller sets it first.  The panic is that programming-error assertion.
    pub fn load(
        &mut self,
        param: &Patchcrawl3DPrePIPParam,
    ) -> Result<(), FortranInputSyntaxException> {
        if !self.convert_to_pip {
            panic!("java.lang.IllegalStateException: !convertToPIP in load()");
        }
        self.base.reset();
        let base = &mut self.base;
        // xPatchSize, yPatchSize, zPatchSize
        base.patch_size_xyz
            .set_index_double(0, param.get_x_patch_size() as f64);
        base.patch_size_xyz
            .set_index_double(1, param.get_y_patch_size() as f64);
        base.patch_size_xyz
            .set_index_double(2, param.get_z_patch_size() as f64);
        // nX, nY, nZ
        base.number_of_patches_xyz
            .set_index_double(0, param.get_nx() as f64);
        base.number_of_patches_xyz
            .set_index_double(1, param.get_ny() as f64);
        base.number_of_patches_xyz
            .set_index_double(2, param.get_nz() as f64);
        // xLow, xHigh
        base.x_min_and_max
            .set_index_double(0, param.get_x_low() as f64);
        base.x_min_and_max
            .set_index_double(1, param.get_x_high() as f64);
        // yLow, yHigh
        base.y_min_and_max
            .set_index_double(0, param.get_y_low() as f64);
        base.y_min_and_max
            .set_index_double(1, param.get_y_high() as f64);
        // zLow, zHigh
        base.z_min_and_max
            .set_index_double(0, param.get_z_low() as f64);
        base.z_min_and_max
            .set_index_double(1, param.get_z_high() as f64);
        // drop maxShift
        // fileA
        base.reference_file = param.get_file_a().map(str::to_owned);
        // fileB
        base.file_to_align = param.get_file_b().map(str::to_owned);
        // outputFile
        base.output_file = param.get_output_file().map(str::to_owned);
        // transformFile
        base.b_source_transform = param.get_transform_file().map(str::to_owned);
        // originalFileB
        base.b_source_or_size_xyz = param.get_original_file_b().map(str::to_owned);
        // borders
        let borders = param.get_borders_fortran_input_string();
        base.b_source_border_x_lo_hi
            .set_index_double(0, borders.get_int(0) as f64);
        base.b_source_border_x_lo_hi
            .set_index_double(1, borders.get_int(1) as f64);
        base.b_source_border_yz_lo_hi
            .set_index_double(0, borders.get_int(2) as f64);
        base.b_source_border_yz_lo_hi
            .set_index_double(1, borders.get_int(3) as f64);
        // boundaryModel
        base.region_model = param.get_boundary_model().map(str::to_owned);
        Ok(())
    }

    /// Java private `getInputArguments`.
    fn get_input_arguments(
        &self,
        script_command: &ComScriptCommand,
    ) -> Result<Vec<Rc<RefCell<ComScriptInputArg>>>, BadComScriptException> {
        let command = script_command.get_command();
        // Check to be sure that it is the right command
        if command != Some(COMMAND) && !(command == Some("patchcrawl3d") && self.convert_to_pip) {
            return Err(BadComScriptException::new(&format!(
                "Not a {COMMAND} command"
            )));
        }
        // Get the input arguments parameters to preserve the comments
        let input_args = script_command.get_input_arguments();
        Ok(input_args)
    }

    /// Java `setNX`.
    pub fn set_nx(&mut self, n_x: i32) {
        self.base
            .number_of_patches_xyz
            .set_index_double(X_INDEX, n_x as f64);
    }

    /// Java `setNY`.
    pub fn set_ny(&mut self, n_y: i32) {
        self.base
            .number_of_patches_xyz
            .set_index_double(Y_INDEX, n_y as f64);
    }

    /// Java `setNZ`.
    pub fn set_nz(&mut self, n_z: i32) {
        self.base
            .number_of_patches_xyz
            .set_index_double(Z_INDEX, n_z as f64);
    }

    /// Java `setXHigh`.
    pub fn set_x_high(&mut self, x_high: i32) {
        self.base.x_min_and_max.set_index_double(1, x_high as f64);
    }

    /// Java `setXLow`.
    pub fn set_x_low(&mut self, x_low: i32) {
        self.base.x_min_and_max.set_index_double(0, x_low as f64);
    }

    /// Java `setXPatchSize`.
    pub fn set_x_patch_size(&mut self, x_patch_size: i32) {
        self.base
            .patch_size_xyz
            .set_index_double(0, x_patch_size as f64);
    }

    /// Java `setYHigh`.
    pub fn set_y_high(&mut self, y_high: i32) {
        self.base.y_min_and_max.set_index_double(1, y_high as f64);
    }

    /// Java `setYLow`.
    pub fn set_y_low(&mut self, y_low: i32) {
        self.base.y_min_and_max.set_index_double(0, y_low as f64);
    }

    /// Java `setYPatchSize`.
    pub fn set_y_patch_size(&mut self, y_patch_size: i32) {
        self.base
            .patch_size_xyz
            .set_index_double(Y_INDEX, y_patch_size as f64);
    }

    /// Java `setZHigh`.
    pub fn set_z_high(&mut self, z_high: i32) {
        self.base.z_min_and_max.set_index_double(1, z_high as f64);
    }

    /// Java `setZLow`.
    pub fn set_z_low(&mut self, z_low: i32) {
        self.base.z_min_and_max.set_index_double(0, z_low as f64);
    }

    /// Java `setZPatchSize`.
    pub fn set_z_patch_size(&mut self, z_patch_size: i32) {
        self.base
            .patch_size_xyz
            .set_index_double(Z_INDEX, z_patch_size as f64);
    }

    /// Java `setUseBoundaryModel`.
    pub fn set_use_boundary_model(&mut self, use_boundary_model: bool) {
        if use_boundary_model {
            self.base.region_model = Some(MatchorwarpParam::get_default_patch_region_model());
        } else {
            self.base.region_model = Some(String::new());
        }
    }

    /// Java `setInitialShiftX`.  A null input (a NullPointerException in Java) is
    /// treated as blank.
    pub fn set_initial_shift_x(&mut self, initial_shift_x: Option<&str>) {
        if initial_shift_x.is_none_or(java_lang_string_matches_whitespace) {
            self.base.initial_shift_xyz.set_default_index(X_INDEX);
        } else {
            self.base
                .initial_shift_xyz
                .set_index_string(X_INDEX, initial_shift_x);
        }
    }

    /// Java `setInitialShiftY`.  A null input is treated as blank.
    pub fn set_initial_shift_y(&mut self, initial_shift_y: Option<&str>) {
        if initial_shift_y.is_none_or(java_lang_string_matches_whitespace) {
            self.base.initial_shift_xyz.set_default_index(Y_INDEX);
        } else {
            self.base
                .initial_shift_xyz
                .set_index_string(Y_INDEX, initial_shift_y);
        }
    }

    /// Java `setInitialShiftZ`.  A null input is treated as blank.
    pub fn set_initial_shift_z(&mut self, initial_shift_z: Option<&str>) {
        if initial_shift_z.is_none_or(java_lang_string_matches_whitespace) {
            self.base.initial_shift_xyz.set_default_index(Z_INDEX);
        } else {
            self.base
                .initial_shift_xyz
                .set_index_string(Z_INDEX, initial_shift_z);
        }
    }

    /// Java `setKernelSigma`.
    pub fn set_kernel_sigma(&mut self, kernel_sigma_active: bool, kernel_sigma: Option<&str>) {
        self.base.kernel_sigma.set_active(kernel_sigma_active);
        self.base.kernel_sigma.set_string(kernel_sigma);
    }

    /// Java static `getTitle`.
    pub fn get_title() -> String {
        COMMAND[0..1].to_uppercase() + &COMMAND[1..]
    }
}

impl CommandParam for Patchcrawl3DParam {
    /// Java `parseComScriptCommand`.
    fn parse_com_script_command(
        &mut self,
        script_command: &ComScriptCommand,
    ) -> Result<(), ParseComScriptError> {
        let _cmd_line_args = script_command.get_command_line_args();
        if !script_command.is_keyword_value_pairs() {
            self.convert_to_pip = true;
            let mut pre_pip_param = Patchcrawl3DPrePIPParam::new();
            pre_pip_param.parse_com_script_command(script_command)?;
            self.load(&pre_pip_param)?;
            return Ok(());
        }
        self.convert_to_pip = false;
        self.base.reset();
        // Java's try block: a NumberFormatException out of it becomes a
        // BadComScriptException carrying its message.
        let result = (|| -> Result<(), ParseComScriptError> {
            let base = &mut self.base;
            base.patch_size_xyz
                .validate_and_set_com_script(script_command)?;
            base.number_of_patches_xyz
                .validate_and_set_com_script(script_command)?;
            base.x_min_and_max
                .validate_and_set_com_script(script_command)?;
            base.y_min_and_max
                .validate_and_set_com_script(script_command)?;
            base.z_min_and_max
                .validate_and_set_com_script(script_command)?;
            if script_command.has_keyword(Some(REGION_MODEL_KEY))? {
                base.region_model = script_command.get_value(Some(REGION_MODEL_KEY))?;
            }
            base.initial_shift_xyz
                .validate_and_set_com_script(script_command)?;
            base.kernel_sigma.parse_set_active(script_command, true)?;
            base.invert_y_limits
                .get_mut()
                .unwrap()
                .parse(script_command)?;
            Ok(())
        })();
        match result {
            Err(ParseComScriptError::NumberFormat(message)) => {
                Err(BadComScriptException::new(&message).into())
            }
            other => other,
        }
    }

    /// Java `updateComScriptCommand`.
    fn update_com_script_command(
        &self,
        script_command: &mut ComScriptCommand,
    ) -> Result<(), BadComScriptException> {
        // get the input arguments from the command
        let _input_args = self.get_input_arguments(script_command)?;
        // Switch to keyword/value pairs
        script_command.use_keyword_value();
        let base = &self.base;
        base.patch_size_xyz.update_script_parameter(script_command);
        base.number_of_patches_xyz
            .update_script_parameter(script_command);
        base.x_min_and_max.update_script_parameter(script_command);
        base.y_min_and_max.update_script_parameter(script_command);
        base.z_min_and_max.update_script_parameter(script_command);
        param_utilities::update_script_parameter_string(
            script_command,
            Some(REGION_MODEL_KEY),
            base.region_model.as_deref(),
        )?;
        base.initial_shift_xyz
            .update_script_parameter(script_command);
        base.kernel_sigma.update_com_script(script_command);
        {
            let mut invert_y_limits = base.invert_y_limits.lock().unwrap();
            if !invert_y_limits.is() && CombineTool::is_invert_y_limits(self.manager) {
                invert_y_limits.set_boolean(true);
            }
            invert_y_limits.update_com_script(script_command);
        }
        if self.convert_to_pip {
            script_command.set_command(Some(COMMAND));
            param_utilities::update_script_parameter_string(
                script_command,
                Some(REFERENCE_FILE_KEY),
                base.reference_file.as_deref(),
            )?;
            param_utilities::update_script_parameter_string(
                script_command,
                Some(FILE_TO_ALIGN_KEY),
                base.file_to_align.as_deref(),
            )?;
            param_utilities::update_script_parameter_string(
                script_command,
                Some(OUTPUT_FILE_KEY),
                base.output_file.as_deref(),
            )?;
            param_utilities::update_script_parameter_string(
                script_command,
                Some(B_SOURCE_TRANSFORM_KEY),
                base.b_source_transform.as_deref(),
            )?;
            param_utilities::update_script_parameter_string(
                script_command,
                Some(B_SOURCE_OR_SIZE_XYZ_KEY),
                base.b_source_or_size_xyz.as_deref(),
            )?;
            base.b_source_border_x_lo_hi
                .update_script_parameter(script_command);
            base.b_source_border_yz_lo_hi
                .update_script_parameter(script_command);
        }
        Ok(())
    }

    /// Java `initializeDefaults`.
    fn initialize_defaults(&mut self) {}
}
