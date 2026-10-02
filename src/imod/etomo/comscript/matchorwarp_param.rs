//! `IMOD/Etomo/src/etomo/comscript/MatchorwarpParam.java`.

use super::bad_com_script_exception::BadComScriptException;
use super::com_script_command::ComScriptCommand;
use super::command_param::{CommandParam, ParseComScriptError};
use super::const_matchorwarp_param::ConstMatchorwarpParam;
use crate::imod::etomo::r#type::const_etomo_number::Type;
use crate::imod::etomo::r#type::etomo_boolean2::EtomoBoolean2;
use crate::imod::etomo::r#type::script_parameter::ScriptParameter;
use crate::imod::etomo::r#type::string_parameter::StringParameter;
use crate::imod::etomo::util::dataset_files;

/// Java final `MatchorwarpParam`.
#[derive(Clone, Debug)]
pub struct MatchorwarpParam {
    /// Java `sizeXYZorVolume`.
    size_xyz_or_volume: StringParameter,
    /// Java `refineLimit`.
    refine_limit: ScriptParameter,
    /// Java `residualFile`.
    residual_file: StringParameter,
    /// Java `vectorModel`.
    vector_model: StringParameter,
    /// Java `clipPlaneBoxSize`.
    clip_plane_box_size: ScriptParameter,
    /// Java `warpLimits`.
    warp_limits: StringParameter,
    /// Java `modelFile`.
    model_file: StringParameter,
    /// Java `patchFile`.
    patch_file: StringParameter,
    /// Java `solveFile`.
    solve_file: StringParameter,
    /// Java `refineFile`.
    refine_file: StringParameter,
    /// Java `inverseFile`.
    inverse_file: StringParameter,
    /// Java `warpFile`.
    warp_file: StringParameter,
    /// Java `temporaryDirectory`.
    temporary_directory: StringParameter,
    /// Java `xLowerExclude`.
    x_lower_exclude: ScriptParameter,
    /// Java `xUpperExclude`.
    x_upper_exclude: ScriptParameter,
    /// Java `zLowerExclude`.
    z_lower_exclude: ScriptParameter,
    /// Java `zUpperExclude`.
    z_upper_exclude: ScriptParameter,
    /// Java `trialMode`.
    trial_mode: EtomoBoolean2,
    /// Java `inputVolume`.
    input_volume: StringParameter,
    /// Java `outputVolume`.
    output_volume: StringParameter,
    /// Java `linearInterpolation`.
    linear_interpolation: EtomoBoolean2,
    /// Java `structureCriteria`.
    structure_criteria: StringParameter,
    /// Java `extentToFit`.
    extent_to_fit: StringParameter,
}

impl Default for MatchorwarpParam {
    fn default() -> MatchorwarpParam {
        MatchorwarpParam::new()
    }
}

impl MatchorwarpParam {
    /// Java package-private `MatchorwarpParam()`.
    pub fn new() -> MatchorwarpParam {
        MatchorwarpParam {
            size_xyz_or_volume: StringParameter::new("SizeXYZorVolume"),
            refine_limit: ScriptParameter::new_with_type_and_name(Type::Double, "RefineLimit"),
            residual_file: StringParameter::new("ResidualFile"),
            vector_model: StringParameter::new("VectorModel"),
            clip_plane_box_size: ScriptParameter::new_with_name("ClipPlaneBoxSize"),
            warp_limits: StringParameter::new("WarpLimits"),
            model_file: StringParameter::new("ModelFile"),
            patch_file: StringParameter::new("PatchFile"),
            solve_file: StringParameter::new("SolveFile"),
            refine_file: StringParameter::new("RefineFile"),
            inverse_file: StringParameter::new("InverseFile"),
            warp_file: StringParameter::new("WarpFile"),
            temporary_directory: StringParameter::new("TemporaryDirectory"),
            x_lower_exclude: ScriptParameter::new_with_name("XLowerExclude"),
            x_upper_exclude: ScriptParameter::new_with_name("XUpperExclude"),
            z_lower_exclude: ScriptParameter::new_with_name("ZLowerExclude"),
            z_upper_exclude: ScriptParameter::new_with_name("ZUpperExclude"),
            trial_mode: EtomoBoolean2::new_with_name("TrialMode"),
            input_volume: StringParameter::new("InputVolume"),
            output_volume: StringParameter::new("OutputVolume"),
            linear_interpolation: EtomoBoolean2::new_with_name("LinearInterpolation"),
            structure_criteria: StringParameter::new("StructureCriteria"),
            extent_to_fit: StringParameter::new("ExtentToFit"),
        }
    }

    /// Java `parseComScriptCommandForBackwardsCompatibility`.
    ///
    /// Java indexes `cmdLineArgs` without bounds checks: an option-taking flag as
    /// the last examined argument, a chain of matching flags in one pass, or fewer
    /// than two arguments throws ArrayIndexOutOfBoundsException, and a null
    /// argument throws NullPointerException.  Fixed in translation: an index past
    /// the end reads as a null argument (which does not match any flag and resets
    /// a parameter when set), a missing argument array is empty, and with fewer than
    /// two arguments the input and output volumes are left unset.
    pub fn parse_com_script_command_for_backwards_compatibility(
        &mut self,
        script_command: &ComScriptCommand,
    ) -> Result<(), ParseComScriptError> {
        // TODO error checking - throw exceptions for bad syntax
        let cmd_line_args = script_command.get_command_line_args().unwrap_or_default();
        let len = cmd_line_args.len();
        let arg = |i: usize| cmd_line_args.get(i).and_then(|arg| arg.as_deref());
        let starts_with =
            |i: usize, prefix: &str| -> bool { arg(i).is_some_and(|arg| arg.starts_with(prefix)) };
        let mut i = 0usize;
        while i + 2 < len {
            if starts_with(i, "-siz") {
                i += 1;
                self.size_xyz_or_volume.set(arg(i));
            }
            if starts_with(i, "-refinel") {
                i += 1;
                self.refine_limit.set_string(arg(i));
            }
            if starts_with(i, "-res") {
                i += 1;
                self.residual_file.set(arg(i));
            }
            if starts_with(i, "-vec") {
                i += 1;
                self.vector_model.set(arg(i));
            }
            if starts_with(i, "-cli") {
                i += 1;
                self.clip_plane_box_size.set_string(arg(i));
            }
            if starts_with(i, "-warpl") {
                i += 1;
                self.warp_limits.set(arg(i));
            }
            if starts_with(i, "-mod") {
                i += 1;
                self.model_file.set(arg(i));
            }
            if starts_with(i, "-pat") {
                i += 1;
                self.patch_file.set(arg(i));
            }
            if starts_with(i, "-sol") {
                i += 1;
                self.solve_file.set(arg(i));
            }
            if starts_with(i, "-refinef") {
                i += 1;
                self.refine_file.set(arg(i));
            }
            if starts_with(i, "-inv") {
                i += 1;
                self.inverse_file.set(arg(i));
            }
            if starts_with(i, "-warpf") {
                i += 1;
                self.warp_file.set(arg(i));
            }
            if starts_with(i, "-tem") {
                i += 1;
                self.temporary_directory.set(arg(i));
            }
            if starts_with(i, "-xlo") {
                i += 1;
                self.x_lower_exclude.set_string(arg(i));
            }
            if starts_with(i, "-xup") {
                i += 1;
                self.x_upper_exclude.set_string(arg(i));
            }
            if starts_with(i, "-zlo") {
                i += 1;
                self.z_lower_exclude.set_string(arg(i));
            }
            if starts_with(i, "-zup") {
                i += 1;
                self.z_upper_exclude.set_string(arg(i));
            }
            if starts_with(i, "-lin") {
                i += 1;
                self.linear_interpolation.set_boolean(true);
            }
            if starts_with(i, "-tri") {
                self.trial_mode.set_boolean(true);
            }
            if starts_with(i, "-str") {
                i += 1;
                self.structure_criteria.set(arg(i));
            }
            if starts_with(i, "-ext") {
                i += 1;
                self.extent_to_fit.set(arg(i));
            }
            i += 1;
        }
        if len >= 2 {
            self.input_volume.set(arg(len - 2));
            self.output_volume.set(arg(len - 1));
        }
        // Backwards compatibility (before 3.8.25):
        // Update so that the user can look at the .resid file.
        if self.residual_file.is_empty()
            && self.vector_model.is_empty()
            && self.clip_plane_box_size.is_null()
        {
            self.residual_file.set(Some("patch.resid"));
            self.vector_model
                .set(Some(dataset_files::PATCH_VECTOR_MODEL));
            self.clip_plane_box_size.set_int(600);
        }
        Ok(())
    }

    /// Java `setModelFile`.
    pub fn set_model_file(&mut self, input: Option<&str>) {
        self.model_file.set(input);
    }

    /// Java `setDefaultModelFile`.
    pub fn set_default_model_file(&mut self) {
        self.model_file
            .set(Some(&MatchorwarpParam::get_default_patch_region_model()));
    }

    /// Java `setRefineLimit`.
    pub fn set_refine_limit(&mut self, input: Option<&str>) {
        self.refine_limit.set_string(input);
    }

    /// Java `setWarpLimits`.
    pub fn set_warp_limits(&mut self, input: Option<&str>) {
        self.warp_limits.set(input);
    }

    /// Java `setXLowerExclude(String)`.
    pub fn set_x_lower_exclude_string(&mut self, input: Option<&str>) {
        self.x_lower_exclude.set_string(input);
    }

    /// Java `setXLowerExclude(int)`.
    pub fn set_x_lower_exclude_int(&mut self, input: i32) {
        self.x_lower_exclude.set_int(input);
    }

    /// Java `resetXLowerExclude`.
    pub fn reset_x_lower_exclude(&mut self) {
        self.x_lower_exclude.reset();
    }

    /// Java `setXUpperExclude(String)`.
    pub fn set_x_upper_exclude_string(&mut self, input: Option<&str>) {
        self.x_upper_exclude.set_string(input);
    }

    /// Java `setXUpperExclude(int)`.
    pub fn set_x_upper_exclude_int(&mut self, input: i32) {
        self.x_upper_exclude.set_int(input);
    }

    /// Java `resetXUpperExclude`.
    pub fn reset_x_upper_exclude(&mut self) {
        self.x_upper_exclude.reset();
    }

    /// Java `setZLowerExclude(String)`.
    pub fn set_z_lower_exclude_string(&mut self, input: Option<&str>) {
        self.z_lower_exclude.set_string(input);
    }

    /// Java `setZLowerExclude(int)`.
    pub fn set_z_lower_exclude_int(&mut self, input: i32) {
        self.z_lower_exclude.set_int(input);
    }

    /// Java `resetZLowerExclude`.
    pub fn reset_z_lower_exclude(&mut self) {
        self.z_lower_exclude.reset();
    }

    /// Java `setZUpperExclude(String)`.
    pub fn set_z_upper_exclude_string(&mut self, input: Option<&str>) {
        self.z_upper_exclude.set_string(input);
    }

    /// Java `setZUpperExclude(int)`.
    pub fn set_z_upper_exclude_int(&mut self, input: i32) {
        self.z_upper_exclude.set_int(input);
    }

    /// Java `resetZUpperExclude`.
    pub fn reset_z_upper_exclude(&mut self) {
        self.z_upper_exclude.reset();
    }

    /// Java `setTrialMode`.
    pub fn set_trial_mode(&mut self, input: bool) {
        self.trial_mode.set_boolean(input);
    }

    /// Java `setLinearInterpolation`.
    pub fn set_linear_interpolation(&mut self, input: bool) {
        self.linear_interpolation.set_boolean(input);
    }

    /// Java static `getDefaultPatchRegionModel`.
    pub fn get_default_patch_region_model() -> String {
        "patch_region.mod".to_owned()
    }
}

impl CommandParam for MatchorwarpParam {
    /// Java `parseComScriptCommand`.
    fn parse_com_script_command(
        &mut self,
        script_command: &ComScriptCommand,
    ) -> Result<(), ParseComScriptError> {
        self.initialize_defaults();
        if !script_command.is_keyword_value_pairs() {
            self.parse_com_script_command_for_backwards_compatibility(script_command)?;
        } else {
            self.size_xyz_or_volume.parse(script_command)?;
            self.refine_limit.parse(script_command)?;
            self.residual_file.parse(script_command)?;
            self.vector_model.parse(script_command)?;
            self.warp_limits.parse(script_command)?;
            self.model_file.parse(script_command)?;
            self.patch_file.parse(script_command)?;
            self.solve_file.parse(script_command)?;
            self.refine_file.parse(script_command)?;
            self.inverse_file.parse(script_command)?;
            self.warp_file.parse(script_command)?;
            self.temporary_directory.parse(script_command)?;
            self.x_lower_exclude.parse(script_command)?;
            self.x_upper_exclude.parse(script_command)?;
            self.z_lower_exclude.parse(script_command)?;
            self.z_upper_exclude.parse(script_command)?;
            self.trial_mode.parse(script_command)?;
            self.input_volume.parse(script_command)?;
            self.output_volume.parse(script_command)?;
            self.linear_interpolation.parse(script_command)?;
            self.structure_criteria.parse(script_command)?;
            self.extent_to_fit.parse(script_command)?;
        }
        Ok(())
    }

    /// Java `updateComScriptCommand`.
    ///
    /// Java stores the default vector model into the `vectorModel` field when it
    /// is empty; this method takes `&self`, so the default goes into a copy that
    /// is written to the script.  Nothing reads the field except this method, which
    /// makes the same substitution every time.
    fn update_com_script_command(
        &self,
        script_command: &mut ComScriptCommand,
    ) -> Result<(), BadComScriptException> {
        let mut vector_model = self.vector_model.clone();
        if vector_model.is_empty() {
            vector_model.set(Some(dataset_files::PATCH_VECTOR_MODEL));
        }
        script_command.use_keyword_value();
        self.size_xyz_or_volume.update_com_script(script_command);
        self.refine_limit.update_com_script(script_command);
        self.residual_file.update_com_script(script_command);
        vector_model.update_com_script(script_command);
        self.warp_limits.update_com_script(script_command);
        self.model_file.update_com_script(script_command);
        self.patch_file.update_com_script(script_command);
        self.solve_file.update_com_script(script_command);
        self.refine_file.update_com_script(script_command);
        self.inverse_file.update_com_script(script_command);
        self.warp_file.update_com_script(script_command);
        self.temporary_directory.update_com_script(script_command);
        self.x_lower_exclude.update_com_script(script_command);
        self.x_upper_exclude.update_com_script(script_command);
        self.z_lower_exclude.update_com_script(script_command);
        self.z_upper_exclude.update_com_script(script_command);
        self.trial_mode.update_com_script(script_command);
        self.input_volume.update_com_script(script_command);
        self.output_volume.update_com_script(script_command);
        self.linear_interpolation.update_com_script(script_command);
        self.structure_criteria.update_com_script(script_command);
        self.extent_to_fit.update_com_script(script_command);
        Ok(())
    }

    /// Java `initializeDefaults`.
    fn initialize_defaults(&mut self) {
        self.size_xyz_or_volume.reset();
        self.refine_limit.reset();
        self.residual_file.reset();
        self.clip_plane_box_size.reset();
        self.warp_limits.reset();
        self.model_file.reset();
        self.patch_file.reset();
        self.solve_file.reset();
        self.refine_file.reset();
        self.inverse_file.reset();
        self.warp_file.reset();
        self.temporary_directory.reset();
        self.x_lower_exclude.reset();
        self.x_upper_exclude.reset();
        self.z_lower_exclude.reset();
        self.z_upper_exclude.reset();
        self.trial_mode.reset();
        self.input_volume.reset();
        self.output_volume.reset();
        self.linear_interpolation.reset();
        self.structure_criteria.reset();
        self.extent_to_fit.reset();
    }
}

impl ConstMatchorwarpParam for MatchorwarpParam {
    fn is_use_model_file(&self) -> bool {
        !self.model_file.is_empty()
    }

    fn get_refine_limit(&self) -> String {
        self.refine_limit.to_string()
    }

    fn get_warp_limits(&self) -> String {
        self.warp_limits.to_string()
    }

    fn get_x_lower_exclude(&self) -> i32 {
        self.x_lower_exclude.get_int()
    }

    fn is_x_lower_exclude_set(&self) -> bool {
        !self.x_lower_exclude.is_null()
    }

    fn get_x_upper_exclude(&self) -> i32 {
        self.x_upper_exclude.get_int()
    }

    fn is_x_upper_exclude_set(&self) -> bool {
        !self.x_upper_exclude.is_null()
    }

    fn get_z_lower_exclude(&self) -> i32 {
        self.z_lower_exclude.get_int()
    }

    fn is_z_lower_exclude_set(&self) -> bool {
        !self.z_lower_exclude.is_null()
    }

    fn get_z_upper_exclude(&self) -> i32 {
        self.z_upper_exclude.get_int()
    }

    fn is_z_upper_exclude_set(&self) -> bool {
        !self.z_upper_exclude.is_null()
    }

    fn is_linear_interpolation(&self) -> bool {
        self.linear_interpolation.is()
    }
}
