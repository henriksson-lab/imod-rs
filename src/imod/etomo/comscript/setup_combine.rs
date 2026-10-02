//! `IMOD/Etomo/src/etomo/comscript/SetupCombine.java`.
//!
//! Runs the `setupcombine` script through a `SystemProgram`.

use std::sync::Mutex;

use super::combine_params;
use super::const_combine_params::ConstCombineParams;
use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::process::base_process_manager::{
    BaseProcessManager, SystemProcessException,
};
use crate::imod::etomo::process::process_messages::ProcessMessages;
use crate::imod::etomo::process::system_program::SystemProgram;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::combine_patch_size::CombinePatchSize;
use crate::imod::etomo::r#type::const_meta_data::ConstMetaData;
use crate::imod::etomo::r#type::fiducial_match::FiducialMatch;
use crate::imod::etomo::r#type::match_mode::MatchMode;
use crate::imod::etomo::r#type::meta_data::MetaData;
use crate::imod::etomo::util::dataset_files;

/// Java private `COMMAND`.
const COMMAND: &str = "setupcombine";

/// Java private static `patchSizes`, guarded as the Java guards it with
/// `synchronized (EtomoDirector.INSTANCE)`.
static PATCH_SIZES: Mutex<Option<Vec<String>>> = Mutex::new(None);

/// Java `EtomoDirector.INSTANCE.getPythonScriptPath()`, written "null" by Java
/// string concatenation when it is null.
fn python_script_path() -> String {
    etomo_director::INSTANCE.get_python_script_path()
        .as_deref()
        .unwrap_or("null")
        .to_owned()
}

/// Java final `SetupCombine`.
pub struct SetupCombine {
    command: Vec<String>,
    setupcombine: SystemProgram,
    /// Java `exitValue`: declared and never used (`run` has a local).
    #[allow(dead_code)]
    exit_value: i32,
    meta_data: &'static MetaData,
    #[allow(dead_code)]
    debug: bool,
    #[allow(dead_code)]
    manager: &'static dyn BaseManager,
    match_mode: Option<MatchMode>,
    transfer: bool,
}

impl SetupCombine {
    /// Java private `SetupCombine(ApplicationManager, boolean)`.
    fn new(
        manager: &'static ApplicationManager,
        only_make_combine_com: bool,
    ) -> Result<SetupCombine, SystemProcessException> {
        let meta_data = manager.get_const_meta_data();
        let debug = etomo_director::ARGUMENTS.lock().unwrap().is_debug();
        // Create a new SystemProgram object for setupcombine, set the
        // working directory and stdin array.
        // Do not use the -e flag for tcsh since David's scripts handle the failure
        // of commands and then report appropriately. The exception to this is the
        // com scripts which require the -e flag. RJG: 2003-11-06
        let mut command = Vec::new();
        command.push("python".to_string());
        command.push("-u".to_string());
        command.push(format!("{}{COMMAND}", python_script_path()));
        let mut match_mode = None;
        let mut transfer = false;
        if !only_make_combine_com {
            SetupCombine::gen_options(&mut command, meta_data, &mut match_mode, &mut transfer)?;
        } else {
            SetupCombine::gen_options_only_make_combine_com(&mut command);
        }
        let mut command_array = Vec::with_capacity(command.len());
        for i in 0..command.len() {
            command_array.push(command[i].clone());
        }
        let setupcombine = SystemProgram::new_array(
            Some(manager as &'static dyn BaseManager),
            manager.get_property_user_dir(),
            Some(command_array),
            AxisID::Only,
        );
        Ok(SetupCombine {
            command,
            setupcombine,
            exit_value: 0,
            meta_data,
            debug,
            manager,
            match_mode,
            transfer,
        })
    }

    /// Java static `getInstance(ApplicationManager)`.
    pub fn get_instance(
        manager: &'static ApplicationManager,
    ) -> Result<SetupCombine, SystemProcessException> {
        SetupCombine::new(manager, false)
    }

    /// Java static `getOnlyMakeCombineComInstance(ApplicationManager)`.
    pub fn get_only_make_combine_com_instance(
        manager: &'static ApplicationManager,
    ) -> Result<SetupCombine, SystemProcessException> {
        SetupCombine::new(manager, true)
    }

    /// Java static `getInfoOnPatchSizes`.
    ///
    /// Java passes a null `AxisID` to `getCommandOutput`; the translated
    /// `getCommandOutput` requires one, and `AxisID.ONLY` stands in for it.
    pub fn get_info_on_patch_sizes() -> Option<Vec<String>> {
        let mut patch_sizes = PATCH_SIZES.lock().unwrap();
        if patch_sizes.is_some() {
            return patch_sizes.clone();
        }
        *patch_sizes = BaseProcessManager::get_command_output(
            vec![
                "python".to_string(),
                "-u".to_string(),
                format!("{}{COMMAND}", python_script_path()),
                "-info".to_string(),
            ],
            AxisID::Only,
            None,
        );
        patch_sizes.clone()
    }

    /// Java `getMatchMode`.
    pub fn get_match_mode(&self) -> Option<MatchMode> {
        self.match_mode
    }

    /// Java `isTransfer`.
    pub fn is_transfer(&self) -> bool {
        self.transfer
    }

    /// Java private `genOptions`.  It runs from the constructor in the Java, before
    /// the `SystemProgram` is made, so it takes the fields it writes.
    fn gen_options(
        command: &mut Vec<String>,
        meta_data: &MetaData,
        match_mode_field: &mut Option<MatchMode>,
        transfer: &mut bool,
    ) -> Result<(), SystemProcessException> {
        let combine_params = meta_data.get_combine_params();
        *match_mode_field = combine_params.get_match_mode();
        let match_mode = *match_mode_field;
        let match_list_to;
        let match_list_from;
        let fiducial_match = combine_params.get_fiducial_match();
        let patch_size = combine_params.get_patch_size(false);
        let auto_patch_final_size = combine_params.get_patch_size(true);
        // dataset name
        command.push("-name".to_string());
        command.push(meta_data.get_dataset_name());
        if meta_data.is_orig_scope_template() {
            command.push("-change".to_string());
            command.push(meta_data.get_orig_scope_template());
        }
        if meta_data.is_orig_system_template() {
            command.push("-change".to_string());
            command.push(meta_data.get_orig_system_template());
        }
        if meta_data.is_orig_user_template() {
            command.push("-change".to_string());
            command.push(meta_data.get_orig_user_template());
        }
        // transfer fid coord file and point list
        if combine_params.is_transfer() {
            *transfer = true;
            command.push("-transfer".to_string());
            command.push(dataset_files::get_transfer_fid_coord_file_name());
            let use_list = combine_params.get_use_list();
            // `useList.matches("\\s*")`
            if !use_list
                .chars()
                .all(|c| matches!(c, ' ' | '\t' | '\n' | '\u{0B}' | '\u{0C}' | '\r'))
            {
                command.push("-uselist".to_string());
                command.push(combine_params.get_use_list());
            }
        }
        // matching relationship
        if match_mode == Some(MatchMode::AToB) {
            command.push("-atob".to_string());
        }
        // corresponding lists
        if !*transfer {
            if match_mode == Some(MatchMode::AToB) {
                match_list_to = combine_params.get_fiducial_match_list_b();
                match_list_from = combine_params.get_fiducial_match_list_a();
            } else {
                match_list_to = combine_params.get_fiducial_match_list_a();
                match_list_from = combine_params.get_fiducial_match_list_b();
            }
            // points lists
            if match_list_to != "" {
                command.push("-tolist".to_string());
                command.push(match_list_to);
                command.push("-fromlist".to_string());
                command.push(match_list_from);
            }
        }
        // fiducial surfaces / use model
        if let Some(fiducial_match) = fiducial_match
            && fiducial_match != FiducialMatch::NotSet
        {
            command.push("-surfaces".to_string());
            // `getOption()` is null only for NOT_SET.
            command.push(fiducial_match.get_option().unwrap().to_string());
        }
        // patch sizes
        if let Some(patch_size) = patch_size {
            command.push("-patchsize".to_string());
            if patch_size != CombinePatchSize::Custom {
                command.push(patch_size.get_option().to_string());
            } else {
                // Java adds the string even when it is null.
                command.push(
                    combine_params
                        .get_patch_size_xyz(false)
                        .unwrap_or_else(|| "null".to_string()),
                );
            }
        }
        if let Some(auto_patch_final_size) = auto_patch_final_size {
            command.push("-AutoPatchFinalSize".to_string());
            if auto_patch_final_size != CombinePatchSize::Custom {
                command.push(auto_patch_final_size.get_option().to_string());
            } else {
                command.push(
                    combine_params
                        .get_patch_size_xyz(true)
                        .unwrap_or_else(|| "null".to_string()),
                );
            }
        }
        if combine_params.is_extra_residual_targets_set() {
            command.push("-ExtraResidualTargets".to_string());
            command.push(
                combine_params
                    .get_extra_residual_targets()
                    .unwrap_or_else(|| "null".to_string()),
            );
        }
        let mut min = combine_params.get_patch_x_min();
        let mut max = combine_params.get_patch_x_max();
        if min != 0 || max != 0 {
            command.push("-xlimits".to_string());
            command.push(format!("{min},{max}"));
        }
        min = combine_params.get_patch_y_min();
        max = combine_params.get_patch_y_max();
        if min != 0 || max != 0 {
            command.push("-ylimits".to_string());
            command.push(format!("{min},{max}"));
        }
        let mut number = combine_params.get_patch_z_min();
        if number.is_null() {
            return Err(SystemProcessException(format!(
                "{} is required.",
                combine_params::PATCH_Z_MIN_LABEL
            )));
        }
        min = combine_params.get_patch_z_min().get_int();
        number = combine_params.get_patch_z_max();
        if number.is_null() {
            return Err(SystemProcessException(format!(
                "{} is required.",
                combine_params::PATCH_Z_MAX_LABEL
            )));
        }
        max = combine_params.get_patch_z_max().get_int();
        if min != 0 || max != 0 {
            command.push("-zlimits".to_string());
            command.push(format!("{min},{max}"));
        }
        // patch region model
        if combine_params.use_patch_region_model() {
            command.push("-regionmod".to_string());
            command.push(combine_params.get_patch_region_model());
        }
        if combine_params.is_low_from_both_radius_set() {
            command.push("-LowFromBothRadius".to_string());
            command.push(
                combine_params
                    .get_low_from_both_radius()
                    .unwrap_or_else(|| "null".to_string()),
            );
        }
        if combine_params.is_wedge_reduction_fraction_set() {
            command.push("-WedgeReductionFraction".to_string());
            command.push(
                combine_params
                    .get_wedge_reduction_fraction()
                    .unwrap_or_else(|| "null".to_string()),
            );
        }
        if combine_params.is_temp_directory_set() {
            command.push("-tempdir".to_string());
            command.push(combine_params.get_temp_directory());
            if combine_params.get_manual_cleanup() {
                command.push("-noclean".to_string());
            }
        }
        command.push("-StackExtension".to_string());
        // `getRawImageStackExtension().toString()` throws NullPointerException for an
        // unrecognised stored extension; "null" is passed instead.
        command.push(
            meta_data
                .get_raw_image_stack_extension()
                .map(|extension| extension.to_string())
                .unwrap_or("null".to_string()),
        );
        command.push("-NamingStyle".to_string());
        command.push(meta_data.get_image_filename_style().to_string());
        Ok(())
    }

    /// Java private `genOptionsOnlyMakeCombineCom`.
    fn gen_options_only_make_combine_com(command: &mut Vec<String>) {
        command.push("-OnlyMakeCombineCom".to_string());
    }

    /// Java private `genStdInputSequence` (deprecated, never called).  Generate
    /// the standard input sequence.
    #[allow(dead_code)]
    fn gen_std_input_sequence(&mut self) {
        let mut combine_params = self.meta_data.get_combine_params();
        self.match_mode = combine_params.get_match_mode();
        // writing the script, so set the script match mode to be the same as the
        // screen match mode
        combine_params.set_match_mode(self.match_mode);
        let mut temp_std_input: Vec<String> = Vec::with_capacity(15);
        // compile the input sequence to setupcombine
        // Dataset name
        temp_std_input.push(self.meta_data.get_dataset_name());
        // Matching relationship
        // SetupCombine.java:473 and :482 compare `getFiducialMatchListA() != ""`
        // by reference, which is always true for the StringList's built string.
        // Fixed in translation: the lists are compared by value, as intended.
        if self.match_mode.is_none() || self.match_mode == Some(MatchMode::BToA) {
            temp_std_input.push("a".to_string());
            if combine_params.get_fiducial_match_list_a() != "" {
                temp_std_input.push(combine_params.get_fiducial_match_list_a());
                temp_std_input.push(combine_params.get_fiducial_match_list_b());
            } else {
                temp_std_input.push(String::new());
            }
        } else {
            temp_std_input.push("b".to_string());
            if combine_params.get_fiducial_match_list_b() != "" {
                temp_std_input.push(combine_params.get_fiducial_match_list_b());
                temp_std_input.push(combine_params.get_fiducial_match_list_a());
            } else {
                temp_std_input.push(String::new());
            }
        }
        // Fiducial surfaces / use model
        if combine_params.get_fiducial_match() == Some(FiducialMatch::BothSides) {
            temp_std_input.push("2".to_string());
        }
        if combine_params.get_fiducial_match() == Some(FiducialMatch::OneSide) {
            temp_std_input.push("1".to_string());
        }
        if combine_params.get_fiducial_match() == Some(FiducialMatch::OneSideInverted) {
            temp_std_input.push("-1".to_string());
        }
        if combine_params.get_fiducial_match() == Some(FiducialMatch::UseModel) {
            temp_std_input.push("0".to_string());
        }
        if combine_params.get_fiducial_match() == Some(FiducialMatch::UseModelOnly) {
            temp_std_input.push("-2".to_string());
        }
        // Patch sizes
        if combine_params.get_patch_size(false) == Some(CombinePatchSize::Large) {
            temp_std_input.push("l".to_string());
        }
        if combine_params.get_patch_size(false) == Some(CombinePatchSize::Medium) {
            temp_std_input.push("m".to_string());
        }
        if combine_params.get_patch_size(false) == Some(CombinePatchSize::Small) {
            temp_std_input.push("s".to_string());
        }
        temp_std_input.push(combine_params.get_patch_x_min().to_string());
        temp_std_input.push(combine_params.get_patch_x_max().to_string());
        temp_std_input.push(combine_params.get_patch_y_min().to_string());
        temp_std_input.push(combine_params.get_patch_y_max().to_string());
        temp_std_input.push(combine_params.get_patch_z_min().to_string());
        temp_std_input.push(combine_params.get_patch_z_max().to_string());
        temp_std_input.push(combine_params.get_patch_region_model());
        temp_std_input.push(combine_params.get_temp_directory());
        if combine_params.is_temp_directory_set() {
            if combine_params.get_manual_cleanup() {
                temp_std_input.push("y".to_string());
            } else {
                temp_std_input.push("n".to_string());
            }
        }
        //
        // Copy the temporary stdInput to the real stdInput to get the number
        // of array elements correct
        let line_count = temp_std_input.len();
        let mut std_input = Vec::with_capacity(line_count);
        for i in 0..line_count {
            std_input.push(temp_std_input[i].clone());
        }
        self.setupcombine.set_std_input(Some(std_input));
    }

    /// Java `run() throws IOException`.  Execute the script and return its exit
    /// value.
    pub fn run(&self) -> i32 {
        let exit_value;
        // Execute the script
        self.setupcombine.run();
        exit_value = self.setupcombine.get_exit_value();
        exit_value
    }

    /// Java `getStdError`.
    pub fn get_std_error(&self) -> Option<Vec<String>> {
        self.setupcombine.get_std_error()
    }

    /// Java final `getProcessMessages`.
    pub fn get_process_messages(&self) -> std::sync::MutexGuard<'_, ProcessMessages> {
        self.setupcombine.get_process_messages()
    }

    /// Java `command` list, as the constructor built it.
    pub fn get_command(&self) -> &[String] {
        &self.command
    }
}
