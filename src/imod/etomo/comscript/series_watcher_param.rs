//! `IMOD/Etomo/src/etomo/comscript/SeriesWatcherParam.java`.
//!
//! Parameters for the serieswatcher application.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::bad_com_script_exception::BadComScriptException;
use super::com_script_command::ComScriptCommand;
use super::command::Command;
use super::command_details::CommandDetails;
use super::command_mode::CommandMode;
use super::command_param::{CommandParam, ParseComScriptError};
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::Type;
use crate::imod::etomo::r#type::etomo_boolean2::EtomoBoolean2;
use crate::imod::etomo::r#type::file_key::FileKey;
use crate::imod::etomo::r#type::file_type::{self, FileType};
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::script_parameter::ScriptParameter;
use crate::imod::etomo::r#type::string_parameter::StringParameter;

/// Java `WATCH_DIRECTORY_KEY`.
pub const WATCH_DIRECTORY_KEY: &str = "WatchDirectory";
/// Java `DUAL_AXIS_KEY`.
pub const DUAL_AXIS_KEY: &str = "DualAxis";
/// Java private `DUAL_AXIS_FALSE`.
const DUAL_AXIS_FALSE: i32 = 0;
/// Java private `DUAL_AXIS_TRUE`.
const DUAL_AXIS_TRUE: i32 = 1;
/// Java `TWO_SURFACES_KEY`.
pub const TWO_SURFACES_KEY: &str = "TwoSurfaces";
/// Java private `TWO_SURFACES_FALSE`.
const TWO_SURFACES_FALSE: i32 = 0;
/// Java private `TWO_SURFACES_TRUE`.
const TWO_SURFACES_TRUE: i32 = 1;
/// Java `MINIMUM_TILT_RANGE_KEY`.
pub const MINIMUM_TILT_RANGE_KEY: &str = "MinimumTiltRange";
/// Java `MINIMUM_NUMBER_OF_VIEWS_KEY`.
pub const MINIMUM_NUMBER_OF_VIEWS_KEY: &str = "MinimumNumberOfViews";
/// Java `MINIMUM_AGE_OF_STACKS_KEY`.
pub const MINIMUM_AGE_OF_STACKS_KEY: &str = "MinimumAgeOfStacks";
/// Java `MINIMUM_TILT_RANGE_DEFAULT`.
pub const MINIMUM_TILT_RANGE_DEFAULT: f64 = 40.0;
/// Java `MINIMUM_NUMBER_OF_VIEWS_DEFAULT`.
pub const MINIMUM_NUMBER_OF_VIEWS_DEFAULT: i32 = 12;
/// Java `MINUMUM_AGE_OF_STACKS_DEFAULT`.
pub const MINUMUM_AGE_OF_STACKS_DEFAULT: f64 = 300.0;

/// Java `CHECK_FILE_VALUE = FileType.SERIES_WATCHER_CHECK_FILE`.
pub fn check_file_value() -> Arc<FileType> {
    file_type::CLASS.series_watcher_check_file.clone()
}

/// Java final `SeriesWatcherParam`.
pub struct SeriesWatcherParam {
    watch_directory: StringParameter,
    dual_axis: ScriptParameter,
    two_surfaces: ScriptParameter,
    match_pattern_or_ext: StringParameter,
    minimum_tilt_range: ScriptParameter,
    minimum_number_of_views: ScriptParameter,
    minimum_age_of_stacks: ScriptParameter,
    etomo_project_root: StringParameter,
    parallel_runs: ScriptParameter,
    debug_mode: Option<EtomoBoolean2>,
    manager: &'static dyn BaseManager,
    axis_id: AxisID,
}

impl SeriesWatcherParam {
    /// Java `SeriesWatcherParam(BaseManager, AxisID)`.
    pub fn new(manager: &'static dyn BaseManager, axis_id: AxisID) -> SeriesWatcherParam {
        let debug_mode = if etomo_director::ARGUMENTS.lock().unwrap().is_debug() {
            Some(EtomoBoolean2::new_with_name("DebugMode"))
        } else {
            None
        };
        SeriesWatcherParam {
            watch_directory: StringParameter::new(WATCH_DIRECTORY_KEY),
            dual_axis: ScriptParameter::new_with_type_and_name(Type::Integer, DUAL_AXIS_KEY),
            two_surfaces: ScriptParameter::new_with_type_and_name(Type::Integer, TWO_SURFACES_KEY),
            match_pattern_or_ext: StringParameter::new("MatchPatternOrExt"),
            minimum_tilt_range: ScriptParameter::new_with_type_and_name(
                Type::Double,
                MINIMUM_TILT_RANGE_KEY,
            ),
            minimum_number_of_views: ScriptParameter::new_with_type_and_name(
                Type::Integer,
                MINIMUM_NUMBER_OF_VIEWS_KEY,
            ),
            minimum_age_of_stacks: ScriptParameter::new_with_type_and_name(
                Type::Double,
                MINIMUM_AGE_OF_STACKS_KEY,
            ),
            etomo_project_root: StringParameter::new("EtomoProjectRoot"),
            parallel_runs: ScriptParameter::new_with_type_and_name(Type::Integer, "ParallelRuns"),
            debug_mode,
            manager,
            axis_id,
        }
    }

    /// Java `setWatchDirectory(File)`.
    pub fn set_watch_directory(&mut self, file: Option<&Path>) {
        self.watch_directory.set_file(file);
    }

    /// Java `setMatchPatternOrExt`.
    pub fn set_match_pattern_or_ext(&mut self, input: Option<&str>) {
        self.match_pattern_or_ext.set(input);
    }

    /// Java `setEtomoProjectRoot`.
    pub fn set_etomo_project_root(&mut self, input: Option<&str>) {
        self.etomo_project_root.set(input);
    }

    /// Java `setParallelRuns`.
    pub fn set_parallel_runs(&mut self, input: Option<&str>) {
        self.parallel_runs.set_string(input);
    }

    /// Java `resetParallelRuns`.
    pub fn reset_parallel_runs(&mut self) {
        self.parallel_runs.reset();
    }

    /// Java `setMinimumTiltRange`.
    pub fn set_minimum_tilt_range(&mut self, input: Option<&str>) {
        self.minimum_tilt_range.set_string(input);
    }

    /// Java `setMinimumNumberOfViews`.
    pub fn set_minimum_number_of_views(&mut self, input: Option<&str>) {
        self.minimum_number_of_views.set_string(input);
    }

    /// Java `setMinimumAgeOfStacks`.
    pub fn set_minimum_age_of_stacks(&mut self, input: Option<&str>) {
        self.minimum_age_of_stacks.set_string(input);
    }

    /// Java `setDualAxis`.
    pub fn set_dual_axis(&mut self, bool: bool) {
        if bool {
            self.dual_axis.set_int(DUAL_AXIS_TRUE);
        } else {
            self.dual_axis.set_int(DUAL_AXIS_FALSE);
        }
    }

    /// Java `setTwoSurfaces`.
    pub fn set_two_surfaces(&mut self, bool: bool) {
        if bool {
            self.two_surfaces.set_int(TWO_SURFACES_TRUE);
        } else {
            self.two_surfaces.set_int(TWO_SURFACES_FALSE);
        }
    }

    /// Java `isWatchDirectorySet`.
    pub fn is_watch_directory_set(&self) -> bool {
        !self.watch_directory.is_empty()
    }

    /// Java `getWatchDirectory`.
    pub fn get_watch_directory(&self) -> String {
        self.watch_directory.to_string()
    }

    /// Java `isMinimumTiltRangeSet`.
    pub fn is_minimum_tilt_range_set(&self) -> bool {
        !self.minimum_tilt_range.is_null()
    }

    /// Java `isMinimumNumberOfViewsSet`.
    pub fn is_minimum_number_of_views_set(&self) -> bool {
        !self.minimum_number_of_views.is_null()
    }

    /// Java `isMinimumAgeOfStacksSet`.
    pub fn is_minimum_age_of_stacks_set(&self) -> bool {
        !self.minimum_age_of_stacks.is_null()
    }

    /// Java `getMinimumTiltRange`.
    pub fn get_minimum_tilt_range(&self) -> String {
        self.minimum_tilt_range.to_string()
    }

    /// Java `getMinimumNumberOfViews`.
    pub fn get_minimum_number_of_views(&self) -> String {
        self.minimum_number_of_views.to_string()
    }

    /// Java `getMinimumAgeOfStacks`.
    pub fn get_minimum_age_of_stacks(&self) -> String {
        self.minimum_age_of_stacks.to_string()
    }

    /// Java `isDualAxisSet`.
    pub fn is_dual_axis_set(&self) -> bool {
        !self.dual_axis.is_null()
    }

    /// Java `isDualAxis`.
    pub fn is_dual_axis(&self) -> bool {
        if self.dual_axis.is_null() || self.dual_axis.equals_int(DUAL_AXIS_FALSE) {
            return false;
        }
        true
    }

    /// Java `isTwoSurfacesSet`.
    pub fn is_two_surfaces_set(&self) -> bool {
        !self.two_surfaces.is_null()
    }

    /// Java `isTwoSurfaces`.  The source compares against `DUAL_AXIS_FALSE`,
    /// which has the same value (0) as `TWO_SURFACES_FALSE`.
    pub fn is_two_surfaces(&self) -> bool {
        if self.two_surfaces.is_null() || self.two_surfaces.equals_int(DUAL_AXIS_FALSE) {
            return false;
        }
        true
    }
}

impl CommandParam for SeriesWatcherParam {
    /// Java `initializeDefaults`.
    fn initialize_defaults(&mut self) {
        if let Some(debug_mode) = &mut self.debug_mode {
            debug_mode.set_boolean(true);
        }
        self.watch_directory.reset();
        self.dual_axis.set_int(DUAL_AXIS_FALSE);
        self.two_surfaces.set_int(TWO_SURFACES_FALSE);
        self.match_pattern_or_ext.reset();
        self.minimum_tilt_range.reset();
        self.minimum_number_of_views.reset();
        self.minimum_age_of_stacks.reset();
        self.etomo_project_root.reset();
        self.parallel_runs.reset();
    }

    /// Java `parseComScriptCommand`.
    fn parse_com_script_command(
        &mut self,
        script_command: &ComScriptCommand,
    ) -> Result<(), ParseComScriptError> {
        self.initialize_defaults();
        self.watch_directory.parse(script_command)?;
        self.dual_axis.parse(script_command)?;
        self.two_surfaces.parse(script_command)?;
        // MatchPatternOrExt is derived from fields stored in metadata.
        self.minimum_tilt_range.parse(script_command)?;
        self.minimum_number_of_views.parse(script_command)?;
        self.minimum_age_of_stacks.parse(script_command)?;
        // EtomoProjectRoot connot be modifield.
        // ParallelRuns doesn't need to be loaded. Relying on Batchruntomo metadata to
        // load split batch fields.
        Ok(())
    }

    /// Java `updateComScriptCommand`.
    fn update_com_script_command(
        &self,
        script_command: &mut ComScriptCommand,
    ) -> Result<(), BadComScriptException> {
        script_command.use_keyword_value();
        self.watch_directory.update_com_script(script_command);
        self.dual_axis.update_com_script(script_command);
        self.two_surfaces.update_com_script(script_command);
        self.match_pattern_or_ext.update_com_script(script_command);
        self.minimum_tilt_range.update_com_script(script_command);
        self.minimum_number_of_views
            .update_com_script(script_command);
        self.minimum_age_of_stacks.update_com_script(script_command);
        self.etomo_project_root.update_com_script(script_command);
        self.parallel_runs.update_com_script(script_command);
        if let Some(debug_mode) = &self.debug_mode {
            debug_mode.update_com_script(script_command);
        }
        script_command.set_value(
            Some("CheckFile"),
            file_type::CLASS
                .series_watcher_check_file
                .get_file_name(Some(self.manager), None)
                .as_deref(),
        );
        // EtomoOptions: Do not rename active .ebt file (1). Use a new (higher) ebt row
        // number (4). (2) Don't strip existing stack ids.
        script_command.set_value(Some("EtomoOptions"), Some("7"));
        Ok(())
    }
}

impl Command for SeriesWatcherParam {
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
        Some(ProcessName::SERIES_WATCHER)
    }

    /// Java `getCommand`.
    fn get_command(&self) -> Option<String> {
        file_type::CLASS
            .series_watcher_comscript
            .get_file_name(Some(self.manager), Some(self.axis_id))
    }

    /// Java `getCommandName`.
    fn get_command_name(&self) -> Option<String> {
        Some(ProcessName::SERIES_WATCHER.to_string())
    }

    /// Java `getCommandLine`.
    fn get_command_line(&self) -> Option<String> {
        Some(ProcessName::SERIES_WATCHER.to_string())
    }

    /// Java `getCommandArray`.  A null command is a one-element array holding
    /// null in the source; it is an empty array here.
    fn get_command_array(&self) -> Option<Vec<String>> {
        Some(self.get_command().into_iter().collect())
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
}
