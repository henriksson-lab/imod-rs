//! `IMOD/Etomo/src/etomo/comscript/AutofidseedParam.java`.
//!
//! Parameters for autofidseed, run from `autofidseed.com`.

use std::path::PathBuf;
use std::sync::Arc;

use super::bad_com_script_exception::BadComScriptException;
use super::com_script_command::ComScriptCommand;
use super::command::Command;
use super::command_details::CommandDetails;
use super::command_mode::CommandMode;
use super::command_param::{CommandParam, ParseComScriptError};
use super::fortran_input_string::FortranInputString;
use super::fortran_input_syntax_exception::FortranInputSyntaxException;
use super::string_list::StringList;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::logic::clustered_points_allowed::ClusteredPointsAllowed;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::{ConstEtomoNumber, Number, Type};
use crate::imod::etomo::r#type::etomo_boolean2::EtomoBoolean2;
use crate::imod::etomo::r#type::file_key::FileKey;
use crate::imod::etomo::r#type::file_type::{self, FileType};
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::script_parameter::ScriptParameter;
use crate::imod::etomo::r#type::string_parameter::StringParameter;

/// Java private static `PROCESS_NAME`.
const PROCESS_NAME: ProcessName = ProcessName::AUTOFIDSEED;
/// Java `BOUNDARY_MODEL_KEY`.
pub const BOUNDARY_MODEL_KEY: &str = "BoundaryModel";
/// Java `EXCLUDE_INSIDE_AREAS_KEY`.
pub const EXCLUDE_INSIDE_AREAS_KEY: &str = "ExcludeInsideAreas";
/// Java `BORDERS_IN_X_AND_Y_KEY`.
pub const BORDERS_IN_X_AND_Y_KEY: &str = "BordersInXandY";
/// Java `MIN_GUESS_NUM_BEADS_KEY`.
pub const MIN_GUESS_NUM_BEADS_KEY: &str = "MinGuessNumBeads";
/// Java `MIN_SPACING_KEY`.
pub const MIN_SPACING_KEY: &str = "MinSpacing";
/// Java `PEAK_STORAGE_FRACTION_KEY`.
pub const PEAK_STORAGE_FRACTION_KEY: &str = "PeakStorageFraction";
/// Java `TARGET_NUMBER_OF_BEADS_KEY`.
pub const TARGET_NUMBER_OF_BEADS_KEY: &str = "TargetNumberOfBeads";
/// Java `TARGET_DENSITY_OF_BEADS_KEY`.
pub const TARGET_DENSITY_OF_BEADS_KEY: &str = "TargetDensityOfBeads";
/// Java `TWO_SURFACES_KEY`.
pub const TWO_SURFACES_KEY: &str = "TwoSurfaces";
/// Java `APPEND_TO_SEED_MODEL_KEY`.
pub const APPEND_TO_SEED_MODEL_KEY: &str = "AppendToSeedModel";
/// Java `IGNORE_SURFACE_DATA_KEY`.
pub const IGNORE_SURFACE_DATA_KEY: &str = "IgnoreSurfaceData";
/// Java `DROP_TRACKS_KEY`.
pub const DROP_TRACKS_KEY: &str = "DropTracks";
/// Java `MAX_MAJOR_TO_MINOR_RATIO_KEY`.
pub const MAX_MAJOR_TO_MINOR_RATIO_KEY: &str = "MaxMajorToMinorRatio";
/// Java `CLUSTERED_POINTS_ALLOWED_KEY`.
pub const CLUSTERED_POINTS_ALLOWED_KEY: &str = "ClusteredPointsAllowed";
/// Java `ADJUST_SIZES_KEY`.
pub const ADJUST_SIZES_KEY: &str = "AdjustSizes";
/// Java `ELONGATED_POINTS_ALLOWED_KEY`.
pub const ELONGATED_POINTS_ALLOWED_KEY: &str = "ElongatedPointsAllowed";
/// Java `LOWER_TARGET_FOR_CLUSTERED_KEY`.
pub const LOWER_TARGET_FOR_CLUSTERED_KEY: &str = "LowerTargetForClustered";
/// Java `JUST_FIND_SHIFTS_NEAR_ZERO_KEY`.
pub const JUST_FIND_SHIFTS_NEAR_ZERO_KEY: &str = "JustFindShiftsNearZero";

/// Java `AutofidseedParam`.
pub struct AutofidseedParam {
    track_command_file: StringParameter,
    min_guess_num_beads: ScriptParameter,
    min_spacing: ScriptParameter,
    peak_storage_fraction: ScriptParameter,
    boundary_model: StringParameter,
    exclude_inside_areas: EtomoBoolean2,
    borders_in_xand_y: FortranInputString,
    two_surfaces: EtomoBoolean2,
    append_to_seed_model: EtomoBoolean2,
    target_number_of_beads: ScriptParameter,
    target_density_of_beads: ScriptParameter,
    max_major_to_minor_ratio: ScriptParameter,
    clustered_points_allowed: ScriptParameter,
    ignore_surface_data: StringList,
    drop_tracks: StringList,
    elongated_points_allowed: ScriptParameter,
    adjust_sizes: EtomoBoolean2,
    lower_target_for_clustered: ScriptParameter,
    just_find_shifts_near_zero: ScriptParameter,

    manager: &'static dyn BaseManager,
    axis_id: AxisID,
}

impl AutofidseedParam {
    /// Java `AutofidseedParam(BaseManager, AxisID)`.
    pub fn new(manager: &'static dyn BaseManager, axis_id: AxisID) -> AutofidseedParam {
        let mut param = AutofidseedParam {
            track_command_file: StringParameter::new("TrackCommandFile"),
            min_guess_num_beads: ScriptParameter::new_with_name(MIN_GUESS_NUM_BEADS_KEY),
            min_spacing: ScriptParameter::new_with_type_and_name(Type::Double, MIN_SPACING_KEY),
            peak_storage_fraction: ScriptParameter::new_with_type_and_name(
                Type::Double,
                PEAK_STORAGE_FRACTION_KEY,
            ),
            boundary_model: StringParameter::new(BOUNDARY_MODEL_KEY),
            exclude_inside_areas: EtomoBoolean2::new_with_name(EXCLUDE_INSIDE_AREAS_KEY),
            borders_in_xand_y: FortranInputString::new_with_key(Some(BORDERS_IN_X_AND_Y_KEY), 2),
            two_surfaces: EtomoBoolean2::new_with_name(TWO_SURFACES_KEY),
            append_to_seed_model: EtomoBoolean2::new_with_name(APPEND_TO_SEED_MODEL_KEY),
            target_number_of_beads: ScriptParameter::new_with_name(TARGET_NUMBER_OF_BEADS_KEY),
            target_density_of_beads: ScriptParameter::new_with_name(TARGET_DENSITY_OF_BEADS_KEY),
            max_major_to_minor_ratio: ScriptParameter::new_with_type_and_name(
                Type::Double,
                MAX_MAJOR_TO_MINOR_RATIO_KEY,
            ),
            clustered_points_allowed: ScriptParameter::new_with_name(CLUSTERED_POINTS_ALLOWED_KEY),
            ignore_surface_data: StringList::new_with_key(Some(IGNORE_SURFACE_DATA_KEY)),
            drop_tracks: StringList::new_with_key(Some(DROP_TRACKS_KEY)),
            elongated_points_allowed: ScriptParameter::new_with_name(ELONGATED_POINTS_ALLOWED_KEY),
            adjust_sizes: EtomoBoolean2::new_with_name(ADJUST_SIZES_KEY),
            lower_target_for_clustered: ScriptParameter::new_with_type_and_name(
                Type::Double,
                LOWER_TARGET_FOR_CLUSTERED_KEY,
            ),
            just_find_shifts_near_zero: ScriptParameter::new_with_type_and_name(
                Type::Integer,
                JUST_FIND_SHIFTS_NEAR_ZERO_KEY,
            ),
            manager,
            axis_id,
        };
        param.borders_in_xand_y.set_integer_type(true);
        param.initialize_defaults();
        param
    }

    /// Java `isAdjustSizes`.
    pub fn is_adjust_sizes(&self) -> bool {
        self.adjust_sizes.is()
    }

    /// Java `isElongatedPointsAllowedSet`.
    pub fn is_elongated_points_allowed_set(&self) -> bool {
        !self.elongated_points_allowed.is_null() && !self.elongated_points_allowed.equals_int(0)
    }

    /// Java `setAdjustSizes`.
    pub fn set_adjust_sizes(&mut self, input: bool) {
        self.adjust_sizes.set_boolean(input);
    }

    /// Java `setElongatedPointsAllowed(Number)`.
    pub fn set_elongated_points_allowed(&mut self, input: Option<Number>) {
        self.elongated_points_allowed.set_number(input);
    }

    /// Java `resetElongatedPointsAllowed`.
    pub fn reset_elongated_points_allowed(&mut self) {
        self.elongated_points_allowed.reset();
    }

    /// Java `getElongatedPointsAllowed`.
    pub fn get_elongated_points_allowed(&self) -> &ConstEtomoNumber {
        &self.elongated_points_allowed
    }

    /// Java `getMinGuessNumBeads`.
    pub fn get_min_guess_num_beads(&self) -> String {
        self.min_guess_num_beads.to_string()
    }

    /// Java `getMinSpacing`.
    pub fn get_min_spacing(&self) -> String {
        self.min_spacing.to_string()
    }

    /// Java `getPeakStorageFraction`.
    pub fn get_peak_storage_fraction(&self) -> String {
        self.peak_storage_fraction.to_string()
    }

    /// Java `isBoundaryModel`.
    pub fn is_boundary_model(&self) -> bool {
        !self.boundary_model.is_empty()
    }

    /// Java `isExcludeInsideAreas`.
    pub fn is_exclude_inside_areas(&self) -> bool {
        self.exclude_inside_areas.is()
    }

    /// Java `getBordersInXandY`.
    pub fn get_borders_in_xand_y(&self) -> String {
        self.borders_in_xand_y.to_string_default_is_blank(true)
    }

    /// Java `isTwoSurfaces`.
    pub fn is_two_surfaces(&self) -> bool {
        self.two_surfaces.is()
    }

    /// Java `isAppendToSeedModel`.
    pub fn is_append_to_seed_model(&self) -> bool {
        self.append_to_seed_model.is()
    }

    /// Java `isTargetNumberOfBeads`.
    pub fn is_target_number_of_beads(&self) -> bool {
        !self.target_number_of_beads.is_null()
    }

    /// Java `getTargetNumberOfBeads`.
    pub fn get_target_number_of_beads(&self) -> String {
        self.target_number_of_beads.to_string()
    }

    /// Java `isTargetDensityOfBeads`.
    pub fn is_target_density_of_beads(&self) -> bool {
        !self.target_density_of_beads.is_null()
    }

    /// Java `getTargetDensityOfBeads`.
    pub fn get_target_density_of_beads(&self) -> String {
        self.target_density_of_beads.to_string()
    }

    /// Java `getMaxMajorToMinorRatio`.
    pub fn get_max_major_to_minor_ratio(&self) -> String {
        self.max_major_to_minor_ratio.to_string()
    }

    /// Java `isClusteredPointsAllowed`.
    pub fn is_clustered_points_allowed(&self) -> bool {
        !self.clustered_points_allowed.is_null()
    }

    /// Java `getClusteredPointsAllowed`.  Null when the value is not one of the
    /// defined instances.
    pub fn get_clustered_points_allowed(&self) -> Option<ClusteredPointsAllowed> {
        ClusteredPointsAllowed::get_instance(self.clustered_points_allowed.get_int())
    }

    /// Java `getIgnoreSurfaceData`.
    pub fn get_ignore_surface_data(&self) -> String {
        self.ignore_surface_data.to_string()
    }

    /// Java `getDropTracks`.
    pub fn get_drop_tracks(&self) -> String {
        self.drop_tracks.to_string()
    }

    /// Java `isLowerTargetForClustered`.
    pub fn is_lower_target_for_clustered(&self) -> bool {
        self.lower_target_for_clustered.is()
    }

    /// Java `getLowerTargetForClustered`.
    pub fn get_lower_target_for_clustered(&self) -> String {
        self.lower_target_for_clustered.to_string()
    }

    /// Java `setMinGuessNumBeads`.
    pub fn set_min_guess_num_beads(&mut self, input: Option<&str>) {
        self.min_guess_num_beads.set_string(input);
    }

    /// Java `setMinSpacing`.
    pub fn set_min_spacing(&mut self, input: Option<&str>) {
        self.min_spacing.set_string(input);
    }

    /// Java `setPeakStorageFraction`.
    pub fn set_peak_storage_fraction(&mut self, input: Option<&str>) {
        self.peak_storage_fraction.set_string(input);
    }

    /// Java `setBoundaryModel`.
    pub fn set_boundary_model(&mut self, set: bool) {
        if set {
            self.boundary_model.set(
                file_type::CLASS
                    .autofidseed_boundary_model
                    .get_file_name(Some(self.manager), Some(self.axis_id))
                    .as_deref(),
            );
        } else {
            self.boundary_model.reset();
        }
    }

    /// Java `setExcludeInsideAreas`.
    pub fn set_exclude_inside_areas(&mut self, input: bool) {
        self.exclude_inside_areas.set_boolean(input);
    }

    /// Java `setBordersInXandY`.
    pub fn set_borders_in_xand_y(
        &mut self,
        input: Option<&str>,
    ) -> Result<(), FortranInputSyntaxException> {
        self.borders_in_xand_y.validate_and_set(input)
    }

    /// Java `setTwoSurfaces`.
    pub fn set_two_surfaces(&mut self, input: bool) {
        self.two_surfaces.set_boolean(input);
    }

    /// Java `setAppendToSeedModel`.
    pub fn set_append_to_seed_model(&mut self, input: bool) {
        self.append_to_seed_model.set_boolean(input);
    }

    /// Java `setTargetNumberOfBeads`.
    pub fn set_target_number_of_beads(&mut self, input: Option<&str>) {
        self.target_number_of_beads.set_string(input);
    }

    /// Java `resetTargetNumberOfBeads`.
    pub fn reset_target_number_of_beads(&mut self) {
        self.target_number_of_beads.reset();
    }

    /// Java `setTargetDensityOfBeads`.
    pub fn set_target_density_of_beads(&mut self, input: Option<&str>) {
        self.target_density_of_beads.set_string(input);
    }

    /// Java `resetTargetDensityOfBeads`.
    pub fn reset_target_density_of_beads(&mut self) {
        self.target_density_of_beads.reset();
    }

    /// Java `setMaxMajorToMinorRatio`.
    pub fn set_max_major_to_minor_ratio(&mut self, input: Option<&str>) {
        self.max_major_to_minor_ratio.set_string(input);
    }

    /// Java `setClusteredPointsAllowed`.
    pub fn set_clustered_points_allowed(&mut self, input: bool) {
        if input {
            self.clustered_points_allowed
                .set_int(ClusteredPointsAllowed::CLUSTERED.get_value());
        } else {
            self.clustered_points_allowed.reset();
        }
    }

    /// Java `resetClusteredPointsAllowed`.
    pub fn reset_clustered_points_allowed(&mut self) {
        self.clustered_points_allowed.reset();
    }

    /// Java `setIgnoreSurfaceData`.
    pub fn set_ignore_surface_data(&mut self, input: Option<&str>) {
        self.ignore_surface_data.parse_string(input);
    }

    /// Java `setDropTracks`.
    pub fn set_drop_tracks(&mut self, input: Option<&str>) {
        self.drop_tracks.parse_string(input);
    }

    /// Java `resetLowerTargetForClustered`.
    pub fn reset_lower_target_for_clustered(&mut self) {
        self.lower_target_for_clustered.reset();
    }

    /// Java `setLowerTargetForClustered`.
    pub fn set_lower_target_for_clustered(&mut self, input: Option<&str>) {
        self.lower_target_for_clustered.set_string(input);
    }

    /// Java `resetJustFindShiftsNearZero`.
    pub fn reset_just_find_shifts_near_zero(&mut self) {
        self.just_find_shifts_near_zero.reset();
    }

    /// Java `setJustFindShiftsNearZero`.
    pub fn set_just_find_shifts_near_zero(&mut self, input: Option<&str>) {
        self.just_find_shifts_near_zero.set_string(input);
    }
}

impl CommandParam for AutofidseedParam {
    /// Java `initializeDefaults`.
    fn initialize_defaults(&mut self) {
        self.track_command_file.set(
            file_type::CLASS
                .track_comscript
                .get_file_name(Some(self.manager), Some(self.axis_id))
                .as_deref(),
        );
        self.min_guess_num_beads.reset();
        self.min_spacing.set_string(Some("0.85"));
        self.peak_storage_fraction.set_string(Some("1"));
        self.boundary_model.reset();
        self.exclude_inside_areas.reset();
        self.borders_in_xand_y.reset();
        self.two_surfaces.reset();
        self.append_to_seed_model.reset();
        self.target_number_of_beads.reset();
        self.target_density_of_beads.reset();
        self.max_major_to_minor_ratio.reset();
        self.clustered_points_allowed.reset();
        self.ignore_surface_data.reset();
        self.drop_tracks.reset();
        self.elongated_points_allowed.reset();
        self.adjust_sizes.reset();
        self.lower_target_for_clustered.reset();
        self.just_find_shifts_near_zero.reset();
    }

    /// Java `parseComScriptCommand`.  Get the parameters from the
    /// ComScriptCommand.
    fn parse_com_script_command(
        &mut self,
        script_command: &ComScriptCommand,
    ) -> Result<(), ParseComScriptError> {
        self.initialize_defaults();
        self.track_command_file.parse(script_command)?;
        self.min_guess_num_beads.parse(script_command)?;
        self.min_spacing.parse(script_command)?;
        self.peak_storage_fraction.parse(script_command)?;
        self.boundary_model.parse(script_command)?;
        self.exclude_inside_areas.parse(script_command)?;
        self.borders_in_xand_y
            .validate_and_set_com_script(script_command)?;
        self.two_surfaces.parse(script_command)?;
        self.append_to_seed_model.parse(script_command)?;
        self.target_number_of_beads.parse(script_command)?;
        self.target_density_of_beads.parse(script_command)?;
        self.max_major_to_minor_ratio.parse(script_command)?;
        self.clustered_points_allowed.parse(script_command)?;
        self.ignore_surface_data.parse(script_command)?;
        self.drop_tracks.parse(script_command)?;
        self.elongated_points_allowed.parse(script_command)?;
        self.adjust_sizes.parse(script_command)?;
        self.lower_target_for_clustered.parse(script_command)?;
        self.just_find_shifts_near_zero.parse(script_command)?;
        Ok(())
    }

    /// Java `updateComScriptCommand`.  Update the script command with the
    /// current values of this object.
    fn update_com_script_command(
        &self,
        script_command: &mut ComScriptCommand,
    ) -> Result<(), BadComScriptException> {
        // Switch to keyword/value pairs
        script_command.use_keyword_value();
        self.track_command_file.update_com_script(script_command);
        self.min_guess_num_beads.update_com_script(script_command);
        self.min_spacing.update_com_script(script_command);
        self.peak_storage_fraction.update_com_script(script_command);
        self.boundary_model.update_com_script(script_command);
        self.exclude_inside_areas.update_com_script(script_command);
        self.borders_in_xand_y
            .update_script_parameter(script_command);
        self.two_surfaces.update_com_script(script_command);
        self.append_to_seed_model.update_com_script(script_command);
        self.target_number_of_beads
            .update_com_script(script_command);
        self.target_density_of_beads
            .update_com_script(script_command);
        self.max_major_to_minor_ratio
            .update_com_script(script_command);
        self.clustered_points_allowed
            .update_com_script(script_command);
        self.ignore_surface_data.update_com_script(script_command)?;
        self.drop_tracks.update_com_script(script_command)?;
        self.elongated_points_allowed
            .update_com_script(script_command);
        self.adjust_sizes.update_com_script(script_command);
        self.lower_target_for_clustered
            .update_com_script(script_command);
        self.just_find_shifts_near_zero
            .update_com_script(script_command);
        Ok(())
    }
}

impl Command for AutofidseedParam {
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
        Some(PROCESS_NAME)
    }

    /// Java `getCommand`.
    fn get_command(&self) -> Option<String> {
        Some(PROCESS_NAME.get_comscript(self.axis_id))
    }

    /// Java `getCommandName`.
    fn get_command_name(&self) -> Option<String> {
        Some(PROCESS_NAME.to_string())
    }

    /// Java `getCommandLine`.
    fn get_command_line(&self) -> Option<String> {
        Some(PROCESS_NAME.get_comscript(self.axis_id))
    }

    /// Java `getCommandArray`.
    fn get_command_array(&self) -> Option<Vec<String>> {
        Some(vec![PROCESS_NAME.get_comscript(self.axis_id)])
    }

    /// Java `getCommandInputFile`.
    fn get_command_input_file(&self) -> Option<PathBuf> {
        file_type::CLASS
            .prealigned_stack
            .get_file(Some(self.manager), Some(self.axis_id))
    }

    /// Java `getCommandOutputFile`.
    fn get_command_output_file(&self) -> Option<PathBuf> {
        file_type::CLASS
            .seed_model
            .get_file(Some(self.manager), Some(self.axis_id))
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
