//! `IMOD/Etomo/src/etomo/comscript/SubtomoSetupParam.java`.

use super::bad_com_script_exception::BadComScriptException;
use super::com_script_command::ComScriptCommand;
use super::command::Command;
use super::command_details::CommandDetails;
use super::command_mode::CommandMode;
use super::command_param::{CommandParam, ParseComScriptError};
use super::ctf3d_setup_param;
use super::fortran_input_string::FortranInputString;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::Type;
use crate::imod::etomo::r#type::etomo_boolean2::EtomoBoolean2;
use crate::imod::etomo::r#type::file_key::FileKey;
use crate::imod::etomo::r#type::file_type::FileType;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::script_parameter::ScriptParameter;
use crate::imod::etomo::r#type::string_parameter::StringParameter;

/// Java private static `PROCESS_NAME`.
const PROCESS_NAME: ProcessName = ProcessName::SUBTOMO_SETUP;
/// Java `VOLUME_MODELED`.
pub const VOLUME_MODELED: &str = "VolumeModeled";
/// Java `REORIENTATION_TYPE`.
pub const REORIENTATION_TYPE: &str = "ReorientionType";
/// Java `CENTER_POSITION_FILE`.
pub const CENTER_POSITION_FILE: &str = "CenterPositionFile";
/// Java `OBJECTS_TO_USE`.
pub const OBJECTS_TO_USE: &str = "ObjectsToUse";
/// Java `SIZE_IN_XYZ`.
pub const SIZE_IN_XYZ: &str = "SizeInXYZ";
/// Java `DIRECTORY_FOR_OUTPUT`.
pub const DIRECTORY_FOR_OUTPUT: &str = "DirectoryForOutput";
/// Java `MAKE_VOLUME_STACKS`.
pub const MAKE_VOLUME_STACKS: &str = "MakeVolumeStacks";
/// Java `SKIP_SUBVOL_NUMBERS`.
pub const SKIP_SUBVOL_NUMBERS: &str = "SkipSubVolNumbers";
/// Java `NEW_ALIGNED_BINNING`.
pub const NEW_ALIGNED_BINNING: &str = "NewAlignedBinning";
/// Java `USE_UNALIGNED_IMAGES`.
pub const USE_UNALIGNED_IMAGES: &str = "UseUnalignedImages";
/// Java `FOURIER_REDUCEBY_FACTOR`.
pub const FOURIER_REDUCEBY_FACTOR: &str = "FourierReduceByFactor";
/// Java `EXTENT_OF_ZLEVELS_IN_NM`.
pub const EXTENT_OF_ZLEVELS_IN_NM: &str = "ExtentOfZLevelsInNm";
/// Java `ERASE_FIDUCIALS`.
pub const ERASE_FIDUCIALS: &str = "EraseFiducials";
/// Java `FILTER_IN_2D`.
pub const FILTER_IN_2D: &str = "FilterIn2D";
/// Java `WHEN_TO_USE_GPU`.
pub const WHEN_TO_USE_GPU: &str = "WhenToUseGPU";
/// Java `PROCESSOR_NUMBER`.
pub const PROCESSOR_NUMBER: &str = "ProcessorNumber";
/// Java `ROOT_NAME`.
pub const ROOT_NAME: &str = "RootName";
/// Java `DIRECTORY_FOR_CHUNK_FILES`.
pub const DIRECTORY_FOR_CHUNK_FILES: &str = "DirectoryForChunkFiles";
/// Java `DIRECTORY_FOR_CHUNK_FILES_NAME`.
pub const DIRECTORY_FOR_CHUNK_FILES_NAME: &str = "subtomo_coms";
/// Java `REORIENTATION_TYPE_NONE`.
pub const REORIENTATION_TYPE_NONE: i32 = 0;
/// Java `REORIENTATION_TYPE_FLIPPED`.
pub const REORIENTATION_TYPE_FLIPPED: i32 = 1;
/// Java `REORIENTATION_TYPE_ROTATED`.
pub const REORIENTATION_TYPE_ROTATED: i32 = -1;
/// Java `SIZE_IN_XYZ_NPARAMS`.
pub const SIZE_IN_XYZ_NPARAMS: i32 = 3;
/// Java `MAKE_VOLUME_STACKS_SPINNER_DEFAULT`.
pub const MAKE_VOLUME_STACKS_SPINNER_DEFAULT: i32 = 1000;
/// Java `MAKE_VOLUME_STACKS_SPINNER_STEP`.
pub const MAKE_VOLUME_STACKS_SPINNER_STEP: i32 = 100;
/// Java `MAKE_VOLUME_STACKS_SPINNER_MIN`.
pub const MAKE_VOLUME_STACKS_SPINNER_MIN: i32 = 0;
/// Java `MAKE_VOLUME_STACKS_SPINNER_MAX`.
pub const MAKE_VOLUME_STACKS_SPINNER_MAX: i32 = 1000000;
/// Java `NEW_ALIGNED_BINNING_SPINNER_DEFAULT`.
pub const NEW_ALIGNED_BINNING_SPINNER_DEFAULT: i32 = 1;
/// Java `NEW_ALIGNED_BINNING_SPINNER_MIN`.
pub const NEW_ALIGNED_BINNING_SPINNER_MIN: i32 = 1;
/// Java `NEW_ALIGNED_BINNING_SPINNER_MAX`.
pub const NEW_ALIGNED_BINNING_SPINNER_MAX: i32 = 8;
/// Java `FOURIER_REDUCEBY_FACTOR_SPINNER_DEFAULT`.
pub const FOURIER_REDUCEBY_FACTOR_SPINNER_DEFAULT: i32 = 1;
/// Java `FOURIER_REDUCEBY_FACTOR_SPINNER_MIN`.
pub const FOURIER_REDUCEBY_FACTOR_SPINNER_MIN: i32 = 1;
/// Java `FOURIER_REDUCEBY_FACTOR_SPINNER_MAX`.
pub const FOURIER_REDUCEBY_FACTOR_SPINNER_MAX: i32 = 8;
/// Java `EXTENT_OF_ZLEVELS_IN_NM_DEFAULT`.
pub const EXTENT_OF_ZLEVELS_IN_NM_DEFAULT: i32 = ctf3d_setup_param::SLAB_THICKNESS_IN_NM_DEFAULT;
/// Java `ERASE_FIDUCIALS_VAL_1`.
pub const ERASE_FIDUCIALS_VAL_1: i32 = 1;
/// Java `FILTER_IN_2D_VAL_1`.
pub const FILTER_IN_2D_VAL_1: i32 = 1;
/// Java `WHEN_TO_USE_GPU_VAL_0`.
pub const WHEN_TO_USE_GPU_VAL_0: i32 = 0;
/// Java `WHEN_TO_USE_GPU_VAL_1`.
pub const WHEN_TO_USE_GPU_VAL_1: i32 = 1;
/// Java `WHEN_TO_USE_GPU_VAL_2`.
pub const WHEN_TO_USE_GPU_VAL_2: i32 = 2;

/// Java `SubtomoSetupParam`.
pub struct SubtomoSetupParam {
    volume_modeled: StringParameter,
    reorientation_type: ScriptParameter,
    center_position_file: StringParameter,
    objects_to_use: StringParameter,
    size_in_xyz: FortranInputString,
    directory_for_output: StringParameter,
    make_volume_stacks: ScriptParameter,
    skip_sub_vol_numbers: EtomoBoolean2,
    new_aligned_binning: ScriptParameter,
    use_unaligned_images: EtomoBoolean2,
    fourier_reduce_by_factor: ScriptParameter,
    extent_of_z_levels_in_nm: ScriptParameter,
    erase_fiducials: EtomoBoolean2,
    filter_in_2d: EtomoBoolean2,
    when_to_use_gpu: ScriptParameter,
    /// Java `processorNumber`: set, but neither parsed nor written by the source.
    processor_number: ScriptParameter,
    root_name: StringParameter,
    directory_for_chunk_files: StringParameter,
    adjust_for_align_z_shift: EtomoBoolean2,
    /// Java `manager`, which the source never reads.
    #[allow(dead_code)]
    manager: &'static dyn BaseManager,
    axis_id: AxisID,
}

impl SubtomoSetupParam {
    /// Java `SubtomoSetupParam(BaseManager, AxisID)`.
    pub fn new(manager: &'static dyn BaseManager, axis_id: AxisID) -> SubtomoSetupParam {
        let mut param = SubtomoSetupParam {
            volume_modeled: StringParameter::new(VOLUME_MODELED),
            reorientation_type: ScriptParameter::new_with_type_and_name(
                Type::Integer,
                REORIENTATION_TYPE,
            ),
            center_position_file: StringParameter::new(CENTER_POSITION_FILE),
            objects_to_use: StringParameter::new(OBJECTS_TO_USE),
            size_in_xyz: FortranInputString::new_with_key(Some(SIZE_IN_XYZ), SIZE_IN_XYZ_NPARAMS),
            directory_for_output: StringParameter::new(DIRECTORY_FOR_OUTPUT),
            make_volume_stacks: ScriptParameter::new_with_type_and_name(
                Type::Integer,
                MAKE_VOLUME_STACKS,
            ),
            skip_sub_vol_numbers: EtomoBoolean2::new_with_name(SKIP_SUBVOL_NUMBERS),
            new_aligned_binning: ScriptParameter::new_with_type_and_name(
                Type::Integer,
                NEW_ALIGNED_BINNING,
            ),
            use_unaligned_images: EtomoBoolean2::new_with_name(USE_UNALIGNED_IMAGES),
            fourier_reduce_by_factor: ScriptParameter::new_with_type_and_name(
                Type::Integer,
                FOURIER_REDUCEBY_FACTOR,
            ),
            extent_of_z_levels_in_nm: ScriptParameter::new_with_type_and_name(
                Type::Integer,
                EXTENT_OF_ZLEVELS_IN_NM,
            ),
            erase_fiducials: EtomoBoolean2::new_with_name(ERASE_FIDUCIALS),
            filter_in_2d: EtomoBoolean2::new_with_name(FILTER_IN_2D),
            when_to_use_gpu: ScriptParameter::new_with_type_and_name(
                Type::Integer,
                WHEN_TO_USE_GPU,
            ),
            processor_number: ScriptParameter::new_with_type_and_name(
                Type::Integer,
                PROCESSOR_NUMBER,
            ),
            root_name: StringParameter::new(ROOT_NAME),
            directory_for_chunk_files: StringParameter::new(DIRECTORY_FOR_CHUNK_FILES),
            adjust_for_align_z_shift: EtomoBoolean2::new_with_name("AdjustForAlignZShift"),
            manager,
            axis_id,
        };
        param.size_in_xyz.set_integer_type(true);
        param
            .directory_for_chunk_files
            .set(Some(DIRECTORY_FOR_CHUNK_FILES_NAME));
        param
    }

    /// Java `isVolumeModeled`.
    pub fn is_volume_modeled(&self) -> bool {
        !self.volume_modeled.is_empty()
    }

    /// Java `isCenterPositionFile`.
    pub fn is_center_position_file(&self) -> bool {
        !self.center_position_file.is_empty()
    }

    /// Java `isDirectoryForOutput`.
    pub fn is_directory_for_output(&self) -> bool {
        !self.directory_for_output.is_empty()
    }

    /// Java `isMakeVolumeStacks`.
    pub fn is_make_volume_stacks(&self) -> bool {
        self.make_volume_stacks.is()
    }

    /// Java `isReorientationType`.
    pub fn is_reorientation_type(&self) -> bool {
        self.reorientation_type.is()
    }

    /// Java `isObjectsToUse`.
    pub fn is_objects_to_use(&self) -> bool {
        !self.objects_to_use.is_empty()
    }

    /// Java `isSizeInXYZ`.
    pub fn is_size_in_xyz(&self) -> bool {
        !self.size_in_xyz.is_empty()
    }

    /// Java `isNewAlignedBinning`.
    pub fn is_new_aligned_binning(&self) -> bool {
        self.new_aligned_binning.is()
    }

    /// Java `isFourierReduceByFactor`.
    pub fn is_fourier_reduce_by_factor(&self) -> bool {
        self.fourier_reduce_by_factor.is()
    }

    /// Java `isExtentOfZLevelsInNm`.
    pub fn is_extent_of_z_levels_in_nm(&self) -> bool {
        self.extent_of_z_levels_in_nm.is()
    }

    /// Java `isAdjustForAlignZShift`.
    pub fn is_adjust_for_align_z_shift(&self) -> bool {
        self.adjust_for_align_z_shift.is()
    }

    /// Java `isWhenToUseGpu`.
    pub fn is_when_to_use_gpu(&self) -> bool {
        self.when_to_use_gpu.is()
    }

    /// Java `getVolumeModeled`.
    pub fn get_volume_modeled(&self) -> String {
        self.volume_modeled.to_string()
    }

    /// Java `getReorientationType`.
    pub fn get_reorientation_type(&self) -> i32 {
        self.reorientation_type.get_int()
    }

    /// Java `getCenterPositionFile`.
    pub fn get_center_position_file(&self) -> String {
        self.center_position_file.to_string()
    }

    /// Java `getOjectsToUse`.
    pub fn get_ojects_to_use(&self) -> String {
        self.objects_to_use.to_string()
    }

    /// Java `getSizeInX`.
    pub fn get_size_in_x(&self) -> i32 {
        self.size_in_xyz.get_int(0)
    }

    /// Java `getSizeInY`.
    pub fn get_size_in_y(&self) -> i32 {
        self.size_in_xyz.get_int(1)
    }

    /// Java `getSizeInZ`.
    pub fn get_size_in_z(&self) -> i32 {
        self.size_in_xyz.get_int(2)
    }

    /// Java `getDirectoryForOutput`.
    pub fn get_directory_for_output(&self) -> String {
        self.directory_for_output.to_string()
    }

    /// Java `getMakeVolumeStacks`.
    pub fn get_make_volume_stacks(&self) -> String {
        self.make_volume_stacks.to_string()
    }

    /// Java `isSkipSubVolNumbers`.
    pub fn is_skip_sub_vol_numbers(&self) -> bool {
        self.skip_sub_vol_numbers.is()
    }

    /// Java `getNewAlignedBinning`.
    pub fn get_new_aligned_binning(&self) -> i32 {
        self.new_aligned_binning.get_int()
    }

    /// Java `isUseUnalignedImages`.
    pub fn is_use_unaligned_images(&self) -> bool {
        self.use_unaligned_images.is()
    }

    /// Java `getFourierReduceByFactor`.
    pub fn get_fourier_reduce_by_factor(&self) -> i32 {
        self.fourier_reduce_by_factor.get_int()
    }

    /// Java `getExtentOfZLevelsInNm`.
    pub fn get_extent_of_z_levels_in_nm(&self) -> String {
        self.extent_of_z_levels_in_nm.to_string()
    }

    /// Java `isEraseFiducials`.
    pub fn is_erase_fiducials(&self) -> bool {
        self.erase_fiducials.is()
    }

    /// Java `isFilterIn2D`.
    pub fn is_filter_in_2d(&self) -> bool {
        self.filter_in_2d.is()
    }

    /// Java `getWhenToUseGpu`.
    pub fn get_when_to_use_gpu(&self) -> i32 {
        self.when_to_use_gpu.get_int()
    }

    /// Java `setVolumeModeled`.
    pub fn set_volume_modeled(&mut self, input: Option<&str>) {
        self.volume_modeled.set(input);
    }

    /// Java `setReorientationType`.
    pub fn set_reorientation_type(&mut self, input: Option<&str>) {
        self.reorientation_type.set_string(input);
    }

    /// Java `resetReorientationType`.
    pub fn reset_reorientation_type(&mut self) {
        self.reorientation_type.reset();
    }

    /// Java `setCenterPositionFile`.
    pub fn set_center_position_file(&mut self, input: Option<&str>) {
        self.center_position_file.set(input);
    }

    /// Java `setOjectsToUse`.
    pub fn set_ojects_to_use(&mut self, input: Option<&str>) {
        self.objects_to_use.set(input);
    }

    /// Java `setSizeInX`.
    pub fn set_size_in_x(&mut self, input: Option<&str>) {
        self.size_in_xyz.set_index_string(0, input);
    }

    /// Java `setSizeInY`.
    pub fn set_size_in_y(&mut self, input: Option<&str>) {
        self.size_in_xyz.set_index_string(1, input);
    }

    /// Java `setSizeInZ`.
    pub fn set_size_in_z(&mut self, input: Option<&str>) {
        self.size_in_xyz.set_index_string(2, input);
    }

    /// Java `setDirectoryForOutput`.
    pub fn set_directory_for_output(&mut self, input: Option<&str>) {
        self.directory_for_output.set(input);
    }

    /// Java `setMakeVolumeStacks`.
    pub fn set_make_volume_stacks(&mut self, input: Option<&str>) {
        self.make_volume_stacks.set_string(input);
    }

    /// Java `resetMakeVolumeStacks`.
    pub fn reset_make_volume_stacks(&mut self) {
        self.make_volume_stacks.reset();
    }

    /// Java `setSkipSubVolNumbers`.
    pub fn set_skip_sub_vol_numbers(&mut self, input: bool) {
        self.skip_sub_vol_numbers.set_boolean(input);
    }

    /// Java `setNewAlignedBinning`.
    pub fn set_new_aligned_binning(&mut self, input: Option<&str>) {
        self.new_aligned_binning.set_string(input);
    }

    /// Java `resetNewAlignedBinning`.
    pub fn reset_new_aligned_binning(&mut self) {
        self.new_aligned_binning.reset();
    }

    /// Java `setUseUnalignedImages`.
    pub fn set_use_unaligned_images(&mut self, input: bool) {
        self.use_unaligned_images.set_boolean(input);
    }

    /// Java `setFourierReduceByFactor`.
    pub fn set_fourier_reduce_by_factor(&mut self, input: Option<&str>) {
        self.fourier_reduce_by_factor.set_string(input);
    }

    /// Java `resetFourierReduceByFactor`.
    pub fn reset_fourier_reduce_by_factor(&mut self) {
        self.fourier_reduce_by_factor.reset();
    }

    /// Java `setExtentOfZLevelsInNm`.
    pub fn set_extent_of_z_levels_in_nm(&mut self, input: Option<&str>) {
        self.extent_of_z_levels_in_nm.set_string(input);
    }

    /// Java `resetExtentOfZLevelsInNm`.
    pub fn reset_extent_of_z_levels_in_nm(&mut self) {
        self.extent_of_z_levels_in_nm.reset();
    }

    /// Java `setAdjustForAlignZShift`.
    pub fn set_adjust_for_align_z_shift(&mut self, input: bool) {
        self.adjust_for_align_z_shift.set_boolean(input);
    }

    /// Java `setEraseFiducials`.
    pub fn set_erase_fiducials(&mut self, input: bool) {
        self.erase_fiducials.set_boolean(input);
    }

    /// Java `setFilterIn2D`.
    pub fn set_filter_in_2d(&mut self, input: bool) {
        self.filter_in_2d.set_boolean(input);
    }

    /// Java `setWhenToUseGpu`.
    pub fn set_when_to_use_gpu(&mut self, input: Option<&str>) {
        self.when_to_use_gpu.set_string(input);
    }

    /// Java `setProcessorNumber`.
    pub fn set_processor_number(&mut self, input: Option<&str>) {
        self.processor_number.set_string(input);
    }

    /// Java `setRootname`.
    pub fn set_rootname(&mut self, input: Option<&str>) {
        self.root_name.set(input);
    }
}

impl CommandParam for SubtomoSetupParam {
    /// Java `parseComScriptCommand`.
    fn parse_com_script_command(
        &mut self,
        script_command: &ComScriptCommand,
    ) -> Result<(), ParseComScriptError> {
        // Java calls `scriptCommand.useKeywordValue()` on the command it parses,
        // converting an old-style command in place.  The trait lends the command
        // immutably, so the conversion is made on a copy, which is what is parsed.
        let mut script_command = ComScriptCommand::new_from(script_command);
        script_command.use_keyword_value();
        let script_command = &script_command;
        self.initialize_defaults();
        self.volume_modeled.parse(script_command)?;
        self.reorientation_type.parse(script_command)?;
        self.center_position_file.parse(script_command)?;
        self.objects_to_use.parse(script_command)?;
        self.size_in_xyz
            .validate_and_set_com_script(script_command)?;
        self.directory_for_output.parse(script_command)?;
        self.make_volume_stacks.parse(script_command)?;
        self.skip_sub_vol_numbers.parse(script_command)?;
        self.new_aligned_binning.parse(script_command)?;
        self.use_unaligned_images.parse(script_command)?;
        self.fourier_reduce_by_factor.parse(script_command)?;
        self.extent_of_z_levels_in_nm.parse(script_command)?;
        self.adjust_for_align_z_shift.parse(script_command)?;
        self.erase_fiducials.parse(script_command)?;
        self.filter_in_2d.parse(script_command)?;
        self.when_to_use_gpu.parse(script_command)?;
        Ok(())
    }

    /// Java `updateComScriptCommand`.
    fn update_com_script_command(
        &self,
        script_command: &mut ComScriptCommand,
    ) -> Result<(), BadComScriptException> {
        script_command.use_keyword_value();
        self.volume_modeled.update_com_script(script_command);
        self.reorientation_type.update_com_script(script_command);
        self.center_position_file.update_com_script(script_command);
        self.objects_to_use.update_com_script(script_command);
        self.size_in_xyz.update_script_parameter(script_command);
        self.directory_for_output.update_com_script(script_command);
        self.make_volume_stacks.update_com_script(script_command);
        self.skip_sub_vol_numbers.update_com_script(script_command);
        self.new_aligned_binning.update_com_script(script_command);
        self.use_unaligned_images.update_com_script(script_command);
        self.fourier_reduce_by_factor
            .update_com_script(script_command);
        self.extent_of_z_levels_in_nm
            .update_com_script(script_command);
        self.adjust_for_align_z_shift
            .update_com_script(script_command);
        self.erase_fiducials.update_com_script(script_command);
        self.filter_in_2d.update_com_script(script_command);
        self.when_to_use_gpu.update_com_script(script_command);
        self.root_name.update_com_script(script_command);
        self.directory_for_chunk_files
            .update_com_script(script_command);
        Ok(())
    }

    /// Java `initializeDefaults`.
    fn initialize_defaults(&mut self) {
        self.volume_modeled.reset();
        self.reorientation_type.reset();
        self.center_position_file.reset();
        self.objects_to_use.reset();
        self.size_in_xyz.reset();
        self.directory_for_output.reset();
        self.make_volume_stacks.reset();
        self.skip_sub_vol_numbers.reset();
        self.new_aligned_binning.reset();
        self.use_unaligned_images.reset();
        self.fourier_reduce_by_factor.reset();
        self.extent_of_z_levels_in_nm.reset();
        self.adjust_for_align_z_shift.reset();
        self.erase_fiducials.reset();
        self.filter_in_2d.reset();
        self.when_to_use_gpu.reset();
    }
}

impl Command for SubtomoSetupParam {
    fn get_axis_id(&self) -> AxisID {
        self.axis_id
    }

    fn get_command_mode(&self) -> Option<&dyn CommandMode> {
        None
    }

    fn get_process_name(&self) -> Option<ProcessName> {
        Some(ProcessName::SUBTOMO_SETUP)
    }

    fn get_command(&self) -> Option<String> {
        Some(PROCESS_NAME.get_comscript(self.axis_id))
    }

    fn get_command_name(&self) -> Option<String> {
        Some(ProcessName::SUBTOMO_SETUP.to_string())
    }

    fn get_command_line(&self) -> Option<String> {
        self.get_command()
    }

    fn get_command_array(&self) -> Option<Vec<String>> {
        Some(PROCESS_NAME.get_comscript_array(self.axis_id))
    }

    fn get_command_input_file(&self) -> Option<std::path::PathBuf> {
        None
    }

    fn get_command_output_file(&self) -> Option<std::path::PathBuf> {
        None
    }

    fn get_output_image_file_type(&self) -> Option<std::sync::Arc<FileType>> {
        None
    }

    fn get_output_image_file_key(&self) -> Option<FileKey> {
        None
    }

    fn get_output_image_file_type2(&self) -> Option<std::sync::Arc<FileType>> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extent_default_is_the_ctf3d_slab_thickness_default() {
        assert_eq!(EXTENT_OF_ZLEVELS_IN_NM_DEFAULT, 15);
        assert_eq!(REORIENTATION_TYPE, "ReorientionType");
    }
}
