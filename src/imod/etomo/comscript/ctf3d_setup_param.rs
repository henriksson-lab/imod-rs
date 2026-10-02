//! `IMOD/Etomo/src/etomo/comscript/Ctf3dSetupParam.java`.

use super::bad_com_script_exception::BadComScriptException;
use super::com_script_command::ComScriptCommand;
use super::command::Command;
use super::command_details::CommandDetails;
use super::command_mode::CommandMode;
use super::command_param::{CommandParam, ParseComScriptError};
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::{Number, Type};
use crate::imod::etomo::r#type::etomo_boolean2::EtomoBoolean2;
use crate::imod::etomo::r#type::file_key::FileKey;
use crate::imod::etomo::r#type::file_type::{self, FileType};
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::script_parameter::ScriptParameter;
use crate::imod::etomo::r#type::string_parameter::StringParameter;
use crate::imod::etomo::util::utilities;

/// Java private static `PROCESS_NAME`.
const PROCESS_NAME: ProcessName = ProcessName::CTF_3D_SETUP;
/// Java `SLAB_THICKNESS_IN_NM_KEY`.
pub const SLAB_THICKNESS_IN_NM_KEY: &str = "SlabThicknessInNm";
/// Java `NUMBER_OF_SLABS_MIN`.
pub const NUMBER_OF_SLABS_MIN: i32 = 3;
/// Java `RUN_SLABS_IN_PARALLEL_KEY`.
pub const RUN_SLABS_IN_PARALLEL_KEY: &str = "RunSlabsInParallel";
/// Java `ERASE_FIDUCIALS_KEY`.
pub const ERASE_FIDUCIALS_KEY: &str = "EraseFiducials";
/// Java `FILTER_IN_2D_KEY`.
pub const FILTER_IN_2D_KEY: &str = "FilterIn2D";
/// Java `USE_UNALIGNED_IMAGES_KEY`.
pub const USE_UNALIGNED_IMAGES_KEY: &str = "UseUnalignedImages";
/// Java `FOURIER_REDUCE_BY_FACTOR_KEY`.
pub const FOURIER_REDUCE_BY_FACTOR_KEY: &str = "FourierReduceByFactor";
/// Java `FOURIER_REDUCE_BY_FACTOR_DEFAULT`.
pub const FOURIER_REDUCE_BY_FACTOR_DEFAULT: i32 = 1;
/// Java `FOURIER_REDUCE_BY_FACTOR_MIN`.
pub const FOURIER_REDUCE_BY_FACTOR_MIN: i32 = 1;
/// Java `FOURIER_REDUCE_BY_FACTOR_MAX`.
pub const FOURIER_REDUCE_BY_FACTOR_MAX: i32 = 8;
/// Java `VERTICAL_SLICES_KEY`.
pub const VERTICAL_SLICES_KEY: &str = "VerticalSlices";
/// Java `OLD_STYLE_X_TILTING_KEY`.
pub const OLD_STYLE_X_TILTING_KEY: &str = "OldStyleXtilting";
/// Java `TEMPORARY_DIRECTORY_KEY`.
pub const TEMPORARY_DIRECTORY_KEY: &str = "TemporaryDirectory";
/// Java `SLAB_THICKNESS_IN_NM_DEFAULT`.
pub const SLAB_THICKNESS_IN_NM_DEFAULT: i32 = 15;

/// Java `Ctf3dSetupParam`.
pub struct Ctf3dSetupParam {
    slab_thickness_in_nm: ScriptParameter,
    run_slabs_in_parallel: EtomoBoolean2,
    erase_fiducials: EtomoBoolean2,
    filter_in_2d: EtomoBoolean2,
    use_unaligned_images: EtomoBoolean2,
    fourier_reduce_by_factor: ScriptParameter,
    vertical_slices: EtomoBoolean2,
    old_style_xtilting: EtomoBoolean2,
    temporary_directory: StringParameter,
    tilt_command_file: StringParameter,
    number_of_processors: ScriptParameter,
    adjust_for_align_z_shift: EtomoBoolean2,
    manager: &'static dyn BaseManager,
    axis_id: AxisID,
}

impl Ctf3dSetupParam {
    /// Java `Ctf3dSetupParam(BaseManager, AxisID)`.
    pub fn new(manager: &'static dyn BaseManager, axis_id: AxisID) -> Ctf3dSetupParam {
        Ctf3dSetupParam {
            slab_thickness_in_nm: ScriptParameter::new_with_type_and_name(
                Type::Integer,
                SLAB_THICKNESS_IN_NM_KEY,
            ),
            run_slabs_in_parallel: EtomoBoolean2::new_with_name(RUN_SLABS_IN_PARALLEL_KEY),
            erase_fiducials: EtomoBoolean2::new_with_name(ERASE_FIDUCIALS_KEY),
            filter_in_2d: EtomoBoolean2::new_with_name(FILTER_IN_2D_KEY),
            use_unaligned_images: EtomoBoolean2::new_with_name(USE_UNALIGNED_IMAGES_KEY),
            fourier_reduce_by_factor: ScriptParameter::new_with_type_and_name(
                Type::Integer,
                FOURIER_REDUCE_BY_FACTOR_KEY,
            ),
            vertical_slices: EtomoBoolean2::new_with_name(VERTICAL_SLICES_KEY),
            old_style_xtilting: EtomoBoolean2::new_with_name(OLD_STYLE_X_TILTING_KEY),
            temporary_directory: StringParameter::new(TEMPORARY_DIRECTORY_KEY),
            tilt_command_file: StringParameter::new("TiltCommandFile"),
            number_of_processors: ScriptParameter::new_with_type_and_name(
                Type::Integer,
                "NumberOfProcessors",
            ),
            adjust_for_align_z_shift: EtomoBoolean2::new_with_name("AdjustForAlignZShift"),
            manager,
            axis_id,
        }
    }

    /// Java static `calcNumberOfSlabs(Long, double, Long)`.
    pub fn calc_number_of_slabs(
        tomo_thickness: Option<i64>,
        pixel_size_nm: f64,
        slab_thickness: Option<i64>,
    ) -> Option<i64> {
        let (Some(tomo_thickness), Some(slab_thickness)) = (tomo_thickness, slab_thickness) else {
            return None;
        };
        Some(utilities::java_lang_math_round(
            ((tomo_thickness as f64 * pixel_size_nm) / slab_thickness as f64) - 0.01,
        ))
    }

    /// Java `getTemporaryDirectory`.
    pub fn get_temporary_directory(&self) -> String {
        self.temporary_directory.to_string()
    }

    /// Java `isSlabThicknessInNmNull`.
    pub fn is_slab_thickness_in_nm_null(&self) -> bool {
        self.slab_thickness_in_nm.is_null()
    }

    /// Java `getSlabThicknessInNm`.
    pub fn get_slab_thickness_in_nm(&self) -> String {
        self.slab_thickness_in_nm.to_string()
    }

    /// Java `isRunSlabsInParallel`.
    pub fn is_run_slabs_in_parallel(&self) -> bool {
        self.run_slabs_in_parallel.is()
    }

    /// Java `isEraseFiducials`.
    pub fn is_erase_fiducials(&self) -> bool {
        self.erase_fiducials.is()
    }

    /// Java `isFilterIn2D`.
    pub fn is_filter_in_2d(&self) -> bool {
        self.filter_in_2d.is()
    }

    /// Java `isUseUnalignedImages`.
    pub fn is_use_unaligned_images(&self) -> bool {
        self.use_unaligned_images.is()
    }

    /// Java `isAdjustForAlignZShift`.
    pub fn is_adjust_for_align_z_shift(&self) -> bool {
        self.adjust_for_align_z_shift.is()
    }

    /// Java `getFourierReduceByFactor`.
    pub fn get_fourier_reduce_by_factor(&self) -> String {
        self.fourier_reduce_by_factor.to_string()
    }

    /// Java `isVerticalSlices`.
    pub fn is_vertical_slices(&self) -> bool {
        self.vertical_slices.is()
    }

    /// Java `isOldStyleXtilting`.
    pub fn is_old_style_xtilting(&self) -> bool {
        self.old_style_xtilting.is()
    }

    /// Java `setSlabThicknessInNm`.
    pub fn set_slab_thickness_in_nm(&mut self, input: Option<&str>) {
        self.slab_thickness_in_nm.set_string(input);
    }

    /// Java `setRunSlabsInParallel`.
    pub fn set_run_slabs_in_parallel(&mut self, input: bool) {
        self.run_slabs_in_parallel.set_boolean(input);
    }

    /// Java `setEraseFiducials`.
    pub fn set_erase_fiducials(&mut self, input: bool) {
        self.erase_fiducials.set_boolean(input);
    }

    /// Java `setFilterIn2D`.
    pub fn set_filter_in_2d(&mut self, input: bool) {
        self.filter_in_2d.set_boolean(input);
    }

    /// Java `setUseUnalignedImages`.
    pub fn set_use_unaligned_images(&mut self, input: bool) {
        self.use_unaligned_images.set_boolean(input);
    }

    /// Java `setAdjustForAlignZShift`.
    pub fn set_adjust_for_align_z_shift(&mut self, input: bool) {
        self.adjust_for_align_z_shift.set_boolean(input);
    }

    /// Java `setFourierReduceByFactor(Number)`.
    pub fn set_fourier_reduce_by_factor(&mut self, input: Option<Number>) {
        self.fourier_reduce_by_factor.set_number(input);
    }

    /// Java `setVerticalSlices`.
    pub fn set_vertical_slices(&mut self, input: bool) {
        self.vertical_slices.set_boolean(input);
    }

    /// Java `setOldStyleXtilting`.
    pub fn set_old_style_xtilting(&mut self, input: bool) {
        self.old_style_xtilting.set_boolean(input);
    }

    /// Java `setTemporaryDirectory`.
    pub fn set_temporary_directory(&mut self, input: Option<&str>) {
        self.temporary_directory.set(input);
    }

    /// Java `setNumberOfProcessors`.
    pub fn set_number_of_processors(&mut self, input: Option<&str>) {
        self.number_of_processors.set_string(input);
    }

    /// Java `resetNumberOfProcessors`.
    pub fn reset_number_of_processors(&mut self) {
        self.number_of_processors.reset();
    }
}

impl CommandParam for Ctf3dSetupParam {
    /// Java `initializeDefaults`.
    fn initialize_defaults(&mut self) {
        self.slab_thickness_in_nm.reset();
        self.run_slabs_in_parallel.reset();
        self.erase_fiducials.reset();
        self.filter_in_2d.reset();
        self.use_unaligned_images.reset();
        self.adjust_for_align_z_shift.reset();
        self.fourier_reduce_by_factor.reset();
        self.vertical_slices.reset();
        self.old_style_xtilting.reset();
        self.temporary_directory.reset();
        let name = file_type::CLASS
            .tilt_comscript
            .get_file_name(Some(self.manager), Some(self.axis_id));
        self.tilt_command_file.set(name.as_deref());
        self.number_of_processors.reset();
    }

    /// Java `parseComScriptCommand`.
    fn parse_com_script_command(
        &mut self,
        script_command: &ComScriptCommand,
    ) -> Result<(), ParseComScriptError> {
        self.initialize_defaults();
        // parse
        // commandFile is read-only
        self.slab_thickness_in_nm.parse(script_command)?;
        self.run_slabs_in_parallel.parse(script_command)?;
        self.erase_fiducials.parse(script_command)?;
        self.filter_in_2d.parse(script_command)?;
        self.use_unaligned_images.parse(script_command)?;
        self.adjust_for_align_z_shift.parse(script_command)?;
        self.fourier_reduce_by_factor.parse(script_command)?;
        self.vertical_slices.parse(script_command)?;
        self.old_style_xtilting.parse(script_command)?;
        self.temporary_directory.parse(script_command)?;
        self.tilt_command_file.parse(script_command)?;
        self.number_of_processors.parse(script_command)?;
        Ok(())
    }

    /// Java `updateComScriptCommand`.
    fn update_com_script_command(
        &self,
        script_command: &mut ComScriptCommand,
    ) -> Result<(), BadComScriptException> {
        script_command.use_keyword_value();
        self.slab_thickness_in_nm.update_com_script(script_command);
        self.run_slabs_in_parallel.update_com_script(script_command);
        self.erase_fiducials.update_com_script(script_command);
        self.filter_in_2d.update_com_script(script_command);
        self.use_unaligned_images.update_com_script(script_command);
        self.adjust_for_align_z_shift
            .update_com_script(script_command);
        self.fourier_reduce_by_factor
            .update_com_script(script_command);
        self.vertical_slices.update_com_script(script_command);
        self.old_style_xtilting.update_com_script(script_command);
        self.temporary_directory.update_com_script(script_command);
        self.tilt_command_file.update_com_script(script_command);
        self.number_of_processors.update_com_script(script_command);
        Ok(())
    }
}

impl Command for Ctf3dSetupParam {
    fn get_axis_id(&self) -> AxisID {
        self.axis_id
    }

    fn get_command(&self) -> Option<String> {
        Some(PROCESS_NAME.get_comscript(self.axis_id))
    }

    fn get_command_array(&self) -> Option<Vec<String>> {
        Some(vec![PROCESS_NAME.get_comscript(self.axis_id)])
    }

    fn get_command_output_file(&self) -> Option<std::path::PathBuf> {
        None
    }

    fn get_command_input_file(&self) -> Option<std::path::PathBuf> {
        file_type::CLASS
            .tilt_comscript
            .get_file(Some(self.manager), Some(self.axis_id))
    }

    fn get_command_line(&self) -> Option<String> {
        self.get_command()
    }

    fn get_command_mode(&self) -> Option<&dyn CommandMode> {
        None
    }

    fn get_command_name(&self) -> Option<String> {
        // In this case the .com file name and the command in the .com file are the same.
        Some(ProcessName::CTF_3D.to_string())
    }

    /// Java returns `OUTPUT_IMAGE_FILE_TYPE` (`FileType.CTF_3D_OUTPUT`) as a `FileKey`.
    fn get_output_image_file_key(&self) -> Option<FileKey> {
        Some(FileKey::clone(&file_type::CLASS.ctf_3d_output))
    }

    fn get_output_image_file_key2(&self) -> Option<FileKey> {
        None
    }

    /// Deprecated 3/15/2019.  Java returns `OUTPUT_IMAGE_FILE_TYPE`.
    fn get_output_image_file_type(&self) -> Option<std::sync::Arc<FileType>> {
        Some(std::sync::Arc::clone(&file_type::CLASS.ctf_3d_output))
    }

    /// Deprecated 3/15/2019.
    fn get_output_image_file_type2(&self) -> Option<std::sync::Arc<FileType>> {
        None
    }

    fn get_process_name(&self) -> Option<ProcessName> {
        Some(ProcessName::CTF_3D_SETUP)
    }

    fn get_subcommand_details(&self) -> Option<&dyn CommandDetails> {
        None
    }

    fn get_subcommand_process_name(&self) -> Option<String> {
        None
    }

    fn is_message_reporter(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn number_of_slabs_is_rounded_less_a_hundredth() {
        assert_eq!(
            Ctf3dSetupParam::calc_number_of_slabs(None, 1.0, Some(15)),
            None
        );
        assert_eq!(
            Ctf3dSetupParam::calc_number_of_slabs(Some(300), 1.0, None),
            None
        );
        // 300 * 1.0 / 15 - .01 = 19.99 -> 20
        assert_eq!(
            Ctf3dSetupParam::calc_number_of_slabs(Some(300), 1.0, Some(15)),
            Some(20)
        );
        // 45 / 10 - .01 = 4.49 -> 4
        assert_eq!(
            Ctf3dSetupParam::calc_number_of_slabs(Some(45), 1.0, Some(10)),
            Some(4)
        );
    }
}
