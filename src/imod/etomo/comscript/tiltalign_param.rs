//! `IMOD/Etomo/src/etomo/comscript/TiltalignParam.java`.
//!
//! **Representation.**  `TiltalignParam extends ConstTiltalignParam`: the superclass
//! state is `ConstTiltalignParam` (`const_tiltalign_param.rs`), held in `base` and
//! reached through `Deref`/`DerefMut`, as `etomo/type/script_parameter.rs` does for its
//! superclass.  The inherited `Command`, `ProcessDetails` and `Loggable` methods are
//! forwarded to `base` so a `TiltalignParam` can be handed to a process as
//! `Arc<dyn Command + Send + Sync>`.

use std::path::PathBuf;
use std::sync::Arc;

use super::bad_com_script_exception::BadComScriptException;
use super::com_script_command::ComScriptCommand;
use super::command::Command;
use super::command_details::CommandDetails;
use super::command_mode::CommandMode;
use super::command_param::{CommandParam, ParseComScriptError};
use super::const_tiltalign_param::*;
use super::field_interface::FieldInterface;
use super::fortran_input_string::FortranInputString;
use super::fortran_input_syntax_exception::FortranInputSyntaxException;
use super::invalid_parameter_exception::InvalidParameterException;
use super::old_tiltalign_param::OldTiltalignParam;
use super::param_utilities;
use super::process_details::ProcessDetails;
use super::tiltalign_solution::TiltalignSolution;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::storage::loggable::{Loggable, LoggableException};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::{
    ConstEtomoNumber, java_lang_string_matches_whitespace,
};
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::r#type::file_key::FileKey;
use crate::imod::etomo::r#type::file_type::{self, FileType};
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::x_tilt_option::XTiltOption;
use crate::imod::etomo::util::dataset_files;
use crate::imod::etomo::util::mrc_header::MRCHeader;
use crate::imod::etomo::util::utilities;

const X_TILT_DEFAULT_GROUPING_DEFAULT: i32 = 2000;

/// Java final `TiltalignParam`.
pub struct TiltalignParam {
    /// Java superclass `ConstTiltalignParam` state.
    pub base: ConstTiltalignParam,
    /// Java private field `z`, which defaults to null.
    z: Option<EtomoNumber>,
}

/// Java inheritance: every `ConstTiltalignParam` member is reachable on a
/// `TiltalignParam`.
impl std::ops::Deref for TiltalignParam {
    type Target = ConstTiltalignParam;

    fn deref(&self) -> &ConstTiltalignParam {
        &self.base
    }
}

impl std::ops::DerefMut for TiltalignParam {
    fn deref_mut(&mut self) -> &mut ConstTiltalignParam {
        &mut self.base
    }
}

/// `outputModelFile.matches("\\s*")` on a `String` field.  The field is null when the
/// com script has the keyword without a value, and the Java throws
/// `NullPointerException` there (TiltalignParam.java:62-69); upstream bug fixed in
/// translation: a null value is treated as blank.
fn is_blank(value: &Option<String>) -> bool {
    java_lang_string_matches_whitespace(value.as_deref().unwrap_or(""))
}

impl TiltalignParam {
    /// Java `TiltalignParam(BaseManager, String, AxisID)`.
    pub fn new(
        manager: &'static dyn BaseManager,
        dataset_name: Option<&str>,
        axis_id: AxisID,
    ) -> TiltalignParam {
        TiltalignParam {
            base: ConstTiltalignParam::new(manager, dataset_name, axis_id),
            z: None,
        }
    }

    /// Java private `convertToPIP`.  Convert from the old style script to PIP:
    /// modelFile = modelFile, imageFile = imageFile, not using imageParameters,
    /// outputModelFile = imodFiducialPosFile + .3dmod, outputResidualFile =
    /// imodFiducialPosFile + .resid, outputFidXYZFile = asciiFiducialPosFile,
    /// outputTiltFile = tiltAngleSolutionFile, outputTransformFile =
    /// transformSolutionFile, outputZFactorFile new, ... (see the Java comment for the
    /// full mapping).
    fn convert_to_pip(
        &mut self,
        old_param: &OldTiltalignParam,
    ) -> Result<(), FortranInputSyntaxException> {
        // Java string concatenation writes a null name as "null".
        let null = |value: Option<&str>| value.unwrap_or("null").to_owned();
        let base = &mut self.base;
        base.model_file = old_param.get_model_file().map(str::to_owned);
        base.image_file = old_param.get_image_file().map(str::to_owned);
        // OldTiltParam only looks for IMODFiducialPosFile. It does not check for
        // the model file or the residual file
        base.output_model_and_residual = old_param.get_imod_fiducial_pos_file().map(str::to_owned);
        base.output_model_file =
            Some(null(old_param.get_imod_fiducial_pos_file()) + MODEL_FILE_EXTENSION);
        base.output_residual_file =
            Some(null(old_param.get_imod_fiducial_pos_file()) + RESIDUAL_FILE_EXTENSION);
        base.output_fid_xyz_file = old_param.get_ascii_fiducial_pos_file().map(str::to_owned);
        base.output_tilt_file = old_param.get_tilt_angle_solution_file().map(str::to_owned);
        base.output_transform_file = old_param.get_transform_solution_file().map(str::to_owned);
        // Set ExcludeList
        // Kept as the Java writes it: all three include/exclude types go to
        // excludeList, although the Java comment maps 1 to includeStartEndInc and 2 to
        // includeList.
        let include_exclude_type = old_param.get_include_exclude_type();
        if include_exclude_type == 1 {
            base.exclude_list = old_param.get_include_exclude_list().clone();
        } else if include_exclude_type == 2 {
            base.exclude_list = old_param.get_include_exclude_list().clone();
        } else if include_exclude_type == 3 {
            base.exclude_list = old_param.get_include_exclude_list().clone();
        }

        base.rotation_angle
            .set_double(old_param.get_initial_image_rotation());
        base.separate_group = old_param.get_separate_view_groups().clone();
        base.separate_group.set_key(Some(SEPARATE_GROUP_KEY));
        base.separate_group.set_successive_entries_accumulate();
        base.tilt_angle_spec
            .set(Some(old_param.get_tilt_angle_spec()));
        base.angle_offset
            .set_double(old_param.get_tilt_angle_offset());
        // Set RotationAngleSolutionType and RotationFixedView
        let rotation_angle_solution_type = old_param.get_rotation_angle_solution_type();
        if rotation_angle_solution_type >= 0 {
            base.rot_option.set_int(1);
            if rotation_angle_solution_type > 0 {
                base.rotation_fixed_view
                    .set_int(rotation_angle_solution_type);
            }
        } else if rotation_angle_solution_type == -2 {
            base.rot_option.set_int(0);
        } else {
            base.rot_option.set_int(rotation_angle_solution_type);
        }

        base.local_rot_nondefault_group = TiltalignParam::set_solution(
            Some(&mut *base.local_rot_option),
            Some(&mut *base.local_rot_default_grouping),
            None,
            Some(old_param.get_local_rotation_solution()),
        )?;
        base.tilt_nondefault_group = TiltalignParam::set_solution(
            Some(&mut *base.tilt_option),
            Some(&mut *base.tilt_default_grouping),
            None,
            Some(old_param.get_tilt_angle_solution()),
        )?;
        base.local_tilt_nondefault_group = TiltalignParam::set_solution(
            Some(&mut *base.local_tilt_option),
            Some(&mut *base.local_tilt_default_grouping),
            None,
            Some(old_param.get_local_tilt_solution()),
        )?;
        base.mag_nondefault_group = TiltalignParam::set_solution(
            Some(&mut *base.mag_option),
            Some(&mut *base.mag_default_grouping),
            Some(&mut *base.mag_reference_view),
            Some(old_param.get_magnification_solution()),
        )?;
        base.local_mag_nondefault_group = TiltalignParam::set_solution(
            Some(&mut *base.local_mag_option),
            Some(&mut *base.local_mag_default_grouping),
            Some(&mut *base.local_mag_reference_view),
            Some(old_param.get_local_magnification_solution()),
        )?;
        base.x_stretch_nondefault_group = TiltalignParam::set_solution(
            Some(&mut *base.x_stretch_option),
            Some(&mut *base.x_stretch_default_grouping),
            None,
            Some(old_param.get_xstretch_solution()),
        )?;
        base.local_x_stretch_nondefault_group = TiltalignParam::set_solution(
            Some(&mut *base.local_x_stretch_option),
            Some(&mut *base.local_x_stretch_default_grouping),
            None,
            Some(old_param.get_local_xstretch_solution()),
        )?;
        base.skew_nondefault_group = TiltalignParam::set_solution(
            Some(&mut *base.skew_option),
            Some(&mut *base.skew_default_grouping),
            None,
            Some(old_param.get_skew_solution()),
        )?;
        base.local_skew_nondefault_group = TiltalignParam::set_solution(
            Some(&mut *base.local_skew_option),
            Some(&mut *base.local_skew_default_grouping),
            None,
            Some(old_param.get_local_skew_solution()),
        )?;
        base.residual_report_criterion
            .set_double(old_param.get_residual_threshold());
        base.surfaces_to_analyze
            .set_int(old_param.get_n_surface_analysis());
        base.metro_factor.set_double(old_param.get_metro_factor());
        base.maximum_cycles.set_int(old_param.get_cycle_limit());
        base.axis_z_shift
            .set_double(old_param.get_tilt_axis_z_shift());
        base.local_alignments
            .set_boolean(old_param.get_local_alignments());
        base.output_local_file = old_param.get_local_transform_file().map(str::to_owned);
        base.number_of_local_patches_xand_y = old_param.get_n_local_patches().clone();
        base.min_size_or_overlap_xand_y = old_param.get_min_local_patch_size().clone();
        base.min_fids_total_and_each_surface = old_param.get_min_local_fiducials().clone();
        base.fix_xyz_coordinates
            .set_boolean(old_param.get_fix_local_fiducial_coodinates());
        base.local_output_options = old_param.get_local_output_selection().clone();
        // set state
        self.set_output_z_factor_file();
        self.base.loaded_from_file = true;
        Ok(())
    }

    /// Java private `setSolution`.  It reads no instance state; it is an associated
    /// function so the fields it sets can be passed in while the others are borrowed.
    fn set_solution(
        option: Option<&mut EtomoNumber>,
        default_grouping: Option<&mut EtomoNumber>,
        reference_view: Option<&mut EtomoNumber>,
        solution: Option<&TiltalignSolution>,
    ) -> Result<Option<Vec<FortranInputString>>, FortranInputSyntaxException> {
        let solution = match solution {
            None => return Ok(None),
            Some(solution) => solution,
        };
        if let Some(option) = option {
            option.set_int(solution.r#type);
        }
        if let Some(default_grouping) = default_grouping
            && solution.params.size() > 0
            && !solution.params.is_default_index(0)
        {
            default_grouping.set_int(solution.params.get_int(0));
        }
        if let Some(reference_view) = reference_view
            && solution.reference_view.size() > 0
            && !solution.reference_view.is_default_index(0)
        {
            reference_view.set_int(solution.reference_view.get_int(0));
        }
        param_utilities::parse_string_list(
            Some(&solution.additional_groups),
            &NONDEFAULT_GROUP_INTEGER_TYPE,
            NONDEFAULT_GROUP_SIZE,
        )
    }

    /// Java `setAngleOffset`.
    pub fn set_angle_offset(&mut self, angle_offset: Option<&str>) {
        self.base.angle_offset.set_string(angle_offset);
    }

    /// Java `setAxisZShift`.
    pub fn set_axis_z_shift(&mut self, axis_z_shift: Option<&str>) {
        self.base.axis_z_shift.set_string(axis_z_shift);
    }

    /// Java `setExcludeList`.
    pub fn set_exclude_list(&mut self, exclude_list: Option<&str>) {
        self.base.exclude_list.parse_string(exclude_list);
    }

    /// Java `updateImageFile`.
    pub fn update_image_file(&mut self) {
        let manager = self.base.manager;
        let axis_id = self.base.axis_id;
        let image_file = &file_type::CLASS.prealigned_stack;
        if image_file.exists(Some(manager), Some(axis_id)) {
            if utilities::is_empty(self.base.image_file.as_deref()) {
                self.base.image_file = image_file.get_file_name(Some(manager), Some(axis_id));
            }
        } else {
            self.base.image_file = Some(String::new());
        }
    }

    /// Java `setImagesAreBinned`.
    pub fn set_images_are_binned(&mut self, images_are_binned: i32) {
        self.base.images_are_binned.set_int(images_are_binned);
    }

    /// Java `setBeamTiltOption`.
    pub fn set_beam_tilt_option(&mut self, beam_tilt_option: i32) {
        self.base.beam_tilt_option.set_int(beam_tilt_option);
    }

    /// Java `setFixedOrInitialBeamTilt`.
    pub fn set_fixed_or_initial_beam_tilt(&mut self, fixed_or_initial_beam_tilt: Option<&str>) {
        self.base
            .fixed_or_initial_beam_tilt
            .get_mut()
            .unwrap()
            .set_string(fixed_or_initial_beam_tilt);
    }

    /// Java `resetFixedOrInitialBeamTilt`.
    pub fn reset_fixed_or_initial_beam_tilt(&mut self) {
        self.base
            .fixed_or_initial_beam_tilt
            .get_mut()
            .unwrap()
            .reset();
    }

    /// Java `resetWeightWholeTracks`.
    pub fn reset_weight_whole_tracks(&mut self) {
        self.base.weight_whole_tracks.reset();
    }

    /// Java `setLocalAlignments`.
    pub fn set_local_alignments(&mut self, local_alignments: bool) {
        self.base.local_alignments.set_boolean(local_alignments);
    }

    /// Java `setLocalMagDefaultGrouping`.
    pub fn set_local_mag_default_grouping(&mut self, local_mag_default_grouping: Option<&str>) {
        self.base
            .local_mag_default_grouping
            .set_string(local_mag_default_grouping);
    }

    /// Java `setLocalMagNondefaultGroup`.
    pub fn set_local_mag_nondefault_group(
        &mut self,
        local_mag_nondefault_group: Option<&str>,
    ) -> Result<(), FortranInputSyntaxException> {
        self.base.local_mag_nondefault_group =
            param_utilities::parse_string(local_mag_nondefault_group, true, NONDEFAULT_GROUP_SIZE)?;
        Ok(())
    }

    /// Java `setLocalMagOption`.
    pub fn set_local_mag_option(&mut self, local_mag_option: i32) {
        self.base.local_mag_option.set_int(local_mag_option);
    }

    /// Java `setLocalRotDefaultGrouping`.
    pub fn set_local_rot_default_grouping(&mut self, local_rot_default_grouping: Option<&str>) {
        self.base
            .local_rot_default_grouping
            .set_string(local_rot_default_grouping);
    }

    /// Java `setLocalRotNondefaultGroup`.
    pub fn set_local_rot_nondefault_group(
        &mut self,
        local_rot_nondefault_group: Option<&str>,
    ) -> Result<(), FortranInputSyntaxException> {
        self.base.local_rot_nondefault_group =
            param_utilities::parse_string(local_rot_nondefault_group, true, NONDEFAULT_GROUP_SIZE)?;
        Ok(())
    }

    /// Java `setLocalRotOption`.
    pub fn set_local_rot_option(&mut self, local_rot_option: i32) {
        self.base.local_rot_option.set_int(local_rot_option);
    }

    /// Java `setLocalSkewDefaultGrouping`.
    pub fn set_local_skew_default_grouping(&mut self, local_skew_default_grouping: Option<&str>) {
        self.base
            .local_skew_default_grouping
            .set_string(local_skew_default_grouping);
    }

    /// Java `setLocalSkewNondefaultGroup`.
    pub fn set_local_skew_nondefault_group(
        &mut self,
        local_skew_nondefault_group: Option<&str>,
    ) -> Result<(), FortranInputSyntaxException> {
        self.base.local_skew_nondefault_group = param_utilities::parse_string(
            local_skew_nondefault_group,
            true,
            NONDEFAULT_GROUP_SIZE,
        )?;
        Ok(())
    }

    /// Java `setLocalSkewOption`.
    pub fn set_local_skew_option(&mut self, local_skew_option: i32) {
        self.base.local_skew_option.set_int(local_skew_option);
    }

    /// Java `setLocalTiltDefaultGrouping`.
    pub fn set_local_tilt_default_grouping(&mut self, local_tilt_default_grouping: Option<&str>) {
        self.base
            .local_tilt_default_grouping
            .set_string(local_tilt_default_grouping);
    }

    /// Java `setLocalTiltNondefaultGroup`.
    pub fn set_local_tilt_nondefault_group(
        &mut self,
        local_tilt_nondefault_group: Option<&str>,
    ) -> Result<(), FortranInputSyntaxException> {
        self.base.local_tilt_nondefault_group = param_utilities::parse_string(
            local_tilt_nondefault_group,
            true,
            NONDEFAULT_GROUP_SIZE,
        )?;
        Ok(())
    }

    /// Java `setLocalTiltOption`.
    pub fn set_local_tilt_option(&mut self, local_tilt_option: i32) {
        self.base.local_tilt_option.set_int(local_tilt_option);
    }

    /// Java `setLocalXStretchDefaultGrouping`.
    pub fn set_local_x_stretch_default_grouping(
        &mut self,
        local_x_stretch_default_grouping: Option<&str>,
    ) {
        self.base
            .local_x_stretch_default_grouping
            .set_string(local_x_stretch_default_grouping);
    }

    /// Java `setLocalXStretchNondefaultGroup`.
    pub fn set_local_x_stretch_nondefault_group(
        &mut self,
        local_x_stretch_nondefault_group: Option<&str>,
    ) -> Result<(), FortranInputSyntaxException> {
        self.base.local_x_stretch_nondefault_group = param_utilities::parse_string(
            local_x_stretch_nondefault_group,
            true,
            NONDEFAULT_GROUP_SIZE,
        )?;
        Ok(())
    }

    /// Java `setLocalXStretchOption`.
    pub fn set_local_x_stretch_option(&mut self, local_x_stretch_option: i32) {
        self.base
            .local_x_stretch_option
            .set_int(local_x_stretch_option);
    }

    /// Java `setMagDefaultGrouping`.
    pub fn set_mag_default_grouping(&mut self, mag_default_grouping: Option<&str>) {
        self.base
            .mag_default_grouping
            .set_string(mag_default_grouping);
    }

    /// Java `setMagNondefaultGroup`.
    pub fn set_mag_nondefault_group(
        &mut self,
        mag_nondefault_group: Option<&str>,
    ) -> Result<(), FortranInputSyntaxException> {
        self.base.mag_nondefault_group =
            param_utilities::parse_string(mag_nondefault_group, true, NONDEFAULT_GROUP_SIZE)?;
        Ok(())
    }

    /// Java `setMagOption`.
    pub fn set_mag_option(&mut self, mag_option: i32) {
        self.base.mag_option.set_int(mag_option);
    }

    /// Java `setMagReferenceView`.
    pub fn set_mag_reference_view(&mut self, mag_reference_view: Option<&str>) {
        self.base.mag_reference_view.set_string(mag_reference_view);
    }

    /// Java `setMaximumCycles`.
    pub fn set_maximum_cycles(&mut self, maximum_cycles: Option<&str>) {
        self.base.maximum_cycles.set_string(maximum_cycles);
    }

    /// Java `setMetroFactor`.
    pub fn set_metro_factor(&mut self, metro_factor: Option<&str>) {
        self.base.metro_factor.set_string(metro_factor);
    }

    /// Java `setMinFidsTotalAndEachSurface`.
    pub fn set_min_fids_total_and_each_surface(
        &mut self,
        min_fids_total_and_each_surface: Option<&str>,
    ) -> Result<(), FortranInputSyntaxException> {
        self.base
            .min_fids_total_and_each_surface
            .validate_and_set(min_fids_total_and_each_surface)
    }

    /// Java `setFixXYZCoordinates`.
    pub fn set_fix_xyz_coordinates(&mut self, fix_xyz_coordinates: bool) -> &ConstEtomoNumber {
        self.base
            .fix_xyz_coordinates
            .set_boolean(fix_xyz_coordinates)
    }

    /// Java `setMinSizeOrOverlapXandY`.
    pub fn set_min_size_or_overlap_xand_y(
        &mut self,
        min_size_or_overlap_xand_y: Option<&str>,
    ) -> Result<(), FortranInputSyntaxException> {
        self.base
            .min_size_or_overlap_xand_y
            .validate_and_set(min_size_or_overlap_xand_y)
    }

    /// Java `setModelFile`.
    pub fn set_model_file(&mut self, model_file: Option<&str>) {
        self.base.model_file = model_file.map(str::to_owned);
    }

    /// Java `setNumberOfLocalPatchesXandY`.
    pub fn set_number_of_local_patches_xand_y(
        &mut self,
        number_of_local_patches_xand_y: Option<&str>,
    ) -> Result<(), FortranInputSyntaxException> {
        self.base
            .number_of_local_patches_xand_y
            .validate_and_set(number_of_local_patches_xand_y)
    }

    /// Java `setTargetPatchSizeXandY`.
    pub fn set_target_patch_size_xand_y(
        &mut self,
        target_patch_size_xand_y: Option<&str>,
    ) -> Result<(), FortranInputSyntaxException> {
        self.base
            .target_patch_size_xand_y
            .validate_and_set(target_patch_size_xand_y)
    }

    /// Java `setNumberOfLocalPatchesXandYActive`.
    pub fn set_number_of_local_patches_xand_y_active(&mut self, active: bool) {
        self.base.number_of_local_patches_xand_y.set_active(active);
    }

    /// Java `setTargetPatchSizeXandYActive`.
    pub fn set_target_patch_size_xand_y_active(&mut self, active: bool) {
        self.base.target_patch_size_xand_y.set_active(active);
    }

    /// Java `setOutputFidXYZFile`.
    pub fn set_output_fid_xyz_file(&mut self, output_fid_xyz_file: Option<&str>) {
        self.base.output_fid_xyz_file = output_fid_xyz_file.map(str::to_owned);
    }

    /// Java `setOutputLocalFile(String)`.
    pub fn set_output_local_file_string(&mut self, output_local_file: Option<&str>) {
        self.base.output_local_file = output_local_file.map(str::to_owned);
    }

    /// Java `setOutputModelFile`.
    pub fn set_output_model_file(&mut self, output_model_file: Option<&str>) {
        self.base.output_model_file = output_model_file.map(str::to_owned);
    }

    /// Java `setOutputResidualFile`.
    pub fn set_output_residual_file(&mut self, output_residual_file: Option<&str>) {
        self.base.output_residual_file = output_residual_file.map(str::to_owned);
    }

    /// Java `setOutputTiltFile`.
    pub fn set_output_tilt_file(&mut self, output_tilt_file: Option<&str>) {
        self.base.output_tilt_file = output_tilt_file.map(str::to_owned);
    }

    /// Java `setOutputTransformFile`.
    pub fn set_output_transform_file(&mut self, output_transform_file: Option<&str>) {
        self.base.output_transform_file = output_transform_file.map(str::to_owned);
    }

    /// Java `setOutputZFactorFile`.  This must called after skewOption, or
    /// localAlignment, and localSkewOption have been set.
    pub fn set_output_z_factor_file(&mut self) {
        if self.use_output_z_factor_file() {
            self.base.output_z_factor_file =
                Some(ConstTiltalignParam::get_output_z_factor_file_name(
                    self.base.dataset_name.as_deref(),
                    self.base.axis_id,
                ));
        } else {
            self.base.output_z_factor_file = Some(String::new());
        }
    }

    /// Java `setOutputLocalFile()`.
    pub fn set_output_local_file(&mut self) {
        if is_blank(&self.base.output_local_file) {
            self.base.output_local_file = Some(ConstTiltalignParam::get_output_local_file_name(
                self.base.dataset_name.as_deref(),
                self.base.axis_id,
            ));
        }
    }

    /// Java `setProjectionStretch`.
    pub fn set_projection_stretch(&mut self, projection_stretch: bool) {
        self.base.projection_stretch.set_boolean(projection_stretch);
    }

    /// Java `setResidualReportCriterion`.
    pub fn set_residual_report_criterion(&mut self, residual_report_criterion: f64) {
        self.base
            .residual_report_criterion
            .set_double(residual_report_criterion);
    }

    /// Java `setRotationAngle`.
    pub fn set_rotation_angle(&mut self, rotation_angle: Option<&str>) {
        self.base.rotation_angle.set_string(rotation_angle);
    }

    /// Java `setRotationFixedView`.
    pub fn set_rotation_fixed_view(&mut self, rotation_fixed_view: i32) {
        self.base.rotation_fixed_view.set_int(rotation_fixed_view);
    }

    /// Java `setRotDefaultGrouping`.
    pub fn set_rot_default_grouping(&mut self, rot_default_grouping: Option<&str>) {
        self.base
            .rot_default_grouping
            .set_string(rot_default_grouping);
    }

    /// Java `setRotNondefaultGroup`.
    pub fn set_rot_nondefault_group(
        &mut self,
        rot_nondefault_group: Option<&str>,
    ) -> Result<(), FortranInputSyntaxException> {
        self.base.rot_nondefault_group =
            param_utilities::parse_string(rot_nondefault_group, true, NONDEFAULT_GROUP_SIZE)?;
        Ok(())
    }

    /// Java `setRotOption`.
    pub fn set_rot_option(&mut self, rot_option: i32) {
        self.base.rot_option.set_int(rot_option);
    }

    /// Java `setSeparateGroup`.
    pub fn set_separate_group(&mut self, separate_group: Option<&str>) {
        self.base.separate_group.parse_string(separate_group);
    }

    /// Java `setSkewDefaultGrouping`.
    pub fn set_skew_default_grouping(&mut self, skew_default_grouping: Option<&str>) {
        self.base
            .skew_default_grouping
            .set_string(skew_default_grouping);
    }

    /// Java `setSkewNondefaultGroup`.
    pub fn set_skew_nondefault_group(
        &mut self,
        skew_nondefault_group: Option<&str>,
    ) -> Result<(), FortranInputSyntaxException> {
        self.base.skew_nondefault_group =
            param_utilities::parse_string(skew_nondefault_group, true, NONDEFAULT_GROUP_SIZE)?;
        Ok(())
    }

    /// Java `setSkewOption`.
    pub fn set_skew_option(&mut self, skew_option: i32) {
        self.base.skew_option.set_int(skew_option);
    }

    /// Java `setSurfacesToAnalyze`.
    pub fn set_surfaces_to_analyze(&mut self, surfaces_to_analyze: i32) {
        self.base.surfaces_to_analyze.set_int(surfaces_to_analyze);
    }

    /// Java `setRobustFitting`.
    pub fn set_robust_fitting(&mut self, input: bool) {
        self.base.robust_fitting.set_boolean(input);
    }

    /// Java `setWeightWholeTracks`.
    pub fn set_weight_whole_tracks(&mut self, input: bool) {
        self.base.weight_whole_tracks.set_boolean(input);
    }

    /// Java `setCrossValidate`.
    pub fn set_cross_validate(&mut self, input: bool) {
        if input {
            self.base.cross_validate.set_int(1);
        } else {
            self.base.cross_validate.reset();
        }
    }

    /// Java `setKFactorScaling`.
    pub fn set_k_factor_scaling(&mut self, input: Option<&str>) {
        self.base.k_factor_scaling.set_string(input);
    }

    /// Java `setTiltDefaultGrouping`.
    pub fn set_tilt_default_grouping(&mut self, tilt_default_grouping: Option<&str>) {
        self.base
            .tilt_default_grouping
            .set_string(tilt_default_grouping);
    }

    /// Java `setTiltNondefaultGroup`.
    pub fn set_tilt_nondefault_group(
        &mut self,
        tilt_nondefault_group: Option<&str>,
    ) -> Result<(), FortranInputSyntaxException> {
        self.base.tilt_nondefault_group =
            param_utilities::parse_string(tilt_nondefault_group, true, NONDEFAULT_GROUP_SIZE)?;
        Ok(())
    }

    /// Java `setTiltOption`.
    pub fn set_tilt_option(&mut self, tilt_option: i32) {
        self.base.tilt_option.set_int(tilt_option);
    }

    /// Java `setXStretchDefaultGrouping`.
    pub fn set_x_stretch_default_grouping(&mut self, stretch_default_grouping: Option<&str>) {
        self.base
            .x_stretch_default_grouping
            .set_string(stretch_default_grouping);
    }

    /// Java `setXStretchNondefaultGroup`.
    pub fn set_x_stretch_nondefault_group(
        &mut self,
        stretch_nondefault_group: Option<&str>,
    ) -> Result<(), FortranInputSyntaxException> {
        self.base.x_stretch_nondefault_group =
            param_utilities::parse_string(stretch_nondefault_group, true, NONDEFAULT_GROUP_SIZE)?;
        Ok(())
    }

    /// Java `setXStretchOption`.
    pub fn set_x_stretch_option(&mut self, stretch_option: i32) {
        self.base.x_stretch_option.set_int(stretch_option);
    }

    /// Java `setXTiltAutomapSameLargeGroup`.
    pub fn set_x_tilt_automap_same_large_group(&mut self, input: bool) {
        if input {
            self.base
                .x_tilt_option
                .set_int(XTiltOption::AUTOMAP_SAME.get_option());
            if self.z.is_none() {
                let mut z = EtomoNumber::new();
                let manager = self.base.manager;
                let header = MRCHeader::get_instance_from_file_type(
                    manager,
                    Some(self.base.axis_id),
                    &file_type::CLASS.prealigned_stack,
                );
                // The Java calls `header.read(manager)` before its `header != null`
                // test (TiltalignParam.java:969-970), so a null header is a
                // NullPointerException.  Upstream bug fixed in translation: a null
                // header leaves z null-valued.
                if let Some(header) = header {
                    let read = header.borrow_mut().read_with_manager(manager);
                    match read {
                        Ok(_) => {
                            let n_sections = header.borrow().get_n_sections();
                            z.set_int(n_sections);
                        }
                        // `e.printStackTrace()` for InvalidParameterException and
                        // IOException.
                        Err(e) => eprintln!("{e}"),
                    }
                }
                self.z = Some(z);
            }
            // large group at least as big as Z
            if let Some(z) = &self.z
                && !z.is_null()
                && self
                    .base
                    .x_tilt_default_grouping
                    .lt_const_etomo_number(Some(&**z))
            {
                self.base
                    .x_tilt_default_grouping
                    .set_const_etomo_number(Some(&**z));
            } else if self.base.x_tilt_default_grouping.is_null() {
                self.base
                    .x_tilt_default_grouping
                    .set_int(X_TILT_DEFAULT_GROUPING_DEFAULT);
            }
        } else {
            self.base
                .x_tilt_option
                .set_int(XTiltOption::FIX.get_option());
        }
    }

    /// Java `upgradeOldVersion`.  Backward compatibility fix.  Unbinned all the
    /// parameters which where binned in the old version.  Ignore parameters with reset
    /// values.  The param should be loaded from a com file before running this
    /// function.  Returns true if changes where made.
    pub fn upgrade_old_version(&mut self, correction_binning: i32, current_binning: i32) -> bool {
        if !self.is_old_version() {
            return false;
        }
        // Set the binning to prevent this function from being called again
        self.base.images_are_binned.set_int(current_binning);
        // Currently this function only multiplies by binning, so there is nothing to
        // do if binning is 1.
        if correction_binning != 1
            && !self.base.axis_z_shift.is_null()
            && !self.base.axis_z_shift.equals_int(0)
        {
            let value = self.base.axis_z_shift.get_double() * correction_binning as f64;
            self.base.axis_z_shift.set_double(value);
        }
        let mut buffer = format!(
            "\nUpgraded align{}.com:\n",
            self.base.axis_id.get_extension()
        );
        if correction_binning > 1 {
            buffer.push_str(&format!(
                "Multiplied binned {} by {}.\n",
                self.base.axis_z_shift.get_name(),
                correction_binning
            ));
        }
        buffer.push_str(&format!(
            "Added {} {}.\n",
            self.base.images_are_binned.get_name(),
            current_binning
        ));
        eprintln!("{buffer}");
        true
    }
}

impl CommandParam for TiltalignParam {
    /// Java `parseComScriptCommand`.  Get the parameters from the ComScriptCommand.
    fn parse_com_script_command(
        &mut self,
        script_command: &ComScriptCommand,
    ) -> Result<(), ParseComScriptError> {
        self.base.reset();
        if !script_command.is_keyword_value_pairs() {
            let mut old_param = OldTiltalignParam::new();
            old_param.parse_com_script_command(script_command)?;
            self.convert_to_pip(&old_param)?;
        } else {
            let base = &mut self.base;
            base.images_are_binned.parse(script_command)?;
            base.model_file = script_command.get_value(Some(MODEL_FILE_STRING))?;
            base.image_file = script_command.get_value(Some(IMAGE_FILE_STRING))?;
            base.output_model_and_residual =
                script_command.get_value(Some(OUTPUT_MODEL_AND_RESIDUAL_STRING))?;
            base.output_model_file = script_command.get_value(Some(OUTPUT_MODEL_FILE_STRING))?;
            base.output_residual_file =
                script_command.get_value(Some(OUTPUT_RESIDUAL_FILE_STRING))?;

            // Use OutputModelAndResidual if OutputModelFile or OutputResidualFile are blank
            // Convert OutputModelAndResidual to OutputModelFile and OutputResidualFile
            if is_blank(&base.output_model_file) || is_blank(&base.output_residual_file) {
                let output_model_and_residual =
                    script_command.get_value(Some(OUTPUT_MODEL_AND_RESIDUAL_STRING))?;
                // Java string concatenation writes a null value as "null".
                let output_model_and_residual =
                    output_model_and_residual.unwrap_or_else(|| "null".to_owned());
                if is_blank(&base.output_model_file) {
                    base.output_model_file =
                        Some(output_model_and_residual.clone() + MODEL_FILE_EXTENSION);
                }
                if is_blank(&base.output_residual_file) {
                    base.output_residual_file =
                        Some(output_model_and_residual + RESIDUAL_FILE_EXTENSION);
                }
            }

            base.output_fid_xyz_file =
                script_command.get_value(Some(OUTPUT_FID_XYZ_FILE_STRING))?;
            base.output_tilt_file = script_command.get_value(Some(OUTPUT_TILT_FILE_STRING))?;
            base.output_transform_file =
                script_command.get_value(Some(OUTPUT_TRANSFORM_FILE_STRING))?;
            base.output_z_factor_file =
                script_command.get_value(Some(OUTPUT_Z_FACTOR_FILE_STRING))?;
            param_utilities::set_param_if_present_fortran_input_string(
                script_command,
                INCLUDE_START_END_INC_STRING,
                &mut base.include_start_end_inc,
            )?;
            base.include_list.parse_string(
                script_command
                    .get_value(Some(INCLUDE_LIST_STRING))?
                    .as_deref(),
            );
            base.exclude_list
                .parse_string(script_command.get_value(Some(EXCLUDE_LIST_KEY))?.as_deref());
            base.rotation_angle.parse(script_command)?;
            base.separate_group
                .parse_string_array(Some(&script_command.get_values(Some(SEPARATE_GROUP_KEY))));
            base.tilt_angle_spec.parse(script_command)?;
            base.angle_offset.parse(script_command)?;
            base.projection_stretch.parse(script_command)?;
            base.rot_option.parse(script_command)?;
            base.rot_default_grouping.parse(script_command)?;
            base.rot_nondefault_group =
                param_utilities::set_param_if_present_fortran_input_string_array(
                    script_command,
                    ROT_NONDEFAULT_GROUP_KEY,
                    NONDEFAULT_GROUP_SIZE,
                    &NONDEFAULT_GROUP_INTEGER_TYPE,
                )?;
            base.rotation_fixed_view.parse(script_command)?;
            base.local_rot_option.parse(script_command)?;
            base.local_rot_default_grouping.parse(script_command)?;
            base.local_rot_nondefault_group =
                param_utilities::set_param_if_present_fortran_input_string_array(
                    script_command,
                    LOCAL_ROT_NONDEFAULT_GROUP_KEY,
                    NONDEFAULT_GROUP_SIZE,
                    &NONDEFAULT_GROUP_INTEGER_TYPE,
                )?;
            base.tilt_option.parse(script_command)?;
            base.tilt_default_grouping.parse(script_command)?;
            base.tilt_nondefault_group =
                param_utilities::set_param_if_present_fortran_input_string_array(
                    script_command,
                    TILT_NONDEFAULT_GROUP_KEY,
                    NONDEFAULT_GROUP_SIZE,
                    &NONDEFAULT_GROUP_INTEGER_TYPE,
                )?;
            base.local_tilt_option.parse(script_command)?;
            base.local_tilt_default_grouping.parse(script_command)?;
            base.local_tilt_nondefault_group =
                param_utilities::set_param_if_present_fortran_input_string_array(
                    script_command,
                    LOCAL_TILT_NONDEFAULT_GROUP_KEY,
                    NONDEFAULT_GROUP_SIZE,
                    &NONDEFAULT_GROUP_INTEGER_TYPE,
                )?;
            base.mag_reference_view.parse(script_command)?;
            base.mag_option.parse(script_command)?;
            base.mag_default_grouping.parse(script_command)?;
            base.mag_nondefault_group =
                param_utilities::set_param_if_present_fortran_input_string_array(
                    script_command,
                    MAG_NONDEFAULT_GROUP_KEY,
                    NONDEFAULT_GROUP_SIZE,
                    &NONDEFAULT_GROUP_INTEGER_TYPE,
                )?;
            base.local_mag_reference_view.parse(script_command)?;
            base.local_mag_option.parse(script_command)?;
            base.local_mag_default_grouping.parse(script_command)?;
            base.local_mag_nondefault_group =
                param_utilities::set_param_if_present_fortran_input_string_array(
                    script_command,
                    LOCAL_MAG_NONDEFAULT_GROUP_KEY,
                    NONDEFAULT_GROUP_SIZE,
                    &NONDEFAULT_GROUP_INTEGER_TYPE,
                )?;
            base.x_stretch_option.parse(script_command)?;
            base.x_stretch_default_grouping.parse(script_command)?;
            base.x_stretch_nondefault_group =
                param_utilities::set_param_if_present_fortran_input_string_array(
                    script_command,
                    X_STRETCH_NONDEFAULT_GROUP_KEY,
                    NONDEFAULT_GROUP_SIZE,
                    &NONDEFAULT_GROUP_INTEGER_TYPE,
                )?;
            base.local_x_stretch_option.parse(script_command)?;
            base.local_x_stretch_default_grouping
                .parse(script_command)?;
            base.local_x_stretch_nondefault_group =
                param_utilities::set_param_if_present_fortran_input_string_array(
                    script_command,
                    LOCAL_X_STRETCH_NONDEFAULT_GROUP_KEY,
                    NONDEFAULT_GROUP_SIZE,
                    &NONDEFAULT_GROUP_INTEGER_TYPE,
                )?;
            base.skew_option.parse(script_command)?;
            base.skew_default_grouping.parse(script_command)?;
            base.skew_nondefault_group =
                param_utilities::set_param_if_present_fortran_input_string_array(
                    script_command,
                    SKEW_NONDEFAULT_GROUP_KEY,
                    NONDEFAULT_GROUP_SIZE,
                    &NONDEFAULT_GROUP_INTEGER_TYPE,
                )?;
            base.local_skew_option.parse(script_command)?;
            base.local_skew_default_grouping.parse(script_command)?;
            base.local_skew_nondefault_group =
                param_utilities::set_param_if_present_fortran_input_string_array(
                    script_command,
                    LOCAL_SKEW_NONDEFAULT_GROUP_KEY,
                    NONDEFAULT_GROUP_SIZE,
                    &NONDEFAULT_GROUP_INTEGER_TYPE,
                )?;
            base.residual_report_criterion.parse(script_command)?;
            base.surfaces_to_analyze.parse(script_command)?;
            base.metro_factor.parse(script_command)?;
            base.maximum_cycles.parse(script_command)?;
            base.axis_z_shift.parse(script_command)?;
            base.local_alignments.parse(script_command)?;
            base.output_local_file = script_command.get_value(Some(OUTPUT_LOCAL_FILE_STRING))?;
            param_utilities::set_param_if_present_fortran_input_string(
                script_command,
                TARGET_PATCH_SIZE_X_AND_Y_KEY,
                &mut base.target_patch_size_xand_y,
            )?;
            param_utilities::set_param_if_present_fortran_input_string(
                script_command,
                NUMBER_OF_LOCAL_PATCHES_X_AND_Y_KEY,
                &mut base.number_of_local_patches_xand_y,
            )?;
            param_utilities::set_param_if_present_fortran_input_string(
                script_command,
                MIN_SIZE_OR_OVERLAP_X_AND_Y_KEY,
                &mut base.min_size_or_overlap_xand_y,
            )?;
            param_utilities::set_param_if_present_fortran_input_string(
                script_command,
                MIN_FIDS_TOTAL_AND_EACH_SURFACE_KEY,
                &mut base.min_fids_total_and_each_surface,
            )?;
            base.fix_xyz_coordinates.parse(script_command)?;
            param_utilities::set_param_if_present_fortran_input_string(
                script_command,
                LOCAL_OUTPUT_OPTIONS_STRING,
                &mut base.local_output_options,
            )?;
            base.beam_tilt_option.parse(script_command)?;
            base.fixed_or_initial_beam_tilt
                .get_mut()
                .unwrap()
                .parse(script_command)?;
            base.robust_fitting.parse(script_command)?;
            base.weight_whole_tracks.parse(script_command)?;
            base.k_factor_scaling.parse(script_command)?;
            base.created_day_stamp.parse(script_command)?;
            // crossValidate: converted from a boolean to an integer, handle backwards
            // compatibility.
            base.cross_validate_deprecated.parse(script_command)?;
            base.cross_validate.parse(script_command)?;
            if base.cross_validate.is_null() && base.cross_validate_deprecated.is() {
                // CrossValidate in the comfile is in boolean form - present and has no value.
                base.cross_validate.set_int(1);
            }
            let param = script_command.get_value(Some(OUTPUT_X_AXIS_TILT_FILE_KEY))?;
            if is_blank(&param) {
                base.output_x_axis_tilt_file = Some(dataset_files::get_x_tilt_file_name(
                    base.manager,
                    Some(base.axis_id),
                ));
            } else {
                base.output_x_axis_tilt_file = param;
            }
            base.x_tilt_option.parse(script_command)?;
            base.x_tilt_default_grouping.parse(script_command)?;
        }
        let invalid_reason = self.base.validate();
        if !java_lang_string_matches_whitespace(&invalid_reason) {
            return Err(InvalidParameterException::new(Some(&invalid_reason)).into());
        }
        // set fields dependent on other fields
        self.set_output_local_file();
        self.base.loaded_from_file = true;
        Ok(())
    }

    /// Java `updateComScriptCommand`.  Update the script command with the current
    /// values.
    fn update_com_script_command(
        &self,
        script_command: &mut ComScriptCommand,
    ) -> Result<(), BadComScriptException> {
        let base = &self.base;
        let invalid_reason = base.validate();
        if !java_lang_string_matches_whitespace(&invalid_reason) {
            return Err(BadComScriptException::new(&invalid_reason));
        }
        // Switch to keyword/value pairs
        script_command.use_keyword_value();

        base.images_are_binned.update_com_script(script_command);
        param_utilities::update_script_parameter_string(
            script_command,
            Some(MODEL_FILE_STRING),
            base.model_file.as_deref(),
        )?;
        param_utilities::update_script_parameter_string(
            script_command,
            Some(IMAGE_FILE_STRING),
            base.image_file.as_deref(),
        )?;
        script_command.delete_key(Some(OUTPUT_MODEL_AND_RESIDUAL_STRING));
        param_utilities::update_script_parameter_string(
            script_command,
            Some(OUTPUT_MODEL_FILE_STRING),
            base.output_model_file.as_deref(),
        )?;
        param_utilities::update_script_parameter_string(
            script_command,
            Some(OUTPUT_RESIDUAL_FILE_STRING),
            base.output_residual_file.as_deref(),
        )?;
        param_utilities::update_script_parameter_string(
            script_command,
            Some(OUTPUT_FID_XYZ_FILE_STRING),
            base.output_fid_xyz_file.as_deref(),
        )?;
        param_utilities::update_script_parameter_string(
            script_command,
            Some(OUTPUT_TILT_FILE_STRING),
            base.output_tilt_file.as_deref(),
        )?;
        param_utilities::update_script_parameter_string(
            script_command,
            Some(OUTPUT_TRANSFORM_FILE_STRING),
            base.output_transform_file.as_deref(),
        )?;
        param_utilities::update_script_parameter_fortran_input_string(
            script_command,
            Some(INCLUDE_START_END_INC_STRING),
            &base.include_start_end_inc,
        );
        param_utilities::update_script_parameter_string_list(
            script_command,
            Some(INCLUDE_LIST_STRING),
            Some(&base.include_list),
        )?;
        param_utilities::update_script_parameter_string_list(
            script_command,
            Some(EXCLUDE_LIST_KEY),
            Some(&base.exclude_list),
        )?;
        base.rotation_angle.update_com_script(script_command);
        base.separate_group.update_com_script(script_command)?;
        base.tilt_angle_spec.update_com_script(script_command)?;
        base.angle_offset.update_com_script(script_command);
        base.projection_stretch.update_com_script(script_command);
        base.rot_option.update_com_script(script_command);
        base.rot_default_grouping.update_com_script(script_command);
        base.rotation_fixed_view.update_com_script(script_command);
        base.tilt_option.update_com_script(script_command);
        base.tilt_default_grouping.update_com_script(script_command);
        base.mag_reference_view.update_com_script(script_command);
        base.mag_option.update_com_script(script_command);
        base.mag_default_grouping.update_com_script(script_command);
        base.x_stretch_option.update_com_script(script_command);
        base.skew_option.update_com_script(script_command);
        base.x_stretch_default_grouping
            .update_com_script(script_command);
        base.skew_default_grouping.update_com_script(script_command);
        base.residual_report_criterion
            .update_com_script(script_command);
        base.surfaces_to_analyze.update_com_script(script_command);
        base.metro_factor.update_com_script(script_command);
        base.maximum_cycles.update_com_script(script_command);
        base.axis_z_shift.update_com_script(script_command);
        // local alignment
        base.local_alignments.update_com_script(script_command);
        param_utilities::update_script_parameter_string(
            script_command,
            Some(OUTPUT_LOCAL_FILE_STRING),
            base.output_local_file.as_deref(),
        )?;
        param_utilities::update_script_parameter_fortran_input_string(
            script_command,
            Some(TARGET_PATCH_SIZE_X_AND_Y_KEY),
            &base.target_patch_size_xand_y,
        );
        param_utilities::update_script_parameter_fortran_input_string(
            script_command,
            Some(NUMBER_OF_LOCAL_PATCHES_X_AND_Y_KEY),
            &base.number_of_local_patches_xand_y,
        );
        param_utilities::update_script_parameter_fortran_input_string(
            script_command,
            Some(MIN_SIZE_OR_OVERLAP_X_AND_Y_KEY),
            &base.min_size_or_overlap_xand_y,
        );
        param_utilities::update_script_parameter_fortran_input_string(
            script_command,
            Some(MIN_FIDS_TOTAL_AND_EACH_SURFACE_KEY),
            &base.min_fids_total_and_each_surface,
        );
        base.fix_xyz_coordinates.update_com_script(script_command);
        param_utilities::update_script_parameter_fortran_input_string(
            script_command,
            Some(LOCAL_OUTPUT_OPTIONS_STRING),
            &base.local_output_options,
        );
        base.local_rot_option.update_com_script(script_command);
        base.local_rot_default_grouping
            .update_com_script(script_command);
        base.local_tilt_option.update_com_script(script_command);
        base.local_tilt_default_grouping
            .update_com_script(script_command);
        base.local_mag_reference_view
            .update_com_script(script_command);
        base.local_mag_option.update_com_script(script_command);
        base.local_mag_default_grouping
            .update_com_script(script_command);
        base.local_x_stretch_option
            .update_com_script(script_command);
        base.local_x_stretch_default_grouping
            .update_com_script(script_command);
        base.local_skew_option.update_com_script(script_command);
        base.local_skew_default_grouping
            .update_com_script(script_command);
        // optional parameters
        param_utilities::update_script_parameter_string(
            script_command,
            Some(OUTPUT_Z_FACTOR_FILE_STRING),
            base.output_z_factor_file.as_deref(),
        )?;
        param_utilities::update_script_parameter_fortran_input_string_array(
            script_command,
            Some(ROT_NONDEFAULT_GROUP_KEY),
            base.rot_nondefault_group.as_deref(),
        );
        param_utilities::update_script_parameter_fortran_input_string_array(
            script_command,
            Some(TILT_NONDEFAULT_GROUP_KEY),
            base.tilt_nondefault_group.as_deref(),
        );
        param_utilities::update_script_parameter_fortran_input_string_array(
            script_command,
            Some(MAG_NONDEFAULT_GROUP_KEY),
            base.mag_nondefault_group.as_deref(),
        );
        param_utilities::update_script_parameter_fortran_input_string_array(
            script_command,
            Some(X_STRETCH_NONDEFAULT_GROUP_KEY),
            base.x_stretch_nondefault_group.as_deref(),
        );
        param_utilities::update_script_parameter_fortran_input_string_array(
            script_command,
            Some(SKEW_NONDEFAULT_GROUP_KEY),
            base.skew_nondefault_group.as_deref(),
        );
        // local optional parameters
        param_utilities::update_script_parameter_fortran_input_string_array(
            script_command,
            Some(LOCAL_ROT_NONDEFAULT_GROUP_KEY),
            base.local_rot_nondefault_group.as_deref(),
        );
        param_utilities::update_script_parameter_fortran_input_string_array(
            script_command,
            Some(LOCAL_TILT_NONDEFAULT_GROUP_KEY),
            base.local_tilt_nondefault_group.as_deref(),
        );
        param_utilities::update_script_parameter_fortran_input_string_array(
            script_command,
            Some(LOCAL_MAG_NONDEFAULT_GROUP_KEY),
            base.local_mag_nondefault_group.as_deref(),
        );
        param_utilities::update_script_parameter_fortran_input_string_array(
            script_command,
            Some(LOCAL_X_STRETCH_NONDEFAULT_GROUP_KEY),
            base.local_x_stretch_nondefault_group.as_deref(),
        );
        param_utilities::update_script_parameter_fortran_input_string_array(
            script_command,
            Some(LOCAL_SKEW_NONDEFAULT_GROUP_KEY),
            base.local_skew_nondefault_group.as_deref(),
        );
        base.beam_tilt_option
            .update_com_script_when_defaulted(script_command, true);
        base.robust_fitting.update_com_script(script_command);
        base.weight_whole_tracks.update_com_script(script_command);
        base.k_factor_scaling.update_com_script(script_command);
        base.cross_validate.update_com_script(script_command);
        {
            let mut fixed_or_initial_beam_tilt = base.fixed_or_initial_beam_tilt.lock().unwrap();
            // Only using FixedOrInitialBeamTilt for fixed beam tilt
            if !base.beam_tilt_option.equals_int(FIXED_OPTION)
                && !fixed_or_initial_beam_tilt.is_default()
            {
                fixed_or_initial_beam_tilt.reset();
            }
            fixed_or_initial_beam_tilt.update_com_script(script_command);
        }
        param_utilities::update_script_parameter_string(
            script_command,
            Some(OUTPUT_X_AXIS_TILT_FILE_KEY),
            base.output_x_axis_tilt_file.as_deref(),
        )?;
        base.x_tilt_option.update_com_script(script_command);
        base.x_tilt_default_grouping
            .update_com_script(script_command);
        Ok(())
    }

    /// Java `initializeDefaults`.
    fn initialize_defaults(&mut self) {
        self.base.reset();
    }
}

/// The inherited `Command` methods, forwarded to the `ConstTiltalignParam` state.
impl Command for TiltalignParam {
    fn get_axis_id(&self) -> AxisID {
        self.base.get_axis_id()
    }
    fn get_command_mode(&self) -> Option<&dyn CommandMode> {
        self.base.get_command_mode()
    }
    fn get_process_name(&self) -> Option<ProcessName> {
        self.base.get_process_name()
    }
    fn get_command(&self) -> Option<String> {
        self.base.get_command()
    }
    fn get_command_name(&self) -> Option<String> {
        self.base.get_command_name()
    }
    fn get_command_line(&self) -> Option<String> {
        self.base.get_command_line()
    }
    fn get_command_array(&self) -> Option<Vec<String>> {
        self.base.get_command_array()
    }
    fn get_command_input_file(&self) -> Option<PathBuf> {
        self.base.get_command_input_file()
    }
    fn get_command_output_file(&self) -> Option<PathBuf> {
        self.base.get_command_output_file()
    }
    fn get_output_image_file_type(&self) -> Option<Arc<FileType>> {
        self.base.get_output_image_file_type()
    }
    fn get_output_image_file_key(&self) -> Option<FileKey> {
        self.base.get_output_image_file_key()
    }
    fn get_output_image_file_type2(&self) -> Option<Arc<FileType>> {
        self.base.get_output_image_file_type2()
    }
    fn get_output_image_file_key2(&self) -> Option<FileKey> {
        self.base.get_output_image_file_key2()
    }
    fn is_message_reporter(&self) -> bool {
        self.base.is_message_reporter()
    }
    fn get_subcommand_process_name(&self) -> Option<String> {
        self.base.get_subcommand_process_name()
    }
    fn get_subcommand_details(&self) -> Option<&dyn CommandDetails> {
        self.base.get_subcommand_details()
    }
    fn get_process_details(&self) -> Option<&dyn ProcessDetails> {
        Some(self)
    }
}

/// The inherited `ProcessDetails` methods, forwarded to the `ConstTiltalignParam`
/// state.
impl ProcessDetails for TiltalignParam {
    fn get_int_value(&self, field: &dyn FieldInterface) -> Option<i32> {
        self.base.get_int_value(field)
    }
    fn get_boolean_value(&self, field: &dyn FieldInterface) -> Option<bool> {
        self.base.get_boolean_value(field)
    }
    fn get_double_value(&self, field: &dyn FieldInterface) -> Option<f64> {
        self.base.get_double_value(field)
    }
    fn get_hashtable(
        &self,
        field: &dyn FieldInterface,
    ) -> Option<super::process_details::Hashtable> {
        self.base.get_hashtable(field)
    }
    fn get_etomo_number(&self, field: &dyn FieldInterface) -> Option<ConstEtomoNumber> {
        self.base.get_etomo_number(field)
    }
    fn get_int_key_list(
        &self,
        field: &dyn FieldInterface,
    ) -> Option<crate::imod::etomo::r#type::int_key_list::IntKeyList> {
        self.base.get_int_key_list(field)
    }
    fn get_string(&self, field: &dyn FieldInterface) -> Option<String> {
        self.base.get_string(field)
    }
    fn get_string_array(&self, field: &dyn FieldInterface) -> Option<Vec<String>> {
        self.base.get_string_array(field)
    }
    fn get_iterator_element_list(
        &self,
        field: &dyn FieldInterface,
    ) -> Option<crate::imod::etomo::r#type::iterator_element_list::IteratorElementList> {
        self.base.get_iterator_element_list(field)
    }
}

/// The inherited `Loggable` methods, forwarded to the `ConstTiltalignParam` state.
impl Loggable for TiltalignParam {
    fn get_name(&self) -> String {
        self.base.get_name()
    }
    fn get_log_message(&self) -> Result<Vec<Option<String>>, LoggableException> {
        self.base.get_log_message()
    }
}
