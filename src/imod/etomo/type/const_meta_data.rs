//! `IMOD/Etomo/src/etomo/type/ConstMetaData.java`.
//!
//! The read-only view of `MetaData`.  The concrete `MetaData`
//! (`etomo/type/meta_data.rs`) implements it by forwarding to its inherent methods, so
//! the signatures here are exactly those; see that module's header for how Java's
//! parameter and return types are represented (a Java `String` parameter is
//! `Option<&str>`, a returned field object is a clone of the field's concrete type).
#![allow(dead_code)]

use super::axis_id::AxisID;
use super::axis_type::AxisType;
use super::data_source::DataSource;
use super::dialog_type::DialogType;
use super::etomo_boolean2::EtomoBoolean2;
use super::etomo_number::EtomoNumber;
use super::extension::Extension;
use super::image_filename_style::ImageFilenameStyle;
use super::int_key_list::IntKeyList;
use super::panel_id::PanelId;
use super::sample_type::SampleType;
use super::tilt_angle_spec::TiltAngleSpec;
use super::view_type::ViewType;
use crate::imod::etomo::comscript::fortran_input_string::FortranInputString;

/// Java `ConstMetaData`.
pub trait ConstMetaData: Send + Sync {
    /// Java `isCtf3dSetupSlabThicknessInNmSet`.
    fn is_ctf_3d_setup_slab_thickness_in_nm_set(&self) -> bool;

    /// Java `getRawImageStackExtension`.
    fn get_raw_image_stack_extension(&self) -> Option<&'static Extension>;

    /// Java `getPostCurTab`.
    fn get_post_cur_tab(&self) -> EtomoNumber;

    /// Java `getGenCurTab`.
    fn get_gen_cur_tab(&self) -> EtomoNumber;

    /// Java `getDatasetName`.
    fn get_dataset_name(&self) -> String;

    /// Java `getComScriptCreated`.
    fn get_com_script_created(&self) -> bool;

    /// Java `getAdjustedFocusA`.
    fn get_adjusted_focus_a(&self) -> EtomoBoolean2;

    /// Java `getAdjustedFocusB`.
    fn get_adjusted_focus_b(&self) -> EtomoBoolean2;

    /// Java `getImageRotation`.
    fn get_image_rotation(&self, axis_id: AxisID) -> EtomoNumber;

    /// Java `getBackupDirectory`.
    fn get_backup_directory(&self) -> String;

    /// Java `getBinning`.
    fn get_binning(&self) -> String;

    /// Java `getViewType`.
    fn get_view_type(&self) -> ViewType;

    /// Java `getPixelSize`.
    fn get_pixel_size(&self) -> f64;

    /// Java `isHalfFloatModeOutputSet`.
    fn is_half_float_mode_output_set(&self) -> bool;

    /// Java `getHalfFloatModeOutput`.
    fn get_half_float_mode_output(&self) -> Option<i32>;

    /// Java `getDataSource`.
    fn get_data_source(&self) -> DataSource;

    /// Java `getFiducialDiameter`.
    fn get_fiducial_diameter(&self) -> f64;

    /// Java `getTiltAngleSpecA`.
    fn get_tilt_angle_spec_a(&self) -> TiltAngleSpec;

    /// Java `getExcludeProjectionsA`.
    fn get_exclude_projections_a(&self) -> String;

    /// Java `getTiltAngleSpecB`.
    fn get_tilt_angle_spec_b(&self) -> TiltAngleSpec;

    /// Java `getDistortionFile`.
    fn get_distortion_file(&self) -> String;

    /// Java `getMagGradientFile`.
    fn get_mag_gradient_file(&self) -> String;

    /// Java `getExcludeProjectionsB`.
    fn get_exclude_projections_b(&self) -> String;

    /// Java `getCombineVolcombineParallel`.
    fn get_combine_volcombine_parallel(&self) -> Option<EtomoBoolean2>;

    /// Java `isDefaultParallel`.
    fn is_default_parallel(&self) -> bool;

    /// Java `isDefaultGpuProcessing`.
    fn is_default_gpu_processing(&self) -> bool;

    /// Java `getFirstAxisPrepend`.
    fn get_first_axis_prepend(&self) -> Option<String>;

    /// Java `getSecondAxisPrepend`.
    fn get_second_axis_prepend(&self) -> Option<String>;

    /// Java `getTargetPatchSizeXandY`.
    fn get_target_patch_size_x_and_y(&self) -> String;

    /// Java `getFixedBeamTiltSelected`.
    fn get_fixed_beam_tilt_selected(&self, axis_id: AxisID) -> EtomoBoolean2;

    /// Java `getNumberOfLocalPatchesXandY`.
    fn get_number_of_local_patches_x_and_y(&self) -> String;

    /// Java `getFixedBeamTilt`.
    fn get_fixed_beam_tilt(&self, axis_id: AxisID) -> EtomoNumber;

    /// Java `getNoBeamTiltSelected`.
    fn get_no_beam_tilt_selected(&self, axis_id: AxisID) -> EtomoBoolean2;

    /// Java `getSampleThickness`.
    fn get_sample_thickness(&self, axis_id: AxisID) -> EtomoNumber;

    /// Java `getSizeToOutputInXandY`.
    fn get_size_to_output_in_x_and_y(&self, axis_id: AxisID) -> FortranInputString;

    /// Java `getPosBinning`.
    fn get_pos_binning(&self, axis_id: AxisID) -> i32;

    /// Java `getStackBinning`.
    fn get_stack_binning(&self, axis_id: AxisID) -> i32;

    /// Java `isStack3dFindBinningSet`.
    fn is_stack_3d_find_binning_set(&self, axis_id: AxisID) -> bool;

    /// Java `getStack3dFindBinning`.
    fn get_stack_3d_find_binning(&self, axis_id: AxisID) -> i32;

    /// Java `getTiltParallel`.
    fn get_tilt_parallel(&self, axis_id: AxisID, panel_id: PanelId) -> Option<EtomoBoolean2>;

    /// Java `getFinalStackCtfCorrectionParallel`.
    fn get_final_stack_ctf_correction_parallel(&self, axis_id: AxisID) -> Option<EtomoBoolean2>;

    /// Java `isDistortionCorrection`.
    fn is_distortion_correction(&self) -> bool;

    /// Java `isFinalStackBetterRadiusEmpty`.
    fn is_final_stack_better_radius_empty(&self, axis_id: AxisID) -> bool;

    /// Java `getFinalStackBetterRadius`.
    fn get_final_stack_better_radius(&self, axis_id: AxisID) -> String;

    /// Java `isFinalStackFiducialDiameterNull`.
    fn is_final_stack_fiducial_diameter_null(&self, axis_id: AxisID) -> bool;

    /// Java `getFinalStackFiducialDiameter`.
    fn get_final_stack_fiducial_diameter(&self, axis_id: AxisID) -> String;

    /// Java `getFinalStackExpandCircleIterations`.
    fn get_final_stack_expand_circle_iterations(&self, axis_id: AxisID) -> i32;

    /// Java `isFinalStackExpandCircleIterationsSet`.
    fn is_final_stack_expand_circle_iterations_set(&self, axis_id: AxisID) -> bool;

    /// Java `getFinalStackPolynomialOrder`.
    fn get_final_stack_polynomial_order(&self, axis_id: AxisID) -> i32;

    /// Java `isFinalAlignedStackDialogSaved`.
    fn is_final_aligned_stack_dialog_saved(&self, axis_id: AxisID) -> bool;

    /// Java `getTomoGenTrialTomogramNameList`.
    fn get_tomo_gen_trial_tomogram_name_list(
        &self,
        axis_id: AxisID,
    ) -> std::sync::Arc<std::sync::Mutex<IntKeyList>>;

    /// Java `getTrackRaptorUseRawStack`.
    fn get_track_raptor_use_raw_stack(&self) -> bool;

    /// Java `getTrackRaptorMark`.
    fn get_track_raptor_mark(&self) -> String;

    /// Java `getTrackRaptorDiam`.
    fn get_track_raptor_diam(&self) -> EtomoNumber;

    /// Java `getEraseGoldModelUseFid`.
    fn get_erase_gold_model_use_fid(&self, axis_id: AxisID) -> bool;

    /// Java `isPostFlattenWarpInputTrimVol`.
    fn is_post_flatten_warp_input_trim_vol(&self) -> bool;

    /// Java `isPostFlattenWarpContoursOnOneSurface`.
    fn is_post_flatten_warp_contours_on_one_surface(&self) -> bool;

    /// Java `getPostFlattenWarpSpacingInX`.
    fn get_post_flatten_warp_spacing_in_x(&self) -> String;

    /// Java `getPostFlattenWarpSpacingInY`.
    fn get_post_flatten_warp_spacing_in_y(&self) -> String;

    /// Java `isPostSqueezeVolInputTrimVol`.
    fn is_post_squeeze_vol_input_trim_vol(&self) -> bool;

    /// Java `isPostTrimvolConvertToBytes`.
    fn is_post_trimvol_convert_to_bytes(&self) -> bool;

    /// Java `isPostTrimvolFixedScaling`.
    fn is_post_trimvol_fixed_scaling(&self) -> bool;

    /// Java `isPostTrimvolRotateX`.
    fn is_post_trimvol_rotate_x(&self) -> bool;

    /// Java `isFiducialessAlignment`.
    fn is_fiducialess_alignment(&self, axis_id: AxisID) -> bool;

    /// Java `getLambdaForSmoothing`.
    fn get_lambda_for_smoothing(&self) -> String;

    /// Java `getLambdaForSmoothingList`.
    fn get_lambda_for_smoothing_list(&self) -> String;

    /// Java `isLambdaForSmoothingListEmpty`.
    fn is_lambda_for_smoothing_list_empty(&self) -> bool;

    /// Java `getTrackOverlapOfPatchesXandY`.
    fn get_track_overlap_of_patches_x_and_y(&self, axis_id: AxisID) -> String;

    /// Java `getTrackNumberOfPatchesXandY`.
    fn get_track_number_of_patches_x_and_y(&self, axis_id: AxisID) -> String;

    /// Java `getTrackLengthAndOverlap`.
    fn get_track_length_and_overlap(&self, axis_id: AxisID) -> String;

    /// Java `isTrackOverlapOfPatchesXandYSet`.
    fn is_track_overlap_of_patches_x_and_y_set(&self, axis_id: AxisID) -> bool;

    /// Java `isTrackLengthAndOverlapSet`.
    fn is_track_length_and_overlap_set(&self, axis_id: AxisID) -> bool;

    /// Java `getTrackMethod`.
    fn get_track_method(&self, axis_id: AxisID) -> String;

    /// Java `getGenLog`.
    fn get_gen_log(&self, axis_id: AxisID) -> String;

    /// Java `getGenScaleFactorLog`.
    fn get_gen_scale_factor_log(&self, axis_id: AxisID) -> String;

    /// Java `getGenScaleOffsetLog`.
    fn get_gen_scale_offset_log(&self, axis_id: AxisID) -> String;

    /// Java `getGenScaleFactorLinear`.
    fn get_gen_scale_factor_linear(&self, axis_id: AxisID) -> String;

    /// Java `getGenScaleOffsetLinear`.
    fn get_gen_scale_offset_linear(&self, axis_id: AxisID) -> String;

    /// Java `isGenScaleFactorLinearSet`.
    fn is_gen_scale_factor_linear_set(&self, axis_id: AxisID) -> bool;

    /// Java `isGenScaleOffsetLinearSet`.
    fn is_gen_scale_offset_linear_set(&self, axis_id: AxisID) -> bool;

    /// Java `getGenSuperSampleFactor`.
    fn get_gen_super_sample_factor(&self, axis_id: AxisID) -> EtomoNumber;

    /// Java `getGenExpandInputLines`.
    fn get_gen_expand_input_lines(&self, axis_id: AxisID) -> EtomoBoolean2;

    /// Java `isGenBackProjection`.
    fn is_gen_back_projection(&self, axis_id: AxisID) -> bool;

    /// Java `isGenFilterTrials`.
    fn is_gen_filter_trials(&self, axis_id: AxisID) -> bool;

    /// Java `getGenSubareaSize`.
    fn get_gen_subarea_size(&self, axis_id: AxisID) -> String;

    /// Java `getGenYOffsetOfSubarea`.
    fn get_gen_y_offset_of_subarea(&self, axis_id: AxisID) -> String;

    /// Java `isGenSubarea`.
    fn is_gen_subarea(&self, axis_id: AxisID) -> bool;

    /// Java `getRadialRadius`.
    fn get_radial_radius(&self, panel_id: PanelId, axis_id: AxisID) -> Option<String>;

    /// Java `getRadialSigma`.
    fn get_radial_sigma(&self, panel_id: PanelId, axis_id: AxisID) -> Option<String>;

    /// Java `isUseFinalStackExpandCircleIterations`.
    fn is_use_final_stack_expand_circle_iterations(&self, axis_id: AxisID) -> bool;

    /// Java `isPostTrimvolSwapYZ`.
    fn is_post_trimvol_swap_yz(&self) -> bool;

    /// Java `getPostTrimvolFixedScaleMax`.
    fn get_post_trimvol_fixed_scale_max(&self) -> String;

    /// Java `getPostTrimvolFixedScaleMin`.
    fn get_post_trimvol_fixed_scale_min(&self) -> String;

    /// Java `getPostTrimvolScaleXMax`.
    fn get_post_trimvol_scale_x_max(&self) -> String;

    /// Java `getPostTrimvolScaleXMin`.
    fn get_post_trimvol_scale_x_min(&self) -> String;

    /// Java `getPostTrimvolScaleYMin`.
    fn get_post_trimvol_scale_y_min(&self) -> String;

    /// Java `getPostTrimvolScaleYMax`.
    fn get_post_trimvol_scale_y_max(&self) -> String;

    /// Java `getPostTrimvolSectionScaleMax`.
    fn get_post_trimvol_section_scale_max(&self) -> String;

    /// Java `getPostTrimvolSectionScaleMin`.
    fn get_post_trimvol_section_scale_min(&self) -> String;

    /// Java `getPostTrimvolXMax`.
    fn get_post_trimvol_x_max(&self) -> String;

    /// Java `getPostTrimvolXMin`.
    fn get_post_trimvol_x_min(&self) -> String;

    /// Java `getPostTrimvolYMin`.
    fn get_post_trimvol_y_min(&self) -> String;

    /// Java `getPostTrimvolYMax`.
    fn get_post_trimvol_y_max(&self) -> String;

    /// Java `getPostTrimvolZMin`.
    fn get_post_trimvol_z_min(&self) -> String;

    /// Java `getPostTrimvolZMax`.
    fn get_post_trimvol_z_max(&self) -> String;

    /// Java `isPostReduceFiltVolReductionFactor`.
    fn is_post_reduce_filt_vol_reduction_factor(&self) -> bool;

    /// Java `getPostReduceFiltVolReductionFactor`.
    fn get_post_reduce_filt_vol_reduction_factor(&self) -> String;

    /// Java `getPostReduceFiltVolReductionFactorEtomoNumber`.
    fn get_post_reduce_filt_vol_reduction_factor_etomo_number(&self) -> EtomoNumber;

    /// Java `isPostReduceFiltVolZReductionFactor`.
    fn is_post_reduce_filt_vol_z_reduction_factor(&self) -> bool;

    /// Java `getPostReduceFiltVolZReductionFactor`.
    fn get_post_reduce_filt_vol_z_reduction_factor(&self) -> String;

    /// Java `getPostReduceFiltVolZReductionFactorEtomoNumber`.
    fn get_post_reduce_filt_vol_z_reduction_factor_etomo_number(&self) -> EtomoNumber;

    /// Java `getPostReduceFiltVolLowPassRadiusSigma`.
    fn get_post_reduce_filt_vol_low_pass_radius_sigma(&self) -> String;

    /// Java `getPostReduceFiltVolDeconvolutionStrength`.
    fn get_post_reduce_filt_vol_deconvolution_strength(&self) -> String;

    /// Java `getPostReduceFiltVolSNRFalloff`.
    fn get_post_reduce_filt_vol_snr_falloff(&self) -> String;

    /// Java `getPostReduceFiltVolHighPassNyquist`.
    fn get_post_reduce_filt_vol_high_pass_nyquist(&self) -> String;

    /// Java `getPostReduceFiltVolDefocusInMicrons`.
    fn get_post_reduce_filt_vol_defocus_in_microns(&self) -> String;

    /// Java `getPostReduceFiltVolPhaseShift`.
    fn get_post_reduce_filt_vol_phase_shift(&self) -> String;

    /// Java `isEraseBeadsInitialized`.
    fn is_erase_beads_initialized(&self) -> bool;

    /// Java `isTrackSeedModelManual`.
    fn is_track_seed_model_manual(&self, axis_id: AxisID) -> bool;

    /// Java `isTrackSeedModelAuto`.
    fn is_track_seed_model_auto(&self, axis_id: AxisID) -> bool;

    /// Java `isTrackSeedModelTransfer`.
    fn is_track_seed_model_transfer(&self, axis_id: AxisID) -> bool;

    /// Java `isTrackExcludeInsideAreas`.
    fn is_track_exclude_inside_areas(&self, axis_id: AxisID) -> bool;

    /// Java `getTrackJustFindShiftsNearZero`.
    fn get_track_just_find_shifts_near_zero(&self, axis_id: AxisID) -> String;

    /// Java `getTrackTargetNumberOfBeads`.
    fn get_track_target_number_of_beads(&self, axis_id: AxisID) -> String;

    /// Java `getTrackTargetDensityOfBeads`.
    fn get_track_target_density_of_beads(&self, axis_id: AxisID) -> String;

    /// Java `isTrackClusteredPointsAllowedElongated`.
    fn is_track_clustered_points_allowed_elongated(&self, axis_id: AxisID) -> bool;

    /// Java `getTrackClusteredPointsAllowedElongatedValue`.
    fn get_track_clustered_points_allowed_elongated_value(&self, axis_id: AxisID) -> i32;

    /// Java `isTrackAdvanced`.
    fn is_track_advanced(&self, axis_id: AxisID) -> bool;

    /// Java `isStack3dFindThicknessSet`.
    fn is_stack_3d_find_thickness_set(&self, axis_id: AxisID) -> bool;

    /// Java `getStack3dFindThickness`.
    fn get_stack_3d_find_thickness(&self, axis_id: AxisID) -> String;

    /// Java `isSetFEIPixelSize`.
    fn is_set_fei_pixel_size(&self) -> bool;

    /// Java `isTwodir`.
    fn is_twodir(&self, axis_id: AxisID) -> bool;

    /// Java `getTwodir`.
    fn get_twodir(&self, axis_id: AxisID) -> String;

    /// Java `getRaptorTab`.
    fn get_raptor_tab(&self, axis_id: AxisID) -> i32;

    /// Java `getSeedAndTrackTab`.
    fn get_seed_and_track_tab(&self, axis_id: AxisID) -> i32;

    /// Java `getAntialiasFilter`.
    fn get_antialias_filter(&self, dialog_type: DialogType, axis_id: AxisID)
    -> Option<EtomoNumber>;

    /// Java `isAntialiasFilterNull`.
    fn is_antialias_filter_null(&self, dialog_type: DialogType, axis_id: AxisID) -> bool;

    /// Java `isTrackElongatedPointsAllowedNull`.
    fn is_track_elongated_points_allowed_null(&self, axis_id: AxisID) -> bool;

    /// Java `getTrackElongatedPointsAllowed`.
    fn get_track_elongated_points_allowed(&self, axis_id: AxisID) -> EtomoNumber;

    /// Java `getTrackLowerTargetForClustered`.
    fn get_track_lower_target_for_clustered(&self, axis_id: AxisID) -> String;

    /// Java `getWeightWholeTracks`.
    fn get_weight_whole_tracks(&self, axis_id: AxisID) -> bool;

    /// Java `getLengthOfPieces`.
    fn get_length_of_pieces(&self, axis_id: AxisID) -> String;

    /// Java `getMinimumOverlap`.
    fn get_minimum_overlap(&self, axis_id: AxisID) -> String;

    /// Java `getTargetMeasurementRatio`.
    fn get_target_measurement_ratio(&self, axis_id: AxisID) -> String;

    /// Java `getMinMeasurementRatio`.
    fn get_min_measurement_ratio(&self, axis_id: AxisID) -> String;

    /// Java `isTargetMeasurementRatioSet`.
    fn is_target_measurement_ratio_set(&self, axis_id: AxisID) -> bool;

    /// Java `isMinMeasurementRatioSet`.
    fn is_min_measurement_ratio_set(&self, axis_id: AxisID) -> bool;

    /// Java `getSampleType`.
    fn get_sample_type(&self, axis_id: AxisID) -> Option<SampleType>;

    /// Java `isHasGoldBeads`.
    fn is_has_gold_beads(&self, axis_id: AxisID) -> bool;

    /// Java `getPositioningFiducialDiameter`.
    fn get_positioning_fiducial_diameter(&self, axis_id: AxisID) -> f64;

    /// Java `getPositioningBeadSize`.
    fn get_positioning_bead_size(&self, axis_id: AxisID) -> String;

    /// Java `isHasGoldBeadsNull`.
    fn is_has_gold_beads_null(&self, axis_id: AxisID) -> bool;

    /// Java `isPositioningFiducialDiameterNull`.
    fn is_positioning_fiducial_diameter_null(&self, axis_id: AxisID) -> bool;

    /// Java `isPositioningBeadSizeNull`.
    fn is_positioning_bead_size_null(&self, axis_id: AxisID) -> bool;

    /// Java `getExtraThickness`.
    fn get_extra_thickness(&self, axis_id: AxisID) -> String;

    /// Java `getExtraThicknessCryo`.
    fn get_extra_thickness_cryo(&self, axis_id: AxisID) -> String;

    /// Java `getHammingLikeFilter`.
    fn get_hamming_like_filter(&self, panel_id: PanelId, axis_id: AxisID) -> Option<String>;

    /// Java `getFakeSIRTiterations`.
    fn get_fake_sirt_iterations(&self, panel_id: PanelId, axis_id: AxisID) -> Option<String>;

    /// Java `getExactFilterSize`.
    fn get_exact_filter_size(&self, panel_id: PanelId, axis_id: AxisID) -> Option<String>;

    /// Java `isOrigScopeTemplate`.
    fn is_orig_scope_template(&self) -> bool;

    /// Java `getOrigScopeTemplate`.
    fn get_orig_scope_template(&self) -> String;

    /// Java `isOrigSystemTemplate`.
    fn is_orig_system_template(&self) -> bool;

    /// Java `getOrigSystemTemplate`.
    fn get_orig_system_template(&self) -> String;

    /// Java `isOrigUserTemplate`.
    fn is_orig_user_template(&self) -> bool;

    /// Java `getOrigUserTemplate`.
    fn get_orig_user_template(&self) -> String;

    /// Java `isFiducialDiameterAvailable`.
    fn is_fiducial_diameter_available(&self) -> bool;

    /// Java `isPositioningNewDialog`.
    fn is_positioning_new_dialog(&self, axis_id: AxisID) -> bool;

    /// Java `getGenFilterTrialsFakeSIRTiterations`.
    fn get_gen_filter_trials_fake_sirt_iterations(&self, axis_id: AxisID) -> String;

    /// Java `getGenFilterTrialsExactObjectSizes`.
    fn get_gen_filter_trials_exact_object_sizes(&self, axis_id: AxisID) -> String;

    /// Java `getGenFilterTrialsGaussianCutoffs`.
    fn get_gen_filter_trials_gaussian_cutoffs(&self, axis_id: AxisID) -> String;

    /// Java `getGenFilterTrialsGaussianFalloffs`.
    fn get_gen_filter_trials_gaussian_falloffs(&self, axis_id: AxisID) -> String;

    /// Java `getGenFilterTrialsHammingLikeStarts`.
    fn get_gen_filter_trials_hamming_like_starts(&self, axis_id: AxisID) -> String;

    /// Java `getStackMtfFilterLowPassRadiusSigma`.
    fn get_stack_mtf_filter_low_pass_radius_sigma(&self, axis_id: AxisID) -> String;

    /// Java `getStackMtfFilterMtfFile`.
    fn get_stack_mtf_filter_mtf_file(&self, axis_id: AxisID) -> String;

    /// Java `getStackMtfFilterMaximumInverse`.
    fn get_stack_mtf_filter_maximum_inverse(&self, axis_id: AxisID) -> String;

    /// Java `getStackMtfFilterInverseRolloffRadiusSigma`.
    fn get_stack_mtf_filter_inverse_rolloff_radius_sigma(&self, axis_id: AxisID) -> String;

    /// Java `isUseStackMtfFilterFixedImageDose`.
    fn is_use_stack_mtf_filter_fixed_image_dose(&self, axis_id: AxisID) -> bool;

    /// Java `getStackMtfFilterFixedImageDose`.
    fn get_stack_mtf_filter_fixed_image_dose(&self, axis_id: AxisID) -> String;

    /// Java `getStackMtfFilterDoseWeightingFile`.
    fn get_stack_mtf_filter_dose_weighting_file(&self, axis_id: AxisID) -> String;

    /// Java `getStackMtfFilterTypeOfDoseFile`.
    fn get_stack_mtf_filter_type_of_dose_file(&self, axis_id: AxisID) -> String;

    /// Java `isStackMtfFilterVoltage200`.
    fn is_stack_mtf_filter_voltage_200(&self, axis_id: AxisID) -> bool;

    /// Java `getStackMtfFilterOptimalDoseScaling`.
    fn get_stack_mtf_filter_optimal_dose_scaling(&self, axis_id: AxisID) -> String;

    /// Java `getStackMtfFilterBidirectionalNumViews`.
    fn get_stack_mtf_filter_bidirectional_num_views(&self, axis_id: AxisID) -> String;

    /// Java `isUseStackCtfPhaseFlipXAxisTilt`.
    fn is_use_stack_ctf_phase_flip_x_axis_tilt(&self, axis_id: AxisID) -> bool;

    /// Java `getStackCtfPhaseFlipXAxisTilt`.
    fn get_stack_ctf_phase_flip_x_axis_tilt(&self, axis_id: AxisID) -> String;

    /// Java `getStackCtfPhaseFlipScaleByCtfPower`.
    fn get_stack_ctf_phase_flip_scale_by_ctf_power(&self, axis_id: AxisID) -> String;

    /// Java `isGenSirt`.
    fn is_gen_sirt(&self, axis_id: AxisID) -> bool;

    /// Java `isGenCtf3dOldStyleXtilting`.
    fn is_gen_ctf_3d_old_style_xtilting(&self, axis_id: AxisID) -> bool;

    /// Java `isGenCtf3dVerticalSlices`.
    fn is_gen_ctf_3d_vertical_slices(&self, axis_id: AxisID) -> bool;

    /// Java `getGenCtf3dFourierReduceByFactor`.
    fn get_gen_ctf_3d_fourier_reduce_by_factor(&self, axis_id: AxisID) -> String;

    /// Java `isSubtomoReorientationTypeNone`.
    fn is_subtomo_reorientation_type_none(&self) -> bool;

    /// Java `isSubtomoReorientationTypeFlipped`.
    fn is_subtomo_reorientation_type_flipped(&self) -> bool;

    /// Java `isSubtomoReorientationTypeRotated`.
    fn is_subtomo_reorientation_type_rotated(&self) -> bool;

    /// Java `isSubtomoMakeVolumeStacks`.
    fn is_subtomo_make_volume_stacks(&self) -> bool;

    /// Java `getSubtomoMakeVolumeStacks`.
    fn get_subtomo_make_volume_stacks(&self) -> String;

    /// Java `getSubtomoExtentOfZLevelsInNm`.
    fn get_subtomo_extent_of_z_levels_in_nm(&self) -> String;

    /// Java `getSubtomoNewAlignedBinning`.
    fn get_subtomo_new_aligned_binning(&self) -> String;

    /// Java `getSubtomoFourierReduceByFactor`.
    fn get_subtomo_fourier_reduce_by_factor(&self) -> String;

    /// Java `getFineLocalAlignValidation`.
    fn get_fine_local_align_validation(&self, axis_id: AxisID) -> String;

    /// Java `isDoseSym`.
    fn is_dose_sym(&self, axis_id: AxisID) -> bool;

    /// Java `getDoseSym`.
    fn get_dose_sym(&self, axis_id: AxisID) -> String;

    /// Java `getAltTomoRootname`.
    fn get_alt_tomo_rootname(&self) -> String;

    /// Java `isAltTomoTrimVolume`.
    fn is_alt_tomo_trim_volume(&self) -> bool;

    /// Java `isAltTomoArchiveOrigStack`.
    fn is_alt_tomo_archive_orig_stack(&self) -> bool;

    /// Java `getImageFilenameStyle`, implemented by `BaseMetaData`.
    fn get_image_filename_style(&self) -> ImageFilenameStyle;

    /// Java `getAxisType`, implemented by `BaseMetaData`.
    fn get_axis_type(&self) -> AxisType;

    /// Java `getCombineParams`.  The field behind its lock.
    fn get_combine_params(
        &self,
    ) -> std::sync::MutexGuard<'_, crate::imod::etomo::comscript::combine_params::CombineParams>;
}
