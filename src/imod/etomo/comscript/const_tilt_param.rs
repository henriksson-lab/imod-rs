//! `IMOD/Etomo/src/etomo/comscript/ConstTiltParam.java`.

use super::command_details::CommandDetails;
use crate::imod::etomo::r#type::const_etomo_number::ConstEtomoNumber;

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `ConstTiltParam extends CommandDetails`.
pub trait ConstTiltParam: CommandDetails {
    /// Java `getInputFile`.
    fn get_input_file(&self) -> String;
    /// Java `hasMode`.
    fn has_mode(&self) -> bool;
    /// Java `getExcludeList2`.
    fn get_exclude_list2(&self) -> String;
    /// Java `getImageBinned`.
    fn get_image_binned(&self) -> &ConstEtomoNumber;
    /// Java `getMode`.
    fn get_mode(&self) -> i32;
    /// Java `getOutputFile`.
    fn get_output_file(&self) -> String;
    /// Java `isFiducialess`.
    fn is_fiducialess(&self) -> bool;
    /// Java `getFullImageX`.
    fn get_full_image_x(&self) -> i32;
    /// Java `getIdxSliceStart`.
    fn get_idx_slice_start(&self) -> i32;
    /// Java `getIdxSliceStop`.
    fn get_idx_slice_stop(&self) -> i32;
    /// Java `getLogShift`.
    fn get_log_shift(&self) -> String;
    /// Java `getRadialBandwidth`.
    fn get_radial_bandwidth(&self) -> String;
    /// Java `getRadialFalloff`.
    fn get_radial_falloff(&self) -> String;
    /// Java `getScaleCoeff`.
    fn get_scale_coeff(&self) -> f64;
    /// Java `getScaleFLevel`.
    fn get_scale_f_level(&self) -> f64;
    /// Java `getThickness`.
    fn get_thickness(&self) -> i32;
    /// Java `getTiltAngleOffset`.
    fn get_tilt_angle_offset(&self) -> &ConstEtomoNumber;
    /// Java `getWidth`.
    fn get_width(&self) -> i32;
    /// Java `getXAxisTilt`.
    fn get_x_axis_tilt(&self) -> f64;
    /// Java `getXAxisTiltString`.
    fn get_x_axis_tilt_string(&self) -> String;
    /// Java `getXShift`.
    fn get_x_shift(&self) -> f64;
    /// Java `getZShift`.
    fn get_z_shift(&self) -> &ConstEtomoNumber;
    /// Java `isSuperSampleFactorSet`.
    fn is_super_sample_factor_set(&self) -> bool;
    /// Java `getSuperSampleFactor`.
    fn get_super_sample_factor(&self) -> String;
    /// Java `isExpandInputLinesSet`.
    fn is_expand_input_lines_set(&self) -> bool;
    /// Java `hasLogOffset`.
    fn has_log_offset(&self) -> bool;
    /// Java `hasRadialWeightingFunction`.
    fn has_radial_weighting_function(&self) -> bool;
    /// Java `hasScale`.
    fn has_scale(&self) -> bool;
    /// Java `hasSlice`.
    fn has_slice(&self) -> bool;
    /// Java `hasThickness`.
    fn has_thickness(&self) -> bool;
    /// Java `hasTiltAngleOffset`.
    fn has_tilt_angle_offset(&self) -> bool;
    /// Java `hasWidth`.
    fn has_width(&self) -> bool;
    /// Java `hasXAxisTilt`.
    fn has_x_axis_tilt(&self) -> bool;
    /// Java `hasXShift`.
    fn has_x_shift(&self) -> bool;
    /// Java `hasZShift`.
    fn has_z_shift(&self) -> bool;
    /// Java `isUseGpu`.
    fn is_use_gpu(&self) -> bool;
    /// Java `hasLocalAlignFile`.
    fn has_local_align_file(&self) -> bool;
    /// Java `hasZFactorFileName`.
    fn has_z_factor_file_name(&self) -> bool;
    /// Java `getHammingLikeFilter`.
    fn get_hamming_like_filter(&self) -> String;
    /// Java `getFakeSIRTiterations`.
    fn get_fake_sirt_iterations(&self) -> String;
    /// Java `getExactFilterSize`.
    fn get_exact_filter_size(&self) -> String;
    /// Java `isFalloffIsTrueSigma`.
    fn is_falloff_is_true_sigma(&self) -> bool;
    /// Java `isHammingLikeFilter`.
    fn is_hamming_like_filter(&self) -> bool;
    /// Java `isFakeSIRTiterations`.
    fn is_fake_sirt_iterations(&self) -> bool;
    /// Java `isExactFilterSize`.
    fn is_exact_filter_size(&self) -> bool;
}
