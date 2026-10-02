//! `IMOD/Etomo/src/etomo/comscript/ConstMTFFilterParam.java`.

use super::command::Command;

/// Java `ConstMTFFilterParam extends Command`.
pub trait ConstMTFFilterParam: Command {
    /// Java `getMtfFile`.  Java null -> `None`.
    fn get_mtf_file(&self) -> Option<String>;

    /// Java `getMaximumInverseString`.
    fn get_maximum_inverse_string(&self) -> String;

    /// Java `getLowPassRadiusSigmaString`.
    fn get_low_pass_radius_sigma_string(&self) -> String;

    /// Java `getStartingAndEndingZString`.
    fn get_starting_and_ending_z_string(&self) -> String;

    /// Java `isStartingZSet`.
    fn is_starting_z_set(&self) -> bool;

    /// Java `isEndingZSet`.
    fn is_ending_z_set(&self) -> bool;

    /// Java `getStartingZ`.
    fn get_starting_z(&self) -> i32;

    /// Java `getEndingZ`.
    fn get_ending_z(&self) -> i32;

    /// Java `getInverseRolloffRadiusSigmaString`.
    fn get_inverse_rolloff_radius_sigma_string(&self) -> String;

    /// Java `getOutputFile`.  Java null -> `None`.
    fn get_output_file(&self) -> Option<String>;

    /// Java `isTypeOfDoseFileSet`.
    fn is_type_of_dose_file_set(&self) -> bool;

    /// Java `isFixedImageDoseSet`.
    fn is_fixed_image_dose_set(&self) -> bool;

    /// Java `isMaximumInverseSet`.
    fn is_maximum_inverse_set(&self) -> bool;

    /// Java `isOptimalDoseScalingSet`.
    fn is_optimal_dose_scaling_set(&self) -> bool;

    /// Java `isBidirectionalNumViewsSet`.
    fn is_bidirectional_num_views_set(&self) -> bool;

    /// Java `isLowPassRadiusSigmaSet`.
    fn is_low_pass_radius_sigma_set(&self) -> bool;

    /// Java `isInverseRolloffRadiusSigmaSet`.
    fn is_inverse_rolloff_radius_sigma_set(&self) -> bool;

    /// Java `isMtfFileSet`.
    fn is_mtf_file_set(&self) -> bool;

    /// Java `getFixedImageDose`.
    fn get_fixed_image_dose(&self) -> String;

    /// Java `getDoseWeightingFile`.
    fn get_dose_weighting_file(&self) -> String;

    /// Java `isDoseWeightingFileSet`.
    fn is_dose_weighting_file_set(&self) -> bool;

    /// Java `getTypeOfDoseFile`.
    fn get_type_of_dose_file(&self) -> String;

    /// Java `isVoltage200`.
    fn is_voltage200(&self) -> bool;

    /// Java `getOptimalDoseScaling`.
    fn get_optimal_dose_scaling(&self) -> String;

    /// Java `getBidirectionalNumViews`.
    fn get_bidirectional_num_views(&self) -> String;
}
