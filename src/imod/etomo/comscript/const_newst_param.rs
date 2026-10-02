//! `IMOD/Etomo/src/etomo/comscript/ConstNewstParam.java`.

use super::command_details::CommandDetails;

/// Java `ConstNewstParam extends CommandDetails`.
pub trait ConstNewstParam: CommandDetails {
    /// Java `getBinByFactor`.  Returns the binByFactor.
    fn get_bin_by_factor(&self) -> i32;
    /// Java `getFloatDensities`.  Returns the floatDensities.
    fn get_float_densities(&self) -> i32;
    /// Java `getInputFile`.  Backward compatibility with pre PIP structure, just
    /// return the first input file.  `None` when there is no input file (see
    /// `NewstParam::get_input_file`).
    fn get_input_file(&self) -> Option<String>;
    /// Java `isLinearInterpolation`.
    fn is_linear_interpolation(&self) -> bool;
    /// Java `getModeToOutput`.
    fn get_mode_to_output(&self) -> i32;
    /// Java `getOutputFile`.  Backward compatibility with pre PIP structure, just
    /// return the first ouput file.
    fn get_output_file(&self) -> String;
    /// Java `getSizeToOutputInX`.
    fn get_size_to_output_in_x(&self) -> i32;
    /// Java `getSizeToOutputInY`.
    fn get_size_to_output_in_y(&self) -> i32;
    /// Java `isSizeToOutputInXandYSet`.
    fn is_size_to_output_in_x_and_y_set(&self) -> bool;
    /// Java `fillValueEquals(int)`.
    fn fill_value_equals(&self, value: i32) -> bool;
    /// Java `getOffsetInX`.
    fn get_offset_in_x(&self) -> String;
    /// Java `getOffsetInY`.
    fn get_offset_in_y(&self) -> String;
    /// Java `isAntialiasFilterNull`.
    fn is_antialias_filter_null(&self) -> bool;
    /// Java `getAntialiasFilter`.
    fn get_antialias_filter(&self) -> String;
}
