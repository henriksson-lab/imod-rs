//! `IMOD/Etomo/src/etomo/comscript/ConstCtfPlotterParam.java`.

use crate::imod::etomo::r#type::const_etomo_number::ConstEtomoNumber;

/// Java `ConstCtfPlotterParam`.
pub trait ConstCtfPlotterParam {
    /// Java `getConfigFile`.
    fn get_config_file(&self) -> String;

    /// Java `getScanDefocusRange`.  Java null -> `None`.
    fn get_scan_defocus_range(&self) -> Option<String>;

    /// Java `getScanDefocusRangeLow`.  Java null -> `None`.
    fn get_scan_defocus_range_low(&self) -> Option<f64>;

    /// Java `getScanDefocusRangeHigh`.  Java null -> `None`.
    fn get_scan_defocus_range_high(&self) -> Option<f64>;

    /// Java `getExpectedDefocus`.  Java null -> `None`.
    fn get_expected_defocus(&self) -> Option<&ConstEtomoNumber>;

    /// Java `getPhaseShiftInDegrees`.
    fn get_phase_shift_in_degrees(&self) -> String;

    /// Java `getOffsetToAdd`.
    fn get_offset_to_add(&self) -> &ConstEtomoNumber;

    /// Java `isScanDefocusRange`.
    fn is_scan_defocus_range(&self) -> bool;

    /// Java `isScanDefocusRangeLow`.
    fn is_scan_defocus_range_low(&self) -> bool;

    /// Java `isScanDefocusRangeHigh`.
    fn is_scan_defocus_range_high(&self) -> bool;
}
