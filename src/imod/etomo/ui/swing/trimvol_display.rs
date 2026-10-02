//! `IMOD/Etomo/src/etomo/ui/swing/TrimvolDisplay.java`.

/// Java `TrimvolDisplay`.  UI objects take `&self` (interior mutability).
pub trait TrimvolDisplay {
    /// Java `setSwapYZ(boolean)`.
    fn set_swap_yz(&self, input: bool);
    /// Java `setRotateX(boolean)`.
    fn set_rotate_x(&self, input: bool);
    /// Java `setConvertToBytes(boolean)`.
    fn set_convert_to_bytes(&self, input: bool);
    /// Java `setSectionScaleMin(String)`.
    fn set_section_scale_min(&self, input: &str);
    /// Java `setSectionScaleMax(String)`.
    fn set_section_scale_max(&self, input: &str);
    /// Java `setXMin(String)`.
    fn set_x_min(&self, input: &str);
    /// Java `setXMax(String)`.
    fn set_x_max(&self, input: &str);
    /// Java `setYMin(String)`.
    fn set_y_min(&self, input: &str);
    /// Java `setYMax(String)`.
    fn set_y_max(&self, input: &str);
    /// Java `setZMin(String)`.
    fn set_z_min(&self, input: &str);
    /// Java `setZMax(String)`.
    fn set_z_max(&self, input: &str);
    /// Java `setScaleXMin(String)`.
    fn set_scale_x_min(&self, input: &str);
    /// Java `setScaleYMin(String)`.
    fn set_scale_y_min(&self, input: &str);
    /// Java `setScaleXMax(String)`.
    fn set_scale_x_max(&self, input: &str);
    /// Java `setScaleYMax(String)`.
    fn set_scale_y_max(&self, input: &str);
}
