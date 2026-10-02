//! `IMOD/Etomo/src/etomo/ui/swing/FinalCombineFields.java`.
//!
//! Package-local interface to the final combine screen fields, implemented by
//! the Setup tab (`SetupCombinePanel`) and the Final Match tab
//! (`FinalCombinePanel`); `TomogramCombinationDialog.synchronize` copies these
//! fields from one to the other.  The implementers are EDT objects, so every
//! method takes `&self`.  Java `String` values are `Option<String>` (null).

/// Java `rcsid`.
pub const RCSID: &str = "$$Id$$";

/// Java package-local `interface FinalCombineFields`.
pub trait FinalCombineFields {
    /// Java `setUsePatchRegionModel(boolean)`.
    fn set_use_patch_region_model(&self, use_patch_region_model: bool);

    /// Java `isUsePatchRegionModel()`.
    fn is_use_patch_region_model(&self) -> bool;

    /// Java `setXMin(String)`.
    fn set_x_min(&self, x_min: Option<&str>);

    /// Java `getXMin()`.
    fn get_x_min(&self) -> Option<String>;

    /// Java `setXMax(String)`.
    fn set_x_max(&self, x_max: Option<&str>);

    /// Java `getXMax()`.
    fn get_x_max(&self) -> Option<String>;

    /// Java `setYMin(String)`.
    fn set_y_min(&self, y_min: Option<&str>);

    /// Java `getYMin()`.
    fn get_y_min(&self) -> Option<String>;

    /// Java `setYMax(String)`.
    fn set_y_max(&self, y_max: Option<&str>);

    /// Java `getYMax()`.
    fn get_y_max(&self) -> Option<String>;

    /// Java `setZMin(String)`.
    fn set_z_min(&self, z_min: Option<&str>);

    /// Java `getZMin()`.
    fn get_z_min(&self) -> Option<String>;

    /// Java `setZMax(String)`.
    fn set_z_max(&self, z_max: Option<&str>);

    /// Java `getZMax()`.
    fn get_z_max(&self) -> Option<String>;

    /// Java `setParallel(boolean)`.
    fn set_parallel(&self, parallel: bool);

    /// Java `isParallel()`.
    fn is_parallel(&self) -> bool;

    /// Java `setParallelEnabled(boolean)`.
    fn set_parallel_enabled(&self, parallel_enabled: bool);

    /// Java `isParallelEnabled()`.
    fn is_parallel_enabled(&self) -> bool;

    /// Java `setNoVolcombine(boolean)`.
    fn set_no_volcombine(&self, no_volcombine: bool);

    /// Java `isNoVolcombine()`.
    fn is_no_volcombine(&self) -> bool;

    /// Java `isEnabled()`.
    fn is_enabled(&self) -> bool;
}
