//! `IMOD/Etomo/src/etomo/comscript/ConstMatchorwarpParam.java`.

/// Java `ConstMatchorwarpParam` interface.
pub trait ConstMatchorwarpParam {
    /// Java `isUseModelFile`.
    fn is_use_model_file(&self) -> bool;
    /// Java `getRefineLimit`.
    fn get_refine_limit(&self) -> String;
    /// Java `getWarpLimits`.
    fn get_warp_limits(&self) -> String;
    /// Java `getXLowerExclude`.
    fn get_x_lower_exclude(&self) -> i32;
    /// Java `getXUpperExclude`.
    fn get_x_upper_exclude(&self) -> i32;
    /// Java `getZLowerExclude`.
    fn get_z_lower_exclude(&self) -> i32;
    /// Java `getZUpperExclude`.
    fn get_z_upper_exclude(&self) -> i32;
    /// Java `isLinearInterpolation`.
    fn is_linear_interpolation(&self) -> bool;
    /// Java `isXLowerExcludeSet`.
    fn is_x_lower_exclude_set(&self) -> bool;
    /// Java `isXUpperExcludeSet`.
    fn is_x_upper_exclude_set(&self) -> bool;
    /// Java `isZLowerExcludeSet`.
    fn is_z_lower_exclude_set(&self) -> bool;
    /// Java `isZUpperExcludeSet`.
    fn is_z_upper_exclude_set(&self) -> bool;
}
