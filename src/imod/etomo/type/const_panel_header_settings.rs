//! `IMOD/Etomo/src/etomo/type/ConstPanelHeaderSettings.java`.

/// Java `rcsid`.
pub const RCSID: &str = "$Id:$";

/// Java `ConstPanelHeaderSettings`.
pub trait ConstPanelHeaderSettings {
    /// Java `isAdvanced()`.
    fn is_advanced(&self) -> bool;
    /// Java `isMore()`.
    fn is_more(&self) -> bool;
    /// Java `isOpen()`.
    fn is_open(&self) -> bool;
    /// Java `isAdvancedNull()`.
    fn is_advanced_null(&self) -> bool;
    /// Java `isOpenNull()`.
    fn is_open_null(&self) -> bool;
    /// Java `isMoreNull()`.
    fn is_more_null(&self) -> bool;
}
