//! `IMOD/Etomo/src/etomo/type/ConstPanelHeaderState.java`.

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `ConstPanelHeaderState`.  Java null is `None`.
pub trait ConstPanelHeaderState {
    /// Java `getOpenCloseState`.
    fn get_open_close_state(&self) -> Option<String>;
    /// Java `getAdvancedBasicState`.
    fn get_advanced_basic_state(&self) -> Option<String>;
    /// Java `getMoreLessState`.
    fn get_more_less_state(&self) -> Option<String>;
}
