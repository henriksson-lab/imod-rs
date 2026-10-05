//! `IMOD/Etomo/src/etomo/type/ConstPeetScreenState.java`.

use super::panel_header_state::PanelHeaderState;

/// Java `public static final String rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `public interface ConstPeetScreenState`.
pub trait ConstPeetScreenState {
    /// Java `getPeetSetupHeaderState()`.
    fn get_peet_setup_header_state(&self) -> &PanelHeaderState;

    /// Java `getPeetRunHeaderState()`.
    fn get_peet_run_header_state(&self) -> &PanelHeaderState;
}
