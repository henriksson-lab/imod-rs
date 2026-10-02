//! `IMOD/Etomo/src/etomo/ui/Run3dmodMenuTarget.java`.

use super::swing::swing_component::SwingComponent;
use crate::imod::etomo::r#type::run_3dmod_menu_options::Run3dmodMenuOptions;

/// Java `Run3dmodMenuTarget.rcsid`.
pub const RCSID: &str = "$Id:$";

/// Java `Run3dmodMenuTarget extends SwingComponent`.  Implementers are EDT objects
/// (`Rc`, `&self` methods).
pub trait Run3dmodMenuTarget: SwingComponent {
    /// Java `menuAction(Run3dmodMenuOptions)`.
    fn menu_action(&self, run_3dmod_menu_options: Run3dmodMenuOptions);

    /// Java `isEnabled()`.
    fn is_enabled(&self) -> bool;
}
