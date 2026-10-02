//! `IMOD/Etomo/src/etomo/ui/UIComponent.java`.

use std::rc::Rc;

use super::swing::swing_component::SwingComponent;
use crate::imod::etomo::jdk::JComponent;

/// Java `UIComponent.rcsid`.
pub const RCSID: &str = "$Id:$";

/// Java `UIComponent`.  Implementers are EDT objects (`Rc`, `&self` methods).
pub trait UIComponent {
    /// Java `getUIComponent()`: the object itself, seen as a `SwingComponent`.
    fn get_ui_component(&self) -> &dyn SwingComponent;

    /// Java `getComponent()`: the `java.awt.Component` this object is drawn as.
    fn get_component(&self) -> Rc<JComponent>;
}
