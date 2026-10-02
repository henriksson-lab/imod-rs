//! `IMOD/Etomo/src/etomo/ui/swing/SwingComponent.java`.

use std::rc::Rc;

use crate::imod::etomo::jdk::JComponent;

/// Java `SwingComponent.rcsid`.
pub const RCSID: &str = "$Id:$";

/// Java `public interface SwingComponent`.
pub trait SwingComponent {
    /// Java `getComponent()`: the `java.awt.Component` this object is drawn as.
    fn get_component(&self) -> Rc<JComponent>;
}
