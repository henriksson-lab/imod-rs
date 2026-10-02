//! `IMOD/Etomo/src/etomo/ui/FieldDisplayer.java`.
//!
//! Interface for a class that makes sure a field or group of fields are displayed.

use super::ui_component::UIComponent;

/// Java `FieldDisplayer`.  Implementers are EDT objects (`Rc`, `&self` methods).
pub trait FieldDisplayer {
    /// Java `display()`.  Function to make a field or group of fields visible.  This
    /// function may be called when the field is already visible.  It should do nothing
    /// in this case.
    fn display_void(&self);

    /// Java `display(UIComponent)`.
    fn display_ui_component(&self, ui_component: Option<&dyn UIComponent>);
}
