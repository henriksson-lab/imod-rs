//! `IMOD/Etomo/src/etomo/ui/swing/TextEfieldInterface.java`.
//!
//! The public interface of a text Efield (implemented by `TextEfield` and the
//! other text Efields), used by `etomo.ui.TextStateExtension` and the directive
//! code.  Implementers are EDT objects (`Rc`, `&self` methods).

use crate::imod::etomo::storage::directive_def::DirectiveDef;

/// Java `TextEfieldInterface`.
pub trait TextEfieldInterface {
    /// Java `getDirectiveDef()`.
    fn get_directive_def(&self) -> Option<DirectiveDef>;

    /// Java `isEnabled()`.
    fn is_enabled(&self) -> bool;

    /// Java `isVisible()`.
    fn is_visible(&self) -> bool;

    /// Java `getText()`.
    fn get_text(&self) -> Option<String>;

    /// Java `setText(String)`.
    fn set_text(&self, text: Option<&str>);

    /// Java `setFieldHighlight(String)`.
    fn set_field_highlight(&self, text: Option<&str>);

    /// Java `setTemplateValue()`.
    fn set_template_value(&self);

    /// Java `equals(String)`.
    fn equals(&self, string: Option<&str>) -> bool;

    /// Java `setDebug(boolean)`.
    fn set_debug(&self, debug: bool);
}
