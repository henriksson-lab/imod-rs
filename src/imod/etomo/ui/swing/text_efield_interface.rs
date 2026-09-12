//! `IMOD/Etomo/src/etomo/ui/swing/TextEfieldInterface.java`.
#![allow(dead_code)]

/// Java public `TextEfieldInterface`.
pub trait TextEfieldInterface {
    fn get_directive_def(&self) -> Option<&str>;
    fn is_enabled(&self) -> bool;
    fn is_visible(&self) -> bool;
    fn get_text(&self) -> String;
    fn set_text(&mut self, text: String);
    fn set_field_highlight(&mut self, text: String);
    fn set_template_value(&mut self);
    fn equals(&self, string: Option<&str>) -> bool;
    fn set_debug(&mut self, debug: bool);
}
