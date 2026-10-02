//! `IMOD/Etomo/src/etomo/ui/FieldSettingInterface.java`.
//!
//! The settings (`BooleanFieldSetting`, `TextFieldSetting`, `FieldSettingBundle`) are
//! plain values held inside a field's cell.  Java's `getBooleanSetting()` and
//! `getTextSetting()` hand out a reference to the held setting; every Java caller only
//! reads it (`BooleanFieldSetting.copy`, `TextFieldSetting.copy`,
//! `FieldSettingBundle.addBooleanSetting`/`addTextSetting`), so the Rust methods return
//! an owned copy, which is the same value.

use super::boolean_field_setting::BooleanFieldSetting;
use super::text_field_setting::TextFieldSetting;

/// Java `FieldSettingInterface`.
pub trait FieldSettingInterface {
    /// Java `getBooleanSetting()`.
    fn get_boolean_setting(&self) -> Option<BooleanFieldSetting>;

    /// Java `getTextSetting()`.
    fn get_text_setting(&self) -> Option<TextFieldSetting>;

    /// Java `isBoolean()`.
    fn is_boolean(&self) -> bool;

    /// Java `isText()`.
    fn is_text(&self) -> bool;

    /// Java `isSet()`.
    fn is_set(&self) -> bool;
}
