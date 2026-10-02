//! `IMOD/Etomo/src/etomo/ui/FieldSettingBundle.java`.
//!
//! Holds a boolean setting and a text setting.  Both are optional.  Built and filled
//! on the event dispatch thread by the combined fields (`RadioTextField`), so the two
//! settings sit in cells and the methods take `&self`.

use std::cell::RefCell;

use super::boolean_field_setting::BooleanFieldSetting;
use super::field_setting_interface::FieldSettingInterface;
use super::text_field_setting::TextFieldSetting;

/// Java `FieldSettingBundle`.
#[derive(Debug, Default)]
pub struct FieldSettingBundle {
    /// Java private `boolSetting`, initialised to null.
    bool_setting: RefCell<Option<BooleanFieldSetting>>,
    /// Java private `textSetting`, initialised to null.
    text_setting: RefCell<Option<TextFieldSetting>>,
}

impl FieldSettingBundle {
    /// Java's implicit `FieldSettingBundle()`.
    pub fn new() -> FieldSettingBundle {
        FieldSettingBundle {
            bool_setting: RefCell::new(None),
            text_setting: RefCell::new(None),
        }
    }

    /// Java `addBooleanSetting(FieldSettingInterface)`.
    pub fn add_boolean_setting(&self, input: Option<&dyn FieldSettingInterface>) {
        if let Some(input) = input {
            *self.bool_setting.borrow_mut() = input.get_boolean_setting();
        } else {
            *self.bool_setting.borrow_mut() = None;
        }
    }

    /// Java `addTextSetting(FieldSettingInterface)`.
    pub fn add_text_setting(&self, input: Option<&dyn FieldSettingInterface>) {
        if let Some(input) = input {
            *self.text_setting.borrow_mut() = input.get_text_setting();
        } else {
            *self.text_setting.borrow_mut() = None;
        }
    }
}

impl FieldSettingInterface for FieldSettingBundle {
    /// Java `isBoolean()`.
    fn is_boolean(&self) -> bool {
        self.bool_setting.borrow().is_some()
    }

    /// Java `isText()`.
    fn is_text(&self) -> bool {
        self.text_setting.borrow().is_some()
    }

    /// Java `isSet()`.
    fn is_set(&self) -> bool {
        self.bool_setting
            .borrow()
            .as_ref()
            .is_some_and(|bool_setting| bool_setting.is_set())
            || self
                .text_setting
                .borrow()
                .as_ref()
                .is_some_and(|text_setting| text_setting.is_set())
    }

    /// Java `getBooleanSetting()`.
    fn get_boolean_setting(&self) -> Option<BooleanFieldSetting> {
        self.bool_setting.borrow().clone()
    }

    /// Java `getTextSetting()`.
    fn get_text_setting(&self) -> Option<TextFieldSetting> {
        self.text_setting.borrow().clone()
    }
}
