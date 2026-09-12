//! `IMOD/Etomo/src/etomo/ui/swing/RadioEbutton.java`.
//!
//! `JRadioButton`, its model, native listener delivery, and painting remain a
//! GUI boundary.  This unit retains the source's naming, selected/checkpoint,
//! warning-flag, and tooltip transitions.
#![allow(dead_code)]

use std::{cell::RefCell, rc::Rc};

use crate::imod::etomo::{
    etomo_director::ARGUMENTS,
    storage::autodoc::autodoc_tokenizer::{DEFAULT_DELIMITER, SEPARATOR_CHAR},
    util::utilities,
};

use super::{
    appearance_extension::FlagType,
    check_box::Color,
    radio_button::{EnumeratedTypeBoundary, RadioButton, RadioButtonGroup},
};

/// Java final package-private `RadioEbutton`.
#[derive(Clone, Debug)]
pub struct RadioEbutton {
    /// Java final `button`; `RadioButton` is the existing canonical radio
    /// widget boundary used by the current Rust eTomo controls.
    pub button: RadioButton,
    /// Java final `enumeratedType`.
    pub enumerated_type: Option<EnumeratedTypeBoundary>,
    /// Java `stateExtension` checkpoint state.
    pub checkpoint_value: Option<bool>,
    /// Java `flagExtension` state at its BooleanFlagExtension boundary.
    pub warning_enabled: Option<bool>,
    /// Java `defaultBackground`.
    pub default_background: Option<Color>,
    /// Java `flagType`.
    pub flag_type: Option<FlagType>,
    /// Java `button.addActionListener(this)` and external registrations.
    pub action_listener_count: usize,
}

impl RadioEbutton {
    /// Java private `RadioEbutton(EnumeratedType,String,EtomoButtonGroup,ComponentStyleExtension)`.
    fn new(
        enumerated_type: Option<EnumeratedTypeBoundary>,
        label: Option<&str>,
        button_group: Option<Rc<RefCell<RadioButtonGroup>>>,
    ) -> Self {
        let source_label = enumerated_type
            .as_ref()
            .filter(|value| !value.label.is_empty())
            .map(|value| value.label.as_str())
            .or(label);
        let mut button = match (&enumerated_type, button_group) {
            (Some(value), group) => RadioButton::new_with_enumerated_type(
                source_label.map(str::to_owned),
                value.clone(),
                group,
            ),
            (None, Some(group)) => {
                RadioButton::new_in_group(source_label.unwrap_or_default(), group)
            }
            (None, None) => RadioButton::new(source_label.unwrap_or_default()),
        };
        if let Some(label) = source_label {
            button.set_name(label);
        }
        Self {
            button,
            enumerated_type,
            checkpoint_value: None,
            warning_enabled: None,
            default_background: None,
            flag_type: None,
            action_listener_count: 0,
        }
    }

    /// Java `getEnumInstance(EnumeratedType, EtomoButtonGroup)`.
    pub fn get_enum_instance(
        enumerated_type: EnumeratedTypeBoundary,
        button_group: Option<Rc<RefCell<RadioButtonGroup>>>,
    ) -> Self {
        Self::new(Some(enumerated_type), None, button_group)
    }

    /// Java `getInstance(String, EtomoButtonGroup)`.
    pub fn get_instance(label: &str, button_group: Option<Rc<RefCell<RadioButtonGroup>>>) -> Self {
        Self::new(None, Some(label), button_group)
    }

    /// Java `setLabel(String)`.
    pub fn set_label(&mut self, label: &str) {
        self.button.set_text(label);
    }

    /// Java `addActionListener(ActionListener)`.
    pub fn add_action_listener(&mut self) {
        self.action_listener_count += 1;
        self.button.add_action_listener();
    }

    /// Java `getComponent()` boundary.
    pub fn get_component(&self) -> &RadioButton {
        &self.button
    }

    /// Java `actionPerformed(ActionEvent)`.
    pub fn action_performed(&mut self) {
        if self.warning_enabled.is_some() {
            self.update_warning();
        }
    }

    /// Java private `getEnumeratedType()`.
    pub fn get_enumerated_type(&self) -> Option<&EnumeratedTypeBoundary> {
        self.enumerated_type.as_ref()
    }

    /// Java `isEnabled()`.
    pub fn is_enabled(&self) -> bool {
        self.button.is_enabled()
    }

    /// Java `setEnabled(boolean)`.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.button.set_enabled(enabled);
        if self.warning_enabled.is_some() {
            self.set_flag(self.flag_type);
        }
    }

    /// Java `isSelected()`.
    pub fn is_selected(&self) -> bool {
        self.button.is_selected()
    }

    /// Java `equals(boolean)`.
    pub fn equals(&self, value: bool) -> bool {
        self.button.is_selected() == value
    }

    /// Java `setSelected(boolean)`.
    pub fn set_selected(&mut self, selected: bool) {
        self.button.set_selected(selected);
        if self.warning_enabled.is_some() {
            self.update_warning();
        }
    }

    /// Java `getLabel()`.
    pub fn get_label(&self) -> &str {
        self.button.get_text()
    }

    /// Java `checkpoint()`.
    pub fn checkpoint(&mut self) {
        self.checkpoint_value = Some(self.is_selected());
    }

    /// Java `isCheckpointValue()`.
    pub fn is_checkpoint_value(&self) -> bool {
        self.checkpoint_value.unwrap_or(false)
    }

    /// Java `enableWarning(boolean)`.
    pub fn enable_warning(&mut self, value: bool) {
        if self.default_background.is_none() {
            self.default_background = Some(Color(128, 128, 128));
        }
        if self.warning_enabled.is_none() {
            self.add_action_listener();
        }
        self.warning_enabled = Some(value);
        self.update_warning();
    }

    /// Java `disableWarning()`.
    pub fn disable_warning(&mut self) {
        if self.warning_enabled.is_none() {
            return;
        }
        self.warning_enabled = Some(false);
        self.update_warning();
    }

    /// Java `setFlag(FlagType)`.
    pub fn set_flag(&mut self, flag_type: Option<FlagType>) {
        self.flag_type = flag_type;
        // Java ComponentStyleExtension updates Swing painting at this boundary.
        // `RadioButton` retains a separate historical Color representation, so
        // the source flag identity (rather than lossy colour conversion) is
        // the cross-unit state retained here.
    }

    /// Java `setFormattedTooltip(String)`.
    pub fn set_formatted_tooltip(&mut self, tooltip: Option<&str>) {
        self.button.set_preformatted_tooltip(tooltip);
    }

    /// Java `setTooltip(String)`.
    pub fn set_tooltip(&mut self, text: &str) {
        self.button.set_tool_tip_text(Some(text));
    }

    /// Java `setTooltip(String, ReadOnlySection)` at the autodoc boundary.
    pub fn set_autodoc_tooltip(&mut self, tooltip: Option<&str>) {
        self.set_tooltip(tooltip.unwrap_or_default());
    }

    /// Java `BooleanFlagExtension.update()` via this unit's selected origin.
    fn update_warning(&mut self) {
        self.set_flag(
            self.warning_enabled
                .filter(|value| *value && self.is_selected())
                .map(|_| FlagType::WARNING),
        );
    }

    /// Source test-name side effect of Java private `setName(String)`.
    pub fn set_name(&mut self, label: &str) {
        self.button.set_name(label);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selected_warning_and_checkpoint_follow_source_order() {
        let mut button = RadioEbutton::get_instance("File", None);
        button.checkpoint();
        button.enable_warning(true);
        assert_eq!(button.action_listener_count, 1);
        assert_eq!(button.flag_type, None);
        button.set_selected(true);
        assert_eq!(button.flag_type, Some(FlagType::WARNING));
        assert!(!button.is_checkpoint_value());
    }
}
