//! `IMOD/Etomo/src/etomo/ui/swing/ExpandButton.java`.
//!
//! `SingleLineButton`, `JPanel`, `GridBagLayout`, and action-listener
//! installation are retained as direct native-Swing boundaries.  This unit
//! keeps the Java button state, text, tooltip, naming, and event order.
#![allow(dead_code)]

use super::expandable::Expandable;
use super::process_dialog::GlobalExpandButton;
use crate::imod::etomo::storage::autodoc::autodoc_tokenizer::SEPARATOR_CHAR;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::util::utilities;

const DEFAULT_TYPE: ExpandButtonType = ExpandButtonType::More;

/// Java static inner `ExpandButton.Type`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExpandButtonType {
    More,
    Advanced,
    Open,
}

impl ExpandButtonType {
    /// Java `Type.getUnformattedText(boolean)`.
    pub fn get_unformatted_text(self, expanded: bool) -> &'static str {
        match (self, expanded) {
            (Self::More, true) => "<",
            (Self::More, false) => ">",
            (Self::Advanced, true) => "B",
            (Self::Advanced, false) => "A",
            (Self::Open, true) => "-",
            (Self::Open, false) => "+",
        }
    }

    /// Java `Type.getState(boolean)`.
    pub fn get_state(self, expanded: bool) -> &'static str {
        match (self, expanded) {
            (Self::More, true) => "more",
            (Self::More, false) => "less",
            (Self::Advanced, true) => "advanced",
            (Self::Advanced, false) => "basic",
            (Self::Open, true) => "open",
            (Self::Open, false) => "closed",
        }
    }

    /// Java `Type.getExpandedState()`.
    pub fn get_expanded_state(self) -> &'static str {
        self.get_state(true)
    }

    /// Java `Type.getContractedState()`.
    pub fn get_contracted_state(self) -> &'static str {
        self.get_state(false)
    }

    /// Java `Type.getSymbol(boolean)`.
    pub fn get_symbol(self, expanded: bool) -> &'static str {
        match (self, expanded) {
            (Self::More, true) => "<html>&lt",
            (Self::More, false) => "<html>&gt",
            _ => self.get_unformatted_text(expanded),
        }
    }

    /// Java `Type.getToolTip(boolean)`.
    pub fn get_tool_tip(self, expanded: bool) -> &'static str {
        match (self, expanded) {
            (Self::More, true) => "Show less.",
            (Self::More, false) => "Show more.",
            (Self::Advanced, true) => "Show basic options.",
            (Self::Advanced, false) => "Show all options.",
            (Self::Open, true) => "Close panel.",
            (Self::Open, false) => "Open panel.",
        }
    }

    /// Java `Type.equals(AbstractButton, String)` at the Swing text boundary.
    pub fn equals(button_text: Option<&str>, input: Option<&str>) -> bool {
        let (Some(button_text), Some(input)) = (button_text, input) else {
            return false;
        };
        let text = utilities::strip_html_tags(Some(button_text));
        let Some(text) = text else { return false };
        let symbol = match text.as_str() {
            "&lt" => "<",
            "&gt" => ">",
            "B" | "A" | "-" | "+" => text.as_str(),
            _ => return false,
        };
        symbol == input
    }
}

/// Java `ExpandButton`, including its explicit native-Swing boundary state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExpandButton {
    pub button_type: ExpandButtonType,
    pub expanded: bool,
    pub name: String,
    pub state_key: Option<String>,
    pub global_expand_button_present: bool,
    pub expandable1_present: bool,
    pub expandable2_present: bool,
    pub text: String,
    pub tool_tip_text: String,
    pub manual_name: bool,
    pub action_listener_present: bool,
    pub raised_bevel_border: bool,
    pub container_present: bool,
    pub original_process_result_display_state: Option<bool>,
    pub debug: bool,
}

impl ExpandButton {
    /// Java first `getInstance(Expandable, Type)` overload.
    pub fn get_instance(
        expandable: Option<&dyn Expandable>,
        button_type: Option<ExpandButtonType>,
    ) -> Self {
        Self::new_with_owners(
            expandable.is_some(),
            false,
            button_type.unwrap_or(DEFAULT_TYPE),
            false,
            false,
        )
    }

    /// Java second `getInstance(Expandable, Expandable, Type)` overload.
    pub fn get_instance_two(
        expandable1: Option<&dyn Expandable>,
        expandable2: Option<&dyn Expandable>,
        button_type: Option<ExpandButtonType>,
    ) -> Self {
        Self::new_with_owners(
            expandable1.is_some(),
            expandable2.is_some(),
            button_type.unwrap_or(DEFAULT_TYPE),
            false,
            false,
        )
    }

    /// Java first `getGlobalInstance` overload.
    pub fn get_global_instance(
        expandable: Option<&dyn Expandable>,
        button_type: Option<ExpandButtonType>,
        global_expand_button: Option<&GlobalExpandButton>,
    ) -> Self {
        Self::new_with_owners(
            expandable.is_some(),
            false,
            button_type.unwrap_or(DEFAULT_TYPE),
            false,
            global_expand_button.is_some(),
        )
    }

    /// Java second `getGlobalInstance` overload.
    pub fn get_global_instance_two(
        expandable1: Option<&dyn Expandable>,
        expandable2: Option<&dyn Expandable>,
        button_type: Option<ExpandButtonType>,
        global_expand_button: Option<&GlobalExpandButton>,
    ) -> Self {
        Self::new_with_owners(
            expandable1.is_some(),
            expandable2.is_some(),
            button_type.unwrap_or(DEFAULT_TYPE),
            false,
            global_expand_button.is_some(),
        )
    }

    /// Java `getExpandedInstance`.
    pub fn get_expanded_instance(
        expandable1: Option<&dyn Expandable>,
        expandable2: Option<&dyn Expandable>,
        button_type: Option<ExpandButtonType>,
    ) -> Self {
        Self::new_with_owners(
            expandable1.is_some(),
            expandable2.is_some(),
            button_type.unwrap_or(DEFAULT_TYPE),
            true,
            false,
        )
    }

    /// Java private constructor result, used by `PanelHeader` too.
    pub fn new(
        button_type: ExpandButtonType,
        expanded: bool,
        global_expand_button_present: bool,
    ) -> Self {
        Self::new_with_owners(
            false,
            false,
            button_type,
            expanded,
            global_expand_button_present,
        )
    }

    /// Java private five-argument constructor, with non-owning Rust event endpoints.
    pub fn new_with_owners(
        expandable1_present: bool,
        expandable2_present: bool,
        button_type: ExpandButtonType,
        expanded: bool,
        global_expand_button_present: bool,
    ) -> Self {
        Self {
            button_type,
            expanded,
            name: String::new(),
            state_key: None,
            global_expand_button_present,
            expandable1_present,
            expandable2_present,
            text: button_type.get_symbol(expanded).to_string(),
            tool_tip_text: button_type.get_tool_tip(expanded).to_string(),
            manual_name: true,
            action_listener_present: true,
            raised_bevel_border: true,
            container_present: false,
            original_process_result_display_state: None,
            debug: false,
        }
    }

    /// Java static `equals(AbstractButton, String)`.
    pub fn equals_text(button_text: Option<&str>, input: Option<&str>) -> bool {
        ExpandButtonType::equals(button_text, input)
    }

    /// Java overridden `setName(String)`.
    pub fn set_name(&mut self, associated_label: &str) {
        self.name = format!(
            "mb{SEPARATOR_CHAR}{}",
            utilities::convert_label_to_name(Some(associated_label), true).unwrap_or_default()
        );
    }

    /// Java `isExpanded()`.
    pub fn is_expanded(&self) -> bool {
        self.expanded
    }

    /// Java overridden `createButtonStateKey(DialogType)`.
    pub fn create_button_state_key(&mut self, dialog_type: DialogType) -> String {
        let state_key = format!(
            "{}.{}.{}",
            dialog_type.get_storable_name(),
            self.name,
            self.button_type.get_expanded_state()
        );
        self.state_key = Some(state_key.clone());
        state_key
    }

    /// Java overridden `getButtonState()`.
    pub fn get_button_state(&self) -> bool {
        self.is_expanded()
    }

    /// Java `setButtonState(boolean)`.
    pub fn set_button_state(&mut self, state: bool) {
        self.original_process_result_display_state = Some(state);
        self.set_expanded(state);
    }

    /// Java package-private `getState()`.
    pub fn get_state(&self) -> String {
        self.button_type.get_state(self.expanded).to_string()
    }

    /// Java package-private `setState(String)`.
    pub fn set_state(&mut self, state: Option<&str>) -> bool {
        let Some(state) = state else { return false };
        if state == self.button_type.get_expanded_state() && !self.expanded {
            self.set_expanded(true);
            return true;
        }
        if state == self.button_type.get_contracted_state() && self.expanded {
            self.set_expanded(false);
            return true;
        }
        false
    }

    /// Java `getPreferredWidth()` at the native preferred-size boundary.
    pub fn get_preferred_width(&self) -> usize {
        self.button_type.get_unformatted_text(self.expanded).len()
    }

    /// Java `add(JPanel, GridBagLayout, GridBagConstraints)` boundary.
    pub fn add(&mut self) {
        self.container_present = true;
    }

    /// Java `remove()`.
    pub fn remove(&mut self) {
        self.container_present = false;
    }

    /// Java identity `equals(ExpandButton)`; Rust callers preserve identity by
    /// comparing the address of the owning source field.
    pub fn equals(&self, that: &Self) -> bool {
        std::ptr::eq(self, that)
    }

    /// Java `setExpanded(boolean)`: force its subsequent action even unchanged.
    pub fn set_expanded(&mut self, expanded: bool) {
        if self.expanded == expanded {
            self.expanded = !expanded;
        }
        self.button_action();
    }

    /// Java `update(boolean)`, deliberately without owner notification.
    pub fn update(&mut self, expanded: bool) {
        if self.expanded == expanded {
            return;
        }
        self.expanded = expanded;
        self.text = self.button_type.get_symbol(expanded).to_string();
        self.tool_tip_text = self.button_type.get_tool_tip(expanded).to_string();
    }

    /// Java private `buttonAction()`; owner/global dispatch stays at the
    /// explicit Rust GUI boundary because Java's retained callback objects are
    /// incompatible with safe self-referential Rust ownership.
    pub fn button_action(&mut self) {
        self.expanded = !self.expanded;
        self.text = self.button_type.get_symbol(self.expanded).to_string();
        self.tool_tip_text = self.button_type.get_tool_tip(self.expanded).to_string();
    }

    /// Java `ExpandButtonActionListener.actionPerformed(ActionEvent)`.
    pub fn action_performed(&mut self) {
        self.button_action();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::etomo::r#type::dialog_type::DialogType;

    #[test]
    fn type_text_state_and_tooltip_match_java() {
        assert_eq!(ExpandButtonType::More.get_symbol(true), "<html>&lt");
        assert_eq!(ExpandButtonType::More.get_unformatted_text(false), ">");
        assert_eq!(ExpandButtonType::Advanced.get_state(false), "basic");
        assert_eq!(ExpandButtonType::Open.get_tool_tip(true), "Close panel.");
        assert!(ExpandButton::equals_text(Some("<html>&lt"), Some("<")));
    }

    #[test]
    fn expanded_forces_action_and_update_does_not() {
        let mut button = ExpandButton::new(ExpandButtonType::More, false, false);
        button.set_expanded(false);
        assert!(!button.expanded);
        assert_eq!(button.text, "<html>&gt");
        button.update(true);
        assert!(button.expanded);
        assert_eq!(button.text, "<html>&lt");
    }

    #[test]
    fn state_name_and_storage_key_follow_source() {
        let mut button = ExpandButton::new(ExpandButtonType::Advanced, false, false);
        button.set_name("Tilt alignment");
        assert_eq!(button.name, "mb.tilt-alignment");
        assert!(button.set_state(Some("advanced")));
        assert_eq!(button.get_state(), "advanced");
        assert_eq!(
            button.create_button_state_key(DialogType::FineAlignment),
            "FineAlign.mb.tilt-alignment.advanced"
        );
    }
}
