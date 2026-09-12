//! `IMOD/Etomo/src/etomo/ui/swing/RadioButton.java`.
//!
//! Swing's `JRadioButton`, `ButtonGroup`, button model, listener dispatch, and
//! painting are kept at the GUI boundary.  The source-owned selection,
//! lock/editability, naming, checkpoint, highlight, and tooltip state remain
//! explicit here.
#![allow(dead_code)]

use std::cell::RefCell;
use std::rc::Rc;

use crate::imod::etomo::etomo_director::ARGUMENTS;
use crate::imod::etomo::storage::autodoc::autodoc_tokenizer::{DEFAULT_DELIMITER, SEPARATOR_CHAR};
use crate::imod::etomo::ui::swing::button_component::{ActionListenerBoundary, ButtonComponent};
use crate::imod::etomo::ui::swing::check_box::{
    BLACK, BooleanFieldSetting, Color, FIELD_HIGHLIGHT,
};
use crate::imod::etomo::util::utilities;

/// Source information obtained through Java `EnumeratedType`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EnumeratedTypeBoundary {
    pub label: String,
    pub default: bool,
    pub value: Option<String>,
}

/// Java `ButtonGroup` selection boundary.
#[derive(Clone, Debug, Default)]
pub struct RadioButtonGroup {
    next_id: usize,
    selected_id: Option<usize>,
}

impl RadioButtonGroup {
    pub fn new() -> Self {
        Self::default()
    }
    fn add(&mut self) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        id
    }
    fn set_selected(&mut self, id: usize, selected: bool) {
        if selected {
            self.selected_id = Some(id);
        } else if self.selected_id == Some(id) {
            // Java ButtonGroup does not deselect its selected button.
        }
    }
    fn is_selected(&self, id: usize, fallback: bool) -> bool {
        self.selected_id
            .map_or(fallback, |selected_id| selected_id == id)
    }
}

/// Source-observable state of the wrapped `JRadioButton`.
#[derive(Clone, Debug, PartialEq)]
pub struct JRadioButtonBoundary {
    pub selected: bool,
    pub text: String,
    pub name: Option<String>,
    pub action_command: Option<String>,
    pub visible: bool,
    pub enabled: bool,
    pub focusable: bool,
    pub foreground: Option<Color>,
    pub tooltip: Option<String>,
    pub border_painted: bool,
    pub alignment_x: f32,
    pub preferred_size: Option<(i32, i32)>,
}

impl JRadioButtonBoundary {
    fn new(text: String) -> Self {
        Self {
            selected: false,
            text,
            name: None,
            action_command: None,
            visible: true,
            enabled: true,
            focusable: false,
            foreground: Some(BLACK),
            tooltip: None,
            border_painted: true,
            alignment_x: 0.5,
            preferred_size: None,
        }
    }
}

/// Java `RadioButton` with direct Swing operations represented as boundary state.
#[derive(Clone, Debug)]
pub struct RadioButton {
    pub radio_button: JRadioButtonBoundary,
    pub enumerated_type: Option<EnumeratedTypeBoundary>,
    pub group: Option<Rc<RefCell<RadioButtonGroup>>>,
    group_id: Option<usize>,
    debug: bool,
    orig_foreground: Option<Color>,
    directive_def: Option<String>,
    backup: Option<BooleanFieldSetting>,
    checkpoint: Option<BooleanFieldSetting>,
    default_value: Option<BooleanFieldSetting>,
    field_highlight: Option<BooleanFieldSetting>,
    selected_string_value: Option<String>,
    unformatted_tooltip: Option<String>,
    enabled: bool,
    editable: bool,
    locked: bool,
    item_listener_count: usize,
    action_listener_count: usize,
    change_listener_count: usize,
    selected_message_count: usize,
}

impl RadioButton {
    /// Java `RadioButton(String)`.
    pub fn new(text: impl Into<String>) -> Self {
        Self::new_full(Some(text.into()), None, None, None)
    }
    /// Java `RadioButton(String, String)`.
    pub fn new_with_tf_label(text: impl Into<String>, tf_label: impl Into<String>) -> Self {
        Self::new_full(Some(text.into()), Some(tf_label.into()), None, None)
    }
    /// Java `RadioButton(String, ButtonGroup)`.
    pub fn new_in_group(text: impl Into<String>, group: Rc<RefCell<RadioButtonGroup>>) -> Self {
        Self::new_full(Some(text.into()), None, Some(group), None)
    }
    /// Java enumerated-type constructors after extracting the interface boundary.
    pub fn new_with_enumerated_type(
        text: Option<String>,
        enumerated_type: EnumeratedTypeBoundary,
        group: Option<Rc<RefCell<RadioButtonGroup>>>,
    ) -> Self {
        let text = text.or_else(|| Some(enumerated_type.label.clone()));
        Self::new_full(text, None, group, Some(enumerated_type))
    }
    /// Shared body of all Java constructors; model construction is a Swing boundary.
    pub fn new_full(
        text: Option<String>,
        name_reference: Option<String>,
        group: Option<Rc<RefCell<RadioButtonGroup>>>,
        enumerated_type: Option<EnumeratedTypeBoundary>,
    ) -> Self {
        let text = text
            .or_else(|| enumerated_type.as_ref().map(|value| value.label.clone()))
            .unwrap_or_default();
        let group_id = group.as_ref().map(|group| group.borrow_mut().add());
        let mut result = Self {
            radio_button: JRadioButtonBoundary::new(text.clone()),
            enumerated_type,
            group,
            group_id,
            debug: false,
            orig_foreground: None,
            directive_def: None,
            backup: None,
            checkpoint: None,
            default_value: None,
            field_highlight: None,
            selected_string_value: None,
            unformatted_tooltip: None,
            enabled: true,
            editable: true,
            locked: false,
            item_listener_count: 0,
            action_listener_count: 0,
            change_listener_count: 0,
            selected_message_count: 0,
        };
        result.set_name(name_reference.as_deref().unwrap_or(&text));
        if let Some(enumerated_type) = result.enumerated_type.clone() {
            if enumerated_type.default {
                result.set_selected(true);
            }
            result.selected_string_value = enumerated_type.value;
        }
        result
    }
    pub fn get_field(&self) -> &Self {
        self
    }
    pub fn is_boolean(&self) -> bool {
        true
    }
    pub fn is_debug(&self) -> bool {
        self.debug || ARGUMENTS.lock().unwrap().is_debug()
    }
    pub fn is_text(&self) -> bool {
        false
    }
    pub fn is_visible(&self) -> bool {
        self.radio_button.visible
    }
    pub fn equals_selected_string_value(&self, value: Option<&str>) -> bool {
        self.selected_string_value.as_deref().map_or_else(
            || value.is_some_and(|value| !value.is_empty()),
            |selected| Some(selected) == value,
        )
    }
    pub fn do_click(&mut self) {
        self.set_selected(true);
        self.item_state_changed();
    }
    pub fn checkpoint(&mut self) {
        let selected = self.is_selected();
        self.checkpoint.get_or_insert_default().set(selected);
    }
    pub fn checkpoint_value(&mut self, value: bool) {
        self.checkpoint.get_or_insert_default().set(value);
    }
    pub fn get_checkpoint(&self) -> Option<&BooleanFieldSetting> {
        self.checkpoint.as_ref()
    }
    pub fn set_checkpoint(&mut self, input: Option<&BooleanFieldSetting>) {
        if self.checkpoint.is_none()
            && input.is_some_and(|value| value.is_set() && value.is_boolean())
        {
            self.checkpoint = Some(BooleanFieldSetting::default());
        }
        if let Some(checkpoint) = &mut self.checkpoint {
            checkpoint.copy(input);
        }
    }
    pub fn backup(&mut self) {
        let selected = self.is_selected();
        self.backup.get_or_insert_default().set(selected);
    }
    pub fn restore_from_backup(&mut self) {
        let value = self
            .backup
            .as_ref()
            .filter(|backup| backup.is_set())
            .map(BooleanFieldSetting::is_value);
        if let Some(value) = value {
            self.set_selected(value);
            self.backup.as_mut().unwrap().reset();
        }
    }
    pub fn set_value(&mut self, input: Option<&Self>) {
        if let Some(input) = input {
            self.set_selected(input.is_selected());
        }
    }
    pub fn set_value_string(&mut self, _value: Option<&str>) {}
    pub fn set_value_boolean(&mut self, value: bool) {
        self.set_selected(value);
    }
    /// Java `clear`: a selected `ButtonGroup` button cannot be cleared.
    pub fn clear(&mut self) {}
    pub fn set_directive_def(&mut self, directive_def: Option<&str>) {
        self.directive_def = directive_def.map(str::to_owned);
    }
    pub fn get_directive_def(&self) -> Option<&str> {
        self.directive_def.as_deref()
    }
    /// Java autodoc default lookup is supplied at the storage boundary.
    pub fn use_default_value(&mut self, autodoc_default: Option<&str>) {
        if self.directive_def.is_none() {
            if let Some(default_value) = &mut self.default_value
                && default_value.is_set()
            {
                default_value.reset();
            }
            return;
        }
        if self.default_value.is_none() {
            let mut value = BooleanFieldSetting::default();
            if let Some(default_value) = autodoc_default {
                value.set_string(Some(default_value));
            }
            self.default_value = Some(value);
        }
        if let Some(default_value) = &self.default_value
            && default_value.is_set()
        {
            self.set_selected(default_value.is_value());
        }
    }
    pub fn equals_default_value(&self) -> bool {
        self.default_value
            .as_ref()
            .is_some_and(|value| value.equals(self.is_selected()))
    }
    pub fn equals_default_value_string(&self, value: Option<&str>) -> bool {
        self.default_value.as_ref().is_some_and(|setting| {
            setting.equals(value.is_some_and(|value| !value.trim().is_empty()))
        })
    }
    pub fn equals_default_value_boolean(&self, input: bool) -> bool {
        self.default_value
            .as_ref()
            .is_some_and(|value| value.equals(input))
    }
    pub fn is_checkpoint_value(&self) -> bool {
        self.checkpoint
            .as_ref()
            .is_some_and(BooleanFieldSetting::is_value)
    }
    pub fn is_different_from_checkpoint(&self, always_check: bool) -> bool {
        if !always_check && (!self.is_enabled() || !self.is_visible()) {
            return false;
        }
        self.checkpoint
            .as_ref()
            .is_none_or(|checkpoint| !checkpoint.equals(self.is_selected()))
    }
    pub fn set_text(&mut self, text: impl Into<String>) {
        self.radio_button.text = text.into();
        self.set_name(&self.radio_button.text.clone());
    }
    pub fn get_text(&self) -> &str {
        &self.radio_button.text
    }
    pub fn get_description(&self) -> Option<String> {
        self.get_quoted_label()
    }
    pub fn get_quoted_label(&self) -> Option<String> {
        utilities::quote_label(Some(self.get_text()))
    }
    pub fn set_border_painted(&mut self, painted: bool) {
        self.radio_button.border_painted = painted;
    }
    /// Java `setBorder(Border)` is a native painting boundary; only presence is observable here.
    pub fn set_border(&mut self, _border_present: bool) {}
    pub fn set_foreground(&mut self, foreground: Color) {
        self.radio_button.foreground = Some(foreground);
    }
    pub fn set_name(&mut self, reference: &str) {
        if let Some(name) = utilities::convert_label_to_name(Some(reference), true) {
            self.radio_button.name = Some(format!("rb{SEPARATOR_CHAR}{name}"));
            if ARGUMENTS.lock().unwrap().is_print_names() {
                println!(
                    "{} {DEFAULT_DELIMITER} ",
                    self.radio_button.name.as_deref().unwrap_or_default()
                );
            }
        }
    }
    pub fn get_name(&self) -> Option<&str> {
        self.radio_button.name.as_deref()
    }
    pub fn equals_enumerated_type(&self, value: Option<&EnumeratedTypeBoundary>) -> bool {
        self.enumerated_type.as_ref() == value
    }
    pub fn set_debug(&mut self, input: bool) {
        self.debug = input;
    }
    pub fn is_field_highlight_set(&self) -> bool {
        self.field_highlight
            .as_ref()
            .is_some_and(BooleanFieldSetting::is_set)
    }
    pub fn set_field_highlight_setting(&mut self, input: Option<&BooleanFieldSetting>) {
        if self.field_highlight.is_none()
            && input.is_some_and(|value| value.is_set() && value.is_boolean())
        {
            self.field_highlight = Some(BooleanFieldSetting::default());
            self.add_field_highlight_action_listeners();
        }
        if let Some(field_highlight) = &mut self.field_highlight {
            field_highlight.copy(input);
            self.update_field_highlight(self.is_selected());
        }
    }
    /// Java group enumeration/listener registrations are represented by counts.
    fn add_field_highlight_action_listeners(&mut self) {
        self.item_listener_count = if self.group.is_some() {
            1
        } else {
            self.item_listener_count.max(1)
        };
    }
    pub fn set_field_highlight(&mut self, value: bool) {
        if self.field_highlight.is_none() {
            self.field_highlight = Some(BooleanFieldSetting::default());
            self.add_field_highlight_action_listeners();
        }
        self.field_highlight.as_mut().unwrap().set(value);
        self.update_field_highlight(self.is_selected());
    }
    pub fn set_field_highlight_string(&mut self, _value: Option<&str>) {}
    pub fn clear_field_highlight(&mut self) {
        if let Some(field_highlight) = &mut self.field_highlight
            && field_highlight.is_set()
        {
            field_highlight.reset();
            self.update_field_highlight(false);
        }
    }
    pub fn get_field_highlight(&self) -> Option<&BooleanFieldSetting> {
        self.field_highlight.as_ref()
    }
    pub fn equals_field_highlight(&self) -> bool {
        self.field_highlight
            .as_ref()
            .is_some_and(|value| value.is_set() && value.equals(self.is_selected()))
    }
    pub fn equals_field_highlight_string(&self, value: Option<&str>) -> bool {
        self.field_highlight.as_ref().is_some_and(|setting| {
            setting.is_set() && setting.equals(value.is_some_and(|value| !value.trim().is_empty()))
        })
    }
    pub fn equals_field_highlight_boolean(&self, input: bool) -> bool {
        self.field_highlight
            .as_ref()
            .is_some_and(|value| value.is_set() && value.equals(input))
    }
    pub fn item_state_changed(&mut self) {
        self.update_field_highlight(self.is_selected());
    }
    pub fn update_field_highlight(&mut self, selected: bool) {
        if self
            .field_highlight
            .as_ref()
            .is_some_and(|value| value.is_set() && value.is_value() == selected)
        {
            if self.orig_foreground.is_none() {
                self.orig_foreground = Some(self.radio_button.foreground.unwrap_or(BLACK));
            }
            self.radio_button.foreground = Some(FIELD_HIGHLIGHT);
        } else if let Some(foreground) = self.orig_foreground {
            self.radio_button.foreground = Some(foreground);
        }
    }
    pub fn set_tool_tip_text(&mut self, text: Option<&str>) {
        self.radio_button.tooltip = text.map(|text| format!("<html>{text}"));
    }
    pub fn set_unformatted_tooltip(&mut self, text: Option<&str>) -> Option<&str> {
        self.unformatted_tooltip = text.map(str::to_owned);
        self.unformatted_tooltip.as_deref()
    }
    pub fn has_unformatted_tooltip(&self) -> bool {
        self.unformatted_tooltip.is_some()
    }
    pub fn use_unformatted_tooltip(
        &mut self,
        param_descr: Option<&str>,
        directive_descr: Option<&str>,
    ) {
        let pieces = [
            self.unformatted_tooltip.as_deref(),
            param_descr,
            directive_descr,
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .join(" ");
        self.set_tool_tip_text((!pieces.is_empty()).then_some(pieces.as_str()));
        self.unformatted_tooltip = None;
    }
    pub fn set_tooltip(&mut self, field_tooltip: Option<&str>) {
        self.radio_button.tooltip = field_tooltip.map(str::to_owned);
    }
    pub fn get_tooltip(&self) -> Option<&str> {
        self.radio_button.tooltip.as_deref()
    }
    pub fn set_preformatted_tooltip(&mut self, tooltip: Option<&str>) {
        self.radio_button.tooltip = tooltip.map(str::to_owned);
    }
    pub fn add_tooltip(&mut self, text: Option<&str>) {
        let Some(text) = text else {
            return;
        };
        self.radio_button.tooltip = Some(match self.radio_button.tooltip.take() {
            Some(tooltip) => format!("{tooltip} & <html>{text}"),
            None => format!("<html>{text}"),
        });
    }
    pub fn set_visible(&mut self, visible: bool) {
        self.radio_button.visible = visible;
    }
    pub fn add_action_listener(&mut self) {
        self.action_listener_count += 1;
    }
    pub fn remove_action_listener(&mut self) {
        self.action_listener_count = self.action_listener_count.saturating_sub(1);
    }
    pub fn add_change_listener(&mut self) {
        self.change_listener_count += 1;
    }
    pub fn set_selected(&mut self, selected: bool) {
        self.radio_button.selected = selected;
        if let (Some(group), Some(id)) = (&self.group, self.group_id) {
            group.borrow_mut().set_selected(id, selected);
        }
        self.msg_selected();
        if self.field_highlight.is_some() {
            self.update_field_highlight(self.is_selected());
        }
    }
    pub fn is_selected(&self) -> bool {
        match (&self.group, self.group_id) {
            (Some(group), Some(id)) => group.borrow().is_selected(id, self.radio_button.selected),
            _ => self.radio_button.selected,
        }
    }
    pub fn msg_selected(&mut self) {
        self.selected_message_count += 1;
    }
    pub fn is_empty(&self) -> bool {
        false
    }
    pub fn set_preferred_size(&mut self, preferred_size: Option<(i32, i32)>) {
        self.radio_button.preferred_size = preferred_size;
    }
    pub fn is_required(&self) -> bool {
        false
    }
    pub fn get_enumerated_type(&self) -> Option<&EnumeratedTypeBoundary> {
        self.enumerated_type.as_ref()
    }
    pub fn get_ui_component(&self) -> &Self {
        self
    }
    pub fn get_component(&self) -> &JRadioButtonBoundary {
        &self.radio_button
    }
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        self.radio_button.enabled = enabled && self.editable && !self.locked;
        if self.radio_button.enabled {
            self.update_field_highlight(self.is_selected());
        }
    }
    pub fn set_locked(&mut self, locked: bool) {
        self.locked = locked;
        self.radio_button.enabled = self.enabled && self.editable && !locked;
        if self.radio_button.enabled {
            self.update_field_highlight(self.is_selected());
        }
    }
    pub fn set_editable(&mut self, editable: bool) {
        self.editable = editable;
        self.radio_button.enabled = self.enabled && editable && !self.locked;
        if self.radio_button.enabled {
            self.update_field_highlight(self.is_selected());
        }
    }
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
    pub fn is_editable(&self) -> bool {
        self.editable
    }
    pub fn is_locked(&self) -> bool {
        self.locked
    }
    pub fn get_action_command(&self) -> &str {
        self.radio_button
            .action_command
            .as_deref()
            .unwrap_or(self.get_text())
    }
    pub fn set_action_command(&mut self, action_command: Option<&str>) {
        self.radio_button.action_command = action_command.map(str::to_owned);
    }
    pub fn set_alignment_x(&mut self, alignment_x: f32) {
        self.radio_button.alignment_x = alignment_x;
    }
    pub fn get_selected_objects(&self) -> Option<Vec<String>> {
        self.is_selected().then(|| vec![self.get_text().to_owned()])
    }
}

/// `RadioButton` is the other Java `ButtonComponent` implementation.
impl ButtonComponent for RadioButton {
    fn add_action_listener(&mut self, _listener: &mut dyn ActionListenerBoundary) {
        RadioButton::add_action_listener(self);
    }

    fn is_selected(&self) -> bool {
        RadioButton::is_selected(self)
    }

    fn get_action_command(&self) -> Option<&str> {
        Some(RadioButton::get_action_command(self))
    }

    fn is_enabled(&self) -> bool {
        RadioButton::is_enabled(self)
    }
}

impl std::fmt::Display for RadioButton {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "{}: {}",
            self.get_text(),
            if self.is_selected() { "On" } else { "Off" }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_uitest_name_and_group_selection_are_preserved() {
        let group = Rc::new(RefCell::new(RadioButtonGroup::new()));
        let mut one = RadioButton::new_in_group("One:", group.clone());
        let mut two = RadioButton::new_in_group("Two:", group);
        assert_eq!(one.get_name(), Some("rb.one"));
        one.set_selected(true);
        two.set_selected(true);
        assert!(!one.is_selected());
        assert!(two.is_selected());
    }

    #[test]
    fn checkpoint_and_highlight_follow_source_visibility_and_lock_rules() {
        let mut button = RadioButton::new("Choice");
        button.checkpoint();
        button.set_selected(true);
        assert!(button.is_different_from_checkpoint(false));
        button.set_visible(false);
        assert!(!button.is_different_from_checkpoint(false));
        button.set_visible(true);
        button.set_field_highlight(true);
        assert_eq!(button.radio_button.foreground, Some(FIELD_HIGHLIGHT));
        button.set_locked(true);
        assert!(!button.radio_button.enabled);
    }

    #[test]
    fn default_and_tooltip_boundaries_preserve_values() {
        let mut button = RadioButton::new("Choice");
        button.set_directive_def(Some("flag"));
        button.use_default_value(Some("true"));
        assert!(button.is_selected());
        assert!(button.equals_default_value());
        button.set_unformatted_tooltip(Some("Tip"));
        button.use_unformatted_tooltip(Some("parameter"), None);
        assert!(button.get_tooltip().is_some());
        assert!(!button.has_unformatted_tooltip());
    }
}
