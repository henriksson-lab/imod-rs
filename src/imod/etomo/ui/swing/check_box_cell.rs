//! `IMOD/Etomo/src/etomo/ui/swing/CheckBoxCell.java`.
//!
//! `JCheckBox`, its border, listener dispatch, colours, and native geometry are
//! retained as an explicit Swing boundary.  The source-owned CheckBoxCell state
//! and decisions are represented here without pretending to render Swing.
#![allow(dead_code)]

use std::sync::atomic::{AtomicUsize, Ordering};

use crate::imod::etomo::etomo_director::ARGUMENTS;
use crate::imod::etomo::storage::autodoc::autodoc_tokenizer::{DEFAULT_DELIMITER, SEPARATOR_CHAR};
use crate::imod::etomo::r#type::const_etomo_number::ConstEtomoNumber;
use crate::imod::etomo::util::utilities;

use super::check_box::{BLACK, BooleanFieldSetting, Color, FIELD_HIGHLIGHT, GRAY};
use super::field_lock_controller::{FieldLockController, JToggleButton};

static NEXT_CHECK_BOX_CELL_IDENTITY_HASH_CODE: AtomicUsize = AtomicUsize::new(1);
const CHECK_BOX: &str = "CheckBox";

/// Source-observable state of Java `JCheckBox`; rendering is a Swing boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CheckBoxCellJCheckBox {
    pub selected: bool,
    pub text: String,
    pub name: Option<String>,
    pub action_command: Option<String>,
    pub visible: bool,
    pub enabled: bool,
    pub foreground: Color,
    pub background: Option<Color>,
    pub tooltip: Option<String>,
    pub border_painted: bool,
    pub border: &'static str,
    pub width: i32,
    pub height: i32,
    pub border_bottom: i32,
    pub border_left: i32,
    pub action_listener_count: usize,
    pub change_listener_count: usize,
}

impl Default for CheckBoxCellJCheckBox {
    fn default() -> Self {
        Self {
            selected: false,
            text: String::new(),
            name: None,
            action_command: None,
            visible: true,
            enabled: true,
            foreground: BLACK,
            background: None,
            tooltip: None,
            border_painted: false,
            border: "EtchedBorder",
            width: 0,
            height: 0,
            border_bottom: 0,
            border_left: 0,
            action_listener_count: 0,
            change_listener_count: 0,
        }
    }
}

/// Java package-private final `CheckBoxCell`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CheckBoxCell {
    pub check_box: CheckBoxCellJCheckBox,
    pub field_lock_controller: FieldLockController,
    /// Java `ToggleCoordinator`; its target-object dispatch remains a GUI boundary.
    pub toggle_coordinator_targets: Option<Vec<usize>>,
    pub header_background: bool,
    pub debug: bool,
    pub unformatted_label: String,
    pub checkpoint: Option<BooleanFieldSetting>,
    pub backup_value: bool,
    pub field_is_backed_up: bool,
    pub field_highlight: Option<BooleanFieldSetting>,
    pub directive_def: Option<String>,
    pub selected_string_value: Option<String>,
    pub unformatted_tooltip: Option<String>,
    pub identity_hash_code: usize,
    pub background_refresh_count: usize,
}

impl CheckBoxCell {
    /// Java private `CheckBoxCell(String, boolean, boolean)`.
    pub fn new(header_label: Option<&str>, header_background: bool, debug: bool) -> Self {
        let mut value = Self {
            check_box: CheckBoxCellJCheckBox {
                border_painted: true,
                ..Default::default()
            },
            field_lock_controller: FieldLockController::get_toggle_button_debug_instance(
                JToggleButton {
                    enabled: true,
                    selected: false,
                },
                debug,
            ),
            toggle_coordinator_targets: header_background.then(Vec::new),
            header_background,
            debug,
            unformatted_label: String::new(),
            checkpoint: None,
            backup_value: false,
            field_is_backed_up: false,
            field_highlight: None,
            directive_def: None,
            selected_string_value: None,
            unformatted_tooltip: None,
            identity_hash_code: NEXT_CHECK_BOX_CELL_IDENTITY_HASH_CODE
                .fetch_add(1, Ordering::Relaxed),
            background_refresh_count: 0,
        };
        value.set_background();
        value.set_foreground();
        if let Some(header_label) = header_label {
            value.set_name(header_label);
        }
        value
    }
    pub fn get_instance() -> Self {
        Self::new(None, false, false)
    }
    pub fn get_header_background_named_instance(
        header_label1: Option<&str>,
        header_label2: Option<&str>,
    ) -> Self {
        let label = utilities::concatenate(header_label1, Some(" "), header_label2, None);
        Self::new(label.as_deref(), true, false)
    }
    pub fn get_named_instance(header_label: Option<&str>) -> Self {
        Self::new(header_label, false, false)
    }
    pub fn get_named_debug_instance(header_label: Option<&str>, debug: bool) -> Self {
        Self::new(header_label, false, debug)
    }
    pub fn get_named_three_instance(
        header_label1: Option<&str>,
        header_label2: Option<&str>,
        header_label3: Option<&str>,
    ) -> Self {
        let label = utilities::concatenate(header_label1, header_label2, header_label3, Some(" "));
        Self::new(label.as_deref(), false, false)
    }
    pub fn add_target(&mut self, target: &Self) {
        if let Some(targets) = &mut self.toggle_coordinator_targets {
            targets.push(target.identity_hash_code);
        }
    }
    pub fn delete_target(&mut self, target: &Self) {
        if let Some(targets) = &mut self.toggle_coordinator_targets {
            targets.retain(|id| *id != target.identity_hash_code);
        }
    }
    pub fn set_name_three(
        &mut self,
        reference1: Option<&str>,
        reference2: Option<&str>,
        reference3: Option<&str>,
    ) {
        if let Some(reference) =
            utilities::concatenate(reference1, reference2, reference3, Some(" "))
        {
            self.set_name(&reference);
        }
    }
    fn set_name(&mut self, reference: &str) {
        if let Some(name) = utilities::convert_label_to_name(Some(reference), true) {
            self.check_box.name = Some(format!("{CHECK_BOX}{SEPARATOR_CHAR}{name}"));
            if ARGUMENTS.lock().unwrap().is_print_names() {
                println!(
                    "{} {DEFAULT_DELIMITER} ",
                    self.check_box.name.as_deref().unwrap()
                );
            }
        }
    }
    pub fn is_debug(&self) -> bool {
        ARGUMENTS.lock().unwrap().is_debug()
    }
    pub fn get_unique_action_command(&self) -> String {
        format!("etomo.ui.swing.CheckBoxCell@{:x}", self.identity_hash_code)
    }
    pub fn get_name(&self) -> Option<&str> {
        self.check_box.name.as_deref()
    }
    pub fn get_component(&self) -> &CheckBoxCellJCheckBox {
        &self.check_box
    }
    pub fn get_ui_component(&self) -> &Self {
        self
    }
    pub fn get_field_type(&self) -> &'static str {
        CHECK_BOX
    }
    pub fn is_text(&self) -> bool {
        false
    }
    pub fn is_boolean(&self) -> bool {
        true
    }
    pub fn is_empty(&self) -> bool {
        false
    }
    pub fn set_selected_string_value(&mut self, value: Option<&str>) {
        self.selected_string_value = value.map(str::to_owned);
    }
    pub fn equals_selected_string_value(&self, value: Option<&str>) -> bool {
        match &self.selected_string_value {
            None => value.is_some_and(|x| !x.is_empty()),
            Some(selected) => value == Some(selected),
        }
    }
    pub fn is_different_from_checkpoint(&self, always_check: bool) -> bool {
        if !always_check && (!self.is_enabled() || !self.check_box.visible) {
            return false;
        }
        self.checkpoint
            .as_ref()
            .is_none_or(|checkpoint| !checkpoint.equals(self.is_selected()))
    }
    pub fn backup(&mut self) {
        self.backup_value = self.is_selected();
        self.field_is_backed_up = true;
    }
    pub fn restore_from_backup(&mut self) {
        if self.field_is_backed_up {
            self.set_selected(self.backup_value);
            self.field_is_backed_up = false;
        }
    }
    pub fn clear(&mut self) {
        self.set_selected(false);
    }
    pub fn checkpoint(&mut self) {
        let selected = self.is_selected();
        self.checkpoint.get_or_insert_default().set(selected);
    }
    pub fn set_checkpoint(&mut self, input: Option<&BooleanFieldSetting>) {
        if self.checkpoint.is_none() && input.is_some_and(|x| x.is_set() && x.is_boolean()) {
            self.checkpoint = Some(BooleanFieldSetting::default());
        }
        if let Some(checkpoint) = &mut self.checkpoint {
            checkpoint.copy(input);
        }
    }
    pub fn get_checkpoint(&self) -> Option<&BooleanFieldSetting> {
        self.checkpoint.as_ref()
    }
    pub fn set_locked(&mut self, locked: bool) {
        if self.field_lock_controller.set_locked(locked) {
            self.check_box.enabled = self
                .field_lock_controller
                .toggle_button
                .as_ref()
                .unwrap()
                .enabled;
            self.background_refresh_count += 1;
            self.set_background();
        }
    }
    pub fn set_editable(&mut self, editable: bool) {
        if self.field_lock_controller.set_editable(editable) {
            self.check_box.enabled = self
                .field_lock_controller
                .toggle_button
                .as_ref()
                .unwrap()
                .enabled;
            self.background_refresh_count += 1;
            self.set_background();
        }
    }
    pub fn set_enabled(&mut self, enabled: bool) {
        if self.field_lock_controller.set_enabled(enabled) {
            self.check_box.enabled = self
                .field_lock_controller
                .toggle_button
                .as_ref()
                .unwrap()
                .enabled;
            self.background_refresh_count += 1;
            self.set_background();
        }
    }
    pub fn is_locked(&self) -> bool {
        self.field_lock_controller.is_locked()
    }
    pub fn is_editable(&self) -> bool {
        self.field_lock_controller.is_editable()
    }
    pub fn is_enabled(&self) -> bool {
        self.field_lock_controller.is_enabled()
    }
    pub fn set_label(&mut self, label: &str) {
        self.unformatted_label = label.to_owned();
        self.set_foreground();
    }
    pub fn get_label(&self) -> &str {
        &self.unformatted_label
    }
    pub fn is_required(&self) -> bool {
        false
    }
    pub fn get_text(&self, _do_validation: bool) -> &str {
        &self.unformatted_label
    }
    pub fn get_text_two_displayers(&self, _do_validation: bool) -> &str {
        &self.unformatted_label
    }
    pub fn get_text_unvalidated(&self) -> &str {
        &self.unformatted_label
    }
    pub fn get_description(&self) -> Option<String> {
        self.get_quoted_label()
    }
    pub fn get_quoted_label(&self) -> Option<String> {
        utilities::quote_label(Some(&self.unformatted_label))
    }
    pub fn set_value_string(&mut self, value: Option<&str>) {
        self.set_selected(BooleanFieldSetting::string_to_boolean(value));
    }
    pub fn is_selected(&self) -> bool {
        self.check_box.selected
    }
    pub fn set_selected(&mut self, selected: bool) {
        self.check_box.selected = selected;
        self.field_lock_controller
            .toggle_button
            .as_mut()
            .unwrap()
            .selected = selected;
    }
    pub fn set_selected_number(&mut self, selected: Option<&ConstEtomoNumber>, allow_empty: bool) {
        if let Some(selected) = selected.filter(|x| !x.is_null()) {
            self.set_selected(selected.is());
        }
        if allow_empty {
            self.set_selected(false);
        }
    }
    pub fn set_value_boolean(&mut self, selected: bool) {
        self.set_selected(selected);
    }
    pub fn set_value_field(&mut self, selected: Option<bool>) {
        self.set_selected(selected.unwrap_or(false));
    }
    pub fn set_action_command(&mut self, input: Option<&str>) {
        self.check_box.action_command = input.map(str::to_owned);
    }
    pub fn get_action_command(&self) -> Option<&str> {
        self.check_box.action_command.as_deref()
    }
    pub fn add_action_listener(&mut self) {
        self.check_box.action_listener_count += 1;
    }
    pub fn add_change_listener(&mut self) {
        self.check_box.change_listener_count += 1;
    }
    fn set_background(&mut self) {
        self.check_box.background = self.header_background.then_some(GRAY);
    }
    fn set_html_label(&mut self, color: Color) {
        self.check_box.text = format!(
            "<html><P style=\"font-weight:normal; color:rgb({},{},{})\">{}</style>",
            color.0, color.1, color.2, self.unformatted_label
        );
    }
    fn set_foreground(&mut self) {
        self.check_box.foreground = BLACK;
        self.set_html_label(BLACK);
    }
    pub fn set_directive_def(&mut self, directive_def: Option<&str>) {
        self.directive_def = directive_def.map(str::to_owned);
    }
    pub fn get_directive_def(&self) -> Option<&str> {
        self.directive_def.as_deref()
    }
    pub fn is_field_highlight_set(&self) -> bool {
        self.field_highlight
            .as_ref()
            .is_some_and(BooleanFieldSetting::is_set)
    }
    pub fn set_field_highlight_boolean(&mut self, value: bool) {
        if self.field_highlight.is_none() {
            self.field_highlight = Some(BooleanFieldSetting::default());
            self.add_action_listener();
        }
        self.field_highlight.as_mut().unwrap().set(value);
        self.update_field_highlight();
    }
    pub fn set_field_highlight_string(&mut self, _value: Option<&str>) {}
    pub fn set_field_highlight_setting(&mut self, input: Option<&BooleanFieldSetting>) {
        if self.field_highlight.is_none() && input.is_some_and(|x| x.is_set() && x.is_boolean()) {
            self.field_highlight = Some(BooleanFieldSetting::default());
            self.add_action_listener();
        }
        if let Some(highlight) = &mut self.field_highlight {
            highlight.copy(input);
            self.update_field_highlight();
        }
    }
    pub fn equals_field_highlight(&self) -> bool {
        self.field_highlight
            .as_ref()
            .is_some_and(|x| x.is_set() && x.equals(self.is_selected()))
    }
    pub fn equals_field_highlight_string(&self, value: Option<&str>) -> bool {
        self.field_highlight
            .as_ref()
            .is_some_and(|x| x.is_set() && x.equals(value.is_some_and(|v| !v.trim().is_empty())))
    }
    pub fn clear_field_highlight(&mut self) {
        if self
            .field_highlight
            .as_ref()
            .is_some_and(BooleanFieldSetting::is_set)
        {
            self.field_highlight.as_mut().unwrap().reset();
            self.update_field_highlight();
        }
    }
    pub fn get_field_highlight(&self) -> Option<&BooleanFieldSetting> {
        self.field_highlight.as_ref()
    }
    pub fn action_performed(&mut self) {
        self.update_field_highlight();
        self.field_lock_controller
            .apply_toggle_button_selection_state();
        self.check_box.enabled = self
            .field_lock_controller
            .toggle_button
            .as_ref()
            .unwrap()
            .enabled;
    }
    pub fn update_field_highlight(&mut self) {
        if let Some(highlight) = &self.field_highlight {
            if highlight.is_set() {
                self.check_box.foreground = if highlight.is_value() == self.is_selected() {
                    FIELD_HIGHLIGHT
                } else {
                    BLACK
                };
            }
        }
    }
    pub fn use_default_value(&self) {
        eprintln!("Warning: CheckBoxCell.useDefaultValue has not been implemented");
    }
    pub fn equals_default_value(&self) -> bool {
        false
    }
    pub fn equals_default_value_string(&self, _value: Option<&str>) -> bool {
        false
    }
    pub fn get_height(&self) -> i32 {
        self.check_box.height + self.check_box.border_bottom - 1
    }
    pub fn get_width(&self) -> i32 {
        self.check_box.width
    }
    pub fn get_left_border(&self) -> i32 {
        self.check_box.border_left
    }
    pub fn set_tool_tip_text(&mut self, text: Option<&str>) {
        self.check_box.tooltip = text.map(str::to_owned);
    }
    pub fn set_tooltip_field(&mut self, tooltip: Option<&str>) {
        if let Some(tooltip) = tooltip {
            self.check_box.tooltip = Some(tooltip.to_owned());
        }
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
        self.check_box.tooltip = Some(
            [
                self.unformatted_tooltip.as_deref(),
                param_descr,
                directive_descr,
            ]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>()
            .join(" "),
        );
        self.unformatted_tooltip = None;
    }
    pub fn get_tooltip(&self) -> Option<&str> {
        self.check_box.tooltip.as_deref()
    }
    pub fn add_tooltip(&mut self, text: Option<&str>) {
        let Some(text) = text else { return };
        if let Some(tooltip) = &mut self.check_box.tooltip {
            tooltip.push_str(" & ");
            tooltip.push_str(text);
        } else {
            self.set_tool_tip_text(Some(text));
        }
    }
}

impl std::fmt::Display for CheckBoxCell {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[name:{},unformattedLabel:{},selected:{}]",
            self.get_name().unwrap_or("null"),
            self.unformatted_label,
            self.is_selected()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn factories_name_checkbox_and_header_coordinator() {
        let header = CheckBoxCell::get_header_background_named_instance(Some("A"), Some("B"));
        let named = CheckBoxCell::get_named_instance(Some("Use item"));
        assert!(header.toggle_coordinator_targets.is_some());
        assert_eq!(header.get_name(), Some("CheckBox.a-b"));
        assert_eq!(named.get_name(), Some("CheckBox.use-item"));
        assert!(named.check_box.border_painted);
    }
    #[test]
    fn checkpoint_backup_and_empty_override_follow_source() {
        let mut cell = CheckBoxCell::get_instance();
        cell.set_selected(true);
        cell.backup();
        cell.clear();
        assert!(cell.is_different_from_checkpoint(true));
        cell.restore_from_backup();
        assert!(cell.is_selected());
        cell.checkpoint();
        assert!(!cell.is_different_from_checkpoint(true));
        cell.set_selected(false);
        assert!(cell.is_different_from_checkpoint(true));
    }
    #[test]
    fn lock_and_highlight_follow_toggle_state() {
        let mut cell = CheckBoxCell::get_instance();
        cell.set_field_highlight_boolean(true);
        cell.set_selected(true);
        assert!(
            cell.field_lock_controller
                .toggle_button
                .as_ref()
                .unwrap()
                .selected
        );
        cell.action_performed();
        assert_eq!(cell.check_box.foreground, FIELD_HIGHLIGHT);
        cell.set_locked(true);
        assert!(!cell.check_box.enabled);
        assert!(
            !cell
                .field_lock_controller
                .toggle_button
                .as_ref()
                .unwrap()
                .enabled
        );
        assert!(cell.is_locked());
        cell.clear_field_highlight();
        assert!(!cell.is_field_highlight_set());
    }
    #[test]
    fn tooltips_and_identity_command_are_source_shaped() {
        let mut cell = CheckBoxCell::get_instance();
        cell.set_unformatted_tooltip(Some("raw"));
        cell.use_unformatted_tooltip(Some("parameter"), Some("directive"));
        cell.add_tooltip(Some("more"));
        assert_eq!(cell.get_tooltip(), Some("raw parameter directive & more"));
        assert!(!cell.has_unformatted_tooltip());
        assert!(
            cell.get_unique_action_command()
                .starts_with("etomo.ui.swing.CheckBoxCell@")
        );
    }
}
