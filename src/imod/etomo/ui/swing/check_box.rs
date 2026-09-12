//! `IMOD/Etomo/src/etomo/ui/swing/CheckBox.java`.
//!
//! `JCheckBox`, its `ButtonModel`, painting, listeners, and component hierarchy
//! remain at the Swing boundary.  This unit retains all source-observable state
//! and the source's naming, two-label, checkpoint, highlight, tooltip, and flag
//! behaviour for the Rust GUI model.
#![allow(dead_code)]

use crate::imod::etomo::etomo_director::ARGUMENTS;
use crate::imod::etomo::storage::autodoc::autodoc_tokenizer::{DEFAULT_DELIMITER, SEPARATOR_CHAR};
use crate::imod::etomo::r#type::const_etomo_number::ConstEtomoNumber;
use crate::imod::etomo::ui::swing::button_component::{ActionListenerBoundary, ButtonComponent};
use crate::imod::etomo::util::utilities;

/// Java `java.awt.Color` values used by this source unit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Color(pub u8, pub u8, pub u8);

pub const FIELD_HIGHLIGHT: Color = Color(0, 0, 185);
pub const BLACK: Color = Color(0, 0, 0);
pub const GRAY: Color = Color(128, 128, 128);

/// Java `etomo.ui.FlagType`, limited to the flag type selected by this unit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FlagType {
    Warning,
}

/// Java `BooleanFieldSetting` as used by `CheckBox`.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct BooleanFieldSetting {
    set: bool,
    value: bool,
    text_setting: Option<String>,
}

impl BooleanFieldSetting {
    pub fn is_set(&self) -> bool {
        self.set
    }
    pub fn is_value(&self) -> bool {
        self.value
    }
    pub fn is_boolean(&self) -> bool {
        true
    }
    pub fn equals(&self, value: bool) -> bool {
        self.set && self.value == value
    }
    pub fn equals_string(&self, value: Option<&str>) -> bool {
        if !self.set {
            return false;
        }
        match &self.text_setting {
            Some(text) => value == Some(text),
            None => self.value == Self::string_to_boolean(value),
        }
    }
    pub fn set(&mut self, value: bool) {
        self.set = true;
        self.value = value;
        self.text_setting = None;
    }
    pub fn set_string(&mut self, value: Option<&str>) {
        self.set = true;
        self.value = Self::string_to_boolean(value);
        self.text_setting = value.map(str::to_owned);
    }
    pub fn reset(&mut self) {
        self.set = false;
        self.value = false;
        self.text_setting = None;
    }
    pub fn copy(&mut self, input: Option<&Self>) {
        *self = input.cloned().unwrap_or_default();
    }
    pub fn string_to_boolean(value: Option<&str>) -> bool {
        let Some(value) = value else {
            return false;
        };
        let value = value.trim();
        if value.is_empty() {
            return true;
        }
        if value.parse::<i32>().is_ok_and(|number| number == 0) {
            return false;
        }
        !["f", "false", "n", "na", "no", "off"]
            .iter()
            .any(|false_value| value.eq_ignore_ascii_case(false_value))
    }
}

/// Source-shaped state for the wrapped `JCheckBox`.
#[derive(Clone, Debug, PartialEq)]
pub struct JCheckBoxBoundary {
    pub selected: bool,
    pub text: Option<String>,
    pub name: Option<String>,
    pub action_command: Option<String>,
    pub visible: bool,
    pub enabled: bool,
    pub foreground: Option<Color>,
    pub background: Option<Color>,
    pub tooltip: Option<String>,
    pub alignment_x: f32,
}

impl Default for JCheckBoxBoundary {
    fn default() -> Self {
        Self {
            selected: false,
            text: None,
            name: None,
            action_command: None,
            visible: true,
            enabled: true,
            foreground: Some(BLACK),
            background: None,
            tooltip: None,
            alignment_x: 0.5,
        }
    }
}

/// Java `CheckBox` fields and methods.
#[derive(Clone, Debug)]
pub struct CheckBox {
    pub check_box: JCheckBoxBoundary,
    debug: bool,
    orig_foreground: Option<Color>,
    directive_def: Option<String>,
    backup: Option<BooleanFieldSetting>,
    default_value: Option<BooleanFieldSetting>,
    checkpoint: Option<BooleanFieldSetting>,
    field_highlight: Option<BooleanFieldSetting>,
    enabled: bool,
    editable: bool,
    selected_string_value: Option<String>,
    tooltip: Option<String>,
    alternate_tooltip: Option<String>,
    text: Option<String>,
    alternate_text: Option<String>,
    warning_enabled: bool,
    warning_value: bool,
    default_background: Option<Color>,
    flag_type: Option<FlagType>,
    unformatted_tooltip: Option<String>,
    label_when_false: Option<String>,
    label_when_true: Option<String>,
    action_command_when_false: Option<String>,
    action_command_when_true: Option<String>,
    action_listener_count: usize,
}

impl Default for CheckBox {
    fn default() -> Self {
        Self {
            check_box: JCheckBoxBoundary::default(),
            debug: false,
            orig_foreground: None,
            directive_def: None,
            backup: None,
            default_value: None,
            checkpoint: None,
            field_highlight: None,
            enabled: true,
            editable: true,
            selected_string_value: None,
            tooltip: None,
            alternate_tooltip: None,
            text: None,
            alternate_text: None,
            warning_enabled: false,
            warning_value: false,
            default_background: None,
            flag_type: None,
            unformatted_tooltip: None,
            label_when_false: None,
            label_when_true: None,
            action_command_when_false: None,
            action_command_when_true: None,
            action_listener_count: 0,
        }
    }
}

impl CheckBox {
    /// Java package-private `CheckBox()`.
    pub fn new() -> Self {
        Self::default()
    }
    /// Java `CheckBox(String)`.
    pub fn new_with_text(text: &str) -> Self {
        let mut value = Self::new();
        value.check_box.text = Some(text.into());
        value.set_name(text);
        value
    }
    /// Java `CheckBox(String, String)`; `None` represents Java null.
    pub fn new_with_texts(text_false: Option<&str>, text_true: Option<&str>) -> Self {
        let mut value = Self::new();
        match (text_false, text_true) {
            (None, None) => {}
            (Some(text), None) | (None, Some(text)) => {
                value.check_box.text = Some(text.into());
                value.set_name(text);
            }
            (Some(text_false), Some(text_true)) => {
                value.label_when_false = Some(text_false.into());
                value.label_when_true = Some(text_true.into());
                value.check_box.text = Some(text_true.into());
                value.action_command_when_true = value.get_action_command().map(str::to_owned);
                value.check_box.text = Some(text_false.into());
                value.action_command_when_false = value.get_action_command().map(str::to_owned);
                value.set_name(text_false);
                value.add_action_listener();
            }
        }
        value
    }
    /// Java `doClick(boolean)`: native button painting/synthetic pointer event boundary.
    pub fn do_click(&mut self, _allow_headless: bool) {
        self.check_box.selected = !self.check_box.selected;
        self.action_performed();
    }
    pub fn get_name(&self) -> Option<&str> {
        self.check_box.name.as_deref()
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
        self.check_box.visible
    }
    pub fn is_empty(&self) -> bool {
        false
    }
    pub fn equals_self(&self, object_is_this: bool) -> bool {
        object_is_this
    }
    pub fn equals_document(&self) -> bool {
        false
    }
    pub fn set_selected_string_value(&mut self, value: Option<&str>) {
        self.selected_string_value = value.map(str::to_owned);
    }
    pub fn equals_selected_string_value(&self, value: Option<&str>) -> bool {
        match &self.selected_string_value {
            None => value.is_some_and(|value| !value.is_empty()),
            Some(selected) => Some(selected.as_str()) == value,
        }
    }
    pub fn is_required(&self) -> bool {
        false
    }
    pub fn is_selected(&self) -> bool {
        self.check_box.selected
    }
    pub fn get_text(&self) -> Option<&str> {
        self.check_box.text.as_deref()
    }
    pub fn get_text_with_validation(&self, _do_validation: bool) -> Option<&str> {
        self.get_text()
    }
    pub fn get_text_with_two_displayers(&self, _do_validation: bool) -> Option<&str> {
        self.get_text()
    }
    pub fn get_description(&self) -> Option<String> {
        self.get_quoted_label()
    }
    pub fn get_quoted_label(&self) -> Option<String> {
        utilities::quote_label(self.get_text().or(self.get_name()))
    }
    pub fn set_text(&mut self, text: Option<&str>) {
        self.check_box.text = text.map(str::to_owned);
        if let Some(text) = text {
            self.set_name(text);
        }
        self.text = text.map(str::to_owned);
        self.label_when_false = None;
        self.label_when_true = None;
    }
    pub fn set_alternate_text(&mut self, text: Option<&str>) {
        self.alternate_text = text.map(str::to_owned);
    }
    pub fn switch_text(&mut self, alternate: bool) {
        self.check_box.text = if alternate && self.alternate_text.is_some() {
            self.alternate_text.clone()
        } else {
            self.text.clone()
        };
    }
    pub fn set_selected(&mut self, input: bool) {
        self.check_box.selected = input;
        self.update_two_label();
        self.update_field_highlight();
    }
    pub fn set_selected_number(&mut self, input: Option<&ConstEtomoNumber>, allow_empty: bool) {
        if allow_empty || input.is_some_and(|input| !input.is_null()) {
            self.set_selected(input.is_some_and(ConstEtomoNumber::is));
        }
    }
    pub fn set_name(&mut self, text: &str) {
        let name = utilities::convert_label_to_name(Some(text), true).unwrap_or_default();
        self.check_box.name = Some(format!("cb{SEPARATOR_CHAR}{name}"));
        if ARGUMENTS.lock().unwrap().is_print_names() {
            println!(
                "{} {DEFAULT_DELIMITER} ",
                self.check_box.name.as_deref().unwrap_or_default()
            );
        }
    }
    pub fn backup(&mut self) {
        let selected = self.is_selected();
        self.backup.get_or_insert_default().set(selected);
    }
    pub fn restore_from_backup(&mut self) {
        if let Some(backup) = &self.backup
            && backup.is_set()
        {
            self.set_selected(backup.is_value());
        }
    }
    pub fn set_action_command(&mut self, input: Option<&str>) {
        self.check_box.action_command = input.map(str::to_owned);
    }
    pub fn set_alignment_x(&mut self, input: f32) {
        self.check_box.alignment_x = input;
    }
    pub fn set_background(&mut self, color: Option<Color>) {
        self.check_box.background = color;
    }
    pub fn clear(&mut self) {
        self.set_selected(false);
    }
    pub fn set_value_checkbox(&mut self, input: Option<&Self>) {
        self.set_selected(input.is_some_and(Self::is_selected));
    }
    pub fn set_value_string(&mut self, _value: Option<&str>) {}
    pub fn set_visible(&mut self, input: bool) {
        self.check_box.visible = input;
    }
    pub fn set_value(&mut self, value: bool) {
        self.set_selected(value);
    }
    pub fn set_directive_def(&mut self, directive_def: Option<&str>) {
        self.directive_def = directive_def.map(str::to_owned);
        if let Some(default_value) = &mut self.default_value {
            default_value.reset();
        }
    }
    pub fn get_directive_def(&self) -> Option<&str> {
        self.directive_def.as_deref()
    }
    /// Java `useDefaultValue`; the autodoc lookup is an explicit storage boundary.
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
            if autodoc_default.is_some() {
                value.set_string(autodoc_default);
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
        self.default_value
            .as_ref()
            .is_some_and(|setting| setting.equals_string(value))
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
    pub fn get_ui_component(&self) -> &Self {
        self
    }
    pub fn get_component(&self) -> &JCheckBoxBoundary {
        &self.check_box
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
    pub fn reset_to_checkpoint(&mut self) {
        if let Some(checkpoint) = &self.checkpoint
            && checkpoint.is_set()
        {
            self.set_selected(checkpoint.is_value());
        }
    }
    pub fn set_debug(&mut self, input: bool) {
        self.debug = input;
    }
    pub fn set_field_highlight(&mut self, value: bool) {
        if self.field_highlight.is_none() {
            self.field_highlight = Some(BooleanFieldSetting::default());
            self.add_action_listener();
        }
        self.field_highlight.as_mut().unwrap().set(value);
        self.update_field_highlight();
    }
    pub fn set_field_highlight_string(&mut self, input: Option<&str>) {
        if self.field_highlight.is_none() && input.is_some() {
            self.field_highlight = Some(BooleanFieldSetting::default());
            self.add_action_listener();
        }
        if let Some(highlight) = &mut self.field_highlight {
            highlight.set_string(input);
            self.update_field_highlight();
        }
    }
    pub fn get_field_highlight(&self) -> Option<&BooleanFieldSetting> {
        self.field_highlight.as_ref()
    }
    pub fn set_field_highlight_setting(&mut self, input: Option<&BooleanFieldSetting>) {
        if self.field_highlight.is_none()
            && input.is_some_and(|value| value.is_set() && value.is_boolean())
        {
            self.field_highlight = Some(BooleanFieldSetting::default());
            self.add_action_listener();
        }
        if let Some(highlight) = &mut self.field_highlight {
            highlight.copy(input);
            self.update_field_highlight();
        }
    }
    pub fn clear_field_highlight(&mut self) {
        if let Some(highlight) = &mut self.field_highlight
            && highlight.is_set()
        {
            highlight.reset();
            self.update_field_highlight();
        }
    }
    pub fn is_field_highlight_set(&self) -> bool {
        self.field_highlight
            .as_ref()
            .is_some_and(BooleanFieldSetting::is_set)
    }
    pub fn equals_field_highlight(&self) -> bool {
        self.field_highlight
            .as_ref()
            .is_some_and(|value| value.is_set() && value.equals(self.is_selected()))
    }
    pub fn equals_field_highlight_string(&self, value: Option<&str>) -> bool {
        self.field_highlight
            .as_ref()
            .is_some_and(|setting| setting.is_set() && setting.equals_string(value))
    }
    pub fn get_action_command(&self) -> Option<&str> {
        self.check_box
            .action_command
            .as_deref()
            .or(self.check_box.text.as_deref())
    }
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        self.check_box.enabled = enabled && self.editable;
        if enabled && self.editable {
            self.update_field_highlight();
        }
    }
    pub fn set_editable(&mut self, editable: bool) {
        self.editable = editable;
        if self.enabled {
            self.check_box.enabled = editable;
        }
        if self.enabled && editable {
            self.update_field_highlight();
        }
    }
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
    pub fn is_editable(&self) -> bool {
        self.editable
    }
    pub fn equals_action_command(&self, action_command: Option<&str>) -> bool {
        let Some(action_command) = action_command else {
            return false;
        };
        self.get_action_command() == Some(action_command)
            || self.action_command_when_false.as_deref() == Some(action_command)
            || self.action_command_when_true.as_deref() == Some(action_command)
    }
    pub fn action_performed(&mut self) {
        self.update_two_label();
        self.update_field_highlight();
        self.update_warning();
    }
    /// Java `addActionListener(ActionListener)` duplicate-elision boundary.
    pub fn add_action_listener(&mut self) {
        self.action_listener_count = self.action_listener_count.max(1);
    }
    pub fn remove_action_listener(&mut self) {
        self.action_listener_count = 0;
    }
    fn update_two_label(&mut self) {
        if self.label_when_false.is_some() {
            let text = if self.is_selected() {
                self.label_when_true.clone()
            } else {
                self.label_when_false.clone()
            };
            self.check_box.text = text.clone();
            if let Some(text) = text {
                self.set_name(&text);
            }
        }
    }
    fn update_field_highlight(&mut self) {
        if self
            .field_highlight
            .as_ref()
            .is_some_and(|value| value.equals(self.is_selected()))
        {
            if self.orig_foreground.is_none() {
                self.orig_foreground = Some(self.check_box.foreground.unwrap_or(BLACK));
            }
            self.check_box.foreground = Some(FIELD_HIGHLIGHT);
        } else if let Some(foreground) = self.orig_foreground {
            self.check_box.foreground = Some(foreground);
        }
    }
    pub fn is_different_from_checkpoint_default(&self) -> bool {
        self.is_different_from_checkpoint(false)
    }
    pub fn is_different_from_checkpoint(&self, always_check: bool) -> bool {
        if !always_check && (!self.is_enabled() || !self.check_box.visible) {
            return false;
        }
        self.checkpoint
            .as_ref()
            .is_none_or(|checkpoint| !checkpoint.equals(self.is_selected()))
    }
    pub fn set_tool_tip_text(&mut self, text: Option<&str>) {
        self.set_preformatted_tooltip(text.map(Self::format_tooltip).as_deref());
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
        let text = Self::build_tooltip(
            self.unformatted_tooltip.as_deref(),
            param_descr,
            directive_descr,
        );
        self.set_tool_tip_text(text.as_deref());
        self.unformatted_tooltip = None;
    }
    pub fn set_tooltip(&mut self, tooltip: Option<&str>) {
        self.set_preformatted_tooltip(tooltip);
    }
    pub fn set_preformatted_tooltip(&mut self, tooltip: Option<&str>) {
        self.check_box.tooltip = tooltip.map(str::to_owned);
        self.tooltip = tooltip.map(str::to_owned);
    }
    pub fn set_alternate_tooltip_text(&mut self, text: Option<&str>) {
        self.alternate_tooltip = text.map(Self::format_tooltip);
    }
    pub fn switch_tooltips(&mut self, alternate: bool) {
        self.check_box.tooltip = if alternate && self.alternate_tooltip.is_some() {
            self.alternate_tooltip.clone()
        } else {
            self.tooltip.clone()
        };
    }
    pub fn get_tooltip(&self) -> Option<&str> {
        self.check_box.tooltip.as_deref()
    }
    pub fn enable_warning(&mut self, value: bool) {
        if self.default_background.is_none() {
            self.default_background = Some(self.check_box.background.unwrap_or(GRAY));
        }
        self.add_action_listener();
        self.warning_enabled = true;
        self.warning_value = value;
        self.update_warning();
    }
    pub fn disable_warning(&mut self) {
        if !self.warning_enabled {
            return;
        }
        self.warning_enabled = false;
        self.update_warning();
    }
    pub fn set_flag(&mut self, flag_type: Option<FlagType>) {
        if flag_type.is_some() && self.check_box.enabled {
            self.check_box.background = Some(Color(255, 255, 204));
        } else if let Some(default_background) = self.default_background {
            self.check_box.background = Some(default_background);
        }
        self.flag_type = flag_type;
    }
    fn update_warning(&mut self) {
        self.set_flag(
            if self.warning_enabled && self.is_selected() == self.warning_value {
                Some(FlagType::Warning)
            } else {
                None
            },
        );
    }
    fn format_tooltip(text: &str) -> String {
        format!("<html>{text}")
    }
    fn build_tooltip(
        unformatted: Option<&str>,
        param: Option<&str>,
        directive: Option<&str>,
    ) -> Option<String> {
        if unformatted.is_none() && param.is_none() && directive.is_none() {
            return None;
        }
        if param.is_none() && directive.is_none() {
            return unformatted.map(str::to_owned);
        }
        let text = unformatted.map(|value| value.trim_end_matches('.').trim().to_string());
        let descriptions = [param, directive]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>()
            .join(", ");
        Some(match text {
            Some(text) => format!("{text} ({descriptions})."),
            None => format!("{descriptions}."),
        })
    }
}

/// `CheckBox` is one of the two Java `ButtonComponent` implementations.
/// Listener ownership and dispatch remain with the GUI frontend; the existing
/// source listener-registration state records the registration here.
impl ButtonComponent for CheckBox {
    fn add_action_listener(&mut self, _listener: &mut dyn ActionListenerBoundary) {
        CheckBox::add_action_listener(self);
    }

    fn is_selected(&self) -> bool {
        CheckBox::is_selected(self)
    }

    fn get_action_command(&self) -> Option<&str> {
        CheckBox::get_action_command(self)
    }

    fn is_enabled(&self) -> bool {
        CheckBox::is_enabled(self)
    }
}

impl std::fmt::Display for CheckBox {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "[text:{}]", self.get_text().unwrap_or("null"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn source_uitest_name_and_two_labels_follow_selection() {
        let mut box_ = CheckBox::new_with_texts(Some("Off:"), Some("On:"));
        assert_eq!(box_.get_name(), Some("cb.off"));
        box_.set_selected(true);
        assert_eq!(box_.get_text(), Some("On:"));
        assert_eq!(box_.get_name(), Some("cb.on"));
    }
    #[test]
    fn checkpoint_respects_invisible_and_disabled_source_exclusion() {
        let mut box_ = CheckBox::new();
        box_.checkpoint();
        box_.set_selected(true);
        assert!(box_.is_different_from_checkpoint_default());
        box_.set_visible(false);
        assert!(!box_.is_different_from_checkpoint_default());
        assert!(box_.is_different_from_checkpoint(true));
    }
    #[test]
    fn highlight_and_warning_follow_source_value() {
        let mut box_ = CheckBox::new();
        box_.set_field_highlight(true);
        assert_eq!(box_.check_box.foreground, Some(BLACK));
        box_.set_selected(true);
        assert_eq!(box_.check_box.foreground, Some(FIELD_HIGHLIGHT));
        box_.enable_warning(true);
        assert_eq!(box_.flag_type, Some(FlagType::Warning));
        box_.set_selected(false);
        // Java `setSelected` updates highlighting but does not synthesize an
        // ActionEvent; the BooleanFlagExtension update occurs in its listener.
        box_.action_performed();
        assert_eq!(box_.flag_type, None);
    }
}
