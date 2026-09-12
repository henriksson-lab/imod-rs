//! `IMOD/Etomo/src/etomo/ui/swing/ComboBox.java`.
//!
//! `JComboBox`, `JLabel`, `JPanel`, sizing, painting, and native listener
//! delivery remain at the explicit Swing boundary. This module owns every
//! source-visible state transition made by `ComboBox.java`.
#![allow(dead_code)]

use crate::imod::etomo::etomo_director::ARGUMENTS;
use crate::imod::etomo::storage::autodoc::autodoc_tokenizer::{DEFAULT_DELIMITER, SEPARATOR_CHAR};
use crate::imod::etomo::r#type::debug_level::DebugLevel;
use crate::imod::etomo::ui::swing::button_component::ActionListenerBoundary;
use crate::imod::etomo::util::utilities;

/// Boundary for Java `FocusListener`; Swing owns focus-event construction and
/// listener lifetime.
pub trait FocusListenerBoundary {
    /// Java `focusGained(FocusEvent)`.
    fn focus_gained(&mut self);
    /// Java `focusLost(FocusEvent)`.
    fn focus_lost(&mut self);
}

/// Java `Object` entries accepted by `JComboBox.addItem(Object)`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ComboBoxObject<T> {
    /// A caller-supplied Java `Object`.
    Object(T),
    /// A source-created Java `String` placeholder.
    String(String),
}

/// Source-observable `JComboBox` state. Drawing and event dispatch belong to
/// the frontend.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JComboBoxBoundary<T> {
    pub items: Vec<Option<ComboBoxObject<T>>>,
    pub selected_index: isize,
    pub name: String,
    pub action_command: Option<String>,
    pub enabled: bool,
    pub editable: bool,
    pub visible: bool,
    pub foreground_field_highlight: bool,
    pub field_highlight_border: bool,
    pub maximum_width: Option<i32>,
    pub tool_tip_text: Option<String>,
    pub action_listener_count: usize,
    pub focus_listener_count: usize,
}

/// Java package-private final `ComboBox`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComboBox<T = String> {
    pub combo_box: JComboBoxBoundary<T>,
    pub label: Option<String>,
    pub label_enabled: bool,
    pub label_field_highlight: bool,
    pub label_tool_tip_text: Option<String>,
    pub pnl_root: bool,
    pub text_entry_policy: bool,
    pub empty_choice: bool,
    pub checkpointed: bool,
    pub checkpoint_index: isize,
    pub debug: DebugLevel,
    pub enabled_policy: bool,
    pub enabled: bool,
    pub editable: bool,
    pub placeholder: bool,
    pub empty_label: Option<String>,
    pub no_selection_label: Option<String>,
}

impl<T> ComboBox<T> {
    /// Java private `ComboBox(String, boolean, boolean, boolean)`.
    fn new(name: &str, labeled: bool, text_entry_policy: bool, empty_choice: bool) -> Self {
        let mut value = Self {
            combo_box: JComboBoxBoundary {
                items: Vec::new(),
                selected_index: -1,
                name: String::new(),
                action_command: None,
                enabled: true,
                editable: false,
                visible: true,
                foreground_field_highlight: false,
                field_highlight_border: false,
                maximum_width: None,
                tool_tip_text: None,
                action_listener_count: 0,
                focus_listener_count: 0,
            },
            label: labeled.then(|| name.into()),
            label_enabled: true,
            label_field_highlight: false,
            label_tool_tip_text: None,
            pnl_root: labeled,
            text_entry_policy,
            empty_choice: false,
            checkpointed: false,
            checkpoint_index: -1,
            debug: ARGUMENTS.lock().unwrap().get_debug_level(),
            enabled_policy: true,
            enabled: true,
            editable: true,
            placeholder: false,
            empty_label: None,
            no_selection_label: None,
        };
        value.set_name(name);
        if text_entry_policy {
            value.empty_choice = false;
        } else {
            value.empty_choice = empty_choice;
            if empty_choice {
                value.combo_box.items.push(None);
                value.combo_box.selected_index = 0;
            }
        }
        value.update_display();
        value
    }

    /// Java `getInstance(String)`.
    pub fn get_instance(name: &str) -> Self {
        let value = Self::new(name, true, false, false);
        value.create_panel();
        value
    }
    /// Java `getUnlabeledInstance(String)`.
    pub fn get_unlabeled_instance(name: &str) -> Self {
        let value = Self::new(name, false, false, false);
        value.create_panel();
        value
    }
    /// Java `getEditableInstance(String)`.
    pub fn get_editable_instance(name: &str) -> Self {
        let value = Self::new(name, true, true, false);
        value.create_panel();
        value
    }
    /// Java `getUnlabeledEmptyChoiceInstance(String)`.
    pub fn get_unlabeled_empty_choice_instance(name: &str) -> Self {
        let value = Self::new(name, false, false, true);
        value.create_panel();
        value
    }
    /// Java `getEmptyChoiceInstance(String)`.
    pub fn get_empty_choice_instance(name: &str) -> Self {
        let value = Self::new(name, true, false, true);
        value.create_panel();
        value
    }

    /// Java private `createPanel()`; Swing layout remains a GUI boundary.
    fn create_panel(&self) {}
    /// Java `addActionListener(ActionListener)`.
    pub fn add_action_listener(&mut self, _listener: &mut dyn ActionListenerBoundary) {
        self.combo_box.action_listener_count += 1;
    }
    /// Java `addFocusListener(FocusListener)`.
    pub fn add_focus_listener(&mut self, _listener: &mut dyn FocusListenerBoundary) {
        self.combo_box.focus_listener_count += 1;
    }

    /// Java `setPlaceholder(String, String)`.
    pub fn set_placeholder(&mut self, empty_label: Option<&str>, no_selection_label: Option<&str>) {
        if self.text_entry_policy || self.empty_choice {
            return;
        }
        self.empty_label = empty_label.map(str::to_owned);
        self.no_selection_label = no_selection_label.map(str::to_owned);
        let no_label = empty_label.is_none() && no_selection_label.is_none();
        if !self.placeholder && no_label {
            return;
        }
        let empty = self.is_empty();
        if empty {
            self.combo_box.items.clear();
            self.combo_box.selected_index = -1;
        }
        self.placeholder = !(self.placeholder && no_label);
        if self.placeholder && empty {
            self.combo_box
                .items
                .push(empty_label.map(|label| ComboBoxObject::String(label.to_owned())));
            self.combo_box.selected_index = 0;
        }
        if !empty {
            let items = std::mem::take(&mut self.combo_box.items);
            self.combo_box.selected_index = -1;
            if self.placeholder {
                self.combo_box
                    .items
                    .push(no_selection_label.map(|label| ComboBoxObject::String(label.to_owned())));
            }
            self.combo_box.items.extend(items);
            self.combo_box.selected_index = if self.combo_box.items.is_empty() {
                -1
            } else {
                0
            };
        }
    }

    /// Java `addItem(Object)`.
    pub fn add_item(&mut self, input: Option<T>) {
        if self.placeholder && self.is_empty() {
            self.combo_box.items.clear();
            self.combo_box.items.push(
                self.no_selection_label
                    .as_ref()
                    .map(|label| ComboBoxObject::String(label.clone())),
            );
            self.combo_box.selected_index = 0;
        }
        self.combo_box.items.push(input.map(ComboBoxObject::Object));
        if self.combo_box.selected_index == -1 {
            self.combo_box.selected_index = 0;
        }
        self.update_display();
    }

    /// Java `setMaximumWidth(int)`; native sizing stays at the GUI boundary.
    pub fn set_maximum_width(&mut self, width: i32) {
        self.combo_box.maximum_width = Some(width);
    }
    /// Java `setFieldHighlight()`.
    pub fn set_field_highlight(&mut self) {
        self.label_field_highlight = true;
        self.combo_box.foreground_field_highlight = true;
        self.combo_box.field_highlight_border = true;
    }
    /// Java `getActionCommand()`.
    pub fn get_action_command(&self) -> Option<&str> {
        self.combo_box.action_command.as_deref()
    }
    /// Source-visible identity of Java `getComponent()`.
    pub fn get_component_is_panel(&self) -> bool {
        self.pnl_root
    }
    /// Java `getSelectedIndex()`.
    pub fn get_selected_index(&self) -> isize {
        let index = self.combo_box.selected_index;
        if (self.empty_choice || self.placeholder) && index > -1 {
            index - 1
        } else {
            index
        }
    }
    /// Java `getSelectedItem()`.
    pub fn get_selected_item(&self) -> Option<&ComboBoxObject<T>> {
        usize::try_from(self.combo_box.selected_index)
            .ok()
            .and_then(|index| self.combo_box.items.get(index))
            .and_then(Option::as_ref)
    }

    /// Java `removeAllItems()`.
    pub fn remove_all_items(&mut self) {
        self.combo_box.items.clear();
        self.combo_box.selected_index = -1;
        if self.empty_choice {
            self.combo_box.items.push(None);
        } else if self.placeholder {
            self.combo_box.items.push(
                self.empty_label
                    .as_ref()
                    .map(|label| ComboBoxObject::String(label.clone())),
            );
        }
        if !self.combo_box.items.is_empty() {
            self.combo_box.selected_index = 0;
        }
        self.update_display();
    }

    /// Java private `updateDisplay()`.
    fn update_display(&mut self) {
        if !self.enabled_policy || (self.is_empty() && !self.text_entry_policy) {
            self.combo_box.enabled = false;
        } else if !self.text_entry_policy {
            self.combo_box.enabled = self.enabled && self.editable;
        } else {
            self.combo_box.enabled = self.enabled;
        }
        self.combo_box.editable = self.text_entry_policy && self.editable;
    }
    /// Java `isEmpty()`.
    pub fn is_empty(&self) -> bool {
        self.combo_box.items.len() <= usize::from(self.empty_choice || self.placeholder)
    }
    /// Java `isEnabled()`.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
    /// Java `setEnabled(boolean)`.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        if self.label.is_some() {
            self.label_enabled = enabled;
        }
        self.update_display();
    }
    /// Java `setEditable(boolean)`.
    pub fn set_editable(&mut self, editable: bool) {
        self.editable = editable;
        self.update_display();
    }
    /// Java `setEnabledPolicy(boolean)`.
    pub fn set_enabled_policy(&mut self, input: bool) {
        self.enabled_policy = input;
        self.update_display();
    }
    /// Java `getLabel()`.
    pub fn get_label(&self) -> Option<&str> {
        self.label.as_deref()
    }
    /// Java `setDebug(DebugLevel)`.
    pub fn set_debug(&mut self, input: DebugLevel) {
        self.debug = input;
    }
    /// Java `setName(String)`.
    pub fn set_name(&mut self, text: &str) {
        const FIELD_TYPE: &str = "cmb";
        let name = utilities::convert_label_to_name(Some(text), false).unwrap_or_default();
        self.combo_box.name = format!("{FIELD_TYPE}{SEPARATOR_CHAR}{name}");
        if ARGUMENTS.lock().unwrap().is_print_names() {
            println!("{} {DEFAULT_DELIMITER} ", self.combo_box.name);
        }
    }
    /// Java `checkpoint()`.
    pub fn checkpoint(&mut self) {
        self.checkpointed = true;
        self.checkpoint_index = self.get_selected_index();
    }
    /// Java `isDifferentFromCheckpoint(boolean)`.
    pub fn is_different_from_checkpoint(&self, always_check: bool) -> bool {
        if !always_check && (!self.is_enabled() || !self.is_visible()) {
            return false;
        }
        if !self.checkpointed {
            return true;
        }
        self.checkpoint_index != self.get_selected_index()
    }
    /// Java `isVisible()`.
    pub fn is_visible(&self) -> bool {
        self.combo_box.visible
    }
    /// Java `setSelectedIndex(int)`.
    pub fn set_selected_index(&mut self, mut index: isize) {
        if self.empty_choice || self.placeholder {
            index += 1;
        }
        self.combo_box.selected_index = index;
    }
    /// Java `unselect()`.
    pub fn unselect(&mut self) {
        if self.empty_choice || self.placeholder {
            self.combo_box.selected_index = 0;
        } else {
            self.combo_box.selected_index = -1;
        }
    }
    /// Java `setToolTipText(String)`.
    pub fn set_tool_tip_text(&mut self, tooltip: &str) {
        if self.label.is_some() {
            self.label_tool_tip_text = Some(tooltip.into());
        }
        self.combo_box.tool_tip_text = Some(tooltip.into());
    }
    /// Java `verifyComboBoxSource(JComboBox)` as frontend identity comparison.
    pub fn verify_combo_box_source(&self, input_is_this_combo_box: bool) -> bool {
        input_is_this_combo_box
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn factories_preserve_label_empty_choice_and_editability_policies() {
        let labeled = ComboBox::<String>::get_instance("Choose item");
        let editable = ComboBox::<String>::get_editable_instance("Type item");
        let empty = ComboBox::<String>::get_unlabeled_empty_choice_instance("Optional");
        assert_eq!(labeled.get_label(), Some("Choose item"));
        assert!(labeled.get_component_is_panel());
        assert!(!labeled.combo_box.enabled);
        assert!(editable.combo_box.editable);
        assert!(editable.combo_box.enabled);
        assert!(empty.empty_choice);
        assert!(empty.is_empty());
        assert_eq!(empty.get_selected_index(), -1);
    }
    #[test]
    fn placeholder_adjusts_indices_and_is_replaced_when_first_item_arrives() {
        let mut combo = ComboBox::<u32>::get_instance("Choice");
        combo.set_placeholder(Some("None available"), Some("Choose one"));
        assert!(combo.is_empty());
        assert_eq!(combo.get_selected_index(), -1);
        combo.add_item(Some(7));
        assert!(!combo.is_empty());
        assert_eq!(combo.get_selected_index(), -1);
        assert!(
            matches!(combo.get_selected_item(), Some(ComboBoxObject::String(value)) if value == "Choose one")
        );
        combo.set_selected_index(0);
        assert_eq!(combo.get_selected_index(), 0);
        assert_eq!(combo.get_selected_item(), Some(&ComboBoxObject::Object(7)));
    }
    #[test]
    fn enable_checkpoint_and_unselect_follow_source_precedence() {
        let mut combo = ComboBox::<String>::get_empty_choice_instance("Choice");
        combo.add_item(Some("first".into()));
        combo.set_selected_index(0);
        combo.checkpoint();
        assert!(!combo.is_different_from_checkpoint(false));
        combo.unselect();
        assert!(combo.is_different_from_checkpoint(false));
        combo.set_enabled(false);
        assert!(!combo.is_different_from_checkpoint(false));
        assert!(combo.is_different_from_checkpoint(true));
        combo.set_enabled_policy(false);
        assert!(!combo.combo_box.enabled);
    }
    #[test]
    fn source_naming_tooltip_and_highlight_are_retained_at_gui_boundary() {
        let mut combo = ComboBox::<String>::get_instance("Input File:");
        combo.set_tool_tip_text("Select a file");
        combo.set_field_highlight();
        combo.set_maximum_width(240);
        assert_eq!(combo.combo_box.name, "cmb.input-file");
        assert_eq!(combo.label_tool_tip_text.as_deref(), Some("Select a file"));
        assert_eq!(
            combo.combo_box.tool_tip_text.as_deref(),
            Some("Select a file")
        );
        assert!(combo.combo_box.field_highlight_border);
        assert_eq!(combo.combo_box.maximum_width, Some(240));
    }
}
