//! `IMOD/Etomo/src/etomo/ui/swing/CheckBoxEfield.java`.
//!
//! `JCheckBox`, Swing action dispatch, and tooltip formatting are GUI-boundary
//! services.  This unit retains the Java object's state transitions, generated
//! test name, and enable-control mediation.
#![allow(dead_code)]

use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;

use crate::imod::etomo::etomo_director::ARGUMENTS;
use crate::imod::etomo::storage::autodoc::autodoc_tokenizer::{DEFAULT_DELIMITER, SEPARATOR_CHAR};
use crate::imod::etomo::ui::swing::appearance_extension::{AppearanceExtension, ComponentBoundary};
use crate::imod::etomo::util::utilities;

/// Java `ControlMode`, represented by the field name used in UI test names.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ControlMode {
    Enable,
}

impl ControlMode {
    /// Java `getFieldName()`.
    pub fn get_field_name(&self) -> &'static str {
        match self {
            Self::Enable => "enable",
        }
    }
}

/// Java `ControlTarget`; file-selection operations remain an explicit GUI boundary.
pub trait ControlTarget {
    fn clear(&mut self);
    fn set_text_file(&mut self, file: PathBuf);
    fn set_text_files(&mut self, files: Vec<PathBuf>);
    fn get_label(&self) -> Option<String>;
    fn set_component_control(&mut self, control: bool);
    fn set_enable_control(&mut self, control: bool);
    fn send_control_event(&mut self);
    fn is_local_dir(&self, current_directory: Option<&str>) -> bool;
}

/// Source-visible `JCheckBox` state.  Native rendering and listener dispatch
/// are intentionally represented at the Swing boundary.
#[derive(Clone, Debug)]
pub struct JCheckBoxBoundary {
    pub text: Option<String>,
    pub name: Option<String>,
    pub selected: bool,
    pub visible: bool,
    pub tooltip: Option<String>,
    pub action_command: Option<String>,
    pub action_listener_count: usize,
    pub component: Rc<RefCell<ComponentBoundary>>,
}

impl Default for JCheckBoxBoundary {
    fn default() -> Self {
        Self {
            text: None,
            name: None,
            selected: false,
            visible: true,
            tooltip: None,
            action_command: None,
            action_listener_count: 0,
            component: Rc::new(RefCell::new(ComponentBoundary::default())),
        }
    }
}

/// Java package-private `CheckBoxEfield`.
pub struct CheckBoxEfield {
    pub check_box: JCheckBoxBoundary,
    control_target: Option<Rc<RefCell<dyn ControlTarget>>>,
    control_mode: Option<ControlMode>,
    appearance_extension: Option<AppearanceExtension>,
}

impl CheckBoxEfield {
    /// Java private `CheckBoxEfield(String, ControlMode, ControlTarget)`.
    fn new(
        mut label: Option<String>,
        control_mode: Option<ControlMode>,
        control_target: Option<Rc<RefCell<dyn ControlTarget>>>,
    ) -> Self {
        if label.is_none() {
            if let Some(target) = &control_target {
                label = target.borrow().get_label();
            }
        }
        if label.is_none() {
            if let Some(control_mode) = &control_mode {
                label = Some(control_mode.get_field_name().to_owned());
            }
        }
        let mut value = Self {
            check_box: JCheckBoxBoundary {
                text: label.clone(),
                ..Default::default()
            },
            control_target,
            control_mode,
            appearance_extension: None,
        };
        value.set_name(label.as_deref());
        value
    }

    /// Java `getInstance(String)`.
    pub fn get_instance(label: impl Into<String>) -> Self {
        Self::new(Some(label.into()), None, None)
    }

    /// Java `getEnableControlInstance(ControlTarget)`.
    pub fn get_enable_control_instance(control_target: Rc<RefCell<dyn ControlTarget>>) -> Self {
        let mut value = Self::new(None, Some(ControlMode::Enable), Some(control_target));
        value.add_listeners();
        value
    }

    /// Java `setText(String)`.
    pub fn set_text(&mut self, label: impl Into<String>) {
        let label = label.into();
        self.check_box.text = Some(label.clone());
        self.set_name(Some(&label));
    }

    /// Java private `setName(String, ControlTarget, ControlMode)`.
    fn set_name(&mut self, label: Option<&str>) {
        let mut name = None;
        if let Some(label) = label {
            name = self.append_to_name(name, utilities::convert_label_to_name(Some(label), true));
        } else {
            if let Some(target) = &self.control_target {
                name =
                    utilities::convert_label_to_name(target.borrow().get_label().as_deref(), true);
            }
            if let Some(control_mode) = &self.control_mode {
                name = self.append_to_name(name, Some(control_mode.get_field_name().to_owned()));
            }
        }
        let Some(name) = name else { return };
        self.check_box.name = Some(format!("cb{SEPARATOR_CHAR}{name}"));
        if ARGUMENTS.lock().unwrap().is_print_names() {
            println!(
                "{} {DEFAULT_DELIMITER} ",
                self.check_box.name.as_deref().unwrap()
            );
        }
    }

    /// Java private `appendToName(String, String)`.
    fn append_to_name(&self, name: Option<String>, append_name: Option<String>) -> Option<String> {
        match (name, append_name) {
            (None, append_name) => append_name,
            (name, None) => name,
            (Some(name), Some(append_name)) => {
                Some(format!("{name}{}{append_name}", utilities::NAME_SEPARATOR))
            }
        }
    }

    /// Java `getUIComponent()`.
    pub fn get_ui_component(&self) -> &Self {
        self
    }

    /// Java `getComponent()`; this boundary object is the JCheckBox component.
    pub fn get_component(&self) -> &JCheckBoxBoundary {
        &self.check_box
    }

    /// Java private `addListeners()`.
    fn add_listeners(&mut self) {
        if self.control_mode.is_some() {
            self.check_box.action_listener_count += 1;
        }
    }

    /// Java `addActionListener(ActionListener)`; callback ownership remains with Swing.
    pub fn add_action_listener(&mut self) {
        self.check_box.action_listener_count += 1;
    }

    /// Java `actionPerformed(ActionEvent)`.
    pub fn action_performed(&mut self) {
        if self.control_mode.is_some() {
            self.control_event();
        }
    }

    /// Java `isControl()`.
    pub fn is_control(&self) -> bool {
        self.is_selected() && self.is_enabled()
    }

    /// Java `selectFile()`.
    pub fn select_file(&self) -> Option<PathBuf> {
        None
    }

    /// Java `setEditable(boolean)`.
    pub fn set_editable(&mut self, editable: bool) {
        self.create_appearance_extension();
        self.appearance_extension
            .as_mut()
            .unwrap()
            .set_editable(editable);
    }

    /// Java private `createAppearanceExtension()`.
    fn create_appearance_extension(&mut self) {
        if self.appearance_extension.is_none() {
            self.appearance_extension =
                Some(AppearanceExtension::new(self.check_box.component.clone()));
        }
    }

    /// Java `isSelected()`.
    pub fn is_selected(&self) -> bool {
        self.check_box.selected
    }

    /// Java `setEnabled(boolean)`.
    pub fn set_enabled(&mut self, enabled: bool) {
        if let Some(appearance_extension) = &mut self.appearance_extension {
            appearance_extension.set_enabled(enabled);
        } else {
            self.check_box.component.borrow_mut().set_enabled(enabled);
        }
        if self.control_mode.is_some() {
            self.control_event();
        }
    }

    /// Java `clear()`.
    pub fn clear(&mut self) {
        self.set_selected(false);
    }

    /// Java `setVisible(boolean)`.
    pub fn set_visible(&mut self, visible: bool) {
        self.check_box.visible = visible;
    }

    /// Java `isVisible()`.
    pub fn is_visible(&self) -> bool {
        self.check_box.visible
    }

    /// Java `isEnabled()`.
    pub fn is_enabled(&self) -> bool {
        self.appearance_extension.as_ref().map_or_else(
            || self.check_box.component.borrow().is_enabled(),
            AppearanceExtension::is_enabled,
        )
    }

    /// Java `isEditable()`.
    pub fn is_editable(&self) -> bool {
        self.appearance_extension
            .as_ref()
            .map_or(true, AppearanceExtension::is_editable)
    }

    /// Java `setTooltip(String)`; `TooltipFormatter` remains a GUI boundary.
    pub fn set_tooltip(&mut self, text: impl Into<String>) {
        self.check_box.tooltip = Some(text.into());
    }

    /// Java `getActionCommand()`.
    pub fn get_action_command(&self) -> Option<&str> {
        self.check_box.action_command.as_deref()
    }

    /// Java `setSelected(boolean)`.
    pub fn set_selected(&mut self, selected: bool) {
        self.check_box.selected = selected;
        if self.control_mode.is_some() {
            self.control_event();
        }
    }

    /// Java `selectMultipleFiles()`.
    pub fn select_multiple_files(&self) -> Vec<PathBuf> {
        Vec::new()
    }

    /// Java `ControlMediator.controlEvent(Controller, ControlTarget, ControlState.ENABLE)`.
    fn control_event(&mut self) {
        if let Some(target) = &self.control_target {
            target.borrow_mut().set_enable_control(self.is_control());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Target {
        label: Option<String>,
        enabled_control: Option<bool>,
    }
    impl ControlTarget for Target {
        fn clear(&mut self) {}
        fn set_text_file(&mut self, _: PathBuf) {}
        fn set_text_files(&mut self, _: Vec<PathBuf>) {}
        fn get_label(&self) -> Option<String> {
            self.label.clone()
        }
        fn set_component_control(&mut self, _: bool) {}
        fn set_enable_control(&mut self, value: bool) {
            self.enabled_control = Some(value);
        }
        fn send_control_event(&mut self) {}
        fn is_local_dir(&self, _: Option<&str>) -> bool {
            false
        }
    }

    #[test]
    fn ordinary_instance_sets_source_ui_test_name() {
        let value = CheckBoxEfield::get_instance("Apply dose weighting");
        assert_eq!(
            value.check_box.name.as_deref(),
            Some("cb.apply-dose-weighting")
        );
        assert!(value.is_enabled());
    }

    #[test]
    fn enable_control_uses_target_label_and_selection_and_enabled_state() {
        let target = Rc::new(RefCell::new(Target {
            label: Some("Use CTF".into()),
            ..Default::default()
        }));
        let mut value = CheckBoxEfield::get_enable_control_instance(target.clone());
        assert_eq!(value.check_box.name.as_deref(), Some("cb.use-ctf"));
        value.set_selected(true);
        assert_eq!(target.borrow().enabled_control, Some(true));
        value.set_enabled(false);
        assert_eq!(target.borrow().enabled_control, Some(false));
    }

    #[test]
    fn appearance_extension_preserves_editability_and_visibility() {
        let mut value = CheckBoxEfield::get_instance("Check");
        value.set_editable(false);
        assert!(!value.is_editable());
        value.set_visible(false);
        assert!(!value.is_visible());
        value.clear();
        assert!(!value.is_selected());
    }
}
