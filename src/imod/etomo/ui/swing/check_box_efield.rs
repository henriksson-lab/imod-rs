//! `IMOD/Etomo/src/etomo/ui/swing/CheckBoxEfield.java`.
//!
//! A `JCheckBox` field that can act as a controller: with a control mode (the enable
//! control instance) every selection and enable change sends a control event to its
//! target through the `ControlMediator`.
//!
//! The control target is held as a `Weak<dyn ControlTarget>`, as in `Ebutton`: the
//! target field and this check box are owned by the same panel.

use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::{Rc, Weak};

use super::appearance_extension::AppearanceExtension;
use super::control_mediator;
use super::control_mode::ControlMode;
use super::control_state;
use super::control_target::ControlTarget;
use super::controller::Controller;
use super::swing_component::SwingComponent;
use super::tooltip_formatter;
use crate::imod::etomo::etomo_director::ARGUMENTS;
use crate::imod::etomo::jdk::{ActionEvent, ActionListener, JComponent};
use crate::imod::etomo::storage::autodoc::autodoc_tokenizer::{DEFAULT_DELIMITER, SEPARATOR_CHAR};
use crate::imod::etomo::r#type::ui_test_field_type::UITestFieldType;
use crate::imod::etomo::ui::ui_component::UIComponent;
use crate::imod::etomo::util::utilities;

/// Java package-private `final class CheckBoxEfield implements ActionListener,
/// Controller, UIComponent, SwingComponent`.
pub struct CheckBoxEfield {
    /// This object, for `checkBox.addActionListener(this)`.
    self_ref: RefCell<Weak<CheckBoxEfield>>,
    /// Java final `checkBox`.
    check_box: Rc<JComponent>,
    /// Java final `controlTarget`.
    control_target: Option<Weak<dyn ControlTarget>>,
    /// Java final `controlMode`.
    control_mode: Option<&'static ControlMode>,
    /// Java `appearanceExtension`.
    appearance_extension: RefCell<Option<Rc<AppearanceExtension>>>,
}

impl CheckBoxEfield {
    /// Java private `CheckBoxEfield(String, ControlMode, ControlTarget)`.
    fn new(
        label: Option<&str>,
        control_mode: Option<&'static ControlMode>,
        control_target: Option<Weak<dyn ControlTarget>>,
    ) -> Rc<CheckBoxEfield> {
        let mut label: Option<String> = label.map(str::to_owned);
        if label.is_none() {
            // Java: `controlTarget != null` (a dropped target counts as null).
            if let Some(target) = control_target.as_ref().and_then(Weak::upgrade) {
                label = target.get_label();
            }
        }
        if label.is_none() {
            if let Some(control_mode) = control_mode {
                label = control_mode.get_field_name().map(str::to_owned);
            }
        }
        let instance = Rc::new(CheckBoxEfield {
            self_ref: RefCell::new(Weak::new()),
            check_box: JComponent::new_check_box(label.as_deref().unwrap_or("")),
            control_target,
            control_mode,
            appearance_extension: RefCell::new(None),
        });
        *instance.self_ref.borrow_mut() = Rc::downgrade(&instance);
        instance.set_name(
            label.as_deref(),
            instance.control_target.as_ref(),
            control_mode,
        );
        instance
    }

    /// Java static `getInstance(String)`.
    pub fn get_instance(label: Option<&str>) -> Rc<CheckBoxEfield> {
        CheckBoxEfield::new(label, None, None)
    }

    /// Java static `getEnableControlInstance(ControlTarget)`.
    pub fn get_enable_control_instance(
        control_target: Option<Weak<dyn ControlTarget>>,
    ) -> Rc<CheckBoxEfield> {
        let instance = CheckBoxEfield::new(None, Some(&**control_state::ENABLE), control_target);
        instance.add_listeners();
        instance
    }

    /// Java `setText(String)`.
    pub fn set_text(&self, label: Option<&str>) {
        self.check_box.set_text(label.unwrap_or(""));
        self.set_name(label, self.control_target.as_ref(), self.control_mode);
    }

    /// Java private `setName(String, ControlTarget, ControlMode)`.
    fn set_name(
        &self,
        label: Option<&str>,
        target: Option<&Weak<dyn ControlTarget>>,
        control_mode: Option<&'static ControlMode>,
    ) {
        let field_type = UITestFieldType::CHECK_BOX;
        // build name
        let mut name: Option<String> = None;
        let unlimited_segments = field_type.is_unlimited_segments();
        if label.is_some() {
            name = self.append_to_name(
                name,
                utilities::convert_label_to_name(label, unlimited_segments),
            );
        } else {
            if let Some(target) = target.and_then(Weak::upgrade) {
                name = utilities::convert_label_to_name(
                    target.get_label().as_deref(),
                    unlimited_segments,
                );
            }
            if let Some(control_mode) = control_mode {
                name = self.append_to_name(name, control_mode.get_field_name().map(str::to_owned));
            }
        }
        let Some(name) = name else {
            return;
        };
        self.check_box
            .set_name(Some(&format!("{}{}{}", field_type, SEPARATOR_CHAR, name)));
        if ARGUMENTS.lock().unwrap().is_print_names() {
            println!(
                "{} {} ",
                self.check_box.get_name().as_deref().unwrap_or("null"),
                DEFAULT_DELIMITER
            );
        }
    }

    /// Java private `appendToName(String, String)`.
    fn append_to_name(&self, name: Option<String>, append_name: Option<String>) -> Option<String> {
        let Some(name) = name else {
            return append_name;
        };
        let Some(append_name) = append_name else {
            return Some(name);
        };
        Some(format!(
            "{}{}{}",
            name,
            utilities::NAME_SEPARATOR,
            append_name
        ))
    }

    /// Java `@Override getComponent()`.
    pub fn get_component(&self) -> Rc<JComponent> {
        self.check_box.clone()
    }

    /// Java private `addListeners()`.
    fn add_listeners(&self) {
        if self.control_mode.is_some() {
            // checkBox.addActionListener(this)
            let this = self.self_ref.borrow().clone();
            self.check_box.add_action_listener(Rc::new(move |event| {
                if let Some(this) = this.upgrade() {
                    this.action_performed(event);
                }
            }));
        }
    }

    /// Java `addActionListener(ActionListener)`.
    pub fn add_action_listener(&self, listener: ActionListener) {
        self.check_box.add_action_listener(listener);
    }

    // controlMediator

    /// Java `@Override actionPerformed(ActionEvent)`.
    pub fn action_performed(&self, _event: &ActionEvent) {
        if let Some(control_mode) = self.control_mode {
            let target = self.control_target.as_ref().and_then(Weak::upgrade);
            control_mediator::INSTANCE.control_event_controller_control_target_control_mode(
                self,
                target.as_deref(),
                Some(control_mode),
            );
        }
    }

    /// Java `@Override isControl()`.
    pub fn is_control(&self) -> bool {
        self.is_selected() && self.is_enabled()
    }

    /// Java `@Override selectFile()`.
    pub fn select_file(&self) -> Option<PathBuf> {
        None
    }

    // AppearanceExtension

    /// Java `@Override setEditable(boolean)`.
    pub fn set_editable(&self, editable: bool) {
        self.create_appearance_extension();
        let appearance_extension = self.appearance_extension.borrow().clone().unwrap();
        appearance_extension.set_editable(editable);
    }

    /// Java private `createAppearanceExtension()`.
    fn create_appearance_extension(&self) {
        if self.appearance_extension.borrow().is_none() {
            *self.appearance_extension.borrow_mut() =
                Some(AppearanceExtension::new_component(&self.check_box));
        }
    }

    /// Java `isSelected()`.
    pub fn is_selected(&self) -> bool {
        self.check_box.is_selected()
    }

    /// Java `@Override setEnabled(boolean)`.
    pub fn set_enabled(&self, enabled: bool) {
        let appearance_extension = self.appearance_extension.borrow().clone();
        if let Some(appearance_extension) = appearance_extension {
            appearance_extension.set_enabled(enabled);
        } else {
            self.check_box.set_enabled(enabled);
        }
        if let Some(control_mode) = self.control_mode {
            let target = self.control_target.as_ref().and_then(Weak::upgrade);
            control_mediator::INSTANCE.control_event_controller_control_target_control_mode(
                self,
                target.as_deref(),
                Some(control_mode),
            );
        }
    }

    /// Java `clear()`.
    pub fn clear(&self) {
        self.set_selected(false);
    }

    /// Java `setVisible(boolean)`.
    pub fn set_visible(&self, visible: bool) {
        self.get_component().set_visible(visible);
    }

    /// Java `isVisible()`.
    pub fn is_visible(&self) -> bool {
        self.get_component().is_visible()
    }

    /// Java `isEnabled()`.
    pub fn is_enabled(&self) -> bool {
        let appearance_extension = self.appearance_extension.borrow().clone();
        if let Some(appearance_extension) = appearance_extension {
            return appearance_extension.is_enabled();
        }
        self.check_box.is_enabled()
    }

    /// Java `isEditable()`.
    pub fn is_editable(&self) -> bool {
        let appearance_extension = self.appearance_extension.borrow().clone();
        if let Some(appearance_extension) = appearance_extension {
            return appearance_extension.is_editable();
        }
        true
    }

    /// Java `setTooltip(String)`.
    pub fn set_tooltip(&self, text: Option<&str>) {
        self.check_box
            .set_tool_tip_text(tooltip_formatter::INSTANCE.format(text).as_deref());
    }

    /// Java `getActionCommand()`.
    pub fn get_action_command(&self) -> Option<String> {
        self.check_box.get_action_command()
    }

    /// Java `setSelected(boolean)`.
    pub fn set_selected(&self, selected: bool) {
        self.check_box.set_selected(selected);
        if let Some(control_mode) = self.control_mode {
            let target = self.control_target.as_ref().and_then(Weak::upgrade);
            control_mediator::INSTANCE.control_event_controller_control_target_control_mode(
                self,
                target.as_deref(),
                Some(control_mode),
            );
        }
    }

    /// Java `@Override selectMultipleFiles()`.
    pub fn select_multiple_files(&self) -> Option<Vec<PathBuf>> {
        // TODO Auto-generated method stub
        None
    }
}

impl Controller for CheckBoxEfield {
    fn is_control(&self) -> bool {
        CheckBoxEfield::is_control(self)
    }
    fn set_editable(&self, editable: bool) {
        CheckBoxEfield::set_editable(self, editable)
    }
    fn set_enabled(&self, enabled: bool) {
        CheckBoxEfield::set_enabled(self, enabled)
    }
    fn select_file(&self) -> Option<PathBuf> {
        CheckBoxEfield::select_file(self)
    }
    fn select_multiple_files(&self) -> Option<Vec<PathBuf>> {
        CheckBoxEfield::select_multiple_files(self)
    }
}

impl UIComponent for CheckBoxEfield {
    /// Java `@Override getUIComponent()`.
    fn get_ui_component(&self) -> &dyn SwingComponent {
        self
    }
    fn get_component(&self) -> Rc<JComponent> {
        CheckBoxEfield::get_component(self)
    }
}

impl SwingComponent for CheckBoxEfield {
    fn get_component(&self) -> Rc<JComponent> {
        CheckBoxEfield::get_component(self)
    }
}
