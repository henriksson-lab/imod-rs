//! `IMOD/Etomo/src/etomo/ui/swing/ControlComponentModule.java`.
//!
//! A label that an efield shows in place of its component while the field is under
//! the control of another field (for example "override").

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use super::control_state::{self, ControlState};
use super::tooltip_formatter;
use crate::imod::etomo::etomo_director::ARGUMENTS;
use crate::imod::etomo::jdk::JComponent;
use crate::imod::etomo::r#type::ui_test_field_type;
use crate::imod::etomo::storage::autodoc::autodoc_tokenizer::{DEFAULT_DELIMITER, SEPARATOR_CHAR};
use crate::imod::etomo::util::utilities;

/// Java `ControlComponentModule`.
pub struct ControlComponentModule {
    /// Java `controlComponent`, a `JLabel`.
    control_component: Rc<JComponent>,
    /// Java `controlState`.  Java never assigns it (see [`Self::set_component_control`]),
    /// so it stays `null`.
    control_state: RefCell<Option<&'static ControlState>>,
    /// Java `debug`.
    #[allow(dead_code)]
    debug: Cell<bool>,
}

impl ControlComponentModule {
    /// Java `ControlComponentModule()`.
    pub fn new() -> Rc<ControlComponentModule> {
        let module = Rc::new(ControlComponentModule {
            control_component: JComponent::new_label(""),
            control_state: RefCell::new(None),
            debug: Cell::new(false),
        });
        // init
        module.set_visible(false);
        module
    }

    /// Java `setToContainerName(String)`.
    pub fn set_to_container_name(&self, container_label: Option<&str>) {
        let field_type = &ui_test_field_type::LABEL;
        if container_label.is_some() {
            let name =
                utilities::convert_label_to_name(container_label, field_type.is_unlimited_segments());
            if let Some(name) = name {
                self.control_component
                    .set_name(Some(&format!("{}{}{}", field_type, SEPARATOR_CHAR, name)));
                if ARGUMENTS.lock().unwrap().is_print_names() {
                    println!(
                        "{} {} ",
                        self.control_component.get_name().as_deref().unwrap_or("null"),
                        DEFAULT_DELIMITER
                    );
                }
                return;
            }
        }
        self.control_component.set_name(None);
    }

    /// Java `getComponent()`.
    pub fn get_component(&self) -> Rc<JComponent> {
        self.control_component.clone()
    }

    /// Java `setTooltip(String)`.
    pub fn set_tooltip(&self, text: Option<&str>) {
        self.control_component
            .set_tool_tip_text(tooltip_formatter::INSTANCE.format(text).as_deref());
    }

    /// Java `setVisible(boolean)`.
    pub fn set_visible(&self, visible: bool) {
        self.control_component.set_visible(visible);
    }

    /// Java `setText(String)`.
    pub fn set_text(&self, text: Option<&str>) {
        self.control_component.set_text(text.unwrap_or(""));
    }

    /// Java `isOverride()`.
    pub fn is_override(&self) -> bool {
        self.control_state
            .borrow()
            .is_some_and(|control_state| std::ptr::eq(control_state, &*control_state::OVERRIDE))
    }

    /// Java `setComponentControl(boolean, ControlState)`.  Returns true if the control
    /// component is in use.
    ///
    /// Kept as in the Java: the `controlState` argument is not stored in the
    /// `controlState` field, so [`Self::is_override`] always answers false.  Storing it
    /// would be a guess about intent (`ControlComponentModule.java:73-82`).
    pub fn set_component_control(
        &self,
        mut control: bool,
        control_state: Option<&'static ControlState>,
    ) -> bool {
        if control_state.is_none() {
            control = false;
        }
        if control {
            self.set_text(control_state.unwrap().get_control_string());
        }
        self.set_visible(control);
        control
    }
}
