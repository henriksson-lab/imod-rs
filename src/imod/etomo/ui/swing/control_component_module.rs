//! `IMOD/Etomo/src/etomo/ui/swing/ControlComponentModule.java`.
//!
//! `JLabel` allocation, painting, and tooltip formatting remain explicit Swing
//! presentation boundaries.  This source unit owns the label's naming, text,
//! visibility, and control-display calls.
#![allow(dead_code)]

use super::combo_box_efield::ControlState;
use crate::imod::etomo::etomo_director::ARGUMENTS;
use crate::imod::etomo::storage::autodoc::autodoc_tokenizer::{DEFAULT_DELIMITER, SEPARATOR_CHAR};
use crate::imod::etomo::ui::shared_strings::OVERRIDE_TEXT;
use crate::imod::etomo::util::utilities;

/// Source-observable `JLabel` state.  Native widget allocation, painting, and
/// `TooltipFormatter.INSTANCE.format` are GUI presentation boundaries.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct JLabelBoundary {
    pub name: Option<String>,
    pub text: Option<String>,
    pub tooltip: Option<String>,
    pub visible: bool,
}

/// Java package-private final `ControlComponentModule`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ControlComponentModule {
    /// Java `JLabel controlComponent`.
    pub control_component: JLabelBoundary,
    /// Java `ControlState controlState`; the Java `setComponentControl`
    /// parameter shadows this field, so this source unit never assigns it.
    pub control_state: Option<ControlState>,
    /// Java `boolean debug`.
    pub debug: bool,
}

impl ControlComponentModule {
    /// Java `ControlComponentModule()`.
    pub fn new() -> Self {
        let mut value = Self {
            control_component: JLabelBoundary::default(),
            control_state: None,
            debug: false,
        };
        value.set_visible(false);
        value
    }

    /// Java `setToContainerName(String)`.
    pub fn set_to_container_name(&mut self, container_label: Option<&str>) {
        const FIELD_TYPE: &str = "l";

        if let Some(container_label) = container_label {
            if let Some(name) = utilities::convert_label_to_name(Some(container_label), true) {
                self.control_component.name = Some(format!("{FIELD_TYPE}{SEPARATOR_CHAR}{name}"));
                if ARGUMENTS.lock().unwrap().is_print_names() {
                    println!(
                        "{} {DEFAULT_DELIMITER} ",
                        self.control_component.name.as_deref().unwrap_or_default()
                    );
                }
                return;
            }
        }
        self.control_component.name = None;
    }

    /// Java `getComponent`; the concrete `Component` remains a GUI boundary.
    pub fn get_component(&self) -> &JLabelBoundary {
        &self.control_component
    }

    /// Java `setTooltip(String)`.  Formatting is delegated to the unported
    /// `TooltipFormatter` presentation boundary, so its input is retained.
    pub fn set_tooltip(&mut self, text: Option<&str>) {
        self.control_component.tooltip = text.map(str::to_owned);
    }

    /// Java `setVisible(boolean)`.
    pub fn set_visible(&mut self, visible: bool) {
        self.control_component.visible = visible;
    }

    /// Java `setText(String)`.
    pub fn set_text(&mut self, text: Option<&str>) {
        self.control_component.text = text.map(str::to_owned);
    }

    /// Java `isOverride()`.
    pub fn is_override(&self) -> bool {
        self.control_state == Some(ControlState::Override)
    }

    /// Java `setComponentControl(boolean, ControlState)`.
    pub fn set_component_control(
        &mut self,
        mut control: bool,
        control_state: Option<ControlState>,
    ) -> bool {
        if control_state.is_none() {
            control = false;
        }
        if control {
            self.set_text(match control_state {
                Some(ControlState::Override) => Some(OVERRIDE_TEXT),
                Some(ControlState::Enable) | None => None,
            });
        }
        self.set_visible(control);
        control
    }
}

impl Default for ControlComponentModule {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructor_hides_the_source_label_boundary() {
        let module = ControlComponentModule::new();

        assert!(!module.control_component.visible);
        assert_eq!(module.control_state, None);
        assert!(!module.debug);
    }

    #[test]
    fn container_name_uses_label_field_type_and_unlimited_segments() {
        let mut module = ControlComponentModule::new();
        module.set_to_container_name(Some("Axis A: Final aligned stack"));

        assert_eq!(module.get_component().name.as_deref(), Some("l.axis-a"));

        module.set_to_container_name(None);
        assert_eq!(module.get_component().name, None);
    }

    #[test]
    fn tooltip_text_and_visibility_delegate_to_the_label_boundary() {
        let mut module = ControlComponentModule::new();
        module.set_tooltip(Some("A tooltip"));
        module.set_text(Some("A label"));
        module.set_visible(true);

        assert_eq!(module.get_component().tooltip.as_deref(), Some("A tooltip"));
        assert_eq!(module.get_component().text.as_deref(), Some("A label"));
        assert!(module.get_component().visible);
    }

    #[test]
    fn component_control_rejects_null_state_and_displays_override_state() {
        let mut module = ControlComponentModule::new();

        assert!(!module.set_component_control(true, None));
        assert!(!module.get_component().visible);

        assert!(module.set_component_control(true, Some(ControlState::Override)));
        assert_eq!(module.get_component().text.as_deref(), Some(OVERRIDE_TEXT));
        assert!(module.get_component().visible);

        assert!(module.set_component_control(true, Some(ControlState::Enable)));
        assert!(module.get_component().visible);
        assert_eq!(module.get_component().text, None);
    }

    #[test]
    fn source_parameter_shadowing_leaves_is_override_false() {
        let mut module = ControlComponentModule::new();
        module.set_component_control(true, Some(ControlState::Override));

        assert!(!module.is_override());
    }
}
