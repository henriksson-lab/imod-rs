//! `IMOD/Etomo/src/etomo/ui/swing/ToggleEbutton.java`.
//!
//! `JToggleButton`, layout, style painting, and native file selection remain
//! GUI boundaries.  This module retains the Java source unit's control,
//! appearance, flag, naming, and insertion state.
#![allow(dead_code)]

use std::{cell::RefCell, path::PathBuf, rc::Rc};

use crate::imod::etomo::{
    etomo_director::ARGUMENTS,
    storage::autodoc::autodoc_tokenizer::{DEFAULT_DELIMITER, SEPARATOR_CHAR},
    util::utilities,
};

use super::{
    appearance_extension::{AppearanceExtension, ComponentBoundary, FlagDisplay, FlagType},
    button_style_extension::ButtonStyleExtensionBoundary,
    control_mediator::ControlMediator,
    control_state::ControlState,
    control_target::ControlTarget,
    controller::Controller,
    fixed_dim::FixedDim,
    grid_bag_extension::GridBagExtension,
};

/// Java `JToggleButton` state directly touched by this source unit.
#[derive(Clone, Debug)]
pub struct JToggleButtonBoundary {
    pub component: Rc<RefCell<ComponentBoundary>>,
    pub name: Option<String>,
    pub selected: bool,
    pub visible: bool,
    pub preferred_size: Option<(i32, i32)>,
    pub maximum_size: Option<(i32, i32)>,
    pub action_listener_registered: bool,
}
impl Default for JToggleButtonBoundary {
    fn default() -> Self {
        Self {
            component: Rc::new(RefCell::new(ComponentBoundary::default())),
            name: None,
            selected: false,
            visible: true,
            preferred_size: None,
            maximum_size: None,
            action_listener_registered: false,
        }
    }
}

/// Java package-private final `ToggleEbutton`.
pub struct ToggleEbutton {
    /// Java final `button`.
    pub button: JToggleButtonBoundary,
    /// Java final `target`; `None` is the explicit field boundary before its
    /// source unit is wired as a mutable `ControlTarget`.
    pub target: Option<Rc<RefCell<dyn ControlTarget>>>,
    /// Java final `controlMode`.
    pub control_mode: Option<ControlState>,
    /// Java final `buttonStyle`.
    pub button_style: Option<ButtonStyleExtensionBoundary>,
    /// Java `gridBagExtension`.
    pub grid_bag_extension: Option<GridBagExtension>,
    /// Java `controlMediator`; retained because the source declares it, though
    /// it is not assigned by this Java source unit.
    pub control_mediator: Option<ControlMediator>,
    /// Java `appearanceExtension`.
    pub appearance_extension: Option<AppearanceExtension>,
    /// Java `debug`.
    pub debug: bool,
}

impl ToggleEbutton {
    /// Java private `ToggleEbutton(ControlMode, ControlTarget, Dimension, ButtonStyleExtension)`.
    fn new(
        control_mode: Option<ControlState>,
        target: Option<Rc<RefCell<dyn ControlTarget>>>,
        fixed_size: Option<(i32, i32)>,
        button_style: Option<ButtonStyleExtensionBoundary>,
    ) -> Self {
        let mut value = Self {
            button: JToggleButtonBoundary::default(),
            target,
            control_mode,
            button_style,
            grid_bag_extension: None,
            control_mediator: None,
            appearance_extension: None,
            debug: false,
        };
        value.set_name();
        if let Some((width, height)) = fixed_size {
            value.button.preferred_size = Some((width, height));
            value.button.maximum_size = Some((width, height));
        }
        if let Some(button_style) = &mut value.button_style {
            button_style.setup(None, value.debug);
        }
        value
    }

    /// Java static `getOverrideInstance(ControlTarget)`.
    pub fn get_override_instance(target: Option<Rc<RefCell<dyn ControlTarget>>>) -> Self {
        let mut value = Self::new(
            Some((*ControlState::OVERRIDE).clone()),
            target,
            Some((
                FixedDim::INLINE_SQUARE_SIZE.width,
                FixedDim::INLINE_SQUARE_SIZE.height,
            )),
            Some(ButtonStyleExtensionBoundary::new("ButtonStyleExtension")),
        );
        value.create_panel();
        value
    }

    /// Java private `setName(ControlTarget, ControlMode)`.
    fn set_name(&mut self) {
        let control_name_set = self
            .control_mode
            .as_ref()
            .is_some_and(|mode| mode.has_field_name());
        let field_type = if control_name_set { "ctb" } else { "bn" };
        let mut name = self.target.as_ref().and_then(|target| {
            utilities::convert_label_to_name(Some(&target.borrow().get_label()), false)
        });
        if let Some(control_mode) = &self.control_mode {
            name = control_mode.append_to_name(name);
        }
        let Some(name) = name else { return };
        self.button.name = Some(format!("{field_type}{SEPARATOR_CHAR}{name}"));
        if ARGUMENTS.lock().unwrap().is_print_names() {
            println!(
                "{} {DEFAULT_DELIMITER} ",
                self.button.name.as_deref().expect("assigned above")
            );
        }
    }

    /// Java private `createPanel()`.
    fn create_panel(&mut self) {
        if self.control_mode.is_some() {
            self.button.action_listener_registered = true;
        }
    }

    /// Java `getComponent()`.
    pub fn get_component(&self) -> &JToggleButtonBoundary {
        &self.button
    }

    /// Java `isControl()`.
    pub fn is_control(&self) -> bool {
        self.is_selected() && self.is_enabled()
    }

    /// Java `selectFile()`.
    pub fn select_file(&mut self) -> Option<PathBuf> {
        None
    }

    /// Java `isSelected()`.
    pub fn is_selected(&self) -> bool {
        self.button.selected
    }

    /// Java `isEnabled()`.
    pub fn is_enabled(&self) -> bool {
        self.appearance_extension
            .as_ref()
            .map(AppearanceExtension::is_enabled)
            .unwrap_or_else(|| self.button.component.borrow().is_enabled())
    }

    /// Java `setDebug(boolean)`.
    pub fn set_debug(&mut self, debug: bool) {
        self.debug = debug;
    }

    /// Java `setSelected(boolean)`.
    pub fn set_selected(&mut self, selected: bool) {
        self.button.selected = selected;
        self.action_performed();
    }

    /// Java `setVisible(boolean)`.
    pub fn set_visible(&mut self, visible: bool) {
        self.button.visible = visible;
    }

    /// Java private `createAppearanceExtension()`.
    fn create_appearance_extension(&mut self) {
        if self.appearance_extension.is_none() {
            self.appearance_extension =
                Some(AppearanceExtension::new(Rc::clone(&self.button.component)));
        }
    }

    /// Java `setEnabled(boolean)`.
    pub fn set_enabled(&mut self, enabled: bool) {
        if let Some(extension) = &mut self.appearance_extension {
            extension.set_enabled(enabled);
        } else {
            self.button.component.borrow_mut().set_enabled(enabled);
        }
    }

    /// Java `setEditable(boolean)`.
    pub fn set_editable(&mut self, editable: bool) {
        self.create_appearance_extension();
        self.appearance_extension
            .as_mut()
            .expect("created above")
            .set_editable(editable);
    }

    /// Java `setFlag(FlagType)`.
    pub fn set_flag(&mut self, flag_type: Option<FlagType>) {
        let selected = self.is_selected();
        if let Some(style) = &mut self.button_style {
            style.update_appearance(flag_type, false, selected);
        }
        self.create_appearance_extension();
        self.appearance_extension
            .as_mut()
            .expect("created above")
            .set_flag(flag_type);
    }

    /// Java `actionPerformed(ActionEvent)`; Java ignores its event argument.
    pub fn action_performed(&mut self) {
        if let Some(control_mode) = self.control_mode.clone() {
            if let Some(target) = self.target.clone() {
                ControlMediator::INSTANCE.control_event_state(
                    Some(self),
                    Some(&mut *target.borrow_mut()),
                    Some(&control_mode),
                );
            }
        }
    }

    /// Java `remove()`.
    pub fn remove(&mut self) {
        if let Some(extension) = &mut self.grid_bag_extension {
            extension.remove();
        }
    }

    /// Java `add(JPanel, GridBagLayout, GridBagConstraints)`.
    pub fn add(&mut self, constraints: (i32, i32)) {
        if self.grid_bag_extension.is_none() {
            self.grid_bag_extension = Some(GridBagExtension::new());
        }
        self.grid_bag_extension
            .as_mut()
            .expect("created above")
            .add(0, constraints);
    }

    /// Java `selectMultipleFiles()`.
    pub fn select_multiple_files(&mut self) -> Option<Vec<PathBuf>> {
        None
    }
}

impl FlagDisplay for ToggleEbutton {
    fn set_flag(&mut self, flag_type: Option<FlagType>) {
        ToggleEbutton::set_flag(self, flag_type);
    }
}
impl Controller for ToggleEbutton {
    fn is_control(&self) -> bool {
        ToggleEbutton::is_control(self)
    }
    fn set_editable(&mut self, editable: bool) {
        ToggleEbutton::set_editable(self, editable);
    }
    fn set_enabled(&mut self, enabled: bool) {
        ToggleEbutton::set_enabled(self, enabled);
    }
    fn select_file(&mut self) -> Option<PathBuf> {
        ToggleEbutton::select_file(self)
    }
    fn select_multiple_files(&mut self) -> Option<Vec<PathBuf>> {
        ToggleEbutton::select_multiple_files(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    #[derive(Default)]
    struct Target {
        component: Option<bool>,
        events: usize,
    }
    impl ControlTarget for Target {
        fn clear(&mut self) {}
        fn set_text_file(&mut self, _: &Path) {}
        fn set_text_files(&mut self, _: &[PathBuf]) {}
        fn get_label(&self) -> String {
            "Value".into()
        }
        fn set_component_control(&mut self, value: bool, _: &ControlState) {
            self.component = Some(value);
        }
        fn set_enable_control(&mut self, _: bool, _: &ControlState) {}
        fn send_control_event(&mut self) {
            self.events += 1;
        }
        fn is_local_dir(&self, _: &str) -> bool {
            false
        }
    }
    #[test]
    fn override_factory_keeps_name_size_style_and_control_listener_order() {
        let target = Rc::new(RefCell::new(Target::default()));
        let button = ToggleEbutton::get_override_instance(Some(target));
        assert_eq!(button.button.name.as_deref(), Some("ctb.value-override"));
        assert_eq!(button.button.preferred_size, Some((22, 22)));
        assert!(button.button.action_listener_registered);
        assert_eq!(button.button_style.as_ref().unwrap().setup_count, 1);
    }
    #[test]
    fn selection_immediately_mediates_override_and_appearance_is_lazy() {
        let target = Rc::new(RefCell::new(Target::default()));
        let mut button = ToggleEbutton::get_override_instance(Some(target.clone()));
        button.set_selected(true);
        assert_eq!(target.borrow().component, Some(true));
        assert_eq!(target.borrow().events, 1);
        button.set_flag(Some(FlagType::WARNING));
        assert!(button.appearance_extension.is_some());
        assert_eq!(
            button.button_style.as_ref().unwrap().last_update,
            Some((Some(FlagType::WARNING), false, true))
        );
        button.add((3, 4));
        button.remove();
        assert!(button.grid_bag_extension.as_ref().unwrap().removed);
    }
    #[test]
    fn null_target_keeps_native_boundary_and_noops_control_event() {
        let mut button = ToggleEbutton::get_override_instance(None);
        button.set_selected(true);
        assert!(button.is_control());
        assert_eq!(button.select_file(), None);
        assert_eq!(button.select_multiple_files(), None);
    }
}
