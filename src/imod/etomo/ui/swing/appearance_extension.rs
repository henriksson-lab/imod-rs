//! `IMOD/Etomo/src/etomo/ui/swing/AppearanceExtension.java`.
//!
//! The Java `Component` is represented by `ComponentBoundary`: the actual
//! toolkit painting remains at the GUI boundary, while this source unit owns
//! the enabled, editable, and foreground transitions made by the Java class.
#![allow(dead_code)]

use std::cell::RefCell;
use std::rc::Rc;

use super::controller::Controller;

/// Java `java.awt.Color`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Color {
    pub red: u8,
    pub green: u8,
    pub blue: u8,
}

pub const BLACK: Color = Color {
    red: 0,
    green: 0,
    blue: 0,
};
pub const FIELD_HIGHLIGHT: Color = Color {
    red: 0,
    green: 0,
    blue: 185,
};
pub const NOT_STARTED: Color = Color {
    red: 255,
    green: 204,
    blue: 204,
};
pub const WARNING_BACKGROUND: Color = Color {
    red: 255,
    green: 255,
    blue: 204,
};

/// Java `etomo.ui.FlagType`.
///
/// The public colors and background bit are the source's corresponding final
/// fields.  The identity-bearing `kind` replaces Java singleton identity.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FlagType {
    pub color: Color,
    pub background: bool,
    kind: FlagTypeKind,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FlagTypeKind {
    Template,
    TemplateError,
    Error,
    Warning,
}

impl FlagType {
    pub const TEMPLATE: Self = Self {
        color: FIELD_HIGHLIGHT,
        background: false,
        kind: FlagTypeKind::Template,
    };
    pub const TEMPLATE_ERROR: Self = Self {
        color: NOT_STARTED,
        background: false,
        kind: FlagTypeKind::TemplateError,
    };
    pub const ERROR: Self = Self {
        color: NOT_STARTED,
        background: false,
        kind: FlagTypeKind::Error,
    };
    pub const WARNING: Self = Self {
        color: WARNING_BACKGROUND,
        background: true,
        kind: FlagTypeKind::Warning,
    };

    /// Java `FlagType.isTemplate()`.
    pub fn is_template(self) -> bool {
        self.kind == FlagTypeKind::Template
    }

    /// Java `FlagType.isError()`.
    pub fn is_error(self) -> bool {
        self.kind == FlagTypeKind::Error || self.kind == FlagTypeKind::TemplateError
    }
}

/// Java `ControlState` values used by this class.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ControlState {
    Override,
    Enable,
}

/// Source-visible state delegated to Swing's `Component` methods.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComponentBoundary {
    pub foreground: Option<Color>,
    pub background: Option<Color>,
    pub enabled: bool,
}

impl Default for ComponentBoundary {
    fn default() -> Self {
        Self {
            foreground: Some(BLACK),
            background: None,
            enabled: true,
        }
    }
}

impl ComponentBoundary {
    /// Java `Component.getForeground()`.
    pub fn get_foreground(&self) -> Option<Color> {
        self.foreground
    }

    /// Java `Component.setForeground(Color)`.
    pub fn set_foreground(&mut self, color: Color) {
        self.foreground = Some(color);
    }

    /// Java `Component.getBackground()`.
    pub fn get_background(&self) -> Option<Color> {
        self.background
    }

    /// Java `Component.setBackground(Color)`.
    pub fn set_background(&mut self, color: Color) {
        self.background = Some(color);
    }

    /// Java `Component.isEnabled()`.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Java `Component.setEnabled(boolean)`.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
}

/// Java `FlagDisplay` interface.
pub trait FlagDisplay {
    fn set_flag(&mut self, flag_type: Option<FlagType>);
}

/// Java package-private `AppearanceExtension`.
pub struct AppearanceExtension {
    pub component: Rc<RefCell<ComponentBoundary>>,
    orig_foreground: Color,
    enabled_field: bool,
    pub editable_component: bool,
    enabled: bool,
    editable: bool,
    flag_type: Option<FlagType>,
    allow_flag_editable_control: bool,
    allow_foreground_change_on_error: bool,
    respond_to_non_error_flags: bool,
    debug: bool,
    enable_control_state: Option<ControlState>,
    child_controllers: Option<Vec<Option<Rc<RefCell<dyn Controller>>>>>,
}

impl AppearanceExtension {
    /// Java `AppearanceExtension(Component)`.
    pub fn new(component: Rc<RefCell<ComponentBoundary>>) -> Self {
        Self::new_with(component, true, true, true)
    }

    /// Java `AppearanceExtension(Component, boolean, boolean, boolean)`.
    pub fn new_with(
        component: Rc<RefCell<ComponentBoundary>>,
        enabled_field: bool,
        editable_component: bool,
        editable: bool,
    ) -> Self {
        let orig_foreground = component.borrow().get_foreground().unwrap_or(BLACK);
        let enabled = component.borrow().is_enabled();
        let mut extension = Self {
            component,
            orig_foreground,
            enabled_field,
            editable_component,
            enabled,
            editable,
            flag_type: None,
            allow_flag_editable_control: true,
            allow_foreground_change_on_error: true,
            respond_to_non_error_flags: true,
            debug: false,
            enable_control_state: None,
            child_controllers: None,
        };
        if !enabled_field {
            extension.set_enabled(false);
        }
        if !editable_component {
            extension.set_component_editable(false);
        }
        extension.set_editable(editable);
        extension
    }

    /// Java `setChildControllers(Controller[])`.
    pub fn set_child_controllers(
        &mut self,
        child_controllers: Option<Vec<Option<Rc<RefCell<dyn Controller>>>>>,
    ) {
        self.child_controllers = child_controllers;
        if let Some(child_controllers) = &self.child_controllers {
            for child_controller in child_controllers.iter().flatten() {
                child_controller.borrow_mut().set_editable(self.editable);
                child_controller.borrow_mut().set_enabled(self.enabled);
            }
        }
    }

    /// Java `setAllowFlagEditableControl(boolean)`.
    pub fn set_allow_flag_editable_control(&mut self, allow: bool) {
        self.allow_flag_editable_control = allow;
    }

    /// Java `setAllowForegroundChangeOnError(boolean)`.
    pub fn set_allow_foreground_change_on_error(&mut self, allow: bool) {
        self.allow_foreground_change_on_error = allow;
    }

    /// Java `setRespondToNonErrorFlags(boolean)`.
    pub fn set_respond_to_non_error_flags(&mut self, respond: bool) {
        self.respond_to_non_error_flags = respond;
    }

    /// Java `setDebug(boolean)`.
    pub fn set_debug(&mut self, debug: bool) {
        self.debug = debug;
    }

    /// Java `getFlagType()`.
    pub fn get_flag_type(&self) -> Option<FlagType> {
        self.flag_type
    }

    /// Java `isNativeSetEditable()`.
    pub fn is_native_set_editable(&self) -> bool {
        false
    }

    /// Java `setEditable(boolean)`.
    pub fn set_editable(&mut self, editable: bool) {
        if self.editable == editable {
            return;
        }
        self.editable = editable;
        self.set_component_editable(editable);
        self.set_foreground();
        if let Some(child_controllers) = &self.child_controllers {
            for child_controller in child_controllers.iter().flatten() {
                child_controller.borrow_mut().set_editable(editable);
            }
        }
    }

    /// Java `setComponentEditable(boolean)`.
    pub fn set_component_editable(&mut self, editable: bool) {
        if !editable || self.editable_component {
            self.component.borrow_mut().set_enabled(editable);
        }
    }

    /// Java `setEnableControlState(ControlState)`.
    pub fn set_enable_control_state(&mut self, enable_control_state: Option<ControlState>) {
        self.enable_control_state = enable_control_state;
        self.set_enabled(self.enabled);
    }

    /// Java `setEnabled(boolean)`.
    pub fn set_enabled(&mut self, mut enabled: bool) {
        if self.enable_control_state == Some(ControlState::Enable) {
            enabled = true;
        } else if !self.enabled_field {
            enabled = false;
        }
        if self.enabled == enabled {
            return;
        }
        self.enabled = enabled;
        if !enabled || self.is_native_set_editable() || (self.editable && self.editable_component) {
            self.component.borrow_mut().set_enabled(enabled);
        }
        self.set_foreground();
        if let Some(child_controllers) = &self.child_controllers {
            for child_controller in child_controllers.iter().flatten() {
                child_controller.borrow_mut().set_enabled(enabled);
            }
        }
    }

    /// Java `setForeground()`.
    pub fn set_foreground(&mut self) {
        if let Some(flag_type) = self.flag_type {
            if !flag_type.background
                && (self.allow_foreground_change_on_error || !flag_type.is_error())
            {
                self.component.borrow_mut().set_foreground(flag_type.color);
                return;
            }
        }
        self.component
            .borrow_mut()
            .set_foreground(self.orig_foreground);
    }

    /// Java `isEditable()`.
    pub fn is_editable(&self) -> bool {
        self.editable
    }

    /// Java `isEnabled()`.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
}

impl FlagDisplay for AppearanceExtension {
    /// Java `setFlag(FlagType)`.
    fn set_flag(&mut self, mut flag_type: Option<FlagType>) {
        if !self.respond_to_non_error_flags && flag_type.is_some_and(|flag| !flag.is_error()) {
            flag_type = None;
        }
        if self.flag_type == flag_type {
            return;
        }
        self.flag_type = flag_type;
        self.set_foreground();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestController {
        editable: bool,
        enabled: bool,
    }

    impl Controller for TestController {
        fn is_control(&self) -> bool {
            false
        }
        fn set_editable(&mut self, editable: bool) {
            self.editable = editable;
        }
        fn set_enabled(&mut self, enabled: bool) {
            self.enabled = enabled;
        }
        fn select_file(&mut self) -> Option<std::path::PathBuf> {
            None
        }
        fn select_multiple_files(&mut self) -> Option<Vec<std::path::PathBuf>> {
            None
        }
    }

    #[test]
    fn flag_foreground_respects_error_controls() {
        let component = Rc::new(RefCell::new(ComponentBoundary::default()));
        let mut extension = AppearanceExtension::new(component.clone());
        extension.set_flag(Some(FlagType::TEMPLATE));
        assert_eq!(component.borrow().foreground, Some(FIELD_HIGHLIGHT));
        extension.set_allow_foreground_change_on_error(false);
        extension.set_flag(Some(FlagType::ERROR));
        assert_eq!(component.borrow().foreground, Some(BLACK));
        extension.set_flag(Some(FlagType::WARNING));
        assert_eq!(component.borrow().foreground, Some(BLACK));
    }

    #[test]
    fn disabled_field_can_only_be_enabled_by_enable_control_state() {
        let component = Rc::new(RefCell::new(ComponentBoundary::default()));
        let mut extension = AppearanceExtension::new_with(component.clone(), false, true, true);
        assert!(!extension.is_enabled());
        extension.set_enabled(true);
        assert!(!extension.is_enabled());
        extension.set_enable_control_state(Some(ControlState::Enable));
        assert!(extension.is_enabled());
        assert!(component.borrow().enabled);
    }

    #[test]
    fn child_controllers_receive_current_and_changed_state() {
        let component = Rc::new(RefCell::new(ComponentBoundary::default()));
        let child = Rc::new(RefCell::new(TestController {
            editable: false,
            enabled: false,
        }));
        let mut extension = AppearanceExtension::new(component);
        extension.set_child_controllers(Some(vec![Some(child.clone()), None]));
        assert!(child.borrow().editable);
        assert!(child.borrow().enabled);
        extension.set_editable(false);
        extension.set_enabled(false);
        assert!(!child.borrow().editable);
        assert!(!child.borrow().enabled);
    }
}
