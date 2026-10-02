//! `IMOD/Etomo/src/etomo/ui/swing/AppearanceExtension.java`.
//!
//! Controls how an efield's component shows the field's enabled, editable and flag
//! states.  Java `AppearanceExtension` is subclassed by `TextComponentAppearanceExtension`,
//! which overrides `setComponentEditable` and `isNativeSetEditable`; those two are in
//! [`AppearanceExtensionVirtual`], and every call from this class dispatches through
//! `this` (the most derived object).
//!
//! Java calls `setComponentEditable` from the constructor, where it already dispatches
//! to the subclass.  So construction is split: [`AppearanceExtension::construct`]
//! assigns the fields (the Java field initialisers and the constructor's assignments),
//! the subclass stores itself with [`AppearanceExtension::set_this`], and
//! [`AppearanceExtension::constructor_body`] runs the rest of the Java constructor.  The
//! `new_*` constructors do all three for a plain `AppearanceExtension`.

use std::cell::{Cell, RefCell};
use std::rc::{Rc, Weak};

use super::control_state::{self, ControlState};
use super::controller::Controller;
use crate::imod::etomo::jdk::{Color, JComponent};
use crate::imod::etomo::ui::flag_display::FlagDisplay;
use crate::imod::etomo::ui::flag_type::FlagType;

/// The methods of Java `AppearanceExtension` that a subclass overrides.
pub trait AppearanceExtensionVirtual {
    /// The embedded `AppearanceExtension`.
    fn appearance_extension(&self) -> &AppearanceExtension;

    /// Java `isNativeSetEditable()`.  Inherit and return true for components which have
    /// a setEditable function that should be used.
    fn is_native_set_editable(&self) -> bool {
        false
    }

    /// Java `setComponentEditable(boolean)`.  Make the component enabled or disabled.
    fn set_component_editable(&self, editable: bool) {
        let base = self.appearance_extension();
        // Using the component's enable/disable function to show editability as well as
        // enableness.
        // Ineditable components are never editable, even when the field is editable.
        if !editable || base.editable_component {
            base.component.set_enabled(editable);
        }
    }
}

/// Java `AppearanceExtension`.
pub struct AppearanceExtension {
    /// The most derived object, for the overridable methods.
    this: RefCell<Weak<dyn AppearanceExtensionVirtual>>,
    /// Java `component`.
    pub component: Rc<JComponent>,
    /// Java `origForeground`.
    orig_foreground: Color,
    /// Java `enabledField`.
    enabled_field: bool,
    /// Java `editableComponent`.
    pub editable_component: bool,
    /// Java `enabled`.
    enabled: Cell<bool>,
    /// Java `editable`.
    editable: Cell<bool>,
    /// Java `flagType`.
    flag_type: Cell<Option<&'static FlagType>>,
    /// Java `allowFlagEditableControl` (set, never read, in the Java too).
    allow_flag_editable_control: Cell<bool>,
    /// Java `allowForegroundChangeOnError`.
    allow_foreground_change_on_error: Cell<bool>,
    /// Java `respondToNonErrorFlags`.
    respond_to_non_error_flags: Cell<bool>,
    /// Java `debug`.
    debug: Cell<bool>,
    /// Java `enableControlState`.  Can enable a field that can't be enabled.
    enable_control_state: Cell<Option<&'static ControlState>>,
    /// Java `childControllers`.
    child_controllers: RefCell<Option<Vec<Option<Rc<dyn Controller>>>>>,
}

impl AppearanceExtensionVirtual for AppearanceExtension {
    fn appearance_extension(&self) -> &AppearanceExtension {
        self
    }
}

impl AppearanceExtension {
    /// Java `AppearanceExtension(Component)`.
    pub fn new_component(component: &Rc<JComponent>) -> Rc<AppearanceExtension> {
        AppearanceExtension::new_component_boolean_boolean_boolean(component, true, true, true)
    }

    /// Java `AppearanceExtension(Component, boolean, boolean, boolean)`.
    pub fn new_component_boolean_boolean_boolean(
        component: &Rc<JComponent>,
        enabled_field: bool,
        editable_component: bool,
        editable: bool,
    ) -> Rc<AppearanceExtension> {
        let extension = Rc::new(AppearanceExtension::construct(
            component,
            enabled_field,
            editable_component,
            editable,
        ));
        extension.set_this(Rc::downgrade(&extension) as Weak<dyn AppearanceExtensionVirtual>);
        extension.constructor_body();
        extension
    }

    /// The field assignments of Java `AppearanceExtension(Component, boolean, boolean,
    /// boolean)` (for a subclass; see the module documentation).
    pub fn construct(
        component: &Rc<JComponent>,
        enabled_field: bool,
        editable_component: bool,
        editable: bool,
    ) -> AppearanceExtension {
        let color = component.get_foreground();
        let orig_foreground = match color {
            Some(color) => color,
            // Color.BLACK
            None => (0, 0, 0),
        };
        AppearanceExtension {
            this: RefCell::new(Weak::<AppearanceExtension>::new() as Weak<dyn AppearanceExtensionVirtual>),
            component: component.clone(),
            orig_foreground,
            enabled_field,
            editable_component,
            enabled: Cell::new(true),
            editable: Cell::new(editable),
            flag_type: Cell::new(None),
            allow_flag_editable_control: Cell::new(true),
            allow_foreground_change_on_error: Cell::new(true),
            respond_to_non_error_flags: Cell::new(true),
            debug: Cell::new(false),
            enable_control_state: Cell::new(None),
            child_controllers: RefCell::new(None),
        }
    }

    /// Stores the most derived object (Java `this`).
    pub fn set_this(&self, this: Weak<dyn AppearanceExtensionVirtual>) {
        *self.this.borrow_mut() = this;
    }

    /// The statements of the Java constructor after its field assignments.
    pub fn constructor_body(&self) {
        // Get the field enabled setting from the component
        self.enabled.set(self.component.is_enabled());
        if !self.enabled_field {
            // Disable the field
            self.set_enabled(false);
        }
        if !self.editable_component {
            // Make the component ineditable
            let this = self.this.borrow().upgrade();
            match this {
                Some(this) => this.set_component_editable(false),
                None => AppearanceExtensionVirtual::set_component_editable(self, false),
            }
        }
        // Get the field editable setting from the parameter
        self.set_editable(self.editable.get());
    }

    /// Java final `setChildControllers(Controller[])`.
    pub fn set_child_controllers(&self, child_controllers: Option<Vec<Option<Rc<dyn Controller>>>>) {
        *self.child_controllers.borrow_mut() = child_controllers.clone();
        if let Some(child_controllers) = child_controllers {
            for child_controller in child_controllers.iter().flatten() {
                child_controller.set_editable(self.editable.get());
                child_controller.set_enabled(self.enabled.get());
            }
        }
    }

    /// Java final `setAllowFlagEditableControl(boolean)`.
    pub fn set_allow_flag_editable_control(&self, allow: bool) {
        self.allow_flag_editable_control.set(allow);
    }

    /// Java final `setAllowForegroundChangeOnError(boolean)`.
    pub fn set_allow_foreground_change_on_error(&self, allow: bool) {
        self.allow_foreground_change_on_error.set(allow);
    }

    /// Java final `setRespondToNonErrorFlags(boolean)`.
    pub fn set_respond_to_non_error_flags(&self, respond: bool) {
        self.respond_to_non_error_flags.set(respond);
    }

    /// Java final `setDebug(boolean)`.
    pub fn set_debug(&self, debug: bool) {
        self.debug.set(debug);
    }

    /// Java final `getFlagType()`.
    pub fn get_flag_type(&self) -> Option<&'static FlagType> {
        self.flag_type.get()
    }

    /// Java final `setEditable(boolean)`.  Make a field editable or ineditable.
    pub fn set_editable(&self, editable: bool) {
        // Ineditable components are never editable but the entire field (including file
        // and clear buttons) can be editable. So editableComponent is ignored here.
        if self.editable.get() == editable {
            return;
        }
        self.editable.set(editable);
        let this = self.this.borrow().upgrade();
        match this {
            Some(this) => this.set_component_editable(editable),
            None => AppearanceExtensionVirtual::set_component_editable(self, editable),
        }
        self.set_foreground();
        let child_controllers = self.child_controllers.borrow().clone();
        if let Some(child_controllers) = child_controllers {
            for child_controller in child_controllers.iter().flatten() {
                child_controller.set_editable(editable);
            }
        }
    }

    /// Java final `setEnableControlState(ControlState)`.
    pub fn set_enable_control_state(&self, enable_control_state: Option<&'static ControlState>) {
        self.enable_control_state.set(enable_control_state);
        self.set_enabled(self.enabled.get());
    }

    /// Java final `setEnabled(boolean)`.  Make a field enabled or disabled.
    pub fn set_enabled(&self, mut enabled: bool) {
        // ControlState.ENABLE keeps the component enabled, even if its a disabled field
        if self
            .enable_control_state
            .get()
            .is_some_and(|state| std::ptr::eq(state, &*control_state::ENABLE))
        {
            enabled = true;
        } else if !self.enabled_field {
            // Prevent a disabled field from being enabled.
            enabled = false;
        }
        if self.enabled.get() == enabled {
            return;
        }
        self.enabled.set(enabled);
        // If this component doesn't have a native editable setting, then component's
        // enabled/disabled functionality is being used to show both states. In this
        // case don't call the components's setEnabled function with TRUE while the field
        // is ineditable or the component is not allowed to be editable.
        let this = self.this.borrow().upgrade();
        let native_set_editable = match this {
            Some(this) => this.is_native_set_editable(),
            None => AppearanceExtensionVirtual::is_native_set_editable(self),
        };
        if !enabled || native_set_editable || (self.editable.get() && self.editable_component) {
            self.component.set_enabled(enabled);
        }
        self.set_foreground();
        let child_controllers = self.child_controllers.borrow().clone();
        if let Some(child_controllers) = child_controllers {
            for child_controller in child_controllers.iter().flatten() {
                child_controller.set_enabled(enabled);
            }
        }
    }

    /// Java final `setForeground()`.  Set the foreground.  This class is not currently
    /// set up to handle background changes.
    pub fn set_foreground(&self) {
        match self.flag_type.get() {
            Some(flag_type)
                if !flag_type.background
                    && (self.allow_foreground_change_on_error.get() || !flag_type.is_error()) =>
            {
                self.component.set_foreground(Some(flag_type.color));
            }
            _ => self.component.set_foreground(Some(self.orig_foreground)),
        }
    }

    /// Java final `isEditable()`.
    pub fn is_editable(&self) -> bool {
        self.editable.get()
    }

    /// Java final `isEnabled()`.
    pub fn is_enabled(&self) -> bool {
        self.enabled.get()
    }
}

impl FlagDisplay for AppearanceExtension {
    /// Java final `setFlag(FlagType)`.
    fn set_flag(&self, mut flag_type: Option<&'static FlagType>) {
        if !self.respond_to_non_error_flags.get()
            && flag_type.is_some_and(|flag_type| !flag_type.is_error())
        {
            // Respond to non-error flags by turning off the flag.
            flag_type = None;
        }
        let same = match (self.flag_type.get(), flag_type) {
            (None, None) => true,
            (Some(a), Some(b)) => std::ptr::eq(a, b),
            _ => false,
        };
        if same {
            return;
        }
        self.flag_type.set(flag_type);
        self.set_foreground();
    }
}
