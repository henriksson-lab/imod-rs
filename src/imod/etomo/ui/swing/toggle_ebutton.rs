//! `IMOD/Etomo/src/etomo/ui/swing/ToggleEbutton.java`.
//!
//! A `JToggleButton` that acts as a controller: when it has a control mode, toggling
//! it sends a control event to its target through the `ControlMediator` (the override
//! toggle of a directive row).
//!
//! The control target is held as a `Weak<dyn ControlTarget>`, as in `Ebutton`: the
//! target field and this button are owned by the same row.  The `fixedSize`
//! constructor argument is layout (`// Swing layout:`), and
//! `add(JPanel, GridBagLayout, GridBagConstraints)` keeps the panel and drops the
//! layout arguments.

use std::cell::{Cell, RefCell};
use std::path::PathBuf;
use std::rc::{Rc, Weak};

use super::appearance_extension::AppearanceExtension;
use super::button_style_extension::ButtonStyleExtensionVirtual;
use super::control_mediator::{self, ControlMediator};
use super::control_mode::ControlMode;
use super::control_state;
use super::control_target::ControlTarget;
use super::controller::Controller;
use super::grid_bag_extension::GridBagExtension;
use super::x_button_style_extension::XButtonStyleExtension;
use crate::imod::etomo::etomo_director::ARGUMENTS;
use crate::imod::etomo::jdk::{ActionEvent, JComponent};
use crate::imod::etomo::storage::autodoc::autodoc_tokenizer::{DEFAULT_DELIMITER, SEPARATOR_CHAR};
use crate::imod::etomo::r#type::ui_test_field_type::UITestFieldType;
use crate::imod::etomo::ui::boolean_efield_interface::BooleanEfieldInterface;
use crate::imod::etomo::ui::flag_display::FlagDisplay;
use crate::imod::etomo::ui::flag_type::FlagType;
use crate::imod::etomo::util::utilities;

/// Java package-private `final class ToggleEbutton implements ActionListener,
/// BooleanEfieldInterface, FlagDisplay, Controller`.
pub struct ToggleEbutton {
    /// This object, for `button.addActionListener(this)`.
    self_ref: RefCell<Weak<ToggleEbutton>>,
    /// Java final `button` (`new JToggleButton()`).
    button: Rc<JComponent>,
    /// Java final `target`.
    target: Option<Weak<dyn ControlTarget>>,
    /// Java final `controlMode`.
    control_mode: Option<&'static ControlMode>,
    /// Java final `buttonStyle`.
    button_style: Option<Rc<dyn ButtonStyleExtensionVirtual>>,
    /// Java `gridBagExtension`.
    grid_bag_extension: RefCell<Option<Rc<GridBagExtension>>>,
    /// Java `controlMediator`.  Never assigned or read in the Java class.
    #[allow(dead_code)]
    control_mediator: Cell<Option<&'static ControlMediator>>,
    /// Java `appearanceExtension`.
    appearance_extension: RefCell<Option<Rc<AppearanceExtension>>>,
    /// Java `debug`.
    debug: Cell<bool>,
}

impl ToggleEbutton {
    /// Java private `ToggleEbutton(ControlMode, ControlTarget, Dimension,
    /// ButtonStyleExtension)`.  `fixedSize` is layout; `has_fixed_size` says whether
    /// the Java caller passed one.
    fn new(
        control_mode: Option<&'static ControlMode>,
        target: Option<Weak<dyn ControlTarget>>,
        has_fixed_size: bool,
        button_style: Option<Rc<dyn ButtonStyleExtensionVirtual>>,
    ) -> Rc<ToggleEbutton> {
        let instance = Rc::new(ToggleEbutton {
            self_ref: RefCell::new(Weak::new()),
            button: JComponent::new_toggle_button(""),
            target,
            control_mode,
            button_style,
            grid_bag_extension: RefCell::new(None),
            control_mediator: Cell::new(None),
            appearance_extension: RefCell::new(None),
            debug: Cell::new(false),
        });
        *instance.self_ref.borrow_mut() = Rc::downgrade(&instance);
        instance.set_name(instance.target.as_ref(), control_mode);
        if has_fixed_size {
            // Swing layout: button.setPreferredSize(fixedSize);
            // button.setMaximumSize(fixedSize).
        }
        if let Some(button_style) = &instance.button_style {
            button_style.setup(Some(&instance.button), None, false);
        }
        instance
    }

    /// Java static `getOverrideInstance(ControlTarget)`.
    pub fn get_override_instance(target: Option<Weak<dyn ControlTarget>>) -> Rc<ToggleEbutton> {
        // new ToggleEbutton(ControlState.OVERRIDE, target, FixedDim.INLINE_SQUARE_SIZE,
        // XButtonStyleExtension.getInstance())
        let instance = ToggleEbutton::new(
            Some(&**control_state::OVERRIDE),
            target,
            true,
            Some(XButtonStyleExtension::get_instance() as Rc<dyn ButtonStyleExtensionVirtual>),
        );
        instance.create_panel();
        instance
    }

    /// Java private `setName(ControlTarget, ControlMode)`.
    fn set_name(
        &self,
        target: Option<&Weak<dyn ControlTarget>>,
        control_mode: Option<&'static ControlMode>,
    ) {
        // build name
        let mut name: Option<String> = None;
        let control_name_set = match control_mode {
            Some(control_mode) => control_mode.has_field_name(),
            None => false,
        };
        let field_type = if control_name_set {
            UITestFieldType::CONTROL_TOGGLE_BUTTON
        } else {
            UITestFieldType::BUTTON
        };
        // Java: `if (target != null)`.  A target that has already been dropped is
        // treated as null.
        if let Some(target) = target.and_then(Weak::upgrade) {
            name = utilities::convert_label_to_name(
                target.get_label().as_deref(),
                field_type.is_unlimited_segments(),
            );
        }
        if control_name_set {
            name = control_mode.unwrap().append_to_name(name.as_deref());
        }
        let Some(name) = name else {
            return;
        };
        // set the name in the field
        self.button
            .set_name(Some(&format!("{}{}{}", field_type, SEPARATOR_CHAR, name)));
        if ARGUMENTS.lock().unwrap().is_print_names() {
            println!(
                "{} {} ",
                self.button.get_name().as_deref().unwrap_or("null"),
                DEFAULT_DELIMITER
            );
        }
    }

    /// Java private `createPanel()`.
    fn create_panel(&self) {
        if self.control_mode.is_some() {
            // button.addActionListener(this)
            let this = self.self_ref.borrow().clone();
            self.button.add_action_listener(Rc::new(move |event| {
                if let Some(this) = this.upgrade() {
                    this.action_performed(Some(event));
                }
            }));
        }
    }

    /// Java `getComponent()`.
    pub fn get_component(&self) -> Rc<JComponent> {
        self.button.clone()
    }

    /// Java `@Override isControl()`.
    pub fn is_control(&self) -> bool {
        self.is_selected() && self.is_enabled()
    }

    /// Java `@Override selectFile()`.
    pub fn select_file(&self) -> Option<PathBuf> {
        None
    }

    /// Java `@Override isSelected()`.
    pub fn is_selected(&self) -> bool {
        self.button.is_selected()
    }

    /// Java `isEnabled()`.
    pub fn is_enabled(&self) -> bool {
        self.button.is_enabled()
    }

    /// Java `setDebug(boolean)`.
    pub fn set_debug(&self, debug: bool) {
        self.debug.set(debug);
    }

    /// Java `@Override setSelected(boolean)`.
    pub fn set_selected(&self, selected: bool) {
        self.button.set_selected(selected);
        self.action_performed(None);
    }

    /// Java `setVisible(boolean)`.
    pub fn set_visible(&self, visible: bool) {
        self.button.set_visible(visible);
    }

    // appearanceExtension

    /// Java private `createAppearanceExtension()`.
    fn create_appearance_extension(&self) {
        if self.appearance_extension.borrow().is_none() {
            *self.appearance_extension.borrow_mut() =
                Some(AppearanceExtension::new_component(&self.button));
        }
    }

    /// Java `@Override setEnabled(boolean)`.
    pub fn set_enabled(&self, enabled: bool) {
        let appearance_extension = self.appearance_extension.borrow().clone();
        if let Some(appearance_extension) = appearance_extension {
            appearance_extension.set_enabled(enabled);
        } else {
            self.button.set_enabled(enabled);
        }
    }

    /// Java `@Override setEditable(boolean)`.
    pub fn set_editable(&self, editable: bool) {
        self.create_appearance_extension();
        let appearance_extension = self.appearance_extension.borrow().clone().unwrap();
        appearance_extension.set_editable(editable);
    }

    /// Java `@Override setFlag(FlagType)`.
    pub fn set_flag(&self, flag_type: Option<&'static FlagType>) {
        if let Some(button_style) = &self.button_style {
            button_style.update_appearance(&self.button, flag_type, false, self.is_selected());
        }
        self.create_appearance_extension();
        let appearance_extension = self.appearance_extension.borrow().clone().unwrap();
        appearance_extension.set_flag(flag_type);
    }

    /// Java `@Override actionPerformed(ActionEvent)`.  `event` is ignored.
    pub fn action_performed(&self, _event: Option<&ActionEvent>) {
        if let Some(control_mode) = self.control_mode {
            let target = self.target.as_ref().and_then(Weak::upgrade);
            control_mediator::INSTANCE.control_event_controller_control_target_control_mode(
                self,
                target.as_deref(),
                Some(control_mode),
            );
        }
    }

    // gridBagExtension

    /// Java `remove()`.
    pub fn remove(&self) {
        let grid_bag_extension = self.grid_bag_extension.borrow().clone();
        if let Some(grid_bag_extension) = grid_bag_extension {
            grid_bag_extension.remove(&self.get_component());
        }
    }

    /// Java `add(JPanel, GridBagLayout, GridBagConstraints)`.  The layout and
    /// constraints are not modelled.
    pub fn add(&self, panel: &Rc<JComponent>) {
        if self.grid_bag_extension.borrow().is_none() {
            *self.grid_bag_extension.borrow_mut() = Some(GridBagExtension::new());
        }
        let grid_bag_extension = self.grid_bag_extension.borrow().clone().unwrap();
        grid_bag_extension.add(&self.get_component(), panel);
    }

    /// Java `@Override selectMultipleFiles()`.
    pub fn select_multiple_files(&self) -> Option<Vec<PathBuf>> {
        // TODO Auto-generated method stub
        None
    }
}

impl BooleanEfieldInterface for ToggleEbutton {
    fn is_selected(&self) -> bool {
        ToggleEbutton::is_selected(self)
    }
    fn set_selected(&self, selected: bool) {
        ToggleEbutton::set_selected(self, selected)
    }
}

impl FlagDisplay for ToggleEbutton {
    fn set_flag(&self, flag_type: Option<&'static FlagType>) {
        ToggleEbutton::set_flag(self, flag_type)
    }
}

impl Controller for ToggleEbutton {
    fn is_control(&self) -> bool {
        ToggleEbutton::is_control(self)
    }
    fn set_editable(&self, editable: bool) {
        ToggleEbutton::set_editable(self, editable)
    }
    fn set_enabled(&self, enabled: bool) {
        ToggleEbutton::set_enabled(self, enabled)
    }
    fn select_file(&self) -> Option<PathBuf> {
        ToggleEbutton::select_file(self)
    }
    fn select_multiple_files(&self) -> Option<Vec<PathBuf>> {
        ToggleEbutton::select_multiple_files(self)
    }
}
