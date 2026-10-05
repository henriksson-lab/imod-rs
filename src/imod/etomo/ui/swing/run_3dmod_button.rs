//! `IMOD/Etomo/src/etomo/ui/swing/Run3dmodButton.java`.
//!
//! A `MultiLineButton` that opens 3dmod, or (deferred) runs its process and
//! opens 3dmod afterwards as chosen from its right-click `Run3dmodMenu`.
//!
//! Java `extends MultiLineButton`: the superclass is field `base` (deref);
//! see `multi_line_button.rs` for the object model.  The container is held as
//! a `Weak<dyn Run3dmodButtonContainer>` (the container owns the button and
//! usually passes itself while it is being constructed).

use std::cell::{Cell, OnceCell, RefCell};
use std::ops::Deref;
use std::rc::{Rc, Weak};

use crate::imod::etomo::jdk::{JComponent, MouseEvent};
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::file_key::FileKey;
use crate::imod::etomo::ui::run_3dmod_menu_target::Run3dmodMenuTarget;
use crate::imod::etomo::ui::ui_component::UIComponent;

use super::binned_xy_3dmod_button::BinnedXY3dmodButton;
use super::context_menu::ContextMenu;
use super::deferred_3dmod_button::Deferred3dmodButton;
use super::multi_line_button::{MultiLineButton, MultiLineButtonVirtual};
use super::run_3dmod_button_container::Run3dmodButtonContainer;
use super::run_3dmod_menu::Run3dmodMenu;
use super::swing_component::SwingComponent;

/// Java public final `Run3dmodButton`.
pub struct Run3dmodButton {
    /// Java superclass `MultiLineButton`.
    pub base: MultiLineButton,
    /// Java final `deferred`.  When the button is not a 3dmod button, then it
    /// may run 3dmod deferred; first running the process associated with the
    /// button and then running 3dmod as directed by the right-click menu.  The
    /// right click menu contains a plain 3dmod option (noMenuOption) when
    /// deferred is true.
    deferred: bool,
    /// Java final `run3dmodMenu` (set in the constructor, after `super`).
    run_3dmod_menu: OnceCell<Rc<Run3dmodMenu>>,
    /// Java `container`.
    container: RefCell<Option<Weak<dyn Run3dmodButtonContainer>>>,
    /// Java `deferred3dmodButton`.  When deferred is true, need a button that
    /// knows how to run the 3dmod command.
    deferred_3dmod_button: RefCell<Option<Rc<dyn Deferred3dmodButton>>>,
    /// Java `processOutputFileKnown`.  Unused in the Java class.
    process_output_file_known: Cell<bool>,
}

impl Deref for Run3dmodButton {
    type Target = MultiLineButton;
    fn deref(&self) -> &MultiLineButton {
        &self.base
    }
}

/// Run3dmodButton overrides no `MultiLineButton` virtual method (its
/// `setActionCommand` override only calls `super.setActionCommand`).
impl MultiLineButtonVirtual for Run3dmodButton {
    fn get_multi_line_button(&self) -> &MultiLineButton {
        &self.base
    }
}

impl Run3dmodButton {
    /// Java private
    /// `Run3dmodButton(String, Run3dmodButtonContainer, boolean, DialogType, boolean, String, FileKey)`.
    fn new(
        label: Option<&str>,
        container: Option<Weak<dyn Run3dmodButtonContainer>>,
        toggle_button: bool,
        dialog_type: Option<DialogType>,
        deferred: bool,
        description: Option<&str>,
        output_image_file_key: Option<FileKey>,
    ) -> Rc<Run3dmodButton> {
        let instance = Rc::new(Run3dmodButton {
            base: MultiLineButton::new_fields(
                label,
                toggle_button,
                dialog_type,
                false,
                false,
                output_image_file_key,
            ),
            deferred,
            run_3dmod_menu: OnceCell::new(),
            container: RefCell::new(None),
            deferred_3dmod_button: RefCell::new(None),
            process_output_file_known: Cell::new(true),
        });
        // Java `super(label, toggleButton, dialogType, false, false, false, outputImageFileKey)`.
        MultiLineButton::construct(&instance, false);
        *instance.container.borrow_mut() = container;
        let target: Rc<dyn Run3dmodMenuTarget> = instance.clone();
        let run_3dmod_menu = if deferred {
            Run3dmodMenu::get_process_button_instance(target, description)
        } else {
            Run3dmodMenu::get_3dmod_button_instance(target, description)
        };
        let _ = instance.run_3dmod_menu.set(run_3dmod_menu);
        instance
    }

    /// Java public static
    /// `get3dmodInstance(String, Run3dmodButtonContainer, FileKey)`.
    pub fn get_3dmod_instance_string_run_3dmod_button_container_file_key(
        label: Option<&str>,
        container: Option<Weak<dyn Run3dmodButtonContainer>>,
        output_image_file_key: Option<FileKey>,
    ) -> Rc<Run3dmodButton> {
        let instance = Run3dmodButton::new(
            label,
            container,
            false,
            None,
            false,
            None,
            output_image_file_key,
        );
        instance.add_listeners();
        instance
    }

    /// Java public static `get3dmodInstance(String, Run3dmodButtonContainer)`.
    pub fn get_3dmod_instance_string_run_3dmod_button_container(
        label: Option<&str>,
        container: Option<Weak<dyn Run3dmodButtonContainer>>,
    ) -> Rc<Run3dmodButton> {
        let instance = Run3dmodButton::new(label, container, false, None, false, None, None);
        instance.add_listeners();
        instance
    }

    /// Java static `get3dmodInstance(String)`.
    pub fn get_3dmod_instance_string(label: Option<&str>) -> Rc<Run3dmodButton> {
        let instance = Run3dmodButton::new(label, None, false, None, false, None, None);
        instance.add_listeners();
        instance
    }

    /// Java static `getToggle3dmodInstance(String, DialogType)`.
    pub fn get_toggle_3dmod_instance(
        label: Option<&str>,
        dialog_type: Option<DialogType>,
    ) -> Rc<Run3dmodButton> {
        let instance = Run3dmodButton::new(label, None, true, dialog_type, false, None, None);
        instance.add_listeners();
        instance
    }

    /// Java public static `getDeferred3dmodInstance(String, Run3dmodButtonContainer)`.
    pub fn get_deferred_3dmod_instance_string_run_3dmod_button_container(
        label: Option<&str>,
        container: Option<Weak<dyn Run3dmodButtonContainer>>,
    ) -> Rc<Run3dmodButton> {
        let instance = Run3dmodButton::new(label, container, false, None, true, None, None);
        instance.add_listeners();
        instance
    }

    /// Java static `getDeferred3dmodInstance(String)`.
    pub fn get_deferred_3dmod_instance_string(label: Option<&str>) -> Rc<Run3dmodButton> {
        let instance = Run3dmodButton::new(label, None, false, None, true, None, None);
        instance.add_listeners();
        instance
    }

    /// Java static
    /// `getDeferred3dmodInstance(String, Run3dmodButtonContainer, String)`.
    pub fn get_deferred_3dmod_instance_string_run_3dmod_button_container_string(
        label: Option<&str>,
        container: Option<Weak<dyn Run3dmodButtonContainer>>,
        description: Option<&str>,
    ) -> Rc<Run3dmodButton> {
        let instance =
            Run3dmodButton::new(label, container, false, None, true, description, None);
        instance.add_listeners();
        instance
    }

    /// Java static `getDeferredToggle3dmodInstance(String)`.
    pub fn get_deferred_toggle_3dmod_instance_string(label: Option<&str>) -> Rc<Run3dmodButton> {
        let instance = Run3dmodButton::new(label, None, true, None, true, None, None);
        instance.add_listeners();
        instance
    }

    /// Java public static `getDeferredToggle3dmodInstance(String, DialogType)`.
    pub fn get_deferred_toggle_3dmod_instance_string_dialog_type(
        label: Option<&str>,
        dialog_type: Option<DialogType>,
    ) -> Rc<Run3dmodButton> {
        let instance = Run3dmodButton::new(label, None, true, dialog_type, true, None, None);
        instance.add_listeners();
        instance
    }

    /// Java private `addListeners()`.
    fn add_listeners(self: &Rc<Self>) {
        let context_menu: Weak<dyn ContextMenu> = Rc::downgrade(self) as Weak<dyn ContextMenu>;
        self.base
            .add_mouse_listener(super::generic_mouse_adapter::GenericMouseAdapter::new(context_menu));
    }

    /// Java public `setDeferred3dmodButton(Deferred3dmodButton)`.
    pub fn set_deferred_3dmod_button_deferred_3dmod_button(
        &self,
        input: Option<Rc<dyn Deferred3dmodButton>>,
    ) {
        if input.is_none() && self.deferred {
            // Java throws `NullPointerException("A deferred instance needs to
            // have a deferred3dmodButton.")`, which Swing reports on the EDT
            // and survives, with nothing assigned.  Fixed in translation: the
            // message is reported and the call leaves the button unchanged,
            // without unwinding the UI thread.
            eprintln!(
                "java.lang.NullPointerException: A deferred instance needs to have a deferred3dmodButton."
            );
            return;
        }
        *self.deferred_3dmod_button.borrow_mut() = input.clone();
        if let Some(deferred_3dmod_button) = input {
            self.set_output_image_file_key(deferred_3dmod_button.get_output_image_file_key());
        }
    }

    /// Java `setFileToOpenKnown(boolean)`.
    pub fn set_file_to_open_known(&self, file_to_open_known: bool) {
        self.run_3dmod_menu()
            .set_file_to_open_known(file_to_open_known);
    }

    /// Java `setDeferred3dmodButton(BinnedXY3dmodButton)`.
    pub fn set_deferred_3dmod_button_binned_xy_3dmod_button(
        &self,
        input: Option<&BinnedXY3dmodButton>,
    ) {
        if input.is_none() && self.deferred {
            // Java throws `NullPointerException` here; see
            // `set_deferred_3dmod_button_deferred_3dmod_button`.
            eprintln!(
                "java.lang.NullPointerException: A deferred instance needs to have a deferred3dmodButton."
            );
            return;
        }
        // Upstream bug fixed (Run3dmodButton.java, setDeferred3dmodButton(
        // BinnedXY3dmodButton)): for a non-deferred button and a null input
        // Java calls `input.getButton()` and throws NullPointerException.  A
        // null input clears the deferred button, as the other overload does.
        *self.deferred_3dmod_button.borrow_mut() = input.map(|input| input.get_button());
    }

    /// Java public `getDeferred3dmodButton()`.
    pub fn get_deferred_3dmod_button(&self) -> Option<Rc<dyn Deferred3dmodButton>> {
        self.deferred_3dmod_button.borrow().clone()
    }

    /// Java public `setContainer(Run3dmodButtonContainer)`.
    pub fn set_container(&self, container: Option<Weak<dyn Run3dmodButtonContainer>>) {
        *self.container.borrow_mut() = container;
    }

    /// Java field `run3dmodMenu`.
    fn run_3dmod_menu(&self) -> Rc<Run3dmodMenu> {
        self.run_3dmod_menu
            .get()
            .expect("Run3dmodButton used before its constructor finished")
            .clone()
    }

    /// Java `popUpContextMenu(MouseEvent)`.
    pub fn pop_up_context_menu(&self, mouse_event: &MouseEvent) {
        self.run_3dmod_menu().pop_up_context_menu(mouse_event);
    }

    /// Java `menuAction(Run3dmodMenuOptions)`.
    pub fn menu_action(&self, run_3dmod_menu_options: Run3dmodMenuOptions) {
        self.action(run_3dmod_menu_options);
        if self.is_toggle_button() {
            self.set_selected(true);
        }
    }

    /// Java `action(Run3dmodMenuOptions)`.
    pub fn action(&self, menu_options: Run3dmodMenuOptions) {
        let container = self.container.borrow().as_ref().and_then(Weak::upgrade);
        if let Some(container) = container {
            // Java passes `getActionCommand()`, which is the button text when no
            // command was set, so it is never null in practice.
            container.action(
                self.get_action_command().as_deref().unwrap_or(""),
                self.get_deferred_3dmod_button(),
                Some(menu_options),
            );
        }
    }
}

/// Java `implements Deferred3dmodButton`.
impl Deferred3dmodButton for Run3dmodButton {
    fn action(&self, menu_options: Run3dmodMenuOptions) {
        Run3dmodButton::action(self, menu_options)
    }
    fn get_output_image_file_key(&self) -> Option<FileKey> {
        self.base.get_output_image_file_key()
    }
}

/// Java `implements Run3dmodMenuTarget`.
impl Run3dmodMenuTarget for Run3dmodButton {
    fn menu_action(&self, run_3dmod_menu_options: Run3dmodMenuOptions) {
        Run3dmodButton::menu_action(self, run_3dmod_menu_options)
    }
    fn is_enabled(&self) -> bool {
        self.base.is_enabled()
    }
}

/// Java `implements ContextMenu`.
impl ContextMenu for Run3dmodButton {
    fn pop_up_context_menu(&self, mouse_event: &MouseEvent) {
        Run3dmodButton::pop_up_context_menu(self, mouse_event)
    }
}

/// Java `SwingComponent.getComponent()`, inherited from `MultiLineButton`
/// (and required by `Run3dmodMenuTarget extends SwingComponent`).
impl SwingComponent for Run3dmodButton {
    fn get_component(&self) -> Rc<JComponent> {
        self.base.get_component()
    }
}

/// Java `UIComponent`, inherited from `MultiLineButton`.
impl UIComponent for Run3dmodButton {
    fn get_ui_component(&self) -> &dyn SwingComponent {
        self
    }
    fn get_component(&self) -> Rc<JComponent> {
        self.base.get_component()
    }
}
