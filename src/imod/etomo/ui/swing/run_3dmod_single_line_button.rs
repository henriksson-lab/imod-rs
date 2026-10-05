//! `IMOD/Etomo/src/etomo/ui/swing/Run3dmodSingleLineButton.java`.
//!
//! A `SingleLineButton` that opens 3dmod, with a right-click `Run3dmodMenu`.
//! Java `extends SingleLineButton implements Deferred3dmodButton,
//! Run3dmodMenuTarget, ContextMenu`: the superclass is field `base` (deref), as in
//! `run_3dmod_button.rs`.  The container is a `Weak<dyn Run3dmodButtonContainer>`
//! (the container owns the button and passes itself while it is being
//! constructed).

use std::cell::{OnceCell, RefCell};
use std::ops::Deref;
use std::rc::{Rc, Weak};

use super::context_menu::ContextMenu;
use super::deferred_3dmod_button::Deferred3dmodButton;
use super::multi_line_button::{MultiLineButton, MultiLineButtonVirtual};
use super::run_3dmod_button_container::Run3dmodButtonContainer;
use super::run_3dmod_menu::Run3dmodMenu;
use super::single_line_button::SingleLineButton;
use super::swing_component::SwingComponent;
use crate::imod::etomo::jdk::{JComponent, MouseEvent};
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::file_key::FileKey;
use crate::imod::etomo::ui::run_3dmod_menu_target::Run3dmodMenuTarget;
use crate::imod::etomo::ui::ui_component::UIComponent;

/// Java package-private `final class Run3dmodSingleLineButton extends
/// SingleLineButton`.
pub struct Run3dmodSingleLineButton {
    /// Java superclass `SingleLineButton`.
    pub base: SingleLineButton,
    /// Java private final `deferred`.  When the button is not a 3dmod button, then it
    /// may run 3dmod deferred; first running the process associated with the button
    /// and then running 3dmod as directed by the right-click menu.
    deferred: bool,
    /// Java private final `run3dmodMenu` (set in the constructor after `super`).
    run_3dmod_menu: OnceCell<Rc<Run3dmodMenu>>,
    /// Java private `container`, initially null.
    container: RefCell<Option<Weak<dyn Run3dmodButtonContainer>>>,
    /// Java private `deferred3dmodButton`, initially null.  When deferred is true,
    /// need a button that knows how to run the 3dmod command.
    deferred_3dmod_button: RefCell<Option<Rc<dyn Deferred3dmodButton>>>,
}

impl Deref for Run3dmodSingleLineButton {
    type Target = SingleLineButton;
    fn deref(&self) -> &SingleLineButton {
        &self.base
    }
}

/// The `SingleLineButton` overrides are inherited.
impl MultiLineButtonVirtual for Run3dmodSingleLineButton {
    fn get_multi_line_button(&self) -> &MultiLineButton {
        &self.base.base
    }
    fn new_button(&self) -> Rc<JComponent> {
        self.base.new_button()
    }
    fn setup_button(&self, set_minimum_size: bool) {
        self.base.setup_button(set_minimum_size)
    }
    fn set_text_label(&self, text: Option<&str>) {
        self.base.set_text_label(text)
    }
}

impl Run3dmodSingleLineButton {
    /// Java private `Run3dmodSingleLineButton(String, Run3dmodButtonContainer, boolean,
    /// DialogType, boolean, String)`.
    fn new(
        label: Option<&str>,
        container: Option<Weak<dyn Run3dmodButtonContainer>>,
        toggle_button: bool,
        dialog_type: Option<DialogType>,
        deferred: bool,
        description: Option<&str>,
    ) -> Rc<Run3dmodSingleLineButton> {
        // super(label, toggleButton, dialogType)
        let instance = Rc::new(Run3dmodSingleLineButton {
            base: SingleLineButton::new_fields(label, toggle_button, dialog_type, false),
            deferred,
            run_3dmod_menu: OnceCell::new(),
            container: RefCell::new(None),
            deferred_3dmod_button: RefCell::new(None),
        });
        MultiLineButton::construct(&instance, false);
        instance.base.constructor_body(label, false);
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

    /// Java static `get3dmodInstance(String, Run3dmodButtonContainer)`.
    pub fn get_3dmod_instance(
        label: Option<&str>,
        container: Option<Weak<dyn Run3dmodButtonContainer>>,
    ) -> Rc<Run3dmodSingleLineButton> {
        let instance = Run3dmodSingleLineButton::new(label, container, false, None, false, None);
        instance.add_listeners();
        instance
    }

    /// Java private `addListeners()`.
    fn add_listeners(self: &Rc<Self>) {
        let context_menu: Weak<dyn ContextMenu> = Rc::downgrade(self) as Weak<dyn ContextMenu>;
        self.base
            .add_mouse_listener(super::generic_mouse_adapter::GenericMouseAdapter::new(context_menu));
    }

    /// Java package-private `getDeferred3dmodButton()`.
    pub fn get_deferred_3dmod_button(&self) -> Option<Rc<dyn Deferred3dmodButton>> {
        self.deferred_3dmod_button.borrow().clone()
    }

    /// Java field read `run3dmodMenu`.
    fn run_3dmod_menu(&self) -> Rc<Run3dmodMenu> {
        self.run_3dmod_menu
            .get()
            .expect("Run3dmodSingleLineButton used before its constructor finished")
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
            container.action(
                self.get_action_command().as_deref().unwrap_or(""),
                self.get_deferred_3dmod_button(),
                Some(menu_options),
            );
        }
    }
}

/// Java `implements Deferred3dmodButton`.
impl Deferred3dmodButton for Run3dmodSingleLineButton {
    fn action(&self, menu_options: Run3dmodMenuOptions) {
        Run3dmodSingleLineButton::action(self, menu_options)
    }
    fn get_output_image_file_key(&self) -> Option<FileKey> {
        self.base.get_output_image_file_key()
    }
}

/// Java `implements Run3dmodMenuTarget`.
impl Run3dmodMenuTarget for Run3dmodSingleLineButton {
    fn menu_action(&self, run_3dmod_menu_options: Run3dmodMenuOptions) {
        Run3dmodSingleLineButton::menu_action(self, run_3dmod_menu_options)
    }
    fn is_enabled(&self) -> bool {
        self.base.is_enabled()
    }
}

/// Java `implements ContextMenu`.
impl ContextMenu for Run3dmodSingleLineButton {
    fn pop_up_context_menu(&self, mouse_event: &MouseEvent) {
        Run3dmodSingleLineButton::pop_up_context_menu(self, mouse_event)
    }
}

/// Java `SwingComponent.getComponent()`, inherited from `MultiLineButton`.
impl SwingComponent for Run3dmodSingleLineButton {
    fn get_component(&self) -> Rc<JComponent> {
        self.base.base.get_component()
    }
}

/// Java `UIComponent`, inherited from `MultiLineButton`.
impl UIComponent for Run3dmodSingleLineButton {
    fn get_ui_component(&self) -> &dyn SwingComponent {
        self
    }
    fn get_component(&self) -> Rc<JComponent> {
        self.base.base.get_component()
    }
}
