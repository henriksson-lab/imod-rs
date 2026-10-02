//! `IMOD/Etomo/src/etomo/ui/swing/MinibuttonCell.java`.
//!
//! A table cell holding a square `Minibutton`, optionally with a 3dmod right-click
//! menu that reports its choice to a `Run3dmodButtonContainer`.
//!
//! Java `final class MinibuttonCell extends InputCell implements UIComponent,
//! SwingComponent, Run3dmodMenuTarget, ContextMenu, FieldSettings`.  Every Java method
//! body is an inherent method; the trait impls at the end bind `CellVirtual`,
//! `InputCellVirtual` and the interfaces to them.  The Java constructor passes `this`
//! to `Run3dmodMenu`, which needs the finished `Rc`; so the final `contextMenu` field
//! is a `OnceCell` set right after allocation, before the rest of the constructor
//! body.  Icons are represented by their image names (see `complete_icon.rs`).

use std::cell::{Cell, OnceCell};
use std::ops::Deref;
use std::rc::{Rc, Weak};

use super::brt_log_button_style_extension::BrtLogButtonStyleExtension;
use super::button_style_extension::ButtonStyleExtensionVirtual;
use super::cell::{Cell as TableCell, CellVirtual};
use super::context_menu::ContextMenu;
use super::etomo_button_style_extension::EtomoButtonStyleExtension;
use super::etomo_log_button_style_extension::EtomoLogButtonStyleExtension;
use super::field_lock_controller::FieldLockController;
use super::generic_mouse_adapter::GenericMouseAdapter;
use super::imod_button_style_extension::ImodButtonStyleExtension;
use super::input_cell::{InputCell, InputCellVirtual};
use super::minibutton::Minibutton;
use super::run_3dmod_button_container::Run3dmodButtonContainer;
use super::run_3dmod_menu::Run3dmodMenu;
use super::swing_component::SwingComponent;
use super::tooltip_formatter;
use crate::imod::etomo::etomo_director::ARGUMENTS;
use crate::imod::etomo::jdk::{ActionListener, Dimension, JComponent, MouseEvent};
use crate::imod::etomo::storage::autodoc::autodoc_tokenizer::{DEFAULT_DELIMITER, SEPARATOR_CHAR};
use crate::imod::etomo::r#type::run_3dmod_menu_options::Run3dmodMenuOptions;
use crate::imod::etomo::r#type::ui_test_field_type::{self, UITestFieldType};
use crate::imod::etomo::ui::run_3dmod_menu_target::Run3dmodMenuTarget;
use crate::imod::etomo::ui::ui_component::UIComponent;
use crate::imod::etomo::util::utilities;

/// Java package-private `final class MinibuttonCell extends InputCell`.
pub struct MinibuttonCell {
    /// Java superclass `InputCell`.
    base: InputCell,
    /// This object, for the listeners and the menu.
    self_ref: Weak<MinibuttonCell>,
    /// Java final `button`.
    button: Rc<Minibutton>,
    /// Java final `contextMenu` (null unless a 3dmod cell).
    context_menu: OnceCell<Option<Rc<Run3dmodMenu>>>,
    /// Java final `container`.
    container: Option<Weak<dyn Run3dmodButtonContainer>>,
    /// Java final `fieldLockController`.
    field_lock_controller: Rc<FieldLockController>,
    /// Java `debug`.
    debug: Cell<bool>,
}

impl Deref for MinibuttonCell {
    type Target = InputCell;
    fn deref(&self) -> &InputCell {
        &self.base
    }
}

impl MinibuttonCell {
    /// Java private `MinibuttonCell(Icon, String, boolean, Run3dmodButtonContainer)`.
    /// The bevel border is painting.
    fn new_icon_string_boolean_run_3dmod_button_container(
        icon: Option<&str>,
        header_label: Option<&str>,
        run_3dmod: bool,
        container: Option<Weak<dyn Run3dmodButtonContainer>>,
    ) -> Rc<MinibuttonCell> {
        // Minibutton.getSquareInstance(icon, BorderFactory.createBevelBorder(
        // BevelBorder.RAISED))
        let button = Minibutton::get_square_instance_icon_border(icon);
        // super(): InputCell()
        let field_lock_controller =
            FieldLockController::get_button_instance(&button.get_component());
        let instance = Rc::new_cyclic(|this: &Weak<MinibuttonCell>| MinibuttonCell {
            base: InputCell::new_void(),
            self_ref: this.clone(),
            button,
            context_menu: OnceCell::new(),
            container,
            field_lock_controller,
            debug: Cell::new(false),
        });
        instance
            .base
            .set_this(Rc::downgrade(&instance) as Weak<dyn InputCellVirtual>);
        if run_3dmod {
            let context_menu = Run3dmodMenu::get_3dmod_button_instance(
                instance.clone() as Rc<dyn Run3dmodMenuTarget>,
                None,
            );
            let _ = instance.context_menu.set(Some(context_menu));
        } else {
            let _ = instance.context_menu.set(None);
        }
        if header_label.is_some() {
            instance.set_name_string(header_label);
        }
        instance
    }

    /// Java private `MinibuttonCell(String, boolean, Run3dmodButtonContainer)`.  The
    /// bevel border is painting.
    fn new_string_boolean_run_3dmod_button_container(
        header_label: Option<&str>,
        run_3dmod: bool,
        container: Option<Weak<dyn Run3dmodButtonContainer>>,
    ) -> Rc<MinibuttonCell> {
        // Minibutton.getSquareInstance(BorderFactory.createBevelBorder(
        // BevelBorder.RAISED))
        let button = Minibutton::get_square_instance_border();
        // super(): InputCell()
        let field_lock_controller =
            FieldLockController::get_button_instance(&button.get_component());
        let instance = Rc::new_cyclic(|this: &Weak<MinibuttonCell>| MinibuttonCell {
            base: InputCell::new_void(),
            self_ref: this.clone(),
            button,
            context_menu: OnceCell::new(),
            container,
            field_lock_controller,
            debug: Cell::new(false),
        });
        instance
            .base
            .set_this(Rc::downgrade(&instance) as Weak<dyn InputCellVirtual>);
        if run_3dmod {
            let context_menu = Run3dmodMenu::get_3dmod_button_instance(
                instance.clone() as Rc<dyn Run3dmodMenuTarget>,
                None,
            );
            let _ = instance.context_menu.set(Some(context_menu));
        } else {
            let _ = instance.context_menu.set(None);
        }
        if header_label.is_some() {
            instance.set_name_string(header_label);
        }
        instance
    }

    /// Java private `setButtonStyle(ButtonStyleExtension)`.
    fn set_button_style(&self, button_style: Option<Rc<dyn ButtonStyleExtensionVirtual>>) {
        if let Some(button_style) = button_style {
            button_style.setup(Some(&self.button.get_component()), None, false);
        }
    }

    /// Java `getPreferredSize()`.  Sizes are not modelled by the Swing stand-in.
    pub fn get_preferred_size(&self) -> Dimension {
        // Swing layout: button.getPreferredSize().
        Dimension::default()
    }

    /// Java `@Override setName(String, String, String)`.
    pub fn set_name_string_string_string(
        &self,
        reference1: Option<&str>,
        reference2: Option<&str>,
        reference3: Option<&str>,
    ) {
        self.set_name_string(
            utilities::concatenate(reference1, reference2, reference3, Some(" ")).as_deref(),
        );
    }

    /// Java private `setName(String)`.
    fn set_name_string(&self, reference: Option<&str>) {
        let field_type = &ui_test_field_type::BUTTON;
        let name = utilities::convert_label_to_name(reference, field_type.is_unlimited_segments());
        if let Some(name) = name {
            // Minibuttons in uitest are two state square buttons with a text showing
            // it's state. So these are treated as regular buttons.
            let button = self.button.get_component();
            button.set_name(Some(&format!("{}{}{}", field_type, SEPARATOR_CHAR, name)));
            if ARGUMENTS.lock().unwrap().is_print_names() {
                println!(
                    "{} {} ",
                    button.get_name().as_deref().unwrap_or("null"),
                    DEFAULT_DELIMITER
                );
            }
        }
    }

    /// Java `@Override getName()`.
    pub fn get_name(&self) -> Option<String> {
        self.button.get_component().get_name()
    }

    /// Java `getUniqueActionCommand()`: the class name and identity hash (Java
    /// `Object.toString`).
    pub fn get_unique_action_command(&self) -> String {
        format!(
            "etomo.ui.swing.MinibuttonCell@{:x}",
            self as *const MinibuttonCell as usize
        )
    }

    /// Java static `getInstance(Icon)`.
    pub fn get_instance(icon: Option<&str>) -> Rc<MinibuttonCell> {
        MinibuttonCell::new_icon_string_boolean_run_3dmod_button_container(icon, None, false, None)
    }

    /// Java static `getNamedInstance(Icon, String, String)`.
    pub fn get_named_instance(
        icon: Option<&str>,
        header_label1: Option<&str>,
        header_label2: Option<&str>,
    ) -> Rc<MinibuttonCell> {
        MinibuttonCell::new_icon_string_boolean_run_3dmod_button_container(
            icon,
            utilities::concatenate(header_label1, header_label2, None, Some(" ")).as_deref(),
            false,
            None,
        )
    }

    /// Java static `getRun3dmodInstance(Icon, Run3dmodButtonContainer)`.
    pub fn get_run_3dmod_instance_icon_run_3dmod_button_container(
        icon: Option<&str>,
        container: Option<Weak<dyn Run3dmodButtonContainer>>,
    ) -> Rc<MinibuttonCell> {
        let instance = MinibuttonCell::new_icon_string_boolean_run_3dmod_button_container(
            icon, None, true, container,
        );
        instance.add_listeners();
        instance
    }

    /// Java static `getNamedRun3dmodInstance(String, String)`.
    pub fn get_named_run_3dmod_instance(
        header_label1: Option<&str>,
        header_label2: Option<&str>,
    ) -> Rc<MinibuttonCell> {
        let instance = MinibuttonCell::new_icon_string_boolean_run_3dmod_button_container(
            None,
            utilities::concatenate(header_label1, header_label2, None, Some(" ")).as_deref(),
            false,
            None,
        );
        instance.set_button_style(Some(ImodButtonStyleExtension::get_instance(
            &instance.button.get_component(),
        )));
        instance
    }

    /// Java static `getNamedEtomoInstance(String, String)`.
    pub fn get_named_etomo_instance(
        header_label1: Option<&str>,
        header_label2: Option<&str>,
    ) -> Rc<MinibuttonCell> {
        let instance = MinibuttonCell::new_icon_string_boolean_run_3dmod_button_container(
            None,
            utilities::concatenate(header_label1, header_label2, None, Some(" ")).as_deref(),
            false,
            None,
        );
        instance.set_button_style(Some(EtomoButtonStyleExtension::get_instance(
            &instance.button.get_component(),
        )));
        instance
    }

    /// Java static `getNamedEtomoLogInstance(String, String)`.
    pub fn get_named_etomo_log_instance(
        header_label1: Option<&str>,
        header_label2: Option<&str>,
    ) -> Rc<MinibuttonCell> {
        let instance = MinibuttonCell::new_icon_string_boolean_run_3dmod_button_container(
            None,
            utilities::concatenate(header_label1, header_label2, None, Some(" ")).as_deref(),
            false,
            None,
        );
        instance.set_button_style(Some(EtomoLogButtonStyleExtension::get_instance(
            &instance.button.get_component(),
        )));
        instance
    }

    /// Java static `getNamedBrtLogInstance(String, String)`.
    pub fn get_named_brt_log_instance(
        header_label1: Option<&str>,
        header_label2: Option<&str>,
    ) -> Rc<MinibuttonCell> {
        let instance = MinibuttonCell::new_icon_string_boolean_run_3dmod_button_container(
            None,
            utilities::concatenate(header_label1, header_label2, None, Some(" ")).as_deref(),
            false,
            None,
        );
        instance.set_button_style(Some(BrtLogButtonStyleExtension::get_instance(
            &instance.button.get_component(),
        )));
        instance
    }

    /// Java static `getRun3dmodInstance(Run3dmodButtonContainer)`.
    pub fn get_run_3dmod_instance_run_3dmod_button_container(
        container: Option<Weak<dyn Run3dmodButtonContainer>>,
    ) -> Rc<MinibuttonCell> {
        let instance =
            MinibuttonCell::new_string_boolean_run_3dmod_button_container(None, true, container);
        instance.set_button_style(Some(ImodButtonStyleExtension::get_instance(
            &instance.button.get_component(),
        )));
        instance.add_listeners();
        instance
    }

    /// Java private `addListeners()`.
    fn add_listeners(&self) {
        if self.context_menu.get().is_some_and(Option::is_some) {
            // button.addMouseListener(new GenericMouseAdapter(this))
            self.button
                .get_component()
                .add_mouse_listener(GenericMouseAdapter::new(
                    self.self_ref.clone() as Weak<dyn ContextMenu>
                ));
        }
    }

    /// Java `@Override getComponent()`.
    pub fn get_component(&self) -> Rc<JComponent> {
        self.button.get_component()
    }

    /// Java `@Override getText()`.
    pub fn get_text(&self) -> Option<String> {
        None
    }

    /// Java `@Override popUpContextMenu(MouseEvent)`.
    ///
    /// Upstream bug fixed in translation (`MinibuttonCell.java:215`): for a cell
    /// without a 3dmod menu `contextMenu` is null and the Java throws a
    /// NullPointerException; here nothing is shown.  (The Java only registers the
    /// mouse listener when the menu exists, so only an outside caller reaches it.)
    pub fn pop_up_context_menu(&self, mouse_event: &MouseEvent) {
        if let Some(Some(context_menu)) = self.context_menu.get() {
            context_menu.pop_up_context_menu(mouse_event);
        }
    }

    /// Java `@Override menuAction(Run3dmodMenuOptions)`.
    pub fn menu_action(&self, run_3dmod_menu_options: Run3dmodMenuOptions) {
        // Java: `if (container != null)`; a container that has been dropped is null.
        if let Some(container) = self.container.as_ref().and_then(Weak::upgrade) {
            container.action(
                self.get_action_command().as_deref().unwrap_or(""),
                None,
                Some(run_3dmod_menu_options),
            );
        }
    }

    /// Java `@Override getFieldType()`.
    pub fn get_field_type(&self) -> &'static UITestFieldType {
        &ui_test_field_type::BUTTON
    }

    /// Java `@Override getWidth()`.
    pub fn get_width(&self) -> i32 {
        // Swing geometry: button.getWidth().  Sizes are not modelled by the jdk
        // stand-in.
        0
    }

    /// Java `addActionListener(ActionListener)`.
    pub fn add_action_listener(&self, listener: ActionListener) {
        self.button.get_component().add_action_listener(listener);
    }

    /// Java `@Override setDebug(boolean)`.
    pub fn set_debug(&self, input: bool) {
        self.base.set_debug_super(input);
        self.debug.set(input);
    }

    /// Java `@Override setLocked(boolean)`.
    pub fn set_locked(&self, locked: bool) {
        if self.field_lock_controller.set_locked(locked) {
            self.base.set_background_void();
        }
    }

    /// Java `@Override setEditable(boolean)`.
    pub fn set_editable(&self, editable: bool) {
        if self.field_lock_controller.set_editable(editable) {
            self.base.set_background_void();
        }
    }

    /// Java `@Override setSelected(boolean)` (empty).
    pub fn set_selected(&self, _dummy: bool) {}

    /// Java `@Override setEnabled(boolean)`.
    pub fn set_enabled(&self, enabled: bool) {
        self.field_lock_controller.set_enabled(enabled);
    }

    /// Java `@Override isSelected()`.
    pub fn is_selected(&self) -> bool {
        false
    }

    /// Java `@Override isLocked()`.
    pub fn is_locked(&self) -> bool {
        self.field_lock_controller.is_locked()
    }

    /// Java `toChangedInternalStateString()`.
    pub fn to_changed_internal_state_string(&self) -> Option<String> {
        self.field_lock_controller
            .to_changed_internal_state_string()
    }

    /// Java `@Override isEditable()`.
    pub fn is_editable(&self) -> bool {
        self.field_lock_controller.is_editable()
    }

    /// Java `@Override isEnabled()`.
    pub fn is_enabled(&self) -> bool {
        self.field_lock_controller.is_enabled()
    }

    /// Java `setDisabledIcon(Icon)`.
    pub fn set_disabled_icon(&self, _icon: Option<&str>) {
        // Swing painting: button.setDisabledIcon(icon).
    }

    /// Java `setPressedIcon(Icon)`.
    pub fn set_pressed_icon(&self, _icon: Option<&str>) {
        // Swing painting: button.setPressedIcon(icon).
    }

    /// Java `@Override setToolTipText(String)`.
    pub fn set_tool_tip_text(&self, text: Option<&str>) {
        self.button
            .get_component()
            .set_tool_tip_text(tooltip_formatter::INSTANCE.format(text).as_deref());
    }

    /// Java `setActionCommand(String)`.
    pub fn set_action_command(&self, input: Option<&str>) {
        self.button.get_component().set_action_command(input);
    }

    /// Java `getActionCommand()`.
    pub fn get_action_command(&self) -> Option<String> {
        self.button.get_component().get_action_command()
    }
}

impl CellVirtual for MinibuttonCell {
    fn cell(&self) -> &TableCell {
        &self.base
    }
    fn set_enabled(&self, enable: bool) {
        MinibuttonCell::set_enabled(self, enable);
    }
    fn msg_label_changed(&self) {
        self.base.msg_label_changed();
    }
    fn add(&self, panel: &Rc<JComponent>) {
        self.base.add(panel);
    }
}

impl InputCellVirtual for MinibuttonCell {
    fn input_cell(&self) -> &InputCell {
        &self.base
    }
    fn get_component(&self) -> Rc<JComponent> {
        MinibuttonCell::get_component(self)
    }
    fn get_field_type(&self) -> &'static UITestFieldType {
        MinibuttonCell::get_field_type(self)
    }
    fn get_width(&self) -> i32 {
        MinibuttonCell::get_width(self)
    }
    fn set_tool_tip_text(&self, tool_tip_text: Option<&str>) {
        MinibuttonCell::set_tool_tip_text(self, tool_tip_text);
    }
    fn get_text(&self) -> Option<String> {
        MinibuttonCell::get_text(self)
    }
    fn set_name_string_string_string(
        &self,
        reference1: Option<&str>,
        reference2: Option<&str>,
        reference3: Option<&str>,
    ) {
        MinibuttonCell::set_name_string_string_string(self, reference1, reference2, reference3);
    }
    fn get_name(&self) -> Option<String> {
        MinibuttonCell::get_name(self)
    }
    fn set_locked(&self, locked: bool) {
        MinibuttonCell::set_locked(self, locked);
    }
    fn set_editable(&self, editable: bool) {
        MinibuttonCell::set_editable(self, editable);
    }
    fn is_locked(&self) -> bool {
        MinibuttonCell::is_locked(self)
    }
    fn is_editable(&self) -> bool {
        MinibuttonCell::is_editable(self)
    }
    fn is_enabled(&self) -> bool {
        MinibuttonCell::is_enabled(self)
    }
    fn set_debug(&self, input: bool) {
        MinibuttonCell::set_debug(self, input);
    }
}

impl UIComponent for MinibuttonCell {
    /// Java `@Override getUIComponent()`.
    fn get_ui_component(&self) -> &dyn SwingComponent {
        self
    }
    fn get_component(&self) -> Rc<JComponent> {
        MinibuttonCell::get_component(self)
    }
}

impl SwingComponent for MinibuttonCell {
    fn get_component(&self) -> Rc<JComponent> {
        MinibuttonCell::get_component(self)
    }
}

impl Run3dmodMenuTarget for MinibuttonCell {
    fn menu_action(&self, run_3dmod_menu_options: Run3dmodMenuOptions) {
        MinibuttonCell::menu_action(self, run_3dmod_menu_options);
    }
    fn is_enabled(&self) -> bool {
        MinibuttonCell::is_enabled(self)
    }
}

impl ContextMenu for MinibuttonCell {
    fn pop_up_context_menu(&self, mouse_event: &MouseEvent) {
        MinibuttonCell::pop_up_context_menu(self, mouse_event);
    }
}

// TODO(unit): needs etomo/type/FieldSettings.java - `implements FieldSettings`
// (setSelected, isSelected, isEditable, setEditable, setEnabled, isEnabled); the
// methods are the inherent ones above.
