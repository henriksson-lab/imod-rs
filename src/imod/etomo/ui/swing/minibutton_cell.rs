//! `IMOD/Etomo/src/etomo/ui/swing/MinibuttonCell.java`.
//!
//! The wrapped `Minibutton`, its right-click 3dmod menu and listener wiring
//! are explicit native GUI boundaries.  The source unit's construction,
//! naming, lock state, and action/menu dispatch state are retained directly.
#![allow(dead_code)]

use std::sync::atomic::{AtomicUsize, Ordering};

use crate::imod::etomo::etomo_director::ARGUMENTS;
use crate::imod::etomo::storage::autodoc::autodoc_tokenizer::{DEFAULT_DELIMITER, SEPARATOR_CHAR};
use crate::imod::etomo::util::utilities;

use super::context_menu::{ContextMenu, MouseEvent};
use super::field_lock_controller::{FieldLockController, JButton};
use super::input_cell::{InputCell, InputCellComponent};
use super::minibutton::Minibutton;
use super::run_3dmod_menu::Run3dmodMenu;
use super::tooltip_formatter::TooltipFormatter;
use super::ui_utilities::{Icon, UiUtilities};

static NEXT_MINIBUTTON_CELL_IDENTITY_HASH_CODE: AtomicUsize = AtomicUsize::new(1);

/// Java package-private final `MinibuttonCell`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MinibuttonCell {
    /// Java superclass `InputCell` state.
    pub input_cell: InputCell,
    pub button: Minibutton,
    pub context_menu: Option<Run3dmodMenu>,
    pub container_attached: bool,
    pub field_lock_controller: FieldLockController,
    pub debug: bool,
    pub identity_hash_code: usize,
    pub background_refresh_count: usize,
    pub last_run_3dmod_menu_action: Option<String>,
    pub style_name: Option<&'static str>,
}

impl MinibuttonCell {
    /// Java private `MinibuttonCell(Icon, String, boolean, Run3dmodButtonContainer)`.
    pub fn new_icon(
        icon: Option<Icon>,
        header_label: Option<&str>,
        run_3dmod: bool,
        container_attached: bool,
    ) -> Self {
        let mut value = Self {
            input_cell: InputCell::new(),
            button: Minibutton::get_square_icon_instance(icon, Some("BevelBorder.RAISED")),
            context_menu: run_3dmod.then(|| Run3dmodMenu::get_3dmod_button_instance(None)),
            container_attached,
            field_lock_controller: FieldLockController::get_button_instance(JButton {
                enabled: true,
            }),
            debug: false,
            identity_hash_code: NEXT_MINIBUTTON_CELL_IDENTITY_HASH_CODE
                .fetch_add(1, Ordering::Relaxed),
            background_refresh_count: 0,
            last_run_3dmod_menu_action: None,
            style_name: None,
        };
        if let Some(header_label) = header_label {
            value.set_name(header_label);
        }
        value
    }

    /// Java private `MinibuttonCell(String, boolean, Run3dmodButtonContainer)`.
    pub fn new_empty(
        header_label: Option<&str>,
        run_3dmod: bool,
        container_attached: bool,
    ) -> Self {
        let mut value = Self {
            input_cell: InputCell::new(),
            button: Minibutton::get_square_empty_instance(Some("BevelBorder.RAISED")),
            context_menu: run_3dmod.then(|| Run3dmodMenu::get_3dmod_button_instance(None)),
            container_attached,
            field_lock_controller: FieldLockController::get_button_instance(JButton {
                enabled: true,
            }),
            debug: false,
            identity_hash_code: NEXT_MINIBUTTON_CELL_IDENTITY_HASH_CODE
                .fetch_add(1, Ordering::Relaxed),
            background_refresh_count: 0,
            last_run_3dmod_menu_action: None,
            style_name: None,
        };
        if let Some(header_label) = header_label {
            value.set_name(header_label);
        }
        value
    }

    /// Java private `setButtonStyle(ButtonStyleExtension)`.
    pub fn set_button_style(&mut self, button_style: Option<&'static str>) {
        if let Some(button_style) = button_style {
            self.style_name = Some(button_style);
            match button_style {
                "ImodButtonStyleExtension" => {
                    self.button.icon = Some(Icon {
                        width: 0,
                        height: 0,
                    })
                }
                "EtomoButtonStyleExtension" => {
                    self.button.icon = Some(Icon {
                        width: 0,
                        height: 0,
                    })
                }
                "EtomoLogButtonStyleExtension" => {
                    self.button.icon = Some(Icon {
                        width: 0,
                        height: 0,
                    })
                }
                "BrtLogButtonStyleExtension" => {
                    self.button.icon = Some(Icon {
                        width: 0,
                        height: 0,
                    })
                }
                _ => {}
            }
            self.button.set_size();
        }
    }

    /// Java `getPreferredSize`.
    pub fn get_preferred_size(&self) -> super::panel::Dimension {
        self.button.preferred_size
    }

    /// Java overridden `setName(String, String, String)`.
    pub fn set_name_three(
        &mut self,
        reference1: Option<&str>,
        reference2: Option<&str>,
        reference3: Option<&str>,
    ) {
        let reference = utilities::concatenate(reference1, reference2, reference3, Some(" "));
        if let Some(reference) = reference {
            self.set_name(&reference);
        }
    }

    /// Java private `setName(String)`.
    pub fn set_name(&mut self, reference: &str) {
        if let Some(name) = utilities::convert_label_to_name(Some(reference), true) {
            self.button.name = Some(format!("bn{SEPARATOR_CHAR}{name}"));
            if ARGUMENTS.lock().unwrap().is_print_names() {
                println!(
                    "{} {DEFAULT_DELIMITER} ",
                    self.button.name.as_deref().unwrap_or_default()
                );
            }
        }
    }

    /// Java `getName`.
    pub fn get_name(&self) -> Option<&str> {
        self.button.name.as_deref()
    }

    /// Java `getUniqueActionCommand`.
    pub fn get_unique_action_command(&self) -> String {
        format!(
            "etomo.ui.swing.MinibuttonCell@{:x}",
            self.identity_hash_code
        )
    }

    /// Java static `getInstance(Icon)`.
    pub fn get_instance(icon: Option<Icon>) -> Self {
        Self::new_icon(icon, None, false, false)
    }

    /// Java static `getNamedInstance(Icon, String, String)`.
    pub fn get_named_instance(
        icon: Option<Icon>,
        header_label1: Option<&str>,
        header_label2: Option<&str>,
    ) -> Self {
        let label = utilities::concatenate(header_label1, header_label2, None, Some(" "));
        Self::new_icon(icon, label.as_deref(), false, false)
    }

    /// Java static `getRun3dmodInstance(Icon, Run3dmodButtonContainer)`.
    pub fn get_run_3dmod_icon_instance(icon: Option<Icon>) -> Self {
        let mut value = Self::new_icon(icon, None, true, true);
        value.add_listeners();
        value
    }

    /// Java static `getNamedRun3dmodInstance(String, String)`.
    pub fn get_named_run_3dmod_instance(
        header_label1: Option<&str>,
        header_label2: Option<&str>,
    ) -> Self {
        let label = utilities::concatenate(header_label1, header_label2, None, Some(" "));
        let mut value = Self::new_icon(None, label.as_deref(), false, false);
        value.set_button_style(Some("ImodButtonStyleExtension"));
        value
    }

    /// Java static `getNamedEtomoInstance(String, String)`.
    pub fn get_named_etomo_instance(
        header_label1: Option<&str>,
        header_label2: Option<&str>,
    ) -> Self {
        let label = utilities::concatenate(header_label1, header_label2, None, Some(" "));
        let mut value = Self::new_icon(None, label.as_deref(), false, false);
        value.set_button_style(Some("EtomoButtonStyleExtension"));
        value
    }

    /// Java static `getNamedEtomoLogInstance(String, String)`.
    pub fn get_named_etomo_log_instance(
        header_label1: Option<&str>,
        header_label2: Option<&str>,
    ) -> Self {
        let label = utilities::concatenate(header_label1, header_label2, None, Some(" "));
        let mut value = Self::new_icon(None, label.as_deref(), false, false);
        value.set_button_style(Some("EtomoLogButtonStyleExtension"));
        value
    }

    /// Java static `getNamedBrtLogInstance(String, String)`.
    pub fn get_named_brt_log_instance(
        header_label1: Option<&str>,
        header_label2: Option<&str>,
    ) -> Self {
        let label = utilities::concatenate(header_label1, header_label2, None, Some(" "));
        let mut value = Self::new_icon(None, label.as_deref(), false, false);
        value.set_button_style(Some("BrtLogButtonStyleExtension"));
        value
    }

    /// Java static `getRun3dmodInstance(Run3dmodButtonContainer)`.
    pub fn get_run_3dmod_instance() -> Self {
        let mut value = Self::new_empty(None, true, true);
        value.set_button_style(Some("ImodButtonStyleExtension"));
        value.add_listeners();
        value
    }

    /// Java private `addListeners`.
    pub fn add_listeners(&mut self) {
        if self.context_menu.is_some() {
            self.button.mouse_listener_count += 1;
        }
    }

    /// Java `getComponent`.
    pub fn get_component(&self) -> &Minibutton {
        &self.button
    }

    /// Java `getUIComponent`.
    pub fn get_ui_component(&self) -> &Self {
        self
    }

    /// Java `getText`.
    pub fn get_text(&self) -> Option<&str> {
        None
    }

    /// Java `popUpContextMenu(MouseEvent)`.
    pub fn pop_up_context_menu(&mut self, mouse_event: MouseEvent) {
        if let Some(context_menu) = &mut self.context_menu {
            context_menu.pop_up_context_menu(self.button.enabled, mouse_event);
        }
    }

    /// Java `menuAction(Run3dmodMenuOptions)`; the enclosing button container dispatch is a boundary.
    pub fn menu_action(&mut self, run_3dmod_menu_options: Option<&str>) {
        if self.container_attached {
            self.last_run_3dmod_menu_action = run_3dmod_menu_options.map(str::to_owned);
        }
    }

    /// Java `getFieldType`.
    pub fn get_field_type(&self) -> &'static str {
        "bn"
    }

    /// Java `getWidth`.
    pub fn get_width(&self) -> i32 {
        self.button.width
    }

    /// Java `addActionListener(ActionListener)`.
    pub fn add_action_listener(&mut self) {
        self.button.action_listener_count += 1;
    }

    /// Java overridden `setDebug(boolean)`.
    pub fn set_debug(&mut self, input: bool) {
        self.input_cell.set_debug(input);
        self.debug = input;
    }

    /// Java `setLocked(boolean)`.
    pub fn set_locked(&mut self, locked: bool) {
        let changed = self.field_lock_controller.set_locked(locked);
        self.button.enabled = self
            .field_lock_controller
            .button
            .as_ref()
            .expect("MinibuttonCell owns Java JButton")
            .enabled;
        if changed {
            self.set_background();
        }
    }

    /// Java `setEditable(boolean)`.
    pub fn set_editable(&mut self, editable: bool) {
        let changed = self.field_lock_controller.set_editable(editable);
        self.button.enabled = self
            .field_lock_controller
            .button
            .as_ref()
            .expect("MinibuttonCell owns Java JButton")
            .enabled;
        if changed {
            self.set_background();
        }
    }

    /// Java `setSelected(boolean)`.
    pub fn set_selected(&mut self, _dummy: bool) {}

    /// Java `setEnabled(boolean)`.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.field_lock_controller.set_enabled(enabled);
        self.button.enabled = self
            .field_lock_controller
            .button
            .as_ref()
            .expect("MinibuttonCell owns Java JButton")
            .enabled;
    }

    /// Java `isSelected`.
    pub fn is_selected(&self) -> bool {
        false
    }

    /// Java `isLocked`.
    pub fn is_locked(&self) -> bool {
        self.field_lock_controller.is_locked()
    }

    /// Java `toChangedInternalStateString`.
    pub fn to_changed_internal_state_string(&mut self) -> Option<String> {
        self.field_lock_controller
            .to_changed_internal_state_string()
    }

    /// Java `isEditable`.
    pub fn is_editable(&self) -> bool {
        self.field_lock_controller.is_editable()
    }

    /// Java `isEnabled`.
    pub fn is_enabled(&self) -> bool {
        self.field_lock_controller.is_enabled()
    }

    /// Java `setDisabledIcon(Icon)`.
    pub fn set_disabled_icon(&mut self, icon: Option<Icon>) {
        self.button.disabled_icon = icon;
    }

    /// Java `setPressedIcon(Icon)`.
    pub fn set_pressed_icon(&mut self, icon: Option<Icon>) {
        self.button.pressed_icon = icon;
    }

    /// Java `setToolTipText(String)`.
    pub fn set_tool_tip_text(&mut self, text: Option<&str>) {
        self.button.tooltip = TooltipFormatter::instance().format(text);
    }

    /// Java `setActionCommand(String)`.
    pub fn set_action_command(&mut self, input: Option<&str>) {
        self.button.action_command = input.map(str::to_owned);
    }

    /// Java `getActionCommand`.
    pub fn get_action_command(&self) -> Option<&str> {
        self.button.action_command.as_deref()
    }

    /// Java inherited `setBackground()` at the InputCell presentation boundary.
    pub fn set_background(&mut self) {
        self.background_refresh_count += 1;
        self.input_cell.clone().set_background(&mut self.button);
    }
}

impl ContextMenu for MinibuttonCell {
    fn pop_up_context_menu(&mut self, mouse_event: MouseEvent) {
        Self::pop_up_context_menu(self, mouse_event);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_factories_preserve_style_name_name_and_run3dmod_listener_paths() {
        let etomo = MinibuttonCell::get_named_etomo_instance(Some("Open"), Some("Set"));
        let run = MinibuttonCell::get_run_3dmod_instance();
        assert_eq!(etomo.get_name(), Some("bn.open-set"));
        assert_eq!(etomo.style_name, Some("EtomoButtonStyleExtension"));
        assert!(run.context_menu.is_some());
        assert_eq!(run.button.mouse_listener_count, 1);
        assert_eq!(run.style_name, Some("ImodButtonStyleExtension"));
    }

    #[test]
    fn lock_tooltip_menu_and_icon_delegates_follow_source_paths() {
        let mut cell = MinibuttonCell::get_run_3dmod_icon_instance(Some(Icon {
            width: 8,
            height: 9,
        }));
        cell.set_locked(true);
        cell.set_tool_tip_text(Some("one two"));
        cell.pop_up_context_menu(MouseEvent {
            x: 3,
            y: 4,
            right_mouse_button: true,
        });
        cell.menu_action(Some("binBy2"));
        cell.set_disabled_icon(Some(Icon {
            width: 2,
            height: 3,
        }));
        assert!(!cell.button.enabled);
        assert_eq!(cell.button.tooltip.as_deref(), Some("<html>one two"));
        // Java `Run3dmodMenu.popUpContextMenu` returns when its target is disabled.
        assert_eq!(cell.context_menu.as_ref().unwrap().popup_events.len(), 0);
        assert_eq!(cell.last_run_3dmod_menu_action.as_deref(), Some("binBy2"));
        assert_eq!(
            cell.button.disabled_icon,
            Some(Icon {
                width: 2,
                height: 3
            })
        );
    }
}
