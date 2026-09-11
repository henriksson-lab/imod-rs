//! `IMOD/Etomo/src/etomo/ui/swing/AbstractFrame.java`.
//!
//! Swing's `JFrame`, `JOptionPane`, and `JDialog` do not have a hidden Rust
//! replacement here.  This unit owns the source-visible frame and popup state;
//! the optional Slint `UIHarness` is the presentation boundary.  In particular,
//! a dialog for which the harness has supplied no answer returns Swing's
//! `CLOSED_OPTION`, rather than fabricating a Yes/No response.
#![allow(dead_code)]

use super::etomo_frame::{ActionEvent, FramePresentation, FrameType};
use super::etomo_menu::MenuTarget;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::etomo_director::ARGUMENTS;
use crate::imod::etomo::logic::popup_tool;
use crate::imod::etomo::process::process_messages::{MessageType, ProcessMessages};
use crate::imod::etomo::storage::autodoc::autodoc_tokenizer::{DEFAULT_DELIMITER, SEPARATOR_CHAR};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::util::utilities;

pub const OK: &str = "OK";
pub const ETOMO_QUESTION: &str = "Etomo question";
pub const YES: &str = "Yes";
pub const NO: &str = "No";
pub const CANCEL: &str = "Cancel";
pub const YES_NO_LABEL_ARRAY: [&str; 2] = [YES, NO];
pub const NO_INDEX: usize = 1;
pub const OK_LABEL_ARRAY: [&str; 1] = [OK];
pub const DELETE_NO_LABEL_ARRAY: [&str; 2] = ["Delete", NO];
pub const DELETE_OPTION: i32 = 0; // JOptionPane.YES_OPTION
pub const YES_NO_CANCEL_LABEL_ARRAY: [&str; 3] = [YES, NO, CANCEL];
pub const DEFAULT_OPTION: i32 = -1;
pub const YES_NO_OPTION: i32 = 0;
pub const YES_NO_CANCEL_OPTION: i32 = 1;
pub const ERROR_MESSAGE: i32 = 0;
pub const INFORMATION_MESSAGE: i32 = 1;
pub const WARNING_MESSAGE: i32 = 2;
pub const QUESTION_MESSAGE: i32 = 3;
pub const CLOSED_OPTION: i32 = -1;

/// Source-used `Component` fields.  A GUI backend owns the real native widget.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ComponentState {
    pub height: i32,
    pub location: (i32, i32),
    pub orientation_left_to_right: bool,
}

/// The complete information `JOptionPane` passes to its presentation boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OptionDialog {
    pub axis_id: Option<AxisID>,
    pub parent: Option<ComponentState>,
    pub message: Vec<String>,
    pub title: Option<String>,
    pub option_type: i32,
    pub message_type: i32,
    pub initial_value: Option<String>,
    pub override_defaults: bool,
    pub options: Option<Vec<String>>,
    pub modal: Option<bool>,
    pub location: Option<(i32, i32)>,
}

/// The Rust equivalent of the abstract Java hooks in `AbstractFrame`.
///
/// The returned `MenuTarget` makes the Java callback's downstream manager or
/// dialog boundary explicit.  `EtomoFrame` implements this trait by forwarding
/// into its existing source-named menu methods.
pub trait AbstractFrameActions {
    fn menu_file_action(&mut self, action_event: &ActionEvent) -> MenuTarget;
    fn menu_tools_action(&mut self, action_event: &ActionEvent) -> MenuTarget;
    fn menu_view_action(&mut self, action_event: &ActionEvent) -> Result<(), String>;
    fn menu_options_action(&mut self, action_event: &ActionEvent) -> Result<(), String>;
    fn menu_help_action(&mut self, action_event: &ActionEvent) -> MenuTarget;
    fn get_frame_type(&self) -> FrameType;
    fn cancel(&mut self);
    fn save(&mut self, axis_id: AxisID) -> Result<(), String>;
    fn save_as(&mut self) -> Result<(), String>;
    fn close(&mut self);
}

/// Fields and implemented methods of Java's abstract `AbstractFrame`.
#[derive(Clone, Debug)]
pub struct AbstractFrame {
    pub verbose: bool,
    pub presentation: FramePresentation,
    pub component: ComponentState,
    pub last_dialog: Option<OptionDialog>,
    /// The next response contributed by the real UI test/backend.  `None` is
    /// the source-equivalent closed dialog, never a guessed answer.
    pub dialog_response: Option<i32>,
}

impl AbstractFrame {
    /// Java field initialisers plus `JFrame()`.
    pub fn new() -> Self {
        Self {
            verbose: false,
            presentation: FramePresentation::default(),
            component: ComponentState {
                orientation_left_to_right: true,
                ..Default::default()
            },
            last_dialog: None,
            dialog_response: None,
        }
    }

    /// `getUIComponent`.
    pub fn get_ui_component(&self) -> &Self {
        self
    }
    /// `getComponent`.
    pub fn get_component(&self) -> &ComponentState {
        &self.component
    }
    /// `setVisible(boolean)`.  UserConfiguration's persisted-location query is
    /// an explicit boundary until `UserConfiguration.java` is translated.
    pub fn set_visible(&mut self, visible: bool) {
        self.presentation.visible = visible;
    }
    /// `setVerbose`.
    pub fn set_verbose(&mut self, verbose: bool) {
        self.verbose = verbose;
    }
    /// `getAxisID`.
    pub fn get_axis_id(&self) -> AxisID {
        AxisID::Only
    }
    /// `pack(boolean)`.
    pub fn pack_force(&mut self, force: bool, auto_fit: bool) {
        if !force && !auto_fit {
            self.set_visible(true);
        } else {
            self.presentation.packed = true;
        }
    }
    /// `repaint(AxisID)`.
    pub fn repaint(&mut self, _axis_id: AxisID) {
        self.presentation.repaint_count += 1;
    }
    /// `pack(AxisID)`.
    pub fn pack_axis(&mut self, _axis_id: AxisID) {
        self.presentation.packed = true;
    }
    /// `pack(AxisID, boolean)`.
    pub fn pack_axis_force(&mut self, _axis_id: AxisID, force: bool, auto_fit: bool) {
        self.pack_force(force, auto_fit);
    }
    /// `repaintWindow`.
    pub fn repaint_window(&mut self) {
        self.repaint_container();
        self.presentation.repaint_count += 1;
    }
    /// `repaintContainer(Container)`.  Native child traversal belongs to Slint.
    pub fn repaint_container(&mut self) {
        self.presentation.repaint_count += 1;
    }
    /// `menuFileMRUListAction(ActionEvent)` is intentionally empty in Java.
    pub fn menu_file_mru_list_action(&self, _event: &ActionEvent) {}

    /// Java overload `openInfoMessageDialog(... String, String[], ProcessMessages, ...)`.
    pub fn open_info_message_dialog_factory(
        &mut self,
        manager: Option<&dyn BaseManager>,
        parent: Option<ComponentState>,
        axis_id: Option<AxisID>,
        message: Option<&str>,
        message_array: Option<&[String]>,
        title: Option<&str>,
        modal: Option<bool>,
    ) -> i32 {
        self.show_option_pane_factory(
            manager,
            parent,
            axis_id,
            message,
            message_array,
            title,
            INFORMATION_MESSAGE,
            modal,
        )
    }
    /// Java overload `openMessageDialog(... String, String[], ProcessMessages, ...)`.
    pub fn open_message_dialog_factory(
        &mut self,
        manager: Option<&dyn BaseManager>,
        parent: Option<ComponentState>,
        axis_id: Option<AxisID>,
        message: Option<&str>,
        message_array: Option<&[String]>,
        title: Option<&str>,
        modal: Option<bool>,
    ) -> i32 {
        self.show_option_pane_factory(
            manager,
            parent,
            axis_id,
            message,
            message_array,
            title,
            ERROR_MESSAGE,
            modal,
        )
    }
    /// Java overload `openWarningMessageDialog(... String, String[], ProcessMessages, ...)`.
    pub fn open_warning_message_dialog_factory(
        &mut self,
        manager: Option<&dyn BaseManager>,
        parent: Option<ComponentState>,
        axis_id: Option<AxisID>,
        message: Option<&str>,
        message_array: Option<&[String]>,
        title: Option<&str>,
        modal: Option<bool>,
    ) -> i32 {
        self.show_option_pane_factory(
            manager,
            parent,
            axis_id,
            message,
            message_array,
            title,
            WARNING_MESSAGE,
            modal,
        )
    }
    /// Java overload `openErrorMessageDialog(... String, String[], ProcessMessages, ...)`.
    pub fn open_error_message_dialog_factory(
        &mut self,
        manager: Option<&dyn BaseManager>,
        parent: Option<ComponentState>,
        axis_id: Option<AxisID>,
        message: Option<&str>,
        message_array: Option<&[String]>,
        title: Option<&str>,
        modal: Option<bool>,
    ) -> i32 {
        self.show_option_pane_factory(
            manager,
            parent,
            axis_id,
            message,
            message_array,
            title,
            ERROR_MESSAGE,
            modal,
        )
    }
    /// `showOptionPane(... wrapFactory ...)`; ProcessMessages is an unported
    /// process unit, so no invented surrogate can enter this source branch.
    pub fn show_option_pane_factory(
        &mut self,
        manager: Option<&dyn BaseManager>,
        parent: Option<ComponentState>,
        axis_id: Option<AxisID>,
        message: Option<&str>,
        message_array: Option<&[String]>,
        title: Option<&str>,
        message_type: i32,
        modal: Option<bool>,
    ) -> i32 {
        let Some(wrapped) = self.wrap_factory(message, message_array) else {
            return CLOSED_OPTION;
        };
        self.show_option_pane_parent(
            manager,
            parent,
            axis_id,
            &wrapped,
            title,
            message_type,
            modal,
        )
    }
    /// `openYesNoDialog(... String, String[], ...)`.
    pub fn open_yes_no_dialog_factory(
        &mut self,
        manager: Option<&dyn BaseManager>,
        parent: Option<ComponentState>,
        axis_id: Option<AxisID>,
        message: Option<&str>,
        message_array: Option<&[String]>,
        title: Option<&str>,
        initial_value: Option<&str>,
    ) -> i32 {
        self.show_option_confirm_pane_factory(
            manager,
            parent,
            axis_id,
            message,
            message_array,
            title,
            YES_NO_OPTION,
            None,
            initial_value,
            &YES_NO_LABEL_ARRAY,
        )
    }
    /// `openYesNoCancelDialog(... String, String[], ...)`.
    pub fn open_yes_no_cancel_dialog_factory(
        &mut self,
        manager: Option<&dyn BaseManager>,
        parent: Option<ComponentState>,
        axis_id: Option<AxisID>,
        message: Option<&str>,
        message_array: Option<&[String]>,
        title: Option<&str>,
        initial_value: Option<&str>,
    ) -> i32 {
        self.show_option_confirm_pane_factory(
            manager,
            parent,
            axis_id,
            message,
            message_array,
            title,
            YES_NO_CANCEL_OPTION,
            None,
            initial_value,
            &YES_NO_CANCEL_LABEL_ARRAY,
        )
    }
    /// `openDeleteDialog(... String, String[], ...)`.
    pub fn open_delete_dialog_factory(
        &mut self,
        manager: Option<&dyn BaseManager>,
        parent: Option<ComponentState>,
        axis_id: Option<AxisID>,
        message: Option<&str>,
        message_array: Option<&[String]>,
        title: Option<&str>,
        initial_value: Option<&str>,
    ) -> i32 {
        self.show_option_confirm_pane_factory(
            manager,
            parent,
            axis_id,
            message,
            message_array,
            title.or(Some("Delete File?")),
            DEFAULT_OPTION,
            None,
            initial_value,
            &DELETE_NO_LABEL_ARRAY,
        )
    }
    /// `openYesNoWarningDialog(... String, String[], ...)`.
    pub fn open_yes_no_warning_dialog_factory(
        &mut self,
        manager: Option<&dyn BaseManager>,
        parent: Option<ComponentState>,
        axis_id: Option<AxisID>,
        message: Option<&str>,
        message_array: Option<&[String]>,
        title: Option<&str>,
        initial_value: Option<&str>,
    ) -> i32 {
        self.show_option_confirm_pane_factory(
            manager,
            parent,
            axis_id,
            message,
            message_array,
            title.or(Some("Etomo Warning")),
            YES_NO_OPTION,
            Some(WARNING_MESSAGE),
            initial_value.or(Some(YES_NO_LABEL_ARRAY[NO_INDEX])),
            &YES_NO_LABEL_ARRAY,
        )
    }
    /// `showOptionConfirmPane(... Component ...)`.
    pub fn show_option_confirm_pane_factory(
        &mut self,
        manager: Option<&dyn BaseManager>,
        parent: Option<ComponentState>,
        axis_id: Option<AxisID>,
        message: Option<&str>,
        message_array: Option<&[String]>,
        title: Option<&str>,
        option_type: i32,
        message_type: Option<i32>,
        initial_value: Option<&str>,
        options: &[&str],
    ) -> i32 {
        let Some(wrapped) = self.wrap_factory(message, message_array) else {
            return CLOSED_OPTION;
        };
        self.show_option_pane_parent_full(
            manager,
            parent,
            axis_id,
            &wrapped,
            title.or(Some(ETOMO_QUESTION)),
            option_type,
            message_type.unwrap_or(QUESTION_MESSAGE),
            initial_value,
            true,
            options,
            None,
        )
    }

    /// `displayMessage(BaseManager, String, String, AxisID)`.
    pub fn display_message(
        &mut self,
        manager: Option<&dyn BaseManager>,
        message: &str,
        title: Option<&str>,
        axis_id: AxisID,
    ) {
        self.open_message_dialog(manager, axis_id, message, title);
    }
    /// `displayMessage(BaseManager, Component, String, String, AxisID)`.
    pub fn display_message_parent(
        &mut self,
        manager: Option<&dyn BaseManager>,
        parent: ComponentState,
        message: &str,
        title: Option<&str>,
        axis_id: Option<AxisID>,
    ) {
        self.open_message_dialog_parent(manager, Some(parent), axis_id, message, title);
    }
    /// `displayYesNoMessage(BaseManager, Component, String, AxisID)`.
    pub fn display_yes_no_message_parent(
        &mut self,
        manager: Option<&dyn BaseManager>,
        parent: ComponentState,
        message: &str,
        axis_id: Option<AxisID>,
    ) -> bool {
        self.open_yes_no_dialog_parent(manager, Some(parent), axis_id, message)
    }
    /// `displayWarningMessage(BaseManager, Component, String, String, AxisID)`.
    pub fn display_warning_message_parent(
        &mut self,
        manager: Option<&dyn BaseManager>,
        parent: ComponentState,
        message: &str,
        title: Option<&str>,
        axis_id: Option<AxisID>,
    ) {
        self.open_warning_message_dialog_parent(manager, Some(parent), axis_id, message, title);
    }
    /// `displayMessage(BaseManager, Component, String[], String, AxisID)`.
    pub fn display_message_lines_parent(
        &mut self,
        manager: Option<&dyn BaseManager>,
        parent: ComponentState,
        message: &[String],
        title: Option<&str>,
        axis_id: Option<AxisID>,
    ) {
        self.open_message_dialog_lines_parent(manager, Some(parent), axis_id, message, title);
    }
    /// `displayMessage(BaseManager, String, String)`.
    pub fn display_message_only(
        &mut self,
        manager: Option<&dyn BaseManager>,
        message: &str,
        title: Option<&str>,
    ) {
        self.open_message_dialog(manager, AxisID::Only, message, title);
    }
    /// `displayInfoMessage`.
    pub fn display_info_message(
        &mut self,
        manager: Option<&dyn BaseManager>,
        parent: ComponentState,
        message: &str,
        title: Option<&str>,
        axis_id: Option<AxisID>,
    ) {
        self.open_info_message_dialog(manager, Some(parent), axis_id, message, title);
    }
    /// `displayYesNoCancelMessage`.
    pub fn display_yes_no_cancel_message(
        &mut self,
        manager: Option<&dyn BaseManager>,
        message: &str,
        axis_id: AxisID,
    ) -> i32 {
        self.open_yes_no_cancel_dialog(manager, axis_id, message)
    }
    /// `displayYesNoMessage(BaseManager, String[], AxisID)`.
    pub fn display_yes_no_message_lines(
        &mut self,
        manager: Option<&dyn BaseManager>,
        message: &[String],
        axis_id: AxisID,
    ) -> bool {
        self.open_yes_no_dialog_lines(manager, axis_id, message)
    }
    /// `displayYesNoMessage(BaseManager, String, AxisID)`.
    pub fn display_yes_no_message(
        &mut self,
        manager: Option<&dyn BaseManager>,
        message: &str,
        axis_id: AxisID,
    ) -> bool {
        self.open_yes_no_dialog(manager, axis_id, message)
    }
    /// `openYesNoDialogWithDefaultNo`.
    pub fn open_yes_no_dialog_with_default_no(
        &mut self,
        manager: Option<&dyn BaseManager>,
        message: &str,
        title: Option<&str>,
        axis_id: AxisID,
    ) -> bool {
        self.open_yes_no_dialog_default(manager, axis_id, message, title, NO_INDEX, true)
    }
    /// `displayDeleteMessage`.
    pub fn display_delete_message(
        &mut self,
        manager: Option<&dyn BaseManager>,
        message: &[String],
        axis_id: AxisID,
    ) -> bool {
        self.open_delete_dialog(manager, axis_id, message)
    }
    /// `displayMessage(BaseManager, String[], String, AxisID)`.
    pub fn display_message_lines(
        &mut self,
        manager: Option<&dyn BaseManager>,
        message: &[String],
        title: Option<&str>,
        axis_id: AxisID,
    ) {
        self.open_message_dialog_lines(manager, axis_id, message, title);
    }
    /// Java `displayErrorMessage(BaseManager, ProcessMessages, String, AxisID)`.
    pub fn display_error_message_process_messages(
        &mut self,
        manager: Option<&dyn BaseManager>,
        process_messages: &ProcessMessages,
        title: Option<&str>,
        axis_id: AxisID,
    ) {
        self.open_error_message_dialog_process_messages(manager, axis_id, process_messages, title);
    }
    /// Java `displayMessage(BaseManager, ProcessMessages, String, AxisID)`.
    pub fn display_message_process_messages(
        &mut self,
        manager: Option<&dyn BaseManager>,
        process_messages: &ProcessMessages,
        title: Option<&str>,
        axis_id: AxisID,
    ) {
        self.open_message_dialog_process_messages(manager, axis_id, process_messages, title);
    }
    /// `displayYesNoWarningDialog`.
    pub fn display_yes_no_warning_dialog(
        &mut self,
        manager: Option<&dyn BaseManager>,
        message: &str,
        axis_id: AxisID,
    ) -> bool {
        self.open_yes_no_warning_dialog(manager, axis_id, message)
    }
    /// Java `displayWarningMessage(BaseManager, ProcessMessages, String, AxisID)`.
    pub fn display_warning_message_process_messages(
        &mut self,
        manager: Option<&dyn BaseManager>,
        process_messages: &ProcessMessages,
        title: Option<&str>,
        axis_id: AxisID,
    ) {
        self.open_warning_message_dialog_process_messages(
            manager,
            axis_id,
            process_messages,
            title,
        );
    }

    /// `openMessageDialog(BaseManager, AxisID, String, String)`.
    pub fn open_message_dialog(
        &mut self,
        manager: Option<&dyn BaseManager>,
        axis_id: AxisID,
        message: &str,
        title: Option<&str>,
    ) {
        let wrapped = self.wrap(message);
        self.show_option_pane(manager, axis_id, &wrapped, title, ERROR_MESSAGE);
    }
    /// `openMessageDialog(BaseManager, Component, AxisID, String, String)`.
    pub fn open_message_dialog_parent(
        &mut self,
        manager: Option<&dyn BaseManager>,
        parent: Option<ComponentState>,
        axis_id: Option<AxisID>,
        message: &str,
        title: Option<&str>,
    ) {
        let wrapped = self.wrap(message);
        self.show_option_pane_parent(
            manager,
            parent,
            axis_id,
            &wrapped,
            title,
            ERROR_MESSAGE,
            None,
        );
    }
    /// `openYesNoDialog(BaseManager, Component, AxisID, String)`.
    pub fn open_yes_no_dialog_parent(
        &mut self,
        manager: Option<&dyn BaseManager>,
        parent: Option<ComponentState>,
        axis_id: Option<AxisID>,
        message: &str,
    ) -> bool {
        let wrapped = self.wrap(message);
        self.show_option_confirm_pane_parent(
            manager,
            parent,
            axis_id,
            &wrapped,
            Some(ETOMO_QUESTION),
            YES_NO_OPTION,
            &YES_NO_LABEL_ARRAY,
        ) == DELETE_OPTION
    }
    /// `openWarningMessageDialog(BaseManager, Component, AxisID, String, String)`.
    pub fn open_warning_message_dialog_parent(
        &mut self,
        manager: Option<&dyn BaseManager>,
        parent: Option<ComponentState>,
        axis_id: Option<AxisID>,
        message: &str,
        title: Option<&str>,
    ) {
        let wrapped = self.wrap(message);
        self.show_option_pane_parent(
            manager,
            parent,
            axis_id,
            &wrapped,
            title,
            WARNING_MESSAGE,
            None,
        );
    }
    /// `openMessageDialog(BaseManager, Component, AxisID, String[], String)`.
    pub fn open_message_dialog_lines_parent(
        &mut self,
        manager: Option<&dyn BaseManager>,
        parent: Option<ComponentState>,
        axis_id: Option<AxisID>,
        message: &[String],
        title: Option<&str>,
    ) {
        let wrapped = self.wrap_lines(message);
        self.show_option_pane_parent(
            manager,
            parent,
            axis_id,
            &wrapped,
            title,
            ERROR_MESSAGE,
            None,
        );
    }
    /// Java `openWarningMessageDialog(BaseManager, AxisID, ProcessMessages, String)`.
    pub fn open_warning_message_dialog_process_messages(
        &mut self,
        manager: Option<&dyn BaseManager>,
        axis_id: AxisID,
        process_messages: &ProcessMessages,
        title: Option<&str>,
    ) {
        let wrapped = self.wrap_warning(process_messages);
        self.show_option_pane(manager, axis_id, &wrapped, title, ERROR_MESSAGE);
    }
    /// Java `openErrorMessageDialog(BaseManager, AxisID, ProcessMessages, String)`.
    pub fn open_error_message_dialog_process_messages(
        &mut self,
        manager: Option<&dyn BaseManager>,
        axis_id: AxisID,
        process_messages: &ProcessMessages,
        title: Option<&str>,
    ) {
        let wrapped = self.wrap_error(process_messages);
        self.show_option_pane(manager, axis_id, &wrapped, title, ERROR_MESSAGE);
    }
    /// Java `openMessageDialog(BaseManager, AxisID, ProcessMessages, String)`.
    pub fn open_message_dialog_process_messages(
        &mut self,
        manager: Option<&dyn BaseManager>,
        axis_id: AxisID,
        process_messages: &ProcessMessages,
        title: Option<&str>,
    ) {
        let wrapped = self.wrap_messages_process_messages(process_messages);
        self.show_option_pane(manager, axis_id, &wrapped, title, ERROR_MESSAGE);
    }
    /// `openYesNoWarningDialog`.
    pub fn open_yes_no_warning_dialog(
        &mut self,
        manager: Option<&dyn BaseManager>,
        axis_id: AxisID,
        message: &str,
    ) -> bool {
        let wrapped = self.wrap(message);
        self.show_option_pane_full(
            manager,
            axis_id,
            &wrapped,
            Some("Etomo Warning"),
            YES_NO_OPTION,
            WARNING_MESSAGE,
            Some(YES_NO_LABEL_ARRAY[NO_INDEX]),
            false,
            &YES_NO_LABEL_ARRAY,
        ) == DELETE_OPTION
    }
    /// `openYesNoDialog(BaseManager, AxisID, String)`.
    pub fn open_yes_no_dialog(
        &mut self,
        manager: Option<&dyn BaseManager>,
        axis_id: AxisID,
        message: &str,
    ) -> bool {
        let wrapped = self.wrap(message);
        self.show_option_confirm_pane(
            manager,
            axis_id,
            &wrapped,
            Some(ETOMO_QUESTION),
            YES_NO_OPTION,
            &YES_NO_LABEL_ARRAY,
        ) == DELETE_OPTION
    }
    /// `openYesNoDialog(BaseManager, AxisID, String, String, int, boolean)`.
    pub fn open_yes_no_dialog_default(
        &mut self,
        manager: Option<&dyn BaseManager>,
        axis_id: AxisID,
        message: &str,
        title: Option<&str>,
        initial_value_index: usize,
        override_default_labels: bool,
    ) -> bool {
        let wrapped = self.wrap(message);
        self.show_option_confirm_pane_default(
            manager,
            axis_id,
            &wrapped,
            title,
            YES_NO_OPTION,
            YES_NO_LABEL_ARRAY[initial_value_index],
            override_default_labels,
            &YES_NO_LABEL_ARRAY,
        ) == DELETE_OPTION
    }
    /// `openDeleteDialog(BaseManager, AxisID, String[])`.
    pub fn open_delete_dialog(
        &mut self,
        manager: Option<&dyn BaseManager>,
        axis_id: AxisID,
        message: &[String],
    ) -> bool {
        let wrapped = self.wrap_lines(message);
        self.show_option_pane_full(
            manager,
            axis_id,
            &wrapped,
            Some("Delete File?"),
            DEFAULT_OPTION,
            QUESTION_MESSAGE,
            None,
            true,
            &DELETE_NO_LABEL_ARRAY,
        ) == DELETE_OPTION
    }
    /// `openYesNoDialog(BaseManager, AxisID, String[])`.
    pub fn open_yes_no_dialog_lines(
        &mut self,
        manager: Option<&dyn BaseManager>,
        axis_id: AxisID,
        message: &[String],
    ) -> bool {
        let wrapped = self.wrap_lines(message);
        self.show_option_confirm_pane(
            manager,
            axis_id,
            &wrapped,
            Some(ETOMO_QUESTION),
            YES_NO_OPTION,
            &YES_NO_LABEL_ARRAY,
        ) == DELETE_OPTION
    }
    /// `openInfoMessageDialog(BaseManager, Component, AxisID, String, String)`.
    pub fn open_info_message_dialog(
        &mut self,
        manager: Option<&dyn BaseManager>,
        parent: Option<ComponentState>,
        axis_id: Option<AxisID>,
        message: &str,
        title: Option<&str>,
    ) {
        let wrapped = self.wrap(message);
        self.show_option_pane_parent(
            manager,
            parent,
            axis_id,
            &wrapped,
            title,
            INFORMATION_MESSAGE,
            None,
        );
    }
    /// `openMessageDialog(BaseManager, AxisID, String[], String)`.
    pub fn open_message_dialog_lines(
        &mut self,
        manager: Option<&dyn BaseManager>,
        axis_id: AxisID,
        message: &[String],
        title: Option<&str>,
    ) {
        let wrapped = self.wrap_lines(message);
        self.show_option_pane(manager, axis_id, &wrapped, title, ERROR_MESSAGE);
    }
    /// `openYesNoCancelDialog(BaseManager, AxisID, String)`.
    pub fn open_yes_no_cancel_dialog(
        &mut self,
        manager: Option<&dyn BaseManager>,
        axis_id: AxisID,
        message: &str,
    ) -> i32 {
        let wrapped = self.wrap(message);
        self.show_option_confirm_pane(
            manager,
            axis_id,
            &wrapped,
            Some(ETOMO_QUESTION),
            YES_NO_CANCEL_OPTION,
            &YES_NO_CANCEL_LABEL_ARRAY,
        )
    }
    /// `showOptionConfirmPane(BaseManager, AxisID, String[], String, int, String[])`.
    pub fn show_option_confirm_pane(
        &mut self,
        manager: Option<&dyn BaseManager>,
        axis_id: AxisID,
        message: &[String],
        title: Option<&str>,
        option_type: i32,
        option_strings: &[&str],
    ) -> i32 {
        self.show_option_pane_full(
            manager,
            axis_id,
            message,
            title,
            option_type,
            QUESTION_MESSAGE,
            None,
            false,
            option_strings,
        )
    }
    /// `showOptionConfirmPane(BaseManager, AxisID, String[], String, int, String, boolean, String[])`.
    pub fn show_option_confirm_pane_default(
        &mut self,
        manager: Option<&dyn BaseManager>,
        axis_id: AxisID,
        message: &[String],
        title: Option<&str>,
        option_type: i32,
        initial_value: &str,
        override_default_labels: bool,
        option_strings: &[&str],
    ) -> i32 {
        self.show_option_pane_full(
            manager,
            axis_id,
            message,
            title,
            option_type,
            QUESTION_MESSAGE,
            Some(initial_value),
            override_default_labels,
            option_strings,
        )
    }
    /// `wrapFactory`; ProcessMessages branch remains a named source boundary.
    pub fn wrap_factory(
        &self,
        message: Option<&str>,
        message_array: Option<&[String]>,
    ) -> Option<Vec<String>> {
        if let Some(message) = message {
            Some(self.wrap(message))
        } else {
            message_array.map(|m| self.wrap_lines(m))
        }
    }
    /// Java `wrapMessages(ProcessMessages, MessageType)`.
    pub fn wrap_messages_process_messages_type(
        &self,
        process_messages: &ProcessMessages,
        process_message_type: Option<MessageType>,
    ) -> Vec<String> {
        match process_message_type {
            Some(message_type) => {
                (0..process_messages.size(message_type)).fold(Vec::new(), |lines, i| {
                    popup_tool::wrap_message(process_messages.get(message_type, i), Some(lines))
                })
            }
            None => self.wrap_messages_process_messages(process_messages),
        }
    }
    /// Java `wrapWarning(ProcessMessages)`.
    pub fn wrap_warning(&self, process_messages: &ProcessMessages) -> Vec<String> {
        self.wrap_messages_process_messages_type(process_messages, Some(MessageType::Warning))
    }
    /// Java `wrapError(ProcessMessages)`.
    pub fn wrap_error(&self, process_messages: &ProcessMessages) -> Vec<String> {
        self.wrap_messages_process_messages_type(process_messages, Some(MessageType::Error))
    }
    /// Java `wrapMessages(ProcessMessages)`.  Java places an empty line after each
    /// non-empty message category.
    pub fn wrap_messages_process_messages(
        &self,
        process_messages: &ProcessMessages,
    ) -> Vec<String> {
        let mut lines = Vec::new();
        for message_type in [
            MessageType::Error,
            MessageType::ChunkError,
            MessageType::Warning,
            MessageType::ChunkWarning,
            MessageType::Info,
        ] {
            let before = lines.len();
            for i in 0..process_messages.size(message_type) {
                lines =
                    popup_tool::wrap_message(process_messages.get(message_type, i), Some(lines));
            }
            if lines.len() > before {
                lines.push(String::new());
            }
        }
        lines
    }
    /// `wrap(String)`.
    pub fn wrap(&self, message: &str) -> Vec<String> {
        popup_tool::wrap_message(Some(message), None)
    }
    /// `wrap(String[])`.
    pub fn wrap_lines(&self, message: &[String]) -> Vec<String> {
        message
            .iter()
            .fold(None, |array, line| {
                Some(popup_tool::wrap_message(Some(line), array))
            })
            .unwrap_or_default()
    }
    /// `toStringArray(ArrayList)`.
    pub fn to_string_array(&self, array_list: Option<Vec<String>>) -> Option<Vec<String>> {
        array_list
    }
    /// `showOptionPane(BaseManager, AxisID, String[], String, int)`.
    pub fn show_option_pane(
        &mut self,
        manager: Option<&dyn BaseManager>,
        axis_id: AxisID,
        message: &[String],
        title: Option<&str>,
        message_type: i32,
    ) {
        let _ = self.show_option_pane_full(
            manager,
            axis_id,
            message,
            title,
            DEFAULT_OPTION,
            message_type,
            None,
            false,
            &OK_LABEL_ARRAY,
        );
    }
    /// `showOptionPane(BaseManager, Component, AxisID, String[], String, int, Boolean)`.
    pub fn show_option_pane_parent(
        &mut self,
        manager: Option<&dyn BaseManager>,
        parent: Option<ComponentState>,
        axis_id: Option<AxisID>,
        message: &[String],
        title: Option<&str>,
        message_type: i32,
        modal: Option<bool>,
    ) -> i32 {
        self.show_option_pane_parent_full(
            manager,
            parent,
            axis_id,
            message,
            title,
            DEFAULT_OPTION,
            message_type,
            None,
            false,
            &OK_LABEL_ARRAY,
            modal,
        )
    }
    /// `showOptionConfirmPane(BaseManager, Component, AxisID, String[], String, int, String[])`.
    pub fn show_option_confirm_pane_parent(
        &mut self,
        manager: Option<&dyn BaseManager>,
        parent: Option<ComponentState>,
        axis_id: Option<AxisID>,
        message: &[String],
        title: Option<&str>,
        option_type: i32,
        options: &[&str],
    ) -> i32 {
        self.show_option_pane_parent_full(
            manager,
            parent,
            axis_id,
            message,
            title,
            option_type,
            QUESTION_MESSAGE,
            None,
            false,
            options,
            None,
        )
    }
    /// `showOptionPane(BaseManager, AxisID, ...)`.
    pub fn show_option_pane_full(
        &mut self,
        manager: Option<&dyn BaseManager>,
        axis_id: AxisID,
        message: &[String],
        title: Option<&str>,
        option_type: i32,
        message_type: i32,
        initial_value: Option<&str>,
        override_defaults: bool,
        options: &[&str],
    ) -> i32 {
        self.show_option_dialog(
            manager,
            Some(axis_id),
            None,
            message,
            title,
            option_type,
            message_type,
            initial_value,
            override_defaults,
            Some(options),
            None,
        )
    }
    /// `showOptionPane(BaseManager, Component, AxisID, ...)`.
    pub fn show_option_pane_parent_full(
        &mut self,
        manager: Option<&dyn BaseManager>,
        parent: Option<ComponentState>,
        axis_id: Option<AxisID>,
        message: &[String],
        title: Option<&str>,
        option_type: i32,
        message_type: i32,
        initial_value: Option<&str>,
        override_defaults: bool,
        options: &[&str],
        modal: Option<bool>,
    ) -> i32 {
        self.show_option_dialog(
            manager,
            axis_id,
            parent,
            message,
            title,
            option_type,
            message_type,
            initial_value,
            override_defaults,
            Some(options),
            modal,
        )
    }
    /// `showOptionDialog`.
    pub fn show_option_dialog(
        &mut self,
        manager: Option<&dyn BaseManager>,
        axis_id: Option<AxisID>,
        parent: Option<ComponentState>,
        message: &[String],
        title: Option<&str>,
        option_type: i32,
        mut message_type: i32,
        initial_value: Option<&str>,
        override_defaults: bool,
        options: Option<&[&str]>,
        modal: Option<bool>,
    ) -> i32 {
        if message.is_empty() {
            return CLOSED_OPTION;
        }
        if let Some(manager) = manager {
            manager.log_message_array(Some(message), title, None, axis_id);
        } else {
            eprintln!(
                "{}\n{} - {} axis:",
                utilities::get_date_time_stamp(),
                title.unwrap_or("null"),
                axis_id.map_or("null", |a| a.get_key())
            );
            for line in message {
                eprintln!("{line}");
            }
        }
        if message_type == ERROR_MESSAGE {
            let title_lc = title.unwrap_or("").to_ascii_lowercase();
            let error = title_lc.contains("error")
                || message
                    .iter()
                    .take(3)
                    .any(|m| m.to_ascii_lowercase().contains("error:"));
            let warning = title_lc.contains("warning")
                || message
                    .iter()
                    .take(3)
                    .any(|m| m.to_ascii_lowercase().contains("warning:"));
            if !error && warning {
                message_type = WARNING_MESSAGE;
            }
        }
        let location = parent.as_ref().and_then(|p| {
            if axis_id.is_none() {
                Some((p.location.0, (p.location.1 - (p.height / 2 + 20)).max(0)))
            } else {
                None
            }
        });
        let option_strings = if override_defaults {
            options.map(|v| v.iter().map(|s| (*s).to_string()).collect())
        } else {
            None
        };
        self.last_dialog = Some(OptionDialog {
            axis_id,
            parent,
            message: message.to_vec(),
            title: title.map(str::to_string),
            option_type,
            message_type,
            initial_value: initial_value.map(str::to_string),
            override_defaults,
            options: option_strings,
            modal,
            location,
        });
        self.print_name(
            utilities::convert_label_to_name(title, true).as_deref(),
            options,
            title,
            message,
        );
        self.dialog_response.take().unwrap_or(CLOSED_OPTION)
    }
    /// `printName`.
    pub fn print_name(
        &self,
        name: Option<&str>,
        options: Option<&[&str]>,
        title: Option<&str>,
        message: &[String],
    ) {
        if ARGUMENTS.lock().unwrap().is_print_names() {
            if let Some(options) = options.filter(|options| !options.is_empty()) {
                let mut buffer = format!(
                    "popup{SEPARATOR_CHAR}{} {DEFAULT_DELIMITER} ",
                    name.unwrap_or("null")
                );
                let mut appended = false;
                for option in options.iter().skip(1) {
                    if appended {
                        buffer.push(',');
                    }
                    buffer.push_str(option);
                    appended = true;
                }
                println!("{buffer}");
            }
        }
        if self.verbose {
            eprintln!("Popup:\n{}", title.unwrap_or("null"));
            for line in message {
                eprintln!("{line}");
            }
        }
    }
}

impl Default for AbstractFrame {
    fn default() -> Self {
        Self::new()
    }
}

impl AbstractFrameActions for super::etomo_frame::EtomoFrame {
    fn menu_file_action(&mut self, event: &ActionEvent) -> MenuTarget {
        super::etomo_frame::EtomoFrame::menu_file_action(self, event)
    }
    fn menu_tools_action(&mut self, event: &ActionEvent) -> MenuTarget {
        super::etomo_frame::EtomoFrame::menu_tools_action(self, event)
    }
    fn menu_view_action(&mut self, event: &ActionEvent) -> Result<(), String> {
        super::etomo_frame::EtomoFrame::menu_view_action(self, event)
    }
    fn menu_options_action(&mut self, event: &ActionEvent) -> Result<(), String> {
        super::etomo_frame::EtomoFrame::menu_options_action(self, event)
    }
    fn menu_help_action(&mut self, event: &ActionEvent) -> MenuTarget {
        super::etomo_frame::EtomoFrame::menu_help_action(self, event)
    }
    fn get_frame_type(&self) -> FrameType {
        super::etomo_frame::EtomoFrame::get_frame_type(self).unwrap_or(FrameType::Main)
    }
    fn cancel(&mut self) {
        super::etomo_frame::EtomoFrame::cancel(self)
    }
    fn save(&mut self, axis_id: AxisID) -> Result<(), String> {
        super::etomo_frame::EtomoFrame::save(self, axis_id)
    }
    fn save_as(&mut self) -> Result<(), String> {
        super::etomo_frame::EtomoFrame::save_as(self)
    }
    fn close(&mut self) {
        super::etomo_frame::EtomoFrame::close(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn wrap_matches_popup_tool_width_and_comma_rule() {
        let f = AbstractFrame::new();
        assert_eq!(f.wrap("a\n\n"), vec!["a"]);
        let line = format!("{} tail", "x".repeat(60));
        assert_eq!(f.wrap(&line), vec!["x".repeat(60), " tail".into()]);
    }
    #[test]
    fn dialog_is_closed_without_gui_answer_and_classifies_warning() {
        let mut f = AbstractFrame::new();
        f.open_message_dialog(None, AxisID::Only, "warning: check", Some("message"));
        assert_eq!(
            f.last_dialog.as_ref().unwrap().message_type,
            WARNING_MESSAGE
        );
        f.dialog_response = Some(DELETE_OPTION);
        assert!(f.open_yes_no_dialog(None, AxisID::Only, "continue?"));
    }
    #[test]
    fn etomo_frame_trait_forwards_menu_listener_path() {
        let mut frame = super::super::etomo_frame::EtomoFrame::new();
        assert_eq!(
            AbstractFrameActions::menu_file_action(&mut frame, &ActionEvent::new("Save")),
            MenuTarget::Save
        );
    }
}
