//! `IMOD/Etomo/src/etomo/ui/swing/ProcessDialog.java`.
//!
//! `ApplicationManager`, the Swing button classes, and queue listeners are
//! deliberately represented at their direct source boundaries.
//! This unit owns the dialog's exit state, initial advanced state, button order,
//! tooltips, and action routing; it does not substitute another dialog toolkit.
#![allow(dead_code)]

pub use super::abstract_parallel_dialog::AbstractParallelDialog;
use super::etomo_frame::ActionEvent;
use super::etomo_panel::EtomoPanel;
pub use super::global_expand_button::GlobalExpandButton;
pub use super::single_line_button::SingleLineButton;
use crate::imod::etomo::comscript::parallel_param::ParallelParam;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::dialog_type::DialogType;
pub use crate::imod::etomo::ui::queue_table_event::QueueTableEvent;
pub use crate::imod::etomo::ui::queue_table_listener::QueueTableListener;
use crate::imod::etomo::util::utilities;

/// `DialogExitState.java`, a direct value dependency of `ProcessDialog`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DialogExitState {
    Cancel,
    Postpone,
    Execute,
    Save,
}

impl std::fmt::Display for DialogExitState {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Cancel => "Cancel",
            Self::Postpone => "Postpone",
            Self::Execute => "Execute",
            Self::Save => "Save",
        })
    }
}

/// Direct `ApplicationManager.isAdvanced(DialogType, AxisID)` boundary.
pub trait ProcessDialogApplicationManager {
    fn is_advanced(&self, dialog_type: DialogType, axis_id: AxisID) -> bool;
}

/// Java `Box.create*` entries in the exact `pnlExitButtons` insertion order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExitButtonLayoutItem {
    HorizontalGlue,
    Cancel,
    Postpone,
    Execute,
    Advanced,
}

/// Native `JPanel` layout state directly read or written by this unit.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExitButtonsPanel {
    pub box_layout_x_axis: bool,
    pub children: Vec<ExitButtonLayoutItem>,
    pub narrow_button_size_applied: bool,
}

/// The callback corresponding to Java's abstract `done()` method.
pub type ProcessDialogDone = Box<dyn FnMut()>;

/// Fields and implemented actions of Java's abstract `ProcessDialog`.
pub struct ProcessDialog<'a> {
    pub application_manager: &'a dyn ProcessDialogApplicationManager,
    pub axis_id: AxisID,
    pub dialog_type: DialogType,
    pub root_panel: EtomoPanel,
    pub pnl_exit_buttons: ExitButtonsPanel,
    pub btn_cancel: SingleLineButton,
    pub btn_execute: SingleLineButton,
    pub btn_advanced: GlobalExpandButton,
    pub btn_postpone: Option<SingleLineButton>,
    exit_state: DialogExitState,
    displayed: bool,
    done: ProcessDialogDone,
}

impl<'a> ProcessDialog<'a> {
    /// Java three-argument constructor, which delegates with `usePostpone = true`.
    pub fn new(
        application_manager: &'a dyn ProcessDialogApplicationManager,
        axis_id: AxisID,
        dialog_type: DialogType,
        done: ProcessDialogDone,
    ) -> Self {
        Self::new_with_postpone(application_manager, axis_id, dialog_type, true, done)
    }

    /// Java four-argument constructor.
    pub fn new_with_postpone(
        application_manager: &'a dyn ProcessDialogApplicationManager,
        axis_id: AxisID,
        dialog_type: DialogType,
        use_postpone: bool,
        done: ProcessDialogDone,
    ) -> Self {
        eprintln!(
            "\n{}\nDialog: {dialog_type}",
            utilities::get_date_time_stamp()
        );
        let mut btn_advanced = GlobalExpandButton::get_instance("Advanced", "Basic");
        btn_advanced.change_state(application_manager.is_advanced(dialog_type, axis_id));
        let mut dialog = Self {
            application_manager,
            axis_id,
            dialog_type,
            root_panel: EtomoPanel::default(),
            pnl_exit_buttons: ExitButtonsPanel {
                box_layout_x_axis: true,
                children: vec![
                    ExitButtonLayoutItem::HorizontalGlue,
                    ExitButtonLayoutItem::Cancel,
                    ExitButtonLayoutItem::HorizontalGlue,
                ],
                narrow_button_size_applied: true,
            },
            btn_cancel: SingleLineButton::new_with_label(Some("Cancel")),
            btn_execute: SingleLineButton::new_with_label(Some("Execute")),
            btn_advanced,
            btn_postpone: use_postpone.then(|| SingleLineButton::new_with_label(Some("Postpone"))),
            exit_state: DialogExitState::Save,
            displayed: true,
            done,
        };
        if use_postpone {
            dialog.pnl_exit_buttons.children.extend([
                ExitButtonLayoutItem::Postpone,
                ExitButtonLayoutItem::HorizontalGlue,
            ]);
        }
        dialog.pnl_exit_buttons.children.extend([
            ExitButtonLayoutItem::Execute,
            ExitButtonLayoutItem::HorizontalGlue,
            ExitButtonLayoutItem::Advanced,
            ExitButtonLayoutItem::HorizontalGlue,
        ]);
        dialog.set_tool_tip_text();
        dialog.btn_cancel.add_action_listener();
        if let Some(button) = &mut dialog.btn_postpone {
            button.add_action_listener();
        }
        dialog.btn_execute.add_action_listener();
        dialog
    }

    /// Java abstract `done()` callback boundary.
    pub fn done(&mut self) {
        (self.done)();
    }
    pub fn set_exit_state(&mut self, exit_state: DialogExitState) {
        self.exit_state = exit_state;
    }
    pub fn get_container(&self) -> &EtomoPanel {
        &self.root_panel
    }
    /// Java implementation is empty.
    pub fn get_parameters(&self, _param: &mut dyn ParallelParam) {}
    pub fn add_exit_buttons(&mut self) {
        self.root_panel.children.extend([
            super::abstract_frame::ComponentState::default(),
            super::abstract_frame::ComponentState {
                height: 10,
                ..Default::default()
            },
            super::abstract_frame::ComponentState::default(),
        ]);
    }
    pub fn get_dialog_type(&self) -> DialogType {
        self.dialog_type
    }
    pub fn is_advanced(&self) -> bool {
        self.btn_advanced.is_expanded()
    }
    pub fn set_displayed(&mut self, displayed: bool) {
        self.displayed = displayed;
    }
    pub fn is_displayed(&self) -> bool {
        self.displayed
    }
    pub fn button_cancel_action(&mut self, _event: &ActionEvent) {
        utilities::button_timestamp_container(Some("cancel"), Some(&self.dialog_type.to_string()));
        self.exit_state = DialogExitState::Cancel;
        self.done();
    }
    pub fn button_postpone_action(&mut self, _event: &ActionEvent) {
        utilities::button_timestamp_container(
            Some("postpone"),
            Some(&self.dialog_type.to_string()),
        );
        self.exit_state = DialogExitState::Postpone;
        self.done();
    }
    /// Java implementation is empty.
    pub fn queue_table_event_action(&mut self, _event: QueueTableEvent) {}
    /// Java implementation is empty.
    pub fn add_queue_table_listener(&mut self, _listener: &mut dyn QueueTableListener) {}
    /// Java implementation is empty.
    pub fn remove_queue_table_listener(&mut self, _listener: &mut dyn QueueTableListener) {}
    pub fn button_execute_action(&mut self) -> bool {
        utilities::button_timestamp_container(Some("done"), Some(&self.dialog_type.to_string()));
        self.exit_state = DialogExitState::Execute;
        self.done();
        true
    }
    pub fn save_action(&mut self) {
        utilities::timestamp_command_status(Some("save"), Some(&self.dialog_type.to_string()));
        self.exit_state = DialogExitState::Save;
        self.done();
    }
    pub fn get_exit_state(&self) -> DialogExitState {
        self.exit_state
    }
    /// Java private `setToolTipText()`.
    fn set_tool_tip_text(&mut self) {
        self.btn_cancel.set_tool_tip_text(Some(
            "This button will abort any changes to the parameters in this dialog box and return you to the main window.",
        ));
        if let Some(button) = &mut self.btn_postpone {
            button.set_tool_tip_text(Some(
                "This button will save any changes to the parameters in this dialog box and return you to the main window without executing any of the processing.  Any parameter changes will also be written to the com scripts.",
            ));
        }
        self.btn_execute.set_tool_tip_text(Some(
            "This button will save any changes to the parameters in this dialog box and execute the specified operation on the data.  Any parameter changes will also be written to the com scripts.",
        ));
        self.btn_advanced.set_tool_tip_text(
            "This button will present a more detailed set of options for each of the underlying processes.",
        );
    }
}

impl AbstractParallelDialog for ProcessDialog<'_> {
    fn get_parameters(&self, param: &mut dyn ParallelParam) {
        ProcessDialog::get_parameters(self, param);
    }
    fn get_dialog_type(&self) -> DialogType {
        ProcessDialog::get_dialog_type(self)
    }
}

/// Java private final `buttonCancelActionAdapter`.
pub struct ButtonCancelActionAdapter;
impl ButtonCancelActionAdapter {
    pub fn action_performed(dialog: &mut ProcessDialog<'_>, event: &ActionEvent) {
        dialog.button_cancel_action(event);
    }
}
/// Java private final `buttonPostponeActionAdapter`.
pub struct ButtonPostponeActionAdapter;
impl ButtonPostponeActionAdapter {
    pub fn action_performed(dialog: &mut ProcessDialog<'_>, event: &ActionEvent) {
        dialog.button_postpone_action(event);
    }
}
/// Java private final `buttonExecuteActionAdapter`.
pub struct ButtonExecuteActionAdapter;
impl ButtonExecuteActionAdapter {
    pub fn action_performed(dialog: &mut ProcessDialog<'_>, _event: &ActionEvent) {
        dialog.button_execute_action();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::rc::Rc;

    struct Manager(bool);
    impl ProcessDialogApplicationManager for Manager {
        fn is_advanced(&self, _dialog_type: DialogType, _axis_id: AxisID) -> bool {
            self.0
        }
    }
    struct Parameter;
    impl ParallelParam for Parameter {
        fn get_subcommand_mode(
            &self,
        ) -> Option<&dyn crate::imod::etomo::comscript::command_mode::CommandMode> {
            None
        }
    }

    #[test]
    fn constructor_preserves_source_button_order_tooltips_and_advanced_state() {
        let manager = Manager(true);
        let mut dialog = ProcessDialog::new(
            &manager,
            AxisID::First,
            DialogType::SetupRecon,
            Box::new(|| {}),
        );
        assert!(dialog.is_displayed());
        assert!(dialog.is_advanced());
        assert_eq!(dialog.get_exit_state(), DialogExitState::Save);
        assert_eq!(
            dialog.pnl_exit_buttons.children,
            vec![
                ExitButtonLayoutItem::HorizontalGlue,
                ExitButtonLayoutItem::Cancel,
                ExitButtonLayoutItem::HorizontalGlue,
                ExitButtonLayoutItem::Postpone,
                ExitButtonLayoutItem::HorizontalGlue,
                ExitButtonLayoutItem::Execute,
                ExitButtonLayoutItem::HorizontalGlue,
                ExitButtonLayoutItem::Advanced,
                ExitButtonLayoutItem::HorizontalGlue,
            ]
        );
        assert_eq!(
            dialog
                .btn_cancel
                .multi_line_button
                .button
                .action_listener_count,
            1
        );
        assert!(
            dialog
                .btn_postpone
                .as_ref()
                .unwrap()
                .multi_line_button
                .button
                .action_listener_count
                == 1
        );
        assert_eq!(
            dialog
                .btn_execute
                .multi_line_button
                .button
                .action_listener_count,
            1
        );
        assert!(
            dialog
                .btn_advanced
                .button
                .multi_line_button
                .button
                .tooltip
                .as_deref()
                .unwrap()
                .contains("detailed")
        );
        dialog.add_exit_buttons();
        let mut param = Parameter;
        dialog.get_parameters(&mut param);
    }

    #[test]
    fn source_exit_actions_set_state_then_invoke_done() {
        let manager = Manager(false);
        let called = Rc::new(Cell::new(0));
        let callback_called = called.clone();
        let mut dialog = ProcessDialog::new_with_postpone(
            &manager,
            AxisID::Only,
            DialogType::Tools,
            false,
            Box::new(move || callback_called.set(callback_called.get() + 1)),
        );
        let event = ActionEvent::new("ignored");
        ButtonCancelActionAdapter::action_performed(&mut dialog, &event);
        assert_eq!(dialog.get_exit_state(), DialogExitState::Cancel);
        dialog.button_postpone_action(&event);
        assert_eq!(dialog.get_exit_state(), DialogExitState::Postpone);
        assert!(dialog.button_execute_action());
        assert_eq!(dialog.get_exit_state(), DialogExitState::Execute);
        dialog.save_action();
        assert_eq!(dialog.get_exit_state(), DialogExitState::Save);
        assert_eq!(called.get(), 4);
        assert!(dialog.btn_postpone.is_none());
    }
}
