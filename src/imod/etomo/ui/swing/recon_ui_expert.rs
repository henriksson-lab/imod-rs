//! `IMOD/Etomo/src/etomo/ui/swing/ReconUIExpert.java`.
//!
//! Java declares this as an abstract class.  Its three abstract methods form
//! the canonical Rust expert contract; concrete experts own manager/dialog
//! process coordination.

/// Java abstract `ReconUIExpert` methods.
pub trait ReconUIExpert {
    type Dialog;
    type DialogExitState;
    /// Java abstract package-private `doneDialog()`.
    fn done_dialog(&mut self);
    /// Java abstract package-private `saveDialog()`.
    fn save_dialog(&mut self);
    /// Java abstract package-private `getDialog()`; `None` is Java null.
    fn get_dialog(&mut self) -> Option<&mut Self::Dialog>;
    /// Java final `doneDialog(DialogExitState)`.
    fn done_dialog_with_exit_state(&mut self, _exit_state: Self::DialogExitState) {
        if self.get_dialog().is_some() {
            self.done_dialog();
        }
    }
    /// Java final `saveAction()`; concrete `ProcessDialog` owns its action.
    fn save_action(&mut self);
    /// Java final `saveDialog(DialogExitState)`.
    fn save_dialog_with_exit_state(&mut self, _exit_state: Self::DialogExitState) {
        if self.get_dialog().is_some() {
            self.save_dialog();
        }
    }
}

/// Native state owned by the shared reconstruction-expert coordinator.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ReconUIExpertCoordinator {
    pub scripts_created: bool,
    pub dialog_out_of_date: bool,
    pub dialog_visible: bool,
    pub dialog_state: Option<String>,
    pub process_start_messages: u32,
    pub process_messages: Vec<String>,
    pub progress_label: Option<String>,
    pub progress_steps: Option<i32>,
    pub progress_running: bool,
}

impl ReconUIExpertCoordinator {
    #[allow(non_snake_case)]
    pub fn canShowDialog(&self) -> bool {
        self.scripts_created
    }
    #[allow(non_snake_case)]
    pub fn showDialog(&mut self, has_dialog: bool) -> bool {
        if self.dialog_out_of_date || !has_dialog {
            return false;
        }
        self.dialog_visible = true;
        true
    }
    #[allow(non_snake_case)]
    pub fn openDialog(&mut self) {
        self.dialog_out_of_date = false;
        self.dialog_visible = true;
    }
    #[allow(non_snake_case)]
    pub fn sendMsgProcessStarting(&mut self, display_present: bool) {
        if display_present {
            self.process_start_messages += 1;
        }
    }
    #[allow(non_snake_case)]
    pub fn sendMsg(&mut self, message: Option<&str>, display_present: bool) {
        if let Some(message) = message.filter(|_| display_present) {
            self.process_messages.push(message.into());
        }
    }
    #[allow(non_snake_case)]
    pub fn leaveDialog(&mut self, exit_state: &str) {
        match exit_state {
            "POSTPONE" => self.setDialogState("INPROGRESS"),
            "EXECUTE" => self.setDialogState("COMPLETE"),
            _ => {}
        }
        self.dialog_visible = false;
        self.dialog_out_of_date = true;
    }
    #[allow(non_snake_case)]
    pub fn setDialogState(&mut self, state: &str) {
        self.dialog_state = Some(state.into());
    }
    #[allow(non_snake_case)]
    pub fn processchunks(
        &mut self,
        dialog_present: bool,
        parallel_present: bool,
        parameters_valid: bool,
    ) -> bool {
        self.sendMsgProcessStarting(dialog_present);
        if !dialog_present || !parallel_present || !parameters_valid {
            self.sendMsg(Some("FAILED_TO_START"), true);
            return false;
        }
        self.setDialogState("INPROGRESS");
        true
    }
    #[allow(non_snake_case)]
    pub fn getParallelPanel(&self) -> bool {
        self.dialog_visible
    }
    #[allow(non_snake_case)]
    pub fn setProgressBar(&mut self, label: &str, steps: i32) {
        self.progress_label = Some(label.into());
        self.progress_steps = Some(steps);
    }
    #[allow(non_snake_case)]
    pub fn startProgressBar(&mut self, label: &str) {
        self.progress_label = Some(label.into());
        self.progress_running = true;
    }
    #[allow(non_snake_case)]
    pub fn stopProgressBar(&mut self) {
        self.progress_running = false;
    }
}

#[cfg(test)]
mod tests {
    use super::ReconUIExpert;
    struct Expert {
        dialog: Option<()>,
        done: usize,
        saved: usize,
    }
    impl ReconUIExpert for Expert {
        type Dialog = ();
        type DialogExitState = ();
        fn done_dialog(&mut self) {
            self.done += 1;
        }
        fn save_dialog(&mut self) {
            self.saved += 1;
        }
        fn get_dialog(&mut self) -> Option<&mut Self::Dialog> {
            self.dialog.as_mut()
        }
        fn save_action(&mut self) {}
    }
    #[test]
    fn final_exit_methods_keep_java_null_dialog_guards() {
        let mut expert = Expert {
            dialog: None,
            done: 0,
            saved: 0,
        };
        expert.done_dialog_with_exit_state(());
        expert.save_dialog_with_exit_state(());
        assert_eq!((expert.done, expert.saved), (0, 0));
        expert.dialog = Some(());
        expert.done_dialog_with_exit_state(());
        expert.save_dialog_with_exit_state(());
        assert_eq!((expert.done, expert.saved), (1, 1));
    }
}
