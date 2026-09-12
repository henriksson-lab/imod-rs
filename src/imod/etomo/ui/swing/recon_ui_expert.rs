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
