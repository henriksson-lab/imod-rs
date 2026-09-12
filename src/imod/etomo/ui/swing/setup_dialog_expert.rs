//! `IMOD/Etomo/src/etomo/ui/swing/SetupDialogExpert.java`.
//!
//! The concrete expert coordinates setup metadata, files, and tilt-angle
//! experts.  `SetupDialog` reaches it through these two Java calls, so this is
//! the canonical source boundary rather than a duplicate dialog-local trait.

/// Java `SetupDialogExpert` methods called from `SetupDialog.java`.
pub trait SetupDialogExpert {
    /// Java `doneSetupDialog(boolean, String, boolean)`.
    fn done_setup_dialog(&mut self, remove: bool, directory: Option<&str>, dual: bool);
    /// Java `msgSetupReconFailed()`.
    fn setup_recon_failed(&mut self);
}

#[cfg(test)]
mod tests {
    use super::SetupDialogExpert;
    #[derive(Default)]
    struct Expert {
        failed: bool,
        done: Option<(bool, Option<String>, bool)>,
    }
    impl SetupDialogExpert for Expert {
        fn done_setup_dialog(&mut self, remove: bool, directory: Option<&str>, dual: bool) {
            self.done = Some((remove, directory.map(str::to_owned), dual));
        }
        fn setup_recon_failed(&mut self) {
            self.failed = true;
        }
    }
    #[test]
    fn preserves_both_dialog_facing_java_signatures() {
        let mut expert = Expert::default();
        expert.done_setup_dialog(true, Some("dir"), false);
        expert.setup_recon_failed();
        assert_eq!(expert.done, Some((true, Some("dir".into()), false)));
        assert!(expert.failed);
    }
}
