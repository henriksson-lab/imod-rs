//! `IMOD/Etomo/src/etomo/ui/swing/SetupDialogExpert.java`.
//!
//! The concrete expert coordinates setup metadata, files, and tilt-angle
//! experts.  `SetupDialog` reaches it through these two Java calls, so this is
//! the canonical source boundary rather than a duplicate dialog-local trait.

use std::path::{Path, PathBuf};

/// Java `SetupDialogExpert` methods called from `SetupDialog.java`.
pub trait SetupDialogExpert {
    /// Java `doneSetupDialog(boolean, String, boolean)`.
    fn done_setup_dialog(&mut self, remove: bool, directory: Option<&str>, dual: bool);
    /// Java `msgSetupReconFailed()`.
    fn setup_recon_failed(&mut self);
}

/// Native dialog boundary owned by the setup coordinator.  This replaces the
/// Java expert's direct Swing calls while keeping its orchestration order.
pub trait SetupDialogExpertDialog {
    fn directive_file_collection(&self) -> Option<String>;
    fn show_progress_panel(&mut self);
    fn update_display(&mut self, process_done: bool);
    fn exclude_views_succeeded(&mut self, axis: &str, running: bool, done: bool);
    fn setup_recon_failed(&mut self);
    fn dataset(&self) -> String;
    fn raw_image_stack(&self) -> String;
    fn set_displayed(&mut self, displayed: bool);
    fn views_to_skip(&self, axis: &str, validate: bool) -> Result<String, String>;
    fn exit_state(&self) -> String;
    fn container_id(&self) -> String;
}

/// Explicit host boundary for operations that were formerly delegated to
/// `ApplicationManager`, `SetupReconUIHarness`, and tilt-angle Swing panels.
pub trait SetupDialogExpertHost {
    fn automation(&mut self, directory: Option<&Path>, dataset: Option<&str>);
    fn set_tilt_angle_fields(&mut self, axis: &str, specification: &str);
    fn get_tilt_angle_fields(&self, axis: &str, validate: bool) -> Result<String, String>;
    fn tilt_angle_type(&self, axis: &str) -> String;
    fn initialize_fields(&mut self);
    fn set_axis_enabled(&mut self, axis: &str, enabled: bool);
    fn view_raw_stack(&mut self, extension: &str, axis: &str, menu_options: Option<&str>);
    fn action(&mut self, command: &str);
    fn load_header(&mut self);
    fn update_template_values(&mut self, directive_files: Option<&str>);
    fn set_tooltips(&mut self);
}

/// Rust coordinator for Java `SetupDialogExpert`.  Platform UI, process, and
/// metadata services enter through the two narrow boundaries above.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SetupDialogExpertCoordinator {
    dir: Option<PathBuf>,
}

impl SetupDialogExpertCoordinator {
    #[allow(non_snake_case)]
    pub fn getInstance(directory: Option<PathBuf>) -> Self {
        Self { dir: directory }
    }

    #[allow(non_snake_case)]
    pub fn doAutomation<H: SetupDialogExpertHost>(
        &mut self,
        host: &mut H,
        directory: Option<PathBuf>,
        dataset: Option<&str>,
    ) {
        if directory.is_some() {
            self.dir = directory;
        }
        host.automation(self.dir.as_deref(), dataset);
    }

    #[allow(non_snake_case)]
    pub fn getDirectiveFileCollection<D: SetupDialogExpertDialog>(
        &self,
        dialog: &D,
    ) -> Option<String> {
        dialog.directive_file_collection()
    }

    #[allow(non_snake_case)]
    pub fn showProgressPanel<D: SetupDialogExpertDialog>(&self, dialog: &mut D) {
        dialog.show_progress_panel();
    }

    #[allow(non_snake_case)]
    pub fn updateDisplay<D: SetupDialogExpertDialog>(&self, dialog: &mut D, process_done: bool) {
        dialog.update_display(process_done);
    }

    #[allow(non_snake_case)]
    pub fn msgExcludeViewsSucceeded<D: SetupDialogExpertDialog>(
        &self,
        dialog: &mut D,
        axis: &str,
        running: bool,
        done: bool,
    ) {
        dialog.exclude_views_succeeded(axis, running, done);
    }

    #[allow(non_snake_case)]
    pub fn msgSetupReconFailed<D: SetupDialogExpertDialog>(&self, dialog: &mut D) {
        dialog.setup_recon_failed();
    }

    #[allow(non_snake_case)]
    pub fn getDir(&self) -> Option<&Path> {
        self.dir.as_deref()
    }

    #[allow(non_snake_case)]
    pub fn getDataset<D: SetupDialogExpertDialog>(&self, dialog: &D) -> String {
        dialog.dataset()
    }

    #[allow(non_snake_case)]
    pub fn getSetupReconInterface<'a, D: SetupDialogExpertDialog>(&self, dialog: &'a D) -> &'a D {
        dialog
    }

    #[allow(non_snake_case)]
    pub fn getExitState<D: SetupDialogExpertDialog>(&self, dialog: &D) -> String {
        dialog.exit_state()
    }

    #[allow(non_snake_case)]
    pub fn getRawImageStack<D: SetupDialogExpertDialog>(&self, dialog: &D) -> String {
        dialog.raw_image_stack()
    }

    #[allow(non_snake_case)]
    pub fn getWorkingDirectory<D: SetupDialogExpertDialog>(&self, dialog: &D) -> Option<PathBuf> {
        let stack = PathBuf::from(dialog.raw_image_stack());
        if stack.is_absolute() {
            stack.parent().map(Path::to_path_buf)
        } else {
            self.dir
                .as_ref()
                .cloned()
                .or_else(|| stack.parent().map(Path::to_path_buf))
        }
    }

    #[allow(non_snake_case)]
    pub fn setDisplayed<D: SetupDialogExpertDialog>(&self, dialog: &mut D, displayed: bool) {
        dialog.set_displayed(displayed);
    }

    #[allow(non_snake_case)]
    pub fn getContainer<D: SetupDialogExpertDialog>(&self, dialog: &D) -> String {
        dialog.container_id()
    }

    #[allow(non_snake_case)]
    pub fn setTooltips<H: SetupDialogExpertHost>(&self, host: &mut H) {
        host.set_tooltips();
    }

    #[allow(non_snake_case)]
    pub fn setTiltAngleFields<H: SetupDialogExpertHost>(
        &self,
        host: &mut H,
        axis: &str,
        spec: &str,
    ) {
        host.set_tilt_angle_fields(axis, spec);
    }

    #[allow(non_snake_case)]
    pub fn initializeFields<H: SetupDialogExpertHost>(&self, host: &mut H) {
        host.initialize_fields();
    }

    #[allow(non_snake_case)]
    pub fn getViewsToSkip<D: SetupDialogExpertDialog>(
        &self,
        dialog: &D,
        axis: &str,
        validate: bool,
    ) -> Result<String, String> {
        dialog.views_to_skip(axis, validate)
    }

    #[allow(non_snake_case)]
    pub fn getTiltAngleType<H: SetupDialogExpertHost>(&self, host: &H, axis: &str) -> String {
        host.tilt_angle_type(axis)
    }

    #[allow(non_snake_case)]
    pub fn getTiltAngleFields<H: SetupDialogExpertHost>(
        &self,
        host: &H,
        axis: &str,
        validate: bool,
    ) -> Result<String, String> {
        host.get_tilt_angle_fields(axis, validate)
    }

    /// The caller supplies decoded EDF stack references; file discovery and
    /// locking stay in the storage adapter rather than the UI coordinator.
    #[allow(non_snake_case)]
    pub fn checkForSharedDirectory<I, P>(&self, proposed: I, saved: I) -> bool
    where
        I: IntoIterator<Item = P>,
        P: AsRef<Path>,
    {
        let proposed: Vec<PathBuf> = proposed
            .into_iter()
            .map(|p| p.as_ref().to_path_buf())
            .collect();
        saved.into_iter().any(|path| {
            let path = path.as_ref();
            path.exists()
                && !proposed
                    .iter()
                    .any(|candidate| candidate.file_name() == path.file_name())
        })
    }
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
