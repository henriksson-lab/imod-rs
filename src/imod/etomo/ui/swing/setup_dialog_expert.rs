//! `IMOD/Etomo/src/etomo/ui/swing/SetupDialogExpert.java`.
//!
//! Java `public final class SetupDialogExpert`.  The expert, its `SetupDialog`
//! and its two `TiltAnglePanelExpert`s live on the event dispatch thread
//! (`util/event_queue.rs`).  The dialog is handed `this` at construction, so
//! the expert is built with `Rc::new_cyclic` and keeps a `this: Weak<Self>`;
//! every method takes `&self` because the dialog calls back into the expert.
//! The dialog is held as `Rc<SetupDialog>` (set once, after the expert
//! exists): its methods take `&self`, since a call such as
//! `buttonExecuteAction` reaches the manager, which reads the dialog again
//! through this expert.

use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::jdk::JComponent;
use crate::imod::etomo::local_arguments::LocalArguments;
use crate::imod::etomo::logic::validation_set::ValidationSet;
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::storage::directive_file_collection::DirectiveFileCollection;
use crate::imod::etomo::storage::etomo_file_filter::EtomoFileFilter;
use crate::imod::etomo::storage::parameter_store::ParameterStore;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::base_meta_data::BaseMetaData;
use crate::imod::etomo::r#type::const_meta_data::ConstMetaData;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::extension::{self, EXTENSION_DIVIDER, Extension};
use crate::imod::etomo::r#type::meta_data::MetaData;
use crate::imod::etomo::r#type::tilt_angle_spec::TiltAngleSpec;
use crate::imod::etomo::r#type::tilt_angle_type::TiltAngleType;
use crate::imod::etomo::r#type::user_configuration::UserConfiguration;
use crate::imod::etomo::r#type::view_type::ViewType;
use crate::imod::etomo::ui::setup_recon_interface::{
    DirectiveFileCollectionHandle, SetupReconInterface,
};
use crate::imod::etomo::ui::setup_recon_ui_harness::SetupReconUIHarness;
use crate::imod::etomo::ui::swing::axis_progress_panel::AxisProgressPanel;
use crate::imod::etomo::ui::swing::process_dialog::DialogExitState;
use crate::imod::etomo::ui::swing::setup_dialog::SetupDialog;
use crate::imod::etomo::ui::swing::tilt_angle_panel_expert::TiltAnglePanelExpert;
use crate::imod::etomo::ui::swing::ui_harness;
use crate::imod::etomo::util::shared_constants;
use crate::imod::etomo::util::utilities;
use std::cell::{OnceCell, RefCell};
use std::path::{Path, PathBuf};
use std::rc::{Rc, Weak};
use std::sync::Arc;

/// Java `public final class SetupDialogExpert`.
pub struct SetupDialogExpert {
    /// Java private final `tiltAnglePanelExpertA`.
    tilt_angle_panel_expert_a: Rc<TiltAnglePanelExpert>,
    /// Java private final `tiltAnglePanelExpertB`.
    tilt_angle_panel_expert_b: Rc<TiltAnglePanelExpert>,
    /// Java private final `dialog`.  Set once, right after the expert is
    /// built: `SetupDialog.getInstance(this, ...)` calls back into the
    /// expert while it constructs the dialog, so the expert must already
    /// exist as an `Rc` when the dialog is made.
    dialog: OnceCell<Rc<SetupDialog>>,
    /// Java private final `manager`.
    manager: &'static ApplicationManager,
    /// Java private final `setupUIHarness`.  The harness owns this expert
    /// (`SetupReconUIHarness.expert`), so the back reference is weak to keep
    /// the pair from being a reference cycle; the manager keeps the harness
    /// for the whole run, so it is always there while the expert is.
    setup_ui_harness: Weak<SetupReconUIHarness>,
    /// Java private `dir`.
    dir: RefCell<Option<PathBuf>>,
    /// Rust-only: the `this` the Java hands to `SetupDialog.getInstance`.
    this: Weak<SetupDialogExpert>,
}

impl SetupDialogExpert {
    /// Java private constructor `SetupDialogExpert(ApplicationManager,
    /// SetupReconUIHarness, boolean, ValidationSet, AxisProgressPanel)`.
    fn new(
        manager: &'static ApplicationManager,
        setup_ui_harness: Weak<SetupReconUIHarness>,
        calibration_available: bool,
        binning_validation_set: Arc<ValidationSet>,
        progress_panel: Rc<AxisProgressPanel>,
    ) -> Rc<SetupDialogExpert> {
        let tilt_angle_panel_expert_a = TiltAnglePanelExpert::new(manager, AxisID::First);
        let tilt_angle_panel_expert_b = TiltAnglePanelExpert::new(manager, AxisID::Second);
        let instance = Rc::new_cyclic(|this: &Weak<SetupDialogExpert>| SetupDialogExpert {
            tilt_angle_panel_expert_a,
            tilt_angle_panel_expert_b,
            dialog: OnceCell::new(),
            manager,
            setup_ui_harness,
            dir: RefCell::new(None),
            this: this.clone(),
        });
        let dialog = SetupDialog::get_instance(
            &instance,
            manager,
            AxisID::Only,
            DialogType::SetupRecon,
            calibration_available,
            binning_validation_set,
            progress_panel,
        );
        let _ = instance.dialog.set(dialog);
        instance
    }

    /// The Java `dialog` field (always set once construction is done).
    fn dialog(&self) -> &Rc<SetupDialog> {
        self.dialog
            .get()
            .expect("SetupDialogExpert.dialog is set by the constructor")
    }

    /// Java public static `getInstance(ApplicationManager, SetupReconUIHarness,
    /// boolean, ValidationSet, AxisProgressPanel)`.
    pub fn get_instance(
        manager: &'static ApplicationManager,
        setup_ui_harness: Weak<SetupReconUIHarness>,
        calibration_available: bool,
        binning_validation_set: Arc<ValidationSet>,
        progress_panel: Rc<AxisProgressPanel>,
    ) -> Rc<SetupDialogExpert> {
        let instance = SetupDialogExpert::new(
            manager,
            setup_ui_harness,
            calibration_available,
            binning_validation_set,
            progress_panel,
        );
        instance.set_tooltips();
        instance
    }

    /// Java `doAutomation(LocalArguments)`: process command line arguments
    /// that pertain to Setup Dialog.  May call functions in
    /// ApplicationManager.
    ///
    /// `EtomoDirector.INSTANCE.getArguments()` is a locked static here; the
    /// lock is taken per read and never held across a call into the dialog
    /// (`buttonExecuteAction` reaches the manager, which reads the arguments
    /// again).
    pub fn do_automation(&self, local_arguments: Option<&LocalArguments>) {
        // build and set dataset
        *self.dir.borrow_mut() = etomo_director::ARGUMENTS
            .lock()
            .unwrap()
            .get_dir()
            .map(Path::to_path_buf);
        if let Some(local_arguments) = local_arguments {
            *self.dir.borrow_mut() = local_arguments.arguments.get_dir().map(Path::to_path_buf);
            // Upstream bug fixed in translation (SetupDialogExpert.java:74):
            // `dir.getAbsolutePath()` throws a NullPointerException when the
            // local arguments carry no directory.  The line is printed with
            // "null" for the path instead, as Java string concatenation would
            // print a null reference.
            let dir = self.dir.borrow().clone();
            println!(
                "LocalArguments directory{}",
                match dir {
                    Some(dir) => utilities::java_io_file_get_absolute_path(&dir.to_string_lossy()),
                    None => "null".to_owned(),
                }
            );
        }
        let mut dataset: Option<String> = etomo_director::ARGUMENTS
            .lock()
            .unwrap()
            .get_raw_image_stack()
            .map(str::to_owned);
        if let Some(local_arguments) = local_arguments {
            dataset = local_arguments
                .arguments
                .get_raw_image_stack()
                .map(str::to_owned);
        }
        if let Some(dataset) = dataset {
            // If the directory was set and dataset is a file (not the dataset
            // name), pass the absolute path to the dialog.
            let dir = self.dir.borrow().clone();
            match dir {
                Some(dir) if Extension::is_input_image_file_path(&dataset) => {
                    self.dialog().set_raw_image_stack(Some(
                        &utilities::java_io_file_get_absolute_path(&utilities::java_io_file_new(
                            &dir.to_string_lossy(),
                            &dataset,
                        )),
                    ));
                }
                _ => {
                    let raw_image_stack = etomo_director::ARGUMENTS
                        .lock()
                        .unwrap()
                        .get_raw_image_stack()
                        .map(str::to_owned);
                    self.dialog()
                        .set_raw_image_stack(raw_image_stack.as_deref());
                }
            }
        }
        // check radio buttons
        let axis = etomo_director::ARGUMENTS.lock().unwrap().get_axis();
        if axis == Some(AxisType::SingleAxis) {
            self.dialog().set_single_axis(true);
        } else if axis == Some(AxisType::DualAxis) {
            self.dialog().set_dual_axis(true);
        }
        let frame = etomo_director::ARGUMENTS.lock().unwrap().get_frame();
        if let Some(frame) = frame {
            self.set_view_type(frame);
        }
        // fill in fields
        let fiducial = etomo_director::ARGUMENTS.lock().unwrap().get_fiducial();
        if let Some(fiducial) = fiducial {
            self.dialog().set_fiducial_diameter(fiducial);
        }
        // scan header
        let scan = etomo_director::ARGUMENTS.lock().unwrap().is_scan();
        if scan && let Some(setup_ui_harness) = self.setup_ui_harness.upgrade() {
            setup_ui_harness.scan_header_action(&**self.dialog(), false);
        }
        let cpus = etomo_director::ARGUMENTS.lock().unwrap().is_cpus();
        if cpus {
            self.dialog().set_parallel_process(true);
        }
        let gpus = etomo_director::ARGUMENTS.lock().unwrap().is_gpus();
        if gpus {
            self.dialog().set_gpu_processing(true);
        }
        // complete the dialog
        let create = etomo_director::ARGUMENTS.lock().unwrap().is_create();
        if create {
            self.dialog().button_execute_action();
        }
    }

    /// Java `getDirectiveFileCollection()`.
    pub fn get_directive_file_collection(&self) -> Option<DirectiveFileCollectionHandle> {
        self.dialog().get_directive_file_collection()
    }

    /// Java `showProgressPanel()`.
    pub fn show_progress_panel(&self) {
        self.dialog().show_progress_panel();
    }

    /// Java `updateDisplay(boolean)`.
    pub fn update_display(&self, process_done: bool) {
        self.dialog().update_display(false, process_done);
    }

    /// Java `msgExcludeViewsSucceeded(AxisID, boolean, boolean)`.
    pub fn msg_exclude_views_succeeded(
        &self,
        axis_id: AxisID,
        process_running: bool,
        process_done: bool,
    ) {
        self.dialog()
            .msg_exclude_views_succeeded(axis_id, process_running, process_done);
    }

    /// Java `msgSetupReconFailed()`.
    pub fn msg_setup_recon_failed(&self) {
        self.dialog().msg_setup_recon_failed();
    }

    /// Java `getDir()`.
    pub fn get_dir(&self) -> Option<PathBuf> {
        self.dir.borrow().clone()
    }

    /// Java `getDataset()` (deprecated 6/17/19).
    #[deprecated]
    pub fn get_dataset(&self) -> Option<String> {
        #[allow(deprecated)]
        self.dialog().get_dataset()
    }

    /// Java `getSetupReconInterface()`: the dialog, as the interface.
    pub fn get_setup_recon_interface(&self) -> Rc<dyn SetupReconInterface> {
        self.dialog().clone()
    }

    /// Java `checkForSharedDirectory(Extension)`: checks for an existing
    /// reconstruction on a different stack in the current directory.
    /// Assumes that the new .edf file for this instance has not been created
    /// yet.  Since .com file names are not stack specific, it is necessary to
    /// prevent interference by doing only one reconstruction per directory.
    /// A secondary goal is to have only one tilt series per directory.
    /// Multiple .edf files accessing the same stacks are allowed so that the
    /// user can back up their .edf file or start a fresh .edf file.  The user
    /// may also have one single and one dual reconstruction in a directory,
    /// as long as they have a stack in common.
    ///
    /// Returns true if there is already an .edf file in the propertyUserDir
    /// and it is referencing a stack other then the one(s) specified in the
    /// setup dialog.  True if the new .edf file and the existing .edf file
    /// are both single axis, even if one file is accessing the A stack and
    /// the other is accessing the B stack.  False if no existing .edf file is
    /// found.  False if the new .edf file and the existing .edf file are
    /// single and dual axis, as long as they have a stack in common.  False
    /// if there is a conflict, but the stacks that one of .edf files
    /// references don't exist.
    pub fn check_for_shared_directory(&self, raw_image_stack_extension: &Extension) -> bool {
        let property_user_dir = self.get_property_user_dir().unwrap_or_default();
        // Get all the edf files in propertyUserDir.  `File.listFiles` returns
        // null when the directory cannot be listed.
        let Ok(entries) = std::fs::read_dir(&property_user_dir) else {
            return false;
        };
        let filter = EtomoFileFilter;
        let edf_files: Vec<PathBuf> = entries
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| filter.accept(path))
            .collect();
        let Some(meta_data) = self.manager.get_base_meta_data() else {
            return false;
        };
        let dataset_name = meta_data.get_dataset_name().unwrap_or_default();
        let axis_type = meta_data.base().get_axis_type();
        let mut first_stack_name: Option<String> = None;
        let mut second_stack_name: Option<String> = None;
        // Create File instances based on the stacks specified in the setup
        // dialog
        let _raw_image_stack = self.get_raw_image_stack();
        if axis_type == AxisType::DualAxis {
            let first_stack = utilities::java_io_file_new(
                &property_user_dir,
                &format!(
                    "{dataset_name}{}{EXTENSION_DIVIDER}{raw_image_stack_extension}",
                    AxisID::First.get_extension()
                ),
            );
            first_stack_name = Some(utilities::java_io_file_get_name(&first_stack));
            let second_stack = utilities::java_io_file_new(
                &property_user_dir,
                &format!(
                    "{dataset_name}{}{EXTENSION_DIVIDER}{raw_image_stack_extension}",
                    AxisID::Second.get_extension()
                ),
            );
            second_stack_name = Some(utilities::java_io_file_get_name(&second_stack));
        } else if axis_type == AxisType::SingleAxis {
            let first_stack = utilities::java_io_file_new(
                &property_user_dir,
                &format!(
                    "{dataset_name}{}{EXTENSION_DIVIDER}{raw_image_stack_extension}",
                    AxisID::Only.get_extension()
                ),
            );
            first_stack_name = Some(utilities::java_io_file_get_name(&first_stack));
        }
        // open any .edf files in propertyUserDir - assuming the .edf file for
        // this instance hasn't been created yet.
        // If there is at least one .edf file that references existing stacks
        // that are not the stacks the will be used in this instance, then the
        // directory is already in use.
        // Doing a dual and single axis on the same stack is not sharing a
        // directory.
        // However doing two single axis reconstructions on the same tilt
        // series, where one is done on A and the other is done on B, would be
        // considered sharing a directory.
        for edf_file in &edf_files {
            let mut saved_meta_data =
                MetaData::new(Some(self.manager), self.manager.get_log_properties(), true);
            match ParameterStore::get_instance(Some(edf_file.clone())) {
                Ok(param_store) => {
                    if let Some(param_store) = param_store {
                        param_store.load(&mut saved_meta_data);
                    }
                }
                // `catch (LogFile.FileException | IOException e)`; the
                // `LockException` arm (`continue` silently) has no counterpart
                // in this `ParameterStore`, which takes no log-file lock.
                Err(e) => {
                    eprintln!("{e:?}");
                    ui_harness::open_message_dialog_from_process(
                        Some(self.manager),
                        &format!("Unable to read .edf files in {property_user_dir}"),
                        "Etomo Error",
                        None,
                    );
                    continue;
                }
            }

            // Create File instances based on the stacks specified in the edf
            // file found in propertyUserDir.
            let saved_axis_type = saved_meta_data.base().get_axis_type();
            let saved_dataset_name = saved_meta_data.get_dataset_name();
            // Upstream bug fixed in translation (SetupDialogExpert.java:216-221):
            // `savedMetaData.getRawImageStackExtension().toString()` throws a
            // NullPointerException when the saved extension is null, so the
            // following `== null` test (meant for a dataset that dates from
            // before the naming style upgrade) can never be reached.  A null
            // extension takes that fallback here: the old standard dataset
            // extension.
            let saved_raw_stack_extension = match saved_meta_data.get_raw_image_stack_extension() {
                Some(extension) => extension.to_string(),
                // This dataset appears to date from before the naming style
                // upgrade so use the old standard dataset extension.
                None => extension::CLASS.st.to_string(),
            };
            if saved_axis_type == AxisType::DualAxis {
                let saved_first_stack = utilities::java_io_file_new(
                    &property_user_dir,
                    &format!(
                        "{saved_dataset_name}{}{EXTENSION_DIVIDER}{saved_raw_stack_extension}",
                        AxisID::First.get_extension()
                    ),
                );
                let saved_first_stack_name = utilities::java_io_file_get_name(&saved_first_stack);
                let saved_second_stack = utilities::java_io_file_new(
                    &property_user_dir,
                    &format!(
                        "{saved_dataset_name}{}{EXTENSION_DIVIDER}{saved_raw_stack_extension}",
                        AxisID::Second.get_extension()
                    ),
                );
                let saved_second_stack_name = utilities::java_io_file_get_name(&saved_second_stack);
                if axis_type == AxisType::DualAxis {
                    // compare dual axis A against saved dual axis A
                    if Path::new(&saved_first_stack).exists()
                        && first_stack_name.as_deref() != Some(saved_first_stack_name.as_str())
                    {
                        return true;
                    }
                    // compare dual axis B against saved dual axis B
                    if Path::new(&saved_second_stack).exists()
                        && second_stack_name.as_deref() != Some(saved_second_stack_name.as_str())
                    {
                        return true;
                    }
                } else if axis_type == AxisType::SingleAxis {
                    // compare single axis against saved dual axis A
                    // compare single axis against saved dual axis B
                    if Path::new(&saved_first_stack).exists()
                        && first_stack_name.as_deref() != Some(saved_first_stack_name.as_str())
                        && (!Path::new(&saved_second_stack).exists()
                            || (Path::new(&saved_second_stack).exists()
                                && first_stack_name.as_deref()
                                    != Some(saved_second_stack_name.as_str())))
                    {
                        return true;
                    }
                }
            } else if saved_axis_type == AxisType::SingleAxis {
                let saved_first_stack = utilities::java_io_file_new(
                    &property_user_dir,
                    &format!(
                        "{saved_dataset_name}{}{EXTENSION_DIVIDER}{saved_raw_stack_extension}",
                        AxisID::Only.get_extension()
                    ),
                );
                let saved_first_stack_name = utilities::java_io_file_get_name(&saved_first_stack);
                if axis_type == AxisType::DualAxis {
                    // compare dual axis A against saved single axis
                    // compare dual axis B against saved single axis
                    if Path::new(&saved_first_stack).exists()
                        && first_stack_name.as_deref() != Some(saved_first_stack_name.as_str())
                        && second_stack_name.as_deref() != Some(saved_first_stack_name.as_str())
                    {
                        return true;
                    }
                } else if axis_type == AxisType::SingleAxis {
                    // compare single axis against saved single axis
                    if Path::new(&saved_first_stack).exists()
                        && first_stack_name.as_deref() != Some(saved_first_stack_name.as_str())
                    {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Java package-private `getPropertyUserDir()`.
    pub fn get_property_user_dir(&self) -> Option<String> {
        self.setup_ui_harness
            .upgrade()
            .and_then(|setup_ui_harness| setup_ui_harness.get_property_user_dir())
    }

    /// Java `getExitState()`.
    pub fn get_exit_state(&self) -> DialogExitState {
        self.dialog().get_exit_state()
    }

    /// Java `getRawImageStack()`.
    pub fn get_raw_image_stack(&self) -> Option<String> {
        self.dialog().get_raw_image_stack()
    }

    /// Java `getWorkingDirectory()`: return the working directory as a File
    /// object (`File.getParentFile()`, which may be null).
    pub fn get_working_directory(&self) -> Option<PathBuf> {
        let dataset_text = self.dialog().get_raw_image_stack().unwrap_or_default();
        let mut dataset = dataset_text.clone();
        if !Path::new(&dataset).is_absolute() {
            dataset = format!(
                "{}{}{}",
                self.get_property_user_dir().as_deref().unwrap_or("null"),
                std::path::MAIN_SEPARATOR,
                dataset_text
            );
        }
        utilities::java_io_file_get_parent(&dataset).map(PathBuf::from)
    }

    /// Java `setDisplayed(boolean)`.
    pub fn set_displayed(&self, displayed: bool) {
        self.dialog().set_displayed(displayed);
    }

    /// Java `setTiltAngleFields(AxisID, TiltAngleSpec, UserConfiguration)`.
    pub fn set_tilt_angle_fields(
        &self,
        axis_id: AxisID,
        tilt_angle_spec: &TiltAngleSpec,
        user_configuration: &UserConfiguration,
    ) {
        if axis_id == AxisID::Second {
            self.tilt_angle_panel_expert_b
                .set_fields(tilt_angle_spec, user_configuration);
        } else {
            self.tilt_angle_panel_expert_a
                .set_fields(tilt_angle_spec, user_configuration);
        }
    }

    /// Java `initializeFields(ConstMetaData, UserConfiguration)`.
    pub fn initialize_fields(
        &self,
        meta_data: &dyn ConstMetaData,
        user_config: &UserConfiguration,
    ) {
        // Java `!metaData.getDatasetName().equals("")`; a null name is treated
        // as empty rather than throwing.
        let dataset_name = meta_data.get_dataset_name();
        if dataset_name != "" {
            let canonical_path = format!(
                "{}/{}",
                self.get_property_user_dir().as_deref().unwrap_or("null"),
                dataset_name
            );
            self.dialog().set_raw_image_stack(Some(&canonical_path));
        }
        // Parallel processing is optional in tomogram reconstruction, so only
        // use it if the user set it up.
        let property_user_dir = self.get_property_user_dir();
        // TODO(unit): needs etomo/logic/UserEnv.java - `isParallelProcessing`,
        // `isGpuProcessingEnabled`, `isGpuProcessing`.
        self.dialog().set_parallel_process(
            crate::imod::etomo::logic::user_env::is_parallel_processing(
                self.manager,
                AxisID::Only,
                property_user_dir.as_deref(),
            ),
        );
        let property_user_dir = self.get_property_user_dir();
        self.dialog().set_gpu_processing_enabled(
            crate::imod::etomo::logic::user_env::is_gpu_processing_enabled(
                self.manager,
                AxisID::Only,
                property_user_dir.as_deref(),
            ),
        );
        let property_user_dir = self.get_property_user_dir();
        self.dialog()
            .set_gpu_processing(crate::imod::etomo::logic::user_env::is_gpu_processing(
                self.manager,
                AxisID::Only,
                property_user_dir.as_deref(),
            ));
        let dialog = self.dialog();
        dialog.set_backup_directory(Some(&meta_data.get_backup_directory()));
        dialog.set_distortion_file(Some(&meta_data.get_distortion_file()));
        dialog.set_mag_gradient_file(Some(&meta_data.get_mag_gradient_file()));
        dialog.set_adjusted_focus(AxisID::First, meta_data.get_adjusted_focus_a().is());
        dialog.set_adjusted_focus(AxisID::Second, meta_data.get_adjusted_focus_b().is());
        if meta_data.get_axis_type() == AxisType::SingleAxis || user_config.get_single_axis() {
            dialog.set_single_axis(true);
        } else {
            dialog.set_dual_axis(true);
        }
        if user_config.get_montage() {
            self.set_view_type(ViewType::Montage);
        } else {
            self.set_view_type(meta_data.get_view_type());
        }
        let dialog = self.dialog();
        if !meta_data.get_pixel_size().is_nan() {
            dialog.set_pixel_size(meta_data.get_pixel_size());
        }
        if meta_data.is_half_float_mode_output_set() {
            dialog.set_half_float_mode_output(meta_data.get_half_float_mode_output());
        }
        if !meta_data.get_fiducial_diameter().is_nan() {
            dialog.set_fiducial_diameter(meta_data.get_fiducial_diameter());
        }
        if !meta_data.get_image_rotation(AxisID::Only).is_null() {
            dialog.set_image_rotation(Some(
                &meta_data.get_image_rotation(AxisID::Only).to_string(),
            ));
        }
        dialog.set_binning(Some(&meta_data.get_binning()));
        dialog.set_exclude_list(AxisID::First, Some(&meta_data.get_exclude_projections_a()));
        dialog.set_exclude_list(AxisID::Second, Some(&meta_data.get_exclude_projections_b()));
        dialog.set_twodir_axis_id_boolean(AxisID::First, meta_data.is_twodir(AxisID::First));
        dialog.set_twodir_axis_id_string(AxisID::First, Some(&meta_data.get_twodir(AxisID::First)));
        dialog.set_twodir_axis_id_boolean(AxisID::Second, meta_data.is_twodir(AxisID::Second));
        dialog
            .set_twodir_axis_id_string(AxisID::Second, Some(&meta_data.get_twodir(AxisID::Second)));
        dialog.set_dose_sym_axis_id_boolean(AxisID::First, meta_data.is_dose_sym(AxisID::First));
        dialog.set_dose_sym_axis_id_string(
            AxisID::First,
            Some(&meta_data.get_dose_sym(AxisID::First)),
        );
        dialog.set_dose_sym_axis_id_boolean(AxisID::Second, meta_data.is_dose_sym(AxisID::Second));
        dialog.set_dose_sym_axis_id_string(
            AxisID::Second,
            Some(&meta_data.get_dose_sym(AxisID::Second)),
        );
        if meta_data.get_axis_type() == AxisType::SingleAxis || user_config.get_single_axis() {
            self.set_tilt_angle_panel_enabled(AxisID::Second, false);
        }
        self.dialog().set_parameters(user_config);
        self.dialog().checkpoint();
        self.tilt_angle_panel_expert_a.checkpoint();
        self.tilt_angle_panel_expert_b.checkpoint();
    }

    /// Java package-private `validateTiltAngle(AxisID, String)`.
    pub fn validate_tilt_angle(&self, axis_id: AxisID, error_title: &str) -> bool {
        self.get_tilt_angles_panel_expert(axis_id)
            .validate(error_title)
    }

    /// Java package-private `getTiltAnglesPanelExpert(AxisID)`.
    pub fn get_tilt_angles_panel_expert(&self, axis_id: AxisID) -> Rc<TiltAnglePanelExpert> {
        if axis_id == AxisID::Second {
            return self.tilt_angle_panel_expert_b.clone();
        }
        self.tilt_angle_panel_expert_a.clone()
    }

    /// Java `getContainer()`: `dialog.getContainer()`, the dialog's root
    /// container (`ProcessDialog.getContainer`).
    pub fn get_container(&self) -> Rc<JComponent> {
        self.dialog().get_container()
    }

    /// Java package-private `getAxisType()`.
    pub fn get_axis_type(&self) -> Option<AxisType> {
        self.setup_ui_harness
            .upgrade()
            .and_then(|setup_ui_harness| setup_ui_harness.get_axis_type())
    }

    /// Java package-private `getDatasetDir()`.
    pub fn get_dataset_dir(&self) -> Option<String> {
        if let Some(dir) = self.dir.borrow().as_ref() {
            return Some(utilities::java_io_file_get_absolute_path(
                &dir.to_string_lossy(),
            ));
        }
        etomo_director::INSTANCE.get_original_user_dir()
    }

    /// Java `getViewsToSkip(AxisID, boolean)`.
    pub fn get_views_to_skip(&self, axis_id: AxisID, do_validation: bool) -> Option<String> {
        self.dialog().get_views_to_skip(axis_id, do_validation)
    }

    /// Java `getTiltAngleType(AxisID)`.
    pub fn get_tilt_angle_type(&self, axis_id: AxisID) -> Option<TiltAngleType> {
        if axis_id == AxisID::Second {
            return self.tilt_angle_panel_expert_b.get_tilt_angle_type();
        }
        self.tilt_angle_panel_expert_a.get_tilt_angle_type()
    }

    /// Java `getTiltAngleFields(AxisID, TiltAngleSpec, boolean)`.
    pub fn get_tilt_angle_fields(
        &self,
        axis_id: AxisID,
        tilt_angle_spec: &mut TiltAngleSpec,
        do_validation: bool,
    ) -> Result<bool, String> {
        if axis_id == AxisID::Second {
            return self
                .tilt_angle_panel_expert_b
                .get_fields(tilt_angle_spec, do_validation);
        }
        self.tilt_angle_panel_expert_a
            .get_fields(tilt_angle_spec, do_validation)
    }

    /// Java package-private `getCurrentBackupDirectory()`.
    pub fn get_current_backup_directory(&self) -> Option<String> {
        // Open up the file chooser in the working directory
        let mut current_backup_directory = self.dialog().get_backup_directory();
        // Java `currentBackupDirectory.equals("")` would throw on null; a null
        // directory is treated as empty.
        if current_backup_directory.as_deref().unwrap_or("") == "" {
            current_backup_directory = etomo_director::INSTANCE.get_original_user_dir();
        }
        current_backup_directory
    }

    /// Java package-private `getCurrentMagGradientDir()`.
    pub fn get_current_mag_gradient_dir(&self) -> Option<String> {
        // Open up the file chooser in the calibration directory, if
        // available, otherwise open in the working directory
        let mut current_mag_gradient_dir = self.dialog().get_mag_gradient_file();
        // Java `currentMagGradientDir.equals("")` would throw on null; a null
        // file is treated as empty.
        if current_mag_gradient_dir.as_deref().unwrap_or("") == "" {
            let calibration_dir = etomo_director::INSTANCE.get_imod_calib_directory();
            // Upstream bug fixed in translation (SetupDialogExpert.java:449-450):
            // `calibrationDir.getAbsolutePath()` throws a NullPointerException
            // when there is no IMOD calibration directory.  With none, the
            // "Distortion" directory cannot exist, so the working directory is
            // used, as the source's own `else` arm does.
            let mag_gradient_dir = calibration_dir.map(|calibration_dir| {
                utilities::java_io_file_new(
                    &utilities::java_io_file_get_absolute_path(&calibration_dir.to_string_lossy()),
                    "Distortion",
                )
            });
            match mag_gradient_dir {
                Some(mag_gradient_dir) if Path::new(&mag_gradient_dir).exists() => {
                    current_mag_gradient_dir =
                        Some(utilities::java_io_file_get_absolute_path(&mag_gradient_dir));
                }
                _ => {
                    current_mag_gradient_dir = self.get_property_user_dir();
                }
            }
        }
        current_mag_gradient_dir
    }

    /// Java package-private `action(String)`.
    pub fn action(&self, action_command: &str) {
        if self
            .dialog()
            .equals_single_axis_action_command(action_command)
        {
            self.set_tilt_angle_panel_enabled(AxisID::Second, false);
            self.dialog().update_display(false, false);
        } else if self
            .dialog()
            .equals_dual_axis_action_command(action_command)
        {
            self.set_tilt_angle_panel_enabled(AxisID::Second, true);
            self.dialog().update_display(false, false);
        } else if self
            .dialog()
            .equals_single_view_action_command(action_command)
        {
            self.dialog()
                .set_adjusted_focus_enabled(AxisID::First, false);
            self.dialog()
                .set_adjusted_focus_enabled(AxisID::Second, false);
        } else if self.dialog().equals_montage_action_command(action_command) {
            self.dialog()
                .set_adjusted_focus_enabled(AxisID::First, true);
            self.dialog()
                .set_adjusted_focus_enabled(AxisID::Second, true);
        } else if self
            .dialog()
            .equals_scan_header_action_command(action_command)
        {
            if let Some(setup_ui_harness) = self.setup_ui_harness.upgrade() {
                setup_ui_harness.scan_header_action(&**self.dialog(), false);
            }
        } else if self.dialog().equals_template_action_command(action_command) {
            self.dialog().update_template_values();
        }
        self.dialog().update_display(false, false);
    }

    /// Java package-private `isFloatModeInput()`.
    pub fn is_float_mode_input(&self) -> bool {
        self.setup_ui_harness
            .upgrade()
            .is_some_and(|setup_ui_harness| setup_ui_harness.is_float_mode_input())
    }

    /// Java package-private `loadHeader()`.
    pub fn load_header(&self) {
        if let Some(setup_ui_harness) = self.setup_ui_harness.upgrade() {
            setup_ui_harness.scan_header_action(&**self.dialog(), true);
        }
    }

    /// Java package-private `updateTiltAnglePanelTemplateValues(
    /// DirectiveFileCollection)`.
    pub fn update_tilt_angle_panel_template_values(
        &self,
        directive_file_collection: &DirectiveFileCollection,
    ) {
        self.tilt_angle_panel_expert_a
            .update_template_values(directive_file_collection);
        self.tilt_angle_panel_expert_b
            .update_template_values(directive_file_collection);
    }

    /// Java package-private `setTiltAnglePanelEnabled(AxisID, boolean)`.
    pub fn set_tilt_angle_panel_enabled(&self, axis_id: AxisID, enable: bool) {
        if axis_id == AxisID::Second {
            self.tilt_angle_panel_expert_b.set_enabled(enable);
        } else {
            self.tilt_angle_panel_expert_a.set_enabled(enable);
        }
        let dialog = self.dialog();
        dialog.set_exclude_list_enabled(axis_id, enable);
        dialog.set_twodir_enabled(axis_id, enable);
        dialog.set_dose_sym_enabled(axis_id, enable);
        dialog.set_view_raw_stack_enabled(axis_id, enable);
    }

    /// Java package-private `viewRawStack(String, AxisID, Run3dmodMenuOptions)`.
    pub fn view_raw_stack(
        &self,
        file_extension: &str,
        axis_id: AxisID,
        menu_options: Option<Run3dmodMenuOptions>,
    ) {
        // A null `Run3dmodMenuOptions` means no menu options were chosen.
        let menu_options = menu_options.unwrap_or_default();
        if axis_id == AxisID::Second {
            self.manager
                .imod_preview(file_extension, AxisID::Second, menu_options);
        } else if self.get_axis_type() == Some(AxisType::SingleAxis) {
            self.manager
                .imod_preview(file_extension, AxisID::Only, menu_options);
        } else {
            self.manager
                .imod_preview(file_extension, AxisID::First, menu_options);
        }
    }

    /// Java private `setTooltips()`.
    fn set_tooltips(&self) {
        let dialog = self.dialog();
        dialog.set_raw_image_stack_tooltip(
            "Enter the name of one of the view data files. You can also select the view \
             data file by pressing the folder button.",
            "This button will open a file chooser dialog box allowing you to \
             select the view data file.",
        );
        dialog.set_backup_directory_tooltip(
            "Enter the name of the directory where you want the small data \
             files .com and .log files to be backed up.  You can use the \
             folder button on the right to create a new directory to \
             store the backups.",
            "This button will open a file chooser dialog box allowing you to \
             select and/or create the backup directory.",
        );
        dialog.set_scan_header_tooltip(
            "Attempt to extract pixel size and tilt axis rotation angle from data stack.",
        );
        dialog.set_axis_type_tooltip(
            "This radio button selector will choose whether the \
             data consists of one or two tilt axis.",
        );
        // TODO(unit): needs etomo/util/SharedConstants.java - `VIEW_TYPE_TOOLTIP`,
        // `DISTORTION_FIELD_TOOLTIP`, `IMAGES_ARE_BINNED_TOOLTIP`.
        dialog.set_view_type_tooltip(shared_constants::VIEW_TYPE_TOOLTIP);

        dialog.set_pixel_size_tooltip("Enter the view image pixel size in nanometers here.");
        dialog.set_half_float_mode_output_tooltip(
            "Output aligned stack and tomogram as 16-bit floats regardless of the input data \
             type.",
            "Output aligned stack and tomogram as 16-bit floats only if the raw stack is \
             floating point.",
        );
        dialog.set_fiducial_diameter_tooltip(
            "Enter the fiducial size in nanometers here, or 0 if there are no fiducials.",
        );
        dialog.set_image_rotation_tooltip(
            "Enter the view image rotation in degrees. This \
             is the rotation (CCW positive) from the Y-axis (the tilt axis \
             after the views are aligned) to the suspected tilt axis in the \
             unaligned views.",
        );
        dialog.set_distortion_file_tooltip(shared_constants::DISTORTION_FIELD_TOOLTIP);
        dialog.set_binning_tooltip(shared_constants::IMAGES_ARE_BINNED_TOOLTIP);

        self.tilt_angle_panel_expert_a.set_tooltips();
        self.tilt_angle_panel_expert_b.set_tooltips();

        let dialog = self.dialog();
        dialog.set_exclude_list_tooltip(
            "Enter the view images to <b>exclude</b> from the \
             processing of this axis.  Ranges are allowed, separate ranges by \
             commas.  For example to exclude the first four and last four \
             images of a 60 view stack enter 1-4,57-60.",
        );
        dialog.set_twodir_tooltip();
        dialog.set_execute_tooltip(
            "This button will create a new set of command scripts \
             overwriting any of the same name in the specified working \
             directory.  Be sure to save the data file after creating the \
             command script if you wish to keep the results.",
        );
        dialog.set_parallel_process_tooltip(
            "Sets the default for parallel processing \
             (distributing processes across multiple computers).",
        );
        dialog.set_gpu_processing_tooltip(
            "Sets the default for GPU processing (sending \
             processes to the graphics card).",
        );

        dialog.set_mag_gradient_file_tooltip(
            "OPTIONAL:  A file with magnification \
             gradients to be applied for each image.",
        );
        dialog.set_view_raw_stack_tooltip("View the current raw image stack.");
        dialog.set_adjusted_focus_tooltip(
            "Set this if \"Change focus with height\" was \
             selected when the montage was acquired in SerialEM.",
        );
        dialog.set_tooltips();
    }

    /// Java package-private `setViewType(ViewType)`: view type radio button.
    pub fn set_view_type(&self, view_type: ViewType) {
        if view_type == ViewType::SingleView {
            self.dialog().set_single_view(true);
        } else {
            self.dialog().set_montage(true);
        }
    }

    /// Rust-only: the `this` the Java passes as `SetupDialogExpert`.
    pub fn this(&self) -> Weak<SetupDialogExpert> {
        self.this.clone()
    }
}
