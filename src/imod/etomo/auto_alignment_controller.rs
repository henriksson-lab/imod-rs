//! `IMOD/Etomo/src/etomo/AutoAlignmentController.java`.
//!
//! The auto-alignment logic shared by the Join and Serial Sections interfaces:
//! `xfalign` (initial and refine), the sample `midas`, reverting the transforms to the
//! midas output or to no transforms, and the boundary model.
//!
//! **Threads.**  The controller is built on the event dispatch thread by the manager
//! and kept for the run; its process manager's `postProcess`/`errorProcess` call back
//! into it on process threads (`copyXfFile`, `msgProcessEnded`).  So it is shared
//! (`&'static`, `Send + Sync`), its display (the dialog, an EDT object) is held in an
//! [`EdtRef`], and `msgProcessEnded` reaches the display on the event dispatch thread.

use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::{Arc, OnceLock};

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::midas_param::{self, MidasParam};
use crate::imod::etomo::comscript::xfalign_param::{self, XfalignParam};
use crate::imod::etomo::process::auto_alignment_process_manager::AutoAlignmentProcessManager;
use crate::imod::etomo::process::base_imod_manager::ImodManagerException;
use crate::imod::etomo::process::base_process_manager::BaseProcessManager;
use crate::imod::etomo::process::imod_manager::{self, ImodManager};
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::process_series::ProcessSeriesHandle;
use crate::imod::etomo::storage::log_file::LogFileError;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::file_type::{self, FileType};
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::view_type::ViewType;
use crate::imod::etomo::ui::auto_alignment_display::AutoAlignmentDisplay;
use crate::imod::etomo::ui::swing::ui_harness;
use crate::imod::etomo::util::event_queue::{self, EdtRef};
use crate::imod::etomo::util::utilities;

/// Java private static final `LOCAL_TRANSFORMATION_LIST_FILES`.
fn local_transformation_list_files() -> [Option<Arc<FileType>>; 4] {
    [
        Some(Arc::clone(&file_type::CLASS.local_transformation_list)),
        Some(Arc::clone(&file_type::CLASS.auto_local_transformation_list)),
        Some(Arc::clone(&file_type::CLASS.empty_local_transformation_list)),
        Some(Arc::clone(&file_type::CLASS.midas_local_transformation_list)),
    ]
}

/// Java `public final class AutoAlignmentController`.
pub struct AutoAlignmentController {
    /// Java private final `manager`.
    manager: &'static dyn BaseManager,
    /// Java private final `display` (an event dispatch thread object).
    display: EdtRef<dyn AutoAlignmentDisplay>,
    /// Java private final `axisID`.
    axis_id: AxisID,
    /// Java private final `processManager`, built in the constructor once the
    /// controller exists (the process manager keeps the controller).
    process_manager: OnceLock<&'static AutoAlignmentProcessManager>,
    /// Java private final `imodManager`.
    imod_manager: &'static ImodManager,
    /// Java private final `serialSectionRawStackName`.
    serial_section_raw_stack_name: Option<String>,
}

/// Owns every controller this module builds (the managers keep theirs for the run).
static INSTANCES: std::sync::Mutex<Vec<&'static AutoAlignmentController>> =
    std::sync::Mutex::new(Vec::new());

impl AutoAlignmentController {
    /// Java `AutoAlignmentController(BaseManager, AutoAlignmentDisplay, ImodManager,
    /// String)`.  Called on the event dispatch thread.
    pub fn new(
        manager: &'static dyn BaseManager,
        display: Rc<dyn AutoAlignmentDisplay>,
        imod_manager: &'static ImodManager,
        serial_section_raw_stack_name: Option<&str>,
    ) -> &'static AutoAlignmentController {
        let axis_id = display.get_axis_id();
        let controller: &'static AutoAlignmentController =
            Box::leak(Box::new(AutoAlignmentController {
                manager,
                display: EdtRef::new(display),
                axis_id,
                process_manager: OnceLock::new(),
                imod_manager,
                serial_section_raw_stack_name: serial_section_raw_stack_name.map(str::to_owned),
            }));
        INSTANCES.lock().unwrap().push(controller);
        let _ = controller
            .process_manager
            .set(AutoAlignmentProcessManager::new(manager, controller));
        controller
    }

    /// Java field read `processManager`.
    fn process_manager(&self) -> &'static AutoAlignmentProcessManager {
        self.process_manager.get().expect("processManager")
    }

    /// Java `xfalignInitial(ProcessSeries, boolean)`.
    pub fn xfalign_initial(
        &self,
        process_series: Option<&ProcessSeriesHandle>,
        join_interface: bool,
    ) {
        if !self.update_meta_data(true) {
            if let Some(process_series) = process_series {
                process_series.borrow().end_series();
            }
            return;
        }
        let Some(auto_alignment_meta_data) = self.manager.get_auto_alignment_meta_data() else {
            // Fixed in translation: a manager without auto-alignment meta data makes
            // `XfalignParam` throw NullPointerException; nothing is run here.
            if let Some(process_series) = process_series {
                process_series.borrow().end_series();
            }
            return;
        };
        let mut xfalign_param = XfalignParam::new(
            self.manager,
            auto_alignment_meta_data,
            xfalign_param::Mode::Initial,
            join_interface,
        );
        if !self
            .display
            .get()
            .get_auto_alignment_parameters_xfalign(&mut xfalign_param, true)
        {
            if let Some(process_series) = process_series {
                process_series.borrow().end_series();
            }
            return;
        }
        match self.process_manager().xfalign(
            Arc::new(xfalign_param),
            self.axis_id,
            process_series.map(|process_series| Arc::new(EdtRef::new(Rc::clone(process_series)))),
        ) {
            Ok(thread_name) => self
                .manager
                .set_thread_name(Some(&thread_name), Some(self.axis_id)),
            Err(except) => {
                eprintln!("{except}");
                ui_harness::with(|harness| {
                    harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self.manager),
                        &format!("Can't run initial xfalign\n{}", except),
                        "SystemProcessException",
                        Some(self.axis_id),
                    )
                });
                self.display.get().msg_process_ended();
                return;
            }
        }
        if let Some(main_panel) = self.manager.get_main_panel() {
            main_panel
                .main_panel()
                .start_progress_bar_string_axis_id_process_name(
                    Some("Initial xfalign"),
                    self.axis_id,
                    Some(&ProcessName::XFALIGN),
                );
        }
    }

    /// Java `xfalignRefine(ProcessSeries, boolean, String)`.
    pub fn xfalign_refine(
        &self,
        process_series: Option<&ProcessSeriesHandle>,
        join_interface: bool,
        description: &str,
    ) {
        if !self.update_meta_data(true) {
            if let Some(process_series) = process_series {
                process_series.borrow().end_series();
            }
            return;
        }
        let Some(auto_alignment_meta_data) = self.manager.get_auto_alignment_meta_data() else {
            // Fixed in translation: see `xfalign_initial`.
            if let Some(process_series) = process_series {
                process_series.borrow().end_series();
            }
            return;
        };
        let mut xfalign_param = XfalignParam::new(
            self.manager,
            auto_alignment_meta_data,
            xfalign_param::Mode::Refine,
            join_interface,
        );
        if !self
            .display
            .get()
            .get_auto_alignment_parameters_xfalign(&mut xfalign_param, true)
        {
            if let Some(process_series) = process_series {
                process_series.borrow().end_series();
            }
            return;
        }
        if !self.copy_most_recent_xf_file(description) {
            if let Some(process_series) = process_series {
                process_series.borrow().end_series();
            }
            return;
        }
        match self.process_manager().xfalign(
            Arc::new(xfalign_param),
            self.axis_id,
            process_series.map(|process_series| Arc::new(EdtRef::new(Rc::clone(process_series)))),
        ) {
            Ok(thread_name) => self
                .manager
                .set_thread_name(Some(&thread_name), Some(self.axis_id)),
            Err(except) => {
                eprintln!("{except}");
                ui_harness::with(|harness| {
                    harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self.manager),
                        &format!("Can't run {}\n{}", description, except),
                        "SystemProcessException",
                        Some(self.axis_id),
                    )
                });
                return;
            }
        }
        if let Some(main_panel) = self.manager.get_main_panel() {
            main_panel
                .main_panel()
                .start_progress_bar_string_axis_id_process_name(
                    Some("Refine xfalign"),
                    self.axis_id,
                    Some(&ProcessName::XFALIGN),
                );
        }
    }

    /// Java `revertXfFileToMidas()`.
    pub fn revert_xf_file_to_midas(&self) {
        let midas_output_file = file_type::CLASS
            .midas_local_transformation_list
            .get_file(Some(self.manager), Some(self.axis_id));
        BaseProcessManager::touch(
            &midas_output_file
                .as_ref()
                .map(|file| utilities::java_io_file_get_absolute_path(&file.to_string_lossy()))
                .unwrap_or_default(),
            Some(self.manager),
        );
        self.copy_xf_file(midas_output_file.as_deref());
    }

    /// Java `revertXfFileToEmpty()`.
    pub fn revert_xf_file_to_empty(&self) {
        let empty_file = file_type::CLASS
            .empty_local_transformation_list
            .get_file(Some(self.manager), Some(self.axis_id));
        BaseProcessManager::touch(
            &empty_file
                .as_ref()
                .map(|file| utilities::java_io_file_get_absolute_path(&file.to_string_lossy()))
                .unwrap_or_default(),
            Some(self.manager),
        );
        self.copy_xf_file(empty_file.as_deref());
    }

    /// Java `msgProcessEnded()`.  Called from process threads, so the display is
    /// reached on the event dispatch thread.
    pub fn msg_process_ended(&'static self) {
        if self.display.is_owner_thread() {
            self.display.get().msg_process_ended();
            return;
        }
        event_queue::invoke_later(move || self.display.get().msg_process_ended());
    }

    /// Java `imodBoundaryModel(Run3dmodMenuOptions)`.
    pub fn imod_boundary_model(&self, menu_options: Option<Run3dmodMenuOptions>) {
        let result = if self.manager.get_view_type() == ViewType::Montage {
            self.imod_manager
                .open_string_axis_id_string_boolean_run3dmod_menu_options(
                    imod_manager::PREBLEND_KEY,
                    Some(self.axis_id),
                    file_type::CLASS
                        .auto_align_boundary_model
                        .get_file_name(Some(self.manager), Some(self.axis_id))
                        .as_deref(),
                    true,
                    menu_options,
                )
        } else {
            let raw_stack = PathBuf::from(utilities::java_io_file_new(
                &self
                    .manager
                    .get_property_user_dir()
                    .unwrap_or_else(|| "null".to_string()),
                self.serial_section_raw_stack_name
                    .as_deref()
                    .unwrap_or("null"),
            ));
            self.imod_manager
                .open_string_axis_id_file_string_boolean_run3dmod_menu_options(
                    imod_manager::RAW_STACK_KEY,
                    Some(self.axis_id),
                    Some(&raw_stack),
                    file_type::CLASS
                        .auto_align_boundary_model
                        .get_file_name(Some(self.manager), Some(self.axis_id))
                        .as_deref(),
                    true,
                    menu_options,
                )
        };
        let open_message = |message: &str, title: &str| {
            ui_harness::with(|harness| {
                harness.open_message_dialog_base_manager_string_string_axis_id(
                    Some(self.manager),
                    message,
                    title,
                    Some(self.axis_id),
                )
            })
        };
        match result {
            Ok(()) => {}
            Err(ImodManagerException::AxisType(except)) => {
                eprintln!("{except}");
                open_message(&except.to_string(), "AxisType problem");
            }
            Err(ImodManagerException::SystemProcess(except)) => {
                eprintln!("{except}");
                open_message(&except.to_string(), "Can't open 3dmod with the tomogram");
            }
            Err(ImodManagerException::Io(e)) => {
                eprintln!("{e}");
                open_message(&e.to_string(), "IO Exception");
            }
            // An uncaught RuntimeException: Swing's handler prints it.
            Err(ImodManagerException::Runtime(message)) => eprintln!("{message}"),
        }
    }

    /// Java `midasSample(String)`.  Run midas on the sample.
    pub fn midas_sample(&self, description: &str) {
        if !self.update_meta_data(true) {
            return;
        }
        let mut midas_param = MidasParam::new(self.manager, self.axis_id, midas_param::Mode::Sample);
        self.display
            .get()
            .get_auto_alignment_parameters_midas(&mut midas_param);
        if !self.copy_most_recent_xf_file(description) {
            return;
        }
        if let Err(except) = self.process_manager().midas_sample(Arc::new(midas_param)) {
            eprintln!("{except}");
            ui_harness::with(|harness| {
                harness.open_message_dialog_base_manager_string_string_axis_id(
                    Some(self.manager),
                    &format!("Can't run{}\n{}", description, except),
                    "SystemProcessException",
                    Some(self.axis_id),
                )
            });
        }
    }

    /// Java `createEmptyXfFile() throws LockException`.  Create the empty xf file and
    /// copy it to root.xf.  This function may before the dataset name is set.  The
    /// declared `LockException` is never thrown by the body.
    pub fn create_empty_xf_file(&self) -> Result<(), LogFileError> {
        let empty_xf_file = file_type::CLASS
            .empty_local_transformation_list
            .get_file(Some(self.manager), Some(self.axis_id));
        if let Some(empty_xf_file) = empty_xf_file.as_ref()
            && !empty_xf_file.exists()
        {
            let empty_line = "   1.0000000   0.0000000   0.0000000   1.0000000       0.000       0.000";
            // `BufferedWriter.newLine()` writes the platform line separator.
            if let Err(e) = std::fs::write(
                empty_xf_file,
                format!("{empty_line}\n{empty_line}\n{empty_line}\n"),
            ) {
                eprintln!("{e}");
                return Ok(());
            }
        }
        let xf_file = file_type::CLASS
            .local_transformation_list
            .get_file(Some(self.manager), Some(self.axis_id));
        if !xf_file.as_ref().is_some_and(|xf_file| xf_file.exists()) {
            match utilities::copy_file(
                Some(self.manager),
                Some(self.axis_id),
                empty_xf_file.as_deref(),
                xf_file.as_deref(),
                false,
                false,
                false,
            ) {
                Ok(()) => {}
                Err(LogFileError::Lock(e)) => return Err(LogFileError::Lock(e)),
                Err(e) => eprintln!("{e:?}"),
            }
        }
        Ok(())
    }

    /// Java `copyXfFile(File)`.  Runs on the event dispatch thread or on a process
    /// thread (`AutoAlignmentProcessManager.postProcess`).
    pub fn copy_xf_file(&self, xf_output_file: Option<&Path>) {
        let xf_file = file_type::CLASS
            .local_transformation_list
            .get_file(Some(self.manager), Some(self.axis_id));
        if let Some(xf_output_file) = xf_output_file
            && xf_output_file.exists()
            && let Err(e) = utilities::copy_file(
                Some(self.manager),
                Some(self.axis_id),
                Some(xf_output_file),
                xf_file.as_deref(),
                false,
                false,
                false,
            )
        {
            eprintln!("{e:?}");
            let xf_file_name = xf_file
                .as_ref()
                .map(|xf_file| utilities::java_io_file_get_name(&xf_file.to_string_lossy()))
                .unwrap_or_else(|| "null".to_string());
            let message = [
                format!(
                    "Unable to copy {} to {}.",
                    utilities::java_io_file_get_absolute_path(&xf_output_file.to_string_lossy()),
                    xf_file_name
                ),
                format!(
                    "Copy {} to {}.",
                    utilities::java_io_file_get_name(&xf_output_file.to_string_lossy()),
                    xf_file_name
                ),
            ];
            ui_harness::open_message_dialog_array_from_process(
                Some(self.manager),
                &message,
                "Cannot Copy File",
                Some(self.axis_id),
            );
        }
    }

    /// Java private `updateMetaData(boolean)`.
    fn update_meta_data(&self, do_validation: bool) -> bool {
        if !self.manager.update_meta_data(
            Some(self.display.get().get_dialog_type()),
            Some(self.axis_id),
            do_validation,
        ) {
            return false;
        }
        // Fixed in translation: a manager without meta data makes the source throw
        // NullPointerException; the update fails here.
        let Some(meta_data) = self.manager.get_base_meta_data() else {
            return false;
        };
        if !meta_data.is_valid() {
            ui_harness::with(|harness| {
                harness.open_message_dialog_base_manager_string_string_axis_id(
                    Some(self.manager),
                    &meta_data.base().get_invalid_reason(),
                    "Invalid Data",
                    Some(self.axis_id),
                )
            });
            return false;
        }
        if !self
            .manager
            .save_meta_data_to_parameter_store(Some(self.axis_id))
        {
            return false;
        }
        true
    }

    /// Java package-private `copyMostRecentXfFile(String)`.
    pub(crate) fn copy_most_recent_xf_file(&self, command_description: &str) -> bool {
        let files = local_transformation_list_files();
        let newest_xf_file_type = utilities::most_recent_file_type(
            Some(self.manager),
            Some(self.axis_id),
            Some(&files),
            2, /* EMPTY_LOCAL_TRANSFORMATION_LIST */
        );
        // If the most recent .xf file is not root.xf, copy it to root.xf
        let Some(newest_xf_file_type) = newest_xf_file_type else {
            return true;
        };
        if !Arc::ptr_eq(
            &newest_xf_file_type,
            &file_type::CLASS.local_transformation_list,
        ) && let Err(e) = utilities::copy_file_file_types(
            &newest_xf_file_type,
            &file_type::CLASS.local_transformation_list,
            Some(self.manager),
            Some(self.axis_id),
            false,
            false,
            false,
        ) {
            eprintln!("{e:?}");
            let local_name = file_type::CLASS
                .local_transformation_list
                .get_file_name(Some(self.manager), Some(self.axis_id))
                .unwrap_or_else(|| "null".to_string());
            let message = [
                format!(
                    "Unable to copy {} to {}.",
                    newest_xf_file_type
                        .get_file(Some(self.manager), Some(self.axis_id))
                        .map(|file| utilities::java_io_file_get_absolute_path(
                            &file.to_string_lossy()
                        ))
                        .unwrap_or_else(|| "null".to_string()),
                    local_name
                ),
                format!(
                    "Copy {} to {}",
                    newest_xf_file_type
                        .get_file_name(Some(self.manager), Some(self.axis_id))
                        .unwrap_or_else(|| "null".to_string()),
                    local_name
                ),
                format!(" and then rerun {}.", command_description),
            ];
            ui_harness::with(|harness| {
                harness.open_message_dialog_base_manager_string_array_string_axis_id(
                    Some(self.manager),
                    &message,
                    "Cannot Run Command",
                    Some(self.axis_id),
                )
            });
            return false;
        }
        true
    }
}
