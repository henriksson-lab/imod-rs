//! `IMOD/Etomo/src/etomo/ToolsManager.java`.
//!
//! The manager of the Tools interface (front page "Flatten Volume", "Test
//! GPU" and "Align Frames"; the Tools menu).  It owns a `ToolsMetaData`, a
//! `ToolsProcessManager`, a `ToolsComScriptManager`, the `MainToolsPanel` and
//! the `ToolsDialog`.
//!
//! **Threads.**  The manager is a process-lifetime singleton shared with
//! process threads (`Send + Sync`); the main panel and the dialog are event
//! dispatch thread objects, held in `EdtCell`s and reached only on that
//! thread.  The align-frames displays the panel passes in are EDT objects as
//! well, and every method taking one runs on the EDT.
//!
//! **Java `null` axis.**  Several align-frames members pass a null `AxisID`
//! (to `ProcessSeries`, `startFailProcess`, `stopProgressBar`, the process
//! manager).  The Tools interface is single axis and those translated
//! routines take an axis by value, so they get `AxisID::Only`; dialogs keep
//! Java's null (`None`).

use std::convert::Infallible;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::{Arc, Mutex, OnceLock};

use crate::imod::etomo::base_manager::{BaseManager, BaseManagerBase};
use crate::imod::etomo::comscript::align_frames_param::AlignFramesParam;
use crate::imod::etomo::comscript::flatten_warp_param::FlattenWarpParam;
use crate::imod::etomo::comscript::gpu_tilt_test_param::GpuTiltTestParam;
use crate::imod::etomo::comscript::sort_tilt_frames_param::{self, SortTiltFramesParam};
use crate::imod::etomo::comscript::tomodataplots_param;
use crate::imod::etomo::comscript::tools_com_script_manager::ToolsComScriptManager;
use crate::imod::etomo::comscript::warp_vol_param::WarpVolParam;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::process::base_imod_manager::ImodManagerException;
use crate::imod::etomo::process::imod_manager;
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::process::process_interface::ProcessResultDisplayRef;
use crate::imod::etomo::process::tools_process_manager::ToolsProcessManager;
use crate::imod::etomo::process_series::{Process, ProcessSeries, ProcessSeriesHandle};
use crate::imod::etomo::storage::log_file::{LogFile, LogFileError};
use crate::imod::etomo::storage::storable::Storable;
use crate::imod::etomo::task_interface::TaskInterface;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::base_meta_data::BaseMetaData;
use crate::imod::etomo::r#type::data_file_type::DataFileType;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::extension;
use crate::imod::etomo::r#type::file_type::{self, FileType};
use crate::imod::etomo::r#type::interface_type::InterfaceType;
use crate::imod::etomo::r#type::process_end_state::ProcessEndState;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::process_result_display::ProcessResultDisplayHandle;
use crate::imod::etomo::r#type::tools_meta_data::ToolsMetaData;
use crate::imod::etomo::ui::UiComponent;
use crate::imod::etomo::ui::swing::align_frames_display::AlignFramesDisplay;
use crate::imod::etomo::ui::swing::deferred_3dmod_button::Deferred3dmodButton;
use crate::imod::etomo::ui::swing::etomo_menu::ToolType;
use crate::imod::etomo::ui::swing::flatten_warp_display::FlattenWarpDisplay;
use crate::imod::etomo::ui::swing::log_interface::LogInterface;
use crate::imod::etomo::ui::swing::log_window::LogWindow;
use crate::imod::etomo::ui::swing::main_panel::MainPanelVirtual;
use crate::imod::etomo::ui::swing::main_tools_panel::MainToolsPanel;
use crate::imod::etomo::ui::swing::process_display::ProcessDisplay;
use crate::imod::etomo::ui::swing::tools_dialog::ToolsDialog;
use crate::imod::etomo::ui::swing::ui_harness;
use crate::imod::etomo::ui::swing::warp_vol_display::WarpVolDisplay;
use crate::imod::etomo::util::event_queue::{self, EdtCell, EdtRef};
use crate::imod::etomo::util::mrc_header::MRCHeader;
use crate::imod::etomo::util::utilities;

/// Java `private static final AxisID AXIS_ID`.
pub const AXIS_ID: AxisID = AxisID::Only;
/// Java `private static final DialogType DIALOG_TYPE`.
pub const DIALOG_TYPE: DialogType = DialogType::Tools;
/// Java `private static final int STATUS_BAR_SIZE`.
pub const STATUS_BAR_SIZE: i32 = 65;

/// Java `public final class ToolsManager extends BaseManager`.
pub struct ToolsManager {
    base: BaseManagerBase,
    // <p>Updates done</p>
    /// Java private `alignFramesTiltAngleFile`, initially false.
    align_frames_tilt_angle_file: Mutex<bool>,
    /// Java private final `metaData`, assigned in the constructor body after
    /// `super()` (which needs the manager at its final address first).
    meta_data: OnceLock<ToolsMetaData>,
    /// Java private final `toolType`.
    tool_type: ToolType,
    /// Java private `mainPanel` (null in headless mode).
    main_panel: EdtCell<Rc<MainToolsPanel>>,
    /// Java private `processMgr`, set by the constructor.
    process_mgr: OnceLock<&'static ToolsProcessManager>,
    /// Java private `toolsDialog`, initially null.
    tools_dialog: EdtCell<Rc<ToolsDialog>>,
    /// Java private `comScriptMgr`, set by `createComScriptManager` (from
    /// `super()`).
    com_script_mgr: OnceLock<&'static ToolsComScriptManager>,
}

/// Owns every `ToolsManager` this module builds.  Java's owner is the collector, by
/// way of `EtomoDirector.managerList`, which keeps each manager for the run;
/// the translation hands out `&'static Self`, so without a root here the
/// allocation is unreachable the moment the constructor returns.
static INSTANCES: Mutex<Vec<&'static ToolsManager>> = Mutex::new(Vec::new());

impl ToolsManager {
    /// Java `ToolsManager(ToolType)`.
    pub fn new(tool_type: ToolType) -> &'static Self {
        let instance: &'static ToolsManager = Box::leak(Box::new(Self {
            base: BaseManagerBase::initial(),
            align_frames_tilt_angle_file: Mutex::new(false),
            meta_data: OnceLock::new(),
            tool_type,
            main_panel: EdtCell::new(),
            process_mgr: OnceLock::new(),
            tools_dialog: EdtCell::new(),
            com_script_mgr: OnceLock::new(),
        }));
        INSTANCES.lock().unwrap().push(instance);
        // `super()`, which calls createComScriptManager and createMainPanel.
        instance.base_manager();
        let _ = instance.meta_data.set(ToolsMetaData::new(
            Some(instance),
            DIALOG_TYPE,
            tool_type,
            instance.get_log_properties(),
            true,
        ));
        instance.create_state();
        let _ = instance.process_mgr.set(ToolsProcessManager::new(instance));
        instance.initialize_ui_parameters(None, Some(AXIS_ID), false);
        // Frame hasn't been created yet so stop here.
        instance
    }

    /// The constructed manager at its final address (Java `this` inside the
    /// overrides `super()` calls, which take `&self`).
    fn this_static(&self) -> &'static ToolsManager {
        INSTANCES
            .lock()
            .unwrap()
            .iter()
            .copied()
            .find(|manager| std::ptr::eq(*manager, self))
            .expect("constructed ToolsManager")
    }

    /// Java field read `metaData`.
    fn meta_data(&self) -> &ToolsMetaData {
        self.meta_data
            .get()
            .expect("metaData is assigned by the constructor")
    }

    /// Java field read `processMgr`.
    fn process_mgr(&self) -> &'static ToolsProcessManager {
        self.process_mgr.get().expect("processMgr")
    }

    /// Java field read `comScriptMgr`.
    fn com_script_mgr(&self) -> &'static ToolsComScriptManager {
        self.com_script_mgr.get().expect("comScriptMgr")
    }

    /// Java field read `propertyUserDir`.
    fn property_user_dir(&self) -> Option<String> {
        self.base().property_user_dir.lock().unwrap().clone()
    }

    /// Java `initialize()`.  Open panel and add dialog.
    pub fn initialize(&'static self) {
        if !etomo_director::ARGUMENTS.lock().unwrap().is_headless() {
            self.open_processing_panel();
            let mut location: Option<String> = None;
            let mut size = STATUS_BAR_SIZE;
            if self.tool_type == ToolType::GpuTiltTest {
                location = self.property_user_dir();
                size = 40;
            }
            if let Some(main_panel) = self.main_panel.get() {
                main_panel.set_status_bar_text(location.as_deref(), size);
            }
            self.open_tools_dialog();
            ui_harness::INSTANCE.with(|harness| harness.to_front(Some(self)));
        }
    }

    /// Java `isConflictingDatasetName(AxisID, File)`.  Checks for .edf, .ejf, or
    /// .epe files with the same dataset name (left side) as file.getName().
    /// Conflicting dataset names can cause file name collisions.  Returns true
    /// if there was a conflict.
    pub fn is_conflicting_dataset_name(&'static self, axis_id: AxisID, file: &Path) -> bool {
        let dir = file.parent().unwrap_or(Path::new(""));
        let name = utilities::java_io_file_get_name(&file.to_string_lossy());
        let filter = ConflictFileFilter::new(&name);
        // dir.listFiles(filter)
        let mut conflict_file_list: Vec<PathBuf> = Vec::new();
        if let Ok(entries) = std::fs::read_dir(if dir.as_os_str().is_empty() {
            Path::new(".")
        } else {
            dir
        }) {
            for entry in entries.flatten() {
                let path = entry.path();
                if filter.accept(&path) {
                    conflict_file_list.push(path);
                }
            }
        }
        // Upstream bug fixed in translation (ToolsManager.java:126): Java's
        // listFiles returns null for an unreadable directory and `.length`
        // throws NullPointerException; an unreadable directory has no
        // conflicts here.
        if !conflict_file_list.is_empty() {
            ui_harness::INSTANCE.with(|harness| {
                harness.open_message_dialog_base_manager_string_string_axis_id(
                    Some(self),
                    &format!(
                        "This file, {}, cannot be opened by the Tools interface in the this \
                         directory because the name conflicts with the dataset file {}.  \
                         Please copy {} to another directory and work on it there, or change \
                         its name.",
                        utilities::java_io_file_get_absolute_path(&file.to_string_lossy()),
                        utilities::java_io_file_get_name(&conflict_file_list[0].to_string_lossy()),
                        name
                    ),
                    "Entry Error",
                    Some(axis_id),
                )
            });
            return true;
        }
        false
    }

    // Updates done

    /// Java `setName(File)`.  Sets the dataset name.
    pub fn set_name(&'static self, input_file: &Path) {
        *self.base().property_user_dir.lock().unwrap() = input_file
            .parent()
            .map(|parent| parent.to_string_lossy().into_owned());
        self.meta_data().set_root_name_file(input_file);
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.set_status_bar_text(self.property_user_dir().as_deref(), STATUS_BAR_SIZE);
        }
        ui_harness::INSTANCE.with(|harness| {
            harness.set_title(
                Some(self),
                &format!(
                    "{} - {}",
                    self.tool_type.label(),
                    self.get_name().unwrap_or_else(|| "null".to_owned())
                ),
            )
        });
        // Open dialog tasks for for Flatten Volume
        if self.tool_type == ToolType::FlattenVolume {
            self.com_script_mgr().load_flatten(AxisID::Only);
            // Set from dataset_flatten.com. Dataset_flatten.com would only exist if
            // the Tools flatten functionality was used before on the same file in the
            // same directory.
            if self
                .com_script_mgr()
                .is_warp_vol_param_in_flatten(AxisID::Only)
            {
                let param = self
                    .com_script_mgr()
                    .get_warp_vol_param_from_flatten(AxisID::Only);
                // Upstream bug fixed in translation (ToolsManager.java:160): Java
                // dereferences toolsDialog, which is null in headless mode; no
                // dialog is set here.
                if let Some(tools_dialog) = self.tools_dialog.get() {
                    tools_dialog.set_parameters(&param);
                }
            }
        }
    }

    // ALIGN_FRAMES
    /// Java `openToolsDialog()`.
    pub fn open_tools_dialog(&'static self) {
        // if toolType = ALIGN_FRAMES,
        // check filetype.exists to check if outputAlignFrames.com exists
        // if exists, construct a file LogFile and do backup
        // call touch3d (creates empty file)
        // load toolsComScriptMgr->loadoutputParam(required=true)
        // exit if load fails
        if self.tool_type == ToolType::AlignFrames {
            // (The source's body here is commented out.)
        }
        if !self.tools_dialog.is_some() {
            self.tools_dialog.set(Some(ToolsDialog::get_instance(
                self,
                AXIS_ID,
                DIALOG_TYPE,
                self.tool_type,
            )));
        }
        // Upstream bug fixed in translation (ToolsManager.java:198): Java
        // dereferences mainPanel, which is null in headless mode; nothing is
        // shown here.
        if let (Some(main_panel), Some(tools_dialog)) =
            (self.main_panel.get(), self.tools_dialog.get())
        {
            main_panel.show_process(&tools_dialog.get_container(), AXIS_ID);
        }
        let action_message =
            utilities::prepare_dialog_action_message(Some(DialogType::Tools), AxisID::Only, None);
        if let Some(action_message) = action_message {
            eprintln!("{action_message}");
        }
    }

    /// Java `gpuTiltTestSuceeded(String[], AxisID)` (the source's spelling).
    pub fn gpu_tilt_test_suceeded(&'static self, output: Option<&[String]>, axis_id: AxisID) {
        if let Some(output) = output
            && !output.is_empty()
        {
            ui_harness::INSTANCE.with(|harness| {
                harness.open_info_message_dialog_base_manager_string_string_axis_id(
                    Some(self),
                    &output[output.len() - 1],
                    "Gputilttest Succeeded",
                    Some(axis_id),
                )
            });
        }
    }

    /// Java `gpuTiltTest(AxisID)`.
    pub fn gpu_tilt_test(&'static self, axis_id: AxisID) {
        let Some(param) = self.update_gpu_tilt_test(axis_id, true) else {
            return;
        };
        let thread_name = match self.process_mgr().gpu_tilt_test(&param, axis_id) {
            Ok(thread_name) => thread_name,
            Err(e) => {
                eprintln!("{e}");
                let message = [
                    format!("Can not execute {}", ProcessName::GPU_TILT_TEST),
                    e.to_string(),
                ];
                ui_harness::INSTANCE.with(|harness| {
                    harness.open_message_dialog_base_manager_string_array_string_axis_id(
                        Some(self),
                        &message,
                        "Unable to execute command",
                        Some(axis_id),
                    )
                });
                return;
            }
        };
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.start_progress_bar_string_axis_id_process_name(
                Some("Running gputilttest"),
                axis_id,
                Some(&ProcessName::GPU_TILT_TEST),
            );
        }
        self.set_thread_name(Some(&thread_name), Some(axis_id));
    }

    /// Java private `updateGpuTiltTest(AxisID, boolean)`.
    fn update_gpu_tilt_test(
        &'static self,
        axis_id: AxisID,
        do_validation: bool,
    ) -> Option<GpuTiltTestParam> {
        let mut param = GpuTiltTestParam::new();
        let Some(tools_dialog) = self.tools_dialog.get() else {
            self.open_message(
                "Unable to get information from the dialog.",
                "Etomo Error",
                axis_id,
            );
            return None;
        };
        if !tools_dialog.get_parameters(&mut param, do_validation) {
            return None;
        }
        Some(param)
    }

    /// Java `flatten`.  Execute flatten.com.
    #[allow(clippy::too_many_arguments)]
    pub fn flatten(
        &'static self,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
        dialog_type: Option<DialogType>,
        axis_id: AxisID,
        display: &dyn WarpVolDisplay,
    ) {
        let process_result_display: Option<ProcessResultDisplayRef> =
            process_result_display.map(|display| Arc::new(EdtRef::new(display)));
        let process_series = match process_series {
            Some(process_series) => process_series,
            None => ProcessSeries::new(self, axis_id, dialog_type, Some("flatten")),
        };
        self.send_msg_process_starting(process_result_display.as_ref());
        let Some(param) = self.update_warp_vol_param(Some(display), axis_id, false) else {
            process_series.borrow().end_series();
            return;
        };
        // Run process
        process_series
            .borrow_mut()
            .set_run_3dmod_deferred(deferred_3dmod_button, run_3dmod_menu_options);
        let thread_name = match self.process_mgr().flatten(
            Arc::new(param),
            axis_id,
            process_result_display,
            Some(Arc::new(EdtRef::new(Rc::clone(&process_series)))),
            &file_type::CLASS.flatten_tool_comscript,
        ) {
            Ok(thread_name) => thread_name,
            Err(e) => {
                eprintln!("{e}");
                let message = [
                    format!("Can not execute {}", ProcessName::FLATTEN),
                    e.to_string(),
                ];
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_array_string_axis_id(
                        Some(self),
                        &message,
                        "Unable to execute command",
                        Some(axis_id),
                    )
                });
                return;
            }
        };
        self.set_thread_name(Some(&thread_name), Some(axis_id));
    }

    /// Java private `updateWarpVolParam`.
    fn update_warp_vol_param(
        &'static self,
        display: Option<&dyn WarpVolDisplay>,
        axis_id: AxisID,
        do_validation: bool,
    ) -> Option<WarpVolParam> {
        let mut param = self
            .com_script_mgr()
            .get_warp_vol_param_from_flatten(axis_id);
        let Some(display) = display else {
            ui_harness::INSTANCE.with(|ui_harness| {
                ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                    Some(self),
                    "Unable to get information from the display.",
                    "Etomo Error",
                    Some(axis_id),
                )
            });
            return None;
        };
        if !display.get_parameters(&mut param, do_validation) {
            return None;
        }
        self.com_script_mgr().save_flatten(&param, axis_id);
        Some(param)
    }

    /// Java `imodFlatten`.  Open flatten.com output.
    pub fn imod_flatten(&'static self, menu_options: Option<Run3dmodMenuOptions>, axis_id: AxisID) {
        let key = imod_manager::FLATTEN_TOOL_OUTPUT_KEY;
        match self
            .get_imod_manager()
            .open_string_axis_id_run3dmod_menu_options(key, Some(axis_id), menu_options)
        {
            Ok(()) => {}
            Err(ImodManagerException::SystemProcess(except)) => {
                eprintln!("{except}");
                self.open_message(
                    &format!("{except}\nCan't open 3dmod on the {key}"),
                    "Cannot Open 3dmod",
                    axis_id,
                );
            }
            Err(ImodManagerException::AxisType(except)) => {
                eprintln!("{except}");
                self.open_message(&except.to_string(), "AxisType problem", axis_id);
            }
            Err(ImodManagerException::Io(e)) => {
                eprintln!("{e}");
                self.open_message(&e.to_string(), "IO Exception", axis_id);
            }
            // An uncaught RuntimeException: Swing's handler prints it.
            Err(ImodManagerException::Runtime(message)) => eprintln!("{message}"),
        }
    }

    /// Java `imodMakeSurfaceModel`.
    pub fn imod_make_surface_model(
        &'static self,
        menu_options: Option<Run3dmodMenuOptions>,
        axis_id: AxisID,
        binning: i32,
        file: Option<&Path>,
    ) {
        // Pick ImodManager key
        // Need to look at tomogram edge on. Use -Y, unless using squeezevol and it
        // is not flipped.
        let key = imod_manager::FLATTEN_INPUT_KEY;
        let use_swap_yz = self.is_flipped(file);
        let imod_manager = self.get_imod_manager();
        let result = (|| -> Result<(), ImodManagerException> {
            imod_manager.set_swap_yz_string_file_boolean(key, file, use_swap_yz)?;
            imod_manager.set_open_contours(key, Some(axis_id), true)?;
            imod_manager.set_start_new_contours_at_new_z(key, Some(axis_id), true)?;
            imod_manager.set_binning_xy_string_int(key, binning)?;
            imod_manager.open_string_file_string_boolean_run3dmod_menu_options(
                key,
                file,
                file_type::CLASS
                    .flatten_warp_input_model
                    .get_file_name(Some(self), Some(AXIS_ID))
                    .as_deref(),
                true,
                menu_options,
            )?;
            Ok(())
        })();
        match result {
            Ok(()) => {}
            Err(ImodManagerException::SystemProcess(except)) => {
                eprintln!("{except}");
                self.open_message(
                    &format!("{except}\nCan't open 3dmod on the {key}"),
                    "Cannot Open 3dmod",
                    axis_id,
                );
            }
            Err(ImodManagerException::AxisType(except)) => {
                eprintln!("{except}");
                self.open_message(&except.to_string(), "AxisType problem", axis_id);
            }
            Err(ImodManagerException::Io(e)) => {
                eprintln!("{e}");
                self.open_message(&e.to_string(), "IO Exception", axis_id);
            }
            // An uncaught RuntimeException: Swing's handler prints it.
            Err(ImodManagerException::Runtime(message)) => eprintln!("{message}"),
        }
    }

    /// Java private `isFlipped`.  Backward compatibility function to decide whether
    /// the file is flipped based on the header.
    ///
    /// Fixed in translation: Java dereferences a null file (`mrcFile.getAbsolutePath()`,
    /// NullPointerException); a missing file is "not flipped".
    fn is_flipped(&'static self, mrc_file: Option<&Path>) -> bool {
        let Some(mrc_file) = mrc_file else {
            return false;
        };
        let absolute_path = utilities::java_io_file_get_absolute_path(&mrc_file.to_string_lossy());
        let Some(header) = MRCHeader::get_instance_in_dir(
            self.get_property_user_dir().as_deref(),
            Some(&absolute_path),
            Some(AXIS_ID),
        ) else {
            return false;
        };
        match header.borrow_mut().read_with_manager(self) {
            Ok(true) => {}
            Ok(false) => return false,
            // catch (IOException e) / catch (Exception e)
            Err(_) => return false,
        }
        let header = header.borrow();
        let name = utilities::java_io_file_get_name(&mrc_file.to_string_lossy());
        if header.get_n_rows() < header.get_n_sections() {
            eprintln!(
                "Assuming that {name} has not been flipped\nbecause the Y is less then Z in the header."
            );
            return false;
        }
        eprintln!(
            "Assuming that {name} has been flipped\nbecause the Y is greater or equal to Z in the header."
        );
        true
    }

    /// Java `flattenWarp`.  Execute flattenwarp.
    #[allow(clippy::too_many_arguments)]
    pub fn flatten_warp(
        &'static self,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
        dialog_type: Option<DialogType>,
        axis_id: AxisID,
        display: &dyn FlattenWarpDisplay,
    ) {
        let busy_status_mediator = self.get_busy_status_mediator();
        let _monitor = busy_status_mediator.synchronized();
        let process_result_display: Option<ProcessResultDisplayRef> =
            process_result_display.map(|display| Arc::new(EdtRef::new(display)));
        let process_series = match process_series {
            Some(process_series) => process_series,
            None => ProcessSeries::new(self, axis_id, dialog_type, Some("flattenWarp")),
        };
        self.send_msg_process_starting(process_result_display.as_ref());
        let Some(param) = self.update_flatten_warp_param(Some(display), axis_id, true) else {
            process_series.borrow().end_series();
            return;
        };
        // Run process
        process_series
            .borrow_mut()
            .set_run_3dmod_deferred(deferred_3dmod_button, run_3dmod_menu_options);
        let thread_name = match self.process_mgr().flatten_warp(
            &param,
            process_result_display,
            Some(Arc::new(EdtRef::new(Rc::clone(&process_series)))),
            axis_id,
        ) {
            Ok(thread_name) => thread_name,
            Err(e) => {
                eprintln!("{e}");
                let message = [
                    format!("Can not execute {}", param.get_process_name()),
                    e.to_string(),
                ];
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_array_string_axis_id(
                        Some(self),
                        &message,
                        "Unable to execute command",
                        Some(axis_id),
                    )
                });
                return;
            }
        };
        self.set_thread_name(Some(&thread_name), Some(axis_id));
        if let Some(main_panel) = self.get_main_panel() {
            let process_name = param.get_process_name();
            main_panel
                .main_panel()
                .start_progress_bar_string_axis_id_process_name(
                    Some(&format!("Running {process_name}")),
                    axis_id,
                    Some(&process_name),
                );
        }
    }

    /// Java private `updateFlattenWarpParam`.
    fn update_flatten_warp_param(
        &'static self,
        display: Option<&dyn FlattenWarpDisplay>,
        axis_id: AxisID,
        do_validation: bool,
    ) -> Option<FlattenWarpParam> {
        let mut param = FlattenWarpParam::new(self);
        let Some(display) = display else {
            self.open_message(
                "Unable to get information from the display.",
                "Etomo Error",
                axis_id,
            );
            return None;
        };
        if !display.get_parameters(&mut param, do_validation) {
            return None;
        }
        Some(param)
    }

    /// `uiHarness.openMessageDialog(this, message, title, axisID)`.
    fn open_message(&'static self, message: &str, title: &str, axis_id: AxisID) {
        ui_harness::INSTANCE.with(|ui_harness| {
            ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                Some(self),
                message,
                title,
                Some(axis_id),
            )
        });
    }

    // ALIGN_FRAMES
    /// Java private `updateAlignFramesOutputParam(AlignFramesDisplay)`.
    fn update_align_frames_output_param(&'static self, display: &dyn AlignFramesDisplay) -> bool {
        // Try loading empty comfile to check loadAlignFramesOutput() works with empty files
        // as well

        let rootname = display.get_rootname_output_files();
        let property_user_dir = self.property_user_dir();
        // Upstream bug fixed in translation (ToolsManager.java:380): Java
        // dereferences the File getFile returns, which is null when no file
        // name can be built; the update fails here.
        let Some(file) = file_type::CLASS
            .align_frames_output_comscript
            .get_file_with_property_user_dir(
                Some(self),
                Some(&rootname),
                None,
                None,
                property_user_dir.as_deref(),
            )
        else {
            return false;
        };
        if file.exists() {
            match LogFile::get_instance_file(
                Some(&file),
                Some(self.get_emergency_monitor(Some(AxisID::Only))),
            ) {
                Ok(log_file) => match log_file.backup() {
                    Ok(_) => {}
                    Err(LogFileError::Lock(_)) => {}
                    Err(e) => eprintln!("{e:?}"),
                },
                Err(LogFileError::Lock(_)) => {}
                Err(e) => eprintln!("{e:?}"),
            }
        }
        if !file.exists() {
            ToolsProcessManager::touch(
                &utilities::java_io_file_get_absolute_path(&file.to_string_lossy()),
                Some(self),
            );
        }
        self.com_script_mgr().reset_align_frames_output();
        if !self.com_script_mgr().load_align_frames_output(&file, true) {
            return false;
        }

        let file_name = utilities::java_io_file_get_name(&file.to_string_lossy());
        let mut param: AlignFramesParam = self
            .com_script_mgr()
            .get_align_frames_output_param(Some(&file_name));
        match display.get_parameters_align_frames_param(&mut param) {
            Ok(true) => {}
            Ok(false) => return false,
            Err(except) => {
                eprintln!("{except}");
                // Java `new String[3]` with only the first element set; the null
                // elements show nothing.
                let error_message = [except.to_string()];
                ui_harness::INSTANCE.with(|harness| {
                    harness.open_message_dialog_base_manager_string_array_string_axis_id(
                        Some(self),
                        &error_message,
                        "AlignFrames Parameter Syntax Error",
                        Some(AxisID::Only),
                    )
                });
                return false;
            }
        }
        if *self.align_frames_tilt_angle_file.lock().unwrap() {
            param.set_tilt_angle_file(Some(&format!(
                "{}{}{}{}",
                display.get_rootname_output_files(),
                sort_tilt_frames_param::OUTPUT_TILT_ANGLE_FILE_MATCHING,
                extension::EXTENSION_DIVIDER,
                extension::CLASS.tlt
            )));
        }
        self.com_script_mgr().save_align_frames_output(&param);
        true
    }

    /// Java `getListOfInputFiles(AlignFramesDisplay, File, boolean) throws
    /// LogFileException, IOException, LockException`.
    pub fn get_list_of_input_files(
        &'static self,
        display: &dyn AlignFramesDisplay,
        subdir: &Path,
        in_list: bool,
    ) -> Result<String, LogFileError> {
        let list = if in_list {
            LogFile::get_instance_file(
                Some(&subdir.join(format!(
                    "{}{}{}{}",
                    display.get_rootname_output_files(),
                    sort_tilt_frames_param::OUTPUT_FILE_LIST_INLIST,
                    extension::EXTENSION_DIVIDER,
                    extension::CLASS.txt
                ))),
                Some(self.get_emergency_monitor(Some(AxisID::Only))),
            )?
        } else {
            LogFile::get_instance_file(
                Some(&subdir.join(format!(
                    "{}{}{}{}",
                    display.get_rootname_output_files(),
                    sort_tilt_frames_param::OUTPUT_FILE_LIST_TEMPLIST,
                    extension::EXTENSION_DIVIDER,
                    extension::CLASS.txt
                ))),
                Some(self.get_emergency_monitor(Some(AxisID::Only))),
            )?
        };
        list.create()?;

        let writer_id = list.open_writer()?;
        list.write(Some(&display.get_text_area_input_files()), &writer_id)?;
        list.close_id(Some(&writer_id));
        Ok(list.get_absolute_path())
    }

    /// Java `getOutputFileList(AlignFramesDisplay, File) throws
    /// LogFileException, IOException, LockException`.
    pub fn get_output_file_list(
        &'static self,
        display: &dyn AlignFramesDisplay,
        subdir: &Path,
    ) -> Result<String, LogFileError> {
        let in_list = LogFile::get_instance_file(
            Some(&subdir.join(format!(
                "{}{}{}{}",
                display.get_rootname_output_files(),
                sort_tilt_frames_param::OUTPUT_FILE_LIST_INLIST,
                extension::EXTENSION_DIVIDER,
                extension::CLASS.txt
            ))),
            Some(self.get_emergency_monitor(Some(AxisID::Only))),
        )?;
        in_list.create()?;
        Ok(in_list.get_absolute_path())
    }

    /// Java `setAlignFramesTiltAngleFileCreated(boolean)`.
    pub fn set_align_frames_tilt_angle_file_created(&self, input: bool) {
        *self.align_frames_tilt_angle_file.lock().unwrap() = input;
    }

    // ALIGN_FRAMES
    /// Java `openAlignFramesInputComFile(AlignFramesDisplay, File, UIComponent)`.
    pub fn open_align_frames_input_com_file(
        &'static self,
        display: &dyn AlignFramesDisplay,
        com_file: Option<&Path>,
        _component: Option<&dyn UiComponent>,
    ) -> bool {
        let Some(com_file) =
            com_file.filter(|com_file| com_file.exists() && std::fs::File::open(com_file).is_ok())
        else {
            ui_harness::INSTANCE.with(|harness| {
                harness.open_message_dialog_base_manager_string_string(
                    Some(self),
                    ".com file is empty/does not exist/not readable",
                    ".com File Error",
                )
            });
            return false;
        };
        if !self
            .com_script_mgr()
            .load_align_frames_input(com_file, false)
        {
            ui_harness::INSTANCE.with(|harness| {
                harness.open_message_dialog_base_manager_string_string(
                    Some(self),
                    &format!(
                        "{} is empty/does not exist/not readable",
                        utilities::java_io_file_get_absolute_path(&com_file.to_string_lossy())
                    ),
                    ".com File Error",
                )
            });
            return false;
        }

        // `getAlignFramesInputParam` never returns null, so the source's
        // "AlignFramesInputParam is empty" branch is unreachable.
        let align_frames_input_param = self.com_script_mgr().get_align_frames_input_param();

        display.set_parameters(&align_frames_input_param);
        true
    }

    // ALIGN_FRAMES
    /// Java `alignFrames(ProcessSeries, AlignFramesDisplay)`.
    pub fn align_frames(
        &'static self,
        process_series: Option<&ProcessSeriesHandle>,
        display: &dyn AlignFramesDisplay,
    ) {
        if !self.update_align_frames_output_param(display) {
            if let Some(process_series) = process_series {
                ProcessSeries::start_fail_process(process_series, AxisID::Only);
            }
            return;
        }

        let thread_name = match self.process_mgr().align_frames(
            process_series.map(|process_series| Arc::new(EdtRef::new(Rc::clone(process_series)))),
            &file_type::CLASS.align_frames_output_comscript,
            &file_type::CLASS.align_frames_log,
        ) {
            Ok(thread_name) => thread_name,
            Err(e) => {
                eprintln!("{e}");
                let message = [
                    format!("Can not execute {}", ProcessName::ALIGN_FRAMES),
                    e.to_string(),
                ];
                ui_harness::INSTANCE.with(|harness| {
                    harness.open_message_dialog_base_manager_string_array_string_axis_id(
                        Some(self),
                        &message,
                        "Unable to execute command",
                        None,
                    )
                });
                return;
            }
        };
        self.set_thread_name(Some(&thread_name), None);
    }

    /// Java `setRootname(String)`.
    pub fn set_rootname(&self, rootname: Option<&str>) {
        if rootname.is_some() {
            self.meta_data().set_root_name_string(rootname);
        }
    }

    // ALIGN_FRAMES
    /// Java `sorttiltframes(AlignFramesDisplay)`.
    pub fn sorttiltframes(&'static self, display: Rc<dyn AlignFramesDisplay>) {
        let rootname = display.get_rootname_output_files();
        let property_user_dir = self.property_user_dir();
        let align_frames_com_file = file_type::CLASS
            .align_frames_output_comscript
            .get_file_with_property_user_dir(
                Some(self),
                Some(&rootname),
                None,
                None,
                property_user_dir.as_deref(),
            );
        // Upstream bug fixed in translation (ToolsManager.java:497): Java
        // dereferences a null File from getFile; the param gets a null com
        // file name here.
        let align_frames_com_file_name = align_frames_com_file
            .map(|file| utilities::java_io_file_get_name(&file.to_string_lossy()));
        let mut param = SortTiltFramesParam::new(self, None, align_frames_com_file_name.as_deref());

        match display.get_parameters_sort_tilt_frames_param(&mut param) {
            Ok(true) => {}
            Ok(false) => return,
            Err(except) => {
                eprintln!("{except}");
                // Java `new String[3]` with only the first element set.
                let error_message = [except.to_string()];
                ui_harness::INSTANCE.with(|harness| {
                    harness.open_message_dialog_base_manager_string_array_string_axis_id(
                        Some(self),
                        &error_message,
                        "AlignFrames Parameter Syntax Error",
                        Some(AxisID::Only),
                    )
                });
                return;
            }
        }
        let process_series = ProcessSeries::new_with_process_display(
            self,
            AxisID::Only,
            None,
            Some(Rc::clone(&display) as Rc<dyn ProcessDisplay>),
            Some("sorttiltframes"),
        );
        process_series
            .borrow_mut()
            .set_next_process_task(Rc::new(Task::AlignFrames) as Rc<dyn TaskInterface>);
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.start_progress_bar_string_axis_id_process_name(
                Some("Sort tilt frames"),
                AxisID::Only,
                Some(&ProcessName::SORT_TILT_FRAMES),
            );
        }
        match self.process_mgr().sort_tilt_frames(
            &mut param,
            Some(Arc::new(EdtRef::new(Rc::clone(&process_series)))),
        ) {
            Ok(thread_name) => self.set_thread_name(Some(&thread_name), None),
            Err(e) => {
                eprintln!("{e}");
                if let Some(main_panel) = self.main_panel.get() {
                    main_panel.stop_progress_bar_axis_id_process_end_state(
                        AxisID::Only,
                        Some(ProcessEndState::Failed),
                    );
                }
                let message = ["Can not execute sorttiltframes".to_owned(), e.to_string()];
                ui_harness::INSTANCE.with(|harness| {
                    harness.open_message_dialog_base_manager_string_array_string_axis_id(
                        Some(self),
                        &message,
                        "Unable to execute script",
                        None,
                    )
                });
            }
        }
    }

    // ALIGN_FRAMES
    /// Java `plotAllResults(AlignFramesDisplay)`.
    pub fn plot_all_results(&'static self, display: &dyn AlignFramesDisplay) {
        // Java's unused local `logFile`.
        let rootname = display.get_rootname_output_files();
        let property_user_dir = self.property_user_dir();
        let _log_file = file_type::CLASS
            .align_frames_log
            .get_file_with_property_user_dir(
                Some(self),
                Some(&rootname),
                None,
                None,
                property_user_dir.as_deref(),
            );

        let process_series = ProcessSeries::new(self, AxisID::Only, None, Some("plotAllResults"));
        process_series
            .borrow_mut()
            .set_next_process_task(Rc::new(tomodataplots_param::Task::AlignFramesMeanResiduals)
                as Rc<dyn TaskInterface>);
        process_series.borrow_mut().add_process(Rc::new(
            tomodataplots_param::Task::AlignFramesMaxofmaxResiduals,
        ) as Rc<dyn TaskInterface>);
        self.tomodataplots(
            Some(&tomodataplots_param::Task::AlignFramesShifts),
            None,
            Some(&process_series),
            None,
        );
    }

    // ALIGN_FRAMES
    /// Java `openOutputTiltSeries(Run3dmodMenuOptions, AlignFramesDisplay)`.
    pub fn open_output_tilt_series(
        &'static self,
        menu_options: Option<Run3dmodMenuOptions>,
        display: &dyn AlignFramesDisplay,
    ) {
        let key = imod_manager::OPEN_OUTPUT_TILT_SERIES_KEY;
        let file = Path::new(&self.property_user_dir().unwrap_or_default())
            .join(display.get_output_image_file_name());
        match self
            .get_imod_manager()
            .open_string_axis_id_file_run3dmod_menu_options(
                key,
                Some(AxisID::Only),
                Some(&file),
                menu_options,
            ) {
            Ok(()) => {}
            Err(ImodManagerException::SystemProcess(except)) => {
                eprintln!("{except}");
                self.open_message(
                    &format!("{except}\nCan't open 3dmod on the {key}"),
                    "Cannot Open 3dmod",
                    AxisID::Only,
                );
            }
            Err(ImodManagerException::AxisType(except)) => {
                eprintln!("{except}");
                self.open_message(&except.to_string(), "AxisType problem", AxisID::Only);
            }
            Err(ImodManagerException::Io(e)) => {
                eprintln!("{e}");
                self.open_message(&e.to_string(), "IO Exception", AxisID::Only);
            }
            // An uncaught RuntimeException: Swing's handler prints it.
            Err(ImodManagerException::Runtime(message)) => eprintln!("{message}"),
        }
    }

    // ALIGN_FRAMES
    /// Java `openTomogram(AlignFramesDisplay)`.
    pub fn open_tomogram(&'static self, display: &dyn AlignFramesDisplay) {
        let local_arguments = display.setup_local_arguments();
        // Move outputImageFile to LocalArguments->getDir() location
        let property_user_dir = self
            .property_user_dir()
            .unwrap_or_else(|| "null".to_owned());
        let dir = local_arguments
            .arguments
            .get_dir()
            .map(|dir| dir.to_string_lossy().into_owned())
            .unwrap_or_else(|| "null".to_owned());
        // `Files.move(source, target)` without REPLACE_EXISTING: a target that
        // exists is a FileAlreadyExistsException (its message is the target).
        let files_move = |source: String, target: String| -> Result<(), MoveError> {
            if Path::new(&target).exists() {
                return Err(MoveError::FileAlreadyExists(target));
            }
            std::fs::rename(&source, &target).map_err(|e| MoveError::Io(format!("{source}: {e}")))
        };
        let result = (|| -> Result<(), MoveError> {
            files_move(
                format!(
                    "{property_user_dir}/{}",
                    display.get_output_image_file_name()
                ),
                format!("{dir}/{}", display.get_output_image_file_name()),
            )?;
            if display.is_metadata_file_selected() {
                files_move(
                    format!("{property_user_dir}/{}", display.get_new_mdoc_file_name()),
                    format!("{dir}/{}", display.get_new_mdoc_file_name()),
                )?;
            }
            etomo_director::INSTANCE.open_tomogram_and_do_automation(
                true,
                Some(AxisID::Only),
                Some(&local_arguments),
            );
            Ok(())
        })();
        match result {
            Ok(()) => {}
            Err(MoveError::FileAlreadyExists(except)) => {
                eprintln!("java.nio.file.FileAlreadyExistsException: {except}");
                self.open_message(
                    &format!(
                        "{except}\nFile with the same name already exists in the selected \
                         directory.\nPlease select another directory."
                    ),
                    "File already exists",
                    AxisID::Only,
                );
            }
            Err(MoveError::Io(except)) => {
                eprintln!("{except}");
                self.open_message(
                    &format!("{except}\nUnable to move file to the selected directory."),
                    "Failed to move file",
                    AxisID::Only,
                );
            }
        }
    }

    /// Java `imodViewModel`.  Open 3dmodv with a model.
    pub fn imod_view_model(&'static self, axis_id: AxisID, model_file_type: &FileType) {
        let file_name = model_file_type.get_file_name(Some(self), Some(axis_id));
        // Fixed in translation: a file type without a 3dmod key makes Java's
        // ImodManager throw NullPointerException; nothing is opened.
        let Some(key) = model_file_type.get_imod_manager_key() else {
            return;
        };
        match self.get_imod_manager().open_string_axis_id_string(
            key,
            Some(axis_id),
            file_name.as_deref(),
        ) {
            Ok(()) => {}
            Err(ImodManagerException::AxisType(except)) => {
                eprintln!("{except}");
                self.open_message(&except.to_string(), "AxisType problem", axis_id);
            }
            Err(ImodManagerException::SystemProcess(except)) => {
                eprintln!("{except}");
                self.open_message(
                    &except.to_string(),
                    &format!(
                        "Can't open 3dmod on {}",
                        file_name.as_deref().unwrap_or("null")
                    ),
                    axis_id,
                );
            }
            Err(ImodManagerException::Io(e)) => {
                eprintln!("{e}");
                self.open_message(&e.to_string(), "IO Exception", axis_id);
            }
            // An uncaught RuntimeException: Swing's handler prints it.
            Err(ImodManagerException::Runtime(message)) => eprintln!("{message}"),
        }
    }

    /// Java `getState()`: null.
    // TODO(unit): needs etomo/type/ParallelState.java - the declared return type;
    // the source always returns null.
    pub fn get_state(&self) -> Option<Infallible> {
        None
    }

    /// Java private `createState()`: empty.
    fn create_state(&self) {}

    /// Java private `openProcessingPanel()`.  MUST run reconnect for all axis.
    fn open_processing_panel(&'static self) {
        // Upstream bug fixed in translation (ToolsManager.java:800): Java
        // dereferences mainPanel, which is null in headless mode (initialize
        // only calls this when not headless); nothing is shown then.
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.show_processing_panel(AxisType::SingleAxis);
        }
        self.set_panel();
        self.reconnect(
            Some(
                self.get_axis_process_data()
                    .get_saved_process_data(AxisID::Only),
            ),
            Some(AxisID::Only),
            false,
            None,
        );
    }
}

impl BaseManager for ToolsManager {
    fn base(&self) -> &BaseManagerBase {
        &self.base
    }

    fn this(&'static self) -> &'static dyn BaseManager {
        self
    }

    /// Java `closeFrame()`.
    fn close_frame(&self) -> bool {
        true
    }

    /// Java `getInterfaceType()`.
    fn get_interface_type(&self) -> Option<InterfaceType> {
        Some(InterfaceType::Tools)
    }

    fn get_tool_type(&self) -> Option<ToolType> {
        Some(self.tool_type)
    }

    /// Java `getLogInterface()`: the Tools dialog.  It is an event dispatch
    /// thread object; off that thread it cannot be reached and reads as null
    /// (as `BaseManager.getLogWindow` does).
    fn get_log_interface(&self) -> Option<Rc<dyn LogInterface>> {
        if !event_queue::is_dispatch_thread() {
            return None;
        }
        self.tools_dialog
            .get()
            .map(|tools_dialog| tools_dialog as Rc<dyn LogInterface>)
    }

    /// Java package-private `createComScriptManager()`.
    fn create_com_script_manager(&self) {
        let this = self.this_static();
        let com_script_mgr: &'static ToolsComScriptManager =
            Box::leak(Box::new(ToolsComScriptManager::new(this)));
        let _ = self.com_script_mgr.set(com_script_mgr);
    }

    /// Java package-private `createMainPanel()`.
    fn create_main_panel(&self) {
        if !etomo_director::ARGUMENTS.lock().unwrap().is_headless() {
            let this = self.this_static();
            self.main_panel.set(Some(MainToolsPanel::new(this)));
        }
    }

    /// Java package-private `createLogWindow()`: null.
    fn create_log_window(&'static self) -> Option<Rc<LogWindow>> {
        None
    }

    /// Java `getBaseMetaData()`.
    fn get_base_meta_data(&self) -> Option<&dyn BaseMetaData> {
        self.meta_data
            .get()
            .map(|meta_data| meta_data as &dyn BaseMetaData)
    }

    /// Java `getMainPanel()`.
    fn get_main_panel(&self) -> Option<Rc<dyn MainPanelVirtual>> {
        self.main_panel
            .get()
            .map(|main_panel| main_panel as Rc<dyn MainPanelVirtual>)
    }

    /// Java package-private `getStorables(int)`: null.
    fn get_storables_with_offset(
        &self,
        _offset: i32,
    ) -> Option<Vec<Option<&'static dyn Storable>>> {
        None
    }

    /// Java `isInManagerFrame()`.
    fn is_in_manager_frame(&self) -> bool {
        // Warning: This does not work with openToolInSeparateFrame(). Should return true if
        // openToolInSeparateFrame was called.
        false
    }

    /// Java `getProcessManager()`.
    fn get_process_manager(
        &self,
    ) -> Option<&'static crate::imod::etomo::process::base_process_manager::BaseProcessManager>
    {
        self.process_mgr.get().map(|process_mgr| &process_mgr.base)
    }

    /// Java `save() throws LogFileException, IOException, LockException`.
    fn save(&'static self) -> Result<bool, LogFileError> {
        self.save_super()?;
        // Upstream bug fixed in translation (ToolsManager.java:770): Java
        // dereferences mainPanel, which is null in headless mode; it is
        // skipped here.
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.done();
        }
        Ok(true)
    }

    /// Java `exitProgram(AxisID)`.
    fn exit_program(&'static self, axis_id: Option<AxisID>) -> bool {
        // try { ... } catch (Throwable e) { e.printStackTrace(); return true; }
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            if self.exit_program_super(axis_id) {
                self.end_threads();
                self.save_param_file()?;
                return Ok(true);
            }
            Ok::<bool, LogFileError>(false)
        }));
        match result {
            Ok(Ok(exit)) => exit,
            Ok(Err(e)) => {
                eprintln!("{e:?}");
                true
            }
            Err(_) => true,
        }
    }

    /// Java `getName()`.
    fn get_name(&self) -> Option<String> {
        self.meta_data
            .get()
            .and_then(|meta_data| meta_data.get_name())
    }

    /// Java package-private `startNextProcess(UIComponent, AxisID,
    /// ProcessSeries.Process, ProcessResultDisplay, ProcessSeries, DialogType,
    /// ProcessDisplay)`.  Returns true if the process is recognized.
    fn start_next_process(
        &'static self,
        ui_component: Option<Rc<dyn UiComponent>>,
        axis_id: AxisID,
        process: &Process,
        process_result_display: Option<ProcessResultDisplayRef>,
        process_series: &ProcessSeriesHandle,
        dialog_type: Option<DialogType>,
        display: Option<Rc<dyn ProcessDisplay>>,
    ) -> bool {
        if self.start_next_process_super(
            ui_component,
            axis_id,
            process,
            process_result_display,
            process_series,
            dialog_type,
            display.clone(),
        ) {
            return true;
        }
        if process.equals_task(&Task::AlignFrames) {
            // Upstream bug fixed in translation (ToolsManager.java:888): Java casts
            // the display to AlignFramesDisplay unchecked (ClassCastException, or a
            // NullPointerException in alignFrames for a null one); a series without
            // an align-frames display fails here.
            match display
                .as_deref()
                .and_then(|display| display.as_align_frames_display())
            {
                Some(align_frames_display) => {
                    self.align_frames(Some(process_series), align_frames_display)
                }
                None => ProcessSeries::start_fail_process(process_series, AxisID::Only),
            }
            return true;
        }
        false
    }
}

/// `Files.move`'s two failures, as `openTomogram` catches them.
enum MoveError {
    /// `java.nio.file.FileAlreadyExistsException`; the message is the target.
    FileAlreadyExists(String),
    /// `java.io.IOException`.
    Io(String),
}

/// Java private static final `ConflictFileFilter extends
/// javax.swing.filechooser.FileFilter implements java.io.FileFilter`.  Class
/// identifies dataset files that conflict with the member variable
/// compareFileName.  Used to avoid file name collisions between tools
/// projects and exclusive datasets.  Ignores parallel processing datasets
/// because these are file oriented datasets like tools projects.
pub struct ConflictFileFilter {
    /// Java private final `compareFileName`.
    compare_file_name: String,
}

impl ConflictFileFilter {
    /// Java private `ConflictFileFilter(String)`.
    fn new(compare_file_name: &str) -> Self {
        Self {
            compare_file_name: compare_file_name.to_owned(),
        }
    }

    /// Java `accept(File)`.  Returns true if file is in conflict with
    /// compareFileName.
    pub fn accept(&self, file: &Path) -> bool {
        // If this file has one of the five exclusive dataset extensions and the
        // left side of the file name is equal to compareFileName, then
        // compareFileName is in conflict with the dataset in this directory and
        // may cause file name collisions.
        if file.is_file() {
            let file_name = utilities::java_io_file_get_name(&file.to_string_lossy());
            let ends_with = |data_file_type: DataFileType| {
                // `fileName.endsWith(null)` throws NullPointerException; the four
                // exclusive types all have an extension.
                data_file_type
                    .extension()
                    .is_some_and(|extension| file_name.ends_with(extension))
            };
            if ends_with(DataFileType::Recon)
                || ends_with(DataFileType::Join)
                || ends_with(DataFileType::Peet)
                || ends_with(DataFileType::SerialSections)
            {
                // The matched extension contains '.', so lastIndexOf finds one.
                let dot = file_name.rfind('.').unwrap_or(file_name.len());
                return file_name[..dot] == self.compare_file_name;
            }
        }
        false
    }

    /// Java `getDescription()`.
    pub fn get_description(&self) -> String {
        format!(
            "Dataset file that conflicts with {}",
            self.compare_file_name
        )
    }
}

/// Java `public static final class Task implements TaskInterface`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Task {
    /// Java private static final `ALIGN_FRAMES = new Task("align frames")`.
    AlignFrames,
}

impl TaskInterface for Task {
    /// Java `getDescr()`.
    fn get_descr(&self) -> Option<String> {
        match self {
            Task::AlignFrames => Some("align frames".to_string()),
        }
    }

    /// Java `okToDrop()`.
    fn ok_to_drop(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conflict_filter_matches_only_exclusive_dataset_extensions() {
        let dir =
            std::env::temp_dir().join(format!("imod-rs-tools-manager-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let conflict = dir.join("dataset.edf");
        let parallel = dir.join("dataset.epp");
        std::fs::File::create(&conflict).unwrap();
        std::fs::File::create(&parallel).unwrap();
        let filter = ConflictFileFilter::new("dataset");
        assert!(filter.accept(&conflict));
        assert!(!filter.accept(&parallel));
        assert_eq!(
            filter.get_description(),
            "Dataset file that conflicts with dataset"
        );
        std::fs::remove_file(conflict).unwrap();
        std::fs::remove_file(parallel).unwrap();
        std::fs::remove_dir(dir).unwrap();
    }

    #[test]
    fn meta_data_name_follows_tool_type_until_root_name_is_set() {
        let manager = event_queue::invoke_and_wait(|| ToolsManager::new(ToolType::GpuTiltTest));
        assert_eq!(manager.get_name().as_deref(), Some("GPU Test"));
        manager.set_rootname(Some("gputest"));
        assert_eq!(manager.get_name().as_deref(), Some("gputest"));
        assert_eq!(manager.get_interface_type(), Some(InterfaceType::Tools));
    }
}
