//! `IMOD/Etomo/src/etomo/DirectiveEditorManager.java`.
//!
//! The manager of the directive file editor (File > Templates > Save ... Template, File
//! > Export Batch Directive File): a `ManagerFrame` of its own holding a
//! `MainDirectiveEditorPanel` with the `DirectiveEditorDialog`, which saves the edited
//! directives through `DirectiveWriter`.
//!
//! The manager itself is shared across threads (`&'static`, like every manager); its
//! panel and dialog are event dispatch thread objects, held in `EdtCell`s and reached
//! only on that thread.

use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::{Mutex, OnceLock};

use crate::imod::etomo::base_manager::{BaseManager, BaseManagerBase};
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::logic::directive_editor_builder::DirectiveEditorBuilder;
use crate::imod::etomo::process::base_process_manager::BaseProcessManager;
use crate::imod::etomo::storage::autodoc::autodoc_factory;
use crate::imod::etomo::storage::directive_map::DirectiveMap;
use crate::imod::etomo::storage::directive_writer::DirectiveWriter;
use crate::imod::etomo::storage::storable::Storable;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::base_meta_data::BaseMetaData;
use crate::imod::etomo::r#type::data_file_type::DataFileType;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::directive_editor_meta_data::DirectiveEditorMetaData;
use crate::imod::etomo::r#type::directive_file_type::DirectiveFileType;
use crate::imod::etomo::r#type::interface_type::InterfaceType;
use crate::imod::etomo::r#type::parallel_state::ParallelState;
use crate::imod::etomo::ui::swing::directive_editor_dialog::DirectiveEditorDialog;
use crate::imod::etomo::ui::swing::log_interface::LogInterface;
use crate::imod::etomo::ui::swing::log_window::LogWindow;
use crate::imod::etomo::ui::swing::main_directive_editor_panel::MainDirectiveEditorPanel;
use crate::imod::etomo::ui::swing::main_panel::MainPanelVirtual;
use crate::imod::etomo::ui::swing::ui_harness;
use crate::imod::etomo::util::event_queue::EdtCell;
use crate::imod::etomo::util::utilities;

/// Java `public static final String rcsid`.
pub const RCSID: &str = "$Id:$";

/// Java private static final `AXIS_ID`.
const AXIS_ID: AxisID = AxisID::Only;
/// Java private static final `DIALOG_TYPE`.
const DIALOG_TYPE: DialogType = DialogType::DirectiveEditor;
/// Java private static final `STATUS_BAR_SIZE`.
const STATUS_BAR_SIZE: usize = 65;

/// Java `public final class DirectiveEditorManager extends BaseManager`.
pub struct DirectiveEditorManager {
    /// Java superclass `BaseManager` state.
    base: BaseManagerBase,
    /// Java private final `sourceManager`.
    source_manager: Option<&'static dyn BaseManager>,
    /// Java private final `metaData` (assigned by the constructor, which passes `this`).
    meta_data: OnceLock<DirectiveEditorMetaData>,
    /// Java private final `type`.
    r#type: Option<DirectiveFileType>,
    /// Java private final `timestamp`.
    timestamp: Option<String>,
    /// Java private `mainPanel`.
    main_panel: EdtCell<Rc<MainDirectiveEditorPanel>>,
    /// Java private `dialog`, initialised to null.
    dialog: EdtCell<Rc<DirectiveEditorDialog>>,
    /// Java private `dialogErrmsg`, initialised to null.
    dialog_errmsg: Mutex<Option<String>>,
    /// Java private `saveFile`, initialised to null.
    save_file: Mutex<Option<PathBuf>>,
}

/// Owns every `DirectiveEditorManager` this module builds.  Java's owner is the
/// collector, by way of `UIHarness.managerFrameTable`, which keeps each manager for
/// the run; the translation hands out `&'static Self`, so without a root here the
/// allocation is unreachable the moment the constructor returns.
static INSTANCES: Mutex<Vec<&'static DirectiveEditorManager>> = Mutex::new(Vec::new());

impl DirectiveEditorManager {
    /// Java `DirectiveEditorManager(DirectiveFileType, BaseManager, String,
    /// StringBuffer)`.
    pub fn new(
        r#type: Option<DirectiveFileType>,
        source_manager: Option<&'static dyn BaseManager>,
        timestamp: Option<&str>,
        errmsg: Option<&str>,
    ) -> &'static Self {
        let instance: &'static DirectiveEditorManager = Box::leak(Box::new(Self {
            base: BaseManagerBase::initial(),
            source_manager,
            meta_data: OnceLock::new(),
            r#type,
            timestamp: timestamp.map(str::to_string),
            main_panel: EdtCell::new(),
            dialog: EdtCell::new(),
            dialog_errmsg: Mutex::new(None),
            save_file: Mutex::new(None),
        }));
        INSTANCES.lock().unwrap().push(instance);
        // `super()`, which calls createMainPanel.
        instance.base_manager();
        let _ = instance.meta_data.set(DirectiveEditorMetaData::new(
            Some(instance),
            r#type,
            instance.get_log_properties(),
            true,
        ));
        *instance.dialog_errmsg.lock().unwrap() = errmsg.map(str::to_string);
        instance.create_state();
        instance.initialize_ui_parameters(None, Some(AXIS_ID), false);
        // Frame hasn't been created yet so stop here.
        instance
    }

    /// The constructed manager at its final address (Java `this` inside the overrides
    /// that take `&self`).
    fn this_static(&self) -> &'static DirectiveEditorManager {
        INSTANCES
            .lock()
            .unwrap()
            .iter()
            .copied()
            .find(|manager| std::ptr::eq(*manager, self))
            .expect("constructed DirectiveEditorManager")
    }

    /// Java field read `metaData`.
    fn meta_data(&self) -> &DirectiveEditorMetaData {
        self.meta_data
            .get()
            .expect("metaData is assigned by the constructor")
    }

    /// Java field read `dialog`.  The frame's Save/Close and the dialog's buttons only
    /// exist once `initialize` has opened the dialog.
    fn dialog(&self) -> Rc<DirectiveEditorDialog> {
        self.dialog
            .get()
            .expect("DirectiveEditorManager: dialog opened by initialize")
    }

    /// Java `initialize()`.  Open panel and add dialog.
    pub fn initialize(&'static self) {
        if !etomo_director::ARGUMENTS.lock().unwrap().is_headless() {
            self.open_processing_panel();
            let location: Option<String> = None;
            let size = STATUS_BAR_SIZE;
            if let Some(main_panel) = self.main_panel.get() {
                main_panel.set_status_bar_text(location.as_deref(), size);
            }
            self.open_directive_editor_dialog();
            ui_harness::with(|harness| harness.to_front(Some(self)));
        }
    }

    /// Java `setName(File)`.  Sets the dataset name.
    pub fn set_name(&'static self, input_file: &Path) {
        // TODO not being called
        let parent = input_file
            .parent()
            .map(|parent| parent.to_string_lossy().into_owned());
        *self.base().property_user_dir.lock().unwrap() = parent.clone();
        self.meta_data().set_root_name(Some(input_file));
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.set_status_bar_text(parent.as_deref(), STATUS_BAR_SIZE);
        }
        let mut label = "Directive File";
        if let Some(r#type) = self.r#type {
            label = r#type.get_label();
        }
        let title = format!(
            "{} - {}",
            label,
            self.get_name().unwrap_or_else(|| "null".to_string())
        );
        ui_harness::with(|harness| harness.set_title(Some(self), &title));
    }

    /// Java `openDirectiveEditorDialog()`.
    pub fn open_directive_editor_dialog(&'static self) {
        if !self.dialog.is_some() {
            let builder = Rc::new(DirectiveEditorBuilder::new(self, self.r#type));
            // Upstream bug fixed in translation: Java dereferences sourceManager
            // unchecked (DirectiveEditorManager.java:110); a manager without a source
            // takes the source's axis type as not set and its status as null.
            let source_axis_type = self
                .source_manager
                .and_then(|source_manager| source_manager.get_base_meta_data())
                .map(|meta_data| meta_data.base().get_axis_type())
                .unwrap_or(AxisType::NotSet);
            let errmsg = self.dialog_errmsg.lock().unwrap().take();
            let errmsg = builder.build(source_axis_type, errmsg);
            *self.dialog_errmsg.lock().unwrap() = Some(errmsg.clone());
            let source_status = self
                .source_manager
                .and_then(|source_manager| source_manager.get_status());
            self.dialog.set(Some(DirectiveEditorDialog::get_instance(
                self,
                self.r#type,
                builder,
                source_axis_type,
                source_status.as_deref(),
                self.timestamp.as_deref(),
                Some(&errmsg),
            )));
        }
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.show_process(&self.dialog().get_container(), AXIS_ID);
        }
        let action_message =
            utilities::prepare_dialog_action_message(Some(DIALOG_TYPE), AxisID::Only, None);
        if let Some(action_message) = action_message {
            eprintln!("{action_message}");
        }
    }

    /// Java `getState()`: null.
    pub fn get_state(&self) -> Option<&ParallelState> {
        None
    }

    /// Java private `createState()`: empty.
    fn create_state(&self) {}

    /// Java private `writeFile()`.
    fn write_file(&'static self) -> bool {
        let save_file = self.save_file.lock().unwrap().clone();
        let mut writer = DirectiveWriter::new(Some(self), AxisID::Only, save_file.clone());
        if !writer.open() {
            return false;
        }
        let dialog = self.dialog();
        // Java also assigns `dialog.getIncludeDirectiveList()` to an unused local.
        let r#type = self.r#type.expect("DirectiveWriter.write: type");
        writer.write(
            r#type,
            Some(&dialog.get_comments()),
            Some(&dialog.get_include_directive_list()),
            Some(&dialog.get_dropped_directives()),
        );
        writer.close();
        // `new Date(saveFile.lastModified())`: 0 when the file cannot be read.
        let last_modified = save_file
            .as_ref()
            .and_then(|save_file| std::fs::metadata(save_file).ok())
            .and_then(|metadata| metadata.modified().ok())
            .and_then(|modified| modified.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|duration| duration.as_millis() as i64)
            .unwrap_or(0);
        dialog.set_file_timestamp(last_modified);
        true
    }

    /// Java private `openProcessingPanel()`.  MUST run reconnect for all axis.
    fn open_processing_panel(&'static self) {
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

    /// `uiHarness.openMessageDialog(this, message, title)`.
    fn open_message_dialog(&'static self, message: &str, title: &str) {
        ui_harness::with(|harness| {
            harness.open_message_dialog_base_manager_string_string(Some(self), message, title)
        });
    }

    /// `uiHarness.openYesNoDialog(this, message, AxisID.ONLY)`.
    fn open_yes_no_dialog(&'static self, message: &str) -> bool {
        ui_harness::with(|harness| {
            harness.open_yes_no_dialog_base_manager_string_axis_id(
                Some(self),
                message,
                Some(AxisID::Only),
            )
        })
    }
}

/// Rust-only plumbing for the Slint bridge (`slint_bridge`): the container of the
/// `DirectiveEditorDialog` of the directive editor `manager`, when `manager` is one
/// and its dialog is open.  Java's frame shows that container itself.
pub fn dialog_container_of(
    manager: &'static dyn BaseManager,
) -> Option<Rc<crate::imod::etomo::jdk::JComponent>> {
    let instance = INSTANCES.lock().unwrap().iter().copied().find(|instance| {
        std::ptr::addr_eq(
            *instance as *const DirectiveEditorManager,
            manager as *const dyn BaseManager,
        )
    })?;
    instance.dialog.get().map(|dialog| dialog.get_container())
}

impl BaseManager for DirectiveEditorManager {
    fn base(&self) -> &BaseManagerBase {
        &self.base
    }

    fn this(&'static self) -> &'static dyn BaseManager {
        self
    }

    /// Java `updateDirectiveMap(DirectiveMap, StringBuffer)`: the source manager's.
    /// Reached through the dispatching overload (see
    /// `BaseManager::update_directive_map_directive_map`).
    fn update_directive_map_directive_map(
        &'static self,
        directive_map: &DirectiveMap,
        errmsg: &mut String,
    ) {
        // Upstream bug fixed in translation: a null sourceManager is a
        // NullPointerException in Java (DirectiveEditorManager.java:102).
        if let Some(source_manager) = self.source_manager {
            source_manager.update_directive_map_directive_map(directive_map, errmsg);
        }
    }

    /// Java `getInterfaceType()`.
    fn get_interface_type(&self) -> Option<InterfaceType> {
        Some(InterfaceType::Tools)
    }

    /// Java package-private `createLogWindow()`: null.
    fn create_log_window(&'static self) -> Option<Rc<LogWindow>> {
        None
    }

    /// Java `getLogInterface()`: null.
    fn get_log_interface(&self) -> Option<Rc<dyn LogInterface>> {
        None
    }

    /// Java package-private `createMainPanel()`.
    fn create_main_panel(&self) {
        if !etomo_director::ARGUMENTS.lock().unwrap().is_headless() {
            let this = self.this_static();
            self.main_panel
                .set(Some(MainDirectiveEditorPanel::new(this)));
        }
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
        true
    }

    /// Java `getProcessManager()`: null.
    fn get_process_manager(&self) -> Option<&'static BaseProcessManager> {
        None
    }

    /// Java `closeFrame()`.
    fn close_frame(&self) -> bool {
        let this = self.this_static();
        if self.dialog().is_different_from_checkpoint(true) {
            let response = ui_harness::with(|harness| {
                harness.open_yes_no_cancel_dialog(
                    Some(this),
                    &format!(
                        "{} has been modified.  Do you want to save you changes?",
                        self.meta_data()
                            .get_name()
                            .unwrap_or_else(|| "null".to_string())
                    ),
                    Some(AXIS_ID),
                )
            });
            let Some(response) = response else {
                return false;
            };
            if response.is() && !self.save_to_file() {
                return false;
            }
        }
        true
    }

    /// Java `saveToFile()`.
    fn save_to_file(&self) -> bool {
        let this = self.this_static();
        if self.save_file.lock().unwrap().is_none() {
            return self.save_as_to_file();
        }
        if this.write_file() {
            self.dialog().checkpoint();
            return true;
        }
        false
    }

    /// Java `saveAsToFile()`.
    fn save_as_to_file(&self) -> bool {
        let this = self.this_static();
        let Some(abs_path) = self.dialog().get_save_file_abs_path() else {
            return false;
        };
        let save_file = if autodoc_factory::ends_with_autodoc_extension(Some(&abs_path)) {
            PathBuf::from(&abs_path)
        } else {
            PathBuf::from(format!(
                "{}{}",
                abs_path,
                autodoc_factory::extension::DEFAULT
            ))
        };
        *self.save_file.lock().unwrap() = Some(save_file.clone());
        let save_file_abs_path =
            utilities::java_io_file_get_absolute_path(&save_file.to_string_lossy());
        if save_file.exists() {
            if !this.open_yes_no_dialog(&format!(
                "The file {save_file_abs_path} already exists.  Overwrite it?"
            )) {
                return false;
            }
            if !save_file.is_file() {
                this.open_message_dialog(
                    &format!("Cannot to write to {save_file_abs_path} because it is not a file."),
                    "Unable to Write to File",
                );
                return false;
            }
            if !utilities::java_io_file_can_write(&save_file.to_string_lossy()) {
                this.open_message_dialog(
                    &format!("Unable to write to {save_file_abs_path}:  permission denied."),
                    "Unable to Write to File",
                );
                return false;
            }
        } else {
            let parent = save_file
                .parent()
                .map(Path::to_path_buf)
                .unwrap_or_default();
            if !parent.exists() {
                let parent_abs_path =
                    utilities::java_io_file_get_absolute_path(&parent.to_string_lossy());
                if this.open_yes_no_dialog(&format!(
                    "Directory, {parent_abs_path}, does not exist.  Create this directory?"
                )) {
                    if std::fs::create_dir_all(&parent).is_err() {
                        this.open_message_dialog(
                            &format!("Unable to create {parent_abs_path}."),
                            "Unable to Create Directory",
                        );
                    }
                } else {
                    return false;
                }
            }
        }
        if this.write_file() {
            self.dialog().checkpoint();
            this.set_name(&save_file);
            return true;
        }
        false
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
            Ok::<bool, crate::imod::etomo::storage::log_file::LogFileError>(false)
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
}

/// Java private static final `ConflictFileFilter extends
/// javax.swing.filechooser.FileFilter implements java.io.FileFilter`.  Identifies
/// dataset files that conflict with `compareFileName`.  Never constructed in the
/// source.
pub struct ConflictFileFilter {
    /// Java private final `compareFileName`.
    compare_file_name: String,
}

impl ConflictFileFilter {
    /// Java private `ConflictFileFilter(String)`.
    pub fn new(compare_file_name: &str) -> Self {
        Self {
            compare_file_name: compare_file_name.to_string(),
        }
    }
}

impl crate::imod::etomo::jdk::FileFilter for ConflictFileFilter {
    /// Java `accept(File)`.  True if file is in conflict with compareFileName.
    fn accept(&self, file: &Path) -> bool {
        // If this file has one of the five exclusive dataset extensions and the left
        // side of the file name is equal to compareFileName, then compareFileName is in
        // conflict with the dataset in this directory and may cause file name
        // collisions.
        if file.is_file() {
            let file_name = utilities::java_io_file_get_name(&file.to_string_lossy());
            let ends_with = |data_file_type: DataFileType| {
                data_file_type
                    .extension()
                    .is_some_and(|extension| file_name.ends_with(extension))
            };
            if ends_with(DataFileType::Recon)
                || ends_with(DataFileType::Join)
                || ends_with(DataFileType::Peet)
                || ends_with(DataFileType::SerialSections)
            {
                let index = file_name.rfind('.').unwrap_or(file_name.len());
                return file_name[..index] == self.compare_file_name;
            }
        }
        false
    }

    /// Java `getDescription()`.
    fn get_description(&self) -> Option<String> {
        Some(format!(
            "Dataset file that conflicts with {}",
            self.compare_file_name
        ))
    }
}
