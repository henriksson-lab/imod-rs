//! `IMOD/Etomo/src/etomo/DirectiveEditorManager.java`.
//!
//! The directive editor has a genuine GUI/process boundary: `DirectiveEditorDialog`,
//! `MainDirectiveEditorPanel`, `DirectiveEditorBuilder`, and `DirectiveWriter` are
//! separate Java units which have not yet been translated.  Their nullable Java fields
//! are therefore represented by `Option<Infallible>` rather than replacement panels or
//! invented write behavior.  The manager, metadata, file naming, conflict filter, and
//! all source methods are nevertheless kept source-shaped here.
#![allow(dead_code)]

use crate::imod::etomo::base_manager::{BaseManager, BaseManagerBase};
use crate::imod::etomo::storage::storable::Storable;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::base_meta_data::BaseMetaData;
use crate::imod::etomo::r#type::data_file_type::DataFileType;
use crate::imod::etomo::r#type::directive_editor_meta_data::DirectiveEditorMetaData;
use crate::imod::etomo::r#type::directive_file_type::DirectiveFileType;
use crate::imod::etomo::r#type::interface_type::InterfaceType;
use std::convert::Infallible;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// Java private static final `AXIS_ID`.
const AXIS_ID: AxisID = AxisID::Only;
/// Java private static final `STATUS_BAR_SIZE`.
const STATUS_BAR_SIZE: i32 = 65;

/// Java `DirectiveEditorManager`.
pub struct DirectiveEditorManager {
    /// Java superclass `BaseManager` state.
    base: BaseManagerBase,
    /// Java final `sourceManager`.
    source_manager: Option<&'static dyn BaseManager>,
    /// Java final `metaData`.
    meta_data: DirectiveEditorMetaData,
    /// Java final `type`.
    file_type: Option<DirectiveFileType>,
    /// Java final `timestamp`.
    timestamp: String,
    /// Java `mainPanel`.
    // TODO(unit): etomo/ui/swing/MainDirectiveEditorPanel.java.
    main_panel: Option<Infallible>,
    /// Java `dialog`, initialised to null.
    // TODO(unit): etomo/ui/swing/DirectiveEditorDialog.java.
    dialog: Option<Infallible>,
    /// Java `dialogErrmsg`, initialised from the constructor argument.
    dialog_errmsg: Mutex<Option<String>>,
    /// Java `saveFile`, initialised to null.
    save_file: Mutex<Option<PathBuf>>,
}

// `DirectiveEditorManager` creates its `DirectiveEditorMetaData` with a null
// `LogProperties`, exactly because this source manager's `createLogWindow()` returns
// null.  `BaseMetaDataBase` nevertheless retains the Java-declared `LogProperties`
// reference type, whose Rust trait has no `Sync` bound.  The stored value is provably
// null for this constructor, while Java shares this manager across UI/process threads.
unsafe impl Send for DirectiveEditorManager {}
// See the `Send` explanation directly above.
unsafe impl Sync for DirectiveEditorManager {}

impl DirectiveEditorManager {
    /// Java `DirectiveEditorManager(DirectiveFileType, BaseManager, String, StringBuffer)`.
    pub fn new(
        file_type: Option<DirectiveFileType>,
        source_manager: Option<&'static dyn BaseManager>,
        timestamp: Option<&str>,
        errmsg: Option<&str>,
    ) -> &'static Self {
        // The Java constructor passes `this` to `DirectiveEditorMetaData` before
        // superclass setup.  This source-shaped Rust representation has to allocate
        // before it can make a `'static` manager reference; the metadata's manager is
        // only used by inherited image-name processing, which directive files do not
        // use, so it remains null until that adjacent path is translated.
        let instance = Box::leak(Box::new(Self {
            base: BaseManagerBase::initial(),
            source_manager,
            meta_data: DirectiveEditorMetaData::new(None, file_type, None, true),
            file_type,
            timestamp: timestamp.unwrap_or_default().to_string(),
            main_panel: None,
            dialog: None,
            dialog_errmsg: Mutex::new(errmsg.map(str::to_string)),
            save_file: Mutex::new(None),
        }));
        instance.base_manager();
        instance.create_state();
        let _ = instance.initialize_ui_parameters(None, Some(AXIS_ID), false);
        instance
    }

    /// Java `initialize`.  Open panel and add dialog.
    pub fn initialize(&self) {
        if !crate::imod::etomo::etomo_director::ARGUMENTS
            .lock()
            .unwrap()
            .is_headless()
        {
            self.open_processing_panel();
            // TODO(unit): MainDirectiveEditorPanel.setStatusBarText,
            // DirectiveEditorDialog, and UIHarness.toFront.
            let _ = STATUS_BAR_SIZE;
            self.open_directive_editor_dialog();
        }
    }

    /// Java `setName(File)`.
    pub fn set_name(&self, input_file: Option<&Path>) {
        let input_file = match input_file {
            Some(input_file) => input_file,
            None => return,
        };
        let parent = input_file
            .parent()
            .map(|parent| parent.to_string_lossy().into_owned());
        self.set_property_user_dir(parent.as_deref());
        self.meta_data.set_root_name(Some(input_file));
        // TODO(unit): MainDirectiveEditorPanel.setStatusBarText and UIHarness.setTitle.
        let _ = STATUS_BAR_SIZE;
    }

    /// Java `updateDirectiveMap(DirectiveMap, StringBuffer)`.
    // TODO(unit): etomo/storage/DirectiveMap.java.  BaseManager's corresponding
    // translated signature is equally blocked on `DirectiveMapInterface`.
    pub fn update_directive_map(&self, directive_map: Option<Infallible>, errmsg: &mut String) {
        let _ = (&self.source_manager, directive_map, errmsg);
    }

    /// Java `openDirectiveEditorDialog`.
    pub fn open_directive_editor_dialog(&self) {
        if self.dialog.is_none() {
            // TODO(unit): DirectiveEditorBuilder.java and DirectiveEditorDialog.java.
            // This is intentionally not replaced with a generic Slint panel: the
            // builder's directive-map rules and dialog checkpoint semantics are source
            // behavior, not presentation plumbing.
        }
        // TODO(unit): MainDirectiveEditorPanel.showProcess and
        // Utilities.prepareDialogActionMessage.
    }

    /// Java `getState`.
    // TODO(unit): etomo/type/ParallelState.java.
    pub fn get_state(&self) -> Option<Infallible> {
        None
    }

    /// Java private `createState`.
    fn create_state(&self) {}

    /// Java private `writeFile`.
    fn write_file(&self) -> bool {
        // TODO(unit): etomo/storage/DirectiveWriter.java, Directive.java and
        // DirectiveEditorDialog.java.  Writing an empty or synthetic autodoc here would
        // silently corrupt directives, so the unavailable boundary fails instead.
        false
    }

    /// Java private `openProcessingPanel`.
    fn open_processing_panel(&self) {
        // TODO(unit): MainDirectiveEditorPanel and AxisProcessData.java.  The source
        // calls showProcessingPanel, setPanel, then reconnect(savedProcessData,...).
    }
}

impl BaseManager for DirectiveEditorManager {
    fn base(&self) -> &BaseManagerBase {
        &self.base
    }

    fn this(&'static self) -> &'static dyn BaseManager {
        self
    }

    /// Java `getInterfaceType`.
    fn get_interface_type(&self) -> Option<InterfaceType> {
        Some(InterfaceType::Tools)
    }

    /// Java `createLogWindow`.
    fn create_log_window(&self) -> Option<Infallible> {
        None
    }

    /// Java `getLogInterface`.
    fn get_log_interface(&self) -> Option<Infallible> {
        None
    }

    /// Java `createMainPanel`.
    fn create_main_panel(&self) {
        if !crate::imod::etomo::etomo_director::ARGUMENTS
            .lock()
            .unwrap()
            .is_headless()
        {
            // TODO(unit): etomo/ui/swing/MainDirectiveEditorPanel.java.
        }
    }

    /// Java `getBaseMetaData`.
    fn get_base_meta_data(&self) -> Option<&dyn BaseMetaData> {
        Some(&self.meta_data)
    }

    /// Java `getMainPanel`.
    fn get_main_panel(&self) -> Option<Infallible> {
        self.main_panel
    }

    /// Java `getStorables(int)`.
    fn get_storables_with_offset(&self, _offset: i32) -> Option<Vec<Box<dyn Storable>>> {
        None
    }

    /// Java `isInManagerFrame`.
    fn is_in_manager_frame(&self) -> bool {
        true
    }

    /// Java `getProcessManager`.
    fn get_process_manager(&self) -> Option<Infallible> {
        None
    }

    /// Java `closeFrame`.
    fn close_frame(&self) -> bool {
        // TODO(unit): DirectiveEditorDialog.isDifferentFromCheckpoint and UIHarness's
        // Yes/No/Cancel dialog.  With no dialog created, Java's source would dereference
        // null; there is no user decision to synthesize in this translation boundary.
        self.dialog.is_none()
    }

    /// Java `saveToFile`.
    fn save_to_file(&self) -> bool {
        if self.save_file.lock().unwrap().is_none() {
            return self.save_as_to_file();
        }
        self.write_file()
    }

    /// Java `saveAsToFile`.
    fn save_as_to_file(&self) -> bool {
        // TODO(unit): DirectiveEditorDialog.getSaveFileAbsPath, UIHarness prompts, and
        // DirectiveWriter.  The Java method takes the proposed path from its dialog, so
        // no sensible Rust argument-free implementation exists before that dialog.
        false
    }

    /// Java `exitProgram(AxisID)`.
    fn exit_program(&self, axis_id: Option<AxisID>) -> bool {
        // The source catches Throwable around its superclass exit, endThreads, and
        // saveParamFile.  The translated super body does not throw.
        if self.exit_program_super(axis_id) {
            self.end_threads();
            self.save_param_file();
            return true;
        }
        false
    }

    /// Java `getName`.
    fn get_name(&self) -> Option<String> {
        self.meta_data.get_name()
    }
}

/// Java private static final `ConflictFileFilter`, which implements both Swing's file
/// chooser filter and `java.io.FileFilter`.
pub struct ConflictFileFilter {
    /// Java final `compareFileName`.
    compare_file_name: String,
}

impl ConflictFileFilter {
    /// Java private `ConflictFileFilter(String)`.
    pub fn new(compare_file_name: Option<&str>) -> Self {
        Self {
            compare_file_name: compare_file_name.unwrap_or_default().to_string(),
        }
    }

    /// Java `accept(File)`.
    pub fn accept(&self, file: Option<&Path>) -> bool {
        let file = match file {
            Some(file) if file.is_file() => file,
            _ => return false,
        };
        let file_name = match file.file_name() {
            Some(file_name) => file_name.to_string_lossy(),
            None => return false,
        };
        let is_exclusive_dataset = [
            DataFileType::Recon,
            DataFileType::Join,
            DataFileType::Peet,
            DataFileType::SerialSections,
        ]
        .into_iter()
        .filter_map(DataFileType::extension)
        .any(|extension| file_name.ends_with(extension));
        if !is_exclusive_dataset {
            return false;
        }
        let extension_index = match file_name.rfind('.') {
            Some(index) => index,
            None => return false,
        };
        file_name[..extension_index] == self.compare_file_name
    }

    /// Java `getDescription`.
    pub fn get_description(&self) -> String {
        format!(
            "Dataset file that conflicts with {}",
            self.compare_file_name
        )
    }
}

#[cfg(test)]
mod tests {
    use super::ConflictFileFilter;
    use std::fs;

    #[test]
    fn conflict_file_filter_only_accepts_matching_exclusive_dataset_file() {
        let directory =
            std::env::temp_dir().join(format!("imod-rs-directive-filter-{}", std::process::id()));
        fs::create_dir_all(&directory).unwrap();
        let matching = directory.join("sample.edf");
        let other = directory.join("other.edf");
        let parallel = directory.join("sample.epp");
        fs::write(&matching, []).unwrap();
        fs::write(&other, []).unwrap();
        fs::write(&parallel, []).unwrap();
        let filter = ConflictFileFilter::new(Some("sample"));
        assert!(filter.accept(Some(&matching)));
        assert!(!filter.accept(Some(&other)));
        assert!(!filter.accept(Some(&parallel)));
        fs::remove_dir_all(directory).unwrap();
    }
}
