//! `IMOD/Etomo/src/etomo/type/ToolsMetaData.java`.
//!
//! Meta data for `ToolsManager`.  Tools projects are not saved
//! (`getMetaDataFileName` returns null), so the inherited `store`/`load` only
//! run if something stores the manager's storables, which `ToolsManager`
//! reports as null.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Mutex;

use super::axis_type::AxisType;
use super::base_meta_data::{BaseMetaData, BaseMetaDataBase};
use super::dialog_type::DialogType;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::storage::storable::Storable;
use crate::imod::etomo::ui::log_properties::LogProperties;
use crate::imod::etomo::ui::swing::etomo_menu::ToolType;

/// Java `public static final String rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `public final class ToolsMetaData extends BaseMetaData`.
pub struct ToolsMetaData {
    /// The Java superclass part.
    base: BaseMetaDataBase,
    /// Java private final `dialogType`.
    dialog_type: DialogType,
    /// Java private final `toolType`.
    tool_type: ToolType,
    /// Java private `rootName`, initialised to null.
    root_name: Mutex<Option<String>>,
}

// Safety: every field is a `Mutex` of owned data or `Copy` except
// `BaseMetaDataBase`'s `&'static` references.  The manager is `Send + Sync`;
// the log properties are the manager's log window, which `ToolsManager`
// never creates (`createLogWindow` returns null), so the reference is always
// null.  Same argument as `front_page_meta_data.rs`.
unsafe impl Send for ToolsMetaData {}
unsafe impl Sync for ToolsMetaData {}

impl ToolsMetaData {
    /// Java `ToolsMetaData(BaseManager, DialogType, ToolType, LogProperties,
    /// boolean)`.
    pub fn new(
        manager: Option<&'static dyn BaseManager>,
        dialog_type: DialogType,
        tool_type: ToolType,
        log_properties: Option<&'static dyn LogProperties>,
        new_dataset: bool,
    ) -> ToolsMetaData {
        let meta_data = ToolsMetaData {
            // super(manager, logProperties, false, newDataset, true)
            base: BaseMetaDataBase::new_force_old_style(
                manager,
                log_properties,
                false,
                new_dataset,
                true,
            ),
            dialog_type,
            tool_type,
            root_name: Mutex::new(None),
        };
        *meta_data.base.axis_type.lock().unwrap() = AxisType::SingleAxis;
        meta_data
    }

    /// Java `setRootName(File)`.
    pub fn set_root_name_file(&self, file: &Path) {
        *self.root_name.lock().unwrap() = Some(
            file.file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_default(),
        );
    }

    /// Java `setRootName(String)`.
    pub fn set_root_name_string(&self, rootname: Option<&str>) {
        *self.root_name.lock().unwrap() = rootname.map(str::to_owned);
    }

    /// Java `validate()`: returns null if valid, otherwise the error message.
    pub fn validate(&self) -> Option<String> {
        if self.root_name.lock().unwrap().is_none() {
            return Some("Missing root name.".to_owned());
        }
        None
    }

    /// Java private final `dialogType` (no getter in the source).
    pub fn dialog_type(&self) -> DialogType {
        self.dialog_type
    }
}

impl Storable for ToolsMetaData {
    /// Java `store(Properties)`, inherited from `BaseMetaData`.
    fn store(&self, properties: &mut BTreeMap<String, String>) {
        self.store_with_prepend(properties, "");
    }

    /// Java `store(Properties, String)`, inherited from `BaseMetaData`.
    fn store_with_prepend(&self, properties: &mut BTreeMap<String, String>, prepend: &str) {
        self.base
            .store_with_created_prepend(properties, self.create_prepend(prepend).as_deref());
    }

    /// Java `load(Properties)`, inherited from `BaseMetaData`.
    fn load(&self, properties: &BTreeMap<String, String>) {
        self.load_with_prepend(properties, "");
    }

    /// Java `load(Properties, String)`, inherited from `BaseMetaData`
    /// (`canCorrectImageFilenameStyle` is true here).
    fn load_with_prepend(&self, properties: &BTreeMap<String, String>, prepend: &str) {
        let prepend = self.create_prepend(prepend);
        if self
            .base
            .load_with_created_prepend(properties, prepend.as_deref())
        {
            self.check_image_filename_style_loaded(prepend.as_deref().unwrap_or("null"));
        }
    }
}

impl BaseMetaData for ToolsMetaData {
    fn base(&self) -> &BaseMetaDataBase {
        &self.base
    }

    /// Java package-private `getGroupKey()`.
    fn get_group_key(&self) -> Option<String> {
        None
    }

    /// Java package-private `createPrepend(String)`.
    fn create_prepend(&self, _prepend: &str) -> Option<String> {
        None
    }

    /// Java `getDatasetName()`.
    fn get_dataset_name(&self) -> Option<String> {
        self.root_name.lock().unwrap().clone()
    }

    /// Java `getMetaDataFileName()`.
    fn get_meta_data_file_name(&self) -> Option<String> {
        // Tools projects are not saved
        None
    }

    /// Java `getName()`.
    fn get_name(&self) -> Option<String> {
        if let Some(root_name) = self.root_name.lock().unwrap().clone() {
            return Some(root_name);
        }
        Some(self.tool_type.label().to_owned())
    }

    /// Java `isValid()`.
    fn is_valid(&self) -> bool {
        self.validate().is_none()
    }
}
