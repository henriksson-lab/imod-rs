//! `IMOD/Etomo/src/etomo/type/DirectiveEditorMetaData.java`.
#![allow(dead_code)]

use super::axis_type::AxisType;
use super::base_meta_data::{BaseMetaData, BaseMetaDataBase};
use super::directive_file_type::DirectiveFileType;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::storage::storable::Storable;
use crate::imod::etomo::ui::log_properties::LogProperties;
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Mutex;

/// Java `DirectiveEditorMetaData`.
pub struct DirectiveEditorMetaData {
    /// Java superclass `BaseMetaData` state.
    base: BaseMetaDataBase,
    /// Java private final `type`.
    file_type: Option<DirectiveFileType>,
    /// Java private `rootName`, initially null.
    root_name: Mutex<Option<String>>,
}

impl DirectiveEditorMetaData {
    /// Java `DirectiveEditorMetaData(BaseManager, DirectiveFileType, LogProperties, boolean)`.
    pub fn new(
        manager: Option<&'static dyn BaseManager>,
        file_type: Option<DirectiveFileType>,
        log_properties: Option<&'static dyn LogProperties>,
        new_dataset: bool,
    ) -> Self {
        let instance = Self {
            base: BaseMetaDataBase::new_force_old_style(
                manager,
                log_properties,
                false,
                new_dataset,
                true,
            ),
            file_type,
            root_name: Mutex::new(None),
        };
        *instance.base.axis_type.lock().unwrap() = AxisType::SingleAxis;
        instance
    }

    /// Java `setRootName(File)`.
    pub fn set_root_name(&self, file: Option<&Path>) {
        *self.root_name.lock().unwrap() = file.and_then(|file| {
            file.file_name()
                .map(|name| name.to_string_lossy().into_owned())
        });
    }

    /// Java `validate`; null in Java means valid.
    pub fn validate(&self) -> Option<String> {
        if self.root_name.lock().unwrap().is_none() {
            return Some("Missing root name.".to_string());
        }
        None
    }
}

impl Storable for DirectiveEditorMetaData {
    /// Java inherited `BaseMetaData.store(Properties)`.
    fn store(&self, properties: &mut BTreeMap<String, String>) {
        self.base
            .store_with_created_prepend(properties, self.get_group_key().as_deref());
    }

    /// Java inherited `BaseMetaData.store(Properties, String)`.
    fn store_with_prepend(&self, properties: &mut BTreeMap<String, String>, prepend: &str) {
        self.base
            .store_with_created_prepend(properties, self.create_prepend(prepend).as_deref());
    }

    /// Java inherited `BaseMetaData.load(Properties)`.
    fn load(&mut self, properties: &BTreeMap<String, String>) {
        if self
            .base
            .load_with_created_prepend(properties, self.get_group_key().as_deref())
        {
            self.check_image_filename_style_loaded("");
        }
    }

    /// Java inherited `BaseMetaData.load(Properties, String)`.
    fn load_with_prepend(&mut self, properties: &BTreeMap<String, String>, prepend: &str) {
        if self
            .base
            .load_with_created_prepend(properties, self.create_prepend(prepend).as_deref())
        {
            self.check_image_filename_style_loaded(prepend);
        }
    }
}

impl BaseMetaData for DirectiveEditorMetaData {
    fn base(&self) -> &BaseMetaDataBase {
        &self.base
    }

    /// Java `getGroupKey`.
    fn get_group_key(&self) -> Option<String> {
        None
    }

    /// Java `createPrepend`.
    fn create_prepend(&self, _prepend: &str) -> Option<String> {
        None
    }

    /// Java `getDatasetName`.
    fn get_dataset_name(&self) -> Option<String> {
        self.root_name.lock().unwrap().clone()
    }

    /// Java `getMetaDataFileName`.
    fn get_meta_data_file_name(&self) -> Option<String> {
        None
    }

    /// Java `getName`.
    fn get_name(&self) -> Option<String> {
        if let Some(root_name) = self.root_name.lock().unwrap().clone() {
            return Some(root_name);
        }
        if let Some(file_type) = self.file_type {
            return Some(format!("{} Editor", file_type.get_label()));
        }
        Some("Directive File Editor".to_string())
    }

    /// Java `isValid`.
    fn is_valid(&self) -> bool {
        self.validate().is_none()
    }
}
