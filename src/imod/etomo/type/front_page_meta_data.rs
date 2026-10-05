//! `IMOD/Etomo/src/etomo/type/FrontPageMetaData.java`.
//!
//! Meta data for FrontPageManager.  MetaData is a required class for a
//! manager.

use std::collections::BTreeMap;
use std::sync::Mutex;

use super::axis_type::AxisType;
use super::base_meta_data::{BaseMetaData, BaseMetaDataBase};
use super::image_filename_style::ImageFilenameStyle;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::storage::storable::{Storable, StorableValue};
use crate::imod::etomo::ui::log_properties::LogProperties;

/// Java private static final `NAME`.
pub const NAME: &str = "Front Page";
/// Java private static final `FRONT_PAGE_GROUP_KEY`.
pub const FRONT_PAGE_GROUP_KEY: &str = "FrontPage";

/// Java public final class `FrontPageMetaData extends BaseMetaData`.
pub struct FrontPageMetaData {
    /// The Java superclass part.
    base: BaseMetaDataBase,
    /// Java private `name`, initialised to null.  For testing.
    name: Mutex<Option<String>>,
}

// Safety: every field is a `Mutex` of owned data except `BaseMetaDataBase`'s
// `&'static` references.  The manager is `Send + Sync` (the `BaseManager` trait
// requires it).  The member that keeps the auto traits from applying is
// `BaseMetaDataBase`'s `Option<&'static dyn LogProperties>` (the log window, an
// EDT object; `FrontPageManager.createLogWindow` returns null, so for the front
// page it is always null).  `BaseMetaDataBase` never calls through it and
// nothing in this module reaches it, so sharing the reference across threads
// never touches the object behind it.  Same argument as `meta_data.rs`.
unsafe impl Send for FrontPageMetaData {}
unsafe impl Sync for FrontPageMetaData {}

impl FrontPageMetaData {
    /// Java `FrontPageMetaData(BaseManager, LogProperties, ImageFilenameStyle,
    /// boolean)`.
    pub fn new(
        manager: Option<&'static dyn BaseManager>,
        log_properties: Option<&'static dyn LogProperties>,
        image_filename_style: Option<ImageFilenameStyle>,
        new_dataset: bool,
    ) -> Self {
        let metadata = Self {
            // super(manager, logProperties, false, imageFilenameStyle, newDataset, true)
            base: BaseMetaDataBase::new_force_old_style_and_image_filename_style(
                manager,
                log_properties,
                false,
                image_filename_style,
                new_dataset,
                true,
            ),
            name: Mutex::new(None),
        };
        *metadata.base.axis_type.lock().unwrap() = AxisType::SingleAxis;
        metadata
    }

    /// Java `toString()`: `getClass().getName() + "[" + paramString() + "]\n"`.
    pub fn to_string(&self) -> String {
        format!("etomo.type.FrontPageMetaData[{}]\n", self.param_string())
    }

    /// Java package-private `paramString()`.
    pub fn param_string(&self) -> String {
        FRONT_PAGE_GROUP_KEY.to_owned()
    }

    /// Java `setAxisType(AxisType)`.  For testing.
    pub fn set_axis_type(&self, axis_type: AxisType) {
        *self.base.axis_type.lock().unwrap() = axis_type;
    }

    /// Java `setName(String)`.  For testing.
    pub fn set_name(&self, name: Option<&str>) {
        *self.name.lock().unwrap() = name.map(str::to_owned);
    }
}

impl Storable for FrontPageMetaData {
    /// Java `store(Properties)`, inherited from `BaseMetaData`:
    /// `store(props, "")`, which reaches this class's override.
    fn store(&self, properties: &mut BTreeMap<String, String>) {
        self.store_with_prepend(properties, "");
    }

    /// Java `store(Properties, String)`.
    fn store_with_prepend(&self, properties: &mut BTreeMap<String, String>, prepend: &str) {
        // super.store(props, prepend)
        self.base
            .store_with_created_prepend(properties, self.create_prepend(prepend).as_deref());
        let prepend = self.create_prepend(prepend).unwrap_or("null".to_owned());
        // Java's unused local `String group = prepend + ".";`.
        let _group = format!("{prepend}.");
    }

    /// Java `load(Properties)`: `load(props, "")`.
    fn load(&self, properties: &mut BTreeMap<String, String>) {
        self.load_with_prepend(properties, "");
    }

    /// Java `load(Properties, String)`.
    fn load_with_prepend(&self, properties: &mut BTreeMap<String, String>, prepend: &str) {
        // super.load(props, prepend): BaseMetaData.load creates the prepend,
        // loads, and then (canCorrectImageFilenameStyle is true here) calls
        // checkImageFilenameStyleLoaded with the *created* prepend.
        let super_prepend = self.create_prepend(prepend);
        if self
            .base
            .load_with_created_prepend(properties, super_prepend.as_deref())
        {
            self.check_image_filename_style_loaded(super_prepend.as_deref().unwrap_or("null"));
        }
        // reset
        // load
        let prepend = self.create_prepend(prepend).unwrap_or("null".to_owned());
        // Java's unused local `String group = prepend + ".";`.
        let _group = format!("{prepend}.");
    }
}

impl BaseMetaData for FrontPageMetaData {
    fn base(&self) -> &BaseMetaDataBase {
        &self.base
    }

    /// Java `getName()`.
    fn get_name(&self) -> Option<String> {
        // for testing
        if let Some(name) = self.name.lock().unwrap().clone() {
            return Some(name);
        }
        Some(NAME.to_owned())
    }

    /// Java `getDatasetName()`.
    fn get_dataset_name(&self) -> Option<String> {
        // for testing
        if let Some(name) = self.name.lock().unwrap().clone() {
            return Some(name);
        }
        Some(NAME.to_owned())
    }

    /// Java `getMetaDataFileName()`.
    fn get_meta_data_file_name(&self) -> Option<String> {
        // for testing
        if let Some(name) = self.name.lock().unwrap().clone() {
            return Some(name);
        }
        Some(NAME.to_owned())
    }

    /// Java `isValid()`.
    fn is_valid(&self) -> bool {
        true
    }

    /// Java package-private `getGroupKey()`.
    fn get_group_key(&self) -> Option<String> {
        Some(FRONT_PAGE_GROUP_KEY.to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn source_default_and_test_name_contracts_hold() {
        let metadata = FrontPageMetaData::new(None, None, None, true);
        assert_eq!(metadata.get_name().as_deref(), Some(NAME));
        assert_eq!(metadata.base().get_axis_type(), AxisType::SingleAxis);
        assert!(metadata.is_valid());
        metadata.set_name(Some("test"));
        assert_eq!(metadata.get_dataset_name().as_deref(), Some("test"));
        assert_eq!(
            metadata.to_string(),
            "etomo.type.FrontPageMetaData[FrontPage]\n"
        );
    }
}
