//! `IMOD/Etomo/src/etomo/type/FrontPageMetaData.java`.

use std::collections::BTreeMap;
use std::sync::Mutex;

use super::axis_type::AxisType;
use super::base_meta_data::{BaseMetaData, BaseMetaDataBase};
use super::image_filename_style::ImageFilenameStyle;
use crate::imod::etomo::storage::storable::Storable;

pub const NAME: &str = "Front Page";
pub const GROUP_KEY: &str = "FrontPage";

pub struct FrontPageMetaData {
    base: BaseMetaDataBase,
    name: Mutex<Option<String>>,
}

// `FrontPageMetaData` is always constructed with `log_properties: None`: its
// no-file front-page source constructor has no LogProperties owner.  The
// BaseMetaDataBase representation permits that optional trait object for other
// metadata classes, which prevents automatic Send/Sync derivation even though
// this concrete instance cannot contain one.
unsafe impl Send for FrontPageMetaData {}
unsafe impl Sync for FrontPageMetaData {}

impl FrontPageMetaData {
    pub fn new(image_filename_style: Option<ImageFilenameStyle>) -> Self {
        let metadata = Self {
            base: BaseMetaDataBase::new_force_old_style_and_image_filename_style(
                None,
                None,
                false,
                image_filename_style,
                true,
                true,
            ),
            name: Mutex::new(None),
        };
        *metadata.base.axis_type.lock().unwrap() = AxisType::SingleAxis;
        metadata
    }

    pub fn set_name(&self, name: impl Into<String>) {
        *self.name.lock().unwrap() = Some(name.into());
    }
    pub fn set_axis_type(&self, axis_type: AxisType) {
        *self.base.axis_type.lock().unwrap() = axis_type;
    }

    /// Java `toString`.  Java's superclass suffix is identity-dependent; as
    /// elsewhere in the translation it is deliberately omitted while the
    /// concrete class/group framing is preserved.
    pub fn to_string(&self) -> String {
        format!("etomo.type.FrontPageMetaData[{GROUP_KEY}]\n")
    }
}

impl Storable for FrontPageMetaData {
    fn store(&self, properties: &mut BTreeMap<String, String>) {
        self.base
            .store_with_created_prepend(properties, Some(GROUP_KEY));
    }
    fn store_with_prepend(&self, properties: &mut BTreeMap<String, String>, prepend: &str) {
        self.base
            .store_with_created_prepend(properties, self.create_prepend(prepend).as_deref());
    }
    fn load(&mut self, properties: &BTreeMap<String, String>) {
        if self
            .base
            .load_with_created_prepend(properties, Some(GROUP_KEY))
        {
            self.check_image_filename_style_loaded("");
        }
    }
    fn load_with_prepend(&mut self, properties: &BTreeMap<String, String>, prepend: &str) {
        if self
            .base
            .load_with_created_prepend(properties, self.create_prepend(prepend).as_deref())
        {
            self.check_image_filename_style_loaded(prepend);
        }
    }
}

impl BaseMetaData for FrontPageMetaData {
    fn base(&self) -> &BaseMetaDataBase {
        &self.base
    }
    fn get_meta_data_file_name(&self) -> Option<String> {
        self.get_name()
    }
    fn get_name(&self) -> Option<String> {
        Some(
            self.name
                .lock()
                .unwrap()
                .clone()
                .unwrap_or_else(|| NAME.to_owned()),
        )
    }
    fn get_dataset_name(&self) -> Option<String> {
        self.get_name()
    }
    fn is_valid(&self) -> bool {
        true
    }
    fn get_group_key(&self) -> Option<String> {
        Some(GROUP_KEY.to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn source_default_and_test_name_contracts_hold() {
        let metadata = FrontPageMetaData::new(None);
        assert_eq!(metadata.get_name().as_deref(), Some(NAME));
        assert!(metadata.is_valid());
        metadata.set_name("test");
        assert_eq!(metadata.get_dataset_name().as_deref(), Some("test"));
        assert_eq!(
            metadata.to_string(),
            "etomo.type.FrontPageMetaData[FrontPage]\n"
        );
    }
}
