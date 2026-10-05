//! `IMOD/Etomo/src/etomo/storage/VolumeFileFilter.java`.
//!
//! `VolumeFileFilter extends ExtensionFileFilter`: the superclass is embedded as
//! `base` and reached through `Deref` (as in `reduce_filt_vol_file_filter.rs`).

use std::path::Path;

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::jdk::FileFilter;
use crate::imod::etomo::storage::extension_file_filter::ExtensionFileFilter;
use crate::imod::etomo::r#type::extension;

/// Java `public class VolumeFileFilter extends ExtensionFileFilter`.
pub struct VolumeFileFilter {
    /// Java superclass `ExtensionFileFilter` state.
    pub base: ExtensionFileFilter,
}

impl std::ops::Deref for VolumeFileFilter {
    type Target = ExtensionFileFilter;

    fn deref(&self) -> &ExtensionFileFilter {
        &self.base
    }
}

impl VolumeFileFilter {
    /// Java private `VolumeFileFilter(BaseManager)`.
    fn new(manager: Option<&'static dyn BaseManager>) -> VolumeFileFilter {
        VolumeFileFilter {
            base: ExtensionFileFilter::new(manager, true, true, false),
        }
    }

    /// Java static `getInstance(BaseManager)`.
    pub fn get_instance(manager: Option<&'static dyn BaseManager>) -> VolumeFileFilter {
        let mut instance = VolumeFileFilter::new(manager);
        instance.base.setup(
            None,
            None,
            None,
            None,
            Some(&[&extension::CLASS.mrc, &extension::CLASS.rec]),
            None,
        );
        instance
    }
}

impl FileFilter for VolumeFileFilter {
    fn accept(&self, file: &Path) -> bool {
        self.base.accept(Some(file))
    }

    /// Java `getDescription()`.
    fn get_description(&self) -> Option<String> {
        Some("Volume".to_owned())
    }
}
