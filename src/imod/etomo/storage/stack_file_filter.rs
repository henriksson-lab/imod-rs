//! `IMOD/Etomo/src/etomo/storage/StackFileFilter.java`.
//!
//! `StackFileFilter extends ExtensionFileFilter`: the superclass is embedded as `base`
//! and reached through `Deref`/`DerefMut`, so every inherited member (`accept`,
//! `accept_in_dir`, `setup`, ...) is callable on a `StackFileFilter`.  The one
//! override, `getDescription`, is an inherent method here and shadows the base's.

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::storage::extension_file_filter::ExtensionFileFilter;
use crate::imod::etomo::r#type::extension;

/// Java `StackFileFilter`.
pub struct StackFileFilter {
    /// Java superclass `ExtensionFileFilter` state.
    pub base: ExtensionFileFilter,
}

impl std::ops::Deref for StackFileFilter {
    type Target = ExtensionFileFilter;

    fn deref(&self) -> &ExtensionFileFilter {
        &self.base
    }
}

impl std::ops::DerefMut for StackFileFilter {
    fn deref_mut(&mut self) -> &mut ExtensionFileFilter {
        &mut self.base
    }
}

impl StackFileFilter {
    /// Java private `StackFileFilter(BaseManager, boolean)`.
    fn new(
        manager: Option<&'static dyn BaseManager>,
        use_all_file_styles: bool,
    ) -> StackFileFilter {
        StackFileFilter {
            base: ExtensionFileFilter::new(manager, true, use_all_file_styles, false),
        }
    }

    /// Java static `getInstance(BaseManager, boolean)`.
    pub fn get_instance(
        manager: Option<&'static dyn BaseManager>,
        use_all_file_styles: bool,
    ) -> StackFileFilter {
        let mut instance = StackFileFilter::new(manager, use_all_file_styles);
        instance.setup(
            None,
            None,
            None,
            None,
            Some(&[
                &extension::CLASS.st,
                &extension::CLASS.mrc,
                &extension::CLASS.hdf,
                &extension::CLASS.tif,
                &extension::CLASS.tiff,
            ]),
            None,
        );
        instance
    }

    /// Java `getDescription()`.
    pub fn get_description(&self) -> &'static str {
        "MRC, HDF, or TIF Image Stack"
    }
}

impl crate::imod::etomo::jdk::FileFilter for StackFileFilter {
    fn accept(&self, file: &std::path::Path) -> bool {
        self.base.accept(Some(file))
    }

    fn get_description(&self) -> Option<String> {
        Some(StackFileFilter::get_description(self).to_string())
    }
}
