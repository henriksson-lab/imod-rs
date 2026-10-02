//! `IMOD/Etomo/src/etomo/storage/ComFileFileFilter.java`.
//!
//! `ComFileFileFilter extends ExtensionFileFilter`: the superclass is embedded as `base`
//! and reached through `Deref`/`DerefMut`.  The one override, `getDescription`,
//! is an inherent method here and shadows the base's.

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::storage::extension_file_filter::ExtensionFileFilter;
use crate::imod::etomo::r#type::extension;

/// Java `public class ComFileFileFilter extends ExtensionFileFilter`.
pub struct ComFileFileFilter {
    /// Java superclass `ExtensionFileFilter` state.
    pub base: ExtensionFileFilter,
}

impl std::ops::Deref for ComFileFileFilter {
    type Target = ExtensionFileFilter;

    fn deref(&self) -> &ExtensionFileFilter {
        &self.base
    }
}

impl std::ops::DerefMut for ComFileFileFilter {
    fn deref_mut(&mut self) -> &mut ExtensionFileFilter {
        &mut self.base
    }
}

impl ComFileFileFilter {
    /// Java private `ComFileFileFilter(BaseManager, boolean)`.
    fn new(
        manager: Option<&'static dyn BaseManager>,
        use_all_file_styles: bool,
    ) -> ComFileFileFilter {
        ComFileFileFilter {
            base: ExtensionFileFilter::new(manager, true, use_all_file_styles, false),
        }
    }

    /// Java static `getInstance(BaseManager, boolean)`.
    pub fn get_instance(
        manager: Option<&'static dyn BaseManager>,
        use_all_file_styles: bool,
    ) -> ComFileFileFilter {
        let mut instance = ComFileFileFilter::new(manager, use_all_file_styles);
        instance.setup(
            None,
            None,
            None,
            None,
            Some(&[&extension::CLASS.com, &extension::CLASS.pcm]),
            None,
        );
        instance
    }

    /// Java `getDescription()`.
    pub fn get_description(&self) -> &'static str {
        "COM or PCM"
    }
}

impl crate::imod::etomo::jdk::FileFilter for ComFileFileFilter {
    fn accept(&self, file: &std::path::Path) -> bool {
        self.base.accept(Some(file))
    }

    fn get_description(&self) -> Option<String> {
        Some(ComFileFileFilter::get_description(self).to_string())
    }
}
