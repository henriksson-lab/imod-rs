//! `IMOD/Etomo/src/etomo/storage/FrameFileFilter.java`.
//!
//! `FrameFileFilter extends ExtensionFileFilter`: the superclass is embedded as `base`
//! and reached through `Deref`/`DerefMut`.  The one override, `getDescription`,
//! is an inherent method here and shadows the base's.

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::storage::extension_file_filter::ExtensionFileFilter;
use crate::imod::etomo::r#type::extension;

/// Java `public class FrameFileFilter extends ExtensionFileFilter`.
pub struct FrameFileFilter {
    /// Java superclass `ExtensionFileFilter` state.
    pub base: ExtensionFileFilter,
}

impl std::ops::Deref for FrameFileFilter {
    type Target = ExtensionFileFilter;

    fn deref(&self) -> &ExtensionFileFilter {
        &self.base
    }
}

impl std::ops::DerefMut for FrameFileFilter {
    fn deref_mut(&mut self) -> &mut ExtensionFileFilter {
        &mut self.base
    }
}

impl FrameFileFilter {
    /// Java private `FrameFileFilter(BaseManager, boolean)`.
    fn new(
        manager: Option<&'static dyn BaseManager>,
        use_all_file_styles: bool,
    ) -> FrameFileFilter {
        FrameFileFilter {
            base: ExtensionFileFilter::new(manager, true, use_all_file_styles, false),
        }
    }

    /// Java static `getInstance(BaseManager, boolean)`.
    pub fn get_instance(
        manager: Option<&'static dyn BaseManager>,
        use_all_file_styles: bool,
    ) -> FrameFileFilter {
        let mut instance = FrameFileFilter::new(manager, use_all_file_styles);
        instance.setup(
            None,
            None,
            None,
            None,
            Some(&[
                &extension::CLASS.mrc,
                &extension::CLASS.tif,
                &extension::CLASS.tiff,
                &extension::CLASS.eer,
            ]),
            None,
        );
        instance
    }

    /// Java `getDescription()`.
    pub fn get_description(&self) -> &'static str {
        "MRC or TIF or EER"
    }
}

impl crate::imod::etomo::jdk::FileFilter for FrameFileFilter {
    fn accept(&self, file: &std::path::Path) -> bool {
        self.base.accept(Some(file))
    }

    fn get_description(&self) -> Option<String> {
        Some(FrameFileFilter::get_description(self).to_string())
    }
}
