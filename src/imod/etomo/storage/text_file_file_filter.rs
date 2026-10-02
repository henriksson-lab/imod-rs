//! `IMOD/Etomo/src/etomo/storage/TextFileFileFilter.java`.
//!
//! `TextFileFileFilter extends ExtensionFileFilter`: the superclass is embedded as `base`
//! and reached through `Deref`/`DerefMut`.  The one override, `getDescription`,
//! is an inherent method here and shadows the base's.

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::storage::extension_file_filter::ExtensionFileFilter;
use crate::imod::etomo::r#type::extension;

/// Java `public class TextFileFileFilter extends ExtensionFileFilter`.
pub struct TextFileFileFilter {
    /// Java superclass `ExtensionFileFilter` state.
    pub base: ExtensionFileFilter,
}

impl std::ops::Deref for TextFileFileFilter {
    type Target = ExtensionFileFilter;

    fn deref(&self) -> &ExtensionFileFilter {
        &self.base
    }
}

impl std::ops::DerefMut for TextFileFileFilter {
    fn deref_mut(&mut self) -> &mut ExtensionFileFilter {
        &mut self.base
    }
}

impl TextFileFileFilter {
    /// Java private `TextFileFileFilter(BaseManager, boolean)`.
    fn new(
        manager: Option<&'static dyn BaseManager>,
        use_all_file_styles: bool,
    ) -> TextFileFileFilter {
        TextFileFileFilter {
            base: ExtensionFileFilter::new(manager, true, use_all_file_styles, false),
        }
    }

    /// Java static `getInstance(BaseManager, boolean)`.
    pub fn get_instance(
        manager: Option<&'static dyn BaseManager>,
        use_all_file_styles: bool,
    ) -> TextFileFileFilter {
        let mut instance = TextFileFileFilter::new(manager, use_all_file_styles);
        instance.setup(None, None, None, None, Some(&[&extension::CLASS.txt]), None);
        instance
    }

    /// Java `getDescription()`.
    pub fn get_description(&self) -> &'static str {
        "TXT"
    }
}

impl crate::imod::etomo::jdk::FileFilter for TextFileFileFilter {
    fn accept(&self, file: &std::path::Path) -> bool {
        self.base.accept(Some(file))
    }

    fn get_description(&self) -> Option<String> {
        Some(TextFileFileFilter::get_description(self).to_string())
    }
}
