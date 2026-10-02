//! `IMOD/Etomo/src/etomo/storage/TiltExtensionFileFilter.java`.
//!
//! `TiltExtensionFileFilter extends ExtensionFileFilter`: the superclass is embedded as `base`
//! and reached through `Deref`/`DerefMut`.  The one override, `getDescription`,
//! is an inherent method here and shadows the base's.

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::storage::extension_file_filter::ExtensionFileFilter;
use crate::imod::etomo::r#type::extension;

/// Java `public class TiltExtensionFileFilter extends ExtensionFileFilter`.
pub struct TiltExtensionFileFilter {
    /// Java superclass `ExtensionFileFilter` state.
    pub base: ExtensionFileFilter,
}

impl std::ops::Deref for TiltExtensionFileFilter {
    type Target = ExtensionFileFilter;

    fn deref(&self) -> &ExtensionFileFilter {
        &self.base
    }
}

impl std::ops::DerefMut for TiltExtensionFileFilter {
    fn deref_mut(&mut self) -> &mut ExtensionFileFilter {
        &mut self.base
    }
}

impl TiltExtensionFileFilter {
    /// Java private `TiltExtensionFileFilter(BaseManager, boolean)`.
    fn new(
        manager: Option<&'static dyn BaseManager>,
        use_all_file_styles: bool,
    ) -> TiltExtensionFileFilter {
        TiltExtensionFileFilter {
            base: ExtensionFileFilter::new(manager, true, use_all_file_styles, false),
        }
    }

    /// Java static `getInstance(BaseManager, boolean)`.
    pub fn get_instance(
        manager: Option<&'static dyn BaseManager>,
        use_all_file_styles: bool,
    ) -> TiltExtensionFileFilter {
        let mut instance = TiltExtensionFileFilter::new(manager, use_all_file_styles);
        instance.setup(
            None,
            None,
            None,
            None,
            Some(&[&extension::CLASS.tlt, &extension::CLASS.rawtlt]),
            None,
        );
        instance
    }

    /// Java `getDescription()`.
    pub fn get_description(&self) -> &'static str {
        "TLT or RAWTLT"
    }
}

impl crate::imod::etomo::jdk::FileFilter for TiltExtensionFileFilter {
    fn accept(&self, file: &std::path::Path) -> bool {
        self.base.accept(Some(file))
    }

    fn get_description(&self) -> Option<String> {
        Some(TiltExtensionFileFilter::get_description(self).to_string())
    }
}
