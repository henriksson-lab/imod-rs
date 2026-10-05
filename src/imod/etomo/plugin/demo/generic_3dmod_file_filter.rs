//! `IMOD/Etomo/src/etomo/plugin/demo/Generic3dmodFileFilter.java`.
//!
//! `Generic3dmodFileFilter extends ExtensionFileFilter`: the superclass is embedded as
//! `base` and reached through `Deref`/`DerefMut`; the one override, `getDescription`,
//! is an inherent method here.  (The Java class comment, "Parameters for the
//! serieswatcher application", is a copy-paste leftover.)

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::storage::extension_file_filter::ExtensionFileFilter;
use crate::imod::etomo::r#type::extension::{self, Extension};

/// Java `final class Generic3dmodFileFilter extends ExtensionFileFilter`.
pub struct Generic3dmodFileFilter {
    /// Java superclass `ExtensionFileFilter` state.
    pub base: ExtensionFileFilter,
    /// Java private final `extensions` (never read in the Java either).
    #[allow(dead_code)]
    extensions: [&'static Extension; 10],
}

impl std::ops::Deref for Generic3dmodFileFilter {
    type Target = ExtensionFileFilter;

    fn deref(&self) -> &ExtensionFileFilter {
        &self.base
    }
}

impl std::ops::DerefMut for Generic3dmodFileFilter {
    fn deref_mut(&mut self) -> &mut ExtensionFileFilter {
        &mut self.base
    }
}

impl Generic3dmodFileFilter {
    /// Java private `Generic3dmodFileFilter(BaseManager)`.
    fn new(manager: Option<&'static dyn BaseManager>) -> Generic3dmodFileFilter {
        let c = &extension::CLASS;
        Generic3dmodFileFilter {
            base: ExtensionFileFilter::new(manager, true, false, false),
            extensions: [
                &c.st, &c.mrc, &c.ali, &c.join, &c.tif, &c.png, &c.rec, &c.r#mod, &c.jpg, &c.hdf,
            ],
        }
    }

    /// Java public static `getInstance(BaseManager)`.
    pub fn get_instance(manager: Option<&'static dyn BaseManager>) -> Generic3dmodFileFilter {
        let mut instance = Generic3dmodFileFilter::new(manager);
        let c = &extension::CLASS;
        // The Java array ends with a null element, which `SearchCollection.add`
        // skips (`if (inputExtensions[i] == null) continue`).
        instance.setup(
            None,
            None,
            None,
            None,
            Some(&[
                &c.st, &c.mrc, &c.ali, &c.join, &c.tif, &c.png, &c.rec, &c.r#mod, &c.jpg, &c.hdf,
            ]),
            None,
        );
        instance
    }

    /// Java `getDescription()`.
    pub fn get_description(&self) -> &'static str {
        "File to open in 3dmod"
    }
}

impl crate::imod::etomo::jdk::FileFilter for Generic3dmodFileFilter {
    fn accept(&self, file: &std::path::Path) -> bool {
        self.base.accept(Some(file))
    }

    fn get_description(&self) -> Option<String> {
        Some(Generic3dmodFileFilter::get_description(self).to_string())
    }
}
