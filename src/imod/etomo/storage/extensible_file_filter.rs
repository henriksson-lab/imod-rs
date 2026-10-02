//! `IMOD/Etomo/src/etomo/storage/ExtensibleFileFilter.java`.
//!
//! `public abstract class ExtensibleFileFilter extends ExtensionFileFilter`:
//! the superclass is embedded as `base` and reached through `Deref`, and the
//! one abstract method is the `ExtensibleFileFilterVirtual` trait, implemented
//! by the subclass.

use std::path::Path;

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::storage::extension_file_filter::ExtensionFileFilter;

/// Java `rcsid`.
pub const RCSID: &str = "$Id:$";

/// Java `ExtensibleFileFilter`'s abstract methods, implemented by subclasses.
pub trait ExtensibleFileFilterVirtual {
    /// Java abstract `addExtension(File)`.
    fn add_extension(&self, file: &Path);
}

/// Java `public abstract class ExtensibleFileFilter extends ExtensionFileFilter`.
pub struct ExtensibleFileFilter {
    /// Java superclass `ExtensionFileFilter` state.
    pub base: ExtensionFileFilter,
}

impl std::ops::Deref for ExtensibleFileFilter {
    type Target = ExtensionFileFilter;

    fn deref(&self) -> &ExtensionFileFilter {
        &self.base
    }
}

impl std::ops::DerefMut for ExtensibleFileFilter {
    fn deref_mut(&mut self) -> &mut ExtensionFileFilter {
        &mut self.base
    }
}

impl ExtensibleFileFilter {
    /// Java `ExtensibleFileFilter(BaseManager, boolean, boolean, boolean)`.
    pub fn new(
        manager: Option<&'static dyn BaseManager>,
        accept_directory: bool,
        use_all_file_styles: bool,
        debug: bool,
    ) -> ExtensibleFileFilter {
        ExtensibleFileFilter {
            base: ExtensionFileFilter::new(manager, accept_directory, use_all_file_styles, debug),
        }
    }
}
