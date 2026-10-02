//! `IMOD/Etomo/src/etomo/storage/ReduceFiltVolFileFilter.java`.
//!
//! `ReduceFiltVolFileFilter extends ExtensionFileFilter`: the superclass is
//! embedded as `base` and reached through `Deref`, so the inherited
//! `accept(File)` and `accept(File, String)` (`accept_in_dir`) are the
//! superclass's.  With `getDescription` they are also the `jdk::FileFilter`
//! implementation.

use std::path::Path;

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::jdk::FileFilter;
use crate::imod::etomo::storage::extension_file_filter::ExtensionFileFilter;
use crate::imod::etomo::r#type::file_type;

/// Java `public class ReduceFiltVolFileFilter extends ExtensionFileFilter`.
pub struct ReduceFiltVolFileFilter {
    /// Java superclass `ExtensionFileFilter` state.
    pub base: ExtensionFileFilter,
}

impl std::ops::Deref for ReduceFiltVolFileFilter {
    type Target = ExtensionFileFilter;

    fn deref(&self) -> &ExtensionFileFilter {
        &self.base
    }
}

impl std::ops::DerefMut for ReduceFiltVolFileFilter {
    fn deref_mut(&mut self) -> &mut ExtensionFileFilter {
        &mut self.base
    }
}

impl ReduceFiltVolFileFilter {
    /// Java private `ReduceFiltVolFileFilter(BaseManager)`.
    fn new(manager: &'static dyn BaseManager) -> ReduceFiltVolFileFilter {
        ReduceFiltVolFileFilter {
            base: ExtensionFileFilter::new(Some(manager), false, false, false),
        }
    }

    /// Java static `getInstance(BaseManager)`.
    pub fn get_instance(manager: &'static dyn BaseManager) -> ReduceFiltVolFileFilter {
        let mut instance = ReduceFiltVolFileFilter::new(manager);
        let name = manager.get_name();
        instance.setup(
            name.as_deref(),
            None,
            None,
            None,
            None,
            Some(&[Some(&*file_type::CLASS.reduce_filt_vol_output_file)]),
        );
        instance
    }

    /// Java `getDescription()`: the source returns null (an auto-generated
    /// stub).
    pub fn get_description(&self) -> Option<String> {
        // TODO Auto-generated method stub
        None
    }
}

impl FileFilter for ReduceFiltVolFileFilter {
    fn accept(&self, file: &Path) -> bool {
        self.base.accept(Some(file))
    }

    fn get_description(&self) -> Option<String> {
        ReduceFiltVolFileFilter::get_description(self)
    }
}
