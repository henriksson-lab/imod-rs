//! `IMOD/Etomo/src/etomo/storage/ChunkFileFilter.java`.
//!
//! `ChunkFileFilter extends ExtensionFileFilter`: the superclass is embedded as `base`
//! and reached through `Deref` (as in `volume_file_filter.rs`).

use std::path::Path;

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::jdk::FileFilter;
use crate::imod::etomo::storage::extension_file_filter::ExtensionFileFilter;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::file_type;

/// Java `public class ChunkFileFilter extends ExtensionFileFilter`.
pub struct ChunkFileFilter {
    /// Java superclass `ExtensionFileFilter` state.
    pub base: ExtensionFileFilter,
}

impl std::ops::Deref for ChunkFileFilter {
    type Target = ExtensionFileFilter;

    fn deref(&self) -> &ExtensionFileFilter {
        &self.base
    }
}

impl ChunkFileFilter {
    /// Java private `ChunkFileFilter(BaseManager, boolean)`.
    fn new(manager: &'static dyn BaseManager, use_all_file_styles: bool) -> ChunkFileFilter {
        ChunkFileFilter {
            base: ExtensionFileFilter::new(Some(manager), true, use_all_file_styles, false),
        }
    }

    /// Java static `getInstance(BaseManager, boolean)`.
    pub fn get_instance(
        manager: &'static dyn BaseManager,
        use_all_file_styles: bool,
    ) -> ChunkFileFilter {
        let mut instance = ChunkFileFilter::new(manager, use_all_file_styles);
        let name = manager.get_name();
        instance.base.setup(
            name.as_deref(),
            Some(AxisType::SingleAxis),
            Some(AxisID::Only),
            None,
            None,
            Some(&[
                Some(&*file_type::CLASS.processchunks_mrc),
                Some(&*file_type::CLASS.processchunks_vol_mrc),
            ]),
        );
        instance
    }
}

impl FileFilter for ChunkFileFilter {
    fn accept(&self, file: &Path) -> bool {
        self.base.accept(Some(file))
    }

    /// Java `getDescription()`.
    fn get_description(&self) -> Option<String> {
        Some("MRC file (.mrc)".to_owned())
    }
}
