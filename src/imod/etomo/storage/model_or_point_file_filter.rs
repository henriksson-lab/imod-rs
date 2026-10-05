//! `IMOD/Etomo/src/etomo/storage/ModelOrPointFileFilter.java`.
//!
//! `ModelOrPointFileFilter extends ExtensionFileFilter`: the superclass is embedded as
//! `base` and reached through `Deref` (as in `volume_file_filter.rs`).

use std::path::Path;

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::jdk::FileFilter;
use crate::imod::etomo::storage::extension_file_filter::ExtensionFileFilter;
use crate::imod::etomo::r#type::extension;

/// Java `public class ModelOrPointFileFilter extends ExtensionFileFilter`.
pub struct ModelOrPointFileFilter {
    /// Java superclass `ExtensionFileFilter` state.
    pub base: ExtensionFileFilter,
}

impl std::ops::Deref for ModelOrPointFileFilter {
    type Target = ExtensionFileFilter;

    fn deref(&self) -> &ExtensionFileFilter {
        &self.base
    }
}

impl ModelOrPointFileFilter {
    /// Java private `ModelOrPointFileFilter(BaseManager, boolean)`.
    fn new(
        manager: Option<&'static dyn BaseManager>,
        use_all_file_styles: bool,
    ) -> ModelOrPointFileFilter {
        ModelOrPointFileFilter {
            base: ExtensionFileFilter::new(manager, true, use_all_file_styles, false),
        }
    }

    /// Java static `getInstance(BaseManager, boolean)`.
    pub fn get_instance(
        manager: Option<&'static dyn BaseManager>,
        use_all_file_styles: bool,
    ) -> ModelOrPointFileFilter {
        let mut instance = ModelOrPointFileFilter::new(manager, use_all_file_styles);
        instance.base.setup(
            None,
            None,
            None,
            None,
            Some(&[
                &extension::CLASS.r#mod,
                &extension::CLASS.txt,
                &extension::CLASS.pt,
            ]),
            None,
        );
        instance
    }
}

impl FileFilter for ModelOrPointFileFilter {
    fn accept(&self, file: &Path) -> bool {
        self.base.accept(Some(file))
    }

    /// Java `getDescription()`.
    fn get_description(&self) -> Option<String> {
        Some("Model/Point file (.mod, .txt, .pt)".to_owned())
    }
}
