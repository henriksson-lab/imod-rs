//! `IMOD/Etomo/src/etomo/plugin/demo/DemoFileType.java`.
//!
//! Extension to the `FileType` class.  This class should contain instances of
//! `FileType` rather then instances of `DemoFileType`.  The instances are created on
//! first use (Java: at class initialization); the private constructor has no
//! counterpart.

use std::sync::{Arc, LazyLock};

use super::demo_process_name;
use crate::imod::etomo::r#type::file_type::FileType;

/// Java package-private static final `DEMO_COMSCRIPT`.
pub static DEMO_COMSCRIPT: LazyLock<Arc<FileType>> = LazyLock::new(|| {
    FileType::construct_instance_descr(
        false,
        true,
        Some("demo"),
        Some(".com"),
        Some("DEMOSETUP_COMSCRIPT"),
    )
});

/// Java package-private static final `ETOMO_DEMO_PLUGIN_AUTODOC`.
pub static ETOMO_DEMO_PLUGIN_AUTODOC: LazyLock<Arc<FileType>> = LazyLock::new(|| {
    FileType::construct_imod_dir_instance(
        false,
        false,
        Some(&demo_process_name::ETOMO_PLUGIN_DEMO.to_string()),
        Some(".adoc"),
        Some("Plugins"),
    )
});
