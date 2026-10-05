//! `IMOD/Etomo/src/etomo/plugin/demo/DemoProcessName.java`.
//!
//! Extension to the `ProcessName` class.  This class should contain instances of
//! `ProcessName` rather then instances of `DemoProcessName`.  Java creates the two
//! instances when the class is initialized (`ProcessName.constructInstance` adds them to
//! `ProcessName`'s instance map); here they are created on first use, which is before
//! anything can look them up by name (`EtomoPluginDemoParam` and the demo plugin are the
//! only users, and they reach them through these statics).  The private constructor
//! (`DemoProcessName()`, only there to allow the inheritance) has no counterpart.

use std::sync::LazyLock;

use crate::imod::etomo::r#type::process_name::ProcessName;

/// Java `public static final ProcessName ETOMO_PLUGIN_DEMO`: script name.
pub static ETOMO_PLUGIN_DEMO: LazyLock<ProcessName> =
    LazyLock::new(|| ProcessName::construct_instance("etomoPluginDemo"));

/// Java `public static final ProcessName DEMO`: comscript name.
pub static DEMO: LazyLock<ProcessName> = LazyLock::new(|| ProcessName::construct_instance("demo"));
