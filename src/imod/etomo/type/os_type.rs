//! `IMOD/Etomo/src/etomo/type/OSType.java`.
//!
//! Class representing the type of OS.  Can either get the OS of the current system or
//! a stored OS.
//!
//! Copyright: Copyright 2008
//!
//! Organization: Boulder Laboratory for 3-Dimensional Electron Microscopy of Cells
//! (BL3DEMC), University of Colorado

use std::collections::BTreeMap;

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `KEY`.
pub const KEY: &str = "OSType";

/// Java `OSType`, a typesafe enum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OSType {
    /// Java `LINUX`.
    Linux,
    /// Java `WINDOWS`.
    Windows,
    /// Java `MAC`.
    Mac,
}

/// Java `DEFAULT`.
pub const DEFAULT: OSType = OSType::Linux;

impl OSType {
    /// Java private field `description`.
    fn description(self) -> &'static str {
        match self {
            OSType::Linux => "Linux",
            OSType::Windows => "Windows",
            OSType::Mac => "Mac",
        }
    }

    /// Java static `getInstance()`.
    ///
    /// `System.getProperty("os.name")` is a JVM property; the Rust target's
    /// `std::env::consts::OS` ("linux", "windows", "macos", ...) stands in for it and
    /// is searched for the same lower-case substrings.
    pub fn get_instance() -> OSType {
        let os_name = std::env::consts::OS.to_lowercase();
        if os_name.contains("windows") {
            return OSType::Windows;
        }
        if os_name.contains("mac") {
            return OSType::Mac;
        }
        OSType::Linux
    }

    /// Java static `getInstance(Properties, String)`.
    pub fn get_instance_from_props(props: &BTreeMap<String, String>, prepend: &str) -> OSType {
        let description = props.get(&format!("{}.{}", prepend, KEY));
        let description = match description {
            // `description.matches("\\*")`: the whole value is a single '*'.
            Some(description) if description != "*" => description,
            _ => return DEFAULT,
        };
        if description == OSType::Linux.description() {
            return OSType::Linux;
        }
        if description == OSType::Windows.description() {
            return OSType::Windows;
        }
        if description == OSType::Mac.description() {
            return OSType::Mac;
        }
        DEFAULT
    }

    /// Java `store(Properties, String)`.
    pub fn store(self, props: &mut BTreeMap<String, String>, prepend: &str) {
        props.insert(
            format!("{}.{}", prepend, KEY),
            self.description().to_string(),
        );
    }
}

/// Java `toString`.
impl std::fmt::Display for OSType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.description())
    }
}
