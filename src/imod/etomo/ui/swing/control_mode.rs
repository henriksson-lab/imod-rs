//! `IMOD/Etomo/src/etomo/ui/swing/ControlMode.java`.
//!
//! Types of control modes.  Currently there is only one mode with state
//! (OVERRIDE).  If a second stateful mode is added, ControlTargets will need to
//! be changed to handle this.
//!
//! Java compares modes by identity (`mode == ControlMode.CLEAR`); the instances
//! are the process-lifetime statics below, so a Rust caller holds
//! `&'static ControlMode` and compares with `std::ptr::eq`.  `ControlState`
//! extends this class; its instances embed a `ControlMode` (see
//! `control_state.rs`).

use std::fmt;
use std::sync::LazyLock;

use crate::imod::etomo::util::utilities;

/// Java `public static ControlMode CLEAR = new ControlMode("clear")`.
pub static CLEAR: LazyLock<ControlMode> = LazyLock::new(|| ControlMode::new(Some("clear")));

/// Java `public static ControlMode SELECT_FILE =
/// new ControlMode("select" + Utilities.NAME_SEPARATOR + "file")`.
pub static SELECT_FILE: LazyLock<ControlMode> = LazyLock::new(|| {
    ControlMode::new(Some(&format!(
        "select{}file",
        utilities::NAME_SEPARATOR
    )))
});

/// Java package-private `static ControlMode SELECT_MULTIPLE_FILES =
/// new ControlMode("select multiple" + Utilities.NAME_SEPARATOR + "files")`.
pub static SELECT_MULTIPLE_FILES: LazyLock<ControlMode> = LazyLock::new(|| {
    ControlMode::new(Some(&format!(
        "select multiple{}files",
        utilities::NAME_SEPARATOR
    )))
});

/// Java `public class ControlMode`.
#[derive(Debug)]
pub struct ControlMode {
    /// Java `private final String fieldName`.
    field_name: Option<String>,
}

impl ControlMode {
    /// Java package-private `ControlMode(String)`.
    pub fn new(field_name: Option<&str>) -> ControlMode {
        ControlMode {
            field_name: field_name.map(str::to_owned),
        }
    }

    /// Java package-private `getFieldName()`.
    pub fn get_field_name(&self) -> Option<&str> {
        self.field_name.as_deref()
    }

    /// Java `hasFieldName()`.
    pub fn has_field_name(&self) -> bool {
        self.field_name.is_some()
    }

    /// Java `appendToName(String)`.
    pub fn append_to_name(&self, name: Option<&str>) -> Option<String> {
        let Some(name) = name else {
            return self.field_name.clone();
        };
        let Some(field_name) = &self.field_name else {
            return Some(name.to_owned());
        };
        Some(format!(
            "{}{}{}",
            name,
            utilities::NAME_SEPARATOR,
            field_name
        ))
    }
}

/// Java `toString()`: returns `fieldName`.  Java returns `null` for a null
/// field name, which string concatenation renders as `"null"`.
impl fmt::Display for ControlMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.field_name.as_deref().unwrap_or("null"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn statics_and_append_to_name() {
        assert_eq!(CLEAR.get_field_name(), Some("clear"));
        assert_eq!(SELECT_FILE.to_string(), "select-file");
        assert_eq!(SELECT_MULTIPLE_FILES.to_string(), "select multiple-files");
        assert_eq!(
            SELECT_FILE.append_to_name(Some("bn")),
            Some("bn-select-file".to_owned())
        );
        let null_mode = ControlMode::new(None);
        assert_eq!(null_mode.append_to_name(None), None);
        assert_eq!(null_mode.append_to_name(Some("x")), Some("x".to_owned()));
    }
}
