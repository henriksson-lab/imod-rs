//! `IMOD/Etomo/src/etomo/ui/swing/ControlMode.java`.
//!
//! This source unit supplies the names used by the Swing control mediator.
//! It has no rendering of its own; callers which act on a mode remain at the
//! explicit GUI-control boundary.
#![allow(dead_code)]

use std::sync::LazyLock;

use crate::imod::etomo::util::utilities;

/// Java `ControlMode`.
///
/// `field_name` is deliberately optional because the Java constructor accepts
/// a nullable `String`, and `appendToName` preserves that distinction.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ControlMode {
    field_name: Option<String>,
}

impl ControlMode {
    /// Java package-private `ControlMode(String)`.
    pub(crate) fn new(field_name: Option<String>) -> Self {
        Self { field_name }
    }

    /// Java `getFieldName()`.
    pub(crate) fn get_field_name(&self) -> Option<&str> {
        self.field_name.as_deref()
    }

    /// Java `toString()`.
    ///
    /// The Java implementation returns its nullable field directly, so this
    /// retains `None` rather than inventing an empty display string.
    pub fn to_string(&self) -> Option<&str> {
        self.field_name.as_deref()
    }

    /// Java `hasFieldName()`.
    pub fn has_field_name(&self) -> bool {
        self.field_name.is_some()
    }

    /// Java `appendToName(String)`.
    pub fn append_to_name(&self, name: Option<String>) -> Option<String> {
        if name.is_none() {
            return self.field_name.clone();
        }
        if self.field_name.is_none() {
            return name;
        }
        Some(format!(
            "{}{}{}",
            name.as_deref().unwrap_or_default(),
            utilities::NAME_SEPARATOR,
            self.field_name.as_deref().unwrap_or_default()
        ))
    }
}

/// Java public static `ControlMode.CLEAR`.
pub static CLEAR: LazyLock<ControlMode> =
    LazyLock::new(|| ControlMode::new(Some("clear".to_owned())));

/// Java public static `ControlMode.SELECT_FILE`.
pub static SELECT_FILE: LazyLock<ControlMode> =
    LazyLock::new(|| ControlMode::new(Some(format!("select{}file", utilities::NAME_SEPARATOR))));

/// Java package-private static `ControlMode.SELECT_MULTIPLE_FILES`.
pub(crate) static SELECT_MULTIPLE_FILES: LazyLock<ControlMode> = LazyLock::new(|| {
    ControlMode::new(Some(format!(
        "select multiple{}files",
        utilities::NAME_SEPARATOR
    )))
});

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn static_modes_keep_the_java_field_names() {
        assert_eq!(CLEAR.get_field_name(), Some("clear"));
        assert_eq!(SELECT_FILE.to_string(), Some("select-file"));
        assert_eq!(
            SELECT_MULTIPLE_FILES.to_string(),
            Some("select multiple-files")
        );
        assert!(CLEAR.has_field_name());
    }

    #[test]
    fn append_to_name_follows_both_null_branches() {
        assert_eq!(CLEAR.append_to_name(None), Some("clear".to_owned()));
        assert_eq!(
            SELECT_FILE.append_to_name(Some("button".to_owned())),
            Some("button-select-file".to_owned())
        );

        let null_mode = ControlMode::new(None);
        assert_eq!(null_mode.append_to_name(None), None);
        assert_eq!(
            null_mode.append_to_name(Some("button".to_owned())),
            Some("button".to_owned())
        );
        assert!(!null_mode.has_field_name());
        assert_eq!(null_mode.to_string(), None);
    }
}
