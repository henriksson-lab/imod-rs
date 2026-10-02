//! `IMOD/Etomo/src/etomo/ui/FlagType.java`.
//!
//! Types of flags.  A Java class with four static instances compared by identity: the
//! instances are `pub static` items here, handed around as `&'static FlagType` and
//! compared with `std::ptr::eq`.  (`ERROR` and `TEMPLATE_ERROR` carry the same colour
//! and background, so only their addresses tell them apart, as in Java.)

use super::swing::colors;
use super::swing::process_control_panel;

/// Java `FlagType`.
#[derive(Debug)]
pub struct FlagType {
    /// Java public final `color`.
    pub color: (u8, u8, u8),
    /// Java public final `background`.
    pub background: bool,
}

/// Java `TEMPLATE`.
pub static TEMPLATE: FlagType = FlagType::new(colors::FIELD_HIGHLIGHT, false);
/// Java `TEMPLATE_ERROR`.
pub static TEMPLATE_ERROR: FlagType =
    FlagType::new(process_control_panel::COLOR_NOT_STARTED, false);
/// Java `ERROR`.
pub static ERROR: FlagType = FlagType::new(process_control_panel::COLOR_NOT_STARTED, false);
/// Java `WARNING`.
pub static WARNING: FlagType = FlagType::new(colors::WARNING_BACKGROUND, true);

impl FlagType {
    /// Java private `FlagType(Color, boolean)`.
    const fn new(color: (u8, u8, u8), background: bool) -> FlagType {
        FlagType { color, background }
    }

    /// Java `isTemplate()`.
    pub fn is_template(&self) -> bool {
        std::ptr::eq(self, &TEMPLATE)
    }

    /// Java `isError()`.
    pub fn is_error(&self) -> bool {
        std::ptr::eq(self, &ERROR) || std::ptr::eq(self, &TEMPLATE_ERROR)
    }
}

/// Java `toString()`.
impl std::fmt::Display for FlagType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if std::ptr::eq(self, &TEMPLATE) {
            return f.write_str("TEMPLATE");
        }
        if std::ptr::eq(self, &TEMPLATE_ERROR) {
            return f.write_str("TEMPLATE_ERROR");
        }
        if std::ptr::eq(self, &ERROR) {
            return f.write_str("ERROR");
        }
        if std::ptr::eq(self, &WARNING) {
            return f.write_str("WARNING");
        }
        // `super.toString()`: `Object.toString()`, the class name and identity hash.
        write!(
            f,
            "etomo.ui.FlagType@{:x}",
            self as *const FlagType as usize
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instances_are_told_apart_by_identity() {
        assert!(ERROR.is_error() && TEMPLATE_ERROR.is_error());
        assert!(!WARNING.is_error() && TEMPLATE.is_template());
        assert_eq!(TEMPLATE_ERROR.to_string(), "TEMPLATE_ERROR");
        assert_eq!(ERROR.to_string(), "ERROR");
    }
}
