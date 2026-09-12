//! `IMOD/Etomo/src/etomo/ui/DirectiveDisplaySettings.java`.
#![allow(dead_code)]

/// Java `DirectiveDisplaySettings`.
pub trait DirectiveDisplaySettings {
    fn is_include(&self, index: i32) -> bool;
    fn is_exclude(&self, index: i32) -> bool;
    fn is_show_unchanged(&self) -> bool;
    fn is_show_hidden(&self) -> bool;
    fn is_show_only_included(&self) -> bool;
}
