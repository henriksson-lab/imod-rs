//! `IMOD/Etomo/src/etomo/storage/DirectiveValue.java`.
#![allow(dead_code)]

/// Java interface `DirectiveValue`.
pub trait DirectiveValue {
    /// Java `getValue`.
    fn get_value(&self) -> Option<String>;

    /// Java `isBatch`.
    fn is_batch(&self) -> bool;

    /// Java `isOverride`.
    fn is_override(&self) -> bool;
}
