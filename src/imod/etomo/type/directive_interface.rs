//! `IMOD/Etomo/src/etomo/type/DirectiveInterface.java`.
//!
//! Implementations are shared (`Arc`) and mutated through `&self`, so the trait
//! requires `Send + Sync`.

/// Java `DirectiveInterface`.
pub trait DirectiveInterface: Send + Sync {
    /// Java `setValue(boolean)`.
    fn set_value_boolean(&self, value: bool);

    /// Java `setValue(String)`.
    fn set_value_string(&self, value: Option<&str>);

    /// Java `resetValue()`.
    fn reset_value(&self);
}
