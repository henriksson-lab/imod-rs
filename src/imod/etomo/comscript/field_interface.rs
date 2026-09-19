//! `IMOD/Etomo/src/etomo/comscript/FieldInterface.java`.

/// Java marker interface implemented by parameter-class inner field enums.
/// The trait deliberately has no string conversion requirement: Java uses the
/// field object's identity/type, not a display label, to select a value.
pub trait FieldInterface {}
