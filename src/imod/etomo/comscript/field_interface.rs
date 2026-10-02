//! `IMOD/Etomo/src/etomo/comscript/FieldInterface.java`.

/// Java marker interface implemented by parameter-class inner field enums.
/// The trait deliberately has no string conversion requirement: Java uses the
/// field object's identity/type, not a display label, to select a value.
pub trait FieldInterface: std::any::Any {}

/// The field as its concrete enum, for a `ProcessDetails` implementation
/// dispatching on the field (Java `field == Fields.X`).
pub fn as_field<F: FieldInterface>(field: &dyn FieldInterface) -> Option<&F> {
    (field as &dyn std::any::Any).downcast_ref::<F>()
}
