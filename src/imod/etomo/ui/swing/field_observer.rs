//! `IMOD/Etomo/src/etomo/ui/swing/FieldObserver.java`.
#![allow(dead_code)]

/// Java deprecated `FieldObserver`.
#[deprecated(note = "Java FieldObserver was deprecated on 2018-08-04")]
pub trait FieldObserver {
    /// Java `msgFieldChanged(boolean)`.
    #[deprecated(note = "Java FieldObserver was deprecated on 2018-08-04")]
    fn msg_field_changed(&mut self, different_from_checkpoint: bool);
}
