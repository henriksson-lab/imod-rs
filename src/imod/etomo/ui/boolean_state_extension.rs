//! `IMOD/Etomo/src/etomo/ui/BooleanStateExtension.java`.
//!
//! Checkpoint state for a toggle field.  An EDT object: built as `Rc<Self>` by the
//! field it extends, and every method takes `&self`.
//!
//! The field owns this extension and this extension refers back to the field, so the
//! back-reference is a `Weak` (Java's collector handles the cycle).  The field always
//! outlives its own extension; a call after the field is gone does nothing, where Java
//! could not reach this object at all.

use std::cell::RefCell;
use std::rc::{Rc, Weak};

use super::boolean_efield_interface::BooleanEfieldInterface;
use super::boolean_field_setting::BooleanFieldSetting;

/// Java `BooleanStateExtension`.
pub struct BooleanStateExtension {
    /// Java private final `field`.
    field: Weak<dyn BooleanEfieldInterface>,
    /// Java private `checkpoint`, initialised to null.
    checkpoint: RefCell<Option<BooleanFieldSetting>>,
}

impl BooleanStateExtension {
    /// Java `BooleanStateExtension(BooleanEfieldInterface)`.
    pub fn new(field: Rc<dyn BooleanEfieldInterface>) -> Rc<BooleanStateExtension> {
        Rc::new(BooleanStateExtension {
            field: Rc::downgrade(&field),
            checkpoint: RefCell::new(None),
        })
    }

    /// Java `checkpoint()`.
    pub fn checkpoint(&self) {
        if self.checkpoint.borrow().is_none() {
            *self.checkpoint.borrow_mut() = Some(BooleanFieldSetting::new());
        }
        let Some(field) = self.field.upgrade() else {
            return;
        };
        let selected = field.is_selected();
        self.checkpoint
            .borrow_mut()
            .as_mut()
            .unwrap()
            .set_boolean(selected);
    }

    /// Java final `isCheckpointValue()`: the checkpoint value.
    pub fn is_checkpoint_value(&self) -> bool {
        let checkpoint = self.checkpoint.borrow();
        let Some(checkpoint) = checkpoint.as_ref() else {
            return false;
        };
        checkpoint.is_value()
    }
}
