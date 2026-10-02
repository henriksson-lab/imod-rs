//! `IMOD/Etomo/src/etomo/ui/TextStateExtension.java`.
//!
//! Backup and checkpoint state for a text field.  An EDT object: built as `Rc<Self>` by
//! the field it extends, and every method takes `&self`.
//!
//! The field owns this extension and this extension refers back to the field, so the
//! back-reference is a `Weak` (Java's collector handles the cycle).  A call after the
//! field is gone does nothing, where Java could not reach this object at all.

use std::cell::RefCell;
use std::rc::{Rc, Weak};

use super::field_setting_interface::FieldSettingInterface;
use super::field_type::FieldType;
use super::swing::text_efield_interface::TextEfieldInterface;
use super::text_field_setting::TextFieldSetting;

/// Java `TextStateExtension`.
pub struct TextStateExtension {
    /// Java private final `field`.
    field: Weak<dyn TextEfieldInterface>,
    /// Java private `backup`, initialised to null.
    backup: RefCell<Option<TextFieldSetting>>,
    /// Java private `checkpoint`, initialised to null.
    checkpoint: RefCell<Option<TextFieldSetting>>,
}

impl TextStateExtension {
    /// Java `TextStateExtension(TextEfieldInterface)`.
    pub fn new(field: Rc<dyn TextEfieldInterface>) -> Rc<TextStateExtension> {
        Rc::new(TextStateExtension {
            field: Rc::downgrade(&field),
            backup: RefCell::new(None),
            checkpoint: RefCell::new(None),
        })
    }

    /// Java `backup()`.
    pub fn backup(&self) {
        if self.backup.borrow().is_none() {
            *self.backup.borrow_mut() = Some(TextFieldSetting::new_field_type(FieldType::String));
        }
        let Some(field) = self.field.upgrade() else {
            return;
        };
        let text = field.get_text();
        self.backup
            .borrow_mut()
            .as_mut()
            .unwrap()
            .set_string(text.as_deref());
    }

    /// Java `checkpoint()`.
    pub fn checkpoint(&self) {
        if self.checkpoint.borrow().is_none() {
            *self.checkpoint.borrow_mut() =
                Some(TextFieldSetting::new_field_type(FieldType::String));
        }
        let Some(field) = self.field.upgrade() else {
            return;
        };
        let text = field.get_text();
        self.checkpoint
            .borrow_mut()
            .as_mut()
            .unwrap()
            .set_string(text.as_deref());
    }

    /// Java `isCheckpointed()`.
    pub fn is_checkpointed(&self) -> bool {
        self.checkpoint
            .borrow()
            .as_ref()
            .is_some_and(|checkpoint| checkpoint.is_set())
    }

    /// Java `restoreFromBackup()`.  If the field was backed up, make the backup value
    /// the displayed value, and turn off the back up.
    pub fn restore_from_backup(&self) {
        let value = {
            let backup = self.backup.borrow();
            match backup.as_ref() {
                Some(backup) if backup.is_set() => Some(backup.get_value()),
                _ => None,
            }
        };
        if let Some(value) = value {
            let Some(field) = self.field.upgrade() else {
                return;
            };
            field.set_text(value.as_deref());
            if let Some(backup) = self.backup.borrow_mut().as_mut() {
                backup.reset();
            }
        }
    }

    /// Java `isDifferentFromCheckpoint(boolean)`.  `always_check` - when false return
    /// false when the field is disabled or invisible.  Returns true if text field is
    /// different from checkpoint.
    pub fn is_different_from_checkpoint(&self, always_check: bool) -> bool {
        let Some(field) = self.field.upgrade() else {
            return false;
        };
        if !always_check && (!field.is_enabled() || !field.is_visible()) {
            return false;
        }
        let text = field.get_text();
        let checkpoint = self.checkpoint.borrow();
        checkpoint.is_none() || !checkpoint.as_ref().unwrap().equals_string(text.as_deref())
    }
}
