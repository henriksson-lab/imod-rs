//! `IMOD/Etomo/src/etomo/ui/swing/MaskingParent.java`.
//!
//! The implementor (`PeetDialog`) is an event dispatch thread object reached through
//! `Rc`, so the methods take `&self`.

use super::file_text_field_interface::FileTextFieldInterface;

/// Java package-private `interface MaskingParent`.
pub trait MaskingParent {
    /// Java `isReferenceFileSelected()`.
    fn is_reference_file_selected(&self) -> bool;

    /// Java `getVolumeTableSize()`.
    fn get_volume_table_size(&self) -> i32;

    /// Java `fixIncorrectPath(FileTextFieldInterface, boolean)`.
    fn fix_incorrect_path(&self, file_text_field: &dyn FileTextFieldInterface, choose_path: bool)
    -> bool;

    /// Java `updateDisplay(boolean)`.
    fn update_display(&self, init: bool);
}
