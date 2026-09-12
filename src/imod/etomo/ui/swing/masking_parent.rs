//! `IMOD/Etomo/src/etomo/ui/swing/MaskingParent.java`.
#![allow(dead_code)]

use super::file_text_field_interface::FileTextFieldInterface;

/// Java `MaskingParent`.
pub trait MaskingParent {
    /// Java `isReferenceFileSelected`.
    fn is_reference_file_selected(&self) -> bool;

    /// Java `getVolumeTableSize`.
    fn get_volume_table_size(&self) -> i32;

    /// Java `fixIncorrectPath(FileTextFieldInterface, boolean)`.
    fn fix_incorrect_path(
        &mut self,
        file_text_field: &mut dyn FileTextFieldInterface,
        choose_path: bool,
    ) -> bool;

    /// Java `updateDisplay(boolean)`.
    fn update_display(&mut self, init: bool);
}
