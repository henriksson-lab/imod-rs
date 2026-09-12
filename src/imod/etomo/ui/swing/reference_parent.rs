//! `IMOD/Etomo/src/etomo/ui/swing/ReferenceParent.java`.
#![allow(dead_code)]

use super::file_text_field_interface::FileTextFieldInterface;

/// Java `ReferenceParent`.
pub trait ReferenceParent {
    /// Java `fixIncorrectPath(FileTextFieldInterface, boolean)`.
    fn fix_incorrect_path(
        &mut self,
        file_text_field: &mut dyn FileTextFieldInterface,
        choose_path: bool,
    ) -> bool;

    /// Java `getVolumeTableSize`.
    fn get_volume_table_size(&self) -> i32;

    /// Java `updateDisplay(boolean)`.
    fn update_display(&mut self, init: bool);

    /// Java `isFlgVolNamesAreTemplates`.
    fn is_flg_vol_names_are_templates(&self) -> bool;
}
