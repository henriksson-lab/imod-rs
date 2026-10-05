//! `IMOD/Etomo/src/etomo/ui/swing/ReferenceParent.java`.
//!
//! The implementor (`PeetDialog`) is an event dispatch thread object reached through
//! `Rc`, so the methods take `&self`.

use super::file_text_field_interface::FileTextFieldInterface;

/// Java package-private `interface ReferenceParent`.
pub trait ReferenceParent {
    /// Java `fixIncorrectPath(FileTextFieldInterface, boolean)`.
    fn fix_incorrect_path(&self, file_text_field: &dyn FileTextFieldInterface, choose_path: bool)
    -> bool;

    /// Java `getVolumeTableSize()`.
    fn get_volume_table_size(&self) -> i32;

    /// Java `updateDisplay(boolean)`.
    fn update_display(&self, init: bool);

    /// Java `isFlgVolNamesAreTemplates()`.
    fn is_flg_vol_names_are_templates(&self) -> bool;
}
