//! `IMOD/Etomo/src/etomo/storage/DirectiveDescr.java`.

use crate::imod::etomo::storage::directive_descr_choice_list::DirectiveDescrChoiceList;
use crate::imod::etomo::storage::directive_descr_etomo_column::DirectiveDescrEtomoColumn;
use crate::imod::etomo::storage::directive_value_type::DirectiveValueType;

/// Java `rcsid`.
pub const RCSID: &str = "$Id:$";

/// Java interface `DirectiveDescr`.
pub trait DirectiveDescr {
    /// Java `getName()`.
    fn get_name(&self) -> Option<String>;

    /// Java `getDescription()`.
    fn get_description(&self) -> Option<String>;

    /// Java `getValueType()`.
    fn get_value_type(&self) -> Option<DirectiveValueType>;

    /// Java `isBatch()`.
    fn is_batch(&self) -> bool;

    /// Java `isTemplate()`.
    fn is_template(&self) -> bool;

    /// Java `getEtomoColumn()`.
    fn get_etomo_column(&self) -> Option<DirectiveDescrEtomoColumn>;

    /// Java `getLabel()`.
    fn get_label(&self) -> Option<String>;

    /// Java `getChoiceList()`.
    fn get_choice_list(&self) -> Option<DirectiveDescrChoiceList>;
}
