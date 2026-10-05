//! `IMOD/Etomo/src/etomo/comscript/ProcessDetails.java`.

use super::field_interface::FieldInterface;
use crate::imod::etomo::r#type::const_etomo_number::ConstEtomoNumber;
use crate::imod::etomo::r#type::int_key_list::IntKeyList;
use crate::imod::etomo::r#type::slicer_angles::SlicerAngles;

/// Java raw `Hashtable` returned by `getHashtable`.  The only implementors that return
/// one (`MakejoincomParam`, `StartJoinParam`) return their rotation-angle tables:
/// `SlicerAngles` keyed by section row index.
pub type Hashtable = std::collections::HashMap<i32, SlicerAngles>;

/// Java `ProcessDetails`.  The values retain their distinct source shapes;
/// implementations decide which field enums they recognize and return `None`
/// for an unavailable optional parameter.
pub trait ProcessDetails {
    fn get_int_value(&self, field: &dyn FieldInterface) -> Option<i32>;
    fn get_boolean_value(&self, field: &dyn FieldInterface) -> Option<bool>;
    fn get_double_value(&self, field: &dyn FieldInterface) -> Option<f64>;
    fn get_hashtable(&self, field: &dyn FieldInterface) -> Option<Hashtable>;
    fn get_etomo_number(&self, field: &dyn FieldInterface) -> Option<ConstEtomoNumber>;
    /// Java `getIntKeyList`: a copy of the `ConstIntKeyList`.
    fn get_int_key_list(&self, field: &dyn FieldInterface) -> Option<IntKeyList>;
    fn get_string(&self, field: &dyn FieldInterface) -> Option<String>;
    fn get_string_array(&self, field: &dyn FieldInterface) -> Option<Vec<String>>;
    fn get_iterator_element_list(&self, field: &dyn FieldInterface) -> Option<crate::imod::etomo::r#type::iterator_element_list::IteratorElementList>;
}
