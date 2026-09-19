//! `IMOD/Etomo/src/etomo/comscript/ProcessDetails.java`.

use super::field_interface::FieldInterface;
use crate::imod::etomo::r#type::const_etomo_number::ConstEtomoNumber;

/// Java `ProcessDetails`.  The values retain their distinct source shapes;
/// implementations decide which field enums they recognize and return `None`
/// for an unavailable optional parameter.
pub trait ProcessDetails {
    fn get_int_value(&self, field: &dyn FieldInterface) -> Option<i32>;
    fn get_boolean_value(&self, field: &dyn FieldInterface) -> Option<bool>;
    fn get_double_value(&self, field: &dyn FieldInterface) -> Option<f64>;
    fn get_hashtable(&self, field: &dyn FieldInterface) -> Option<Vec<(String, String)>>;
    fn get_etomo_number(&self, field: &dyn FieldInterface) -> Option<ConstEtomoNumber>;
    fn get_int_key_list(&self, field: &dyn FieldInterface) -> Option<Vec<(i32, String)>>;
    fn get_string(&self, field: &dyn FieldInterface) -> Option<String>;
    fn get_string_array(&self, field: &dyn FieldInterface) -> Option<Vec<String>>;
    fn get_iterator_element_list(&self, field: &dyn FieldInterface) -> Option<Vec<i32>>;
}
