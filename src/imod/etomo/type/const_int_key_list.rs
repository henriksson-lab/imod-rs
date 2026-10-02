//! `IMOD/Etomo/src/etomo/type/ConstIntKeyList.java`.

use super::etomo_number::EtomoNumber;
use super::int_key_list::Walker;

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `ConstIntKeyList`.  `getEtomoNumber` is declared `ConstEtomoNumber` in Java;
/// the implementation builds a fresh `EtomoNumber`, which is returned by value.
pub trait ConstIntKeyList {
    /// Java `getFirstKey`.
    fn get_first_key(&self) -> i32;
    /// Java `getLastKey`.
    fn get_last_key(&self) -> i32;
    /// Java `getString`; null is `None`.
    fn get_string(&self, key: i32) -> Option<String>;
    /// Java `getEtomoNumber`; null is `None`.
    fn get_etomo_number(&self, key: i32) -> Option<EtomoNumber>;
    /// Java `containsKey`.
    fn contains_key(&self, key: i32) -> bool;
    /// Java `size`.
    fn size(&self) -> i32;
    /// Java `getWalker`.
    fn get_walker(&self) -> Walker<'_>;
    /// Java `isEmpty`.
    fn is_empty(&self) -> bool;
}
