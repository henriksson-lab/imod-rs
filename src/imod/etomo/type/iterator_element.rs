//! `IMOD/Etomo/src/etomo/type/IteratorElement.java`.
//!
//! One element of an iterator list ("2" or "4 - 9"): one or two numbers.  Immutable.

use super::etomo_number::EtomoNumber;

/// Java `public static final String rcsid`.
pub const RCSID: &str = "$Id$";

/// Java package-private `final class IteratorElement`.
#[derive(Clone, Debug)]
pub struct IteratorElement {
    /// Java private final `first = new EtomoNumber()`.
    first: EtomoNumber,
    /// Java private final `second`; null for a single number.
    second: Option<EtomoNumber>,
}

impl IteratorElement {
    /// Java `IteratorElement(String)`.
    pub fn new_string(first: Option<&str>) -> IteratorElement {
        let mut instance = IteratorElement {
            first: EtomoNumber::new(),
            second: None,
        };
        instance.first.set_string(first);
        instance
    }

    /// Java `IteratorElement(String, String)`.
    pub fn new_string_string(first: Option<&str>, second: Option<&str>) -> IteratorElement {
        let mut instance = IteratorElement {
            first: EtomoNumber::new(),
            second: Some(EtomoNumber::new()),
        };
        instance.first.set_string(first);
        instance.second.as_mut().unwrap().set_string(second);
        instance
    }

    /// Java package-private `isRange()`.
    pub fn is_range(&self) -> bool {
        !self.first.is_null() && self.second.as_ref().is_some_and(|second| !second.is_null())
    }

    /// Java package-private `getNumber()`.  Returns a String version of the first
    /// number or, if it is null, the second number.  If both numbers are null, returns
    /// null.
    pub fn get_number(&self) -> Option<String> {
        if !self.first.is_null() {
            Some(self.first.to_string())
        } else if let Some(second) = self.second.as_ref().filter(|second| !second.is_null()) {
            Some(second.to_string())
        } else {
            None
        }
    }

    /// Java package-private `getRange()`.  Returns String values of a range from
    /// first to second.
    pub fn get_range(&self) -> Vec<String> {
        let mut range = Vec::new();
        if !self.is_range()
            || self
                .first
                .equals_const_etomo_number(self.second.as_ref().map(|second| &**second))
        {
            if let Some(number) = self.get_number() {
                range.push(number);
            }
        } else {
            let second = self.second.as_ref().unwrap();
            range.push(self.first.to_string());
            if self.first.lt_const_etomo_number(Some(second)) {
                let mut i = self.first.get_int() + 1;
                while i < second.get_int() {
                    range.push(i.to_string());
                    i += 1;
                }
            } else {
                let mut i = self.first.get_int() - 1;
                while i > second.get_int() {
                    range.push(i.to_string());
                    i -= 1;
                }
            }
            range.push(second.to_string());
        }
        range
    }
}

/// Java `toString()`.  Convert the one or two numbers to strings and return.
impl std::fmt::Display for IteratorElement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let second = self.second.as_ref().filter(|second| !second.is_null());
        if !self.first.is_null() {
            match second {
                Some(second) => write!(f, "{} - {}", self.first, second),
                None => write!(f, "{}", self.first),
            }
        } else if let Some(second) = second {
            write!(f, "{second}")
        } else {
            Ok(())
        }
    }
}
