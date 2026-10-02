//! `IMOD/Etomo/src/etomo/type/EraseGold.java`.
//!
//! A typesafe enum whose `value` is an `EtomoNumber`, mirrored as a Rust enum with the
//! value built on demand.

use super::const_etomo_number::ConstEtomoNumber;
use super::enumerated_type::EnumeratedType;
use super::etomo_number::EtomoNumber;

/// Java `EraseGold`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EraseGold {
    /// Java `FID = new EraseGold(1)`.
    Fid,
    /// Java `FIND_3D = new EraseGold(2)`.
    Find3d,
}

impl EraseGold {
    /// Java private static `DEFAULT`.
    const DEFAULT: EraseGold = EraseGold::Find3d;

    /// Java field `value`, filled in by the constructor with `this.value.set(value)`.
    fn value(self) -> EtomoNumber {
        let mut value = EtomoNumber::new();
        value.set_int(match self {
            Self::Fid => 1,
            Self::Find3d => 2,
        });
        value
    }

    /// Java `getInstance(String)`.
    pub fn get_instance(value: Option<&str>) -> Option<EraseGold> {
        if Self::Fid.value().equals_string(value) {
            return Some(Self::Fid);
        }
        if Self::Find3d.value().equals_string(value) {
            return Some(Self::Find3d);
        }
        None
    }
}

impl EnumeratedType for EraseGold {
    /// Java `isDefault`.
    fn is_default(&self) -> bool {
        *self == Self::DEFAULT
    }

    /// Java `getValue`.
    fn get_value(&self) -> ConstEtomoNumber {
        self.value().base
    }

    /// Java `getLabel`.
    fn get_label(&self) -> Option<String> {
        None
    }
}

/// Java `toString`: `value.toString()`.
impl std::fmt::Display for EraseGold {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.value())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values_and_lookup() {
        assert_eq!(EraseGold::Fid.to_string(), "1");
        assert!(EraseGold::Find3d.is_default());
        assert_eq!(EraseGold::get_instance(Some("2")), Some(EraseGold::Find3d));
        assert_eq!(EraseGold::get_instance(Some("3")), None);
    }
}
