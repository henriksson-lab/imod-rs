//! `IMOD/Etomo/src/etomo/type/SampleType.java`.
//!
//! A typesafe enum whose `value` is an `EtomoNumber`.  The three singletons are Rust
//! variants; `value` is built on demand, as `view_type.rs` does for its `index`.

use super::const_etomo_number::ConstEtomoNumber;
use super::enumerated_type::EnumeratedType;
use super::etomo_number::EtomoNumber;

/// Java `SampleType`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SampleType {
    /// Java `NONE = new SampleType(0)`.
    None,
    /// Java `PLASTIC_SECTION = new SampleType(1)`.
    PlasticSection,
    /// Java `CRYO = new SampleType(2)`.
    Cryo,
}

impl SampleType {
    /// Java private static `DEFAULT`.
    const DEFAULT: SampleType = SampleType::PlasticSection;

    /// Java field `value`, a `private final EtomoNumber` the constructor fills in with
    /// `this.value.set(value)`.
    fn value(self) -> EtomoNumber {
        let mut value = EtomoNumber::new();
        value.set_int(match self {
            Self::None => 0,
            Self::PlasticSection => 1,
            Self::Cryo => 2,
        });
        value
    }

    /// Java `getInstance(ConstEtomoNumber, boolean)`.
    pub fn get_instance(input: Option<&ConstEtomoNumber>, use_default: bool) -> Option<SampleType> {
        let input = match input {
            None => {
                if use_default {
                    return Some(Self::DEFAULT);
                }
                return None;
            }
            Some(input) => input,
        };
        if input.equals_const_etomo_number(Some(&Self::None.value().base)) {
            return Some(Self::None);
        }
        if input.equals_const_etomo_number(Some(&Self::PlasticSection.value().base)) {
            return Some(Self::PlasticSection);
        }
        if input.equals_const_etomo_number(Some(&Self::Cryo.value().base)) {
            return Some(Self::Cryo);
        }
        None
    }

    /// Java `getInstance(String)`.
    pub fn get_instance_from_string(value: Option<&str>) -> Option<SampleType> {
        if Self::None.value().equals_string(value) {
            return Some(Self::None);
        }
        if Self::PlasticSection.value().equals_string(value) {
            return Some(Self::PlasticSection);
        }
        if Self::Cryo.value().equals_string(value) {
            return Some(Self::Cryo);
        }
        None
    }
}

impl EnumeratedType for SampleType {
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
impl std::fmt::Display for SampleType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.value())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values_and_lookup() {
        assert_eq!(SampleType::Cryo.to_string(), "2");
        assert!(SampleType::PlasticSection.is_default());
        let mut number = EtomoNumber::new();
        number.set_int(0);
        assert_eq!(
            SampleType::get_instance(Some(&number.base), false),
            Some(SampleType::None)
        );
        assert_eq!(
            SampleType::get_instance(None, true),
            Some(SampleType::PlasticSection)
        );
        assert_eq!(SampleType::get_instance(None, false), None);
        assert_eq!(
            SampleType::get_instance_from_string(Some("2")),
            Some(SampleType::Cryo)
        );
        assert_eq!(SampleType::get_instance_from_string(Some("7")), None);
    }
}
