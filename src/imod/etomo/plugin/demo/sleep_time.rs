//! `IMOD/Etomo/src/etomo/plugin/demo/SleepTime.java`.
//!
//! Demonstates the use of `EnumeratedType` and radio buttons.  `EnumeratedType`s are
//! actually used for sets of values or string.  The four Java singletons are the enum's
//! variants (`EnumeratedTypeRef` compares them as Java's `==` does).

use crate::imod::etomo::r#type::const_etomo_number::ConstEtomoNumber;
use crate::imod::etomo::r#type::enumerated_type::EnumeratedType;
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;

/// Java `final class SleepTime implements EnumeratedType`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SleepTime {
    /// Java private static final `ONE = new SleepTime(1, "1 sec")`.
    One,
    /// Java static final `TWO = new SleepTime(2, "2 sec")`.
    Two,
    /// Java static final `THREE = new SleepTime(3, "3 sec")`.
    Three,
    /// Java static final `USER_ENTRY = new SleepTime("Sleep time:")` (no value).
    UserEntry,
}

impl SleepTime {
    /// Java static final `DEFAULT = ONE`.
    pub const DEFAULT: SleepTime = SleepTime::One;

    /// Java private final field `value`: the instance's number, null for `USER_ENTRY`.
    pub fn value(self) -> Option<ConstEtomoNumber> {
        let value = match self {
            SleepTime::One => 1,
            SleepTime::Two => 2,
            SleepTime::Three => 3,
            SleepTime::UserEntry => return None,
        };
        let mut number = EtomoNumber::new();
        number.set_int(value);
        Some((*number).clone())
    }

    /// Java private final field `label`.
    fn label(self) -> &'static str {
        match self {
            SleepTime::One => "1 sec",
            SleepTime::Two => "2 sec",
            SleepTime::Three => "3 sec",
            // Not necessary to put the label here - can be placed directly in the radio
            // button.
            SleepTime::UserEntry => "Sleep time:",
        }
    }

    /// Java static `getInstance(ConstEtomoNumber)`.
    pub fn get_instance(value: Option<&ConstEtomoNumber>) -> SleepTime {
        let Some(value) = value else {
            return SleepTime::DEFAULT;
        };
        if value.is_null() || !value.is_valid() {
            return SleepTime::DEFAULT;
        }
        if value.equals_const_etomo_number(SleepTime::One.value().as_ref()) {
            return SleepTime::One;
        }
        if value.equals_const_etomo_number(SleepTime::Two.value().as_ref()) {
            return SleepTime::Two;
        }
        if value.equals_const_etomo_number(SleepTime::Three.value().as_ref()) {
            return SleepTime::Three;
        }
        SleepTime::UserEntry
    }
}

impl EnumeratedType for SleepTime {
    /// Java `isDefault()`.
    fn is_default(&self) -> bool {
        *self == SleepTime::DEFAULT
    }

    /// Java `getValue()`.  `USER_ENTRY`'s Java value is null; the trait returns a number,
    /// so an empty (null) number stands in for it.  [`SleepTime::value`] keeps the
    /// null.
    fn get_value(&self) -> ConstEtomoNumber {
        self.value()
            .unwrap_or_else(|| (*EtomoNumber::new()).clone())
    }

    /// Java `getLabel()`.
    fn get_label(&self) -> Option<String> {
        Some(self.label().to_owned())
    }
}

/// Java `toString()`: the label.
impl std::fmt::Display for SleepTime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}
