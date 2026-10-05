//! `IMOD/Etomo/src/etomo/type/SlicerAngles.java`.
//!
//! The three rotation angles 3dmod's slicer reports for a join section, as
//! `JoinManager.imodGetSlicerAngles` collects them and `JoinState` stores them per
//! section row.
//!
//! **`prepend == ""`.**  `createPrepend` tests `prepend == ""`, a reference comparison
//! true for the interned literal; the callers pass a prepend built by
//! `SectionTableRowData.createPrepend`, so it is translated as `prepend.is_empty()`.

use std::collections::BTreeMap;

use super::const_etomo_number::{ConstEtomoNumber, Type};
use super::etomo_number::EtomoNumber;

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java private static `NAME`.
const NAME: &str = "SlicerAngles";

/// Java `public final class SlicerAngles`.
#[derive(Clone, Debug)]
pub struct SlicerAngles {
    /// Java private final `x = new EtomoNumber(EtomoNumber.Type.DOUBLE, "X")`.
    x: EtomoNumber,
    /// Java private final `y = new EtomoNumber(EtomoNumber.Type.DOUBLE, "Y")`.
    y: EtomoNumber,
    /// Java private final `z = new EtomoNumber(EtomoNumber.Type.DOUBLE, "Z")`.
    z: EtomoNumber,
}

impl Default for SlicerAngles {
    fn default() -> SlicerAngles {
        SlicerAngles::new()
    }
}

impl SlicerAngles {
    /// Java implicit constructor `SlicerAngles()`.
    pub fn new() -> SlicerAngles {
        SlicerAngles {
            x: EtomoNumber::new_with_type_and_name(Type::Double, "X"),
            y: EtomoNumber::new_with_type_and_name(Type::Double, "Y"),
            z: EtomoNumber::new_with_type_and_name(Type::Double, "Z"),
        }
    }

    /// Java `isComplete()`.
    pub fn is_complete(&self) -> bool {
        !self.x.is_null() && !self.y.is_null() && !self.z.is_null()
    }

    /// Java `add(String) throws NumberFormatException`.  `EtomoNumber.set(String)`
    /// records an unparsable value as invalid instead of throwing, so the declared
    /// exception is never raised (the caller's `catch` is dead in the source too).
    pub fn add(&mut self, angle: Option<&str>) {
        if self.is_complete() {
            return;
        }
        if self.x.is_null() {
            self.x.set_string(angle);
        } else if self.y.is_null() {
            self.y.set_string(angle);
        } else {
            self.z.set_string(angle);
        }
    }

    /// Java `store(Properties, String)`.
    pub fn store(&self, props: &mut BTreeMap<String, String>, prepend: &str) {
        let prepend = SlicerAngles::create_prepend(prepend);
        self.x.store_with_prepend(props, Some(&prepend));
        self.y.store_with_prepend(props, Some(&prepend));
        self.z.store_with_prepend(props, Some(&prepend));
    }

    /// Java `load(Properties, String)`.
    pub fn load(&mut self, props: &BTreeMap<String, String>, prepend: &str) {
        let prepend = SlicerAngles::create_prepend(prepend);
        self.x.load_with_prepend(props, Some(&prepend));
        self.y.load_with_prepend(props, Some(&prepend));
        self.z.load_with_prepend(props, Some(&prepend));
    }

    /// Java package-private static `createPrepend(String)`.
    pub(crate) fn create_prepend(prepend: &str) -> String {
        if prepend.is_empty() {
            return NAME.to_string();
        }
        format!("{}.{}", prepend, NAME)
    }

    /// Java `isEmpty()`.
    pub fn is_empty(&self) -> bool {
        self.x.is_null() && self.y.is_null() && self.z.is_null()
    }

    /// Java `setX(ConstEtomoNumber)`.
    pub fn set_x(&mut self, x: Option<&ConstEtomoNumber>) {
        self.x.set_const_etomo_number(x);
    }

    /// Java `setY(ConstEtomoNumber)`.
    pub fn set_y(&mut self, y: Option<&ConstEtomoNumber>) {
        self.y.set_const_etomo_number(y);
    }

    /// Java `setZ(ConstEtomoNumber)`.
    pub fn set_z(&mut self, z: Option<&ConstEtomoNumber>) {
        self.z.set_const_etomo_number(z);
    }

    /// Java `getX()`.
    pub fn get_x(&self) -> &ConstEtomoNumber {
        &self.x
    }

    /// Java `getY()`.
    pub fn get_y(&self) -> &ConstEtomoNumber {
        &self.y
    }

    /// Java `getZ()`.
    pub fn get_z(&self) -> &ConstEtomoNumber {
        &self.z
    }
}

/// Java `toString()`.
impl std::fmt::Display for SlicerAngles {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "x={},y={},z={}", self.x, self.y, self.z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_fills_x_then_y_then_z_and_stores_under_the_name() {
        let mut angles = SlicerAngles::new();
        assert!(angles.is_empty());
        angles.add(Some("1.5"));
        angles.add(Some("-2"));
        assert!(!angles.is_complete());
        angles.add(Some("3"));
        assert!(angles.is_complete());
        let mut props = BTreeMap::new();
        angles.store(&mut props, "JoinState.SectionTableRow.1");
        assert_eq!(
            props.get("JoinState.SectionTableRow.1.SlicerAngles.X"),
            Some(&"1.5".to_string())
        );
        let mut loaded = SlicerAngles::new();
        loaded.load(&props, "JoinState.SectionTableRow.1");
        assert_eq!(loaded.to_string(), angles.to_string());
    }
}
