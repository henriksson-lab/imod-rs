//! `IMOD/Etomo/src/etomo/type/SerialSectionsState.java`.
//!
//! The state (`.ess` "SerialSectionsState" group) of the Serial Sections interface:
//! whether the edge functions are invalid and the preblend settings the last
//! successful preblend ran with.  `SerialSectionsState extends BaseState`.  Shared by
//! the manager (EDT) and its process manager (process threads), so each field sits in
//! a `Mutex`.

use std::collections::BTreeMap;
use std::sync::Mutex;

use super::base_state::BaseState;
use super::const_etomo_number::{ConstEtomoNumber, Number, Type};
use super::const_string_property::ConstStringProperty;
use super::etomo_number::EtomoNumber;
use super::etomo_state::{self, EtomoState};
use super::string_property::StringProperty;
use crate::imod::etomo::storage::storable::Storable;

/// Java `rcsid`.
pub const RCSID: &str = "$Id:$";

/// Java private static final `groupString`.
const GROUP_STRING: &str = "SerialSectionsState";

/// Java `public class SerialSectionsState extends BaseState`.
pub struct SerialSectionsState {
    /// Java package-private `invalidEdgeFunctions`.
    invalid_edge_functions: Mutex<EtomoState>,
    /// Java private final `preblendRobustFitting`.
    preblend_robust_fitting: Mutex<EtomoNumber>,
    /// Java private final `preblendFixIntensityFromEdges`.
    preblend_fix_intensity_from_edges: Mutex<EtomoNumber>,
    /// Java private final `preblendSumPiecesForGradient`.
    preblend_sum_pieces_for_gradient: Mutex<EtomoNumber>,
    /// Java private final `preblendOtherSumGradientFile`.
    preblend_other_sum_gradient_file: Mutex<StringProperty>,
}

impl Default for SerialSectionsState {
    fn default() -> Self {
        Self::new()
    }
}

impl SerialSectionsState {
    /// Java implicit `SerialSectionsState()`.
    pub fn new() -> SerialSectionsState {
        SerialSectionsState {
            invalid_edge_functions: Mutex::new(EtomoState::new_with_name("InvalidEdgeFunctions")),
            preblend_robust_fitting: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Double,
                "Preblend.RobustFitting",
            )),
            preblend_fix_intensity_from_edges: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Integer,
                "Preblend.FixIntensityFromEdges",
            )),
            preblend_sum_pieces_for_gradient: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Integer,
                "Preblend.SumPiecesForGradient",
            )),
            preblend_other_sum_gradient_file: Mutex::new(StringProperty::new_with_key(Some(
                "Preblend.OtherSumGradientFile",
            ))),
        }
    }

    /// Java `initialize()`.
    pub fn initialize(&self) {
        self.invalid_edge_functions
            .lock()
            .unwrap()
            .set_int(etomo_state::NO_RESULT_VALUE);
    }

    /// Java `setInvalidEdgeFunctions(boolean)`.  Returns a copy of the number (Java
    /// returns the field itself as a `ConstEtomoNumber`).
    pub fn set_invalid_edge_functions(&self, invalid_edge_functions: bool) -> ConstEtomoNumber {
        let mut field = self.invalid_edge_functions.lock().unwrap();
        field.set_boolean(invalid_edge_functions);
        (***field).clone()
    }

    /// Java `getInvalidEdgeFunctions()`, a copy of the `EtomoState`.
    pub fn get_invalid_edge_functions(&self) -> EtomoState {
        self.invalid_edge_functions.lock().unwrap().clone()
    }

    /// Java `setPreblendRobustFitting(double)`.
    pub fn set_preblend_robust_fitting(&self, value: f64) {
        self.preblend_robust_fitting
            .lock()
            .unwrap()
            .set_double(value);
    }

    /// Java `equalsPreblendRobustFitting(boolean, String)`.
    pub fn equals_preblend_robust_fitting(&self, set: bool, value: Option<&str>) -> bool {
        let field = self.preblend_robust_fitting.lock().unwrap();
        if !set && field.is_null() {
            return true;
        }
        set && field.equals_string(value)
    }

    /// Java `setPreblendFixIntensityFromEdges(int)`.
    pub fn set_preblend_fix_intensity_from_edges(&self, value: i32) {
        self.preblend_fix_intensity_from_edges
            .lock()
            .unwrap()
            .set_int(value);
    }

    /// Java `equalsPreblendFixIntensityFromEdges(boolean, Integer)`: Java resolves
    /// `equals(value)` with an `Integer` to `ConstEtomoNumber.equals(Number)`.
    pub fn equals_preblend_fix_intensity_from_edges(&self, set: bool, value: Option<i32>) -> bool {
        let field = self.preblend_fix_intensity_from_edges.lock().unwrap();
        if !set && field.is_null() {
            return true;
        }
        set && field.equals_number(value.map(Number::Integer))
    }

    /// Java `setPreblendSumPiecesForGradient(int)`.
    pub fn set_preblend_sum_pieces_for_gradient(&self, value: i32) {
        self.preblend_sum_pieces_for_gradient
            .lock()
            .unwrap()
            .set_int(value);
    }

    /// Java `equalsPreblendSumPiecesForGradient(boolean, Integer)`.
    pub fn equals_preblend_sum_pieces_for_gradient(&self, set: bool, value: Option<i32>) -> bool {
        let field = self.preblend_sum_pieces_for_gradient.lock().unwrap();
        if !set && field.is_null() {
            return true;
        }
        set && field.equals_number(value.map(Number::Integer))
    }

    /// Java `setPreblendOtherSumGradientFile(String)`.
    pub fn set_preblend_other_sum_gradient_file(&self, value: Option<&str>) {
        self.preblend_other_sum_gradient_file
            .lock()
            .unwrap()
            .set(value);
    }

    /// Java `equalsPreblendOtherSumGradientFile(boolean, String)`.
    pub fn equals_preblend_other_sum_gradient_file(&self, set: bool, value: Option<&str>) -> bool {
        let field = self.preblend_other_sum_gradient_file.lock().unwrap();
        if !set && field.is_empty() {
            return true;
        }
        set && field.equals(value)
    }
}

impl Storable for SerialSectionsState {
    /// Java `store(Properties)`.
    fn store(&self, props: &mut BTreeMap<String, String>) {
        self.store_with_prepend(props, "");
    }

    /// Java `store(Properties, String)`.  (`super.store(props, prepend)` is a no-op;
    /// see `base_state.rs`.)
    fn store_with_prepend(&self, props: &mut BTreeMap<String, String>, prepend: &str) {
        let prepend = self.create_prepend(prepend);
        // String group = prepend + "."; (unused)
        self.invalid_edge_functions
            .lock()
            .unwrap()
            .store_with_prepend(props, &prepend);
        self.preblend_robust_fitting
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.preblend_fix_intensity_from_edges
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.preblend_sum_pieces_for_gradient
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.preblend_other_sum_gradient_file
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), Some(&prepend));
    }

    /// Java `load(Properties)`.
    fn load(&self, props: &mut BTreeMap<String, String>) {
        self.load_with_prepend(props, "");
    }

    /// Java `load(Properties, String)`.  (`super.load(props, prepend)` is a no-op.)
    ///
    /// Upstream bug fixed in translation (SerialSectionsState.java:68-82): `store`
    /// writes the fields under `createPrepend(prepend)` ("SerialSectionsState.") but
    /// `load` reads them under the bare `prepend`, so a reopened dataset never gets
    /// its state back.  Here `load` reads under `createPrepend(prepend)`, where
    /// `store` wrote them (`BUGS.md`, "Serial sections").
    fn load_with_prepend(&self, props: &mut BTreeMap<String, String>, prepend: &str) {
        let prepend = self.create_prepend(prepend);
        let prepend = prepend.as_str();
        // reset
        self.invalid_edge_functions.lock().unwrap().reset();
        self.preblend_robust_fitting.lock().unwrap().reset();
        self.preblend_fix_intensity_from_edges
            .lock()
            .unwrap()
            .reset();
        self.preblend_sum_pieces_for_gradient
            .lock()
            .unwrap()
            .reset();
        self.preblend_other_sum_gradient_file
            .lock()
            .unwrap()
            .reset();
        // load
        self.invalid_edge_functions
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(prepend));
        self.preblend_robust_fitting
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(prepend));
        self.preblend_fix_intensity_from_edges
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(prepend));
        self.preblend_sum_pieces_for_gradient
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(prepend));
        // `StringProperty.load` may remove a backward-compatible key from the Java
        // `Properties`; this one declares none, so it loads from a copy.
        let mut props_copy = props.clone();
        self.preblend_other_sum_gradient_file
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), Some(prepend));
    }
}

impl BaseState for SerialSectionsState {
    /// Java package-private `createPrepend(String)`.  Java compares the strings with
    /// `==`; every caller passes the literal `""` or a built string, and `""` literals
    /// are interned, so an empty prepend is the `==` case.
    fn create_prepend(&self, prepend: &str) -> String {
        if prepend.is_empty() {
            return GROUP_STRING.to_string();
        }
        format!("{prepend}.{GROUP_STRING}")
    }
}
