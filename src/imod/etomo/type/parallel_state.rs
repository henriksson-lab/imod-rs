//! `IMOD/Etomo/src/etomo/type/ParallelState.java`.
//!
//! The state (`.epp` "State" group) of the anisotropic diffusion interface: the K
//! values and iterations of the last tests run.  `ParallelState extends BaseState`.
//! Shared by the manager (EDT) and its process manager (process threads), so each
//! field sits in a `Mutex`; getters of value objects return a copy.

use std::collections::BTreeMap;
use std::sync::Mutex;

use super::axis_id::AxisID;
use super::base_state::BaseState;
use super::const_etomo_number::{ConstEtomoNumber, Type};
use super::etomo_number::EtomoNumber;
use super::iterator_element_list::IteratorElementList;
use super::parallel_meta_data;
use super::parsed_array::ParsedArray;
use super::parsed_element::ParsedElement;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::storage::storable::Storable;

/// Java private static final `GROUP_KEY`.
const GROUP_KEY: &str = "State";

/// Java `public final class ParallelState extends BaseState`.
pub struct ParallelState {
    /// Java private final `testKValueList`.
    test_k_value_list: Mutex<ParsedArray>,
    /// Java private final `testIteration`.
    test_iteration: Mutex<EtomoNumber>,
    /// Java private final `testKValue`.
    test_k_value: Mutex<EtomoNumber>,
    /// Java private final `testIterationList`.  IterationList may contain array
    /// descriptors in the form start-end.  Example: "2,4 - 9,10".
    test_iteration_list: Mutex<IteratorElementList>,
}

impl ParallelState {
    /// Java `ParallelState(BaseManager, AxisID)`.
    pub fn new(manager: Option<&'static dyn BaseManager>, axis_id: AxisID) -> ParallelState {
        let mut test_k_value_list =
            ParsedArray::get_instance(Some(Type::Double), Some("TestKValueList"), None);
        test_k_value_list.set_backward_compatible_null_key();
        ParallelState {
            test_k_value_list: Mutex::new(test_k_value_list),
            test_iteration: Mutex::new(EtomoNumber::new_with_name("TestIteration")),
            test_k_value: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Double,
                "TestKValue",
            )),
            test_iteration_list: Mutex::new(IteratorElementList::new(
                manager,
                Some(axis_id),
                Some("TestIterationList"),
            )),
        }
    }

    /// Java `setTestKValueList(String)`.
    pub fn set_test_k_value_list(&self, input: Option<&str>) {
        self.test_k_value_list
            .lock()
            .unwrap()
            .set_raw_string_string(input);
    }

    /// Java `getTestKValueList()`.
    pub fn get_test_k_value_list(&self) -> ParsedArray {
        self.test_k_value_list.lock().unwrap().clone()
    }

    /// Java `setTestIteration(int)`.
    pub fn set_test_iteration(&self, input: i32) {
        self.test_iteration.lock().unwrap().set_int(input);
    }

    /// Java `getTestIteration()`.
    pub fn get_test_iteration(&self) -> ConstEtomoNumber {
        (**self.test_iteration.lock().unwrap()).clone()
    }

    /// Java `setTestIterationList(IteratorElementList)`.
    pub fn set_test_iteration_list(&self, input: Option<&IteratorElementList>) {
        self.test_iteration_list.lock().unwrap().set_list(input);
    }

    /// Java `getTestIterationList()`.
    pub fn get_test_iteration_list(&self) -> IteratorElementList {
        self.test_iteration_list.lock().unwrap().clone()
    }

    /// Java `setTestKValue(double)`.
    pub fn set_test_k_value(&self, input: f64) {
        self.test_k_value.lock().unwrap().set_double(input);
    }

    /// Java `getTestKValue()`.
    pub fn get_test_k_value(&self) -> ConstEtomoNumber {
        (**self.test_k_value.lock().unwrap()).clone()
    }
}

impl Storable for ParallelState {
    /// Java `store(Properties)`.
    fn store(&self, props: &mut BTreeMap<String, String>) {
        self.store_with_prepend(props, "");
    }

    /// Java `store(Properties, String)`.  (`super.store(props, prepend)` is a no-op;
    /// see `base_state.rs`.)
    fn store_with_prepend(&self, props: &mut BTreeMap<String, String>, prepend: &str) {
        let prepend = self.create_prepend(prepend);
        self.test_k_value_list
            .lock()
            .unwrap()
            .store(props, Some(&prepend));
        self.test_iteration
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.test_iteration_list
            .lock()
            .unwrap()
            .store(Some(props), Some(&prepend));
        self.test_k_value
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
    }

    /// Java `load(Properties)`.
    fn load(&self, props: &mut BTreeMap<String, String>) {
        self.load_with_prepend(props, "");
    }

    /// Java `load(Properties, String)`.  (`super.load(props, prepend)` is a no-op.)
    fn load_with_prepend(&self, props: &mut BTreeMap<String, String>, prepend: &str) {
        let prepend = self.create_prepend(prepend);
        self.test_k_value_list
            .lock()
            .unwrap()
            .load(props, Some(&prepend));
        self.test_iteration
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.test_iteration_list
            .lock()
            .unwrap()
            .load(Some(props), Some(&prepend));
        self.test_k_value
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
    }
}

impl BaseState for ParallelState {
    /// Java package-private `createPrepend(String)`.
    fn create_prepend(&self, prepend: &str) -> String {
        if prepend.is_empty() {
            return format!(
                "{GROUP_KEY}.{}",
                parallel_meta_data::ANISOTROPIC_DIFFUSION_GROUP_KEY
            );
        }
        format!(
            "{prepend}.{GROUP_KEY}.{}",
            parallel_meta_data::ANISOTROPIC_DIFFUSION_GROUP_KEY
        )
    }
}
