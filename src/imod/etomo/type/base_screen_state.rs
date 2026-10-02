//! `IMOD/Etomo/src/etomo/type/BaseScreenState.java`.
//!
//! Per-axis screen state (button states and the parallel-processing panel header),
//! stored under `ScreenState<A|B|>`.
//!
//! Held by the managers and read from process threads as well as the event dispatch
//! thread, so each mutable field carries its own lock and every method takes `&self`.
//!
//! **`localProperties`.**  Java's `load` keeps a reference to the caller's
//! `Properties` object, so later `getButtonState` calls read, and `setButtonState`
//! calls write, the live data-file properties.  Here `load` keeps a copy of the map:
//! `getButtonState` reads what was loaded, and `setButtonState` writes into the copy,
//! from which `store` copies every key that `getButtonState` has registered - which is
//! how the source's own `store` publishes button states.  A button state set but never
//! read back through `getButtonState` is therefore not stored here, where Java would
//! already have written it into the shared object.
//!
//! **`prepend == ""`.**  `getPrepend` tests `prepend == ""`, a reference comparison
//! true for the interned literal that `store(Properties)`/`load(Properties)` pass; it is
//! translated as `prepend.is_empty()`.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Mutex;

use super::axis_id::AxisID;
use super::axis_type::AxisType;
use super::etomo_state::EtomoState;
use super::panel_header_state::PanelHeaderState;
use crate::imod::etomo::storage::storable::{Storable, StorableValue};
use crate::imod::etomo::util::utilities;

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";
/// Java `PARALLEL_HEADER_GROUP`.
pub const PARALLEL_HEADER_GROUP: &str = "ParallelProcess.Header";

/// Java `BaseScreenState`.
pub struct BaseScreenState {
    /// Java field `parallelHeaderState`.
    parallel_header_state: PanelHeaderState,
    /// Java field `additionalStorables`, initialised to null.
    additional_storables: Mutex<Option<Vec<Box<dyn Storable + Send>>>>,
    /// Java package-private final field `axisID`.
    pub(crate) axis_id: AxisID,
    /// Java field `axisType`.
    axis_type: AxisType,
    /// Java field `group`.
    group: String,
    /// Java field `localProperties`, initialised to null.  See the module header.
    local_properties: Mutex<Option<BTreeMap<String, String>>>,
    /// Java field `localPrepend`: the prepend used to save to localProperties.
    local_prepend: Mutex<Option<String>>,
    /// Java field `keys` (a raw `HashSet`), initialised to null.  A `BTreeSet` keeps
    /// `store`'s iteration deterministic.
    keys: Mutex<Option<BTreeSet<String>>>,
}

impl BaseScreenState {
    /// Java `BaseScreenState(AxisID, AxisType)`.
    pub fn new(axis_id: AxisID, axis_type: AxisType) -> BaseScreenState {
        let mut axis_id = axis_id;
        if axis_id == AxisID::Only && axis_type == AxisType::DualAxis {
            axis_id = AxisID::First;
        } else if axis_id == AxisID::First && axis_type == AxisType::SingleAxis {
            axis_id = AxisID::Only;
        }
        BaseScreenState {
            parallel_header_state: PanelHeaderState::new(PARALLEL_HEADER_GROUP),
            additional_storables: Mutex::new(None),
            group: format!("ScreenState{}", axis_id.get_extension().to_uppercase()),
            axis_id,
            axis_type,
            local_properties: Mutex::new(None),
            local_prepend: Mutex::new(None),
            keys: Mutex::new(None),
        }
    }

    /// Java `getMemberVariables`.
    pub fn get_member_variables(&self) -> String {
        format!(
            "{}[{}]",
            utilities::get_extension(Some("class etomo.type.BaseScreenState")).unwrap_or_default(),
            self.parallel_header_state.get_member_variables()
        )
    }

    /// Java final `getButtonState(String)`.  Get a button state out of the local
    /// Properties object and return it as a boolean.
    pub fn get_button_state(&self, key: Option<&str>) -> bool {
        self.get_button_state_with_default(key, false)
    }

    /// Java final `getButtonState(String, boolean)`.  If the state is not available,
    /// return defaultState.
    pub fn get_button_state_with_default(&self, key: Option<&str>, default_state: bool) -> bool {
        let key = match key {
            None => return default_state,
            Some(key) => key,
        };
        {
            let mut keys = self.keys.lock().unwrap();
            if keys.is_none() {
                *keys = Some(BTreeSet::new());
            }
            keys.as_mut().unwrap().insert(key.to_string());
        }
        let local_properties = self.local_properties.lock().unwrap();
        let local_properties = match local_properties.as_ref() {
            None => return default_state,
            Some(local_properties) => local_properties,
        };
        let local_prepend = self.local_prepend.lock().unwrap().clone();
        let mut button_state = EtomoState::new_with_name(key);
        button_state.load_with_prepend(local_properties, local_prepend.as_deref());
        if button_state.is_null() {
            return default_state;
        }
        button_state.is()
    }

    /// Java final `setButtonState`.  Sets the button state in a local Properties
    /// object.  Java concatenates the local prepend and `'.'` and the key.
    pub fn set_button_state(&self, key: Option<&str>, state: bool) {
        let key = match key {
            None => return,
            Some(key) => key,
        };
        let mut local_properties = self.local_properties.lock().unwrap();
        if local_properties.is_none() {
            *local_properties = Some(BTreeMap::new());
            *self.local_prepend.lock().unwrap() = Some(self.get_prepend(""));
        }
        let local_prepend = self.local_prepend.lock().unwrap().clone();
        local_properties.as_mut().unwrap().insert(
            format!("{}.{}", local_prepend.as_deref().unwrap_or("null"), key),
            state.to_string(),
        );
    }

    /// Java package-private final `getPrepend`.
    pub(crate) fn get_prepend(&self, prepend: &str) -> String {
        if prepend.is_empty() {
            self.group.clone()
        } else {
            format!("{}.{}", prepend, self.group)
        }
    }

    /// Java `insert`.  Inserts additional items to be stored with items in this class,
    /// under its prepend.
    pub fn insert(&self, input: Option<Box<dyn Storable + Send>>) {
        let input = match input {
            None => return,
            Some(input) => input,
        };
        let mut additional_storables = self.additional_storables.lock().unwrap();
        if additional_storables.is_none() {
            *additional_storables = Some(Vec::new());
        }
        additional_storables.as_mut().unwrap().push(input);
    }

    /// Java `store(Properties)`.
    pub fn store(&self, props: &mut BTreeMap<String, String>) {
        self.store_with_prepend(props, "");
    }

    /// Java `store(Properties, String)`.  Store the values in localProperties in props.
    pub fn store_with_prepend(&self, props: &mut BTreeMap<String, String>, prepend: &str) {
        let prepend = self.get_prepend(prepend);
        self.parallel_header_state
            .store_with_prepend(props, &prepend);
        if let Some(additional_storables) = self.additional_storables.lock().unwrap().as_ref() {
            for storable in additional_storables.iter() {
                storable.store_with_prepend(props, &prepend);
            }
        }
        let keys = self.keys.lock().unwrap();
        let local_properties = self.local_properties.lock().unwrap();
        let (keys, local_properties) = match (keys.as_ref(), local_properties.as_ref()) {
            (Some(keys), Some(local_properties)) => (keys, local_properties),
            // nothing to store
            _ => return,
        };
        // `synchronized (this)`: the three locks held here.
        let local_prepend = self.local_prepend.lock().unwrap().clone();
        for key in keys.iter() {
            // get the value from the local property using the local prepend
            let state = local_properties
                .get(&format!(
                    "{}.{}",
                    local_prepend.as_deref().unwrap_or("null"),
                    key
                ))
                .cloned();
            if let Some(state) = state {
                // store the value in props using the modified prepend parameter
                props.insert(format!("{}.{}", prepend, key), state);
            }
        }
    }

    /// Java `load(Properties)`.
    pub fn load(&self, props: &BTreeMap<String, String>) {
        self.load_with_prepend(props, "");
    }

    /// Java `load(Properties, String)`.  Point localProperties to props (a copy here;
    /// see the module header) and set localPrepend to prepend.
    pub fn load_with_prepend(&self, props: &BTreeMap<String, String>, prepend: &str) {
        let prepend = self.get_prepend(prepend);
        self.parallel_header_state
            .load_with_prepend(props, &prepend);
        *self.local_properties.lock().unwrap() = Some(props.clone());
        *self.local_prepend.lock().unwrap() = Some(prepend.clone());
        if let Some(additional_storables) = self.additional_storables.lock().unwrap().as_mut() {
            for storable in additional_storables.iter_mut() {
                storable.load_with_prepend(props, &prepend);
            }
        }
    }

    /// Java final `getParallelHeaderState`.
    pub fn get_parallel_header_state(&self) -> &PanelHeaderState {
        &self.parallel_header_state
    }
}

/// Java `Storable`.
impl Storable for BaseScreenState {
    fn store(&self, properties: &mut BTreeMap<String, String>) {
        BaseScreenState::store(self, properties);
    }

    fn store_with_prepend(&self, properties: &mut BTreeMap<String, String>, prepend: &str) {
        BaseScreenState::store_with_prepend(self, properties, prepend);
    }

    fn load(&self, properties: &BTreeMap<String, String>) {
        BaseScreenState::load(self, properties);
    }

    fn load_with_prepend(&self, properties: &BTreeMap<String, String>, prepend: &str) {
        BaseScreenState::load_with_prepend(self, properties, prepend);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn button_states_and_header() {
        let state = BaseScreenState::new(AxisID::First, AxisType::DualAxis);
        assert_eq!(
            state.get_member_variables(),
            "BaseScreenState[PanelHeaderState[group:ParallelProcess.Header,openCloseState:null]]"
        );
        state
            .get_parallel_header_state()
            .set_open_close_state(Some("closed"));
        state.set_button_state(Some("Button1"), true);
        assert!(state.get_button_state(Some("Button1")));
        assert!(state.get_button_state_with_default(Some("Button2"), true));
        let mut props = BTreeMap::new();
        state.store(&mut props);
        assert_eq!(props.get("ScreenStateA.Button1").unwrap(), "true");
        assert_eq!(
            props
                .get("ScreenStateA.ParallelProcess.Header.OpenClose")
                .unwrap(),
            "closed"
        );
        assert_eq!(props.len(), 2);

        let loaded = BaseScreenState::new(AxisID::First, AxisType::DualAxis);
        loaded.load(&props);
        assert!(loaded.get_button_state(Some("Button1")));
        assert!(!loaded.get_button_state(Some("Button2")));
        let single = BaseScreenState::new(AxisID::First, AxisType::SingleAxis);
        assert_eq!(single.get_prepend(""), "ScreenState");
    }
}
