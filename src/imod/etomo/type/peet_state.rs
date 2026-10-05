//! `IMOD/Etomo/src/etomo/type/PeetState.java`.
//!
//! Stores the states of variables used in processes after the processes are
//! finished.  `PeetState extends BaseState`; shared by the manager (event dispatch
//! thread) and its process manager (process threads), so each field sits in a
//! `Mutex`.

use std::collections::BTreeMap;
use std::sync::Mutex;

use super::base_state::BaseState;
use super::etomo_number::EtomoNumber;
use super::etomo_version::EtomoVersion;
use super::int_key_list::IntKeyList;
use crate::imod::etomo::storage::storable::{Storable, StorableValue};

/// Java `public static final String rcsid`.
pub const RCSID: &str = "$Id$";

/// Java private static final `KEY`.
const KEY: &str = "PeetState";
/// Java private static final `CURRENT_VERSION`.
const CURRENT_VERSION: &str = "1.1";
/// Java private static final `PARSER_KEY`.
const PARSER_KEY: &str = "Parser";
/// Java private static final `ITERATION_LIST_SIZE_KEY`.
const ITERATION_LIST_SIZE_KEY: &str = "IterationListSize";
/// Java private static final `LST_THRESHOLDS_KEY`.
const LST_THRESHOLDS_KEY: &str = "LstThresholds";

/// Java `public final class PeetState extends BaseState`.
pub struct PeetState {
    /// Java private final `parserIterationListSize` (deprecated; backward
    /// compatibility for version 1.0).
    parser_iteration_list_size: Mutex<EtomoNumber>,
    /// Java private final `parserLstThresholdsArray` (deprecated).
    parser_lst_thresholds_array: Mutex<IntKeyList>,
    /// Java private final `iterationListSize`.
    iteration_list_size: Mutex<EtomoNumber>,
    /// Java private final `version`.
    version: Mutex<EtomoVersion>,
}

impl PeetState {
    /// Java `PeetState()` with its field initialisers.
    pub fn new() -> PeetState {
        PeetState {
            parser_iteration_list_size: Mutex::new(EtomoNumber::new_with_name(&format!(
                "{PARSER_KEY}.{ITERATION_LIST_SIZE_KEY}"
            ))),
            parser_lst_thresholds_array: Mutex::new(IntKeyList::get_string_instance_with_key(
                &format!("{PARSER_KEY}.{LST_THRESHOLDS_KEY}"),
            )),
            iteration_list_size: Mutex::new(EtomoNumber::new_with_name(ITERATION_LIST_SIZE_KEY)),
            version: Mutex::new(EtomoVersion::get_empty_instance(Some("Version"))),
        }
    }

    /// Java `setIterationListSize(int)`.
    pub fn set_iteration_list_size(&self, input: i32) {
        self.iteration_list_size.lock().unwrap().set_int(input);
    }

    /// Java `getIterationListSize()`.
    pub fn get_iteration_list_size(&self) -> i32 {
        self.iteration_list_size.lock().unwrap().get_int()
    }

    /// Java `store(Properties, String)`.  (`super.store` is a no-op.)
    pub fn store_with_prepend(&self, props: &mut BTreeMap<String, String>, prepend: &str) {
        let prepend = self.create_prepend(prepend);
        let mut version = self.version.lock().unwrap();
        version.set(Some(CURRENT_VERSION));
        version.store_with_prepend(props, &prepend);
        drop(version);
        self.iteration_list_size
            .lock()
            .unwrap()
            .base
            .store_with_prepend(props, Some(&prepend));
    }

    /// Java `load(Properties, String)`.  (`super.load` is a no-op.)
    pub fn load_with_prepend(&self, props: &mut BTreeMap<String, String>, prepend: &str) {
        // reset
        self.parser_iteration_list_size.lock().unwrap().reset();
        self.parser_lst_thresholds_array.lock().unwrap().reset();
        self.iteration_list_size.lock().unwrap().reset();
        self.version.lock().unwrap().reset();
        // load
        let prepend = self.create_prepend(prepend);
        self.version
            .lock()
            .unwrap()
            .load_with_prepend(props, &prepend);
        // backward compatibility for version 1.0
        let le_1_0 = self.version.lock().unwrap().le(Some(
            &EtomoVersion::get_default_instance_with_version(Some("1.0")),
        ));
        if le_1_0 {
            self.parser_iteration_list_size
                .lock()
                .unwrap()
                .load_with_prepend(props, Some(&prepend));
            let parser = (**self.parser_iteration_list_size.lock().unwrap()).clone();
            self.iteration_list_size
                .lock()
                .unwrap()
                .set_const_etomo_number(Some(&parser));
            self.parser_lst_thresholds_array
                .lock()
                .unwrap()
                .load(props, &prepend);
        } else {
            self.iteration_list_size
                .lock()
                .unwrap()
                .load_with_prepend(props, Some(&prepend));
        }
    }
}

impl Default for PeetState {
    fn default() -> PeetState {
        PeetState::new()
    }
}

impl Storable for PeetState {
    /// Java `store(Properties)`.
    fn store(&self, props: &mut BTreeMap<String, String>) {
        PeetState::store_with_prepend(self, props, "");
    }

    fn store_with_prepend(&self, props: &mut BTreeMap<String, String>, prepend: &str) {
        PeetState::store_with_prepend(self, props, prepend);
    }

    /// Java `load(Properties)`.
    fn load(&self, props: &mut BTreeMap<String, String>) {
        PeetState::load_with_prepend(self, props, "");
    }

    fn load_with_prepend(&self, props: &mut BTreeMap<String, String>, prepend: &str) {
        PeetState::load_with_prepend(self, props, prepend);
    }
}

impl BaseState for PeetState {
    /// Java package-private `createPrepend(String)`.
    fn create_prepend(&self, prepend: &str) -> String {
        if prepend.is_empty() {
            return KEY.to_owned();
        }
        format!("{prepend}.{KEY}")
    }
}
