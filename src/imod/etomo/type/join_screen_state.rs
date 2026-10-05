//! `IMOD/Etomo/src/etomo/type/JoinScreenState.java`.
//!
//! The Join interface's screen state: the "Refine with trial" check box and the
//! boundary table's xfjointomo results (best gap, mean and max error per boundary),
//! stored under `ScreenState`.
//!
//! **Representation.**  `JoinScreenState extends BaseScreenState`: the superclass is
//! the `base` field, reached through `Deref` (as in `recon_screen_state.rs`).  The
//! object is owned by `JoinManager` and stored from process threads, so each field
//! carries its own lock and every method takes `&self`.

use std::collections::BTreeMap;
use std::sync::Mutex;

use super::axis_id::AxisID;
use super::axis_type::AxisType;
use super::base_screen_state::BaseScreenState;
use super::const_etomo_number::ConstEtomoNumber;
use super::const_int_key_list::ConstIntKeyList;
use super::etomo_boolean2::EtomoBoolean2;
use super::int_key_list::IntKeyList;
use crate::imod::etomo::storage::storable::Storable;

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `public class JoinScreenState extends BaseScreenState`.
pub struct JoinScreenState {
    /// Java superclass `BaseScreenState` state.
    pub base: BaseScreenState,
    /// Java private final `refineWithTrial = new EtomoBoolean2("RefineWithTrial")`.
    refine_with_trial: Mutex<EtomoBoolean2>,
    /// Java private final `bestGapList = IntKeyList.getStringInstance("BestGap")`.
    best_gap_list: Mutex<IntKeyList>,
    /// Java private final `meanErrorList = IntKeyList.getStringInstance("MeanError")`.
    mean_error_list: Mutex<IntKeyList>,
    /// Java private final `maxErrorList = IntKeyList.getStringInstance("MaxError")`.
    max_error_list: Mutex<IntKeyList>,
}

/// Java inheritance: every `BaseScreenState` member is reachable on a
/// `JoinScreenState`.
impl std::ops::Deref for JoinScreenState {
    type Target = BaseScreenState;

    fn deref(&self) -> &BaseScreenState {
        &self.base
    }
}

impl JoinScreenState {
    /// Java `JoinScreenState(AxisID, AxisType)`.
    pub fn new(axis_id: AxisID, axis_type: AxisType) -> JoinScreenState {
        JoinScreenState {
            base: BaseScreenState::new(axis_id, axis_type),
            refine_with_trial: Mutex::new(EtomoBoolean2::new_with_name("RefineWithTrial")),
            best_gap_list: Mutex::new(IntKeyList::get_string_instance_with_key("BestGap")),
            mean_error_list: Mutex::new(IntKeyList::get_string_instance_with_key("MeanError")),
            max_error_list: Mutex::new(IntKeyList::get_string_instance_with_key("MaxError")),
        }
    }

    /// Java `store(Properties, String)`.
    pub fn store_with_prepend(&self, props: &mut BTreeMap<String, String>, prepend: &str) {
        self.base.store_with_prepend(props, prepend);
        let prepend = self.base.get_prepend(prepend);
        self.refine_with_trial
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.best_gap_list.lock().unwrap().store(props, &prepend);
        self.mean_error_list.lock().unwrap().store(props, &prepend);
        self.max_error_list.lock().unwrap().store(props, &prepend);
    }

    /// Java `load(Properties, String)`.
    pub fn load_with_prepend(&self, props: &mut BTreeMap<String, String>, prepend: &str) {
        self.base.load_with_prepend(props, prepend);
        let prepend = self.base.get_prepend(prepend);
        self.refine_with_trial
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.best_gap_list.lock().unwrap().load(props, &prepend);
        self.mean_error_list.lock().unwrap().load(props, &prepend);
        self.max_error_list.lock().unwrap().load(props, &prepend);
    }

    /// Java `getRefineWithTrial()`: a copy of the field.
    pub fn get_refine_with_trial(&self) -> ConstEtomoNumber {
        let guard = self.refine_with_trial.lock().unwrap();
        let number: &ConstEtomoNumber = &guard;
        number.clone()
    }

    /// Java `setRefineWithTrial(boolean)`.
    pub fn set_refine_with_trial(&self, refine_with_trial: bool) {
        self.refine_with_trial
            .lock()
            .unwrap()
            .set_boolean(refine_with_trial);
    }

    /// Java `resetBestGap()`.
    pub fn reset_best_gap(&self) {
        self.best_gap_list.lock().unwrap().reset();
    }

    /// Java `setBestGap(int, String)`.
    pub fn set_best_gap(&self, key: i32, best_gap: Option<&str>) {
        self.best_gap_list.lock().unwrap().put_string(key, best_gap);
    }

    /// Java `getBestGap(int)`.
    pub fn get_best_gap(&self, key: i32) -> Option<String> {
        self.best_gap_list.lock().unwrap().get_string(key)
    }

    /// Java `resetMeanError()`.
    pub fn reset_mean_error(&self) {
        self.mean_error_list.lock().unwrap().reset();
    }

    /// Java `setMeanError(int, String)`.
    pub fn set_mean_error(&self, key: i32, mean_error: Option<&str>) {
        self.mean_error_list
            .lock()
            .unwrap()
            .put_string(key, mean_error);
    }

    /// Java `getMeanError(int)`.  The source's `meanErrorList == null` test is on a
    /// final field that is never null.
    pub fn get_mean_error(&self, key: i32) -> Option<String> {
        self.mean_error_list.lock().unwrap().get_string(key)
    }

    /// Java `resetMaxError()`.
    pub fn reset_max_error(&self) {
        self.max_error_list.lock().unwrap().reset();
    }

    /// Java `setMaxError(int, String)`.
    pub fn set_max_error(&self, key: i32, max_error: Option<&str>) {
        self.max_error_list
            .lock()
            .unwrap()
            .put_string(key, max_error);
    }

    /// Java `getMaxError(int)`.
    pub fn get_max_error(&self, key: i32) -> Option<String> {
        self.max_error_list.lock().unwrap().get_string(key)
    }
}

/// Java `Storable`: `store(Properties)`/`load(Properties)` are `BaseScreenState`'s,
/// which dispatch to this class's two-argument forms with `""`.
impl Storable for JoinScreenState {
    fn store(&self, properties: &mut BTreeMap<String, String>) {
        self.store_with_prepend(properties, "");
    }

    fn store_with_prepend(&self, properties: &mut BTreeMap<String, String>, prepend: &str) {
        JoinScreenState::store_with_prepend(self, properties, prepend);
    }

    fn load(&self, properties: &mut BTreeMap<String, String>) {
        self.load_with_prepend(properties, "");
    }

    fn load_with_prepend(&self, properties: &mut BTreeMap<String, String>, prepend: &str) {
        JoinScreenState::load_with_prepend(self, properties, prepend);
    }
}
