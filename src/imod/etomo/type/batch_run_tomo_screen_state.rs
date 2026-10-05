//! `IMOD/Etomo/src/etomo/type/BatchRunTomoScreenState.java`.
//!
//! **Representation.**  `BatchRunTomoScreenState extends BaseScreenState`: the
//! superclass is the `base` field, reached through `Deref`.  Java's constructor
//! `insert`s the three panel header states into the superclass's storable list *and*
//! keeps them as fields; the fields here are `Arc`s whose clones are what the base
//! list holds, so loading the screen state fills the same objects the dialog reads.

use std::collections::BTreeMap;
use std::sync::Arc;

use super::axis_id::AxisID;
use super::axis_type::AxisType;
use super::base_screen_state::BaseScreenState;
use super::panel_header_state::{self, PanelHeaderState};
use crate::imod::etomo::storage::storable::Storable;
use crate::imod::etomo::ui::swing::batch_run_tomo_dialog;

/// Java `public final class BatchRunTomoScreenState extends BaseScreenState`.
pub struct BatchRunTomoScreenState {
    /// Java superclass `BaseScreenState` state.
    pub base: BaseScreenState,
    /// Java private final `datasetHeaderState = new
    /// PanelHeaderState(BatchRunTomoDialog.TABLE_LABEL + "." + PanelHeaderState.KEY)`.
    dataset_header_state: Arc<PanelHeaderState>,
    /// Java private final `runHeaderState = new
    /// PanelHeaderState(BatchRunTomoDialog.TABLE_LABEL + ".Run." + PanelHeaderState.KEY)`.
    run_header_state: Arc<PanelHeaderState>,
    /// Java private final `seriesWatcherHeaderState = new
    /// PanelHeaderState("SeriesWatcher." + PanelHeaderState.KEY)`.
    series_watcher_header_state: Arc<PanelHeaderState>,
}

impl std::ops::Deref for BatchRunTomoScreenState {
    type Target = BaseScreenState;

    fn deref(&self) -> &BaseScreenState {
        &self.base
    }
}

impl BatchRunTomoScreenState {
    /// Java `BatchRunTomoScreenState(AxisID, AxisType)`.
    pub fn new(axis_id: AxisID, axis_type: AxisType) -> BatchRunTomoScreenState {
        let dataset_header_state = Arc::new(PanelHeaderState::new(&format!(
            "{}.{}",
            batch_run_tomo_dialog::TABLE_LABEL,
            panel_header_state::KEY
        )));
        let run_header_state = Arc::new(PanelHeaderState::new(&format!(
            "{}.Run.{}",
            batch_run_tomo_dialog::TABLE_LABEL,
            panel_header_state::KEY
        )));
        let series_watcher_header_state = Arc::new(PanelHeaderState::new(&format!(
            "SeriesWatcher.{}",
            panel_header_state::KEY
        )));
        let base = BaseScreenState::new(axis_id, axis_type);
        base.insert(Some(Box::new(dataset_header_state.clone())));
        base.insert(Some(Box::new(run_header_state.clone())));
        base.insert(Some(Box::new(series_watcher_header_state.clone())));
        BatchRunTomoScreenState {
            base,
            dataset_header_state,
            run_header_state,
            series_watcher_header_state,
        }
    }

    /// Java `getMemberVariables()`.
    pub fn get_member_variables(&self) -> String {
        format!(
            "{}BatchRunTomoScreenState[{}]",
            self.base.get_member_variables(),
            self.dataset_header_state.get_member_variables()
        )
    }

    /// Java `getDatasetHeaderState()`.
    pub fn get_dataset_header_state(&self) -> &PanelHeaderState {
        &self.dataset_header_state
    }

    /// Java `getRunHeaderState()`.
    pub fn get_run_header_state(&self) -> &PanelHeaderState {
        &self.run_header_state
    }

    /// Java `getSeriesWatcherHeaderState()`.
    pub fn get_series_watcher_header_state(&self) -> &PanelHeaderState {
        &self.series_watcher_header_state
    }

    /// Java `load(Properties, String)`.
    pub fn load_with_prepend(&self, props: &mut BTreeMap<String, String>, prepend: &str) {
        self.base.load_with_prepend(props, prepend);
        let _prepend = self.base.get_prepend(prepend);
    }

    /// Java `store(Properties, String)`.
    pub fn store_with_prepend(&self, props: &mut BTreeMap<String, String>, prepend: &str) {
        self.base.store_with_prepend(props, prepend);
        let _prepend = self.base.get_prepend(prepend);
    }
}

/// Java `Storable`: `load(Properties)`/`store(Properties)` are this class's
/// overrides, which call the two-argument forms with `""`.
impl Storable for BatchRunTomoScreenState {
    fn store(&self, properties: &mut BTreeMap<String, String>) {
        self.store_with_prepend(properties, "");
    }

    fn store_with_prepend(&self, properties: &mut BTreeMap<String, String>, prepend: &str) {
        BatchRunTomoScreenState::store_with_prepend(self, properties, prepend);
    }

    fn load(&self, properties: &mut BTreeMap<String, String>) {
        self.load_with_prepend(properties, "");
    }

    fn load_with_prepend(&self, properties: &mut BTreeMap<String, String>, prepend: &str) {
        BatchRunTomoScreenState::load_with_prepend(self, properties, prepend);
    }
}
