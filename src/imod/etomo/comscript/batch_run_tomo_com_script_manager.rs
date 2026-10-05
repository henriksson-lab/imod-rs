//! `IMOD/Etomo/src/etomo/comscript/BatchRunTomoComScriptManager.java`.
//!
//! Loads and saves the batchruntomo and serieswatcher command files.  Used on the
//! event dispatch thread; the scripts sit behind a re-entrant lock, as in
//! `join_comscript_manager.rs`.

use std::cell::RefCell;

use super::batchruntomo_param::BatchruntomoParam;
use super::com_script::ComScript;
use super::com_script_util::ComScriptUtil;
use super::series_watcher_param::SeriesWatcherParam;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::batch_run_tomo_manager::BatchRunTomoManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::file_type;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::status::Status;
use crate::imod::etomo::util::event_queue::ReentrantLock;

/// Java `public final class BatchRunTomoComScriptManager`.
pub struct BatchRunTomoComScriptManager {
    /// Java private final `manager`.
    manager: &'static BatchRunTomoManager,
    /// Java private `scriptBatchRunTomo`, initially null.
    script_batch_run_tomo: RefCell<Option<ComScript>>,
    /// Java private `scriptSeriesWatcher`, initially null.
    script_series_watcher: RefCell<Option<ComScript>>,
    /// Java's unguarded field access, made exclusive.
    lock: ReentrantLock,
}

// SAFETY: the scripts are only reached under `lock`.
unsafe impl Send for BatchRunTomoComScriptManager {}
unsafe impl Sync for BatchRunTomoComScriptManager {}

impl BatchRunTomoComScriptManager {
    /// Java `BatchRunTomoComScriptManager(BatchRunTomoManager)`.
    pub fn new(manager: &'static BatchRunTomoManager) -> BatchRunTomoComScriptManager {
        BatchRunTomoComScriptManager {
            manager,
            script_batch_run_tomo: RefCell::new(None),
            script_series_watcher: RefCell::new(None),
            lock: ReentrantLock::new(),
        }
    }

    fn base_manager(&self) -> &'static dyn BaseManager {
        self.manager
    }

    /// Java `loadBatchRunTomo(AxisID)`.
    pub fn load_batch_run_tomo(&self, axis_id: AxisID) {
        let _lock = self.lock.lock();
        let file_name = file_type::CLASS
            .batch_run_tomo_comscript
            .get_file_name(Some(self.base_manager()), Some(axis_id));
        let script = ComScriptUtil::load_com_script_file_name(
            self.base_manager(),
            file_name.as_deref(),
            axis_id,
            true,
            true,
            false,
            false,
        );
        *self.script_batch_run_tomo.borrow_mut() = script;
    }

    /// Java `loadBatchRunTomo(AxisID, String)`.
    pub fn load_batch_run_tomo_root_name(&self, axis_id: AxisID, root_name: Option<&str>) {
        let _lock = self.lock.lock();
        let file_name = file_type::CLASS
            .batch_run_tomo_comscript
            .get_file_name_with_root_name(Some(self.base_manager()), root_name, Some(axis_id));
        let script = ComScriptUtil::load_com_script_file_name(
            self.base_manager(),
            file_name.as_deref(),
            axis_id,
            true,
            true,
            false,
            false,
        );
        *self.script_batch_run_tomo.borrow_mut() = script;
    }

    /// Java `loadSeriesWatcher(AxisID, boolean)`.
    pub fn load_series_watcher(&self, axis_id: AxisID, required: bool) -> bool {
        let _lock = self.lock.lock();
        let file_name = file_type::CLASS
            .series_watcher_comscript
            .get_file_name(Some(self.base_manager()), Some(axis_id));
        let script = ComScriptUtil::load_com_script_file_name(
            self.base_manager(),
            file_name.as_deref(),
            axis_id,
            true,
            required,
            false,
            false,
        );
        let loaded = script.is_some();
        *self.script_series_watcher.borrow_mut() = script;
        loaded
    }

    /// Java `isBatchRunTomoLoaded()`.
    pub fn is_batch_run_tomo_loaded(&self) -> bool {
        let _lock = self.lock.lock();
        self.script_batch_run_tomo.borrow().is_some()
    }

    /// Java `saveBatchRunTomo(BatchruntomoParam, AxisID)`.
    pub fn save_batch_run_tomo(&self, param: &BatchruntomoParam, axis_id: AxisID) {
        let _lock = self.lock.lock();
        ComScriptUtil::modify_command(
            self.base_manager(),
            self.script_batch_run_tomo.borrow_mut().as_mut(),
            param,
            "batchruntomo",
            axis_id,
            false,
            false,
        );
    }

    /// Java `getBatchRunTomoParam(AxisID, boolean, boolean)`.
    pub fn get_batch_run_tomo_param(
        &self,
        axis_id: AxisID,
        _do_validation: bool,
        parallel_processing: bool,
    ) -> BatchruntomoParam {
        let _lock = self.lock.lock();
        let mut param =
            BatchruntomoParam::get_instance(self.base_manager(), axis_id, parallel_processing);
        ComScriptUtil::initialize(
            self.base_manager(),
            &mut param,
            self.script_batch_run_tomo.borrow_mut().as_mut(),
            "batchruntomo",
            axis_id,
            false,
            false,
            true,
        );
        param
    }

    /// Java `getSeriesWatcherParam(AxisID, boolean)`.
    pub fn get_series_watcher_param(
        &self,
        axis_id: AxisID,
        _do_validation: bool,
    ) -> SeriesWatcherParam {
        let _lock = self.lock.lock();
        let mut param = SeriesWatcherParam::new(self.base_manager(), axis_id);
        ComScriptUtil::initialize(
            self.base_manager(),
            &mut param,
            self.script_series_watcher.borrow_mut().as_mut(),
            ProcessName::SERIES_WATCHER.get_text().unwrap_or(""),
            axis_id,
            false,
            false,
            true,
        );
        param
    }

    /// Java `saveSeriesWatcher(SeriesWatcherParam, AxisID)`.
    pub fn save_series_watcher(&self, param: &SeriesWatcherParam, axis_id: AxisID) {
        let _lock = self.lock.lock();
        ComScriptUtil::modify_command(
            self.base_manager(),
            self.script_series_watcher.borrow_mut().as_mut(),
            param,
            ProcessName::SERIES_WATCHER.get_text().unwrap_or(""),
            axis_id,
            false,
            false,
        );
    }
}
