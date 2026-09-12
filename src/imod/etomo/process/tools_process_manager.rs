//! `IMOD/Etomo/src/etomo/process/ToolsProcessManager.java`.
//!
//! The named command-parameter, process, and monitor source units remain direct
//! execution boundaries.  This coordinator nevertheless preserves the Java fields,
//! method surface, inherited `BaseProcessManager`, and source command/file-name
//! selection.  It never reports an unstarted process as started.
#![allow(dead_code)]

use std::convert::Infallible;

use crate::imod::etomo::process::base_process_manager::BaseProcessManager;
use crate::imod::etomo::tools_manager::ToolsManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::file_type::FileType;

/// Java final `ToolsProcessManager extends BaseProcessManager`.
pub struct ToolsProcessManager {
    /// Java superclass state.
    base: BaseProcessManager,
    /// Java final `manager`.
    manager: &'static ToolsManager,
}

impl ToolsProcessManager {
    /// Java `ToolsProcessManager(ToolsManager)`.
    pub fn new(manager: &'static ToolsManager) -> Self {
        Self {
            base: BaseProcessManager::new(None),
            manager,
        }
    }

    /// Java `gpuTiltTest(GpuTiltTestParam, AxisID)`.
    pub fn gpu_tilt_test(&self, param: Option<Infallible>, axis_id: AxisID) -> Option<String> {
        let _ = (self.base.in_use(axis_id, None, false), param, axis_id);
        // TODO(unit): GpuTiltTestParam.java, BackgroundProcess.java, and
        // BaseProcessManager.startBackgroundProcess.
        None
    }

    /// Java `flatten(WarpVolParam, AxisID, ProcessResultDisplay, ProcessSeries, FileType)`.
    pub fn flatten(
        &self,
        param: Option<Infallible>,
        axis_id: AxisID,
        process_result_display: Option<Infallible>,
        process_series: Option<Infallible>,
        file_type: &FileType,
    ) -> Option<String> {
        let command = file_type.get_file_name(Some(self.manager), Some(AxisID::Only));
        let _ = (
            param,
            axis_id,
            process_result_display,
            process_series,
            command,
        );
        // TODO(unit): WarpVolParam.java, Matchvol1ProcessMonitor.java,
        // ComScriptProcess.java, and BaseProcessManager.startComScript.
        None
    }

    /// Java `flattenWarp(FlattenWarpParam, ProcessResultDisplay, ProcessSeries, AxisID)`.
    pub fn flatten_warp(
        &self,
        param: Option<Infallible>,
        process_result_display: Option<Infallible>,
        process_series: Option<Infallible>,
        axis_id: AxisID,
    ) -> Option<String> {
        let _ = (param, process_result_display, process_series, axis_id);
        // TODO(unit): FlattenWarpParam.java, BackgroundProcess.java, and
        // BaseProcessManager.startBackgroundProcess.
        None
    }

    /// Java override `postProcess(BackgroundProcess)`.
    pub fn post_process_background(&self, process: Option<Infallible>) {
        self.base.post_process_background(None);
        let _ = process;
        // TODO(unit): BackgroundProcess.java and GpuTiltTestParam.java; their output,
        // process-name and axis getters implement the source conditional/logging path.
    }

    /// Java override `errorProcess(BackgroundProcess)`.
    pub fn error_process_background(&self, process: Option<Infallible>) {
        let _ = process;
        // TODO(unit): BackgroundProcess.java and GpuTiltTestParam.java.
    }

    /// Java package-private `getManager()`.
    pub fn get_manager(&self) -> &'static ToolsManager {
        self.manager
    }

    /// Java override `postProcess(DetachedProcess)`.
    pub fn post_process_detached(&self, process: Option<Infallible>) {
        let _ = process;
    }

    /// Java `alignFrames(ProcessSeries, FileType, FileType)`.
    pub fn align_frames(
        &self,
        process_series: Option<Infallible>,
        file_type: &FileType,
        log_file_type: &FileType,
    ) -> Option<String> {
        let log_file_name = log_file_type.get_file_name(Some(self.manager), Some(AxisID::Only));
        let command = file_type.get_file_name(Some(self.manager), Some(AxisID::Only));
        let _ = (process_series, log_file_name, command);
        // TODO(unit): AlignFramesProcessMonitor.java, ComScriptProcess.java, and
        // BaseProcessManager.startComScript.
        None
    }

    /// Java `sortTiltFrames(SortTiltFramesParam, ProcessSeries)`.
    pub fn sort_tilt_frames(
        &self,
        param: Option<Infallible>,
        process_series: Option<Infallible>,
    ) -> Option<String> {
        let _ = (param, process_series);
        // TODO(unit): SortTiltFramesParam.java, BackgroundProcess.java, and
        // BaseProcessManager.startBackgroundProcess.
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::etomo::ui::swing::etomo_menu::ToolType;

    #[test]
    fn manager_identity_is_preserved() {
        let manager = ToolsManager::new(ToolType::FlattenVolume);
        let process_manager = ToolsProcessManager::new(manager);
        assert!(std::ptr::eq(process_manager.get_manager(), manager));
    }
}
