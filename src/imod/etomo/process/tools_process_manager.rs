//! `IMOD/Etomo/src/etomo/process/ToolsProcessManager.java`.
//!
//! The process manager of the Tools interface (`ToolsManager`).  It embeds the
//! `BaseProcessManager` superclass as `base` and installs itself as the base's
//! [`BaseProcessManagerHooks`] for its `postProcess`/`errorProcess` overrides.
//!
//! **Threads.**  The overrides run on the process thread; the logging they do
//! reaches the Tools dialog (the manager's `LogInterface`, an event dispatch
//! thread object) and `gpuTiltTestSuceeded` opens a dialog, so those calls are
//! posted to the event dispatch thread, in the source's order.

use std::sync::Arc;

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::command::Command;
use crate::imod::etomo::comscript::flatten_warp_param::FlattenWarpParam;
use crate::imod::etomo::comscript::gpu_tilt_test_param::{self, GpuTiltTestParam};
use crate::imod::etomo::comscript::sort_tilt_frames_param::SortTiltFramesParam;
use crate::imod::etomo::comscript::warp_vol_param::WarpVolParam;
use crate::imod::etomo::process::align_frames_process_monitor::AlignFramesProcessMonitor;
use crate::imod::etomo::process::background_process::BackgroundProcess;
use crate::imod::etomo::process::base_process_manager::{
    AxisBusyException, BaseProcessManager, BaseProcessManagerHooks,
};
use crate::imod::etomo::process::matchvol1_process_monitor::Matchvol1ProcessMonitor;
use crate::imod::etomo::process::monitor::ProcessMonitor;
use crate::imod::etomo::process::process_interface::{
    ProcessResultDisplayRef, ProcessSeriesRef, SystemProcessInterface,
};
use crate::imod::etomo::tools_manager::ToolsManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::file_type::{self, FileType};
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::util::event_queue;

/// Java `public static final String rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `public final class ToolsProcessManager extends BaseProcessManager`.
pub struct ToolsProcessManager {
    /// The `BaseProcessManager` superclass.
    pub base: BaseProcessManager,
    /// Java private final `manager`.
    manager: &'static ToolsManager,
}

impl ToolsProcessManager {
    /// Java `ToolsProcessManager(ToolsManager)`.  The manager keeps it for the
    /// run, and the base class's start functions take `&'static self`.
    pub fn new(manager: &'static ToolsManager) -> &'static Self {
        let process_manager: &'static ToolsProcessManager = Box::leak(Box::new(Self {
            base: BaseProcessManager::new(manager),
            manager,
        }));
        process_manager.base.set_hooks(process_manager);
        process_manager
    }

    /// Java `touch(String, BaseManager)`, the static `BaseProcessManager`
    /// member the source reaches as `ToolsProcessManager.touch`.
    pub fn touch(absolute_path: &str, manager: Option<&'static dyn BaseManager>) {
        BaseProcessManager::touch(absolute_path, manager);
    }

    /// Java `gpuTiltTest(GpuTiltTestParam, AxisID)`.  Run gpuTiltTest.
    pub fn gpu_tilt_test(
        &'static self,
        param: &GpuTiltTestParam,
        axis_id: AxisID,
    ) -> Result<String, AxisBusyException> {
        let background_process = self.base.start_background_process_array(
            param.get_command(),
            axis_id,
            Some(ProcessName::GPU_TILT_TEST),
            None,
        )?;
        Ok(background_process.get_name())
    }

    /// Java `flatten(WarpVolParam, AxisID, ProcessResultDisplay, ProcessSeries, FileType)`.
    /// Run the appropriate flatten com file for the given axis ID.
    pub fn flatten(
        &'static self,
        param: Arc<WarpVolParam>,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayRef>,
        process_series: Option<ProcessSeriesRef>,
        file_type: &'static FileType,
    ) -> Result<String, AxisBusyException> {
        // Create the required tilt command
        let command = file_type
            .get_file_name(Some(self.manager), Some(AxisID::Only))
            .unwrap_or_else(|| "null".to_owned());
        // Instantiate the process monitor
        let monitor =
            Matchvol1ProcessMonitor::get_flatten_instance(self.manager, axis_id, Some(file_type));
        // Start the com script in the background
        let com_script_process = self.base.start_com_script_command_file_type(
            &command,
            Some(monitor as Arc<dyn ProcessMonitor>),
            axis_id,
            process_result_display,
            Some(param as Arc<dyn Command + Send + Sync>),
            process_series,
            Some(file_type),
        )?;
        Ok(com_script_process.get_name())
    }

    /// Java `flattenWarp(FlattenWarpParam, ProcessResultDisplay, ProcessSeries, AxisID)`.
    pub fn flatten_warp(
        &'static self,
        param: &FlattenWarpParam,
        process_result_display: Option<ProcessResultDisplayRef>,
        process_series: Option<ProcessSeriesRef>,
        axis_id: AxisID,
    ) -> Result<String, AxisBusyException> {
        let background_process = self.base.start_background_process_array_display(
            param.get_command_array(),
            axis_id,
            process_result_display,
            Some(param.get_process_name()),
            process_series,
        )?;
        Ok(background_process.get_name())
    }

    /// Java package-private `getManager()`.
    pub fn get_manager(&self) -> &'static ToolsManager {
        self.manager
    }

    /// Java `alignFrames(ProcessSeries, FileType, FileType)`.
    pub fn align_frames(
        &'static self,
        process_series: Option<ProcessSeriesRef>,
        file_type: &'static FileType,
        log_file_type: &FileType,
    ) -> Result<String, AxisBusyException> {
        // Create the process monitor
        let align_frames_process_monitor = AlignFramesProcessMonitor::new(
            self.manager,
            AxisID::Only,
            log_file_type.get_file_name(Some(self.manager), Some(AxisID::Only)),
        );
        let com_script_process = self.base.start_com_script_command_file_type(
            &file_type
                .get_file_name(Some(self.manager), Some(AxisID::Only))
                .unwrap_or_else(|| "null".to_owned()),
            Some(align_frames_process_monitor as Arc<dyn ProcessMonitor>),
            AxisID::Only,
            None,
            None,
            process_series,
            Some(file_type),
        )?;
        Ok(com_script_process.get_name())
    }

    /// Java `sortTiltFrames(SortTiltFramesParam, ProcessSeries)`.
    ///
    /// Java passes a null axis to `startBackgroundProcess`; the Tools
    /// interface is single axis, so the process runs on `AxisID.ONLY`, which
    /// is what the base class's null-axis paths resolve to.
    pub fn sort_tilt_frames(
        &'static self,
        param: &mut SortTiltFramesParam,
        process_series: Option<ProcessSeriesRef>,
    ) -> Result<String, AxisBusyException> {
        let background_process = self.base.start_background_process_array_display(
            param.get_command_array(),
            AxisID::Only,
            None,
            Some(ProcessName::SORT_TILT_FRAMES),
            process_series,
        )?;
        Ok(background_process.get_name())
    }
}

impl BaseProcessManagerHooks for ToolsProcessManager {
    /// Java override `postProcess(BackgroundProcess)`.
    fn post_process_background(&self, base: &BaseProcessManager, process: &BackgroundProcess) {
        base.post_process_background_base(process);
        if process.get_process_name() == Some(ProcessName::GPU_TILT_TEST) {
            let output = process.get_std_output();
            let std_error = process.get_std_error();
            let axis_id = process.get_axis_id();
            let manager = self.manager;
            // Logged and reported on the event dispatch thread, where the Tools
            // dialog's log is.
            event_queue::invoke_later(move || {
                let title = format!("{}:", ProcessName::GPU_TILT_TEST);
                manager.log_message_with_keyword(
                    output.as_deref(),
                    Some(gpu_tilt_test_param::OUTPUT_KEYWORD),
                    Some(&title),
                    Some(axis_id),
                );
                manager.log_message_with_keyword(
                    std_error.as_deref(),
                    Some(gpu_tilt_test_param::OUTPUT_KEYWORD),
                    Some(&title),
                    Some(axis_id),
                );
                manager.gpu_tilt_test_suceeded(output.as_deref(), axis_id);
            });
        }
    }

    /// Java override `errorProcess(BackgroundProcess)`.
    fn error_process_background(&self, _base: &BaseProcessManager, process: &BackgroundProcess) {
        if process.get_process_name() == Some(ProcessName::GPU_TILT_TEST) {
            let output = process.get_std_output();
            let std_error = process.get_std_error();
            let axis_id = process.get_axis_id();
            let manager = self.manager;
            // Logged on the event dispatch thread, where the Tools dialog's
            // log is.
            event_queue::invoke_later(move || {
                let title = format!("{}:", ProcessName::GPU_TILT_TEST);
                manager.log_message_with_keyword(
                    output.as_deref(),
                    Some(gpu_tilt_test_param::OUTPUT_KEYWORD),
                    Some(&title),
                    Some(axis_id),
                );
                manager.log_message_with_keyword(
                    std_error.as_deref(),
                    Some(gpu_tilt_test_param::OUTPUT_KEYWORD),
                    Some(&title),
                    Some(axis_id),
                );
                manager.log_message_with_keyword_file_type(
                    Some(&file_type::CLASS.gpu_test_log),
                    Some(gpu_tilt_test_param::OUTPUT_KEYWORD),
                    Some(&title),
                    Some(axis_id),
                );
            });
        }
    }

    /// Java override `postProcess(DetachedProcess)`: empty.
    fn post_process_detached(&self, _base: &BaseProcessManager, _process: &BackgroundProcess) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::etomo::ui::swing::etomo_menu::ToolType;

    #[test]
    fn manager_identity_is_preserved() {
        let manager = crate::imod::etomo::util::event_queue::invoke_and_wait(|| {
            ToolsManager::new(ToolType::FlattenVolume)
        });
        let process_manager = ToolsProcessManager::new(manager);
        assert!(std::ptr::eq(process_manager.get_manager(), manager));
    }
}
