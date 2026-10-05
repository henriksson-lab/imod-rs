//! `IMOD/Etomo/src/etomo/process/AutoAlignmentProcessManager.java`.
//!
//! The process manager of `AutoAlignmentController`: it runs `xfalign` and the
//! sample `midas` for the Join and Serial Sections interfaces.  It embeds the
//! `BaseProcessManager` superclass as `base` and installs itself as the base's
//! [`BaseProcessManagerHooks`] for its `postProcess`/`errorProcess` overrides, which
//! run on the process thread.

use std::sync::Arc;

use super::background_process::BackgroundProcess;
use super::base_process_manager::{
    AxisBusyException, BaseProcessManager, BaseProcessManagerHooks, SystemProcessException,
};
use super::interactive_system_program::InteractiveSystemProgram;
use super::process_interface::ProcessSeriesRef;
use crate::imod::etomo::auto_alignment_controller::AutoAlignmentController;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::command::Command;
use crate::imod::etomo::comscript::midas_param::MidasParam;
use crate::imod::etomo::comscript::xfalign_param::XfalignParam;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::process_name::ProcessName;

/// Java `rcsid`.
pub const RCSID: &str = "$Id:$";

/// Java `public final class AutoAlignmentProcessManager extends BaseProcessManager`.
pub struct AutoAlignmentProcessManager {
    /// The `BaseProcessManager` superclass.
    pub base: BaseProcessManager,
    /// Java private final `manager`.
    manager: &'static dyn BaseManager,
    /// Java private final `controller`.
    controller: &'static AutoAlignmentController,
}

impl AutoAlignmentProcessManager {
    /// Java `AutoAlignmentProcessManager(BaseManager, AutoAlignmentController)`.  The
    /// controller keeps it for the run, and the base class's start functions take
    /// `&'static self`.
    pub fn new(
        manager: &'static dyn BaseManager,
        controller: &'static AutoAlignmentController,
    ) -> &'static AutoAlignmentProcessManager {
        let process_manager: &'static AutoAlignmentProcessManager =
            Box::leak(Box::new(AutoAlignmentProcessManager {
                base: BaseProcessManager::new(manager),
                manager,
                controller,
            }));
        process_manager.base.set_hooks(process_manager);
        process_manager
    }

    /// Java `xfalign(XfalignParam, AxisID, ProcessSeries) throws AxisBusyException`.
    /// Run xfalign.
    pub fn xfalign(
        &'static self,
        xfalign_param: Arc<XfalignParam>,
        axis_id: AxisID,
        process_series: Option<ProcessSeriesRef>,
    ) -> Result<String, AxisBusyException> {
        let background_process = self.base.start_background_process_command(
            xfalign_param as Arc<dyn Command + Send + Sync>,
            false,
            axis_id,
            Some(ProcessName::XFALIGN),
            None,
            process_series,
            false,
            true, // POPUP_CHUNK_WARNINGS_DEFAULT
        )?;
        Ok(background_process.get_name())
    }

    /// Java `midasSample(MidasParam) throws SystemProcessException`.  Run midas on the
    /// sample file.
    pub fn midas_sample(
        &'static self,
        midas_param: Arc<MidasParam>,
    ) -> Result<Option<String>, SystemProcessException> {
        let program = self
            .base
            .start_interactive_system_program(midas_param as Arc<dyn Command + Send + Sync>)?;
        Ok(program.get_name())
    }
}

impl BaseProcessManagerHooks for AutoAlignmentProcessManager {
    /// Java override `postProcess(BackgroundProcess)`.  Non-generic post processing
    /// for a successful BackgroundProcess.
    fn post_process_background(&self, base: &BaseProcessManager, process: &BackgroundProcess) {
        base.post_process_background_base(process);
        let Some(command_name) = process.get_command_name() else {
            return;
        };
        // Java's unused local `processDetails`.
        let _process_details = process
            .get_command_details()
            .and_then(|details| details.get_process_details());
        let command = process.get_command();
        if command_name == XfalignParam::get_name() {
            let Some(command) = command else {
                return;
            };
            base.write_log_file(
                process,
                process.get_axis_id(),
                &format!("{}.log", XfalignParam::get_name()),
            );
            self.controller
                .copy_xf_file(command.get_command_output_file().as_deref());
            self.controller.msg_process_ended();
        }
    }

    /// Java override `errorProcess(BackgroundProcess)`.
    fn error_process_background(&self, base: &BaseProcessManager, process: &BackgroundProcess) {
        let Some(command_name) = process.get_command_name() else {
            return;
        };
        if command_name == XfalignParam::get_name() {
            base.write_log_file(
                process,
                process.get_axis_id(),
                &format!("{}.log", XfalignParam::get_name()),
            );
            self.controller.msg_process_ended();
        }
    }

    /// Java override `postProcess(InteractiveSystemProgram)`.
    fn post_process_interactive(
        &self,
        _base: &BaseProcessManager,
        program: &InteractiveSystemProgram,
    ) {
        let Some(command_name) = program.get_command_name() else {
            return;
        };
        let Some(command) = program.get_command() else {
            return;
        };
        if command_name == MidasParam::get_name() {
            let output_file = command.get_command_output_file();
            // `program.getOutputFileLastModified().getLong()`: a null long reads as
            // `Long.MIN_VALUE`.
            let output_file_last_modified =
                program.get_output_file_last_modified().unwrap_or(i64::MIN);
            if let Some(output_file) = output_file
                && output_file.exists()
                && std::fs::metadata(&output_file)
                    .and_then(|metadata| metadata.modified())
                    .ok()
                    .and_then(|modified| modified.duration_since(std::time::UNIX_EPOCH).ok())
                    .map_or(0, |modified| modified.as_millis() as i64)
                    > output_file_last_modified
            {
                self.controller.copy_xf_file(Some(&output_file));
            }
        }
    }
}

impl AutoAlignmentProcessManager {
    /// Java `manager` field read (the superclass also holds it).
    pub fn get_manager(&self) -> &'static dyn BaseManager {
        self.manager
    }
}
