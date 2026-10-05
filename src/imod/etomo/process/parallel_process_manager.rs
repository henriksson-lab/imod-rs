//! `IMOD/Etomo/src/etomo/process/ParallelProcessManager.java`.
//!
//! The process manager of the generic parallel process and anisotropic diffusion
//! interfaces (`ParallelManager`).  It embeds the `BaseProcessManager` superclass as
//! `base` and installs itself as the base's [`BaseProcessManagerHooks`] for its two
//! `postProcess` overrides.
//!
//! **Threads.**  The overrides run on the process thread.  The state they write is
//! the manager's `ParallelState` (a `Mutex` per field); the chunksetup results go to
//! the `ParallelDialog`, an event dispatch thread object, so those two manager calls
//! are posted to the event dispatch thread, in the source's order.

use std::sync::Arc;

use crate::imod::etomo::comscript::anisotropic_diffusion_param::{self, AnisotropicDiffusionParam};
use crate::imod::etomo::comscript::chunksetup_param::{self, ChunksetupParam};
use crate::imod::etomo::comscript::command::Command;
use crate::imod::etomo::comscript::command_mode;
use crate::imod::etomo::comscript::trimvol_param::TrimvolParam;
use crate::imod::etomo::parallel_manager::ParallelManager;
use crate::imod::etomo::process::background_process::BackgroundProcess;
use crate::imod::etomo::process::base_process_manager::{
    AxisBusyException, BaseProcessManager, BaseProcessManagerHooks,
};
use crate::imod::etomo::process::process_interface::{ProcessSeriesRef, SystemProcessInterface};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::util::event_queue;

/// Java `public final class ParallelProcessManager extends BaseProcessManager`.
pub struct ParallelProcessManager {
    /// The `BaseProcessManager` superclass.
    pub base: BaseProcessManager,
    /// Java private final `manager`.
    manager: &'static ParallelManager,
}

impl ParallelProcessManager {
    /// Java `ParallelProcessManager(ParallelManager)`.  The manager keeps it for the
    /// run, and the base class's start functions take `&'static self`.
    pub fn new(manager: &'static ParallelManager) -> &'static ParallelProcessManager {
        let process_manager: &'static ParallelProcessManager = Box::leak(Box::new(Self {
            base: BaseProcessManager::new(manager),
            manager,
        }));
        process_manager.base.set_hooks(process_manager);
        process_manager
    }

    /// Java `trimVolume(TrimvolParam, ProcessSeries) throws AxisBusyException`.  Run
    /// trimvol.
    pub fn trim_volume(
        &'static self,
        trimvol_param: Arc<TrimvolParam>,
        process_series: Option<ProcessSeriesRef>,
    ) -> Result<String, AxisBusyException> {
        let background_process = self.base.start_background_process_command(
            trimvol_param as Arc<dyn Command + Send + Sync>,
            true,
            AxisID::Only,
            Some(ProcessName::TRIMVOL),
            None,
            process_series,
            false,
            true, // POPUP_CHUNK_WARNINGS_DEFAULT
        )?;
        Ok(background_process.get_name())
    }

    /// Java `anisotropicDiffusion(AnisotropicDiffusionParam, ProcessSeries) throws
    /// AxisBusyException`.
    pub fn anisotropic_diffusion(
        &'static self,
        param: Arc<AnisotropicDiffusionParam>,
        process_series: Option<ProcessSeriesRef>,
    ) -> Result<String, AxisBusyException> {
        let background_process = self.base.start_background_process_command(
            param as Arc<dyn Command + Send + Sync>,
            true,
            AxisID::Only,
            Some(ProcessName::ANISOTROPIC_DIFFUSION),
            None,
            process_series,
            false,
            true, // POPUP_CHUNK_WARNINGS_DEFAULT
        )?;
        Ok(background_process.get_name())
    }

    /// Java `chunksetup(ChunksetupParam, ProcessSeries) throws AxisBusyException`.
    pub fn chunksetup(
        &'static self,
        param: Arc<ChunksetupParam>,
        process_series: Option<ProcessSeriesRef>,
    ) -> Result<String, AxisBusyException> {
        let background_process = self.base.start_background_process_command(
            param as Arc<dyn Command + Send + Sync>,
            true,
            AxisID::Only,
            Some(ProcessName::CHUNKSETUP),
            None,
            process_series,
            false,
            true, // POPUP_CHUNK_WARNINGS_DEFAULT
        )?;
        Ok(background_process.get_name())
    }

    /// Java package-private `getManager()`.
    pub fn get_manager(&self) -> &'static ParallelManager {
        self.manager
    }
}

impl BaseProcessManagerHooks for ParallelProcessManager {
    /// Java override `postProcess(BackgroundProcess)`.  (The source does not call
    /// `super.postProcess`.)
    fn post_process_background(&self, _base: &BaseProcessManager, process: &BackgroundProcess) {
        let Some(command_details) = process.get_command_details() else {
            return;
        };
        if command_details.get_command_name().as_deref()
            == Some(&ProcessName::ANISOTROPIC_DIFFUSION.to_string())
        {
            let state = self.manager.get_state();
            let Some(details) = command_details.get_process_details() else {
                return;
            };
            // Java unboxes the double; an unknown field throws there.
            if let Some(k_value) =
                details.get_double_value(&anisotropic_diffusion_param::Field::KValue)
            {
                state.set_test_k_value(k_value);
            }
            state.set_test_iteration_list(
                details
                    .get_iterator_element_list(&anisotropic_diffusion_param::Field::IterationList)
                    .as_ref(),
            );
        } else if command_details.get_process_name() == Some(ProcessName::CHUNKSETUP) {
            let std_output = process.get_std_output();
            let one_line_command_program =
                command_details.get_process_details().and_then(|details| {
                    details.get_string(&chunksetup_param::Field::OneLineCommandProgram)
                });
            let manager = self.manager;
            // The dialog is an event dispatch thread object.
            event_queue::invoke_later(move || {
                manager.set_chunk_setup_output_file(std_output.as_deref());
                manager.set_parallel_process_name(one_line_command_program.as_deref());
            });
        }
    }

    /// Java override `postProcess(DetachedProcess)`.
    fn post_process_detached(&self, base: &BaseProcessManager, process: &BackgroundProcess) {
        base.post_process_detached_base(process);
        let Some(command) = process.get_command() else {
            return;
        };
        if command.get_command_name().as_deref() == Some(&ProcessName::PROCESSCHUNKS.to_string()) {
            let subcommand_details = command.get_subcommand_details();
            if let Some(subcommand_details) = subcommand_details
                && subcommand_details.get_command_name().as_deref()
                    == Some(&ProcessName::ANISOTROPIC_DIFFUSION.to_string())
                && command_mode::equals_mode(
                    subcommand_details.get_command_mode(),
                    &anisotropic_diffusion_param::Mode::VaryingK,
                )
            {
                let state = self.manager.get_state();
                state.set_test_k_value_list(
                    subcommand_details
                        .get_string(&anisotropic_diffusion_param::Field::KValueList)
                        .as_deref(),
                );
                // Java unboxes the int; an unknown field throws there.
                if let Some(iteration) =
                    subcommand_details.get_int_value(&anisotropic_diffusion_param::Field::Iteration)
                {
                    state.set_test_iteration(iteration);
                }
            }
        }
    }
}
