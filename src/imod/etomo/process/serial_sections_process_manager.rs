//! `IMOD/Etomo/src/etomo/process/SerialSectionsProcessManager.java`.
//!
//! The process manager of the Serial Sections interface (`SerialSectionsManager`):
//! extractpieces, the preblend/blend com scripts, xftoxg and newst.  It embeds the
//! `BaseProcessManager` superclass as `base` and installs itself as the base's
//! [`BaseProcessManagerHooks`] for its `postProcess`/`errorProcess` overrides.
//!
//! **Threads.**  The overrides run on the process thread.  The state is shared (its
//! fields carry their own locks) and is updated there, as in the source;
//! `msgPreblendSucceeded`, which reaches the dialog, is made on the event dispatch
//! thread.

use std::sync::Arc;

use super::background_process::BackgroundProcess;
use super::base_process_manager::{AxisBusyException, BaseProcessManager, BaseProcessManagerHooks};
use super::blendmont_process_monitor::BlendmontProcessMonitor;
use super::com_script_process::ComScriptProcess;
use super::monitor::ProcessMonitor;
use super::newst_process_monitor::NewstProcessMonitor;
use super::process_interface::{
    ProcessResultDisplayRef, ProcessSeriesRef, SystemProcessInterface as _,
};
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::blendmont_param::{self, BlendmontParam};
use crate::imod::etomo::comscript::command::Command;
use crate::imod::etomo::comscript::command_mode;
use crate::imod::etomo::comscript::const_newst_param::ConstNewstParam;
use crate::imod::etomo::comscript::extractpieces_param::ExtractpiecesParam;
use crate::imod::etomo::comscript::newst_param::NewstParam;
use crate::imod::etomo::comscript::xftoxg_param::XftoxgParam;
use crate::imod::etomo::serial_sections_manager::SerialSectionsManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::view_type::ViewType;
use crate::imod::etomo::util::event_queue;

/// Java `public final class SerialSectionsProcessManager extends BaseProcessManager`.
pub struct SerialSectionsProcessManager {
    /// The `BaseProcessManager` superclass.
    pub base: BaseProcessManager,
    /// Java private final `manager`.
    manager: &'static SerialSectionsManager,
}

impl SerialSectionsProcessManager {
    /// Java `SerialSectionsProcessManager(SerialSectionsManager)`.  The manager keeps
    /// it for the run, and the base class's start functions take `&'static self`.
    pub fn new(manager: &'static SerialSectionsManager) -> &'static Self {
        let process_manager: &'static SerialSectionsProcessManager = Box::leak(Box::new(Self {
            base: BaseProcessManager::new(manager),
            manager,
        }));
        process_manager.base.set_hooks(process_manager);
        process_manager
    }

    /// Java `createNewFile(String)`, the inherited `BaseProcessManager` member.
    pub fn create_new_file(&self, absolute_path: &str) {
        self.base.create_new_file(absolute_path);
    }

    /// Java `extractpieces(ExtractpiecesParam, AxisID, ProcessSeries) throws
    /// AxisBusyException`.  Run extractpieces.
    pub fn extractpieces(
        &'static self,
        param: &mut ExtractpiecesParam,
        axis_id: AxisID,
        process_series: Option<ProcessSeriesRef>,
    ) -> Result<String, AxisBusyException> {
        let background_process = self.base.start_background_process_array_force(
            param.get_command(),
            axis_id,
            false,
            None,
            process_series,
            Some(ProcessName::EXTRACTPIECES),
        )?;
        Ok(background_process.get_name())
    }

    /// Java `blend(BlendmontParam, ProcessResultDisplay, AxisID, ProcessSeries) throws
    /// AxisBusyException`.  Run the blend comscript.
    pub fn blend(
        &'static self,
        blendmont_param: Arc<BlendmontParam>,
        process_result_display: Option<ProcessResultDisplayRef>,
        axis_id: AxisID,
        process_series: Option<ProcessSeriesRef>,
    ) -> Result<String, AxisBusyException> {
        // Start the com script in the background
        let blendmont_process_monitor =
            BlendmontProcessMonitor::new(self.manager, axis_id, blendmont_param.get_mode());
        // Start the com script in the background
        let com_script_process = self.base.start_com_script_param(
            blendmont_param as Arc<dyn Command + Send + Sync>,
            true, // BlendmontParam is a CommandDetails: startComScript(CommandDetails, ...)
            Some(blendmont_process_monitor as Arc<dyn ProcessMonitor>),
            axis_id,
            process_result_display,
            process_series,
            None,
        )?;
        Ok(com_script_process.get_name())
    }

    /// Java `xftoxg(XftoxgParam, ProcessResultDisplay, AxisID, ProcessSeries) throws
    /// AxisBusyException`.  Run xftoxg.
    pub fn xftoxg(
        &'static self,
        param: Arc<XftoxgParam>,
        process_result_display: Option<ProcessResultDisplayRef>,
        axis_id: AxisID,
        process_series: Option<ProcessSeriesRef>,
    ) -> Result<String, AxisBusyException> {
        let background_process = self.base.start_background_process_command_display(
            param as Arc<dyn Command + Send + Sync>,
            axis_id,
            process_result_display,
            Some(ProcessName::XFTOXG),
            process_series,
        )?;
        Ok(background_process.get_name())
    }

    /// Java `newst(ConstNewstParam, ProcessResultDisplay, AxisID, ProcessSeries)
    /// throws AxisBusyException`.  Run newst.com.
    pub fn newst(
        &'static self,
        newst_param: Arc<NewstParam>,
        process_result_display: Option<ProcessResultDisplayRef>,
        axis_id: AxisID,
        process_series: Option<ProcessSeriesRef>,
    ) -> Result<String, AxisBusyException> {
        // Start the com script in the background
        let newst_process_monitor = NewstProcessMonitor::new(
            self.manager,
            axis_id,
            ProcessName::NEWST,
            Arc::clone(&newst_param) as Arc<dyn ConstNewstParam + Send + Sync>,
        );
        // Start the com script in the background
        let com_script_process = self.base.start_com_script_param(
            newst_param as Arc<dyn Command + Send + Sync>,
            true, // ConstNewstParam extends CommandDetails
            Some(newst_process_monitor as Arc<dyn ProcessMonitor>),
            axis_id,
            process_result_display,
            process_series,
            None,
        )?;
        Ok(com_script_process.get_name())
    }

    /// Java private `setInvalidEdgeFunctions(Command, boolean)`.
    fn set_invalid_edge_functions(
        &self,
        command: Option<&Arc<dyn Command + Send + Sync>>,
        succeeded: bool,
    ) {
        // Fixed in translation: a com script started without its command makes the
        // source dereference null; there is nothing to record then.
        let Some(command) = command else {
            return;
        };
        let mode = command.get_command_mode();
        if BaseManager::get_view_type(self.manager) == ViewType::Montage
            && command.get_command_name().as_deref() == Some(blendmont_param::COMMAND_NAME)
            && (command_mode::equals_mode(mode, &blendmont_param::Mode::SerialSectionPreblend)
                || command_mode::equals_mode(mode, &blendmont_param::Mode::SerialSectionBlend))
        {
            self.manager
                .get_state()
                .set_invalid_edge_functions(!succeeded);
        }
    }
}

impl BaseProcessManagerHooks for SerialSectionsProcessManager {
    /// Java override `postProcess(ComScriptProcess)`.
    fn post_process_com_script(&self, _base: &BaseProcessManager, script: &ComScriptProcess) {
        // Script specific post processing
        let process_name = script.get_process_name();
        let process_details = script
            .get_command_details()
            .and_then(|details| details.get_process_details());
        let state = self.manager.get_state();
        if process_name == Some(ProcessName::PREBLEND) {
            self.set_invalid_edge_functions(script.get_command(), true);
            // try { ... } catch (Exception e): a com script started without its
            // details (null processDetails) or a field the details do not carry.
            let recorded = (|| -> Option<()> {
                let process_details = process_details?;
                state.set_preblend_robust_fitting(
                    process_details.get_double_value(&blendmont_param::Field::RobustFitting)?,
                );
                state.set_preblend_fix_intensity_from_edges(
                    process_details
                        .get_int_value(&blendmont_param::Field::FixIntensityFromEdges)?,
                );
                state.set_preblend_sum_pieces_for_gradient(
                    process_details.get_int_value(&blendmont_param::Field::SumPiecesForGradient)?,
                );
                state.set_preblend_other_sum_gradient_file(
                    process_details
                        .get_string(&blendmont_param::Field::OtherSumGradientFile)
                        .as_deref(),
                );
                Some(())
            })();
            if recorded.is_none() {
                eprintln!("java.lang.NullPointerException");
                eprintln!("ERROR:  Unable to record state.");
                return;
            }
            let manager = self.manager;
            if event_queue::is_dispatch_thread() {
                manager.msg_preblend_succeeded();
            } else {
                event_queue::invoke_later(move || manager.msg_preblend_succeeded());
            }
        } else if process_name == Some(ProcessName::BLEND) {
            self.set_invalid_edge_functions(script.get_command(), true);
        }
    }

    /// Java package-private override `errorProcess(ComScriptProcess)`.
    fn error_process_com_script(&self, _base: &BaseProcessManager, script: &ComScriptProcess) {
        let process_name = script.get_process_name();
        if process_name == Some(ProcessName::BLEND) {
            self.set_invalid_edge_functions(script.get_command(), false);
        } else if process_name == Some(ProcessName::PREBLEND) {
            self.set_invalid_edge_functions(script.get_command(), false);
        }
    }

    /// Java `postProcess(BackgroundProcess)`: not overridden.
    fn post_process_background(&self, base: &BaseProcessManager, process: &BackgroundProcess) {
        base.post_process_background_base(process);
    }
}
