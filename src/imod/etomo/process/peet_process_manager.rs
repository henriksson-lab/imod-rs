//! `IMOD/Etomo/src/etomo/process/PeetProcessManager.java`.
//!
//! The process manager of the PEET interface (`PeetManager`).  It embeds the
//! `BaseProcessManager` superclass as `base` and installs itself as the base's
//! [`BaseProcessManagerHooks`] for its `postProcess(BackgroundProcess)` override.
//!
//! **Threads.**  `postProcess` runs on the process thread.  The iteration list size
//! goes to the manager's `PeetState` (a `Mutex`); `manager.logMessage` writes to the
//! log window, an event dispatch thread object, so it is posted to that thread.

use std::any::Any;
use std::sync::Arc;

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::average_all_param::{self, AverageAllParam};
use crate::imod::etomo::comscript::command::Command;
use crate::imod::etomo::comscript::peet_parser_param::{self, PeetParserParam};
use crate::imod::etomo::peet_manager::PeetManager;
use crate::imod::etomo::process::background_process::BackgroundProcess;
use crate::imod::etomo::process::base_process_manager::{
    AxisBusyException, BaseProcessManager, BaseProcessManagerHooks,
};
use crate::imod::etomo::process::process_interface::ProcessSeriesRef;
use crate::imod::etomo::storage::loggable::Loggable;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::util::event_queue;

/// Java `public static final String rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `public final class PeetProcessManager extends BaseProcessManager`.
pub struct PeetProcessManager {
    /// The `BaseProcessManager` superclass.
    pub base: BaseProcessManager,
    /// Java private final `manager`.
    manager: &'static PeetManager,
}

impl PeetProcessManager {
    /// Java `PeetProcessManager(PeetManager)`.  The manager keeps it for the run.
    pub fn new(manager: &'static PeetManager) -> &'static PeetProcessManager {
        let process_manager: &'static PeetProcessManager = Box::leak(Box::new(Self {
            base: BaseProcessManager::new(manager),
            manager,
        }));
        process_manager.base.set_hooks(process_manager);
        process_manager
    }

    /// Java `peetParser(PeetParserParam, ProcessSeries) throws AxisBusyException`.
    pub fn peet_parser(
        &'static self,
        param: Arc<PeetParserParam>,
        process_series: Option<ProcessSeriesRef>,
    ) -> Result<String, AxisBusyException> {
        let background_process = self.base.start_background_process_command(
            param as Arc<dyn Command + Send + Sync>,
            true,
            AxisID::Only,
            Some(ProcessName::PEET_PARSER),
            None,
            process_series,
            false,
            true,
        )?;
        Ok(background_process.get_name())
    }

    /// Java `averageAll(AverageAllParam, ProcessSeries) throws AxisBusyException`.
    pub fn average_all(
        &'static self,
        param: Arc<AverageAllParam>,
        process_series: Option<ProcessSeriesRef>,
    ) -> Result<String, AxisBusyException> {
        let background_process = self.base.start_background_process_command(
            param as Arc<dyn Command + Send + Sync>,
            true,
            AxisID::Only,
            Some(ProcessName::AVERAGE_ALL),
            None,
            process_series,
            false,
            false,
        )?;
        Ok(background_process.get_name())
    }
}

impl BaseProcessManagerHooks for PeetProcessManager {
    /// Java override `postProcess(BackgroundProcess)`.
    fn post_process_background(&self, base: &BaseProcessManager, process: &BackgroundProcess) {
        base.post_process_background_base(process);
        let process_name = process.get_process_name();
        let process_details = process.get_command_details().cloned();
        let state = self.manager.get_state();
        let Some(process_name) = process_name else {
            return;
        };
        let manager = self.manager;
        if process_name == ProcessName::PEET_PARSER {
            let Some(process_details) = process_details else {
                return;
            };
            let Some(details) = process_details.get_process_details() else {
                return;
            };
            // Java unboxes the int; an unknown field throws there.
            if let Some(size) = details.get_int_value(&peet_parser_param::Fields::IterationListSize)
            {
                state.set_iteration_list_size(size);
            }
            event_queue::invoke_later(move || {
                let any: &dyn Any = &*process_details;
                if let Some(param) = any.downcast_ref::<PeetParserParam>() {
                    manager.log_message_loggable(Some(param as &dyn Loggable), Some(AxisID::Only));
                }
            });
        } else if process_name == ProcessName::AVERAGE_ALL {
            // Java dereferences processDetails unchecked; averageAll is always
            // started with its param.
            let Some(process_details) = process_details else {
                return;
            };
            let Some(details) = process_details.get_process_details() else {
                return;
            };
            if let Some(size) = details.get_int_value(&average_all_param::Fields::IterationListSize)
            {
                state.set_iteration_list_size(size);
            }
            event_queue::invoke_later(move || {
                let any: &dyn Any = &*process_details;
                if let Some(param) = any.downcast_ref::<AverageAllParam>() {
                    manager.log_message_loggable(Some(param as &dyn Loggable), Some(AxisID::Only));
                }
            });
        }
    }
}
