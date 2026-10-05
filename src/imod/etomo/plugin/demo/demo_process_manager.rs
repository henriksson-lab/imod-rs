//! `IMOD/Etomo/src/etomo/plugin/demo/DemoProcessManager.java`.
//!
//! Runs processes.  `DemoProcessManager extends BaseProcessManager`: the superclass is
//! embedded as `base` and this class installs itself as the base's
//! [`BaseProcessManagerHooks`] for its `postProcess(ComScriptProcess)` override.  It
//! shares the axis blocking information with the manager's own process manager (the
//! base class takes it from the manager).
//!
//! **Threads.**  `postProcess` runs on the process thread, and the Java calls
//! `pluginManager.updateValues`, which writes the demo panel's fields, from there.  The
//! plugin manager and its panel are event-dispatch-thread objects, so that call is
//! posted to the event dispatch thread.

use std::sync::Arc;

use super::demo_plugin_manager::DemoPluginManager;
use super::demo_process_name;
use super::etomo_plugin_demo_param::{self, EtomoPluginDemoParam};
use super::sleep_time::SleepTime;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::command::Command;
use crate::imod::etomo::process::base_process_manager::{
    AxisBusyException, BaseProcessManager, BaseProcessManagerHooks,
};
use crate::imod::etomo::process::com_script_process::ComScriptProcess;
use crate::imod::etomo::process::process_interface::{ProcessResultDisplayRef, ProcessSeriesRef};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::util::event_queue::{self, EdtRef};

/// Java `final class DemoProcessManager extends BaseProcessManager`.
pub struct DemoProcessManager {
    /// The `BaseProcessManager` superclass.
    pub base: BaseProcessManager,
    /// Java private final `axisID`.
    axis_id: AxisID,
    /// Java package-private final `pluginManager` (an event-dispatch-thread object; see
    /// the module comment).
    plugin_manager: Arc<EdtRef<DemoPluginManager>>,
}

impl DemoProcessManager {
    /// Java package-private `DemoProcessManager(BaseManager, AxisID,
    /// DemoPluginManager)`.  It lives as long as the dataset (the base class's start
    /// functions take `&'static self`).
    pub fn new(
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        plugin_manager: Arc<EdtRef<DemoPluginManager>>,
    ) -> &'static DemoProcessManager {
        let process_manager: &'static DemoProcessManager =
            Box::leak(Box::new(DemoProcessManager {
                base: BaseProcessManager::new(manager),
                axis_id,
                plugin_manager,
            }));
        process_manager.base.set_hooks(process_manager);
        process_manager
    }

    /// Java package-private `etomoPluginDemo(ProcessResultDisplay, ProcessSeries,
    /// EtomoPluginDemoParam) throws AxisBusyException`.
    pub fn etomo_plugin_demo(
        &'static self,
        process_result_display: Option<ProcessResultDisplayRef>,
        process_series: Option<ProcessSeriesRef>,
        param: EtomoPluginDemoParam,
    ) -> Result<String, AxisBusyException> {
        // Start the com script in the background
        // `startComScript(CommandDetails, ProcessMonitor, AxisID, ProcessResultDisplay,
        // ProcessSeries)`: the com script is the param's command line.
        let com_script_process = self.base.start_com_script_param(
            Arc::new(param) as Arc<dyn Command + Send + Sync>,
            true,
            None,
            self.axis_id,
            process_result_display,
            process_series,
            None,
        )?;
        Ok(com_script_process.get_name())
    }
}

impl BaseProcessManagerHooks for DemoProcessManager {
    /// Java protected override `postProcess(ComScriptProcess)`.
    fn post_process_com_script(&self, _base: &BaseProcessManager, script: &ComScriptProcess) {
        if script.get_process_name() == Some(*demo_process_name::DEMO) {
            let details = script
                .get_command_details()
                .and_then(|details| details.get_process_details());
            // `details != null ? details.getIntValue(Field.SLEEP_TIME) :
            // SleepTime.DEFAULT.getValue().getInt()`.
            let sleep_time = match details {
                Some(details) => details
                    .get_int_value(&etomo_plugin_demo_param::Field::SleepTime)
                    .unwrap_or_default(),
                None => SleepTime::DEFAULT
                    .value()
                    .map(|value| value.get_int())
                    .unwrap_or_default(),
            };
            let plugin_manager = Arc::clone(&self.plugin_manager);
            event_queue::invoke_later(move || {
                plugin_manager.get().update_values(sleep_time);
            });
        }
    }
}
