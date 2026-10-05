//! `IMOD/Etomo/src/etomo/plugin/demo/DemoComScriptManager.java`.
//!
//! Stores and manages comscripts.  This class just uses a utility class, so its
//! functionality could be handled by any class.  An event-dispatch-thread object owned
//! by the demo plugin manager; the loaded script sits in a `RefCell`.

use std::cell::RefCell;

use super::demo_file_type;
use super::etomo_plugin_demo_param::{self, EtomoPluginDemoParam};
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::com_script::ComScript;
use crate::imod::etomo::comscript::com_script_util::ComScriptUtil;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;

/// Java `final class DemoComScriptManager`.
pub struct DemoComScriptManager {
    /// Java private final `manager`.
    manager: &'static dyn BaseManager,
    /// Java private final `axisID`.
    axis_id: Option<AxisID>,
    /// Java private final `axisType` (never read in the Java either).
    #[allow(dead_code)]
    axis_type: Option<AxisType>,
    /// Java private `scriptDemo`, initialised to null.
    script_demo: RefCell<Option<ComScript>>,
}

impl DemoComScriptManager {
    /// Java `DemoComScriptManager(BaseManager, AxisID, AxisType)`.
    pub fn new(
        manager: &'static dyn BaseManager,
        axis_id: Option<AxisID>,
        axis_type: Option<AxisType>,
    ) -> DemoComScriptManager {
        DemoComScriptManager {
            manager,
            axis_id,
            axis_type,
            script_demo: RefCell::new(None),
        }
    }

    /// Java package-private `loadDemo()`.
    pub fn load_demo(&self) -> bool {
        // Java passes the possibly-null axisID; the per-axis file name and the loader
        // take a real axis, and the plugin always has one (`init` receives the
        // expert's).
        let axis_id = self.axis_id.unwrap_or(AxisID::Only);
        // Assign the new ComScriptObject object to the appropriate reference
        let script = ComScriptUtil::load_com_script_file_name(
            self.manager,
            demo_file_type::DEMO_COMSCRIPT
                .get_file_name(Some(self.manager), Some(axis_id))
                .as_deref(),
            axis_id,
            true,
            false,
            false,
            false,
        );
        let loaded = script.is_some();
        *self.script_demo.borrow_mut() = script;
        loaded
    }

    /// Java package-private `getEtomoDemoPluginParam()`.
    pub fn get_etomo_demo_plugin_param(&self) -> EtomoPluginDemoParam {
        let axis_id = self.axis_id.unwrap_or(AxisID::Only);
        // Initialize a DemosetupParam object from the com script command object
        let mut param = EtomoPluginDemoParam::new(axis_id);
        ComScriptUtil::initialize(
            self.manager,
            &mut param,
            self.script_demo.borrow_mut().as_mut(),
            &etomo_plugin_demo_param::command_process_name().to_string(),
            axis_id,
            true,
            true,
            true,
        );
        param
    }

    /// Java package-private `saveDemo(EtomoPluginDemoParam, AxisID)`.
    pub fn save_demo(&self, param: &EtomoPluginDemoParam, axis_id: AxisID) {
        // Get a reference to the appropriate script object
        let mut script = self.script_demo.borrow_mut();
        ComScriptUtil::modify_command(
            self.manager,
            script.as_mut(),
            param,
            &etomo_plugin_demo_param::command_process_name().to_string(),
            axis_id,
            false,
            false,
        );
    }
}
