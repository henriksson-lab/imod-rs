//! `IMOD/Etomo/src/etomo/comscript/JoinComscriptManager.java`.
//!
//! Loads and saves the join's com script (`joinwarp2model.com`).
//!
//! The manager is shared with process threads like the reconstruction
//! `ComScriptManager`; the `ComScript` holds `Rc`s, so it is only reached while the
//! re-entrant lock is held (the `tools_com_script_manager.rs` arrangement).

use std::cell::RefCell;

use super::com_script::ComScript;
use super::com_script_util::ComScriptUtil;
use super::joinwarp2model_param::{self, Joinwarp2modelParam};
use crate::imod::etomo::join_manager::JoinManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::file_type;
use crate::imod::etomo::util::event_queue::ReentrantLock;

/// Java `public class JoinComscriptManager`.
pub struct JoinComscriptManager {
    /// Java private final `manager`.
    manager: &'static JoinManager,
    /// Java private `scriptJoinWarp2Model`, initially null.
    script_join_warp_2_model: RefCell<Option<ComScript>>,
    /// Java's unguarded field access, made exclusive.
    lock: ReentrantLock,
}

// SAFETY: the `ComScript` cell (the only non-`Send` state) is touched only while the
// calling thread holds `lock`, so no two threads reach its `Rc`s at once, and every
// `Rc` a script creates stays inside it.
unsafe impl Send for JoinComscriptManager {}
unsafe impl Sync for JoinComscriptManager {}

impl JoinComscriptManager {
    /// Java `JoinComscriptManager(JoinManager)`.
    pub fn new(manager: &'static JoinManager) -> JoinComscriptManager {
        JoinComscriptManager {
            manager,
            script_join_warp_2_model: RefCell::new(None),
            lock: ReentrantLock::new(),
        }
    }

    /// Java `loadJoinWarp2Model(AxisID)`.
    pub fn load_join_warp_2_model(&self, axis_id: AxisID) {
        let _lock = self.lock.lock();
        let script = ComScriptUtil::load_com_script_file_type(
            self.manager,
            &file_type::CLASS.join_warp_2_model_comscript,
            axis_id,
            true,
            false,
            false,
        );
        *self.script_join_warp_2_model.borrow_mut() = script;
    }

    /// Java `getJoinwarp2modelParam(AxisID)`.
    pub fn get_joinwarp2model_param(&self, axis_id: AxisID) -> Option<Joinwarp2modelParam> {
        let _lock = self.lock.lock();
        // Initialize a SetEnvParam object from the com script command
        // object
        let mut param = Joinwarp2modelParam::new(self.manager);
        // Assuming its the first setenv command in the comfile.
        if !ComScriptUtil::initialize_previous_command_required_option(
            self.manager,
            &mut param,
            self.script_join_warp_2_model.borrow_mut().as_mut(),
            joinwarp2model_param::COMMAND_NAME,
            axis_id,
            true,
            None,
            true,
            false,
            false,
            false,
            None,
        ) {
            return None;
        }
        Some(param)
    }

    /// Java `saveJoinwarp2model(Joinwarp2modelParam, AxisID)`.
    pub fn save_joinwarp2model(&self, param: &Joinwarp2modelParam, axis_id: AxisID) {
        let _lock = self.lock.lock();
        ComScriptUtil::add_modify_command_required_option(
            self.manager,
            self.script_join_warp_2_model.borrow_mut().as_mut(),
            param,
            joinwarp2model_param::COMMAND_NAME,
            axis_id,
            false,
            false,
            None,
        );
    }
}
