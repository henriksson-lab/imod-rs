//! `IMOD/Etomo/src/etomo/comscript/SerialSectionsComScriptManager.java`.
//!
//! Description: Stores and manages comscripts (`preblend.com`, `blend.com` and
//! `newst.com` of the Serial Sections interface).

use std::cell::RefCell;

use crate::imod::etomo::util::event_queue::ReentrantLock;

use super::blendmont_param::{self, BlendmontParam};
use super::com_script::ComScript;
use super::com_script_util::ComScriptUtil;
use super::command::Command;
use super::command_param::CommandParam;
use super::newst_param::NewstParam;
use super::set_env_param::{self, SetEnvParam};
use crate::imod::etomo::serial_sections_manager::SerialSectionsManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::file_type;

/// Java `rcsid`.
pub const RCSID: &str = "$Id:$";

/// Java `public final class SerialSectionsComScriptManager`.
pub struct SerialSectionsComScriptManager {
    /// Java private final `manager`.
    manager: &'static SerialSectionsManager,
    /// Java private `scriptPreblend`, initially null.  A `ComScript` holds `Rc`s and
    /// is not `Send`; the three scripts are only reached while `lock` is held.
    script_preblend: RefCell<Option<ComScript>>,
    /// Java private `scriptBlend`, initially null.
    script_blend: RefCell<Option<ComScript>>,
    /// Java private `scriptNewst`, initially null.
    script_newst: RefCell<Option<ComScript>>,
    /// Java's unguarded field access, made exclusive (as in `ToolsComScriptManager`).
    lock: ReentrantLock,
}

// SAFETY: the three `ComScript` cells (the only non-`Send` state) are touched only
// while the calling thread holds `lock`, so no two threads reach their `Rc`s at once,
// and every `Rc` a script creates stays inside it.
unsafe impl Send for SerialSectionsComScriptManager {}
unsafe impl Sync for SerialSectionsComScriptManager {}

impl SerialSectionsComScriptManager {
    /// Java `SerialSectionsComScriptManager(SerialSectionsManager)`.
    pub fn new(manager: &'static SerialSectionsManager) -> Self {
        Self {
            manager,
            script_preblend: RefCell::new(None),
            script_blend: RefCell::new(None),
            script_newst: RefCell::new(None),
            lock: ReentrantLock::new(),
        }
    }

    /// Java `loadPreblend(AxisID)`.
    pub fn load_preblend(&self, axis_id: AxisID) {
        let _lock = self.lock.lock();
        let script = ComScriptUtil::load_com_script_file_type(
            self.manager,
            &file_type::CLASS.preblend_comscript,
            axis_id,
            true,
            false,
            false,
        );
        *self.script_preblend.borrow_mut() = script;
    }

    /// The `getSetEnvParamFrom*` bodies: Java writes the same body out three times.
    fn get_set_env_param(
        &self,
        script: &RefCell<Option<ComScript>>,
        axis_id: AxisID,
        env_var: &str,
    ) -> Option<SetEnvParam> {
        let _lock = self.lock.lock();
        // Initialize a SetEnvParam object from the com script command
        // object
        let mut param = SetEnvParam::new(Some(env_var));
        // Assuming its the first setenv command in the comfile.
        if !ComScriptUtil::initialize_previous_command_required_option(
            self.manager,
            &mut param,
            script.borrow_mut().as_mut(),
            set_env_param::COMMAND_NAME,
            axis_id,
            true,
            None,
            true,
            false,
            false,
            false,
            Some(env_var),
        ) {
            return None;
        }
        Some(param)
    }

    /// Java `getSetEnvParamFromPreblend(AxisID, String)`.
    pub fn get_set_env_param_from_preblend(
        &self,
        axis_id: AxisID,
        env_var: &str,
    ) -> Option<SetEnvParam> {
        self.get_set_env_param(&self.script_preblend, axis_id, env_var)
    }

    /// The `getBlendmontParamFrom*` bodies.
    fn get_blendmont_param(
        &self,
        script: &RefCell<Option<ComScript>>,
        axis_id: AxisID,
        root_name: Option<&str>,
        mode: blendmont_param::Mode,
    ) -> BlendmontParam {
        let _lock = self.lock.lock();
        let mut param = BlendmontParam::new_with_mode(self.manager, root_name, axis_id, mode);
        ComScriptUtil::initialize(
            self.manager,
            &mut param,
            script.borrow_mut().as_mut(),
            blendmont_param::COMMAND_NAME,
            axis_id,
            false,
            false,
            true,
        );
        param
    }

    /// Java `getBlendmontParamFromPreblend(AxisID, String)`.
    pub fn get_blendmont_param_from_preblend(
        &self,
        axis_id: AxisID,
        root_name: Option<&str>,
    ) -> BlendmontParam {
        self.get_blendmont_param(
            &self.script_preblend,
            axis_id,
            root_name,
            blendmont_param::Mode::SerialSectionPreblend,
        )
    }

    /// The `save*(SetEnvParam, AxisID, String)` bodies.
    fn save_set_env(
        &self,
        script: &RefCell<Option<ComScript>>,
        param: &SetEnvParam,
        axis_id: AxisID,
        env_var: &str,
    ) {
        let _lock = self.lock.lock();
        ComScriptUtil::add_modify_command_required_option(
            self.manager,
            script.borrow_mut().as_mut(),
            param,
            set_env_param::COMMAND_NAME,
            axis_id,
            false,
            false,
            Some(env_var),
        );
    }

    /// The `save*(CommandParam, AxisID)` bodies.
    fn save_command(
        &self,
        script: &RefCell<Option<ComScript>>,
        param: &dyn CommandParam,
        command: &str,
        axis_id: AxisID,
    ) {
        let _lock = self.lock.lock();
        ComScriptUtil::modify_command(
            self.manager,
            script.borrow_mut().as_mut(),
            param,
            command,
            axis_id,
            true,
            false,
        );
    }

    /// Java `savePreblend(SetEnvParam, AxisID, String)`.
    pub fn save_preblend_set_env(&self, param: &SetEnvParam, axis_id: AxisID, env_var: &str) {
        self.save_set_env(&self.script_preblend, param, axis_id, env_var);
    }

    /// Java `savePreblend(BlendmontParam, AxisID)`.
    pub fn save_preblend(&self, param: &BlendmontParam, axis_id: AxisID) {
        self.save_command(
            &self.script_preblend,
            param,
            blendmont_param::COMMAND_NAME,
            axis_id,
        );
    }

    /// Java `loadBlend(AxisID)`.
    pub fn load_blend(&self, axis_id: AxisID) {
        let _lock = self.lock.lock();
        let script = ComScriptUtil::load_com_script_file_type(
            self.manager,
            &file_type::CLASS.blend_comscript,
            axis_id,
            true,
            false,
            false,
        );
        *self.script_blend.borrow_mut() = script;
    }

    /// Java `getSetEnvParamFromBlend(AxisID, String)`.
    pub fn get_set_env_param_from_blend(
        &self,
        axis_id: AxisID,
        env_var: &str,
    ) -> Option<SetEnvParam> {
        self.get_set_env_param(&self.script_blend, axis_id, env_var)
    }

    /// Java `getBlendmontParamFromBlend(AxisID, String)`.
    pub fn get_blendmont_param_from_blend(
        &self,
        axis_id: AxisID,
        root_name: Option<&str>,
    ) -> BlendmontParam {
        self.get_blendmont_param(
            &self.script_blend,
            axis_id,
            root_name,
            blendmont_param::Mode::SerialSectionBlend,
        )
    }

    /// Java `saveBlend(SetEnvParam, AxisID, String)`.
    pub fn save_blend_set_env(&self, param: &SetEnvParam, axis_id: AxisID, env_var: &str) {
        self.save_set_env(&self.script_blend, param, axis_id, env_var);
    }

    /// Java `saveBlend(BlendmontParam, AxisID)`.
    pub fn save_blend(&self, param: &BlendmontParam, axis_id: AxisID) {
        self.save_command(
            &self.script_blend,
            param,
            blendmont_param::COMMAND_NAME,
            axis_id,
        );
    }

    /// Java `loadNewst(AxisID)`.
    pub fn load_newst(&self, axis_id: AxisID) {
        let _lock = self.lock.lock();
        let script = ComScriptUtil::load_com_script_file_type(
            self.manager,
            &file_type::CLASS.newst_comscript,
            axis_id,
            true,
            false,
            false,
        );
        *self.script_newst.borrow_mut() = script;
    }

    /// Java `getSetEnvParamFromNewst(AxisID, String)`.
    pub fn get_set_env_param_from_newst(
        &self,
        axis_id: AxisID,
        env_var: &str,
    ) -> Option<SetEnvParam> {
        self.get_set_env_param(&self.script_newst, axis_id, env_var)
    }

    /// Java `getNewstackParam(AxisID, String)`.
    pub fn get_newstack_param(&self, axis_id: AxisID, _root_name: Option<&str>) -> NewstParam {
        let _lock = self.lock.lock();
        let mut param = NewstParam::get_color_instance(self.manager, axis_id);
        let command_name = param.get_command_name().unwrap_or_default();
        ComScriptUtil::initialize(
            self.manager,
            &mut param,
            self.script_newst.borrow_mut().as_mut(),
            &command_name,
            axis_id,
            false,
            false,
            true,
        );
        param
    }

    /// Java `saveNewst(SetEnvParam, AxisID, String)`.
    pub fn save_newst_set_env(&self, param: &SetEnvParam, axis_id: AxisID, env_var: &str) {
        self.save_set_env(&self.script_newst, param, axis_id, env_var);
    }

    /// Java `saveNewst(NewstParam, AxisID)`.
    pub fn save_newst(&self, param: &NewstParam, axis_id: AxisID) {
        let command_name = param.get_command_name().unwrap_or_default();
        self.save_command(&self.script_newst, param, &command_name, axis_id);
    }
}
