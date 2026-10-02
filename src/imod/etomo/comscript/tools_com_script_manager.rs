//! `IMOD/Etomo/src/etomo/comscript/ToolsComScriptManager.java`.
//!
//! Description: Stores and manages comscripts (for the Tools interface).
//!
//! **Java `null` axis.**  The align-frames members pass a null `AxisID` to
//! `ComScriptUtil`; the Tools interface is single axis and the translated
//! `ComScriptUtil` takes an axis by value, so they pass `AxisID::Only`.

use std::cell::RefCell;
use std::path::Path;

use crate::imod::etomo::util::event_queue::ReentrantLock;

use crate::imod::etomo::comscript::align_frames_param::{self, AlignFramesParam};
use crate::imod::etomo::comscript::com_script::ComScript;
use crate::imod::etomo::comscript::com_script_util::ComScriptUtil;
use crate::imod::etomo::comscript::warp_vol_param::{self, WarpVolParam};
use crate::imod::etomo::tools_manager::ToolsManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::file_type;

/// Java `public final class ToolsComScriptManager`.
pub struct ToolsComScriptManager {
    /// Java private final `manager`.
    manager: &'static ToolsManager,
    /// Java private `scriptFlatten`, initially null.  A `ComScript` holds
    /// `Rc`s and is not `Send`; it is only reached while `lock` is held.
    script_flatten: RefCell<Option<ComScript>>,
    /// Java private `scriptAlignFramesInput`, initially null.
    script_align_frames_input: RefCell<Option<ComScript>>,
    /// Java private `scriptAlignFramesOutput`, initially null.
    script_align_frames_output: RefCell<Option<ComScript>>,
    /// Java's unguarded field access, made exclusive: the same re-entrant lock the
    /// reconstruction `ComScriptManager` is reached through.
    lock: ReentrantLock,
}

// SAFETY: the three `ComScript` cells (the only non-`Send` state) are touched
// only while the calling thread holds `lock`, so no two threads reach their
// `Rc`s at once, and every `Rc` a script creates stays inside it.
unsafe impl Send for ToolsComScriptManager {}
unsafe impl Sync for ToolsComScriptManager {}

impl ToolsComScriptManager {
    /// Java `ToolsComScriptManager(ToolsManager)`.
    pub fn new(manager: &'static ToolsManager) -> Self {
        Self {
            manager,
            script_flatten: RefCell::new(None),
            script_align_frames_input: RefCell::new(None),
            script_align_frames_output: RefCell::new(None),
            lock: ReentrantLock::new(),
        }
    }

    /// Java `loadFlatten(AxisID)`.
    pub fn load_flatten(&self, axis_id: AxisID) {
        let _lock = self.lock.lock();
        let script = ComScriptUtil::load_com_script_file_type(
            self.manager,
            &file_type::CLASS.flatten_tool_comscript,
            axis_id,
            true,
            false,
            false,
        );
        *self.script_flatten.borrow_mut() = script;
    }

    /// Java `loadAlignFramesInput(File, boolean)`.
    pub fn load_align_frames_input(&self, com_file: &Path, required: bool) -> bool {
        let _lock = self.lock.lock();
        let parent = com_file
            .parent()
            .map(|parent| parent.to_string_lossy().into_owned());
        let name = com_file
            .file_name()
            .map(|name| name.to_string_lossy().into_owned());
        let script = ComScriptUtil::load_com_script(
            self.manager,
            parent.as_deref(),
            name.as_deref(),
            AxisID::Only,
            true,
            required,
            false,
            false,
        );
        *self.script_align_frames_input.borrow_mut() = script;
        self.script_align_frames_input.borrow().is_some()
    }

    // public boolean loadAlignFramesOutput(File comFile, boolean required) {
    // scriptAlignFramesOutput = ComScriptUtil.loadComScript(manager,
    // comFile.getAbsolutePath(), null, true, required, false, false);
    // return scriptAlignFramesOutput != null;
    // }

    /// Java `loadAlignFramesOutput(File, boolean)`.
    pub fn load_align_frames_output(&self, com_file: &Path, required: bool) -> bool {
        let _lock = self.lock.lock();
        let parent = com_file
            .parent()
            .map(|parent| parent.to_string_lossy().into_owned());
        let name = com_file
            .file_name()
            .map(|name| name.to_string_lossy().into_owned());
        let script = ComScriptUtil::load_com_script(
            self.manager,
            parent.as_deref(),
            name.as_deref(),
            AxisID::Only,
            true,
            required,
            false,
            false,
        );
        *self.script_align_frames_output.borrow_mut() = script;
        self.script_align_frames_output.borrow().is_some()
    }

    /// Java `resetAlignFramesOutput()`.
    pub fn reset_align_frames_output(&self) {
        let _lock = self.lock.lock();
        *self.script_align_frames_output.borrow_mut() = None;
    }

    /// Java `isWarpVolParamInFlatten(AxisID)`.
    ///
    /// Fixed in translation: `loadComScript` returns null when the com file cannot
    /// be parsed, and Java's `.isCommandLoaded()` then throws NullPointerException;
    /// a script that did not load answers false (as in `ComScriptManager`).
    pub fn is_warp_vol_param_in_flatten(&self, axis_id: AxisID) -> bool {
        match ComScriptUtil::load_com_script_file_type(
            self.manager,
            &file_type::CLASS.flatten_tool_comscript,
            axis_id,
            true,
            false,
            false,
        ) {
            None => false,
            Some(com_script) => com_script.is_command_loaded(),
        }
    }

    /// Java `getWarpVolParamFromFlatten(AxisID)`.
    pub fn get_warp_vol_param_from_flatten(&self, axis_id: AxisID) -> WarpVolParam {
        // Initialize a WarpVolParam object from the com script command
        // object
        let _lock = self.lock.lock();
        let mut param = WarpVolParam::new(self.manager, axis_id, Some(warp_vol_param::Mode::Tools));
        ComScriptUtil::initialize(
            self.manager,
            &mut param,
            self.script_flatten.borrow_mut().as_mut(),
            warp_vol_param::COMMAND,
            axis_id,
            false,
            false,
            true,
        );
        param
    }

    /// Java `getAlignFramesInputParam()`.
    pub fn get_align_frames_input_param(&self) -> AlignFramesParam {
        let _lock = self.lock.lock();
        let mut param = AlignFramesParam::new(self.manager, None);
        ComScriptUtil::initialize(
            self.manager,
            &mut param,
            self.script_align_frames_input.borrow_mut().as_mut(),
            align_frames_param::COMMAND,
            AxisID::Only,
            false,
            false,
            true,
        );
        param
    }

    /// Java `getAlignFramesOutputParam(String)`.
    pub fn get_align_frames_output_param(&self, com_filename: Option<&str>) -> AlignFramesParam {
        let _lock = self.lock.lock();
        let mut param = AlignFramesParam::new(self.manager, com_filename);
        ComScriptUtil::initialize(
            self.manager,
            &mut param,
            self.script_align_frames_output.borrow_mut().as_mut(),
            align_frames_param::COMMAND,
            AxisID::Only,
            false,
            false,
            true,
        );
        param
    }

    /// Java `saveFlatten(WarpVolParam, AxisID)`.
    /// Save the WarpVolParam command to the flatten com script.
    pub fn save_flatten(&self, param: &WarpVolParam, axis_id: AxisID) {
        let _lock = self.lock.lock();
        ComScriptUtil::modify_command(
            self.manager,
            self.script_flatten.borrow_mut().as_mut(),
            param,
            warp_vol_param::COMMAND,
            axis_id,
            true,
            false,
        );
    }

    /// Java `saveAlignFramesOutput(AlignFramesParam)`.
    pub fn save_align_frames_output(&self, param: &AlignFramesParam) {
        let _lock = self.lock.lock();
        ComScriptUtil::modify_command(
            self.manager,
            self.script_align_frames_output.borrow_mut().as_mut(),
            param,
            align_frames_param::COMMAND,
            AxisID::Only,
            true,
            false,
        );
    }
}
