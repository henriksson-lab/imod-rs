//! `IMOD/Etomo/src/etomo/comscript/ToolsComScriptManager.java`.
//!
//! `ComScript`, `ComScriptUtil`, `WarpVolParam`, and `AlignFramesParam` are
//! declared in source units that have not yet crossed the translation frontier.
//! Their fields retain Java's initial `null` representation rather than inventing a
//! second comscript parser or parameter model in this coordinator.
#![allow(dead_code)]

use std::convert::Infallible;
use std::path::Path;
use std::sync::Mutex;

use crate::imod::etomo::comscript::com_script_file::ComScriptFile;
use crate::imod::etomo::tools_manager::ToolsManager;
use crate::imod::etomo::r#type::axis_id::AxisID;

/// Java final `ToolsComScriptManager`.
pub struct ToolsComScriptManager {
    /// Java final `manager`.
    manager: &'static ToolsManager,
    /// Java `scriptFlatten`, initially null.
    script_flatten: Mutex<Option<Infallible>>,
    /// Java `scriptAlignFramesInput`, initially null.
    script_align_frames_input: Mutex<Option<Infallible>>,
    /// Java `scriptAlignFramesOutput`, initially null.
    script_align_frames_output: Mutex<Option<Infallible>>,
    /// Typed COM documents used by the translated manager workflow.  The
    /// parameter objects can be layered on these documents as they land.
    flatten_document: Mutex<Option<ComScriptFile>>,
    align_frames_input_document: Mutex<Option<ComScriptFile>>,
    align_frames_output_document: Mutex<Option<ComScriptFile>>,
}

impl ToolsComScriptManager {
    /// Java `ToolsComScriptManager(ToolsManager)`.
    pub fn new(manager: &'static ToolsManager) -> Self {
        Self {
            manager,
            script_flatten: Mutex::new(None),
            script_align_frames_input: Mutex::new(None),
            script_align_frames_output: Mutex::new(None),
            flatten_document: Mutex::new(None),
            align_frames_input_document: Mutex::new(None),
            align_frames_output_document: Mutex::new(None),
        }
    }

    /// Java `loadFlatten(AxisID)`.
    pub fn load_flatten(&self, axis_id: AxisID) {
        let _ = (self.manager, axis_id);
        // TODO(unit): ComScriptUtil.java and ComScript.java.
        *self.script_flatten.lock().unwrap() = None;
    }

    /// Java `loadAlignFramesInput(File, boolean)`.
    pub fn load_align_frames_input(&self, com_file: &Path, required: bool) -> bool {
        let _ = (self.manager, required);
        *self.align_frames_input_document.lock().unwrap() = ComScriptFile::load(com_file).ok();
        *self.script_align_frames_input.lock().unwrap() = None;
        self.align_frames_input_document.lock().unwrap().is_some()
    }

    /// Java `loadAlignFramesOutput(File, boolean)`.
    pub fn load_align_frames_output(&self, com_file: &Path, required: bool) -> bool {
        let _ = (self.manager, required);
        *self.align_frames_output_document.lock().unwrap() = ComScriptFile::load(com_file).ok();
        *self.script_align_frames_output.lock().unwrap() = None;
        self.align_frames_output_document.lock().unwrap().is_some()
    }

    /// Java `resetAlignFramesOutput()`.
    pub fn reset_align_frames_output(&self) {
        *self.script_align_frames_output.lock().unwrap() = None;
        *self.align_frames_output_document.lock().unwrap() = None;
    }

    pub fn align_frames_input_document(&self) -> Option<ComScriptFile> {
        self.align_frames_input_document.lock().unwrap().clone()
    }
    pub fn align_frames_output_document(&self) -> Option<ComScriptFile> {
        self.align_frames_output_document.lock().unwrap().clone()
    }

    /// Java `isWarpVolParamInFlatten(AxisID)`.
    pub fn is_warp_vol_param_in_flatten(&self, axis_id: AxisID) -> bool {
        let _ = (self.manager, axis_id);
        // TODO(unit): ComScriptUtil.loadComScript and ComScript.isCommandLoaded.
        false
    }

    /// Java `getWarpVolParamFromFlatten(AxisID)`.
    pub fn get_warp_vol_param_from_flatten(&self, axis_id: AxisID) -> Option<Infallible> {
        let _ = (
            self.manager,
            axis_id,
            self.script_flatten.lock().unwrap().is_some(),
        );
        // TODO(unit): WarpVolParam.java and ComScriptUtil.initialize.
        None
    }

    /// Java `getAlignFramesInputParam()`.
    pub fn get_align_frames_input_param(&self) -> Option<Infallible> {
        let _ = (
            self.manager,
            self.script_align_frames_input.lock().unwrap().is_some(),
        );
        // TODO(unit): AlignFramesParam.java and ComScriptUtil.initialize.
        None
    }

    /// Java `getAlignFramesOutputParam(String)`.
    pub fn get_align_frames_output_param(&self, com_filename: Option<&str>) -> Option<Infallible> {
        let _ = (
            self.manager,
            com_filename,
            self.script_align_frames_output.lock().unwrap().is_some(),
        );
        // TODO(unit): AlignFramesParam.java and ComScriptUtil.initialize.
        None
    }

    /// Java `saveFlatten(WarpVolParam, AxisID)`.
    pub fn save_flatten(&self, param: Option<Infallible>, axis_id: AxisID) {
        let _ = (
            self.manager,
            param,
            axis_id,
            self.script_flatten.lock().unwrap().is_some(),
        );
        // TODO(unit): ComScriptUtil.modifyCommand and WarpVolParam.java.
    }

    /// Java `saveAlignFramesOutput(AlignFramesParam)`.
    pub fn save_align_frames_output(&self, param: Option<Infallible>) {
        let _ = (
            self.manager,
            param,
            self.script_align_frames_output.lock().unwrap().is_some(),
        );
        // TODO(unit): ComScriptUtil.modifyCommand and AlignFramesParam.java.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::etomo::ui::swing::etomo_menu::ToolType;

    #[test]
    fn reset_align_frames_output_retains_java_null_state() {
        let manager = ToolsManager::new(ToolType::AlignFrames);
        let manager = ToolsComScriptManager::new(manager);
        assert!(!manager.load_align_frames_output(Path::new("alignframes.com"), false));
        manager.reset_align_frames_output();
        assert!(
            manager
                .get_align_frames_output_param(Some("alignframes.com"))
                .is_none()
        );
    }
}
