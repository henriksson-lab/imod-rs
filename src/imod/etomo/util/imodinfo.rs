//! `IMOD/Etomo/src/etomo/util/Imodinfo.java`.
//!
//! Runs the external `imodinfo -h` on a model through an `etomo.process.SystemProgram`
//! and reports whether the model is a patch tracking model.

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::process::system_program::SystemProgram;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::java_lang_string_trim;
use crate::imod::etomo::r#type::file_type::FileType;
use std::sync::Arc;

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `Imodinfo`.
pub struct Imodinfo {
    /// Java private final field `fileType`.
    file_type: Arc<FileType>,
    /// Java private field `patchTracking`, initialised to false.
    patch_tracking: bool,
}

impl Imodinfo {
    /// Java `Imodinfo(FileType)`.
    pub fn new(file_type: &Arc<FileType>) -> Imodinfo {
        Imodinfo {
            file_type: Arc::clone(file_type),
            patch_tracking: false,
        }
    }

    /// Java `isPatchTracking(BaseManager, AxisID)`.
    pub fn is_patch_tracking(
        &mut self,
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
    ) -> bool {
        self.run(manager, axis_id);
        self.patch_tracking
    }

    /// Java private `run(BaseManager, AxisID)`.
    fn run(&mut self, manager: &'static dyn BaseManager, axis_id: AxisID) {
        let system_program = SystemProgram::new_array(
            Some(manager),
            manager.get_property_user_dir(),
            Some(vec![
                "imodinfo".to_string(),
                "-h".to_string(),
                self.file_type
                    .get_file_name(Some(manager), Some(axis_id))
                    .unwrap_or_else(|| "null".to_string()),
            ]),
            axis_id,
        );
        system_program.run();
        let stdout = system_program.get_std_output();
        if let Some(stdout) = stdout {
            for i in 0..stdout.len() {
                if java_lang_string_trim(&stdout[i]).starts_with("# NAME") {
                    self.patch_tracking = stdout[i].contains("Patch Tracking Model");
                }
            }
        }
    }
}
