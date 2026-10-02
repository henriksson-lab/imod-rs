//! `IMOD/Etomo/src/etomo/process/Tilt3dFindProcessMonitor.java`.
//!
//! `Tilt3dFindProcessMonitor extends TiltProcessMonitor` and overrides only
//! `getTiltParam`.  Its state is held by the `TiltProcessMonitor` it extends
//! (`tilt_process_monitor.rs`), whose `get_tilt_param` dispatches here.

use super::file_size_process_monitor::FileSizeProcessMonitorOf;
use super::tilt_process_monitor::TiltProcessMonitor;
use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::comscript::const_tilt_param::ConstTiltParam;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::process_name::ProcessName;
use std::sync::Arc;

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `Tilt3dFindProcessMonitor`'s own fields.
pub struct Tilt3dFindProcessMonitor {
    /// Java field `tiltParam`.
    tilt_param: Arc<dyn ConstTiltParam + Send + Sync>,
}

impl Tilt3dFindProcessMonitor {
    /// Java `Tilt3dFindProcessMonitor(ApplicationManager, AxisID, ProcessName,
    /// ConstTiltParam)`.
    pub fn new(
        app_mgr: &'static ApplicationManager,
        id: AxisID,
        process_name: ProcessName,
        tilt_param: Arc<dyn ConstTiltParam + Send + Sync>,
    ) -> Arc<FileSizeProcessMonitorOf<TiltProcessMonitor>> {
        TiltProcessMonitor::new_subclass(
            app_mgr,
            id,
            process_name,
            Some(Tilt3dFindProcessMonitor { tilt_param }),
        )
    }

    /// Java `getTiltParam`.  Returns the instance of TiltParam that was passed
    /// into the constructor.
    pub fn get_tilt_param(&self) -> Arc<dyn ConstTiltParam + Send + Sync> {
        self.tilt_param.clone()
    }
}
