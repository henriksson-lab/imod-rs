//! `IMOD/Etomo/src/etomo/process/PrenewstProcessMonitor.java`.
//!
//! A `FileSizeProcessMonitor` subclass; see the module comment of
//! `file_size_process_monitor.rs` for how subclasses are represented.

use super::file_size_process_monitor::{
    CalcFileSizeError, FileSizeProcessMonitor, FileSizeProcessMonitorImpl, FileSizeProcessMonitorOf,
};
use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::comscript::const_newst_param::ConstNewstParam;
use crate::imod::etomo::comscript::newst_param::NewstParam;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::file_type;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::util::mrc_header::MRCHeader;
use std::sync::Arc;
use std::sync::atomic::Ordering;

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `PrenewstProcessMonitor`'s own fields.
pub struct PrenewstProcessMonitor {
    /// Java field `applicationManager`.
    application_manager: &'static ApplicationManager,
}

impl PrenewstProcessMonitor {
    /// Java `PrenewstProcessMonitor(ApplicationManager, AxisID)`.
    pub fn new(
        app_mgr: &'static ApplicationManager,
        id: AxisID,
    ) -> Arc<FileSizeProcessMonitorOf<PrenewstProcessMonitor>> {
        FileSizeProcessMonitorOf::new(
            app_mgr,
            id,
            ProcessName::PRENEWST,
            PrenewstProcessMonitor {
                application_manager: app_mgr,
            },
        )
    }
}

impl FileSizeProcessMonitorImpl for PrenewstProcessMonitor {
    /// Java `calcFileSize`.  Calculate the expected files size in kBytes from
    /// the size of the current stack and newstack binBy parameter.  The
    /// assumption for prenewst.com is that the mode is always 0 (1 byte per
    /// pixel).
    fn calc_file_size(&self, base: &FileSizeProcessMonitor) -> Result<bool, CalcFileSizeError> {
        let manager = base.manager;
        let axis_id = base.axis_id;
        let mut n_x: i32;
        let mut n_y: i32;
        let n_z: i32;
        let mut mode_bytes: i32 = 1;

        // Get the header from the raw stack to calculate the aligned stack size
        let raw_stack = MRCHeader::get_instance_from_file_type(
            manager,
            Some(axis_id),
            &file_type::CLASS.raw_stack,
        )
        .unwrap();
        let mut raw_stack = raw_stack.borrow_mut();
        if !raw_stack
            .read_with_manager(manager)
            .map_err(CalcFileSizeError::from)?
        {
            return Ok(false);
        }

        n_x = raw_stack.get_n_rows();
        n_y = raw_stack.get_n_columns();
        n_z = raw_stack.get_n_sections();

        // Get the binByFactor from prenewst.com script
        let com_script_manager = self.application_manager.get_com_script_manager();
        com_script_manager.load_prenewst(axis_id);
        let prenewst_param = com_script_manager.get_prenewst_param(axis_id);
        let bin_by = prenewst_param.get_bin_by_factor();
        // If the bin by factor is unspecified it defaults to 1
        if bin_by > 1 {
            n_x /= bin_by;
            n_y /= bin_by;
        }
        if prenewst_param.get_mode_to_output()
            != crate::imod::etomo::comscript::newst_param::DATA_MODE_BYTE
        {
            mode_bytes = base.get_mode_bytes(raw_stack.get_mode())?;
        }
        let file_size: i64 = 1024i64.wrapping_add(
            (n_x as i64)
                .wrapping_mul(n_y as i64)
                .wrapping_mul(n_z as i64)
                .wrapping_mul(mode_bytes as i64),
        );
        let n_k_bytes = (file_size / 1024) as i32;
        base.n_k_bytes.store(n_k_bytes, Ordering::SeqCst);
        let title = self.get_title(base);
        manager.post_main_panel(Box::new(move |panel| {
            panel.set_progress_bar_string_int_boolean_axis_id(
                Some(&title),
                n_k_bytes,
                false,
                axis_id,
            );
        }));
        Ok(true)
    }

    /// Java `getTitle`.
    fn get_title(&self, _base: &FileSizeProcessMonitor) -> String {
        "Creating coarse stack".to_string()
    }

    /// Java `reloadWatchedFile`.
    fn reload_watched_file(&self, base: &FileSizeProcessMonitor) {
        // Create a file object describing the file to be monitored
        *base.watched_file.lock().unwrap() = file_type::CLASS
            .prealigned_stack
            .get_file(Some(base.manager), Some(base.axis_id));
    }
}
