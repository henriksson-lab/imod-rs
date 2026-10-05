//! `IMOD/Etomo/src/etomo/process/CtfCorrectionMonitor.java`.
//!
//! A `FileSizeProcessMonitor` subclass; see the module comment of
//! `file_size_process_monitor.rs` for how subclasses are represented.

use super::file_size_process_monitor::{
    CalcFileSizeError, FileSizeProcessMonitor, FileSizeProcessMonitorImpl, FileSizeProcessMonitorOf,
};
use crate::imod::etomo::application_manager::{ApplicationManager, ComScriptManagerGuard};
use crate::imod::etomo::comscript::com_script_manager::ComScriptManager;
use crate::imod::etomo::comscript::const_newst_param::ConstNewstParam;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::view_type::ViewType;
use crate::imod::etomo::util::dataset_files;
use crate::imod::etomo::util::mrc_header::MRCHeader;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `CtfCorrectionMonitor`'s own fields.
pub struct CtfCorrectionMonitor {
    /// Java field `applicationManager`.
    application_manager: &'static ApplicationManager,
    // Java field `comScriptManager` caches `applicationManager.getComScriptManager()`;
    // the translation takes the guarded manager afresh on each call instead,
    // since a cached reference would outlive its lock (see
    // `ApplicationManager::get_com_script_manager`).
}

impl CtfCorrectionMonitor {
    /// Java `CtfCorrectionMonitor(ApplicationManager, AxisID)`.
    pub fn new(
        app_mgr: &'static ApplicationManager,
        id: AxisID,
    ) -> Arc<FileSizeProcessMonitorOf<CtfCorrectionMonitor>> {
        let monitor = FileSizeProcessMonitorOf::new(
            app_mgr,
            id,
            ProcessName::CTF_CORRECTION,
            CtfCorrectionMonitor {
                application_manager: app_mgr,
            },
        );
        monitor.set_find_watched_file_name(false);
        monitor
    }

    /// Java private `loadComScriptManager`.
    fn load_com_script_manager(&self) -> ComScriptManagerGuard {
        self.application_manager.get_com_script_manager()
    }
}

impl FileSizeProcessMonitorImpl for CtfCorrectionMonitor {
    /// Java `calcFileSize`.
    fn calc_file_size(&self, base: &FileSizeProcessMonitor) -> Result<bool, CalcFileSizeError> {
        let manager = base.manager;
        let axis_id = base.axis_id;
        let n_x: f64;
        let n_y: f64;
        let n_z: f64;
        let mode_bytes: f64;

        // Get the depth, mode, any mods to the X and Y size from the tilt
        // command script and the input and output filenames.
        let com_script_manager = self.load_com_script_manager();
        let property_user_dir = manager.get_property_user_dir();
        let output_filename: String = if manager.get_view_type() == ViewType::Montage {
            com_script_manager.load_blend(axis_id);
            let blendmont_param = com_script_manager.get_blend_param(axis_id);

            // Get the header from the raw stack to calculate the aligned stack stize
            property_user_dir
                .clone()
                .unwrap_or_else(|| "null".to_string())
                + "/"
                + &blendmont_param
                    .get_image_output_file()
                    .unwrap_or_else(|| "null".to_string())
        } else {
            com_script_manager.load_newst(axis_id);
            let newst_param = com_script_manager.get_newst_com_newst_param(axis_id);

            // Get the header from the raw stack to calculate the aligned stack stize
            property_user_dir
                .clone()
                .unwrap_or_else(|| "null".to_string())
                + "/"
                + &newst_param.get_output_file()
        };
        let output_header = MRCHeader::get_instance_in_dir(
            property_user_dir.as_deref(),
            Some(&output_filename),
            Some(axis_id),
        )
        .unwrap();
        let mut output_header = output_header.borrow_mut();
        if !output_header
            .read_with_manager(manager)
            .map_err(CalcFileSizeError::from)?
        {
            return Ok(false);
        }
        n_x = output_header.get_n_rows() as f64;
        n_y = output_header.get_n_columns() as f64;
        n_z = output_header.get_n_sections() as f64;
        mode_bytes = base.get_mode_bytes(output_header.get_mode())? as f64;
        // Assumption: newst will write the output file with the same mode as the
        // the input file
        let file_size = 1024.0f64 + n_x * n_y * n_z * mode_bytes;
        let n_k_bytes = (file_size / 1024.0) as i32;
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
        "Running CTF Correction".to_string()
    }

    /// Java `reloadWatchedFile`.
    fn reload_watched_file(&self, base: &FileSizeProcessMonitor) {
        *base.watched_file.lock().unwrap() = Some(dataset_files::get_ctf_correction_file(
            self.application_manager,
            Some(base.axis_id),
        ));
    }
}
