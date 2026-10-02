//! `IMOD/Etomo/src/etomo/process/MtffilterProcessMonitor.java`.
//!
//! A `FileSizeProcessMonitor` subclass; see the module comment of
//! `file_size_process_monitor.rs` for how subclasses are represented.

use super::file_size_process_monitor::{
    CalcFileSizeError, FileSizeProcessMonitor, FileSizeProcessMonitorImpl, FileSizeProcessMonitorOf,
};
use crate::imod::etomo::application_manager::{ApplicationManager, ComScriptManagerGuard};
use crate::imod::etomo::comscript::com_script_manager::ComScriptManager;
use crate::imod::etomo::comscript::const_mtf_filter_param::ConstMTFFilterParam;
use crate::imod::etomo::comscript::const_newst_param::ConstNewstParam;
use crate::imod::etomo::comscript::mtf_filter_param::MTFFilterParam;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::view_type::ViewType;
use crate::imod::etomo::util::mrc_header::MRCHeader;
use crate::imod::etomo::util::utilities;
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `MtffilterProcessMonitor`'s own fields.
pub struct MtffilterProcessMonitor {
    /// Java field `applicationManager`.
    application_manager: &'static ApplicationManager,
    /// Java field `mtfFilterParam`.
    mtf_filter_param: Mutex<Option<Arc<MTFFilterParam>>>,
    // Java field `comScriptManager` caches `applicationManager.getComScriptManager()`;
    // the translation takes the guarded manager afresh on each call instead,
    // since a cached reference would outlive its lock (see
    // `ApplicationManager::get_com_script_manager`).
}

impl MtffilterProcessMonitor {
    /// Java `MtffilterProcessMonitor(ApplicationManager, AxisID)`.
    pub fn new(
        app_mgr: &'static ApplicationManager,
        id: AxisID,
    ) -> Arc<FileSizeProcessMonitorOf<MtffilterProcessMonitor>> {
        FileSizeProcessMonitorOf::new(
            app_mgr,
            id,
            ProcessName::MTFFILTER,
            MtffilterProcessMonitor {
                application_manager: app_mgr,
                mtf_filter_param: Mutex::new(None),
            },
        )
    }

    /// Java private `loadComScriptManager`.
    fn load_com_script_manager(&self) -> ComScriptManagerGuard {
        self.application_manager.get_com_script_manager()
    }

    /// Java private `loadMtfFilterParam`.
    fn load_mtf_filter_param(&self, axis_id: AxisID) -> Arc<MTFFilterParam> {
        let mut mtf_filter_param = self.mtf_filter_param.lock().unwrap();
        if let Some(mtf_filter_param) = &*mtf_filter_param {
            return mtf_filter_param.clone();
        }
        let com_script_manager = self.load_com_script_manager();
        com_script_manager.load_mtf_filter(axis_id);
        let param = Arc::new(com_script_manager.get_mtf_filter_param(axis_id));
        *mtf_filter_param = Some(param.clone());
        param
    }
}

impl FileSizeProcessMonitorImpl for MtffilterProcessMonitor {
    /// Java `calcFileSize`.
    fn calc_file_size(&self, base: &FileSizeProcessMonitor) -> Result<bool, CalcFileSizeError> {
        let manager = base.manager;
        let axis_id = base.axis_id;
        let n_x: f64;
        let n_y: f64;
        let mut n_z: f64;
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
            .map_err(CalcFileSizeError::Io)?
        {
            return Ok(false);
        }
        n_x = output_header.get_n_rows() as f64;
        n_y = output_header.get_n_columns() as f64;
        n_z = output_header.get_n_sections() as f64;
        mode_bytes = base.get_mode_bytes(output_header.get_mode())? as f64;

        let mtf_filter_param = self.load_mtf_filter_param(axis_id);
        // take starting and ending Z into account
        if mtf_filter_param.is_starting_z_set() {
            if mtf_filter_param.is_ending_z_set() {
                n_z = mtf_filter_param.get_ending_z() as f64
                    - mtf_filter_param.get_starting_z() as f64
                    + 1.0f64;
            } else {
                n_z = n_z - mtf_filter_param.get_starting_z() as f64 + 1.0f64;
            }
        } else if mtf_filter_param.is_ending_z_set() {
            n_z = mtf_filter_param.get_ending_z() as f64;
        }

        // Assumption: newst will write the output file with the same mode as the
        // the input file
        let file_size = 1024.0 + n_x * n_y * n_z * mode_bytes;
        let n_k_bytes = (file_size / 1024.0) as i32;
        base.n_k_bytes.store(n_k_bytes, Ordering::SeqCst);
        let title = self.get_title(base);
        manager.post_main_panel(Box::new(move |panel| {
            panel.set_progress_bar_string_int_boolean_axis_id(Some(&title), n_k_bytes, false, axis_id);
        }));
        Ok(true)
    }

    /// Java `getTitle`.
    fn get_title(&self, _base: &FileSizeProcessMonitor) -> String {
        "Running MTF filter".to_string()
    }

    /// Java `reloadWatchedFile`.
    fn reload_watched_file(&self, base: &FileSizeProcessMonitor) {
        let mtf_filter_param = self.load_mtf_filter_param(base.axis_id);
        // `new File(String parent, String child)`: a null parent is the child
        // alone.  A null child is a NullPointerException in the source; it is read
        // as the empty name here.
        let output_file = mtf_filter_param.get_output_file().unwrap_or_default();
        *base.watched_file.lock().unwrap() =
            Some(PathBuf::from(match base.manager.get_property_user_dir() {
                None => output_file,
                Some(parent) => utilities::java_io_file_new(&parent, &output_file),
            }));
    }
}
