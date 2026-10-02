//! `IMOD/Etomo/src/etomo/process/TiltProcessMonitor.java`.
//!
//! A `FileSizeProcessMonitor` subclass; see the module comment of
//! `file_size_process_monitor.rs` for how subclasses are represented.
//!
//! `Tilt3dFindProcessMonitor extends TiltProcessMonitor` and overrides only
//! `getTiltParam`; that one-level-deeper subclass is carried in
//! `tilt_3d_find`, and `get_tilt_param` dispatches to it the way Java's
//! virtual call does.

use super::file_size_process_monitor::{
    CalcFileSizeError, FileSizeProcessMonitor, FileSizeProcessMonitorImpl, FileSizeProcessMonitorOf,
};
use super::tilt3d_find_process_monitor::Tilt3dFindProcessMonitor;
use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::comscript::const_tilt_param::ConstTiltParam;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::java_lang_string_matches_whitespace;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::util::mrc_header::MRCHeader;
use crate::imod::etomo::util::utilities;
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `TiltProcessMonitor`'s own fields.
pub struct TiltProcessMonitor {
    /// Java field `applicationManager`.
    application_manager: &'static ApplicationManager,
    /// Java field `tiltParam`.
    tilt_param: Mutex<Option<Arc<dyn ConstTiltParam + Send + Sync>>>,
    /// Java field `processTitle`.
    process_title: Mutex<String>,
    /// The `Tilt3dFindProcessMonitor` subclass state, when this is one.
    tilt_3d_find: Option<Tilt3dFindProcessMonitor>,
}

impl TiltProcessMonitor {
    /// Java `TiltProcessMonitor(ApplicationManager, AxisID, ProcessName)`.
    pub fn new(
        app_mgr: &'static ApplicationManager,
        id: AxisID,
        process_name: ProcessName,
    ) -> Arc<FileSizeProcessMonitorOf<TiltProcessMonitor>> {
        TiltProcessMonitor::new_subclass(app_mgr, id, process_name, None)
    }

    /// The Java constructor as reached from a subclass constructor
    /// (`Tilt3dFindProcessMonitor`'s `super(appMgr, id, processName)`), with
    /// that subclass's own state.
    pub fn new_subclass(
        app_mgr: &'static ApplicationManager,
        id: AxisID,
        process_name: ProcessName,
        tilt_3d_find: Option<Tilt3dFindProcessMonitor>,
    ) -> Arc<FileSizeProcessMonitorOf<TiltProcessMonitor>> {
        FileSizeProcessMonitorOf::new(
            app_mgr,
            id,
            process_name,
            TiltProcessMonitor {
                application_manager: app_mgr,
                tilt_param: Mutex::new(None),
                process_title: Mutex::new("Calculating tomogram".to_string()),
                tilt_3d_find,
            },
        )
    }

    /// Java static `getReconnectInstance`.
    pub fn get_reconnect_instance(
        app_mgr: &'static ApplicationManager,
        id: AxisID,
    ) -> Arc<FileSizeProcessMonitorOf<TiltProcessMonitor>> {
        let instance = TiltProcessMonitor::new(app_mgr, id, ProcessName::TILT);
        instance.set_reconnect(true);
        instance
    }

    /// Java final `setProcessTitle`.
    pub fn set_process_title(&self, input: Option<&str>) {
        // Fixed in translation: TiltProcessMonitor.java:236 tests
        // `!input.matches("\\*s")`, which only rejects the literal string "*s";
        // the evident intent (a blank title leaves the default) is
        // `matches("\\s*")`, which is what the translation tests.
        if let Some(input) = input {
            if !java_lang_string_matches_whitespace(input) {
                *self.process_title.lock().unwrap() = input.to_string();
            }
        }
    }

    /// Java private final `loadTiltParam`.
    fn load_tilt_param(&self, axis_id: AxisID) -> Arc<dyn ConstTiltParam + Send + Sync> {
        let mut tilt_param = self.tilt_param.lock().unwrap();
        if let Some(tilt_param) = &*tilt_param {
            return tilt_param.clone();
        }
        let param = self.get_tilt_param(axis_id);
        *tilt_param = Some(param.clone());
        self.application_manager
            .get_meta_data()
            .set_fiducialess(axis_id, param.is_fiducialess());
        param
    }

    /// Java `getTiltParam`.  Returns a loaded instance of TiltParam from
    /// ComScriptManager.
    pub fn get_tilt_param(&self, axis_id: AxisID) -> Arc<dyn ConstTiltParam + Send + Sync> {
        if let Some(tilt_3d_find) = &self.tilt_3d_find {
            return tilt_3d_find.get_tilt_param();
        }
        let com_script_manager = self.application_manager.get_com_script_manager();
        com_script_manager.load_tilt(axis_id);
        Arc::new(com_script_manager.get_tilt_param(axis_id))
    }
}

impl FileSizeProcessMonitorImpl for TiltProcessMonitor {
    /// Java final `calcFileSize`.
    fn calc_file_size(&self, base: &FileSizeProcessMonitor) -> Result<bool, CalcFileSizeError> {
        let manager = base.manager;
        let axis_id = base.axis_id;
        let mut n_x: i32;
        let mut n_y: i32;
        let n_z: i32;
        let mut mode_bytes: i32 = 4;

        // Get the depth, mode, any mods to the X and Y size from the tilt
        // command script and the input and output filenames.
        let tilt_param = self.load_tilt_param(axis_id);
        // Get the header from the aligned stack to use as default nX and
        // nY parameters
        let property_user_dir = manager.get_property_user_dir();
        let aligned_filename = property_user_dir
            .clone()
            .unwrap_or_else(|| "null".to_string())
            + "/"
            + &tilt_param.get_input_file();

        let aligned_stack = MRCHeader::get_instance_in_dir(
            property_user_dir.as_deref(),
            Some(&aligned_filename),
            Some(axis_id),
        )
        .unwrap();
        let mut aligned_stack = aligned_stack.borrow_mut();
        if !aligned_stack
            .read_with_manager(manager)
            .map_err(CalcFileSizeError::Io)?
        {
            return Ok(false);
        }

        n_x = aligned_stack.get_n_columns();
        n_y = aligned_stack.get_n_rows();

        n_z = tilt_param.get_thickness();
        if tilt_param.has_mode() {
            mode_bytes = base.get_mode_bytes(tilt_param.get_mode())?;
        }
        // Get the imageBinned from prenewst.com script
        let mut image_binned: i32 = 1;
        // Java tests `getImageBinned()` for null; the Rust getter always returns a
        // number, so the test is always true.
        let number = tilt_param.get_image_binned();
        image_binned = number.get_int();
        // adjust x and y
        if tilt_param.has_width() {
            n_x = tilt_param.get_width() / image_binned;
        }
        if tilt_param.has_slice() {
            let slice_range = tilt_param
                .get_idx_slice_stop()
                .wrapping_sub(tilt_param.get_idx_slice_start())
                .wrapping_add(1);
            // Divide by the step size if present
            n_y = slice_range / image_binned;
        }
        let file_size: i64 = 1024i64.wrapping_add(
            (n_x as i64)
                .wrapping_mul(n_y as i64)
                .wrapping_mul((n_z / image_binned) as i64)
                .wrapping_mul(mode_bytes as i64),
        );
        let n_k_bytes = (file_size / 1024) as i32;
        base.n_k_bytes.store(n_k_bytes, Ordering::SeqCst);

        if etomo_director::ARGUMENTS.lock().unwrap().is_debug() {
            eprintln!(
                "TiltProcessMonitor.calcFileSize:fileSize={},nX={},nY={},nZ={},imageBinned={}",
                file_size, n_x, n_y, n_z, image_binned
            );
        }
        let title = self.get_title(base);
        manager.post_main_panel(Box::new(move |panel| {
            panel.set_progress_bar_string_int_boolean_axis_id(Some(&title), n_k_bytes, false, axis_id);
        }));
        Ok(true)
    }

    /// Java `getTitle`.
    fn get_title(&self, _base: &FileSizeProcessMonitor) -> String {
        self.process_title.lock().unwrap().clone()
    }

    /// Java final `reloadWatchedFile`.
    fn reload_watched_file(&self, base: &FileSizeProcessMonitor) {
        let tilt_param = self.load_tilt_param(base.axis_id);
        // Create a file object describing the file to be monitored
        // `new File(String parent, String child)`: a null parent is the child
        // alone.
        let output_file = tilt_param.get_output_file();
        *base.watched_file.lock().unwrap() =
            Some(PathBuf::from(match base.manager.get_property_user_dir() {
                None => output_file,
                Some(parent) => utilities::java_io_file_new(&parent, &output_file),
            }));
    }
}
