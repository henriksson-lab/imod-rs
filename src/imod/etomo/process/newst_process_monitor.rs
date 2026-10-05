//! `IMOD/Etomo/src/etomo/process/NewstProcessMonitor.java`.
//!
//! A `FileSizeProcessMonitor` subclass; see the module comment of
//! `file_size_process_monitor.rs` for how subclasses are represented.

use super::file_size_process_monitor::{
    CalcFileSizeError, FileSizeProcessMonitor, FileSizeProcessMonitorImpl, FileSizeProcessMonitorOf,
};
use super::monitor_tool_kit::{self, InterruptedException};
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::const_newst_param::ConstNewstParam;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::util::mrc_header::MRCHeader;
use crate::imod::etomo::util::utilities;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::Ordering;

/// Java `NewstProcessMonitor`'s own fields.
pub struct NewstProcessMonitor {
    /// Java field `newstParam`.  NewstParam must be passed in because it can
    /// be loaded from more then one com file.
    newst_param: Arc<dyn ConstNewstParam + Send + Sync>,
}

impl NewstProcessMonitor {
    /// Java `NewstProcessMonitor(BaseManager, AxisID, ProcessName,
    /// ConstNewstParam)`.
    pub fn new(
        manager: &'static dyn BaseManager,
        id: AxisID,
        process_name: ProcessName,
        newst_param: Arc<dyn ConstNewstParam + Send + Sync>,
    ) -> Arc<FileSizeProcessMonitorOf<NewstProcessMonitor>> {
        FileSizeProcessMonitorOf::new(
            manager,
            id,
            process_name,
            NewstProcessMonitor { newst_param },
        )
    }
}

impl FileSizeProcessMonitorImpl for NewstProcessMonitor {
    /// Java `getStatusFromLog`.  Set gettingStatusFromLog to true and return
    /// true if mrctaper is being run.  This is for backward compatibility since
    /// mrctaper has been replace by the -taper newstack parameter.
    fn get_status_from_log(
        &self,
        base: &FileSizeProcessMonitor,
    ) -> Result<(), InterruptedException> {
        base.get_status_from_log()?;
        if base.is_ending() {
            return Ok(());
        }
        // read the lines available in the log and look for a line shows that
        // mrctaper started
        let mut exception = false;
        loop {
            match base.read_line_from_log() {
                Ok(None) => break,
                Ok(Some(line)) => {
                    if line.starts_with("Tapering over") {
                        // mrctaper started
                        let axis_id = base.axis_id;
                        base.manager.post_main_panel(Box::new(move |panel| {
                            panel.set_progress_bar_value_int_string_axis_id(
                                0,
                                Some("mrctaper"),
                                axis_id,
                            );
                        }));
                        base.set_ending();
                        break;
                    }
                }
                Err(e) => {
                    // LogFileException | FileNotFoundException | IOException: the
                    // throw skips the `Thread.sleep(1)` below.
                    eprintln!("{e}");
                    exception = true;
                    break;
                }
            }
        }
        if !exception {
            monitor_tool_kit::sleep(&base.interrupted, 1)?;
        }
        if exception || base.is_ending() || !base.is_file_writing() {
            base.close_log_reader();
        }
        Ok(())
    }

    /// Java `calcFileSize`.
    fn calc_file_size(&self, base: &FileSizeProcessMonitor) -> Result<bool, CalcFileSizeError> {
        let manager = base.manager;
        let axis_id = base.axis_id;
        let mut n_x: i32;
        let mut n_y: i32;
        let n_z: i32;
        let mode_bytes: i32;

        // Get the depth, mode, any mods to the X and Y size from the tilt
        // command script and the input and output filenames.
        // Get the header from the raw stack to calculate the aligned stack stize
        let property_user_dir = manager.get_property_user_dir();
        let raw_stack_filename = property_user_dir
            .clone()
            .unwrap_or_else(|| "null".to_string())
            + "/"
            + &self
                .newst_param
                .get_input_file()
                .unwrap_or_else(|| "null".to_string());
        let raw_stack = MRCHeader::get_instance_in_dir(
            property_user_dir.as_deref(),
            Some(&raw_stack_filename),
            Some(axis_id),
        )
        .unwrap();
        let mut raw_stack = raw_stack.borrow_mut();
        if !raw_stack
            .read_with_manager(manager)
            .map_err(CalcFileSizeError::from)?
        {
            return Ok(false);
        }
        let mut binning_already_applied = false;
        if self.newst_param.is_size_to_output_in_x_and_y_set() {
            binning_already_applied = true;
            n_x = self.newst_param.get_size_to_output_in_x();
            n_y = self.newst_param.get_size_to_output_in_y();
        } else {
            n_x = raw_stack.get_n_rows();
            n_y = raw_stack.get_n_columns();
        }
        n_z = raw_stack.get_n_sections();
        mode_bytes = base.get_mode_bytes(raw_stack.get_mode())?;

        // Get the binByFactor from newst.com script
        let bin_by = self.newst_param.get_bin_by_factor();
        // If the bin by factor is unspecified it defaults to 1
        if !binning_already_applied && bin_by > 1 {
            n_x /= bin_by;
            n_y /= bin_by;
        }

        // Assumption: newst will write the output file with the same mode as the
        // the input file
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
        "Creating aligned stack".to_string()
    }

    /// Java `reloadWatchedFile`.
    fn reload_watched_file(&self, base: &FileSizeProcessMonitor) {
        // Create a file object describing the file to be monitored
        // `new File(String parent, String child)`: a null parent is the child
        // alone.
        let output_file = self.newst_param.get_output_file();
        *base.watched_file.lock().unwrap() =
            Some(PathBuf::from(match base.manager.get_property_user_dir() {
                None => output_file,
                Some(parent) => utilities::java_io_file_new(&parent, &output_file),
            }));
    }
}
