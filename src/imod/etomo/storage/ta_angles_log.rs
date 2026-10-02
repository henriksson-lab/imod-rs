//! `IMOD/Etomo/src/etomo/storage/TaAnglesLog.java`.
//!
//! Reads values from the tiltalign angles log (`taAngles<axis>.log`), which
//! `AlignLogGenerator` splits out of the align log.

use std::sync::LazyLock;

use regex::Regex;

use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::process::align_log_generator;
use crate::imod::etomo::storage::log_file::{LogFile, LogFileError};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::{
    ConstEtomoNumber, Type, java_lang_string_trim,
};
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::r#type::file_type;
use crate::imod::etomo::util::utilities::java_lang_string_split;

/// Java private static final `CENTER_TO_CENTER_THICKNESS_TAG` (declared, not
/// read: the method repeats the literal).
const CENTER_TO_CENTER_THICKNESS_TAG: &str =
    "Unbinned thickness needed to contain centers of all fiducials";

/// Java `"\\s+"`.  Java's `\s` is `[ \t\n\x0B\f\r]`.
static WHITESPACE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[ \t\n\x0B\x0C\r]+").unwrap());

/// Java `public final class TaAnglesLog`.
pub struct TaAnglesLog {
    /// Java `private final String userDir`.
    user_dir: Option<String>,
    /// Java `private final ApplicationManager manager` (never null here).
    manager: &'static ApplicationManager,
    /// Java `private final AxisID axisID`.
    axis_id: AxisID,
}

impl TaAnglesLog {
    /// Java private `TaAnglesLog(String, ApplicationManager, AxisID)`.
    fn new(user_dir: Option<&str>, manager: &'static ApplicationManager, axis_id: AxisID) -> Self {
        TaAnglesLog {
            user_dir: user_dir.map(str::to_owned),
            manager,
            axis_id,
        }
    }

    /// Java static `getInstance(String, ApplicationManager, AxisID)`.
    pub fn get_instance(
        user_dir: Option<&str>,
        manager: &'static ApplicationManager,
        axis_id: AxisID,
    ) -> TaAnglesLog {
        let instance = TaAnglesLog::new(user_dir, manager, axis_id);
        instance.create_log();
        instance
    }

    /// Java private `createLog()`.
    fn create_log(&self) {
        let manager: &'static dyn BaseManager = self.manager;
        let ta_log = &file_type::CLASS.align_angles_log;
        if !ta_log.exists(Some(manager), Some(self.axis_id))
            || ta_log.last_modified(Some(manager), Some(self.axis_id))
                < file_type::CLASS
                    .tilt_align_log
                    .last_modified(Some(manager), Some(self.axis_id))
        {
            self.manager.generate_align_logs(self.axis_id);
        }
    }

    /// Java `getCenterToCenterThickness()`.  Get center to center thickness from
    /// the log.
    pub fn get_center_to_center_thickness(&self) -> Result<ConstEtomoNumber, LogFileError> {
        let mut center_to_center_thickness = EtomoNumber::new_with_type(Some(Type::Double));
        // refresh the log file
        // `manager != null ? manager.getEmergencyMonitor(axisID) : null`: the
        // manager is never null here.  A null userDir is `new File(null, name)`.
        let ta_angles_log = LogFile::get_instance_name(
            self.user_dir.as_deref().unwrap_or(""),
            self.axis_id,
            align_log_generator::ANGLES_LOG_NAME,
            Some(self.manager.get_emergency_monitor(Some(self.axis_id))),
        )?;
        if ta_angles_log.exists() {
            let reader_id = ta_angles_log.open_reader()?;
            if let Some(reader_id) = reader_id
                && !reader_id.is_empty()
            {
                let mut line = ta_angles_log.read_line(&reader_id)?;
                while let Some(current) = line {
                    let current = java_lang_string_trim(&current).to_owned();
                    if current.starts_with(
                        "Unbinned thickness needed to contain centers of all fiducials",
                    ) {
                        let string_array = java_lang_string_split(&current, &WHITESPACE);
                        // Upstream bug fixed in translation (TaAnglesLog.java:85):
                        // `stringArray[10]` throws ArrayIndexOutOfBoundsException on
                        // a short line; here the thickness stays null.
                        if let Some(value) = string_array.get(10) {
                            center_to_center_thickness.set_string(Some(value));
                        }
                        ta_angles_log.close_id(Some(&*reader_id));
                        return Ok(center_to_center_thickness.base);
                    }
                    line = ta_angles_log.read_line(&reader_id)?;
                }
                ta_angles_log.close_id(Some(&*reader_id));
            }
        }
        Ok(center_to_center_thickness.base)
    }

    /// Java `getIncrementalShiftToCenter()`.  Get incremental shift to center
    /// from the log.
    pub fn get_incremental_shift_to_center(&self) -> Result<ConstEtomoNumber, LogFileError> {
        let mut incremental_shift_to_center = EtomoNumber::new_with_type(Some(Type::Double));
        // refresh the log file
        let ta_angles_log = LogFile::get_instance_name(
            self.user_dir.as_deref().unwrap_or(""),
            self.axis_id,
            align_log_generator::ANGLES_LOG_NAME,
            Some(self.manager.get_emergency_monitor(Some(self.axis_id))),
        )?;
        if ta_angles_log.exists() {
            let reader_id = ta_angles_log.open_reader()?;
            if let Some(reader_id) = reader_id
                && !reader_id.is_empty()
            {
                let mut line = ta_angles_log.read_line(&reader_id)?;
                while let Some(current) = line {
                    let current = java_lang_string_trim(&current).to_owned();
                    if current.starts_with(
                        "Incremental unbinned shift needed to center range of fiducials in Z",
                    ) {
                        let string_array = java_lang_string_split(&current, &WHITESPACE);
                        // Upstream bug fixed in translation (TaAnglesLog.java:113):
                        // `stringArray[12]` throws ArrayIndexOutOfBoundsException on
                        // a short line; here the shift stays null.
                        if let Some(value) = string_array.get(12) {
                            incremental_shift_to_center.set_string(Some(value));
                        }
                        ta_angles_log.close_id(Some(&*reader_id));
                        return Ok(incremental_shift_to_center.base);
                    }
                    line = ta_angles_log.read_line(&reader_id)?;
                }
                ta_angles_log.close_id(Some(&*reader_id));
            }
        }
        Ok(incremental_shift_to_center.base)
    }
}
