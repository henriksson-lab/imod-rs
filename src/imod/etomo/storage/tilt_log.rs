//! `IMOD/Etomo/src/etomo/storage/TiltLog.java`.
//!
//! Reads the first and last angle of the "Projection angles:" list in a tilt log.

use std::path::Path;
use std::sync::Arc;

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::{Type, java_lang_string_matches_whitespace};
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::storage::log_file::{Handle, LogFile, LogFileError, ReaderId};

/// Java `public final class TiltLog`.
pub struct TiltLog {
    /// Java private `file` (created in initialization).
    file: Option<Arc<Handle>>,
    /// Java private final `minAngle`.
    min_angle: EtomoNumber,
    /// Java private final `maxAngle`.
    max_angle: EtomoNumber,
}

impl TiltLog {
    /// Java private `TiltLog()`.
    fn new() -> TiltLog {
        TiltLog {
            file: None,
            min_angle: EtomoNumber::new_with_type(Some(Type::Double)),
            max_angle: EtomoNumber::new_with_type(Some(Type::Double)),
        }
    }

    /// Java static `getInstance(BaseManager, AxisID, File) throws
    /// LogFile.FileException, IOException`.
    pub fn get_instance(
        manager: Option<&'static dyn BaseManager>,
        axis_id: AxisID,
        file: &Path,
    ) -> Result<TiltLog, LogFileError> {
        let mut instance = TiltLog::new();
        instance.file = Some(LogFile::get_instance_file(
            Some(file),
            manager.map(|manager| manager.get_emergency_monitor(Some(axis_id))),
        )?);
        Ok(instance)
    }

    /// Java `read()`.  Resets the data, and opens, reads, and closes the log file.
    pub fn read(&mut self) -> bool {
        self.reset();
        let Some(file) = self.file.clone() else {
            return false;
        };
        let result = (|| -> Result<bool, LogFileError> {
            let reader_id = file.open_reader()?;
            let Some(reader_id) = reader_id else {
                return Ok(false);
            };
            let retval = self.read_reader_id(&file, &reader_id);
            file.close_id(Some(&*reader_id));
            Ok(retval)
        })();
        match result {
            Ok(retval) => retval,
            Err(LogFileError::Lock(_)) => false,
            Err(e) => {
                eprintln!("{e}");
                false
            }
        }
    }

    /// Java `getMinAngle()`.
    pub fn get_min_angle(&self) -> String {
        self.min_angle.to_string()
    }

    /// Java `getMaxAngle()`.
    pub fn get_max_angle(&self) -> String {
        self.max_angle.to_string()
    }

    /// Java private `read(LogFile.ReaderId)`.  Reads data from the log file.
    fn read_reader_id(&mut self, file: &Arc<Handle>, reader_id: &ReaderId) -> bool {
        let result = (|| -> Result<bool, LogFileError> {
            // find angle list
            let mut line;
            loop {
                line = file.read_line(reader_id)?;
                match &line {
                    Some(text) if !text.ends_with("Projection angles:") => {}
                    _ => break,
                }
            }
            if line.is_none() {
                return Ok(false);
            }
            // remove blank line at start of angle list
            loop {
                line = file.read_line(reader_id)?;
                match &line {
                    Some(text) if !java_lang_string_matches_whitespace(text) => {}
                    _ => break,
                }
            }
            if line.is_none() {
                return Ok(false);
            }
            // set minAngle to the first angle in the angle list
            line = file.read_line(reader_id)?;
            // Fixed in translation (TiltLog.java:100): Java dereferences a null line
            // (the log ends after the blank line) and throws NullPointerException; the
            // read fails instead.
            let Some(first) = line.clone() else {
                return Ok(false);
            };
            // `line.trim().split("\\s+")[0]`
            let trimmed = first.trim_matches(|c: char| (c as u32) <= 0x20);
            let token = trimmed.split_whitespace().next().unwrap_or("");
            self.min_angle.set_string(Some(token));
            // find the last line of the angle list (assume there is a blank line after
            // the angle list)
            let mut prev_line = first;
            loop {
                line = file.read_line(reader_id)?;
                match &line {
                    Some(text) if !java_lang_string_matches_whitespace(text) => {
                        prev_line = text.clone();
                    }
                    _ => break,
                }
            }
            // set maxAngle to the last angle in the angle list
            // `prevLine.split("\\s+")`: a leading separator gives an empty first
            // element, trailing empties are dropped.
            let mut angle_array: Vec<&str> = prev_line.split(char::is_whitespace).collect();
            while angle_array.len() > 1 && angle_array.last() == Some(&"") {
                angle_array.pop();
            }
            self.max_angle
                .set_string(angle_array.last().copied());
            Ok(true)
        })();
        match result {
            Ok(retval) => retval,
            Err(e) => {
                eprintln!("{e}");
                false
            }
        }
    }

    /// Java private `reset()`.
    fn reset(&mut self) {
        self.min_angle.reset();
        self.max_angle.reset();
    }
}
