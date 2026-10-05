//! `IMOD/Etomo/src/etomo/storage/TiltFile.java`.
//!
//! Reads the minimum and maximum angle of a tilt angle file (`.tlt`): the first and
//! the last line.

use std::path::{Path, PathBuf};

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::{ConstEtomoNumber, Type};
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::storage::log_file::{LogFile, LogFileError};

/// Java `public final class TiltFile`.
pub struct TiltFile {
    /// Java private final `minAngle`.
    min_angle: EtomoNumber,
    /// Java private final `maxAngle`.
    max_angle: EtomoNumber,
    /// Java private final `file`.
    file: PathBuf,
}

impl TiltFile {
    /// Java private `TiltFile(File)`.
    fn new(file: &Path) -> TiltFile {
        TiltFile {
            min_angle: EtomoNumber::new_with_type(Some(Type::Double)),
            max_angle: EtomoNumber::new_with_type(Some(Type::Double)),
            file: file.to_path_buf(),
        }
    }

    /// Java static `getInstance(BaseManager, AxisID, File)`.
    pub fn get_instance(
        manager: Option<&'static dyn BaseManager>,
        axis_id: AxisID,
        file: &Path,
    ) -> TiltFile {
        let mut instance = TiltFile::new(file);
        instance.initialize(manager, axis_id);
        instance
    }

    /// Java private `initialize(BaseManager, AxisID)`.
    fn initialize(&mut self, manager: Option<&'static dyn BaseManager>, axis_id: AxisID) {
        let result = (|| -> Result<(), LogFileError> {
            let file_reader = LogFile::get_instance_file(
                Some(&self.file),
                manager.map(|manager| manager.get_emergency_monitor(Some(axis_id))),
            )?;
            let Some(reader_id) = file_reader.open_reader()? else {
                // Java's readLine(null) throws; nothing is read.
                return Ok(());
            };
            let line = file_reader.read_line(&reader_id)?;
            self.min_angle.set_string(line.as_deref());
            // read until end of file, preserving last line read
            let mut prev_line = None;
            while let Some(line) = file_reader.read_line(&reader_id)? {
                prev_line = Some(line);
            }
            self.max_angle.set_string(prev_line.as_deref());
            // minAngle must be smaller then maxAngle
            if self
                .max_angle
                .lt_const_etomo_number(Some(&self.min_angle))
            {
                let temp = self.min_angle.get_double();
                let max: ConstEtomoNumber = (*self.max_angle).clone();
                self.min_angle.set_const_etomo_number(Some(&max));
                self.max_angle.set_double(temp);
            }
            Ok(())
        })();
        match result {
            Ok(()) | Err(LogFileError::Lock(_)) => {}
            Err(e) => eprintln!("{e}"),
        }
    }

    /// Java `getMinAngle()`.
    pub fn get_min_angle(&self) -> &ConstEtomoNumber {
        &self.min_angle
    }

    /// Java `getMaxAngle()`.
    pub fn get_max_angle(&self) -> &ConstEtomoNumber {
        &self.max_angle
    }
}
