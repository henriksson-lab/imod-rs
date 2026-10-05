//! `IMOD/Etomo/src/etomo/storage/TiltLogFileFilter.java`.

use std::path::Path;

use crate::imod::etomo::jdk::FileFilter;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::util::dataset_files;
use crate::imod::etomo::util::utilities;

/// Java `public static final String rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `public class TiltLogFileFilter extends FileFilter`.
#[derive(Clone, Copy, Debug, Default)]
pub struct TiltLogFileFilter;

impl TiltLogFileFilter {
    /// Java implicit constructor.
    pub fn new() -> TiltLogFileFilter {
        TiltLogFileFilter
    }
}

impl FileFilter for TiltLogFileFilter {
    /// Java `accept(File)`.
    fn accept(&self, f: &Path) -> bool {
        let name = utilities::java_io_file_get_name(&f.to_string_lossy());
        let tilt = ProcessName::TILT.to_string();
        if !f.is_file()
            || name.ends_with(&format!("{tilt}{}", dataset_files::LOG_EXT))
            || name.ends_with(&format!(
                "{tilt}{}{}",
                AxisID::First.get_extension(),
                dataset_files::LOG_EXT
            ))
            || name.ends_with(&format!(
                "{tilt}{}{}",
                AxisID::Second.get_extension(),
                dataset_files::LOG_EXT
            ))
        {
            return true;
        }
        false
    }

    /// Java `getDescription()`.
    fn get_description(&self) -> Option<String> {
        Some(format!(
            "Tilt Log ({}{})",
            ProcessName::TILT,
            dataset_files::LOG_EXT
        ))
    }
}
