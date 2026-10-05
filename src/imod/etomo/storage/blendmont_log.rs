//! `IMOD/Etomo/src/etomo/storage/BlendmontLog.java`.
//!
//! Finds blendmont's "Starting coordinates of output in X and Y" line in a log
//! (`preblend.log`) and parses the two coordinates, which `SerialSectionsManager`
//! passes to the blend (`UnalignedStartingXandY`).

use std::sync::{Arc, LazyLock};

use regex::Regex;

use super::log_file::{Handle, LogFileError};
use crate::imod::etomo::r#type::const_etomo_number::java_lang_string_trim;
use crate::imod::etomo::util::utilities;

/// The pattern `"\\s*=\\s*"`.
static EQUALS: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[ \t\n\x0B\x0C\r]*=[ \t\n\x0B\x0C\r]*").unwrap());
/// The pattern `"\\s+"`.
static WHITESPACE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[ \t\n\x0B\x0C\r]+").unwrap());

/// Java `public final class BlendmontLog`.
#[derive(Default)]
pub struct BlendmontLog {
    /// Java private `unalignedStartingXandYLine`, initially null.
    unaligned_starting_xand_y_line: Option<String>,
}

impl BlendmontLog {
    /// Java implicit `BlendmontLog()`.
    pub fn new() -> BlendmontLog {
        BlendmontLog {
            unaligned_starting_xand_y_line: None,
        }
    }

    /// Java `findUnalignedStartingXandY(LogFile.Handle)`.  Finds and saves the line
    /// containing the data for unalignedStartingXandY.  Returns true if found.
    pub fn find_unaligned_starting_xand_y(&mut self, log_file: Option<&Arc<Handle>>) -> bool {
        let Some(log_file) = log_file else {
            return false;
        };
        let result = (|| -> Result<bool, LogFileError> {
            let id = log_file.open_reader()?;
            // `openReader` answers null when the file cannot be opened; Java then
            // passes the null id to `readLine`, which returns null.
            let Some(id) = id else {
                return Ok(false);
            };
            while let Some(line) = log_file.read_line(&id)? {
                if line.contains("Starting coordinates of output in X and Y") {
                    self.unaligned_starting_xand_y_line = Some(line);
                    return Ok(true);
                }
            }
            Ok(false)
        })();
        match result {
            Ok(found) => found,
            Err(LogFileError::Lock(_)) => false,
            Err(e) => {
                eprintln!("{e:?}");
                false
            }
        }
    }

    /// Java `getUnalignedStartingXandY()`.  Parses unalignedStartingXandYLine and
    /// returns X and Y.  `findUnalignedStartingXandY` must be run first.  Returns
    /// `String[2]` or null.
    pub fn get_unaligned_starting_xand_y(&self) -> Option<Vec<String>> {
        let line = self.unaligned_starting_xand_y_line.as_ref()?;
        let array = utilities::java_lang_string_split(java_lang_string_trim(line), &EQUALS);
        if array.len() < 2 {
            println!("WARNING: Unrecognized blendmont output:\n{}", line);
            return None;
        }
        let array = utilities::java_lang_string_split(&array[1], &WHITESPACE);
        if array.len() < 2 {
            println!("WARNING: Unrecognized blendmont output:\n{}", line);
            return None;
        }
        Some(array)
    }
}
