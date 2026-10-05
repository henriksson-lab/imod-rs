//! `IMOD/Etomo/src/etomo/storage/AveragedFileNames.java`.
//!
//! Reads `averagedFilenames.txt`, the list of averaged volumes PEET writes; a file
//! name may be duplicated at the end of the file.

use std::path::Path;

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::storage::log_file::{LogFile, LogFileError};
use crate::imod::etomo::ui::swing::ui_harness;

/// Java `public final class AveragedFileNames`.
#[derive(Clone, Copy, Debug, Default)]
pub struct AveragedFileNames;

impl AveragedFileNames {
    /// Java implicit constructor.
    pub fn new() -> AveragedFileNames {
        AveragedFileNames
    }

    /// Java `getList(BaseManager, AxisID, String, String)`.  Build a list from the
    /// lines in averagedFilenames.txt.  Duplicates are only at the end of the file, so
    /// stop building the list when a duplicate is found.  Returns null (`None`) on a
    /// file error.
    pub fn get_list(
        &self,
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        error_message: &str,
        error_title: &str,
    ) -> Option<Vec<String>> {
        let opened = (|| {
            let file = LogFile::get_instance_file(
                Some(
                    &Path::new(&manager.get_property_user_dir().unwrap_or_default())
                        .join("averagedFilenames.txt"),
                ),
                Some(manager.get_emergency_monitor(Some(axis_id))),
            )?;
            let id = file.open_reader()?;
            Ok::<_, LogFileError>((file, id))
        })();
        let (file, id) = match opened {
            Ok((file, id)) => (Some(file), id),
            Err(LogFileError::Io(e)) if e.kind() == std::io::ErrorKind::NotFound => {
                eprintln!("{e}");
                ui_harness::with(|harness| {
                    harness.open_message_dialog_base_manager_string_string(
                        Some(manager),
                        &format!("File not found. {error_message}"),
                        error_title,
                    )
                });
                (None, None)
            }
            Err(LogFileError::Lock(_)) => return None,
            Err(e) => {
                eprintln!("{e}");
                return None;
            }
        };
        let mut list = Vec::new();
        // Java goes on to `file.readLine(id)` after a FileNotFoundException with a null
        // id, which throws a caught exception; the list stays empty.  A file that is
        // not there (openReader returned null) is read as empty the same way.
        let (Some(file), Some(id)) = (file, id) else {
            return Some(list);
        };
        let mut prev_line: Option<String> = None;
        let result = (|| -> Result<(), LogFileError> {
            while let Some(line) = file.read_line(&id)? {
                if Some(&line) == prev_line.as_ref() {
                    // Duplicate are all of the last file name so that rest of the file
                    // should be nothing but duplicates.
                    break;
                }
                // Fixed in translation (AveragedFileNames.java:60-68): the source never
                // assigns `prevLine`, so a duplicate is never found and the trailing
                // duplicates are opened in 3dmod too; the previous line is kept here,
                // as the comments above describe.  (BUGS.md)
                prev_line = Some(line.clone());
                list.push(line);
            }
            Ok(())
        })();
        if let Err(e) = result {
            eprintln!("{e}");
        }
        Some(list)
    }
}
