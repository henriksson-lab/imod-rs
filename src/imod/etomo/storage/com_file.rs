//! `IMOD/Etomo/src/etomo/storage/ComFile.java`.
//!
//! Reads the standard-input parameters of one program out of a `.com` file (or the
//! `origcoms` copy of it), for the directive editor.  Java's `HashMap<String, String>`
//! with null values for parameters that have no value is
//! `HashMap<String, Option<String>>`.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, LazyLock};

use regex::Regex;

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::storage::log_file::{Handle, LogFile, LogFileError};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::java_lang_string_trim;
use crate::imod::etomo::util::utilities::{java_io_file_get_absolute_path, java_lang_string_split};

/// Java `"\\s+"`, the `split` regex in `getCommandMap`.
static WHITESPACE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[ \t\n\x0B\x0C\r]+").unwrap());

/// Java `public final class ComFile`.
pub struct ComFile {
    /// Java private final `axisID`.
    axis_id: AxisID,
    /// Java private final `manager`.
    manager: &'static dyn BaseManager,
    /// Java private final `subdirectory`.
    subdirectory: Option<String>,
    /// Java private `logFile`, initialised to null.
    log_file: Option<Arc<Handle>>,
    /// Java private `comFileName`, initialised to null.
    com_file_name: Option<String>,
    /// Java private `programName`, initialised to null.
    program_name: Option<String>,
}

impl ComFile {
    /// Java `ComFile(BaseManager, AxisID)`.
    pub fn new(manager: &'static dyn BaseManager, axis_id: AxisID) -> ComFile {
        ComFile {
            axis_id,
            manager,
            subdirectory: None,
            log_file: None,
            com_file_name: None,
            program_name: None,
        }
    }

    /// Java `ComFile(BaseManager, AxisID, String)`.
    pub fn new_subdirectory(
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        subdirectory: Option<&str>,
    ) -> ComFile {
        ComFile {
            axis_id,
            manager,
            subdirectory: subdirectory.map(str::to_string),
            log_file: None,
            com_file_name: None,
            program_name: None,
        }
    }

    /// Java `equalsComFileName(String)`.  True if comFileName has been set and is the
    /// same as input.
    pub fn equals_com_file_name(&self, input: Option<&str>) -> bool {
        match &self.com_file_name {
            None => false,
            Some(com_file_name) => Some(com_file_name.as_str()) == input,
        }
    }

    /// Java `equalsProgramName(String)`.  True if programName has been set and is the
    /// same as input.
    pub fn equals_program_name(&self, input: Option<&str>) -> bool {
        match &self.program_name {
            None => false,
            Some(program_name) => Some(program_name.as_str()) == input,
        }
    }

    /// Java `setComFileName(String)`.  Closes the current reader, if it exists, and
    /// resets to null the other member variables involved with reading.
    pub fn set_com_file_name(&mut self, input: Option<&str>) {
        if !self.equals_com_file_name(input) {
            self.log_file = None;
            self.com_file_name = input.map(str::to_string);
            self.program_name = None;
        }
    }

    /// Java `getCommandMap(String, StringBuffer)`.  Attempts to find an instance of
    /// programName in the .com file.  Opens the file for reading and searches for the
    /// program name.  If it finds the program name it fills a map with parameter
    /// names/values and returns it.
    pub fn get_command_map(
        &mut self,
        program_name: Option<&str>,
        errmsg: &mut String,
    ) -> Option<HashMap<String, Option<String>>> {
        self.program_name = program_name.map(str::to_string);
        // Create LogFile instance if is hasn't already been created
        if self.log_file.is_none() {
            let property_user_dir = self.manager.get_property_user_dir();
            let property_user_dir = property_user_dir.as_deref().unwrap_or("null");
            let dir_path = match &self.subdirectory {
                Some(subdirectory) => java_io_file_get_absolute_path(
                    &Path::new(property_user_dir)
                        .join(subdirectory)
                        .to_string_lossy(),
                ),
                None => property_user_dir.to_string(),
            };
            let file = Path::new(&dir_path).join(format!(
                "{}{}.com",
                self.com_file_name.as_deref().unwrap_or("null"),
                self.axis_id.get_extension()
            ));
            match LogFile::get_instance_file(
                Some(&file),
                Some(self.manager.get_emergency_monitor(Some(self.axis_id))),
            ) {
                Ok(log_file) => self.log_file = Some(log_file),
                // `catch (final FileException | IOException e)`
                Err(e) => {
                    // `e.printStackTrace()`; see etomo/util/stack_trace.rs.
                    eprintln!("{}", e);
                    errmsg.push_str(&e.get_message());
                    return None;
                }
            }
        }
        let log_file = self.log_file.clone().unwrap();
        let mut command_map: Option<HashMap<String, Option<String>>> = None;
        let id = match log_file.open_reader() {
            Ok(id) => id,
            // Not every com file requested will exist
            Err(LogFileError::File(_)) | Err(LogFileError::Io(_)) | Err(LogFileError::Lock(_)) => {
                return None;
            }
            Err(e) => {
                eprintln!("{}", e);
                errmsg.push_str(&format!(
                    "unable to open {}\n{}",
                    log_file.get_absolute_path(),
                    e.get_message()
                ));
                return None;
            }
        };
        let result: Result<(), LogFileError> = (|| {
            let Some(id) = id.as_ref() else {
                // A null reader id: `readLine` finds no lock and throws
                // `UnlockedException`, a `LogFileException`.
                return Err(LogFileError::Unlocked(
                    crate::imod::etomo::storage::log_file::UnlockedException::new_id(None, None),
                ));
            };
            // Find the command.
            let pattern = Regex::new(&format!(
                r"^\${}[ \t\n\x0B\x0C\r]+{}$",
                regex::escape(program_name.unwrap_or("null")),
                regex::escape("-StandardInput")
            ))
            .unwrap();
            let mut line;
            loop {
                line = log_file.read_line(id)?;
                match &line {
                    Some(text) if !pattern.is_match(text) => {}
                    _ => break,
                }
            }
            if line.is_some() {
                // Load the command's standard input parameters.
                let map = command_map.insert(HashMap::new());
                while let Some(text) = log_file.read_line(id)? {
                    let text = java_lang_string_trim(&text);
                    if text.starts_with('$') {
                        // End of command
                        break;
                    }
                    if text.is_empty() || text.starts_with('#') {
                        // Ignoring comments and empty lines
                        continue;
                    }
                    // Load the parameter
                    let array = java_lang_string_split(text, &WHITESPACE);
                    if array.is_empty() {
                        continue;
                    }
                    let key = array[0].clone();
                    if array.len() == 1 {
                        // Parameter has no value
                        map.insert(key, None);
                    } else {
                        // Parameter has a value
                        let value = java_lang_string_trim(&text[key.len()..]).to_string();
                        map.insert(key, Some(value));
                    }
                }
            }
            Ok(())
        })();
        if let Err(e) = result {
            // `catch (final LogFileException e)` and `catch (IOException e)`.
            eprintln!("{}", e);
            errmsg.push_str(&format!(
                "unable to read {}\n{}",
                log_file.get_absolute_path(),
                e.get_message()
            ));
        }
        if let Some(id) = id.as_ref() {
            log_file.close_id(Some(&**id));
        }
        command_map
    }
}
