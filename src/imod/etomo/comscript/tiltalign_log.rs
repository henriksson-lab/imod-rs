//! `IMOD/Etomo/src/etomo/comscript/TiltalignLog.java`.

use std::sync::Arc;

use super::const_tiltalign_param::EXCLUDE_LIST_KEY;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::process::process_output_strings::SUCCESS_TAG;
use crate::imod::etomo::storage::log_file::{Handle, LogFile, LogFileError};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::file_type;

/// Java `TiltalignLog`.
pub struct TiltalignLog {
    /// Java `log`.
    log: Option<Arc<Handle>>,
}

impl TiltalignLog {
    /// Java `getInstance(BaseManager, AxisID)`.
    pub fn get_instance(manager: &'static dyn BaseManager, axis_id: AxisID) -> TiltalignLog {
        let mut instance = TiltalignLog { log: None };
        instance.init(Some(manager), axis_id);
        instance
    }

    /// Java private `init(BaseManager, AxisID)`.
    fn init(&mut self, manager: Option<&'static dyn BaseManager>, axis_id: AxisID) {
        if self.log.is_some() {
            return;
        }
        let file = file_type::CLASS
            .tilt_align_log
            .get_file(manager, Some(axis_id));
        match LogFile::get_instance_file(
            file.as_deref(),
            manager.map(|manager| manager.get_emergency_monitor(Some(axis_id))),
        ) {
            Ok(log) => self.log = Some(log),
            // `catch (FileException | IOException e) { e.printStackTrace(); log = null; }`
            Err(e) => {
                eprintln!("{e}");
                self.log = None;
            }
        }
    }

    /// Java `exists()`.
    pub fn exists(&self) -> bool {
        let Some(log) = &self.log else {
            return false;
        };
        log.exists()
    }

    /// Java `isSuccess()`.
    pub fn is_success(&self) -> bool {
        let Some(log) = &self.log else {
            return false;
        };
        let mut id = None;
        match log.open_big_buffer_reader() {
            Ok(opened) => id = opened,
            // `catch (final LockException e) {}`
            Err(LogFileError::Lock(_)) => {}
            // `catch (final LogFileException | IOException e)`
            Err(e) => eprintln!("{e}"),
        }
        if let Some(id) = &id
            && !id.is_empty()
        {
            match log.search_for_last_line(id, SUCCESS_TAG) {
                Ok(success) => {
                    log.close_id(Some(&**id));
                    return success;
                }
                Err(e) => eprintln!("{e}"),
            }
            if !id.is_empty() {
                log.close_id(Some(&**id));
            }
        }
        false
    }

    /// Java `getExcludeList()`.
    pub fn get_exclude_list(&self) -> Option<String> {
        let log = self.log.as_ref()?;
        let mut id = None;
        match log.open_reader() {
            Ok(opened) => id = opened,
            // `catch (final LockException e) {}`
            Err(LogFileError::Lock(_)) => {}
            Err(e) => eprintln!("{e}"),
        }
        if let Some(id) = &id
            && !id.is_empty()
        {
            let mut found_param_list = false;
            // `try { ... } catch (LogFileException | IOException e)`: an error ends the
            // loop and falls through to the close below.
            let mut line = match log.read_line(id) {
                Ok(line) => line,
                Err(e) => {
                    eprintln!("{e}");
                    None
                }
            };
            while let Some(current) = line {
                let current = current.trim().to_owned();
                if current == "*** End of entries ***" {
                    log.close_id(Some(&**id));
                    return None;
                }
                if found_param_list {
                    if current.contains(EXCLUDE_LIST_KEY) {
                        // May be the right parameter - make sure by matching it exactly
                        let pair: Vec<&str> = regex::Regex::new(r"\s*=\s*")
                            .unwrap()
                            .split(&current)
                            .collect();
                        // Java `String.split` drops trailing empty strings.
                        let mut len = pair.len();
                        while len > 0 && pair[len - 1].is_empty() {
                            len -= 1;
                        }
                        if len > 0 && pair[0] == EXCLUDE_LIST_KEY {
                            log.close_id(Some(&**id));
                            if len > 1 {
                                return Some(pair[1].to_owned());
                            }
                            return None;
                        }
                    }
                } else if current == "*** Entries to program tiltalign ***" {
                    found_param_list = true;
                }
                line = match log.read_line(id) {
                    Ok(line) => line,
                    Err(e) => {
                        eprintln!("{e}");
                        None
                    }
                };
            }
        }
        if let Some(id) = &id
            && !id.is_empty()
        {
            log.close_id(Some(&**id));
        }
        None
    }
}
