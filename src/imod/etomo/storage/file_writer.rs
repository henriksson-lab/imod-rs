//! `IMOD/Etomo/src/etomo/storage/FileWriter.java`.
//!
//! Writes lines to a log file (a secondary project log) through `LogFile`.  Java's
//! `synchronized` methods are the `Mutex` a shared [`FileWriterRef`] holds.

use std::path::Path;
use std::sync::{Arc, Mutex};

use super::log_file::{Handle, LogFile, LogFileError, WriterId};
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::r#type::axis_id::AxisID;

/// A Java reference to a `FileWriter`, whose methods are `synchronized`.
pub type FileWriterRef = Arc<Mutex<FileWriter>>;

/// Java `public final class FileWriter`.
#[derive(Default)]
pub struct FileWriter {
    /// Java private `logFile`, initially null.
    log_file: Option<Arc<Handle>>,
    /// Java private `writerId`, initially null.
    writer_id: Option<WriterId>,
    /// Java private `prevLine`, initially null.
    prev_line: Option<String>,
}

impl FileWriter {
    /// Java `FileWriter()`.
    pub fn new() -> FileWriter {
        FileWriter::default()
    }

    /// Java `FileWriter(File)`.  The source ignores the file.
    pub fn new_file(_file: Option<&Path>) -> FileWriter {
        FileWriter::default()
    }

    /// Java `equals(File)`.
    pub fn equals_file(&self, file: Option<&Path>) -> bool {
        let Some(log_file) = &self.log_file else {
            return false;
        };
        log_file.equals_file(file)
    }

    /// Java `setFile(BaseManager, AxisID, File)`.
    pub fn set_file(
        &mut self,
        manager: Option<&'static dyn BaseManager>,
        axis_id: Option<AxisID>,
        file: Option<&Path>,
    ) {
        self.reset();
        let Some(file) = file else {
            return;
        };
        let result = (|| -> Result<(), LogFileError> {
            let log_file = LogFile::get_instance_file(
                Some(file),
                manager.map(|manager| manager.get_emergency_monitor(axis_id)),
            )?;
            self.log_file = Some(log_file.clone());
            self.writer_id = Some(log_file.open_writer_append(true)?);
            Ok(())
        })();
        match result {
            Ok(()) => {}
            // `catch (final LockException e) {}`
            Err(LogFileError::Lock(_)) => {}
            // `catch (final LogFileException | IOException e) { e.printStackTrace(); }`
            Err(e) => eprintln!("{e}"),
        }
    }

    /// Java package-private `getFile()`.
    pub(crate) fn get_file(&self) -> Option<Arc<Handle>> {
        self.log_file.clone()
    }

    /// Java `reset()`.
    pub fn reset(&mut self) {
        self.close();
        self.log_file = None;
    }

    /// Java `isEmpty()`.
    pub fn is_empty(&self) -> bool {
        self.log_file.is_none()
    }

    /// Java `append(String)`.
    pub fn append(&mut self, line: &str) -> bool {
        let mut ret_val = true;
        if let (Some(writer_id), Some(log_file)) = (&self.writer_id, &self.log_file) {
            if let Err(e) = log_file.write(Some(line), writer_id) {
                // `catch (IOException e)` / `catch (LogFile.UnlockedException e)`
                ret_val = false;
                eprintln!("{e}");
            }
        }
        self.prev_line = Some(line.to_owned());
        ret_val
    }

    /// Java `getPrevLineEndOffset()`.  `String.length()` counts UTF-16 units.
    pub fn get_prev_line_end_offset(&self) -> i32 {
        if let Some(prev_line) = &self.prev_line {
            return prev_line.encode_utf16().count() as i32;
        }
        0
    }

    /// Java `flush()`.
    pub fn flush(&mut self) {
        if let (Some(writer_id), Some(log_file)) = (&self.writer_id, &self.log_file) {
            if let Err(e) = log_file.flush(writer_id) {
                eprintln!("{e}");
            }
        }
    }

    /// Java `close()`.
    pub fn close(&mut self) {
        self.flush();
        if let (Some(writer_id), Some(log_file)) = (&self.writer_id, &self.log_file) {
            log_file.close_id(Some(&**writer_id));
            self.writer_id = None;
        }
    }

    /// Java `isOpen()`.
    pub fn is_open(&self) -> bool {
        self.writer_id.is_some()
    }
}

/// Java `toString()`.
impl std::fmt::Display for FileWriter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[logFile:{},writerId:{},prevLine:{}]",
            self.log_file
                .as_ref()
                .map_or("null".to_owned(), |log_file| log_file.to_string()),
            self.writer_id
                .as_ref()
                .map_or("null".to_owned(), |writer_id| (***writer_id).to_string()),
            self.prev_line.as_deref().unwrap_or("null")
        )
    }
}
