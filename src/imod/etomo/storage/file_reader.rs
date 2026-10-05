//! `IMOD/Etomo/src/etomo/storage/FileReader.java`.
//!
//! Reads the end of a reasonably small log file: `goToLast` saves the lines from the
//! last line containing a token to the end of the file, and `readLine` hands them out.

use std::sync::{Arc, Mutex};

use super::file_writer::FileWriter;
use super::log_file::{LogFileError, ReaderId};
use super::log_file_interface::LogFileInterface;

/// A Java reference to a `FileReader`.
pub type FileReaderRef = Arc<Mutex<FileReader>>;

/// Java `public final class FileReader`.
pub struct FileReader {
    /// Java private final `maxLinesAllowed`.
    max_lines_allowed: i32,
    /// Java private `logFile`, initially null.
    log_file: Option<Box<dyn LogFileInterface>>,
    /// Java private `readerId`, initially null.
    reader_id: Option<ReaderId>,
    /// Java private `savedLines`, initially null.
    saved_lines: Option<Vec<String>>,
    /// Java private `curLine`, initially -1.
    cur_line: i32,
}

impl Default for FileReader {
    fn default() -> FileReader {
        FileReader::new()
    }
}

impl FileReader {
    /// Java `FileReader()`.
    pub fn new() -> FileReader {
        FileReader::new_max_lines_allowed(1000)
    }

    /// Java package-private `FileReader(int)`.
    pub(crate) fn new_max_lines_allowed(max_lines_allowed: i32) -> FileReader {
        FileReader {
            max_lines_allowed,
            log_file: None,
            reader_id: None,
            saved_lines: None,
            cur_line: -1,
        }
    }

    /// Java `setFile(FileWriter)`.
    pub fn set_file(&mut self, file_writer: Option<&mut FileWriter>) -> bool {
        self.log_file = None;
        let Some(file_writer) = file_writer else {
            return false;
        };
        if file_writer.is_empty() {
            return false;
        }
        file_writer.flush();
        self.log_file = file_writer
            .get_file()
            .map(|log_file| Box::new(log_file) as Box<dyn LogFileInterface>);
        self.log_file.is_some()
    }

    /// Java package-private `setFile(LogFileInterface)`.
    pub(crate) fn set_file_interface(
        &mut self,
        log_file: Option<Box<dyn LogFileInterface>>,
    ) -> bool {
        self.log_file = log_file;
        self.log_file.is_some()
    }

    /// Java private `open()`.
    fn open(&mut self) -> bool {
        if self.reader_id.is_some() {
            return true;
        }
        let Some(log_file) = &self.log_file else {
            return false;
        };
        match log_file.open_reader() {
            Ok(reader_id) => self.reader_id = reader_id,
            // `catch (final LockException e) {}`
            Err(LogFileError::Lock(_)) => {}
            // `catch (final LogFileException | IOException e) { e.printStackTrace(); }`
            Err(e) => eprintln!("{e}"),
        }
        self.reader_id.is_some()
    }

    /// Java `close()`.
    pub fn close(&mut self) {
        if let Some(reader_id) = self.reader_id.take() {
            if let Some(log_file) = &self.log_file {
                log_file.close_id(Some(&*reader_id));
            }
        }
    }

    /// Java `goToLast(String)`.  Saves the last line containing `token` and the lines
    /// after it.
    pub fn go_to_last(&mut self, token: Option<&str>) -> bool {
        self.cur_line = -1;
        if let Some(saved_lines) = &mut self.saved_lines {
            saved_lines.clear();
        }
        self.close();
        let token = match token {
            Some(token) if !token.is_empty() => token,
            _ => return false,
        };
        if !self.open() {
            return false;
        }
        let mut found = false;
        let log_file = self.log_file.as_ref().unwrap();
        let reader_id = self.reader_id.as_ref().unwrap();
        let result = (|| -> Result<Option<bool>, LogFileError> {
            // Find the first instance of token
            let mut line = None;
            while let Some(next) = log_file.read_line(reader_id)? {
                if next.contains(token) {
                    found = true;
                    line = Some(next);
                    break;
                }
            }
            // Fail if token was not found.
            if !found {
                return Ok(Some(false));
            }
            // Save the lines starting with token and look for the next instance.  Limit
            // the number of lines saved to maxLinesAllowed.  This function is designed
            // for reading a reasonably small file.
            let saved_lines = self.saved_lines.get_or_insert_with(Vec::new);
            saved_lines.push(line.unwrap());
            let mut line_count = 1;
            while let Some(line) = log_file.read_line(reader_id)? {
                if line.contains(token) {
                    saved_lines.clear();
                    line_count = 0;
                }
                saved_lines.push(line);
                line_count += 1;
                if line_count > self.max_lines_allowed {
                    eprintln!(
                        "java.lang.IllegalArgumentException: Giving up searching for {} because file ({}) is too large.",
                        token,
                        log_file.get_absolute_path()
                    );
                    return Ok(None);
                }
            }
            // Token found.  Last token and following lines is contained in savedLines.
            Ok(Some(true))
        })();
        match result {
            // `close(); return false;` when the token was not found.
            Ok(Some(false)) => {
                self.close();
                return false;
            }
            // Too large: `return false` without closing the reader.
            Ok(None) => return false,
            Ok(Some(true)) => {}
            // `catch (final LogFileException e)` / `catch (final IOException e)`
            Err(e) => eprintln!("{e}"),
        }
        self.close();
        found
    }

    /// Java `isReadable()`.
    pub fn is_readable(&self) -> bool {
        self.saved_lines
            .as_ref()
            .is_some_and(|saved_lines| saved_lines.len() as i32 > self.cur_line + 1)
    }

    /// Java `readLine()`.  A null `savedLines` is the source's caught
    /// `NullPointerException`, which prints its stack trace and returns null.
    pub fn read_line(&mut self) -> Option<String> {
        let Some(saved_lines) = &self.saved_lines else {
            eprintln!("java.lang.NullPointerException");
            return None;
        };
        if self.cur_line + 1 >= saved_lines.len() as i32 {
            return None;
        }
        self.cur_line += 1;
        Some(saved_lines[self.cur_line as usize].clone())
    }
}
