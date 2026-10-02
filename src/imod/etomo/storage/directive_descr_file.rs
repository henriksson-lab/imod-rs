//! `IMOD/Etomo/src/etomo/storage/DirectiveDescrFile.java` (with its nested
//! `Iterator`).
//!
//! Copyright: Copyright 2013 - 2025 by the Regents of the University of Colorado
//!
//! Organization: Dept. of MCD Biology, University of Colorado
//!
//! Reads the directives description file, `FileType.DIRECTIVES_DESCR`
//! (`$IMOD_DIR/com/directives.csv`), through `LogFile`, one line at a time, each line
//! split on `\s*,\s*` as Java's `String.split` does.
//!
//! **Shape.**  `INSTANCE` is a process-global singleton whose three fields Java mutates
//! (`logFile`, `alternateFile`, `directiveMap`); each is behind its own `Mutex` so the
//! singleton is `Send + Sync`.  Java's `synchronized (this)` in `get` is the lock on
//! `directive_map`, and `synchronized open` is the lock on `log_file`.  `get` returns a
//! copy of the stored element, since the map owns it.
#![allow(dead_code)]

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, LazyLock, Mutex};

use regex::Regex;

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::storage::directive_descr_element::DirectiveDescrElement;
use crate::imod::etomo::storage::log_file::{Handle, LogFile, LogFileError, ReaderId};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::file_type;
use crate::imod::etomo::ui::swing::ui_harness;
use crate::imod::etomo::util::utilities::{java_io_file_get_absolute_path, java_lang_string_split};

/// Java `"\\s*,\\s*"`, the `split` regex in `Iterator.next`.  Java's `\s` is
/// `[ \t\n\x0B\f\r]`.
static LINE_SPLIT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[ \t\n\x0B\x0C\r]*,[ \t\n\x0B\x0C\r]*").unwrap());

/// Java `DirectiveDescrFile`.
pub struct DirectiveDescrFile {
    /// Java private field `logFile`, initialised to null.
    log_file: Mutex<Option<Arc<Handle>>>,
    /// Java private field `alternateFile`, initialised to null.
    alternate_file: Mutex<Option<PathBuf>>,
    /// Java private field `directiveMap`, initialised to null.
    directive_map: Mutex<Option<HashMap<String, DirectiveDescrElement>>>,
}

/// Java `INSTANCE`.
pub static INSTANCE: DirectiveDescrFile = DirectiveDescrFile::new();

impl DirectiveDescrFile {
    /// Java private `DirectiveDescrFile()`.
    const fn new() -> DirectiveDescrFile {
        DirectiveDescrFile {
            log_file: Mutex::new(None),
            alternate_file: Mutex::new(None),
            directive_map: Mutex::new(None),
        }
    }

    /// Java `releaseIterator(Iterator)`.
    ///
    /// DirectiveDescrFile.java:37-39 tests `logFile.isLocked()` outside the
    /// `logFile != null` guard, so a release while `logFile` is null throws a
    /// `NullPointerException`.  Fixed in translation: a null `logFile` stays null.
    pub fn release_iterator(&self, iterator: &Iterator) {
        let mut log_file = self.log_file.lock().unwrap();
        if let Some(handle) = log_file.as_ref() {
            handle.close_id(iterator.id.as_deref());
        }
        if log_file.as_ref().is_some_and(|handle| !handle.is_locked()) {
            *log_file = None;
        }
    }

    /// Java package-private `setFile(File)`.  Sets an alternative directives description
    /// file.  Pass null to remove the alternative file.  Has no effect while the log file
    /// is in use.  Returns true if alternateFile was set.
    pub(crate) fn set_file(&self, input: Option<PathBuf>) -> bool {
        match &input {
            None => {
                *self.alternate_file.lock().unwrap() = None;
            }
            Some(input) => {
                if self.log_file.lock().unwrap().is_some() {
                    eprintln!(
                        "Warning: unable to use {} as the directives description file because the default file is already in use.",
                        java_io_file_get_absolute_path(&input.to_string_lossy())
                    );
                    return false;
                }
            }
        }
        *self.alternate_file.lock().unwrap() = input;
        true
    }

    /// Java `getIterator(BaseManager, AxisID)`.  Opens the directives description file if
    /// necessary and returns an iterator for the file.  When done with the iterator, call
    /// releaseIterator.
    pub fn get_iterator(
        &self,
        manager: Option<&'static dyn BaseManager>,
        axis_id: Option<AxisID>,
    ) -> Option<Iterator> {
        if self.log_file.lock().unwrap().is_none() {
            if !self.open(manager, axis_id) {
                return None;
            }
        }
        let log_file = self.log_file.lock().unwrap().clone();
        // `open` succeeded, so `logFile` is set.
        let log_file = log_file?;
        match log_file.open_reader() {
            Ok(id) => Some(Iterator::new(manager, axis_id, log_file, id)),
            // `catch (final LockException e) { return null; }`
            Err(LogFileError::Lock(_)) => None,
            // `catch (final LogFileException | IOException e)`
            Err(e) => {
                // `e.printStackTrace()`; see etomo/util/stack_trace.rs.
                eprintln!("{}", e);
                ui_harness::open_message_dialog_from_process(
                    manager,
                    &format!(
                        "Unable to get a reader for {}.\n{}",
                        // Java would throw a `NullPointerException` if `getFile` returned
                        // null; the path then reads "null".
                        match file_type::CLASS.directives_descr.get_file(manager, axis_id) {
                            None => "null".to_string(),
                            Some(file) => java_io_file_get_absolute_path(&file.to_string_lossy()),
                        },
                        e.get_message()
                    ),
                    "Open File Failed",
                    axis_id,
                );
                None
            }
        }
    }

    /// Java private synchronized `open(BaseManager, AxisID)`.
    fn open(&self, manager: Option<&'static dyn BaseManager>, axis_id: Option<AxisID>) -> bool {
        let mut log_file = self.log_file.lock().unwrap();
        if log_file.is_some() {
            return true;
        }
        let file: Option<PathBuf>;
        let alternate_file = self.alternate_file.lock().unwrap().clone();
        if alternate_file.is_none() {
            file = file_type::CLASS.directives_descr.get_file(manager, axis_id);
        } else {
            file = alternate_file;
        }
        match LogFile::get_instance_file(
            file.as_deref(),
            manager.map(|manager| manager.get_emergency_monitor(axis_id)),
        ) {
            Ok(handle) => {
                *log_file = Some(handle);
                true
            }
            // `catch (final LogFile.FileException | IOException e)`
            Err(e) => {
                // `e.printStackTrace()`; see etomo/util/stack_trace.rs.
                eprintln!("{}", e);
                ui_harness::open_message_dialog_from_process(
                    manager,
                    &format!(
                        "Unable to open {}.\n{}",
                        match &file {
                            None => "null".to_string(),
                            Some(file) => java_io_file_get_absolute_path(&file.to_string_lossy()),
                        },
                        e.get_message()
                    ),
                    "Open File Failed",
                    axis_id,
                );
                false
            }
        }
    }

    /// Java package-private `get(String)`.  Returns the description of a directive.
    /// `key` matches the first column of the directive.cvs file.
    ///
    /// `Hashtable.get(null)` throws a `NullPointerException`
    /// (DirectiveDescrFile.java:139); `DirectiveDef.loadDirectiveDescr` passes
    /// `switchStandardKey`'s result, which can be null.  Fixed in translation: a null
    /// key finds nothing, after the map has been loaded as the source loads it.
    pub(crate) fn get(&self, key: Option<&str>) -> Option<DirectiveDescrElement> {
        let mut directive_map = self.directive_map.lock().unwrap();
        if directive_map.is_none() {
            let mut map = HashMap::new();
            let iterator = self.get_iterator(None, None);
            if let Some(mut iterator) = iterator {
                while iterator.has_next() {
                    let element = iterator.next_element();
                    if let Some(element) = element
                        && element.is_directive()
                    {
                        map.insert(
                            element.get_name().unwrap(),
                            DirectiveDescrElement::new_with_line_array(element.get_line_array()),
                        );
                    }
                }
                self.release_iterator(&iterator);
            }
            *directive_map = Some(map);
        }
        directive_map.as_ref().unwrap().get(key?).cloned()
    }
}

/// Java public static final nested class `Iterator`.  Readonly iterator.
pub struct Iterator {
    /// Java private final field `curElement`.  Only one instance used - don't hang onto
    /// it.
    cur_element: DirectiveDescrElement,
    /// Java private final field `axisID`.
    axis_id: Option<AxisID>,
    /// Java private final field `id`.  `LogFile.Handle.openReader` returns null when it
    /// has nothing to read.
    id: Option<ReaderId>,
    /// Java private final field `logFile`.
    log_file: Arc<Handle>,
    /// Java private final field `manager`.
    manager: Option<&'static dyn BaseManager>,
    /// Java private field `nextLine`, initialised to null.
    next_line: Option<String>,
}

impl Iterator {
    /// Java private `Iterator(BaseManager, AxisID, LogFile.Handle, LogFile.ReaderId)`.
    fn new(
        manager: Option<&'static dyn BaseManager>,
        axis_id: Option<AxisID>,
        log_file: Arc<Handle>,
        id: Option<ReaderId>,
    ) -> Iterator {
        Iterator {
            cur_element: DirectiveDescrElement::new(),
            axis_id,
            id,
            log_file,
            manager,
            next_line: None,
        }
    }

    /// Java `hasNext()`.  Moves to the next line if nextLine is null.  Returns true if
    /// there is another line to read and nextLine has been set.
    ///
    /// A null reader id reaches `LogFile.readLine`, whose lock test fails and throws
    /// `UnlockedException`, a `LogFileException`; that is the `catch` arm taken here.
    pub fn has_next(&mut self) -> bool {
        // hasNext has been run, but next was not, so line isn't incremented.
        if self.next_line.is_some() {
            return true;
        }
        let result = match &self.id {
            Some(id) => self.log_file.read_line(id),
            None => Err(LogFileError::Unlocked(
                crate::imod::etomo::storage::log_file::UnlockedException::new_id(None, None),
            )),
        };
        match result {
            Ok(line) => {
                self.next_line = line;
                self.next_line.is_some()
            }
            // `catch (final LogFileException e)` and `catch (IOException e)`: the two
            // arms are identical.
            Err(e) => {
                // `e.printStackTrace()`; see etomo/util/stack_trace.rs.
                eprintln!("{}", e);
                ui_harness::open_message_dialog_from_process(
                    self.manager,
                    &format!(
                        "Unable to read {}.\n{}",
                        // Java would throw a `NullPointerException` if `getFile` returned
                        // null; the path then reads "null".
                        match file_type::CLASS
                            .directives_descr
                            .get_file(self.manager, self.axis_id)
                        {
                            None => "null".to_string(),
                            Some(file) => java_io_file_get_absolute_path(&file.to_string_lossy()),
                        },
                        e.get_message()
                    ),
                    "Open File Failed",
                    self.axis_id,
                );
                false
            }
        }
    }

    /// Java `next()`.  Increment the line, and null out nextLine.
    pub fn next(&mut self) -> Option<Vec<String>> {
        if self.has_next() {
            let line_array = java_lang_string_split(self.next_line.as_ref().unwrap(), &LINE_SPLIT);
            self.next_line = None;
            return Some(line_array);
        }
        None
    }

    /// Java `nextElement()`.
    pub fn next_element(&mut self) -> Option<&DirectiveDescrElement> {
        let line_array = self.next();
        if let Some(line_array) = line_array {
            self.cur_element.set_line_array(Some(line_array));
            return Some(&self.cur_element);
        }
        None
    }
}
