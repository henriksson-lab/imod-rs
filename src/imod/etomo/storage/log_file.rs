//! `IMOD/Etomo/src/etomo/storage/LogFile.java`.
//!
//! Class which controls the opening and closing of a log file.  It also handles reading,
//! writing, copying, etc.  It can also hold a lock in order to give an outside reader or
//! writer safe access.  The goal is for the software to work properly on both Linux/Mac
//! and Windows, which have very different file locking schemes.
//!
//! This class contains a `Lock` inner class, which contains three types of locks: Read,
//! Write and File.  File locks are completely exclusive.  Only one Write lock can exist
//! at a time, and they can coexist with Read locks.  Read locks are not exclusive.
//!
//! `LogFile` is an n'ton, and should only have one instance per physical file.
//!
//! **Frontier.**  The lock machinery is translated: `Handle` and its members, the `Id`
//! hierarchy (`Id`, `ReadId` and the nine concrete id classes), `Lock`, `Reporter`,
//! `BlockingIdIterator`, `Lockable`, `LockException`, and every `LogFile` member that
//! takes an `Id`.  Two names are left, both `copyToNumberedFile` (LogFile.java:2106 and
//! its `Handle` wrapper at 2417): they need `NumberedFileType.contructInstance`,
//! `searchDirectory`, `SearchResults.getNextUnusedNumber`, `getFile` and `getOldestFile`,
//! none of which `src/imod/etomo/type/numbered_file_type.rs` has, and `searchDirectory`
//! additionally takes an `etomo/BaseManager.java`.  `getInstance(BaseManager, AxisID,
//! FileType, EmergencyMonitor)` is blocked on the same manager and on the untranslated
//! `FileType.getFile(BaseManager, AxisID)`; its six sibling overloads are here.  `copy`
//! and `rename` are translated - only their `UIHarness.openWarningMessageDialog` branch
//! is marked in place, and the `BaseManager` they take is `Option<Infallible>`, the Rust
//! type with exactly the one inhabitant (`null`) a translated caller can supply.
//!
//! **Java's instance monitors are not reproduced.**  Nearly every `LogFile` and `Lock`
//! method is `synchronized`, and Java's monitors are reentrant while Rust's `Mutex` is
//! not - `lock()` calls `isLocked()` calls `isLocked(LockType)`, and one instance-wide
//! `Mutex` would deadlock on the first such call.  Each mutable field carries its own
//! lock instead, which is what the fields translated before the lock layer already did.
//!
#![allow(dead_code)]

use crate::imod::etomo::base_manager::BaseManager;
use std::collections::HashMap;
use std::io::{BufRead, Read, Write};
use std::sync::{Arc, LazyLock, Mutex};

use crate::imod::etomo::etomo_director;
use crate::imod::etomo::process::emergency_monitor::EmergencyMonitor;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::debug_level::DebugLevel;
use crate::imod::etomo::r#type::extension_marker::ExtensionMarker;
use crate::imod::etomo::r#type::numbered_file_type::NumberedFileType;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::ui::standard_bar_string::StandardBarString;
use crate::imod::etomo::util::clean_print::CleanPrint;
use crate::imod::etomo::util::dataset_files;
use crate::imod::etomo::util::stack_trace::StackTrace;
use crate::imod::etomo::util::utilities;

/// Java `LOG_FILE_MAP`.
static LOG_FILE_MAP: LazyLock<Mutex<HashMap<String, Arc<LogFile>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Java `DEBUG`: `EtomoDirector.INSTANCE.getArguments().getDebugLevel()`.
static DEBUG: LazyLock<DebugLevel> =
    LazyLock::new(|| etomo_director::ARGUMENTS.lock().unwrap().get_debug_level());

/// Java `CLEAN_PRINT`:
/// `CleanPrint.getInstance(true, EtomoDirector.FILE_INFO_CLEAN_PRINT_LABEL)`.
static CLEAN_PRINT: LazyLock<CleanPrint> = LazyLock::new(|| {
    CleanPrint::get_instance_blockable(true, Some(etomo_director::FILE_INFO_CLEAN_PRINT_LABEL))
});

/// Java `SLEEP_LIMIT`.  Maximum waiting for a file is 15 seconds.
const SLEEP_LIMIT: i64 = 15000;
/// Java `SLEEP`.
const SLEEP: i64 = 100;

/// Java `LogFile`.
#[derive(Debug)]
pub struct LogFile {
    /// Java field `readingTokenList`.
    reading_token_list: ReadingTokenList,
    /// Java field `lock`.
    lock: Lock,
    /// Java field `file`.
    file: std::path::PathBuf,
    /// Java field `tempFile`.
    temp_file: Option<std::path::PathBuf>,
    /// Java field `backupTempFile`.
    backup_temp_file: Option<std::path::PathBuf>,
    /// Java field `fileAbsolutePath`.
    file_absolute_path: String,
    /// Java field `fileName`.
    file_name: String,
    /// Java fields `fileWriter` and `bufferedWriter`.  Java holds a `FileWriter` and a
    /// `BufferedWriter` wrapped around it; the wrapping is what `createWriter` and
    /// `closeResource(WriterId)` manage, so both are one `BufWriter` here and the pair's
    /// null-ness is the `Option`.
    file_writer: Mutex<Option<std::io::BufWriter<std::fs::File>>>,
    /// Java field `inputStream`.
    input_stream: Mutex<Option<std::fs::File>>,
    /// Java field `outputStream`.
    output_stream: Mutex<Option<std::fs::File>>,
    /// Java field `backedUp`.
    backed_up: Mutex<bool>,
    /// Java field `debug`.
    debug: Mutex<bool>,
    /// Java field `lockNumberToTrack`.
    lock_number_to_track: Mutex<Option<i32>>,
    /// Java field `verbose`.
    verbose: Mutex<bool>,
    /// Java field `stressTestHandle`.
    stress_test_handle: Mutex<Option<Arc<Handle>>>,
}

impl LogFile {
    /// Java `dumpState`.
    pub fn dump_state(&self) {
        if !DEBUG.is_verbose() {
            return;
        }
        eprintln!("[fileAbsolutePath:{},file:", self.file_absolute_path);
        eprintln!(
            "{}",
            utilities::java_io_file_get_absolute_path(&self.file.to_string_lossy())
        );
        eprintln!(",backupFile:");
        let backup_file = self.create_backup_file_var();
        eprintln!(
            "{}",
            utilities::java_io_file_get_absolute_path(&backup_file.to_string_lossy())
        );
        eprintln!(
            ",backedUp:{},debug:{}]",
            *self.backed_up.lock().unwrap(),
            *self.debug.lock().unwrap()
        );
    }

    /// Java `LogFile(File, File, File)`.  Java's constructor hands `this` to
    /// `new Lock(this)` before the object is published; `Arc::new_cyclic` is how a Rust
    /// value gives out a reference to itself from inside its own constructor, so this
    /// returns the `Arc` that `createInstance` would otherwise have wrapped it in.
    fn new(
        file: &std::path::Path,
        temp_file: Option<&std::path::Path>,
        backup_temp_file: Option<&std::path::Path>,
    ) -> Arc<LogFile> {
        Arc::new_cyclic(|log_file| LogFile {
            reading_token_list: ReadingTokenList::new(),
            lock: Lock::new(log_file.clone()),
            file: file.to_path_buf(),
            temp_file: temp_file.map(|temp_file| temp_file.to_path_buf()),
            backup_temp_file: backup_temp_file.map(|file| file.to_path_buf()),
            file_absolute_path: utilities::java_io_file_get_absolute_path(&file.to_string_lossy()),
            file_name: utilities::java_io_file_get_name(&file.to_string_lossy()),
            file_writer: Mutex::new(None),
            input_stream: Mutex::new(None),
            output_stream: Mutex::new(None),
            backed_up: Mutex::new(false),
            debug: Mutex::new(false),
            lock_number_to_track: Mutex::new(None),
            verbose: Mutex::new(false),
            stress_test_handle: Mutex::new(None),
        })
    }

    /// Java `getFile`.
    pub fn get_file(&self) -> &std::path::Path {
        &self.file
    }

    /// Java `getInstance(String, AxisID, ProcessName, EmergencyMonitor)`.  Get an
    /// instance of `LogFile` based on a key constructed from parameters describing the
    /// log file.
    pub fn get_instance_process_name(
        user_dir: &str,
        axis_id: AxisID,
        process_name: ProcessName,
        emergency_monitor: Option<Arc<EmergencyMonitor>>,
    ) -> Result<Arc<Handle>, LogFileError> {
        LogFile::get_instance_name(
            user_dir,
            axis_id,
            &process_name.to_string(),
            emergency_monitor,
        )
    }

    /// Java `getInstance(String, AxisID, String, EmergencyMonitor)`.
    pub fn get_instance_name(
        user_dir: &str,
        axis_id: AxisID,
        name: &str,
        emergency_monitor: Option<Arc<EmergencyMonitor>>,
    ) -> Result<Arc<Handle>, LogFileError> {
        LogFile::get_instance_user_dir(
            user_dir,
            &(name.to_string() + &axis_id.get_extension() + dataset_files::LOG_EXT),
            emergency_monitor,
        )
    }

    /// Java `getInstance(String, String, EmergencyMonitor)`.
    pub fn get_instance_user_dir(
        user_dir: &str,
        file_name: &str,
        emergency_monitor: Option<Arc<EmergencyMonitor>>,
    ) -> Result<Arc<Handle>, LogFileError> {
        LogFile::get_instance_stress_test(
            Some(&std::path::Path::new(user_dir).join(file_name)),
            false,
            emergency_monitor,
        )
    }

    /// Java `getInstance(File, String, EmergencyMonitor)`.
    pub fn get_instance_dir(
        dir: &std::path::Path,
        file_name: &str,
        emergency_monitor: Option<Arc<EmergencyMonitor>>,
    ) -> Result<Arc<Handle>, LogFileError> {
        LogFile::get_instance_stress_test(Some(&dir.join(file_name)), false, emergency_monitor)
    }

    /// Java `getInstance(File, EmergencyMonitor)`.
    pub fn get_instance_file(
        file: Option<&std::path::Path>,
        emergency_monitor: Option<Arc<EmergencyMonitor>>,
    ) -> Result<Arc<Handle>, LogFileError> {
        LogFile::get_instance_stress_test(file, false, emergency_monitor)
    }

    /// Java `getStressTestInstance`.
    pub fn get_stress_test_instance(
        file: Option<&std::path::Path>,
    ) -> Result<Arc<Handle>, LogFileError> {
        LogFile::get_instance_stress_test(file, true, None)
    }

    // TODO(unit): needs etomo/BaseManager.java - Java
    // `getInstance(BaseManager, AxisID, FileType, EmergencyMonitor)` (LogFile.java:203)
    // reads `manager.getEmergencyMonitor(axisID)` and calls the untranslated
    // `FileType.getFile(BaseManager, AxisID)`.

    /// Java `getInstance(File, boolean, EmergencyMonitor)`.
    pub fn get_instance_stress_test(
        file: Option<&std::path::Path>,
        stress_test_blocking_ids: bool,
        emergency_monitor: Option<Arc<EmergencyMonitor>>,
    ) -> Result<Arc<Handle>, LogFileError> {
        let file = match file {
            None => {
                let e = FileException::new("Cannot create LogFile, file is null.");
                // `e.printStackTrace()`; see etomo/util/stack_trace.rs.
                eprintln!("{}", e);
                return Err(LogFileError::File(e));
            }
            Some(file) => file,
        };
        let key = LogFile::get_map_key(Some(file))?;
        let log_file: Arc<LogFile>;
        // `synchronized (LOG_FILE_MAP)`; `createInstance` locks the same map and Rust's
        // `Mutex` is not reentrant, so the lookup releases it before calling on.
        let mapped = LOG_FILE_MAP.lock().unwrap().get(&key).cloned();
        match mapped {
            Some(mapped) => log_file = mapped,
            None => log_file = LogFile::create_instance(Some(file))?,
        }
        Ok(Handle::new(
            &log_file,
            stress_test_blocking_ids,
            emergency_monitor,
        ))
    }

    /// Java `getMapKey`.  Returns a unique key for `file`.
    fn get_map_key(file: Option<&std::path::Path>) -> Result<String, FileException> {
        let file = match file {
            None => {
                let e = FileException::new("Cannot create LogFile, file is null.");
                // `e.printStackTrace()`; see etomo/util/stack_trace.rs.
                eprintln!("{}", e);
                return Err(e);
            }
            Some(file) => file,
        };
        // `file.getCanonicalPath()`, falling back to `getAbsolutePath()` on IOException.
        match std::fs::canonicalize(file) {
            Ok(canonical) => Ok(canonical.to_string_lossy().to_string()),
            Err(_) => Ok(utilities::java_io_file_get_absolute_path(
                &file.to_string_lossy(),
            )),
        }
    }

    /// Java `resetAll`.  For testing.  Removes all instances of `LogFile`.
    pub fn reset_all() {
        if !etomo_director::ARGUMENTS.lock().unwrap().is_test() {
            return;
        }
        // LOG_FILE_HASH_TABLE.clear();
        LOG_FILE_MAP.lock().unwrap().clear();
    }

    /// Java `createInstance`.  Check for the existence of the instance, because another
    /// thread could have created it before this thread called `createInstance()`.  If the
    /// instance isn't there, create an instance of `LogFile` and a key and add them to
    /// the map.
    fn create_instance(file: Option<&std::path::Path>) -> Result<Arc<LogFile>, FileException> {
        let file = match file {
            None => {
                let e = FileException::new("Cannot create LogFile, file is null.");
                eprintln!("{}", e);
                return Err(e);
            }
            Some(file) => file,
        };
        let key = LogFile::get_map_key(Some(file))?;
        let mut map = LOG_FILE_MAP.lock().unwrap();
        if let Some(log_file) = map.get(&key) {
            return Ok(log_file.clone());
        }
        let unique_temp_file_path = LogFile::get_unique_temp_file_path(Some(file), &map);
        let log_file: Arc<LogFile>;
        if let Some(unique_temp_file_path) = unique_temp_file_path {
            log_file = LogFile::new(
                file,
                Some(std::path::Path::new(&unique_temp_file_path)),
                Some(std::path::Path::new(&(unique_temp_file_path.clone() + "~"))),
            );
        } else {
            log_file = LogFile::new(file, None, None);
        }
        map.insert(key, log_file.clone());
        Ok(log_file)
    }

    /// Java `getUniqueTempFilePath`.  Create a random temp file path name which is not
    /// already stored in `LOG_FILE_MAP`, and doesn't exist.  A backup name for this path
    /// is also not stored and doesn't exist.
    ///
    /// Java synchronizes on `LOG_FILE_MAP`; the map is already held by the caller here,
    /// so it is passed in rather than re-locked (Rust's `Mutex` is not reentrant).
    fn get_unique_temp_file_path(
        file: Option<&std::path::Path>,
        map: &HashMap<String, Arc<LogFile>>,
    ) -> Option<String> {
        let file = file?;
        // `file.getCanonicalPath()`, falling back to `getAbsolutePath()`.
        let path = match std::fs::canonicalize(file) {
            Ok(canonical) => canonical.to_string_lossy().to_string(),
            Err(_) => utilities::java_io_file_get_absolute_path(&file.to_string_lossy()),
        };
        if path.is_empty() {
            // Java tests `path == null`; `getAbsolutePath` never returns null.
            return None;
        }
        let mut temp_file_path: String;
        // Create a random temp file and backup file that aren't LogFile instances and
        // don't exist.
        loop {
            let random_string = java_util_uuid_random_uuid_no_dashes();
            temp_file_path = format!("{}_{}.tmp", path, random_string);
            let temp_file = std::path::Path::new(&temp_file_path);
            let backup_temp_file_path = temp_file_path.clone() + "~";
            let backup_temp_file = std::path::Path::new(&backup_temp_file_path);
            let temp_key = match LogFile::get_map_key(Some(temp_file)) {
                Ok(key) => key,
                Err(_) => return None,
            };
            let backup_key = match LogFile::get_map_key(Some(backup_temp_file)) {
                Ok(key) => key,
                Err(_) => return None,
            };
            if !(map.contains_key(&temp_key)
                || temp_file.exists()
                || map.contains_key(&backup_key)
                || backup_temp_file.exists())
            {
                break;
            }
        }
        Some(temp_file_path)
    }

    /// Java `getExceptionMessage`.
    pub fn get_exception_message(
        &self,
        message: &str,
        new_id: Option<&Id>,
        blocking_id: Option<&Id>,
        extended_info: bool,
    ) -> String {
        if !extended_info || (new_id.is_none() && blocking_id.is_none()) {
            return message.to_string();
        }
        let mut builder = String::new();
        builder.push_str(message);
        builder.push('\n');
        if let Some(new_id) = new_id {
            builder.push_str("New Id thread: ");
            builder.push_str(&new_id.get_handle().unwrap().get_thread_id());
        }
        if let Some(blocking_id) = blocking_id {
            if new_id.is_some() {
                builder.push_str(", ");
            }
            builder.push_str("Blocking Id thread: ");
            builder.push_str(&blocking_id.get_handle().unwrap().get_thread_id());
        }
        builder
    }

    /// Java `equals(LogFile)`.
    pub fn equals_log_file(self: &Arc<LogFile>, obj_log_file: Option<&Arc<LogFile>>) -> bool {
        // One instance per physical file.
        match obj_log_file {
            None => false,
            Some(obj_log_file) => Arc::ptr_eq(self, obj_log_file),
        }
    }

    /// Java `equals(LogFile.Handle)`.
    pub fn equals_handle(self: &Arc<LogFile>, obj_handle: Option<&Arc<Handle>>) -> bool {
        let obj_handle = match obj_handle {
            None => return false,
            Some(obj_handle) => obj_handle,
        };
        self.equals_log_file(Some(&obj_handle.log_file))
    }

    /// Java `equals(Path)`.
    pub fn equals_path(&self, obj_path: Option<&std::path::Path>) -> bool {
        let obj_path = match obj_path {
            None => return false,
            Some(obj_path) => obj_path,
        };
        // `Files.isSameFile(thisPath, objPath)`: equal paths are the same file, and
        // otherwise the two files' keys (device and inode) are compared; a missing file
        // is an IOException, which the source turns into false.
        if self.file == obj_path {
            return true;
        }
        java_nio_file_files_is_same_file(&self.file, obj_path).unwrap_or(false)
    }

    /// Java `equals(File)`.
    pub fn equals_file(&self, obj_file: Option<&std::path::Path>) -> bool {
        match obj_file {
            None => false,
            Some(obj_file) => self.file == obj_file,
        }
    }

    /// Java `equals(String)`.
    pub fn equals_string(&self, obj_string: Option<&str>) -> bool {
        let obj_string = match obj_string {
            None => return false,
            Some(obj_string) => obj_string,
        };
        // `Paths.get(objString)` throws InvalidPathException only for a string with a NUL
        // byte on this platform, which a Rust `&str` can still carry.
        if obj_string.contains('\0') {
            return self.equals_file(Some(std::path::Path::new(obj_string)));
        }
        self.equals_path(Some(std::path::Path::new(obj_string)))
    }

    /// Java `selfTestEquals`.  `assert` is a no-op in the JVM unless assertions are
    /// enabled, which is what `debug_assert!` is.
    fn self_test_equals(self: &Arc<LogFile>, handle: &Arc<Handle>, equals: bool) {
        debug_assert!(self.equals_log_file(Some(&handle.log_file)) == equals);
        debug_assert!(self.equals_handle(Some(handle)) == equals);
        debug_assert!(self.equals_path(Some(&handle.log_file.file)) == equals);
        debug_assert!(self.equals_file(Some(&handle.log_file.file)) == equals);
        debug_assert!(
            self.equals_string(Some(&match std::fs::canonicalize(&handle.log_file.file) {
                Ok(canonical) => canonical.to_string_lossy().to_string(),
                Err(_) => utilities::java_io_file_get_absolute_path(
                    &handle.log_file.file.to_string_lossy()
                ),
            })) == equals
        );
        debug_assert!(
            self.equals_string(Some(&utilities::java_io_file_get_absolute_path(
                &handle.log_file.file.to_string_lossy()
            ))) == equals
        );
        // `assert (!equals(1));`: an Integer matches none of the `equals(Object)` arms.
    }

    /// Java `getLineContaining(LogFile.Handle, ReaderId, String, SleepTimer)`.
    fn get_line_containing_handle(
        &self,
        handle: &Arc<Handle>,
        reader_id: &ReaderId,
        search_string: &str,
        timer: Option<&mut SleepTimer>,
    ) -> Result<Option<String>, LogFileError> {
        let mut id: Option<ReaderId> = None;
        let mut return_value: Option<String> = None;
        let mut lock_exception: Option<LogFileError> = None;
        match self.open_reader(reader_id, timer) {
            Ok(opened) => {
                id = opened;
                let empty = match &id {
                    None => true,
                    Some(id) => id.is_empty(),
                };
                if !empty {
                    match self.get_line_containing(id.as_ref().unwrap(), search_string) {
                        Ok(line) => return_value = line,
                        Err(LogFileError::Lock(e)) => lock_exception = Some(LogFileError::Lock(e)),
                        Err(e) => {
                            if *self.debug.lock().unwrap() || DEBUG.is_on() {
                                eprintln!("{}", e);
                            }
                        }
                    }
                }
            }
            Err(LogFileError::Lock(e)) => lock_exception = Some(LogFileError::Lock(e)),
            Err(e) => {
                if *self.debug.lock().unwrap() || DEBUG.is_on() {
                    eprintln!("{}", e);
                }
            }
        }
        // `finally`
        self.close_id(Some(handle), id.as_ref().map(|id| &**id));
        if let Some(lock_exception) = lock_exception {
            return Err(lock_exception);
        }
        Ok(return_value)
    }

    /// Java `backupOnce`.  Try to do a backup and set `backedUp` to true.  This prevents
    /// more than one backup being done on the file during the lifetime of the instance,
    /// which prevents the loss of data from a previous session because of too many
    /// backups being done.  Returns true if backup happened or there was no file to back
    /// up.  False if it can't back up for any reason including one time backup.
    fn backup_once(&self, handle: &Arc<Handle>) -> Result<bool, LogFileError> {
        if *self.backed_up.lock().unwrap() {
            return Ok(false);
        }
        if self.backup(handle)? {
            *self.backed_up.lock().unwrap() = true;
            return Ok(true);
        }
        Ok(false)
    }

    /// Java `backup(LogFile.Handle, boolean)`.  Works like `backupOnce` but can back up
    /// once per handle.  `backedUpHandle` should be set to true by the calling handle if
    /// this function returns true.
    fn backup_backed_up(
        &self,
        handle: &Arc<Handle>,
        backed_up: bool,
    ) -> Result<bool, LogFileError> {
        if backed_up {
            return Ok(false);
        }
        self.backup(handle)
    }

    /// Java `doubleBackupOnce`.  Returns true if backup happened or there was no file to
    /// back up.  False if it can't back up for any reason including one time backup.
    fn double_backup_once(&self, handle: &Arc<Handle>) -> Result<bool, LogFileError> {
        if *self.backed_up.lock().unwrap() {
            return Ok(false);
        }
        // BackupOnce functions must set backedUp.
        let backed_up = self.double_backup(handle)?;
        *self.backed_up.lock().unwrap() = backed_up;
        Ok(backed_up)
    }

    /// Java `doubleBackup`.  Returns true if backup happened or there was no file to back
    /// up.  False if it can't back up for any reason including one time backup.
    fn double_backup(&self, handle: &Arc<Handle>) -> Result<bool, LogFileError> {
        if *self.backed_up.lock().unwrap() {
            // One time backup is in use.
            return Ok(false);
        }
        // Back up the backup file first.
        let backup_file = self.create_backup_file_var();
        let backup_file_handle = LogFile::get_instance_stress_test(
            Some(&backup_file),
            false,
            handle.emergency_monitor.clone(),
        )?;
        backup_file_handle.backup()?;
        // Back up.
        handle.backup()
    }

    /// Java `isLocked()`.
    fn is_locked(&self) -> bool {
        self.lock.is_locked()
    }

    /// Java `backup(LogFile.Handle)`.  Delete the current backup file and rename the
    /// current file to be the new backup file.  The current backup file will not be
    /// deleted unless the current file exists.  Will not backup if `backedUp` is true
    /// (doesn't set `backedUp`).  Returns true if backup happened or there was no file to
    /// back up.  False if it can't back up for any reason including one time backup.
    fn backup(&self, handle: &Arc<Handle>) -> Result<bool, LogFileError> {
        let backup_file = self.create_backup_file_var();
        if *self.backed_up.lock().unwrap() {
            // A backupOnce function was called.
            return Ok(false);
        }
        handle.rename(None, None, Some(&backup_file), false, false, false)
    }

    /// Java private synchronized `copyToNumberedFile`.
    fn copy_to_numbered_file(
        &self,
        manager: Option<&'static dyn BaseManager>,
        extension_marker: Option<ExtensionMarker>,
        num_digits: i32,
        copy_from_id: &CopyFromId,
        timer: Option<&mut SleepTimer>,
    ) -> Result<bool, LogFileError> {
        // `if (file == null)`: the field is a `PathBuf` here and cannot be null.
        if !self.file.exists() {
            return Ok(true);
        }
        // Get the numbered file.
        let file_type = NumberedFileType::contruct_instance(
            Some(&crate::imod::etomo::util::utilities::java_io_file_get_name(
                &self.file.to_string_lossy(),
            )),
            extension_marker,
            num_digits,
        );
        // The source dereferences the result without a null check.
        let file_type = file_type.expect("java.lang.NullPointerException");
        let dir = self.file.parent().map(|dir| dir.to_path_buf());
        let search_results = file_type.search_directory(manager, dir.as_deref());
        let number = search_results.get_next_unused_number(&file_type);
        let numbered_file = match number {
            Some(number) => file_type.get_file(
                manager,
                Some(
                    &crate::imod::etomo::util::utilities::java_io_file_get_absolute_path(
                        &dir.clone()
                            .map(|dir| dir.to_string_lossy().to_string())
                            .unwrap_or("null".to_string()),
                    ),
                ),
                Some(&number.to_string()),
            ),
            None => search_results.get_oldest_file(),
        };
        let numbered_file = match numbered_file {
            None => {
                Reporter::new_log_file_exception(
                    self,
                    LogFileError::File(FileException::new(&format!(
                        "Warning:  Unable to copy {} to a numbered file.",
                        crate::imod::etomo::util::utilities::java_io_file_get_name(
                            &self.file.to_string_lossy()
                        )
                    ))),
                )
                .print();
                return Ok(false);
            }
            Some(numbered_file) => numbered_file,
        };
        self.copy(
            manager,
            None,
            copy_from_id,
            Some(&numbered_file),
            false,
            false,
            timer,
        )
    }

    /// Java `copy`.  `preserveCopyToFile`: don't preserve files that signify that a copy
    /// error has happened, unless there's another way to stop the user from continuing.
    fn copy(
        &self,
        manager: Option<&'static dyn BaseManager>,
        axis_id: Option<AxisID>,
        copy_from_id: &CopyFromId,
        copy_to_file: Option<&std::path::Path>,
        preserve_copy_to_file: bool,
        popup_err_msg: bool,
        mut timer: Option<&mut SleepTimer>,
    ) -> Result<bool, LogFileError> {
        let mut copy_to_file_id: Option<FileId> = None;
        let mut copy_to_handle: Option<Arc<Handle>> = None;
        let mut copy_to_safe_delete_flag: Option<SafeDeleteFlag> = None;
        // The source's `try`/`finally`; the `finally` block is repeated before each exit.
        let result = (|| -> Result<bool, LogFileError> {
            // `if (file == null)`: the field is a `PathBuf` here and cannot be null.
            let copy_to_file = match copy_to_file {
                None => {
                    eprintln!(
                        "Unable to copy {}.  destination parameter is null.",
                        self.file_name
                    );
                    return Ok(false);
                }
                Some(copy_to_file) => copy_to_file,
            };
            if !self.file.exists() {
                return Ok(true);
            }
            if self.equals_file(Some(copy_to_file)) {
                eprintln!("Unable to copy {} to itself.", self.file_name);
                return Ok(true);
            }
            self.lock.lock(
                Some(&*copy_from_id),
                Some(StandardBarString::CopyingFrom),
                None,
                timer.as_deref_mut(),
            )?;
            if !self.file.exists() {
                // nothing to copy
                return Ok(true);
            }

            // Lock the numbered file.
            copy_to_file_id = LogFile::build_handle_and_do_file_lock(
                Some(copy_to_file),
                copy_from_id
                    .handle
                    .as_ref()
                    .unwrap()
                    .emergency_monitor
                    .clone(),
                Some(StandardBarString::CopyingTo),
                timer.as_deref_mut(),
            )?;
            let copy_to_file_id = match &copy_to_file_id {
                None => return Ok(false),
                Some(copy_to_file_id) => copy_to_file_id,
            };
            copy_to_handle = copy_to_file_id.handle.clone();
            let copy_to_handle = match &copy_to_handle {
                None => return Ok(false),
                Some(copy_to_handle) => copy_to_handle,
            };

            // Delete destination file and copy.
            copy_to_safe_delete_flag =
                Some(copy_to_handle.log_file.safe_delete(preserve_copy_to_file));
            match LogFile::copy_file(Some(&self.file), Some(copy_to_file)) {
                Ok(()) => {
                    copy_to_handle.log_file.expunge(copy_to_safe_delete_flag);
                }
                Err(e) => {
                    copy_to_handle.log_file.undelete(copy_to_safe_delete_flag);
                    let mut stack_trace = StackTrace::new();
                    let title = "File Copy Failed";
                    let message = format!(
                        "Unable to copy {} to {}.{}",
                        self.file_name,
                        utilities::java_io_file_get_name(&copy_to_file.to_string_lossy()),
                        if utilities::is_windows_os()
                            || etomo_director::EtomoDirector::is_simulate_windows()
                        {
                            "  Close both files and try again."
                        } else {
                            ""
                        }
                    );
                    if popup_err_msg && !stack_trace.is_starting() && !stack_trace.is_exiting() {
                        // TODO(unit): needs etomo/ui/swing/UIHarness.java -
                        // `UIHarness.INSTANCE.openWarningMessageDialog(manager, message,
                        // title, axisID)`.
                        let _ = (&message, title, manager, axis_id);
                    } else {
                        eprintln!("{}\n{}", title, message);
                    }
                    return Err(LogFileError::LogFile(e));
                }
            }
            Ok(true)
        })();
        // `finally`
        self.lock.unlock(Some(&*copy_from_id));
        if let Some(copy_to_handle) = &copy_to_handle {
            copy_to_handle
                .log_file
                .lock
                .unlock(copy_to_file_id.as_ref().map(|id| &**id));
            copy_to_handle.log_file.expunge(copy_to_safe_delete_flag);
        }
        result
    }

    /// Java `rename`.  Delete the current destination file and rename the current file to
    /// be the new destination file.  The current destination file will not be deleted
    /// unless the current file exists.
    ///
    /// `preserveDestFile`: don't preserve files that signify that a copy error has
    /// happened, unless there's another way to stop the user from continuing.
    fn rename(
        &self,
        manager: Option<&'static dyn BaseManager>,
        axis_id: Option<AxisID>,
        file_id: &FileId,
        dest_file: Option<&std::path::Path>,
        allow_delete_dest_file: bool,
        preserve_dest_file: bool,
        popup_err_msg: bool,
        mut timer: Option<&mut SleepTimer>,
    ) -> Result<bool, LogFileError> {
        let mut dest_file_id: Option<FileId> = None;
        let mut dest_handle: Option<Arc<Handle>> = None;
        let result = (|| -> Result<bool, LogFileError> {
            // `if (file == null || destFile == null)`: `file` is a `PathBuf` here.
            let dest_file = match dest_file {
                None => {
                    Reporter::new_log_file_exception(
                        self,
                        LogFileError::File(FileException::new(&format!(
                            "Error:  null parameters.  File parameters must be set before \
                             running this function.  File:{},destFile:null",
                            self.file.to_string_lossy()
                        ))),
                    )
                    .print();
                    return Ok(false);
                }
                Some(dest_file) => dest_file,
            };
            if self.equals_file(Some(dest_file)) {
                // Nothing to do
                return Ok(true);
            }
            if !self.file.exists() {
                return Ok(true);
            }
            self.lock.lock(
                Some(&*file_id),
                Some(StandardBarString::Renaming),
                Some(&utilities::java_io_file_get_name(
                    &dest_file.to_string_lossy(),
                )),
                timer.as_deref_mut(),
            )?;
            if !self.file.exists() {
                // nothing to rename
                return Ok(true);
            }
            // Attempt to delete destination file.
            dest_file_id = LogFile::build_handle_and_do_file_lock(
                Some(dest_file),
                file_id.handle.as_ref().unwrap().emergency_monitor.clone(),
                Some(StandardBarString::RenamingTo),
                timer.as_deref_mut(),
            )?;
            let dest_file_id = match &dest_file_id {
                None => return Ok(false),
                Some(dest_file_id) => dest_file_id,
            };
            dest_handle = dest_file_id.handle.clone();
            let dest_handle = match &dest_handle {
                None => return Ok(false),
                Some(dest_handle) => dest_handle,
            };
            let mut deleted_dest_file = false;
            let mut dest_safe_delete_flag: Option<SafeDeleteFlag> = None;
            if dest_file.exists() {
                if !allow_delete_dest_file {
                    eprintln!(
                        "Warning: unable to rename {} to {}.  {} still exists.",
                        self.file_name,
                        utilities::java_io_file_get_name(&dest_file.to_string_lossy()),
                        utilities::java_io_file_get_name(&dest_file.to_string_lossy())
                    );
                    return Ok(false);
                }
                dest_safe_delete_flag = Some(dest_handle.log_file.safe_delete(preserve_dest_file));
                if let Some(dest_safe_delete_flag) = dest_safe_delete_flag {
                    deleted_dest_file = dest_safe_delete_flag.succeeded();
                }
            }
            let _ = deleted_dest_file;
            std::thread::sleep(std::time::Duration::from_millis(100));
            CLEAN_PRINT.print(Some(&format!(
                "(1) Rename {} to {}.",
                self.file_name,
                utilities::java_io_file_get_name(&dest_file.to_string_lossy())
            )));

            let success = std::fs::rename(&self.file, dest_file).is_ok();

            if !success {
                dest_handle.log_file.undelete(dest_safe_delete_flag);
                let errmsg = format!(
                    "Warning: Unable to rename {} to {}",
                    utilities::java_io_file_get_absolute_path(&self.file.to_string_lossy()),
                    utilities::java_io_file_get_absolute_path(&dest_file.to_string_lossy())
                );
                Reporter::new_log_file_exception_blocking_id(
                    self,
                    LogFileError::File(FileException::new(&errmsg)),
                    Some(&*file_id),
                )
                .throw_log_file_exception()?;
            }
            dest_handle.log_file.expunge(dest_safe_delete_flag);
            Ok(success)
        })();
        let result = match result {
            Err(e) => {
                let mut stack_trace = StackTrace::new();
                let err_title = "File Copy Failed";
                let err_message = format!(
                    "Unable to copy {} to {}.  {} both files and try again.",
                    self.file_name,
                    match dest_file {
                        None => "null".to_string(),
                        Some(dest_file) =>
                            utilities::java_io_file_get_name(&dest_file.to_string_lossy()),
                    },
                    if utilities::is_windows_os()
                        || etomo_director::EtomoDirector::is_simulate_windows()
                    {
                        "Close"
                    } else {
                        "Check"
                    }
                );
                if popup_err_msg && !stack_trace.is_starting() && !stack_trace.is_exiting() {
                    // TODO(unit): needs etomo/ui/swing/UIHarness.java -
                    // `UIHarness.INSTANCE.openWarningMessageDialog(manager, errMessage,
                    // errTitle, axisID)`.
                    let _ = (&err_message, err_title, manager, axis_id);
                } else {
                    eprintln!("\n{}\n{}\n", err_title, err_message);
                }
                Err(e)
            }
            Ok(success) => Ok(success),
        };
        // `finally`
        self.lock.unlock(Some(&*file_id));
        if let Some(dest_handle) = &dest_handle {
            dest_handle
                .log_file
                .lock
                .unlock(dest_file_id.as_ref().map(|id| &**id));
        }
        result
    }

    /// Java `buildHandleAndDoFileLock`.  Working with a handle to a totally different
    /// file is tricky.  This function makes it easier to avoid bugs.  It must remain
    /// static.
    fn build_handle_and_do_file_lock(
        another_file: Option<&std::path::Path>,
        emergency_monitor: Option<Arc<EmergencyMonitor>>,
        standard_bar_string: Option<StandardBarString>,
        timer: Option<&mut SleepTimer>,
    ) -> Result<Option<FileId>, LogFileError> {
        let mut file_id: Option<FileId> = None;
        let another_file = match another_file {
            None => return Ok(None),
            Some(another_file) => another_file,
        };
        let handle =
            LogFile::get_instance_stress_test(Some(another_file), false, emergency_monitor);
        let handle = match handle {
            Ok(handle) => handle,
            Err(e) => {
                eprintln!(
                    "Unable to open a file lock on {}.\n{}",
                    utilities::java_io_file_get_name(&another_file.to_string_lossy()),
                    e.get_message()
                );
                return Err(e);
            }
        };
        // `handle.logFile.new FileId(handle)`: the inner class instantiated against an
        // explicit enclosing `LogFile`, which is the handle's own.
        file_id = Some(FileId::new(&handle));
        let new_file_id = file_id.as_ref().unwrap();
        match handle
            .log_file
            .lock
            .lock(Some(&**new_file_id), standard_bar_string, None, timer)
        {
            Ok(()) => Ok(file_id),
            Err(e) => {
                eprintln!(
                    "Unable to open a file lock on {}.\n{}",
                    utilities::java_io_file_get_name(&another_file.to_string_lossy()),
                    e.get_message()
                );
                new_file_id.close();
                Err(e)
            }
        }
    }

    /// Java `create`.  Creates the file if it doesn't already exist.  Returns true if the
    /// file is created or already exists.
    fn create(
        &self,
        file_id: &FileId,
        timer: Option<&mut SleepTimer>,
    ) -> Result<bool, LogFileError> {
        let result = (|| -> Result<bool, LogFileError> {
            if self.file.exists() {
                return Ok(true);
            }
            self.lock.lock(
                Some(&**file_id),
                Some(StandardBarString::Creating),
                None,
                timer,
            )?;
            if self.file.exists() {
                // nothing to create
                return Ok(true);
            }
            let mut success = false;
            CLEAN_PRINT.print(Some(&format!("Create {}.", self.file_name)));
            for _i in 0..20 {
                let _ = std::fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&self.file);
                std::thread::sleep(std::time::Duration::from_millis(25));
                if self.file.exists() {
                    success = true;
                    break;
                }
            }
            if success {
                return Ok(true);
            }
            Reporter::new_log_file_exception_id_verbose(
                self,
                LogFileError::File(FileException::new(&format!(
                    "Error: unable to create {}",
                    utilities::java_io_file_get_absolute_path(&self.file.to_string_lossy())
                ))),
                Some(&**file_id),
                true,
            )
            .print()
            .throw_log_file_exception()?;
            Ok(false)
        })();
        // `finally`
        self.lock.unlock(Some(&**file_id));
        result
    }

    /// Java `delete(FileId, SleepTimer)`.  Returns true if the file does not exist or has
    /// been deleted.
    fn delete_file_id(
        &self,
        file_id: &FileId,
        timer: Option<&mut SleepTimer>,
    ) -> Result<bool, LogFileError> {
        let result = (|| -> Result<bool, LogFileError> {
            if !self.file.exists() {
                return Ok(true);
            }
            self.lock.lock(
                Some(&**file_id),
                Some(StandardBarString::Deleting),
                None,
                timer,
            )?;
            if !self.file.exists() {
                // nothing to delete
                return Ok(true);
            }
            LogFile::delete(&self.file);
            let success = !self.file.exists();
            if success {
                return Ok(true);
            }
            Reporter::new_log_file_exception_id_verbose(
                self,
                LogFileError::File(FileException::new(&format!(
                    "Error: unable to delete {}",
                    utilities::java_io_file_get_absolute_path(&self.file.to_string_lossy())
                ))),
                Some(&**file_id),
                true,
            )
            .print()
            .throw_log_file_exception()?;
            Ok(false)
        })();
        // `finally`
        self.lock.unlock(Some(&**file_id));
        result
    }

    /// Java `overrideLockIdResource`.  Close the resources associated with this id, and
    /// unlock the id.  Do this only for an id handle that is different from the
    /// `thisHandle` parameter.
    fn override_lock_id_resource(
        &self,
        this_handle: Option<&Arc<Handle>>,
        lock_id: Option<&Arc<Id>>,
    ) -> Result<bool, LogFileError> {
        if this_handle.is_none() || lock_id.is_none() || !self.lock.is_locked_id(lock_id) {
            return Ok(false);
        }
        self.close_resource(lock_id)?;
        Ok(true)
    }

    /// Java `openWriter(WriterId, boolean, SleepTimer)`.
    fn open_writer_append(
        &self,
        writer_id: &WriterId,
        append: bool,
        timer: Option<&mut SleepTimer>,
    ) -> Result<WriterId, LogFileError> {
        match (|| -> Result<WriterId, LogFileError> {
            self.lock.lock(
                Some(&**writer_id),
                Some(StandardBarString::Writing),
                None,
                timer,
            )?;
            self.create_writer(append, writer_id)?;
            Ok(writer_id.clone())
        })() {
            Ok(writer_id) => Ok(writer_id),
            Err(e) => {
                self.lock.unlock(Some(&**writer_id));
                Err(e)
            }
        }
    }

    /// Java `openWriter(WriterId, SleepTimer)`.
    fn open_writer(
        &self,
        writer_id: &WriterId,
        timer: Option<&mut SleepTimer>,
    ) -> Result<WriterId, LogFileError> {
        match (|| -> Result<WriterId, LogFileError> {
            self.lock.lock(
                Some(&**writer_id),
                Some(StandardBarString::Writing),
                None,
                timer,
            )?;
            self.create_writer(false, writer_id)?;
            Ok(writer_id.clone())
        })() {
            Ok(writer_id) => Ok(writer_id),
            Err(e) => {
                self.lock.unlock(Some(&**writer_id));
                Err(e)
            }
        }
    }

    /// Java `openForWriting`.  Run open with no wait limit.  This function can cause
    /// deadlock.
    fn open_for_writing(
        &self,
        writing_id: &WritingId,
        timer: Option<&mut SleepTimer>,
    ) -> Result<WritingId, LogFileError> {
        match self.lock.lock(
            Some(&**writing_id),
            Some(StandardBarString::Writing),
            None,
            timer,
        ) {
            Ok(()) => Ok(writing_id.clone()),
            Err(e) => {
                self.lock.unlock(Some(&**writing_id));
                Err(e)
            }
        }
    }

    /// Java `openInputStream`.  Opens the input stream.  Although this is a reader, it
    /// needs to exclude writers because it is used to read the entire file.  So the
    /// writer lock is used for now.
    fn open_input_stream(
        &self,
        input_stream_id: &InputStreamId,
        timer: Option<&mut SleepTimer>,
    ) -> Result<InputStreamId, LogFileError> {
        self.lock.lock(
            Some(&**input_stream_id),
            Some(StandardBarString::Reading),
            None,
            timer,
        )?;
        match self.create_input_stream(input_stream_id) {
            Ok(()) => Ok(input_stream_id.clone()),
            Err(e) => {
                self.lock.unlock(Some(&**input_stream_id));
                Err(e)
            }
        }
    }

    /// Java `openOutputStream`.  Opens the output stream.  Locks the WRITE lock and
    /// returns a writeId.
    fn open_output_stream(
        &self,
        output_stream_id: &OutputStreamId,
        timer: Option<&mut SleepTimer>,
    ) -> Result<OutputStreamId, LogFileError> {
        match (|| -> Result<OutputStreamId, LogFileError> {
            self.lock.lock(
                Some(&**output_stream_id),
                Some(StandardBarString::Writing),
                None,
                timer,
            )?;
            self.create_output_stream(output_stream_id)?;
            Ok(output_stream_id.clone())
        })() {
            Ok(output_stream_id) => Ok(output_stream_id),
            Err(e) => {
                self.lock.unlock(Some(&**output_stream_id));
                Err(e)
            }
        }
    }

    /// Java `openReader(ReaderId, SleepTimer)`.
    fn open_reader(
        &self,
        reader_id: &ReaderId,
        timer: Option<&mut SleepTimer>,
    ) -> Result<Option<ReaderId>, LogFileError> {
        self.open_reader_required(reader_id, true, timer)
    }

    /// Java `openReader(ReaderId, boolean, SleepTimer)`.
    fn open_reader_required(
        &self,
        reader_id: &ReaderId,
        required: bool,
        timer: Option<&mut SleepTimer>,
    ) -> Result<Option<ReaderId>, LogFileError> {
        match (|| -> Result<Option<ReaderId>, LogFileError> {
            self.lock.lock(
                Some(&**reader_id),
                Some(StandardBarString::Reading),
                None,
                timer,
            )?;
            let id_key = self.reading_token_list.make_key(self, Some(&**reader_id))?;
            if !self.reading_token_list.open_reading_token_required(
                &id_key,
                &self.file,
                ReaderType::Reader,
                required,
            )? {
                self.lock.unlock(Some(&**reader_id));
                return Ok(None);
            }
            Ok(Some(reader_id.clone()))
        })() {
            Ok(reader_id) => Ok(reader_id),
            Err(e) => {
                self.lock.unlock(Some(&**reader_id));
                Err(e)
            }
        }
    }

    /// Java `openBigBufferReader`.
    fn open_big_buffer_reader(
        &self,
        big_buffer_reader_id: &BigBufferReaderId,
        timer: Option<&mut SleepTimer>,
    ) -> Result<Option<BigBufferReaderId>, LogFileError> {
        match (|| -> Result<Option<BigBufferReaderId>, LogFileError> {
            self.lock.lock(
                Some(&**big_buffer_reader_id),
                Some(StandardBarString::Reading),
                None,
                timer,
            )?;
            let id_key = self
                .reading_token_list
                .make_key(self, Some(&**big_buffer_reader_id))?;
            if !self.reading_token_list.open_reading_token(
                &id_key,
                &self.file,
                ReaderType::BigBufferReader,
            )? {
                self.lock.unlock(Some(&**big_buffer_reader_id));
                return Ok(None);
            }
            Ok(Some(big_buffer_reader_id.clone()))
        })() {
            Ok(big_buffer_reader_id) => Ok(big_buffer_reader_id),
            Err(e) => {
                self.lock.unlock(Some(&**big_buffer_reader_id));
                Err(e)
            }
        }
    }

    /// Java `createStressTestHandle`.
    fn create_stress_test_handle(&self) -> Option<Arc<Handle>> {
        let mut stress_test_handle = self.stress_test_handle.lock().unwrap();
        if stress_test_handle.is_none() {
            // The stress test handle should not have an emergency monitor because it has
            // nothing to do with etomo functionality. It should just silently block.
            match LogFile::get_stress_test_instance(Some(&self.file)) {
                Ok(handle) => *stress_test_handle = Some(handle),
                Err(e) => eprintln!("{}", e),
            }
        }
        stress_test_handle.clone()
    }

    /// Java `openFileLock`.
    fn open_file_lock(&self, file_id: &FileId, timer: Option<&mut SleepTimer>) -> FileId {
        if self.lock.lock(Some(&**file_id), None, None, timer).is_err() {
            self.lock.unlock(Some(&**file_id));
        }
        file_id.clone()
    }

    /// Java `closeFileLockWithWait`.
    fn close_file_lock_with_wait(
        &self,
        file_id: &FileId,
        time_to_close: i64,
        _timer: Option<&mut SleepTimer>,
    ) {
        if self.lock.is_locked_id(Some(&**file_id)) {
            LogFile::do_required_sleep(time_to_close);
            self.lock.unlock(Some(&**file_id));
        }
        // `finally`
        self.lock.unlock(Some(&**file_id));
    }

    /// Java `openForReading`.
    fn open_for_reading(
        &self,
        reading_id: &ReadingId,
        timer: Option<&mut SleepTimer>,
    ) -> Result<Option<ReadingId>, LogFileError> {
        match (|| -> Result<Option<ReadingId>, LogFileError> {
            self.lock.lock(
                Some(&**reading_id),
                Some(StandardBarString::Reading),
                None,
                timer,
            )?;
            let id_key = self
                .reading_token_list
                .make_key(self, Some(&**reading_id))?;
            if !self.reading_token_list.open_reading_token(
                &id_key,
                &self.file,
                ReaderType::Reading,
            )? {
                self.lock.unlock(Some(&**reading_id));
                return Ok(None);
            }
            Ok(Some(reading_id.clone()))
        })() {
            Ok(reading_id) => Ok(reading_id),
            Err(e) => {
                self.lock.unlock(Some(&**reading_id));
                Err(e)
            }
        }
    }

    /// Java `openForCopyingFrom`.
    fn open_for_copying_from(
        &self,
        copy_from_id: &CopyFromId,
        timer: Option<&mut SleepTimer>,
    ) -> Result<Option<CopyFromId>, LogFileError> {
        match (|| -> Result<Option<CopyFromId>, LogFileError> {
            self.lock.lock(
                Some(&**copy_from_id),
                Some(StandardBarString::CopyingFrom),
                None,
                timer,
            )?;
            let id_key = self
                .reading_token_list
                .make_key(self, Some(&**copy_from_id))?;
            if !self.reading_token_list.open_reading_token(
                &id_key,
                &self.file,
                ReaderType::Reading,
            )? {
                self.lock.unlock(Some(&**copy_from_id));
                return Ok(None);
            }
            Ok(Some(copy_from_id.clone()))
        })() {
            Ok(copy_from_id) => Ok(copy_from_id),
            Err(e) => {
                self.lock.unlock(Some(&**copy_from_id));
                Err(e)
            }
        }
    }

    /// Java `closeId`.  Unlocks the id and closes the resource.
    fn close_id(&self, handle: Option<&Arc<Handle>>, id: Option<&Arc<Id>>) {
        // A handle can only close it's own ids.
        let handle = match handle {
            None => {
                eprintln!("Error: Missing handle");
                return;
            }
            Some(handle) => handle,
        };
        let id = match id {
            None => return,
            Some(id) => id,
        };
        if !id.equals_handle(Some(handle)) {
            eprintln!("Warning: Id ({}) does not match handle ({})", id, handle);
            return;
        }
        let _ = self.close_resource(Some(id));
        self.lock.unlock(Some(id));
    }

    /// Java `closeResource(Id)`.
    fn close_resource(&self, id: Option<&Arc<Id>>) -> Result<(), LogFileError> {
        let id = match id {
            None => return Ok(()),
            Some(id) => id,
        };
        let id_type = id.get_id_type();
        let lock_type = id_type.get_lock_type();
        if lock_type == LockType::Read {
            self.close_resource_read_id(Some(id))?;
        }
        if lock_type != LockType::Write {
            return Ok(());
        }
        if id_type == IdType::Writer {
            self.close_resource_writer_id()?;
        }
        if id_type == IdType::InputStream {
            self.close_resource_input_stream_id()?;
        }
        if id_type == IdType::OutputStream {
            self.close_resource_output_stream_id()?;
        }
        Ok(())
    }

    /// Java `closeResource(ReadId)`.
    fn close_resource_read_id(&self, read_id: Option<&Arc<Id>>) -> Result<(), LogFileError> {
        let mut reading_token: Option<usize> = None;
        let result = (|| -> Result<(), LogFileError> {
            let read_id = match read_id {
                None => return Ok(()),
                Some(read_id) => read_id,
            };
            let id_type = read_id.get_id_type();
            if id_type == IdType::Reading || id_type == IdType::CopyFrom {
                return Ok(());
            }
            let key = self.reading_token_list.make_key(self, Some(read_id))?;
            reading_token = self.reading_token_list.get_reading_token(&key);
            Ok(())
        })();
        // `finally`
        if let Some(reading_token) = reading_token {
            self.reading_token_list.array_list.lock().unwrap()[reading_token].close();
        }
        result
    }

    /// Java `readLine`.  Reads a line from the file.  Returns null when there are no more
    /// lines to read.
    fn read_line(&self, read_id: &ReaderId) -> Result<Option<String>, LogFileError> {
        if !self.lock.is_locked_id(Some(&**read_id)) {
            return Err(LogFileError::Unlocked(UnlockedException::new_id(
                Some(&**read_id),
                Some(self),
            )));
        }
        let key = self.reading_token_list.make_key(self, Some(&**read_id))?;
        let reader = self.reading_token_list.get_reader(&key);
        match reader {
            Some(reader) => {
                Ok(self.reading_token_list.array_list.lock().unwrap()[reader].read_line()?)
            }
            None => Ok(None),
        }
    }

    /// Java `searchForLastLine`.  Returns true if the last line of the file equals the
    /// line parameter.
    fn search_for_last_line(
        &self,
        id: &BigBufferReaderId,
        line: &str,
    ) -> Result<bool, LogFileError> {
        if !self.lock.is_locked_id(Some(&**id)) {
            return Err(LogFileError::Unlocked(UnlockedException::new_id(
                Some(&**id),
                Some(self),
            )));
        }
        let key = self.reading_token_list.make_key(self, Some(&**id))?;
        let reader = self.reading_token_list.get_big_buffer_reader(&key);
        match reader {
            Some(reader) => Ok(self.reading_token_list.array_list.lock().unwrap()[reader]
                .search_for_last_line(line)?),
            None => Ok(false),
        }
    }

    /// Java `getLineContaining(ReaderId, String)`.  Returns the first line that contains
    /// `searchString`.
    fn get_line_containing(
        &self,
        id: &ReaderId,
        search_string: &str,
    ) -> Result<Option<String>, LogFileError> {
        if !self.lock.is_locked_id(Some(&**id)) {
            return Err(LogFileError::Unlocked(UnlockedException::new_id(
                Some(&**id),
                Some(self),
            )));
        }
        let key = self.reading_token_list.make_key(self, Some(&**id))?;
        let reader = self.reading_token_list.get_reader(&key);
        if let Some(reader) = reader {
            let mut array_list = self.reading_token_list.array_list.lock().unwrap();
            let mut line = array_list[reader].read_line()?;
            while let Some(current_line) = line {
                if current_line.contains(search_string) {
                    return Ok(Some(current_line));
                }
                line = array_list[reader].read_line()?;
            }
        }
        Ok(None)
    }

    /// Java `load`.  `java.util.Properties.load(InputStream)` over the open input stream.
    fn load(
        &self,
        properties: &mut std::collections::BTreeMap<String, String>,
        input_stream_id: &InputStreamId,
    ) -> Result<(), LogFileError> {
        let mut input_stream = self.input_stream.lock().unwrap();
        if input_stream.is_none() || !self.lock.is_locked_id(Some(&**input_stream_id)) {
            return Err(LogFileError::Unlocked(UnlockedException::new_id(
                Some(&**input_stream_id),
                Some(self),
            )));
        }
        // `properties.load(inputStream)`: ISO-8859-1 lines, `#` and `!` comments, and a
        // `=`, `:` or whitespace separator, as etomo/storage/parameter_store.rs reads
        // them.
        let mut input = String::new();
        input_stream.as_mut().unwrap().read_to_string(&mut input)?;
        for line in input.lines() {
            let line = line.trim_start();
            if line.starts_with('#') || line.starts_with('!') || line.is_empty() {
                continue;
            }
            if let Some((key, value)) = line.split_once('=') {
                properties.insert(key.trim().to_owned(), value.trim().to_owned());
            } else if let Some((key, value)) = line.split_once(':') {
                properties.insert(key.trim().to_owned(), value.trim().to_owned());
            }
        }
        Ok(())
    }

    /// Java `store`.  `java.util.Properties.store(OutputStream, null)` over the open
    /// output stream.
    fn store(
        &self,
        properties: &std::collections::BTreeMap<String, String>,
        output_stream_id: &OutputStreamId,
    ) -> Result<(), LogFileError> {
        let mut output_stream = self.output_stream.lock().unwrap();
        if output_stream.is_none() || !self.lock.is_locked_id(Some(&**output_stream_id)) {
            return Err(LogFileError::Unlocked(UnlockedException::new_id(
                Some(&**output_stream_id),
                Some(self),
            )));
        }
        // `properties.store(outputStream, null)` writes a `#` line carrying
        // `new Date().toString()` and then one `key=value` line per property.
        let mut date = [0i8; 64];
        let now = unsafe { libc::time(std::ptr::null_mut()) };
        let mut broken_down: libc::tm = unsafe { std::mem::zeroed() };
        unsafe {
            libc::localtime_r(&now, &mut broken_down);
            libc::strftime(
                date.as_mut_ptr() as *mut libc::c_char,
                date.len(),
                c"%a %b %d %H:%M:%S %Z %Y".as_ptr(),
                &broken_down,
            );
        }
        let date = unsafe { std::ffi::CStr::from_ptr(date.as_ptr() as *const libc::c_char) };
        let mut output = String::new();
        output.push('#');
        output.push_str(&date.to_string_lossy());
        output.push('\n');
        for (key, value) in properties {
            output.push_str(key);
            output.push('=');
            output.push_str(value);
            output.push('\n');
        }
        output_stream
            .as_mut()
            .unwrap()
            .write_all(output.as_bytes())?;
        Ok(())
    }

    /// Java `write(String, WriterId)`.
    fn write(&self, string: Option<&str>, writer_id: &WriterId) -> Result<(), LogFileError> {
        let string = match string {
            None => return Ok(()),
            Some(string) => string,
        };
        let mut file_writer = self.file_writer.lock().unwrap();
        if file_writer.is_none() {
            return Err(LogFileError::Unlocked(UnlockedException::new_message_id(
                &format!("fileWriter is null.  Unable to write: {}", string),
                Some(&**writer_id),
                Some(self),
            )));
        }
        if !self.lock.is_locked_id(Some(&**writer_id)) {
            return Err(LogFileError::Unlocked(UnlockedException::new_message_id(
                "not locked",
                Some(&**writer_id),
                Some(self),
            )));
        }
        file_writer.as_mut().unwrap().write_all(string.as_bytes())?;
        Ok(())
    }

    /// Java `write(char, WriterId)`.
    fn write_char(&self, ch: char, writer_id: &WriterId) -> Result<(), LogFileError> {
        let mut file_writer = self.file_writer.lock().unwrap();
        if file_writer.is_none() || !self.lock.is_locked_id(Some(&**writer_id)) {
            return Err(LogFileError::Unlocked(UnlockedException::new_id(
                Some(&**writer_id),
                Some(self),
            )));
        }
        file_writer
            .as_mut()
            .unwrap()
            .write_all(ch.to_string().as_bytes())?;
        Ok(())
    }

    /// Java `write(Character, WriterId)`.
    fn write_character(&self, ch: Option<char>, writer_id: &WriterId) -> Result<(), LogFileError> {
        self.write_char(ch.unwrap(), writer_id)
    }

    /// Java `newLine`.
    fn new_line(&self, writer_id: &WriterId) -> Result<(), LogFileError> {
        let mut file_writer = self.file_writer.lock().unwrap();
        if file_writer.is_none() || !self.lock.is_locked_id(Some(&**writer_id)) {
            return Err(LogFileError::Unlocked(UnlockedException::new_id(
                Some(&**writer_id),
                Some(self),
            )));
        }
        // `BufferedWriter.newLine()` writes the `line.separator` property.
        file_writer.as_mut().unwrap().write_all(b"\n")?;
        Ok(())
    }

    /// Java `writeStackTrace`.
    fn write_stack_trace(&self, title: &str, writer_id: &WriterId) -> Result<(), LogFileError> {
        if self.output_stream.lock().unwrap().is_none()
            || !self.lock.is_locked_id(Some(&**writer_id))
        {
            return Err(LogFileError::Unlocked(UnlockedException::new_id(
                Some(&**writer_id),
                Some(self),
            )));
        }
        // `new Exception(title).getStackTrace()`; see etomo/util/stack_trace.rs for why
        // this process contributes no `StackTraceElement`s.
        let trace: Vec<String> = Vec::new();
        self.write(Some(title), writer_id)?;
        for element in trace.iter() {
            self.write(Some(element), writer_id)?;
        }
        Ok(())
    }

    /// Java `flush`.
    fn flush(&self, writer_id: &WriterId) -> Result<(), LogFileError> {
        if !self.lock.is_locked_id(Some(&**writer_id)) {
            return Err(LogFileError::Unlocked(UnlockedException::new_id(
                Some(&**writer_id),
                Some(self),
            )));
        }
        CLEAN_PRINT.print(Some(&format!("Flush {} buffered writer.", self.file_name)));
        let mut file_writer = self.file_writer.lock().unwrap();
        match file_writer.as_mut() {
            Some(file_writer) => {
                file_writer.flush()?;
            }
            None => {
                // `catch (final NullPointerException e)`
                if *self.debug.lock().unwrap() || DEBUG.is_on() {
                    eprintln!("Nothing to flush:{}", self);
                }
            }
        }
        Ok(())
    }

    /// Java `createWriter`.
    fn create_writer(&self, append: bool, writer_id: &WriterId) -> Result<(), LogFileError> {
        let mut file_writer = self.file_writer.lock().unwrap();
        if file_writer.is_none() {
            CLEAN_PRINT.print(Some(&format!("Open writer for {}.", self.file_name)));
            match std::fs::OpenOptions::new()
                .write(true)
                .create(true)
                .append(append)
                .truncate(!append)
                .open(&self.file)
            {
                Ok(opened) => {
                    CLEAN_PRINT.print(Some(&format!(
                        "Open buffered writer for {}.",
                        self.file_name
                    )));
                    *file_writer = Some(std::io::BufWriter::new(opened));
                }
                Err(e) => {
                    drop(file_writer);
                    if let Err(e0) = self.close_resource_writer_id() {
                        if *self.debug.lock().unwrap() || DEBUG.is_on() {
                            eprintln!("{}", e0);
                        }
                    }
                    return Err(LogFileError::Io(e));
                }
            }
        }
        Ok(())
    }

    /// Java `createInputStream`.
    fn create_input_stream(&self, _input_stream_id: &InputStreamId) -> Result<(), LogFileError> {
        let mut input_stream = self.input_stream.lock().unwrap();
        if input_stream.is_none() {
            CLEAN_PRINT.print(Some(&format!("Open input stream for {}.", self.file_name)));
            match std::fs::File::open(&self.file) {
                Ok(opened) => *input_stream = Some(opened),
                Err(e) => {
                    drop(input_stream);
                    if let Err(e0) = self.close_resource_input_stream_id() {
                        if *self.debug.lock().unwrap() || DEBUG.is_on() {
                            eprintln!("{}", e0);
                        }
                    }
                    return Err(LogFileError::Io(e));
                }
            }
        }
        Ok(())
    }

    /// Java `createOutputStream`.
    fn create_output_stream(&self, _output_stream_id: &OutputStreamId) -> Result<(), LogFileError> {
        let mut output_stream = self.output_stream.lock().unwrap();
        if output_stream.is_none() {
            CLEAN_PRINT.print(Some(&format!("Open output stream for {}.", self.file_name)));
            match std::fs::File::create(&self.file) {
                Ok(opened) => *output_stream = Some(opened),
                Err(e) => {
                    drop(output_stream);
                    if let Err(e0) = self.close_resource_output_stream_id() {
                        if *self.debug.lock().unwrap() || DEBUG.is_on() {
                            eprintln!("{}", e0);
                        }
                    }
                    return Err(LogFileError::Io(e));
                }
            }
        }
        Ok(())
    }

    /// Java `closeResource(WriterId)`.
    fn close_resource_writer_id(&self) -> Result<(), LogFileError> {
        // Java closes the `BufferedWriter` in the `try` and the `FileWriter` in the
        // `finally`; one `BufWriter` here holds both, and dropping it closes the file.
        let mut file_writer = self.file_writer.lock().unwrap();
        if file_writer.is_some() {
            CLEAN_PRINT.print(Some(&format!(
                "Close buffered writer for {}.",
                self.file_name
            )));
            file_writer.as_mut().unwrap().flush()?;
            CLEAN_PRINT.print(Some(&format!("Close writer for {}.", self.file_name)));
            *file_writer = None;
        }
        Ok(())
    }

    /// Java `closeResource(InputStreamId)`.
    fn close_resource_input_stream_id(&self) -> Result<(), LogFileError> {
        let mut input_stream = self.input_stream.lock().unwrap();
        if input_stream.is_some() {
            CLEAN_PRINT.print(Some(&format!("Close input stream for {}.", self.file_name)));
            *input_stream = None;
        }
        Ok(())
    }

    /// Java `closeResource(OutputStreamId)`.
    fn close_resource_output_stream_id(&self) -> Result<(), LogFileError> {
        let mut output_stream = self.output_stream.lock().unwrap();
        if output_stream.is_some() {
            CLEAN_PRINT.print(Some(&format!(
                "Close output stream for {}.",
                self.file_name
            )));
            *output_stream = None;
        }
        Ok(())
    }

    /// Java `isLocked(Id)`.  Returns true if the id is part of the lock.
    fn is_locked_id(&self, id: Option<&Arc<Id>>) -> bool {
        self.lock.is_locked_id(id)
    }

    /// Java `noLocks`.
    fn no_locks(&self) -> bool {
        !self.is_locked()
    }

    /// Java `getWriterIdForTest`.
    fn get_writer_id_for_test(&self, handle: &Arc<Handle>) -> Option<WriterId> {
        if etomo_director::ARGUMENTS.lock().unwrap().is_test() {
            return Some(WriterId::new(handle));
        }
        None
    }

    /// Java `getWritingIdForTest`.
    fn get_writing_id_for_test(&self, handle: &Arc<Handle>) -> Option<WritingId> {
        if etomo_director::ARGUMENTS.lock().unwrap().is_test() {
            return Some(WritingId::new(handle));
        }
        None
    }

    /// Java `getReaderIdForTest`.
    fn get_reader_id_for_test(&self, handle: &Arc<Handle>) -> Option<ReaderId> {
        if etomo_director::ARGUMENTS.lock().unwrap().is_test() {
            return Some(ReaderId::new(handle));
        }
        None
    }

    /// Java `isBackedup`.
    pub fn is_backedup(&self) -> bool {
        *self.backed_up.lock().unwrap()
    }

    /// Java `copyFile(File, File)`.
    fn copy_file(
        source: Option<&std::path::Path>,
        destination: Option<&std::path::Path>,
    ) -> Result<(), LogFileException> {
        let (source, destination) = match (source, destination) {
            (None, _) | (_, None) => return Ok(()),
            (Some(source), Some(destination)) if source == destination => return Ok(()),
            (Some(source), Some(destination)) => (source, destination),
        };
        let source_name = utilities::java_io_file_get_name(&source.to_string_lossy());
        let destination_name = utilities::java_io_file_get_name(&destination.to_string_lossy());
        // Copy file
        let result = (|| -> std::io::Result<()> {
            let mut source_stream = std::fs::File::open(source)?;
            let mut dest_stream = std::fs::File::create(destination)?;
            // `sourceChannel.transferTo(0L, sourceChannel.size(), destChannel)`; the
            // buffered fallback the source keeps for an IOException from the channel copy
            // has the same result.
            CLEAN_PRINT.print(Some(&format!(
                "(1) Copy {} to {}.",
                source_name, destination_name
            )));
            std::io::copy(&mut source_stream, &mut dest_stream)?;
            Ok(())
        })();
        if let Err(e) = result {
            eprintln!(
                "Unable to copy from {} to {}.\n{}",
                source_name, destination_name, e
            );
            return Err(LogFileException::new(&e.to_string()));
        }
        Ok(())
    }

    /// Java `safeDelete`.  Renames `file` to `tempFile` or deletes `file`.  Rename is
    /// done when `preserveFile` is true.  Delete is done when `preserveFile` is false and
    /// as a back up to renaming.
    fn safe_delete(&self, preserve_file: bool) -> SafeDeleteFlag {
        if !self.file.exists() {
            return SafeDeleteFlag::DoesNotExist;
        }
        if !preserve_file || self.temp_file.is_none() {
            if LogFile::delete(&self.file) {
                return SafeDeleteFlag::Deleted;
            }
            return SafeDeleteFlag::FailedDelete;
        }
        let temp_file = self.temp_file.as_ref().unwrap();
        if temp_file.exists() {
            // Make it a little easier to rescue a temp file.
            if let Some(backup_temp_file) = self.backup_temp_file.as_ref() {
                LogFile::delete(backup_temp_file);
                if std::fs::rename(temp_file, backup_temp_file).is_err() {
                    LogFile::delete(temp_file);
                }
            } else {
                LogFile::delete(temp_file);
            }
        }
        CLEAN_PRINT.print(Some(&format!(
            "Safe Delete: rename {} to {}.",
            self.file_name,
            utilities::java_io_file_get_name(&temp_file.to_string_lossy())
        )));
        if std::fs::rename(&self.file, temp_file).is_ok() {
            for _i in 0..50 {
                std::thread::sleep(std::time::Duration::from_millis(25));
                if !self.file.exists() {
                    return SafeDeleteFlag::Renamed;
                }
            }
            // `new Reporter(...)` is blocked; its `print()` writes the message and the
            // lock state to stderr.  See the Reporter entry in the frontier marker above.
            eprintln!(
                "Error: Unable to rename  {} to {}.",
                utilities::java_io_file_get_name(&self.file.to_string_lossy()),
                utilities::java_io_file_get_name(&temp_file.to_string_lossy())
            );
            return SafeDeleteFlag::RenamedButStillExists;
        } else if LogFile::delete(&self.file) {
            return SafeDeleteFlag::Deleted;
        }
        SafeDeleteFlag::FailedDelete
    }

    /// Java `undelete`.  If `safeDelete` renamed `file` to the temp file, this function
    /// renames the temp file back to `file`.  Returns true if this is done.
    fn undelete(&self, safe_delete_flag: Option<SafeDeleteFlag>) -> bool {
        let temp_file = match (&self.temp_file, safe_delete_flag) {
            (None, _) | (_, None) => return false,
            (_, Some(safe_delete_flag)) if !safe_delete_flag.renamed() => return false,
            (Some(temp_file), _) => temp_file,
        };
        if self.file.exists() {
            // Could this be an invalid, partially created file that's blocking tempFile?
            eprintln!(
                "Unable to rename {} back to {}",
                utilities::java_io_file_get_name(&temp_file.to_string_lossy()),
                self.file_name
            );
            return false;
        }
        CLEAN_PRINT.print(Some(&format!(
            "Undelete: rename {} to {}.",
            utilities::java_io_file_get_name(&temp_file.to_string_lossy()),
            self.file_name
        )));
        std::fs::rename(temp_file, &self.file).is_ok()
    }

    /// Java `expunge`.  If `safeDelete` renamed `file` to the temp file, this function
    /// removes the temp file.  Returns true if this was done.
    fn expunge(&self, safe_delete_flag: Option<SafeDeleteFlag>) -> bool {
        let temp_file = match (&self.temp_file, safe_delete_flag) {
            (None, _) | (_, None) => return true,
            (Some(temp_file), _) if !temp_file.exists() => return true,
            (_, Some(safe_delete_flag)) if !safe_delete_flag.renamed() => return true,
            (Some(temp_file), _) => temp_file,
        };
        CLEAN_PRINT.print(Some(&format!(
            "Expunge: delete {}.",
            utilities::java_io_file_get_name(&temp_file.to_string_lossy())
        )));
        LogFile::delete(temp_file)
    }

    /// Java `delete(File)`.  Deletes `file` with multiple tries.  Returns true if the
    /// file doesn't exist or was deleted.
    fn delete(file: &std::path::Path) -> bool {
        if !file.exists() {
            return true;
        }
        let mut i: i32;
        CLEAN_PRINT.print(Some(&format!(
            "(2) Delete {}.",
            utilities::java_io_file_get_name(&file.to_string_lossy())
        )));
        i = 0;
        while i < 50 {
            let _ = std::fs::remove_file(file);
            std::thread::sleep(std::time::Duration::from_millis(25));
            if !file.exists() {
                return true;
            }
            i += 1;
        }
        if !file.exists() {
            return true;
        }
        // `new Reporter(...).print()`; see the frontier marker above.
        eprintln!(
            "Error: Unable to remove {} after {} tries.",
            utilities::java_io_file_get_name(&file.to_string_lossy()),
            i
        );
        false
    }

    /// Java `doRequiredSleep`.
    fn do_required_sleep(required_sleep: i64) {
        // Rust's `thread::sleep` is not interruptible, so the source's
        // InterruptedException retry loop cannot be entered.
        if required_sleep > 0 {
            std::thread::sleep(std::time::Duration::from_millis(required_sleep as u64));
        }
    }

    /// Java `setDebug`.
    fn set_debug(&self, input: bool) {
        *self.debug.lock().unwrap() = input;
        self.reading_token_list.set_debug(input);
    }

    /// Java `createBackupFileVar`.
    fn create_backup_file_var(&self) -> std::path::PathBuf {
        std::path::PathBuf::from(
            utilities::java_io_file_get_absolute_path(&self.file.to_string_lossy())
                + &dataset_files::BACKUP_CHAR.to_string(),
        )
    }

    /// Java `createDoubleBackupFileVar`.
    fn create_double_backup_file_var(&self) -> std::path::PathBuf {
        std::path::PathBuf::from(
            utilities::java_io_file_get_absolute_path(&self.file.to_string_lossy())
                + &dataset_files::BACKUP_CHAR.to_string()
                + &dataset_files::BACKUP_CHAR.to_string(),
        )
    }

    /// Java `exists`.
    fn exists(&self) -> bool {
        self.file.exists()
    }

    /// Java `lastModified`.
    fn last_modified(&self) -> i64 {
        utilities::java_io_file_last_modified(&self.file.to_string_lossy())
    }

    /// Java `getAbsolutePath`.
    fn get_absolute_path(&self) -> String {
        utilities::java_io_file_get_absolute_path(&self.file.to_string_lossy())
    }

    /// Java `isDirectory`.
    fn is_directory(&self) -> bool {
        self.file.is_dir()
    }

    /// Java `getName`.
    fn get_name(&self) -> String {
        utilities::java_io_file_get_name(&self.file.to_string_lossy())
    }
}

/// Java `toString` on `LogFile`.
impl std::fmt::Display for LogFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[fileAbsolutePath={}]", self.file_absolute_path)
    }
}

/// `java.util.UUID.randomUUID().toString().replace("-", "")`: 32 lower-case hexadecimal
/// digits of a version-4 UUID.
fn java_util_uuid_random_uuid_no_dashes() -> String {
    // A process-local counter mixed with the clock; the source only needs a value that
    // does not collide with an existing file or map key, and retries until it does not.
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let counter = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_nanos() as u64)
        .unwrap_or(0);
    let mut bytes = [0u8; 16];
    bytes[0..8].copy_from_slice(&nanos.to_be_bytes());
    bytes[8..16].copy_from_slice(&(counter ^ std::process::id() as u64).to_be_bytes());
    // Version 4 and the IETF variant, as java.util.UUID.randomUUID sets them.
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

/// `java.nio.file.Files.isSameFile(Path, Path)`: equal paths are the same file, and
/// otherwise the two files' keys are compared.  `Err` is the source's `IOException`.
fn java_nio_file_files_is_same_file(
    path1: &std::path::Path,
    path2: &std::path::Path,
) -> std::io::Result<bool> {
    use std::os::unix::fs::MetadataExt;
    if path1 == path2 {
        return Ok(true);
    }
    let metadata1 = std::fs::metadata(path1)?;
    let metadata2 = std::fs::metadata(path2)?;
    Ok(metadata1.dev() == metadata2.dev() && metadata1.ino() == metadata2.ino())
}

/// Java `SafeDeleteFlag`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SafeDeleteFlag {
    /// Java `RENAMED(true, true)`.
    Renamed,
    /// Java `DELETED(true, false)`.
    Deleted,
    /// Java `DOES_NOT_EXIST(true, false)`.
    DoesNotExist,
    /// Java `RENAMED_BUT_STILL_EXISTS(false, true)`.
    RenamedButStillExists,
    /// Java `FAILED_DELETE(false, false)`.
    FailedDelete,
}

impl SafeDeleteFlag {
    /// Java field `succeeded`.
    pub fn succeeded(self) -> bool {
        matches!(
            self,
            SafeDeleteFlag::Renamed | SafeDeleteFlag::Deleted | SafeDeleteFlag::DoesNotExist
        )
    }

    /// Java field `renamed`.
    pub fn renamed(self) -> bool {
        matches!(
            self,
            SafeDeleteFlag::Renamed | SafeDeleteFlag::RenamedButStillExists
        )
    }
}

/// Java `LockType`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LockType {
    /// Java `READ = new LockType("read")`.
    Read,
    /// Java `WRITE = new LockType("write")`.
    Write,
    /// Java `FILE = new LockType("file")`.
    File,
}

/// Java `toString` on `LockType`, which returns the `name` field.
impl std::fmt::Display for LockType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            LockType::Read => "read",
            LockType::Write => "write",
            LockType::File => "file",
        })
    }
}

/// Java `IdType`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IdType {
    /// Java `FILE = new IdType("file", LockType.FILE)`.
    File,
    /// Java `WRITING = new IdType("writing", LockType.WRITE)`.
    Writing,
    /// Java `WRITER = new IdType("writer", LockType.WRITE)`.
    Writer,
    /// Java `OUTPUT_STREAM = new IdType("output stream", LockType.WRITE)`.
    OutputStream,
    /// Java `INPUT_STREAM = new IdType("input stream", LockType.WRITE)`.  The input
    /// stream can't coexist with other input streams, so it works to classify it as a
    /// writer.
    InputStream,
    /// Java `COPY_FROM = new IdType("copy from", LockType.READ)`.  The file being copied
    /// needs a lock that allows read access while preventing modifications or deletions.
    CopyFrom,
    /// Java `READING = new IdType("reading", LockType.READ)`.
    Reading,
    /// Java `READER = new IdType("reader", LockType.READ)`.
    Reader,
    /// Java `BIG_BUFFER_READER = new IdType("high capacity reader", LockType.READ)`.
    BigBufferReader,
}

impl IdType {
    /// Java field `lockType`, and `getLockType`.
    pub fn get_lock_type(self) -> LockType {
        match self {
            IdType::File => LockType::File,
            IdType::Writing | IdType::Writer | IdType::OutputStream | IdType::InputStream => {
                LockType::Write
            }
            IdType::CopyFrom | IdType::Reading | IdType::Reader | IdType::BigBufferReader => {
                LockType::Read
            }
        }
    }

    /// Java `getDescr`, which returns the `name` field.
    pub fn get_descr(self) -> &'static str {
        match self {
            IdType::File => "file",
            IdType::Writing => "writing",
            IdType::Writer => "writer",
            IdType::OutputStream => "output stream",
            IdType::InputStream => "input stream",
            IdType::CopyFrom => "copy from",
            IdType::Reading => "reading",
            IdType::Reader => "reader",
            IdType::BigBufferReader => "high capacity reader",
        }
    }
}

/// Java `toString` on `IdType`.
impl std::fmt::Display for IdType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.get_descr())
    }
}

/// Java `ReaderType`.  The source declares three identity-only singletons.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReaderType {
    /// Java `READING`.
    Reading,
    /// Java `READER`.
    Reader,
    /// Java `BIG_BUFFER_READER`.
    BigBufferReader,
}

/// Java `overrideLocks`.  Tells the `Lock` inner class what to do with blocking locks.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(non_camel_case_types)]
pub enum overrideLocks {
    /// Java `NO_OVERRIDE("Waiting for blocking file locks to be removed.")`.  Prevents
    /// overrides and can't be escalated.
    NoOverride,
    /// Java `OVERRIDE_ABANDONED_LOCKS_FROM_OTHER_HANDLES("Removing blocking file locks
    /// left over from previous runs.")`.
    OverrideAbandonedLocksFromOtherHandles,
    /// Java `OVERRIDE_ABANDONED_LOCKS_FROM_ALL_HANDLES("Removing blocking file locks that
    /// are inactive.")`.
    OverrideAbandonedLocksFromAllHandles,
    /// Java `OVERRIDE_ALL("Removing blocking file locks.")`.
    OverrideAll,
}

impl overrideLocks {
    /// Java field `descr`.
    pub fn descr(self) -> &'static str {
        match self {
            overrideLocks::NoOverride => "Waiting for blocking file locks to be removed.",
            overrideLocks::OverrideAbandonedLocksFromOtherHandles => {
                "Removing blocking file locks left over from previous runs."
            }
            overrideLocks::OverrideAbandonedLocksFromAllHandles => {
                "Removing blocking file locks that are inactive."
            }
            overrideLocks::OverrideAll => "Removing blocking file locks.",
        }
    }

    /// Java `escalate`.
    pub fn escalate(self) -> Option<overrideLocks> {
        if self == overrideLocks::OverrideAbandonedLocksFromOtherHandles {
            return Some(overrideLocks::OverrideAbandonedLocksFromAllHandles);
        }
        if self == overrideLocks::OverrideAbandonedLocksFromAllHandles {
            return Some(overrideLocks::OverrideAll);
        }
        None
    }
}

/// Java `SleepTimer`.
pub struct SleepTimer {
    /// Java field `testLevel`.
    test_level: Option<crate::imod::etomo::arguments::TestLevel>,
    /// Java field `sleepLimit`: the total amount of wait time allowed.
    sleep_limit: i64,
    /// Java field `waitSleepRunningTotal`.
    wait_sleep_running_total: i64,
    /// Java field `error`.
    error: bool,
}

impl SleepTimer {
    /// Java `SleepTimer()`.
    pub fn new() -> SleepTimer {
        let test_level = etomo_director::ARGUMENTS.lock().unwrap().get_test_level();
        let sleep_limit = match test_level {
            None => SLEEP_LIMIT,
            Some(test_level) if test_level.is_sleep_allowed() => SLEEP_LIMIT,
            // When sleepAllowed is off, any blocking file lock causes an immediate
            // failure.
            Some(_) => 0,
        };
        SleepTimer {
            test_level,
            sleep_limit,
            wait_sleep_running_total: 0,
            error: false,
        }
    }

    /// Java `isSleepLimitReached`.
    pub fn is_sleep_limit_reached(&self) -> bool {
        self.wait_sleep_running_total >= self.sleep_limit
    }

    /// Java `doSleep`.
    pub fn do_sleep(&mut self) {
        if self.is_sleep_limit_reached() {
            return;
        }
        // Rust's `thread::sleep` is not interruptible, so the source's
        // InterruptedException branch - which credits the partial sleep and, when nothing
        // was slept, adds SLEEP/10 and sets `error` - cannot be entered.
        std::thread::sleep(std::time::Duration::from_millis(SLEEP as u64));
        let slept = SLEEP;
        if slept > 0 {
            self.wait_sleep_running_total += slept;
        } else {
            // Something went wrong. Avoid an infinite loop.
            self.wait_sleep_running_total += SLEEP / 10;
            self.error = true;
        }
    }
}

/// Java `LogFileException`.  An abstract checked exception; `UnlockedException` and
/// `FileException` extend it, and `copyFile` throws an anonymous subclass of it.
#[derive(Clone, Debug)]
pub struct LogFileException {
    /// `java.lang.Throwable.getMessage()`.
    message: String,
    /// `java.lang.Throwable.getCause()`.
    cause: Option<String>,
}

impl LogFileException {
    /// Java `LogFileException(String)`.
    pub fn new(message: &str) -> LogFileException {
        LogFileException {
            message: message.to_string(),
            cause: None,
        }
    }

    /// Java `LogFileException(String, Throwable)`.
    pub fn new_with_cause(message: &str, cause: &str) -> LogFileException {
        LogFileException {
            message: message.to_string(),
            cause: Some(cause.to_string()),
        }
    }

    /// `java.lang.Throwable.getMessage()`.
    pub fn get_message(&self) -> &str {
        &self.message
    }

    /// `java.lang.Throwable.getCause()`.
    pub fn get_cause(&self) -> Option<&str> {
        self.cause.as_deref()
    }
}

impl std::fmt::Display for LogFileException {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for LogFileException {}

/// Java `FileException`.  For a file problem.
#[derive(Clone, Debug)]
pub struct FileException {
    /// The `LogFileException` superclass state.
    log_file_exception: LogFileException,
}

impl std::ops::Deref for FileException {
    type Target = LogFileException;
    fn deref(&self) -> &LogFileException {
        &self.log_file_exception
    }
}

impl FileException {
    /// Java `FileException(String)`.
    pub fn new(message: &str) -> FileException {
        FileException {
            log_file_exception: LogFileException::new(message),
        }
    }

    /// Java `FileException(String, Throwable)`.
    pub fn new_with_cause(message: &str, cause: &str) -> FileException {
        FileException {
            log_file_exception: LogFileException::new_with_cause(message, cause),
        }
    }
}

impl std::fmt::Display for FileException {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.log_file_exception.message)
    }
}

impl std::error::Error for FileException {}

/// Java `UnlockedException`.  For a missing id, an id that is not saved in the lock, or a
/// missing resource.
#[derive(Clone, Debug)]
pub struct UnlockedException {
    /// The `LogFileException` superclass state.
    log_file_exception: LogFileException,
}

impl std::ops::Deref for UnlockedException {
    type Target = LogFileException;
    fn deref(&self) -> &LogFileException {
        &self.log_file_exception
    }
}

impl UnlockedException {
    /// Java `UnlockedException(String)`, which forwards to
    /// `UnlockedException(String, LockType, Id, LogFile)` with three nulls; `toString`
    /// then returns the message unchanged.
    pub fn new(message: &str) -> UnlockedException {
        UnlockedException {
            log_file_exception: LogFileException::new(&UnlockedException::to_string_static(
                Some(message),
                None,
                None,
                None,
            )),
        }
    }

    /// Java `UnlockedException(String, Throwable)`.
    pub fn new_with_cause(message: &str, cause: &str) -> UnlockedException {
        UnlockedException {
            log_file_exception: LogFileException::new_with_cause(
                &UnlockedException::to_string_static(Some(message), None, None, None),
                cause,
            ),
        }
    }

    /// Java `UnlockedException(String, LockType, Id, LogFile)`.
    fn new_lock_type(
        message: Option<&str>,
        lock_type: Option<LockType>,
        id: Option<&Arc<Id>>,
        log_file: Option<&LogFile>,
    ) -> UnlockedException {
        UnlockedException {
            log_file_exception: LogFileException::new(&UnlockedException::to_string_static(
                message, lock_type, id, log_file,
            )),
        }
    }

    /// Java `UnlockedException(String, Throwable, LockType, Id, LogFile)`.
    fn new_lock_type_with_cause(
        message: Option<&str>,
        cause: &str,
        lock_type: Option<LockType>,
        id: Option<&Arc<Id>>,
        log_file: Option<&LogFile>,
    ) -> UnlockedException {
        UnlockedException {
            log_file_exception: LogFileException::new_with_cause(
                &UnlockedException::to_string_static(message, lock_type, id, log_file),
                cause,
            ),
        }
    }

    /// Java `UnlockedException(String, Id, LogFile)`.
    pub fn new_message_id(
        message: &str,
        id: Option<&Arc<Id>>,
        log_file: Option<&LogFile>,
    ) -> UnlockedException {
        UnlockedException::new_lock_type(Some(message), None, id, log_file)
    }

    /// Java `UnlockedException(Id, LogFile)`.
    pub fn new_id(id: Option<&Arc<Id>>, log_file: Option<&LogFile>) -> UnlockedException {
        UnlockedException::new_lock_type(None, None, id, log_file)
    }

    /// Java `toString(String, LockType, Id, LogFile)`.
    fn to_string_static(
        message: Option<&str>,
        lock_type: Option<LockType>,
        id: Option<&Arc<Id>>,
        log_file: Option<&LogFile>,
    ) -> String {
        let message = match message {
            None => "Unlocked log file.  Cannot access.",
            Some(message) => message,
        };
        if log_file.is_none() && id.is_none() && lock_type.is_none() {
            return message.to_string();
        }
        message.to_string()
            + "\n"
            + &(match lock_type {
                None => String::new(),
                Some(lock_type) => format!("lockType={}", lock_type),
            })
            + &(match id {
                None => String::new(),
                Some(id) => format!("id={}", id),
            })
            + &(match log_file {
                None => String::new(),
                Some(log_file) => format!("logFile={}", log_file),
            })
    }
}

impl std::fmt::Display for UnlockedException {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.log_file_exception.message)
    }
}

impl std::error::Error for UnlockedException {}

/// Java `getTestUnlockedException`.
pub fn get_test_unlocked_exception() -> UnlockedException {
    UnlockedException::new("Test exception")
}

/// Java `ReadingToken`.
#[derive(Debug)]
struct ReadingToken {
    /// Java field `readerType`.
    reader_type: ReaderType,
    /// Java field `open`, initialised to true.
    open: bool,
    /// Java field `key`.
    key: Option<String>,
    /// The `Reader` / `BigBufferReader` subclass state, when this token is one of them.
    subclass: ReadingTokenSubclass,
}

/// The state the two `ReadingToken` subclasses add.  Java models them as subclasses that
/// override `open()` and `close()`; the reader they hold is the only extra state.
#[derive(Debug)]
enum ReadingTokenSubclass {
    /// The base `ReadingToken` itself, which `newReadingToken` builds for
    /// `ReaderType.READING`.
    None,
    /// Java `Reader`.
    Reader {
        /// Java field `file`.
        file: std::path::PathBuf,
        /// Java field `required`.
        required: bool,
        /// Java fields `fileReader` and `bufferedReader`.
        buffered_reader: Option<std::io::BufReader<std::fs::File>>,
        /// Java field `fileNotFoundExceptionCount`.
        file_not_found_exception_count: i32,
    },
    /// Java `BigBufferReader`.
    BigBufferReader {
        /// Java field `file`.
        file: std::path::PathBuf,
        /// Java fields `fileReader` and `bufferedReader`.
        buffered_reader: Option<std::io::BufReader<std::fs::File>>,
    },
}

/// Java `Reader.DUMP_FILE_NOT_FOUND_EXCEPTION_AT`.
const DUMP_FILE_NOT_FOUND_EXCEPTION_AT: i32 = 5;
/// Java `BigBufferReader.SIZE`.
const BIG_BUFFER_READER_SIZE: usize = 4096;

impl ReadingToken {
    /// Java `ReadingToken(ReaderType)`.
    fn new(reader_type: ReaderType) -> ReadingToken {
        ReadingToken {
            reader_type,
            open: true,
            key: None,
            subclass: ReadingTokenSubclass::None,
        }
    }

    /// Java `Reader(File, boolean)`.
    fn new_reader(file: &std::path::Path, required: bool) -> ReadingToken {
        ReadingToken {
            reader_type: ReaderType::Reader,
            open: true,
            key: None,
            subclass: ReadingTokenSubclass::Reader {
                file: file.to_path_buf(),
                required,
                buffered_reader: None,
                file_not_found_exception_count: 0,
            },
        }
    }

    /// Java `BigBufferReader(File)`.
    fn new_big_buffer_reader(file: &std::path::Path) -> ReadingToken {
        ReadingToken {
            reader_type: ReaderType::BigBufferReader,
            open: true,
            key: None,
            subclass: ReadingTokenSubclass::BigBufferReader {
                file: file.to_path_buf(),
                buffered_reader: None,
            },
        }
    }

    /// Java `getReaderType`.
    fn get_reader_type(&self) -> ReaderType {
        self.reader_type
    }

    /// Java `ReadingToken.open()`, `Reader.open()` and `BigBufferReader.open()`.
    fn open(&mut self) -> Result<bool, std::io::Error> {
        match &mut self.subclass {
            ReadingTokenSubclass::None => {
                self.open = true;
                Ok(true)
            }
            ReadingTokenSubclass::Reader {
                file,
                required,
                buffered_reader,
                file_not_found_exception_count,
            } => {
                let file_name = utilities::java_io_file_get_name(&file.to_string_lossy());
                let mut exception: Option<std::io::Error> = None;
                if buffered_reader.is_none() {
                    // Having trouble opening the file even though it exists.
                    let max_tries = 3;
                    let mut i = 1;
                    while i <= max_tries {
                        if i > 1 {
                            if DEBUG.is_extra() {
                                eprintln!("Attempt # {} to open {}.", i, file.display());
                                eprintln!(
                                    "Does {} exist?  {}",
                                    utilities::java_io_file_get_absolute_path(
                                        &file.to_string_lossy()
                                    ),
                                    file.exists()
                                );
                            }
                            let parent_file =
                                utilities::java_io_file_get_parent(&file.to_string_lossy());
                            match parent_file {
                                Some(parent_file) => match std::fs::read_dir(&parent_file) {
                                    Ok(entries) => {
                                        let name = utilities::java_io_file_get_name(
                                            &file.to_string_lossy(),
                                        );
                                        for entry in entries.flatten() {
                                            if name
                                                == entry.file_name().to_string_lossy().to_string()
                                            {
                                                eprintln!(
                                                    "File found in {}:{}",
                                                    parent_file,
                                                    entry.file_name().to_string_lossy()
                                                );
                                            }
                                        }
                                    }
                                    Err(_) => eprintln!("Can't get directory list"),
                                },
                                None => eprintln!("Can't get parent directory"),
                            }
                            exception = None;
                        }
                        CLEAN_PRINT.print(Some(&format!("(1) Open reader for {}.", file_name)));
                        match std::fs::File::open(utilities::java_io_file_get_absolute_path(
                            &file.to_string_lossy(),
                        )) {
                            Ok(handle) => {
                                *buffered_reader = Some(std::io::BufReader::new(handle));
                                break;
                            }
                            Err(e) => {
                                *file_not_found_exception_count += 1;
                                let is_required = *required;
                                let count = *file_not_found_exception_count;
                                exception = Some(e);
                                if is_required
                                    && i >= max_tries
                                    && count > DUMP_FILE_NOT_FOUND_EXCEPTION_AT
                                {
                                    *file_not_found_exception_count = 0;
                                    eprintln!("{}", exception.as_ref().unwrap());
                                }
                            }
                        }
                        std::thread::sleep(std::time::Duration::from_millis(5));
                        i += 1;
                    }
                    if *required {
                        if let Some(exception) = exception {
                            return Err(exception);
                        }
                        exception = None;
                    }
                }
                if exception.is_some() {
                    return Ok(false);
                }
                // Java wraps the FileReader in a BufferedReader here; the two are one
                // object in this translation, so there is nothing further to open.
                CLEAN_PRINT.print(Some(&format!(
                    "(1) Open buffered reader for {}.",
                    file_name
                )));
                self.open = true;
                Ok(true)
            }
            ReadingTokenSubclass::BigBufferReader {
                file,
                buffered_reader,
            } => {
                let file_name = utilities::java_io_file_get_name(&file.to_string_lossy());
                if buffered_reader.is_none() {
                    CLEAN_PRINT.print(Some(&format!("(2) Open reader for {}.", file_name)));
                    let handle = std::fs::File::open(utilities::java_io_file_get_absolute_path(
                        &file.to_string_lossy(),
                    ))?;
                    CLEAN_PRINT.print(Some(&format!(
                        "(2) Open buffered reader for {}.",
                        file_name
                    )));
                    *buffered_reader = Some(std::io::BufReader::with_capacity(
                        BIG_BUFFER_READER_SIZE,
                        handle,
                    ));
                }
                self.open = true;
                Ok(true)
            }
        }
    }

    /// Java `ReadingToken.close()`, `Reader.close()` and `BigBufferReader.close()`.
    fn close(&mut self) {
        match &mut self.subclass {
            ReadingTokenSubclass::None => {}
            ReadingTokenSubclass::Reader {
                file,
                buffered_reader,
                ..
            } => {
                let file_name = utilities::java_io_file_get_name(&file.to_string_lossy());
                if buffered_reader.is_some() {
                    CLEAN_PRINT.print(Some(&format!("(1) Close reader for {}.", file_name)));
                    CLEAN_PRINT.print(Some(&format!(
                        "(1) Close buffered reader for {}.",
                        file_name
                    )));
                    *buffered_reader = None;
                }
            }
            ReadingTokenSubclass::BigBufferReader {
                file,
                buffered_reader,
            } => {
                let file_name = utilities::java_io_file_get_name(&file.to_string_lossy());
                if buffered_reader.is_some() {
                    CLEAN_PRINT.print(Some(&format!("(2) Close reader for {}.", file_name)));
                    CLEAN_PRINT.print(Some(&format!(
                        "(2) Close buffered reader for {}.",
                        file_name
                    )));
                    *buffered_reader = None;
                }
            }
        }
        self.open = false;
    }

    /// Java `isOpen`.
    fn is_open(&self) -> bool {
        self.open
    }

    /// Java `setKey`.
    fn set_key(&mut self, key: &str) {
        self.key = Some(key.to_string());
    }

    /// Java `getKey`.
    fn get_key(&self) -> Option<&str> {
        self.key.as_deref()
    }

    /// Java `Reader.readLine()` and `BigBufferReader.readLine()`:
    /// `java.io.BufferedReader.readLine()`, which returns null at end of file and strips
    /// a trailing "\n", "\r" or "\r\n".
    fn read_line(&mut self) -> Result<Option<String>, std::io::Error> {
        let reader = match &mut self.subclass {
            ReadingTokenSubclass::Reader {
                buffered_reader, ..
            } => buffered_reader,
            ReadingTokenSubclass::BigBufferReader {
                buffered_reader, ..
            } => buffered_reader,
            // Java's base ReadingToken has no readLine; calling it is a compile error
            // there.
            ReadingTokenSubclass::None => return Ok(None),
        };
        let reader = match reader {
            None => return Ok(None),
            Some(reader) => reader,
        };
        let mut line = Vec::new();
        let mut saw_any = false;
        loop {
            let mut byte = [0u8; 1];
            let read = reader.read(&mut byte)?;
            if read == 0 {
                break;
            }
            saw_any = true;
            if byte[0] == b'\n' {
                break;
            }
            if byte[0] == b'\r' {
                // Consume a following "\n" as part of the same terminator.
                let available = reader.fill_buf()?;
                if available.first() == Some(&b'\n') {
                    reader.consume(1);
                }
                break;
            }
            line.push(byte[0]);
        }
        if !saw_any {
            return Ok(None);
        }
        Ok(Some(String::from_utf8_lossy(&line).to_string()))
    }

    /// Java `BigBufferReader.searchForLastLine`.  Searches for a string in the last
    /// characters in a file.  Finds the last characters of the file equal to 2 times the
    /// length of `lastLine` and searches them for `lastLine`.  Returns true if `lastLine`
    /// is found in this span of characters.
    fn search_for_last_line(&mut self, last_line: &str) -> Result<bool, std::io::Error> {
        let reader = match &mut self.subclass {
            ReadingTokenSubclass::BigBufferReader {
                buffered_reader, ..
            } => buffered_reader,
            _ => return Ok(false),
        };
        let reader = match reader {
            None => return Ok(false),
            Some(reader) => reader,
        };
        // Increase the size to handle a final end of line (linux or windows).
        let length_to_scan = last_line.len() * 2;
        if length_to_scan > BIG_BUFFER_READER_SIZE {
            eprintln!(
                "java.lang.IllegalArgumentException: String too large to scan for:{}",
                last_line
            );
            return Ok(false);
        }
        let mut n_read_temp: i64;
        let mut n_read: i64 = -1;
        let mut n_read_prev: i64 = -1;
        // Not sure if read reallocates the char array.  Avoid assigning one char array to
        // the other by using alternating reads.
        let mut first = true;
        let mut first_char_array = vec![0u8; BIG_BUFFER_READER_SIZE];
        let mut second_char_array = vec![0u8; BIG_BUFFER_READER_SIZE];
        loop {
            let target = if first {
                &mut first_char_array
            } else {
                &mut second_char_array
            };
            // `bufferedReader.read(char[])` fills as much of the array as the underlying
            // reader can supply.
            let mut filled = 0usize;
            while filled < BIG_BUFFER_READER_SIZE {
                let read = reader.read(&mut target[filled..])?;
                if read == 0 {
                    break;
                }
                filled += read;
            }
            n_read_temp = if filled == 0 { -1 } else { filled as i64 };
            if n_read_temp == -1 {
                break;
            }
            // Save the previous number of characters read
            n_read_prev = n_read;
            // Save the number of characters read this time
            n_read = n_read_temp;
            first = !first;
        }
        if n_read == -1 {
            // Nothing was read
            return Ok(false);
        }
        // Assign the apropriate char array
        let (char_array, char_array_prev) = if !first {
            (&first_char_array, &second_char_array)
        } else {
            (&second_char_array, &first_char_array)
        };
        // Add the last characters equal to usefulLength to buffer.
        let mut buffer: Vec<u8> = Vec::new();
        let mut useful_length: i64;
        if n_read < length_to_scan as i64 && n_read_prev > 0 {
            // The last characters have been split in half
            useful_length = n_read_prev.min(length_to_scan as i64 - n_read);
            buffer.extend_from_slice(
                &char_array_prev[(n_read_prev - useful_length) as usize..n_read_prev as usize],
            );
        }
        useful_length = n_read.min(length_to_scan as i64);
        buffer.extend_from_slice(&char_array[(n_read - useful_length) as usize..n_read as usize]);
        Ok(String::from_utf8_lossy(&buffer).contains(last_line))
    }
}

/// Java `ReadingTokenList`.
#[derive(Debug)]
struct ReadingTokenList {
    /// Java field `hashMap`.
    hash_map: Mutex<HashMap<String, usize>>,
    /// Java field `arrayList`.
    array_list: Mutex<Vec<ReadingToken>>,
    /// Java field `debug`.
    debug: Mutex<bool>,
}

impl ReadingTokenList {
    /// Java `ReadingTokenList()`.
    fn new() -> ReadingTokenList {
        ReadingTokenList {
            hash_map: Mutex::new(HashMap::new()),
            array_list: Mutex::new(Vec::new()),
            debug: Mutex::new(false),
        }
    }

    /// Java `makeKey(ReadId)`.  The `hashMap` key is the lock number's `toString`; the
    /// map is keyed by `String` here and the index into `arrayList` is the value, because
    /// Rust cannot store two aliases to the same `ReadingToken`.  The enclosing
    /// `LogFile.this` that `new Reporter(...)` needs is passed in.
    fn make_key(&self, log_file: &LogFile, id: Option<&Arc<Id>>) -> Result<String, LogFileError> {
        let id = match id {
            None => {
                Reporter::new_log_file_exception(
                    log_file,
                    LogFileError::Unlocked(UnlockedException::new("id is null")),
                )
                .print()
                .throw_log_file_exception()?;
                return Ok(String::new());
            }
            Some(id) => id,
        };
        if !id.is_lock_number_set() {
            Reporter::new_log_file_exception(
                log_file,
                LogFileError::Unlocked(UnlockedException::new("id is null")),
            )
            .print()
            .throw_log_file_exception()?;
        }
        Ok(id.get_lock_number().unwrap().to_string())
    }

    /// Java `setDebug`.
    fn set_debug(&self, input: bool) {
        *self.debug.lock().unwrap() = input;
    }

    /// Java `getReadingToken`.
    fn get_reading_token(&self, key: &str) -> Option<usize> {
        self.hash_map.lock().unwrap().get(key).copied()
    }

    /// Java `getReader`.
    fn get_reader(&self, key: &str) -> Option<usize> {
        let index = *self.hash_map.lock().unwrap().get(key)?;
        if self.array_list.lock().unwrap()[index].get_reader_type() == ReaderType::Reader {
            return Some(index);
        }
        None
    }

    /// Java `getBigBufferReader`.
    fn get_big_buffer_reader(&self, key: &str) -> Option<usize> {
        let index = *self.hash_map.lock().unwrap().get(key)?;
        if self.array_list.lock().unwrap()[index].get_reader_type() == ReaderType::BigBufferReader {
            return Some(index);
        }
        None
    }

    /// Java `openReadingToken(String, File, ReaderType)`.
    fn open_reading_token(
        &self,
        current_key: &str,
        file: &std::path::Path,
        reader_type: ReaderType,
    ) -> Result<bool, std::io::Error> {
        self.open_reading_token_required(current_key, file, reader_type, true)
    }

    /// Java `openReadingToken(String, File, ReaderType, boolean)`.  `reader_required`,
    /// when false, allows a silent fail when opening a `Reader`.
    fn open_reading_token_required(
        &self,
        current_key: &str,
        file: &std::path::Path,
        reader_type: ReaderType,
        reader_required: bool,
    ) -> Result<bool, std::io::Error> {
        {
            let mut array_list = self.array_list.lock().unwrap();
            for i in 0..array_list.len() {
                if array_list[i].get_reader_type() == reader_type && !array_list[i].is_open() {
                    // open the reader to get exclusive access to it
                    if let Err(e) = array_list[i].open() {
                        eprintln!("{}\ncurrentKey={}", e, current_key);
                        return Err(e);
                    }
                    // Found a closed reader, so reuse it.  Get the old key from the reader
                    // and rekey the reader in the hash map with the current key.
                    let old_key = array_list[i].get_key().map(|key| key.to_string());
                    let mut hash_map = self.hash_map.lock().unwrap();
                    if let Some(old_key) = old_key {
                        hash_map.remove(&old_key);
                    }
                    array_list[i].set_key(current_key);
                    hash_map.insert(current_key.to_string(), i);
                }
            }
        }
        // Can't find a closed reader, so create a new one.  (The loop above does not
        // return; the source always falls through to here.)
        let mut reading_token =
            ReadingTokenList::new_reading_token(file, reader_type, reader_required);
        // open the reader to get exclusive access to it
        if !reading_token.open()? {
            return Ok(false);
        }
        // store the current key in the reader and store it in the array list and hash map
        reading_token.set_key(current_key);
        let mut array_list = self.array_list.lock().unwrap();
        array_list.push(reading_token);
        let index = array_list.len() - 1;
        self.hash_map
            .lock()
            .unwrap()
            .insert(current_key.to_string(), index);
        Ok(true)
    }

    /// Java `newReadingToken`.  `reader_required`, when false, allows a silent fail when
    /// opening a `Reader`.
    fn new_reading_token(
        file: &std::path::Path,
        reader_type: ReaderType,
        reader_required: bool,
    ) -> ReadingToken {
        if reader_type == ReaderType::Reader {
            return ReadingToken::new_reader(file, reader_required);
        }
        if reader_type == ReaderType::BigBufferReader {
            return ReadingToken::new_big_buffer_reader(file);
        }
        ReadingToken::new(reader_type)
    }
}

/// Java's checked-exception set, as one Rust error type.  A `LogFile` member declares up
/// to four of `LogFileException` (with its `FileException` and `UnlockedException`
/// subclasses), the inner-class `LockException`, `java.io.IOException` and
/// `java.io.FileNotFoundException` in one `throws` clause; a Rust function returns one
/// `Result`, so the thrown types are the variants here.  `FileNotFoundException` is an
/// `IOException` subclass and arrives as `Io` with `ErrorKind::NotFound`, which is what
/// the source's `catch (final FileNotFoundException e)` arms test for.
#[derive(Debug)]
pub enum LogFileError {
    /// `LogFile.LogFileException`, including the anonymous subclass `copyFile` throws.
    LogFile(LogFileException),
    /// `LogFile.FileException`.
    File(FileException),
    /// `LogFile.UnlockedException`.
    Unlocked(UnlockedException),
    /// `LogFile.LockException`.
    Lock(LockException),
    /// `java.io.IOException` and `java.io.FileNotFoundException`.
    Io(std::io::Error),
}

/// `java.lang.Exception` is a reference in Java, and `Reporter.throwLogFileException`
/// rethrows the object it holds; the Rust `Reporter` keeps its exception and hands the
/// caller a copy.  `std::io::Error` is not `Clone`, so its copy carries the kind and the
/// message.
impl Clone for LogFileError {
    fn clone(&self) -> LogFileError {
        match self {
            LogFileError::LogFile(e) => LogFileError::LogFile(e.clone()),
            LogFileError::File(e) => LogFileError::File(e.clone()),
            LogFileError::Unlocked(e) => LogFileError::Unlocked(e.clone()),
            LogFileError::Lock(e) => LogFileError::Lock(e.clone()),
            LogFileError::Io(e) => LogFileError::Io(std::io::Error::new(e.kind(), e.to_string())),
        }
    }
}

impl From<LogFileException> for LogFileError {
    fn from(e: LogFileException) -> LogFileError {
        LogFileError::LogFile(e)
    }
}

impl From<FileException> for LogFileError {
    fn from(e: FileException) -> LogFileError {
        LogFileError::File(e)
    }
}

impl From<UnlockedException> for LogFileError {
    fn from(e: UnlockedException) -> LogFileError {
        LogFileError::Unlocked(e)
    }
}

impl From<LockException> for LogFileError {
    fn from(e: LockException) -> LogFileError {
        LogFileError::Lock(e)
    }
}

impl From<std::io::Error> for LogFileError {
    fn from(e: std::io::Error) -> LogFileError {
        LogFileError::Io(e)
    }
}

impl std::fmt::Display for LogFileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LogFileError::LogFile(e) => std::fmt::Display::fmt(e, f),
            LogFileError::File(e) => std::fmt::Display::fmt(e, f),
            LogFileError::Unlocked(e) => std::fmt::Display::fmt(e, f),
            LogFileError::Lock(e) => std::fmt::Display::fmt(e, f),
            LogFileError::Io(e) => std::fmt::Display::fmt(e, f),
        }
    }
}

impl std::error::Error for LogFileError {}

impl LogFileError {
    /// `java.lang.Throwable.getMessage()` on whichever exception this is.
    pub fn get_message(&self) -> String {
        match self {
            LogFileError::LogFile(e) => e.get_message().to_string(),
            LogFileError::File(e) => e.get_message().to_string(),
            LogFileError::Unlocked(e) => e.get_message().to_string(),
            LogFileError::Lock(e) => e.get_message().to_string(),
            LogFileError::Io(e) => e.to_string(),
        }
    }
}

/// The `java.lang.Throwable` cause a `LockException` carries.  `popupMessage`
/// (`EmergencyMonitor.java:77`) tests `cause instanceof OverlappingFileLockException`, so
/// the cause's Java class is part of the behaviour and not just its message.
#[derive(Clone, Debug)]
pub enum LockExceptionCause {
    /// `java.nio.channels.OverlappingFileLockException`: the file is already locked
    /// within this process.
    OverlappingFileLock(String),
    /// `java.io.IOException`.
    Io(String),
}

impl std::fmt::Display for LockExceptionCause {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LockExceptionCause::OverlappingFileLock(message) => f.write_str(message),
            LockExceptionCause::Io(message) => f.write_str(message),
        }
    }
}

/// Java `LockException`.  Only used for blocking locks that can't be overridden.  Causes
/// a retry if any are left.  Private exception: when retries are exhausted, its stack is
/// printed and an `UnlockedException` is thrown.
pub struct LockException {
    /// The inner class's enclosing `LogFile.this`.
    log_file: Arc<LogFile>,
    /// `java.lang.Throwable.getMessage()`, which is
    /// `getExceptionMessage(message, newId, blockingId, extendedInfo)`.
    exception_message: String,
    /// `java.lang.Throwable.getCause()`.
    cause: Option<LockExceptionCause>,
    /// Java field `newId`.
    new_id: Option<Arc<Id>>,
    /// Java field `blockingId`.
    blocking_id: Option<Arc<Id>>,
    /// Java field `toFileName`.
    to_file_name: Option<String>,
    /// Java field `extendedInfo`.
    extended_info: bool,
    /// Java field `message`.
    message: String,
    /// Java field `action`, which `Lockable.printDiagnostics` assigns to.
    action: Mutex<Option<StandardBarString>>,
}

impl Clone for LockException {
    fn clone(&self) -> LockException {
        LockException {
            log_file: self.log_file.clone(),
            exception_message: self.exception_message.clone(),
            cause: self.cause.clone(),
            new_id: self.new_id.clone(),
            blocking_id: self.blocking_id.clone(),
            to_file_name: self.to_file_name.clone(),
            extended_info: self.extended_info,
            message: self.message.clone(),
            action: Mutex::new(*self.action.lock().unwrap()),
        }
    }
}

impl std::fmt::Debug for LockException {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "LockException({})", self.exception_message)
    }
}

impl std::fmt::Display for LockException {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.exception_message)
    }
}

impl std::error::Error for LockException {}

impl LockException {
    /// Java `LockException(String, Id, Id, StandardBarString, String, boolean,
    /// Throwable)`.
    pub fn new(
        log_file: &Arc<LogFile>,
        message: &str,
        new_id: Option<&Arc<Id>>,
        blocking_id: Option<&Arc<Id>>,
        action: Option<StandardBarString>,
        to_file_name: Option<&str>,
        extended_info: bool,
        cause: Option<LockExceptionCause>,
    ) -> LockException {
        LockException {
            exception_message: log_file.get_exception_message(
                message,
                new_id.map(|new_id| &**new_id),
                blocking_id.map(|blocking_id| &**blocking_id),
                extended_info,
            ),
            log_file: log_file.clone(),
            cause,
            new_id: new_id.cloned(),
            blocking_id: blocking_id.cloned(),
            action: Mutex::new(action),
            to_file_name: to_file_name.map(|to_file_name| to_file_name.to_string()),
            extended_info,
            message: message.to_string(),
        }
    }

    /// `java.lang.Throwable.getMessage()`.
    pub fn get_message(&self) -> &str {
        &self.exception_message
    }

    /// `java.lang.Throwable.getCause()`.
    pub fn get_cause(&self) -> Option<&LockExceptionCause> {
        self.cause.as_ref()
    }

    /// Java `printStackTrace`.
    pub fn print_stack_trace(&self) {
        // `super.printStackTrace()`: the JVM's own dump of the throwable and its frames.
        eprintln!("{}", self.exception_message);
        if let Some(cause) = &self.cause {
            eprintln!("Caused by: {}", cause);
        }
        if self.extended_info {
            self.new_id
                .as_ref()
                .unwrap()
                .get_handle()
                .unwrap()
                .print_owner_stack_trace_title(Some("New Id"));
            self.blocking_id
                .as_ref()
                .unwrap()
                .get_handle()
                .unwrap()
                .print_owner_stack_trace_title(Some("Blocking Id"));
        }
    }

    /// Java `equalsFile(LogFile.Handle)`.
    pub fn equals_file_handle(&self, handle: Option<&Arc<Handle>>) -> bool {
        self.log_file.equals_handle(handle)
    }

    /// Java `equalsFile(File)`.
    pub fn equals_file(&self, file: Option<&std::path::Path>) -> bool {
        self.log_file.equals_file(file)
    }

    /// Java `getAction`.
    pub fn get_action(&self) -> Option<StandardBarString> {
        *self.action.lock().unwrap()
    }

    /// Java `getFileName`, which returns the enclosing instance's `fileName` field.
    pub fn get_file_name(&self) -> String {
        self.log_file.file_name.clone()
    }

    /// Java `getToFileName`.
    pub fn get_to_file_name(&self) -> Option<String> {
        self.to_file_name.clone()
    }

    /// Java `getBlockingId`.
    fn get_blocking_id(&self) -> Option<Arc<Id>> {
        self.blocking_id.clone()
    }
}

/// Java `Reporter`.
struct Reporter<'a> {
    /// The inner class's enclosing `LogFile.this`.
    log_file: &'a LogFile,
    /// Java field `message`.
    message: Option<String>,
    /// Java field `logFileException`.  `LogFileException` is abstract, so the value is
    /// always one of its subclasses; the Rust type for "one of them" is the error union.
    log_file_exception: Option<LogFileError>,
    /// Java field `lockException`.
    lock_exception: Option<LockException>,
    /// Java field `blockingId`.
    blocking_id: Option<Arc<Id>>,
    /// Java field `verbose`.
    verbose: bool,
}

impl<'a> Reporter<'a> {
    /// Java `Reporter(String, LogFileException, LockException, Id, boolean)`.
    fn new_full(
        log_file: &'a LogFile,
        message: Option<&str>,
        log_file_exception: Option<LogFileError>,
        lock_exception: Option<LockException>,
        blocking_id: Option<&Arc<Id>>,
        verbose: bool,
    ) -> Reporter<'a> {
        let stored_message: Option<String>;
        if let Some(message) = message {
            stored_message = Some(message.to_string());
        } else if let Some(log_file_exception) = &log_file_exception {
            stored_message = Some(log_file_exception.get_message().to_string());
        } else if let Some(lock_exception) = &lock_exception {
            stored_message = Some(lock_exception.get_message().to_string());
        } else {
            stored_message = None;
        }
        Reporter {
            log_file,
            message: stored_message,
            log_file_exception,
            lock_exception,
            blocking_id: blocking_id.cloned(),
            verbose,
        }
    }

    /// Java `Reporter(String)`.
    fn new_message(log_file: &'a LogFile, message: &str) -> Reporter<'a> {
        Reporter::new_full(log_file, Some(message), None, None, None, false)
    }

    /// Java `Reporter(String, Id)`.
    fn new_message_blocking_id(
        log_file: &'a LogFile,
        message: &str,
        blocking_id: Option<&Arc<Id>>,
    ) -> Reporter<'a> {
        Reporter::new_full(log_file, Some(message), None, None, blocking_id, false)
    }

    /// Java `Reporter(String, Id, boolean)`.
    fn new_message_blocking_id_verbose(
        log_file: &'a LogFile,
        message: &str,
        blocking_id: Option<&Arc<Id>>,
        verbose: bool,
    ) -> Reporter<'a> {
        Reporter::new_full(log_file, Some(message), None, None, blocking_id, verbose)
    }

    /// Java `Reporter(LogFileException)`.
    fn new_log_file_exception(
        log_file: &'a LogFile,
        log_file_exception: LogFileError,
    ) -> Reporter<'a> {
        Reporter::new_full(log_file, None, Some(log_file_exception), None, None, false)
    }

    /// Java `Reporter(LogFileException, Id)`.
    fn new_log_file_exception_blocking_id(
        log_file: &'a LogFile,
        log_file_exception: LogFileError,
        blocking_id: Option<&Arc<Id>>,
    ) -> Reporter<'a> {
        Reporter::new_full(
            log_file,
            None,
            Some(log_file_exception),
            None,
            blocking_id,
            false,
        )
    }

    /// Java `Reporter(LogFileException, boolean)`.
    fn new_log_file_exception_verbose(
        log_file: &'a LogFile,
        log_file_exception: LogFileError,
        verbose: bool,
    ) -> Reporter<'a> {
        Reporter::new_full(
            log_file,
            None,
            Some(log_file_exception),
            None,
            None,
            verbose,
        )
    }

    /// Java `Reporter(LogFileException, Id, boolean)`.
    fn new_log_file_exception_id_verbose(
        log_file: &'a LogFile,
        log_file_exception: LogFileError,
        id: Option<&Arc<Id>>,
        verbose: bool,
    ) -> Reporter<'a> {
        Reporter::new_full(log_file, None, Some(log_file_exception), None, id, verbose)
    }

    /// Java `Reporter(LockException, boolean)`.
    fn new_lock_exception_verbose(
        log_file: &'a LogFile,
        lock_exception: LockException,
        verbose: bool,
    ) -> Reporter<'a> {
        Reporter::new_full(log_file, None, None, Some(lock_exception), None, verbose)
    }

    /// Java `Reporter(LockException, Id)`.
    fn new_lock_exception_blocking_id(
        log_file: &'a LogFile,
        lock_exception: LockException,
        blocking_id: Option<&Arc<Id>>,
    ) -> Reporter<'a> {
        Reporter::new_full(
            log_file,
            None,
            None,
            Some(lock_exception),
            blocking_id,
            false,
        )
    }

    /// Java `Reporter(LockException, Id, boolean)`.
    fn new_lock_exception_blocking_id_verbose(
        log_file: &'a LogFile,
        lock_exception: LockException,
        blocking_id: Option<&Arc<Id>>,
        verbose: bool,
    ) -> Reporter<'a> {
        Reporter::new_full(
            log_file,
            None,
            None,
            Some(lock_exception),
            blocking_id,
            verbose,
        )
    }

    /// Java `print()`.
    fn print(&self) -> &Reporter<'a> {
        self.print_on_debug(false)
    }

    /// Java `debugPrint`.
    fn debug_print(&self) -> &Reporter<'a> {
        self.print_on_debug(true)
    }

    /// Java `print(boolean)`.
    fn print_on_debug(&self, print_on_debug: bool) -> &Reporter<'a> {
        if print_on_debug && !*self.log_file.debug.lock().unwrap() && !DEBUG.is_on() {
            return self;
        }
        let error = match &self.message {
            None => false,
            Some(message) => message.trim().to_lowercase().starts_with("error"),
        };
        let verbose =
            self.verbose || error || *self.log_file.debug.lock().unwrap() || DEBUG.is_on();
        let mut builder = String::new();
        builder.push_str(&format!(
            "\n{}\n",
            match &self.message {
                None => "null".to_string(),
                Some(message) => message.clone(),
            }
        ));
        if let Some(blocking_id) = &self.blocking_id {
            builder.push_str(&format!(
                "blockingId:{}",
                blocking_id.to_string_verbose(!self.log_file.lock.contains(Some(blocking_id)))
            ));
        } else {
            builder.push_str(&format!(
                "lock.curLockNumber:{}",
                *self.log_file.lock.cur_lock_number.lock().unwrap()
            ));
        }
        builder.push_str(&format!(
            "\nlock:{}",
            self.log_file
                .lock
                .to_string_verbose(verbose || self.blocking_id.is_none())
        ));
        eprintln!("{}", builder);
        // Print stack
        if verbose
            && !self.print_exception(self.log_file_exception.as_ref())
            && !self.print_exception(
                self.lock_exception
                    .as_ref()
                    .map(|lock_exception| LogFileError::Lock(lock_exception.clone()))
                    .as_ref(),
            )
        {
            // `Thread.dumpStack()`; see etomo/util/stack_trace.rs for why this process
            // contributes no frames.
            StackTrace::new_with_thread(None, None).print(Some("Stack trace"), false);
        }
        self
    }

    /// Java `printException`.
    fn print_exception(&self, exception: Option<&LogFileError>) -> bool {
        let exception = match exception {
            None => return false,
            Some(exception) => exception,
        };
        // `exception.getCause()`, whose `printStackTrace` runs before the exception's own.
        let cause = match exception {
            LogFileError::LogFile(e) => e.get_cause().map(|cause| cause.to_string()),
            LogFileError::File(e) => e.get_cause().map(|cause| cause.to_string()),
            LogFileError::Unlocked(e) => e.get_cause().map(|cause| cause.to_string()),
            LogFileError::Lock(e) => e.get_cause().map(|cause| cause.to_string()),
            LogFileError::Io(_) => None,
        };
        if let Some(cause) = cause {
            eprintln!("{}", cause);
        }
        eprintln!("{}", exception);
        true
    }

    /// Java `throwLogFileException`.
    fn throw_log_file_exception(&self) -> Result<(), LogFileError> {
        if let Some(log_file_exception) = &self.log_file_exception {
            return Err(log_file_exception.clone());
        }
        Ok(())
    }

    /// Java `throwLockException`.
    fn throw_lock_exception(&self) -> Result<(), LogFileError> {
        if let Some(lock_exception) = &self.lock_exception {
            return Err(LogFileError::Lock(lock_exception.clone()));
        }
        Ok(())
    }
}

/// Java `BlockingIdIterator`.  Iterates through `Id`s that can block a specific lock
/// type.
struct BlockingIdIterator {
    /// Java field `readIdIterator`.  `ConcurrentHashMap.values().iterator()` is weakly
    /// consistent; the closest Rust shape is a snapshot taken under the field's lock.
    read_id_iterator: Option<std::vec::IntoIter<Arc<Id>>>,
    /// Java field `fileId`.
    file_id: Option<Arc<Id>>,
    /// Java field `writeId`.
    write_id: Option<Arc<Id>>,
}

impl BlockingIdIterator {
    /// Java `BlockingIdIterator(Lock, LockType)`.
    fn new(lock: Option<&Lock>, lock_type: Option<LockType>) -> BlockingIdIterator {
        let mut file_id = None;
        let mut write_id = None;
        let read_id_iterator;
        // Only add Ids that block.
        match (lock, lock_type) {
            (Some(lock), Some(lock_type)) => {
                // fileId always blocks.
                file_id = lock.file_id.lock().unwrap().clone();
                if lock_type == LockType::File || lock_type == LockType::Write {
                    write_id = lock.write_id.lock().unwrap().clone();
                }
                if lock_type == LockType::File && !lock.read_id_hash_map.lock().unwrap().is_empty()
                {
                    read_id_iterator = Some(
                        lock.read_id_hash_map
                            .lock()
                            .unwrap()
                            .values()
                            .cloned()
                            .collect::<Vec<Arc<Id>>>()
                            .into_iter(),
                    );
                } else {
                    read_id_iterator = None;
                }
            }
            _ => {
                read_id_iterator = None;
            }
        }
        BlockingIdIterator {
            read_id_iterator,
            file_id,
            write_id,
        }
    }

    /// Java `hasNext`.  All `Id`s that have been saved are blocking `Id`s.  The order of
    /// `Id`s is always File, Write, Read.
    fn has_next(&self) -> bool {
        self.file_id.is_some()
            || self.write_id.is_some()
            || match &self.read_id_iterator {
                None => false,
                Some(read_id_iterator) => !read_id_iterator.as_slice().is_empty(),
            }
    }

    /// Java `next`.
    fn next(&mut self) -> Option<Arc<Id>> {
        let lock_id;
        if self.file_id.is_some() {
            lock_id = self.file_id.clone();
            self.file_id = None;
            return lock_id;
        }
        if self.write_id.is_some() {
            lock_id = self.write_id.clone();
            self.write_id = None;
            return lock_id;
        }
        if let Some(read_id_iterator) = &mut self.read_id_iterator {
            return read_id_iterator.next();
        }
        None
    }
}

/// `java.nio.file.StandardOpenOption`, the two members `waitForFileSystemLocks` uses.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum StandardOpenOption {
    Read,
    Write,
}

/// The JVM's per-process table of held `java.nio.channels.FileLock`s, keyed by the
/// device and inode of the locked file.  `FileChannel.tryLock` throws
/// `OverlappingFileLockException` when *this* virtual machine already holds an
/// overlapping lock, and returns null - not an error - when another *process* holds one;
/// the operating system's own lock cannot carry that distinction, so the same-process
/// half is this table.
static FILE_LOCK_TABLE: LazyLock<Mutex<Vec<(u64, u64)>>> = LazyLock::new(|| Mutex::new(Vec::new()));

/// Java `Lock`.
#[derive(Debug)]
struct Lock {
    /// Java field `readIdHashMap`.  The key is the lock number.
    read_id_hash_map: Mutex<HashMap<i32, Arc<Id>>>,
    /// Java field `logFile`.  The enclosing instance is not yet reachable through an
    /// `Arc` when the constructor runs, so `LogFile::new` builds it with
    /// `Arc::new_cyclic` and the back-reference is weak.
    log_file: std::sync::Weak<LogFile>,
    /// Java field `warningDisplayed`.
    warning_displayed: Mutex<bool>,
    /// Java field `curLockNumber`.  Java's `synchronized` methods are reentrant and
    /// Rust's `Mutex` is not, so each mutable field carries its own lock rather than the
    /// instance carrying one.
    cur_lock_number: Mutex<i32>,
    /// Java field `fileId`.
    file_id: Mutex<Option<Arc<Id>>>,
    /// Java field `writeId`.
    write_id: Mutex<Option<Arc<Id>>>,
}

/// Java `Lock.LOCK_NUMBER_MAX`.
const LOCK_NUMBER_MAX: i32 = 1000;

impl Lock {
    /// Java `Lock(LogFile)`.
    fn new(log_file: std::sync::Weak<LogFile>) -> Lock {
        Lock {
            read_id_hash_map: Mutex::new(HashMap::new()),
            log_file,
            warning_displayed: Mutex::new(false),
            cur_lock_number: Mutex::new(NO_ID),
            file_id: Mutex::new(None),
            write_id: Mutex::new(None),
        }
    }

    /// Java `toString(boolean)`.
    fn to_string_verbose(&self, verbose: bool) -> String {
        let mut builder = String::new();
        builder.push('[');
        let mut appended = false;
        if let Some(file_id) = &*self.file_id.lock().unwrap() {
            builder.push_str(&format!("fileId:{}", file_id.to_string_verbose(verbose)));
            appended = true;
        }
        if let Some(write_id) = &*self.write_id.lock().unwrap() {
            if appended {
                builder.push(',');
            } else {
                appended = true;
            }
            builder.push_str(&format!("writeId:{}", write_id.to_string_verbose(verbose)));
        }
        let read_id_hash_map = self.read_id_hash_map.lock().unwrap();
        if !read_id_hash_map.is_empty() {
            if appended {
                builder.push_str(",\n");
            }
            builder.push_str("readIds:[");
            appended = false;
            let mut iterator = read_id_hash_map.values();
            while let Some(read_id) = iterator.next() {
                if appended {
                    builder.push(',');
                } else {
                    appended = true;
                }
                builder.push_str(&read_id.to_string_verbose(verbose));
            }
            builder.push(']');
            if read_id_hash_map.len() > 3 {
                builder.push_str(&format!(
                    ",\ncurLockNumber:{}",
                    *self.cur_lock_number.lock().unwrap()
                ));
            }
        }
        builder.push(']');
        builder
    }

    /// Java `setLockNumber`.  Sets a unique lockNumber in the `id` parameter.
    fn set_lock_number(&self, id: &Arc<Id>) -> Result<(), LogFileError> {
        let log_file = self.log_file.upgrade().unwrap();
        let mut overflow_count = 0;
        let mut done = false;
        while !done {
            std::thread::sleep(std::time::Duration::from_millis(1));
            // Increment lock number. Avoid overflow.
            {
                let mut cur_lock_number = self.cur_lock_number.lock().unwrap();
                if *cur_lock_number < LOCK_NUMBER_MAX - 1 {
                    *cur_lock_number += 1;
                } else {
                    *cur_lock_number = 0;
                    overflow_count += 1;
                    if overflow_count >= 2 {
                        // This means that ids containing every positive number available
                        // in a long are stored in this lock instance. Which most likely
                        // means that there is an infinite loop, and the computer this is
                        // running on is in swap.
                        drop(cur_lock_number);
                        return Reporter::new_log_file_exception_verbose(
                            &log_file,
                            LogFileError::File(FileException::new(
                                &("Error: No lockNumber available.  LockNumber overflow.  \
                                   Infinite loop "
                                    .to_string()
                                    + "suspected.  Shut down software."),
                            )),
                            true,
                        )
                        .print()
                        .throw_log_file_exception();
                    }
                }
            }
            let cur_lock_number = *self.cur_lock_number.lock().unwrap();
            // Make sure this lock number isn't in use.
            if let Some(file_id) = &*self.file_id.lock().unwrap() {
                if file_id.equals_integer(Some(cur_lock_number)) {
                    continue;
                }
            }
            if let Some(write_id) = &*self.write_id.lock().unwrap() {
                if write_id.equals_integer(Some(cur_lock_number)) {
                    continue;
                }
            }
            if self
                .read_id_hash_map
                .lock()
                .unwrap()
                .contains_key(&cur_lock_number)
            {
                continue;
            }
            id.set_lock_number(Some(cur_lock_number));
            if let Some(lock_number_to_track) = *log_file.lock_number_to_track.lock().unwrap() {
                if lock_number_to_track == cur_lock_number {
                    Reporter::new_message_blocking_id_verbose(
                        &log_file,
                        "lock number set",
                        Some(id),
                        true,
                    )
                    .debug_print();
                }
            }
            done = true;
        }
        Ok(())
    }

    /// Java `lock(Id, StandardBarString, String, SleepTimer)`.  Adds an id to the lock
    /// instance.  An id that's identical to an id in this lock will cause a lock
    /// exception to be thrown.  If the lock type is blocked by existing lock ids which
    /// have a different handle, this function will attempt to override them.  If the id
    /// can be added to the lock, it will be given a unique lock number.
    fn lock(
        &self,
        id: Option<&Arc<Id>>,
        action: Option<StandardBarString>,
        to_file_name: Option<&str>,
        timer: Option<&mut SleepTimer>,
    ) -> Result<(), LogFileError> {
        let log_file = self.log_file.upgrade().unwrap();
        let id = match id {
            None => {
                Reporter::new_log_file_exception(
                    &log_file,
                    LogFileError::Unlocked(UnlockedException::new("Warning: null id parameter")),
                )
                .print()
                .throw_log_file_exception()?;
                return Ok(());
            }
            Some(id) => id,
        };
        if !id.is_valid() {
            Reporter::new_log_file_exception_blocking_id(
                &log_file,
                LogFileError::Unlocked(UnlockedException::new("Warning: invalid id")),
                Some(id),
            )
            .print()
            .throw_log_file_exception()?;
        }
        // Don't use an id that's already stored in the lock instance.
        if self.is_locked_id(Some(id)) {
            Reporter::new_message_blocking_id_verbose(
                &log_file,
                "Warning: already locked",
                Some(id),
                true,
            )
            .print();
            return Ok(());
        }
        // Check for other locks that prevent this lock. Attempt to fix the problem.
        let mut locked = false;
        // `synchronized (LogFile.this)`; see the note on `Lock.curLockNumber` for why the
        // enclosing instance carries no monitor here.
        if self.is_lockable(Some(id)) || self.check_blocking_ids(Some(id), action, to_file_name)? {
            self.set_lock_number(id)?;
            if id.equals_lock_type(LockType::File) {
                *self.file_id.lock().unwrap() = Some(id.clone());
                if etomo_director::ARGUMENTS.lock().unwrap().is_debug() {
                    eprintln!(
                        "(1) fileId set to {}",
                        self.file_id.lock().unwrap().as_ref().unwrap()
                    );
                }
            } else if id.equals_lock_type(LockType::Write) {
                *self.write_id.lock().unwrap() = Some(id.clone());
            } else if id.equals_lock_type(LockType::Read) {
                match id.get_lock_number() {
                    Some(lock_number) => {
                        self.read_id_hash_map
                            .lock()
                            .unwrap()
                            .insert(lock_number, id.clone());
                    }
                    None => {
                        // `catch (final NullPointerException e)`: a null lockNumber is the
                        // `Map.put` key that throws.
                        Reporter::new_log_file_exception_blocking_id(
                            &log_file,
                            LogFileError::Unlocked(UnlockedException::new(
                                "Error: Lock failed because of null lockNumber",
                            )),
                            Some(id),
                        )
                        .print()
                        .throw_log_file_exception()?;
                    }
                }
            } else {
                Reporter::new_log_file_exception_id_verbose(
                    &log_file,
                    LogFileError::Unlocked(UnlockedException::new("Error: bad lockType")),
                    Some(id),
                    true,
                )
                .print()
                .throw_log_file_exception()?;
            }
            locked = true;
        }
        if locked {
            if let Err(e) = self.wait_for_file_system_locks(Some(id), action, to_file_name, timer) {
                self.unlock(Some(id));
                return Err(LogFileError::Lock(e));
            }
        }
        Ok(())
    }

    /// Java `waitForFileSystemLocks`.  Waits for the file system to give up any file
    /// locks that block this id.
    fn wait_for_file_system_locks(
        &self,
        id: Option<&Arc<Id>>,
        action: Option<StandardBarString>,
        to_file_name: Option<&str>,
        timer: Option<&mut SleepTimer>,
    ) -> Result<(), LockException> {
        let log_file = self.log_file.upgrade().unwrap();
        let file = &log_file.file;
        let id = match id {
            None => return Ok(()),
            Some(id) => id,
        };
        if !file.exists() {
            // A file that doesn't exist isn't locked in the file system.
            return Ok(());
        }
        // If no state was passed in, then this will be able to wait the full sleep time
        // and will be slower.
        let mut own_timer;
        let timer = match timer {
            Some(timer) => timer,
            None => {
                own_timer = SleepTimer::new();
                &mut own_timer
            }
        };

        // Set the type of lock.
        let mut option1: Option<StandardOpenOption> = None;
        let mut option2: Option<StandardOpenOption> = None;
        let mut shared = false;
        if id.get_id_type() == IdType::File {
            option1 = Some(StandardOpenOption::Read);
            option2 = Some(StandardOpenOption::Write);
        } else if id.get_id_type().get_lock_type() == LockType::Read
            || id.get_id_type() == IdType::InputStream
        {
            option1 = Some(StandardOpenOption::Read);
            shared = true;
        } else {
            option1 = Some(StandardOpenOption::Write);
        }

        // Test for blocking file system locks.
        let mut lock_exception: Option<LockException> = None;

        // Try nio.FileChannel first (first try). If it doesn't work because an existing
        // file cannot be found, then the Path functionality may not work on this
        // computer. Use FileReader and/or FileWriter as a fallback (second try).
        let mut use_nio = true;
        let num_tries = 2;
        for _i in 0..num_tries {
            if !file.exists() {
                return Ok(());
            }
            let channel: Option<std::fs::File>;
            if use_nio {
                let mut open_options = std::fs::OpenOptions::new();
                open_options.read(option1 == Some(StandardOpenOption::Read));
                open_options.write(
                    option1 == Some(StandardOpenOption::Write)
                        || option2 == Some(StandardOpenOption::Write),
                );
                match open_options.open(file) {
                    Ok(open_file) => channel = Some(open_file),
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                        if !file.exists() {
                            return Ok(());
                        }
                        // The Path functionality may not work on this computer. Use
                        // FileReader and/or FileWriter as a fallback for the second try.
                        use_nio = false;
                        continue;
                    }
                    Err(_) => {
                        // Something else is wrong. Let the real file manipulation uncover
                        // and document it.
                        return Ok(());
                    }
                }
            } else {
                channel = None;
            }
            // Use FileChannel.tryLock to test for a locked file. If that doesn't work,
            // use FileReader/FileWriter.
            let read = option1 == Some(StandardOpenOption::Read);
            let write = option1 == Some(StandardOpenOption::Write)
                || option2 == Some(StandardOpenOption::Write);
            loop {
                lock_exception = None;
                if !file.exists() {
                    return Ok(());
                }
                // `channel.tryLock(0L, Long.MAX_VALUE, shared)`, then, when NIO is not in
                // use, `new FileReader(file)` and `new FileWriter(file)`.
                let mut overlapping_file_lock = false;
                let mut io_exception: Option<std::io::Error> = None;
                if let Some(channel) = &channel {
                    use std::os::unix::fs::MetadataExt;
                    use std::os::unix::io::AsRawFd;
                    let key = match channel.metadata() {
                        Ok(metadata) => (metadata.dev(), metadata.ino()),
                        Err(_) => return Ok(()),
                    };
                    let mut table = FILE_LOCK_TABLE.lock().unwrap();
                    if table.contains(&key) {
                        overlapping_file_lock = true;
                    } else {
                        table.push(key);
                        drop(table);
                        unsafe {
                            let mut flock: libc::flock = std::mem::zeroed();
                            flock.l_type = if shared {
                                libc::F_RDLCK as libc::c_short
                            } else {
                                libc::F_WRLCK as libc::c_short
                            };
                            flock.l_whence = libc::SEEK_SET as libc::c_short;
                            flock.l_start = 0;
                            flock.l_len = 0;
                            // A refusal here is another process's lock, which `tryLock`
                            // reports by returning null - the source then falls through to
                            // its `return`.
                            libc::fcntl(channel.as_raw_fd(), libc::F_SETLK, &flock);
                            // The source's try-with-resources releases the lock at the end
                            // of a block whose only statement is `return`.
                            flock.l_type = libc::F_UNLCK as libc::c_short;
                            libc::fcntl(channel.as_raw_fd(), libc::F_SETLK, &flock);
                        }
                        FILE_LOCK_TABLE
                            .lock()
                            .unwrap()
                            .retain(|entry| *entry != key);
                    }
                }
                if !overlapping_file_lock && io_exception.is_none() && !use_nio && read {
                    if let Err(e) = std::fs::File::open(file) {
                        if e.kind() == std::io::ErrorKind::NotFound {
                            return Ok(());
                        }
                        io_exception = Some(e);
                    }
                }
                if !overlapping_file_lock && io_exception.is_none() && !use_nio && write {
                    if let Err(e) = std::fs::OpenOptions::new()
                        .write(true)
                        .create(true)
                        .truncate(true)
                        .open(file)
                    {
                        if e.kind() == std::io::ErrorKind::NotFound {
                            return Ok(());
                        }
                        io_exception = Some(e);
                    }
                }
                if overlapping_file_lock {
                    // File is already locked within etomo.
                    lock_exception = Some(LockException::new(
                        &log_file,
                        "File already opened in etomo.",
                        None,
                        None,
                        action,
                        to_file_name,
                        false,
                        Some(LockExceptionCause::OverlappingFileLock(
                            "java.nio.channels.OverlappingFileLockException".to_string(),
                        )),
                    ));
                } else if let Some(io_exception) = io_exception {
                    // Assuming that the file is already locked.
                    lock_exception = Some(LockException::new(
                        &log_file,
                        &format!(
                            "File may have already been opened{}",
                            if use_nio {
                                " in another application"
                            } else {
                                "null"
                            }
                        ),
                        None,
                        None,
                        action,
                        to_file_name,
                        false,
                        Some(LockExceptionCause::Io(io_exception.to_string())),
                    ));
                } else {
                    // Able to lock.
                    return Ok(());
                }
                timer.do_sleep();
                if timer.is_sleep_limit_reached() {
                    break;
                }
            }
        }
        match lock_exception {
            None => Ok(()),
            Some(lock_exception) => Err(lock_exception),
        }
    }

    /// Java `unlock`.  Unlock or just check if unlockable.
    fn unlock(&self, id: Option<&Arc<Id>>) {
        let log_file = self.log_file.upgrade().unwrap();
        let id = match id {
            None => return,
            Some(id) => id,
        };
        if !id.is_lock_number_set() {
            return;
        }
        let real_id = match self.get_real_id(Some(id)) {
            None => {
                eprintln!("Warning: real id not set. id:{}", id);
                return;
            }
            Some(real_id) => real_id,
        };
        if !self.is_locked_id(Some(&real_id)) {
            eprintln!("Warning: real id not locked. id:{},realId:{}", id, real_id);
            return;
        }
        if let Some(lock_number_to_track) = *log_file.lock_number_to_track.lock().unwrap() {
            if real_id.equals_integer(Some(lock_number_to_track)) {
                Reporter::new_message_blocking_id_verbose(
                    &log_file,
                    &format!("Lock track:unlocked id:{}", real_id.to_string_verbose(true)),
                    Some(id),
                    true,
                )
                .print();
            }
        }
        // Remove the lock id
        let lock_type = real_id.get_lock_type();
        if lock_type == LockType::File {
            *self.file_id.lock().unwrap() = None;
            if etomo_director::ARGUMENTS.lock().unwrap().is_debug() {
                eprintln!("(2) fileId set to null");
            }
        } else if lock_type == LockType::Write {
            *self.write_id.lock().unwrap() = None;
        } else if lock_type == LockType::Read {
            if let Some(lock_number) = real_id.get_lock_number() {
                let mut read_id_hash_map = self.read_id_hash_map.lock().unwrap();
                if let Some(mapped) = read_id_hash_map.get(&lock_number) {
                    if Arc::ptr_eq(mapped, &real_id) {
                        read_id_hash_map.remove(&lock_number);
                    }
                }
            }
        }
        id.reset_lock_number();
        real_id.reset_lock_number();
    }

    /// Java `isLocked()`.  True if this lock instance has any locks.
    fn is_locked(&self) -> bool {
        self.is_locked_lock_type(Some(LockType::File))
            || self.is_locked_lock_type(Some(LockType::Write))
            || self.is_locked_lock_type(Some(LockType::Read))
    }

    /// Java `isLocked(Id)`.  Return true if `id` completely matches a lock id, including
    /// the handle.  Used for gaining permission to read or modify a file, or to unlock
    /// this file.  If `id` has a lock number, but it isn't one of this `Lock` instance's
    /// locks, then it most likely belongs to another file, and shouldn't be used to lock
    /// this file.
    fn is_locked_id(&self, id: Option<&Arc<Id>>) -> bool {
        let id = match id {
            None => return false,
            Some(id) => id,
        };
        if !id.is_valid() || !id.is_lock_number_set() {
            return false;
        }
        let lock_number = match id.get_lock_number() {
            None => return false,
            Some(lock_number) => lock_number,
        };
        if id.equals_id(self.file_id.lock().unwrap().as_ref())
            || id.equals_id(self.write_id.lock().unwrap().as_ref())
        {
            return true;
        }
        let read_id = self
            .read_id_hash_map
            .lock()
            .unwrap()
            .get(&lock_number)
            .cloned();
        id.equals_id(read_id.as_ref())
    }

    /// Java `contains`.  Returns true if the lock contains this `Id` instance.
    fn contains(&self, id: Option<&Arc<Id>>) -> bool {
        let id = match id {
            None => return false,
            Some(id) => id,
        };
        if let Some(file_id) = &*self.file_id.lock().unwrap() {
            if Arc::ptr_eq(id, file_id) {
                return true;
            }
        }
        if let Some(write_id) = &*self.write_id.lock().unwrap() {
            if Arc::ptr_eq(id, write_id) {
                return true;
            }
        }
        let lock_number = match id.get_lock_number() {
            None => return false,
            Some(lock_number) => lock_number,
        };
        match self.read_id_hash_map.lock().unwrap().get(&lock_number) {
            None => false,
            Some(read_id) => Arc::ptr_eq(id, read_id),
        }
    }

    /// Java `getRealId`.  The `id` parameter may be an identical copy of a real `Id`.  Or
    /// it might be out of date or belong to a different `LogFile` or handle.  Return the
    /// real `Id` if the `id` parameter is identical to it.
    fn get_real_id(&self, id: Option<&Arc<Id>>) -> Option<Arc<Id>> {
        let id = id?;
        if !id.is_valid() {
            return None;
        }
        let file_id = self.file_id.lock().unwrap().clone();
        if id.equals_id(file_id.as_ref()) {
            return file_id;
        }
        let write_id = self.write_id.lock().unwrap().clone();
        if id.equals_id(write_id.as_ref()) {
            return write_id;
        }
        let read_id = match id.get_lock_number() {
            None => None,
            Some(lock_number) => self
                .read_id_hash_map
                .lock()
                .unwrap()
                .get(&lock_number)
                .cloned(),
        };
        if id.equals_id(read_id.as_ref()) {
            return read_id;
        }
        None
    }

    /// Java `isLocked(Integer, IdType)`.  Return true if the id matches everything but
    /// the handle of a lock id.  Used for locking.
    fn is_locked_lock_number(&self, lock_number: Option<i32>, id_type: Option<IdType>) -> bool {
        let (lock_number, id_type) = match (lock_number, id_type) {
            (Some(lock_number), Some(id_type)) => (lock_number, id_type),
            _ => return false,
        };
        let file_id = self.file_id.lock().unwrap().clone();
        let write_id = self.write_id.lock().unwrap().clone();
        if match &file_id {
            None => false,
            Some(file_id) => {
                file_id.is_valid()
                    && file_id.equals_integer(Some(lock_number))
                    && file_id.equals_id_type(id_type)
            }
        } || match &write_id {
            None => false,
            Some(write_id) => {
                write_id.is_valid()
                    && write_id.equals_integer(Some(lock_number))
                    && write_id.equals_id_type(id_type)
            }
        } {
            return true;
        }
        let read_id = self
            .read_id_hash_map
            .lock()
            .unwrap()
            .get(&lock_number)
            .cloned();
        match read_id {
            None => false,
            Some(read_id) => {
                read_id.is_valid()
                    && read_id.equals_integer(Some(lock_number))
                    && read_id.equals_id_type(id_type)
            }
        }
    }

    /// Java `isLocked(LockType)`.  Return true if this lock instance has any locks for a
    /// lockType.
    fn is_locked_lock_type(&self, lock_type: Option<LockType>) -> bool {
        let lock_type = match lock_type {
            None => return self.is_locked(),
            Some(lock_type) => lock_type,
        };
        if lock_type == LockType::File {
            return match &*self.file_id.lock().unwrap() {
                None => false,
                Some(file_id) => file_id.is_lock_number_set() && file_id.is_valid(),
            };
        }
        if lock_type == LockType::Write {
            return match &*self.write_id.lock().unwrap() {
                None => false,
                Some(write_id) => write_id.is_lock_number_set() && write_id.is_valid(),
            };
        } else if lock_type == LockType::Read {
            let read_id_hash_map = self.read_id_hash_map.lock().unwrap();
            let mut iterator = read_id_hash_map.values();
            while let Some(read_id) = iterator.next() {
                if read_id.is_lock_number_set() && read_id.is_valid() {
                    return true;
                }
            }
        }
        false
    }

    /// Java `isLockable`.  Returns true if the id parameter can be added to the lock
    /// instance.
    fn is_lockable(&self, id: Option<&Arc<Id>>) -> bool {
        let id = match id {
            None => return false,
            Some(id) => id,
        };
        if !id.is_valid() {
            return false;
        }
        // File locks are exclusive.
        if id.equals_lock_type(LockType::File) {
            return !self.is_locked();
        }
        // Write locks can only coexist with read locks.
        if id.equals_lock_type(LockType::Write) {
            return !self.is_locked_lock_type(Some(LockType::File))
                && !self.is_locked_lock_type(Some(LockType::Write));
        }
        // Read locks can coexist with one write lock and multiple read locks.
        !self.is_locked_lock_type(Some(LockType::File))
    }

    /// Java `checkBlockingIds`.  If possible, remove `Id`s that prevent the `Id`
    /// parameter from locking.  `overrideLocks` allows `Id`s to be removed.
    fn check_blocking_ids(
        &self,
        new_id: Option<&Arc<Id>>,
        action: Option<StandardBarString>,
        to_file_name: Option<&str>,
    ) -> Result<bool, LogFileError> {
        let log_file = self.log_file.upgrade().unwrap();
        let new_id = match new_id {
            None => return Ok(true),
            Some(new_id) => new_id,
        };
        if !new_id.is_valid() {
            Reporter::new_message_blocking_id_verbose(
                &log_file,
                "Warning: invalid id",
                Some(new_id),
                true,
            )
            .print();
            return Ok(true);
        }
        if !self.is_locked() || self.is_lockable(Some(new_id)) {
            return Ok(true);
        }
        let mut iterator = BlockingIdIterator::new(Some(self), Some(new_id.get_lock_type()));
        // Make sure an id really blocks newId. Throws LockException for a real block.
        while iterator.has_next() {
            self.check_blocking_id(Some(new_id), iterator.next().as_ref(), action, to_file_name)?;
        }
        Ok(true)
    }

    /// Java `checkBlockingId`.  Make sure `blockingId` is really a valid, blocking lock.
    fn check_blocking_id(
        &self,
        new_id: Option<&Arc<Id>>,
        blocking_id: Option<&Arc<Id>>,
        action: Option<StandardBarString>,
        to_file_name: Option<&str>,
    ) -> Result<bool, LogFileError> {
        let log_file = self.log_file.upgrade().unwrap();
        let (new_id, blocking_id) = match (new_id, blocking_id) {
            (Some(new_id), Some(blocking_id)) => (new_id, blocking_id),
            // Nothing to override
            _ => return Ok(true),
        };
        if !blocking_id.is_valid() || !blocking_id.is_lock_number_set() {
            return Ok(true);
        }
        let mut blocking_lock_type: Option<LockType> = None;
        // Make sure this is really an id from the lock instance, and get the lockType.
        let blocking_lock_number = blocking_id.get_lock_number();
        let blocking_read_id;
        let file_id = self.file_id.lock().unwrap().clone();
        let write_id = self.write_id.lock().unwrap().clone();
        if match &file_id {
            None => false,
            Some(file_id) => Arc::ptr_eq(blocking_id, file_id),
        } || blocking_id.equals_id(file_id.as_ref())
        {
            blocking_lock_type = Some(LockType::File);
        } else if match &write_id {
            None => false,
            Some(write_id) => Arc::ptr_eq(blocking_id, write_id),
        } || blocking_id.equals_id(write_id.as_ref())
        {
            blocking_lock_type = Some(LockType::Write);
        } else {
            // Check lock number
            blocking_read_id = match blocking_lock_number {
                None => None,
                Some(blocking_lock_number) => self
                    .read_id_hash_map
                    .lock()
                    .unwrap()
                    .get(&blocking_lock_number)
                    .cloned(),
            };
            if match &blocking_read_id {
                None => false,
                Some(blocking_read_id) => Arc::ptr_eq(blocking_id, blocking_read_id),
            } || blocking_id.equals_id(blocking_read_id.as_ref())
            {
                blocking_lock_type = Some(LockType::Read);
            }
        }
        if blocking_lock_type.is_none() {
            // blockingLockId is not in lock instance, so there's nothing to override.
            return Ok(true);
        }
        // System.err.println("Error:" + newId.getHandle().getName());
        // newId.getHandle().printOwnerStackTrace("newId owner");
        // blockingId.getHandle().printOwnerStackTrace("blockingId owner");
        Err(LogFileError::Lock(LockException::new(
            &log_file,
            &format!(
                "Unable to lock {} because of another etomo lock.",
                new_id.get_handle().unwrap().get_name()
            ),
            Some(new_id),
            Some(blocking_id),
            action,
            to_file_name,
            true,
            None,
        )))
    }

    /// Java `getWriteId`.
    fn get_write_id(&self) -> i32 {
        self.write_id
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .get_lock_number()
            .unwrap()
    }
}

/// Java `toString()` on `Lock`.
impl std::fmt::Display for Lock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.to_string_verbose(false))
    }
}

/// Java `Id.NO_ID`.
const NO_ID: i32 = -1;

/// Java `Id`.  Keep constructors for `Id` classes private to avoid having to keep track
/// of which kind of id is being used.
#[derive(Debug)]
pub struct Id {
    /// Java field `handle`.
    handle: Option<Arc<Handle>>,
    /// Java field `idType`.
    id_type: IdType,
    /// Java field `callStackTrace`.
    call_stack_trace: Mutex<StackTrace>,
    /// Java field `lockNumber`.
    lock_number: Mutex<Option<i32>>,
    /// Java field `lockStackTrace`.
    lock_stack_trace: Mutex<Option<StackTrace>>,
}

impl Id {
    /// Java `Id(LogFile.Handle, IdType)`.  `idType` is required.
    fn new(handle: Option<&Arc<Handle>>, id_type: IdType) -> Arc<Id> {
        Arc::new(Id {
            handle: handle.cloned(),
            id_type,
            call_stack_trace: Mutex::new(StackTrace::new_with_title(Some("contructed"))),
            lock_number: Mutex::new(None),
            lock_stack_trace: Mutex::new(None),
        })
    }

    /// Java `close`, from `AutoCloseable`.
    pub fn close(self: &Arc<Id>) {
        self.handle
            .as_ref()
            .unwrap()
            .log_file
            .lock
            .unlock(Some(self));
    }

    /// Java `getDescr`, including every subclass override: `ReaderId`, `BigBufferReaderId`,
    /// `CopyFromId`, `ReadingId`, `InputStreamId`, `WritingId` and `OutputStreamId` each
    /// replace the `idType`'s description, and each subclass has exactly one `IdType`, so
    /// the dynamic dispatch is a match on that field.
    fn get_descr(&self) -> &'static str {
        match self.id_type {
            // `ReaderId.getDescr`.
            IdType::Reader => "reader",
            // `BigBufferReaderId.getDescr`.
            IdType::BigBufferReader => " buffered reader",
            // `CopyFromId.getDescr`.
            IdType::CopyFrom => " buffered reader",
            // `ReadingId.getDescr`.
            IdType::Reading => " for reading",
            // `InputStreamId.getDescr`.
            IdType::InputStream => "input stream",
            // `WritingId.getDescr`: `"for " + super.getDescr()`.
            IdType::Writing => "for writing",
            // `OutputStreamId.getDescr`.
            IdType::OutputStream => "output stream",
            // `FileId` and `WriterId` do not override it.
            IdType::File | IdType::Writer => self.id_type.get_descr(),
        }
    }

    /// Java `getCallStackTrace`.
    fn get_call_stack_trace(&self) -> &Mutex<StackTrace> {
        &self.call_stack_trace
    }

    /// Java `printStackTrace(String)`.
    fn print_stack_trace(&self, title: Option<&str>) {
        let mut lock_stack_trace = self.lock_stack_trace.lock().unwrap();
        if let Some(lock_stack_trace) = &mut *lock_stack_trace {
            lock_stack_trace.print(title, false);
        } else {
            self.call_stack_trace.lock().unwrap().print(title, false);
        }
    }

    /// Java `isValid`.
    fn is_valid(&self) -> bool {
        self.handle.is_some()
    }

    /// Java `getLockType`.
    fn get_lock_type(&self) -> LockType {
        self.id_type.get_lock_type()
    }

    /// Java `getIdType`.
    fn get_id_type(&self) -> IdType {
        self.id_type
    }

    /// Java `getHandle`.
    fn get_handle(&self) -> Option<&Arc<Handle>> {
        self.handle.as_ref()
    }

    /// Java `setLockNumber`.
    fn set_lock_number(&self, lock_number: Option<i32>) {
        *self.lock_number.lock().unwrap() = lock_number;
        *self.lock_stack_trace.lock().unwrap() = Some(StackTrace::new_with_title(Some("locked")));
    }

    /// Java `isLockNumberSet`.
    fn is_lock_number_set(&self) -> bool {
        match *self.lock_number.lock().unwrap() {
            None => false,
            Some(lock_number) => lock_number > NO_ID,
        }
    }

    /// Java `isEmpty`.
    pub fn is_empty(&self) -> bool {
        !self.is_lock_number_set()
    }

    /// Java `getLockNumber`.
    fn get_lock_number(&self) -> Option<i32> {
        match *self.lock_number.lock().unwrap() {
            None => None,
            Some(lock_number) if lock_number <= NO_ID => None,
            Some(lock_number) => Some(lock_number),
        }
    }

    /// Java `getNumber`.
    fn get_number(&self) -> i32 {
        match *self.lock_number.lock().unwrap() {
            None => NO_ID,
            Some(lock_number) if lock_number < NO_ID => NO_ID,
            Some(lock_number) => lock_number,
        }
    }

    /// Java `resetLockNumber`.
    fn reset_lock_number(&self) {
        *self.lock_number.lock().unwrap() = None;
        *self.lock_stack_trace.lock().unwrap() = None;
    }

    /// Java `equals(Object)`, `instanceof Id` arm.
    fn equals_id(&self, object: Option<&Arc<Id>>) -> bool {
        let compare_id = match object {
            None => return false,
            Some(compare_id) => compare_id,
        };
        if !self.equals_handle(compare_id.handle.as_ref()) {
            return false;
        }
        // `compare_id` may be this very `Id` - `Lock.isLocked(Id)` compares an id against
        // the lock's own copy of it - so the compared lock number is read out before
        // `equals_integer` locks this id's.  Java's field read needs no such care.
        let compare_lock_number = *compare_id.lock_number.lock().unwrap();
        self.equals_integer(compare_lock_number) && self.id_type == compare_id.id_type
    }

    /// Java `equals(Object)`, `instanceof LogFile.Handle` arm.
    fn equals_handle(&self, object: Option<&Arc<Handle>>) -> bool {
        match (&self.handle, object) {
            (Some(handle), Some(object)) => Arc::ptr_eq(handle, object),
            _ => false,
        }
    }

    /// Java `equals(Object)`, `instanceof String` arm, which assumes the string is
    /// `LogFile.fileAbsolutePath`.
    fn equals_string(&self, object: Option<&str>) -> bool {
        let object = match object {
            None => return false,
            Some(object) => object,
        };
        let handle = match &self.handle {
            None => return false,
            Some(handle) => handle,
        };
        handle.log_file.equals_string(Some(object))
    }

    /// Java `equals(Object)`, `instanceof Integer` arm.  A null lockNumber means that the
    /// id isn't unique; ids with no lockNumber can't be used to retrieve an id.
    fn equals_integer(&self, object: Option<i32>) -> bool {
        let compare_lock_number = match object {
            None => return false,
            Some(compare_lock_number) => compare_lock_number,
        };
        let lock_number = *self.lock_number.lock().unwrap();
        self.is_lock_number_set()
            && match lock_number {
                None => false,
                Some(lock_number) => lock_number > NO_ID && lock_number == compare_lock_number,
            }
    }

    /// Java `equals(Object)`, `instanceof IdType` arm.
    fn equals_id_type(&self, object: IdType) -> bool {
        self.id_type == object
    }

    /// Java `equals(Object)`, `instanceof LockType` arm.
    fn equals_lock_type(&self, object: LockType) -> bool {
        self.get_lock_type() == object
    }

    /// Java `toString(boolean)`.
    fn to_string_verbose(&self, verbose: bool) -> String {
        format!(
            "[{},{}{}]",
            match *self.lock_number.lock().unwrap() {
                None => "null".to_string(),
                Some(lock_number) => lock_number.to_string(),
            },
            self.id_type,
            match &self.handle {
                None => String::new(),
                Some(handle) => format!(",{}", handle.to_string_verbose(verbose)),
            }
        )
    }

    /// Java `print(String)`.
    pub fn print(&self, descr: &str) {
        eprintln!("{}:{}", descr, self);
    }
}

/// Java `toString()` on `Id`.
impl std::fmt::Display for Id {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.to_string_verbose(false))
    }
}

/// Java `FileId`.  Keep constructors for `Id` classes private to avoid having to keep
/// track of which kind of id is being used.
#[derive(Clone, Debug)]
pub struct FileId {
    /// The `Id` superclass state.  Java's `Id`s are compared by reference in
    /// `Lock.contains`, so the subclass wrapper shares one allocation.
    id: Arc<Id>,
}

impl std::ops::Deref for FileId {
    type Target = Arc<Id>;
    fn deref(&self) -> &Arc<Id> {
        &self.id
    }
}

impl FileId {
    /// Java `FileId(LogFile.Handle)`.
    fn new(handle: &Arc<Handle>) -> FileId {
        FileId {
            id: Id::new(Some(handle), IdType::File),
        }
    }
}

/// Java `ReadId`, the abstract superclass of `ReaderId`, `BigBufferReaderId`,
/// `CopyFromId` and `ReadingId`.  Rust has no subtyping, so a parameter the source
/// declares as `ReadId` takes the `Id` superclass here and this type carries only the
/// constructor.
pub struct ReadId;

impl ReadId {
    /// Java `ReadId(LogFile.Handle, IdType)`.
    fn new(handle: &Arc<Handle>, id_type: Option<IdType>) -> Arc<Id> {
        Id::new(
            Some(handle),
            match id_type {
                None => IdType::Reading,
                Some(id_type) => id_type,
            },
        )
    }
}

/// Java `ReaderId`.
#[derive(Clone, Debug)]
pub struct ReaderId {
    /// The `ReadId` superclass state.
    id: Arc<Id>,
}

impl std::ops::Deref for ReaderId {
    type Target = Arc<Id>;
    fn deref(&self) -> &Arc<Id> {
        &self.id
    }
}

impl ReaderId {
    /// Java `ReaderId(LogFile.Handle)`.
    fn new(handle: &Arc<Handle>) -> ReaderId {
        ReaderId {
            id: ReadId::new(handle, Some(IdType::Reader)),
        }
    }
}

/// Java `BigBufferReaderId`.
#[derive(Clone, Debug)]
pub struct BigBufferReaderId {
    /// The `ReadId` superclass state.
    id: Arc<Id>,
}

impl std::ops::Deref for BigBufferReaderId {
    type Target = Arc<Id>;
    fn deref(&self) -> &Arc<Id> {
        &self.id
    }
}

impl BigBufferReaderId {
    /// Java `BigBufferReaderId(LogFile.Handle)`.
    fn new(handle: &Arc<Handle>) -> BigBufferReaderId {
        BigBufferReaderId {
            id: ReadId::new(handle, Some(IdType::BigBufferReader)),
        }
    }
}

/// Java `CopyFromId`.
#[derive(Clone, Debug)]
pub struct CopyFromId {
    /// The `ReadId` superclass state.
    id: Arc<Id>,
}

impl std::ops::Deref for CopyFromId {
    type Target = Arc<Id>;
    fn deref(&self) -> &Arc<Id> {
        &self.id
    }
}

impl CopyFromId {
    /// Java `CopyFromId(LogFile.Handle)`.
    fn new(handle: &Arc<Handle>) -> CopyFromId {
        CopyFromId {
            id: ReadId::new(handle, Some(IdType::CopyFrom)),
        }
    }
}

/// Java `ReadingId`.
#[derive(Clone, Debug)]
pub struct ReadingId {
    /// The `ReadId` superclass state.
    id: Arc<Id>,
}

impl std::ops::Deref for ReadingId {
    type Target = Arc<Id>;
    fn deref(&self) -> &Arc<Id> {
        &self.id
    }
}

impl ReadingId {
    /// Java `ReadingId(LogFile.Handle)`.
    fn new(handle: &Arc<Handle>) -> ReadingId {
        ReadingId {
            id: ReadId::new(handle, Some(IdType::Reading)),
        }
    }
}

/// Java `InputStreamId`.  Only one input stream can be used at a time, so it works just
/// like an output stream or a writer - and is locked using the lockType WRITE.
#[derive(Clone, Debug)]
pub struct InputStreamId {
    /// The `Id` superclass state.
    id: Arc<Id>,
}

impl std::ops::Deref for InputStreamId {
    type Target = Arc<Id>;
    fn deref(&self) -> &Arc<Id> {
        &self.id
    }
}

impl InputStreamId {
    /// Java `InputStreamId(LogFile.Handle)`.
    fn new(handle: &Arc<Handle>) -> InputStreamId {
        InputStreamId {
            id: Id::new(Some(handle), IdType::InputStream),
        }
    }
}

/// Java `WriterId`.
#[derive(Clone, Debug)]
pub struct WriterId {
    /// The `Id` superclass state.
    id: Arc<Id>,
}

impl std::ops::Deref for WriterId {
    type Target = Arc<Id>;
    fn deref(&self) -> &Arc<Id> {
        &self.id
    }
}

impl WriterId {
    /// Java `WriterId(LogFile.Handle)`.
    fn new(handle: &Arc<Handle>) -> WriterId {
        WriterId {
            id: Id::new(Some(handle), IdType::Writer),
        }
    }
}

/// Java `WritingId`.
#[derive(Clone, Debug)]
pub struct WritingId {
    /// The `Id` superclass state.
    id: Arc<Id>,
}

impl std::ops::Deref for WritingId {
    type Target = Arc<Id>;
    fn deref(&self) -> &Arc<Id> {
        &self.id
    }
}

impl WritingId {
    /// Java `WritingId(LogFile.Handle)`.
    fn new(handle: &Arc<Handle>) -> WritingId {
        WritingId {
            id: Id::new(Some(handle), IdType::Writing),
        }
    }
}

/// Java `OutputStreamId`.
#[derive(Clone, Debug)]
pub struct OutputStreamId {
    /// The `Id` superclass state.
    id: Arc<Id>,
}

impl std::ops::Deref for OutputStreamId {
    type Target = Arc<Id>;
    fn deref(&self) -> &Arc<Id> {
        &self.id
    }
}

impl OutputStreamId {
    /// Java `OutputStreamId(LogFile.Handle)`.
    fn new(handle: &Arc<Handle>) -> OutputStreamId {
        OutputStreamId {
            id: Id::new(Some(handle), IdType::OutputStream),
        }
    }
}

/// Java `Lockable<V>`.  Class for avoiding problems with single thread mode.
///
/// Override the `attempt` function in public functions which may cause an id to be added
/// to the lock (see `Lock.lock(Id)`).  The `retry` function goes in and out of
/// synchronization (single thread mode) for each `attempt()` call.
///
/// When the blocking lock id has a different handle, staying in single thread mode during
/// repeated lock attempts prevents the other handle from doing anything that changes its
/// timestamp.  This will make the blocking lock id look abandoned (and removeable) when
/// it's actually active.
///
/// For a lock id with the same handle, there are always at least two threads involved
/// with running a process - the thread that ran the process and the monitor thread.  So
/// staying in single thread mode could alter Etomo's behavior.
///
/// Java's callers all subclass this anonymously to override one abstract method, so the
/// override is a closure field here.
struct Lockable<'a, V> {
    /// Java field `handle`.
    handle: Arc<Handle>,
    /// The anonymous subclass's `attempt` override.
    attempt: Box<dyn FnMut(&mut SleepTimer) -> Result<V, LogFileError> + 'a>,
}

impl<'a, V> Lockable<'a, V> {
    /// Java `Lockable(LogFile.Handle)`.
    fn new(
        handle: &Arc<Handle>,
        attempt: Box<dyn FnMut(&mut SleepTimer) -> Result<V, LogFileError> + 'a>,
    ) -> Lockable<'a, V> {
        Lockable {
            handle: handle.clone(),
            attempt,
        }
    }

    /// Java `attempt(SleepTimer)`.  Override this to run `LogFile` functions that run
    /// `Lock.lock()` and are not private or are called from `LogFile.Handle`.
    fn attempt(&mut self, timer: &mut SleepTimer) -> Result<V, LogFileError> {
        (self.attempt)(timer)
    }

    /// Java `retry(Id)`.
    fn retry(&mut self, new_id: &Id) -> Result<V, LogFileError> {
        self.retry_blocking_id(new_id, false, false)
    }

    /// Java `retry(Id, boolean, boolean)`.  Runs the `attempt()` function multiple times.
    /// The total runs take up enough time for an inactive handle to be considered
    /// abandoned.  And it unsynchronizes after each attempt to allow this `LogFile`
    /// instance to clean up blocking locks.  If the retries fail, it throws the
    /// `LockException`.  `Lockable.attempt()` should call a `LogFile` function that
    /// attempts to create a file lock.
    fn retry_blocking_id(
        &mut self,
        new_id: &Id,
        is_blocking_id_for_test: bool,
        primary_monitor: bool,
    ) -> Result<V, LogFileError> {
        // Set up blocking IDs for stress test and fail test.
        if !is_blocking_id_for_test {
            self.lock_blocking_ids_for_test(new_id.handle.as_ref().unwrap());
        }
        let mut timer = SleepTimer::new();
        let mut lock_exception: Option<LockException>;
        let mut variable: Option<V> = None;
        loop {
            lock_exception = None;
            match self.attempt(&mut timer) {
                Ok(value) => {
                    variable = Some(value);
                    break;
                }
                Err(LogFileError::Lock(e)) => {
                    lock_exception = Some(e);
                }
                Err(e) => return Err(e),
            }
            timer.do_sleep();
            if timer.is_sleep_limit_reached() {
                break;
            }
        }

        // The lock exception was thrown, and there no sleep time left.
        if let Some(lock_exception) = lock_exception {
            self.print_diagnostics(Some(new_id), &lock_exception);
            // Use the emergency monitor to inform the user.
            if let Some(emergency_monitor) = &self.handle.emergency_monitor {
                emergency_monitor.alert(Some(&lock_exception), primary_monitor);
            }
            return Err(LogFileError::Lock(lock_exception));
        }
        Ok(variable.unwrap())
    }

    /// Java `lockBlockingIdsForTest`.
    fn lock_blocking_ids_for_test(&self, handle: &Arc<Handle>) {
        let test_level = etomo_director::ARGUMENTS.lock().unwrap().get_test_level();
        let test_level = match test_level {
            None => return,
            Some(test_level) => test_level,
        };
        let num_blocking_ids = test_level.get_num_blocking_ids();
        if num_blocking_ids == 0 {
            return;
        }

        // All purpose stress test. Blocks any lock, including the other stress test
        // locks, for a limited amount of time.
        let stress_test_handle = handle.log_file.create_stress_test_handle();
        *handle.log_file.stress_test_handle.lock().unwrap() = stress_test_handle.clone();
        for _i in 0..num_blocking_ids {
            let stress_test_handle = match &stress_test_handle {
                None => break,
                Some(stress_test_handle) => stress_test_handle,
            };
            match stress_test_handle.open_file_lock(true) {
                Ok(blocking_id) => {
                    if let Err(e) = stress_test_handle.close_file_lock_with_wait(
                        &blocking_id,
                        ((SLEEP_LIMIT as f64) / 10.0).round() as i64,
                        true,
                    ) {
                        eprintln!("{}", e);
                    }
                }
                Err(e) => eprintln!("{}", e),
            }
        }
    }

    /// Java `printDiagnostics`.
    fn print_diagnostics(&self, new_id: Option<&Id>, lock_exception: &LockException) {
        // SleepLimit has been exceeded and the attempt() function never succeeded.
        // Diagnostics for file lock collision.
        let action = lock_exception.get_action();
        {
            let mut standard_bar_string = self.handle.standard_bar_string.lock().unwrap();
            match *standard_bar_string {
                None => *standard_bar_string = action,
                Some(current) => {
                    // `replaceWithPrecedence(action)` dereferences its argument, so a null
                    // action throws here.
                    let replaced = current.replace_with_precedence(action.unwrap());
                    *standard_bar_string = Some(replaced);
                    *lock_exception.action.lock().unwrap() = Some(replaced);
                }
            }
        }
        // `DebugLevel debugLevel = EtomoDirector.ARGUMENTS.getDebugLevel();` and its
        // `if (debugLevel == null) debugLevel = DebugLevel.OFF;`, which cannot be entered
        // here: `getDebugLevel` returns the enum, not a reference.
        let debug_level = etomo_director::ARGUMENTS.lock().unwrap().get_debug_level();
        eprintln!("Error: lock failed.  {}", lock_exception.get_message());
        lock_exception.print_stack_trace();
        if let Some(new_id) = new_id {
            eprintln!("\nNew ID:{}", new_id);
            if debug_level.is_on() {
                new_id
                    .handle
                    .as_ref()
                    .unwrap()
                    .print_owner_stack_trace_title(Some("New ID Owner"));
            }
            new_id.print_stack_trace(Some("New ID"));
        }
        if let Some(blocking_id) = lock_exception.get_blocking_id() {
            eprintln!("\nBlocking ID:{}", blocking_id);
            if debug_level.is_on() {
                blocking_id
                    .handle
                    .as_ref()
                    .unwrap()
                    .print_owner_stack_trace_title(Some("Blocking ID Owner"));
            }
            blocking_id.print_stack_trace(Some("Blocking ID"));
        }
        if debug_level.is_on() {
            eprintln!("Handle attempting to lock:{}", self.handle);
            self.handle
                .owner_stack_trace
                .lock()
                .unwrap()
                .print(Some("Handle owner"), false);
        }
        if debug_level.is_verbose() {
            StackTrace::print_all(false);
        }
        if let Some(cause) = lock_exception.get_cause() {
            eprintln!("Cause: {}", cause);
            eprintln!("{}", cause);
        }
    }
}

/// Java `Handle`, which implements `LogFileInterface`.
pub struct Handle {
    /// Java field `logFile`.
    log_file: Arc<LogFile>,
    /// Java field `ownerStackTrace`.
    owner_stack_trace: Mutex<StackTrace>,
    /// Java field `emergencyMonitor`.  Java hands one monitor to several handles
    /// (`doubleBackup` and `buildHandleAndDoFileLock` both pass their handle's), so it is
    /// shared here rather than owned.
    emergency_monitor: Option<Arc<EmergencyMonitor>>,
    /// Java field `backedUpHandle`.
    backed_up_handle: Mutex<bool>,
    /// Java field `standardBarString`, filled in when a process fails.
    standard_bar_string: Mutex<Option<StandardBarString>>,
}

/// The `Debug` of a `Handle` prints only its file: `Id` holds a `Handle`, the `Handle`
/// holds the `LogFile`, and the `LogFile`'s `Lock` holds `Id`s, so a derived `Debug`
/// would not terminate.
impl std::fmt::Debug for Handle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Handle[{}]", self.log_file.file_absolute_path)
    }
}

impl Handle {
    /// Java `Handle(LogFile, boolean, EmergencyMonitor)`.
    fn new(
        log_file: &Arc<LogFile>,
        stress_test_blocking_ids: bool,
        emergency_monitor: Option<Arc<EmergencyMonitor>>,
    ) -> Arc<Handle> {
        let mut emergency_monitor = emergency_monitor;
        if emergency_monitor.is_none() && !stress_test_blocking_ids {
            emergency_monitor = Some(Arc::new(EmergencyMonitor::new(None, None)));
        }
        Arc::new(Handle {
            log_file: log_file.clone(),
            owner_stack_trace: Mutex::new(StackTrace::new_with_title(Some("Handle owner"))),
            emergency_monitor,
            backed_up_handle: Mutex::new(false),
            standard_bar_string: Mutex::new(None),
        })
    }

    /// Java `printOwnerStackTrace()`.
    fn print_owner_stack_trace(&self) {
        self.print_owner_stack_trace_title(None);
    }

    /// Java `printOwnerStackTrace(String)`.
    fn print_owner_stack_trace_title(&self, title: Option<&str>) {
        self.owner_stack_trace.lock().unwrap().print(title, false);
    }

    /// Java `getThreadId`.
    fn get_thread_id(&self) -> String {
        match self.owner_stack_trace.lock().unwrap().get_thread_id() {
            None => String::new(),
            Some(thread_id) => thread_id.to_string(),
        }
    }

    /// Java `setDebug`.
    pub fn set_debug(&self, debug: bool) {
        self.log_file.set_debug(debug);
    }

    /// Java `getWriterIdForTest`.
    pub fn get_writer_id_for_test(self: &Arc<Handle>) -> Option<WriterId> {
        self.log_file.get_writer_id_for_test(self)
    }

    /// Java `getWritingIdForTest`.
    pub fn get_writing_id_for_test(self: &Arc<Handle>) -> Option<WritingId> {
        self.log_file.get_writing_id_for_test(self)
    }

    /// Java `getReaderIdForTest`.
    pub fn get_reader_id_for_test(self: &Arc<Handle>) -> Option<ReaderId> {
        self.log_file.get_reader_id_for_test(self)
    }

    /// Java `getBackupFile`.
    pub fn get_backup_file(&self) -> std::path::PathBuf {
        self.log_file.create_backup_file_var()
    }

    /// Java `selfTestEquals`.  To run a self test of `LogFile.equals` functions, pass a
    /// handle and a boolean based on whether the handle points to the same file or not.
    pub fn self_test_equals(&self, handle: &Arc<Handle>, equals: bool) {
        self.log_file.self_test_equals(handle, equals);
    }

    /// Java `toString(boolean)`.
    pub fn to_string_verbose(&self, verbose: bool) -> String {
        if verbose {
            format!(
                "[{},{}]",
                self.log_file.file_absolute_path,
                java_lang_thread_current_thread_get_id()
            )
        } else {
            self.to_string()
        }
    }

    /// Java `write(Character, WriterId)`.
    pub fn write_character(
        &self,
        ch: Option<char>,
        writer_id: &WriterId,
    ) -> Result<(), LogFileError> {
        self.log_file.write_character(ch, writer_id)
    }

    /// Java `searchForLastLine`.
    pub fn search_for_last_line(
        &self,
        id: &BigBufferReaderId,
        line: &str,
    ) -> Result<bool, LogFileError> {
        self.log_file.search_for_last_line(id, line)
    }

    /// Java `newFileId`.
    fn new_file_id(&self, handle: &Arc<Handle>) -> FileId {
        FileId::new(handle)
    }

    /// Java `openFileLock`.  Opens a fileId lock.  Does not manipulate the file.
    pub fn open_file_lock(
        self: &Arc<Handle>,
        is_blocking_id: bool,
    ) -> Result<FileId, LogFileError> {
        let file_id = FileId::new(self);
        let mut lockable = Lockable::new(
            self,
            Box::new(|timer: &mut SleepTimer| {
                Ok(self.log_file.open_file_lock(&file_id, Some(timer)))
            }),
        );
        lockable.retry_blocking_id(&file_id, is_blocking_id, false)
    }

    /// Java `closeFileLockWithWait`.  Waits for the period specified by the `timeToClose`
    /// parameter, and then unlocks.  Does not block.  The anonymous `Thread`'s `run` body
    /// is the closure handed to `std::thread::spawn` below.
    pub fn close_file_lock_with_wait(
        self: &Arc<Handle>,
        file_id: &FileId,
        time_to_close: i64,
        is_blocking_id: bool,
    ) -> Result<(), LogFileError> {
        let mut lockable = Lockable::new(
            self,
            Box::new(|_timer: &mut SleepTimer| {
                // `new Thread() { public void run() { ... } }.start()`.  The Java body
                // captures `timer`, which `closeFileLockWithWait` never reads.
                let log_file = self.log_file.clone();
                let file_id = file_id.clone();
                std::thread::spawn(move || {
                    log_file.close_file_lock_with_wait(&file_id, time_to_close, None);
                });
                Ok(())
            }),
        );
        lockable.retry_blocking_id(file_id, is_blocking_id, false)
    }

    /// Java `openForReading`.  Functions which cause a lock should only be synchronized
    /// in `Lockable.retry`.
    pub fn open_for_reading(self: &Arc<Handle>) -> Result<Option<ReadingId>, LogFileError> {
        let reading_id = ReadingId::new(self);
        let mut lockable = Lockable::new(
            self,
            Box::new(|timer: &mut SleepTimer| {
                self.log_file.open_for_reading(&reading_id, Some(timer))
            }),
        );
        lockable.retry(&reading_id)
    }

    /// Java `getLineContaining`.
    pub fn get_line_containing(
        self: &Arc<Handle>,
        search_string: &str,
    ) -> Result<Option<String>, LogFileError> {
        let reader_id = ReaderId::new(self);
        let mut lockable = Lockable::new(
            self,
            Box::new(|timer: &mut SleepTimer| {
                self.log_file.get_line_containing_handle(
                    self,
                    &reader_id,
                    search_string,
                    Some(timer),
                )
            }),
        );
        lockable.retry(&reader_id)
    }

    /// Java `openBigBufferReader`.
    pub fn open_big_buffer_reader(
        self: &Arc<Handle>,
    ) -> Result<Option<BigBufferReaderId>, LogFileError> {
        let big_buffer_reader_id = BigBufferReaderId::new(self);
        let mut lockable = Lockable::new(
            self,
            Box::new(|timer: &mut SleepTimer| {
                self.log_file
                    .open_big_buffer_reader(&big_buffer_reader_id, Some(timer))
            }),
        );
        lockable.retry(&big_buffer_reader_id)
    }

    /// Java `isDirectory`.
    pub fn is_directory(&self) -> bool {
        self.log_file.is_directory()
    }

    /// Java `doubleBackupOnce`.  Returns true if backup happened or there was no file to
    /// back up.  False if it can't back up for any reason including one time backup.
    pub fn double_backup_once(self: &Arc<Handle>) -> Result<bool, LogFileError> {
        self.log_file.double_backup_once(self)
    }

    /// Java `isBackedup`.
    pub fn is_backedup(&self) -> bool {
        self.log_file.is_backedup()
    }

    /// Java `openOutputStream`.
    pub fn open_output_stream(self: &Arc<Handle>) -> Result<OutputStreamId, LogFileError> {
        let output_stream_id = OutputStreamId::new(self);
        let mut lockable = Lockable::new(
            self,
            Box::new(|timer: &mut SleepTimer| {
                self.log_file
                    .open_output_stream(&output_stream_id, Some(timer))
            }),
        );
        lockable.retry(&output_stream_id)
    }

    /// Java `store`.
    pub fn store(
        &self,
        properties: &std::collections::BTreeMap<String, String>,
        output_stream_id: &OutputStreamId,
    ) -> Result<(), LogFileError> {
        self.log_file.store(properties, output_stream_id)
    }

    /// Java `load`.
    pub fn load(
        &self,
        properties: &mut std::collections::BTreeMap<String, String>,
        input_stream_id: &InputStreamId,
    ) -> Result<(), LogFileError> {
        self.log_file.load(properties, input_stream_id)
    }

    /// Java `closeId`.
    pub fn close_id(self: &Arc<Handle>, id: Option<&Arc<Id>>) {
        self.log_file.close_id(Some(self), id);
    }

    /// Java `openInputStream`.
    pub fn open_input_stream(self: &Arc<Handle>) -> Result<InputStreamId, LogFileError> {
        let input_stream_id = InputStreamId::new(self);
        let mut lockable = Lockable::new(
            self,
            Box::new(|timer: &mut SleepTimer| {
                self.log_file
                    .open_input_stream(&input_stream_id, Some(timer))
            }),
        );
        lockable.retry(&input_stream_id)
    }

    /// Java `backupOnce`.  Returns true if backup happened or there was no file to back
    /// up.  False if it can't back up for any reason including one time backup.
    pub fn backup_once(self: &Arc<Handle>) -> Result<bool, LogFileError> {
        self.log_file.backup_once(self)
    }

    /// Java `backupOncePerHandle`.
    pub fn backup_once_per_handle(self: &Arc<Handle>) -> Result<bool, LogFileError> {
        let retval = self
            .log_file
            .backup_backed_up(self, *self.backed_up_handle.lock().unwrap())?;
        if retval {
            *self.backed_up_handle.lock().unwrap() = true;
        }
        Ok(retval)
    }

    /// Java `isLocked(Id)`.  Called by unit tests.
    pub fn is_locked_id(&self, id: Option<&Arc<Id>>) -> bool {
        self.log_file.is_locked_id(id)
    }

    /// Java `equals(Object)`, the `instanceof LogFile.Handle` arm.  Need to distinguish
    /// between handles.
    pub fn equals_handle(self: &Arc<Handle>, object: Option<&Arc<Handle>>) -> bool {
        match object {
            None => false,
            Some(object) => Arc::ptr_eq(self, object),
        }
    }

    /// Java `equals(Object)`, the `logFile.equals(object)` fall-through for a `LogFile`.
    pub fn equals_log_file(&self, object: Option<&Arc<LogFile>>) -> bool {
        self.log_file.equals_log_file(object)
    }

    /// Java `equals(Object)`, the `logFile.equals(object)` fall-through for a `Path`.
    pub fn equals_path(&self, object: Option<&std::path::Path>) -> bool {
        self.log_file.equals_path(object)
    }

    /// Java `equals(Object)`, the `logFile.equals(object)` fall-through for a `File`.
    pub fn equals_file(&self, object: Option<&std::path::Path>) -> bool {
        self.log_file.equals_file(object)
    }

    /// Java `equals(Object)`, the `logFile.equals(object)` fall-through for a `String`.
    pub fn equals_string(&self, object: Option<&str>) -> bool {
        self.log_file.equals_string(object)
    }

    /// Java `delete`.  Does a real delete that can't be undone.
    pub fn delete(self: &Arc<Handle>) -> Result<bool, LogFileError> {
        let file_id = FileId::new(self);
        let mut lockable = Lockable::new(
            self,
            Box::new(|timer: &mut SleepTimer| self.log_file.delete_file_id(&file_id, Some(timer))),
        );
        lockable.retry(&file_id)
    }

    /// Java `flush`.
    pub fn flush(&self, writer_id: &WriterId) -> Result<(), LogFileError> {
        self.log_file.flush(writer_id)
    }

    /// Java `isLocked()`.
    pub fn is_locked(&self) -> bool {
        self.log_file.is_locked()
    }

    /// Java `lastModified`.
    pub fn last_modified(&self) -> i64 {
        self.log_file.last_modified()
    }

    /// Java `getName`.
    pub fn get_name(&self) -> String {
        self.log_file.get_name()
    }

    /// Java `openForWriting`.
    pub fn open_for_writing(self: &Arc<Handle>) -> Result<WritingId, LogFileError> {
        let writing_id = WritingId::new(self);
        let mut lockable = Lockable::new(
            self,
            Box::new(|timer: &mut SleepTimer| {
                self.log_file.open_for_writing(&writing_id, Some(timer))
            }),
        );
        lockable.retry(&writing_id)
    }

    /// Java `dumpState`.
    pub fn dump_state(&self) {
        self.log_file.dump_state();
    }

    /// Java `openWriter(boolean)`.
    pub fn open_writer_append(self: &Arc<Handle>, append: bool) -> Result<WriterId, LogFileError> {
        let writer_id = WriterId::new(self);
        let mut lockable = Lockable::new(
            self,
            Box::new(|timer: &mut SleepTimer| {
                self.log_file
                    .open_writer_append(&writer_id, append, Some(timer))
            }),
        );
        lockable.retry(&writer_id)
    }

    /// Java `newLine`.
    pub fn new_line(&self, writer_id: &WriterId) -> Result<(), LogFileError> {
        self.log_file.new_line(writer_id)
    }

    /// Java `create`.
    pub fn create(self: &Arc<Handle>) -> Result<bool, LogFileError> {
        let file_id = FileId::new(self);
        let mut lockable = Lockable::new(
            self,
            Box::new(|timer: &mut SleepTimer| self.log_file.create(&file_id, Some(timer))),
        );
        lockable.retry(&file_id)
    }

    /// Java `exists`.
    pub fn exists(&self) -> bool {
        self.log_file.exists()
    }

    /// Java `openReader()`.
    pub fn open_reader(self: &Arc<Handle>) -> Result<Option<ReaderId>, LogFileError> {
        let reader_id = ReaderId::new(self);
        let mut lockable = Lockable::new(
            self,
            Box::new(|timer: &mut SleepTimer| self.log_file.open_reader(&reader_id, Some(timer))),
        );
        lockable.retry(&reader_id)
    }

    /// Java `openReader(boolean)`.
    pub fn open_reader_required(
        self: &Arc<Handle>,
        required: bool,
    ) -> Result<Option<ReaderId>, LogFileError> {
        let reader_id = ReaderId::new(self);
        let mut lockable = Lockable::new(
            self,
            Box::new(|timer: &mut SleepTimer| {
                self.log_file
                    .open_reader_required(&reader_id, required, Some(timer))
            }),
        );
        lockable.retry(&reader_id)
    }

    /// Java `readLine`.
    pub fn read_line(&self, read_id: &ReaderId) -> Result<Option<String>, LogFileError> {
        self.log_file.read_line(read_id)
    }

    /// Java `backup`.  Returns true if backup happened or there was no file to back up.
    /// False if it can't back up for any reason including one time backup.
    pub fn backup(self: &Arc<Handle>) -> Result<bool, LogFileError> {
        self.log_file.backup(self)
    }

    /// Java `copyToNumberedFile`.
    pub fn copy_to_numbered_file(
        self: &Arc<Handle>,
        manager: Option<&'static dyn BaseManager>,
        extension_marker: Option<ExtensionMarker>,
        num_digits: i32,
    ) -> Result<bool, LogFileError> {
        let copy_from_id = CopyFromId::new(self);
        let mut lockable = Lockable::new(
            self,
            Box::new(|timer: &mut SleepTimer| {
                self.log_file.copy_to_numbered_file(
                    manager,
                    extension_marker,
                    num_digits,
                    &copy_from_id,
                    Some(timer),
                )
            }),
        );
        lockable.retry(&copy_from_id)
    }

    /// Java `copy`.  `preserveCopyToFile`: don't preserve files that signify that a copy
    /// error has happened, unless there's another way to stop the user from continuing.
    pub fn copy(
        self: &Arc<Handle>,
        manager: Option<&'static dyn BaseManager>,
        axis_id: Option<AxisID>,
        copy_to_file: Option<&std::path::Path>,
        preserve_copy_to_file: bool,
        popup_err_msg: bool,
        primary_monitor: bool,
    ) -> Result<bool, LogFileError> {
        let copy_from_id = CopyFromId::new(self);
        let mut lockable = Lockable::new(
            self,
            Box::new(|timer: &mut SleepTimer| {
                self.log_file.copy(
                    manager,
                    axis_id,
                    &copy_from_id,
                    copy_to_file,
                    preserve_copy_to_file,
                    popup_err_msg,
                    Some(timer),
                )
            }),
        );
        lockable.retry_blocking_id(&copy_from_id, false, primary_monitor)
    }

    /// Java `rename`.  `preserveDestFile`: don't preserve files that signify that a
    /// rename error has happened, unless there's another way to stop the user from
    /// continuing.
    pub fn rename(
        self: &Arc<Handle>,
        manager: Option<&'static dyn BaseManager>,
        axis_id: Option<AxisID>,
        dest_file: Option<&std::path::Path>,
        preserve_dest_file: bool,
        popup_err_msg: bool,
        primary_monitor: bool,
    ) -> Result<bool, LogFileError> {
        let file_id = FileId::new(self);
        let mut lockable = Lockable::new(
            self,
            Box::new(|timer: &mut SleepTimer| {
                self.log_file.rename(
                    manager,
                    axis_id,
                    &file_id,
                    dest_file,
                    true,
                    preserve_dest_file,
                    popup_err_msg,
                    Some(timer),
                )
            }),
        );
        lockable.retry_blocking_id(&file_id, false, primary_monitor)
    }

    /// Java `renameSafely`.
    pub fn rename_safely(
        self: &Arc<Handle>,
        dest_file: Option<&std::path::Path>,
    ) -> Result<bool, LogFileError> {
        let file_id = FileId::new(self);
        let mut lockable = Lockable::new(
            self,
            Box::new(|timer: &mut SleepTimer| {
                self.log_file.rename(
                    None,
                    None,
                    &file_id,
                    dest_file,
                    false,
                    true,
                    false,
                    Some(timer),
                )
            }),
        );
        lockable.retry(&file_id)
    }

    /// Java `openWriter()`.
    pub fn open_writer(self: &Arc<Handle>) -> Result<WriterId, LogFileError> {
        let writer_id = WriterId::new(self);
        let mut lockable = Lockable::new(
            self,
            Box::new(|timer: &mut SleepTimer| self.log_file.open_writer(&writer_id, Some(timer))),
        );

        lockable.retry(&writer_id)
    }

    /// Java `write(String, WriterId)`.
    pub fn write(&self, string: Option<&str>, writer_id: &WriterId) -> Result<(), LogFileError> {
        self.log_file.write(string, writer_id)
    }

    /// Java `getAbsolutePath`.
    pub fn get_absolute_path(&self) -> String {
        self.log_file.get_absolute_path()
    }

    /// Java `getFile`.
    pub fn get_file(&self) -> &std::path::Path {
        &self.log_file.file
    }
}

/// Java `toString()` on `Handle`.
impl std::fmt::Display for Handle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}({})",
            self.log_file.file_name,
            java_lang_thread_current_thread_get_id()
        )
    }
}

/// `java.lang.Thread.currentThread().getId()`.  See etomo/util/stack_trace.rs for why
/// this is not the JVM's number.
fn java_lang_thread_current_thread_get_id() -> i64 {
    let debug = format!("{:?}", std::thread::current().id());
    // `ThreadId(3)`
    let digits: String = debug.chars().filter(|c| c.is_ascii_digit()).collect();
    digits.parse().unwrap_or(0)
}
