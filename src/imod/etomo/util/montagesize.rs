//! `IMOD/Etomo/src/etomo/util/Montagesize.java`.
//!
//! Description: Runs montagesize on .st files.  Creates once instance per file.
//! Rereads only when file has changed.
//!
//! Copyright: Copyright 2005 - 2015 by the Regents of the University of Colorado
//!
//! Organization: Dept. of MCD Biology, University of Colorado
//!
//! **Shape.**  The n'ton table hands one shared instance to every caller (parameter
//! classes on the event dispatch thread, `BlendmontProcessMonitor` on its own thread),
//! and the Java methods that mutate it are `synchronized`.  The instances are therefore
//! `Arc<Montagesize>` with the mutable fields behind one `Mutex`, and every method
//! takes `&self`.  `read`'s `IOException`, `InvalidParameterException` and
//! `NumberFormatException` all arrive as their message (`Err(String)`), as
//! `MRCHeader.read`'s do.

use crate::imod::etomo::base_manager::{self, BaseManager};
use crate::imod::etomo::process::process_messages::MessageType;
use crate::imod::etomo::process::system_program::SystemProgram;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::{
    ConstEtomoNumber, Type, java_lang_string_matches_whitespace, java_lang_string_trim,
};
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::r#type::file_type::FileType;
use crate::imod::etomo::util::file_modified_flag::FileModifiedFlag;
use crate::imod::etomo::util::utilities::{
    self, java_io_file_get_absolute_path, java_lang_string_split,
};
use regex::Regex;
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, LazyLock, Mutex};

/// Java private `EXT`.
const EXT: &str = ".pl";

/// Java `"\\s+"`, with Java's `\s` class.
static WHITESPACE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new("[ \t\n\u{0B}\u{0C}\r]+").unwrap());

// n'ton member variables

/// Java private static `instances`, a `Hashtable` keyed by absolute path.
static INSTANCES: LazyLock<Mutex<HashMap<String, Arc<Montagesize>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
/// Java's `synchronized createInstance` lock, which is on the class.
static CREATE_INSTANCE_LOCK: Mutex<()> = Mutex::new(());

/// The fields the source mutates, guarded together as the `synchronized` methods do.
struct State {
    // member variables to prevent unnecessary reads
    /// Java private field `modifiedFlag`.
    modified_flag: FileModifiedFlag,
    // other member variables
    /// Java private field `x`.
    x: EtomoNumber,
    /// Java private field `y`.
    y: EtomoNumber,
    /// Java private field `z`.
    z: EtomoNumber,
    /// Java private field `fileExists`, initialised to false.
    file_exists: bool,
    /// Java package-private field `commandArray`, initialised to null.
    command_array: Option<Vec<String>>,
    /// Java private field `ignorePieceListFile`, initialised to false.
    ignore_piece_list_file: bool,
    /// Java private field `exitValue`, initialised to -1.
    exit_value: i32,
}

/// Java `Montagesize`.
pub struct Montagesize {
    /// Java private final field `file`, a `java.io.File` (a path holder).
    file: String,
    /// Java private final field `propertyUserDir`.
    property_user_dir: Option<String>,
    /// Java private final field `axisID`.
    axis_id: AxisID,
    /// The mutable fields.
    state: Mutex<State>,
}

impl Montagesize {
    // n'ton functions

    /// Java private constructor `Montagesize(String, File, AxisID)`.
    fn new(property_user_dir: Option<String>, file: &str, axis_id: AxisID) -> Montagesize {
        Montagesize {
            file: file.to_string(),
            property_user_dir,
            axis_id,
            state: Mutex::new(State {
                modified_flag: FileModifiedFlag::new(file),
                x: EtomoNumber::new_with_type(Some(Type::Integer)),
                y: EtomoNumber::new_with_type(Some(Type::Integer)),
                z: EtomoNumber::new_with_type(Some(Type::Integer)),
                file_exists: false,
                command_array: None,
                ignore_piece_list_file: false,
                exit_value: -1,
            }),
        }
    }

    /// Java static `getInstance(BaseManager, AxisID, FileType, boolean)`.  Function to
    /// get an instance of the class.
    ///
    /// `fileType.getFile` returns null when the file type cannot name a file, and
    /// Montagesize.java:80 then dereferences it in `makeKey`.  Fixed in translation:
    /// no instance (`None`) is returned.
    pub fn get_instance(
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        file_type: &Arc<FileType>,
        debug: bool,
    ) -> Option<Arc<Montagesize>> {
        let key_file = file_type.get_file(Some(manager), Some(axis_id))?;
        let key_file = key_file.to_string_lossy().to_string();
        let key = Montagesize::make_key(&key_file);
        let montagesize = INSTANCES.lock().unwrap().get(&key).cloned();
        match montagesize {
            None => Some(Montagesize::create_instance(
                manager.get_property_user_dir(),
                &key,
                &key_file,
                axis_id,
                debug,
            )),
            Some(montagesize) => {
                montagesize.set_to_defaults();
                Some(montagesize)
            }
        }
    }

    /// Java static `getInstance(String, String, AxisID)`.  Function to get an instance
    /// of the class.
    pub fn get_instance_in_dir(
        file_location: &str,
        filename: Option<&str>,
        axis_id: AxisID,
    ) -> Arc<Montagesize> {
        let key_file = utilities::get_file(file_location, filename)
            .to_string_lossy()
            .to_string();
        let key = Montagesize::make_key(&key_file);
        let montagesize = INSTANCES.lock().unwrap().get(&key).cloned();
        match montagesize {
            None => Montagesize::create_instance(
                Some(file_location.to_string()),
                &key,
                &key_file,
                axis_id,
                false,
            ),
            Some(montagesize) => {
                montagesize.set_to_defaults();
                montagesize
            }
        }
    }

    /// Java private static synchronized `createInstance`.  Function to create and save
    /// an instance of the class.  Just returns the instance if it already exists.
    fn create_instance(
        property_user_dir: Option<String>,
        key: &str,
        file: &str,
        axis_id: AxisID,
        _debug: bool,
    ) -> Arc<Montagesize> {
        let _lock = CREATE_INSTANCE_LOCK.lock().unwrap();
        if let Some(montagesize) = INSTANCES.lock().unwrap().get(key) {
            return Arc::clone(montagesize);
        }
        let montagesize = Arc::new(Montagesize::new(property_user_dir, file, axis_id));
        INSTANCES
            .lock()
            .unwrap()
            .insert(key.to_string(), Arc::clone(&montagesize));
        montagesize.self_test_invariants();
        montagesize
    }

    /// Java private static `makeKey(File)`.  Make a unique key from a file.
    fn make_key(file: &str) -> String {
        java_io_file_get_absolute_path(file)
    }

    /// Java `getFile`.
    pub fn get_file(&self) -> String {
        java_io_file_get_absolute_path(&self.file)
    }

    // other functions

    /// Java private `makePieceListFile`.  Construct a piece list file.
    fn make_piece_list_file(&self) -> String {
        let file_path = java_io_file_get_absolute_path(&self.file);
        let extension_index = file_path.rfind('.');
        match extension_index {
            None | Some(0) => format!("{}{}", file_path, EXT),
            Some(extension_index) => format!("{}{}", &file_path[..extension_index], EXT),
        }
    }

    /// Java `pieceListFileExists`.
    pub fn piece_list_file_exists(&self) -> bool {
        Path::new(&self.make_piece_list_file()).exists()
    }

    /// Java private `reset`.  Reset results.
    fn reset(state: &mut State) {
        state.file_exists = false;
        state.x.reset();
        state.y.reset();
        state.z.reset();
        state.command_array = None;
        state.exit_value = -1;
    }

    /// Java private final `buildCommand`.
    fn build_command(&self, state: &mut State) {
        if state.command_array.is_some() {
            return;
        }
        let piece_list_file = self.make_piece_list_file();
        let piece_list_file_exists = Path::new(&piece_list_file).exists();
        let mut command_array: Vec<String> = Vec::new();
        command_array.push(format!(
            "{}montagesize",
            base_manager::get_imod_bin_path().unwrap_or_else(|| "null".to_string())
        ));
        command_array.push(java_io_file_get_absolute_path(&self.file));
        if piece_list_file_exists && !state.ignore_piece_list_file {
            // bug# 1336
            command_array.push(java_io_file_get_absolute_path(&piece_list_file));
        }
        state.command_array = Some(command_array);
    }

    /// Java private `setToDefaults`.
    fn set_to_defaults(&self) {
        self.set_ignore_piece_list_file(false);
    }

    /// Java synchronized `setIgnorePieceListFile(boolean)`.
    pub fn set_ignore_piece_list_file(&self, input: bool) {
        let mut state = self.state.lock().unwrap();
        if state.ignore_piece_list_file != input {
            state.modified_flag.reset();
            Montagesize::reset(&mut state);
        }
        state.ignore_piece_list_file = input;
    }

    /// Java `getExitValue`.
    pub fn get_exit_value(&self) -> i32 {
        self.state.lock().unwrap().exit_value
    }

    /// Java synchronized `read(BaseManager)`.  Run montagesize on the file.  Returns
    /// true if attempted to read.
    pub fn read(&self, manager: &'static dyn BaseManager) -> Result<bool, String> {
        let mut state = self.state.lock().unwrap();
        let path = Path::new(&self.file);
        if path.is_dir() {
            return Err(format!("{}is not a file.", self.file));
        }
        if !path.exists() {
            Montagesize::reset(&mut state);
            return Ok(false);
        }
        state.file_exists = true;
        // If the file hasn't been modified, don't reread
        if !state.modified_flag.is_modified_since_last_read() {
            return Ok(false);
        }
        let failed = || {
            utilities::timestamp_file(
                Some("read"),
                Some("montagesize"),
                Some(path),
                Some(utilities::FAILED_STATUS),
            );
        };
        // put first timestamp after decide to read
        utilities::timestamp_file(
            Some("read"),
            Some("montagesize"),
            Some(path),
            Some(utilities::STARTED_STATUS),
        );
        // Run the montagesize command on the file.
        self.build_command(&mut state);
        let montagesize = SystemProgram::new_array(
            Some(manager),
            self.property_user_dir.clone(),
            state.command_array.clone(),
            self.axis_id,
        );
        state.modified_flag.set_reading_now();
        montagesize.run();

        state.exit_value = montagesize.get_exit_value();
        let absolute_path = java_io_file_get_absolute_path(&self.file);
        if state.exit_value != 0 {
            let std_output = montagesize.get_std_output();
            if let Some(std_output) = &std_output
                && !std_output.is_empty()
            {
                let messages = montagesize.get_process_messages();
                if messages.size(MessageType::Error) > 0 {
                    let mut message = format!(
                        "montagesize returned an error while reading{}:\n",
                        absolute_path
                    );
                    for i in 0..messages.size(MessageType::Error) {
                        message =
                            message + messages.get(MessageType::Error, i).unwrap_or("null") + "\n";
                    }
                    failed();
                    return Err(message);
                }
            }
        }
        // Throw an exception if the file can not be read
        let std_error = montagesize.get_std_error();
        if let Some(std_error) = &std_error
            && !std_error.is_empty()
        {
            let mut message = format!(
                "montagesize returned an error while reading{}:\n",
                absolute_path
            );
            for i in 0..std_error.len() {
                message = message + &std_error[i] + "\n";
            }
            failed();
            return Err(message);
        }

        // Parse the output
        let std_output = montagesize.get_std_output();
        let std_output = match std_output {
            Some(std_output) if !std_output.is_empty() => std_output,
            _ => {
                failed();
                return Err(format!(
                    "montagesize returned no data while reading{}",
                    absolute_path
                ));
            }
        };

        for i in 0..std_output.len() {
            // Parse the size of the data
            // Note the initial space in the string below
            let output_line = java_lang_string_trim(&std_output[i]);
            if output_line.starts_with("Total NX, NY, NZ:") {
                let tokens = java_lang_string_split(output_line, &WHITESPACE);
                if tokens.len() < 7 {
                    failed();
                    return Err(format!(
                        "Montagesize returned less than three parameters for image size while reading{}",
                        absolute_path
                    ));
                }
                state.x.set_string(Some(&tokens[4]));
                state.y.set_string(Some(&tokens[5]));
                state.z.set_string(Some(&tokens[6]));
                if !state.x.is_valid() || state.x.is_null() {
                    failed();
                    return Err(format!(
                        "NX is not set, token is {}\n{}",
                        tokens[4],
                        state.x.get_invalid_reason()
                    ));
                }
                if !state.y.is_valid() || state.y.is_null() {
                    failed();
                    return Err(format!(
                        "NY is not set, token is {}\n{}",
                        tokens[5],
                        state.y.get_invalid_reason()
                    ));
                }
                if !state.z.is_valid() || state.z.is_null() {
                    failed();
                    return Err(format!(
                        "NZ is not set, token is {}\n{}",
                        tokens[6],
                        state.z.get_invalid_reason()
                    ));
                }
            }
        }
        utilities::timestamp_file(
            Some("read"),
            Some("montagesize"),
            Some(path),
            Some(utilities::FINISHED_STATUS),
        );
        Ok(true)
    }

    /// Java `getX`.
    pub fn get_x(&self) -> ConstEtomoNumber {
        self.state.lock().unwrap().x.base.clone()
    }

    /// Java `getY`.
    pub fn get_y(&self) -> ConstEtomoNumber {
        self.state.lock().unwrap().y.base.clone()
    }

    /// Java `getZ`.
    pub fn get_z(&self) -> ConstEtomoNumber {
        self.state.lock().unwrap().z.base.clone()
    }

    /// Java `isFileExists`.
    pub fn is_file_exists(&self) -> bool {
        self.state.lock().unwrap().file_exists
    }

    // self test functions

    /// Java package-private `selfTestInvariants`.
    fn self_test_invariants(&self) {
        if !utilities::is_self_test() {
            return;
        }
        if Path::new(&self.file).is_dir() {
            panic!("java.lang.IllegalStateException: file is null");
        }
        let key = Montagesize::make_key(&self.file);
        if java_lang_string_matches_whitespace(&key) {
            panic!(
                "java.lang.IllegalStateException: unable to make key: filename={}",
                java_io_file_get_absolute_path(&self.file)
            );
        }
        if !INSTANCES.lock().unwrap().contains_key(&key) {
            panic!(
                "java.lang.IllegalStateException: this instance is not in instances: key={}",
                key
            );
        }
    }

    /// Java package-private `paramString`.
    fn param_string(&self) -> String {
        let state = self.state.lock().unwrap();
        format!(
            ",file={},fileExists={},x={},y={},z={},axisID={}",
            self.file, state.file_exists, state.x, state.y, state.z, self.axis_id
        )
    }
}

/// Java `toString`: `getClass().getName() + "[" + super.toString() + paramString() +
/// "]"`.  `Object.toString`'s identity hash is a JVM value; the instance address stands
/// in for it.
impl std::fmt::Display for Montagesize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "etomo.util.Montagesize[etomo.util.Montagesize@{:x}{}]",
            self as *const Montagesize as usize,
            self.param_string()
        )
    }
}
