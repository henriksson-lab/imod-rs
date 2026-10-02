//! `IMOD/Etomo/src/etomo/storage/SimpleDefocusFile.java`.
//!
//! A simple defocus file (_simple.defocus) that can only contain the expected defocus
//! and one phase plate shift in degrees.  The data from this file is ignored, but it is
//! checked to avoid writing the file unnecessarily.  See the ctfphaseflip man page;
//! Defocus File Format section.
//!
//! The entire file, without the phase plate shift value:
//! `1 1 0 0 expected_defocus_in_nanometers`
//!
//! With the phase plate shift value:
//! `4 0 0. 0. 0 3`
//! `1 1 0 0 expected_defocus_in_nanometers phase_plate_shift_in_degrees`
//!
//! The "4" in the first line is the flag showing the phase plate shifts are included.
//!
//! Copyright: Copyright 2019 - 2025 by the Regents of the University of Colorado
//!
//! Organization: Dept. of MCD Biology, University of Colorado
//!
//! **Shape.**  The package-private overloads take the `LogFileInterface` the public
//! ones build; its only implementation is `LogFile.Handle`, so they take
//! `Option<&Arc<Handle>>`.  The two string parameters are `&str`: every caller passes a
//! string, and the source treats null exactly as it treats an empty string (the
//! `matches("\\s*")` tests), except that `array[PHASE_SHIFT_INDEX].equals(null)` and
//! `.equals("")` are both false for a token of a split line.
// TODO(unit): needs etomo/storage/LogFileInterface.java - the interface the
// package-private overloads take; `LogFile.Handle` is its implementation.
#![allow(dead_code)]

use std::sync::{Arc, LazyLock};

use regex::Regex;

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::storage::log_file::{
    Handle, LogFile, LogFileError, ReaderId, UnlockedException, WriterId,
};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::java_lang_string_matches_whitespace;
use crate::imod::etomo::util::dataset_files;
use crate::imod::etomo::util::stack_trace::StackTrace;
use crate::imod::etomo::util::utilities::java_lang_string_split;

/// Java package-private `FLAG_LINE`.  Only uses flag 4 (000100) phase shifts included.
pub(crate) const FLAG_LINE: &str = "4 0 0. 0. 0 3";
/// Java package-private `DATA_PREFIX`.
pub(crate) const DATA_PREFIX: &str = "1 1 0 0 ";
/// Java private `EXPECTED_DEFOCUS_INDEX`.
const EXPECTED_DEFOCUS_INDEX: usize = 4;
/// Java private `PHASE_SHIFT_INDEX`.
const PHASE_SHIFT_INDEX: usize = 5;

/// Java `"\\s+"`.  Java's `\s` is `[ \t\n\x0B\f\r]`.
static WHITESPACE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[ \t\n\x0B\x0C\r]+").unwrap());

/// Java static `isUpToDate(BaseManager, AxisID, String, String)`.  Returns true if the
/// file exists and matches the parameters.  `expected_defocus_in_nanometers` is
/// required.
pub fn is_up_to_date(
    manager: &'static dyn BaseManager,
    axis_id: AxisID,
    expected_defocus_in_nanometers: &str,
    phase_shift_in_degrees: &str,
) -> bool {
    match LogFile::get_instance_file(
        Some(dataset_files::get_simple_defocus_file(manager, Some(axis_id)).as_path()),
        Some(manager.get_emergency_monitor(Some(axis_id))),
    ) {
        Ok(file) => {
            return is_up_to_date_file(
                Some(manager),
                Some(axis_id),
                expected_defocus_in_nanometers,
                phase_shift_in_degrees,
                Some(&file),
                false,
            );
        }
        // `catch (final LogFile.FileException | IOException e)`
        Err(e) => {
            // `e.printStackTrace()`; see etomo/util/stack_trace.rs.
            eprintln!("{}", e);
        }
    }
    false
}

/// Java package-private static `isUpToDate(BaseManager, AxisID, String, String,
/// LogFileInterface, boolean)`.
pub(crate) fn is_up_to_date_file(
    _manager: Option<&'static dyn BaseManager>,
    _axis_id: Option<AxisID>,
    expected_defocus_in_nanometers: &str,
    phase_shift_in_degrees: &str,
    file: Option<&Arc<Handle>>,
    debug: bool,
) -> bool {
    if debug {
        println!(
            "IsUpToDate0:expectedDefocusInNanometers:{},phaseShiftInDegrees:{}",
            expected_defocus_in_nanometers, phase_shift_in_degrees
        );
    }
    if java_lang_string_matches_whitespace(expected_defocus_in_nanometers) {
        // `Thread.dumpStack()`; see etomo/util/stack_trace.rs.
        StackTrace::new_with_thread(None, None).print(Some("Stack trace"), false);
        eprintln!("Warning: expectedDefocusInNanometers is empty.  Unable to proceed.");
        return false;
    }
    let file = match file {
        Some(file) if file.exists() => file,
        _ => return false,
    };
    let mut up_to_date = false;
    let mut reader_id: Option<ReaderId> = None;
    // The `try` block.  `Ok(Some(value))` is a `return value` from inside the block
    // (which closes the reader itself first), `Ok(None)` falls out of the block.
    let result: Result<Option<bool>, LogFileError> = (|| -> Result<Option<bool>, LogFileError> {
        reader_id = file.open_reader()?;
        // `LogFile.readLine` with a null reader id fails its lock test and throws
        // `UnlockedException`.
        let read_line = |reader_id: &Option<ReaderId>| -> Result<Option<String>, LogFileError> {
            match reader_id {
                Some(reader_id) => file.read_line(reader_id),
                None => Err(LogFileError::Unlocked(UnlockedException::new_id(
                    None, None,
                ))),
            }
        };
        // Look for the flag
        let mut line = read_line(&reader_id)?;
        let mut num_reads = 1;
        if line.is_none() {
            file.close_id(reader_id.as_deref());
            return Ok(Some(false));
        }
        let phase_shift_set = !java_lang_string_matches_whitespace(phase_shift_in_degrees);
        if debug {
            println!(
                "IsUpToDate1:line:{},phaseShiftSet:{}",
                line.as_deref().unwrap(),
                phase_shift_set
            );
        }
        if line.as_deref() == Some(FLAG_LINE) {
            // Found the flag line. Should only be there if the phase plate shift value is
            // available.
            if !phase_shift_set {
                file.close_id(reader_id.as_deref());
                return Ok(Some(false));
            }
            // Get the data line
            line = read_line(&reader_id)?;
            num_reads += 1;
        }
        // Check the data line against parameters
        let line = match line {
            None => {
                file.close_id(reader_id.as_deref());
                return Ok(Some(false));
            }
            Some(line) => line,
        };
        if line.starts_with(DATA_PREFIX) {
            if phase_shift_set && num_reads == 1 {
                // Missing flag line
                file.close_id(reader_id.as_deref());
                return Ok(Some(false));
            }
        } else {
            file.close_id(reader_id.as_deref());
            return Ok(Some(false));
        }
        let array = java_lang_string_split(&line, &WHITESPACE);
        if debug {
            print!("IsUpToDate2:line:{},numReads:{},array:[", line, num_reads);
            if array.len() > 0 {
                print!("{}", array[0]);
            }
            for i in 1..array.len() {
                print!(",{}", array[i]);
            }
            println!("]");
        }
        // Check the required expected defocus
        if array.len() <= EXPECTED_DEFOCUS_INDEX
            || array[EXPECTED_DEFOCUS_INDEX] != expected_defocus_in_nanometers
        {
            file.close_id(reader_id.as_deref());
            return Ok(Some(false));
        }
        // Check the optional phase shift
        if array.len() > PHASE_SHIFT_INDEX && array[PHASE_SHIFT_INDEX] != phase_shift_in_degrees {
            file.close_id(reader_id.as_deref());
            return Ok(Some(false));
        }
        Ok(None)
    })();
    match result {
        Ok(Some(value)) => return value,
        Ok(None) => {
            up_to_date = true;
            if debug {
                println!("IsUpToDate3:upToDate:{}", up_to_date);
            }
        }
        // `catch (FileNotFoundException | LockException e) {}`
        Err(LogFileError::Lock(_)) => {}
        Err(LogFileError::Io(ref e)) if e.kind() == std::io::ErrorKind::NotFound => {}
        // `catch (LogFileException | IOException e)`
        Err(e) => {
            // `e.printStackTrace()`.
            eprintln!("{}", e);
        }
    }
    file.close_id(reader_id.as_deref());
    up_to_date
}

/// Java static `writeFile(BaseManager, AxisID, String, String)`.  Returns true if file
/// was successfully written.  `expected_defocus_in_nanometers` is required.
pub fn write_file(
    manager: &'static dyn BaseManager,
    axis_id: AxisID,
    expected_defocus_in_nanometers: &str,
    phase_shift_in_degrees: &str,
) -> bool {
    match LogFile::get_instance_file(
        Some(dataset_files::get_simple_defocus_file(manager, Some(axis_id)).as_path()),
        Some(manager.get_emergency_monitor(Some(axis_id))),
    ) {
        Ok(file) => {
            return write_file_file(
                Some(manager),
                Some(axis_id),
                expected_defocus_in_nanometers,
                phase_shift_in_degrees,
                Some(&file),
                false,
            );
        }
        // `catch (final LogFile.FileException | IOException e)`
        Err(e) => {
            // `e.printStackTrace()`.
            eprintln!("{}", e);
        }
    }
    false
}

/// Java package-private static `writeFile(BaseManager, AxisID, String, String,
/// LogFileInterface, boolean)`.
pub(crate) fn write_file_file(
    _manager: Option<&'static dyn BaseManager>,
    _axis_id: Option<AxisID>,
    expected_defocus_in_nanometers: &str,
    phase_shift_in_degrees: &str,
    file: Option<&Arc<Handle>>,
    debug: bool,
) -> bool {
    if debug {
        println!(
            "WriteFile0:expectedDefocusInNanometers:{},phaseShiftInDegrees:{}",
            expected_defocus_in_nanometers, phase_shift_in_degrees
        );
    }
    if java_lang_string_matches_whitespace(expected_defocus_in_nanometers) {
        // `Thread.dumpStack()`.
        StackTrace::new_with_thread(None, None).print(Some("Stack trace"), false);
        eprintln!("Warning: expectedDefocusInNanometers is empty.  Unable to write file.");
        return false;
    }
    let file = match file {
        None => return false,
        Some(file) => file,
    };
    let mut succeeded = false;
    let mut writer_id: Option<WriterId> = None;
    // The `try` block.
    let result: Result<(), LogFileError> = (|| -> Result<(), LogFileError> {
        if file.exists() {
            file.backup()?;
        }
        writer_id = Some(file.open_writer()?);
        let phase_shift = !java_lang_string_matches_whitespace(phase_shift_in_degrees);
        if debug {
            println!("WriteFile1:phaseShift:{}", phase_shift);
        }
        if phase_shift {
            file.write(Some(FLAG_LINE), writer_id.as_ref().unwrap())?;
        }
        file.write(
            Some(
                &(DATA_PREFIX.to_string()
                    + expected_defocus_in_nanometers
                    + &(if phase_shift {
                        " ".to_string() + phase_shift_in_degrees
                    } else {
                        String::new()
                    })),
            ),
            writer_id.as_ref().unwrap(),
        )?;
        Ok(())
    })();
    match result {
        Ok(()) => {
            succeeded = true;
        }
        // `catch (final LockException e) {}`
        Err(LogFileError::Lock(_)) => {}
        // `catch (final LogFileException | IOException e)`
        Err(e) => {
            // `e.printStackTrace()`.
            eprintln!("{}", e);
        }
    }
    file.close_id(writer_id.as_deref());
    succeeded
}
