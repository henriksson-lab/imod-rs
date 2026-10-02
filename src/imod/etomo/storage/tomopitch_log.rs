//! `IMOD/Etomo/src/etomo/storage/TomopitchLog.java`.
//!
//! Parses the tomopitch log file.
//!
//! **Shape.**  Every getter calls the private `read()` first, which fills the fields
//! once; the getters take `&self` (the dialog reads the log through
//! `TomopitchLogBoundary`, which takes `&self`), so the lazily filled state is in a
//! `RefCell`.  The object is built and read on the event dispatch thread.  Java's
//! getters return the `ConstEtomoNumber` field itself; here they return a copy.
#![allow(dead_code)]

use std::cell::RefCell;
use std::path::PathBuf;
use std::sync::LazyLock;

use regex::Regex;

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::{
    ConstEtomoNumber, Type, java_lang_string_trim,
};
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::util::dataset_files;
use crate::imod::etomo::util::utilities::{java_io_file_new, java_lang_string_split};

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java private `WHITESPACE`, `"\\s+"`.  Java's `\s` is `[ \t\n\x0B\f\r]`.
static WHITESPACE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[ \t\n\x0B\x0C\r]+").unwrap());
/// Java private `ORIGINAL_LABEL_INDEX`.
const ORIGINAL_LABEL_INDEX: usize = 1;
/// Java private `ORIGINAL_LABEL`.
const ORIGINAL_LABEL: &str = "Original:";
/// Java private `ADDED_LABEL_INDEX`.
const ADDED_LABEL_INDEX: usize = 3;
/// Java private `ADDED_LABEL`.
const ADDED_LABEL: &str = "Added:";
/// Java private `TOTAL_LABEL_INDEX`.
const TOTAL_LABEL_INDEX: usize = 5;
/// Java private `TOTAL_LABEL`.
const TOTAL_LABEL: &str = "Total:";
/// Java private `ANGLE_OFFSET_TAG`.
const ANGLE_OFFSET_TAG: &str = "Angle offset";
/// Java private `ANGLE_OFFSET_OFFSET`, `ANGLE_OFFSET_TAG.split(WHITESPACE).length`.
static ANGLE_OFFSET_OFFSET: LazyLock<usize> =
    LazyLock::new(|| java_lang_string_split(ANGLE_OFFSET_TAG, &WHITESPACE).len());
/// Java private `AXIS_Z_SHIFT_TAG`.
const AXIS_Z_SHIFT_TAG: &str = "Z shift";
/// Java private `AXIS_Z_SHIFT_OFFSET`.
static AXIS_Z_SHIFT_OFFSET: LazyLock<usize> =
    LazyLock::new(|| java_lang_string_split(AXIS_Z_SHIFT_TAG, &WHITESPACE).len());
/// Java private `X_AXIS_TILT_TAG`.
const X_AXIS_TILT_TAG: &str = "X axis tilt";
/// Java private `X_AXIS_TILT_OFFSET`.
static X_AXIS_TILT_OFFSET: LazyLock<usize> =
    LazyLock::new(|| java_lang_string_split(X_AXIS_TILT_TAG, &WHITESPACE).len());
/// Java private `NO_X_THICKNESS_TAG`.
const NO_X_THICKNESS_TAG: &str = "rotated";
/// Java private `THICKNESS_TAG`.
const THICKNESS_TAG: &str = "x-tilted";
/// Java private `THICKNESS_LABEL`.
const THICKNESS_LABEL: &str = "to";
/// Java private `THICKNESS_LABEL_INDEX`.
const THICKNESS_LABEL_INDEX: usize = 12;

/// The mutable fields of Java `TomopitchLog`.
struct State {
    /// Java private final field `angleOffsetOriginal`.
    angle_offset_original: EtomoNumber,
    /// Java private final field `angleOffsetAdded`.
    angle_offset_added: EtomoNumber,
    /// Java private final field `angleOffsetTotal`.
    angle_offset_total: EtomoNumber,
    /// Java private final field `axisZShiftOriginal`.
    axis_z_shift_original: EtomoNumber,
    /// Java private final field `axisZShiftAdded`.
    axis_z_shift_added: EtomoNumber,
    /// Java private final field `axisZShiftTotal`.
    axis_z_shift_total: EtomoNumber,
    /// Java private final field `xAxisTiltOriginal`.
    x_axis_tilt_original: EtomoNumber,
    /// Java private final field `xAxisTiltAdded`.
    x_axis_tilt_added: EtomoNumber,
    /// Java private final field `xAxisTiltTotal`.
    x_axis_tilt_total: EtomoNumber,
    /// Java private final field `thickness`.
    thickness: EtomoNumber,
    /// Java private field `logFile`, initialised to null.
    log_file: Option<PathBuf>,
    /// Java private field `containsData`, initialised to false.
    contains_data: bool,
}

/// Java `TomopitchLog`.
pub struct TomopitchLog {
    /// Java private final field `manager`.
    manager: &'static dyn BaseManager,
    /// Java private final field `axisID`.
    axis_id: AxisID,
    state: RefCell<State>,
}

impl TomopitchLog {
    /// Java `TomopitchLog(BaseManager, AxisID)`.
    pub fn new(manager: &'static dyn BaseManager, axis_id: AxisID) -> TomopitchLog {
        TomopitchLog {
            manager,
            axis_id,
            state: RefCell::new(State {
                angle_offset_original: EtomoNumber::new_with_type(Some(Type::Double)),
                angle_offset_added: EtomoNumber::new_with_type(Some(Type::Double)),
                angle_offset_total: EtomoNumber::new_with_type(Some(Type::Double)),
                axis_z_shift_original: EtomoNumber::new_with_type(Some(Type::Double)),
                axis_z_shift_added: EtomoNumber::new_with_type(Some(Type::Double)),
                axis_z_shift_total: EtomoNumber::new_with_type(Some(Type::Double)),
                x_axis_tilt_original: EtomoNumber::new_with_type(Some(Type::Double)),
                x_axis_tilt_added: EtomoNumber::new_with_type(Some(Type::Double)),
                x_axis_tilt_total: EtomoNumber::new_with_type(Some(Type::Double)),
                thickness: EtomoNumber::new(),
                log_file: None,
                contains_data: false,
            }),
        }
    }

    /// Java private `read()`.
    ///
    /// TomopitchLog.java:104-105 and :113-114 index `array[THICKNESS_LABEL_INDEX]` (and
    /// `+ 1`) without checking the length, so a short "x-tilted" or "rotated" line
    /// throws `ArrayIndexOutOfBoundsException` out of every getter.  Fixed in
    /// translation: a line too short to hold the label carries no thickness, and one
    /// with the label but no value sets it to null.
    fn read(&self) {
        let mut state = self.state.borrow_mut();
        if state.log_file.is_some() {
            return;
        }
        // `new File(manager.getPropertyUserDir(), ...)`; a null parent is `new
        // File(child)`.
        let child = dataset_files::get_tomopitch_log_file_name(self.manager, Some(self.axis_id));
        let log_file = match self.manager.get_property_user_dir() {
            None => child,
            Some(property_user_dir) => java_io_file_new(&property_user_dir, &child),
        };
        state.log_file = Some(PathBuf::from(&log_file));
        // `new BufferedReader(new FileReader(logFile))` and `readLine()` until null:
        // lines end at "\n", "\r" or "\r\n".
        let contents = match std::fs::read(&log_file) {
            Ok(contents) => contents,
            // `catch (IOException e) { ... e.printStackTrace(); }`
            Err(e) => {
                eprintln!("java.io.FileNotFoundException: {} ({})", log_file, e);
                return;
            }
        };
        let contents = String::from_utf8_lossy(&contents).replace("\r\n", "\n");
        let mut lines: Vec<&str> = contents.split(['\n', '\r']).collect();
        // The text after the last line terminator is a line only when it is not empty.
        if lines.last().is_some_and(|line| line.is_empty()) {
            lines.pop();
        }
        for line in lines {
            let line = java_lang_string_trim(line);
            if line.starts_with(ANGLE_OFFSET_TAG) {
                let array = java_lang_string_split(line, &WHITESPACE);
                let value = Self::get(
                    &mut state,
                    &array,
                    *ANGLE_OFFSET_OFFSET,
                    ORIGINAL_LABEL_INDEX,
                    ORIGINAL_LABEL,
                );
                state.angle_offset_original.set_string(value.as_deref());
                let value = Self::get(
                    &mut state,
                    &array,
                    *ANGLE_OFFSET_OFFSET,
                    ADDED_LABEL_INDEX,
                    ADDED_LABEL,
                );
                state.angle_offset_added.set_string(value.as_deref());
                let value = Self::get(
                    &mut state,
                    &array,
                    *ANGLE_OFFSET_OFFSET,
                    TOTAL_LABEL_INDEX,
                    TOTAL_LABEL,
                );
                state.angle_offset_total.set_string(value.as_deref());
            } else if line.starts_with(AXIS_Z_SHIFT_TAG) {
                let array = java_lang_string_split(line, &WHITESPACE);
                let value = Self::get(
                    &mut state,
                    &array,
                    *AXIS_Z_SHIFT_OFFSET,
                    ORIGINAL_LABEL_INDEX,
                    ORIGINAL_LABEL,
                );
                state.axis_z_shift_original.set_string(value.as_deref());
                let value = Self::get(
                    &mut state,
                    &array,
                    *AXIS_Z_SHIFT_OFFSET,
                    ADDED_LABEL_INDEX,
                    ADDED_LABEL,
                );
                state.axis_z_shift_added.set_string(value.as_deref());
                let value = Self::get(
                    &mut state,
                    &array,
                    *AXIS_Z_SHIFT_OFFSET,
                    TOTAL_LABEL_INDEX,
                    TOTAL_LABEL,
                );
                state.axis_z_shift_total.set_string(value.as_deref());
            } else if line.starts_with(X_AXIS_TILT_TAG) {
                let array = java_lang_string_split(line, &WHITESPACE);
                let value = Self::get(
                    &mut state,
                    &array,
                    *X_AXIS_TILT_OFFSET,
                    ORIGINAL_LABEL_INDEX,
                    ORIGINAL_LABEL,
                );
                state.x_axis_tilt_original.set_string(value.as_deref());
                let value = Self::get(
                    &mut state,
                    &array,
                    *X_AXIS_TILT_OFFSET,
                    ADDED_LABEL_INDEX,
                    ADDED_LABEL,
                );
                state.x_axis_tilt_added.set_string(value.as_deref());
                let value = Self::get(
                    &mut state,
                    &array,
                    *X_AXIS_TILT_OFFSET,
                    TOTAL_LABEL_INDEX,
                    TOTAL_LABEL,
                );
                state.x_axis_tilt_total.set_string(value.as_deref());
            } else if line.starts_with(THICKNESS_TAG) {
                let array = java_lang_string_split(line, &WHITESPACE);
                if array.get(THICKNESS_LABEL_INDEX).map(String::as_str) == Some(THICKNESS_LABEL) {
                    state.contains_data = true;
                    let value = array.get(THICKNESS_LABEL_INDEX + 1).cloned();
                    state.thickness.set_string(value.as_deref());
                }
            }
            // DNM 6/27/24: get thickness from the "rotated" lines in case no X tilt
            else if line.starts_with(NO_X_THICKNESS_TAG) {
                let array = java_lang_string_split(line, &WHITESPACE);
                if array.get(THICKNESS_LABEL_INDEX).map(String::as_str) == Some(THICKNESS_LABEL) {
                    state.contains_data = true;
                    let value = array.get(THICKNESS_LABEL_INDEX + 1).cloned();
                    state.thickness.set_string(value.as_deref());
                }
            }
        }
    }

    /// Java private `get(String[], int, int, String)`.
    ///
    /// TomopitchLog.java:141-142 indexes `array[index]` and `array[index + 1]` without
    /// checking the length (`ArrayIndexOutOfBoundsException` on a short line).  Fixed
    /// in translation: a missing label finds nothing, and a label with no value
    /// returns null.
    fn get(
        state: &mut State,
        array: &[String],
        offset: usize,
        label_index: usize,
        label: &str,
    ) -> Option<String> {
        let index = offset + label_index;
        if array.get(index).map(String::as_str) == Some(label) {
            state.contains_data = true;
            return array.get(index + 1).cloned();
        }
        None
    }

    /// Java `containsData()`.
    pub fn contains_data(&self) -> bool {
        self.read();
        self.state.borrow().contains_data
    }

    /// Java `getAngleOffsetAdded()`.
    pub fn get_angle_offset_added(&self) -> ConstEtomoNumber {
        self.read();
        self.state.borrow().angle_offset_added.base.clone()
    }

    /// Java `getAngleOffsetOriginal()`.
    pub fn get_angle_offset_original(&self) -> ConstEtomoNumber {
        self.read();
        self.state.borrow().angle_offset_original.base.clone()
    }

    /// Java `getAngleOffsetTotal()`.
    pub fn get_angle_offset_total(&self) -> ConstEtomoNumber {
        self.read();
        self.state.borrow().angle_offset_total.base.clone()
    }

    /// Java `getAxisZShiftAdded()`.
    pub fn get_axis_z_shift_added(&self) -> ConstEtomoNumber {
        self.read();
        self.state.borrow().axis_z_shift_added.base.clone()
    }

    /// Java `getAxisZShiftOriginal()`.
    pub fn get_axis_z_shift_original(&self) -> ConstEtomoNumber {
        self.read();
        self.state.borrow().axis_z_shift_original.base.clone()
    }

    /// Java `getAxisZShiftTotal()`.
    pub fn get_axis_z_shift_total(&self) -> ConstEtomoNumber {
        self.read();
        self.state.borrow().axis_z_shift_total.base.clone()
    }

    /// Java `getXAxisTiltAdded()`.
    pub fn get_x_axis_tilt_added(&self) -> ConstEtomoNumber {
        self.read();
        self.state.borrow().x_axis_tilt_added.base.clone()
    }

    /// Java `getXAxisTiltOriginal()`.
    pub fn get_x_axis_tilt_original(&self) -> ConstEtomoNumber {
        self.read();
        self.state.borrow().x_axis_tilt_original.base.clone()
    }

    /// Java `getXAxisTiltTotal()`.
    pub fn get_x_axis_tilt_total(&self) -> ConstEtomoNumber {
        self.read();
        self.state.borrow().x_axis_tilt_total.base.clone()
    }

    /// Java `getThickness()`.
    pub fn get_thickness(&self) -> ConstEtomoNumber {
        self.read();
        self.state.borrow().thickness.base.clone()
    }
}

