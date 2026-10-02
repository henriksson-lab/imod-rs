//! `IMOD/Etomo/src/etomo/util/FidXyz.java`.
//!
//! An interface to the fid.xyz file.
//!
//! **Exceptions.**  `read()` declares `IOException`, which is the `Err` here.  It can
//! also let two unchecked exceptions escape from `setPixelSize`: the
//! `IllegalStateException` for a bad fid.xyz format and a `NumberFormatException` from
//! `Double.parseDouble`.  Those would propagate through every caller (none catches
//! them) and abort the operation; here they are returned as `Err` with the exception's
//! text, so the caller's `IOException` handling reports them instead (fixed in
//! translation, a crash).
#![allow(dead_code)]

use std::sync::LazyLock;

use regex::Regex;

use crate::imod::etomo::r#type::const_etomo_number::java_lang_double_value_of;
use crate::imod::etomo::util::utilities::{
    self, FAILED_STATUS, FINISHED_STATUS, STARTED_STATUS, java_io_file_new, java_lang_string_split,
};

/// Java `rcsid`.
pub const RCSID: &str = "$$Id$$";

/// Java private `OLD_PIXEL_SIZE_LABEL`.
const OLD_PIXEL_SIZE_LABEL: &str = "pixel size:";
/// Java private `NEW_PIXEL_SIZE_LABEL`.
const NEW_PIXEL_SIZE_LABEL: &str = "pix:";
/// Java private `OLD_PIXEL_SIZE_INDEX`.
const OLD_PIXEL_SIZE_INDEX: usize = 9;
/// Java private `NEW_PIXEL_SIZE_INDEX`.
const NEW_PIXEL_SIZE_INDEX: usize = 7;

/// Java `"\\s+"`, the `split` regex in `setPixelSize`.  Java's `\s` is
/// `[ \t\n\x0B\f\r]`.
static WHITESPACE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[ \t\n\x0B\x0C\r]+").unwrap());

/// Java `FidXyz`.
#[derive(Clone, Debug)]
pub struct FidXyz {
    /// Java private field `filename`.
    filename: Option<String>,
    /// Java private field `exists`, initialised to false.
    exists: bool,
    /// Java private field `empty`, initialised to false.
    empty: bool,
    /// Java private field `pixelSize`, initialised to `Double.NaN`.
    pixel_size: f64,
    /// Java private final field `propertyUserDir`.
    property_user_dir: Option<String>,
}

impl FidXyz {
    /// Java `FidXyz(String, String)`.
    pub fn new(property_user_dir: Option<&str>, name: &str) -> FidXyz {
        FidXyz {
            filename: Some(name.to_string()),
            exists: false,
            empty: false,
            pixel_size: f64::NAN,
            property_user_dir: property_user_dir.map(|dir| dir.to_string()),
        }
    }

    /// Java `read()`.
    pub fn read(&mut self) -> Result<(), String> {
        let filename = match &self.filename {
            Some(filename) if !filename.is_empty() => filename.clone(),
            _ => return Err("No filename specified".to_string()),
        };
        utilities::timestamp_process_container_status(
            Some("read"),
            Some(&filename),
            Some(STARTED_STATUS),
        );
        // `new File(propertyUserDir, filename)`; a null parent is `new File(filename)`.
        let fid_xyz_file = match &self.property_user_dir {
            None => filename.clone(),
            Some(property_user_dir) => java_io_file_new(property_user_dir, &filename),
        };
        let path = std::path::Path::new(&fid_xyz_file);
        if !path.exists() || path.is_dir() {
            utilities::timestamp_process_container_status(
                Some("read"),
                Some(&filename),
                Some(FAILED_STATUS),
            );
            return Ok(());
        }
        self.exists = true;
        // `File.length()` is 0 when it cannot be read.
        if std::fs::metadata(path)
            .map(|metadata| metadata.len())
            .unwrap_or(0)
            == 0
        {
            self.empty = true;
            utilities::timestamp_process_container_status(
                Some("read"),
                Some(&filename),
                Some(FAILED_STATUS),
            );
            return Ok(());
        }
        // `new BufferedReader(new FileReader(fidXyzFile)).readLine()`: the first line,
        // ended by "\n", "\r" or "\r\n"; null for an empty file (not reached here).
        let contents = std::fs::read(path).map_err(|e| e.to_string())?;
        let contents = String::from_utf8_lossy(&contents);
        let line: Option<String> = if contents.is_empty() {
            None
        } else {
            Some(
                contents
                    .split(['\n', '\r'])
                    .next()
                    .unwrap_or_default()
                    .to_string(),
            )
        };
        // The first line contains the pixel size
        if !self.set_pixel_size(line.as_deref(), NEW_PIXEL_SIZE_LABEL, NEW_PIXEL_SIZE_INDEX)? {
            if !self.set_pixel_size(line.as_deref(), OLD_PIXEL_SIZE_LABEL, OLD_PIXEL_SIZE_INDEX)? {
                utilities::timestamp_process_container_status(
                    Some("read"),
                    Some(&filename),
                    Some(FAILED_STATUS),
                );
            }
        }
        utilities::timestamp_process_container_status(
            Some("read"),
            Some(&filename),
            Some(FINISHED_STATUS),
        );
        Ok(())
    }

    /// Java private `setPixelSize(String, String, int)`.  Handle format change in
    /// fid.xyz.  `Err` carries an unchecked exception's text; see the module header.
    ///
    /// FidXyz.java:128-130 builds the `IllegalStateException` message from
    /// `tokens[pixelSizeIndex]`, which is out of bounds exactly when the "too few
    /// tokens" half of the test is what failed, so an
    /// `ArrayIndexOutOfBoundsException` replaces the intended exception.  Fixed in
    /// translation: a missing token prints as "null".
    fn set_pixel_size(
        &mut self,
        line: Option<&str>,
        pixel_size_label: &str,
        pixel_size_index: usize,
    ) -> Result<bool, String> {
        let line = match line {
            None => return Ok(false),
            Some(line) => line,
        };
        let mut line = line.to_lowercase();
        if pixel_size_label == NEW_PIXEL_SIZE_LABEL {
            line = crate::imod::etomo::r#type::const_etomo_number::java_lang_string_trim(&line)
                .to_string();
        }
        if line.contains(pixel_size_label) {
            let tokens = java_lang_string_split(&line, &WHITESPACE);
            if tokens.len() < pixel_size_index + 1
                || tokens[pixel_size_index - 1] != pixel_size_label
            {
                utilities::timestamp_process_container_status(
                    Some("read"),
                    self.filename.as_deref(),
                    Some(FAILED_STATUS),
                );
                return Err(format!(
                    "java.lang.IllegalStateException: bad fid.xyz format: ,\ntokens[{}]={},pixelSizeLabel={},line={}",
                    pixel_size_index,
                    tokens
                        .get(pixel_size_index)
                        .map(String::as_str)
                        .unwrap_or("null"),
                    pixel_size_label,
                    line
                ));
            }
            self.pixel_size = java_lang_double_value_of(&tokens[pixel_size_index])
                .map_err(|e| format!("java.lang.NumberFormatException: {}", e))?;
            return Ok(true);
        }
        Ok(false)
    }

    /// Java `exists()`.
    pub fn exists(&self) -> bool {
        self.exists
    }

    /// Java `isEmpty()`.  Returns true if zero length file.
    pub fn is_empty(&self) -> bool {
        self.empty
    }

    /// Java `isPixelSizeSet()`.
    pub fn is_pixel_size_set(&self) -> bool {
        !self.pixel_size.is_nan()
    }

    /// Java `getPixelSize()`.
    pub fn get_pixel_size(&self) -> f64 {
        self.pixel_size
    }
}
