//! `IMOD/Etomo/src/etomo/process/TomosetextsOutput.java`.
//!
//! Parses the output of `b3dtomosetexts`.  The whole unit is translated; nothing in it
//! reaches an untranslated source unit.
#![allow(dead_code)]

use crate::imod::etomo::r#type::image_filename_style::ImageFilenameStyle;
use crate::imod::etomo::util::utilities::java_lang_string_split;
use regex::Regex;
use std::sync::LazyLock;

/// The literal `"\\s+"` the source hands `String.split`.  Java's `\s` is the five
/// ASCII characters `[ \t\n\x0B\f\r]`, which the Rust regex crate's `\s` widens to
/// Unicode whitespace, so the class is written out.
static WHITESPACE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new("[ \t\n\u{0B}\u{0C}\r]+").unwrap());

/// Java `TomosetextsOutput`.
pub struct TomosetextsOutput {
    /// Java field `output`.
    output: Option<String>,
}

impl TomosetextsOutput {
    /// Java package-private `TomosetextsOutput(String[])`.
    pub fn new(stdout: Option<&[String]>) -> TomosetextsOutput {
        let output = match stdout {
            Some(stdout) if !stdout.is_empty() => Some(stdout[0].clone()),
            _ => None,
        };
        TomosetextsOutput { output }
    }

    /// Java `getImageFilenameStyle`.  Get the image filename style from the dataset.
    pub fn get_image_filename_style(&self) -> Option<ImageFilenameStyle> {
        let output = match &self.output {
            None => return None,
            Some(output) => output,
        };
        let split = java_lang_string_split(output, &WHITESPACE);
        if !split.is_empty() {
            return ImageFilenameStyle::get_instance(&split[0], false);
        }
        None
    }
}

/// Java `toString`.
impl std::fmt::Display for TomosetextsOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[output:{}]",
            match &self.output {
                None => "null".to_string(),
                Some(output) => output.clone(),
            }
        )
    }
}
