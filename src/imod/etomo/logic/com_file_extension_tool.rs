//! `IMOD/Etomo/src/etomo/logic/ComFileExtensionTool.java`.
//!
//! The plan is to replace the current comfile extension (.com) with something less
//! problematic.  Use this library to generically work with the comfile extension.  Will
//! be modified to handle changes, including a transition period.

use crate::imod::etomo::util::utilities::java_lang_string_last_index_of_from;

/// Java private `EXTENSION`.
const EXTENSION: &str = ".com";

/// Java final `ComFileExtensionTool` (private constructor; static members only).
pub struct ComFileExtensionTool;

impl ComFileExtensionTool {
    /// Java static `lastIndexOf(String, int)`.
    pub fn last_index_of(str: Option<&str>, from_index: i32) -> i32 {
        let str = match str {
            None => return -1,
            Some(str) => str,
        };
        java_lang_string_last_index_of_from(str, EXTENSION, from_index as i64) as i32
    }
}
