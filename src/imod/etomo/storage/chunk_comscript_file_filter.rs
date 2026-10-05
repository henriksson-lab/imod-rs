//! `IMOD/Etomo/src/etomo/storage/ChunkComscriptFileFilter.java`.
//!
//! A file filter for parallel processing .com scripts: only the first chunk.

use std::path::Path;
use std::sync::LazyLock;

use regex::Regex;

use crate::imod::etomo::jdk::FileFilter;

/// Java `public static final String rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `"\\S+-0+1.com"` with `String.matches` (whole string).
static FIRST_CHUNK: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^\S+-0+1.com$").unwrap());
/// Java `"\\S+-0+1-sync.com"` with `String.matches` (whole string).
static FIRST_SYNC_CHUNK: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\S+-0+1-sync.com$").unwrap());

/// Java `public class ChunkComscriptFileFilter extends
/// javax.swing.filechooser.FileFilter implements java.io.FileFilter`.
#[derive(Default)]
pub struct ChunkComscriptFileFilter;

impl ChunkComscriptFileFilter {
    /// Java implicit `ChunkComscriptFileFilter()`.
    pub fn new() -> ChunkComscriptFileFilter {
        ChunkComscriptFileFilter
    }
}

impl FileFilter for ChunkComscriptFileFilter {
    /// Java `accept(File)`.  Returns true if a file is a chunk com file.  File must
    /// be in the form:  {rootname}-*.com.  Example: tilta-001.com
    fn accept(&self, file: &Path) -> bool {
        if file.is_dir() {
            return true;
        }
        let name = file
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        // only match -001.com or -001-sync.com. At least one 0 is required. Any
        // number of 0s is valid.
        if !FIRST_CHUNK.is_match(&name) && !FIRST_SYNC_CHUNK.is_match(&name) {
            return false;
        }
        true
    }

    /// Java `getDescription()`.
    fn get_description(&self) -> Option<String> {
        Some("First chunk scripts".to_owned())
    }
}
