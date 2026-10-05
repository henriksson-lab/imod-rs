//! `IMOD/Etomo/src/etomo/storage/TestNADFileFilter.java`.
//!
//! Accepts the `nad_eed_3d-NNN.com` test command files the anisotropic diffusion
//! interface writes.

use std::path::Path;

/// Java `public static final String rcsid`.
pub const RCSID: &str = "$Id$";
/// Java `FILE_NAME_BODY`.
pub const FILE_NAME_BODY: &str = "nad_eed_3d-";
/// Java `FILE_NAME_EXT`.
pub const FILE_NAME_EXT: &str = ".com";

/// Java `public final class TestNADFileFilter implements java.io.FileFilter`.
#[derive(Default)]
pub struct TestNADFileFilter;

impl TestNADFileFilter {
    /// Java implicit `TestNADFileFilter()`.
    pub fn new() -> TestNADFileFilter {
        TestNADFileFilter
    }

    /// Java `accept(File)`.
    pub fn accept(&self, file: &Path) -> bool {
        if file.is_dir() {
            return false;
        }
        let file_name = file
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        if file_name.starts_with(FILE_NAME_BODY) && file_name.ends_with(FILE_NAME_EXT) {
            return true;
        }
        false
    }
}
