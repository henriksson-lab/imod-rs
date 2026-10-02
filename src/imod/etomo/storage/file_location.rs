//! `IMOD/Etomo/src/etomo/storage/FileLocation.java`.
//!
//! Holds a File instance that exists.  Does a one-time search for the file in
//! a list of paths.  The one static instance is shared by every thread, so
//! the lazily searched state sits behind a `Mutex`.

use std::path::{Path, PathBuf};
use std::sync::{LazyLock, Mutex};

/// Java `rcsid`.
pub const RCSID: &str = "$Id:$";

/// Java `LSOF`.
pub static LSOF: LazyLock<FileLocation> =
    LazyLock::new(|| FileLocation::new("lsof", Some(&["/usr/bin", "/usr/sbin"])));

/// The fields `searchForFile` changes: Java `searched` and `file`.
struct Searched {
    searched: bool,
    file: Option<PathBuf>,
}

/// Java final `FileLocation`.
pub struct FileLocation {
    /// Java private final `fileName`.
    file_name: &'static str,
    /// Java package-private final `pathArray`.
    path_array: Option<&'static [&'static str]>,
    state: Mutex<Searched>,
}

impl FileLocation {
    /// Java private `FileLocation(String, String[])`.
    fn new(file_name: &'static str, path_array: Option<&'static [&'static str]>) -> FileLocation {
        FileLocation {
            file_name,
            path_array,
            state: Mutex::new(Searched {
                searched: false,
                file: None,
            }),
        }
    }

    /// Java `exists`: true if the file exists.
    pub fn exists(&self) -> bool {
        let mut state = self.state.lock().unwrap();
        if !state.searched {
            self.search_for_file(&mut state);
        }
        state.file.is_some()
    }

    /// Java `getAbsolutePath`: the file's absolute path, or null if it
    /// doesn't exist.
    pub fn get_absolute_path(&self) -> Option<String> {
        let mut state = self.state.lock().unwrap();
        if !state.searched {
            self.search_for_file(&mut state);
        }
        let file = state.file.as_ref()?;
        Some(
            crate::imod::etomo::util::utilities::java_io_file_get_absolute_path(
                &file.to_string_lossy(),
            ),
        )
    }

    /// Java private `searchForFile`: search one time for the file in the
    /// paths in pathArray.  If the file isn't found, set the file member
    /// variable to null.  Changes searched to true.
    fn search_for_file(&self, state: &mut Searched) {
        if state.searched {
            return;
        }
        state.searched = true;
        let Some(path_array) = self.path_array else {
            return;
        };
        for path in path_array {
            state.file = Some(Path::new(path).join(self.file_name));
            if state.file.as_ref().is_some_and(|file| file.exists()) {
                return;
            }
        }
        state.file = None;
    }
}
