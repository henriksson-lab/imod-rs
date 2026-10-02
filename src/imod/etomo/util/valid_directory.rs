//! `IMOD/Etomo/src/etomo/util/ValidDirectory.java`.
//!
//! Takes a file or a string.  Always contains either null or a directory that is
//! readable and exists.  Will use the parent directory if it is given a file.  See
//! `ui/browsing_directory.rs`.
//!
//! A plain value (a manager keeps one behind its own lock), so the mutating methods
//! take `&mut self`; it is `Send + Sync`.  `java.io.File` is a `PathBuf` holding the
//! File's path string.

use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::r#type::const_etomo_number::java_lang_string_matches_whitespace;
use crate::imod::etomo::ui::browsing_directory::BrowsingDirectory;
use crate::imod::etomo::util::utilities::{
    java_io_file_can_read, java_io_file_get_absolute_path, java_io_file_get_parent,
};

/// Java private static final `DEBUG`.
static DEBUG: LazyLock<bool> =
    LazyLock::new(|| etomo_director::ARGUMENTS.lock().unwrap().is_debug());

/// Java `ValidDirectory`.
pub struct ValidDirectory {
    /// Java private final `manager`.
    manager: Option<&'static dyn BaseManager>,
    /// Java private `dir`, initialised to null.
    dir: Option<PathBuf>,
    /// Java package-private `allowGetParent`, initialised to true.
    pub allow_get_parent: bool,
}

impl ValidDirectory {
    /// Java `ValidDirectory(BaseManager)`.
    pub fn new(manager: Option<&'static dyn BaseManager>) -> ValidDirectory {
        ValidDirectory {
            manager,
            dir: None,
            allow_get_parent: true,
        }
    }

    /// Java `get()`.
    pub fn get_void(&self) -> Option<PathBuf> {
        self.dir.clone()
    }

    /// Java `getParent()`.
    pub fn get_parent(&self) -> Option<PathBuf> {
        if !self.allow_get_parent {
            return self.get_void();
        }
        let dir = self.dir.as_ref()?;
        let parent = java_io_file_get_parent(&dir.to_string_lossy());
        let Some(parent) = parent else {
            return Some(dir.clone());
        };
        Some(PathBuf::from(parent))
    }

    /// Java `isNull()`.
    pub fn is_null(&self) -> bool {
        self.dir.is_none()
    }

    /// Java `setToPropertyUserDir()`.
    pub fn set_to_property_user_dir(&mut self) {
        self.allow_get_parent = false;
        if let Some(manager) = self.manager {
            let property_user_dir = manager.get_property_user_dir();
            if let Some(property_user_dir) = property_user_dir {
                let user_dir = PathBuf::from(&property_user_dir);
                if user_dir.is_dir()
                    && user_dir.exists()
                    && java_io_file_can_read(&property_user_dir)
                {
                    self.dir = Some(user_dir);
                    return;
                }
            }
        }
        self.dir = Some(PathBuf::from("."));
    }

    /// Java static `isValid(File)`.  Returns true if dir or dir's parent is a valid
    /// readable directory.
    ///
    /// Upstream bug fixed (ValidDirectory.java:87-90): a non-directory with no parent
    /// (a bare file name) leaves `dir` null and `dir.exists()` throws a
    /// NullPointerException.  Such a file has no directory to be valid, so false is
    /// returned.
    pub fn is_valid_file(dir: Option<&Path>) -> bool {
        let Some(dir) = dir else {
            return false;
        };
        let mut dir: PathBuf = dir.to_path_buf();
        if !dir.is_dir() {
            let Some(parent) = java_io_file_get_parent(&dir.to_string_lossy()) else {
                return false;
            };
            dir = PathBuf::from(parent);
        }
        dir.exists() && dir.is_dir() && java_io_file_can_read(&dir.to_string_lossy())
    }

    /// Java static `isValid(BrowsingDirectory)`.  Returns true if browsing dir or dir's
    /// parent is a valid readable directory.
    pub fn is_valid_browsing_directory(browsing_directory: Option<&dyn BrowsingDirectory>) -> bool {
        let Some(browsing_directory) = browsing_directory else {
            return false;
        };
        ValidDirectory::is_valid_file(browsing_directory.get_browsing_dir().as_deref())
    }

    /// Java static `get(File)`.  Returns dir or dir's parent if one is a valid readable
    /// directory.
    ///
    /// Upstream bug fixed (ValidDirectory.java:114-117): as in `isValid(File)`, a
    /// non-directory with no parent throws a NullPointerException; null is returned.
    pub fn get_file(dir: Option<&Path>) -> Option<PathBuf> {
        let dir = dir?;
        let mut dir: PathBuf = dir.to_path_buf();
        if !dir.is_dir() {
            dir = PathBuf::from(java_io_file_get_parent(&dir.to_string_lossy())?);
        }
        if dir.exists() && dir.is_dir() && java_io_file_can_read(&dir.to_string_lossy()) {
            return Some(dir);
        }
        None
    }

    /// Java static `get(BrowsingDirectory)`.  Returns browsing dir or dir's parent if
    /// one is a valid readable directory.
    pub fn get_browsing_directory(
        browsing_directory: Option<&dyn BrowsingDirectory>,
    ) -> Option<PathBuf> {
        let browsing_directory = browsing_directory?;
        ValidDirectory::get_file(browsing_directory.get_browsing_dir().as_deref())
    }

    /// Java `set(File)`.  Attempts to set a valid, readable directory.  If input is
    /// null, the instance resets.  The parent directory will be used if the input is a
    /// file.  If the input is not null and no directory can be set, nothing is changed.
    pub fn set_file(&mut self, input: Option<&Path>) {
        self.allow_get_parent = true;
        let Some(input) = input else {
            self.dir = None;
            return;
        };
        let mut temp: Option<PathBuf> = Some(input.to_path_buf());
        if !input.is_dir() {
            temp = java_io_file_get_parent(&input.to_string_lossy()).map(PathBuf::from);
        }
        let Some(temp) = temp else {
            self.set_to_property_user_dir();
            return;
        };
        if temp.is_dir() && temp.exists() && java_io_file_can_read(&temp.to_string_lossy()) {
            self.dir = Some(temp);
        } else {
            if *DEBUG {
                eprintln!(
                    "Warning:  unable to set browsing directory from: {}.",
                    java_io_file_get_absolute_path(&input.to_string_lossy())
                );
            }
        }
    }

    /// Java `set(String)`.  Attempts to call set(File) with new File(input).  Null
    /// input causes a reset.  Empty or blank input is treated as the current directory.
    pub fn set_string(&mut self, input: Option<&str>) {
        self.allow_get_parent = true;
        let Some(input) = input else {
            self.dir = None;
            return;
        };
        // A File class instance created from an empty or blank string does not have a
        // parent.  Use the current directory.
        if java_lang_string_matches_whitespace(input) {
            self.set_to_property_user_dir();
            return;
        }
        self.set_file(Some(Path::new(input)));
    }
}

/// Java `toString()`.
impl std::fmt::Display for ValidDirectory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.dir {
            None => f.write_str("[dir:]"),
            Some(dir) => write!(
                f,
                "[dir:{}]",
                java_io_file_get_absolute_path(&dir.to_string_lossy())
            ),
        }
    }
}
