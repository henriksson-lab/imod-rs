//! `IMOD/Etomo/src/etomo/util/FilePath.java`.
//!
//! Manipulates file paths.
//!
//! Windows path formats: absolute paths are standard DOS (`\Documents\Newsletter`,
//! `C:\Documents\Newsletter`), UNC (`\\system7\C$\Documents\Newsletters`) and WIN32
//! device paths (`\\.\C:\Documents\Newsletter`, `\\?\Volume{...}\Documents`); relative
//! paths are standard DOS (`Documents\Newsletter`, `.\Documents`, `..\Documents`,
//! `C:Documents\Newsletter`), and WIN32 device paths with a relative drive are treated
//! as absolute by this class.  The File class does not seem to handle the UNC and WIN32
//! formats well, and there haven't been any requests to support them.  So support for
//! them is currently limited to avoiding serious failure.
//!
//! A Java `File` is represented by its path string (normalized as `new File(String)`
//! normalizes it) or a `PathBuf` holding it; the `java.io.File` operations are the ones
//! in `util/utilities.rs` (`java_io_file_*`).  `File.separator` is the platform's
//! separator.  Java string indexes are UTF-16 units; `removeFirst`/`removeLast` count
//! characters, which is the same for every path in the Basic Multilingual Plane.
//!
//! Overloads carry a suffix naming their parameter types
//! (`build_absolute_file_string_string`, `build_absolute_file_string_file`).

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use regex::Regex;

use crate::imod::etomo::r#type::const_etomo_number::{
    java_lang_string_matches_whitespace, java_lang_string_trim,
};
use crate::imod::etomo::util::utilities;

/// Java `File.separator`.
const FILE_SEPARATOR: &str = std::path::MAIN_SEPARATOR_STR;
/// Java private static final `CURRENT_DIR_SYMBOL`.
const CURRENT_DIR_SYMBOL: &str = ".";
/// Java private static final `UP_DIR_SYMBOL`.
const UP_DIR_SYMBOL: &str = "..";
/// Java private static final `WIN_DRIVE_SEPARATOR`.
const WIN_DRIVE_SEPARATOR: &str = ":";
/// Java private static `debug`, initialised to false.
static DEBUG: AtomicBool = AtomicBool::new(false);

/// Java public final class `FilePath`.
#[derive(Debug)]
pub struct FilePath {
    /// Java private final `origPath`, a `StringBuffer`: the original, unmodified path.
    orig_path: String,
    /// Java private final `pathArray`: array of the canonical path or the original
    /// filePath parameter if the canonical path can't be created.
    path_array: Option<Vec<String>>,
    /// Java private final `regExpSeparator`.
    reg_exp_separator: String,
    /// Java private final `drive`.
    drive: Option<String>,
    /// Java private final `origPathWithoutDrive`.
    orig_path_without_drive: Option<String>,
    /// Java private final `alternativeFormat`.
    alternative_format: bool,
}

impl FilePath {
    /// Java private `FilePath(FilePath)`.  Deep copy constructor.
    fn copy(file_path: &FilePath) -> FilePath {
        let reg_exp_separator = file_path.reg_exp_separator.clone();
        let drive = file_path.drive.clone();
        let orig_path_without_drive = file_path.orig_path_without_drive.clone();
        let alternative_format = file_path.alternative_format;
        let path_array = match &file_path.path_array {
            Some(source) => {
                let mut path_array = Vec::with_capacity(source.len());
                for i in 0..source.len() {
                    path_array.push(source[i].clone());
                }
                Some(path_array)
            }
            None => None,
        };
        let mut orig_path = String::new();
        orig_path.push_str(&file_path.orig_path);
        FilePath {
            orig_path,
            path_array,
            reg_exp_separator,
            drive,
            orig_path_without_drive,
            alternative_format,
        }
    }

    /// Java private `FilePath(String)`.
    fn new(file_path: Option<&str>) -> FilePath {
        let reg_exp_separator = if FILE_SEPARATOR == "\\" {
            "\\".to_string() + FILE_SEPARATOR
        } else {
            FILE_SEPARATOR.to_string()
        };
        let file_path = file_path.map(java_lang_string_trim);
        let mut orig_path = String::new();
        // Do not create a current location path from an empty parameter.
        if utilities::is_empty(file_path) {
            return FilePath {
                orig_path,
                path_array: None,
                reg_exp_separator,
                drive: None,
                orig_path_without_drive: None,
                alternative_format: false,
            };
        }
        let file_path = file_path.unwrap();
        orig_path.push_str(file_path);
        let unc_and_win32_prefix = FILE_SEPARATOR.to_string() + FILE_SEPARATOR;
        let alternative_format =
            utilities::is_windows_os() && file_path.starts_with(&unc_and_win32_prefix);

        // Save the drive, and the original path minus the drive. WIN32 path formats are
        // not supported.
        let drive: Option<String>;
        let orig_path_without_drive: Option<String>;
        let drive_index = orig_path.find(WIN_DRIVE_SEPARATOR);
        if utilities::is_windows_os()
            && !alternative_format
            && let Some(drive_index) = drive_index
        {
            let array = utilities::java_lang_string_split(
                &orig_path,
                &Regex::new(WIN_DRIVE_SEPARATOR).unwrap(),
            );
            if array.len() > 1 && !array[0].is_empty() {
                drive = Some(array[0].clone() + WIN_DRIVE_SEPARATOR);
                orig_path_without_drive = Some(orig_path[drive_index + 1..].to_string());
            } else {
                drive = None;
                orig_path_without_drive = Some(orig_path.clone());
            }
        } else {
            drive = None;
            orig_path_without_drive = Some(orig_path.clone());
        }

        // The canonical path is useful for manipulating the path because is doesn't
        // contain "." or "..". It's always absolute.
        //
        // `new File(filePath).getCanonicalPath()`, as the JDK's
        // `UnixFileSystem.canonicalize` computes it: the absolute path through
        // realpath(3); when that fails, realpath of the longest existing parent with the
        // rest appended and "." and ".." collapsed.
        let absolute = utilities::java_io_file_get_absolute_path(file_path);
        let mut canonical_path: Option<String> = match std::fs::canonicalize(&absolute) {
            Ok(canonical) => Some(canonical.to_string_lossy().to_string()),
            Err(_) => {
                let mut prefix = absolute.clone();
                let mut remainder = String::new();
                let mut joined: Option<String> = None;
                while let Some(index) = prefix.rfind('/') {
                    remainder = prefix[index..].to_string() + &remainder;
                    prefix.truncate(index);
                    let base = if prefix.is_empty() {
                        "/"
                    } else {
                        prefix.as_str()
                    };
                    if let Ok(canonical) = std::fs::canonicalize(base) {
                        joined = Some(canonical.to_string_lossy().to_string() + &remainder);
                        break;
                    }
                }
                joined.map(|joined| {
                    let mut names: Vec<&str> = Vec::new();
                    for name in joined.split('/') {
                        if name.is_empty() || name == "." {
                            continue;
                        }
                        if name == ".." {
                            names.pop();
                            continue;
                        }
                        names.push(name);
                    }
                    "/".to_string() + &names.join("/")
                })
            }
        };
        if canonical_path.is_none() {
            // Use the orginal path if canonical path isn't available.
            canonical_path = Some(file_path.to_string());
        }
        let canonical_path = canonical_path.unwrap();
        // Remove the drive from the canonical path, and use the result to make the path
        // array. Removing the drive is only necessary on Windows, and only works with
        // paths with a drive designator.(example: C:). UNC or Win32 paths without the
        // drive designator are always absolute, and the code works even though their
        // drives/volumes/etc can't be split off.
        let mut canonical_path_without_drive = canonical_path.as_str();
        let canonical_path_drive_index = canonical_path.find(WIN_DRIVE_SEPARATOR);
        if utilities::is_windows_os()
            && let Some(canonical_path_drive_index) = canonical_path_drive_index
        {
            canonical_path_without_drive = &canonical_path[canonical_path_drive_index + 1..];
        }
        // Build the pathArray
        let temp_array = utilities::java_lang_string_split(
            canonical_path_without_drive,
            &Regex::new(&reg_exp_separator).unwrap(),
        );
        let mut path_array = Vec::with_capacity(temp_array.len());
        for i in 0..temp_array.len() {
            path_array.push(temp_array[i].clone());
        }
        FilePath {
            orig_path,
            path_array: Some(path_array),
            reg_exp_separator,
            drive,
            orig_path_without_drive,
            alternative_format,
        }
    }

    /// Java private `FilePath(File)`: `this(file != null ? file.getPath() : null)`.
    fn new_file(file: Option<&Path>) -> FilePath {
        let path = file.map(|file| file.to_string_lossy().to_string());
        FilePath::new(path.as_deref())
    }

    /// Java private `removeLast()`.
    ///
    /// Upstream bug fixed (FilePath.java:171): the length removed is that of the last
    /// *canonical* element, but it is removed from the *original* path, which for a
    /// relative from-path is shorter than the canonical one; `StringBuffer.delete`
    /// then gets a negative start and throws StringIndexOutOfBoundsException, which
    /// nothing catches.  The start is clamped to 0, so the whole remaining original
    /// path is removed.
    fn remove_last(&mut self) {
        let Some(path_array) = self.path_array.as_mut() else {
            return;
        };
        if path_array.is_empty() {
            return;
        }
        let mut amount_to_remove = path_array.pop().unwrap().chars().count() as i64;
        if self.orig_path.ends_with(FILE_SEPARATOR) {
            amount_to_remove += 1;
        }
        let length = self.orig_path.chars().count() as i64;
        let start = (length - amount_to_remove).max(0) as usize;
        self.orig_path = self.orig_path.chars().take(start).collect();
    }

    /// Java private `removeFirst()`.  `StringBuffer.delete(0, end)` clamps an end past
    /// the length to the length.
    fn remove_first(&mut self) {
        let Some(path_array) = self.path_array.as_mut() else {
            return;
        };
        if path_array.is_empty() {
            return;
        }
        let mut amount_to_remove = path_array.remove(0).chars().count() + 1;
        if self.orig_path.starts_with(FILE_SEPARATOR) {
            amount_to_remove += 1;
        }
        self.orig_path = self.orig_path.chars().skip(amount_to_remove).collect();
    }

    /// Java public static `isPath(String)`.  Returns true if name contains the file
    /// separator.  In Windows also returns true if name contain ":".
    pub fn is_path(name: Option<&str>) -> bool {
        let name = match name {
            Some(name) if !java_lang_string_matches_whitespace(name) => name,
            _ => return false,
        };
        name.contains(FILE_SEPARATOR)
            || (utilities::is_windows_os() && name.find(":").is_some_and(|index| index > 0))
    }

    /// Java public static `getFileName(String)`.
    pub fn get_file_name(path: Option<&str>) -> Option<String> {
        if FilePath::is_path(path) {
            return Some(utilities::java_io_file_get_name(path.unwrap()));
        }
        path.map(|path| path.to_string())
    }

    /// Java public static `getFileParent(String)`.
    pub fn get_file_parent(path: Option<&str>) -> Option<PathBuf> {
        if FilePath::is_path(path) {
            return utilities::java_io_file_get_parent(path.unwrap()).map(PathBuf::from);
        }
        None
    }

    /// Java public static `getRelativePath(String, File)`.  Returns a relative path
    /// going from fromAbsolutePath to toAbsoluteFile.  Uses the canonical absolute path
    /// to avoid problems with "." and "..".  If one of the parameters is null or the
    /// two paths do not shared a root directory, an absolute path is returned.
    pub fn get_relative_path(
        from_absolute_path: Option<&str>,
        to_absolute_file: Option<&Path>,
    ) -> Option<String> {
        let from_absolute_path = match from_absolute_path {
            Some(from_absolute_path)
                if !java_lang_string_matches_whitespace(from_absolute_path) =>
            {
                from_absolute_path
            }
            _ => {
                if let Some(to_absolute_file) = to_absolute_file {
                    return Some(utilities::java_io_file_get_absolute_path(
                        &to_absolute_file.to_string_lossy(),
                    ));
                }
                return None;
            }
        };
        let Some(to_absolute_file) = to_absolute_file else {
            return Some(from_absolute_path.to_string());
        };
        let from_path = FilePath::new(Some(from_absolute_path));
        let to_path = FilePath::new(Some(&utilities::java_io_file_get_absolute_path(
            &to_absolute_file.to_string_lossy(),
        )));
        Some(from_path.get_relative_path_to(Some(&to_path)))
    }

    /// Java public static `buildAbsoluteFile(String, String)`.  Builds and returns an
    /// absolute file if possible.  Returns a File instance of filePath if filePath is
    /// an absolute path or dir is empty, otherwise returns a File instance of
    /// dir/filePath.
    pub fn build_absolute_file_string_string(dir: Option<&str>, file_path: &str) -> PathBuf {
        let file = if file_path == CURRENT_DIR_SYMBOL {
            utilities::java_io_file_normalize("")
        } else {
            utilities::java_io_file_normalize(file_path)
        };
        let dir = match dir {
            Some(dir)
                if !Path::new(&file).is_absolute() && !java_lang_string_matches_whitespace(dir) =>
            {
                dir
            }
            _ => return PathBuf::from(file),
        };
        if file_path == CURRENT_DIR_SYMBOL {
            return PathBuf::from(utilities::java_io_file_new(dir, &file));
        }
        PathBuf::from(utilities::java_io_file_new(dir, file_path))
    }

    /// Java public static `buildAbsoluteFile(String, File)`.  Builds and returns an
    /// absolute file if possible.  Return file if file is absolute or dir is empty,
    /// otherwise returns a File instance of dir/file.
    pub fn build_absolute_file_string_file(dir: Option<&str>, file: &Path) -> PathBuf {
        let mut file = utilities::java_io_file_normalize(&file.to_string_lossy());
        if file == CURRENT_DIR_SYMBOL {
            file = utilities::java_io_file_normalize("");
        }
        let dir = match dir {
            Some(dir)
                if !Path::new(&file).is_absolute() && !java_lang_string_matches_whitespace(dir) =>
            {
                dir
            }
            _ => return PathBuf::from(file),
        };
        PathBuf::from(utilities::java_io_file_new(dir, &file))
    }

    /// Java public static `getRerootedRelativePath(String, String, String)`.  Make a
    /// new relative path from the new root to a file whose relative path is relative to
    /// the old root.  If the file's path is absolute, or if newRoot is missing, return
    /// filePath as is.
    ///
    /// Upstream bug fixed (FilePath.java:293): a null filePath with a non-empty newRoot
    /// reaches `new File(filePath)` and throws a NullPointerException.  A null path has
    /// nothing to reroot, so null is returned, as it already is when newRoot is empty.
    pub fn get_rerooted_relative_path(
        old_root: Option<&str>,
        new_root: Option<&str>,
        file_path: Option<&str>,
    ) -> Option<String> {
        if utilities::is_empty(new_root) {
            return file_path.map(|file_path| file_path.to_string());
        }
        let file_path = file_path?;
        if Path::new(&utilities::java_io_file_normalize(file_path)).is_absolute()
            || FilePath::new(Some(file_path)).alternative_format
        {
            return Some(file_path.to_string());
        }
        FilePath::get_relative_path(
            new_root,
            Some(&FilePath::get_absolute_file(old_root, Some(file_path))),
        )
    }

    /// Java public static `setDebug(boolean)`.
    pub fn set_debug(input: bool) {
        DEBUG.store(input, Ordering::Relaxed);
    }

    /// Java private static `getAbsoluteFile(String, String)`.  Returns a file with an
    /// absolute path going from fromAbsolutePath to filePath.  fromAbsolutePath is
    /// assumed to be absolute.
    ///
    /// Upstream bug fixed (FilePath.java:312): an empty toRelativePath with a null
    /// fromAbsolutePath calls `new File(null)`, a NullPointerException.  A null
    /// from-path is taken as the empty path, as `new FilePath(null)` already takes it
    /// on the other branch.
    fn get_absolute_file(
        from_absolute_path: Option<&str>,
        to_relative_path: Option<&str>,
    ) -> PathBuf {
        match to_relative_path {
            Some(to_relative_path) if !java_lang_string_matches_whitespace(to_relative_path) => {}
            _ => {
                return PathBuf::from(utilities::java_io_file_normalize(
                    from_absolute_path.unwrap_or(""),
                ));
            }
        }
        let from_path = FilePath::new(from_absolute_path);
        let to_path = FilePath::new(to_relative_path);
        let output = from_path.get_absolute_path_to(&to_path);
        PathBuf::from(utilities::java_io_file_normalize(&output))
    }

    /// Java private `getDrive()`.
    fn get_drive(&self) -> String {
        match &self.drive {
            None => String::new(),
            Some(drive) => drive.clone(),
        }
    }

    /// Java private `getAbsolutePathTo(FilePath)`.  Returns an absolute path going from
    /// this to toRelPath.  If this is impossible, returns the absolute path in
    /// toRelPath.
    fn get_absolute_path_to(&self, to_rel_path: &FilePath) -> String {
        // Alternative path formats (Windows UNC and WIN32) are not supported and assumed
        // to be absolute.
        if self.alternative_format {
            return to_rel_path.orig_path.clone();
        }
        // Make sure that the drives are the same
        if utilities::is_windows_os()
            && (to_rel_path.drive.is_some()
                && self.drive.is_some()
                && self.drive != to_rel_path.drive)
        {
            return utilities::java_io_file_get_absolute_path(&to_rel_path.orig_path);
        }
        let path_array_empty = match &self.path_array {
            None => true,
            Some(path_array) => path_array.is_empty(),
        };
        if path_array_empty {
            // No from-path - return absolute to-path
            return utilities::java_io_file_get_absolute_path(&to_rel_path.orig_path);
        }
        // If the to path doesn't contain ..'s then use path/toRelPath.path.
        if !to_rel_path.orig_path.contains(UP_DIR_SYMBOL) {
            // Upstream bug fixed (FilePath.java:350): an empty to-path (one `trim`
            // empties although `matches("\\s*")` did not reject it, e.g. control
            // characters) has a null `origPathWithoutDrive`, and `new File(String,
            // null)` throws a NullPointerException.  It is taken as the empty path.
            return utilities::java_io_file_get_absolute_path(&utilities::java_io_file_new(
                &self.orig_path,
                &to_rel_path.get_orig_path(true).unwrap_or_default(),
            ));
        }
        // Go up the from path by removing ..'s at the beginning of the to path.
        let mut temp_from_path = FilePath::copy(self);
        let mut temp_to_path = FilePath::copy(to_rel_path);
        let to_path_array: &[String] = to_rel_path.path_array.as_deref().unwrap_or(&[]);
        let mut i = 0;
        while i < to_path_array.len() {
            if to_path_array[i] == UP_DIR_SYMBOL {
                temp_from_path.remove_last();
                temp_to_path.remove_first();
            } else {
                break;
            }
            i += 1;
        }
        utilities::java_io_file_get_absolute_path(&utilities::java_io_file_new(
            &temp_from_path.orig_path,
            &temp_to_path.orig_path,
        ))
    }

    /// Java private `getOrigPath(boolean)`.
    fn get_orig_path(&self, remove_drive: bool) -> Option<String> {
        if !remove_drive {
            return Some(self.orig_path.clone());
        }
        self.orig_path_without_drive.clone()
    }

    /// Java private `getRelativePathTo(FilePath)`.  Returns a relative path going from
    /// the path in this instance to toAbsPath.  If this is impossible, returns the
    /// absolute path in toAbsPath.
    ///
    /// Java wraps the computation in a `try` that returns the absolute to-path on an
    /// index exception; every index here is bounds-checked, so no such exception can
    /// arise and the arm has no counterpart.  A to-path with no path array (Java would
    /// throw an uncaught NullPointerException; `getRelativePath` always passes an
    /// absolute, non-empty one) is taken as empty.
    fn get_relative_path_to(&self, to_abs_path: Option<&FilePath>) -> String {
        let Some(to_abs_path) = to_abs_path else {
            return self.orig_path.clone();
        };
        // Make sure that the drives are the same. This is case sensitive and probably
        // shouldn't be. But it's hard to find a definitive statement that Windows drives
        // are not case sensitive, so I'm taking the safest route.
        if (self.drive.is_some() || to_abs_path.drive.is_some())
            && (self.drive.is_none() || self.drive != to_abs_path.drive)
        {
            // Drives are different - return absolute to-path
            return to_abs_path.orig_path.clone();
        }
        // Alternative path formats (Windows UNC and WIN32) are not supported and assumed
        // to be absolute.
        if self.alternative_format {
            return to_abs_path.orig_path.clone();
        }
        let path_array: &[String] = match &self.path_array {
            Some(path_array) if !path_array.is_empty() => path_array,
            // No from-path - return absolute to-path
            _ => return to_abs_path.orig_path.clone(),
        };
        let to_path_array: &[String] = to_abs_path.path_array.as_deref().unwrap_or(&[]);
        let mut same = true;
        let mut skip: i64 = -1;
        let mut go_up = 0;
        let mut from_index = 0usize;
        let mut to_index = 0usize;
        while from_index < path_array.len() {
            // Load an element from the "from" path.
            let from_dir = &path_array[from_index];
            // Handle ".". It doesn't change the directory so it can be ignored.
            if from_dir == CURRENT_DIR_SYMBOL {
                from_index += 1;
                continue;
            }
            if from_dir == UP_DIR_SYMBOL {
                // Getting the canonical path failed. Handling a file path that contains
                // ".." is not worth it for this rare case.
                return to_abs_path.orig_path.clone();
            }

            // Load an element from the "to" path.
            let mut to_dir: Option<&String> = None;
            if to_index < to_path_array.len() {
                to_dir = Some(&to_path_array[to_index]);
                if to_dir.unwrap() == CURRENT_DIR_SYMBOL {
                    to_index += 1;
                    continue;
                }
                if to_dir.unwrap() == UP_DIR_SYMBOL {
                    // Getting the canonical path failed. Handling a file path that
                    // contains ".." is not worth it for this rare case.
                    return to_abs_path.orig_path.clone();
                }
            }
            // Continue even if the if toAbsPath has run out. Must generate ".." to move
            // out of the part of the from path that doesn't exist in the to path.

            // Figure out which part of the from path is duplicated, and where to insert
            // ".."'s.
            if same && Some(from_dir) == to_dir {
                // Keep track of the identical places to ignore.
                skip = to_index as i64;
            } else {
                // The "from" path is going down a different branch from the "to" path.
                // Use ".." to go up.
                same = false;
                go_up += 1;
            }
            from_index += 1;
            to_index += 1;
        }

        // Build relative path.
        let mut buffer = String::new();
        // Add the ".."'s to get out of the "from" path's separate branch.
        for i in 0..go_up {
            if i > 0 {
                buffer.push_str(FILE_SEPARATOR);
            }
            buffer.push_str(UP_DIR_SYMBOL);
        }
        // Add the unique part of the "to" path.
        let mut i = (skip + 1) as usize;
        while i < to_path_array.len() {
            if go_up > 0 {
                buffer.push_str(FILE_SEPARATOR);
                go_up = 0;
            }
            let to_dir = &to_path_array[i];
            // If the "from" path is shorter then the "to" path, from the "from" path
            // still needs to be checked.
            if to_dir != CURRENT_DIR_SYMBOL {
                buffer.push_str(&to_path_array[i]);
                if i < to_path_array.len() - 1 {
                    buffer.push_str(FILE_SEPARATOR);
                }
            }
            i += 1;
        }
        buffer
    }
}

/// Java `toString()`: the original path.
impl std::fmt::Display for FilePath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.orig_path)
    }
}
