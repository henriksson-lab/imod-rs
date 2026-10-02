//! `IMOD/Etomo/src/etomo/logic/ConfigTool.java`.
//!
//! Tool for the retrieval of configuration files.  This includes the three levels of
//! template files, which contain directives, and the distortion file.  A Java class of
//! static methods: module-level functions here.
//!
//! `java.io.File[]` is `Option<Vec<PathBuf>>` (null when the directory cannot be
//! listed, or when no files were found).  `File.listFiles(FileFilter)` is a
//! `read_dir` filtered through the filter's `accept`.  Java's `TreeMap<String, File>`
//! keyed on the file name is a `BTreeMap`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::storage::autodoc_filter::AutodocFilter;
use crate::imod::etomo::util::utilities::{
    java_io_file_can_read, java_io_file_get_absolute_path, java_io_file_get_name,
    java_io_file_get_parent, java_io_file_new,
};

/// Java private static final `DEFAULT_SYSTEM_TEMPLATE_DIR`.
const DEFAULT_SYSTEM_TEMPLATE_DIR: &str = "SystemTemplate";

/// Java `getScopeTemplateFiles()`.  Returns a sorted list of the scope template files.
pub fn get_scope_template_files() -> Option<Vec<PathBuf>> {
    let mut map: BTreeMap<String, PathBuf> = BTreeMap::new();
    // `EtomoDirector.getIMODCalibDirectory()` returns a new absolute File.
    let calib_dir = java_io_file_get_absolute_path(
        &etomo_director::INSTANCE.get_imod_calib_directory()
            .map(|dir| dir.to_string_lossy().to_string())
            .unwrap_or_default(),
    );
    let dir = java_io_file_new(&calib_dir, "ScopeTemplate");
    let filter = AutodocFilter::new_exclude_hidden(true);
    let file_array: Option<Vec<PathBuf>> = std::fs::read_dir(&dir).ok().map(|entries| {
        entries
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| filter.accept(path))
            .collect()
    });
    let file_array = file_array?;
    for i in 0..file_array.len() {
        map.insert(
            java_io_file_get_name(&file_array[i].to_string_lossy()),
            file_array[i].clone(),
        );
    }
    let size = map.len();
    if size == 0 {
        return None;
    }
    if size == 1 {
        return Some(vec![map.values().next().unwrap().clone()]);
    }
    Some(map.into_values().collect())
}

/// Java `getSystemTemplateFiles()`.  Returns a sorted list of the system template
/// files.  File names in ImodCalib override files in IMOD_DIR.
pub fn get_system_template_files() -> Option<Vec<PathBuf>> {
    let filter = AutodocFilter::new_exclude_hidden(true);
    // `EtomoDirector.getIMODDirectory()` returns a new absolute File.
    let imod_dir = java_io_file_get_absolute_path(
        &etomo_director::INSTANCE.get_imod_directory()
            .map(|dir| dir.to_string_lossy().to_string())
            .unwrap_or_default(),
    );
    let mut file_array: Option<Vec<PathBuf>> =
        std::fs::read_dir(java_io_file_new(&imod_dir, DEFAULT_SYSTEM_TEMPLATE_DIR))
            .ok()
            .map(|entries| {
                entries
                    .filter_map(|entry| entry.ok())
                    .map(|entry| entry.path())
                    .filter(|path| filter.accept(path))
                    .collect()
            });
    let mut map: Option<BTreeMap<String, PathBuf>> = None;
    if let Some(file_array) = &file_array {
        let mut new_map = BTreeMap::new();
        for i in 0..file_array.len() {
            new_map.insert(
                java_io_file_get_name(&file_array[i].to_string_lossy()),
                file_array[i].clone(),
            );
        }
        map = Some(new_map);
    }
    let calib_dir = java_io_file_get_absolute_path(
        &etomo_director::INSTANCE.get_imod_calib_directory()
            .map(|dir| dir.to_string_lossy().to_string())
            .unwrap_or_default(),
    );
    let filter = AutodocFilter::new_exclude_hidden(true);
    file_array = std::fs::read_dir(java_io_file_new(&calib_dir, DEFAULT_SYSTEM_TEMPLATE_DIR))
        .ok()
        .map(|entries| {
            entries
                .filter_map(|entry| entry.ok())
                .map(|entry| entry.path())
                .filter(|path| filter.accept(path))
                .collect()
        });
    if let Some(file_array) = &file_array {
        if map.is_none() {
            map = Some(BTreeMap::new());
        }
        let map = map.as_mut().unwrap();
        for i in 0..file_array.len() {
            let key = java_io_file_get_name(&file_array[i].to_string_lossy());
            if map.contains_key(&key) {
                map.remove(&key);
            }
            map.insert(key, file_array[i].clone());
        }
    }
    let map = map?;
    let size = map.len();
    if size == 0 {
        return None;
    }
    if size == 1 {
        return Some(vec![map.values().next().unwrap().clone()]);
    }
    Some(map.into_values().collect())
}

/// Java `getDefaultUserTemplateDir()`.  Returns the user's .etomotemplate directory,
/// or null if the user's home directory is not known.  (`System.getProperty
/// ("user.home")` is the JVM's `$HOME`.)
pub fn get_default_user_template_dir() -> Option<PathBuf> {
    let home_directory = std::env::var("HOME").ok()?;
    Some(PathBuf::from(java_io_file_new(
        &home_directory,
        ".etomotemplate",
    )))
}

/// Java `getUserTemplateFiles(File)`.  Returns a sorted list of the user template
/// files.  User template files are stored either in .etomotemplate, or in a directory
/// specified in the Settings dialog.  `new_user_template_dir` overrides the user
/// template directory from User Configuration.
pub fn get_user_template_files(new_user_template_dir: Option<&Path>) -> Option<Vec<PathBuf>> {
    let user_template_dir_path: Option<String>;
    if let Some(new_user_template_dir) = new_user_template_dir {
        user_template_dir_path = Some(java_io_file_get_absolute_path(
            &new_user_template_dir.to_string_lossy(),
        ));
    } else {
        // `EtomoDirector.INSTANCE.getUserConfiguration().getUserTemplateDir()`.
        user_template_dir_path = etomo_director::INSTANCE.with_user_configuration(|c| c.get_user_template_dir());
    }
    let dir: Option<PathBuf>;
    if let Some(user_template_dir_path) = user_template_dir_path {
        dir = Some(PathBuf::from(user_template_dir_path));
    } else {
        dir = get_default_user_template_dir();
    }
    let mut map = sort_autodoc_files(dir.as_deref(), None);
    let second_user_template_dir: Option<PathBuf> = etomo_director::ARGUMENTS
        .lock()
        .unwrap()
        .get_user_template_loc()
        .map(Path::to_path_buf);
    if let Some(second_user_template_dir) = second_user_template_dir {
        map = sort_autodoc_files(Some(&second_user_template_dir), map);
    }
    let map = map?;
    let size = map.len();
    if size == 0 {
        return None;
    }
    if size == 1 {
        return Some(vec![map.values().next().unwrap().clone()]);
    }
    Some(map.into_values().collect())
}

/// Java private `sortAutodocFiles(File, SortedMap)`.  Get the autodoc files from dir
/// and add them to a sorted map.  Returns map.
fn sort_autodoc_files(
    dir: Option<&Path>,
    mut map: Option<BTreeMap<String, PathBuf>>,
) -> Option<BTreeMap<String, PathBuf>> {
    let Some(dir) = dir else {
        return map;
    };
    if etomo_director::ARGUMENTS.lock().unwrap().is_debug() {
        eprintln!(
            "User template location:{}",
            java_io_file_get_absolute_path(&dir.to_string_lossy())
        );
    }
    let filter = AutodocFilter::new();
    let file_array: Option<Vec<PathBuf>> = std::fs::read_dir(dir).ok().map(|entries| {
        entries
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| filter.accept(path))
            .collect()
    });
    let file_array = match file_array {
        Some(file_array) if !file_array.is_empty() => file_array,
        _ => return map,
    };
    if map.is_none() {
        map = Some(BTreeMap::new());
    }
    for i in 0..file_array.len() {
        map.as_mut().unwrap().insert(
            java_io_file_get_name(&file_array[i].to_string_lossy()),
            file_array[i].clone(),
        );
    }
    map
}

/// Java `getDistortionDir(BaseManager, File)`.  Returns either the parent of
/// currentDistortionFile, the distortion directory in ImodCalib, or the property user
/// directory.
pub fn get_distortion_dir(
    manager: &'static dyn BaseManager,
    curt_distortion_file: Option<&Path>,
) -> Option<String> {
    if let Some(curt_distortion_file) = curt_distortion_file {
        let dir = java_io_file_get_parent(&curt_distortion_file.to_string_lossy());
        if let Some(dir) = dir {
            let dir_path = Path::new(&dir);
            if dir_path.exists() && dir_path.is_dir() && java_io_file_can_read(&dir) {
                return Some(java_io_file_get_absolute_path(&dir));
            }
        }
    }
    let calib_dir = java_io_file_get_absolute_path(
        &etomo_director::INSTANCE.get_imod_calib_directory()
            .map(|dir| dir.to_string_lossy().to_string())
            .unwrap_or_default(),
    );
    let dir = java_io_file_new(&calib_dir, "Distortion");
    let dir_path = Path::new(&dir);
    if dir_path.exists() && dir_path.is_dir() && java_io_file_can_read(&dir) {
        return Some(java_io_file_get_absolute_path(&dir));
    }
    manager.get_property_user_dir()
}
