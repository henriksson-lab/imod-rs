//! `IMOD/Etomo/src/etomo/logic/DatasetTool.java` (with its nested `StackNameFilter` and
//! `StackInfo`).
//!
//! Copyright: Copyright 2012 - 2025 by the Regents of the University of Colorado
//!
//! Organization: Dept. of MCD Biology, University of Colorado
//!
//! **Shape.**  A Java `File` is a path string here (`&str`/`&Path` in, `PathBuf` out),
//! handled through the `java.io.File` shims in util/utilities.rs.  Java overloads carry
//! descriptive suffixes; the overload the Rust callers already use keeps the plain
//! name.  Parameters the Java allows to be null and that callers pass both as values
//! and as `Option`s are `impl Into<Option<T>>`.  A Java `BaseManager` is
//! `&'static dyn BaseManager`; no Java caller of these methods passes null except
//! through `validateDatasetName`'s `manager != null ?` test, which is always true here.
//!
//! `StackInfo`s refer to each other (`matchingStack`), so the list
//! `removeMatchingBStacks` returns holds `Rc<RefCell<StackInfo>>` and the back
//! references are `Weak`: the list owns every instance, as the Java list does, and the
//! pair does not keep itself alive.  They are Swing-side objects (file chooser), never
//! handed to a process.
//!
//! **`UIComponent`** is `ui_harness::UiComponentBoundary`, the boundary the translated
//! `UIHarness` uses for it.
#![allow(dead_code)]

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::{Rc, Weak};
use std::sync::LazyLock;

use regex::Regex;

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::storage::data_file_filter::DataFileFilter;
use crate::imod::etomo::storage::log_file::{LogFile, LogFileError};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::const_etomo_number::java_lang_string_matches_whitespace;
use crate::imod::etomo::r#type::data_file_type::DataFileType;
use crate::imod::etomo::r#type::extension::{EXTENSION_DIVIDER, Extension};
use crate::imod::etomo::r#type::file_type;
use crate::imod::etomo::r#type::string_property::StringProperty;
use crate::imod::etomo::r#type::view_type::ViewType;
use crate::imod::etomo::ui::swing::ui_harness::{self, UiComponentBoundary};
use crate::imod::etomo::util::montagesize::Montagesize;
use crate::imod::etomo::util::mrc_header::MRCHeader;
use crate::imod::etomo::util::stack_trace::StackTrace;
use crate::imod::etomo::util::utilities::{
    self, java_io_file_can_read, java_io_file_can_write, java_io_file_get_absolute_path,
    java_io_file_get_name, java_io_file_get_parent, java_io_file_new,
};

/// Java `STANDARD_DATASET_EXT` (deprecated 5/9/2019).
pub const STANDARD_DATASET_EXT: &str = ".st";
/// Java `ALTERNATE_DATASET_EXT` (deprecated 5/9/2019).
pub const ALTERNATE_DATASET_EXT: &str = ".mrc";

/// Java private `MESSAGE_TITLE`.
const MESSAGE_TITLE: &str = "Invalid Dataset Directory";
/// Java private `DEBUG`.
static DEBUG: LazyLock<bool> =
    LazyLock::new(|| etomo_director::ARGUMENTS.lock().unwrap().is_debug());

/// Java `"\\s*\\S+\\s+\\S+(\\s+\\S+)*\\s*"` used with `String.matches` (anchored at both
/// ends).  Java's `\s` is `[ \t\n\x0B\f\r]` and `\S` its complement.
static EMBEDDED_SPACES: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"^[ \t\n\x0B\x0C\r]*[^ \t\n\x0B\x0C\r]+[ \t\n\x0B\x0C\r]+[^ \t\n\x0B\x0C\r]+([ \t\n\x0B\x0C\r]+[^ \t\n\x0B\x0C\r]+)*[ \t\n\x0B\x0C\r]*$",
    )
    .unwrap()
});

/// Java `substituteExtension(String, Extension)`.  Substitutes newExtension for the
/// extension in fileName.  Works on both regular and standardized file names.
/// Substitutes standardized style when the file is standardized and the new extension is
/// standardizable (dataset_join.mrc => dataset_rot.mrc).  Can handle unrecognized
/// regular extensions.  ImageFilenameStyle is ignored so fileName can come from another
/// dataset.
pub fn substitute_extension(
    file_name: Option<&str>,
    new_extension: Option<&Extension>,
) -> Option<String> {
    let new_file_name = Extension::substitute_standardized_extension(file_name, new_extension);
    if new_file_name.is_some() {
        return new_file_name;
    }
    Extension::substitute_extension(file_name, new_extension)
}

// Updates done

/// Java `isValidInputImageFile(String, boolean, StringBuilder)`.
pub fn is_valid_input_image_file(
    input_image_file: &str,
    dual_axis: bool,
    mut err_msg: Option<&mut String>,
) -> bool {
    // Validate file name
    if !is_valid_input_image_file_name(Some(input_image_file), dual_axis, err_msg.as_deref_mut()) {
        return false;
    }
    let mut input_image_file = input_image_file.to_string();
    // Validate file
    if !input_image_file.contains('/') {
        // Work around. File.exists() fails when the File instance was constructed with only
        // a file name - even though the instance knows the correct absolute path.
        input_image_file = java_io_file_get_absolute_path(&input_image_file);
    }
    let file = Path::new(&input_image_file);
    if !file.exists() {
        if let Some(err_msg) = err_msg.as_deref_mut() {
            err_msg.push_str(&format!(
                "{} does not exist.  ",
                java_io_file_get_absolute_path(&input_image_file)
            ));
        }
        return false;
    }
    if file.is_dir() {
        if let Some(err_msg) = err_msg.as_deref_mut() {
            err_msg.push_str(&format!("{} is not a file.  ", input_image_file));
        }
        return false;
    }
    if !java_io_file_can_read(&input_image_file) {
        if let Some(err_msg) = err_msg.as_deref_mut() {
            err_msg.push_str(&format!("{} is not readable.  ", input_image_file));
        }
        return false;
    }
    if !java_io_file_can_write(&input_image_file) {
        if let Some(err_msg) = err_msg.as_deref_mut() {
            err_msg.push_str(&format!("{} is not writable.  ", input_image_file));
        }
        return false;
    }
    // Validate directory name
    let directory = java_io_file_get_parent(&input_image_file);
    if let Some(directory) = directory {
        let directory_path = java_io_file_get_absolute_path(&directory);
        if directory_path.ends_with(' ') {
            if let Some(err_msg) = err_msg.as_deref_mut() {
                err_msg.push_str(&format!(
                    "The directory for the dataset must not end in a space ({}).  ",
                    directory_path
                ));
            }
            return false;
        }
        // Validate directory
        if !java_io_file_can_write(&directory) {
            if let Some(err_msg) = err_msg.as_deref_mut() {
                err_msg.push_str(&format!("{} is not writable.  ", directory_path));
            }
            return false;
        }
    }
    true
}

/// Java `isValidInputImageFileName(String, boolean, StringBuilder)`.  Validates an input
/// image file.  When dual axis is true the axis letter is required.  `input_image_file_path`
/// is an input image file name or file path; only the file name is validated.  The
/// location and the file's status in that location are ignored.  If `err_msg` is not
/// null, a message will be added to it if the return value is false.
pub fn is_valid_input_image_file_name(
    input_image_file_path: Option<&str>,
    dual_axis: bool,
    mut err_msg: Option<&mut String>,
) -> bool {
    let input_image_file_path = match input_image_file_path {
        Some(input_image_file_path) if !input_image_file_path.is_empty() => input_image_file_path,
        _ => {
            if let Some(err_msg) = err_msg.as_deref_mut() {
                err_msg.push_str("Empty file path - not a valid input image file.  ");
            }
            return false;
        }
    };
    let name = java_io_file_get_name(input_image_file_path);
    // Check for illegal characters
    if !file_type::contains_valid_dataset_name(Some(&name), err_msg.as_deref_mut()) {
        return false;
    }
    // Check the extension
    if !name.contains(EXTENSION_DIVIDER) {
        if let Some(err_msg) = err_msg.as_deref_mut() {
            err_msg.push_str(&format!("{} does not contain an extension.  ", name));
        }
        return false;
    }
    // `Extension.getInstance(null)` returns null.
    let extension = utilities::get_extension(Some(&name))
        .and_then(|extension| Extension::get_instance(&extension));
    if extension.is_none_or(|extension| !extension.is_input_image_file()) {
        if let Some(err_msg) = err_msg.as_deref_mut() {
            err_msg.push_str(&format!(
                "{} does not contain a valid extension.  Valid extensions are:  {}.  ",
                name,
                Extension::get_input_image_file_descr()
            ));
        }
        return false;
    }
    let left_side = utilities::remove_extension(Some(&name));
    let first_extension = AxisID::First.get_extension();
    let second_extension = AxisID::Second.get_extension();
    let left_side = match left_side {
        Some(left_side)
            if !left_side.is_empty()
                && left_side != first_extension
                && left_side != second_extension =>
        {
            left_side
        }
        _ => {
            if let Some(err_msg) = err_msg.as_deref_mut() {
                err_msg.push_str(&format!(
                    "{} not a valid input image file name because the dataset name must be derived from this file name.  ",
                    name
                ));
            }
            return false;
        }
    };
    // Check the axis letter
    if !dual_axis {
        return true;
    }
    if left_side.encode_utf16().count() <= 1
        || (!left_side.ends_with(&first_extension) && !left_side.ends_with(&second_extension))
    {
        if let Some(err_msg) = err_msg.as_deref_mut() {
            err_msg.push_str(&format!(
                "{} must contain an axis letter ('{}' or '{}') before the extension for a dual axis dataset.  ",
                input_image_file_path, first_extension, second_extension
            ));
        }
        return false;
    }
    true
}

/// Java `getBAxisInputImageFile(String, boolean)`.  Returns B axis file name derived from
/// inputImageFilePath.  If dualAxis is false returns null.  Add the B axis letter if it
/// is not there.  Expects a file where the axis letter comes just before the extension.
///
/// DatasetTool.java:215-217 dereferences `Utilities.removeExtension`'s result, which is
/// null for a null path (`NullPointerException`).  Fixed in translation: null is
/// returned.
pub fn get_b_axis_input_image_file(
    input_image_file_path: Option<&str>,
    dual_axis: bool,
) -> Option<String> {
    if !dual_axis {
        return None;
    }
    let mut left_side = utilities::remove_extension(input_image_file_path)?;
    let second_extension = AxisID::Second.get_extension();
    if left_side.ends_with(&second_extension) {
        // Already is a B axis file
        return input_image_file_path.map(|path| path.to_string());
    }
    let first_extension = AxisID::First.get_extension();
    if left_side.ends_with(&first_extension) {
        // Remove A axis letter
        left_side = left_side[..left_side.len() - first_extension.len()].to_string();
    }
    // Reconstructuct the file path with the B axis letter.
    let mut divider = EXTENSION_DIVIDER;
    let mut extension = utilities::get_extension(input_image_file_path);
    if extension.is_none() {
        divider = "";
        extension = Some(String::new());
    }
    Some(left_side + &second_extension + divider + &extension.unwrap())
}

/// Java `getDatasetName(String, boolean)`.  Extract a dataset name from filePath.  Does
/// not check for errors.  Returns the dataset name; only returns null if filePath is
/// null.
pub fn get_dataset_name(file_path: Option<&str>, dual_axis: bool) -> Option<String> {
    let file_path = match file_path {
        None => return None,
        Some(file_path) if file_path.is_empty() => return Some(file_path.to_string()),
        Some(file_path) => file_path,
    };
    let left_side = utilities::remove_extension(Some(&java_io_file_get_name(file_path)));
    if !dual_axis {
        return left_side;
    }
    if let Some(left_side) = &left_side
        && left_side.encode_utf16().count() > 1
        && (left_side.ends_with(&AxisID::First.get_extension())
            || left_side.ends_with(&AxisID::Second.get_extension()))
    {
        // The last character is the ASCII axis letter.
        return Some(left_side[..left_side.len() - 1].to_string());
    }
    // No axis letter - treat as single axis
    left_side
}

/// Java `switchExtension(String)` (deprecated 5/8/2019).
pub fn switch_extension(file: &str) -> String {
    if file.ends_with(STANDARD_DATASET_EXT) {
        return file[..file.rfind('.').unwrap()].to_string() + ALTERNATE_DATASET_EXT;
    }
    if file.ends_with(ALTERNATE_DATASET_EXT) {
        return file[..file.rfind('.').unwrap()].to_string() + STANDARD_DATASET_EXT;
    }
    file.to_string()
}

/// Java `getExtension(String)` (deprecated 5/8/2019).  Returns one of the two
/// extensions, defaulting to the standard extension.
pub fn get_extension(file: Option<&str>) -> &'static str {
    let file = match file {
        None => return STANDARD_DATASET_EXT,
        Some(file) => file,
    };
    if file.ends_with(ALTERNATE_DATASET_EXT) {
        return ALTERNATE_DATASET_EXT;
    }
    STANDARD_DATASET_EXT
}

/// Java `removeMatchingBStacks(UIComponent, File[])`.  Prevent B stacks in the stackList
/// from being passed by.  Do the same to files with name collision.  See `StackInfo`.
pub fn remove_matching_b_stacks(
    component: Option<&dyn UiComponentBoundary>,
    stack_list: Option<&[Option<PathBuf>]>,
) -> Option<Vec<Rc<RefCell<StackInfo>>>> {
    let stack_list = stack_list?;
    // The filtered stack list will be returned.
    let mut filtered_stack_list: Vec<Rc<RefCell<StackInfo>>> = Vec::new();
    // Stack map allows for searching for B stacks and collisions.
    let mut stack_map: std::collections::HashMap<String, Option<Rc<RefCell<StackInfo>>>> =
        std::collections::HashMap::new();
    // Add non-dual axis stacks to the filtered stack list. Filter out matching B stacks.
    for i in 0..stack_list.len() {
        let stack = match &stack_list[i] {
            None => continue,
            Some(stack) => stack,
        };
        let stack_info = Rc::new(RefCell::new(StackInfo::new(stack.clone())));
        filtered_stack_list.push(Rc::clone(&stack_info));
        // A non-null stack always has a dataset file, so the key is never null (a null
        // `Hashtable` key would throw).
        let key = match stack_info.borrow_mut().get_key() {
            None => continue,
            Some(key) => key,
        };
        if !stack_map.contains_key(&key) {
            // Nothing matches this key - can save the current stack in stackMap
            stack_map.insert(key, Some(stack_info));
        } else {
            let mapped_stack_info = stack_map.get(&key).cloned().flatten();
            match mapped_stack_info {
                None => {
                    // This should not happen.
                    stack_map.remove(&key);
                    stack_map.insert(key, Some(stack_info));
                }
                Some(mapped_stack_info) => {
                    // Attach the new stackInfo to the saved one with the same key, or set a
                    // collision.
                    StackInfo::match_stack(&mapped_stack_info, Some(&stack_info));
                }
            }
        }
    }
    // Return filtered stack list. Stack info instances should be set up to ignore B
    // stacks and collided names. Report ignored stacks and collisions.
    let mut err_msg = String::new();
    let len = filtered_stack_list.len();
    for i in 0..len {
        filtered_stack_list[i].borrow().report(&mut err_msg);
    }
    if err_msg.len() > 0 {
        ui_harness::open_message_dialog_with_component_from_process(
            None,
            component,
            &err_msg,
            "File Name Collision",
            None,
        );
    }
    Some(filtered_stack_list)
}

/// Java private static `getAxisIDFromRawImageStackName(String)`.
///
/// `Utilities.removeExtension` can return null, which DatasetTool.java:353
/// dereferences.  Fixed in translation: a null prefix has no axis letter.
fn get_axis_id_from_raw_image_stack_name(raw_image_stack_name: Option<&str>) -> AxisID {
    if raw_image_stack_name.is_none() {
        return AxisID::Only;
    }
    let prefix = utilities::remove_extension(raw_image_stack_name).unwrap_or_default();
    if prefix.ends_with(&AxisID::First.get_extension()) {
        return AxisID::First;
    }
    if prefix.ends_with(&AxisID::Second.get_extension()) {
        return AxisID::Second;
    }
    AxisID::Only
}

/// Java `getStackFile(String, AxisID, boolean)`.  If dualAxis is true, get the stack
/// that matches axisID.  Otherwise return a file made of stackAbsPath.
pub fn get_stack_file(
    stack_abs_path: Option<&str>,
    axis_id: Option<AxisID>,
    dual_axis: bool,
) -> Option<PathBuf> {
    let stack_abs_path = match stack_abs_path {
        Some(stack_abs_path) if !stack_abs_path.is_empty() => stack_abs_path,
        _ => return None,
    };
    // Use the unchanged stackAbsPath parameter for single axis, or if it already matches
    // axisID.
    if !dual_axis
        || axis_id.is_none()
        || axis_id == Some(AxisID::Only)
        || stack_abs_path.encode_utf16().count() == 1
    {
        return Some(PathBuf::from(stack_abs_path));
    }
    // The last operand of the source's `||` chain, which also assigns `stackAxisID`.
    let stack_axis_id = get_axis_id_from_raw_image_stack_name(Some(stack_abs_path));
    if Some(stack_axis_id) == axis_id {
        return Some(PathBuf::from(stack_abs_path));
    }
    let axis_id = axis_id.unwrap();
    let ext = utilities::get_extension(Some(stack_abs_path)).unwrap_or_default();
    // If stackAbsPath is single axis but the call is looking for dual axis, return a file
    // path that matches the axisID.
    let mut stack_abs_path = utilities::remove_extension(Some(stack_abs_path)).unwrap_or_default();
    if stack_axis_id != AxisID::Only {
        // The last character is the ASCII axis letter.
        stack_abs_path = stack_abs_path[..stack_abs_path.len() - 1].to_string();
    }
    // Return a file path that matches axisID.
    Some(PathBuf::from(
        stack_abs_path + &axis_id.get_extension() + EXTENSION_DIVIDER + &ext,
    ))
}

/// Java `getStackFileName(String, String, AxisID, boolean)`.  Returns the input image
/// stack name for the axisID parameter.  Deprecated 10/19/22: this function does not
/// return the correct extension for file types like .tif and .hdf.
pub fn get_stack_file_name(
    dataset_directory: Option<&str>,
    root_name: &str,
    mut axis_id: Option<AxisID>,
    dual_axis: bool,
) -> String {
    let mut right_side = String::new();
    right_side.push_str(root_name);
    if dual_axis {
        if axis_id.is_none() || axis_id == Some(AxisID::Only) {
            axis_id = Some(AxisID::First);
        }
        right_side.push_str(&axis_id.unwrap().get_extension());
    }
    let stack_file;
    if let Some(dataset_directory) = dataset_directory {
        stack_file = java_io_file_new(
            dataset_directory,
            &(right_side.clone() + STANDARD_DATASET_EXT),
        );
    } else {
        stack_file = right_side.clone() + STANDARD_DATASET_EXT;
    }
    if Path::new(&stack_file).exists() {
        return java_io_file_get_name(&stack_file);
    }
    let alt_stack_file;
    if let Some(dataset_directory) = dataset_directory {
        alt_stack_file = java_io_file_new(
            dataset_directory,
            &(right_side.clone() + ALTERNATE_DATASET_EXT),
        );
    } else {
        alt_stack_file = right_side.clone() + ALTERNATE_DATASET_EXT;
    }
    if Path::new(&alt_stack_file).exists() {
        return java_io_file_get_name(&alt_stack_file);
    }
    java_io_file_get_name(&stack_file)
}

/// Java `getDatasetNameForRegressionTest(String, boolean)`.
pub fn get_dataset_name_for_regression_test(
    stack_name: Option<&str>,
    dual_axis: bool,
) -> Option<String> {
    let stack_name = stack_name?;
    let mut ext: Option<&str> = None;
    if stack_name.ends_with(STANDARD_DATASET_EXT) {
        ext = Some(STANDARD_DATASET_EXT);
    } else if stack_name.ends_with(ALTERNATE_DATASET_EXT) {
        ext = Some(ALTERNATE_DATASET_EXT);
    }
    let mut remove_chars = 0;
    if let Some(ext) = ext {
        remove_chars = ext.len();
    }
    if dual_axis && remove_chars > 0 {
        let ext = ext.unwrap();
        if stack_name.ends_with(&(AxisID::First.get_extension() + ext)) {
            remove_chars += AxisID::First.get_extension().len();
        } else if stack_name.ends_with(&(AxisID::Second.get_extension() + ext)) {
            remove_chars += AxisID::Second.get_extension().len();
        }
    }
    // Every removed character is ASCII.
    Some(stack_name[..stack_name.len() - remove_chars].to_string())
}

/// Java `getDatasetFile(File, boolean)`.  Gets a dataset (.edf) file that is in the same
/// directory as the stackFile.
pub fn get_dataset_file(stack_file: Option<&Path>, dual_axis: bool) -> Option<PathBuf> {
    let stack_file = stack_file?.to_string_lossy().to_string();
    let dataset_name = get_dataset_name(Some(&java_io_file_get_name(&stack_file)), dual_axis)?;
    let child = dataset_name + DataFileType::Recon.extension().unwrap();
    // `new File((String) null, child)` is `new File(child)`.
    Some(PathBuf::from(match java_io_file_get_parent(&stack_file) {
        None => child,
        Some(parent) => java_io_file_new(&parent, &child),
    }))
}

/// Java `standardizeExtension(BaseManager, AxisType, File, StringProperty)` (deprecated
/// 6/18/19).  Rename the inputFile if it is an .mrc file.  If this is dual axis, also
/// rename the other axis .mrc file.  If inputFile is actually a dataset name, rename the
/// associated .mrc file(s), if they exist.  If an .mrc file name was passed in, return
/// true and set standardizedFilePath to the renamed file path.  If the rename fails,
/// return true and keep standardizedFilePath empty so that the user will reexamine the
/// directory.  Otherwise return true.
///
/// Deprecated!  This type of rename is no longer done.  Batchruntomo handles rename to
/// give both stacks the same extension.
pub fn standardize_extension(
    manager: &'static dyn BaseManager,
    axis_type: Option<AxisType>,
    input_file: Option<&Path>,
    standardized_file_path: &mut StringProperty,
) -> bool {
    standardized_file_path.reset();
    // Nothing to do
    let input_file = match input_file {
        None => return false,
        Some(input_file) => input_file.to_string_lossy().to_string(),
    };
    // The standard extension was used - nothing to do
    let name = java_io_file_get_name(&input_file);
    if name.ends_with(STANDARD_DATASET_EXT) {
        return false;
    }
    // The entry is a dataset name, not a file
    let abs_path = java_io_file_get_absolute_path(&input_file);
    if !name.ends_with(ALTERNATE_DATASET_EXT) {
        if axis_type != Some(AxisType::DualAxis) {
            // single axis dataset
            if !Path::new(&(abs_path.clone() + STANDARD_DATASET_EXT)).exists() {
                let alt_dataset_file = abs_path.clone() + ALTERNATE_DATASET_EXT;
                if Path::new(&alt_dataset_file).exists() {
                    // rename the .mrc file associated with this dataset name
                    if rename_to_standard_extension(manager, &alt_dataset_file).is_none() {
                        // Need to reset the dataset field because rename failed
                        return true;
                    }
                }
            }
        } else if !Path::new(
            &(abs_path.clone() + &AxisID::First.get_extension() + STANDARD_DATASET_EXT),
        )
        .exists()
            && !Path::new(
                &(abs_path.clone() + &AxisID::Second.get_extension() + STANDARD_DATASET_EXT),
            )
            .exists()
        {
            // dual axis dataset
            // if the .mrc files associated with this dataset name exist, rename them
            let mut alt_dataset_file =
                abs_path.clone() + &AxisID::First.get_extension() + ALTERNATE_DATASET_EXT;
            // must return true if the rename failed, files may be in an unknown state and
            // the user should reexamine the directory.
            let mut rename_failed = false;
            if Path::new(&alt_dataset_file).exists() {
                rename_failed = rename_to_standard_extension(manager, &alt_dataset_file).is_none();
            }
            alt_dataset_file =
                abs_path.clone() + &AxisID::Second.get_extension() + ALTERNATE_DATASET_EXT;
            if Path::new(&alt_dataset_file).exists() {
                if rename_to_standard_extension(manager, &alt_dataset_file).is_none()
                    || rename_failed
                {
                    return true;
                }
            } else if rename_failed {
                return true;
            }
        }
        // no need to change the dataset field, since its a dataset name without an
        // extension and any renames succeeded.
        return false;
    }
    // The file has the alternative extension - must be renamed
    let new_name = rename_to_standard_extension(manager, &input_file);
    if let Some(new_name) = new_name {
        standardized_file_path.set(Some(&new_name));
    }
    // Handle the second .mrc file in a dual axis dataset
    if axis_type == Some(AxisType::DualAxis) {
        // Java indexes UTF-16 units; the name is handled as characters.
        let chars: Vec<char> = name.chars().collect();
        // Find out if this is an A axis or B axis file
        let mut index: i64 = match chars.iter().rposition(|&c| c == '.') {
            Some(index) => index as i64,
            None => -1,
        };
        if index != -1 {
            index -= 1;
        } else {
            index = chars.len() as i64 - 1;
        }
        // `name.charAt(-1)` throws `StringIndexOutOfBoundsException` for a name that
        // starts with its only "." (DatasetTool.java:578).  Fixed in translation: such a
        // name has no axis letter.
        let axis_id = if index < 0 {
            None
        } else {
            AxisID::get_instance_from_char(chars[index as usize])
        };
        let mut axis_id = match axis_id {
            // Has neither an a or b extension - giving up
            None => return false,
            Some(axis_id) => axis_id,
        };
        // Attempt to rename the second file
        if axis_id == AxisID::First {
            axis_id = AxisID::Second;
        } else {
            axis_id = AxisID::First;
        }
        let index = index as usize;
        let mut second_file_name = String::new();
        second_file_name
            .push_str(&(chars[..index].iter().collect::<String>() + &axis_id.get_extension()));
        if second_file_name.encode_utf16().count() < name.encode_utf16().count() {
            second_file_name.push_str(&chars[index + 1..].iter().collect::<String>());
        }
        let second_file = match java_io_file_get_parent(&input_file) {
            None => second_file_name,
            Some(parent) => java_io_file_new(&parent, &second_file_name),
        };
        rename_to_standard_extension(manager, &second_file);
    }
    true
}

/// Java private static `renameToStandardExtension(BaseManager, File)` (deprecated).
/// Rename the input file that is not an .st file.  If the file has the .mrc extension,
/// it will substitute the .st extension for the .mrc extension.  Returns the path of the
/// image file as it exists at the end of the this function.  If something has gone
/// wrong, null will be returned.  This forces the user to find the correct file.
fn rename_to_standard_extension(
    manager: &'static dyn BaseManager,
    input_file: &str,
) -> Option<String> {
    let mut name = java_io_file_get_name(input_file);
    if name.ends_with(ALTERNATE_DATASET_EXT) {
        name = name[..name.len() - ALTERNATE_DATASET_EXT.len()].to_string();
    }
    let child = name + STANDARD_DATASET_EXT;
    let new_file = match java_io_file_get_parent(input_file) {
        None => child,
        Some(parent) => java_io_file_new(&parent, &child),
    };
    // MUST fail and return false if the destination file already exists.
    eprintln!(
        "Renaming {} to {}",
        java_io_file_get_absolute_path(input_file),
        java_io_file_get_absolute_path(&new_file)
    );
    match utilities::rename_file_safely(
        Some(manager),
        Some(AxisID::Only),
        Some(Path::new(input_file)),
        Some(Path::new(&new_file)),
    ) {
        Ok(true) => return Some(java_io_file_get_absolute_path(&new_file)),
        Ok(false) => {}
        // `catch (final LockException e) { return null; }`
        Err(LogFileError::Lock(_)) => return None,
        // `catch (IOException | LogFileException e)`
        Err(e) => {
            // `e.printStackTrace()`; see etomo/util/stack_trace.rs.
            eprintln!("{}", e);
            ui_harness::open_message_dialog_from_process(
                Some(manager),
                &e.get_message(),
                "Rename Failed",
                None,
            );
            // Rename appears to have failed, but we don't absolutely know the state of the
            // files. The .st file may or may not exist now, so the user needs to reselect
            // the correct image file or dataset name.
            return None;
        }
    }
    Some(java_io_file_get_absolute_path(input_file))
}

/// Java `validateDatasetName(BaseManager, UIComponent, AxisID, File, DataFileType,
/// AxisType)`.  Validates the dataset directory, including sharing.  `input_file` is the
/// input file (such as .st file) or the data file; `axis_type` is only required for
/// reconstructions.
///
/// DatasetTool.java:686 and :692 read `inputFile.getParent()` / `getParentFile()`,
/// which are null for a bare file name: `getParent().endsWith` throws a
/// `NullPointerException`, and a null directory would throw in the overload it is
/// passed to.  Fixed in translation: a bare file name's directory is the directory
/// of its absolute path, which is where `exists()` found it.
pub fn validate_dataset_name_input_file_component(
    manager: &'static dyn BaseManager,
    ui_component: Option<&dyn UiComponentBoundary>,
    axis_id: impl Into<Option<AxisID>>,
    input_file: Option<&Path>,
    data_file_type: impl Into<Option<DataFileType>>,
    axis_type: impl Into<Option<AxisType>>,
) -> bool {
    let axis_id = axis_id.into();
    let data_file_type = data_file_type.into();
    let axis_type = axis_type.into();
    let mut error_message: Option<String> = None;
    let input_file_string = input_file.map(|input_file| input_file.to_string_lossy().to_string());
    match &input_file_string {
        None => {
            error_message = Some("No input file specified.".to_string());
        }
        Some(input_file) => {
            let path = Path::new(input_file);
            if !path.exists() {
                // `Thread.dumpStack()`; see etomo/util/stack_trace.rs for why this process
                // contributes no frames.
                StackTrace::new_with_thread(None, None).print(Some("Stack trace"), false);
                error_message = Some(format!(
                    "Input file does not exist: {}",
                    java_io_file_get_absolute_path(input_file)
                ));
            } else if !path.is_file() {
                error_message = Some(format!(
                    "{} must be a file.",
                    java_io_file_get_absolute_path(input_file)
                ));
            } else if !java_io_file_can_read(input_file) {
                error_message = Some(format!(
                    "Unreadable input file: {}",
                    java_io_file_get_absolute_path(input_file)
                ));
            }
        }
    }
    if let Some(error_message) = error_message {
        ui_harness::open_message_dialog_with_component_from_process(
            Some(manager),
            ui_component,
            &error_message,
            MESSAGE_TITLE,
            axis_id,
        );
        return false;
    }
    let input_file = input_file_string.unwrap();
    let input_file_name = java_io_file_get_name(&input_file);
    let ext_index = input_file_name.rfind('.');
    let mut input_file_root = input_file_name.clone();
    if let Some(ext_index) = ext_index {
        input_file_root = input_file_name[..ext_index].to_string();
    }
    // Check for embedded spaces
    if EMBEDDED_SPACES.is_match(&input_file_name) {
        ui_harness::open_message_dialog_with_component_from_process(
            Some(manager),
            ui_component,
            &format!(
                "The dataset name cannot contain embedded spaces: {}",
                java_io_file_get_absolute_path(&input_file)
            ),
            MESSAGE_TITLE,
            axis_id,
        );
        return false;
    }
    let parent = java_io_file_get_parent(&input_file).unwrap_or_else(|| {
        java_io_file_get_parent(&java_io_file_get_absolute_path(&input_file)).unwrap_or_default()
    });
    if parent.ends_with(' ') {
        ui_harness::open_message_dialog_with_component_from_process(
            Some(manager),
            ui_component,
            &format!(
                "The dataset directory cannot end in a space: {}",
                java_io_file_get_absolute_path(&input_file)
            ),
            MESSAGE_TITLE,
            axis_id,
        );
        return false;
    }
    let directory = PathBuf::from(parent);
    validate_dataset_name(
        manager,
        ui_component,
        axis_id,
        &directory,
        Some(&input_file_root),
        data_file_type,
        axis_type,
        false,
    )
}

/// Java `validateDatasetName(BaseManager, AxisID, File, DataFileType, AxisType)`.
/// Validates the dataset directory, including sharing.
///
/// DatasetTool.java:715 passes `inputFile.getParentFile()`, null for a bare file name,
/// to an overload that dereferences it.  Fixed in translation as in
/// [`validate_dataset_name_input_file_component`].
pub fn validate_dataset_name_input_file(
    manager: &'static dyn BaseManager,
    axis_id: impl Into<Option<AxisID>>,
    input_file: Option<&Path>,
    data_file_type: impl Into<Option<DataFileType>>,
    axis_type: impl Into<Option<AxisType>>,
) -> bool {
    let axis_id = axis_id.into();
    let input_file = match input_file {
        None => {
            ui_harness::open_message_dialog_with_component_from_process(
                Some(manager),
                None,
                "The input file is empty.",
                MESSAGE_TITLE,
                axis_id,
            );
            return false;
        }
        Some(input_file) => input_file.to_string_lossy().to_string(),
    };
    let parent = java_io_file_get_parent(&input_file).unwrap_or_else(|| {
        java_io_file_get_parent(&java_io_file_get_absolute_path(&input_file)).unwrap_or_default()
    });
    validate_dataset_name(
        manager,
        None,
        axis_id,
        Path::new(&parent),
        Some(&java_io_file_get_name(&input_file)),
        data_file_type,
        axis_type,
        false,
    )
}

/// Java `validateDatasetName(BaseManager, AxisID, File, String, DataFileType, AxisType,
/// boolean)`.  Validates the dataset directory, including sharing.
pub fn validate_dataset_name_directory(
    manager: &'static dyn BaseManager,
    axis_id: impl Into<Option<AxisID>>,
    directory: &Path,
    input_file_root: Option<&str>,
    data_file_type: impl Into<Option<DataFileType>>,
    axis_type: impl Into<Option<AxisType>>,
    dataset_name: bool,
) -> bool {
    validate_dataset_name(
        manager,
        None,
        axis_id,
        directory,
        input_file_root,
        data_file_type,
        axis_type,
        dataset_name,
    )
}

/// Java `validateDatasetName(BaseManager, UIComponent, AxisID, File, String,
/// DataFileType, AxisType, boolean)`.  Validates the dataset directory, including
/// sharing.  `directory` is the directory in which the new dataset will be created,
/// `input_file_root` the root name of the new dataset, `data_file_type` the type of the
/// new dataset, and `axis_type` the axis type of the new dataset (only required for
/// reconstructions).
pub fn validate_dataset_name(
    manager: &'static dyn BaseManager,
    ui_component: Option<&dyn UiComponentBoundary>,
    axis_id: impl Into<Option<AxisID>>,
    directory: &Path,
    input_file_root: Option<&str>,
    data_file_type: impl Into<Option<DataFileType>>,
    axis_type: impl Into<Option<AxisType>>,
    dataset_name: bool,
) -> bool {
    let axis_id = axis_id.into();
    let data_file_type = data_file_type.into();
    let axis_type = axis_type.into();
    let directory_string = directory.to_string_lossy().to_string();
    let mut error_message: Option<String> = None;
    if !directory.exists() {
        error_message = Some(format!(
            "Directory does not exist: {}",
            java_io_file_get_absolute_path(&directory_string)
        ));
    } else if !directory.is_dir() {
        error_message = Some(format!(
            "{} must be a directory.",
            java_io_file_get_absolute_path(&directory_string)
        ));
    } else if !java_io_file_can_read(&directory_string) {
        error_message = Some(format!(
            "Unreadable directory: {}",
            java_io_file_get_absolute_path(&directory_string)
        ));
    } else if !java_io_file_can_write(&directory_string) {
        error_message = Some(format!(
            "Unwritable directory: {}",
            java_io_file_get_absolute_path(&directory_string)
        ));
    } else if data_file_type.is_none() {
        error_message = Some("No data file type specified".to_string());
    } else if input_file_root.is_none() {
        error_message = Some("Missing dataset name.".to_string());
    }
    // Check for embedded spaces
    else if EMBEDDED_SPACES.is_match(input_file_root.unwrap()) {
        error_message = Some(format!(
            "The dataset name cannot contain embedded spaces: {}",
            input_file_root.unwrap()
        ));
    } else {
        let data_file_type = data_file_type.unwrap();
        let input_file_root = input_file_root.unwrap();
        // `directory.listFiles(new DataFileFilter(true))`: null when the directory
        // cannot be listed.
        let filter = DataFileFilter::new_with_files_only(true);
        let file_list: Option<Vec<PathBuf>> = std::fs::read_dir(directory).ok().map(|entries| {
            entries
                .filter_map(|entry| entry.ok())
                .map(|entry| entry.path())
                .filter(|path| filter.accept(path))
                .collect()
        });
        let file_list = match file_list {
            Some(file_list) if file_list.len() > 0 => file_list,
            _ => return true,
        };
        for i in 0..file_list.len() {
            let file_name = java_io_file_get_name(&file_list[i].to_string_lossy());
            let can_share;
            if data_file_type.has_axis_type() {
                let mut file_axis_type: Option<AxisType> = None;
                // If the existing data file does not have an axis type, the axis types
                // don't matter.  (The filter only accepts data files, so the type is
                // never null here.)
                if DataFileType::get_instance(Some(&file_name))
                    .is_some_and(|file_type| file_type.has_axis_type())
                {
                    match LogFile::get_instance_file(
                        Some(file_list[i].as_path()),
                        Some(manager.get_emergency_monitor(axis_id)),
                    )
                    .and_then(|log_file| log_file.get_line_containing("Setup.AxisType"))
                    {
                        Ok(line) => {
                            file_axis_type = AxisType::get_instance(line.as_deref());
                        }
                        // `catch (final LogFileException | IOException e)` and
                        // `catch (final LockException e)`: both print the trace.
                        Err(e) => {
                            eprintln!("{}", e);
                        }
                    }
                }
                can_share = can_share_with_axis_type(
                    data_file_type,
                    Some(input_file_root),
                    axis_type,
                    Some(&file_name),
                    file_axis_type,
                    dataset_name,
                );
            } else {
                can_share = can_share_with(
                    data_file_type,
                    Some(input_file_root),
                    Some(&file_name),
                    dataset_name,
                );
            }
            if !can_share {
                error_message = Some(format!(
                    "Cannot create {} dataset {} in {} because {} cannot share a directory with this new dataset.  Please select another directory.",
                    data_file_type, input_file_root, directory_string, file_name
                ));
                break;
            }
        }
    }
    let error_message = match error_message {
        None => return true,
        Some(error_message) => error_message,
    };
    ui_harness::open_message_dialog_with_component_from_process(
        Some(manager),
        ui_component,
        &error_message,
        MESSAGE_TITLE,
        axis_id,
    );
    false
}

/// Java package-private static `canShareWith(DataFileType, String, String, boolean)`.
/// Returns true if newDataFileType can share a directory with another data file
/// (existingDataFileName).  This function cannot allow .edf files to share a directory.
/// Call canShareWith(...AxisType...) to allow .edf file sharing.  `new_root` is the root
/// of the project to be created; `existing_data_file_name` a data file in the directory
/// to be shared.
pub(crate) fn can_share_with(
    new_data_file_type: DataFileType,
    new_root: Option<&str>,
    existing_data_file_name: Option<&str>,
    dataset_name: bool,
) -> bool {
    if new_data_file_type.has_axis_type() {
        // handle incorrect data file type
        // `new InvalidParameterException(...).printStackTrace()`; see
        // etomo/util/stack_trace.rs.
        eprintln!(
            "etomo.util.InvalidParameterException: Warning: unable to share directories containing {} file types.  Wrong canShareWith called.  Calling correct canShareWith without axis type information.",
            new_data_file_type
        );
        return can_share_with_axis_type(
            new_data_file_type,
            new_root,
            None,
            existing_data_file_name,
            None,
            dataset_name,
        );
    }
    // Get the type of the existing data file
    let existing_data_file_type = match DataFileType::get_instance(existing_data_file_name) {
        // Its not a data file
        None => return true,
        Some(existing_data_file_type) => existing_data_file_type,
    };
    let existing_data_file_name = existing_data_file_name.unwrap();
    // Get existing data file root
    let ext_index = existing_data_file_name.rfind('.');
    let mut root = existing_data_file_name;
    if let Some(ext_index) = ext_index {
        root = &existing_data_file_name[..ext_index];
    }
    if new_data_file_type == DataFileType::Join {
        if existing_data_file_type == DataFileType::SerialSections {
            return false;
        }
        return true;
    }
    if new_data_file_type == DataFileType::Parallel {
        return true;
    }
    if new_data_file_type == DataFileType::Peet {
        if existing_data_file_type == DataFileType::Peet {
            // Can share a PEET directory if the root is the same
            return Some(root) == new_root;
        }
        if existing_data_file_type == DataFileType::SerialSections {
            return false;
        }
        return true;
    }
    if new_data_file_type == DataFileType::SerialSections {
        if existing_data_file_type == DataFileType::Recon
            || existing_data_file_type == DataFileType::Join
            || existing_data_file_type == DataFileType::Peet
        {
            return false;
        }
        if existing_data_file_type == DataFileType::SerialSections {
            // Can share a SERIAL_SECTIONS directory if the root is the same
            return Some(root) == new_root;
        }
        return true;
    }
    if new_data_file_type == DataFileType::Tools {
        if existing_data_file_type == DataFileType::Recon
            || existing_data_file_type == DataFileType::Join
            || existing_data_file_type == DataFileType::Peet
            || existing_data_file_type == DataFileType::SerialSections
        {
            // Can share a major project directory, if the root is different
            return Some(root) != new_root;
        }
        return true;
    }
    false
}

/// Java package-private static `canShareWith(DataFileType, String, AxisType, String,
/// AxisType, boolean)`.  CanShareWith function for DataFileTypes that have an axis type
/// (.edf).  Returns true if newDataFileType can share a directory with another data
/// file (existingDataFileName).  The Axis Type parameters can be null if
/// existingDataFileName is not an .edf file.
///
/// DatasetTool.java:927 falls back to the other overload as
/// `canShareWith(newDataFileType, existingDataFileName, newRoot, datasetName)`, with the
/// new root and the existing file name swapped against that overload's
/// `(newRoot, existingDataFileName)` parameters.  Fixed in translation: the arguments
/// are passed in the order the called overload declares.
pub(crate) fn can_share_with_axis_type(
    new_data_file_type: DataFileType,
    new_root: Option<&str>,
    new_axis_type: Option<AxisType>,
    existing_data_file_name: Option<&str>,
    existing_axis_type: Option<AxisType>,
    dataset_name: bool,
) -> bool {
    // check new root
    let mut new_root = match new_root {
        Some(new_root) if !java_lang_string_matches_whitespace(new_root) => new_root.to_string(),
        _ => return false,
    };
    // handle incorrect data file types
    if !new_data_file_type.has_axis_type() {
        // `new IllegalStateException(...).printStackTrace()`.
        eprintln!(
            "java.lang.IllegalStateException: Wrong canShareWith function called - {} does not have an axis type.  Calling correct canShareWith.  newRoot:{}",
            new_data_file_type, new_root
        );
        return can_share_with(
            new_data_file_type,
            Some(&new_root),
            existing_data_file_name,
            dataset_name,
        );
    }
    // Get the type of the existing data file
    let existing_data_file_type = match DataFileType::get_instance(existing_data_file_name) {
        // Its not a data file
        None => return true,
        Some(existing_data_file_type) => existing_data_file_type,
    };
    let existing_data_file_name = existing_data_file_name.unwrap();
    // Get existing data file root
    let ext_index = existing_data_file_name.rfind('.');
    let mut root = existing_data_file_name;
    if let Some(ext_index) = ext_index {
        root = &existing_data_file_name[..ext_index];
    }
    // Can't share if the newAxisType is missing
    if (new_axis_type.is_none() || new_axis_type == Some(AxisType::NotSet))
        && existing_data_file_type.has_axis_type()
    {
        // `new InvalidParameterException(...).printStackTrace()`.
        eprintln!(
            "etomo.util.InvalidParameterException: Warning: dual and single axis reconstructions of the same stack cannot share a directory.\nNewAxisType wasn't set for a {} data file.  hasAxisType:{},existingDataFileName:{},newRoot:{},root:{}",
            new_data_file_type,
            new_data_file_type.has_axis_type(),
            existing_data_file_name,
            new_root,
            root
        );
        return false;
    }
    // If the data file type uses the axis letter, and the new root (newRoot) ends in "a"
    // or "b", strip the axis letter (BBa -> BB, jawa -> jaw).
    let mut stripped = false;
    let mut stripped_letter: Option<String> = None;
    if new_data_file_type.has_axis_type() && (new_root.ends_with('a') || new_root.ends_with('b')) {
        stripped = true;
        if new_axis_type == Some(AxisType::SingleAxis) {
            // Record the letter that is stripped because b and a may not be compatible for
            // single axis data file types.
            stripped_letter = Some(new_root[new_root.len() - 1..].to_string());
        }
        new_root = new_root[..new_root.len() - 1].to_string();
    }
    // check for sharing with another .edf file
    // Can share a RECON directory if the root is the same
    if existing_data_file_type == DataFileType::Recon {
        // Can't share if the existingAxisType is missing
        // Don't have enough information to avoid matching stacks with similar names
        if (existing_axis_type.is_none() || existing_axis_type == Some(AxisType::NotSet))
            && existing_data_file_type.has_axis_type()
        {
            // `new InvalidAlgorithmParameterException(...).printStackTrace()`.
            eprintln!(
                "java.security.InvalidAlgorithmParameterException: Warning: dual and single axis reconstructions of the same stack cannot share a directory.\nExistingAxisType wasn't set for a {} data file.  hasAxisType:{},existingDataFileName:{},newRoot:{},root:{}",
                existing_data_file_type,
                new_data_file_type.has_axis_type(),
                existing_data_file_name,
                new_root,
                root
            );
            return false;
        }
        // Match the root without an axis letter
        if root == new_root {
            if stripped {
                if existing_axis_type == Some(AxisType::SingleAxis) {
                    // The existing data file is associated with root.st, while the new .edf
                    // will be associated with roota.st/rootb.st, so they cannot share the
                    // directory.
                    return false;
                }
                return true;
            }
            if existing_axis_type == Some(AxisType::DualAxis) && !dataset_name {
                // The existing data file is associated with roota.st/rootb.st, while the
                // new .edf with be associated with root.st, so they cannot share the
                // directory.
                return false;
            }
            // If it is dual axis and it is the dataset name, then root is associated with
            // roota.st/rootb.st.
            return true;
        }
        // Don't add an axis letter to a root that didn't originally have one
        if !stripped {
            return false;
        }
        // single axis can match the same single axis .edf file, or a dual axis file
        if new_axis_type == Some(AxisType::SingleAxis) {
            if let Some(stripped_letter) = &stripped_letter {
                if root == new_root.clone() + stripped_letter {
                    if existing_axis_type == Some(AxisType::DualAxis) {
                        // The existing data file is associated with rootxa.st/rootxb.st,
                        // while the new .edf with be associated with rootx.st, so they
                        // cannot share the directory.
                        return false;
                    }
                    return true;
                }
                return false;
            }
            // `new IllegalStateException(...).printStackTrace()`.
            eprintln!(
                "java.lang.IllegalStateException: Letter was stripped, but not recorded.  usesAxisID:{},existingDataFileName:{},newRoot:{},root:{}",
                new_data_file_type.has_axis_type(),
                existing_data_file_name,
                new_root,
                root
            );
            return false;
        }
        // Dual axis can match the same dual axis file, or both single axis files
        // Add the stripped axis letters back to find a match with root. This is because
        // dual can share a dataset with single or dual if they use the same stack(s).
        if root == new_root.clone() + &AxisID::First.get_extension() {
            if !dataset_name {
                if existing_axis_type == Some(AxisType::DualAxis) {
                    // The existing data file is associated with rootaa.st/rootab.st, while
                    // the new .edf with be associated with roota.st, so they cannot share
                    // the directory.
                    return false;
                }
            } else if existing_axis_type == Some(AxisType::SingleAxis) {
                // The existing data file is associated with roota.edf/roota.st, while the
                // new .edf with be associated with rootaa.st/rootab.st, so they cannot
                // share the directory.
                return false;
            }
            // If its a dataset name then roota is associated with rootaa.st/rootab.st.
            // If its not a dataset name then roota is associated with roota.edf/roota.st.
            return true;
        }
        if root == new_root.clone() + &AxisID::Second.get_extension() {
            if !dataset_name {
                if existing_axis_type == Some(AxisType::DualAxis) {
                    // The existing data file is associated with rootba.st/rootbb.st, while
                    // the new .edf with be associated with rootb.st, so they cannot share
                    // the directory.
                    return false;
                }
            } else if existing_axis_type == Some(AxisType::SingleAxis) {
                // The existing data file is associated with roota.edf/roota.st, while the
                // new .edf with be associated with rootaa.st/rootab.st, so they cannot
                // share the directory.
                return false;
            }
            // If its a dataset name then rootb is associated with rootba.st/rootbb.st.
            // If its not a dataset name then rootb is associated with rootb.edf/rootb.st.
            return true;
        }
        return false;
    }
    if existing_data_file_type == DataFileType::SerialSections {
        return false;
    }
    true
}

/// Java `validateViewType(ViewType, String, String, BaseManager, UIComponent, AxisID)`.
/// Pops up an error message and returns false if the view type doesn't match the stack
/// type.
pub fn validate_view_type(
    view_type: ViewType,
    absolute_path: Option<&str>,
    stack_file_name: Option<&str>,
    manager: &'static dyn BaseManager,
    ui_component: Option<&dyn UiComponentBoundary>,
    axis_id: AxisID,
) -> bool {
    if stack_file_name.is_none() {
        return true;
    }
    // `Montagesize::get_instance_in_dir` takes the location as `&str`; a null
    // `absolutePath` is passed as "null", as `MRCHeader`'s translation does with its
    // null location (Java's `Utilities.getFile(null, name)` would be
    // `new File((String) null, name)`).
    let montagesize =
        Montagesize::get_instance_in_dir(absolute_path.unwrap_or("null"), stack_file_name, axis_id);
    // Run montagesize without the piece list file to see what the stack looks like.
    montagesize.set_ignore_piece_list_file(true);
    let mut exit_value = read_montagesize(&montagesize, manager);
    if !montagesize.piece_list_file_exists() {
        if exit_value == 0 {
            return validate_montage(
                view_type,
                &montagesize,
                absolute_path,
                stack_file_name,
                manager,
                ui_component,
                axis_id,
            );
        } else if exit_value == 1 && view_type == ViewType::Montage {
            ui_harness::open_message_dialog_with_component_from_process(
                Some(manager),
                ui_component,
                "The dataset is not a montage.  Please select single frame type.",
                "Incorrect Frame Type",
                Some(axis_id),
            );
            return false;
        }
    }
    // Ignored existing piece list file.
    else if exit_value == 0 {
        return validate_montage(
            view_type,
            &montagesize,
            absolute_path,
            stack_file_name,
            manager,
            ui_component,
            axis_id,
        );
    } else if exit_value == 1 {
        // No piece list information available in the stack - run montagesize with with
        montagesize.set_ignore_piece_list_file(false);
        exit_value = read_montagesize(&montagesize, manager);
        if exit_value == 0 {
            return validate_montage(
                view_type,
                &montagesize,
                absolute_path,
                stack_file_name,
                manager,
                ui_component,
                axis_id,
            );
        } else if exit_value == 2 || exit_value == 3 {
            // If they selected single view, go with that and ignore the piece list file
            if view_type == ViewType::Montage {
                ui_harness::open_message_dialog_with_component_from_process(
                    Some(manager),
                    ui_component,
                    "The piece list file associated with this dataset does not match and the stack does not contain piece list information.  Please select single frame type.",
                    "Incorrect Frame Type",
                    Some(axis_id),
                );
                return false;
            }
        }
    }
    true
}

/// Java private static `readMontagesize(Montagesize, BaseManager)`.
fn read_montagesize(montagesize: &Montagesize, manager: &'static dyn BaseManager) -> i32 {
    let exit_value;
    match montagesize.read(manager) {
        Ok(_) => montagesize.get_exit_value(),
        // `catch (InvalidParameterException e)` and `catch (IOException e)`: the two
        // arms are identical.
        Err(_) => {
            exit_value = montagesize.get_exit_value();
            if montagesize.get_exit_value() == 0 {
                return 1;
            }
            exit_value
        }
    }
}

/// Java private static `validateMontage(ViewType, Montagesize, String, String,
/// BaseManager, UIComponent, AxisID)`.
///
/// `MRCHeader.read` throws `InvalidParameterException` (title "Invalid Parameter
/// Exception") or `IOException` (title "IO Exception"); its translation returns either
/// one's message in the same `Err`, so the IO title is used for both.
fn validate_montage(
    view_type: ViewType,
    montagesize: &Montagesize,
    absolute_path: Option<&str>,
    stack_file_name: Option<&str>,
    manager: &'static dyn BaseManager,
    ui_component: Option<&dyn UiComponentBoundary>,
    axis_id: AxisID,
) -> bool {
    if view_type != ViewType::Montage {
        // Currently 1x1 montage works with single view, so only fail if X or Y are
        // different.
        let header =
            match MRCHeader::get_instance_in_dir(absolute_path, stack_file_name, Some(axis_id)) {
                None => return true,
                Some(header) => header,
            };
        let mut header = header.borrow_mut();
        match header.read_with_manager(manager) {
            Ok(false) => {
                ui_harness::open_message_dialog_from_process(
                    Some(manager),
                    "File does not exist.",
                    "Entry Error",
                    Some(axis_id),
                );
                return false;
            }
            Ok(true) => {}
            Err(message) => {
                ui_harness::open_message_dialog_from_process(
                    Some(manager),
                    &message,
                    "IO Exception",
                    Some(axis_id),
                );
                return false;
            }
        }
        if montagesize.get_x().get_int() > header.get_n_columns()
            || montagesize.get_y().get_int() > header.get_n_rows()
        {
            ui_harness::open_message_dialog_with_component_from_process(
                Some(manager),
                ui_component,
                "The dataset is a montage.  Please select montage frame type.",
                "Incorrect Frame Type",
                Some(axis_id),
            );
            return false;
        }
    }
    true
}

/// Java `isOneBy(String, String, BaseManager, AxisID)`.  Returns true if the stack is a
/// not a montage, or is a 1xn or nx1 montage.
pub fn is_one_by(
    absolute_path: Option<&str>,
    stack_file_name: Option<&str>,
    manager: &'static dyn BaseManager,
    axis_id: AxisID,
) -> bool {
    // montagesize
    let montagesize =
        Montagesize::get_instance_in_dir(absolute_path.unwrap_or("null"), stack_file_name, axis_id);
    // Run montagesize without the piece list file to see what the stack looks like.
    montagesize.set_ignore_piece_list_file(true);
    let exit_value = read_montagesize(&montagesize, manager);
    if exit_value == 1 {
        return true;
    }
    // header
    let header = match MRCHeader::get_instance_in_dir(absolute_path, stack_file_name, Some(axis_id))
    {
        None => return true,
        Some(header) => header,
    };
    let mut header = header.borrow_mut();
    match header.read_with_manager(manager) {
        Ok(false) => return true,
        Ok(true) => {}
        // `catch (InvalidParameterException | IOException except)`:
        // `except.printStackTrace()`.
        Err(message) => {
            eprintln!("{}", message);
            return true;
        }
    }
    if montagesize.get_x().get_int() == header.get_n_columns()
        || montagesize.get_y().get_int() == header.get_n_rows()
    {
        return true;
    }
    false
}

/// Java `validateTiltAngle(BaseManager, AxisID, String, AxisID, boolean, String,
/// String)`.
pub fn validate_tilt_angle(
    manager: &'static dyn BaseManager,
    message_axis_id: AxisID,
    error_title: Option<&str>,
    axis_id: Option<AxisID>,
    manual: bool,
    angle: Option<&str>,
    increment: Option<&str>,
) -> bool {
    if !manual {
        return true;
    }
    let axis_descr = get_axis_descr(axis_id);
    let mut message: Option<&str> = None;
    if angle.is_none_or(java_lang_string_matches_whitespace) {
        message = Some("Starting angle cannot be empty");
    } else if increment.is_none_or(java_lang_string_matches_whitespace) {
        message = Some("Increment cannot be empty");
    }
    if let Some(message) = message {
        ui_harness::open_message_dialog_from_process(
            Some(manager),
            &(message.to_string() + axis_descr.unwrap_or(".")),
            // A null Java title is shown as an empty title.
            error_title.unwrap_or(""),
            Some(message_axis_id),
        );
        return false;
    }
    true
}

/// Java private static `getAxisDescr(AxisID)`.
fn get_axis_descr(axis_id: Option<AxisID>) -> Option<&'static str> {
    if axis_id == Some(AxisID::First) {
        return Some(" in Axis A.");
    }
    if axis_id == Some(AxisID::Second) {
        return Some(" in Axis B.");
    }
    None
}

/// Java private static final nested class `StackNameFilter implements FilenameFilter`.
struct StackNameFilter {
    /// Java private final field `stackName`.
    stack_name: Option<String>,
}

impl StackNameFilter {
    /// Java `StackNameFilter(String)`.
    fn new(stack_name: Option<String>) -> StackNameFilter {
        StackNameFilter { stack_name }
    }

    /// Java `accept(File, String)`.
    fn accept(&self, _dir: &Path, name: &str) -> bool {
        match &self.stack_name {
            None => false,
            Some(stack_name) => stack_name == name,
        }
    }
}

/// Java public static final nested class `StackInfo`.  Stores a stack file and
/// information about other stack file with the same root name, path, and axis type.  For
/// axis type, any stack ending in the axis letters "a" or "b" is considered to be dual
/// axis type.  A matching stack is a dual axis type stack file which has the same root
/// name, path, and axis type - but a different axis letter.  The assumption is that it
/// will be part of the same tomogram, so the B matching stack is not returned by
/// getStack.
#[derive(Debug)]
pub struct StackInfo {
    /// Java private field `stack`.
    stack: Option<PathBuf>,
    /// Java private field `axisID`, initialised to null.
    axis_id: Option<AxisID>,
    /// Java private field `key`, initialised to null.
    key: Option<String>,
    /// Java private field `collision`.  A collision is caused by another file with the
    /// same root name, path, axis type, and axis letter (for dual axis type), but a
    /// different extension.
    collision: bool,
    /// Java private field `matchingStack`, initialised to null.  See the module header
    /// for why this is `Weak`.
    matching_stack: Option<Weak<RefCell<StackInfo>>>,
    /// Java private field `dirChecked`.
    dir_checked: bool,
    /// Java private field `dirMatchingStack`.
    dir_matching_stack: bool,
}

impl StackInfo {
    /// Java package-private `StackInfo(File)`.
    pub(crate) fn new(stack: PathBuf) -> StackInfo {
        StackInfo {
            stack: Some(stack),
            axis_id: None,
            key: None,
            collision: false,
            matching_stack: None,
            dir_checked: false,
            dir_matching_stack: false,
        }
    }

    /// Java private `getKey()`.
    fn get_key(&mut self) -> Option<String> {
        self.set_key();
        self.key.clone()
    }

    /// Java private `match(StackInfo)`.  Matches two files against each other.
    fn match_stack(this: &Rc<RefCell<StackInfo>>, stack_info: Option<&Rc<RefCell<StackInfo>>>) {
        let stack_info = match stack_info {
            None => {
                eprintln!("Warning: empty stackInfo parameter");
                // `Thread.dumpStack()`; see etomo/util/stack_trace.rs.
                StackTrace::new_with_thread(None, None).print(Some("Stack trace"), false);
                return;
            }
            Some(stack_info) => stack_info,
        };
        this.borrow_mut().set_key();
        stack_info.borrow_mut().set_key();
        let key = this.borrow().key.clone();
        let other_key = stack_info.borrow().key.clone();
        if key.is_none() || key != other_key {
            eprintln!(
                "Warning: key, {} is not equal to {}",
                key.as_deref().unwrap_or("null"),
                other_key.as_deref().unwrap_or("null")
            );
            // `Thread.dumpStack()`.
            StackTrace::new_with_thread(None, None).print(Some("Stack trace"), false);
            return;
        }
        this.borrow_mut().set_axis_id();
        stack_info.borrow_mut().set_axis_id();
        let axis_id = this.borrow().axis_id;
        if axis_id.is_none()
            || axis_id == Some(AxisID::Only)
            || axis_id == stack_info.borrow().axis_id
        {
            // this stack name is identical to another stack name, but with a different
            // extension. This is a name collision. Files with a name collision can not be
            // loaded together. Neither file will be added to the table.
            StackInfo::set_collision(this);
            StackInfo::set_collision(stack_info);
        } else {
            // Both stacks are dual axis, and they have different axis letters. Match the
            // stacks to each other.
            this.borrow_mut().matching_stack = Some(Rc::downgrade(stack_info));
            stack_info.borrow_mut().matching_stack = Some(Rc::downgrade(this));
        }
    }

    /// Java private `setCollision()`.
    fn set_collision(this: &Rc<RefCell<StackInfo>>) {
        let matching_stack = {
            let mut stack_info = this.borrow_mut();
            stack_info.collision = true;
            // Release the matched stack. It is now considered to NOT be part of a dual axis
            // tomogram.
            stack_info.matching_stack.take()
        };
        if let Some(matching_stack) = matching_stack
            && let Some(matching_stack) = matching_stack.upgrade()
        {
            matching_stack.borrow_mut().matching_stack = None;
        }
    }

    /// Java private `report(StringBuilder)`.
    fn report(&self, err_msg: &mut String) {
        let stack = self
            .stack
            .as_ref()
            .map(|stack| stack.to_string_lossy().to_string())
            .unwrap_or_else(|| "null".to_string());
        if self.collision {
            err_msg.push_str(&format!(
                "Warning: Name collision: {}.  ",
                java_io_file_get_absolute_path(&stack)
            ));
        } else if self.matching_stack.is_some() && self.axis_id == Some(AxisID::Second) {
            eprintln!(
                "\nINFO: Assuming {} is a B axis stack.\nIt will not be added to the table.  To add it to the table, open it separately from the corresponding A axis stack,\nand uncheck Dual Axis checkbox for both stacks.",
                stack
            );
        }
    }

    /// Java `getStack()`.  Returns a stack if the stack can be put into the table.  For
    /// stacks with a name collision, and B stacks that have a matching stack, null is
    /// returned.
    pub fn get_stack(&self) -> Option<PathBuf> {
        if !self.collision
            && (self.matching_stack.is_none() || self.axis_id != Some(AxisID::Second))
        {
            return self.stack.clone();
        }
        None
    }

    /// Java `isMatched()`.  Returns true if a matching stack was found in the selection
    /// list, or if a matching is in the same directory as the stack.
    pub fn is_matched(&mut self) -> bool {
        if self.matching_stack.is_some() {
            return true;
        }
        self.is_dir_matching_stack()
    }

    /// Java private `isDirMatchingStack()`.
    ///
    /// DatasetTool.java:1415 calls `list` on `stack.getParentFile()`, which is null for
    /// a bare file name (`NullPointerException`).  Fixed in translation: such a stack is
    /// looked for in the directory of its absolute path.
    fn is_dir_matching_stack(&mut self) -> bool {
        if self.dir_checked {
            return self.dir_matching_stack;
        }
        self.dir_checked = true;
        let stack = match &self.stack {
            None => return false,
            Some(stack) => stack.to_string_lossy().to_string(),
        };
        // Check for the matching stack name in the directory containing the stack.
        let stack_name = java_io_file_get_name(&stack);
        let axis_id = get_axis_id_from_raw_image_stack_name(Some(&stack_name));
        if axis_id == AxisID::Only {
            return false;
        }
        let dataset_name = get_dataset_name(Some(&java_io_file_get_name(&stack)), true);
        let extension = match Extension::get_instance(&stack_name) {
            None => return false,
            Some(extension) => extension,
        };
        let ext = EXTENSION_DIVIDER.to_string() + &extension.to_string();
        let matching_stack_name = dataset_name.unwrap_or_else(|| "null".to_string())
            + &(if axis_id == AxisID::First {
                AxisID::Second.get_extension()
            } else {
                AxisID::First.get_extension()
            })
            + &ext;
        let parent = java_io_file_get_parent(&stack)
            .or_else(|| java_io_file_get_parent(&java_io_file_get_absolute_path(&stack)))
            .unwrap_or_default();
        let filter = StackNameFilter::new(Some(matching_stack_name));
        // `File.list(FilenameFilter)`: null when the directory cannot be listed.
        let list: Option<Vec<String>> = std::fs::read_dir(&parent).ok().map(|entries| {
            entries
                .filter_map(|entry| entry.ok())
                .map(|entry| entry.file_name().to_string_lossy().to_string())
                .filter(|name| filter.accept(Path::new(&parent), name))
                .collect()
        });
        if list.is_none_or(|list| list.len() == 0) {
            return false;
        }
        self.dir_matching_stack = true;
        true
    }

    /// Java `isSingleAxis()`.
    pub fn is_single_axis(&mut self) -> bool {
        self.set_axis_id();
        self.axis_id.is_none() || self.axis_id == Some(AxisID::Only)
    }

    /// Java private `setAxisID()`.
    fn set_axis_id(&mut self) {
        if self.axis_id.is_none() {
            if let Some(stack) = &self.stack {
                self.axis_id = Some(get_axis_id_from_raw_image_stack_name(Some(
                    &java_io_file_get_name(&stack.to_string_lossy()),
                )));
            }
        }
    }

    /// Java private `setKey()`.  xxxa.st, xxxa.mrc, xxxb.st, and xxxb.mrc have the same
    /// key.  A different key will be generated for .xxx.st and xxx.mrc
    fn set_key(&mut self) {
        self.set_axis_id();
        if self.key.is_none() {
            let axis_type;
            if self.axis_id.is_none() || self.axis_id == Some(AxisID::Only) {
                axis_type = AxisType::SingleAxis;
            } else {
                axis_type = AxisType::DualAxis;
            }
            let mut dataset_file: Option<PathBuf> = None;
            if let Some(stack) = &self.stack {
                dataset_file =
                    get_dataset_file(Some(stack.as_path()), axis_type == AxisType::DualAxis);
            }
            if let Some(dataset_file) = dataset_file {
                self.key = Some(
                    java_io_file_get_absolute_path(&dataset_file.to_string_lossy())
                        + &axis_type.to_string(),
                );
            }
        }
    }
}
