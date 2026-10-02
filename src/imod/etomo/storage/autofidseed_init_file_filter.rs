//! `IMOD/Etomo/src/etomo/storage/AutofidseedInitFileFilter.java`.
//!
//! A `java.io.FileFilter` accepting the initial bead-finding model autofidseed
//! leaves in its directory (`afs<n>.pkmod`).  The Java static methods are module
//! functions.

use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use regex::Regex;

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::file_type;
use crate::imod::etomo::ui::swing::ui_harness;
use crate::imod::etomo::util::utilities;

/// Java `rcsid`.
pub const RCSID: &str = "$Id:$";

/// Java private static final `NAME`.
const NAME: &str = "afs";
/// Java private static final `EXT`.
const EXT: &str = ".pkmod";

/// Java `NAME + "\\d+\\" + EXT` as a whole-string match (`String.matches`).
/// Java's `\d` is ASCII only.
static PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(&format!(r"^(?:{}[0-9]+\{})$", NAME, EXT)).unwrap());

/// Java `public final class AutofidseedInitFileFilter implements FileFilter`.
pub struct AutofidseedInitFileFilter;

impl AutofidseedInitFileFilter {
    /// Java private constructor.
    fn new() -> AutofidseedInitFileFilter {
        AutofidseedInitFileFilter
    }

    /// Java `accept(File)`.
    pub fn accept(&self, pathname: Option<&Path>) -> bool {
        pathname.is_some_and(|pathname| {
            pathname.is_file()
                && PATTERN.is_match(&utilities::java_io_file_get_name(
                    &pathname.to_string_lossy(),
                ))
        })
    }
}

/// Java static `exists(BaseManager, AxisID)`.
pub fn exists(manager: &'static dyn BaseManager, axis_id: AxisID) -> bool {
    // `FileType.AUTOFIDSEED_DIR.getFile(manager, axisID).listFiles(new
    // AutofidseedInitFileFilter())`: null when the directory cannot be listed.
    // Upstream bug fixed in translation (AutofidseedInitFileFilter.java:36):
    // `getFile` may return null, which `listFiles` dereferences; a null directory
    // lists nothing here.
    let filter = AutofidseedInitFileFilter::new();
    let file_list: Option<Vec<PathBuf>> = file_type::CLASS
        .autofidseed_dir
        .get_file(Some(manager), Some(axis_id))
        .and_then(|dir| std::fs::read_dir(dir).ok())
        .map(|entries| {
            entries
                .filter_map(Result::ok)
                .map(|entry| entry.path())
                .filter(|path| filter.accept(Some(path)))
                .collect()
        });
    file_list.is_some_and(|file_list| !file_list.is_empty())
}

/// Java static `getFileName(BaseManager, AxisID)`.
pub fn get_file_name(manager: &'static dyn BaseManager, axis_id: AxisID) -> Option<String> {
    // `FileType.AUTOFIDSEED_DIR.getFile(manager, axisID).listFiles(new
    // AutofidseedInitFileFilter())`: null when the directory cannot be listed.
    // Upstream bug fixed in translation (AutofidseedInitFileFilter.java:42):
    // `getFile` may return null, which `listFiles` dereferences; a null directory
    // lists nothing here.
    let filter = AutofidseedInitFileFilter::new();
    let file_list: Option<Vec<PathBuf>> = file_type::CLASS
        .autofidseed_dir
        .get_file(Some(manager), Some(axis_id))
        .and_then(|dir| std::fs::read_dir(dir).ok())
        .map(|entries| {
            entries
                .filter_map(Result::ok)
                .map(|entry| entry.path())
                .filter(|path| filter.accept(Some(path)))
                .collect()
        });
    let Some(file_list) = file_list.filter(|file_list| !file_list.is_empty()) else {
        let message = format!("No {} file available.", get_descr(manager, axis_id));
        ui_harness::with(|harness| {
            harness.open_message_dialog_base_manager_string_string_axis_id(
                Some(manager),
                &message,
                "No Such File",
                Some(axis_id),
            )
        });
        return None;
    };
    // there should be only one .pkmod file
    Some(format!(
        "{}{}{}",
        file_type::CLASS
            .autofidseed_dir
            .get_file_name(Some(manager), Some(axis_id))
            .as_deref()
            .unwrap_or("null"),
        std::path::MAIN_SEPARATOR,
        utilities::java_io_file_get_name(&file_list[0].to_string_lossy())
    ))
}

/// Java private static `getDescr(BaseManager, AxisID)`.
fn get_descr(manager: &'static dyn BaseManager, axis_id: AxisID) -> String {
    format!(
        "{}/{}nnn{}",
        file_type::CLASS
            .autofidseed_dir
            .get_file_name(Some(manager), Some(axis_id))
            .as_deref()
            .unwrap_or("null"),
        NAME,
        EXT
    )
}
