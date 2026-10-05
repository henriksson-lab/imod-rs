//! `IMOD/Etomo/src/etomo/logic/TransformsTool.java`.
//!
//! Checks that the edge function file (`.xef`) is up to date with the edge
//! displacement file (`.ecd`) before the blended montage is used.

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::file_type;
use crate::imod::etomo::ui::swing::ui_harness;
use crate::imod::etomo::util::utilities;

/// Java `rcsid`.
pub const RCSID: &str = "$Id:$";

/// Java private static final `TITLE`.
const TITLE: &str = "Dataset Consistency Warning";

/// Java static `checkUpToDateEdgeFunctionsFile(boolean, BaseManager, AxisID, String,
/// String)`.  Pops up a warning and returns false if the edge function file (.xef)
/// was not successfully created after the .ecd file.  Ignores the Y edge function
/// file (.yef).
pub fn check_up_to_date_edge_functions_file(
    invalid_edge_functions: bool,
    manager: &'static dyn BaseManager,
    axis_id: AxisID,
    location_descr: Option<&str>,
    create_edge_functions_descr: Option<&str>,
) -> bool {
    let instructions = format!(
        "  Please open {} and run {}.",
        location_descr.unwrap_or("null"),
        create_edge_functions_descr.unwrap_or("null")
    );
    let message = |text: String| {
        ui_harness::with(|harness| {
            harness.open_message_dialog_base_manager_string_string_axis_id(
                Some(manager),
                &text,
                TITLE,
                Some(axis_id),
            )
        });
    };
    let edge_functions_file = file_type::CLASS
        .edge_functions_x
        .get_file(Some(manager), Some(axis_id));
    if !edge_functions_file
        .as_ref()
        .is_some_and(|file| file.exists())
    {
        message(format!(
            "Warning: Edge functions do not exist.{}",
            instructions
        ));
        return false;
    }
    let piece_shifts_file = file_type::CLASS
        .piece_shifts
        .get_file(Some(manager), Some(axis_id));
    let Some(piece_shifts_file) = piece_shifts_file.filter(|file| file.exists()) else {
        return true;
    };
    if invalid_edge_functions {
        message(format!(
            "Warning: Edge functions are invalid.{}",
            instructions
        ));
        return false;
    }
    let edge_functions_file = edge_functions_file.unwrap_or_default();
    if utilities::java_io_file_last_modified(&piece_shifts_file.to_string_lossy())
        > utilities::java_io_file_last_modified(&edge_functions_file.to_string_lossy())
    {
        message(format!(
            "Warning: Edge functions are out of date.{}",
            instructions
        ));
        return false;
    }
    true
}
