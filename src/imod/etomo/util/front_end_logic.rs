//! `IMOD/Etomo/src/etomo/util/FrontEndLogic.java`.
//!
//! "Business logic" functions for the UI.

use std::path::Path;

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::etomo_boolean2::EtomoBoolean2;
use crate::imod::etomo::util::mrc_header::MRCHeader;
use crate::imod::etomo::util::utilities;

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java static `isRotated(BaseManager, AxisID, File)`.  Reads the MRC header.
/// Returns true if rows (Y) are greater or equal to sections (Z).  Returns
/// false if Y < Z.  Returns `None` (Java null) when the header is unreadable.
pub fn is_rotated(
    manager: &'static dyn BaseManager,
    axis_id: AxisID,
    file: &Path,
) -> Option<EtomoBoolean2> {
    let path = file.to_string_lossy();
    let header = MRCHeader::get_instance_in_dir(
        utilities::java_io_file_get_parent(&path).as_deref(),
        Some(&utilities::java_io_file_get_name(&path)),
        Some(axis_id),
    )?;
    // try {
    let read = header.borrow_mut().read_with_manager(manager);
    match read {
        Ok(_) => {
            let mut retval = EtomoBoolean2::new();
            let (n_rows, n_sections) = {
                let header = header.borrow();
                (header.get_n_rows(), header.get_n_sections())
            };
            retval.set_boolean(n_rows >= n_sections);
            Some(retval)
        }
        // catch (IOException e) / catch (InvalidParameterException e):
        // e.printStackTrace();
        Err(e) => {
            eprintln!("{e}");
            None
        }
    }
}
