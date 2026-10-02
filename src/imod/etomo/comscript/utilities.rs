//! `IMOD/Etomo/src/etomo/comscript/Utilities.java`.
//!
//! Copyright: Copyright 2008
//!
//! Organization: Boulder Laboratory for 3-Dimensional Electron Microscopy of Cells
//! (BL3DEMC), University of Colorado

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::file_type;
use crate::imod::etomo::util::goodframe::Goodframe;
use crate::imod::etomo::util::montagesize::Montagesize;

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `MONTAGE_SEPARATION`.
pub const MONTAGE_SEPARATION: &str = "-10";

/// Java package-private static `is90DegreeImageRotation(double)`.
pub fn is_90_degree_image_rotation(image_rotation: f64) -> bool {
    (image_rotation > 45.0 && image_rotation < 135.0)
        || (image_rotation < -45.0 && image_rotation > -135.0)
}

/// Java package-private static `getGoodframeFromMontageSize(AxisID, BaseManager)`.
///
/// The source catches `InvalidParameterException` and `IOException` (printing the stack
/// trace) but not the `NumberFormatException` that `Montagesize.read` and
/// `Goodframe.run` throw for an unparsable size, which then escapes to the caller as an
/// uncaught runtime exception (Utilities.java:41-58).  Fixed in translation: every
/// failure is reported the same way and answers null.  `Montagesize.getInstance`
/// returning no instance (see its docs) also answers null.
pub fn get_goodframe_from_montage_size(
    axis_id: AxisID,
    manager: &'static dyn BaseManager,
) -> Option<Goodframe> {
    let montagesize =
        Montagesize::get_instance(manager, axis_id, &file_type::CLASS.raw_stack, false)?;
    // The Java `try` block.
    match montagesize.read(manager) {
        Ok(_) => {
            if montagesize.is_file_exists() {
                let mut goodframe = Goodframe::new(manager.get_property_user_dir(), axis_id);
                match goodframe.run_int(
                    manager,
                    montagesize.get_x().get_int(),
                    montagesize.get_y().get_int(),
                ) {
                    Ok(()) => return Some(goodframe),
                    // `e.printStackTrace()`.
                    Err(e) => eprintln!("{}", e),
                }
            }
        }
        // `e.printStackTrace()`.
        Err(e) => eprintln!("{}", e),
    }
    None
}
