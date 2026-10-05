//! `IMOD/Etomo/src/etomo/ui/swing/FilterFullVolumeParent.java`.
//!
//! What a `FilterFullVolumePanel` asks of the dialog that holds it
//! (`AnisotropicDiffusionDialog`).

use super::process_interface::ProcessInterface;

/// Java `public static final String rcsid`.
pub const RCSID: &str = "$Id$";

/// Java package-private `interface FilterFullVolumeParent extends ProcessInterface`.
pub trait FilterFullVolumeParent: ProcessInterface {
    /// Java `cleanUp()`.
    fn clean_up(&self);

    /// Java `getVolume()`.
    fn get_volume(&self) -> Option<String>;

    /// Java `initSubdir()`.
    fn init_subdir(&self) -> bool;

    /// Java `isLoadWithFlipping()`.
    fn is_load_with_flipping(&self) -> bool;
}
