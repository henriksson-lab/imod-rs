//! `IMOD/Etomo/src/etomo/ui/swing/RubberbandContainer.java`.
//!
//! GUI object that contains a `RubberbandPanel` and needs to be able to set
//! coordinate values in the parent.  Can currently set Z min and max.

/// Java `RubberbandContainer.rcsid`.
pub const RCSID: &str = "$Id$";

/// Java package-private `interface RubberbandContainer`.
pub trait RubberbandContainer {
    /// Java `setRubberbandContainerZMin(String)`.
    fn set_rubberband_container_z_min(&self, z_min: Option<&str>);

    /// Java `setRubberbandContainerZMax(String)`.
    fn set_rubberband_container_z_max(&self, z_max: Option<&str>);
}
