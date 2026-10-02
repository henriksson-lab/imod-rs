//! `IMOD/Etomo/src/etomo/ui/swing/ProcessDisplay.java`.
//!
//! A marker interface for panels which contain process parameters (used with
//! the next-process functionality).  It declares nothing but its `rcsid`.

/// Java `ProcessDisplay.rcsid`.
pub const RCSID: &str = "$Id$";

/// Java public interface `ProcessDisplay`.
pub trait ProcessDisplay {
    /// Java cast `(TiltXcorrDisplay) display`: `None` where the display is not one.
    fn as_tilt_xcorr_display(&self) -> Option<&dyn super::tiltxcorr_display::TiltXcorrDisplay> {
        None
    }
    /// Java cast `(CcdEraserDisplay) display`.
    fn as_ccd_eraser_display(&self) -> Option<&dyn super::ccd_eraser_display::CcdEraserDisplay> {
        None
    }
    /// Java cast `(TiltDisplay) display`.
    fn as_tilt_display(&self) -> Option<&dyn super::tilt_display::TiltDisplay> {
        None
    }
    /// Java cast `(AlignFramesDisplay) display`.
    fn as_align_frames_display(
        &self,
    ) -> Option<&dyn super::align_frames_display::AlignFramesDisplay> {
        None
    }
}
