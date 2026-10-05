//! `IMOD/Etomo/src/etomo/ui/AutoAlignmentDisplay.java`.
//!
//! What `AutoAlignmentController` needs from the dialog it serves (the Join dialog or
//! the Serial Sections dialog).  An event dispatch thread object.

use crate::imod::etomo::comscript::midas_param::MidasParam;
use crate::imod::etomo::comscript::xfalign_param::XfalignParam;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::dialog_type::DialogType;

/// Java `rcsid`.
pub const RCSID: &str = "$Id:$";

/// Java `public interface AutoAlignmentDisplay`.
pub trait AutoAlignmentDisplay {
    /// Java `msgProcessEnded()`.  The purpose of this function is to have Midas button
    /// enabled whether or not Initial Auto-Alignment succeeds.
    fn msg_process_ended(&self);

    /// Java `getDialogType()`.
    fn get_dialog_type(&self) -> DialogType;

    /// Java `getAxisID()`.
    fn get_axis_id(&self) -> AxisID;

    /// Java `getAutoAlignmentParameters(MidasParam)`.
    fn get_auto_alignment_parameters_midas(&self, param: &mut MidasParam);

    /// Java `getAutoAlignmentParameters(XfalignParam, boolean)`.
    fn get_auto_alignment_parameters_xfalign(
        &self,
        param: &mut XfalignParam,
        do_validation: bool,
    ) -> bool;
}
