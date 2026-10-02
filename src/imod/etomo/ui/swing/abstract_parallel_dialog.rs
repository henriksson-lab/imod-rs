//! `IMOD/Etomo/src/etomo/ui/swing/AbstractParallelDialog.java`.
//!
//! A generic parent for the dialogs which can run a parallel process: the
//! processing code asks the dialog for its parallel parameters through it.

use crate::imod::etomo::comscript::parallel_param::ParallelParam;
use crate::imod::etomo::r#type::dialog_type::DialogType;

/// Java `AbstractParallelDialog.rcsid`.
pub const RCSID: &str = "$Id$";

/// Java public interface `AbstractParallelDialog`.
pub trait AbstractParallelDialog {
    /// Java `getParameters(ParallelParam)`.
    fn get_parameters(&self, param: &mut dyn ParallelParam);
    /// Java `getDialogType()`.
    fn get_dialog_type(&self) -> DialogType;
}
