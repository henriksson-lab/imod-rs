//! `IMOD/Etomo/src/etomo/ui/swing/AbstractParallelDialog.java`.

use crate::imod::etomo::comscript::parallel_param::ParallelParam;
use crate::imod::etomo::r#type::dialog_type::DialogType;

/// Java `AbstractParallelDialog`.
///
/// This is intentionally only the two-method source interface.  Dialogs own
/// their parameter filling logic; the interface merely lets processing code
/// request it through a common parent type.
pub trait AbstractParallelDialog {
    fn get_parameters(&self, param: &mut dyn ParallelParam);
    fn get_dialog_type(&self) -> DialogType;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::etomo::comscript::command_mode::CommandMode;

    struct Param;
    impl ParallelParam for Param {
        fn get_subcommand_mode(&self) -> Option<&dyn CommandMode> {
            None
        }
    }

    struct Dialog;
    impl AbstractParallelDialog for Dialog {
        fn get_parameters(&self, _param: &mut dyn ParallelParam) {}

        fn get_dialog_type(&self) -> DialogType {
            DialogType::Tools
        }
    }

    #[test]
    fn exposes_both_source_interface_methods() {
        let dialog: &dyn AbstractParallelDialog = &Dialog;
        let mut param = Param;
        dialog.get_parameters(&mut param);
        assert_eq!(dialog.get_dialog_type(), DialogType::Tools);
    }
}
