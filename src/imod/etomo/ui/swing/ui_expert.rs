//! `IMOD/Etomo/src/etomo/ui/swing/UIExpert.java`.

use crate::imod::etomo::r#type::dialog_type::DialogType;

/// Java `UIExpert`.
///
/// The three process collaborators are associated types because their Java
/// concrete source units own their state.  This preserves every interface
/// signature without substituting a common process model.
pub trait UIExpert {
    type Process;
    type ProcessResultDisplay;
    type ProcessSeries;
    type DialogExitState;
    type ProcessDisplay;

    /// Java `openDialog()`.
    fn open_dialog(&mut self);
    /// Java `startNextProcess(ProcessSeries.Process, ProcessResultDisplay,
    /// ProcessSeries, DialogType, ProcessDisplay)`.
    fn start_next_process(
        &mut self,
        process: &mut Self::Process,
        process_result_display: &mut Self::ProcessResultDisplay,
        process_series: &mut Self::ProcessSeries,
        dialog_type: DialogType,
        display: &mut Self::ProcessDisplay,
    ) -> bool;
    /// Java `saveAction()`.
    fn save_action(&mut self);
    /// Java `saveDialog(DialogExitState)`.
    fn save_dialog(&mut self, exit_state: Self::DialogExitState);
}

#[cfg(test)]
mod tests {
    use super::UIExpert;
    use crate::imod::etomo::r#type::dialog_type::DialogType;
    struct Expert;
    impl UIExpert for Expert {
        type Process = ();
        type ProcessResultDisplay = ();
        type ProcessSeries = ();
        type DialogExitState = ();
        type ProcessDisplay = ();
        fn open_dialog(&mut self) {}
        fn start_next_process(
            &mut self,
            _: &mut (),
            _: &mut (),
            _: &mut (),
            _: DialogType,
            _: &mut (),
        ) -> bool {
            true
        }
        fn save_action(&mut self) {}
        fn save_dialog(&mut self, _: ()) {}
    }
    #[test]
    fn all_java_interface_methods_are_implementable() {
        let mut expert = Expert;
        let (mut process, mut result, mut series, mut display) = ((), (), (), ());
        expert.open_dialog();
        assert!(expert.start_next_process(
            &mut process,
            &mut result,
            &mut series,
            DialogType::TomogramGeneration,
            &mut display
        ));
        expert.save_action();
        expert.save_dialog(());
    }
}
