//! `IMOD/Etomo/src/etomo/process/ProcessResultDisplayFactoryInterface.java`.
#![allow(dead_code)]

/// Java `ProcessResultDisplayFactoryInterface`.
///
/// Java fixes the return type to `ProcessResultDisplay`; the associated type preserves
/// that contract while allowing the concrete Rust factory to return its typed
/// `MultiLineButton` display rather than erase it behind a duplicate wrapper.
pub trait ProcessResultDisplayFactoryInterface {
    type ProcessResultDisplay;

    /// Java `getProcessResultDisplay(int, String)`.
    fn get_process_result_display(
        &self,
        display_id: i32,
        factory_id: Option<&str>,
    ) -> Option<&Self::ProcessResultDisplay>;
}
