//! `IMOD/Etomo/src/etomo/process/ProcessResultDisplayFactoryInterface.java`.

use crate::imod::etomo::r#type::process_result_display::ProcessResultDisplayHandle;

/// Java `ProcessResultDisplayFactoryInterface`.  The factory and its displays
/// are EDT objects.
pub trait ProcessResultDisplayFactoryInterface {
    /// Java `getProcessResultDisplay(int, String)`.
    fn get_process_result_display(
        &self,
        display_id: i32,
        factory_id: &str,
    ) -> Option<ProcessResultDisplayHandle>;
}
