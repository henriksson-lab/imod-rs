//! `IMOD/Etomo/src/etomo/ui/swing/UIExpert.java`.
//!
//! Experts live on the event dispatch thread with the dialogs they drive.  The
//! manager holds each one as an `Rc` of the concrete expert and hands it out as
//! `Rc<dyn UIExpert>` (Java `getUIExpert`).  Every method takes `&self`: a
//! Java expert is re-entered from its own dialog (`saveAction` ->
//! `ProcessDialog.saveAction` -> `done` -> `expert.doneDialog`), so the mutable
//! state of a concrete expert sits in `Cell`/`RefCell` fields and no borrow of
//! the expert is held across a call out.

use crate::imod::etomo::process_series::{Process, ProcessSeriesHandle};
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::process_result_display::ProcessResultDisplayHandle;
use crate::imod::etomo::ui::swing::process_dialog::DialogExitState;
use crate::imod::etomo::ui::swing::process_display::ProcessDisplay;
use std::rc::Rc;

/// Java `public static final String rcsid`.
pub const RCSID: &str = "$Id$";

/// Java interface `UIExpert`.
pub trait UIExpert: std::any::Any {
    /// Java `openDialog()`.
    fn open_dialog(&self);

    /// Java `startNextProcess(ProcessSeries.Process, ProcessResultDisplay,
    /// ProcessSeries, DialogType, ProcessDisplay)`.
    fn start_next_process(
        &self,
        process: &Process,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
        dialog_type: Option<DialogType>,
        display: Option<Rc<dyn ProcessDisplay>>,
    ) -> bool;

    /// Java `saveAction()`.
    fn save_action(&self);

    /// Java `saveDialog(DialogExitState)`.
    fn save_dialog(&self, exit_state: DialogExitState);

    /// Rust-only: the Java callers cast the `UIExpert` returned by
    /// `ApplicationManager.getUIExpert` to its concrete class
    /// (`(TomogramPositioningExpert) getUIExpert(...)`); this is the cast.
    fn as_any(&self) -> &dyn std::any::Any;
}
