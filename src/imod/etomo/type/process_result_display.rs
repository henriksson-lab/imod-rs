//! `IMOD/Etomo/src/etomo/type/ProcessResultDisplay.java`.
//!
//! Java hands the same display object to a process, a process series and a
//! display-factory dependency graph.  Displays are EDT objects with `&self`
//! methods; the handle is an `Rc`.  `as_any_rc` stands in for Java's casts of a
//! display to its concrete button class.

use super::file_key::FileKey;
use super::process_end_state::ProcessEndState;
use super::process_result::ProcessResult;
use std::rc::Rc;

pub type ProcessResultDisplayHandle = Rc<dyn ProcessResultDisplay>;

/// Java `ProcessResultDisplay` interface.
pub trait ProcessResultDisplay {
    /// Java cast support: the display as `Any`, for `(Run3dmodButton) display`.
    fn as_any_rc(self: Rc<Self>) -> Rc<dyn std::any::Any>;
    fn get_output_image_file_key(&self) -> Option<FileKey>;
    fn set_next(&self, display: Option<ProcessResultDisplayHandle>);
    fn set_use_global_dependency_list(&self, input: bool);
    fn get_next(&self) -> Option<ProcessResultDisplayHandle>;
    fn set_debug(&self, input: bool);
    fn dump_state(&self);
    fn get_original_state(&self) -> bool;
    fn set_original_state(&self, original_state: bool);
    fn set_process_done(&self, done: bool);
    /// Java `setScreenState(BaseScreenState)`.
    fn set_screen_state(&self, screen_state: &'static super::base_screen_state::BaseScreenState);
    fn msg_process_result(&self, display_state: ProcessResult);
    fn msg_process_end_state(&self, end_state: ProcessEndState);
    fn msg_process_starting(&self);
    fn msg_process_succeeded(&self);
    fn msg_process_failed(&self);
    fn msg_process_failed_to_start(&self);
    fn msg_secondary_process(&self);
    fn add_dependent_display(&self, dependent_display: ProcessResultDisplayHandle);
    fn add_failure_display(&self, failure_display: ProcessResultDisplayHandle);
    fn add_success_display(&self, success_display: ProcessResultDisplayHandle);
    fn equals_id(&self, display_id: i32, factory_id: &str) -> bool;
    fn set_id(&self, display_id: i32, factory_id: String);
    fn get_display_id(&self) -> i32;
    fn get_factory_id(&self) -> Option<String>;
    fn set_factory_id(&self, factory_id: String);
    fn get_button_state_key(&self) -> Option<String>;
}
