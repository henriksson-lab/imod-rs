//! `IMOD/Etomo/src/etomo/type/ProcessResultDisplay.java`.
//!
//! Java hands the same mutable display object to a process, a process series,
//! and a display-factory dependency graph.  `Rc<RefCell<_>>` is the matching
//! single-UI-thread ownership shape: it preserves aliasing and mutation without
//! pretending the display is a copyable process result.

use super::file_key::FileKey;
use super::process_end_state::ProcessEndState;
use super::process_result::ProcessResult;
use std::cell::RefCell;
use std::rc::Rc;

pub type ProcessResultDisplayHandle = Rc<RefCell<dyn ProcessResultDisplay>>;

/// Java `ProcessResultDisplay` interface.
pub trait ProcessResultDisplay {
    fn get_output_image_file_key(&self) -> Option<FileKey>;
    fn set_next(&mut self, display: Option<ProcessResultDisplayHandle>);
    fn set_use_global_dependency_list(&mut self, input: bool);
    fn get_next(&self) -> Option<ProcessResultDisplayHandle>;
    fn set_debug(&mut self, input: bool);
    fn dump_state(&self);
    fn get_original_state(&self) -> bool;
    fn set_original_state(&mut self, original_state: bool);
    fn set_process_done(&mut self, done: bool);
    fn msg_process_result(&mut self, display_state: ProcessResult);
    fn msg_process_end_state(&mut self, end_state: ProcessEndState);
    fn msg_process_starting(&mut self);
    fn msg_process_succeeded(&mut self);
    fn msg_process_failed(&mut self);
    fn msg_process_failed_to_start(&mut self);
    fn msg_secondary_process(&mut self);
    fn add_dependent_display(&mut self, dependent_display: ProcessResultDisplayHandle);
    fn add_failure_display(&mut self, failure_display: ProcessResultDisplayHandle);
    fn add_success_display(&mut self, success_display: ProcessResultDisplayHandle);
    fn equals_id(&self, display_id: i32, factory_id: &str) -> bool;
    fn set_id(&mut self, display_id: i32, factory_id: String);
    fn get_display_id(&self) -> i32;
    fn get_factory_id(&self) -> Option<&str>;
    fn set_factory_id(&mut self, factory_id: String);
    fn get_button_state_key(&self) -> Option<String>;
}
