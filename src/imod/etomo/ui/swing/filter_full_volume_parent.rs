//! `IMOD/Etomo/src/etomo/ui/swing/FilterFullVolumeParent.java`.
//!
//! `FilterFullVolumeParent` extends the separately translated
//! `ProcessInterface` exactly as its Java source does.
#![allow(dead_code)]

use super::check_box::CheckBox;
use super::process_interface::ProcessInterface;
use crate::imod::etomo::r#type::processing_method::ProcessingMethod;

/// Java `FilterFullVolumeParent`.
pub trait FilterFullVolumeParent: ProcessInterface<QueueCheckBox = CheckBox> {
    /// Java `cleanUp`.
    fn clean_up(&mut self);

    /// Java `getVolume`.
    fn get_volume(&self) -> String;

    /// Java `initSubdir`.
    fn init_subdir(&mut self) -> bool;

    /// Java `isLoadWithFlipping`.
    fn is_load_with_flipping(&self) -> bool;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Parent;

    impl crate::imod::etomo::ui::queue_table_listener::QueueTableListener for Parent {
        fn queue_table_event_action(
            &mut self,
            _event: crate::imod::etomo::ui::queue_table_event::QueueTableEvent,
        ) {
        }
    }

    impl ProcessInterface for Parent {
        type QueueCheckBox = CheckBox;
        fn update_gpu(&mut self, _disable_gpu: bool) {}
        fn get_processing_method(&self) -> ProcessingMethod {
            ProcessingMethod::PpCpu
        }
        fn get_secondary_processing_method(&self) -> Option<ProcessingMethod> {
            None
        }
        fn lock_processing_method(&mut self, _lock: bool) {}
        fn set_method(&mut self, _processing_method: ProcessingMethod) {}
        fn is_use_gpu(&self) -> bool {
            false
        }
        fn set_use_queue_check_box(&mut self, _use_queue_checkbox: Option<CheckBox>) {}
        fn add_queue_table_listener(
            &mut self,
            _listener: &mut dyn crate::imod::etomo::ui::queue_table_listener::QueueTableListener,
        ) {
        }
        fn remove_queue_table_listener(
            &mut self,
            _listener: &mut dyn crate::imod::etomo::ui::queue_table_listener::QueueTableListener,
        ) {
        }
    }

    impl FilterFullVolumeParent for Parent {
        fn clean_up(&mut self) {}
        fn get_volume(&self) -> String {
            "volume.rec".into()
        }
        fn init_subdir(&mut self) -> bool {
            true
        }
        fn is_load_with_flipping(&self) -> bool {
            false
        }
    }

    #[test]
    fn preserves_parent_and_inherited_process_contract() {
        let mut parent = Parent;
        assert!(parent.init_subdir());
        assert_eq!(parent.get_volume(), "volume.rec");
        assert!(!parent.is_load_with_flipping());
        assert_eq!(parent.get_processing_method(), ProcessingMethod::PpCpu);
        parent.clean_up();
    }
}
