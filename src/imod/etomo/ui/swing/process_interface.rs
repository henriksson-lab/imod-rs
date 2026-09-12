//! `IMOD/Etomo/src/etomo/ui/swing/ProcessInterface.java`.
//!
//! The Java interface stores a `ButtonComponent` supplied by its caller.  Rust
//! needs that owned component's concrete type to retain the same lifetime, so
//! the direct equivalent is an associated type constrained to that interface.
#![allow(dead_code)]

use super::button_component::ButtonComponent;
use crate::imod::etomo::r#type::processing_method::ProcessingMethod;
use crate::imod::etomo::ui::queue_table_listener::QueueTableListener;

/// Java `ProcessInterface`.
pub trait ProcessInterface: QueueTableListener {
    /// Concrete owned Java `ButtonComponent` at this implementation boundary.
    type QueueCheckBox: ButtonComponent;

    /// Java `updateGpu`.
    fn update_gpu(&mut self, disable_gpu: bool);
    /// Java `getProcessingMethod`.
    fn get_processing_method(&self) -> ProcessingMethod;
    /// Java `getSecondaryProcessingMethod`; Java permits a null result.
    fn get_secondary_processing_method(&self) -> Option<ProcessingMethod>;
    /// Java `lockProcessingMethod`.
    fn lock_processing_method(&mut self, lock: bool);
    /// Java `setMethod`.
    fn set_method(&mut self, processing_method: ProcessingMethod);
    /// Java `isUseGpu`.
    fn is_use_gpu(&self) -> bool;
    /// Java `setUseQueueCheckBox`.
    fn set_use_queue_check_box(&mut self, use_queue_checkbox: Option<Self::QueueCheckBox>);
    /// Java `addQueueTableListener`.
    fn add_queue_table_listener(&mut self, listener: &mut dyn QueueTableListener);
    /// Java `removeQueueTableListener`.
    fn remove_queue_table_listener(&mut self, listener: &mut dyn QueueTableListener);
}

#[cfg(test)]
mod tests {
    use super::ProcessInterface;
    use crate::imod::etomo::r#type::processing_method::ProcessingMethod;
    use crate::imod::etomo::ui::queue_table_event::QueueTableEvent;
    use crate::imod::etomo::ui::queue_table_listener::QueueTableListener;
    use crate::imod::etomo::ui::swing::check_box::CheckBox;

    struct Interface;

    impl QueueTableListener for Interface {
        fn queue_table_event_action(&mut self, _event: QueueTableEvent) {}
    }

    impl ProcessInterface for Interface {
        type QueueCheckBox = CheckBox;
        fn update_gpu(&mut self, _disable_gpu: bool) {}
        fn get_processing_method(&self) -> ProcessingMethod { ProcessingMethod::PpCpu }
        fn get_secondary_processing_method(&self) -> Option<ProcessingMethod> { None }
        fn lock_processing_method(&mut self, _lock: bool) {}
        fn set_method(&mut self, _processing_method: ProcessingMethod) {}
        fn is_use_gpu(&self) -> bool { false }
        fn set_use_queue_check_box(&mut self, _use_queue_checkbox: Option<CheckBox>) {}
        fn add_queue_table_listener(&mut self, _listener: &mut dyn QueueTableListener) {}
        fn remove_queue_table_listener(&mut self, _listener: &mut dyn QueueTableListener) {}
    }

    #[test]
    fn source_nullable_secondary_method_is_retained() {
        assert_eq!(Interface.get_secondary_processing_method(), None);
    }
}
