//! `IMOD/Etomo/src/etomo/ui/swing/ProcessInterface.java`.
//!
//! The interface a dialog or panel implements so that the
//! `ProcessingMethodMediator` and the parallel panel can query and change its
//! processing method.  Swing-side objects: every method takes `&self`, and the
//! objects passed through it are shared `Rc` handles.

use std::rc::Rc;

use super::button_component::ButtonComponent;
use crate::imod::etomo::r#type::processing_method::ProcessingMethod;
use crate::imod::etomo::ui::queue_table_listener::QueueTableListener;

/// Java public interface `ProcessInterface extends QueueTableListener`.
pub trait ProcessInterface: QueueTableListener {
    /// Java `updateGpu(boolean)`.  Tell the interface that the user has checked
    /// or unchecked the parallel panel's cluster checkbox.  This only has an
    /// effect on a dialog which can do parallel GPU processing.
    fn update_gpu(&self, disable_gpu: bool);

    /// Java `getProcessingMethod()`.  The display processing method; it should
    /// change depending on the tab.  Should never return QUEUE.
    fn get_processing_method(&self) -> ProcessingMethod;

    /// Java `getSecondaryProcessingMethod()`.  Java may return null.
    fn get_secondary_processing_method(&self) -> Option<ProcessingMethod>;

    /// Java `lockProcessingMethod(boolean)`.  When lock is true, disable check
    /// boxes that can change the processing method; when false, enable them.
    fn lock_processing_method(&self, lock: bool);

    /// Java `setMethod(ProcessingMethod)`.
    fn set_method(&self, processing_method: ProcessingMethod);

    /// Java `isUseGpu()`.
    fn is_use_gpu(&self) -> bool;

    /// Java `setUseQueueCheckBox(ButtonComponent)`; Java may pass null.
    fn set_use_queue_check_box(&self, use_queue_checkbox: Option<Rc<dyn ButtonComponent>>);

    /// Java `addQueueTableListener(QueueTableListener)`.
    fn add_queue_table_listener(&self, listener: Rc<dyn QueueTableListener>);

    /// Java `removeQueueTableListener(QueueTableListener)`, by identity.
    fn remove_queue_table_listener(&self, listener: &Rc<dyn QueueTableListener>);
}
