//! `IMOD/Etomo/src/etomo/plugin/PluginPanel.java`.
//!
//! This is not a plugin interface.  It is a panel belonging to a plugin.  This
//! interface is currently used with `TomogramGenerationDialog`.  It can be added to
//! other locations on demand.
//!
//! Implementers are event-dispatch-thread objects (`Rc`, `&self` methods).

use std::rc::Rc;

use crate::imod::etomo::jdk::JComponent;
use crate::imod::etomo::r#type::base_screen_state::BaseScreenState;

/// Java `public interface PluginPanel`.
pub trait PluginPanel {
    /// Java `getButtonTitle()`.
    fn get_button_title(&self) -> Option<String>;

    /// Java `getComponent()`.
    fn get_component(&self) -> Rc<JComponent>;

    /// Java `getParameters(BaseScreenState)`.
    fn get_parameters(&self, screen_state: &BaseScreenState);

    /// Java `setParameters(BaseScreenState)`.
    fn set_parameters(&self, screen_state: &BaseScreenState);

    /// Java `updateDisplay()`.
    fn update_display(&self);

    /// Java `done()`.
    fn done(&self);

    /// Java `msgVisibilityChanged(boolean)`.
    fn msg_visibility_changed(&self, visible: bool);
}
