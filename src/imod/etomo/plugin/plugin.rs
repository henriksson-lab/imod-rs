//! `IMOD/Etomo/src/etomo/plugin/Plugin.java`.
//!
//! The base interface for eTomo-compatible plugins.  External plugins must have a
//! public default constructor.  (The Java javadoc's long how-to on writing a plugin -
//! classes to inherit, process series, autodocs, post-processing - describes the Java
//! plugin API and is not repeated here; see the source.)
//!
//! Implementers are event-dispatch-thread objects (`Rc`, `&self` methods).

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::dialog_type::DialogType;

/// Java `public interface Plugin`.
pub trait Plugin {
    /// Java `getKey()`: unique key for retrieving plugin information.
    fn get_key(&self) -> Option<String>;

    /// Java `getTitle()`.
    fn get_title(&self) -> Option<String>;

    /// Java `getVersion()`.
    fn get_version(&self) -> Option<String>;

    /// Java `getDescription()`.
    fn get_description(&self) -> Option<String>;

    /// Java `init(BaseManager, AxisID, AxisType, DialogType)`.  Will be called as soon
    /// as the plugin instance is constructed.
    fn init(
        &self,
        manager: &'static dyn BaseManager,
        axis_id: Option<AxisID>,
        axis_type: Option<AxisType>,
        dialog_type: Option<DialogType>,
    );

    /// Java `setParameters()`.  Called by an openDialog function.  Signal that the
    /// plugin should load data from files.
    fn set_parameters(&self);

    /// Java `save()`.  Called by a saveDialog function.  Signal that the plugin should
    /// save data to files.
    fn save(&self);
}
