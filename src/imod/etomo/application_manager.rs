//! Initial executable ownership slice of `etomo/ApplicationManager.java`.

use crate::imod::etomo::base_manager::{BaseManager, BaseManagerBase};
use crate::imod::etomo::process::emergency_monitor::EmergencyMonitor;
use crate::imod::etomo::storage::storable::Storable;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::base_meta_data::BaseMetaData;
use crate::imod::etomo::r#type::dialog_type::{DialogType, TOTAL_RECON};
use crate::imod::etomo::r#type::extension::Extension;
use crate::imod::etomo::r#type::interface_type::InterfaceType;
use std::convert::Infallible;
use std::sync::Arc;
use std::sync::Mutex;

pub struct ApplicationManager {
    base: BaseManagerBase,
    name: String,
    advanced_a: Mutex<[bool; TOTAL_RECON as usize]>,
    advanced_b: Mutex<[bool; TOTAL_RECON as usize]>,
    new_manager: bool,
    setup_raw_image_stack: Mutex<Option<String>>,
    raw_image_stack_extension: Mutex<Option<String>>,
}
impl ApplicationManager {
    /// Java `ApplicationManager(String, AxisID)`: retain the dataset identity
    /// and run the shared manager construction before typed metadata/process
    /// state is attached by the subsequent translation units.
    pub fn new(param_file_name: Option<&str>, axis_id: AxisID) -> &'static Self {
        let manager = Box::leak(Box::new(Self {
            base: BaseManagerBase::initial(),
            name: param_file_name
                .filter(|name| !name.is_empty())
                .unwrap_or("Setup Tomogram")
                .to_owned(),
            advanced_a: Mutex::new([false; TOTAL_RECON as usize]),
            advanced_b: Mutex::new([false; TOTAL_RECON as usize]),
            new_manager: param_file_name.is_none_or(str::is_empty),
            setup_raw_image_stack: Mutex::new(None),
            raw_image_stack_extension: Mutex::new(None),
        }));
        manager.base_manager();
        manager.initialize_ui_parameters_from_name(param_file_name, Some(axis_id));
        manager.initialize_advanced(false);
        manager
    }
    /// Java private `initializeAdvanced`, parameterized by the director's
    /// current preference so the construction dependency remains explicit.
    pub fn initialize_advanced(&self, advanced: bool) {
        *self.advanced_a.lock().unwrap() = [advanced; TOTAL_RECON as usize];
        *self.advanced_b.lock().unwrap() = [advanced; TOTAL_RECON as usize];
    }
    pub fn is_advanced(&self, dialog_type: DialogType, axis_id: AxisID) -> bool {
        let index = dialog_type.to_index() as usize;
        if axis_id == AxisID::Second {
            self.advanced_b.lock().unwrap()[index]
        } else {
            self.advanced_a.lock().unwrap()[index]
        }
    }
    pub fn set_advanced(&self, dialog_type: DialogType, axis_id: AxisID, advanced: bool) {
        let index = dialog_type.to_index() as usize;
        if axis_id == AxisID::Second {
            self.advanced_b.lock().unwrap()[index] = advanced;
        }
        self.advanced_a.lock().unwrap()[index] = advanced;
    }
    pub fn set_advanced_a(&self, dialog_type: DialogType, advanced: bool) {
        self.advanced_a.lock().unwrap()[dialog_type.to_index() as usize] = advanced;
    }
    pub const fn is_new_manager(&self) -> bool {
        self.new_manager
    }
    /// Source `isSetupChanged`: the setup dialog's raw-image text is changed
    /// only when it contains a non-whitespace dataset value.
    pub fn is_setup_changed(&self) -> bool {
        self.setup_raw_image_stack
            .lock()
            .unwrap()
            .as_deref()
            .is_some_and(|value| !value.trim().is_empty())
    }
    pub fn set_setup_raw_image_stack(&self, value: Option<&str>) {
        *self.setup_raw_image_stack.lock().unwrap() = value.map(str::to_owned);
    }
    /// Java `setRawImageStackExtension`: reject non-input-image extensions
    /// before setup metadata is available to a process series.
    pub fn set_raw_image_stack_extension(&self, file_name: Option<&str>) -> bool {
        let Some(file_name) = file_name else {
            return false;
        };
        let Some(extension) = Extension::get_instance(file_name) else {
            return false;
        };
        if !extension.is_input_image_file() {
            return false;
        }
        *self.raw_image_stack_extension.lock().unwrap() = std::path::Path::new(file_name)
            .extension()
            .map(|value| format!(".{}", value.to_string_lossy()));
        true
    }
}
impl BaseManager for ApplicationManager {
    fn base(&self) -> &BaseManagerBase {
        &self.base
    }
    fn this(&'static self) -> &'static dyn BaseManager {
        self
    }
    fn get_interface_type(&self) -> Option<InterfaceType> {
        Some(InterfaceType::Recon)
    }
    /// Java `getEmergencyMonitor(AxisID)`: retain the application-manager
    /// override point while the shared typed monitor owns lazy construction.
    fn get_emergency_monitor(&'static self, axis_id: Option<AxisID>) -> Arc<EmergencyMonitor> {
        BaseManager::get_emergency_monitor(self, axis_id)
    }
    fn create_main_panel(&self) {}
    fn get_base_meta_data(&self) -> Option<&dyn BaseMetaData> {
        None
    }
    fn get_main_panel(&self) -> Option<Infallible> {
        None
    }
    fn get_process_manager(&self) -> Option<Infallible> {
        None
    }
    fn get_storables_with_offset(&self, _: i32) -> Option<Vec<Box<dyn Storable>>> {
        None
    }
    fn get_name(&self) -> Option<String> {
        Some(self.name.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn advanced_state_preserves_source_axis_update_rules() {
        let manager = ApplicationManager::new(None, AxisID::Only);
        assert!(manager.is_new_manager());
        manager.set_advanced(DialogType::SetupRecon, AxisID::Second, true);
        // Java's source sets B and then unconditionally sets A.
        assert!(manager.is_advanced(DialogType::SetupRecon, AxisID::Second));
        assert!(manager.is_advanced(DialogType::SetupRecon, AxisID::First));
        manager.set_advanced_a(DialogType::SetupRecon, false);
        assert!(!manager.is_advanced(DialogType::SetupRecon, AxisID::First));
        assert!(manager.is_advanced(DialogType::SetupRecon, AxisID::Second));
        assert!(!ApplicationManager::new(Some("data.edf"), AxisID::Only).is_new_manager());
    }

    #[test]
    fn setup_changed_matches_source_whitespace_rule() {
        let manager = ApplicationManager::new(None, AxisID::Only);
        assert!(!manager.is_setup_changed());
        manager.set_setup_raw_image_stack(Some(" \t\n "));
        assert!(!manager.is_setup_changed());
        manager.set_setup_raw_image_stack(Some("raw.mrc"));
        assert!(manager.is_setup_changed());
    }
}
