//! `IMOD/Etomo/src/etomo/FrontPageManager.java`.
//!
//! The front page is a real manager-list entry, not a synthetic startup
//! marker.  Its presentation collaborators are progressively translated in
//! `ui/swing`; this unit owns the source manager lifetime, identity, and close
//! semantics that the director requires to make the default window usable.

use std::convert::Infallible;

use crate::imod::etomo::base_manager::{BaseManager, BaseManagerBase};
use crate::imod::etomo::storage::storable::Storable;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::base_meta_data::BaseMetaData;
use crate::imod::etomo::r#type::front_page_meta_data::FrontPageMetaData;
use crate::imod::etomo::r#type::interface_type::InterfaceType;

/// Java private static final `AXIS_ID`.
pub const AXIS_ID: AxisID = AxisID::Only;
/// `FrontPageMetaData.NAME`.
pub const NAME: &str = "Front Page";

/// Java final `FrontPageManager`.
pub struct FrontPageManager {
    base: BaseManagerBase,
    /// Java final `metaData`.
    meta_data: FrontPageMetaData,
    /// Java lazy `processManager`; its concrete FrontPageProcessManager is
    /// not yet a Rust source unit.
    process_manager: Option<Infallible>,
}

impl FrontPageManager {
    /// Java `FrontPageManager()` and its image-style overload.  Image style is
    /// metadata input; it does not alter the no-file front-page lifecycle.
    pub fn new() -> &'static Self {
        let manager = Box::leak(Box::new(Self {
            base: BaseManagerBase::initial(),
            meta_data: FrontPageMetaData::new(None),
            process_manager: None,
        }));
        manager.base_manager();
        manager.initialize_ui_parameters_from_name(Some(""), Some(AXIS_ID));
        manager.create_state();
        manager
    }

    /// Java private `createState`, whose source body is empty.
    pub fn create_state(&self) {}

    /// Java private `openProcessingPanel`; the concrete panel is constructed
    /// by the GUI frontend after this manager becomes current.
    pub fn open_processing_panel(&self) {}

    /// Java private `openFrontPageDialog`; same ownership boundary as the
    /// front-page presentation factory.
    pub fn open_front_page_dialog(&self) {}

    /// Java `getMetaData`.
    pub fn get_meta_data(&self) -> &FrontPageMetaData {
        &self.meta_data
    }
}

impl BaseManager for FrontPageManager {
    fn base(&self) -> &BaseManagerBase {
        &self.base
    }

    fn this(&'static self) -> &'static dyn BaseManager {
        self
    }

    fn get_interface_type(&self) -> Option<InterfaceType> {
        Some(InterfaceType::FrontPage)
    }

    fn create_main_panel(&self) {
        // `MainFrontPagePanel` needs an owned presentation slot in
        // BaseManager; the current trait still carries its historical
        // Infallible placeholder.  The manager lifecycle is nevertheless
        // concrete and the GUI factory owns the panel construction.
    }

    fn get_base_meta_data(&self) -> Option<&dyn BaseMetaData> {
        Some(&self.meta_data)
    }

    fn get_main_panel(&self) -> Option<Infallible> {
        None
    }

    fn get_process_manager(&self) -> Option<Infallible> {
        self.process_manager
    }

    /// Java `getLogInterface`, whose source body returns null.
    fn get_log_interface(&self) -> Option<Infallible> {
        None
    }

    fn get_storables_with_offset(&self, _offset: i32) -> Option<Vec<Box<dyn Storable>>> {
        None
    }

    fn get_name(&self) -> Option<String> {
        self.meta_data.get_name()
    }

    fn allow_process_watching(&self) -> bool {
        false
    }

    /// Java `kill(AxisID)`, whose source body is intentionally empty: the
    /// front page owns no killable process.
    fn kill(&self, _axis_id: Option<AxisID>) {}

    fn pause(&self, _axis_id: Option<AxisID>) -> bool {
        false
    }

    fn save(&self) -> bool {
        // Java calls `super.save()` then `mainPanel.done()`.  There is no
        // parameter file for a front page, so there is nothing persistent to
        // save until FrontPageMetaData is translated.
        true
    }

    fn exit_program(&self, axis_id: Option<AxisID>) -> bool {
        if self.exit_program_super(axis_id) {
            self.end_threads();
            self.save_param_file();
            return true;
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn front_page_is_a_non_watching_single_axis_manager() {
        let manager = FrontPageManager::new();
        assert_eq!(manager.get_name().as_deref(), Some(NAME));
        assert_eq!(manager.get_interface_type(), Some(InterfaceType::FrontPage));
        assert!(!manager.allow_process_watching());
        assert!(!manager.pause(Some(AXIS_ID)));
    }
}
