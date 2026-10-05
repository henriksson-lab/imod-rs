//! `IMOD/Etomo/src/etomo/plugin/demo/DemoScreenState.java`.
//!
//! Extension to the `ReconScreenState` class.  `getInstance` inserts the instance into
//! the manager's screen state, which then stores and loads it with its own properties;
//! the inserted storable and the panel share the instance (`Arc`).

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::storage::storable::Storable;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::panel_header_state::PanelHeaderState;
use crate::imod::etomo::r#type::recon_screen_state;

/// Java `final class DemoScreenState implements Storable`.
pub struct DemoScreenState {
    /// Java private final `tomoGenDemoHeaderState`.
    tomo_gen_demo_header_state: PanelHeaderState,
}

impl DemoScreenState {
    /// Java private `DemoScreenState()`.
    fn new() -> DemoScreenState {
        DemoScreenState {
            tomo_gen_demo_header_state: PanelHeaderState::new(&format!(
                "{}.Demo{}",
                DialogType::TomogramGeneration.get_storable_name(),
                recon_screen_state::HEADER_GROUP
            )),
        }
    }

    /// Java static `getInstance(ApplicationManager, AxisID)`.  Uses
    /// `ReconScreenState.insert` to add an instance of this class to `ReconScreenState`.
    pub fn get_instance(
        manager: &'static ApplicationManager,
        axis_id: AxisID,
    ) -> Arc<DemoScreenState> {
        let instance = Arc::new(DemoScreenState::new());
        manager
            .get_screen_state(axis_id)
            .insert(Some(Box::new(instance.clone())));
        instance
    }

    /// Java package-private `getTomoGenDemoHeaderState()`.
    pub fn get_tomo_gen_demo_header_state(&self) -> &PanelHeaderState {
        &self.tomo_gen_demo_header_state
    }
}

impl Storable for DemoScreenState {
    /// Java `store(Properties)`.
    fn store(&self, props: &mut BTreeMap<String, String>) {
        self.store_with_prepend(props, "");
    }

    /// Java `store(Properties, String)`.
    fn store_with_prepend(&self, props: &mut BTreeMap<String, String>, prepend: &str) {
        self.tomo_gen_demo_header_state
            .store_with_prepend(props, prepend);
    }

    /// Java `load(Properties)`.
    fn load(&self, props: &mut BTreeMap<String, String>) {
        self.load_with_prepend(props, "");
    }

    /// Java `load(Properties, String)`.
    fn load_with_prepend(&self, props: &mut BTreeMap<String, String>, prepend: &str) {
        self.tomo_gen_demo_header_state
            .load_with_prepend(props, prepend);
    }
}
