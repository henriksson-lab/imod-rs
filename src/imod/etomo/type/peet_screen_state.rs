//! `IMOD/Etomo/src/etomo/type/PeetScreenState.java`.
//!
//! The screen state of the PEET interface.  `PeetScreenState extends BaseScreenState
//! implements ConstPeetScreenState`: the superclass is the `base` field, reached
//! through `Deref` (as in `join_screen_state.rs`).

use std::collections::BTreeMap;

use super::axis_id::AxisID;
use super::axis_type::AxisType;
use super::base_screen_state::BaseScreenState;
use super::const_peet_screen_state::ConstPeetScreenState;
use super::dialog_type::DialogType;
use super::panel_header_state::{self, PanelHeaderState};
use crate::imod::etomo::storage::storable::Storable;

/// Java `public static final String rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `public final class PeetScreenState extends BaseScreenState implements
/// ConstPeetScreenState`.
pub struct PeetScreenState {
    /// Java superclass `BaseScreenState` state.
    pub base: BaseScreenState,
    /// Java private final `peetSetupHeaderState`.
    peet_setup_header_state: PanelHeaderState,
    /// Java private final `peetRunHeaderState`.
    peet_run_header_state: PanelHeaderState,
}

/// Java inheritance: every `BaseScreenState` member is reachable on a
/// `PeetScreenState`.
impl std::ops::Deref for PeetScreenState {
    type Target = BaseScreenState;

    fn deref(&self) -> &BaseScreenState {
        &self.base
    }
}

impl PeetScreenState {
    /// Java `PeetScreenState(AxisID, AxisType)`.
    pub fn new(axis_id: AxisID, axis_type: AxisType) -> PeetScreenState {
        PeetScreenState {
            base: BaseScreenState::new(axis_id, axis_type),
            peet_setup_header_state: PanelHeaderState::new(&format!(
                "{}.Setup.{}",
                DialogType::Peet.get_storable_name(),
                panel_header_state::KEY
            )),
            peet_run_header_state: PanelHeaderState::new(&format!(
                "{}.Run.{}",
                DialogType::Peet.get_storable_name(),
                panel_header_state::KEY
            )),
        }
    }

    /// Java `load(Properties, String)`.
    pub fn load_with_prepend(&self, props: &mut BTreeMap<String, String>, prepend: &str) {
        self.base.load_with_prepend(props, prepend);
        let prepend = self.base.get_prepend(prepend);
        self.peet_setup_header_state
            .load_with_prepend(props, &prepend);
        self.peet_run_header_state
            .load_with_prepend(props, &prepend);
    }

    /// Java `store(Properties, String)`.
    pub fn store_with_prepend(&self, props: &mut BTreeMap<String, String>, prepend: &str) {
        self.base.store_with_prepend(props, prepend);
        let prepend = self.base.get_prepend(prepend);
        self.peet_setup_header_state
            .store_with_prepend(props, &prepend);
        self.peet_run_header_state
            .store_with_prepend(props, &prepend);
    }
}

impl ConstPeetScreenState for PeetScreenState {
    fn get_peet_setup_header_state(&self) -> &PanelHeaderState {
        &self.peet_setup_header_state
    }

    fn get_peet_run_header_state(&self) -> &PanelHeaderState {
        &self.peet_run_header_state
    }
}

/// Java `Storable`: `load(Properties)` and `store(Properties)` are overridden to call
/// the two-argument forms with `""`.
impl Storable for PeetScreenState {
    fn store(&self, properties: &mut BTreeMap<String, String>) {
        PeetScreenState::store_with_prepend(self, properties, "");
    }

    fn store_with_prepend(&self, properties: &mut BTreeMap<String, String>, prepend: &str) {
        PeetScreenState::store_with_prepend(self, properties, prepend);
    }

    fn load(&self, properties: &mut BTreeMap<String, String>) {
        PeetScreenState::load_with_prepend(self, properties, "");
    }

    fn load_with_prepend(&self, properties: &mut BTreeMap<String, String>, prepend: &str) {
        PeetScreenState::load_with_prepend(self, properties, prepend);
    }
}
