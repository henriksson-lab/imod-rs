//! `IMOD/Etomo/src/etomo/ui/swing/MultifiltSetupDisplay.java`.

use crate::imod::etomo::comscript::multifilt_setup_param::MultifiltSetupParam;

/// Java `MultifiltSetupDisplay` (does not extend `ProcessDisplay`).
pub trait MultifiltSetupDisplay {
    /// Java `getParameters(MultifiltSetupParam, boolean)`.
    fn get_parameters(&self, param: &mut MultifiltSetupParam, do_validation: bool) -> bool;
}

// TODO(unit): MultifiltPanel.java implements MultifiltSetupDisplay; the Rust
// `multifilt_panel.rs` writes into a boundary trait, so the impl waits.
