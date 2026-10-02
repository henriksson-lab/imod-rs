//! `IMOD/Etomo/src/etomo/ui/swing/WarpVolDisplay.java`.

use crate::imod::etomo::comscript::warp_vol_param::WarpVolParam;

/// Java `WarpVolDisplay` (does not extend `ProcessDisplay`).
pub trait WarpVolDisplay {
    /// Java `getParameters(WarpVolParam, boolean)`.
    fn get_parameters(&self, param: &mut WarpVolParam, do_validation: bool) -> bool;
}

// TODO(unit): FlattenVolumePanel.java implements WarpVolDisplay; the Rust
// `flatten_volume_panel.rs` takes an explicit manager and a boundary param.
