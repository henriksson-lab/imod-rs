//! `IMOD/Etomo/src/etomo/ui/swing/TrialTiltDisplay.java`.

use super::process_display::ProcessDisplay;
use super::tilt_display::TiltDisplay;
use super::trial_tilt_panel::TrialTiltPanel;
use super::trial_tilt_parent::TrialTiltParent;

/// Java `TrialTiltDisplay extends TiltDisplay`.  Methods take `&self`;
/// implementing panels keep their state interior-mutable.
pub trait TrialTiltDisplay: TiltDisplay {
    /// Java `getTrialTomogramName()`.
    fn get_trial_tomogram_name(&self) -> Option<String>;

    /// Java `containsTrialTomogramName(String)`.
    fn contains_trial_tomogram_name(&self, trial_tomogram_name: Option<&str>) -> bool;

    /// Java `addTrialTomogramName(String)`.
    fn add_trial_tomogram_name(&self, trial_tomogram_name: Option<&str>);
}



// TODO(unit): TrialTiltPanel.java implements TrialTiltDisplay; the Rust
// `trial_tilt_panel.rs` reads local `TrialTiltParam`/`SplittiltParam`
// boundaries rather than the comscript `TiltParam`/`SplittiltParam`.
