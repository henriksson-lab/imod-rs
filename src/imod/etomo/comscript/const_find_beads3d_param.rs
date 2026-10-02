//! `IMOD/Etomo/src/etomo/comscript/ConstFindBeads3dParam.java`.

use super::command_details::CommandDetails;
use crate::imod::etomo::r#type::const_etomo_number::ConstEtomoNumber;

/// Java interface `ConstFindBeads3dParam extends CommandDetails`.
pub trait ConstFindBeads3dParam: CommandDetails {
    /// Java `getBeadSize`.
    fn get_bead_size(&self) -> String;

    /// Java `getMinRelativeStrength`.
    fn get_min_relative_strength(&self) -> String;

    /// Java `getThresholdForAveraging`.
    fn get_threshold_for_averaging(&self) -> String;

    /// Java `getStorageThreshold`.
    fn get_storage_threshold(&self) -> &ConstEtomoNumber;

    /// Java `getMinSpacing`.
    fn get_min_spacing(&self) -> String;

    /// Java `getGuessNumBeads`.
    fn get_guess_num_beads(&self) -> String;

    /// Java `getMaxNumBeads`.
    fn get_max_num_beads(&self) -> String;
}
