//! `IMOD/Etomo/src/etomo/comscript/ConstSqueezevolParam.java`.

use super::command_details::CommandDetails;
use crate::imod::etomo::storage::storable::StorableValue;
use crate::imod::etomo::r#type::const_etomo_number::ConstEtomoNumber;

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java interface `ConstSqueezevolParam extends CommandDetails, Storable`.
pub trait ConstSqueezevolParam: CommandDetails + StorableValue {
    /// Java `getReductionFactorX`.
    fn get_reduction_factor_x(&self) -> &ConstEtomoNumber;

    /// Java `getReductionFactorY`.
    fn get_reduction_factor_y(&self) -> &ConstEtomoNumber;

    /// Java `getReductionFactorZ`.
    fn get_reduction_factor_z(&self) -> &ConstEtomoNumber;
}
