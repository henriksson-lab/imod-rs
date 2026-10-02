//! `IMOD/Etomo/src/etomo/comscript/ConstWarpVolParam.java`.

use super::command::Command;

/// Java `ConstWarpVolParam extends Command`.
pub trait ConstWarpVolParam: Command {
    /// Java `getTemporaryDirectory`.
    fn get_temporary_directory(&self) -> String;
    /// Java `getOutputSizeZ`.
    fn get_output_size_z(&self) -> String;
    /// Java `isInterpolationOrderLinear`.
    fn is_interpolation_order_linear(&self) -> bool;
}
