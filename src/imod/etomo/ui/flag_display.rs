//! `IMOD/Etomo/src/etomo/ui/FlagDisplay.java`.
//!
//! Interface for anything that needs to respond to a flag (see `TextFlagExtension`).

use super::flag_type::FlagType;

/// Java `FlagDisplay`.  Implementers are EDT objects (`Rc`, `&self` methods).
pub trait FlagDisplay {
    /// Java `setFlag(FlagType)`.  `None` is Java's null (no flag).
    fn set_flag(&self, flag_type: Option<&'static FlagType>);
}
