//! `IMOD/Etomo/src/etomo/comscript/ConstSplitCorrectionParam.java`.

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java interface `ConstSplitCorrectionParam`.  `getCommand` builds the array
/// on first use; the implementation caches it behind a lock, so it takes
/// `&self`.
pub trait ConstSplitCorrectionParam {
    /// Java `getCommand`.
    fn get_command(&self) -> Vec<String>;
}
