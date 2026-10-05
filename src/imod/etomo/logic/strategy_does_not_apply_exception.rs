//! `IMOD/Etomo/src/etomo/logic/StrategyDoesNotApplyException.java`.
//!
//! Used to modify how a comparison is done.

/// Java `public final class StrategyDoesNotApplyException extends Exception`; the
/// message is null for the no-argument constructor.
#[derive(Clone, Debug, Default)]
pub struct StrategyDoesNotApplyException(pub Option<String>);

impl StrategyDoesNotApplyException {
    /// Java `StrategyDoesNotApplyException()`.
    pub fn new() -> StrategyDoesNotApplyException {
        StrategyDoesNotApplyException(None)
    }

    /// Java `StrategyDoesNotApplyException(String)`.
    pub fn new_with_message(message: Option<&str>) -> StrategyDoesNotApplyException {
        StrategyDoesNotApplyException(message.map(str::to_owned))
    }
}
