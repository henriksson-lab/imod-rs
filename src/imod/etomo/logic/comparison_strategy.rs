//! `IMOD/Etomo/src/etomo/logic/ComparisonStrategy.java`.
//!
//! Used to modify how a comparison is done.

use super::strategy_does_not_apply_exception::StrategyDoesNotApplyException;

/// Java `public interface ComparisonStrategy`.
pub trait ComparisonStrategy {
    /// Java `equals(String, String) throws StrategyDoesNotApplyException`.
    fn equals(
        &self,
        left_string: Option<&str>,
        right_string: Option<&str>,
    ) -> Result<bool, StrategyDoesNotApplyException>;
}
