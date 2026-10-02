//! `IMOD/Etomo/src/etomo/process/FailureReason.java`.
//!
//! The reasons an intermittent background process (a load-average `w` over ssh) can
//! fail, with the tooltip the processor table shows for each.  A Java type-safe enum:
//! the three instances are statics, compared by identity.

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java final package-private class `FailureReason`.
#[derive(Debug)]
pub struct FailureReason {
    /// Java private final `reason`.
    reason: &'static str,
    /// Java private final `tooltip`.
    tooltip: &'static str,
}

/// Java static final `UNKOWN` (sic).
pub static UNKOWN: FailureReason =
    FailureReason::new("", "Unable to get the load averages for this computer.");

/// Java static final `COMPUTER_DOWN`.
pub static COMPUTER_DOWN: FailureReason =
    FailureReason::new("down", "This computer in not running.");

/// Java static final `LOGIN_FAILED`.
pub static LOGIN_FAILED: FailureReason = FailureReason::new(
    "no login",
    concat!(
        "You must have an account on this computer.  ",
        "You must also have a passwordless login on this computer."
    ),
);

impl FailureReason {
    /// Java private `FailureReason(String, String)`.
    const fn new(reason: &'static str, tooltip: &'static str) -> FailureReason {
        FailureReason { reason, tooltip }
    }

    /// Java package-private `getReason()`.
    pub fn get_reason(&self) -> &str {
        self.reason
    }

    /// Java package-private `getTooltip()`.
    pub fn get_tooltip(&self) -> &str {
        self.tooltip
    }
}

/// Java `toString()`: the reason.
impl std::fmt::Display for FailureReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.reason)
    }
}
