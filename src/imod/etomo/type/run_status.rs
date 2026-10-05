//! `IMOD/Etomo/src/etomo/type/RunStatus.java`.
//!
//! Status before and after a run.  Java's typesafe-enum pattern (a private
//! constructor plus `static final` singletons compared by identity) is a Rust enum.

/// Java package-private static final `NAME`.
pub const NAME: &str = "RunStatus";

/// Java `public final class RunStatus`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum RunStatus {
    /// Java `TO_RUN = new RunStatus("ToRun")`.
    ToRun,
    /// Java `RAN = new RunStatus("Ran")`.
    Ran,
    /// Java `FAILED = new RunStatus("Failed")`.
    Failed,
    /// Java `KILLED = new RunStatus("Killed")`.
    Killed,
}

impl RunStatus {
    /// Java package-private final field `value`.
    pub fn value(self) -> &'static str {
        match self {
            RunStatus::ToRun => "ToRun",
            RunStatus::Ran => "Ran",
            RunStatus::Failed => "Failed",
            RunStatus::Killed => "Killed",
        }
    }

    /// Java package-private static `getInstance(String)`.
    pub fn get_instance(value: Option<&str>) -> Option<RunStatus> {
        let value = value?;
        if RunStatus::ToRun.value() == value {
            return Some(RunStatus::ToRun);
        }
        if RunStatus::Ran.value() == value {
            return Some(RunStatus::Ran);
        }
        if RunStatus::Failed.value() == value {
            return Some(RunStatus::Failed);
        }
        if RunStatus::Killed.value() == value {
            return Some(RunStatus::Killed);
        }
        None
    }
}

/// Java `toString()`.
impl std::fmt::Display for RunStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.value())
    }
}
