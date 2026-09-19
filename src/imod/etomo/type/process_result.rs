//! `IMOD/Etomo/src/etomo/type/ProcessResult.java`.
//!
//! Java uses identity-only singleton objects.  A Rust enum preserves the same
//! three closed outcomes while making result routing exhaustive.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProcessResult {
    FailedToStart,
    Failed,
    Succeeded,
}

impl ProcessResult {
    pub const FAILED_TO_START: Self = Self::FailedToStart;
    pub const FAILED: Self = Self::Failed;
    pub const SUCCEEDED: Self = Self::Succeeded;
}

#[cfg(test)]
mod tests {
    use super::ProcessResult;

    #[test]
    fn java_singleton_outcomes_are_closed_and_distinct() {
        assert_ne!(ProcessResult::FAILED_TO_START, ProcessResult::FAILED);
        assert_ne!(ProcessResult::FAILED, ProcessResult::SUCCEEDED);
    }
}
