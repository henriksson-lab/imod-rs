//! `IMOD/Etomo/src/etomo/type/ProcessEndState.java`.

/// Java's typesafe process-end-state enum.  `FileLockFailure` deliberately
/// keeps the source `failed` progress-bar string while retaining its distinct
/// serialized name and precedence behavior.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProcessEndState {
    Done,
    Failed,
    Killed,
    Paused,
    Cancelled,
    FileLockFailure,
}

impl ProcessEndState {
    /// Java `TOTAL`; the Java source intentionally excludes the file-lock
    /// state from this count.
    pub const TOTAL: i32 = 5;

    /// Java `toIndex`.
    pub const fn to_index(self) -> i32 {
        match self {
            Self::Done => 0,
            Self::Failed => 1,
            Self::Killed => 2,
            Self::Paused => 3,
            Self::Cancelled => 4,
            Self::FileLockFailure => 5,
        }
    }
    /// Java `isValid`.
    pub fn is_valid(value: &str) -> bool {
        Self::get_instance(Some(value)).is_some()
    }
    /// Java static `precedence`.
    pub const fn precedence(existing: Self, new_state: Self) -> Self {
        match (existing, new_state) {
            (Self::Killed | Self::Paused, _) => existing,
            (Self::Failed | Self::FileLockFailure, Self::Done) => existing,
            (_, state) => state,
        }
    }
    /// Java `getInstance`, including its trim-before-lookup behavior.
    pub fn get_instance(value: Option<&str>) -> Option<Self> {
        match value?.trim() {
            "done" => Some(Self::Done),
            "failed" => Some(Self::Failed),
            "killed" => Some(Self::Killed),
            "paused" => Some(Self::Paused),
            "cancelled" => Some(Self::Cancelled),
            "file lock failure" => Some(Self::FileLockFailure),
            _ => None,
        }
    }
    /// Java `getBarString`.
    pub const fn get_bar_string(self) -> &'static str {
        match self {
            Self::Done => "done",
            Self::Failed | Self::FileLockFailure => "failed",
            Self::Killed => "killed",
            Self::Paused => "paused",
            Self::Cancelled => "cancelled",
        }
    }
    /// Java `equals(String)`.
    pub fn equals_string(self, value: &str) -> bool {
        self.to_string() == value
    }

    /// Java `equals(ProcessEndState)`.  Rust's derived equality is the
    /// normal spelling, while this name keeps callers translated directly
    /// from the source explicit.
    pub const fn equals(self, other: Option<Self>) -> bool {
        matches!(other, Some(other) if self.to_index() == other.to_index())
    }

    /// Java `dumpState` diagnostic formatting.
    pub fn dump_state(self) {
        eprint!("[name:{self},index:{}]", self.to_index());
    }
}

impl std::fmt::Display for ProcessEndState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Done => "done",
            Self::Failed => "failed",
            Self::Killed => "killed",
            Self::Paused => "paused",
            Self::Cancelled => "cancelled",
            Self::FileLockFailure => "file lock failure",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::ProcessEndState;

    #[test]
    fn java_names_indices_and_precedence_are_retained() {
        assert_eq!(ProcessEndState::TOTAL, 5);
        assert_eq!(ProcessEndState::FileLockFailure.to_index(), 5);
        assert_eq!(
            ProcessEndState::get_instance(Some(" file lock failure ")),
            Some(ProcessEndState::FileLockFailure)
        );
        assert_eq!(ProcessEndState::FileLockFailure.get_bar_string(), "failed");
        assert_eq!(
            ProcessEndState::precedence(ProcessEndState::Failed, ProcessEndState::Done),
            ProcessEndState::Failed
        );
        assert!(ProcessEndState::Done.equals(Some(ProcessEndState::Done)));
        assert!(!ProcessEndState::Done.equals(Some(ProcessEndState::Failed)));
        assert!(!ProcessEndState::Done.equals(None));
    }
}
