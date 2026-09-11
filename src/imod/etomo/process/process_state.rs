//! `IMOD/Etomo/src/etomo/process/ProcessState.java`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProcessState {
    NotStarted,
    InProgress,
    Complete,
}
impl ProcessState {
    pub fn from_string(name: Option<&str>) -> Option<Self> {
        match name? {
            value if value.eq_ignore_ascii_case("Not started") => Some(Self::NotStarted),
            value if value.eq_ignore_ascii_case("In progress") => Some(Self::InProgress),
            value if value.eq_ignore_ascii_case("Complete") => Some(Self::Complete),
            _ => None,
        }
    }
}
impl std::fmt::Display for ProcessState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::NotStarted => "Not started",
            Self::InProgress => "In progress",
            Self::Complete => "Complete",
        })
    }
}
