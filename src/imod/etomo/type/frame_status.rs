//! `IMOD/Etomo/src/etomo/type/FrameStatus.java`.

use super::status::Status;
use crate::imod::etomo::util::utilities::to_string_if_set;

/// Java `public final class FrameStatus implements Status`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum FrameStatus {
    /// Java `SINGLE = new FrameStatus("Single")`.
    Single,
    /// Java `MONTAGE = new FrameStatus("Montage")`.
    Montage,
}

impl FrameStatus {
    /// Java private final `descr`.
    fn descr(self) -> &'static str {
        match self {
            FrameStatus::Single => "Single",
            FrameStatus::Montage => "Montage",
        }
    }
}

impl Status for FrameStatus {
    /// Java `getText()`.
    fn get_text(&self) -> Option<&'static str> {
        None
    }
}

/// Java `toString()`.
impl std::fmt::Display for FrameStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[FrameStatus{}]",
            to_string_if_set(Some(":descr:"), Some(self.descr()))
        )
    }
}

// <p>Updates done</p>
