//! `IMOD/Etomo/src/etomo/type/BatchRunTomoDatasetStatus.java`.
//!
//! Miscellaneous dataset status values.

use super::status::Status;
use crate::imod::etomo::util::utilities::to_string_if_set;

/// Java `public class BatchRunTomoDatasetStatus implements Status`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum BatchRunTomoDatasetStatus {
    /// Java `DELIVERED = new BatchRunTomoDatasetStatus("Delivered")`.
    Delivered,
    /// Java `RENAMED = new BatchRunTomoDatasetStatus("Renamed")`.
    Renamed,
}

impl BatchRunTomoDatasetStatus {
    /// Java private final `descr`.
    fn descr(self) -> &'static str {
        match self {
            BatchRunTomoDatasetStatus::Delivered => "Delivered",
            BatchRunTomoDatasetStatus::Renamed => "Renamed",
        }
    }
}

impl Status for BatchRunTomoDatasetStatus {
    /// Java `getText()`.
    fn get_text(&self) -> Option<&'static str> {
        None
    }
}

/// Java `toString()`.
impl std::fmt::Display for BatchRunTomoDatasetStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[BatchRunTomoDatasetStatus{}]",
            to_string_if_set(Some(":descr:"), Some(self.descr()))
        )
    }
}
