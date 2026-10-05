//! `IMOD/Etomo/src/etomo/type/BatchRunTomoRowStatus.java`.
//!
//! Events coming from BatchRunTomoRow.

use super::status::Status;
use crate::imod::etomo::util::utilities::to_string_if_set;

/// Java `public class BatchRunTomoRowStatus implements Status`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum BatchRunTomoRowStatus {
    /// Java `RUN = new BatchRunTomoRowStatus("Run")`.
    Run,
}

impl Status for BatchRunTomoRowStatus {
    /// Java `getText()`.
    fn get_text(&self) -> Option<&'static str> {
        Some("Run checkbox")
    }
}

/// Java `toString()`.
impl std::fmt::Display for BatchRunTomoRowStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[BatchRunTomoRowStatus{}]",
            to_string_if_set(Some(":descr:"), Some("Run"))
        )
    }
}
