//! `IMOD/Etomo/src/etomo/ui/BatchRunTomoTab.java`.
//!
//! The tabs of the batchruntomo dialog.  The Java singletons are an enum.

use crate::imod::etomo::util::utilities;

/// Java `public final class BatchRunTomoTab`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum BatchRunTomoTab {
    /// Java `BATCH = new BatchRunTomoTab(0, "Batch Setup")`.
    Batch,
    /// Java `STACKS = new BatchRunTomoTab(1, "Stacks")`.
    Stacks,
    /// Java `DATASET = new BatchRunTomoTab(2, "Dataset Values")`.
    Dataset,
    /// Java `RUN = new BatchRunTomoTab(3, "Run")`.
    Run,
}

/// Java `SIZE = RUN.index + 1`.
pub const SIZE: usize = 4;

/// Java `DEFAULT = BATCH`.
pub const DEFAULT: BatchRunTomoTab = BatchRunTomoTab::Batch;

impl BatchRunTomoTab {
    /// Java static `getInstance(int)`.
    pub fn get_instance(index: i32) -> BatchRunTomoTab {
        for tab in [
            BatchRunTomoTab::Batch,
            BatchRunTomoTab::Stacks,
            BatchRunTomoTab::Dataset,
            BatchRunTomoTab::Run,
        ] {
            if index == tab.get_index() {
                return tab;
            }
        }
        DEFAULT
    }

    /// Java `getTitle()`.
    pub fn get_title(self) -> &'static str {
        match self {
            BatchRunTomoTab::Batch => "Batch Setup",
            BatchRunTomoTab::Stacks => "Stacks",
            BatchRunTomoTab::Dataset => "Dataset Values",
            BatchRunTomoTab::Run => "Run",
        }
    }

    /// Java `getQuotedLabel()`.
    pub fn get_quoted_label(self) -> String {
        utilities::quote_label(Some(self.get_title())).unwrap_or_default()
    }

    /// Java `getIndex()`.
    pub fn get_index(self) -> i32 {
        match self {
            BatchRunTomoTab::Batch => 0,
            BatchRunTomoTab::Stacks => 1,
            BatchRunTomoTab::Dataset => 2,
            BatchRunTomoTab::Run => 3,
        }
    }
}

/// Java `toString()`: the title.
impl std::fmt::Display for BatchRunTomoTab {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.get_title())
    }
}
