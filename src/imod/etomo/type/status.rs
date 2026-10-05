//! `IMOD/Etomo/src/etomo/type/Status.java`.
#![allow(dead_code)]

/// Java `Status`.  Works with StatusChangeListener.
pub trait Status {
    /// Java `getText`.
    fn get_text(&self) -> Option<&'static str>;
}

use super::batch_run_tomo_dataset_state::BatchRunTomoDatasetState;
use super::batch_run_tomo_dataset_status::BatchRunTomoDatasetStatus;
use super::batch_run_tomo_row_status::BatchRunTomoRowStatus;
use super::batch_run_tomo_status::BatchRunTomoStatus;
use super::ending_step::EndingStep;
use super::frame_status::FrameStatus;
use super::process_name::ProcessName;
use super::step::Step;

/// A Java reference to a `Status` instance, as it travels through the
/// `StatusChangeListener` family.  The Java implementors are singletons that the
/// listeners tell apart with `instanceof` and compare with `==`; the Rust
/// implementors are `Copy` values, so a reference is one enum over them, whose
/// variant is the `instanceof` test and whose `==` is Java's identity.  (Not a
/// source member: it is the representation of the interface type.)
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StatusRef {
    BatchRunTomoStatus(BatchRunTomoStatus),
    BatchRunTomoDatasetState(BatchRunTomoDatasetState),
    BatchRunTomoDatasetStatus(BatchRunTomoDatasetStatus),
    BatchRunTomoRowStatus(BatchRunTomoRowStatus),
    EndingStep(EndingStep),
    Step(Step),
    ProcessName(ProcessName),
    FrameStatus(FrameStatus),
}

impl Status for StatusRef {
    /// The referenced instance's `getText()`.
    fn get_text(&self) -> Option<&'static str> {
        match self {
            StatusRef::BatchRunTomoStatus(status) => status.get_text(),
            StatusRef::BatchRunTomoDatasetState(status) => status.get_text(),
            StatusRef::BatchRunTomoDatasetStatus(status) => status.get_text(),
            StatusRef::BatchRunTomoRowStatus(status) => status.get_text(),
            StatusRef::EndingStep(status) => status.get_text(),
            StatusRef::Step(status) => status.get_text(),
            StatusRef::ProcessName(status) => status.get_text(),
            StatusRef::FrameStatus(status) => status.get_text(),
        }
    }
}

/// The referenced instance's `toString()`.
impl std::fmt::Display for StatusRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StatusRef::BatchRunTomoStatus(status) => write!(f, "{status}"),
            StatusRef::BatchRunTomoDatasetState(status) => write!(f, "{status}"),
            StatusRef::BatchRunTomoDatasetStatus(status) => write!(f, "{status}"),
            StatusRef::BatchRunTomoRowStatus(status) => write!(f, "{status}"),
            StatusRef::EndingStep(status) => write!(f, "{status}"),
            StatusRef::Step(status) => write!(f, "{status}"),
            StatusRef::ProcessName(status) => write!(f, "{status}"),
            StatusRef::FrameStatus(status) => write!(f, "{status}"),
        }
    }
}

impl From<BatchRunTomoStatus> for StatusRef {
    fn from(status: BatchRunTomoStatus) -> StatusRef {
        StatusRef::BatchRunTomoStatus(status)
    }
}
impl From<BatchRunTomoDatasetState> for StatusRef {
    fn from(status: BatchRunTomoDatasetState) -> StatusRef {
        StatusRef::BatchRunTomoDatasetState(status)
    }
}
impl From<BatchRunTomoDatasetStatus> for StatusRef {
    fn from(status: BatchRunTomoDatasetStatus) -> StatusRef {
        StatusRef::BatchRunTomoDatasetStatus(status)
    }
}
impl From<BatchRunTomoRowStatus> for StatusRef {
    fn from(status: BatchRunTomoRowStatus) -> StatusRef {
        StatusRef::BatchRunTomoRowStatus(status)
    }
}
impl From<EndingStep> for StatusRef {
    fn from(status: EndingStep) -> StatusRef {
        StatusRef::EndingStep(status)
    }
}
impl From<Step> for StatusRef {
    fn from(status: Step) -> StatusRef {
        StatusRef::Step(status)
    }
}
impl From<ProcessName> for StatusRef {
    fn from(status: ProcessName) -> StatusRef {
        StatusRef::ProcessName(status)
    }
}
impl From<FrameStatus> for StatusRef {
    fn from(status: FrameStatus) -> StatusRef {
        StatusRef::FrameStatus(status)
    }
}
