//! `IMOD/Etomo/src/etomo/type/QueueMode.java`.
//!
//! Description: The queue mode as described in the splitbatch man page and Bug# 2413.
//!
//! Copyright: Copyright 2022 - 2023 by the Regents of the University of Colorado
//!
//! Organization: Dept. of MCD Biology, University of Colorado

use super::queue_type::QueueType;

/// Java `QueueMode`, a typesafe enum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueueMode {
    /// Java `QUEUE_WITH_SINGLE_CPU`.  Mode 1: Queues with one CPU and no GPUs, or
    /// exclusive allocation mode queues (slurm queue with the initialize value set).
    QueueWithSingleCpu,
    /// Java `NODE_WITH_GPU`.  Mode 2.
    NodeWithGpu,
    /// Java `NODE_WITHOUT_GPU`.  Mode 2C.
    NodeWithoutGpu,
    /// Java `INVALID`.  Invalid - mode 1 queue with 1 or 2 computers.
    Invalid,
}

impl QueueMode {
    /// Java private field `label`.
    fn label(self) -> &'static str {
        match self {
            QueueMode::QueueWithSingleCpu => "QUEUE_WITH_SINGLE_CPU",
            QueueMode::NodeWithGpu => "NODE_WITH_GPU",
            QueueMode::NodeWithoutGpu => "NODE_WITHOUT_GPU",
            QueueMode::Invalid => "INVALID",
        }
    }

    /// Java private fields `type1` and `type2`.
    fn types(self) -> (Option<QueueType>, Option<QueueType>) {
        match self {
            QueueMode::QueueWithSingleCpu => (Some(QueueType::Queue), None),
            QueueMode::NodeWithGpu => (Some(QueueType::Node), None),
            QueueMode::NodeWithoutGpu => (Some(QueueType::Node), Some(QueueType::NodeWithoutGpu)),
            QueueMode::Invalid => (None, None),
        }
    }

    /// Java `isType(QueueType)`.
    pub fn is_type(self, queue_type: QueueType) -> bool {
        self.is_type_nullable(Some(queue_type))
    }

    /// Java `isType(QueueType)` with the source's nullable argument: a null queueType
    /// is a match.
    pub fn is_type_nullable(self, queue_type: Option<QueueType>) -> bool {
        let (type1, type2) = self.types();
        if queue_type.is_none() || (type1.is_none() && type2.is_none()) {
            return true;
        }
        queue_type == type1 || queue_type == type2
    }
}

/// Java `toString`.
impl std::fmt::Display for QueueMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}
