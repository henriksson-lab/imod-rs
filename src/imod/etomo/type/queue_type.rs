//! `IMOD/Etomo/src/etomo/type/QueueType.java`.
//!
//! Description: The type of queue.
//!
//! Copyright: Copyright 2022 by the Regents of the University of Colorado
//!
//! Organization: Dept. of MCD Biology, University of Colorado

/// Java `QueueType`, a typesafe enum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueueType {
    /// Java `QUEUE`.
    Queue,
    /// Java `NODE`.
    Node,
    /// Java `NODE_WITHOUT_GPU`.
    NodeWithoutGpu,
}

impl QueueType {
    /// Java package-private static `getInstance(String)`.
    pub(crate) fn get_instance(label: Option<&str>) -> Option<QueueType> {
        let label = label?;
        if label == QueueType::Queue.get_label() {
            return Some(QueueType::Queue);
        }
        if label == QueueType::Node.get_label() {
            return Some(QueueType::Node);
        }
        if label == QueueType::NodeWithoutGpu.get_label() {
            return Some(QueueType::NodeWithoutGpu);
        }
        None
    }

    /// Java package-private `getLabel`.
    pub(crate) fn get_label(self) -> &'static str {
        match self {
            QueueType::Queue => "QUEUE",
            QueueType::Node => "NODE",
            QueueType::NodeWithoutGpu => "NODE_WITHOUT_GPU",
        }
    }
}
