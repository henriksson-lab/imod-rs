//! `IMOD/Etomo/src/etomo/logic/ProcessorType.java`.

/// Java `ProcessorType`, a typesafe enum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProcessorType {
    /// Java `CPU`, named "cpu".
    Cpu,
    /// Java `GPU`, named "gpu".
    Gpu,
    /// Java `QUEUE`, named "queue".
    Queue,
}

/// Java `toString`.
impl std::fmt::Display for ProcessorType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            ProcessorType::Cpu => "cpu",
            ProcessorType::Gpu => "gpu",
            ProcessorType::Queue => "queue",
        })
    }
}
