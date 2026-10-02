//! `IMOD/Etomo/src/etomo/type/RunType.java`.
//!
//! An enumeration written as a Java class with static instances; each
//! instance is compared by identity, which is enum equality here.

/// Java final `RunType`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum RunType {
    /// Java `RUN`.
    Run,
    /// Java `RECONNECT`.
    Reconnect,
    /// Java `RESUME`.
    Resume,
    /// Java `RESUME_PROCESS_CHUNKS`.
    ResumeProcessChunks,
    /// Java `SERIES_WATCHER`.
    SeriesWatcher,
}

impl RunType {
    /// Java private final `descr`.
    fn descr(self) -> &'static str {
        match self {
            RunType::Run => "Run",
            RunType::Reconnect => "Reconnect",
            RunType::Resume => "Resume",
            RunType::ResumeProcessChunks => "ResumeProcessChunks",
            RunType::SeriesWatcher => "SeriesWatcher",
        }
    }
}

/// Java `toString`.
impl std::fmt::Display for RunType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.descr())
    }
}
