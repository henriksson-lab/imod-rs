//! `IMOD/Etomo/src/etomo/storage/PcOptionType.java`.
//!
//! The Java class is a typesafe enum with two instances; a Java `null` instance is
//! `Option<PcOptionType>` at the sites that can carry one.

/// Java `PcOptionType`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PcOptionType {
    /// Java `PC_OPTION_TYPE_COMPUTER`, named "computer".
    PcOptionTypeComputer,
    /// Java `PC_OPTION_TYPE_QUEUE`, named "queue".
    PcOptionTypeQueue,
}

impl PcOptionType {
    /// Java private field `name`.
    fn name(self) -> &'static str {
        match self {
            PcOptionType::PcOptionTypeComputer => "computer",
            PcOptionType::PcOptionTypeQueue => "queue",
        }
    }

    /// Java static `getInstance(String)`.
    pub fn get_instance(line: Option<&str>) -> Option<PcOptionType> {
        let line = line?;
        if line == PcOptionType::PcOptionTypeComputer.to_string() {
            return Some(PcOptionType::PcOptionTypeComputer);
        }
        if line == PcOptionType::PcOptionTypeQueue.to_string() {
            return Some(PcOptionType::PcOptionTypeQueue);
        }
        None
    }
}

/// Java `toString`.
impl std::fmt::Display for PcOptionType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}
