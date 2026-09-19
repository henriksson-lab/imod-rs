//! `IMOD/Etomo/src/etomo/comscript/CommandDetails.java`.

use super::command::Command;
use super::process_details::ProcessDetails;

/// Java `CommandDetails extends Command, ProcessDetails`.
pub trait CommandDetails: Command + ProcessDetails {}

impl<T: Command + ProcessDetails> CommandDetails for T {}
