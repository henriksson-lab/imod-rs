//! `IMOD/Etomo/src/etomo/comscript/ParallelParam.java`.

use super::command_mode::CommandMode;

/// Java `ParallelParam`.
///
/// Java permits `getSubcommandMode` to return null; the source's
/// `ProcesschunksParam` initializes that member to null.  `None` preserves
/// that state without introducing an invented default command mode.
pub trait ParallelParam {
    fn get_subcommand_mode(&self) -> Option<&dyn CommandMode>;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Param;

    impl ParallelParam for Param {
        fn get_subcommand_mode(&self) -> Option<&dyn CommandMode> {
            None
        }
    }

    #[test]
    fn null_java_subcommand_mode_maps_to_none() {
        assert!(Param.get_subcommand_mode().is_none());
    }
}
