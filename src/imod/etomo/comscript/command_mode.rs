//! `IMOD/Etomo/src/etomo/comscript/CommandMode.java`.
//!
//! Java's interface has only its explicit `toString` contract.  Rust's
//! `Display` is the direct equivalent: `ToString` is supplied by the standard
//! library for every `Display` implementation.

/// Java `CommandMode`.
pub trait CommandMode: std::fmt::Display + std::any::Any {}

/// Java `command.getCommandMode() == SomeParam.Mode.X`: modes are enum
/// singletons compared by identity, which is the same concrete type and value.
pub fn equals_mode<M: CommandMode + PartialEq>(mode: Option<&dyn CommandMode>, other: &M) -> bool {
    mode.and_then(|mode| (mode as &dyn std::any::Any).downcast_ref::<M>())
        .is_some_and(|mode| mode == other)
}

#[cfg(test)]
mod tests {
    use super::CommandMode;

    struct Mode;

    impl std::fmt::Display for Mode {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("subcommand")
        }
    }

    impl CommandMode for Mode {}

    #[test]
    fn source_to_string_contract_is_available() {
        let mode: &dyn CommandMode = &Mode;
        assert_eq!(mode.to_string(), "subcommand");
    }
}
