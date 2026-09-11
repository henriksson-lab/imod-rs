//! `IMOD/Etomo/src/etomo/comscript/CommandMode.java`.
//!
//! Java's interface has only its explicit `toString` contract.  Rust's
//! `Display` is the direct equivalent: `ToString` is supplied by the standard
//! library for every `Display` implementation.

/// Java `CommandMode`.
pub trait CommandMode: std::fmt::Display {}

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
