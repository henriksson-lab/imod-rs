//! `IMOD/Etomo/src/etomo/storage/autodoc/ReadOnlyStatement.java`.

/// `Statement.Type` from the direct statement implementation unit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StatementType {
    NameValuePair,
    Subsection,
    Comment,
    EmptyLine,
}
impl std::fmt::Display for StatementType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::NameValuePair => "NAME_VALUE_PAIR",
            Self::Subsection => "SUBSECTION",
            Self::Comment => "COMMENT",
            Self::EmptyLine => "EMPTY_LINE",
        })
    }
}

/// Complete `ReadOnlyStatement` source interface.
pub trait ReadOnlyStatement {
    type Section;
    fn get_type(&self) -> StatementType;
    fn get_string(&self) -> String;
    fn size_left_side(&self) -> i32;
    fn get_left_side(&self) -> Option<String>;
    fn get_left_side_at(&self, index: i32) -> Option<String>;
    fn get_right_side(&self) -> Option<String>;
    fn get_subsection(&self) -> Option<&Self::Section>;
    fn get_line_num(&self) -> i32;
}

#[cfg(test)]
mod tests {
    use super::{ReadOnlyStatement, StatementType};
    struct Statement;
    impl ReadOnlyStatement for Statement {
        type Section = ();
        fn get_type(&self) -> StatementType {
            StatementType::Comment
        }
        fn get_string(&self) -> String {
            "# x".into()
        }
        fn size_left_side(&self) -> i32 {
            0
        }
        fn get_left_side(&self) -> Option<String> {
            None
        }
        fn get_left_side_at(&self, _: i32) -> Option<String> {
            None
        }
        fn get_right_side(&self) -> Option<String> {
            Some("# x".into())
        }
        fn get_subsection(&self) -> Option<&()> {
            None
        }
        fn get_line_num(&self) -> i32 {
            2
        }
    }
    #[test]
    fn statement_contract_preserves_comment_view() {
        let s = Statement;
        assert_eq!(s.get_type(), StatementType::Comment);
        assert_eq!(s.get_right_side().as_deref(), Some("# x"));
    }
}
