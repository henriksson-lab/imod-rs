//! `IMOD/Etomo/src/etomo/storage/DirectiveType.java`.
//!
//! Java's typesafe-enum pattern is mirrored as a Rust enum with one variant per
//! `public static final` singleton; the singletons are also associated constants under
//! their Java names.
#![allow(dead_code)]

use regex::Regex;

use crate::imod::etomo::storage::autodoc::autodoc_tokenizer::SEPARATOR_CHAR;
use crate::imod::etomo::r#type::const_etomo_number::java_lang_string_trim;
use crate::imod::etomo::util::utilities::java_lang_string_split;

/// Java `rcsid`.
pub const RCSID: &str = "$Id:$";

/// Java `DirectiveType`.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DirectiveType {
    /// Java `SETUP_SET = new DirectiveType("setupset")`.
    SetupSet,
    /// Java `RUN_TIME = new DirectiveType("runtime")`.
    RunTime,
    /// Java `COM_PARAM = new DirectiveType("comparam")`.
    ComParam,
    /// Java `COPY_ARG = new DirectiveType("copyarg")`.
    CopyArg,
}

/// Java `directive.split("\\" + AutodocTokenizer.SEPARATOR_CHAR)`, which drops trailing
/// empty strings.
fn split_directive(directive: &str) -> Vec<String> {
    java_lang_string_split(
        directive,
        &Regex::new(&regex::escape(SEPARATOR_CHAR)).unwrap(),
    )
}

impl DirectiveType {
    /// Java `SETUP_SET`.
    pub const SETUP_SET: DirectiveType = DirectiveType::SetupSet;
    /// Java `RUN_TIME`.
    pub const RUN_TIME: DirectiveType = DirectiveType::RunTime;
    /// Java `COM_PARAM`.
    pub const COM_PARAM: DirectiveType = DirectiveType::ComParam;
    /// Java `COPY_ARG`.
    pub const COPY_ARG: DirectiveType = DirectiveType::CopyArg;

    /// Java field `tag`.
    fn tag(self) -> &'static str {
        match self {
            Self::SetupSet => "setupset",
            Self::RunTime => "runtime",
            Self::ComParam => "comparam",
            Self::CopyArg => "copyarg",
        }
    }

    /// Java package-private static `getInstance(String)`.
    pub(crate) fn get_instance(directive: &str) -> Option<DirectiveType> {
        Self::get_instance_from_array(Some(split_directive(directive).as_slice()))
    }

    /// Java package-private static `getInstance(String[])`.
    pub(crate) fn get_instance_from_array(
        directive_array: Option<&[String]>,
    ) -> Option<DirectiveType> {
        let directive_array = directive_array?;
        let mut index = 0;
        if directive_array.len() > index {
            if Self::SETUP_SET.tag() == directive_array[index] {
                index += 1;
                if directive_array.len() > index {
                    if Self::COPY_ARG.tag() == directive_array[index] {
                        return Some(Self::COPY_ARG);
                    }
                }
                return Some(Self::SETUP_SET);
            }
            if Self::RUN_TIME.tag() == directive_array[index] {
                return Some(Self::RUN_TIME);
            }
            if Self::COM_PARAM.tag() == directive_array[index] {
                return Some(Self::COM_PARAM);
            }
        }
        None
    }

    /// Java package-private static `getFirstSectionInstance(String)`.  `input` is a
    /// directive name or first part of the name; returns the instance matching the first
    /// section of input.
    pub(crate) fn get_first_section_instance(input: Option<&str>) -> Option<DirectiveType> {
        let input = input?;
        let mut input = input.to_string();
        if input.contains('.') {
            let array = split_directive(&input);
            if array.is_empty() {
                return None;
            }
            input = array[0].clone();
        }
        let input = java_lang_string_trim(&input);
        if Self::SETUP_SET.equals(Some(input)) {
            return Some(Self::SETUP_SET);
        }
        if Self::RUN_TIME.equals(Some(input)) {
            return Some(Self::RUN_TIME);
        }
        if Self::COM_PARAM.equals(Some(input)) {
            return Some(Self::COM_PARAM);
        }
        None
    }

    /// Java `equals(String)`.  `input` is a directive name or first part of the name;
    /// returns true if this instance matches the first section of input.
    pub fn equals(self, input: Option<&str>) -> bool {
        let input = match input {
            None => return false,
            Some(input) => input,
        };
        let mut input = input.to_string();
        if input.contains('.') {
            let array = split_directive(&input);
            if array.is_empty() {
                return false;
            }
            input = array[0].clone();
        }
        java_lang_string_trim(&input) == self.tag()
    }

    /// Java `getKey`.
    pub fn get_key(self) -> String {
        if self == Self::COPY_ARG {
            return format!("{}{}{}", Self::SETUP_SET.tag(), SEPARATOR_CHAR, self.tag());
        }
        self.tag().to_string()
    }
}

/// Java `toString`: the tag.
impl std::fmt::Display for DirectiveType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.tag())
    }
}
