//! `IMOD/Etomo/src/etomo/comscript/ComScriptInputArg.java`.
//!
//! Models a single input argument in an IMOD com script.  The associate comments are
//! also handled with each line returned as a separate element of a String array.
#![allow(dead_code)]

use crate::imod::etomo::comscript::fortran_input_string::FortranInputString;
use crate::imod::etomo::r#type::const_etomo_number::java_lang_double_to_string;
use crate::imod::etomo::util::utilities::java_lang_string_split_limit;
use regex::Regex;

/// Java package-private class `ComScriptInputArg`.
#[derive(Clone, Debug)]
pub struct ComScriptInputArg {
    /// Java field `comments`, initialised to `new String[0]`.
    comments: Vec<Option<String>>,
    /// Java field `argument`, initialised to null.
    argument: Option<String>,
    /// Java field `debug`, initialised to false.
    debug: bool,
}

impl ComScriptInputArg {
    /// Java `ComScriptInputArg()`, the default constructor.  A zero length String is
    /// created to represent the comments and the argument is null.
    pub fn new() -> ComScriptInputArg {
        ComScriptInputArg {
            comments: Vec::new(),
            argument: None,
            debug: false,
        }
    }

    /// Java `ComScriptInputArg(ComScriptInputArg)`, the copy constructor.
    pub fn new_from(src: &ComScriptInputArg) -> ComScriptInputArg {
        let mut input_arg = ComScriptInputArg::new();
        input_arg.argument = src.get_argument().map(|argument| argument.to_string());
        input_arg.comments = src.get_comments();
        input_arg
    }

    /// Java `setArgument(String)`.  The whole line is considerered the argument,
    /// comments are not parsed from the line.
    pub fn set_argument(&mut self, argument: Option<&str>) {
        self.set_argument_parse_comments(argument, false);
    }

    /// Java `setArgument(String, boolean)`.  The argument is considered the first
    /// whitespace separated token in the line.  The remainder of the line is appended
    /// onto the comments for this input argument.
    pub fn set_argument_parse_comments(&mut self, argument: Option<&str>, parse_comments: bool) {
        if parse_comments {
            let argument = match argument {
                None => panic!("java.lang.NullPointerException"),
                Some(argument) => argument,
            };
            let parse = java_lang_string_split_limit(argument, &Regex::new(r"\s+").unwrap(), 2);
            self.argument = Some(parse[0].clone());
            if parse.len() > 1 {
                let mut append: Vec<Option<String>> = vec![None];
                if parse[1].starts_with('#') {
                    append[0] = Some(parse[1].clone());
                } else {
                    append[0] = Some(format!("# {}", parse[1]));
                }
                self.add_comments(&append);
            }
        } else {
            self.argument = argument.map(|argument| argument.to_string());
        }
    }

    /// Java `setArgument(boolean)`.  Set the argument line with an integer.
    pub fn set_argument_boolean(&mut self, arg: bool) {
        if arg {
            self.argument = Some("1".to_string())
        } else {
            self.argument = Some("0".to_string())
        }
    }

    /// Java `setArgument(int)`.  Set the argument line with an integer.
    pub fn set_argument_int(&mut self, arg: i32) {
        self.argument = Some(arg.to_string());
    }

    /// Java `setArgument(double)`.  Set the argument line with a double.
    pub fn set_argument_double(&mut self, arg: f64) {
        self.argument = Some(java_lang_double_to_string(arg));
    }

    /// Java `setArgument(FortranInputString)`.  Set the argument line with
    /// FortranInputString.
    pub fn set_argument_fortran_input_string(&mut self, arg: &FortranInputString) {
        self.argument = Some(arg.to_string());
    }

    /// Java `getArgument`.
    pub fn get_argument(&self) -> Option<&str> {
        self.argument.as_deref()
    }

    /// Java `setComments`.  Each element of the array represents a line of comments.
    pub fn set_comments(&mut self, comments: &[Option<String>]) {
        // make a defensive copy of the array
        self.comments = vec![None; comments.len()];
        for i in 0..comments.len() {
            self.comments[i] = comments[i].clone();
        }
    }

    /// Java `setDebug`.  Returns the previous setting of debug.
    pub fn set_debug(&mut self, input: bool) -> bool {
        let old_debug = self.debug;
        self.debug = input;
        old_debug
    }

    /// Java `getComments`.  Each comment line is a separate element in the array; this
    /// is a zero length array if there are no comments.
    pub fn get_comments(&self) -> Vec<Option<String>> {
        // make a defensive copy of the array
        let mut safe_array = vec![None; self.comments.len()];
        for i in 0..self.comments.len() {
            safe_array[i] = self.comments[i].clone();
        }
        safe_array
    }

    /// Java `addComments`.  Add the array of comments on to the end of the existing
    /// comments.
    pub fn add_comments(&mut self, append: &[Option<String>]) {
        let existing_comments = std::mem::take(&mut self.comments);
        self.comments = vec![None; existing_comments.len() + append.len()];
        for i in 0..existing_comments.len() {
            self.comments[i] = existing_comments[i].clone();
        }
        for i in 0..append.len() {
            self.comments[i + existing_comments.len()] = append[i].clone();
        }
    }
}

impl Default for ComScriptInputArg {
    fn default() -> ComScriptInputArg {
        ComScriptInputArg::new()
    }
}

/// Java `toString`.  Java renders a null `argument` as `null`.
impl std::fmt::Display for ComScriptInputArg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}]", self.argument.as_deref().unwrap_or("null"))
    }
}
