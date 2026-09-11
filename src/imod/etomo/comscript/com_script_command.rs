//! `IMOD/Etomo/src/etomo/comscript/ComScriptCommand.java`.
//!
//! Description: This class models a single command within an IMOD com script
//! file.  It handles header comments, the command line, command line arguments
//! and standard input for the command.
//!
//! Copyright: Copyright 2002 - 2019 by the Regents of the University of Colorado
//!
//! Organization: Dept. of MCD Biology, University of Colorado
//!
//! **Aliasing.**  `stdinArgs` is a `LinkedList` of `ComScriptInputArg` *references*, and
//! two members deliberately share the objects rather than copy them: the copy
//! constructor adds the very `ComScriptInputArg`s `getInputArguments()` returned, and
//! `setValue`/`addKey`/`setValues` mutate an element already on the list.  The elements
//! are therefore `Rc<RefCell<ComScriptInputArg>>`, which is what a Java reference is
//! here.  `String[]` fields keep the source's null-vs-empty distinction as
//! `Option<Vec<Option<String>>>`.
#![allow(dead_code)]

use super::com_script_input_arg::ComScriptInputArg;
use super::invalid_parameter_exception::InvalidParameterException;
use crate::imod::etomo::r#type::const_etomo_number::java_lang_string_trim;
use crate::imod::etomo::util::utilities::java_lang_string_split_limit;
use regex::Regex;
use std::cell::RefCell;
use std::rc::Rc;

/// Java `ComScriptCommand`.
pub struct ComScriptCommand {
    /// Java field `caseInsensitive`.  Case insensitive when finding keywords.
    case_insensitive: bool,
    /// Java field `divider`.  Use a space instead of `\t` - tab doesn't work when
    /// tilt.com to create samples in tomo pos.
    divider: String,
    /// Java field `keywordValuePairs`, initialised to false.
    keyword_value_pairs: bool,
    /// Java field `headerComments`, initialised to `new String[0]`.
    header_comments: Option<Vec<Option<String>>>,
    /// Java field `command`, initialised to null.
    command: Option<String>,
    /// Java field `commandLineArgs`, initialised to `new String[0]`.
    command_line_args: Option<Vec<Option<String>>>,
    /// Java field `stdinArgs`, initialised to `new LinkedList()`.
    stdin_args: Vec<Rc<RefCell<ComScriptInputArg>>>,
    /// Java field `debug`, initialised to false.
    debug: bool,
}

impl ComScriptCommand {
    /// Java `ComScriptCommand(boolean, boolean)`, the package-private default
    /// constructor.  A zero length String array is created to represent the header
    /// comments and command line arguments, the command is null.  A zero length String
    /// array will be returned for the input argument array.
    pub fn new(case_insensitive: bool, separate_with_a_space: bool) -> ComScriptCommand {
        ComScriptCommand {
            case_insensitive,
            divider: if separate_with_a_space {
                " ".to_string()
            } else {
                "\t".to_string()
            },
            keyword_value_pairs: false,
            header_comments: Some(Vec::new()),
            command: None,
            command_line_args: Some(Vec::new()),
            stdin_args: Vec::new(),
            debug: false,
        }
    }

    /// Java `ComScriptCommand(ComScriptCommand)`, the copy constructor.  The
    /// `stdinArgs` elements are shared with `src`, not copied - `getInputArguments`
    /// copies only the array.
    pub fn new_from(src: &ComScriptCommand) -> ComScriptCommand {
        let mut command = ComScriptCommand {
            case_insensitive: src.case_insensitive,
            divider: src.divider.clone(),
            keyword_value_pairs: false,
            header_comments: Some(Vec::new()),
            command: None,
            command_line_args: Some(Vec::new()),
            stdin_args: Vec::new(),
            debug: false,
        };
        command.keyword_value_pairs = src.is_keyword_value_pairs();
        command.header_comments = src.get_header_comments();
        command.command = src.get_command().map(|command| command.to_string());
        command.command_line_args = src.get_command_line_args();

        let input_args = src.get_input_arguments();
        for input_arg in input_args.iter() {
            command.stdin_args.push(Rc::clone(input_arg));
        }
        command
    }

    /// Java `setHeaderComments`.  Set the header comments, the comment block which
    /// preceeds the command line.
    pub fn set_header_comments(&mut self, header_comments: &[Option<String>]) {
        // make a defensive copy of the array
        let mut copy = vec![None; header_comments.len()];
        for i in 0..header_comments.len() {
            copy[i] = header_comments[i].clone();
        }
        self.header_comments = Some(copy);
    }

    /// Java `appendHeaderComments`.
    pub fn append_header_comments(&mut self, more_comments: &[Option<String>]) {
        let old_comments = match self.header_comments.take() {
            None => panic!("java.lang.NullPointerException"),
            Some(old_comments) => old_comments,
        };
        let mut header_comments = vec![None; old_comments.len() + more_comments.len()];
        for i in 0..old_comments.len() {
            header_comments[i] = old_comments[i].clone();
        }
        for i in 0..more_comments.len() {
            header_comments[i + old_comments.len()] = more_comments[i].clone();
        }
        self.header_comments = Some(header_comments);
    }

    /// Java `getHeaderComments`.  Returns a copy of the header comments as a String
    /// array.
    pub fn get_header_comments(&self) -> Option<Vec<Option<String>>> {
        let header_comments = match &self.header_comments {
            None => return None,
            Some(header_comments) => header_comments,
        };
        // make a defensive copy of the array
        let mut safe_array = vec![None; header_comments.len()];
        for i in 0..header_comments.len() {
            safe_array[i] = header_comments[i].clone();
        }
        Some(safe_array)
    }

    /// Java `setCommand`.  Set the command string.
    pub fn set_command(&mut self, command: Option<&str>) {
        self.command = command.map(|command| command.to_string());
    }

    /// Java `getCommand`.  Get the command string.
    pub fn get_command(&self) -> Option<&str> {
        self.command.as_deref()
    }

    /// Java `setCommandLineArgs`.  Set the command line arguments for the script
    /// command.
    pub fn set_command_line_args(&mut self, args: &[Option<String>]) {
        let first = || match &args[0] {
            None => panic!("java.lang.NullPointerException"),
            Some(first) => first.clone(),
        };
        if args.len() == 1
            && (first() == "-StandardInput"
                || (self.case_insensitive && first().eq_ignore_ascii_case("-StandardInput")))
        {
            let mut command_line_args = vec![None; args.len()];
            command_line_args[0] = args[0].clone();
            self.command_line_args = Some(command_line_args);
            self.keyword_value_pairs = true;
        } else {
            // make a defensive copy of the array
            let mut command_line_args = vec![None; args.len()];
            for i in 0..args.len() {
                command_line_args[i] = args[i].clone();
            }
            self.command_line_args = Some(command_line_args);
        }
    }

    /// Java `setDebug`.  Sets debug and returns the previous setting of debug.
    pub fn set_debug(&mut self, input: bool) -> bool {
        let old_debug = self.debug;
        self.debug = input;
        old_debug
    }

    /// Java `appendCommandLineArgs`.  Append addition arguments onto the command line
    /// argument list.
    pub fn append_command_line_args(&mut self, append: &[Option<String>]) {
        let existing_args = match self.command_line_args.take() {
            None => panic!("java.lang.NullPointerException"),
            Some(existing_args) => existing_args,
        };
        let mut command_line_args = vec![None; existing_args.len() + append.len()];
        for i in 0..existing_args.len() {
            command_line_args[i] = existing_args[i].clone();
        }
        for i in 0..append.len() {
            command_line_args[i + existing_args.len()] = append[i].clone();
        }
        self.command_line_args = Some(command_line_args);
    }

    /// Java `getCommandLineArgs`.  Each argument (white space separated command line
    /// entry) is store in a separate element of the return String[].
    pub fn get_command_line_args(&self) -> Option<Vec<Option<String>>> {
        let command_line_args = match &self.command_line_args {
            None => return None,
            Some(command_line_args) => command_line_args,
        };
        // make a defensive copy of the array
        let mut safe_array = vec![None; command_line_args.len()];
        for i in 0..command_line_args.len() {
            safe_array[i] = command_line_args[i].clone();
        }
        Some(safe_array)
    }

    /// Java `getCommandLineLength`.
    pub fn get_command_line_length(&self) -> i32 {
        match &self.command_line_args {
            None => 0,
            Some(command_line_args) => command_line_args.len() as i32,
        }
    }

    /// Java `appendInputArgument`.  Append to the input argument to the input argument
    /// list; `inputArg` is copied into the internal input argument list.
    pub fn append_input_argument(&mut self, input_arg: &ComScriptInputArg) {
        self.stdin_args
            .push(Rc::new(RefCell::new(ComScriptInputArg::new_from(
                input_arg,
            ))));
    }

    /// Java `getInputArguments`.  These are the values (and associated comments) that
    /// are meant for the standard input to the command.  `LinkedList.toArray` copies the
    /// array, not the elements, so the returned handles alias the list's.
    pub fn get_input_arguments(&self) -> Vec<Rc<RefCell<ComScriptInputArg>>> {
        let mut input_args: Vec<Rc<RefCell<ComScriptInputArg>>> =
            Vec::with_capacity(self.stdin_args.len());
        for input_arg in self.stdin_args.iter() {
            input_args.push(Rc::clone(input_arg));
        }
        input_args
    }

    /// Java `setInputArgument`.  Set the ith input argument to the supplied parameter.
    pub fn set_input_argument(&mut self, index: i32, input_arg: &ComScriptInputArg) {
        // create a defensive copy of the input argument object
        self.stdin_args[index as usize] =
            Rc::new(RefCell::new(ComScriptInputArg::new_from(input_arg)));
    }

    /// Java `setInputArguments`.  Set all of the input arguments, erasing any existing
    /// input arguments.
    pub fn set_input_arguments(&mut self, input_args: &[Rc<RefCell<ComScriptInputArg>>]) {
        // Clear the current list
        self.stdin_args.clear();

        // create a defensive copy of the input argument object
        for input_arg in input_args.iter() {
            self.stdin_args
                .push(Rc::new(RefCell::new(ComScriptInputArg::new_from(
                    &input_arg.borrow(),
                ))));
        }
    }

    /// Java `useKeywordValue`.
    ///
    /// Switch to using keyword/value pairs.  All of the current input arguments
    /// are deleted, and -StandardInput is added to the command line.  This
    /// should be done right before updating all of the values for converting a
    /// old style script to new.  The function has no effect if the script is
    /// already using keyword/value pairs.
    pub fn use_keyword_value(&mut self) {
        if !self.keyword_value_pairs {
            self.stdin_args.clear();
            self.keyword_value_pairs = true;
            self.command_line_args = Some(vec![Some("-StandardInput".to_string())]);
        }
    }

    /// Java `hasKeyword`.  Returns true if the command uses keyword/value pairs and the
    /// specified keyword is being used.
    pub fn has_keyword(&self, keyword: Option<&str>) -> Result<bool, InvalidParameterException> {
        if !self.keyword_value_pairs {
            return Err(InvalidParameterException::new(Some(&format!(
                "Command {} does not use keyword/value pairs",
                self.command.as_deref().unwrap_or("null")
            ))));
        }
        if keyword.is_none() {
            return Ok(false);
        }
        if self.find_key(keyword) >= 0 {
            return Ok(true);
        }
        Ok(false)
    }

    /// Java `getValue`.  Returns the (first) value associated with the specified keyword
    /// or an empty string if the keyowrd is not present, and null when the keyword is
    /// present with no value.  The declared `InvalidParameterException` is never thrown.
    pub fn get_value(
        &self,
        keyword: Option<&str>,
    ) -> Result<Option<String>, InvalidParameterException> {
        let idx = self.find_key(keyword);
        if idx >= 0 {
            let input_arg = self.stdin_args[idx as usize].borrow();
            let argument = match input_arg.get_argument() {
                None => panic!("java.lang.NullPointerException"),
                Some(argument) => argument,
            };
            let tokens = java_lang_string_split_limit(
                java_lang_string_trim(argument),
                &Regex::new(r"\s+").unwrap(),
                2,
            );
            if tokens.len() < 2 {
                return Ok(None);
            }
            return Ok(Some(tokens[1].clone()));
        }
        Ok(Some(String::new()))
    }

    /// Java `getValues`.  Returns all of the values associate with a keyword.
    pub fn get_values(&self, keyword: Option<&str>) -> Vec<Option<String>> {
        let mut values: Vec<Option<String>> = Vec::new();
        for input_arg in self.stdin_args.iter() {
            let input_arg = input_arg.borrow();
            let argument = match input_arg.get_argument() {
                None => panic!("java.lang.NullPointerException"),
                Some(argument) => argument,
            };
            let tokens = java_lang_string_split_limit(
                java_lang_string_trim(argument),
                &Regex::new(r"\s+").unwrap(),
                2,
            );
            let matches = match keyword {
                // `tokens[0].equals(null)` is false, and `equalsIgnoreCase(null)` is too.
                None => false,
                Some(keyword) => {
                    tokens[0] == keyword
                        || (self.case_insensitive && tokens[0].eq_ignore_ascii_case(keyword))
                }
            };
            if matches && tokens.len() > 1 {
                values.push(Some(tokens[1].clone()));
            }
        }
        if !values.is_empty() {
            return values;
        }
        Vec::new()
    }

    /// Java `setValue`.  Sets the specified key to the specified value.  The key will be
    /// created if it does not exist.
    ///
    /// Note that the source's `oldDebug` is initialised to false and never assigned from
    /// `inputArg.setDebug`'s return value, so the restore always sets debug off.
    pub fn set_value(&mut self, keyword: Option<&str>, value: Option<&str>) {
        let idx = self.find_key(keyword);
        let input_arg: Rc<RefCell<ComScriptInputArg>> = if idx == -1 {
            let input_arg = Rc::new(RefCell::new(ComScriptInputArg::new()));
            self.stdin_args.push(Rc::clone(&input_arg));
            input_arg
        } else {
            Rc::clone(&self.stdin_args[idx as usize])
        };
        let old_debug = false;
        if self.debug {
            input_arg.borrow_mut().set_debug(self.debug);
        }
        input_arg.borrow_mut().set_argument(Some(&format!(
            "{}{}{}",
            keyword.unwrap_or("null"),
            self.divider,
            value.unwrap_or("null")
        )));
        if self.debug {
            input_arg.borrow_mut().set_debug(old_debug);
        }
    }

    /// Java `setValues`.  Sets the keyword the values specified replacing any existing
    /// values.
    pub fn set_values(&mut self, keyword: Option<&str>, values: &[Option<String>]) {
        self.delete_key_all(keyword);
        for value in values.iter() {
            let input_arg = Rc::new(RefCell::new(ComScriptInputArg::new()));
            self.stdin_args.push(Rc::clone(&input_arg));
            input_arg.borrow_mut().set_argument(Some(&format!(
                "{}{}{}",
                keyword.unwrap_or("null"),
                self.divider,
                value.as_deref().unwrap_or("null")
            )));
        }
    }

    /// Java `setValuesInterleaved`.
    ///
    /// Adds values in an interleaved fashion with the first values of each element,
    /// followed by second values, and so on.  Elements of valuesArray may be null.
    ///
    /// Note that the source reads `valuesArray.length` before its own null check, so a
    /// null `valuesArray` throws a NullPointerException there; and that the `i == 0`
    /// branch raises the outer loop's own bound as it goes.
    pub fn set_values_interleaved(
        &mut self,
        keyword_array: Option<&[Option<String>]>,
        values_array: Option<&[Option<Vec<Option<String>>>]>,
    ) {
        let keyword_array = match keyword_array {
            None => return,
            Some(keyword_array) => keyword_array,
        };
        if keyword_array.is_empty() {
            return;
        }
        let values_array_length = match values_array {
            None => panic!("java.lang.NullPointerException"),
            Some(values_array) => values_array.len(),
        };
        let max_array_len = std::cmp::max(keyword_array.len() as i32, values_array_length as i32);
        let mut max_values_len = 1i32;
        if let Some(values_array) = values_array {
            if !values_array.is_empty() {
                if let Some(first) = &values_array[0] {
                    max_values_len = std::cmp::max(max_values_len, first.len() as i32);
                }
            }
        }
        // Interleave arguments - order is the first element from each parameter, then the
        // second from each, and so on
        let mut i = 0i32;
        while i < max_values_len {
            for j in 0..max_array_len {
                let mut keyword: Option<&str> = None;
                if j < keyword_array.len() as i32 {
                    keyword = keyword_array[j as usize].as_deref();
                }
                // Nothing to do if there is no keyword
                let keyword = match keyword {
                    None => continue,
                    Some(keyword) => keyword,
                };
                let mut values_len = -1i32;
                // Add an entry when there is value available
                if let Some(values_array) = values_array {
                    if j < values_array.len() as i32 {
                        if let Some(values) = &values_array[j as usize] {
                            values_len = values.len() as i32;
                        }
                    }
                }
                if i == 0 {
                    // always delete old parameters, even if the new values weren't passed
                    self.delete_key_all(Some(keyword));
                    if j > 0 {
                        // On the first loop of the outer loop, increase the limit for the
                        // outer loop, so that all the values of this parameter will be
                        // included.
                        let values_array = match values_array {
                            None => panic!("java.lang.NullPointerException"),
                            Some(values_array) => values_array,
                        };
                        if values_array[j as usize].is_some() {
                            max_values_len = std::cmp::max(max_values_len, values_len);
                        }
                    }
                }
                // Check if this parameter contains the ith value (parameters may have
                // different numbers of values.
                if i < values_len {
                    let input_arg = Rc::new(RefCell::new(ComScriptInputArg::new()));
                    self.stdin_args.push(Rc::clone(&input_arg));
                    let value = match values_array {
                        None => "null".to_string(),
                        Some(values_array) => match &values_array[j as usize] {
                            None => "null".to_string(),
                            Some(values) => {
                                values[i as usize].as_deref().unwrap_or("null").to_string()
                            }
                        },
                    };
                    input_arg
                        .borrow_mut()
                        .set_argument(Some(&format!("{}{}{}", keyword, self.divider, value)));
                }
            }
            i += 1;
        }
    }

    /// Java `addKey`.  Add a new key and value to the standard input argument list.
    pub fn add_key(&mut self, keyword: Option<&str>, value: Option<&str>) {
        let new_arg = Rc::new(RefCell::new(ComScriptInputArg::new()));
        new_arg.borrow_mut().set_argument(Some(&format!(
            "{}{}{}",
            keyword.unwrap_or("null"),
            self.divider,
            value.unwrap_or("null")
        )));
        self.stdin_args.push(new_arg);
    }

    /// Java `deleteKey`.  Delete the comScriptInputArg elements it contains the
    /// specified key.  Returns true if the keyword was found, false if it was not found.
    pub fn delete_key(&mut self, keyword: Option<&str>) -> bool {
        let idx = self.find_key(keyword);
        if idx >= 0 {
            self.stdin_args.remove(idx as usize);
            return true;
        }
        false
    }

    /// Java `deleteKeyAll`.  Delete all instances of the the keyword the std input
    /// arguments.
    pub fn delete_key_all(&mut self, keyword: Option<&str>) {
        let case_insensitive = self.case_insensitive;
        self.stdin_args.retain(|input_arg| {
            let input_arg = input_arg.borrow();
            let argument = match input_arg.get_argument() {
                None => panic!("java.lang.NullPointerException"),
                Some(argument) => argument,
            };
            let tokens = java_lang_string_split_limit(
                java_lang_string_trim(argument),
                &Regex::new(r"\s+").unwrap(),
                2,
            );
            let matches = match keyword {
                None => false,
                Some(keyword) => {
                    tokens[0] == keyword
                        || (case_insensitive && tokens[0].eq_ignore_ascii_case(keyword))
                }
            };
            !matches
        });
    }

    /// Java private `findKey`.  Returns the index within the stdinArgs list of the first
    /// ComScriptInputArg containing the specified key or -1 if the keyword is not found.
    fn find_key(&self, keyword: Option<&str>) -> i32 {
        let keyword = match keyword {
            None => return -1,
            Some(keyword) => keyword,
        };
        for i in 0..self.stdin_args.len() {
            let input_arg = self.stdin_args[i].borrow();
            let argument = match input_arg.get_argument() {
                None => panic!("java.lang.NullPointerException"),
                Some(argument) => argument,
            };
            let tokens = java_lang_string_split_limit(
                java_lang_string_trim(argument),
                &Regex::new(r"\s+").unwrap(),
                2,
            );
            if tokens[0] == keyword
                || (self.case_insensitive && tokens[0].eq_ignore_ascii_case(keyword))
            {
                return i as i32;
            }
        }
        -1
    }

    /// Java `containsOption`.  Returns true if option is a keyword in this command or in
    /// the command line.  Checks the command line if option is not one of the keywords.
    ///
    /// **The command-line loop can never match.**  The source writes
    /// `commandLineArgs.equals(option)` - comparing the `String[]` itself, not
    /// `commandLineArgs[i]`, to a `String` - so `Object.equals` is false for every
    /// element.  The loop is translated as written.
    pub fn contains_option(&self, option: Option<&str>) -> bool {
        if self.keyword_value_pairs {
            // Check the keywords.
            match self.has_keyword(option) {
                Ok(true) => return true,
                Ok(false) => {}
                Err(e) => {
                    // KeywordValuePairs is set so this should not happen.
                    eprintln!("Error:  Unexpected exception.");
                    // `e.printStackTrace()`.  A Java stack trace is a property of the
                    // JVM, not of the program.
                    eprintln!("{}", e);
                }
            }
        }
        // Check the command line options
        let command_line_args = match &self.command_line_args {
            None => return false,
            Some(command_line_args) => command_line_args,
        };
        if command_line_args.is_empty() {
            return false;
        }
        for i in 0..command_line_args.len() {
            // `commandLineArgs[i] != null && commandLineArgs.equals(option)`: the second
            // operand compares the `String[]` object itself against a `String`, which
            // `Object.equals` answers false for, so the condition never holds.
            let command_line_args_equals_option = false;
            if command_line_args[i].is_some() && command_line_args_equals_option {
                return true;
            }
        }
        false
    }

    /// Java `isKeywordValuePairs`.
    pub fn is_keyword_value_pairs(&self) -> bool {
        self.keyword_value_pairs
    }
}

/// Java `toString`.  `"[" + getCommand() + " " + stdinArgs + "]"`, where
/// `LinkedList.toString()` renders as `[e1, e2]`.
impl std::fmt::Display for ComScriptCommand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut list = String::from("[");
        for (index, input_arg) in self.stdin_args.iter().enumerate() {
            if index > 0 {
                list.push_str(", ");
            }
            list.push_str(&input_arg.borrow().to_string());
        }
        list.push(']');
        write!(f, "[{} {}]", self.get_command().unwrap_or("null"), list)
    }
}
