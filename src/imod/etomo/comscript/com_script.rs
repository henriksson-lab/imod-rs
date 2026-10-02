//! `IMOD/Etomo/src/etomo/comscript/ComScript.java`.
//!
//! Description: This object models a IMOD Com script.
//!
//! Copyright: Copyright 2002 - 2025 by the Regents of the University of Colorado
//!
//! Organization: Dept. of MCD Biology, University of Colorado
//!
//! **Aliasing.**  `scriptCommands` is an `ArrayList` of `ComScriptCommand`
//! *references*: `getScriptCommand` hands out the very object in the list, callers
//! mutate it in place (`CommandParam.updateComScriptCommand`), and the copy constructor
//! shares the source's objects.  The elements are therefore
//! `Rc<RefCell<ComScriptCommand>>`, which is what a Java reference is here.  A
//! `ComScript` lives on the event dispatch thread (`ComScriptManager` is driven from
//! dialog actions) and is not `Send`.
//!
//! **Index checks.**  Java `ArrayList.get`/`add` throw the unchecked
//! `IndexOutOfBoundsException` for an index outside the list.  Where `ComScriptUtil`
//! can reach that with a command index one past the last command (a previous command
//! that is the last command in the script), the Java aborts the Swing action with an
//! uncaught exception.  Fixed in translation (`BUGS.md`): an index outside the list
//! does not name the command (`getScriptCommandIndex` returns -1), and
//! `getScriptCommand(String, int, ...)` creates the command at the end of the list when
//! `addNew` is set and the index is exactly the list size, and otherwise throws the
//! method's own `BadComScriptException`.

use super::bad_com_script_exception::BadComScriptException;
use super::com_script_command::ComScriptCommand;
use super::com_script_input_arg::ComScriptInputArg;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::r#type::const_etomo_number::java_lang_string_trim;
use crate::imod::etomo::util::utilities::{
    java_io_file_get_absolute_path, java_io_file_get_name, java_io_file_normalize,
    java_lang_string_split,
};
use regex::Regex;
use std::cell::RefCell;
use std::io::Write;
use std::rc::Rc;

/// The checked exceptions `readComFile` declares: `FileNotFoundException` and
/// `IOException` (`Io`), and `BadComScriptException`.
#[derive(Debug)]
pub enum ReadComFileError {
    Io(std::io::Error),
    BadComScript(BadComScriptException),
}

impl std::fmt::Display for ReadComFileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReadComFileError::Io(e) => write!(f, "{e}"),
            ReadComFileError::BadComScript(e) => write!(f, "{e}"),
        }
    }
}

impl From<std::io::Error> for ReadComFileError {
    fn from(e: std::io::Error) -> ReadComFileError {
        ReadComFileError::Io(e)
    }
}

impl From<BadComScriptException> for ReadComFileError {
    fn from(e: BadComScriptException) -> ReadComFileError {
        ReadComFileError::BadComScript(e)
    }
}

/// Java `ComScript`.
pub struct ComScript {
    /// Java field `comFile` (a `java.io.File`, held as its normalized path).
    // TODO check for necessary defensive copying
    com_file: String,
    /// Java field `scriptCommands`, initialised to `new ArrayList()`.
    script_commands: Vec<Rc<RefCell<ComScriptCommand>>>,
    /// Java field `parseComments`, initialised to true.
    parse_comments: bool,
    /// Java field `commandLoaded`, initialised to false.  True when at least one
    /// command has be found or created.
    command_loaded: bool,
}

impl ComScript {
    /// Java `ComScript(File)`.
    pub fn new(com_file: &str) -> ComScript {
        ComScript {
            com_file: java_io_file_normalize(com_file),
            script_commands: Vec::new(),
            parse_comments: true,
            command_loaded: false,
        }
    }

    /// Java `ComScript(File, ComScript)`.  Copy constructor, creates a deep copy of the
    /// supplied ComScript object.  (The source's "deep copy" shares the source's
    /// `ComScriptCommand` objects, and leaves `commandLoaded` false.)
    pub fn new_from(com_file: &str, src_com_script: &ComScript) -> ComScript {
        let mut com_script = ComScript::new(com_file);
        // Copy the ComScriptCommands from the source object
        let n_commands = src_com_script.get_command_count();
        com_script.script_commands.reserve(n_commands as usize);
        for i in 0..n_commands {
            com_script
                .script_commands
                .push(src_com_script.get_script_command(i).unwrap());
        }
        com_script
    }

    /// Java `readComFile`.  Read in the specified com script from the file system
    /// parsing the command, comments and arguments into an internal representation.
    ///
    /// `BufferedReader.readLine` ends a line at `\n`, `\r` or `\r\n`, and `FileReader`
    /// decodes with the platform charset (UTF-8, malformed input replaced).
    pub fn read_com_file(
        &mut self,
        case_insensitive: bool,
        separate_with_a_space: bool,
    ) -> Result<(), ReadComFileError> {
        // Open the com file for reading using a buffered reader
        if !std::path::Path::new(&self.com_file).exists() {
            return Ok(());
        }
        let bytes = std::fs::read(&self.com_file)?;
        let text = String::from_utf8_lossy(&bytes).into_owned();
        let mut lines: Vec<String> = Vec::new();
        {
            let mut current = String::new();
            let mut chars = text.chars().peekable();
            let mut pending = false;
            while let Some(c) = chars.next() {
                if c == '\n' {
                    lines.push(std::mem::take(&mut current));
                    pending = false;
                } else if c == '\r' {
                    lines.push(std::mem::take(&mut current));
                    pending = false;
                    if chars.peek() == Some(&'\n') {
                        chars.next();
                    }
                } else {
                    current.push(c);
                    pending = true;
                }
            }
            if pending {
                lines.push(current);
            }
        }
        let whitespace = Regex::new("[ \t\n\u{0B}\u{0C}\r]+").unwrap();

        // Read in the lines of the command file, assigning each one to the correct
        // list
        let mut flg_continuation = false;
        let mut current_script_command: Option<Rc<RefCell<ComScriptCommand>>> = None;
        let mut current_comment_block: Vec<Option<String>> = Vec::new();

        let mut line_number = 0;
        for line in lines.iter() {
            line_number += 1;

            // `line.matches("^\\$[^!].*")`: a `$`, then any character but `!`, then any
            // characters that are not line terminators (`.` in a Java regex).
            let mut line_chars = line.chars();
            let is_command_line = line_chars.next() == Some('$')
                && matches!(line_chars.next(), Some(second) if second != '!')
                && !line_chars
                    .any(|c| matches!(c, '\n' | '\r' | '\u{85}' | '\u{2028}' | '\u{2029}'));

            // If the first character is a pound place on the comment list
            if line.starts_with('#') {
                current_comment_block.push(Some(line.clone()));
            }
            // If the first character is a $ not followed by !
            // create a new ComScriptCommand object and insert onto scriptCommands
            // insert the current comment block into the header comments then clear
            // the current comment block
            // parse the command line setting the command and command line
            // arguments
            else if is_command_line {
                let command = Rc::new(RefCell::new(ComScriptCommand::new(
                    case_insensitive,
                    separate_with_a_space,
                )));
                self.script_commands.push(Rc::clone(&command));
                current_script_command = Some(Rc::clone(&command));
                let mut current = command.borrow_mut();

                if !current_comment_block.is_empty() {
                    let comment_array = current_comment_block.clone();
                    current.set_header_comments(&comment_array);
                    current_comment_block.clear();
                }

                // Split the line into tokens at whitespace boundaries
                let no_dollar_sign = &line['$'.len_utf8()..];
                let no_dollar_sign = java_lang_string_trim(no_dollar_sign);
                let tokens = java_lang_string_split(no_dollar_sign, &whitespace);
                current.set_command(Some(tokens[0].as_str()));
                self.command_loaded = true;

                // Check to if see if this line is continued
                let mut n_shrink = 1;
                if tokens[tokens.len() - 1] == "\\" {
                    n_shrink = 2;
                    flg_continuation = true;
                }

                // `new String[tokens.length - nShrink]`: a line holding only `$\` gives
                // -1, which throws `NegativeArraySizeException("-1")`.  Every caller
                // (`ComScriptUtil.loadComScript`) catches it as an `Exception` and
                // reports its message, so it is returned as an error with the same
                // message (only the exception class differs).
                let n_args = tokens.len() as i64 - n_shrink;
                if n_args < 0 {
                    return Err(BadComScriptException::new(&n_args.to_string()).into());
                }
                let mut cmd_line_args: Vec<Option<String>> = vec![None; n_args as usize];
                for i in 0..cmd_line_args.len() {
                    cmd_line_args[i] = Some(tokens[i + 1].clone());
                }
                current.set_command_line_args(&cmd_line_args);
                // Force the comment parsing from the standard input lines to off
                // if a keyword/value pair input format is detected
                if current.is_keyword_value_pairs() {
                    self.parse_comments = false;
                }
            }
            // Otherwise the line is assumed to be an input parmeter to the current
            // command or a continuation line
            else if flg_continuation {
                let current_script_command = current_script_command.as_ref().unwrap();
                let mut current = current_script_command.borrow_mut();
                // Get any comments associated with the continuation line and add
                // them to the header comments
                if !current_comment_block.is_empty() {
                    let comment_array = current_comment_block.clone();
                    current.append_header_comments(&comment_array);
                    current_comment_block.clear();
                }

                // Split the line into tokens checking to see if the last token is
                // another line continuation
                let tokens = java_lang_string_split(java_lang_string_trim(line), &whitespace);
                let mut n_shrink = 0;
                if tokens[tokens.len() - 1] == "\\" {
                    n_shrink = 1;
                    flg_continuation = true;
                } else {
                    flg_continuation = false;
                }

                let mut cmd_line_args: Vec<Option<String>> = vec![None; tokens.len() - n_shrink];
                for i in 0..cmd_line_args.len() {
                    cmd_line_args[i] = Some(tokens[i].clone());
                }
                current.append_command_line_args(&cmd_line_args);
            } else {
                let current_script_command = match &current_script_command {
                    None => {
                        let description = format!(
                            "Input parameter found before command in {} line: {}",
                            java_io_file_get_absolute_path(&self.com_file),
                            line_number
                        );
                        return Err(BadComScriptException::new(&description).into());
                    }
                    Some(current_script_command) => current_script_command,
                };

                let mut input_arg = ComScriptInputArg::new();

                if !current_comment_block.is_empty() {
                    let comment_array = current_comment_block.clone();
                    input_arg.set_comments(&comment_array);
                    current_comment_block.clear();
                }

                input_arg.set_argument_parse_comments(Some(line), self.parse_comments);

                current_script_command
                    .borrow_mut()
                    .append_input_argument(&input_arg);
            }
        }
        // Close the com script
        Ok(())
    }

    /// Java `getCommandArray`.  Get the command names in script as a string array.  The
    /// array returned is in the same order as the commands in the script.
    pub fn get_command_array(&self) -> Vec<Option<String>> {
        let mut command_array: Vec<Option<String>> = vec![None; self.script_commands.len()];
        for i in 0..command_array.len() {
            let command = self.script_commands[i].borrow();
            command_array[i] = command.get_command().map(|command| command.to_string());
        }
        command_array
    }

    /// Java `getScriptCommand(int)`.  Return the specified ComSrciptCommand element
    /// according to the commandArray described in getCommandArray.  `None` where
    /// `ArrayList.get` would throw `IndexOutOfBoundsException`.
    pub fn get_script_command(&self, index: i32) -> Option<Rc<RefCell<ComScriptCommand>>> {
        if index < 0 || index as usize >= self.script_commands.len() {
            return None;
        }
        Some(Rc::clone(&self.script_commands[index as usize]))
    }

    /// Java `getScriptCommand(String, boolean, boolean)`.  Return the first instance of
    /// ComScriptCommand with the specified command.
    pub fn get_script_command_named(
        &mut self,
        cmd_name: &str,
        case_insensitive: bool,
        separate_with_a_space: bool,
    ) -> Result<Rc<RefCell<ComScriptCommand>>, BadComScriptException> {
        if !self.command_loaded {
            self.create_command(cmd_name, case_insensitive, separate_with_a_space);
        }
        for i in 0..self.script_commands.len() {
            let command = &self.script_commands[i];
            if command.borrow().get_command() == Some(cmd_name) {
                return Ok(Rc::clone(command));
            }
        }
        Err(BadComScriptException::new(&format!(
            "Did not find command: {}",
            cmd_name
        )))
    }

    /// Java `getScriptCommand(String, int, boolean, boolean, boolean)`.  Return the
    /// instance of ComScriptCommand with the specified command corresponding to
    /// commandIndex.  See the module comment for an index outside the list.
    pub fn get_script_command_named_at(
        &mut self,
        cmd_name: &str,
        command_index: i32,
        add_new: bool,
        case_insensitive: bool,
        separate_with_a_space: bool,
    ) -> Result<Rc<RefCell<ComScriptCommand>>, BadComScriptException> {
        if !self.command_loaded {
            self.create_command_at(
                cmd_name,
                command_index,
                case_insensitive,
                separate_with_a_space,
            );
        }
        if let Some(command) = self.get_script_command(command_index) {
            if command.borrow().get_command() == Some(cmd_name) {
                return Ok(command);
            }
        }
        if add_new && command_index >= 0 && command_index as usize <= self.script_commands.len() {
            self.create_command_at(
                cmd_name,
                command_index,
                case_insensitive,
                separate_with_a_space,
            );
            let command = self.get_script_command(command_index).unwrap();
            if command.borrow().get_command() == Some(cmd_name) {
                return Ok(command);
            }
        }
        Err(BadComScriptException::new(&format!(
            "Did not find command: {} at index {}",
            cmd_name, command_index
        )))
    }

    /// Java package-private `createCommand(String, boolean, boolean)`.
    pub fn create_command(
        &mut self,
        cmd_name: &str,
        case_insensitive: bool,
        separate_with_a_space: bool,
    ) -> i32 {
        let current_script_command = Rc::new(RefCell::new(ComScriptCommand::new(
            case_insensitive,
            separate_with_a_space,
        )));
        self.script_commands
            .push(Rc::clone(&current_script_command));
        current_script_command
            .borrow_mut()
            .set_command(Some(cmd_name));
        self.command_loaded = true;
        self.script_commands.len() as i32 - 1
    }

    /// Java package-private `createCommand(String, int, boolean, boolean)`.
    ///
    /// Fixed in translation: `ArrayList.add(int, E)` throws for an index outside
    /// `0..=size`; such an index appends instead (only reachable from
    /// `getScriptCommand(String, int, ...)` on a script with no command loaded).
    pub fn create_command_at(
        &mut self,
        cmd_name: &str,
        command_index: i32,
        case_insensitive: bool,
        separate_with_a_space: bool,
    ) {
        let current_script_command = Rc::new(RefCell::new(ComScriptCommand::new(
            case_insensitive,
            separate_with_a_space,
        )));
        if command_index >= 0 && command_index as usize <= self.script_commands.len() {
            self.script_commands
                .insert(command_index as usize, Rc::clone(&current_script_command));
        } else {
            self.script_commands
                .push(Rc::clone(&current_script_command));
        }
        current_script_command
            .borrow_mut()
            .set_command(Some(cmd_name));
        self.command_loaded = true;
    }

    /// Java `deleteCommand`.
    pub fn delete_command(&mut self, command_index: i32) {
        self.script_commands.remove(command_index as usize);
        if self.script_commands.is_empty() {
            self.command_loaded = true;
        }
    }

    /// Java `getScriptCommandIndex(String, boolean, boolean)`.  Return the index of the
    /// specified command or -1 if the command is not present in the script.
    pub fn get_script_command_index(
        &mut self,
        cmd_name: &str,
        case_insensitive: bool,
        separate_with_a_space: bool,
    ) -> i32 {
        self.get_script_command_index_add_new(
            cmd_name,
            false,
            case_insensitive,
            separate_with_a_space,
            None,
        )
    }

    /// Java `getScriptCommandIndex(String, boolean, boolean, String)`.  Return the index
    /// of the specified command or -1 if the command is not present in the script.
    pub fn get_script_command_index_required_option(
        &mut self,
        cmd_name: &str,
        case_insensitive: bool,
        separate_with_a_space: bool,
        required_option: Option<&str>,
    ) -> i32 {
        self.get_script_command_index_add_new(
            cmd_name,
            false,
            case_insensitive,
            separate_with_a_space,
            required_option,
        )
    }

    /// Java `getScriptCommandIndex(String, boolean, boolean, boolean, String)`.
    /// `requiredOption` - option name which must exist in command for it to match.
    pub fn get_script_command_index_add_new(
        &mut self,
        cmd_name: &str,
        add_new: bool,
        case_insensitive: bool,
        separate_with_a_space: bool,
        required_option: Option<&str>,
    ) -> i32 {
        if !self.command_loaded {
            self.create_command(cmd_name, case_insensitive, separate_with_a_space);
        }
        for i in 0..self.script_commands.len() {
            let command = self.script_commands[i].borrow();
            if command.get_command() == Some(cmd_name)
                && (required_option.is_none() || command.contains_option(required_option))
            {
                return i as i32;
            }
        }
        if add_new {
            return self.create_command(cmd_name, case_insensitive, separate_with_a_space);
        }
        -1
    }

    /// Java `getScriptCommandIndex(String, int, boolean, boolean, boolean)`.
    pub fn get_script_command_index_at_add_new(
        &mut self,
        cmd_name: &str,
        command_index: i32,
        add_new: bool,
        case_insensitive: bool,
        separate_with_a_space: bool,
    ) -> i32 {
        if !self.command_loaded {
            self.create_command(cmd_name, case_insensitive, separate_with_a_space);
        }
        if let Some(command) = self.get_script_command(command_index) {
            if command.borrow().get_command() == Some(cmd_name) {
                return command_index;
            }
        }
        if add_new {
            return self.create_command(cmd_name, case_insensitive, separate_with_a_space);
        }
        -1
    }

    /// Java `getScriptCommandIndex(String, int, boolean, boolean)`.
    pub fn get_script_command_index_at(
        &mut self,
        cmd_name: &str,
        command_index: i32,
        case_insensitive: bool,
        separate_with_a_space: bool,
    ) -> i32 {
        if !self.command_loaded {
            self.create_command(cmd_name, case_insensitive, separate_with_a_space);
        }
        if let Some(command) = self.get_script_command(command_index) {
            if command.borrow().get_command() == Some(cmd_name) {
                return command_index;
            }
        }
        -1
    }

    /// Java `writeComFile`.  Write out the command file currently represented by this
    /// object.  (`BadComScriptException` is declared but never thrown.)
    ///
    /// `Writer.write(null)` throws `NullPointerException`, which every caller catches
    /// as an `Exception`; a null input argument is returned as an `io::Error` with that
    /// name.  Null header comments/command-line arrays (never produced by
    /// `ComScriptCommand`) are written as empty.
    pub fn write_com_file(&self) -> std::io::Result<()> {
        if !etomo_director::INSTANCE.is_memory_available() {
            return Ok(());
        }
        // Open the com file for writing using a buffered writer
        let file = std::fs::File::create(&self.com_file)
            .map_err(|e| std::io::Error::new(e.kind(), format!("{} ({})", self.com_file, e)))?;
        let mut out = std::io::BufWriter::new(file);

        // Write out the the list of sript commands
        for script_command in self.script_commands.iter() {
            let script_command = script_command.borrow();

            // Write out the header comments if they exist
            let header_comments = script_command.get_header_comments().unwrap_or_default();
            for i in 0..header_comments.len() {
                out.write_all(header_comments[i].as_deref().unwrap_or("null").as_bytes())?;
                out.write_all(b"\n")?;
            }

            // Write out the the command and any command line arguments
            out.write_all(
                ("$".to_string() + script_command.get_command().unwrap_or("null")).as_bytes(),
            )?;
            let command_arguments = script_command.get_command_line_args().unwrap_or_default();
            for i in 0..command_arguments.len() {
                out.write_all(
                    (" ".to_string() + command_arguments[i].as_deref().unwrap_or("null"))
                        .as_bytes(),
                )?;
            }
            out.write_all(b"\n")?;

            // Write the input arguments for the command
            let input_args = script_command.get_input_arguments();
            for i in 0..input_args.len() {
                let input_arg = input_args[i].borrow();
                let arg_comments = input_arg.get_comments();
                for j in 0..arg_comments.len() {
                    out.write_all(arg_comments[j].as_deref().unwrap_or("null").as_bytes())?;
                    out.write_all(b"\n")?;
                }
                match input_arg.get_argument() {
                    None => {
                        return Err(std::io::Error::other("java.lang.NullPointerException"));
                    }
                    Some(argument) => out.write_all(argument.as_bytes())?,
                }
                out.write_all(b"\n")?;
            }
        }
        out.flush()?;
        Ok(())
    }

    /// Java `setScriptComand`.  Set the specified ScriptCommand object to a new
    /// ScriptCommand object.
    pub fn set_script_comand(&mut self, index: i32, new_script_command: &ComScriptCommand) {
        // make a defensive copy of the scriptCommand object
        self.script_commands[index as usize] =
            Rc::new(RefCell::new(ComScriptCommand::new_from(new_script_command)));
    }

    /// Java `addScriptComand`.  Adds a new command at index, shifting the existing
    /// commands to make room for it.
    pub fn add_script_comand(&mut self, index: i32, new_script_command: &ComScriptCommand) {
        // make a defensive copy of the scriptCommand object
        self.script_commands.insert(
            index as usize,
            Rc::new(RefCell::new(ComScriptCommand::new_from(new_script_command))),
        );
    }

    /// Java `removeScriptCommand`.  Remove the specified ComScriptCommand from the
    /// collection.
    pub fn remove_script_command(&mut self, index: i32) {
        self.script_commands.remove(index as usize);
    }

    /// Java `addScriptCommand`.  Add the specified ComScriptCommand to the end of the
    /// collection.
    pub fn add_script_command(&mut self, new_script_command: &ComScriptCommand) {
        self.script_commands
            .push(Rc::new(RefCell::new(ComScriptCommand::new_from(
                new_script_command,
            ))));
    }

    /// Java `getComFileName`.  Get the com file name: the absolute path of com file.
    pub fn get_com_file_name(&self) -> String {
        java_io_file_get_absolute_path(&self.com_file)
    }

    /// Java `getName`.  Get the com file name: the name of the com file.
    pub fn get_name(&self) -> String {
        java_io_file_get_name(&self.com_file)
    }

    /// Java `isCommandLoaded`.
    pub fn is_command_loaded(&self) -> bool {
        self.command_loaded
    }

    /// Java `getCommandCount`.  Get the number of commands in the script.
    pub fn get_command_count(&self) -> i32 {
        self.script_commands.len() as i32
    }

    /// Java `setParseComments`.
    pub fn set_parse_comments(&mut self, state: bool) {
        self.parse_comments = state;
    }
}

/// Java `toString`: `comFile.getAbsolutePath()`.
impl std::fmt::Display for ComScript {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&java_io_file_get_absolute_path(&self.com_file))
    }
}

#[cfg(test)]
mod tests {
    use super::ComScript;

    /// `IMOD/com/xcorr.com`, verbatim.
    const XCORR_COM: &str = "#
# TO RUN TILTXCORR
#
####CreatedVersion#### 3.4.4
#
# Add BordersInXandY to use a centered region smaller than the default
# or XMinAndMax and YMinAndMax  to specify a non-centered region
#
$tiltxcorr -StandardInput
InputFile\tg5a.st
OutputFile\tg5a.prexf
FirstTiltAngle\t-60.
TiltIncrement\t1.5
RotationAngle\t0.
FilterSigma1\t0.03
FilterRadius2\t0.25
FilterSigma2\t0.05
#SkipViews
";

    fn temp_dir(name: &str) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("imod-rs-comscript-{}-{}", name, std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Reading and writing back `xcorr.com` reproduces it, except the trailing comment
    /// block: `readComFile` attaches a comment block to the *next* command or input
    /// line, so a comment block after the last line is dropped, as in the source.
    #[test]
    fn xcorr_template_round_trip() {
        let dir = temp_dir("xcorr");
        let path = dir.join("xcorr.com");
        std::fs::write(&path, XCORR_COM).unwrap();

        let mut com_script = ComScript::new(path.to_str().unwrap());
        com_script.set_parse_comments(true);
        com_script.read_com_file(false, false).unwrap();
        assert!(com_script.is_command_loaded());
        assert_eq!(com_script.get_command_count(), 1);
        assert_eq!(
            com_script.get_command_array(),
            vec![Some("tiltxcorr".to_string())]
        );
        let command = com_script
            .get_script_command_named("tiltxcorr", false, false)
            .unwrap();
        assert!(command.borrow().is_keyword_value_pairs());
        assert_eq!(
            command.borrow().get_value(Some("FirstTiltAngle")).unwrap(),
            Some("-60.".to_string())
        );
        assert_eq!(
            com_script.get_script_command_index("tiltxcorr", false, false),
            0
        );
        assert_eq!(
            com_script.get_script_command_index("newstack", false, false),
            -1
        );

        com_script.write_com_file().unwrap();
        let written = std::fs::read_to_string(&path).unwrap();
        let expected = XCORR_COM.strip_suffix("#SkipViews\n").unwrap();
        assert_eq!(written, expected);

        // A second read/write of the written file is a fixed point.
        let mut again = ComScript::new(path.to_str().unwrap());
        again.read_com_file(false, false).unwrap();
        again.write_com_file().unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), expected);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A keyword with no value (`YAxisElongated`, as makecomfile writes it) parses as
    /// an EtomoBoolean2 that is on, and is written back as `YAxisElongated\t`.
    #[test]
    fn keyword_without_value_is_a_true_boolean() {
        use crate::imod::etomo::r#type::etomo_boolean2::EtomoBoolean2;
        let dir = temp_dir("kwnoval");
        let path = dir.join("findbeads3d.com");
        std::fs::write(
            &path,
            "$findbeads3d -StandardInput\nInputFile\ta.mrc\nTiltFile ts8.tlt\nYAxisElongated\n",
        )
        .unwrap();
        let mut com_script = ComScript::new(path.to_str().unwrap());
        com_script.read_com_file(false, false).unwrap();
        let command = com_script
            .get_script_command_named("findbeads3d", false, false)
            .unwrap();
        assert!(
            command
                .borrow()
                .has_keyword(Some("YAxisElongated"))
                .unwrap()
        );
        let mut b = EtomoBoolean2::new_with_name("YAxisElongated");
        b.set_display_value_boolean(true);
        b.reset();
        b.parse(&command.borrow()).unwrap();
        assert!(b.is());
        b.update_com_script(&mut command.borrow_mut());
        assert!(
            command
                .borrow()
                .has_keyword(Some("YAxisElongated"))
                .unwrap()
        );
        // Unparsed (a com file the param creates): the display value (on) is
        // written, as FindBeads3dParam's new findbeads3d.com has it.
        let mut fresh = EtomoBoolean2::new_with_name("YAxisElongated");
        fresh.set_display_value_boolean(true);
        fresh.reset();
        command.borrow_mut().delete_key(Some("YAxisElongated"));
        fresh.update_com_script(&mut command.borrow_mut());
        assert!(
            command
                .borrow()
                .has_keyword(Some("YAxisElongated"))
                .unwrap()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A continuation line is appended to the command line, and an input line before
    /// any command is `BadComScriptException`.
    #[test]
    fn continuation_and_orphan_input() {
        let dir = temp_dir("cont");
        let path = dir.join("cont.com");
        std::fs::write(&path, "$goto \\\n  label\n$echo a b # c\n").unwrap();
        let mut com_script = ComScript::new(path.to_str().unwrap());
        com_script.read_com_file(false, false).unwrap();
        com_script.write_com_file().unwrap();
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "$goto label\n$echo a b # c\n"
        );

        let orphan = dir.join("orphan.com");
        std::fs::write(&orphan, "input\n$cmd\n").unwrap();
        let mut com_script = ComScript::new(orphan.to_str().unwrap());
        let error = com_script.read_com_file(false, false).unwrap_err();
        assert!(
            error
                .to_string()
                .starts_with("Input parameter found before command in ")
        );
        assert!(error.to_string().ends_with("orphan.com line: 1"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
