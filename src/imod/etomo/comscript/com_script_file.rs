//! Typed, lossless-enough COM-script document boundary.
//!
//! IMOD COM files are command blocks beginning with `$`; following lines are
//! the command's standard input.  Keeping blocks typed lets managers inspect
//! and schedule them without passing a shell string around.

use crate::imod::etomo::process::system_program::ProcessCommand;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ComScriptBlock {
    /// Comment lines immediately preceding the `$command`, corresponding to
    /// `ComScriptCommand.headerComments`.
    pub header_comments: Vec<String>,
    pub program: String,
    pub args: Vec<String>,
    pub stdin: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ComScriptFile {
    pub path: Option<PathBuf>,
    pub preamble: Vec<String>,
    pub blocks: Vec<ComScriptBlock>,
}

impl ComScriptFile {
    pub fn parse(text: &str) -> Result<Self, String> {
        let mut script = Self::default();
        let mut current: Option<ComScriptBlock> = None;
        let mut comments = Vec::new();
        let mut continuation = false;
        for (number, line) in text.lines().enumerate() {
            // ComScript.java treats only a `$` in column zero (and never `$!`)
            // as a command.  This distinction matters for conventional COM
            // shell-comment lines and for standard-input indentation.
            if line.starts_with('#') {
                comments.push(line.to_owned());
            } else if line.starts_with('$') && !line.starts_with("$!") {
                if continuation {
                    return Err(format!(
                        "command starts before continuation at line {}",
                        number + 1
                    ));
                }
                if let Some(block) = current.take() {
                    script.blocks.push(block);
                }
                let words: Vec<_> = line[1..].split_whitespace().map(str::to_owned).collect();
                let Some((program, args)) = words.split_first() else {
                    return Err(format!("empty COM command at line {}", number + 1));
                };
                let mut args = args.to_vec();
                continuation = args.last().is_some_and(|arg| arg == "\\");
                if continuation {
                    args.pop();
                }
                current = Some(ComScriptBlock {
                    header_comments: std::mem::take(&mut comments),
                    program: program.clone(),
                    args,
                    stdin: Vec::new(),
                });
            } else if continuation {
                let Some(block) = current.as_mut() else {
                    return Err(format!(
                        "continuation without a command at line {}",
                        number + 1
                    ));
                };
                // Java moves comments encountered in a command continuation
                // into the command header, then appends non-comment tokens.
                block.header_comments.append(&mut comments);
                let mut args: Vec<_> = line.split_whitespace().map(str::to_owned).collect();
                continuation = args.last().is_some_and(|arg| arg == "\\");
                if continuation {
                    args.pop();
                }
                block.args.extend(args);
            } else if let Some(block) = current.as_mut() {
                // Comments associated with a standard-input parameter remain
                // immediately before it when written, as in ComScriptInputArg.
                block.stdin.append(&mut comments);
                block.stdin.push(line.to_owned());
            } else {
                script.preamble.append(&mut comments);
                script.preamble.push(line.to_owned());
            }
        }
        if continuation {
            return Err("COM command continuation at end of file".to_owned());
        }
        if let Some(block) = current {
            script.blocks.push(block);
        } else {
            script.preamble.append(&mut comments);
        }
        Ok(script)
    }
    pub fn load(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref();
        let mut script = Self::parse(&fs::read_to_string(path)?)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        script.path = Some(path.to_owned());
        Ok(script)
    }
    pub fn to_text(&self) -> String {
        let mut lines = self.preamble.clone();
        for block in &self.blocks {
            lines.extend(block.header_comments.iter().cloned());
            let mut line = format!("${}", block.program);
            if !block.args.is_empty() {
                line.push(' ');
                line.push_str(&block.args.join(" "));
            }
            lines.push(line);
            lines.extend(block.stdin.iter().cloned());
        }
        format!("{}\n", lines.join("\n"))
    }
    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        fs::write(path, self.to_text())
    }
    /// Convert each COM block into an argv-based command.  No shell is used;
    /// the block's remaining source lines are sent through standard input.
    pub fn process_commands(&self) -> Vec<(String, ProcessCommand)> {
        self.blocks
            .iter()
            .enumerate()
            .map(|(index, block)| {
                let mut command = ProcessCommand::new(&block.program)
                    .args(block.args.iter().cloned())
                    .stdin_lines(block.stdin.iter().cloned());
                if let Some(path) = self.path.as_ref().and_then(|path| path.parent()) {
                    command = command.current_dir(path);
                }
                (format!("{}:{}", block.program, index + 1), command)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn preserves_blocks_and_standard_input() {
        let script = ComScriptFile::parse(
            "# preamble\n$tilt -StandardInput\nInputFile a.st\n$trimvol in out\n",
        )
        .unwrap();
        assert_eq!(script.blocks.len(), 2);
        assert_eq!(script.blocks[0].program, "tilt");
        assert_eq!(script.blocks[0].stdin, ["InputFile a.st"]);
        assert_eq!(ComScriptFile::parse(&script.to_text()).unwrap(), script);
    }
    #[test]
    fn command_comments_and_continuations_follow_comscript_rules() {
        let script = ComScriptFile::parse(
            "# head\n$tilt -a one \\\ntwo\n# input comment\nvalue\n$! shell note\n",
        )
        .unwrap();
        assert_eq!(script.blocks[0].header_comments, ["# head"]);
        assert_eq!(script.blocks[0].args, ["-a", "one", "two"]);
        assert_eq!(
            script.blocks[0].stdin,
            ["# input comment", "value", "$! shell note"]
        );
    }
}
