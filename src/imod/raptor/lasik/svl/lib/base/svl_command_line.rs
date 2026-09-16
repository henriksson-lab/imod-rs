//! Owned replacement for the command-line macros in `svlCommandLine.h`.
//!
//! The C++ interface spliced parser control flow into each caller with macros.
//! This iterator retains the same standard switches and token consumption while
//! making application-specific option handling ordinary Rust code.

use super::svl_config_manager::SvlConfigurationManager;
use super::svl_logger::{SvlLogLevel, initialize, log_message, set_log_level};

pub const STANDARD_OPTIONS_USAGE: &str = "  -help             :: display application usage\n\
  -config <xml>     :: configure SVL from XML file\n\
  -set <m> <n> <v>  :: set (configuration) <m>::<n> to value <v>\n\
  -profile          :: profile code\n\
  -quiet            :: only show warnings and errors\n\
  -verbose          :: show verbose messages\n\
  -debug            :: show debug messages\n\
  -log <filename>   :: log filename\n\
  -threads <max>    :: set maximum number of threads\n";

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SvlCommandLineItem {
    /// The source `-config` form without a filename prints the registry.
    ShowRegistry,
    Help,
    ApplicationOption(String),
    Positional(String),
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SvlCommandLineState {
    pub profiling_enabled: bool,
    pub max_threads: i32,
}

/// Token iterator corresponding to `SVL_BEGIN_CMDLINE_PROCESSING`.
pub struct SvlCommandLine<'a> {
    args: &'a [String],
    index: usize,
    pub state: SvlCommandLineState,
}

impl<'a> SvlCommandLine<'a> {
    /// The executable name is intentionally skipped, just as `ARGV + 1` is.
    pub fn new(argv: &'a [String]) -> Self {
        Self {
            args: argv,
            index: 1,
            state: SvlCommandLineState::default(),
        }
    }

    pub fn remaining(&self) -> &[String] {
        &self.args[self.index..]
    }

    pub fn take_string(&mut self) -> Result<String, String> {
        let value = self
            .args
            .get(self.index)
            .cloned()
            .ok_or_else(|| "not enough command-line arguments".to_string())?;
        self.index += 1;
        Ok(value)
    }

    pub fn take_int(&mut self) -> Result<i32, String> {
        Ok(self.take_string()?.parse::<i32>().unwrap_or(0))
    }

    pub fn take_real(&mut self) -> Result<f64, String> {
        Ok(self.take_string()?.parse::<f64>().unwrap_or(0.0))
    }

    /// Process one source standard switch or yield the next client token.
    pub fn next(
        &mut self,
        configuration: &mut SvlConfigurationManager,
    ) -> Result<Option<SvlCommandLineItem>, String> {
        let argument = match self.take_string() {
            Ok(argument) => argument,
            Err(_) => return Ok(None),
        };
        let _ = log_message(
            SvlLogLevel::Debug,
            &format!("processing command line argument {argument}"),
        );
        match argument.as_str() {
            "-config" => {
                if self.index == self.args.len() {
                    Ok(Some(SvlCommandLineItem::ShowRegistry))
                } else {
                    configuration.configure_file(self.take_string()?)?;
                    Ok(None)
                }
            }
            "-set" => {
                let module = self.take_string()?;
                let name = self.take_string()?;
                let value = self.take_string()?;
                configuration.configure(&module, &name, &value)?;
                Ok(None)
            }
            "-profile" => {
                self.state.profiling_enabled = true;
                Ok(None)
            }
            "-quiet" => {
                set_log_level(SvlLogLevel::Warning);
                Ok(None)
            }
            "-verbose" | "-v" => {
                set_log_level(SvlLogLevel::Verbose);
                Ok(None)
            }
            "-debug" => {
                set_log_level(SvlLogLevel::Debug);
                Ok(None)
            }
            "-log" => {
                initialize(Some(&self.take_string()?), true, None)
                    .map_err(|error| error.to_string())?;
                Ok(None)
            }
            "-threads" => {
                self.state.max_threads = self.take_int()?;
                Ok(None)
            }
            "-help" => Ok(Some(SvlCommandLineItem::Help)),
            // The macro expansion lets the application-specific option arms
            // run before `SVL_END_CMDLINE_PROCESSING` diagnoses an unknown
            // dash option, so expose it to the caller here.
            _ if argument.starts_with('-') => {
                Ok(Some(SvlCommandLineItem::ApplicationOption(argument)))
            }
            _ => Ok(Some(SvlCommandLineItem::Positional(argument))),
        }
    }

    /// Source `SVL_CMDLINE_STR_OPTION`: consumes a following value on match.
    pub fn string_option(
        &mut self,
        option: &str,
        argument: &str,
    ) -> Result<Option<String>, String> {
        (argument == option).then(|| self.take_string()).transpose()
    }

    /// Source `SVL_CMDLINE_INT_OPTION`.
    pub fn int_option(&mut self, option: &str, argument: &str) -> Result<Option<i32>, String> {
        (argument == option).then(|| self.take_int()).transpose()
    }

    /// Source `SVL_CMDLINE_REAL_OPTION`.
    pub fn real_option(&mut self, option: &str, argument: &str) -> Result<Option<f64>, String> {
        (argument == option).then(|| self.take_real()).transpose()
    }

    /// Source `SVL_CMDLINE_BOOL_OPTION`.
    pub fn bool_option(option: &str, argument: &str) -> bool {
        argument == option
    }

    /// Source `SVL_CMDLINE_BOOL_TOGGLE_OPTION`.
    pub fn bool_toggle_option(option: &str, argument: &str, value: &mut bool) -> bool {
        if argument == option {
            *value = !*value;
            true
        } else {
            false
        }
    }

    /// Source `SVL_CMDLINE_VEC_OPTION`.
    pub fn vector_option(
        &mut self,
        option: &str,
        argument: &str,
        values: &mut Vec<String>,
    ) -> Result<bool, String> {
        if argument != option {
            return Ok(false);
        }
        values.push(self.take_string()?);
        Ok(true)
    }

    /// Source `SVL_CMDLINE_OPTION_BEGIN/END`, returning exactly `count` following tokens.
    pub fn option_arguments(
        &mut self,
        option: &str,
        argument: &str,
        count: usize,
    ) -> Result<Option<Vec<String>>, String> {
        if argument != option {
            return Ok(None);
        }
        (0..count)
            .map(|_| self.take_string())
            .collect::<Result<Vec<_>, _>>()
            .map(Some)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_options_consume_their_source_argument_counts() {
        let argv = vec!["app", "-profile", "-threads", "12", "-v", "file"]
            .into_iter()
            .map(str::to_string)
            .collect::<Vec<_>>();
        let mut command_line = SvlCommandLine::new(&argv);
        let mut manager = SvlConfigurationManager::new();
        assert_eq!(command_line.next(&mut manager).unwrap(), None);
        assert!(command_line.state.profiling_enabled);
        assert_eq!(command_line.next(&mut manager).unwrap(), None);
        assert_eq!(command_line.state.max_threads, 12);
        assert_eq!(command_line.next(&mut manager).unwrap(), None);
        assert_eq!(
            command_line.next(&mut manager).unwrap(),
            Some(SvlCommandLineItem::Positional("file".into()))
        );
    }

    #[test]
    fn application_options_keep_source_atoi_and_atof_fallbacks() {
        let argv = vec!["app", "nope", "bad"]
            .into_iter()
            .map(str::to_string)
            .collect::<Vec<_>>();
        let mut command_line = SvlCommandLine::new(&argv);
        let argument = command_line.take_string().unwrap();
        assert_eq!(command_line.int_option("nope", &argument).unwrap(), Some(0));
        assert_eq!(command_line.take_real().unwrap(), 0.0);
    }
}
