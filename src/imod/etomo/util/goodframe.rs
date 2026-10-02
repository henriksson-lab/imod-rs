//! `IMOD/Etomo/src/etomo/util/Goodframe.java`.
//!
//! Runs the external `goodframe` program through an `etomo.process.SystemProgram` and
//! parses its output.
//!
//! Copyright: Copyright 2005 - 2015 by the Regents of the University of Colorado
//!
//! Organization: Dept. of MCD Biology, University of Colorado

use crate::imod::etomo::base_manager::{self, BaseManager};
use crate::imod::etomo::process::process_messages::MessageType;
use crate::imod::etomo::process::system_program::SystemProgram;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::{ConstEtomoNumber, java_lang_string_trim};
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::util::utilities::{self, java_lang_string_split};
use regex::Regex;
use std::sync::LazyLock;

/// Java `"\\s+"`, with Java's `\s` class.
static WHITESPACE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new("[ \t\n\u{0B}\u{0C}\r]+").unwrap());

/// The three exceptions `run` declares, each carrying its message.
#[derive(Clone, Debug)]
pub enum GoodframeError {
    /// `java.io.IOException`.
    Io(String),
    /// `etomo.util.InvalidParameterException`.
    InvalidParameter(String),
    /// `java.lang.NumberFormatException`.
    NumberFormat(String),
}

impl GoodframeError {
    /// `Throwable.getMessage()`.
    pub fn get_message(&self) -> &str {
        match self {
            GoodframeError::Io(message)
            | GoodframeError::InvalidParameter(message)
            | GoodframeError::NumberFormat(message) => message,
        }
    }
}

impl std::fmt::Display for GoodframeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.get_message())
    }
}

/// Java `Goodframe`.
#[derive(Clone, Debug)]
pub struct Goodframe {
    /// Java private final field `axisID`.
    axis_id: AxisID,
    /// Java private final field `propertyUserDir`.
    property_user_dir: Option<String>,
    /// Java private field `output`.
    output: Option<Vec<EtomoNumber>>,
}

impl Goodframe {
    /// Java `Goodframe(String, AxisID)`.
    pub fn new(property_user_dir: Option<String>, axis_id: AxisID) -> Goodframe {
        Goodframe {
            axis_id,
            property_user_dir,
            output: None,
        }
    }

    /// Java `run(BaseManager, int, int)`.
    pub fn run_int(
        &mut self,
        manager: &'static dyn BaseManager,
        first_input: i32,
        second_input: i32,
    ) -> Result<(), GoodframeError> {
        self.run(
            manager,
            &[first_input.to_string(), second_input.to_string()],
        )
    }

    /// Java `run(BaseManager, String[])`.
    pub fn run(
        &mut self,
        manager: &'static dyn BaseManager,
        input: &[String],
    ) -> Result<(), GoodframeError> {
        utilities::timestamp_process_container_status(
            Some("run"),
            Some("goodframe"),
            Some(utilities::STARTED_STATUS),
        );
        // Run the goodframe command.
        let mut command_array: Vec<String> = Vec::with_capacity(input.len() + 1);
        command_array.push(format!(
            "{}goodframe",
            base_manager::get_imod_bin_path().unwrap_or_else(|| "null".to_string())
        ));
        for i in 0..input.len() {
            command_array.push(input[i].clone());
        }
        let groupframe = SystemProgram::new_array(
            Some(manager),
            self.property_user_dir.clone(),
            Some(command_array),
            self.axis_id,
        );
        groupframe.run();

        if groupframe.get_exit_value() != 0 {
            let messages = groupframe.get_process_messages();
            if messages.size(MessageType::Error) > 0 {
                let mut message = "groupframe returned an error:\n".to_string();
                for i in 0..messages.size(MessageType::Error) {
                    message =
                        message + messages.get(MessageType::Error, i).unwrap_or("null") + "\n";
                }
                utilities::timestamp_process_container_status(
                    Some("run"),
                    Some("goodframe"),
                    Some(utilities::FAILED_STATUS),
                );
                return Err(GoodframeError::InvalidParameter(message));
            }
        }
        // Throw an exception if the file can not be read
        let std_error = groupframe.get_std_error();
        if let Some(std_error) = &std_error
            && !std_error.is_empty()
        {
            let mut message = "groupframe returned an error:\n".to_string();
            for i in 0..std_error.len() {
                message = message + &std_error[i] + "\n";
            }
            utilities::timestamp_process_container_status(
                Some("run"),
                Some("goodframe"),
                Some(utilities::FAILED_STATUS),
            );
            return Err(GoodframeError::InvalidParameter(message));
        }

        // Parse the output
        let std_output = groupframe.get_std_output();
        // Goodframe.java:106-111 tests `stdOutput != null && stdOutput.length < 1` and
        // then reads `stdOutput[0]`, so a null output throws a NullPointerException.
        // Fixed in translation: a null output is treated like an empty one.
        let std_output = match std_output {
            Some(std_output) if !std_output.is_empty() => std_output,
            _ => {
                utilities::timestamp_process_container_status(
                    Some("run"),
                    Some("goodframe"),
                    Some(utilities::FAILED_STATUS),
                );
                return Err(GoodframeError::Io(
                    "groupframe returned no data".to_string(),
                ));
            }
        };
        // Parse the size of the data
        // Note the initial space in the string below
        let output_line = java_lang_string_trim(&std_output[0]);
        let tokens = java_lang_string_split(output_line, &WHITESPACE);
        if tokens.len() < input.len() {
            utilities::timestamp_process_container_status(
                Some("run"),
                Some("goodframe"),
                Some(utilities::FAILED_STATUS),
            );
            return Err(GoodframeError::Io(format!(
                "groupframe returned less than {} outputs",
                input.len()
            )));
        }
        let mut output: Vec<EtomoNumber> = Vec::with_capacity(input.len());
        for i in 0..input.len() {
            let mut number = EtomoNumber::new();
            number.set_string(Some(&tokens[i]));
            let invalid = !number.is_valid() || number.is_null();
            let invalid_reason = number.get_invalid_reason();
            output.push(number);
            if invalid {
                // Java has already stored the partly filled `output` array.
                self.output = Some(output);
                utilities::timestamp_process_container_status(
                    Some("run"),
                    Some("goodframe"),
                    Some(utilities::FAILED_STATUS),
                );
                return Err(GoodframeError::NumberFormat(format!(
                    "Output {} is not set, token is {}\n{}",
                    i, tokens[i], invalid_reason
                )));
            }
        }
        self.output = Some(output);
        utilities::timestamp_process_container_status(
            Some("run"),
            Some("goodframe"),
            Some(utilities::FINISHED_STATUS),
        );
        Ok(())
    }

    /// Java `getOutput(int)`.  Returns `output[index]` if it exists, otherwise null.
    ///
    /// Every caller dereferences the result (`TiltParam`, `BlendmontParam`), so a
    /// missing output is a NullPointerException there.  Fixed in translation: a missing
    /// output is returned as a null number (`getInt` then answers
    /// `INTEGER_NULL_VALUE`); [`Goodframe::get_output_nullable`] keeps the source's
    /// null.
    pub fn get_output(&self, index: i32) -> ConstEtomoNumber {
        match self.get_output_nullable(index) {
            Some(output) => output,
            None => EtomoNumber::new().base,
        }
    }

    /// Java `getOutput(int)` with the source's null result.
    pub fn get_output_nullable(&self, index: i32) -> Option<ConstEtomoNumber> {
        let output = self.output.as_ref()?;
        if index < 0 || (output.len() as i64) < index as i64 + 1 {
            return None;
        }
        Some(output[index as usize].base.clone())
    }
}
