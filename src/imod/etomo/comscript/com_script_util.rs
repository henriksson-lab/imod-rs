//! `IMOD/Etomo/src/etomo/comscript/ComScriptUtil.java`.
//!
//! Description:  Utility class for comscript managers.  Replaces BaseComScriptManager.
//!
//! Copyright: Copyright 2016 - 2024 by the Regents of the University of Colorado
//!
//! Organization: Dept. of MCD Biology, University of Colorado
//!
//! **Dialogs.**  These routines run on the event dispatch thread (they are called from
//! dialog actions through `ComScriptManager`).  `UIHarness.INSTANCE.openMessageDialog(
//! manager, String[], title, axisID)` and the bare `JOptionPane.showMessageDialog(null,
//! message, title, ERROR_MESSAGE)` are both shown synchronously through
//! `ui_harness::open_message_dialog_from_process`, which runs the harness on the
//! current thread (installed presentation, or the headless log).  A `String[]` message
//! is shown one element per line, as `JOptionPane` lays out an array; the
//! `JOptionPane` error icon is not reproduced (messages need not mirror exactly).
//!
//! **Exceptions.**  `except.printStackTrace()` prints the exception's description to
//! standard error; a Java stack trace is a property of the JVM, not of the program.
//!
//! **Parameters.**  A Java `ComScript` argument that may be null is
//! `Option<&mut ComScript>`; a `CommandParam` argument is `&mut dyn CommandParam` where
//! it is parsed into and `&dyn CommandParam` where it only updates a command.

use super::bad_com_script_exception::BadComScriptException;
use super::com_script::ComScript;
use super::com_script_command::ComScriptCommand;
use super::command_param::{CommandParam, ParseComScriptError};
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::storage::log_file::LogFileError;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::file_type::FileType;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::ui::swing::ui_harness;
use crate::imod::etomo::util::utilities::{
    self, FAILED_STATUS, FINISHED_STATUS, NOTHING_TO_DO_STATUS, STARTED_STATUS,
    java_io_file_get_absolute_path, java_io_file_get_name, java_io_file_new,
};
use std::cell::RefCell;
use std::io::Write;
use std::rc::Rc;

/// The checked exceptions the two `useTemplate` overloads declare:
/// `BadComScriptException`, `IOException`, and (`useTemplate(BaseManager, String,
/// AxisType, AxisID, boolean)` only) `LogFileException`/`LockException`, which reach
/// the translation as `LogFileError`.
#[derive(Debug)]
pub enum UseTemplateError {
    BadComScript(BadComScriptException),
    Io(std::io::Error),
    LogFile(LogFileError),
}

impl std::fmt::Display for UseTemplateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UseTemplateError::BadComScript(e) => write!(f, "{e}"),
            UseTemplateError::Io(e) => write!(f, "{e}"),
            UseTemplateError::LogFile(e) => write!(f, "{e}"),
        }
    }
}

impl From<BadComScriptException> for UseTemplateError {
    fn from(e: BadComScriptException) -> UseTemplateError {
        UseTemplateError::BadComScript(e)
    }
}

impl From<std::io::Error> for UseTemplateError {
    fn from(e: std::io::Error) -> UseTemplateError {
        UseTemplateError::Io(e)
    }
}

impl From<LogFileError> for UseTemplateError {
    fn from(e: LogFileError) -> UseTemplateError {
        UseTemplateError::LogFile(e)
    }
}

/// Java `ComScriptUtil` (a final class of static methods).
pub struct ComScriptUtil;

impl ComScriptUtil {
    /// Java package-private `loadComScript(BaseManager, FileType, AxisID, boolean,
    /// boolean, boolean)`.
    pub fn load_com_script_file_type(
        manager: &'static dyn BaseManager,
        script_file_type: &FileType,
        axis_id: AxisID,
        parse_comments: bool,
        case_insensitive: bool,
        separate_with_a_space: bool,
    ) -> Option<ComScript> {
        let file_name = script_file_type.get_file_name(Some(manager), Some(axis_id));
        ComScriptUtil::load_com_script_file_name(
            manager,
            file_name.as_deref(),
            axis_id,
            parse_comments,
            true,
            case_insensitive,
            separate_with_a_space,
        )
    }

    /// Java `loadComScript(BaseManager, String, AxisID, boolean, boolean, boolean,
    /// boolean)`.
    pub fn load_com_script_file_name(
        manager: &'static dyn BaseManager,
        command_com_script_file_name: Option<&str>,
        axis_id: AxisID,
        parse_comments: bool,
        required: bool,
        case_insensitive: bool,
        separate_with_a_space: bool,
    ) -> Option<ComScript> {
        ComScriptUtil::load_com_script(
            manager,
            None,
            command_com_script_file_name,
            axis_id,
            parse_comments,
            required,
            case_insensitive,
            separate_with_a_space,
        )
    }

    /// Java `loadComScript(BaseManager, String, String, AxisID, boolean, boolean,
    /// boolean, boolean)`.  Load the comscript file in commandComScript.
    /// `comScriptAltDir` - optional - absolute path of the directory containing
    /// commandComScriptFileName.
    #[allow(clippy::too_many_arguments)]
    pub fn load_com_script(
        manager: &'static dyn BaseManager,
        com_script_alt_dir: Option<&str>,
        command_com_script_file_name: Option<&str>,
        axis_id: AxisID,
        parse_comments: bool,
        required: bool,
        case_insensitive: bool,
        separate_with_a_space: bool,
    ) -> Option<ComScript> {
        utilities::timestamp_process_container_status(
            Some("load"),
            command_com_script_file_name,
            Some(STARTED_STATUS),
        );
        // `new File(String parent, String child)`: a null parent is `new File(child)`;
        // a null child throws `NullPointerException`, which no caller can reach (every
        // file name here is built from a non-null name).
        let child = command_com_script_file_name.unwrap_or("null");
        let com_file = match com_script_alt_dir {
            None => match manager.get_property_user_dir() {
                None => child.to_string(),
                Some(parent) => java_io_file_new(&parent, child),
            },
            Some(com_script_alt_dir) => java_io_file_new(com_script_alt_dir, child),
        };
        // If the file isn't there and its not required then just return null without
        // any error messages.
        if !required && !std::path::Path::new(&com_file).exists() {
            utilities::timestamp_process_container_status(
                Some("load"),
                command_com_script_file_name,
                Some(NOTHING_TO_DO_STATUS),
            );
            return None;
        }
        let mut com_script = ComScript::new(&com_file);
        com_script.set_parse_comments(parse_comments);
        if let Err(except) = com_script.read_com_file(case_insensitive, separate_with_a_space) {
            eprintln!("{}", except);
            let error_message = [
                "Com file: ".to_string() + &com_script.get_com_file_name(),
                except.to_string(),
            ];
            ui_harness::open_message_dialog_from_process(
                None,
                &error_message.join("\n"),
                &format!(
                    "Can't parse {}{}.com file: {}",
                    child,
                    axis_id.get_extension(),
                    com_script.get_com_file_name()
                ),
                None,
            );
            utilities::timestamp_process_container_status(
                Some("load"),
                command_com_script_file_name,
                Some(FAILED_STATUS),
            );
            return None;
        }
        utilities::timestamp_process_container_status(
            Some("load"),
            command_com_script_file_name,
            Some(FINISHED_STATUS),
        );
        Some(com_script)
    }

    /// Java `initialize(BaseManager, CommandParam, ComScript, String, AxisID, boolean,
    /// boolean, boolean)`.  Initialize the CommandParam object from the specified
    /// command in the comscript.  True is returned if the initialization is successful,
    /// false if the initialization fails.
    #[allow(clippy::too_many_arguments)]
    pub fn initialize(
        manager: &'static dyn BaseManager,
        param: &mut dyn CommandParam,
        com_script: Option<&mut ComScript>,
        command: &str,
        axis_id: AxisID,
        case_insensitive: bool,
        separate_with_a_space: bool,
        required: bool,
    ) -> bool {
        ComScriptUtil::initialize_optional_command(
            manager,
            param,
            com_script,
            command,
            axis_id,
            false,
            case_insensitive,
            separate_with_a_space,
            required,
            None,
        )
    }

    /// Java `initialize(BaseManager, CommandParam, ComScript, String, AxisID, boolean,
    /// boolean, boolean, String)`.
    #[allow(clippy::too_many_arguments)]
    pub fn initialize_required_option(
        manager: &'static dyn BaseManager,
        param: &mut dyn CommandParam,
        com_script: Option<&mut ComScript>,
        command: &str,
        axis_id: AxisID,
        case_insensitive: bool,
        separate_with_a_space: bool,
        required: bool,
        required_option: Option<&str>,
    ) -> bool {
        ComScriptUtil::initialize_optional_command(
            manager,
            param,
            com_script,
            command,
            axis_id,
            false,
            case_insensitive,
            separate_with_a_space,
            required,
            required_option,
        )
    }

    /// Java package-private `initialize(BaseManager, CommandParam, ComScript, String,
    /// AxisID, boolean optionalCommand, boolean, boolean, boolean, String)`.
    #[allow(clippy::too_many_arguments)]
    pub fn initialize_optional_command(
        manager: &'static dyn BaseManager,
        param: &mut dyn CommandParam,
        com_script: Option<&mut ComScript>,
        command: &str,
        axis_id: AxisID,
        optional_command: bool,
        case_insensitive: bool,
        separate_with_a_space: bool,
        required: bool,
        required_option: Option<&str>,
    ) -> bool {
        let com_script = match com_script {
            None => return false,
            Some(com_script) => com_script,
        };
        utilities::timestamp_com_script(
            Some("initialize"),
            Some(command),
            Some(&*com_script),
            Some(STARTED_STATUS),
        );
        if !com_script.is_command_loaded() {
            param.initialize_defaults();
        } else {
            let index = com_script.get_script_command_index_required_option(
                command,
                case_insensitive,
                separate_with_a_space,
                required_option,
            );
            if optional_command && index == -1 {
                utilities::timestamp_com_script(
                    Some("initialize"),
                    Some(command),
                    Some(&*com_script),
                    Some(FAILED_STATUS),
                );
                return false;
            }
            let result: Result<(), ParseComScriptError> = match com_script.get_script_command_named(
                command,
                case_insensitive,
                separate_with_a_space,
            ) {
                Err(except) => Err(except.into()),
                Ok(script_command) => {
                    let parsed = param.parse_com_script_command(&script_command.borrow());
                    parsed
                }
            };
            if let Err(except) = result {
                if required {
                    eprintln!("{}", except);
                    let error_message = [
                        "Com file: ".to_string() + &com_script.get_com_file_name(),
                        "Command: ".to_string() + command,
                        match &except {
                            ParseComScriptError::BadComScript(_) => {
                                "etomo.comscript.BadComScriptException".to_string()
                            }
                            ParseComScriptError::FortranInputSyntax(_) => {
                                "etomo.comscript.FortranInputSyntaxException".to_string()
                            }
                            ParseComScriptError::InvalidParameter(_) => {
                                "etomo.comscript.InvalidParameterException".to_string()
                            }
                            ParseComScriptError::NumberFormat(_) => {
                                "java.lang.NumberFormatException".to_string()
                            }
                        },
                        match &except {
                            ParseComScriptError::BadComScript(e) => e.get_message().to_string(),
                            ParseComScriptError::FortranInputSyntax(e) => {
                                e.get_message().unwrap_or("null").to_string()
                            }
                            ParseComScriptError::InvalidParameter(e) => {
                                e.get_message().unwrap_or("null").to_string()
                            }
                            ParseComScriptError::NumberFormat(e) => e.clone(),
                        },
                    ];
                    ui_harness::open_message_dialog_from_process(
                        None,
                        &error_message.join("\n"),
                        "Com Script Command Parse Error",
                        None,
                    );
                    utilities::timestamp_com_script(
                        Some("initialize"),
                        Some(command),
                        Some(&*com_script),
                        Some(FAILED_STATUS),
                    );
                }
                return false;
            }
        }
        utilities::timestamp_com_script(
            Some("initialize"),
            Some(command),
            Some(&*com_script),
            Some(FINISHED_STATUS),
        );
        true
    }

    /// Java `modifyCommand(BaseManager, ComScript, CommandParam, String, AxisID,
    /// boolean, boolean)`.  Update the specified comscript.
    pub fn modify_command(
        manager: &'static dyn BaseManager,
        script: Option<&mut ComScript>,
        params: &dyn CommandParam,
        command: &str,
        axis_id: AxisID,
        case_insensitive: bool,
        separate_with_a_space: bool,
    ) {
        ComScriptUtil::modify_command_add_new(
            manager,
            script,
            params,
            command,
            axis_id,
            false,
            false,
            case_insensitive,
            separate_with_a_space,
            None,
        );
    }

    /// Java `modifyCommand(BaseManager, ComScript, CommandParam, String, AxisID,
    /// boolean, boolean, String)`.  Update the specified comscript.
    #[allow(clippy::too_many_arguments)]
    pub fn modify_command_required_option(
        manager: &'static dyn BaseManager,
        script: Option<&mut ComScript>,
        params: &dyn CommandParam,
        command: &str,
        axis_id: AxisID,
        case_insensitive: bool,
        separate_with_a_space: bool,
        required_option: Option<&str>,
    ) {
        ComScriptUtil::modify_command_add_new(
            manager,
            script,
            params,
            command,
            axis_id,
            false,
            false,
            case_insensitive,
            separate_with_a_space,
            required_option,
        );
    }

    /// Java package-private `modifyCommand(BaseManager, ComScript, CommandParam,
    /// String, AxisID, boolean addNew, boolean optional, boolean, boolean, String)`.
    /// Modify and/or add command (depending on addNew boolean).  May treate command as
    /// optional, depending on optional boolean.
    ///
    /// Fixed in translation: with a `requiredOption`, `getScriptCommandIndex` can return
    /// -1 while `getScriptCommand(String, ...)` (which ignores the option) finds the
    /// command, and `setScriptComand(-1, ...)` then throws an uncaught
    /// `IndexOutOfBoundsException`.  That case fails here with the `update` failed
    /// timestamp and returns false, writing nothing (`BUGS.md`).
    #[allow(clippy::too_many_arguments)]
    pub fn modify_command_add_new(
        manager: &'static dyn BaseManager,
        script: Option<&mut ComScript>,
        params: &dyn CommandParam,
        command: &str,
        axis_id: AxisID,
        add_new: bool,
        optional: bool,
        case_insensitive: bool,
        separate_with_a_space: bool,
        required_option: Option<&str>,
    ) -> bool {
        let script = match script {
            None => {
                eprintln!("java.lang.IllegalStateException");
                let error_message = [
                    "Unable to update comscript.".to_string(),
                    "\nscript=null\ncommand=".to_string() + command,
                ];
                ui_harness::open_message_dialog_from_process(
                    Some(manager),
                    &error_message.join("\n"),
                    "ComScriptManager Error",
                    Some(axis_id),
                );
                return false;
            }
            Some(script) => script,
        };
        utilities::timestamp_com_script(
            Some("update"),
            Some(command),
            Some(&*script),
            Some(STARTED_STATUS),
        );

        // Update the specified com script command from the CommandParam object
        let command_index = script.get_script_command_index_add_new(
            command,
            add_new,
            case_insensitive,
            separate_with_a_space,
            required_option,
        );
        // optional return false if failed
        if optional && command_index == -1 {
            utilities::timestamp_com_script(
                Some("updateComScript"),
                Some(command),
                Some(&*script),
                Some(FAILED_STATUS),
            );
            return false;
        }
        let result: Result<Rc<RefCell<ComScriptCommand>>, BadComScriptException> = match script
            .get_script_command_named(command, case_insensitive, separate_with_a_space)
        {
            Err(except) => Err(except),
            Ok(com_script_command) => {
                let updated =
                    params.update_com_script_command(&mut com_script_command.borrow_mut());
                match updated {
                    Err(except) => Err(except),
                    Ok(()) => Ok(com_script_command),
                }
            }
        };
        let com_script_command = match result {
            Err(except) => {
                eprintln!("{}", except);
                let error_message = [
                    "Com file: ".to_string() + &script.get_com_file_name(),
                    "Command: ".to_string() + command,
                    except.get_message().to_string(),
                ];
                ui_harness::open_message_dialog_from_process(
                    None,
                    &error_message.join("\n"),
                    &format!("Can't update {} in {}", command, script.get_com_file_name()),
                    None,
                );
                utilities::timestamp_com_script(
                    Some("update"),
                    Some(command),
                    Some(&*script),
                    Some(FAILED_STATUS),
                );
                return false;
            }
            Ok(com_script_command) => com_script_command,
        };

        // See the doc comment: `ArrayList.set(-1, ...)` throws.
        if command_index == -1 {
            utilities::timestamp_com_script(
                Some("update"),
                Some(command),
                Some(&*script),
                Some(FAILED_STATUS),
            );
            return false;
        }
        // Replace the specified command by the updated comScriptCommand
        script.set_script_comand(command_index, &com_script_command.borrow());

        // Write the script back out to disk
        if let Err(except) = script.write_com_file() {
            eprintln!("{}", except);
            let _error_message = [
                "Com file: ".to_string() + &script.get_com_file_name(),
                "Command: ".to_string() + command,
                except.to_string(),
            ];
            ui_harness::open_message_dialog_from_process(
                None,
                &except.to_string(),
                &format!("Can't write {}{}.com", command, axis_id.get_extension()),
                None,
            );
            utilities::timestamp_com_script(
                Some("update"),
                Some(command),
                Some(&*script),
                Some(FAILED_STATUS),
            );
            return false;
        }
        utilities::timestamp_com_script(
            Some("update"),
            Some(command),
            Some(&*script),
            Some(FINISHED_STATUS),
        );
        true
    }

    /// Java package-private `useTemplate(BaseManager, String, String, AxisType,
    /// AxisID)`.  Use a template script from the $IMOD_DIR/com directory.  This is
    /// useful if we encounter an old data set that does not have a current complete set
    /// of com scripts.  `axisID` is ignored for single axis tomogram.  For a dual axis
    /// tomogram AxisID.FIRST will create an a.com script, AxisID.SECOND will create a
    /// b.com script.  AxisID.ONLY will create a .com script replacing the g5a and g5b
    /// tags.
    ///
    /// `String.replaceAll` treats `$` and `\` in the replacement specially; a dataset
    /// name holds neither, so the replacement is literal here.
    ///
    /// Fixed in translation: `EtomoDirector.INSTANCE.getIMODDirectory()` is null when
    /// `IMOD_DIR` is not set, and `.getAbsolutePath()` then throws
    /// `NullPointerException`; that is returned as an `IOException` here.
    pub fn use_template_dataset_name(
        manager: &'static dyn BaseManager,
        script_name: &str,
        dataset_name: &str,
        axis_type: AxisType,
        axis_id: AxisID,
    ) -> Result<(), UseTemplateError> {
        let mut dataset_name = dataset_name.to_string();
        utilities::timestamp_process_container_status(
            Some("copy from template"),
            Some(script_name),
            Some(STARTED_STATUS),
        );
        // Read in the template file from the IMOD_DIR/com directory replacing all
        // instances of the tag g5a and g5b with the appropriate dataset name
        let imod_directory = etomo_director::INSTANCE
            .get_imod_directory()
            .map(|directory| directory.to_string_lossy().to_string());
        let imod_directory = match imod_directory {
            None => {
                return Err(std::io::Error::other("java.lang.NullPointerException").into());
            }
            Some(imod_directory) => imod_directory,
        };
        let com_directory =
            java_io_file_get_absolute_path(&imod_directory) + std::path::MAIN_SEPARATOR_STR + "com";

        let template = java_io_file_new(&com_directory, &(script_name.to_string() + ".com"));
        if !std::path::Path::new(&template).exists() {
            let message = "Unknown template: ".to_string() + script_name;
            utilities::timestamp_process_container_status(
                Some("copy from template"),
                Some(script_name),
                Some(FAILED_STATUS),
            );
            return Err(BadComScriptException::new(&message).into());
        }
        let template_bytes = std::fs::read(&template)?;
        let template_text = String::from_utf8_lossy(&template_bytes).into_owned();

        // The ouput script
        let property_user_dir = manager.get_property_user_dir();

        // Open the appropriate output script and change the dataset name if
        // necessary.
        let child = if axis_type == AxisType::SingleAxis {
            script_name.to_string() + ".com"
        } else {
            dataset_name = dataset_name + &axis_id.get_extension();
            script_name.to_string() + &axis_id.get_extension() + ".com"
        };
        let script = match &property_user_dir {
            // `new File((String) null, child)` is `new File(child)`.
            None => child,
            Some(parent) => java_io_file_new(parent, &child),
        };
        let mut script_writer = std::io::BufWriter::new(std::fs::File::create(&script)?);

        // `BufferedReader.readLine`: a line ends at `\n`, `\r` or `\r\n`.
        let mut lines: Vec<String> = Vec::new();
        {
            let mut current = String::new();
            let mut chars = template_text.chars().peekable();
            let mut pending = false;
            while let Some(c) = chars.next() {
                if c == '\n' || c == '\r' {
                    lines.push(std::mem::take(&mut current));
                    pending = false;
                    if c == '\r' && chars.peek() == Some(&'\n') {
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

        if axis_type == AxisType::DualAxis && axis_id == AxisID::Only {
            for line in lines.iter() {
                let line = line.replace("g5a", &(dataset_name.clone() + "a"));
                let line = line.replace("g5b", &(dataset_name.clone() + "b"));
                script_writer.write_all(line.as_bytes())?;
                script_writer.write_all(b"\n")?;
            }
        } else {
            for line in lines.iter() {
                let line = line.replace("g5a", &dataset_name);
                script_writer.write_all(line.as_bytes())?;
                script_writer.write_all(b"\n")?;
            }
        }
        script_writer.flush()?;
        utilities::timestamp_process_container_status(
            Some("copy from template"),
            Some(script_name),
            Some(FINISHED_STATUS),
        );
        Ok(())
    }

    /// Java `useTemplate(BaseManager, String, AxisType, AxisID, boolean)`.  Use a
    /// template script from the $IMOD_DIR/com directory.  This is useful if we
    /// encounter an old data set that does not have a current complete set of com
    /// scripts.  `rename` - rename the original script file.
    ///
    /// Fixed in translation: a null `getIMODDirectory()` (see
    /// `use_template_dataset_name`) is returned as an `IOException`.
    pub fn use_template(
        manager: &'static dyn BaseManager,
        script_name: &str,
        axis_type: AxisType,
        axis_id: AxisID,
        rename: bool,
    ) -> Result<(), UseTemplateError> {
        utilities::timestamp_process_container_status(
            Some("copy from template"),
            Some(script_name),
            Some(STARTED_STATUS),
        );
        // Copy the template file from the IMOD_DIR/com directory to the script
        let imod_directory = etomo_director::INSTANCE
            .get_imod_directory()
            .map(|directory| directory.to_string_lossy().to_string());
        let imod_directory = match imod_directory {
            None => {
                return Err(std::io::Error::other("java.lang.NullPointerException").into());
            }
            Some(imod_directory) => imod_directory,
        };
        let com_directory =
            java_io_file_get_absolute_path(&imod_directory) + std::path::MAIN_SEPARATOR_STR + "com";

        let template = java_io_file_new(&com_directory, &(script_name.to_string() + ".com"));
        if !std::path::Path::new(&template).exists() {
            let message = "Unknown template: ".to_string() + script_name;
            utilities::timestamp_process_container_status(
                Some("copy from template"),
                Some(script_name),
                Some(FAILED_STATUS),
            );
            return Err(BadComScriptException::new(&message).into());
        }

        // The ouput script
        let property_user_dir = manager.get_property_user_dir();
        let child = if axis_type == AxisType::SingleAxis {
            script_name.to_string() + ".com"
        } else {
            script_name.to_string() + &axis_id.get_extension() + ".com"
        };
        let script = match &property_user_dir {
            // `new File((String) null, child)` is `new File(child)`.
            None => child,
            Some(parent) => java_io_file_new(parent, &child),
        };

        if rename {
            if let Err(_e) = utilities::rename_file(
                Some(manager),
                Some(axis_id),
                Some(std::path::Path::new(&script)),
                Some(std::path::Path::new(
                    &(java_io_file_get_absolute_path(&script) + "~"),
                )),
                false,
                false,
                false,
            ) {
                eprintln!("Unable to backup {}.", java_io_file_get_name(&script));
            }
        }
        utilities::copy_file(
            Some(manager),
            Some(axis_id),
            Some(std::path::Path::new(&template)),
            Some(std::path::Path::new(&script)),
            false,
            false,
            false,
        )?;
        utilities::timestamp_process_container_status(
            Some("copy from template"),
            Some(script_name),
            Some(FINISHED_STATUS),
        );
        Ok(())
    }

    /// Java package-private `loadComScript(BaseManager, ProcessName, AxisID, boolean,
    /// boolean, boolean)`.
    pub fn load_com_script_process_name(
        manager: &'static dyn BaseManager,
        process_name: ProcessName,
        axis_id: AxisID,
        parse_comments: bool,
        case_insensitive: bool,
        separate_with_a_space: bool,
    ) -> Option<ComScript> {
        ComScriptUtil::load_com_script_file_name(
            manager,
            Some(&process_name.get_comscript(axis_id)),
            axis_id,
            parse_comments,
            true,
            case_insensitive,
            separate_with_a_space,
        )
    }

    /// Java package-private `loadComScript(BaseManager, String, AxisID, boolean,
    /// boolean, boolean)`.
    ///
    /// Fixed in translation: `ProcessName.getInstance(scriptName, axisID)` is null for a
    /// name that is not a process name, and `.getComscript` then throws
    /// `NullPointerException`.  Every caller passes a built-in process name; for any
    /// other name nothing is loaded and `None` is returned (`BUGS.md`).
    pub fn load_com_script_script_name(
        manager: &'static dyn BaseManager,
        script_name: &str,
        axis_id: AxisID,
        parse_comments: bool,
        case_insensitive: bool,
        separate_with_a_space: bool,
    ) -> Option<ComScript> {
        let process_name = ProcessName::get_instance_with_axis(script_name, axis_id)?;
        ComScriptUtil::load_com_script_file_name(
            manager,
            Some(&process_name.get_comscript(axis_id)),
            axis_id,
            parse_comments,
            true,
            case_insensitive,
            separate_with_a_space,
        )
    }

    /// Java package-private `modifyOptionalCommand`.
    pub fn modify_optional_command(
        manager: &'static dyn BaseManager,
        script: Option<&mut ComScript>,
        params: &dyn CommandParam,
        command: &str,
        axis_id: AxisID,
        case_insensitive: bool,
        separate_with_a_space: bool,
    ) -> bool {
        ComScriptUtil::modify_command_add_new(
            manager,
            script,
            params,
            command,
            axis_id,
            false,
            true,
            case_insensitive,
            separate_with_a_space,
            None,
        )
    }

    /// Java package-private `modifyCommand(BaseManager, ComScript, CommandParam,
    /// String, AxisID, boolean addNew, boolean optional, String previousCommand,
    /// boolean, boolean)`.
    ///
    /// Fixed in translation: with neither `addNew` nor `optional`, a missing command
    /// leaves `commandIndex` at -1 and `script.getScriptCommand(-1)` throws an uncaught
    /// `IndexOutOfBoundsException`.  That case takes the method's own "Can't update"
    /// path with `BadComScriptException("Did not find command: ...")` (`BUGS.md`).
    #[allow(clippy::too_many_arguments)]
    pub fn modify_command_previous_command(
        manager: &'static dyn BaseManager,
        script: Option<&mut ComScript>,
        params: &dyn CommandParam,
        command: &str,
        axis_id: AxisID,
        add_new: bool,
        optional: bool,
        previous_command: &str,
        case_insensitive: bool,
        separate_with_a_space: bool,
    ) -> bool {
        let script = match script {
            None => {
                eprintln!("java.lang.IllegalStateException");
                let error_message = [
                    "Unable to update comscript.".to_string(),
                    "\nscript=null\ncommand=".to_string() + command,
                ];
                ui_harness::open_message_dialog_from_process(
                    Some(manager),
                    &error_message.join("\n"),
                    "ComScriptManager Error",
                    Some(axis_id),
                );
                return false;
            }
            Some(script) => script,
        };
        utilities::timestamp_com_script(
            Some("update"),
            Some(command),
            Some(&*script),
            Some(STARTED_STATUS),
        );

        // locate previous command
        let previous_command_index = script.get_script_command_index(
            previous_command,
            case_insensitive,
            separate_with_a_space,
        );

        // previous command must exist
        if previous_command_index == -1 {
            let error_message = [
                "Unable to update ".to_string() + &script.get_name() + ".  ",
                format!(
                    "Unable to update {} because the previous command, {}, is missing.",
                    command, previous_command
                ),
            ];
            ui_harness::open_message_dialog_from_process(
                Some(manager),
                &error_message.join("\n"),
                "ComScriptManager Error",
                Some(axis_id),
            );
            utilities::timestamp_com_script(
                Some("update"),
                Some(command),
                Some(&*script),
                Some(FAILED_STATUS),
            );
            return false;
        }
        // Update the specified com script command from the CommandParam object
        let command_index = script.get_script_command_index_at_add_new(
            command,
            previous_command_index + 1,
            add_new,
            case_insensitive,
            separate_with_a_space,
        );
        // optional return false if failed
        if optional && command_index == -1 {
            utilities::timestamp_com_script(
                Some("updateComScript"),
                Some(command),
                Some(&*script),
                Some(FAILED_STATUS),
            );
            return false;
        }
        let result: Result<Rc<RefCell<ComScriptCommand>>, BadComScriptException> =
            match script.get_script_command(command_index) {
                None => Err(BadComScriptException::new(&format!(
                    "Did not find command: {} at index {}",
                    command, command_index
                ))),
                Some(com_script_command) => {
                    let updated =
                        params.update_com_script_command(&mut com_script_command.borrow_mut());
                    match updated {
                        Err(except) => Err(except),
                        Ok(()) => Ok(com_script_command),
                    }
                }
            };
        let com_script_command = match result {
            Err(except) => {
                eprintln!("{}", except);
                let error_message = [
                    "Com file: ".to_string() + &script.get_com_file_name(),
                    "Command: ".to_string() + command,
                    except.get_message().to_string(),
                ];
                ui_harness::open_message_dialog_from_process(
                    None,
                    &error_message.join("\n"),
                    &format!("Can't update {} in {}", command, script.get_com_file_name()),
                    None,
                );
                utilities::timestamp_com_script(
                    Some("update"),
                    Some(command),
                    Some(&*script),
                    Some(FAILED_STATUS),
                );
                return false;
            }
            Ok(com_script_command) => com_script_command,
        };

        // Replace the specified command by the updated comScriptCommand
        script.set_script_comand(command_index, &com_script_command.borrow());

        // Write the script back out to disk
        if let Err(except) = script.write_com_file() {
            eprintln!("{}", except);
            let _error_message = [
                "Com file: ".to_string() + &script.get_com_file_name(),
                "Command: ".to_string() + command,
                except.to_string(),
            ];
            ui_harness::open_message_dialog_from_process(
                None,
                &except.to_string(),
                &format!("Can't write {}{}.com", command, axis_id.get_extension()),
                None,
            );
            utilities::timestamp_com_script(
                Some("update"),
                Some(command),
                Some(&*script),
                Some(FAILED_STATUS),
            );
            return false;
        }
        utilities::timestamp_com_script(
            Some("update"),
            Some(command),
            Some(&*script),
            Some(FINISHED_STATUS),
        );
        true
    }

    /// Java package-private `addModifyCommand(BaseManager, ComScript fromScript,
    /// ComScript toScript, CommandParam, String, AxisID, boolean, boolean)`.  Find the
    /// command in fromScript, update it, and write it to toScript.  AddNew is only used
    /// with the toScript.  The fromScript should not be changed.  If the fromScript is
    /// not null, the command must be in it.
    #[allow(clippy::too_many_arguments)]
    pub fn add_modify_command_from_to(
        manager: &'static dyn BaseManager,
        from_script: Option<&mut ComScript>,
        to_script: Option<&mut ComScript>,
        params: &dyn CommandParam,
        command: &str,
        axis_id: AxisID,
        case_insensitive: bool,
        separate_with_a_space: bool,
    ) {
        let to_script = match to_script {
            None => {
                eprintln!("java.lang.IllegalStateException");
                let error_message = [
                    "Unable to update comscript.  ".to_string(),
                    "\ntoScript=nullncommand=".to_string() + command,
                ];
                ui_harness::open_message_dialog_from_process(
                    Some(manager),
                    &error_message.join("\n"),
                    "ComScriptManager Error",
                    Some(axis_id),
                );
                return;
            }
            Some(to_script) => to_script,
        };
        utilities::timestamp_com_script(
            Some("update"),
            Some(command),
            Some(&*to_script),
            Some(STARTED_STATUS),
        );
        // If no fromScript, then default to reading from and writing to the
        // toScript.
        let from_script = match from_script {
            None => {
                ComScriptUtil::modify_command_add_new(
                    manager,
                    Some(&mut *to_script),
                    params,
                    command,
                    axis_id,
                    true,
                    false,
                    case_insensitive,
                    separate_with_a_space,
                    None,
                );
                utilities::timestamp_com_script(
                    Some("update"),
                    Some(command),
                    Some(&*to_script),
                    Some(FINISHED_STATUS),
                );
                return;
            }
            Some(from_script) => from_script,
        };
        // Update the specified com script command from the CommandParam object
        let to_script_command_index = to_script.get_script_command_index_add_new(
            command,
            true,
            case_insensitive,
            separate_with_a_space,
            None,
        );
        // Get comScriptCommand from fromScript
        let com_script_command = match from_script.get_script_command_named(
            command,
            case_insensitive,
            separate_with_a_space,
        ) {
            Err(except) => {
                eprintln!("{}", except);
                let error_message = [
                    "Com file: ".to_string() + &from_script.get_com_file_name(),
                    "Command: ".to_string() + command,
                    except.get_message().to_string(),
                ];
                ui_harness::open_message_dialog_from_process(
                    None,
                    &error_message.join("\n"),
                    &format!(
                        "Can't read {} in {}",
                        command,
                        from_script.get_com_file_name()
                    ),
                    None,
                );
                utilities::timestamp_com_script(
                    Some("update"),
                    Some(command),
                    Some(&*to_script),
                    Some(FAILED_STATUS),
                );
                return;
            }
            Ok(com_script_command) => com_script_command,
        };
        // Update comScriptCommand
        if let Err(except) = params.update_com_script_command(&mut com_script_command.borrow_mut())
        {
            eprintln!("{}", except);
            let error_message = [
                "Com file: ".to_string() + &to_script.get_com_file_name(),
                "Command: ".to_string() + command,
                except.get_message().to_string(),
            ];
            ui_harness::open_message_dialog_from_process(
                None,
                &error_message.join("\n"),
                &format!(
                    "Can't update {} in {}",
                    command,
                    to_script.get_com_file_name()
                ),
                None,
            );
            utilities::timestamp_com_script(
                Some("update"),
                Some(command),
                Some(&*to_script),
                Some(FAILED_STATUS),
            );
            return;
        }

        // Replace the specified command by the updated comScriptCommand in toScript
        to_script.set_script_comand(to_script_command_index, &com_script_command.borrow());

        // Write toScript back out to disk
        if let Err(except) = to_script.write_com_file() {
            eprintln!("{}", except);
            let _error_message = [
                "Com file: ".to_string() + &to_script.get_com_file_name(),
                "Command: ".to_string() + command,
                except.to_string(),
            ];
            ui_harness::open_message_dialog_from_process(
                None,
                &except.to_string(),
                &format!("Can't write {}{}.com", command, axis_id.get_extension()),
                None,
            );
            utilities::timestamp_com_script(
                Some("update"),
                Some(command),
                Some(&*to_script),
                Some(FAILED_STATUS),
            );
            return;
        }
        utilities::timestamp_com_script(
            Some("update"),
            Some(command),
            Some(&*to_script),
            Some(FINISHED_STATUS),
        );
    }

    /// Java package-private `addModifyCommand(BaseManager, ComScript, CommandParam,
    /// String, AxisID, String previousCommand, boolean, boolean)`.  Modify and/or add
    /// command; returns the index of the updated command.
    #[allow(clippy::too_many_arguments)]
    pub fn add_modify_command_previous_command(
        manager: &'static dyn BaseManager,
        script: Option<&mut ComScript>,
        params: &dyn CommandParam,
        command: &str,
        axis_id: AxisID,
        previous_command: &str,
        case_insensitive: bool,
        separate_with_a_space: bool,
    ) -> i32 {
        let script = match script {
            None => {
                let error_message =
                    ["Unable to update comscript.\nscript=null\ncommand=".to_string() + command];
                ui_harness::open_message_dialog_from_process(
                    Some(manager),
                    &error_message.join("\n"),
                    "ComScriptManager Error",
                    Some(axis_id),
                );
                return -1;
            }
            Some(script) => script,
        };
        utilities::timestamp_com_script(
            Some("update"),
            Some(command),
            Some(&*script),
            Some(STARTED_STATUS),
        );
        // locate previous command
        let previous_command_index = script.get_script_command_index(
            previous_command,
            case_insensitive,
            separate_with_a_space,
        );

        if previous_command_index == -1 {
            let error_message = [
                "Unable to update ".to_string() + &script.get_name() + ".  ",
                format!(
                    "Unable to update {} because the previous command, {}, is missing.",
                    command, previous_command
                ),
            ];
            ui_harness::open_message_dialog_from_process(
                Some(manager),
                &error_message.join("\n"),
                "ComScriptManager Error",
                Some(axis_id),
            );
            utilities::timestamp_com_script(
                Some("update"),
                Some(command),
                Some(&*script),
                Some(FAILED_STATUS),
            );
            return -1;
        }

        ComScriptUtil::add_modify_command_previous_index(
            manager,
            Some(script),
            params,
            command,
            axis_id,
            previous_command_index,
            true,
            case_insensitive,
            separate_with_a_space,
        )
    }

    /// Java package-private `addModifyCommand(BaseManager, ComScript, CommandParam,
    /// String, AxisID, boolean, boolean, String requiredOption)`.  Modify and/or add
    /// command.
    #[allow(clippy::too_many_arguments)]
    pub fn add_modify_command_required_option(
        manager: &'static dyn BaseManager,
        script: Option<&mut ComScript>,
        params: &dyn CommandParam,
        command: &str,
        axis_id: AxisID,
        case_insensitive: bool,
        separate_with_a_space: bool,
        required_option: Option<&str>,
    ) {
        let script = match script {
            None => {
                let error_message = [
                    "Unable to update comscript.".to_string(),
                    "\nscript=null\ncommand=".to_string() + command,
                ];
                ui_harness::open_message_dialog_from_process(
                    Some(manager),
                    &error_message.join("\n"),
                    "ComScriptManager Error",
                    Some(axis_id),
                );
                return;
            }
            Some(script) => script,
        };
        utilities::timestamp_com_script(
            Some("update"),
            Some(command),
            Some(&*script),
            Some(STARTED_STATUS),
        );
        let index = script.get_script_command_index_required_option(
            command,
            case_insensitive,
            separate_with_a_space,
            required_option,
        );
        // Update the specified com script command from the CommandParam object
        let result: Result<Rc<RefCell<ComScriptCommand>>, BadComScriptException> = (|| {
            let com_script_command = if index >= 0 {
                script.get_script_command_named_at(
                    command,
                    index,
                    true,
                    case_insensitive,
                    separate_with_a_space,
                )?
            } else {
                // Command is not in the script.
                let com_script_command = Rc::new(RefCell::new(ComScriptCommand::new(
                    case_insensitive,
                    separate_with_a_space,
                )));
                com_script_command.borrow_mut().set_command(Some(command));
                com_script_command
            };
            params.update_com_script_command(&mut com_script_command.borrow_mut())?;
            Ok(com_script_command)
        })();
        let com_script_command = match result {
            Err(except) => {
                eprintln!("{}", except);
                let error_message = [
                    "Com file: ".to_string() + &script.get_com_file_name(),
                    "Command: ".to_string() + command,
                    except.get_message().to_string(),
                ];
                ui_harness::open_message_dialog_from_process(
                    None,
                    &error_message.join("\n"),
                    &format!("Can't update {} in {}", command, script.get_com_file_name()),
                    None,
                );
                utilities::timestamp_com_script(
                    Some("update"),
                    Some(command),
                    Some(&*script),
                    Some(FAILED_STATUS),
                );
                return;
            }
            Ok(com_script_command) => com_script_command,
        };

        // Replace the specified command by the updated comScriptCommand
        if index >= 0 {
            script.set_script_comand(index, &com_script_command.borrow());
        } else {
            // Add the command to the beginning of script.
            script.add_script_comand(0, &com_script_command.borrow());
        }
        // Write the script back out to disk
        if let Err(except) = script.write_com_file() {
            eprintln!("{}", except);
            let _error_message = [
                "Com file: ".to_string() + &script.get_com_file_name(),
                "Command: ".to_string() + command,
                except.to_string(),
            ];
            ui_harness::open_message_dialog_from_process(
                None,
                &except.to_string(),
                &format!("Can't write {}{}.com", command, axis_id.get_extension()),
                None,
            );
            utilities::timestamp_com_script(
                Some("update"),
                Some(command),
                Some(&*script),
                Some(FAILED_STATUS),
            );
            return;
        }
        utilities::timestamp_com_script(
            Some("update"),
            Some(command),
            Some(&*script),
            Some(FINISHED_STATUS),
        );
    }

    /// Java package-private `addModifyCommand(BaseManager, ComScript, CommandParam,
    /// String, AxisID, int previousCommandIndex, boolean updateStarted, boolean,
    /// boolean)`.  Modify and/or add command.
    #[allow(clippy::too_many_arguments)]
    pub fn add_modify_command_previous_index(
        manager: &'static dyn BaseManager,
        script: Option<&mut ComScript>,
        params: &dyn CommandParam,
        command: &str,
        axis_id: AxisID,
        previous_command_index: i32,
        update_started: bool,
        case_insensitive: bool,
        separate_with_a_space: bool,
    ) -> i32 {
        let script = match script {
            None => {
                let error_message = [
                    "Unable to update comscript.".to_string(),
                    "\nscript=null\ncommand=".to_string() + command,
                ];
                ui_harness::open_message_dialog_from_process(
                    Some(manager),
                    &error_message.join("\n"),
                    "ComScriptManager Error",
                    Some(axis_id),
                );
                return -1;
            }
            Some(script) => script,
        };
        if !update_started {
            utilities::timestamp_com_script(
                Some("update"),
                Some(command),
                Some(&*script),
                Some(STARTED_STATUS),
            );
        }
        if previous_command_index < -1 {
            let error_message = [
                "Unable to update ".to_string() + &script.get_name() + ".  ",
                format!(
                    "Can't find {} because the previous command index, {}, is invalid.\npreviousCommandIndex={}",
                    command, previous_command_index, previous_command_index
                ),
            ];
            ui_harness::open_message_dialog_from_process(
                Some(manager),
                &error_message.join("\n"),
                "ComScriptManager Error",
                Some(axis_id),
            );
            utilities::timestamp_com_script(
                Some("update"),
                Some(command),
                Some(&*script),
                Some(FAILED_STATUS),
            );
            return -1;
        }

        // Update the specified com script command from the CommandParam object
        let command_index = previous_command_index + 1;

        let result: Result<Rc<RefCell<ComScriptCommand>>, BadComScriptException> = match script
            .get_script_command_named_at(
                command,
                command_index,
                true,
                case_insensitive,
                separate_with_a_space,
            ) {
            Err(except) => Err(except),
            Ok(com_script_command) => {
                let updated =
                    params.update_com_script_command(&mut com_script_command.borrow_mut());
                match updated {
                    Err(except) => Err(except),
                    Ok(()) => Ok(com_script_command),
                }
            }
        };
        let com_script_command = match result {
            Err(except) => {
                eprintln!("{}", except);
                let error_message = [
                    "Com file: ".to_string() + &script.get_com_file_name(),
                    "Command: ".to_string() + command,
                    except.get_message().to_string(),
                ];
                ui_harness::open_message_dialog_from_process(
                    None,
                    &error_message.join("\n"),
                    &format!("Can't update {} in {}", command, script.get_com_file_name()),
                    None,
                );
                utilities::timestamp_com_script(
                    Some("update"),
                    Some(command),
                    Some(&*script),
                    Some(FAILED_STATUS),
                );
                return command_index;
            }
            Ok(com_script_command) => com_script_command,
        };

        // Replace the specified command by the updated comScriptCommand
        script.set_script_comand(command_index, &com_script_command.borrow());
        // Write the script back out to disk
        if let Err(except) = script.write_com_file() {
            eprintln!("{}", except);
            let _error_message = [
                "Com file: ".to_string() + &script.get_com_file_name(),
                "Command: ".to_string() + command,
                except.to_string(),
            ];
            ui_harness::open_message_dialog_from_process(
                None,
                &except.to_string(),
                &format!("Can't write {}{}.com", command, axis_id.get_extension()),
                None,
            );
            utilities::timestamp_com_script(
                Some("update"),
                Some(command),
                Some(&*script),
                Some(FAILED_STATUS),
            );
            return command_index;
        }
        utilities::timestamp_com_script(
            Some("update"),
            Some(command),
            Some(&*script),
            Some(FINISHED_STATUS),
        );
        command_index
    }

    /// Java package-private `deleteCommand(BaseManager, ComScript, String, AxisID,
    /// String previousCommand, boolean, boolean)`.
    #[allow(clippy::too_many_arguments)]
    pub fn delete_command_previous_command(
        manager: &'static dyn BaseManager,
        script: Option<&mut ComScript>,
        command: &str,
        axis_id: AxisID,
        previous_command: &str,
        case_insensitive: bool,
        separate_with_a_space: bool,
    ) {
        let script = match script {
            None => {
                let error_message = [
                    "Unable to update comscript.  ".to_string(),
                    "Cannot delete".to_string() + command + ".\nscript=null",
                ];
                ui_harness::open_message_dialog_from_process(
                    Some(manager),
                    &error_message.join("\n"),
                    "ComScriptManager Error",
                    Some(axis_id),
                );
                return;
            }
            Some(script) => script,
        };
        utilities::timestamp_com_script(
            Some("delete"),
            Some(command),
            Some(&*script),
            Some(STARTED_STATUS),
        );

        // locate previous command
        let previous_command_index = script.get_script_command_index(
            previous_command,
            case_insensitive,
            separate_with_a_space,
        );

        if previous_command_index == -1 {
            let error_message = [
                "Unable to update ".to_string() + &script.get_name() + ".  ",
                format!(
                    "Unable to delete {} because the previous command, {}, is missing.",
                    command, previous_command
                ),
            ];
            ui_harness::open_message_dialog_from_process(
                Some(manager),
                &error_message.join("\n"),
                "ComScriptManager Error",
                Some(axis_id),
            );
            utilities::timestamp_com_script(
                Some("delete"),
                Some(command),
                Some(&*script),
                Some(FAILED_STATUS),
            );
            return;
        }

        // Update the specified com script command from the CommandParam object
        let command_index = script.get_script_command_index_at(
            command,
            previous_command_index + 1,
            case_insensitive,
            separate_with_a_space,
        );

        if command_index == -1 {
            utilities::timestamp_com_script(
                Some("delete"),
                Some(command),
                Some(&*script),
                Some(FAILED_STATUS),
            );
            return;
        }
        script.delete_command(command_index);

        // Write the script back out to disk
        if let Err(except) = script.write_com_file() {
            eprintln!("{}", except);
            let _error_message = [
                "Com file: ".to_string() + &script.get_com_file_name(),
                "Command: ".to_string() + command,
                except.to_string(),
            ];
            ui_harness::open_message_dialog_from_process(
                None,
                &except.to_string(),
                &format!("Can't write {}{}.com", command, axis_id.get_extension()),
                None,
            );
            utilities::timestamp_com_script(
                Some("delete"),
                Some(command),
                Some(&*script),
                Some(FAILED_STATUS),
            );
        }
        utilities::timestamp_com_script(
            Some("delete"),
            Some(command),
            Some(&*script),
            Some(FINISHED_STATUS),
        );
    }

    /// Java package-private `deleteCommand(BaseManager, ComScript, String, AxisID,
    /// boolean, boolean)`.
    ///
    /// Fixed in translation: `getScriptCommandIndex` and `getScriptCommand(String, ...)`
    /// agree on whether the command exists (both search by name, and both create it on
    /// a script with no command loaded), so `deleteCommand(-1)` is unreachable from
    /// here; the index is checked anyway rather than letting `ArrayList.remove(-1)`
    /// throw.
    pub fn delete_command(
        manager: &'static dyn BaseManager,
        script: Option<&mut ComScript>,
        command: &str,
        axis_id: AxisID,
        case_insensitive: bool,
        separate_with_a_space: bool,
    ) {
        let script = match script {
            None => {
                eprintln!("java.lang.IllegalStateException");
                let error_message = [
                    "Unable to update comscript.".to_string(),
                    "\nscript=null\ncommand=".to_string() + command,
                ];
                ui_harness::open_message_dialog_from_process(
                    Some(manager),
                    &error_message.join("\n"),
                    "ComScriptManager Error",
                    Some(axis_id),
                );
                return;
            }
            Some(script) => script,
        };
        utilities::timestamp_com_script(
            Some("delete"),
            Some(command),
            Some(&*script),
            Some(STARTED_STATUS),
        );

        // Update the specified com script command from the CommandParam object
        let command_index =
            script.get_script_command_index(command, case_insensitive, separate_with_a_space);

        if let Err(except) =
            script.get_script_command_named(command, case_insensitive, separate_with_a_space)
        {
            eprintln!("{}", except);
            let error_message = [
                "Com file: ".to_string() + &script.get_com_file_name(),
                "Command: ".to_string() + command,
                except.get_message().to_string(),
            ];
            ui_harness::open_message_dialog_from_process(
                None,
                &error_message.join("\n"),
                &format!("Can't delete {} in {}", command, script.get_com_file_name()),
                None,
            );
            utilities::timestamp_com_script(
                Some("delete"),
                Some(command),
                Some(&*script),
                Some(FAILED_STATUS),
            );
            return;
        }

        // Delete the specified command
        if command_index >= 0 {
            script.delete_command(command_index);
        }

        // Write the script back out to disk
        if let Err(except) = script.write_com_file() {
            eprintln!("{}", except);
            let _error_message = [
                "Com file: ".to_string() + &script.get_com_file_name(),
                "Command: ".to_string() + command,
                except.to_string(),
            ];
            ui_harness::open_message_dialog_from_process(
                None,
                &except.to_string(),
                &format!("Can't write {}{}.com", command, axis_id.get_extension()),
                None,
            );
            utilities::timestamp_com_script(
                Some("delete"),
                Some(command),
                Some(&*script),
                Some(FAILED_STATUS),
            );
            return;
        }
        utilities::timestamp_com_script(
            Some("delete"),
            Some(command),
            Some(&*script),
            Some(FINISHED_STATUS),
        );
    }

    /// Java package-private `initialize(BaseManager, CommandParam, ComScript, String,
    /// AxisID, boolean optionalCommand, String previousCommand, boolean
    /// optionalPreviousCommand, boolean, boolean, boolean)`.  Initialize the
    /// CommandParam object from the specified command in the comscript.  True is
    /// returned if the initialization is successful, false if the initialization fails.
    /// `optionalCommand` - missing command returns false, no exception thrown;
    /// `optionalPreviousCommand` - missing previous command returns false, no exception
    /// thrown.
    ///
    /// Fixed in translation: unlike the overload it delegates to, this one does not
    /// check `comScript` for null, and `comScript.isCommandLoaded()` throws
    /// `NullPointerException` (e.g. `getEchoParamFromCombine` before combine.com is
    /// loaded).  A null script returns false, as the other overloads do (`BUGS.md`).
    #[allow(clippy::too_many_arguments)]
    pub fn initialize_previous_command(
        manager: &'static dyn BaseManager,
        param: &mut dyn CommandParam,
        com_script: Option<&mut ComScript>,
        command: &str,
        axis_id: AxisID,
        optional_command: bool,
        previous_command: Option<&str>,
        optional_previous_command: bool,
        case_insensitive: bool,
        separate_with_a_space: bool,
        required: bool,
    ) -> bool {
        let previous_command = match previous_command {
            None => {
                return ComScriptUtil::initialize(
                    manager,
                    param,
                    com_script,
                    command,
                    axis_id,
                    case_insensitive,
                    separate_with_a_space,
                    required,
                );
            }
            Some(previous_command) => previous_command,
        };
        let com_script = match com_script {
            None => return false,
            Some(com_script) => com_script,
        };
        utilities::timestamp_com_script(
            Some("initialize"),
            Some(command),
            Some(&*com_script),
            Some(STARTED_STATUS),
        );
        if !com_script.is_command_loaded() {
            param.initialize_defaults();
        } else {
            // locate previous command
            let previous_command_index = com_script.get_script_command_index(
                previous_command,
                case_insensitive,
                separate_with_a_space,
            );

            if previous_command_index == -1 {
                if optional_previous_command {
                    return false;
                }
                let error_message = [
                    "Unable to read ".to_string() + &com_script.get_name() + ".  ",
                    format!(
                        "Unable to read {} because the previous command, {}, is missing.",
                        command, previous_command
                    ),
                ];
                ui_harness::open_message_dialog_from_process(
                    Some(manager),
                    &error_message.join("\n"),
                    "ComScriptManager Error",
                    Some(axis_id),
                );
                utilities::timestamp_com_script(
                    Some("initialize"),
                    Some(command),
                    Some(&*com_script),
                    Some(FAILED_STATUS),
                );
                return false;
            }

            let command_index = com_script.get_script_command_index_at(
                command,
                previous_command_index + 1,
                case_insensitive,
                separate_with_a_space,
            );

            if command_index == -1 {
                utilities::timestamp_com_script(
                    Some("initialize"),
                    Some(command),
                    Some(&*com_script),
                    Some(FAILED_STATUS),
                );
                return false;
            }
            if optional_command
                && com_script.get_script_command_index_at(
                    command,
                    previous_command_index + 1,
                    case_insensitive,
                    separate_with_a_space,
                ) == -1
            {
                utilities::timestamp_com_script(
                    Some("initialize"),
                    Some(command),
                    Some(&*com_script),
                    Some(FAILED_STATUS),
                );
                return false;
            }
            let result: Result<(), ParseComScriptError> = match com_script
                .get_script_command_named_at(
                    command,
                    command_index,
                    false,
                    case_insensitive,
                    separate_with_a_space,
                ) {
                Err(except) => Err(except.into()),
                Ok(script_command) => {
                    let parsed = param.parse_com_script_command(&script_command.borrow());
                    parsed
                }
            };
            if let Err(except) = result {
                eprintln!("{}", except);
                let error_message = [
                    "Com file: ".to_string() + &com_script.get_com_file_name(),
                    "Command: ".to_string() + command + " after " + previous_command,
                    match &except {
                        ParseComScriptError::BadComScript(_) => {
                            "etomo.comscript.BadComScriptException".to_string()
                        }
                        ParseComScriptError::FortranInputSyntax(_) => {
                            "etomo.comscript.FortranInputSyntaxException".to_string()
                        }
                        ParseComScriptError::InvalidParameter(_) => {
                            "etomo.comscript.InvalidParameterException".to_string()
                        }
                        ParseComScriptError::NumberFormat(_) => {
                            "java.lang.NumberFormatException".to_string()
                        }
                    },
                    match &except {
                        ParseComScriptError::BadComScript(e) => e.get_message().to_string(),
                        ParseComScriptError::FortranInputSyntax(e) => {
                            e.get_message().unwrap_or("null").to_string()
                        }
                        ParseComScriptError::InvalidParameter(e) => {
                            e.get_message().unwrap_or("null").to_string()
                        }
                        ParseComScriptError::NumberFormat(e) => e.clone(),
                    },
                ];
                ui_harness::open_message_dialog_from_process(
                    None,
                    &error_message.join("\n"),
                    "Com Script Command Parse Error",
                    None,
                );
                utilities::timestamp_com_script(
                    Some("initialize"),
                    Some(command),
                    Some(&*com_script),
                    Some(FAILED_STATUS),
                );
                return false;
            }
        }
        utilities::timestamp_com_script(
            Some("initialize"),
            Some(command),
            Some(&*com_script),
            Some(FINISHED_STATUS),
        );
        true
    }

    /// Java package-private `initialize(BaseManager, CommandParam, ComScript, String,
    /// AxisID, boolean optionalCommand, String previousCommand, boolean
    /// optionalPreviousCommand, boolean, boolean, boolean, String requiredOption)`.
    /// Same as `initialize_previous_command`, but a null `previousCommand` delegates to
    /// the `requiredOption` overload.  The same null-script fix applies.
    #[allow(clippy::too_many_arguments)]
    pub fn initialize_previous_command_required_option(
        manager: &'static dyn BaseManager,
        param: &mut dyn CommandParam,
        com_script: Option<&mut ComScript>,
        command: &str,
        axis_id: AxisID,
        optional_command: bool,
        previous_command: Option<&str>,
        optional_previous_command: bool,
        case_insensitive: bool,
        separate_with_a_space: bool,
        required: bool,
        required_option: Option<&str>,
    ) -> bool {
        let previous_command = match previous_command {
            None => {
                return ComScriptUtil::initialize_required_option(
                    manager,
                    param,
                    com_script,
                    command,
                    axis_id,
                    case_insensitive,
                    separate_with_a_space,
                    required,
                    required_option,
                );
            }
            Some(previous_command) => previous_command,
        };
        let com_script = match com_script {
            None => return false,
            Some(com_script) => com_script,
        };
        utilities::timestamp_com_script(
            Some("initialize"),
            Some(command),
            Some(&*com_script),
            Some(STARTED_STATUS),
        );
        if !com_script.is_command_loaded() {
            param.initialize_defaults();
        } else {
            // locate previous command
            let previous_command_index = com_script.get_script_command_index(
                previous_command,
                case_insensitive,
                separate_with_a_space,
            );

            if previous_command_index == -1 {
                if optional_previous_command {
                    return false;
                }
                let error_message = [
                    "Unable to read ".to_string() + &com_script.get_name() + ".  ",
                    format!(
                        "Unable to read {} because the previous command, {}, is missing.",
                        command, previous_command
                    ),
                ];
                ui_harness::open_message_dialog_from_process(
                    Some(manager),
                    &error_message.join("\n"),
                    "ComScriptManager Error",
                    Some(axis_id),
                );
                utilities::timestamp_com_script(
                    Some("initialize"),
                    Some(command),
                    Some(&*com_script),
                    Some(FAILED_STATUS),
                );
                return false;
            }

            let command_index = com_script.get_script_command_index_at(
                command,
                previous_command_index + 1,
                case_insensitive,
                separate_with_a_space,
            );

            if command_index == -1 {
                utilities::timestamp_com_script(
                    Some("initialize"),
                    Some(command),
                    Some(&*com_script),
                    Some(FAILED_STATUS),
                );
                return false;
            }
            if optional_command
                && com_script.get_script_command_index_at(
                    command,
                    previous_command_index + 1,
                    case_insensitive,
                    separate_with_a_space,
                ) == -1
            {
                utilities::timestamp_com_script(
                    Some("initialize"),
                    Some(command),
                    Some(&*com_script),
                    Some(FAILED_STATUS),
                );
                return false;
            }
            let result: Result<(), ParseComScriptError> = match com_script
                .get_script_command_named_at(
                    command,
                    command_index,
                    false,
                    case_insensitive,
                    separate_with_a_space,
                ) {
                Err(except) => Err(except.into()),
                Ok(script_command) => {
                    let parsed = param.parse_com_script_command(&script_command.borrow());
                    parsed
                }
            };
            if let Err(except) = result {
                eprintln!("{}", except);
                let error_message = [
                    "Com file: ".to_string() + &com_script.get_com_file_name(),
                    "Command: ".to_string() + command + " after " + previous_command,
                    match &except {
                        ParseComScriptError::BadComScript(_) => {
                            "etomo.comscript.BadComScriptException".to_string()
                        }
                        ParseComScriptError::FortranInputSyntax(_) => {
                            "etomo.comscript.FortranInputSyntaxException".to_string()
                        }
                        ParseComScriptError::InvalidParameter(_) => {
                            "etomo.comscript.InvalidParameterException".to_string()
                        }
                        ParseComScriptError::NumberFormat(_) => {
                            "java.lang.NumberFormatException".to_string()
                        }
                    },
                    match &except {
                        ParseComScriptError::BadComScript(e) => e.get_message().to_string(),
                        ParseComScriptError::FortranInputSyntax(e) => {
                            e.get_message().unwrap_or("null").to_string()
                        }
                        ParseComScriptError::InvalidParameter(e) => {
                            e.get_message().unwrap_or("null").to_string()
                        }
                        ParseComScriptError::NumberFormat(e) => e.clone(),
                    },
                ];
                ui_harness::open_message_dialog_from_process(
                    None,
                    &error_message.join("\n"),
                    "Com Script Command Parse Error",
                    None,
                );
                utilities::timestamp_com_script(
                    Some("initialize"),
                    Some(command),
                    Some(&*com_script),
                    Some(FAILED_STATUS),
                );
                return false;
            }
        }
        utilities::timestamp_com_script(
            Some("initialize"),
            Some(command),
            Some(&*com_script),
            Some(FINISHED_STATUS),
        );
        true
    }
}
