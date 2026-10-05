//! `IMOD/Etomo/src/etomo/comscript/PeetParserParam.java`.
//!
//! The `sh $PARTICLE_DIR/bin/prmParser <name>.prm` command, run before the PEET
//! processchunks run.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::command::Command;
use super::command_details::CommandDetails;
use super::command_mode::CommandMode;
use super::field_interface::{self, FieldInterface};
use super::process_details::ProcessDetails;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::storage::loggable::{Loggable, LoggableException};
use crate::imod::etomo::storage::matlab_param::{self, MatlabParam};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::{
    ConstEtomoNumber, java_lang_string_matches_whitespace,
};
use crate::imod::etomo::r#type::file_key::{self, FileKey};
use crate::imod::etomo::r#type::file_type::FileType;
use crate::imod::etomo::r#type::int_key_list::IntKeyList;
use crate::imod::etomo::r#type::iterator_element_list::IteratorElementList;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::ui::swing::ui_harness;
use crate::imod::etomo::util::environment_variable;
use crate::imod::etomo::util::utilities;

/// Java `public static final String rcsid`.
pub const RCSID: &str = "$Id$";

/// Java private static final `PROCESS_NAME`.
const PROCESS_NAME: ProcessName = ProcessName::PEET_PARSER;

/// Java `public static final class Fields implements FieldInterface`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fields {
    /// Java `ITERATION_LIST_SIZE`.
    IterationListSize,
}

impl FieldInterface for Fields {}

/// Java `public final class PeetParserParam implements CommandDetails`.
pub struct PeetParserParam {
    /// Java private final `prmFile`.
    prm_file: PathBuf,
    /// Java private final `manager`.
    manager: &'static dyn BaseManager,
    /// Java private `debug`, initially false.
    debug: bool,
    /// Java private `iterationListSize`.
    iteration_list_size: i32,
    /// Java private `lstThresholdsArray`.
    lst_thresholds_array: Option<Vec<String>>,
    /// Java private `szVol`, initially null.
    sz_vol: Option<String>,
    /// Java private `lstFlagAllTom`, initially null.
    lst_flag_all_tom: Option<String>,
}

impl PeetParserParam {
    /// Java `PeetParserParam(BaseManager, File)`.
    pub fn new(manager: &'static dyn BaseManager, prm_file: &Path) -> PeetParserParam {
        PeetParserParam {
            prm_file: prm_file.to_path_buf(),
            manager,
            debug: false,
            iteration_list_size: 0,
            lst_thresholds_array: None,
            sz_vol: None,
            lst_flag_all_tom: None,
        }
    }

    /// Java `getLogFile()`.
    pub fn get_log_file(&self) -> PathBuf {
        PathBuf::from(format!(
            "{}.log",
            utilities::java_io_file_get_absolute_path(&self.prm_file.to_string_lossy())
        ))
    }

    /// Java `setParameters(MatlabParam)`.
    pub fn set_parameters(&mut self, matlab_param: &MatlabParam) {
        self.iteration_list_size = matlab_param.get_iteration_list_size();
        self.lst_thresholds_array = Some(matlab_param.get_lst_thresholds_expanded_array());
        self.sz_vol = matlab_param.get_sz_vol();
        self.lst_flag_all_tom = matlab_param.get_lst_flag_all_tom();
    }
}

impl Command for PeetParserParam {
    /// Java `getCommandArray()`.
    fn get_command_array(&self) -> Option<Vec<String>> {
        let mut command = vec![String::new(); 3];
        command[0] = "sh".to_owned();
        let particle_dir = environment_variable::INSTANCE.get_value(
            Some(self.manager),
            self.manager.get_property_user_dir().as_deref(),
            environment_variable::PARTICLE_DIR,
            Some(AxisID::Only),
        );
        if java_lang_string_matches_whitespace(&particle_dir) {
            ui_harness::open_message_dialog_from_process(
                Some(self.manager),
                "The environment variables PARTICLE_DIR has not been set.  Set it to the location of the directory containing the PEET software.  Make sure the PEET package is installed (typically installed in /usr/local/Particle).  To download PEET, go to ftp://bio3d.colorado.edu/PEET.",
                "Environment Error",
                None,
            );
            return None;
        }
        let command_file = Path::new(&environment_variable::INSTANCE.get_value(
            Some(self.manager),
            self.manager.get_property_user_dir().as_deref(),
            environment_variable::PARTICLE_DIR,
            Some(AxisID::Only),
        ))
        .join("bin")
        .join(PROCESS_NAME.to_string());
        command[1] = utilities::java_io_file_get_absolute_path(&command_file.to_string_lossy());
        command[2] = utilities::java_io_file_get_name(&self.prm_file.to_string_lossy());
        if self.debug {
            eprintln!("{}", command.join(" "));
        }
        Some(command)
    }

    fn get_command_mode(&self) -> Option<&dyn CommandMode> {
        None
    }

    fn is_message_reporter(&self) -> bool {
        false
    }

    fn get_command_output_file(&self) -> Option<PathBuf> {
        None
    }

    fn get_command_input_file(&self) -> Option<PathBuf> {
        None
    }

    fn get_command_name(&self) -> Option<String> {
        Some(PROCESS_NAME.to_string())
    }

    /// Java deprecated `getOutputImageFileType()`.
    fn get_output_image_file_type(&self) -> Option<Arc<FileType>> {
        None
    }

    fn get_output_image_file_key(&self) -> Option<FileKey> {
        Some(file_key::AVERAGED_VOLUMES.clone())
    }

    /// Java deprecated `getOutputImageFileType2()`.
    fn get_output_image_file_type2(&self) -> Option<Arc<FileType>> {
        None
    }

    fn get_output_image_file_key2(&self) -> Option<FileKey> {
        Some(file_key::REFERENCE_VOLUMES.clone())
    }

    fn get_process_name(&self) -> Option<ProcessName> {
        Some(PROCESS_NAME)
    }

    fn get_command_line(&self) -> Option<String> {
        self.get_command()
    }

    fn get_subcommand_details(&self) -> Option<&dyn CommandDetails> {
        None
    }

    fn get_subcommand_process_name(&self) -> Option<String> {
        None
    }

    fn get_axis_id(&self) -> AxisID {
        AxisID::Only
    }

    /// Java `getCommand()`.
    fn get_command(&self) -> Option<String> {
        let command = PROCESS_NAME.to_string();
        if self.debug {
            eprintln!("{command}");
        }
        Some(command)
    }

    fn get_process_details(&self) -> Option<&dyn ProcessDetails> {
        Some(self)
    }
}

impl Loggable for PeetParserParam {
    /// Java `getName()`.
    fn get_name(&self) -> String {
        PROCESS_NAME.to_string()
    }

    /// Java `getLogMessage()`.
    fn get_log_message(&self) -> Result<Vec<Option<String>>, LoggableException> {
        let mut message = Vec::new();
        message.push(Some(format!(
            "{} = {}",
            matlab_param::SZ_VOL_KEY,
            self.sz_vol.as_deref().unwrap_or("null")
        )));
        message.push(Some(format!(
            "{} = {}",
            matlab_param::LST_FLAG_ALL_TOM_KEY,
            self.lst_flag_all_tom.as_deref().unwrap_or("null")
        )));
        let mut buffer = format!("{} = ", matlab_param::LST_THRESHOLDS_KEY);
        if let Some(lst_thresholds_array) = &self.lst_thresholds_array {
            buffer.push_str(&lst_thresholds_array.join(", "));
        }
        message.push(Some(buffer));
        Ok(message)
    }
}

/// Every Java `ProcessDetails` getter except `getIntValue(ITERATION_LIST_SIZE)` throws
/// `IllegalArgumentException("field=" + field)`; that is `None` here.
impl ProcessDetails for PeetParserParam {
    fn get_int_value(&self, field: &dyn FieldInterface) -> Option<i32> {
        if field_interface::as_field::<Fields>(field) == Some(&Fields::IterationListSize) {
            return Some(self.iteration_list_size);
        }
        None
    }

    fn get_boolean_value(&self, _field: &dyn FieldInterface) -> Option<bool> {
        None
    }

    fn get_double_value(&self, _field: &dyn FieldInterface) -> Option<f64> {
        None
    }

    fn get_hashtable(
        &self,
        _field: &dyn FieldInterface,
    ) -> Option<super::process_details::Hashtable> {
        None
    }

    fn get_etomo_number(&self, _field: &dyn FieldInterface) -> Option<ConstEtomoNumber> {
        None
    }

    fn get_int_key_list(&self, _field: &dyn FieldInterface) -> Option<IntKeyList> {
        None
    }

    fn get_string(&self, _field: &dyn FieldInterface) -> Option<String> {
        None
    }

    fn get_string_array(&self, _field: &dyn FieldInterface) -> Option<Vec<String>> {
        None
    }

    fn get_iterator_element_list(
        &self,
        _field: &dyn FieldInterface,
    ) -> Option<IteratorElementList> {
        None
    }
}
