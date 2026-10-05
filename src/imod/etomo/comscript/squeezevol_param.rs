//! `IMOD/Etomo/src/etomo/comscript/SqueezevolParam.java`.
//!
//! The command array and output file are rebuilt by `getCommandArray`, which
//! the `Command` trait calls through `&self` on a shared param, so those two
//! fields sit behind a `Mutex`.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use super::command::Command;
use super::command_details::CommandDetails;
use super::command_mode::CommandMode;
use super::const_squeezevol_param::ConstSqueezevolParam;
use super::field_interface::{self, FieldInterface};
use super::process_details::ProcessDetails;
use super::trimvol_param::TrimvolParam;
use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::storage::loggable::{Loggable, LoggableException};
use crate::imod::etomo::storage::storable::StorableValue;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::{
    ConstEtomoNumber, Type, java_lang_float_to_string,
};
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::r#type::file_key::FileKey;
use crate::imod::etomo::r#type::file_type::{self, FileType};
use crate::imod::etomo::r#type::image_file_type::ImageFileType;
use crate::imod::etomo::r#type::process_name::ProcessName;

/// Java private static `GROUP_STRING`.
const GROUP_STRING: &str = "Squeezevol";
/// Java private static `COMMAND_SIZE`.
const COMMAND_SIZE: usize = 4;
/// Java `COMMAND_NAME`.
pub const COMMAND_NAME: &str = "squeezevol";
/// Java `REDUCTION_FACTOR_DEFAULT` (a `Float`).
pub const REDUCTION_FACTOR_DEFAULT: f32 = 1.25f32;

/// Java final `SqueezevolParam`.
pub struct SqueezevolParam {
    reduction_factor_x: EtomoNumber,
    reduction_factor_y: EtomoNumber,
    reduction_factor_z: EtomoNumber,
    manager: &'static ApplicationManager,
    command_array: Mutex<Option<Vec<String>>>,
    output_file: Mutex<Option<PathBuf>>,
    input_file: Option<String>,
    /// Java package-private `flipped`.
    pub(crate) flipped: bool,
}

impl SqueezevolParam {
    /// Java `SqueezevolParam(ApplicationManager)`.
    pub fn new(manager: &'static ApplicationManager) -> SqueezevolParam {
        let mut param = SqueezevolParam {
            reduction_factor_x: EtomoNumber::new_with_type_and_name(
                Type::Double,
                "ReductionFactorX",
            ),
            reduction_factor_y: EtomoNumber::new_with_type_and_name(
                Type::Double,
                "ReductionFactorY",
            ),
            reduction_factor_z: EtomoNumber::new_with_type_and_name(
                Type::Double,
                "ReductionFactorZ",
            ),
            manager,
            command_array: Mutex::new(None),
            output_file: Mutex::new(None),
            input_file: None,
            flipped: false,
        };
        param.reset();
        param
    }

    /// Java `setReductionFactorX(String)`.
    pub fn set_reduction_factor_x(
        &mut self,
        reduction_factor_x: Option<&str>,
    ) -> &ConstEtomoNumber {
        self.reduction_factor_x.set_string(reduction_factor_x);
        &self.reduction_factor_x
    }

    /// Java `setReductionFactorY(String)`.
    pub fn set_reduction_factor_y(&mut self, reduction_factor_y: Option<&str>) {
        self.reduction_factor_y.set_string(reduction_factor_y);
    }

    /// Java `setReductionFactorZ(String)`.
    pub fn set_reduction_factor_z(&mut self, reduction_factor_z: Option<&str>) {
        self.reduction_factor_z.set_string(reduction_factor_z);
    }

    /// Java `setFlipped(boolean)`.
    pub fn set_flipped(&mut self, flipped: bool) -> bool {
        self.flipped = flipped;
        self.flipped
    }

    /// Java `setInputFile(ImageFileType)`.
    pub fn set_input_file(&mut self, image_file_type: &ImageFileType) {
        self.input_file = image_file_type.get_file_name(self.manager);
    }

    /// Java private `reset`.
    fn reset(&mut self) {
        self.reduction_factor_x.reset();
        self.reduction_factor_y.reset();
        self.reduction_factor_z.reset();
    }

    /// Java private `genOptions`.
    fn gen_options(&self) -> Vec<Option<String>> {
        let mut options: Vec<Option<String>> = Vec::new();
        // Squeezevol no long has an interface, and reducefiltvol only needs values from
        // actual squeezevol runs.
        options.push(Some("-x".to_owned()));
        if !self.reduction_factor_x.is_null() {
            options.push(Some(self.reduction_factor_x.to_string()));
        } else {
            options.push(Some(java_lang_float_to_string(REDUCTION_FACTOR_DEFAULT)));
        }
        options.push(Some("-y".to_owned()));
        // SqueezevolParam.java:142 and :149 test `reductionFactorX.isNull()` before
        // writing the Y and Z factors - a copy-paste typo, so a stored X with a null Y
        // or Z wrote "null" for that factor.  Fixed in translation: each factor tests
        // its own null state.
        if !self.reduction_factor_y.is_null() {
            options.push(Some(self.reduction_factor_y.to_string()));
        } else {
            options.push(Some(java_lang_float_to_string(REDUCTION_FACTOR_DEFAULT)));
        }
        options.push(Some("-z".to_owned()));
        if !self.reduction_factor_z.is_null() {
            options.push(Some(self.reduction_factor_z.to_string()));
        } else {
            options.push(Some(java_lang_float_to_string(REDUCTION_FACTOR_DEFAULT)));
        }
        options.push(self.input_file.clone());
        // output is dataset.sqz
        let output_file = ImageFileType::SqueezeVolOutput.get_file(self.manager);
        // `outputFile.getName()`: getFile is never null for a non-null manager and
        // SQUEEZE_VOL_OUTPUT; an absent name would be Java's NullPointerException and
        // is left out here.
        options.push(
            output_file
                .as_ref()
                .and_then(|file| file.file_name())
                .map(|name| name.to_string_lossy().into_owned()),
        );
        *self.output_file.lock().unwrap() = output_file;
        options
    }

    /// Java private `getInputFileName(String)`.
    #[allow(dead_code)]
    fn get_input_file_name(&self, _dataset_name: Option<&str>) -> Option<String> {
        TrimvolParam::get_output_file_name_for(self.manager, AxisID::Only)
    }

    /// Java private static `createPrepend(String)`.
    fn create_prepend(prepend: &str) -> String {
        // `prepend == ""` is a reference comparison; every caller passes the interned
        // literal for the empty case.
        if prepend.is_empty() {
            return GROUP_STRING.to_owned();
        }
        format!("{prepend}.{GROUP_STRING}")
    }

    /// Java private `createCommand`.  The array is sized `options.size() +
    /// COMMAND_SIZE`; a null option stays a null element, which `ProcessBuilder`
    /// would reject, so null options are dropped here.
    fn create_command(&self) {
        let options = self.gen_options();
        let mut command_array: Vec<String> = Vec::with_capacity(options.len() + COMMAND_SIZE);
        command_array.push("python".to_owned());
        command_array.push("-u".to_owned());
        // Java string concatenation writes a null path as "null"
        let script_path = etomo_director::INSTANCE
            .get_python_script_path()
            .as_deref()
            .unwrap_or("null")
            .to_owned();
        command_array.push(format!("{script_path}{COMMAND_NAME}"));
        command_array.push("-PID".to_owned());
        for option in options.into_iter().flatten() {
            command_array.push(option);
        }
        *self.command_array.lock().unwrap() = Some(command_array);
    }

    /// Java `isFlipped`.
    pub fn is_flipped(&self) -> bool {
        self.flipped
    }

    /// Java `equals(ConstSqueezevolParam)`.
    pub fn equals(&self, param: &dyn ConstSqueezevolParam) -> bool {
        if !self
            .reduction_factor_x
            .equals_const_etomo_number(Some(param.get_reduction_factor_x()))
        {
            return false;
        }
        if !self
            .reduction_factor_y
            .equals_const_etomo_number(Some(param.get_reduction_factor_y()))
        {
            return false;
        }
        if !self
            .reduction_factor_z
            .equals_const_etomo_number(Some(param.get_reduction_factor_z()))
        {
            return false;
        }
        true
    }
}

impl StorableValue for SqueezevolParam {
    /// Java `store(Properties)`.
    fn store(&self, props: &mut BTreeMap<String, String>) {
        self.store_with_prepend(props, "");
    }

    /// Java `store(Properties, String)`.
    fn store_with_prepend(&self, props: &mut BTreeMap<String, String>, prepend: &str) {
        let prepend = SqueezevolParam::create_prepend(prepend);
        let _group = format!("{prepend}.");
        self.reduction_factor_x
            .base
            .store_with_prepend(props, Some(&prepend));
        self.reduction_factor_y
            .base
            .store_with_prepend(props, Some(&prepend));
        self.reduction_factor_z
            .base
            .store_with_prepend(props, Some(&prepend));
    }

    /// Java `load(Properties)`.  Get the objects attributes from the properties
    /// object.
    fn load(&mut self, props: &mut BTreeMap<String, String>) {
        self.load_with_prepend(props, "");
    }

    /// Java `load(Properties, String)`.
    fn load_with_prepend(&mut self, props: &mut BTreeMap<String, String>, prepend: &str) {
        self.reset();
        let prepend = SqueezevolParam::create_prepend(prepend);
        let _group = format!("{prepend}.");
        EtomoNumber::load_with_prepend(&mut self.reduction_factor_x, props, Some(&prepend));
        EtomoNumber::load_with_prepend(&mut self.reduction_factor_y, props, Some(&prepend));
        EtomoNumber::load_with_prepend(&mut self.reduction_factor_z, props, Some(&prepend));
    }
}

impl Command for SqueezevolParam {
    /// Java `command instanceof ProcessDetails`: this class is a `CommandDetails`.
    fn get_process_details(&self) -> Option<&dyn ProcessDetails> {
        Some(self)
    }

    fn get_axis_id(&self) -> AxisID {
        AxisID::Only
    }

    fn get_command_mode(&self) -> Option<&dyn CommandMode> {
        None
    }

    fn get_process_name(&self) -> Option<ProcessName> {
        Some(ProcessName::SQUEEZEVOL)
    }

    fn get_command(&self) -> Option<String> {
        Some(COMMAND_NAME.to_owned())
    }

    fn get_command_name(&self) -> Option<String> {
        Some(COMMAND_NAME.to_owned())
    }

    /// Java `getCommandLine`.  Get command array used to run command.  Not for
    /// running the command.
    fn get_command_line(&self) -> Option<String> {
        let command_array = self.command_array.lock().unwrap();
        let command_array = match command_array.as_ref() {
            None => return Some(String::new()),
            Some(command_array) => command_array,
        };
        let mut buffer = String::new();
        for element in command_array {
            buffer.push_str(&format!("{element} "));
        }
        Some(buffer)
    }

    /// Java `getCommandArray`.  Get command array to run.
    fn get_command_array(&self) -> Option<Vec<String>> {
        self.create_command();
        self.command_array.lock().unwrap().clone()
    }

    fn get_command_input_file(&self) -> Option<PathBuf> {
        None
    }

    fn get_command_output_file(&self) -> Option<PathBuf> {
        self.output_file.lock().unwrap().clone()
    }

    /// Java `getOutputImageFileType` (deprecated 3/15/2019).
    fn get_output_image_file_type(&self) -> Option<Arc<FileType>> {
        Some(file_type::CLASS.squeeze_vol_output.clone())
    }

    fn get_output_image_file_key(&self) -> Option<FileKey> {
        Some(FileKey::clone(&file_type::CLASS.squeeze_vol_output))
    }

    /// Java `getOutputImageFileType2` (deprecated 3/15/2019).
    fn get_output_image_file_type2(&self) -> Option<Arc<FileType>> {
        None
    }

    fn get_output_image_file_key2(&self) -> Option<FileKey> {
        None
    }

    fn is_message_reporter(&self) -> bool {
        false
    }

    fn get_subcommand_process_name(&self) -> Option<String> {
        None
    }

    fn get_subcommand_details(&self) -> Option<&dyn CommandDetails> {
        None
    }
}

/// Every getter but `getBooleanValue(FLIPPED)` throws
/// `IllegalArgumentException("field=" + field)` in Java; they return `None`.
impl ProcessDetails for SqueezevolParam {
    fn get_int_value(&self, _field: &dyn FieldInterface) -> Option<i32> {
        None
    }

    fn get_boolean_value(&self, field: &dyn FieldInterface) -> Option<bool> {
        if field_interface::as_field::<Fields>(field) == Some(&Fields::Flipped) {
            return Some(self.flipped);
        }
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

    fn get_int_key_list(
        &self,
        _field: &dyn FieldInterface,
    ) -> Option<crate::imod::etomo::r#type::int_key_list::IntKeyList> {
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
    ) -> Option<crate::imod::etomo::r#type::iterator_element_list::IteratorElementList> {
        None
    }
}

impl Loggable for SqueezevolParam {
    /// Java `getName`.
    fn get_name(&self) -> String {
        COMMAND_NAME.to_owned()
    }

    /// Java `getLogMessage`, which returns null: no message lines.
    fn get_log_message(&self) -> Result<Vec<Option<String>>, LoggableException> {
        Ok(Vec::new())
    }
}

impl ConstSqueezevolParam for SqueezevolParam {
    fn get_reduction_factor_x(&self) -> &ConstEtomoNumber {
        &self.reduction_factor_x
    }

    fn get_reduction_factor_y(&self) -> &ConstEtomoNumber {
        &self.reduction_factor_y
    }

    fn get_reduction_factor_z(&self) -> &ConstEtomoNumber {
        &self.reduction_factor_z
    }
}

/// Java nested class `SqueezevolParam.Fields`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fields {
    /// Java `FLIPPED`.
    Flipped,
}

impl FieldInterface for Fields {}
