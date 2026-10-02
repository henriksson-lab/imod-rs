//! `IMOD/Etomo/src/etomo/storage/Directive.java`.
//!
//! This class represents information about a directive required by the Directive File
//! Editor.  Directives which can have multiple axes (other then setupset.copyarg
//! directives), are handled with a single Directive instance.
//!
//! A `Directive` is shared (the `DirectiveMap` hands out `Arc<Directive>`) and mutated
//! through `&self`, so its mutable fields sit behind a `Mutex`.  Overloads carry the
//! suffix of their parameter types (`setValue(int)` is `set_value_int`,
//! `setDefaultValue(String)` is `set_default_value_string`, ...).
//!
//! The nested abstract class `Directive.Value` and its four subclasses are the enum
//! `Value` over the structs `BooleanValue`, `NumericValue`, `NumericPairValue` and
//! `StringValue`; `ValueFactory` is a unit struct with the one static method.
//!
//! **`DOUBLE_NULL_VALUE` is `Double.NaN`.**  Every `input == EtomoNumber.DOUBLE_NULL_VALUE`
//! in this unit is therefore false for every input in Java, NaN included, so the
//! "null double" branches can never run.  Fixed in translation: the comparison is
//! `is_nan()`, which is what the null test evidently means (see each site).

use std::sync::{Arc, Mutex, MutexGuard};

use crate::imod::etomo::comscript::fortran_input_string::FortranInputString;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::storage::autodoc::autodoc_tokenizer::DEFAULT_DELIMITER;
use crate::imod::etomo::storage::directive_descr::DirectiveDescr;
use crate::imod::etomo::storage::directive_descr_choice_list::DirectiveDescrChoiceList;
use crate::imod::etomo::storage::directive_descr_etomo_column::DirectiveDescrEtomoColumn;
use crate::imod::etomo::storage::directive_name::DirectiveName;
use crate::imod::etomo::storage::directive_type::DirectiveType;
use crate::imod::etomo::storage::directive_value_type::DirectiveValueType;
use crate::imod::etomo::storage::directive_values::DirectiveValues;
use crate::imod::etomo::storage::log_file::{Handle, LogFileError, WriterId};
use crate::imod::etomo::r#type::const_etomo_number::{
    ConstEtomoNumber, INTEGER_NULL_VALUE, Type, java_lang_double_to_string,
    java_lang_string_matches_whitespace, java_lang_string_trim,
};
use crate::imod::etomo::r#type::const_string_parameter::ConstStringParameter;
use crate::imod::etomo::r#type::debug_level::DebugLevel;
use crate::imod::etomo::r#type::directive_file_type::{self, DirectiveFileType};
use crate::imod::etomo::r#type::directive_interface::DirectiveInterface;
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;

/// Java final `Directive implements DirectiveInterface`.
pub struct Directive {
    /// Java private final field `directiveName = new DirectiveName()`.
    directive_name: DirectiveName,
    /// Java private final field `batch`.
    batch: bool,
    /// Java private final field `description`.
    description: Option<String>,
    /// Java private final field `etomoColumn`.
    etomo_column: Option<DirectiveDescrEtomoColumn>,
    /// Java private final field `template`.
    template: bool,
    /// Java private final field `values`.
    values: Mutex<DirectiveValues>,
    /// Java private final field `valueType`.
    value_type: Option<DirectiveValueType>,
    /// Java private final field `label`.
    label: Option<String>,
    /// Java private final field `choiceList`.
    choice_list: Option<DirectiveDescrChoiceList>,
    /// Java private field `inDirectiveFile`, initialised to null.
    in_directive_file: Mutex<Option<Vec<bool>>>,
    /// Java private field `include`, initialised to false.
    include: Mutex<bool>,
    /// Java private field `debug`, initialised from the arguments' debug level.  Never
    /// read in the source.
    debug: DebugLevel,
}

impl Directive {
    /// Java `Directive(DirectiveDescr)`.
    pub fn new_directive_descr(descr: &dyn DirectiveDescr) -> Directive {
        let mut directive_name = DirectiveName::new();
        let debug = etomo_director::ARGUMENTS.lock().unwrap().get_debug_level();
        directive_name.set_key_directive_descr(descr);
        let value_type = descr.get_value_type();
        let values = DirectiveValues::new(value_type);
        let description = descr.get_description();
        let batch = descr.is_batch();
        let template = descr.is_template();
        let etomo_column = descr.get_etomo_column();
        let label = descr.get_label();
        let choice_list = descr.get_choice_list();
        Directive {
            directive_name,
            batch,
            description,
            etomo_column,
            template,
            values: Mutex::new(values),
            value_type,
            label,
            choice_list,
            in_directive_file: Mutex::new(None),
            include: Mutex::new(false),
            debug,
        }
    }

    /// Java `Directive(DirectiveName)`.  Constructor for undefined directives.  The
    /// constructor does a deep copy of the parameter.
    pub fn new_directive_name(directive_name: &DirectiveName) -> Directive {
        let mut this_directive_name = DirectiveName::new();
        let debug = etomo_director::ARGUMENTS.lock().unwrap().get_debug_level();
        this_directive_name.deep_copy(directive_name);
        let value_type = Some(DirectiveValueType::Unknown);
        let values = DirectiveValues::new(value_type);
        Directive {
            directive_name: this_directive_name,
            // No description of this directive, so allow it to exist in any type of
            // directive
            // file.
            batch: true,
            description: None,
            etomo_column: None,
            template: true,
            values: Mutex::new(values),
            value_type,
            label: None,
            choice_list: None,
            in_directive_file: Mutex::new(None),
            include: Mutex::new(false),
            debug,
        }
    }

    /// Java package-private `write(LogFile.Handle, LogFile.WriterId)`.
    pub(crate) fn write(&self, log_file: &Arc<Handle>, id: &WriterId) -> Result<(), LogFileError> {
        let value_string = {
            let values = self.values.lock().unwrap();
            let value = values.get_value();
            let mut value_string = String::new();
            if let Some(value) = value {
                value_string = value.to_string();
            }
            value_string
        };
        log_file.write(
            Some(
                format!(
                    "{}{}{}",
                    self.directive_name
                        .get_name()
                        .unwrap_or_else(|| "null".to_string()),
                    DEFAULT_DELIMITER,
                    value_string
                )
                .as_str(),
            ),
            id,
        )?;
        log_file.new_line(id)?;
        Ok(())
    }

    /// Java `setDebug(DebugLevel)`.
    pub fn set_debug(&self, input: DebugLevel) {
        self.values.lock().unwrap().set_debug(input);
    }

    /// Java `resetDebug()`.
    pub fn reset_debug(&self) {
        self.values.lock().unwrap().reset_debug();
    }

    /// Java `equals(DirectiveType)`.
    pub fn equals_directive_type(&self, input: Option<DirectiveType>) -> bool {
        self.directive_name.equals_directive_type(input)
    }

    /// Java `getEtomoColumn()`.
    pub fn get_etomo_column(&self) -> Option<DirectiveDescrEtomoColumn> {
        self.etomo_column
    }

    /// Java `getInDirectiveFileDebugString()`.
    pub fn get_in_directive_file_debug_string(&self) -> Option<String> {
        let in_directive_file = self.in_directive_file.lock().unwrap();
        if let Some(in_directive_file) = in_directive_file.as_ref() {
            let mut buffer = String::new();
            for i in 0..in_directive_file.len() {
                if in_directive_file[i] {
                    let file_type = match DirectiveFileType::get_instance_from_index(i as i32) {
                        None => "null".to_string(),
                        Some(file_type) => file_type.to_string(),
                    };
                    buffer.push_str(&format!(
                        "{}in {} directive file",
                        if !buffer.is_empty() { ", " } else { "" },
                        file_type
                    ));
                }
            }
            if !buffer.is_empty() {
                return Some(buffer);
            }
        }
        None
    }

    /// Java `getChoiceList()`.
    pub fn get_choice_list(&self) -> Option<&DirectiveDescrChoiceList> {
        self.choice_list.as_ref()
    }

    /// Java `getDescription()`.
    pub fn get_description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    /// Java `getKey()`.
    pub fn get_key(&self) -> Option<String> {
        self.directive_name.get_key()
    }

    /// Java `getKeyDescription()`.
    pub fn get_key_description(&self) -> Option<String> {
        self.directive_name.get_key_description()
    }

    /// Java `getName()`.
    pub fn get_name(&self) -> Option<String> {
        self.directive_name.get_name()
    }

    /// Java `getTitle()`.
    pub fn get_title(&self) -> Option<String> {
        if let Some(label) = &self.label {
            return Some(label.clone());
        }
        self.directive_name.get_title()
    }

    /// Java `getType()`.
    pub fn get_type(&self) -> Option<DirectiveType> {
        self.directive_name.get_type()
    }

    /// Java `getValues()`.  Java returns the shared `DirectiveValues`; the lock guard
    /// stands in for the reference.
    pub fn get_values(&self) -> MutexGuard<'_, DirectiveValues> {
        self.values.lock().unwrap()
    }

    /// Java `getValueType()`.
    pub fn get_value_type(&self) -> Option<DirectiveValueType> {
        self.value_type
    }

    /// Java `isBatch()`.
    pub fn is_batch(&self) -> bool {
        self.batch
    }

    /// Java `isChoiceList()`.
    pub fn is_choice_list(&self) -> bool {
        self.choice_list.is_some() && !self.choice_list.as_ref().unwrap().is_empty()
    }

    /// Java `isInclude()`.
    pub fn is_include(&self) -> bool {
        *self.include.lock().unwrap()
    }

    /// Java `isCopyArg()`.
    pub fn is_copy_arg(&self) -> bool {
        self.directive_name.is_copy_arg()
    }

    /// Java `isValid()`.
    pub fn is_valid(&self) -> bool {
        self.directive_name.is_valid()
    }

    /// Java `isInDirectiveFile(int)`.  Returns true if inDirectiveFile is true for the
    /// index.
    pub fn is_in_directive_file(&self, index: i32) -> bool {
        let in_directive_file = self.in_directive_file.lock().unwrap();
        if in_directive_file.is_none() || index < 0 || index >= directive_file_type::NUM {
            return false;
        }
        in_directive_file.as_ref().unwrap()[index as usize]
    }

    /// Java `isTemplate()`.
    pub fn is_template(&self) -> bool {
        self.template
    }

    /// Java `setDefaultValue(boolean)`.
    pub fn set_default_value_boolean(&self, input: bool) {
        self.values.lock().unwrap().set_default_value_boolean(input);
    }

    /// Java `setDefaultValue(ConstEtomoNumber)`.
    pub fn set_default_value_const_etomo_number(&self, input: Option<&ConstEtomoNumber>) {
        self.values
            .lock()
            .unwrap()
            .set_default_value_const_etomo_number(input);
    }

    /// Java `setDefaultValue(String)`.
    pub fn set_default_value_string(&self, input: Option<&str>) {
        self.values.lock().unwrap().set_default_value_string(input);
    }

    /// Java `setDefaultValue(int)`.
    pub fn set_default_value_int(&self, input: i32) {
        self.values.lock().unwrap().set_default_value_int(input);
    }

    /// Java `setInclude(boolean)`.
    pub fn set_include(&self, input: bool) {
        *self.include.lock().unwrap() = input;
    }

    /// Java `setInDirectiveFile(DirectiveFileType, boolean)`.
    pub fn set_in_directive_file(&self, r#type: Option<DirectiveFileType>, input: bool) {
        let r#type = match r#type {
            None => return,
            Some(r#type) => r#type,
        };
        let index = r#type.get_index();
        let mut in_directive_file = self.in_directive_file.lock().unwrap();
        if in_directive_file.is_none() {
            // Initialize inDirectiveFile, and add input value.
            let mut array = vec![false; directive_file_type::NUM as usize];
            for i in 0..array.len() {
                if i as i32 == index {
                    array[i] = input;
                } else {
                    array[i] = false;
                }
            }
            *in_directive_file = Some(array);
        } else if index >= 0 && index < in_directive_file.as_ref().unwrap().len() as i32 {
            // Add input value.
            in_directive_file.as_mut().unwrap()[index as usize] = input;
        }
    }

    /// Java `setValue(File)`.
    pub fn set_value_file(&self, input: Option<&std::path::Path>) {
        match input {
            Some(input) => {
                // `File.getAbsolutePath()`: a relative path is resolved against the
                // current directory without normalisation.
                let absolute = if input.is_absolute() {
                    input.to_path_buf()
                } else {
                    std::env::current_dir()
                        .map(|dir| dir.join(input))
                        .unwrap_or_else(|_| input.to_path_buf())
                };
                self.values
                    .lock()
                    .unwrap()
                    .set_value_string(Some(absolute.to_string_lossy().as_ref()));
            }
            None => {
                self.values.lock().unwrap().set_value_string(Some(""));
            }
        }
    }

    /// Java `resetValue()`.
    pub fn reset_value(&self) {
        self.values.lock().unwrap().reset_value();
    }

    /// Java `setValue(boolean)`.
    pub fn set_value_boolean(&self, input: bool) {
        self.values.lock().unwrap().set_value_boolean(input);
    }

    /// Java `setValue(ConstEtomoNumber)`.
    pub fn set_value_const_etomo_number(&self, input: Option<&ConstEtomoNumber>) {
        self.values
            .lock()
            .unwrap()
            .set_value_const_etomo_number(input);
    }

    /// Java `setValue(String)`.
    pub fn set_value_string(&self, input: Option<&str>) {
        self.values.lock().unwrap().set_value_string(input);
    }

    /// Java `setValue(ConstStringParameter)`.
    pub fn set_value_const_string_parameter(&self, input: Option<&dyn ConstStringParameter>) {
        self.values
            .lock()
            .unwrap()
            .set_value_const_string_parameter(input);
    }

    /// Java `setValue(double)`.
    pub fn set_value_double(&self, input: f64) {
        self.values.lock().unwrap().set_value_double(input);
    }

    /// Java `setValue(double[])`.
    pub fn set_value_double_array(&self, input: Option<&[f64]>) {
        self.values.lock().unwrap().set_value_double_array(input);
    }

    /// Java `setValue(FortranInputString)`.
    pub fn set_value_fortran_input_string(&self, input: Option<&FortranInputString>) {
        self.values
            .lock()
            .unwrap()
            .set_value_fortran_input_string(input);
    }

    /// Java `setValue(int)`.
    pub fn set_value_int(&self, input: i32) {
        self.values.lock().unwrap().set_value_int(input);
    }
}

impl DirectiveInterface for Directive {
    /// Java `@Override setValue(boolean)`.
    fn set_value_boolean(&self, value: bool) {
        Directive::set_value_boolean(self, value);
    }

    /// Java `@Override setValue(String)`.
    fn set_value_string(&self, value: Option<&str>) {
        Directive::set_value_string(self, value);
    }

    /// Java `@Override resetValue()`.
    fn reset_value(&self) {
        Directive::reset_value(self);
    }
}

/// Java `toString()`.  `DirectiveValues` does not override `toString`, so Java prints
/// `Object.toString()` (`etomo.storage.DirectiveValues@<identity hash>`); the address of
/// the shared instance stands in for the identity hash.
impl std::fmt::Display for Directive {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[directiveName:{},batch:{},description:{},etomoColumn:{},template:{},valueType:{},values:etomo.storage.DirectiveValues@{:x}]",
            self.directive_name,
            self.batch,
            self.description.as_deref().unwrap_or("null"),
            match self.etomo_column {
                None => "null".to_string(),
                Some(etomo_column) => etomo_column.to_string(),
            },
            self.template,
            match self.value_type {
                None => "null".to_string(),
                Some(value_type) => value_type.to_string(),
            },
            &self.values as *const Mutex<DirectiveValues> as usize
        )
    }
}

/// Java private static final nested class `AxisLevelData`.  Unused in the source.
struct AxisLevelData {
    /// Java private field `inDirectiveFile = new boolean[DirectiveFileType.NUM]`.
    in_directive_file: Vec<bool>,
}

impl AxisLevelData {
    /// Java private `AxisLevelData()`.
    fn new() -> AxisLevelData {
        let mut in_directive_file = vec![false; directive_file_type::NUM as usize];
        for i in 0..in_directive_file.len() {
            in_directive_file[i] = false;
        }
        AxisLevelData { in_directive_file }
    }

    /// Java `toString()`.  Returns null when nothing is set.
    fn to_string_option(&self) -> Option<String> {
        let mut buffer = String::new();
        for i in 0..self.in_directive_file.len() {
            if self.in_directive_file[i] {
                let file_type = match DirectiveFileType::get_instance_from_index(i as i32) {
                    None => "null".to_string(),
                    Some(file_type) => file_type.to_string(),
                };
                buffer.push_str(&format!(
                    "{}in {} directive file",
                    if !buffer.is_empty() { ", " } else { "" },
                    file_type
                ));
            }
        }
        if !buffer.is_empty() {
            return Some(buffer);
        }
        None
    }
}

/// Java public static abstract nested class `Directive.Value`, closed over its four
/// subclasses.  The abstract methods dispatch to the subclass.
pub enum Value {
    /// `BooleanValue`.
    Boolean(BooleanValue),
    /// `NumericValue`.
    Numeric(NumericValue),
    /// `NumericPairValue`.
    NumericPair(NumericPairValue),
    /// `StringValue`.
    String(StringValue),
}

impl Value {
    /// Java abstract `set(boolean)`.
    pub(crate) fn set_boolean(&mut self, input: bool) {
        match self {
            Value::Boolean(value) => value.set_boolean(input),
            Value::Numeric(value) => value.set_boolean(input),
            Value::NumericPair(value) => value.set_boolean(input),
            Value::String(value) => value.set_boolean(input),
        }
    }

    /// Java abstract `set(ConstEtomoNumber)`.
    pub(crate) fn set_const_etomo_number(&mut self, input: Option<&ConstEtomoNumber>) {
        match self {
            Value::Boolean(value) => value.set_const_etomo_number(input),
            Value::Numeric(value) => value.set_const_etomo_number(input),
            Value::NumericPair(value) => value.set_const_etomo_number(input),
            Value::String(value) => value.set_const_etomo_number(input),
        }
    }

    /// Java abstract `set(ConstStringParameter)`.
    pub(crate) fn set_const_string_parameter(&mut self, input: Option<&dyn ConstStringParameter>) {
        match self {
            Value::Boolean(value) => value.set_const_string_parameter(input),
            Value::Numeric(value) => value.set_const_string_parameter(input),
            Value::NumericPair(value) => value.set_const_string_parameter(input),
            Value::String(value) => value.set_const_string_parameter(input),
        }
    }

    /// Java abstract `set(double)`.
    pub(crate) fn set_double(&mut self, input: f64) {
        match self {
            Value::Boolean(value) => value.set_double(input),
            Value::Numeric(value) => value.set_double(input),
            Value::NumericPair(value) => value.set_double(input),
            Value::String(value) => value.set_double(input),
        }
    }

    /// Java abstract `set(double[])`.
    pub(crate) fn set_double_array(&mut self, input: Option<&[f64]>) {
        match self {
            Value::Boolean(value) => value.set_double_array(input),
            Value::Numeric(value) => value.set_double_array(input),
            Value::NumericPair(value) => value.set_double_array(input),
            Value::String(value) => value.set_double_array(input),
        }
    }

    /// Java abstract `set(FortranInputString)`.
    pub(crate) fn set_fortran_input_string(&mut self, input: Option<&FortranInputString>) {
        match self {
            Value::Boolean(value) => value.set_fortran_input_string(input),
            Value::Numeric(value) => value.set_fortran_input_string(input),
            Value::NumericPair(value) => value.set_fortran_input_string(input),
            Value::String(value) => value.set_fortran_input_string(input),
        }
    }

    /// Java abstract `set(int)`.
    pub(crate) fn set_int(&mut self, input: i32) {
        match self {
            Value::Boolean(value) => value.set_int(input),
            Value::Numeric(value) => value.set_int(input),
            Value::NumericPair(value) => value.set_int(input),
            Value::String(value) => value.set_int(input),
        }
    }

    /// Java abstract `set(String)`.
    pub(crate) fn set_string(&mut self, input: Option<&str>) {
        match self {
            Value::Boolean(value) => value.set_string(input),
            Value::Numeric(value) => value.set_string(input),
            Value::NumericPair(value) => value.set_string(input),
            Value::String(value) => value.set_string(input),
        }
    }

    /// Java abstract `toBoolean()`.
    pub fn to_boolean(&self) -> bool {
        match self {
            Value::Boolean(value) => value.to_boolean(),
            Value::Numeric(value) => value.to_boolean(),
            Value::NumericPair(value) => value.to_boolean(),
            Value::String(value) => value.to_boolean(),
        }
    }

    /// Java abstract `isEmpty()`.
    pub fn is_empty(&self) -> bool {
        match self {
            Value::Boolean(value) => value.is_empty(),
            Value::Numeric(value) => value.is_empty(),
            Value::NumericPair(value) => value.is_empty(),
            Value::String(value) => value.is_empty(),
        }
    }

    /// Java package-private `setDebug(DebugLevel)`.
    pub(crate) fn set_debug(&self, input: DebugLevel) {
        let debug = match self {
            Value::Boolean(value) => &value.debug,
            Value::Numeric(value) => &value.debug,
            Value::NumericPair(value) => &value.debug,
            Value::String(value) => &value.debug,
        };
        *debug.lock().unwrap() = input;
    }
}

/// Java abstract `toString()`.
impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Boolean(value) => std::fmt::Display::fmt(value, f),
            Value::Numeric(value) => std::fmt::Display::fmt(value, f),
            Value::NumericPair(value) => std::fmt::Display::fmt(value, f),
            Value::String(value) => std::fmt::Display::fmt(value, f),
        }
    }
}

/// Java field initialiser `debug = EtomoDirector.INSTANCE.getArguments().getDebugLevel()`
/// of the abstract `Value`, run by every subclass constructor.
fn new_value_debug() -> Mutex<DebugLevel> {
    Mutex::new(etomo_director::ARGUMENTS.lock().unwrap().get_debug_level())
}

/// Java package-private static final nested class `ValueFactory`.
pub(crate) struct ValueFactory;

impl ValueFactory {
    /// Java static `getValue(DirectiveValueType)`.
    pub(crate) fn get_value(value_type: Option<DirectiveValueType>) -> Value {
        if value_type == Some(DirectiveValueType::Boolean) {
            return Value::Boolean(BooleanValue::new());
        } else if value_type == Some(DirectiveValueType::FloatingPoint)
            || value_type == Some(DirectiveValueType::Integer)
        {
            return Value::Numeric(NumericValue::new(value_type));
        } else if value_type == Some(DirectiveValueType::FloatingPointPair)
            || value_type == Some(DirectiveValueType::IntegerPair)
        {
            return Value::NumericPair(NumericPairValue::new(value_type));
        }
        Value::String(StringValue::new())
    }
}

/// Java package-private static final nested class `BooleanValue extends Value`.
pub struct BooleanValue {
    /// Java field `debug` inherited from `Value`.
    debug: Mutex<DebugLevel>,
    /// Java private field `value`, initialised to false.
    value: bool,
}

impl BooleanValue {
    /// Java private `BooleanValue()`.
    fn new() -> BooleanValue {
        BooleanValue {
            debug: new_value_debug(),
            value: false,
        }
    }

    /// Java `isEmpty()`.
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Java `equals(BooleanValue)`.
    pub fn equals_boolean_value(&self, input: Option<&BooleanValue>) -> bool {
        if self.debug.lock().unwrap().is_extra() {
            eprintln!(
                "equals:value:{},input:{}",
                self.value,
                match input {
                    None => "null".to_string(),
                    Some(input) => input.to_string(),
                }
            );
        }
        let input = match input {
            // the default is false
            None => return !self.value,
            Some(input) => input,
        };
        self.value == input.value
    }

    /// Java `set(boolean)`.
    pub(crate) fn set_boolean(&mut self, input: bool) {
        self.value = input;
    }

    /// Java `set(ConstEtomoNumber)`.
    pub(crate) fn set_const_etomo_number(&mut self, input: Option<&ConstEtomoNumber>) {
        match input {
            None => self.value = false,
            Some(input) => self.value = input.is(),
        }
    }

    /// Java `set(ConstStringParameter)`.
    pub(crate) fn set_const_string_parameter(&mut self, input: Option<&dyn ConstStringParameter>) {
        match input {
            None => self.value = false,
            Some(input) => self.set_string(Some(input.to_string().as_str())),
        }
    }

    /// Java `set(double)`.
    pub(crate) fn set_double(&mut self, input: f64) {
        if input == 1.0 {
            self.value = true;
        } else if input == 0.0 || input.is_nan() {
            // Directive.java:421 tests `input == EtomoNumber.DOUBLE_NULL_VALUE`, which is
            // NaN and never equal, so native leaves the value unchanged for a null
            // double.  Fixed in translation: a null (NaN) double sets false, as the
            // source intends.
            self.value = false;
        }
    }

    /// Java `set(double[])`.
    pub(crate) fn set_double_array(&mut self, input: Option<&[f64]>) {
        match input {
            Some(input) if !input.is_empty() => self.set_double(input[0]),
            _ => self.value = false,
        }
    }

    /// Java `set(FortranInputString)`.
    pub(crate) fn set_fortran_input_string(&mut self, input: Option<&FortranInputString>) {
        match input {
            Some(input) if input.size() != 0 => {
                if input.is_integer_type(0) {
                    self.set_int(input.get_int(0));
                } else {
                    self.set_double(input.get_double_index(0));
                }
            }
            _ => self.value = false,
        }
    }

    /// Java `set(int)`.
    pub(crate) fn set_int(&mut self, input: i32) {
        if input == 1 {
            self.value = true;
        } else if input == 0 || input == INTEGER_NULL_VALUE {
            self.value = false;
        }
    }

    /// Java `set(String)`.
    pub(crate) fn set_string(&mut self, input: Option<&str>) {
        match input {
            None => self.value = false,
            Some(input) => {
                let input = java_lang_string_trim(input);
                if input == "1" {
                    self.value = true;
                } else if input == "0" || input.is_empty() {
                    self.value = false;
                }
            }
        }
    }

    /// Java `toBoolean()`.
    pub fn to_boolean(&self) -> bool {
        self.value
    }
}

/// Java `toString()`.
impl std::fmt::Display for BooleanValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.value {
            return f.write_str("1");
        }
        f.write_str("0")
    }
}

/// Java package-private static final nested class `NumericValue extends Value`.
pub struct NumericValue {
    /// Java field `debug` inherited from `Value`.
    debug: Mutex<DebugLevel>,
    /// Java private final field `value`.
    value: EtomoNumber,
}

impl NumericValue {
    /// Java private `NumericValue(DirectiveValueType)`.
    fn new(value_type: Option<DirectiveValueType>) -> NumericValue {
        let debug = new_value_debug();
        let value = if value_type == Some(DirectiveValueType::FloatingPoint) {
            EtomoNumber::new_with_type(Some(Type::Double))
        } else {
            EtomoNumber::new()
        };
        NumericValue { debug, value }
    }

    /// Java `isEmpty()`.  (`value == null` cannot be true: the field is final and
    /// always constructed.)
    pub fn is_empty(&self) -> bool {
        self.value.is_null()
    }

    /// Java `equals(NumericValue)`.
    pub fn equals_numeric_value(&self, input: Option<&NumericValue>) -> bool {
        let input = match input {
            None => return self.value.is_null(),
            Some(input) => input,
        };
        self.value.equals_const_etomo_number(Some(&input.value))
    }

    /// Java `set(boolean)`.
    pub(crate) fn set_boolean(&mut self, input: bool) {
        self.value.set_boolean(input);
    }

    /// Java `set(ConstEtomoNumber)`.
    pub(crate) fn set_const_etomo_number(&mut self, input: Option<&ConstEtomoNumber>) {
        self.value.set_const_etomo_number(input);
    }

    /// Java `set(ConstStringParameter)`.
    pub(crate) fn set_const_string_parameter(&mut self, input: Option<&dyn ConstStringParameter>) {
        match input {
            None => {
                self.value.reset();
            }
            Some(input) => {
                self.value.set_string(Some(input.to_string().as_str()));
            }
        }
    }

    /// Java `set(double)`.
    pub(crate) fn set_double(&mut self, input: f64) {
        self.value.set_double(input);
    }

    /// Java `set(double[])`.
    pub(crate) fn set_double_array(&mut self, input: Option<&[f64]>) {
        match input {
            Some(input) if !input.is_empty() => {
                self.value.set_double(input[0]);
            }
            _ => {
                self.value.reset();
            }
        }
    }

    /// Java `set(FortranInputString)`.
    pub(crate) fn set_fortran_input_string(&mut self, input: Option<&FortranInputString>) {
        match input {
            Some(input) if input.size() != 0 => {
                if input.is_integer_type(0) {
                    self.set_int(input.get_int(0));
                } else {
                    self.set_double(input.get_double_index(0));
                }
            }
            _ => {
                self.value.reset();
            }
        }
    }

    /// Java `set(int)`.
    pub(crate) fn set_int(&mut self, input: i32) {
        self.value.set_int(input);
    }

    /// Java `set(String)`.
    pub(crate) fn set_string(&mut self, input: Option<&str>) {
        self.value.set_string(input);
    }

    /// Java `toBoolean()`.
    pub fn to_boolean(&self) -> bool {
        self.value.is()
    }
}

/// Java `toString()`.
impl std::fmt::Display for NumericValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.value, f)
    }
}

/// Java package-private static final nested class `NumericPairValue extends Value`.
pub struct NumericPairValue {
    /// Java field `debug` inherited from `Value`.
    debug: Mutex<DebugLevel>,
    /// Java private final field `value = new FortranInputString(2)`.
    value: FortranInputString,
}

impl NumericPairValue {
    /// Java private `NumericPairValue(DirectiveValueType)`.
    fn new(value_type: Option<DirectiveValueType>) -> NumericPairValue {
        let debug = new_value_debug();
        let mut value = FortranInputString::new(2);
        if value_type != Some(DirectiveValueType::FloatingPointPair) {
            value.set_integer_type(true);
        }
        NumericPairValue { debug, value }
    }

    /// Java `isEmpty()`.
    pub fn is_empty(&self) -> bool {
        self.value.is_null()
    }

    /// Java `equals(NumericPairValue)`.
    pub fn equals_numeric_pair_value(&self, input: Option<&NumericPairValue>) -> bool {
        let input = match input {
            None => return self.value.is_null(),
            Some(input) => input,
        };
        self.value.equals_fortran_input_string(Some(&input.value))
    }

    /// Java `set(boolean)`: `value.set(0, input ? 1 : 0)`, which resolves to
    /// `FortranInputString.set(int, double)`.
    pub(crate) fn set_boolean(&mut self, input: bool) {
        self.value
            .set_index_double(0, (if input { 1 } else { 0 }) as f64);
    }

    /// Java `set(ConstEtomoNumber)`.
    pub(crate) fn set_const_etomo_number(&mut self, input: Option<&ConstEtomoNumber>) {
        match input {
            Some(input) => self.value.set_index_const_etomo_number(0, input),
            // Directive.java:634 passes a null input to `FortranInputString.set(int,
            // ConstEtomoNumber)`, which dereferences it (NullPointerException).  Not
            // reachable through `DirectiveValues`, which filters null first.  Fixed in
            // translation: a null input resets the pair to its default, as
            // `set(ConstStringParameter)` does for null.
            None => self.value.set_default(),
        }
    }

    /// Java `set(ConstStringParameter)`.
    pub(crate) fn set_const_string_parameter(&mut self, input: Option<&dyn ConstStringParameter>) {
        match input {
            None => self.value.set_default(),
            Some(input) => {
                if let Err(e) = self
                    .value
                    .validate_and_set(Some(input.to_string().as_str()))
                {
                    e.print_stack_trace();
                }
            }
        }
    }

    /// Java `set(double)`.
    pub(crate) fn set_double(&mut self, input: f64) {
        self.value.set_index_double(0, input);
    }

    /// Java `set(double[])`.
    pub(crate) fn set_double_array(&mut self, input: Option<&[f64]>) {
        self.value.set_default();
        if let Some(input) = input {
            for i in 0..input.len() {
                // Directive.java:640-643 writes every element into the two-element
                // FortranInputString, so an array longer than 2 throws
                // ArrayIndexOutOfBoundsException.  Fixed in translation: elements past
                // the pair's size are ignored.
                if i as i32 >= self.value.size() {
                    break;
                }
                self.value.set_index_double(i as i32, input[i]);
            }
        }
    }

    /// Java `set(FortranInputString)`.
    pub(crate) fn set_fortran_input_string(&mut self, input: Option<&FortranInputString>) {
        self.value.set_default();
        if let Some(input) = input {
            for i in 0..input.size() {
                // Directive.java:650-653: same out-of-bounds write as `set(double[])`
                // for an input with more than two elements.  Fixed in translation:
                // elements past the pair's size are ignored.
                if i >= self.value.size() {
                    break;
                }
                self.value.set_index_string(
                    i,
                    Some(input.to_string_index_default_is_blank(i, true).as_str()),
                );
            }
        }
    }

    /// Java `set(int)`: resolves to `FortranInputString.set(int, double)`.
    pub(crate) fn set_int(&mut self, input: i32) {
        self.value.set_index_double(0, input as f64);
    }

    /// Java `set(String)`.
    pub(crate) fn set_string(&mut self, input: Option<&str>) {
        if let Err(e) = self.value.validate_and_set(input) {
            e.print_stack_trace();
        }
    }

    /// Java `toBoolean()`.
    pub fn to_boolean(&self) -> bool {
        if self.value.is_null_index(0) {
            return false;
        }
        let element = self.value.get_int(0);
        if element == 0 {
            return false;
        }
        true
    }
}

/// Java `toString()`.
impl std::fmt::Display for NumericPairValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.value.is_null() {
            return f.write_str("");
        }
        std::fmt::Display::fmt(&self.value, f)
    }
}

/// Java package-private static final nested class `StringValue extends Value`.
pub struct StringValue {
    /// Java field `debug` inherited from `Value`.
    debug: Mutex<DebugLevel>,
    /// Java private field `value`, initialised to null.
    value: Option<String>,
}

impl StringValue {
    /// Java private `StringValue()`.
    fn new() -> StringValue {
        StringValue {
            debug: new_value_debug(),
            value: None,
        }
    }

    /// Java `isEmpty()`.
    pub fn is_empty(&self) -> bool {
        match &self.value {
            None => true,
            Some(value) => java_lang_string_matches_whitespace(value),
        }
    }

    /// Java `equals(StringValue)`.
    pub fn equals_string_value(&self, input: Option<&StringValue>) -> bool {
        let input_value = match input {
            None => None,
            Some(input) => match &input.value {
                None => None,
                Some(value) if java_lang_string_matches_whitespace(value) => None,
                Some(value) => Some(value),
            },
        };
        let input_value = match input_value {
            None => {
                return match &self.value {
                    None => true,
                    Some(value) => java_lang_string_matches_whitespace(value),
                };
            }
            Some(input_value) => input_value,
        };
        // Directive.java:707 calls `value.trim()` on a possibly null `value` here
        // (NullPointerException when this value is unset and the input is set).  Fixed
        // in translation: an unset value is not equal to a non-empty input.
        match &self.value {
            None => false,
            Some(value) => java_lang_string_trim(value) == java_lang_string_trim(input_value),
        }
    }

    /// Java `set(boolean)`.
    pub(crate) fn set_boolean(&mut self, input: bool) {
        self.value = Some(if input { "1" } else { "0" }.to_string());
    }

    /// Java `set(ConstEtomoNumber)`.
    pub(crate) fn set_const_etomo_number(&mut self, input: Option<&ConstEtomoNumber>) {
        match input {
            Some(input) if !input.is_null() => self.value = Some(input.to_string()),
            _ => self.value = None,
        }
    }

    /// Java `set(ConstStringParameter)`.
    pub(crate) fn set_const_string_parameter(&mut self, input: Option<&dyn ConstStringParameter>) {
        match input {
            Some(input) if !input.is_empty() => self.value = Some(input.to_string()),
            _ => self.value = None,
        }
    }

    /// Java `set(double)`.
    pub(crate) fn set_double(&mut self, input: f64) {
        // Directive.java:737 tests `input == EtomoNumber.DOUBLE_NULL_VALUE` (NaN, never
        // equal), so native stores the string "NaN" for a null double.  Fixed in
        // translation: a null (NaN) double unsets the value, as the source intends.
        if input.is_nan() {
            self.value = None;
        } else {
            self.value = Some(java_lang_double_to_string(input));
        }
    }

    /// Java `set(double[])`.
    pub(crate) fn set_double_array(&mut self, input: Option<&[f64]>) {
        match input {
            None => self.value = None,
            Some(input) => {
                let mut buffer = String::new();
                for i in 0..input.len() {
                    // The element is compared with the integer null value, widened to
                    // double, as the source writes it.
                    buffer.push_str(&format!(
                        "{}{}",
                        if i > 0 { "," } else { "" },
                        if input[i] == INTEGER_NULL_VALUE as f64 {
                            " ".to_string()
                        } else {
                            java_lang_double_to_string(input[i])
                        }
                    ));
                }
                if !buffer.is_empty() {
                    self.value = Some(buffer);
                }
            }
        }
    }

    /// Java `set(FortranInputString)`.
    pub(crate) fn set_fortran_input_string(&mut self, input: Option<&FortranInputString>) {
        match input {
            Some(input) if !input.is_null() => self.value = Some(input.to_string()),
            _ => self.value = None,
        }
    }

    /// Java `set(int)`.
    pub(crate) fn set_int(&mut self, input: i32) {
        if input == INTEGER_NULL_VALUE {
            self.value = None;
        } else {
            self.value = Some(input.to_string());
        }
    }

    /// Java `set(String)`.
    pub(crate) fn set_string(&mut self, input: Option<&str>) {
        self.value = input.map(str::to_string);
    }

    /// Java `toBoolean()`.
    pub fn to_boolean(&self) -> bool {
        if self.value.is_some() && self.value.as_deref() == Some("1") {
            return true;
        }
        false
    }
}

/// Java `toString()`.
impl std::fmt::Display for StringValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.value {
            None => f.write_str(""),
            Some(value) => f.write_str(value),
        }
    }
}
