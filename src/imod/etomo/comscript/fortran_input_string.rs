//! `IMOD/Etomo/src/etomo/comscript/FortranInputString.java`.
//!
//! The FortranInputString class models the multiple parameter FORTRAN input formatting
//! present in the IMOD utilities.  It also allows for range and type validation of the
//! parameters.
//!
//! Implementation details: `Double.NaN` is used internaly to represent David's remaining
//! default input specifier `/`.  `Double.NEGATIVE_INFINITY` is used to represent an
//! uninitialized value.
//!
//! **Boundaries.**  `updateScriptParameter` (both overloads) and
//! `validateAndSet(ComScriptCommand)` need `etomo/comscript/ComScriptCommand.java` and
//! `etomo/comscript/ParamUtilities.java`; `regressionTest` (both overloads) needs
//! `EtomoDirector.INSTANCE`.  Each carries a `TODO(unit)` marker below.
#![allow(dead_code)]

use std::collections::BTreeMap;

use super::fortran_input_syntax_exception::FortranInputSyntaxException;
use crate::imod::etomo::r#type::const_etomo_number::{
    ConstEtomoNumber, java_lang_double_to_string, java_lang_double_value_of,
    java_lang_string_matches_whitespace,
};

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `DEFAULT_DIVIDER`.
pub(crate) const DEFAULT_DIVIDER: char = ',';
/// Java `VALUE_ENDS`.
const VALUE_ENDS: char = '/';
/// Java `VALUE_ENDS_STRING`.
const VALUE_ENDS_STRING: &str = "/";

/// The two throwables `validateAndSet(String, char)` can raise.  Java distinguishes them
/// by type - the unchecked `NumberFormatException` that `Double.valueOf` throws, which
/// `validateAndSet(String)` catches to retry with the alternate divider, and the checked
/// `FortranInputSyntaxException` that `rangeCheck` throws, which propagates.  Rust has
/// one error channel, so the distinction is carried in this enum.
#[derive(Clone, Debug)]
pub enum ValidateAndSetError {
    /// `java.lang.NumberFormatException`, carrying its message.
    NumberFormatException(String),
    /// `etomo.comscript.FortranInputSyntaxException`.
    FortranInputSyntaxException(FortranInputSyntaxException),
}

/// Java `FortranInputString`.
#[derive(Clone, Debug)]
pub struct FortranInputString {
    /// Java field `nParams`.
    n_params: i32,
    /// Java field `isInteger`.
    is_integer: Vec<bool>,
    /// Java field `minimum`.
    minimum: Vec<f64>,
    /// Java field `maximum`.
    maximum: Vec<f64>,
    /// Java field `value`.  A `Double[]`, so an element can be null as well as NaN.
    value: Vec<Option<f64>>,
    /// Java field `key`.
    key: Option<String>,
    /// Java field `divider`.
    divider: char,
    /// Java field `active`.
    active: bool,
    /// Java field `propertiesKey`.
    properties_key: Option<String>,
}

impl FortranInputString {
    /// Java `FortranInputString(int)`.  Create a FortranInputString with nParams
    /// parameters.
    pub fn new(n_params: i32) -> FortranInputString {
        let mut instance = FortranInputString::allocate();
        instance.initialize(n_params);
        instance
    }

    /// Java `FortranInputString(String, int)`.
    pub(crate) fn new_with_key(key: Option<&str>, n_params: i32) -> FortranInputString {
        let mut instance = FortranInputString::allocate();
        instance.key = key.map(|key| key.to_string());
        instance.initialize(n_params);
        instance
    }

    /// Java `FortranInputString(FortranInputString)`.  Copy constructor.
    pub(crate) fn new_from_instance(src: &FortranInputString) -> FortranInputString {
        let mut instance = FortranInputString::allocate();
        instance.n_params = src.n_params;
        instance.value = vec![None; src.n_params as usize];
        instance.minimum = vec![0.0; src.n_params as usize];
        instance.maximum = vec![0.0; src.n_params as usize];
        instance.is_integer = vec![false; src.n_params as usize];
        for i in 0..src.n_params as usize {
            instance.value[i] = src.value[i];
            instance.minimum[i] = src.minimum[i];
            instance.maximum[i] = src.maximum[i];
            instance.is_integer[i] = src.is_integer[i];
        }
        instance
    }

    /// Java `FortranInputString(double[])`.
    ///
    /// FortranInputString.java:174 reads `initialize(value.length)` - the *field*
    /// `value`, which is still null at that point - rather than `values.length`, so this
    /// constructor throws NullPointerException for any non-null argument.  The panic
    /// below is that throw.
    pub(crate) fn new_from_values(values: Option<&[f64]>) -> FortranInputString {
        let mut instance = FortranInputString::allocate();
        if values.is_none() {
            instance.initialize(0);
            return instance;
        }
        panic!("java.lang.NullPointerException");
    }

    /// The Java object allocation the four constructors share: every field at its
    /// declared initial value.  The array fields are the ones `initialize` builds, and
    /// are the Java `null` stand-in (an empty vector) here.
    fn allocate() -> FortranInputString {
        FortranInputString {
            n_params: 0,
            is_integer: Vec::new(),
            minimum: Vec::new(),
            maximum: Vec::new(),
            value: Vec::new(),
            key: None,
            divider: DEFAULT_DIVIDER,
            active: true,
            properties_key: None,
        }
    }

    /// Java `getInstance(Properties, String)`.  Returns an instance of
    /// FortranInputString loaded from props.
    pub fn get_instance(
        props: &BTreeMap<String, String>,
        group: &str,
    ) -> Result<FortranInputString, FortranInputSyntaxException> {
        let list = props.get(group).cloned();
        FortranInputString::get_instance_from_list(list.as_deref())
    }

    /// Java `getInstance(String)`.
    pub(crate) fn get_instance_from_list(
        list: Option<&str>,
    ) -> Result<FortranInputString, FortranInputSyntaxException> {
        let list = match list {
            None => return Ok(FortranInputString::new(0)),
            Some(list) if list.is_empty() => return Ok(FortranInputString::new(0)),
            Some(list) => list,
        };
        // java.lang.String.split(",") with the default limit drops trailing empty
        // strings.
        let mut parts: Vec<&str> = list.split(',').collect();
        while let Some(last) = parts.last() {
            if last.is_empty() {
                parts.pop();
            } else {
                break;
            }
        }
        let n_params = parts.len() as i32;
        let mut instance = FortranInputString::new(n_params);
        instance.validate_and_set(Some(list))?;
        Ok(instance)
    }

    /// Java `getInstance(double[])`.
    pub fn get_instance_from_doubles(list: Option<&[f64]>) -> FortranInputString {
        let list = match list {
            None => return FortranInputString::new(0),
            Some(list) => list,
        };
        let mut instance = FortranInputString::new(list.len() as i32);
        for i in 0..list.len() {
            instance.set_index_double(i as i32, list[i]);
        }
        instance
    }

    /// Java `load(Properties, String)`.  Loads from props, does not set nParams.
    pub fn load(&mut self, props: &BTreeMap<String, String>, group: Option<&str>) {
        let prepend = self.make_prepend(group);
        let list = match &prepend {
            None => None,
            Some(prepend) => props.get(prepend).cloned(),
        };
        if self.validate_and_set(list.as_deref()).is_err() {
            self.reset();
        }
    }

    /// Java `makePrepend`.
    fn make_prepend(&self, group: Option<&str>) -> Option<String> {
        let mut group = group.map(|group| group.to_string());
        if let Some(properties_key) = &self.properties_key {
            group = match &group {
                None => Some(properties_key.clone()),
                Some(g) if java_lang_string_matches_whitespace(g) => Some(properties_key.clone()),
                Some(g) => Some(format!("{}.{}", g, properties_key)),
            };
        }
        group
    }

    /// Java `store(Properties, String)`.
    pub fn store(&self, props: &mut BTreeMap<String, String>, group: Option<&str>) {
        // java.util.Properties.setProperty throws NullPointerException for a null key,
        // which is what makePrepend returns when both group and propertiesKey are null.
        let key = self
            .make_prepend(group)
            .expect("java.lang.NullPointerException");
        props.insert(key, self.to_string());
    }

    /// Java `reset`.
    pub(crate) fn reset(&mut self) {
        self.value = vec![None; self.n_params as usize];
        for i in 0..self.n_params as usize {
            self.value[i] = Some(f64::NEG_INFINITY);
        }
    }

    /// Java `setActive`.
    pub(crate) fn set_active(&mut self, active: bool) {
        self.active = active;
    }

    /// Java `isActive`.
    pub(crate) fn is_active(&self) -> bool {
        self.active
    }

    /// Java `initialize`.
    fn initialize(&mut self, n_params: i32) {
        self.n_params = n_params;
        self.minimum = vec![0.0; n_params as usize];
        self.maximum = vec![0.0; n_params as usize];
        self.is_integer = vec![false; n_params as usize];
        self.value = vec![None; n_params as usize];
        for i in 0..n_params as usize {
            self.minimum[i] = -1.0 * f64::MAX;
            self.maximum[i] = f64::MAX;
            self.is_integer[i] = false;
            self.value[i] = Some(f64::NEG_INFINITY);
        }
    }

    /// Java `setRange`.  Set the valid range for all of the values.
    pub(crate) fn set_range(&mut self, min: f64, max: f64) {
        for i in 0..self.minimum.len() {
            self.minimum[i] = min;
        }
        for i in 0..self.maximum.len() {
            self.maximum[i] = max;
        }
    }

    /// Java `setRangeByIndex`.
    pub(crate) fn set_range_by_index(&mut self, idx: i32, min: f64, max: f64) {
        if (idx as usize) < self.minimum.len() {
            self.minimum[idx as usize] = min;
        }
        if (idx as usize) < self.maximum.len() {
            self.maximum[idx as usize] = max;
        }
    }

    /// Java `size`.
    pub fn size(&self) -> i32 {
        self.n_params
    }

    // TODO(unit): needs etomo/comscript/ComScriptCommand.java and
    // etomo/comscript/ParamUtilities.java - both Java `updateScriptParameter` overloads
    // (FortranInputString.java:299-308) forward to
    // `ParamUtilities.updateScriptParameter(ComScriptCommand, key, this, ...)`, and
    // neither of those units has a Rust module.

    /// Java `setDivider`.
    pub(crate) fn set_divider(&mut self, divider: char) {
        self.divider = divider;
    }

    /// Java `resetDivider`.
    pub(crate) fn reset_divider(&mut self) {
        self.divider = DEFAULT_DIVIDER;
    }

    // TODO(unit): needs etomo/comscript/ComScriptCommand.java and
    // etomo/comscript/InvalidParameterException.java - Java
    // `validateAndSet(ComScriptCommand)` (FortranInputString.java:318-326) reads
    // `scriptCommand.hasKeyword(key)` and `scriptCommand.getValue(key)`.

    /// Java `validateAndSet(String, String)`.  Concatenates newValues1 and newValues2.
    /// Instance must be large enough to hold newValue1 and newValue2.  It is isn't, then
    /// newValue2 will not be added, and FortranInputSyntaxException will be thrown.
    /// Allow two different dividers (the default and a space) without calling
    /// setDivider.
    pub fn validate_and_set_two(
        &mut self,
        new_values1: Option<&str>,
        new_values2: Option<&str>,
    ) -> Result<(), FortranInputSyntaxException> {
        self.reset();
        let mut temp = FortranInputString::new_from_instance(self);
        self.validate_and_set(new_values1)?;
        temp.validate_and_set(new_values2)?;
        // Find the size of newValues1.
        let mut size1 = 0;
        for i in (0..self.value.len() as i32).rev() {
            if !self.is_null_index(i) && !self.is_empty_index(i) && !self.is_default_index(i) {
                size1 = i + 1;
                break;
            }
        }
        // Find the size of newValues2.
        let mut size2 = 0;
        for i in (0..temp.value.len() as i32).rev() {
            if !temp.is_null_index(i) && !temp.is_empty_index(i) && !temp.is_default_index(i) {
                size2 = i + 1;
                break;
            }
        }
        // Make sure there is enough space.
        if (self.value.len() as i32) < size1 + size2 {
            return Err(FortranInputSyntaxException::new(&format!(
                "Value array's length ({}) is too small to hold {} and {}",
                self.value.len(),
                match new_values1 {
                    None => "null",
                    Some(new_values1) => new_values1,
                },
                match new_values2 {
                    None => "null",
                    Some(new_values2) => new_values2,
                }
            )));
        }
        // concatenate newValues2
        let mut index2 = 0;
        for index1 in size1..size1 + size2 {
            self.value[index1 as usize] = temp.value[index2 as usize];
            index2 += 1;
        }
        Ok(())
    }

    /// Java `validateAndSet(String)`.  Allow two different dividers (the default and a
    /// space) without calling setDivider.
    pub fn validate_and_set(
        &mut self,
        new_values: Option<&str>,
    ) -> Result<(), FortranInputSyntaxException> {
        let divider = self.divider;
        match self.validate_and_set_with_divider(new_values, divider) {
            Ok(()) => Ok(()),
            Err(ValidateAndSetError::FortranInputSyntaxException(exception)) => Err(exception),
            Err(ValidateAndSetError::NumberFormatException(divider_exception)) => {
                // Didn't work, try an alternative divider
                // FortranInputString.java:393 assigns `origDivider` and never reads it.
                let _orig_divider = self.divider;
                let attempt = if self.divider == DEFAULT_DIVIDER {
                    self.validate_and_set_with_divider(new_values, ' ')
                } else if self.divider == ' ' {
                    self.validate_and_set_with_divider(new_values, DEFAULT_DIVIDER)
                } else {
                    // Unfamiliar divider, throw a FortranInputSyntaxException so the
                    // problem will be handled with a pop up.
                    return Err(FortranInputSyntaxException::new(&divider_exception));
                };
                match attempt {
                    Ok(()) => Ok(()),
                    Err(ValidateAndSetError::FortranInputSyntaxException(exception)) => {
                        Err(exception)
                    }
                    Err(ValidateAndSetError::NumberFormatException(message)) => {
                        // Error happened with the alternative divider, throw a
                        // FortranInputSyntaxException so the problem will be handled
                        // with a pop up.
                        Err(FortranInputSyntaxException::new(&message))
                    }
                }
            }
        }
    }

    /// Java `validateAndSet(String, char)`.  Set the String representation of the
    /// parameters and validate it against the specified rules.
    ///
    /// Java indexes a `String` by UTF-16 code unit; the char vector below indexes by
    /// code point, which is the same for every character below U+10000.
    pub(crate) fn validate_and_set_with_divider(
        &mut self,
        new_values: Option<&str>,
        divider_param: char,
    ) -> Result<(), ValidateAndSetError> {
        let new_values = match new_values {
            None => String::new(),
            Some(new_values) => new_values.to_string(),
        };
        // Handle a simple default string
        if new_values == VALUE_ENDS_STRING {
            for i in 0..self.value.len() {
                self.value[i] = Some(f64::NAN);
            }
            return Ok(());
        }
        // Walk through the newValues string parsing the values
        let mut temp_value: Vec<Option<f64>> = vec![None; self.value.len()];
        for i in 0..self.value.len() {
            temp_value[i] = Some(f64::NAN);
        }
        let chars: Vec<char> = new_values.chars().collect();
        // The Java `try` block; its `catch (ArrayIndexOutOfBoundsException e)` arm calls
        // `e.printStackTrace()` and then constructs - and discards - a
        // FortranInputSyntaxException, so an out-of-range index leaves tempValue partly
        // filled and still assigns it to `value`.
        let mut out_of_bounds = false;
        {
            let mut idx_value = 0usize; // current index of tempValue
            let mut idx_start = 0usize; // current index of newValues
            while idx_start < chars.len() {
                let idx_delim = chars[idx_start..]
                    .iter()
                    .position(|c| *c == divider_param)
                    .map(|offset| (idx_start + offset) as i32)
                    .unwrap_or(-1);
                if idx_delim != -1 {
                    let current_token: String =
                        chars[idx_start..idx_delim as usize].iter().collect();

                    // A default value
                    if current_token.is_empty() {
                        if idx_value >= temp_value.len() {
                            out_of_bounds = true;
                            break;
                        }
                        temp_value[idx_value] = Some(f64::NAN);
                    } else {
                        if idx_value >= temp_value.len() {
                            out_of_bounds = true;
                            break;
                        }
                        temp_value[idx_value] = Some(
                            java_lang_double_value_of(&current_token)
                                .map_err(ValidateAndSetError::NumberFormatException)?,
                        );
                        self.range_check(
                            temp_value[idx_value].unwrap(),
                            idx_value as i32,
                            &new_values,
                        )
                        .map_err(ValidateAndSetError::FortranInputSyntaxException)?;
                    }
                    idx_value += 1;
                    idx_start = idx_delim as usize + 1;
                }
                // This should be the last value
                else {
                    let current_token: String = chars[idx_start..].iter().collect();
                    if current_token.ends_with(VALUE_ENDS_STRING) {
                        if idx_value >= temp_value.len() {
                            out_of_bounds = true;
                            break;
                        }
                        temp_value[idx_value] = Some(
                            java_lang_double_value_of(
                                &current_token[..current_token.len() - VALUE_ENDS_STRING.len()],
                            )
                            .map_err(ValidateAndSetError::NumberFormatException)?,
                        );
                        self.range_check(
                            temp_value[idx_value].unwrap(),
                            idx_value as i32,
                            &new_values,
                        )
                        .map_err(ValidateAndSetError::FortranInputSyntaxException)?;
                        idx_value += 1;
                        while (idx_value as i32) < self.n_params {
                            if idx_value >= temp_value.len() {
                                out_of_bounds = true;
                                break;
                            }
                            temp_value[idx_value] = Some(f64::NAN);
                            idx_value += 1;
                        }
                    } else {
                        if idx_value >= temp_value.len() {
                            out_of_bounds = true;
                            break;
                        }
                        let tail: String = chars[idx_start..].iter().collect();
                        temp_value[idx_value] = Some(
                            java_lang_double_value_of(&tail)
                                .map_err(ValidateAndSetError::NumberFormatException)?,
                        );
                    }
                    break;
                }
            }
        }
        if out_of_bounds {
            // `e.printStackTrace()`.  A Java stack trace is a property of the JVM, not of
            // the program (CLAUDE.md's non-reproducible class), so only the exception's
            // first line is written to stderr where the source writes the whole trace.
            eprintln!("java.lang.ArrayIndexOutOfBoundsException");
        }
        self.value = temp_value;
        Ok(())
    }

    /// Java `getDouble(int)`.  Get a specific value as a double.
    pub fn get_double_index(&self, index: i32) -> f64 {
        self.value[index as usize].expect("java.lang.NullPointerException")
    }

    /// Java `getDouble()`.  Get input string as doubles.
    pub fn get_double(&self) -> Vec<f64> {
        let mut array = vec![0.0f64; self.n_params as usize];
        for i in 0..self.n_params as usize {
            array[i] = self.value[i].expect("java.lang.NullPointerException");
        }
        array
    }

    /// Java `getInt(int)`.  Get a specific value as an integer.
    pub fn get_int(&self, index: i32) -> i32 {
        // CHECK value here
        self.value[index as usize].expect("java.lang.NullPointerException") as i32
    }

    /// Java `set(int, double)`.  Set the value of given parameter.
    pub fn set_index_double(&mut self, index: i32, new_value: f64) {
        self.value[index as usize] = Some(new_value);
    }

    /// Java `set(int, String)`.  `Double.valueOf` throws the unchecked
    /// NumberFormatException, which this method does not declare or catch.
    pub fn set_index_string(&mut self, index: i32, new_value: Option<&str>) {
        let blank = match new_value {
            None => true,
            Some(new_value) => java_lang_string_matches_whitespace(new_value),
        };
        if blank {
            self.set_default_index(index);
        } else {
            self.value[index as usize] = Some(
                java_lang_double_value_of(new_value.unwrap()).unwrap_or_else(|message| {
                    panic!("java.lang.NumberFormatException: {}", message)
                }),
            );
        }
    }

    /// Java `set(int, ConstEtomoNumber)`.
    pub fn set_index_const_etomo_number(&mut self, index: i32, new_value: &ConstEtomoNumber) {
        self.value[index as usize] = Some(new_value.get_double());
    }

    /// Java `set(int, FortranInputString)`.
    pub(crate) fn set_index_fortran_input_string(
        &mut self,
        index: i32,
        new_value: &FortranInputString,
    ) {
        self.value[index as usize] = Some(new_value.get_double_index(index));
    }

    /// Java `set(FortranInputString)`.
    pub(crate) fn set_fortran_input_string(&mut self, new_value: &FortranInputString) {
        for i in 0..self.n_params {
            self.value[i as usize] = Some(new_value.get_double_index(i));
        }
    }

    /// Java `setPropertiesKey`.
    pub fn set_properties_key(&mut self, input: Option<&str>) {
        self.properties_key = input.map(|input| input.to_string());
    }

    /// Java `setDefault()`.
    pub fn set_default(&mut self) {
        for i in 0..self.n_params as usize {
            self.value[i] = Some(f64::NAN);
        }
    }

    /// Java `setDefault(int)`.
    pub(crate) fn set_default_index(&mut self, index: i32) {
        self.value[index as usize] = Some(f64::NAN);
    }

    /// Java `substring(int, int)`.  Return the string representation of the parameters.
    /// Does NOT return null.
    pub fn substring(&self, begin_index: i32, end_index: i32) -> String {
        let mut begin_index = begin_index;
        let mut end_index = end_index;
        if begin_index < 0 {
            begin_index = 0;
        }
        if end_index > self.value.len() as i32 {
            end_index = self.value.len() as i32;
        }
        if begin_index >= end_index {
            return String::new();
        }
        if begin_index == 0 && self.value[0].is_none() {
            return "Unitialized!".to_string();
        }

        // Walk backwards from the end of the array collecting any default value
        // into /

        let mut trailing_default = false;
        let mut idx_value = end_index - 1;
        while idx_value >= begin_index
            && self.value[idx_value as usize]
                .expect("java.lang.NullPointerException")
                .is_nan()
        {
            trailing_default = true;
            idx_value -= 1;
        }
        let mut buffer = String::new();
        if trailing_default {
            buffer.push(VALUE_ENDS);
        }
        while idx_value >= begin_index {
            let element = self.value[idx_value as usize].expect("java.lang.NullPointerException");
            if !element.is_nan() {
                if self.is_integer[idx_value as usize] {
                    buffer.insert_str(0, &(element as i32).to_string());
                } else {
                    buffer.insert_str(0, &java_lang_double_to_string(element));
                }
            }
            if idx_value != begin_index {
                buffer.insert(0, self.divider);
            }
            idx_value -= 1;
        }
        buffer
    }

    // TODO(unit): needs etomo/EtomoDirector.java - both Java `regressionTest` overloads
    // (FortranInputString.java:635-651) open with
    // `EtomoDirector.INSTANCE.getArguments().isTest()`, and the crate's etomo_director
    // module has no `INSTANCE` singleton with an `Arguments` attached.  Every call site
    // below carries an inline marker where the check would run; the value each caller
    // returns is unaffected, because the check only throws
    // `etomo/util/RegressionTestFailedError.java` when the two agree.

    /// Java `toStringForRegressionTest()`.  Return the string representation of the
    /// parameters.  Does NOT return null.  Deprecated 5/24/21: for unit test only,
    /// `toString()` must always return the same result as this function.
    fn to_string_for_regression_test(&self) -> String {
        if self.value[0].is_none() {
            return "Unitialized!".to_string();
        }

        // Walk backwards from the end of the array collecting any default value
        // into /

        let mut trailing_default = false;
        let mut idx_value = self.value.len() as i32 - 1;
        while idx_value >= 0
            && self.value[idx_value as usize]
                .expect("java.lang.NullPointerException")
                .is_nan()
        {
            trailing_default = true;
            idx_value -= 1;
        }
        let mut buffer = String::new();
        if trailing_default {
            buffer.push(VALUE_ENDS);
        }
        while idx_value >= 0 {
            let element = self.value[idx_value as usize].expect("java.lang.NullPointerException");
            if !element.is_nan() {
                if self.is_integer[idx_value as usize] {
                    buffer.insert_str(0, &(element as i32).to_string());
                } else {
                    buffer.insert_str(0, &java_lang_double_to_string(element));
                }
            }
            if idx_value != 0 {
                buffer.insert(0, self.divider);
            }
            idx_value -= 1;
        }
        buffer
    }

    /// Java `substring(int, int, boolean)`.
    pub fn substring_default_is_blank(
        &self,
        begin_index: i32,
        end_index: i32,
        default_is_blank: bool,
    ) -> String {
        let mut begin_index = begin_index;
        let mut end_index = end_index;
        if begin_index < 0 {
            begin_index = 0;
        }
        if end_index > self.value.len() as i32 {
            end_index = self.value.len() as i32;
        }
        if begin_index >= end_index {
            return String::new();
        }
        if !default_is_blank {
            return self.substring(begin_index, end_index);
        }
        if self.is_substring_default(begin_index, end_index) {
            return String::new();
        }
        if self.is_substring_null(begin_index, end_index) {
            return String::new();
        }
        if self.is_substring_empty(begin_index, end_index) {
            return String::new();
        }
        let string = self.substring(begin_index, end_index);
        if string == VALUE_ENDS_STRING {
            return String::new();
        }
        string
    }

    /// Java `toString(boolean)`.
    pub fn to_string_default_is_blank(&self, default_is_blank: bool) -> String {
        let substring =
            self.substring_default_is_blank(0, self.value.len() as i32, default_is_blank);
        // TODO(unit): needs etomo/EtomoDirector.java - Java calls
        // `regressionTest(toStringForRegressionTest(defaultIsBlank), substring)` here.
        let _ = self.to_string_for_regression_test_default_is_blank(default_is_blank);
        substring
    }

    /// Java `toStringForRegressionTest(boolean)`.  Deprecated 5/24/21: for test only.
    fn to_string_for_regression_test_default_is_blank(&self, default_is_blank: bool) -> String {
        // FortranInputString.java:706 declares `Error err = null;` and never reads it.
        if !default_is_blank {
            return self.to_string();
        }
        if self.is_default() {
            return String::new();
        }
        if self.is_null() {
            return String::new();
        }
        if self.is_empty() {
            return String::new();
        }
        let string = self.to_string();
        if string == VALUE_ENDS_STRING {
            return String::new();
        }
        string
    }

    /// Java `substring(int, int, boolean, boolean)`.
    pub fn substring_strip(
        &self,
        begin_index: i32,
        end_index: i32,
        default_is_blank: bool,
        strip_value_ends_char: bool,
    ) -> String {
        let mut begin_index = begin_index;
        let mut end_index = end_index;
        if begin_index < 0 {
            begin_index = 0;
        }
        if end_index > self.value.len() as i32 {
            end_index = self.value.len() as i32;
        }
        if begin_index >= end_index {
            return String::new();
        }
        FortranInputString::process_to_string(
            Some(&self.substring_default_is_blank(begin_index, end_index, default_is_blank)),
            strip_value_ends_char,
        )
    }

    /// Java `substring(int, boolean, boolean)`.
    pub fn substring_from(
        &self,
        begin_index: i32,
        default_is_blank: bool,
        strip_value_ends_char: bool,
    ) -> String {
        self.substring_strip(
            begin_index,
            self.value.len() as i32,
            default_is_blank,
            strip_value_ends_char,
        )
    }

    /// Java `toString(boolean, boolean)`.
    pub fn to_string_strip(&self, default_is_blank: bool, strip_value_ends_char: bool) -> String {
        self.substring_strip(
            0,
            self.value.len() as i32,
            default_is_blank,
            strip_value_ends_char,
        )
    }

    /// Java `processToString`.
    fn process_to_string(string: Option<&str>, strip_value_ends_char: bool) -> String {
        let string = match string {
            None => "",
            Some(string) => string,
        };
        let string_len = string.chars().count();
        if !strip_value_ends_char || string_len == 0 || !string.ends_with(VALUE_ENDS_STRING) {
            return string.to_string();
        }
        // Strip off the "/" from the end of the string.
        string[..string.len() - VALUE_ENDS_STRING.len()].to_string()
    }

    /// Java `toString(int, boolean)`.
    pub fn to_string_index_default_is_blank(&self, index: i32, default_is_blank: bool) -> String {
        if !default_is_blank {
            return self.to_string_index(index);
        }
        let string = self.to_string_index(index);
        if string == "Uninitialized!" {
            return String::new();
        }
        string
    }

    /// Java `toString(int)`.
    pub(crate) fn to_string_index(&self, index: i32) -> String {
        if !self.value_set(index) {
            return "Uninitialized!".to_string();
        }
        if self.is_default_index(index) {
            return String::new();
        }
        let element = self.value[index as usize].expect("java.lang.NullPointerException");
        if self.is_integer[index as usize] {
            (element as i32).to_string()
        } else {
            java_lang_double_to_string(element)
        }
    }

    /// Java `setIntegerType(boolean)`.  Sets all the elements to either integer or not
    /// integer.
    pub fn set_integer_type(&mut self, is_integer: bool) {
        for i in 0..self.is_integer.len() {
            self.is_integer[i] = is_integer;
        }
    }

    /// Java `setIntegerType(int, boolean)`.  Set the specified element to be an integer.
    pub(crate) fn set_integer_type_index(&mut self, index: i32, is_integer: bool) {
        self.is_integer[index as usize] = is_integer;
    }

    /// Java `setIntegerType(boolean[])`.  Set the integer state for all values.
    pub fn set_integer_type_array(&mut self, is_int_array: &[bool]) {
        for i in 0..is_int_array.len() {
            self.is_integer[i] = is_int_array[i];
        }
    }

    /// Java `valuesSet`.  Are any of the values unset.
    pub(crate) fn values_set(&self) -> bool {
        for i in 0..self.n_params as usize {
            if self.value[i]
                .expect("java.lang.NullPointerException")
                .is_infinite()
            {
                return false;
            }
        }
        true
    }

    /// Java `valueSet(int)`.
    pub(crate) fn value_set(&self, index: i32) -> bool {
        self.value[index as usize].is_some() && !self.value[index as usize].unwrap().is_infinite()
    }

    /// Java `isSubstringDefault`.  Are all of the values set to their defaults.
    pub(crate) fn is_substring_default(&self, begin_index: i32, end_index: i32) -> bool {
        let mut begin_index = begin_index;
        let mut end_index = end_index;
        if begin_index < 0 {
            begin_index = 0;
        }
        if end_index > self.value.len() as i32 {
            end_index = self.value.len() as i32;
        }
        for i in begin_index..end_index {
            if !self.value[i as usize]
                .expect("java.lang.NullPointerException")
                .is_nan()
            {
                return false;
            }
        }
        true
    }

    /// Java `isDefault()`.  Are all of the values set to their defaults.
    pub(crate) fn is_default(&self) -> bool {
        let is_default = self.is_substring_default(0, self.n_params);
        // TODO(unit): needs etomo/EtomoDirector.java - Java calls
        // `regressionTest(isDefaultForRegressionTest(), isDefault)` here.
        let _ = self.is_default_for_regression_test();
        is_default
    }

    /// Java `isDefaultForRegressionTest`.  Are all of the values set to their defaults.
    /// Deprecated 5/25/21: for test only.
    fn is_default_for_regression_test(&self) -> bool {
        for i in 0..self.n_params as usize {
            if !self.value[i]
                .expect("java.lang.NullPointerException")
                .is_nan()
            {
                return false;
            }
        }
        true
    }

    /// Java `isDefault(int)`.  Is value defaulted at index.
    pub fn is_default_index(&self, index: i32) -> bool {
        match self.value[index as usize] {
            None => panic!("java.lang.NullPointerException: value[{}]", index),
            Some(element) => element.is_nan(),
        }
    }

    /// Java `isEmpty(int)`.  Is value not initialized at index.
    pub fn is_empty_index(&self, index: i32) -> bool {
        match self.value[index as usize] {
            None => panic!("java.lang.NullPointerException: value[{}]", index),
            Some(element) => element.is_infinite(),
        }
    }

    /// Java `equals(FortranInputString)`.  Compares this.value to input.value.  The two
    /// arrays can be different sizes; the smaller array is equal if the larger contains
    /// nulls for the outsized portion of the array.
    pub fn equals_fortran_input_string(&self, input: Option<&FortranInputString>) -> bool {
        let input = match input {
            None => return self.is_null(),
            Some(input) => input,
        };
        let max = std::cmp::max(self.n_params, input.n_params);
        for i in 0..max {
            if !self.equals_index(i, input) {
                return false;
            }
        }
        true
    }

    /// Java `equals(int, FortranInputString)`.  Compares this.value[index] to
    /// input.value[index].  Handles an out-of-range index by comparing to null.  Null
    /// and `isNull()` are considered equal.
    pub fn equals_index(&self, index: i32, input: &FortranInputString) -> bool {
        let this_element = if index < self.n_params {
            self.value[index as usize]
        } else {
            None
        };
        let input_element = if index < input.n_params {
            input.value[index as usize]
        } else {
            None
        };
        // handle out of range and null values
        if (this_element.is_none() || self.is_null_index(index))
            && (input_element.is_none() || input.is_null_index(index))
        {
            return true;
        }
        if this_element.is_none()
            || self.is_null_index(index)
            || input_element.is_none()
            || input.is_null_index(index)
        {
            return false;
        }
        let this_element = this_element.unwrap();
        let input_element = input_element.unwrap();
        // compare
        if self.is_integer[index as usize] && input.is_integer[index as usize] {
            return this_element as i32 == input_element as i32;
        }
        if self.is_integer[index as usize] && !input.is_integer[index as usize] {
            return this_element as i32 as f64 == input_element;
        }
        if !self.is_integer[index as usize] && input.is_integer[index as usize] {
            return this_element == input_element as i32 as f64;
        }
        if !self.is_integer[index as usize] && !input.is_integer[index as usize] {
            return this_element == input_element;
        }
        false
    }

    /// Java `isNull(int)`.  Checks for both NaN and infinity.  Returns true for range
    /// indices.
    pub fn is_null_index(&self, index: i32) -> bool {
        if index >= self.n_params {
            return true;
        }
        if self.value[index as usize].is_none() {
            return true;
        }
        let v = self.value[index as usize].unwrap();
        v.is_infinite() || v.is_nan()
    }

    /// Java `isSubstringNull`.  Checks for both NaN and infinity.  Returns true unless
    /// an element of value does not contain infinity or NaN.
    pub fn is_substring_null(&self, begin_index: i32, end_index: i32) -> bool {
        let mut begin_index = begin_index;
        let mut end_index = end_index;
        if begin_index < 0 {
            begin_index = 0;
        }
        if end_index > self.n_params {
            end_index = self.n_params;
        }
        for i in begin_index..end_index {
            let element = self.value[i as usize].expect("java.lang.NullPointerException");
            if !element.is_infinite() && !element.is_nan() {
                return false;
            }
        }
        true
    }

    /// Java `isNull()`.  Checks for both NaN and infinity.  Returns true unless an
    /// element of value does not contain infinity or NaN.
    pub fn is_null(&self) -> bool {
        let is_null = self.is_substring_null(0, self.n_params);
        // TODO(unit): needs etomo/EtomoDirector.java - Java calls
        // `regressionTest(isNullForRegressionTest(), isNull)` here.
        let _ = self.is_null_for_regression_test();
        is_null
    }

    /// Java `isNullForRegressionTest`.  Deprecated 5/25/21: for test only.
    fn is_null_for_regression_test(&self) -> bool {
        for i in 0..self.n_params as usize {
            let element = self.value[i].expect("java.lang.NullPointerException");
            if !element.is_infinite() && !element.is_nan() {
                return false;
            }
        }
        true
    }

    /// Java `isSubstringEmpty`.  Returns false if any value is set.
    pub(crate) fn is_substring_empty(&self, begin_index: i32, end_index: i32) -> bool {
        let mut begin_index = begin_index;
        let mut end_index = end_index;
        if begin_index < 0 {
            begin_index = 0;
        }
        if end_index > self.value.len() as i32 {
            end_index = self.value.len() as i32;
        }
        for i in begin_index..end_index {
            if !self.value[i as usize]
                .expect("java.lang.NullPointerException")
                .is_infinite()
            {
                return false;
            }
        }
        true
    }

    /// Java `isEmpty()`.  Returns false if any value is set.
    pub(crate) fn is_empty(&self) -> bool {
        let empty = self.is_substring_empty(0, self.n_params);
        // TODO(unit): needs etomo/EtomoDirector.java - Java calls
        // `regressionTest(isEmptyForRegressionTest(), empty)` here.
        let _ = self.is_empty_for_regression_test();
        empty
    }

    /// Java `isEmptyForRegressionTest`.  Deprecated 5/25/21: for test only.
    pub(crate) fn is_empty_for_regression_test(&self) -> bool {
        for i in 0..self.n_params as usize {
            if !self.value[i]
                .expect("java.lang.NullPointerException")
                .is_infinite()
            {
                return false;
            }
        }
        true
    }

    /// Java `isIntegerType(int)`.  Is value treated as an integer at index.
    pub fn is_integer_type(&self, index: i32) -> bool {
        if self.value[index as usize].is_none() {
            panic!("java.lang.NullPointerException: value[{}]", index);
        }
        self.is_integer[index as usize]
    }

    /// Java `rangeCheck`.  Compare given value range specified for index.
    fn range_check(
        &self,
        value: f64,
        index: i32,
        new_values: &str,
    ) -> Result<(), FortranInputSyntaxException> {
        if value < self.minimum[index as usize] {
            let message = match &self.key {
                None => String::new(),
                Some(key) => format!(
                    "{}: Value below minimum.  Acceptable range: [{},{}] got {}",
                    key,
                    java_lang_double_to_string(self.minimum[index as usize]),
                    java_lang_double_to_string(self.maximum[index as usize]),
                    java_lang_double_to_string(value)
                ),
            };
            return Err(FortranInputSyntaxException::new_with_new_values(
                &message, new_values,
            ));
        }
        if value > self.maximum[index as usize] {
            let message = match &self.key {
                None => String::new(),
                Some(key) => format!(
                    "{}: Value above maximum.  Acceptable range: [{},{}] got {}",
                    key,
                    java_lang_double_to_string(self.minimum[index as usize]),
                    java_lang_double_to_string(self.maximum[index as usize]),
                    java_lang_double_to_string(value)
                ),
            };
            return Err(FortranInputSyntaxException::new_with_new_values(
                &message, new_values,
            ));
        }
        Ok(())
    }
}

/// Java `toString()`.  Return the string representation of the parameters.  Does NOT
/// return null.
impl std::fmt::Display for FortranInputString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let substring = self.substring(0, self.value.len() as i32);
        // TODO(unit): needs etomo/EtomoDirector.java - Java calls
        // `regressionTest(toStringForRegressionTest(), substring)` here.
        let _ = self.to_string_for_regression_test();
        f.write_str(&substring)
    }
}
