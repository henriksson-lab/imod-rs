//! `IMOD/Etomo/src/etomo/comscript/StringList.java`.
//!
//! Copyright: Copyright 2002 - 2023 by the Regents of the University of Colorado
//!
//! Organization: Dept. of MCD Biology, University of Colorado
//!
//! `mElements` is a `String[]` whose elements may be null (`StringList(int)` allocates
//! an array of nulls), so it is `Vec<Option<String>>`.  Every constructor assigns the
//! array, so the array itself is never null and is not an `Option`.

use super::bad_com_script_exception::BadComScriptException;
use super::com_script_command::ComScriptCommand;
use super::invalid_parameter_exception::InvalidParameterException;
use crate::imod::etomo::r#type::const_etomo_number::java_lang_string_matches_whitespace;
use crate::imod::etomo::util::utilities::java_lang_string_split;
use regex::Regex;

/// Java `StringList`.
#[derive(Clone, Debug)]
pub struct StringList {
    /// Java field `mElements`, initialised to null (every constructor then assigns it).
    m_elements: Vec<Option<String>>,
    /// Java field `mKey`.
    m_key: Option<String>,
    /// Java field `mSuccessiveEntriesAccumulate`, initialised to false.
    m_successive_entries_accumulate: bool,
    /// Java field `mConvertToSingleEntry`, initialised to false.
    m_convert_to_single_entry: bool,
}

impl Default for StringList {
    fn default() -> Self {
        StringList::new()
    }
}

impl StringList {
    /// Java `StringList()`.
    pub fn new() -> StringList {
        StringList {
            m_elements: Vec::new(),
            m_key: None,
            m_successive_entries_accumulate: false,
            m_convert_to_single_entry: false,
        }
    }

    /// Java `StringList(String key)`.
    pub fn new_with_key(key: Option<&str>) -> StringList {
        let mut string_list = StringList::new();
        string_list.m_elements = Vec::new();
        string_list.m_key = key.map(|key| key.to_string());
        string_list
    }

    /// Java `StringList(int nElements)`.
    pub fn new_with_n_elements(n_elements: i32) -> StringList {
        let mut string_list = StringList::new();
        string_list.m_elements = vec![None; n_elements as usize];
        string_list
    }

    /// Java `isEmpty`.  Empty if mElements can't or doesn't hold any elements.
    pub fn is_empty(&self) -> bool {
        if self.m_elements.is_empty() {
            return true;
        }
        for i in 0..self.m_elements.len() {
            if self.m_elements[i].is_some() {
                return false;
            }
        }
        true
    }

    /// Java `reset`.
    pub fn reset(&mut self) {
        self.m_elements = vec![None; self.m_elements.len()];
    }

    /// Java `setSuccessiveEntriesAccumulate`.  MSuccessiveEntriesAccumulate defaults to
    /// false.  Turning on mSuccessiveEntriesAccumulate means that all entries matching
    /// mKey will be loaded.
    pub fn set_successive_entries_accumulate(&mut self) {
        self.m_successive_entries_accumulate = true;
    }

    /// Java `setConvertToSingleEntry`.  MConvertToSingleEntry defaults to false.  Turning
    /// on mConvertToSingleEntry means that all entries matching mKey will be deleted and
    /// then mElement will be saved into one entry.  If mSuccessiveEntriesAccumulate is
    /// false, mConvertToSingleEntry will be ignored.
    pub fn set_convert_to_single_entry(&mut self) {
        self.m_convert_to_single_entry = true;
    }

    /// Java `StringList(StringList src)`, the copy constructor.
    pub fn new_from(src: &StringList) -> StringList {
        let mut string_list = StringList::new();
        string_list.m_elements = vec![None; src.get_n_elements() as usize];
        for i in 0..string_list.m_elements.len() {
            string_list.m_elements[i] = src.get(i as i32).map(|value| value.to_string());
        }
        string_list.m_key = src.m_key.clone();
        string_list
    }

    /// Java `StringList(String[] stringArray)`.
    pub fn new_from_array(string_array: Option<&[Option<String>]>) -> StringList {
        let mut string_list = StringList::new();
        string_list.parse_string_array(string_array);
        string_list
    }

    /// Java `setKey`.
    pub fn set_key(&mut self, key: Option<&str>) {
        self.m_key = key.map(|key| key.to_string());
    }

    /// Java `getKey`.
    pub fn get_key(&self) -> Option<&str> {
        self.m_key.as_deref()
    }

    /// Java `setNElements`.
    pub fn set_n_elements(&mut self, n_elements: i32) {
        // Allocate a new string array
        self.m_elements = vec![None; n_elements as usize];
    }

    /// Java `setAll(Iterator<String>)`.  Count and store the elements.
    pub fn set_all(&mut self, iterator: Option<&mut dyn Iterator<Item = Option<String>>>) {
        let iterator = match iterator {
            None => {
                self.m_elements = Vec::new();
                return;
            }
            Some(iterator) => iterator,
        };
        let mut temp: Vec<Option<String>> = Vec::new();
        for next in iterator {
            temp.push(next);
        }
        let size = temp.len();
        if size != self.m_elements.len() {
            self.m_elements = vec![None; size];
        }
        if size == 0 {
            return;
        }
        if size == 1 {
            // ArrayList.toArray may not work on a single element.
            self.m_elements[0] = temp[0].clone();
            return;
        }
        // `temp.toArray(mElements)`: mElements already has exactly `size` slots, so it
        // is filled in place and returned.
        for i in 0..size {
            self.m_elements[i] = temp[i].clone();
        }
    }

    /// Java `set(int, String)`.
    pub fn set(&mut self, index: i32, value: Option<&str>) {
        self.m_elements[index as usize] = value.map(|value| value.to_string());
    }

    /// Java `get(int)`.
    pub fn get(&self, index: i32) -> Option<&str> {
        self.m_elements[index as usize].as_deref()
    }

    /// Java `getNElements`.
    pub fn get_n_elements(&self) -> i32 {
        self.m_elements.len() as i32
    }

    /// Java `parseString(String)`.  Parse a space delimited string into the StringList.
    pub fn parse_string(&mut self, new_list: Option<&str>) {
        // If the string is only white space set the StringList to the null set
        let new_list = match new_list {
            None => {
                self.m_elements = Vec::new();
                return;
            }
            Some(new_list) if java_lang_string_matches_whitespace(new_list) => {
                self.m_elements = Vec::new();
                return;
            }
            Some(new_list) => new_list,
        };
        self.m_elements = java_lang_string_split(new_list, &Regex::new(" +").unwrap())
            .into_iter()
            .map(Some)
            .collect();
    }

    /// Java `parseString(String[])`.  Parse a space delimited string into the
    /// StringList.
    ///
    /// Fixed in translation: `newList[i].matches(...)` throws `NullPointerException`
    /// on a null element; a null element is skipped here as if it were white space.
    pub fn parse_string_array(&mut self, new_list: Option<&[Option<String>]>) {
        // If the string is only white space set the StringList to the null set
        let new_list = match new_list {
            None => {
                self.m_elements = Vec::new();
                return;
            }
            Some(new_list) if new_list.is_empty() => {
                self.m_elements = Vec::new();
                return;
            }
            Some(new_list) => new_list,
        };
        let mut element_array: Vec<String> = Vec::new();
        for i in 0..new_list.len() {
            let element = match &new_list[i] {
                None => continue,
                Some(element) => element,
            };
            if !java_lang_string_matches_whitespace(element) {
                let string_array = java_lang_string_split(element, &Regex::new(" +").unwrap());
                for string_index in 0..string_array.len() {
                    element_array.push(string_array[string_index].clone());
                }
            }
        }
        if element_array.is_empty() {
            self.m_elements = Vec::new();
        } else if element_array.len() == 1 {
            self.m_elements = vec![None; 1];
            self.m_elements[0] = Some(element_array[0].clone());
        } else {
            self.m_elements = element_array.into_iter().map(Some).collect();
        }
    }

    /// Java `parseString(StringList)`.
    pub fn parse_string_string_list(&mut self, string_list: Option<&StringList>) {
        if let Some(string_list) = string_list {
            let n_elements = string_list.m_elements.len() as i32;
            if n_elements < 0 {
                self.m_elements = Vec::new();
            } else {
                self.m_elements = vec![None; n_elements as usize];
                for i in 0..n_elements as usize {
                    self.m_elements[i] = string_list.m_elements[i].clone();
                }
            }
        }
    }

    /// Java `parse(ComScriptCommand)`.
    pub fn parse(
        &mut self,
        script_command: &ComScriptCommand,
    ) -> Result<(), InvalidParameterException> {
        if self.m_successive_entries_accumulate {
            let values = script_command.get_values(self.m_key.as_deref());
            self.parse_string_array(Some(&values));
        } else {
            let value = script_command.get_value(self.m_key.as_deref())?;
            self.parse_string(value.as_deref());
        }
        Ok(())
    }

    /// Java `deleteAllFromComScript`.
    pub fn delete_all_from_com_script(&self, script_command: &mut ComScriptCommand) {
        script_command.delete_key_all(self.m_key.as_deref());
    }

    /// Java `updateComScript`.
    ///
    /// A null or blank `mKey` throws the unchecked `IllegalArgumentException`, which is a
    /// programming error (every caller sets a constant key); it stays a panic here.
    pub fn update_com_script(
        &self,
        script_command: &mut ComScriptCommand,
    ) -> Result<(), BadComScriptException> {
        let key = match &self.m_key {
            None => panic!("java.lang.IllegalArgumentException"),
            Some(key) if java_lang_string_matches_whitespace(key) => {
                panic!("java.lang.IllegalArgumentException")
            }
            Some(key) => key.clone(),
        };
        if self.m_successive_entries_accumulate && self.m_convert_to_single_entry {
            // Remove all entries before saving
            while script_command.delete_key(Some(&key)) {}
        }
        if self.m_successive_entries_accumulate
            && !self.m_convert_to_single_entry
            && self.get_n_elements() > 1
        {
            // output separate entries
            script_command.set_values(Some(&key), &self.m_elements);
        } else if self.get_n_elements() > 0 {
            // output one entry
            script_command.set_value(Some(&key), Some(&self.to_string()));
        } else {
            script_command.delete_key(Some(&key));
        }
        Ok(())
    }
}

/// Java `toString`.
impl std::fmt::Display for StringList {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.m_elements.is_empty() {
            return f.write_str("");
        }
        let mut buffer = String::new();
        for i in 0..self.m_elements.len() {
            buffer.push_str(self.m_elements[i].as_deref().unwrap_or("null"));
            buffer.push(' ');
        }
        f.write_str(&buffer)
    }
}
