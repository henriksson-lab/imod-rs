//! `IMOD/Etomo/src/etomo/storage/DirectiveDescrChoiceList.java`.

use regex::Regex;

use crate::imod::etomo::r#type::option::Option as TypeOption;
use crate::imod::etomo::util::utilities::java_lang_string_split;

/// Java final `DirectiveDescrChoiceList`.
#[derive(Clone, Debug)]
pub struct DirectiveDescrChoiceList {
    /// Java private final field `choiceList`.
    choice_list: Vec<TypeOption>,
}

impl DirectiveDescrChoiceList {
    /// Java private `DirectiveDescrChoiceList(String[])`.  A Java `String[]` from
    /// `String.split` holds no null element, so the elements are non-null here.
    fn new(array: &[String]) -> DirectiveDescrChoiceList {
        let mut choice_list = Vec::new();
        let pattern = Regex::new(r"\s*\:\s*").unwrap();
        for i in 0..array.len() {
            if !array[i].is_empty() {
                let pair = java_lang_string_split(&array[i], &pattern);
                // `pair != null && pair.length >= 0` is always true.
                choice_list.push(TypeOption::new_string_array(Some(&pair)));
            }
        }
        DirectiveDescrChoiceList { choice_list }
    }

    /// Java package-private static `getInstance(String)`.
    pub(crate) fn get_instance(choices: Option<&str>) -> Option<DirectiveDescrChoiceList> {
        let choices = choices?;
        let array = java_lang_string_split(choices, &Regex::new(r"\s*;\s*").unwrap());
        if array.is_empty() {
            return None;
        }
        Some(DirectiveDescrChoiceList::new(&array))
    }

    /// Java package-private `isEmpty()`.
    pub(crate) fn is_empty(&self) -> bool {
        self.choice_list.is_empty()
    }

    /// Java `iterator()`.
    pub fn iterator(&self) -> std::slice::Iter<'_, TypeOption> {
        self.choice_list.iter()
    }

    /// Java `size()`.
    pub fn size(&self) -> i32 {
        self.choice_list.len() as i32
    }

    /// Java `getDescr(int)`.
    pub fn get_descr(&self, index: i32) -> Option<String> {
        if index >= 0 && index < self.choice_list.len() as i32 {
            return self.choice_list[index as usize].get_display_string();
        }
        None
    }

    /// Java `getValue(int)`.
    pub fn get_value(&self, index: i32) -> Option<String> {
        if index >= 0 && index < self.choice_list.len() as i32 {
            return self.choice_list[index as usize]
                .get_value()
                .map(str::to_string);
        }
        None
    }
}

/// Java `toString()`: `"[" + choiceList + "]"`, with `AbstractCollection.toString`'s
/// `[a, b]` form inside.
impl std::fmt::Display for DirectiveDescrChoiceList {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("[[")?;
        for i in 0..self.choice_list.len() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{}", self.choice_list[i])?;
        }
        f.write_str("]]")
    }
}
