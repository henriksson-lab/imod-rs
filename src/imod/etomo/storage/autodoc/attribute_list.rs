//! `IMOD/Etomo/src/etomo/storage/autodoc/AttributeList.java`.
//!
//! Description: The Autodoc and each Section contain an AttributeList which holds
//! the attributes with the first attribute in each of their name/value pairs.
//! If the name of a name/value pair contains multiple attributes, then each
//! attribute, except for the last one, will also contain an AttributeList called
//! children.  So Autodocs and sections each contain a tree structure of Attributes.
//!
//! **Limitation.**  `print` and `paramString` walk `map`, a `java.util.HashMap`, whose
//! iteration order is a property of the JVM's bucket layout.  The Rust `HashMap` here
//! keys and looks up identically but does not reproduce that order, so the *order* of
//! the lines `print` emits and of the entries `paramString` renders is not matchable
//! when a level holds more than one attribute.  Every other member is order-independent
//! (`list` carries the source's insertion order and drives `iterator` and
//! `getFirstAttribute`).
#![allow(dead_code)]

use super::attribute::{self, Attribute};
use super::name_value_pair::NameValuePair;
use super::read_only_attribute::ReadOnlyAttribute;
use super::read_only_attribute_iterator::ReadOnlyAttributeIterator;
use super::read_only_attribute_list::ReadOnlyAttributeList;
use super::write_only_attribute_list::WriteOnlyAttributeList;
use crate::imod::etomo::ui::swing::token::{self, Token};
use std::collections::HashMap;

/// Java package-private final `AttributeList implements ReadOnlyAttributeList`.
pub struct AttributeList {
    /// Java field `parent`.
    parent: *mut dyn WriteOnlyAttributeList,
    /// Java field `map`.
    ///
    /// map contains Attributes.  Each Attribute instance stands for 0 or more
    /// occurrences of a name in this attribute list.  Attributes are never removed,
    /// but the number of occurrences they contain can be reduced to 0.
    map: HashMap<String, *mut Attribute>,
    /// Java field `list`.
    list: Vec<*mut Attribute>,
}

impl AttributeList {
    /// Java package-private `AttributeList(WriteOnlyAttributeList)`.
    pub fn new(parent: *mut dyn WriteOnlyAttributeList) -> AttributeList {
        AttributeList {
            parent,
            map: HashMap::new(),
            list: Vec::new(),
        }
    }

    /// Java package-private `addAttribute(Token, int)`.
    ///
    /// Adds a new attribute, or increments an existing one.
    ///
    /// # Safety
    /// `name` must point to a live `Token` link list.
    pub unsafe fn add_attribute(&mut self, name: *mut Token, line_num: i32) -> *mut Attribute {
        let key = unsafe { attribute::get_key_of_token(name) };
        let mut attribute: *mut Attribute = match &key {
            None => std::ptr::null_mut(),
            Some(key) => *self.map.get(key).unwrap_or(&std::ptr::null_mut()),
        };
        if attribute.is_null() {
            attribute = unsafe { Attribute::new(self.parent, name, line_num) };
            self.map.insert(
                match key {
                    // Java's `HashMap` accepts a null key; no source path reaches it,
                    // since `name` is never null here.
                    None => panic!("java.lang.NullPointerException"),
                    Some(key) => key,
                },
                attribute,
            );
            self.list.push(attribute);
        } else {
            // add another occurrence of this attribute
            unsafe { (*attribute).add() };
        }
        attribute
    }

    /// Java package-private `addAttribute(int, String[], int, String, NameValuePair)`.
    ///
    /// Adds a multi-level name/value pair.  Adds all attributes and the name/value pair
    /// with the last attribute.
    ///
    /// # Safety
    /// `name_value_pair` must point to a live `NameValuePair`.
    pub unsafe fn add_attribute_multi(
        &mut self,
        mut index: i32,
        name: Option<&[Option<String>]>,
        line_num: i32,
        value: Option<&str>,
        name_value_pair: *mut NameValuePair,
    ) {
        if name.is_none() || index >= name.unwrap().len() as i32 {
            return;
        }
        let name = name.unwrap();
        // Find the next valid attribute name
        if index < 0 {
            index = 0;
        }
        let mut cur_index: i32 = -1;
        let mut next_index: i32 = -1;
        for i in index..name.len() as i32 {
            // Java's `matches("\\s*")` is true when the whole string is zero or more
            // characters of `\s`: space, tab, newline, vertical tab, form feed, return.
            if name[i as usize].is_some()
                && !name[i as usize]
                    .as_ref()
                    .unwrap()
                    .chars()
                    .all(|c| matches!(c, ' ' | '\t' | '\n' | '\u{0B}' | '\u{0C}' | '\r'))
            {
                if cur_index == -1 {
                    cur_index = i;
                } else if next_index == -1 {
                    next_index = i;
                    break;
                }
            }
        }
        if cur_index == -1 {
            // no valid attribute names left
            return;
        }
        // Get or create the attribute
        let name_token = Box::into_raw(Box::new(Token::new()));
        unsafe {
            (*name_token).set_type_and_string(
                token::Type::Anything,
                name[cur_index as usize].as_ref().unwrap(),
            )
        };
        let attribute = unsafe { self.add_attribute(name_token, line_num) };
        // Add each attribute to the name/value pair
        unsafe { (*name_value_pair).add_attribute(attribute) };
        if next_index == -1 {
            // Added the value when the last attribute is added
            let value_token = Box::into_raw(Box::new(Token::new()));
            unsafe {
                (*value_token).set_type_and_string(
                    token::Type::Anything,
                    match value {
                        None => panic!("java.lang.NullPointerException"),
                        Some(value) => value,
                    },
                )
            };
            unsafe { (*name_value_pair).add_value(value_token) };
        } else {
            // Add the next attributes
            unsafe {
                (*attribute).add_attribute_multi(
                    next_index,
                    Some(name),
                    line_num,
                    value,
                    name_value_pair,
                )
            };
        }
    }

    /// Java package-private `getAttribute(String)`.
    ///
    /// # Safety
    /// Every attribute in the map must be live.
    pub unsafe fn get_attribute(&self, name: Option<&str>) -> *mut Attribute {
        // The source's `map == null` guard cannot fail: the field is final and assigned
        // at its declaration.
        let attribute: *mut Attribute = match attribute::get_key_of_string(name) {
            None => std::ptr::null_mut(),
            Some(key) => *self.map.get(&key).unwrap_or(&std::ptr::null_mut()),
        };
        if attribute.is_null() || !unsafe { (*attribute).exists() } {
            // if !exists(), then all occurrences of this attribute have been removed
            return std::ptr::null_mut();
        }
        attribute
    }

    /// Java package-private `getFirstAttribute()`.
    ///
    /// Returns the first attribute which exists.
    ///
    /// # Safety
    /// Every attribute in the list must be live.
    pub unsafe fn get_first_attribute(&self) -> *mut Attribute {
        for i in 0..self.list.len() {
            let attribute = self.list[i];
            if unsafe { (*attribute).exists() } {
                return attribute;
            }
        }
        std::ptr::null_mut()
    }

    /// Java package-private `print(int)`.  See the module header for the `HashMap`
    /// iteration-order limitation.
    ///
    /// # Safety
    /// Every attribute in the map must be live.
    pub unsafe fn print(&self, level: i32) {
        // The source's `map != null` guard cannot fail; the field is final.
        let mut attribute: *mut Attribute;
        let collection = self.map.values();
        let mut iterator = collection.into_iter();
        // The source calls `hasNext()` once before the loop, which the loop repeats.
        let mut current = iterator.next();
        if current.is_some() {
            while let Some(entry) = current {
                // This bypasses the exists() check, but Attribute.print() also checks
                // exists()
                attribute = *entry;
                unsafe { (*attribute).print(level) };
                current = iterator.next();
            }
        }
    }

    /// Java `toString()`.
    ///
    /// # Safety
    /// Every attribute in the map must be live.
    pub unsafe fn to_string(&self) -> String {
        format!("etomo.storage.autodoc.AttributeList[{}]", unsafe {
            self.param_string()
        })
    }

    /// Java package-private `paramString()`.  `"map=" + map` renders through
    /// `java.util.AbstractMap.toString`; see the module header for its ordering.
    ///
    /// # Safety
    /// Every attribute in the map must be live.
    pub unsafe fn param_string(&self) -> String {
        let mut buffer = String::from("map={");
        let mut first = true;
        for (key, attribute) in &self.map {
            if !first {
                buffer.push_str(", ");
            }
            first = false;
            buffer.push_str(key);
            buffer.push('=');
            buffer.push_str(&unsafe { (**attribute).to_string() });
        }
        buffer.push('}');
        buffer
    }
}

impl ReadOnlyAttributeList for AttributeList {
    /// Java `iterator()`.  Returns an iterator for the list of attributes.
    fn iterator(&self) -> ReadOnlyAttributeIterator<'_, *mut Attribute> {
        ReadOnlyAttributeIterator::new(&self.list)
    }
}

#[cfg(test)]
mod tests {
    use crate::imod::etomo::storage::autodoc::autodoc::Autodoc;
    use crate::imod::etomo::storage::autodoc::read_only_autodoc::ReadOnlyAutodoc;
    use crate::imod::etomo::storage::autodoc::writable_autodoc::WritableAutodoc;

    /// Attributes are never removed from the list; `addAttribute` increments an
    /// existing entry's occurrence count and `getAttribute` hides one whose count has
    /// fallen below one.
    #[test]
    fn occurrences_gate_lookup_rather_than_deleting_the_attribute() {
        unsafe {
            let autodoc = Autodoc::new(Some("occurrences"), std::ptr::null_mut());
            (*autodoc).add_name_value_pair_attribute(Some("dup"), Some("1"));
            (*autodoc).add_name_value_pair_attribute(Some("dup"), Some("2"));
            assert!(!(*autodoc).get_attribute(Some("dup")).is_null());
            (*autodoc).remove_name_value_pair(Some("dup"));
            assert!(!(*autodoc).get_attribute(Some("dup")).is_null());
            (*autodoc).remove_name_value_pair(Some("dup"));
            assert!((*autodoc).get_attribute(Some("dup")).is_null());
        }
    }
}
