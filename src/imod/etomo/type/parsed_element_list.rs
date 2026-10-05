//! `IMOD/Etomo/src/etomo/type/ParsedElementList.java`.
//!
//! Expandable array that allows sparse population: the elements sit in a map keyed by
//! their index, and `size` is one past the largest index set.
//!
//! **Ownership.**  Java stores element references; the translation owns its elements
//! as `Box<dyn ParsedElement>` (see `ParsedElement::clone_element`).  Java's
//! `HashMap<Integer, ParsedElement>` is a `BTreeMap`: its `toString` lists small
//! non-negative integer keys in ascending order, which is the order a `BTreeMap`
//! iterates in.

use std::collections::BTreeMap;

use super::const_etomo_number::Type;
use super::etomo_number::EtomoNumber;
use super::parsed_element::ParsedElement;
use super::parsed_element_type::{self, ParsedElementType};
use super::parsed_number::ParsedNumber;
use super::parsed_quoted_string::ParsedQuotedString;

/// Java `public static final String rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `public final class ParsedElementList`.
pub struct ParsedElementList {
    /// Java private final `type`.
    r#type: &'static ParsedElementType,
    /// Java private final `map = new HashMap()`.
    map: BTreeMap<i32, Box<dyn ParsedElement>>,
    /// Java private final `etomoNumberType`.
    etomo_number_type: Option<Type>,
    /// Java private final `descr`.
    descr: Option<String>,
    /// Java private `size`, initially 0.
    size: i32,
    /// Java private `debug`, initially false.
    debug: bool,
    /// Java private `defaultValue`, initially null.
    default_value: Option<EtomoNumber>,
    /// Java private `minSize`, initially -1 (its setter is commented out in the
    /// source).
    #[allow(dead_code)]
    min_size: i32,
}

impl ParsedElementList {
    /// Java package-private `ParsedElementList(ParsedElementType, EtomoNumber.Type,
    /// boolean, EtomoNumber, String)`.
    pub fn new(
        r#type: &'static ParsedElementType,
        etomo_number_type: Option<Type>,
        debug: bool,
        default_value: Option<&EtomoNumber>,
        descr: Option<&str>,
    ) -> ParsedElementList {
        ParsedElementList {
            r#type,
            map: BTreeMap::new(),
            etomo_number_type,
            descr: descr.map(str::to_owned),
            size: 0,
            debug,
            default_value: default_value.cloned(),
            min_size: -1,
        }
    }

    /// Java `size()`.  Don't call this function with the class since it uses minSize
    /// instead of giving the real size and would cause new elements to be added after
    /// minSize in an empty list.  (The minSize branch is commented out in the source.)
    pub fn size(&self) -> i32 {
        self.size
    }

    /// Java synchronized package-private `add(ParsedElement)`.  Add an element.  The
    /// key is the current size.  Size is incremented by one.
    pub fn add(&mut self, element: Option<Box<dyn ParsedElement>>) {
        let Some(element) = element else {
            return;
        };
        let key = self.size;
        self.size += 1;
        self.map.insert(key, element);
    }

    /// Java synchronized `get(int)`.
    pub fn get(&self, index: i32) -> Option<&dyn ParsedElement> {
        self.map.get(&index).map(|element| element.as_ref())
    }

    /// Mutable form of Java `get(int)`: Java mutates the element `get` returns.
    pub fn get_mut(&mut self, index: i32) -> Option<&mut Box<dyn ParsedElement>> {
        self.map.get_mut(&index)
    }

    /// Java package-private `setDebug(boolean)`.
    pub fn set_debug(&mut self, input: bool) {
        self.debug = input;
    }

    /// Java package-private `setDefault(EtomoNumber)`.
    pub fn set_default(&mut self, input: Option<&EtomoNumber>) {
        self.default_value = input.cloned();
    }

    /// Java synchronized package-private `set(int, ParsedElement)`.  Add or change an
    /// element.  Puts the element in the map using index as the key.  If the location
    /// is larger then the current size, the size will be increased.
    pub fn set(&mut self, index: i32, element: Box<dyn ParsedElement>) {
        self.map.insert(index, element);
        // if index is equal to size, this is the same a calling add()
        if index == self.size {
            self.size += 1;
        }
        // if index is ahead of size
        else if index > self.size {
            self.size = index + 1;
        }
    }

    /// Java synchronized package-private `clear()`.
    pub fn clear(&mut self) {
        self.map.clear();
        self.size = 0;
    }

    /// Java package-private `isEmpty()`.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Java synchronized package-private `addEmptyElement()`.
    pub fn add_empty_element(&mut self) -> &dyn ParsedElement {
        let index = self.size;
        self.set_empty_element(index)
    }

    /// Java package-private `setEmptyElement(int)`.
    pub fn set_empty_element(&mut self, index: i32) -> &dyn ParsedElement {
        let element: Box<dyn ParsedElement> =
            if std::ptr::eq(self.r#type, &parsed_element_type::STRING) {
                Box::new(ParsedQuotedString::get_instance_boolean(
                    self.debug,
                    self.descr.as_deref(),
                ))
            } else {
                Box::new(ParsedNumber::get_instance(
                    self.r#type,
                    self.etomo_number_type,
                    self.debug,
                    self.default_value.as_ref(),
                    self.descr.as_deref(),
                ))
            };
        self.set(index, element);
        self.map.get(&index).unwrap().as_ref()
    }

    /// Java `remove(int)`.
    pub fn remove(&mut self, index: i32) -> Option<Box<dyn ParsedElement>> {
        self.map.remove(&index)
    }
}

impl Clone for ParsedElementList {
    fn clone(&self) -> ParsedElementList {
        ParsedElementList {
            r#type: self.r#type,
            map: self
                .map
                .iter()
                .map(|(key, element)| (*key, element.clone_element()))
                .collect(),
            etomo_number_type: self.etomo_number_type,
            descr: self.descr.clone(),
            size: self.size,
            debug: self.debug,
            default_value: self.default_value.clone(),
            min_size: self.min_size,
        }
    }
}

/// Java `toString()`: `"[map:" + map + "]"`, where `HashMap.toString()` is
/// `{key=value, ...}`.
impl std::fmt::Display for ParsedElementList {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("[map:{")?;
        let mut first = true;
        for (key, element) in &self.map {
            if !first {
                f.write_str(", ")?;
            }
            first = false;
            write!(f, "{key}={element}")?;
        }
        f.write_str("}]")
    }
}
