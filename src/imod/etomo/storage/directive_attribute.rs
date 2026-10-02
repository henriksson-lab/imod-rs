//! `IMOD/Etomo/src/etomo/storage/DirectiveAttribute.java` (with its nested
//! `AttributeMatch`, `Key` and `Match`).
//!
//! Returns a primary directive attribute match and also a secondary one.  Contains a
//! table of instances that are likely to be accessed again.  Only the TABLE is thread
//! safe.
//!
//! **Shape.**  The autodoc translation keeps its attributes per thread and hands out
//! raw pointers (see storage/directive_file.rs), so `AttributeMatch.attribute` is a
//! `*mut Attribute` (null for Java null) and `TABLE`, which holds `AttributeMatch`es,
//! is per thread as well.
//!
//! `Key` holds a `DirectiveFile` only to read its `toString()` and
//! `getDirectiveFileType()`, both fixed once the file has been loaded
//! (`DirectiveFile.setFile` is private and runs only from its factory methods), so the
//! key records those two values when it is built instead of a reference to the file.
//! That keeps an `AttributeMatch` free of borrows (`'static`), as its callers need.
//!
//! **`TABLE` is never read back.**  `Key` overrides `hashCode` but not `equals`, so
//! `Hashtable.containsKey(key)` compares identity, and `getMatch` always asks with a key
//! it has just built: the lookup never hits.  The only other operations are
//! `put(key, this)` and `remove(key)` with the match's own key object.  That is odd but
//! harmless (every match is loaded afresh, which is the correct answer), so it is kept:
//! a `Key` carries an identity number that stands for Java's object identity, and the
//! table is keyed on it.
#![allow(dead_code)]

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::imod::etomo::storage::autodoc::attribute::Attribute;
use crate::imod::etomo::storage::autodoc::read_only_attribute::ReadOnlyAttribute;
use crate::imod::etomo::storage::directive_def::DirectiveDef;
use crate::imod::etomo::storage::directive_file::DirectiveFile;
use crate::imod::etomo::storage::directive_type::DirectiveType;
use crate::imod::etomo::storage::directive_value::DirectiveValue;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::{
    java_lang_string_matches_whitespace, java_lang_string_trim,
};
use crate::imod::etomo::r#type::directive_file_type::DirectiveFileType;

/// Java `rcsid`.
pub const RCSID: &str = "$Id:$";

/// Java `TRUE_VALUE`.
pub const TRUE_VALUE: &str = "1";
/// Java `FALSE_VALUE`.
pub const FALSE_VALUE: &str = "0";

thread_local! {
    /// Java private static final `TABLE`.  Table of directives that are likely to be
    /// accessed again.  Keyed on `Key` identity; see the module header.
    static TABLE: RefCell<HashMap<u64, AttributeMatch>> = RefCell::new(HashMap::new());
}

/// Source of `Key` identities (Java object identity).
static NEXT_KEY_IDENTITY: AtomicU64 = AtomicU64::new(0);

/// Java package-private static final `INSTANCE`.  The class has no instance state, so
/// the singleton is the unit struct.
pub(crate) static INSTANCE: DirectiveAttribute = DirectiveAttribute;

/// Java `DirectiveAttribute`.
pub struct DirectiveAttribute;

/// Java package-private static `getMatch(Match, DirectiveFile, ReadOnlyAttribute,
/// DirectiveDef, AxisID)`.  Returns the AttributeMatch that matches the match parameter.
/// The AttributeMatch is either found in the TABLE or is constructed.
///
/// # Safety (of `parent_attribute`)
/// The attribute must be live, as the autodoc that owns it is for the rest of the thread.
pub(crate) fn get_match(
    r#match: Match,
    directive_file: &DirectiveFile,
    parent_attribute: Option<&Attribute>,
    directive_def: DirectiveDef,
    axis_id: Option<AxisID>,
) -> Option<AttributeMatch> {
    let key;
    if r#match == Match::Primary {
        key = Key::new(directive_file, directive_def, axis_id, Match::Primary);
    } else if r#match == Match::Secondary && directive_def.has_secondary_match(axis_id) {
        key = Key::new(directive_file, directive_def, axis_id, Match::Secondary);
    } else {
        return None;
    }
    let found = TABLE.with(|table| table.borrow().get(&key.identity).cloned());
    if let Some(found) = found {
        return Some(found);
    }
    let mut attribute_match = AttributeMatch::new(r#match, key, directive_def);
    attribute_match.load_attribute(parent_attribute, axis_id);
    Some(attribute_match)
}

/// Java package-private static `toBoolean(String)`.  Translates the value of the
/// attribute into a boolean.  A null value returns false.  A "0" value returns false.  A
/// "1" returns true.  Everything else returns true.
pub(crate) fn to_boolean(value: Option<&str>) -> bool {
    let value = match value {
        None => return false,
        Some(value) => value,
    };
    let value = java_lang_string_trim(value);
    if value == FALSE_VALUE {
        return false;
    }
    true
}

/// Java package-private static final nested class `AttributeMatch implements
/// DirectiveValue`.
#[derive(Clone)]
pub struct AttributeMatch {
    /// Java private final field `match`.
    r#match: Match,
    /// Java private final field `key`.
    key: Key,
    /// Java private final field `directiveDef`.  Every construction passes the
    /// non-null `DirectiveDef` given to `getMatch` (Java's `getMatch` dereferences it
    /// for a secondary match anyway).
    directive_def: DirectiveDef,
    /// Java private field `loaded`, initialised to false.
    loaded: bool,
    /// Java private field `attribute`, initialised to null.
    attribute: *mut Attribute,
}

impl AttributeMatch {
    /// Java private `AttributeMatch(Match, Key, DirectiveDef)`.
    fn new(r#match: Match, key: Key, directive_def: DirectiveDef) -> AttributeMatch {
        AttributeMatch {
            r#match,
            key,
            directive_def,
            loaded: false,
            attribute: std::ptr::null_mut(),
        }
    }

    /// Java private `loadAttribute(ReadOnlyAttribute, AxisID)`.
    fn load_attribute(&mut self, parent_attribute: Option<&Attribute>, axis_id: Option<AxisID>) {
        if self.loaded {
            return;
        }
        self.loaded = true;
        // `directiveDef == null` cannot happen here; see the field.
        let parent_attribute = match parent_attribute {
            None => return,
            Some(parent_attribute) => parent_attribute,
        };
        self.loaded = true;
        let r#type = self.directive_def.get_directive_type();
        let name = self.directive_def.get_name_for_match(self.r#match, axis_id);
        // copyarg and setupset
        if r#type == DirectiveType::COPY_ARG || r#type == DirectiveType::SETUP_SET {
            self.attribute = unsafe {
                ReadOnlyAttribute::get_attribute_by_name(parent_attribute, name.as_deref())
            };
        }
        // runtime
        else if r#type == DirectiveType::RUN_TIME {
            self.attribute = self.find_attribute(
                parent_attribute as *const Attribute as *mut Attribute,
                self.directive_def.get_module().as_deref(),
                self.directive_def
                    .get_axis(self.r#match, axis_id)
                    .as_deref(),
                name.as_deref(),
            );
        }
        // comparam
        else if r#type == DirectiveType::COM_PARAM {
            self.attribute = self.find_attribute(
                parent_attribute as *const Attribute as *mut Attribute,
                self.directive_def
                    .get_comfile(self.r#match, axis_id)
                    .as_deref(),
                self.directive_def.get_command().as_deref(),
                name.as_deref(),
            );
        }
        if !self.attribute.is_null() {
            // This instance may be reused - save it
            let saved = self.clone();
            TABLE.with(|table| table.borrow_mut().insert(self.key.identity, saved));
        }
    }

    /// Java private `findAttribute(ReadOnlyAttribute, String, String, String)`.  Get the
    /// three-level down descendant of an attribute.
    fn find_attribute(
        &self,
        mut attribute: *mut Attribute,
        name1: Option<&str>,
        name2: Option<&str>,
        name3: Option<&str>,
    ) -> *mut Attribute {
        if attribute.is_null() || name1.is_none() {
            return std::ptr::null_mut();
        }
        attribute = unsafe { ReadOnlyAttribute::get_attribute_by_name(&*attribute, name1) };
        if attribute.is_null() || name2.is_none() {
            return std::ptr::null_mut();
        }
        attribute = unsafe { ReadOnlyAttribute::get_attribute_by_name(&*attribute, name2) };
        if attribute.is_null() || name3.is_none() {
            return std::ptr::null_mut();
        }
        unsafe { ReadOnlyAttribute::get_attribute_by_name(&*attribute, name3) }
    }

    /// Java package-private `isEmpty()`.  Returns true if no attribute has been loaded.
    pub(crate) fn is_empty(&self) -> bool {
        self.attribute.is_null()
    }

    /// Java package-private `isValue()`.  Returns the boolean value of the primary or
    /// secondary attribute.
    pub(crate) fn is_value(&self) -> bool {
        let mut value = self.get_value();
        if self.directive_def.is_boolean() {
            match &value {
                None => {
                    eprintln!(
                        "Warning: {} is boolean and its value should not be null",
                        self.directive_def
                    );
                }
                Some(v) => {
                    let trimmed = java_lang_string_trim(v).to_string();
                    if trimmed != FALSE_VALUE && trimmed != TRUE_VALUE {
                        eprintln!(
                            "Warning: {} is boolean and its value is invalid: {}",
                            self.directive_def, trimmed
                        );
                    }
                    value = Some(trimmed);
                }
            }
        } else {
            eprintln!("Warning: {} is not a boolean", self.directive_def);
        }
        to_boolean(value.as_deref())
    }
}

impl DirectiveValue for AttributeMatch {
    /// Java `getValue()`.  Returns the value of the attribute.
    fn get_value(&self) -> Option<String> {
        if self.attribute.is_null() {
            return None;
        }
        // The value has been retrieved, so no further use for this instance.
        TABLE.with(|table| table.borrow_mut().remove(&self.key.identity));
        unsafe { ReadOnlyAttribute::get_value(&*self.attribute) }
    }

    /// Java `isBatch()`.
    fn is_batch(&self) -> bool {
        let file_type = match self.key.directive_file_type {
            None => return false,
            Some(file_type) => file_type,
        };
        file_type.is_batch()
    }

    /// Java `isOverride()`.  Returns true if the directive is not boolean and the
    /// attribute value is null.
    fn is_override(&self) -> bool {
        if self.directive_def.is_boolean() || self.attribute.is_null() {
            return false;
        }
        let value = unsafe { ReadOnlyAttribute::get_value(&*self.attribute) };
        match value {
            None => true,
            Some(value) => java_lang_string_matches_whitespace(&value),
        }
    }
}

/// Java `AttributeMatch.toString()`.
impl std::fmt::Display for AttributeMatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[match:{},\nkey:{},\ndirectiveDef:{},loaded:{},attribute:{},value:{}]",
            self.r#match,
            self.key,
            self.directive_def,
            self.loaded,
            if self.attribute.is_null() {
                "null".to_string()
            } else {
                unsafe { ReadOnlyAttribute::to_string(&*self.attribute) }
            },
            self.get_value().unwrap_or_else(|| "null".to_string())
        )
    }
}

/// Java private static final nested class `Key`.
#[derive(Clone)]
struct Key {
    /// Java object identity; see the module header.
    identity: u64,
    /// `directiveFile.toString()`, recorded when the key is built.
    directive_file_string: String,
    /// `directiveFile.getDirectiveFileType()`, recorded when the key is built.
    directive_file_type: Option<DirectiveFileType>,
    /// Java private final field `directiveDef`.
    directive_def: DirectiveDef,
    /// Java private final field `axisID`.
    axis_id: Option<AxisID>,
    /// Java private final field `match`.
    r#match: Match,
}

impl Key {
    /// Java private `Key(DirectiveFile, DirectiveDef, AxisID, Match)`.
    fn new(
        directive_file: &DirectiveFile,
        directive_def: DirectiveDef,
        axis_id: Option<AxisID>,
        r#match: Match,
    ) -> Key {
        Key {
            identity: NEXT_KEY_IDENTITY.fetch_add(1, Ordering::Relaxed),
            directive_file_string: directive_file.to_string(),
            directive_file_type: directive_file.get_directive_file_type(),
            directive_def,
            axis_id,
            r#match,
        }
    }
}

/// Java `Key.toString()`.  (`hashCode` hashes this string; it is not used, see the
/// module header.)
impl std::fmt::Display for Key {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}{}{}{}",
            self.directive_file_string,
            self.directive_def,
            match self.axis_id {
                Some(axis_id) => format!(",{}", axis_id),
                None => String::new(),
            },
            self.r#match
        )
    }
}

/// Java package-private static final nested class `Match`.  How well an attribute
/// matches an axis.  Primary is the first match, secondary is the second match.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Match {
    /// Java `PRIMARY`, tag "primary".
    Primary,
    /// Java `SECONDARY`, tag "secondary".
    Secondary,
}

/// Java `Match.toString()`.
impl std::fmt::Display for Match {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Match::Primary => "primary",
            Match::Secondary => "secondary",
        })
    }
}
