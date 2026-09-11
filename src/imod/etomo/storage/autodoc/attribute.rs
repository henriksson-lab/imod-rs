//! `IMOD/Etomo/src/etomo/storage/autodoc/Attribute.java`.
//!
//! **Ownership.**  An `Attribute` is aliased by its `AttributeList`'s map and list, by
//! every `NameValuePair` whose name contains it, and by its children's `parent`.  As in
//! `statement.rs`, the allocation is a leaked `Box` and the aliases are raw pointers.
#![allow(dead_code)]

use crate::imod::etomo::storage::log_file;

use super::attribute_list::AttributeList;
use super::autodoc;
use super::name_value_pair::NameValuePair;
use super::read_only_attribute::ReadOnlyAttribute;
use super::writable_attribute::WritableAttribute;
use super::write_only_attribute_list::WriteOnlyAttributeList;
use crate::imod::etomo::ui::swing::token::{self, Token};

/// Java package-private final `Attribute extends WriteOnlyAttributeList implements
/// WritableAttribute`.
pub struct Attribute {
    /// Java field `parent`.
    parent: *mut dyn WriteOnlyAttributeList,
    /// Java field `name`.
    name: *mut Token,
    /// Java field `key`.
    key: String,
    /// Java field `lineNum`.
    line_num: i32,
    /// Java field `occurrences`, initialised to 1.
    ///
    /// An attribute can occur more then once.  When occurrences is less then 1,
    /// this attribute no longer exists.
    occurrences: i32,
    /// Java field `nameValuePairList`, initialised to null.
    ///
    /// nameValuePairList will be instantiated if the attribute is the last attribute
    /// in the name of at least one name/value pair.  The value of the attribute is
    /// retrieved through the name/value pair.
    name_value_pair_list: Option<Vec<*mut NameValuePair>>,
    /// Java field `children`, initialised to null.
    ///
    /// children will be instantiated if the attribute is the not the last attribute
    /// in the name of at least one name/value pair.
    children: *mut AttributeList,
}

/// Java static `getKey(Token name)`.
///
/// # Safety
/// `name` must be null or point to a live `Token` link list.
pub unsafe fn get_key_of_token(name: *mut Token) -> Option<String> {
    if name.is_null() {
        return None;
    }
    Some(unsafe { (*name).get_key() })
}

/// Java static `getKey(String name)`.
pub fn get_key_of_string(name: Option<&str>) -> Option<String> {
    name?;
    Some(token::convert_to_key(name.unwrap()))
}

impl Attribute {
    /// Java package-private `Attribute(WriteOnlyAttributeList, Token, int)`.  The
    /// allocation is leaked; see the module header.
    ///
    /// # Safety
    /// `parent` must point to a live attribute-list owner and `name` to a live `Token`
    /// link list - Java's `name.getKey()` throws a `NullPointerException` on a null
    /// name.
    pub unsafe fn new(
        parent: *mut dyn WriteOnlyAttributeList,
        name: *mut Token,
        line_num: i32,
    ) -> *mut Attribute {
        Box::into_raw(Box::new(Attribute {
            parent,
            name,
            key: unsafe { (*name).get_key() },
            line_num,
            occurrences: 1,
            name_value_pair_list: None,
            children: std::ptr::null_mut(),
        }))
    }

    /// Java package-private `getKey()`.
    pub fn get_key(&self) -> String {
        self.key.clone()
    }

    /// Java package-private `isBase()`.
    ///
    /// First attribute in name value pair - parent is a section or an autodoc.
    ///
    /// # Safety
    /// `parent` must point to a live attribute-list owner.
    pub unsafe fn is_base(&self) -> bool {
        !unsafe { (*self.parent).is_attribute() }
    }

    /// Java package-private `add()`.
    pub fn add(&mut self) {
        self.occurrences += 1;
    }

    /// Java package-private `remove()`.
    pub fn remove(&mut self) {
        self.occurrences -= 1;
    }

    /// Java package-private `exists()`.
    pub fn exists(&self) -> bool {
        self.occurrences >= 1
    }

    /// Java package-private `addAttribute(int, String[], int, String, NameValuePair)`.
    ///
    /// Added a multi-attribute name/value pair to the children member variable.
    ///
    /// # Safety
    /// `name_value_pair` must point to a live `NameValuePair`.
    pub unsafe fn add_attribute_multi(
        &mut self,
        index: i32,
        name: Option<&[Option<String>]>,
        line_num: i32,
        value: Option<&str>,
        name_value_pair: *mut NameValuePair,
    ) {
        if self.children.is_null() {
            let this: *mut Attribute = self;
            self.children = Box::into_raw(Box::new(AttributeList::new(this)));
        }
        unsafe {
            (*self.children).add_attribute_multi(index, name, line_num, value, name_value_pair)
        };
    }

    /// Java `addNameValuePair(NameValuePair)`, declared `synchronized`.
    ///
    /// # Safety
    /// `name_value_pair` must point to a live `NameValuePair`.
    pub unsafe fn add_name_value_pair(&mut self, name_value_pair: *mut NameValuePair) {
        if self.name_value_pair_list.is_none() {
            // complete construction before assigning to keep the unsynchronized
            // functions from seeing a partially constructed instance.
            self.name_value_pair_list = Some(Vec::new());
        }
        self.name_value_pair_list
            .as_mut()
            .unwrap()
            .push(name_value_pair);
    }

    /// Java package-private `getFirstAttribute()`.
    ///
    /// Gets the first attribute in children with an onameValuePairccurrences value of
    /// at least one.
    ///
    /// # Safety
    /// The children must be live.
    pub unsafe fn get_first_attribute(&self) -> *mut Attribute {
        if self.children.is_null() {
            return std::ptr::null_mut();
        }
        unsafe { (*self.children).get_first_attribute() }
    }

    /// Java package-private `write(LogFile.Handle, LogFile.WriterId)`.
    ///
    /// # Safety
    /// The name token must be live.
    pub unsafe fn write(
        &self,
        file: &std::sync::Arc<log_file::Handle>,
        writer_id: &log_file::WriterId,
    ) -> Result<(), log_file::LogFileError> {
        if !self.exists() {
            return Ok(());
        }
        unsafe { (*self.name).write(file, writer_id) }
    }

    /// Java package-private `print(int)`.
    ///
    /// # Safety
    /// The name token, the name/value pair list and the children must be live.
    pub unsafe fn print(&self, mut level: i32) {
        if self.exists() {
            autodoc::print_indent(level);
            level += 1;
            print!("{}", unsafe { (*self.name).get_values() });
            match &self.name_value_pair_list {
                None => println!("."),
                Some(name_value_pair_list) => {
                    let len = name_value_pair_list.len();
                    if len > 0 {
                        for i in 0..len {
                            if i > 0 {
                                autodoc::print_indent(level);
                            }
                            let name_value_pair = name_value_pair_list[i];
                            if !name_value_pair.is_null() {
                                print!(" = ");
                                let value = unsafe { (*name_value_pair).get_token_value() };
                                if value.is_null() {
                                    println!("null");
                                } else {
                                    println!("{}", unsafe { (*value).get_values() });
                                }
                            }
                        }
                    } else {
                        println!();
                    }
                }
            }
        }
        if !self.children.is_null() {
            unsafe { (*self.children).print(level) };
        }
    }

    /// Java package-private `getParent()`.
    pub fn get_parent(&self) -> *mut dyn WriteOnlyAttributeList {
        self.parent
    }

    /// Java package-private `getNameToken()`.
    pub fn get_name_token(&self) -> *mut Token {
        self.name
    }

    /// Java package-private `removeNameValuePair(NameValuePair)`.
    ///
    /// # Safety
    /// `pair` must point to a live `NameValuePair`; Java throws a
    /// `NullPointerException` when the list was never created.
    pub unsafe fn remove_name_value_pair(&mut self, pair: *mut NameValuePair) {
        let list = match self.name_value_pair_list.as_mut() {
            None => panic!("java.lang.NullPointerException"),
            Some(list) => list,
        };
        if let Some(index) = list.iter().position(|entry| std::ptr::eq(*entry, pair)) {
            list.remove(index);
        }
    }

    /// Java package-private `getNameValuePair()`.
    ///
    /// Gets the last nameValuePair in the nameValuePairList.
    pub fn get_name_value_pair(&self) -> *mut NameValuePair {
        if let Some(list) = &self.name_value_pair_list {
            if !list.is_empty() {
                return list[list.len() - 1];
            }
        }
        std::ptr::null_mut()
    }

    /// Java package-private `getValueToken()`.
    ///
    /// # Safety
    /// The name/value pair list must be live.
    pub unsafe fn get_value_token(&self) -> *mut Token {
        let name_value_pair = self.get_name_value_pair();
        if !name_value_pair.is_null() {
            return unsafe { (*name_value_pair).get_token_value() };
        }
        std::ptr::null_mut()
    }

    /// Java `hashCode()`, which is `key.hashCode()` - `java.lang.String.hashCode`, an
    /// `int` accumulation of `31 * h + charAt(i)` over UTF-16 code units.
    pub fn hash_code(&self) -> i32 {
        let mut h: i32 = 0;
        for unit in self.key.encode_utf16() {
            h = h.wrapping_mul(31).wrapping_add(unit as i32);
        }
        h
    }
}

impl WriteOnlyAttributeList for Attribute {
    /// Java `addAttribute(Token, int)`.
    unsafe fn add_attribute(&mut self, name: *mut Token, line_num: i32) -> *mut Attribute {
        if self.children.is_null() {
            let this: *mut Attribute = self;
            self.children = Box::into_raw(Box::new(AttributeList::new(this)));
        }
        unsafe { (*self.children).add_attribute(name, line_num) }
    }

    /// Java `isGlobal()`.  Global attributes are not in sections.
    fn is_global(&self) -> bool {
        unsafe { (*self.parent).is_global() }
    }

    /// Java `isAttribute()`.
    fn is_attribute(&self) -> bool {
        true
    }
}

impl WritableAttribute for Attribute {
    /// Java `setValue(String)`, declared `synchronized`.
    fn set_value(&mut self, new_value: Option<&str>) {
        if self.name_value_pair_list.is_none() {
            // This attribute is never the last attribute in a name/value pair.
            // Therefore there is no value to change
            // To add a value to this attribute you would have to create the name/value
            // pair where this attribute is the last attribute in the name.
            return;
        }
        // current we can only modify the last name/value pair found
        let name_value_pair = self.get_name_value_pair();
        let value = Box::into_raw(Box::new(Token::new()));
        unsafe {
            (*value).set_type_and_string(
                token::Type::Anything,
                match new_value {
                    None => panic!("java.lang.NullPointerException"),
                    Some(new_value) => new_value,
                },
            )
        };
        unsafe { (*name_value_pair).set_value(value) };
    }
}

impl ReadOnlyAttribute for Attribute {
    /// Java `getAttribute(int)`.
    unsafe fn get_attribute_by_index(&self, name: i32) -> *mut Attribute {
        if self.children.is_null() {
            return std::ptr::null_mut();
        }
        unsafe { (*self.children).get_attribute(Some(&name.to_string())) }
    }

    /// Java `getAttribute(String)`.
    unsafe fn get_attribute_by_name(&self, name: Option<&str>) -> *mut Attribute {
        if self.children.is_null() {
            return std::ptr::null_mut();
        }
        unsafe { (*self.children).get_attribute(name) }
    }

    /// Java `getName()`.
    fn get_name(&self) -> String {
        unsafe { (*self.name).get_values() }
    }

    /// Java `getChildren()`.
    fn get_children(&self) -> *mut AttributeList {
        self.children
    }

    /// Java `getMultiLineValue()`.
    fn get_multi_line_value(&self) -> Option<String> {
        let name_value_pair = self.get_name_value_pair();
        if !name_value_pair.is_null() {
            let value = unsafe { (*name_value_pair).get_token_value() };
            if !value.is_null() {
                return Some(unsafe { (*value).get_multi_line_values() });
            }
        }
        None
    }

    /// Java `getValue()`.
    ///
    /// Gets the value from the first nameValuePair in the nameValuePairList.
    fn get_value(&self) -> Option<String> {
        let name_value_pair = self.get_name_value_pair();
        if !name_value_pair.is_null() {
            let value = unsafe { (*name_value_pair).get_token_value() };
            if !value.is_null() {
                return Some(unsafe { (*value).get_values() });
            }
        }
        None
    }

    /// Java `getLineNum()`.
    fn get_line_num(&self) -> i32 {
        self.line_num
    }

    /// Java `toString()`.  The children render through `AttributeList.toString()`,
    /// whose `HashMap` rendering order is the limitation that module documents.
    fn to_string(&self) -> String {
        format!(
            "etomo.storage.autodoc.Attribute[,name={},\nchildren={}]",
            unsafe { (*self.name).to_string() },
            if self.children.is_null() {
                "null".to_string()
            } else {
                unsafe { (*self.children).to_string() }
            }
        )
    }
}
