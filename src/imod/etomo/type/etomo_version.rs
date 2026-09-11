//! `IMOD/Etomo/src/etomo/type/EtomoVersion.java`.
//!
//! Version of an object or file.  Used when necessary.  Treats null as the earliest
//! version.
//!
//! **Representation.**  `SectionList.list` is a raw Java `List` holding either an
//! `EtomoNumber` or a `String` per element, and every comparison in the nested class
//! branches on `instanceof EtomoNumber`.  Rust has no heterogeneous list, so the element
//! is the `Section` enum with exactly those two shapes; `instanceof EtomoNumber` becomes
//! a match on the variant.
//!
//! Java `Properties` is modelled by a deterministic `BTreeMap<String, String>`, as
//! `etomo/storage/storable.rs` does.
#![allow(dead_code)]

use super::const_etomo_number::java_lang_string_trim;
use super::const_etomo_version::ConstEtomoVersion;
use super::etomo_number::EtomoNumber;
use crate::imod::etomo::storage::storable::Storable;
use crate::imod::etomo::util::utilities::{EMPTY_PATTERN, java_lang_string_split};
use regex::Regex;
use std::collections::BTreeMap;
use std::sync::LazyLock;

/// The literal `"\\s+"` that `set(String)` hands `String.split`.  Java's `\s` is the
/// five ASCII characters `[ \t\n\x0B\f\r]`, which the Rust regex crate's `\s` would
/// widen to Unicode whitespace, so the class is written out.
static WHITESPACE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new("[ \t\n\u{0B}\u{0C}\r]+").unwrap());

/// The literal `"\\" + DIVIDER` that `SectionList.parse` hands `String.split`: an
/// escaped `.`, so the pattern matches one literal period.
static DIVIDER_PATTERN: LazyLock<Regex> = LazyLock::new(|| Regex::new("\\.").unwrap());

/// One element of `SectionList.list`.  See the module header.
#[derive(Clone)]
enum Section {
    /// An `EtomoNumber` element, which `add(String)` stores for a valid, non-null
    /// numeric section.
    Numeric(EtomoNumber),
    /// A `String` element.
    Text(String),
}

/// `Object.toString()` on a list element.
impl std::fmt::Display for Section {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Section::Numeric(number) => write!(f, "{}", number),
            Section::Text(string) => f.write_str(string),
        }
    }
}

/// Java private static nested class `SectionList`.
#[derive(Clone)]
struct SectionList {
    /// Java field `list`.
    list: Vec<Section>,
}

impl SectionList {
    /// Java `DIVIDER`.
    const DIVIDER: &'static str = ".";

    /// The field initialiser `new SectionList()` with `list = new ArrayList()`.
    fn new() -> SectionList {
        SectionList { list: Vec::new() }
    }

    /// Java `parse`.  Parses a string into sections.
    fn parse(&mut self, string: Option<&str>) {
        let string = match string {
            Some(string) if string.contains(Self::DIVIDER) => string,
            _ => {
                eprintln!("WARNING: Bad version: {}", string.unwrap_or("null"));
                return;
            }
        };
        self.reset();
        // sections are divided by ".".
        let array = java_lang_string_split(string, &DIVIDER_PATTERN);
        if array.is_empty() {
            return;
        }
        for item in &array {
            self.add(Some(java_lang_string_trim(item)));
        }
    }

    /// Java `isNull`.
    fn is_null(&self) -> bool {
        self.list.is_empty()
    }

    /// Java `size`.
    fn size(&self) -> i32 {
        self.list.len() as i32
    }

    /// Java `reset`.
    fn reset(&mut self) {
        self.list.clear();
    }

    /// Java `equals(SectionList, int)`.  Returns true if `list[index]` equals
    /// `sectionList.list[index]`.
    fn equals(&self, section_list: &SectionList, index: i32) -> bool {
        let section = &self.list[index as usize];
        let other_section = &section_list.list[index as usize];
        if let Section::Numeric(numeric_section) = section {
            // Do a numeric comparison.
            if let Section::Numeric(numeric_other_section) = other_section {
                return numeric_section
                    .base
                    .equals_const_etomo_number(Some(&numeric_other_section.base));
            } else {
                // One is numeric and the other non-numeric; can't be equal.
                return false;
            }
        }
        // Non-numeric - do a string comparison.
        section.to_string() == other_section.to_string()
    }

    /// Java `gt(SectionList, int)`.  Returns true if `list[index]` is greater then
    /// `sectionList.list[index]`.  Does a numeric comparison if both elements are
    /// numeric.
    fn gt(&self, section_list: &SectionList, index: i32) -> bool {
        let section = &self.list[index as usize];
        let other_section = &section_list.list[index as usize];
        if let Section::Numeric(numeric_section) = section {
            // Do a numeric comparison.
            if let Section::Numeric(numeric_other_section) = other_section {
                return numeric_section
                    .base
                    .gt_const_etomo_number(Some(&numeric_other_section.base));
            }
        }
        // One or both are non-numeric - do a string comparison.  `String.compareTo`
        // orders by UTF-16 code unit, which `encode_utf16` reproduces.
        section
            .to_string()
            .encode_utf16()
            .cmp(other_section.to_string().encode_utf16())
            == std::cmp::Ordering::Greater
    }

    /// Java `lt(SectionList, int)`.  Returns true if `list[index]` is less then
    /// `sectionList.list[index]`.  Does a numeric comparison if both elements are
    /// numeric.
    fn lt(&self, section_list: &SectionList, index: i32) -> bool {
        let section = &self.list[index as usize];
        let other_section = &section_list.list[index as usize];
        if let Section::Numeric(numeric_section) = section {
            // Do a numeric comparison.
            if let Section::Numeric(numeric_other_section) = other_section {
                return numeric_section
                    .base
                    .lt_const_etomo_number(Some(&numeric_other_section.base));
            }
        }
        // One or both are non-numeric - do a string comparison.
        section
            .to_string()
            .encode_utf16()
            .cmp(other_section.to_string().encode_utf16())
            == std::cmp::Ordering::Less
    }

    /// Java `add(String)`.  Adds section to list.  Doesn't add empty sections.  Adds
    /// numeric sections as `EtomoNumber`s.
    fn add(&mut self, section: Option<&str>) {
        // Ignore empty sections
        let section = match section {
            None => return,
            Some(section) if section.is_empty() || EMPTY_PATTERN.is_match(section) => return,
            Some(section) => section,
        };
        // Try to store a numeric section
        let mut numeric_section = EtomoNumber::new();
        numeric_section.set_string(Some(section));
        if numeric_section.base.is_valid() && !numeric_section.base.is_null() {
            // Store a numeric section.
            self.list.push(Section::Numeric(numeric_section));
        } else {
            // Store a non-numeric section.
            self.list.push(Section::Text(section.to_string()));
        }
    }

    /// Java `isNumeric`.  Return true if each element in list is an `EtomoNumber`.
    fn is_numeric(&self) -> bool {
        if self.list.is_empty() {
            return false;
        }
        for section in &self.list {
            if !matches!(section, Section::Numeric(_)) {
                return false;
            }
        }
        true
    }

    /// Java `add(SectionList)`.  Add the sections from one sectionList to another.
    /// Note that the source adds `sectionList.get(i)`, which is a `String` even for a
    /// numeric section, so every copied element becomes non-numeric.
    fn add_section_list(&mut self, section_list: &SectionList) {
        for i in 0..section_list.size() {
            self.list.push(Section::Text(section_list.get(i)));
        }
    }

    /// Java `get(int)`.
    fn get(&self, index: i32) -> String {
        let section = &self.list[index as usize];
        match section {
            Section::Numeric(number) => number.to_string(),
            Section::Text(string) => string.clone(),
        }
    }
}

/// Java `SectionList.toString`.
impl std::fmt::Display for SectionList {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut builder = String::new();
        let size = self.list.len();
        for i in 0..size {
            builder.push_str(&self.list[i].to_string());
            if i < size - 1 {
                builder.push_str(SectionList::DIVIDER);
            }
        }
        f.write_str(&builder)
    }
}

/// Java `EtomoVersion`.
#[derive(Clone)]
pub struct EtomoVersion {
    /// Java field `sectionList`.
    section_list: SectionList,
    /// Java field `key`.
    key: Option<String>,
    /// Java field `debug`, initialised to false.
    debug: bool,
    /// Java field `extra`, initialised to null.
    extra: Option<String>,
}

impl EtomoVersion {
    /// Java `DEFAULT_KEY`.
    pub const DEFAULT_KEY: &'static str = "Version";

    /// Java `getDefaultInstance()`.
    pub fn get_default_instance() -> EtomoVersion {
        EtomoVersion::new()
    }

    /// Java `getDefaultInstance(String)`.
    pub fn get_default_instance_with_version(version: Option<&str>) -> EtomoVersion {
        let mut instance = EtomoVersion::new();
        instance.set(version);
        instance
    }

    /// Java `getEmptyInstance(String)`.
    pub fn get_empty_instance(key: Option<&str>) -> EtomoVersion {
        let mut instance = EtomoVersion::new();
        instance.key = key.map(|key| key.to_string());
        instance
    }

    /// Java `getInstance(String, String)`.
    pub fn get_instance(key: Option<&str>, version: Option<&str>) -> EtomoVersion {
        let mut instance = EtomoVersion::get_empty_instance(key);
        instance.set(version);
        instance
    }

    /// Java `getKey`.
    pub fn get_key(&self) -> Option<&str> {
        self.key.as_deref()
    }

    /// Java `setDebug`.
    pub fn set_debug(&mut self, debug: bool) {
        self.debug = debug;
    }

    /// Java private `EtomoVersion()`.
    fn new() -> EtomoVersion {
        EtomoVersion {
            section_list: SectionList::new(),
            key: Some(EtomoVersion::DEFAULT_KEY.to_string()),
            debug: false,
            extra: None,
        }
    }

    /// Java `hashCode`.  Returns `sectionList.hashCode()`, the identity hash of the
    /// `ArrayList`, which is `List.hashCode()`: `31 * result + element.hashCode()` over
    /// the elements starting from 1.  Java `String.hashCode` is `s[0]*31^(n-1) + ...`
    /// over UTF-16 code units; `EtomoNumber` inherits `Object.hashCode`, which is
    /// identity-based and not reproducible, so a list holding one is not comparable
    /// across runs in Java either.
    pub fn hash_code(&self) -> i32 {
        let mut result: i32 = 1;
        for section in &self.section_list.list {
            let element_hash: i32 = match section {
                Section::Text(string) => {
                    let mut hash: i32 = 0;
                    for unit in string.encode_utf16() {
                        hash = hash.wrapping_mul(31).wrapping_add(unit as i32);
                    }
                    hash
                }
                // `Object.hashCode()` - identity based, not reproducible.
                Section::Numeric(_) => 0,
            };
            result = result.wrapping_mul(31).wrapping_add(element_hash);
        }
        result
    }

    /// Java `equals(EtomoVersion)`.  Returns true if equal to the parameter.  Ignores
    /// extra string.
    pub fn equals(&self, version: Option<&EtomoVersion>) -> bool {
        // treat null as the earliest version
        let version_is_null = match version {
            None => true,
            Some(version) => version.is_null(),
        };
        if version_is_null && self.is_null() {
            return true;
        }
        if version_is_null || self.is_null() {
            return false;
        }
        let version = version.unwrap();
        if self.section_list.size() != version.section_list.size() {
            return false;
        }
        for i in 0..self.section_list.size() {
            if !self.section_list.equals(&version.section_list, i) {
                return false;
            }
        }
        true
    }

    /// Java `lt(EtomoVersion)`.  Returns true if less then the parameter.
    pub fn lt(&self, version: Option<&EtomoVersion>) -> bool {
        // treat null as the earliest version
        let version_is_null = match version {
            None => true,
            Some(version) => version.is_null(),
        };
        if self.is_null() && !version_is_null {
            return true;
        }
        if version_is_null || self.is_null() {
            return false;
        }
        let version = version.unwrap();
        let length = std::cmp::min(self.section_list.size(), version.section_list.size());
        // loop until a section is not equal to corresponding version section
        for i in 0..length {
            if self.section_list.gt(&version.section_list, i) {
                return false;
            }
            if self.section_list.lt(&version.section_list, i) {
                return true;
            }
        }
        // equal so far - shorter one is less then
        if self.section_list.size() < version.section_list.size() {
            return true;
        }
        false
    }

    /// Java `isNumeric`.
    pub fn is_numeric(&self) -> bool {
        self.section_list.is_numeric()
    }

    /// Java `le(String)`.
    pub fn le_string(&self, version: Option<&str>) -> bool {
        self.le(Some(&EtomoVersion::get_default_instance_with_version(
            version,
        )))
    }

    /// Java `le(EtomoVersion)`.  Returns true if less then or equal to the parameter.
    pub fn le(&self, version: Option<&EtomoVersion>) -> bool {
        // treat null as the earliest version
        if self.is_null() {
            return true;
        }
        let version_is_null = match version {
            None => true,
            Some(version) => version.is_null(),
        };
        if version_is_null {
            return false;
        }
        let version = version.unwrap();
        let length = std::cmp::min(self.section_list.size(), version.section_list.size());
        // loop until a section is not equal to corresponding version section
        for i in 0..length {
            if self.section_list.gt(&version.section_list, i) {
                return false;
            }
            if self.section_list.lt(&version.section_list, i) {
                return true;
            }
        }
        // equal so far - shorter one is less then
        if self.section_list.size() <= version.section_list.size() {
            return true;
        }
        false
    }

    /// Java `gt(EtomoVersion)`.  Returns true if greater then the parameter.
    pub fn gt(&self, version: Option<&EtomoVersion>) -> bool {
        // treat null as the earliest version
        let version_is_null = match version {
            None => true,
            Some(version) => version.is_null(),
        };
        if version_is_null && !self.is_null() {
            return true;
        }
        if version_is_null || self.is_null() {
            return false;
        }
        let version = version.unwrap();
        let length = std::cmp::min(self.section_list.size(), version.section_list.size());
        // loop until a section is not equal then corresponding version section
        for i in 0..length {
            if self.section_list.gt(&version.section_list, i) {
                return true;
            }
            if self.section_list.lt(&version.section_list, i) {
                return false;
            }
        }
        // equal so far - longer one is greater then
        if self.section_list.size() > version.section_list.size() {
            return true;
        }
        false
    }

    /// Java `isNull`, package-private.
    pub fn is_null(&self) -> bool {
        self.section_list.is_null()
    }

    /// Java `get(int)`.  Gets a single section.  Example: if the version is 3.12.5,
    /// `get(1)` returns 12.
    pub fn get(&self, section_index: i32) -> Option<String> {
        if section_index < self.section_list.size() {
            return Some(self.section_list.get(section_index));
        }
        None
    }

    /// Java `set(String)`.  Parses version, and saves any extra strings.  If null
    /// resets the value.
    pub fn set(&mut self, version: Option<&str>) {
        self.reset();
        let version = match version {
            None => return,
            Some(version) => version,
        };
        let array = java_lang_string_split(java_lang_string_trim(version), &WHITESPACE);
        if array.is_empty() {
            return;
        }
        // assume then that the version string comes first
        self.section_list.parse(Some(&array[0]));
        if self.section_list.is_null() {
            self.extra = Some(version.to_string());
        } else if array.len() > 1 {
            // Java `String.substring(int)` indexes UTF-16 code units, and
            // `array[0].length()` is that same count.
            let units: Vec<u16> = version.encode_utf16().collect();
            let skip = array[0].encode_utf16().count();
            self.extra = Some(String::from_utf16_lossy(&units[skip..]));
            if let Some(extra) = &self.extra {
                self.extra = Some(java_lang_string_trim(extra).to_string());
            }
        }
    }

    /// Java `set(EtomoVersion)`.
    pub fn set_etomo_version(&mut self, etomo_version: Option<&EtomoVersion>) {
        self.reset();
        let etomo_version = match etomo_version {
            None => return,
            Some(etomo_version) if etomo_version.is_null() => {
                let _ = etomo_version;
                return;
            }
            Some(etomo_version) => etomo_version,
        };
        self.section_list
            .add_section_list(&etomo_version.section_list);
        self.extra = etomo_version.extra.clone();
    }

    /// Java `reset`.
    pub fn reset(&mut self) {
        self.section_list.reset();
        self.extra = None;
    }
}

/// Java `toString`.  Returns the parseable format: `v.v...[ ...]`.
impl std::fmt::Display for EtomoVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_null() {
            return f.write_str("");
        }
        if let Some(extra) = &self.extra {
            if !extra.is_empty() {
                return write!(f, "{} {}", self.section_list, extra);
            }
        }
        write!(f, "{}", self.section_list)
    }
}

/// Java `ge(String)`, `ge(EtomoVersion)` and `lt(String)`, declared by
/// `ConstEtomoVersion`.
impl ConstEtomoVersion for EtomoVersion {
    /// Java `ge(String)`.  Returns true if greater or equal to the parameter.
    fn ge_string(&self, version: Option<&str>) -> bool {
        self.ge(Some(&EtomoVersion::get_default_instance_with_version(
            version,
        )))
    }

    /// Java `ge(EtomoVersion)`.  Returns true if greater or equal to the parameter.
    fn ge(&self, version: Option<&EtomoVersion>) -> bool {
        // treat null as the earliest version
        let version_is_null = match version {
            None => true,
            Some(version) => version.is_null(),
        };
        if version_is_null {
            return true;
        }
        if self.is_null() {
            return false;
        }
        let version = version.unwrap();
        let length = std::cmp::min(self.section_list.size(), version.section_list.size());
        // loop until a section is not equal then corresponding version section
        for i in 0..length {
            if self.section_list.gt(&version.section_list, i) {
                return true;
            }
            if self.section_list.lt(&version.section_list, i) {
                return false;
            }
        }
        // equal so far - longer one is greater then
        if self.section_list.size() >= version.section_list.size() {
            return true;
        }
        false
    }

    /// Java `lt(String)`.  Returns true if less then the parameter.
    fn lt_string(&self, version: Option<&str>) -> bool {
        self.lt(Some(&EtomoVersion::get_default_instance_with_version(
            version,
        )))
    }
}

/// Java `store(Properties)`, `store(Properties, String)`, `load(Properties)` and
/// `load(Properties, String)`, declared by `Storable`.
impl Storable for EtomoVersion {
    /// Java `store(Properties)`.
    fn store(&self, props: &mut BTreeMap<String, String>) {
        let key = match &self.key {
            None => return,
            Some(key) => key.clone(),
        };
        if self.is_null() {
            props.remove(&key);
        } else {
            props.insert(key, self.to_string());
        }
    }

    /// Java `store(Properties, String)`.
    fn store_with_prepend(&self, props: &mut BTreeMap<String, String>, prepend: &str) {
        let key = match &self.key {
            None => return,
            Some(key) => key.clone(),
        };
        let props_key = if prepend.is_empty() || EMPTY_PATTERN.is_match(prepend) {
            key.clone()
        } else if prepend.ends_with('.') {
            format!("{}{}", prepend, key)
        } else {
            format!("{}.{}", prepend, key)
        };
        if self.debug {
            eprintln!("store:prepend={},key={},toString()={}", prepend, key, self);
        }
        if self.is_null() {
            if self.debug {
                eprintln!("isNull");
            }
            props.remove(&props_key);
        } else {
            props.insert(props_key.clone(), self.to_string());
        }
        if self.debug {
            eprintln!(
                "props:{}",
                props.get(&props_key).cloned().unwrap_or("null".to_string())
            );
        }
    }

    /// Java `load(Properties)`.
    fn load(&mut self, props: &BTreeMap<String, String>) {
        let value = match &self.key {
            None => None,
            Some(key) => props.get(key).cloned(),
        };
        self.set(value.as_deref());
    }

    /// Java `load(Properties, String)`.
    fn load_with_prepend(&mut self, props: &BTreeMap<String, String>, prepend: &str) {
        let key = match &self.key {
            None => return,
            Some(key) => key.clone(),
        };
        let props_key = if prepend.is_empty() || EMPTY_PATTERN.is_match(prepend) {
            key
        } else if prepend.ends_with('.') {
            format!("{}{}", prepend, key)
        } else {
            format!("{}.{}", prepend, key)
        };
        let value = props.get(&props_key).cloned();
        self.set(value.as_deref());
    }
}
