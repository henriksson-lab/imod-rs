//! `IMOD/Etomo/src/etomo/type/IntKeyList.java`.
//!
//! A list of values (strings, or numbers of one `EtomoNumber.Type`) keyed by integers,
//! stored in properties as `<prepend>.<listKey>.<key>=<value>` plus the `First` and
//! `Last` keys of the range.
//!
//! **Map.**  Java keeps the pairs in a `HashMap` keyed by `String.valueOf(key)`.  A
//! `BTreeMap` with the same keys is used here so iteration is deterministic; the only
//! places the order shows are `store` (whose output is a property map, where order does
//! not matter) and `toString`.
//!
//! **Synchronisation.**  Java's `synchronized` methods mutate the list; here they take
//! `&mut self`, and a shared list lives behind its owner's lock (`MetaData` holds its
//! lists in a `Mutex`).
//!
//! **`prepend == ""`.**  `getPrepend` tests `prepend == ""`, a reference comparison that
//! is true only for the interned literal; the callers pass either that literal or a
//! non-empty prepend, so it is translated as `prepend.is_empty()`.

use std::collections::BTreeMap;

use super::const_etomo_number::{Number, Type, java_lang_string_matches_whitespace};
use super::const_int_key_list::ConstIntKeyList;
use super::etomo_number::EtomoNumber;

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `IntKeyList`.
#[derive(Clone, Debug)]
pub struct IntKeyList {
    /// Java field `map`.
    map: BTreeMap<String, Pair>,
    /// Java field `rowKey`.
    row_key: RowKey,
    /// Java field `etomoNumberType`.  If etomoNumberType is null, then the value is a
    /// string.
    etomo_number_type: Option<Type>,
    /// Java field `listKey`.
    list_key: Option<String>,
    /// Java field `debug`.
    debug: bool,
}

impl IntKeyList {
    /// Java private `IntKeyList(String, EtomoNumber.Type)`.
    fn new(list_key: Option<&str>, etomo_number_type: Option<Type>) -> IntKeyList {
        IntKeyList {
            map: BTreeMap::new(),
            row_key: RowKey::new(),
            etomo_number_type,
            list_key: list_key.map(|s| s.to_string()),
            debug: false,
        }
    }

    /// Java static `getStringInstance()`.
    pub fn get_string_instance() -> IntKeyList {
        IntKeyList::new(None, None)
    }

    /// Java static `getStringInstance(String)`.
    pub fn get_string_instance_with_key(list_key: &str) -> IntKeyList {
        IntKeyList::new(Some(list_key), None)
    }

    /// Java static `getNumberInstance()`.
    pub fn get_number_instance() -> IntKeyList {
        IntKeyList::new(None, Some(Type::get_default()))
    }

    /// Java static `getNumberInstance(String)`.
    pub fn get_number_instance_with_key(list_key: &str) -> IntKeyList {
        IntKeyList::new(Some(list_key), Some(Type::get_default()))
    }

    /// Java static `getNumberInstance(String, EtomoNumber.Type)`.
    pub fn get_number_instance_with_type(
        list_key: &str,
        etomo_number_type: Option<Type>,
    ) -> IntKeyList {
        let etomo_number_type = match etomo_number_type {
            None => Type::get_default(),
            Some(etomo_number_type) => etomo_number_type,
        };
        IntKeyList::new(Some(list_key), Some(etomo_number_type))
    }

    /// Java `reset()`.
    pub fn reset(&mut self) {
        self.map.clear();
        self.row_key.reset();
    }

    /// Java `reset(int)`.  Clear with a non default startKey.  The startKey is only
    /// used when lastKey is null and put(String) or put(ConstEtomoNumber) is used.
    pub fn reset_with_start_key(&mut self, start_key: i32) {
        self.reset();
        self.row_key.reset_with_start_key(start_key);
    }

    /// Java `set(String[])`.  A null element is Java null (`None`).
    pub fn set_string_array(&mut self, input: Option<&[Option<String>]>) {
        let input = match input {
            None => return,
            Some(input) => input,
        };
        for i in 0..input.len() {
            if let Some(value) = &input[i] {
                self.put_string(i as i32, Some(value));
            }
        }
    }

    /// Java `set(List)`.  The elements are `String`s for a string list and `Number`s
    /// (from `ConstEtomoNumber`s) otherwise; `put(int, Object)` casts accordingly.
    pub fn set_list(&mut self, input: Option<&[Option<ListValue>]>) {
        let input = match input {
            None => return,
            Some(input) => input,
        };
        for i in 0..input.len() {
            if let Some(object) = &input[i] {
                self.put_object(i as i32, object.clone());
            }
        }
    }

    /// Java `set(ConstIntKeyList)`.  Added intKeyList to the instance.  This function
    /// does not replace the existing list, though it may replace individual elements.
    pub fn set_int_key_list(&mut self, int_key_list: Option<&dyn ConstIntKeyList>) {
        let int_key_list = match int_key_list {
            None => return,
            Some(int_key_list) => int_key_list,
        };
        let mut i = int_key_list.get_first_key();
        while i <= int_key_list.get_last_key() {
            let value = int_key_list.get_string(i);
            if let Some(value) = value {
                self.put_string(i, Some(&value));
            }
            i += 1;
        }
    }

    /// Java `put(int, String)`.
    pub fn put_string(&mut self, key: i32, value: Option<&str>) {
        self.row_key.adjust_first_last_keys(key);
        self.map.insert(
            IntKeyList::build_key(key),
            Pair::new_string(key, value, self.etomo_number_type),
        );
        if self.debug {
            println!("map.size()={}", self.map.len());
        }
    }

    /// Java `put(int, Object)`.  The source casts the object to `String` for a string
    /// list and to `ConstEtomoNumber` otherwise (a `ClassCastException` for any other
    /// object); `ListValue` carries exactly those two shapes.
    pub fn put_object(&mut self, key: i32, value: ListValue) {
        self.row_key.adjust_first_last_keys(key);
        let pair;
        if self.etomo_number_type.is_none() {
            let value = match value {
                ListValue::String(value) => value,
                ListValue::EtomoNumber(_) => panic!("java.lang.ClassCastException"),
            };
            pair = Pair::new_string(key, value.as_deref(), self.etomo_number_type);
        } else {
            let value = match value {
                ListValue::EtomoNumber(value) => value,
                ListValue::String(_) => panic!("java.lang.ClassCastException"),
            };
            pair = Pair::new_etomo_number(key, &value, self.etomo_number_type);
        }
        self.map.insert(IntKeyList::build_key(key), pair);
        if self.debug {
            println!("map.size()={}", self.map.len());
        }
    }

    /// Java `put(int, ConstEtomoNumber)`.
    pub fn put_etomo_number(&mut self, key: i32, value: &EtomoNumber) {
        self.row_key.adjust_first_last_keys(key);
        self.map.insert(
            IntKeyList::build_key(key),
            Pair::new_etomo_number(key, value, self.etomo_number_type),
        );
        if self.debug {
            println!("map.size()={}", self.map.len());
        }
    }

    /// Java `put(int, int)`.
    pub fn put_int(&mut self, key: i32, value: i32) {
        self.row_key.adjust_first_last_keys(key);
        self.map.insert(
            IntKeyList::build_key(key),
            Pair::new_int(key, value, self.etomo_number_type),
        );
        if self.debug {
            println!("map.size()={}", self.map.len());
        }
    }

    /// Java `add(ConstEtomoNumber)`.  Puts the value, generates its own key
    /// (lastKey+1).
    pub fn add_etomo_number(&mut self, value: &EtomoNumber) {
        let key = self.row_key.gen_key();
        self.map.insert(
            IntKeyList::build_key(key),
            Pair::new_etomo_number(key, value, self.etomo_number_type),
        );
        if self.debug {
            println!("map.size()={}", self.map.len());
        }
    }

    /// Java `add(String)`.  Puts the value, generates its own key (lastKey+1).
    pub fn add_string(&mut self, value: Option<&str>) {
        let key = self.row_key.gen_key();
        self.map.insert(
            IntKeyList::build_key(key),
            Pair::new_string(key, value, self.etomo_number_type),
        );
        if self.debug {
            println!("map.size()={}", self.map.len());
        }
    }

    /// Java `store(Properties, String)`.
    pub fn store(&self, props: &mut BTreeMap<String, String>, prepend: &str) {
        if self.list_key.is_none() {
            return;
        }
        // Remove everything from this map from properties. This means that
        // renumbering is unnecessary.
        self.remove(props, prepend);
        let prepend = self.get_prepend(prepend);
        let group = format!("{}.", prepend);
        self.row_key.store(props, &prepend);
        if self.map.is_empty() {
            return;
        }
        for pair in self.map.values() {
            if !pair.is_null() {
                // `isNull()` is false, so the value string is not null.
                props.insert(
                    format!("{}{}", group, pair.get_key()),
                    pair.get_value_string().unwrap_or_default(),
                );
            }
        }
    }

    /// Java `load(Properties, String)`.
    pub fn load(&mut self, props: &BTreeMap<String, String>, prepend: &str) {
        if self.debug {
            println!("load:prepend={}", prepend);
        }
        if self.list_key.is_none() {
            return;
        }
        let prepend = self.get_prepend(prepend);
        if self.debug {
            println!("prepend={}", prepend);
        }
        let group = format!("{}.", prepend);
        self.row_key.load(props, &prepend);
        let mut i = self.get_first_key();
        while i <= self.get_last_key() {
            let value = props.get(&format!("{}{}", group, i)).cloned();
            if let Some(value) = value {
                self.map.insert(
                    i.to_string(),
                    Pair::new_string(i, Some(&value), self.etomo_number_type),
                );
                if self.debug {
                    println!("map.size()={}", self.map.len());
                }
            }
            i += 1;
        }
    }

    /// Java `load(Properties, String, String)`.
    pub fn load_with_temp_key(
        &mut self,
        props: &BTreeMap<String, String>,
        prepend: &str,
        temp_key: &str,
    ) {
        if self.list_key.is_none() {
            return;
        }
        let prepend = IntKeyList::get_prepend_with_temp_key(prepend, temp_key);
        let group = format!("{}.", prepend);
        self.row_key.load(props, &prepend);
        let mut i = self.get_first_key();
        while i <= self.get_last_key() {
            let value = props.get(&format!("{}{}", group, i)).cloned();
            if let Some(value) = value {
                self.map.insert(
                    i.to_string(),
                    Pair::new_string(i, Some(&value), self.etomo_number_type),
                );
                if self.debug {
                    println!("map.size()={}", self.map.len());
                }
            }
            i += 1;
        }
    }

    /// Java `remove(Properties, String, String)`.
    pub fn remove_with_temp_key(
        &self,
        props: &mut BTreeMap<String, String>,
        prepend: &str,
        temp_key: Option<&str>,
    ) {
        let temp_key = match temp_key {
            None => return,
            Some(temp_key) => temp_key,
        };
        let prepend = IntKeyList::get_prepend_with_temp_key(prepend, temp_key);
        let group = format!("{}.", prepend);
        let mut old_row_key = RowKey::new_from_instance(&self.row_key);
        old_row_key.load(props, &prepend);
        let mut i = old_row_key.get_first_key();
        while i <= old_row_key.get_last_key() {
            props.remove(&format!("{}{}", group, i));
            i += 1;
        }
        old_row_key.remove(props, &prepend);
    }

    /// Java `containsValue(String)`.
    ///
    /// Upstream bug fixed in translation (IntKeyList.java:339): the source asks the
    /// `HashMap` whether it contains the `String` as a value, but the values are
    /// `Pair` objects, so the answer was always false.  The element is now compared
    /// with each pair's value string.
    pub fn contains_value(&self, element: Option<&str>) -> bool {
        let element = match element {
            None => return false,
            Some(element) => element,
        };
        self.map
            .values()
            .any(|pair| pair.get_value_string().as_deref() == Some(element))
    }

    /// Java `setDebug`.
    pub fn set_debug(&mut self, debug: bool) {
        self.debug = debug;
        self.row_key.set_debug(debug);
    }

    /// Java private `remove(Properties, String)`.  Remove the data associated with the
    /// keys of this instance, without changing the value of this instance.
    fn remove(&self, props: &mut BTreeMap<String, String>, prepend: &str) {
        if self.list_key.is_none() {
            return;
        }
        let prepend = self.get_prepend(prepend);
        let group = format!("{}.", prepend);
        let mut old_row_key = RowKey::new_from_instance(&self.row_key);
        old_row_key.load(props, &prepend);
        let mut i = old_row_key.get_first_key();
        while i <= old_row_key.get_last_key() {
            props.remove(&format!("{}{}", group, i));
            i += 1;
        }
        old_row_key.remove(props, &prepend);
    }

    /// Java private final `getPrepend(String)`.  Only reached with a non-null
    /// `listKey`.
    fn get_prepend(&self, prepend: &str) -> String {
        let list_key = self.list_key.as_deref().unwrap_or("null");
        if prepend.is_empty() {
            list_key.to_string()
        } else {
            format!("{}.{}", prepend, list_key)
        }
    }

    /// Java private final `getPrepend(String, String)`.
    fn get_prepend_with_temp_key(prepend: &str, temp_key: &str) -> String {
        if prepend.is_empty() {
            temp_key.to_string()
        } else {
            format!("{}.{}", prepend, temp_key)
        }
    }

    /// Java package-private static `buildKey`.
    pub(crate) fn build_key(key: i32) -> String {
        key.to_string()
    }
}

impl ConstIntKeyList for IntKeyList {
    /// Java `getFirstKey`.
    fn get_first_key(&self) -> i32 {
        self.row_key.get_first_key()
    }

    /// Java `getLastKey`.
    fn get_last_key(&self) -> i32 {
        self.row_key.get_last_key()
    }

    /// Java `getString`.
    fn get_string(&self, key: i32) -> Option<String> {
        let pair = self.map.get(&IntKeyList::build_key(key))?;
        pair.get_value_string()
    }

    /// Java `getEtomoNumber`.
    fn get_etomo_number(&self, key: i32) -> Option<EtomoNumber> {
        let pair = self.map.get(&IntKeyList::build_key(key))?;
        let mut etomo_number;
        match self.etomo_number_type {
            None => {
                etomo_number = EtomoNumber::new_with_type(Some(Type::Double));
                etomo_number.set_string(pair.get_value_string().as_deref());
                Some(etomo_number)
            }
            Some(etomo_number_type) => {
                etomo_number = EtomoNumber::new_with_type(Some(etomo_number_type));
                etomo_number.set_number(pair.get_value_number());
                Some(etomo_number)
            }
        }
    }

    /// Java `containsKey`.
    fn contains_key(&self, key: i32) -> bool {
        self.map.contains_key(&IntKeyList::build_key(key))
    }

    /// Java `size`.
    fn size(&self) -> i32 {
        self.map.len() as i32
    }

    /// Java `getWalker`.  Iterator that doesn't allow changes to the IntKeyList.
    fn get_walker(&self) -> Walker<'_> {
        Walker::new(self)
    }

    /// Java `isEmpty`.
    fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

/// Java `toString`: `"[map=" + map + "]"`, with `AbstractMap.toString`'s
/// `{key=value, ...}` layout (in key order; see the module header).
impl std::fmt::Display for IntKeyList {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let entries: Vec<String> = self
            .map
            .iter()
            .map(|(key, pair)| format!("{}={}", key, pair))
            .collect();
        write!(f, "[map={{{}}}]", entries.join(", "))
    }
}

/// An element of the `List` given to Java `set(List)` / `put(int, Object)`: a `String`
/// for a string list, a `ConstEtomoNumber` for a number list.
#[derive(Clone, Debug)]
pub enum ListValue {
    /// A Java `String` (possibly null).
    String(Option<String>),
    /// A Java `ConstEtomoNumber`.
    EtomoNumber(EtomoNumber),
}

/// Java public static final nested class `Walker`.
pub struct Walker<'a> {
    /// Java field `started`.
    started: bool,
    /// Java field `debug`.
    debug: bool,
    /// Java field `key`.
    key: i32,
    /// Java field `list`.
    list: &'a dyn ConstIntKeyList,
}

impl<'a> Walker<'a> {
    /// Java package-private `Walker(ConstIntKeyList)`.
    pub(crate) fn new(list: &'a dyn ConstIntKeyList) -> Walker<'a> {
        Walker {
            started: false,
            debug: false,
            key: 0,
            list,
        }
    }

    /// Java `isEmpty`.
    pub fn is_empty(&self) -> bool {
        self.list.is_empty()
    }

    /// Java `hasNext`.  Increments the key until it points to the next value.  Returns
    /// true if there are values left.
    pub fn has_next(&mut self) -> bool {
        self.start();
        let mut has_next = false;
        while !has_next && self.key <= self.list.get_last_key() {
            has_next = self.list.contains_key(self.key);
            if !has_next {
                self.key += 1;
            }
        }
        has_next
    }

    /// Java `nextEtomoNumber`.  Increments the key until it points to the next value.
    /// Gets the value and increments the key.
    pub fn next_etomo_number(&mut self) -> Option<EtomoNumber> {
        self.start();
        let mut value = None;
        while value.is_none() && self.key <= self.list.get_last_key() {
            value = self.list.get_etomo_number(self.key);
            self.key += 1;
        }
        value
    }

    /// Java `nextString`.  Returns the next value as a string.
    pub fn next_string(&mut self) -> Option<String> {
        self.start();
        let mut value = None;
        while value.is_none() && self.key <= self.list.get_last_key() {
            value = self.list.get_string(self.key);
            self.key += 1;
        }
        value
    }

    /// Java `getLastEtomoNumber`.
    pub fn get_last_etomo_number(&self) -> Option<EtomoNumber> {
        self.list.get_etomo_number(self.list.get_last_key())
    }

    /// Java `size`.
    pub fn size(&self) -> i32 {
        if self.debug {
            println!("list.size()={}", self.list.size());
        }
        self.list.size()
    }

    /// Java `getFirstKey`.
    pub fn get_first_key(&self) -> i32 {
        self.list.get_first_key()
    }

    /// Java `setDebug`.
    pub fn set_debug(&mut self, debug: bool) {
        self.debug = debug;
    }

    /// Java private `start`.  Start walking through the list.
    fn start(&mut self) {
        if !self.started {
            self.key = self.list.get_first_key();
            self.started = true;
        }
    }
}

/// Java `Walker.toString`.  The list is printed as its first and last keys and size;
/// the Java concatenation calls the list's `toString`, which a `&dyn ConstIntKeyList`
/// does not expose.
impl std::fmt::Display for Walker<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.started {
            return write!(f, "[key={},list=[size={}]]", self.key, self.list.size());
        }
        write!(f, "[list=[size={}]]", self.list.size())
    }
}

/// Java private static final nested class `Pair`.
#[derive(Clone, Debug)]
struct Pair {
    /// Java field `key`.
    key: i32,
    /// Java field `stringValue`.
    string_value: Option<String>,
    /// Java field `number`; can only be null for a number list through `put(int,
    /// Object)` with a null element, which `set(List)` never passes.
    number: Option<Number>,
    /// Java field `numericType`.
    numeric_type: Option<Type>,
}

impl Pair {
    /// Java private `Pair(int, String, EtomoNumber.Type)`.
    fn new_string(key: i32, value: Option<&str>, etomo_number_type: Option<Type>) -> Pair {
        match etomo_number_type {
            None => Pair {
                key,
                string_value: value.map(|s| s.to_string()),
                number: None,
                numeric_type: etomo_number_type,
            },
            Some(etomo_number_type) => {
                let mut etomo_number = EtomoNumber::new_with_type(Some(etomo_number_type));
                let number = etomo_number.set_string(value).get_number();
                Pair {
                    key,
                    string_value: None,
                    number: Some(number),
                    numeric_type: Some(etomo_number_type),
                }
            }
        }
    }

    /// Java private `Pair(int, int, EtomoNumber.Type)`: `this(key, new Integer(value),
    /// etomoNumberType)`.
    fn new_int(key: i32, value: i32, etomo_number_type: Option<Type>) -> Pair {
        Pair::new_number(key, Some(Number::Integer(value)), etomo_number_type)
    }

    /// Java private `Pair(int, ConstEtomoNumber, EtomoNumber.Type)`: `this(key,
    /// value.getNumber(), etomoNumberType)`.
    fn new_etomo_number(key: i32, value: &EtomoNumber, etomo_number_type: Option<Type>) -> Pair {
        Pair::new_number(key, Some(value.get_number()), etomo_number_type)
    }

    /// Java private `Pair(int, Number, EtomoNumber.Type)`.
    fn new_number(key: i32, value: Option<Number>, etomo_number_type: Option<Type>) -> Pair {
        match etomo_number_type {
            None => Pair {
                key,
                string_value: Some(match value {
                    None => String::new(),
                    Some(value) => value.to_string(),
                }),
                number: None,
                numeric_type: None,
            },
            Some(etomo_number_type) => Pair {
                key,
                string_value: None,
                number: value,
                numeric_type: Some(etomo_number_type),
            },
        }
    }

    /// Java `getKey`.
    fn get_key(&self) -> i32 {
        self.key
    }

    /// Java `getValueString`; null is `None`.  A number pair's `number` is never
    /// null in practice (see the field); Java would throw `NullPointerException` in
    /// `number.toString()`, and `None` is returned instead.
    fn get_value_string(&self) -> Option<String> {
        if self.numeric_type.is_none() {
            return self.string_value.clone();
        }
        self.number.map(|number| number.to_string())
    }

    /// Java `getValueNumber`.
    fn get_value_number(&self) -> Option<Number> {
        if self.numeric_type.is_none() {
            return None;
        }
        self.number
    }

    /// Java `isNull`.
    fn is_null(&self) -> bool {
        match self.numeric_type {
            None => match &self.string_value {
                None => true,
                Some(string_value) => java_lang_string_matches_whitespace(string_value),
            },
            Some(numeric_type) => {
                let mut etomo_number = EtomoNumber::new_with_type(Some(numeric_type));
                etomo_number.set_number(self.number).is_null()
            }
        }
    }
}

/// Java `Pair.toString`.
impl std::fmt::Display for Pair {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.numeric_type.is_none() {
            return write!(
                f,
                "[key={},value={}]",
                self.key,
                self.string_value.as_deref().unwrap_or("null")
            );
        }
        write!(
            f,
            "[key={},value={}]",
            self.key,
            match self.number {
                None => "null".to_string(),
                Some(number) => number.to_string(),
            }
        )
    }
}

/// Java private static final nested class `RowKey`.
#[derive(Clone, Debug)]
struct RowKey {
    /// Java field `firstKey`.
    first_key: EtomoNumber,
    /// Java field `lastKey`.
    last_key: EtomoNumber,
    /// Java field `debug`.
    debug: bool,
    /// Java field `startKey`.
    start_key: i32,
}

/// Java `RowKey.FIRST_KEY`.
const FIRST_KEY: &str = "First";
/// Java `RowKey.LAST_KEY`.
const LAST_KEY: &str = "Last";
/// Java `RowKey.DEFAULT_START_KEY`.
const DEFAULT_START_KEY: i32 = 0;

impl RowKey {
    /// Java private `RowKey()`.
    fn new() -> RowKey {
        let mut row_key = RowKey {
            first_key: EtomoNumber::new_with_name(FIRST_KEY),
            last_key: EtomoNumber::new_with_name(LAST_KEY),
            debug: false,
            start_key: DEFAULT_START_KEY,
        };
        row_key.first_key.set_display_value_int(DEFAULT_START_KEY);
        row_key.last_key.set_display_value_int(DEFAULT_START_KEY);
        row_key
    }

    /// Java private `RowKey(RowKey)`.  The copy's display values are not set, as in
    /// the source.
    fn new_from_instance(row_key: &RowKey) -> RowKey {
        let mut copy = RowKey {
            first_key: EtomoNumber::new_with_name(FIRST_KEY),
            last_key: EtomoNumber::new_with_name(LAST_KEY),
            debug: false,
            start_key: DEFAULT_START_KEY,
        };
        copy.first_key
            .set_const_etomo_number(Some(&row_key.first_key.base));
        copy.last_key
            .set_const_etomo_number(Some(&row_key.last_key.base));
        copy.start_key = row_key.start_key;
        copy
    }

    /// Java `reset()`.
    fn reset(&mut self) {
        self.first_key.reset();
        self.last_key.reset();
        self.start_key = DEFAULT_START_KEY;
    }

    /// Java `setDebug`.
    fn set_debug(&mut self, debug: bool) {
        self.debug = debug;
    }

    /// Java `reset(int)`.
    fn reset_with_start_key(&mut self, start_key: i32) {
        self.start_key = start_key;
    }

    /// Java `genKey`.
    fn gen_key(&mut self) -> i32 {
        let key;
        if self.last_key.is_null() {
            key = self.start_key;
        } else {
            key = self.get_last_key() + 1;
        }
        self.adjust_first_last_keys(key);
        key
    }

    /// Java `getLastKey`.
    fn get_last_key(&self) -> i32 {
        self.last_key.get_int()
    }

    /// Java `adjustFirstLastKeys`.
    fn adjust_first_last_keys(&mut self, key: i32) {
        if self.first_key.is_null() || self.first_key.gt_int(key) {
            self.first_key.set_int(key);
        }
        if self.last_key.is_null() || self.last_key.lt_int(key) {
            self.last_key.set_int(key);
        }
    }

    /// Java `getFirstKey`.
    fn get_first_key(&self) -> i32 {
        self.first_key.get_int()
    }

    /// Java `store`.
    ///
    /// Upstream bug fixed in translation (IntKeyList.java:614-618): the source throws
    /// `IllegalStateException("StartKey must be not be greater then endKey...")` when
    /// the first key is greater than the last, which aborts the whole data-file save
    /// that is storing this list.  The keys can only be out of order after loading a
    /// hand-edited file.  The message is now printed to standard error and the two
    /// keys are not stored; the rest of the save goes ahead.
    fn store(&self, props: &mut BTreeMap<String, String>, prepend: &str) {
        if self
            .first_key
            .gt_const_etomo_number(Some(&self.last_key.base))
        {
            eprintln!(
                "StartKey must be not be greater then endKey.\nstartKey={},endKey={}",
                self.first_key, self.last_key
            );
            return;
        }
        self.first_key.store_with_prepend(props, Some(prepend));
        self.last_key.store_with_prepend(props, Some(prepend));
    }

    /// Java `load`.
    fn load(&mut self, props: &BTreeMap<String, String>, prepend: &str) {
        self.first_key.load_with_prepend(props, Some(prepend));
        self.last_key.load_with_prepend(props, Some(prepend));
        if self.debug {
            println!("firstKey={},lastKey={}", self.first_key, self.last_key);
        }
    }

    /// Java `remove`.
    fn remove(&self, props: &mut BTreeMap<String, String>, prepend: &str) {
        self.first_key.remove_with_prepend(props, Some(prepend));
        self.last_key.remove_with_prepend(props, Some(prepend));
    }
}

/// Java `RowKey.toString`, including the source's missing closing bracket.
impl std::fmt::Display for RowKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[firstKey={},lastKey={}", self.first_key, self.last_key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn string_list_store_load_round_trip() {
        let mut list = IntKeyList::get_string_instance_with_key("Gen.a.TrialTomogramName");
        list.add_string(Some("trial1"));
        list.add_string(Some("trial2"));
        list.put_string(5, Some("trial5"));
        let mut props = BTreeMap::new();
        list.store(&mut props, "Setup");
        assert_eq!(
            props.get("Setup.Gen.a.TrialTomogramName.First").unwrap(),
            "0"
        );
        assert_eq!(
            props.get("Setup.Gen.a.TrialTomogramName.Last").unwrap(),
            "5"
        );
        assert_eq!(
            props.get("Setup.Gen.a.TrialTomogramName.0").unwrap(),
            "trial1"
        );
        assert_eq!(
            props.get("Setup.Gen.a.TrialTomogramName.1").unwrap(),
            "trial2"
        );
        assert_eq!(
            props.get("Setup.Gen.a.TrialTomogramName.5").unwrap(),
            "trial5"
        );
        assert_eq!(props.len(), 5);

        let mut loaded = IntKeyList::get_string_instance_with_key("Gen.a.TrialTomogramName");
        loaded.load(&props, "Setup");
        assert_eq!(loaded.size(), 3);
        assert_eq!(loaded.get_string(1).as_deref(), Some("trial2"));
        assert!(loaded.contains_value(Some("trial5")));
        let mut walker = loaded.get_walker();
        assert_eq!(walker.next_string().as_deref(), Some("trial1"));
        assert_eq!(walker.next_string().as_deref(), Some("trial2"));
        assert!(walker.has_next());
        assert_eq!(walker.next_string().as_deref(), Some("trial5"));
        assert!(!walker.has_next());

        // A store after shrinking the list removes the old entries first.
        loaded.reset();
        loaded.add_string(Some("only"));
        let mut again = props.clone();
        loaded.store(&mut again, "Setup");
        assert_eq!(again.len(), 3);
        assert_eq!(
            again.get("Setup.Gen.a.TrialTomogramName.Last").unwrap(),
            "0"
        );
    }

    #[test]
    fn empty_list_stores_nothing_and_listless_is_inert() {
        let list = IntKeyList::get_string_instance_with_key("K");
        let mut props = BTreeMap::new();
        list.store(&mut props, "Setup");
        assert!(props.is_empty());
        let keyless = IntKeyList::get_string_instance();
        keyless.store(&mut props, "Setup");
        assert!(props.is_empty());
    }

    #[test]
    fn number_list() {
        let mut list = IntKeyList::get_number_instance_with_type("N", Some(Type::Double));
        list.put_int(2, 7);
        list.put_string(3, Some("1.5"));
        assert_eq!(list.get_first_key(), 2);
        assert_eq!(list.get_string(2).as_deref(), Some("7"));
        assert_eq!(list.get_string(3).as_deref(), Some("1.5"));
        assert_eq!(list.get_etomo_number(3).unwrap().get_double(), 1.5);
        assert!(list.get_etomo_number(4).is_none());
    }
}
