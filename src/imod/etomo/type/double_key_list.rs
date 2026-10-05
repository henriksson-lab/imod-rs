//! `IMOD/Etomo/src/etomo/type/DoubleKeyList.java`.
//!
//! A list of values keyed by integers, stored in properties as
//! `<prepend>.<listKey>.<key>=<value>` plus the `First` and `Last` keys of the range.
//! The class is a reduced copy of `IntKeyList` (only `getStringInstance` exists).
//!
//! **Map.**  Java keeps the pairs in a `HashMap` keyed by `String.valueOf(key)`; a
//! `BTreeMap` with the same keys is used here so iteration is deterministic.  The
//! only place the order shows is `store`, whose output is a property map.
//!
//! **`prepend == ""`.**  As in `IntKeyList`, the reference comparison is translated as
//! `prepend.is_empty()`.

use std::collections::BTreeMap;

use super::const_etomo_number::{Number, Type, java_lang_string_matches_whitespace};
use super::etomo_number::EtomoNumber;

/// Java `public final class DoubleKeyList`.
#[derive(Clone, Debug)]
pub struct DoubleKeyList {
    /// Java private final `map`.
    map: BTreeMap<String, Pair>,
    /// Java private final `rowKey`.
    row_key: RowKey,
    /// Java private final `etomoNumberType`.  If etomoNumberType is null, then the
    /// value is a string.
    etomo_number_type: Option<Type>,
    /// Java private `listKey`.
    list_key: Option<String>,
    /// Java private `debug`.
    debug: bool,
}

impl DoubleKeyList {
    /// Java private `DoubleKeyList(String, EtomoNumber.Type)`.
    fn new(list_key: Option<&str>, etomo_number_type: Option<Type>) -> DoubleKeyList {
        DoubleKeyList {
            map: BTreeMap::new(),
            row_key: RowKey::new(),
            etomo_number_type,
            list_key: list_key.map(str::to_owned),
            debug: false,
        }
    }

    /// Java static `getStringInstance(String)`.
    pub fn get_string_instance(list_key: &str) -> DoubleKeyList {
        DoubleKeyList::new(Some(list_key), None)
    }

    /// Java synchronized `put(int, String)`.
    pub fn put(&mut self, key: i32, value: Option<&str>) {
        self.row_key.adjust_first_last_keys(key);
        self.map.insert(
            DoubleKeyList::build_key(key),
            Pair::new(key, value, self.etomo_number_type),
        );
        if self.debug {
            println!("map.size()={}", self.map.len());
        }
    }

    /// Java package-private static `buildKey(int)`.
    fn build_key(key: i32) -> String {
        key.to_string()
    }

    /// Java synchronized `reset()`.
    pub fn reset(&mut self) {
        self.map.clear();
        self.row_key.reset();
    }

    /// Java `set(DoubleKeyList)`.
    pub fn set(&mut self, double_key_list: Option<&DoubleKeyList>) {
        let Some(double_key_list) = double_key_list else {
            return;
        };
        let mut i = double_key_list.get_first_key();
        while i <= double_key_list.get_last_key() {
            let value = double_key_list.get_string(i);
            if let Some(value) = value {
                self.put(i, Some(&value));
            }
            i += 1;
        }
    }

    /// Java `getFirstKey()`.
    pub fn get_first_key(&self) -> i32 {
        self.row_key.get_first_key()
    }

    /// Java `getLastKey()`.
    pub fn get_last_key(&self) -> i32 {
        self.row_key.get_last_key()
    }

    /// Java `getString(int)`.
    pub fn get_string(&self, key: i32) -> Option<String> {
        let pair = self.map.get(&DoubleKeyList::build_key(key))?;
        pair.get_value_string()
    }

    /// Java private final `getPrepend(String)`.  Only reached with a non-null
    /// `listKey`.
    fn get_prepend(&self, prepend: &str) -> String {
        let list_key = self.list_key.as_deref().unwrap_or("null");
        if prepend.is_empty() {
            list_key.to_owned()
        } else {
            format!("{prepend}.{list_key}")
        }
    }

    /// Java `load(Properties, String)`.
    pub fn load(&mut self, props: &BTreeMap<String, String>, prepend: &str) {
        if self.debug {
            println!("load:prepend={prepend}");
        }
        if self.list_key.is_none() {
            return;
        }
        let prepend = self.get_prepend(prepend);
        if self.debug {
            println!("prepend={prepend}");
        }
        let group = format!("{prepend}.");
        self.row_key.load(props, &prepend);
        let mut i = self.get_first_key();
        while i <= self.get_last_key() {
            let value = props.get(&format!("{group}{i}")).cloned();
            if let Some(value) = value {
                self.map
                    .insert(i.to_string(), Pair::new(i, Some(&value), self.etomo_number_type));
                if self.debug {
                    println!("map.size()={}", self.map.len());
                }
            }
            i += 1;
        }
    }

    /// Java private `remove(Properties, String)`.
    fn remove(&self, props: &mut BTreeMap<String, String>, prepend: &str) {
        if self.list_key.is_none() {
            return;
        }
        let prepend = self.get_prepend(prepend);
        let group = format!("{prepend}.");
        let mut old_row_key = RowKey::new_from_instance(&self.row_key);
        old_row_key.load(props, &prepend);
        let mut i = old_row_key.get_first_key();
        while i <= old_row_key.get_last_key() {
            props.remove(&format!("{group}{i}"));
            i += 1;
        }
        old_row_key.remove(props, &prepend);
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
        let group = format!("{prepend}.");
        self.row_key.store(props, &prepend);
        if self.map.is_empty() {
            return;
        }
        for pair in self.map.values() {
            if !pair.is_null() {
                props.insert(
                    format!("{group}{}", pair.get_key()),
                    pair.get_value_string().unwrap_or_default(),
                );
            }
        }
    }
}

/// Java private static final nested class `Pair`.
#[derive(Clone, Debug)]
struct Pair {
    /// Java private final `key`.
    key: i32,
    /// Java private final `stringValue`.
    string_value: Option<String>,
    /// Java private final `number`.
    number: Option<Number>,
    /// Java private final `numericType`.
    numeric_type: Option<Type>,
}

impl Pair {
    /// Java private `Pair(int, String, EtomoNumber.Type)`.
    fn new(key: i32, value: Option<&str>, etomo_number_type: Option<Type>) -> Pair {
        match etomo_number_type {
            None => Pair {
                key,
                string_value: value.map(str::to_owned),
                number: None,
                numeric_type: None,
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

    /// Java `getValueString()`.
    fn get_value_string(&self) -> Option<String> {
        if self.numeric_type.is_none() {
            return self.string_value.clone();
        }
        self.number.map(|number| number.to_string())
    }

    /// Java `isNull()`.
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

    /// Java `getKey()`.
    fn get_key(&self) -> i32 {
        self.key
    }
}

/// Java `RowKey.FIRST_KEY`.
const FIRST_KEY: &str = "First";
/// Java `RowKey.LAST_KEY`.
const LAST_KEY: &str = "Last";
/// Java `RowKey.DEFAULT_START_KEY`.
const DEFAULT_START_KEY: i32 = 0;

/// Java private static final nested class `RowKey`.
#[derive(Clone, Debug)]
struct RowKey {
    /// Java private final `firstKey`.
    first_key: EtomoNumber,
    /// Java private final `lastKey`.
    last_key: EtomoNumber,
    /// Java private `debug`.
    debug: bool,
    /// Java private `startKey`.
    start_key: i32,
}

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

    /// Java private `RowKey(RowKey)`.
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

    /// Java `getLastKey()`.
    fn get_last_key(&self) -> i32 {
        self.last_key.get_int()
    }

    /// Java `adjustFirstLastKeys(int)`.
    fn adjust_first_last_keys(&mut self, key: i32) {
        if self.first_key.is_null() || self.first_key.gt_int(key) {
            self.first_key.set_int(key);
        }
        if self.last_key.is_null() || self.last_key.lt_int(key) {
            self.last_key.set_int(key);
        }
    }

    /// Java `getFirstKey()`.
    fn get_first_key(&self) -> i32 {
        self.first_key.get_int()
    }

    /// Java `store(Properties, String)`.
    ///
    /// Upstream bug fixed in translation (DoubleKeyList.java:250), as in
    /// `IntKeyList`: Java throws `IllegalStateException` for a first key greater
    /// than the last, which aborts the data-file save; the message is printed and
    /// the two keys are not stored.
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

    /// Java `load(Properties, String)`.
    fn load(&mut self, props: &BTreeMap<String, String>, prepend: &str) {
        self.first_key.load_with_prepend(props, Some(prepend));
        self.last_key.load_with_prepend(props, Some(prepend));
        if self.debug {
            println!("firstKey={},lastKey={}", self.first_key, self.last_key);
        }
    }

    /// Java `remove(Properties, String)`.
    fn remove(&self, props: &mut BTreeMap<String, String>, prepend: &str) {
        self.first_key.remove_with_prepend(props, Some(prepend));
        self.last_key.remove_with_prepend(props, Some(prepend));
    }
}
