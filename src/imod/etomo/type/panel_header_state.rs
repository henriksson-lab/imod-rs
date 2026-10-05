//! `IMOD/Etomo/src/etomo/type/PanelHeaderState.java`.
//!
//! The open/closed, advanced/basic and more/less state of one panel header, stored as
//! `<prepend>.<group>.OpenClose` etc.
//!
//! Screen states are held by the managers and read from process threads as well as
//! the event dispatch thread, so each mutable field carries its own lock and every
//! method takes `&self`.

use std::collections::BTreeMap;
use std::sync::Mutex;

use super::const_etomo_number::java_lang_string_matches_whitespace;
use super::const_panel_header_state::ConstPanelHeaderState;
use crate::imod::etomo::storage::storable::{Storable, StorableValue};
use crate::imod::etomo::util::utilities;

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";
/// Java package-private static `KEY`.
pub(crate) const KEY: &str = "Header";
/// Java private static `OPEN_CLOSE_NAME`.
const OPEN_CLOSE_NAME: &str = "OpenClose";
/// Java private static `ADVANCED_BASIC_NAME`.
const ADVANCED_BASIC_NAME: &str = "AdvancedBasic";
/// Java private static `MORE_LESS_NAME`.
const MORE_LESS_NAME: &str = "MoreLess";

/// Java `PanelHeaderState`.
#[derive(Debug)]
pub struct PanelHeaderState {
    /// Java field `group`.
    group: String,
    /// Java field `openCloseState`, initialised to null.
    open_close_state: Mutex<Option<String>>,
    /// Java field `advancedBasicState`, initialised to null.
    advanced_basic_state: Mutex<Option<String>>,
    /// Java field `moreLessState`, initialised to null.
    more_less_state: Mutex<Option<String>>,
    /// Java field `debug`, initialised to false.
    debug: Mutex<bool>,
}

impl PanelHeaderState {
    /// Java `PanelHeaderState(String)`.
    pub fn new(group: &str) -> PanelHeaderState {
        PanelHeaderState {
            group: group.to_string(),
            open_close_state: Mutex::new(None),
            advanced_basic_state: Mutex::new(None),
            more_less_state: Mutex::new(None),
            debug: Mutex::new(false),
        }
    }

    /// Java `getMemberVariables`: `Utilities.getClassString(getClass()) + "[group:" +
    /// group + ",openCloseState:" + openCloseState + "]"`.
    pub fn get_member_variables(&self) -> String {
        format!(
            "{}[group:{},openCloseState:{}]",
            utilities::get_extension(Some("class etomo.type.PanelHeaderState")).unwrap_or_default(),
            self.group,
            self.open_close_state
                .lock()
                .unwrap()
                .as_deref()
                .unwrap_or("null")
        )
    }

    /// Java package-private `set(PanelHeaderState)`.
    pub(crate) fn set(&self, input: &PanelHeaderState) {
        let open_close_state = input.open_close_state.lock().unwrap().clone();
        let advanced_basic_state = input.advanced_basic_state.lock().unwrap().clone();
        let more_less_state = input.more_less_state.lock().unwrap().clone();
        *self.open_close_state.lock().unwrap() = open_close_state;
        *self.advanced_basic_state.lock().unwrap() = advanced_basic_state;
        *self.more_less_state.lock().unwrap() = more_less_state;
    }

    /// Java `setDebug`.
    pub fn set_debug(&self, input: bool) {
        *self.debug.lock().unwrap() = input;
    }

    /// Java private `getGroup`.
    fn get_group(&self, prepend: Option<&str>, key: &str) -> String {
        match prepend {
            None => format!("{}.", key),
            Some(prepend) if java_lang_string_matches_whitespace(prepend) => format!("{}.", key),
            Some(prepend) => format!("{}.{}.", prepend, key),
        }
    }

    /// Java `store(Properties)`.
    pub fn store(&self, props: &mut BTreeMap<String, String>) {
        self.store_with_prepend(props, "");
    }

    /// Java `store(Properties, String)`.
    pub fn store_with_prepend(&self, props: &mut BTreeMap<String, String>, prepend: &str) {
        let group = self.get_group(Some(prepend), &self.group);
        if let Some(open_close_state) = self.open_close_state.lock().unwrap().as_ref() {
            props.insert(
                format!("{}{}", group, OPEN_CLOSE_NAME),
                open_close_state.clone(),
            );
        }
        if let Some(advanced_basic_state) = self.advanced_basic_state.lock().unwrap().as_ref() {
            props.insert(
                format!("{}{}", group, ADVANCED_BASIC_NAME),
                advanced_basic_state.clone(),
            );
        }
        if let Some(more_less_state) = self.more_less_state.lock().unwrap().as_ref() {
            props.insert(
                format!("{}{}", group, MORE_LESS_NAME),
                more_less_state.clone(),
            );
        }
    }

    /// Java `load(Properties)`.
    pub fn load(&self, props: &BTreeMap<String, String>) {
        self.load_with_prepend(props, "");
    }

    /// Java `load(Properties, String)`.
    pub fn load_with_prepend(&self, props: &BTreeMap<String, String>, prepend: &str) {
        let group = self.get_group(Some(prepend), &self.group);
        *self.open_close_state.lock().unwrap() =
            props.get(&format!("{}{}", group, OPEN_CLOSE_NAME)).cloned();
        *self.advanced_basic_state.lock().unwrap() = props
            .get(&format!("{}{}", group, ADVANCED_BASIC_NAME))
            .cloned();
        *self.more_less_state.lock().unwrap() =
            props.get(&format!("{}{}", group, MORE_LESS_NAME)).cloned();
    }

    /// Java `load(Properties, String, String)`.  Load with key instead of this.group.
    pub fn load_with_key(&self, props: &BTreeMap<String, String>, prepend: &str, key: &str) {
        let group = self.get_group(Some(prepend), key);
        *self.open_close_state.lock().unwrap() =
            props.get(&format!("{}{}", group, OPEN_CLOSE_NAME)).cloned();
        *self.advanced_basic_state.lock().unwrap() = props
            .get(&format!("{}{}", group, ADVANCED_BASIC_NAME))
            .cloned();
        *self.more_less_state.lock().unwrap() =
            props.get(&format!("{}{}", group, MORE_LESS_NAME)).cloned();
    }

    /// Java `reset`.
    pub fn reset(&self) {
        *self.open_close_state.lock().unwrap() = None;
        *self.advanced_basic_state.lock().unwrap() = None;
        *self.more_less_state.lock().unwrap() = None;
    }

    /// Java `isNull`.
    pub fn is_null(&self) -> bool {
        self.open_close_state.lock().unwrap().is_none()
            && self.advanced_basic_state.lock().unwrap().is_none()
            && self.more_less_state.lock().unwrap().is_none()
    }

    /// Java final `setAdvancedBasicState`.
    pub fn set_advanced_basic_state(&self, advanced_basic_state: Option<&str>) {
        *self.advanced_basic_state.lock().unwrap() = advanced_basic_state.map(|s| s.to_string());
    }

    /// Java final `setMoreLessState`.
    pub fn set_more_less_state(&self, more_less_state: Option<&str>) {
        *self.more_less_state.lock().unwrap() = more_less_state.map(|s| s.to_string());
    }

    /// Java final `setOpenCloseState`.
    pub fn set_open_close_state(&self, open_close_state: Option<&str>) {
        *self.open_close_state.lock().unwrap() = open_close_state.map(|s| s.to_string());
    }
}

impl ConstPanelHeaderState for PanelHeaderState {
    /// Java final `getAdvancedBasicState`.
    fn get_advanced_basic_state(&self) -> Option<String> {
        self.advanced_basic_state.lock().unwrap().clone()
    }

    /// Java final `getMoreLessState`.
    fn get_more_less_state(&self) -> Option<String> {
        self.more_less_state.lock().unwrap().clone()
    }

    /// Java final `getOpenCloseState`.
    fn get_open_close_state(&self) -> Option<String> {
        self.open_close_state.lock().unwrap().clone()
    }
}

/// Java `Storable`.
impl Storable for PanelHeaderState {
    fn store(&self, properties: &mut BTreeMap<String, String>) {
        PanelHeaderState::store(self, properties);
    }

    fn store_with_prepend(&self, properties: &mut BTreeMap<String, String>, prepend: &str) {
        PanelHeaderState::store_with_prepend(self, properties, prepend);
    }

    fn load(&self, properties: &mut BTreeMap<String, String>) {
        PanelHeaderState::load(self, properties);
    }

    fn load_with_prepend(&self, properties: &mut BTreeMap<String, String>, prepend: &str) {
        PanelHeaderState::load_with_prepend(self, properties, prepend);
    }
}

/// Java `toString`.
impl std::fmt::Display for PanelHeaderState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[group={},openCloseState={},advancedBasicState={},moreLessState={}]",
            self.group,
            self.open_close_state
                .lock()
                .unwrap()
                .as_deref()
                .unwrap_or("null"),
            self.advanced_basic_state
                .lock()
                .unwrap()
                .as_deref()
                .unwrap_or("null"),
            self.more_less_state
                .lock()
                .unwrap()
                .as_deref()
                .unwrap_or("null")
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn store_load_round_trip() {
        let state = PanelHeaderState::new("FinalStack.Newst.Header");
        state.set_open_close_state(Some("open"));
        state.set_more_less_state(Some("less"));
        let mut props = BTreeMap::new();
        state.store_with_prepend(&mut props, "ScreenStateA");
        assert_eq!(
            props
                .get("ScreenStateA.FinalStack.Newst.Header.OpenClose")
                .unwrap(),
            "open"
        );
        assert_eq!(
            props
                .get("ScreenStateA.FinalStack.Newst.Header.MoreLess")
                .unwrap(),
            "less"
        );
        assert_eq!(props.len(), 2);
        let loaded = PanelHeaderState::new("FinalStack.Newst.Header");
        loaded.load_with_prepend(&props, "ScreenStateA");
        assert_eq!(loaded.get_open_close_state().as_deref(), Some("open"));
        assert!(loaded.get_advanced_basic_state().is_none());
        assert!(!loaded.is_null());
        loaded.reset();
        assert!(loaded.is_null());
        assert_eq!(
            state.get_member_variables(),
            "PanelHeaderState[group:FinalStack.Newst.Header,openCloseState:open]"
        );
    }
}
