//! `IMOD/Etomo/src/etomo/type/FieldProperties.java`.
//!
//! Saves multiple field characteristics to Properties.

use std::collections::BTreeMap;

use crate::imod::etomo::util::utilities;

/// Java private static final `DELIMETER`.
const DELIMETER: &str = ".";

/// Java package-private `static class PropInstr`.  Use REMOVE to permanently remove a
/// property that's no longer needed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PropInstr {
    /// Java `STORE = new PropInstr("store")`.
    Store,
    /// Java `REMOVE = new PropInstr("remove")`.
    Remove,
}

/// Java `PropInstr.toString()`.
impl std::fmt::Display for PropInstr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            PropInstr::Store => "store",
            PropInstr::Remove => "remove",
        })
    }
}

/// Java private static nested `BooleanProp`.
#[derive(Clone, Debug)]
struct BooleanProp {
    /// Java private final `instr`.
    instr: PropInstr,
    /// Java private final `key`.
    key: Option<&'static str>,
    /// Java private `prop`, initially null.
    prop: Option<bool>,
}

impl BooleanProp {
    /// Java private `BooleanProp(PropInstr, String)`.
    fn new_with_key(instr: PropInstr, key: Option<&'static str>) -> BooleanProp {
        BooleanProp {
            instr,
            key,
            prop: None,
        }
    }

    /// Java private `BooleanProp(PropInstr)`.
    fn new(instr: PropInstr) -> BooleanProp {
        Self::new_with_key(instr, None)
    }

    /// Java private `set(boolean)`.
    fn set(&mut self, input: bool) {
        self.prop = Some(input);
    }

    /// Java private `get()`.
    fn get(&self) -> Option<bool> {
        self.prop
    }

    /// Java private `reset()`.
    fn reset(&mut self) {
        self.prop = None;
    }

    /// Java private `load(Properties, String)`.
    fn load(&mut self, props: &BTreeMap<String, String>, prepend: Option<&str>) {
        if self.instr == PropInstr::Store {
            self.prop = FieldProperties::load_prop(self.key, Some(props), prepend);
        } else if self.instr == PropInstr::Remove {
            self.prop = None;
        }
    }

    /// Java private `store(Properties, String)`.
    fn store(&self, props: &mut BTreeMap<String, String>, prepend: Option<&str>) {
        if self.instr == PropInstr::Store {
            FieldProperties::store_prop(self.prop, self.key, Some(props), prepend);
        } else if self.instr == PropInstr::Remove {
            FieldProperties::store_prop(None, self.key, Some(props), prepend);
        }
    }
}

/// Java `BooleanProp.toString()`.
impl std::fmt::Display for BooleanProp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.prop {
            Some(prop) => write!(f, "{prop}"),
            None => Ok(()),
        }
    }
}

/// Java package-private `final class FieldProperties`.
#[derive(Clone, Debug)]
pub struct FieldProperties {
    /// Java private final `key`.
    key: Option<String>,
    /// Java private final `selectedProp`.
    selected_prop: Option<BooleanProp>,
    /// Java private final `editableProp`.
    editable_prop: Option<BooleanProp>,
    /// Java private final `enabledProp`.
    enabled_prop: Option<BooleanProp>,
}

impl FieldProperties {
    /// Java private `FieldProperties(String, PropInstr, PropInstr, PropInstr)`.  A null
    /// PropInstr parameter omits the property.
    fn new(
        key: &str,
        selected_instr: Option<PropInstr>,
        editable_instr: Option<PropInstr>,
        enabled_instr: Option<PropInstr>,
    ) -> FieldProperties {
        // Remove leading and trailing .'s.
        FieldProperties {
            key: utilities::strip_repeating_string(Some(key), Some(DELIMETER)),
            selected_prop: selected_instr.map(BooleanProp::new),
            editable_prop: editable_instr
                .map(|instr| BooleanProp::new_with_key(instr, Some("Editable"))),
            enabled_prop: enabled_instr
                .map(|instr| BooleanProp::new_with_key(instr, Some("Enabled"))),
        }
    }

    /// Java package-private static `getToggleButtonInstance(String, PropInstr,
    /// PropInstr, PropInstr)`.
    pub fn get_toggle_button_instance(
        key: &str,
        selected_instr: Option<PropInstr>,
        editable_instr: Option<PropInstr>,
        enabled_instr: Option<PropInstr>,
    ) -> FieldProperties {
        Self::new(key, selected_instr, editable_instr, enabled_instr)
    }

    /// Java package-private static `getButtonInstance(String, PropInstr, PropInstr)`.
    pub fn get_button_instance(
        key: &str,
        editable_instr: Option<PropInstr>,
        enabled_instr: Option<PropInstr>,
    ) -> FieldProperties {
        Self::new(key, None, editable_instr, enabled_instr)
    }

    /// Java package-private `reset()`.  Turns off set, and sets values to null.
    /// Properties that match prepend and key are removed when set is false.
    pub fn reset(&mut self) {
        if let Some(prop) = &mut self.selected_prop {
            prop.reset();
        }
        if let Some(prop) = &mut self.editable_prop {
            prop.reset();
        }
        if let Some(prop) = &mut self.enabled_prop {
            prop.reset();
        }
    }

    /// Java package-private `isToggleButton()`.
    pub fn is_toggle_button(&self) -> bool {
        self.selected_prop.is_some()
    }

    /// Java `setSelectedProperty(boolean)`.
    pub fn set_selected_property(&mut self, selected: bool) {
        if let Some(prop) = &mut self.selected_prop {
            prop.set(selected);
        }
    }

    /// Java `getSelectedProperty()`.
    pub fn get_selected_property(&self) -> Option<bool> {
        self.selected_prop.as_ref().and_then(BooleanProp::get)
    }

    /// Java `setEditableProperty(boolean)`.
    pub fn set_editable_property(&mut self, editable: bool) {
        if let Some(prop) = &mut self.editable_prop {
            prop.set(editable);
        }
    }

    /// Java `getEditableProperty()`.
    pub fn get_editable_property(&self) -> Option<bool> {
        self.editable_prop.as_ref().and_then(BooleanProp::get)
    }

    /// Java `setEnabledProperty(boolean)`.
    pub fn set_enabled_property(&mut self, enabled: bool) {
        if let Some(prop) = &mut self.enabled_prop {
            prop.set(enabled);
        }
    }

    /// Java `getEnabledProperty()`.
    pub fn get_enabled_property(&self) -> Option<bool> {
        self.enabled_prop.as_ref().and_then(BooleanProp::get)
    }

    /// Java package-private `createPrepend(String)`.
    pub fn create_prepend(&self, prepend: Option<&str>) -> Option<String> {
        utilities::build_string(
            Some(&[
                utilities::strip_repeating_string(prepend, Some(DELIMETER)),
                self.key.clone(),
            ]),
            Some(DELIMETER),
        )
    }

    /// Java `load(Properties, String)`.
    pub fn load(&mut self, props: &BTreeMap<String, String>, prepend: Option<&str>) {
        // reset
        self.reset();
        let prepend = self.create_prepend(prepend);
        if let Some(prop) = &mut self.selected_prop {
            prop.load(props, prepend.as_deref());
        }
        if let Some(prop) = &mut self.editable_prop {
            prop.load(props, prepend.as_deref());
        }
        if let Some(prop) = &mut self.enabled_prop {
            prop.load(props, prepend.as_deref());
        }
    }

    /// Java `store(Properties, String)`.
    pub fn store(&self, props: &mut BTreeMap<String, String>, prepend: Option<&str>) {
        let prepend = self.create_prepend(prepend);
        if let Some(prop) = &self.selected_prop {
            prop.store(props, prepend.as_deref());
        }
        if let Some(prop) = &self.editable_prop {
            prop.store(props, prepend.as_deref());
        }
        if let Some(prop) = &self.enabled_prop {
            prop.store(props, prepend.as_deref());
        }
    }

    /// Java private static `load(String, Properties, String)`.
    fn load_prop(
        prop_key: Option<&str>,
        props: Option<&BTreeMap<String, String>>,
        prepend: Option<&str>,
    ) -> Option<bool> {
        let props = props?;
        let key = Self::build_complete_key(prop_key, prepend)?;
        let property = props.get(&key)?;
        // Boolean.parseBoolean
        Some(property.eq_ignore_ascii_case("true"))
    }

    /// Java private static `store(Boolean, String, Properties, String)`.
    fn store_prop(
        prop: Option<bool>,
        prop_key: Option<&str>,
        props: Option<&mut BTreeMap<String, String>>,
        prepend: Option<&str>,
    ) {
        let Some(props) = props else {
            return;
        };
        let Some(complete_key) = Self::build_complete_key(prop_key, prepend) else {
            return;
        };
        match prop {
            Some(prop) => {
                props.insert(complete_key, prop.to_string());
            }
            None => {
                props.remove(&complete_key);
            }
        }
    }

    /// Java private static `buildCompleteKey(String, String)`.
    fn build_complete_key(prop_key: Option<&str>, prepend: Option<&str>) -> Option<String> {
        utilities::build_string(
            Some(&[prepend.map(str::to_owned), prop_key.map(str::to_owned)]),
            Some(DELIMETER),
        )
    }
}
