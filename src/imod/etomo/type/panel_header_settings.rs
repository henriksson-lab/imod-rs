//! `IMOD/Etomo/src/etomo/type/PanelHeaderSettings.java`.

use std::collections::BTreeMap;

use super::const_etomo_number::java_lang_string_matches_whitespace;
use super::const_panel_header_settings::ConstPanelHeaderSettings;
use super::etomo_boolean2::EtomoBoolean2;

/// Java `rcsid`.
pub const RCSID: &str = "$Id:$";

/// Java private static final `OPEN_KEY`.
const OPEN_KEY: &str = "open";
/// Java private static final `ADVANCED_KEY`.
const ADVANCED_KEY: &str = "advanced";
/// Java private static final `MORE_KEY`.
const MORE_KEY: &str = "more";

/// Java package-private `final class PanelHeaderSettings implements
/// ConstPanelHeaderSettings`.
#[derive(Clone, Debug)]
pub struct PanelHeaderSettings {
    /// Java private final `name`.
    name: String,
    /// Java package-private `open`, initially null.
    open: Option<EtomoBoolean2>,
    /// Java package-private `advanced`, initially null.
    advanced: Option<EtomoBoolean2>,
    /// Java package-private `more`, initially null.
    more: Option<EtomoBoolean2>,
}

impl PanelHeaderSettings {
    /// Java package-private `PanelHeaderSettings(String)`.
    pub fn new(name: &str) -> PanelHeaderSettings {
        PanelHeaderSettings {
            name: name.to_owned(),
            open: None,
            advanced: None,
            more: None,
        }
    }

    /// Java private static `createPrepend(String, String)`.
    fn create_prepend(prepend: Option<&str>, name: &str) -> String {
        let Some(prepend) = prepend else {
            return name.to_owned();
        };
        if java_lang_string_matches_whitespace(prepend) {
            return name.to_owned();
        }
        if prepend.ends_with('.') {
            return format!("{}{}", prepend, name);
        }
        format!("{}.{}", prepend, name)
    }

    /// Java static `load(PanelHeaderSettings, String, Properties, String)`.  Attempt to
    /// get the properties specified by prepend and name.  If it doesn't exist, return
    /// null.  If it does, set them in instance (create instance if it doesn't exist).
    /// Return the instance.
    pub fn load_instance(
        instance: Option<PanelHeaderSettings>,
        name: &str,
        props: &BTreeMap<String, String>,
        prepend: Option<&str>,
    ) -> Option<PanelHeaderSettings> {
        let prepend = Self::create_prepend(prepend, name);
        match instance {
            Some(mut instance) => {
                instance.open =
                    EtomoBoolean2::load_instance(instance.open.take(), OPEN_KEY, props, Some(&prepend));
                instance.advanced = EtomoBoolean2::load_instance(
                    instance.advanced.take(),
                    ADVANCED_KEY,
                    props,
                    Some(&prepend),
                );
                instance.more =
                    EtomoBoolean2::load_instance(instance.more.take(), MORE_KEY, props, Some(&prepend));
                if instance.open.is_none() && instance.advanced.is_none() && instance.more.is_none()
                {
                    return None;
                }
                Some(instance)
            }
            None => {
                let open = EtomoBoolean2::load_instance(None, OPEN_KEY, props, Some(&prepend));
                let advanced =
                    EtomoBoolean2::load_instance(None, ADVANCED_KEY, props, Some(&prepend));
                let more = EtomoBoolean2::load_instance(None, MORE_KEY, props, Some(&prepend));
                if open.is_none() && advanced.is_none() && more.is_none() {
                    return None;
                }
                let mut instance = PanelHeaderSettings::new(name);
                instance.open = open;
                instance.advanced = advanced;
                instance.more = more;
                Some(instance)
            }
        }
    }

    /// Java `reset()`.
    pub fn reset(&mut self) {
        if let Some(open) = &mut self.open {
            open.reset();
        }
        if let Some(advanced) = &mut self.advanced {
            advanced.reset();
        }
        if let Some(more) = &mut self.more {
            more.reset();
        }
    }

    /// Java `set(ConstPanelHeaderSettings)`.
    pub fn set(&mut self, input: &dyn ConstPanelHeaderSettings) {
        if !input.is_open_null() {
            if self.open.is_none() {
                self.open = Some(EtomoBoolean2::new_with_name(OPEN_KEY));
            }
            self.open.as_mut().unwrap().set_boolean(input.is_open());
        }
        if !input.is_advanced_null() {
            if self.advanced.is_none() {
                self.advanced = Some(EtomoBoolean2::new_with_name(ADVANCED_KEY));
            }
            self.advanced
                .as_mut()
                .unwrap()
                .set_boolean(input.is_advanced());
        }
        if !input.is_more_null() {
            if self.more.is_none() {
                self.more = Some(EtomoBoolean2::new_with_name(MORE_KEY));
            }
            self.more.as_mut().unwrap().set_boolean(input.is_more());
        }
    }

    /// Java `load(Properties, String)`.
    pub fn load(&mut self, props: &BTreeMap<String, String>, prepend: Option<&str>) {
        let prepend = Self::create_prepend(prepend, &self.name);
        self.open = EtomoBoolean2::load_instance(self.open.take(), OPEN_KEY, props, Some(&prepend));
        self.advanced =
            EtomoBoolean2::load_instance(self.advanced.take(), ADVANCED_KEY, props, Some(&prepend));
        self.more = EtomoBoolean2::load_instance(self.more.take(), MORE_KEY, props, Some(&prepend));
    }

    /// Java `store(Properties, String)`.
    pub fn store(&self, props: &mut BTreeMap<String, String>, prepend: Option<&str>) {
        let prepend = Self::create_prepend(prepend, &self.name);
        match &self.open {
            Some(open) if !open.is_null() => open.store_with_prepend(props, Some(&prepend)),
            _ => EtomoBoolean2::remove(OPEN_KEY, props, Some(&prepend)),
        }
        match &self.advanced {
            Some(advanced) if !advanced.is_null() => {
                advanced.store_with_prepend(props, Some(&prepend))
            }
            _ => EtomoBoolean2::remove(ADVANCED_KEY, props, Some(&prepend)),
        }
        match &self.more {
            Some(more) if !more.is_null() => more.store_with_prepend(props, Some(&prepend)),
            _ => EtomoBoolean2::remove(MORE_KEY, props, Some(&prepend)),
        }
    }

    /// Java `remove(Properties, String)`.  Upstream bug fixed in translation
    /// (`PanelHeaderSettings.java:110-123`): the source calls `store` on a non-null
    /// `advanced` and `more`, re-writing the properties this method exists to remove;
    /// here they are removed like `open` (`BUGS.md`).
    pub fn remove(&self, props: &mut BTreeMap<String, String>, prepend: Option<&str>) {
        let prepend = Self::create_prepend(prepend, &self.name);
        match &self.open {
            Some(open) => open.remove_with_prepend(props, Some(&prepend)),
            None => EtomoBoolean2::remove(OPEN_KEY, props, Some(&prepend)),
        }
        match &self.advanced {
            Some(advanced) => advanced.remove_with_prepend(props, Some(&prepend)),
            None => EtomoBoolean2::remove(ADVANCED_KEY, props, Some(&prepend)),
        }
        match &self.more {
            Some(more) => more.remove_with_prepend(props, Some(&prepend)),
            None => EtomoBoolean2::remove(MORE_KEY, props, Some(&prepend)),
        }
    }
}

impl ConstPanelHeaderSettings for PanelHeaderSettings {
    /// Java `isOpenNull()`.
    fn is_open_null(&self) -> bool {
        self.open.as_ref().is_none_or(|open| open.is_null())
    }

    /// Java `isOpen()`.
    fn is_open(&self) -> bool {
        self.open.as_ref().is_some_and(|open| open.is())
    }

    /// Java `isAdvancedNull()`.
    fn is_advanced_null(&self) -> bool {
        self.advanced.as_ref().is_none_or(|advanced| advanced.is_null())
    }

    /// Java `isAdvanced()`.
    fn is_advanced(&self) -> bool {
        self.advanced.as_ref().is_some_and(|advanced| advanced.is())
    }

    /// Java `isMoreNull()`.
    fn is_more_null(&self) -> bool {
        self.more.as_ref().is_none_or(|more| more.is_null())
    }

    /// Java `isMore()`.
    fn is_more(&self) -> bool {
        self.more.as_ref().is_some_and(|more| more.is())
    }
}
