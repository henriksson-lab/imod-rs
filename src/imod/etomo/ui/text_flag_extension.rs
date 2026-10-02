//! `IMOD/Etomo/src/etomo/ui/TextFlagExtension.java`.
//!
//! Tells one or more fields about a change in the flag type.  Flag types are template
//! error, error, and template.  The field can then change the display if it knows how
//! to react to the flag setting.  For instance a template flag causes some part of a
//! field to turn blue.  And error flag causes a field to turn red in the same way.
//! Replaces FlagExtension.
//!
//! An EDT object: built as `Rc<Self>` (the constructor registers `this` with the
//! origin), and every method takes `&self`.  The origin is the field that owns this
//! extension and holds it as a listener, so the origin is held here as a `Weak` (Java's
//! collector handles the cycle); `update` after the origin is gone does nothing, where
//! Java could not reach this object at all.

use std::cell::{Cell, RefCell};
use std::rc::{Rc, Weak};

use super::flag_display::FlagDisplay;
use super::flag_origin_listener::FlagOriginListener;
use super::flag_type::{self, FlagType};
use super::text_flag_origin::TextFlagOrigin;
use crate::imod::etomo::jdk::ItemEvent;

/// Java `TextFlagExtension`.
pub struct TextFlagExtension {
    /// Java private final `flagOrigin`.
    flag_origin: Weak<dyn TextFlagOrigin>,
    /// Java private `flagDisplays`, initialised to null.
    flag_displays: RefCell<Option<Vec<Rc<dyn FlagDisplay>>>>,
    /// Java private `finalFlagDisplays`, initialised to null.
    final_flag_displays: RefCell<Option<Vec<Rc<dyn FlagDisplay>>>>,
    /// Java private `flaggedTemplateValue`, initialised to null.
    flagged_template_value: RefCell<Option<String>>,
    /// Java private `flagErrors`, initialised to false.
    flag_errors: Cell<bool>,
    /// Java private `debug`, initialised to false.
    debug: Cell<bool>,
}

impl TextFlagExtension {
    /// Java `TextFlagExtension(TextFlagOrigin)`.
    pub fn new(flag_origin: Rc<dyn TextFlagOrigin>) -> Rc<TextFlagExtension> {
        let this = Rc::new(TextFlagExtension {
            flag_origin: Rc::downgrade(&flag_origin),
            flag_displays: RefCell::new(None),
            final_flag_displays: RefCell::new(None),
            flagged_template_value: RefCell::new(None),
            flag_errors: Cell::new(false),
            debug: Cell::new(false),
        });
        flag_origin.add_flag_origin_listener(this.clone() as Rc<dyn FlagOriginListener>);
        this
    }

    /// Java `setDebug(boolean)`.
    pub fn set_debug(&self, debug: bool) {
        self.debug.set(debug);
    }

    /// Java `update()`.
    pub fn update_void(&self) {
        self.update_boolean(false);
    }

    /// Java private `update(boolean)`.
    ///
    /// Template error:  matches  flaggedTemplateValue and not valid
    /// Error: does not match flaggedTemplateValue and not valid
    /// template: matches flaggedTemplateValue
    ///
    /// Appearance change precedence order: template error, error, template, standard
    ///
    /// `force` - call setFlag when no flag checking is in use.  Used after flags are
    /// cleared.
    fn update_boolean(&self, force: bool) {
        if !force && !self.flag_errors.get() && self.flagged_template_value.borrow().is_none() {
            return;
        }
        let Some(flag_origin) = self.flag_origin.upgrade() else {
            return;
        };
        let mut flag_type: Option<&'static FlagType> = None;
        let flagged_template_value = self.flagged_template_value.borrow().clone();
        let template_flag = flagged_template_value.is_some()
            && flag_origin.equals(flagged_template_value.as_deref());
        if self.flag_errors.get() && !flag_origin.is_valid() {
            if template_flag {
                flag_type = Some(&flag_type::TEMPLATE_ERROR);
            } else {
                flag_type = Some(&flag_type::ERROR);
            }
        } else if template_flag {
            flag_type = Some(&flag_type::TEMPLATE);
        }
        // Each list is copied before it is walked: a display may re-enter this
        // extension.
        let flag_displays = self.flag_displays.borrow().clone();
        if let Some(flag_displays) = flag_displays {
            for flag_display in flag_displays.iter() {
                flag_display.set_flag(flag_type);
            }
        }
        let final_flag_displays = self.final_flag_displays.borrow().clone();
        if let Some(final_flag_displays) = final_flag_displays {
            for flag_display in final_flag_displays.iter() {
                flag_display.set_flag(flag_type);
            }
        }
    }

    /// Java `getFlaggedTemplateValue()`.
    pub fn get_flagged_template_value(&self) -> Option<String> {
        self.flagged_template_value.borrow().clone()
    }

    /// Java `setFlagErrors()`.
    pub fn set_flag_errors(&self) {
        self.flag_errors.set(true);
    }

    /// Java `flagTemplate(String)`.  Change appearance when value matches the template
    /// value.
    pub fn flag_template(&self, template_value: Option<&str>) {
        let Some(template_value) = template_value else {
            return;
        };
        *self.flagged_template_value.borrow_mut() = Some(template_value.to_string());
    }

    /// Java `addFinalFlagDisplay(FlagDisplay)`.
    pub fn add_final_flag_display(&self, flag_display: Option<Rc<dyn FlagDisplay>>) {
        let Some(flag_display) = flag_display else {
            return;
        };
        let mut final_flag_displays = self.final_flag_displays.borrow_mut();
        if final_flag_displays.is_none() {
            *final_flag_displays = Some(Vec::new());
        }
        final_flag_displays.as_mut().unwrap().push(flag_display);
    }

    /// Java `addFlagDisplay(FlagDisplay)`.
    pub fn add_flag_display(&self, flag_display: Option<Rc<dyn FlagDisplay>>) {
        let Some(flag_display) = flag_display else {
            return;
        };
        let mut flag_displays = self.flag_displays.borrow_mut();
        if flag_displays.is_none() {
            *flag_displays = Some(Vec::new());
        }
        flag_displays.as_mut().unwrap().push(flag_display);
    }

    /// Java `clearFlags()`.
    pub fn clear_flags(&self) {
        *self.flagged_template_value.borrow_mut() = None;
        self.update_boolean(true);
    }
}

/// Java `implements FlagOriginListener`.
impl FlagOriginListener for TextFlagExtension {
    /// Java `itemStateChanged(ItemEvent)`.
    fn item_state_changed(&self, _event: &ItemEvent) {
        self.update_void();
    }

    /// Java `focusLost(FocusEvent)`.
    fn focus_lost(&self) {
        self.update_void();
    }

    /// Java `focusGained(FocusEvent)`.
    fn focus_gained(&self) {}
}
