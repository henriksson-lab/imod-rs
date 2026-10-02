//! `IMOD/Etomo/src/etomo/ui/BooleanFlagExtension.java`.
//!
//! Chooses a flag type and informs flag displays.  An EDT object: built as `Rc<Self>`
//! by the field it extends, and every method takes `&self`.
//!
//! The origin is the field that owns this extension, so it is held as a `Weak` (Java's
//! collector handles the cycle).  `update` after the origin is gone does nothing, where
//! Java could not reach this object at all.

use std::cell::{Cell, RefCell};
use std::rc::{Rc, Weak};

use super::boolean_flag_origin::BooleanFlagOrigin;
use super::flag_display::FlagDisplay;
use super::flag_type::{self, FlagType};

/// Java `BooleanFlagExtension`.
pub struct BooleanFlagExtension {
    /// Java private final `flagOrigin`.
    flag_origin: Weak<dyn BooleanFlagOrigin>,
    /// Java private `flagDisplays`, initialised to null.
    flag_displays: RefCell<Option<Vec<Rc<dyn FlagDisplay>>>>,
    /// Java private `warningEnabled`, initialised to false.
    warning_enabled: Cell<bool>,
    /// Java private `warningValue`, initialised to false.
    warning_value: Cell<bool>,
}

impl BooleanFlagExtension {
    /// Java `BooleanFlagExtension(BooleanFlagOrigin)`.
    pub fn new(flag_origin: Rc<dyn BooleanFlagOrigin>) -> Rc<BooleanFlagExtension> {
        Rc::new(BooleanFlagExtension {
            flag_origin: Rc::downgrade(&flag_origin),
            flag_displays: RefCell::new(None),
            warning_enabled: Cell::new(false),
            warning_value: Cell::new(false),
        })
    }

    /// Java `enableWarning(boolean)`.
    pub fn enable_warning(&self, value: bool) {
        self.warning_enabled.set(true);
        self.warning_value.set(value);
    }

    /// Java `disableWarning()`.
    pub fn disable_warning(&self) {
        self.warning_enabled.set(false);
    }

    /// Java `update()`.
    pub fn update(&self) {
        let Some(flag_origin) = self.flag_origin.upgrade() else {
            return;
        };
        let mut flag_type: Option<&'static FlagType> = None;
        if self.warning_enabled.get() && flag_origin.is_selected() == self.warning_value.get() {
            flag_type = Some(&flag_type::WARNING);
        }
        // The list is copied before it is walked: a display may re-enter this extension.
        let flag_displays = self.flag_displays.borrow().clone();
        if let Some(flag_displays) = flag_displays {
            for flag_display in flag_displays.iter() {
                flag_display.set_flag(flag_type);
            }
        }
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
}
