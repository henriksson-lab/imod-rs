//! `IMOD/Etomo/src/etomo/ui/swing/ComponentStyleExtension.java`.
//!
//! A stateless singleton that changes a component's foreground (or background)
//! colour to show a flag.  Backgrounds are painting, which `jdk.rs` does not model;
//! those statements are comments.

use std::rc::Rc;

use crate::imod::etomo::jdk::{Color, JComponent};
use crate::imod::etomo::ui::flag_type::FlagType;

/// Java `static ComponentStyleExtension INSTANCE = new ComponentStyleExtension()`.
pub static INSTANCE: ComponentStyleExtension = ComponentStyleExtension {};

/// Java `final class ComponentStyleExtension`.
pub struct ComponentStyleExtension {}

impl ComponentStyleExtension {
    /// Java private `ComponentStyleExtension()`.
    #[allow(dead_code)]
    const fn new() -> ComponentStyleExtension {
        ComponentStyleExtension {}
    }

    /// Java `updateAppearance(Component, FlagType, Color, Color)`.  Changes the
    /// foreground color based on the flag type and whether the field is enabled.
    pub fn update_appearance(
        &self,
        component: Option<&Rc<JComponent>>,
        flag_type: Option<&'static FlagType>,
        default_foreground: Option<Color>,
        default_background: Option<Color>,
    ) {
        let Some(component) = component else {
            return;
        };
        // Remove the flag color when there is no flag or the component is disabled.
        match flag_type {
            Some(flag_type) if component.is_enabled() => {
                if !flag_type.background {
                    component.set_foreground(Some(flag_type.color));
                } else {
                    // Swing painting: component.setBackground(flagType.color).
                }
            }
            _ => {
                if let Some(default_foreground) = default_foreground {
                    component.set_foreground(Some(default_foreground));
                }
                if let Some(_default_background) = default_background {
                    // Swing painting: component.setBackground(defaultBackground).
                }
            }
        }
    }
}
