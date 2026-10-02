//! `IMOD/Etomo/src/etomo/ui/swing/ButtonStyleExtension.java`.
//!
//! A class to handle the appearance of buttons.  Contains CompleteIcons
//! corresponding to different types of flags.  Specifically for handling
//! icons, but setup may be overridden to modify the button in other ways
//! (such as size, background, or boundary).  For foreground text changes, and
//! editable and enabled settings use AppearanceExtension.
//!
//! Statelessness: this class is designed to be inherited by singletons which
//! can be used to give a large number of the same type of button the same
//! style.  Do not add any state information related to a button instance to
//! a singleton.
//!
//! Java `abstract class`: subclasses (`SingleLineButtonStyleExtension`,
//! `OpenCloseButtonStyleExtension`, `CloseButtonStyleExtension`,
//! `FileOpenButtonStyleExtension`, `ClearButtonStyleExtension`,
//! `HeaderButtonStyleExtension`, ...) embed this struct as field `base` and
//! implement [`ButtonStyleExtensionVirtual`]; a button holds its style as
//! `Rc<dyn ButtonStyleExtensionVirtual>` and calls `setup` virtually.  This
//! class never calls an overridable method itself, so it keeps no `this`.
//!
//! Icons are painting and sizes are layout, neither modelled by `jdk.rs`;
//! the calls into `CompleteIcon` are kept (that unit decides what they do),
//! the size computation is kept, and the Swing size/alignment setters are
//! comments.

use std::cell::Cell;
use std::rc::Rc;

use crate::imod::etomo::jdk::{Dimension, JComponent};
use crate::imod::etomo::ui::flag_type::FlagType;

use super::complete_icon::CompleteIcon;

/// Java private static final `PADDING_FOR_ICON_BUTTON`.
const PADDING_FOR_ICON_BUTTON: i32 = 5;

/// The overridable members of Java `ButtonStyleExtension`.
pub trait ButtonStyleExtensionVirtual {
    /// The embedded `ButtonStyleExtension`.
    fn get_button_style_extension(&self) -> &ButtonStyleExtension;

    /// Java `synchronized setup(AbstractButton, String, boolean)`: call this
    /// function to make the button conform to its style.
    fn setup(&self, button: Option<&Rc<JComponent>>, label: Option<&str>, debug: bool) {
        self.get_button_style_extension()
            .setup(button, label, debug)
    }

    /// Java public final `updateAppearance(AbstractButton, FlagType, boolean, boolean)`.
    fn update_appearance(
        &self,
        button: &Rc<JComponent>,
        flag_type: Option<&'static FlagType>,
        implement_toggle: bool,
        selected: bool,
    ) {
        self.get_button_style_extension().update_appearance(
            button,
            flag_type,
            implement_toggle,
            selected,
        )
    }
}

/// Java package-private abstract `ButtonStyleExtension`.
pub struct ButtonStyleExtension {
    /// Java final `textGap`.
    text_gap: bool,
    /// Java final `icon`.
    icon: Option<Rc<CompleteIcon>>,
    /// Java final `templateIcon`.
    template_icon: Option<Rc<CompleteIcon>>,
    /// Java final `errorIcon`.
    error_icon: Option<Rc<CompleteIcon>>,
    /// Java final `sizeFromImage`.
    size_from_image: bool,
    /// Java `preferredSize`.
    preferred_size: Cell<Option<Dimension>>,
}

impl ButtonStyleExtension {
    /// Java package-private
    /// `ButtonStyleExtension(boolean, CompleteIcon, CompleteIcon, CompleteIcon, Dimension, boolean)`.
    /// All CompleteIcon parameters may be set to null.
    pub fn new(
        text_gap: bool,
        icon: Option<Rc<CompleteIcon>>,
        template_icon: Option<Rc<CompleteIcon>>,
        error_icon: Option<Rc<CompleteIcon>>,
        preferred_size: Option<Dimension>,
        size_from_image: bool,
    ) -> ButtonStyleExtension {
        ButtonStyleExtension {
            text_gap,
            icon,
            template_icon,
            error_icon,
            size_from_image,
            preferred_size: Cell::new(preferred_size),
        }
    }

    /// Java `synchronized setup(AbstractButton, String, boolean)`.  The lock
    /// has no Rust counterpart: styles are only used on the EDT.
    pub fn setup(&self, button: Option<&Rc<JComponent>>, label: Option<&str>, _debug: bool) {
        let Some(button) = button else {
            return;
        };
        if let Some(icon) = &self.icon {
            icon.setup(Some(button));
        }
        // Handle an buttons with a label and an icon
        if self.text_gap && label.is_some() && (self.icon.is_some() || self.template_icon.is_some())
        {
            // Swing layout: button.setIconTextGap(10);
            // button.setHorizontalAlignment(SwingConstants.LEFT).
        }
        // Setting preferred size for buttons without labels.
        // If preferredSize wasn't set, get the preferredSize from the icons.
        if !self.text_gap
            && label.is_none()
            && self.preferred_size.get().is_none()
            && self.size_from_image
        {
            let mut icon_size: Option<Dimension> = None;
            if let Some(icon) = &self.icon {
                icon_size = self.max_icon_size(Some(icon), icon_size);
            }
            if let Some(template_icon) = &self.template_icon {
                icon_size = self.max_icon_size(Some(template_icon), icon_size);
            }
            // Java checks templateIcon a second time; the repeat changes nothing.
            if let Some(template_icon) = &self.template_icon {
                icon_size = self.max_icon_size(Some(template_icon), icon_size);
            }
            if let Some(error_icon) = &self.error_icon {
                icon_size = self.max_icon_size(Some(error_icon), icon_size);
            }
            // Set the preferredSize
            if let Some(mut icon_size) = icon_size {
                if icon_size.width > 0 && icon_size.height > 0 {
                    // Convert icon size to a button size.  (Java adds the
                    // padding to the Dimension object the CompleteIcon caches,
                    // so the icon's cached size grows too; Dimension is a value
                    // here, and only layout ever reads either.)
                    icon_size.width += PADDING_FOR_ICON_BUTTON;
                    icon_size.height += PADDING_FOR_ICON_BUTTON;
                    self.preferred_size.set(Some(icon_size));
                }
            }
        }
        // Set the preferredSize if available.
        if self.preferred_size.get().is_some() {
            // Swing layout: button.setPreferredSize(preferredSize);
            // button.setMaximumSize(preferredSize).
        }
    }

    /// Java private `maxIconSize(CompleteIcon, Dimension)`: return the maximum
    /// width and height, comparing completeIcon with iconSize.
    fn max_icon_size(
        &self,
        complete_icon: Option<&Rc<CompleteIcon>>,
        icon_size: Option<Dimension>,
    ) -> Option<Dimension> {
        let Some(complete_icon) = complete_icon else {
            return icon_size;
        };
        let Some(new_icon_size) = complete_icon.get_icon_size() else {
            return icon_size;
        };
        let Some(mut icon_size) = icon_size else {
            return Some(new_icon_size);
        };
        if new_icon_size.width > icon_size.width {
            icon_size.width = new_icon_size.width;
        }
        if new_icon_size.height > icon_size.height {
            icon_size.height = new_icon_size.height;
        }
        Some(icon_size)
    }

    /// Java public final `updateAppearance(AbstractButton, FlagType, boolean, boolean)`.
    /// Sets the appearance.  Can create a appearance of a toggle button.  Can
    /// change the icon in response to the flagType.
    pub fn update_appearance(
        &self,
        button: &Rc<JComponent>,
        flag_type: Option<&'static FlagType>,
        implement_toggle: bool,
        selected: bool,
    ) {
        let mut cur_icon: Option<&Rc<CompleteIcon>> = None;
        if let Some(flag_type) = flag_type {
            if flag_type.is_template() {
                cur_icon = self.template_icon.as_ref();
            } else if flag_type.is_error() {
                cur_icon = self.error_icon.as_ref();
            }
        }
        if cur_icon.is_none() {
            cur_icon = self.icon.as_ref();
        }
        if let Some(cur_icon) = cur_icon {
            cur_icon.setup(Some(button));
            if implement_toggle {
                cur_icon.set_selected_appearance(Some(button), selected);
            }
        }
    }
}
