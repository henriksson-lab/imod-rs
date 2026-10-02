//! `IMOD/Etomo/src/etomo/ui/swing/SingleLineButton.java`.
//!
//! A `MultiLineButton` that always shows its label on one line, optionally
//! formatted as bold centred HTML.
//!
//! Java `extends MultiLineButton`: the superclass is field `base` (deref).
//! The overrides of `newButton`, `setupButton` and `setTextLabel` are
//! inherent methods here (so `ExpandButton`, a subclass, reuses them) and are
//! wired into [`MultiLineButtonVirtual`].  See `multi_line_button.rs` for the
//! construction order.
//!
//! Sizes are not modelled by `jdk.rs`: `setSize()`, `setSize(Dimension)`,
//! `getPreferredSize`, `setToPreferredSize()` and
//! `setToPreferredSize(Dimension)` only set preferred and maximum sizes, and
//! have no Rust counterpart; the size statements in the constructor,
//! `newButton` and `setTextLabel` are layout comments.

use std::ops::Deref;
use std::rc::Rc;

use crate::imod::etomo::jdk::JComponent;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::ui::ui_component::UIComponent;

use super::multi_line_button::{MultiLineButton, MultiLineButtonVirtual};
use super::swing_component::SwingComponent;

/// Java package-private `SingleLineButton`.
pub struct SingleLineButton {
    /// Java superclass `MultiLineButton`.
    pub base: MultiLineButton,
}

impl Deref for SingleLineButton {
    type Target = MultiLineButton;
    fn deref(&self) -> &MultiLineButton {
        &self.base
    }
}

impl MultiLineButtonVirtual for SingleLineButton {
    fn get_multi_line_button(&self) -> &MultiLineButton {
        &self.base
    }
    fn new_button(&self) -> Rc<JComponent> {
        SingleLineButton::new_button(self)
    }
    fn setup_button(&self, set_minimum_size: bool) {
        SingleLineButton::setup_button(self, set_minimum_size)
    }
    fn set_text_label(&self, text: Option<&str>) {
        SingleLineButton::set_text_label(self, text)
    }
}

impl SingleLineButton {
    /// The field part of Java
    /// `SingleLineButton(String, boolean, DialogType, boolean)`: its
    /// `super(label, toggleButton, dialogType, false, html, false, null)`
    /// up to `newButton()`.  A subclass (`ExpandButton`) embeds the result,
    /// runs `MultiLineButton::construct` and then
    /// [`SingleLineButton::constructor_body`].
    pub fn new_fields(
        label: Option<&str>,
        toggle_button: bool,
        dialog_type: Option<DialogType>,
        html: bool,
    ) -> SingleLineButton {
        SingleLineButton {
            base: MultiLineButton::new_fields(label, toggle_button, dialog_type, html, false, None),
        }
    }

    /// The body of Java `SingleLineButton(String, boolean, DialogType,
    /// boolean)` after `super(...)`.
    pub fn constructor_body(&self, label: Option<&str>, html: bool) {
        let _button = self.get_button();
        if label.is_some() {
            if html {
                // Swing layout: button.setPreferredSize(getPreferredSize()).
            } else {
                // Swing layout: button.setPreferredSize(
                //   UIUtilities.getPreferredSize(button, getUnformattedLabel())).
            }
        }
    }

    /// Java package-private `SingleLineButton(String, boolean, DialogType, boolean)`.
    pub fn new_string_boolean_dialog_type_boolean(
        label: Option<&str>,
        toggle_button: bool,
        dialog_type: Option<DialogType>,
        html: bool,
    ) -> Rc<SingleLineButton> {
        let instance = Rc::new(SingleLineButton::new_fields(
            label,
            toggle_button,
            dialog_type,
            html,
        ));
        MultiLineButton::construct(&instance, false);
        instance.constructor_body(label, html);
        instance
    }

    /// Java package-private `SingleLineButton()`.
    pub fn new_void() -> Rc<SingleLineButton> {
        Self::new_string_boolean_dialog_type_boolean(None, false, None, false)
    }

    /// Java package-private `SingleLineButton(String)`.
    pub fn new_string(label: Option<&str>) -> Rc<SingleLineButton> {
        Self::new_string_boolean_dialog_type_boolean(label, false, None, false)
    }

    /// Java package-private `SingleLineButton(String, boolean, DialogType)`.
    pub fn new_string_boolean_dialog_type(
        label: Option<&str>,
        toggle_button: bool,
        dialog_type: Option<DialogType>,
    ) -> Rc<SingleLineButton> {
        Self::new_string_boolean_dialog_type_boolean(label, toggle_button, dialog_type, false)
    }

    /// Java static `getHtmlInstance(String)`.
    pub fn get_html_instance(label: Option<&str>) -> Rc<SingleLineButton> {
        Self::new_string_boolean_dialog_type_boolean(label, false, None, true)
    }

    /// Java `newButton()` (overrides `MultiLineButton.newButton`).
    pub fn new_button(&self) -> Rc<JComponent> {
        let unformatted_label = self.get_unformatted_label();
        let label: Option<String> = if self.is_html() {
            self.format(unformatted_label.as_deref())
        } else {
            unformatted_label.clone()
        };
        let new_button: Rc<JComponent> = if self.is_toggle_button() {
            // Java `new JToggleButton(label)`; the stand-in holds no null text.
            JComponent::new_toggle_button(label.as_deref().unwrap_or(""))
        } else {
            JComponent::new_button(label.as_deref().unwrap_or(""))
        };
        if unformatted_label.is_some() {
            if self.is_html() {
                // Swing layout: newButton.setPreferredSize(newButton.getPreferredSize()).
            } else {
                // Swing layout: newButton.setPreferredSize(
                //   UIUtilities.getPreferredSize(newButton, unformattedLabel)).
            }
        }
        new_button
    }

    /// Java `setupButton(boolean)` (overrides `MultiLineButton.setupButton`).
    pub fn setup_button(&self, _set_minimum_size: bool) {
        // Virtual `setName` (ExpandButton overrides it): through the base's
        // dispatcher, not this type's trait method.
        self.base.set_name(self.get_unformatted_label().as_deref());
    }

    /// Java final `setTextLabel(String)` (overrides
    /// `MultiLineButton.setTextLabel`).
    pub fn set_text_label(&self, text: Option<&str>) {
        let button = self.get_button();
        if !self.is_html() {
            // Java `button.setText(text)`; the stand-in holds no null text.
            button.set_text(text.unwrap_or(""));
        } else {
            button.set_text(self.format(text).as_deref().unwrap_or(""));
        }
        if text.is_some() {
            if self.is_html() {
                // Swing layout: button.setPreferredSize(button.getPreferredSize()).
            } else {
                // Swing layout: button.setPreferredSize(UIUtilities.getPreferredSize(button, text)).
            }
        }
    }

    /// Java private final `format(String)`.
    fn format(&self, label: Option<&str>) -> Option<String> {
        let label = label?;
        if label.to_lowercase().starts_with("<html>") {
            return Some(label.to_owned());
        }
        Some(format!("<html><b><center>{}</center></b>", label))
    }
}

/// Java `SwingComponent.getComponent()`, inherited from `MultiLineButton`.
impl SwingComponent for SingleLineButton {
    fn get_component(&self) -> Rc<JComponent> {
        self.base.get_component()
    }
}

/// Java `UIComponent`, inherited from `MultiLineButton`.
impl UIComponent for SingleLineButton {
    fn get_ui_component(&self) -> &dyn SwingComponent {
        self
    }
    fn get_component(&self) -> Rc<JComponent> {
        self.base.get_component()
    }
}
