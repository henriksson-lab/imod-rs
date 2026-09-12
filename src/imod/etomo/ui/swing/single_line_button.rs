//! `IMOD/Etomo/src/etomo/ui/swing/SingleLineButton.java`.
#![allow(dead_code)]

use super::multi_line_button::{ButtonBoundary, MultiLineButton};
use super::panel::Dimension;
use super::ui_utilities::UiUtilities;
use crate::imod::etomo::r#type::dialog_type::DialogType;

/// Java package-private `SingleLineButton` with its inherited button state.
#[derive(Clone, Debug, PartialEq)]
pub struct SingleLineButton {
    pub multi_line_button: MultiLineButton,
}

impl SingleLineButton {
    /// Java `SingleLineButton()`.
    pub fn new() -> Self {
        Self::new_full(None, false, None, false)
    }
    /// Java `SingleLineButton(String)`.
    pub fn new_with_label(label: Option<&str>) -> Self {
        Self::new_full(label, false, None, false)
    }
    /// Java `SingleLineButton(String, boolean, DialogType)`.
    pub fn new_with_toggle_button(
        label: Option<&str>,
        toggle_button: bool,
        dialog_type: Option<DialogType>,
    ) -> Self {
        Self::new_full(label, toggle_button, dialog_type, false)
    }
    /// Java four-argument constructor.
    pub fn new_full(
        label: Option<&str>,
        toggle_button: bool,
        dialog_type: Option<DialogType>,
        html: bool,
    ) -> Self {
        let mut value = Self {
            multi_line_button: MultiLineButton::new_full(
                label,
                toggle_button,
                dialog_type,
                false,
                html,
                false,
                None,
            ),
        };
        value.setup_button(false);
        if label.is_some() {
            value
                .multi_line_button
                .button
                .abstract_button
                .preferred_size = Some(value.get_preferred_size());
        }
        value
    }
    /// Java `getHtmlInstance(String)`.
    pub fn get_html_instance(label: Option<&str>) -> Self {
        Self::new_full(label, false, None, true)
    }
    /// Java overridden `newButton()` at the native Swing construction boundary.
    pub fn new_button(&self) -> ButtonBoundary {
        let label = self.multi_line_button.get_unformatted_label();
        let mut button = ButtonBoundary {
            text: if self.multi_line_button.is_html() {
                Self::format(label)
            } else {
                label.map(str::to_owned)
            },
            ..Default::default()
        };
        if let Some(label) = label {
            button.abstract_button.preferred_size = Some(if self.multi_line_button.is_html() {
                self.multi_line_button
                    .button
                    .abstract_button
                    .preferred_size
                    .unwrap_or_default()
            } else {
                UiUtilities::get_preferred_size(&button.abstract_button, Some(label))
            });
        }
        button
    }
    /// Java overridden `setupButton(boolean)`.
    pub fn setup_button(&mut self, _set_minimum_size: bool) {
        let label = self
            .multi_line_button
            .get_unformatted_label()
            .map(str::to_owned);
        self.multi_line_button.set_name(label.as_deref());
    }
    /// Java overridden `setTextLabel(String)`.
    pub fn set_text_label(&mut self, text: Option<&str>) {
        self.multi_line_button.button.text = if self.multi_line_button.is_html() {
            Self::format(text)
        } else {
            text.map(str::to_owned)
        };
        if let Some(text) = text {
            self.multi_line_button.button.abstract_button.preferred_size =
                Some(if self.multi_line_button.is_html() {
                    self.multi_line_button
                        .button
                        .abstract_button
                        .preferred_size
                        .unwrap_or_default()
                } else {
                    UiUtilities::get_preferred_size(
                        &self.multi_line_button.button.abstract_button,
                        Some(text),
                    )
                });
        }
    }
    /// Java inherited `setText(String)` with virtual `setTextLabel` dispatch.
    pub fn set_text(&mut self, text: &str) {
        if !self.multi_line_button.manual_name {
            self.multi_line_button.set_name(Some(text));
        }
        self.multi_line_button.unformatted_label = Some(text.to_owned());
        self.set_text_label(Some(text));
    }
    /// Java private `format(String)`.
    fn format(label: Option<&str>) -> Option<String> {
        let label = label?;
        if label.to_lowercase().starts_with("<html>") {
            Some(label.to_owned())
        } else {
            Some(format!("<html><b><center>{label}</center></b>"))
        }
    }
    /// Java `setSize()`.
    pub fn set_size(&mut self) {
        let size = if self.multi_line_button.is_html()
            || self.multi_line_button.get_unformatted_label().is_none()
        {
            Dimension {
                width: 90,
                height: 27,
            }
        } else {
            self.get_preferred_size()
        };
        self.set_size_dimension(size);
    }
    /// Java overloaded `setSize(Dimension)`.
    pub fn set_size_dimension(&mut self, size: Dimension) {
        self.multi_line_button.button.abstract_button.preferred_size = Some(size);
        self.multi_line_button.button.abstract_button.maximum_size = Some(size);
    }
    /// Java overridden `getPreferredSize()`.
    pub fn get_preferred_size(&self) -> Dimension {
        let label = self.multi_line_button.get_unformatted_label();
        if self.multi_line_button.is_html() || label.is_none() {
            self.multi_line_button
                .button
                .abstract_button
                .preferred_size
                .unwrap_or_default()
        } else {
            UiUtilities::get_preferred_size(&self.multi_line_button.button.abstract_button, label)
        }
    }
    /// Java `setToPreferredSize()`.
    pub fn set_to_preferred_size(&mut self) {
        self.set_size_dimension(self.get_preferred_size());
    }
    /// Java overloaded `setToPreferredSize(Dimension)`.
    pub fn set_to_preferred_size_dimension(&mut self, size: Option<Dimension>) {
        if let Some(size) = size {
            self.set_size_dimension(size);
        } else {
            self.set_to_preferred_size();
        }
    }
    /// Java inherited Swing calls.
    pub fn add_action_listener(&mut self) {
        self.multi_line_button.add_action_listener();
    }
    pub fn set_tool_tip_text(&mut self, text: Option<&str>) {
        self.multi_line_button.set_tool_tip_text(text);
    }
    pub fn set_visible(&mut self, visible: bool) {
        self.multi_line_button.set_visible(visible);
    }
    pub fn get_component(&self) -> &ButtonBoundary {
        self.multi_line_button.get_component()
    }
    pub fn do_click(&mut self) {
        self.multi_line_button.do_click();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn constructor_uses_single_line_name_and_size() {
        let button = SingleLineButton::new_with_label(Some("Run process"));
        assert_eq!(button.multi_line_button.get_name(), Some("bn.run-process"));
        assert_eq!(button.multi_line_button.get_text(), Some("Run process"));
    }
    #[test]
    fn html_text_is_wrapped_once() {
        let mut button = SingleLineButton::get_html_instance(Some("Advanced"));
        button.set_text_label(Some("<HTML>Basic"));
        assert_eq!(
            button.multi_line_button.button.text.as_deref(),
            Some("<HTML>Basic")
        );
    }
}
