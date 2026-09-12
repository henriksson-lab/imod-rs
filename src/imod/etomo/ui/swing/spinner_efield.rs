//! `IMOD/Etomo/src/etomo/ui/swing/SpinnerEfield.java`.
#![allow(dead_code)]
use super::appearance_extension::{AppearanceExtension, ComponentBoundary};
use crate::imod::etomo::storage::autodoc::autodoc_tokenizer::SEPARATOR_CHAR;
use crate::imod::etomo::util::utilities;
use std::cell::RefCell;
use std::rc::Rc;
/// Java package-private final `SpinnerEfield`.
pub struct SpinnerEfield {
    pub value: i32,
    pub minimum: i32,
    pub maximum: i32,
    pub default_value: i32,
    pub default_minimum: i32,
    pub label: Option<String>,
    pub root_panel: bool,
    pub spinner_name: Option<String>,
    pub spinner_enabled: bool,
    pub label_enabled: bool,
    pub spinner_visible: bool,
    pub root_visible: bool,
    pub preferred_width: Option<i32>,
    pub change_listener_count: usize,
    pub focus_listener_count: usize,
    pub appearance_extension: Option<AppearanceExtension>,
    pub component_boundary: Rc<RefCell<ComponentBoundary>>,
    pub identity: usize,
}
impl SpinnerEfield {
    /// Java private `SpinnerEfield(String,int,int,int)`.
    fn new(label: Option<&str>, value: i32, minimum: i32, maximum: i32) -> Self {
        let component_boundary = Rc::new(RefCell::new(ComponentBoundary::default()));
        let mut value_out = Self {
            value,
            minimum,
            maximum,
            default_value: value,
            default_minimum: minimum,
            label: label.map(str::to_owned),
            root_panel: label.is_some(),
            spinner_name: None,
            spinner_enabled: true,
            label_enabled: true,
            spinner_visible: true,
            root_visible: true,
            preferred_width: None,
            change_listener_count: 0,
            focus_listener_count: 0,
            appearance_extension: None,
            component_boundary,
            identity: 0,
        };
        value_out.identity = (&value_out as *const Self) as usize;
        value_out.set_name();
        value_out
    }
    /// Java `getInstance(String,int,int,int)`.
    pub fn get_instance(label: Option<&str>, value: i32, minimum: i32, maximum: i32) -> Self {
        let mut instance = Self::new(label, value, minimum, maximum);
        instance.create_panel();
        instance
    }
    /// Java `createPanel()`.
    fn create_panel(&mut self) {
        if self.root_panel {}
    }
    /// Java `getComponent()` represented by its source return branch.
    pub fn get_component_is_root_panel(&self) -> bool {
        self.root_panel
    }
    /// Java `setName()`.
    fn set_name(&mut self) {
        if let Some(label) = &self.label {
            self.spinner_name = utilities::convert_label_to_name(Some(label), true)
                .map(|name| format!("sp{SEPARATOR_CHAR}{name}"));
        }
    }
    /// Java `addChangeListener(ChangeListener)`.
    pub fn add_change_listener(&mut self) {
        self.change_listener_count += 1;
    }
    /// Java `addFocusListener(FocusListener)`.
    pub fn add_focus_listener(&mut self) {
        self.focus_listener_count += 1;
    }
    /// Java `equalsSource(EventObject)` represented by source identity.
    pub fn equals_source(&self, source_identity: Option<usize>) -> bool {
        source_identity == Some(self.identity)
    }
    /// Java `setMaximum(Integer)`.
    pub fn set_maximum(&mut self, maximum: Option<i32>) {
        if let Some(maximum) = maximum {
            self.maximum = maximum;
        }
    }
    /// Java `adjustToMaximum()`.
    pub fn adjust_to_maximum(&mut self) {
        if self.value > self.maximum {
            self.value = self.maximum;
        }
        if self.minimum > self.maximum {
            self.minimum = if self.default_minimum <= self.maximum {
                self.default_minimum
            } else {
                self.maximum
            };
        } else if self.minimum != self.default_minimum
            && self.default_minimum <= self.maximum
            && self.default_minimum <= self.value
        {
            self.minimum = self.default_minimum;
        }
        if self.value < self.minimum {
            self.value = self.minimum;
        }
    }
    /// Java `setText(Number)`.
    pub fn set_text_number(&mut self, number: i32) {
        self.value = number;
    }
    /// Java `setText(String)`.
    pub fn set_text(&mut self, string: Option<&str>) {
        let value = string.map_or(Some(self.default_value), |string| {
            string.trim().parse().ok()
        });
        if let Some(value) = value {
            if value >= self.minimum && value <= self.maximum {
                self.value = value;
            }
        }
    }
    /// Java `getText()`.
    pub fn get_text(&self) -> String {
        self.value.to_string()
    }
    /// Java `createAppearanceExtension()`.
    fn create_appearance_extension(&mut self) {
        if self.appearance_extension.is_none() {
            let mut extension = AppearanceExtension::new(self.component_boundary.clone());
            extension.set_allow_foreground_change_on_error(false);
            self.appearance_extension = Some(extension);
        }
    }
    /// Java `setEnabled(boolean)`.
    pub fn set_enabled(&mut self, enabled: bool) {
        if let Some(extension) = &mut self.appearance_extension {
            extension.set_enabled(enabled);
            self.spinner_enabled = extension.is_enabled();
        } else {
            self.spinner_enabled = enabled;
            self.component_boundary.borrow_mut().enabled = enabled;
        }
        if self.label.is_some() {
            self.label_enabled = self.is_enabled();
        }
    }
    /// Java `isEnabled()`.
    pub fn is_enabled(&self) -> bool {
        self.appearance_extension
            .as_ref()
            .map_or(self.spinner_enabled, AppearanceExtension::is_enabled)
    }
    /// Java `setPreferredWidth(int)`.
    pub fn set_preferred_width(&mut self, width: i32) {
        self.preferred_width = Some(width);
    }
    /// Java `setEditable(boolean)`.
    pub fn set_editable(&mut self, editable: bool) {
        if editable && self.appearance_extension.is_none() {
            return;
        }
        self.create_appearance_extension();
        self.appearance_extension
            .as_mut()
            .unwrap()
            .set_editable(editable);
    }
    /// Java `isVisible()`.
    pub fn is_visible(&self) -> bool {
        if self.root_panel {
            self.root_visible
        } else {
            self.spinner_visible
        }
    }
    /// Java `setVisible(boolean)`.
    pub fn set_visible(&mut self, visible: bool) {
        if self.root_panel {
            self.root_visible = visible;
        } else {
            self.spinner_visible = visible;
        }
    }
    /// Java `isEditable()`.
    pub fn is_editable(&self) -> bool {
        self.appearance_extension
            .as_ref()
            .map_or(true, AppearanceExtension::is_editable)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn maximum_adjustment_restores_default_minimum_when_possible() {
        let mut field = SpinnerEfield::get_instance(Some("Number"), 8, 2, 10);
        field.minimum = 7;
        field.set_maximum(Some(9));
        field.adjust_to_maximum();
        assert_eq!(field.minimum, 2);
    }
    #[test]
    fn editable_creation_is_lazy() {
        let mut field = SpinnerEfield::get_instance(None, 1, 0, 4);
        field.set_editable(true);
        assert!(field.appearance_extension.is_none());
        field.set_editable(false);
        assert!(!field.is_editable());
    }
}
