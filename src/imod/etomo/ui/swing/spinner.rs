//! `IMOD/Etomo/src/etomo/ui/swing/Spinner.java`.
//!
//! `JSpinner`, `SpinnerNumberModel`, Swing listener registration, and component
//! painting remain native GUI boundaries; source-owned spinner state and listener
//! routing stay explicit in this unit.
#![allow(dead_code)]

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SpinnerNumberModel {
    pub value: i32,
    pub minimum: i32,
    pub maximum: i32,
    pub step: i32,
}

pub struct Spinner {
    pub text: String,
    pub default_value: i32,
    pub labeled: bool,
    pub minimum: i32,
    pub model: SpinnerNumberModel,
    pub panel_visible: bool,
    pub label_enabled: bool,
    pub spinner_enabled: bool,
    pub debug: bool,
    pub change_listener_count: usize,
    pub spinner_change_listening: bool,
    pub maximum: i32,
    pub tooltip: Option<String>,
    pub maximum_width: Option<(i32, bool)>,
}
impl Spinner {
    pub fn new(
        text: &str,
        labeled: bool,
        value: i32,
        minimum: i32,
        maximum: i32,
        step: i32,
    ) -> Self {
        Self {
            text: text.into(),
            default_value: value,
            labeled,
            minimum,
            model: SpinnerNumberModel {
                value,
                minimum,
                maximum,
                step,
            },
            panel_visible: true,
            label_enabled: true,
            spinner_enabled: true,
            debug: false,
            change_listener_count: 0,
            spinner_change_listening: false,
            maximum,
            tooltip: None,
            maximum_width: None,
        }
    }
    pub fn get_instance(text: &str, value: i32, minimum: i32, maximum: i32, step: i32) -> Self {
        Self::new(text, false, value, minimum, maximum, step)
    }
    pub fn get_labeled_instance(
        label: &str,
        value: i32,
        minimum: i32,
        maximum: i32,
        step: i32,
    ) -> Self {
        Self::new(label, true, value, minimum, maximum, step)
    }
    pub fn set_tool_tip_text(&mut self, text: Option<&str>) {
        self.tooltip = text.map(str::to_owned);
    }
    pub fn get_ui_component(&self) -> &Self {
        self
    }
    pub fn get_component(&self) -> &Self {
        self
    }
    pub fn get_container(&self) -> &Self {
        self
    }
    pub fn get_label(&self) -> &str {
        &self.text
    }
    pub fn is_enabled(&self) -> bool {
        self.spinner_enabled
    }
    pub fn set_enabled(&mut self, enabled: bool) {
        self.spinner_enabled = enabled;
        if self.labeled {
            self.label_enabled = enabled;
        }
    }
    pub fn set_visible(&mut self, visible: bool) {
        self.panel_visible = visible;
    }
    pub fn is_visible(&self) -> bool {
        self.panel_visible
    }
    pub fn set_maximum_width(&mut self, width: i32, adjust_by_font: bool) {
        if width > 0 {
            self.maximum_width = Some((width, adjust_by_font));
        }
    }
    pub fn reset(&mut self) {
        self.model.value = self.default_value;
    }
    pub fn set_max(&mut self, maximum: i32) {
        self.maximum = maximum;
        self.model.maximum = maximum;
    }
    pub fn set_debug(&mut self, input: bool) {
        self.debug = input;
    }
    pub fn set_value(&mut self, value: i32) {
        self.model.value = if value == i32::MIN {
            self.model.minimum
        } else {
            value
        };
    }
    pub fn get_value(&self) -> i32 {
        self.model.value
    }
    pub fn get_int_value(&self) -> i32 {
        self.model.value
    }
    /// Java private `getTextField()`.  The concrete formatted-text control is
    /// owned by the Rust GUI backend; its source-visible editor value is this
    /// spinner model formatted for binding.
    pub fn get_text_field(&self) -> String {
        self.model.value.to_string()
    }
    pub fn add_change_listener(&mut self) {
        self.change_listener_count += 1;
        self.spinner_change_listening = true;
    }
    pub fn state_changed(&self) -> usize {
        self.change_listener_count
    }
    pub fn verify_spinner_source(&self, same_spinner: bool) -> bool {
        same_spinner
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reset_returns_to_constructor_value() {
        let mut spinner = Spinner::get_labeled_instance("Size", 8, 0, 32, 1);
        spinner.set_value(3);
        spinner.reset();
        assert_eq!(spinner.get_value(), 8);
        assert_eq!(spinner.get_text_field(), "8");
    }
}
