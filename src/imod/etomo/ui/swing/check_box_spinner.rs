//! `IMOD/Etomo/src/etomo/ui/swing/CheckBoxSpinner.java`.
//!
//! `JPanel`, `BoxLayout`, glue, Swing action dispatch, and text-component
//! highlighting remain native GUI boundaries.  This unit keeps the composite's
//! source-owned checkbox/spinner state and enable/highlight transitions.
#![allow(dead_code)]

use crate::imod::etomo::r#type::const_etomo_number::{ConstEtomoNumber, Number};

use super::check_box::{CheckBox, Color};
use super::spinner::{Spinner, SpinnerNumberModel};

const BACKGROUND: Color = Color(255, 255, 255);
const HIGHLIGHT_BACKGROUND: Color = Color(204, 255, 255);

/// Java `JPanel` state used by `CheckBoxSpinner`; widget creation and painting
/// are native GUI boundaries.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CheckBoxSpinnerPanel {
    pub visible: bool,
    pub background: Color,
    pub box_layout_x_axis: bool,
    pub leading_horizontal_glue: bool,
    pub check_box_component: bool,
    pub spinner_container: bool,
    pub trailing_horizontal_glue: bool,
    pub text_components_highlighted: bool,
}

impl Default for CheckBoxSpinnerPanel {
    fn default() -> Self {
        Self {
            visible: true,
            background: BACKGROUND,
            box_layout_x_axis: false,
            leading_horizontal_glue: false,
            check_box_component: false,
            spinner_container: false,
            trailing_horizontal_glue: false,
            text_components_highlighted: false,
        }
    }
}

/// Java package-private final `CheckBoxSpinner`.
pub struct CheckBoxSpinner {
    pub panel: CheckBoxSpinnerPanel,
    pub spinner: Spinner,
    pub check_box: CheckBox,
    pub panel_background: Option<Color>,
    pub panel_highlight_background: Option<Color>,
}

impl CheckBoxSpinner {
    /// Java private `CheckBoxSpinner(String)`.
    fn new(text: &str) -> Self {
        let check_box = CheckBox::new_with_text(text);
        let spinner = Spinner::get_instance(check_box.get_text().unwrap_or_default(), 1, 1, 1, 1);
        Self {
            panel: CheckBoxSpinnerPanel::default(),
            spinner,
            check_box,
            panel_background: None,
            panel_highlight_background: None,
        }
    }

    /// Java private `CheckBoxSpinner(String,int,int,int)`.
    fn new_with_value(text: &str, value: i32, minimum: i32, maximum: i32) -> Self {
        Self {
            panel: CheckBoxSpinnerPanel::default(),
            spinner: Spinner::get_instance(text, value, minimum, maximum, 1),
            check_box: CheckBox::new_with_text(text),
            panel_background: None,
            panel_highlight_background: None,
        }
    }

    /// Java `getInstance(String)`.
    pub fn get_instance(text: &str) -> Self {
        let mut instance = Self::new(text);
        instance.create_panel();
        instance.add_listeners();
        instance
    }

    /// Java `getInstance(String,int,int,int)`.
    pub fn get_instance_with_value(text: &str, value: i32, minimum: i32, maximum: i32) -> Self {
        let mut instance = Self::new_with_value(text, value, minimum, maximum);
        instance.create_panel();
        instance.add_listeners();
        instance
    }

    /// Java private `createPanel()`.
    fn create_panel(&mut self) {
        self.spinner.set_enabled(false);
        self.panel.box_layout_x_axis = true;
        self.panel.leading_horizontal_glue = true;
        self.panel.check_box_component = true;
        self.panel.spinner_container = true;
        self.panel.trailing_horizontal_glue = true;
    }

    /// Java private `addListeners()`.
    fn add_listeners(&mut self) {
        self.check_box.add_action_listener();
    }

    /// Java `getContainer()`; native hierarchy attachment remains at the GUI boundary.
    pub fn get_container(&self) -> &CheckBoxSpinnerPanel {
        &self.panel
    }

    /// Java `setEnabled(boolean)`.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.check_box.set_enabled(enabled);
        self.enable_spinner();
    }

    /// Java `addActionListener(ActionListener)`; dispatch is a native GUI boundary.
    pub fn add_action_listener(&mut self) {
        self.check_box.add_action_listener();
    }

    /// Java private `enableSpinner()`.
    fn enable_spinner(&mut self) {
        self.spinner
            .set_enabled(self.check_box.is_selected() && self.check_box.is_enabled());
    }

    /// Java `addCheckBoxActionListener(ActionListener)`.
    pub fn add_check_box_action_listener(&mut self) {
        self.check_box.add_action_listener();
    }

    /// Java `getCheckBoxActionCommand()`.
    pub fn get_check_box_action_command(&self) -> Option<&str> {
        self.check_box.get_action_command()
    }

    /// Java `setVisible(boolean)`.
    pub fn set_visible(&mut self, input: bool) {
        self.panel.visible = input;
    }

    /// Java `setCheckBoxEnabled(boolean)`.
    pub fn set_check_box_enabled(&mut self, enabled: bool) {
        self.check_box.set_enabled(enabled);
        self.enable_spinner();
    }

    /// Java `isCheckBoxEnabled()`.
    pub fn is_check_box_enabled(&self) -> bool {
        self.check_box.is_enabled()
    }

    /// Java `setModel(SpinnerNumberModel)`.
    pub fn set_model(&mut self, model: SpinnerNumberModel) {
        self.spinner.minimum = model.minimum;
        self.spinner.maximum = model.maximum;
        self.spinner.model = model;
    }

    /// Java `setMax(int)`.
    pub fn set_max(&mut self, max: i32) {
        self.spinner.set_max(max);
    }

    /// Java `setMaximumWidth(int,boolean)`.
    pub fn set_maximum_width(&mut self, width: i32, adjust_by_font: bool) {
        self.spinner.set_maximum_width(width, adjust_by_font);
    }

    /// Java `getValue()`.
    pub fn get_value(&self) -> Number {
        Number::Integer(self.spinner.get_value())
    }

    /// Java `isSelected()`.
    pub fn is_selected(&self) -> bool {
        self.check_box.is_selected()
    }

    /// Java `setSelected(boolean)`.
    pub fn set_selected(&mut self, selected: bool) {
        self.check_box.set_selected(selected);
        self.enable_spinner();
    }

    /// Java `setValue(String)`.
    pub fn set_value_string(&mut self, value: &str) {
        self.spinner
            .set_value(value.parse::<i32>().unwrap_or(self.spinner.model.minimum));
    }

    /// Java `setValue(int)`.
    pub fn set_value_int(&mut self, value: i32) {
        self.spinner.set_value(value);
    }

    /// Java `setValue(Number)`.
    pub fn set_value_number(&mut self, value: Number) {
        self.spinner.set_value(value.int_value());
    }

    /// Java `setValue(ConstEtomoNumber)`.
    pub fn set_value_const_etomo_number(&mut self, value: &ConstEtomoNumber) {
        self.spinner.set_value(if value.is_null() {
            self.spinner.model.minimum
        } else {
            value.get_number().int_value()
        });
    }

    /// Java `setHighlight(boolean)`.
    pub fn set_highlight(&mut self, highlight: bool) {
        if self.panel_background.is_none() {
            self.panel_background = Some(self.panel.background);
            let panel_background = self.panel.background;
            self.panel_highlight_background = Some(Color(
                HIGHLIGHT_BACKGROUND.0 - (BACKGROUND.0 - panel_background.0),
                HIGHLIGHT_BACKGROUND.1 - (BACKGROUND.1 - panel_background.1),
                HIGHLIGHT_BACKGROUND.2 - (BACKGROUND.2 - panel_background.2),
            ));
        }
        self.check_box.set_background(if highlight {
            self.panel_highlight_background
        } else {
            self.panel_background
        });
        self.panel.text_components_highlighted = highlight;
    }

    /// Java `setToolTipText(String)`.
    pub fn set_tool_tip_text(&mut self, text: &str) {
        self.set_check_box_tool_tip_text(text);
        self.set_spinner_tool_tip_text(text);
    }

    /// Java `setCheckBoxToolTipText(String)`.
    pub fn set_check_box_tool_tip_text(&mut self, text: &str) {
        self.check_box.set_tool_tip_text(Some(text));
    }

    /// Java `setSpinnerToolTipText(String)`; TooltipFormatter is a presentation boundary.
    pub fn set_spinner_tool_tip_text(&mut self, text: &str) {
        self.spinner.set_tool_tip_text(Some(text));
    }
}

/// Java private inner `CheckBoxSpinnerActionListener`; actual Swing event delivery
/// is a GUI boundary, while its source callback remains explicit.
pub struct CheckBoxSpinnerActionListener;

impl CheckBoxSpinnerActionListener {
    /// Java `CheckBoxSpinnerActionListener(CheckBoxSpinner)`.
    pub const fn new() -> Self {
        Self
    }

    /// Java `actionPerformed(ActionEvent)`.
    pub fn action_performed(&self, adaptee: &mut CheckBoxSpinner) {
        adaptee.enable_spinner();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construction_uses_glue_and_disables_spinner() {
        let value = CheckBoxSpinner::get_instance_with_value("Iterations", 2, 1, 5);
        assert!(value.panel.box_layout_x_axis);
        assert!(value.panel.leading_horizontal_glue && value.panel.trailing_horizontal_glue);
        assert_eq!(value.spinner.get_value(), 2);
        assert!(!value.spinner.is_enabled());
    }

    #[test]
    fn selection_and_checkbox_enablement_control_spinner() {
        let mut value = CheckBoxSpinner::get_instance("Use count");
        value.set_selected(true);
        assert!(value.spinner.is_enabled());
        value.set_check_box_enabled(false);
        assert!(!value.spinner.is_enabled());
    }

    #[test]
    fn highlight_and_values_follow_source_transitions() {
        let mut value = CheckBoxSpinner::get_instance_with_value("Count", 2, 1, 5);
        value.set_highlight(true);
        assert_eq!(
            value.check_box.check_box.background,
            Some(HIGHLIGHT_BACKGROUND)
        );
        assert!(value.panel.text_components_highlighted);
        value.set_highlight(false);
        assert_eq!(value.check_box.check_box.background, Some(BACKGROUND));
        value.set_value_string("4");
        assert_eq!(value.get_value().int_value(), 4);
        value.set_value_string("invalid");
        assert_eq!(value.get_value().int_value(), 1);
    }
}
