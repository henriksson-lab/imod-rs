//! `IMOD/Etomo/src/etomo/ui/swing/CheckBoxSpinner.java`: a check box followed by
//! a spinner that is enabled only while the box is checked.
//!
//! `final class CheckBoxSpinner` (with the inner `CheckBoxSpinnerActionListener`).

use std::rc::{Rc, Weak};

use crate::imod::etomo::jdk::{ActionEvent, ActionListener, JComponent, SpinnerNumberModel};
use crate::imod::etomo::r#type::const_etomo_number::{ConstEtomoNumber, Number};
use crate::imod::etomo::ui::swing::check_box::CheckBox;
use crate::imod::etomo::ui::swing::spinner::Spinner;
use crate::imod::etomo::ui::swing::tooltip_formatter;
use crate::imod::etomo::ui::swing::ui_utilities;

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `CheckBoxSpinner`.
pub struct CheckBoxSpinner {
    panel: Rc<JComponent>,

    spinner: Rc<Spinner>,
    check_box: Rc<CheckBox>,
    // Java fields `panelBackground` and `panelHighlightBackground`: background
    // colours, which the jdk stand-in does not model (painting).
}

impl CheckBoxSpinner {
    /// Java `CheckBoxSpinner(String text)`.
    fn new_string(text: Option<&str>) -> CheckBoxSpinner {
        let check_box = CheckBox::new_string(text);
        let spinner = Spinner::get_instance_string(check_box.get_text_void().as_deref());
        CheckBoxSpinner {
            panel: JComponent::new_panel(),
            spinner,
            check_box,
        }
    }

    /// Java `CheckBoxSpinner(String text, int value, int minimum, int maximum)`.
    fn new_string_int_int_int(
        text: Option<&str>,
        value: i32,
        minimum: i32,
        maximum: i32,
    ) -> CheckBoxSpinner {
        let check_box = CheckBox::new_string(text);
        let spinner = Spinner::get_instance_string_int_int_int(text, value, minimum, maximum);
        CheckBoxSpinner {
            panel: JComponent::new_panel(),
            spinner,
            check_box,
        }
    }

    /// Java `getInstance(String)`.
    pub fn get_instance_string(text: Option<&str>) -> Rc<CheckBoxSpinner> {
        let instance = Rc::new(CheckBoxSpinner::new_string(text));
        instance.create_panel();
        instance.add_listeners(&instance);
        instance
    }

    /// Java `getInstance(String, int value, int minimum, int maximum)`.
    pub fn get_instance_string_int_int_int(
        text: Option<&str>,
        value: i32,
        minimum: i32,
        maximum: i32,
    ) -> Rc<CheckBoxSpinner> {
        let instance = Rc::new(CheckBoxSpinner::new_string_int_int_int(
            text, value, minimum, maximum,
        ));
        instance.create_panel();
        instance.add_listeners(&instance);
        instance
    }

    /// Java `createPanel()`.
    fn create_panel(&self) {
        // init
        self.spinner.set_enabled(false);
        // panel
        // Swing layout: BoxLayout X_AXIS; horizontal glue before and after.
        self.panel.add(&self.check_box.get_component());
        self.panel.add(&self.spinner.get_container());
    }

    /// Java `addListeners()`.
    fn add_listeners(&self, this: &Rc<CheckBoxSpinner>) {
        // new CheckBoxSpinnerActionListener(this): its `actionPerformed` calls
        // `adaptee.enableSpinner()`.
        let adaptee: Weak<CheckBoxSpinner> = Rc::downgrade(this);
        let listener: ActionListener = Rc::new(move |event: &ActionEvent| {
            let _ = event;
            if let Some(adaptee) = adaptee.upgrade() {
                adaptee.enable_spinner();
            }
        });
        self.check_box.add_action_listener(Some(listener));
    }

    /// Java `getContainer()`.
    pub fn get_container(&self) -> Rc<JComponent> {
        self.panel.clone()
    }

    /// Java `setEnabled(boolean)`.
    pub fn set_enabled(&self, enabled: bool) {
        self.check_box.set_enabled(enabled);
        self.enable_spinner();
    }

    /// Java `addActionListener(ActionListener)`.
    pub fn add_action_listener(&self, listener: Option<ActionListener>) {
        self.check_box.add_action_listener(listener);
    }

    /// Java `enableSpinner()`.
    fn enable_spinner(&self) {
        self.spinner
            .set_enabled(self.check_box.is_selected() && self.check_box.is_enabled());
    }

    /// Java `addCheckBoxActionListener(ActionListener)`.
    pub fn add_check_box_action_listener(&self, action_listener: Option<ActionListener>) {
        self.check_box.add_action_listener(action_listener);
    }

    /// Java `getCheckBoxActionCommand()`.
    pub fn get_check_box_action_command(&self) -> Option<String> {
        self.check_box.get_action_command()
    }

    /// Java `setVisible(boolean)`.
    pub fn set_visible(&self, input: bool) {
        self.panel.set_visible(input);
    }

    /// Java `setCheckBoxEnabled(boolean)`.
    pub fn set_check_box_enabled(&self, enabled: bool) {
        self.check_box.set_enabled(enabled);
        self.enable_spinner();
    }

    /// Java `isCheckBoxEnabled()`.
    pub fn is_check_box_enabled(&self) -> bool {
        self.check_box.is_enabled()
    }

    /// Java `setModel(SpinnerNumberModel)`.
    pub fn set_model(&self, model: SpinnerNumberModel) {
        self.spinner.set_model(model);
    }

    /// Java `setMax(int)`.
    pub fn set_max(&self, max: i32) {
        self.spinner.set_max(max);
    }

    /// Java `setMaximumWidth(int, boolean adjustByFont)`.
    pub fn set_maximum_width(&self, width: i32, adjust_by_font: bool) {
        self.spinner.set_maximum_width_int_boolean(width, adjust_by_font);
    }

    /// Java `getValue()`.
    pub fn get_value(&self) -> Number {
        self.spinner.get_value()
    }

    /// Java `isSelected()`.
    pub fn is_selected(&self) -> bool {
        self.check_box.is_selected()
    }

    /// Java `setSelected(boolean)`.
    pub fn set_selected(&self, selected: bool) {
        self.check_box.set_selected_boolean(selected);
        self.enable_spinner();
    }

    /// Java `setValue(String)`.
    pub fn set_value_string(&self, value: Option<&str>) {
        self.spinner.set_value_string(value);
    }

    /// Java `setValue(int)`.
    pub fn set_value_int(&self, value: i32) {
        self.spinner.set_value_int(value);
    }

    /// Java `setValue(Number)`.
    pub fn set_value_number(&self, value: Number) {
        self.spinner.set_value_number(value);
    }

    /// Java `setValue(ConstEtomoNumber)`.
    pub fn set_value_const_etomo_number(&self, value: &ConstEtomoNumber) {
        self.spinner.set_value_const_etomo_number(value);
    }

    /// Java `setHighlight(boolean)`.
    pub fn set_highlight(&self, highlight: bool) {
        // Swing painting: on first use, panelBackground = panel.getBackground() and
        // panelHighlightBackground = Colors.subtractColor(
        // Colors.HIGHLIGHT_BACKGROUND, Colors.subtractColor(Colors.BACKGROUND,
        // panelBackground)) (greying out the highlight color to match the panel's
        // original color); then checkBox.setBackground(panelHighlightBackground)
        // when highlighting, else checkBox.setBackground(panelBackground).
        // Backgrounds are not modelled.
        let _ = self.get_container();
        ui_utilities::highlight_j_text_components(highlight, &self.panel);
    }

    /// Java `setToolTipText(String)`.
    pub fn set_tool_tip_text(&self, text: Option<&str>) {
        self.set_check_box_tool_tip_text(text);
        self.set_spinner_tool_tip_text(text);
    }

    /// Java `setCheckBoxToolTipText(String)`.
    pub fn set_check_box_tool_tip_text(&self, text: Option<&str>) {
        self.check_box.set_tool_tip_text_string(text);
    }

    /// Java `setSpinnerToolTipText(String)`.
    pub fn set_spinner_tool_tip_text(&self, text: Option<&str>) {
        self.spinner
            .set_tool_tip_text(tooltip_formatter::INSTANCE.format(text).as_deref());
    }
}
