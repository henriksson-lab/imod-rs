//! `IMOD/Etomo/src/etomo/ui/swing/Spinner.java`: a self-naming integer spinner,
//! optionally with a label.
//!
//! `final class Spinner implements UIComponent, SwingComponent, ChangeListener`.
//! The `JSpinner` (and its `SpinnerNumberModel`) is a jdk stand-in
//! [`JComponent`].  Java `Number` values are
//! [`Number`](crate::imod::etomo::r#type::const_etomo_number::Number): the jdk
//! spinner stores an `f64` and a flag saying whether its model was built from
//! `int`s, which is how `getValue()` knows to answer an `Integer`.

use std::cell::{Cell, RefCell};
use std::rc::{Rc, Weak};

use crate::imod::etomo::etomo_director::ARGUMENTS;
use crate::imod::etomo::jdk::{ChangeEvent, ChangeListener, JComponent, SpinnerNumberModel};
use crate::imod::etomo::storage::autodoc::autodoc_tokenizer::{DEFAULT_DELIMITER, SEPARATOR_CHAR};
use crate::imod::etomo::r#type::const_etomo_number::{
    ConstEtomoNumber, INTEGER_NULL_VALUE, Number,
};
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::r#type::parsed_element::ParsedElement;
use crate::imod::etomo::r#type::ui_test_field_type;
use crate::imod::etomo::ui::swing::swing_component::SwingComponent;
use crate::imod::etomo::ui::swing::tooltip_formatter;
use crate::imod::etomo::ui::ui_component::UIComponent;
use crate::imod::etomo::util::utilities;

/// Java `Spinner`.
pub struct Spinner {
    /// Java `this`, for `spinner.addChangeListener(this)` and `getUIComponent`.
    this: Weak<Spinner>,
    spinner: Rc<JComponent>,
    default_value: Number,
    labeled: bool,
    minimum: i32,

    // Java field `model`: the `SpinnerNumberModel` shared with the `JSpinner`.
    // The jdk stand-in keeps the model inside the spinner component, so it is
    // read and written there (`get_spinner_model` / `set_spinner_model`).
    panel: Option<Rc<JComponent>>,
    label: Option<Rc<JComponent>>,
    debug: Cell<bool>,
    change_listeners: RefCell<Option<Vec<ChangeListener>>>,
    spinner_change_listening: Cell<bool>,

    maximum: Cell<i32>,
}

impl Spinner {
    /// Java `Spinner(String text, boolean labeled, int value, int minimum,
    /// int maximum, int step)`.
    fn new(
        text: Option<&str>,
        labeled: bool,
        value: i32,
        minimum: i32,
        maximum: i32,
        step: i32,
    ) -> Rc<Spinner> {
        let model = SpinnerNumberModel::new_int(value, minimum, maximum, step);
        let spinner = JComponent::new_spinner(model);
        let default_value = Number::Integer(value);
        let (panel, label) = if labeled {
            let panel = JComponent::new_panel();
            // Swing layout: panel.setLayout(new BoxLayout(panel, BoxLayout.X_AXIS)).
            let label = JComponent::new_label(text.unwrap_or(""));
            panel.add(&label);
            panel.add(&spinner);
            (Some(panel), Some(label))
        } else {
            (None, None)
        };
        // Swing layout: set the maximum height of the text field box to twice the
        // font size (of the label if larger, else of the spinner) since it is not
        // set by default - spinner.setMaximumSize(maxSize).
        let instance = Rc::new_cyclic(|this| Spinner {
            this: this.clone(),
            spinner,
            default_value,
            labeled,
            minimum,
            panel,
            label,
            debug: Cell::new(false),
            change_listeners: RefCell::new(None),
            spinner_change_listening: Cell::new(false),
            maximum: Cell::new(maximum),
        });
        instance.set_name(text);
        instance
    }

    /// Java `setName(String)`.
    pub fn set_name(&self, text: Option<&str>) {
        let field_type = &ui_test_field_type::SPINNER;
        // Java string concatenation of a null name gives "null".
        let name = format!(
            "{}{}{}",
            field_type,
            SEPARATOR_CHAR,
            utilities::convert_label_to_name(text, field_type.is_unlimited_segments())
                .as_deref()
                .unwrap_or("null")
        );
        self.spinner.set_name(Some(&name));
        if ARGUMENTS.lock().unwrap().is_print_names() {
            println!(
                "{} {} ",
                self.spinner.get_name().as_deref().unwrap_or("null"),
                DEFAULT_DELIMITER
            );
        }
    }

    /// Java `setModel(SpinnerNumberModel)`.
    pub fn set_model(&self, input: SpinnerNumberModel) {
        // model = input;
        self.spinner.set_spinner_model(input);
    }

    // Java `getTextField()`: the spinner editor's `JFormattedTextField` (an
    // internal Swing child, not modelled; nothing in this class calls it).

    /// Java `getInstance(String)`.
    pub fn get_instance_string(text: Option<&str>) -> Rc<Spinner> {
        Spinner::new(text, false, 1, 1, 1, 1)
    }

    /// Java `getInstance(String, int value, int minimum, int maximum)`.
    pub fn get_instance_string_int_int_int(
        text: Option<&str>,
        value: i32,
        minimum: i32,
        maximum: i32,
    ) -> Rc<Spinner> {
        Spinner::new(text, false, value, minimum, maximum, 1)
    }

    /// Java `getInstance(String, int value, int minimum, int maximum, int step)`.
    pub fn get_instance_string_int_int_int_int(
        text: Option<&str>,
        value: i32,
        minimum: i32,
        maximum: i32,
        step: i32,
    ) -> Rc<Spinner> {
        Spinner::new(text, false, value, minimum, maximum, step)
    }

    /// Java `getLabeledInstance(String)`.
    pub fn get_labeled_instance_string(label: Option<&str>) -> Rc<Spinner> {
        Spinner::new(label, true, 1, 1, 1, 1)
    }

    /// Java `getLabeledInstance(String, int value, int minimum, int maximum)`.
    pub fn get_labeled_instance_string_int_int_int(
        label: Option<&str>,
        value: i32,
        minimum: i32,
        maximum: i32,
    ) -> Rc<Spinner> {
        Spinner::new(label, true, value, minimum, maximum, 1)
    }

    /// Java `getLabeledInstance(String, int value, int minimum, int maximum,
    /// int step)`.
    pub fn get_labeled_instance_string_int_int_int_int(
        label: Option<&str>,
        value: i32,
        minimum: i32,
        maximum: i32,
        step: i32,
    ) -> Rc<Spinner> {
        Spinner::new(label, true, value, minimum, maximum, step)
    }

    /// Java `getLabeledInstance(String, int maximum)`.  Sets minimum, current
    /// value, and step to 1.
    pub fn get_labeled_instance_string_int(label: Option<&str>, maximum: i32) -> Rc<Spinner> {
        Spinner::new(label, true, 1, 1, maximum, 1)
    }

    /// Java `setToolTipText(String)`.
    pub fn set_tool_tip_text(&self, text: Option<&str>) {
        let tooltip = tooltip_formatter::INSTANCE.format(text);
        self.spinner.set_tool_tip_text(tooltip.as_deref());
        if let Some(label) = &self.label {
            label.set_tool_tip_text(tooltip.as_deref());
        }
    }

    /// Java `setAlignmentX(float)`.
    pub fn set_alignment_x(&self, alignment_x: f32) {
        // Swing layout: panel.setAlignmentX(alignmentX).
        let _ = alignment_x;
    }

    /// Java `getUIComponent()`.
    pub fn get_ui_component(&self) -> Option<Rc<dyn SwingComponent>> {
        self.this.upgrade().map(|this| this as Rc<dyn SwingComponent>)
    }

    /// Java `getComponent()`.
    pub fn get_component(&self) -> Rc<JComponent> {
        self.get_container()
    }

    /// Java `getContainer()`.
    pub fn get_container(&self) -> Rc<JComponent> {
        match &self.panel {
            None => self.spinner.clone(),
            Some(panel) => panel.clone(),
        }
    }

    /// Java `getLabel()`.
    pub fn get_label(&self) -> Option<String> {
        // Upstream bug fixed (Spinner.java:228-230): `label.getText()` throws a
        // NullPointerException on an unlabeled spinner; that answers null here.
        self.label.as_ref().map(|label| label.get_text())
    }

    /// Java `isEnabled()`.
    pub fn is_enabled(&self) -> bool {
        self.spinner.is_enabled()
    }

    /// Java `setEnabled(boolean)`.
    pub fn set_enabled(&self, enabled: bool) {
        self.spinner.set_enabled(enabled);
        if let Some(label) = &self.label {
            label.set_enabled(enabled);
        }
    }

    /// Java `setVisible(boolean)`.
    pub fn set_visible(&self, visible: bool) {
        match &self.panel {
            None => self.spinner.set_visible(visible),
            Some(panel) => panel.set_visible(visible),
        }
    }

    /// Java `isVisible()`.
    pub fn is_visible(&self) -> bool {
        // Upstream bug fixed (Spinner.java:254-256): `panel.isVisible()` throws a
        // NullPointerException on an unlabeled spinner.  As `setVisible` does, the
        // spinner itself is the visible component when there is no panel.
        match &self.panel {
            None => self.spinner.is_visible(),
            Some(panel) => panel.is_visible(),
        }
    }

    /// Java `setMaximumWidth(int)`.
    pub fn set_maximum_width_int(&self, width: i32) {
        self.set_maximum_width_int_boolean(width, true);
    }

    /// Java `setMaximumWidth(int, boolean adjustByFont)`.
    pub fn set_maximum_width_int_boolean(&self, width: i32, adjust_by_font: bool) {
        if width <= 0 {
            return;
        }
        // Swing layout: spinner.setMaximumSize(UIUtilities.calcNewTextFieldSize(
        // spinner.getPreferredSize(), width, adjustByFont)).
        let _ = adjust_by_font;
    }

    /// Java `reset()`.
    pub fn reset(&self) {
        self.spinner
            .set_spinner_value(self.default_value.double_value());
    }

    /// Java `setMax(int)`.
    pub fn set_max(&self, max: i32) {
        self.maximum.set(max);
        // model.setMaximum((Integer) max)
        if let Some(mut model) = self.spinner.get_spinner_model() {
            model.maximum = Some(max as f64);
            self.spinner.set_spinner_model(model);
        }
    }

    /// Java `setDebug(boolean)`.
    pub fn set_debug(&self, input: bool) {
        self.debug.set(input);
    }

    /// Java `setValue(ParsedElement)`.  (`(Integer) model.getMinimum()` is the
    /// jdk model's `minimum`.)
    pub fn set_value_parsed_element(&self, value: Option<&dyn ParsedElement>) {
        match value {
            Some(value) if !value.is_empty() => {
                let raw_number = value.get_raw_number();
                if let Some(raw_number) = raw_number {
                    self.spinner.set_spinner_value(raw_number.double_value());
                } else if let Some(minimum) = self.spinner.get_spinner_model().and_then(|model| model.minimum) {
                    self.spinner.set_spinner_value(minimum);
                }
            }
            _ => {
                if let Some(minimum) = self.spinner.get_spinner_model().and_then(|model| model.minimum) {
                    self.spinner.set_spinner_value(minimum);
                }
            }
        }
    }

    /// Java `setValue(int)`.
    pub fn set_value_int(&self, value: i32) {
        if value == INTEGER_NULL_VALUE {
            if let Some(minimum) = self.spinner.get_spinner_model().and_then(|model| model.minimum) {
                self.spinner.set_spinner_value(minimum);
            }
        } else {
            self.spinner.set_spinner_value(value as f64);
        }
    }

    /// Java `setValue(String)`.
    pub fn set_value_string(&self, value: Option<&str>) {
        let mut n_value = EtomoNumber::new();
        n_value.set_string(value);
        self.set_value_const_etomo_number(&n_value);
    }

    /// Java `setValue(ConstEtomoNumber)`.
    pub fn set_value_const_etomo_number(&self, value: &ConstEtomoNumber) {
        if value.is_null() {
            if let Some(minimum) = self.spinner.get_spinner_model().and_then(|model| model.minimum) {
                self.spinner.set_spinner_value(minimum);
            }
        } else {
            self.spinner
                .set_spinner_value(value.get_number().double_value());
        }
    }

    /// Java `setValue(Number)`.
    pub fn set_value_number(&self, value: Number) {
        self.spinner.set_spinner_value(value.double_value());
    }

    /// Java `getValue()`.
    pub fn get_value(&self) -> Number {
        let value = self.spinner.get_spinner_value();
        if self
            .spinner
            .get_spinner_model()
            .is_none_or(|model| model.integer)
        {
            Number::Integer(value as i32)
        } else {
            Number::Double(value)
        }
    }

    /// Java `getIntValue()`.
    pub fn get_int_value(&self) -> Option<i32> {
        // Only integers are used in this spinner.  (The jdk spinner always has a
        // value, so Java's null case does not arise.)
        let number = self.get_value();
        Some(number.int_value())
    }

    /// Java `addChangeListener(ChangeListener)`.
    pub fn add_change_listener_change_listener(&self, listener: Option<ChangeListener>) {
        let Some(listener) = listener else {
            return;
        };
        if self.change_listeners.borrow().is_none() {
            self.add_change_listener_void();
            *self.change_listeners.borrow_mut() = Some(Vec::new());
        }
        self.change_listeners
            .borrow_mut()
            .as_mut()
            .unwrap()
            .push(listener);
    }

    /// Java `addChangeListener()`.
    fn add_change_listener_void(&self) {
        if !self.spinner_change_listening.get() {
            let this = self.this.clone();
            self.spinner.add_change_listener(Rc::new(move |event: &ChangeEvent| {
                if let Some(this) = this.upgrade() {
                    this.state_changed(event);
                }
            }));
            self.spinner_change_listening.set(true);
        }
    }

    /// Java `stateChanged(ChangeEvent)`.
    pub fn state_changed(&self, event: &ChangeEvent) {
        let change_listeners = self.change_listeners.borrow().clone();
        if let Some(change_listeners) = change_listeners {
            for listener in change_listeners.iter() {
                listener(event);
            }
        }
    }

    /// Java `verifySpinnerSource(JSpinner)`.
    pub fn verify_spinner_source(&self, input: &Rc<JComponent>) -> bool {
        Rc::ptr_eq(&self.spinner, input)
    }
}

impl UIComponent for Spinner {
    fn get_ui_component(&self) -> &dyn SwingComponent {
        self
    }
    fn get_component(&self) -> Rc<JComponent> {
        Spinner::get_component(self)
    }
}

impl SwingComponent for Spinner {
    fn get_component(&self) -> Rc<JComponent> {
        Spinner::get_component(self)
    }
}
