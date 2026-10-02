//! `IMOD/Etomo/src/etomo/ui/swing/SpinnerEfield.java`.
//!
//! An integer `JSpinner` field, optionally labeled, that can be made ineditable and
//! whose minimum follows a changeable maximum.
//!
//! The spinner's `SpinnerNumberModel` lives in the `JComponent` (`jdk.rs`); the
//! Java's `model` field is that model, read with `get_spinner_model` and written back
//! with `set_spinner_model` (which, like the Swing model, fires the spinner's change
//! listeners - only called here when the Swing setter would have changed something).
//! Focus events and sizes are not modelled.

use std::cell::RefCell;
use std::rc::Rc;

use super::appearance_extension::AppearanceExtension;
use crate::imod::etomo::etomo_director::ARGUMENTS;
use crate::imod::etomo::jdk::{ChangeListener, FocusListener, JComponent, SpinnerNumberModel};
use crate::imod::etomo::logic::converter;
use crate::imod::etomo::storage::autodoc::autodoc_tokenizer::{DEFAULT_DELIMITER, SEPARATOR_CHAR};
use crate::imod::etomo::r#type::ui_test_field_type::UITestFieldType;
use crate::imod::etomo::util::utilities;

/// Java package-private `final class SpinnerEfield`.
pub struct SpinnerEfield {
    /// Java final `spinner`.
    spinner: Rc<JComponent>,
    /// Java final `pnlRoot` (null without a label).
    pnl_root: Option<Rc<JComponent>>,
    /// Java final `jLabel` (null without a label).
    j_label: Option<Rc<JComponent>>,
    /// Java final `defaultValue`.
    default_value: Option<i32>,
    /// Java final `defaultMinimum`.
    default_minimum: Option<i32>,
    /// Java `appearanceExtension`.
    appearance_extension: RefCell<Option<Rc<AppearanceExtension>>>,
}

impl SpinnerEfield {
    /// Java private `SpinnerEfield(String, int, int, int)`.
    fn new(label: Option<&str>, value: i32, minimum: i32, maximum: i32) -> SpinnerEfield {
        // model = new SpinnerNumberModel(value, minimum, maximum, 1)
        let model = SpinnerNumberModel::new_int(value, minimum, maximum, 1);
        let spinner = JComponent::new_spinner(model);
        // defaultValue = (Integer) spinner.getValue()
        let default_value = Some(spinner.get_spinner_value() as i32);
        // defaultMinimum = (Integer) model.getMinimum()
        let default_minimum = spinner
            .get_spinner_model()
            .and_then(|model| model.minimum)
            .map(|minimum| minimum as i32);
        let (j_label, pnl_root) = if let Some(label) = label {
            (
                Some(JComponent::new_label(label)),
                Some(JComponent::new_panel()),
            )
        } else {
            (None, None)
        };
        let instance = SpinnerEfield {
            spinner,
            pnl_root,
            j_label,
            default_value,
            default_minimum,
            appearance_extension: RefCell::new(None),
        };
        if label.is_some() {
            instance.set_name();
        }
        instance
    }

    /// Java static `getInstance(String, int, int, int)`.
    pub fn get_instance(
        label: Option<&str>,
        value: i32,
        minimum: i32,
        maximum: i32,
    ) -> Rc<SpinnerEfield> {
        let instance = SpinnerEfield::new(label, value, minimum, maximum);
        instance.create_panel();
        Rc::new(instance)
    }

    /// Java private `createPanel()`.
    fn create_panel(&self) {
        if let Some(pnl_root) = &self.pnl_root {
            // Swing layout: pnlRoot.setLayout(new BoxLayout(pnlRoot, BoxLayout.X_AXIS)).
            pnl_root.add(self.j_label.as_ref().unwrap());
            pnl_root.add(&self.spinner);
        }
    }

    /// Java `getComponent()`.
    pub fn get_component(&self) -> Rc<JComponent> {
        if let Some(pnl_root) = &self.pnl_root {
            return pnl_root.clone();
        }
        self.spinner.clone()
    }

    /// Java private `setName()`.
    fn set_name(&self) {
        let Some(j_label) = &self.j_label else {
            return;
        };
        let field_type = UITestFieldType::SPINNER;
        let name = utilities::convert_label_to_name(
            Some(&j_label.get_text()),
            field_type.is_unlimited_segments(),
        );
        if let Some(name) = name {
            self.spinner
                .set_name(Some(&format!("{}{}{}", field_type, SEPARATOR_CHAR, name)));
            if ARGUMENTS.lock().unwrap().is_print_names() {
                println!(
                    "{} {} ",
                    self.spinner.get_name().as_deref().unwrap_or("null"),
                    DEFAULT_DELIMITER
                );
            }
        }
    }

    /// Java `addChangeListener(ChangeListener)`.
    pub fn add_change_listener(&self, listener: ChangeListener) {
        self.spinner.add_change_listener(listener);
    }

    /// Java `addFocusListener(FocusListener)`.
    pub fn add_focus_listener(&self, listener: FocusListener) {
        self.spinner.add_focus_listener(listener);
    }

    /// Java `equalsSource(EventObject)`, given the event's `getSource()`.
    pub fn equals_source(&self, event_source: Option<&Rc<JComponent>>) -> bool {
        if let Some(source) = event_source {
            // System.out.println("A:spinner:" + ((Object) spinner).toString() + ",source:"
            // + event.getSource().toString());
            return Rc::ptr_eq(&self.spinner, source);
        }
        false
    }

    /// Java public `setMaximum(Integer)`.  Set maximum.  Make sure that value and
    /// minimum are less then or equal to maximum.
    pub fn set_maximum(&self, maximum: Option<i32>) {
        if let Some(maximum) = maximum {
            // model.setMaximum(maximum)
            if let Some(mut model) = self.spinner.get_spinner_model() {
                if model.maximum != Some(maximum as f64) {
                    model.maximum = Some(maximum as f64);
                    self.spinner.set_spinner_model(model);
                }
            }
        }
    }

    /// Java public `adjustToMaximum()`.  Adjust value and minimum so that: minimum <=
    /// value <= maximum.  If minimum has to be changed, prefer defaultMinimum.
    pub fn adjust_to_maximum(&self) {
        let Some(maximum) = self
            .spinner
            .get_spinner_model()
            .and_then(|model| model.maximum)
            .map(|maximum| maximum as i32)
        else {
            return;
        };
        // Ensure that value <= maximum.
        let mut value: Option<i32> = Some(self.spinner.get_spinner_value() as i32);
        if value.is_some_and(|value| value > maximum) {
            // model.setValue(maximum): the model sets the value without a range check and
            // fires a change when it differs.
            if let Some(mut model) = self.spinner.get_spinner_model() {
                if model.value != maximum as f64 {
                    model.value = maximum as f64;
                    self.spinner.set_spinner_model(model);
                }
            }
            value = Some(maximum);
        }
        let Some(mut minimum) = self
            .spinner
            .get_spinner_model()
            .and_then(|model| model.minimum)
            .map(|minimum| minimum as i32)
        else {
            return;
        };
        // Ensure that minimum <= maximum.
        if minimum > maximum {
            // Set minimum back to it's original value if possible.
            if let Some(default_minimum) = self.default_minimum.filter(|m| *m <= maximum) {
                // model.setMinimum(default_minimum): fires a change when the minimum differs.
                if let Some(mut model) = self.spinner.get_spinner_model() {
                    if model.minimum != Some(default_minimum as f64) {
                        model.minimum = Some(default_minimum as f64);
                        self.spinner.set_spinner_model(model);
                    }
                }
                minimum = default_minimum;
            } else {
                // model.setMinimum(maximum): fires a change when the minimum differs.
                if let Some(mut model) = self.spinner.get_spinner_model() {
                    if model.minimum != Some(maximum as f64) {
                        model.minimum = Some(maximum as f64);
                        self.spinner.set_spinner_model(model);
                    }
                }
                minimum = maximum;
            }
        } else {
            // If minimum was changed in the past, set it back to it's original value if
            // possible. Don't do an equivalence comparison between Integers as this will
            // compare addresses rather then integer values.
            // Make this change only if it wouldn't force value to change.
            if let Some(default_minimum) = self.default_minimum {
                if minimum != default_minimum
                    && default_minimum <= maximum
                    && value.is_some_and(|value| default_minimum <= value)
                {
                    // model.setMinimum(default_minimum): fires a change when the minimum differs.
                    if let Some(mut model) = self.spinner.get_spinner_model() {
                        if model.minimum != Some(default_minimum as f64) {
                            model.minimum = Some(default_minimum as f64);
                            self.spinner.set_spinner_model(model);
                        }
                    }
                    minimum = default_minimum;
                }
            }
        }
        // Ensure that minimum <= value.
        // (Java unboxes `value` here; it cannot be null, since it was read from the
        // spinner's Integer value.)
        if value.is_some_and(|value| value < minimum) {
            // model.setValue(minimum): the model sets the value without a range check and
            // fires a change when it differs.
            if let Some(mut model) = self.spinner.get_spinner_model() {
                if model.value != minimum as f64 {
                    model.value = minimum as f64;
                    self.spinner.set_spinner_model(model);
                }
            }
        }
    }

    /// Java public `setText(Number)`.
    pub fn set_text_number(&self, number: f64) {
        self.spinner.set_spinner_value(number);
    }

    /// Java `setText(String)`.
    pub fn set_text_string(&self, string: Option<&str>) {
        // If string is null, then reset the field to the value it was created with.
        let value = if string.is_none() {
            self.default_value
        } else {
            converter::to_integer_with_round(string, false)
        };
        let Some(value) = value else {
            // Invalid string
            return;
        };
        // spinner.setValue(value); the IllegalArgumentException the Java catches is
        // thrown only for a non-Number value, which an Integer never is.
        self.spinner.set_spinner_value(value as f64);
    }

    /// Java public `getText()`: `Integer.toString` of the value.
    pub fn get_text(&self) -> String {
        (self.spinner.get_spinner_value() as i32).to_string()
    }

    // appearanceExtension

    /// Java private `createAppearanceExtension()`.
    fn create_appearance_extension(&self) {
        if self.appearance_extension.borrow().is_none() {
            let appearance_extension = AppearanceExtension::new_component(&self.spinner);
            appearance_extension.set_allow_foreground_change_on_error(false);
            *self.appearance_extension.borrow_mut() = Some(appearance_extension);
        }
    }

    /// Java `setEnabled(boolean)`.
    pub fn set_enabled(&self, enabled: bool) {
        let appearance_extension = self.appearance_extension.borrow().clone();
        match appearance_extension {
            None => self.spinner.set_enabled(enabled),
            Some(appearance_extension) => appearance_extension.set_enabled(enabled),
        }
        if let Some(j_label) = &self.j_label {
            j_label.set_enabled(self.is_enabled());
        }
    }

    /// Java public `isEnabled()`.
    pub fn is_enabled(&self) -> bool {
        let appearance_extension = self.appearance_extension.borrow().clone();
        match appearance_extension {
            None => self.spinner.is_enabled(),
            Some(appearance_extension) => appearance_extension.is_enabled(),
        }
    }

    /// Java `setPreferredWidth(int)`.  Set the preferred text field width in pixels.
    pub fn set_preferred_width(&self, _width: i32) {
        // Swing layout: dim = spinner.getPreferredSize(); dim.width = width *
        // (int) Math.round(UIParameters.getInstance().getFontSizeAdjustment());
        // spinner.setPreferredSize(dim); spinner.setMaximumSize(dim).
    }

    /// Java `setEditable(boolean)`.
    pub fn set_editable(&self, editable: bool) {
        // Always starts as editable. AppearanceExtension is necessary for making it
        // ineditable.
        if editable && self.appearance_extension.borrow().is_none() {
            return;
        }
        self.create_appearance_extension();
        let appearance_extension = self.appearance_extension.borrow().clone().unwrap();
        appearance_extension.set_editable(editable);
    }

    /// Java `isVisible()`.
    pub fn is_visible(&self) -> bool {
        if let Some(pnl_root) = &self.pnl_root {
            return pnl_root.is_visible();
        }
        self.spinner.is_visible()
    }

    /// Java `setVisible(boolean)`.
    pub fn set_visible(&self, visible: bool) {
        if let Some(pnl_root) = &self.pnl_root {
            pnl_root.set_visible(visible);
        } else {
            self.spinner.set_visible(visible);
        }
    }

    /// Java `isEditable()`.
    pub fn is_editable(&self) -> bool {
        // Always starts as editable. AppearanceExtension is necessary for making it
        // ineditable.
        let appearance_extension = self.appearance_extension.borrow().clone();
        match appearance_extension {
            None => true,
            Some(appearance_extension) => appearance_extension.is_editable(),
        }
    }
}
