//! `IMOD/Etomo/src/etomo/ui/swing/Popup.java`.
//!
//! A popup built from a `JOptionPane`, kept as an object so the caller can read
//! the answer and the check box afterwards.  Pass it to `UIHarness.openPopup`,
//! or call `open`/`log`.
//!
//! The modal `JOptionPane`/`JDialog` pair is replaced by the Rust-only
//! presentation hook in `ui_harness.rs` ([`ui_harness::present_popup`]), as
//! `AbstractFrame.showOptionDialog` does: the request carries everything the
//! Java pane was built from, and the answer is reduced to the value
//! `pane.getValue()` would have held.
//!
//! `Popup` is an EDT object (`Rc`, `&self` methods).

use std::cell::RefCell;
use std::rc::{Rc, Weak};

use super::abstract_frame::{
    DEFAULT_OPTION, ERROR_MESSAGE, QUESTION_MESSAGE, YES_NO_OPTION, YES_OPTION,
};
use super::check_box::CheckBox;
use super::ui_harness::{self, PopupAnswer, PopupRequest};
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::jdk::JComponent;
use crate::imod::etomo::logic::popup_tool;
use crate::imod::etomo::storage::autodoc::autodoc_tokenizer::{DEFAULT_DELIMITER, SEPARATOR_CHAR};
use crate::imod::etomo::r#type::ui_test_field_type::UITestFieldType;
use crate::imod::etomo::ui::field_displayer::FieldDisplayer;
use crate::imod::etomo::ui::ui_component::UIComponent;
use crate::imod::etomo::util::event_queue::{self, EdtRef};
use crate::imod::etomo::util::utilities;

/// Java's untyped `JOptionPane.getValue()` result, restricted to the two
/// classes this unit inspects: an `Integer` (the index of the default button
/// pressed, when no button labels were passed, or `YES_OPTION` from `log()`),
/// or the `String` label of the button pressed.  Java `null` is `None`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PopupValue {
    Integer(i32),
    String(String),
}

/// Java `public final class Popup`.
pub struct Popup {
    /// Java `private final Component component`.
    component: Option<Rc<JComponent>>,
    /// Java `private final String message`.
    message: Option<String>,
    /// Java `private final CheckBox checkBox`.
    check_box: Option<Rc<CheckBox>>,
    /// Java `private final String title`.
    title: Option<String>,
    /// Java `private final String[] buttonLabels`.
    button_labels: Option<Vec<String>>,
    /// Java `private final int optionType`.
    option_type: i32,
    /// Java `private final int messageType`.
    message_type: i32,
    /// Java `private final boolean formatMessage`.
    format_message: bool,
    /// Java `private final FieldDisplayer fieldDisplayer1`.
    field_displayer1: Option<Rc<dyn FieldDisplayer>>,
    /// Java `private final FieldDisplayer fieldDisplayer2`.
    field_displayer2: Option<Rc<dyn FieldDisplayer>>,

    /// Java `private Object value = null`.
    value: RefCell<Option<PopupValue>>,

    /// Java `this`, for the `Runnable`s `open()` queues.
    this: Weak<Popup>,
}

impl Popup {
    /// Java private constructor `Popup(UIComponent, int, String, String, String,
    /// int, String[], boolean, FieldDisplayer, FieldDisplayer)`.
    fn new(
        ui_component: Option<&dyn UIComponent>,
        message_type: i32,
        title: Option<&str>,
        message: Option<&str>,
        check_box_title: Option<&str>,
        option_type: i32,
        button_labels: Option<Vec<String>>,
        format_message: bool,
        field_displayer1: Option<Rc<dyn FieldDisplayer>>,
        field_displayer2: Option<Rc<dyn FieldDisplayer>>,
    ) -> Rc<Popup> {
        let component = if let Some(ui_component) = ui_component {
            // `SwingComponent swingComponent = uiComponent.getUIComponent();` is
            // never null in the Rust trait, so `swingComponent.getComponent()`.
            let swing_component = ui_component.get_ui_component();
            Some(swing_component.get_component())
        } else {
            None
        };
        let check_box = if check_box_title.is_some() {
            Some(CheckBox::new_string(check_box_title))
        } else {
            None
        };
        Rc::new_cyclic(|this| Popup {
            message_type,
            format_message,
            field_displayer1,
            field_displayer2,
            component,
            title: title.map(str::to_owned),
            message: message.map(str::to_owned),
            option_type,
            button_labels,
            check_box,
            value: RefCell::new(None),
            this: this.clone(),
        })
    }

    /// Java static `getErrorInstance(UIComponent, String, String, FieldDisplayer,
    /// FieldDisplayer)`.
    pub fn get_error_instance(
        ui_component: Option<&dyn UIComponent>,
        title: Option<&str>,
        message: Option<&str>,
        field_displayer1: Option<Rc<dyn FieldDisplayer>>,
        field_displayer2: Option<Rc<dyn FieldDisplayer>>,
    ) -> Rc<Popup> {
        Popup::new(
            ui_component,
            ERROR_MESSAGE,
            title,
            message,
            None,
            DEFAULT_OPTION,
            None,
            true,
            field_displayer1,
            field_displayer2,
        )
    }

    /// Java static `getYesNoInstance(UIComponent, String, String, String)`.  Use
    /// this instance by passing it to UIHarness.openPopup, or calling open or log.
    pub fn get_yes_no_instance(
        ui_component: Option<&dyn UIComponent>,
        title: Option<&str>,
        message: Option<&str>,
        check_box_message: Option<&str>,
    ) -> Rc<Popup> {
        Popup::new(
            ui_component,
            QUESTION_MESSAGE,
            title,
            message,
            check_box_message,
            YES_NO_OPTION,
            None,
            true,
            None,
            None,
        )
    }

    /// Java static `getCustomButtonInstance(UIComponent, String, String, String,
    /// String[])`.  Use this instance by passing it to UIHarness.openPopup, or
    /// calling open or log.  Uses custom buttons and can have a checkbox.
    pub fn get_custom_button_instance(
        ui_component: Option<&dyn UIComponent>,
        title: Option<&str>,
        message: Option<&str>,
        check_box_message: Option<&str>,
        button_labels: Option<Vec<String>>,
    ) -> Rc<Popup> {
        Popup::new(
            ui_component,
            QUESTION_MESSAGE,
            title,
            message,
            check_box_message,
            DEFAULT_OPTION,
            button_labels,
            true,
            None,
            None,
        )
    }

    /// Java static `getUnformattedErrorInstance(UIComponent, String, String)`.
    pub fn get_unformatted_error_instance(
        ui_component: Option<&dyn UIComponent>,
        title: Option<&str>,
        message: Option<&str>,
    ) -> Rc<Popup> {
        Popup::new(
            ui_component,
            ERROR_MESSAGE,
            title,
            message,
            None,
            DEFAULT_OPTION,
            None,
            false,
            None,
            None,
        )
    }

    /// Java `open()`.
    pub fn open(&self) {
        // SwingUtilities.invokeLater(new Runnable() { run() { fieldDisplayer1.display();
        // fieldDisplayer2.display(); } });
        let field_displayer1 = self.field_displayer1.clone().map(EdtRef::new);
        let field_displayer2 = self.field_displayer2.clone().map(EdtRef::new);
        event_queue::invoke_later(move || {
            if let Some(field_displayer1) = &field_displayer1 {
                field_displayer1.get().display_void();
            }
            if let Some(field_displayer2) = &field_displayer2 {
                field_displayer2.get().display_void();
            }
        });
        if self.message_type == QUESTION_MESSAGE {
            // Leave popup on the current thread so it can block execution until it has
            // been closed.
            self.display();
        } else {
            // SwingUtilities.invokeLater(new Runnable() { run() { display(); } });
            if let Some(this) = self.this.upgrade() {
                let this = EdtRef::new(this);
                event_queue::invoke_later(move || {
                    this.get().display();
                });
            }
        }
    }

    /// Java `display()`.
    pub fn display(&self) {
        // `Object popupMessage`: the lines the pane shows.  A formatted message is
        // the wrapped array (null when empty); an unformatted one is the message
        // string itself, one object.
        let popup_message: Option<Vec<String>> = if self.format_message {
            // Wrap the message
            let wrapped_message = popup_tool::wrap_message(self.message.as_deref(), None);
            let message_array: Option<Vec<String>> = if wrapped_message.is_empty() {
                None
            } else if wrapped_message.len() == 1 {
                Some(vec![wrapped_message[0].clone()])
            } else {
                Some(wrapped_message)
            };
            message_array
        } else {
            self.message.clone().map(|message| vec![message])
        };
        // Create the pane.
        // `if (checkBox != null) popupMessage = new Object[] { popupMessage,
        // Box.createRigidArea(FixedDim.x0_y10), checkBox.getComponent() };`: the
        // check box should travel with the request (see NEEDS below); the rigid area is Swing
        // layout.
        let check_box_component = self
            .check_box
            .as_ref()
            .map(|check_box| check_box.get_component());
        // final JOptionPane pane = new JOptionPane(popupMessage, messageType,
        // optionType, /*icon*/null, buttonLabels, /*Object initialValue*/null);
        // With null button labels the pane shows the look and feel's default
        // buttons for the option type.
        let buttons: Vec<String> = match &self.button_labels {
            Some(button_labels) => button_labels.clone(),
            None => match self.option_type {
                YES_NO_OPTION => vec!["Yes".to_owned(), "No".to_owned()],
                super::abstract_frame::YES_NO_CANCEL_OPTION => {
                    vec!["Yes".to_owned(), "No".to_owned(), "Cancel".to_owned()]
                }
                super::abstract_frame::OK_CANCEL_OPTION => {
                    vec!["OK".to_owned(), "Cancel".to_owned()]
                }
                _ => vec!["OK".to_owned()],
            },
        };
        // Build and display the dialog.
        // final JDialog dialog = pane.createDialog(component, title);
        if self.component.is_some() {
            // Swing layout: adjust the location of the dialog so it will be entirely
            // visible (PopupTool.adjustLocationY(location.y, component.getHeight(),
            // dialog.getHeight()); dialog.setLocation(location)).
        }
        // Give the dialog a name for uitest
        let name = utilities::convert_label_to_name(
            self.title.as_deref(),
            UITestFieldType::POPUP.is_unlimited_segments(),
        );
        // pane.setName(name): carried by the request.
        self.print_name(name.as_deref());
        // Display dialog
        // dialog.setDefaultCloseOperation(JDialog.DISPOSE_ON_CLOSE);
        // dialog.setVisible(true); dialog.toFront();
        let request = PopupRequest {
            name,
            title: self.title.clone(),
            message: popup_message.unwrap_or_default(),
            options: buttons.clone(),
            option_type: self.option_type,
            message_type: self.message_type,
            initial_value: None,
            axis_id: None,
            parent_component: self.component.clone(),
        };
        // NEEDS: ui_harness::PopupRequest has no field for the check box the
        // Java adds to the pane's message array (`new Object[] { message,
        // checkBox }`); until it has `check_box: Option<Rc<JComponent>>`, the
        // responder cannot see or toggle it and isChecked() reads its initial
        // (unselected) state.
        let _ = check_box_component;
        let answer = ui_harness::present_popup(&request);
        // Get the selected value
        // value = pane.getValue(): the index of the default button pressed when no
        // labels were passed, the label of the button pressed otherwise, null when
        // the dialog was closed.
        let value = match answer {
            PopupAnswer::Closed => None,
            PopupAnswer::Selected(index) => match &self.button_labels {
                None => Some(PopupValue::Integer(index as i32)),
                Some(button_labels) => button_labels
                    .get(index)
                    .map(|label| PopupValue::String(label.clone())),
            },
        };
        *self.value.borrow_mut() = value;
    }

    /// Java private final `printName(String)`.
    fn print_name(&self, name: Option<&str>) {
        if etomo_director::ARGUMENTS.lock().unwrap().is_print_names() {
            // print popup name/value pair
            let mut builder = format!(
                "{}{}{} {} ",
                UITestFieldType::POPUP,
                SEPARATOR_CHAR,
                name.unwrap_or("null"),
                DEFAULT_DELIMITER
            );
            // if there are options, then print a popup name/value pair
            if self.option_type == YES_NO_OPTION {
                builder.push_str("Yes,No");
            }
            println!("{}", builder);
        }
    }

    /// Java package-private `log()`.
    pub fn log(&self) {
        eprintln!("Popup:{}", self.message.as_deref().unwrap_or("null"));
        eprintln!();
        // System.err.flush(): eprintln! is unbuffered.
        if etomo_director::is_test_failed() {
            *self.value.borrow_mut() = Some(PopupValue::Integer(YES_OPTION));
        }
    }

    /// Java `getValue()`.
    pub fn get_value(&self) -> Option<PopupValue> {
        self.value.borrow().clone()
    }

    /// Java `isYes()`.
    pub fn is_yes(&self) -> bool {
        match &*self.value.borrow() {
            Some(PopupValue::Integer(value)) => *value == YES_OPTION,
            _ => false,
        }
    }

    /// Java `getSelectedButtonIndex()`.
    pub fn get_selected_button_index(&self) -> i32 {
        let value = self.value.borrow().clone();
        let Some(value) = value else {
            return -1;
        };
        if let PopupValue::Integer(value) = value {
            return value;
        }
        let (Some(button_labels), PopupValue::String(value)) = (&self.button_labels, &value) else {
            return -1;
        };
        for (i, button_label) in button_labels.iter().enumerate() {
            if value == button_label {
                return i as i32;
            }
        }
        -1
    }

    /// Java `isCheckboxSelected()`.
    pub fn is_checkbox_selected(&self) -> bool {
        self.check_box
            .as_ref()
            .is_some_and(|check_box| check_box.is_selected())
    }
}
