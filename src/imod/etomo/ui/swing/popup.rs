//! `IMOD/Etomo/src/etomo/ui/swing/Popup.java`.
#![allow(dead_code)]

use super::{
    abstract_frame::{DEFAULT_OPTION, ERROR_MESSAGE, QUESTION_MESSAGE, YES_NO_OPTION},
    check_box::CheckBox,
};
use crate::imod::etomo::{
    etomo_director::{ARGUMENTS, INSTANCE},
    logic::popup_tool,
    storage::autodoc::autodoc_tokenizer::{DEFAULT_DELIMITER, SEPARATOR_CHAR},
    util::utilities,
};

pub const YES_OPTION: i32 = 0;

/// Source-observable native parent `Component` state.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ComponentBoundary {
    pub height: i32,
}

/// Direct `FieldDisplayer` dependency boundary.  Its `display` call is retained
/// so headless and native frontends observe the source ordering.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct FieldDisplayer {
    pub display_count: usize,
}
impl FieldDisplayer {
    /// Java `display()`.
    pub fn display(&mut self) {
        self.display_count += 1;
    }
}

/// Java's untyped `JOptionPane.getValue()` result, restricted to values this
/// source unit inspects.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PopupValue {
    Integer(i32),
    String(String),
    Other,
}

/// Native `JOptionPane`/`JDialog` state, retained as one presentation boundary.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct JOptionPaneBoundary {
    pub message: Vec<String>,
    pub checkbox_present: bool,
    pub message_type: i32,
    pub option_type: i32,
    pub button_labels: Option<Vec<String>>,
    pub name: Option<String>,
    pub visible: bool,
    pub to_front: bool,
    pub location_y: Option<i32>,
    pub selected_value: Option<PopupValue>,
}

/// Java public final `Popup`.
#[derive(Clone, Debug)]
pub struct Popup {
    pub component: Option<ComponentBoundary>,
    pub message: String,
    pub check_box: Option<CheckBox>,
    pub title: String,
    pub button_labels: Option<Vec<String>>,
    pub option_type: i32,
    pub message_type: i32,
    pub format_message: bool,
    pub field_displayer1: Option<FieldDisplayer>,
    pub field_displayer2: Option<FieldDisplayer>,
    pub value: Option<PopupValue>,
    pub pane: Option<JOptionPaneBoundary>,
    pub open_queued: bool,
}

impl Popup {
    /// Java private constructor.
    pub fn new(
        component: Option<ComponentBoundary>,
        message_type: i32,
        title: &str,
        message: &str,
        check_box_title: Option<&str>,
        option_type: i32,
        button_labels: Option<Vec<String>>,
        format_message: bool,
        field_displayer1: Option<FieldDisplayer>,
        field_displayer2: Option<FieldDisplayer>,
    ) -> Self {
        Self {
            component,
            message: message.into(),
            check_box: check_box_title.map(|title| {
                let mut check_box = CheckBox::new();
                check_box.set_text(Some(title));
                check_box
            }),
            title: title.into(),
            button_labels,
            option_type,
            message_type,
            format_message,
            field_displayer1,
            field_displayer2,
            value: None,
            pane: None,
            open_queued: false,
        }
    }

    /// Java `getErrorInstance`.
    pub fn get_error_instance(
        component: Option<ComponentBoundary>,
        title: &str,
        message: &str,
        field_displayer1: Option<FieldDisplayer>,
        field_displayer2: Option<FieldDisplayer>,
    ) -> Self {
        Self::new(
            component,
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

    /// Java `getYesNoInstance`.
    pub fn get_yes_no_instance(
        component: Option<ComponentBoundary>,
        title: &str,
        message: &str,
        check_box_message: Option<&str>,
    ) -> Self {
        Self::new(
            component,
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

    /// Java `getCustomButtonInstance`.
    pub fn get_custom_button_instance(
        component: Option<ComponentBoundary>,
        title: &str,
        message: &str,
        check_box_message: Option<&str>,
        button_labels: Vec<String>,
    ) -> Self {
        Self::new(
            component,
            QUESTION_MESSAGE,
            title,
            message,
            check_box_message,
            DEFAULT_OPTION,
            Some(button_labels),
            true,
            None,
            None,
        )
    }

    /// Java `getUnformattedErrorInstance`.
    pub fn get_unformatted_error_instance(
        component: Option<ComponentBoundary>,
        title: &str,
        message: &str,
    ) -> Self {
        Self::new(
            component,
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

    /// Java `open()`.  A question is displayed synchronously; other message
    /// kinds are queued at the native Swing boundary and represented immediately.
    pub fn open(&mut self) {
        if let Some(displayer) = &mut self.field_displayer1 {
            displayer.display();
        }
        if let Some(displayer) = &mut self.field_displayer2 {
            displayer.display();
        }
        if self.message_type == QUESTION_MESSAGE {
            self.display();
        } else {
            self.open_queued = true;
            self.display();
        }
    }

    /// Java `display()`.
    pub fn display(&mut self) {
        let message = if self.format_message {
            popup_tool::wrap_message(Some(&self.message), None)
        } else {
            vec![self.message.clone()]
        };
        let name = utilities::convert_label_to_name(Some(&self.title), true).unwrap_or_default();
        self.print_name(&name);
        let selected_value = self
            .pane
            .as_ref()
            .and_then(|pane| pane.selected_value.clone());
        self.pane = Some(JOptionPaneBoundary {
            message,
            checkbox_present: self.check_box.is_some(),
            message_type: self.message_type,
            option_type: self.option_type,
            button_labels: self.button_labels.clone(),
            name: Some(name),
            visible: true,
            to_front: true,
            location_y: self
                .component
                .as_ref()
                .map(|component| popup_tool::adjust_location_y(0, component.height, 0)),
            selected_value: selected_value.clone(),
        });
        self.value = selected_value;
    }

    /// Java private `printName(String)`.
    pub fn print_name(&self, name: &str) {
        if ARGUMENTS.lock().unwrap().is_print_names() {
            let mut output = format!("popup{SEPARATOR_CHAR}{name} {DEFAULT_DELIMITER} ");
            if self.option_type == YES_NO_OPTION {
                output.push_str("Yes,No");
            }
            println!("{output}");
        }
    }

    /// Java `log()`.
    pub fn log(&mut self) {
        eprintln!("Popup:{}\n", self.message);
        if INSTANCE.lock().unwrap().test_failed {
            self.value = Some(PopupValue::Integer(YES_OPTION));
        }
    }

    /// Native `JOptionPane` completion boundary: set the exact value Java would
    /// subsequently receive from `pane.getValue()`.
    pub fn set_pane_value(&mut self, value: Option<PopupValue>) {
        if let Some(pane) = &mut self.pane {
            pane.selected_value = value.clone();
        }
        self.value = value;
    }

    /// Java `getValue()`.
    pub fn get_value(&self) -> Option<&PopupValue> {
        self.value.as_ref()
    }

    /// Java `isYes()`.
    pub fn is_yes(&self) -> bool {
        self.value == Some(PopupValue::Integer(YES_OPTION))
    }

    /// Java `getSelectedButtonIndex()`.
    pub fn get_selected_button_index(&self) -> i32 {
        match &self.value {
            None => -1,
            Some(PopupValue::Integer(value)) => *value,
            Some(PopupValue::String(value)) => self
                .button_labels
                .as_ref()
                .and_then(|labels| {
                    labels
                        .iter()
                        .position(|label| label == value)
                        .map(|index| index as i32)
                })
                .unwrap_or(-1),
            Some(PopupValue::Other) => -1,
        }
    }

    /// Java `isCheckboxSelected()`.
    pub fn is_checkbox_selected(&self) -> bool {
        self.check_box.as_ref().is_some_and(CheckBox::is_selected)
    }
}

/// Native queue callback for the first Java `SwingUtilities.invokeLater`
/// runnable in `Popup.open`.
pub struct FieldDisplayRunnable;

impl FieldDisplayRunnable {
    #[allow(non_snake_case)]
    pub fn run(popup: &mut Popup) {
        if let Some(displayer) = &mut popup.field_displayer1 {
            displayer.display();
        }
        if let Some(displayer) = &mut popup.field_displayer2 {
            displayer.display();
        }
    }
}

/// Native queue callback for the non-question popup display runnable.
pub struct DisplayRunnable;

impl DisplayRunnable {
    #[allow(non_snake_case)]
    pub fn run(popup: &mut Popup) {
        popup.display();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn custom_button_value_has_source_index() {
        let mut popup = Popup::get_custom_button_instance(
            None,
            "Question",
            "Use it?",
            None,
            vec!["One".into(), "Two".into()],
        );
        popup.open();
        popup.set_pane_value(Some(PopupValue::String("Two".into())));
        assert_eq!(popup.get_selected_button_index(), 1);
    }
    #[test]
    fn formatted_popup_displays_field_displayers_before_pane() {
        let mut popup = Popup::get_error_instance(
            None,
            "Problem",
            "a long message",
            Some(FieldDisplayer::default()),
            Some(FieldDisplayer::default()),
        );
        popup.open();
        assert_eq!(popup.field_displayer1.unwrap().display_count, 1);
        assert_eq!(popup.pane.unwrap().name.as_deref(), Some("problem"));
    }
}
