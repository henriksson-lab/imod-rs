//! `IMOD/Etomo/src/etomo/ui/swing/ProcessControlPanel.java`.
//!
//! One process button of the tomogram process panel: a toggle button whose
//! label shows the process name and, below it, the process state ("Not
//! Started", "In Progress", "Complete") in the state's colour.
//!
//! `java.awt.Color` is represented as the RGB triple the Swing stand-in
//! (`jdk.rs`) uses for foreground colours.

use std::cell::RefCell;
use std::rc::Rc;

use super::colored_state_text::ColoredStateText;
use super::generic_mouse_adapter::GenericMouseAdapter;
use super::simple_toggle_button::SimpleToggleButton;
use super::tooltip_formatter;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::jdk::{ActionListener, JComponent};
use crate::imod::etomo::process::process_state::ProcessState;
use crate::imod::etomo::r#type::dialog_type::DialogType;

// Java `static Dimension dimPanelProcess = FixedDim.processPanel;` - Swing
// layout (a size), not modelled.

/// Java `static String[] textStates`.
pub static TEXT_STATES: [&str; 3] = ["Not Started", "In Progress", "Complete"];
/// Java `colorNotStarted = new Color(0.75f, 0.0f, 0.0f)`; `Color(float...)`
/// stores `(int)(v * 255 + 0.5)`.
pub const COLOR_NOT_STARTED: (u8, u8, u8) = (191, 0, 0);
/// Java `colorInProgress = new Color(0.75f, 0.0f, 0.75f)`.
pub const COLOR_IN_PROGRESS: (u8, u8, u8) = (191, 0, 191);
// static Color colorComplete = new Color(0.0f, 0.75f, 0.0f);
/// Java `colorComplete = new Color(0, 153, 0)`.
pub const COLOR_COMPLETE: (u8, u8, u8) = (0, 153, 0);
/// Java `static Color[] colorState`.
pub static COLOR_STATE: [(u8, u8, u8); 3] = [COLOR_NOT_STARTED, COLOR_IN_PROGRESS, COLOR_COMPLETE];

/// Java public class `ProcessControlPanel`.
pub struct ProcessControlPanel {
    /// Java `command`.
    command: String,
    /// Java `panelRoot = new JPanel()`.
    panel_root: Rc<JComponent>,
    /// Java `buttonRun = new SimpleToggleButton()`.
    button_run: Rc<SimpleToggleButton>,
    /// Java `panelState`: declared and never assigned in the Java.
    #[allow(dead_code)]
    panel_state: Option<Rc<JComponent>>,
    /// Java `highlightState`; stays null when its construction throws.
    highlight_state: RefCell<Option<ColoredStateText>>,
    /// Java `dialogType`.
    dialog_type: DialogType,
}

impl ProcessControlPanel {
    /// Java `ProcessControlPanel(DialogType)`.
    pub fn new(dialog_type: DialogType) -> Rc<ProcessControlPanel> {
        let compact_display = etomo_director::INSTANCE.with_user_configuration(|c| c.get_compact_display());
        let command = if compact_display {
            dialog_type.get_compact_label()
        } else {
            dialog_type.to_string()
        };
        let panel_root = JComponent::new_panel();
        // Swing layout: panelRoot.setLayout(new BoxLayout(panelRoot, BoxLayout.Y_AXIS)).

        let mut highlight_state: Option<ColoredStateText> = None;
        let created = if compact_display {
            Ok(ColoredStateText::new_color_array(&COLOR_STATE))
        } else {
            ColoredStateText::new_string_array_color_array(&TEXT_STATES, &COLOR_STATE)
        };
        let result = match created {
            Ok(state) => {
                highlight_state = Some(state);
                highlight_state.as_mut().unwrap().set_selected(0)
            }
            Err(e) => Err(e),
        };
        if let Err(e) = result {
            // e.printStackTrace()
            eprintln!("{e}");
            eprintln!("Unable to create or set highlightState object");
            eprintln!("{}", e.get_message());
        }

        let button_run = SimpleToggleButton::new_void();
        let this = Rc::new(ProcessControlPanel {
            command,
            panel_root,
            button_run,
            panel_state: None,
            highlight_state: RefCell::new(highlight_state),
            dialog_type,
        });
        this.panel_root.add(&this.button_run.get_component());
        this.button_run
            .get_component()
            .set_action_command(Some(&dialog_type.to_string()));
        this.update_label();
        this
    }

    /// Java `getCommand()`.
    pub fn get_command(&self) -> String {
        self.dialog_type.to_string()
    }

    /// Java `getDialogType()`.
    pub fn get_dialog_type(&self) -> DialogType {
        self.dialog_type
    }

    /// Java `setButtonActionListener(ActionListener)`.
    pub fn set_button_action_listener(&self, action_listener: ActionListener) {
        self.button_run
            .get_component()
            .add_action_listener(action_listener);
    }

    /// Java `getContainer()`.
    pub fn get_container(&self) -> Rc<JComponent> {
        self.panel_root.clone()
    }

    /// Java `setState(ProcessState)`.
    pub fn set_state(&self, state: ProcessState) {
        let result = {
            let mut highlight_state = self.highlight_state.borrow_mut();
            // Upstream bug fixed in translation (ProcessControlPanel.java:128-138):
            // when the constructor's ColoredStateText creation threw, highlightState
            // is null and the Java throws a NullPointerException here.  The
            // translation skips the selection instead (unreachable with the
            // source's constant arrays, which always agree in length).
            match highlight_state.as_mut() {
                None => Ok(()),
                Some(highlight_state) => {
                    let mut result = Ok(());
                    if state == ProcessState::NotStarted {
                        result = highlight_state.set_selected(0);
                    }
                    if result.is_ok() && state == ProcessState::InProgress {
                        result = highlight_state.set_selected(1);
                    }
                    if result.is_ok() && state == ProcessState::Complete {
                        result = highlight_state.set_selected(2);
                    }
                    result
                }
            }
        };
        if let Err(e) = result {
            // e.printStackTrace()
            eprintln!("{e}");
            eprintln!("Unable to set highlightState object");
            eprintln!("{}", e.get_message());
        }
        self.update_label();
    }

    /// Java `setSelected(boolean)`.  Set the selected state of the button.
    pub fn set_selected(&self, state: bool) {
        self.button_run.get_component().set_selected(state);
    }

    /// Java private `updateLabel()`.
    fn update_label(&self) {
        // Upstream bug fixed in translation (ProcessControlPanel.java:160-167): a
        // null highlightState (see `set_state`) throws a NullPointerException in
        // the Java; here the label is built without the state line and the
        // foreground is left unchanged.
        let (highlight_text, highlight_color) = match self.highlight_state.borrow().as_ref() {
            Some(highlight_state) => (
                highlight_state.get_selected_text(),
                Some(highlight_state.get_selected_color()),
            ),
            None => (None, None),
        };
        let mut text = format!("<HTML><CENTER>{}", self.command);
        if let Some(highlight_text) = highlight_text {
            text.push_str(&format!("<br>{highlight_text}"));
        }
        self.button_run.set_text(&format!("{text}</CENTER>"));
        if let Some(highlight_color) = highlight_color {
            self.button_run
                .get_component()
                .set_foreground(highlight_color);
        }
    }

    /// Java `addMouseListener(MouseListener)`.
    pub fn add_mouse_listener(&self, listener: &Rc<GenericMouseAdapter>) {
        self.panel_root.add_mouse_listener(listener.clone());
        self.button_run.get_component().add_mouse_listener(listener.clone());
    }

    /// Java `setToolTipText(String)`.
    pub fn set_tool_tip_text(&self, text: &str) {
        let tooltip = tooltip_formatter::INSTANCE.format(Some(text));
        self.panel_root.set_tool_tip_text(tooltip.as_deref());
        self.button_run
            .get_component()
            .set_tool_tip_text(tooltip.as_deref());
    }
}
