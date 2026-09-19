//! `IMOD/Etomo/src/etomo/ui/swing/MultiLineButton.java`.
//!
//! `AbstractButton`, `JButton`, `JToggleButton`, `JLabel`, and their AWT
//! layout/listener calls are an explicit Swing boundary.  This module retains
//! the complete source-owned state and label-dividing algorithm so a Rust
//! frontend can present the same component without changing eTomo policy.
#![allow(dead_code)]

use std::collections::BTreeMap;

use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::file_key::FileKey;
use crate::imod::etomo::r#type::process_end_state::ProcessEndState;
use crate::imod::etomo::r#type::process_result::ProcessResult;
use crate::imod::etomo::util::utilities;

use super::panel::Dimension;
use super::ui_utilities::{AbstractButton, Color, FontMetrics, Icon, Insets, UiUtilities};

pub const SEPARATOR_CHAR: char = '.';
pub const BUTTON_FIELD_TYPE: &str = "bn";
pub const PADDING: i32 = 9;

/// Source-visible state of the Java `AbstractButton` owned by this wrapper.
#[derive(Clone, Debug, PartialEq)]
pub struct ButtonBoundary {
    pub abstract_button: AbstractButton,
    pub text: Option<String>,
    pub name: Option<String>,
    pub action_command: Option<String>,
    pub selected: bool,
    pub enabled: bool,
    pub visible: bool,
    pub displayable: bool,
    pub focusable: bool,
    pub foreground: Option<Color>,
    pub background: Option<Color>,
    pub margin: Insets,
    pub minimum_size: Option<Dimension>,
    pub width: i32,
    pub height: i32,
    pub alignment_x: f32,
    pub alignment_y: f32,
    pub icon: Option<Icon>,
    /// `AbstractButton.disabledIcon`; rendering remains a native GUI boundary.
    pub disabled_icon: Option<Icon>,
    pub border: Option<String>,
    pub border_painted: bool,
    pub tooltip: Option<String>,
    pub action_listener_count: usize,
    pub mouse_listener_count: usize,
    pub has_border_layout: bool,
    pub label1: Option<String>,
    pub label2: Option<String>,
    pub click_count: usize,
}

impl Default for ButtonBoundary {
    fn default() -> Self {
        Self {
            abstract_button: AbstractButton::default(),
            text: None,
            name: None,
            action_command: None,
            selected: false,
            enabled: true,
            visible: true,
            displayable: true,
            focusable: true,
            foreground: None,
            background: None,
            margin: Insets::default(),
            minimum_size: None,
            width: 0,
            height: 0,
            alignment_x: 0.5,
            alignment_y: 0.5,
            icon: None,
            disabled_icon: None,
            border: None,
            border_painted: true,
            tooltip: None,
            action_listener_count: 0,
            mouse_listener_count: 0,
            has_border_layout: false,
            label1: None,
            label2: None,
            click_count: 0,
        }
    }
}

/// Java `BaseScreenState` calls used by `MultiLineButton`.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct BaseScreenState {
    pub button_states: BTreeMap<String, bool>,
}
impl BaseScreenState {
    pub fn get_button_state(&self, key: Option<&str>) -> bool {
        key.and_then(|key| self.button_states.get(key))
            .copied()
            .unwrap_or(false)
    }
    pub fn set_button_state(&mut self, key: Option<&str>, state: bool) {
        if let Some(key) = key {
            self.button_states.insert(key.to_owned(), state);
        }
    }
}

/// State owned by Java `ProcessResultDisplayState` at this unit's dependency boundary.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ProcessResultDisplayState {
    pub debug: bool,
    pub original_state: bool,
    /// Java `secondaryProcess`: changes failed-to-start handling after the
    /// first command in a process series.
    pub secondary_process: bool,
    /// Java `processRunning`: terminal messages are ignored until a process
    /// has actually started, preventing stale callbacks from changing a UI.
    pub process_running: bool,
    pub display_id: i32,
    pub factory_id: Option<String>,
    pub next_present: bool,
    pub use_global_dependency_list: bool,
    pub dependent_display_count: usize,
    pub failure_display_count: usize,
    pub success_display_count: usize,
    pub messages: Vec<&'static str>,
}
impl ProcessResultDisplayState {
    pub fn equals_id(&self, display_id: i32, factory_id: Option<&str>) -> bool {
        self.display_id == display_id && self.factory_id.as_deref() == factory_id
    }
}

/// Java `MultiLineButton`.
#[derive(Clone, Debug, PartialEq)]
pub struct MultiLineButton {
    pub button: ButtonBoundary,
    pub toggle_button: bool,
    pub process_result_display_state: ProcessResultDisplayState,
    pub background: Option<Color>,
    pub html: bool,
    pub screen_state: Option<BaseScreenState>,
    pub dialog_type: Option<DialogType>,
    pub state_key: Option<String>,
    pub manual_name: bool,
    pub button_foreground: Option<Color>,
    pub button_highlight_foreground: Option<Color>,
    pub debug: bool,
    pub unformatted_label: Option<String>,
    pub font_metrics: Option<FontMetrics>,
    pub enabled: bool,
    pub editable: bool,
    pub width: i32,
    pub action_command_set: bool,
    pub output_image_file_key: Option<FileKey>,
}

impl MultiLineButton {
    #[allow(non_snake_case)]
    /// Java diagnostic `dumpState`, returned for the Rust frontend/log sink.
    pub fn dumpState(&self) -> String {
        format!(
            "[toggleButton:{},stateKey:{:?},\nmanualName:{},buttonForeground:{:?},\nbuttonHighlightForeground:{:?},debug:{},\nunformattedLabel:{:?}]",
            self.toggle_button,
            self.state_key,
            self.manual_name,
            self.button_foreground,
            self.button_highlight_foreground,
            self.debug,
            self.unformatted_label,
        )
    }

    /// Java package-private `MultiLineButton()`.
    pub fn new() -> Self {
        Self::new_full(None, false, None, false, false, false, None)
    }
    /// Java package-private `MultiLineButton(String)`.
    pub fn new_with_label(label: Option<&str>) -> Self {
        Self::new_full(label, false, None, false, false, false, None)
    }
    /// Java package-private `MultiLineButton(String, FileKey)`.
    pub fn new_with_output_image_file_key(label: Option<&str>, key: Option<FileKey>) -> Self {
        Self::new_full(label, false, None, false, false, false, key)
    }
    /// Java package-private `MultiLineButton(boolean, String)`.
    pub fn new_with_minimum_size(set_minimum_size: bool, label: Option<&str>) -> Self {
        Self::new_full(label, false, None, set_minimum_size, false, false, None)
    }
    /// Java static `getDebugInstance(String)`.
    pub fn get_debug_instance(label: Option<&str>) -> Self {
        Self::new_full(label, false, None, false, false, true, None)
    }
    /// Java private `MultiLineButton(String, boolean)`.
    pub fn new_toggle(label: Option<&str>, toggle_button: bool) -> Self {
        Self::new_full(label, toggle_button, None, false, false, true, None)
    }
    /// Java primary constructor.
    pub fn new_full(
        label: Option<&str>,
        toggle_button: bool,
        dialog_type: Option<DialogType>,
        set_minimum_size: bool,
        html: bool,
        debug: bool,
        output_image_file_key: Option<FileKey>,
    ) -> Self {
        let mut value = Self {
            button: Self::new_button(toggle_button),
            toggle_button,
            process_result_display_state: ProcessResultDisplayState::default(),
            background: None,
            html,
            screen_state: None,
            dialog_type,
            state_key: None,
            manual_name: false,
            button_foreground: None,
            button_highlight_foreground: None,
            debug,
            unformatted_label: label.map(str::to_owned),
            font_metrics: None,
            enabled: true,
            editable: true,
            width: -1,
            action_command_set: false,
            output_image_file_key,
        };
        value.setup_button(set_minimum_size);
        value.init();
        value.background = value.button.background;
        value
    }
    /// Java `getOutputImageFileKey`.
    pub fn get_output_image_file_key(&self) -> Option<&FileKey> {
        self.output_image_file_key.as_ref()
    }
    /// Java `setOutputImageFileKey`.
    pub fn set_output_image_file_key(&mut self, key: Option<FileKey>) {
        self.output_image_file_key = key;
    }
    /// Java `newButton`.
    pub fn new_button(_toggle_button: bool) -> ButtonBoundary {
        ButtonBoundary::default()
    }
    /// Java `isHtml`.
    pub fn is_html(&self) -> bool {
        self.html
    }
    /// Java `doClick`.
    pub fn do_click(&mut self) {
        self.button.click_count += 1;
        if self.toggle_button {
            self.set_selected(!self.button.selected);
        }
    }
    /// Java `setupButton`.
    pub fn setup_button(&mut self, set_minimum_size: bool) {
        self.set_size(set_minimum_size);
        if let Some(label) = self.unformatted_label.clone() {
            self.set_text(&label);
        }
    }
    /// Java `setHighlight`; Java body is empty.
    pub fn set_highlight(&mut self, _highlight: bool) {}
    /// Java static `getToggleButtonInstance()`.
    pub fn get_toggle_button_instance() -> Self {
        Self::new_full(None, true, None, false, false, false, None)
    }
    /// Java static `getToggleButtonInstance(String, DialogType)`.
    pub fn get_toggle_button_instance_with_dialog(
        label: Option<&str>,
        dialog_type: Option<DialogType>,
    ) -> Self {
        Self::new_full(label, true, dialog_type, false, false, false, None)
    }
    /// Java static `getToggleButtonInstance(String)`.
    pub fn get_toggle_button_instance_with_label(label: Option<&str>) -> Self {
        Self::new_toggle(label, true)
    }
    /// Java `setFocusable`.
    pub fn set_focusable(&mut self, input: bool) {
        self.button.focusable = input;
    }
    /// Java `getPreferredWidth()`.
    pub fn get_preferred_width(&self) -> i32 {
        UiUtilities::get_preferred_width_button(
            &self.button.abstract_button,
            self.unformatted_label.as_deref(),
        )
    }
    /// Java `setDebug`.
    pub fn set_debug(&mut self, input: bool) {
        self.debug = input;
        self.process_result_display_state.debug = input;
    }
    /// Java `isDebug`.
    pub fn is_debug(&self) -> bool {
        self.debug
    }
    /// Java `getPreferredWidth(String)`.
    pub fn get_preferred_width_text(&self, text: Option<&str>) -> i32 {
        UiUtilities::get_preferred_width_button(&self.button.abstract_button, text)
    }
    /// Java `getWidth`.
    pub fn get_width(&self) -> i32 {
        self.button.width
    }
    /// Java `createButtonStateKey`.
    pub fn create_button_state_key(&mut self, dialog_type: Option<DialogType>) -> Option<String> {
        if let Some(dialog_type) = dialog_type {
            let name = self.button.name.clone().unwrap_or_default();
            self.state_key = Some(format!("{}.{}.done", dialog_type.get_storable_name(), name));
        }
        self.state_key.clone()
    }
    /// Java `getButtonStateKey`.
    pub fn get_button_state_key(&mut self) -> Option<String> {
        if self.state_key.is_none() && self.dialog_type.is_some() {
            self.create_button_state_key(self.dialog_type);
        }
        self.state_key.clone()
    }
    /// Java `setButtonState`.
    pub fn set_button_state(&mut self, state: bool) {
        self.set_original_process_result_display_state(state);
        self.set_selected(state);
    }
    /// Java `setOriginalProcessResultDisplayState`.
    pub fn set_original_process_result_display_state(&mut self, state: bool) {
        self.process_result_display_state.original_state = state;
    }
    /// Java `getButtonState`.
    pub fn get_button_state(&self) -> bool {
        self.is_selected()
    }
    /// Java `equalsID`.
    pub fn equals_id(&self, display_id: i32, factory_id: Option<&str>) -> bool {
        self.process_result_display_state
            .equals_id(display_id, factory_id)
    }
    /// Java `setID`.
    pub fn set_id(&mut self, display_id: i32, factory_id: Option<&str>) {
        self.process_result_display_state.display_id = display_id;
        self.process_result_display_state.factory_id = factory_id.map(str::to_owned);
    }
    /// Java `setFactoryID`.
    pub fn set_factory_id(&mut self, factory_id: Option<&str>) {
        self.process_result_display_state.factory_id = factory_id.map(str::to_owned);
    }
    /// Java `getDisplayID`.
    pub fn get_display_id(&self) -> i32 {
        self.process_result_display_state.display_id
    }
    /// Java `getFactoryID`.
    pub fn get_factory_id(&self) -> Option<&str> {
        self.process_result_display_state.factory_id.as_deref()
    }
    /// Java `getNext`; downstream instance identity remains a display boundary.
    pub fn get_next(&self) -> bool {
        self.process_result_display_state.next_present
    }
    /// Java `setNext`.
    pub fn set_next(&mut self, present: bool) {
        self.process_result_display_state.next_present = present;
    }
    /// Java `setUseGlobalDependencyList`.
    pub fn set_use_global_dependency_list(&mut self, use_global: bool) {
        self.process_result_display_state.use_global_dependency_list = use_global;
    }
    /// Java `setManualName`.
    pub fn set_manual_name(&mut self) {
        self.manual_name = true;
    }
    /// Java `setIcon`.
    pub fn set_icon(&mut self, icon: Option<Icon>) {
        self.button.icon = icon;
        self.button.abstract_button.icon = icon;
    }
    /// Java `setName`.
    pub fn set_name(&mut self, label: Option<&str>) {
        let name = utilities::convert_label_to_name(label, false).unwrap_or_default();
        self.button.name = Some(format!("{BUTTON_FIELD_TYPE}{SEPARATOR_CHAR}{name}"));
    }
    /// Java `getButton`; direct native widget state.
    pub fn get_button(&self) -> &ButtonBoundary {
        &self.button
    }
    /// Java `getName`.
    pub fn get_name(&self) -> Option<&str> {
        self.button.name.as_deref()
    }
    /// Java `setStateKey`.
    pub fn set_state_key(&mut self, state_key: Option<&str>) {
        self.state_key = state_key.map(str::to_owned);
    }
    /// Java `setEnabled`.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        self.button.enabled = enabled && self.editable;
        if !self.toggle_button || self.html {
            self.button.foreground = self.button_foreground;
        }
    }
    /// Java `setEditable`.
    pub fn set_editable(&mut self, editable: bool) {
        self.editable = editable;
        if self.enabled {
            self.button.enabled = editable;
        }
    }
    /// Java `isEnabled`.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
    /// Java `isToggleButton`.
    pub fn is_toggle_button(&self) -> bool {
        self.toggle_button
    }
    /// Java `setText`.
    pub fn set_text(&mut self, text: &str) {
        if !self.manual_name {
            self.set_name(Some(text));
        }
        self.unformatted_label = Some(text.to_owned());
        self.set_text_label(Some(text));
    }
    /// Java `setTextLabel`.
    pub fn set_text_label(&mut self, text: Option<&str>) {
        let mut text1 = None;
        let mut text2 = None;
        if let Some(text) = text.filter(|text| text.contains("\\n")) {
            let labels: Vec<_> = text.split("\\n").map(str::trim).collect();
            if labels.len() >= 2 {
                text1 = Some(labels[0].to_owned());
                text2 = Some(labels[1].to_owned());
            }
        } else if let Some(text) = text.filter(|_| self.width != -1) {
            if self.font_metrics.is_none() {
                self.font_metrics =
                    UiUtilities::get_font_metrics_button(&self.button.abstract_button);
            }
            if let Some(font_metrics) = self.font_metrics {
                let mut divider = LabelDivider::new(font_metrics, self.width);
                divider.divide(text);
                text1 = divider.get_line1().map(str::to_owned);
                text2 = divider.get_line2().map(str::to_owned);
            }
        }
        if text2.is_none() {
            self.button.text = text.map(str::to_owned);
            // `JButton.getActionCommand()` defaults to its displayed text.
            // Java's chooser relies on that default for one-line labels.
            if !self.action_command_set {
                self.button.action_command = text.map(str::to_owned);
            }
            self.button.label1 = None;
            self.button.label2 = None;
            self.button.has_border_layout = false;
        } else {
            self.button.text = Some(String::new());
            if !self.action_command_set {
                self.button.action_command = text.map(str::to_owned);
            }
            self.button.has_border_layout = true;
            self.button.label1 = text1;
            self.button.label2 = text2;
        }
    }
    /// Java `toString`.
    pub fn to_string(&self) -> Option<&str> {
        self.get_text()
    }
    /// Java `getUnformattedLabel`.
    pub fn get_unformatted_label(&self) -> Option<&str> {
        self.unformatted_label.as_deref()
    }
    /// Java `addActionListener`.
    pub fn add_action_listener(&mut self) {
        self.button.action_listener_count += 1;
    }
    /// Java `setActionCommand`.
    pub fn set_action_command(&mut self, action_command: Option<&str>) {
        self.action_command_set = true;
        self.button.action_command = action_command.map(str::to_owned);
    }
    /// Java `getActionCommand`.
    pub fn get_action_command(&self) -> Option<&str> {
        self.button.action_command.as_deref()
    }
    /// Java `getComponent`; native component boundary.
    pub fn get_component(&self) -> &ButtonBoundary {
        &self.button
    }
    /// Java `setVisible`.
    pub fn set_visible(&mut self, visible: bool) {
        self.button.visible = visible;
    }
    /// Java `setBorder`.
    pub fn set_border(&mut self, border: Option<&str>) {
        self.button.border = border.map(str::to_owned);
    }
    /// Java `setBorderPainted`.
    pub fn set_border_painted(&mut self, border_painted: bool) {
        self.button.border_painted = border_painted;
    }
    /// Java `getHeight`.
    pub fn get_height(&self) -> i32 {
        self.button.height
    }
    /// Java `getBorder`.
    pub fn get_border(&self) -> Option<&str> {
        self.button.border.as_deref()
    }
    /// Java `setToolTipText`; `TooltipFormatter` is presentation boundary.
    pub fn set_tool_tip_text(&mut self, text: Option<&str>) {
        self.button.tooltip = text.map(str::to_owned);
    }
    /// Java `setTooltip`.
    pub fn set_tooltip(&mut self, multi_line_button: &Self) {
        self.button.tooltip = multi_line_button.button.tooltip.clone();
    }
    /// Java `getPreferredSize`.
    pub fn get_preferred_size(&self) -> Option<Dimension> {
        self.button.abstract_button.preferred_size
    }
    /// Java `setAlignmentX`.
    pub fn set_alignment_x(&mut self, alignment_x: f32) {
        self.button.alignment_x = alignment_x;
    }
    /// Java `setAlignmentY`.
    pub fn set_alignment_y(&mut self, alignment_y: f32) {
        self.button.alignment_y = alignment_y;
    }
    /// Java `getQuotedLabel`.
    pub fn get_quoted_label(&self) -> Option<String> {
        utilities::quote_label(self.unformatted_label.as_deref())
    }
    /// Java `getText`.
    pub fn get_text(&self) -> Option<&str> {
        if self.button.label1.is_none() {
            self.button.text.as_deref()
        } else {
            self.unformatted_label.as_deref()
        }
    }
    /// Java `addMouseListener`.
    pub fn add_mouse_listener(&mut self) {
        self.button.mouse_listener_count += 1;
    }
    /// Java `removeActionListener`.
    pub fn remove_action_listener(&mut self) {
        self.button.action_listener_count = self.button.action_listener_count.saturating_sub(1);
    }
    /// Java `getFontMetrics`.
    pub fn get_font_metrics(&self) -> Option<FontMetrics> {
        self.font_metrics
    }
    /// Java `setSize`.
    pub fn set_size(&mut self, set_minimum: bool) {
        let font_metrics = self.button.abstract_button.font_metrics;
        let size = Some(Dimension {
            width: 90,
            height: 27,
        }); // UIParameters widget-theme boundary
        if let Some(size) = size {
            self.width = size.width;
            self.button.width = size.width;
            self.button.height = size.height;
            self.button.abstract_button.preferred_size = Some(size);
            self.button.abstract_button.maximum_size = Some(size);
            if set_minimum {
                self.button.minimum_size = Some(size);
            }
        }
        if self.font_metrics.is_none() {
            self.font_metrics = font_metrics;
        }
    }
    /// Java `setProcessDone`.
    pub fn set_process_done(&mut self, done: bool) {
        self.set_selected(done);
    }
    /// Java `setScreenState`.
    pub fn set_screen_state(&mut self, screen_state: BaseScreenState) {
        self.screen_state = Some(screen_state);
        let key = self.get_button_state_key();
        let state = self
            .screen_state
            .as_ref()
            .unwrap()
            .get_button_state(key.as_deref());
        self.button.selected = state;
    }
    /// Java `setSelected`.
    pub fn set_selected(&mut self, selected: bool) {
        self.button.selected = selected;
        let key = self.get_button_state_key();
        let button_state = self.get_button_state();
        if let Some(screen_state) = self.screen_state.as_mut() {
            screen_state.set_button_state(key.as_deref(), button_state);
        }
    }
    /// Java `getOriginalState`.
    pub fn get_original_state(&self) -> bool {
        !self.is_selected()
    }
    /// Java `isSelected`.
    pub fn is_selected(&self) -> bool {
        self.button.selected
    }
    /// Java `isVisible`.
    pub fn is_visible(&self) -> bool {
        self.button.visible
    }
    /// Java `isDisplayable`.
    pub fn is_displayable(&self) -> bool {
        self.button.displayable
    }
    /// Java `init`.
    pub fn init(&mut self) {
        self.button.margin = Insets {
            top: 2,
            left: 2,
            bottom: 2,
            right: 2,
        };
        self.button.abstract_button.insets = self.button.margin;
    }
    /// Java `msgProcessStarting`.
    pub fn msg_process_starting(&mut self) {
        if !self.process_result_display_state.process_running {
            // Java snapshots `display.getOriginalState()` before it marks the
            // display done, in case launching ultimately fails.
            let original_state = !self.is_selected();
            self.set_process_done(true);
            let state = &mut self.process_result_display_state;
            state.original_state = original_state;
            state.secondary_process = false;
        }
        self.process_result_display_state.process_running = true;
    }
    /// Java `msg(ProcessResult)`.
    pub fn msg(&mut self, process_result: ProcessResult) {
        match process_result {
            ProcessResult::Succeeded => self.msg_process_succeeded(),
            ProcessResult::Failed => self.msg_process_failed(),
            ProcessResult::FailedToStart => self.msg_process_failed_to_start(),
        }
    }
    /// Java `msg(ProcessEndState)`.
    pub fn msg_end_state(&mut self, end_state: ProcessEndState) {
        match end_state {
            ProcessEndState::Done | ProcessEndState::Killed | ProcessEndState::Paused => {
                self.msg_process_succeeded()
            }
            ProcessEndState::Cancelled => self.msg_process_failed_to_start(),
            ProcessEndState::Failed | ProcessEndState::FileLockFailure => self.msg_process_failed(),
        }
    }
    /// Java `msgProcessSucceeded`.
    pub fn msg_process_succeeded(&mut self) {
        if !self.process_result_display_state.process_running {
            return;
        }
        self.set_process_done(true);
        self.process_result_display_state.process_running = false;
    }
    /// Java `msgProcessFailed`.
    pub fn msg_process_failed(&mut self) {
        if !self.process_result_display_state.process_running {
            return;
        }
        self.set_process_done(false);
        self.process_result_display_state.process_running = false;
    }
    /// Java `msgProcessFailedToStart`.
    pub fn msg_process_failed_to_start(&mut self) {
        if !self.process_result_display_state.process_running {
            return;
        }
        if self.process_result_display_state.secondary_process {
            self.msg_process_failed();
        } else {
            self.set_process_done(self.process_result_display_state.original_state);
            self.process_result_display_state.process_running = false;
        }
    }
    /// Java `msgSecondaryProcess`.
    pub fn msg_secondary_process(&mut self) {
        self.process_result_display_state.secondary_process = true;
    }
    /// Java `addDependentDisplay`.
    pub fn add_dependent_display(&mut self) {
        self.process_result_display_state.dependent_display_count += 1;
    }
    /// Java `setOriginalState`.
    pub fn set_original_state(&mut self, original_state: bool) {
        self.process_result_display_state.original_state = original_state;
    }
    /// Java `addFailureDisplay`.
    pub fn add_failure_display(&mut self) {
        self.process_result_display_state.failure_display_count += 1;
    }
    /// Java `addSuccessDisplay`.
    pub fn add_success_display(&mut self) {
        self.process_result_display_state.success_display_count += 1;
    }
    /// Java `getUIComponent`.
    pub fn get_ui_component(&self) -> &Self {
        self
    }
}

/// Java private inner `LabelDivider`.
#[derive(Clone, Debug)]
pub struct LabelDivider {
    pub font_metrics: FontMetrics,
    pub width: i32,
    pub line1: Option<String>,
    pub line2: Option<String>,
}
impl LabelDivider {
    pub fn new(font_metrics: FontMetrics, width: i32) -> Self {
        Self {
            font_metrics,
            width,
            line1: None,
            line2: None,
        }
    }
    pub fn get_line1(&self) -> Option<&str> {
        self.line1.as_deref()
    }
    pub fn get_line2(&self) -> Option<&str> {
        self.line2.as_deref()
    }
    /// Java `divide`.
    pub fn divide(&mut self, label_text: &str) {
        self.line1 = Some(label_text.to_owned());
        self.line2 = None;
        let label_space = self.width - PADDING;
        if self.font_metrics.string_width(label_text) <= 0
            || self.font_metrics.string_width(label_text) < label_space
        {
            return;
        }
        let word_array: Vec<String> = label_text.split_whitespace().map(str::to_owned).collect();
        if word_array.is_empty() {
            return;
        }
        let mut label_line1 =
            LabelLine::new(word_array.clone(), true, label_space, self.font_metrics);
        let mut label_line2 = LabelLine::new(word_array, false, label_space, self.font_metrics);
        while label_line1.iterator_le(&label_line2) {
            let potential_width1 = label_line1.try_next_word();
            let overflow1 = label_line1.is_overflow_on_next_word();
            let potential_width2 = label_line2.try_next_word();
            let overflow2 = label_line2.is_overflow_on_next_word();
            if overflow1 || overflow2 {
                if !overflow1 {
                    label_line1.build();
                } else if !overflow2 {
                    label_line2.build();
                } else if potential_width1 <= potential_width2 {
                    label_line1.build();
                } else {
                    label_line2.build();
                }
                continue;
            }
            if potential_width1 <= potential_width2 {
                label_line1.build();
            } else {
                label_line2.build();
            }
        }
        self.line1 = Some(label_line1.get_line().to_owned());
        self.line2 = Some(label_line2.get_line().to_owned());
    }
}

/// Java private inner `LabelLine`.
#[derive(Clone, Debug)]
pub struct LabelLine {
    pub iterator: WordArrayIterator,
    pub first_half: bool,
    pub label_space: i32,
    pub label: String,
    pub font_metrics: FontMetrics,
}
impl LabelLine {
    pub fn new(
        word_array: Vec<String>,
        first_half: bool,
        label_space: i32,
        font_metrics: FontMetrics,
    ) -> Self {
        let mut iterator = WordArrayIterator::new(word_array, first_half);
        let label = iterator.next();
        Self {
            iterator,
            first_half,
            label_space,
            label,
            font_metrics,
        }
    }
    pub fn iterator_le(&self, label_line: &Self) -> bool {
        self.iterator.le(&label_line.iterator)
    }
    pub fn try_next_word(&self) -> i32 {
        self.font_metrics
            .string_width(&(self.label.clone() + &self.iterator.peek()))
    }
    pub fn is_overflow_on_next_word(&self) -> bool {
        self.try_next_word() > self.label_space
    }
    pub fn build(&mut self) {
        if self.iterator.has_next() {
            let next = self.iterator.next();
            if self.first_half {
                self.label.push_str(&next);
            } else {
                self.label = next + &self.label;
            }
        }
    }
    pub fn get_line(&self) -> &str {
        &self.label
    }
}

/// Java private static inner `WordArrayIterator`.
#[derive(Clone, Debug)]
pub struct WordArrayIterator {
    pub word_array: Vec<String>,
    pub forwards: bool,
    pub space_index: isize,
    pub hyphen_index: isize,
    pub cur_hyphenated_array: Option<Vec<String>>,
}
impl WordArrayIterator {
    pub fn new(word_array: Vec<String>, forwards: bool) -> Self {
        let space_index = if word_array.is_empty() {
            -1
        } else if forwards {
            0
        } else {
            word_array.len() as isize - 1
        };
        Self {
            word_array,
            forwards,
            space_index,
            hyphen_index: -1,
            cur_hyphenated_array: None,
        }
    }
    pub fn has_next(&self) -> bool {
        (self.space_index >= 0 && (self.space_index as usize) < self.word_array.len())
            || self.cur_hyphenated_array.as_ref().is_some_and(|array| {
                self.hyphen_index >= 0 && (self.hyphen_index as usize) < array.len()
            })
    }
    pub fn peek(&self) -> String {
        let mut value = self.clone();
        value.next_internal(false)
    }
    pub fn next(&mut self) -> String {
        self.next_internal(true)
    }
    /// Java overloaded private `next(boolean)`.
    pub fn next_internal(&mut self, increment: bool) -> String {
        if !self.has_next() {
            return String::new();
        }
        if let Some(element) = self.next_hyphenated_element(increment) {
            return element;
        }
        let cur_word = self.word_array[self.space_index as usize].clone();
        if self.cur_hyphenated_array.is_none()
            && !cur_word.is_empty()
            && !cur_word.starts_with('-')
            && cur_word.contains('-')
        {
            let array: Vec<String> = cur_word.split('-').map(str::to_owned).collect();
            if !array.is_empty() {
                self.hyphen_index = if self.forwards {
                    0
                } else {
                    array.len() as isize - 1
                };
                self.cur_hyphenated_array = Some(array);
                if let Some(element) = self.next_hyphenated_element(increment) {
                    return element;
                }
                return String::new();
            }
        }
        let element = self.add_split_char(cur_word);
        if increment {
            self.space_index = self.increment_index(self.space_index);
        }
        element
    }
    pub fn increment_index(&self, index: isize) -> isize {
        if self.forwards { index + 1 } else { index - 1 }
    }
    pub fn next_hyphenated_element(&mut self, increment: bool) -> Option<String> {
        let array = self.cur_hyphenated_array.clone()?;
        if self.hyphen_index < 0 || self.hyphen_index as usize >= array.len() {
            if increment {
                self.cur_hyphenated_array = None;
                self.hyphen_index = -1;
            }
            return None;
        }
        let element = array[self.hyphen_index as usize].clone();
        if element.is_empty() {
            return None;
        }
        let element = self.add_split_char(element);
        if increment {
            self.hyphen_index = self.increment_index(self.hyphen_index);
        }
        Some(element)
    }
    /// Java `addSplitChar`.
    pub fn add_split_char(&self, mut element: String) -> String {
        if element.is_empty() {
            return element;
        }
        if self.forwards {
            if let Some(array) = &self.cur_hyphenated_array {
                if self.hyphen_index == 0 {
                    element = format!(" {element}-");
                } else if self.hyphen_index < array.len() as isize - 1 {
                    element.push('-');
                }
            } else if self.space_index > 0 {
                element.insert(0, ' ');
            }
        } else if let Some(array) = &self.cur_hyphenated_array {
            if self.hyphen_index == array.len() as isize - 1 {
                element.push(' ');
            } else {
                element.push('-');
            }
        } else if self.space_index < self.word_array.len() as isize - 1 {
            element.push(' ');
        }
        element
    }
    /// Java `le`.
    pub fn le(&self, iterator: &Self) -> bool {
        if self.space_index == -1
            || iterator.space_index == -1
            || self.space_index > iterator.space_index
        {
            false
        } else if self.space_index < iterator.space_index {
            true
        } else if self.hyphen_index == -1 {
            true
        } else {
            self.hyphen_index <= iterator.hyphen_index
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn divider_preserves_hyphen_and_balances_lines() {
        let mut button =
            MultiLineButton::new_with_minimum_size(false, Some("word1 word2 wor-d-3 word4"));
        button.button.abstract_button.font_metrics = Some(FontMetrics {
            average_char_width: 5,
            wide_char_width: 5,
            height: 10,
        });
        button.width = 55;
        button.set_text_label(Some("word1 word2 wor-d-3 word4"));
        assert_eq!(button.get_text(), Some("word1 word2 wor-d-3 word4"));
        assert!(button.button.label1.is_some());
        assert!(button.button.label2.is_some());
    }
    #[test]
    fn state_key_and_screen_state_follow_java_order() {
        let mut button = MultiLineButton::get_toggle_button_instance_with_dialog(
            Some("Run process"),
            Some(DialogType::Tools),
        );
        let key = button.get_button_state_key().unwrap();
        let mut screen = BaseScreenState::default();
        screen.set_button_state(Some(&key), true);
        button.set_screen_state(screen);
        assert!(button.is_selected());
        button.set_selected(false);
        assert!(!button.screen_state.unwrap().get_button_state(Some(&key)));
    }

    #[test]
    fn process_result_state_restores_first_launch_but_fails_secondary_launch() {
        let mut button = MultiLineButton::new();
        assert!(!button.is_selected());
        button.msg_process_starting();
        assert!(button.is_selected());
        assert!(button.process_result_display_state.process_running);
        button.msg_process_failed_to_start();
        // Java's `getOriginalState` is `!isSelected()`, so the first launch
        // restores that source-defined state rather than the visual flag.
        assert!(button.is_selected());
        assert!(!button.process_result_display_state.process_running);

        button.msg_process_starting();
        button.msg_secondary_process();
        button.msg_process_failed_to_start();
        assert!(!button.is_selected());
        assert!(!button.process_result_display_state.process_running);

        button.msg_process_starting();
        button.msg_process_succeeded();
        assert!(button.is_selected());
    }

    #[test]
    fn source_result_and_end_state_messages_dispatch_to_terminal_lifecycle() {
        let mut button = MultiLineButton::new();
        button.msg_process_starting();
        button.msg(ProcessResult::FAILED);
        assert!(!button.is_selected());

        button.msg_process_starting();
        button.msg_end_state(ProcessEndState::Paused);
        assert!(button.is_selected());

        button.msg_process_starting();
        button.msg_end_state(ProcessEndState::Cancelled);
        // Source maps CANCELLED to failed-to-start and therefore restores
        // the stored original-state convention.
        assert!(!button.process_result_display_state.process_running);
    }
}
