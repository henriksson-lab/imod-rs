//! `IMOD/Etomo/src/etomo/ui/swing/DirectivePanel.java`.
//!
//! Swing layout, the shared `UIHarness` file chooser, `DirectiveTool`, and
//! directive storage are boundaries in the Java program.  This source unit
//! keeps their state and calls explicit, rather than replacing them with a
//! different native policy.
#![allow(dead_code)]

use std::path::PathBuf;

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::debug_level::DebugLevel;
use crate::imod::etomo::ui::field_type::FieldType;

use super::check_box::CheckBox;
use super::simple_button::SimpleButton;
use super::text_efield::TextEfield;

/// Java `DirectiveValueType` values used by `DirectivePanel`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DirectiveValueType {
    Boolean,
    File,
    String,
    Integer,
    FloatingPoint,
}

impl DirectiveValueType {
    /// Java `FieldType.getInstance(DirectiveValueType)`.
    pub fn field_type(self) -> FieldType {
        match self {
            Self::Boolean | Self::String => FieldType::String,
            Self::File => FieldType::File,
            Self::Integer => FieldType::Integer,
            Self::FloatingPoint => FieldType::FloatingPoint,
        }
    }

    /// Java `getColumns`, delegated through `FieldType` in the source.
    pub fn get_columns(self) -> i32 {
        self.field_type().get_columns()
    }
}

/// Java `DirectiveType`; `DirectivePanel` fetches it in its constructor but
/// does not otherwise inspect it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DirectiveType {
    Setupset,
    Other,
}

/// Java `Directive.Value`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DirectiveValue {
    Boolean(bool),
    String(String),
}

impl DirectiveValue {
    /// Java `Value.toBoolean`.
    pub fn to_boolean(&self) -> bool {
        match self {
            Self::Boolean(value) => *value,
            Self::String(value) => value.eq_ignore_ascii_case("true") || value == "1",
        }
    }

    /// Java `Value.toString`.
    pub fn to_string_value(&self) -> String {
        match self {
            Self::Boolean(value) => value.to_string(),
            Self::String(value) => value.clone(),
        }
    }
}

/// Java `DirectiveValues` as consumed by this panel.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct DirectiveValues {
    pub value: Option<DirectiveValue>,
    pub default_value: Option<DirectiveValue>,
}

/// Java `DirectiveDescrChoiceList`.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct DirectiveDescrChoiceList {
    pub descriptions: Vec<String>,
    pub values: Vec<String>,
}

impl DirectiveDescrChoiceList {
    /// Java `size`.
    pub fn size(&self) -> usize {
        self.descriptions.len()
    }
    /// Java `getDescr`.
    pub fn get_descr(&self, index: usize) -> &str {
        &self.descriptions[index]
    }
    /// Java `getValue`.
    pub fn get_value(&self, index: usize) -> &str {
        &self.values[index]
    }
}

/// Java `storage.Directive` fields and accessors reached from this source unit.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Directive {
    pub directive_type: DirectiveType,
    pub title: String,
    pub value_type: DirectiveValueType,
    pub choice_list: Option<DirectiveDescrChoiceList>,
    pub values: DirectiveValues,
    pub include: bool,
    pub copy_arg: bool,
    pub key_description: String,
    pub description: String,
    pub batch: bool,
    pub template: bool,
    pub etomo_column: String,
    pub in_directive_file_debug_string: String,
}

impl Directive {
    /// Java `isChoiceList`.
    pub fn is_choice_list(&self) -> bool {
        self.choice_list.is_some()
    }
    /// Java `setInclude`.
    pub fn set_include(&mut self, include: bool) {
        self.include = include;
    }
    /// Java overloaded `setValue(boolean)`.
    pub fn set_value_boolean(&mut self, value: bool) {
        self.values.value = Some(DirectiveValue::Boolean(value));
    }
    /// Java overloaded `setValue(String)`, where `None` is Java null.
    pub fn set_value_string(&mut self, value: Option<String>) {
        self.values.value = value.map(DirectiveValue::String);
    }
}

/// Java `DirectiveTool` boundary.  Its source unit owns visibility and toggle
/// precedence; this panel merely passes its source arguments through.
pub trait DirectiveTool {
    fn set_debug(&mut self, debug: DebugLevel);
    fn reset_debug(&mut self);
    fn is_directive_visible(
        &mut self,
        directive: &Directive,
        included: bool,
        different_from_checkpoint: bool,
    ) -> bool;
    fn is_toggle_directive_included(&mut self, directive: &Directive, included: bool) -> bool;
}

/// Swing `JComboBox` state operated by this source unit.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComboBox {
    pub title: String,
    pub items: Vec<String>,
    pub selected_index: isize,
    pub enabled: bool,
    pub tooltip: Option<String>,
    checkpoint: isize,
}

impl ComboBox {
    /// Java `getUnlabeledInstance` / `getUnlabeledEmptyChoiceInstance`.
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            items: Vec::new(),
            selected_index: -1,
            enabled: true,
            tooltip: None,
            checkpoint: -1,
        }
    }
    /// Java `addItem`.
    pub fn add_item(&mut self, item: impl Into<String>) {
        self.items.push(item.into());
    }
    /// Java `setSelectedIndex`.
    pub fn set_selected_index(&mut self, index: isize) {
        self.selected_index = index;
    }
    /// Java `getSelectedIndex`.
    pub fn get_selected_index(&self) -> isize {
        self.selected_index
    }
    /// Java `setEnabled`.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
    /// Java `checkpoint`.
    pub fn checkpoint(&mut self) {
        self.checkpoint = self.selected_index;
    }
    /// Java `isDifferentFromCheckpoint`.
    pub fn is_different_from_checkpoint(&self, _always_check: bool) -> bool {
        self.selected_index != self.checkpoint
    }
    /// Java `setToolTipText`.
    pub fn set_tool_tip_text(&mut self, text: Option<&str>) {
        self.tooltip = text.map(str::to_owned);
    }
}

/// Native state corresponding to the root `JPanel` and its X-axis `BoxLayout`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JPanelBoundary {
    pub visible: bool,
    pub x_axis_layout: bool,
    pub children: Vec<String>,
}

impl Default for JPanelBoundary {
    fn default() -> Self {
        Self {
            visible: true,
            x_axis_layout: false,
            children: Vec::new(),
        }
    }
}

/// Java `JFileChooser` state and native `showOpenDialog` boundary.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct FileChooserBoundary {
    pub current_directory: Option<PathBuf>,
    pub dialog_title: Option<String>,
    pub selected_file: Option<PathBuf>,
    pub approve_option: bool,
}

/// Java `FieldValidator.equals(FieldType, String, String)`.
pub struct FieldValidator;
impl FieldValidator {
    pub fn equals(_field_type: FieldType, first: &str, second: &str) -> bool {
        first.trim() == second.trim()
    }
}

/// Java final `DirectivePanel`.
pub struct DirectivePanel {
    pub pnl_root: JPanelBoundary,
    pub cb_include: CheckBox,
    pub cb_value: Option<ComboBox>,
    pub tf_value: Option<TextEfield>,
    pub sb_file_value: Option<SimpleButton>,
    pub tool: Box<dyn DirectiveTool>,
    pub directive: Directive,
    pub field_type: FieldType,
    pub action_command: String,
    pub source_axis_type: AxisType,
    pub value_list: Option<Vec<String>>,
    pub boolean_value_type: bool,
    pub debug: DebugLevel,
    pub last_file_chooser_location: Option<PathBuf>,
    /// `UIHarness.INSTANCE.getFileChooser()` boundary and its configured result.
    pub file_chooser: Option<FileChooserBoundary>,
    /// `EtomoDirector.INSTANCE.getIMODCalibDirectory()` boundary.
    pub imod_calib_directory: Option<PathBuf>,
    /// `EtomoDirector.INSTANCE.getHomeDirectory()` boundary.
    pub home_directory: Option<PathBuf>,
}

impl DirectivePanel {
    const NO_INDEX: isize = 1;
    const YES_INDEX: isize = 0;

    /// Java private `DirectivePanel(BaseManager, Directive, DirectiveTool, AxisType)`.
    pub fn new(
        _manager: Option<&'static dyn BaseManager>,
        directive: Directive,
        tool: Box<dyn DirectiveTool>,
        source_axis_type: AxisType,
    ) -> Self {
        let field_type = directive.value_type.field_type();
        let title = directive.title.clone();
        let mut cb_include = CheckBox::new_with_text(&format!(" -  {title}: "));
        let action_command = "include".to_owned();
        cb_include.set_action_command(Some(&action_command));
        let boolean_value_type = directive.value_type == DirectiveValueType::Boolean;
        let (cb_value, tf_value, sb_file_value, value_list) =
            if boolean_value_type || directive.is_choice_list() {
                if boolean_value_type {
                    let mut cb_value = ComboBox::new(title);
                    cb_value.add_item("Yes");
                    cb_value.add_item("No");
                    (Some(cb_value), None, None, None)
                } else {
                    let mut cb_value = ComboBox::new(title);
                    let choice_list = directive
                        .choice_list
                        .as_ref()
                        .expect("Directive.choiceList is null");
                    let mut value_list = Vec::with_capacity(choice_list.size());
                    for index in 0..choice_list.size() {
                        cb_value.add_item(choice_list.get_descr(index));
                        value_list.push(choice_list.get_value(index).to_owned());
                    }
                    (Some(cb_value), None, None, Some(value_list))
                }
            } else {
                let tf_value = TextEfield::get_instance(title, field_type);
                let sb_file_value = (directive.value_type == DirectiveValueType::File)
                    .then(|| SimpleButton::new_with_scaled_image(Some("OPEN_FILE")));
                (None, Some(tf_value), sb_file_value, None)
            };
        Self {
            pnl_root: JPanelBoundary::default(),
            cb_include,
            cb_value,
            tf_value,
            sb_file_value,
            tool,
            directive,
            field_type,
            action_command,
            source_axis_type,
            value_list,
            boolean_value_type,
            debug: DebugLevel::Off,
            last_file_chooser_location: None,
            file_chooser: None,
            imod_calib_directory: None,
            home_directory: None,
        }
    }

    /// Java static `getInstance`.
    pub fn get_instance(
        manager: Option<&'static dyn BaseManager>,
        directive: Directive,
        tool: Box<dyn DirectiveTool>,
        source_axis_type: AxisType,
    ) -> Self {
        let mut instance = Self::new(manager, directive, tool, source_axis_type);
        instance.create_panel();
        instance.set_tooltips();
        instance.add_listeners();
        instance
    }

    /// Java private `createPanel(Directive)`.
    pub fn create_panel(&mut self) {
        if let Some(button) = &mut self.sb_file_value {
            let size = (24, 24); // Java UIUtilities.getScaledFolderButtonDimension boundary.
            button.button.abstract_button.preferred_size = Some(super::panel::Dimension {
                width: size.0,
                height: size.1,
            });
            button.button.abstract_button.maximum_size = Some(super::panel::Dimension {
                width: size.0,
                height: size.1,
            });
        }
        self.pnl_root.x_axis_layout = true;
        self.pnl_root.children.push("cbInclude".into());
        if self.cb_value.is_some() {
            self.pnl_root.children.push("cbValue".into());
        } else {
            self.pnl_root.children.push("tfValue".into());
            if self.sb_file_value.is_some() {
                self.pnl_root.children.push("sbFileValue".into());
            }
        }
        self.pnl_root.children.push("horizontalGlue".into());
        self.init();
    }

    /// Java `getComponent`.
    pub fn get_component(&self) -> &JPanelBoundary {
        &self.pnl_root
    }

    /// Java private `addListeners`.
    pub fn add_listeners(&mut self) {
        self.cb_include.add_action_listener();
        if let Some(button) = &mut self.sb_file_value {
            button.button.action_listener_count += 1;
        }
    }

    /// Java `getState`.
    pub fn get_state(&mut self) -> &Directive {
        self.set_state_in_directive();
        &self.directive
    }

    /// Java `setStateInDirective`.
    pub fn set_state_in_directive(&mut self) {
        self.directive.set_include(self.is_include());
        if let Some(cb_value) = &self.cb_value {
            if self.boolean_value_type {
                self.directive
                    .set_value_boolean(cb_value.get_selected_index() == Self::YES_INDEX);
            } else {
                let index = cb_value.get_selected_index();
                self.directive.set_value_string(
                    (index >= 0).then(|| self.value_list.as_ref().unwrap()[index as usize].clone()),
                );
            }
        } else {
            self.directive
                .set_value_string(self.tf_value.as_ref().map(TextEfield::get_text));
        }
    }

    /// Java `init`.
    pub fn init(&mut self) {
        if let Some(tf_value) = &mut self.tf_value {
            tf_value.text_field.columns = self.directive.value_type.get_columns();
        }
        self.set_included();
        self.init_value();
        self.tool.set_debug(self.debug);
        if !self
            .tool
            .is_directive_visible(&self.directive, self.cb_include.is_selected(), false)
        {
            self.pnl_root.visible = false;
        }
        self.tool.reset_debug();
        self.checkpoint();
    }

    /// Java private `action`.
    pub fn action(&mut self) {
        self.enable_value();
    }

    /// Java private `fileValueAction`.
    pub fn file_value_action(&mut self) {
        let Some(chooser) = &mut self.file_chooser else {
            return;
        };
        if !self
            .tf_value
            .as_ref()
            .expect("DirectivePanel.tfValue is null")
            .is_empty()
        {
            chooser.current_directory =
                Some(PathBuf::from(self.tf_value.as_ref().unwrap().get_text()));
        } else if self.last_file_chooser_location.is_some() {
            chooser.current_directory = self.last_file_chooser_location.clone();
        } else if self.directive.copy_arg {
            chooser.current_directory = self.imod_calib_directory.clone();
        } else if self.home_directory.is_some() {
            chooser.current_directory = self.home_directory.clone();
        }
        chooser.dialog_title = Some(format!("Open {}", self.directive.title));
        if chooser.approve_option {
            if let Some(file) = chooser.selected_file.clone() {
                self.tf_value.as_mut().unwrap().set_text_file(&file);
            }
        }
        self.last_file_chooser_location = chooser.current_directory.clone();
    }

    /// Java `msgControlChanged`.
    pub fn msg_control_changed(&mut self, include_change: bool, _expand_change: bool) -> bool {
        if include_change {
            self.set_included();
        }
        self.tool.set_debug(self.debug);
        let visible = self.tool.is_directive_visible(
            &self.directive,
            self.cb_include.is_selected(),
            self.is_different_from_checkpoint(false),
        );
        self.pnl_root.visible = visible;
        visible
    }

    /// Java private `copy`.
    pub fn copy(&mut self, input: Option<&DirectivePanel>) {
        self.cb_include
            .set_selected(input.is_some_and(|input| input.cb_include.is_selected()));
        self.enable_value();
        self.copy_value(input);
    }

    /// Java private `copyValue`.
    pub fn copy_value(&mut self, input: Option<&DirectivePanel>) {
        match (self.cb_value.as_mut(), self.tf_value.as_mut(), input) {
            (Some(cb), _, None) if self.boolean_value_type => cb.set_selected_index(Self::NO_INDEX),
            (Some(cb), _, None) => cb.set_selected_index(-1),
            (None, Some(tf), None) => tf.set_text(""),
            (Some(cb), _, Some(input)) => cb.set_selected_index(
                input
                    .cb_value
                    .as_ref()
                    .expect("DirectivePanel.cbValue is null")
                    .get_selected_index(),
            ),
            (None, Some(tf), Some(input)) => tf.set_text(
                input
                    .tf_value
                    .as_ref()
                    .expect("DirectivePanel.tfValue is null")
                    .get_text(),
            ),
            _ => unreachable!(),
        }
    }

    /// Java `checkpoint`.
    pub fn checkpoint(&mut self) {
        self.cb_include.checkpoint();
        if let Some(cb_value) = &mut self.cb_value {
            cb_value.checkpoint();
        } else if let Some(tf_value) = &mut self.tf_value {
            tf_value.checkpoint();
        }
    }

    /// Java private `enableValue`.
    pub fn enable_value(&mut self) {
        let enable = self.cb_include.is_selected() && self.cb_include.is_enabled();
        if let Some(cb_value) = &mut self.cb_value {
            cb_value.set_enabled(enable);
        } else {
            self.tf_value
                .as_mut()
                .expect("DirectivePanel.tfValue is null")
                .set_enabled(enable);
            if let Some(button) = &mut self.sb_file_value {
                button.button.enabled = enable;
            }
        }
    }

    /// Java `equals(DirectivePanel)`.
    pub fn equals(&self, input: Option<&DirectivePanel>) -> bool {
        input.is_some_and(|input| {
            self.cb_include.is_selected() == input.cb_include.is_selected()
                && self.equals_value(Some(input))
        })
    }

    /// Java private `equalsValue`.
    pub fn equals_value(&self, input: Option<&DirectivePanel>) -> bool {
        let Some(input) = input else {
            return false;
        };
        if let Some(cb_value) = &self.cb_value {
            return cb_value.get_selected_index()
                == input
                    .cb_value
                    .as_ref()
                    .expect("DirectivePanel.cbValue is null")
                    .get_selected_index();
        }
        FieldValidator::equals(
            self.field_type,
            &self.tf_value.as_ref().unwrap().get_text(),
            &input
                .tf_value
                .as_ref()
                .expect("DirectivePanel.tfValue is null")
                .get_text(),
        )
    }

    /// Java private `initValue(Directive)`.
    pub fn init_value(&mut self) {
        let Some(value) = self.directive.values.value.clone() else {
            return;
        };
        if let Some(cb_value) = &mut self.cb_value {
            if self.boolean_value_type {
                cb_value.set_selected_index(if value.to_boolean() {
                    Self::YES_INDEX
                } else {
                    Self::NO_INDEX
                });
            } else {
                let value = value.to_string_value();
                let index = (!value.trim().is_empty())
                    .then(|| {
                        self.value_list
                            .as_ref()
                            .unwrap()
                            .iter()
                            .position(|candidate| candidate == &value)
                            .map(|index| index as isize)
                    })
                    .flatten()
                    .unwrap_or(-1);
                cb_value.set_selected_index(index);
            }
        } else {
            self.tf_value
                .as_mut()
                .unwrap()
                .set_text(value.to_string_value());
        }
    }

    /// Java `isDifferentFromCheckpoint`.
    pub fn is_different_from_checkpoint(&self, check_include: bool) -> bool {
        (check_include && self.cb_include.is_different_from_checkpoint(true))
            || self.cb_value.as_ref().map_or_else(
                || {
                    self.tf_value
                        .as_ref()
                        .unwrap()
                        .is_different_from_checkpoint(true)
                },
                |cb| cb.is_different_from_checkpoint(true),
            )
    }
    /// Java `isEnabled`.
    pub fn is_enabled(&self) -> bool {
        self.cb_include.is_enabled()
    }
    /// Java `isInclude`.
    pub fn is_include(&self) -> bool {
        self.cb_include.is_selected() && self.cb_include.is_enabled()
    }
    /// Java `isVisible`.
    pub fn is_visible(&self) -> bool {
        self.pnl_root.visible
    }

    /// Java private `resetValue`.
    pub fn reset_value(&mut self) {
        if let Some(cb_value) = &mut self.cb_value {
            cb_value.set_selected_index(if self.boolean_value_type {
                Self::NO_INDEX
            } else {
                -1
            });
        } else {
            self.tf_value.as_mut().unwrap().set_text("");
        }
    }
    /// Java private `setEnabled`.
    pub fn set_enabled(&mut self, enable: bool) {
        self.cb_include.set_enabled(enable);
        self.enable_value();
    }

    /// Java private `setIncluded`.
    pub fn set_included(&mut self) {
        let included = self.cb_include.is_selected();
        self.tool.set_debug(self.debug);
        if self
            .tool
            .is_toggle_directive_included(&self.directive, included)
        {
            self.cb_include.set_selected(!included);
        }
        self.tool.reset_debug();
        self.enable_value();
    }
    /// Java `setVisible`.
    pub fn set_visible(&mut self, input: bool) {
        self.pnl_root.visible = input;
    }

    /// Java `toString`.
    pub fn to_string_value(&self) -> &str {
        &self.directive.title
    }

    /// Java private `setTooltips`.
    pub fn set_tooltips(&mut self) {
        let value_string = self
            .directive
            .values
            .value
            .as_ref()
            .map(DirectiveValue::to_string_value);
        let default_value_string = self
            .directive
            .values
            .default_value
            .as_ref()
            .map(DirectiveValue::to_string_value);
        let debug_string = if self.debug.is_extra_verbose() {
            format!(
                "  Type:{:?}, Batch:{}, Tmplt:{}, Etomo:{}, AxisLevelData:{}",
                self.directive.value_type,
                self.directive.batch,
                self.directive.template,
                self.directive.etomo_column,
                self.directive.in_directive_file_debug_string
            )
        } else {
            String::new()
        };
        let tooltip = format!(
            "{}:  {}.{}{}{}",
            self.directive.key_description,
            self.directive.description,
            value_string.map_or_else(String::new, |value| format!("  Dataset value:{value}")),
            default_value_string
                .map_or_else(String::new, |value| format!("  Original value:{value}")),
            debug_string
        );
        self.cb_include.set_tool_tip_text(Some(&tooltip));
        if let Some(cb_value) = &mut self.cb_value {
            cb_value.set_tool_tip_text(Some(&tooltip));
        } else {
            self.tf_value.as_mut().unwrap().set_tooltip(tooltip.clone());
            if let Some(button) = &mut self.sb_file_value {
                button.button.tooltip = Some(tooltip);
            }
        }
    }

    /// Java nested `DirectiveListener.actionPerformed`.
    pub fn directive_listener_action_performed(&mut self) {
        self.action();
    }
    /// Java nested `DirectiveFileValueListener.actionPerformed`.
    pub fn directive_file_value_listener_action_performed(&mut self) {
        self.file_value_action();
    }
}

/// Native callback adapter for Java `DirectiveListener`.
pub struct DirectiveListener;

impl DirectiveListener {
    #[allow(non_snake_case)]
    pub fn actionPerformed(panel: &mut DirectivePanel) {
        panel.action();
    }
}

/// Native callback adapter for Java `DirectiveFileValueListener`.
pub struct DirectiveFileValueListener;

impl DirectiveFileValueListener {
    #[allow(non_snake_case)]
    pub fn actionPerformed(panel: &mut DirectivePanel) {
        panel.file_value_action();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Tool {
        visible: bool,
        toggle: bool,
        debug: Vec<DebugLevel>,
    }
    impl DirectiveTool for Tool {
        fn set_debug(&mut self, debug: DebugLevel) {
            self.debug.push(debug);
        }
        fn reset_debug(&mut self) {}
        fn is_directive_visible(&mut self, _: &Directive, _: bool, _: bool) -> bool {
            self.visible
        }
        fn is_toggle_directive_included(&mut self, _: &Directive, _: bool) -> bool {
            self.toggle
        }
    }
    fn directive(value_type: DirectiveValueType) -> Directive {
        Directive {
            directive_type: DirectiveType::Setupset,
            title: "Test".into(),
            value_type,
            choice_list: None,
            values: DirectiveValues::default(),
            include: false,
            copy_arg: false,
            key_description: "key".into(),
            description: "description".into(),
            batch: false,
            template: false,
            etomo_column: "column".into(),
            in_directive_file_debug_string: "axis".into(),
        }
    }
    #[test]
    fn boolean_state_checkpoint_and_tool_visibility_follow_source() {
        let mut panel = DirectivePanel::get_instance(
            None,
            directive(DirectiveValueType::Boolean),
            Box::new(Tool {
                visible: true,
                toggle: false,
                debug: vec![],
            }),
            AxisType::NotSet,
        );
        panel.cb_include.set_selected(true);
        panel
            .cb_value
            .as_mut()
            .unwrap()
            .set_selected_index(DirectivePanel::YES_INDEX);
        assert!(panel.is_different_from_checkpoint(true));
        assert_eq!(
            panel.get_state().values.value,
            Some(DirectiveValue::Boolean(true))
        );
        panel.checkpoint();
        assert!(!panel.is_different_from_checkpoint(true));
        assert!(panel.msg_control_changed(false, true));
    }
    #[test]
    fn choice_and_file_values_preserve_java_empty_and_chooser_paths() {
        let mut chosen = directive(DirectiveValueType::String);
        chosen.choice_list = Some(DirectiveDescrChoiceList {
            descriptions: vec!["one".into()],
            values: vec!["1".into()],
        });
        let mut choices = DirectivePanel::get_instance(
            None,
            chosen,
            Box::new(Tool {
                visible: true,
                toggle: false,
                debug: vec![],
            }),
            AxisType::NotSet,
        );
        choices.cb_value.as_mut().unwrap().set_selected_index(-1);
        assert_eq!(choices.get_state().values.value, None);
        let mut file = DirectivePanel::get_instance(
            None,
            directive(DirectiveValueType::File),
            Box::new(Tool {
                visible: true,
                toggle: false,
                debug: vec![],
            }),
            AxisType::NotSet,
        );
        file.file_chooser = Some(FileChooserBoundary {
            current_directory: None,
            dialog_title: None,
            selected_file: Some(PathBuf::from("/tmp/input.mrc")),
            approve_option: true,
        });
        file.directive_file_value_listener_action_performed();
        assert_eq!(file.tf_value.unwrap().get_text(), "/tmp/input.mrc");
    }
    #[test]
    fn copy_enable_reset_and_equality_follow_source_value_type() {
        let mut first = DirectivePanel::get_instance(
            None,
            directive(DirectiveValueType::Integer),
            Box::new(Tool {
                visible: true,
                toggle: false,
                debug: vec![],
            }),
            AxisType::NotSet,
        );
        let mut second = DirectivePanel::get_instance(
            None,
            directive(DirectiveValueType::Integer),
            Box::new(Tool {
                visible: true,
                toggle: false,
                debug: vec![],
            }),
            AxisType::NotSet,
        );
        first.cb_include.set_selected(true);
        first.enable_value();
        first.tf_value.as_mut().unwrap().set_text(" 7 ");
        second.copy(Some(&first));
        assert!(second.equals(Some(&first)));
        second.reset_value();
        assert!(!second.equals_value(Some(&first)));
        second.set_enabled(false);
        assert!(!second.is_include());
    }
}
