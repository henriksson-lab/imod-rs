//! `IMOD/Etomo/src/etomo/ui/swing/DirectivePanel.java`.
//!
//! One directive in the directive editor: an include check box (" -  <title>: ") and
//! the value field matching the directive's value type (a Yes/No or choice combo box,
//! or a text field, with a file chooser button for file values).  An event dispatch
//! thread object, created as `Rc<Self>` by [`DirectivePanel::get_instance`].

use std::cell::{Cell, RefCell};
use std::path::{Path, PathBuf};
use std::rc::{Rc, Weak};
use std::sync::Arc;

use super::check_box::CheckBox;
use super::combo_box::ComboBox;
use super::file_chooser;
use super::scaled_image;
use super::simple_button::SimpleButton;
use super::text_field::TextField;
use super::tooltip_formatter;
use super::ui_harness;
use super::ui_utilities;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::jdk::{ActionEvent, JComponent};
use crate::imod::etomo::logic::directive_tool::DirectiveTool;
use crate::imod::etomo::logic::field_validator::FieldValidator;
use crate::imod::etomo::storage::directive::Directive;
use crate::imod::etomo::storage::directive_value_type::DirectiveValueType;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::debug_level::DebugLevel;
use crate::imod::etomo::ui::field::Field;
use crate::imod::etomo::ui::field_type::FieldType;
use crate::imod::etomo::util::utilities;

/// Java private static final `NO_INDEX`.
const NO_INDEX: i32 = 1;
/// Java private static final `YES_INDEX`.
const YES_INDEX: i32 = 0;

/// Java package-private `final class DirectivePanel`.
pub struct DirectivePanel {
    /// Java private final `pnlRoot`.
    pnl_root: Rc<JComponent>,
    /// Java private final `cbInclude`.
    cb_include: Rc<CheckBox>,
    /// Java private final `cbValue` (null unless the value is boolean or a choice).
    cb_value: Option<Rc<ComboBox>>,
    /// Java private final `tfValue` (null when `cbValue` is used).
    tf_value: Option<Rc<TextField>>,
    /// Java private final `sbFileValue` (only for file values).
    sb_file_value: Option<Rc<SimpleButton>>,
    /// Java private final `tool`.
    tool: Rc<DirectiveTool>,
    /// Java private final `directive`.
    directive: Arc<Directive>,
    /// Java private final `fieldType`.
    field_type: Option<FieldType>,
    /// Java private final `actionCommand`.
    action_command: String,
    /// Java private final `sourceAxisType`.  Never read in the source.
    source_axis_type: AxisType,
    /// Java private final `valueList`.
    value_list: Option<Vec<String>>,
    /// Java private final `booleanValueType`.
    boolean_value_type: bool,
    /// Java private `debug`, initialised from the arguments' debug level.
    debug: Cell<DebugLevel>,
    /// Java private `lastFileChooserLocation`, initialised to null.
    last_file_chooser_location: RefCell<Option<PathBuf>>,
    /// Java `this`.
    self_ref: Weak<DirectivePanel>,
}

impl DirectivePanel {
    /// Java private `DirectivePanel(BaseManager, Directive, DirectiveTool, AxisType)`.
    fn new(
        _manager: &'static dyn BaseManager,
        directive: Arc<Directive>,
        tool: Rc<DirectiveTool>,
        source_axis_type: AxisType,
    ) -> Rc<DirectivePanel> {
        let field_type = FieldType::get_instance(directive.get_value_type());
        // Initialize the value field that matches the value type. Set the other one to
        // null.
        let title = directive.get_title();
        let cb_include = CheckBox::new_string(Some(&format!(
            " -  {}: ",
            title.as_deref().unwrap_or("null")
        )));
        // Need to be able to distinguish the include checkbox action commands from the
        // three directives in each directive set.
        let action_command = "include".to_string();
        cb_include.set_action_command(Some(&action_command));
        let value_type = directive.get_value_type();
        let local_field_type = FieldType::get_instance(value_type);
        let boolean_value_type = value_type == Some(DirectiveValueType::Boolean);
        let cb_value;
        let tf_value;
        let sb_file_value;
        let value_list;
        if boolean_value_type || directive.is_choice_list() {
            if boolean_value_type {
                let combo_box = ComboBox::get_unlabeled_instance(title.as_deref());
                combo_box.add_item(Some("Yes"));
                combo_box.add_item(Some("No"));
                cb_value = Some(combo_box);
                value_list = None;
            } else {
                let combo_box = ComboBox::get_unlabeled_empty_choice_instance(title.as_deref());
                let choice_list = directive.get_choice_list().unwrap();
                let size = choice_list.size();
                let mut list = Vec::with_capacity(size as usize);
                for i in 0..size {
                    combo_box.add_item(choice_list.get_descr(i).as_deref());
                    list.push(choice_list.get_value(i).unwrap_or_default());
                }
                cb_value = Some(combo_box);
                value_list = Some(list);
            }
            tf_value = None;
            sb_file_value = None;
        } else {
            // A non-boolean directive value type always has a field type.
            tf_value = Some(TextField::new(
                local_field_type.expect("FieldType for a non-boolean directive"),
                title.as_deref(),
                None,
            ));
            cb_value = None;
            value_list = None;
            if value_type == Some(DirectiveValueType::File) {
                sb_file_value = Some(SimpleButton::new_scaled_image(Some(
                    if !*utilities::APRIL_FOOLS {
                        &scaled_image::OPEN_FILE
                    } else {
                        &scaled_image::OPEN_FILE_FOOL
                    },
                )));
            } else {
                sb_file_value = None;
            }
        }
        Rc::new_cyclic(|self_ref| DirectivePanel {
            pnl_root: JComponent::new_panel(),
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
            debug: Cell::new(etomo_director::ARGUMENTS.lock().unwrap().get_debug_level()),
            last_file_chooser_location: RefCell::new(None),
            self_ref: self_ref.clone(),
        })
    }

    /// Java package-private static `getInstance(BaseManager, Directive, DirectiveTool,
    /// AxisType)`.
    pub fn get_instance(
        manager: &'static dyn BaseManager,
        directive: Arc<Directive>,
        tool: Rc<DirectiveTool>,
        source_axis_type: AxisType,
    ) -> Rc<DirectivePanel> {
        let instance = DirectivePanel::new(manager, directive, tool, source_axis_type);
        instance.create_panel();
        instance.set_tooltips();
        instance.add_listeners();
        instance
    }

    /// Java private `createPanel(Directive)`.
    fn create_panel(&self) {
        // init
        if self.sb_file_value.is_some() {
            // Swing layout: sbFileValue.setPreferredSize/setMaximumSize(
            // UIUtilities.getScaledFolderButtonDimension()).
            let _size = ui_utilities::get_scaled_folder_button_dimension();
        }
        // root panel (BoxLayout X_AXIS)
        self.pnl_root.add(&self.cb_include.get_component());
        if let Some(cb_value) = &self.cb_value {
            self.pnl_root.add(&cb_value.get_component());
        } else {
            self.pnl_root
                .add(&self.tf_value.as_ref().unwrap().get_component());
            if let Some(sb_file_value) = &self.sb_file_value {
                self.pnl_root.add(&sb_file_value.get_component());
            }
        }
        // Swing layout: pnlRoot.add(Box.createHorizontalGlue()).
        self.init();
    }

    /// Java package-private `getComponent()`.
    pub fn get_component(&self) -> Rc<JComponent> {
        self.pnl_root.clone()
    }

    /// Java private `addListeners()`.
    fn add_listeners(&self) {
        let adaptee = self.self_ref.clone();
        self.cb_include
            .add_action_listener(Some(Rc::new(move |_event: &ActionEvent| {
                if let Some(adaptee) = adaptee.upgrade() {
                    adaptee.action();
                }
            })));
        if let Some(sb_file_value) = &self.sb_file_value {
            let adaptee = self.self_ref.clone();
            sb_file_value.get_component().add_action_listener(Rc::new(
                move |_event: &ActionEvent| {
                    if let Some(adaptee) = adaptee.upgrade() {
                        adaptee.file_value_action();
                    }
                },
            ));
        }
    }

    /// Java package-private `getState()`.
    pub fn get_state(&self) -> Arc<Directive> {
        self.set_state_in_directive();
        self.directive.clone()
    }

    /// Java package-private `setStateInDirective()`.
    pub fn set_state_in_directive(&self) {
        self.directive.set_include(self.is_include());
        if let Some(cb_value) = &self.cb_value {
            if self.boolean_value_type {
                self.directive
                    .set_value_boolean(cb_value.get_selected_index() == YES_INDEX);
            } else {
                let index = cb_value.get_selected_index();
                if index >= 0 {
                    self.directive
                        .set_value_string(Some(&self.value_list.as_ref().unwrap()[index as usize]));
                } else {
                    // Handle empty pulldown choice
                    self.directive.set_value_string(None);
                }
            }
        } else {
            self.directive
                .set_value_string(self.tf_value.as_ref().unwrap().get_text_void().as_deref());
        }
    }

    /// Java package-private `init()`.  Initialize directive - set include, set and
    /// checkpoint the value, and set root panel visibility.
    pub fn init(&self) {
        if let Some(tf_value) = &self.tf_value {
            tf_value.set_columns_int(
                self.directive
                    .get_value_type()
                    .map(|value_type| value_type.get_columns())
                    .unwrap_or_default(),
            );
        }
        self.set_included();
        self.init_value();
        self.tool.set_debug(self.debug.get());
        if !self
            .tool
            .is_directive_visible(&self.directive, self.cb_include.is_selected(), false)
        {
            self.pnl_root.set_visible(false);
        }
        self.tool.reset_debug();
        self.checkpoint();
    }

    /// Java private `action()`.
    fn action(&self) {
        self.enable_value();
    }

    /// Java private `fileValueAction()`.
    fn file_value_action(&self) {
        let Some(chooser) = ui_harness::with(|harness| harness.get_file_chooser()) else {
            return;
        };
        let tf_value = self.tf_value.as_ref().unwrap();
        if !tf_value.is_empty() {
            chooser.set_current_directory(Some(Path::new(
                &tf_value.get_text_void().unwrap_or_default(),
            )));
        } else if let Some(location) = self.last_file_chooser_location.borrow().clone() {
            chooser.set_current_directory(Some(&location));
        } else if self.directive.is_copy_arg() {
            chooser.set_current_directory(
                etomo_director::INSTANCE
                    .get_imod_calib_directory()
                    .as_deref(),
            );
        } else {
            let home = etomo_director::INSTANCE.get_home_directory().to_string();
            chooser.set_current_directory(Some(Path::new(&home)));
        }
        chooser.set_dialog_title(Some(&format!(
            "Open {}",
            self.directive.get_title().as_deref().unwrap_or("null")
        )));
        if chooser.show_open_dialog(Some(&self.pnl_root)) == file_chooser::APPROVE_OPTION
            && let Some(file) = chooser.get_selected_file()
        {
            tf_value.set_text_file(&file);
        }
        *self.last_file_chooser_location.borrow_mut() = chooser.get_current_directory();
    }

    /// Java package-private `msgControlChanged(boolean, boolean)`.  Returns true if the
    /// directive is visible.
    pub fn msg_control_changed(&self, include_change: bool, _expand_change: bool) -> bool {
        if include_change {
            self.set_included();
        }
        self.tool.set_debug(self.debug.get());
        let visible = self.tool.is_directive_visible(
            &self.directive,
            self.cb_include.is_selected(),
            self.is_different_from_checkpoint(false),
        );
        self.pnl_root.set_visible(visible);
        visible
    }

    /// Java private `copy(DirectivePanel)`.  Unused in the source.
    fn copy(&self, input: Option<&DirectivePanel>) {
        match input {
            None => self.cb_include.set_selected_boolean(false),
            Some(input) => self
                .cb_include
                .set_selected_boolean(input.cb_include.is_selected()),
        }
        self.enable_value();
        self.copy_value(input);
    }

    /// Java private `copyValue(DirectivePanel)`.
    fn copy_value(&self, input: Option<&DirectivePanel>) {
        match input {
            None => {
                if let Some(cb_value) = &self.cb_value {
                    if self.boolean_value_type {
                        cb_value.set_selected_index(NO_INDEX);
                    } else {
                        cb_value.set_selected_index(-1);
                    }
                } else {
                    self.tf_value.as_ref().unwrap().set_text_string(Some(""));
                }
            }
            Some(input) => {
                if let Some(cb_value) = &self.cb_value {
                    cb_value
                        .set_selected_index(input.cb_value.as_ref().unwrap().get_selected_index());
                } else {
                    self.tf_value.as_ref().unwrap().set_text_string(
                        input.tf_value.as_ref().unwrap().get_text_void().as_deref(),
                    );
                }
            }
        }
    }

    /// Java package-private `checkpoint()`.
    pub fn checkpoint(&self) {
        // Checkpoint include to use when closing.
        self.cb_include.checkpoint_void();
        // Checkpoint value to use when closing and to for deciding whether the directive
        // should be visible.
        if let Some(cb_value) = &self.cb_value {
            cb_value.checkpoint();
        } else if let Some(tf_value) = &self.tf_value {
            tf_value.checkpoint_void();
        }
    }

    /// Java private `enableValue()`.
    fn enable_value(&self) {
        let enable = self.cb_include.is_selected() && self.cb_include.is_enabled();
        if let Some(cb_value) = &self.cb_value {
            cb_value.set_enabled(enable);
        } else {
            self.tf_value.as_ref().unwrap().set_enabled(enable);
            if let Some(sb_file_value) = &self.sb_file_value {
                sb_file_value.get_component().set_enabled(enable);
            }
        }
    }

    /// Java `equals(DirectivePanel)`.  Unused in the source.
    pub fn equals(&self, input: Option<&DirectivePanel>) -> bool {
        let Some(input) = input else {
            return false;
        };
        if self.cb_include.is_selected() != input.cb_include.is_selected() {
            return false;
        }
        self.equals_value(Some(input))
    }

    /// Java private `equalsValue(DirectivePanel)`.
    fn equals_value(&self, input: Option<&DirectivePanel>) -> bool {
        let Some(input) = input else {
            return false;
        };
        if let Some(cb_value) = &self.cb_value {
            if cb_value.get_selected_index()
                != input.cb_value.as_ref().unwrap().get_selected_index()
            {
                return false;
            }
        } else if !FieldValidator::equals(
            self.field_type,
            self.tf_value.as_ref().unwrap().get_text_void().as_deref(),
            input.tf_value.as_ref().unwrap().get_text_void().as_deref(),
        ) {
            return false;
        }
        true
    }

    /// Java private `initValue(Directive)`.
    fn init_value(&self) {
        let values = self.directive.get_values();
        let value = values.get_value();
        if let Some(value) = value {
            if let Some(cb_value) = &self.cb_value {
                if self.boolean_value_type {
                    cb_value.set_selected_index(if value.to_boolean() {
                        YES_INDEX
                    } else {
                        NO_INDEX
                    });
                } else {
                    // Find the matching value and display for corresponding description.
                    let mut index = -1;
                    let s_value = value.to_string();
                    if !s_value
                        .chars()
                        .all(|c| matches!(c, ' ' | '\t' | '\n' | '\u{0B}' | '\u{0C}' | '\r'))
                    {
                        let value_list = self.value_list.as_ref().unwrap();
                        for (i, listed) in value_list.iter().enumerate() {
                            if *listed == s_value {
                                index = i as i32;
                                break;
                            }
                        }
                    }
                    cb_value.set_selected_index(index);
                }
            } else {
                self.tf_value
                    .as_ref()
                    .unwrap()
                    .set_text_string(Some(&value.to_string()));
            }
        }
    }

    /// Java package-private `isDifferentFromCheckpoint(boolean)`.
    pub fn is_different_from_checkpoint(&self, check_include: bool) -> bool {
        if check_include && self.cb_include.is_different_from_checkpoint_boolean(true) {
            return true;
        }
        if let Some(cb_value) = &self.cb_value {
            return cb_value.is_different_from_checkpoint(true);
        }
        self.tf_value
            .as_ref()
            .unwrap()
            .is_different_from_checkpoint(true)
    }

    /// Java package-private `isEnabled()`.
    pub fn is_enabled(&self) -> bool {
        self.cb_include.is_enabled()
    }

    /// Java package-private `isInclude()`.  True if include is checked and enabled.
    pub fn is_include(&self) -> bool {
        self.cb_include.is_selected() && self.cb_include.is_enabled()
    }

    /// Java package-private `isVisible()`.
    pub fn is_visible(&self) -> bool {
        self.pnl_root.is_visible()
    }

    /// Java private `resetValue()`.  Unused in the source.
    fn reset_value(&self) {
        if let Some(cb_value) = &self.cb_value {
            if self.boolean_value_type {
                cb_value.set_selected_index(NO_INDEX);
            } else {
                cb_value.set_selected_index(-1);
            }
        } else {
            self.tf_value.as_ref().unwrap().set_text_string(Some(""));
        }
    }

    /// Java private `setEnabled(boolean)`.  Unused in the source.
    fn set_enabled(&self, enable: bool) {
        self.cb_include.set_enabled(enable);
        self.enable_value();
    }

    /// Java private `setIncluded()`.  Sets the include checkbox from the display
    /// settings.
    fn set_included(&self) {
        let included = self.cb_include.is_selected();
        self.tool.set_debug(self.debug.get());
        if self
            .tool
            .is_toggle_directive_included(Some(&self.directive), included)
        {
            self.cb_include.set_selected_boolean(!included);
        }
        self.tool.reset_debug();
        self.enable_value();
    }

    /// Java package-private `setVisible(boolean)`.
    pub fn set_visible(&self, input: bool) {
        self.pnl_root.set_visible(input);
    }

    /// Java private `setTooltips()`.
    fn set_tooltips(&self) {
        let (value_string, default_value_string) = {
            let values = self.directive.get_values();
            (
                values.get_value().map(|value| value.to_string()),
                values.get_default_value().map(|value| value.to_string()),
            )
        };
        let mut debug_string = String::new();
        if self.debug.get().is_extra_verbose() {
            debug_string = format!(
                "  Type:{}, Batch:{}, Tmplt:{}, Etomo:{}, AxisLevelData:{}",
                self.directive
                    .get_value_type()
                    .map(|value_type| value_type.to_string())
                    .unwrap_or_else(|| "null".to_string()),
                self.directive.is_batch(),
                self.directive.is_template(),
                self.directive
                    .get_etomo_column()
                    .map(|column| column.to_string())
                    .unwrap_or_else(|| "null".to_string()),
                self.directive
                    .get_in_directive_file_debug_string()
                    .unwrap_or_else(|| "null".to_string())
            );
        }
        let tooltip = format!(
            "{}:  {}.{}{}{}",
            self.directive
                .get_key_description()
                .unwrap_or_else(|| "null".to_string()),
            self.directive.get_description().unwrap_or("null"),
            match &value_string {
                Some(value_string) => format!("  Dataset value:{value_string}"),
                None => String::new(),
            },
            match &default_value_string {
                Some(default_value_string) => format!("  Original value:{default_value_string}"),
                None => String::new(),
            },
            debug_string
        );
        self.cb_include.set_tool_tip_text_string(Some(&tooltip));
        if let Some(cb_value) = &self.cb_value {
            cb_value.set_tool_tip_text(Some(&tooltip));
        } else {
            self.tf_value
                .as_ref()
                .unwrap()
                .set_tool_tip_text(Some(&tooltip));
            if let Some(sb_file_value) = &self.sb_file_value {
                sb_file_value.get_component().set_tool_tip_text(
                    tooltip_formatter::INSTANCE
                        .format(Some(&tooltip))
                        .as_deref(),
                );
            }
        }
    }
}

/// Java `toString()`: the directive's title.
impl std::fmt::Display for DirectivePanel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.directive.get_title().as_deref().unwrap_or("null"))
    }
}
