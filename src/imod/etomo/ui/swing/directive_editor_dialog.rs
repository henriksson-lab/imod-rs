//! `IMOD/Etomo/src/etomo/ui/swing/DirectiveEditorDialog.java`.
//!
//! Swing layout, the shared `UIHarness`, and `JFileChooser` are presentation
//! boundaries.  This unit retains the dialog's construction, filtering,
//! section distribution, action dispatch, checkpoint, and save-file policy.
#![allow(dead_code)]

use std::path::PathBuf;

use super::check_box::CheckBox;
use super::directive_section_panel::{
    Directive, DirectiveDescrSection, DirectiveMap, DirectivePanel, DirectiveSectionPanel,
    DirectiveTool,
};
use super::expand_button::ExpandButton;
use super::expandable::Expandable;
use super::labeled_text_field::LabeledTextField;
use super::multi_line_button::MultiLineButton;
use super::panel_header::PanelHeader;
use super::process_dialog::GlobalExpandButton;
use super::single_line_button::SingleLineButton;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::directive_file_type::{DirectiveFileType, NUM};
use crate::imod::etomo::ui::directive_display_settings::DirectiveDisplaySettings;
use crate::imod::etomo::ui::field_type::FieldType;

/// The direct `DirectiveEditorBuilder` calls made by this source unit.
pub trait DirectiveEditorBuilder {
    fn get_file_type_exists(&self) -> [bool; NUM as usize];
    fn get_section_array(&self) -> Vec<DirectiveDescrSection>;
    fn get_directive_map(&self) -> DirectiveMap;
    fn get_default_save_location(&self) -> Option<PathBuf>;
    fn get_dropped_directives(&self) -> Vec<String>;
}

/// The source-visible `JFileChooser` calls made by `getSaveFileAbsPath`.
pub trait DirectiveEditorFileChooser {
    fn set_current_directory(&mut self, directory: Option<&std::path::Path>);
    fn get_current_directory(&self) -> Option<PathBuf>;
    fn set_dialog_title(&mut self, title: &str);
    fn set_autodoc_filter(&mut self);
    fn set_file_hiding_enabled(&mut self, enabled: bool);
    fn show_open_dialog(&mut self) -> bool;
    fn get_selected_file(&self) -> Option<PathBuf>;
}

/// Direct global `UIHarness.INSTANCE` calls made by this source unit.
pub trait DirectiveEditorUiHarness {
    fn open_message_dialog(
        &mut self,
        manager: &'static dyn BaseManager,
        message: &str,
        title: &str,
    );
    fn pack(&mut self, manager: &'static dyn BaseManager);
    fn cancel(&mut self, manager: &'static dyn BaseManager);
    fn save(&mut self, manager: &'static dyn BaseManager, axis_id: AxisID);
}

/// Source-visible Swing container/layout state owned by this dialog.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct DirectiveEditorDialogLayout {
    pub tooltip_manager_registered: bool,
    pub root_order: Vec<String>,
    pub source_order: Vec<String>,
    pub control_order: Vec<String>,
    pub control_body_order: Vec<String>,
    pub include_checkbox_order: Vec<String>,
    pub include_label_enabled: [bool; NUM as usize],
    pub show_order: Vec<String>,
    pub show_checkbox_order: Vec<String>,
    pub buttons_order: Vec<String>,
    pub section_column_orders: [Vec<String>; 3],
    pub section_separators: usize,
    pub control_body_visible: bool,
    pub source_body_visible: bool,
}

/// Java final `DirectiveEditorDialog`.
pub struct DirectiveEditorDialog<P: DirectivePanel, B: DirectiveEditorBuilder> {
    pub rcsid: &'static str,
    pub cb_include: [CheckBox; NUM as usize],
    pub cb_exclude: [CheckBox; NUM as usize],
    pub cb_show_hidden: CheckBox,
    pub cb_show_unchanged: CheckBox,
    pub cb_show_only_included: CheckBox,
    pub section_array: Vec<DirectiveSectionPanel<P>>,
    pub btn_close_all: SingleLineButton,
    pub btn_save: MultiLineButton,
    pub btn_cancel: MultiLineButton,
    pub ltf_source: LabeledTextField,
    pub ltf_file_timestamp: LabeledTextField,
    pub ltf_dataset_timestamp: LabeledTextField,
    pub last_file_chooser_location: Option<PathBuf>,
    pub manager: &'static dyn BaseManager,
    pub ph_control: PanelHeader,
    pub ph_source: PanelHeader,
    pub tool: &'static DirectiveTool,
    pub file_type_exists: [bool; NUM as usize],
    pub builder: B,
    pub directive_file_type: Option<DirectiveFileType>,
    pub layout: DirectiveEditorDialogLayout,
    pub listeners_added: bool,
    /// Native UIHarness pack boundary requested by Java `expand(ExpandButton)`.
    pub pack_requested: bool,
}

impl<P: DirectivePanel, B: DirectiveEditorBuilder> DirectiveEditorDialog<P, B> {
    /// Java private `DirectiveEditorDialog(BaseManager, DirectiveFileType,
    /// DirectiveEditorBuilder)`.
    fn new(
        manager: &'static dyn BaseManager,
        directive_file_type: Option<DirectiveFileType>,
        builder: B,
    ) -> Self {
        let file_type_exists = builder.get_file_type_exists();
        let tool = Box::leak(Box::new(DirectiveTool));
        Self {
            rcsid: "$Id:$",
            cb_include: std::array::from_fn(|_| CheckBox::new()),
            cb_exclude: std::array::from_fn(|_| CheckBox::new()),
            cb_show_hidden: CheckBox::new_with_text("Show hidden"),
            cb_show_unchanged: CheckBox::new_with_text("Show unchanged"),
            cb_show_only_included: CheckBox::new_with_text("Show only included"),
            section_array: Vec::new(),
            btn_close_all: SingleLineButton::new_with_label(Some("Close All Sections")),
            btn_save: MultiLineButton::new_with_label(Some("Save")),
            btn_cancel: MultiLineButton::new_with_label(Some("Cancel")),
            ltf_source: LabeledTextField::new(FieldType::String, "Dataset: "),
            ltf_file_timestamp: LabeledTextField::new(FieldType::String, "File saved at: "),
            ltf_dataset_timestamp: LabeledTextField::new(FieldType::String, "Dataset saved at: "),
            last_file_chooser_location: None,
            manager,
            ph_control: PanelHeader::new(
                "Control Panel",
                false,
                false,
                DialogType::DirectiveEditor,
                true,
                false,
                true,
                false,
                true,
            ),
            ph_source: PanelHeader::new(
                "Source",
                false,
                false,
                DialogType::DirectiveEditor,
                true,
                false,
                true,
                false,
                true,
            ),
            tool,
            file_type_exists,
            builder,
            directive_file_type,
            layout: DirectiveEditorDialogLayout {
                control_body_visible: true,
                source_body_visible: true,
                ..Default::default()
            },
            listeners_added: false,
            pack_requested: false,
        }
    }

    /// Java static `getInstance(BaseManager, DirectiveFileType,
    /// DirectiveEditorBuilder, AxisType, String, String, StringBuffer)`.
    pub fn get_instance(
        manager: &'static dyn BaseManager,
        directive_file_type: Option<DirectiveFileType>,
        builder: B,
        source_axis_type: AxisType,
        source_status: &str,
        save_timestamp: &str,
        errmsg: Option<&str>,
        ui_harness: &mut dyn DirectiveEditorUiHarness,
    ) -> Self {
        let mut instance = Self::new(manager, directive_file_type, builder);
        instance.create_panel(
            source_axis_type,
            source_status,
            save_timestamp,
            errmsg,
            ui_harness,
        );
        instance.set_tooltips();
        instance.add_listeners();
        instance
    }

    /// Java private `createPanel(AxisType, String, String, StringBuffer)`.
    fn create_panel(
        &mut self,
        source_axis_type: AxisType,
        source_status: &str,
        save_timestamp: &str,
        errmsg: Option<&str>,
        ui_harness: &mut dyn DirectiveEditorUiHarness,
    ) {
        let index = self
            .directive_file_type
            .map_or(-1, DirectiveFileType::get_index);
        for i in 0..NUM as usize {
            let directive_file_type = DirectiveFileType::get_instance_from_index(i as i32)
                .expect("DirectiveFileType.NUM matches its source enum instances");
            self.cb_include[i].set_action_command(Some(directive_file_type.get_label()));
            self.cb_exclude[i].set_action_command(Some(directive_file_type.get_label()));
            if !self.file_type_exists[i] {
                self.cb_include[i].set_enabled(false);
                self.cb_exclude[i].set_enabled(false);
                self.layout.include_label_enabled[i] = false;
            } else if i as i32 == index {
                self.cb_include[i].set_selected(true);
            } else if (i as i32) < index {
                self.cb_exclude[i].set_selected(true);
            }
            self.layout.include_checkbox_order.extend([
                format!("include:{i}"),
                format!("exclude:{i}"),
                format!("label:{i}"),
            ]);
        }
        if self.directive_file_type == Some(DirectiveFileType::Batch) {
            self.cb_show_hidden.set_enabled(false);
        }
        self.ltf_source.set_editable(false);
        self.ltf_source.set_text(source_status);
        self.ltf_dataset_timestamp.set_editable(false);
        self.ltf_dataset_timestamp.set_text(save_timestamp);
        self.ltf_file_timestamp.set_editable(false);
        self.btn_close_all.set_size();

        let directive_map = self.builder.get_directive_map();
        for descr_section in self.builder.get_section_array() {
            if descr_section.is_contains_editable_directives() {
                self.section_array.push(DirectiveSectionPanel::get_instance(
                    self.manager,
                    descr_section,
                    directive_map.clone(),
                    source_axis_type,
                    self.tool,
                ));
            }
        }

        self.layout.tooltip_manager_registered = true;
        self.layout.root_order = vec!["source".into(), "control".into(), "sections".into()];
        self.layout.source_order = vec!["header".into(), "body".into()];
        self.layout.control_order = vec!["header".into(), "body".into()];
        self.layout.control_body_order = vec![
            "include-settings".into(),
            "horizontal-glue".into(),
            "show".into(),
            "horizontal-glue".into(),
            "buttons".into(),
        ];
        self.layout.show_order = vec![
            "show-checkboxes".into(),
            "rigid-x0-y2".into(),
            "close-all".into(),
            "rigid-x0-y2".into(),
            "vertical-glue".into(),
        ];
        self.layout.show_checkbox_order = vec![
            "show-unchanged".into(),
            "show-hidden".into(),
            "rigid-x0-y10".into(),
            "show-only-included".into(),
            "rigid-x0-y2".into(),
        ];
        self.layout.buttons_order = vec![
            "vertical-glue".into(),
            "save".into(),
            "vertical-glue".into(),
            "cancel".into(),
            "vertical-glue".into(),
        ];

        let n_sections = self.section_array.len();
        let n_rows = n_sections / 3;
        let mut remainder = n_sections % 3;
        let mut section_index = 0;
        for i in 0..3 {
            if i < 2 {
                self.layout.section_separators += 1;
            }
            let extra_row = remainder > 0;
            remainder = remainder.saturating_sub(1);
            let mut column_section_indices = Vec::new();
            for _ in 0..n_rows + usize::from(extra_row) {
                if section_index == n_sections {
                    break;
                }
                self.layout.section_column_orders[i].push(format!("show-checkbox:{section_index}"));
                column_section_indices.push(section_index);
                section_index += 1;
            }
            self.layout.section_column_orders[i].push(if extra_row {
                "rigid-x0-y5".into()
            } else {
                "rigid-x0-y20".into()
            });
            for section_index in column_section_indices {
                self.layout.section_column_orders[i].push(format!("section:{section_index}"));
            }
            self.layout.section_column_orders[i].push("vertical-glue".into());
        }
        if let Some(errmsg) = errmsg.filter(|message| !message.is_empty()) {
            ui_harness.open_message_dialog(self.manager, errmsg, "Problems Building Editor");
        }
    }

    /// Java private `addListeners()`.
    fn add_listeners(&mut self) {
        self.cb_show_unchanged.add_action_listener();
        self.cb_show_hidden.add_action_listener();
        self.cb_show_only_included.add_action_listener();
        self.btn_close_all.add_action_listener();
        self.btn_cancel.add_action_listener();
        self.btn_save.add_action_listener();
        for i in 0..NUM as usize {
            self.cb_include[i].add_action_listener();
            self.cb_exclude[i].add_action_listener();
        }
        self.listeners_added = true;
    }

    /// Java private `setTooltips()`.
    fn set_tooltips(&mut self) {
        for i in 0..NUM as usize {
            let directive_file_type = DirectiveFileType::get_instance_from_index(i as i32)
                .expect("DirectiveFileType.NUM matches its source enum instances");
            let file_name = directive_file_type
                .get_local_file(Some(self.manager), Some(AxisID::Only))
                .and_then(|file| {
                    file.file_name()
                        .map(|name| name.to_string_lossy().into_owned())
                })
                .unwrap_or_default();
            self.cb_include[i]
                .set_tool_tip_text(Some(&format!("Include directives found in {file_name}")));
            self.cb_exclude[i]
                .set_tool_tip_text(Some(&format!("Exclude directives found in {file_name}")));
        }
        self.cb_show_unchanged
            .set_tool_tip_text(Some("Show directives whose values have not changed."));
        self.cb_show_hidden.set_tool_tip_text(Some(
            "Show directives that are usually not included in a directive file.",
        ));
    }

    /// Java `getContainer()` at the Swing boundary.
    pub fn get_container(&self) -> &DirectiveEditorDialogLayout {
        &self.layout
    }

    /// Java `isInclude(int)`.
    pub fn is_include(&self, index: i32) -> bool {
        (0..NUM).contains(&index) && self.cb_include[index as usize].is_selected()
    }

    /// Java `getSaveFileAbsPath()`.
    pub fn get_save_file_abs_path(
        &mut self,
        file_chooser: Option<&mut dyn DirectiveEditorFileChooser>,
    ) -> Option<String> {
        let chooser = file_chooser?;
        let default_save_location = self.builder.get_default_save_location();
        chooser.set_current_directory(
            self.last_file_chooser_location
                .as_deref()
                .or(default_save_location.as_deref()),
        );
        chooser.set_dialog_title("Save As");
        chooser.set_autodoc_filter();
        if self.directive_file_type == Some(DirectiveFileType::User) {
            chooser.set_file_hiding_enabled(false);
        }
        if chooser.show_open_dialog() {
            if let Some(file) = chooser.get_selected_file() {
                return Some(file.to_string_lossy().into_owned());
            }
        }
        self.last_file_chooser_location = chooser.get_current_directory();
        None
    }

    /// Java `getComments()`.
    pub fn get_comments(&self) -> Vec<String> {
        vec![
            format!(
                "{} {}",
                self.ltf_source.get_label(),
                self.ltf_source.get_text()
            ),
            format!(
                "{} {}",
                self.ltf_dataset_timestamp.get_label(),
                self.ltf_dataset_timestamp.get_text()
            ),
        ]
    }

    /// Java `getDroppedDirectives()`.
    pub fn get_dropped_directives(&self) -> Vec<String> {
        self.builder.get_dropped_directives()
    }

    /// Java `setFileTimestamp(Date)`.  `ToString` is the Java `Date.toString`
    /// boundary used by this method.
    pub fn set_file_timestamp(&mut self, date: impl ToString) {
        let date = date.to_string();
        let date_array: Vec<_> = date.split_whitespace().collect();
        self.ltf_file_timestamp
            .set_text(&date_array[..date_array.len().saturating_sub(2)].join(" "));
    }

    /// Java `getIncludeDirectiveList()`.
    pub fn get_include_directive_list(&mut self) -> Vec<Directive> {
        let mut directive_list = Vec::new();
        for section in &mut self.section_array {
            directive_list.extend(section.get_include_directive_list());
        }
        directive_list
    }

    /// Java `checkpoint()`.
    pub fn checkpoint(&mut self) {
        for section in &mut self.section_array {
            section.checkpoint();
        }
    }

    /// Java `isExclude(int)`.
    pub fn is_exclude(&self, index: i32) -> bool {
        (0..NUM).contains(&index) && self.cb_exclude[index as usize].is_selected()
    }

    /// Java `isShowUnchanged()`.
    pub fn is_show_unchanged(&self) -> bool {
        self.cb_show_unchanged.is_selected()
    }

    /// Java `isShowHidden()`.
    pub fn is_show_hidden(&self) -> bool {
        self.cb_show_hidden.is_selected()
    }

    /// Java package-private `getSectionList()`.
    pub fn get_section_list(&self) -> &[DirectiveSectionPanel<P>] {
        &self.section_array
    }

    /// Java private `action(String)`.
    pub fn action(&mut self, action_command: &str, ui_harness: &mut dyn DirectiveEditorUiHarness) {
        if self.btn_close_all.multi_line_button.get_action_command() == Some(action_command) {
            for section in &mut self.section_array {
                section.close();
            }
            ui_harness.pack(self.manager);
        } else if self.btn_cancel.get_action_command() == Some(action_command) {
            ui_harness.cancel(self.manager);
        } else if self.btn_save.get_action_command() == Some(action_command) {
            ui_harness.save(self.manager, AxisID::Only);
        } else {
            if self.cb_show_only_included.get_action_command() == Some(action_command) {
                let enable = !self.cb_show_only_included.is_selected();
                self.cb_show_unchanged.set_enabled(enable);
                self.cb_show_hidden.set_enabled(
                    enable && self.directive_file_type != Some(DirectiveFileType::Batch),
                );
            }
            self.msg_control_changed(false, ui_harness);
        }
    }

    /// Java `isShowOnlyIncluded()`.
    pub fn is_show_only_included(&self) -> bool {
        self.cb_show_only_included.is_selected()
    }

    /// Java `isDifferentFromCheckpoint(boolean)`.
    pub fn is_different_from_checkpoint(&self, check_include: bool) -> bool {
        self.section_array
            .iter()
            .any(|section| section.is_different_from_checkpoint(check_include))
    }

    /// Java private `msgControlChanged(boolean)`.
    fn msg_control_changed(
        &mut self,
        include_change: bool,
        ui_harness: &mut dyn DirectiveEditorUiHarness,
    ) {
        let show_only_included = self.cb_show_only_included.is_selected();
        for section in &mut self.section_array {
            section.msg_control_changed(include_change, show_only_included, show_only_included);
        }
        ui_harness.pack(self.manager);
    }

    /// Java private `includeAction(String)`.
    pub fn include_action(
        &mut self,
        action_command: &str,
        ui_harness: &mut dyn DirectiveEditorUiHarness,
    ) {
        if let Some(directive_file_type) = DirectiveFileType::get_instance(Some(action_command)) {
            let index = directive_file_type.get_index() as usize;
            if self.cb_include[index].is_selected() || self.cb_exclude[index].is_selected() {
                self.cb_exclude[index].set_selected(false);
            }
        }
        self.msg_control_changed(true, ui_harness);
    }

    /// Java private `excludeAction(String)`.
    pub fn exclude_action(
        &mut self,
        action_command: &str,
        ui_harness: &mut dyn DirectiveEditorUiHarness,
    ) {
        if let Some(directive_file_type) = DirectiveFileType::get_instance(Some(action_command)) {
            let index = directive_file_type.get_index() as usize;
            if self.cb_exclude[index].is_selected() || self.cb_include[index].is_selected() {
                self.cb_include[index].set_selected(false);
            }
        }
        self.msg_control_changed(true, ui_harness);
    }
}

impl<P: DirectivePanel, B: DirectiveEditorBuilder> Expandable for DirectiveEditorDialog<P, B> {
    /// Java `expand(ExpandButton)`.
    fn expand_expand_button(&mut self, button: &ExpandButton) {
        if self.ph_control.equals_open_close(button) {
            self.layout.control_body_visible = button.is_expanded();
        }
        if self.ph_source.equals_open_close(button) {
            self.layout.source_body_visible = button.is_expanded();
        }
        self.pack_requested = true;
    }

    /// Java `expand(GlobalExpandButton)`, intentionally empty in the source.
    fn expand_global_button(&mut self, _button: &GlobalExpandButton) {}
}

impl<P: DirectivePanel, B: DirectiveEditorBuilder> DirectiveDisplaySettings
    for DirectiveEditorDialog<P, B>
{
    fn is_include(&self, index: i32) -> bool {
        self.is_include(index)
    }

    fn is_exclude(&self, index: i32) -> bool {
        self.is_exclude(index)
    }

    fn is_show_unchanged(&self) -> bool {
        self.is_show_unchanged()
    }

    fn is_show_hidden(&self) -> bool {
        self.is_show_hidden()
    }

    fn is_show_only_included(&self) -> bool {
        self.is_show_only_included()
    }
}

pub struct DirectiveListener;
impl DirectiveListener {
    #[allow(non_snake_case)]
    pub fn actionPerformed<P: DirectivePanel, B: DirectiveEditorBuilder>(
        dialog: &mut DirectiveEditorDialog<P, B>,
        command: &str,
        ui: &mut dyn DirectiveEditorUiHarness,
    ) {
        dialog.action(command, ui);
    }
}

pub struct IncludeListener;
impl IncludeListener {
    #[allow(non_snake_case)]
    pub fn actionPerformed<P: DirectivePanel, B: DirectiveEditorBuilder>(
        dialog: &mut DirectiveEditorDialog<P, B>,
        command: &str,
        ui: &mut dyn DirectiveEditorUiHarness,
    ) {
        dialog.include_action(command, ui);
    }
}

pub struct ExcludeListener;
impl ExcludeListener {
    #[allow(non_snake_case)]
    pub fn actionPerformed<P: DirectivePanel, B: DirectiveEditorBuilder>(
        dialog: &mut DirectiveEditorDialog<P, B>,
        command: &str,
        ui: &mut dyn DirectiveEditorUiHarness,
    ) {
        dialog.exclude_action(command, ui);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::etomo::directive_editor_manager::DirectiveEditorManager;
    use std::collections::HashMap;

    #[derive(Default)]
    struct Builder;
    impl DirectiveEditorBuilder for Builder {
        fn get_file_type_exists(&self) -> [bool; NUM as usize] {
            [true; NUM as usize]
        }
        fn get_section_array(&self) -> Vec<DirectiveDescrSection> {
            vec![DirectiveDescrSection {
                title: "One".into(),
                names: vec!["included".into()],
                contains_editable_directives: true,
            }]
        }
        fn get_directive_map(&self) -> DirectiveMap {
            DirectiveMap {
                directives: HashMap::from([(
                    "included".into(),
                    Directive {
                        name: "included".into(),
                    },
                )]),
            }
        }
        fn get_default_save_location(&self) -> Option<PathBuf> {
            Some(PathBuf::from("/tmp"))
        }
        fn get_dropped_directives(&self) -> Vec<String> {
            vec!["dropped".into()]
        }
    }

    struct TestPanel {
        include: bool,
        changed: bool,
        checkpoint_count: usize,
    }
    impl DirectivePanel for TestPanel {
        fn get_instance(
            _: &'static dyn BaseManager,
            _: &Directive,
            _: &'static DirectiveTool,
            _: AxisType,
        ) -> Self {
            Self {
                include: true,
                changed: false,
                checkpoint_count: 0,
            }
        }
        fn msg_control_changed(&mut self, _: bool, _: bool) -> bool {
            true
        }
        fn is_include(&self) -> bool {
            self.include
        }
        fn is_different_from_checkpoint(&self, _: bool) -> bool {
            self.changed
        }
        fn get_state(&mut self) -> Directive {
            Directive {
                name: "included".into(),
            }
        }
        fn checkpoint(&mut self) {
            self.checkpoint_count += 1;
        }
    }

    #[derive(Default)]
    struct Harness {
        packs: usize,
        saves: usize,
        cancels: usize,
        messages: Vec<String>,
    }
    impl DirectiveEditorUiHarness for Harness {
        fn open_message_dialog(&mut self, _: &'static dyn BaseManager, message: &str, _: &str) {
            self.messages.push(message.into());
        }
        fn pack(&mut self, _: &'static dyn BaseManager) {
            self.packs += 1;
        }
        fn cancel(&mut self, _: &'static dyn BaseManager) {
            self.cancels += 1;
        }
        fn save(&mut self, _: &'static dyn BaseManager, _: AxisID) {
            self.saves += 1;
        }
    }

    fn dialog(harness: &mut Harness) -> DirectiveEditorDialog<TestPanel, Builder> {
        let manager: &'static dyn BaseManager = DirectiveEditorManager::new(None, None, None, None);
        DirectiveEditorDialog::get_instance(
            manager,
            Some(DirectiveFileType::User),
            Builder,
            AxisType::SingleAxis,
            "dataset",
            "saved",
            Some("warning"),
            harness,
        )
    }

    #[test]
    fn factory_retains_source_initial_priority_layout_and_error_message() {
        let mut harness = Harness::default();
        let dialog = dialog(&mut harness);
        assert!(dialog.is_include(3));
        assert!(dialog.is_exclude(0));
        assert_eq!(dialog.layout.root_order, ["source", "control", "sections"]);
        assert_eq!(dialog.layout.section_separators, 2);
        assert!(dialog.listeners_added);
        assert_eq!(harness.messages, ["warning"]);
        assert_eq!(
            dialog.get_comments(),
            ["Dataset:  dataset", "Dataset saved at:  saved"]
        );
    }

    #[test]
    fn include_exclude_show_and_button_actions_follow_source_dispatch() {
        let mut harness = Harness::default();
        let mut dialog = dialog(&mut harness);
        dialog.cb_exclude[0].set_selected(true);
        dialog.cb_include[0].set_selected(true);
        dialog.include_action(DirectiveFileType::BatchDefaults.get_label(), &mut harness);
        assert!(!dialog.is_exclude(0));
        dialog.cb_show_only_included.set_selected(true);
        dialog.action("Show only included", &mut harness);
        assert!(!dialog.cb_show_unchanged.check_box.enabled);
        dialog.action("Save", &mut harness);
        dialog.action("Cancel", &mut harness);
        dialog.action("Close All Sections", &mut harness);
        assert_eq!((harness.saves, harness.cancels), (1, 1));
        assert!(!dialog.section_array[0].cb_show.is_selected());
        assert!(harness.packs >= 3);
    }

    #[test]
    fn checkpoint_collection_timestamp_and_difference_visit_all_sections() {
        let mut harness = Harness::default();
        let mut dialog = dialog(&mut harness);
        assert_eq!(dialog.get_dropped_directives(), ["dropped"]);
        assert_eq!(
            dialog.get_include_directive_list(),
            [Directive {
                name: "included".into()
            }]
        );
        dialog.set_file_timestamp("Wed Jan 01 12:34:56 CET 2020");
        assert_eq!(dialog.ltf_file_timestamp.get_text(), "Wed Jan 01 12:34:56");
        dialog.section_array[0].directive_panel_array[0].changed = true;
        assert!(dialog.is_different_from_checkpoint(false));
        dialog.checkpoint();
        assert_eq!(
            dialog.section_array[0].directive_panel_array[0].checkpoint_count,
            1
        );
    }
}
