//! `IMOD/Etomo/src/etomo/ui/swing/DirectiveEditorDialog.java`.
//!
//! The directive editor: a Source panel (dataset, save time stamps), a Control Panel
//! (include/exclude check boxes per directive file, show settings, Close All Sections,
//! Save and Cancel) and three columns of `DirectiveSectionPanel`s built from the
//! `DirectiveEditorBuilder`'s sections.  Implements `Expandable` and
//! `DirectiveDisplaySettings`.  An event dispatch thread object, created as `Rc<Self>`
//! by [`DirectiveEditorDialog::get_instance`].

use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::{Rc, Weak};
use std::sync::Arc;

use super::check_box::CheckBox;
use super::directive_section_panel::DirectiveSectionPanel;
use super::etched_border::EtchedBorder;
use super::expand_button::ExpandButton;
use super::expandable::Expandable;
use super::file_chooser;
use super::global_expand_button::GlobalExpandButton;
use super::labeled_text_field::LabeledTextField;
use super::multi_line_button::MultiLineButton;
use super::panel_header::PanelHeader;
use super::single_line_button::SingleLineButton;
use super::ui_harness;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::jdk::{ActionEvent, JComponent};
use crate::imod::etomo::logic::directive_editor_builder::DirectiveEditorBuilder;
use crate::imod::etomo::logic::directive_tool::DirectiveTool;
use crate::imod::etomo::storage::autodoc_filter::AutodocFilter;
use crate::imod::etomo::storage::directive::Directive;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::directive_file_type::{self, DirectiveFileType};
use crate::imod::etomo::ui::directive_display_settings::DirectiveDisplaySettings;
use crate::imod::etomo::ui::field::Field;
use crate::imod::etomo::ui::field_type::FieldType;
use crate::imod::etomo::util::utilities;

/// Java `public static final String rcsid`.
pub const RCSID: &str = "$Id:$";

/// Java `public final class DirectiveEditorDialog implements Expandable,
/// DirectiveDisplaySettings`.
pub struct DirectiveEditorDialog {
    /// Java private final `cbInclude = new CheckBox[DirectiveFileType.NUM]`.
    cb_include: Vec<Rc<CheckBox>>,
    /// Java private final `cbExclude = new CheckBox[DirectiveFileType.NUM]`.
    cb_exclude: Vec<Rc<CheckBox>>,
    /// Java private final `cbShowHidden`.
    cb_show_hidden: Rc<CheckBox>,
    /// Java private final `cbShowUnchanged`.
    cb_show_unchanged: Rc<CheckBox>,
    /// Java private final `cbShowOnlyIncluded`.
    cb_show_only_included: Rc<CheckBox>,
    /// Java private final `pnlControlBody`.
    pnl_control_body: Rc<JComponent>,
    /// Java private final `pnlRoot`.
    pnl_root: Rc<JComponent>,
    /// Java private final `pnlSourceBody`.
    pnl_source_body: Rc<JComponent>,
    /// Java private final `sectionArray`.
    section_array: RefCell<Vec<Rc<DirectiveSectionPanel>>>,
    /// Java private final `btnCloseAll`.
    btn_close_all: Rc<SingleLineButton>,
    /// Java private final `btnSave`.
    btn_save: Rc<MultiLineButton>,
    /// Java private final `btnCancel`.
    btn_cancel: Rc<MultiLineButton>,
    /// Java private final `ltfSource`.
    ltf_source: Rc<LabeledTextField>,
    /// Java private final `ltfFileTimestamp`.
    ltf_file_timestamp: Rc<LabeledTextField>,
    /// Java private final `ltfDatasetTimestamp`.
    ltf_dataset_timestamp: Rc<LabeledTextField>,
    /// Java private `lastFileChooserLocation`, initialised to null.
    last_file_chooser_location: RefCell<Option<PathBuf>>,
    /// Java private final `manager`.
    manager: &'static dyn BaseManager,
    /// Java private final `phControl`.
    ph_control: Rc<PanelHeader>,
    /// Java private final `phSource`.
    ph_source: Rc<PanelHeader>,
    /// Java private final `tool`.
    tool: Rc<DirectiveTool>,
    /// Java private final `fileTypeExists`.
    file_type_exists: Vec<bool>,
    /// Java private final `builder`.
    builder: Rc<DirectiveEditorBuilder>,
    /// Java private final `type`.
    r#type: Option<DirectiveFileType>,
    /// Java `this`.
    self_ref: Weak<DirectiveEditorDialog>,
}

impl DirectiveEditorDialog {
    /// Java private `DirectiveEditorDialog(BaseManager, DirectiveFileType,
    /// DirectiveEditorBuilder)`.
    fn new(
        manager: &'static dyn BaseManager,
        r#type: Option<DirectiveFileType>,
        builder: Rc<DirectiveEditorBuilder>,
    ) -> Rc<DirectiveEditorDialog> {
        Rc::new_cyclic(|self_ref: &Weak<DirectiveEditorDialog>| {
            let expandable: Weak<dyn Expandable> = self_ref.clone();
            let display_settings: Weak<dyn DirectiveDisplaySettings> = self_ref.clone();
            let ph_control = PanelHeader::get_instance(
                Some("Control Panel"),
                Some(expandable.clone()),
                Some(DialogType::DirectiveEditor),
            );
            let ph_source = PanelHeader::get_instance(
                Some("Source"),
                Some(expandable),
                Some(DialogType::DirectiveEditor),
            );
            let file_type_exists = builder.get_file_type_exists();
            let tool = match r#type {
                None => DirectiveTool::new(None, false, display_settings),
                Some(r#type) => DirectiveTool::new(
                    Some(r#type),
                    file_type_exists[r#type.get_index() as usize],
                    display_settings,
                ),
            };
            DirectiveEditorDialog {
                // The arrays' elements are created by createPanel.
                cb_include: (0..directive_file_type::NUM)
                    .map(|_| CheckBox::new_void())
                    .collect(),
                cb_exclude: (0..directive_file_type::NUM)
                    .map(|_| CheckBox::new_void())
                    .collect(),
                cb_show_hidden: CheckBox::new_string(Some("Show hidden")),
                cb_show_unchanged: CheckBox::new_string(Some("Show unchanged")),
                cb_show_only_included: CheckBox::new_string(Some("Show only included")),
                pnl_control_body: JComponent::new_panel(),
                pnl_root: JComponent::new_panel(),
                pnl_source_body: JComponent::new_panel(),
                section_array: RefCell::new(Vec::new()),
                btn_close_all: SingleLineButton::new_string(Some("Close All Sections")),
                btn_save: MultiLineButton::new_string(Some("Save")),
                btn_cancel: MultiLineButton::new_string(Some("Cancel")),
                ltf_source: LabeledTextField::new_field_type_string(
                    FieldType::String,
                    Some("Dataset: "),
                ),
                ltf_file_timestamp: LabeledTextField::new_field_type_string(
                    FieldType::String,
                    Some("File saved at: "),
                ),
                ltf_dataset_timestamp: LabeledTextField::new_field_type_string(
                    FieldType::String,
                    Some("Dataset saved at: "),
                ),
                last_file_chooser_location: RefCell::new(None),
                manager,
                ph_control,
                ph_source,
                tool: Rc::new(tool),
                file_type_exists,
                builder,
                r#type,
                self_ref: self_ref.clone(),
            }
        })
    }

    /// Java static `getInstance(BaseManager, DirectiveFileType, DirectiveEditorBuilder,
    /// AxisType, String, String, StringBuffer)`.
    #[allow(clippy::too_many_arguments)]
    pub fn get_instance(
        manager: &'static dyn BaseManager,
        r#type: Option<DirectiveFileType>,
        builder: Rc<DirectiveEditorBuilder>,
        source_axis_type: AxisType,
        source_status: Option<&str>,
        save_timestamp: Option<&str>,
        errmsg: Option<&str>,
    ) -> Rc<DirectiveEditorDialog> {
        let instance = DirectiveEditorDialog::new(manager, r#type, builder);
        instance.create_panel(source_axis_type, source_status, save_timestamp, errmsg);
        instance.set_tooltips();
        instance.add_listeners();
        instance
    }

    /// Java private `createPanel(AxisType, String, String, StringBuffer)`.
    fn create_panel(
        &self,
        source_axis_type: AxisType,
        source_status: Option<&str>,
        save_timestamp: Option<&str>,
        errmsg: Option<&str>,
    ) {
        // construct
        let n_columns = 3;
        // Swing: JScrollPane scrollPane = new JScrollPane(pnlRoot); the scroll pane is
        // never added anywhere (the container is pnlRoot).
        let pnl_control = JComponent::new_panel();
        let pnl_show_checkboxes = JComponent::new_panel();
        let pnl_source = JComponent::new_panel();
        let pnl_include_checkboxes = JComponent::new_panel();
        let pnl_show = JComponent::new_panel();
        let pnl_include_settings = JComponent::new_panel();
        let pnl_sections = JComponent::new_panel();
        let mut pnl_section_columns: Vec<Rc<JComponent>> = Vec::with_capacity(n_columns);
        let pnl_buttons = JComponent::new_panel();
        let mut l_include: Vec<Rc<JComponent>> = Vec::new();
        let mut index = -1;
        if let Some(r#type) = self.r#type {
            index = r#type.get_index();
        }
        for i in 0..directive_file_type::NUM {
            let iu = i as usize;
            // cbInclude[i] = new CheckBox(); cbExclude[i] = new CheckBox()
            self.cb_include[iu].set_action_command(DirectiveFileType::get_label_from_index(i));
            self.cb_exclude[iu].set_action_command(DirectiveFileType::get_label_from_index(i));
            // `new JLabel(null)` has no text.
            l_include.push(JComponent::new_label(
                DirectiveFileType::to_string_from_index(i).unwrap_or(""),
            ));
            // init
            if !self.file_type_exists[iu] {
                self.cb_include[iu].set_enabled(false);
                self.cb_exclude[iu].set_enabled(false);
                l_include[iu].set_enabled(false);
            } else if i == index {
                // The matching file type should be included
                self.cb_include[iu].set_selected_boolean(true);
            } else {
                // Lower priority file types should be excluded. This class is indexed in
                // order of priority.
                if i < index {
                    self.cb_exclude[iu].set_selected_boolean(true);
                }
            }
        }
        // init
        if self.r#type == Some(DirectiveFileType::Batch) {
            self.cb_show_hidden.set_enabled(false);
        }
        self.ltf_source.set_editable(false);
        self.ltf_source.set_text_string(source_status);
        self.ltf_dataset_timestamp.set_editable(false);
        self.ltf_dataset_timestamp.set_text_string(save_timestamp);
        self.ltf_file_timestamp.set_editable(false);
        // Swing layout: btnCloseAll.setSize().
        // Fill sectionArray
        {
            let directive_map = self.builder.get_directive_map();
            for descr_section in self.builder.get_section_array() {
                if descr_section.is_contains_editable_directives() {
                    let section = DirectiveSectionPanel::get_instance(
                        self.manager,
                        descr_section,
                        &directive_map,
                        source_axis_type,
                        self.tool.clone(),
                    );
                    self.section_array.borrow_mut().push(section);
                }
            }
        }
        // root panel (BoxLayout Y_AXIS)
        // Swing: ToolTipManager.sharedInstance().registerComponent(pnlRoot).
        self.pnl_root.add(&pnl_source);
        self.pnl_root.add(&pnl_control);
        self.pnl_root.add(&pnl_sections);
        // source panel (BoxLayout Y_AXIS, an untitled etched border)
        pnl_source.add(&self.ph_source.get_container());
        pnl_source.add(&self.pnl_source_body);
        // source body panel (BoxLayout Y_AXIS)
        self.pnl_source_body.add(&self.ltf_source.get_component());
        self.pnl_source_body
            .add(&self.ltf_dataset_timestamp.get_component());
        self.pnl_source_body
            .add(&self.ltf_file_timestamp.get_component());
        // control panel (BoxLayout Y_AXIS, an untitled etched border)
        pnl_control.add(&self.ph_control.get_container());
        pnl_control.add(&self.pnl_control_body);
        // control body panel (BoxLayout X_AXIS, horizontal glue between the three)
        self.pnl_control_body.add(&pnl_include_settings);
        self.pnl_control_body.add(&pnl_show);
        self.pnl_control_body.add(&pnl_buttons);
        // include settings (BoxLayout Y_AXIS)
        pnl_include_settings.set_border_title(
            EtchedBorder::new(Some("Include Based on Directives in Files"))
                .get_title()
                .as_deref(),
        );
        pnl_include_settings.add(&pnl_include_checkboxes);
        // include checkboxes panel (GridLayout(0, 3))
        pnl_include_checkboxes.add(&JComponent::new_label("Include"));
        pnl_include_checkboxes.add(&JComponent::new_label("Exclude"));
        pnl_include_checkboxes.add(&JComponent::new_label("File"));
        for i in 0..directive_file_type::NUM as usize {
            pnl_include_checkboxes.add(&self.cb_include[i].get_component());
            pnl_include_checkboxes.add(&self.cb_exclude[i].get_component());
            pnl_include_checkboxes.add(&l_include[i]);
        }
        // show glue panel (BoxLayout Y_AXIS; rigid areas and glue are layout)
        pnl_show.add(&pnl_show_checkboxes);
        pnl_show.add(&self.btn_close_all.get_component());
        // show panel (BoxLayout Y_AXIS)
        pnl_show_checkboxes.set_border_title(
            EtchedBorder::new(Some("Show Directives"))
                .get_title()
                .as_deref(),
        );
        pnl_show_checkboxes.add(&self.cb_show_unchanged.get_component());
        pnl_show_checkboxes.add(&self.cb_show_hidden.get_component());
        pnl_show_checkboxes.add(&self.cb_show_only_included.get_component());
        // Buttons panel (BoxLayout Y_AXIS, vertical glue between)
        pnl_buttons.add(&self.btn_save.get_component());
        pnl_buttons.add(&self.btn_cancel.get_component());
        // sections (BoxLayout X_AXIS)
        // columns
        let section_array = self.section_array.borrow().clone();
        let n_sections = section_array.len();
        let n_rows = n_sections / n_columns;
        let mut remainder = (n_sections % n_columns) as i32;
        let mut iterator = section_array.iter();
        let mut column_section_array: Vec<Rc<DirectiveSectionPanel>> = Vec::new();
        for i in 0..n_columns {
            // Build the columns (BoxLayout Y_AXIS).
            let column = JComponent::new_panel();
            pnl_section_columns.push(column.clone());
            pnl_sections.add(&column);
            if i < n_columns - 1 {
                // Swing layout: rigid areas around the separator.
                pnl_sections.add(&JComponent::new_other());
            }
            // Add the checkboxes and tempoarily store the sections in this column.
            column_section_array.clear();
            let extra_row = if remainder > 0 { 1 } else { 0 };
            remainder -= 1;
            for _j in 0..n_rows + extra_row {
                let Some(section_panel) = iterator.next() else {
                    break;
                };
                column_section_array.push(section_panel.clone());
                let panel = JComponent::new_panel();
                // panel (BoxLayout X_AXIS, horizontal glue after the check box)
                panel.add(&section_panel.get_show_check_box());
                column.add(&panel);
            }
            // Swing layout: padding between the chckboxes and the section panels
            // (x0_y20 without an extra row, x0_y5 with one).
            // Add the section panels.
            for section_panel in &column_section_array {
                column.add(&section_panel.get_component());
            }
            // Swing layout: vertical glue.
        }
        if let Some(errmsg) = errmsg
            && !errmsg.is_empty()
        {
            let manager = self.manager;
            ui_harness::with(|harness| {
                harness.open_message_dialog_base_manager_string_string(
                    Some(manager),
                    errmsg,
                    "Problems Building Editor",
                )
            });
        }
    }

    /// Java private `addListeners()`.
    fn add_listeners(&self) {
        // DirectiveListener
        let adaptee = self.self_ref.clone();
        let listener: crate::imod::etomo::jdk::ActionListener =
            Rc::new(move |event: &ActionEvent| {
                if let Some(adaptee) = adaptee.upgrade() {
                    adaptee.action(event.get_action_command());
                }
            });
        self.cb_show_unchanged
            .add_action_listener(Some(listener.clone()));
        self.cb_show_hidden
            .add_action_listener(Some(listener.clone()));
        self.cb_show_only_included
            .add_action_listener(Some(listener.clone()));
        self.btn_close_all.add_action_listener(listener.clone());
        self.btn_cancel.add_action_listener(listener.clone());
        self.btn_save.add_action_listener(listener);
        // IncludeListener, ExcludeListener
        let adaptee = self.self_ref.clone();
        let include_listener: crate::imod::etomo::jdk::ActionListener =
            Rc::new(move |event: &ActionEvent| {
                if let Some(adaptee) = adaptee.upgrade() {
                    adaptee.include_action(event.get_action_command());
                }
            });
        let adaptee = self.self_ref.clone();
        let exclude_listener: crate::imod::etomo::jdk::ActionListener =
            Rc::new(move |event: &ActionEvent| {
                if let Some(adaptee) = adaptee.upgrade() {
                    adaptee.exclude_action(event.get_action_command());
                }
            });
        for i in 0..directive_file_type::NUM as usize {
            self.cb_include[i].add_action_listener(Some(include_listener.clone()));
            self.cb_exclude[i].add_action_listener(Some(exclude_listener.clone()));
        }
    }

    /// Java private `setTooltips()`.
    fn set_tooltips(&self) {
        for i in 0..directive_file_type::NUM {
            // `DirectiveFileType.getInstance(i).getLocalFile(manager, AxisID.ONLY)
            // .getName()`.
            let file_name = DirectiveFileType::get_instance_from_index(i)
                .and_then(|file_type| {
                    file_type.get_local_file(Some(self.manager), Some(AxisID::Only))
                })
                .map(|file| utilities::java_io_file_get_name(&file.to_string_lossy()))
                .unwrap_or_else(|| "null".to_string());
            self.cb_include[i as usize].set_tool_tip_text_string(Some(&format!(
                "Include directives found in {file_name}"
            )));
            self.cb_exclude[i as usize].set_tool_tip_text_string(Some(&format!(
                "Exclude directives found in {file_name}"
            )));
        }
        self.cb_show_unchanged
            .set_tool_tip_text_string(Some("Show directives whose values have not changed."));
        self.cb_show_hidden.set_tool_tip_text_string(Some(
            "Show directives that are usually not included in a directive file.",
        ));
    }

    /// Java `getContainer()`.
    pub fn get_container(&self) -> Rc<JComponent> {
        self.pnl_root.clone()
    }

    /// Java `getSaveFileAbsPath()`.
    pub fn get_save_file_abs_path(&self) -> Option<String> {
        let chooser = ui_harness::with(|harness| harness.get_file_chooser())?;
        let last_file_chooser_location = self.last_file_chooser_location.borrow().clone();
        if let Some(location) = last_file_chooser_location {
            chooser.set_current_directory(Some(&location));
        } else {
            chooser.set_current_directory(Some(&self.builder.get_default_save_location()));
        }
        chooser.set_dialog_title(Some("Save As"));
        chooser.set_file_filter(Some(Rc::new(AutodocFilter::new())));
        if self.r#type == Some(DirectiveFileType::User) {
            chooser.set_file_hiding_enabled(false);
        }
        if chooser.show_open_dialog(Some(&self.pnl_root)) == file_chooser::APPROVE_OPTION {
            let file = chooser.get_selected_file();
            if let Some(file) = file {
                return Some(utilities::java_io_file_get_absolute_path(
                    &file.to_string_lossy(),
                ));
            }
        }
        *self.last_file_chooser_location.borrow_mut() = chooser.get_current_directory();
        None
    }

    /// Java `getComments()`.
    pub fn get_comments(&self) -> Vec<String> {
        vec![
            format!(
                "{} {}",
                self.ltf_source.get_label(),
                self.ltf_source.get_text_void().unwrap_or_default()
            ),
            format!(
                "{} {}",
                self.ltf_dataset_timestamp.get_label(),
                self.ltf_dataset_timestamp
                    .get_text_void()
                    .unwrap_or_default()
            ),
        ]
    }

    /// Java `getDroppedDirectives()`.
    pub fn get_dropped_directives(&self) -> Vec<String> {
        self.builder.get_dropped_directives()
    }

    /// Java `setFileTimestamp(Date)`, with the date as milliseconds since the epoch.
    pub fn set_file_timestamp(&self, date_millis: i64) {
        let date = utilities::java_util_date_to_string(date_millis);
        let date_array: Vec<&str> = date
            .split(|c: char| matches!(c, ' ' | '\t' | '\n' | '\u{0B}' | '\u{0C}' | '\r'))
            .filter(|part| !part.is_empty())
            .collect();
        let mut buffer = String::new();
        let n = date_array.len().saturating_sub(2);
        for part in &date_array[..n] {
            buffer.push_str(part);
            buffer.push(' ');
        }
        self.ltf_file_timestamp
            .set_text_string(Some(buffer.trim_matches(|c: char| c <= ' ')));
    }

    /// Java `getIncludeDirectiveList()`.
    pub fn get_include_directive_list(&self) -> Vec<Arc<Directive>> {
        let mut directive_list = Vec::new();
        for section in self.section_array.borrow().iter() {
            directive_list.extend(section.get_include_directive_list());
        }
        directive_list
    }

    /// Java `checkpoint()`.
    pub fn checkpoint(&self) {
        for section in self.section_array.borrow().iter() {
            section.checkpoint();
        }
    }

    /// Java package-private `getSectionList()`.
    pub fn get_section_list(&self) -> Vec<Rc<DirectiveSectionPanel>> {
        self.section_array.borrow().clone()
    }

    /// Java private `action(String)`.
    fn action(&self, action_command: Option<&str>) {
        let manager = self.manager;
        if self.btn_close_all.get_action_command().as_deref() == action_command {
            for section in self.section_array.borrow().iter() {
                section.close();
            }
            ui_harness::with(|harness| harness.pack_base_manager(Some(manager)));
        } else if self.btn_cancel.get_action_command().as_deref() == action_command {
            ui_harness::with(|harness| harness.cancel(Some(manager)));
        } else if self.btn_save.get_action_command().as_deref() == action_command {
            ui_harness::with(|harness| {
                harness.save_base_manager_axis_id(Some(manager), Some(AxisID::Only))
            });
        } else {
            if self.cb_show_only_included.get_action_command().as_deref() == action_command {
                let enable = !self.cb_show_only_included.is_selected();
                self.cb_show_unchanged.set_enabled(enable);
                self.cb_show_hidden
                    .set_enabled(enable && self.r#type != Some(DirectiveFileType::Batch));
            }
            self.msg_control_changed(false);
        }
    }

    /// Java `isDifferentFromCheckpoint(boolean)`.
    pub fn is_different_from_checkpoint(&self, check_include: bool) -> bool {
        self.section_array
            .borrow()
            .iter()
            .any(|section| section.is_different_from_checkpoint(check_include))
    }

    /// Java private `msgControlChanged(boolean)`.
    fn msg_control_changed(&self, include_change: bool) {
        let show_only_included = self.cb_show_only_included.is_selected();
        for section in self.section_array.borrow().iter() {
            section.msg_control_changed(include_change, show_only_included, show_only_included);
        }
        let manager = self.manager;
        ui_harness::with(|harness| harness.pack_base_manager(Some(manager)));
    }

    /// Java private `includeAction(String)`.
    fn include_action(&self, action_command: Option<&str>) {
        if let Some(directive_file_type) = DirectiveFileType::get_instance(action_command) {
            let index = directive_file_type.get_index() as usize;
            if self.cb_include[index].is_selected() || self.cb_exclude[index].is_selected() {
                self.cb_exclude[index].set_selected_boolean(false);
            }
        }
        self.msg_control_changed(true);
    }

    /// Java private `excludeAction(String)`.
    fn exclude_action(&self, action_command: Option<&str>) {
        if let Some(directive_file_type) = DirectiveFileType::get_instance(action_command) {
            let index = directive_file_type.get_index() as usize;
            if self.cb_exclude[index].is_selected() || self.cb_include[index].is_selected() {
                self.cb_include[index].set_selected_boolean(false);
            }
        }
        self.msg_control_changed(true);
    }
}

impl DirectiveDisplaySettings for DirectiveEditorDialog {
    /// Java `isInclude(int)`.
    fn is_include(&self, index: i32) -> bool {
        if index >= 0 && index < directive_file_type::NUM {
            return self.cb_include[index as usize].is_selected();
        }
        false
    }

    /// Java `isExclude(int)`.
    fn is_exclude(&self, index: i32) -> bool {
        if index >= 0 && index < directive_file_type::NUM {
            return self.cb_exclude[index as usize].is_selected();
        }
        false
    }

    /// Java `isShowUnchanged()`.
    fn is_show_unchanged(&self) -> bool {
        self.cb_show_unchanged.is_selected()
    }

    /// Java `isShowHidden()`.
    fn is_show_hidden(&self) -> bool {
        self.cb_show_hidden.is_selected()
    }

    /// Java `isShowOnlyIncluded()`.
    fn is_show_only_included(&self) -> bool {
        self.cb_show_only_included.is_selected()
    }
}

impl Expandable for DirectiveEditorDialog {
    /// Java `expand(ExpandButton)`.
    fn expand_expand_button(&self, button: &Rc<ExpandButton>) {
        if self.ph_control.equals_open_close(button) {
            self.pnl_control_body.set_visible(button.is_expanded());
        }
        if self.ph_source.equals_open_close(button) {
            self.pnl_source_body.set_visible(button.is_expanded());
        }
        let manager = self.manager;
        ui_harness::with(|harness| harness.pack_base_manager(Some(manager)));
    }

    /// Java final `expand(GlobalExpandButton)`: empty.
    fn expand_global_expand_button(&self, _button: &Rc<GlobalExpandButton>) {}
}
