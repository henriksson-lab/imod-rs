//! `IMOD/Etomo/src/etomo/ui/swing/TemplatePanel.java`.
//!
//! The Swing widgets, `ConfigTool`, `BaseManager`, settings dialog, directive
//! storage, and autodoc writer are explicit boundaries.  This unit keeps the
//! source panel's three-template ordering and selection rules intact.
#![allow(dead_code)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::imod::etomo::storage::autodoc::writable_autodoc::WritableAutodoc;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::directive_file_type::DirectiveFileType;

use super::settings_dialog::SettingsDialog;

const EMPTY_OPTION: &str = "None available";
const NO_SELECTION1: &str = "No selection (";
const NO_SELECTION2: &str = " available)";
const NUM_TEMPLATES: usize = 3;

/// Java `ActionListener` boundary used by the source's three ComboBoxes.
pub trait TemplateActionListener {
    fn action_performed(&mut self, action_command: &str);
}

/// The state TemplatePanel observes from Java `ComboBox`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TemplateComboBox {
    pub label: String,
    pub items: Vec<String>,
    pub selected_index: isize,
    pub enabled: bool,
    pub editable: bool,
    pub placeholder: Option<(String, String)>,
    pub tooltip: Option<String>,
    pub field_highlight: bool,
    pub action_listener_count: usize,
    pub focus_listener_count: usize,
}

impl TemplateComboBox {
    /// Java `ComboBox.getInstance(String)`.
    pub fn get_instance(label: &str) -> Self {
        Self {
            label: label.to_owned(),
            items: Vec::new(),
            selected_index: -1,
            enabled: true,
            editable: false,
            placeholder: None,
            tooltip: None,
            field_highlight: false,
            action_listener_count: 0,
            focus_listener_count: 0,
        }
    }

    /// Java `setPlaceholder`.
    pub fn set_placeholder(&mut self, empty_option: &str, no_selection: &str) {
        self.placeholder = Some((empty_option.to_owned(), no_selection.to_owned()));
    }

    /// Java `addItem`.
    pub fn add_item(&mut self, item: &str) {
        self.items.push(item.to_owned());
    }

    /// Java `unselect`.
    pub fn unselect(&mut self) {
        self.selected_index = -1;
    }

    /// Java `removeAllItems`.
    pub fn remove_all_items(&mut self) {
        self.items.clear();
        self.selected_index = -1;
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

    /// Java `isEnabled`.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Java `setEditable`.
    pub fn set_editable(&mut self, editable: bool) {
        self.editable = editable;
    }

    /// Java `setFieldHighlight`.
    pub fn set_field_highlight(&mut self) {
        self.field_highlight = true;
    }

    /// Java `setToolTipText`.
    pub fn set_tool_tip_text(&mut self, text: Option<&str>) {
        self.tooltip = text.map(str::to_owned);
    }

    /// Java `getActionCommand`.
    pub fn get_action_command(&self) -> &str {
        &self.label
    }
}

/// `JPanel` construction data owned by this source unit.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TemplatePanelLayout {
    pub y_axis_layout: bool,
    pub etched_border: bool,
    pub border_title: Option<String>,
    pub component_order: Vec<&'static str>,
}

impl Default for TemplatePanelLayout {
    fn default() -> Self {
        Self {
            y_axis_layout: false,
            etched_border: false,
            border_title: None,
            component_order: Vec::new(),
        }
    }
}

/// Direct `ConfigTool` filesystem boundary for TemplatePanel.
pub struct TemplateConfigTool;

impl TemplateConfigTool {
    /// Java `ConfigTool.getScopeTemplateFiles`.
    pub fn get_scope_template_files() -> Option<Vec<PathBuf>> {
        let calib = std::env::var_os("IMOD_CALIB_DIR")?;
        let directory = PathBuf::from(calib).join("ScopeTemplate");
        let entries = std::fs::read_dir(directory).ok()?;
        let mut files = BTreeMap::new();
        for entry in entries.flatten() {
            let path = entry.path();
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if path.is_file() && name.ends_with(".adoc") && !name.starts_with('.') {
                files.insert(name.into_owned(), path);
            }
        }
        (!files.is_empty()).then(|| files.into_values().collect())
    }

    /// Java `ConfigTool.getSystemTemplateFiles`; calib entries override IMOD_DIR entries.
    pub fn get_system_template_files() -> Option<Vec<PathBuf>> {
        let mut files = BTreeMap::new();
        if let Some(imod) = std::env::var_os("IMOD_DIR") {
            if let Ok(entries) = std::fs::read_dir(PathBuf::from(imod).join("SystemTemplate")) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    let name = entry.file_name();
                    let name = name.to_string_lossy();
                    if path.is_file() && name.ends_with(".adoc") && !name.starts_with('.') {
                        files.insert(name.into_owned(), path);
                    }
                }
            }
        }
        if let Some(calib) = std::env::var_os("IMOD_CALIB_DIR") {
            if let Ok(entries) = std::fs::read_dir(PathBuf::from(calib).join("SystemTemplate")) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    let name = entry.file_name();
                    let name = name.to_string_lossy();
                    if path.is_file() && name.ends_with(".adoc") && !name.starts_with('.') {
                        files.insert(name.into_owned(), path);
                    }
                }
            }
        }
        (!files.is_empty()).then(|| files.into_values().collect())
    }

    /// Java `ConfigTool.getUserTemplateFiles(File)`; the optional second command-line
    /// user-template location remains an EtomoDirector boundary.
    pub fn get_user_template_files(new_user_template_dir: Option<&Path>) -> Option<Vec<PathBuf>> {
        let directory = new_user_template_dir
            .map(Path::to_path_buf)
            .or_else(|| std::env::var_os("IMOD_USER_TEMPLATE_DIR").map(PathBuf::from))
            .or_else(|| {
                std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".etomotemplate"))
            })?;
        let entries = std::fs::read_dir(directory).ok()?;
        let mut files = BTreeMap::new();
        for entry in entries.flatten() {
            let path = entry.path();
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if path.is_file() && name.ends_with(".adoc") {
                files.insert(name.into_owned(), path);
            }
        }
        (!files.is_empty()).then(|| files.into_values().collect())
    }
}

/// Source `DirectiveFile` values consumed by this panel.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TemplateDirectiveFile {
    pub values: BTreeMap<String, String>,
}

impl TemplateDirectiveFile {
    /// Java `contains(DirectiveDef)`.
    pub fn contains(&self, directive: &str) -> bool {
        self.values.contains_key(directive)
    }

    /// Java `getValue(DirectiveDef)`.
    pub fn get_value(&self, directive: &str) -> Option<&str> {
        self.values.get(directive).map(String::as_str)
    }
}

/// Source `DirectiveFileCollection` state reached by TemplatePanel.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TemplateDirectiveFileCollection {
    pub manager_present: bool,
    pub axis_id: AxisID,
    pub batch_instance: bool,
    pub scope: Option<PathBuf>,
    pub system: Option<PathBuf>,
    pub user: Option<PathBuf>,
}

impl TemplateDirectiveFileCollection {
    /// Java `new DirectiveFileCollection(manager, axisID)`.
    pub fn new(axis_id: AxisID) -> Self {
        Self {
            manager_present: true,
            axis_id,
            batch_instance: false,
            scope: None,
            system: None,
            user: None,
        }
    }

    /// Java `DirectiveFileCollection.getBatchInstance(manager, axisID)`.
    pub fn get_batch_instance(axis_id: AxisID) -> Self {
        Self {
            batch_instance: true,
            ..Self::new(axis_id)
        }
    }

    /// Java `setDirectiveFile(File, DirectiveFileType)`.
    pub fn set_directive_file(&mut self, file: Option<PathBuf>, file_type: DirectiveFileType) {
        match file_type {
            DirectiveFileType::Scope => self.scope = file,
            DirectiveFileType::System => self.system = file,
            DirectiveFileType::User => self.user = file,
            DirectiveFileType::BatchDefaults | DirectiveFileType::Batch => {}
        }
    }
}

/// Source `UserConfiguration` subset reached by TemplatePanel.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TemplateUserConfiguration {
    pub scope_template: Option<PathBuf>,
    pub system_template: Option<PathBuf>,
    pub user_template: Option<PathBuf>,
}

impl TemplateUserConfiguration {
    pub fn set_scope_template(&mut self, file: Option<PathBuf>) {
        self.scope_template = file;
    }
    pub fn set_system_template(&mut self, file: Option<PathBuf>) {
        self.system_template = file;
    }
    pub fn set_user_template(&mut self, file: Option<PathBuf>) {
        self.user_template = file;
    }
    pub fn equals_scope_template(&self, file: Option<&Path>) -> bool {
        self.scope_template.as_deref() == file
    }
    pub fn equals_system_template(&self, file: Option<&Path>) -> bool {
        self.system_template.as_deref() == file
    }
    pub fn equals_user_template(&self, file: Option<&Path>) -> bool {
        self.user_template.as_deref() == file
    }
    pub fn is_scope_template_set(&self) -> bool {
        self.scope_template.is_some()
    }
    pub fn is_system_template_set(&self) -> bool {
        self.system_template.is_some()
    }
    pub fn is_user_template_set(&self) -> bool {
        self.user_template.is_some()
    }
    pub fn get_scope_template(&self) -> Option<&Path> {
        self.scope_template.as_deref()
    }
    pub fn get_system_template(&self) -> Option<&Path> {
        self.system_template.as_deref()
    }
    pub fn get_user_template(&self) -> Option<&Path> {
        self.user_template.as_deref()
    }
}

/// Java final `TemplatePanel`.
pub struct TemplatePanel<'a> {
    pub pnl_root: TemplatePanelLayout,
    pub cmb_scope_template: TemplateComboBox,
    pub cmb_system_template: TemplateComboBox,
    pub cmb_user_template: TemplateComboBox,
    pub listener: Option<&'a mut dyn TemplateActionListener>,
    pub manager_present: bool,
    pub axis_id: AxisID,
    pub settings: Option<&'a SettingsDialog>,
    pub directive_file_collection: TemplateDirectiveFileCollection,
    pub draw_border: bool,
    pub scope_template_file_list: Option<Vec<PathBuf>>,
    pub system_template_file_list: Option<Vec<PathBuf>>,
    pub user_template_file_list: Option<Vec<PathBuf>>,
    pub new_user_template_dir: Option<PathBuf>,
    pub actions_active: bool,
}

impl<'a> TemplatePanel<'a> {
    /// Java private `TemplatePanel(...)` constructor.
    pub fn new(
        axis_id: AxisID,
        listener: Option<&'a mut dyn TemplateActionListener>,
        settings: Option<&'a SettingsDialog>,
        draw_border: bool,
        directive_file_collection: Option<TemplateDirectiveFileCollection>,
        batch_interface: bool,
    ) -> Self {
        let directive_file_collection = directive_file_collection.unwrap_or_else(|| {
            if batch_interface {
                TemplateDirectiveFileCollection::get_batch_instance(axis_id)
            } else {
                TemplateDirectiveFileCollection::new(axis_id)
            }
        });
        Self {
            pnl_root: TemplatePanelLayout::default(),
            cmb_scope_template: TemplateComboBox::get_instance("Scope template:"),
            cmb_system_template: TemplateComboBox::get_instance("System template:"),
            cmb_user_template: TemplateComboBox::get_instance("User template:"),
            listener,
            manager_present: true,
            axis_id,
            settings,
            directive_file_collection,
            draw_border,
            scope_template_file_list: None,
            system_template_file_list: None,
            user_template_file_list: None,
            new_user_template_dir: None,
            actions_active: true,
        }
    }

    /// Java static `getInstance(...)`.
    pub fn get_instance(
        axis_id: AxisID,
        listener: Option<&'a mut dyn TemplateActionListener>,
        title: Option<&str>,
        settings: Option<&'a SettingsDialog>,
        batch_interface: bool,
    ) -> Self {
        let mut instance = Self::new(axis_id, listener, settings, true, None, batch_interface);
        instance.create_panel(title);
        instance.add_listeners();
        instance
    }

    /// Java static `getBorderlessInstance(...)`.
    pub fn get_borderless_instance(
        axis_id: AxisID,
        listener: Option<&'a mut dyn TemplateActionListener>,
        title: Option<&str>,
        settings: Option<&'a SettingsDialog>,
        directive_file_collection: Option<TemplateDirectiveFileCollection>,
        delay_listeners: bool,
        batch_interface: bool,
    ) -> Self {
        let mut instance = Self::new(
            axis_id,
            listener,
            settings,
            false,
            directive_file_collection,
            batch_interface,
        );
        instance.create_panel(title);
        if !delay_listeners {
            instance.add_listeners();
        }
        instance
    }

    /// Java private `fillComboBox`.
    pub fn fill_combo_box(combo_box: &mut TemplateComboBox, file_list: Option<&[PathBuf]>) {
        let len = file_list.map_or(0, <[PathBuf]>::len);
        combo_box.set_placeholder(
            EMPTY_OPTION,
            &format!("{NO_SELECTION1}{len}{NO_SELECTION2}"),
        );
        if let Some(file_list) = file_list {
            for file in file_list {
                if let Some(name) = file.file_name().and_then(|name| name.to_str()) {
                    combo_box.add_item(name);
                }
            }
        }
        combo_box.unselect();
    }

    /// Java private `createPanel`.
    pub fn create_panel(&mut self, title: Option<&str>) {
        self.scope_template_file_list = TemplateConfigTool::get_scope_template_files();
        Self::fill_combo_box(
            &mut self.cmb_scope_template,
            self.scope_template_file_list.as_deref(),
        );
        self.system_template_file_list = TemplateConfigTool::get_system_template_files();
        Self::fill_combo_box(
            &mut self.cmb_system_template,
            self.system_template_file_list.as_deref(),
        );
        self.load_user_template();
        self.pnl_root.y_axis_layout = true;
        if self.draw_border {
            self.pnl_root.etched_border = true;
            self.pnl_root.border_title = title.map(str::to_owned);
        }
        self.pnl_root.component_order = vec![
            "x0_y2",
            "cmbScopeTemplate",
            "x0_y3",
            "cmbSystemTemplate",
            "x0_y3",
            "cmbUserTemplate",
            "x0_y2",
        ];
    }

    /// Java `addListeners`.
    pub fn add_listeners(&mut self) {
        self.add_action_listener();
        self.cmb_user_template.focus_listener_count += 1;
    }

    /// Java `getComponent` GUI boundary.
    pub fn get_component(&self) -> &TemplatePanelLayout {
        &self.pnl_root
    }

    /// Java `addActionListener(ActionListener)`.
    pub fn add_action_listener(&mut self) {
        self.cmb_scope_template.action_listener_count += 1;
        self.cmb_system_template.action_listener_count += 1;
        self.cmb_user_template.action_listener_count += 1;
    }

    /// Java `setEnabled`.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.cmb_scope_template.set_enabled(enabled);
        self.cmb_system_template.set_enabled(enabled);
        self.cmb_user_template.set_enabled(enabled);
    }
    /// Java `setEditable`.
    pub fn set_editable(&mut self, editable: bool) {
        self.cmb_scope_template.set_editable(editable);
        self.cmb_system_template.set_editable(editable);
        self.cmb_user_template.set_editable(editable);
    }

    /// Java `saveAutodoc`.
    pub fn save_autodoc(&self, autodoc: &mut dyn WritableAutodoc, validate_only: bool) {
        if validate_only {
            return;
        }
        for (directive, file) in [
            ("scopeTemplate", self.get_scope_template_file()),
            ("systemTemplate", self.get_system_template_file()),
            ("userTemplate", self.get_user_template_file()),
        ] {
            if let Some(file) = file {
                unsafe {
                    autodoc.add_name_value_pair_attribute(Some(directive), file.to_str());
                }
            }
        }
    }

    /// Java `setFieldHighlight`.
    pub fn set_field_highlight(&mut self) {
        self.cmb_scope_template.set_field_highlight();
        self.cmb_system_template.set_field_highlight();
        self.cmb_user_template.set_field_highlight();
    }

    /// Java private `loadUserTemplate`.
    pub fn load_user_template(&mut self) {
        self.user_template_file_list = None;
        self.cmb_user_template.remove_all_items();
        self.user_template_file_list =
            TemplateConfigTool::get_user_template_files(self.new_user_template_dir.as_deref());
        Self::fill_combo_box(
            &mut self.cmb_user_template,
            self.user_template_file_list.as_deref(),
        );
    }

    /// Java private `getTemplateFile`.
    pub fn get_template_file(
        combo_box: &TemplateComboBox,
        file_list: Option<&[PathBuf]>,
    ) -> Option<PathBuf> {
        if combo_box.is_enabled() && combo_box.get_selected_index() != -1 {
            return file_list
                .and_then(|files| files.get(combo_box.get_selected_index() as usize))
                .cloned();
        }
        None
    }

    /// Java private `selectTemplate`.
    pub fn select_template(
        template: Option<&Path>,
        template_file_list: Option<&[PathBuf]>,
        combo_box: &mut TemplateComboBox,
    ) {
        let (Some(template), Some(files)) = (template, template_file_list) else {
            return;
        };
        let absolute_path = template.components().count() > 1;
        for (index, file) in files.iter().enumerate() {
            if (absolute_path && file == template)
                || (!absolute_path && file.file_name() == template.file_name())
            {
                combo_box.set_selected_index(index as isize);
                break;
            }
        }
    }

    /// Java `clear`.
    pub fn clear(&mut self) {
        self.cmb_scope_template.unselect();
        self.cmb_system_template.unselect();
        self.cmb_user_template.unselect();
    }
    /// Java `activateActions`.
    pub fn activate_actions(&mut self, input: bool) {
        self.actions_active = input;
    }
    /// Java `equalsActionCommand`.
    pub fn equals_action_command(&self, action_command: &str) -> bool {
        self.actions_active
            && (action_command == self.cmb_scope_template.get_action_command()
                || action_command == self.cmb_system_template.get_action_command()
                || action_command == self.cmb_user_template.get_action_command())
    }

    /// Java `getParameters(UserConfiguration)`.
    pub fn get_parameters(&self, user_config: &mut TemplateUserConfiguration) {
        user_config.set_scope_template(self.get_scope_template_file());
        user_config.set_system_template(self.get_system_template_file());
        user_config.set_user_template(self.get_user_template_file());
    }
    /// Java `getDirectiveFileCollection`.
    pub fn get_directive_file_collection(&mut self) -> &TemplateDirectiveFileCollection {
        self.refresh_directive_file_collection();
        &self.directive_file_collection
    }
    /// Java `refreshDirectiveFileCollection`.
    pub fn refresh_directive_file_collection(&mut self) {
        self.directive_file_collection
            .set_directive_file(self.get_scope_template_file(), DirectiveFileType::Scope);
        self.directive_file_collection
            .set_directive_file(self.get_system_template_file(), DirectiveFileType::System);
        self.directive_file_collection
            .set_directive_file(self.get_user_template_file(), DirectiveFileType::User);
    }
    /// Java `getFiles`.
    pub fn get_files(&self) -> [Option<PathBuf>; NUM_TEMPLATES] {
        [
            self.get_scope_template_file(),
            self.get_system_template_file(),
            self.get_user_template_file(),
        ]
    }
    /// Java private `getScopeTemplateFile`.
    pub fn get_scope_template_file(&self) -> Option<PathBuf> {
        Self::get_template_file(
            &self.cmb_scope_template,
            self.scope_template_file_list.as_deref(),
        )
    }
    /// Java private `getSystemTemplateFile`.
    pub fn get_system_template_file(&self) -> Option<PathBuf> {
        Self::get_template_file(
            &self.cmb_system_template,
            self.system_template_file_list.as_deref(),
        )
    }
    /// Java private `getUserTemplateFile`.
    pub fn get_user_template_file(&self) -> Option<PathBuf> {
        Self::get_template_file(
            &self.cmb_user_template,
            self.user_template_file_list.as_deref(),
        )
    }
    /// Java private `focusGained`.
    pub fn focus_gained(&mut self) {
        self.reload_user_template();
    }

    /// Java private `reloadUserTemplate`.
    pub fn reload_user_template(&mut self) {
        if let Some(settings) = self.settings {
            if !settings.equals_user_template_dir(self.new_user_template_dir.as_deref()) {
                self.new_user_template_dir =
                    settings.get_user_template_dir().map(Path::to_path_buf);
                self.load_user_template();
            }
        }
    }

    /// Java `isAppearanceSettingChanged`.
    pub fn is_appearance_setting_changed(&self, user_config: &TemplateUserConfiguration) -> bool {
        !user_config.equals_scope_template(self.get_scope_template_file().as_deref())
            || !user_config.equals_system_template(self.get_system_template_file().as_deref())
            || !user_config.equals_user_template(self.get_user_template_file().as_deref())
    }

    /// Java overloaded `setParameters(UserConfiguration)`.
    pub fn set_parameters_user_configuration(&mut self, user_config: &TemplateUserConfiguration) {
        if user_config.is_scope_template_set() {
            Self::select_template(
                user_config.get_scope_template(),
                self.scope_template_file_list.as_deref(),
                &mut self.cmb_scope_template,
            );
        }
        if user_config.is_system_template_set() {
            Self::select_template(
                user_config.get_system_template(),
                self.system_template_file_list.as_deref(),
                &mut self.cmb_system_template,
            );
        }
        if user_config.is_user_template_set() {
            Self::select_template(
                user_config.get_user_template(),
                self.user_template_file_list.as_deref(),
                &mut self.cmb_user_template,
            );
        }
    }

    /// Java overloaded `setParameters(DirectiveFile)`.
    pub fn set_parameters_directive_file(
        &mut self,
        directive_file: Option<&TemplateDirectiveFile>,
    ) {
        let Some(directive_file) = directive_file else {
            return;
        };
        if directive_file.contains("scopeTemplate") {
            Self::select_template(
                directive_file.get_value("scopeTemplate").map(Path::new),
                self.scope_template_file_list.as_deref(),
                &mut self.cmb_scope_template,
            );
        } else {
            self.cmb_scope_template.unselect();
        }
        if directive_file.contains("systemTemplate") {
            Self::select_template(
                directive_file.get_value("systemTemplate").map(Path::new),
                self.system_template_file_list.as_deref(),
                &mut self.cmb_system_template,
            );
        } else {
            self.cmb_system_template.unselect();
        }
        if directive_file.contains("userTemplate") {
            self.reload_user_template();
            Self::select_template(
                directive_file.get_value("userTemplate").map(Path::new),
                self.user_template_file_list.as_deref(),
                &mut self.cmb_user_template,
            );
        } else {
            self.cmb_user_template.unselect();
        }
        self.refresh_directive_file_collection();
    }

    /// Java `setScopeTooltip`.
    pub fn set_scope_tooltip(&mut self, tooltip_text: Option<&str>) {
        self.cmb_scope_template.set_tool_tip_text(tooltip_text);
    }
    /// Java `setSystemTooltip`.
    pub fn set_system_tooltip(&mut self, tooltip_text: Option<&str>) {
        self.cmb_system_template.set_tool_tip_text(tooltip_text);
    }
    /// Java `setUserTooltip`.
    pub fn set_user_tooltip(&mut self, tooltip_text: Option<&str>) {
        self.cmb_user_template.set_tool_tip_text(tooltip_text);
    }
}

/// Java private static `TemplateFocusListener`.  Focus dispatch is retained by
/// `TemplatePanel::focus_gained`; Swing event ownership is a GUI boundary.
pub struct TemplateFocusListener;
impl TemplateFocusListener {
    /// Java `focusGained`.
    pub fn focus_gained(panel: &mut TemplatePanel<'_>) {
        panel.focus_gained();
    }
    /// Java `focusLost`.
    pub fn focus_lost(&self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    fn panel() -> TemplatePanel<'static> {
        let mut panel = TemplatePanel::new(AxisID::Only, None, None, false, None, false);
        panel.scope_template_file_list = Some(vec![PathBuf::from("/scope/a.adoc")]);
        panel.system_template_file_list = Some(vec![PathBuf::from("/system/b.adoc")]);
        panel.user_template_file_list = Some(vec![PathBuf::from("/user/c.adoc")]);
        TemplatePanel::fill_combo_box(
            &mut panel.cmb_scope_template,
            panel.scope_template_file_list.as_deref(),
        );
        TemplatePanel::fill_combo_box(
            &mut panel.cmb_system_template,
            panel.system_template_file_list.as_deref(),
        );
        TemplatePanel::fill_combo_box(
            &mut panel.cmb_user_template,
            panel.user_template_file_list.as_deref(),
        );
        panel
    }

    #[test]
    fn set_parameters_selects_file_names_and_refreshes_collection() {
        let mut panel = panel();
        let mut directive_file = TemplateDirectiveFile::default();
        directive_file
            .values
            .insert("scopeTemplate".into(), "a.adoc".into());
        directive_file
            .values
            .insert("systemTemplate".into(), "/system/b.adoc".into());
        directive_file
            .values
            .insert("userTemplate".into(), "c.adoc".into());
        panel.set_parameters_directive_file(Some(&directive_file));
        assert_eq!(
            panel.get_files(),
            [
                Some(PathBuf::from("/scope/a.adoc")),
                Some(PathBuf::from("/system/b.adoc")),
                Some(PathBuf::from("/user/c.adoc"))
            ]
        );
        assert_eq!(
            panel.directive_file_collection.system,
            Some(PathBuf::from("/system/b.adoc"))
        );
    }

    #[test]
    fn disabled_templates_are_excluded_and_action_gate_is_honored() {
        let mut panel = panel();
        panel.cmb_scope_template.set_selected_index(0);
        panel.cmb_scope_template.set_enabled(false);
        assert_eq!(panel.get_scope_template_file(), None);
        assert!(panel.equals_action_command("Scope template:"));
        panel.activate_actions(false);
        assert!(!panel.equals_action_command("Scope template:"));
    }

    #[test]
    fn borderless_constructor_honors_listener_delay() {
        let panel = TemplatePanel::get_borderless_instance(
            AxisID::First,
            None,
            None,
            None,
            None,
            true,
            false,
        );
        assert!(!panel.pnl_root.etched_border);
        assert_eq!(panel.cmb_user_template.focus_listener_count, 0);
    }
}
