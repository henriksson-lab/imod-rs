//! `IMOD/Etomo/src/etomo/ui/swing/TemplatePanel.java`.
//!
//! The scope/system/user template combo boxes of the setup and batch dialogs.
//! Box layout and borders are Swing layout (recorded as comments); the combo
//! boxes' names and contents are kept.

use std::cell::{Cell, RefCell};
use std::path::{Path, PathBuf};
use std::rc::{Rc, Weak};

use super::combo_box::ComboBox;
use super::settings_dialog::SettingsDialog;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::jdk::{ActionListener, FocusEvent, JComponent};
use crate::imod::etomo::logic::config_tool;
use crate::imod::etomo::storage::autodoc::writable_autodoc::WritableAutodoc;
use crate::imod::etomo::storage::directive_def::DirectiveDef;
use crate::imod::etomo::storage::directive_file::DirectiveFile;
use crate::imod::etomo::storage::directive_file_collection::DirectiveFileCollection;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::directive_file_type::DirectiveFileType;
use crate::imod::etomo::r#type::user_configuration::UserConfiguration;
use crate::imod::etomo::util::utilities;

/// Java private `EMPTY_OPTION`.
const EMPTY_OPTION: &str = "None available";
/// Java private `NO_SELECTION1`.
const NO_SELECTION1: &str = "No selection (";
/// Java private `NO_SELECTION2`.
const NO_SELECTION2: &str = " available)";
/// Java private `NUM_TEMPLATES`.
const NUM_TEMPLATES: usize = 3;

/// Java `TemplatePanel`.
pub struct TemplatePanel {
    /// Java `pnlRoot`.
    pnl_root: Rc<JComponent>,
    /// Java `cmbScopeTemplate`.
    cmb_scope_template: Rc<ComboBox>,
    /// Java `cmbSystemTemplate`.
    cmb_system_template: Rc<ComboBox>,
    /// Java `cmbUserTemplate`.
    cmb_user_template: Rc<ComboBox>,
    /// Java `listener` (a `TemplateActionListener`, which is an `ActionListener`).
    listener: ActionListener,
    /// Java `manager`.
    manager: &'static dyn BaseManager,
    /// Java `axisID`.
    axis_id: AxisID,
    /// Java package-private `settings`.
    pub settings: Option<Rc<SettingsDialog>>,
    /// Java `directiveFileCollection`.
    directive_file_collection: Rc<RefCell<DirectiveFileCollection>>,
    /// Java `drawBorder`.
    draw_border: bool,
    /// Java `scopeTemplateFileList`.
    scope_template_file_list: RefCell<Option<Vec<PathBuf>>>,
    /// Java `systemTemplateFileList`.
    system_template_file_list: RefCell<Option<Vec<PathBuf>>>,
    /// Java `userTemplateFileList`.
    user_template_file_list: RefCell<Option<Vec<PathBuf>>>,
    /// Java `newUserTemplateDir`.
    new_user_template_dir: RefCell<Option<PathBuf>>,
    /// Java `actionsActive`.
    actions_active: Cell<bool>,
    this: Weak<TemplatePanel>,
}

impl TemplatePanel {
    /// Java private `TemplatePanel(BaseManager, AxisID, TemplateActionListener,
    /// SettingsDialog, boolean, DirectiveFileCollection, boolean)`.
    fn new(
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        listener: ActionListener,
        settings: Option<Rc<SettingsDialog>>,
        draw_border: bool,
        directive_file_collection: Option<Rc<RefCell<DirectiveFileCollection>>>,
        batch_interface: bool,
    ) -> Rc<TemplatePanel> {
        let directive_file_collection = match directive_file_collection {
            None => {
                if batch_interface {
                    Rc::new(RefCell::new(DirectiveFileCollection::get_batch_instance(
                        manager,
                        Some(axis_id),
                    )))
                } else {
                    Rc::new(RefCell::new(DirectiveFileCollection::new(
                        manager,
                        Some(axis_id),
                    )))
                }
            }
            Some(directive_file_collection) => directive_file_collection,
        };
        Rc::new_cyclic(|this| TemplatePanel {
            pnl_root: JComponent::new_panel(),
            cmb_scope_template: ComboBox::get_instance(Some("Scope template:")),
            cmb_system_template: ComboBox::get_instance(Some("System template:")),
            cmb_user_template: ComboBox::get_instance(Some("User template:")),
            listener,
            manager,
            axis_id,
            settings,
            directive_file_collection,
            draw_border,
            scope_template_file_list: RefCell::new(None),
            system_template_file_list: RefCell::new(None),
            user_template_file_list: RefCell::new(None),
            new_user_template_dir: RefCell::new(None),
            actions_active: Cell::new(true),
            this: this.clone(),
        })
    }

    /// Java `getInstance(BaseManager, AxisID, TemplateActionListener, String,
    /// SettingsDialog, boolean)`.
    pub fn get_instance(
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        listener: ActionListener,
        title: Option<&str>,
        settings: Option<Rc<SettingsDialog>>,
        batch_interface: bool,
    ) -> Rc<TemplatePanel> {
        let instance = Self::new(
            manager,
            axis_id,
            listener,
            settings,
            true,
            None,
            batch_interface,
        );
        instance.create_panel(title);
        instance.add_listeners();
        instance
    }

    /// Java `getBorderlessInstance(BaseManager, AxisID, TemplateActionListener,
    /// String, SettingsDialog, DirectiveFileCollection, boolean, boolean)`.
    #[allow(clippy::too_many_arguments)]
    pub fn get_borderless_instance(
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        listener: ActionListener,
        title: Option<&str>,
        settings: Option<Rc<SettingsDialog>>,
        directive_file_collection: Option<Rc<RefCell<DirectiveFileCollection>>>,
        delay_listeners: bool,
        batch_interface: bool,
    ) -> Rc<TemplatePanel> {
        let instance = Self::new(
            manager,
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

    /// Java private `fillComboBox(ComboBox, File[])`.
    fn fill_combo_box(combo_box: &ComboBox, file_list: Option<&[PathBuf]>) {
        let len = file_list.map_or(0, |file_list| file_list.len());
        combo_box.set_placeholder(
            Some(EMPTY_OPTION),
            Some(&format!("{NO_SELECTION1}{len}{NO_SELECTION2}")),
        );
        if len > 0 {
            for file in file_list.unwrap() {
                combo_box.add_item(Some(&utilities::java_io_file_get_name(
                    &file.to_string_lossy(),
                )));
            }
        }
        combo_box.unselect();
    }

    /// Java private `createPanel(String)`.
    fn create_panel(&self, title: Option<&str>) {
        // init
        *self.scope_template_file_list.borrow_mut() = config_tool::get_scope_template_files();
        Self::fill_combo_box(
            &self.cmb_scope_template,
            self.scope_template_file_list.borrow().as_deref(),
        );
        *self.system_template_file_list.borrow_mut() = config_tool::get_system_template_files();
        Self::fill_combo_box(
            &self.cmb_system_template,
            self.system_template_file_list.borrow().as_deref(),
        );
        self.load_user_template();
        // Swing layout: pnlRoot BoxLayout Y_AXIS.
        if self.draw_border {
            if let Some(title) = title {
                // `new EtchedBorder(title).getBorder()`: a titled border.
                self.pnl_root.set_border_title(Some(title));
            } else {
                // Swing painting: an untitled etched border.
            }
        }
        // Swing layout: rigid areas between the combo boxes.
        self.pnl_root.add(&self.cmb_scope_template.get_component());
        self.pnl_root.add(&self.cmb_system_template.get_component());
        self.pnl_root.add(&self.cmb_user_template.get_component());
    }

    /// Java `addListeners()`.
    pub fn add_listeners(&self) {
        self.add_action_listener(self.listener.clone());
        // Java `cmbUserTemplate.addFocusListener(new TemplateFocusListener(this))`:
        // `TemplateFocusListener` forwards focusGained and ignores focusLost.
        let template = self.this.clone();
        self.cmb_user_template
            .add_focus_listener(Rc::new(move |event: &FocusEvent| {
                if event.gained
                    && let Some(template) = template.upgrade()
                {
                    template.focus_gained();
                }
            }));
    }

    /// Java `getComponent()`.
    pub fn get_component(&self) -> Rc<JComponent> {
        self.pnl_root.clone()
    }

    /// Java `addActionListener(ActionListener)`.
    pub fn add_action_listener(&self, listener: ActionListener) {
        self.cmb_scope_template
            .add_action_listener(listener.clone());
        self.cmb_system_template
            .add_action_listener(listener.clone());
        self.cmb_user_template.add_action_listener(listener);
    }

    /// Java `setEnabled(boolean)`.
    pub fn set_enabled(&self, enabled: bool) {
        self.cmb_scope_template.set_enabled(enabled);
        self.cmb_system_template.set_enabled(enabled);
        self.cmb_user_template.set_enabled(enabled);
    }

    /// Java `setEditable(boolean)`.
    pub fn set_editable(&self, editable: bool) {
        self.cmb_scope_template.set_editable(editable);
        self.cmb_system_template.set_editable(editable);
        self.cmb_user_template.set_editable(editable);
    }

    /// Java `saveAutodoc(WritableAutodoc, boolean)`.
    pub fn save_autodoc(&self, autodoc: &mut dyn WritableAutodoc, validate_only: bool) {
        if validate_only {
            return;
        }
        let mut template_file = self.get_scope_template_file();
        if let Some(file) = &template_file {
            autodoc.add_name_value_pair_attribute(
                Some(&DirectiveDef::SCOPE_TEMPLATE.get_directive()),
                Some(&utilities::java_io_file_get_absolute_path(
                    &file.to_string_lossy(),
                )),
            );
        }
        template_file = self.get_system_template_file();
        if let Some(file) = &template_file {
            autodoc.add_name_value_pair_attribute(
                Some(&DirectiveDef::SYSTEM_TEMPLATE.get_directive()),
                Some(&utilities::java_io_file_get_absolute_path(
                    &file.to_string_lossy(),
                )),
            );
        }
        template_file = self.get_user_template_file();
        if let Some(file) = &template_file {
            autodoc.add_name_value_pair_attribute(
                Some(&DirectiveDef::USER_TEMPLATE.get_directive()),
                Some(&utilities::java_io_file_get_absolute_path(
                    &file.to_string_lossy(),
                )),
            );
        }
    }

    /// Java `setFieldHighlight()`.
    pub fn set_field_highlight(&self) {
        self.cmb_scope_template.set_field_highlight();
        self.cmb_system_template.set_field_highlight();
        self.cmb_user_template.set_field_highlight();
    }

    /// Java private `loadUserTemplate()`.
    fn load_user_template(&self) {
        *self.user_template_file_list.borrow_mut() = None;
        self.cmb_user_template.remove_all_items();
        // If the user template directory is in a different directory from the
        // location of the default user template, the user template will not be
        // loaded.
        let new_user_template_dir = self.new_user_template_dir.borrow().clone();
        *self.user_template_file_list.borrow_mut() =
            config_tool::get_user_template_files(new_user_template_dir.as_deref());
        Self::fill_combo_box(
            &self.cmb_user_template,
            self.user_template_file_list.borrow().as_deref(),
        );
    }

    /// Java private `getTemplateFile(ComboBox, File[])`.
    fn get_template_file(
        cmb_template: &ComboBox,
        template_file_list: Option<&[PathBuf]>,
    ) -> Option<PathBuf> {
        if cmb_template.is_enabled() {
            let i = cmb_template.get_selected_index();
            if i != -1
                && let Some(template_file_list) = template_file_list
            {
                return template_file_list.get(i as usize).cloned();
            }
        }
        None
    }

    /// Java private `selectTemplate(String, File[], ComboBox)`.
    fn select_template(
        template: Option<&str>,
        template_file_list: Option<&[PathBuf]>,
        cmb_template: &ComboBox,
    ) {
        let (Some(template_file_list), Some(template)) = (template_file_list, template) else {
            return;
        };
        let abs_path = template.contains('/');
        // If template doesn't match something in templateFileList, nothing will be
        // selected in the combobox.
        for (i, file) in template_file_list.iter().enumerate() {
            let path = file.to_string_lossy();
            if (abs_path && utilities::java_io_file_get_absolute_path(&path) == template)
                || (!abs_path && utilities::java_io_file_get_name(&path) == template)
            {
                cmb_template.set_selected_index(i as i32);
                break;
            }
        }
    }

    /// Java `clear()`.
    pub fn clear(&self) {
        self.cmb_scope_template.unselect();
        self.cmb_system_template.unselect();
        self.cmb_user_template.unselect();
    }

    /// Java `activateActions(boolean)`.
    pub fn activate_actions(&self, input: bool) {
        self.actions_active.set(input);
    }

    /// Java `equalsActionCommand(String)`.
    pub fn equals_action_command(&self, action_command: Option<&str>) -> bool {
        if !self.actions_active.get() {
            return false;
        }
        // Java dereferences the command; a null command matches nothing here.
        let Some(action_command) = action_command else {
            return false;
        };
        Some(action_command) == self.cmb_scope_template.get_action_command().as_deref()
            || Some(action_command) == self.cmb_system_template.get_action_command().as_deref()
            || Some(action_command) == self.cmb_user_template.get_action_command().as_deref()
    }

    /// Java `getParameters(UserConfiguration)`.
    pub fn get_parameters(&self, user_config: &mut UserConfiguration) {
        user_config.set_scope_template(self.get_scope_template_file().as_deref());
        user_config.set_system_template(self.get_system_template_file().as_deref());
        user_config.set_user_template(self.get_user_template_file().as_deref());
    }

    /// Java `getDirectiveFileCollection()`.
    pub fn get_directive_file_collection(&self) -> Option<Rc<RefCell<DirectiveFileCollection>>> {
        self.refresh_directive_file_collection();
        Some(self.directive_file_collection.clone())
    }

    /// Java `refreshDirectiveFileCollection()`.
    pub fn refresh_directive_file_collection(&self) {
        let scope = self.get_scope_template_file();
        let system = self.get_system_template_file();
        let user = self.get_user_template_file();
        let mut collection = self.directive_file_collection.borrow_mut();
        collection.set_directive_file(scope.as_deref(), DirectiveFileType::Scope);
        collection.set_directive_file(system.as_deref(), DirectiveFileType::System);
        collection.set_directive_file(user.as_deref(), DirectiveFileType::User);
    }

    /// Java `getFiles()`.
    pub fn get_files(&self) -> [Option<PathBuf>; NUM_TEMPLATES] {
        [
            self.get_scope_template_file(),
            self.get_system_template_file(),
            self.get_user_template_file(),
        ]
    }

    /// Java private `getScopeTemplateFile()`.
    fn get_scope_template_file(&self) -> Option<PathBuf> {
        Self::get_template_file(
            &self.cmb_scope_template,
            self.scope_template_file_list.borrow().as_deref(),
        )
    }

    /// Java private `getSystemTemplateFile()`.
    fn get_system_template_file(&self) -> Option<PathBuf> {
        Self::get_template_file(
            &self.cmb_system_template,
            self.system_template_file_list.borrow().as_deref(),
        )
    }

    /// Java private `getUserTemplateFile()`.
    fn get_user_template_file(&self) -> Option<PathBuf> {
        Self::get_template_file(
            &self.cmb_user_template,
            self.user_template_file_list.borrow().as_deref(),
        )
    }

    /// Java private `focusGained()`.  Only need to listen to user template
    /// combobox.
    pub fn focus_gained(&self) {
        self.reload_user_template();
    }

    /// Java private `reloadUserTemplate()`.
    fn reload_user_template(&self) {
        let new_dir = self.new_user_template_dir.borrow().clone();
        if let Some(settings) = &self.settings
            && !settings.equals_user_template_dir(new_dir.as_deref())
        {
            // If a new user template directory has been entered, reload the user
            // template combo box.
            *self.new_user_template_dir.borrow_mut() = settings.get_user_template_dir();
            self.load_user_template();
        }
    }

    /// Java `isAppearanceSettingChanged(UserConfiguration)`.
    pub fn is_appearance_setting_changed(&self, user_config: &UserConfiguration) -> bool {
        !user_config.equals_scope_template(self.get_scope_template_file().as_deref())
            || !user_config.equals_system_template(self.get_system_template_file().as_deref())
            || !user_config.equals_user_template(self.get_user_template_file().as_deref())
    }

    /// Java `setParameters(UserConfiguration)`.
    pub fn set_parameters_user_configuration(&self, user_config: &UserConfiguration) {
        if user_config.is_scope_template_set() {
            Self::select_template(
                user_config.get_scope_template().as_deref(),
                self.scope_template_file_list.borrow().as_deref(),
                &self.cmb_scope_template,
            );
        }
        if user_config.is_system_template_set() {
            Self::select_template(
                user_config.get_system_template().as_deref(),
                self.system_template_file_list.borrow().as_deref(),
                &self.cmb_system_template,
            );
        }
        if user_config.is_user_template_set() {
            Self::select_template(
                user_config.get_user_template().as_deref(),
                self.user_template_file_list.borrow().as_deref(),
                &self.cmb_user_template,
            );
        }
    }

    /// Java `setParameters(DirectiveFile)`.
    pub fn set_parameters_directive_file(&self, directive_file: Option<&DirectiveFile>) {
        let Some(directive_file) = directive_file else {
            return;
        };
        let mut directive_def = DirectiveDef::SCOPE_TEMPLATE;
        if directive_file.contains(Some(directive_def)) {
            Self::select_template(
                directive_file.get_value(Some(directive_def)).as_deref(),
                self.scope_template_file_list.borrow().as_deref(),
                &self.cmb_scope_template,
            );
        } else {
            self.cmb_scope_template.unselect();
        }
        directive_def = DirectiveDef::SYSTEM_TEMPLATE;
        if directive_file.contains(Some(directive_def)) {
            Self::select_template(
                directive_file.get_value(Some(directive_def)).as_deref(),
                self.system_template_file_list.borrow().as_deref(),
                &self.cmb_system_template,
            );
        } else {
            self.cmb_system_template.unselect();
        }
        directive_def = DirectiveDef::USER_TEMPLATE;
        if directive_file.contains(Some(directive_def)) {
            self.reload_user_template();
            Self::select_template(
                directive_file.get_value(Some(directive_def)).as_deref(),
                self.user_template_file_list.borrow().as_deref(),
                &self.cmb_user_template,
            );
        } else {
            self.cmb_user_template.unselect();
        }
        self.refresh_directive_file_collection();
    }

    /// Java `setScopeTooltip(String)`.
    pub fn set_scope_tooltip(&self, tooltip_text: Option<&str>) {
        self.cmb_scope_template.set_tool_tip_text(tooltip_text);
    }

    /// Java `setSystemTooltip(String)`.
    pub fn set_system_tooltip(&self, tooltip_text: Option<&str>) {
        self.cmb_system_template.set_tool_tip_text(tooltip_text);
    }

    /// Java `setUserTooltip(String)`.
    pub fn set_user_tooltip(&self, tooltip_text: Option<&str>) {
        self.cmb_user_template.set_tool_tip_text(tooltip_text);
    }
}
