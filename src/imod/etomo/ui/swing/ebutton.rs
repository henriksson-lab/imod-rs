//! `IMOD/Etomo/src/etomo/ui/swing/Ebutton.java`.
//!
//! `JButton`, `GridBagLayout`, button styles, tooltip formatting, and file
//! chooser presentation are native GUI boundaries.  This module retains the
//! source-owned construction, state transitions, control mediation, and
//! listener-delivery order without replacing Swing's file chooser or painter.
#![allow(dead_code)]

use std::{cell::RefCell, path::PathBuf, rc::Rc};

use crate::imod::etomo::{
    etomo_director::ARGUMENTS,
    storage::autodoc::autodoc_tokenizer::{DEFAULT_DELIMITER, SEPARATOR_CHAR},
    ui::browsing_directory::BrowsingDirectory,
    util::utilities,
};

use super::{
    appearance_extension::{AppearanceExtension, ComponentBoundary, FlagDisplay, FlagType},
    button_component::ActionListenerBoundary,
    button_style_extension::ButtonStyleExtensionBoundary,
    control_mediator::ControlMediator,
    control_mode::{CLEAR, ControlMode, SELECT_FILE, SELECT_MULTIPLE_FILES},
    control_target::ControlTarget,
    controller::Controller,
    file_text_field_interface::FileFilter,
    grid_bag_extension::GridBagExtension,
    select_file_extension::SelectFileExtension,
};

/// Java `JButton` state read or changed by this source unit.
#[derive(Clone, Debug)]
pub struct JButtonBoundary {
    pub component: Rc<RefCell<ComponentBoundary>>,
    pub name: Option<String>,
    pub text: Option<String>,
    pub preferred_size: Option<(i32, i32)>,
    pub maximum_size: Option<(i32, i32)>,
    pub empty_border: bool,
    pub horizontal_alignment: Option<i32>,
    pub visible: bool,
    pub action_command: Option<String>,
    pub tooltip: Option<String>,
    pub action_listener_registered: bool,
}

impl Default for JButtonBoundary {
    fn default() -> Self {
        Self {
            component: Rc::new(RefCell::new(ComponentBoundary::default())),
            name: None,
            text: None,
            preferred_size: None,
            maximum_size: None,
            empty_border: false,
            horizontal_alignment: None,
            visible: true,
            action_command: None,
            tooltip: None,
            action_listener_registered: false,
        }
    }
}

/// Java package-private final `Ebutton`.
pub struct Ebutton {
    /// Java final `button`.
    pub button: JButtonBoundary,
    /// Java final `target`.
    pub target: Option<Rc<RefCell<dyn ControlTarget>>>,
    /// Java final `controlMode`.
    pub control_mode: Option<&'static ControlMode>,
    /// Java final `selectFileExtension`.
    pub select_file_extension: Option<Rc<RefCell<SelectFileExtension>>>,
    /// Java final `implementToggle`.
    pub implement_toggle: bool,
    /// Java `buttonStyle`.
    pub button_style: Option<ButtonStyleExtensionBoundary>,
    /// Java `gridBagExtension`.
    pub grid_bag_extension: Option<GridBagExtension>,
    /// Java `selected`.
    pub selected: bool,
    /// Java `appearanceExtension`.
    pub appearance_extension: Option<AppearanceExtension>,
    /// Java `actionListeners`.
    pub action_listeners: Option<Vec<Rc<RefCell<dyn ActionListenerBoundary>>>>,
    /// Java `allowFlagEditableControl`.
    pub allow_flag_editable_control: bool,
    /// Java `respondToNonErrorFlags`.
    pub respond_to_non_error_flags: bool,
    /// Java `flagType`.
    pub flag_type: Option<FlagType>,
    /// Java `buttonActionListening`.
    pub button_action_listening: bool,
    /// Java `overrideSelectFileDirectory`.
    pub override_select_file_directory: Option<PathBuf>,
    /// Java `debug`.
    pub debug: bool,
}

impl Ebutton {
    /// Java private `Ebutton(ControlMode, ControlTarget, String, Dimension, Border, boolean,
    /// SelectFileExtension, boolean)`.
    #[allow(clippy::too_many_arguments)]
    fn new(
        control_mode: Option<&'static ControlMode>,
        target: Option<Rc<RefCell<dyn ControlTarget>>>,
        label: Option<&str>,
        fixed_size: Option<(i32, i32)>,
        empty_border: bool,
        implement_toggle: bool,
        shared_select_file_extension: Option<Rc<RefCell<SelectFileExtension>>>,
        debug: bool,
    ) -> Self {
        let mut instance = Self {
            button: JButtonBoundary::default(),
            target,
            control_mode,
            select_file_extension: None,
            implement_toggle,
            button_style: None,
            grid_bag_extension: None,
            selected: false,
            appearance_extension: None,
            action_listeners: None,
            allow_flag_editable_control: true,
            respond_to_non_error_flags: true,
            flag_type: None,
            button_action_listening: false,
            override_select_file_directory: None,
            debug,
        };
        instance.button.text = label.map(str::to_owned);
        instance.set_name(label);
        if let Some(fixed_size) = fixed_size {
            instance.button.preferred_size = Some(fixed_size);
            instance.button.maximum_size = Some(fixed_size);
        }
        instance.button.empty_border = empty_border;
        if control_mode.is_some_and(|mode| std::ptr::eq(mode, &*SELECT_FILE))
            || control_mode.is_some_and(|mode| std::ptr::eq(mode, &*SELECT_MULTIPLE_FILES))
        {
            instance.select_file_extension = Some(
                shared_select_file_extension
                    .unwrap_or_else(|| Rc::new(RefCell::new(SelectFileExtension::new()))),
            );
        }
        instance
    }

    /// Java private `setButtonStyle(ButtonStyleExtension, String)`.
    fn set_button_style(&mut self, style_name: &'static str, label: Option<&str>) {
        let mut button_style = ButtonStyleExtensionBoundary::new(style_name);
        button_style.setup(label, self.debug);
        self.button_style = Some(button_style);
    }

    /// Java `setDebug(boolean)`.
    pub fn set_debug(&mut self, debug: bool) {
        self.debug = debug;
    }

    /// Java `toString()`.
    pub fn to_string(&self) -> String {
        format!(
            "[{},{}]",
            self.button.name.as_deref().unwrap_or("null"),
            self.button.text.as_deref().unwrap_or("null")
        )
    }

    /// Java private `setName(String, ControlTarget, ControlMode)`.
    fn set_name(&mut self, label: Option<&str>) {
        let mut name = None;
        let mut field_type = "bn";
        if let Some(label) = label {
            name = self.append_to_name(name, utilities::convert_label_to_name(Some(label), true));
        } else if let Some(control_mode) = self.control_mode {
            if control_mode.has_field_name() {
                field_type = "ctb";
            }
            if let Some(target) = &self.target {
                name = utilities::convert_label_to_name(Some(&target.borrow().get_label()), true);
            }
            name = control_mode.append_to_name(name);
        }
        if let Some(name) = name {
            self.button.name = Some(format!("{field_type}{SEPARATOR_CHAR}{name}"));
            if ARGUMENTS.lock().unwrap().is_print_names() {
                println!(
                    "{} {DEFAULT_DELIMITER} ",
                    self.button.name.as_deref().expect("assigned above")
                );
            }
        }
    }

    /// Java private `appendToName(String, String)`.
    fn append_to_name(&self, name: Option<String>, append_name: Option<String>) -> Option<String> {
        match (name, append_name) {
            (None, append_name) => append_name,
            (name, None) => name,
            (Some(name), Some(append_name)) => {
                Some(format!("{name}{}{append_name}", utilities::NAME_SEPARATOR))
            }
        }
    }

    /// Java static `getSingleLineInstance(String)`.
    pub fn get_single_line_instance(label: impl AsRef<str>) -> Self {
        let label = label.as_ref();
        let mut instance = Self::new(None, None, Some(label), None, false, false, None, false);
        instance.set_button_style("SingleLineButtonStyleExtension", Some(label));
        instance.create_panel();
        instance
    }

    /// Java static `getOpenCloseInstance(String)`.
    pub fn get_open_close_instance(label: impl AsRef<str>) -> Self {
        let label = label.as_ref();
        let mut instance = Self::new(None, None, Some(label), None, false, true, None, false);
        instance.set_button_style("OpenCloseButtonStyleExtension", Some(label));
        instance.create_panel();
        instance
    }

    /// Java static `getCloseInstance()`.
    pub fn get_close_instance() -> Self {
        let mut instance = Self::new(None, None, None, None, true, false, None, false);
        instance.set_button_style("CloseButtonStyleExtension", None);
        instance.create_panel();
        instance
    }

    /// Java static `getSelectFileInstance(ControlTarget, SelectFileExtension, boolean)`.
    pub fn get_select_file_instance(
        target: Rc<RefCell<dyn ControlTarget>>,
        shared_select_file_extension: Option<Rc<RefCell<SelectFileExtension>>>,
        debug: bool,
    ) -> Self {
        let mut instance = Self::new(
            Some(&SELECT_FILE),
            Some(target),
            None,
            None,
            false,
            false,
            shared_select_file_extension,
            debug,
        );
        instance.set_button_style("FileOpenButtonStyleExtension", None);
        instance.create_panel();
        instance
    }

    /// Java static `getSelectFileInstance(ControlTarget, SelectFileExtension, String)`.
    pub fn get_select_file_instance_with_label(
        target: Rc<RefCell<dyn ControlTarget>>,
        shared_select_file_extension: Option<Rc<RefCell<SelectFileExtension>>>,
        label: impl AsRef<str>,
    ) -> Self {
        let label = label.as_ref();
        let mut instance = Self::new(
            Some(&SELECT_FILE),
            Some(target),
            Some(label),
            None,
            false,
            false,
            shared_select_file_extension,
            false,
        );
        instance.set_button_style("FileOpenButtonStyleExtension", Some(label));
        instance.create_panel();
        instance
    }

    /// Java static `getSelectMultipleFilesInstance(ControlTarget, SelectFileExtension)`.
    pub fn get_select_multiple_files_instance(
        target: Rc<RefCell<dyn ControlTarget>>,
        shared_select_file_extension: Option<Rc<RefCell<SelectFileExtension>>>,
    ) -> Self {
        let mut instance = Self::new(
            Some(&SELECT_MULTIPLE_FILES),
            Some(target),
            None,
            None,
            false,
            false,
            shared_select_file_extension,
            false,
        );
        instance.set_button_style("FileOpenButtonStyleExtension", None);
        instance.create_panel();
        instance
    }

    /// Java static `getSelectMultipleFilesInstance(String, ControlTarget, SelectFileExtension)`.
    pub fn get_select_multiple_files_instance_with_label(
        label: impl AsRef<str>,
        target: Rc<RefCell<dyn ControlTarget>>,
        shared_select_file_extension: Option<Rc<RefCell<SelectFileExtension>>>,
    ) -> Self {
        let label = label.as_ref();
        let mut instance = Self::new(
            Some(&SELECT_MULTIPLE_FILES),
            Some(target),
            None,
            Some((20, 20)),
            false,
            false,
            shared_select_file_extension,
            false,
        );
        instance.set_button_style("FileOpenButtonStyleExtension", Some(label));
        instance.set_name(Some(label));
        instance.create_panel();
        instance
    }

    /// Java static `getClearInstance(ControlTarget)`.
    pub fn get_clear_instance(target: Rc<RefCell<dyn ControlTarget>>) -> Self {
        let mut instance = Self::new(
            Some(&CLEAR),
            Some(target),
            None,
            Some((20, 20)),
            false,
            false,
            None,
            false,
        );
        instance.set_button_style("ClearButtonStyleExtension", None);
        instance.create_panel();
        instance
    }

    /// Java static `getHeaderInstance(String)`.
    pub fn get_header_instance(label: impl AsRef<str>) -> Self {
        let label = label.as_ref();
        let mut instance = Self::new(None, None, Some(label), None, false, false, None, false);
        instance.set_button_style("HeaderButtonStyleExtension", Some(label));
        instance.create_panel();
        instance
    }

    /// Java static `getHeaderInstance()`.
    pub fn get_empty_header_instance() -> Self {
        let mut instance = Self::new(None, None, None, None, false, false, None, false);
        instance.set_button_style("HeaderButtonStyleExtension", None);
        instance.create_panel();
        instance
    }

    /// Java `setAllowFlagEditableControl(boolean)`.
    pub fn set_allow_flag_editable_control(&mut self, allow: bool) {
        self.allow_flag_editable_control = allow;
    }

    /// Java `setRespondToNonErrorFlags(boolean)`.
    pub fn set_respond_to_non_error_flags(&mut self, respond: bool) {
        self.respond_to_non_error_flags = respond;
    }

    /// Java private `createPanel()`.
    fn create_panel(&mut self) {
        if (self.implement_toggle && self.button_style.is_some()) || self.control_mode.is_some() {
            self.add_button_action_listener();
        }
    }

    /// Java private `addActionListener()`.
    fn add_button_action_listener(&mut self) {
        if !self.button_action_listening {
            self.button.action_listener_registered = true;
            self.button_action_listening = true;
        }
    }

    /// Java `getComponent()`.
    pub fn get_component(&self) -> &JButtonBoundary {
        &self.button
    }

    /// Java `setHorizontalAlignment(int)`.
    pub fn set_horizontal_alignment(&mut self, alignment: i32) {
        self.button.horizontal_alignment = Some(alignment);
    }

    /// Java `doClick()`.
    pub fn do_click(&mut self) {
        self.action_performed();
    }

    /// Java `addActionListener(ActionListener)`.
    pub fn add_action_listener(
        &mut self,
        listener: Option<Rc<RefCell<dyn ActionListenerBoundary>>>,
    ) {
        let Some(listener) = listener else { return };
        if self.action_listeners.is_none() {
            self.add_button_action_listener();
            self.action_listeners = Some(Vec::new());
        }
        self.action_listeners
            .as_mut()
            .expect("initialized above")
            .push(listener);
    }

    /// Java `getActionCommand()`.
    pub fn get_action_command(&self) -> Option<&str> {
        self.button.action_command.as_deref()
    }

    /// Java `getText()`.
    pub fn get_text(&self) -> Option<&str> {
        self.button.text.as_deref()
    }

    /// Java `setVisible(boolean)`.
    pub fn set_visible(&mut self, visible: bool) {
        self.button.visible = visible;
    }

    /// Java `setOverrideFileOpenDirectory(File)`.
    pub fn set_override_file_open_directory(&mut self, directory: Option<PathBuf>) {
        self.override_select_file_directory = directory;
    }

    /// Java `setFileFilter(FileFilter)`.
    pub fn set_file_filter(&mut self, file_filter: Rc<dyn FileFilter>) {
        if let Some(extension) = &self.select_file_extension {
            extension.borrow_mut().set_file_filter(file_filter);
        }
    }

    /// Java `setFileSelectionMode(int)`.
    pub fn set_file_selection_mode(&mut self, file_selection_mode: i32) {
        if let Some(extension) = &self.select_file_extension {
            extension
                .borrow_mut()
                .set_file_selection_mode(file_selection_mode);
        }
    }

    /// Java `setSelectFileDir(String)`.
    pub fn set_select_file_dir(&mut self, dir: impl Into<String>) {
        if let Some(extension) = &self.select_file_extension {
            extension.borrow_mut().set_dir(dir);
        }
    }

    /// Java `getSelectFileExtension()`.
    pub fn get_select_file_extension(&self) -> Option<Rc<RefCell<SelectFileExtension>>> {
        self.select_file_extension.clone()
    }

    /// Java `setSelected(boolean)`.
    pub fn set_selected(&mut self, selected: bool) {
        self.selected = selected;
        if self.implement_toggle {
            if let Some(button_style) = &mut self.button_style {
                button_style.update_appearance(
                    self.flag_type,
                    self.implement_toggle,
                    self.selected,
                );
            }
        }
    }

    /// Java `isControl()`.
    pub fn is_control(&self) -> bool {
        self.is_selected() && self.is_enabled()
    }

    /// Java `isSelected()`.
    pub fn is_selected(&self) -> bool {
        self.selected
    }

    /// Java `setTooltip(String)`.
    pub fn set_tooltip(&mut self, tooltip: Option<impl Into<String>>) {
        self.button.tooltip = tooltip.map(Into::into);
    }

    /// Java `actionPerformed(ActionEvent)`.
    pub fn action_performed(&mut self) {
        if self.implement_toggle && self.button_style.is_some() {
            self.selected = !self.selected;
            self.button_style
                .as_mut()
                .expect("checked above")
                .update_appearance(self.flag_type, self.implement_toggle, self.selected);
        }
        if let Some(control_mode) = self.control_mode {
            if let Some(target) = self.target.clone() {
                ControlMediator::INSTANCE.control_event(
                    self,
                    Some(&mut *target.borrow_mut()),
                    control_mode,
                );
            }
        }
        if let Some(action_listeners) = &self.action_listeners {
            let action_command = self.button.action_command.clone();
            for listener in action_listeners {
                listener
                    .borrow_mut()
                    .action_performed(action_command.as_deref());
            }
        }
    }

    /// Java `selectFile()`; native chooser presentation remains a GUI boundary.
    pub fn select_file(&mut self) -> Option<PathBuf> {
        None
    }

    /// Java `selectMultipleFiles()`; native chooser presentation remains a GUI boundary.
    pub fn select_multiple_files(&mut self) -> Option<Vec<PathBuf>> {
        None
    }

    /// Java private `createAppearanceExtension()`.
    fn create_appearance_extension(&mut self) {
        if self.appearance_extension.is_none() {
            let mut extension = AppearanceExtension::new(Rc::clone(&self.button.component));
            extension.set_allow_flag_editable_control(self.allow_flag_editable_control);
            extension.set_respond_to_non_error_flags(self.respond_to_non_error_flags);
            self.appearance_extension = Some(extension);
        }
    }

    /// Java `setFlag(FlagType)`.
    pub fn set_flag(&mut self, flag_type: Option<FlagType>) {
        self.flag_type = flag_type;
        if let Some(button_style) = &mut self.button_style {
            button_style.update_appearance(self.flag_type, self.implement_toggle, self.selected);
        }
        self.create_appearance_extension();
        self.appearance_extension
            .as_mut()
            .expect("created above")
            .set_flag(flag_type);
    }

    /// Java `setEnabled(boolean)`.
    pub fn set_enabled(&mut self, enabled: bool) {
        if let Some(appearance_extension) = &mut self.appearance_extension {
            appearance_extension.set_enabled(enabled);
        } else {
            self.button.component.borrow_mut().set_enabled(enabled);
        }
    }

    /// Java `isEnabled()`.
    pub fn is_enabled(&self) -> bool {
        self.appearance_extension
            .as_ref()
            .map(AppearanceExtension::is_enabled)
            .unwrap_or_else(|| self.button.component.borrow().is_enabled())
    }

    /// Java `setEditable(boolean)`.
    pub fn set_editable(&mut self, editable: bool) {
        self.create_appearance_extension();
        self.appearance_extension
            .as_mut()
            .expect("created above")
            .set_editable(editable);
    }

    /// Java `remove()`.
    pub fn remove(&mut self) {
        if let Some(grid_bag_extension) = &mut self.grid_bag_extension {
            grid_bag_extension.remove();
        }
    }

    /// Java `add(JPanel, GridBagLayout, GridBagConstraints)`.
    pub fn add(&mut self, constraints: (i32, i32)) {
        if self.grid_bag_extension.is_none() {
            self.grid_bag_extension = Some(GridBagExtension::new());
        }
        self.grid_bag_extension
            .as_mut()
            .expect("created above")
            .add(0, constraints);
    }

    /// Java `setAltBrowsingDirectory(BrowsingDirectory)`.
    pub fn set_alt_browsing_directory(&mut self, browsing_directory: Rc<dyn BrowsingDirectory>) {
        if let Some(extension) = &self.select_file_extension {
            extension
                .borrow_mut()
                .set_alt_browsing_directory(browsing_directory);
        }
    }
}

impl FlagDisplay for Ebutton {
    fn set_flag(&mut self, flag_type: Option<FlagType>) {
        Ebutton::set_flag(self, flag_type);
    }
}

impl Controller for Ebutton {
    fn is_control(&self) -> bool {
        Ebutton::is_control(self)
    }
    fn set_editable(&mut self, editable: bool) {
        Ebutton::set_editable(self, editable);
    }
    fn set_enabled(&mut self, enabled: bool) {
        Ebutton::set_enabled(self, enabled);
    }
    fn select_file(&mut self) -> Option<PathBuf> {
        Ebutton::select_file(self)
    }
    fn select_multiple_files(&mut self) -> Option<Vec<PathBuf>> {
        Ebutton::select_multiple_files(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Listener {
        calls: Vec<Option<String>>,
    }
    impl ActionListenerBoundary for Listener {
        fn action_performed(&mut self, action_command: Option<&str>) {
            self.calls.push(action_command.map(str::to_owned));
        }
    }

    #[derive(Default)]
    struct Target {
        clears: usize,
        events: usize,
    }
    impl ControlTarget for Target {
        fn clear(&mut self) {
            self.clears += 1;
        }
        fn set_text_file(&mut self, _: &std::path::Path) {}
        fn set_text_files(&mut self, _: &[PathBuf]) {}
        fn get_label(&self) -> String {
            "input files".to_owned()
        }
        fn set_component_control(
            &mut self,
            _: bool,
            _: &super::super::control_state::ControlState,
        ) {
        }
        fn set_enable_control(&mut self, _: bool, _: &super::super::control_state::ControlState) {}
        fn send_control_event(&mut self) {
            self.events += 1;
        }
        fn is_local_dir(&self, _: &str) -> bool {
            false
        }
    }

    #[test]
    fn toggle_style_is_updated_before_external_listeners() {
        let mut button = Ebutton::get_open_close_instance("Open");
        button.button.action_command = Some("open".to_owned());
        let listener = Rc::new(RefCell::new(Listener::default()));
        button.add_action_listener(Some(listener.clone()));
        button.do_click();
        assert!(button.selected);
        assert_eq!(button.button_style.as_ref().unwrap().update_count, 1);
        assert_eq!(listener.borrow().calls, vec![Some("open".to_owned())]);
    }

    #[test]
    fn clear_mode_uses_control_mediator_then_notifies_target() {
        let target = Rc::new(RefCell::new(Target::default()));
        let mut button = Ebutton::get_clear_instance(target.clone());
        button.do_click();
        assert_eq!(target.borrow().clears, 1);
        assert_eq!(target.borrow().events, 1);
        assert!(button.button_action_listening);
    }

    #[test]
    fn appearance_and_grid_bag_source_transitions_are_retained() {
        let mut button = Ebutton::get_single_line_instance("Run test");
        button.set_respond_to_non_error_flags(false);
        button.set_flag(Some(FlagType::WARNING));
        assert_eq!(
            button
                .appearance_extension
                .as_ref()
                .unwrap()
                .get_flag_type(),
            None
        );
        button.set_enabled(false);
        assert!(!button.is_enabled());
        button.add((2, 3));
        button.remove();
        assert!(button.grid_bag_extension.as_ref().unwrap().removed);
    }

    #[test]
    fn select_factories_share_the_java_extension_and_preserve_control_name() {
        let target: Rc<RefCell<dyn ControlTarget>> = Rc::new(RefCell::new(Target::default()));
        let shared = Rc::new(RefCell::new(SelectFileExtension::new()));
        let button = Ebutton::get_select_file_instance(target, Some(shared.clone()), true);
        assert!(Rc::ptr_eq(
            &button.get_select_file_extension().unwrap(),
            &shared
        ));
        assert_eq!(
            button.button.name.as_deref(),
            Some("ctb.input-files-select-file")
        );
        assert!(button.debug);
    }
}
