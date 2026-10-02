//! `IMOD/Etomo/src/etomo/ui/swing/Ebutton.java`.
//!
//! An extensible `JButton`: a button style (icons, and a toggle appearance
//! for a plain `JButton`), a control mode (clear a target, select a file or
//! files for it, through the `ControlMediator`), flag appearance, and grid bag
//! placement.
//!
//! The control target is held as a `Weak<dyn ControlTarget>`: the target (a
//! file text field, ...) owns its buttons and usually passes itself while it
//! is being constructed.  The Java `fixedSize` and `border` constructor
//! arguments are layout, so the private constructor drops them and each
//! factory says in a comment what it passed; `setHorizontalAlignment` is
//! layout too.  `add(JPanel, GridBagLayout, GridBagConstraints)` keeps the
//! panel and drops the layout arguments.

use std::cell::{Cell, RefCell};
use std::path::PathBuf;
use std::rc::{Rc, Weak};

use crate::imod::etomo::etomo_director::ARGUMENTS;
use crate::imod::etomo::jdk::{ActionEvent, ActionListener, JComponent};
use crate::imod::etomo::storage::autodoc::autodoc_tokenizer::{DEFAULT_DELIMITER, SEPARATOR_CHAR};
use crate::imod::etomo::r#type::ui_test_field_type::UITestFieldType;
use crate::imod::etomo::ui::browsing_directory::BrowsingDirectory;
use crate::imod::etomo::ui::flag_display::FlagDisplay;
use crate::imod::etomo::ui::flag_type::FlagType;
use crate::imod::etomo::util::utilities;

use super::appearance_extension::AppearanceExtension;
use super::button_style_extension::ButtonStyleExtensionVirtual;
use super::clear_button_style_extension::ClearButtonStyleExtension;
use super::close_button_style_extension::CloseButtonStyleExtension;
use super::control_mediator::{self, ControlMediator};
use super::control_mode::{self, ControlMode};
use super::control_target::ControlTarget;
use super::controller::Controller;
use super::file_open_button_style_extension::FileOpenButtonStyleExtension;
// use super::file_text_field_interface::FileFilter;
use super::grid_bag_extension::GridBagExtension;
use super::header_button_style_extension::HeaderButtonStyleExtension;
use super::open_close_button_style_extension::OpenCloseButtonStyleExtension;
use super::select_file_extension::SelectFileExtension;
use super::single_line_button_style_extension::SingleLineButtonStyleExtension;
use super::tooltip_formatter::TooltipFormatter;
use crate::imod::etomo::jdk::FileFilter;

/// Java package-private final `Ebutton`.
pub struct Ebutton {
    /// This object, for the Java calls that pass `this`.
    self_ref: RefCell<Weak<Ebutton>>,
    /// Java final `button` (`new JButton()`).
    button: Rc<JComponent>,
    /// Java final `target`.
    target: Option<Weak<dyn ControlTarget>>,
    /// Java final `controlMode`.
    control_mode: Option<&'static ControlMode>,
    /// Java final `selectFileExtension`.
    select_file_extension: Option<Rc<SelectFileExtension>>,
    /// Java final `implementToggle`: gives a JButton the appearance of
    /// toggling, the button style used comtains a complete icon with a
    /// selected image.
    implement_toggle: bool,
    /// Java `buttonStyle`.
    button_style: RefCell<Option<Rc<dyn ButtonStyleExtensionVirtual>>>,
    /// Java `gridBagExtension`.
    grid_bag_extension: RefCell<Option<Rc<GridBagExtension>>>,
    /// Java `selected`.
    selected: Cell<bool>,
    /// Java `appearanceExtension`.
    appearance_extension: RefCell<Option<Rc<AppearanceExtension>>>,
    /// Java `actionListeners`.
    action_listeners: RefCell<Option<Vec<ActionListener>>>,
    /// Java `allowFlagEditableControl`.
    allow_flag_editable_control: Cell<bool>,
    /// Java `respondToNonErrorFlags`.
    respond_to_non_error_flags: Cell<bool>,
    /// Java `flagType`.
    flag_type: Cell<Option<&'static FlagType>>,
    /// Java `controlMediator`.  Never assigned or read in the Java class.
    control_mediator: Cell<Option<&'static ControlMediator>>,
    /// Java `buttonActionListening`.
    button_action_listening: Cell<bool>,
    /// Java `overrideSelectFileDirectory`.
    override_select_file_directory: RefCell<Option<PathBuf>>,
    /// Java `debug`.
    debug: Cell<bool>,
}

impl Ebutton {
    /// Java private
    /// `Ebutton(ControlMode, ControlTarget, String, Dimension, Border, boolean, SelectFileExtension, boolean)`.
    /// `fixedSize` (preferred and maximum size) and `border` are layout and
    /// are not passed.
    fn new(
        control_mode: Option<&'static ControlMode>,
        target: Option<Weak<dyn ControlTarget>>,
        label: Option<&str>,
        implement_toggle: bool,
        shared_select_file_extension: Option<Rc<SelectFileExtension>>,
        debug: bool,
    ) -> Rc<Ebutton> {
        let select_file_extension = if control_mode
            .is_some_and(|mode| std::ptr::eq(mode, &*control_mode::SELECT_FILE))
            || control_mode
                .is_some_and(|mode| std::ptr::eq(mode, &*control_mode::SELECT_MULTIPLE_FILES))
        {
            if shared_select_file_extension.is_some() {
                // Shared extension avaiable
                shared_select_file_extension
            } else {
                Some(SelectFileExtension::new())
            }
        } else {
            None
        };
        let instance = Rc::new(Ebutton {
            self_ref: RefCell::new(Weak::new()),
            button: JComponent::new_button(""),
            target,
            control_mode,
            select_file_extension,
            implement_toggle,
            button_style: RefCell::new(None),
            grid_bag_extension: RefCell::new(None),
            selected: Cell::new(false),
            appearance_extension: RefCell::new(None),
            action_listeners: RefCell::new(None),
            allow_flag_editable_control: Cell::new(true),
            respond_to_non_error_flags: Cell::new(true),
            flag_type: Cell::new(None),
            control_mediator: Cell::new(None),
            button_action_listening: Cell::new(false),
            override_select_file_directory: RefCell::new(None),
            debug: Cell::new(debug),
        });
        *instance.self_ref.borrow_mut() = Rc::downgrade(&instance);
        if let Some(label) = label {
            instance.button.set_text(label);
        }
        let target = instance.target.clone();
        instance.set_name(label, target.as_ref(), control_mode);
        // Swing layout: when fixedSize != null, button.setPreferredSize(fixedSize)
        // and button.setMaximumSize(fixedSize); when border != null,
        // button.setBorder(border).
        // (The Java assigns selectFileExtension after these; nothing between
        // reads it.)
        instance
    }

    /// Java private `setButtonStyle(ButtonStyleExtension, String)`.
    fn set_button_style(
        &self,
        button_style: Option<Rc<dyn ButtonStyleExtensionVirtual>>,
        label: Option<&str>,
    ) {
        *self.button_style.borrow_mut() = button_style.clone();
        if let Some(button_style) = button_style {
            button_style.setup(Some(&self.button), label, self.debug.get());
        }
    }

    /// Java public `setDebug(boolean)`.
    pub fn set_debug(&self, debug: bool) {
        self.debug.set(debug);
    }

    /// Java `toString()`.
    pub fn to_string(&self) -> String {
        format!(
            "[{},{}]",
            self.button.get_name().as_deref().unwrap_or("null"),
            self.button.get_text()
        )
    }

    /// Java private `setName(String, ControlTarget, ControlMode)`.
    fn set_name(
        &self,
        label: Option<&str>,
        target: Option<&Weak<dyn ControlTarget>>,
        control_mode: Option<&'static ControlMode>,
    ) {
        // build name
        let mut name: Option<String> = None;
        let mut control_name_set = false;
        let mut field_type = UITestFieldType::BUTTON;
        if label.is_some() {
            name = self.append_to_name(
                name,
                utilities::convert_label_to_name(label, field_type.is_unlimited_segments()),
            );
        } else if let Some(control_mode) = control_mode {
            control_name_set = control_mode.has_field_name();
            if control_name_set {
                if control_name_set {
                    field_type = UITestFieldType::CONTROL_BUTTON;
                }
                if let Some(target) = target.and_then(Weak::upgrade) {
                    name = utilities::convert_label_to_name(
                        target.get_label().as_deref(),
                        field_type.is_unlimited_segments(),
                    );
                }
                name = control_mode.append_to_name(name.as_deref());
            }
        }
        let Some(name) = name else {
            return;
        };
        // set the name in the field
        self.button.set_name(Some(&format!(
            "{}{}{}",
            field_type.to_string(),
            SEPARATOR_CHAR,
            name
        )));
        // Java `EtomoDirector.INSTANCE.getArguments()` is the `ARGUMENTS` static.
        if ARGUMENTS.lock().unwrap().is_print_names() {
            println!(
                "{} {} ",
                self.button.get_name().as_deref().unwrap_or("null"),
                DEFAULT_DELIMITER
            );
        }
    }

    /// Java private `appendToName(String, String)`.
    fn append_to_name(&self, name: Option<String>, append_name: Option<String>) -> Option<String> {
        let Some(name) = name else {
            return append_name;
        };
        let Some(append_name) = append_name else {
            return Some(name);
        };
        Some(format!(
            "{}{}{}",
            name,
            utilities::NAME_SEPARATOR,
            append_name
        ))
    }

    /// Java static `getSingleLineInstance(String)`.
    pub fn get_single_line_instance(label: Option<&str>) -> Rc<Ebutton> {
        let instance = Ebutton::new(None, None, label, false, None, false);
        instance.set_button_style(Some(SingleLineButtonStyleExtension::get_instance()), label);
        instance.create_panel();
        instance
    }

    /// Java static `getOpenCloseInstance(String)`.
    pub fn get_open_close_instance(label: Option<&str>) -> Rc<Ebutton> {
        let instance = Ebutton::new(None, None, label, true, None, false);
        instance.set_button_style(Some(OpenCloseButtonStyleExtension::get_instance()), label);
        instance.create_panel();
        instance
    }

    /// Java static `getCloseInstance()`.
    pub fn get_close_instance() -> Rc<Ebutton> {
        // Java passes border BorderFactory.createEmptyBorder() (layout).
        let instance = Ebutton::new(None, None, None, false, None, false);
        instance.set_button_style(
            Some(CloseButtonStyleExtension::get_instance(&instance.button)),
            None,
        );
        instance.create_panel();
        instance
    }

    /// Java static `getSelectFileInstance(ControlTarget, SelectFileExtension, boolean)`.
    /// `sharedSelectFileExtension` (optional) is for a group of buttons that
    /// can use the same file chooser.
    pub fn get_select_file_instance_control_target_select_file_extension_boolean(
        target: Option<Weak<dyn ControlTarget>>,
        shared_select_file_extension: Option<Rc<SelectFileExtension>>,
        debug: bool,
    ) -> Rc<Ebutton> {
        let instance = Ebutton::new(
            Some(&*control_mode::SELECT_FILE),
            target,
            None,
            false,
            shared_select_file_extension,
            debug,
        );
        instance.set_button_style(
            Some(FileOpenButtonStyleExtension::get_instance(&instance.button)),
            None,
        );
        instance.create_panel();
        instance
    }

    /// Java static `getSelectFileInstance(ControlTarget, SelectFileExtension, String)`.
    pub fn get_select_file_instance_control_target_select_file_extension_string(
        target: Option<Weak<dyn ControlTarget>>,
        shared_select_file_extension: Option<Rc<SelectFileExtension>>,
        label: Option<&str>,
    ) -> Rc<Ebutton> {
        let instance = Ebutton::new(
            Some(&*control_mode::SELECT_FILE),
            target,
            label,
            false,
            shared_select_file_extension,
            false,
        );
        instance.set_button_style(
            Some(FileOpenButtonStyleExtension::get_instance(&instance.button)),
            label,
        );
        instance.create_panel();
        instance
    }

    /// Java static `getSelectMultipleFilesInstance(ControlTarget, SelectFileExtension)`.
    pub fn get_select_multiple_files_instance_control_target_select_file_extension(
        target: Option<Weak<dyn ControlTarget>>,
        shared_select_file_extension: Option<Rc<SelectFileExtension>>,
    ) -> Rc<Ebutton> {
        // Java passes fixedSize null (a commented-out FixedDim.INLINE_SQUARE_SIZE).
        let instance = Ebutton::new(
            Some(&*control_mode::SELECT_MULTIPLE_FILES),
            target,
            None,
            false,
            shared_select_file_extension,
            false,
        );
        instance.set_button_style(
            Some(FileOpenButtonStyleExtension::get_instance(&instance.button)),
            None,
        );
        instance.create_panel();
        instance
    }

    /// Java static
    /// `getSelectMultipleFilesInstance(String, ControlTarget, SelectFileExtension)`.
    pub fn get_select_multiple_files_instance_string_control_target_select_file_extension(
        label: Option<&str>,
        target: Option<Weak<dyn ControlTarget>>,
        shared_select_file_extension: Option<Rc<SelectFileExtension>>,
    ) -> Rc<Ebutton> {
        // Java passes fixedSize FixedDim.INLINE_SQUARE_SIZE (layout).
        let instance = Ebutton::new(
            Some(&*control_mode::SELECT_MULTIPLE_FILES),
            target.clone(),
            None,
            false,
            shared_select_file_extension,
            false,
        );
        instance.set_button_style(
            Some(FileOpenButtonStyleExtension::get_instance(&instance.button)),
            label,
        );
        instance.set_name(
            label,
            target.as_ref(),
            Some(&*control_mode::SELECT_MULTIPLE_FILES),
        );
        instance.create_panel();
        instance
    }

    /// Java static `getClearInstance(ControlTarget)`.
    pub fn get_clear_instance(target: Option<Weak<dyn ControlTarget>>) -> Rc<Ebutton> {
        // Java passes fixedSize FixedDim.INLINE_SQUARE_SIZE (layout).
        let instance = Ebutton::new(
            Some(&*control_mode::CLEAR),
            target,
            None,
            false,
            None,
            false,
        );
        instance.set_button_style(
            Some(ClearButtonStyleExtension::get_instance(&instance.button)),
            None,
        );
        instance.create_panel();
        instance
    }

    /// Java static `getHeaderInstance(String)`.
    pub fn get_header_instance_string(label: Option<&str>) -> Rc<Ebutton> {
        let instance = Ebutton::new(None, None, label, false, None, false);
        instance.set_button_style(Some(HeaderButtonStyleExtension::get_instance()), label);
        instance.create_panel();
        instance
    }

    /// Java static `getHeaderInstance()`.
    pub fn get_header_instance_void() -> Rc<Ebutton> {
        let instance = Ebutton::new(None, None, None, false, None, false);
        instance.set_button_style(Some(HeaderButtonStyleExtension::get_instance()), None);
        instance.create_panel();
        instance
    }

    /// Java public `setAllowFlagEditableControl(boolean)`.
    pub fn set_allow_flag_editable_control(&self, allow: bool) {
        self.allow_flag_editable_control.set(allow);
    }

    /// Java public `setRespondToNonErrorFlags(boolean)`.
    pub fn set_respond_to_non_error_flags(&self, respond: bool) {
        self.respond_to_non_error_flags.set(respond);
    }

    /// Java private `createPanel()`.
    fn create_panel(&self) {
        if (self.implement_toggle && self.button_style.borrow().is_some())
            || self.control_mode.is_some()
        {
            self.add_action_listener_void();
        }
    }

    /// Java private `addActionListener()`: `button.addActionListener(this)`.
    fn add_action_listener_void(&self) {
        if !self.button_action_listening.get() {
            let this = self.self_ref.borrow().clone();
            self.button.add_action_listener(Rc::new(move |event| {
                if let Some(this) = this.upgrade() {
                    this.action_performed(event);
                }
            }));
            self.button_action_listening.set(true);
        }
    }

    /// Java `getComponent()`.
    pub fn get_component(&self) -> Rc<JComponent> {
        self.button.clone()
    }

    /// Java `setHorizontalAlignment(int)`.
    pub fn set_horizontal_alignment(&self, _alignment: i32) {
        // Swing layout: button.setHorizontalAlignment(alignment).
    }

    /// Java `doClick()`.
    pub fn do_click(&self) {
        self.button.do_click();
    }

    /// Java `addActionListener(ActionListener)`.
    pub fn add_action_listener_action_listener(&self, listener: Option<ActionListener>) {
        let Some(listener) = listener else {
            return;
        };
        if self.action_listeners.borrow().is_none() {
            self.add_action_listener_void();
            *self.action_listeners.borrow_mut() = Some(Vec::new());
        }
        self.action_listeners
            .borrow_mut()
            .as_mut()
            .unwrap()
            .push(listener);
    }

    /// Java `getActionCommand()`.
    pub fn get_action_command(&self) -> Option<String> {
        self.button.get_action_command()
    }

    /// Java `getText()`.
    pub fn get_text(&self) -> String {
        self.button.get_text()
    }

    /// Java `setVisible(boolean)`.
    pub fn set_visible(&self, visible: bool) {
        self.get_component().set_visible(visible);
    }

    /// Java `setOverrideFileOpenDirectory(File)`.
    pub fn set_override_file_open_directory(
        &self,
        override_select_file_directory: Option<PathBuf>,
    ) {
        *self.override_select_file_directory.borrow_mut() = override_select_file_directory;
    }

    /// Java `setFileFilter(FileFilter)`.
    pub fn set_file_filter(&self, file_filter: Option<Rc<dyn FileFilter>>) {
        if let Some(select_file_extension) = &self.select_file_extension {
            select_file_extension.set_file_filter(file_filter);
        }
    }

    /// Java `setFileSelectionMode(int)`.
    pub fn set_file_selection_mode(&self, file_selection_mode: i32) {
        if let Some(select_file_extension) = &self.select_file_extension {
            select_file_extension.set_file_selection_mode(file_selection_mode);
        }
    }

    /// Java `setSelectFileDir(String)`.
    pub fn set_select_file_dir(&self, dir: Option<&str>) {
        if let Some(select_file_extension) = &self.select_file_extension {
            select_file_extension.set_dir(dir);
        }
    }

    /// Java `getSelectFileExtension()`.
    pub fn get_select_file_extension(&self) -> Option<Rc<SelectFileExtension>> {
        self.select_file_extension.clone()
    }

    /// Java `setSelected(boolean)`.
    pub fn set_selected(&self, selected: bool) {
        self.selected.set(selected);
        let button_style = self.button_style.borrow().clone();
        if let Some(button_style) = button_style {
            if self.implement_toggle {
                button_style.update_appearance(
                    &self.button,
                    self.flag_type.get(),
                    self.implement_toggle,
                    selected,
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
        self.selected.get()
    }

    /// Java `setTooltip(String)`.
    pub fn set_tooltip(&self, tooltip: Option<&str>) {
        self.button.set_tool_tip_text(
            super::tooltip_formatter::INSTANCE
                .format(tooltip)
                .as_deref(),
        );
    }

    /// Java `actionPerformed(ActionEvent)` (`implements ActionListener`).
    pub fn action_performed(&self, event: &ActionEvent) {
        let button_style = self.button_style.borrow().clone();
        if self.implement_toggle {
            if let Some(button_style) = button_style {
                self.selected.set(!self.selected.get());
                button_style.update_appearance(
                    &self.button,
                    self.flag_type.get(),
                    self.implement_toggle,
                    self.selected.get(),
                );
            }
        }
        if let Some(control_mode) = self.control_mode {
            let target = self.target.as_ref().and_then(Weak::upgrade);
            control_mediator::INSTANCE.control_event_controller_control_target_control_mode(
                self,
                target.as_deref(),
                Some(control_mode),
            );
        }
        let action_listeners = self.action_listeners.borrow().clone();
        if let Some(action_listeners) = action_listeners {
            for listener in &action_listeners {
                listener(event);
            }
        }
    }

    /// Java `selectFile()`.
    pub fn select_file(&self) -> Option<PathBuf> {
        if let Some(select_file_extension) = &self.select_file_extension {
            return select_file_extension.select_file(
                Some(&self.button),
                self.override_select_file_directory.borrow().clone(),
            );
        }
        None
    }

    /// Java `selectMultipleFiles()`.  ALIGN_FRAMES.
    pub fn select_multiple_files(&self) -> Option<Vec<PathBuf>> {
        if let Some(select_file_extension) = &self.select_file_extension {
            return select_file_extension.select_multiple_files(
                Some(&self.button),
                self.override_select_file_directory.borrow().clone(),
            );
        }
        None
    }

    /// Java private `createAppearanceExtension()`.
    fn create_appearance_extension(&self) -> Rc<AppearanceExtension> {
        if self.appearance_extension.borrow().is_none() {
            let appearance_extension = AppearanceExtension::new_component(&self.button);
            appearance_extension
                .set_allow_flag_editable_control(self.allow_flag_editable_control.get());
            appearance_extension
                .set_respond_to_non_error_flags(self.respond_to_non_error_flags.get());
            *self.appearance_extension.borrow_mut() = Some(appearance_extension);
        }
        self.appearance_extension.borrow().clone().unwrap()
    }

    /// Java `setFlag(FlagType)`.
    pub fn set_flag(&self, flag_type: Option<&'static FlagType>) {
        self.flag_type.set(flag_type);
        let button_style = self.button_style.borrow().clone();
        if let Some(button_style) = button_style {
            button_style.update_appearance(
                &self.button,
                flag_type,
                self.implement_toggle,
                self.selected.get(),
            );
        }
        let appearance_extension = self.create_appearance_extension();
        appearance_extension.set_flag(flag_type);
    }

    /// Java `setEnabled(boolean)`.
    pub fn set_enabled(&self, enabled: bool) {
        let appearance_extension = self.appearance_extension.borrow().clone();
        if let Some(appearance_extension) = appearance_extension {
            appearance_extension.set_enabled(enabled);
        } else {
            self.button.set_enabled(enabled);
        }
    }

    /// Java `isEnabled()`.
    pub fn is_enabled(&self) -> bool {
        let appearance_extension = self.appearance_extension.borrow().clone();
        if let Some(appearance_extension) = appearance_extension {
            return appearance_extension.is_enabled();
        }
        self.button.is_enabled()
    }

    /// Java `setEditable(boolean)`.
    pub fn set_editable(&self, editable: bool) {
        let appearance_extension = self.create_appearance_extension();
        appearance_extension.set_editable(editable);
    }

    /// Java `remove()` (gridBagExtension).
    pub fn remove(&self) {
        let grid_bag_extension = self.grid_bag_extension.borrow().clone();
        if let Some(grid_bag_extension) = grid_bag_extension {
            grid_bag_extension.remove(&self.get_component());
        }
    }

    /// Java `add(JPanel, GridBagLayout, GridBagConstraints)`.  The layout and
    /// constraints are not modelled.
    pub fn add(&self, panel: &Rc<JComponent>) {
        if self.grid_bag_extension.borrow().is_none() {
            *self.grid_bag_extension.borrow_mut() = Some(GridBagExtension::new());
        }
        let grid_bag_extension = self.grid_bag_extension.borrow().clone().unwrap();
        grid_bag_extension.add(&self.get_component(), panel);
    }

    /// Java public `setAltBrowsingDirectory(BrowsingDirectory)`.
    pub fn set_alt_browsing_directory(
        &self,
        browsing_directory: Option<Rc<dyn BrowsingDirectory>>,
    ) {
        if let Some(select_file_extension) = &self.select_file_extension {
            select_file_extension.set_alt_browsing_directory(browsing_directory);
        }
    }
}

/// Java `implements FlagDisplay`.
impl FlagDisplay for Ebutton {
    fn set_flag(&self, flag_type: Option<&'static FlagType>) {
        Ebutton::set_flag(self, flag_type)
    }
}

/// Java `implements Controller`.
impl Controller for Ebutton {
    fn is_control(&self) -> bool {
        Ebutton::is_control(self)
    }
    fn set_editable(&self, editable: bool) {
        Ebutton::set_editable(self, editable)
    }
    fn set_enabled(&self, enabled: bool) {
        Ebutton::set_enabled(self, enabled)
    }
    fn select_file(&self) -> Option<PathBuf> {
        Ebutton::select_file(self)
    }
    fn select_multiple_files(&self) -> Option<Vec<PathBuf>> {
        Ebutton::select_multiple_files(self)
    }
}
