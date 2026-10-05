//! `IMOD/Etomo/src/etomo/plugin/demo/DemoPanel.java`.
//!
//! The demo plugin's panel in the Tomogram Generation dialog (selected by its "Demo"
//! radio button).  An event-dispatch-thread object created as `Rc<Self>` by
//! [`DemoPanel::get_instance`]; every method takes `&self`.  It is its own
//! `ActionListener` (a closure holding a weak reference) and hands itself to its 3dmod
//! buttons as their `Run3dmodButtonContainer` and to its panel header as its
//! `Expandable`.

use std::cell::OnceCell;
use std::rc::{Rc, Weak};

use super::demo_plugin_manager::DemoPluginManager;
use super::etomo_plugin_demo_param::{self, EtomoPluginDemoParam};
use super::generic_3dmod_file_filter::Generic3dmodFileFilter;
use super::sleep_time::SleepTime;
use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::jdk::{
    ActionEvent, ActionListener, ButtonGroup, FileFilter, JComponent, MouseEvent, MouseListener,
};
use crate::imod::etomo::plugin::plugin_panel::PluginPanel;
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::storage::autodoc::read_only_autodoc::ReadOnlyAutodoc;
use crate::imod::etomo::storage::log_file::LogFileError;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::base_screen_state::BaseScreenState;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::enumerated_type::EnumeratedType;
use crate::imod::etomo::r#type::etomo_autodoc;
use crate::imod::etomo::r#type::process_result_display::ProcessResultDisplay;
use crate::imod::etomo::ui::field::Field;
use crate::imod::etomo::ui::field_type::FieldType;
use crate::imod::etomo::ui::swing::abstract_radio_button_model::AbstractRadioButtonModel;
use crate::imod::etomo::ui::swing::check_box::CheckBox;
use crate::imod::etomo::ui::swing::check_text_field::CheckTextField;
use crate::imod::etomo::ui::swing::context_menu::ContextMenu;
use crate::imod::etomo::ui::swing::context_popup::{self, ContextPopup};
use crate::imod::etomo::ui::swing::deferred_3dmod_button::Deferred3dmodButton;
use crate::imod::etomo::ui::swing::expand_button::ExpandButton;
use crate::imod::etomo::ui::swing::expandable::Expandable;
use crate::imod::etomo::ui::swing::file_text_field_interface::FileTextFieldInterface;
use crate::imod::etomo::ui::swing::file_text_field2::FileTextField2;
use crate::imod::etomo::ui::swing::generic_mouse_adapter::GenericMouseAdapter;
use crate::imod::etomo::ui::swing::global_expand_button::GlobalExpandButton;
use crate::imod::etomo::ui::swing::labeled_text_field::LabeledTextField;
use crate::imod::etomo::ui::swing::multi_line_button::MultiLineButton;
use crate::imod::etomo::ui::swing::panel_header::PanelHeader;
use crate::imod::etomo::ui::swing::radio_button::RadioButton;
use crate::imod::etomo::ui::swing::radio_button::RadioButtonModel;
use crate::imod::etomo::ui::swing::radio_button_interface::EnumeratedTypeRef;
use crate::imod::etomo::ui::swing::radio_text_field::RadioTextField;
use crate::imod::etomo::ui::swing::run_3dmod_button::Run3dmodButton;
use crate::imod::etomo::ui::swing::run_3dmod_button_container::Run3dmodButtonContainer;
use crate::imod::etomo::ui::swing::swing_component::SwingComponent;
use crate::imod::etomo::ui::swing::tomogram_generation_dialog::TomogramGenerationDialog;
use crate::imod::etomo::ui::swing::ui_harness;
use crate::imod::etomo::ui::ui_component::UIComponent;

/// Java private static final `TITLE`.
const TITLE: &str = "Demo";

/// Java `final class DemoPanel implements PluginPanel, ContextMenu,
/// Run3dmodButtonContainer, Expandable, ActionListener, UIComponent, SwingComponent`.
pub struct DemoPanel {
    /// Java private final `pnlRoot = new JPanel()`.
    pnl_root: Rc<JComponent>,
    /// Java private final `pnlDemoBody = new JPanel(true)`.
    pnl_demo_body: Rc<JComponent>,

    /// Java private final `cbSleepTime`.
    cb_sleep_time: Rc<CheckBox>,
    /// Java private final `bgSleepTime`.
    bg_sleep_time: Rc<ButtonGroup>,
    /// Java private final `rbSleepTime2`.
    rb_sleep_time2: Rc<RadioButton>,
    /// Java private final `rbSleepTime3`.
    rb_sleep_time3: Rc<RadioButton>,
    /// Java private final `rtfSleepTime`.
    rtf_sleep_time: Rc<RadioTextField>,

    /// Java private final `ctfMessage`.
    ctf_message: Rc<CheckTextField>,
    /// Java private final `cbSwapYZ`.
    cb_swap_yz: Rc<CheckBox>,
    /// Java private final `btn3dmod = Run3dmodButton.get3dmodInstance("Open File",
    /// this)`.
    btn_3dmod: Rc<Run3dmodButton>,
    /// Java private final `ltfSleepTimeUsed`.
    ltf_sleep_time_used: Rc<LabeledTextField>,
    /// Java private final `ltfCpus`.
    ltf_cpus: Rc<LabeledTextField>,

    /// Java private final `manager`.
    manager: &'static ApplicationManager,
    /// Java private final `axisID`.
    axis_id: AxisID,
    /// Java private final `pluginManager` (it owns this panel).
    plugin_manager: Weak<DemoPluginManager>,
    /// Java private final `parent` (it owns this panel).
    parent: Weak<TomogramGenerationDialog>,
    /// Java private final `phDemo`.
    ph_demo: Rc<PanelHeader>,
    /// Java private final `ftfFile`.
    ftf_file: Rc<FileTextField2>,
    /// Java private final `btnEtomoPluginDemo`.  Run3dmodButton is used for this run
    /// button so that it has a right-click menu.
    btn_etomo_plugin_demo: Rc<Run3dmodButton>,
    /// Java private final `btnUnselectDependencies`.
    btn_unselect_dependencies: Rc<MultiLineButton>,

    /// Java `this` as the `ActionListener` added to the buttons (one object, so
    /// `done()` can remove it).
    action_listener: OnceCell<ActionListener>,
}

impl DemoPanel {
    /// Java private `DemoPanel(DemoPluginManager, ApplicationManager, AxisID,
    /// DialogType, GlobalExpandButton, TomogramGenerationDialog)`.
    fn new(
        plugin_manager: &Rc<DemoPluginManager>,
        manager: &'static ApplicationManager,
        axis_id: AxisID,
        dialog_type: Option<DialogType>,
        global_advanced_button: &Rc<GlobalExpandButton>,
        parent: Weak<TomogramGenerationDialog>,
    ) -> Rc<DemoPanel> {
        Rc::new_cyclic(|this: &Weak<DemoPanel>| {
            // Field initializers, in declaration order.
            let pnl_root = JComponent::new_panel();
            let pnl_demo_body = JComponent::new_panel();
            let cb_sleep_time = CheckBox::new_string(Some("Modify sleep time"));
            let bg_sleep_time = ButtonGroup::new();
            let rb_sleep_time2 = RadioButton::new_enumerated_type_button_group(
                EnumeratedTypeRef::new(SleepTime::Two),
                Some(&bg_sleep_time),
            );
            let rb_sleep_time3 = RadioButton::new_enumerated_type_button_group(
                EnumeratedTypeRef::new(SleepTime::Three),
                Some(&bg_sleep_time),
            );
            let rtf_sleep_time =
                RadioTextField::get_instance_field_type_enumerated_type_button_group(
                    FieldType::Integer,
                    Some(EnumeratedTypeRef::new(SleepTime::UserEntry)),
                    Some(&bg_sleep_time),
                );
            let ctf_message =
                CheckTextField::get_instance(FieldType::String, "Override default message: ");
            let cb_swap_yz = CheckBox::new_string(Some("Swap Y and Z axes"));
            let container: Weak<dyn Run3dmodButtonContainer> = this.clone();
            let btn_3dmod = Run3dmodButton::get_3dmod_instance_string_run_3dmod_button_container(
                Some("Open File"),
                Some(container),
            );
            let ltf_sleep_time_used = LabeledTextField::new_field_type_string(
                FieldType::String,
                Some("Sleep time used: "),
            );
            let ltf_cpus = LabeledTextField::new_field_type_string(
                FieldType::String,
                Some("Cores selected: "),
            );
            // Constructor body.
            let expandable: Weak<dyn Expandable> = this.clone();
            let ph_demo =
                PanelHeader::get_advanced_basic_instance_string_expandable_dialog_type_global_expand_button(
                    Some(TITLE),
                    Some(expandable),
                    dialog_type,
                    Some(global_advanced_button.clone()),
                );
            let ftf_file = FileTextField2::get_instance(Some(manager), Some("Choose a file: "));
            let factory = plugin_manager.get_process_result_display_factory();
            let btn_etomo_plugin_demo = factory
                .get_etomo_plugin_demo()
                .as_any_rc()
                .downcast::<Run3dmodButton>()
                .ok()
                .expect("etomoPluginDemo is a Run3dmodButton");
            let btn_unselect_dependencies = factory
                .get_unselect_dependencies()
                .as_any_rc()
                .downcast::<MultiLineButton>()
                .ok()
                .expect("unselectDependencies is a MultiLineButton");
            DemoPanel {
                pnl_root,
                pnl_demo_body,
                cb_sleep_time,
                bg_sleep_time,
                rb_sleep_time2,
                rb_sleep_time3,
                rtf_sleep_time,
                ctf_message,
                cb_swap_yz,
                btn_3dmod,
                ltf_sleep_time_used,
                ltf_cpus,
                manager,
                axis_id,
                plugin_manager: Rc::downgrade(plugin_manager),
                parent,
                ph_demo,
                ftf_file,
                btn_etomo_plugin_demo,
                btn_unselect_dependencies,
                action_listener: OnceCell::new(),
            }
        })
    }

    /// Java public static `getInstance(DemoPluginManager, ApplicationManager, AxisID,
    /// DialogType, GlobalExpandButton, TomogramGenerationDialog)`.
    pub fn get_instance(
        plugin_manager: &Rc<DemoPluginManager>,
        manager: &'static ApplicationManager,
        axis_id: AxisID,
        dialog_type: Option<DialogType>,
        global_advanced_button: &Rc<GlobalExpandButton>,
        parent: Weak<TomogramGenerationDialog>,
    ) -> Rc<DemoPanel> {
        let instance = DemoPanel::new(
            plugin_manager,
            manager,
            axis_id,
            dialog_type,
            global_advanced_button,
            parent,
        );
        instance.create_panel();
        instance.set_tool_tip_text();
        instance.add_listeners();
        instance
    }

    /// Rust-only: Java's `pluginManager` field (the plugin manager owns this panel and
    /// outlives it).
    fn plugin_manager(&self) -> Rc<DemoPluginManager> {
        self.plugin_manager
            .upgrade()
            .expect("the plugin manager owns its panel")
    }

    // <p>Updates done</p>

    /// Java private `createPanel()`.
    fn create_panel(self: &Rc<Self>) {
        // local panels
        let pnl_demo = JComponent::new_panel();
        let pnl_buttons = JComponent::new_panel();
        let pnl_sleep_time = JComponent::new_panel();
        let pnl_file = JComponent::new_panel();
        let pnl_output = JComponent::new_panel();
        // init
        self.rb_sleep_time2.set_selected_boolean(true);
        self.ftf_file
            .set_file_filter(Some(
                Rc::new(Generic3dmodFileFilter::get_instance(Some(self.manager)))
                    as Rc<dyn FileFilter>,
            ));
        // Allow the demo button to run 3dmod via its right-click menu. The 3dmod run
        // command is sent to the action functon in the container.
        let container: Weak<dyn Run3dmodButtonContainer> =
            Rc::downgrade(self) as Weak<dyn Run3dmodButtonContainer>;
        self.btn_etomo_plugin_demo.set_container(Some(container));
        self.btn_etomo_plugin_demo
            .set_deferred_3dmod_button_deferred_3dmod_button(Some(
                self.btn_3dmod.clone() as Rc<dyn Deferred3dmodButton>
            ));
        self.ltf_cpus.set_editable(false);
        self.ltf_sleep_time_used.set_editable(false);
        // Root
        // Swing layout: pnlRoot BoxLayout Y_AXIS; a 0x10 rigid area between pnlDemo
        // and pnlButtons.
        self.pnl_root.add(&pnl_demo);
        self.pnl_root.add(&pnl_buttons);
        // Demo
        // Swing layout: pnlDemo BoxLayout Y_AXIS, etched border.
        pnl_demo.add(&self.ph_demo.get_component());
        pnl_demo.add(&self.pnl_demo_body);
        // DemoBody
        // Swing layout: pnlDemoBody BoxLayout Y_AXIS; 0x2 rigid areas between the rows
        // and a 0x10 one before pnlOutput.
        self.pnl_demo_body.add(&pnl_sleep_time);
        self.pnl_demo_body.add(&self.ctf_message.get_component());
        self.pnl_demo_body.add(&pnl_file);
        self.pnl_demo_body.add(&pnl_output);
        // SleepTime
        // Swing layout: pnlSleepTime BoxLayout X_AXIS.
        pnl_sleep_time.add(&self.cb_sleep_time.get_component());
        pnl_sleep_time.add(&self.rb_sleep_time2.get_component());
        pnl_sleep_time.add(&self.rb_sleep_time3.get_component());
        pnl_sleep_time.add(&self.rtf_sleep_time.get_container());
        // File
        // Swing layout: pnlFile BoxLayout X_AXIS.
        pnl_file.add(&self.ftf_file.get_root_panel());
        pnl_file.add(&self.cb_swap_yz.get_component());
        // Output
        // Swing layout: pnlOutput BoxLayout X_AXIS; a 10x0 rigid area between the two
        // fields.
        pnl_output.add(&self.ltf_sleep_time_used.get_component());
        pnl_output.add(&self.ltf_cpus.get_component());
        // Buttons
        // Swing layout: pnlButtons BoxLayout X_AXIS.
        pnl_buttons.add(&MultiLineButton::get_component(&self.btn_etomo_plugin_demo));
        pnl_buttons.add(&MultiLineButton::get_component(&self.btn_3dmod));
        pnl_buttons.add(&self.btn_unselect_dependencies.get_component());
        PluginPanel::update_display(&**self);
    }

    /// Java private `addListeners()`.
    fn add_listeners(self: &Rc<Self>) {
        let context_menu: Weak<dyn ContextMenu> = Rc::downgrade(self) as Weak<dyn ContextMenu>;
        let mouse_adapter: Rc<dyn MouseListener> = GenericMouseAdapter::new(context_menu);
        self.pnl_root.add_mouse_listener(mouse_adapter);
        // An action listener is so simple that it makes sense for this class to
        // implement it.
        let adaptee = Rc::downgrade(self);
        let listener: ActionListener = Rc::new(move |event: &ActionEvent| {
            if let Some(adaptee) = adaptee.upgrade() {
                adaptee.action_performed(event);
            }
        });
        let _ = self.action_listener.set(listener.clone());
        self.btn_etomo_plugin_demo
            .add_action_listener(listener.clone());
        self.btn_3dmod.add_action_listener(listener.clone());
        self.btn_unselect_dependencies
            .add_action_listener(listener.clone());
        self.cb_sleep_time.add_action_listener(Some(listener));
    }

    /// Java public `msgFieldChanged(boolean)`; empty.
    pub fn msg_field_changed(&self, _different_from_checkpoint: bool) {}

    /// Java public `getParameters(EtomoPluginDemoParam, boolean)`.
    pub fn get_parameters_param(
        &self,
        param: &mut EtomoPluginDemoParam,
        do_validation: bool,
    ) -> bool {
        // try { ... } catch (FieldValidationFailedException e): the validation has its
        // own, built-in message popups.
        let result = (|| {
            if self.cb_sleep_time.is_selected() {
                // Demonstration of the use of EnumeratedType with RadioButton
                // `((RadioButton.RadioButtonModel) bgSleepTime.getSelection())
                // .getEnumeratedType()`.
                let sleep_time = self
                    .bg_sleep_time
                    .get_selection()
                    .and_then(|selection| selection.get_model())
                    .and_then(|model| {
                        model
                            .as_any()
                            .downcast_ref::<RadioButtonModel>()
                            .and_then(AbstractRadioButtonModel::get_enumerated_type)
                    });
                if let Some(sleep_time) = sleep_time {
                    let user_entry =
                        sleep_time.downcast_ref::<SleepTime>() == Some(&SleepTime::UserEntry);
                    // `ConstEtomoNumber value = sleepTime.getValue()`: null only for
                    // USER_ENTRY.
                    let value = sleep_time
                        .downcast_ref::<SleepTime>()
                        .map_or_else(|| Some(sleep_time.get_value()), |s| s.value());
                    if value.is_none() && user_entry {
                        param.set_sleep_time_string(
                            self.rtf_sleep_time
                                .get_text_boolean(do_validation)?
                                .as_deref(),
                        );
                    } else {
                        param.set_sleep_time_const_etomo_number(value.as_ref());
                    }
                }
            } else {
                param.reset_sleep_time();
            }
            if self.ctf_message.is_selected() {
                param.set_message(self.ctf_message.get_text_boolean(do_validation)?.as_deref());
            } else {
                param.reset_message();
            }
            Ok::<bool, crate::imod::etomo::ui::field_validation_failed_exception::FieldValidationFailedException>(true)
        })();
        result.unwrap_or(false)
    }

    /// Java package-private `setParameters(EtomoPluginDemoParam)`.
    pub fn set_parameters_param(&self, param: &EtomoPluginDemoParam) {
        let sleep_time = param.get_sleep_time();
        // `sleepTime == null` cannot happen: `SleepTime.getInstance` never returns null.
        if sleep_time.is_default() {
            self.cb_sleep_time.set_selected_boolean(false);
        } else {
            self.cb_sleep_time.set_selected_boolean(true);
            if sleep_time == SleepTime::Two {
                self.rb_sleep_time2.set_selected_boolean(true);
            } else if sleep_time == SleepTime::Three {
                self.rb_sleep_time3.set_selected_boolean(true);
            } else if sleep_time == SleepTime::UserEntry {
                self.rtf_sleep_time.set_selected_boolean(true);
                self.rtf_sleep_time
                    .set_text_int(param.get_sleep_time_value());
            }
        }
        self.ctf_message
            .set_selected_boolean(param.is_message_set());
        if self.ctf_message.is_selected() {
            self.ctf_message.set_text_string(Some(&param.get_message()));
        }
        PluginPanel::update_display(self);
    }

    /// Java package-private `setSleepTimeUsed(int)`.
    pub fn set_sleep_time_used(&self, sleep_time: i32) {
        self.ltf_sleep_time_used.set_text_int(sleep_time);
    }

    /// Java package-private `getSleepTimeUsed()`.
    pub fn get_sleep_time_used(&self) -> Option<String> {
        Field::get_text_void(&*self.ltf_sleep_time_used)
    }

    /// Java package-private `setCpus(String)`.
    pub fn set_cpus(&self, input: Option<&str>) {
        self.ltf_cpus.set_text_string(input);
    }

    /// Java `actionPerformed(ActionEvent)`.
    fn action_performed(&self, event: &ActionEvent) {
        Run3dmodButtonContainer::action(
            self,
            event.get_action_command().as_deref().unwrap_or(""),
            None,
            None,
        );
    }

    /// Java private `setToolTipText()`.
    fn set_tool_tip_text(&self) {
        let mut autodoc: Option<&dyn ReadOnlyAutodoc> = None;
        match self.plugin_manager().get_autodoc() {
            Ok(instance) => autodoc = instance,
            // catch (final LockException except) {}
            Err(LogFileError::Lock(_)) => {}
            // catch (final LogFileException | IOException except):
            // except.printStackTrace().
            Err(except) => eprintln!("{except}"),
        }
        let tooltip_text =
            etomo_autodoc::get_tooltip(autodoc, Some(etomo_plugin_demo_param::SLEEP_TIME_KEY));
        self.cb_sleep_time
            .set_tool_tip_text_string(tooltip_text.as_deref());
        self.rb_sleep_time2
            .set_tool_tip_text_string(tooltip_text.as_deref());
        self.rb_sleep_time3
            .set_tool_tip_text_string(tooltip_text.as_deref());
        self.rtf_sleep_time
            .set_tool_tip_text(tooltip_text.as_deref());
        self.ctf_message.set_tool_tip_text(
            etomo_autodoc::get_tooltip(autodoc, Some(etomo_plugin_demo_param::MESSAGE_KEY))
                .as_deref(),
        );
        self.ltf_cpus
            .set_tool_tip_text(Some("The cores selected when the process completed"));
    }
}

impl PluginPanel for DemoPanel {
    /// Java `getButtonTitle()`.
    fn get_button_title(&self) -> Option<String> {
        Some(TITLE.to_string())
    }

    /// Java `getComponent()`.
    fn get_component(&self) -> Rc<JComponent> {
        self.pnl_root.clone()
    }

    /// Java `getParameters(BaseScreenState)`.  Created `DemoScreenState` for custom
    /// fields.  This class can be inserted into the manager's screen state, and so does
    /// not require management within the plugin.
    fn get_parameters(&self, screen_state: &BaseScreenState) {
        let demo_screen_state = self.plugin_manager().get_demo_screen_state();
        self.btn_etomo_plugin_demo.set_button_state(
            screen_state
                .get_button_state(self.btn_etomo_plugin_demo.get_button_state_key().as_deref()),
        );
        self.btn_unselect_dependencies.set_button_state(
            screen_state.get_button_state(
                self.btn_unselect_dependencies
                    .get_button_state_key()
                    .as_deref(),
            ),
        );
        self.ph_demo.get_state(
            demo_screen_state
                .as_deref()
                .map(|state| state.get_tomo_gen_demo_header_state()),
        );
    }

    /// Java `setParameters(BaseScreenState)`.
    fn set_parameters(&self, screen_state: &BaseScreenState) {
        let demo_screen_state = self.plugin_manager().get_demo_screen_state();
        self.ph_demo.set_state(
            demo_screen_state
                .as_deref()
                .map(|state| state.get_tomo_gen_demo_header_state() as _),
        );
        self.btn_etomo_plugin_demo.set_button_state(
            screen_state
                .get_button_state(self.btn_etomo_plugin_demo.get_button_state_key().as_deref()),
        );
        self.btn_unselect_dependencies.set_button_state(
            screen_state.get_button_state(
                self.btn_unselect_dependencies
                    .get_button_state_key()
                    .as_deref(),
            ),
        );
    }

    /// Java `updateDisplay()`.
    fn update_display(&self) {
        let advanced = self
            .parent
            .upgrade()
            .is_some_and(|parent| parent.is_advanced());
        self.ctf_message.set_visible(advanced);
        let sleep_time = self.cb_sleep_time.is_selected();
        self.rb_sleep_time2.set_enabled(sleep_time);
        self.rb_sleep_time3.set_enabled(sleep_time);
        self.rtf_sleep_time.set_enabled(sleep_time);
    }

    /// Java `done()`.
    fn done(&self) {
        if let Some(listener) = self.action_listener.get() {
            self.btn_etomo_plugin_demo.remove_action_listener(listener);
            self.btn_unselect_dependencies
                .remove_action_listener(listener);
        }
    }

    /// Java `msgVisibilityChanged(boolean)`.
    fn msg_visibility_changed(&self, visible: bool) {
        self.pnl_root.set_visible(visible);
    }
}

impl ContextMenu for DemoPanel {
    /// Java `popUpContextMenu(MouseEvent)`.  This context menu will override the
    /// dialog's menu.
    fn pop_up_context_menu(&self, mouse_event: &MouseEvent) {
        let mut man_page_label: Vec<String> = vec![String::new(); 1];
        let mut man_page: Vec<String> = vec![String::new(); 1];
        let mut log_file_label: Vec<String> = vec![String::new(); 1];
        let mut log_file: Vec<String> = vec![String::new(); 1];
        let i = 0;
        man_page_label[i] = "3dmod".to_string();
        man_page[i] = "3dmod.html".to_string();
        log_file_label[i] = "EtomoPluginDemo".to_string();
        log_file[i] = format!("demo{}.log", self.axis_id.get_extension());
        let manager: &'static dyn BaseManager = self.manager;
        let _context_popup = ContextPopup::new_component_mouse_event_string_string_string_array_string_array_string_array_string_array_base_manager_axis_id(
            &self.pnl_root,
            mouse_event,
            Some("TOMOGRAM GENERATION"),
            Some(context_popup::TOMO_GUIDE),
            &man_page_label,
            &man_page,
            Some(&log_file_label),
            Some(&log_file),
            manager,
            self.axis_id,
        );
    }
}

impl Expandable for DemoPanel {
    /// Java `expand(GlobalExpandButton)`.  Handled by the panel header, to which the
    /// global advanced button was added.
    fn expand_global_expand_button(&self, _button: &Rc<GlobalExpandButton>) {}

    /// Java `expand(ExpandButton)`.
    fn expand_expand_button(&self, button: &Rc<ExpandButton>) {
        if self.ph_demo.equals_open_close(button) {
            self.pnl_demo_body.set_visible(button.is_expanded());
        } else if self.ph_demo.equals_advanced_basic(button) {
            PluginPanel::update_display(self);
        }
        let manager: &'static dyn BaseManager = self.manager;
        ui_harness::INSTANCE
            .with(|harness| harness.pack_axis_id_base_manager(Some(self.axis_id), Some(manager)));
    }
}

impl Run3dmodButtonContainer for DemoPanel {
    /// Java `action(String, Deferred3dmodButton, Run3dmodMenuOptions)`.
    /// `deferred3dmodButton`: the button assigned to run 3dmod after the process series
    /// associated with this event is complete; `run3dmodMenuOptions`: result of
    /// right-click menu of the button.
    fn action(
        &self,
        command: &str,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    ) {
        if Some(command) == self.btn_etomo_plugin_demo.get_action_command().as_deref() {
            self.ltf_sleep_time_used.set_text_string(Some(""));
            self.plugin_manager().etomo_plugin_demo(
                Some(self.btn_etomo_plugin_demo.clone() as Rc<dyn ProcessResultDisplay>),
                None,
                deferred_3dmod_button,
                run_3dmod_menu_options,
            );
        } else if Some(command) == self.btn_3dmod.get_action_command().as_deref() {
            self.plugin_manager().open_3dmod(
                self.ftf_file.get_file().as_deref(),
                self.cb_swap_yz.is_selected(),
                run_3dmod_menu_options,
            );
        } else if Some(command)
            == self
                .btn_unselect_dependencies
                .get_action_command()
                .as_deref()
        {
            self.plugin_manager().unselect_dependencies(
                Some(self.btn_unselect_dependencies.clone() as Rc<dyn ProcessResultDisplay>),
                None,
            );
        } else {
            PluginPanel::update_display(self);
        }
    }
}

impl UIComponent for DemoPanel {
    /// Java `getUIComponent()`.  Implementing SwingComponent and UIComponent allows
    /// UIHarness and Popup to pop up messages without using the manager and the axisID.
    fn get_ui_component(&self) -> &dyn SwingComponent {
        self
    }

    /// Java `getComponent()`.
    fn get_component(&self) -> Rc<JComponent> {
        self.pnl_root.clone()
    }
}

impl SwingComponent for DemoPanel {
    /// Java `getComponent()`.
    fn get_component(&self) -> Rc<JComponent> {
        self.pnl_root.clone()
    }
}
