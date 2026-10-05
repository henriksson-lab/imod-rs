//! `IMOD/Etomo/src/etomo/ui/swing/SettingsDialog.java`.
//!
//! The Options > Settings dialog: font, tooltip delays, appearance flags, the
//! enhanced-local-processing (CPU/GPU) fields, the dataset defaults, the table
//! sizes, the user template directory and the default templates.  It reads
//! and writes the process-wide `etomo.type.UserConfiguration`
//! (`src/imod/etomo/type/user_configuration.rs`), which `EtomoDirector` owns.
//!
//! Java `final class SettingsDialog extends JDialog`: the dialog is the Swing
//! stand-in's `jdk::JDialog` (content pane, title, visibility; the Slint window
//! draws a showing one over the frame), with the modality and location the Java
//! sets kept here.  Layout (`BoxLayout`, rigid areas, glue, scroll
//! bar increments, `pack`) is recorded as `// Swing layout:` comments.
//!
//! Object model (ui.md): the dialog is an `Rc<SettingsDialog>` living on the
//! event dispatch thread; every method takes `&self`.  The Java constructor
//! hands `this` to `TemplatePanel.getInstance`, whose Rust form stores an
//! `Rc<SettingsDialog>`; so the panel is created right after the `Rc` and kept
//! in a `OnceCell`.  That is a reference cycle, the same object graph as the
//! Java's; the director never releases its settings dialog either
//! (`EtomoDirector.settingsDialog` is only ever assigned once).
//!
//! JVM-only facts with no Rust counterpart:
//! * `GraphicsEnvironment.getAvailableFontFamilyNames()` (in `FontFamilies`):
//!   the JVM lists the host's font families plus its five logical families.
//!   Only the logical families, which every JVM returns, are listed here; the
//!   Slint frontend renders with its own fonts.
//! * `UIManager.getDefaults()` (in `setParameters`): the first `FontUIResource`
//!   in the Swing defaults is the font `EtomoDirector.setUIFont` installed from
//!   the user configuration, so its family and size are read from the user
//!   configuration.

use std::cell::{Cell, OnceCell, RefCell};
use std::path::{Path, PathBuf};
use std::rc::{Rc, Weak};

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::jdk::{ActionEvent, ActionListener, JComponent, Point};
use crate::imod::etomo::logic::config_tool;
use crate::imod::etomo::storage::cpu_adoc;
use crate::imod::etomo::storage::network::Network;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::{
    Number, java_lang_double_value_of, java_lang_integer_parse_int,
};
use crate::imod::etomo::r#type::tilt_angle_type::TiltAngleType;
use crate::imod::etomo::r#type::user_configuration::UserConfiguration;
use crate::imod::etomo::ui::field::Field;
use crate::imod::etomo::ui::field_type::FieldType;
use crate::imod::etomo::ui::swing::check_box::CheckBox;
use crate::imod::etomo::ui::swing::etched_border::EtchedBorder;
use crate::imod::etomo::ui::swing::etomo_panel::EtomoPanel;
use crate::imod::etomo::ui::swing::file_chooser::DIRECTORIES_ONLY;
use crate::imod::etomo::ui::swing::file_text_field_interface::FileTextFieldInterface;
use crate::imod::etomo::ui::swing::file_text_field2::FileTextField2;
use crate::imod::etomo::ui::swing::labeled_text_field::LabeledTextField;
use crate::imod::etomo::ui::swing::parallel_panel;
use crate::imod::etomo::ui::swing::setup_dialog;
use crate::imod::etomo::ui::swing::spaced_panel::{self, SpacedPanel};
use crate::imod::etomo::ui::swing::template_panel::TemplatePanel;
use crate::imod::etomo::ui::swing::trimvol_panel;

/// `javax.swing.Box.LEFT_ALIGNMENT`.
const LEFT_ALIGNMENT: f32 = 0.0;
/// `javax.swing.Box.CENTER_ALIGNMENT`.
const CENTER_ALIGNMENT: f32 = 0.5;

/// Java `public final class SettingsDialog extends JDialog`.
pub struct SettingsDialog {
    // --- javax.swing.JDialog state ---
    /// The `JDialog` itself (content pane, title, visibility).
    dialog: Rc<crate::imod::etomo::jdk::JDialog>,
    /// `Dialog.setModal`.
    modal: Cell<bool>,
    /// `Component.setLocation`.
    location: Cell<Point>,

    // --- Java fields, in declaration order ---
    /// Java private final `fontFamilies`.  Font selection panel.
    font_families: FontFamilies,
    /// Java private final `listFontFamily`, a `JList` over the usable font
    /// families.  The stand-in is a plain node carrying the item list and the
    /// selected index.
    list_font_family: Rc<JComponent>,
    /// Java private final `ltfFontSize`.
    ltf_font_size: Rc<LabeledTextField>,
    /// Java private final `ltfTooltipsInitialDelay`.
    ltf_tooltips_initial_delay: Rc<LabeledTextField>,
    /// Java private final `ltfTooltipsDismissDelay`.
    ltf_tooltips_dismiss_delay: Rc<LabeledTextField>,
    /// Java private final `cbNativeLAF`.
    cb_native_laf: Rc<CheckBox>,
    /// Java private final `cbAdvancedDialogs`.
    cb_advanced_dialogs: Rc<CheckBox>,
    /// Java private final `cbAutoFit`.
    cb_auto_fit: Rc<CheckBox>,
    /// Java private final `cbCompactDisplay`.
    cb_compact_display: Rc<CheckBox>,
    /// Java private final `buttonCancel`.
    button_cancel: Rc<JComponent>,
    /// Java private final `buttonApply`.
    button_apply: Rc<JComponent>,
    /// Java private final `buttonDone`.
    button_done: Rc<JComponent>,
    /// Java private final `cbParallelProcessing`.
    cb_parallel_processing: Rc<CheckBox>,
    /// Java private final `cbGpuProcessing`.
    cb_gpu_processing: Rc<CheckBox>,
    /// Java private final `ltfCpus`.
    ltf_cpus: Rc<LabeledTextField>,
    /// Java private final `ltfNumberOfLocalGPUs`.
    ltf_number_of_local_gpus: Rc<LabeledTextField>,
    /// Java private final `cbSingleAxis`.
    cb_single_axis: Rc<CheckBox>,
    /// Java private final `cbMontage`.
    cb_montage: Rc<CheckBox>,
    /// Java private final `cbNoParallelProcessing`.
    cb_no_parallel_processing: Rc<CheckBox>,
    /// Java private final `cbGpuProcessingDefault`.
    cb_gpu_processing_default: Rc<CheckBox>,
    /// Java private final `cbTiltAnglesRawtltFile`.
    cb_tilt_angles_rawtlt_file: Rc<CheckBox>,
    /// Java private final `cbSwapYAndZ`.
    cb_swap_y_and_z: Rc<CheckBox>,
    /// Java private final `ltfParallelTableSize`.
    ltf_parallel_table_size: Rc<LabeledTextField>,
    /// Java private final `ltfJoinTableSize`.
    ltf_join_table_size: Rc<LabeledTextField>,
    /// Java private final `ltfPeetTableSize`.
    ltf_peet_table_size: Rc<LabeledTextField>,
    /// Java private final `ltfBatchTableSize`.
    ltf_batch_table_size: Rc<LabeledTextField>,
    /// Java private final `cbSetFEIPixelSize`.
    cb_set_fei_pixel_size: Rc<CheckBox>,
    /// Java private final `ftfUserTemplateDir`.
    ftf_user_template_dir: Rc<FileTextField2>,
    /// Java private final `listener`, a `SettingsDialogListener`.
    listener: ActionListener,
    /// Java private final `ltfSmtpServer`.
    ltf_smtp_server: Rc<LabeledTextField>,
    /// Java private final `cbRemoveExcludedViews`.
    cb_remove_excluded_views: Rc<CheckBox>,
    /// Java private final `cpuAdocViable = CpuAdoc.INSTANCE.isViable()`.
    cpu_adoc_viable: bool,

    /// Java private final `templatePanel`, assigned in the constructor (see the
    /// module comment for the `OnceCell`).
    template_panel: OnceCell<Rc<TemplatePanel>>,
    /// Java private final `propertyUserDir`.
    property_user_dir: Option<String>,
    /// Java private final `manager`.  Non-null here: the translated
    /// `TemplatePanel` (and the `DirectiveFileCollection` it builds) requires a
    /// manager, so `EtomoDirector.openSettingsDialog` only builds the dialog
    /// for a current manager.
    manager: &'static dyn BaseManager,
}

impl SettingsDialog {
    /// Java private `SettingsDialog(BaseManager, String)`, with the field
    /// initializers.
    fn new(
        manager: &'static dyn BaseManager,
        property_user_dir: Option<&str>,
    ) -> Rc<SettingsDialog> {
        let instance = Rc::new_cyclic(|this: &Weak<SettingsDialog>| {
            let font_families = FontFamilies::new();
            // new JList(fontFamilies.getFontFamilies())
            let list_font_family = JComponent::new_other();
            for family in font_families.get_font_families() {
                list_font_family.add_item(family);
            }
            // A JList starts with nothing selected; the stand-in's item model
            // selects the first item added, so clear it.
            list_font_family.set_selected_index(-1);
            // new SettingsDialogListener(this)
            let adaptee = this.clone();
            let listener: ActionListener = Rc::new(move |action_event: &ActionEvent| {
                // SettingsDialogListener.actionPerformed
                if let Some(adaptee) = adaptee.upgrade() {
                    adaptee.action(action_event.get_action_command());
                }
            });
            SettingsDialog {
                // A JDialog is created invisible (and, with `super()`, not modal).
                dialog: crate::imod::etomo::jdk::JDialog::new("", false),
                modal: Cell::new(false),
                location: Cell::new(Point { x: 0, y: 0 }),
                font_families,
                list_font_family,
                ltf_font_size: LabeledTextField::new_field_type_string(
                    FieldType::Integer,
                    Some("Size: "),
                ),
                ltf_tooltips_initial_delay: LabeledTextField::new_field_type_string(
                    FieldType::FloatingPoint,
                    Some("Tooltips initial delay: "),
                ),
                ltf_tooltips_dismiss_delay: LabeledTextField::new_field_type_string(
                    FieldType::FloatingPoint,
                    Some("Tooltips dismiss delay: "),
                ),
                cb_native_laf: CheckBox::new_string(Some("Native look & feel")),
                cb_advanced_dialogs: CheckBox::new_string(Some("Always use advanced dialogs")),
                cb_auto_fit: CheckBox::new_string(Some("Auto-fit")),
                cb_compact_display: CheckBox::new_string(Some("Compact Display")),
                button_cancel: JComponent::new_button("Cancel"),
                button_apply: JComponent::new_button("Apply"),
                button_done: JComponent::new_button("Done"),
                cb_parallel_processing: CheckBox::new_string(Some("Enable parallel processing")),
                cb_gpu_processing: CheckBox::new_string(Some("Enable graphics processing")),
                ltf_cpus: LabeledTextField::new_field_type_string(
                    FieldType::Integer,
                    Some("# of true CPU cores: "),
                ),
                ltf_number_of_local_gpus: LabeledTextField::new_field_type_string(
                    FieldType::Integer,
                    Some("# of local GPUs: "),
                ),
                cb_single_axis: CheckBox::new_string(Some(
                    format!(
                        "{}:  {}",
                        setup_dialog::AXIS_TYPE_LABEL,
                        setup_dialog::SINGLE_AXIS_LABEL
                    )
                    .as_str(),
                )),
                cb_montage: CheckBox::new_string(Some(
                    format!(
                        "{}:  {}",
                        setup_dialog::FRAME_TYPE_LABEL,
                        setup_dialog::MONTAGE_LABEL
                    )
                    .as_str(),
                )),
                cb_no_parallel_processing: CheckBox::new_string(Some(
                    format!("Start with {} off", parallel_panel::FIELD_LABEL).as_str(),
                )),
                cb_gpu_processing_default: CheckBox::new_string(Some(
                    "Start with graphics card processing on",
                )),
                cb_tilt_angles_rawtlt_file: CheckBox::new_string(Some(
                    format!("Angle Source:  {}", TiltAngleType::File.get_descr()).as_str(),
                )),
                cb_swap_y_and_z: CheckBox::new_string(Some(
                    format!(
                        "{}  {}",
                        trimvol_panel::REORIENTATION_GROUP_LABEL,
                        trimvol_panel::SWAP_YZ_LABEL
                    )
                    .as_str(),
                )),
                ltf_parallel_table_size: LabeledTextField::new_field_type_string(
                    FieldType::Integer,
                    Some("Parallel table size: "),
                ),
                ltf_join_table_size: LabeledTextField::new_field_type_string(
                    FieldType::Integer,
                    Some("Join tables size: "),
                ),
                ltf_peet_table_size: LabeledTextField::new_field_type_string(
                    FieldType::Integer,
                    Some("PEET table size: "),
                ),
                ltf_batch_table_size: LabeledTextField::new_field_type_string(
                    FieldType::Integer,
                    Some("Batchruntomo table size: "),
                ),
                cb_set_fei_pixel_size: CheckBox::new_string(Some(
                    "Set pixel size from mdoc file or in file from FEI",
                )),
                ftf_user_template_dir: FileTextField2::get_instance(
                    None,
                    Some("User templates directory: "),
                ),
                listener,
                ltf_smtp_server: LabeledTextField::new_field_type_string(
                    FieldType::String,
                    Some("Outgoing mail server: "),
                ),
                cb_remove_excluded_views: CheckBox::new_string(Some(
                    "Default to removing excluded views",
                )),
                cpu_adoc_viable: cpu_adoc::INSTANCE.is_viable(),
                template_panel: OnceCell::new(),
                // this.propertyUserDir = propertyUserDir;
                property_user_dir: property_user_dir.map(str::to_owned),
                // this.manager = manager;
                manager,
            }
        });
        // templatePanel = TemplatePanel.getInstance(manager, AxisID.ONLY, listener,
        // "Default Templates", this, false);
        let template_panel = TemplatePanel::get_instance(
            manager,
            AxisID::Only,
            Rc::clone(&instance.listener),
            Some("Default Templates"),
            Some(Rc::clone(&instance)),
            false,
        );
        let _ = instance.template_panel.set(template_panel);
        instance
    }

    /// Java `templatePanel` field read.
    fn template_panel(&self) -> &Rc<TemplatePanel> {
        self.template_panel
            .get()
            .expect("SettingsDialog.templatePanel is assigned by the constructor")
    }

    /// Java static `getInstance(BaseManager, String)`.
    pub fn get_instance(
        manager: &'static dyn BaseManager,
        property_user_dir: Option<&str>,
    ) -> Rc<SettingsDialog> {
        let instance = SettingsDialog::new(manager, property_user_dir);
        instance.build_dialog();
        instance.set_tooltips();
        instance.add_listeners();
        instance
    }

    /// Java private `buildDialog()`.
    fn build_dialog(&self) {
        // panels
        let pnl_gpu_processing = JComponent::new_panel();
        let pnl_number_of_local_gpus = JComponent::new_panel();
        // init
        self.ftf_user_template_dir.set_absolute_path(true);
        self.ftf_user_template_dir
            .set_use_text_as_file_chooser_dir(true);
        self.ftf_user_template_dir
            .set_file_selection_mode(DIRECTORIES_ONLY);
        self.ftf_user_template_dir
            .set_file(config_tool::get_default_user_template_dir());
        self.ftf_user_template_dir.set_turn_off_file_hiding(true);
        self.set_title("Etomo Settings");
        let pnl_main = SpacedPanel::get_instance_void();
        // Scroll bar
        let scroll_pane = JComponent::new_scroll_pane(Some(&pnl_main.get_j_panel()));
        self.set_scroll_bar_increments(&scroll_pane);
        self.set_scroll_bar_increments(&scroll_pane);
        // Swing layout: scrollPane horizontal and vertical scroll bar policies
        // AS_NEEDED.
        self.dialog.get_content_pane().add(&scroll_pane);
        pnl_main.set_box_layout(spaced_panel::Y_AXIS);
        // Layout the font panel
        let panel_font_select = SpacedPanel::get_instance_void();
        panel_font_select.set_box_layout(spaced_panel::X_AXIS);
        panel_font_select.add_j_label(&JComponent::new_label("Font family:"));
        // Swing layout: listFontFamily single selection mode, 3 visible rows;
        // ltfFontSize.setColumns(3).
        self.ltf_font_size.set_columns(3);
        let scroll_font_family = JComponent::new_scroll_pane(Some(&self.list_font_family));
        panel_font_select.add_j_scroll_pane(&scroll_font_family);
        panel_font_select.add_container(&self.ltf_font_size.get_container());
        pnl_main.add_spaced_panel(&panel_font_select);
        pnl_main.add_container(&self.ltf_tooltips_initial_delay.get_container());
        pnl_main.add_container(&self.ltf_tooltips_dismiss_delay.get_container());
        // Swing layout: pnlMain.add(Box.createRigidArea(FixedDim.x0_y5)).
        pnl_main.add_labeled_text_field(&self.ltf_smtp_server);
        // Swing layout: pnlMain.add(Box.createRigidArea(FixedDim.x0_y5)).
        // Settings
        let pnl_settings = SpacedPanel::get_instance_void();
        pnl_settings.set_box_layout(spaced_panel::X_AXIS);
        pnl_settings.set_component_alignment_x(LEFT_ALIGNMENT);
        pnl_main.add_spaced_panel(&pnl_settings);
        let pnl_general_settings = JComponent::new_panel();
        // Swing layout: pnlGeneralSettings BoxLayout Y_AXIS.
        pnl_settings.add_j_panel(&pnl_general_settings);
        pnl_general_settings.add(&self.cb_auto_fit.get_component());
        // TEMP bug# 614
        self.cb_auto_fit.set_enabled(false);
        // TEMP
        pnl_general_settings.add(&self.cb_native_laf.get_component());
        pnl_general_settings.add(&self.cb_advanced_dialogs.get_component());
        pnl_general_settings.add(&self.cb_compact_display.get_component());
        // enhanced processing settings
        let pnl_enhanced_processing = EtomoPanel::new();
        // Swing layout: pnlEnhancedProcessing BoxLayout Y_AXIS.
        pnl_enhanced_processing
            .set_border(&EtchedBorder::new(Some("Enhanced Local Processing")).get_border());
        pnl_settings.add_j_panel(&pnl_enhanced_processing.get_component());
        let pnl_check_box_parallel_processing = JComponent::new_panel();
        // Swing layout: pnlCheckBoxParallelProcessing BoxLayout X_AXIS, horizontal
        // glue.
        pnl_check_box_parallel_processing.add(&self.cb_parallel_processing.get_component());
        pnl_enhanced_processing
            .get_component()
            .add(&pnl_check_box_parallel_processing);
        pnl_enhanced_processing
            .get_component()
            .add(&self.ltf_cpus.get_container());
        pnl_enhanced_processing
            .get_component()
            .add(&pnl_gpu_processing);
        pnl_enhanced_processing
            .get_component()
            .add(&pnl_number_of_local_gpus);
        // GpuProcessing
        // Swing layout: pnlGpuProcessing BoxLayout X_AXIS, horizontal glue.
        pnl_gpu_processing.add(&self.cb_gpu_processing.get_component());
        // NumberOfLocalGPUs
        // Swing layout: pnlNumberOfLocalGPUs BoxLayout X_AXIS, horizontal glue.
        pnl_number_of_local_gpus.add(&self.ltf_number_of_local_gpus.get_component());
        // default settings
        let panel_defaults = SpacedPanel::get_instance_void();
        panel_defaults.set_box_layout(spaced_panel::Y_AXIS);
        panel_defaults.set_component_alignment_x(LEFT_ALIGNMENT);
        panel_defaults.set_border(&EtchedBorder::new(Some("Defaults")).get_border());
        panel_defaults.add_check_box(&self.cb_single_axis);
        panel_defaults.add_check_box(&self.cb_montage);
        panel_defaults.add_check_box(&self.cb_no_parallel_processing);
        panel_defaults.add_check_box(&self.cb_gpu_processing_default);
        panel_defaults.add_component(&self.cb_remove_excluded_views.get_component());
        panel_defaults.add_check_box(&self.cb_tilt_angles_rawtlt_file);
        panel_defaults.add_check_box(&self.cb_swap_y_and_z);
        panel_defaults.add_check_box(&self.cb_set_fei_pixel_size);
        pnl_main.add_container(&panel_defaults.get_container());
        // table settings
        let pnl_table_size = EtomoPanel::new();
        // Swing layout: pnlTableSize BoxLayout Y_AXIS.
        pnl_table_size.set_border(&EtchedBorder::new(Some("Table Sizes")).get_border());
        pnl_table_size
            .get_component()
            .add(&self.ltf_parallel_table_size.get_container());
        pnl_table_size
            .get_component()
            .add(&self.ltf_join_table_size.get_container());
        pnl_table_size
            .get_component()
            .add(&self.ltf_peet_table_size.get_container());
        pnl_table_size
            .get_component()
            .add(&self.ltf_batch_table_size.get_container());
        pnl_main.add_j_panel(&pnl_table_size.get_component());
        pnl_main.add_component(&self.ftf_user_template_dir.get_root_panel());
        pnl_main.add_component(&self.template_panel().get_component());
        // buttons
        let panel_buttons = SpacedPanel::get_instance_void();
        panel_buttons.set_box_layout(spaced_panel::X_AXIS);
        panel_buttons.set_alignment_x(CENTER_ALIGNMENT);
        panel_buttons.add_j_button(&self.button_cancel);
        panel_buttons.add_j_button(&self.button_apply);
        panel_buttons.add_j_button(&self.button_done);
        pnl_main.add_spaced_panel(&panel_buttons);
        // Swing layout: pack().
    }

    /// Java private final `setScrollBarIncrements(JScrollBar)`.  The stand-in
    /// has no scroll bar node; the Java is called with the scroll pane's
    /// vertical and horizontal bars.
    fn set_scroll_bar_increments(&self, _scroll_bar: &Rc<JComponent>) {
        // Swing layout: scrollBar.setUnitIncrement(10); scrollBar.setBlockIncrement(50).
    }

    /// Java private `setParametersFromNetwork()`.  Modified parallel processing
    /// panel settings.  If parallel processing is being handled by the cpu.adoc
    /// or the IMOD_PROCESSORS environment variable, disable the option and set
    /// it to match how the local host is being used.  Also do this for GPUs.
    ///
    /// Returns true if the CPUs text field was set by this function.
    fn set_parameters_from_network(&self) -> bool {
        let axis_id = AxisID::Only;
        if !Network::is_parallel_processing_set_externally(
            self.manager,
            axis_id,
            self.property_user_dir.as_deref(),
        ) {
            // There is no cpu.adoc, and no IMOD_PROCESSORS environment variable. So CPU
            // and GPU settings for this machine can be controlled by this panel.
            return false;
        }
        self.cb_parallel_processing.set_enabled(false);
        self.cb_parallel_processing.set_selected_boolean(true);
        // Disabling the GPU checkbox is handled in updateDisplay.
        let local_host =
            Network::get_local_host(self.manager, axis_id, self.property_user_dir.as_deref());
        if let Some(local_host) = local_host {
            // CPUs
            let cpus = local_host.get_cpus();
            let mut local_host_cpus: Option<i32> = None;
            if !cpus.is_null() {
                local_host_cpus = Some(cpus.get_int());
            }
            if let Some(local_host_cpus) = local_host_cpus {
                // setText(Integer) resolves to setText(Number).
                self.ltf_cpus
                    .set_text_number(Some(Number::Integer(local_host_cpus)));
            } else {
                self.ltf_cpus.set_text_string(Some(""));
            }
            // GPUs
            let local_host_gpus =
                local_host.get_total_gpus(self.manager, axis_id, self.property_user_dir.as_deref());
            if local_host_gpus > 0 {
                self.cb_gpu_processing.set_selected_boolean(true);
                self.ltf_number_of_local_gpus.set_text_int(local_host_gpus);
            } else {
                self.cb_gpu_processing.set_selected_boolean(false);
                if self.cpu_adoc_viable {
                    self.ltf_number_of_local_gpus.set_text_string(Some(""));
                }
            }
        } else if self.cpu_adoc_viable {
            // If there is a cpu.adoc but no local host, uncheck and blank out this panel
            // (which only refers to the local host).
            self.cb_parallel_processing.set_selected_boolean(false);
            self.ltf_cpus.set_text_string(Some(""));
            self.cb_gpu_processing.set_selected_boolean(false);
            self.ltf_number_of_local_gpus.set_text_string(Some(""));
        }
        true
    }

    /// Java private `updateDisplay()`.
    fn update_display(&self) {
        self.ltf_cpus.set_enabled(
            self.cb_parallel_processing.is_enabled() && self.cb_parallel_processing.is_selected(),
        );
        // When IMOD_PROCESSORS is in use, the GPUs can still be set from this dialog.
        self.cb_gpu_processing
            .set_enabled(!self.cpu_adoc_viable && self.cb_parallel_processing.is_selected());
        self.ltf_number_of_local_gpus.set_enabled(
            self.cb_parallel_processing.is_selected()
                && self.cb_gpu_processing.is_enabled()
                && self.cb_gpu_processing.is_selected(),
        );
    }

    /// Java private `addListeners()`.
    fn add_listeners(&self) {
        self.button_cancel
            .add_action_listener(Rc::clone(&self.listener));
        self.button_apply
            .add_action_listener(Rc::clone(&self.listener));
        self.button_done
            .add_action_listener(Rc::clone(&self.listener));
        self.cb_parallel_processing
            .add_action_listener(Some(Rc::clone(&self.listener)));
        self.cb_gpu_processing
            .add_action_listener(Some(Rc::clone(&self.listener)));
    }

    /// Java `setParameters(UserConfiguration)`.
    pub fn set_parameters(&self, user_config: &UserConfiguration) {
        // Convert the tooltips times to seconds
        self.ltf_tooltips_initial_delay
            .set_text_int(user_config.get_tool_tips_initial_delay() / 1000);
        self.ltf_tooltips_dismiss_delay
            .set_text_int(user_config.get_tool_tips_dismiss_delay() / 1000);
        self.cb_auto_fit
            .set_selected_boolean(user_config.is_auto_fit());
        self.cb_native_laf
            .set_selected_boolean(user_config.get_native_look_and_feel());
        self.cb_advanced_dialogs
            .set_selected_boolean(user_config.get_advanced_dialogs());
        self.cb_compact_display
            .set_selected_boolean(user_config.get_compact_display());
        self.cb_single_axis
            .set_selected_boolean(user_config.get_single_axis());
        self.cb_montage
            .set_selected_boolean(user_config.get_montage());
        self.cb_no_parallel_processing
            .set_selected_boolean(user_config.get_no_parallel_processing());
        self.cb_gpu_processing_default
            .set_selected_boolean(user_config.get_gpu_processing_default());
        self.cb_remove_excluded_views
            .set_selected_boolean(user_config.is_remove_excluded_views());
        self.cb_tilt_angles_rawtlt_file
            .set_selected_boolean(user_config.is_tilt_angles_rawtlt_file());
        self.cb_swap_y_and_z
            .set_selected_boolean(user_config.get_swap_y_and_z());
        self.cb_set_fei_pixel_size
            .set_selected_boolean(user_config.is_set_fei_pixel_size());
        self.cb_parallel_processing
            .set_selected_boolean(user_config.is_parallel_processing());
        self.cb_gpu_processing
            .set_selected_boolean(user_config.is_gpu_processing());
        self.ltf_cpus
            .set_text_const_etomo_number(Some(user_config.get_cpus()));
        self.ltf_number_of_local_gpus
            .set_text_const_etomo_number(Some(user_config.get_local_gpus()));
        self.ltf_parallel_table_size
            .set_text_const_etomo_number(Some(user_config.get_parallel_table_size()));
        self.ltf_join_table_size
            .set_text_const_etomo_number(Some(user_config.get_join_table_size()));
        self.ltf_peet_table_size
            .set_text_const_etomo_number(Some(user_config.get_peet_table_size()));
        self.ltf_batch_table_size
            .set_text_const_etomo_number(Some(user_config.get_batch_table_size()));
        self.ltf_smtp_server
            .set_text_string(user_config.get_smtp_server().as_deref());
        let dir = user_config.get_user_template_dir();
        // dir != null && !dir.matches("\\s*")
        if let Some(dir) = &dir
            && !dir
                .chars()
                .all(|c| matches!(c, ' ' | '\t' | '\n' | '\x0B' | '\x0C' | '\r'))
        {
            self.ftf_user_template_dir
                .set_text_string(user_config.get_user_template_dir().as_deref());
        }
        self.template_panel()
            .set_parameters_user_configuration(user_config);

        // Get the current font parameters to set the UI
        // Since they may not be all the same make the assumption that the first
        // object contains the font
        // UIManager.getDefaults(): the first FontUIResource is the font that
        // EtomoDirector.setUIFont installed from the user configuration (see the
        // module comment).
        let current_font_family: String = user_config.get_font_family().unwrap_or_default();
        let current_font_size: i32 = user_config.get_font_size();

        self.list_font_family
            .set_selected_index(self.font_families.get_index(&current_font_family));
        self.ltf_font_size.set_text_int(current_font_size);

        // Order of precedence for parallel processing fields:
        // 1. cpu.adoc
        // 2. IMOD_PROCESSORS
        // 3. This dialog
        // Attempt to load from cpu.adoc or IMOD_PROCESSORS
        if !self.set_parameters_from_network() && !self.cb_parallel_processing.is_selected() {
            // Parallel processing was not set by either the system or the user. Get the
            // number of cores from imodqtassist.
            let physical_cores = self.manager.get_physical_cores(Some(AxisID::Only));
            if let Some(physical_cores) = physical_cores {
                // setText(Integer) resolves to setText(Number).
                self.ltf_cpus
                    .set_text_number(Some(Number::Integer(physical_cores)));
            }
        }

        self.update_display();
    }

    /// Java package-private `equalsUserTemplateDir(File)`.
    pub fn equals_user_template_dir(&self, input: Option<&Path>) -> bool {
        let user_template_dir = self.ftf_user_template_dir.get_file();
        if user_template_dir.is_none() && input.is_none() {
            return true;
        }
        let Some(user_template_dir) = user_template_dir else {
            return false;
        };
        // File.equals(null) is false.
        input.is_some_and(|input| user_template_dir.as_path() == input)
    }

    /// Java package-private `getUserTemplateDir()`.
    pub fn get_user_template_dir(&self) -> Option<PathBuf> {
        self.ftf_user_template_dir.get_file()
    }

    /// Java `getParameters(UserConfiguration)`.
    ///
    /// Upstream bug fixed in translation (SettingsDialog.java:397, 400, 406, 407):
    /// `Double.parseDouble`/`Integer.parseInt` throw `NumberFormatException` on
    /// a delay or font size that is not a number (an empty field included), and
    /// `fontFamilies.getName(-1)` throws `ArrayIndexOutOfBoundsException` when no
    /// font family is selected (the default family "Dialog" missing).  The
    /// exception escapes to the event dispatch thread, abandoning the rest of
    /// the update with the earlier settings already changed.  Here a value that
    /// does not parse, or no selected family, leaves that one setting as it was
    /// and the rest are still read.
    pub fn get_parameters(&self, user_config: &mut UserConfiguration) {
        // Convert the tooltips times to milliseconds
        if let Ok(delay) = java_lang_double_value_of(
            &Field::get_text_void(&*self.ltf_tooltips_initial_delay).unwrap_or_default(),
        ) {
            user_config.set_tool_tips_initial_delay((delay * 1000.0) as i32);
        }

        if let Ok(delay) = java_lang_double_value_of(
            &Field::get_text_void(&*self.ltf_tooltips_dismiss_delay).unwrap_or_default(),
        ) {
            user_config.set_tool_tips_dismiss_delay((delay * 1000.0) as i32);
        }
        user_config.set_auto_fit(self.cb_auto_fit.is_selected());
        user_config.set_native_look_and_feel(self.cb_native_laf.is_selected());
        user_config.set_advanced_dialogs(self.cb_advanced_dialogs.is_selected());
        user_config.set_compact_display(self.cb_compact_display.is_selected());
        if let Ok(font_size) = java_lang_integer_parse_int(
            &Field::get_text_void(&*self.ltf_font_size).unwrap_or_default(),
        ) {
            user_config.set_font_size(font_size);
        }
        if let Some(font_family) = self
            .font_families
            .get_name(self.list_font_family.get_selected_index())
        {
            user_config.set_font_family(Some(font_family));
        }
        user_config.set_single_axis(self.cb_single_axis.is_selected());
        user_config.set_montage(self.cb_montage.is_selected());
        user_config.set_no_parallel_processing(self.cb_no_parallel_processing.is_selected());
        user_config.set_gpu_processing_default(self.cb_gpu_processing_default.is_selected());
        user_config.set_remove_excluded_views(self.cb_remove_excluded_views.is_selected());
        user_config.set_tilt_angles_rawtlt_file(self.cb_tilt_angles_rawtlt_file.is_selected());
        user_config.set_swap_y_and_z(self.cb_swap_y_and_z.is_selected());
        user_config.set_set_fei_pixel_size(self.cb_set_fei_pixel_size.is_selected());
        user_config.set_parallel_processing(self.cb_parallel_processing.is_selected());
        user_config.set_gpu_processing(self.cb_gpu_processing.is_selected());
        user_config.set_cpus(Field::get_text_void(&*self.ltf_cpus).as_deref());
        user_config
            .set_local_gpus(Field::get_text_void(&*self.ltf_number_of_local_gpus).as_deref());
        user_config.set_parallel_table_size(
            Field::get_text_void(&*self.ltf_parallel_table_size).as_deref(),
        );
        user_config
            .set_join_table_size(Field::get_text_void(&*self.ltf_join_table_size).as_deref());
        user_config
            .set_peet_table_size(Field::get_text_void(&*self.ltf_peet_table_size).as_deref());
        user_config
            .set_batch_table_size(Field::get_text_void(&*self.ltf_batch_table_size).as_deref());
        user_config.set_user_template_dir(self.ftf_user_template_dir.get_file().as_deref());
        user_config.set_smtp_server(Field::get_text_void(&*self.ltf_smtp_server).as_deref());
        self.template_panel().get_parameters(user_config);
    }

    /// Java `isAppearanceSettingChanged(UserConfiguration)`.
    ///
    /// Upstream bugs fixed in translation (SettingsDialog.java:443-454):
    /// `Integer.parseInt(ltfFontSize.getText())` throws on a font size that is
    /// not a number, `fontFamilies.getName(-1)` throws with no selected family,
    /// and `userConfig.getFontFamily().equals(...)` /
    /// `userConfig.getSmtpServer().toString()` throw `NullPointerException` for
    /// an unset family or server; each exception escapes to the event dispatch
    /// thread and the Apply/Done action is abandoned.  Here an unparsable font
    /// size, a missing selection or a null family counts as a change (the
    /// restart notice is shown and `getParameters` then keeps the old value),
    /// and a null server compares as an empty one.
    pub fn is_appearance_setting_changed(&self, user_config: &UserConfiguration) -> bool {
        if self
            .template_panel()
            .is_appearance_setting_changed(user_config)
        {
            return true;
        }
        let font_size = java_lang_integer_parse_int(
            &Field::get_text_void(&*self.ltf_font_size).unwrap_or_default(),
        );
        let font_family = self
            .font_families
            .get_name(self.list_font_family.get_selected_index());
        if user_config.get_native_look_and_feel() != self.cb_native_laf.is_selected()
            || user_config.get_compact_display() != self.cb_compact_display.is_selected()
            || user_config.get_single_axis() != self.cb_single_axis.is_selected()
            || user_config.get_montage() != self.cb_montage.is_selected()
            || user_config.get_no_parallel_processing()
                != self.cb_no_parallel_processing.is_selected()
            || user_config.get_gpu_processing_default()
                != self.cb_gpu_processing_default.is_selected()
            || user_config.is_remove_excluded_views() != self.cb_remove_excluded_views.is_selected()
            || user_config.is_tilt_angles_rawtlt_file()
                != self.cb_tilt_angles_rawtlt_file.is_selected()
            || user_config.get_swap_y_and_z() != self.cb_swap_y_and_z.is_selected()
            || user_config.is_set_fei_pixel_size() != self.cb_set_fei_pixel_size.is_selected()
            || font_size.map_or(true, |font_size| user_config.get_font_size() != font_size)
            || user_config.get_font_family().is_none()
            || font_family.is_none()
            || user_config.get_font_family().as_deref() != font_family
            || user_config.is_parallel_processing() != self.cb_parallel_processing.is_selected()
            || user_config.is_gpu_processing() != self.cb_gpu_processing.is_selected()
            || user_config.get_cpus().to_string()
                != Field::get_text_void(&*self.ltf_cpus).unwrap_or_default()
            || user_config.get_parallel_table_size().to_string()
                != Field::get_text_void(&*self.ltf_parallel_table_size).unwrap_or_default()
            || user_config.get_join_table_size().to_string()
                != Field::get_text_void(&*self.ltf_join_table_size).unwrap_or_default()
            || user_config.get_peet_table_size().to_string()
                != Field::get_text_void(&*self.ltf_peet_table_size).unwrap_or_default()
            || user_config.get_batch_table_size().to_string()
                != Field::get_text_void(&*self.ltf_batch_table_size).unwrap_or_default()
            || user_config.get_smtp_server().unwrap_or_default()
                != Field::get_text_void(&*self.ltf_smtp_server).unwrap_or_default()
        {
            return true;
        }
        false
    }

    /// Java package-private `action(String)`.
    ///
    /// Upstream bug fixed in translation (SettingsDialog.java:461): a null action
    /// command makes `command.equals` throw `NullPointerException`; here it
    /// matches no button and only `updateDisplay` runs.
    pub fn action(&self, command: Option<&str>) {
        if command.is_some() && command == self.button_cancel.get_action_command().as_deref() {
            etomo_director::INSTANCE.close_settings_dialog();
        } else if command.is_some() && command == self.button_apply.get_action_command().as_deref()
        {
            etomo_director::INSTANCE.get_settings_parameters();
        } else if command.is_some() && command == self.button_done.get_action_command().as_deref() {
            etomo_director::INSTANCE.get_settings_parameters();
            etomo_director::INSTANCE.save_settings_dialog();
            etomo_director::INSTANCE.close_settings_dialog();
        }
        self.update_display();
    }

    /// Java package-private `setTooltips()`.
    pub fn set_tooltips(&self) {
        self.cb_set_fei_pixel_size.set_tool_tip_text_string(Some(
            "During tomogram setup, transfer pixel size from an mdoc file to main header if \
             pixel spacing is 1.0 in main header, or from extended header to main \
             header for a file from old FEI software",
        ));
    }

    // --- javax.swing.JDialog / java.awt.Dialog / java.awt.Window members ---

    /// Java `JDialog.getContentPane()`.
    pub fn get_content_pane(&self) -> Rc<JComponent> {
        self.dialog.get_content_pane()
    }

    /// Java `Dialog.setTitle(String)`.
    pub fn set_title(&self, title: &str) {
        self.dialog.set_title(title);
    }

    /// Java `Dialog.getTitle()`.
    pub fn get_title(&self) -> String {
        self.dialog.get_title()
    }

    /// Java `Component.setLocation(int, int)`.
    pub fn set_location(&self, x: i32, y: i32) {
        self.location.set(Point { x, y });
    }

    /// Java `Component.getLocation()`.
    pub fn get_location(&self) -> Point {
        self.location.get()
    }

    /// Java `Dialog.setModal(boolean)`.
    pub fn set_modal(&self, modal: bool) {
        self.modal.set(modal);
    }

    /// Java `Dialog.isModal()`.
    pub fn is_modal(&self) -> bool {
        self.modal.get()
    }

    /// Java `Dialog.setVisible(boolean)`.
    pub fn set_visible(&self, visible: bool) {
        self.dialog.set_visible(visible);
    }

    /// Java `Window.isVisible()`.
    pub fn is_visible(&self) -> bool {
        self.dialog.is_visible()
    }

    /// Java `Window.dispose()`.
    pub fn dispose(&self) {
        self.dialog.dispose();
    }
}

/// Java private static final `FontFamilies`.  Gets the available font family
/// name list, which may contain "'" (a character that caused a nasty Java look
/// and feel failure).  Put the usable font families (ones that don't contain
/// "'") into the usable font families list.
struct FontFamilies {
    /// Java private final `usable`.
    usable: Vec<String>,
    /// Java private `defaultIndex`, initially -1.
    default_index: i32,
}

impl FontFamilies {
    /// Java private `FontFamilies()`.  Creates the usable font family list from
    /// the available font family list.  Does not use fonts with "'" in their
    /// name.  Sets the default font family to "Dialog".
    ///
    /// `GraphicsEnvironment.getAvailableFontFamilyNames()` is a JVM query; the
    /// five logical families every JVM lists stand in for it (module comment).
    ///
    /// Upstream bug fixed in translation (SettingsDialog.java:519):
    /// `defaultIndex = i` stores the index into the *available* list, which is
    /// used to index the *usable* list; once a family containing "'" has been
    /// skipped before "Dialog", the default points one or more entries past it
    /// (possibly past the end).  Here the default is the index "Dialog" gets in
    /// the usable list.
    fn new() -> FontFamilies {
        let available: [&str; 5] = ["Dialog", "DialogInput", "Monospaced", "SansSerif", "Serif"];
        let mut usable: Vec<String> = Vec::new();
        let mut default_index = -1;
        for name in available {
            if !name.contains('\'') {
                usable.push(name.to_owned());
                if name.eq_ignore_ascii_case("dialog") {
                    default_index = usable.len() as i32 - 1;
                }
            } else {
                eprintln!("Removing unusable font family:{}", name);
            }
        }
        FontFamilies {
            usable,
            default_index,
        }
    }

    /// Java private `getFontFamilies()`.  Get the usable font families.
    fn get_font_families(&self) -> &[String] {
        &self.usable
    }

    /// Java private `getIndex(String)`.  Gets an index to fontFamilyName in the
    /// usable font family list.  If not found, returns the default font index.
    fn get_index(&self, font_family_name: &str) -> i32 {
        // Find the font family index from the available fontFamilies
        for i in 0..self.usable.len() {
            // String.compareToIgnoreCase == 0
            if self.usable[i].to_lowercase() == font_family_name.to_lowercase() {
                return i as i32;
            }
        }
        self.default_index
    }

    /// Java private `getName(int)`.  `Vector.get` throws for an index out of
    /// range (-1 when nothing is selected); here that is `None` (see
    /// `getParameters`).
    fn get_name(&self, i: i32) -> Option<&str> {
        if i < 0 {
            return None;
        }
        self.usable.get(i as usize).map(String::as_str)
    }
}
