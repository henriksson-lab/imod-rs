//! `IMOD/Etomo/src/etomo/ui/swing/SirtPanel.java`.
//!
//! Java `final class SirtPanel implements Run3dmodButtonContainer,
//! SirtsetupDisplay, Expandable, RadialParent`: the SIRT part of the Tomogram
//! Generation dialog.  An EDT object created as `Rc<Self>` by
//! [`SirtPanel::get_instance`]; every method takes `&self`.  The inner
//! listener classes `SirtActionListener` and `SirtDocumentListener` are
//! closures holding a weak reference to the panel.  The parent dialog is held
//! weakly (it owns this panel).

use std::cell::{Cell, RefCell};
use std::path::PathBuf;
use std::rc::{Rc, Weak};

use super::check_box::CheckBox;
use super::combo_box::ComboBox;
use super::deferred_3dmod_button::Deferred3dmodButton;
use super::expand_button::ExpandButton;
use super::expandable::Expandable;
use super::file_chooser::{self, FileChooser};
use super::global_expand_button::GlobalExpandButton;
use super::labeled_text_field::LabeledTextField;
use super::multi_line_button::MultiLineButton;
use super::panel_header::PanelHeader;
use super::radial_panel::RadialPanel;
use super::radial_parent::RadialParent;
use super::radio_button::RadioButton;
use super::run_3dmod_button::Run3dmodButton;
use super::run_3dmod_button_container::Run3dmodButtonContainer;
use super::sirtsetup_display::SirtsetupDisplay;
use super::spaced_panel::{self, SpacedPanel};
use super::tomogram_generation_dialog::TomogramGenerationDialog;
use super::tomogram_generation_parent::TomogramGenerationParent;
use super::ui_harness;
use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::sirtsetup_param::{self, SirtsetupParam};
use crate::imod::etomo::jdk::{
    ActionEvent, ActionListener, ButtonGroup, DocumentEvent, DocumentListener, FileFilter,
    JComponent,
};
use crate::imod::etomo::process::imod_manager;
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::storage::autodoc::autodoc::Autodoc;
use crate::imod::etomo::storage::autodoc::autodoc_factory;
use crate::imod::etomo::storage::autodoc::read_only_autodoc::ReadOnlyAutodoc;
use crate::imod::etomo::storage::log_file::LogFileError;
use crate::imod::etomo::storage::sirt_output_file_filter::SirtOutputFileFilter;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_meta_data::ConstMetaData;
use crate::imod::etomo::r#type::const_panel_header_settings::ConstPanelHeaderSettings;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::etomo_autodoc;
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::r#type::extension::Extension;
use crate::imod::etomo::r#type::image_filename_style::ImageFilenameStyle;
use crate::imod::etomo::r#type::meta_data::MetaData;
use crate::imod::etomo::r#type::panel_id::PanelId;
use crate::imod::etomo::r#type::process_result_display::ProcessResultDisplayHandle;
use crate::imod::etomo::r#type::recon_screen_state::ReconScreenState;
use crate::imod::etomo::r#type::tomogram_state::TomogramState;
use crate::imod::etomo::ui::field::Field;
use crate::imod::etomo::ui::field_type::FieldType;

/// Java private static final `RESUME_FROM_LAST_ITERATION_LABEL`.
const RESUME_FROM_LAST_ITERATION_LABEL: &str = "Resume from last iteration";

/// Java `final class SirtPanel`.
pub struct SirtPanel {
    /// Java private final `pnlRoot = new JPanel()`.
    pnl_root: Rc<JComponent>,
    /// Java private final `cbSubarea`.
    cb_subarea: Rc<CheckBox>,
    /// Java private final `ltfYOffsetOfSubarea`.
    ltf_y_offset_of_subarea: Rc<LabeledTextField>,
    /// Java private final `ltfSubareaSize`.
    ltf_subarea_size: Rc<LabeledTextField>,
    /// Java private final `ltfLeaveIterations`.
    ltf_leave_iterations: Rc<LabeledTextField>,
    /// Java private final `cbScaleToInteger`.
    cb_scale_to_integer: Rc<CheckBox>,
    /// Java private final `listener` (`SirtActionListener`).
    listener: ActionListener,
    /// Java private final `btn3dmodSirt`.
    btn_3dmod_sirt: Rc<Run3dmodButton>,
    /// Java private final `cbCleanUpPastStart`.
    cb_clean_up_past_start: Rc<CheckBox>,
    /// Java private final `ltfFlatFilterFraction`.
    ltf_flat_filter_fraction: Rc<LabeledTextField>,
    /// Java private final `pnlSirtsetupParamsBody = SpacedPanel.getInstance(true)`.
    pnl_sirtsetup_params_body: Rc<SpacedPanel>,
    /// Java private final `bgStartingIteration`.
    #[allow(dead_code)]
    bg_starting_iteration: Rc<ButtonGroup>,
    /// Java private final `rbStartFromZero`.
    rb_start_from_zero: Rc<RadioButton>,
    /// Java private final `rbResumeFromLastIteration`.
    rb_resume_from_last_iteration: Rc<RadioButton>,
    /// Java private final `rbResumeFromIteration`.
    rb_resume_from_iteration: Rc<RadioButton>,
    /// Java private final `cmbResumeFromIteration`.  The Java combo box holds
    /// `EtomoNumber` items; the Rust `ComboBox` holds their string forms.
    cmb_resume_from_iteration: Rc<ComboBox>,
    /// Java private `cbSkipVertSliceOutput` (never reassigned).
    cb_skip_vert_slice_output: Rc<CheckBox>,

    /// Java private final `axisID`.
    axis_id: AxisID,
    /// Java private final `manager`.
    manager: &'static ApplicationManager,
    /// Java private final `btnSirt`.
    btn_sirt: Rc<Run3dmodButton>,
    /// Java private final `btnUseSirt`.
    btn_use_sirt: Rc<MultiLineButton>,
    /// Java private final `parent` (held weakly; the dialog owns this panel).
    parent: Weak<TomogramGenerationDialog>,
    /// Java private final `dialogType`.
    dialog_type: DialogType,
    /// Java private final `radiusAndSigmaPanel`.
    radius_and_sigma_panel: Rc<RadialPanel>,
    /// Java private final `sirtSetupParamsHeader`.
    sirt_setup_params_header: Rc<PanelHeader>,
    /// Java private `imageFilenameStyle` (assigned only in the constructor).
    /// `None` is Java null (no base metadata).
    image_filename_style: Option<ImageFilenameStyle>,

    /// Java private `numFiles = 0`.
    num_files: Cell<usize>,
    /// Java private `differentFromCheckpointFlag = false`.
    different_from_checkpoint_flag: Cell<bool>,
    /// Java `this`, for the listeners registered after construction.
    this: Weak<SirtPanel>,
}

impl SirtPanel {
    /// Java private constructor `SirtPanel(ApplicationManager, AxisID,
    /// DialogType, GlobalExpandButton, TomogramGenerationDialog)`.
    fn new(
        manager: &'static ApplicationManager,
        axis_id: AxisID,
        dialog_type: DialogType,
        global_advanced_button: &Rc<GlobalExpandButton>,
        parent: Weak<TomogramGenerationDialog>,
    ) -> Rc<SirtPanel> {
        Rc::new_cyclic(|this: &Weak<SirtPanel>| {
            // Field initializers, in declaration order.
            let pnl_root = JComponent::new_panel();
            let cb_subarea = CheckBox::new_string(Some("Reconstruct subarea"));
            let ltf_y_offset_of_subarea =
                LabeledTextField::get_numeric_instance_string(Some(" Offset in Y: "));
            let ltf_subarea_size = LabeledTextField::new_field_type_string(
                FieldType::IntegerPair,
                Some(&format!("{}{}", "Size in X and Y", ": ")),
            );
            let ltf_leave_iterations = LabeledTextField::new_field_type_string(
                FieldType::IntegerList,
                Some("Iteration #'s to retain: "),
            );
            let cb_scale_to_integer =
                CheckBox::new_string(Some("Scale retained volumes to integers"));
            // Java `new SirtActionListener(this)`.
            let adaptee = this.clone();
            let listener: ActionListener = Rc::new(move |event: &ActionEvent| {
                if let Some(adaptee) = adaptee.upgrade() {
                    adaptee.action(event.get_action_command().unwrap_or(""), None, None);
                }
            });
            let container: Weak<dyn Run3dmodButtonContainer> = this.clone();
            let btn_3dmod_sirt =
                Run3dmodButton::get_3dmod_instance_string_run_3dmod_button_container(
                    Some("View Tomogram(s) In 3dmod"),
                    Some(container),
                );
            let cb_clean_up_past_start =
                CheckBox::new_string(Some("Delete existing reconstructions after starting point"));
            let ltf_flat_filter_fraction = LabeledTextField::new_field_type_string(
                FieldType::FloatingPoint,
                Some("Flat filter fraction: "),
            );
            let pnl_sirtsetup_params_body = SpacedPanel::get_instance_boolean(true);
            let bg_starting_iteration = ButtonGroup::new();
            let rb_start_from_zero = RadioButton::new_string_button_group(
                Some("Start from beginning"),
                Some(&bg_starting_iteration),
            );
            let rb_resume_from_last_iteration = RadioButton::new_string_button_group(
                Some(RESUME_FROM_LAST_ITERATION_LABEL),
                Some(&bg_starting_iteration),
            );
            let rb_resume_from_iteration = RadioButton::new_string_button_group(
                Some("Go back, resume from iteration:"),
                Some(&bg_starting_iteration),
            );
            let cmb_resume_from_iteration = ComboBox::get_unlabeled_instance(
                rb_resume_from_iteration.get_text_void().as_deref(),
            );
            let cb_skip_vert_slice_output = CheckBox::new_string(Some(
                "Do not make vertical slice output files used for resuming",
            ));

            // Constructor body.
            let expandable: Weak<dyn Expandable> = this.clone();
            let sirt_setup_params_header =
                PanelHeader::get_advanced_basic_instance_string_expandable_dialog_type_global_expand_button(
                    Some("SIRT"),
                    Some(expandable),
                    Some(dialog_type),
                    Some(global_advanced_button.clone()),
                );
            let base_manager: &'static dyn BaseManager = manager;
            let radial_parent: Weak<dyn RadialParent> = this.clone();
            let radius_and_sigma_panel =
                RadialPanel::get_instance(base_manager, axis_id, PanelId::Sirtsetup, radial_parent);
            let factory = manager.get_process_result_display_factory(axis_id);
            // Java casts `(Run3dmodButton) factory.getSirtsetup()` and
            // `(MultiLineButton) factory.getUseSirt()`; the factory returns the
            // concrete buttons.
            let btn_sirt = factory.get_sirtsetup();
            let btn_use_sirt = factory.get_use_sirt();
            // Java dereferences getBaseMetaData() unguarded; a missing one leaves
            // the style null.
            let image_filename_style = base_manager
                .get_base_meta_data()
                .map(|meta_data| meta_data.base().get_image_filename_style());
            SirtPanel {
                pnl_root,
                cb_subarea,
                ltf_y_offset_of_subarea,
                ltf_subarea_size,
                ltf_leave_iterations,
                cb_scale_to_integer,
                listener,
                btn_3dmod_sirt,
                cb_clean_up_past_start,
                ltf_flat_filter_fraction,
                pnl_sirtsetup_params_body,
                bg_starting_iteration,
                rb_start_from_zero,
                rb_resume_from_last_iteration,
                rb_resume_from_iteration,
                cmb_resume_from_iteration,
                cb_skip_vert_slice_output,
                axis_id,
                manager,
                btn_sirt,
                btn_use_sirt,
                parent,
                dialog_type,
                radius_and_sigma_panel,
                sirt_setup_params_header,
                image_filename_style,
                num_files: Cell::new(0),
                different_from_checkpoint_flag: Cell::new(false),
                this: this.clone(),
            }
        })
    }

    /// Java static `getInstance(ApplicationManager, AxisID, DialogType,
    /// GlobalExpandButton, TomogramGenerationDialog)`.
    pub fn get_instance(
        manager: &'static ApplicationManager,
        axis_id: AxisID,
        dialog_type: DialogType,
        global_advanced_button: &Rc<GlobalExpandButton>,
        parent: Weak<TomogramGenerationDialog>,
    ) -> Rc<SirtPanel> {
        let instance = SirtPanel::new(
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

    /// Java `getRoot()`.
    pub fn get_root(&self) -> Rc<JComponent> {
        self.pnl_root.clone()
    }

    /// Java private `createPanel()`.
    fn create_panel(&self) {
        // initialize
        let pnl_subarea = SpacedPanel::get_instance_void();
        let pnl_size_and_offset = JComponent::new_panel();
        let pnl_sirtsetup_params = JComponent::new_panel();
        let pnl_scale_to_integer = JComponent::new_panel();
        let pnl_skip_vert_slice_output = JComponent::new_panel();
        let pnl_clean_up_past_start = JComponent::new_panel();
        let pnl_start_from = JComponent::new_panel();
        let pnl_start_from_zero = JComponent::new_panel();
        let pnl_resume_from_last_iteration = JComponent::new_panel();
        let pnl_resume_from_iteration = JComponent::new_panel();
        let pnl_buttons = JComponent::new_panel();
        let container: Weak<dyn Run3dmodButtonContainer> = self.this.clone();
        self.btn_sirt.set_container(Some(container));
        let deferred: Rc<dyn Deferred3dmodButton> = self.btn_3dmod_sirt.clone();
        self.btn_sirt
            .set_deferred_3dmod_button_deferred_3dmod_button(Some(deferred));
        // root panel
        // Swing layout: pnlRoot BoxLayout Y_AXIS.
        self.pnl_root.add(&pnl_subarea.get_container());
        self.pnl_root.add(&pnl_sirtsetup_params);
        self.pnl_root.add(&pnl_buttons);
        // SIRT panel
        // Subarea panel
        pnl_subarea.set_box_layout(spaced_panel::Y_AXIS);
        // Swing layout: pnlSubarea.setBorder(BorderFactory.createEtchedBorder()).
        // Component.LEFT_ALIGNMENT
        pnl_subarea.set_component_alignment_x(0.0);
        pnl_subarea.add_check_box(&self.cb_subarea);
        pnl_subarea.add_j_panel(&pnl_size_and_offset);
        // Offset and size panel
        // Swing layout: pnlSizeAndOffset BoxLayout X_AXIS.
        pnl_size_and_offset.add(&self.ltf_subarea_size.get_container());
        pnl_size_and_offset.add(&self.ltf_y_offset_of_subarea.get_container());
        // SIRT params panel
        // Swing layout: pnlSirtsetupParams BoxLayout Y_AXIS, etched border.
        pnl_sirtsetup_params.add(&self.sirt_setup_params_header.get_container());
        pnl_sirtsetup_params.add(&self.pnl_sirtsetup_params_body.get_container());
        // SIRT params body panel
        self.pnl_sirtsetup_params_body
            .set_box_layout(spaced_panel::Y_AXIS);
        self.pnl_sirtsetup_params_body
            .add_container(&self.radius_and_sigma_panel.get_root());
        self.pnl_sirtsetup_params_body
            .add_labeled_text_field(&self.ltf_leave_iterations);
        self.pnl_sirtsetup_params_body
            .add_j_panel(&pnl_scale_to_integer);
        self.pnl_sirtsetup_params_body
            .add_j_panel(&pnl_skip_vert_slice_output);
        self.pnl_sirtsetup_params_body
            .add_j_panel(&pnl_clean_up_past_start);
        self.pnl_sirtsetup_params_body
            .add_labeled_text_field(&self.ltf_flat_filter_fraction);
        self.pnl_sirtsetup_params_body.add_j_panel(&pnl_start_from);
        // ScaleToInteger panel
        // Swing layout: BoxLayout X_AXIS, CENTER_ALIGNMENT, horizontal glue.
        pnl_scale_to_integer.add(&self.cb_scale_to_integer.get_component());
        // SkipVertSliceOutput panel
        // Swing layout: BoxLayout X_AXIS, CENTER_ALIGNMENT, horizontal glue.
        pnl_skip_vert_slice_output.add(&self.cb_skip_vert_slice_output.get_component());
        // CleanUpPastStart panel
        // Swing layout: BoxLayout X_AXIS, CENTER_ALIGNMENT, horizontal glue.
        pnl_clean_up_past_start.add(&self.cb_clean_up_past_start.get_component());
        // start from panel
        // Swing layout: pnlStartFrom BoxLayout Y_AXIS.
        pnl_start_from.add(&pnl_start_from_zero);
        pnl_start_from.add(&pnl_resume_from_last_iteration);
        pnl_start_from.add(&pnl_resume_from_iteration);
        // StartFromZero panel
        // Swing layout: BoxLayout X_AXIS, CENTER_ALIGNMENT, horizontal glue.
        pnl_start_from_zero.add(&self.rb_start_from_zero.get_component());
        // ResumeFromLastIteration panel
        // Swing layout: BoxLayout X_AXIS, CENTER_ALIGNMENT, horizontal glue.
        pnl_resume_from_last_iteration.add(&self.rb_resume_from_last_iteration.get_component());
        // Resume from iteration panel
        // Swing layout: BoxLayout X_AXIS.
        pnl_resume_from_iteration.add(&self.rb_resume_from_iteration.get_component());
        pnl_resume_from_iteration.add(&self.cmb_resume_from_iteration.get_component());
        // Buttons panel
        // Swing layout: pnlButtons BoxLayout X_AXIS.
        pnl_buttons.add(&self.btn_sirt.get_component());
        pnl_buttons.add(&self.btn_3dmod_sirt.get_component());
        pnl_buttons.add(&self.btn_use_sirt.get_component());
        // defaults
        self.rb_start_from_zero.set_selected_boolean(true);
        self.cb_clean_up_past_start.set_selected_boolean(true);
        self.update_display();
    }

    /// Java private `addListeners()`.
    fn add_listeners(&self) {
        self.btn_sirt.add_action_listener(self.listener.clone());
        self.btn_3dmod_sirt
            .add_action_listener(self.listener.clone());
        self.btn_use_sirt.add_action_listener(self.listener.clone());
        self.rb_start_from_zero
            .add_action_listener(self.listener.clone());
        self.rb_resume_from_last_iteration
            .add_action_listener(self.listener.clone());
        self.rb_resume_from_iteration
            .add_action_listener(self.listener.clone());
        self.cb_subarea
            .add_action_listener(Some(self.listener.clone()));
        // Java `new SirtDocumentListener(this)`: changedUpdate, insertUpdate and
        // removeUpdate all call `adaptee.documentAction()`.
        let adaptee = self.this.clone();
        let document_listener: DocumentListener = Rc::new(move |_event: &DocumentEvent| {
            if let Some(adaptee) = adaptee.upgrade() {
                adaptee.document_action();
            }
        });
        self.ltf_subarea_size
            .add_document_listener(document_listener.clone());
        self.ltf_y_offset_of_subarea
            .add_document_listener(document_listener);
    }

    /// Java `@Deprecated msgFieldChanged(boolean)` (8/6/18 Saving tilt.com no
    /// longer effects SIRT resume - Bug# 2098, comment 10).
    pub fn msg_field_changed(&self, different_from_checkpoint: bool) {
        self.different_from_checkpoint_flag
            .set(different_from_checkpoint);
        self.update_display();
    }

    /// Java `updateDisplay()`.
    pub fn update_display(&self) {
        self.ltf_flat_filter_fraction
            .set_visible(self.is_advanced());
        // Update checkpointed fields - disabled fields are not checked for checkpoint
        // difference.
        let subarea = self.cb_subarea.is_selected();
        self.ltf_subarea_size.set_enabled(subarea);
        self.ltf_y_offset_of_subarea.set_enabled(subarea);
        // Enable resume if there are files to resume from and this class has no
        // checkpoint differences, and classes that this class is observing have no
        // checkpoint differences.
        let enable_resume = self.num_files.get() > 0
            && !self.is_different_from_checkpoint()
            && !self.different_from_checkpoint_flag.get();
        self.rb_resume_from_last_iteration
            .set_enabled(enable_resume);
        self.rb_resume_from_iteration.set_enabled(enable_resume);
        self.cmb_resume_from_iteration
            .set_enabled(enable_resume && self.rb_resume_from_iteration.is_selected());
        // Don't allow the resume radio buttons to be selected when they are disabled
        if !enable_resume
            && (self.rb_resume_from_last_iteration.is_selected()
                || self.rb_resume_from_iteration.is_selected())
        {
            self.rb_start_from_zero.set_selected_boolean(true);
        }
        let resume = self.is_resume();
        // Correct checkpointed fields now that resume is available
        self.ltf_subarea_size.set_enabled(subarea && !resume);
        self.ltf_y_offset_of_subarea.set_enabled(subarea && !resume);
        self.radius_and_sigma_panel.set_editable(!resume);
        self.cmb_resume_from_iteration.set_enabled(
            self.rb_resume_from_iteration.is_enabled()
                && self.rb_resume_from_iteration.is_selected(),
        );
    }

    /// Java private `isResumeEnabled()` (unused in the Java).
    #[allow(dead_code)]
    fn is_resume_enabled(&self) -> bool {
        self.rb_resume_from_last_iteration.is_enabled()
    }

    /// Java `isResume()`.
    pub fn is_resume(&self) -> bool {
        self.rb_resume_from_last_iteration.is_selected()
            || self.rb_resume_from_iteration.is_selected()
    }

    /// Java `msgSirtSucceeded()`.
    pub fn msg_sirt_succeeded(&self) {
        self.load_resume_from();
    }

    /// Java `msgMethodChanged()`.  A parent that is gone answers false (Java's
    /// parent outlives the panel).
    pub fn msg_method_changed(&self) {
        self.pnl_root
            .set_visible(self.parent.upgrade().is_some_and(|parent| parent.is_sirt()));
    }

    /// Java `done()`.
    pub fn done(&self) {
        self.btn_sirt.remove_action_listener(&self.listener);
        self.btn_use_sirt.remove_action_listener(&self.listener);
    }

    /// Java `getParameters(ReconScreenState)`.
    pub fn get_parameters_recon_screen_state(&self, screen_state: &ReconScreenState) {
        self.btn_sirt.set_button_state(
            screen_state.get_button_state(self.btn_sirt.get_button_state_key().as_deref()),
        );
        self.btn_use_sirt.set_button_state(
            screen_state.get_button_state(self.btn_use_sirt.get_button_state_key().as_deref()),
        );
        self.sirt_setup_params_header
            .get_state(Some(screen_state.get_tomo_gen_sirt_header_state()));
    }

    /// Java `setParameters(ReconScreenState)`.
    pub fn set_parameters_recon_screen_state(&self, screen_state: &ReconScreenState) {
        self.sirt_setup_params_header
            .set_state(Some(screen_state.get_tomo_gen_sirt_header_state()));
        self.btn_sirt.set_button_state(
            screen_state.get_button_state(self.btn_sirt.get_button_state_key().as_deref()),
        );
        self.btn_use_sirt.set_button_state(
            screen_state.get_button_state(self.btn_use_sirt.get_button_state_key().as_deref()),
        );
    }

    /// Java `getParameters(MetaData)`.
    pub fn get_parameters_meta_data(&self, meta_data: &MetaData) {
        meta_data.set_gen_subarea(self.axis_id, self.cb_subarea.is_selected());
        meta_data.set_gen_subarea_size(
            self.axis_id,
            self.ltf_subarea_size.get_text_void().as_deref(),
        );
        meta_data.set_gen_y_offset_of_subarea(
            self.axis_id,
            self.ltf_y_offset_of_subarea.get_text_void().as_deref(),
        );
        self.radius_and_sigma_panel
            .get_parameters_meta_data(meta_data);
    }

    /// Java `setParameters(ConstMetaData)`.
    pub fn set_parameters_const_meta_data(&self, meta_data: &dyn ConstMetaData) {
        self.cb_subarea
            .set_selected_boolean(meta_data.is_gen_subarea(self.axis_id));
        self.ltf_subarea_size
            .set_text_string(Some(&meta_data.get_gen_subarea_size(self.axis_id)));
        self.ltf_y_offset_of_subarea
            .set_text_string(Some(&meta_data.get_gen_y_offset_of_subarea(self.axis_id)));
        self.radius_and_sigma_panel
            .set_parameters_const_meta_data(meta_data);
        self.load_resume_from();
    }

    /// Java `getParameters(SirtsetupParam, boolean)` (implements
    /// `SirtsetupDisplay`).
    pub fn get_parameters_sirtsetup_param_boolean(
        &self,
        param: &mut SirtsetupParam,
        do_validation: bool,
    ) -> bool {
        /// What the Java try blocks catch.
        enum Thrown {
            /// `FieldValidationFailedException` (caught by the outer try).
            FieldValidationFailed,
            /// `FortranInputSyntaxException` (caught by the inner try).
            FortranInputSyntax,
        }
        let base_manager: &'static dyn BaseManager = self.manager;
        // try { try {
        let inner = (|| -> Result<bool, Thrown> {
            if do_validation && self.ltf_leave_iterations.is_empty() {
                ui_harness::with(|harness| {
                    harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(base_manager),
                        &format!("{} is empty.", self.ltf_leave_iterations.get_label()),
                        "Entry Error",
                        Some(self.axis_id),
                    )
                });
                // Java `Thread.dumpStack()`.
                eprintln!(
                    "java.lang.Exception: Stack trace\n{}",
                    std::backtrace::Backtrace::force_capture()
                );
                return Ok(false);
            }
            let text = self
                .ltf_leave_iterations
                .get_text_boolean(do_validation)
                .map_err(|_| Thrown::FieldValidationFailed)?;
            param.set_leave_iterations(text.as_deref());
            if self.cb_subarea.is_selected() {
                if self.ltf_subarea_size.is_empty() {
                    ui_harness::with(|harness| {
                        harness.open_message_dialog_base_manager_string_string_axis_id(
                            Some(base_manager),
                            &format!("{} is empty.", self.ltf_subarea_size.get_label()),
                            "Entry Error",
                            Some(self.axis_id),
                        )
                    });
                    return Ok(false);
                }
                let text = self
                    .ltf_subarea_size
                    .get_text_boolean(do_validation)
                    .map_err(|_| Thrown::FieldValidationFailed)?;
                param
                    .set_subarea_size(text.as_deref())
                    .map_err(|_| Thrown::FortranInputSyntax)?;
                let text = self
                    .ltf_y_offset_of_subarea
                    .get_text_boolean(do_validation)
                    .map_err(|_| Thrown::FieldValidationFailed)?;
                param.set_y_offset_of_subarea(text.as_deref());
            } else {
                param.reset_subarea_size();
                param.reset_y_offset_of_subarea();
            }
            param.set_scale_to_integer(self.cb_scale_to_integer.is_selected());
            if !self
                .radius_and_sigma_panel
                .get_parameters_sirtsetup_param_boolean(param, do_validation)
            {
                return Ok(false);
            }
            param.set_clean_up_past_start(self.cb_clean_up_past_start.is_selected());
            let text = self
                .ltf_flat_filter_fraction
                .get_text_boolean(do_validation)
                .map_err(|_| Thrown::FieldValidationFailed)?;
            param.set_flat_filter_fraction(text.as_deref());
            param.set_skip_vert_slice_output(self.cb_skip_vert_slice_output.is_selected());
            Ok(true)
        })();
        match inner {
            Ok(true) => {}
            Ok(false) => return false,
            // catch (FortranInputSyntaxException e)
            Err(Thrown::FortranInputSyntax) => {
                ui_harness::with(|harness| {
                    harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(base_manager),
                        &format!(
                            "{} is an invalid list of integers.",
                            self.ltf_subarea_size.get_label()
                        ),
                        "Entry Error",
                        Some(self.axis_id),
                    )
                });
                return false;
            }
            // } catch (FieldValidationFailedException e) { return false; }
            Err(Thrown::FieldValidationFailed) => return false,
        }
        let mut resume = false;
        if self.rb_start_from_zero.is_selected() {
            param.set_start_from_zero(true);
            param.reset_resume_from_iteration();
        } else if self.rb_resume_from_last_iteration.is_enabled()
            && self.rb_resume_from_last_iteration.is_selected()
        {
            param.set_start_from_zero(false);
            param.reset_resume_from_iteration();
            resume = true;
        } else if self.rb_resume_from_iteration.is_enabled()
            && self.rb_resume_from_iteration.is_selected()
        {
            param.set_start_from_zero(false);
            // Java `(ConstEtomoNumber) cmbResumeFromIteration.getSelectedItem()`:
            // the combo box holds the string form of each EtomoNumber item.
            let selected_item = self.cmb_resume_from_iteration.get_selected_item();
            let file_number = selected_item.map(|item| {
                let mut file_number = EtomoNumber::new();
                file_number.set_string(Some(&item));
                file_number
            });
            param.set_resume_from_iteration(file_number.as_ref().map(|number| &number.base));
            resume = true;
        } else {
            ui_harness::with(|harness| {
                harness.open_message_dialog_base_manager_string_string_axis_id(
                    Some(base_manager),
                    "Please select an enabled starting option.",
                    "Entry Error",
                    Some(self.axis_id),
                )
            });
            return false;
        }
        param.set_resume(resume);
        true
    }

    /// Java `setParameters(SirtsetupParam)`.
    pub fn set_parameters_sirtsetup_param(&self, param: &SirtsetupParam) {
        self.ltf_leave_iterations
            .set_text_string(Some(&param.get_leave_iterations()));
        if !param.is_subarea_size_null() {
            self.ltf_subarea_size
                .set_text_string(Some(&param.get_subarea_size()));
        }
        if !param.is_y_offset_of_subarea_null() {
            self.ltf_y_offset_of_subarea
                .set_text_string(Some(&param.get_y_offset_of_subarea()));
        }
        self.cb_scale_to_integer
            .set_selected_boolean(!param.is_scale_to_integer_null());
        self.radius_and_sigma_panel
            .set_parameters_sirtsetup_param(param);
        self.cb_clean_up_past_start
            .set_selected_boolean(param.is_clean_up_past_start());
        self.ltf_flat_filter_fraction
            .set_text_string(Some(&param.get_flat_filter_fraction()));
        self.cb_skip_vert_slice_output
            .set_selected_boolean(param.is_skip_vert_slice_output());
        if param.is_start_from_zero() {
            self.rb_start_from_zero.set_selected_boolean(true);
        } else if !param.is_resume_from_iteration_null() {
            self.rb_resume_from_iteration.set_selected_boolean(true);
        } else {
            self.rb_resume_from_last_iteration
                .set_selected_boolean(true);
        }
        self.update_display();
    }

    /// Java private `loadResumeFrom()`.
    fn load_resume_from(&self) {
        let base_manager: &'static dyn BaseManager = self.manager;
        // Clear pulldown list.
        self.cmb_resume_from_iteration.remove_all_items();
        // Get file names.
        let subarea = self.cb_subarea.is_selected();
        // Ignore .sint## files - they cannot be resumed from.
        let filter: Rc<SirtOutputFileFilter> = if subarea {
            SirtOutputFileFilter::get_subarea_instance(
                base_manager,
                self.image_filename_style,
                self.axis_id,
                false,
            )
        } else {
            SirtOutputFileFilter::get_full_instance(
                base_manager,
                self.image_filename_style,
                self.axis_id,
                false,
            )
        };
        // Java `new File(manager.getPropertyUserDir()).list(filter)`: the names
        // the FilenameFilter accepts, or null when the directory cannot be read.
        // (Java would throw a NullPointerException on a null user dir; that is
        // read as an unreadable directory here.)
        let file_name_list: Option<Vec<String>> =
            base_manager.get_property_user_dir().and_then(|dir| {
                let dir = PathBuf::from(dir);
                std::fs::read_dir(&dir).ok().map(|entries| {
                    entries
                        .filter_map(|entry| entry.ok())
                        .map(|entry| entry.file_name().to_string_lossy().into_owned())
                        .filter(|file_name| filter.accept_file_string(&dir, file_name))
                        .collect()
                })
            });
        // Extract iteration numbers, sort them, and add them to the pulldown list.
        let mut file_number_list: Option<Vec<i32>> = None;
        match &file_name_list {
            Some(file_name_list) if !file_name_list.is_empty() => {
                let mut numbers = vec![0i32; file_name_list.len()];
                for (i, file_name) in file_name_list.iter().enumerate() {
                    let extension = Extension::get_instance_with_style(
                        Some(base_manager),
                        Some(file_name),
                        self.image_filename_style,
                    );
                    if let Some(extension) = extension {
                        let file_number =
                            extension.get_file_number(Some(file_name), self.image_filename_style);
                        if let Some(file_number) = file_number {
                            numbers[i] = file_number.get_int();
                        }
                    }
                }
                // Java `Arrays.sort(int[])`.
                numbers.sort_unstable();
                // Add sorted numbers to the pulldown list.
                for i in (0..numbers.len()).rev() {
                    let mut file_number = EtomoNumber::new();
                    file_number.set_int(numbers[i]);
                    if i == numbers.len() - 1 {
                        // TODO 2206
                        self.rb_resume_from_last_iteration.set_text(Some(&format!(
                            "{}: {}",
                            RESUME_FROM_LAST_ITERATION_LABEL, file_number
                        )));
                    }
                    self.cmb_resume_from_iteration
                        .add_item(Some(&file_number.to_string()));
                }
                if numbers.len() > 1 {
                    self.cmb_resume_from_iteration.set_selected_index(1);
                }
                file_number_list = Some(numbers);
            }
            _ => {
                self.rb_resume_from_last_iteration
                    .set_text(Some(RESUME_FROM_LAST_ITERATION_LABEL));
            }
        }
        // Keep numFiles up to date
        match &file_number_list {
            None => self.num_files.set(0),
            Some(list) if list.is_empty() => self.num_files.set(0),
            Some(list) => self.num_files.set(list.len()),
        }
        self.update_display();
    }

    /// Java private `openFilesInImod(Run3dmodMenuOptions)`.
    fn open_files_in_imod(&self, run_3dmod_menu_options: Option<Run3dmodMenuOptions>) {
        let base_manager: &'static dyn BaseManager = self.manager;
        // Don't open the file chooser if there is only one file to choose
        let sirt_output_file_filter = SirtOutputFileFilter::get_instance(
            base_manager,
            self.image_filename_style,
            self.axis_id,
            true,
            true,
            true,
        );
        // Java `new File(manager.getPropertyUserDir()).listFiles((FilenameFilter)
        // sirtOutputFileFilter)`: the accepted files, or null when the directory
        // cannot be read (a null user dir, a Java NullPointerException, is read
        // the same way).
        let mut file_list = base_manager.get_property_user_dir().and_then(|dir| {
            let dir = PathBuf::from(dir);
            std::fs::read_dir(&dir).ok().map(|entries| {
                entries
                    .filter_map(|entry| entry.ok())
                    .map(|entry| entry.file_name().to_string_lossy().into_owned())
                    .filter(|file_name| sirt_output_file_filter.accept_file_string(&dir, file_name))
                    .map(|file_name| dir.join(file_name))
                    .collect::<Vec<PathBuf>>()
            })
        });
        if file_list.as_ref().is_none_or(|list| list.len() != 1) {
            let chooser = FileChooser::new_base_manager(Some(base_manager));
            // Swing layout: chooser.setPreferredSize(UIParameters.getInstance()
            // .getFileChooserDimension()).
            chooser.set_file_selection_mode(file_chooser::FILES_ONLY);
            chooser.set_multi_selection_enabled(true);
            let filter: Rc<dyn FileFilter> = sirt_output_file_filter.clone();
            chooser.set_file_filter(Some(filter));
            let return_val = chooser.show_open_dialog(Some(&self.pnl_root));
            if return_val != file_chooser::APPROVE_OPTION {
                return;
            }
            let selected = chooser.get_selected_files();
            if selected.is_empty() {
                return;
            }
            file_list = Some(selected);
        }
        let file_list = file_list.unwrap_or_default();
        // Java passes the (possibly null) options through; a null is read as
        // the default options.
        self.manager
            .open_files_in_imod_axis_id_string_file_array_run3dmod_menu_options(
                self.axis_id,
                imod_manager::SIRT_KEY,
                &file_list,
                run_3dmod_menu_options.unwrap_or_default(),
            );
    }

    /// Java `useSirt()`.
    pub fn use_sirt(&self) {
        let base_manager: &'static dyn BaseManager = self.manager;
        let sirt_output_file_filter = SirtOutputFileFilter::get_instance(
            base_manager,
            self.image_filename_style,
            self.axis_id,
            true,
            true,
            true,
        );
        // Java `new File(manager.getPropertyUserDir()).listFiles((FilenameFilter)
        // sirtOutputFileFilter)`: the accepted files, or null when the directory
        // cannot be read (a null user dir, a Java NullPointerException, is read
        // the same way).
        let file_list = base_manager.get_property_user_dir().and_then(|dir| {
            let dir = PathBuf::from(dir);
            std::fs::read_dir(&dir).ok().map(|entries| {
                entries
                    .filter_map(|entry| entry.ok())
                    .map(|entry| entry.file_name().to_string_lossy().into_owned())
                    .filter(|file_name| sirt_output_file_filter.accept_file_string(&dir, file_name))
                    .map(|file_name| dir.join(file_name))
                    .collect::<Vec<PathBuf>>()
            })
        });
        if let Some(file_list) = &file_list
            && file_list.len() == 1
        {
            let name = file_list[0]
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_default();
            if ui_harness::with(|harness| {
                harness.open_yes_no_dialog_base_manager_string_axis_id(
                    Some(base_manager),
                    &format!("Use {} as the tomogram?", name),
                    Some(self.axis_id),
                )
            }) {
                let display: ProcessResultDisplayHandle = self.btn_use_sirt.clone();
                self.manager.use_sirt(
                    Some(display),
                    Some(file_list[0].clone()),
                    &self.btn_sirt.get_unformatted_label().unwrap_or_default(),
                    self.axis_id,
                    self.dialog_type,
                );
            }
            return;
        }
        let chooser = FileChooser::new_base_manager(Some(base_manager));
        // Swing layout: chooser.setPreferredSize(UIParameters.getInstance()
        // .getFileChooserDimension()).
        chooser.set_file_selection_mode(file_chooser::FILES_ONLY);
        let filter: Rc<dyn FileFilter> = sirt_output_file_filter.clone();
        chooser.set_file_filter(Some(filter));
        let return_val = chooser.show_open_dialog(Some(&self.pnl_root));
        if return_val != file_chooser::APPROVE_OPTION {
            return;
        }
        let Some(file) = chooser.get_selected_file() else {
            return;
        };
        if !file.exists() {
            return;
        }
        let display: ProcessResultDisplayHandle = self.btn_use_sirt.clone();
        self.manager.use_sirt(
            Some(display),
            Some(file),
            &self.btn_sirt.get_unformatted_label().unwrap_or_default(),
            self.axis_id,
            self.dialog_type,
        );
    }

    /// Java public `isAdvanced()` (implements `RadialParent`).
    pub fn is_advanced(&self) -> bool {
        self.sirt_setup_params_header.is_advanced()
    }

    /// Java `checkpoint(TomogramState)`.
    pub fn checkpoint(&self, state: &TomogramState) {
        self.ltf_subarea_size
            .checkpoint_string(Some(&state.get_gen_sirtsetup_subarea_size(self.axis_id)));
        self.ltf_y_offset_of_subarea.checkpoint_string(Some(
            &state.get_gen_sirtsetupy_offset_of_subarea(self.axis_id),
        ));
        self.update_display();
    }

    /// Java `isDifferentFromCheckpoint()`.
    pub fn is_different_from_checkpoint(&self) -> bool {
        if self.ltf_subarea_size.is_different_from_checkpoint_void()
            || self
                .ltf_y_offset_of_subarea
                .is_different_from_checkpoint_void()
        {
            return true;
        }
        false
    }

    /// Java private `documentAction()`.
    fn document_action(&self) {
        self.update_display();
    }

    /// Java private `setToolTipText()`.
    fn set_tool_tip_text(&self) {
        // Java `ReadOnlyAutodoc autodoc = null;` then the try/catch.
        let mut autodoc: *const dyn ReadOnlyAutodoc = std::ptr::null::<Autodoc>();
        let base_manager: &'static dyn BaseManager = self.manager;
        // SAFETY: the factory returns an autodoc it keeps for the life of the
        // process.
        match unsafe {
            autodoc_factory::get_instance(
                Some(base_manager),
                Some(autodoc_factory::SIRTSETUP),
                self.axis_id,
                false,
            )
        } {
            Ok(instance) => autodoc = instance as *const Autodoc,
            // `catch (final LockException except) {}`.
            Err(LogFileError::Lock(_)) => {}
            // `catch (final LogFileException | IOException except)`:
            // `except.printStackTrace()`.
            Err(except) => eprintln!("{}", except),
        }
        // SAFETY: `autodoc` is null or an autodoc the factory keeps for the life
        // of the process.
        let autodoc: Option<&dyn ReadOnlyAutodoc> = if autodoc.is_null() {
            None
        } else {
            Some(unsafe { &*autodoc })
        };
        self.cb_subarea
            .set_tool_tip_text_string(Some("Subarea to use from the aligned stack"));
        self.ltf_y_offset_of_subarea.set_tool_tip_text(
            etomo_autodoc::get_tooltip(autodoc, Some(sirtsetup_param::Y_OFFSET_OF_SUBAREA_KEY))
                .as_deref(),
        );
        self.ltf_subarea_size.set_tool_tip_text(
            etomo_autodoc::get_tooltip(autodoc, Some(sirtsetup_param::SUBAREA_SIZE_KEY)).as_deref(),
        );
        self.ltf_leave_iterations.set_tool_tip_text(
            etomo_autodoc::get_tooltip(autodoc, Some(sirtsetup_param::LEAVE_ITERATIONS_KEY))
                .as_deref(),
        );
        self.cb_scale_to_integer.set_tool_tip_text_string(
            etomo_autodoc::get_tooltip(autodoc, Some(sirtsetup_param::SCALE_TO_INTEGER_KEY))
                .as_deref(),
        );
        self.ltf_flat_filter_fraction.set_tool_tip_text(
            etomo_autodoc::get_tooltip(autodoc, Some(sirtsetup_param::FLAT_FILTER_FRACTION_KEY))
                .as_deref(),
        );
        self.btn_3dmod_sirt.set_tool_tip_text(Some(
            "Opens a file chooser for picking SIRT iteration files to open together in 3dmod",
        ));
        self.cb_clean_up_past_start.set_tool_tip_text_string(
            etomo_autodoc::get_tooltip(autodoc, Some(sirtsetup_param::CLEAN_UP_PAST_START_KEY))
                .as_deref(),
        );
        self.cb_skip_vert_slice_output.set_tool_tip_text_string(
            etomo_autodoc::get_tooltip(autodoc, Some(sirtsetup_param::SKIP_VERT_SLICE_OUTPUT_KEY))
                .as_deref(),
        );
        self.rb_start_from_zero.set_tool_tip_text_string(
            etomo_autodoc::get_tooltip(autodoc, Some(sirtsetup_param::START_FROM_ZERO_KEY))
                .as_deref(),
        );
        self.rb_resume_from_last_iteration
            .set_tool_tip_text_string(Some("Iterate from the last existing reconstruction."));
        let tooltip =
            etomo_autodoc::get_tooltip(autodoc, Some(sirtsetup_param::RESUME_FROM_ITERATION_KEY));
        self.rb_resume_from_iteration
            .set_tool_tip_text_string(tooltip.as_deref());
        self.cmb_resume_from_iteration
            .set_tool_tip_text(tooltip.as_deref());
        self.btn_sirt.set_tool_tip_text(Some(
            "Run sirtsetup, and then run the resulting .com files with processchunks.",
        ));
        self.btn_use_sirt.set_tool_tip_text(Some(
            "Use a SIRT result as the tomogram (change the extension to .rec).",
        ));
    }
}

impl SirtsetupDisplay for SirtPanel {
    /// Java `getParameters(SirtsetupParam, boolean)`.
    fn get_parameters(&self, param: &mut SirtsetupParam, do_validation: bool) -> bool {
        self.get_parameters_sirtsetup_param_boolean(param, do_validation)
    }
}

impl Expandable for SirtPanel {
    /// Java `expand(ExpandButton)`.
    fn expand_expand_button(&self, button: &Rc<ExpandButton>) {
        // Java `if (sirtSetupParamsHeader != null)`: final and always set.
        if self.sirt_setup_params_header.equals_open_close(button) {
            self.pnl_sirtsetup_params_body
                .set_visible(button.is_expanded());
        } else if self.sirt_setup_params_header.equals_advanced_basic(button) {
            self.update_display();
        }
        let base_manager: &'static dyn BaseManager = self.manager;
        ui_harness::with(|harness| {
            harness.pack_axis_id_base_manager(Some(self.axis_id), Some(base_manager))
        });
    }

    /// Java `expand(GlobalExpandButton)`; empty.
    fn expand_global_expand_button(&self, _button: &Rc<GlobalExpandButton>) {}
}

impl RadialParent for SirtPanel {
    /// Java `isMultifilt()`.
    fn is_multifilt(&self) -> bool {
        false
    }

    /// Java `isCtf3d()`.
    fn is_ctf3d(&self) -> bool {
        false
    }

    /// Java `isAdvanced()`.
    fn is_advanced(&self) -> bool {
        SirtPanel::is_advanced(self)
    }
}

impl Run3dmodButtonContainer for SirtPanel {
    /// Java `action(String, Deferred3dmodButton, Run3dmodMenuOptions)`.
    fn action(
        &self,
        action_command: &str,
        _deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    ) {
        if Some(action_command) == self.btn_sirt.get_action_command().as_deref() {
            let display: ProcessResultDisplayHandle = self.btn_sirt.clone();
            let processing_method = self
                .parent
                .upgrade()
                .map(|parent| parent.get_processing_method());
            self.manager.sirtsetup(
                self.axis_id,
                Some(display),
                None,
                self.dialog_type,
                processing_method,
                self,
            );
        } else if Some(action_command) == self.btn_3dmod_sirt.get_action_command().as_deref() {
            self.open_files_in_imod(run_3dmod_menu_options);
        } else if Some(action_command) == self.btn_use_sirt.get_action_command().as_deref() {
            self.use_sirt();
        } else if Some(action_command) == self.rb_start_from_zero.get_action_command().as_deref()
            || Some(action_command)
                == self
                    .rb_resume_from_last_iteration
                    .get_action_command()
                    .as_deref()
            || Some(action_command)
                == self
                    .rb_resume_from_iteration
                    .get_action_command()
                    .as_deref()
        {
            self.update_display();
        } else if Some(action_command) == self.cb_subarea.get_action_command().as_deref() {
            self.load_resume_from();
        }
    }
}
