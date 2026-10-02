//! `IMOD/Etomo/src/etomo/ui/swing/AltStackPanel.java`.
//!
//! Java `public final class AltStackPanel implements ActionListener,
//! ContextMenu, Run3dmodButtonContainer, ProcessInterface, AltStackDisplay`:
//! the "Alt Stack" tab of the Post Processing dialog.  Runs alttomosetup, which
//! reprocesses an alternative stack (or even/odd pairs) through the
//! reconstruction.
//!
//! An EDT object (`ui.md`): created as `Rc<Self>` by
//! [`AltStackPanel::get_instance`]; every method takes `&self`.  The panel is
//! its own `ActionListener`: the registered closure holds a weak reference and
//! calls [`AltStackPanel::action_performed`].  The dialog's
//! `ProcessInterface` (the Post Processing dialog, which owns this panel) is
//! held weakly and upgraded where the Java passes it on.

use std::cell::Cell;
use std::path::PathBuf;
use std::rc::{Rc, Weak};

use super::alt_stack_display::AltStackDisplay;
use super::beveled_border::BeveledBorder;
use super::button_component::ButtonComponent;
use super::check_box::CheckBox;
use super::context_menu::ContextMenu;
use super::context_popup::{self, ContextPopup};
use super::cpu_gpu_panel::CpuGpuPanel;
use super::deferred_3dmod_button::Deferred3dmodButton;
use super::etched_border::EtchedBorder;
use super::labeled_text_field::LabeledTextField;
use super::multi_line_button::MultiLineButton;
use super::process_display::ProcessDisplay;
use super::process_interface::ProcessInterface;
use super::radio_button::RadioButton;
use super::run_3dmod_button::Run3dmodButton;
use super::run_3dmod_button_container::Run3dmodButtonContainer;
use super::ui_harness;
use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::alt_tomo_setup_param::{self, AltTomoSetupParam};
use crate::imod::etomo::comscript::const_tilt_param::ConstTiltParam;
use crate::imod::etomo::comscript::fortran_input_syntax_exception::FortranInputSyntaxException;
use crate::imod::etomo::comscript::tilt_param::TiltParam;
use crate::imod::etomo::jdk::{ActionEvent, ActionListener, ButtonGroup, JComponent, MouseEvent};
use crate::imod::etomo::process::imod_manager;
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::processing_method_mediator::ProcessingMethodMediator;
use crate::imod::etomo::storage::autodoc::autodoc::Autodoc;
use crate::imod::etomo::storage::autodoc::autodoc_factory;
use crate::imod::etomo::storage::autodoc::read_only_autodoc::ReadOnlyAutodoc;
use crate::imod::etomo::storage::log_file::LogFileError;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::const_etomo_number::ConstEtomoNumber;
use crate::imod::etomo::r#type::const_meta_data::ConstMetaData;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::etomo_autodoc;
use crate::imod::etomo::r#type::file_type;
use crate::imod::etomo::r#type::meta_data::MetaData;
use crate::imod::etomo::r#type::panel_id::PanelId;
use crate::imod::etomo::r#type::processing_method::ProcessingMethod;
use crate::imod::etomo::r#type::tomogram_state::TomogramState;
use crate::imod::etomo::ui::field::Field;
use crate::imod::etomo::ui::field_type::FieldType;
use crate::imod::etomo::ui::field_validation_failed_exception::FieldValidationFailedException;
use crate::imod::etomo::ui::queue_table_event::QueueTableEvent;
use crate::imod::etomo::ui::queue_table_listener::QueueTableListener;

/// Java static final `EVEN_ODD_PAIRS_LABEL`.
pub const EVEN_ODD_PAIRS_LABEL: &str = "Process even and odd pairs";
/// Java static final `ROOTNAME_ALTERNATIVE_STACK_LABEL`.
pub const ROOTNAME_ALTERNATIVE_STACK_LABEL: &str = "Rootname of alternative stack ";
/// Java static final `AXES_TO_PROCESS_LABEL`.
pub const AXES_TO_PROCESS_LABEL: &str = "Axes to process:";
/// Java static final `AXES_TO_PROCESS_BOTH_LABEL`.
pub const AXES_TO_PROCESS_BOTH_LABEL: &str = "Both";
/// Java static final `AXES_TO_PROCESS_A_ONLY_LABEL`.
pub const AXES_TO_PROCESS_A_ONLY_LABEL: &str = "A only";
/// Java static final `AXES_TO_PROCESS_B_ONLY_LABEL`.
pub const AXES_TO_PROCESS_B_ONLY_LABEL: &str = "B only";
/// Java static final `PREPROCESS_LABEL`.
pub const PREPROCESS_LABEL: &str = "Preprocess";
/// Java static final `ARCHIVE_ORIGINAL_STACK_LABEL`.
pub const ARCHIVE_ORIGINAL_STACK_LABEL: &str = "Archive original stack";
/// Java static final `CORRECT_CTF_LABEL`.
pub const CORRECT_CTF_LABEL: &str = "Correct CTF";
/// Java static final `ERASE_GOLD_LABEL`.
pub const ERASE_GOLD_LABEL: &str = "Erase gold";
/// Java static final `FILTER_IN_2D_LABEL`.
pub const FILTER_IN_2D_LABEL: &str = "Filter in 2D";
/// Java static final `TRIM_VOLUME_LABEL`.
pub const TRIM_VOLUME_LABEL: &str = "Trim volume";
/// Java static final `CLEAN_UP_INTERMEDIATE_FILES_LABEL`.
pub const CLEAN_UP_INTERMEDIATE_FILES_LABEL: &str = "Clean up intermediate files";
/// Java static final `BUTTON_OPEN_RECON_ALTERNATIVE_LABEL`.
pub const BUTTON_OPEN_RECON_ALTERNATIVE_LABEL: &str = "Open Alternative Tomogram in 3dmod";
/// Java static final `BUTTON_OPEN_RECON_EVEN_ODD_LABEL`.
pub const BUTTON_OPEN_RECON_EVEN_ODD_LABEL: &str = "Open Even and Odd Tomograms in 3dmod";
/// Java static final `BUTTON_OPEN_RECON_AXIS_A_LABEL`.
pub const BUTTON_OPEN_RECON_AXIS_A_LABEL: &str = "Open Axis A Tomogram in 3dmod";
/// Java static final `BUTTON_OPEN_RECON_AXIS_B_LABEL`.
pub const BUTTON_OPEN_RECON_AXIS_B_LABEL: &str = "Open Axis B Tomogram in 3dmod";

/// Java `public final class AltStackPanel`.
pub struct AltStackPanel {
    /// Rust-only: Java `this` (the mediator's ProcessInterface).
    self_ref: Weak<AltStackPanel>,
    /// Rust-only: Java `this` as the ActionListener registered on the
    /// components.
    action_listener: ActionListener,

    /// Java private final `pnlRoot = new JPanel()`.
    pnl_root: Rc<JComponent>,
    /// Java private final `cbProcessEvenOddPairs`.
    cb_process_even_odd_pairs: Rc<CheckBox>,
    /// Java private final `ltfRootnameOfAltStack`.
    ltf_rootname_of_alt_stack: Rc<LabeledTextField>,
    /// Java private final `strAxesToProcess = new JLabel(AXES_TO_PROCESS_LABEL)`.
    str_axes_to_process: Rc<JComponent>,
    /// Java private final `cbPreprocess`.
    cb_preprocess: Rc<CheckBox>,
    /// Java private final `cbArchiveOrigStack`.
    cb_archive_orig_stack: Rc<CheckBox>,
    /// Java private final `cbCorrectCTF`.
    cb_correct_ctf: Rc<CheckBox>,
    /// Java private final `cbEraseGold`.
    cb_erase_gold: Rc<CheckBox>,
    /// Java private final `cbFilterIn2D`.
    cb_filter_in_2d: Rc<CheckBox>,
    /// Java private final `cbTrimVolume`.
    cb_trim_volume: Rc<CheckBox>,
    /// Java private final `cbCleanUpIntermediateFiles`.
    cb_clean_up_intermediate_files: Rc<CheckBox>,

    /// Java private final `btnRunAltTomoSetup`.
    btn_run_alt_tomo_setup: Rc<MultiLineButton>,
    /// Java private final `btnOpenReconIn3dmod1`.
    btn_open_recon_in_3dmod1: Rc<Run3dmodButton>,
    /// Java private final `btnOpenReconIn3dmod2`.
    btn_open_recon_in_3dmod2: Rc<Run3dmodButton>,
    /// Java private final `btnRestoreSwappedFiles`.
    btn_restore_swapped_files: Rc<MultiLineButton>,
    /// Java private final `manager`.
    manager: &'static ApplicationManager,
    /// Java private final `axisID`.
    axis_id: AxisID,
    /// Java private final `dialogType`.
    dialog_type: DialogType,
    /// Java private final `cpuGpuPanel`.
    cpu_gpu_panel: Rc<CpuGpuPanel>,
    /// Java private final `mediator`.
    mediator: Rc<ProcessingMethodMediator>,
    /// Java private final `processInterface` (the owning dialog, held weakly).
    process_interface: Weak<dyn ProcessInterface>,
    /// Java private final `tomogramState`.
    tomogram_state: &'static TomogramState,
    /// Java private final `panelId`.
    panel_id: PanelId,
    /// Java private final `bgAxesToProcess`; null for a single axis dataset.
    bg_axes_to_process: Option<Rc<ButtonGroup>>,
    /// Java private final `rbBoth`; null for a single axis dataset.
    rb_both: Option<Rc<RadioButton>>,
    /// Java private final `rbAonly`; null for a single axis dataset.
    rb_aonly: Option<Rc<RadioButton>>,
    /// Java private final `rbBonly`; null for a single axis dataset.
    rb_bonly: Option<Rc<RadioButton>>,

    /// Java private `isRootnameEvenFile`.
    is_rootname_even_file: Cell<bool>,
    /// Java private `isRootnameOddFile`.
    is_rootname_odd_file: Cell<bool>,
    /// Java private `evenOddPairsFilesFound`.
    even_odd_pairs_files_found: Cell<bool>,
    /// Java private `isGoldEraserComFile`.
    is_gold_eraser_com_file: Cell<bool>,
    /// Java private `isGoldEraserComFileA`.
    is_gold_eraser_com_file_a: Cell<bool>,
    /// Java private `isGoldEraserComFileB`.
    is_gold_eraser_com_file_b: Cell<bool>,
    /// Java private `isEraserLogFile`.
    is_eraser_log_file: Cell<bool>,
    /// Java private `isEraserLogFileA`.
    is_eraser_log_file_a: Cell<bool>,
    /// Java private `isEraserLogFileB`.
    is_eraser_log_file_b: Cell<bool>,
    /// Java private `isCtfCorrectionLogFile`.
    is_ctf_correction_log_file: Cell<bool>,
    /// Java private `isCtfCorrectionLogFileA`.
    is_ctf_correction_log_file_a: Cell<bool>,
    /// Java private `isCtfCorrectionLogFileB`.
    is_ctf_correction_log_file_b: Cell<bool>,
    /// Java private `isGoldEraserLogFile`.
    is_gold_eraser_log_file: Cell<bool>,
    /// Java private `isGoldEraserLogFileA`.
    is_gold_eraser_log_file_a: Cell<bool>,
    /// Java private `isGoldEraserLogFileB`.
    is_gold_eraser_log_file_b: Cell<bool>,
    /// Java private `isMtfFilterLogFile`.
    is_mtf_filter_log_file: Cell<bool>,
    /// Java private `isMtfFilterLogFileA`.
    is_mtf_filter_log_file_a: Cell<bool>,
    /// Java private `isMtfFilterLogFileB`.
    is_mtf_filter_log_file_b: Cell<bool>,
    /// Java private `isDialogOpenedFirstTime`.
    is_dialog_opened_first_time: Cell<bool>,
}

impl AltStackPanel {
    /// Java private constructor `AltStackPanel(ApplicationManager, AxisID,
    /// DialogType, ProcessInterface)`, with the field initializers.
    fn new(
        manager: &'static ApplicationManager,
        axis_id: AxisID,
        dialog_type: DialogType,
        process_interface: Weak<dyn ProcessInterface>,
    ) -> Rc<AltStackPanel> {
        let base_manager: &'static dyn BaseManager = manager;
        let instance = Rc::new_cyclic(|self_ref: &Weak<AltStackPanel>| {
            let container: Weak<dyn Run3dmodButtonContainer> = self_ref.clone();
            // Java `this` as an ActionListener.
            let adaptee = self_ref.clone();
            let action_listener: ActionListener = Rc::new(move |event: &ActionEvent| {
                if let Some(adaptee) = adaptee.upgrade() {
                    adaptee.action_performed(Some(event));
                }
            });
            // Field initializers.
            let pnl_root = JComponent::new_panel();
            let cb_process_even_odd_pairs = CheckBox::new_string(Some(EVEN_ODD_PAIRS_LABEL));
            let ltf_rootname_of_alt_stack = LabeledTextField::new_field_type_string(
                FieldType::String,
                Some(ROOTNAME_ALTERNATIVE_STACK_LABEL),
            );
            let str_axes_to_process = JComponent::new_label(AXES_TO_PROCESS_LABEL);
            let cb_preprocess = CheckBox::new_string(Some(PREPROCESS_LABEL));
            let cb_archive_orig_stack = CheckBox::new_string(Some(ARCHIVE_ORIGINAL_STACK_LABEL));
            let cb_correct_ctf = CheckBox::new_string(Some(CORRECT_CTF_LABEL));
            let cb_erase_gold = CheckBox::new_string(Some(ERASE_GOLD_LABEL));
            let cb_filter_in_2d = CheckBox::new_string(Some(FILTER_IN_2D_LABEL));
            let cb_trim_volume = CheckBox::new_string(Some(TRIM_VOLUME_LABEL));
            let cb_clean_up_intermediate_files =
                CheckBox::new_string(Some(CLEAN_UP_INTERMEDIATE_FILES_LABEL));
            // Constructor body.
            let tomogram_state = manager.get_state();
            let panel_id = PanelId::PostAltStack;
            let (bg_axes_to_process, rb_both, rb_aonly, rb_bonly) = if manager.is_dual_axis() {
                let bg_axes_to_process = ButtonGroup::new();
                let rb_both = RadioButton::new_string_button_group(
                    Some(AXES_TO_PROCESS_BOTH_LABEL),
                    Some(&bg_axes_to_process),
                );
                let rb_aonly = RadioButton::new_string_button_group(
                    Some(AXES_TO_PROCESS_A_ONLY_LABEL),
                    Some(&bg_axes_to_process),
                );
                let rb_bonly = RadioButton::new_string_button_group(
                    Some(AXES_TO_PROCESS_B_ONLY_LABEL),
                    Some(&bg_axes_to_process),
                );
                (
                    Some(bg_axes_to_process),
                    Some(rb_both),
                    Some(rb_aonly),
                    Some(rb_bonly),
                )
            } else {
                (None, None, None, None)
            };
            let btn_run_alt_tomo_setup = MultiLineButton::new_string(Some("Run"));
            let btn_open_recon_in_3dmod1 =
                Run3dmodButton::get_3dmod_instance_string_run_3dmod_button_container(
                    Some("Open Tomogram in 3dmod"),
                    Some(container.clone()),
                );
            let btn_open_recon_in_3dmod2 =
                Run3dmodButton::get_3dmod_instance_string_run_3dmod_button_container(
                    Some("Open Tomogram B in 3dmod"),
                    Some(container),
                );
            let btn_restore_swapped_files =
                MultiLineButton::new_string(Some("Restore Swapped Files"));
            let cpu_gpu_panel =
                CpuGpuPanel::get_instance(manager, axis_id, PanelId::PostAltStack, 3, true, -1);
            let mediator = manager
                .get_processing_method_mediator(Some(axis_id))
                .expect("processing method mediator on the event dispatch thread");
            AltStackPanel {
                self_ref: self_ref.clone(),
                action_listener,
                pnl_root,
                cb_process_even_odd_pairs,
                ltf_rootname_of_alt_stack,
                str_axes_to_process,
                cb_preprocess,
                cb_archive_orig_stack,
                cb_correct_ctf,
                cb_erase_gold,
                cb_filter_in_2d,
                cb_trim_volume,
                cb_clean_up_intermediate_files,
                btn_run_alt_tomo_setup,
                btn_open_recon_in_3dmod1,
                btn_open_recon_in_3dmod2,
                btn_restore_swapped_files,
                manager,
                axis_id,
                dialog_type,
                cpu_gpu_panel,
                mediator,
                process_interface,
                tomogram_state,
                panel_id,
                bg_axes_to_process,
                rb_both,
                rb_aonly,
                rb_bonly,
                is_rootname_even_file: Cell::new(false),
                is_rootname_odd_file: Cell::new(false),
                even_odd_pairs_files_found: Cell::new(false),
                is_gold_eraser_com_file: Cell::new(false),
                is_gold_eraser_com_file_a: Cell::new(false),
                is_gold_eraser_com_file_b: Cell::new(false),
                is_eraser_log_file: Cell::new(false),
                is_eraser_log_file_a: Cell::new(false),
                is_eraser_log_file_b: Cell::new(false),
                is_ctf_correction_log_file: Cell::new(false),
                is_ctf_correction_log_file_a: Cell::new(false),
                is_ctf_correction_log_file_b: Cell::new(false),
                is_gold_eraser_log_file: Cell::new(false),
                is_gold_eraser_log_file_a: Cell::new(false),
                is_gold_eraser_log_file_b: Cell::new(false),
                is_mtf_filter_log_file: Cell::new(false),
                is_mtf_filter_log_file_a: Cell::new(false),
                is_mtf_filter_log_file_b: Cell::new(false),
                is_dialog_opened_first_time: Cell::new(true),
            }
        });
        // The rest of the Java constructor body.
        let process_interface = instance.process_interface.upgrade();
        if let Some(process_interface) = &process_interface {
            instance
                .mediator
                .register_process_interface(process_interface.clone());
        }
        instance
            .cpu_gpu_panel
            .set_alt_stack_process_interface(process_interface);
        instance
            .cb_process_even_odd_pairs
            .set_text(Some(EVEN_ODD_PAIRS_LABEL));
        instance
            .cb_process_even_odd_pairs
            .set_alternate_text(Some(&format!("{EVEN_ODD_PAIRS_LABEL} (Files not found)")));
        instance.is_rootname_even_file.set(
            file_type::CLASS
                .alt_stack_rootname_even_file
                .exists(Some(base_manager), Some(AxisID::Only)),
        );
        instance.is_rootname_odd_file.set(
            file_type::CLASS
                .alt_stack_rootname_odd_file
                .exists(Some(base_manager), Some(AxisID::Only)),
        );
        if instance.is_rootname_even_file.get() && instance.is_rootname_odd_file.get() {
            instance.even_odd_pairs_files_found.set(true);
        } else {
            instance.cb_process_even_odd_pairs.switch_text(true);
        }
        instance
    }

    /// Java package-private static `getInstance(ApplicationManager, AxisID,
    /// DialogType, ProcessInterface)`.
    pub fn get_instance(
        manager: &'static ApplicationManager,
        axis_id: AxisID,
        dialog_type: DialogType,
        process_interface: Weak<dyn ProcessInterface>,
    ) -> Rc<AltStackPanel> {
        let instance = AltStackPanel::new(manager, axis_id, dialog_type, process_interface);
        instance.create_panel();
        instance.set_tool_tip_text();
        instance.add_listeners();
        instance
    }

    /// Java package-private `getComponent()`.
    pub fn get_component(&self) -> Rc<JComponent> {
        self.pnl_root.clone()
    }

    /// Java private `addListeners()`.
    fn add_listeners(&self) {
        // Swing mouse: pnlRoot.addMouseListener(new GenericMouseAdapter(this)).
        // Mouse events are not modelled; the adapter's only effect is to call
        // popUpContextMenu on a right-button press, which a driver calls
        // directly.
        self.cb_process_even_odd_pairs
            .add_action_listener(Some(self.action_listener.clone()));
        if let (Some(rb_both), Some(rb_aonly), Some(rb_bonly)) =
            (&self.rb_both, &self.rb_aonly, &self.rb_bonly)
        {
            rb_both.add_action_listener(self.action_listener.clone());
            rb_aonly.add_action_listener(self.action_listener.clone());
            rb_bonly.add_action_listener(self.action_listener.clone());
        }
        self.cb_preprocess
            .add_action_listener(Some(self.action_listener.clone()));
        self.cb_archive_orig_stack
            .add_action_listener(Some(self.action_listener.clone()));
        self.cb_correct_ctf
            .add_action_listener(Some(self.action_listener.clone()));
        self.cb_erase_gold
            .add_action_listener(Some(self.action_listener.clone()));
        self.cb_filter_in_2d
            .add_action_listener(Some(self.action_listener.clone()));
        self.cb_trim_volume
            .add_action_listener(Some(self.action_listener.clone()));
        self.cb_clean_up_intermediate_files
            .add_action_listener(Some(self.action_listener.clone()));
        self.btn_run_alt_tomo_setup
            .add_action_listener(self.action_listener.clone());
        self.btn_open_recon_in_3dmod1
            .add_action_listener(self.action_listener.clone());
        self.btn_open_recon_in_3dmod2
            .add_action_listener(self.action_listener.clone());
        self.btn_restore_swapped_files
            .add_action_listener(self.action_listener.clone());
    }

    /// Java private `createPanel()`.
    fn create_panel(&self) {
        let pnl_outer_panel = JComponent::new_panel();
        let _pnl_use_gpu = JComponent::new_panel();
        let pnl_process_even_odd_pairs = JComponent::new_panel();
        let pnl_rootname_alternative_stack = JComponent::new_panel();
        let pnl_axes_to_process = JComponent::new_panel();
        let pnl_processing_steps_to_run = JComponent::new_panel();
        let pnl_preprocess = JComponent::new_panel();
        let pnl_ctf_correct = JComponent::new_panel();
        let pnl_erase_gold = JComponent::new_panel();
        let pnl_filter_in_2d = JComponent::new_panel();
        let pnl_trim_volume = JComponent::new_panel();
        let pnl_clean_up_intermediate_files = JComponent::new_panel();
        let pnl_buttons = JComponent::new_panel();
        let pnl_buttons_line1 = JComponent::new_panel();
        let pnl_buttons_line2 = JComponent::new_panel();
        // init
        if let Some(rb_both) = &self.rb_both {
            rb_both.set_selected_boolean(true);
        }
        self.cpu_gpu_panel.msg_processing_method_changed(true, true);
        self.set_checkboxes_first_time_only();
        self.cb_trim_volume
            .set_enabled(!self.manager.is_dual_axis());
        self.ltf_rootname_of_alt_stack
            .set_required(self.manager.is_dual_axis());
        // Root
        // Swing layout: pnlRoot X_AXIS BoxLayout.
        self.pnl_root.set_border_title(
            BeveledBorder::new(Some("Alternative Stack"))
                .get_border()
                .get_title()
                .as_deref(),
        );
        self.pnl_root.add(&pnl_outer_panel);
        // Swing layout: pnlOuterPanel Y_AXIS BoxLayout; rigid areas (x0_y5)
        // between its children.
        if !self.manager.is_dual_axis() {
            pnl_outer_panel.add(&pnl_process_even_odd_pairs);
        }
        pnl_outer_panel.add(&pnl_rootname_alternative_stack);
        if self.manager.is_dual_axis() {
            pnl_outer_panel.add(&pnl_axes_to_process);
        }
        pnl_outer_panel.add(&self.cpu_gpu_panel.get_component());
        pnl_outer_panel.add(&pnl_processing_steps_to_run);
        pnl_outer_panel.add(&pnl_clean_up_intermediate_files);
        pnl_outer_panel.add(&pnl_buttons);
        // Process even and odd pairs
        // Swing layout: pnlProcessEvenOddPairs X_AXIS BoxLayout, trailing glue.
        pnl_process_even_odd_pairs.add(&self.cb_process_even_odd_pairs.get_component());
        // Rootname of alternative stack
        // Swing layout: pnlRootnameAlternativeStack X_AXIS BoxLayout, leading
        // rigid area (x2_y0), trailing glue.
        pnl_rootname_alternative_stack.add(&self.ltf_rootname_of_alt_stack.get_component());
        // Axes to process
        if let (Some(rb_both), Some(rb_aonly), Some(rb_bonly)) =
            (&self.rb_both, &self.rb_aonly, &self.rb_bonly)
        {
            // Swing layout: pnlAxesToProcess X_AXIS BoxLayout, leading rigid area
            // (x2_y0), trailing glue.
            pnl_axes_to_process.add(&self.str_axes_to_process);
            pnl_axes_to_process.add(&rb_both.get_component());
            pnl_axes_to_process.add(&rb_aonly.get_component());
            pnl_axes_to_process.add(&rb_bonly.get_component());
        }
        // Processing steps to run
        // Swing layout: pnlProcessingStepsToRun Y_AXIS BoxLayout; rigid areas
        // (x0_y2) between the step panels, trailing glue.
        pnl_processing_steps_to_run.set_border_title(
            EtchedBorder::new(Some("Processing Steps to Run"))
                .get_border()
                .get_title()
                .as_deref(),
        );
        pnl_processing_steps_to_run.add(&pnl_preprocess);
        pnl_processing_steps_to_run.add(&pnl_ctf_correct);
        pnl_processing_steps_to_run.add(&pnl_erase_gold);
        pnl_processing_steps_to_run.add(&pnl_filter_in_2d);
        pnl_processing_steps_to_run.add(&pnl_trim_volume);
        // Swing layout: pnlPreprocess X_AXIS BoxLayout, rigid area (x20_y0)
        // between the check boxes, trailing glue.
        pnl_preprocess.add(&self.cb_preprocess.get_component());
        pnl_preprocess.add(&self.cb_archive_orig_stack.get_component());
        // Swing layout: X_AXIS BoxLayouts with trailing glue for the remaining
        // step panels.
        pnl_ctf_correct.add(&self.cb_correct_ctf.get_component());
        pnl_erase_gold.add(&self.cb_erase_gold.get_component());
        pnl_filter_in_2d.add(&self.cb_filter_in_2d.get_component());
        pnl_trim_volume.add(&self.cb_trim_volume.get_component());
        // Clean up intermediate files
        // Swing layout: pnlCleanUpIntermediateFiles X_AXIS BoxLayout, trailing glue.
        pnl_clean_up_intermediate_files.add(&self.cb_clean_up_intermediate_files.get_component());
        // buttons
        // Swing layout: pnlButtons Y_AXIS BoxLayout, rigid area (x0_y5) between
        // the lines.
        pnl_buttons.add(&pnl_buttons_line1);
        pnl_buttons.add(&pnl_buttons_line2);
        // Swing layout: pnlButtonsLine1 X_AXIS BoxLayout, glue and rigid areas
        // (x5_y0) around the buttons.
        pnl_buttons_line1.add(&self.btn_run_alt_tomo_setup.get_component());
        pnl_buttons_line1.add(&self.btn_open_recon_in_3dmod1.get_component());
        if self.manager.is_dual_axis() {
            pnl_buttons_line1.add(&self.btn_open_recon_in_3dmod2.get_component());
        }
        // Swing layout: pnlButtonsLine2 X_AXIS BoxLayout.
        pnl_buttons_line2.add(&self.btn_restore_swapped_files.get_component());
        // Swing layout: UIUtilities.setButtonSizeAll(pnlButtonsLine1/2,
        // UIParameters.getInstance().getButtonDimension()).
        self.update_display();
    }

    /// Java private `updateDisplay()`.
    fn update_display(&self) {
        self.ltf_rootname_of_alt_stack.set_enabled(
            !(self.cb_process_even_odd_pairs.is_enabled()
                && self.cb_process_even_odd_pairs.is_selected()),
        );
        if !self.manager.is_dual_axis() {
            self.ltf_rootname_of_alt_stack
                .set_required(self.ltf_rootname_of_alt_stack.is_enabled());
        }
        self.cb_archive_orig_stack
            .set_enabled(self.cb_preprocess.is_enabled() && self.cb_preprocess.is_selected());
        if self.manager.is_dual_axis() {
            self.btn_open_recon_in_3dmod1
                .set_text(Some(BUTTON_OPEN_RECON_AXIS_A_LABEL));
            self.btn_open_recon_in_3dmod2
                .set_text(Some(BUTTON_OPEN_RECON_AXIS_B_LABEL));
        } else if self.cb_process_even_odd_pairs.is_enabled()
            && self.cb_process_even_odd_pairs.is_selected()
        {
            self.btn_open_recon_in_3dmod1
                .set_text(Some(BUTTON_OPEN_RECON_EVEN_ODD_LABEL));
        } else {
            self.btn_open_recon_in_3dmod1
                .set_text(Some(BUTTON_OPEN_RECON_ALTERNATIVE_LABEL));
        }
        let manager: &'static dyn BaseManager = self.manager;
        ui_harness::with(|harness| {
            harness.pack_axis_id_base_manager(Some(self.axis_id), Some(manager))
        });
    }

    /// Java private `setToolTipText()`.
    fn set_tool_tip_text(&self) {
        // Java `ReadOnlyAutodoc autodoc = null;` then the try/catch.
        let mut autodoc: *const dyn ReadOnlyAutodoc = std::ptr::null::<Autodoc>();
        let manager: &'static dyn BaseManager = self.manager;
        // Java passes a null AxisID; `AutodocFactory.getInstance` takes the axis
        // by value here (see NEEDS).
        match unsafe {
            autodoc_factory::get_instance(
                Some(manager),
                Some(autodoc_factory::ALT_TOMO_SETUP),
                AxisID::Only,
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
        if let Some(read_only_autodoc) = autodoc {
            let _autodoc_name = read_only_autodoc.get_autodoc_name();
            self.cb_process_even_odd_pairs.set_tool_tip_text_string(
                etomo_autodoc::get_tooltip(autodoc, Some(alt_tomo_setup_param::EVEN_AND_ODD_PAIRS))
                    .as_deref(),
            );
            self.ltf_rootname_of_alt_stack.set_tool_tip_text(
                etomo_autodoc::get_tooltip(
                    autodoc,
                    Some(alt_tomo_setup_param::ROOTNAME_TO_PROCESS),
                )
                .as_deref(),
            );
            if let (Some(rb_both), Some(rb_aonly), Some(rb_bonly)) =
                (&self.rb_both, &self.rb_aonly, &self.rb_bonly)
            {
                rb_both.set_tool_tip_text_string(
                    etomo_autodoc::get_tooltip(
                        autodoc,
                        Some(alt_tomo_setup_param::AXIS_TO_PROCESS),
                    )
                    .as_deref(),
                );
                rb_aonly.set_tool_tip_text_string(
                    etomo_autodoc::get_tooltip(
                        autodoc,
                        Some(alt_tomo_setup_param::AXIS_TO_PROCESS),
                    )
                    .as_deref(),
                );
                rb_bonly.set_tool_tip_text_string(
                    etomo_autodoc::get_tooltip(
                        autodoc,
                        Some(alt_tomo_setup_param::AXIS_TO_PROCESS),
                    )
                    .as_deref(),
                );
            }
            self.cb_preprocess.set_tool_tip_text_string(
                etomo_autodoc::get_tooltip(
                    autodoc,
                    Some(alt_tomo_setup_param::PREPROCESS_FOR_EXTREMES),
                )
                .as_deref(),
            );
            self.cb_correct_ctf.set_tool_tip_text_string(
                etomo_autodoc::get_tooltip(autodoc, Some(alt_tomo_setup_param::CORRECT_CTF))
                    .as_deref(),
            );
            self.cb_erase_gold.set_tool_tip_text_string(
                etomo_autodoc::get_tooltip(autodoc, Some(alt_tomo_setup_param::ERASE_FIDUCIALS))
                    .as_deref(),
            );
            self.cb_filter_in_2d.set_tool_tip_text_string(
                etomo_autodoc::get_tooltip(autodoc, Some(alt_tomo_setup_param::FILTER_IN_2D))
                    .as_deref(),
            );
            self.cb_trim_volume.set_tool_tip_text_string(
                etomo_autodoc::get_tooltip(autodoc, Some(alt_tomo_setup_param::TRIM_VOLUME))
                    .as_deref(),
            );
            self.cb_clean_up_intermediate_files
                .set_tool_tip_text_string(
                    etomo_autodoc::get_tooltip(
                        autodoc,
                        Some(alt_tomo_setup_param::CLEAN_UP_INTERMEDIATES),
                    )
                    .as_deref(),
                );
            self.btn_run_alt_tomo_setup
                .set_tool_tip_text(Some("Run Alttomosetup then process the alternative stack"));
            self.btn_open_recon_in_3dmod1.set_tool_tip_text(Some(
                "Open reconstruction from the alternative stack in 3dmod",
            ));
            self.btn_open_recon_in_3dmod2.set_tool_tip_text(Some(
                "Open reconstruction from the alternative stack in 3dmod",
            ));
            self.btn_restore_swapped_files.set_tool_tip_text(
                etomo_autodoc::get_tooltip(
                    autodoc,
                    Some(alt_tomo_setup_param::JUST_RESTORE_INITIAL_SET),
                )
                .as_deref(),
            );
        }
    }

    /// Java public `openAltStackSingleAxisTomograms(Run3dmodMenuOptions)`.
    pub fn open_alt_stack_single_axis_tomograms(
        &self,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    ) {
        let tomogram_state = self.manager.get_state();
        let is_alt_tomo_trim_vol_checked = tomogram_state.is_alt_tomo_trim_vol_checked();
        if self.cb_process_even_odd_pairs.is_visible()
            && self.cb_process_even_odd_pairs.is_enabled()
            && self.cb_process_even_odd_pairs.is_selected()
        {
            // Java passes a null Run3dmodMenuOptions from the action listener;
            // ImodState.open replaces null with a new Run3dmodMenuOptions() (the
            // default value).
            if !is_alt_tomo_trim_vol_checked {
                self.manager.open_even_odd_files_in_imod(
                    self.axis_id,
                    imod_manager::ALT_TOMO_SETUP_EVEN_ODD_FULL_TOMOGRAM_KEY,
                    run_3dmod_menu_options.unwrap_or_default(),
                    is_alt_tomo_trim_vol_checked,
                );
            } else {
                self.manager.open_even_odd_files_in_imod(
                    self.axis_id,
                    imod_manager::ALT_TOMO_SETUP_EVEN_ODD_TOMOGRAM_KEY,
                    run_3dmod_menu_options.unwrap_or_default(),
                    is_alt_tomo_trim_vol_checked,
                );
            }
        } else {
            let manager: &'static dyn BaseManager = self.manager;
            let rootname = self.ltf_rootname_of_alt_stack.get_text_void();
            let property_user_dir = self.manager.get_property_user_dir();
            let file: Option<PathBuf> = if !is_alt_tomo_trim_vol_checked {
                file_type::CLASS
                    .tilt_output_single
                    .get_file_with_property_user_dir(
                        Some(manager),
                        rootname.as_deref(),
                        Some(AxisType::SingleAxis),
                        Some(self.axis_id),
                        property_user_dir.as_deref(),
                    )
            } else {
                file_type::CLASS
                    .alt_stack_tomogram
                    .get_file_with_property_user_dir(
                        Some(manager),
                        rootname.as_deref(),
                        Some(AxisType::SingleAxis),
                        Some(self.axis_id),
                        property_user_dir.as_deref(),
                    )
            };
            // Java passes the (never null in practice) file on; a null file
            // would throw in the manager.  The translation skips the open.
            if let Some(file) = file {
                self.manager.open_alternative_tomogram(
                    &file,
                    self.axis_id,
                    AxisType::SingleAxis,
                    run_3dmod_menu_options,
                    imod_manager::ALT_TOMO_SETUP_TOMOGRAM_KEY,
                    rootname.as_deref(),
                    is_alt_tomo_trim_vol_checked,
                );
            }
        }
    }

    /// Java public `actionPerformed(ActionEvent)` (the panel is its own
    /// ActionListener).  `None` is a null event.
    pub fn action_performed(&self, event: Option<&ActionEvent>) {
        // Upstream bug fixed in translation (AltStackPanel.java:522): a null
        // event or action command reaches `actionCommand.equals(...)` in
        // `action` and throws NullPointerException.  The translation passes an
        // empty command, which matches no component.
        let action_command = event.and_then(|event| event.get_action_command());
        self.action(action_command.unwrap_or(""), None, None);
    }

    /// Java public `addQueueListener(ButtonComponent)`; empty.
    pub fn add_queue_listener(&self, _component: Option<Rc<dyn ButtonComponent>>) {}

    /// Java package-private `reregisterProcessingMethodMediator()`.
    pub fn reregister_processing_method_mediator(&self) {
        self.cpu_gpu_panel.reregister_processing_method_mediator();
    }

    /// Java package-private `checkIfFilesExist()`.
    pub fn check_if_files_exist(&self) {
        let manager: &'static dyn BaseManager = self.manager;
        if self.manager.is_dual_axis() {
            // golderaser.com is checked every time we visit the tab
            self.is_gold_eraser_com_file_a.set(
                file_type::CLASS
                    .gold_eraser_comscript
                    .exists(Some(manager), Some(AxisID::First)),
            );
            self.is_gold_eraser_com_file_b.set(
                file_type::CLASS
                    .gold_eraser_comscript
                    .exists(Some(manager), Some(AxisID::Second)),
            );
            if self.is_dialog_opened_first_time.get() {
                // Below .com files are checked every time we visit the tab
                self.is_eraser_log_file_a.set(
                    file_type::CLASS
                        .eraser_log
                        .exists(Some(manager), Some(AxisID::First)),
                );
                self.is_eraser_log_file_b.set(
                    file_type::CLASS
                        .eraser_log
                        .exists(Some(manager), Some(AxisID::Second)),
                );
                self.is_ctf_correction_log_file_a.set(
                    file_type::CLASS
                        .ctf_correction_log
                        .exists(Some(manager), Some(AxisID::First)),
                );
                self.is_ctf_correction_log_file_b.set(
                    file_type::CLASS
                        .ctf_correction_log
                        .exists(Some(manager), Some(AxisID::Second)),
                );
                self.is_gold_eraser_log_file_a.set(
                    file_type::CLASS
                        .gold_eraser_log
                        .exists(Some(manager), Some(AxisID::First)),
                );
                self.is_gold_eraser_log_file_b.set(
                    file_type::CLASS
                        .gold_eraser_log
                        .exists(Some(manager), Some(AxisID::Second)),
                );
                self.is_mtf_filter_log_file_a.set(
                    file_type::CLASS
                        .mtf_filter_log
                        .exists(Some(manager), Some(AxisID::First)),
                );
                self.is_mtf_filter_log_file_b.set(
                    file_type::CLASS
                        .mtf_filter_log
                        .exists(Some(manager), Some(AxisID::Second)),
                );
                self.is_dialog_opened_first_time.set(false);
            }
        } else {
            self.is_gold_eraser_com_file.set(
                file_type::CLASS
                    .gold_eraser_comscript
                    .exists(Some(manager), Some(AxisID::Only)),
            );
            if self.is_dialog_opened_first_time.get() {
                self.is_eraser_log_file.set(
                    file_type::CLASS
                        .eraser_log
                        .exists(Some(manager), Some(AxisID::Only)),
                );

                self.is_ctf_correction_log_file.set(
                    file_type::CLASS
                        .ctf_correction_log
                        .exists(Some(manager), Some(AxisID::Only)),
                );
                self.is_gold_eraser_log_file.set(
                    file_type::CLASS
                        .gold_eraser_log
                        .exists(Some(manager), Some(AxisID::Only)),
                );
                self.is_mtf_filter_log_file.set(
                    file_type::CLASS
                        .mtf_filter_log
                        .exists(Some(manager), Some(AxisID::Only)),
                );
                self.is_dialog_opened_first_time.set(false);
            }
        }

        self.enable_or_disable_cb_erase_gold();
        self.update_display();
    }

    /// Java private `setCheckboxesFirstTimeOnly()`.
    fn set_checkboxes_first_time_only(&self) {
        if self.manager.is_dual_axis() {
            self.cb_preprocess.set_selected_boolean(
                self.is_eraser_log_file_a.get() && self.is_eraser_log_file_b.get(),
            );
            self.cb_correct_ctf.set_selected_boolean(
                self.is_ctf_correction_log_file_a.get() && self.is_ctf_correction_log_file_b.get(),
            );
            self.cb_erase_gold.set_selected_boolean(
                self.is_gold_eraser_log_file_a.get()
                    && self.is_gold_eraser_log_file_b.get()
                    && self.is_gold_eraser_com_file_a.get()
                    && self.is_gold_eraser_com_file_b.get(),
            );
            self.cb_filter_in_2d.set_selected_boolean(
                self.is_mtf_filter_log_file_a.get() && self.is_mtf_filter_log_file_b.get(),
            );
        } else {
            self.cb_preprocess
                .set_selected_boolean(self.is_eraser_log_file.get());
            self.cb_correct_ctf
                .set_selected_boolean(self.is_ctf_correction_log_file.get());
            self.cb_erase_gold.set_selected_boolean(
                self.is_gold_eraser_log_file.get() && self.is_gold_eraser_com_file.get(),
            );
            self.cb_filter_in_2d
                .set_selected_boolean(self.is_mtf_filter_log_file.get());
            self.cb_trim_volume.set_selected_boolean(
                !self
                    .tomogram_state
                    .is_post_proc_trim_vol_input_n_rows_null(),
            );
        }
    }

    /// Java package-private `getParameters(MetaData) throws
    /// FortranInputSyntaxException`.
    pub fn get_parameters_meta_data(
        &self,
        meta_data: &MetaData,
    ) -> Result<(), FortranInputSyntaxException> {
        self.cpu_gpu_panel
            .get_parameters_panel_id_meta_data(self.panel_id, meta_data)?;
        meta_data.set_alt_tomo_rootname(self.ltf_rootname_of_alt_stack.get_text_void().as_deref());
        meta_data.set_alt_tomo_trim_volume(self.cb_trim_volume.is_selected());
        meta_data.set_alt_tomo_archive_orig_stack(self.cb_archive_orig_stack.is_selected());
        Ok(())
    }

    /// Java package-private `setParameters(ConstMetaData)`.
    pub fn set_parameters_const_meta_data(&self, meta_data: &dyn ConstMetaData) {
        self.cpu_gpu_panel
            .set_alt_stack_process_interface(self.process_interface.upgrade());
        let tilt_parallel = meta_data.get_tilt_parallel(self.axis_id, self.panel_id);
        self.cpu_gpu_panel
            .set_parameters_const_meta_data_const_etomo_number(
                meta_data,
                tilt_parallel.as_ref().map(|value| {
                    let value: &ConstEtomoNumber = value;
                    value
                }),
            );
        self.ltf_rootname_of_alt_stack
            .set_text_string(Some(&meta_data.get_alt_tomo_rootname()));
        self.cb_trim_volume
            .set_selected_boolean(meta_data.is_alt_tomo_trim_volume());
        self.cb_archive_orig_stack
            .set_selected_boolean(meta_data.is_alt_tomo_archive_orig_stack());
        self.update_display();
    }

    /// Java package-private `setParameters(ConstTiltParam, boolean)`.
    pub fn set_parameters_const_tilt_param_boolean(
        &self,
        tilt_param: &dyn ConstTiltParam,
        initialize: bool,
    ) {
        self.cpu_gpu_panel
            .set_alt_stack_process_interface(self.process_interface.upgrade());
        self.cpu_gpu_panel
            .set_parameters_const_tilt_param_boolean(tilt_param, initialize);
    }

    /// Java package-private `setParameters(AltTomoSetupParam)`.
    pub fn set_parameters_alt_tomo_setup_param(&self, param: &AltTomoSetupParam) {
        self.cb_process_even_odd_pairs
            .set_selected_boolean(param.is_even_and_odd_pairs());
        if param.is_rootname_to_process() {
            self.ltf_rootname_of_alt_stack
                .set_text_string(Some(&param.get_rootname_to_process()));
        }
        if let (Some(_rb_both), Some(rb_aonly), Some(rb_bonly)) =
            (&self.rb_both, &self.rb_aonly, &self.rb_bonly)
        {
            let mut _axis_to_process = String::new();
            if param.is_axis_to_process() {
                let axis_to_process = param.get_axis_to_process();
                if axis_to_process == "B" || axis_to_process == "b" {
                    rb_bonly.set_selected_boolean(true);
                }
                if axis_to_process == "A" || axis_to_process == "a" {
                    rb_aonly.set_selected_boolean(true);
                }
                _axis_to_process = axis_to_process;
            }
        }
        let mut _preprocess_for_extremes = 0;
        if param.is_preprocess_for_extremes() {
            let preprocess_for_extremes = param.get_preprocess_for_extremes();
            if preprocess_for_extremes == 0 {
                self.cb_preprocess.set_selected_boolean(false);
                self.cb_archive_orig_stack.set_enabled(false);
            }
            if preprocess_for_extremes == 1 {
                self.cb_preprocess.set_selected_boolean(true);
                self.cb_archive_orig_stack.set_enabled(false);
            }
            if preprocess_for_extremes == 2 {
                self.cb_preprocess.set_selected_boolean(true);
                self.cb_archive_orig_stack.set_enabled(true);
                self.cb_archive_orig_stack.set_selected_boolean(true);
            }
            _preprocess_for_extremes = preprocess_for_extremes;
        }
        self.cb_correct_ctf
            .set_selected_boolean(param.is_correct_ctf());
        self.cb_erase_gold
            .set_selected_boolean(param.is_erase_fiducials());
        self.cb_filter_in_2d
            .set_selected_boolean(param.is_filter_in_2d());
        self.cb_trim_volume
            .set_selected_boolean(param.is_trim_volume());
        self.cb_clean_up_intermediate_files
            .set_selected_boolean(param.is_clean_up_intermediates());
    }

    /// Java private `enableOrDisableCbEraseGold()`.
    fn enable_or_disable_cb_erase_gold(&self) {
        if self.manager.is_dual_axis() {
            if let (Some(rb_both), Some(rb_aonly), Some(rb_bonly)) =
                (&self.rb_both, &self.rb_aonly, &self.rb_bonly)
            {
                if rb_both.is_selected() {
                    self.cb_erase_gold.set_enabled(
                        self.is_gold_eraser_com_file_a.get()
                            && self.is_gold_eraser_com_file_b.get(),
                    );
                } else if rb_aonly.is_selected() {
                    self.cb_erase_gold
                        .set_enabled(self.is_gold_eraser_com_file_a.get());
                } else if rb_bonly.is_selected() {
                    self.cb_erase_gold
                        .set_enabled(self.is_gold_eraser_com_file_b.get());
                }
            }
        } else {
            self.cb_erase_gold
                .set_enabled(self.is_gold_eraser_com_file.get());
        }
    }

    /// Java public `getAltStackDisplay()`: `this`.
    pub fn get_alt_stack_display(&self) -> Rc<dyn AltStackDisplay> {
        self.self_ref
            .upgrade()
            .expect("AltStackPanel used after it was dropped")
    }
}

impl ContextMenu for AltStackPanel {
    /// Java public `popUpContextMenu(MouseEvent)`.  Right mouse button context
    /// menu.
    fn pop_up_context_menu(&self, mouse_event: &MouseEvent) {
        let man_pagelabel = [
            "AltTomoSetup".to_string(),
            "SwapTomoStacks".to_string(),
            "NewStack".to_string(),
            "Tilt".to_string(),
        ];
        let man_page = [
            "alttomosetup.html".to_string(),
            "swaptomostacks.html".to_string(),
            "newstack.html".to_string(),
            "tilt.html".to_string(),
        ];

        let log_file_label = ["AltTomoSetup".to_string()];
        let manager: &'static dyn BaseManager = self.manager;
        let log_file = [file_type::CLASS
            .alt_tomo_setup_log
            .get_file_name(Some(manager), Some(self.axis_id))
            .unwrap_or_else(|| "null".to_string())];

        let _context_popup = ContextPopup::new_component_mouse_event_string_string_string_array_string_array_string_array_string_array_base_manager_axis_id(
            &self.pnl_root,
            mouse_event,
            Some("AltTomo"),
            Some(context_popup::TOMO_GUIDE),
            &man_pagelabel,
            &man_page,
            Some(&log_file_label),
            Some(&log_file),
            manager,
            self.axis_id,
        );
    }
}

impl Run3dmodButtonContainer for AltStackPanel {
    /// Java public `action(String, Deferred3dmodButton, Run3dmodMenuOptions)`.
    fn action(
        &self,
        action_command: &str,
        _deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    ) {
        let manager: &'static dyn BaseManager = self.manager;
        if Some(action_command)
            == self
                .cb_process_even_odd_pairs
                .get_action_command()
                .as_deref()
        {
            if self.cb_process_even_odd_pairs.is_selected() {
                if !self.even_odd_pairs_files_found.get() {
                    self.is_rootname_even_file.set(
                        file_type::CLASS
                            .alt_stack_rootname_even_file
                            .exists(Some(manager), Some(AxisID::Only)),
                    );
                    self.is_rootname_odd_file.set(
                        file_type::CLASS
                            .alt_stack_rootname_odd_file
                            .exists(Some(manager), Some(AxisID::Only)),
                    );
                    if self.is_rootname_even_file.get() && self.is_rootname_odd_file.get() {
                        self.even_odd_pairs_files_found.set(true);
                        self.cb_process_even_odd_pairs.switch_text(false);
                        self.cb_process_even_odd_pairs.disable_warning();
                    } else {
                        self.cb_process_even_odd_pairs.enable_warning(true);
                    }
                } else {
                    self.cb_process_even_odd_pairs.disable_warning();
                }
            } else {
                self.cb_process_even_odd_pairs.disable_warning();
            }
        } else if self.rb_both.as_ref().is_some_and(|rb_both| {
            // `rbBoth != null && (actionCommand.equals(rbBoth.getActionCommand())
            // || ...rbAonly... || ...rbBonly...)`: the three buttons are created
            // together.
            Some(action_command) == rb_both.get_action_command().as_deref()
                || self.rb_aonly.as_ref().is_some_and(|rb_aonly| {
                    Some(action_command) == rb_aonly.get_action_command().as_deref()
                })
                || self.rb_bonly.as_ref().is_some_and(|rb_bonly| {
                    Some(action_command) == rb_bonly.get_action_command().as_deref()
                })
        }) {
            self.enable_or_disable_cb_erase_gold();
        } else if Some(action_command)
            == self.btn_run_alt_tomo_setup.get_action_command().as_deref()
        {
            self.manager.alt_tomo_setup(
                self.axis_id,
                self.dialog_type,
                Some(ProcessInterface::get_processing_method(self)),
                &*self.get_alt_stack_display(),
            );
        } else if Some(action_command)
            == self
                .btn_open_recon_in_3dmod1
                .get_action_command()
                .as_deref()
        {
            if self.manager.is_dual_axis() {
                let rootname = self.ltf_rootname_of_alt_stack.get_text_void();
                let property_user_dir = self.manager.get_property_user_dir();
                // Java passes the (never null in practice) file on; the
                // translation skips the open for a null file.
                if let Some(file) = file_type::CLASS
                    .alt_stack_tomogram
                    .get_file_with_property_user_dir(
                        Some(manager),
                        rootname.as_deref(),
                        Some(AxisType::DualAxis),
                        Some(AxisID::First),
                        property_user_dir.as_deref(),
                    )
                {
                    self.manager.open_alternative_tomogram(
                        &file,
                        AxisID::First,
                        AxisType::DualAxis,
                        run_3dmod_menu_options,
                        imod_manager::ALT_TOMO_SETUP_TOMOGRAM_KEY,
                        self.ltf_rootname_of_alt_stack.get_text_void().as_deref(),
                        false,
                    );
                }
            } else {
                self.open_alt_stack_single_axis_tomograms(run_3dmod_menu_options);
            }
        } else if Some(action_command)
            == self
                .btn_open_recon_in_3dmod2
                .get_action_command()
                .as_deref()
        {
            let rootname = self.ltf_rootname_of_alt_stack.get_text_void();
            let property_user_dir = self.manager.get_property_user_dir();
            if let Some(file) = file_type::CLASS
                .alt_stack_tomogram
                .get_file_with_property_user_dir(
                    Some(manager),
                    rootname.as_deref(),
                    Some(AxisType::DualAxis),
                    Some(AxisID::Second),
                    property_user_dir.as_deref(),
                )
            {
                self.manager.open_alternative_tomogram(
                    &file,
                    AxisID::Second,
                    AxisType::DualAxis,
                    run_3dmod_menu_options,
                    imod_manager::ALT_TOMO_SETUP_TOMOGRAM_KEY,
                    self.ltf_rootname_of_alt_stack.get_text_void().as_deref(),
                    false,
                );
            }
        } else if Some(action_command)
            == self
                .btn_restore_swapped_files
                .get_action_command()
                .as_deref()
        {
            self.manager.alt_tomo_setup_restore_swapped_files(
                self.axis_id,
                self.dialog_type,
                Some(ProcessInterface::get_processing_method(self)),
                &*self.get_alt_stack_display(),
            );
        }
        self.update_display();
    }
}

impl QueueTableListener for AltStackPanel {
    /// Java public `queueTableEventAction(QueueTableEvent)`; empty.
    fn queue_table_event_action(&self, _event: &QueueTableEvent) {}
}

impl ProcessInterface for AltStackPanel {
    /// Java public `updateGpu(boolean)`.
    fn update_gpu(&self, disable: bool) {
        self.update_display();
        ProcessInterface::update_gpu(&*self.cpu_gpu_panel, disable);
    }

    /// Java public `getProcessingMethod()`.
    fn get_processing_method(&self) -> ProcessingMethod {
        self.cpu_gpu_panel.get_processing_method_void()
    }

    /// Java public `getSecondaryProcessingMethod()`; returns null.
    fn get_secondary_processing_method(&self) -> Option<ProcessingMethod> {
        // TODO Auto-generated method stub
        None
    }

    /// Java public `lockProcessingMethod(boolean)`; empty.
    fn lock_processing_method(&self, _lock: bool) {
        // TODO Auto-generated method stub
    }

    /// Java public `setMethod(ProcessingMethod)`.  (The Java null test on the
    /// final `mediator` field always passes.)
    fn set_method(&self, processing_method: ProcessingMethod) {
        if let Some(this) = self.self_ref.upgrade() {
            let this: Rc<dyn ProcessInterface> = this;
            self.mediator
                .set_method_process_interface_processing_method(&this, processing_method);
        }
    }

    /// Java public `isUseGpu()`; returns false.
    fn is_use_gpu(&self) -> bool {
        // TODO Auto-generated method stub
        false
    }

    /// Java public `setUseQueueCheckBox(ButtonComponent)`; empty.
    fn set_use_queue_check_box(&self, _use_queue_checkbox: Option<Rc<dyn ButtonComponent>>) {}

    /// Java public `addQueueTableListener(QueueTableListener)`; empty.
    fn add_queue_table_listener(&self, _listener: Rc<dyn QueueTableListener>) {}

    /// Java public `removeQueueTableListener(QueueTableListener)`; empty.
    fn remove_queue_table_listener(&self, _listener: &Rc<dyn QueueTableListener>) {}
}

impl ProcessDisplay for AltStackPanel {}

impl AltStackDisplay for AltStackPanel {
    /// Java public `getParameters(TiltParam)`.
    fn get_parameters_tilt_param(&self, tilt_param: &mut TiltParam) -> bool {
        if !self.is_dialog_opened_first_time.get() {
            self.cpu_gpu_panel.get_parameters_tilt_param(tilt_param);
            return true;
        }
        false
    }

    /// Java public `getAxisID()`.  `None` (Java null) when Both is selected.
    fn get_axis_id(&self) -> Option<AxisID> {
        if self.rb_both.is_none() {
            return Some(AxisID::Only);
        }

        if self
            .rb_aonly
            .as_ref()
            .is_some_and(|rb_aonly| rb_aonly.is_selected())
        {
            return Some(AxisID::First);
        }
        if self
            .rb_bonly
            .as_ref()
            .is_some_and(|rb_bonly| rb_bonly.is_selected())
        {
            return Some(AxisID::Second);
        }

        // In case Both is Selected, return null
        // Document returning null

        None
    }

    /// Java public `getParameters(AltTomoSetupParam, boolean)`.
    fn get_parameters_alt_tomo_setup_param_boolean(
        &self,
        param: &mut AltTomoSetupParam,
        do_validation: bool,
    ) -> bool {
        // try { ... } catch (FieldValidationFailedException e)
        //   { e.printStackTrace(); return false; }
        let result = (|| -> Result<(), FieldValidationFailedException> {
            if self.cb_process_even_odd_pairs.is_visible()
                && self.cb_process_even_odd_pairs.is_enabled()
                && self.cb_process_even_odd_pairs.is_selected()
            {
                param.set_even_and_odd_pairs(
                    self.cb_process_even_odd_pairs.is_enabled()
                        && self.cb_process_even_odd_pairs.is_selected(),
                );
                param.reset_rootname_to_process();
            } else {
                param.set_rootname_to_process(
                    self.ltf_rootname_of_alt_stack
                        .get_text_boolean(do_validation)?
                        .as_deref(),
                );
                param.reset_even_and_odd_pairs();
            }
            let mut axis_to_process: Option<&str> = None;
            let _ = axis_to_process;
            // Leave out AxisToProcess if 'Both' option is selected
            if self.rb_both.is_some() {
                param.reset_axis_to_process();
                if let Some(rb_aonly) = &self.rb_aonly {
                    if rb_aonly.is_enabled() && rb_aonly.is_selected() {
                        axis_to_process = Some("A");
                        param.set_axis_to_process(axis_to_process);
                    }
                }
                if let Some(rb_bonly) = &self.rb_bonly {
                    if rb_bonly.is_enabled() && rb_bonly.is_selected() {
                        axis_to_process = Some("B");
                        param.set_axis_to_process(axis_to_process);
                    }
                }
            }
            let mut preprocess_for_extremes = 0;
            if self.cb_preprocess.is_enabled() && self.cb_preprocess.is_selected() {
                if self.cb_archive_orig_stack.is_selected() {
                    preprocess_for_extremes = 2;
                } else {
                    preprocess_for_extremes = 1;
                }
            }
            param.set_preprocess_for_extremes(preprocess_for_extremes);
            param.set_correct_ctf(
                self.cb_correct_ctf.is_enabled() && self.cb_correct_ctf.is_selected(),
            );
            param.set_erase_fiducials(
                self.cb_erase_gold.is_enabled() && self.cb_erase_gold.is_selected(),
            );
            param.set_filter_in_2d(
                self.cb_filter_in_2d.is_enabled() && self.cb_filter_in_2d.is_selected(),
            );
            param.set_trim_volume(
                self.cb_trim_volume.is_enabled() && self.cb_trim_volume.is_selected(),
            );
            param.set_clean_up_intermediates(
                self.cb_clean_up_intermediate_files.is_enabled()
                    && self.cb_clean_up_intermediate_files.is_selected(),
            );
            Ok(())
        })();
        if let Err(e) = result {
            eprintln!("{e:?}");
            return false;
        }
        true
    }
}
