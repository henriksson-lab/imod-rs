//! `IMOD/Etomo/src/etomo/ui/swing/PeetDialog.java`.
//!
//! The one dialog of the PEET interface: a beveled "PEET" panel holding a tabbed
//! pane with the Setup, Run and More Options tabs.  Setup carries the project
//! fields, the fix-paths panel, the volume table, the reference, missing wedge
//! compensation, masking, Y axis type and initial motive list panels; Run the
//! particles-per-CPU spinner, the iteration table, the spherical sampling panel,
//! the particle thresholds and the run buttons; More Options the remaining
//! `.prm` flags.
//!
//! Object model (ui.md): the dialog is an `Rc<PeetDialog>` living on the event
//! dispatch thread, every method takes `&self`.  The sub-panels, which the Java
//! constructor hands `this` to, are created right after the `Rc` exists and kept
//! in `OnceCell`s.  Swing layout (`BoxLayout`, rigid areas, glue, preferred
//! widths) is recorded as `// Swing layout:` comments.

use std::cell::{OnceCell, RefCell};
use std::path::{Path, PathBuf};
use std::rc::{Rc, Weak};

use super::abstract_parallel_dialog::AbstractParallelDialog;
use super::abstract_radio_button_model::AbstractRadioButtonModel;
use super::beveled_border::BeveledBorder;
use super::button_component::ButtonComponent;
use super::check_box::CheckBox;
use super::context_menu::ContextMenu;
use super::context_popup::ContextPopup;
use super::deferred_3dmod_button::Deferred3dmodButton;
use super::etched_border::EtchedBorder;
use super::etomo_panel::EtomoPanel;
use super::file_chooser::{self, FileChooser};
use super::file_container::FileContainer;
use super::file_text_field_interface::FileTextFieldInterface;
use super::fix_paths_panel::FixPathsPanel;
use super::generic_mouse_adapter::GenericMouseAdapter;
use super::iteration_parent::IterationParent;
use super::iteration_table::IterationTable;
use super::labeled_spinner::LabeledSpinner;
use super::labeled_text_field::LabeledTextField;
use super::masking_panel::MaskingPanel;
use super::masking_parent::MaskingParent;
use super::missing_wedge_compensation_panel::MissingWedgeCompensationPanel;
use super::missing_wedge_compensation_parent::MissingWedgeCompensationParent;
use super::process_interface::ProcessInterface;
use super::radio_button::{RadioButton, RadioButtonModel};
use super::radio_button_interface::EnumeratedTypeRef;
use super::reference_panel::ReferencePanel;
use super::reference_parent::ReferenceParent;
use super::run_3dmod_button_container::Run3dmodButtonContainer;
use super::run_3dmod_single_line_button::Run3dmodSingleLineButton;
use super::single_line_button::SingleLineButton;
use super::spaced_panel::{self, SpacedPanel};
use super::spherical_sampling_for_theta_and_psi_panel::SphericalSamplingForThetaAndPsiPanel;
use super::spherical_sampling_for_theta_and_psi_parent::SphericalSamplingForThetaAndPsiParent;
use super::spinner::Spinner;
use super::swing_component::SwingComponent;
use super::tabbed_pane::TabbedPane;
use super::ui_harness;
use super::ui_parameters::UIParameters;
use super::volume_table::VolumeTable;
use super::y_axis_type_panel::YAxisTypePanel;
use super::y_axis_type_parent::YAxisTypeParent;
use crate::imod::etomo::base_manager::{BaseManager, ManagerBrowsingDirectory};
use crate::imod::etomo::comscript::average_all_param::AverageAllParam;
use crate::imod::etomo::comscript::parallel_param::ParallelParam;
use crate::imod::etomo::jdk::{
    ActionEvent, ActionListener, ButtonGroup, ChangeEvent, ChangeListener, JComponent, MouseEvent,
    MouseListener,
};
use crate::imod::etomo::peet_manager::PeetManager;
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::processing_method_mediator::ProcessingMethodMediator;
use crate::imod::etomo::storage::autodoc::autodoc_factory;
use crate::imod::etomo::storage::autodoc::read_only_autodoc::ReadOnlyAutodoc;
use crate::imod::etomo::storage::autodoc::read_only_section_list::ReadOnlySectionList;
use crate::imod::etomo::storage::log_file::LogFileError;
use crate::imod::etomo::storage::matlab_param::{self, InitMotlCode, MatlabParam, YAxisType};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_peet_meta_data::ConstPeetMetaData;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::etomo_autodoc;
use crate::imod::etomo::r#type::peet_meta_data::PeetMetaData;
use crate::imod::etomo::r#type::processing_method::ProcessingMethod;
use crate::imod::etomo::ui::field::Field;
use crate::imod::etomo::ui::field_displayer::FieldDisplayer;
use crate::imod::etomo::ui::field_type::FieldType;
use crate::imod::etomo::ui::field_validation_failed_exception::FieldValidationFailedException;
use crate::imod::etomo::ui::queue_table_event::QueueTableEvent;
use crate::imod::etomo::ui::queue_table_listener::QueueTableListener;
use crate::imod::etomo::ui::shared_strings;
use crate::imod::etomo::ui::ui_component::UIComponent;
use crate::imod::etomo::util::utilities;

/// Java `public static final String FN_OUTPUT_LABEL`.
pub const FN_OUTPUT_LABEL: &str = "Root name for output";
/// Java `public static final String DIRECTORY_LABEL`.
pub const DIRECTORY_LABEL: &str = "Directory";
/// Java `public static final String RUN_LABEL`.
pub const RUN_LABEL: &str = "Run";
/// Java `public static final String AVERAGE_ALL_LABEL`.
pub const AVERAGE_ALL_LABEL: &str = "Remake Averages";
/// Java package-private `static final String SETUP_LOCATION_DESCR`.
pub const SETUP_LOCATION_DESCR: &str = "the Setup tab";
/// Java private `ALIGNED_BASE_NAME`.
const ALIGNED_BASE_NAME: &str = "aligned";

/// Java private `DIALOG_TYPE`.
const DIALOG_TYPE: DialogType = DialogType::Peet;
const LST_THRESHOLD_START_TITLE: &str = "Start";
const LST_THRESHOLD_INCREMENT_TITLE: &str = "Incr.";
const LST_THRESHOLD_END_TITLE: &str = "End";
const LST_THRESHOLD_ADDITIONAL_NUMBERS_TITLE: &str = "Additional numbers";
const LST_THRESHOLDS_LABEL: &str = "Number of Particles to Average";
const SETUP_TAB_LABEL: &str = "Setup";
const RUN_TAB_LABEL: &str = "Run";
const MORE_OPTIONS_TAB_LABEL: &str = "More Options";
const ALIGNED_BASE_NAME_LABEL: &str = "Save individual aligned particles";
const PARTICLE_PER_CPU_LABEL: &str = "Particles per CPU (core)";

/// Java private static final class `Tab`: represents the tabs belonging to this
/// dialog.  The index corresponds to the index of the tab.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tab {
    Setup,
    Run,
    MoreOptions,
}

impl Tab {
    /// Java `index`.
    fn index(self) -> i32 {
        match self {
            Tab::Setup => 0,
            Tab::Run => 1,
            Tab::MoreOptions => 2,
        }
    }
}

/// Java `public final class PeetDialog implements ContextMenu, AbstractParallelDialog,
/// Run3dmodButtonContainer, FileContainer, ReferenceParent,
/// MissingWedgeCompensationParent, IterationParent, MaskingParent, YAxisTypeParent,
/// SphericalSamplingForThetaAndPsiParent, ProcessInterface, UIComponent,
/// SwingComponent`.
pub struct PeetDialog {
    /// Java `this`.
    this: Weak<PeetDialog>,
    root_panel: Rc<EtomoPanel>,
    ltf_directory: Rc<LabeledTextField>,
    ltf_fn_output: Rc<LabeledTextField>,
    pnl_setup_body: Rc<SpacedPanel>,
    cb_aligned_base_name: Rc<CheckBox>,
    cb_flg_no_reference_refinement: Rc<CheckBox>,
    cb_ref_flag_all_tom: Rc<CheckBox>,
    ltf_lst_thresholds_start: Rc<LabeledTextField>,
    ltf_lst_thresholds_increment: Rc<LabeledTextField>,
    ltf_lst_thresholds_end: Rc<LabeledTextField>,
    ltf_lst_thresholds_additional: Rc<LabeledTextField>,
    cb_lst_flag_all_tom: Rc<CheckBox>,
    pnl_run_body: Rc<SpacedPanel>,
    btn_run: Rc<SingleLineButton>,
    ls_particle_per_cpu: Rc<LabeledSpinner>,
    iteration_table: OnceCell<Rc<IterationTable>>,
    bg_init_motl: Rc<ButtonGroup>,
    rb_init_motl_zero: Rc<RadioButton>,
    rb_init_align_particle_y_axes: Rc<RadioButton>,
    rb_init_motl_random_rotations: Rc<RadioButton>,
    rb_init_motl_random_axial_rotations: Rc<RadioButton>,
    rb_init_motl_files: Rc<RadioButton>,
    ls_debug_level: Rc<LabeledSpinner>,
    btn_avg_vol: Rc<Run3dmodSingleLineButton>,
    pnl_init_motl: Rc<EtomoPanel>,
    tab_pane: Rc<TabbedPane>,
    pnl_setup: Rc<SpacedPanel>,
    pnl_run: Rc<EtomoPanel>,
    btn_ref: Rc<Run3dmodSingleLineButton>,
    btn_average_all: Rc<SingleLineButton>,
    cbflg_align_averages: Rc<CheckBox>,
    cb_flg_abs_value: Rc<CheckBox>,
    ltf_select_class_id: Rc<LabeledTextField>,
    cb_flg_randomize: Rc<CheckBox>,
    pnl_more_options: Rc<SpacedPanel>,
    pnl_more_options_body: Rc<SpacedPanel>,
    ltf_exclude_list: Rc<LabeledTextField>,
    ltf_include_list: Rc<LabeledTextField>,
    cb_flg_elevation_compensation: Rc<CheckBox>,
    cb_flg_frm: Rc<CheckBox>,
    cb_flg_allow_masked_correlation: Rc<CheckBox>,
    cb_flg_filter_ref_only: Rc<CheckBox>,
    cb_flg_search_along_particle_axes: Rc<CheckBox>,
    cb_flg_fp_wedge_mask: Rc<CheckBox>,
    ltf_y_axis_symmetry: Rc<LabeledTextField>,
    cb_flg_use_extracted_particles: Rc<CheckBox>,
    cb_cn_symmetric_averaging: Rc<CheckBox>,
    l_cn_symmetric_averaging: Rc<JComponent>,
    sp_cn_symmetric_averaging: Rc<Spinner>,
    cb_flg_cn_masking: Rc<CheckBox>,
    ltf_user_commands: Rc<LabeledTextField>,
    setup_field_displayer: Rc<dyn FieldDisplayer>,
    more_options_field_displayer: Rc<dyn FieldDisplayer>,

    spherical_sampling_for_theta_and_psi_panel: OnceCell<Rc<SphericalSamplingForThetaAndPsiPanel>>,
    y_axis_type_panel: OnceCell<Rc<YAxisTypePanel>>,
    masking_panel: OnceCell<Rc<MaskingPanel>>,
    missing_wedge_compensation_panel: OnceCell<Rc<MissingWedgeCompensationPanel>>,
    reference_panel: OnceCell<Rc<ReferencePanel>>,
    volume_table: OnceCell<Rc<VolumeTable>>,
    manager: &'static PeetManager,
    axis_id: AxisID,
    fix_paths_panel: OnceCell<Rc<FixPathsPanel>>,
    mediator: Option<Rc<ProcessingMethodMediator>>,

    last_location: RefCell<Option<PathBuf>>,
    correct_path: RefCell<Option<String>>,
}

/// Java private static final class `PDFieldDisplayer implements FieldDisplayer`.
struct PDFieldDisplayer {
    peet_dialog: Weak<PeetDialog>,
    tab: Tab,
}

impl FieldDisplayer for PDFieldDisplayer {
    /// Java `display()`.
    fn display_void(&self) {
        if let Some(peet_dialog) = self.peet_dialog.upgrade() {
            peet_dialog.display(Some(self.tab));
        }
    }

    /// Java `display(UIComponent)`.
    fn display_ui_component(&self, _ui_component: Option<&dyn UIComponent>) {
        self.display_void();
    }
}

impl PeetDialog {
    /// Java private `PeetDialog(PeetManager, AxisID)`: the field initializers.
    fn new(manager: &'static PeetManager, axis_id: AxisID) -> Rc<PeetDialog> {
        Rc::new_cyclic(|this: &Weak<PeetDialog>| {
            let bg_init_motl = ButtonGroup::new();
            let container: Weak<dyn Run3dmodButtonContainer> = this.clone();
            let btn_avg_vol = Run3dmodSingleLineButton::get_3dmod_instance(
                Some("Open averages in 3dmod"),
                Some(container.clone()),
            );
            let btn_ref = Run3dmodSingleLineButton::get_3dmod_instance(
                Some("Open references in 3dmod"),
                Some(container),
            );
            PeetDialog {
                this: this.clone(),
                root_panel: EtomoPanel::new(),
                ltf_directory: LabeledTextField::new_field_type_string(
                    FieldType::String,
                    Some(&format!("{DIRECTORY_LABEL}: ")),
                ),
                ltf_fn_output: LabeledTextField::new_field_type_string(
                    FieldType::String,
                    Some(&format!("{FN_OUTPUT_LABEL}: ")),
                ),
                pnl_setup_body: SpacedPanel::get_instance_void(),
                cb_aligned_base_name: CheckBox::new_string(Some(ALIGNED_BASE_NAME_LABEL)),
                cb_flg_no_reference_refinement: CheckBox::new_string(Some(
                    shared_strings::FLG_NO_REFERENCE_REFINEMENT_LABEL,
                )),
                cb_ref_flag_all_tom: CheckBox::new_string(Some("For new references")),
                ltf_lst_thresholds_start: LabeledTextField::new_field_type_string(
                    FieldType::Integer,
                    Some(&format!("{LST_THRESHOLD_START_TITLE}: ")),
                ),
                ltf_lst_thresholds_increment: LabeledTextField::new_field_type_string(
                    FieldType::Integer,
                    Some(&format!("{LST_THRESHOLD_INCREMENT_TITLE}: ")),
                ),
                ltf_lst_thresholds_end: LabeledTextField::new_field_type_string(
                    FieldType::Integer,
                    Some(&format!("{LST_THRESHOLD_END_TITLE}: ")),
                ),
                ltf_lst_thresholds_additional: LabeledTextField::new_field_type_string(
                    FieldType::IntegerArray,
                    Some(&format!(" {LST_THRESHOLD_ADDITIONAL_NUMBERS_TITLE}: ")),
                ),
                cb_lst_flag_all_tom: CheckBox::new_string(Some("For average volumes")),
                pnl_run_body: SpacedPanel::get_instance_boolean(true),
                btn_run: SingleLineButton::new_string(Some(RUN_LABEL)),
                ls_particle_per_cpu: LabeledSpinner::get_instance_string_int_int_int_int_int(
                    Some(&format!("{PARTICLE_PER_CPU_LABEL}: ")),
                    matlab_param::PARTICLE_PER_CPU_DEFAULT,
                    matlab_param::PARTICLE_PER_CPU_MIN,
                    matlab_param::PARTICLE_PER_CPU_MAX,
                    1,
                    28,
                ),
                iteration_table: OnceCell::new(),
                rb_init_motl_zero: RadioButton::new_enumerated_type_button_group(
                    EnumeratedTypeRef::new(InitMotlCode::Zero),
                    Some(&bg_init_motl),
                ),
                rb_init_align_particle_y_axes: RadioButton::new_enumerated_type_button_group(
                    EnumeratedTypeRef::new(InitMotlCode::XAndZAxis),
                    Some(&bg_init_motl),
                ),
                rb_init_motl_random_rotations: RadioButton::new_enumerated_type_button_group(
                    EnumeratedTypeRef::new(InitMotlCode::RandomRotations),
                    Some(&bg_init_motl),
                ),
                rb_init_motl_random_axial_rotations: RadioButton::new_enumerated_type_button_group(
                    EnumeratedTypeRef::new(InitMotlCode::RandomAxialRotations),
                    Some(&bg_init_motl),
                ),
                rb_init_motl_files: RadioButton::new_string_button_group(
                    Some(shared_strings::CSV_FILES_LABEL),
                    Some(&bg_init_motl),
                ),
                bg_init_motl,
                ls_debug_level: LabeledSpinner::get_instance_string_int_int_int_int_int(
                    Some(&format!("{}: ", shared_strings::DEBUG_LEVEL_LABEL)),
                    matlab_param::DEBUG_LEVEL_DEFAULT,
                    matlab_param::DEBUG_LEVEL_MIN,
                    matlab_param::DEBUG_LEVEL_MAX,
                    1,
                    59,
                ),
                btn_avg_vol,
                pnl_init_motl: EtomoPanel::new(),
                tab_pane: TabbedPane::new(),
                pnl_setup: SpacedPanel::get_instance_void(),
                pnl_run: EtomoPanel::new(),
                btn_ref,
                btn_average_all: SingleLineButton::new_string(Some(AVERAGE_ALL_LABEL)),
                cbflg_align_averages: CheckBox::new_string(Some(
                    shared_strings::FLG_ALIGN_AVERAGES_LABEL,
                )),
                cb_flg_abs_value: CheckBox::new_string(Some(shared_strings::FLG_ABS_VALUE_LABEL)),
                ltf_select_class_id: LabeledTextField::new_field_type_string(
                    FieldType::MatlabIntegerArray,
                    Some(shared_strings::SELECT_CLASS_ID_LABEL),
                ),
                cb_flg_randomize: CheckBox::new_string(Some(shared_strings::FLG_RANDOMIZE_LABEL)),
                pnl_more_options: SpacedPanel::get_instance_void(),
                pnl_more_options_body: SpacedPanel::get_instance_void(),
                ltf_exclude_list: LabeledTextField::new_field_type_string(
                    FieldType::MatlabIntegerArray,
                    Some(shared_strings::EXCLUDE_LIST_LABEL),
                ),
                ltf_include_list: LabeledTextField::new_field_type_string(
                    FieldType::MatlabIntegerArray,
                    Some(shared_strings::INCLUDE_LIST_LABEL),
                ),
                cb_flg_elevation_compensation: CheckBox::new_string(Some(
                    shared_strings::FLG_ELEVATION_COMPENSATION_LABEL,
                )),
                cb_flg_frm: CheckBox::new_string(Some(shared_strings::FLG_FRM_LABEL)),
                cb_flg_allow_masked_correlation: CheckBox::new_string(Some(
                    shared_strings::FLG_ALLOW_MASKED_CORRELATION_LABEL,
                )),
                cb_flg_filter_ref_only: CheckBox::new_string(Some(
                    shared_strings::FLG_FILTER_REF_ONLY_LABEL,
                )),
                cb_flg_search_along_particle_axes: CheckBox::new_string(Some(
                    shared_strings::FLG_SEARCH_ALONG_PARTICLE_AXES_LABEL,
                )),
                cb_flg_fp_wedge_mask: CheckBox::new_string(Some(
                    shared_strings::FLG_FP_WEDGE_MASK_LABEL,
                )),
                ltf_y_axis_symmetry: LabeledTextField::new_field_type_string(
                    FieldType::IntegerArray,
                    Some(shared_strings::YAXIS_SYMMETRY_LABEL),
                ),
                cb_flg_use_extracted_particles: CheckBox::new_string(Some(
                    shared_strings::FLG_USE_EXTRACTED_PARTICLES_LABEL,
                )),
                cb_cn_symmetric_averaging: CheckBox::new_string(Some(
                    shared_strings::CN_SYMMETRIC_AVERAGING_LABEL,
                )),
                l_cn_symmetric_averaging: JComponent::new_label("="),
                sp_cn_symmetric_averaging: Spinner::get_instance_string_int_int_int_int(
                    Some(shared_strings::CN_SYMMETRIC_AVERAGING_LABEL),
                    matlab_param::CN_SYMMETRIC_AVERAGING_DEFAULT,
                    matlab_param::CN_SYMMETRIC_AVERAGING_MIN,
                    matlab_param::CN_SYMMETRIC_AVERAGING_MAX,
                    matlab_param::CN_SYMMETRIC_AVERAGING_STEP,
                ),
                cb_flg_cn_masking: CheckBox::new_string(Some(shared_strings::FLG_CN_MASKING_LABEL)),
                ltf_user_commands: LabeledTextField::new_field_type_string(
                    FieldType::StringArray,
                    Some(shared_strings::USER_COMMANDS_LABEL),
                ),
                setup_field_displayer: Rc::new(PDFieldDisplayer {
                    peet_dialog: this.clone(),
                    tab: Tab::Setup,
                }),
                more_options_field_displayer: Rc::new(PDFieldDisplayer {
                    peet_dialog: this.clone(),
                    tab: Tab::MoreOptions,
                }),
                spherical_sampling_for_theta_and_psi_panel: OnceCell::new(),
                y_axis_type_panel: OnceCell::new(),
                masking_panel: OnceCell::new(),
                missing_wedge_compensation_panel: OnceCell::new(),
                reference_panel: OnceCell::new(),
                volume_table: OnceCell::new(),
                manager,
                axis_id,
                fix_paths_panel: OnceCell::new(),
                mediator: None,
                last_location: RefCell::new(None),
                correct_path: RefCell::new(None),
            }
        })
    }

    fn this_rc(&self) -> Rc<PeetDialog> {
        self.this.upgrade().expect("the PEET dialog is alive")
    }
    fn iteration_table(&self) -> &Rc<IterationTable> {
        self.iteration_table.get().expect("iterationTable")
    }
    fn spherical_sampling_for_theta_and_psi_panel(
        &self,
    ) -> &Rc<SphericalSamplingForThetaAndPsiPanel> {
        self.spherical_sampling_for_theta_and_psi_panel
            .get()
            .expect("sphericalSamplingForThetaAndPsiPanel")
    }
    fn y_axis_type_panel(&self) -> &Rc<YAxisTypePanel> {
        self.y_axis_type_panel.get().expect("yAxisTypePanel")
    }
    fn masking_panel(&self) -> &Rc<MaskingPanel> {
        self.masking_panel.get().expect("maskingPanel")
    }
    fn missing_wedge_compensation_panel(&self) -> &Rc<MissingWedgeCompensationPanel> {
        self.missing_wedge_compensation_panel
            .get()
            .expect("missingWedgeCompensationPanel")
    }
    fn reference_panel(&self) -> &Rc<ReferencePanel> {
        self.reference_panel.get().expect("referencePanel")
    }
    fn volume_table(&self) -> &Rc<VolumeTable> {
        self.volume_table.get().expect("volumeTable")
    }
    fn fix_paths_panel(&self) -> &Rc<FixPathsPanel> {
        self.fix_paths_panel.get().expect("fixPathsPanel")
    }

    /// Java `mediator`: set in the constructor body (a field the Rust struct
    /// receives after `Rc::new_cyclic`).
    fn mediator(&self) -> Option<Rc<ProcessingMethodMediator>> {
        self.mediator.clone().or_else(|| {
            self.manager
                .get_processing_method_mediator(Some(self.axis_id))
        })
    }

    /// The Java constructor body after the field initializers.
    fn construct(&self) {
        eprintln!(
            "{}\nDialog: {}",
            utilities::get_date_time_stamp(),
            DialogType::Peet
        );
        let this = self.this_rc();
        let manager: &'static dyn BaseManager = self.manager;
        let _ = self.reference_panel.set(ReferencePanel::get_instance(
            Rc::downgrade(&this) as Weak<dyn ReferenceParent>,
            manager,
        ));
        let _ =
            self.missing_wedge_compensation_panel
                .set(MissingWedgeCompensationPanel::get_instance(
                    Rc::downgrade(&this) as Weak<dyn MissingWedgeCompensationParent>,
                    Some(self.setup_field_displayer.clone()),
                ));
        let _ = self.masking_panel.set(MaskingPanel::get_instance(
            manager,
            Rc::downgrade(&this) as Weak<dyn MaskingParent>,
            Some(self.setup_field_displayer.clone()),
        ));
        let _ = self.y_axis_type_panel.set(YAxisTypePanel::get_instance(
            manager,
            Rc::downgrade(&this) as Weak<dyn YAxisTypeParent>,
        ));
        let _ = self.fix_paths_panel.set(FixPathsPanel::get_instance(
            Rc::downgrade(&this) as Weak<dyn FileContainer>,
            manager,
            self.axis_id,
            Some(DIALOG_TYPE),
        ));
        let _ = self.spherical_sampling_for_theta_and_psi_panel.set(
            SphericalSamplingForThetaAndPsiPanel::get_instance(
                manager,
                Rc::downgrade(&this) as Weak<dyn SphericalSamplingForThetaAndPsiParent>,
            ),
        );
        let _ = self
            .volume_table
            .set(VolumeTable::get_instance(self.manager, &this));
        let _ = self.iteration_table.set(IterationTable::get_instance(
            manager,
            Rc::downgrade(&this) as Weak<dyn IterationParent>,
        ));
        // panels
        // Swing layout: rootPanel BoxLayout Y_AXIS.
        self.root_panel
            .set_border(&BeveledBorder::new(Some("PEET")).get_border());
        self.root_panel
            .get_component()
            .add(&self.tab_pane.get_component());
        self.create_setup_panel();
        self.create_run_panel();
        self.create_more_options_panel();
        // `JTabbedPane.add(String, Component)` is `addTab`, so TabbedPane's
        // override names the pane from the first tab.
        self.tab_pane
            .add_tab_string_component(SETUP_TAB_LABEL, &self.pnl_setup.get_container());
        self.tab_pane
            .add_tab_string_component(RUN_TAB_LABEL, &self.pnl_run.get_component());
        self.tab_pane.add_tab_string_component(
            MORE_OPTIONS_TAB_LABEL,
            &self.pnl_more_options.get_container(),
        );
        let context_menu: Weak<dyn ContextMenu> = self.this.clone();
        let mouse_adapter: Rc<dyn MouseListener> = GenericMouseAdapter::new(context_menu);
        self.tab_pane
            .get_component()
            .add_mouse_listener(mouse_adapter);
        self.change_tab_void();
        self.set_defaults();
        self.update_display(true);
        self.set_tooltip_text();
        if let Some(mediator) = self.mediator() {
            let origin: Rc<dyn ProcessInterface> = this.clone();
            mediator.register_process_interface(origin.clone());
            mediator.set_method_process_interface_processing_method(
                &origin,
                self.get_processing_method(),
            );
        }
    }

    /// Java `getFocusComponent()`.
    pub fn get_focus_component(&self) -> Rc<JComponent> {
        self.pnl_setup.get_container()
    }

    /// Java `getSetupJComponent()`.
    pub fn get_setup_j_component(&self) -> Rc<JComponent> {
        self.pnl_setup.get_j_panel()
    }

    /// Java static `getInstance(PeetManager, AxisID)`.
    pub fn get_instance(manager: &'static PeetManager, axis_id: AxisID) -> Rc<PeetDialog> {
        let instance = PeetDialog::new(manager, axis_id);
        instance.construct();
        instance.add_listeners();
        instance
    }

    /// Java `updateMode(boolean)`.  Toggles between a setup-like mode where the
    /// location and root name being chosen, and a regular mode.
    pub fn update_mode(&self, param_file_set: bool) {
        self.ltf_directory.set_editable(!param_file_set);
        self.ltf_fn_output.set_editable(!param_file_set);
        self.btn_run.set_enabled(param_file_set);
    }

    /// Java `getComponent()`.
    pub fn get_component(&self) -> Rc<JComponent> {
        self.root_panel.get_component()
    }

    /// Java `pack()`.
    pub fn pack(&self) {
        self.volume_table().pack();
    }

    /// Java `convertCopiedPaths(String)`.
    pub fn convert_copied_paths(&self, orig_dataset_dir: &str) {
        self.volume_table().convert_copied_paths(orig_dataset_dir);
        self.reference_panel()
            .convert_copied_paths(orig_dataset_dir);
        self.masking_panel().convert_copied_paths(orig_dataset_dir);
    }

    /// Java `checkIncorrectPaths()`.
    pub fn check_incorrect_paths(&self) {
        let mut incorrect_paths = false;
        if self.volume_table().is_incorrect_paths() {
            incorrect_paths = true;
        } else if self.reference_panel().is_incorrect_paths() {
            incorrect_paths = true;
        } else if self.masking_panel().is_incorrect_paths() {
            incorrect_paths = true;
        }
        self.fix_paths_panel().set_incorrect_paths(incorrect_paths);
    }

    /// Java package-private `getFileChooserInstance()`.
    pub fn get_file_chooser_instance(&self) -> Rc<FileChooser> {
        let last_location = self.last_location.borrow().clone();
        FileChooser::new_base_manager_file_browsing_directory(
            Some(self.manager),
            last_location.as_deref(),
            Some(&ManagerBrowsingDirectory(self.manager)),
        )
    }

    /// Java package-private `setLastLocation(File)`.
    pub fn set_last_location(&self, input: Option<PathBuf>) {
        *self.last_location.borrow_mut() = input;
    }

    /// Java package-private `isCorrectPathNull()`.
    pub fn is_correct_path_null(&self) -> bool {
        self.correct_path.borrow().is_none()
    }

    /// Java package-private `setCorrectPath(String)`.
    pub fn set_correct_path(&self, correct_path: Option<String>) {
        *self.last_location.borrow_mut() = Some(PathBuf::from(
            correct_path.clone().unwrap_or_else(|| "null".to_owned()),
        ));
        *self.correct_path.borrow_mut() = correct_path;
    }

    /// Java package-private `getCorrectPath()`.
    pub fn get_correct_path(&self) -> Option<String> {
        self.correct_path.borrow().clone()
    }

    /// Java `getParameters(PeetMetaData)`.
    pub fn get_parameters_meta_data(&self, meta_data: &PeetMetaData) {
        self.volume_table().get_parameters_meta_data(meta_data);
        self.reference_panel().get_parameters_meta_data(meta_data);
        self.missing_wedge_compensation_panel()
            .get_parameters_meta_data(meta_data);
        self.masking_panel().get_parameters_meta_data(meta_data);
        self.iteration_table().get_parameters_meta_data(meta_data);
        meta_data.set_flg_align_averages(self.cbflg_align_averages.is_selected());
        meta_data.set_cn_symmetric_averaging(Some(self.sp_cn_symmetric_averaging.get_value()));
    }

    /// Java `getParameters(AverageAllParam)`.
    pub fn get_parameters_average_all_param(&self, param: &mut AverageAllParam) {
        param.set_iteration_number(self.iteration_table().size());
    }

    /// Java `setParameters(ConstPeetMetaData)`.  Set parameters from metaData and then
    /// overwrite them with parameters from MatlabParamFile.  This allows inactive data
    /// to appear on the screen but allows MatlabParamFile's active data to override
    /// active metaData.
    pub fn set_parameters_meta_data(&self, meta_data: &'static PeetMetaData) {
        self.ltf_fn_output
            .set_text_string(ConstPeetMetaData::get_name(meta_data).as_deref());
        self.volume_table().set_parameters_meta_data(meta_data);
        self.reference_panel().set_parameters_meta_data(meta_data);
        self.missing_wedge_compensation_panel()
            .set_parameters_meta_data(meta_data);
        self.masking_panel().set_parameters_meta_data(meta_data);
        self.iteration_table().set_parameters_meta_data(meta_data);
        self.cbflg_align_averages
            .set_selected_boolean(meta_data.is_flg_align_averages());
        if meta_data.is_cn_symmetric_averaging() {
            self.sp_cn_symmetric_averaging
                .set_value_const_etomo_number(&meta_data.get_cn_symmetric_averaging());
        }
    }

    /// Java `setParameters(MatlabParam, File)`.  Load data from MatlabParamFile.  Load
    /// only active data after the meta data has been loaded.  Do not load fnOutput.
    pub fn set_parameters_matlab_param(
        &self,
        matlab_param: &mut MatlabParam,
        import_dir: Option<&Path>,
    ) {
        self.iteration_table()
            .set_parameters_matlab_param(matlab_param);
        self.ltf_user_commands
            .set_text_string(Some(&matlab_param.get_user_commands()));
        self.missing_wedge_compensation_panel()
            .set_parameters_matlab_param(matlab_param);
        let init_motl_code = matlab_param.get_init_motl_code();
        match init_motl_code {
            None => self.rb_init_motl_files.set_selected_boolean(true),
            Some(InitMotlCode::Zero) => self.rb_init_motl_zero.set_selected_boolean(true),
            Some(InitMotlCode::XAndZAxis) => self
                .rb_init_align_particle_y_axes
                .set_selected_boolean(true),
            Some(InitMotlCode::RandomRotations) => self
                .rb_init_motl_random_rotations
                .set_selected_boolean(true),
            Some(InitMotlCode::RandomAxialRotations) => self
                .rb_init_motl_random_axial_rotations
                .set_selected_boolean(true),
            Some(_) => {}
        }
        if !matlab_param.is_aligned_base_name_empty() {
            self.cb_aligned_base_name.set_selected_boolean(true);
            let aligned_base_name = matlab_param.get_aligned_base_name();
            if aligned_base_name.as_deref() != Some(ALIGNED_BASE_NAME) {
                ui_harness::with(|harness| {
                    harness.open_problem_value_message_dialog(
                        Some(self.manager),
                        Some(self as &dyn UIComponent),
                        "Invalid",
                        Some(matlab_param::ALIGNED_BASE_NAME_KEY),
                        None,
                        Some(ALIGNED_BASE_NAME_LABEL),
                        aligned_base_name.as_deref(),
                        Some(ALIGNED_BASE_NAME),
                        None,
                    )
                });
            }
        } else {
            self.cb_aligned_base_name.set_selected_boolean(false);
        }
        self.cb_flg_no_reference_refinement
            .set_selected_boolean(matlab_param.is_flg_no_reference_refinement());
        self.cb_flg_randomize
            .set_selected_boolean(matlab_param.is_flg_randomize());
        self.ls_debug_level
            .set_value_const_etomo_number(&matlab_param.get_debug_level());
        self.ltf_lst_thresholds_start
            .set_text_string(matlab_param.get_lst_thresholds_start().as_deref());
        self.ltf_lst_thresholds_increment
            .set_text_string(matlab_param.get_lst_thresholds_increment().as_deref());
        self.ltf_lst_thresholds_end
            .set_text_string(matlab_param.get_lst_thresholds_end().as_deref());
        self.ltf_lst_thresholds_additional
            .set_text_string(Some(&matlab_param.get_lst_thresholds_additional()));
        self.cb_ref_flag_all_tom
            .set_selected_boolean(!matlab_param.is_ref_flag_all_tom());
        self.cb_lst_flag_all_tom
            .set_selected_boolean(!matlab_param.is_lst_flag_all_tom());
        // particle per cpu
        let number = matlab_param.get_particle_per_cpu();
        if self.ls_particle_per_cpu.is_in_range(Some(&number)) {
            self.ls_particle_per_cpu
                .set_value_const_etomo_number(&number);
        } else {
            ui_harness::with(|harness| {
                harness.open_problem_value_message_dialog(
                    Some(self.manager),
                    Some(self as &dyn UIComponent),
                    "Unknown",
                    Some(matlab_param::PARTICLE_PER_CPU_KEY),
                    None,
                    Some(PARTICLE_PER_CPU_LABEL),
                    Some(&number.to_string()),
                    Some(&matlab_param::PARTICLE_PER_CPU_DEFAULT.to_string()),
                    None,
                )
            });
            self.ls_particle_per_cpu
                .set_value_int(matlab_param::PARTICLE_PER_CPU_DEFAULT);
        }
        self.y_axis_type_panel().set_parameters(matlab_param);
        self.volume_table().set_parameters_matlab_param(
            matlab_param,
            self.rb_init_motl_files.is_selected(),
            self.missing_wedge_compensation_panel()
                .is_tilt_range_required(),
            self.missing_wedge_compensation_panel()
                .is_tilt_range_multi_axes(),
            import_dir,
        );
        self.spherical_sampling_for_theta_and_psi_panel()
            .set_parameters(matlab_param);
        self.masking_panel()
            .set_parameters_matlab_param(matlab_param);
        if !matlab_param.is_empty_flg_align_averages() {
            self.cbflg_align_averages
                .set_selected_boolean(matlab_param.is_flg_align_averages());
        }
        self.cb_flg_abs_value
            .set_selected_boolean(matlab_param.is_flg_abs_value());
        self.ltf_select_class_id
            .set_text_string(matlab_param.get_select_class_id().as_deref());
        self.ltf_exclude_list
            .set_text_string(matlab_param.get_exclude_list().as_deref());
        self.ltf_include_list
            .set_text_string(matlab_param.get_include_list().as_deref());
        self.cb_flg_elevation_compensation
            .set_selected_boolean(matlab_param.is_flg_elevation_compensation());
        self.cb_flg_frm
            .set_selected_boolean(matlab_param.is_flg_frm());
        self.cb_flg_allow_masked_correlation
            .set_selected_boolean(matlab_param.is_flg_allow_masked_correlation());
        self.cb_flg_filter_ref_only
            .set_selected_boolean(matlab_param.is_flg_filter_ref_only());
        self.cb_flg_search_along_particle_axes
            .set_selected_boolean(matlab_param.is_flg_search_along_particle_axes());
        self.cb_flg_fp_wedge_mask
            .set_selected_boolean(matlab_param.is_flg_fp_wedge_mask());
        self.ltf_y_axis_symmetry
            .set_text_string(matlab_param.get_y_axis_symmetry().as_deref());
        self.cb_flg_use_extracted_particles
            .set_selected_boolean(matlab_param.is_flg_use_extracted_particles());
        let is_cn_symmetric_averaging = matlab_param.is_cn_symmetric_averaging();
        self.cb_cn_symmetric_averaging
            .set_selected_boolean(is_cn_symmetric_averaging);
        if is_cn_symmetric_averaging {
            self.sp_cn_symmetric_averaging
                .set_value_string(matlab_param.get_cn_symmetric_averaging().as_deref());
        }
        self.cb_flg_cn_masking
            .set_selected_boolean(matlab_param.is_flg_cn_masking());
        self.update_display(true);
        self.reference_panel()
            .set_parameters_matlab_param(matlab_param);
        self.update_display(true);
    }

    /// Java `checkLowCutoffBackwardsCompatibility(MatlabParam)`.
    pub fn check_low_cutoff_backwards_compatibility(&self, matlab_param: &mut MatlabParam) {
        self.iteration_table()
            .check_low_cutoff_backwards_compatibility(matlab_param);
    }

    /// Java `getParameters(MatlabParam, boolean, boolean)`.
    pub fn get_parameters_matlab_param(
        &self,
        matlab_param: &mut MatlabParam,
        for_run: bool,
        do_validation: bool,
    ) -> bool {
        if !matlab_param.validate(for_run) {
            return false;
        }
        matlab_param.clear();
        self.volume_table()
            .get_parameters_matlab_param(matlab_param);
        self.iteration_table()
            .get_parameters_matlab_param(matlab_param);
        let result = (|| -> Result<bool, FieldValidationFailedException> {
            matlab_param.set_user_commands(
                self.ltf_user_commands
                    .get_text_boolean(do_validation)?
                    .as_deref(),
            );
            matlab_param.set_fn_output(
                self.ltf_fn_output
                    .get_text_boolean(do_validation)?
                    .as_deref(),
            );
            if !self
                .reference_panel()
                .get_parameters_matlab_param(matlab_param, for_run)
            {
                return Ok(false);
            }
            self.missing_wedge_compensation_panel()
                .get_parameters_matlab_param(matlab_param, do_validation);
            matlab_param.set_init_motl_code(self.selected_init_motl_code());
            if self.cb_aligned_base_name.is_selected() {
                matlab_param.set_aligned_base_name(Some(ALIGNED_BASE_NAME));
            } else {
                matlab_param.reset_aligned_base_name();
            }
            matlab_param
                .set_flg_no_reference_refinement(self.cb_flg_no_reference_refinement.is_selected());
            matlab_param.set_flg_randomize(self.cb_flg_randomize.is_selected());
            matlab_param.set_debug_level(self.ls_debug_level.get_value());
            matlab_param.set_lst_thresholds_start(
                self.ltf_lst_thresholds_start
                    .get_text_boolean(do_validation)?
                    .as_deref(),
            );
            matlab_param.set_lst_thresholds_increment(
                self.ltf_lst_thresholds_increment
                    .get_text_boolean(do_validation)?
                    .as_deref(),
            );
            matlab_param.set_lst_thresholds_end(
                self.ltf_lst_thresholds_end
                    .get_text_boolean(do_validation)?
                    .as_deref(),
            );
            matlab_param.set_lst_thresholds_additional(
                self.ltf_lst_thresholds_additional
                    .get_text_boolean(do_validation)?
                    .as_deref(),
            );
            matlab_param.set_ref_flag_all_tom(!self.cb_ref_flag_all_tom.is_selected());
            matlab_param.set_lst_flag_all_tom(!self.cb_lst_flag_all_tom.is_selected());
            matlab_param.set_particle_per_cpu(self.ls_particle_per_cpu.get_value());
            self.y_axis_type_panel().get_parameters(matlab_param);
            if !self
                .spherical_sampling_for_theta_and_psi_panel()
                .get_parameters(matlab_param, do_validation)
            {
                return Ok(false);
            }
            if !self
                .masking_panel()
                .get_parameters_matlab_param(matlab_param, do_validation)
            {
                return Ok(false);
            }
            if self.cbflg_align_averages.is_enabled() {
                matlab_param.set_flg_align_averages(self.cbflg_align_averages.is_selected());
            } else {
                matlab_param.reset_flg_align_averages();
            }
            matlab_param.set_flg_abs_value(self.cb_flg_abs_value.is_selected());
            matlab_param.set_select_class_id(
                self.ltf_select_class_id
                    .get_text_boolean(do_validation)?
                    .as_deref(),
            );
            matlab_param.set_exclude_list(
                self.ltf_exclude_list
                    .get_text_boolean(do_validation)?
                    .as_deref(),
            );
            matlab_param.set_include_list(
                self.ltf_include_list
                    .get_text_boolean(do_validation)?
                    .as_deref(),
            );
            matlab_param
                .set_flg_elevation_compensation(self.cb_flg_elevation_compensation.is_selected());
            matlab_param.set_flg_frm(self.cb_flg_frm.is_selected());
            matlab_param.set_flg_allow_masked_correlation(
                self.cb_flg_allow_masked_correlation.is_selected(),
            );
            matlab_param.set_flg_filter_ref_only(self.cb_flg_filter_ref_only.is_selected());
            matlab_param.set_flg_search_along_particle_axes(
                self.cb_flg_search_along_particle_axes.is_selected(),
            );
            matlab_param.set_flg_fp_wedge_mask(self.cb_flg_fp_wedge_mask.is_selected());
            matlab_param.set_y_axis_symmetry(
                self.ltf_y_axis_symmetry
                    .get_text_boolean(do_validation)?
                    .as_deref(),
            );
            matlab_param
                .set_flg_use_extracted_particles(self.cb_flg_use_extracted_particles.is_selected());
            if self.cb_cn_symmetric_averaging.is_selected() {
                matlab_param.set_cn_symmetric_averaging(self.sp_cn_symmetric_averaging.get_value());
            } else {
                matlab_param.reset_cn_symmetric_averaging();
            }
            matlab_param.set_flg_cn_masking(self.cb_flg_cn_masking.is_selected());
            if do_validation
                && (!self.ltf_y_axis_symmetry.handle_validation(
                    matlab_param.validate_y_axis_symmetry().as_deref(),
                    None,
                    None,
                ) || !self.ltf_user_commands.handle_validation(
                    matlab_param.validate_user_commands().as_deref(),
                    None,
                    None,
                ))
            {
                return Ok(false);
            }
            Ok(true)
        })();
        match result {
            Ok(result) => result,
            Err(e) => {
                eprintln!("{e}");
                false
            }
        }
    }

    /// `((RadioButton.RadioButtonModel) bgInitMotl.getSelection()).getEnumeratedType()`.
    /// (The "User supplied csv files" button has no enumerated type: null.)
    fn selected_init_motl_code(&self) -> Option<InitMotlCode> {
        self.bg_init_motl
            .get_selection()
            .and_then(|button| button.get_model())
            .and_then(|model| {
                model
                    .as_any()
                    .downcast_ref::<RadioButtonModel>()
                    .and_then(|model| model.get_enumerated_type())
            })
            .and_then(|enumerated_type| enumerated_type.downcast_ref::<InitMotlCode>().copied())
    }

    /// Java `getFnOutput(boolean) throws FieldValidationFailedException`.
    pub fn get_fn_output(
        &self,
        do_validation: bool,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        self.ltf_fn_output.get_text_boolean(do_validation)
    }

    /// Java `setDirectory(String)`.
    pub fn set_directory(&self, directory: Option<&str>) {
        self.ltf_directory.set_text_string(directory);
    }

    /// Java `setFnOutput(String)`.
    pub fn set_fn_output(&self, output: Option<&str>) {
        self.ltf_fn_output.set_text_string(output);
    }

    /// Java package-private `msgVolumeTableSizeChanged(boolean)`.
    pub fn msg_volume_table_size_changed(&self, init: bool) {
        self.update_display(init);
    }

    /// Java package-private `setUsingInitMotlFile()`.
    pub fn set_using_init_motl_file(&self) {
        self.rb_init_motl_files.set_selected_boolean(true);
    }

    /// Java private `setTooltipText()`.
    ///
    /// Fixed in translation (PeetDialog.java:654): with the peetprm autodoc missing
    /// (null), Java dereferences it and the dialog fails to construct; the autodoc
    /// tooltips are left unset instead.  (BUGS.md)
    fn set_tooltip_text(&self) {
        let autodoc = match unsafe {
            autodoc_factory::get_instance(
                Some(self.manager),
                Some(autodoc_factory::PEET_PRM),
                self.axis_id,
                false,
            )
        } {
            Ok(autodoc) => autodoc,
            Err(LogFileError::Lock(_)) => std::ptr::null_mut(),
            Err(e) => {
                eprintln!("{e}");
                std::ptr::null_mut()
            }
        };
        let autodoc_ref = unsafe { autodoc.as_ref() };
        let read_only = autodoc_ref.map(|autodoc| autodoc as &dyn ReadOnlyAutodoc);
        let tooltip =
            |key: &str| etomo_autodoc::get_tooltip_autodoc_add_source(read_only, Some(key), false);
        if let Some(autodoc) = autodoc_ref {
            let autodoc_name = ReadOnlyAutodoc::get_autodoc_name(autodoc);
            self.pnl_init_motl
                .get_component()
                .set_tool_tip_text(tooltip(InitMotlCode::KEY).as_deref());
            let section = unsafe {
                ReadOnlySectionList::get_section(
                    autodoc,
                    Some(etomo_autodoc::FIELD_SECTION_NAME),
                    Some(InitMotlCode::KEY),
                )
            };
            // Java passes a null section on, which formats as no tooltip.
            if let Some(section) = unsafe { section.as_ref() } {
                self.rb_init_motl_zero
                    .set_tool_tip_text_string_read_only_section(Some(&autodoc_name), section);
                self.rb_init_align_particle_y_axes
                    .set_tool_tip_text_string_read_only_section(Some(&autodoc_name), section);
                self.rb_init_motl_random_rotations
                    .set_tool_tip_text_string_read_only_section(Some(&autodoc_name), section);
                self.rb_init_motl_random_axial_rotations
                    .set_tool_tip_text_string_read_only_section(Some(&autodoc_name), section);
            }
            let section = unsafe {
                ReadOnlySectionList::get_section(
                    autodoc,
                    Some(etomo_autodoc::FIELD_SECTION_NAME),
                    Some(matlab_param::REF_FLAG_ALL_TOM_KEY),
                )
            };
            if let Some(section) = unsafe { section.as_ref() } {
                self.cb_ref_flag_all_tom
                    .set_tool_tip_text_string_read_only_section_string(
                        Some(&autodoc_name),
                        section,
                        Some("0"),
                    );
            }
            let section = unsafe {
                ReadOnlySectionList::get_section(
                    autodoc,
                    Some(etomo_autodoc::FIELD_SECTION_NAME),
                    Some(matlab_param::LST_FLAG_ALL_TOM_KEY),
                )
            };
            if let Some(section) = unsafe { section.as_ref() } {
                self.cb_lst_flag_all_tom
                    .set_tool_tip_text_string_read_only_section_string(
                        Some(&autodoc_name),
                        section,
                        Some("0"),
                    );
            }
            self.cbflg_align_averages
                .set_tool_tip_text_string(tooltip(matlab_param::FLG_ALIGN_AVERAGES_KEY).as_deref());
            let section = unsafe {
                ReadOnlySectionList::get_section(
                    autodoc,
                    Some(etomo_autodoc::FIELD_SECTION_NAME),
                    Some(matlab_param::FLG_ABS_VALUE_KEY),
                )
            };
            if let Some(section) = unsafe { section.as_ref() } {
                self.cb_flg_abs_value
                    .set_tool_tip_text_string_read_only_section_string(
                        Some(&autodoc_name),
                        section,
                        Some("1"),
                    );
            }
            self.cb_flg_no_reference_refinement
                .set_tool_tip_text_string(
                    tooltip(matlab_param::FLG_NO_REFERENCE_REFINEMENT_KEY).as_deref(),
                );
            self.cb_flg_randomize
                .set_tool_tip_text_string(tooltip(matlab_param::FLG_RANDOMIZE_KEY).as_deref());
        }
        self.ltf_directory.set_tool_tip_text(Some(
            "The directory which will contain the parameter and project files, logs, intermediate files, and results. Data files can also be located in this directory, but are not required to be.",
        ));
        self.ltf_fn_output.set_tool_tip_text(Some(
            "The base name of the output files for the average volumes, the reference volumes, and the transformation parameters.",
        ));
        self.btn_run.set_tool_tip_text(Some(
            "Perform the alignment search and create averaged volumes.",
        ));
        self.btn_avg_vol
            .set_tool_tip_text(Some("Open the computed averages in 3dmod."));
        self.btn_ref
            .set_tool_tip_text(Some("Open the references in 3dmod."));
        self.btn_average_all
            .set_tool_tip_text(Some("Recompute the averaged volumes"));
        self.rb_init_motl_files.set_tool_tip_text_string(Some(
            "Use the Initial MOTL file(s) specified in the Volume Table.",
        ));
        self.ls_particle_per_cpu.set_tool_tip_text(Some(
            "The maximum number of particles distributed simultaneously to a single CPU during parallel processing.",
        ));
        self.cb_aligned_base_name.set_tool_tip_text_string(Some(
            "Save individual aligned particles to files aligned*.mrc.",
        ));
        self.ls_debug_level.set_tool_tip_text(Some(
            "Larger numbers result in more debug information in the log files.",
        ));
        let tooltip_text = "Start, Incr, and End determine the numbers of particles in an arithmetic sequence for which averages will be created.  I.e. averages will be created containing Start particles, Start + Incr, and so on up to End.";
        self.ltf_lst_thresholds_start
            .set_tool_tip_text(Some(tooltip_text));
        self.ltf_lst_thresholds_increment
            .set_tool_tip_text(Some(tooltip_text));
        self.ltf_lst_thresholds_end
            .set_tool_tip_text(Some(tooltip_text));
        self.ltf_lst_thresholds_additional.set_tool_tip_text(Some(
            "Additional numbers of particles for which averages are desired.  Values must be listed in increasing order and must be larger than End.",
        ));
        self.ltf_select_class_id.set_tool_tip_text(Some(
            "Restrict averaging to members of the specified classes. This is useful only when the motive list contains class numbers (e.g. generated by clusterPca). WARNING: if accidentally set when running a new alignment (or at any other time when class numbers have not been assigned in the motive list), you will get no particles in the new averages / references. Format: a comma or space separated list of integers and/or descriptions (start:optional increment:end)",
        ));
        self.ltf_exclude_list
            .set_tool_tip_text(tooltip(matlab_param::EXCLUDE_LIST_KEY).as_deref());
        self.ltf_include_list
            .set_tool_tip_text(tooltip(matlab_param::INCLUDE_LIST_KEY).as_deref());
        self.cb_flg_elevation_compensation.set_tool_tip_text_string(
            tooltip(matlab_param::FLG_ELEVATION_COMPENSATION_KEY).as_deref(),
        );
        self.cb_flg_frm
            .set_tool_tip_text_string(tooltip(matlab_param::FLG_FRM_KEY).as_deref());
        self.cb_flg_allow_masked_correlation
            .set_tool_tip_text_string(
                tooltip(matlab_param::FLG_ALLOW_MASKED_CORRELATION_KEY).as_deref(),
            );
        self.cb_flg_filter_ref_only
            .set_tool_tip_text_string(tooltip(matlab_param::FLG_FILTER_REF_ONLY_KEY).as_deref());
        self.cb_flg_search_along_particle_axes
            .set_tool_tip_text_string(
                tooltip(matlab_param::FLG_SEARCH_ALONG_PARTICLE_AXES_KEY).as_deref(),
            );
        self.cb_flg_fp_wedge_mask
            .set_tool_tip_text_string(tooltip(matlab_param::FLG_FP_WEDGE_MASK_KEY).as_deref());
        self.ltf_y_axis_symmetry
            .set_tool_tip_text(tooltip(matlab_param::Y_AXIS_SYMMETRY_KEY).as_deref());
        self.cb_flg_use_extracted_particles
            .set_tool_tip_text_string(
                tooltip(matlab_param::FLG_USE_EXTRACTED_PARTICLES_KEY).as_deref(),
            );
        self.cb_cn_symmetric_averaging.set_tool_tip_text_string(Some(
            "Check this box to use an integer N requesting cN axial symmetrization about the tomogram Y axis during all averaging and related operations",
        ));
        self.sp_cn_symmetric_averaging
            .set_tool_tip_text(tooltip(matlab_param::CN_SYMMETRIC_AVERAGING_KEY).as_deref());
        self.cb_flg_cn_masking
            .set_tool_tip_text_string(tooltip(matlab_param::FLG_CN_MASKING_KEY).as_deref());
        // `getTooltip(...) + "  Enclose ..."`: a null tooltip concatenates as "null".
        self.ltf_user_commands.set_tool_tip_text(Some(&format!(
            "{}  Enclose array elements in single quotes.",
            tooltip(matlab_param::USER_COMMANDS_KEY).unwrap_or_else(|| "null".to_owned())
        )));
    }

    /// Java private `setDefaults()`.
    fn set_defaults(&self) {
        self.ls_debug_level
            .set_value_int(matlab_param::DEBUG_LEVEL_DEFAULT);
        self.reference_panel().set_defaults();
        self.missing_wedge_compensation_panel().set_defaults();
        self.spherical_sampling_for_theta_and_psi_panel()
            .set_defaults();
        self.ls_particle_per_cpu
            .set_value_int(matlab_param::PARTICLE_PER_CPU_DEFAULT);
        self.masking_panel().set_defaults();
    }

    /// Java `display(Tab)`.
    pub fn display(&self, tab: Option<Tab>) {
        self.change_tab_tab(tab);
    }

    /// Java private `createSetupPanel()`.
    fn create_setup_panel(&self) {
        // panels
        let pnl_project = JComponent::new_panel();
        let pnl_reference_and_missing_wedge_compensation = JComponent::new_panel();
        let pnl_init_motl_and_y_axis_type = JComponent::new_panel();
        let pnl_init_motl_x = JComponent::new_panel();
        // (APRIL_FOOLS: pnlInitMotlX background colour, painting only.)
        // Initialize
        self.ltf_directory
            .set_overridable_field_displayers(None, Some(self.setup_field_displayer.clone()));
        self.ltf_fn_output
            .set_overridable_field_displayers(None, Some(self.setup_field_displayer.clone()));
        // tab panel
        self.pnl_setup.set_box_layout(spaced_panel::Y_AXIS);
        // Swing layout: pnlSetup.setBorder(BorderFactory.createEtchedBorder()).
        // body
        self.pnl_setup_body.set_box_layout(spaced_panel::Y_AXIS);
        self.pnl_setup_body.set_component_alignment_x(0.5);
        self.pnl_setup_body.add_j_panel(&pnl_project);
        self.pnl_setup_body
            .add_component(&self.fix_paths_panel().get_root_component());
        self.pnl_setup_body
            .add_container(&self.volume_table().get_container());
        self.pnl_setup_body
            .add_j_panel(&pnl_reference_and_missing_wedge_compensation);
        self.pnl_setup_body
            .add_component(&SwingComponent::get_component(&**self.masking_panel()));
        self.pnl_setup_body
            .add_j_panel(&pnl_init_motl_and_y_axis_type);
        // project (BoxLayout X_AXIS, a 10 pixel rigid area and a 20 pixel strut)
        pnl_project.add(&self.ltf_directory.get_container());
        pnl_project.add(&self.ltf_fn_output.get_container());
        // reference and missing wedge compensation (BoxLayout X_AXIS)
        pnl_reference_and_missing_wedge_compensation
            .add(&SwingComponent::get_component(&**self.reference_panel()));
        pnl_reference_and_missing_wedge_compensation.add(&SwingComponent::get_component(
            &**self.missing_wedge_compensation_panel(),
        ));
        // init MOTL and Y axis type (BoxLayout X_AXIS, glue between)
        pnl_init_motl_and_y_axis_type.add(&self.y_axis_type_panel().get_component());
        pnl_init_motl_and_y_axis_type.add(&pnl_init_motl_x);
        // init motl x (BoxLayout X_AXIS, a 167 pixel rigid area after)
        pnl_init_motl_x.set_border_title(
            EtchedBorder::new(Some(shared_strings::INIT_MOTL_LABEL))
                .get_border()
                .get_title()
                .as_deref(),
        );
        pnl_init_motl_x.add(&self.pnl_init_motl.get_component());
        // init MOTL (BoxLayout Y_AXIS)
        let pnl_init_motl = self.pnl_init_motl.get_component();
        pnl_init_motl.add(&self.rb_init_motl_zero.get_component());
        pnl_init_motl.add(&self.rb_init_align_particle_y_axes.get_component());
        pnl_init_motl.add(&self.rb_init_motl_files.get_component());
        pnl_init_motl.add(&self.rb_init_motl_random_rotations.get_component());
        pnl_init_motl.add(&self.rb_init_motl_random_axial_rotations.get_component());
    }

    /// Java private `createRunPanel()`.
    fn create_run_panel(&self) {
        // panels
        let pnl_lst_thresholds = SpacedPanel::get_instance_void();
        let pnl_equal_number = JComponent::new_panel();
        let pnl_button = JComponent::new_panel();
        let pnl_ref_flag_all_tom = JComponent::new_panel();
        let pnl_lst_flag_all_tom = JComponent::new_panel();
        // initialize
        self.ltf_lst_thresholds_start.set_preferred_width(60);
        self.ltf_lst_thresholds_increment.set_preferred_width(60);
        self.ltf_lst_thresholds_end.set_preferred_width(60);
        self.ls_particle_per_cpu.set_preferred_width(60);
        // panel for tab (BoxLayout Y_AXIS, etched border)
        // body
        self.pnl_run_body.set_box_layout(spaced_panel::Y_AXIS);
        self.pnl_run_body.set_component_alignment_x(0.5);
        self.pnl_run_body
            .add_container(&self.ls_particle_per_cpu.get_container());
        self.pnl_run_body
            .add_container(&self.iteration_table().get_container());
        self.pnl_run_body
            .add_component(&SwingComponent::get_component(
                &**self.spherical_sampling_for_theta_and_psi_panel(),
            ));
        self.pnl_run_body.add_spaced_panel(&pnl_lst_thresholds);
        self.pnl_run_body.add_j_panel(&pnl_equal_number);
        self.pnl_run_body.add_rigid_area_dimension((0, 20));
        self.pnl_run_body.add_j_panel(&pnl_button);
        // lstThresholds
        pnl_lst_thresholds.set_box_layout(spaced_panel::X_AXIS);
        pnl_lst_thresholds.set_border(&EtchedBorder::new(Some(LST_THRESHOLDS_LABEL)).get_border());
        pnl_lst_thresholds.add_rigid_area_dimension((20, 0));
        pnl_lst_thresholds.add_container(&self.ltf_lst_thresholds_start.get_container());
        pnl_lst_thresholds.add_rigid_area_dimension((10, 0));
        pnl_lst_thresholds.add_container(&self.ltf_lst_thresholds_increment.get_container());
        pnl_lst_thresholds.add_horizontal_glue();
        pnl_lst_thresholds.add_rigid_area_dimension((10, 0));
        pnl_lst_thresholds.add_container(&self.ltf_lst_thresholds_end.get_container());
        pnl_lst_thresholds.add_rigid_area_dimension((20, 0));
        pnl_lst_thresholds.add_container(&self.ltf_lst_thresholds_additional.get_container());
        pnl_lst_thresholds.add_horizontal_glue();
        // equals numbers (BoxLayout X_AXIS, glue first)
        pnl_equal_number.set_border_title(
            EtchedBorder::new(Some("Use Equal Numbers of Particles from All Tomograms"))
                .get_border()
                .get_title()
                .as_deref(),
        );
        pnl_equal_number.add(&pnl_lst_flag_all_tom);
        pnl_equal_number.add(&pnl_ref_flag_all_tom);
        // LstFlagAllTom (BoxLayout X_AXIS, centered, glue after)
        pnl_lst_flag_all_tom.add(&self.cb_lst_flag_all_tom.get_component());
        // RefFlagAllTom (BoxLayout X_AXIS, centered, glue after)
        pnl_ref_flag_all_tom.add(&self.cb_ref_flag_all_tom.get_component());
        // button panel (BoxLayout X_AXIS, rigid areas 30, 40, 30, 30, 30)
        pnl_button.add(&self.btn_run.base.get_component());
        pnl_button.add(&self.btn_avg_vol.base.base.get_component());
        pnl_button.add(&self.btn_ref.base.base.get_component());
        pnl_button.add(&self.btn_average_all.base.get_component());
    }

    /// Java package-private `msgFlgVolNamesAreTemplates(boolean, boolean)`.
    pub fn msg_flg_vol_names_are_templates(&self, init: bool, on: bool) {
        self.reference_panel()
            .msg_flg_vol_names_are_templates(init, on);
    }

    /// Java private `createMoreOptionsPanel()`.
    fn create_more_options_panel(&self) {
        let pnl_particle_selection = JComponent::new_panel();
        let pnl_exclude_list = JComponent::new_panel();
        let pnl_include_list = JComponent::new_panel();
        let pnl_select_class_id = JComponent::new_panel();
        let pnl_particle_selection_flags = JComponent::new_panel();
        let pnl_elevation_compensation = JComponent::new_panel();
        let pnl_alignment_outer = JComponent::new_panel();
        let pnl_alignment = JComponent::new_panel();
        let pnl_frm = JComponent::new_panel();
        let pnl_filter_ref_only = JComponent::new_panel();
        let pnl_fp_wedge_mask = JComponent::new_panel();
        let pnl_y_axis_symmetry = JComponent::new_panel();
        let pnl_processing_outer = JComponent::new_panel();
        let pnl_processing = JComponent::new_panel();
        let pnl_align_averages = JComponent::new_panel();
        let pnl_no_reference_refinement = JComponent::new_panel();
        let pnl_cn_symmetric_averaging = JComponent::new_panel();
        let pnl_user_commands = JComponent::new_panel();
        let pnl_debug_level = JComponent::new_panel();
        // (FlowLayout LEADING and LEFT: layout only.)
        // initialize
        self.ls_debug_level.set_preferred_width(50);
        self.cb_flg_abs_value
            .set_selected_boolean(matlab_param::FLG_ABS_VALUE_DEFAULT);
        self.cb_flg_frm.set_selected_boolean(true);
        self.cb_flg_cn_masking.set_selected_boolean(true);
        for field in [
            &self.ltf_exclude_list,
            &self.ltf_include_list,
            &self.ltf_select_class_id,
            &self.ltf_y_axis_symmetry,
            &self.ltf_user_commands,
        ] {
            field.set_overridable_field_displayers(
                None,
                Some(self.more_options_field_displayer.clone()),
            );
        }
        self.ltf_user_commands.set_parsable_string(true);
        //
        self.pnl_more_options_body
            .set_box_layout(spaced_panel::Y_AXIS);
        self.pnl_more_options_body.set_component_alignment_x(0.0);
        self.pnl_more_options_body
            .add_j_panel(&pnl_particle_selection);
        self.pnl_more_options_body.add_j_panel(&pnl_alignment_outer);
        self.pnl_more_options_body
            .add_j_panel(&pnl_processing_outer);
        // Particle Selection (BoxLayout Y_AXIS, rigid areas between)
        pnl_particle_selection.set_border_title(
            EtchedBorder::new(Some("Particle Selection"))
                .get_border()
                .get_title()
                .as_deref(),
        );
        pnl_particle_selection.add(&pnl_exclude_list);
        pnl_particle_selection.add(&pnl_include_list);
        pnl_particle_selection.add(&pnl_select_class_id);
        pnl_particle_selection.add(&pnl_particle_selection_flags);
        // exclude list
        pnl_exclude_list.add(&self.ltf_exclude_list.get_component());
        // include list
        pnl_include_list.add(&self.ltf_include_list.get_component());
        // select class ID
        pnl_select_class_id.add(&self.ltf_select_class_id.get_component());
        // particle selection flags (GridLayout 1 x 2)
        pnl_particle_selection_flags.add(&pnl_elevation_compensation);
        pnl_particle_selection_flags.add(&self.cb_flg_randomize.get_component());
        // elevation compensation
        pnl_elevation_compensation.add(&self.cb_flg_elevation_compensation.get_component());

        // Alignment (BoxLayout Y_AXIS)
        pnl_alignment_outer.set_border_title(
            EtchedBorder::new(Some("Alignment"))
                .get_border()
                .get_title()
                .as_deref(),
        );
        pnl_alignment_outer.add(&pnl_alignment);
        pnl_alignment_outer.add(&pnl_y_axis_symmetry);
        // (GridLayout 3 x 3)
        pnl_alignment.add(&pnl_frm);
        pnl_alignment.add(&self.cb_flg_allow_masked_correlation.get_component());
        pnl_alignment.add(&pnl_filter_ref_only);
        pnl_alignment.add(&self.cb_flg_search_along_particle_axes.get_component());
        pnl_alignment.add(&pnl_fp_wedge_mask);
        pnl_alignment.add(&self.cb_flg_abs_value.get_component());
        // FRM
        pnl_frm.add(&self.cb_flg_frm.get_component());
        // filter reference only
        pnl_filter_ref_only.add(&self.cb_flg_filter_ref_only.get_component());
        // FP wedge mask
        pnl_fp_wedge_mask.add(&self.cb_flg_fp_wedge_mask.get_component());
        // y axis symmetry
        pnl_y_axis_symmetry.add(&self.ltf_y_axis_symmetry.get_component());

        // Processing (BoxLayout Y_AXIS)
        pnl_processing_outer.set_border_title(
            EtchedBorder::new(Some("Processing"))
                .get_border()
                .get_title()
                .as_deref(),
        );
        pnl_processing_outer.add(&pnl_processing);
        pnl_processing_outer.add(&pnl_user_commands);
        pnl_processing_outer.add(&pnl_debug_level);
        // (GridLayout 3 x 3)
        pnl_processing.add(&pnl_align_averages);
        pnl_processing.add(&self.cb_flg_use_extracted_particles.get_component());
        pnl_processing.add(&pnl_no_reference_refinement);
        pnl_processing.add(&self.cb_aligned_base_name.get_component());
        pnl_processing.add(&pnl_cn_symmetric_averaging);
        pnl_processing.add(&self.cb_flg_cn_masking.get_component());
        // align averages
        pnl_align_averages.add(&self.cbflg_align_averages.get_component());
        // template matching/no reference refinement
        pnl_no_reference_refinement.add(&self.cb_flg_no_reference_refinement.get_component());
        // c<N> symmetric averaging
        pnl_cn_symmetric_averaging.add(&self.cb_cn_symmetric_averaging.get_component());
        pnl_cn_symmetric_averaging.add(&self.l_cn_symmetric_averaging);
        pnl_cn_symmetric_averaging.add(&self.sp_cn_symmetric_averaging.get_container());
        // user commands
        pnl_user_commands.add(&self.ltf_user_commands.get_component());
        // debug level
        pnl_debug_level.add(&self.ls_debug_level.get_container());
    }

    /// Java private `validateRun()`.
    fn validate_run(&self) -> bool {
        let message = |text: &str| {
            ui_harness::with(|harness| {
                harness.open_message_dialog_base_manager_string_string(
                    Some(self.manager),
                    text,
                    "Entry Error",
                )
            })
        };
        // Setup tab
        // Must have a directory
        if self.ltf_directory.is_empty() {
            self.goto_setup_tab();
            message(&format!("Please set the {DIRECTORY_LABEL} field."));
            return false;
        }
        // Must have an output name
        if self.ltf_fn_output.is_empty() {
            self.goto_setup_tab();
            message(&format!("Please set the {FN_OUTPUT_LABEL} field."));
            return false;
        }
        // Validate volume table
        let error_message = self.volume_table().validate_run(
            self.missing_wedge_compensation_panel()
                .is_tilt_range_required(),
        );
        if let Some(error_message) = error_message {
            self.goto_setup_tab();
            message(&error_message);
            return false;
        }
        // Must either have a volume and particle or a reference file.
        if let Some(error_message) = self.reference_panel().validate_run() {
            self.goto_setup_tab();
            message(&error_message);
            return false;
        }
        // Validate missing wedge compensation panel
        if let Some(error_message) = self.missing_wedge_compensation_panel().validate_run() {
            self.goto_setup_tab();
            message(&error_message);
            return false;
        }
        // Validate masking
        if let Some(error_message) = self.masking_panel().validate_run() {
            self.goto_setup_tab();
            message(&error_message);
            return false;
        }
        // Run tab
        if !self.iteration_table().validate_run() {
            return false;
        }
        // spherical sampling for theta and psi:
        if !self
            .spherical_sampling_for_theta_and_psi_panel()
            .validate_run()
        {
            return false;
        }
        // Number of particles
        let start_is_empty = self.ltf_lst_thresholds_start.is_empty();
        let increment_is_empty = self.ltf_lst_thresholds_increment.is_empty();
        let end_is_empty = self.ltf_lst_thresholds_end.is_empty();
        let additional_is_empty = self.ltf_lst_thresholds_additional.is_empty();
        if start_is_empty && increment_is_empty {
            // lst thesholds is required
            if additional_is_empty {
                message(&format!("{LST_THRESHOLDS_LABEL} is required."));
                return false;
            }
            // check empty list descriptor
            if !end_is_empty {
                message(&format!(
                    "In {LST_THRESHOLDS_LABEL}, invalid list description."
                ));
                return false;
            }
        }
        // check list descriptor
        else if start_is_empty || end_is_empty {
            message(&format!(
                "In {LST_THRESHOLDS_LABEL}, invalid list description."
            ));
            return false;
        }
        true
    }

    /// Java private `gotoSetupTab()`.
    fn goto_setup_tab(&self) {
        self.tab_pane.get_component().set_selected_tab(0);
        self.change_tab_void();
    }

    /// Java private `changeTab(Tab)`.
    fn change_tab_tab(&self, tab: Option<Tab>) {
        let Some(tab) = tab else {
            return;
        };
        self.tab_pane.get_component().set_selected_tab(tab.index());
        self.change_tab_void();
    }

    /// Java private `changeTab()`.
    fn change_tab_void(&self) {
        let selected_index = self.tab_pane.get_component().get_selected_tab();
        if selected_index == 0 {
            self.pnl_setup.add_spaced_panel(&self.pnl_setup_body);
            self.pnl_run
                .get_component()
                .remove(&self.pnl_run_body.get_container());
            self.pnl_more_options.remove(&self.pnl_more_options_body);
        } else if selected_index == 1 {
            self.pnl_run
                .get_component()
                .add(&self.pnl_run_body.get_container());
            self.pnl_setup.remove(&self.pnl_setup_body);
            self.pnl_more_options.remove(&self.pnl_more_options_body);
        } else {
            self.pnl_more_options
                .add_spaced_panel(&self.pnl_more_options_body);
            self.pnl_run
                .get_component()
                .remove(&self.pnl_run_body.get_container());
            self.pnl_setup.remove(&self.pnl_setup_body);
        }
        if let (Some(mediator), Some(this)) = (self.mediator(), self.this.upgrade()) {
            let origin: Rc<dyn ProcessInterface> = this;
            mediator.set_method_process_interface_processing_method(
                &origin,
                self.get_processing_method(),
            );
        }
        ui_harness::with(|harness| {
            harness.pack_axis_id_base_manager(Some(self.axis_id), Some(self.manager))
        });
        ui_harness::with(|harness| harness.move_sub_frame());
    }

    /// Java private `addListeners()`.
    fn add_listeners(&self) {
        let adaptee = self.this.clone();
        let action_listener: ActionListener = Rc::new(move |event: &ActionEvent| {
            if let Some(adaptee) = adaptee.upgrade() {
                adaptee.action(event.get_action_command().unwrap_or(""), None, None);
            }
        });
        self.rb_init_motl_zero
            .add_action_listener(action_listener.clone());
        self.rb_init_align_particle_y_axes
            .add_action_listener(action_listener.clone());
        self.rb_init_motl_random_rotations
            .add_action_listener(action_listener.clone());
        self.rb_init_motl_random_axial_rotations
            .add_action_listener(action_listener.clone());
        self.rb_init_motl_files
            .add_action_listener(action_listener.clone());
        self.btn_run.add_action_listener(action_listener.clone());
        let adaptee = self.this.clone();
        let tab_change_listener: ChangeListener = Rc::new(move |_event: &ChangeEvent| {
            if let Some(adaptee) = adaptee.upgrade() {
                adaptee.change_tab_void();
            }
        });
        self.tab_pane
            .get_component()
            .add_change_listener(tab_change_listener);
        self.btn_avg_vol
            .add_action_listener(action_listener.clone());
        self.btn_ref.add_action_listener(action_listener.clone());
        self.btn_average_all
            .add_action_listener(action_listener.clone());
        self.cb_cn_symmetric_averaging
            .add_action_listener(Some(action_listener));
    }

    /// Java `setIterationRows(MatlabParam)`.
    pub fn set_iteration_rows(&self, matlab_param: &mut MatlabParam) {
        self.iteration_table().add_iteration_rows(matlab_param);
    }

    /// Java `updateDisplay(boolean)`.  Enabled/disables fields.  Calls updateDisplay()
    /// in subordinate panels.
    pub fn update_display(&self, init: bool) {
        // tilt range
        let _volume_rows = self.volume_table().size() > 0;
        self.reference_panel().update_display(init);
        self.missing_wedge_compensation_panel().update_display();
        self.spherical_sampling_for_theta_and_psi_panel()
            .update_display();
        // iteration table - spherical sampling and FlgRemoveDuplicates
        self.iteration_table().update_display(
            !self
                .spherical_sampling_for_theta_and_psi_panel()
                .is_sample_sphere_none_selected(),
        );
        // volume table
        self.volume_table().update_display(
            self.rb_init_motl_files.is_selected(),
            self.missing_wedge_compensation_panel()
                .is_tilt_range_required(),
            self.missing_wedge_compensation_panel()
                .is_tilt_range_multi_axes(),
        );
        self.masking_panel().update_display();
        self.cbflg_align_averages
            .set_enabled(self.y_axis_type_panel().get_y_axis_type() != Some(YAxisType::YAxis));
        self.sp_cn_symmetric_averaging
            .set_enabled(self.cb_cn_symmetric_averaging.is_selected());
    }
}

impl ContextMenu for PeetDialog {
    /// Java `popUpContextMenu(MouseEvent)`.  Right mouse button context menu.
    fn pop_up_context_menu(&self, mouse_event: &MouseEvent) {
        let man_pagelabel = ["Processchunks".to_owned(), "3dmod".to_owned()];
        let man_page = ["processchunks.html".to_owned(), "3dmod.html".to_owned()];
        if let Err(e) =
            ContextPopup::new_component_mouse_event_string_array_string_array_boolean_base_manager_axis_id(
                &self.root_panel.get_component(),
                mouse_event,
                &man_pagelabel,
                &man_page,
                true,
                self.manager,
                self.axis_id,
            )
        {
            eprintln!("{e}");
        }
    }
}

impl AbstractParallelDialog for PeetDialog {
    /// Java `getParameters(ParallelParam)`: casts to ProcesschunksParam and sets
    /// nothing.
    fn get_parameters(&self, _param: &mut dyn ParallelParam) {}

    /// Java `getDialogType()`.
    fn get_dialog_type(&self) -> DialogType {
        DIALOG_TYPE
    }
}

impl Run3dmodButtonContainer for PeetDialog {
    /// Java `action(String, Deferred3dmodButton, Run3dmodMenuOptions)`.
    fn action(
        &self,
        action_command: &str,
        _deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    ) {
        let is = |command: Option<String>| command.as_deref() == Some(action_command);
        if is(self.btn_run.get_action_command()) {
            if self.validate_run() {
                let run_method = self.mediator().map(|mediator| {
                    mediator.get_run_method_for_process_interface(self.get_processing_method())
                });
                self.manager
                    .peet_parser(None, Some(DIALOG_TYPE), run_method);
            }
        } else if is(self.rb_init_motl_zero.get_action_command())
            || is(self.rb_init_align_particle_y_axes.get_action_command())
            || is(self.rb_init_motl_random_rotations.get_action_command())
            || is(self
                .rb_init_motl_random_axial_rotations
                .get_action_command())
            || is(self.rb_init_motl_files.get_action_command())
        {
            self.update_display(false);
        } else if is(self.btn_avg_vol.get_action_command()) {
            self.manager.imod_avg_vol(run_3dmod_menu_options);
        } else if is(self.btn_ref.get_action_command()) {
            self.manager.imod_ref(run_3dmod_menu_options);
        } else if is(self.btn_average_all.get_action_command()) {
            self.manager.average_all(None, Some(DIALOG_TYPE));
        } else if is(self.cb_cn_symmetric_averaging.get_action_command()) {
            self.sp_cn_symmetric_averaging
                .set_enabled(self.cb_cn_symmetric_averaging.is_selected());
        }
    }
}

impl FileContainer for PeetDialog {
    /// Java `fixIncorrectPaths(boolean)`.  Fix all of the incorrect paths until the
    /// user cancels a file chooser.
    fn fix_incorrect_paths(&self, choose_path_every_row: bool) {
        if !self
            .volume_table()
            .fix_incorrect_paths(choose_path_every_row)
        {
            return;
        }
        if !self
            .reference_panel()
            .fix_incorrect_paths(choose_path_every_row)
        {
            return;
        }
        if !self
            .masking_panel()
            .fix_incorrect_paths(choose_path_every_row)
        {
            return;
        }
        self.check_incorrect_paths();
    }
}

impl PeetDialog {
    /// Java `fixIncorrectPath(FileTextFieldInterface, boolean)`.  Returns false if the
    /// user cancels the file selector, otherwise true.
    fn fix_incorrect_path_impl(
        &self,
        file_text_field: &dyn FileTextFieldInterface,
        choose_path: bool,
    ) -> bool {
        let mut new_file: Option<PathBuf> = None;
        while new_file.as_ref().is_none_or(|file| !file.exists()) {
            // Have the user choose the location of the file if they haven't chosen
            // before or they want to choose most of the files individuallly, otherwise
            // just use the current correctPath.
            let correct_path = self.correct_path.borrow().clone();
            if correct_path.is_none()
                || choose_path
                || new_file.as_ref().is_some_and(|file| !file.exists())
            {
                let file_chooser = self.get_file_chooser_instance();
                file_chooser.set_selected_file(file_text_field.get_file().as_deref());
                // Swing layout: fileChooser.setPreferredSize(
                // UIParameters.getInstance().getFileChooserDimension()).
                let _ = UIParameters::get_instance_void().get_file_chooser_dimension();
                file_chooser.set_file_selection_mode(file_chooser::FILES_ONLY);
                file_chooser.set_file_filter(file_text_field.get_file_filter());
                let return_val =
                    file_chooser.show_open_dialog(Some(&self.root_panel.get_component()));
                if return_val != file_chooser::APPROVE_OPTION {
                    return false;
                }
                new_file = file_chooser.get_selected_file();
                if let Some(file) = new_file.as_ref().filter(|file| file.exists()) {
                    let parent = file
                        .parent()
                        .map(|parent| parent.to_string_lossy().into_owned());
                    *self.last_location.borrow_mut() = parent.as_ref().map(PathBuf::from);
                    *self.correct_path.borrow_mut() = parent;
                    file_text_field.set_file(Some(file.clone()));
                }
            } else if let Some(correct_path) = correct_path {
                // Fixed in translation (PeetDialog.java:302): a field with no file
                // makes Java's `getFile().getName()` throw NullPointerException; the
                // field's name is taken as empty.
                let name = file_text_field
                    .get_file()
                    .and_then(|file| file.file_name().map(|name| name.to_owned()))
                    .unwrap_or_default();
                let file = Path::new(&correct_path).join(name);
                if file.exists() {
                    file_text_field.set_file(Some(file.clone()));
                }
                new_file = Some(file);
            }
        }
        true
    }
}

impl ReferenceParent for PeetDialog {
    fn fix_incorrect_path(
        &self,
        file_text_field: &dyn FileTextFieldInterface,
        choose_path: bool,
    ) -> bool {
        self.fix_incorrect_path_impl(file_text_field, choose_path)
    }
    /// Java `getVolumeTableSize()`.
    fn get_volume_table_size(&self) -> i32 {
        self.volume_table().size()
    }
    fn update_display(&self, init: bool) {
        PeetDialog::update_display(self, init)
    }
    /// Java `isFlgVolNamesAreTemplates()`.
    fn is_flg_vol_names_are_templates(&self) -> bool {
        self.volume_table().is_flg_vol_names_are_templates()
    }
}

impl MaskingParent for PeetDialog {
    /// Java `isReferenceFileSelected()`.
    fn is_reference_file_selected(&self) -> bool {
        self.reference_panel().is_reference_file_selected()
    }
    fn get_volume_table_size(&self) -> i32 {
        self.volume_table().size()
    }
    fn fix_incorrect_path(
        &self,
        file_text_field: &dyn FileTextFieldInterface,
        choose_path: bool,
    ) -> bool {
        self.fix_incorrect_path_impl(file_text_field, choose_path)
    }
    fn update_display(&self, init: bool) {
        PeetDialog::update_display(self, init)
    }
}

impl MissingWedgeCompensationParent for PeetDialog {
    /// Java `isVolumeTableEmpty()`.
    fn is_volume_table_empty(&self) -> bool {
        self.volume_table().is_empty()
    }
    /// Java `isReferenceParticleSelected()`.
    fn is_reference_particle_selected(&self) -> bool {
        self.reference_panel().is_reference_particle_selected()
    }
    fn update_display(&self, init: bool) {
        PeetDialog::update_display(self, init)
    }
}

impl IterationParent for PeetDialog {
    fn update_display(&self, init: bool) {
        PeetDialog::update_display(self, init)
    }
    /// Java `isSampleSphere()`.
    fn is_sample_sphere(&self) -> bool {
        !self
            .spherical_sampling_for_theta_and_psi_panel()
            .is_sample_sphere_none_selected()
    }
}

impl YAxisTypeParent for PeetDialog {
    fn update_display(&self, init: bool) {
        PeetDialog::update_display(self, init)
    }
    fn is_volume_table_empty(&self) -> bool {
        self.volume_table().is_empty()
    }
}

impl SphericalSamplingForThetaAndPsiParent for PeetDialog {
    fn update_display(&self, init: bool) {
        PeetDialog::update_display(self, init)
    }
}

impl QueueTableListener for PeetDialog {
    /// Java `queueTableEventAction(QueueTableEvent)`: empty.
    fn queue_table_event_action(&self, _event: &QueueTableEvent) {}
}

impl ProcessInterface for PeetDialog {
    /// Java `updateGpu(boolean)`: empty.
    fn update_gpu(&self, _input: bool) {}

    /// Java `getProcessingMethod()`.  Get the processing method based on the dialogs
    /// settings.
    fn get_processing_method(&self) -> ProcessingMethod {
        if self.tab_pane.get_component().get_selected_tab() == 1 {
            return ProcessingMethod::PpCpu;
        }
        ProcessingMethod::LocalCpu
    }

    /// Java `getSecondaryProcessingMethod()`.
    fn get_secondary_processing_method(&self) -> Option<ProcessingMethod> {
        None
    }

    /// Java `lockProcessingMethod(boolean)`: empty.
    fn lock_processing_method(&self, _lock: bool) {}

    /// Java `setMethod(ProcessingMethod)`.
    fn set_method(&self, processing_method: ProcessingMethod) {
        if let (Some(mediator), Some(this)) = (self.mediator(), self.this.upgrade()) {
            let origin: Rc<dyn ProcessInterface> = this;
            mediator.set_method_process_interface_processing_method(&origin, processing_method);
        }
    }

    /// Java `isUseGpu()`.
    fn is_use_gpu(&self) -> bool {
        false
    }

    /// Java `setUseQueueCheckBox(ButtonComponent)`: empty.
    fn set_use_queue_check_box(&self, _use_queue_check_box: Option<Rc<dyn ButtonComponent>>) {}

    /// Java `addQueueTableListener(QueueTableListener)`: empty.
    fn add_queue_table_listener(&self, _listener: Rc<dyn QueueTableListener>) {}

    /// Java `removeQueueTableListener(QueueTableListener)`: empty.
    fn remove_queue_table_listener(&self, _listener: &Rc<dyn QueueTableListener>) {}
}

impl SwingComponent for PeetDialog {
    /// Java `getComponent()`.
    fn get_component(&self) -> Rc<JComponent> {
        self.root_panel.get_component()
    }
}

impl UIComponent for PeetDialog {
    /// Java `getUIComponent()`.
    fn get_ui_component(&self) -> &dyn SwingComponent {
        self
    }

    fn get_component(&self) -> Rc<JComponent> {
        self.root_panel.get_component()
    }
}
