//! `IMOD/Etomo/src/etomo/ui/swing/JoinDialog.java`.
//!
//! The Join interface's dialog: a tabbed pane with the Setup, Align, Join, Model and
//! Rejoin tabs.  Only the current tab's panel holds its components; changing tabs
//! moves the shared section table (and the boundary table) to the new tab.
//!
//! **Representation.**  An event dispatch thread object, created as `Rc<Self>` by
//! [`JoinDialog::get_instance_meta_data`] / [`JoinDialog::get_instance_working_dir`].
//! The Java field initialisers run in the `Rc::new_cyclic` closure; the fields the
//! constructor body (or a `create...Panel` method) assigns are `RefCell<Option<..>>`
//! / `OnceCell`, set once the dialog exists, because the section and boundary tables
//! call back into the dialog while they are being built.  The listener classes are
//! closures holding a weak reference to the dialog.

use std::cell::{Cell, OnceCell, RefCell};
use std::path::{Path, PathBuf};
use std::rc::{Rc, Weak};

use super::auto_alignment_panel::AutoAlignmentPanel;
use super::binned_xy_3dmod_button::BinnedXY3dmodButton;
use super::boundary_table::BoundaryTable;
use super::check_box::CheckBox;
use super::check_box_spinner::CheckBoxSpinner;
use super::context_menu::ContextMenu;
use super::context_popup::{self, ContextPopup};
use super::deferred_3dmod_button::Deferred3dmodButton;
use super::etched_border::EtchedBorder;
use super::etomo_panel::EtomoPanel;
use super::file_chooser::{self, FileChooser};
use super::file_text_field::FileTextField;
use super::file_text_field_interface::FileTextFieldInterface;
use super::generic_mouse_adapter::GenericMouseAdapter;
use super::labeled_spinner::LabeledSpinner;
use super::labeled_text_field::LabeledTextField;
use super::multi_line_button::MultiLineButton;
use super::run_3dmod_button::Run3dmodButton;
use super::run_3dmod_button_container::Run3dmodButtonContainer;
use super::section_table_panel::SectionTablePanel;
use super::spaced_panel::{self, SpacedPanel};
use super::tabbed_pane::TabbedPane;
use super::transform_chooser_panel::TransformChooserPanel;
use super::ui_harness;
use super::ui_parameters::UIParameters;
use crate::imod::etomo::auto_alignment_controller::AutoAlignmentController;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::finishjoin_param;
use crate::imod::etomo::comscript::joinwarp2model_param::Joinwarp2modelParam;
use crate::imod::etomo::comscript::midas_param::MidasParam;
use crate::imod::etomo::comscript::xfalign_param::XfalignParam;
use crate::imod::etomo::comscript::xfjointomo_param::{self, XfjointomoParam};
use crate::imod::etomo::jdk::{
    ActionEvent, ActionListener, ChangeEvent, ChangeListener, FileFilter, JComponent, MouseEvent,
    SpinnerNumberModel,
};
use crate::imod::etomo::join_manager::JoinManager;
use crate::imod::etomo::logic::dataset_tool;
use crate::imod::etomo::process::imod_manager;
use crate::imod::etomo::process::imod_process::{self, Run3dmodMenuOptions};
use crate::imod::etomo::storage::autodoc::autodoc::Autodoc;
use crate::imod::etomo::storage::autodoc::autodoc_factory;
use crate::imod::etomo::storage::autodoc::read_only_autodoc::ReadOnlyAutodoc;
use crate::imod::etomo::storage::log_file::LogFileError;
use crate::imod::etomo::storage::model_file_filter::ModelFileFilter;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::{ConstEtomoNumber, Number, Type};
use crate::imod::etomo::r#type::const_join_meta_data::ConstJoinMetaData;
use crate::imod::etomo::r#type::data_file_type::DataFileType;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::etomo_autodoc;
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::r#type::file_type;
use crate::imod::etomo::r#type::join_meta_data::JoinMetaData;
use crate::imod::etomo::r#type::join_screen_state::JoinScreenState;
use crate::imod::etomo::r#type::join_state::JoinState;
use crate::imod::etomo::r#type::null_required_number_exception::NullRequiredNumberException;
use crate::imod::etomo::ui::auto_alignment_display::AutoAlignmentDisplay;
use crate::imod::etomo::ui::field::Field;
use crate::imod::etomo::ui::field_type::FieldType;
use crate::imod::etomo::ui::field_validation_failed_exception::FieldValidationFailedException;
use crate::imod::etomo::util::dataset_files;
use crate::imod::etomo::util::utilities;

/// Java `public static final int SETUP_MODE`.
pub const SETUP_MODE: i32 = -1;
/// Java `public static final int SAMPLE_NOT_PRODUCED_MODE`.
pub const SAMPLE_NOT_PRODUCED_MODE: i32 = -2;
/// Java `public static final int SAMPLE_PRODUCED_MODE`.
pub const SAMPLE_PRODUCED_MODE: i32 = -3;
/// Java `public static final int CHANGING_SAMPLE_MODE`.
pub const CHANGING_SAMPLE_MODE: i32 = -4;
/// Java `public static final DialogType DIALOG_TYPE`.
pub const DIALOG_TYPE: DialogType = DialogType::Join;
/// Java `public static final String FINISH_JOIN_TEXT`.
pub const FINISH_JOIN_TEXT: &str = "Finish Join";
/// Java `public static final String WORKING_DIRECTORY_TEXT`.
pub const WORKING_DIRECTORY_TEXT: &str = "Working directory";
/// Java `public static final String GET_MAX_SIZE_TEXT`.
pub const GET_MAX_SIZE_TEXT: &str = "Get Max Size and Shift";
/// Java `public static final String TRIAL_JOIN_TEXT`.
pub const TRIAL_JOIN_TEXT: &str = "Trial Join";
/// Java `public static final String REJOIN_TEXT`.
pub const REJOIN_TEXT: &str = "Rejoin";
/// Java `public static final String TRIAL_REJOIN_TEXT`.
pub const TRIAL_REJOIN_TEXT: &str = "Trial Rejoin";
/// Java private static final `REFINE_JOIN_TEXT`.
const REFINE_JOIN_TEXT: &str = "Refine Join";
/// Java private static final `OPEN_IN_3DMOD`.
const OPEN_IN_3DMOD: &str = "Open in 3dmod";

/// Java package-private static final nested class `Tab`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Tab {
    /// Java `SETUP`.
    Setup,
    /// Java `ALIGN`.
    Align,
    /// Java `JOIN`.
    Join,
    /// Java `MODEL`.
    Model,
    /// Java `REJOIN`.
    Rejoin,
}

impl Tab {
    /// Java `getIndex()`.
    pub fn get_index(self) -> i32 {
        match self {
            Tab::Setup => 0,
            Tab::Align => 1,
            Tab::Join => 2,
            Tab::Model => 3,
            Tab::Rejoin => 4,
        }
    }

    /// Java static `getInstance(int)`.  Get the tab associated with the index.
    /// Default: SETUP.
    pub fn get_instance(index: i32) -> Tab {
        match index {
            0 => Tab::Setup,
            1 => Tab::Align,
            2 => Tab::Join,
            3 => Tab::Model,
            4 => Tab::Rejoin,
            _ => Tab::Setup,
        }
    }
}

/// Java `public final class JoinDialog implements ContextMenu,
/// Run3dmodButtonContainer, AutoAlignmentDisplay`.
pub struct JoinDialog {
    /// Java private final `ltfRootName`.
    ltf_root_name: Rc<LabeledTextField>,
    /// Java private `rootPanel`.
    root_panel: OnceCell<Rc<JComponent>>,
    /// Java private `tabPane`.
    tab_pane: OnceCell<Rc<TabbedPane>>,
    /// Java private final `pnlSetupTab = SpacedPanel.getFocusableInstance()`.
    pnl_setup_tab: Rc<SpacedPanel>,
    /// Java private `pnlSectionTable`.
    pnl_section_table: OnceCell<Rc<SectionTablePanel>>,
    /// Java private final `pnlAlignTab = SpacedPanel.getFocusableInstance()`.
    pnl_align_tab: Rc<SpacedPanel>,
    /// Java private final `pnlJoinTab = SpacedPanel.getFocusableInstance()`.
    pnl_join_tab: Rc<SpacedPanel>,
    /// Java private `setupPanel2`.
    setup_panel2: OnceCell<Rc<SpacedPanel>>,
    /// Java private `alignPanel1`.
    align_panel1: OnceCell<Rc<SpacedPanel>>,
    /// Java private `pnlFinishJoin`.
    pnl_finish_join: OnceCell<Rc<SpacedPanel>>,
    /// Java private `pnlMidasLimit = SpacedPanel.getInstance()`.
    pnl_midas_limit: Rc<SpacedPanel>,
    /// Java private `ftfWorkingDir`.
    ftf_working_dir: OnceCell<Rc<FileTextField>>,
    /// Java private `btnOpenSample`.
    btn_open_sample: OnceCell<Rc<Run3dmodButton>>,
    /// Java private `btnGetMaxSize`.
    btn_get_max_size: OnceCell<Rc<MultiLineButton>>,
    /// Java private `btnGetSubarea`.
    btn_get_subarea: OnceCell<Rc<MultiLineButton>>,
    /// Java private `btnChangeSetup`.
    btn_change_setup: OnceCell<Rc<MultiLineButton>>,
    /// Java private `btnRevertToLastSetup`.
    btn_revert_to_last_setup: OnceCell<Rc<MultiLineButton>>,
    /// Java private `ltfSizeInX`.
    ltf_size_in_x: OnceCell<Rc<LabeledTextField>>,
    /// Java private `ltfSizeInY`.
    ltf_size_in_y: OnceCell<Rc<LabeledTextField>>,
    /// Java private `ltfShiftInX`.
    ltf_shift_in_x: OnceCell<Rc<LabeledTextField>>,
    /// Java private `ltfShiftInY`.
    ltf_shift_in_y: OnceCell<Rc<LabeledTextField>>,
    /// Java private final `cbLocalFits`.
    cb_local_fits: Rc<CheckBox>,
    /// Java private `ltfMidasLimit`.
    ltf_midas_limit: Rc<LabeledTextField>,
    /// Java private `lblMidasLimit = new JLabel("pixels if bigger.")`.
    lbl_midas_limit: Rc<JComponent>,
    /// Java private `cbsAlignmentRefSection`.
    cbs_alignment_ref_section: OnceCell<Rc<CheckBoxSpinner>>,
    /// Java private `spinDensityRefSection`.
    spin_density_ref_section: OnceCell<Rc<LabeledSpinner>>,
    /// Java private `spinTrialBinning`.
    spin_trial_binning: OnceCell<Rc<LabeledSpinner>>,
    /// Java private `spinRejoinTrialBinning`.
    spin_rejoin_trial_binning: OnceCell<Rc<LabeledSpinner>>,
    /// Java private `spinUseEveryNSlices`.
    spin_use_every_n_slices: OnceCell<Rc<LabeledSpinner>>,
    /// Java private `spinRejoinUseEveryNSlices`.
    spin_rejoin_use_every_n_slices: OnceCell<Rc<LabeledSpinner>>,
    /// Java private final `autoAlignmentPanel`.
    auto_alignment_panel: OnceCell<Rc<AutoAlignmentPanel>>,

    // state
    /// Java private `numSections`, initially 0.
    num_sections: Cell<i32>,
    /// Java private `curTab`, initially `Tab.SETUP`.
    cur_tab: Cell<Tab>,
    /// Java private `invalidReason`, initially null.
    invalid_reason: RefCell<Option<String>>,

    /// Java private `joinActionListener = new JoinActionListener(this)`.
    join_action_listener: ActionListener,
    /// Java private `workingDirActionListener = new WorkingDirActionListener(this)`.
    working_dir_action_listener: ActionListener,
    /// Java private final `axisID`.
    axis_id: AxisID,
    /// Java private final `manager`.
    manager: &'static JoinManager,

    /// Java private `defaultXSize`, initially 0.
    default_x_size: Cell<i32>,
    /// Java private `defaultYSize`, initially 0.
    default_y_size: Cell<i32>,

    /// Java private final `cbRefineWithTrial`.
    cb_refine_with_trial: Rc<CheckBox>,
    /// Java private final `btnRefineJoin`.
    btn_refine_join: Rc<MultiLineButton>,
    /// Java private final `btnMakeRefiningModel`.
    btn_make_refining_model: Rc<Run3dmodButton>,
    /// Java private final `btnXfjointomo`.
    btn_xfjointomo: Rc<MultiLineButton>,

    /// Java private final `boundaryTable`.
    boundary_table: OnceCell<Rc<BoundaryTable>>,
    /// Java private `refiningJoin`, initially false.
    refining_join: Cell<bool>,
    /// Java private `pnlTransformations`, initially null.
    pnl_transformations: RefCell<Option<Rc<SpacedPanel>>>,
    /// Java private final `tcModel = TransformChooserPanel.getJoinModelInstance()`.
    tc_model: Rc<TransformChooserPanel>,
    /// Java private final `ltfBoundariesToAnalyze`.
    ltf_boundaries_to_analyze: Rc<LabeledTextField>,
    /// Java private final `ltfObjectsToInclude`.
    ltf_objects_to_include: Rc<LabeledTextField>,
    /// Java private final `ltfGapStart`.
    ltf_gap_start: Rc<LabeledTextField>,
    /// Java private final `ltfGapEnd`.
    ltf_gap_end: Rc<LabeledTextField>,
    /// Java private final `ltfGapInc`.
    ltf_gap_inc: Rc<LabeledTextField>,
    /// Java private final `ltfPointsToFitMin`.
    ltf_points_to_fit_min: Rc<LabeledTextField>,
    /// Java private final `ltfPointsToFitMax`.
    ltf_points_to_fit_max: Rc<LabeledTextField>,
    /// Java private `pnlTables`, initially null.
    pnl_tables: RefCell<Option<Rc<JComponent>>>,
    /// Java private `pnlRejoin`, initially null.
    pnl_rejoin: RefCell<Option<Rc<JComponent>>>,

    /// Java private `ltfTransformedModel`.
    ltf_transformed_model: RefCell<Option<Rc<LabeledTextField>>>,
    /// Java private final `b3bOpenRejoinWithModel`.
    b3b_open_rejoin_with_model: Rc<BinnedXY3dmodButton>,
    /// Java private final `btnTransformModel`.
    btn_transform_model: Rc<Run3dmodButton>,
    /// Java private final `state`.
    state: &'static JoinState,
    /// Java private final `btnTransformAndViewModel`.
    btn_transform_and_view_model: Rc<MultiLineButton>,
    /// Java private final `btnOpenSampleAverages`.
    btn_open_sample_averages: Rc<Run3dmodButton>,
    /// Java private final `btnMakeSamples`.
    btn_make_samples: Rc<Run3dmodButton>,
    /// Java private final `pnlModelTab = SpacedPanel.getFocusableInstance(true)`.
    pnl_model_tab: Rc<SpacedPanel>,
    /// Java private final `pnlRejoinTab = new EtomoPanel()`.
    pnl_rejoin_tab: Rc<EtomoPanel>,
    /// Java private final `b3bOpenTrialIn3dmod`.
    b3b_open_trial_in_3dmod: Rc<BinnedXY3dmodButton>,
    /// Java private final `btnTrialJoin`.
    btn_trial_join: Rc<Run3dmodButton>,
    /// Java private final `ftfModelFile`.
    ftf_model_file: Rc<FileTextField>,
    /// Java private final `cbGap`.
    cb_gap: Rc<CheckBox>,
    /// Java private final `b3bOpenIn3dmod`.
    b3b_open_in_3dmod: Rc<BinnedXY3dmodButton>,
    /// Java private final `btnFinishJoin`.
    btn_finish_join: Rc<Run3dmodButton>,
    /// Java private final `b3bOpenRejoin`.
    b3b_open_rejoin: Rc<BinnedXY3dmodButton>,
    /// Java private final `btnRejoin`.
    btn_rejoin: Rc<Run3dmodButton>,
    /// Java private final `b3bOpenTrialRejoin`.
    b3b_open_trial_rejoin: Rc<BinnedXY3dmodButton>,
    /// Java private final `btnTrialRejoin`.
    btn_trial_rejoin: Rc<Run3dmodButton>,
    /// Rust-only: Java `this`.
    self_ref: Weak<JoinDialog>,
}

impl JoinDialog {
    /// Java private `JoinDialog(JoinManager, String, ConstJoinMetaData, JoinState)`.
    /// Create JoinDialog with workingDirName equal to the location of the .ejf file.
    fn new(
        manager: &'static JoinManager,
        working_dir_name: Option<&str>,
        meta_data: &dyn ConstJoinMetaData,
        state: &'static JoinState,
    ) -> Rc<JoinDialog> {
        let this = Rc::new_cyclic(|self_ref: &Weak<JoinDialog>| {
            let container: Weak<dyn Run3dmodButtonContainer> = self_ref.clone();
            // Java `new JoinActionListener(this)`.
            let adaptee = self_ref.clone();
            let join_action_listener: ActionListener = Rc::new(move |event: &ActionEvent| {
                if let Some(adaptee) = adaptee.upgrade() {
                    adaptee.action(event.get_action_command().unwrap_or(""), None, None);
                }
            });
            // Java `new WorkingDirActionListener(this)`.
            let adaptee = self_ref.clone();
            let working_dir_action_listener: ActionListener =
                Rc::new(move |_event: &ActionEvent| {
                    if let Some(adaptee) = adaptee.upgrade() {
                        adaptee.working_dir_action();
                    }
                });
            JoinDialog {
                ltf_root_name: LabeledTextField::new_field_type_string(
                    FieldType::String,
                    Some("Root name for output file: "),
                ),
                root_panel: OnceCell::new(),
                tab_pane: OnceCell::new(),
                pnl_setup_tab: SpacedPanel::get_focusable_instance_void(),
                pnl_section_table: OnceCell::new(),
                pnl_align_tab: SpacedPanel::get_focusable_instance_void(),
                pnl_join_tab: SpacedPanel::get_focusable_instance_void(),
                setup_panel2: OnceCell::new(),
                align_panel1: OnceCell::new(),
                pnl_finish_join: OnceCell::new(),
                pnl_midas_limit: SpacedPanel::get_instance_void(),
                ftf_working_dir: OnceCell::new(),
                btn_open_sample: OnceCell::new(),
                btn_get_max_size: OnceCell::new(),
                btn_get_subarea: OnceCell::new(),
                btn_change_setup: OnceCell::new(),
                btn_revert_to_last_setup: OnceCell::new(),
                ltf_size_in_x: OnceCell::new(),
                ltf_size_in_y: OnceCell::new(),
                ltf_shift_in_x: OnceCell::new(),
                ltf_shift_in_y: OnceCell::new(),
                cb_local_fits: CheckBox::new_string(Some("Do local linear fits")),
                ltf_midas_limit: LabeledTextField::new_field_type_string(
                    FieldType::Integer,
                    Some("Squeeze samples to "),
                ),
                lbl_midas_limit: JComponent::new_label("pixels if bigger."),
                cbs_alignment_ref_section: OnceCell::new(),
                spin_density_ref_section: OnceCell::new(),
                spin_trial_binning: OnceCell::new(),
                spin_rejoin_trial_binning: OnceCell::new(),
                spin_use_every_n_slices: OnceCell::new(),
                spin_rejoin_use_every_n_slices: OnceCell::new(),
                auto_alignment_panel: OnceCell::new(),
                num_sections: Cell::new(0),
                cur_tab: Cell::new(Tab::Setup),
                invalid_reason: RefCell::new(None),
                join_action_listener,
                working_dir_action_listener,
                axis_id: AxisID::Only,
                manager,
                default_x_size: Cell::new(0),
                default_y_size: Cell::new(0),
                cb_refine_with_trial: CheckBox::new_string(Some("Refine with trial")),
                btn_refine_join: MultiLineButton::new_string(Some(REFINE_JOIN_TEXT)),
                btn_make_refining_model:
                    Run3dmodButton::get_3dmod_instance_string_run_3dmod_button_container(
                        Some("Make Refining Model"),
                        Some(container.clone()),
                    ),
                btn_xfjointomo: MultiLineButton::new_string(Some("Find Transformations")),
                boundary_table: OnceCell::new(),
                refining_join: Cell::new(false),
                pnl_transformations: RefCell::new(None),
                tc_model: TransformChooserPanel::get_join_model_instance(),
                ltf_boundaries_to_analyze: LabeledTextField::new_field_type_string(
                    FieldType::IntegerList,
                    Some("Boundaries to analyze: "),
                ),
                ltf_objects_to_include: LabeledTextField::new_field_type_string(
                    FieldType::IntegerList,
                    Some("Objects to include: "),
                ),
                ltf_gap_start: LabeledTextField::new_field_type_string(
                    FieldType::FloatingPoint,
                    Some("Start: "),
                ),
                ltf_gap_end: LabeledTextField::new_field_type_string(
                    FieldType::FloatingPoint,
                    Some("End: "),
                ),
                ltf_gap_inc: LabeledTextField::new_field_type_string(
                    FieldType::FloatingPoint,
                    Some("Increment: "),
                ),
                ltf_points_to_fit_min: LabeledTextField::new_field_type_string(
                    FieldType::Integer,
                    Some("Min: "),
                ),
                ltf_points_to_fit_max: LabeledTextField::new_field_type_string(
                    FieldType::Integer,
                    Some("Max: "),
                ),
                pnl_tables: RefCell::new(None),
                pnl_rejoin: RefCell::new(None),
                ltf_transformed_model: RefCell::new(None),
                b3b_open_rejoin_with_model: BinnedXY3dmodButton::new(
                    Some("Open Rejoin with Transformed Model"),
                    Some(container.clone()),
                ),
                btn_transform_model:
                    Run3dmodButton::get_deferred_3dmod_instance_string_run_3dmod_button_container(
                        Some("Transform Model"),
                        Some(container.clone()),
                    ),
                state,
                btn_transform_and_view_model: MultiLineButton::new_string(Some(
                    "Transform & View Model",
                )),
                btn_open_sample_averages:
                    Run3dmodButton::get_3dmod_instance_string_run_3dmod_button_container(
                        Some("Open Sample Averages in 3dmod"),
                        Some(container.clone()),
                    ),
                btn_make_samples:
                    Run3dmodButton::get_deferred_3dmod_instance_string_run_3dmod_button_container_string(
                        Some("Make Samples"),
                        Some(container.clone()),
                        Some("averages in 3dmod"),
                    ),
                pnl_model_tab: SpacedPanel::get_focusable_instance_boolean(true),
                pnl_rejoin_tab: EtomoPanel::new(),
                b3b_open_trial_in_3dmod: BinnedXY3dmodButton::new(
                    Some("Open Trial in 3dmod"),
                    Some(container.clone()),
                ),
                btn_trial_join:
                    Run3dmodButton::get_deferred_3dmod_instance_string_run_3dmod_button_container(
                        Some(TRIAL_JOIN_TEXT),
                        Some(container.clone()),
                    ),
                ftf_model_file: FileTextField::new("Model file: "),
                cb_gap: CheckBox::new_string(Some("Try gaps: ")),
                b3b_open_in_3dmod: BinnedXY3dmodButton::new(
                    Some(OPEN_IN_3DMOD),
                    Some(container.clone()),
                ),
                btn_finish_join:
                    Run3dmodButton::get_deferred_3dmod_instance_string_run_3dmod_button_container(
                        Some(FINISH_JOIN_TEXT),
                        Some(container.clone()),
                    ),
                b3b_open_rejoin: BinnedXY3dmodButton::new(
                    Some("Open Rejoin in 3dmod"),
                    Some(container.clone()),
                ),
                btn_rejoin:
                    Run3dmodButton::get_deferred_3dmod_instance_string_run_3dmod_button_container(
                        Some(REJOIN_TEXT),
                        Some(container.clone()),
                    ),
                b3b_open_trial_rejoin: BinnedXY3dmodButton::new(
                    Some("Open Trial Rejoin in 3dmod"),
                    Some(container.clone()),
                ),
                btn_trial_rejoin:
                    Run3dmodButton::get_deferred_3dmod_instance_string_run_3dmod_button_container(
                        Some(TRIAL_REJOIN_TEXT),
                        Some(container),
                    ),
                self_ref: self_ref.clone(),
            }
        });
        // Constructor body.
        eprintln!(
            "{}\nDialog: {}",
            utilities::get_date_time_stamp(),
            DialogType::Join
        );
        let _ = this.boundary_table.set(BoundaryTable::new(manager, &this));
        let _ = this
            .auto_alignment_panel
            .set(AutoAlignmentPanel::get_join_instance(manager));
        this.set_refining_join_void();
        this.create_root_panel(working_dir_name);
        this.set_meta_data(meta_data);
        ui_harness::with(|harness| harness.pack_base_manager(Some(manager)));
        this.set_screen_state(manager.get_screen_state());
        this.init();
        this.set_tool_tip_text();
        this
    }

    // Field reads of the fields the constructor body and the create methods assign.
    fn pnl_section_table(&self) -> &Rc<SectionTablePanel> {
        self.pnl_section_table.get().expect("pnlSectionTable")
    }
    fn tab_pane(&self) -> &Rc<TabbedPane> {
        self.tab_pane.get().expect("tabPane")
    }
    fn ftf_working_dir(&self) -> &Rc<FileTextField> {
        self.ftf_working_dir.get().expect("ftfWorkingDir")
    }
    fn btn_change_setup(&self) -> &Rc<MultiLineButton> {
        self.btn_change_setup.get().expect("btnChangeSetup")
    }
    fn btn_revert_to_last_setup(&self) -> &Rc<MultiLineButton> {
        self.btn_revert_to_last_setup
            .get()
            .expect("btnRevertToLastSetup")
    }
    fn btn_open_sample(&self) -> &Rc<Run3dmodButton> {
        self.btn_open_sample.get().expect("btnOpenSample")
    }
    fn btn_get_max_size(&self) -> &Rc<MultiLineButton> {
        self.btn_get_max_size.get().expect("btnGetMaxSize")
    }
    fn btn_get_subarea(&self) -> &Rc<MultiLineButton> {
        self.btn_get_subarea.get().expect("btnGetSubarea")
    }
    fn ltf_size_in_x(&self) -> &Rc<LabeledTextField> {
        self.ltf_size_in_x.get().expect("ltfSizeInX")
    }
    fn ltf_size_in_y(&self) -> &Rc<LabeledTextField> {
        self.ltf_size_in_y.get().expect("ltfSizeInY")
    }
    fn ltf_shift_in_x(&self) -> &Rc<LabeledTextField> {
        self.ltf_shift_in_x.get().expect("ltfShiftInX")
    }
    fn ltf_shift_in_y(&self) -> &Rc<LabeledTextField> {
        self.ltf_shift_in_y.get().expect("ltfShiftInY")
    }
    fn cbs_alignment_ref_section(&self) -> &Rc<CheckBoxSpinner> {
        self.cbs_alignment_ref_section
            .get()
            .expect("cbsAlignmentRefSection")
    }
    fn spin_density_ref_section(&self) -> &Rc<LabeledSpinner> {
        self.spin_density_ref_section
            .get()
            .expect("spinDensityRefSection")
    }
    fn spin_trial_binning(&self) -> &Rc<LabeledSpinner> {
        self.spin_trial_binning.get().expect("spinTrialBinning")
    }
    fn spin_rejoin_trial_binning(&self) -> &Rc<LabeledSpinner> {
        self.spin_rejoin_trial_binning
            .get()
            .expect("spinRejoinTrialBinning")
    }
    fn spin_use_every_n_slices(&self) -> &Rc<LabeledSpinner> {
        self.spin_use_every_n_slices
            .get()
            .expect("spinUseEveryNSlices")
    }
    fn spin_rejoin_use_every_n_slices(&self) -> &Rc<LabeledSpinner> {
        self.spin_rejoin_use_every_n_slices
            .get()
            .expect("spinRejoinUseEveryNSlices")
    }
    fn auto_alignment_panel(&self) -> &Rc<AutoAlignmentPanel> {
        self.auto_alignment_panel.get().expect("autoAlignmentPanel")
    }
    fn boundary_table(&self) -> &Rc<BoundaryTable> {
        self.boundary_table.get().expect("boundaryTable")
    }

    /// Java `getFocusComponent()`.
    pub fn get_focus_component(&self) -> Option<Rc<JComponent>> {
        if self.is_setup_tab() {
            return Some(self.pnl_setup_tab.get_container());
        }
        if self.is_align_tab() {
            return Some(self.pnl_align_tab.get_container());
        }
        if self.is_join_tab() {
            return Some(self.pnl_join_tab.get_container());
        }
        if self.is_model_tab() {
            return Some(self.pnl_model_tab.get_container());
        }
        if self.is_rejoin_tab() {
            return Some(self.pnl_rejoin_tab.get_component());
        }
        None
    }

    /// Java `setAutoAlignmentController(AutoAlignmentController)`.
    pub fn set_auto_alignment_controller(
        &self,
        auto_alignment_controller: &'static AutoAlignmentController,
    ) {
        self.auto_alignment_panel()
            .set_controller(auto_alignment_controller);
    }

    /// Java static `getInstance(JoinManager, ConstJoinMetaData, JoinState)`.  Create
    /// JoinDialog without an .ejf file.
    pub fn get_instance_meta_data(
        manager: &'static JoinManager,
        meta_data: &dyn ConstJoinMetaData,
        state: &'static JoinState,
    ) -> Rc<JoinDialog> {
        let instance = JoinDialog::new(manager, None, meta_data, state);
        instance.add_listeners();
        instance
    }

    /// Java static `getInstance(JoinManager, String, ConstJoinMetaData, JoinState)`.
    /// Create JoinDialog with workingDirName equal to the location of the .ejf file.
    pub fn get_instance_working_dir(
        manager: &'static JoinManager,
        working_dir_name: Option<&str>,
        meta_data: &dyn ConstJoinMetaData,
        state: &'static JoinState,
    ) -> Rc<JoinDialog> {
        let instance = JoinDialog::new(manager, working_dir_name, meta_data, state);
        instance.add_listeners();
        instance
    }

    /// Java package-private `paramString()`.
    pub fn param_string(&self) -> String {
        format!(
            "ltfRootName={},ltfSizeInX={},\nltfSizeInY={},ltfShiftInX={},\nltfShiftInY={},\
             ltfMidasLimit={},\nspinDensityRefSection={},\nspinTrialBinning={},\n\
             spinUseEveryNSlices={},\nnumSections{},curTab={:?},invalidReason{},\naxisID={},null",
            self.ltf_root_name.get_text_void().unwrap_or_default(),
            self.ltf_size_in_x().get_text_void().unwrap_or_default(),
            self.ltf_size_in_y().get_text_void().unwrap_or_default(),
            self.ltf_shift_in_x().get_text_void().unwrap_or_default(),
            self.ltf_shift_in_y().get_text_void().unwrap_or_default(),
            self.ltf_midas_limit.get_text_void().unwrap_or_default(),
            self.spin_density_ref_section().get_value(),
            self.spin_trial_binning().get_value(),
            self.spin_use_every_n_slices().get_value(),
            self.num_sections.get(),
            self.cur_tab.get(),
            self.invalid_reason
                .borrow()
                .clone()
                .unwrap_or_else(|| "null".to_string()),
            self.axis_id
        )
    }

    /// Java package-private `getModelTabJComponent()`.
    pub fn get_model_tab_j_component(&self) -> Rc<JComponent> {
        self.pnl_model_tab.get_j_panel()
    }

    /// Java package-private `getRejoinTabJComponent()`.
    pub fn get_rejoin_tab_j_component(&self) -> Rc<JComponent> {
        self.pnl_rejoin_tab.get_component()
    }

    /// Java package-private `getSetupTabJComponent()`.
    pub fn get_setup_tab_j_component(&self) -> Rc<JComponent> {
        self.pnl_setup_tab.get_j_panel()
    }

    /// Java package-private `getAlignTabJComponent()`.
    pub fn get_align_tab_j_component(&self) -> Rc<JComponent> {
        self.pnl_align_tab.get_j_panel()
    }

    /// Java package-private `getJoinTabJComponent()`.
    pub fn get_join_tab_j_component(&self) -> Rc<JComponent> {
        self.pnl_join_tab.get_j_panel()
    }

    /// Java private `createRootPanel(String)`.  Create the root panel.
    fn create_root_panel(&self, working_dir_name: Option<&str>) {
        let root_panel = JComponent::new_panel();
        let _ = self.root_panel.set(root_panel.clone());
        self.create_tab_pane(working_dir_name);
        root_panel.add(&self.tab_pane().get_component());
    }

    /// Java private `addListeners()`.
    fn add_listeners(&self) {
        let context_menu: Weak<dyn ContextMenu> = self.self_ref.clone();
        self.tab_pane()
            .get_component()
            .add_mouse_listener(GenericMouseAdapter::new(context_menu));
        // Java `new TabChangeListener(this)`.
        let adaptee = self.self_ref.clone();
        let tab_change_listener: ChangeListener = Rc::new(move |event: &ChangeEvent| {
            if let Some(adaptee) = adaptee.upgrade() {
                adaptee.change_tab(event);
            }
        });
        self.tab_pane()
            .get_component()
            .add_change_listener(tab_change_listener);
        self.ftf_working_dir()
            .add_action_listener(self.working_dir_action_listener.clone());
        self.btn_change_setup()
            .add_action_listener(self.join_action_listener.clone());
        self.btn_revert_to_last_setup()
            .add_action_listener(self.join_action_listener.clone());
        self.btn_make_samples
            .add_action_listener(self.join_action_listener.clone());
        self.btn_open_sample()
            .add_action_listener(self.join_action_listener.clone());
        self.btn_open_sample_averages
            .add_action_listener(self.join_action_listener.clone());

        self.btn_get_max_size()
            .add_action_listener(self.join_action_listener.clone());
        self.btn_finish_join
            .add_action_listener(self.join_action_listener.clone());
        self.b3b_open_in_3dmod
            .add_action_listener(self.join_action_listener.clone());
        self.btn_trial_join
            .add_action_listener(self.join_action_listener.clone());
        self.b3b_open_trial_in_3dmod
            .add_action_listener(self.join_action_listener.clone());
        self.btn_get_subarea()
            .add_action_listener(self.join_action_listener.clone());
        self.btn_refine_join
            .add_action_listener(self.join_action_listener.clone());
        self.btn_make_refining_model
            .add_action_listener(self.join_action_listener.clone());
        self.btn_xfjointomo
            .add_action_listener(self.join_action_listener.clone());
        self.btn_transform_and_view_model
            .add_action_listener(self.join_action_listener.clone());
        self.btn_rejoin
            .add_action_listener(self.join_action_listener.clone());
        self.btn_trial_rejoin
            .add_action_listener(self.join_action_listener.clone());
        // Java `new ModelFileActionListener(this)`.
        let adaptee = self.self_ref.clone();
        self.ftf_model_file
            .add_action_listener(Rc::new(move |_event: &ActionEvent| {
                if let Some(adaptee) = adaptee.upgrade() {
                    adaptee.model_file_action();
                }
            }));
        self.cb_gap
            .add_action_listener(Some(self.join_action_listener.clone()));
        self.b3b_open_rejoin
            .add_action_listener(self.join_action_listener.clone());
        self.b3b_open_trial_rejoin
            .add_action_listener(self.join_action_listener.clone());
        self.b3b_open_rejoin_with_model
            .add_action_listener(self.join_action_listener.clone());
        self.btn_transform_model
            .add_action_listener(self.join_action_listener.clone());
        self.cb_local_fits
            .add_action_listener(Some(self.join_action_listener.clone()));
        self.cbs_alignment_ref_section()
            .add_action_listener(Some(self.join_action_listener.clone()));
    }

    /// Java private `createTabPane(String)`.  Create the tabbed pane.
    fn create_tab_pane(&self, working_dir_name: Option<&str>) {
        let tab_pane = TabbedPane::new();
        let _ = self.tab_pane.set(tab_pane.clone());
        // tabPane.addMouseListener(new GenericMouseAdapter(this));
        // TabChangeListener tabChangeListener = new TabChangeListener(this);
        // tabPane.addChangeListener(tabChangeListener);
        self.create_setup_panel(working_dir_name);
        tab_pane.add_tab_string_spaced_panel("Setup", &self.pnl_setup_tab);
        self.create_align_panel();
        tab_pane.add_tab_string_spaced_panel("Align", &self.pnl_align_tab);
        self.create_join_panel();
        tab_pane.add_tab_string_spaced_panel("Join", &self.pnl_join_tab);
        self.create_model_panel();
        tab_pane.add_tab_string_spaced_panel("Model", &self.pnl_model_tab);
        self.create_rejoin_panel();
        tab_pane.add_tab_string_component("Rejoin", &self.pnl_rejoin_tab.get_component());
        self.add_panel_components(Tab::Setup);
        self.update_display();
    }

    /// Java `updateDisplay()`.
    pub fn update_display(&self) {
        let tab_pane = self.tab_pane().get_component();
        // must be refining join to have access to model and rejoin tabs
        tab_pane.set_enabled_at(Tab::Model.get_index() as usize, self.refining_join.get());
        tab_pane.set_enabled_at(Tab::Rejoin.get_index() as usize, self.refining_join.get());
        // control gap text fields with checkbox
        let enable = self.cb_gap.is_selected();
        let integer_width = UIParameters::get_instance_void().get_integer_width() as f64;
        self.ltf_gap_start.set_enabled(enable);
        self.ltf_gap_start.set_text_preferred_width(integer_width);
        self.ltf_gap_end.set_enabled(enable);
        self.ltf_gap_end.set_text_preferred_width(integer_width);
        self.ltf_gap_inc.set_enabled(enable);
        self.ltf_gap_inc.set_text_preferred_width(integer_width);
        // .join file must exist before it can be opened
        let enable = dataset_files::get_join_file(false, self.manager).exists();
        self.b3b_open_rejoin_with_model.set_enabled(enable);
        self.b3b_open_rejoin.set_enabled(enable);
        // Need boundary list to transform model
        let enable = !self.state.is_refine_start_list_empty();
        self.btn_transform_model.set_enabled(enable);
        // Can't only refine if a join exists
        // A trial join is only refinable if it was created with all slices
        let trial_join_refinable = dataset_files::get_join_file(true, self.manager).exists()
            && self.state.get_join_trial_use_every_n_slices().equals_int(1);
        self.cb_refine_with_trial.set_enabled(trial_join_refinable);
        if !trial_join_refinable {
            self.cb_refine_with_trial.set_selected_boolean(false);
        }
        self.btn_refine_join.set_enabled(
            dataset_files::get_join_file(false, self.manager).exists() || trial_join_refinable,
        );
        // If local fitting and a reference section are both selected, the local fitting
        // has no effect. Reference section has priority.
        self.cb_local_fits
            .set_enabled(!self.cbs_alignment_ref_section().is_selected());
        self.cbs_alignment_ref_section()
            .set_enabled(!self.cb_local_fits.is_selected() || !self.cb_local_fits.is_enabled());
    }

    /// Java private `addPanelComponents(Tab)`.  Add components to the current tab.
    fn add_panel_components(&self, tab: Tab) -> Option<Rc<JComponent>> {
        match tab {
            Tab::Setup => {
                self.add_setup_panel_components();
                Some(self.pnl_setup_tab.get_container())
            }
            Tab::Align => {
                self.add_align_panel_components();
                Some(self.pnl_align_tab.get_container())
            }
            Tab::Join => {
                self.add_join_panel_components();
                Some(self.pnl_join_tab.get_container())
            }
            Tab::Model => {
                self.add_model_panel_components();
                Some(self.pnl_model_tab.get_container())
            }
            Tab::Rejoin => {
                self.add_rejoin_panel_components();
                Some(self.pnl_rejoin_tab.get_component())
            }
        }
    }

    /// Java private `setSizeAndShift(Vector) throws NullRequiredNumberException`.
    /// Get rubberband coordinates and calculate a new size and shift based on the last
    /// finishjoin trial.
    fn set_size_and_shift(
        &self,
        coordinates: Option<&[String]>,
    ) -> Result<(), NullRequiredNumberException> {
        let Some(coordinates) = coordinates else {
            return Ok(());
        };
        let size = coordinates.len();
        if size == 0 {
            return Ok(());
        }
        let mut index = 0;
        let mut est_x_min = EtomoNumber::new_with_type(Some(Type::Integer));
        let mut est_y_min = EtomoNumber::new_with_type(Some(Type::Integer));
        let mut est_x_max = EtomoNumber::new_with_type(Some(Type::Integer));
        let mut est_y_max = EtomoNumber::new_with_type(Some(Type::Integer));
        while index < size {
            let line = &coordinates[index];
            index += 1;
            if imod_process::RUBBERBAND_RESULTS_STRING == line {
                est_x_min.set_string(Some(&coordinates[index]));
                index += 1;
                if index >= size {
                    break;
                }
                est_y_min.set_string(Some(&coordinates[index]));
                index += 1;
                if index >= size {
                    break;
                }
                est_x_max.set_string(Some(&coordinates[index]));
                index += 1;
                if index >= size {
                    break;
                }
                est_y_max.set_string(Some(&coordinates[index]));
                index += 1;
            }
        }
        let meta_data = self.manager.get_const_meta_data();
        if !est_x_min.is_null() && !est_x_max.is_null() {
            let min = meta_data.get_coordinate(&est_x_min, self.state)?;
            let max = meta_data.get_coordinate(&est_x_max, self.state)?;
            self.ltf_size_in_x()
                .set_text_int(JoinMetaData::get_size(min, max));
            self.ltf_shift_in_x()
                .set_text_int(self.state.get_new_shift_in_x(min, max)?);
        }
        if !est_y_min.is_null() && !est_y_max.is_null() {
            let min = meta_data.get_coordinate(&est_y_min, self.state)?;
            let max = meta_data.get_coordinate(&est_y_max, self.state)?;
            self.ltf_size_in_y()
                .set_text_int(JoinMetaData::get_size(min, max));
            self.ltf_shift_in_y()
                .set_text_int(self.state.get_new_shift_in_y(min, max)?);
        }
        Ok(())
    }

    /// Java private `removePanelComponents(Tab)`.
    fn remove_panel_components(&self, tab: Tab) {
        match tab {
            Tab::Setup => self.pnl_setup_tab.remove_all(),
            Tab::Align => self.pnl_align_tab.remove_all(),
            Tab::Join => self.pnl_join_tab.remove_all(),
            Tab::Model => self.pnl_model_tab.remove_all(),
            Tab::Rejoin => self.pnl_rejoin_tab.get_component().remove_all(),
        }
    }

    /// Java package-private final `isSetupTab()`.
    pub fn is_setup_tab(&self) -> bool {
        self.cur_tab.get() == Tab::Setup
    }

    /// Java package-private final `isAlignTab()`.
    pub fn is_align_tab(&self) -> bool {
        self.cur_tab.get() == Tab::Align
    }

    /// Java package-private final `isJoinTab()`.
    pub fn is_join_tab(&self) -> bool {
        self.cur_tab.get() == Tab::Join
    }

    /// Java package-private final `isModelTab()`.
    pub fn is_model_tab(&self) -> bool {
        self.cur_tab.get() == Tab::Model
    }

    /// Java package-private final `isRejoinTab()`.
    pub fn is_rejoin_tab(&self) -> bool {
        self.cur_tab.get() == Tab::Rejoin
    }

    /// Java package-private `getTab()`.
    pub fn get_tab(&self) -> Tab {
        self.cur_tab.get()
    }

    /// Java package-private `getSectionTableSize()`.
    pub fn get_section_table_size(&self) -> i32 {
        self.pnl_section_table().size()
    }

    /// Java private final `changeTab(ChangeEvent)`.
    fn change_tab(&self, _event: &ChangeEvent) {
        let prev_tab = self.cur_tab.get();
        self.remove_panel_components(prev_tab);
        self.cur_tab.set(Tab::get_instance(
            self.tab_pane().get_component().get_selected_tab(),
        ));
        self.synchronize_tab(prev_tab);
        // Java's unused local `displayedComponent`.
        let _displayed_component = self.add_panel_components(self.cur_tab.get());
        ui_harness::with(|harness| harness.pack_base_manager(Some(self.manager)));
        ui_harness::with(|harness| harness.move_sub_frame());
    }

    /// Java `setInverted() throws FileException, IOException`.
    pub fn set_inverted(&self) -> Result<(), LogFileError> {
        self.pnl_section_table().set_inverted()
    }

    /// Java public final `setMode(int)`.  The source's `IllegalStateException` for an
    /// unknown mode is a panic.
    pub fn set_mode(&self, mode: i32) {
        if mode == SETUP_MODE {
            self.ftf_working_dir().set_editable(true);
            self.ltf_root_name.set_editable(true);
        } else {
            self.ftf_working_dir().set_editable(false);
            self.ltf_root_name.set_editable(false);
        }
        let tab_pane = self.tab_pane().get_component();
        match mode {
            SETUP_MODE | SAMPLE_NOT_PRODUCED_MODE => {
                tab_pane.set_enabled_at(1, false);
                tab_pane.set_enabled_at(2, false);
                self.ltf_midas_limit.set_enabled(true);
                self.lbl_midas_limit.set_enabled(true);
                self.spin_density_ref_section().set_enabled(true);
                self.btn_change_setup().set_enabled(false);
                self.set_revert_state(false);
                self.btn_make_samples.set_enabled(true);
            }
            SAMPLE_PRODUCED_MODE => {
                tab_pane.set_enabled_at(1, true);
                tab_pane.set_enabled_at(2, true);
                self.ltf_midas_limit.set_enabled(false);
                self.lbl_midas_limit.set_enabled(false);
                self.spin_density_ref_section().set_enabled(false);
                self.btn_change_setup().set_enabled(true);
                self.set_revert_state(false);
                self.btn_make_samples.set_enabled(false);
            }
            CHANGING_SAMPLE_MODE => {
                tab_pane.set_enabled_at(1, false);
                tab_pane.set_enabled_at(2, false);
                self.ltf_midas_limit.set_enabled(true);
                self.lbl_midas_limit.set_enabled(true);
                self.spin_density_ref_section().set_enabled(true);
                self.btn_change_setup().set_enabled(false);
                self.set_revert_state(true);
                self.btn_make_samples.set_enabled(true);
            }
            _ => panic!("java.lang.IllegalStateException: mode={mode}"),
        }
        self.pnl_section_table().set_mode(mode);
    }

    /// Java private `setRevertState(boolean)`.
    fn set_revert_state(&self, enable_revert: bool) {
        self.btn_revert_to_last_setup().set_enabled(enable_revert);
        self.state.set_revert_state(enable_revert);
    }

    /// Java private `createSetupPanel(String)`.
    fn create_setup_panel(&self, working_dir_name: Option<&str>) {
        self.pnl_setup_tab.set_box_layout(spaced_panel::Y_AXIS);
        // first component
        let ftf_working_dir = FileTextField::new(&format!("{}: ", WORKING_DIRECTORY_TEXT));
        ftf_working_dir.set_text_string(working_dir_name);
        let _ = self.ftf_working_dir.set(ftf_working_dir);
        // third component
        let this = self.self_ref.upgrade().expect("JoinDialog");
        let _ = self
            .pnl_section_table
            .set(SectionTablePanel::new(&this, self.manager, self.state));
        // midas limit panel
        self.pnl_midas_limit.set_box_layout(spaced_panel::X_AXIS);
        self.pnl_midas_limit
            .add_container(&self.ltf_midas_limit.get_container());
        self.pnl_midas_limit.add_j_label(&self.lbl_midas_limit);
        // fifth component
        let num_sections = self.num_sections.get();
        let _ =
            self.spin_density_ref_section
                .set(LabeledSpinner::get_instance_string_int_int_int_int(
                    Some("Reference section for density matching: "),
                    1,
                    1,
                    if num_sections < 1 { 1 } else { num_sections },
                    1,
                ));
        // sixth component
        let setup_panel2 = SpacedPanel::get_instance_void();
        setup_panel2.set_box_layout(spaced_panel::X_AXIS);
        let btn_change_setup = MultiLineButton::new_string(Some("Change Setup"));
        // btnChangeSetup.addActionListener(joinActionListener);
        setup_panel2.add_multi_line_button(&btn_change_setup);
        let _ = self.btn_change_setup.set(btn_change_setup);
        let btn_revert_to_last_setup = MultiLineButton::new_string(Some("Revert to Last Setup"));
        let _ = self
            .btn_revert_to_last_setup
            .set(btn_revert_to_last_setup.clone());
        self.set_revert_state(true);
        // btnRevertToLastSetup.addActionListener(joinActionListener);
        setup_panel2.add_multi_line_button(&btn_revert_to_last_setup);
        let _ = self.setup_panel2.set(setup_panel2);
        // seventh component
        self.btn_make_samples
            .set_deferred_3dmod_button_deferred_3dmod_button(Some(
                self.btn_open_sample_averages.clone() as Rc<dyn Deferred3dmodButton>,
            ));
        // Swing layout: btnMakeSamples.setAlignmentX(CENTER_ALIGNMENT).
    }

    /// Java private `addSetupPanelComponents()`.
    fn add_setup_panel_components(&self) {
        self.pnl_setup_tab
            .add_container(&self.ftf_working_dir().get_container());
        self.pnl_setup_tab
            .add_labeled_text_field(&self.ltf_root_name);
        self.pnl_setup_tab
            .add_j_panel(&self.pnl_section_table().get_root_panel());
        self.pnl_section_table().display_cur_tab();
        self.pnl_setup_tab.add_spaced_panel(&self.pnl_midas_limit);
        let pnl_density_ref_section = JComponent::new_panel();
        // Swing layout: BoxLayout X_AXIS, CENTER_ALIGNMENT, horizontal glue.
        pnl_density_ref_section.add(&self.spin_density_ref_section().get_container());
        self.pnl_setup_tab.add_j_panel(&pnl_density_ref_section);
        // Swing layout: Box.createRigidArea(FixedDim.x0_y5).
        self.pnl_setup_tab
            .add_spaced_panel(self.setup_panel2.get().expect("setupPanel2"));
        self.pnl_setup_tab
            .add_multi_line_button(&self.btn_make_samples);
    }

    /// Java private `createAlignPanel()`.
    fn create_align_panel(&self) {
        self.pnl_align_tab.set_box_layout(spaced_panel::Y_AXIS);
        // second component
        let align_panel1 = SpacedPanel::get_instance_void();
        align_panel1.set_box_layout(spaced_panel::X_AXIS);
        let container: Weak<dyn Run3dmodButtonContainer> = self.self_ref.clone();
        let btn_open_sample = Run3dmodButton::get_3dmod_instance_string_run_3dmod_button_container(
            Some("Open Sample in 3dmod"),
            Some(container),
        );
        align_panel1.add_multi_line_button(&btn_open_sample);
        let _ = self.btn_open_sample.set(btn_open_sample);
        align_panel1.add_multi_line_button(&self.btn_open_sample_averages);
        let _ = self.align_panel1.set(align_panel1);
        // fourth component
    }

    /// Java private `addAlignPanelComponents()`.
    fn add_align_panel_components(&self) {
        // first component
        self.pnl_align_tab
            .add_j_panel(&self.pnl_section_table().get_root_panel());
        self.pnl_section_table().display_cur_tab();
        // second component
        self.pnl_align_tab
            .add_spaced_panel(self.align_panel1.get().expect("alignPanel1"));
        // auto alignment
        self.pnl_align_tab
            .add_component(&self.auto_alignment_panel().get_root_component());
    }

    /// Java private `createModelPanel()`.
    fn create_model_panel(&self) {
        // create model panel only once
        if self.pnl_transformations.borrow().is_some() {
            return;
        }
        // construct panels
        let pnl_transformations = SpacedPanel::get_instance_void();
        *self.pnl_transformations.borrow_mut() = Some(pnl_transformations.clone());
        let pnl_gap_start_end_inc = SpacedPanel::get_instance_void();
        let pnl_points_to_fit = SpacedPanel::get_instance_void();
        // build panels
        self.pnl_model_tab.set_box_layout(spaced_panel::Y_AXIS);
        self.pnl_model_tab.align_components_x(0.5);
        // transformations panel
        pnl_transformations.set_box_layout(spaced_panel::Y_AXIS);
        pnl_transformations.set_border(&EtchedBorder::new(Some("Transformations")).get_border());
        pnl_transformations.add_component(&self.tc_model.get_component());
        pnl_transformations.add_container(&self.ltf_boundaries_to_analyze.get_container());
        pnl_transformations.add_container(&self.ltf_objects_to_include.get_container());
        pnl_transformations.add_spaced_panel(&pnl_gap_start_end_inc);
        pnl_transformations.add_spaced_panel(&pnl_points_to_fit);
        let pnl_xfjointomo = JComponent::new_panel();
        // Swing layout: BoxLayout X_AXIS, CENTER_ALIGNMENT, horizontal glue around.
        pnl_xfjointomo.add(&self.btn_xfjointomo.get_component());
        pnl_transformations.add_j_panel(&pnl_xfjointomo);
        // Swing layout: Box.createRigidArea(FixedDim.x0_y5).
        let pnl_transform_and_view_model = JComponent::new_panel();
        pnl_transform_and_view_model.add(&self.btn_transform_and_view_model.get_component());
        pnl_transformations.add_j_panel(&pnl_transform_and_view_model);
        // gap panel
        pnl_gap_start_end_inc.set_box_layout(spaced_panel::X_AXIS);
        pnl_gap_start_end_inc.add_check_box(&self.cb_gap);
        pnl_gap_start_end_inc.add_labeled_text_field(&self.ltf_gap_start);
        pnl_gap_start_end_inc.add_labeled_text_field(&self.ltf_gap_end);
        pnl_gap_start_end_inc.add_container(&self.ltf_gap_inc.get_container());
        // points to fit panel
        pnl_points_to_fit.set_box_layout(spaced_panel::X_AXIS);
        pnl_points_to_fit.add_j_label(&JComponent::new_label("Points to fit: "));
        pnl_points_to_fit.add_labeled_text_field(&self.ltf_points_to_fit_min);
        pnl_points_to_fit.add_labeled_text_field(&self.ltf_points_to_fit_max);
    }

    /// Java private `addModelPanelComponents()`.
    fn add_model_panel_components(&self) {
        self.create_model_panel();
        self.pnl_model_tab
            .add_multi_line_button(&self.btn_make_refining_model);
        self.pnl_model_tab
            .add_container(&self.boundary_table().get_container());
        self.boundary_table().display();
        let pnl_transformations = self.pnl_transformations.borrow().clone();
        if let Some(pnl_transformations) = pnl_transformations {
            self.pnl_model_tab.add_spaced_panel(&pnl_transformations);
        }
    }

    /// Java private `addRejoinPanelComponents()`.
    fn add_rejoin_panel_components(&self) {
        self.create_rejoin_panel();
        let pnl_tables = self.pnl_tables.borrow().clone().expect("pnlTables");
        let pnl_rejoin = self.pnl_rejoin.borrow().clone().expect("pnlRejoin");
        self.pnl_rejoin_tab.get_component().add(&pnl_tables);
        pnl_tables.add(&self.pnl_section_table().get_container());
        self.pnl_section_table().display_cur_tab();
        pnl_tables.add(&self.boundary_table().get_container());
        self.boundary_table().display();
        self.pnl_rejoin_tab.get_component().add(&pnl_rejoin);
    }

    /// Java private `createRejoinPanel()`.
    fn create_rejoin_panel(&self) {
        if self.pnl_tables.borrow().is_some() {
            return;
        }
        // Swing layout: pnlRejoinTab BoxLayout Y_AXIS.
        let pnl_tables = JComponent::new_panel();
        // Swing layout: pnlTables BoxLayout X_AXIS.
        *self.pnl_tables.borrow_mut() = Some(pnl_tables);
        let pnl_rejoin = JComponent::new_panel();
        // Swing layout: pnlRejoin BoxLayout Y_AXIS.
        *self.pnl_rejoin.borrow_mut() = Some(pnl_rejoin.clone());
        // use every spinner
        let pnl_use_every = SpacedPanel::get_instance_void();
        pnl_use_every.set_box_layout(spaced_panel::X_AXIS);
        let z_max = self.pnl_section_table().get_z_max();
        let current_value = if z_max < 1 {
            1
        } else if z_max < 10 {
            z_max
        } else {
            10
        };
        let spin_rejoin_use_every_n_slices = LabeledSpinner::get_instance_string_int_int_int_int(
            Some("Use every "),
            current_value,
            1,
            if z_max < 1 { 1 } else { z_max },
            1,
        );
        pnl_use_every.add_labeled_spinner(&spin_rejoin_use_every_n_slices);
        let _ = self
            .spin_rejoin_use_every_n_slices
            .set(spin_rejoin_use_every_n_slices);
        pnl_use_every.add_j_label(&JComponent::new_label("slices"));
        // trial rejoin button
        let pnl_trial_rejoin_button = SpacedPanel::get_instance_void();
        pnl_trial_rejoin_button.set_box_layout(spaced_panel::Y_AXIS);
        // Swing painting: BorderFactory.createEtchedBorder().
        pnl_trial_rejoin_button.add_container(&pnl_use_every.get_container());
        let spin_rejoin_trial_binning = LabeledSpinner::get_instance_string_int_int_int_int(
            Some("Binning in X and Y: "),
            1,
            1,
            50,
            1,
        );
        pnl_trial_rejoin_button.add_container(&spin_rejoin_trial_binning.get_container());
        let _ = self
            .spin_rejoin_trial_binning
            .set(spin_rejoin_trial_binning);
        self.btn_trial_rejoin
            .set_deferred_3dmod_button_binned_xy_3dmod_button(Some(&self.b3b_open_trial_rejoin));
        // Swing layout: btnTrialRejoin.setAlignmentX(CENTER_ALIGNMENT).
        pnl_trial_rejoin_button.add_multi_line_button(&self.btn_trial_rejoin);
        // trial rejoin buttons
        let pnl_trial_rejoin_buttons = EtomoPanel::new();
        // Swing layout: BoxLayout X_AXIS, horizontal glue.
        pnl_trial_rejoin_buttons.set_border(&EtchedBorder::new(Some("Trial Rejoin")).get_border());
        pnl_trial_rejoin_buttons
            .get_component()
            .add(&pnl_trial_rejoin_button.get_container());
        pnl_trial_rejoin_buttons
            .get_component()
            .add(&self.b3b_open_trial_rejoin.get_container());
        pnl_rejoin.add(&pnl_trial_rejoin_buttons.get_component());
        // rejoin buttons
        let pnl_rejoin_buttons = EtomoPanel::new();
        // Swing layout: BoxLayout X_AXIS, horizontal glue.
        pnl_rejoin_buttons.set_border(&EtchedBorder::new(Some("Final Rejoin")).get_border());
        pnl_rejoin_buttons
            .get_component()
            .add(&self.btn_rejoin.get_component());
        self.btn_rejoin
            .set_deferred_3dmod_button_binned_xy_3dmod_button(Some(&self.b3b_open_rejoin));
        pnl_rejoin_buttons
            .get_component()
            .add(&self.b3b_open_rejoin.get_container());
        pnl_rejoin.add(&pnl_rejoin_buttons.get_component());
        // transform model
        let pnl_transform_model = SpacedPanel::get_instance_void();
        pnl_transform_model.set_box_layout(spaced_panel::Y_AXIS);
        pnl_transform_model.set_border(&EtchedBorder::new(Some("Transform Model")).get_border());
        pnl_transform_model.set_component_alignment_x(0.5);
        pnl_transform_model.add_container(&self.ftf_model_file.get_container());
        let ltf_transformed_model =
            LabeledTextField::new_field_type_string(FieldType::String, Some("Output file: "));
        pnl_transform_model.add_container(&ltf_transformed_model.get_container());
        *self.ltf_transformed_model.borrow_mut() = Some(ltf_transformed_model);
        // transform model buttons
        let pnl_transform_model_buttons = JComponent::new_panel();
        // Swing layout: BoxLayout X_AXIS, horizontal glue.
        self.btn_transform_model
            .set_deferred_3dmod_button_binned_xy_3dmod_button(Some(
                &self.b3b_open_rejoin_with_model,
            ));
        pnl_transform_model_buttons.add(&self.btn_transform_model.get_component());
        pnl_transform_model_buttons.add(&self.b3b_open_rejoin_with_model.get_container());
        pnl_transform_model.add_j_panel(&pnl_transform_model_buttons);
        pnl_rejoin.add(&pnl_transform_model.get_container());
    }

    /// Java private `createJoinPanel()`.
    fn create_join_panel(&self) {
        self.pnl_join_tab.set_box_layout(spaced_panel::X_AXIS);
        // second component
        self.create_finish_join_panel();
    }

    /// Java private `addJoinPanelComponents()`.
    fn add_join_panel_components(&self) {
        // first component
        self.pnl_join_tab
            .add_j_panel(&self.pnl_section_table().get_root_panel());
        self.pnl_section_table().display_cur_tab();
        // second component
        self.pnl_join_tab
            .add_spaced_panel(self.pnl_finish_join.get().expect("pnlFinishJoin"));
    }

    /// Java private `createFinishJoinPanel()`.
    fn create_finish_join_panel(&self) {
        let pnl_finish_join = SpacedPanel::get_instance_void();
        let _ = self.pnl_finish_join.set(pnl_finish_join.clone());
        pnl_finish_join.set_box_layout(spaced_panel::Y_AXIS);
        // Swing painting: BorderFactory.createEtchedBorder().
        pnl_finish_join.set_component_alignment_x(0.5);
        // first component
        let cbs_alignment_ref_section =
            CheckBoxSpinner::get_instance_string(Some("Reference section for alignment: "));
        let num_sections = self.num_sections.get();
        let spinner_model =
            SpinnerNumberModel::new_int(1, 1, if num_sections < 1 { 1 } else { num_sections }, 1);
        cbs_alignment_ref_section.set_model(spinner_model);
        cbs_alignment_ref_section.set_maximum_width(
            UIParameters::get_instance_void()
                .get_spinner_dimension()
                .width,
            false,
        );
        pnl_finish_join.add_container(&cbs_alignment_ref_section.get_container());
        let _ = self
            .cbs_alignment_ref_section
            .set(cbs_alignment_ref_section);
        // second component
        let btn_get_max_size = MultiLineButton::new_string(Some(GET_MAX_SIZE_TEXT));
        pnl_finish_join.add_multi_line_button(&btn_get_max_size);
        let _ = self.btn_get_max_size.set(btn_get_max_size);
        // third component
        let finish_join_panel2 = SpacedPanel::get_instance_void();
        finish_join_panel2.set_box_layout(spaced_panel::X_AXIS);
        let ltf_size_in_x =
            LabeledTextField::new_field_type_string(FieldType::Integer, Some("Size in X: "));
        finish_join_panel2.add_labeled_text_field(&ltf_size_in_x);
        let _ = self.ltf_size_in_x.set(ltf_size_in_x);
        let ltf_size_in_y =
            LabeledTextField::new_field_type_string(FieldType::Integer, Some("Y: "));
        finish_join_panel2.add_labeled_text_field(&ltf_size_in_y);
        let _ = self.ltf_size_in_y.set(ltf_size_in_y);
        pnl_finish_join.add_spaced_panel(&finish_join_panel2);
        // fourth component
        let finish_join_panel3 = SpacedPanel::get_instance_void();
        finish_join_panel3.set_box_layout(spaced_panel::X_AXIS);
        let ltf_shift_in_x =
            LabeledTextField::new_field_type_string(FieldType::Integer, Some("Shift in X: "));
        finish_join_panel3.add_labeled_text_field(&ltf_shift_in_x);
        let _ = self.ltf_shift_in_x.set(ltf_shift_in_x);
        let ltf_shift_in_y =
            LabeledTextField::new_field_type_string(FieldType::Integer, Some("Y: "));
        finish_join_panel3.add_labeled_text_field(&ltf_shift_in_y);
        let _ = self.ltf_shift_in_y.set(ltf_shift_in_y);
        pnl_finish_join.add_spaced_panel(&finish_join_panel3);
        let pnl_local_fits = JComponent::new_panel();
        // Swing layout: BoxLayout X_AXIS, CENTER_ALIGNMENT, horizontal glue.
        pnl_local_fits.add(&self.cb_local_fits.get_component());
        pnl_finish_join.add_j_panel(&pnl_local_fits);
        // fifth component
        self.create_trial_join_panel();
        // sixth component
        self.btn_finish_join
            .set_deferred_3dmod_button_binned_xy_3dmod_button(Some(&self.b3b_open_in_3dmod));
        pnl_finish_join.add_multi_line_button(&self.btn_finish_join);
        // seventh component
        self.b3b_open_in_3dmod.set_spinner_tool_tip_text(Some(
            "The binning to use when opening the joined tomogram in 3dmod.",
        ));
        pnl_finish_join.add_container(&self.b3b_open_in_3dmod.get_container());
        // eight component
        let pnl_refine_with_trial = JComponent::new_panel();
        // Swing layout: BoxLayout X_AXIS, CENTER_ALIGNMENT, horizontal glue.
        pnl_refine_with_trial.add(&self.cb_refine_with_trial.get_component());
        pnl_finish_join.add_j_panel(&pnl_refine_with_trial);
        pnl_finish_join.add_multi_line_button(&self.btn_refine_join);
    }

    /// Java private `createTrialJoinPanel()`.
    fn create_trial_join_panel(&self) {
        let pnl_trial_join = SpacedPanel::get_instance_void();
        pnl_trial_join.set_box_layout(spaced_panel::Y_AXIS);
        pnl_trial_join.set_border(&EtchedBorder::new(Some(TRIAL_JOIN_TEXT)).get_border());
        pnl_trial_join.set_component_alignment_x(0.5);
        // first component
        let trial_join_panel1 = SpacedPanel::get_instance_void();
        trial_join_panel1.set_box_layout(spaced_panel::X_AXIS);
        let z_max = self.pnl_section_table().get_z_max();
        let current_value = if z_max < 1 {
            1
        } else if z_max < 10 {
            z_max
        } else {
            10
        };
        let spin_use_every_n_slices = LabeledSpinner::get_instance_string_int_int_int_int(
            Some("Use every "),
            current_value,
            1,
            if z_max < 1 { 1 } else { z_max },
            1,
        );
        trial_join_panel1.add_labeled_spinner(&spin_use_every_n_slices);
        let _ = self.spin_use_every_n_slices.set(spin_use_every_n_slices);
        trial_join_panel1.add_j_label(&JComponent::new_label("slices"));
        pnl_trial_join.add_spaced_panel(&trial_join_panel1);
        // second component
        let spin_trial_binning = LabeledSpinner::get_instance_string_int_int_int_int(
            Some("Binning in X and Y: "),
            1,
            1,
            50,
            1,
        );
        pnl_trial_join.add_labeled_spinner(&spin_trial_binning);
        let _ = self.spin_trial_binning.set(spin_trial_binning);
        // third component
        self.btn_trial_join
            .set_deferred_3dmod_button_binned_xy_3dmod_button(Some(&self.b3b_open_trial_in_3dmod));
        pnl_trial_join.add_multi_line_button(&self.btn_trial_join);
        // fourth component
        self.b3b_open_trial_in_3dmod.set_spinner_tool_tip_text(Some(
            "The binning to use when opening the trial joined tomogram in 3dmod.",
        ));
        pnl_trial_join.add_container(&self.b3b_open_trial_in_3dmod.get_container());
        // fifth component
        let btn_get_subarea = MultiLineButton::new_string(Some("Get Subarea Size And Shift"));
        pnl_trial_join.add_multi_line_button(&btn_get_subarea);
        let _ = self.btn_get_subarea.set(btn_get_subarea);
        self.pnl_finish_join
            .get()
            .expect("pnlFinishJoin")
            .add_spaced_panel(&pnl_trial_join);
    }

    /// Java package-private `msgRowChange()`.
    pub fn msg_row_change(&self) {
        self.boundary_table().msg_row_change();
    }

    /// Java package-private `setNumSections(int, boolean)`.  Change the model for
    /// spinners when the number of sections changes, but preserve any value set by the
    /// user.
    pub fn set_num_sections(&self, num_sections: i32, init: bool) {
        self.num_sections.set(num_sections);
        let max_sections = if num_sections < 1 { 1 } else { num_sections };
        // density matching (setup)
        let mut spinner_value = EtomoNumber::new_with_type(Some(Type::Integer));
        spinner_value.set_number(Some(self.spin_density_ref_section().get_value()));
        spinner_value.set_display_value_int(1);
        self.spin_density_ref_section()
            .set_model(spinner_value.get_int(), 1, max_sections, 1);
        // alignment (join)
        spinner_value.set_number(Some(self.cbs_alignment_ref_section().get_value()));
        let spinner_model =
            SpinnerNumberModel::new_int(spinner_value.get_int(), 1, max_sections, 1);
        self.cbs_alignment_ref_section().set_model(spinner_model);
        // every n sections (join)
        let z_max = self.pnl_section_table().get_z_max();
        if z_max == 0 {
            spinner_value.set_int(1);
        } else {
            spinner_value.set_ceiling(z_max);
            spinner_value.set_number(Some(self.spin_use_every_n_slices().get_value()));
        }
        let default_every = if z_max < 1 {
            1
        } else if z_max < 10 {
            z_max
        } else {
            10
        };
        if init && z_max > 0 && spinner_value.equals_int(1) {
            // The spinner is being set before the rows are loaded, so reset it if this
            // is during initialization.
            spinner_value.set_int(default_every);
        }
        spinner_value.set_display_value_int(default_every);
        self.spin_use_every_n_slices().set_model(
            spinner_value.get_int(),
            1,
            if z_max < 1 { 1 } else { z_max },
            1,
        );
        // Fixed in translation: the rejoin spinner is created with the Rejoin tab's
        // panel, which the constructor builds before any section is added, so it is
        // always there; Java reads it unchecked.
        if let Some(spin_rejoin_use_every_n_slices) = self.spin_rejoin_use_every_n_slices.get() {
            spinner_value.set_number(Some(spin_rejoin_use_every_n_slices.get_value()));
            if init && z_max > 0 && spinner_value.equals_int(1) {
                // The spinner is being set before the rows are loaded, so reset it if
                // this is during initialization.
                spinner_value.set_int(default_every);
            }
            let min = spinner_value.get_int();
            spin_rejoin_use_every_n_slices.set_model(
                min,
                1,
                if z_max < min { min } else { z_max },
                1,
            );
        }
        self.default_size_in_xy();
    }

    /// Java private `init()`.
    fn init(&self) {
        self.default_x_size
            .set(self.pnl_section_table().get_x_max());
        self.default_y_size
            .set(self.pnl_section_table().get_y_max());
    }

    /// Java package-private `defaultSizeInXY()`.  Call this when the number of sections
    /// has changed or a section rotation has changed.  Will override the values on the
    /// screen if xMax and yMax have changed.
    pub fn default_size_in_xy(&self) {
        // update size in X and Y defaults
        // Java's unused local `metaData`.
        let _meta_data = self.manager.get_const_meta_data();
        let x_max = self.pnl_section_table().get_x_max();
        let y_max = self.pnl_section_table().get_y_max();
        if x_max == self.default_x_size.get() && y_max == self.default_y_size.get() {
            return;
        }
        self.default_x_size.set(x_max);
        self.default_y_size.set(y_max);
        self.ltf_size_in_x().set_text_int(self.default_x_size.get());
        self.ltf_size_in_y().set_text_int(self.default_y_size.get());
    }

    /// Java `setSizeInX(ConstEtomoNumber)`.
    pub fn set_size_in_x(&self, size_in_x: &ConstEtomoNumber) {
        self.ltf_size_in_x()
            .set_text_string(Some(&size_in_x.to_string()));
    }

    /// Java `setSizeInY(ConstEtomoNumber)`.
    pub fn set_size_in_y(&self, size_in_y: &ConstEtomoNumber) {
        self.ltf_size_in_y()
            .set_text_string(Some(&size_in_y.to_string()));
    }

    /// Java `setShiftInX(int)`.
    pub fn set_shift_in_x(&self, shift_in_x: i32) {
        self.ltf_shift_in_x().set_text_int(shift_in_x);
    }

    /// Java `setShiftInY(int)`.
    pub fn set_shift_in_y(&self, shift_in_y: i32) {
        self.ltf_shift_in_y().set_text_int(shift_in_y);
    }

    /// Java package-private `getInvalidReason()`.
    pub fn get_invalid_reason(&self) -> Option<String> {
        if let Some(invalid_reason) = self.invalid_reason.borrow().clone() {
            return Some(invalid_reason);
        }
        self.pnl_section_table().get_invalid_reason()
    }

    /// Java `getMode()`.
    pub fn get_mode(&self) -> i32 {
        self.pnl_section_table().get_mode()
    }

    /// Java `getParameters(XfjointomoParam, boolean)`.
    pub fn get_parameters_xfjointomo_param(
        &self,
        param: &mut XfjointomoParam,
        do_validation: bool,
    ) -> bool {
        let result = (|| -> Result<(), FieldValidationFailedException> {
            param.set_transform(Some(self.tc_model.get_transform()));
            param.set_boundaries_to_analyze(
                self.ltf_boundaries_to_analyze
                    .get_text_boolean(do_validation)?
                    .as_deref(),
            );
            param.set_objects_to_include(
                self.ltf_objects_to_include
                    .get_text_boolean(do_validation)?
                    .as_deref(),
            );
            param.set_points_to_fit(
                self.ltf_points_to_fit_min
                    .get_text_boolean(do_validation)?
                    .as_deref(),
                self.ltf_points_to_fit_max
                    .get_text_boolean(do_validation)?
                    .as_deref(),
            );
            if self.cb_gap.is_selected() {
                param.set_gap_start_end_inc(
                    self.ltf_gap_start
                        .get_text_boolean(do_validation)?
                        .as_deref(),
                    self.ltf_gap_end.get_text_boolean(do_validation)?.as_deref(),
                    self.ltf_gap_inc.get_text_boolean(do_validation)?.as_deref(),
                );
            }
            Ok(())
        })();
        result.is_ok()
    }

    /// Java `getParameters(Joinwarp2modelParam, boolean)`.
    ///
    /// Fixed in translation: a shift that is not an integer makes the source's
    /// `Integer.valueOf` throw NumberFormatException out of the action; the
    /// parameters are not taken here (false), as for a validation failure.
    pub fn get_parameters_joinwarp2model_param(
        &self,
        param: &mut Joinwarp2modelParam,
        do_validation: bool,
    ) -> bool {
        let result = (|| -> Result<bool, FieldValidationFailedException> {
            param.set_binning_of_join(Some(&self.spin_trial_binning().get_value().to_string()));
            let shift_in_x = self.ltf_shift_in_x().get_text_boolean(do_validation)?;
            let shift_in_y = self.ltf_shift_in_y().get_text_boolean(do_validation)?;
            let (Some(offset_in_x), Some(offset_in_y)) = (
                shift_in_x
                    .as_deref()
                    .and_then(|text| text.parse::<i32>().ok())
                    .map(|value| value.wrapping_mul(-1)),
                shift_in_y
                    .as_deref()
                    .and_then(|text| text.parse::<i32>().ok())
                    .map(|value| value.wrapping_mul(-1)),
            ) else {
                return Ok(false);
            };
            let offset_in_xand_y = format!("{}, {}", offset_in_x, offset_in_y);
            param.set_offset_in_xand_y(Some(&offset_in_xand_y));
            param.set_chunk_sizes(Some(&self.pnl_section_table().get_chunk_sizes()));
            Ok(true)
        })();
        result.unwrap_or(false)
    }

    /// Java `setXfjointomoResult() throws LogFileException, IOException,
    /// LockException`.
    pub fn set_xfjointomo_result(&self) -> Result<(), LogFileError> {
        self.boundary_table().set_xfjointomo_result()
    }

    /// Java `getMetaData(JoinMetaData, boolean)`.
    pub fn get_meta_data(&self, meta_data: &JoinMetaData, do_validation: bool) -> bool {
        self.synchronize_void();
        let result = (|| -> Result<bool, FieldValidationFailedException> {
            meta_data.set_root_name(
                self.ltf_root_name
                    .get_text_boolean(do_validation)?
                    .as_deref(),
            );
            meta_data.set_density_ref_section(Some(self.spin_density_ref_section().get_value()));
            if !self.auto_alignment_panel().get_parameters_meta_data(
                &mut ConstJoinMetaData::get_auto_alignment_meta_data(meta_data)
                    .lock()
                    .unwrap(),
                do_validation,
            ) {
                return Ok(false);
            }
            meta_data.set_use_alignment_ref_section(self.cbs_alignment_ref_section().is_selected());
            meta_data.set_alignment_ref_section(Some(self.cbs_alignment_ref_section().get_value()));
            meta_data.set_size_in_x(
                self.ltf_size_in_x()
                    .get_text_boolean(do_validation)?
                    .as_deref(),
            );
            meta_data.set_size_in_y(
                self.ltf_size_in_y()
                    .get_text_boolean(do_validation)?
                    .as_deref(),
            );
            meta_data.set_shift_in_x(
                self.ltf_shift_in_x()
                    .get_text_boolean(do_validation)?
                    .as_deref(),
            );
            meta_data.set_shift_in_y(
                self.ltf_shift_in_y()
                    .get_text_boolean(do_validation)?
                    .as_deref(),
            );
            meta_data.set_local_fits(self.cb_local_fits.is_selected());
            meta_data.set_use_every_n_slices(Some(self.spin_use_every_n_slices().get_value()));
            meta_data.set_rejoin_use_every_n_slices(Some(
                self.spin_rejoin_use_every_n_slices().get_value(),
            ));
            meta_data.set_trial_binning(Some(self.spin_trial_binning().get_value()));
            meta_data.set_midas_limit(
                self.ltf_midas_limit
                    .get_text_boolean(do_validation)?
                    .as_deref(),
            );
            meta_data.set_model_transform(self.tc_model.get_transform());
            meta_data.set_boundaries_to_analyze(
                self.ltf_boundaries_to_analyze
                    .get_text_boolean(do_validation)?
                    .as_deref(),
            );
            meta_data.set_objects_to_include(
                self.ltf_objects_to_include
                    .get_text_boolean(do_validation)?
                    .as_deref(),
            );
            meta_data.set_gap(self.cb_gap.is_selected());
            meta_data.set_gap_start(
                self.ltf_gap_start
                    .get_text_boolean(do_validation)?
                    .as_deref(),
            );
            meta_data.set_gap_end(self.ltf_gap_end.get_text_boolean(do_validation)?.as_deref());
            meta_data.set_gap_inc(self.ltf_gap_inc.get_text_boolean(do_validation)?.as_deref());
            meta_data.set_points_to_fit_min(
                self.ltf_points_to_fit_min
                    .get_text_boolean(do_validation)?
                    .as_deref(),
            );
            meta_data.set_points_to_fit_max(
                self.ltf_points_to_fit_max
                    .get_text_boolean(do_validation)?
                    .as_deref(),
            );
            meta_data.set_rejoin_trial_binning(Some(self.spin_rejoin_trial_binning().get_value()));
            self.boundary_table().get_meta_data();
            Ok(self.pnl_section_table().get_meta_data(meta_data))
        })();
        result.unwrap_or(false)
    }

    /// Java package-private `getSectionTable()`.
    pub fn get_section_table(&self) -> Rc<SectionTablePanel> {
        self.pnl_section_table().clone()
    }

    /// Java `getScreenState(JoinScreenState)`.
    pub fn get_screen_state(&self, screen_state: &JoinScreenState) {
        screen_state.set_refine_with_trial(self.cb_refine_with_trial.is_selected());
        self.boundary_table().get_screen_state();
    }

    /// Java package-private `setMetaData(ConstJoinMetaData)`.
    pub fn set_meta_data(&self, meta_data: &dyn ConstJoinMetaData) {
        self.ltf_root_name
            .set_text_string(Some(&meta_data.get_dataset_name()));
        self.spin_density_ref_section()
            .set_value_int(meta_data.get_density_ref_section().get_int());
        self.auto_alignment_panel()
            .set_parameters(&meta_data.get_auto_alignment_meta_data().lock().unwrap());
        self.cbs_alignment_ref_section()
            .set_selected(meta_data.is_use_alignment_ref_section());
        self.cbs_alignment_ref_section()
            .set_value_int(meta_data.get_alignment_ref_section().get_int());
        self.ltf_shift_in_x()
            .set_text_string(Some(&meta_data.get_shift_in_x().to_string()));
        self.ltf_shift_in_y()
            .set_text_string(Some(&meta_data.get_shift_in_y().to_string()));
        self.spin_use_every_n_slices()
            .set_value_const_etomo_number(&meta_data.get_use_every_n_slices());
        self.spin_rejoin_use_every_n_slices()
            .set_value_const_etomo_number(&meta_data.get_rejoin_use_every_n_slices());
        self.spin_trial_binning()
            .set_value_const_etomo_number(&meta_data.get_trial_binning());
        self.ltf_midas_limit
            .set_text_string(Some(&meta_data.get_midas_limit().to_string()));
        self.pnl_section_table().set_meta_data(meta_data);
        self.ltf_size_in_x()
            .set_text_string(Some(&meta_data.get_size_in_x().to_string()));
        self.ltf_size_in_y()
            .set_text_string(Some(&meta_data.get_size_in_y().to_string()));
        self.cb_local_fits
            .set_selected_boolean(meta_data.is_local_fits());
        self.tc_model
            .set_transform(Some(meta_data.get_model_transform()));
        self.ltf_boundaries_to_analyze
            .set_text_string(meta_data.get_boundaries_to_analyze().as_deref());
        self.ltf_objects_to_include
            .set_text_string(meta_data.get_objects_to_include().as_deref());
        self.cb_gap.set_selected_boolean(meta_data.get_gap().is());
        self.update_display();
        self.ltf_gap_start
            .set_text_const_etomo_number(Some(&meta_data.get_gap_start()));
        self.ltf_gap_end
            .set_text_const_etomo_number(Some(&meta_data.get_gap_end()));
        self.ltf_gap_inc
            .set_text_const_etomo_number(Some(&meta_data.get_gap_inc()));
        self.ltf_points_to_fit_min
            .set_text_const_etomo_number(Some(&meta_data.get_points_to_fit_min()));
        self.ltf_points_to_fit_max
            .set_text_const_etomo_number(Some(&meta_data.get_points_to_fit_max()));
        self.spin_rejoin_trial_binning()
            .set_value_const_etomo_number(&meta_data.get_rejoin_trial_binning());
        self.update_display();
    }

    /// Java package-private `setScreenState(JoinScreenState)`.
    pub fn set_screen_state(&self, screen_state: &JoinScreenState) {
        if self.cb_refine_with_trial.is_enabled() {
            self.cb_refine_with_trial
                .set_selected_boolean(screen_state.get_refine_with_trial().is());
        }
    }

    /// Java `isRefineWithTrial()`.
    pub fn is_refine_with_trial(&self) -> bool {
        self.cb_refine_with_trial.is_selected()
    }

    /// Java `getContainer()`.
    pub fn get_container(&self) -> Rc<JComponent> {
        self.root_panel.get().expect("rootPanel").clone()
    }

    /// Java `validateMakejoincom()`.
    pub fn validate_makejoincom(&self) -> bool {
        self.pnl_section_table().validate_makejoincom()
    }

    /// Java `validateFinishjoin()`.
    pub fn validate_finishjoin(&self) -> bool {
        self.pnl_section_table().validate_finishjoin()
    }

    /// Java `getWorkingDirName()`.
    pub fn get_working_dir_name(&self) -> Option<String> {
        self.ftf_working_dir().get_text()
    }

    /// Java `getWorkingDir()`.
    pub fn get_working_dir(&self) -> Option<PathBuf> {
        let working_dir_name = self.ftf_working_dir().get_text();
        let Some(working_dir_name) = working_dir_name.filter(|name| {
            !name.is_empty()
                && !name
                    .chars()
                    .all(|c| matches!(c, ' ' | '\t' | '\n' | '\x0B' | '\x0C' | '\r'))
        }) else {
            return None;
        };
        if working_dir_name.ends_with(' ') {
            ui_harness::with(|harness| {
                harness.open_message_dialog_base_manager_string_string_axis_id(
                    Some(self.manager),
                    &format!(
                        "The directory, {}, cannot be used because it ends with a space.",
                        working_dir_name
                    ),
                    "Unusable Directory Name",
                    Some(AxisID::Only),
                )
            });
            return None;
        }
        self.ftf_working_dir().get_text().map(PathBuf::from)
    }

    /// Java `getRootName()`.
    pub fn get_root_name(&self) -> Option<String> {
        self.ltf_root_name.get_text_void()
    }

    /// Java `abortAddSection()`.
    pub fn abort_add_section(&self) {
        self.pnl_section_table().enable_add_section();
    }

    /// Java `equals(ConstJoinMetaData)`.  Checking if dialog is equal to meta data.
    /// Set useDefault to match how useDefault is used in setMetaData().
    ///
    /// Fixed in translation (JoinDialog.java:1147): `autoAlignmentPanel.equals(metaData)`
    /// resolves to `Object.equals(Object)` (the panel's own `equals` takes an
    /// `AutoAlignmentMetaData`), which is false for every meta data, so the method
    /// always returned false.  The panel is compared with the meta data's
    /// auto-alignment parameters here.  No caller in the source.
    pub fn equals(&self, meta_data: &dyn ConstJoinMetaData) -> bool {
        if !self
            .ltf_root_name
            .equals_string(Some(&meta_data.get_dataset_name()))
        {
            return false;
        }
        if !meta_data
            .get_density_ref_section()
            .equals_number(Some(self.spin_density_ref_section().get_value()))
        {
            return false;
        }
        if !self
            .auto_alignment_panel()
            .equals(&meta_data.get_auto_alignment_meta_data().lock().unwrap())
        {
            return false;
        }
        if self.cbs_alignment_ref_section().is_selected()
            != meta_data.is_use_alignment_ref_section()
        {
            return false;
        }
        if !meta_data
            .get_alignment_ref_section()
            .equals_number(Some(self.cbs_alignment_ref_section().get_value()))
        {
            return false;
        }
        if !meta_data
            .get_size_in_x()
            .equals_string(self.ltf_size_in_x().get_text_void().as_deref())
        {
            return false;
        }
        if !meta_data
            .get_size_in_y()
            .equals_string(self.ltf_size_in_y().get_text_void().as_deref())
        {
            return false;
        }
        if !meta_data
            .get_shift_in_x()
            .equals_string(self.ltf_shift_in_x().get_text_void().as_deref())
        {
            return false;
        }
        if !meta_data
            .get_shift_in_y()
            .equals_string(self.ltf_shift_in_y().get_text_void().as_deref())
        {
            return false;
        }
        if !meta_data
            .get_use_every_n_slices()
            .equals_number(Some(self.spin_use_every_n_slices().get_value()))
        {
            return false;
        }
        if !meta_data
            .get_rejoin_use_every_n_slices()
            .equals_number(Some(self.spin_rejoin_use_every_n_slices().get_value()))
        {
            return false;
        }
        if !meta_data
            .get_trial_binning()
            .equals_number(Some(self.spin_trial_binning().get_value()))
        {
            return false;
        }
        if !self.pnl_section_table().equals(meta_data) {
            return false;
        }
        true
    }

    /// Java package-private `equalsSample(ConstJoinMetaData)`.  Checking if dialog
    /// fields used to make the sample are equal to the fields in meta data.
    pub fn equals_sample(&self, meta_data: &dyn ConstJoinMetaData) -> bool {
        if !self
            .ltf_root_name
            .equals_string(Some(&meta_data.get_dataset_name()))
        {
            return false;
        }
        if !meta_data
            .get_density_ref_section()
            .equals_number(Some(self.spin_density_ref_section().get_value()))
        {
            return false;
        }
        if !self.pnl_section_table().equals_sample(meta_data) {
            return false;
        }
        true
    }

    /// Java `addSection(File)`.
    pub fn add_section(&self, tomogram: &Path) {
        self.pnl_section_table().add_section_file(tomogram);
    }

    /// Java `setRefineDataHighlight(boolean)`.
    pub fn set_refine_data_highlight(&self, highlight: bool) {
        self.ltf_size_in_x().set_highlight(highlight);
        self.ltf_size_in_y().set_highlight(highlight);
        self.ltf_shift_in_x().set_highlight(highlight);
        self.ltf_shift_in_y().set_highlight(highlight);
        self.cbs_alignment_ref_section().set_highlight(highlight);
        self.pnl_section_table()
            .set_join_final_start_highlight(highlight);
        self.pnl_section_table()
            .set_join_final_end_highlight(highlight);
        self.btn_get_subarea().set_highlight(highlight);
        self.btn_get_max_size().set_highlight(highlight);
        if self.cb_refine_with_trial.is_selected() {
            self.spin_trial_binning().set_highlight(highlight);
            self.spin_use_every_n_slices().set_highlight(highlight);
        }
    }

    /// Java `getSectionTableMetaData()`.
    pub fn get_section_table_meta_data(&self) -> bool {
        if !self
            .pnl_section_table()
            .get_meta_data(self.manager.get_join_meta_data())
        {
            ui_harness::with(|harness| {
                harness.open_message_dialog_base_manager_string_string(
                    Some(self.manager),
                    "Unable to proceed.  Screen data is invalid.",
                    "Data Error",
                )
            });
            return false;
        }
        true
    }

    /// Java `getButtonName(boolean)`.
    pub fn get_button_name(&self, use_trial: bool) -> &'static str {
        if use_trial {
            TRIAL_JOIN_TEXT
        } else {
            FINISH_JOIN_TEXT
        }
    }

    /// Java private `setRefiningJoin()`.
    fn set_refining_join_void(&self) {
        self.refining_join.set(
            file_type::CLASS
                .modeled_join
                .get_file(Some(self.manager), Some(AxisID::Only))
                .is_some_and(|file| file.exists()),
        );
        // refiningJoin = DatasetFiles.getModeledJoinFile(manager).exists();
    }

    /// Java `setRefiningJoin(boolean)`.
    pub fn set_refining_join(&self, input: bool) {
        self.refining_join.set(input);
    }

    /// Java package-private `workingDirAction()`.
    fn working_dir_action(&self) {
        // Open up the file chooser in the current working directory
        let chooser = FileChooser::new_base_manager(Some(self.manager));
        // Swing layout: chooser.setPreferredSize(fileChooserDimension).
        chooser.set_file_selection_mode(file_chooser::DIRECTORIES_ONLY);
        let return_val = chooser.show_open_dialog(Some(&self.get_container()));
        if return_val == file_chooser::APPROVE_OPTION
            && let Some(working_dir) = chooser.get_selected_file()
        {
            self.ftf_working_dir().set_text_string(Some(
                &utilities::java_io_file_get_absolute_path(&working_dir.to_string_lossy()),
            ));
        }
    }

    /// Java package-private `modelFileAction()`.
    fn model_file_action(&self) {
        // Open up the file chooser in the current working directory
        let chooser = FileChooser::new_base_manager(Some(self.manager));
        let model_filter = ModelFileFilter::new();
        chooser.set_file_filter(Some(Rc::new(model_filter) as Rc<dyn FileFilter>));
        // Swing layout: chooser.setPreferredSize(fileChooserDimension).
        let return_val = chooser.show_open_dialog(Some(&self.ftf_model_file.get_container()));
        if return_val == file_chooser::APPROVE_OPTION
            && let Some(model_file) = chooser.get_selected_file()
        {
            self.ftf_model_file
                .set_text_string(Some(&utilities::java_io_file_get_absolute_path(
                    &model_file.to_string_lossy(),
                )));
        }
    }

    /// Java package-private final `synchronize(Tab)`.  Synchronize when changing tabs.
    fn synchronize_tab(&self, prev_tab: Tab) {
        self.pnl_section_table()
            .synchronize(Some(prev_tab), Some(self.cur_tab.get()));
    }

    /// Java package-private final `synchronize()`.  Synchronize when saving to the
    /// .ejf file.
    fn synchronize_void(&self) {
        self.pnl_section_table()
            .synchronize(Some(self.cur_tab.get()), None);
    }

    /// Java private `setToolTipText()`.
    fn set_tool_tip_text(&self) {
        let text = "Enter the directory where you wish to place the joined tomogram.";
        self.ftf_working_dir().set_tool_tip_text(Some(text));
        self.ftf_working_dir().set_tool_tip_text(Some(text));
        self.ltf_root_name
            .set_tool_tip_text(Some("Enter the root name for the joined tomogram."));
        let text = "The size to which samples will be squeezed if they are bigger (default 1024).";
        self.ltf_midas_limit.set_tool_tip_text(Some(text));
        self.lbl_midas_limit.set_tool_tip_text(
            super::tooltip_formatter::INSTANCE
                .format(Some(text))
                .as_deref(),
        );
        self.spin_density_ref_section().set_tool_tip_text(Some(
            "Select a section to use as a reference for density scaling.",
        ));
        self.btn_change_setup()
            .set_tool_tip_text(Some("Press to redo an existing sample."));
        self.btn_revert_to_last_setup()
            .set_tool_tip_text(Some("Press to go back to the existing sample."));
        self.btn_make_samples
            .set_tool_tip_text(Some("Press to make a sample."));
        self.btn_open_sample_averages
            .set_tool_tip_text(Some("Press to the sample averages file in 3dmod."));
        self.btn_open_sample()
            .set_tool_tip_text(Some("Press to the sample file in 3dmod."));
        self.btn_open_sample()
            .set_tool_tip_text(Some("Press to the sample file in 3dmod."));
        self.cbs_alignment_ref_section()
            .set_check_box_tool_tip_text(Some(
                "Make a section the reference for alignment.  This means that the chosen section \
             will not be transformed, and the other sections will be transformed into \
             alignment with it.",
            ));
        self.cbs_alignment_ref_section()
            .set_spinner_tool_tip_text(Some(
                "Choose a section to be the reference for alignment.  This means that it will not \
             be transformed, and the other sections will be transformed into alignment with \
             it.",
            ));
        self.btn_get_max_size().set_tool_tip_text(Some(
            "Compute the maximum size and offsets needed to contain the transformed images \
             from all of the sections, given the current transformations.",
        ));
        self.ltf_size_in_x().set_tool_tip_text(Some(
            "The size in X parameter for the trial and final joined tomograms.",
        ));
        self.ltf_size_in_y().set_tool_tip_text(Some(
            "The size in Y parameter for the trial and final joined tomograms.",
        ));
        self.ltf_shift_in_x().set_tool_tip_text(Some(
            "The X offset parameter for the trial and final joined tomograms.",
        ));
        self.ltf_shift_in_y().set_tool_tip_text(Some(
            "The Y offset parameter for the trial and final joined tomograms.",
        ));
        self.cb_local_fits.set_tool_tip_text_string(Some(
            "When running Xftoxg(1) on the primary alignment transforms, run the program in \
             its default mode, which does local fits to 7 adjacent sections.  This option may \
             eliminate unwanted trends in data sets with many sections.  When it is not \
             entered, Xftoxg(1) is run with \"-nfit 0\", which computes a global alignment.",
        ));
        let text = "Slices to use when creating the trial joined tomogram.";
        self.spin_use_every_n_slices().set_tool_tip_text(Some(text));
        self.spin_rejoin_use_every_n_slices()
            .set_tool_tip_text(Some(text));
        let text = "The binning to use when creating the trial joined tomogram.";
        self.spin_trial_binning().set_tool_tip_text(Some(text));
        self.spin_rejoin_trial_binning()
            .set_tool_tip_text(Some(text));
        self.btn_trial_join.set_tool_tip_text(Some(
            "Press to make a trial version of the joined tomogram.",
        ));
        self.b3b_open_trial_in_3dmod
            .set_button_tool_tip_text(Some("Press to open the trial joined tomogram."));
        self.btn_get_subarea().set_tool_tip_text(Some(
            "Press to get maximum size and shift from the trail joined tomogram using the \
             rubber band functionality in 3dmod.",
        ));
        self.btn_finish_join
            .set_tool_tip_text(Some("Press to make the joined tomogram."));
        self.b3b_open_in_3dmod
            .set_button_tool_tip_text(Some("Press to open the joined tomogram in 3dmod."));
        self.cb_refine_with_trial.set_tool_tip_text_string(Some(
            "Check to make the refining model using the trial join.",
        ));
        self.btn_refine_join.set_tool_tip_text(Some(
            "Press to refine the serial section join using a refining model.",
        ));
        self.btn_make_refining_model
            .set_tool_tip_text(Some("Press to create the refining model."));
        let mut autodoc: *const dyn ReadOnlyAutodoc = std::ptr::null::<Autodoc>();
        let result = (|| -> Result<(), LogFileError> {
            autodoc = unsafe {
                autodoc_factory::get_instance(
                    Some(self.manager),
                    Some(autodoc_factory::XFJOINTOMO),
                    AxisID::Only,
                    false,
                )
            }? as *const Autodoc;
            Ok(())
        })();
        let autodoc_loaded = match result {
            Ok(()) => true,
            Err(LogFileError::Lock(_)) => false,
            Err(except) => {
                eprintln!("{}", except);
                false
            }
        };
        // SAFETY: the pointer is null or an autodoc the factory keeps for the life of
        // the process.
        let autodoc: Option<&dyn ReadOnlyAutodoc> = if autodoc.is_null() {
            None
        } else {
            Some(unsafe { &*autodoc })
        };
        if autodoc_loaded {
            self.ltf_boundaries_to_analyze.set_tool_tip_text(
                etomo_autodoc::get_tooltip(
                    autodoc,
                    Some(xfjointomo_param::BOUNDARIES_TO_ANALYZE_KEY),
                )
                .as_deref(),
            );
            self.ltf_objects_to_include.set_tool_tip_text(
                etomo_autodoc::get_tooltip(autodoc, Some(xfjointomo_param::OBJECTS_TO_INCLUDE))
                    .as_deref(),
            );
        }
        self.cb_gap.set_tool_tip_text_string(Some(
            "Check to allow the the final start and end values to change when the join is \
             recreated.  Uncheck keep the existing final start and end values",
        ));
        if autodoc.is_some() {
            if let Some(text) =
                etomo_autodoc::get_tooltip(autodoc, Some(xfjointomo_param::GAP_START_END_INC))
            {
                self.ltf_gap_start.set_tool_tip_text(Some(&text));
                self.ltf_gap_end.set_tool_tip_text(Some(&text));
                self.ltf_gap_inc.set_tool_tip_text(Some(&text));
            }
            if let Some(text) =
                etomo_autodoc::get_tooltip(autodoc, Some(xfjointomo_param::POINTS_TO_FIT))
            {
                self.ltf_points_to_fit_min.set_tool_tip_text(Some(&text));
                self.ltf_points_to_fit_max.set_tool_tip_text(Some(&text));
            }
        }
        self.btn_xfjointomo.set_tool_tip_text(Some(
            "Press to run xfjointomo, which computes transforms for aligning tomograms of \
             serial sections from features modeled on an initial joined tomogram.",
        ));
        self.btn_transform_and_view_model.set_tool_tip_text(Some(
            "Press to apply tranformations to the refining model and view the result.",
        ));
        self.btn_rejoin.set_tool_tip_text(Some(
            "Press to make the joined tomogram using the adjusted end and start values.",
        ));
        self.btn_trial_rejoin.set_tool_tip_text(Some(
            "Press to make a trial version of the joined tomogram using the adjusted end and \
             start values.",
        ));
        self.ftf_model_file
            .set_tool_tip_text(Some("The model to transform."));
        if let Some(ltf_transformed_model) = self.ltf_transformed_model.borrow().as_ref() {
            ltf_transformed_model.set_tool_tip_text(Some("The transformed model."));
        }
        self.btn_transform_model
            .set_tool_tip_text(Some("Press to transform the model"));
        self.b3b_open_rejoin_with_model
            .set_button_tool_tip_text(Some(
                "Press to open the joined tomogram with the transformed model.",
            ));
        self.b3b_open_trial_rejoin.set_button_tool_tip_text(Some(
            "Press to open the trial joined tomogram with the transformed refining model.",
        ));
        self.b3b_open_rejoin.set_button_tool_tip_text(Some(
            "Press to open the joined tomogram with the transformed refining model.",
        ));
    }

    /// Java field read `ltfTransformedModel`.
    fn ltf_transformed_model(&self) -> Rc<LabeledTextField> {
        self.ltf_transformed_model
            .borrow()
            .clone()
            .expect("ltfTransformedModel")
    }
}

impl ContextMenu for JoinDialog {
    /// Java `popUpContextMenu(MouseEvent)`.  Right mouse button context menu.
    fn pop_up_context_menu(&self, mouse_event: &MouseEvent) {
        let root_panel = self.get_container();
        let strings = |values: &[&str]| -> Vec<String> {
            values.iter().map(|value| value.to_string()).collect()
        };
        let result = match self.cur_tab.get() {
            Tab::Setup => {
                let man_page_label = strings(&["3dmod"]);
                let man_page = strings(&["3dmod.html"]);
                let log_file_label = strings(&["startjoin"]);
                let log_file = strings(&["startjoin.log"]);
                ContextPopup::new_component_mouse_event_string_string_string_array_string_array_string_array_string_array_base_manager_axis_id(
                    &root_panel,
                    mouse_event,
                    Some("Setup"),
                    Some(context_popup::JOIN_GUIDE),
                    &man_page_label,
                    &man_page,
                    Some(&log_file_label),
                    Some(&log_file),
                    self.manager,
                    self.axis_id,
                )
                .map(|_| ())
            }
            Tab::Align => {
                let man_page_label = strings(&["Xfalign", "Midas", "3dmod"]);
                let man_page = strings(&["xfalign.html", "midas.html", "3dmod.html"]);
                ContextPopup::new_component_mouse_event_string_string_string_array_string_array_base_manager_axis_id(
                    &root_panel,
                    mouse_event,
                    Some("Align"),
                    Some(context_popup::JOIN_GUIDE),
                    &man_page_label,
                    &man_page,
                    self.manager,
                    self.axis_id,
                )
                .map(|_| ())
            }
            Tab::Join => {
                let man_page_label = strings(&["Finishjoin", "3dmod"]);
                let man_page = strings(&["finishjoin.html", "3dmod.html"]);
                ContextPopup::new_component_mouse_event_string_string_string_array_string_array_base_manager_axis_id(
                    &root_panel,
                    mouse_event,
                    Some("Joining"),
                    Some(context_popup::JOIN_GUIDE),
                    &man_page_label,
                    &man_page,
                    self.manager,
                    self.axis_id,
                )
                .map(|_| ())
            }
            Tab::Model => {
                let man_page_label = strings(&["Xfjointomo", "3dmod"]);
                let man_page = strings(&["xfjointomo.html", "3dmod.html"]);
                let log_file_label = strings(&["xfjointomo"]);
                let log_file = strings(&["xfjointomo.log"]);
                ContextPopup::new_component_mouse_event_string_string_string_array_string_array_string_array_string_array_base_manager_axis_id(
                    &root_panel,
                    mouse_event,
                    Some("Joining"),
                    Some(context_popup::JOIN_GUIDE),
                    &man_page_label,
                    &man_page,
                    Some(&log_file_label),
                    Some(&log_file),
                    self.manager,
                    self.axis_id,
                )
                .map(|_| ())
            }
            Tab::Rejoin => Ok(()),
        };
        // An uncaught IllegalStateException from the popup's validation: Swing's
        // handler prints it.
        if let Err(message) = result {
            eprintln!("java.lang.IllegalStateException: {message}");
        }
    }
}

impl Run3dmodButtonContainer for JoinDialog {
    /// Java `action(String, Deferred3dmodButton, Run3dmodMenuOptions)`.  Handle
    /// actions.
    fn action(
        &self,
        command: &str,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    ) {
        let manager = self.manager;
        let command = Some(command);
        let result = (|| -> Result<(), FieldValidationFailedException> {
            if command == self.btn_make_samples.get_action_command().as_deref() {
                if self.ftf_working_dir().is_editable() {
                    let working_dir = self.ftf_working_dir().get_file();
                    // Fixed in translation: an empty working directory makes the source
                    // pass a null File to `validateDatasetName`, which dereferences it
                    // (NullPointerException); the dataset name is not valid then.
                    let valid = match working_dir.as_deref() {
                        None => false,
                        Some(working_dir) => dataset_tool::validate_dataset_name_directory(
                            manager,
                            self.axis_id,
                            working_dir,
                            self.ltf_root_name.get_text_void().as_deref(),
                            DataFileType::Join,
                            None,
                            true,
                        ),
                    };
                    if !valid {
                        return Ok(());
                    }
                }
                manager.makejoincom(
                    None,
                    deferred_3dmod_button,
                    run_3dmod_menu_options,
                    Some(DIALOG_TYPE),
                );
            } else if command == self.btn_finish_join.get_action_command().as_deref() {
                manager.finishjoin(
                    finishjoin_param::Mode::FinishJoin,
                    FINISH_JOIN_TEXT,
                    None,
                    deferred_3dmod_button,
                    run_3dmod_menu_options,
                    Some(DIALOG_TYPE),
                );
            } else if command == self.btn_get_max_size().get_action_command().as_deref() {
                manager.finishjoin(
                    finishjoin_param::Mode::MaxSize,
                    GET_MAX_SIZE_TEXT,
                    None,
                    None,
                    None,
                    Some(DIALOG_TYPE),
                );
            } else if command == self.btn_trial_join.get_action_command().as_deref() {
                manager.finishjoin(
                    finishjoin_param::Mode::Trial,
                    TRIAL_JOIN_TEXT,
                    None,
                    deferred_3dmod_button,
                    run_3dmod_menu_options,
                    Some(DIALOG_TYPE),
                );
            } else if command == self.btn_get_subarea().get_action_command().as_deref() {
                let coordinates = manager.imod_get_rubberband_coordinates(
                    Some(imod_manager::TRIAL_JOIN_KEY),
                    Some(AxisID::Only),
                );
                if let Err(e) = self.set_size_and_shift(coordinates.as_deref()) {
                    ui_harness::with(|harness| {
                        harness.open_message_dialog_base_manager_string_string(
                            Some(manager),
                            &format!(
                                "Unable to retrieve the trial join's binning, size and/or \
                                 shift.  Please press {} and then rebuild the trial join in \
                                 order get a correct subarea size and shift.\n{}",
                                self.btn_get_max_size()
                                    .get_unformatted_label()
                                    .unwrap_or_else(|| "null".to_string()),
                                e.get_message()
                            ),
                            "Rerun Trial Join",
                        )
                    });
                }
            } else if command == self.btn_change_setup().get_action_command().as_deref() {
                // Prepare for Revert: meta data file should match the screen
                let meta_data = manager.get_join_meta_data();
                self.get_meta_data(meta_data, false);
                match manager.get_parameter_store(Some(self.axis_id)) {
                    Ok(Some(parameter_store)) => {
                        if let Err(e) = parameter_store.lock().unwrap().save(Some(
                            meta_data as &dyn crate::imod::etomo::storage::storable::Storable,
                        )) {
                            match e {
                                LogFileError::Lock(_) => {}
                                e => ui_harness::with(|harness| {
                                    harness.open_message_dialog_base_manager_string_string(
                                        Some(manager),
                                        &format!(
                                            "Unable to save or write JoinMetaData.\n{}",
                                            e.get_message()
                                        ),
                                        "Etomo Error",
                                    )
                                }),
                            }
                        }
                    }
                    Ok(None) => {}
                    Err(LogFileError::Lock(_)) => {}
                    Err(e) => ui_harness::with(|harness| {
                        harness.open_message_dialog_base_manager_string_string(
                            Some(manager),
                            &format!("Unable to save or write JoinMetaData.\n{}", e.get_message()),
                            "Etomo Error",
                        )
                    }),
                }
                self.set_mode(CHANGING_SAMPLE_MODE);
            } else if command
                == self
                    .btn_revert_to_last_setup()
                    .get_action_command()
                    .as_deref()
            {
                // Java's unused local `metaData`.
                let _meta_data = manager.get_const_meta_data();
                if !self.state.is_sample_produced() {
                    panic!(
                        "java.lang.IllegalStateException: sample produced is false but Revert \
                         to Last Setup is enabled"
                    );
                }
                self.pnl_section_table().delete_sections();
                self.set_meta_data(manager.get_const_meta_data());
                self.state.revert();
                self.set_mode(SAMPLE_PRODUCED_MODE);
            } else if command == self.btn_refine_join.get_action_command().as_deref() {
                manager.start_refine();
            } else if command == self.btn_xfjointomo.get_action_command().as_deref() {
                manager.xfjointomo(None);
            } else if command == self.btn_rejoin.get_action_command().as_deref() {
                manager.finishjoin(
                    finishjoin_param::Mode::Rejoin,
                    REJOIN_TEXT,
                    None,
                    deferred_3dmod_button,
                    run_3dmod_menu_options,
                    Some(DIALOG_TYPE),
                );
            } else if command == self.btn_trial_rejoin.get_action_command().as_deref() {
                manager.finishjoin(
                    finishjoin_param::Mode::TrialRejoin,
                    TRIAL_REJOIN_TEXT,
                    None,
                    deferred_3dmod_button,
                    run_3dmod_menu_options,
                    Some(DIALOG_TYPE),
                );
            } else if command == self.cb_gap.get_action_command().as_deref() {
                self.update_display();
            } else if command == self.btn_transform_model.get_action_command().as_deref() {
                manager.xfmodel_with_files(
                    self.ftf_model_file.get_text().as_deref(),
                    self.ltf_transformed_model()
                        .get_text_boolean(true)?
                        .as_deref(),
                    None,
                    deferred_3dmod_button,
                    run_3dmod_menu_options,
                    Some(DIALOG_TYPE),
                );
            } else if command
                == self
                    .btn_transform_and_view_model
                    .get_action_command()
                    .as_deref()
            {
                manager.finishjoin(
                    finishjoin_param::Mode::SuppressExecution,
                    REJOIN_TEXT,
                    None,
                    None,
                    None,
                    Some(DIALOG_TYPE),
                );
            } else if command == self.btn_open_sample().get_action_command().as_deref() {
                manager.imod_open(Some(imod_manager::JOIN_SAMPLES_KEY), run_3dmod_menu_options);
            } else if command
                == self
                    .btn_open_sample_averages
                    .get_action_command()
                    .as_deref()
            {
                manager.imod_open(
                    Some(imod_manager::JOIN_SAMPLE_AVERAGES_KEY),
                    run_3dmod_menu_options,
                );
            } else if command == self.b3b_open_in_3dmod.get_action_command().as_deref() {
                manager.imod_open_binning(
                    Some(imod_manager::JOIN_KEY),
                    self.b3b_open_in_3dmod.get_binning_in_xand_y(),
                    run_3dmod_menu_options,
                );
            } else if command == self.b3b_open_trial_in_3dmod.get_action_command().as_deref() {
                manager.imod_open_binning(
                    Some(imod_manager::TRIAL_JOIN_KEY),
                    self.b3b_open_trial_in_3dmod.get_binning_in_xand_y(),
                    run_3dmod_menu_options,
                );
            } else if command == self.btn_make_refining_model.get_action_command().as_deref() {
                manager.imod_open_axis(
                    Some(AxisID::Only),
                    Some(imod_manager::MODELED_JOIN_KEY),
                    Some(&dataset_files::get_refine_model_file_name(manager)),
                    run_3dmod_menu_options,
                    true,
                );
            } else if command == self.b3b_open_rejoin.get_action_command().as_deref() {
                manager.imod_open_binning_model(
                    Some(imod_manager::JOIN_KEY),
                    self.b3b_open_rejoin.get_binning_in_xand_y(),
                    Some(&dataset_files::get_refine_aligned_model_file_name(manager)),
                    run_3dmod_menu_options,
                );
            } else if command == self.b3b_open_trial_rejoin.get_action_command().as_deref() {
                let use_every_n_slices = self.state.get_refine_trial_use_every_n_slices();
                if use_every_n_slices.is_null() || use_every_n_slices.gt_int(1) {
                    // don't open the model if all the slices have not been included
                    manager.imod_open_binning(
                        Some(imod_manager::TRIAL_JOIN_KEY),
                        self.b3b_open_trial_rejoin.get_binning_in_xand_y(),
                        run_3dmod_menu_options,
                    );
                } else {
                    manager.imod_open_binning_model(
                        Some(imod_manager::TRIAL_JOIN_KEY),
                        self.b3b_open_trial_rejoin.get_binning_in_xand_y(),
                        Some(&dataset_files::get_refine_aligned_model_file_name(manager)),
                        run_3dmod_menu_options,
                    );
                }
            } else if command
                == self
                    .b3b_open_rejoin_with_model
                    .get_action_command()
                    .as_deref()
            {
                manager.imod_open_binning_model(
                    Some(imod_manager::JOIN_KEY),
                    self.b3b_open_rejoin_with_model.get_binning_in_xand_y(),
                    self.ltf_transformed_model()
                        .get_text_boolean(true)?
                        .as_deref(),
                    run_3dmod_menu_options,
                );
            } else {
                self.update_display();
            }
            Ok(())
        })();
        // catch (FieldValidationFailedException e) {}
        let _ = result;
    }
}

impl AutoAlignmentDisplay for JoinDialog {
    /// Java `msgProcessEnded()`.  The purpose of this function is to have Midas button
    /// enabled whether or not Initial Auto-Alignment succeeds.
    fn msg_process_ended(&self) {
        self.auto_alignment_panel().msg_process_change(true);
    }

    /// Java `getDialogType()`.
    fn get_dialog_type(&self) -> DialogType {
        DialogType::Join
    }

    /// Java `getAxisID()`.
    fn get_axis_id(&self) -> AxisID {
        self.axis_id
    }

    /// Java `getAutoAlignmentParameters(MidasParam)`.
    fn get_auto_alignment_parameters_midas(&self, param: &mut MidasParam) {
        self.manager.get_parameters_midas(param, self.axis_id);
    }

    /// Java `getAutoAlignmentParameters(XfalignParam, boolean)`.
    fn get_auto_alignment_parameters_xfalign(
        &self,
        param: &mut XfalignParam,
        _do_validation: bool,
    ) -> bool {
        self.manager.get_parameters_xfalign(param, self.axis_id);
        true
    }
}

/// Java `toString()`.
impl std::fmt::Display for JoinDialog {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "etomo.ui.swing.JoinDialog[{}]", self.param_string())
    }
}

/// `Number` values the spinners report.
#[allow(dead_code)]
fn _number(_: Number) {}
