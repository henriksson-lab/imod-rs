//! `IMOD/Etomo/src/etomo/ui/swing/BatchRunTomoDialog.java`.
//!
//! The one dialog of the batchruntomo interface: a beveled "Batchruntomo
//! Interface" panel holding a tabbed pane with the Batch Setup, Stacks, Dataset
//! Values and Run tabs.  The dataset table (`BatchRunTomoTable`) moves between the
//! Stacks, Dataset and Run tabs; the global `BatchRunTomoDatasetDialog` is the body
//! of the Dataset tab; the Run tab holds the parallel status panel, the resources,
//! the series watcher, the step panel, the email field and the run buttons.
//!
//! Object model (ui.md): the dialog is an `Rc<BatchRunTomoDialog>` living on the
//! event dispatch thread, every method takes `&self`.  The sub-objects the Java
//! constructor hands `this` to (series watcher panel, table, dataset dialog, step
//! panel) are created right after the `Rc` exists and kept in `OnceCell`s, in the
//! constructor's order.  Swing layout (`BoxLayout`, rigid areas, glue, preferred
//! sizes, alignment) is not modelled; it is recorded as `// Swing layout:` comments.
//!
//! `getGlobalDirectivesDialog()` has no caller in the Java and is not translated
//! (DEAD_CODE.md).

use std::cell::{Cell, OnceCell, RefCell};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::rc::{Rc, Weak};
use std::sync::{Arc, Mutex};

use super::abstract_parallel_dialog::AbstractParallelDialog;
use super::batch_run_tomo_dataset_dialog::BatchRunTomoDatasetDialog;
use super::batch_run_tomo_row::{BasicDirectives, BatchRunTomoRow};
use super::batch_run_tomo_step_panel::BatchRunTomoStepPanel;
use super::batch_run_tomo_table::BatchRunTomoTable;
use super::beveled_border::BeveledBorder;
use super::button_component::ButtonComponent;
use super::check_box::CheckBox;
use super::check_box_efield::CheckBoxEfield;
use super::check_text_field::CheckTextField;
use super::context_menu::ContextMenu;
use super::context_popup::{self, ContextPopup};
use super::etched_border::EtchedBorder;
use super::expand_button::ExpandButton;
use super::expandable::Expandable;
use super::file_text_field::FileTextField;
use super::file_text_field2::FileTextField2;
use super::generic_mouse_adapter::GenericMouseAdapter;
use super::global_expand_button::GlobalExpandButton;
use super::labeled_text_field::LabeledTextField;
use super::panel_header::PanelHeader;
use super::parallel_panel::{self, ParallelPanel};
use super::popup::Popup;
use super::process_interface::ProcessInterface;
use super::radio_button::RadioButton;
use super::result_listener::ResultListener;
use super::series_watcher_panel::SeriesWatcherPanel;
use super::series_watcher_parent::SeriesWatcherParent;
use super::single_line_button::SingleLineButton;
use super::spinner_efield::SpinnerEfield;
use super::swing_component::SwingComponent;
use super::tabbed_pane::TabbedPane;
use super::template_panel::TemplatePanel;
use super::ui_harness;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::batch_run_tomo_manager::BatchRunTomoManager;
use crate::imod::etomo::comscript::batchruntomo_param::{self, BatchruntomoParam};
use crate::imod::etomo::comscript::parallel_param::ParallelParam;
use crate::imod::etomo::comscript::processchunks_param::ProcesschunksParam;
use crate::imod::etomo::comscript::series_watcher_param::{self, SeriesWatcherParam};
use crate::imod::etomo::comscript::split_batch_param::SplitBatchParam;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::jdk::{
    ActionEvent, ActionListener, ButtonGroup, ChangeEvent, ChangeListener, FocusEvent,
    FocusListener, JComponent, MouseEvent, MouseListener,
};
use crate::imod::etomo::logic::batch_tool::{self, TemplateValues};
use crate::imod::etomo::logic::converter;
use crate::imod::etomo::logic::user_env;
use crate::imod::etomo::processing_method_mediator::ProcessingMethodMediator;
use crate::imod::etomo::storage::autodoc::autodoc::Autodoc;
use crate::imod::etomo::storage::autodoc::autodoc_factory;
use crate::imod::etomo::storage::autodoc::read_only_autodoc::ReadOnlyAutodoc;
use crate::imod::etomo::storage::autodoc_filter::AutodocFilter;
use crate::imod::etomo::storage::dataset_file_builder::DatasetFileBuilder;
use crate::imod::etomo::storage::directive_def::DirectiveDef;
use crate::imod::etomo::storage::directive_file::DirectiveFile;
use crate::imod::etomo::storage::directive_file_collection::DirectiveFileCollection;
use crate::imod::etomo::storage::log_file::LogFileError;
use crate::imod::etomo::storage::name_value_pair_list::NameValuePairList;
use crate::imod::etomo::storage::network::Network;
use crate::imod::etomo::storage::parameter_store::ParameterStore;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::batch_run_tomo_meta_data::BatchRunTomoMetaData;
use crate::imod::etomo::r#type::batch_run_tomo_screen_state::BatchRunTomoScreenState;
use crate::imod::etomo::r#type::batch_run_tomo_status::BatchRunTomoStatus;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::directive_file_type::DirectiveFileType;
use crate::imod::etomo::r#type::etomo_autodoc;
use crate::imod::etomo::r#type::file_type;
use crate::imod::etomo::r#type::image_filename_style::ImageFilenameStyle;
use crate::imod::etomo::r#type::processing_method::ProcessingMethod;
use crate::imod::etomo::r#type::queue_mode::QueueMode;
use crate::imod::etomo::r#type::queue_type::QueueType;
use crate::imod::etomo::r#type::run_list::RunList;
use crate::imod::etomo::r#type::run_type::RunType;
use crate::imod::etomo::r#type::series_watcher_meta_data::SeriesWatcherMetaData;
use crate::imod::etomo::r#type::status::StatusRef;
use crate::imod::etomo::r#type::status_change_event::StatusChangeEvent;
use crate::imod::etomo::r#type::status_change_event_sender::{
    StatusChangeEventSender, StatusChangeListeners,
};
use crate::imod::etomo::r#type::status_change_listener::StatusChangeListener;
use crate::imod::etomo::r#type::status_changer::StatusChanger;
use crate::imod::etomo::r#type::table_reference::TableReference;
use crate::imod::etomo::r#type::user_configuration::UserConfiguration;
use crate::imod::etomo::ui::batch_run_tomo_state::BatchRunTomoState;
use crate::imod::etomo::ui::batch_run_tomo_tab::{self, BatchRunTomoTab};
use crate::imod::etomo::ui::browsing_directory::BrowsingDirectory;
use crate::imod::etomo::ui::dataset_info_display::DatasetInfoDisplay;
use crate::imod::etomo::ui::field::Field;
use crate::imod::etomo::ui::field_displayer::FieldDisplayer;
use crate::imod::etomo::ui::field_type::FieldType;
use crate::imod::etomo::ui::field_validation_failed_exception::FieldValidationFailedException;
use crate::imod::etomo::ui::queue_table_data_event::QueueTableDataEvent;
use crate::imod::etomo::ui::queue_table_event::QueueTableEvent;
use crate::imod::etomo::ui::queue_table_listener::QueueTableListener;
use crate::imod::etomo::ui::table_listener::TableListener;
use crate::imod::etomo::ui::ui_component::UIComponent;
use crate::imod::etomo::util::event_queue::{self, EdtRef};
use crate::imod::etomo::util::utilities;
use crate::imod::etomo::util::valid_directory::ValidDirectory;

/// Java private static final `DELIVER_TO_DIRECTORY_LABEL`.
const DELIVER_TO_DIRECTORY_LABEL: &str = "Move all stacks to dataset directories under: ";
/// Java private static final `MAX_GPUS_LABEL`.
const MAX_GPUS_LABEL: &str = "Max # of GPUs to use by one job: ";
/// Java private static final `SPLIT_BATCH_DEFAULT_LABEL`.
const SPLIT_BATCH_DEFAULT_LABEL: &str = "Run multiple batch jobs in parallel";
/// Java private static final `SPLIT_BATCH_CLUSTER_ONLY_LABEL`.
const SPLIT_BATCH_CLUSTER_ONLY_LABEL: &str = "Run multiple batch jobs on cluster";
/// Java private static final `DEFAULT_CORES`.
const DEFAULT_CORES: i32 = 4;
/// Java private static final `MINIMUM_CORES`.
const MINIMUM_CORES: i32 = 2;
/// Java public static final `DIALOG_TYPE`.
pub const DIALOG_TYPE: DialogType = DialogType::BatchRunTomo;
/// Java private static final `NUMBER_OF_JOBS_TO_MAKE_LABEL`.
const NUMBER_OF_JOBS_TO_MAKE_LABEL: &str = "Run up to ";
/// Java private static final `DUAL_SELECTION_MAX_DIVISOR`.
const DUAL_SELECTION_MAX_DIVISOR: i32 = 4;
/// Java private static final `JOBS_FIELD_WIDTH`.
const JOBS_FIELD_WIDTH: i32 = 50;
/// Java private static final `USE_SERIES_WATCHER_LABEL`.
const USE_SERIES_WATCHER_LABEL: &str = "Watch for stacks to run";
/// Java private static final `WATCH_DIRECTORY_LABEL`.
const WATCH_DIRECTORY_LABEL: &str = " in: ";
/// Java private static final `RESET_BUTTON_LABEL` (never read).
#[allow(dead_code)]
const RESET_BUTTON_LABEL: &str = "Reset";
/// Java public static final `TABLE_LABEL`.
pub const TABLE_LABEL: &str = "Datasets";

/// Java `public final class BatchRunTomoDialog implements ActionListener,
/// ResultListener, ChangeListener, Expandable, ProcessInterface, StatusChanger,
/// StatusChangeListener, ContextMenu, BrowsingDirectory, FieldDisplayer,
/// AbstractParallelDialog, FocusListener, SeriesWatcherParent, DatasetInfoDisplay`.
pub struct BatchRunTomoDialog {
    /// Java private final `pnlRoot`.
    pnl_root: Rc<JComponent>,
    /// Java private final `ltfRootName`.
    ltf_root_name: Rc<LabeledTextField>,
    /// Java private final `bgDeliver`.
    #[allow(dead_code)]
    bg_deliver: Rc<ButtonGroup>,
    /// Java private final `rbDeliverOff`.
    rb_deliver_off: Rc<RadioButton>,
    /// Java private final `rbDeliverToDirectory`.
    rb_deliver_to_directory: Rc<RadioButton>,
    /// Java private final `rbDeliverMakeSubDirectory`.
    rb_deliver_make_sub_directory: Rc<RadioButton>,
    /// Java private final `ctfEmailAddress`.
    ctf_email_address: Rc<CheckTextField>,
    /// Java private final `cbCPUMachineList`.
    cb_cpu_machine_list: Rc<CheckBox>,
    /// Java private final `bgGPUMachineList`.
    #[allow(dead_code)]
    bg_gpu_machine_list: Rc<ButtonGroup>,
    /// Java private final `rbGPUMachineListOff`.
    rb_gpu_machine_list_off: Rc<RadioButton>,
    /// Java private final `rbGPUMachineListLocal`.
    rb_gpu_machine_list_local: Rc<RadioButton>,
    /// Java private final `rbGPUMachineList`.
    rb_gpu_machine_list: Rc<RadioButton>,
    /// Java private final `tabbedPane`.
    tabbed_pane: Rc<TabbedPane>,
    /// Java private final `pnlTabs = new JPanel[BatchRunTomoTab.SIZE]`.
    pnl_tabs: RefCell<Vec<Rc<JComponent>>>,
    /// Java private final `tabDisplayed = new boolean[BatchRunTomoTab.SIZE]`.
    tab_displayed: RefCell<[bool; batch_run_tomo_tab::SIZE]>,
    /// Java private final `pnlBatch`.
    pnl_batch: Rc<JComponent>,
    /// Java private final `pnlStacks`.
    pnl_stacks: Rc<JComponent>,
    /// Java private final `pnlDataset`.
    pnl_dataset: Rc<JComponent>,
    /// Java private final `pnlRun`.
    pnl_run: Rc<JComponent>,
    /// Java private final `pnlStacksTable`.
    pnl_stacks_table: Rc<JComponent>,
    /// Java private final `btnRun`.
    btn_run: Rc<SingleLineButton>,
    /// Java private final `pnlDatasetTableBody`.
    pnl_dataset_table_body: Rc<JComponent>,
    /// Java private final `pnlRunTableBody`.
    pnl_run_table_body: Rc<JComponent>,
    /// Java private final `btnReset`.
    btn_reset: Rc<SingleLineButton>,
    /// Java private final `btnPause`.
    btn_pause: Rc<SingleLineButton>,
    /// Java private final `btnResume`.
    btn_resume: Rc<SingleLineButton>,
    /// Java private final `btnClearInputDirectiveFile`.
    btn_clear_input_directive_file: Rc<SingleLineButton>,
    /// Java private final `pnlDatasetTable`.
    pnl_dataset_table: Rc<JComponent>,
    /// Java private final `runFieldDisplayer` (never read).
    #[allow(dead_code)]
    run_field_displayer: Rc<RunFieldDisplayer>,
    /// Java private final `templateValues`.
    template_values: Rc<RefCell<TemplateValues>>,
    /// Java private final `basicDirectives`.
    basic_directives: BasicDirectives,
    /// Java private final `cbSplitBatch`.
    cb_split_batch: Rc<CheckBoxEfield>,
    /// Java private final `lMaxGPUsForOneJobOne`.
    l_max_gpus_for_one_job_one: Rc<JComponent>,
    /// Java private final `pnlMaxGPUsForOneJob`.
    pnl_max_gpus_for_one_job: Rc<JComponent>,
    /// Java private final `pnlMaxGPUsForOneJobOne`.
    pnl_max_gpus_for_one_job_one: Rc<JComponent>,
    /// Java private final `lNumberOfJobsToMake`.
    l_number_of_jobs_to_make: Rc<JComponent>,
    /// Java private final `cbUseSeriesWatcher`.
    cb_use_series_watcher: Rc<CheckBox>,
    /// Java private final `btnStartSeriesWatcher`.
    btn_start_series_watcher: Rc<SingleLineButton>,
    /// Java private final `btnFinishSeriesWatcher`.
    btn_finish_series_watcher: Rc<SingleLineButton>,
    /// Java private final `btnResetSeriesWatcher`.
    btn_reset_series_watcher: Rc<SingleLineButton>,
    /// Java private final `pnlSplitBatch`.
    pnl_split_batch: Rc<JComponent>,
    /// Java private final `pnlRunSettings`.
    pnl_run_settings: Rc<JComponent>,

    /// Java private final `ftfRootDir`.
    ftf_root_dir: Rc<FileTextField2>,
    /// Java private final `ftfInputDirectiveFile`.
    ftf_input_directive_file: Rc<FileTextField2>,
    /// Java private final `templatePanel`.
    template_panel: Rc<TemplatePanel>,
    /// Java private final `ftfDeliverToDirectory`.
    ftf_deliver_to_directory: Rc<FileTextField2>,
    /// Java private final `table` (needs `this`).
    table: OnceCell<Rc<BatchRunTomoTable>>,
    /// Java private final `manager`.
    manager: &'static BatchRunTomoManager,
    /// Java private final `axisID`.
    axis_id: AxisID,
    /// Java private final `datasetDialog` (needs `this`).
    dataset_dialog: OnceCell<Rc<BatchRunTomoDatasetDialog>>,
    /// Java private final `directiveFileCollection`.
    directive_file_collection: Rc<RefCell<DirectiveFileCollection>>,
    /// Java private final `phDatasetTable`.
    ph_dataset_table: Rc<PanelHeader>,
    /// Java private final `phRunTable`.
    ph_run_table: Rc<PanelHeader>,
    /// Java private final `mediator`.
    mediator: Option<Rc<ProcessingMethodMediator>>,
    /// Java private final `stepPanel` (needs `this`).
    step_panel: OnceCell<Rc<BatchRunTomoStepPanel>>,
    /// Java private final `parallelStatusPanel`.
    parallel_status_panel: Option<Rc<JComponent>>,
    /// Java private final `spMaxGPUsForOneJob`.
    sp_max_gpus_for_one_job: Rc<SpinnerEfield>,
    /// Java private final `spNumberOfJobsToMake`.
    sp_number_of_jobs_to_make: Rc<SpinnerEfield>,
    /// Java private final `spQueueNumberOfJobsToMake`.
    sp_queue_number_of_jobs_to_make: Rc<SpinnerEfield>,
    /// Java private final `btnParallelPause`.
    btn_parallel_pause: OnceCell<Option<Rc<SingleLineButton>>>,
    /// Java private final `btnParallelResume`.
    btn_parallel_resume: OnceCell<Option<Rc<SingleLineButton>>>,
    /// Java private final `rbQueueTypeQueue`.
    rb_queue_type_queue: Option<Rc<RadioButton>>,
    /// Java private final `rbQueueTypeNode`.
    rb_queue_type_node: Option<Rc<RadioButton>>,
    /// Java private final `cbQueueSecondaryQueue`.
    cb_queue_secondary_queue: Option<Rc<CheckBox>>,
    /// Java private final `queuesAvailable`.
    queues_available: bool,
    /// Java private final `queueTypeNodeWithGpuAvailable`.
    queue_type_node_with_gpu_available: bool,
    /// Java private final `queueTypeNodeWithoutGpuAvailable`.
    queue_type_node_without_gpu_available: bool,
    /// Java private final `queueWithSingleCPUAvailable`.
    queue_with_single_cpu_available: bool,
    /// Java private final `localGpuAvailable`.
    local_gpu_available: bool,
    /// Java private final `gpuAvailable`.
    gpu_available: bool,
    /// Java private final `totalCPUs`.
    total_cpus: i32,
    /// Java private final `numberQueueCPUsAvailable`.
    number_queue_cpus_available: i32,
    /// Java private final `seriesWatcherPanel` (needs `this`).
    series_watcher_panel: OnceCell<Rc<SeriesWatcherPanel>>,
    /// Java private final `ftfWatchDirectory`.
    ftf_watch_directory: Rc<FileTextField>,
    /// Java private final `pnlRunButtons`.
    pnl_run_buttons: Rc<JComponent>,
    /// Java private final `pnlSeriesWatcherRunButtons`.
    pnl_series_watcher_run_buttons: Rc<JComponent>,
    /// Java private final `pnlResources`.
    pnl_resources: Rc<JComponent>,
    /// Java private final `batchRunTomoState = new BatchRunTomoState(true)`.
    batch_run_tomo_state: RefCell<BatchRunTomoState>,

    /// Java private `curTab`, initially null.
    cur_tab: Cell<Option<BatchRunTomoTab>>,
    /// Java private `listeners`, initially null (shared with the event senders).
    listeners: StatusChangeListeners,
    /// Java private `validbrowsingDirectory`, initially null.
    validbrowsing_directory: RefCell<Option<ValidDirectory>>,
    /// Java private `advancedStartingBatch`, initially null.
    advanced_starting_batch: RefCell<Option<NameValuePairList>>,
    /// Java private `parallelPanel`, initially null.
    parallel_panel: RefCell<Option<Rc<ParallelPanel>>>,
    /// Java private `queueTableListenerArray`, initially null.
    queue_table_listener_array: RefCell<Option<Vec<Rc<dyn QueueTableListener>>>>,
    /// Java private `queueTableDisplayed`, initially false.
    queue_table_displayed: Cell<bool>,
    /// Java private `queueMode`, initially null.
    queue_mode: Cell<Option<QueueMode>>,
    /// Java private `queueTypeQueueEnabled`, initially false.
    queue_type_queue_enabled: Cell<bool>,
    /// Java private `secondaryQueues`, initially false.
    secondary_queues: Cell<bool>,
    /// Java private `killedPaused`, initially false.
    killed_paused: Cell<bool>,
    /// Java `this`.
    this: Weak<BatchRunTomoDialog>,
}

/// Java private final inner class `RunFieldDisplayer implements FieldDisplayer`.
pub struct RunFieldDisplayer {
    /// Java private final `dialog`.
    dialog: Weak<BatchRunTomoDialog>,
}

impl FieldDisplayer for RunFieldDisplayer {
    /// Java `display()`.
    fn display_void(&self) {
        if let Some(dialog) = self.dialog.upgrade() {
            dialog.display_tab(Some(BatchRunTomoTab::Run));
        }
    }

    /// Java `display(UIComponent)`.
    fn display_ui_component(&self, _ui_component: Option<&dyn UIComponent>) {
        self.display_void();
    }
}

/// The dialog as the `ResultListener` `FileTextField2.addResultListener` takes
/// (Java passes `this`).
struct DialogResultListener(Weak<BatchRunTomoDialog>);

impl ResultListener for DialogResultListener {
    fn process_result(&mut self, result_origin: &dyn std::any::Any, init: bool) {
        if let Some(dialog) = self.0.upgrade() {
            dialog.process_result(Some(result_origin), init);
        }
    }
}

impl BatchRunTomoDialog {
    /// Java private `BatchRunTomoDialog(BatchRunTomoManager, AxisID, TableReference,
    /// JPanel)`: the field initializers and the constructor body up to the
    /// sub-objects that need `this` (see [`Self::construct`]).
    fn new(
        manager: &'static BatchRunTomoManager,
        axis_id: AxisID,
        parallel_status_panel: Option<Rc<JComponent>>,
    ) -> Rc<BatchRunTomoDialog> {
        let base_manager: &'static dyn BaseManager = manager;
        let bg_deliver = ButtonGroup::new();
        let bg_gpu_machine_list = ButtonGroup::new();
        // Set the default
        let ftf_watch_directory = FileTextField::get_unlabeled_partial_path_instance(&format!(
            "{USE_SERIES_WATCHER_LABEL}{WATCH_DIRECTORY_LABEL}"
        ));
        let ftf_root_dir =
            FileTextField2::get_alt_layout_instance(Some(base_manager), Some("Location: "));
        let ftf_input_directive_file = FileTextField2::get_alt_layout_instance(
            Some(base_manager),
            Some("Starting directive file: "),
        );
        ftf_input_directive_file.checkpoint();
        let ftf_deliver_to_directory = FileTextField2::get_unlabeled_instance(
            Some(base_manager),
            Some(&format!("{DELIVER_TO_DIRECTORY_LABEL}: ")),
        );
        let directive_file_collection = Rc::new(RefCell::new(
            DirectiveFileCollection::get_batch_instance(base_manager, Some(axis_id)),
        ));
        // `TemplatePanel.getBorderlessInstance(manager, axisID, null, null, null,
        // directiveFileCollection, true, true)`: a null TemplateActionListener, which
        // Swing's addActionListener ignores.
        let null_listener: ActionListener = Rc::new(|_event: &ActionEvent| {});
        let template_panel = TemplatePanel::get_borderless_instance(
            base_manager,
            axis_id,
            null_listener,
            None,
            None,
            Some(Rc::clone(&directive_file_collection)),
            true,
            true,
        );
        let mediator = manager.get_processing_method_mediator(Some(axis_id));
        let property_user_dir = manager.get_property_user_dir();
        let total_cpus =
            Network::get_total_cpus(base_manager, axis_id, property_user_dir.as_deref());
        let sp_number_of_jobs_to_make = SpinnerEfield::get_instance(
            Some(NUMBER_OF_JOBS_TO_MAKE_LABEL),
            DEFAULT_CORES.min(total_cpus),
            MINIMUM_CORES.min(total_cpus),
            total_cpus,
        );
        let sp_queue_number_of_jobs_to_make = SpinnerEfield::get_instance(
            Some(NUMBER_OF_JOBS_TO_MAKE_LABEL),
            DEFAULT_CORES,
            MINIMUM_CORES,
            DEFAULT_CORES,
        );
        let total_gpus =
            Network::get_total_gpus(base_manager, AxisID::Only, property_user_dir.as_deref());
        let sp_max_gpus_for_one_job = SpinnerEfield::get_instance(
            Some(MAX_GPUS_LABEL),
            DEFAULT_CORES.min(total_gpus),
            1.min(total_gpus),
            total_gpus,
        );
        let local_gpu_available = Network::is_local_host_gpu_processing_enabled(
            base_manager,
            axis_id,
            property_user_dir.as_deref(),
        );
        let gpu_available = total_gpus >= 1;
        let number_queue_cpus_available =
            Network::get_total_queue_cpus(base_manager, axis_id, property_user_dir.as_deref());
        let queues_available = Network::has_queues();
        let queue_type_node_without_gpu_available =
            Network::has_queues_mode(Some(QueueMode::NodeWithoutGpu));
        let queue_type_node_with_gpu_available =
            Network::has_queues_mode(Some(QueueMode::NodeWithGpu));
        let queue_with_single_cpu_available =
            Network::has_queues_mode(Some(QueueMode::QueueWithSingleCpu));
        let (rb_queue_type_queue, rb_queue_type_node, cb_queue_secondary_queue) =
            if queues_available {
                let bg_queue_type = ButtonGroup::new();
                (
                    Some(RadioButton::new_string_button_group(
                        Some("Job submits processes to single-core queue"),
                        Some(&bg_queue_type),
                    )),
                    Some(RadioButton::new_string_button_group(
                        Some("Job and its processes run directly on one node"),
                        Some(&bg_queue_type),
                    )),
                    Some(CheckBox::new_string(Some("Use one GPU on secondary queue"))),
                )
            } else {
                (None, None, None)
            };
        Rc::new_cyclic(|this: &Weak<BatchRunTomoDialog>| {
            let expandable: Weak<dyn Expandable> = this.clone();
            BatchRunTomoDialog {
                pnl_root: JComponent::new_panel(),
                ltf_root_name: LabeledTextField::new_field_type_string(
                    FieldType::String,
                    Some("Batchruntomo root name: "),
                ),
                rb_deliver_off: RadioButton::new_string_button_group(
                    Some("Stacks are already in dataset directories"),
                    Some(&bg_deliver),
                ),
                rb_deliver_to_directory: RadioButton::new_string_button_group(
                    Some(DELIVER_TO_DIRECTORY_LABEL),
                    Some(&bg_deliver),
                ),
                rb_deliver_make_sub_directory: RadioButton::new_string_button_group(
                    Some("Move stacks to dataset directories under their current locations"),
                    Some(&bg_deliver),
                ),
                bg_deliver,
                ctf_email_address: CheckTextField::get_instance(
                    FieldType::String,
                    "Email notification: ",
                ),
                cb_cpu_machine_list: CheckBox::new_string(Some("Use multiple cores")),
                rb_gpu_machine_list_off: RadioButton::new_string_button_group(
                    Some("No GPU"),
                    Some(&bg_gpu_machine_list),
                ),
                rb_gpu_machine_list_local: RadioButton::new_string_button_group(
                    Some("Local GPU"),
                    Some(&bg_gpu_machine_list),
                ),
                rb_gpu_machine_list: RadioButton::new_string_button_group(
                    Some("Parallel GPUs"),
                    Some(&bg_gpu_machine_list),
                ),
                bg_gpu_machine_list,
                tabbed_pane: TabbedPane::new(),
                pnl_tabs: RefCell::new(Vec::new()),
                tab_displayed: RefCell::new([false; batch_run_tomo_tab::SIZE]),
                pnl_batch: JComponent::new_panel(),
                pnl_stacks: JComponent::new_panel(),
                pnl_dataset: JComponent::new_panel(),
                pnl_run: JComponent::new_panel(),
                pnl_stacks_table: JComponent::new_panel(),
                btn_run: SingleLineButton::new_string(Some("Run")),
                pnl_dataset_table_body: JComponent::new_panel(),
                pnl_run_table_body: JComponent::new_panel(),
                btn_reset: SingleLineButton::new_string(Some("Reset")),
                btn_pause: SingleLineButton::new_string(Some(parallel_panel::PAUSE_LABEL)),
                btn_resume: SingleLineButton::new_string(Some(parallel_panel::RESUME_LABEL)),
                btn_clear_input_directive_file: SingleLineButton::get_html_instance(Some("Clear")),
                pnl_dataset_table: JComponent::new_panel(),
                run_field_displayer: Rc::new(RunFieldDisplayer {
                    dialog: this.clone(),
                }),
                template_values: Rc::new(RefCell::new(TemplateValues::new())),
                basic_directives: Rc::new(RefCell::new(HashSet::new())),
                cb_split_batch: CheckBoxEfield::get_instance(Some(SPLIT_BATCH_DEFAULT_LABEL)),
                l_max_gpus_for_one_job_one: JComponent::new_label(&format!("{MAX_GPUS_LABEL}1")),
                pnl_max_gpus_for_one_job: JComponent::new_panel(),
                pnl_max_gpus_for_one_job_one: JComponent::new_panel(),
                l_number_of_jobs_to_make: JComponent::new_label(" jobs"),
                cb_use_series_watcher: CheckBox::new_string(Some(USE_SERIES_WATCHER_LABEL)),
                btn_start_series_watcher: SingleLineButton::new_string(Some("Start Watching")),
                btn_finish_series_watcher: SingleLineButton::new_string(Some("Finish")),
                btn_reset_series_watcher: SingleLineButton::new_string(Some("Reset")),
                pnl_split_batch: JComponent::new_panel(),
                pnl_run_settings: JComponent::new_panel(),
                ftf_root_dir,
                ftf_input_directive_file,
                template_panel,
                ftf_deliver_to_directory,
                table: OnceCell::new(),
                manager,
                axis_id,
                dataset_dialog: OnceCell::new(),
                directive_file_collection,
                ph_dataset_table: PanelHeader::get_instance(
                    Some(TABLE_LABEL),
                    Some(expandable.clone()),
                    Some(DialogType::BatchRunTomo),
                ),
                ph_run_table: PanelHeader::get_instance(
                    Some(TABLE_LABEL),
                    Some(expandable),
                    Some(DialogType::BatchRunTomo),
                ),
                mediator,
                step_panel: OnceCell::new(),
                parallel_status_panel,
                sp_max_gpus_for_one_job,
                sp_number_of_jobs_to_make,
                sp_queue_number_of_jobs_to_make,
                btn_parallel_pause: OnceCell::new(),
                btn_parallel_resume: OnceCell::new(),
                rb_queue_type_queue,
                rb_queue_type_node,
                cb_queue_secondary_queue,
                queues_available,
                queue_type_node_with_gpu_available,
                queue_type_node_without_gpu_available,
                queue_with_single_cpu_available,
                local_gpu_available,
                gpu_available,
                total_cpus,
                number_queue_cpus_available,
                series_watcher_panel: OnceCell::new(),
                ftf_watch_directory,
                pnl_run_buttons: JComponent::new_panel(),
                pnl_series_watcher_run_buttons: JComponent::new_panel(),
                pnl_resources: JComponent::new_panel(),
                batch_run_tomo_state: RefCell::new(BatchRunTomoState::new(true)),
                cur_tab: Cell::new(None),
                listeners: Arc::new(Mutex::new(None)),
                validbrowsing_directory: RefCell::new(None),
                advanced_starting_batch: RefCell::new(None),
                parallel_panel: RefCell::new(None),
                queue_table_listener_array: RefCell::new(None),
                queue_table_displayed: Cell::new(false),
                queue_mode: Cell::new(None),
                queue_type_queue_enabled: Cell::new(false),
                secondary_queues: Cell::new(false),
                killed_paused: Cell::new(false),
                this: this.clone(),
            }
        })
    }

    /// The Java constructor's statements that hand out `this`, in source order, and
    /// the rest of the constructor body.
    fn construct(&self, table_reference: Arc<TableReference>) {
        let this = self.this.upgrade().expect("the dialog is alive");
        let series_watcher_parent: Weak<dyn SeriesWatcherParent> = self.this.clone();
        let _ = self
            .series_watcher_panel
            .set(SeriesWatcherPanel::get_instance(
                self.manager,
                self.axis_id,
                Some(DialogType::BatchRunTomo),
                series_watcher_parent.clone(),
            ));
        let _ = self.table.set(BatchRunTomoTable::get_instance(
            self.manager,
            self.this.clone(),
            Rc::clone(&self.basic_directives),
            table_reference,
            Rc::clone(&self.pnl_root),
            series_watcher_parent.clone(),
        ));
        let browsing_directory: Weak<dyn BrowsingDirectory> = self.this.clone();
        let _ = self
            .dataset_dialog
            .set(BatchRunTomoDatasetDialog::get_global_instance(
                self.manager,
                self.this.clone(),
                Some(Rc::clone(&self.template_values)),
                Some(Rc::clone(&self.basic_directives)),
                Some(browsing_directory),
            ));
        let _ = self.step_panel.set(BatchRunTomoStepPanel::get_instance(
            self.manager,
            self.axis_id,
            Rc::downgrade(self.table()),
            series_watcher_parent,
        ));
        let main_panel = self
            .manager
            .get_main_panel()
            .expect("the batchruntomo dialog is built with a main panel");
        let main_panel = main_panel.main_panel();

        // Load screen state
        let screen_state = self.manager.get_batch_run_tomo_screen_state();
        match ParameterStore::get_instance_manager(
            Some(self.manager),
            Some(AxisID::First),
            self.manager.get_param_file(),
        ) {
            Ok(Some(mut param_store)) => param_store.load(screen_state),
            Ok(None) | Err(LogFileError::Lock(_)) => {}
            Err(e) => eprintln!("{e}"),
        }
        main_panel.create_parallel_panel(AxisID::Only);

        *self.parallel_panel.borrow_mut() = main_panel.get_parallel_panel(self.axis_id);
        let _ = self
            .btn_parallel_pause
            .set(main_panel.get_parallel_pause_button(AxisID::Only));
        let _ = self
            .btn_parallel_resume
            .set(main_panel.get_parallel_resume_button(AxisID::Only));
        if let Some(mediator) = &self.mediator {
            mediator.register_process_interface(this as Rc<dyn ProcessInterface>);
        }
    }

    /// Java public static `getInstance(BatchRunTomoManager, AxisID, TableReference,
    /// JPanel)`.
    pub fn get_instance(
        manager: &'static BatchRunTomoManager,
        axis_id: AxisID,
        table_reference: Arc<TableReference>,
        parallel_status_panel: Option<Rc<JComponent>>,
    ) -> Rc<BatchRunTomoDialog> {
        let instance = BatchRunTomoDialog::new(manager, axis_id, parallel_status_panel);
        instance.construct(table_reference);
        instance.create_panel();
        instance.set_tooltips();
        instance
    }

    fn base_manager(&self) -> &'static dyn BaseManager {
        self.manager
    }

    fn table(&self) -> &Rc<BatchRunTomoTable> {
        self.table.get().expect("table")
    }

    fn step_panel(&self) -> &Rc<BatchRunTomoStepPanel> {
        self.step_panel.get().expect("stepPanel")
    }

    fn series_watcher_panel(&self) -> &Rc<SeriesWatcherPanel> {
        self.series_watcher_panel.get().expect("seriesWatcherPanel")
    }

    fn btn_parallel_pause(&self) -> Option<Rc<SingleLineButton>> {
        self.btn_parallel_pause.get().cloned().flatten()
    }

    fn btn_parallel_resume(&self) -> Option<Rc<SingleLineButton>> {
        self.btn_parallel_resume.get().cloned().flatten()
    }

    fn this_rc(&self) -> Rc<BatchRunTomoDialog> {
        self.this.upgrade().expect("the dialog is alive")
    }

    /// This dialog as the `ActionListener` Java registers as `this`.
    fn action_listener(&self) -> ActionListener {
        let this = self.this.clone();
        Rc::new(move |event: &ActionEvent| {
            if let Some(dialog) = this.upgrade() {
                dialog.action_performed(Some(event));
            }
        })
    }

    /// Java `getDialogType()`.
    pub fn get_dialog_type(&self) -> DialogType {
        DIALOG_TYPE
    }

    /// Java package-private `hasDual()`.
    pub fn has_dual(&self) -> bool {
        self.table().has_dual()
    }

    /// Java private `createPanel()`.
    fn create_panel(&self) {
        // local panels
        let pnl_root_name = JComponent::new_panel();
        let pnl_deliver = JComponent::new_panel();
        let pnl_templates = JComponent::new_panel();
        let pnl_email = JComponent::new_panel();
        let pnl_deliver_to_directory = JComponent::new_panel();
        let pnl_input_directive_file = JComponent::new_panel();
        let pnl_number_of_jobs_to_make = JComponent::new_panel();
        let pnl_parallel_settings = JComponent::new_panel();
        let pnl_split_batch_check_box = JComponent::new_panel();
        let pnl_split_batch_outer = JComponent::new_panel();
        let pnl_run_table = JComponent::new_panel();
        let pnl_use_series_watcher = JComponent::new_panel();
        //
        let (pnl_queue_mode_outer, pnl_queue_mode) = if self.rb_queue_type_queue.is_some() {
            (Some(JComponent::new_panel()), Some(JComponent::new_panel()))
        } else {
            (None, None)
        };
        // init
        {
            let mut basic_directives = self.basic_directives.borrow_mut();
            basic_directives.insert(DirectiveDef::NAME);
            basic_directives.insert(DirectiveDef::SCOPE_TEMPLATE);
            basic_directives.insert(DirectiveDef::SYSTEM_TEMPLATE);
            basic_directives.insert(DirectiveDef::USER_TEMPLATE);
            basic_directives.insert(DirectiveDef::DATASET_DIRECTORY);
        }
        self.template_panel.set_field_highlight();
        self.ftf_input_directive_file.set_absolute_path(true);
        self.ftf_input_directive_file.set_text_entry_policy(false);
        self.ftf_input_directive_file
            .set_file_filter(Some(Rc::new(AutodocFilter::new())));
        self.ftf_deliver_to_directory.set_absolute_path(true);
        self.ftf_deliver_to_directory
            .set_file_selection_mode(super::file_chooser::DIRECTORIES_ONLY);
        self.ftf_deliver_to_directory.set_required(true);
        // Swing layout: btnReset, btnPause, btnRun (to btnPause's preferred size),
        // btnResume, btnStartSeriesWatcher and btnFinishSeriesWatcher
        // setToPreferredSize(); tabbedPane.setTabLayoutPolicy(SCROLL_TAB_LAYOUT).
        self.ftf_root_dir.set_absolute_path(true);
        self.ftf_root_dir
            .set_file_selection_mode(super::file_chooser::DIRECTORIES_ONLY);
        // `new File(System.getProperty("user.dir")).getAbsolutePath()`.
        let user_dir = std::env::var("PWD").ok().unwrap_or_default();
        self.ftf_root_dir
            .set_text_string(Some(&utilities::java_io_file_get_absolute_path(&user_dir)));
        // Setting the origin to the run directory. The run directory can be changed
        // until the batch is first run.
        self.ftf_input_directive_file
            .set_origin_file(
                crate::imod::etomo::ui::swing::file_text_field_interface::FileTextFieldInterface::get_file(
                    &*self.ftf_root_dir,
                )
                .as_deref(),
            );
        self.ftf_input_directive_file
            .set_origin_reference(Some(Rc::clone(&self.ftf_root_dir)));
        self.ctf_email_address.set_required(true);
        self.ctf_email_address.set_text_preferred_width(300);
        // ftfWatchDirectory.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
        // ftfWatchDirectory.setTextEntryPolicy(false);
        // ftfWatchDirectory.setAbsolutePath(true);
        let property_user_dir = self.manager.get_property_user_dir();
        crate::imod::etomo::ui::swing::file_text_field_interface::FileTextFieldInterface::set_file(
            &*self.ftf_watch_directory,
            Some(PathBuf::from(property_user_dir.clone().unwrap_or_default())),
        );
        self.ftf_watch_directory.add_action(
            property_user_dir.as_deref(),
            Some(Rc::clone(&self.pnl_root)),
            super::file_chooser::DIRECTORIES_ONLY,
        );
        self.ftf_watch_directory.set_use_prev_chooser_dir(true);
        self.ftf_watch_directory.set_text_preferred_width(595);
        self.ftf_watch_directory.set_required(true);
        self.ftf_watch_directory.set_file_must_exist(true);
        self.sp_number_of_jobs_to_make
            .set_preferred_width(JOBS_FIELD_WIDTH);
        self.sp_queue_number_of_jobs_to_make.set_visible(false);
        self.sp_queue_number_of_jobs_to_make
            .set_preferred_width(JOBS_FIELD_WIDTH);
        self.sp_max_gpus_for_one_job
            .set_preferred_width(JOBS_FIELD_WIDTH);
        if self.queue_with_single_cpu_available {
            self.queue_type_queue_enabled.set(true);
            if let Some(rb_queue_type_queue) = &self.rb_queue_type_queue {
                rb_queue_type_queue.set_selected_boolean(true);
            }
        }
        if (self.queue_type_node_without_gpu_available || self.queue_type_node_with_gpu_available)
            && !self.queue_type_queue_enabled.get()
            && let Some(rb_queue_type_node) = &self.rb_queue_type_node
        {
            rb_queue_type_node.set_selected_boolean(true);
        }
        // Secondary queues only work with primary queues that don't have a GPU (modes 1
        // and 2c).
        if Network::has_secondary_queues()
            && (self.queue_type_queue_enabled.get() || self.queue_type_node_without_gpu_available)
        {
            self.secondary_queues.set(true);
        }
        // Make sure that the machine lists from the batchruntomo .com file get loaded.
        self.cb_cpu_machine_list.set_selected_boolean(true);
        self.rb_gpu_machine_list.set_selected_boolean(true);
        self.rb_deliver_off.set_selected_boolean(true);
        // Swing layout: btnClearInputDirectiveFile.setToPreferredSize().
        self.pnl_max_gpus_for_one_job_one.set_visible(false);
        // root panel
        // Swing layout: pnlRoot Y_AXIS BoxLayout.
        self.pnl_root.set_border_title(
            BeveledBorder::new(Some("Batchruntomo Interface"))
                .get_border()
                .get_title()
                .as_deref(),
        );
        self.pnl_root.add(&self.tabbed_pane.get_component());
        // tabbedPane
        for i in 0..batch_run_tomo_tab::SIZE {
            self.tab_displayed.borrow_mut()[i] = false;
            let pnl_tab = JComponent::new_panel();
            self.pnl_tabs.borrow_mut().push(Rc::clone(&pnl_tab));
            let tab = BatchRunTomoTab::get_instance(i as i32);
            self.tabbed_pane
                .add_tab_string_component(tab.get_title(), &pnl_tab);
        }
        // Batch
        // Swing layout: pnlBatch Y_AXIS BoxLayout; rigid areas x0_y20, x0_y6, x0_y15.
        self.pnl_batch.set_border_title(
            EtchedBorder::new(Some("Batch Setup Parameters"))
                .get_border()
                .get_title()
                .as_deref(),
        );
        self.pnl_batch.add(&pnl_deliver);
        self.pnl_batch.add(&pnl_input_directive_file);
        self.pnl_batch.add(&pnl_templates);
        self.pnl_batch.add(&pnl_root_name);
        // Stacks
        // Swing layout: pnlStacks Y_AXIS BoxLayout, etched border.
        self.pnl_stacks.add(&self.pnl_stacks_table);
        // StacksTable
        // Swing layout: pnlStacksTable Y_AXIS BoxLayout.
        self.pnl_stacks_table.set_border_title(
            EtchedBorder::new(Some(TABLE_LABEL))
                .get_border()
                .get_title()
                .as_deref(),
        );
        // panel created on tab change
        // Dataset
        // Swing layout: pnlDataset Y_AXIS BoxLayout, etched border.
        self.pnl_dataset
            .add(&self.get_dataset_dialog().get_component());
        self.pnl_dataset.add(&self.pnl_dataset_table);
        // UseSeriesWatcher
        // Swing layout: pnlUseSeriesWatcher X_AXIS BoxLayout, trailing glue, left
        // aligned.
        pnl_use_series_watcher.add(&self.cb_use_series_watcher.get_component());
        pnl_use_series_watcher.add(&self.ftf_watch_directory.get_container());
        // Run
        // Swing layout: pnlRun Y_AXIS BoxLayout; rigid areas x0_y2, x0_y20, x0_y5,
        // x0_y15, a 25 pixel strut and x0_y10 between the parts below.
        self.pnl_run.set_border_title(None);
        if let Some(parallel_status_panel) = &self.parallel_status_panel {
            self.pnl_run.add(parallel_status_panel);
        }
        self.pnl_run.add(&self.pnl_resources);
        // if (EtomoDirector.INSTANCE.getArguments().isNewstuff()) {
        self.pnl_run.add(&pnl_use_series_watcher);
        // }
        self.pnl_run.add(&self.pnl_run_settings);
        self.pnl_run.add(&pnl_email);
        self.pnl_run.add(&self.pnl_run_buttons);
        self.pnl_run.add(&self.pnl_series_watcher_run_buttons);
        self.pnl_run.add(&pnl_run_table);
        // RunTable
        // Swing layout: pnlRunTable Y_AXIS BoxLayout, etched border, rigid area
        // x0_y2.
        pnl_run_table.add(&self.ph_run_table.get_container());
        pnl_run_table.add(&self.pnl_run_table_body);
        // RunTableBody
        // Swing layout: pnlRunTableBody X_AXIS BoxLayout.
        // Resources
        // Swing layout: pnlResources X_AXIS BoxLayout with glue between the parts.
        self.pnl_resources.add(&pnl_parallel_settings);
        self.pnl_resources.add(&pnl_split_batch_outer);
        if let Some(pnl_queue_mode_outer) = &pnl_queue_mode_outer {
            self.pnl_resources.add(pnl_queue_mode_outer);
        }
        // DatasetTable
        // Swing layout: pnlDatasetTable Y_AXIS BoxLayout, etched border, rigid area
        // x0_y2.
        self.pnl_dataset_table
            .add(&self.ph_dataset_table.get_container());
        self.pnl_dataset_table.add(&self.pnl_dataset_table_body);
        // DatasetTableBody
        // Swing layout: pnlDatasetTableBody X_AXIS BoxLayout.
        // Email
        // Swing layout: pnlEmail X_AXIS BoxLayout.
        pnl_email.add(&self.ctf_email_address.get_component());
        // pnlSplitBatchOuter
        // Swing layout: Y_AXIS BoxLayout, trailing vertical glue.
        pnl_split_batch_outer.add(&self.pnl_split_batch);
        // SplitBatch
        // Swing layout: pnlSplitBatch Y_AXIS BoxLayout.
        self.pnl_split_batch.set_border_title(
            EtchedBorder::new(Some("Multiple Jobs"))
                .get_border()
                .get_title()
                .as_deref(),
        );
        self.pnl_split_batch.add(&pnl_split_batch_check_box);
        self.pnl_split_batch.add(&pnl_number_of_jobs_to_make);
        self.pnl_split_batch.add(&self.pnl_max_gpus_for_one_job);
        self.pnl_split_batch.add(&self.pnl_max_gpus_for_one_job_one);
        // QueueModeOuter
        if let (Some(pnl_queue_mode_outer), Some(pnl_queue_mode)) =
            (&pnl_queue_mode_outer, &pnl_queue_mode)
        {
            // Swing layout: Y_AXIS BoxLayout, trailing vertical glue.
            pnl_queue_mode_outer.add(pnl_queue_mode);
        }
        // QueueMode
        if let Some(pnl_queue_mode) = &pnl_queue_mode {
            // Swing layout: Y_AXIS BoxLayout.
            pnl_queue_mode.set_border_title(
                EtchedBorder::new(Some("How a Batch Job Should Run Processes"))
                    .get_border()
                    .get_title()
                    .as_deref(),
            );
            if let (
                Some(rb_queue_type_queue),
                Some(rb_queue_type_node),
                Some(cb_queue_secondary_queue),
            ) = (
                &self.rb_queue_type_queue,
                &self.rb_queue_type_node,
                &self.cb_queue_secondary_queue,
            ) {
                pnl_queue_mode.add(&rb_queue_type_queue.get_component());
                pnl_queue_mode.add(&rb_queue_type_node.get_component());
                pnl_queue_mode.add(&cb_queue_secondary_queue.get_component());
            }
        }
        // NumberOfJobsToMakePanel
        // Swing layout: X_AXIS BoxLayout, rigid area x1_y0, trailing glue.
        pnl_number_of_jobs_to_make.add(&self.sp_number_of_jobs_to_make.get_component());
        if self.queues_available {
            pnl_number_of_jobs_to_make.add(&self.sp_queue_number_of_jobs_to_make.get_component());
        }
        pnl_number_of_jobs_to_make.add(&self.l_number_of_jobs_to_make);
        // SplitBatchCheckBox
        // Swing layout: X_AXIS BoxLayout, trailing glue.
        pnl_split_batch_check_box.add(&self.cb_split_batch.get_component());
        // MaxGPUsForOneJob
        // Swing layout: X_AXIS BoxLayout, trailing glue.
        self.pnl_max_gpus_for_one_job
            .add(&self.sp_max_gpus_for_one_job.get_component());
        // MaxGPUsForOneJobOne
        // Swing layout: X_AXIS BoxLayout, trailing glue.
        self.pnl_max_gpus_for_one_job_one
            .add(&self.l_max_gpus_for_one_job_one);
        // RunSettings
        // Swing layout: pnlRunSettings X_AXIS BoxLayout.
        // RunButton
        // Swing layout: pnlRunButtons X_AXIS BoxLayout with glue between the buttons.
        self.pnl_run_buttons
            .add(&SwingComponent::get_component(&*self.btn_run));
        self.pnl_run_buttons
            .add(&SwingComponent::get_component(&*self.btn_pause));
        if let Some(btn_parallel_pause) = self.btn_parallel_pause() {
            self.pnl_run_buttons
                .add(&SwingComponent::get_component(&*btn_parallel_pause));
        }
        self.pnl_run_buttons
            .add(&SwingComponent::get_component(&*self.btn_resume));
        if let Some(btn_parallel_resume) = self.btn_parallel_resume() {
            self.pnl_run_buttons
                .add(&SwingComponent::get_component(&*btn_parallel_resume));
        }
        self.pnl_run_buttons
            .add(&SwingComponent::get_component(&*self.btn_reset));
        // SeriesWatcherRunButtons
        // Swing layout: X_AXIS BoxLayout with glue between the buttons.
        self.pnl_series_watcher_run_buttons
            .add(&SwingComponent::get_component(
                &*self.btn_start_series_watcher,
            ));
        self.pnl_series_watcher_run_buttons
            .add(&SwingComponent::get_component(
                &*self.btn_finish_series_watcher,
            ));
        self.pnl_series_watcher_run_buttons
            .add(&SwingComponent::get_component(
                &*self.btn_reset_series_watcher,
            ));
        // ParallelSettings
        // Swing layout: Y_AXIS BoxLayout.
        pnl_parallel_settings.set_border_title(
            EtchedBorder::new(Some("Resources to Use"))
                .get_border()
                .get_title()
                .as_deref(),
        );
        pnl_parallel_settings.add(&self.cb_cpu_machine_list.get_component());
        pnl_parallel_settings.add(&self.rb_gpu_machine_list_off.get_component());
        pnl_parallel_settings.add(&self.rb_gpu_machine_list_local.get_component());
        pnl_parallel_settings.add(&self.rb_gpu_machine_list.get_component());
        // RootName
        // Swing layout: Y_AXIS BoxLayout, rigid area x0_y2.
        pnl_root_name.set_border_title(
            EtchedBorder::new(Some("Batchruntomo Project Files"))
                .get_border()
                .get_title()
                .as_deref(),
        );
        pnl_root_name.add(&self.ltf_root_name.get_component());
        pnl_root_name.add(&self.ftf_root_dir.get_root_panel());
        // Templates
        // Swing layout: X_AXIS BoxLayout, trailing glue.
        pnl_templates.add(&self.template_panel.get_component());
        // Deliver
        // Swing layout: Y_AXIS BoxLayout.
        pnl_deliver.add(&self.rb_deliver_off.get_component());
        pnl_deliver.add(&pnl_deliver_to_directory);
        pnl_deliver.add(&self.rb_deliver_make_sub_directory.get_component());
        // InputDirectiveFile
        // Swing layout: X_AXIS BoxLayout, a 100 pixel rigid area, trailing glue.
        pnl_input_directive_file.add(&self.ftf_input_directive_file.get_root_panel());
        pnl_input_directive_file.add(&SwingComponent::get_component(
            &*self.btn_clear_input_directive_file,
        ));
        // DeliverToDirectory
        // Swing layout: X_AXIS BoxLayout.
        pnl_deliver_to_directory.add(&self.rb_deliver_to_directory.get_component());
        pnl_deliver_to_directory.add(&self.ftf_deliver_to_directory.get_root_panel());
        // align
        // Swing layout: UIUtilities.alignComponentsX(pnlBatch, pnlRoot, pnlDeliver,
        // LEFT_ALIGNMENT).

        // update
        let ftf_root_dir: &dyn std::any::Any = &*self.ftf_root_dir;
        self.process_result(Some(ftf_root_dir), true);
        self.state_changed(None);
        let status = self
            .batch_run_tomo_state
            .borrow()
            .get_batch_run_tomo_status();
        self.status_changed_status(status.map(StatusRef::BatchRunTomoStatus));
        if let Some(mediator) = &self.mediator {
            mediator.set_method_process_interface_processing_method(
                &(self.this_rc() as Rc<dyn ProcessInterface>),
                self.get_processing_method(),
            );
        }
        self.update_display_void();
    }

    /// Java `retrieveScreenStateFromDialog(BatchRunTomoScreenState)`.
    pub fn retrieve_screen_state_from_dialog(&self, screen_state: &BatchRunTomoScreenState) {
        self.ph_dataset_table
            .get_state(Some(screen_state.get_dataset_header_state()));
        self.ph_run_table
            .get_state(Some(screen_state.get_run_header_state()));
        self.series_watcher_panel()
            .retrieve_screen_state_from_dialog(screen_state);
    }

    /// Java `applyScreenStateToDialog(BatchRunTomoScreenState)`.
    pub fn apply_screen_state_to_dialog(&self, screen_state: &BatchRunTomoScreenState) {
        self.ph_dataset_table
            .set_state(Some(screen_state.get_dataset_header_state()));
        self.ph_run_table
            .set_state(Some(screen_state.get_run_header_state()));
        self.series_watcher_panel()
            .apply_screen_state_to_dialog(screen_state);
    }

    /// Java private `buildRunSettingsPanel()`.
    fn build_run_settings_panel(&self) {
        let series_watcher = self.cb_use_series_watcher.is_selected();
        if series_watcher {
            self.pnl_run_settings
                .add(&self.series_watcher_panel().get_component());
            // Swing layout: horizontal strut 2.
        }
        // Swing layout: horizontal glue (and a strut of 2 when watching).
        self.pnl_run_settings
            .add(&self.step_panel().get_component());
        ui_harness::with(|harness| harness.pack_base_manager(Some(self.base_manager())));
    }

    /// Java private `addListeners()`.
    fn add_listeners(&self) {
        let listener = self.action_listener();
        self.cb_use_series_watcher
            .add_action_listener(Some(listener.clone()));
        self.cb_use_series_watcher
            .add_action_listener(Some(self.series_watcher_panel().get_action_listener()));
        let step_panel = Rc::downgrade(self.step_panel());
        self.cb_use_series_watcher.add_action_listener(Some(Rc::new(
            move |event: &ActionEvent| {
                if let Some(step_panel) = step_panel.upgrade() {
                    step_panel.action_performed(event);
                }
            },
        )));
        self.cb_use_series_watcher
            .add_action_listener(Some(self.table().action_listener()));
        self.btn_start_series_watcher
            .add_action_listener(listener.clone());
        self.btn_finish_series_watcher
            .add_action_listener(listener.clone());
        self.btn_reset_series_watcher
            .add_action_listener(listener.clone());
        self.template_panel.add_listeners();
        self.rb_deliver_off.add_action_listener(listener.clone());
        self.rb_deliver_to_directory
            .add_action_listener(listener.clone());
        self.rb_deliver_make_sub_directory
            .add_action_listener(listener.clone());
        self.template_panel.add_action_listener(listener.clone());
        self.cb_cpu_machine_list
            .add_action_listener(Some(listener.clone()));
        self.rb_gpu_machine_list_off
            .add_action_listener(listener.clone());
        self.rb_gpu_machine_list_local
            .add_action_listener(listener.clone());
        self.rb_gpu_machine_list
            .add_action_listener(listener.clone());
        self.btn_run.add_action_listener(listener.clone());
        self.ftf_input_directive_file
            .add_result_listener(Some(Rc::new(RefCell::new(DialogResultListener(
                self.this.clone(),
            )))));
        let this = self.this.clone();
        let change_listener: ChangeListener = Rc::new(move |event: &ChangeEvent| {
            if let Some(dialog) = this.upgrade() {
                dialog.state_changed(Some(event));
            }
        });
        self.tabbed_pane
            .get_component()
            .add_change_listener(change_listener.clone());
        self.table().set_table_listener(Some(
            Rc::clone(self.get_dataset_dialog()) as Rc<dyn TableListener>
        ));
        self.btn_reset.add_action_listener(listener.clone());
        self.ctf_email_address.add_action_listener(listener.clone());
        self.btn_resume.add_action_listener(listener.clone());
        self.btn_pause.add_action_listener(listener.clone());
        self.btn_clear_input_directive_file
            .add_action_listener(listener.clone());
        self.cb_split_batch.add_action_listener(listener.clone());
        self.sp_queue_number_of_jobs_to_make
            .add_change_listener(change_listener);
        let this = self.this.clone();
        let focus_listener: FocusListener = Rc::new(move |event: &FocusEvent| {
            if let Some(dialog) = this.upgrade() {
                if event.gained {
                    dialog.focus_gained(Some(event));
                } else {
                    dialog.focus_lost(Some(event));
                }
            }
        });
        self.sp_queue_number_of_jobs_to_make
            .add_focus_listener(focus_listener);
        if let (
            Some(rb_queue_type_queue),
            Some(rb_queue_type_node),
            Some(cb_queue_secondary_queue),
        ) = (
            &self.rb_queue_type_queue,
            &self.rb_queue_type_node,
            &self.cb_queue_secondary_queue,
        ) {
            rb_queue_type_queue.add_action_listener(listener.clone());
            rb_queue_type_node.add_action_listener(listener.clone());
            cb_queue_secondary_queue.add_action_listener(Some(listener));
        }
        // This dialog can set the global status to open.
        self.add_status_change_listener(Some(
            Rc::clone(self.step_panel()) as Rc<dyn StatusChangeListener>
        ));
        let changer: Rc<dyn StatusChanger> = self.this_rc();
        self.table().msg_status_changer_started(&changer);
        // The step panel needs to listen for changes to the earliestRunStep.
        let listener: Rc<dyn StatusChangeListener> = self.this_rc();
        self.table()
            .add_status_change_listener(Some(Rc::clone(&listener)));
        self.table()
            .add_status_change_listener_to_row_list(Some(listener));
        let context_menu: Weak<dyn ContextMenu> = self.this.clone();
        let mouse_adapter: Rc<dyn MouseListener> = GenericMouseAdapter::new(context_menu);
        self.tabbed_pane
            .get_component()
            .add_mouse_listener(mouse_adapter);
    }

    /// Java package-private `getBasicDirectives()`.
    pub fn get_basic_directives(&self) -> BasicDirectives {
        Rc::clone(&self.basic_directives)
    }

    /// Java `msgStatusChangerStarted(StatusChanger, boolean)`.
    pub fn msg_status_changer_started(&self, changer: &Rc<dyn StatusChanger>, table_only: bool) {
        // Listen to the monitor.
        if !table_only {
            changer
                .add_status_change_listener(Some(self.this_rc() as Rc<dyn StatusChangeListener>));
            changer.add_status_change_listener(Some(
                Rc::clone(self.step_panel()) as Rc<dyn StatusChangeListener>
            ));
        }
        self.table().msg_status_changer_started(changer);
    }

    /// Java `getContainer()`.
    pub fn get_container(&self) -> Rc<JComponent> {
        Rc::clone(&self.pnl_root)
    }

    /// Java `msgLoadDone()`.
    pub fn msg_load_done(&self) {
        self.add_listeners();
        self.table().set_frame(true);
    }

    /// Java package-private `getTemplateValues()`.
    pub fn get_template_values(&self) -> Rc<RefCell<TemplateValues>> {
        Rc::clone(&self.template_values)
    }

    /// Java package-private `getBrowsingDirectory()`.  The dialog replaces the
    /// manager as the browsing directory.
    pub fn get_browsing_directory(&self) -> Weak<dyn BrowsingDirectory> {
        self.this.clone()
    }

    /// Java package-private `getDatasetDialog()`.
    pub fn get_dataset_dialog(&self) -> &Rc<BatchRunTomoDatasetDialog> {
        self.dataset_dialog.get().expect("datasetDialog")
    }

    /// Java package-private `isSeriesWatcherAOnly()`.
    pub fn is_series_watcher_a_only(&self) -> bool {
        self.is_series_watcher_on() && self.series_watcher_panel().is_a_only()
    }

    /// Java package-private `setBrowsingDir(String)`.
    pub fn set_browsing_dir_string(&self, input: Option<&str>) {
        let mut validbrowsing_directory = self.validbrowsing_directory.borrow_mut();
        if validbrowsing_directory.is_none() && input.is_some_and(|input| {
            !crate::imod::etomo::r#type::const_etomo_number::java_lang_string_matches_whitespace(
                input,
            )
        }) {
            *validbrowsing_directory = Some(ValidDirectory::new(Some(self.base_manager())));
        }
        if let Some(validbrowsing_directory) = validbrowsing_directory.as_mut() {
            validbrowsing_directory.set_string(input);
        }
    }

    /// Java `setParameters(BatchRunTomoMetaData, String, boolean, boolean)`.
    /// `onlyStackIDDatasetDialog`: only loading the dataset dialog attached to the row
    /// with this stackID; `onlyAdvancedDatasetDialog`: only loading an advanced
    /// dataset dialog.
    pub fn set_parameters_meta_data(
        &self,
        meta_data: &BatchRunTomoMetaData,
        only_stack_id_dataset_dialog: Option<&str>,
        only_advanced_dataset_dialog: bool,
        init: bool,
    ) {
        let init_entire_dialog =
            only_stack_id_dataset_dialog.is_none() && !only_advanced_dataset_dialog;
        let only_global_dataset_dialog =
            only_stack_id_dataset_dialog.is_none() && only_advanced_dataset_dialog;
        if init_entire_dialog {
            self.set_browsing_dir_string(Some(&meta_data.get_browsing_directory()));
            self.ftf_deliver_to_directory
                .set_text_string(Some(&meta_data.get_deliver_to_directory()));
        }
        if !meta_data.is_root_name_null() {
            if init_entire_dialog {
                self.ltf_root_name
                    .set_text_string(Some(&meta_data.get_root_name()));
                self.ftf_root_dir
                    .set_text_string(self.manager.get_property_user_dir().as_deref());
                self.ftf_input_directive_file
                    .set_text_string(Some(&meta_data.get_input_directive_file()));
                self.ftf_input_directive_file.checkpoint();
                if meta_data.is_split_batch_set() {
                    self.cb_split_batch.set_selected(meta_data.is_split_batch());
                }
                if meta_data.is_max_gpus_for_one_job_set() {
                    self.sp_max_gpus_for_one_job
                        .set_text_string(Some(&meta_data.get_max_gpus_for_one_job()));
                }
                if meta_data.is_number_of_jobs_to_make_set() {
                    self.sp_number_of_jobs_to_make
                        .set_text_string(Some(&meta_data.get_number_of_jobs_to_make()));
                }
                if meta_data.is_queue_number_of_jobs_to_make_set() {
                    self.sp_queue_number_of_jobs_to_make
                        .set_text_string(Some(&meta_data.get_queue_number_of_jobs_to_make()));
                }
                if let (
                    Some(rb_queue_type_queue),
                    Some(rb_queue_type_node),
                    Some(cb_queue_secondary_queue),
                ) = (
                    &self.rb_queue_type_queue,
                    &self.rb_queue_type_node,
                    &self.cb_queue_secondary_queue,
                ) {
                    let queue_type = meta_data.get_queue_type();
                    if queue_type == Some(QueueType::Queue) {
                        rb_queue_type_queue.set_selected_boolean(true);
                    } else if queue_type == Some(QueueType::Node) {
                        rb_queue_type_node.set_selected_boolean(true);
                    }
                    cb_queue_secondary_queue
                        .set_selected_boolean(meta_data.is_use_secondary_queue());
                }
                self.cb_cpu_machine_list
                    .set_selected_boolean(meta_data.is_use_cpu_machine_list());
                let use_gpu_machine_list_parallel = meta_data.get_use_gpu_machine_list_parallel();
                match use_gpu_machine_list_parallel {
                    None => self.rb_gpu_machine_list_off.set_selected_boolean(true),
                    Some(value) if !value.is() => {
                        self.rb_gpu_machine_list_local.set_selected_boolean(true)
                    }
                    Some(_) => self.rb_gpu_machine_list.set_selected_boolean(true),
                }
            }
            if !only_global_dataset_dialog {
                self.table().set_parameters_meta_data(
                    meta_data,
                    only_stack_id_dataset_dialog,
                    only_advanced_dataset_dialog,
                    init,
                );
            }
            if init_entire_dialog {
                self.step_panel().set_parameters_meta_data(meta_data);
            }
            if init_entire_dialog {
                self.get_dataset_dialog()
                    .set_parameters_dataset_meta_data(&meta_data.get_dataset_meta_data());
                self.apply_screen_state_to_dialog(self.manager.get_batch_run_tomo_screen_state());
                self.disable_root_fields();
                self.status_changed_status(
                    meta_data.get_status().map(StatusRef::BatchRunTomoStatus),
                );
            }
        } else if init_entire_dialog {
            self.ltf_root_name.set_text_string(Some(&format!(
                "{}{}",
                batchruntomo_param::ROOT_NAME_PREFIX,
                utilities::get_date_time_stamp_root_name()
            )));
        }
        if init_entire_dialog && meta_data.is_delivered() {
            self.disable_delivery_fields();
        }
        // SeriesWatcher
        self.cb_use_series_watcher
            .set_selected_boolean(meta_data.is_use_series_watcher());
        self.build_run_settings_panel();
        self.ftf_watch_directory
            .set_text_string(Some(&meta_data.get_watch_directory()));
        self.series_watcher_panel()
            .set_parameters_meta_data(meta_data);
        self.update_display_void();
        self.table().update_row_display();
        self.send_queue_table_events();
    }

    /// Java `setParameters(SeriesWatcherMetaData, String, boolean)`.
    pub fn set_parameters_series_watcher_meta_data(
        &self,
        meta_data: &SeriesWatcherMetaData,
        stack_id: Option<&str>,
        init: bool,
    ) {
        self.table()
            .set_parameters_series_watcher_meta_data(meta_data, stack_id, init);
        self.update_display_void();
        self.table().update_row_display();
        self.send_queue_table_events();
    }

    /// Java `isParamFileModifiable()`.
    pub fn is_param_file_modifiable(&self) -> bool {
        self.ltf_root_name.is_editable()
    }

    /// Java `isParamFileEmpty()`.
    pub fn is_param_file_empty(&self) -> bool {
        Field::is_empty(&*self.ltf_root_name) || Field::is_empty(&*self.ftf_root_dir)
    }

    /// Java `disableRootFields()`.
    pub fn disable_root_fields(&self) {
        self.ltf_root_name.set_editable(false);
        self.ftf_root_dir.set_editable(false);
    }

    /// Java private `disableDeliveryFields()`.
    fn disable_delivery_fields(&self) {
        self.manager.get_meta_data().set_delivered(true);
        self.rb_deliver_off.set_editable(false);
        self.rb_deliver_to_directory.set_editable(false);
        self.ftf_deliver_to_directory.set_editable(false);
        self.rb_deliver_make_sub_directory.set_editable(false);
    }

    /// Java `getParameters(UserConfiguration)`.
    pub fn get_parameters_user_configuration(&self, user_configuration: &mut UserConfiguration) {
        user_configuration.set_use_email_address(self.ctf_email_address.is_selected());
        user_configuration
            .set_email_address(Field::get_text_void(&*self.ctf_email_address).as_deref());
        // Nothing to pull out of SeriesWatcherPanel.
    }

    /// Java `setParameters()`.  Set environment parameters.
    pub fn set_parameters_void(&self) {
        self.cb_cpu_machine_list
            .set_selected_boolean(user_env::is_parallel_processing(
                self.base_manager(),
                AxisID::Only,
                None,
            ));
        if user_env::is_gpu_processing(self.base_manager(), AxisID::Only, None) {
            self.rb_gpu_machine_list_local.set_selected_boolean(true);
        } else {
            self.rb_gpu_machine_list_off.set_selected_boolean(true);
        }
    }

    /// Java `setParameters(UserConfiguration, boolean)`.
    pub fn set_parameters_user_configuration(
        &self,
        user_configuration: &UserConfiguration,
        new_dataset: bool,
    ) {
        if new_dataset {
            self.template_panel
                .set_parameters_user_configuration(user_configuration);
        }
        self.ctf_email_address
            .set_selected_boolean(user_configuration.is_use_email_address());
        self.ctf_email_address
            .set_text_string(user_configuration.get_email_address().as_deref());
    }

    /// Java `getParameters(BatchRunTomoMetaData)`.
    pub fn get_parameters_meta_data(&self, meta_data: &BatchRunTomoMetaData) {
        meta_data.set_use_series_watcher(self.cb_use_series_watcher.is_selected());
        if let Some(validbrowsing_directory) = self.validbrowsing_directory.borrow().as_ref() {
            meta_data.set_browsing_directory(validbrowsing_directory.get_void().as_deref());
        }
        meta_data.set_deliver_to_directory(
            crate::imod::etomo::ui::swing::file_text_field_interface::FileTextFieldInterface::get_file(
                &*self.ftf_deliver_to_directory,
            )
            .as_deref(),
        );
        meta_data.set_input_directive_file(
            crate::imod::etomo::ui::swing::file_text_field_interface::FileTextFieldInterface::get_file(
                &*self.ftf_input_directive_file,
            )
            .as_deref(),
        );
        meta_data.set_split_batch(self.cb_split_batch.is_selected());
        meta_data.set_max_gpus_for_one_job(Some(&self.sp_max_gpus_for_one_job.get_text()));
        meta_data.set_number_of_jobs_to_make(Some(&self.sp_number_of_jobs_to_make.get_text()));
        meta_data.set_queue_number_of_jobs_to_make(Some(
            &self.sp_queue_number_of_jobs_to_make.get_text(),
        ));
        let mut queue_type: Option<QueueType> = None;
        if let (Some(rb_queue_type_queue), Some(rb_queue_type_node)) =
            (&self.rb_queue_type_queue, &self.rb_queue_type_node)
        {
            if rb_queue_type_queue.is_selected() {
                queue_type = Some(QueueType::Queue);
            } else if rb_queue_type_node.is_selected() {
                queue_type = Some(QueueType::Node);
            }
        }
        meta_data.set_queue_type(queue_type);
        match &self.cb_queue_secondary_queue {
            Some(cb_queue_secondary_queue) => {
                meta_data.set_use_secondary_queue(cb_queue_secondary_queue.is_selected())
            }
            None => meta_data.reset_use_secondary_queue(),
        }
        meta_data.set_use_cpu_machine_list(self.cb_cpu_machine_list.is_selected());
        if self.rb_gpu_machine_list_local.is_selected() {
            meta_data.set_use_gpu_machine_list_parallel(false);
        } else if self.rb_gpu_machine_list.is_selected() {
            meta_data.set_use_gpu_machine_list_parallel(true);
        } else {
            meta_data.set_use_gpu_machine_list_parallel_null();
        }
        meta_data.set_watch_directory(
            crate::imod::etomo::ui::swing::file_text_field_interface::FileTextFieldInterface::get_file(
                &*self.ftf_watch_directory,
            )
            .as_deref(),
        );
        self.table().get_parameters_meta_data(meta_data);
        self.step_panel().get_parameters_meta_data(meta_data);
        self.series_watcher_panel()
            .get_parameters_meta_data(meta_data);
        self.get_dataset_dialog()
            .get_parameters_dataset_meta_data(&meta_data.get_dataset_meta_data());
        let status = self
            .batch_run_tomo_state
            .borrow()
            .get_batch_run_tomo_status();
        meta_data.set_status(status);
    }

    /// Java `getParameters(SplitBatchParam)`.
    pub fn get_parameters_split_batch_param(&self, param: &mut SplitBatchParam) {
        if self.sp_max_gpus_for_one_job.is_enabled() {
            param.set_max_gpus_for_one_job_string(Some(&self.sp_max_gpus_for_one_job.get_text()));
        } else if self.rb_gpu_machine_list_local.is_selected() {
            param.set_max_gpus_for_one_job_int(1);
        } else {
            param.reset_max_gpus_for_one_job();
        }
    }

    /// Java `setParameters(BatchruntomoParam)`.
    pub fn set_parameters_batchruntomo_param(&self, param: &BatchruntomoParam) {
        self.rb_deliver_off.set_selected_boolean(true);
        if param.is_deliver_to_directory_set() {
            self.rb_deliver_to_directory.set_selected_boolean(true);
            self.ftf_deliver_to_directory
                .set_text_string(Some(&param.get_deliver_to_directory()));
        }
        if param.is_make_sub_directory() {
            self.rb_deliver_make_sub_directory
                .set_selected_boolean(true);
        }
        self.cb_cpu_machine_list
            .set_selected_boolean(!param.is_cpu_machine_list_null());
        if param.is_gpu_machine_list_null() {
            self.rb_gpu_machine_list_off.set_selected_boolean(true);
        } else if param.gpu_machine_list_equals(Some(batchruntomo_param::MACHINE_LIST_LOCAL_VALUE))
        {
            self.rb_gpu_machine_list_local.set_selected_boolean(true);
        } else {
            self.rb_gpu_machine_list.set_selected_boolean(true);
        }
        if !param.is_email_address_null() {
            self.ctf_email_address.set_selected_boolean(true);
            self.ctf_email_address
                .set_text_string(Some(&param.get_email_address()));
        }
        self.step_panel().set_parameters_param(param);
        self.update_display_void();
        self.send_queue_table_events();
    }

    /// Java `setParameters(SeriesWatcherParam)`.
    pub fn set_parameters_series_watcher_param(&self, param: &SeriesWatcherParam) {
        if param.is_watch_directory_set() {
            self.ftf_watch_directory
                .set_text_string(Some(&param.get_watch_directory()));
        }
        self.series_watcher_panel().set_parameters_param(param);
    }

    /// Java package-private `isDeliver()`.
    pub fn is_deliver(&self) -> bool {
        !self.rb_deliver_off.is_selected()
    }

    /// Java `isParallelProcessing()`.
    pub fn is_parallel_processing(&self) -> bool {
        self.cb_split_batch.is_visible()
            && self.cb_split_batch.is_enabled()
            && self.cb_split_batch.is_selected()
    }

    /// Java `getParameters(BatchruntomoParam, RunType, boolean, boolean, boolean)`.
    pub fn get_parameters_batchruntomo_param(
        &self,
        param: &mut BatchruntomoParam,
        run_type: Option<RunType>,
        do_validation: bool,
        for_update: bool,
        validate_only: bool,
    ) -> bool {
        let mut err_msg = String::new();
        let result = (|| -> Result<bool, FieldValidationFailedException> {
            if !for_update {
                if self.rb_deliver_off.is_selected() {
                    param.reset_deliver();
                } else if self.rb_deliver_to_directory.is_selected() {
                    param.set_deliver_to_directory(
                        self.ftf_deliver_to_directory
                            .get_file_boolean_field_displayer(
                                do_validation,
                                self.this
                                    .upgrade()
                                    .map(|this| this as Rc<dyn FieldDisplayer>),
                            )?
                            .as_deref(),
                    );
                } else if self.rb_deliver_make_sub_directory.is_selected() {
                    param.set_make_sub_directory(true);
                }
                if self.ctf_email_address.is_selected() {
                    param.set_email_address(
                        self.ctf_email_address
                            .get_text_boolean(do_validation)?
                            .as_deref(),
                    );
                } else {
                    param.reset_email_address();
                }
            }
            if !self.cb_cpu_machine_list.is_selected() {
                param.set_cpu_machine_list(Some(batchruntomo_param::MACHINE_LIST_LOCAL_VALUE));
            }
            if self.rb_gpu_machine_list_off.is_selected() {
                param.reset_gpu_machine_list();
            } else if self.rb_gpu_machine_list_local.is_selected() {
                param.set_gpu_machine_list(Some(batchruntomo_param::MACHINE_LIST_LOCAL_VALUE));
            }
            if !self.table().get_parameters_param(
                param,
                self.rb_deliver_off.is_selected(),
                self.rb_deliver_to_directory.is_selected(),
                run_type,
                &mut err_msg,
                do_validation,
                validate_only,
            ) {
                return Ok(false);
            }
            self.step_panel().get_parameters_param(param, validate_only);
            if do_validation {
                if !err_msg.is_empty() {
                    ui_harness::with(|harness| {
                        harness.open_message_dialog_base_manager_string_string(
                            Some(self.base_manager()),
                            &err_msg,
                            "Unable to Set Up Directories",
                        )
                    });
                    return Ok(false);
                }
                if !validate_only {
                    self.disable_delivery_fields();
                }
            }
            Ok(true)
        })();
        result.unwrap_or(false)
    }

    /// Java `getParameters(SeriesWatcherParam, boolean)`.
    pub fn get_parameters_series_watcher_param(
        &self,
        param: &mut SeriesWatcherParam,
        do_validation: bool,
    ) -> bool {
        let result = (|| -> Result<bool, FieldValidationFailedException> {
            // Set whether or not enabled. This field is only disabled when serieswatcher
            // is not in use, and the com file will be saved either way.
            param.set_watch_directory(
                self.ftf_watch_directory
                    .get_file_boolean(do_validation)?
                    .as_deref(),
            );
            param.set_etomo_project_root(Field::get_text_void(&*self.ltf_root_name).as_deref());

            if self.cb_use_series_watcher.is_selected() {
                if self.cb_split_batch.is_enabled() && self.cb_split_batch.is_selected() {
                    if self.queue_table_displayed.get() {
                        param.set_parallel_runs(Some(
                            &self.sp_queue_number_of_jobs_to_make.get_text(),
                        ));
                    } else {
                        param.set_parallel_runs(Some(&self.sp_number_of_jobs_to_make.get_text()));
                    }
                } else {
                    param.reset_parallel_runs();
                }
            }
            Ok(self
                .series_watcher_panel()
                .get_parameters_param(param, do_validation))
        })();
        // SeriesWatcherPanel.getParameters throws FieldValidationFailedException
        // (here: returns false) and the Java returns false from the catch.
        result.unwrap_or(false)
    }

    // <p>Updates done</p>

    /// Java `loadTemplates()`.
    pub fn load_templates(&self) {
        // load templates from global autodoc
        let directive_file = DirectiveFile::get_instance(
            self.base_manager(),
            Some(self.axis_id),
            file_type::CLASS
                .batch_run_tomo_global_autodoc
                .get_file(Some(self.base_manager()), Some(self.axis_id))
                .as_deref(),
            DirectiveFileType::Batch,
        );
        self.template_panel
            .set_parameters_directive_file(directive_file.as_ref());
    }

    /// Java `loadAutodocs(String, boolean)`.  `onlyStackIDDatasetDialog`: only loading
    /// the dataset dialog attached to the row with this stackID;
    /// `onlyAdvancedDatasetDialog`: only loading an advanced dataset dialog.
    pub fn load_autodocs(
        &self,
        only_stack_id_dataset_dialog: Option<&str>,
        only_advanced_dataset_dialog: bool,
    ) {
        if only_stack_id_dataset_dialog.is_none() {
            // load global autodoc
            let directive_file = DirectiveFile::get_instance(
                self.base_manager(),
                Some(self.axis_id),
                file_type::CLASS
                    .batch_run_tomo_global_autodoc
                    .get_file(Some(self.base_manager()), Some(self.axis_id))
                    .as_deref(),
                DirectiveFileType::Batch,
            );
            // Java passes the (possibly null) directive file on; DirectiveFile's
            // getInstance returns null for a missing file and setValues then
            // dereferences it.  Fixed in translation: nothing is loaded.
            if let Some(directive_file) = &directive_file {
                self.get_dataset_dialog().set_values(
                    directive_file,
                    false,
                    only_advanced_dataset_dialog,
                    true,
                );
            }
        }
        // load dataset autodocs
        self.table()
            .load_autodocs(only_stack_id_dataset_dialog, only_advanced_dataset_dialog);
    }

    /// Java `saveAutodocs(DatasetFileBuilder, boolean, boolean, String, boolean,
    /// boolean)`.  `autodocStackID`: don't save all dataset autodocs - just this one;
    /// `onlyGlobalAutodoc`: don't save any dataset autodocs.
    pub fn save_autodocs(
        &self,
        dataset_file_builder: &DatasetFileBuilder,
        do_validation: bool,
        init: bool,
        autodoc_stack_id: Option<&str>,
        only_global_autodoc: bool,
        validate_only: bool,
    ) -> bool {
        if validate_only && !do_validation {
            return true;
        }
        // save global autodoc
        let batch_file = dataset_file_builder.build_file(
            Some(&file_type::CLASS.batch_run_tomo_global_autodoc),
            self.axis_id,
        );
        let mut templates: Option<NameValuePairList> = None;
        // If the advanced dialog was never created, then load the save file first as
        // not all of it was loaded into the dialog.
        let advanced_dialog_exists = self.get_dataset_dialog().is_advanced_dialog_exists();
        let mut loaded_batch_list: Option<NameValuePairList> = None;
        let mut save_batch_list: Option<NameValuePairList> = None;
        let result = (|| -> Result<Option<bool>, LogFileError> {
            if !validate_only
                && let Some(batch_file) = &batch_file
                && batch_file.exists()
            {
                if !advanced_dialog_exists {
                    let autodoc = unsafe {
                        autodoc_factory::get_autodoc_instance(
                            Some(self.base_manager()),
                            Some(batch_file),
                        )
                    }?;
                    loaded_batch_list = Some(unsafe { NameValuePairList::new_autodoc(autodoc) });
                }
                // `Utilities.deleteFileOrDirectory(batchFile, manager, axisID)`.
                utilities::delete_file_or_directory(
                    batch_file,
                    Some(self.base_manager()),
                    Some(self.axis_id),
                );
            }
            let batch_autodoc: *mut Autodoc = unsafe {
                autodoc_factory::get_writable_autodoc_instance(
                    Some(self.base_manager()),
                    batch_file.as_deref(),
                )
            }?;
            self.template_panel
                .save_autodoc(unsafe { &mut *batch_autodoc }, validate_only);
            if !self
                .get_dataset_dialog()
                .save_autodoc(batch_autodoc, do_validation, validate_only)
            {
                return Ok(Some(false));
            }
            if !validate_only {
                let template_files = self.template_panel.get_files();
                templates = batch_tool::merge_templates(self.base_manager(), Some(&template_files));
                let list = if advanced_dialog_exists {
                    unsafe {
                        batch_tool::create_batch_file(
                            self.base_manager(),
                            batch_autodoc,
                            advanced_dialog_exists,
                            None,
                            None,
                            None,
                            None,
                            templates.as_ref(),
                        )
                    }
                } else {
                    // No advanced dialog - take the advanced directives from the files
                    let basic_directives = self.basic_directives.borrow().clone();
                    let advanced_starting_batch = self.get_advanced_starting_batch();
                    unsafe {
                        batch_tool::create_batch_file(
                            self.base_manager(),
                            batch_autodoc,
                            advanced_dialog_exists,
                            None,
                            loaded_batch_list.as_mut(),
                            Some(&basic_directives),
                            advanced_starting_batch.as_ref(),
                            templates.as_ref(),
                        )
                    }
                };
                let log_file = unsafe { (*batch_autodoc).get_log_file() };
                save_batch_list = Some(list);
                save_batch_list.as_ref().unwrap().write(log_file.as_ref());
            }
            Ok(None)
        })();
        match result {
            Ok(Some(retval)) => return retval,
            Ok(None) | Err(LogFileError::Lock(_)) => {}
            Err(e) => eprintln!("{e}"),
        }
        // save dataset autodocs with the starting batch and default batch directive
        // files grafted on.
        if autodoc_stack_id.is_some() || !only_global_autodoc {
            let deliver_to_directory = if self.rb_deliver_to_directory.is_selected() {
                crate::imod::etomo::ui::swing::file_text_field_interface::FileTextFieldInterface::get_file(
                    &*self.ftf_deliver_to_directory,
                )
            } else {
                None
            };
            return self.table().save_autodocs(
                Some(&self.template_panel),
                save_batch_list.as_ref(),
                templates.as_ref(),
                do_validation,
                init,
                deliver_to_directory.as_deref(),
                autodoc_stack_id,
                validate_only,
            );
        }
        true
    }

    /// Java private `getInputDirectiveAutodoc(StringBuilder)`.
    fn get_input_directive_autodoc(&self, err_msg: Option<&mut String>) -> Option<*mut Autodoc> {
        let file = crate::imod::etomo::ui::swing::file_text_field_interface::FileTextFieldInterface::get_file(
            &*self.ftf_input_directive_file,
        );
        let file = file.filter(|file| file.exists())?;
        let result = unsafe {
            match err_msg {
                Some(err_msg) => autodoc_factory::get_autodoc_instance_err_msg(
                    Some(self.base_manager()),
                    Some(&file),
                    err_msg as *mut String,
                ),
                None => autodoc_factory::get_autodoc_instance_err_msg(
                    Some(self.base_manager()),
                    Some(&file),
                    std::ptr::null_mut(),
                ),
            }
        };
        match result {
            Ok(autodoc) if !autodoc.is_null() => Some(autodoc),
            Ok(_) | Err(LogFileError::Lock(_)) => None,
            Err(e) => {
                eprintln!("{e}");
                None
            }
        }
    }

    /// Java private `validateInputDirectiveFile()`.
    fn validate_input_directive_file(&self) -> bool {
        let mut err_msg = String::new();
        let Some(autodoc) = self.get_input_directive_autodoc(Some(&mut err_msg)) else {
            return true;
        };
        if unsafe { (*autodoc).is_error() } {
            self.display_void();
            Popup::get_unformatted_error_instance(
                Some(&*self.ftf_input_directive_file as &dyn UIComponent),
                Some("Errors in Starting Directive File"),
                Some(&format!(
                    "Syntax error in the Starting Directive File - unable to load.\nPlease correct the file and reload.\n\n{err_msg}"
                )),
            )
            .open();
            return false;
        }
        true
    }

    /// Java package-private `getAdvancedStartingBatch()`.
    pub fn get_advanced_starting_batch(&self) -> Option<NameValuePairList> {
        if self.advanced_starting_batch.borrow().is_none()
            && let Some(autodoc) = self.get_input_directive_autodoc(None)
        {
            let mut advanced_starting_batch = unsafe { NameValuePairList::new_autodoc(autodoc) };
            advanced_starting_batch.subtract(Some(&self.basic_directives.borrow()));
            *self.advanced_starting_batch.borrow_mut() = Some(advanced_starting_batch);
        }
        self.advanced_starting_batch
            .borrow()
            .as_ref()
            .map(|list| NameValuePairList::new_copy(Some(list)))
    }

    /// Java package-private `getFirstRow()`.
    pub fn get_first_row(&self) -> Option<Rc<BatchRunTomoRow>> {
        self.table().get_first_row()
    }

    /// Java `getDatasetImageFilenameStyle()`.
    pub fn get_dataset_image_filename_style(&self) -> Option<ImageFilenameStyle> {
        self.table().get_image_filename_style()
    }

    /// Java `updateDirectives(boolean, String, boolean)`.  `onlyStackIDDatasetDialog`:
    /// only loading the dataset dialog attached to the row with this stackID;
    /// `onlyAdvancedDatasetDialog`: only loading an advanced dataset dialog.
    pub fn update_directives(
        &self,
        init: bool,
        only_stack_id_dataset_dialog: Option<&str>,
        only_advanced_dataset_dialog: bool,
    ) {
        let only_global_dataset_dialog =
            only_stack_id_dataset_dialog.is_none() && only_advanced_dataset_dialog;
        let mut retain_user_values = false;
        if !init {
            // See if the user has changed any values (and back up the changed values).
            let mut changed = false;
            if !only_global_dataset_dialog
                && self
                    .table()
                    .backup_if_changed(only_stack_id_dataset_dialog, only_advanced_dataset_dialog)
            {
                changed = true;
            }
            if only_stack_id_dataset_dialog.is_none() {
                if self
                    .get_dataset_dialog()
                    .backup_if_changed(only_advanced_dataset_dialog)
                {
                    changed = true;
                }
                if self
                    .series_watcher_panel()
                    .backup_if_changed(only_advanced_dataset_dialog)
                {
                    changed = true;
                }
            }
            if !retain_user_values && changed {
                // Ask the user whether they want to keep the values they changed.
                retain_user_values = ui_harness::with(|harness| {
                    harness.open_yes_no_dialog_base_manager_string_axis_id(
                        Some(self.base_manager()),
                        "New batch directive/template values will be applied.  Keep your changed values?",
                        Some(self.axis_id),
                    )
                });
            }
        }
        let directive_file_collection = self.directive_file_collection.borrow();
        if !only_global_dataset_dialog {
            self.table().apply_values(
                init,
                retain_user_values,
                &directive_file_collection,
                only_stack_id_dataset_dialog,
                only_advanced_dataset_dialog,
            );
        }
        if only_stack_id_dataset_dialog.is_none() {
            self.get_dataset_dialog().apply_values(
                init,
                retain_user_values,
                &directive_file_collection,
                only_advanced_dataset_dialog,
            );
            self.series_watcher_panel().apply_values(
                init,
                retain_user_values,
                &directive_file_collection,
                only_stack_id_dataset_dialog,
                only_advanced_dataset_dialog,
            );
        }
    }

    /// Java private `validate()`.
    fn validate(&self) -> bool {
        if !self.validate_input_directive_file() {
            return false;
        }
        if !self.get_dataset_dialog().validate() {
            return false;
        }
        if !self.table().validate(Some(self.get_dataset_dialog())) {
            return false;
        }
        true
    }

    /// Java `getDirectiveFileCollection()`.
    pub fn get_directive_file_collection(&self) -> Rc<RefCell<DirectiveFileCollection>> {
        Rc::clone(&self.directive_file_collection)
    }

    /// Java `actionPerformed(ActionEvent)`.
    pub fn action_performed(&self, event: Option<&ActionEvent>) {
        let Some(event) = event else {
            return;
        };
        let Some(action_command) = event.get_action_command().map(str::to_owned) else {
            return;
        };
        let action_command_option = Some(action_command.clone());
        let mut paused_failed = true;
        let this: Rc<dyn ProcessInterface> = self.this_rc();
        if self
            .template_panel
            .equals_action_command(Some(&action_command))
        {
            // refresh the shared directive file collection
            self.template_panel.refresh_directive_file_collection();
            self.update_directives(false, None, false);
        } else if action_command_option == self.btn_run.get_action_command() {
            if self.validate() {
                if !self.cb_split_batch.is_enabled() || !self.cb_split_batch.is_selected() {
                    self.manager
                        .batchruntomo(Some(self.get_processing_method()));
                } else {
                    self.manager.split_batch();
                }
            }
        } else if action_command_option == self.btn_resume.get_action_command() {
            self.manager
                .resume_batchruntomo(Some(self.get_processing_method()));
        } else if action_command_option == self.rb_gpu_machine_list_off.get_action_command()
            || action_command_option == self.rb_gpu_machine_list_local.get_action_command()
            || action_command_option == self.rb_gpu_machine_list.get_action_command()
        {
            self.update_display_void();
            self.mediator_set_method(&this);
        } else if action_command_option == self.cb_cpu_machine_list.get_action_command() {
            self.update_display_void();
            self.send_queue_table_event(Some(self.get_split_batch_queue_table_event()));
            self.mediator_set_method(&this);
        } else if action_command_option == self.cb_split_batch.get_action_command() {
            self.update_display_void();
            self.send_queue_table_event(Some(self.get_split_batch_queue_table_event()));
            self.mediator_set_method(&this);
        } else if action_command_option == self.btn_reset.get_action_command()
            || action_command_option == self.btn_reset_series_watcher.get_action_command()
        {
            self.start_over();
        } else if action_command_option == self.btn_pause.get_action_command()
            || action_command_option == self.btn_finish_series_watcher.get_action_command()
        {
            paused_failed = !self.manager.pause(Some(self.axis_id));
        } else if action_command_option == self.btn_clear_input_directive_file.get_action_command()
        {
            Field::clear(&*self.ftf_input_directive_file);
            self.ftf_input_directive_file.checkpoint();
        } else if let (Some(rb_queue_type_queue), Some(rb_queue_type_node)) =
            (&self.rb_queue_type_queue, &self.rb_queue_type_node)
            && (action_command_option == rb_queue_type_queue.get_action_command()
                || action_command_option == rb_queue_type_node.get_action_command())
        {
            self.update_display_void();
            self.send_queue_table_event(self.get_only_queue_type_queue_table_event());
            self.send_queue_table_event(self.get_secondary_queue_table_event());
            self.send_queue_table_event(Some(self.get_number_jobs_changed_queue_table_event()));
        } else if let Some(cb_queue_secondary_queue) = &self.cb_queue_secondary_queue
            && action_command_option == cb_queue_secondary_queue.get_action_command()
        {
            self.update_display_void();
            // The secondary queue can only work with certain types of primary queues.
            self.send_queue_table_event(self.get_only_queue_type_queue_table_event());
            self.send_queue_table_event(self.get_secondary_queue_table_event());
            self.send_queue_table_event(Some(self.get_number_jobs_changed_queue_table_event()));
        } else if action_command_option == self.btn_start_series_watcher.get_action_command() {
            self.manager
                .series_watcher(Some(self.get_processing_method()));
        } else if self.equals_series_watcher_action_command(&action_command) {
            self.pnl_run_settings.remove_all();
            self.build_run_settings_panel();
            self.table().update_row_display();
            self.update_display_void();
            ui_harness::with(|harness| harness.pack_base_manager(Some(self.base_manager())));
        }
        self.update_display(paused_failed);
    }

    /// `mediator.setMethod(this, getProcessingMethod(), getSecondaryProcessingMethod(),
    /// curTab == BatchRunTomoTab.RUN)`.
    fn mediator_set_method(&self, this: &Rc<dyn ProcessInterface>) {
        if let Some(mediator) = &self.mediator {
            let processing_method = self.get_processing_method();
            let secondary_processing_method = self.get_secondary_processing_method();
            mediator.set_method_process_interface_processing_method_processing_method_boolean(
                this,
                Some(processing_method),
                secondary_processing_method,
                self.cur_tab.get() == Some(BatchRunTomoTab::Run),
            );
        }
    }

    /// Java private `getSplitBatchQueueTableEvent()`.
    fn get_split_batch_queue_table_event(&self) -> QueueTableEvent {
        // If CPUs are not selected and split batch is, then switch the processor table
        // to queue table.
        if self.cb_split_batch.is_enabled() && self.cb_split_batch.is_selected() {
            return if self.cb_cpu_machine_list.is_selected() {
                QueueTableEvent::AllowDisplay
            } else {
                QueueTableEvent::Display
            };
        }
        QueueTableEvent::PreventDisplay
    }

    /// Java private `getOnlyQueueTypeQueueTableEvent()`.
    fn get_only_queue_type_queue_table_event(&self) -> Option<QueueTableEvent> {
        let (Some(rb_queue_type_queue), Some(rb_queue_type_node), Some(cb_queue_secondary_queue)) = (
            &self.rb_queue_type_queue,
            &self.rb_queue_type_node,
            &self.cb_queue_secondary_queue,
        ) else {
            return None;
        };
        if rb_queue_type_queue.is_enabled() && rb_queue_type_queue.is_selected() {
            return Some(QueueTableDataEvent::get_only_queue_type_instance(
                QueueType::Queue,
            ));
        }
        if rb_queue_type_node.is_enabled() && rb_queue_type_node.is_selected() {
            if cb_queue_secondary_queue.is_enabled() && cb_queue_secondary_queue.is_selected() {
                return Some(QueueTableDataEvent::get_only_queue_type_instance(
                    QueueType::NodeWithoutGpu,
                ));
            }
            return Some(QueueTableDataEvent::get_only_queue_type_instance(
                QueueType::Node,
            ));
        }
        None
    }

    /// Java private `getSecondaryQueueTableEvent()`.
    fn get_secondary_queue_table_event(&self) -> Option<QueueTableEvent> {
        if let Some(cb_queue_secondary_queue) = &self.cb_queue_secondary_queue {
            return Some(
                if cb_queue_secondary_queue.is_enabled() && cb_queue_secondary_queue.is_selected() {
                    QueueTableEvent::EnableSecondaryQueue
                } else {
                    QueueTableEvent::DisableSecondaryQueue
                },
            );
        }
        None
    }

    /// Java private `getNumberJobsChangedQueueTableEvent()`.
    fn get_number_jobs_changed_queue_table_event(&self) -> QueueTableEvent {
        QueueTableDataEvent::get_number_jobs_changed_instance(
            self.sp_queue_number_of_jobs_to_make.get_text(),
        )
    }

    /// Java private `sendQueueTableEvent(QueueTableEvent)`.  A null event (the Java
    /// sends one when there are no queues) is not sent: the listeners' event type
    /// takes an event, and the only listener (`ParallelPanel`) does nothing with a
    /// null one when it has no queue table.
    fn send_queue_table_event(&self, event: Option<QueueTableEvent>) {
        let queue_table_listener_array = self.queue_table_listener_array.borrow().clone();
        let Some(queue_table_listener_array) = queue_table_listener_array else {
            return;
        };
        let Some(event) = event else {
            return;
        };
        for listener in &queue_table_listener_array {
            listener.queue_table_event_action(&event);
        }
    }

    /// Java `startOver()` (StatusChangeListener).
    fn start_over_impl(&self) {
        self.status_changed_status(Some(StatusRef::BatchRunTomoStatus(
            BatchRunTomoStatus::Open,
        )));
        self.step_panel().start_over();
        let listeners = self.listeners.lock().unwrap().clone();
        if let Some(listeners) = listeners {
            for listener in &listeners {
                listener.get().start_over();
            }
        }
    }

    /// Java `createRunList(RunType)`.
    pub fn create_run_list(&self, run_type: Option<RunType>) -> RunList {
        self.table().create_run_list(run_type)
    }

    /// Java `findRow(String, String)`.
    pub fn find_row(&self, location: Option<&str>, root_name: Option<&str>) -> Option<String> {
        self.table().find_row(location, root_name)
    }

    /// Java `getStack(String)`.
    pub fn get_stack(&self, stack_id: Option<&str>) -> Option<PathBuf> {
        self.table().get_stack(stack_id)
    }

    /// Java `getRootName()`.
    pub fn get_root_name(&self) -> Option<String> {
        Field::get_text_void(&*self.ltf_root_name)
    }

    /// Java `getRootDir()`.
    pub fn get_root_dir(&self) -> Option<PathBuf> {
        crate::imod::etomo::ui::swing::file_text_field_interface::FileTextFieldInterface::get_file(
            &*self.ftf_root_dir,
        )
    }

    /// Java package-private `isTrackingMethodSeed()`.
    pub fn is_tracking_method_seed(&self) -> bool {
        self.get_dataset_dialog().is_tracking_method_seed()
    }

    /// Java `processResult(Object, boolean)`.  Processes a result from object.
    /// `init` - true when result was caused by the creating a dialog, rather then by a
    /// direct user action.
    pub fn process_result(&self, object: Option<&dyn std::any::Any>, init: bool) {
        let is_input_directive_file = object
            .and_then(|object| object.downcast_ref::<FileTextField2>())
            .is_some_and(|object| std::ptr::eq(object, &*self.ftf_input_directive_file));
        if is_input_directive_file
            && Field::is_different_from_checkpoint(&*self.ftf_input_directive_file, false)
        {
            self.ftf_input_directive_file.checkpoint();
            if self.validate_input_directive_file() {
                self.directive_file_collection.borrow_mut().set_directive_file(
                    crate::imod::etomo::ui::swing::file_text_field_interface::FileTextFieldInterface::get_file(
                        &*self.ftf_input_directive_file,
                    )
                    .as_deref(),
                    DirectiveFileType::Batch,
                );
                // The templates in the starting batch are valid with the start batch
                // values and should be used.
                self.template_panel.activate_actions(false);
                self.template_panel.clear();
                let directive_file = self
                    .directive_file_collection
                    .borrow()
                    .get_directive_file(DirectiveFileType::Batch);
                self.template_panel
                    .set_parameters_directive_file(directive_file.as_deref());
                self.template_panel.activate_actions(true);
                self.update_directives(init, None, false);
            }
        }
    }

    /// Java `pack()`.
    pub fn pack(&self) {
        // The dataset table is usually very narrow. Prevent it from bloating out
        // horizontally.
        if self.cur_tab.get() == Some(BatchRunTomoTab::Dataset) {
            self.pnl_dataset_table_body.remove_all();
            let dataset_width = self.get_dataset_dialog().get_preferred_width();
            let table_width = self.table().get_preferred_width();
            if dataset_width != 0 && table_width != 0 && dataset_width > table_width {
                // Swing layout: horizontal struts of (datasetWidth - tableWidth) / 2 on
                // both sides of the table.
                self.pnl_dataset_table_body
                    .add(&self.table().get_component());
            } else {
                self.pnl_dataset_table_body
                    .add(&self.table().get_component());
            }
        }
    }

    /// Java package-private `display(BatchRunTomoTab)`.  Displays the tab, if it is
    /// not displayed.
    pub fn display_tab(&self, tab: Option<BatchRunTomoTab>) {
        if let Some(tab) = tab
            && self.cur_tab.get() != Some(tab)
        {
            self.tabbed_pane
                .get_component()
                .set_selected_tab(tab.get_index());
        }
    }

    /// Java package-private `setDatasetTableVisible(boolean)`.
    pub fn set_dataset_table_visible(&self, visible: bool) {
        self.pnl_dataset_table.set_visible(visible);
    }

    /// Java `focusGained(FocusEvent)`: empty.
    fn focus_gained(&self, _event: Option<&FocusEvent>) {}

    /// Java `focusLost(FocusEvent)`.
    fn focus_lost(&self, event: Option<&FocusEvent>) {
        if self
            .sp_queue_number_of_jobs_to_make
            .equals_source(event.map(|event| &event.source))
        {
            self.send_queue_table_event(Some(self.get_number_jobs_changed_queue_table_event()));
        }
    }

    /// Java `stateChanged(ChangeEvent)`.  Handle tab change event.
    pub fn state_changed(&self, event: Option<&ChangeEvent>) {
        if self
            .sp_queue_number_of_jobs_to_make
            .equals_source(event.map(|event| &event.source))
        {
            self.send_queue_table_event(Some(self.get_number_jobs_changed_queue_table_event()));
            return;
        }
        let cur_index;
        match self.cur_tab.get() {
            None => {
                self.tabbed_pane
                    .get_component()
                    .set_selected_tab(batch_run_tomo_tab::DEFAULT.get_index());
                self.cur_tab.set(Some(batch_run_tomo_tab::DEFAULT));
                cur_index = batch_run_tomo_tab::DEFAULT.get_index();
            }
            Some(cur_tab) => {
                self.pnl_tabs.borrow()[cur_tab.get_index() as usize].remove_all();
                let cur_tab = BatchRunTomoTab::get_instance(
                    self.tabbed_pane.get_component().get_selected_tab(),
                );
                self.cur_tab.set(Some(cur_tab));
                cur_index = cur_tab.get_index();
            }
        }
        let cur_tab = self.cur_tab.get();
        let pnl_tab = Rc::clone(&self.pnl_tabs.borrow()[cur_index as usize]);
        if cur_tab == Some(BatchRunTomoTab::Batch) {
            pnl_tab.add(&self.pnl_batch);
        } else if cur_tab == Some(BatchRunTomoTab::Stacks) {
            pnl_tab.add(&self.pnl_stacks);
            self.table().msg_tab_changed(cur_tab);
            self.pnl_stacks_table.add(&self.table().get_component());
        } else if cur_tab == Some(BatchRunTomoTab::Dataset) {
            pnl_tab.add(&self.pnl_dataset);
            self.table().msg_tab_changed(cur_tab);
            self.pnl_dataset_table_body
                .add(&self.table().get_component());
            // Do the formatting that can only be done once the fields are displayed and
            // get their correct size.
            let mut tab_displayed = self.tab_displayed.borrow_mut();
            if !tab_displayed[cur_index as usize] {
                tab_displayed[cur_index as usize] = true;
                // Swing layout: UIUtilities.alignComponentsX(pnlDataset, LEFT).
            }
        } else if cur_tab == Some(BatchRunTomoTab::Run) {
            pnl_tab.add(&self.pnl_run);
            self.table().msg_tab_changed(cur_tab);
            self.pnl_run_table_body.add(&self.table().get_component());
            // Do the formatting that can only be done once the fields are displayed and
            // get their correct size.
            let mut tab_displayed = self.tab_displayed.borrow_mut();
            if !tab_displayed[cur_index as usize] {
                tab_displayed[cur_index as usize] = true;
                // Swing layout: alignComponentsX(pnlRun, LEFT); shrinkWrapHorizontal(
                // pnlSplitBatch); and, without queues, widen pnlResources to the wider
                // of itself and pnlRunTableBody plus scaleByFontSize(200).
            }
        }
        self.set_method();
    }

    /// Java private `setMethod()`.
    fn set_method(&self) {
        let this: Rc<dyn ProcessInterface> = self.this_rc();
        self.mediator_set_method(&this);
        ui_harness::with(|harness| {
            harness.pack_axis_id_base_manager(Some(self.axis_id), Some(self.base_manager()))
        });
    }

    /// Java private `updateDisplay()`.
    fn update_display_void(&self) {
        self.update_display(false);
    }

    /// Java private `updateDisplay(boolean)`.
    fn update_display(&self, paused_failed: bool) {
        // seriesWatcher
        let use_series_watcher = self.cb_use_series_watcher.is_selected();
        if !use_series_watcher {
            self.cb_use_series_watcher
                .set_text(Some(USE_SERIES_WATCHER_LABEL));
        } else {
            self.cb_use_series_watcher.set_text(Some(&format!(
                "{USE_SERIES_WATCHER_LABEL}{WATCH_DIRECTORY_LABEL}"
            )));
        }
        self.ftf_watch_directory.set_enabled(use_series_watcher);
        self.ftf_watch_directory.set_visible(use_series_watcher);
        self.pnl_run_buttons.set_visible(!use_series_watcher);
        self.pnl_series_watcher_run_buttons
            .set_visible(use_series_watcher);
        let parallel_panel = self.parallel_panel.borrow().clone();
        if let Some(parallel_panel) = &parallel_panel {
            parallel_panel.set_limited(use_series_watcher);
        }
        // SplitBatch
        self.cb_split_batch.set_enabled(
            self.total_cpus > 1 || (self.queues_available && self.number_queue_cpus_available > 1),
        );
        if self.queue_table_displayed.get() {
            self.cb_split_batch
                .set_text(Some(SPLIT_BATCH_CLUSTER_ONLY_LABEL));
        } else {
            self.cb_split_batch
                .set_text(Some(SPLIT_BATCH_DEFAULT_LABEL));
        }
        let split_batch = self.cb_split_batch.is_enabled() && self.cb_split_batch.is_selected();
        // NumberOfJobsToMake
        let queue_table_displayed = self.queue_table_displayed.get();
        self.sp_number_of_jobs_to_make
            .set_visible(!queue_table_displayed);
        self.sp_queue_number_of_jobs_to_make
            .set_visible(queue_table_displayed);
        self.sp_number_of_jobs_to_make.set_enabled(split_batch);
        self.sp_queue_number_of_jobs_to_make
            .set_enabled(split_batch);
        self.l_number_of_jobs_to_make.set_enabled(split_batch);
        // Queue
        if let (
            Some(rb_queue_type_queue),
            Some(rb_queue_type_node),
            Some(cb_queue_secondary_queue),
        ) = (
            &self.rb_queue_type_queue,
            &self.rb_queue_type_node,
            &self.cb_queue_secondary_queue,
        ) {
            rb_queue_type_queue
                .set_enabled(queue_table_displayed && self.queue_type_queue_enabled.get());
            rb_queue_type_node.set_enabled(
                queue_table_displayed
                    && (self.queue_type_node_without_gpu_available
                        || self.queue_type_node_with_gpu_available),
            );
            // The secondary queue is not available for nodes if mode 2c queues (ones
            // without GPUS) are not available.
            cb_queue_secondary_queue.set_enabled(
                queue_table_displayed
                    && self.secondary_queues.get()
                    && (!rb_queue_type_node.is_selected()
                        || self.queue_type_node_without_gpu_available),
            );
        }
        // CPUMachineList
        self.cb_cpu_machine_list.set_enabled(!split_batch);
        // GPU
        let enable_gpu = !queue_table_displayed;
        self.rb_gpu_machine_list_off.set_enabled(enable_gpu);
        self.rb_gpu_machine_list_local
            .set_enabled(enable_gpu && self.local_gpu_available);
        self.rb_gpu_machine_list
            .set_enabled(enable_gpu && self.gpu_available);
        let gpu_machine_list_local = self.rb_gpu_machine_list_local.is_selected();
        self.l_max_gpus_for_one_job_one
            .set_enabled(split_batch && gpu_machine_list_local);
        if gpu_machine_list_local {
            self.pnl_max_gpus_for_one_job_one.set_visible(true);
            self.pnl_max_gpus_for_one_job.set_visible(false);
        }
        let gpu_machine_list =
            self.rb_gpu_machine_list.is_enabled() && self.rb_gpu_machine_list.is_selected();
        self.sp_max_gpus_for_one_job
            .set_enabled(split_batch && gpu_machine_list);
        if gpu_machine_list {
            self.pnl_max_gpus_for_one_job_one.set_visible(false);
            self.pnl_max_gpus_for_one_job.set_visible(true);
        }
        //
        self.ftf_deliver_to_directory
            .set_enabled(self.rb_deliver_to_directory.is_selected());
        // parallel batch
        if let Some(parallel_panel) = &parallel_panel {
            parallel_panel.set_runnable(split_batch);
        }
        self.btn_pause.set_visible(!split_batch);
        self.btn_resume.set_visible(!split_batch);
        let btn_parallel_pause = self.btn_parallel_pause();
        let btn_parallel_resume = self.btn_parallel_resume();
        if let Some(btn_parallel_pause) = &btn_parallel_pause {
            btn_parallel_pause.set_visible(split_batch);
        }
        if let Some(btn_parallel_resume) = &btn_parallel_resume {
            btn_parallel_resume.set_visible(split_batch);
        }

        // run buttons
        let (cur_status, resume_enabled, processchunk_resume_enabled) = {
            let state = self.batch_run_tomo_state.borrow();
            (
                state.get_batch_run_tomo_status(),
                state.is_resume_enabled(),
                state.is_processchunk_resume_enabled(),
            )
        };
        // Run is available when the process is not running and wasn't paused or
        // killed.
        let run = cur_status == Some(BatchRunTomoStatus::Open)
            || cur_status == Some(BatchRunTomoStatus::Done)
            || cur_status == Some(BatchRunTomoStatus::Stopped)
            || cur_status == Some(BatchRunTomoStatus::Failed);
        self.btn_run.set_enabled(run);
        self.btn_start_series_watcher.set_enabled(run);
        // Pause shouldn't be used while the process is in the process of exiting.
        let pause = cur_status == Some(BatchRunTomoStatus::Running);
        self.btn_pause.set_enabled(pause);
        if let Some(btn_parallel_pause) = &btn_parallel_pause {
            btn_parallel_pause.set_enabled(pause);
        }
        self.btn_finish_series_watcher.set_enabled(pause);
        // Resume is available if a matching pause or kill was done. Fields that change
        // what will run are disabled while Resume is available, so the Run button is
        // also disabled until Reset is used.
        self.btn_resume.set_enabled(resume_enabled);
        if let Some(btn_parallel_resume) = &btn_parallel_resume {
            btn_parallel_resume.set_enabled(processchunk_resume_enabled);
        }
        // Reset sets the status to OPEN (which is also the default status). Reset can
        // be done any time the process is not running. It shuts off Resume. If the
        // pause failed, then turn on Reset to avoid being trapped with an incorrect run
        // setting.
        let reset = cur_status == Some(BatchRunTomoStatus::Open)
            || (cur_status.is_some_and(|cur_status| cur_status.is_end_status()) || paused_failed);
        self.btn_reset.set_enabled(reset);
        self.btn_reset_series_watcher.set_enabled(reset);
    }

    /// Java public final `sendStatusChanged(Status)`.
    pub fn send_status_changed(&self, status: Option<StatusRef>) {
        let listeners = Arc::clone(&self.listeners);
        event_queue::invoke_later(move || {
            StatusChangeEventSender::new_status(listeners, status).run();
        });
    }

    /// Java `statusChanged(Status)`.
    pub fn status_changed_status(&self, new_status: Option<StatusRef>) {
        self.batch_run_tomo_state
            .borrow_mut()
            .handle_status_event(new_status.clone());
        let Some(StatusRef::BatchRunTomoStatus(_)) = new_status else {
            return;
        };
        // Avoid overriding an end state with an error state.
        // The rows will not be included in a resume if the status is open, so resume
        // must be restricted more then it would be for a regular processchunks run
        let temp_status = self
            .batch_run_tomo_state
            .borrow()
            .get_batch_run_tomo_status();
        let open = temp_status == Some(BatchRunTomoStatus::Open)
            || temp_status == Some(BatchRunTomoStatus::Failed)
            || temp_status == Some(BatchRunTomoStatus::Done)
            || temp_status == Some(BatchRunTomoStatus::Stopped);
        // Split batch is available for every finished status, except for
        // KILLED_PAUSED (brt only) because processchunks can't be resumed unless split
        // is done first.
        self.cb_split_batch.set_editable(
            open || temp_status == Some(BatchRunTomoStatus::KilledOrPausedProcessChunks),
        );
        self.ctf_email_address.set_editable(open);
        self.ftf_input_directive_file.set_editable(open);
        self.btn_clear_input_directive_file.set_editable(open);
        self.template_panel.set_editable(open);
        let locked = temp_status == Some(BatchRunTomoStatus::Running)
            || temp_status == Some(BatchRunTomoStatus::Pausing)
            || temp_status == Some(BatchRunTomoStatus::Killing);
        self.cb_cpu_machine_list.set_editable(!locked);
        self.rb_gpu_machine_list_off.set_editable(!locked);
        self.rb_gpu_machine_list_local.set_editable(!locked);
        self.rb_gpu_machine_list.set_editable(!locked);
        self.cb_split_batch.set_editable(!locked);
        self.sp_max_gpus_for_one_job.set_editable(!locked);
        self.sp_number_of_jobs_to_make.set_editable(!locked);
        self.sp_queue_number_of_jobs_to_make.set_editable(!locked);
        self.cb_use_series_watcher.set_editable(!locked);
        self.ftf_watch_directory.set_editable(!locked);
        self.series_watcher_panel().set_editable(!locked);

        // Control the resume buttons together, since the user can switch from
        // processchunks to brt.
        // Once the run has been killed or paused, don't shut off the Resume button
        // until a run has started
        if temp_status == Some(BatchRunTomoStatus::Running)
            || temp_status == Some(BatchRunTomoStatus::Open)
        {
            self.killed_paused.set(false);
        }
        if !self.killed_paused.get() {
            self.killed_paused.set(
                temp_status == Some(BatchRunTomoStatus::KilledOrPaused)
                    || temp_status == Some(BatchRunTomoStatus::KilledOrPausedProcessChunks),
            );
        }
        let status = self
            .batch_run_tomo_state
            .borrow()
            .get_batch_run_tomo_status();
        self.get_dataset_dialog()
            .status_changed_status(status.map(StatusRef::BatchRunTomoStatus));
        self.set_method();
    }

    /// Java `statusChanged(StatusChangeEvent)`.
    pub fn status_changed_event_impl(&self, status_change_event: Option<&dyn StatusChangeEvent>) {
        if let Some(status_change_event) = status_change_event {
            self.status_changed_status(status_change_event.get_status());
            self.get_dataset_dialog()
                .status_changed_event(Some(status_change_event));
        }
    }

    /// Java private `setTooltips()`.
    fn set_tooltips(&self) {
        let mut autodoc: *mut Autodoc = std::ptr::null_mut();
        match unsafe {
            autodoc_factory::get_instance(
                Some(self.base_manager()),
                Some(autodoc_factory::BATCH_RUN_TOMO),
                self.axis_id,
                false,
            )
        } {
            Ok(instance) => autodoc = instance,
            Err(LogFileError::Lock(_)) => {}
            Err(except) => eprintln!("{except}"),
        }
        let read_only = |autodoc: *mut Autodoc| -> Option<&dyn ReadOnlyAutodoc> {
            if autodoc.is_null() {
                None
            } else {
                Some(unsafe { &*autodoc } as &dyn ReadOnlyAutodoc)
            }
        };
        self.rb_deliver_off.set_tool_tip_text_string(Some(
            "No delivery.  Dataset will be processed in the original location of the stack; stacks must all be in separate directories.",
        ));
        let tooltip = etomo_autodoc::get_tooltip(
            read_only(autodoc),
            Some(batchruntomo_param::DELIVER_TO_DIRECTORY_TAG),
        );
        self.rb_deliver_to_directory
            .set_tool_tip_text_string(tooltip.as_deref());
        Field::set_tool_tip_text(&*self.ftf_deliver_to_directory, tooltip.as_deref());
        self.rb_deliver_make_sub_directory.set_tool_tip_text_string(Some(
            "Make a subdirectory for each dataset under the directory where the stack is currently located",
        ));
        self.ctf_email_address
            .set_tool_tip_text(Some("Send emails on failure or final completion."));
        self.cb_cpu_machine_list.set_tool_tip_text_string(Some(
            "Use multiple cores or multiple computers for the processing.",
        ));
        self.rb_gpu_machine_list_local
            .set_tool_tip_text_string(Some("Use one GPU on the local machine for reconstruction."));
        self.rb_gpu_machine_list
            .set_tool_tip_text_string(Some("Use multiple GPUs for reconstruction."));
        Field::set_tool_tip_text(
            &*self.ltf_root_name,
            Some("Root name for batch project files (.com, .adoc, .ebt)."),
        );
        Field::set_tool_tip_text(
            &*self.ftf_root_dir,
            Some("Location into which batch project files will be written"),
        );
        self.rb_gpu_machine_list_off
            .set_tool_tip_text_string(Some("No GPU will be used."));
        self.btn_run
            .set_tool_tip_text(Some("Saves with validation and runs batchruntomo."));
        Field::set_tool_tip_text(
            &*self.ftf_input_directive_file,
            Some(
                "Select an existing batch directive file to set initial values of parameters, after applying template values",
            ),
        );
        self.btn_clear_input_directive_file
            .set_tool_tip_text(Some(&format!(
                "Clears the entry in {}.",
                Field::get_quoted_label(&*self.ftf_input_directive_file)
                    .unwrap_or_else(|| "null".to_owned())
            )));
        self.btn_reset.set_tool_tip_text(Some(
            "Makes fields editable. Cancels ability to Resume.  You must uncheck Run checkboxes or change Start From entries, if you wish to avoid rerunning Stopped datasets.",
        ));
        self.btn_pause
            .set_tool_tip_text(Some("Finishes the current dataset and then stops."));
        self.btn_resume
            .set_tool_tip_text(Some("Continues the current batchruntomo run."));
        //
        self.template_panel.set_scope_tooltip(Some(
            "Select the first system-wide template file from which parameters will be set.",
        ));
        self.template_panel.set_system_tooltip(Some(
            "Select the second system-wide template file from which parameters will be set.",
        ));
        self.template_panel.set_user_tooltip(Some(
            "Select a personal template file from which parameters will be set.",
        ));
        //
        self.cb_use_series_watcher.set_tool_tip_text_string(Some(
            "process eligible tilt series stacks already present or as they appear in the directory",
        ));

        match unsafe {
            autodoc_factory::get_instance(
                Some(self.base_manager()),
                Some(autodoc_factory::SERIES_WATCHER),
                self.axis_id,
                false,
            )
        } {
            Ok(instance) => autodoc = instance,
            Err(LogFileError::Lock(_)) => {}
            Err(except) => eprintln!("{except}"),
        }
        self.ftf_watch_directory.set_tool_tip_text(
            etomo_autodoc::get_tooltip(
                read_only(autodoc),
                Some(series_watcher_param::WATCH_DIRECTORY_KEY),
            )
            .as_deref(),
        );
    }

    /// Java package-private `getSeriesWatcherAxisType()`.
    pub fn get_series_watcher_axis_type(&self) -> Option<AxisType> {
        if !self.is_series_watcher_on() {
            return None;
        }
        Some(self.series_watcher_panel().get_axis_type())
    }

    /// Java package-private `getSeriesWatcherSurfacesToAnalyze2()`.
    pub fn get_series_watcher_surfaces_to_analyze2(&self) -> Option<bool> {
        if !self.is_series_watcher_on() {
            return None;
        }
        Some(self.series_watcher_panel().is_two_surfaces())
    }

    /// Java private `sendQueueTableEvents()`.
    fn send_queue_table_events(&self) {
        self.send_queue_table_event(Some(self.get_split_batch_queue_table_event()));
        self.send_queue_table_event(self.get_only_queue_type_queue_table_event());
        self.send_queue_table_event(self.get_secondary_queue_table_event());
        self.send_queue_table_event(Some(self.get_number_jobs_changed_queue_table_event()));
    }

    /// Java `display()` (FieldDisplayer).  Display the correct tab.
    pub fn display_void(&self) {
        self.display_tab(Some(BatchRunTomoTab::Batch));
    }

    /// The tabbed pane (Slint bridge).
    pub fn get_tabbed_pane(&self) -> Rc<TabbedPane> {
        Rc::clone(&self.tabbed_pane)
    }

    /// The dataset table (Slint bridge and driver).
    pub fn get_table(&self) -> Rc<BatchRunTomoTable> {
        Rc::clone(self.table())
    }

    /// The step panel (Slint bridge and driver).
    pub fn get_step_panel(&self) -> Rc<BatchRunTomoStepPanel> {
        Rc::clone(self.step_panel())
    }

    /// The series watcher panel (Slint bridge and driver).
    pub fn get_series_watcher_panel(&self) -> Rc<SeriesWatcherPanel> {
        Rc::clone(self.series_watcher_panel())
    }
}

impl SeriesWatcherParent for BatchRunTomoDialog {
    /// Java `isSeriesWatcherOn()`.
    fn is_series_watcher_on(&self) -> bool {
        self.cb_use_series_watcher.is_selected()
    }

    /// Java `equalsSeriesWatcherActionCommand(String)`.
    fn equals_series_watcher_action_command(&self, action_command: &str) -> bool {
        self.cb_use_series_watcher
            .equals_action_command(Some(action_command))
    }
}

impl BatchRunTomoDialog {
    /// Java `isSeriesWatcherOn()`.
    pub fn is_series_watcher_on(&self) -> bool {
        SeriesWatcherParent::is_series_watcher_on(self)
    }

    /// Java `equalsSeriesWatcherActionCommand(String)`.
    pub fn equals_series_watcher_action_command(&self, action_command: &str) -> bool {
        SeriesWatcherParent::equals_series_watcher_action_command(self, action_command)
    }

    /// Java `startOver()`.
    pub fn start_over(&self) {
        self.start_over_impl();
    }
}

impl BrowsingDirectory for BatchRunTomoDialog {
    /// Java `getBrowsingDir()`.
    fn get_browsing_dir(&self) -> Option<PathBuf> {
        let mut validbrowsing_directory = self.validbrowsing_directory.borrow_mut();
        if validbrowsing_directory.is_none() {
            let mut directory = ValidDirectory::new(Some(self.base_manager()));
            directory.set_to_property_user_dir();
            *validbrowsing_directory = Some(directory);
        }
        let validbrowsing_directory = validbrowsing_directory.as_ref().unwrap();
        if !self.rb_deliver_off.is_selected() {
            return validbrowsing_directory.get_void();
        }
        // If the stacks are not going to be delivered there must only one stack in
        // each directory.
        validbrowsing_directory.get_parent()
    }

    /// Java `setBrowsingDir(File)`.  Set valid browsing directory.
    fn set_browsing_dir(&self, input: Option<&Path>) {
        let mut validbrowsing_directory = self.validbrowsing_directory.borrow_mut();
        if validbrowsing_directory.is_none() && input.is_some() {
            *validbrowsing_directory = Some(ValidDirectory::new(Some(self.base_manager())));
        }
        if let Some(validbrowsing_directory) = validbrowsing_directory.as_mut() {
            validbrowsing_directory.set_file(input);
        }
    }
}

impl ContextMenu for BatchRunTomoDialog {
    /// Java `popUpContextMenu(MouseEvent)`.  Right mouse button context menu.
    fn pop_up_context_menu(&self, mouse_event: &MouseEvent) {
        let man_pagelabel = ["batchruntomo".to_owned(), "3dmod".to_owned()];
        let man_page = ["batchruntomo.html".to_owned(), "3dmod.html".to_owned()];
        let series_watcher_mode = self.is_series_watcher_on();
        let log_file_label = [if !series_watcher_mode {
            "batchruntomo".to_owned()
        } else {
            "serieswatcher".to_owned()
        }];
        let log_file = [if !series_watcher_mode {
            format!(
                "{}.log",
                self.get_root_name().unwrap_or_else(|| "null".to_owned())
            )
        } else {
            "serieswatcher.log".to_owned()
        }];
        let cur_tab = self.cur_tab.get();
        let anchor = if cur_tab == Some(BatchRunTomoTab::Batch) {
            Some("BatchSetup")
        } else if cur_tab == Some(BatchRunTomoTab::Stacks) {
            Some("Stacks")
        } else if cur_tab == Some(BatchRunTomoTab::Dataset) {
            Some("SetValues")
        } else if cur_tab == Some(BatchRunTomoTab::Run) {
            Some("Run")
        } else {
            None
        };
        if let Err(e) = ContextPopup::new_component_mouse_event_string_string_string_array_string_array_string_array_string_array_base_manager_axis_id(
            &self.pnl_root,
            mouse_event,
            anchor,
            Some(context_popup::BATCHRUNTOMO_GUIDE),
            &man_pagelabel,
            &man_page,
            Some(&log_file_label),
            Some(&log_file),
            self.manager,
            self.axis_id,
        ) {
            eprintln!("{e}");
        }
    }
}

impl Expandable for BatchRunTomoDialog {
    /// Java `expand(ExpandButton)`.
    fn expand_expand_button(&self, button: &Rc<ExpandButton>) {
        let expanded = button.is_expanded();
        // Dataset and Run tabs have separate panel headers. They should be kept up to
        // date with each other.
        if self.ph_dataset_table.equals_open_close(button) {
            self.pnl_dataset_table_body.set_visible(expanded);
        } else if self.ph_run_table.equals_open_close(button) {
            self.pnl_run_table_body.set_visible(expanded);
        }
        ui_harness::with(|harness| harness.pack_base_manager(Some(self.base_manager())));
    }

    /// Java `expand(GlobalExpandButton)`: empty.
    fn expand_global_expand_button(&self, _button: &Rc<GlobalExpandButton>) {}
}

impl FieldDisplayer for BatchRunTomoDialog {
    /// Java `display()`.  Display the correct tab.
    fn display_void(&self) {
        BatchRunTomoDialog::display_void(self);
    }

    /// Java `display(UIComponent)`.  No field specific displaying required.
    fn display_ui_component(&self, _ui_component: Option<&dyn UIComponent>) {
        BatchRunTomoDialog::display_void(self);
    }
}

impl StatusChanger for BatchRunTomoDialog {
    /// Java `addStatusChangeListener(StatusChangeListener)`.  Add listeners for
    /// StartOver button.
    fn add_status_change_listener(&self, listener: Option<Rc<dyn StatusChangeListener>>) {
        let Some(listener) = listener else {
            return;
        };
        let mut listeners = self.listeners.lock().unwrap();
        let mut new_element = false;
        if listeners.is_none() {
            *listeners = Some(Vec::new());
            new_element = true;
        }
        let listeners = listeners.as_mut().unwrap();
        if !new_element
            && listeners
                .iter()
                .any(|existing| Rc::ptr_eq(existing.get(), &listener))
        {
            return;
        }
        listeners.push(EdtRef::new(listener));
    }
}

impl StatusChangeListener for BatchRunTomoDialog {
    /// Java `statusChanged(Status)`.
    fn status_changed_status(&self, status: Option<StatusRef>) {
        BatchRunTomoDialog::status_changed_status(self, status);
    }

    /// Java `statusChanged(StatusChangeEvent)`.
    fn status_changed_event(&self, status_change_event: Option<&dyn StatusChangeEvent>) {
        self.status_changed_event_impl(status_change_event);
    }

    /// Java `startOver()`.
    fn start_over(&self) {
        self.start_over_impl();
    }
}

impl DatasetInfoDisplay for BatchRunTomoDialog {
    /// Java `getDatasetName()`.  Returns null if empty.
    fn get_dataset_name(&self) -> Option<String> {
        let dataset_name = self.get_root_name();
        if utilities::is_empty(dataset_name.as_deref()) {
            return None;
        }
        dataset_name
    }

    /// Java `getDatasetAbsolutePath()`.  Returns null if empty.
    fn get_dataset_absolute_path(&self) -> Option<String> {
        let root_dir = self.get_root_dir()?;
        Some(utilities::java_io_file_get_absolute_path(
            &root_dir.to_string_lossy(),
        ))
    }
}

impl AbstractParallelDialog for BatchRunTomoDialog {
    /// Java `getParameters(ParallelParam)`.
    fn get_parameters(&self, parallel_param: &mut dyn ParallelParam) {
        // `parallelParam instanceof ProcesschunksParam`, then the cast.
        let Some(param) =
            (parallel_param as &mut dyn std::any::Any).downcast_mut::<ProcesschunksParam>()
        else {
            return;
        };
        if self.sp_number_of_jobs_to_make.is_visible()
            && self.sp_number_of_jobs_to_make.is_enabled()
        {
            param.set_multi_proc_string(Some(&self.sp_number_of_jobs_to_make.get_text()));
        } else if self.sp_queue_number_of_jobs_to_make.is_visible()
            && self.sp_queue_number_of_jobs_to_make.is_enabled()
        {
            param.set_multi_proc_string(Some(&self.sp_queue_number_of_jobs_to_make.get_text()));
        } else {
            param.set_multi_proc_int(1);
        }
        if self.rb_gpu_machine_list_off.is_selected() {
            param.reset_gpu_machine_list();
        } else if self.rb_gpu_machine_list_local.is_selected() {
            param.set_gpu_machine_list(
                Network::get_local_host_name(
                    self.base_manager(),
                    self.axis_id,
                    self.manager.get_property_user_dir().as_deref(),
                )
                .as_deref(),
            );
        }
    }

    /// Java `getDialogType()`.
    fn get_dialog_type(&self) -> DialogType {
        DIALOG_TYPE
    }
}

impl QueueTableListener for BatchRunTomoDialog {
    /// Java `queueTableEventAction(QueueTableEvent)`.
    fn queue_table_event_action(&self, event: &QueueTableEvent) {
        // batchRunTomoState.handleStatusEvent(event);
        // Check the child class first because a QueueTableDataEvent is also an
        // instance of QueueTableEvent.
        match event {
            QueueTableEvent::QueueSelected {
                queue_mode,
                maximum,
            } => {
                self.queue_mode.set(Some(*queue_mode));
                if let Some(s_maximum) = maximum {
                    // Set the maximum number of the jobs to the current queue's maximum.
                    let maximum = converter::to_integer(Some(s_maximum));
                    if let Some(mut maximum) = maximum {
                        // The maximum jobs need to be reduced to at least 1/2 to prevent
                        // deadlock. If the user selects all nodes, then processchunks
                        // will submit batchruntomo to every node. Then deadlock will
                        // happen when each batchruntomo submit its processchunks runs to
                        // the same queue. This is also true for mode 3-1 runs because
                        // the secondary queue is only used for GPU runs.
                        if self.queue_mode.get() == Some(QueueMode::QueueWithSingleCpu) {
                            maximum /= DUAL_SELECTION_MAX_DIVISOR;
                        }
                        self.sp_queue_number_of_jobs_to_make
                            .set_maximum(Some(maximum));
                        // The new maximum may have invalidated the minimum and/or value.
                        self.sp_queue_number_of_jobs_to_make.adjust_to_maximum();
                        self.send_queue_table_event(Some(
                            self.get_number_jobs_changed_queue_table_event(),
                        ));
                    }
                }
            }
            QueueTableEvent::NumberJobsChanged(_) | QueueTableEvent::OnlyQueueType(_) => {}
            QueueTableEvent::Displayed => {
                self.queue_table_displayed.set(true);
                self.update_display_void();
                self.send_queue_table_event(self.get_only_queue_type_queue_table_event());
                self.send_queue_table_event(self.get_secondary_queue_table_event());
            }
            QueueTableEvent::Hidden => {
                self.queue_table_displayed.set(false);
                // Prevent an invalid state. If the queue table has been hidden and split
                // batch is selected, then treat this as selecting multiple CPUs.
                // if (cbSplitBatch.isEnabled() && cbSplitBatch.isSelected()) {
                // cbCPUMachineList.setSelected(true);
                // }
            }
            _ => {}
        }
        let this: Rc<dyn ProcessInterface> = self.this_rc();
        self.mediator_set_method(&this);
        self.update_display_void();
    }
}

impl ProcessInterface for BatchRunTomoDialog {
    /// Java `updateGpu(boolean)`: no effect because queue is not available.
    fn update_gpu(&self, _disable_gpu: bool) {}

    /// Java `getProcessingMethod()`.  Always returns a processing method.
    fn get_processing_method(&self) -> ProcessingMethod {
        if self.queue_table_displayed.get() {
            return ProcessingMethod::Queue;
        }
        if (self.cb_split_batch.is_enabled() && self.cb_split_batch.is_selected())
            || (self.cb_cpu_machine_list.is_enabled() && self.cb_cpu_machine_list.is_selected())
        {
            return ProcessingMethod::PpCpu;
        }
        if self.rb_gpu_machine_list.is_enabled() && self.rb_gpu_machine_list.is_selected() {
            return ProcessingMethod::PpGpu;
        }
        if self.rb_gpu_machine_list_local.is_enabled()
            && self.rb_gpu_machine_list_local.is_selected()
        {
            return ProcessingMethod::LocalGpu;
        }
        ProcessingMethod::LocalCpu
    }

    /// Java `getSecondaryProcessingMethod()`.  Returns a processing method when there
    /// are two non-default methods in force, otherwise returns null.
    fn get_secondary_processing_method(&self) -> Option<ProcessingMethod> {
        self.update_display_void();
        let method = self.get_processing_method();
        if method != ProcessingMethod::PpCpu {
            return None;
        }
        // two non-default processing methods are in force
        if self.rb_gpu_machine_list.is_enabled() && self.rb_gpu_machine_list.is_selected() {
            return Some(ProcessingMethod::PpGpu);
        }
        if self.rb_gpu_machine_list_local.is_enabled()
            && self.rb_gpu_machine_list_local.is_selected()
        {
            return Some(ProcessingMethod::LocalGpu);
        }
        None
    }

    /// Java `lockProcessingMethod(boolean)`: no effect because the processing method
    /// is not used for running processes by etomo.
    fn lock_processing_method(&self, _lock: bool) {}

    /// Java `setMethod(ProcessingMethod)`.
    fn set_method(&self, processing_method: ProcessingMethod) {
        if let Some(mediator) = &self.mediator {
            mediator.set_method_process_interface_processing_method(
                &(self.this_rc() as Rc<dyn ProcessInterface>),
                processing_method,
            );
        }
    }

    /// Java `isUseGpu()`.
    fn is_use_gpu(&self) -> bool {
        true
    }

    /// Java `setUseQueueCheckBox(ButtonComponent)`: empty.
    fn set_use_queue_check_box(&self, _use_queue_check_box: Option<Rc<dyn ButtonComponent>>) {}

    /// Java `addQueueTableListener(QueueTableListener)`.
    fn add_queue_table_listener(&self, listener: Rc<dyn QueueTableListener>) {
        {
            let mut queue_table_listener_array = self.queue_table_listener_array.borrow_mut();
            queue_table_listener_array
                .get_or_insert_with(Vec::new)
                .push(Rc::clone(&listener));
        }
        // Since listeners can be added at any time, send the status to each new
        // listener.  (A null event is not sent; see sendQueueTableEvent.)
        listener.queue_table_event_action(&self.get_split_batch_queue_table_event());
        if let Some(event) = self.get_only_queue_type_queue_table_event() {
            listener.queue_table_event_action(&event);
        }
        if let Some(event) = self.get_secondary_queue_table_event() {
            listener.queue_table_event_action(&event);
        }
        listener.queue_table_event_action(&self.get_number_jobs_changed_queue_table_event());
    }

    /// Java `removeQueueTableListener(QueueTableListener)`.
    fn remove_queue_table_listener(&self, listener: &Rc<dyn QueueTableListener>) {
        if let Some(queue_table_listener_array) =
            self.queue_table_listener_array.borrow_mut().as_mut()
            && let Some(index) = queue_table_listener_array
                .iter()
                .position(|existing| Rc::ptr_eq(existing, listener))
        {
            queue_table_listener_array.remove(index);
        }
    }
}

impl SwingComponent for BatchRunTomoDialog {
    fn get_component(&self) -> Rc<JComponent> {
        Rc::clone(&self.pnl_root)
    }
}

impl UIComponent for BatchRunTomoDialog {
    fn get_ui_component(&self) -> &dyn SwingComponent {
        self
    }

    fn get_component(&self) -> Rc<JComponent> {
        Rc::clone(&self.pnl_root)
    }
}
