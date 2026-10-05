//! `IMOD/Etomo/src/etomo/ui/swing/BatchRunTomoTable.java`.
//!
//! The dataset table of the batchruntomo dialog: one `BatchRunTomoRow` per stack, with
//! three layouts (the Stacks, Dataset and Run tabs).  An event dispatch thread object,
//! created as `Rc<Self>` by [`BatchRunTomoTable::get_instance`].
//! `BatchRunTomoTable extends HighlightableTable`: the superclass is the `base` field;
//! the private inner class `RowList` is [`RowList`].

use std::cell::{Cell, RefCell};
use std::path::PathBuf;
use std::rc::{Rc, Weak};
use std::sync::Arc;

use super::batch_run_tomo_dataset_dialog::BatchRunTomoDatasetDialog;
use super::batch_run_tomo_dialog::BatchRunTomoDialog;
use super::batch_run_tomo_row::{BasicDirectives, BatchRunTomoRow};
use super::cell::CellVirtual;
use super::check_box_cell::CheckBoxCell;
use super::expand_button::{self, ExpandButton};
use super::expandable::Expandable;
use super::file_chooser::{self, FileChooser};
use super::global_expand_button::GlobalExpandButton;
use super::header_cell::HeaderCell;
use super::highlightable::Highlightable;
use super::highlightable_table::{HighlightableTable, HighlightableTableVirtual};
use super::popup::Popup;
use super::series_watcher_parent::SeriesWatcherParent;
use super::single_line_button::SingleLineButton;
use super::swing_component::SwingComponent;
use super::template_panel::TemplatePanel;
use super::ui_harness;
use super::viewable::Viewable;
use super::viewport::Viewport;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::batch_run_tomo_manager::BatchRunTomoManager;
use crate::imod::etomo::comscript::batchruntomo_param::BatchruntomoParam;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::jdk::{
    ActionEvent, ActionListener, GRID_BAG_BOTH, GRID_BAG_CENTER, GRID_BAG_REMAINDER,
    GridBagConstraints, GridBagLayout, JComponent,
};
use crate::imod::etomo::logic::batch_tool::TemplateValues;
use crate::imod::etomo::logic::dataset_tool;
use crate::imod::etomo::storage::directive_file_collection::DirectiveFileCollection;
use crate::imod::etomo::storage::name_value_pair_list::NameValuePairList;
use crate::imod::etomo::storage::stack_file_filter::StackFileFilter;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::batch_run_tomo_meta_data::BatchRunTomoMetaData;
use crate::imod::etomo::r#type::batch_run_tomo_status::{self, BatchRunTomoStatus};
use crate::imod::etomo::r#type::ending_step::EndingStep;
use crate::imod::etomo::r#type::frame_status::FrameStatus;
use crate::imod::etomo::r#type::image_filename_style::ImageFilenameStyle;
use crate::imod::etomo::r#type::run_list::RunList;
use crate::imod::etomo::r#type::run_type::RunType;
use crate::imod::etomo::r#type::series_watcher_meta_data::SeriesWatcherMetaData;
use crate::imod::etomo::r#type::status::StatusRef;
use crate::imod::etomo::r#type::status_change_boolean_event::StatusChangeBooleanEvent;
use crate::imod::etomo::r#type::status_change_event::StatusChangeEvent;
use crate::imod::etomo::r#type::status_change_listener::StatusChangeListener;
use crate::imod::etomo::r#type::status_changer::StatusChanger;
use crate::imod::etomo::r#type::table_reference::{PutError, TableReference};
use crate::imod::etomo::ui::batch_run_tomo_tab::BatchRunTomoTab;
use crate::imod::etomo::ui::browsing_directory::BrowsingDirectory;
use crate::imod::etomo::ui::field_displayer::FieldDisplayer;
use crate::imod::etomo::ui::preferred_table_size::PreferredTableSize;
use crate::imod::etomo::ui::shared_strings;
use crate::imod::etomo::ui::table_component::TableComponent;
use crate::imod::etomo::ui::table_listener::TableListener;
use crate::imod::etomo::ui::ui_component::UIComponent;
use crate::imod::etomo::util::unique_key::UniqueKey;

/// Java private static final `STACK_TITLE`.
const STACK_TITLE: &str = "Stack";
/// Java private static final `MAX_HEADER_ROWS`.
const MAX_HEADER_ROWS: usize = 3;
/// Java private static final `NUM_STACKS_HEADER_ROWS`.
const NUM_STACKS_HEADER_ROWS: usize = MAX_HEADER_ROWS;
/// Java private static final `NUM_DATASET_HEADER_ROWS`.
const NUM_DATASET_HEADER_ROWS: usize = 1;
/// Java private static final `NUM_RUN_HEADER_ROWS`.
const NUM_RUN_HEADER_ROWS: usize = 2;
/// Java private static final `AXIS_ID`.
const AXIS_ID: AxisID = AxisID::Only;

/// Java package-private static final `STATUS_LABEL`.
pub const STATUS_LABEL: &str = "Status";
/// Java package-private static final `STEP_LABEL`.
pub const STEP_LABEL: &str = "Reached";
/// Java package-private static final `RUN_LABEL`.
pub const RUN_LABEL: &str = "Run";
/// Java private static final `UNIQUE_KEY`.
const UNIQUE_KEY: &str = "BatchRunTomo";
/// Java package-private static final `SURFACES_TO_ANALYZE_LABEL1`.
pub const SURFACES_TO_ANALYZE_LABEL1: &str = "Beads";
/// Java package-private static final `SURFACES_TO_ANALYZE_LABEL2`.
pub const SURFACES_TO_ANALYZE_LABEL2: &str = "on Two";
/// Java package-private static final `SURFACES_TO_ANALYZE_LABEL3`.
pub const SURFACES_TO_ANALYZE_LABEL3: &str = "Surfaces";
/// Java package-private static final `DUAL_LABEL1`.
pub const DUAL_LABEL1: &str = "Dual";
/// Java package-private static final `DUAL_LABEL2`.
pub const DUAL_LABEL2: &str = "Axis";
/// Java private static final `OPEN_LABEL`.
const OPEN_LABEL: &str = "Open";
/// Java package-private static final `DATASET_LABEL1`.
pub const DATASET_LABEL1: &str = OPEN_LABEL;
/// Java package-private static final `DATASET_LABEL2`.
pub const DATASET_LABEL2: &str = "Set";
/// Java package-private static final `REC_LABEL1`.
pub const REC_LABEL1: &str = OPEN_LABEL;
/// Java package-private static final `REC_LABEL2`.
pub const REC_LABEL2: &str = "Rec";
/// Java package-private static final `LOG_LABEL`.
pub const LOG_LABEL: &str = "Log";
/// Java package-private static final `PROJ_LOG_LABEL1`.
pub const PROJ_LOG_LABEL1: &str = "Proj";
/// Java package-private static final `PROJ_LOG_LABEL2`.
pub const PROJ_LOG_LABEL2: &str = LOG_LABEL;
/// Java package-private static final `LOG_LABEL1`.
pub const LOG_LABEL1: &str = "BRT";
/// Java package-private static final `LOG_LABEL2`.
pub const LOG_LABEL2: &str = LOG_LABEL;
/// Java package-private static final `CUR_AXIS_LABEL`.
pub const CUR_AXIS_LABEL: &str = "Axis";

/// Java package-private static final nested class `DatasetColumn`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DatasetColumn {
    /// Java private final `index`.
    index: i32,
}

impl DatasetColumn {
    /// Java `NUMBER`.
    pub const NUMBER: DatasetColumn = DatasetColumn { index: 0 };
    /// Java `STACK`.
    pub const STACK: DatasetColumn = DatasetColumn { index: 1 };
    /// Java `EDIT_DATASET`.
    pub const EDIT_DATASET: DatasetColumn = DatasetColumn { index: 2 };
    /// Java `TOTAL`.
    pub const TOTAL: i32 = 3;

    /// Java package-private `getIndex()`.
    pub fn get_index(&self) -> i32 {
        self.index
    }
}

/// `HeaderCell[]` of a fixed size, filled by `createPanel`.
fn header_array(size: usize) -> Vec<Rc<HeaderCell>> {
    (0..size).map(|_| HeaderCell::new_void()).collect()
}

/// Java `final class BatchRunTomoTable extends HighlightableTable implements Viewable,
/// Expandable, ActionListener, StatusChangeListener, StatusChanger, UIComponent,
/// SwingComponent, FieldDisplayer`.
pub struct BatchRunTomoTable {
    /// Java superclass `HighlightableTable`.
    base: HighlightableTable,
    /// Java private final `pnlRoot`.
    pnl_root: Rc<JComponent>,
    /// Java private final `pnlTable`.
    pnl_table: Rc<JComponent>,
    /// Java private final `layout`.
    layout: GridBagLayout,
    /// Java private final `constraints`.
    constraints: RefCell<GridBagConstraints>,
    /// Java private final `hcNumber[]` (both tabs).
    hc_number: RefCell<Vec<Rc<HeaderCell>>>,
    /// Java private final `hcStack[]`.
    hc_stack: RefCell<Vec<Rc<HeaderCell>>>,
    /// Java private final `hcDual[]` (stacks tab).
    hc_dual: RefCell<Vec<Rc<HeaderCell>>>,
    /// Java private final `hcMontage[]`.
    hc_montage: RefCell<Vec<Rc<HeaderCell>>>,
    /// Java private final `hcSkip[]`.
    hc_skip: RefCell<Vec<Rc<HeaderCell>>>,
    /// Java private final `hcSkipB`.
    hc_skip_b: Rc<HeaderCell>,
    /// Java private final `hcBoundaryModel[]`.
    hc_boundary_model: RefCell<Vec<Rc<HeaderCell>>>,
    /// Java private final `hcSurfacesToAnalyze[]`.
    hc_surfaces_to_analyze: RefCell<Vec<Rc<HeaderCell>>>,
    /// Java private final `hc3dmod[]`.
    hc3dmod: RefCell<Vec<Rc<HeaderCell>>>,
    /// Java private final `hc3dmodB`.
    hc3dmod_b: Rc<HeaderCell>,
    /// Java private final `hcEditDataset[]` (dataset tab).
    hc_edit_dataset: RefCell<Vec<Rc<HeaderCell>>>,
    /// Java private final `hcStatus[]` (run tab).
    hc_status: RefCell<Vec<Rc<HeaderCell>>>,
    /// Java private final `hcStep[]`.
    hc_step: RefCell<Vec<Rc<HeaderCell>>>,
    /// Java private final `hcCurAxis[]`.
    hc_cur_axis: RefCell<Vec<Rc<HeaderCell>>>,
    /// Java private final `hcRun`.
    hc_run: Rc<HeaderCell>,
    /// Java private final `cbcRunToggle`.
    cbc_run_toggle: Rc<CheckBoxCell>,
    /// Java private final `hcDataset[]`.
    hc_dataset: RefCell<Vec<Rc<HeaderCell>>>,
    /// Java private final `hcRec[]`.
    hc_rec: RefCell<Vec<Rc<HeaderCell>>>,
    /// Java private final `hcProjLog[]`.
    hc_proj_log: RefCell<Vec<Rc<HeaderCell>>>,
    /// Java private final `hcLog[]`.
    hc_log: RefCell<Vec<Rc<HeaderCell>>>,
    /// Java private final `btnAdd`.
    btn_add: Rc<SingleLineButton>,
    /// Java private final `btnCopyDown`.
    btn_copy_down: Rc<SingleLineButton>,
    /// Java private final `btnDelete`.
    btn_delete: Rc<SingleLineButton>,
    /// Java private final `pnlStackButtons`.
    pnl_stack_buttons: Rc<JComponent>,
    /// Java private final `preferredTableSize`.
    preferred_table_size: PreferredTableSize,
    /// Java private final `viewport`.
    viewport: Rc<Viewport>,

    /// Java private final `rowList`.
    row_list: RowList,
    /// Java private final `manager`.
    manager: &'static BatchRunTomoManager,
    /// Java private final `btnStack`.
    btn_stack: Rc<ExpandButton>,
    /// Java private final `focusableParents`.
    focusable_parents: Vec<Rc<JComponent>>,
    /// Java private final `dialog`.
    dialog: Weak<BatchRunTomoDialog>,
    /// Java private final `basicDirectives`.
    basic_directives: BasicDirectives,
    /// Java private final `seriesWatcherParent`.
    series_watcher_parent: Weak<dyn SeriesWatcherParent>,

    /// Java private `listeners`, initially null.
    listeners: RefCell<Option<Vec<Rc<dyn StatusChangeListener>>>>,
    /// Java private `curTab`, initially null.
    cur_tab: Cell<Option<BatchRunTomoTab>>,
    /// Java private `status`, initially `BatchRunTomoStatus.DEFAULT`.
    status: Cell<Option<BatchRunTomoStatus>>,
    /// Java private `montageFrame`, initially false.
    montage_frame: Cell<bool>,
    /// Java private `singleFrame`, initially false.
    single_frame: Cell<bool>,
    /// Java private `rowRunning`, initially false.
    row_running: Cell<bool>,
    /// Java private `seriesWatcher`, initially false.
    #[allow(dead_code)]
    series_watcher: Cell<bool>,
    /// Java `this`.
    this: Weak<BatchRunTomoTable>,
}

impl BatchRunTomoTable {
    /// Java private `BatchRunTomoTable(BatchRunTomoManager, BatchRunTomoDialog,
    /// Set<DirectiveDef>, TableReference, JComponent, SeriesWatcherParent)`, with the
    /// field initialisers.
    fn new(
        manager: &'static BatchRunTomoManager,
        dialog: Weak<BatchRunTomoDialog>,
        basic_directives: BasicDirectives,
        table_reference: Arc<TableReference>,
        focusable_parent: Rc<JComponent>,
        series_watcher_parent: Weak<dyn SeriesWatcherParent>,
    ) -> Rc<BatchRunTomoTable> {
        let batch_table_size = etomo_director::INSTANCE
            .with_user_configuration(|user_config| user_config.get_batch_table_size().get_int());
        Rc::new_cyclic(|this: &Weak<BatchRunTomoTable>| {
            let pnl_root = JComponent::new_panel();
            let viewable: Weak<dyn Viewable> = this.clone();
            let expandable: Weak<dyn Expandable> = this.clone();
            let dialog_expandable: Weak<dyn Expandable> = dialog.clone();
            BatchRunTomoTable {
                // super(UNIQUE_KEY)
                base: HighlightableTable::new(UNIQUE_KEY),
                pnl_table: JComponent::new_panel(),
                layout: GridBagLayout::new(),
                constraints: RefCell::new(GridBagConstraints::default()),
                hc_number: RefCell::new(header_array(NUM_STACKS_HEADER_ROWS)),
                hc_stack: RefCell::new(header_array(NUM_STACKS_HEADER_ROWS)),
                hc_dual: RefCell::new(header_array(NUM_STACKS_HEADER_ROWS)),
                hc_montage: RefCell::new(header_array(NUM_STACKS_HEADER_ROWS)),
                hc_skip: RefCell::new(header_array(NUM_STACKS_HEADER_ROWS)),
                hc_skip_b: HeaderCell::new_string(Some("from B")),
                hc_boundary_model: RefCell::new(header_array(NUM_STACKS_HEADER_ROWS)),
                hc_surfaces_to_analyze: RefCell::new(header_array(NUM_STACKS_HEADER_ROWS)),
                hc3dmod: RefCell::new(header_array(NUM_STACKS_HEADER_ROWS)),
                hc3dmod_b: HeaderCell::new_string(Some("B")),
                hc_edit_dataset: RefCell::new(header_array(NUM_DATASET_HEADER_ROWS)),
                hc_status: RefCell::new(header_array(NUM_RUN_HEADER_ROWS)),
                hc_step: RefCell::new(header_array(NUM_RUN_HEADER_ROWS)),
                hc_cur_axis: RefCell::new(header_array(NUM_RUN_HEADER_ROWS)),
                hc_run: HeaderCell::new_string(Some(RUN_LABEL)),
                cbc_run_toggle: CheckBoxCell::get_header_background_named_instance(
                    Some(RUN_LABEL),
                    Some("Toggle"),
                ),
                hc_dataset: RefCell::new(header_array(NUM_RUN_HEADER_ROWS)),
                hc_rec: RefCell::new(header_array(NUM_RUN_HEADER_ROWS)),
                hc_proj_log: RefCell::new(header_array(NUM_RUN_HEADER_ROWS)),
                hc_log: RefCell::new(header_array(NUM_RUN_HEADER_ROWS)),
                btn_add: SingleLineButton::new_string(Some("Add Stack(s)")),
                btn_copy_down: SingleLineButton::new_string(Some("Copy Down")),
                btn_delete: SingleLineButton::new_string(Some("Delete")),
                pnl_stack_buttons: JComponent::new_panel(),
                preferred_table_size: PreferredTableSize::new(DatasetColumn::TOTAL),
                viewport: Viewport::new(viewable, batch_table_size, Some(UNIQUE_KEY)),
                row_list: RowList::new(this.clone(), table_reference),
                manager,
                btn_stack: ExpandButton::get_instance_expandable_expandable_type(
                    Some(expandable),
                    Some(dialog_expandable),
                    Some(&expand_button::Type::MORE),
                ),
                focusable_parents: vec![pnl_root.clone(), focusable_parent],
                pnl_root,
                dialog,
                basic_directives,
                series_watcher_parent,
                listeners: RefCell::new(None),
                cur_tab: Cell::new(None),
                status: Cell::new(Some(batch_run_tomo_status::DEFAULT)),
                montage_frame: Cell::new(false),
                single_frame: Cell::new(false),
                row_running: Cell::new(false),
                series_watcher: Cell::new(false),
                this: this.clone(),
            }
        })
    }

    /// Java package-private static `getInstance(BatchRunTomoManager, BatchRunTomoDialog,
    /// Set<DirectiveDef>, TableReference, JComponent, SeriesWatcherParent)`.
    pub fn get_instance(
        manager: &'static BatchRunTomoManager,
        dialog: Weak<BatchRunTomoDialog>,
        basic_directives: BasicDirectives,
        table_reference: Arc<TableReference>,
        focusable_parent: Rc<JComponent>,
        series_watcher_parent: Weak<dyn SeriesWatcherParent>,
    ) -> Rc<BatchRunTomoTable> {
        let instance = BatchRunTomoTable::new(
            manager,
            dialog,
            basic_directives,
            table_reference,
            focusable_parent,
            series_watcher_parent,
        );
        instance.create_panel();
        instance.set_tooltips();
        instance.add_listeners();
        instance
    }

    fn dialog(&self) -> Option<Rc<BatchRunTomoDialog>> {
        self.dialog.upgrade()
    }

    fn base_manager(&self) -> &'static dyn BaseManager {
        self.manager
    }

    /// Java package-private `msgRowRunning(boolean)`.
    pub fn msg_row_running(&self, running: bool) {
        if self.row_running.get() != running {
            self.row_running.set(running);
            std::thread::sleep(std::time::Duration::from_millis(1));
            ui_harness::with(|harness| harness.pack_base_manager(Some(self.base_manager())));
        }
    }

    /// Java package-private `hasDual()`.
    pub fn has_dual(&self) -> bool {
        self.row_list.has_dual()
    }

    /// Java private `createPanel()`.
    fn create_panel(&self) {
        // init
        let pnl_view = JComponent::new_panel();
        // Swing layout: btnAdd, btnCopyDown and btnDelete setToPreferredSize().
        self.btn_stack.set_name(Some(STACK_TITLE));
        self.viewport.init_paging();
        if let Some(this) = self.this.upgrade() {
            let table: Weak<dyn HighlightableTableVirtual> = Rc::downgrade(&this) as _;
            self.base.init_highlight_hotkeys(table);
        }
        // all table tabs
        {
            let mut hc_number = self.hc_number.borrow_mut();
            hc_number[0] = HeaderCell::new_string(Some("#"));
            hc_number[1] = HeaderCell::new_void();
            hc_number[2] = HeaderCell::new_void();
            let mut hc_stack = self.hc_stack.borrow_mut();
            hc_stack[0] = HeaderCell::new_string(Some(STACK_TITLE));
            hc_stack[1] = HeaderCell::new_void();
            hc_stack[2] = HeaderCell::new_void();
            // stacks tab
            let mut hc_dual = self.hc_dual.borrow_mut();
            hc_dual[0] = HeaderCell::new_string(Some(DUAL_LABEL1));
            hc_dual[1] = HeaderCell::new_string(Some(DUAL_LABEL2));
            hc_dual[2] = HeaderCell::new_void();
            let mut hc_montage = self.hc_montage.borrow_mut();
            hc_montage[0] = HeaderCell::new_string(Some("Montage"));
            hc_montage[1] = HeaderCell::new_void();
            hc_montage[2] = HeaderCell::new_void();
            let mut hc_skip = self.hc_skip.borrow_mut();
            hc_skip[0] = HeaderCell::new_string(Some("Exclude"));
            hc_skip[1] = HeaderCell::new_string(Some("Views"));
            hc_skip[2] = HeaderCell::new_string(Some("from A"));
            let mut hc_boundary_model = self.hc_boundary_model.borrow_mut();
            hc_boundary_model[0] = HeaderCell::new_string(Some("Boundary"));
            hc_boundary_model[1] = HeaderCell::new_string(Some("Model"));
            hc_boundary_model[2] = HeaderCell::new_void();
            let mut hc_surfaces_to_analyze = self.hc_surfaces_to_analyze.borrow_mut();
            hc_surfaces_to_analyze[0] = HeaderCell::new_string(Some(SURFACES_TO_ANALYZE_LABEL1));
            hc_surfaces_to_analyze[1] = HeaderCell::new_string(Some(SURFACES_TO_ANALYZE_LABEL2));
            hc_surfaces_to_analyze[2] = HeaderCell::new_string(Some("Surfaces"));
            let mut hc3dmod = self.hc3dmod.borrow_mut();
            hc3dmod[0] = HeaderCell::new_string(Some(OPEN_LABEL));
            hc3dmod[1] = HeaderCell::new_string(Some("Stack"));
            hc3dmod[2] = HeaderCell::new_string(Some("A"));
            // dataset tab
            self.hc_edit_dataset.borrow_mut()[0] = HeaderCell::new_string(Some("Specific Values"));
            // run tab
            let mut hc_status = self.hc_status.borrow_mut();
            hc_status[0] = HeaderCell::new_string(Some(STATUS_LABEL));
            hc_status[1] = HeaderCell::new_void();
            crate::imod::etomo::r#type::batch_run_tomo_dataset_state::BatchRunTomoDatasetState::set_preferred_width(
                Some(&hc_status[0].get_button()),
            );
            let mut hc_step = self.hc_step.borrow_mut();
            hc_step[0] = HeaderCell::new_string(Some(STEP_LABEL));
            hc_step[1] = HeaderCell::new_void();
            let mut hc_cur_axis = self.hc_cur_axis.borrow_mut();
            hc_cur_axis[0] = HeaderCell::new_string(Some(CUR_AXIS_LABEL));
            hc_cur_axis[1] = HeaderCell::new_void();
            let mut hc_dataset = self.hc_dataset.borrow_mut();
            hc_dataset[0] = HeaderCell::new_string(Some(DATASET_LABEL1));
            hc_dataset[1] = HeaderCell::new_string(Some(DATASET_LABEL2));
            let mut hc_rec = self.hc_rec.borrow_mut();
            hc_rec[0] = HeaderCell::new_string(Some(REC_LABEL1));
            hc_rec[1] = HeaderCell::new_string(Some(REC_LABEL2));
            let mut hc_proj_log = self.hc_proj_log.borrow_mut();
            hc_proj_log[0] = HeaderCell::new_string(Some(PROJ_LOG_LABEL1));
            hc_proj_log[1] = HeaderCell::new_string(Some(PROJ_LOG_LABEL2));
            let mut hc_log = self.hc_log.borrow_mut();
            hc_log[0] = HeaderCell::new_string(Some("BRT"));
            hc_log[1] = HeaderCell::new_string(Some("Log"));
            // preferred width of the dataset view
            self.preferred_table_size.add_column(
                DatasetColumn::NUMBER.index,
                Some(hc_number[0].clone() as Rc<dyn TableComponent>),
            );
            self.preferred_table_size.add_column_pair(
                DatasetColumn::STACK.index,
                Some(hc_stack[0].clone() as Rc<dyn TableComponent>),
                Some(self.btn_stack.clone() as Rc<dyn TableComponent>),
            );
            self.preferred_table_size.add_column(
                DatasetColumn::EDIT_DATASET.index,
                Some(self.hc_edit_dataset.borrow()[0].clone() as Rc<dyn TableComponent>),
            );
        }
        // Root
        self.pnl_root.add(&pnl_view);
        self.pnl_root.add(&self.pnl_stack_buttons);
        // View
        if let Some(paging_panel) = self.viewport.get_paging_panel() {
            pnl_view.add(&paging_panel);
        }
        pnl_view.add(&self.pnl_table);
        // stack Buttons
        self.pnl_stack_buttons
            .add(&SwingComponent::get_component(&*self.btn_add));
        self.pnl_stack_buttons
            .add(&SwingComponent::get_component(&*self.btn_copy_down));
        self.pnl_stack_buttons
            .add(&SwingComponent::get_component(&*self.btn_delete));
        // Table: `pnlTable.setLayout(layout)`; `setBorder(LineBorder.createBlackLineBorder())`.
        // constraints
        {
            let mut constraints = self.constraints.borrow_mut();
            constraints.fill = GRID_BAG_BOTH;
            constraints.anchor = GRID_BAG_CENTER;
            constraints.gridheight = 1;
            constraints.weighty = 1.0;
        }
        // update
        self.viewport.adjust_viewport(-1);
        self.update_display();
        self.status_changed_status(self.status.get().map(StatusRef::BatchRunTomoStatus));
    }

    /// Java `display()` (FieldDisplayer).
    pub fn display_void(&self) {
        if let Some(dialog) = self.dialog() {
            dialog.display_tab(Some(BatchRunTomoTab::Stacks));
        }
    }

    /// Java package-private `getAdvancedStartingBatch()`.
    pub fn get_advanced_starting_batch(&self) -> Option<NameValuePairList> {
        self.dialog()
            .and_then(|dialog| dialog.get_advanced_starting_batch())
    }

    /// Java package-private `getPreferredWidth()`.
    pub fn get_preferred_width(&self) -> i32 {
        self.preferred_table_size.get_preferred_width()
    }

    /// Java package-private `getDatasetDialog()`.
    pub fn get_dataset_dialog(&self) -> Rc<BatchRunTomoDatasetDialog> {
        self.dialog()
            .expect("the dialog owns its table")
            .get_dataset_dialog()
            .clone()
    }

    /// Java package-private `isParallelProcessing()`.
    pub fn is_parallel_processing(&self) -> bool {
        self.dialog()
            .is_some_and(|dialog| dialog.is_parallel_processing())
    }

    /// Java package-private `isTrackingMethodSeed()`.
    pub fn is_tracking_method_seed(&self) -> bool {
        self.dialog()
            .is_some_and(|dialog| dialog.is_tracking_method_seed())
    }

    /// Java package-private `getEarliestRunEndingStep()`.
    pub fn get_earliest_run_ending_step(&self) -> Option<EndingStep> {
        self.row_list.get_earliest_run_ending_step()
    }

    /// Java public `getImageFilenameStyle()`.
    pub fn get_image_filename_style(&self) -> Option<ImageFilenameStyle> {
        self.row_list.get_image_filename_style()
    }

    /// `cell.add(pnlTable, layout, constraints)`: the cell's `add` and the
    /// `layout.setConstraints` it makes.
    fn add_cell(&self, cell: &dyn CellVirtual, component: Rc<JComponent>) {
        cell.add(&self.pnl_table);
        self.layout
            .set_constraints(&component, &self.constraints.borrow());
    }

    /// `HeaderCell.add(pnlTable, layout, constraints)`.
    fn add_header(&self, cell: &Rc<HeaderCell>) {
        self.add_cell(&**cell, cell.get_component());
    }

    /// Java private `rebuildTable()`.
    fn rebuild_table(&self) {
        // remove table
        self.row_list.remove_all();
        self.pnl_table.remove_all();
        // header
        self.constraints.borrow_mut().weightx = 0.0;
        let cur_tab = self.cur_tab.get();
        if cur_tab == Some(BatchRunTomoTab::Stacks) {
            let num_rows = NUM_STACKS_HEADER_ROWS;
            for i in 0..num_rows {
                self.add_standard_headers(i);
                self.constraints.borrow_mut().gridwidth = 1;
                self.add_header(&self.hc_dual.borrow()[i].clone());
                self.add_header(&self.hc_montage.borrow()[i].clone());
                // exclude views has A and B columns
                if i < num_rows - 1 {
                    self.constraints.borrow_mut().gridwidth = 2;
                }
                self.add_header(&self.hc_skip.borrow()[i].clone());
                self.constraints.borrow_mut().gridwidth = 1;
                if i == num_rows - 1 {
                    self.add_header(&self.hc_skip_b);
                }
                self.add_header(&self.hc_boundary_model.borrow()[i].clone());
                self.add_header(&self.hc_surfaces_to_analyze.borrow()[i].clone());
                if i < num_rows - 1 {
                    self.constraints.borrow_mut().gridwidth = GRID_BAG_REMAINDER;
                }
                self.add_header(&self.hc3dmod.borrow()[i].clone());
                if i == num_rows - 1 {
                    self.constraints.borrow_mut().gridwidth = GRID_BAG_REMAINDER;
                    self.add_header(&self.hc3dmod_b);
                }
            }
        } else if cur_tab == Some(BatchRunTomoTab::Dataset) {
            let num_rows = NUM_DATASET_HEADER_ROWS;
            for i in 0..num_rows {
                self.add_standard_headers(i);
                {
                    let mut constraints = self.constraints.borrow_mut();
                    constraints.gridwidth = GRID_BAG_REMAINDER;
                    constraints.weightx = 100.0;
                }
                self.add_header(&self.hc_edit_dataset.borrow()[i].clone());
                self.constraints.borrow_mut().weightx = 0.0;
            }
        } else if cur_tab == Some(BatchRunTomoTab::Run) {
            let num_rows = NUM_RUN_HEADER_ROWS;
            for i in 0..num_rows {
                self.add_standard_headers(i);
                {
                    let mut constraints = self.constraints.borrow_mut();
                    constraints.gridwidth = 1;
                    // Swing layout: constraints.ipadx = 1.
                }
                self.add_header(&self.hc_status.borrow()[i].clone());
                self.add_header(&self.hc_step.borrow()[i].clone());
                self.add_header(&self.hc_cur_axis.borrow()[i].clone());
                // Swing layout: constraints.ipadx = 0.
                if i != 1 {
                    self.add_header(&self.hc_run);
                } else {
                    self.add_cell(
                        &*self.cbc_run_toggle,
                        super::input_cell::InputCellVirtual::get_component(&*self.cbc_run_toggle),
                    );
                }
                self.add_header(&self.hc_dataset.borrow()[i].clone());
                self.add_header(&self.hc_rec.borrow()[i].clone());
                self.add_header(&self.hc_proj_log.borrow()[i].clone());
                self.constraints.borrow_mut().gridwidth = GRID_BAG_REMAINDER;
                self.add_header(&self.hc_log.borrow()[i].clone());
            }
        }
        // rows
        self.row_list.remove_all();
        self.row_list.display(&self.viewport);
        // buttons
        self.pnl_stack_buttons
            .set_visible(cur_tab == Some(BatchRunTomoTab::Stacks));
    }

    /// Java private `addStandardHeaders(int)`.
    fn add_standard_headers(&self, index: usize) {
        // the stacks tab rows are highlightable
        if self.cur_tab.get() == Some(BatchRunTomoTab::Stacks) {
            self.constraints.borrow_mut().gridwidth = 2;
        } else {
            self.constraints.borrow_mut().gridwidth = 1;
        }
        self.add_header(&self.hc_number.borrow()[index].clone());
        // The stack header has a button on the bottom row
        if index == 0 {
            self.constraints.borrow_mut().gridwidth = 1;
        } else {
            self.constraints.borrow_mut().gridwidth = 2;
        }
        self.constraints.borrow_mut().weightx = 10.0;
        let mut _width = 0;
        self.add_header(&self.hc_stack.borrow()[index].clone());
        if self.cur_tab.get() == Some(BatchRunTomoTab::Dataset) {
            // Save the maximum width of the stack column - this is the minimum width of
            // the stack in each row.
            _width = self.hc_stack.borrow()[index].get_preferred_width();
        }
        self.constraints.borrow_mut().weightx = 0.0;
        if index == 0 {
            let mut constraints = *self.constraints.borrow();
            // `btnStack.add(pnlTable, layout, constraints)`: weightx 0 for the add.
            let old_weightx = constraints.weightx;
            constraints.weightx = 0.0;
            self.btn_stack.add(&self.pnl_table);
            self.layout.set_constraints(
                &SwingComponent::get_component(&*self.btn_stack),
                &constraints,
            );
            constraints.weightx = old_weightx;
            *self.constraints.borrow_mut() = constraints;
        }
    }

    /// Java private `addListeners()`.
    fn add_listeners(&self) {
        let listener = self.action_listener();
        self.btn_add.add_action_listener(listener.clone());
        self.btn_copy_down.add_action_listener(listener.clone());
        self.btn_delete.add_action_listener(listener);
    }

    /// This table as the `ActionListener` Java registers as `this` (the dialog also adds
    /// it to its series watcher checkbox).
    pub fn action_listener(&self) -> ActionListener {
        let this = self.this.clone();
        Rc::new(move |event: &ActionEvent| {
            if let Some(table) = this.upgrade() {
                table.action_performed(Some(event));
            }
        })
    }

    /// Java package-private `validate(BatchRunTomoDatasetDialog)`.
    pub fn validate(&self, global_dataset_dialog: Option<&Rc<BatchRunTomoDatasetDialog>>) -> bool {
        self.row_list.validate(global_dataset_dialog)
    }

    /// Java package-private `msgStatusChangerStarted(StatusChanger)`.
    pub fn msg_status_changer_started(&self, changer: &Rc<dyn StatusChanger>) {
        if let Some(this) = self.this.upgrade() {
            changer.add_status_change_listener(Some(this as Rc<dyn StatusChangeListener>));
        }
        self.row_list.msg_status_changer_started(changer);
    }

    /// Java package-private `addStatusChangeListenerToRowList(StatusChangeListener)`.
    pub fn add_status_change_listener_to_row_list(
        &self,
        listener: Option<Rc<dyn StatusChangeListener>>,
    ) {
        self.row_list.add_status_change_listener(listener);
    }

    /// Java package-private `addStatusChangeListenerToRows(StatusChangeListener)`.
    pub fn add_status_change_listener_to_rows(
        &self,
        listener: Option<Rc<dyn StatusChangeListener>>,
    ) {
        self.row_list.add_status_change_listener_to_rows(listener);
    }

    /// Java package-private `setTableListener(TableListener)`.  Currently only have
    /// room for one table listener.
    pub fn set_table_listener(&self, table_listener: Option<Rc<dyn TableListener>>) {
        self.row_list.set_table_listener(table_listener);
    }

    /// Java package-private `msgMontage(boolean)`.
    pub fn msg_montage(&self, montage: bool) {
        if montage {
            if !self.montage_frame.get() {
                self.montage_frame.set(true);
                self.send_status_changed(true, Some(StatusRef::FrameStatus(FrameStatus::Montage)));
            }
            // See if any single frame settings are left
            self.set_single_frame();
        } else {
            if !self.single_frame.get() {
                self.single_frame.set(true);
                self.send_status_changed(true, Some(StatusRef::FrameStatus(FrameStatus::Single)));
            }
            self.set_montage_frame();
        }
    }

    /// Java private `sendStatusChanged(boolean, Status)`.
    fn send_status_changed(&self, bool_: bool, status: Option<StatusRef>) {
        let event = StatusChangeBooleanEvent::new(bool_, status);
        let listeners = self.listeners.borrow().clone();
        if let Some(listeners) = listeners {
            for listener in &listeners {
                listener.status_changed_event(Some(&event));
            }
        }
    }

    /// Java private `setSingleFrame()`.
    fn set_single_frame(&self) {
        let orig_single_frame = self.single_frame.get();
        if self.row_list.is_empty() {
            // Default to enabled
            self.single_frame.set(true);
        } else {
            self.single_frame.set(self.row_list.is_single_frame());
        }
        if orig_single_frame != self.single_frame.get() {
            self.send_status_changed(
                self.single_frame.get(),
                Some(StatusRef::FrameStatus(FrameStatus::Single)),
            );
        }
    }

    /// Java private `setMontageFrame()`.
    fn set_montage_frame(&self) {
        let orig_montage_frame = self.montage_frame.get();
        if self.row_list.is_empty() {
            // Default to enabled
            self.montage_frame.set(true);
        } else {
            self.montage_frame.set(self.row_list.is_montage_frame());
        }
        if orig_montage_frame != self.montage_frame.get() {
            self.send_status_changed(
                self.montage_frame.get(),
                Some(StatusRef::FrameStatus(FrameStatus::Montage)),
            );
        }
    }

    /// Java package-private `setFrame(boolean)`.
    pub fn set_frame(&self, force: bool) {
        self.row_list.set_frame(force);
    }

    /// Java package-private `setParameters(BatchRunTomoMetaData, String, boolean,
    /// boolean)`.
    pub fn set_parameters_meta_data(
        &self,
        meta_data: &BatchRunTomoMetaData,
        only_stack_id_dataset_dialog: Option<&str>,
        _only_advanced_dataset_dialog: bool,
        init: bool,
    ) {
        // onlyAdvancedDatasetDialog refers to the global dataset dialog if the stackID is
        // null.
        self.row_list
            .set_parameters_meta_data(meta_data, only_stack_id_dataset_dialog, init);
        if only_stack_id_dataset_dialog.is_none() {
            self.status_changed_status(meta_data.get_status().map(StatusRef::BatchRunTomoStatus));
        }
    }

    /// Java package-private `setParameters(SeriesWatcherMetaData, String, boolean)`.
    pub fn set_parameters_series_watcher_meta_data(
        &self,
        meta_data: &SeriesWatcherMetaData,
        stack_id: Option<&str>,
        init: bool,
    ) {
        self.row_list
            .set_parameters_series_watcher_meta_data(meta_data, stack_id, init);
    }

    /// Java package-private `getParameters(BatchRunTomoMetaData)`.
    pub fn get_parameters_meta_data(&self, meta_data: &BatchRunTomoMetaData) {
        self.row_list.get_parameters_meta_data(meta_data);
        meta_data.set_earliest_run_ending_step(self.get_earliest_run_ending_step());
    }

    /// Java package-private `getParameters(BatchruntomoParam, boolean, boolean,
    /// RunType, StringBuilder, boolean, boolean)`.
    #[allow(clippy::too_many_arguments)]
    pub fn get_parameters_param(
        &self,
        param: &mut BatchruntomoParam,
        deliver_off: bool,
        deliver_to_directory: bool,
        run_type: Option<RunType>,
        err_msg: &mut String,
        do_validation: bool,
        validate_only: bool,
    ) -> bool {
        let mut manager_key_list: Option<Vec<UniqueKey>> = None;
        if do_validation {
            manager_key_list = Some(Vec::new());
        }
        let retval = self.row_list.get_parameters_param(
            param,
            deliver_off,
            deliver_to_directory,
            run_type,
            err_msg,
            do_validation,
            manager_key_list.as_mut(),
            validate_only,
        );
        if retval
            && let Some(manager_key_list) = &manager_key_list
            && !manager_key_list.is_empty()
        {
            let batch_close_datasets = etomo_director::INSTANCE
                .with_user_configuration(|user_config| user_config.is_batch_close_datasets());
            if !batch_close_datasets {
                let popup = Popup::get_yes_no_instance(
                    Some(self as &dyn UIComponent),
                    Some("Close Datasets?"),
                    Some("Datasets must be closed before batchruntomo is run.  Close datasets?"),
                    Some("Always close"),
                );
                ui_harness::with(|harness| harness.open_popup(&popup));
                if !popup.is_yes() {
                    // Cannot run batchruntomo unless datasets are closed, so Checkbox in
                    // popup must be ignored when No is selected.
                    return false;
                }
                if popup.is_checkbox_selected() {
                    etomo_director::INSTANCE.with_user_configuration_mut(|user_config| {
                        user_config.set_batch_close_datasets(true)
                    });
                }
            }
            etomo_director::INSTANCE.close_managers(Some(manager_key_list));
        }
        retval
    }

    /// Java package-private `saveAutodocs(TemplatePanel, NameValuePairList,
    /// NameValuePairList, boolean, boolean, File, String, boolean)`.
    #[allow(clippy::too_many_arguments)]
    pub fn save_autodocs(
        &self,
        template_panel: Option<&TemplatePanel>,
        global_batch_list: Option<&NameValuePairList>,
        templates: Option<&NameValuePairList>,
        do_validation: bool,
        init: bool,
        deliver_to_directory: Option<&std::path::Path>,
        autodoc_stack_id: Option<&str>,
        validate_only: bool,
    ) -> bool {
        self.row_list.save_autodocs(
            template_panel,
            global_batch_list,
            templates,
            do_validation,
            init,
            deliver_to_directory,
            autodoc_stack_id,
            validate_only,
        )
    }

    /// Java package-private `loadAutodocs(String, boolean)`.
    pub fn load_autodocs(
        &self,
        only_stack_id_dataset_dialog: Option<&str>,
        mut only_advanced_dataset_dialog: bool,
    ) {
        // onlyAdvancedDatasetDialog refers to the global dataset dialog if the stackID is
        // null.
        if only_stack_id_dataset_dialog.is_none() {
            only_advanced_dataset_dialog = false;
        }
        self.row_list
            .load_autodocs(only_stack_id_dataset_dialog, only_advanced_dataset_dialog);
    }

    /// Java package-private `getFirstRow()`.
    pub fn get_first_row(&self) -> Option<Rc<BatchRunTomoRow>> {
        self.row_list.get_first_row()
    }

    /// Java package-private `createRunList(RunType)`.
    pub fn create_run_list(&self, run_type: Option<RunType>) -> RunList {
        self.row_list.create_run_list(run_type)
    }

    /// Java public `findRow(String, String)`.
    pub fn find_row(&self, location: Option<&str>, root_name: Option<&str>) -> Option<String> {
        self.row_list.find_row(location, root_name)
    }

    /// Java public `getStack(String)`.
    pub fn get_stack(&self, stack_id: Option<&str>) -> Option<PathBuf> {
        self.row_list.get_stack(stack_id)
    }

    /// Java package-private `backupIfChanged(String, boolean)`.  Check each field to see
    /// if it has been changed from its checkpoint.  If it has changed, then back up its
    /// current value.  Returns true if any field has been changed from its checkpoint.
    pub fn backup_if_changed(
        &self,
        only_stack_id_dataset_dialog: Option<&str>,
        mut only_advanced_dataset_dialog: bool,
    ) -> bool {
        // onlyAdvancedDatasetDialog refers to the global dataset dialog if the stackID is
        // null.
        if only_stack_id_dataset_dialog.is_none() {
            only_advanced_dataset_dialog = false;
        }
        self.row_list
            .backup_if_changed(only_stack_id_dataset_dialog, only_advanced_dataset_dialog)
    }

    /// Java package-private `applyValues(boolean, boolean, DirectiveFileCollection,
    /// String, boolean)`.
    pub fn apply_values(
        &self,
        init: bool,
        retain_user_values: bool,
        directive_file_collection: &DirectiveFileCollection,
        only_stack_id_dataset_dialog: Option<&str>,
        mut only_advanced_dataset_dialog: bool,
    ) {
        // onlyAdvancedDatasetDialog refers to the global dataset dialog if the stackID is
        // null.
        if only_stack_id_dataset_dialog.is_none() {
            only_advanced_dataset_dialog = false;
        }
        self.row_list.apply_values(
            init,
            retain_user_values,
            directive_file_collection,
            only_stack_id_dataset_dialog,
            only_advanced_dataset_dialog,
        );
    }

    /// Java package-private `isSeriesWatcherOn()`.
    pub fn is_series_watcher_on(&self) -> bool {
        self.series_watcher_parent
            .upgrade()
            .is_some_and(|parent| parent.is_series_watcher_on())
    }

    /// Java private `updateDisplay()`.
    fn update_display(&self) {
        let size = self.row_list.size();
        let enable = size > 0 && self.row_list.is_highlighted();
        self.btn_delete.set_enabled(enable);
        let index = self.row_list.get_highlighted_index();
        self.btn_copy_down
            .set_enabled(index != -1 && index < size - 1);
    }

    /// Java package-private `updateRowDisplay()`.
    pub fn update_row_display(&self) {
        self.row_list.update_display();
    }

    /// Java package-private `msgTabChanged(BatchRunTomoTab)`.
    pub fn msg_tab_changed(&self, tab: Option<BatchRunTomoTab>) {
        if (tab == Some(BatchRunTomoTab::Stacks)
            || tab == Some(BatchRunTomoTab::Dataset)
            || tab == Some(BatchRunTomoTab::Run))
            && tab != self.cur_tab.get()
        {
            self.cur_tab.set(tab);
            self.rebuild_table();
        }
    }

    /// Java `actionPerformed(ActionEvent)`.
    pub fn action_performed(&self, action_event: Option<&ActionEvent>) {
        let Some(action_event) = action_event else {
            return;
        };
        let Some(action_command) = action_event.get_action_command() else {
            return;
        };
        let action_command = Some(action_command.to_owned());
        if action_command == self.btn_add.get_action_command() {
            let dialog = self.dialog();
            let chooser = FileChooser::new_base_manager_axis_id_string_browsing_directory(
                Some(self.base_manager()),
                Some(AxisID::Only),
                None,
                dialog
                    .as_deref()
                    .map(|dialog| dialog as &dyn BrowsingDirectory),
            );
            chooser.set_dialog_title(Some("Select a stack for each dataset"));
            chooser.set_file_filter(Some(Rc::new(StackFileFilter::get_instance(
                Some(self.base_manager()),
                true,
            ))
                as Rc<dyn crate::imod::etomo::jdk::FileFilter>));
            chooser.set_multi_selection_enabled(true);
            // `chooser.setPreferredSize(UIParameters.getInstance().getFileChooserDimension())`:
            // sizes are not modelled by the Swing stand-in.
            let return_val =
                chooser.show_open_dialog(Some(&SwingComponent::get_component(&*self.btn_add)));
            if return_val == file_chooser::APPROVE_OPTION {
                let mut stack_list = chooser.get_selected_files();
                if stack_list.is_empty() {
                    // Workaround: If one file was chosen and the chooser was closed with
                    // FileChooser.approveSelection (happens when uitest is run), then
                    // getSelectedFiles will return a zero length array, and the selected
                    // file will be available from getSelectedFile. It seems like this is
                    // a Java bug.
                    if let Some(stack) = chooser.get_selected_file() {
                        stack_list = vec![stack];
                    }
                }
                if !stack_list.is_empty() {
                    if let Some(dialog) = &dialog {
                        dialog.set_browsing_dir(stack_list.last().map(PathBuf::as_path));
                    }
                    // Remove matching B stacks and set dual to true for the A stack
                    let stack_list: Vec<Option<PathBuf>> =
                        stack_list.into_iter().map(Some).collect();
                    let hc_stack0 = self.hc_stack.borrow()[0].clone();
                    let filtered_stack_list = dataset_tool::remove_matching_b_stacks(
                        Some(&*hc_stack0 as &dyn UIComponent),
                        Some(&stack_list),
                    );
                    self.row_list.add_new(filtered_stack_list);
                    self.set_frame(true);
                }
            }
        } else if action_command == self.btn_delete.get_action_command() {
            if self.row_list.remove_highlighted() {
                self.rebuild_table();
                self.update_display();
                ui_harness::with(|harness| harness.pack_base_manager(Some(self.base_manager())));
            }
        } else if action_command == self.btn_copy_down.get_action_command() {
            self.row_list.copy_down();
        } else if self.series_watcher_parent.upgrade().is_some_and(|parent| {
            parent.equals_series_watcher_action_command(action_command.as_deref().unwrap_or(""))
        }) {
            self.update_display();
            self.row_list.update_display();
        }
    }

    /// Java `size()`.
    pub fn size(&self) -> i32 {
        self.row_list.size()
    }

    /// Java private `setTooltips()`.
    fn set_tooltips(&self) {
        let imod_tooltip = "Opens the stacks";
        for i in 0..NUM_STACKS_HEADER_ROWS {
            self.hc_stack.borrow()[i].set_tool_tip_text(Some(shared_strings::STACK_TOOLTIP));
            self.hc_dual.borrow()[i].set_tool_tip_text(Some(shared_strings::DUAL_TOOLTIP));
            self.hc_montage.borrow()[i].set_tool_tip_text(Some(shared_strings::MONTAGE_TOOLTIP));
            self.hc_boundary_model.borrow()[i]
                .set_tool_tip_text(Some(shared_strings::RAW_BOUNDARY_MODEL));
            self.hc_surfaces_to_analyze.borrow()[i]
                .add_tooltip(Some(shared_strings::SURFACES_TO_ANALYZE_2_TOOLTIP));
            if i < NUM_STACKS_HEADER_ROWS - 1 {
                self.hc_skip.borrow()[i].set_tool_tip_text(Some("Views to exclude"));
                self.hc3dmod.borrow()[i].set_tool_tip_text(Some(imod_tooltip));
            } else {
                self.hc_skip.borrow()[i].set_tool_tip_text(Some(shared_strings::SKIP_TOOLTIP));
            }
        }
        for i in 0..NUM_RUN_HEADER_ROWS {
            self.hc_step.borrow()[i].set_tool_tip_text(Some(shared_strings::STEP_TOOLTIP));
        }
        self.hc_skip_b
            .set_tool_tip_text(Some(shared_strings::BSKIP_TOOLTIP));
        self.hc3dmod.borrow()[NUM_STACKS_HEADER_ROWS - 1]
            .set_tool_tip_text(Some(shared_strings::IMOD_A_TOOLTIP));
        self.hc3dmod_b
            .set_tool_tip_text(Some(shared_strings::IMOD_B_TOOLTIP));
    }

    /// Java package-private `getTablePanel()`.
    pub fn get_table_panel(&self) -> Rc<JComponent> {
        self.pnl_table.clone()
    }

    /// Java package-private `getGridBagLayout()`.
    pub fn get_grid_bag_layout(&self) -> &GridBagLayout {
        &self.layout
    }

    /// Java package-private `getManager()`.
    pub fn get_manager(&self) -> &'static BatchRunTomoManager {
        self.manager
    }

    /// Java package-private `getGridBagConstraints()`: a copy of the shared
    /// constraints.
    pub fn get_grid_bag_constraints(&self) -> GridBagConstraints {
        *self.constraints.borrow()
    }

    /// The shared `constraints` object the rows change.
    pub fn with_constraints<R>(&self, f: impl FnOnce(&mut GridBagConstraints) -> R) -> R {
        let mut constraints = *self.constraints.borrow();
        let result = f(&mut constraints);
        *self.constraints.borrow_mut() = constraints;
        result
    }

    /// Java package-private `getTableReference()`.
    pub fn get_table_reference(&self) -> Arc<TableReference> {
        self.row_list.get_table_reference()
    }

    /// Java package-private `getTemplateValues()`.
    pub fn get_template_values(&self) -> Rc<RefCell<TemplateValues>> {
        self.dialog()
            .expect("the dialog owns its table")
            .get_template_values()
    }

    /// Java package-private `getBrowsingDirectory()`.
    pub fn get_browsing_directory(&self) -> Weak<dyn BrowsingDirectory> {
        self.dialog.clone() as Weak<dyn BrowsingDirectory>
    }

    /// Java `getComponent()` (SwingComponent).
    pub fn get_component(&self) -> Rc<JComponent> {
        self.pnl_root.clone()
    }
}

impl Highlightable for BatchRunTomoTable {
    /// Java `highlight(boolean)`.
    fn highlight(&self, _highlight: bool) {
        self.update_display();
    }
}

impl HighlightableTableVirtual for BatchRunTomoTable {
    fn highlightable_table(&self) -> &HighlightableTable {
        &self.base
    }

    /// Java `highlightUpActionPerformed()`.
    fn highlight_up_action_performed(&self) {
        // If the highlight is not visible, don't change the viewport
        let adjust_viewport = self
            .viewport
            .in_viewport(self.row_list.get_highlighted_index());
        let index = self.row_list.highlight_up();
        if index != -1 && adjust_viewport && !self.viewport.in_viewport(index) {
            // The highlight is visible - keep it in the viewport
            if index == self.row_list.size() - 1 {
                self.viewport.end_button_action();
            } else {
                self.viewport.up_button_action();
            }
        }
    }

    /// Java `highlightDownActionPerformed()`.
    fn highlight_down_action_performed(&self) {
        // If the highlight is not visible, don't change the viewport
        let adjust_viewport = self
            .viewport
            .in_viewport(self.row_list.get_highlighted_index());
        let index = self.row_list.highlight_down();
        if index != -1 && adjust_viewport && !self.viewport.in_viewport(index) {
            // The highlight is visible - keep it in the viewport
            if index == 0 {
                self.viewport.home_button_action();
            } else {
                self.viewport.down_button_action();
            }
        }
    }

    /// Java `getFocusableParents()`.
    fn get_focusable_parents(&self) -> Vec<Option<Rc<JComponent>>> {
        self.focusable_parents.iter().cloned().map(Some).collect()
    }
}

impl Viewable for BatchRunTomoTable {
    /// Java `msgViewportPaged()`.
    fn msg_viewport_paged(&self) {
        self.row_list.remove_all();
        self.row_list.display(&self.viewport);
        ui_harness::with(|harness| harness.pack_base_manager(Some(self.base_manager())));
    }

    /// Java `size()`.
    fn size(&self) -> i32 {
        BatchRunTomoTable::size(self)
    }

    /// Java `getFocusableParents()`.
    fn get_focusable_parents(&self) -> Vec<Rc<JComponent>> {
        self.focusable_parents.clone()
    }
}

impl Expandable for BatchRunTomoTable {
    /// Java `expand(ExpandButton)`.
    fn expand_expand_button(&self, button: &Rc<ExpandButton>) {
        if Rc::ptr_eq(button, &self.btn_stack) {
            self.row_list.expand_stack(self.btn_stack.is_expanded());
        }
        ui_harness::with(|harness| harness.pack_base_manager(Some(self.base_manager())));
    }

    /// Java `expand(GlobalExpandButton)`: empty.
    fn expand_global_expand_button(&self, _button: &Rc<GlobalExpandButton>) {}
}

impl StatusChangeListener for BatchRunTomoTable {
    /// Java `statusChanged(Status)`.
    fn status_changed_status(&self, new_status: Option<StatusRef>) {
        let Some(StatusRef::BatchRunTomoStatus(new_status)) = new_status else {
            return;
        };
        // Avoid overriding an end state with an error state.
        let status =
            BatchRunTomoStatus::get_instance_from_statuses(self.status.get(), Some(new_status));
        self.status.set(status);
        let open = status == Some(BatchRunTomoStatus::Open);
        self.btn_add.set_editable(open);
        self.btn_copy_down.set_editable(open);
        self.btn_delete.set_editable(open);
    }

    /// Java `statusChanged(StatusChangeEvent)`: does not respond to dataset-level or
    /// row-level status changes.
    fn status_changed_event(&self, _status_change_event: Option<&dyn StatusChangeEvent>) {}

    /// Java `startOver()`.
    fn start_over(&self) {
        self.status_changed_status(Some(StatusRef::BatchRunTomoStatus(
            BatchRunTomoStatus::Open,
        )));
    }
}

impl StatusChanger for BatchRunTomoTable {
    /// Java `addStatusChangeListener(StatusChangeListener)`.
    fn add_status_change_listener(&self, listener: Option<Rc<dyn StatusChangeListener>>) {
        let Some(listener) = listener else {
            return;
        };
        let mut new_collection = false;
        let mut listeners = self.listeners.borrow_mut();
        if listeners.is_none() {
            *listeners = Some(Vec::new());
            new_collection = true;
        }
        let list = listeners.as_mut().unwrap();
        if !new_collection && list.iter().any(|existing| Rc::ptr_eq(existing, &listener)) {
            return;
        }
        list.push(listener);
    }
}

impl SwingComponent for BatchRunTomoTable {
    /// Java `getComponent()`.
    fn get_component(&self) -> Rc<JComponent> {
        self.pnl_root.clone()
    }
}

impl UIComponent for BatchRunTomoTable {
    /// Java `getUIComponent()`.
    fn get_ui_component(&self) -> &dyn SwingComponent {
        self
    }

    fn get_component(&self) -> Rc<JComponent> {
        self.pnl_root.clone()
    }
}

impl FieldDisplayer for BatchRunTomoTable {
    /// Java `display()`.
    fn display_void(&self) {
        BatchRunTomoTable::display_void(self);
    }

    /// Java `display(UIComponent)`.
    fn display_ui_component(&self, _ui_component: Option<&dyn UIComponent>) {
        BatchRunTomoTable::display_void(self);
    }
}

/// Java private inner class `RowList implements StatusChangeListener, StatusChanger`.
pub struct RowList {
    /// Java private final `list`.
    list: RefCell<Vec<Rc<BatchRunTomoRow>>>,
    /// Java private final `tableReference`.
    table_reference: Arc<TableReference>,
    /// Java private final `table` (also the enclosing instance).
    table: Weak<BatchRunTomoTable>,
    /// Java private `initialValueRow`, initially null.
    initial_value_row: RefCell<Option<Rc<BatchRunTomoRow>>>,
    /// Java private `tableListener`, initially null.  Currently only one table listener
    /// is required.
    table_listener: RefCell<Option<Rc<dyn TableListener>>>,
    /// Java private `eventObject`, initially null: `new EventObject(table)`, whose
    /// source the listeners never read.
    event_object: Cell<bool>,
    /// Java private `listeners`, initially null.
    listeners: RefCell<Option<Vec<Rc<dyn StatusChangeListener>>>>,
    /// Java private `rowListeners`, initially null.
    row_listeners: RefCell<Option<Vec<Rc<dyn StatusChangeListener>>>>,
    /// Java private `rowChangers`, initially null.
    row_changers: RefCell<Option<Vec<Rc<dyn StatusChanger>>>>,
}

/// The `RowList` as the `StatusChangeListener` the rows and monitors are given (Java
/// passes the inner-class instance `this`).
struct RowListListener(Weak<BatchRunTomoTable>);

impl StatusChangeListener for RowListListener {
    fn status_changed_status(&self, status: Option<StatusRef>) {
        if let Some(table) = self.0.upgrade() {
            table.row_list.status_changed_status(status);
        }
    }
    fn status_changed_event(&self, _event: Option<&dyn StatusChangeEvent>) {}
    fn start_over(&self) {}
}

impl RowList {
    /// Java private `RowList(BatchRunTomoTable, TableReference)`.
    fn new(table: Weak<BatchRunTomoTable>, table_reference: Arc<TableReference>) -> RowList {
        RowList {
            list: RefCell::new(Vec::new()),
            table_reference,
            table,
            initial_value_row: RefCell::new(None),
            table_listener: RefCell::new(None),
            event_object: Cell::new(false),
            listeners: RefCell::new(None),
            row_listeners: RefCell::new(None),
            row_changers: RefCell::new(None),
        }
    }

    fn table(&self) -> Rc<BatchRunTomoTable> {
        self.table
            .upgrade()
            .expect("the row list belongs to its table")
    }

    /// The row list as a listener (Java `this`).
    fn as_listener(&self) -> Rc<dyn StatusChangeListener> {
        Rc::new(RowListListener(self.table.clone()))
    }

    /// Java private `setTableListener(TableListener)`.
    fn set_table_listener(&self, table_listener: Option<Rc<dyn TableListener>>) {
        *self.table_listener.borrow_mut() = table_listener;
        self.event_object.set(true);
    }

    /// Java private `msgStatusChangerStarted(StatusChanger)`.
    fn msg_status_changer_started(&self, changer: &Rc<dyn StatusChanger>) {
        changer.add_status_change_listener(Some(self.as_listener()));
        let list = self.list.borrow().clone();
        for row in &list {
            changer.add_status_change_listener(Some(row.clone() as Rc<dyn StatusChangeListener>));
        }
        if self.row_changers.borrow().is_none() {
            *self.row_changers.borrow_mut() = Some(Vec::new());
        }
        self.row_changers
            .borrow_mut()
            .as_mut()
            .unwrap()
            .push(changer.clone());
    }

    /// Java `addStatusChangeListener(StatusChangeListener)`.  Adds listeners for
    /// earliestRunStep.
    fn add_status_change_listener(&self, listener: Option<Rc<dyn StatusChangeListener>>) {
        let Some(listener) = listener else {
            return;
        };
        self.listeners
            .borrow_mut()
            .get_or_insert_with(Vec::new)
            .push(listener);
    }

    /// Java private `addStatusChangeListenerToRows(StatusChangeListener)`.
    fn add_status_change_listener_to_rows(&self, listener: Option<Rc<dyn StatusChangeListener>>) {
        let Some(listener) = listener else {
            return;
        };
        self.row_listeners
            .borrow_mut()
            .get_or_insert_with(Vec::new)
            .push(listener.clone());
        let list = self.list.borrow().clone();
        for row in &list {
            row.add_status_change_listener(Some(listener.clone()));
        }
    }

    /// Java package-private `getTableReference()`.
    fn get_table_reference(&self) -> Arc<TableReference> {
        self.table_reference.clone()
    }

    /// Adds the row-list listeners to a new row (`row.addStatusChangeListener(...)`).
    fn add_row_listeners(&self, row: &Rc<BatchRunTomoRow>) {
        let row_listeners = self.row_listeners.borrow().clone();
        if let Some(row_listeners) = row_listeners {
            for listener in row_listeners {
                row.add_status_change_listener(Some(listener));
            }
        }
    }

    /// Adds a new row to the row changers (`rowChangers.get(j).addStatusChangeListener(row)`).
    fn add_to_row_changers(&self, row: &Rc<BatchRunTomoRow>) {
        let row_changers = self.row_changers.borrow().clone();
        if let Some(row_changers) = row_changers {
            for changer in row_changers {
                changer
                    .add_status_change_listener(Some(row.clone() as Rc<dyn StatusChangeListener>));
            }
        }
    }

    /// `tableListener.firstRowAdded(eventObject)`.
    fn first_row_added(&self) {
        let table_listener = self.table_listener.borrow().clone();
        if let Some(table_listener) = table_listener {
            table_listener.first_row_added(Some(&()));
        }
    }

    /// Java private `addNew(List<DatasetTool.StackInfo>)`.
    fn add_new(&self, stack_info_list: Option<Vec<Rc<RefCell<dataset_tool::StackInfo>>>>) {
        let Some(stack_info_list) = stack_info_list else {
            return;
        };
        let table = self.table();
        let first_index = self.list.borrow().len() as i32;
        let mut not_added: Vec<String> = Vec::new();
        let mut file_added = false;
        for (i, stack_info) in stack_info_list.iter().enumerate() {
            let stack = stack_info.borrow().get_stack();
            let Some(stack) = stack else {
                continue;
            };
            // See if there is an ID for this stack.
            let abs_path = crate::imod::etomo::util::utilities::java_io_file_get_absolute_path(
                &stack.to_string_lossy(),
            );
            // Get the stackID from absPath minus the extension
            let mut stack_id = self.table_reference.get_id(Some(&abs_path));
            if let Some(id) = &stack_id {
                // Check for duplicate files
                if self.row_exists(Some(id)) {
                    not_added.push(abs_path);
                    continue;
                }
            } else {
                match self.table_reference.put(&abs_path) {
                    Ok(id) => stack_id = Some(id),
                    Err(PutError::Duplicate(e)) => {
                        eprintln!("{e:?}");
                        continue;
                    }
                    Err(PutError::NotLoaded(e)) => {
                        eprintln!("{e:?}");
                        if !file_added {
                            return;
                        } else {
                            continue;
                        }
                    }
                }
            }
            // Put settings from the previous row.
            let index = self.list.borrow().len();
            let prev_row = if index > 0 {
                Some(self.list.borrow()[index - 1].clone())
            } else {
                self.initial_value_row.borrow().clone()
            };
            // Decide how to set the "dual" checkbox.
            let dual = stack_info.borrow_mut().is_matched();
            let single_axis = stack_info.borrow_mut().is_single_axis();
            let _override_prev_row = dual || single_axis;
            let mut axis_type = None;
            if single_axis {
                axis_type = Some(AxisType::SingleAxis);
            } else if stack_info.borrow_mut().is_matched() {
                axis_type = Some(AxisType::DualAxis);
            }
            // Add the row.
            let row = BatchRunTomoRow::get_instance(
                table.manager,
                table.dialog.clone(),
                self.table.clone(),
                Some(table.basic_directives.clone()),
                index as i32 + 1,
                Some(&stack),
                prev_row.as_ref(),
                axis_type,
                stack_id.as_deref(),
                Some(&table.preferred_table_size),
                true,
                Some(table.cbc_run_toggle.clone()),
            );
            // OrigStack is essential for limiting batchruntomo to one delivery. Should
            // only be set when the user chooses the file.
            row.set_orig_stack(Some(&stack));
            row.add_status_change_listener(Some(self.as_listener()));
            self.add_to_row_changers(&row);
            self.add_row_listeners(&row);
            row.expand_stack(table.btn_stack.is_expanded());
            self.list.borrow_mut().push(row.clone());
            file_added = true;
            row.display(&table.viewport, table.cur_tab.get());
            if self.table_listener.borrow().is_some() && i == 0 && first_index == 0 {
                self.first_row_added();
            }
        }
        if file_added {
            table.viewport.adjust_viewport(first_index);
            self.remove_all();
            self.display(&table.viewport);
            ui_harness::with(|harness| harness.pack_base_manager(Some(table.base_manager())));
            table.update_display();
        }
        // Pop up a warning if there where any duplicate files.
        if !not_added.is_empty() {
            let mut warning = String::new();
            warning.push_str("The stack table already contains file(s) that match ");
            let mut iterator = not_added.iter().peekable();
            if let Some(first) = iterator.next() {
                warning.push_str(first);
            }
            while let Some(abs_path) = iterator.next() {
                warning.push_str(", ");
                warning.push_str(if iterator.peek().is_none() {
                    " and "
                } else {
                    ""
                });
                warning.push_str(abs_path);
            }
            warning.push('.');
            ui_harness::with(|harness| {
                harness.open_message_dialog_base_manager_string_string(
                    Some(table.base_manager()),
                    &warning,
                    "Unable to Add File(s)",
                )
            });
        }
    }

    /// Java private `load(BatchRunTomoMetaData)`.
    fn load(&self, meta_data: &BatchRunTomoMetaData) {
        let table = self.table();
        let first_index = self.list.borrow().len() as i32;
        let array = meta_data.get_ordered_rows();
        let mut file_added = false;
        if let Some(array) = array {
            for (i, row_meta_data) in array.iter().enumerate() {
                let Some(row_meta_data) = row_meta_data else {
                    continue;
                };
                let index = self.list.borrow().len();
                let stack_id = row_meta_data.get_stack_id().to_owned();
                let file_path = self.table_reference.get_file_path(Some(&stack_id));
                let initial_value_row = self.initial_value_row.borrow().clone();
                let row = BatchRunTomoRow::get_instance(
                    table.manager,
                    table.dialog.clone(),
                    self.table.clone(),
                    Some(table.basic_directives.clone()),
                    index as i32 + 1,
                    // Java `new File(null)` throws for a stack ID without a path.
                    Some(&PathBuf::from(file_path.unwrap_or_default())),
                    initial_value_row.as_ref(),
                    None,
                    Some(&stack_id),
                    Some(&table.preferred_table_size),
                    false,
                    Some(table.cbc_run_toggle.clone()),
                );
                row.add_status_change_listener(Some(self.as_listener()));
                row.expand_stack(table.btn_stack.is_expanded());
                self.add_row_listeners(&row);
                self.list.borrow_mut().push(row.clone());
                file_added = true;
                row.display(&table.viewport, table.cur_tab.get());
                if self.table_listener.borrow().is_some() && i == 0 && first_index == 0 {
                    self.first_row_added();
                }
            }
        }
        if file_added {
            table.viewport.adjust_viewport(first_index);
            self.remove_all();
            self.display(&table.viewport);
            ui_harness::with(|harness| harness.pack_base_manager(Some(table.base_manager())));
            table.update_display();
        }
    }

    /// Java private `getFirstRow()`.
    fn get_first_row(&self) -> Option<Rc<BatchRunTomoRow>> {
        self.list.borrow().first().cloned()
    }

    /// Java private `validate(BatchRunTomoDatasetDialog)`.
    fn validate(&self, global_dataset_dialog: Option<&Rc<BatchRunTomoDatasetDialog>>) -> bool {
        let list = self.list.borrow().clone();
        for row in &list {
            if !row.validate(global_dataset_dialog) {
                return false;
            }
        }
        true
    }

    /// Java public `findRow(String, String)`.  Returns stackID if the parameter values
    /// can be found in a row.  Otherwise returns null.  location is first matched
    /// against original location.
    fn find_row(&self, location: Option<&str>, root_name: Option<&str>) -> Option<String> {
        if location.is_none() || root_name.is_none() {
            return None;
        }
        let list = self.list.borrow().clone();
        for row in &list {
            if row.equals_location_root_name(location, root_name) {
                return row.get_stack_id();
            }
        }
        None
    }

    /// Java public `getStack(String)`.
    fn get_stack(&self, stack_id: Option<&str>) -> Option<PathBuf> {
        stack_id?;
        let list = self.list.borrow().clone();
        for row in &list {
            if row.equals_string(stack_id) {
                return row.get_stack();
            }
        }
        None
    }

    /// Java private `createRunList(RunType)`.
    fn create_run_list(&self, run_type: Option<RunType>) -> RunList {
        let run_list = RunList::new();
        let list = self.list.borrow().clone();
        for row in &list {
            if row.setup_run(run_type, false) {
                run_list.add(
                    row.get_stack_id().as_deref(),
                    row.get_run_status(),
                    row.is_dual(),
                );
            }
        }
        run_list
    }

    /// Java private `getRow(String)`.
    fn get_row(&self, stack_id: Option<&str>) -> Option<Rc<BatchRunTomoRow>> {
        let list = self.list.borrow().clone();
        list.into_iter().find(|row| row.equals_stack_id(stack_id))
    }

    /// Java private `rowExists(String)`.
    fn row_exists(&self, stack_id: Option<&str>) -> bool {
        self.list
            .borrow()
            .iter()
            .any(|row| row.equals_stack_id(stack_id))
    }

    /// Java private `removeHighlighted()`.  Returns true if a row was deleted.
    fn remove_highlighted(&self) -> bool {
        let mut index = self.get_highlighted_index();
        if index != -1 {
            let table = self.table();
            let row = self.list.borrow()[index as usize].clone();
            if ui_harness::with(|harness| {
                harness.open_yes_no_dialog_base_manager_string_axis_id(
                    Some(table.base_manager()),
                    "Delete the highlighted row?",
                    Some(AXIS_ID),
                )
            }) {
                let _montage = row.is_montage();
                row.remove();
                row.delete();
                self.list.borrow_mut().remove(index as usize);
                table.viewport.adjust_viewport(index);
                self.set_frame(false);
            }
            let size = self.list.borrow().len() as i32;
            for i in index..size {
                self.list.borrow()[i as usize].set_number(i + 1);
            }
            // Highlight the row after the deleted row, or the previous one if at the end
            // of the table.
            if index == size {
                index -= 1;
            }
            self.highlight(index);
            if self.table_listener.borrow().is_some() && self.list.borrow().is_empty() {
                let table_listener = self.table_listener.borrow().clone();
                if let Some(table_listener) = table_listener {
                    table_listener.last_row_deleted(Some(&()));
                }
            }
            return true;
        }
        false
    }

    /// Java private `highlight(int)`.
    fn highlight(&self, index: i32) {
        if index < 0 || index >= self.list.borrow().len() as i32 {
            return;
        }
        let row = self.list.borrow()[index as usize].clone();
        row.select_highlight_button();
    }

    /// Java private `highlightDown()`.
    fn highlight_down(&self) -> i32 {
        let mut index = self.get_highlighted_index();
        if index < 0 {
            return -1;
        }
        index += 1;
        if index >= self.size() {
            index = 0;
        }
        self.highlight(index);
        index
    }

    /// Java private `highlightUp()`.
    fn highlight_up(&self) -> i32 {
        let mut index = self.get_highlighted_index();
        if index < 0 {
            return -1;
        }
        index -= 1;
        let size = self.size();
        if index < 0 || index >= size {
            index = size - 1;
        }
        self.highlight(index);
        index
    }

    /// Java private `expandStack(boolean)`.
    fn expand_stack(&self, expanded: bool) {
        for row in self.list.borrow().iter() {
            row.expand_stack(expanded);
        }
    }

    /// Java private `removeAll()`.
    fn remove_all(&self) {
        for row in self.list.borrow().iter() {
            row.remove();
        }
    }

    /// Java private `display(Viewport)`.
    fn display(&self, viewport: &Viewport) {
        let cur_tab = self.table().cur_tab.get();
        let list = self.list.borrow().clone();
        for row in &list {
            row.display(viewport, cur_tab);
        }
    }

    /// Java private `size()`.
    fn size(&self) -> i32 {
        self.list.borrow().len() as i32
    }

    /// Java private `isEmpty()`.
    fn is_empty(&self) -> bool {
        self.list.borrow().is_empty()
    }

    /// Java private `isSingleFrame()`.
    fn is_single_frame(&self) -> bool {
        self.list.borrow().iter().any(|row| !row.is_montage())
    }

    /// Java private `isMontageFrame()`.
    fn is_montage_frame(&self) -> bool {
        self.list.borrow().iter().any(|row| row.is_montage())
    }

    /// Java package-private `hasDual()`.
    fn has_dual(&self) -> bool {
        self.list.borrow().iter().any(|row| row.is_dual())
    }

    /// Java private `setFrame(boolean)`.  Sets singleFrame and montageFrame and sends
    /// status change events.
    fn set_frame(&self, force: bool) {
        let table = self.table();
        let orig_single_frame = table.single_frame.get();
        let orig_montage_frame = table.montage_frame.get();
        table.single_frame.set(false);
        table.montage_frame.set(false);
        let mut count = 2;
        for row in self.list.borrow().iter() {
            if row.is_montage() {
                if !table.montage_frame.get() {
                    count -= 1;
                    table.montage_frame.set(true);
                }
            } else if !table.single_frame.get() {
                count -= 1;
                table.single_frame.set(true);
            }
            if count <= 0 {
                break;
            }
        }
        if !table.single_frame.get() && !table.montage_frame.get() {
            // There are no rows so enable single and montage.
            table.single_frame.set(true);
            table.montage_frame.set(true);
        }
        if force || orig_single_frame != table.single_frame.get() {
            self.send_status_changed(
                table.single_frame.get(),
                Some(StatusRef::FrameStatus(FrameStatus::Single)),
            );
        }
        if force || orig_montage_frame != table.montage_frame.get() {
            self.send_status_changed(
                table.montage_frame.get(),
                Some(StatusRef::FrameStatus(FrameStatus::Montage)),
            );
        }
    }

    /// Java package-private `updateDisplay()`.
    fn update_display(&self) {
        let list = self.list.borrow().clone();
        for row in &list {
            row.update_display(None, None);
        }
    }

    /// Java `statusChanged(Status)`: nothing to do.
    fn status_changed_status(&self, _status: Option<StatusRef>) {}

    /// Java private `sendStatusChanged(boolean, Status)`.
    fn send_status_changed(&self, bool_: bool, status: Option<StatusRef>) {
        let event = StatusChangeBooleanEvent::new(bool_, status);
        let listeners = self.listeners.borrow().clone();
        if let Some(listeners) = listeners {
            for listener in &listeners {
                listener.status_changed_event(Some(&event));
            }
        }
    }

    /// Java private `getEarliestRunEndingStep()`.  Gets the earliest ending step in the
    /// row list where the Run checkbox is checked.  Return null if any run enabled row
    /// hasn't gotten to an ending step.
    fn get_earliest_run_ending_step(&self) -> Option<EndingStep> {
        let mut earliest: Option<EndingStep> = None;
        for row in self.list.borrow().iter() {
            if row.is_run() {
                let ending_step = row.get_ending_step()?;
                // Look for the earliest ending step and set it to the earliest ending
                // step.
                match earliest {
                    None => earliest = Some(ending_step),
                    Some(current) => {
                        if !current.is_first() && ending_step.lt(Some(current)) {
                            earliest = Some(ending_step);
                        }
                    }
                }
            }
        }
        earliest
    }

    /// Java private `getImageFilenameStyle()`.  Gets the image file name style from the
    /// first dataset that has a valid one.
    fn get_image_filename_style(&self) -> Option<ImageFilenameStyle> {
        let table = self.table();
        let list = self.list.borrow().clone();
        for row in &list {
            let mut image_filename_style = None;
            let output = table.manager.tomosetexts(row.get_stack_path().as_deref());
            if let Some(output) = output {
                image_filename_style = output.get_image_filename_style();
            }
            if image_filename_style.is_some() {
                return image_filename_style;
            }
        }
        None
    }

    /// Java private `isHighlighted()`.
    fn is_highlighted(&self) -> bool {
        self.list.borrow().iter().any(|row| row.is_highlighted())
    }

    /// Java private `getHighlightedIndex()`.
    fn get_highlighted_index(&self) -> i32 {
        for (i, row) in self.list.borrow().iter().enumerate() {
            if row.is_highlighted() {
                return i as i32;
            }
        }
        -1
    }

    /// Java private `copyDown()`.
    fn copy_down(&self) {
        let index = self.get_highlighted_index();
        let list = self.list.borrow().clone();
        if index != -1 && index < list.len() as i32 - 1 {
            list[index as usize + 1].copy(Some(&list[index as usize]));
        }
    }

    /// Java private `backupIfChanged(String, boolean)`.  Returns true if any field has
    /// been changed from its checkpoint.
    fn backup_if_changed(
        &self,
        only_stack_id_dataset_dialog: Option<&str>,
        only_advanced_dataset_dialog: bool,
    ) -> bool {
        let only_dataset_dialog = only_stack_id_dataset_dialog.is_some();
        if !only_dataset_dialog {
            let mut changed = false;
            let list = self.list.borrow().clone();
            for row in &list {
                if row.backup_if_changed(only_dataset_dialog, only_advanced_dataset_dialog) {
                    changed = true;
                }
            }
            return changed;
        } else if let Some(row) = self.get_row(only_stack_id_dataset_dialog) {
            return row.backup_if_changed(only_dataset_dialog, only_advanced_dataset_dialog);
        }
        true
    }

    /// Java private `applyValues(boolean, boolean, DirectiveFileCollection, String,
    /// boolean)`.
    fn apply_values(
        &self,
        init: bool,
        retain_user_values: bool,
        directive_file_collection: &DirectiveFileCollection,
        only_stack_id_dataset_dialog: Option<&str>,
        only_advanced_dataset_dialog: bool,
    ) {
        let only_dataset_dialog = only_stack_id_dataset_dialog.is_some();
        etomo_director::INSTANCE.with_user_configuration(|user_configuration| {
            if !only_dataset_dialog {
                let table = self.table();
                if self.initial_value_row.borrow().is_none() {
                    *self.initial_value_row.borrow_mut() =
                        Some(BatchRunTomoRow::get_defaults_instance(
                            table.manager,
                            table.dialog.clone(),
                            Some(table.basic_directives.clone()),
                        ));
                }
                let initial_value_row = self.initial_value_row.borrow().clone().unwrap();
                initial_value_row.set_values_user_configuration(user_configuration);
                initial_value_row.set_values_directive_files(directive_file_collection);
                let list = self.list.borrow().clone();
                for row in &list {
                    row.apply_values(
                        init,
                        retain_user_values,
                        user_configuration,
                        directive_file_collection,
                        only_dataset_dialog,
                        only_advanced_dataset_dialog,
                    );
                }
                self.set_frame(false);
            } else if let Some(row) = self.get_row(only_stack_id_dataset_dialog) {
                row.apply_values(
                    init,
                    retain_user_values,
                    user_configuration,
                    directive_file_collection,
                    only_dataset_dialog,
                    only_advanced_dataset_dialog,
                );
            }
        });
    }

    /// Java private `setParameters(BatchRunTomoMetaData, String, boolean)`.
    fn set_parameters_meta_data(
        &self,
        meta_data: &BatchRunTomoMetaData,
        only_stack_id_dataset_dialog: Option<&str>,
        init: bool,
    ) {
        let only_dataset_dialog = only_stack_id_dataset_dialog.is_some();
        if !only_dataset_dialog {
            self.load(meta_data);
            let list = self.list.borrow().clone();
            for row in &list {
                row.set_parameters_meta_data(meta_data, only_dataset_dialog, init);
            }
            self.status_changed_status(meta_data.get_status().map(StatusRef::BatchRunTomoStatus));
        } else if let Some(row) = self.get_row(only_stack_id_dataset_dialog) {
            row.set_parameters_meta_data(meta_data, only_dataset_dialog, init);
        }
    }

    /// Java private `setParameters(SeriesWatcherMetaData, String, boolean)`.
    fn set_parameters_series_watcher_meta_data(
        &self,
        meta_data: &SeriesWatcherMetaData,
        stack_id: Option<&str>,
        init: bool,
    ) {
        let mut row = self.get_row(stack_id);
        if row.is_none() {
            // Create a blank row for the new stack. Data must be added from the
            // serieswatcher meta data.
            row = self.add_blank_row(stack_id);
        }
        if let Some(row) = row {
            row.set_parameters_series_watcher_meta_data(
                meta_data,
                self.table_reference.get_file_path(stack_id).as_deref(),
                init,
            );
        }
    }

    /// Java private `addBlankRow(String)`.
    fn add_blank_row(&self, stack_id: Option<&str>) -> Option<Rc<BatchRunTomoRow>> {
        stack_id?;
        let table = self.table();
        let first_index = self.list.borrow().len() as i32;
        // Add the row.
        let row = BatchRunTomoRow::get_series_watcher_instance(
            table.manager,
            table.dialog.clone(),
            self.table.clone(),
            Some(table.basic_directives.clone()),
            first_index + 1,
            stack_id,
            Some(&table.preferred_table_size),
            Some(table.cbc_run_toggle.clone()),
        );
        // OrigStack is essential for limiting batchruntomo to one delivery. Should only
        // be set when the user chooses the file.
        row.add_status_change_listener(Some(self.as_listener()));
        self.add_to_row_changers(&row);
        self.add_row_listeners(&row);
        row.expand_stack(table.btn_stack.is_expanded());
        self.list.borrow_mut().push(row.clone());
        row.display(&table.viewport, table.cur_tab.get());
        if self.table_listener.borrow().is_some() && first_index == 0 {
            self.first_row_added();
        }
        table.viewport.adjust_viewport(first_index);
        self.remove_all();
        self.display(&table.viewport);
        ui_harness::with(|harness| harness.pack_base_manager(Some(table.base_manager())));
        table.update_display();
        Some(row)
    }

    /// Java private `getParameters(BatchRunTomoMetaData)`.
    fn get_parameters_meta_data(&self, meta_data: &BatchRunTomoMetaData) {
        let list = self.list.borrow().clone();
        for row in &list {
            row.get_parameters_meta_data(meta_data);
        }
    }

    /// Java private `getParameters(BatchruntomoParam, boolean, boolean, RunType,
    /// StringBuilder, boolean, List<UniqueKey>, boolean)`.
    #[allow(clippy::too_many_arguments)]
    fn get_parameters_param(
        &self,
        param: &mut BatchruntomoParam,
        deliver_off: bool,
        deliver_to_directory: bool,
        run_type: Option<RunType>,
        err_msg: &mut String,
        do_validation: bool,
        mut manager_key_list: Option<&mut Vec<UniqueKey>>,
        validate_only: bool,
    ) -> bool {
        let mut run = false;
        let mut num_errors = 0;
        let list = self.list.borrow().clone();
        for row in &list {
            run = row.get_parameters_param(
                param,
                deliver_off,
                deliver_to_directory,
                run_type,
                err_msg,
                &mut num_errors,
                do_validation,
                manager_key_list.as_deref_mut(),
                validate_only,
            ) || run;
        }
        if do_validation && !run && run_type != Some(RunType::SeriesWatcher) {
            let table = self.table();
            ui_harness::with(|harness| {
                harness.open_message_dialog_base_manager_ui_component_string_string_axis_id(
                    Some(table.base_manager()),
                    Some(&*table as &dyn UIComponent),
                    &format!("Must check at least one {} checkbox", RUN_LABEL),
                    "Nothing to Do",
                    Some(AXIS_ID),
                )
            });
            return false;
        }
        true
    }

    /// Java private `saveAutodocs(TemplatePanel, NameValuePairList, NameValuePairList,
    /// boolean, boolean, File, String, boolean)`.
    #[allow(clippy::too_many_arguments)]
    fn save_autodocs(
        &self,
        template_panel: Option<&TemplatePanel>,
        global_batch_list: Option<&NameValuePairList>,
        templates: Option<&NameValuePairList>,
        do_validation: bool,
        init: bool,
        deliver_to_directory: Option<&std::path::Path>,
        autodoc_stack_id: Option<&str>,
        validate_only: bool,
    ) -> bool {
        let table = self.table();
        if autodoc_stack_id.is_none() {
            let list = self.list.borrow().clone();
            for row in &list {
                if !row.save_autodoc(
                    template_panel,
                    global_batch_list,
                    templates,
                    do_validation,
                    init,
                    deliver_to_directory,
                    &*table,
                    validate_only,
                ) {
                    return false;
                }
            }
        } else if let Some(row) = self.get_row(autodoc_stack_id)
            && !row.save_autodoc(
                template_panel,
                global_batch_list,
                templates,
                do_validation,
                init,
                deliver_to_directory,
                &*table,
                validate_only,
            )
        {
            return false;
        }
        true
    }

    /// Java private `loadAutodocs(String, boolean)`.
    fn load_autodocs(
        &self,
        only_stack_id_dataset_dialog: Option<&str>,
        only_advanced_dataset_dialog: bool,
    ) {
        let only_dataset_dialog = only_stack_id_dataset_dialog.is_some();
        if !only_dataset_dialog {
            let list = self.list.borrow().clone();
            for row in &list {
                row.load_autodoc(only_dataset_dialog, only_advanced_dataset_dialog);
            }
        } else if let Some(row) = self.get_row(only_stack_id_dataset_dialog) {
            row.load_autodoc(only_dataset_dialog, only_advanced_dataset_dialog);
        }
    }
}
