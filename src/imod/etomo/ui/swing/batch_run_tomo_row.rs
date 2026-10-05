//! `IMOD/Etomo/src/etomo/ui/swing/BatchRunTomoRow.java`.
//!
//! A row of the BatchRunTomo dataset table.  An event dispatch thread object, created
//! as `Rc<Self>` by the `get_*instance` functions; it keeps weak references to its
//! table and dialog.
//!
//! The Java class's private `statusChangedOldVersion(BatchRunTomoStatus)`,
//! `statusChangedOldVersion(StatusChangeEvent, boolean)`, `resetEndingStep()`,
//! `setEndingStep(EndingStep)` and `setCurAxisID(boolean)` are only reachable from each
//! other ("Don't use this"); they are dead in the Java too and are not translated
//! (DEAD_CODE.md).

use std::cell::{Cell, RefCell};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::rc::{Rc, Weak};
use std::sync::Arc;

use super::batch_run_tomo_dataset_dialog::BatchRunTomoDatasetDialog;
use super::batch_run_tomo_dialog::BatchRunTomoDialog;
use super::batch_run_tomo_step_panel;
use super::batch_run_tomo_table::{self, BatchRunTomoTable, DatasetColumn};
use super::button_cell::ButtonCell;
use super::cell::CellVirtual;
use super::check_box_cell::CheckBoxCell;
use super::deferred_3dmod_button::Deferred3dmodButton;
use super::field_cell::FieldCell;
use super::header_cell::HeaderCell;
use super::highlightable::Highlightable;
use super::highlighter_button::HighlighterButton;
use super::input_cell::InputCellVirtual;
use super::minibutton_cell::MinibuttonCell;
use super::run_3dmod_button_container::Run3dmodButtonContainer;
use super::template_panel::TemplatePanel;
use super::ui_harness;
use super::viewport::Viewport;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::batch_run_tomo_manager::BatchRunTomoManager;
use crate::imod::etomo::comscript::batchruntomo_param::BatchruntomoParam;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::jdk::{ActionEvent, ActionListener, GRID_BAG_REMAINDER, JComponent};
use crate::imod::etomo::logic::batch_tool;
use crate::imod::etomo::logic::dataset_tool;
use crate::imod::etomo::process::imod_manager;
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::process::process_output_strings;
use crate::imod::etomo::storage::autodoc::autodoc_factory;
use crate::imod::etomo::storage::directive_def::DirectiveDef;
use crate::imod::etomo::storage::directive_file::DirectiveFile;
use crate::imod::etomo::storage::directive_file_collection::DirectiveFileCollection;
use crate::imod::etomo::storage::directive_file_interface::DirectiveFileInterface;
use crate::imod::etomo::storage::log_file::LogFileError;
use crate::imod::etomo::storage::name_value_pair_list::NameValuePairList;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::batch_run_tomo_dataset_state::BatchRunTomoDatasetState;
use crate::imod::etomo::r#type::batch_run_tomo_dataset_status::BatchRunTomoDatasetStatus;
use crate::imod::etomo::r#type::batch_run_tomo_meta_data::BatchRunTomoMetaData;
use crate::imod::etomo::r#type::batch_run_tomo_row_meta_data::BatchRunTomoRowMetaData;
use crate::imod::etomo::r#type::batch_run_tomo_row_status::BatchRunTomoRowStatus;
use crate::imod::etomo::r#type::batch_run_tomo_status::{self, BatchRunTomoStatus};
use crate::imod::etomo::r#type::directive_file_type::DirectiveFileType;
use crate::imod::etomo::r#type::ending_step::EndingStep;
use crate::imod::etomo::r#type::extension::Extension;
use crate::imod::etomo::r#type::field_properties_adapter::FieldPropertiesAdapter;
use crate::imod::etomo::r#type::field_settings::FieldSettings;
use crate::imod::etomo::r#type::file_type::{self, FileType};
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::run_status::RunStatus;
use crate::imod::etomo::r#type::run_type::RunType;
use crate::imod::etomo::r#type::series_watcher_meta_data::SeriesWatcherMetaData;
use crate::imod::etomo::r#type::status::StatusRef;
use crate::imod::etomo::r#type::status_change_event::StatusChangeEvent;
use crate::imod::etomo::r#type::status_change_listener::StatusChangeListener;
use crate::imod::etomo::r#type::status_change_row_event::StatusChangeRowEvent;
use crate::imod::etomo::r#type::status_changer::StatusChanger;
use crate::imod::etomo::r#type::step::Step;
use crate::imod::etomo::r#type::user_configuration::UserConfiguration;
use crate::imod::etomo::ui::batch_run_tomo_row_state::BatchRunTomoRowState;
use crate::imod::etomo::ui::batch_run_tomo_tab::BatchRunTomoTab;
use crate::imod::etomo::ui::field::Field;
use crate::imod::etomo::ui::field_displayer::FieldDisplayer;
use crate::imod::etomo::ui::field_validation_failed_exception::FieldValidationFailedException;
use crate::imod::etomo::ui::preferred_table_size::PreferredTableSize;
use crate::imod::etomo::ui::shared_strings;
use crate::imod::etomo::ui::table_component::TableComponent;
use crate::imod::etomo::ui::ui_component::UIComponent;
use crate::imod::etomo::util::unique_key::UniqueKey;
use crate::imod::etomo::util::utilities;

/// Java private static final `SURFACES_TO_ANALYZE_TWO`.
const SURFACES_TO_ANALYZE_TWO: &str = "2";
/// Java private static final `SURFACES_TO_ANALYZE_ONE`.
const SURFACES_TO_ANALYZE_ONE: &str = "1";
/// Java private static final `EDIT_DATASET_VALUE`.
const EDIT_DATASET_VALUE: &str = "   Set";

/// The set of basic directives the dialog, the table and every row share (Java
/// `Set<DirectiveDef>` passed by reference).
pub type BasicDirectives = Rc<RefCell<HashSet<DirectiveDef>>>;

/// Java `final class BatchRunTomoRow implements Highlightable, Run3dmodButtonContainer,
/// ActionListener, StatusChangeListener, StatusChanger`.
pub struct BatchRunTomoRow {
    /// Java private final `cbcBoundaryModel`.
    cbc_boundary_model: Rc<CheckBoxCell>,
    /// Java private final `cbcDual`.
    cbc_dual: Rc<CheckBoxCell>,
    /// Java private final `cbcMontage`.
    cbc_montage: Rc<CheckBoxCell>,
    /// Java private final `fcSkip`.
    fc_skip: Rc<FieldCell>,
    /// Java private final `fcBskip`.
    fc_bskip: Rc<FieldCell>,
    /// Java private final `cbcSurfacesToAnalyze2`.
    cbc_surfaces_to_analyze2: Rc<CheckBoxCell>,
    /// Java private final `fcEditDataset`.
    fc_edit_dataset: Rc<FieldCell>,
    /// Java private final `fcDatasetState`.
    fc_dataset_state: Rc<FieldCell>,
    /// Java private final `fcEndingStep`.
    fc_ending_step: Rc<FieldCell>,
    /// Java private final `fcCurAxisLetter`.
    fc_cur_axis_letter: Rc<FieldCell>,
    /// Java private final `cbcRun`.
    cbc_run: Rc<CheckBoxCell>,
    /// Java private final `mbcOpenDataset`.
    mbc_open_dataset: Rc<MinibuttonCell>,
    /// Java private final `mbcImageStackA`.
    mbc_image_stack_a: Rc<MinibuttonCell>,
    /// Java private final `mbcImageStackB`.
    mbc_image_stack_b: Rc<MinibuttonCell>,
    /// Java private final `hcNumber`.
    hc_number: Rc<HeaderCell>,
    /// Java private final `bcEditDataset`.
    bc_edit_dataset: Rc<ButtonCell>,
    /// Java private final `firstSteps`.
    first_steps: RefCell<[bool; batch_run_tomo_step_panel::STEP_PAIRS]>,
    /// Java private final `mbcTomogram`.
    mbc_tomogram: Rc<MinibuttonCell>,
    /// Java private final `mbcProjLog`.
    mbc_proj_log: Rc<MinibuttonCell>,
    /// Java private final `mbcBRTLog`.
    mbc_brt_log: Rc<MinibuttonCell>,

    /// Java private `listeners`, initially null.
    listeners: RefCell<Option<Vec<Rc<dyn StatusChangeListener>>>>,

    /// Java private final `manager`.
    manager: &'static BatchRunTomoManager,
    /// Java private final `hbRow`.
    hb_row: Rc<HighlighterButton>,
    /// Java private final `fcStack`.
    fc_stack: Rc<FieldCell>,
    /// Java private final `stackID`.
    stack_id: Option<String>,
    /// Java private final `table` (null for the defaults instance).
    table: Option<Weak<BatchRunTomoTable>>,
    /// Java private final `basicDirectives`.
    basic_directives: Option<BasicDirectives>,
    /// Java private final `newVersion`.
    #[allow(dead_code)]
    new_version: bool,
    /// Java private final `dialog`.
    dialog: Weak<BatchRunTomoDialog>,
    /// Java private final `rowState`.
    row_state: RefCell<BatchRunTomoRowState>,
    /// Java private final `cbcRunToggle`.
    cbc_run_toggle: Option<Rc<CheckBoxCell>>,

    /// Java private `imodIndexA`, initially -1.
    imod_index_a: Cell<i32>,
    /// Java private `imodIndexB`, initially -1.
    imod_index_b: Cell<i32>,
    /// Java private `imodRec`, initially -1.
    imod_rec: Cell<i32>,
    /// Java private `imodTrimVol`, initially -1.
    #[allow(dead_code)]
    imod_trim_vol: Cell<i32>,
    /// Java private `datasetDialog`, initially null.
    dataset_dialog: RefCell<Option<Rc<BatchRunTomoDatasetDialog>>>,
    /// Java private `metaData`, initially null.
    meta_data: RefCell<Option<Arc<BatchRunTomoRowMetaData>>>,
    /// Java private `status`, initially `BatchRunTomoStatus.DEFAULT`.
    status: Cell<Option<BatchRunTomoStatus>>,
    /// Java private `debug`, initially false.
    #[allow(dead_code)]
    debug: Cell<bool>,
    /// Java private `origStack`, initially null.
    orig_stack: RefCell<Option<String>>,
    /// Java private `datasetState`, initially null (only the dead old-version status
    /// handler set it).
    dataset_state: Cell<Option<BatchRunTomoDatasetState>>,
    /// Java private `tomogramDone`, initially false.
    tomogram_done: Cell<bool>,
    /// Java private `trimvolDone`, initially false.
    trimvol_done: Cell<bool>,
    /// Java private `log`, initially null.
    log: RefCell<Option<PathBuf>>,
    /// Java private `managerKey`, initially null.
    manager_key: RefCell<Option<UniqueKey>>,
    /// Java private `runStatus`, initially null.
    run_status: Cell<Option<RunStatus>>,
    /// Java private `curEndingStepA`, initially null.  Associated with AxisID.FIRST.
    cur_ending_step_a: Cell<Option<EndingStep>>,
    /// Java private `curEndingStep`, initially null.  Associated with AxisID.ONLY if
    /// single, AxisID.SECOND if dual.
    cur_ending_step: Cell<Option<EndingStep>>,
    /// Java private `curAxisID`, initially null: null, AxisID.ONLY if single, or
    /// AxisID.FIRST or SECOND if dual.
    cur_axis_id: Cell<Option<AxisID>>,
    /// Java `this`.
    this: Weak<BatchRunTomoRow>,
}

impl BatchRunTomoRow {
    /// Java private constructor `BatchRunTomoRow(BatchRunTomoManager, BatchRunTomoDialog,
    /// BatchRunTomoTable, Set<DirectiveDef>, int, File, BatchRunTomoRow, AxisType,
    /// String, Boolean, PreferredTableSize, boolean, boolean, CheckBoxCell)`.
    #[allow(clippy::too_many_arguments)]
    fn new(
        manager: &'static BatchRunTomoManager,
        dialog: Weak<BatchRunTomoDialog>,
        table: Option<Weak<BatchRunTomoTable>>,
        basic_directives: Option<BasicDirectives>,
        number: i32,
        stack: Option<&Path>,
        prev_row: Option<&Rc<BatchRunTomoRow>>,
        axis_type: Option<AxisType>,
        stack_id: Option<&str>,
        surfaces_to_analyze2: Option<bool>,
        preferred_table_size: Option<&PreferredTableSize>,
        _new_version: bool,
        new_row: bool,
        cbc_run_toggle: Option<Rc<CheckBoxCell>>,
    ) -> Rc<BatchRunTomoRow> {
        let instance = Rc::new_cyclic(|this: &Weak<BatchRunTomoRow>| {
            let container: Weak<dyn Run3dmodButtonContainer> = this.clone();
            let highlightable: Weak<dyn Highlightable> = this.clone();
            let group: Option<Weak<dyn Highlightable>> =
                table.clone().map(|table| table as Weak<dyn Highlightable>);
            BatchRunTomoRow {
                cbc_boundary_model: CheckBoxCell::get_instance(),
                cbc_dual: CheckBoxCell::get_named_instance_string_string_string(
                    Some(batch_run_tomo_table::DUAL_LABEL1),
                    Some(batch_run_tomo_table::DUAL_LABEL2),
                    None,
                ),
                cbc_montage: CheckBoxCell::get_instance(),
                fc_skip: FieldCell::get_editable_instance(),
                fc_bskip: FieldCell::get_editable_instance(),
                cbc_surfaces_to_analyze2: CheckBoxCell::get_named_instance_string_string_string(
                    Some(batch_run_tomo_table::SURFACES_TO_ANALYZE_LABEL1),
                    Some(batch_run_tomo_table::SURFACES_TO_ANALYZE_LABEL2),
                    Some(batch_run_tomo_table::SURFACES_TO_ANALYZE_LABEL3),
                ),
                fc_edit_dataset: FieldCell::get_ineditable_instance(),
                fc_dataset_state: FieldCell::get_named_ineditable_instance_string(Some(
                    batch_run_tomo_table::STATUS_LABEL,
                )),
                fc_ending_step: FieldCell::get_named_ineditable_instance_string(Some(
                    batch_run_tomo_table::STEP_LABEL,
                )),
                fc_cur_axis_letter: FieldCell::get_named_ineditable_instance_string(Some(
                    batch_run_tomo_table::CUR_AXIS_LABEL,
                )),
                cbc_run: CheckBoxCell::get_named_instance_string(Some(
                    batch_run_tomo_table::RUN_LABEL,
                )),
                mbc_open_dataset: MinibuttonCell::get_named_etomo_instance(
                    Some(batch_run_tomo_table::DATASET_LABEL1),
                    Some(batch_run_tomo_table::DATASET_LABEL2),
                ),
                mbc_image_stack_a:
                    MinibuttonCell::get_run_3dmod_instance_run_3dmod_button_container(Some(
                        container.clone(),
                    )),
                mbc_image_stack_b:
                    MinibuttonCell::get_run_3dmod_instance_run_3dmod_button_container(Some(
                        container,
                    )),
                hc_number: HeaderCell::new_void(),
                bc_edit_dataset: ButtonCell::get_toggle_instance(Some("Open")),
                first_steps: RefCell::new([false; batch_run_tomo_step_panel::STEP_PAIRS]),
                mbc_tomogram: MinibuttonCell::get_named_run_3dmod_instance(
                    Some(batch_run_tomo_table::REC_LABEL1),
                    Some(batch_run_tomo_table::REC_LABEL2),
                ),
                mbc_proj_log: MinibuttonCell::get_named_etomo_log_instance(
                    Some(batch_run_tomo_table::PROJ_LOG_LABEL1),
                    Some(batch_run_tomo_table::PROJ_LOG_LABEL2),
                ),
                mbc_brt_log: MinibuttonCell::get_named_brt_log_instance(
                    Some(batch_run_tomo_table::LOG_LABEL1),
                    Some(batch_run_tomo_table::LOG_LABEL2),
                ),
                listeners: RefCell::new(None),
                manager,
                hb_row: HighlighterButton::get_instance(highlightable, group),
                fc_stack: FieldCell::get_expandable_ineditable_instance(None),
                stack_id: stack_id.map(str::to_owned),
                table: table.clone(),
                basic_directives: basic_directives.clone(),
                // `this.newVersion = true`.
                new_version: true,
                dialog: dialog.clone(),
                row_state: RefCell::new(BatchRunTomoRowState::new(stack_id)),
                cbc_run_toggle: cbc_run_toggle.clone(),
                imod_index_a: Cell::new(-1),
                imod_index_b: Cell::new(-1),
                imod_rec: Cell::new(-1),
                imod_trim_vol: Cell::new(-1),
                dataset_dialog: RefCell::new(None),
                meta_data: RefCell::new(None),
                status: Cell::new(Some(batch_run_tomo_status::DEFAULT)),
                debug: Cell::new(false),
                orig_stack: RefCell::new(None),
                dataset_state: Cell::new(None),
                tomogram_done: Cell::new(false),
                trimvol_done: Cell::new(false),
                log: RefCell::new(None),
                manager_key: RefCell::new(None),
                run_status: Cell::new(None),
                cur_ending_step_a: Cell::new(None),
                cur_ending_step: Cell::new(None),
                cur_axis_id: Cell::new(None),
                this: this.clone(),
            }
        });
        let this = &instance;
        if let Some(cbc_run_toggle) = &this.cbc_run_toggle
            && number >= 0
        {
            cbc_run_toggle.add_target(&this.cbc_run);
        }
        this.hc_number.set_text_int(number);
        this.fc_stack.set_value_file(stack);
        this.set_log(stack);
        // preferred width
        if let Some(preferred_table_size) = preferred_table_size {
            preferred_table_size.add_column(
                DatasetColumn::NUMBER.get_index(),
                Some(this.hc_number.clone() as Rc<dyn TableComponent>),
            );
            preferred_table_size.add_column(
                DatasetColumn::STACK.get_index(),
                Some(this.fc_stack.clone() as Rc<dyn TableComponent>),
            );
            preferred_table_size.add_column_pair(
                DatasetColumn::EDIT_DATASET.get_index(),
                Some(this.bc_edit_dataset.clone() as Rc<dyn TableComponent>),
                Some(this.fc_edit_dataset.clone() as Rc<dyn TableComponent>),
            );
        }
        // init
        this.fc_ending_step
            .set_horizontal_alignment(JTEXTFIELD_CENTER);
        this.fc_cur_axis_letter
            .set_horizontal_alignment(JTEXTFIELD_CENTER);
        this.copy(prev_row.map(|row| &**row));
        this.set_tooltips(prev_row);
        // If axisType is set, use it to set dual checkbox, otherwise keep the default
        // axisType from the previous row.
        if axis_type == Some(AxisType::DualAxis) {
            this.cbc_dual.set_selected_boolean(true);
        } else if axis_type == Some(AxisType::SingleAxis) {
            this.cbc_dual.set_selected_boolean(false);
        }
        this.cbc_run.set_selected_boolean(true);
        // set directives
        let axis_id = AxisID::get_instance_from_file_name(
            Some(if this.cbc_dual.is_selected() {
                AxisType::DualAxis
            } else {
                AxisType::SingleAxis
            }),
            this.fc_stack.get_contracted_value().as_deref(),
        );
        this.setup_field(
            &*this.fc_stack,
            Some(if axis_id == AxisID::Second {
                DirectiveDef::CURRENT_B_STACK_EXT
            } else {
                DirectiveDef::CURRENT_STACK_EXT
            }),
            None,
        );
        this.setup_field(&*this.cbc_dual, Some(DirectiveDef::DUAL), None);
        this.setup_field(&*this.cbc_montage, Some(DirectiveDef::MONTAGE), None);
        this.setup_field(
            &*this.cbc_surfaces_to_analyze2,
            Some(DirectiveDef::SURFACES_TO_ANALYZE),
            Some(DirectiveDef::TWO_SURFACES),
        );
        this.setup_field(&*this.fc_skip, Some(DirectiveDef::SKIP), None);
        this.setup_field(
            &*this.fc_bskip,
            DirectiveDef::get_instance_from_csv(
                Some("setupset.copyarg.bskip"),
                Some(DirectiveDef::SKIP),
            ),
            None,
        );
        // Also works with DirectiveDef.RAW_BOUNDARY_MODEL_FOR_PATCH_TRACKING
        this.setup_field(
            &*this.cbc_boundary_model,
            Some(DirectiveDef::RAW_BOUNDARY_MODEL_FOR_SEED_FINDING),
            Some(DirectiveDef::RAW_BOUNDARY_MODEL_FOR_PATCH_TRACKING),
        );
        *this.first_steps.borrow_mut() = [false; batch_run_tomo_step_panel::STEP_PAIRS];
        if new_row {
            this.cbc_run.set_selected_boolean(true);
            this.mbc_open_dataset.set_editable(false);
            this.mbc_tomogram.set_editable(false);
            if let Some(surfaces_to_analyze2) = surfaces_to_analyze2 {
                this.cbc_surfaces_to_analyze2
                    .set_selected_boolean(surfaces_to_analyze2);
            }
            this.mbc_proj_log.set_editable(false);
            this.mbc_brt_log.set_editable(false);
        }
        this.update_display(None, None);
        this.status_changed_status(this.status.get().map(StatusRef::BatchRunTomoStatus));
        instance
    }

    /// Java private `setupField(Field, DirectiveDef, DirectiveDef)` (and the
    /// two-argument overload, with a null second directive).
    fn setup_field(
        &self,
        field: &dyn Field,
        directive_def: Option<DirectiveDef>,
        directive_def2: Option<DirectiveDef>,
    ) {
        field.set_directive_def(directive_def);
        if let Some(basic_directives) = &self.basic_directives {
            let mut basic_directives = basic_directives.borrow_mut();
            if let Some(directive_def) = directive_def
                && !basic_directives.contains(&directive_def)
            {
                basic_directives.insert(directive_def);
            }
            if let Some(directive_def2) = directive_def2
                && !basic_directives.contains(&directive_def2)
            {
                basic_directives.insert(directive_def2);
            }
        }
    }

    /// Java package-private static `getInstance(BatchRunTomoManager, BatchRunTomoDialog,
    /// BatchRunTomoTable, Set<DirectiveDef>, int, File, BatchRunTomoRow, AxisType,
    /// String, PreferredTableSize, boolean, CheckBoxCell)`.
    #[allow(clippy::too_many_arguments)]
    pub fn get_instance(
        manager: &'static BatchRunTomoManager,
        dialog: Weak<BatchRunTomoDialog>,
        table: Weak<BatchRunTomoTable>,
        basic_directives: Option<BasicDirectives>,
        number: i32,
        stack: Option<&Path>,
        prev_row: Option<&Rc<BatchRunTomoRow>>,
        axis_type: Option<AxisType>,
        stack_id: Option<&str>,
        dataset_width: Option<&PreferredTableSize>,
        new_row: bool,
        cbc_run_toggle: Option<Rc<CheckBoxCell>>,
    ) -> Rc<BatchRunTomoRow> {
        let instance = BatchRunTomoRow::new(
            manager,
            dialog,
            Some(table),
            basic_directives,
            number,
            stack,
            prev_row,
            axis_type,
            stack_id,
            None,
            dataset_width,
            true,
            new_row,
            cbc_run_toggle,
        );
        instance.add_listeners();
        instance
    }

    /// Java package-private static `getDefaultsInstance(BatchRunTomoManager,
    /// BatchRunTomoDialog, Set<DirectiveDef>)`.
    pub fn get_defaults_instance(
        manager: &'static BatchRunTomoManager,
        dialog: Weak<BatchRunTomoDialog>,
        basic_directives: Option<BasicDirectives>,
    ) -> Rc<BatchRunTomoRow> {
        BatchRunTomoRow::new(
            manager,
            dialog,
            None,
            basic_directives,
            -1,
            None,
            None,
            None,
            None,
            None,
            None,
            true,
            false,
            None,
        )
    }

    /// Java package-private static `getSeriesWatcherInstance(BatchRunTomoManager,
    /// BatchRunTomoDialog, BatchRunTomoTable, Set<DirectiveDef>, int, String,
    /// PreferredTableSize, CheckBoxCell)`.  Create an instance that will be filled with
    /// data from the serieswatcher project file.
    #[allow(clippy::too_many_arguments)]
    pub fn get_series_watcher_instance(
        manager: &'static BatchRunTomoManager,
        dialog: Weak<BatchRunTomoDialog>,
        table: Weak<BatchRunTomoTable>,
        basic_directives: Option<BasicDirectives>,
        number: i32,
        stack_id: Option<&str>,
        dataset_width: Option<&PreferredTableSize>,
        cbc_run_toggle: Option<Rc<CheckBoxCell>>,
    ) -> Rc<BatchRunTomoRow> {
        let (axis_type, surfaces_to_analyze2) = match dialog.upgrade() {
            Some(dialog) => (
                dialog.get_series_watcher_axis_type(),
                dialog.get_series_watcher_surfaces_to_analyze2(),
            ),
            None => (None, None),
        };
        let instance = BatchRunTomoRow::new(
            manager,
            dialog,
            Some(table),
            basic_directives,
            number,
            None,
            None,
            axis_type,
            stack_id,
            surfaces_to_analyze2,
            dataset_width,
            true,
            true,
            cbc_run_toggle,
        );
        instance.add_listeners();
        instance
    }

    fn table(&self) -> Option<Rc<BatchRunTomoTable>> {
        self.table.as_ref().and_then(Weak::upgrade)
    }

    fn dialog(&self) -> Option<Rc<BatchRunTomoDialog>> {
        self.dialog.upgrade()
    }

    fn base_manager(&self) -> &'static dyn BaseManager {
        self.manager
    }

    /// The expanded stack path, Java `new File(fcStack.getExpandedValue())`.
    fn stack_file(&self) -> PathBuf {
        PathBuf::from(self.fc_stack.get_expanded_value().unwrap_or_default())
    }

    /// Java package-private `copy(BatchRunTomoRow)`.
    pub fn copy(&self, prev_row: Option<&BatchRunTomoRow>) {
        if let Some(prev_row) = prev_row {
            self.cbc_dual
                .set_selected_boolean(prev_row.cbc_dual.is_selected());
            self.cbc_montage
                .set_selected_boolean(prev_row.cbc_montage.is_selected());
            self.cbc_surfaces_to_analyze2
                .set_selected_boolean(prev_row.cbc_surfaces_to_analyze2.is_selected());
        }
        self.update_display(None, None);
    }

    /// Java package-private `validate(BatchRunTomoDatasetDialog)`.
    pub fn validate(&self, global_dataset_dialog: Option<&Rc<BatchRunTomoDatasetDialog>>) -> bool {
        let surfaces_to_analyze2 = self.cbc_surfaces_to_analyze2.is_selected();
        if let Some(dataset_dialog) = self.dataset_dialog.borrow().clone() {
            return dataset_dialog.validate_boolean(surfaces_to_analyze2);
        }
        if let Some(global_dataset_dialog) = global_dataset_dialog {
            return global_dataset_dialog.validate_boolean(surfaces_to_analyze2);
        }
        true
    }

    /// Java package-private `isParallelProcessing()`.
    pub fn is_parallel_processing(&self) -> bool {
        if let Some(table) = self.table() {
            return table.is_parallel_processing();
        }
        false
    }

    /// Java package-private `isDual()`.
    pub fn is_dual(&self) -> bool {
        self.cbc_dual.is_selected()
    }

    /// Java package-private `isRun()`.
    pub fn is_run(&self) -> bool {
        self.cbc_run.is_enabled() && self.cbc_run.is_selected()
    }

    /// Java package-private `isMontage()`.
    pub fn is_montage(&self) -> bool {
        self.cbc_montage.is_selected()
    }

    /// Java package-private `getEndingStep()`.  Get current ending step for the ending
    /// step panel.  The row must be incomplete.  And for dual axis, return only the B
    /// axis ending steps.
    pub fn get_ending_step(&self) -> Option<EndingStep> {
        let row_state = self.row_state.borrow();
        if row_state.get_dataset_state() == Some(BatchRunTomoDatasetState::Done) {
            return None;
        }
        if self.cbc_dual.is_selected()
            && row_state.get_ending_step_axis_id() != Some(AxisID::Second)
        {
            return None;
        }
        row_state.get_ending_step()
    }

    /// Java package-private `getDatasetState()`.
    pub fn get_dataset_state(&self) -> Option<BatchRunTomoDatasetState> {
        if self.fc_dataset_state.is_empty() {
            return None;
        }
        BatchRunTomoDatasetState::get_instance(self.fc_dataset_state.get_text_void().as_deref())
    }

    /// Java package-private `getRunStatus()`.
    pub fn get_run_status(&self) -> Option<RunStatus> {
        self.run_status.get()
    }

    /// Java package-private `getStackID()`.
    pub fn get_stack_id(&self) -> Option<String> {
        self.stack_id.clone()
    }

    /// Java package-private `getStack()`.
    pub fn get_stack(&self) -> Option<PathBuf> {
        self.fc_stack.get_file()
    }

    /// Java package-private `getStackPath()`.
    pub fn get_stack_path(&self) -> Option<PathBuf> {
        let file = self.fc_stack.get_file()?;
        file.parent().map(Path::to_path_buf)
    }

    /// This row as the `ActionListener` Java registers as `this`.
    fn action_listener(&self) -> ActionListener {
        let this = self.this.clone();
        Rc::new(move |event: &ActionEvent| {
            if let Some(row) = this.upgrade() {
                row.action_performed(event);
            }
        })
    }

    /// Java private `addListeners()`.
    fn add_listeners(&self) {
        // give each listened to field an unique action command
        self.mbc_image_stack_a
            .set_action_command(Some(&self.mbc_image_stack_a.get_unique_action_command()));
        self.mbc_image_stack_b
            .set_action_command(Some(&self.mbc_image_stack_b.get_unique_action_command()));
        self.cbc_dual
            .set_action_command(self.cbc_dual.get_unique_action_command().as_deref());
        self.mbc_open_dataset
            .set_action_command(Some(&self.mbc_open_dataset.get_unique_action_command()));
        self.mbc_tomogram
            .set_action_command(Some(&self.mbc_tomogram.get_unique_action_command()));
        self.mbc_proj_log
            .set_action_command(Some(&self.mbc_proj_log.get_unique_action_command()));
        self.mbc_brt_log
            .set_action_command(Some(&self.mbc_brt_log.get_unique_action_command()));
        self.cbc_boundary_model.set_action_command(
            self.cbc_boundary_model
                .get_unique_action_command()
                .as_deref(),
        );
        self.bc_edit_dataset
            .set_action_command(Some(&self.bc_edit_dataset.get_unique_action_command()));
        self.cbc_run
            .set_action_command(self.cbc_run.get_unique_action_command().as_deref());
        self.cbc_montage
            .set_action_command(self.cbc_montage.get_unique_action_command().as_deref());
        // set listeners
        let listener = self.action_listener();
        self.mbc_image_stack_a.add_action_listener(listener.clone());
        self.mbc_image_stack_b.add_action_listener(listener.clone());
        self.cbc_dual.add_action_listener(listener.clone());
        self.mbc_open_dataset.add_action_listener(listener.clone());
        self.mbc_tomogram.add_action_listener(listener.clone());
        self.mbc_proj_log.add_action_listener(listener.clone());
        self.mbc_brt_log.add_action_listener(listener.clone());
        self.cbc_boundary_model
            .add_action_listener(listener.clone());
        self.bc_edit_dataset.add_action_listener(listener.clone());
        self.cbc_run.add_action_listener(listener.clone());
        self.cbc_montage.add_action_listener(listener);
    }

    /// Java package-private `imodStack(FileType)`.  Opens 3dmod on the A axis; returns
    /// the file created from modelFileType.
    pub fn imod_stack_file_type(&self, model_file_type: Option<&FileType>) -> Option<PathBuf> {
        self.imod_stack_full(
            model_file_type,
            AxisID::First,
            self.cbc_dual.is_selected(),
            None,
        )
    }

    /// Java package-private `imodStack(File)`.  Opens 3dmod on the A axis.
    pub fn imod_stack_model_file(&self, model_file: Option<&Path>) {
        self.imod_stack_model(model_file, AxisID::First, self.cbc_dual.is_selected(), None);
    }

    /// Java private `imodStack(FileType, AxisID, boolean, Run3dmodMenuOptions)`.  Opens
    /// 3dmod; returns the file created from modelFileType.
    fn imod_stack_full(
        &self,
        model_file_type: Option<&FileType>,
        axis_id: AxisID,
        dual: bool,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    ) -> Option<PathBuf> {
        let stack = self.stack_file();
        if let Some(model_file_type) = model_file_type {
            let model_file = java_io_file_new_file(
                stack.parent(),
                &batch_tool::get_model_file_name(
                    self.manager,
                    Some(model_file_type),
                    &stack
                        .file_name()
                        .map(|name| name.to_string_lossy().into_owned())
                        .unwrap_or_default(),
                    dual,
                )
                .unwrap_or_else(|| "null".to_owned()),
            );
            self.imod_stack_model(Some(&model_file), axis_id, dual, run_3dmod_menu_options);
            return Some(model_file);
        }
        // No model is required
        self.imod_stack_no_model(axis_id, dual, run_3dmod_menu_options);
        None
    }

    /// Java private `imodStack(AxisID, boolean, Run3dmodMenuOptions)`.  Opens 3dmod
    /// stack without a model.
    fn imod_stack_no_model(
        &self,
        axis_id: AxisID,
        dual: bool,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    ) {
        self.imod_stack_model(None, axis_id, dual, run_3dmod_menu_options);
    }

    /// Java private `imodStack(File, AxisID, boolean, Run3dmodMenuOptions)`.  Opens
    /// 3dmod, with a model if modelFile is set.
    fn imod_stack_model(
        &self,
        model_file: Option<&Path>,
        axis_id: AxisID,
        dual: bool,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    ) {
        let stack_file = dataset_tool::get_stack_file(
            self.fc_stack.get_expanded_value().as_deref(),
            Some(axis_id),
            dual,
        );
        if axis_id == AxisID::Second {
            self.imod_index_b.set(self.manager.imod_stack(
                stack_file.as_deref(),
                axis_id,
                self.imod_index_b.get(),
                model_file,
                dual,
                run_3dmod_menu_options,
            ));
        } else {
            self.imod_index_a.set(self.manager.imod_stack(
                stack_file.as_deref(),
                axis_id,
                self.imod_index_a.get(),
                model_file,
                dual,
                run_3dmod_menu_options,
            ));
        }
    }

    /// Java `actionPerformed(ActionEvent)`.
    pub fn action_performed(&self, event: &ActionEvent) {
        if let Some(action_command) = event.get_action_command() {
            self.action(action_command, None, None);
        }
    }

    /// Java private `getDatasetFile()`.
    fn get_dataset_file(&self) -> Option<PathBuf> {
        let dual = self.cbc_dual.is_selected();
        dataset_tool::get_dataset_file(
            dataset_tool::get_stack_file(
                self.fc_stack.get_expanded_value().as_deref(),
                Some(AxisID::First),
                dual,
            )
            .as_deref(),
            dual,
        )
    }

    /// Java package-private `remove()`.
    pub fn remove(&self) {
        self.hc_number.remove();
        self.hb_row.remove();
        self.fc_stack.remove();
        self.cbc_boundary_model.remove();
        self.cbc_dual.remove();
        self.cbc_montage.remove();
        self.fc_skip.remove();
        self.fc_bskip.remove();
        self.cbc_surfaces_to_analyze2.remove();
        self.fc_edit_dataset.remove();
        self.fc_dataset_state.remove();
        self.fc_ending_step.remove();
        self.fc_cur_axis_letter.remove();
        self.cbc_run.remove();
        self.mbc_open_dataset.remove();
        self.mbc_tomogram.remove();
        self.mbc_proj_log.remove();
        self.mbc_brt_log.remove();
        self.mbc_image_stack_a.remove();
        self.mbc_image_stack_b.remove();
        self.bc_edit_dataset.remove();
    }

    /// Java package-private `delete()`.
    pub fn delete(&self) {
        if let Some(meta_data) = self.meta_data.borrow().as_ref() {
            meta_data.set_row_number(None);
        }
        if let Some(cbc_run_toggle) = &self.cbc_run_toggle {
            cbc_run_toggle.delete_target(&self.cbc_run);
        }
        self.delete_dataset();
    }

    /// Java package-private `deleteDataset()`.
    pub fn delete_dataset(&self) {
        let dataset_dialog = self.dataset_dialog.borrow().clone();
        if let Some(dataset_dialog) = dataset_dialog {
            dataset_dialog.set_visible(false);
            *self.dataset_dialog.borrow_mut() = None;
            self.bc_edit_dataset.set_selected(false);
            self.fc_edit_dataset.set_value_string(Some(""));
            self.manager.save_batch_run_tomo_dialog(
                None,
                false,
                false,
                self.stack_id.as_deref(),
                false,
                self.is_parallel_processing(),
                false,
            );
        }
    }

    /// Java package-private `isEditDataset()`.
    pub fn is_edit_dataset(&self) -> bool {
        self.bc_edit_dataset.is_selected()
    }

    /// Java package-private `updateDisplay(RunType, Boolean)`.
    pub fn update_display(&self, run_type: Option<RunType>, validate_only: Option<bool>) {
        // Enabled/disabled
        let dual = self.cbc_dual.is_selected();
        self.fc_bskip.set_enabled(dual);
        self.mbc_image_stack_b.set_enabled(dual);

        // Fields effected by serieswatcher.
        let series_watcher = self
            .table()
            .is_some_and(|table| table.is_series_watcher_on());

        // Preserve Resume on a processchunks pause/kill status. The run checkbox is
        // locked, the resume button is enabled. And the Subset of Steps to run stays
        // locked. Cancelled by the Reset button, which sends an open status.
        self.cbc_run.set_editable(
            !series_watcher && !self.row_state.borrow().is_processchunk_resume_enabled(),
        );

        // Copy from state information.
        let run_status = self.row_state.borrow().get_run_status(
            self.run_status.get(),
            run_type,
            self.is_run(),
            validate_only,
        );
        self.run_status.set(run_status);
        let row_state = self.row_state.borrow();
        self.fc_dataset_state
            .set_value_string(Some(&row_state.get_dataset_state_value()));
        self.fc_dataset_state
            .set_run_highlight(row_state.is_run_highlight());
        self.fc_dataset_state
            .set_error_boolean(row_state.is_error());
        self.fc_ending_step
            .set_value_string(Some(&row_state.get_ending_step_value()));
        self.fc_cur_axis_letter
            .set_value_string(Some(&row_state.get_ending_step_axis_letter()));

        let cur_recon_step = row_state.get_cur_recon_step();
        let dataset_directory_set = cur_recon_step.is_some();

        // The dataset directory name is different for single and dual datsets.
        self.cbc_dual.set_editable(!dataset_directory_set);

        // If the images will be moved and this is Windows, leaving the image files open
        // will make it impossible for the files to be moved to their new location. In
        // this case, image files open buttons behavior should be:
        // - Enabled when first loaded.
        // - Ask to close image file before the first run starts.
        // - Disabled when the first run starts.
        // - Enabled after the delivery is done.
        if utilities::is_windows_os()
            && self.dialog().is_some_and(|dialog| dialog.is_deliver())
            && run_type.is_some()
            && validate_only == Some(false)
            && !dataset_directory_set
        {
            let stack = self.stack_file();
            if self.mbc_image_stack_a.is_editable() {
                self.mbc_image_stack_a.set_editable(false);
                self.manager.close_imod_key_file_move(
                    Some(imod_manager::BATCH_RUN_TOMO_STACK_KEY),
                    Some(&stack),
                    Some(if self.is_dual() {
                        AxisID::First
                    } else {
                        AxisID::Only
                    }),
                    Some(imod_manager::BATCH_RUN_TOMO_STACK_KEY),
                    false,
                    true,
                );
            }
            if self.is_dual() && self.mbc_image_stack_b.is_editable() {
                self.mbc_image_stack_b.set_editable(false);
                self.manager.close_imod_key_file_move(
                    Some(imod_manager::BATCH_RUN_TOMO_STACK_KEY),
                    Some(&stack),
                    Some(if self.is_dual() {
                        AxisID::Second
                    } else {
                        AxisID::Only
                    }),
                    Some(imod_manager::BATCH_RUN_TOMO_STACK_KEY),
                    false,
                    true,
                );
            }
        } else {
            self.mbc_image_stack_a.set_editable(true);
            self.mbc_image_stack_b.set_editable(true);
        }
        // Wait until the files have been moved to the dataset directory before opening
        // them.
        self.mbc_open_dataset.set_editable(dataset_directory_set);

        // When dataset build is first started, the dataset directory has not been set.
        // Once the dataset directory is set, the log files can be opened.
        if !dataset_directory_set
            && row_state.get_dataset_state() == Some(BatchRunTomoDatasetState::Running)
        {
            self.mbc_proj_log.set_editable(false);
            self.mbc_brt_log.set_editable(false);
        } else {
            self.mbc_proj_log.set_editable(dataset_directory_set);
            self.mbc_brt_log.set_editable(dataset_directory_set);
        }

        // Based more detail from curReconStep
        self.mbc_tomogram.set_editable(
            (cur_recon_step == Some(StatusRef::Step(Step::RECONSTRUCTION))
                && (!self.cbc_dual.is_selected() || self.tomogram_done.get()))
                || cur_recon_step == Some(StatusRef::ProcessName(ProcessName::VOLCOMBINE))
                || cur_recon_step == Some(StatusRef::ProcessName(ProcessName::TRIMVOL)),
        );
        // Locking
        // During a run, lock buttons that open files for writing. And lock process data.
        let active = row_state.is_active();
        drop(row_state);
        self.cbc_dual.set_locked(active);
        self.mbc_open_dataset.set_locked(active);
        self.mbc_tomogram.set_locked(active);
        self.cbc_run.set_locked(active);
        self.cbc_boundary_model.set_locked(active);
        self.cbc_montage.set_locked(active);
        self.fc_skip.set_locked(active);
        self.fc_bskip.set_locked(active);
        self.cbc_surfaces_to_analyze2.set_locked(active);
        self.bc_edit_dataset.set_locked(active);
    }

    /// Java private `statusChanged(Status, String, String, AxisID, boolean)`.
    fn status_changed_full(
        &self,
        status: Option<StatusRef>,
        event_stack_id: Option<&str>,
        file_string: Option<&str>,
        event_axis_id: Option<AxisID>,
        init: bool,
    ) {
        // Avoid processing events for other rows.
        if let (Some(stack_id), Some(event_stack_id)) = (&self.stack_id, event_stack_id)
            && stack_id != event_stack_id
        {
            return;
        }

        self.row_state
            .borrow_mut()
            .handle_status_event(status, event_axis_id, init);

        // Respond to BatchRunTomoStatus if it was accepted by rowState.
        if let Some(StatusRef::BatchRunTomoStatus(_)) = status
            && self.row_state.borrow().equals_batch_run_tomo_status(status)
        {
            let dataset_dialog = self.dataset_dialog.borrow().clone();
            if let Some(dataset_dialog) = dataset_dialog {
                dataset_dialog
                    .status_changed_status(self.status.get().map(StatusRef::BatchRunTomoStatus));
            }
            self.update_display(None, None);
            return;
        }

        // Only BatchRunTomoStatus can be processed without the event stackID.
        if event_stack_id.is_none() {
            return;
        }

        // Respond to BatchRunTomoDatasetState if it was accepted by rowState.
        if let Some(StatusRef::BatchRunTomoDatasetState(dataset_state)) = status {
            if self
                .row_state
                .borrow()
                .equals_batch_run_tomo_dataset_state(status)
            {
                if dataset_state == BatchRunTomoDatasetState::Done {
                    self.cbc_run.set_selected_boolean(false);
                }
                self.update_display(None, None);
                return;
            } else if dataset_state == BatchRunTomoDatasetState::Running {
                self.tomogram_done.set(false);
                self.trimvol_done.set(false);
            }
        }

        if let Some(StatusRef::BatchRunTomoDatasetStatus(dataset_status)) = status {
            let current_file = self.fc_stack.get_expanded_value().unwrap_or_default();
            let mut new_file_abs_path = file_string.map(str::to_owned);
            if self.orig_stack.borrow().is_none() {
                *self.orig_stack.borrow_mut() = Some(current_file.clone());
            }
            if dataset_status == BatchRunTomoDatasetStatus::Delivered {
                // Ignore a delivery that differs only by the axis letter.
                let cur_file = PathBuf::from(&current_file);
                let new_file = new_file_abs_path.as_ref().map(PathBuf::from);
                if !self.cbc_dual.is_selected()
                    || !Extension::equals_files(Some(&cur_file), new_file.as_deref())
                    || !utilities::equals_dataset(true, Some(&cur_file), new_file.as_deref())
                {
                    // New location
                    self.set_log(new_file.as_deref());
                    if let Some(table) = self.table() {
                        table.get_table_reference().change_file_path(
                            self.stack_id.as_deref(),
                            new_file_abs_path.as_deref(),
                        );
                        self.fc_stack.set_value_string(new_file_abs_path.as_deref());
                        eprintln!(
                            "Delivered to {}",
                            new_file_abs_path.as_deref().unwrap_or("null")
                        );
                    }
                }
            } else if dataset_status == BatchRunTomoDatasetStatus::Renamed {
                // Switching only the file name.
                new_file_abs_path = Some(utilities::java_io_file_get_absolute_path(
                    &java_io_file_new_file(
                        Path::new(&current_file).parent(),
                        new_file_abs_path.as_deref().unwrap_or("null"),
                    )
                    .to_string_lossy(),
                ));
                if let Some(table) = self.table() {
                    table
                        .get_table_reference()
                        .change_file_path(self.stack_id.as_deref(), new_file_abs_path.as_deref());
                }
                self.fc_stack.set_value_string(new_file_abs_path.as_deref());
                eprintln!(
                    "Renamed to {}",
                    new_file_abs_path.as_deref().unwrap_or("null")
                );
            }
            return;
        }

        if let Some(StatusRef::EndingStep(_)) = status
            && self.row_state.borrow().equals_ending_step(status)
        {
            self.update_display(None, None);
            return;
        }

        if let Some(StatusRef::Step(step)) = status
            && self.row_state.borrow().equals_step(status)
        {
            if step == Step::RECONSTRUCTION && !self.cbc_dual.is_selected() {
                self.mbc_tomogram.set_editable(true);
            }
            self.update_display(None, None);
            return;
        }

        if let Some(StatusRef::ProcessName(process_name)) = status {
            if process_name == ProcessName::VOLCOMBINE || process_name == ProcessName::TRIMVOL {
                self.mbc_tomogram.set_editable(true);
                if process_name == ProcessName::VOLCOMBINE {
                    // Volcombine is done
                    if self.is_dual() {
                        self.tomogram_done.set(true);
                    }
                } else if process_name == ProcessName::TRIMVOL {
                    // Trimvol is done
                    self.trimvol_done.set(true);
                }
                self.fc_cur_axis_letter.set_value_void();
            }
            self.update_display(None, None);
        }
    }

    /// Java package-private `setOrigStack(File)`.
    pub fn set_orig_stack(&self, orig_stack: Option<&Path>) {
        let Some(orig_stack) = orig_stack else {
            eprintln!("WARNING:  OrigStack is null.");
            return;
        };
        *self.orig_stack.borrow_mut() = Some(utilities::java_io_file_get_absolute_path(
            &orig_stack.to_string_lossy(),
        ));
    }

    /// Java private `setCurAxisID(AxisID)`.
    fn set_cur_axis_id(&self, axis_id: Option<AxisID>) {
        if self.cbc_dual.is_selected() {
            if axis_id.is_some() && axis_id != Some(AxisID::Only) {
                self.cur_axis_id.set(axis_id);
            }
        } else if axis_id == Some(AxisID::Only) {
            self.cur_axis_id.set(axis_id);
        }
    }

    /// Java private `setCurAxisLetter(AxisID)`.
    fn set_cur_axis_letter(&self, axis_id: Option<AxisID>) {
        let Some(axis_id) = axis_id else {
            return;
        };
        self.fc_cur_axis_letter
            .set_value_string(Some(&axis_id.get_capitalized_key_string()));
    }

    /// Java public `statusChanged(StatusChangeEvent, boolean)`.
    pub fn status_changed_event_init(&self, event: &dyn StatusChangeEvent, init: bool) {
        if let Some(row_event) = event.as_any().downcast_ref::<StatusChangeRowEvent>() {
            self.status_changed_full(
                event.get_status(),
                row_event.get_stack_id(),
                row_event.get_file_string(),
                row_event.get_cur_axis_id(),
                init,
            );
        } else {
            self.status_changed_full(event.get_status(), None, None, None, init);
        }
        // Events which do not contain this stackID have no effect.
    }

    /// Java private `setLog(File)`.
    fn set_log(&self, stack: Option<&Path>) {
        if let Some(stack) = stack {
            // The dataset log does not contain the dataset name.
            *self.log.borrow_mut() = file_type::CLASS.batch_run_tomo_dataset_log.get_file_in_dir(
                Some(self.base_manager()),
                stack.parent(),
                None,
                Some(self.get_axis_type()),
                None,
            );
        }
    }

    /// Java private `getAxisType()`.
    fn get_axis_type(&self) -> AxisType {
        if self.cbc_dual.is_selected() {
            return AxisType::DualAxis;
        }
        AxisType::SingleAxis
    }

    /// Java private `sendStatusChange(StatusChangeEvent)`.
    #[allow(dead_code)]
    fn send_status_change_event(&self, status_change_event: &dyn StatusChangeEvent) {
        let listeners = self.listeners.borrow().clone();
        if let Some(listeners) = listeners {
            for listener in &listeners {
                listener.status_changed_event(Some(status_change_event));
            }
        }
    }

    /// Java private `sendStatusChange(Status)`.
    fn send_status_change_status(&self, status: Option<StatusRef>) {
        let listeners = self.listeners.borrow().clone();
        if let Some(listeners) = listeners {
            for listener in &listeners {
                listener.status_changed_status(status);
            }
        }
    }

    /// Java package-private `display(Viewport, BatchRunTomoTab)`.
    pub fn display(&self, viewport: &Viewport, tab: Option<BatchRunTomoTab>) {
        let Some(table) = self.table() else {
            return;
        };
        // See if index is in the viewport
        if viewport.in_viewport(self.hc_number.get_int() - 1) {
            let panel = table.get_table_panel();
            let add = |cell: &dyn CellVirtual, component: Rc<JComponent>| {
                cell.add(&panel);
                table
                    .get_grid_bag_layout()
                    .set_constraints(&component, &table.get_grid_bag_constraints());
            };
            table.with_constraints(|constraints| constraints.gridwidth = 1);
            add(&*self.hc_number, self.hc_number.get_component());
            if tab == Some(BatchRunTomoTab::Stacks) {
                table.with_constraints(|constraints| {
                    self.hb_row
                        .add(&panel, table.get_grid_bag_layout(), constraints)
                });
            }
            table.with_constraints(|constraints| constraints.gridwidth = 2);
            add(
                &*self.fc_stack,
                InputCellVirtual::get_component(&*self.fc_stack),
            );
            table.with_constraints(|constraints| constraints.gridwidth = 1);
            if tab == Some(BatchRunTomoTab::Stacks) {
                add(
                    &*self.cbc_dual,
                    InputCellVirtual::get_component(&*self.cbc_dual),
                );
                add(
                    &*self.cbc_montage,
                    InputCellVirtual::get_component(&*self.cbc_montage),
                );
                add(
                    &*self.fc_skip,
                    InputCellVirtual::get_component(&*self.fc_skip),
                );
                add(
                    &*self.fc_bskip,
                    InputCellVirtual::get_component(&*self.fc_bskip),
                );
                add(
                    &*self.cbc_boundary_model,
                    InputCellVirtual::get_component(&*self.cbc_boundary_model),
                );
                add(
                    &*self.cbc_surfaces_to_analyze2,
                    InputCellVirtual::get_component(&*self.cbc_surfaces_to_analyze2),
                );
                add(
                    &*self.mbc_image_stack_a,
                    InputCellVirtual::get_component(&*self.mbc_image_stack_a),
                );
                table.with_constraints(|constraints| constraints.gridwidth = GRID_BAG_REMAINDER);
                add(
                    &*self.mbc_image_stack_b,
                    InputCellVirtual::get_component(&*self.mbc_image_stack_b),
                );
            } else if tab == Some(BatchRunTomoTab::Dataset) {
                add(
                    &*self.bc_edit_dataset,
                    InputCellVirtual::get_component(&*self.bc_edit_dataset),
                );
                table.with_constraints(|constraints| constraints.gridwidth = GRID_BAG_REMAINDER);
                add(
                    &*self.fc_edit_dataset,
                    InputCellVirtual::get_component(&*self.fc_edit_dataset),
                );
            } else {
                // Swing layout: constraints.ipadx = 1.
                add(
                    &*self.fc_dataset_state,
                    InputCellVirtual::get_component(&*self.fc_dataset_state),
                );
                add(
                    &*self.fc_ending_step,
                    InputCellVirtual::get_component(&*self.fc_ending_step),
                );
                add(
                    &*self.fc_cur_axis_letter,
                    InputCellVirtual::get_component(&*self.fc_cur_axis_letter),
                );
                // Swing layout: constraints.ipadx = 0.
                add(
                    &*self.cbc_run,
                    InputCellVirtual::get_component(&*self.cbc_run),
                );
                add(
                    &*self.mbc_open_dataset,
                    InputCellVirtual::get_component(&*self.mbc_open_dataset),
                );
                add(
                    &*self.mbc_tomogram,
                    InputCellVirtual::get_component(&*self.mbc_tomogram),
                );
                add(
                    &*self.mbc_proj_log,
                    InputCellVirtual::get_component(&*self.mbc_proj_log),
                );
                table.with_constraints(|constraints| constraints.gridwidth = GRID_BAG_REMAINDER);
                add(
                    &*self.mbc_brt_log,
                    InputCellVirtual::get_component(&*self.mbc_brt_log),
                );
            }
        }
    }

    /// Java package-private `expandStack(boolean)`.
    pub fn expand_stack(&self, expanded: bool) {
        self.fc_stack.expand(expanded);
    }

    /// Java package-private `setError(boolean)`.
    pub fn set_error(&self, error: bool) {
        self.fc_stack.set_error_boolean(error);
        self.cbc_boundary_model.set_error_boolean(error);
        self.cbc_dual.set_error_boolean(error);
        self.cbc_montage.set_error_boolean(error);
        self.fc_skip.set_error_boolean(error);
        self.fc_bskip.set_error_boolean(error);
        self.cbc_surfaces_to_analyze2.set_error_boolean(error);
        self.fc_edit_dataset.set_error_boolean(error);
        self.fc_dataset_state.set_error_boolean(error);
        self.fc_ending_step.set_error_boolean(error);
        self.fc_cur_axis_letter.set_error_boolean(error);
        self.cbc_run.set_error_boolean(error);
    }

    /// Java package-private `equalsStackID(String)`.  Java `this.stackID.equals(...)`
    /// dereferences the row's own stack ID, which a table row always has.
    pub fn equals_stack_id(&self, stack_id: Option<&str>) -> bool {
        self.stack_id
            .as_deref()
            .is_some_and(|own| Some(own) == stack_id)
    }

    /// Java private `setParameters(BatchRunTomoRowMetaData, boolean, boolean, boolean)`.
    fn set_parameters_row_meta_data(
        &self,
        row_meta_data: &Arc<BatchRunTomoRowMetaData>,
        only_dataset_dialog: bool,
        series_watcher: bool,
        init: bool,
    ) {
        let is_dataset_dialog = row_meta_data.is_dataset_dialog();
        if only_dataset_dialog && !series_watcher {
            return;
        }
        *self.meta_data.borrow_mut() = Some(row_meta_data.clone());
        row_meta_data.get_row_properties(|properties| {
            self.row_state.borrow_mut().copy_from(Some(properties))
        });
        row_meta_data.get_dual_check_box(|props| {
            FieldPropertiesAdapter::apply_to_field(
                Some(props),
                Some(&*self.cbc_dual as &dyn FieldSettings),
            )
        });
        row_meta_data.get_run_check_box(|props| {
            FieldPropertiesAdapter::apply_to_field(
                Some(props),
                Some(&*self.cbc_run as &dyn FieldSettings),
            )
        });
        row_meta_data.get_open_dataset_button(|props| {
            FieldPropertiesAdapter::apply_to_field(
                Some(props),
                Some(&*self.mbc_open_dataset as &dyn FieldSettings),
            )
        });
        self.fc_bskip
            .set_value_string_boolean(Some(&row_meta_data.get_bskip()), false);

        self.bc_edit_dataset.set_selected(is_dataset_dialog);
        if is_dataset_dialog {
            self.fc_edit_dataset
                .set_value_string(Some(EDIT_DATASET_VALUE));
        } else {
            self.fc_edit_dataset.set_value_void();
        }
        *self.orig_stack.borrow_mut() = Some(row_meta_data.get_orig_stack());
        if init {
            row_meta_data.get_tomogram_button(|props| {
                FieldPropertiesAdapter::apply_to_field(
                    Some(props),
                    Some(&*self.mbc_tomogram as &dyn FieldSettings),
                )
            });
        }
        row_meta_data.get_proj_log_button(|props| {
            FieldPropertiesAdapter::apply_to_field(
                Some(props),
                Some(&*self.mbc_proj_log as &dyn FieldSettings),
            )
        });
        row_meta_data.get_brt_log_button(|props| {
            FieldPropertiesAdapter::apply_to_field(
                Some(props),
                Some(&*self.mbc_brt_log as &dyn FieldSettings),
            )
        });
        self.mbc_image_stack_a
            .set_editable(row_meta_data.is_image_stack_a_editable());
        self.mbc_image_stack_b
            .set_editable(row_meta_data.is_image_stack_b_editable());
        self.tomogram_done.set(row_meta_data.is_tomogram_done());
        self.trimvol_done.set(row_meta_data.is_trimvol_done());
        self.run_status.set(row_meta_data.get_run_status());
        if is_dataset_dialog {
            let dual = self.cbc_dual.is_selected();
            let dataset_dialog = BatchRunTomoDatasetDialog::get_saved_row_instance(
                self.manager,
                dataset_tool::get_dataset_file(
                    dataset_tool::get_stack_file(
                        self.fc_stack.get_expanded_value().as_deref(),
                        Some(AxisID::First),
                        dual,
                    )
                    .as_deref(),
                    dual,
                ),
                self.this.clone(),
                self.table().map(|table| table.get_template_values()),
                self.basic_directives.clone(),
                self.table().map(|table| table.get_browsing_directory()),
                self.stack_id.as_deref(),
            );
            *self.dataset_dialog.borrow_mut() = Some(dataset_dialog);
        }
        if self.fc_cur_axis_letter.is_empty() {
            self.fc_ending_step
                .set_value_string(Some(&row_meta_data.get_cur_ending_step()));
            self.set_cur_axis_id(AxisID::get_instance_ignore_case(Some(
                &row_meta_data.get_cur_axis_letter(),
            )));
            let cur_axis_id = self.cur_axis_id.get();
            if cur_axis_id.is_some() && cur_axis_id != Some(AxisID::Only) {
                self.set_cur_axis_letter(cur_axis_id);
            }
            let temp_ending_step = row_meta_data.get_ending_step_a();
            if temp_ending_step.is_some() {
                self.cur_ending_step_a.set(temp_ending_step);
            }
            let temp_ending_step = row_meta_data.get_ending_step();
            if temp_ending_step.is_some() {
                self.cur_ending_step.set(temp_ending_step);
            }
        }
    }

    /// Java public `setParameters(BatchRunTomoMetaData, boolean, boolean)`.
    pub fn set_parameters_meta_data(
        &self,
        meta_data: &BatchRunTomoMetaData,
        only_dataset_dialog: bool,
        init: bool,
    ) {
        let Some(stack_id) = self.stack_id.as_deref() else {
            return;
        };
        let row_meta_data = meta_data.get_row_meta_data(stack_id);
        let is_dataset_dialog = row_meta_data.is_dataset_dialog();
        let dataset_dialog = self.dataset_dialog.borrow().clone();
        if !is_dataset_dialog && only_dataset_dialog {
            // For a new row-level dataset dialog initialize with data from the main
            // dataset dialog.
            if let Some(dataset_dialog) = &dataset_dialog {
                dataset_dialog.set_parameters_dataset_meta_data(&meta_data.get_dataset_meta_data());
            }
        }
        if (is_dataset_dialog || only_dataset_dialog)
            && let Some(dataset_dialog) = &dataset_dialog
        {
            dataset_dialog.set_parameters_dataset_meta_data(&row_meta_data.get_dataset_meta_data());
        }
        self.set_parameters_row_meta_data(&row_meta_data, only_dataset_dialog, false, init);
        if !only_dataset_dialog {
            // Set off internal status changed events.
            self.status_changed_status(meta_data.get_status().map(StatusRef::BatchRunTomoStatus));
            let meta_data_dataset_state = row_meta_data.get_dataset_state();
            if let Some(meta_data_dataset_state) = meta_data_dataset_state {
                let event = StatusChangeRowEvent::new(
                    Some(stack_id),
                    None,
                    Some(StatusRef::BatchRunTomoDatasetState(meta_data_dataset_state)),
                );
                self.status_changed_event_init(&event, true);
            }
        }
        self.update_display(None, None);
    }

    /// Java public `setParameters(SeriesWatcherMetaData, String, boolean)`.
    pub fn set_parameters_series_watcher_meta_data(
        &self,
        meta_data: &SeriesWatcherMetaData,
        stack_absolute_path: Option<&str>,
        init: bool,
    ) {
        if let Some(stack_absolute_path) = stack_absolute_path {
            self.fc_stack
                .set_value_file(Some(Path::new(stack_absolute_path)));
        } else if let Some(orig_stack) = self.orig_stack.borrow().clone() {
            self.fc_stack.set_value_file(Some(Path::new(&orig_stack)));
        }
        let Some(stack_id) = self.stack_id.as_deref() else {
            return;
        };
        let row_meta_data = meta_data.get_row_meta_data(stack_id);
        self.set_parameters_row_meta_data(&row_meta_data, false, true, init);
        // Set off internal status changed events.
        let dataset_state = row_meta_data.get_dataset_state();
        if let Some(dataset_state) = dataset_state {
            let event = StatusChangeRowEvent::new(
                Some(stack_id),
                None,
                Some(StatusRef::BatchRunTomoDatasetState(dataset_state)),
            );
            self.status_changed_event_init(&event, true);
        }
        self.update_display(None, None);
    }

    /// Java package-private `getParameters(BatchRunTomoRowMetaData)`.
    pub fn get_parameters_row_meta_data(&self, row_meta_data: &Arc<BatchRunTomoRowMetaData>) {
        *self.meta_data.borrow_mut() = Some(row_meta_data.clone());
        row_meta_data
            .get_row_properties(|properties| self.row_state.borrow().copy_to(Some(properties)));
        row_meta_data.set_row_number(self.hc_number.get_text().as_deref());
        row_meta_data.get_dual_check_box(|props| {
            FieldPropertiesAdapter::apply_to_props(
                Some(&*self.cbc_dual as &dyn FieldSettings),
                Some(props),
            )
        });
        row_meta_data.set_bskip(self.fc_bskip.get_value().as_deref());
        row_meta_data.get_run_check_box(|props| {
            FieldPropertiesAdapter::apply_to_props(
                Some(&*self.cbc_run as &dyn FieldSettings),
                Some(props),
            )
        });
        row_meta_data.set_bskip(self.fc_bskip.get_value().as_deref());
        row_meta_data.get_open_dataset_button(|props| {
            FieldPropertiesAdapter::apply_to_props(
                Some(&*self.mbc_open_dataset as &dyn FieldSettings),
                Some(props),
            )
        });
        let dataset_dialog = self.dataset_dialog.borrow().clone();
        row_meta_data.set_dataset_dialog(dataset_dialog.is_some());
        row_meta_data.set_orig_stack(self.orig_stack.borrow().as_deref());
        row_meta_data.get_tomogram_button(|props| {
            FieldPropertiesAdapter::apply_to_props(
                Some(&*self.mbc_tomogram as &dyn FieldSettings),
                Some(props),
            )
        });
        row_meta_data.get_proj_log_button(|props| {
            FieldPropertiesAdapter::apply_to_props(
                Some(&*self.mbc_proj_log as &dyn FieldSettings),
                Some(props),
            )
        });
        row_meta_data.get_brt_log_button(|props| {
            FieldPropertiesAdapter::apply_to_props(
                Some(&*self.mbc_brt_log as &dyn FieldSettings),
                Some(props),
            )
        });
        row_meta_data.set_image_stack_a_editable(self.mbc_image_stack_a.is_editable());
        row_meta_data.set_image_stack_b_editable(self.mbc_image_stack_b.is_editable());
        row_meta_data.set_tomogram_done(self.tomogram_done.get());
        row_meta_data.set_trimvol_done(self.trimvol_done.get());
        if let Some(dataset_dialog) = &dataset_dialog {
            dataset_dialog.get_parameters_dataset_meta_data(&row_meta_data.get_dataset_meta_data());
        }
        row_meta_data.set_dataset_state(self.dataset_state.get());
        row_meta_data.set_ending_step_a(self.cur_ending_step_a.get());
        row_meta_data.set_ending_step(self.cur_ending_step.get());
        row_meta_data.set_cur_ending_step(self.fc_ending_step.get_value().as_deref());
        row_meta_data.set_cur_axis_letter(self.fc_cur_axis_letter.get_value().as_deref());
        row_meta_data.set_run_status(self.run_status.get());
    }

    /// Java package-private `getParameters(BatchRunTomoMetaData)`.
    pub fn get_parameters_meta_data(&self, meta_data: &BatchRunTomoMetaData) {
        if let Some(stack_id) = self.stack_id.as_deref() {
            self.get_parameters_row_meta_data(&meta_data.get_row_meta_data(stack_id));
        }
    }

    /// Java package-private `getParameters(SeriesWatcherMetaData)`.
    pub fn get_parameters_series_watcher_meta_data(&self, meta_data: &SeriesWatcherMetaData) {
        if let Some(stack_id) = self.stack_id.as_deref() {
            self.get_parameters_row_meta_data(&meta_data.get_row_meta_data(stack_id));
        }
    }

    /// Java package-private `equals(String)`.
    pub fn equals_string(&self, stack_id: Option<&str>) -> bool {
        stack_id.is_some() && stack_id == self.stack_id.as_deref()
    }

    /// Java package-private `equals(String, String)`.
    pub fn equals_location_root_name(
        &self,
        location: Option<&str>,
        root_name: Option<&str>,
    ) -> bool {
        let (Some(location), Some(root_name)) = (location, root_name) else {
            return false;
        };
        let stack = self.stack_file();
        // test rootName
        if Some(root_name.to_owned())
            != dataset_tool::get_dataset_name(
                stack
                    .file_name()
                    .map(|name| name.to_string_lossy())
                    .as_deref(),
                self.cbc_dual.is_selected(),
            )
        {
            return false;
        }
        // test location
        let f_location = PathBuf::from(location);
        // Java dereferences `metaData`, which is null until the row's parameters have
        // been read; fixed in translation: a row without one uses its stack's
        // directory (BUGS.md).
        let orig_stack = self
            .meta_data
            .borrow()
            .as_ref()
            .map(|meta_data| meta_data.get_orig_stack());
        let rowlocation = match orig_stack {
            Some(orig_stack) if !orig_stack.is_empty() => {
                Path::new(&orig_stack).parent().map(Path::to_path_buf)
            }
            _ => stack.parent().map(Path::to_path_buf),
        };
        Some(f_location) == rowlocation
    }

    /// Java package-private `getParameters(BatchruntomoParam, boolean, boolean, RunType,
    /// StringBuilder, AtomicInteger, boolean, List<UniqueKey>, boolean)`.  Update param
    /// and return true if the row can be run.
    #[allow(clippy::too_many_arguments)]
    pub fn get_parameters_param(
        &self,
        param: &mut BatchruntomoParam,
        deliver_off: bool,
        deliver_to_directory: bool,
        run_type: Option<RunType>,
        err_msg: &mut String,
        num_errors: &mut i32,
        do_validation: bool,
        manager_key_list: Option<&mut Vec<UniqueKey>>,
        validate_only: bool,
    ) -> bool {
        if !self.setup_run(run_type, validate_only) {
            return false;
        }
        let mut delivery_err_msg = String::new();
        if !validate_only {
            // Collect all dataset file error messages.
            let stack = self.stack_file();
            param.add_directive_file(Some(
                &self.get_autodoc_file(self.fc_stack.get_contracted_value().as_deref()),
            ));
            let root_name = dataset_tool::get_dataset_name(
                stack
                    .file_name()
                    .map(|name| name.to_string_lossy())
                    .as_deref(),
                self.cbc_dual.is_selected(),
            );
            param.add_root_name(
                root_name.as_deref(),
                deliver_to_directory,
                self.cbc_dual.is_selected(),
                do_validation,
                Some(&mut delivery_err_msg),
            );
            let mut orig_stack_location = None;
            // Java dereferences `metaData` unguarded; the manager reads the row's
            // parameters (`getParameters(metaData)`) before every run.
            let orig_stack = self
                .meta_data
                .borrow()
                .as_ref()
                .map(|meta_data| meta_data.get_orig_stack());
            if let Some(orig_stack) = orig_stack {
                orig_stack_location = Path::new(&orig_stack)
                    .parent()
                    .map(|parent| parent.to_string_lossy().into_owned());
            }

            if !param.add_current_location(
                orig_stack_location.as_deref(),
                stack
                    .parent()
                    .map(|parent| parent.to_string_lossy())
                    .as_deref(),
                deliver_off,
                do_validation,
                Some(&mut delivery_err_msg),
            ) {
                delivery_err_msg.push_str(&format!(
                    ": {}.  ",
                    utilities::java_io_file_get_absolute_path(&stack.to_string_lossy())
                ));
            }
        }
        // If doValidation is true, will be run
        if do_validation {
            let mut manager_key_list = manager_key_list;
            for i in 0..batch_run_tomo_step_panel::STEP_PAIRS {
                self.first_steps.borrow_mut()[i] = false;
                if let Some(manager_key_list) = manager_key_list.as_deref_mut()
                    && etomo_director::INSTANCE.is_open(self.manager_key.borrow().as_ref())
                    && let Some(manager_key) = self.manager_key.borrow().clone()
                {
                    manager_key_list.push(manager_key);
                }
            }
        }
        if !validate_only {
            // Save error message. Only save the first error found in the row list.
            if !delivery_err_msg.is_empty() && {
                *num_errors += 1;
                *num_errors == 1
            } {
                if delivery_err_msg.trim().starts_with(&format!(
                    "{}{}",
                    process_output_strings::BRT_DATASET_DIR_NOT_UNIQUE_ERR1,
                    process_output_strings::BRT_DATASET_DIR_NOT_UNIQUE_ERR2
                )) {
                    err_msg.push_str(&format!(
                        "{}s{}.\nChange the option for moving the stack in the {} tab.\n",
                        process_output_strings::BRT_DATASET_DIR_NOT_UNIQUE_ERR1,
                        process_output_strings::BRT_DATASET_DIR_NOT_UNIQUE_ERR2,
                        BatchRunTomoTab::Batch.get_quoted_label()
                    ));
                } else {
                    err_msg.push_str(&format!("{}\n", delivery_err_msg));
                }
            }
        }
        true
    }

    /// Java package-private `setupRun(RunType, boolean)`.  Returns true if row can be
    /// run.  For a new run or resume, this is based on the run checkbox.  For a
    /// reconnect it's based on runStatus.  In this case it is just part of the group of
    /// rows that are being run, and may not be run itself.  RunStatus will be set if it
    /// is a new run.  This function can be run multiple times before the run happens.
    ///
    /// If runType is resume, then killed rows need to be run because they make have been
    /// killed during the pause or kill.
    pub fn setup_run(&self, run_type: Option<RunType>, validate_only: bool) -> bool {
        let run = self.is_run();
        let mut temp_run_status = self.run_status.get();
        let mut retval: Option<bool> = None;
        if run_type == Some(RunType::Run) {
            // For a new run, change every row to TO_RUN or null depending on the
            // checkbox. The null rows are not part of this run.
            temp_run_status = if run { Some(RunStatus::ToRun) } else { None };
            if run {
                // Clear the dataset state for everything that's part of the run to avoid
                // confusion about what's been run already.
                if let Some(table) = self.table() {
                    table.msg_row_running(false);
                }
            }
            retval = Some(run);
        } else {
            if run_type == Some(RunType::Resume) || run_type == Some(RunType::ResumeProcessChunks) {
                if run {
                    // If the run type is RESUME, then the user may change the run
                    // checkboxes. They may add a new row to the resume by checking the
                    // run checkbox. The run checkboxes are not enabled for processchunks.
                    //
                    // Turn the killed row back into a to_run row so it will run after a
                    // resume.
                    if temp_run_status.is_none() || temp_run_status == Some(RunStatus::Killed) {
                        temp_run_status = Some(RunStatus::ToRun);
                    }
                    // RESUME: Don't resume anything that failed or is already finished
                    if run_type == Some(RunType::Resume) {
                        retval = Some(temp_run_status == Some(RunStatus::ToRun));
                    } else {
                        // RESUME_PROCESS_CHUNKS: Include all of the original run rows to
                        // match the chunks.
                        retval = Some(temp_run_status.is_some());
                    }
                } else {
                    // These rows have been removed from the run, or were never in it.
                    // They are not in the brt .com file.
                    temp_run_status = None;
                    retval = Some(false);
                }
            }
            if retval.is_none() {
                // runType == RECONNECT
                // With reconnect the user can't modify what runs. This function can't
                // know what's already been run. It doesn't have to know because the
                // monitor runs through the whole log and uses the last known line number
                // to start monitoring actively.
                retval = Some(temp_run_status.is_some());
            }
        }
        if !validate_only {
            self.run_status.set(temp_run_status);
        }
        self.update_display(run_type, Some(validate_only));
        retval.unwrap_or(false)
    }

    /// Java private `getAutodocFile(String)`.
    fn get_autodoc_file(&self, file_name: Option<&str>) -> PathBuf {
        let orig_stack = self.orig_stack.borrow().clone();
        let parent = match orig_stack {
            Some(orig_stack) if !orig_stack.is_empty() => Path::new(&orig_stack)
                .parent()
                .map(|parent| parent.to_string_lossy().into_owned()),
            _ => self
                .stack_file()
                .parent()
                .map(|parent| parent.to_string_lossy().into_owned()),
        };
        java_io_file_new_string_parent(
            parent.as_deref(),
            &format!(
                "{}_{}.adoc",
                self.manager.get_name().unwrap_or_else(|| "null".to_owned()),
                dataset_tool::get_dataset_name(file_name, self.cbc_dual.is_selected())
                    .unwrap_or_else(|| "null".to_owned())
            ),
        )
    }

    /// Java private `getProjectLog(BaseManager)`.  Get the project log for the dataset.
    /// Return null if the project log does not exist.
    fn get_project_log(&self, manager: &'static dyn BaseManager) -> Option<PathBuf> {
        let stack = self.stack_file();
        let dual = self.cbc_dual.is_selected();
        let root_name = dataset_tool::get_dataset_name(
            stack
                .file_name()
                .map(|name| name.to_string_lossy())
                .as_deref(),
            dual,
        );
        let location = stack.parent()?.to_string_lossy().into_owned();
        let project_log = file_type::CLASS
            .project_log
            .get_file_with_property_user_dir(
                Some(manager),
                root_name.as_deref(),
                Some(if dual {
                    AxisType::DualAxis
                } else {
                    AxisType::SingleAxis
                }),
                Some(AxisID::First),
                Some(&location),
            )?;
        if !project_log.exists() {
            return None;
        }
        Some(project_log)
    }

    /// Java package-private `loadAutodoc(boolean, boolean)`.
    pub fn load_autodoc(&self, only_load_dataset: bool, only_advanced_dataset_dialog: bool) {
        let directive_file = DirectiveFile::get_instance(
            self.base_manager(),
            None,
            Some(&self.get_autodoc_file(self.fc_stack.get_contracted_value().as_deref())),
            DirectiveFileType::Batch,
        );
        // Java dereferences the null instance of a missing file; fixed in translation:
        // there is nothing to load (BUGS.md).
        let Some(directive_file) = directive_file else {
            return;
        };
        if !only_load_dataset {
            self.set_values_directive_files(&directive_file);
            batch_tool::set_text_value(Some(&*self.fc_skip), &directive_file, false, None);
            batch_tool::set_text_value_axis(
                Some(&*self.fc_bskip),
                &directive_file,
                false,
                None,
                Some(AxisID::Second),
            );
            if directive_file
                .contains_value(Some(DirectiveDef::RAW_BOUNDARY_MODEL_FOR_SEED_FINDING))
                || directive_file
                    .contains_value(Some(DirectiveDef::RAW_BOUNDARY_MODEL_FOR_PATCH_TRACKING))
            {
                batch_tool::set_boolean_value_selected(
                    Some(&*self.cbc_boundary_model),
                    true,
                    false,
                );
            }
        }
        let dataset_dialog = self.dataset_dialog.borrow().clone();
        if let Some(dataset_dialog) = dataset_dialog {
            dataset_dialog.set_values(&directive_file, false, only_advanced_dataset_dialog, true);
        }
    }

    /// Java package-private `saveAutodoc(TemplatePanel, NameValuePairList,
    /// NameValuePairList, boolean, boolean, File, FieldDisplayer, boolean)`.
    #[allow(clippy::too_many_arguments)]
    pub fn save_autodoc(
        &self,
        template_panel: Option<&TemplatePanel>,
        global_batch_list: Option<&NameValuePairList>,
        templates: Option<&NameValuePairList>,
        do_validation: bool,
        init: bool,
        deliver_to_directory: Option<&Path>,
        field_displayer: &dyn FieldDisplayer,
        validate_only: bool,
    ) -> bool {
        let stack = self.stack_file();
        let batch_file = self.get_autodoc_file(self.fc_stack.get_contracted_value().as_deref());
        // If the advanced dialog was never created, then load the save file first as not
        // all of it was loaded into the dialog.
        let dataset_dialog = self.dataset_dialog.borrow().clone();
        let advanced_dialog_exists = dataset_dialog
            .as_ref()
            .is_some_and(|dataset_dialog| dataset_dialog.is_advanced_dialog_exists());
        let mut loaded_batch_list: Option<NameValuePairList> = None;
        // try
        let result: Result<bool, SaveError> = (|| {
            if !validate_only && batch_file.exists() {
                if init && !advanced_dialog_exists {
                    // Include the local batch list in the save if it happens while loading
                    // a dataset where the advanced dialog is already in use.
                    let autodoc = unsafe {
                        autodoc_factory::get_autodoc_instance(
                            Some(self.base_manager()),
                            Some(&batch_file),
                        )
                    }?;
                    loaded_batch_list = Some(unsafe { NameValuePairList::new_autodoc(autodoc) });
                }
                // `Utilities.deleteFileOrDirectory(batchFile, manager, null)`.
                utilities::delete_file_or_directory(&batch_file, Some(self.base_manager()), None);
            }
            let batch_autodoc = unsafe {
                autodoc_factory::get_writable_autodoc_instance(
                    Some(self.base_manager()),
                    Some(&batch_file),
                )
            }?;

            let stack_name = stack
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_default();
            let extension = Extension::get_instance(&stack_name);
            if let Some(extension) = extension {
                // Save setupset.currentStackExt or currentBStackExt.
                batch_tool::save_text_to_autodoc(
                    true,
                    self.fc_stack.get_directive_def(),
                    Some(&extension.to_string()),
                    None,
                    batch_autodoc,
                    None,
                    validate_only,
                )?;
                if extension.is_input_image_file() {
                    batch_tool::save_text_to_autodoc(
                        true,
                        Some(DirectiveDef::STACK_EXT),
                        Some(&extension.to_string()),
                        None,
                        batch_autodoc,
                        None,
                        validate_only,
                    )?;
                }
            } else if do_validation {
                ui_harness::with(|harness| {
                    harness.open_message_dialog_base_manager_ui_component_string_string_field_displayer(
                        Some(self.base_manager()),
                        Some(&*self.cbc_boundary_model as &dyn UIComponent),
                        &format!(
                            "Row# {}:  Missing or invalid file extension - {}",
                            self.hc_number.get_text().unwrap_or_else(|| "null".to_owned()),
                            stack_name
                        ),
                        "Invalid Extension",
                        Some(field_displayer),
                    )
                });
                return Ok(false);
            }
            batch_tool::save_boolean_to_autodoc(
                Some(&*self.cbc_dual),
                batch_autodoc,
                None,
                validate_only,
            )?;
            batch_tool::save_boolean_to_autodoc(
                Some(&*self.cbc_montage),
                batch_autodoc,
                None,
                validate_only,
            )?;
            batch_tool::save_text_field_to_autodoc(
                Some(&*self.fc_skip),
                batch_autodoc,
                do_validation,
                None,
                None,
                validate_only,
            )?;
            batch_tool::save_text_field_to_autodoc(
                Some(&*self.fc_bskip),
                batch_autodoc,
                do_validation,
                None,
                None,
                validate_only,
            )?;
            let mut boundary_model_name = String::new();
            if self.cbc_boundary_model.is_selected() {
                boundary_model_name = batch_tool::get_model_file_name(
                    self.manager,
                    Some(&file_type::CLASS.batch_run_tomo_boundary_model),
                    &stack_name,
                    self.cbc_dual.is_selected(),
                )
                .unwrap_or_else(|| "null".to_owned());
                // Validation: make sure the boundary model file exists
                if do_validation
                    && !java_io_file_new_file(stack.parent(), &boundary_model_name).exists()
                    && deliver_to_directory.is_none_or(|deliver_to_directory| {
                        !deliver_to_directory
                            .join(
                                dataset_tool::get_dataset_name(
                                    Some(&stack_name),
                                    self.cbc_dual.is_selected(),
                                )
                                .unwrap_or_else(|| "null".to_owned()),
                            )
                            .join(&boundary_model_name)
                            .exists()
                    })
                {
                    ui_harness::with(|harness| {
                        harness.open_message_dialog_base_manager_ui_component_string_string_field_displayer(
                            Some(self.base_manager()),
                            Some(&*self.cbc_boundary_model as &dyn UIComponent),
                            &format!(
                                "Row# {}:  Missing boundary model file - {}",
                                self.hc_number.get_text().unwrap_or_else(|| "null".to_owned()),
                                boundary_model_name
                            ),
                            "Missing File",
                            Some(field_displayer),
                        )
                    });
                    return Ok(false);
                }
            }
            // Save the correct boundary model directive.
            let mut tracking_method_seed = false;
            if let Some(dataset_dialog) = &dataset_dialog {
                tracking_method_seed = dataset_dialog.is_tracking_method_seed();
            } else if let Some(table) = self.table() {
                tracking_method_seed = table.is_tracking_method_seed();
            }
            if tracking_method_seed {
                batch_tool::save_boolean_text_to_autodoc_no_template(
                    Some(&*self.cbc_boundary_model),
                    Some(&boundary_model_name),
                    batch_autodoc,
                    validate_only,
                )?;
            } else {
                batch_tool::save_text_to_autodoc(
                    true,
                    Some(DirectiveDef::RAW_BOUNDARY_MODEL_FOR_PATCH_TRACKING),
                    Some(&boundary_model_name),
                    None,
                    batch_autodoc,
                    None,
                    validate_only,
                )?;
            }
            //
            batch_tool::save_boolean_text_to_autodoc_selected(
                Some(&*self.cbc_surfaces_to_analyze2),
                batch_autodoc,
                Some(SURFACES_TO_ANALYZE_TWO),
                Some(SURFACES_TO_ANALYZE_ONE),
                validate_only,
            )?;
            if let Some(template_panel) = template_panel {
                template_panel.save_autodoc(unsafe { &mut *batch_autodoc }, validate_only);
            }
            if let Some(dataset_dialog) = &dataset_dialog {
                if !dataset_dialog.save_autodoc(batch_autodoc, do_validation, validate_only) {
                    return Ok(false);
                }
            } else {
                let global_dataset_dialog = self.table().map(|table| table.get_dataset_dialog());
                if let Some(global_dataset_dialog) = global_dataset_dialog {
                    // The global is validated when the main .adoc file is saved
                    if !global_dataset_dialog.save_autodoc(batch_autodoc, false, validate_only) {
                        return Ok(false);
                    }
                }
            }
            if !validate_only {
                // Create and write a dataset-level batch file.
                let save_batch_list = if advanced_dialog_exists
                    || (loaded_batch_list.is_none() && global_batch_list.is_some())
                {
                    unsafe {
                        batch_tool::create_batch_file(
                            self.base_manager(),
                            batch_autodoc,
                            advanced_dialog_exists,
                            global_batch_list,
                            None,
                            None,
                            None,
                            templates,
                        )
                    }
                } else {
                    // No advanced dialog - take the advanced directives from the files
                    let basic_directives = self
                        .basic_directives
                        .as_ref()
                        .map(|basic_directives| basic_directives.borrow().clone());
                    let advanced_starting_batch = self
                        .table()
                        .and_then(|table| table.get_advanced_starting_batch());
                    unsafe {
                        batch_tool::create_batch_file(
                            self.base_manager(),
                            batch_autodoc,
                            advanced_dialog_exists,
                            None,
                            loaded_batch_list.as_mut(),
                            basic_directives.as_ref(),
                            advanced_starting_batch.as_ref(),
                            templates,
                        )
                    }
                };
                let log_file = unsafe { (*batch_autodoc).get_log_file() };
                save_batch_list.write(log_file.as_ref());
            }
            Ok(true)
        })();
        match result {
            Ok(value) => value,
            // catch (final LogFileException | IOException e)
            Err(SaveError::LogFile(LogFileError::Lock(_))) => false,
            Err(SaveError::LogFile(e)) => {
                eprintln!("{e}");
                true
            }
            // catch (final FieldValidationFailedException e)
            Err(SaveError::FieldValidationFailed(e)) => {
                eprintln!("{e:?}");
                false
            }
        }
    }

    /// Java package-private `isHighlighted()`.
    pub fn is_highlighted(&self) -> bool {
        self.hb_row.is_highlighted()
    }

    /// Java package-private `selectHighlightButton()`.
    pub fn select_highlight_button(&self) {
        self.hb_row.set_selected(true);
    }

    /// Java package-private `backupIfChanged(boolean, boolean)`.  Check
    /// isDifferentFromCheckpoint on all data entry fields that are loaded from directive
    /// files; returns true if any field's isDifferentFromCheckpoint returned true.
    pub fn backup_if_changed(
        &self,
        only_dataset_dialog: bool,
        only_advanced_dataset_dialog: bool,
    ) -> bool {
        let mut changed = false;
        if !only_dataset_dialog {
            if self.cbc_dual.is_different_from_checkpoint(true) {
                self.cbc_dual.backup();
                changed = true;
            }
            if self.cbc_montage.is_different_from_checkpoint(true) {
                self.cbc_montage.backup();
                changed = true;
            }
            if self
                .cbc_surfaces_to_analyze2
                .is_different_from_checkpoint(true)
            {
                self.cbc_surfaces_to_analyze2.backup();
                changed = true;
            }
        }
        let dataset_dialog = self.dataset_dialog.borrow().clone();
        if let Some(dataset_dialog) = dataset_dialog
            && dataset_dialog.backup_if_changed(only_advanced_dataset_dialog)
        {
            changed = true;
        }
        changed
    }

    /// Java package-private `applyValues(boolean, boolean, UserConfiguration,
    /// DirectiveFileCollection, boolean, boolean)`.
    pub fn apply_values(
        &self,
        init: bool,
        retain_user_values: bool,
        user_configuration: &UserConfiguration,
        directive_file_collection: &DirectiveFileCollection,
        only_dataset_dialog: bool,
        only_advanced_dataset_dialog: bool,
    ) {
        if !only_dataset_dialog {
            // to apply values and highlights, start with a clean slate
            if !init {
                self.cbc_dual.clear();
                self.cbc_montage.clear();
                self.cbc_surfaces_to_analyze2.clear();
            }
            // no default values to apply to table
            // Apply settings values
            self.set_values_user_configuration(user_configuration);
            // Apply the directive collection values
            self.set_values_directive_files(directive_file_collection);
            // checkpoint template/starting batch
            self.cbc_dual.checkpoint();
            self.cbc_montage.checkpoint();
            self.cbc_surfaces_to_analyze2.checkpoint();
            // If the user wants to retain their values, apply backed up values and then
            // delete them.
            if retain_user_values {
                self.cbc_dual.restore_from_backup();
                self.cbc_montage.restore_from_backup();
                self.cbc_surfaces_to_analyze2.restore_from_backup();
                self.update_display(None, None);
            }
            // no field highlight values to set in table
        }
        // Dataset dialog
        let dataset_dialog = self.dataset_dialog.borrow().clone();
        if let Some(dataset_dialog) = dataset_dialog {
            dataset_dialog.apply_values(
                init,
                retain_user_values,
                directive_file_collection,
                only_advanced_dataset_dialog,
            );
        }
        if !only_dataset_dialog {
            self.update_display(None, None);
        }
    }

    /// Java package-private `setNumber(int)`.
    pub fn set_number(&self, input: i32) {
        self.hc_number.set_text_int(input);
    }

    /// Java package-private `setValues(DirectiveFileInterface)`.  Set values from the
    /// directive file collection - only for directives that are present.
    pub fn set_values_directive_files(&self, directive_files: &dyn DirectiveFileInterface) {
        batch_tool::set_boolean_value(Some(&*self.cbc_dual), directive_files, false, None);
        batch_tool::set_boolean_value(Some(&*self.cbc_montage), directive_files, false, None);
        batch_tool::set_boolean_value_from_selected_text(
            Some(&*self.cbc_surfaces_to_analyze2),
            Some(SURFACES_TO_ANALYZE_TWO),
            directive_files,
            false,
            None,
        );
    }

    /// Java package-private `setValues(UserConfiguration)`.
    pub fn set_values_user_configuration(&self, user_configuration: &UserConfiguration) {
        // For series watcher, this setting has already been used on the series watcher
        // dual checkbox.
        // Only use this if the Settings checkbox is checked. Otherwise ignore it.
        let series_watcher_on = self
            .dialog()
            .is_some_and(|dialog| dialog.is_series_watcher_on());
        if !series_watcher_on && user_configuration.get_single_axis() {
            self.cbc_dual.set_selected_boolean(false);
        }
        self.cbc_montage
            .set_selected_boolean(user_configuration.get_montage());
        self.update_display(None, None);
    }

    /// Java private `setTooltips(BatchRunTomoRow)`.
    fn set_tooltips(&self, _prev_row: Option<&Rc<BatchRunTomoRow>>) {
        self.cbc_boundary_model
            .set_tool_tip_text(Some(shared_strings::RAW_BOUNDARY_MODEL));
        self.cbc_dual
            .set_tool_tip_text(Some(shared_strings::DUAL_TOOLTIP));
        self.cbc_montage
            .set_tool_tip_text(Some(shared_strings::MONTAGE_TOOLTIP));
        self.fc_skip
            .set_tool_tip_text(Some(shared_strings::SKIP_TOOLTIP));
        self.fc_bskip
            .set_tool_tip_text(Some(shared_strings::BSKIP_TOOLTIP));
        self.cbc_surfaces_to_analyze2
            .set_tool_tip_text(Some(shared_strings::SURFACES_TO_ANALYZE_2_TOOLTIP));
        self.fc_edit_dataset
            .set_tool_tip_text(Some(shared_strings::EDIT_DATASET_TOOLTIP));
        self.fc_dataset_state
            .set_tool_tip_text(Some("Completion status of the dataset"));
        self.cbc_run.set_tool_tip_text(Some(
            "This dataset will be included in the batchruntomo run",
        ));
        self.mbc_open_dataset
            .set_tool_tip_text(Some("Opens a tab in Etomo contain this dataset"));
        self.mbc_tomogram
            .set_tool_tip_text(Some("Opens the current .rec file for this dataset"));
        self.mbc_proj_log
            .set_tool_tip_text(Some("Opens the project log file for this dataset"));
        self.mbc_brt_log
            .set_tool_tip_text(Some("Opens the current log file for this dataset"));
        self.mbc_image_stack_a
            .set_tool_tip_text(Some(shared_strings::IMOD_A_TOOLTIP));
        self.mbc_image_stack_b
            .set_tool_tip_text(Some(shared_strings::IMOD_B_TOOLTIP));
        self.bc_edit_dataset
            .set_tool_tip_text(Some("Open dialog to set dataset-specific values."));
        self.fc_stack
            .set_tool_tip_text(Some(shared_strings::STACK_TOOLTIP));
        self.fc_ending_step
            .set_tool_tip_text(Some(shared_strings::STEP_TOOLTIP));
        self.fc_cur_axis_letter
            .set_tool_tip_text(Some("Current axis being processed"));
    }
}

/// What the Java `saveAutodoc` try block can throw: `LogFileException`/`IOException`/
/// `LockException` and `FieldValidationFailedException`.
enum SaveError {
    LogFile(LogFileError),
    FieldValidationFailed(FieldValidationFailedException),
}

impl From<LogFileError> for SaveError {
    fn from(e: LogFileError) -> SaveError {
        SaveError::LogFile(e)
    }
}

impl From<FieldValidationFailedException> for SaveError {
    fn from(e: FieldValidationFailedException) -> SaveError {
        SaveError::FieldValidationFailed(e)
    }
}

impl Run3dmodButtonContainer for BatchRunTomoRow {
    /// Java `action(String, Deferred3dmodButton, Run3dmodMenuOptions)`.
    fn action(
        &self,
        action_command: &str,
        _deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    ) {
        let action_command = Some(action_command.to_owned());
        if action_command == self.cbc_dual.get_action_command() {
            self.update_display(None, None);
        } else {
            let stack = self.stack_file();
            let dual = self.cbc_dual.is_selected();
            if action_command == self.mbc_image_stack_a.get_action_command() {
                let mut model_file = None;
                if self.cbc_boundary_model.is_selected() {
                    model_file = Some(&*file_type::CLASS.batch_run_tomo_boundary_model);
                }
                self.imod_stack_full(model_file, AxisID::First, dual, run_3dmod_menu_options);
            } else if action_command == self.mbc_image_stack_b.get_action_command() {
                // The model is only opened for the A axis
                self.imod_stack_no_model(AxisID::Second, dual, run_3dmod_menu_options);
            } else if action_command == self.mbc_open_dataset.get_action_command() {
                let manager_key = self.manager_key.borrow().clone();
                if manager_key.is_some() && etomo_director::INSTANCE.is_open(manager_key.as_ref()) {
                    etomo_director::INSTANCE.set_current_manager_unique_key(manager_key.as_ref());
                } else {
                    let key = etomo_director::INSTANCE
                        .open_tomogram_file_boolean_axis_id_ui_component(
                            self.get_dataset_file().as_deref(),
                            true,
                            None,
                            Some(&*self.mbc_open_dataset as &dyn UIComponent),
                        );
                    *self.manager_key.borrow_mut() = key;
                }
            } else if action_command == self.mbc_tomogram.get_action_command() {
                let dataset_name = dataset_tool::get_dataset_name(
                    stack
                        .file_name()
                        .map(|name| name.to_string_lossy())
                        .as_deref(),
                    self.cbc_dual.is_selected(),
                );
                if !self.trimvol_done.get() {
                    self.imod_rec.set(self.manager.imod_rec(
                        stack.parent(),
                        dataset_name.as_deref(),
                        self.get_axis_type(),
                        self.imod_rec.get(),
                        run_3dmod_menu_options,
                    ));
                } else {
                    self.imod_rec.set(self.manager.imod_trimvol(
                        stack.parent(),
                        dataset_name.as_deref(),
                        self.get_axis_type(),
                        self.imod_rec.get(),
                        run_3dmod_menu_options,
                    ));
                }
            } else if action_command == self.mbc_brt_log.get_action_command() {
                let log = self.log.borrow().clone();
                self.manager.open_log(log.as_deref());
            } else if action_command == self.mbc_proj_log.get_action_command() {
                self.manager
                    .open_log(self.get_project_log(self.base_manager()).as_deref());
            } else if action_command == self.cbc_boundary_model.get_action_command()
                && self.cbc_boundary_model.is_selected()
            {
                // The model is only opened for the A axis
                if self.imod_index_a.get() != -1 {
                    self.manager.imod_model(
                        AxisID::First,
                        self.imod_index_a.get(),
                        stack.parent(),
                        stack
                            .file_name()
                            .map(|name| name.to_string_lossy())
                            .as_deref(),
                        Some(&file_type::CLASS.batch_run_tomo_boundary_model),
                        dual,
                    );
                }
            } else if action_command == self.bc_edit_dataset.get_action_command() {
                let dataset_dialog = self.dataset_dialog.borrow().clone();
                if let Some(dataset_dialog) = dataset_dialog {
                    dataset_dialog.set_visible(true);
                    self.bc_edit_dataset.set_selected(true);
                } else {
                    // Save the state. This call will only save this row's autodoc.
                    let table = self.table();
                    let dataset_dialog = BatchRunTomoDatasetDialog::get_row_instance(
                        self.manager,
                        self.get_dataset_file(),
                        self.this.clone(),
                        table.as_ref().map(|table| table.get_template_values()),
                        self.basic_directives.clone(),
                        table.as_ref().map(|table| table.get_browsing_directory()),
                        self.stack_id.as_deref(),
                    );
                    *self.dataset_dialog.borrow_mut() = Some(dataset_dialog.clone());
                    self.manager.init_dialog(self.stack_id.as_deref(), false);
                    self.fc_edit_dataset.set_value_string(Some("   Set"));
                    dataset_dialog.set_montage(self.cbc_montage.is_selected());
                }
            } else if action_command == self.cbc_run.get_action_command() {
                self.send_status_change_status(Some(StatusRef::BatchRunTomoRowStatus(
                    BatchRunTomoRowStatus::Run,
                )));
            } else if action_command == self.cbc_montage.get_action_command() {
                let montage = self.cbc_montage.is_selected();
                if let Some(table) = self.table() {
                    table.msg_montage(montage);
                }
                let dataset_dialog = self.dataset_dialog.borrow().clone();
                if let Some(dataset_dialog) = dataset_dialog {
                    dataset_dialog.set_montage(montage);
                }
            }
        }
    }
}

impl Highlightable for BatchRunTomoRow {
    /// Java `highlight(boolean)`.
    fn highlight(&self, highlight: bool) {
        self.fc_stack.set_highlight(highlight);
        self.cbc_boundary_model.set_highlight(highlight);
        self.cbc_dual.set_highlight(highlight);
        self.cbc_montage.set_highlight(highlight);
        self.fc_skip.set_highlight(highlight);
        self.fc_bskip.set_highlight(highlight);
        self.cbc_surfaces_to_analyze2.set_highlight(highlight);
        self.fc_edit_dataset.set_highlight(highlight);
        self.fc_dataset_state.set_highlight(highlight);
        self.fc_ending_step.set_highlight(highlight);
        self.fc_cur_axis_letter.set_highlight(highlight);
        self.cbc_run.set_highlight(highlight);
    }
}

impl StatusChangeListener for BatchRunTomoRow {
    /// Java `statusChanged(Status)`.  Handles global status changes.
    fn status_changed_status(&self, new_status: Option<StatusRef>) {
        self.status_changed_full(new_status, None, None, None, false);
    }

    /// Java `statusChanged(StatusChangeEvent)`.  Handles dataset-level status changes
    /// and changes that the row-level dataset dialog responds to.
    fn status_changed_event(&self, event: Option<&dyn StatusChangeEvent>) {
        // Java dereferences the event; senders never pass null.
        if let Some(event) = event {
            self.status_changed_event_init(event, false);
        }
    }

    /// Java `startOver()`.
    fn start_over(&self) {
        self.status_changed_status(Some(StatusRef::BatchRunTomoStatus(
            BatchRunTomoStatus::Open,
        )));
        self.run_status.set(None);
    }
}

impl StatusChanger for BatchRunTomoRow {
    /// Java `addStatusChangeListener(StatusChangeListener)`.
    fn add_status_change_listener(&self, listener: Option<Rc<dyn StatusChangeListener>>) {
        let Some(listener) = listener else {
            return;
        };
        self.listeners
            .borrow_mut()
            .get_or_insert_with(Vec::new)
            .push(listener);
    }
}

/// `javax.swing.JTextField.CENTER` (`SwingConstants.CENTER`).
const JTEXTFIELD_CENTER: i32 = 0;

/// Java `new File(File parent, String child)`: a null parent is `new File(child)`.
fn java_io_file_new_file(parent: Option<&Path>, child: &str) -> PathBuf {
    match parent {
        Some(parent) => PathBuf::from(utilities::java_io_file_new(
            &parent.to_string_lossy(),
            child,
        )),
        None => PathBuf::from(child),
    }
}

/// Java `new File(String parent, String child)`: a null parent is `new File(child)`.
fn java_io_file_new_string_parent(parent: Option<&str>, child: &str) -> PathBuf {
    match parent {
        Some(parent) => PathBuf::from(utilities::java_io_file_new(parent, child)),
        None => PathBuf::from(child),
    }
}
