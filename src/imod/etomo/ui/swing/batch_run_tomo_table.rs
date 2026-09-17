//! `IMOD/Etomo/src/etomo/ui/swing/BatchRunTomoTable.java`.
//!
//! Swing construction and the `BatchRunTomoManager`/autodoc/file-chooser calls are
//! deliberately narrow boundaries.  The table's source-owned rows, paging,
//! highlighting, frame state, tab rebuilding, and action dispatch are retained here.
#![allow(dead_code)]

use std::path::PathBuf;

use super::batch_run_tomo_row::RunType;
pub use super::header_cell::HeaderCell;
use crate::imod::etomo::r#type::image_filename_style::ImageFilenameStyle;

pub const STACK_TITLE: &str = "Stack";
pub const STATUS_LABEL: &str = "Status";
pub const STEP_LABEL: &str = "Reached";
pub const RUN_LABEL: &str = "Run";
pub const UNIQUE_KEY: &str = "BatchRunTomo";
pub const SURFACES_TO_ANALYZE_LABEL1: &str = "Beads";
pub const SURFACES_TO_ANALYZE_LABEL2: &str = "on Two";
pub const SURFACES_TO_ANALYZE_LABEL3: &str = "Surfaces";
pub const DUAL_LABEL1: &str = "Dual";
pub const DUAL_LABEL2: &str = "Axis";
pub const DATASET_LABEL1: &str = "Open";
pub const DATASET_LABEL2: &str = "Set";
pub const REC_LABEL1: &str = "Open";
pub const REC_LABEL2: &str = "Rec";
pub const LOG_LABEL1: &str = "BRT";
pub const LOG_LABEL2: &str = "Log";
pub const CUR_AXIS_LABEL: &str = "Axis";

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum BatchRunTomoTab {
    #[default]
    Stacks,
    Dataset,
    Run,
}
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum BatchRunTomoStatus {
    #[default]
    Default,
    Open,
    Done,
    Failed,
}
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum BatchRunTomoTableAction {
    #[default]
    AddStacks,
    CopyDown,
    Delete,
    RunToggle,
}
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PreferredTableSize {
    pub columns: Vec<(DatasetColumn, String)>,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DatasetColumn {
    Number,
    Stack,
    EditDataset,
    Total,
}
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Viewport {
    pub table_size: usize,
    pub first_row: usize,
    pub paging_initialized: bool,
}
impl Viewport {
    pub fn new(table_size: usize) -> Self {
        Self {
            table_size,
            ..Self::default()
        }
    }
    pub fn init_paging(&mut self) {
        self.paging_initialized = true;
    }
    pub fn in_viewport(&self, index: isize) -> bool {
        index >= self.first_row as isize
            && index < self.first_row.saturating_add(self.table_size) as isize
    }
    pub fn adjust_viewport(&mut self, index: isize) {
        if index >= 0 {
            let index = index as usize;
            if index < self.first_row {
                self.first_row = index
            } else if self.table_size > 0 && index >= self.first_row + self.table_size {
                self.first_row = index + 1 - self.table_size
            }
        }
    }
    pub fn home_button_action(&mut self) {
        self.first_row = 0;
    }
    pub fn end_button_action(&mut self, size: usize) {
        self.first_row = size.saturating_sub(self.table_size);
    }
    pub fn up_button_action(&mut self) {
        self.first_row = self.first_row.saturating_sub(1);
    }
    pub fn down_button_action(&mut self) {
        self.first_row = self.first_row.saturating_add(1);
    }
}

/// Direct table-facing state of `BatchRunTomoRow.java`; that source unit is the
/// widget/autodoc boundary, while these are all fields this table reads or writes.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct BatchRunTomoRow {
    pub number: usize,
    pub stack_id: String,
    pub stack: PathBuf,
    pub original_location: String,
    pub root_name: String,
    pub dual: bool,
    pub montage: bool,
    pub run: bool,
    pub highlighted: bool,
    pub displayed: bool,
    pub stack_expanded: bool,
    pub ending_step: Option<usize>,
    pub changed: bool,
    pub valid: bool,
    /// The first valid value is Java `RowList.getImageFilenameStyle()`'s result.
    pub image_filename_style: Option<ImageFilenameStyle>,
}

/// Native serialized form of the `BatchRunTomoRowMetaData` fields that this table loads.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct BatchRunTomoTableRowMetaData {
    pub stack_id: String,
    pub stack: PathBuf,
    pub original_location: String,
    pub root_name: String,
    pub dual: bool,
    pub montage: bool,
    pub run: bool,
    pub ending_step: Option<usize>,
    pub image_filename_style: Option<ImageFilenameStyle>,
}

/// Native serialized form of Java `BatchRunTomoMetaData.getOrderedRows()`.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct BatchRunTomoTableMetaData {
    pub rows: Vec<BatchRunTomoTableRowMetaData>,
}

/// One Java `RunList.add(stackID, runStatus, dual)` entry.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BatchRunTomoRunEntry {
    pub stack_id: String,
    pub dual: bool,
    pub run_type: RunType,
}

/// Native `RunList` representation.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct BatchRunTomoRunList {
    pub entries: Vec<BatchRunTomoRunEntry>,
}
impl BatchRunTomoRow {
    pub fn select_highlight_button(&mut self) {
        self.highlighted = true;
    }
    pub fn remove(&mut self) {
        self.displayed = false;
    }
    pub fn display(&mut self, viewport: &Viewport) {
        self.displayed = viewport.in_viewport(self.number as isize - 1);
    }
    pub fn copy(&mut self, source: &Self) {
        let number = self.number;
        let stack_id = self.stack_id.clone();
        let stack = self.stack.clone();
        *self = source.clone();
        self.number = number;
        self.stack_id = stack_id;
        self.stack = stack;
        self.highlighted = false;
    }
}

/// Java private inner `RowList`.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RowList {
    pub list: Vec<BatchRunTomoRow>,
    pub table_listener_set: bool,
    pub row_listener_count: usize,
    pub changer_count: usize,
    pub status_listener_count: usize,
    pub image_filename_style: Option<ImageFilenameStyle>,
}
impl RowList {
    pub fn add_new(&mut self, stacks: Vec<(PathBuf, bool, bool)>) -> Vec<PathBuf> {
        let mut duplicates = vec![];
        for (stack, dual, single_axis) in stacks {
            let stack_id = stack.with_extension("").to_string_lossy().into_owned();
            if self.row_exists(&stack_id) {
                duplicates.push(stack);
                continue;
            }
            let number = self.list.len() + 1;
            self.list.push(BatchRunTomoRow {
                number,
                stack_id,
                root_name: stack
                    .file_stem()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into_owned(),
                original_location: stack
                    .parent()
                    .unwrap_or_else(|| std::path::Path::new(""))
                    .to_string_lossy()
                    .into_owned(),
                stack,
                dual: if single_axis { false } else { dual },
                valid: true,
                ..Default::default()
            });
        }
        duplicates
    }
    pub fn get_first_row(&self) -> Option<&BatchRunTomoRow> {
        self.list.first()
    }
    pub fn get_row(&self, stack_id: &str) -> Option<&BatchRunTomoRow> {
        self.list.iter().find(|row| row.stack_id == stack_id)
    }
    pub fn get_row_mut(&mut self, stack_id: &str) -> Option<&mut BatchRunTomoRow> {
        self.list.iter_mut().find(|row| row.stack_id == stack_id)
    }
    pub fn row_exists(&self, stack_id: &str) -> bool {
        self.get_row(stack_id).is_some()
    }
    pub fn find_row(&self, location: &str, root_name: &str) -> Option<String> {
        self.list
            .iter()
            .find(|row| row.original_location == location && row.root_name == root_name)
            .map(|row| row.stack_id.clone())
    }
    pub fn get_stack(&self, stack_id: &str) -> Option<PathBuf> {
        self.get_row(stack_id).map(|row| row.stack.clone())
    }
    pub fn remove_highlighted(&mut self) -> bool {
        let Some(mut index) = self.get_highlighted_index().map(|i| i as usize) else {
            return false;
        };
        self.list.remove(index);
        for (i, row) in self.list.iter_mut().enumerate().skip(index) {
            row.number = i + 1;
        }
        if index == self.list.len() {
            index = index.saturating_sub(1);
        }
        self.highlight(index as isize);
        true
    }
    pub fn highlight(&mut self, index: isize) {
        for row in &mut self.list {
            row.highlighted = false;
        }
        if let Some(row) = self.list.get_mut(index.max(0) as usize) {
            row.select_highlight_button();
        }
    }
    pub fn highlight_down(&mut self) -> isize {
        let Some(index) = self.get_highlighted_index() else {
            return -1;
        };
        let next = (index as usize + 1) % self.size();
        self.highlight(next as isize);
        next as isize
    }
    pub fn highlight_up(&mut self) -> isize {
        let Some(index) = self.get_highlighted_index() else {
            return -1;
        };
        let next = if index == 0 {
            self.size() - 1
        } else {
            index as usize - 1
        };
        self.highlight(next as isize);
        next as isize
    }
    pub fn copy_down(&mut self) {
        if let Some(index) = self
            .get_highlighted_index()
            .filter(|i| *i >= 0)
            .map(|i| i as usize)
        {
            if index + 1 < self.list.len() {
                let source = self.list[index].clone();
                self.list[index + 1].copy(&source);
            }
        }
    }
    pub fn remove_all(&mut self) {
        for row in &mut self.list {
            row.remove();
        }
    }
    pub fn display(&mut self, viewport: &Viewport) {
        for row in &mut self.list {
            row.display(viewport);
        }
    }
    pub fn size(&self) -> usize {
        self.list.len()
    }
    pub fn is_empty(&self) -> bool {
        self.list.is_empty()
    }
    pub fn is_single_frame(&self) -> bool {
        self.list.iter().any(|row| !row.montage)
    }
    pub fn is_montage_frame(&self) -> bool {
        self.list.iter().any(|row| row.montage)
    }
    pub fn has_dual(&self) -> bool {
        self.list.iter().any(|row| row.dual)
    }
    pub fn is_highlighted(&self) -> bool {
        self.list.iter().any(|row| row.highlighted)
    }
    pub fn get_highlighted_index(&self) -> Option<isize> {
        self.list
            .iter()
            .position(|row| row.highlighted)
            .map(|i| i as isize)
    }
    pub fn get_earliest_run_ending_step(&self) -> Option<usize> {
        let mut earliest = None;
        for row in self.list.iter().filter(|row| row.run) {
            let step = row.ending_step?;
            earliest = Some(earliest.map_or(step, |e: usize| e.min(step)));
        }
        earliest
    }
    pub fn validate(&self) -> bool {
        self.list.iter().all(|row| row.valid)
    }
    pub fn get_image_filename_style(&self) -> Option<ImageFilenameStyle> {
        self.image_filename_style.clone().or_else(|| {
            self.list
                .iter()
                .find_map(|row| row.image_filename_style.clone())
        })
    }
    pub fn load(&mut self, meta_data: &BatchRunTomoTableMetaData, expanded: bool) {
        for row_meta_data in &meta_data.rows {
            let number = self.list.len() + 1;
            self.list.push(BatchRunTomoRow {
                number,
                stack_id: row_meta_data.stack_id.clone(),
                stack: row_meta_data.stack.clone(),
                original_location: row_meta_data.original_location.clone(),
                root_name: row_meta_data.root_name.clone(),
                dual: row_meta_data.dual,
                montage: row_meta_data.montage,
                run: row_meta_data.run,
                ending_step: row_meta_data.ending_step,
                image_filename_style: row_meta_data.image_filename_style.clone(),
                stack_expanded: expanded,
                valid: true,
                ..Default::default()
            });
        }
    }
    pub fn create_run_list(&self, run_type: RunType) -> BatchRunTomoRunList {
        BatchRunTomoRunList {
            entries: self
                .list
                .iter()
                .filter(|row| row.run)
                .map(|row| BatchRunTomoRunEntry {
                    stack_id: row.stack_id.clone(),
                    dual: row.dual,
                    run_type,
                })
                .collect(),
        }
    }
    pub fn expand_stack(&mut self, expanded: bool) {
        for row in &mut self.list {
            row.stack_expanded = expanded;
        }
    }
    pub fn backup_if_changed(&mut self, stack_id: Option<&str>) -> bool {
        self.list
            .iter_mut()
            .filter(|row| stack_id.is_none_or(|id| id == row.stack_id))
            .any(|row| {
                let changed = row.changed;
                row.changed = false;
                changed
            })
    }
}

/// Java final `BatchRunTomoTable`, retaining every direct source-owned state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BatchRunTomoTable {
    pub headers: Vec<Vec<HeaderCell>>,
    pub row_list: RowList,
    pub viewport: Viewport,
    pub preferred_table_size: PreferredTableSize,
    pub cur_tab: BatchRunTomoTab,
    pub status: BatchRunTomoStatus,
    pub montage_frame: bool,
    pub single_frame: bool,
    pub row_running: bool,
    pub series_watcher: bool,
    pub stack_expanded: bool,
    pub add_editable: bool,
    pub copy_down_editable: bool,
    pub delete_editable: bool,
    pub delete_enabled: bool,
    pub copy_down_enabled: bool,
    pub stack_buttons_visible: bool,
    pub packed: bool,
    pub repaint_requested: bool,
    pub status_events: Vec<(bool, &'static str)>,
    pub duplicate_warnings: Vec<PathBuf>,
    pub table_reference: String,
    pub table_listener_set: bool,
    pub status_listener_count: usize,
    pub row_status_listener_count: usize,
    pub values_applied: bool,
    pub autodocs_saved: usize,
    pub autodocs_loaded: usize,
}
impl BatchRunTomoTable {
    /// Java `getInstance(...)`, `createPanel`, `setTooltips`, and `addListeners`.
    pub fn get_instance(table_size: usize, table_reference: impl Into<String>) -> Self {
        let mut table = Self {
            headers: vec![],
            row_list: RowList::default(),
            viewport: Viewport::new(table_size),
            preferred_table_size: PreferredTableSize::default(),
            cur_tab: BatchRunTomoTab::Stacks,
            status: BatchRunTomoStatus::Default,
            montage_frame: false,
            single_frame: false,
            row_running: false,
            series_watcher: false,
            stack_expanded: false,
            add_editable: true,
            copy_down_editable: true,
            delete_editable: true,
            delete_enabled: false,
            copy_down_enabled: false,
            stack_buttons_visible: true,
            packed: false,
            repaint_requested: false,
            status_events: vec![],
            duplicate_warnings: vec![],
            table_reference: table_reference.into(),
            table_listener_set: false,
            status_listener_count: 0,
            row_status_listener_count: 0,
            values_applied: false,
            autodocs_saved: 0,
            autodocs_loaded: 0,
        };
        table.create_panel();
        table.set_tooltips();
        table.add_listeners();
        table
    }
    pub fn msg_row_running(&mut self, running: bool) {
        if self.row_running != running {
            self.row_running = running;
            self.packed = true;
        }
    }
    pub fn has_dual(&self) -> bool {
        self.row_list.has_dual()
    }
    /// Native focus-parent identities corresponding to Java `getFocusableParents()`.
    pub fn get_focusable_parents(&self) -> [&'static str; 4] {
        ["viewport", "table", "stack-buttons", "root"]
    }
    /// Java `getImageFilenameStyle()`.
    pub fn get_image_filename_style(&self) -> Option<ImageFilenameStyle> {
        self.row_list.get_image_filename_style()
    }
    pub fn create_panel(&mut self) {
        self.viewport.init_paging();
        self.headers = vec![
            vec![
                HeaderCell::new("#"),
                HeaderCell::new(""),
                HeaderCell::new(""),
            ],
            vec![
                HeaderCell::new(STACK_TITLE),
                HeaderCell::new(""),
                HeaderCell::new(""),
            ],
        ];
        self.preferred_table_size.columns = vec![
            (DatasetColumn::Number, "#".into()),
            (DatasetColumn::Stack, STACK_TITLE.into()),
            (DatasetColumn::EditDataset, "Specific Values".into()),
        ];
        self.update_display();
        self.status_changed(self.status);
    }
    pub fn display(&mut self) {
        self.cur_tab = BatchRunTomoTab::Stacks;
        self.msg_tab_changed(self.cur_tab);
    }
    pub fn rebuild_table(&mut self) {
        self.row_list.remove_all();
        self.headers.clear();
        match self.cur_tab {
            BatchRunTomoTab::Stacks => {
                self.headers = vec![vec![
                    HeaderCell::new("#"),
                    HeaderCell::new(STACK_TITLE),
                    HeaderCell::new(DUAL_LABEL1),
                    HeaderCell::new("Montage"),
                    HeaderCell::new("Exclude"),
                    HeaderCell::new("Boundary"),
                    HeaderCell::new(SURFACES_TO_ANALYZE_LABEL1),
                    HeaderCell::new("Open"),
                ]]
            }
            BatchRunTomoTab::Dataset => {
                self.headers = vec![vec![
                    HeaderCell::new("#"),
                    HeaderCell::new(STACK_TITLE),
                    HeaderCell::new("Specific Values"),
                ]]
            }
            BatchRunTomoTab::Run => {
                self.headers = vec![vec![
                    HeaderCell::new(STATUS_LABEL),
                    HeaderCell::new(STEP_LABEL),
                    HeaderCell::new(CUR_AXIS_LABEL),
                    HeaderCell::new(RUN_LABEL),
                    HeaderCell::new(DATASET_LABEL1),
                    HeaderCell::new(REC_LABEL1),
                    HeaderCell::new("Proj"),
                    HeaderCell::new(LOG_LABEL1),
                ]]
            }
        };
        self.row_list.display(&self.viewport);
        self.stack_buttons_visible = self.cur_tab == BatchRunTomoTab::Stacks;
    }
    /// Java `addStandardHeaders(int)`, represented by the common table header cells.
    pub fn add_standard_headers(&mut self, index: usize) {
        let number = HeaderCell::new("#");
        let stack = HeaderCell::new(STACK_TITLE);
        if self.headers.len() <= index {
            self.headers.resize_with(index + 1, Vec::new);
        }
        self.headers[index].extend([number, stack]);
    }
    pub fn add_listeners(&mut self) {}
    /// Java `addStatusChangeListener(StatusChangeListener)`.
    pub fn add_status_change_listener(&mut self, present: bool) {
        if present {
            self.status_listener_count += 1;
        }
    }
    /// Java `msgStatusChangerStarted(StatusChanger)`.
    pub fn msg_status_changer_started(&mut self, present: bool) {
        if present {
            self.row_list.changer_count += 1;
        }
    }
    /// Java `addStatusChangeListenerToRowList(StatusChangeListener)`.
    pub fn add_status_change_listener_to_row_list(&mut self, present: bool) {
        if present {
            self.row_list.status_listener_count += 1;
        }
    }
    /// Java `addStatusChangeListenerToRows(StatusChangeListener)`.
    pub fn add_status_change_listener_to_rows(&mut self, present: bool) {
        if present {
            self.row_status_listener_count += 1;
            self.row_list.row_listener_count += 1;
        }
    }
    /// Java `getUIComponent()`; native frontends use the retained table state itself.
    pub fn get_ui_component(&self) -> &Self {
        self
    }
    /// Java `getComponent()`; returns the native root component identity.
    pub fn get_component(&self) -> &'static str {
        "batch-run-tomo-table-root"
    }
    /// Java `setTableListener(TableListener)`.
    pub fn set_table_listener(&mut self, present: bool) {
        self.table_listener_set = present;
        self.row_list.table_listener_set = present;
    }
    pub fn validate(&self) -> bool {
        self.row_list.validate()
    }
    pub fn msg_montage(&mut self, montage: bool) {
        if montage {
            if !self.montage_frame {
                self.montage_frame = true;
                self.send_status_changed(true, "montage");
            }
            self.set_single_frame();
        } else {
            if !self.single_frame {
                self.single_frame = true;
                self.send_status_changed(true, "single");
            }
            self.set_montage_frame();
        }
    }
    pub fn send_status_changed(&mut self, input: bool, status: &'static str) {
        self.status_events.push((input, status));
    }
    pub fn set_single_frame(&mut self) {
        let original = self.single_frame;
        self.single_frame = if self.row_list.is_empty() {
            true
        } else {
            self.row_list.is_single_frame()
        };
        if original != self.single_frame {
            self.send_status_changed(self.single_frame, "single");
        }
    }
    pub fn set_montage_frame(&mut self) {
        let original = self.montage_frame;
        self.montage_frame = if self.row_list.is_empty() {
            true
        } else {
            self.row_list.is_montage_frame()
        };
        if original != self.montage_frame {
            self.send_status_changed(self.montage_frame, "montage");
        }
    }
    pub fn set_frame(&mut self, force: bool) {
        let single = self.row_list.is_empty() || self.row_list.is_single_frame();
        let montage = self.row_list.is_empty() || self.row_list.is_montage_frame();
        if force || single != self.single_frame {
            self.send_status_changed(single, "single");
        }
        if force || montage != self.montage_frame {
            self.send_status_changed(montage, "montage");
        }
        self.single_frame = single;
        self.montage_frame = montage;
    }
    pub fn highlight_down_action_performed(&mut self) {
        let adjust = self
            .viewport
            .in_viewport(self.row_list.get_highlighted_index().unwrap_or(-1));
        let index = self.row_list.highlight_down();
        if index >= 0 && adjust && !self.viewport.in_viewport(index) {
            if index == 0 {
                self.viewport.home_button_action()
            } else {
                self.viewport.down_button_action()
            }
        }
    }
    pub fn highlight_up_action_performed(&mut self) {
        let adjust = self
            .viewport
            .in_viewport(self.row_list.get_highlighted_index().unwrap_or(-1));
        let index = self.row_list.highlight_up();
        if index >= 0 && adjust && !self.viewport.in_viewport(index) {
            if index as usize == self.row_list.size() - 1 {
                self.viewport.end_button_action(self.row_list.size())
            } else {
                self.viewport.up_button_action()
            }
        }
    }
    pub fn get_first_row(&self) -> Option<&BatchRunTomoRow> {
        self.row_list.get_first_row()
    }
    pub fn find_row(&self, location: &str, root_name: &str) -> Option<String> {
        self.row_list.find_row(location, root_name)
    }
    pub fn get_stack(&self, stack_id: &str) -> Option<PathBuf> {
        self.row_list.get_stack(stack_id)
    }
    pub fn backup_if_changed(&mut self, stack_id: Option<&str>) -> bool {
        self.row_list.backup_if_changed(stack_id)
    }
    pub fn is_series_watcher_on(&self) -> bool {
        self.series_watcher
    }
    pub fn update_display(&mut self) {
        let size = self.row_list.size();
        self.delete_enabled = size > 0 && self.row_list.is_highlighted();
        self.copy_down_enabled = self
            .row_list
            .get_highlighted_index()
            .is_some_and(|index| index >= 0 && (index as usize) < size.saturating_sub(1));
    }
    pub fn update_row_display(&mut self) {
        self.update_display();
    }
    pub fn start_over(&mut self) {
        self.status_changed(BatchRunTomoStatus::Open);
    }
    pub fn status_changed(&mut self, status: BatchRunTomoStatus) {
        if self.status != BatchRunTomoStatus::Done || status != BatchRunTomoStatus::Failed {
            self.status = status;
        }
        let open = self.status == BatchRunTomoStatus::Open;
        self.add_editable = open;
        self.copy_down_editable = open;
        self.delete_editable = open;
    }
    pub fn msg_tab_changed(&mut self, tab: BatchRunTomoTab) {
        if self.cur_tab != tab {
            self.cur_tab = tab;
            self.rebuild_table();
            self.packed = true;
        }
    }
    pub fn msg_viewport_paged(&mut self) {
        self.row_list.remove_all();
        self.row_list.display(&self.viewport);
        self.packed = true;
    }
    pub fn highlight(&mut self, _: bool) {
        self.update_display();
    }
    pub fn action_performed(
        &mut self,
        action: BatchRunTomoTableAction,
        stacks: Vec<(PathBuf, bool, bool)>,
    ) {
        match action {
            BatchRunTomoTableAction::AddStacks => {
                self.duplicate_warnings = self.row_list.add_new(stacks);
                self.viewport
                    .adjust_viewport((self.row_list.size() as isize) - 1);
                self.row_list.display(&self.viewport);
                self.set_frame(false);
                self.update_display();
            }
            BatchRunTomoTableAction::CopyDown => {
                self.row_list.copy_down();
            }
            BatchRunTomoTableAction::Delete => {
                if self.row_list.remove_highlighted() {
                    self.viewport
                        .adjust_viewport(self.row_list.get_highlighted_index().unwrap_or(-1));
                    self.set_frame(false);
                    self.update_display();
                }
            }
            BatchRunTomoTableAction::RunToggle => {
                for row in &mut self.row_list.list {
                    row.run = !row.run;
                }
            }
        }
    }
    pub fn expand(&mut self, expanded: bool) {
        self.stack_expanded = expanded;
        self.row_list.expand_stack(expanded);
        self.packed = true;
    }
    /// Java `expandStack(boolean)`.
    pub fn expand_stack(&mut self, expanded: bool) {
        self.expand(expanded);
    }
    /// Java `load(BatchRunTomoMetaData)`.
    pub fn load(&mut self, meta_data: &BatchRunTomoTableMetaData) {
        let first_index = self.row_list.size();
        self.row_list.load(meta_data, self.stack_expanded);
        if self.row_list.size() != first_index {
            self.viewport.adjust_viewport(first_index as isize);
            self.row_list.display(&self.viewport);
            self.update_display();
            self.packed = true;
        }
    }
    /// Java `setParameters(BatchRunTomoMetaData, String, boolean, boolean)`.
    /// A selected stack updates only that row; otherwise the serialized table is loaded.
    pub fn set_parameters(
        &mut self,
        meta_data: &BatchRunTomoTableMetaData,
        only_stack_id: Option<&str>,
        _init: bool,
    ) {
        if let Some(stack_id) = only_stack_id {
            if let (Some(row), Some(values)) = (
                self.row_list.get_row_mut(stack_id),
                meta_data.rows.iter().find(|row| row.stack_id == stack_id),
            ) {
                row.dual = values.dual;
                row.montage = values.montage;
                row.run = values.run;
                row.ending_step = values.ending_step;
                row.image_filename_style = values.image_filename_style.clone();
            }
        } else {
            self.row_list.list.clear();
            self.load(meta_data);
        }
    }
    /// Java series-watcher `setParameters` path: create a blank row when needed.
    pub fn add_blank_row(&mut self, stack_id: &str) -> bool {
        if stack_id.is_empty() || self.row_list.row_exists(stack_id) {
            return false;
        }
        let number = self.row_list.size() + 1;
        self.row_list.list.push(BatchRunTomoRow {
            number,
            stack_id: stack_id.to_string(),
            stack_expanded: self.stack_expanded,
            valid: true,
            ..Default::default()
        });
        self.viewport.adjust_viewport((number - 1) as isize);
        self.row_list.display(&self.viewport);
        self.update_display();
        true
    }
    /// Java `getParameters(BatchRunTomoMetaData)`.
    pub fn get_parameters(&self) -> BatchRunTomoTableMetaData {
        BatchRunTomoTableMetaData {
            rows: self
                .row_list
                .list
                .iter()
                .map(|row| BatchRunTomoTableRowMetaData {
                    stack_id: row.stack_id.clone(),
                    stack: row.stack.clone(),
                    original_location: row.original_location.clone(),
                    root_name: row.root_name.clone(),
                    dual: row.dual,
                    montage: row.montage,
                    run: row.run,
                    ending_step: row.ending_step,
                    image_filename_style: row.image_filename_style.clone(),
                })
                .collect(),
        }
    }
    /// Native form of Java row `applyValues`; directives are resolved by the frontend.
    pub fn apply_values(
        &mut self,
        _init: bool,
        _retain_user_values: bool,
        only_stack_id: Option<&str>,
    ) {
        self.values_applied = true;
        if only_stack_id.is_none() {
            self.set_frame(false);
        }
    }
    /// Native form of Java `saveAutodocs`; callers own the actual filesystem writer.
    pub fn save_autodocs(&mut self, only_stack_id: Option<&str>) -> bool {
        let count = only_stack_id
            .and_then(|id| self.row_list.row_exists(id).then_some(1))
            .unwrap_or(self.row_list.size());
        self.autodocs_saved += count;
        true
    }
    /// Native form of Java `loadAutodocs`; callers supply decoded values separately.
    pub fn load_autodocs(&mut self, only_stack_id: Option<&str>, _only_advanced: bool) {
        self.autodocs_loaded += only_stack_id
            .and_then(|id| self.row_list.row_exists(id).then_some(1))
            .unwrap_or(self.row_list.size());
    }
    /// Java `createRunList(RunType)`.
    pub fn create_run_list(&self, run_type: RunType) -> BatchRunTomoRunList {
        self.row_list.create_run_list(run_type)
    }
    pub fn size(&self) -> usize {
        self.row_list.size()
    }
    pub fn set_tooltips(&mut self) {}
    pub fn get_table_reference(&self) -> &str {
        &self.table_reference
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn add_duplicate_copy_and_delete_follow_java_row_numbering() {
        let mut t = BatchRunTomoTable::get_instance(2, "r");
        t.action_performed(
            BatchRunTomoTableAction::AddStacks,
            vec![
                (PathBuf::from("/a.st"), false, true),
                (PathBuf::from("/b.st"), true, false),
            ],
        );
        assert_eq!(t.size(), 2);
        t.row_list.highlight(0);
        t.action_performed(BatchRunTomoTableAction::CopyDown, vec![]);
        assert!(t.row_list.list[1].stack_id.contains("b"));
        t.action_performed(BatchRunTomoTableAction::Delete, vec![]);
        assert_eq!(t.row_list.list[0].number, 1);
    }
    #[test]
    fn tab_rebuild_and_viewport_highlight_wrap_are_source_shaped() {
        let mut t = BatchRunTomoTable::get_instance(1, "r");
        t.action_performed(
            BatchRunTomoTableAction::AddStacks,
            vec![
                (PathBuf::from("/a.st"), false, true),
                (PathBuf::from("/b.st"), false, true),
            ],
        );
        t.row_list.highlight(0);
        t.highlight_up_action_performed();
        assert_eq!(t.row_list.get_highlighted_index(), Some(1));
        t.msg_tab_changed(BatchRunTomoTab::Run);
        assert!(!t.stack_buttons_visible);
        assert_eq!(t.headers[0][0].text, STATUS_LABEL);
    }
    #[test]
    fn frame_notifications_and_status_editability_match_table_rules() {
        let mut t = BatchRunTomoTable::get_instance(2, "r");
        t.set_frame(true);
        assert!(t.single_frame && t.montage_frame);
        t.status_changed(BatchRunTomoStatus::Open);
        assert!(t.add_editable);
        t.status_changed(BatchRunTomoStatus::Done);
        assert!(!t.delete_editable);
    }
}
