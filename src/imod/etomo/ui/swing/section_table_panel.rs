//! `IMOD/Etomo/src/etomo/ui/swing/SectionTablePanel.java`.
//!
//! Swing layout, file selection, MRC-header reading, `JoinDialog`, `JoinManager`,
//! and 3dmod calls are presentation/application boundaries.  The Java source's table
//! ownership, row ordering, paging, mode transitions, action dispatch, and metadata
//! rules are retained here.  `SectionTableRow.java` is represented by precisely the
//! state and operations observed by this source unit until that source unit is ported.
#![allow(dead_code)]

use std::path::{Path, PathBuf};

pub const HEADER1_SECTIONS_LABEL: &str = "Sections";
pub const LABEL: &str = "Section Table";
pub const UNIQUE_KEY: &str = "Section";
pub const SETUP_MODE: i32 = 0;
pub const SAMPLE_PRODUCED_MODE: i32 = 1;
pub const SAMPLE_NOT_PRODUCED_MODE: i32 = 2;
pub const CHANGING_SAMPLE_MODE: i32 = 3;
pub const INVERTED_WARNING: &str = "The section will be inverted when the join is run.";
pub const FLIP_WARNING: [&str; 2] = [
    "Tomograms have to be rotated after generation",
    "in order to be in the right orientation for joining serial sections.",
];

/// Java `JoinDialog.Tab` values examined by this unit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Tab {
    Setup,
    Align,
    Join,
    Rejoin,
    Model,
}

/// Java `HeaderCell`; component creation/layout remains at the GUI boundary.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct HeaderCell {
    pub text: String,
    pub width: Option<usize>,
    pub tooltip: String,
}
impl HeaderCell {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            ..Self::default()
        }
    }
    pub fn with_width(text: impl Into<String>, width: usize) -> Self {
        Self {
            text: text.into(),
            width: Some(width),
            tooltip: String::new(),
        }
    }
    pub fn set_text(&mut self, text: impl Into<String>) {
        self.text = text.into();
    }
    pub fn set_tool_tip_text(&mut self, text: impl Into<String>) {
        self.tooltip = text.into();
    }
}

/// `ConstSectionTableRowData` and `SectionTableRowData` fields reached here.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SectionTableRowData {
    pub row_index: usize,
    pub setup_section: PathBuf,
    pub join_final_start: i32,
    pub join_final_end: i32,
    pub x_max: i32,
    pub y_max: i32,
    pub z_max: i32,
    pub rotated: bool,
    pub inverted: bool,
    pub valid: bool,
    pub invalid_reason: Option<String>,
}
pub type ConstSectionTableRowData = SectionTableRowData;

/// Java `ConstJoinMetaData` / `JoinMetaData` direct section-table access.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct JoinMetaData {
    pub section_table_data: Option<Vec<SectionTableRowData>>,
}

/// Direct source-facing representation of the separate `SectionTableRow.java` unit.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SectionTableRow {
    pub data: SectionTableRowData,
    pub highlighted: bool,
    pub displayed: bool,
    pub expanded: bool,
    pub mode: i32,
    pub setup_to_join_synchronized: bool,
    pub join_to_setup_synchronized: bool,
    pub imod_opened: Option<(bool, i32)>,
    pub imod_removed: bool,
    pub angles_retrieved: bool,
}
impl SectionTableRow {
    pub fn new(row_number: usize, section: impl Into<PathBuf>, expanded: bool) -> Self {
        Self {
            data: SectionTableRowData {
                row_index: row_number - 1,
                setup_section: section.into(),
                valid: true,
                ..Default::default()
            },
            expanded,
            ..Default::default()
        }
    }
    pub fn set_mode(&mut self, mode: i32) {
        self.mode = mode;
    }
    pub fn set_names(&mut self) {}
    pub fn is_rotated(&self) -> bool {
        self.data.rotated
    }
    pub fn setup_cur_tab(&mut self, _previous: Option<&SectionTableRow>, _size: usize) {}
    pub fn display(&mut self, _index: usize, viewport: &Viewport) {
        self.displayed = viewport.in_viewport(self.data.row_index);
    }
    pub fn is_highlighted(&self) -> bool {
        self.highlighted
    }
    pub fn select_highlight_button(&mut self) {
        self.highlighted = true;
    }
    pub fn set_inverted(&mut self, inverted: bool) {
        self.data.inverted = inverted;
    }
    pub fn set_join_final_start_highlight(&mut self, _highlight: bool) {}
    pub fn set_join_final_end_highlight(&mut self, _highlight: bool) {}
    pub fn expand_section(&mut self, expand: bool) {
        self.expanded = expand;
    }
    pub fn equals(&self, data: &ConstSectionTableRowData) -> bool {
        self.data == *data
    }
    pub fn equals_sample(&self, data: &ConstSectionTableRowData) -> bool {
        self.data.setup_section == data.setup_section
    }
    pub fn equals_setup_section(&self, section: &Path) -> bool {
        self.data.setup_section == section
    }
    pub fn remove(&mut self) {
        self.displayed = false;
    }
    pub fn remove_imod(&mut self) {
        self.imod_removed = true;
    }
    pub fn set_row_number(&mut self, row_number: usize) {
        self.data.row_index = row_number - 1;
    }
    pub fn swap_bottom_top(&mut self) {}
    pub fn is_valid(&self) -> bool {
        self.data.valid
    }
    pub fn get_data(&self) -> &ConstSectionTableRowData {
        &self.data
    }
    pub fn validate_makejoincom(&self, _max_row: &str) -> bool {
        self.data.valid
    }
    pub fn validate_finishjoin(&self) -> bool {
        self.data.valid
    }
    pub fn set_in_use(&mut self) {}
    pub fn get_invalid_reason(&self) -> Option<&str> {
        self.data.invalid_reason.as_deref()
    }
    pub fn get_x_max(&self) -> i32 {
        self.data.x_max
    }
    pub fn get_y_max(&self) -> i32 {
        self.data.y_max
    }
    pub fn get_z_max(&self) -> i32 {
        self.data.z_max
    }
    pub fn get_setup_section_text(&self) -> String {
        self.data.setup_section.display().to_string()
    }
    pub fn synchronize_setup_to_join(&mut self) {
        self.setup_to_join_synchronized = true;
    }
    pub fn synchronize_join_to_setup(&mut self) {
        self.join_to_setup_synchronized = true;
    }
    pub fn imod_open_setup_section_file(&mut self, binning: i32) {
        self.imod_opened = Some((true, binning));
    }
    pub fn imod_open_join_section_file(&mut self, binning: i32) {
        self.imod_opened = Some((false, binning));
    }
    pub fn imod_get_angles(&mut self) -> bool {
        self.angles_retrieved = true;
        true
    }
}

/// Java private final inner `RowList`.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RowList {
    pub list: Vec<SectionTableRow>,
}
impl RowList {
    pub fn has_rotated_section(&self) -> bool {
        self.list.iter().any(SectionTableRow::is_rotated)
    }
    pub fn display_cur_tab(&mut self, viewport: &Viewport) {
        let size = self.list.len();
        for index in 0..size {
            self.list[index].setup_cur_tab(None, size);
            self.list[index].display(index, viewport);
        }
    }
    pub fn size(&self) -> usize {
        self.list.len()
    }
    pub fn get_highlighted_index(&self) -> isize {
        self.list
            .iter()
            .position(SectionTableRow::is_highlighted)
            .map_or(-1, |i| i as isize)
    }
    pub fn highlight(&mut self, index: usize) {
        if let Some(row) = self.list.get_mut(index) {
            row.select_highlight_button();
        }
    }
    pub fn highlight_down(&mut self) -> isize {
        let index = self.get_highlighted_index();
        if index < 0 {
            return -1;
        }
        let index = (index as usize + 1) % self.size();
        self.highlight(index);
        index as isize
    }
    pub fn highlight_up(&mut self) -> isize {
        let index = self.get_highlighted_index();
        if index < 0 {
            return -1;
        }
        let index = if index == 0 {
            self.size() - 1
        } else {
            index as usize - 1
        };
        self.highlight(index);
        index as isize
    }
    pub fn set_inverted(&mut self, inverted: &[bool]) -> usize {
        let mut count = 0;
        for (index, row) in self.list.iter_mut().enumerate() {
            if let Some(value) = inverted.get(index) {
                if *value {
                    count += 1;
                }
                row.set_inverted(*value);
            }
        }
        count
    }
    pub fn set_mode(&mut self, mode: i32) {
        for row in &mut self.list {
            row.set_mode(mode);
        }
    }
    pub fn set_join_final_start_highlight(&mut self, highlight: bool) {
        for row in &mut self.list {
            row.set_join_final_start_highlight(highlight);
        }
    }
    pub fn set_join_final_end_highlight(&mut self, highlight: bool) {
        for row in &mut self.list {
            row.set_join_final_end_highlight(highlight);
        }
    }
    pub fn expand(&mut self, expand: bool) {
        for row in &mut self.list {
            row.expand_section(expand);
        }
    }
    pub fn equals(&self, data: &[SectionTableRowData]) -> bool {
        self.list.len() == data.len()
            && self
                .list
                .iter()
                .zip(data)
                .all(|(row, data)| row.equals(data))
    }
    pub fn equals_sample(&self, data: &[SectionTableRowData]) -> bool {
        self.list.len() == data.len()
            && self
                .list
                .iter()
                .zip(data)
                .all(|(row, data)| row.equals_sample(data))
    }
    pub fn move_section_up(&mut self, index: usize) {
        self.list.swap(index, index - 1);
    }
    pub fn move_section_down(&mut self, index: usize) {
        self.list.swap(index, index + 1);
    }
    pub fn is_duplicate(&self, section: &Path) -> bool {
        self.list
            .iter()
            .any(|row| row.equals_setup_section(section))
    }
    pub fn add(&mut self, section: impl Into<PathBuf>, expanded: bool, mode: i32) -> usize {
        let index = self.list.len();
        let mut row = SectionTableRow::new(index + 1, section, expanded);
        row.set_mode(mode);
        row.set_names();
        self.list.push(row);
        index
    }
    pub fn get(&self, index: usize) -> Option<&SectionTableRow> {
        self.list.get(index)
    }
    pub fn get_mut(&mut self, index: usize) -> Option<&mut SectionTableRow> {
        self.list.get_mut(index)
    }
    pub fn delete_section(&mut self, index: usize) {
        if index < self.list.len() {
            let mut row = self.list.remove(index);
            row.remove();
            row.remove_imod();
        }
    }
    pub fn delete_sections(&mut self) {
        for row in &mut self.list {
            row.remove();
        }
        self.list.clear();
    }
    pub fn renumber_table(&mut self, start: usize) {
        for (index, row) in self.list.iter_mut().enumerate().skip(start) {
            row.set_row_number(index + 1);
        }
    }
    pub fn remove_rows(&mut self) {
        for row in &mut self.list {
            row.remove();
        }
    }
    pub fn display_rows(&mut self, viewport: &Viewport) {
        for (index, row) in self.list.iter_mut().enumerate() {
            row.display(index, viewport);
        }
    }
    pub fn invert_table(&mut self) {
        self.list.reverse();
        for (index, row) in self.list.iter_mut().enumerate() {
            row.set_row_number(index + 1);
            row.swap_bottom_top();
        }
    }
    pub fn get_meta_data(&self, meta_data: &mut JoinMetaData) -> bool {
        meta_data.section_table_data = Some(self.list.iter().map(|row| row.data.clone()).collect());
        self.list.iter().all(SectionTableRow::is_valid)
    }
    pub fn set_meta_data(&mut self, data: &[SectionTableRowData], mode: i32, viewport: &Viewport) {
        for data in data {
            let mut row = SectionTableRow {
                data: data.clone(),
                ..Default::default()
            };
            row.set_names();
            row.set_mode(mode);
            row.display(data.row_index, viewport);
            self.list.insert(data.row_index.min(self.list.len()), row);
        }
    }
    pub fn validate_makejoincom(&self) -> bool {
        let max = self.list.len().to_string();
        self.list.iter().all(|row| row.validate_makejoincom(&max))
    }
    pub fn validate_finishjoin(&self) -> bool {
        self.list.iter().all(SectionTableRow::validate_finishjoin)
    }
    pub fn configure_rows(&mut self) {
        for row in &mut self.list {
            row.set_in_use();
        }
    }
    pub fn get_invalid_reason(&self) -> Option<String> {
        self.list
            .iter()
            .find_map(|row| row.get_invalid_reason().map(str::to_owned))
    }
    pub fn get_x_max(&self) -> i32 {
        self.list
            .iter()
            .map(SectionTableRow::get_x_max)
            .max()
            .unwrap_or(0)
    }
    pub fn get_y_max(&self) -> i32 {
        self.list
            .iter()
            .map(SectionTableRow::get_y_max)
            .max()
            .unwrap_or(0)
    }
    pub fn get_z_max(&self) -> i32 {
        self.list
            .iter()
            .map(SectionTableRow::get_z_max)
            .max()
            .unwrap_or(0)
    }
    pub fn get_setup_section_text(&self, index: usize) -> String {
        self.get(index)
            .map_or_else(String::new, SectionTableRow::get_setup_section_text)
    }
    pub fn synchronize_setup_to_join(&mut self) {
        for row in &mut self.list {
            row.synchronize_setup_to_join();
        }
    }
    pub fn synchronize_join_to_setup(&mut self) {
        for row in &mut self.list {
            row.synchronize_join_to_setup();
        }
    }
}

/// Java `Viewport`; paging-widget construction remains at the GUI boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Viewport {
    pub table_size: usize,
    pub first_row: usize,
    pub unique_key: String,
}
impl Viewport {
    pub fn new(table_size: usize) -> Self {
        Self {
            table_size,
            first_row: 0,
            unique_key: UNIQUE_KEY.into(),
        }
    }
    pub fn init_paging(&mut self) {}
    pub fn in_viewport(&self, index: usize) -> bool {
        index >= self.first_row && index < self.first_row.saturating_add(self.table_size)
    }
    pub fn adjust_viewport(&mut self, index: usize) {
        if index < self.first_row {
            self.first_row = index;
        } else if self.table_size > 0 && index >= self.first_row + self.table_size {
            self.first_row = index + 1 - self.table_size;
        }
    }
    pub fn home_button_action(&mut self) {
        self.first_row = 0;
    }
    pub fn end_button_action(&mut self, size: usize) {
        self.first_row = size.saturating_sub(self.table_size);
    }
    pub fn down_button_action(&mut self) {
        self.first_row += 1;
    }
    pub fn up_button_action(&mut self) {
        self.first_row = self.first_row.saturating_sub(1);
    }
}

/// Source-owned `SectionTableActionListener` command identities.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SectionTableAction {
    MoveSectionUp,
    MoveSectionDown,
    AddSection,
    DeleteSection,
    GetAngles,
    InvertTable,
    OpenIn3dmod,
}

/// Java `GridBagLayout` state returned by `getTableLayout`.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TableLayout;

/// Java `GridBagConstraints` state returned by `getTableConstraints`.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TableConstraints {
    pub fill_both: bool,
    pub anchor_center: bool,
    pub weight_x: i32,
    pub weight_y: i32,
    pub grid_width: i32,
    pub grid_height: i32,
}

/// Java private static `SectionTableActionListener` adapter.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SectionTableActionListener;
impl SectionTableActionListener {
    pub fn action_performed(
        &self,
        adaptee: &mut SectionTablePanel,
        action: SectionTableAction,
        binning: i32,
    ) {
        adaptee.action(action, binning);
    }
}

/// Java final `SectionTablePanel`.  Notifications record calls that cross into the
/// unported manager/dialog/UI harness rather than replacing them with unrelated logic.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SectionTablePanel {
    pub row_list: RowList,
    pub viewport: Viewport,
    pub current_tab: Tab,
    pub mode: i32,
    pub rotating: bool,
    pub last_location: Option<PathBuf>,
    pub sections_expanded: bool,
    pub controls: SectionTableControls,
    pub sample_header_cell: HeaderCell,
    pub rotation_header_cell: HeaderCell,
    pub join_final_header_cell: HeaderCell,
    pub messages: Vec<(String, String)>,
    pub manager_repainted: bool,
    pub dialog_row_changed: bool,
    pub dialog_num_sections: Option<(usize, bool)>,
    pub packed: bool,
    pub selected_file: Option<PathBuf>,
    pub layout: TableLayout,
    pub constraints: TableConstraints,
    pub root_panel_present: bool,
}
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SectionTableControls {
    pub move_up: bool,
    pub move_down: bool,
    pub add: bool,
    pub delete: bool,
    pub get_angles: bool,
    pub invert: bool,
    pub open_3dmod: bool,
    pub expand_sections: bool,
}
impl SectionTablePanel {
    /// Java constructor through `setToolTipText`; layout construction is retained as
    /// source state because Swing is an explicit GUI boundary.
    pub fn new(tab: Tab, table_size: usize) -> Self {
        let mut panel = Self {
            row_list: RowList::default(),
            viewport: Viewport::new(table_size),
            current_tab: tab,
            mode: SETUP_MODE,
            rotating: false,
            last_location: None,
            sections_expanded: false,
            controls: SectionTableControls::default(),
            sample_header_cell: HeaderCell::new("Sample Slices"),
            rotation_header_cell: HeaderCell::new("Rotation Angles"),
            join_final_header_cell: HeaderCell::new("Final"),
            messages: vec![],
            manager_repainted: false,
            dialog_row_changed: false,
            dialog_num_sections: None,
            packed: false,
            selected_file: None,
            layout: TableLayout,
            constraints: TableConstraints {
                fill_both: true,
                ..Default::default()
            },
            root_panel_present: true,
        };
        panel.viewport.init_paging();
        panel.add_table_panel_components();
        panel.create_buttons_panel();
        panel.add_buttons_panel_components();
        panel.add_root_panel_components();
        panel.set_tool_tip_text();
        panel
    }
    pub fn get_focusable_parents(&self) -> [Tab; 3] {
        [Tab::Setup, Tab::Align, Tab::Join]
    }
    pub fn is_setup_tab(&self) -> bool {
        self.current_tab == Tab::Setup
    }
    pub fn is_align_tab(&self) -> bool {
        self.current_tab == Tab::Align
    }
    pub fn is_join_tab(&self) -> bool {
        self.current_tab == Tab::Join
    }
    pub fn is_rejoin_tab(&self) -> bool {
        self.current_tab == Tab::Rejoin
    }
    pub fn add_root_panel_components(&mut self) {
        if !self.is_align_tab() {
            self.add_buttons_panel_components();
        }
    }
    pub fn add_table_panel_components(&mut self) {
        if self.is_setup_tab() {
            self.add_setup_table_panel_components();
        } else if self.is_align_tab() {
            self.add_align_table_panel_components();
        } else if self.is_join_tab() || self.is_rejoin_tab() {
            self.add_join_table_panel_components();
        }
    }
    pub fn add_setup_table_panel_components(&mut self) {}
    pub fn get_mode(&self) -> i32 {
        self.mode
    }
    pub fn add_align_table_panel_components(&mut self) {}
    pub fn add_join_table_panel_components(&mut self) {
        if self.has_rotated_section() {
            self.join_final_header_cell.set_text("Final");
        }
    }
    pub fn has_rotated_section(&self) -> bool {
        self.row_list.has_rotated_section()
    }
    pub fn create_buttons_panel(&mut self) {}
    pub fn add_buttons_panel_components(&mut self) {}
    pub fn msg_viewport_paged(&mut self) {
        self.display_cur_tab();
        self.manager_repainted = true;
    }
    pub fn display_cur_tab(&mut self) {
        self.add_root_panel_components();
        self.add_buttons_panel_components();
        self.add_table_panel_components();
        self.row_list.display_cur_tab(&self.viewport);
    }
    pub fn size(&self) -> usize {
        self.row_list.size()
    }
    pub fn get_chunk_sizes(&self) -> String {
        self.row_list
            .list
            .iter()
            .map(|row| (row.data.join_final_end - row.data.join_final_start + 1).to_string())
            .collect::<Vec<_>>()
            .join(",")
    }
    pub fn set_inverted(&mut self, inverted: &[bool]) {
        let count = self.row_list.set_inverted(inverted);
        if count > self.row_list.size() / 2 {
            self.messages.push(("Join Warning".into(), format!("Most of the sections in this join will be inverted.  {INVERTED_WARNING}  If you don't want these inversions, push the \"Change Setup\" button and then push the \"Invert Table\" button.")));
        }
    }
    pub fn highlight(&mut self, _highlight: bool) {
        self.set_mode();
    }
    pub fn highlight_down_action_performed(&mut self) {
        let highlighted_index = self.row_list.get_highlighted_index();
        let adjust =
            highlighted_index >= 0 && self.viewport.in_viewport(highlighted_index as usize);
        let index = self.row_list.highlight_down();
        if index != -1 && adjust && !self.viewport.in_viewport(index as usize) {
            if index == 0 {
                self.viewport.home_button_action();
            } else {
                self.viewport.down_button_action();
            }
        }
    }
    pub fn highlight_up_action_performed(&mut self) {
        let highlighted_index = self.row_list.get_highlighted_index();
        let adjust =
            highlighted_index >= 0 && self.viewport.in_viewport(highlighted_index as usize);
        let index = self.row_list.highlight_up();
        if index != -1 && adjust && !self.viewport.in_viewport(index as usize) {
            if index as usize == self.row_list.size() - 1 {
                self.viewport.end_button_action(self.row_list.size());
            } else {
                self.viewport.up_button_action();
            }
        }
    }
    pub fn set_mode(&mut self) {
        self.set_mode_to(self.mode);
    }
    pub fn set_mode_to(&mut self, mode: i32) {
        self.mode = mode;
        match mode {
            SAMPLE_PRODUCED_MODE => {
                self.controls.add = false;
                self.controls.move_up = false;
                self.controls.move_down = false;
                self.controls.delete = false;
                self.controls.get_angles = false;
                self.controls.invert = false;
            }
            SETUP_MODE | SAMPLE_NOT_PRODUCED_MODE | CHANGING_SAMPLE_MODE => {
                if !self.rotating {
                    self.controls.add = true;
                    self.controls.invert = true;
                }
            }
            _ => panic!("mode={mode}"),
        }
        self.enable_row_buttons(self.row_list.get_highlighted_index());
        self.row_list.set_mode(mode);
    }
    pub fn set_join_final_start_highlight(&mut self, highlight: bool) {
        self.row_list.set_join_final_start_highlight(highlight);
    }
    pub fn set_join_final_end_highlight(&mut self, highlight: bool) {
        self.row_list.set_join_final_end_highlight(highlight);
    }
    pub fn enable_row_buttons(&mut self, index: isize) {
        let size = self.row_list.size();
        if size == 0 {
            self.controls.open_3dmod = false;
            self.controls.expand_sections = false;
            if self.mode != SAMPLE_PRODUCED_MODE {
                self.controls.move_up = false;
                self.controls.move_down = false;
                self.controls.delete = false;
                self.controls.get_angles = false;
            }
            return;
        }
        self.controls.open_3dmod = index > -1;
        self.controls.expand_sections = true;
        if self.mode != SAMPLE_PRODUCED_MODE {
            self.controls.move_up = index > 0;
            self.controls.move_down = index > -1 && index < size as isize - 1;
            self.controls.delete = index > -1;
            self.controls.get_angles = index > -1;
        }
    }
    pub fn expand_global(&mut self) {}
    pub fn expand(&mut self, expanded: bool) {
        self.sections_expanded = expanded;
        self.row_list.expand(expanded);
    }
    pub fn get_table_layout(&self) -> &TableLayout {
        &self.layout
    }
    pub fn enable_add_section(&mut self) {
        self.rotating = false;
        self.set_mode();
    }
    pub fn equals(&self, metadata: &JoinMetaData) -> bool {
        metadata
            .section_table_data
            .as_ref()
            .is_some_and(|data| self.row_list.equals(data))
    }
    pub fn equals_sample(&self, metadata: &JoinMetaData) -> bool {
        metadata
            .section_table_data
            .as_ref()
            .is_some_and(|data| self.row_list.equals_sample(data))
    }
    pub fn get_table_constraints(&self) -> &TableConstraints {
        &self.constraints
    }
    pub fn move_section_up(&mut self) {
        let index = self.row_list.get_highlighted_index();
        if index < 0 {
            return;
        }
        let index = index as usize;
        if index == 0 {
            self.messages.push((
                "Wrong Row".into(),
                "Can't move the row up.  Its at the top.".into(),
            ));
            return;
        }
        self.row_list.move_section_up(index);
        self.viewport.adjust_viewport(index - 1);
        self.row_list.remove_rows();
        self.row_list.display_rows(&self.viewport);
        self.row_list.renumber_table(index - 1);
        self.row_list.configure_rows();
        self.enable_row_buttons(index as isize - 1);
        self.dialog_row_changed = true;
        self.manager_repainted = true;
    }
    pub fn move_section_down(&mut self) {
        let index = self.row_list.get_highlighted_index();
        if index < 0 {
            return;
        }
        let index = index as usize;
        if index == self.row_list.size() - 1 {
            self.messages.push((
                "Wrong Row".into(),
                "Can't move the row down.  Its at the bottom.".into(),
            ));
            return;
        }
        self.row_list.move_section_down(index);
        self.viewport.adjust_viewport(index + 1);
        self.row_list.remove_rows();
        self.row_list.display_rows(&self.viewport);
        self.row_list.renumber_table(index);
        self.row_list.configure_rows();
        self.enable_row_buttons(index as isize + 1);
        self.dialog_row_changed = true;
        self.manager_repainted = true;
    }
    /// Java chooser action; selected-file/valid-working-directory/MRC-header operations are GUI and manager boundaries.
    pub fn add_section_action(
        &mut self,
        working_directory_valid: bool,
        header_read: bool,
        n_rows: i32,
        n_sections: i32,
        rotate_accepted: bool,
    ) {
        if !working_directory_valid {
            self.messages.push((
                "Unable to Add Section".into(),
                "Invalid working directory".into(),
            ));
            return;
        }
        let Some(path) = self.selected_file.clone() else {
            return;
        };
        self.last_location = path.parent().map(Path::to_path_buf);
        if self.row_list.is_duplicate(&path) {
            self.messages.push((
                "Add Section Failed".into(),
                format!("The file, {}, is already in the table.", path.display()),
            ));
            return;
        }
        if !header_read {
            self.messages
                .push(("System Error".into(), "File does not exist".into()));
            return;
        }
        self.rotating = true;
        self.controls.add = false;
        self.controls.invert = false;
        if n_rows < n_sections && rotate_accepted {
            return;
        }
        self.add_section(path);
        self.packed = true;
    }
    pub fn read_header(&mut self, read: bool, dimensions_known: bool, user_accepts: bool) -> bool {
        if !read {
            self.messages
                .push(("System Error".into(), "File does not exist".into()));
            return false;
        }
        dimensions_known || user_accepts
    }
    pub fn is_duplicate(&mut self, section: &Path) -> bool {
        if self.row_list.is_duplicate(section) {
            self.messages.push((
                "Add Section Failed".into(),
                format!("The file, {}, is already in the table.", section.display()),
            ));
            true
        } else {
            false
        }
    }
    pub fn add_section(&mut self, tomogram: impl Into<PathBuf>) {
        self.rotating = false;
        self.set_mode();
        let tomogram = tomogram.into();
        if !tomogram.exists() {
            self.messages.push((
                "File Error".into(),
                format!("{} does not exist.", tomogram.display()),
            ));
            return;
        }
        if !tomogram.is_file() {
            self.messages.push((
                "File Error".into(),
                format!("{} is not a file.", tomogram.display()),
            ));
            return;
        }
        let index = self
            .row_list
            .add(tomogram, self.sections_expanded, self.mode);
        self.viewport.adjust_viewport(index);
        self.row_list.remove_rows();
        self.row_list.display_rows(&self.viewport);
        self.row_list.configure_rows();
        self.dialog_num_sections = Some((self.row_list.size(), false));
        self.dialog_row_changed = true;
        self.manager_repainted = true;
    }
    pub fn delete_section(&mut self, accepted: bool) {
        let index = self.row_list.get_highlighted_index();
        if index < 0 || !accepted {
            return;
        }
        let index = index as usize;
        self.row_list.delete_section(index);
        self.row_list.remove_rows();
        self.viewport.adjust_viewport(index);
        self.row_list.display_rows(&self.viewport);
        self.row_list.renumber_table(index);
        self.row_list.configure_rows();
        self.dialog_num_sections = Some((self.row_list.size(), false));
        self.enable_row_buttons(-1);
        self.dialog_row_changed = true;
        self.manager_repainted = true;
    }
    pub fn delete_sections(&mut self) {
        self.row_list.delete_sections();
        self.manager_repainted = true;
    }
    pub fn imod_section(&mut self, binning: i32) {
        let index = self.row_list.get_highlighted_index();
        if index < 0 {
            return;
        }
        let is_setup_tab = self.is_setup_tab();
        let row = self.row_list.get_mut(index as usize).unwrap();
        if is_setup_tab {
            row.imod_open_setup_section_file(binning);
        } else {
            row.imod_open_join_section_file(binning);
        }
    }
    pub fn imod_get_angles(&mut self) {
        if let Some(row) = self
            .row_list
            .get_mut(self.row_list.get_highlighted_index().max(0) as usize)
        {
            if row.imod_get_angles() {
                self.manager_repainted = true;
            }
        }
    }
    pub fn invert_table(&mut self) {
        self.row_list.invert_table();
        self.row_list.remove_rows();
        self.row_list.display_rows(&self.viewport);
        self.row_list.configure_rows();
        self.enable_row_buttons(self.row_list.get_highlighted_index());
        self.dialog_row_changed = true;
        self.manager_repainted = true;
    }
    pub fn get_meta_data(&self, metadata: &mut JoinMetaData) -> bool {
        self.row_list.get_meta_data(metadata)
    }
    pub fn validate_makejoincom(&self) -> bool {
        self.row_list.validate_makejoincom()
    }
    pub fn validate_finishjoin(&self) -> bool {
        self.row_list.validate_finishjoin()
    }
    pub fn set_meta_data(&mut self, metadata: &JoinMetaData) {
        let Some(data) = metadata.section_table_data.as_deref() else {
            return;
        };
        self.row_list.set_meta_data(data, self.mode, &self.viewport);
        self.row_list.configure_rows();
        self.dialog_num_sections = Some((self.row_list.size(), true));
        self.manager_repainted = true;
    }
    pub fn get_sample_header_cell(&self) -> &HeaderCell {
        &self.sample_header_cell
    }
    pub fn get_rotation_header_cell(&self) -> &HeaderCell {
        &self.rotation_header_cell
    }
    pub fn get_join_final_header_cell(&self) -> &HeaderCell {
        &self.join_final_header_cell
    }
    pub fn get_invalid_reason(&self) -> Option<String> {
        self.row_list.get_invalid_reason()
    }
    pub fn get_x_max(&self) -> i32 {
        self.row_list.get_x_max()
    }
    pub fn get_y_max(&self) -> i32 {
        self.row_list.get_y_max()
    }
    pub fn get_z_max(&self) -> i32 {
        self.row_list.get_z_max()
    }
    pub fn remove_cell(&mut self) {}
    pub fn pop_up_context_menu(&mut self) {}
    pub fn get_container(&self) -> bool {
        self.root_panel_present
    }
    pub fn get_root_panel(&self) -> bool {
        self.root_panel_present
    }
    pub fn action(&mut self, action: SectionTableAction, binning: i32) {
        match action {
            SectionTableAction::MoveSectionUp => self.move_section_up(),
            SectionTableAction::MoveSectionDown => self.move_section_down(),
            SectionTableAction::AddSection => self.add_section_action(true, true, 0, 0, false),
            SectionTableAction::DeleteSection => self.delete_section(true),
            SectionTableAction::GetAngles => self.imod_get_angles(),
            SectionTableAction::InvertTable => self.invert_table(),
            SectionTableAction::OpenIn3dmod => self.imod_section(binning),
        }
    }
    pub fn synchronize(&mut self, previous: Tab, current: Tab) {
        if self.row_list.size() == 0 {
            return;
        }
        if matches!(current, Tab::Join | Tab::Rejoin | Tab::Model) {
            self.row_list.synchronize_setup_to_join();
        } else if matches!(previous, Tab::Join | Tab::Rejoin) {
            self.row_list.synchronize_join_to_setup();
        }
    }
    pub fn set_tool_tip_text(&mut self) {
        self.sample_header_cell
            .set_tool_tip_text("The slices to be used in the sample.");
        self.rotation_header_cell
            .set_tool_tip_text("The rotation in X, Y, and Z of each section.");
        self.join_final_header_cell.set_tool_tip_text("Enter starting and ending Z values to trim each section or rotated section in the joined tomogram.");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn panel() -> SectionTablePanel {
        SectionTablePanel::new(Tab::Setup, 2)
    }
    fn file(name: &str) -> PathBuf {
        std::env::temp_dir().join(name)
    }
    #[test]
    fn row_moves_invert_and_chunk_sizes_follow_java_ordering() {
        let mut table = panel();
        table.row_list.add(file("a"), false, SETUP_MODE);
        table.row_list.add(file("b"), false, SETUP_MODE);
        table.row_list.list[0].highlighted = true;
        table.row_list.list[0].data.join_final_start = 1;
        table.row_list.list[0].data.join_final_end = 3;
        table.row_list.list[1].data.join_final_start = 4;
        table.row_list.list[1].data.join_final_end = 7;
        table.move_section_down();
        assert_eq!(table.row_list.list[1].data.setup_section, file("a"));
        assert_eq!(table.get_chunk_sizes(), "4,3");
        table.invert_table();
        assert_eq!(table.row_list.list[0].data.setup_section, file("a"));
    }
    #[test]
    fn mode_and_highlight_enablements_match_source_cases() {
        let mut table = panel();
        table.set_mode_to(SAMPLE_PRODUCED_MODE);
        assert!(!table.controls.add);
        table.row_list.add(file("a"), false, SETUP_MODE);
        table.row_list.list[0].highlighted = true;
        table.set_mode_to(SETUP_MODE);
        assert!(table.controls.add);
        assert!(table.controls.open_3dmod);
        assert!(table.controls.delete);
        assert!(!table.controls.move_up);
    }
    #[test]
    fn metadata_round_trip_and_tab_synchronization_preserve_row_behavior() {
        let mut table = panel();
        let data = vec![SectionTableRowData {
            row_index: 0,
            setup_section: file("a"),
            x_max: 10,
            y_max: 11,
            z_max: 12,
            valid: true,
            ..Default::default()
        }];
        table.set_meta_data(&JoinMetaData {
            section_table_data: Some(data.clone()),
        });
        let mut out = JoinMetaData::default();
        assert!(table.get_meta_data(&mut out));
        assert_eq!(out.section_table_data, Some(data));
        table.synchronize(Tab::Setup, Tab::Join);
        assert!(table.row_list.list[0].setup_to_join_synchronized);
        table.synchronize(Tab::Join, Tab::Setup);
        assert!(table.row_list.list[0].join_to_setup_synchronized);
    }
}
