//! `IMOD/Etomo/src/etomo/ui/swing/BoundaryTable.java`.
//!
//! Swing construction and GridBag insertion, `JoinManager`, `JoinDialog`, and repaint
//! are boundaries.  The source table's headers, `RowList`, paging, reset and transfer
//! behavior are retained directly; `BoundaryRow.java` remains its own source unit.
#![allow(dead_code)]

use super::boundary_row::{
    BoundaryRow, BoundaryRowMetaData, BoundaryRowScreenState, BoundaryTable as BoundaryRowTable,
    XfjointomoLog,
};
use super::section_table_panel::{HeaderCell, Tab, Viewport};

pub const TABLE_LABEL: &str = "Boundary Table";

/// Java private static final `BoundaryTable.RowList`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RowList {
    pub list: Vec<BoundaryRow>,
}
impl RowList {
    pub fn clear(&mut self, _viewport: &Viewport) {
        self.list.clear();
    }
    pub fn size(&self) -> usize {
        self.list.len()
    }
    pub fn remove_display(&mut self) {
        for row in &mut self.list {
            row.remove_display();
        }
    }
    pub fn display(&mut self, tab: Tab, viewport: &Viewport) {
        for (index, row) in self.list.iter_mut().enumerate() {
            row.display(index, viewport, tab);
        }
    }
    pub fn set_xfjointomo_result(&mut self, log: &XfjointomoLog) {
        for row in &mut self.list {
            row.set_xfjointomo_result(log);
        }
    }
    pub fn get_screen_state(&self, state: &mut BoundaryRowScreenState) {
        for row in &self.list {
            row.get_screen_state(state);
        }
    }
    pub fn get_meta_data(&self, data: &mut BoundaryRowMetaData) {
        for row in &self.list {
            row.get_meta_data(data);
        }
    }
    pub fn add(
        &mut self,
        size: usize,
        data: &BoundaryRowMetaData,
        state: &BoundaryRowScreenState,
        table: BoundaryRowTable,
    ) {
        for index in 0..size {
            let mut row = BoundaryRow::new(index as i32 + 1, data, state, table.clone());
            row.set_names();
            self.list.push(row);
        }
    }
    pub fn get(&self, index: usize) -> Option<&BoundaryRow> {
        self.list.get(index)
    }
}

/// Java final `BoundaryTable`.
#[derive(Clone, Debug, PartialEq)]
pub struct BoundaryTable {
    pub header1_boundaries: HeaderCell,
    pub header1_sections: HeaderCell,
    pub header1_best_gap: HeaderCell,
    pub header1_error: HeaderCell,
    pub header1_original: HeaderCell,
    pub header1_adjusted: HeaderCell,
    pub header2_boundaries: HeaderCell,
    pub header2_sections: HeaderCell,
    pub header2_best_gap: HeaderCell,
    pub header2_mean_error: HeaderCell,
    pub header2_max_error: HeaderCell,
    pub header2_original_end: HeaderCell,
    pub header2_original_start: HeaderCell,
    pub header2_adjusted_end: HeaderCell,
    pub header2_adjusted_start: HeaderCell,
    pub header3_sections: HeaderCell,
    pub header3_best_gap: HeaderCell,
    pub header3_original_end: HeaderCell,
    pub header3_original_start: HeaderCell,
    pub header3_adjusted_end: HeaderCell,
    pub header3_adjusted_start: HeaderCell,
    pub row_list: RowList,
    pub viewport: Viewport,
    pub row_change: bool,
    pub tab: Option<Tab>,
    /// Java `getFocusableParents`' two tab components.
    pub focusable_parents: [bool; 2],
    /// Native Swing panels/layout remain boundary-owned.
    pub root_panel_present: bool,
    pub table_panel_present: bool,
    /// `manager.getMainPanel().repaint()` notification.
    pub repaint_requested: bool,
    /// Source header insertion order at the Swing layout boundary.
    pub displayed_headers: Vec<String>,
}
impl BoundaryTable {
    /// Java package-private constructor after the manager/dialog inputs cross boundaries.
    pub fn new(table_size: usize) -> Self {
        let mut table = Self {
            header1_boundaries: HeaderCell::new("Boundaries"),
            header1_sections: HeaderCell::new("Sections"),
            header1_best_gap: HeaderCell::new("Best"),
            header1_error: HeaderCell::new("Error"),
            header1_original: HeaderCell::new("Original"),
            header1_adjusted: HeaderCell::new("Adjusted"),
            header2_boundaries: HeaderCell::default(),
            header2_sections: HeaderCell::default(),
            header2_best_gap: HeaderCell::new("Gap"),
            header2_mean_error: HeaderCell::new("Mean"),
            header2_max_error: HeaderCell::new("Max"),
            header2_original_end: HeaderCell::default(),
            header2_original_start: HeaderCell::default(),
            header2_adjusted_end: HeaderCell::default(),
            header2_adjusted_start: HeaderCell::default(),
            header3_sections: HeaderCell::default(),
            header3_best_gap: HeaderCell::default(),
            header3_original_end: HeaderCell::new("End"),
            header3_original_start: HeaderCell::new("Start"),
            header3_adjusted_end: HeaderCell::new("End"),
            header3_adjusted_start: HeaderCell::new("Start"),
            row_list: RowList::default(),
            viewport: Viewport::new(table_size),
            row_change: true,
            tab: None,
            focusable_parents: [true, true],
            root_panel_present: true,
            table_panel_present: true,
            repaint_requested: false,
            displayed_headers: Vec::new(),
        };
        table.viewport.init_paging();
        table.set_tool_tip_text();
        table
    }
    pub fn get_focusable_parents(&self) -> &[bool; 2] {
        &self.focusable_parents
    }
    pub fn set_xfjointomo_result(&mut self, log: &XfjointomoLog) {
        self.row_list.set_xfjointomo_result(log);
    }
    pub fn display(
        &mut self,
        tab: Tab,
        section_table_size: usize,
        data: &BoundaryRowMetaData,
        state: &BoundaryRowScreenState,
    ) {
        self.display_force(false, tab, section_table_size, data, state);
    }
    pub fn msg_viewport_paged(
        &mut self,
        tab: Tab,
        section_table_size: usize,
        data: &BoundaryRowMetaData,
        state: &BoundaryRowScreenState,
    ) {
        self.display_force(true, tab, section_table_size, data, state);
    }
    pub fn size(&self) -> usize {
        self.row_list.size()
    }
    pub fn display_force(
        &mut self,
        force: bool,
        tab: Tab,
        section_table_size: usize,
        data: &BoundaryRowMetaData,
        state: &BoundaryRowScreenState,
    ) {
        let old_tab = self.tab;
        self.tab = Some(tab);
        if !force && old_tab == self.tab && !self.row_change {
            return;
        }
        self.row_list.remove_display();
        self.remove_all();
        self.add_header(tab);
        self.add_rows(section_table_size, data, state);
        self.repaint_requested = true;
    }
    pub fn msg_row_change(
        &mut self,
        state: &mut BoundaryRowScreenState,
        data: &mut BoundaryRowMetaData,
    ) {
        self.row_change = true;
        BoundaryRow::reset_screen_state(state);
        BoundaryRow::reset_meta_data(data);
    }
    pub fn get_screen_state(&self, state: &mut BoundaryRowScreenState) {
        BoundaryRow::reset_screen_state(state);
        self.row_list.get_screen_state(state);
    }
    /// The sole real `BoundaryTable` member read from `BoundaryRow.setNames`.
    pub fn get_adjusted_header_cell(&self) -> &HeaderCell {
        &self.header1_adjusted
    }
    pub fn get_meta_data(&self, data: &mut BoundaryRowMetaData) {
        BoundaryRow::reset_meta_data(data);
        self.row_list.get_meta_data(data);
    }
    pub fn get_container(&self) -> bool {
        self.root_panel_present
    }
    pub fn add_header(&mut self, tab: Tab) {
        if tab == Tab::Model {
            self.add_model_header();
        } else if tab == Tab::Rejoin {
            self.add_rejoin_header();
        }
    }
    pub fn add_model_header(&mut self) {
        self.displayed_headers.extend([
            self.header1_boundaries.text.clone(),
            self.header1_best_gap.text.clone(),
            self.header1_error.text.clone(),
            self.header2_boundaries.text.clone(),
            self.header2_best_gap.text.clone(),
            self.header2_mean_error.text.clone(),
            self.header2_max_error.text.clone(),
        ]);
    }
    pub fn add_rejoin_header(&mut self) {
        self.displayed_headers.extend([
            self.header1_sections.text.clone(),
            self.header1_original.text.clone(),
            self.header1_best_gap.text.clone(),
            self.header1_adjusted.text.clone(),
            self.header2_sections.text.clone(),
            self.header2_original_end.text.clone(),
            self.header2_original_start.text.clone(),
            self.header2_best_gap.text.clone(),
            self.header2_adjusted_end.text.clone(),
            self.header2_adjusted_start.text.clone(),
            self.header3_sections.text.clone(),
            self.header3_original_end.text.clone(),
            self.header3_original_start.text.clone(),
            self.header3_best_gap.text.clone(),
            self.header3_adjusted_end.text.clone(),
            self.header3_adjusted_start.text.clone(),
        ]);
    }
    pub fn set_tool_tip_text(&mut self) {
        let text = "Boundaries between sections.";
        self.header1_boundaries.set_tool_tip_text(text);
        self.header2_boundaries.set_tool_tip_text(text);
        let text = "The pairs of sections which define each boundary.";
        self.header1_sections.set_tool_tip_text(text);
        self.header2_sections.set_tool_tip_text(text);
        self.header3_sections.set_tool_tip_text(text);
        let text = "Describes how the final start and end values will change when the join is recreated, with a positive gap adding slices and a negative gap removing slices at the corresponding boundary.";
        self.header1_best_gap.set_tool_tip_text(text);
        self.header2_best_gap.set_tool_tip_text(text);
        self.header3_best_gap.set_tool_tip_text(text);
        self.header1_error.set_tool_tip_text("Deviations between transformed points extrapolated from above and below the corresponding boundary.");
        self.header2_mean_error
            .set_tool_tip_text("Mean deviations.");
        self.header2_max_error
            .set_tool_tip_text("Maximum deviations.");
        let text = "End and start values used to create the original join.";
        self.header1_original.set_tool_tip_text(text);
        self.header2_original_end.set_tool_tip_text(text);
        self.header2_original_start.set_tool_tip_text(text);
        self.header3_original_end
            .set_tool_tip_text("End values used to create the original join.");
        self.header3_original_start
            .set_tool_tip_text("Start values used to create the original join.");
        let text = "End and start values which will be used to create the new join.";
        self.header1_adjusted.set_tool_tip_text(text);
        self.header2_adjusted_end.set_tool_tip_text(text);
        self.header2_adjusted_start.set_tool_tip_text(text);
        self.header3_adjusted_end
            .set_tool_tip_text("End values which will be used to create the new join.");
        self.header3_adjusted_start
            .set_tool_tip_text("Start values which will be used to create the new join.");
    }
    pub fn add_rows(
        &mut self,
        section_table_size: usize,
        data: &BoundaryRowMetaData,
        state: &BoundaryRowScreenState,
    ) {
        if self.row_change {
            self.row_change = false;
            self.row_list.clear(&self.viewport);
            self.row_list.add(
                section_table_size.saturating_sub(1),
                data,
                state,
                BoundaryRowTable {
                    adjusted_header_cell: self.header1_adjusted.clone(),
                },
            );
        }
        if let Some(tab) = self.tab {
            self.row_list.display(tab, &self.viewport);
        }
    }
    pub fn remove_all(&mut self) {
        self.displayed_headers.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::etomo::ui::swing::section_table_panel::SectionTableRowData;
    use std::collections::BTreeMap;
    fn data() -> BoundaryRowMetaData {
        BoundaryRowMetaData {
            section_table_data: vec![
                SectionTableRowData {
                    join_final_end: 10,
                    join_final_start: 1,
                    z_max: 20,
                    ..Default::default()
                },
                SectionTableRowData {
                    join_final_end: 11,
                    join_final_start: 2,
                    z_max: 20,
                    ..Default::default()
                },
                SectionTableRowData {
                    join_final_end: 12,
                    join_final_start: 3,
                    z_max: 20,
                    ..Default::default()
                },
            ],
            ..Default::default()
        }
    }
    #[test]
    fn headers_and_canonical_rows_follow_source_tabs() {
        let mut table = BoundaryTable::new(2);
        let data = data();
        let state = BoundaryRowScreenState::default();
        table.display(Tab::Model, 3, &data, &state);
        assert_eq!(
            table.displayed_headers,
            ["Boundaries", "Best", "Error", "", "Gap", "Mean", "Max"]
        );
        assert_eq!(table.size(), 2);
        assert!(table.row_list.list.iter().all(|row| row.model_displayed));
        table.display(Tab::Rejoin, 3, &data, &state);
        assert_eq!(table.displayed_headers.len(), 16);
        assert!(table.row_list.list.iter().all(|row| row.rejoin_displayed));
    }
    #[test]
    fn reset_and_metadata_round_trip_use_boundary_row_source_types() {
        let mut table = BoundaryTable::new(2);
        let mut data = data();
        data.boundary_row_end.insert(1, 7);
        let mut state = BoundaryRowScreenState {
            best_gap: BTreeMap::from([(1, 5.0)]),
            mean_error: BTreeMap::from([(1, 6.0)]),
            max_error: BTreeMap::from([(1, 7.0)]),
        };
        table.msg_row_change(&mut state, &mut data);
        assert_eq!(state, BoundaryRowScreenState::default());
        assert!(data.boundary_row_end.is_empty());
        table.display(Tab::Rejoin, 3, &data, &state);
        table.row_list.list[0].adjusted_end.set_value(13);
        table.get_meta_data(&mut data);
        assert_eq!(data.boundary_row_end.get(&1), Some(&13));
    }
}
