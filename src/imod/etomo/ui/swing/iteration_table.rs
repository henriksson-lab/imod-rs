//! `IMOD/Etomo/src/etomo/ui/swing/IterationTable.java`.
//!
//! Swing panel/layout construction, `UIHarness`, autodoc tooltips, and the owning
//! `PeetDialog` remain explicit boundaries.  The table's complete row-list and action
//! state live here; `IterationRow.java` is represented by its direct table-facing
//! state until that separate source unit is translated.
#![allow(dead_code)]

use std::collections::BTreeMap;

use super::iteration_parent::IterationParent;

pub const D_PHI_D_THETA_D_PSI_HEADER1: &str = "Angular Search Range";
pub const INCR_HEADER3: &str = "Step";
pub const SEARCH_RADIUS_HEADER1: &str = "Search";
pub const SEARCH_RADIUS_HEADER2: &str = "Distance";
pub const LABEL: &str = "Iteration Table";
pub const MAX_HEADER3: &str = "Max";
pub const HICUTOFF_HEADER1: &str = "Low-pass";
pub const HICUTOFF_HEADER2: &str = "Filter";
pub const HICUTOFF_CUTOFF_HEADER3: &str = "Cutoff";
pub const HICUTOFF_SIGMA_HEADER3: &str = "Sigma";
pub const LOWCUTOFF_HEADER1: &str = "High-pass";
pub const LOWCUTOFF_HEADER2: &str = "Filter";
pub const LOWCUTOFF_CUTOFF_HEADER3: &str = "Cutoff";
pub const LOWCUTOFF_SIGMA_HEADER3: &str = "Sigma";
pub const REF_THRESHOLD_HEADER1: &str = "Ref";
pub const REF_THRESHOLD_HEADER2: &str = "Threshold";
pub const DUPLICATE_TOLERANCE_HEADER1: &str = "Duplicate";
pub const DUPLICATE_TOLERANCE_HEADER2: &str = "Tolerance";
pub const DUPLICATE_SHIFT_TOLERANCE_HEADER3: &str = "Shift";
pub const DUPLICATE_ANGULAR_TOLERANCE_HEADER3: &str = "Angle";
pub const LOW_CUTOFF_DEFAULT: &str = "0";
pub const LOW_CUTOFF_SIGMA_DEFAULT: &str = "0.05";

/// Java `HeaderCell`; Swing component creation is a boundary.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct HeaderCell {
    pub text: String,
    pub width: Option<usize>,
    pub visible: bool,
    pub tooltip: Option<String>,
}
impl HeaderCell {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            visible: true,
            ..Self::default()
        }
    }
    pub fn with_width(text: impl Into<String>, width: usize) -> Self {
        Self {
            text: text.into(),
            width: Some(width),
            visible: true,
            tooltip: None,
        }
    }
}

/// The fields `IterationRow.java` exposes directly to this table.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct IterationRow {
    pub index: usize,
    pub highlighted: bool,
    pub displayed: bool,
    pub low_cutoff_rows_visible: bool,
    pub d_theta_enabled: bool,
    pub d_psi_enabled: bool,
    pub duplicate_tolerance_enabled: bool,
    pub values: BTreeMap<String, String>,
}
impl IterationRow {
    pub fn new(index: usize, is_low_cutoff: bool) -> Self {
        Self {
            index,
            low_cutoff_rows_visible: is_low_cutoff,
            d_theta_enabled: true,
            d_psi_enabled: true,
            duplicate_tolerance_enabled: false,
            ..Self::default()
        }
    }
    pub fn set_low_cutoff_sigma(&mut self, value: impl Into<String>) {
        self.values.insert("lowCutoffSigma".into(), value.into());
    }
    pub fn set_visible_low_cutoff_rows(&mut self, input: bool) {
        self.low_cutoff_rows_visible = input;
    }
    pub fn update_display(&mut self, sample_sphere: bool, remove_duplicates: bool) {
        if self.index == 0 {
            self.d_theta_enabled = !sample_sphere;
            self.d_psi_enabled = !sample_sphere;
        }
        self.duplicate_tolerance_enabled = remove_duplicates;
    }
    pub fn check_low_cutoff_backwards_compatibility(&self) -> bool {
        self.values
            .get("lowCutoffCutoff")
            .is_some_and(|v| v == LOW_CUTOFF_DEFAULT)
            && self
                .values
                .get("lowCutoffSigma")
                .is_some_and(|v| v == LOW_CUTOFF_SIGMA_DEFAULT)
    }
    pub fn set_index(&mut self, index: usize) {
        self.index = index;
    }
    pub fn remove(&mut self) {
        self.displayed = false;
    }
    pub fn display(&mut self) {
        self.displayed = true;
    }
    pub fn validate_run(&self, low_cutoff: bool) -> bool {
        let required = [
            "dPhiMax",
            "dPhiIncrement",
            "searchRadius",
            "hiCutoff",
            "hiCutoffSigma",
            "refThreshold",
        ];
        if required
            .into_iter()
            .any(|key| self.values.get(key).is_none_or(|v| v.trim().is_empty()))
        {
            return false;
        }
        if low_cutoff
            && ["lowCutoff", "lowCutoffSigma"]
                .into_iter()
                .any(|key| self.values.get(key).is_none_or(|v| v.trim().is_empty()))
        {
            return false;
        }
        true
    }
}

/// Java `MatlabParam.Iteration` data touched by this unit.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Iteration {
    pub values: BTreeMap<String, String>,
}
/// Java `MatlabParam`, restricted to direct `IterationTable` accesses.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MatlabParam {
    pub iterations: Vec<Iteration>,
    pub flg_remove_duplicates: bool,
    pub flg_strict_search_limits: bool,
}
impl MatlabParam {
    pub fn set_iteration_list_size(&mut self, size: usize) {
        self.iterations.resize_with(size, Iteration::default);
    }
    pub fn get_iteration_list_size(&self) -> usize {
        self.iterations.len()
    }
}
/// Java `PeetMetaData` / `ConstPeetMetaData` fields reached by this source unit.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PeetMetaData {
    pub low_cutoff: bool,
    pub low_cutoff_values: Vec<BTreeMap<String, String>>,
}
pub type ConstPeetMetaData = PeetMetaData;

/// Java private static inner `RowList`.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RowList {
    pub list: Vec<IterationRow>,
    pub meta_data: Option<ConstPeetMetaData>,
}
impl RowList {
    pub fn add(&mut self, is_low_cutoff: bool) -> &mut IterationRow {
        let index = self.list.len();
        self.list.push(IterationRow::new(index, is_low_cutoff));
        self.list.last_mut().unwrap()
    }
    pub fn get_parameters(&self, matlab: &mut MatlabParam, low_cutoff: bool) {
        matlab.set_iteration_list_size(self.list.len());
        for (index, row) in self.list.iter().enumerate() {
            matlab.iterations[index].values = row.values.clone();
            if !low_cutoff {
                matlab.iterations[index]
                    .values
                    .insert("lowCutoffCutoff".into(), LOW_CUTOFF_DEFAULT.into());
                matlab.iterations[index]
                    .values
                    .insert("lowCutoffSigma".into(), LOW_CUTOFF_SIGMA_DEFAULT.into());
            }
        }
    }
    pub fn get_peet_parameters(&self, meta_data: &mut PeetMetaData) {
        meta_data.low_cutoff_values = self.list.iter().map(|row| row.values.clone()).collect();
    }
    pub fn set_parameters(&mut self, meta_data: &ConstPeetMetaData) {
        self.meta_data = Some(meta_data.clone());
    }
    pub fn delete(&mut self, row_index: usize) -> usize {
        self.list.remove(row_index);
        for (index, row) in self.list.iter_mut().enumerate().skip(row_index) {
            row.set_index(index);
        }
        row_index
    }
    pub fn validate_run(&self, is_low_cutoff: bool) -> bool {
        !self.list.is_empty() && self.list.iter().all(|row| row.validate_run(is_low_cutoff))
    }
    pub fn update_display(&mut self, sample_sphere: bool, remove_duplicates: bool) {
        for row in &mut self.list {
            row.update_display(sample_sphere, remove_duplicates);
        }
    }
    pub fn remove(&mut self) {
        for row in &mut self.list {
            row.remove();
        }
    }
    pub fn copy(&mut self, row_index: usize, is_low_cutoff: bool) {
        let mut copy = self.list[row_index].clone();
        copy.index = self.list.len();
        copy.highlighted = false;
        copy.low_cutoff_rows_visible = is_low_cutoff;
        self.list.push(copy);
    }
    pub fn size(&self) -> usize {
        self.list.len()
    }
    pub fn display(&mut self) {
        for row in &mut self.list {
            row.display();
        }
    }
    pub fn get_row(&self, index: usize) -> Option<&IterationRow> {
        self.list.get(index)
    }
    pub fn get_row_mut(&mut self, index: usize) -> Option<&mut IterationRow> {
        self.list.get_mut(index)
    }
    pub fn get_highlighted_row(&self) -> Option<usize> {
        self.list.iter().position(|row| row.highlighted)
    }
    pub fn highlight_down(&mut self) {
        let Some(mut index) = self.get_highlight_index() else {
            return;
        };
        index += 1;
        if index >= self.size() {
            index = 0;
        }
        self.highlight(index);
    }
    pub fn highlight_up(&mut self) {
        let Some(mut index) = self.get_highlight_index() else {
            return;
        };
        if index == 0 {
            index = self.size() - 1;
        } else {
            index -= 1;
        }
        self.highlight(index);
    }
    pub fn get_highlight_index(&self) -> Option<usize> {
        self.get_highlighted_row()
    }
    pub fn highlight(&mut self, row_index: usize) {
        if let Some(row) = self.list.get_mut(row_index) {
            row.highlighted = true;
        }
    }
    pub fn move_row_up(&mut self, row_index: usize) {
        self.list.swap(row_index, row_index - 1);
    }
    pub fn move_row_down(&mut self, row_index: usize) {
        self.list.swap(row_index, row_index + 1);
    }
    pub fn reindex(&mut self, start_index: usize) {
        for (index, row) in self.list.iter_mut().enumerate().skip(start_index) {
            row.set_index(index);
        }
    }
}

/// Java buttons and Swing `ActionEvent` commands at the boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Button {
    pub action_command: String,
    pub enabled: bool,
    pub tooltip: Option<String>,
}
impl Button {
    pub fn new(command: &str) -> Self {
        Self {
            action_command: command.into(),
            enabled: true,
            tooltip: None,
        }
    }
}

/// Java `IterationTable` fields and source operations.
#[derive(Clone, Debug)]
pub struct IterationTable {
    pub row_list: RowList,
    pub header1_iteration_number: HeaderCell,
    pub header2_iteration_number: HeaderCell,
    pub header3_iteration_number: HeaderCell,
    pub header1_d_phi_d_theta_d_psi: HeaderCell,
    pub header2_d_phi: HeaderCell,
    pub header2_d_theta: HeaderCell,
    pub header2_d_psi: HeaderCell,
    pub header3_d_phi_max: HeaderCell,
    pub header3_d_phi_increment: HeaderCell,
    pub header3_d_theta_max: HeaderCell,
    pub header3_d_theta_increment: HeaderCell,
    pub header3_d_psi_max: HeaderCell,
    pub header3_d_psi_increment: HeaderCell,
    pub header1_search_radius: HeaderCell,
    pub header2_search_radius: HeaderCell,
    pub header3_search_radius: HeaderCell,
    pub header1_hi_cutoff: HeaderCell,
    pub header2_hi_cutoff: HeaderCell,
    pub header3_hi_cutoff: HeaderCell,
    pub header3_hi_cutoff_sigma: HeaderCell,
    pub header1_low_cutoff: HeaderCell,
    pub header2_low_cutoff: HeaderCell,
    pub header3_low_cutoff: HeaderCell,
    pub header3_low_cutoff_sigma: HeaderCell,
    pub header1_ref_threshold: HeaderCell,
    pub header2_ref_threshold: HeaderCell,
    pub header3_ref_threshold: HeaderCell,
    pub header1_duplicate_tolerance: HeaderCell,
    pub header2_duplicate_tolerance: HeaderCell,
    pub header3_duplicate_shift_tolerance: HeaderCell,
    pub header3_duplicate_angular_tolerance: HeaderCell,
    pub btn_move_up: Button,
    pub btn_move_down: Button,
    pub btn_add_row: Button,
    pub btn_delete_row: Button,
    pub btn_copy_row: Button,
    pub flg_remove_duplicates: bool,
    pub low_cutoff: bool,
    pub flg_strict_search_limits: bool,
    pub table_visible: bool,
    pub vertical_padding_height: usize,
    pub pack_count: usize,
    pub repaint_count: usize,
    pub last_message: Option<(String, String)>,
}
impl IterationTable {
    pub fn new<P: IterationParent>(parent: &mut P) -> Self {
        let mut table = Self {
            row_list: RowList::default(),
            header1_iteration_number: HeaderCell::new("Run #"),
            header2_iteration_number: HeaderCell::default(),
            header3_iteration_number: HeaderCell::default(),
            header1_d_phi_d_theta_d_psi: HeaderCell::new(D_PHI_D_THETA_D_PSI_HEADER1),
            header2_d_phi: HeaderCell::new("Phi"),
            header2_d_theta: HeaderCell::new("Theta"),
            header2_d_psi: HeaderCell::new("Psi"),
            header3_d_phi_max: HeaderCell::new(MAX_HEADER3),
            header3_d_phi_increment: HeaderCell::new(INCR_HEADER3),
            header3_d_theta_max: HeaderCell::new(MAX_HEADER3),
            header3_d_theta_increment: HeaderCell::new(INCR_HEADER3),
            header3_d_psi_max: HeaderCell::new(MAX_HEADER3),
            header3_d_psi_increment: HeaderCell::new(INCR_HEADER3),
            header1_search_radius: HeaderCell::new(SEARCH_RADIUS_HEADER1),
            header2_search_radius: HeaderCell::new(SEARCH_RADIUS_HEADER2),
            header3_search_radius: HeaderCell::default(),
            header1_hi_cutoff: HeaderCell::new(HICUTOFF_HEADER1),
            header2_hi_cutoff: HeaderCell::new(HICUTOFF_HEADER2),
            header3_hi_cutoff: HeaderCell::new(HICUTOFF_CUTOFF_HEADER3),
            header3_hi_cutoff_sigma: HeaderCell::new(HICUTOFF_SIGMA_HEADER3),
            header1_low_cutoff: HeaderCell::new(LOWCUTOFF_HEADER1),
            header2_low_cutoff: HeaderCell::new(LOWCUTOFF_HEADER2),
            header3_low_cutoff: HeaderCell::new(LOWCUTOFF_CUTOFF_HEADER3),
            header3_low_cutoff_sigma: HeaderCell::new(LOWCUTOFF_SIGMA_HEADER3),
            header1_ref_threshold: HeaderCell::new(REF_THRESHOLD_HEADER1),
            header2_ref_threshold: HeaderCell::new(REF_THRESHOLD_HEADER2),
            header3_ref_threshold: HeaderCell::default(),
            header1_duplicate_tolerance: HeaderCell::new(DUPLICATE_TOLERANCE_HEADER1),
            header2_duplicate_tolerance: HeaderCell::new(DUPLICATE_TOLERANCE_HEADER2),
            header3_duplicate_shift_tolerance: HeaderCell::new(DUPLICATE_SHIFT_TOLERANCE_HEADER3),
            header3_duplicate_angular_tolerance: HeaderCell::new(
                DUPLICATE_ANGULAR_TOLERANCE_HEADER3,
            ),
            btn_move_up: Button::new("Up"),
            btn_move_down: Button::new("Down"),
            btn_add_row: Button::new("Insert"),
            btn_delete_row: Button::new("Delete"),
            btn_copy_row: Button::new("Dup"),
            flg_remove_duplicates: false,
            low_cutoff: false,
            flg_strict_search_limits: false,
            table_visible: true,
            vertical_padding_height: 0,
            pack_count: 0,
            repaint_count: 0,
            last_message: None,
        };
        table.create_table();
        table.add_row(true, false, parent);
        table.update_display();
        table.refresh_vertical_padding();
        table.set_tool_tip_text();
        table
    }
    pub fn highlight(&mut self, _highlight: bool) {
        self.update_display();
    }
    pub fn highlight_up_action_performed(&mut self) {
        self.row_list.highlight_up();
    }
    pub fn highlight_down_action_performed(&mut self) {
        self.row_list.highlight_down();
    }
    /// Java `getFocusableParents`; the sole Swing root component is retained as
    /// the table's focusable-root identity at the UI boundary.
    pub fn get_focusable_parents(&self) -> usize {
        1
    }
    pub fn validate_run(&mut self) -> bool {
        let valid = self.row_list.validate_run(self.low_cutoff);
        if !valid && self.row_list.size() == 0 {
            self.last_message = Some((
                format!("Must enter at least one row in {LABEL}"),
                "Entry Error".into(),
            ));
        }
        valid
    }
    /// Java `getContainer`; native `JPanel` ownership is the Swing boundary.
    pub fn get_container(&self) -> bool {
        self.table_visible
    }
    pub fn reset<P: IterationParent>(&mut self, init: bool, parent: &mut P) {
        self.row_list.remove();
        self.row_list.list.clear();
        self.add_row(init, false, parent);
        self.flg_remove_duplicates = false;
        self.update_display();
        self.pack_count += 1;
    }
    pub fn get_parameters(&self, matlab: &mut MatlabParam) {
        self.row_list.get_parameters(matlab, self.low_cutoff);
        matlab.flg_remove_duplicates = self.flg_remove_duplicates;
        matlab.flg_strict_search_limits = self.flg_strict_search_limits;
    }
    pub fn get_peet_parameters(&self, meta_data: &mut PeetMetaData) {
        meta_data.low_cutoff = self.low_cutoff;
        self.row_list.get_peet_parameters(meta_data);
    }
    pub fn set_peet_parameters(&mut self, meta_data: &ConstPeetMetaData) {
        self.low_cutoff = meta_data.low_cutoff;
        self.row_list.set_parameters(meta_data);
        for (index, row) in self.row_list.list.iter_mut().enumerate() {
            if let Some(values) = meta_data.low_cutoff_values.get(index) {
                row.values.extend(values.clone());
            }
        }
    }
    pub fn update_rows_display(&mut self, sample_sphere: bool) {
        self.row_list
            .update_display(sample_sphere, self.flg_remove_duplicates);
    }
    pub fn check_low_cutoff_backwards_compatibility(&mut self) {
        if self
            .row_list
            .list
            .iter()
            .any(|row| !row.check_low_cutoff_backwards_compatibility())
        {
            self.low_cutoff = true;
        }
        if !self.low_cutoff {
            for row in &mut self.row_list.list {
                row.set_low_cutoff_sigma(LOW_CUTOFF_SIGMA_DEFAULT);
            }
        }
    }
    pub fn set_parameters(&mut self, matlab: &MatlabParam) {
        self.set_visible_high_pass_filter_column(self.low_cutoff);
        for (index, row) in self.row_list.list.iter_mut().enumerate() {
            if let Some(iteration) = matlab.iterations.get(index) {
                row.values = iteration.values.clone();
            }
            row.set_visible_low_cutoff_rows(self.low_cutoff);
        }
        self.flg_remove_duplicates = matlab.flg_remove_duplicates;
        self.flg_strict_search_limits = matlab.flg_strict_search_limits;
        self.update_display();
        self.pack_count += 1;
    }
    pub fn add_iteration_rows<P: IterationParent>(&mut self, matlab: &MatlabParam, parent: &mut P) {
        for index in self.row_list.size()..matlab.get_iteration_list_size() {
            let row = self.add_row(true, false, parent);
            row.values = matlab.iterations[index].values.clone();
        }
    }
    pub fn add_row<P: IterationParent>(
        &mut self,
        init: bool,
        action_insert_btn: bool,
        parent: &mut P,
    ) -> &mut IterationRow {
        let low_cutoff = self.low_cutoff;
        let index = self.row_list.size();
        let row = self.row_list.add(low_cutoff);
        if action_insert_btn {
            row.set_low_cutoff_sigma(LOW_CUTOFF_SIGMA_DEFAULT);
        }
        row.display();
        parent.update_display(init);
        self.refresh_vertical_padding();
        &mut self.row_list.list[index]
    }
    pub fn size(&self) -> usize {
        self.row_list.size()
    }
    pub fn set_tool_tip_text(&mut self) {
        self.btn_add_row.tooltip = Some("Add a new iteration row to the table.".into());
        self.btn_copy_row.tooltip =
            Some("Create a new row that is a duplicate of the highlighted row.".into());
        self.btn_move_up.tooltip = Some("Move highlighted row up in the table.".into());
        self.btn_move_down.tooltip = Some("Move highlighted row down in the table".into());
        self.btn_delete_row.tooltip = Some("Remove highlighted row from table.".into());
    }
    pub fn action<P: IterationParent>(&mut self, action_command: &str, parent: &mut P) {
        if action_command == self.btn_add_row.action_command {
            self.add_row(false, true, parent);
            self.pack_count += 1;
        } else if action_command == self.btn_copy_row.action_command {
            if let Some(row) = self.row_list.get_highlighted_row() {
                self.copy_row(row, parent);
            }
        } else if action_command == self.btn_delete_row.action_command {
            if let Some(row) = self.row_list.get_highlighted_row() {
                self.delete_row(row);
            }
        } else if action_command == self.btn_move_up.action_command {
            self.move_row_up();
        } else if action_command == self.btn_move_down.action_command {
            self.move_row_down();
        } else if action_command == "Remove duplicates" {
            self.update_rows_display(parent.is_sample_sphere());
        } else if action_command == "Bandpass filtering" {
            self.set_visible_high_pass_filter_column(self.low_cutoff);
            self.pack_count += 1;
        }
    }
    pub fn copy_row<P: IterationParent>(&mut self, row: usize, parent: &mut P) {
        self.row_list.copy(row, self.low_cutoff);
        parent.update_display(false);
        self.refresh_vertical_padding();
        self.pack_count += 1;
    }
    pub fn delete_row(&mut self, row: usize) {
        self.row_list.remove();
        let index = self.row_list.delete(row);
        self.row_list.highlight(index);
        self.row_list.display();
        self.update_display();
        self.refresh_vertical_padding();
        self.pack_count += 1;
    }
    pub fn move_row_up(&mut self) {
        let Some(index) = self.row_list.get_highlight_index() else {
            return;
        };
        if index == 0 {
            self.last_message = Some((
                "Can't move the row up.  Its at the top.".into(),
                "Wrong Row".into(),
            ));
            return;
        }
        self.row_list.move_row_up(index);
        self.row_list.remove();
        self.row_list.reindex(index - 1);
        self.row_list.display();
        self.update_display();
        self.repaint_count += 1;
    }
    pub fn move_row_down(&mut self) {
        let Some(index) = self.row_list.get_highlight_index() else {
            return;
        };
        if index == self.row_list.size() - 1 {
            self.last_message = Some((
                "Can't move the row down.  Its at the bottom.".into(),
                "Wrong Row".into(),
            ));
            return;
        }
        self.row_list.move_row_down(index);
        self.row_list.remove();
        self.row_list.reindex(index);
        self.row_list.display();
        self.update_display();
        self.repaint_count += 1;
    }
    pub fn update_display(&mut self) {
        let index = self.row_list.get_highlight_index();
        self.btn_copy_row.enabled = index.is_some();
        self.btn_delete_row.enabled = index.is_some();
        self.btn_move_up.enabled = index.is_some_and(|i| i > 0);
        self.btn_move_down.enabled = index.is_some_and(|i| i < self.row_list.size() - 1);
    }
    pub fn create_table(&mut self) {
        self.set_visible_high_pass_filter_column(self.low_cutoff);
        self.pack_count += 1;
    }
    pub fn refresh_vertical_padding(&mut self) {
        self.vertical_padding_height = 3usize.saturating_sub(self.row_list.size()) * 22;
    }
    pub fn display(&mut self) {
        self.row_list.display();
    }
    pub fn get_d_phi_d_theta_d_psi_header_cell(&self) -> &HeaderCell {
        &self.header1_d_phi_d_theta_d_psi
    }
    pub fn get_search_radius_header_cell(&self) -> &HeaderCell {
        &self.header1_search_radius
    }
    pub fn get_hi_cutoff_header_cell(&self) -> &HeaderCell {
        &self.header1_hi_cutoff
    }
    pub fn get_low_cutoff_header_cell(&self) -> &HeaderCell {
        &self.header1_low_cutoff
    }
    pub fn get_ref_threshold_header_cell(&self) -> &HeaderCell {
        &self.header1_ref_threshold
    }
    pub fn get_duplicate_tolerance_header_cell(&self) -> &HeaderCell {
        &self.header1_duplicate_tolerance
    }
    pub fn get_iteration_number_header_cell(&self) -> &HeaderCell {
        &self.header1_iteration_number
    }
    pub fn set_visible_high_pass_filter_column(&mut self, input: bool) {
        self.header1_low_cutoff.visible = input;
        self.header2_low_cutoff.visible = input;
        self.header3_low_cutoff.visible = input;
        self.header3_low_cutoff_sigma.visible = input;
        for row in &mut self.row_list.list {
            row.set_visible_low_cutoff_rows(input);
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct Parent {
        sample_sphere: bool,
        display_updates: Vec<bool>,
    }
    impl IterationParent for Parent {
        fn update_display(&mut self, init: bool) {
            self.display_updates.push(init);
        }
        fn is_sample_sphere(&self) -> bool {
            self.sample_sphere
        }
    }
    fn populated_row(table: &mut IterationTable) {
        for key in [
            "dPhiMax",
            "dPhiIncrement",
            "searchRadius",
            "hiCutoff",
            "hiCutoffSigma",
            "refThreshold",
        ] {
            table.row_list.list[0].values.insert(key.into(), "1".into());
        }
    }
    #[test]
    fn insert_copy_move_and_delete_follow_source_row_order() {
        let mut parent = Parent::default();
        let mut table = IterationTable::new(&mut parent);
        table.action("Insert", &mut parent);
        table.row_list.highlight(1);
        table.action("Up", &mut parent);
        assert_eq!(table.row_list.list[0].index, 0);
        table.action("Dup", &mut parent);
        assert_eq!(table.size(), 3);
        table.action("Delete", &mut parent);
        assert_eq!(table.size(), 2);
    }
    #[test]
    fn parameter_and_low_cutoff_paths_are_retained() {
        let mut parent = Parent::default();
        let mut table = IterationTable::new(&mut parent);
        populated_row(&mut table);
        assert!(table.validate_run());
        table.low_cutoff = true;
        table.row_list.list[0]
            .values
            .insert("lowCutoff".into(), "0".into());
        table.row_list.list[0]
            .values
            .insert("lowCutoffSigma".into(), "0.05".into());
        let mut matlab = MatlabParam::default();
        table.get_parameters(&mut matlab);
        assert_eq!(matlab.get_iteration_list_size(), 1);
        assert!(table.validate_run());
    }
    #[test]
    fn high_pass_visibility_and_highlight_buttons_update() {
        let mut parent = Parent::default();
        let mut table = IterationTable::new(&mut parent);
        assert!(!table.header1_low_cutoff.visible);
        table.set_visible_high_pass_filter_column(true);
        assert!(table.row_list.list[0].low_cutoff_rows_visible);
        table.row_list.highlight(0);
        table.update_display();
        assert!(table.btn_copy_row.enabled);
        assert!(!table.btn_move_up.enabled);
    }
}
