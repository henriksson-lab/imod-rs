//! `IMOD/Etomo/src/etomo/ui/swing/IterationTable.java`.
//!
//! Swing panel/layout construction, `UIHarness`, autodoc tooltips, and the owning
//! `PeetDialog` remain explicit boundaries.  `IterationRow.java` lives in its
//! matching canonical module; this source owns the table and its `RowList` only.
#![allow(dead_code)]

use std::collections::BTreeMap;

pub use super::header_cell::HeaderCell;
use super::highlightable::Highlightable;
use super::highlightable_table::{HighlightFocusableParent, HighlightableTable};
use super::iteration_parent::IterationParent;
pub use super::iteration_row::IterationRow;

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

/// Java `MatlabParam.Iteration` data touched by this unit.
#[derive(Clone, Debug, Default)]
pub struct Iteration {
    pub values: BTreeMap<String, String>,
}
/// Java `MatlabParam`, restricted to direct `IterationTable` accesses.
#[derive(Clone, Debug, Default)]
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
#[derive(Clone, Debug, Default)]
pub struct PeetMetaData {
    pub low_cutoff: bool,
    pub low_cutoff_values: Vec<BTreeMap<String, String>>,
}
pub type ConstPeetMetaData = PeetMetaData;

/// Java private static inner `RowList`.
#[derive(Clone, Debug, Default)]
pub struct RowList {
    pub list: Vec<IterationRow>,
    pub meta_data: Option<ConstPeetMetaData>,
    /// Direct `UIHarness.openMessageDialog` result from Java
    /// `getHighlightedRow()` when no selected highlighter exists.
    pub last_message: Option<(String, String)>,
}
impl RowList {
    pub fn add(&mut self, is_low_cutoff: bool) -> &mut IterationRow {
        let index = self.list.len();
        let mut row = IterationRow::new(index, is_low_cutoff);
        row.set_visible_low_cutoff_rows(is_low_cutoff);
        row.set_names();
        self.list.push(row);
        self.list.last_mut().unwrap()
    }
    pub fn get_parameters(&self, matlab: &mut MatlabParam, low_cutoff: bool) {
        matlab.set_iteration_list_size(self.list.len());
        for (index, row) in self.list.iter().enumerate() {
            row.get_parameters_matlab(&mut matlab.iterations[index], low_cutoff);
        }
    }
    pub fn get_peet_parameters(&self, meta_data: &mut PeetMetaData) {
        meta_data.low_cutoff_values.clear();
        for row in &self.list {
            row.get_parameters_peet(meta_data);
        }
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
    pub fn validate_run(&mut self, is_low_cutoff: bool) -> bool {
        !self.list.is_empty()
            && self
                .list
                .iter_mut()
                .all(|row| row.validate_run(is_low_cutoff))
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
        self.list
            .push(IterationRow::copy(self.list.len(), &self.list[row_index]));
        let copy = self.list.last_mut().unwrap();
        copy.set_visible_low_cutoff_rows(is_low_cutoff);
        copy.set_names();
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
    pub fn get_highlighted_row(&mut self) -> Option<usize> {
        let value = self.list.iter().position(IterationRow::is_highlighted);
        if value.is_none() {
            self.last_message = Some(("Please highlight a row.".into(), "Entry Error".into()));
        }
        value
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
        self.list.iter().position(IterationRow::is_highlighted)
    }
    pub fn highlight(&mut self, row_index: usize) {
        if let Some(row) = self.list.get_mut(row_index) {
            row.set_highlighter_selected(true);
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

/// The native `JPanel`, `GridBagLayout`, `GridBagConstraints`, `BoxLayout`,
/// `Box`, `LineBorder`, and `EtchedBorder` graph created by Java `createTable`.
/// It is deliberately data rather than a replacement widget toolkit.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct IterationTableLayoutBoundary {
    pub table_grid_bag_layout: bool,
    pub table_black_line_border: bool,
    pub button_box_y_axis: bool,
    pub table_and_checkbox_box_y_axis: bool,
    pub root_box_x_axis: bool,
    pub root_etched_border_label: Option<String>,
    pub button_order: Vec<String>,
    pub checkbox_order: Vec<String>,
    pub table_and_checkbox_has_vertical_padding: bool,
    pub table_and_checkbox_has_checkbox_panel: bool,
}

/// Java private inner `ITActionListener`.  Native `ActionEvent` construction is
/// a Swing boundary; its source dispatch is retained directly.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ItActionListener;
impl ItActionListener {
    /// Java `ITActionListener.actionPerformed(ActionEvent)`.
    pub fn action_performed<P: IterationParent>(
        &self,
        iteration_table: &mut IterationTable,
        action_command: &str,
        parent: &mut P,
    ) {
        iteration_table.action(action_command, parent);
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
    pub layout: IterationTableLayoutBoundary,
    /// Java `focusableParents = new JComponent[] { rootPanel }`.  Native Swing
    /// input/action-map installation is represented by `HighlightableTable`.
    pub focusable_parents: Vec<HighlightFocusableParent>,
    /// Java private `addListeners` installs the single `ITActionListener` on five
    /// buttons and the two source check boxes (not strict-search-limits).
    pub action_listener_count: usize,
    /// Results of the direct `AutodocFactory` / `EtomoAutodoc.getTooltip` manager
    /// boundary used only by Java `setToolTipText`.
    pub duplicate_shift_tolerance_autodoc_tooltip: Option<String>,
    pub duplicate_angular_tolerance_autodoc_tooltip: Option<String>,
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
            header3_d_phi_max: HeaderCell::new_with_text_width(MAX_HEADER3, 40),
            header3_d_phi_increment: HeaderCell::new_with_text_width(INCR_HEADER3, 40),
            header3_d_theta_max: HeaderCell::new_with_text_width(MAX_HEADER3, 40),
            header3_d_theta_increment: HeaderCell::new_with_text_width(INCR_HEADER3, 40),
            header3_d_psi_max: HeaderCell::new_with_text_width(MAX_HEADER3, 40),
            header3_d_psi_increment: HeaderCell::new_with_text_width(INCR_HEADER3, 40),
            header1_search_radius: HeaderCell::new(SEARCH_RADIUS_HEADER1),
            header2_search_radius: HeaderCell::new(SEARCH_RADIUS_HEADER2),
            header3_search_radius: HeaderCell::new_with_width(75),
            header1_hi_cutoff: HeaderCell::new(HICUTOFF_HEADER1),
            header2_hi_cutoff: HeaderCell::new(HICUTOFF_HEADER2),
            header3_hi_cutoff: HeaderCell::new_with_text_width(HICUTOFF_CUTOFF_HEADER3, 50),
            header3_hi_cutoff_sigma: HeaderCell::new_with_text_width(HICUTOFF_SIGMA_HEADER3, 50),
            header1_low_cutoff: HeaderCell::new(LOWCUTOFF_HEADER1),
            header2_low_cutoff: HeaderCell::new(LOWCUTOFF_HEADER2),
            header3_low_cutoff: HeaderCell::new_with_text_width(LOWCUTOFF_CUTOFF_HEADER3, 50),
            header3_low_cutoff_sigma: HeaderCell::new_with_text_width(LOWCUTOFF_SIGMA_HEADER3, 50),
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
            layout: IterationTableLayoutBoundary::default(),
            focusable_parents: vec![HighlightFocusableParent::default()],
            action_listener_count: 0,
            duplicate_shift_tolerance_autodoc_tooltip: None,
            duplicate_angular_tolerance_autodoc_tooltip: None,
            table_visible: true,
            vertical_padding_height: 0,
            pack_count: 0,
            repaint_count: 0,
            last_message: None,
        };
        table.create_table();
        table.add_row(true, false, parent);
        table.display();
        table.update_display();
        table.refresh_vertical_padding();
        table.set_tool_tip_text();
        table
    }
    /// Java static `getInstance(BaseManager, IterationParent)`.
    pub fn get_instance<P: IterationParent>(parent: &mut P) -> Self {
        let mut value = Self::new(parent);
        value.add_listeners();
        value.init_highlight_hotkeys();
        value
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
    pub fn validate_run(&mut self) -> bool {
        let valid = self.row_list.validate_run(self.low_cutoff);
        if !valid && self.row_list.size() == 0 {
            self.last_message = Some((
                format!("Must enter at least one row in {LABEL}"),
                "Entry Error".into(),
            ));
        }
        if !valid && self.row_list.size() > 0 {
            self.last_message = self
                .row_list
                .list
                .iter()
                .find_map(|row| row.last_message.clone());
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
        for row in &mut self.row_list.list {
            row.set_parameters_peet(Some(meta_data));
        }
    }
    pub fn update_rows_display(&mut self, sample_sphere: bool) {
        self.row_list
            .update_display(sample_sphere, self.flg_remove_duplicates);
    }
    pub fn check_low_cutoff_backwards_compatibility(&mut self, matlab: &MatlabParam) {
        if self
            .row_list
            .list
            .iter()
            .any(|row| !row.check_low_cutoff_backwards_compatibility(matlab))
        {
            self.low_cutoff = true;
        }
        if !self.low_cutoff {
            for row in &mut self.row_list.list {
                row.set_low_cutoff_sigma(Some(LOW_CUTOFF_SIGMA_DEFAULT));
            }
        }
    }
    pub fn set_parameters(&mut self, matlab: &MatlabParam) {
        self.set_visible_high_pass_filter_column(self.low_cutoff);
        for row in &mut self.row_list.list {
            row.set_parameters_matlab(matlab, self.low_cutoff);
            row.set_visible_low_cutoff_rows(self.low_cutoff);
        }
        self.flg_remove_duplicates = matlab.flg_remove_duplicates;
        self.flg_strict_search_limits = matlab.flg_strict_search_limits;
        self.update_display();
        self.pack_count += 1;
    }
    pub fn add_iteration_rows<P: IterationParent>(&mut self, matlab: &MatlabParam, parent: &mut P) {
        for index in self.row_list.size()..matlab.get_iteration_list_size() {
            let low_cutoff = self.low_cutoff;
            let row = self.add_row(true, false, parent);
            row.set_parameters_matlab(matlab, low_cutoff);
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
            row.set_low_cutoff_sigma(Some(LOW_CUTOFF_SIGMA_DEFAULT));
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
        self.header3_duplicate_shift_tolerance.set_tool_tip_text(
            self.duplicate_shift_tolerance_autodoc_tooltip
                .as_deref()
                .unwrap_or_default(),
        );
        self.header3_duplicate_angular_tolerance.set_tool_tip_text(
            self.duplicate_angular_tolerance_autodoc_tooltip
                .as_deref()
                .unwrap_or_default(),
        );
        self.btn_add_row.tooltip = Some("Add a new iteration row to the table.".into());
        self.btn_copy_row.tooltip =
            Some("Create a new row that is a duplicate of the highlighted row.".into());
        self.btn_move_up.tooltip = Some("Move highlighted row up in the table.".into());
        self.btn_move_down.tooltip = Some("Move highlighted row down in the table".into());
        self.btn_delete_row.tooltip = Some("Remove highlighted row from table.".into());
    }
    /// Java private `addListeners`; actual listener objects and Swing dispatch stay
    /// at the GUI boundary, while the source attachment cardinality is retained.
    pub fn add_listeners(&mut self) {
        self.action_listener_count += 7;
    }
    pub fn action<P: IterationParent>(&mut self, action_command: &str, parent: &mut P) {
        if action_command == self.btn_add_row.action_command {
            self.add_row(false, true, parent);
            self.pack_count += 1;
        } else if action_command == self.btn_copy_row.action_command {
            if let Some(row) = self.row_list.get_highlighted_row() {
                self.copy_row(row, parent);
            } else {
                self.last_message = self.row_list.last_message.clone();
            }
        } else if action_command == self.btn_delete_row.action_command {
            if let Some(row) = self.row_list.get_highlighted_row() {
                self.delete_row(row);
            } else {
                self.last_message = self.row_list.last_message.clone();
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
        self.layout.table_grid_bag_layout = true;
        self.layout.table_black_line_border = true;
        self.layout.button_box_y_axis = true;
        self.layout.table_and_checkbox_box_y_axis = true;
        self.layout.root_box_x_axis = true;
        self.layout.root_etched_border_label = Some(LABEL.into());
        self.layout.button_order = vec![
            "Up".into(),
            "Down".into(),
            "Insert".into(),
            "Delete".into(),
            "Dup".into(),
        ];
        self.layout.checkbox_order = vec![
            "Remove duplicates".into(),
            "Bandpass filtering".into(),
            "Strict search limit checking".into(),
        ];
        self.pack_count += 1;
    }
    pub fn refresh_vertical_padding(&mut self) {
        self.vertical_padding_height = 3usize.saturating_sub(self.row_list.size()) * 22;
        self.layout.table_and_checkbox_has_vertical_padding = true;
        self.layout.table_and_checkbox_has_checkbox_panel = true;
    }
    pub fn display(&mut self) {
        self.header1_iteration_number.add();
        self.header2_iteration_number.add();
        self.header3_iteration_number.add();
        self.header1_d_phi_d_theta_d_psi.add();
        self.header2_d_phi.add();
        self.header2_d_theta.add();
        self.header2_d_psi.add();
        self.header3_d_phi_max.add();
        self.header3_d_phi_increment.add();
        self.header3_d_theta_max.add();
        self.header3_d_theta_increment.add();
        self.header3_d_psi_max.add();
        self.header3_d_psi_increment.add();
        self.header1_search_radius.add();
        self.header2_search_radius.add();
        self.header3_search_radius.add();
        self.header1_hi_cutoff.add();
        self.header2_hi_cutoff.add();
        self.header3_hi_cutoff.add();
        self.header3_hi_cutoff_sigma.add();
        self.header1_low_cutoff.add();
        self.header2_low_cutoff.add();
        self.header3_low_cutoff.add();
        self.header3_low_cutoff_sigma.add();
        self.header1_ref_threshold.add();
        self.header2_ref_threshold.add();
        self.header3_ref_threshold.add();
        self.header1_duplicate_tolerance.add();
        self.header2_duplicate_tolerance.add();
        self.header3_duplicate_shift_tolerance.add();
        self.header3_duplicate_angular_tolerance.add();
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
        self.header1_low_cutoff.set_visible(input);
        self.header2_low_cutoff.set_visible(input);
        self.header3_low_cutoff.set_visible(input);
        self.header3_low_cutoff_sigma.set_visible(input);
        for row in &mut self.row_list.list {
            row.set_visible_low_cutoff_rows(input);
        }
    }
}

impl Highlightable for IterationTable {
    fn highlight(&mut self, highlight: bool) {
        IterationTable::highlight(self, highlight);
    }
}

impl HighlightableTable for IterationTable {
    fn highlight_up_action_performed(&mut self) {
        IterationTable::highlight_up_action_performed(self);
    }
    fn highlight_down_action_performed(&mut self) {
        IterationTable::highlight_down_action_performed(self);
    }
    fn get_focusable_parents(&mut self) -> Option<&mut [HighlightFocusableParent]> {
        Some(&mut self.focusable_parents)
    }
    fn unique_key(&self) -> &str {
        "Interation"
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
        let row = &mut table.row_list.list[0];
        row.d_phi_max.set_value("1");
        row.d_phi_increment.set_value("1");
        row.d_theta_max.set_value("1");
        row.d_theta_increment.set_value("1");
        row.d_psi_max.set_value("1");
        row.d_psi_increment.set_value("1");
        row.search_radius.set_value("1");
        row.hi_cutoff.set_value("1");
        row.hi_cutoff_sigma.set_value("1");
        row.ref_threshold.set_value("1");
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
        table.update_rows_display(true);
        assert!(
            table.validate_run(),
            "{:?}",
            table.row_list.list[0].last_message
        );
        table.low_cutoff = true;
        table.row_list.list[0].low_cutoff.set_value("0");
        table.row_list.list[0].low_cutoff_sigma.set_value("0.05");
        let mut matlab = MatlabParam::default();
        table.get_parameters(&mut matlab);
        assert_eq!(matlab.get_iteration_list_size(), 1);
        assert!(table.validate_run());
    }
    #[test]
    fn high_pass_visibility_and_highlight_buttons_update() {
        let mut parent = Parent::default();
        let mut table = IterationTable::new(&mut parent);
        assert!(!table.header1_low_cutoff.cell.visible);
        table.set_visible_high_pass_filter_column(true);
        assert!(table.row_list.list[0].low_cutoff.text_field.visible);
        table.row_list.highlight(0);
        table.update_display();
        assert!(table.btn_copy_row.enabled);
        assert!(!table.btn_move_up.enabled);
    }
    #[test]
    fn source_factory_installs_listener_and_all_three_hotkey_maps() {
        let mut parent = Parent::default();
        let table = IterationTable::get_instance(&mut parent);
        assert_eq!(table.action_listener_count, 7);
        let root = &table.focusable_parents[0];
        assert!(root.focusable);
        assert_eq!(root.actions, ["InterationALT_UP", "InterationALT_DOWN"]);
        assert_eq!(root.focused_keys, ["ALT+UP", "ALT+DOWN"]);
        assert_eq!(root.window_keys, ["ALT+UP", "ALT+DOWN"]);
        assert_eq!(root.ancestor_keys, ["ALT+UP", "ALT+DOWN"]);
    }
    #[test]
    fn display_places_all_three_header_rows_and_unselected_actions_report_source_error() {
        let mut parent = Parent::default();
        let mut table = IterationTable::new(&mut parent);
        assert!(table.header1_iteration_number.jpanel_container);
        assert!(table.header3_duplicate_angular_tolerance.jpanel_container);
        table.action("Dup", &mut parent);
        assert_eq!(
            table.last_message,
            Some(("Please highlight a row.".into(), "Entry Error".into()))
        );
        assert!(table.layout.table_grid_bag_layout);
        assert!(table.layout.table_black_line_border);
        assert_eq!(
            table.layout.root_etched_border_label.as_deref(),
            Some(LABEL)
        );
        assert_eq!(
            table.layout.button_order,
            ["Up", "Down", "Insert", "Delete", "Dup"]
        );
        assert!(table.layout.table_and_checkbox_has_vertical_padding);
    }
    #[test]
    fn insertion_and_backwards_compatibility_follow_default_low_cutoff_rules() {
        let mut parent = Parent::default();
        let mut table = IterationTable::new(&mut parent);
        table.action("Insert", &mut parent);
        assert_eq!(
            table.row_list.list[1].low_cutoff_sigma.get_value(),
            LOW_CUTOFF_SIGMA_DEFAULT
        );
        let mut matlab = MatlabParam::default();
        matlab.set_iteration_list_size(2);
        for iteration in &mut matlab.iterations {
            iteration
                .values
                .insert("lowCutoffCutoff".into(), LOW_CUTOFF_DEFAULT.into());
            iteration
                .values
                .insert("lowCutoffSigma".into(), LOW_CUTOFF_SIGMA_DEFAULT.into());
        }
        table.check_low_cutoff_backwards_compatibility(&matlab);
        assert!(!table.low_cutoff);
        assert_eq!(
            table.row_list.list[0].low_cutoff_sigma.get_value(),
            LOW_CUTOFF_SIGMA_DEFAULT
        );
        matlab.iterations[1]
            .values
            .insert("lowCutoffCutoff".into(), "0.1".into());
        table.check_low_cutoff_backwards_compatibility(&matlab);
        assert!(table.low_cutoff);
    }
    #[test]
    fn inner_listener_dispatches_to_the_table_action() {
        let mut parent = Parent::default();
        let mut table = IterationTable::get_instance(&mut parent);
        ItActionListener.action_performed(&mut table, "Insert", &mut parent);
        assert_eq!(table.size(), 2);
        assert_eq!(parent.display_updates, [true, false]);
    }
}
