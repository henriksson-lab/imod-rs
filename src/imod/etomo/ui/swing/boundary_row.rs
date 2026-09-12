//! `IMOD/Etomo/src/etomo/ui/swing/BoundaryRow.java`.
//!
//! Swing layout/listener attachment and `XfjointomoLog` I/O remain explicit
//! boundaries.  The row's gap arithmetic, row state, and metadata/screen-state
//! transfers are kept source-shaped in this unit.
#![allow(dead_code)]

use std::collections::BTreeMap;

use super::section_table_panel::{HeaderCell, SectionTableRowData, Tab, Viewport};
use super::spinner_cell::SpinnerCell as BoundaryRowSpinnerCell;

pub const INVERTED_TOOLTIP: &str = "This value comes from an inverted section.";
pub const EMPTY_SLICE_WARNING: &str = "Empty slices will be added to the section.";
pub const TABLE_LABEL: &str = "Boundary Table";

/// `FieldCell` state used by this source unit at the unported widget boundary.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct BoundaryRowFieldCell {
    pub value: String,
    pub warning: bool,
    pub warning_tooltip: Option<String>,
    pub displayed: bool,
}
impl BoundaryRowFieldCell {
    pub fn set_value(&mut self, value: impl ToString) {
        self.value = value.to_string();
    }
    pub fn get_value(&self) -> &str {
        &self.value
    }
    pub fn get_int_value(&self) -> i32 {
        self.value.parse().unwrap_or(i32::MIN)
    }
    pub fn get_double_value(&self) -> f64 {
        self.value.parse().unwrap_or(f64::NAN)
    }
    pub fn set_warning(&mut self, warning: bool, tooltip: Option<&str>) {
        self.warning = warning;
        self.warning_tooltip = tooltip.map(str::to_owned);
    }
    pub fn remove(&mut self) {
        self.displayed = false;
    }
}

/// `ConstJoinMetaData` / `JoinMetaData` members read by `BoundaryRow`.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct BoundaryRowMetaData {
    pub section_table_data: Vec<SectionTableRowData>,
    pub boundary_row_end: BTreeMap<i32, i32>,
    pub boundary_row_start: BTreeMap<i32, i32>,
}
impl BoundaryRowMetaData {
    pub fn is_boundary_row_end_list_empty(&self) -> bool {
        self.boundary_row_end.is_empty()
    }
    pub fn get_boundary_row_end(&self, key: i32) -> Option<i32> {
        self.boundary_row_end.get(&key).copied()
    }
    pub fn reset_boundary_row_start_list(&mut self) {
        self.boundary_row_start.clear();
    }
    pub fn reset_boundary_row_end_list(&mut self) {
        self.boundary_row_end.clear();
    }
    pub fn set_boundary_row_end(&mut self, key: i32, value: String) {
        if let Ok(value) = value.parse() {
            self.boundary_row_end.insert(key, value);
        }
    }
    pub fn set_boundary_row_start(&mut self, key: i32, value: String) {
        if let Ok(value) = value.parse() {
            self.boundary_row_start.insert(key, value);
        }
    }
}

/// `JoinScreenState` values accessed by this row.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BoundaryRowScreenState {
    pub best_gap: BTreeMap<i32, f64>,
    pub mean_error: BTreeMap<i32, f64>,
    pub max_error: BTreeMap<i32, f64>,
}
impl BoundaryRowScreenState {
    pub fn get_best_gap(&self, key: i32) -> f64 {
        self.best_gap.get(&key).copied().unwrap_or(f64::NAN)
    }
    pub fn get_mean_error(&self, key: i32) -> f64 {
        self.mean_error.get(&key).copied().unwrap_or(f64::NAN)
    }
    pub fn get_max_error(&self, key: i32) -> f64 {
        self.max_error.get(&key).copied().unwrap_or(f64::NAN)
    }
    pub fn reset_best_gap(&mut self) {
        self.best_gap.clear();
    }
    pub fn reset_mean_error(&mut self) {
        self.mean_error.clear();
    }
    pub fn reset_max_error(&mut self) {
        self.max_error.clear();
    }
    pub fn set_best_gap(&mut self, key: i32, value: &str) {
        if let Ok(value) = value.parse() {
            self.best_gap.insert(key, value);
        }
    }
    pub fn set_mean_error(&mut self, key: i32, value: &str) {
        if let Ok(value) = value.parse() {
            self.mean_error.insert(key, value);
        }
    }
    pub fn set_max_error(&mut self, key: i32, value: &str) {
        if let Ok(value) = value.parse() {
            self.max_error.insert(key, value);
        }
    }
}

/// `BoundaryTable` header at this row's table boundary.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct BoundaryTable {
    pub adjusted_header_cell: HeaderCell,
}
impl BoundaryTable {
    pub fn get_adjusted_header_cell(&self) -> &HeaderCell {
        &self.adjusted_header_cell
    }
}

/// Data returned by `XfjointomoLog` after the file/application boundary.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct XfjointomoLogRow {
    pub best_gap: f64,
    pub mean_error: f64,
    pub max_error: f64,
}
pub type XfjointomoLog = BTreeMap<String, XfjointomoLogRow>;

/// Java package-private final `BoundaryRow`.
#[derive(Clone, Debug, PartialEq)]
pub struct BoundaryRow {
    pub boundary: HeaderCell,
    pub sections: HeaderCell,
    pub best_gap: BoundaryRowFieldCell,
    pub mean_error: BoundaryRowFieldCell,
    pub max_error: BoundaryRowFieldCell,
    pub orig_end: BoundaryRowFieldCell,
    pub orig_start: BoundaryRowFieldCell,
    pub adjusted_end: BoundaryRowSpinnerCell,
    pub adjusted_start: BoundaryRowSpinnerCell,
    pub z_max_end: i32,
    pub z_max_start: i32,
    pub table: BoundaryTable,
    pub gap: Option<Gap>,
    pub end_inverted: bool,
    pub start_inverted: bool,
    /// Native `HeaderCell.add/remove` state retained at the Swing boundary.
    pub boundary_displayed: bool,
    /// Native `HeaderCell.add/remove` state retained at the Swing boundary.
    pub sections_displayed: bool,
    pub model_displayed: bool,
    pub rejoin_displayed: bool,
}
impl BoundaryRow {
    pub fn new(
        key: i32,
        meta_data: &BoundaryRowMetaData,
        screen_state: &BoundaryRowScreenState,
        table: BoundaryTable,
    ) -> Self {
        let first_section = key.to_string();
        let end_data = &meta_data.section_table_data[(key - 1) as usize];
        let start_data = &meta_data.section_table_data[key as usize];
        let mut orig_end = BoundaryRowFieldCell::default();
        orig_end.set_value(end_data.join_final_end);
        orig_end.set_warning(
            end_data.inverted,
            end_data.inverted.then_some(INVERTED_TOOLTIP),
        );
        let mut orig_start = BoundaryRowFieldCell::default();
        orig_start.set_value(start_data.join_final_start);
        orig_start.set_warning(
            start_data.inverted,
            start_data.inverted.then_some(INVERTED_TOOLTIP),
        );
        let mut row = Self {
            boundary: HeaderCell::new(&first_section),
            sections: HeaderCell::new(format!("{first_section} & {}", key + 1)),
            best_gap: BoundaryRowFieldCell::default(),
            mean_error: BoundaryRowFieldCell::default(),
            max_error: BoundaryRowFieldCell::default(),
            orig_end,
            orig_start,
            adjusted_end: BoundaryRowSpinnerCell::get_int_instance(
                end_data.z_max * -2,
                end_data.z_max * 2,
            ),
            adjusted_start: BoundaryRowSpinnerCell::get_int_instance(
                start_data.z_max * -2,
                start_data.z_max * 2,
            ),
            z_max_end: end_data.z_max,
            z_max_start: start_data.z_max,
            table,
            gap: None,
            end_inverted: end_data.inverted,
            start_inverted: start_data.inverted,
            boundary_displayed: false,
            sections_displayed: false,
            model_displayed: false,
            rejoin_displayed: false,
        };
        row.best_gap.set_value(screen_state.get_best_gap(key));
        row.mean_error.set_value(screen_state.get_mean_error(key));
        row.max_error.set_value(screen_state.get_max_error(key));
        row.set_adjusted_values(Some(meta_data));
        row.adjusted_end.add_change_listener();
        row.adjusted_start.add_change_listener();
        row
    }
    pub fn set_names(&mut self) {
        self.adjusted_start.set_headers(
            TABLE_LABEL,
            &self.sections,
            self.table.get_adjusted_header_cell(),
        );
        self.adjusted_end.set_headers(
            TABLE_LABEL,
            &self.sections,
            self.table.get_adjusted_header_cell(),
        );
    }
    pub fn set_adjusted_values(&mut self, meta_data: Option<&BoundaryRowMetaData>) {
        self.adjusted_end.remove_change_listener();
        self.adjusted_start.remove_change_listener();
        self.gap = Some(Gap::new(
            self.best_gap.get_double_value().round() as i32,
            self.orig_end.get_int_value(),
            self.orig_start.get_int_value(),
            self.end_inverted,
            self.start_inverted,
            self.z_max_end,
            self.z_max_start,
        ));
        let gap = self.gap.as_mut().unwrap();
        if meta_data.is_none_or(|metadata| metadata.is_boundary_row_end_list_empty())
            || meta_data
                .and_then(|metadata| {
                    metadata.get_boundary_row_end(self.boundary.text.parse().unwrap())
                })
                .is_none()
        {
            self.adjusted_end.set_value(gap.get_adjusted_left());
            self.adjusted_start.set_value(gap.get_adjusted_right());
        } else {
            let end = meta_data
                .unwrap()
                .get_boundary_row_end(self.boundary.text.parse().unwrap())
                .unwrap();
            self.adjusted_end.set_value(end);
            gap.msg_left_gap_boundary_changed(end);
            self.adjusted_start.set_value(gap.get_adjusted_right());
        }
        self.set_adjusted_value_warnings();
        self.adjusted_end.add_change_listener();
        self.adjusted_start.add_change_listener();
    }
    pub fn set_adjusted_value_warnings(&mut self) {
        let gap = self.gap.as_ref().unwrap();
        self.adjusted_end.set_warning(
            !gap.is_left_ok(),
            (!gap.is_left_ok()).then_some(EMPTY_SLICE_WARNING),
        );
        self.adjusted_start.set_warning(
            !gap.is_right_ok(),
            (!gap.is_right_ok()).then_some(EMPTY_SLICE_WARNING),
        );
    }
    pub fn calculate_negative_adjustment(&self, rounded_best_gap: i32, orig: i32) -> i32 {
        assert!(
            rounded_best_gap >= 0 && orig > 0,
            "Only pass the absolute value of roundedBestGap. Orig is either origEnd or origStart and must be at least 1."
        );
        -(rounded_best_gap / 2).min(orig - 1)
    }
    pub fn display(&mut self, index: usize, viewport: &Viewport, tab: Tab) {
        if !viewport.in_viewport(index) {
            return;
        }
        if tab == Tab::Model {
            self.display_model();
        } else if tab == Tab::Rejoin {
            self.display_rejoin();
        }
    }
    pub fn display_model(&mut self) {
        self.model_displayed = true;
        self.boundary_displayed = true;
        self.best_gap.displayed = true;
        self.mean_error.displayed = true;
        self.max_error.displayed = true;
    }
    pub fn display_rejoin(&mut self) {
        self.rejoin_displayed = true;
        self.sections_displayed = true;
        self.orig_end.displayed = true;
        self.orig_start.displayed = true;
        self.best_gap.displayed = true;
        self.adjusted_end.spinner.visible = true;
        self.adjusted_start.spinner.visible = true;
    }
    pub fn remove_display(&mut self) {
        self.boundary_displayed = false;
        self.best_gap.remove();
        self.mean_error.remove();
        self.max_error.remove();
        self.orig_end.remove();
        self.orig_end.remove();
        self.adjusted_end.remove();
        self.adjusted_start.remove();
    }
    pub fn set_xfjointomo_result(&mut self, xfjointomo_log: &XfjointomoLog) {
        let boundary = &self.boundary.text;
        let Some(row) = xfjointomo_log.get(boundary) else {
            return;
        };
        self.best_gap.set_value(row.best_gap);
        self.mean_error.set_value(row.mean_error);
        self.max_error.set_value(row.max_error);
        self.set_adjusted_values(None);
    }
    pub fn reset_screen_state(screen_state: &mut BoundaryRowScreenState) {
        screen_state.reset_best_gap();
        screen_state.reset_mean_error();
        screen_state.reset_max_error();
    }
    pub fn get_screen_state(&self, screen_state: &mut BoundaryRowScreenState) {
        let key = self.boundary.text.parse().unwrap();
        screen_state.set_best_gap(key, self.best_gap.get_value());
        screen_state.set_mean_error(key, self.mean_error.get_value());
        screen_state.set_max_error(key, self.max_error.get_value());
    }
    pub fn reset_meta_data(meta_data: &mut BoundaryRowMetaData) {
        meta_data.reset_boundary_row_start_list();
        meta_data.reset_boundary_row_end_list();
    }
    pub fn get_meta_data(&self, meta_data: &mut BoundaryRowMetaData) {
        let key = self.boundary.text.parse().unwrap();
        meta_data.set_boundary_row_end(key, self.adjusted_end.get_string_value());
        meta_data.set_boundary_row_start(key, self.adjusted_start.get_string_value());
    }
    pub fn adjusted_end_state_changed(&mut self) {
        if self.gap.is_none() {
            self.set_adjusted_values(None);
        }
        self.adjusted_start.remove_change_listener();
        let adjusted_end = self.adjusted_end.get_int_value();
        self.gap
            .as_mut()
            .unwrap()
            .msg_left_gap_boundary_changed(adjusted_end);
        self.adjusted_start
            .set_value(self.gap.as_ref().unwrap().get_adjusted_right());
        self.set_adjusted_value_warnings();
        self.adjusted_start.add_change_listener();
    }
    pub fn adjusted_start_state_changed(&mut self) {
        if self.gap.is_none() {
            self.set_adjusted_values(None);
        }
        self.adjusted_end.remove_change_listener();
        let adjusted_start = self.adjusted_start.get_int_value();
        self.gap
            .as_mut()
            .unwrap()
            .msg_right_gap_boundary_changed(adjusted_start);
        self.adjusted_end
            .set_value(self.gap.as_ref().unwrap().get_adjusted_left());
        self.set_adjusted_value_warnings();
        self.adjusted_end.add_change_listener();
    }
}

/// Java private static final inner `Gap`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Gap {
    pub gap: i32,
    pub left_boundary: GapBoundary,
    pub right_boundary: GapBoundary,
}
impl Gap {
    pub fn new(
        gap: i32,
        orig_left: i32,
        orig_right: i32,
        left_inverted: bool,
        right_inverted: bool,
        ok_max_left: i32,
        ok_max_right: i32,
    ) -> Self {
        let mut value = Self {
            gap,
            left_boundary: GapBoundary::new(orig_left, left_inverted, ok_max_left, true),
            right_boundary: GapBoundary::new(orig_right, right_inverted, ok_max_right, false),
        };
        if gap == 0 {
            return value;
        }
        let positive_gap = gap >= 0;
        let abs_gap = gap.abs();
        let mut right_succeeded = true;
        let mut stay_in_ok_range = true;
        while value.left_boundary.get_adjustment_abs_value()
            + value.right_boundary.get_adjustment_abs_value()
            < abs_gap
        {
            let left_succeeded = value.left_boundary.adjust(positive_gap, stay_in_ok_range);
            if value.left_boundary.get_adjustment_abs_value()
                + value.right_boundary.get_adjustment_abs_value()
                < abs_gap
            {
                right_succeeded = value.right_boundary.adjust(positive_gap, stay_in_ok_range);
            }
            stay_in_ok_range = left_succeeded || right_succeeded;
        }
        value
    }
    pub fn msg_left_gap_boundary_changed(&mut self, adjusted_left: i32) {
        let gap_change = self.left_boundary.r#move(adjusted_left);
        for _ in 0..gap_change.abs() {
            self.right_boundary.adjust(gap_change > 0, false);
        }
    }
    pub fn msg_right_gap_boundary_changed(&mut self, adjusted_right: i32) {
        let gap_change = self.right_boundary.r#move(adjusted_right);
        for _ in 0..gap_change.abs() {
            self.left_boundary.adjust(gap_change > 0, false);
        }
    }
    pub fn get_adjusted_left(&self) -> i32 {
        self.left_boundary.get_adjusted()
    }
    pub fn get_adjusted_right(&self) -> i32 {
        self.right_boundary.get_adjusted()
    }
    pub fn is_left_ok(&self) -> bool {
        self.left_boundary.is_ok()
    }
    pub fn is_right_ok(&self) -> bool {
        self.right_boundary.is_ok()
    }
}

/// Java `Gap.GapBoundary`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GapBoundary {
    pub orig: i32,
    pub inverted: bool,
    pub left_side: bool,
    pub ok_min: i32,
    pub ok_max: i32,
    pub adjustment: i32,
}
impl GapBoundary {
    pub fn new(orig: i32, inverted: bool, ok_max: i32, left_side: bool) -> Self {
        Self {
            orig,
            inverted,
            left_side,
            ok_min: 1,
            ok_max,
            adjustment: 0,
        }
    }
    pub fn adjust(&mut self, positive_gap: bool, stay_in_ok_range: bool) -> bool {
        if (positive_gap && self.left_side && !self.inverted)
            || (positive_gap && !self.left_side && self.inverted)
            || (!positive_gap && self.left_side && self.inverted)
            || (!positive_gap && !self.left_side && !self.inverted)
        {
            if !stay_in_ok_range || self.orig + self.adjustment < self.ok_max {
                self.adjustment += 1;
                return true;
            }
            return false;
        }
        if (positive_gap && !self.left_side && !self.inverted)
            || (positive_gap && self.left_side && self.inverted)
            || (!positive_gap && self.left_side && !self.inverted)
            || (!positive_gap && !self.left_side && self.inverted)
        {
            if !stay_in_ok_range || self.orig + self.adjustment > self.ok_min {
                self.adjustment -= 1;
                return true;
            }
            return false;
        }
        false
    }
    pub fn r#move(&mut self, adjusted: i32) -> i32 {
        let new_adjustment = adjusted - self.orig;
        let change = new_adjustment - self.adjustment;
        self.adjustment = new_adjustment;
        if change == 0 {
            return 0;
        }
        if (change > 0 && !self.left_side && !self.inverted)
            || (change > 0 && self.left_side && self.inverted)
            || (change < 0 && self.left_side && !self.inverted)
            || (change < 0 && !self.left_side && self.inverted)
        {
            return change.abs();
        }
        if (change > 0 && self.left_side && !self.inverted)
            || (change > 0 && !self.left_side && self.inverted)
            || (change < 0 && self.left_side && self.inverted)
            || (change < 0 && !self.left_side && !self.inverted)
        {
            return -change.abs();
        }
        unreachable!()
    }
    pub fn is_ok(&self) -> bool {
        let adjusted = self.orig + self.adjustment;
        adjusted >= self.ok_min && adjusted <= self.ok_max
    }
    pub fn get_adjustment_abs_value(&self) -> i32 {
        self.adjustment.abs()
    }
    pub fn get_adjusted(&self) -> i32 {
        self.orig + self.adjustment
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn row(gap: f64) -> BoundaryRow {
        BoundaryRow::new(
            1,
            &BoundaryRowMetaData {
                section_table_data: vec![
                    SectionTableRowData {
                        join_final_end: 5,
                        z_max: 8,
                        ..Default::default()
                    },
                    SectionTableRowData {
                        join_final_start: 4,
                        z_max: 8,
                        ..Default::default()
                    },
                ],
                ..Default::default()
            },
            &BoundaryRowScreenState {
                best_gap: BTreeMap::from([(1, gap)]),
                ..Default::default()
            },
            BoundaryTable::default(),
        )
    }
    #[test]
    fn positive_gap_is_shared_between_boundaries() {
        let row = row(3.0);
        assert_eq!(
            (
                row.adjusted_end.get_int_value(),
                row.adjusted_start.get_int_value()
            ),
            (7, 3)
        );
    }
    #[test]
    fn edited_left_boundary_preserves_gap_by_moving_right() {
        let mut row = row(4.0);
        row.adjusted_end.set_value(8);
        row.adjusted_end_state_changed();
        assert_eq!(row.adjusted_start.get_int_value(), 3);
    }
    #[test]
    fn metadata_and_screen_state_round_trip() {
        let mut row = row(-2.0);
        let mut metadata = BoundaryRowMetaData::default();
        let mut screen = BoundaryRowScreenState::default();
        row.get_meta_data(&mut metadata);
        row.get_screen_state(&mut screen);
        assert_eq!(metadata.boundary_row_end.get(&1), Some(&4));
        assert_eq!(screen.best_gap.get(&1), Some(&-2.0));
    }
    #[test]
    fn model_and_rejoin_display_follow_viewport_and_tab() {
        let mut row = row(0.0);
        row.display(1, &Viewport::new(1), Tab::Model);
        assert!(!row.model_displayed);
        row.display(0, &Viewport::new(1), Tab::Rejoin);
        assert!(row.rejoin_displayed && row.adjusted_end.spinner.visible);
    }
}
