//! `IMOD/Etomo/src/etomo/ui/swing/BoundaryRow.java`.
//!
//! One row of the Join dialog's boundary table: a boundary between two sections, its
//! xfjointomo best gap and errors, the original end/start of the two sections, and
//! the adjusted end/start spinners, which keep the gap between them.  An event
//! dispatch thread object, created as `Rc<Self>`; the table owns it.

use std::cell::RefCell;
use std::rc::{Rc, Weak};

use super::boundary_table::{self, BoundaryTable};
use super::cell::CellVirtual;
use super::field_cell::FieldCell;
use super::header_cell::HeaderCell;
use super::join_dialog::Tab;
use super::spinner_cell::SpinnerCell;
use super::viewport::Viewport;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::jdk::{ChangeEvent, ChangeListener, GRID_BAG_REMAINDER, JComponent};
use crate::imod::etomo::storage::log_file::LogFileError;
use crate::imod::etomo::storage::xfjointomo_log::XfjointomoLog;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_join_meta_data::ConstJoinMetaData;
use crate::imod::etomo::r#type::const_section_table_row_data::ConstSectionTableRowData;
use crate::imod::etomo::r#type::join_meta_data::JoinMetaData;
use crate::imod::etomo::r#type::join_screen_state::JoinScreenState;
use crate::imod::etomo::util::utilities;

/// Java private static final `INVERTED_TOOLTIP`.
const INVERTED_TOOLTIP: &str = "This value comes from an inverted section.";
/// Java private static final `EMPTY_SLICE_WARNING`.
const EMPTY_SLICE_WARNING: &str = "Empty slices will be added to the section.";

/// Java package-private `final class BoundaryRow`.
pub struct BoundaryRow {
    /// Java private final `boundary = new HeaderCell()`.
    boundary: Rc<HeaderCell>,
    /// Java private final `sections = new HeaderCell()`.
    sections: Rc<HeaderCell>,
    /// Java private final `bestGap = FieldCell.getIneditableInstance()`.
    best_gap: Rc<FieldCell>,
    /// Java private final `meanError = FieldCell.getIneditableInstance()`.
    mean_error: Rc<FieldCell>,
    /// Java private final `maxError = FieldCell.getIneditableInstance()`.
    max_error: Rc<FieldCell>,
    /// Java private final `origEnd = FieldCell.getIneditableInstance()`.
    orig_end: Rc<FieldCell>,
    /// Java private final `origStart = FieldCell.getIneditableInstance()`.
    orig_start: Rc<FieldCell>,
    /// Java private final `adjustedEnd`.
    adjusted_end: Rc<SpinnerCell>,
    /// Java private final `adjustedStart`.
    adjusted_start: Rc<SpinnerCell>,
    /// Java private final `zMaxEnd`.
    z_max_end: i32,
    /// Java private final `zMaxStart`.
    z_max_start: i32,
    // `panel`, `layout` and `constraints` are the table's (`table`).
    /// Java private final `adjustedEndChangeListener`.
    adjusted_end_change_listener: ChangeListener,
    /// Java private final `adjustedStartChangeListener`.
    adjusted_start_change_listener: ChangeListener,
    /// Java private final `table` (the table owns the row).
    table: Weak<BoundaryTable>,
    /// Java private `gap`, initially null.
    gap: RefCell<Option<Gap>>,
    /// Java private `endInverted`.
    end_inverted: bool,
    /// Java private `startInverted`.
    start_inverted: bool,
}

impl BoundaryRow {
    /// Java package-private `BoundaryRow(int, ConstJoinMetaData, JoinScreenState,
    /// JPanel, GridBagLayout, GridBagConstraints, BoundaryTable)`.
    pub fn new(
        key: i32,
        meta_data: &dyn ConstJoinMetaData,
        screen_state: &JoinScreenState,
        table: &Rc<BoundaryTable>,
    ) -> Rc<BoundaryRow> {
        let this = Rc::new_cyclic(|self_ref: &Weak<BoundaryRow>| {
            let boundary = HeaderCell::new_void();
            let sections = HeaderCell::new_void();
            let best_gap = FieldCell::get_ineditable_instance();
            let mean_error = FieldCell::get_ineditable_instance();
            let max_error = FieldCell::get_ineditable_instance();
            let orig_end = FieldCell::get_ineditable_instance();
            let orig_start = FieldCell::get_ineditable_instance();
            // boundary
            let first_section = key.to_string();
            boundary.set_text_string(Some(&first_section));
            sections.set_text_string(Some(&format!(
                "{} & {}",
                first_section,
                key.wrapping_add(1)
            )));
            // bestGap
            best_gap.set_value_string(screen_state.get_best_gap(key).as_deref());
            // meanError
            mean_error.set_value_string(screen_state.get_mean_error(key).as_deref());
            // maxError
            max_error.set_value_string(screen_state.get_max_error(key).as_deref());
            // origEnd
            let section_table_data = meta_data.get_section_table_data().unwrap_or_default();
            let data = &section_table_data[(key - 1) as usize];
            orig_end.set_value_int(data.get_join_final_end().get_int());
            let end_inverted = data.get_inverted().is();
            if end_inverted {
                orig_end.set_warning_boolean_string(true, Some(INVERTED_TOOLTIP));
            } else {
                orig_end.set_warning_boolean_string(false, None);
            }
            let z_max_end = data.get_setup_z_max();
            // origStart
            let data = &section_table_data[key as usize];
            orig_start.set_value_int(data.get_join_final_start().get_int());
            let start_inverted = data.get_inverted().is();
            if start_inverted {
                orig_start.set_warning_boolean_string(true, Some(INVERTED_TOOLTIP));
            } else {
                orig_start.set_warning_boolean_string(false, None);
            }
            let z_max_start = data.get_setup_z_max();
            // adjustedEnd and adjustedStart
            let adjusted_end = SpinnerCell::get_int_instance(
                z_max_end.wrapping_mul(2).wrapping_mul(-1),
                z_max_end.wrapping_mul(2),
            );
            let adjusted_start = SpinnerCell::get_int_instance(
                z_max_start.wrapping_mul(2).wrapping_mul(-1),
                z_max_start.wrapping_mul(2),
            );
            // listeners
            let adaptee = self_ref.clone();
            let adjusted_end_change_listener: ChangeListener =
                Rc::new(move |event: &ChangeEvent| {
                    if let Some(adaptee) = adaptee.upgrade() {
                        adaptee.adjusted_end_state_changed(event);
                    }
                });
            let adaptee = self_ref.clone();
            let adjusted_start_change_listener: ChangeListener =
                Rc::new(move |event: &ChangeEvent| {
                    if let Some(adaptee) = adaptee.upgrade() {
                        adaptee.adjusted_start_state_changed(event);
                    }
                });
            BoundaryRow {
                boundary,
                sections,
                best_gap,
                mean_error,
                max_error,
                orig_end,
                orig_start,
                adjusted_end,
                adjusted_start,
                z_max_end,
                z_max_start,
                adjusted_end_change_listener,
                adjusted_start_change_listener,
                table: Rc::downgrade(table),
                gap: RefCell::new(None),
                end_inverted,
                start_inverted,
            }
        });
        this.set_adjusted_values(Some(meta_data));
        // listeners
        this.adjusted_end
            .add_change_listener(this.adjusted_end_change_listener.clone());
        this.adjusted_start
            .add_change_listener(this.adjusted_start_change_listener.clone());
        this
    }

    /// Java field read `table`.
    fn table(&self) -> Rc<BoundaryTable> {
        self.table
            .upgrade()
            .expect("the boundary table owns its rows")
    }

    /// Java package-private `setNames()`.
    pub fn set_names(&self) {
        let adjusted_header_cell = self.table().get_adjusted_header_cell();
        self.adjusted_start.set_headers(
            Some(boundary_table::TABLE_LABEL),
            &self.sections,
            &adjusted_header_cell,
        );
        self.adjusted_end.set_headers(
            Some(boundary_table::TABLE_LABEL),
            &self.sections,
            &adjusted_header_cell,
        );
    }

    /// Java private synchronized `setAdjustedValues(ConstJoinMetaData)`.  Sets
    /// adjustedEnd and adjustedStart.  Should be called whenever best gap is Changed.
    fn set_adjusted_values(&self, meta_data: Option<&dyn ConstJoinMetaData>) {
        self.adjusted_end
            .remove_change_listener(&self.adjusted_end_change_listener);
        self.adjusted_start
            .remove_change_listener(&self.adjusted_start_change_listener);
        let gap = Gap::new(
            utilities::java_lang_math_round(self.best_gap.get_double_value()) as i32,
            self.orig_end.get_int_value(),
            self.orig_start.get_int_value(),
            self.end_inverted,
            self.start_inverted,
            self.z_max_end,
            self.z_max_start,
        );
        *self.gap.borrow_mut() = Some(gap);
        let end_number = match meta_data {
            None => None,
            Some(meta_data) if meta_data.is_boundary_row_end_list_empty() => None,
            Some(meta_data) => meta_data.get_boundary_row_end(self.boundary.get_int()),
        };
        match end_number {
            None => {
                let gap = self.gap.borrow();
                let gap = gap.as_ref().unwrap();
                self.adjusted_end.set_value_int(gap.get_adjusted_left());
                self.adjusted_start.set_value_int(gap.get_adjusted_right());
            }
            Some(end_number) => {
                let end = end_number.get_int();
                self.adjusted_end.set_value_int(end);
                let mut gap = self.gap.borrow_mut();
                let gap = gap.as_mut().unwrap();
                gap.msg_left_gap_boundary_changed(end);
                self.adjusted_start.set_value_int(gap.get_adjusted_right());
            }
        }
        self.set_adjusted_value_warnings();
        self.adjusted_end
            .add_change_listener(self.adjusted_end_change_listener.clone());
        self.adjusted_start
            .add_change_listener(self.adjusted_start_change_listener.clone());
    }

    /// Java private `setAdjustedValueWarnings()`.
    fn set_adjusted_value_warnings(&self) {
        let gap = self.gap.borrow();
        let gap = gap.as_ref().unwrap();
        if gap.is_left_ok() {
            self.adjusted_end.set_warning_boolean_string(false, None);
        } else {
            self.adjusted_end
                .set_warning_boolean_string(true, Some(EMPTY_SLICE_WARNING));
        }
        if gap.is_right_ok() {
            self.adjusted_start.set_warning_boolean_string(false, None);
        } else {
            self.adjusted_start
                .set_warning_boolean_string(true, Some(EMPTY_SLICE_WARNING));
        }
    }

    /// Java private `calculateNegativeAdjustment(int, int)`.  Calculated negative
    /// adjustment for end and start.  Not called in the source.  The source's
    /// `IllegalStateException` is a panic.
    #[allow(dead_code)]
    fn calculate_negative_adjustment(rounded_best_gap: i32, orig: i32) -> i32 {
        if rounded_best_gap < 0 || orig <= 0 {
            panic!(
                "java.lang.IllegalStateException: Only pass the absolute value of \
                 roundedBestGap.  Orig is either origEnd or origStart and must be at least \
                 1.\nroundedBestGap={rounded_best_gap},orig={orig}"
            );
        }
        let mut adjustment = rounded_best_gap / 2;
        // Will have to add negative adjustment to orig and the result must be at
        // least 1.
        if adjustment > orig - 1 {
            adjustment = orig - 1;
        }
        // Make adjustment negative.
        adjustment * -1
    }

    /// Java package-private `display(int, Viewport, JoinDialog.Tab)`.
    pub fn display(&self, index: i32, viewport: &Viewport, tab: Option<Tab>) {
        if !viewport.in_viewport(index) {
            return;
        }
        if tab == Some(Tab::Model) {
            self.display_model();
        } else if tab == Some(Tab::Rejoin) {
            self.display_rejoin();
        }
    }

    /// `cell.add(panel, layout, constraints)`: the cell's `add` and the
    /// `layout.setConstraints` it makes.
    fn add_cell(&self, table: &BoundaryTable, cell: &dyn CellVirtual, component: Rc<JComponent>) {
        let panel = table.get_table_panel();
        cell.add(&panel);
        table
            .get_layout()
            .set_constraints(&component, &table.get_constraints());
    }

    /// Java private `displayModel()`.
    fn display_model(&self) {
        let table = self.table();
        table.with_constraints(|constraints| {
            constraints.weightx = 0.0;
            constraints.weighty = 0.1;
            constraints.gridwidth = 1;
        });
        self.add_cell(&table, &*self.boundary, self.boundary.get_component());
        table.with_constraints(|constraints| constraints.weightx = 0.1);
        self.add_cell(&table, &*self.best_gap, self.best_gap.get_component());
        self.add_cell(&table, &*self.mean_error, self.mean_error.get_component());
        table.with_constraints(|constraints| constraints.gridwidth = GRID_BAG_REMAINDER);
        self.add_cell(&table, &*self.max_error, self.max_error.get_component());
    }

    /// Java private `displayRejoin()`.
    fn display_rejoin(&self) {
        let table = self.table();
        table.with_constraints(|constraints| {
            constraints.weightx = 0.0;
            constraints.weighty = 0.1;
            constraints.gridwidth = 1;
        });
        self.add_cell(&table, &*self.sections, self.sections.get_component());
        table.with_constraints(|constraints| constraints.weightx = 0.1);
        self.add_cell(&table, &*self.orig_end, self.orig_end.get_component());
        self.add_cell(&table, &*self.orig_start, self.orig_start.get_component());
        self.add_cell(&table, &*self.best_gap, self.best_gap.get_component());
        self.add_cell(
            &table,
            &*self.adjusted_end,
            self.adjusted_end.get_component(),
        );
        table.with_constraints(|constraints| constraints.gridwidth = GRID_BAG_REMAINDER);
        self.add_cell(
            &table,
            &*self.adjusted_start,
            self.adjusted_start.get_component(),
        );
    }

    /// Java package-private `removeDisplay()`.
    ///
    /// Fixed in translation (BoundaryRow.java:240-248): the source removes `origEnd`
    /// twice and never removes `sections` or `origStart` (a copy-and-paste slip); every
    /// displayed cell is removed here.  The table empties its panel right after, so the
    /// display is the same.
    pub fn remove_display(&self) {
        self.boundary.remove();
        self.sections.remove();
        self.best_gap.remove();
        self.mean_error.remove();
        self.max_error.remove();
        self.orig_end.remove();
        self.orig_start.remove();
        self.adjusted_end.remove();
        self.adjusted_start.remove();
    }

    /// Java package-private `setXfjointomoResult(BaseManager) throws
    /// LogFileException, IOException, LockException`.
    pub fn set_xfjointomo_result(
        &self,
        manager: &'static dyn BaseManager,
    ) -> Result<(), LogFileError> {
        let xfjointomo_log = XfjointomoLog::get_instance(manager, AxisID::Only);
        let boundary = self
            .boundary
            .get_text()
            .unwrap_or_else(|| "null".to_string());
        if !xfjointomo_log.row_exists(&boundary)? {
            return Ok(());
        }
        self.best_gap
            .set_value_string(xfjointomo_log.get_best_gap(&boundary)?.as_deref());
        self.mean_error
            .set_value_string(xfjointomo_log.get_mean_error(&boundary).as_deref());
        self.max_error
            .set_value_string(xfjointomo_log.get_max_error(&boundary)?.as_deref());
        self.set_adjusted_values(None);
        Ok(())
    }

    /// Java static package-private `resetScreenState(JoinScreenState)`.
    pub fn reset_screen_state(screen_state: &JoinScreenState) {
        screen_state.reset_best_gap();
        screen_state.reset_mean_error();
        screen_state.reset_max_error();
    }

    /// Java package-private `getScreenState(JoinScreenState)`.
    pub fn get_screen_state(&self, screen_state: &JoinScreenState) {
        let key = self.boundary.get_int();
        screen_state.set_best_gap(key, self.best_gap.get_value().as_deref());
        screen_state.set_mean_error(key, self.mean_error.get_value().as_deref());
        screen_state.set_max_error(key, self.max_error.get_value().as_deref());
    }

    /// Java static package-private `resetMetaData(JoinMetaData)`.
    pub fn reset_meta_data(meta_data: &JoinMetaData) {
        meta_data.reset_boundary_row_start_list();
        meta_data.reset_boundary_row_end_list();
    }

    /// Java package-private `getMetaData(JoinMetaData)`.
    pub fn get_meta_data(&self, meta_data: &JoinMetaData) {
        meta_data.set_boundary_row_end(
            self.boundary.get_int(),
            self.adjusted_end.get_string_value().as_deref(),
        );
        meta_data.set_boundary_row_start(
            self.boundary.get_int(),
            self.adjusted_start.get_string_value().as_deref(),
        );
    }

    /// Java package-private synchronized `adjustedEndStateChanged(ChangeEvent)`.
    fn adjusted_end_state_changed(&self, _event: &ChangeEvent) {
        if self.gap.borrow().is_none() {
            self.set_adjusted_values(None);
        }
        self.adjusted_start
            .remove_change_listener(&self.adjusted_start_change_listener);
        let adjusted_right = {
            let mut gap = self.gap.borrow_mut();
            let gap = gap.as_mut().unwrap();
            gap.msg_left_gap_boundary_changed(self.adjusted_end.get_int_value());
            gap.get_adjusted_right()
        };
        self.adjusted_start.set_value_int(adjusted_right);
        self.set_adjusted_value_warnings();
        self.adjusted_start
            .add_change_listener(self.adjusted_start_change_listener.clone());
    }

    /// Java package-private synchronized `adjustedStartStateChanged(ChangeEvent)`.
    fn adjusted_start_state_changed(&self, _event: &ChangeEvent) {
        if self.gap.borrow().is_none() {
            self.set_adjusted_values(None);
        }
        self.adjusted_end
            .remove_change_listener(&self.adjusted_end_change_listener);
        let adjusted_left = {
            let mut gap = self.gap.borrow_mut();
            let gap = gap.as_mut().unwrap();
            gap.msg_right_gap_boundary_changed(self.adjusted_start.get_int_value());
            gap.get_adjusted_left()
        };
        self.adjusted_end.set_value_int(adjusted_left);
        self.set_adjusted_value_warnings();
        self.adjusted_end
            .add_change_listener(self.adjusted_end_change_listener.clone());
    }
}

/// Java private static final nested class `Gap`.
struct Gap {
    /// Java private final `gap` (stored, not read again).
    #[allow(dead_code)]
    gap: i32,
    /// Java private final `leftBoundary`.
    left_boundary: GapBoundary,
    /// Java private final `rightBoundary`.
    right_boundary: GapBoundary,
}

impl Gap {
    /// Java private `Gap(int, int, int, boolean, boolean, int, int)`.
    fn new(
        gap: i32,
        orig_left: i32,
        orig_right: i32,
        left_inverted: bool,
        right_inverted: bool,
        ok_max_left: i32,
        ok_max_right: i32,
    ) -> Gap {
        let mut this = Gap {
            gap,
            left_boundary: GapBoundary::new(orig_left, left_inverted, ok_max_left, true),
            right_boundary: GapBoundary::new(orig_right, right_inverted, ok_max_right, false),
        };
        if gap == 0 {
            return this;
        }
        let mut positive_gap = true;
        if gap < 0 {
            positive_gap = false;
        }
        // Increment the absolute values of the left and right boundary
        // until the absolute values of the adjustments equal the absolute gap.
        // Try to stay in the ok range.
        let abs_gap = gap.wrapping_abs();
        let mut left_succeeded;
        let mut right_succeeded = true;
        let mut stay_in_ok_range = true;
        while this.left_boundary.get_adjustment_abs_value()
            + this.right_boundary.get_adjustment_abs_value()
            < abs_gap
        {
            left_succeeded = this.left_boundary.adjust(positive_gap, stay_in_ok_range);
            if this.left_boundary.get_adjustment_abs_value()
                + this.right_boundary.get_adjustment_abs_value()
                < abs_gap
            {
                right_succeeded = this.right_boundary.adjust(positive_gap, stay_in_ok_range);
            }
            // see if have to go outside of ok range
            stay_in_ok_range = left_succeeded || right_succeeded;
        }
        this
    }

    /// Java `msgLeftGapBoundaryChanged(int)`.
    fn msg_left_gap_boundary_changed(&mut self, adjusted_left: i32) {
        let gap_change = self.left_boundary.move_to(adjusted_left);
        // The gap has been changed on the left side, change the right side to get
        // back to the original gap
        let increased_gap = gap_change > 0;
        let abs_gap_change = gap_change.wrapping_abs();
        for _ in 0..abs_gap_change {
            self.right_boundary.adjust(increased_gap, false);
        }
    }

    /// Java `msgRightGapBoundaryChanged(int)`.
    fn msg_right_gap_boundary_changed(&mut self, adjusted_right: i32) {
        let gap_change = self.right_boundary.move_to(adjusted_right);
        // The gap has been changed on the right side, change the left side to get
        // back to the original gap
        let increased_gap = gap_change > 0;
        let abs_gap_change = gap_change.wrapping_abs();
        for _ in 0..abs_gap_change {
            self.left_boundary.adjust(increased_gap, false);
        }
    }

    /// Java `getAdjustedLeft()`.
    fn get_adjusted_left(&self) -> i32 {
        self.left_boundary.get_adjusted()
    }

    /// Java `getAdjustedRight()`.
    fn get_adjusted_right(&self) -> i32 {
        self.right_boundary.get_adjusted()
    }

    /// Java `isLeftOk()`.
    fn is_left_ok(&self) -> bool {
        self.left_boundary.is_ok()
    }

    /// Java `isRightOk()`.
    fn is_right_ok(&self) -> bool {
        self.right_boundary.is_ok()
    }
}

/// Java private static final nested class `Gap.GapBoundary`.
struct GapBoundary {
    /// Java private final `orig`.
    orig: i32,
    /// Java private final `inverted`.
    inverted: bool,
    /// Java private final `leftSide`.
    left_side: bool,
    /// Java private final `okMin = 1`.
    ok_min: i32,
    /// Java private final `okMax`.
    ok_max: i32,
    /// Java private `adjustment`, initially 0.
    adjustment: i32,
}

impl GapBoundary {
    /// Java private `GapBoundary(int, boolean, int, boolean)`.
    fn new(orig: i32, inverted: bool, ok_max: i32, left_side: bool) -> GapBoundary {
        GapBoundary {
            orig,
            inverted,
            left_side,
            ok_min: 1,
            ok_max,
            adjustment: 0,
        }
    }

    /// Java `adjust(boolean, boolean)`.  Change adjustment by 1 to work with either a
    /// positive or negative gap.  Widen a section to fill in a positive gap.  Narrow
    /// a section to handle a negative gap.
    fn adjust(&mut self, positive_gap: bool, stay_in_ok_range: bool) -> bool {
        let left_side = self.left_side;
        let inverted = self.inverted;
        if (positive_gap && left_side && !inverted)
            || (positive_gap && !left_side && inverted)
            || (!positive_gap && left_side && inverted)
            || (!positive_gap && !left_side && !inverted)
        {
            if !stay_in_ok_range || self.orig + self.adjustment < self.ok_max {
                self.adjustment += 1;
                return true;
            }
            return false;
        }
        if (positive_gap && !left_side && !inverted)
            || (positive_gap && left_side && inverted)
            || (!positive_gap && left_side && !inverted)
            || (!positive_gap && !left_side && inverted)
        {
            if !stay_in_ok_range || self.orig + self.adjustment > self.ok_min {
                self.adjustment -= 1;
                return true;
            }
            return false;
        }
        false
    }

    /// Java `move(int)`.  Change the gap size by moving one of the boundaries.  Set
    /// the new adjustment and get the change in adjustment.  Returns the change in gap
    /// size -- positive if gap was increased, negative if gap was decreased.  The
    /// source's unreachable `IllegalStateException` is a panic.
    fn move_to(&mut self, adjusted: i32) -> i32 {
        let new_adjustment = adjusted.wrapping_sub(self.orig);
        let change = new_adjustment.wrapping_sub(self.adjustment);
        self.adjustment = new_adjustment;
        if change == 0 {
            return 0;
        }
        let left_side = self.left_side;
        let inverted = self.inverted;
        // Find out if this move increased the gap size or decreased it
        if (change > 0 && !left_side && !inverted)
            || (change > 0 && left_side && inverted)
            || (change < 0 && left_side && !inverted)
            || (change < 0 && !left_side && inverted)
        {
            return change.wrapping_abs();
        }
        if (change > 0 && left_side && !inverted)
            || (change > 0 && !left_side && inverted)
            || (change < 0 && left_side && inverted)
            || (change < 0 && !left_side && !inverted)
        {
            return change.wrapping_abs() * -1;
        }
        panic!("java.lang.IllegalStateException");
    }

    /// Java `isOk()`.
    fn is_ok(&self) -> bool {
        let adjusted = self.orig + self.adjustment;
        adjusted >= self.ok_min && adjusted <= self.ok_max
    }

    /// Java `getAdjustmentAbsValue()`.
    fn get_adjustment_abs_value(&self) -> i32 {
        self.adjustment.wrapping_abs()
    }

    /// Java `getAdjusted()`.
    fn get_adjusted(&self) -> i32 {
        self.orig + self.adjustment
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_positive_gap_widens_both_sections() {
        let gap = Gap::new(3, 10, 1, false, false, 20, 20);
        // The right start cannot shrink below 1 while the loop stays in the ok
        // range, and the left side still can, so the left end takes the whole
        // gap (Gap.java's loop: stayInOkRange = leftSucceeded || rightSucceeded).
        assert_eq!(gap.get_adjusted_left(), 13);
        assert_eq!(gap.get_adjusted_right(), 1);
        assert!(gap.is_left_ok());
        assert!(gap.is_right_ok());
    }
}
