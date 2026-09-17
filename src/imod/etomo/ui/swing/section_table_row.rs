//! `IMOD/Etomo/src/etomo/ui/swing/SectionTableRow.java`.
//!
//! Swing widgets, JoinManager/3dmod calls, and message dialogs are explicit UI
//! boundaries.  This unit owns the row's source fields and preserves the Java
//! row's display, validation, highlighting, section-path, sample-range, and
//! synchronization state.
#![allow(dead_code)]

use std::path::{Path, PathBuf};

use super::colors::{FOREGROUND, HIGHLIGHT_BACKGROUND};
use super::field_cell::FieldCell;
use super::highlightable::Highlightable;
use super::highlighter_button::HighlighterButton;
use super::section_table_panel::{
    CHANGING_SAMPLE_MODE, ConstSectionTableRowData, SAMPLE_NOT_PRODUCED_MODE, SAMPLE_PRODUCED_MODE,
    SETUP_MODE, SectionTableRowData, Tab, Viewport,
};

pub const INVERTED_WARNING: &str = "The handedness of structures will change in inverted sections.";

/// Java `HeaderCell currentChunk` state, whose Swing component is a boundary.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CurrentChunk {
    pub text: String,
    pub displayed: bool,
    pub tooltip: String,
}

/// Java package-private final `SectionTableRow`.
#[derive(Clone, Debug)]
pub struct SectionTableRow {
    pub setup_section: FieldCell,
    pub join_section: FieldCell,
    pub sample_bottom_start: FieldCell,
    pub sample_bottom_end: FieldCell,
    pub sample_top_start: FieldCell,
    pub sample_top_end: FieldCell,
    pub slices_in_sample: FieldCell,
    pub current_chunk: CurrentChunk,
    pub reference_section: FieldCell,
    pub current_section: FieldCell,
    pub setup_final_start: FieldCell,
    pub setup_final_end: FieldCell,
    pub join_final_start: FieldCell,
    pub join_final_end: FieldCell,
    pub rotation_angle_x: FieldCell,
    pub rotation_angle_y: FieldCell,
    pub rotation_angle_z: FieldCell,
    pub data: SectionTableRowData,
    pub highlighter_button: HighlighterButton,
    pub imod_index: i32,
    pub imod_rot_index: i32,
    pub section_expanded: bool,
    pub valid: bool,
    pub inverted: bool,
    pub mode: i32,
    pub displayed: bool,
    pub displayed_tab: Option<Tab>,
    /// Application-boundary call recording, preserving the existing panel API.
    pub imod_opened: Option<(bool, i32)>,
    pub imod_removed: bool,
    pub angles_retrieved: bool,
    pub setup_to_join_synchronized: bool,
    pub join_to_setup_synchronized: bool,
    pub warnings: Vec<String>,
}

impl Default for SectionTableRow {
    fn default() -> Self {
        Self {
            setup_section: FieldCell::get_ineditable_instance(),
            join_section: FieldCell::get_ineditable_instance(),
            sample_bottom_start: FieldCell::get_editable_instance(),
            sample_bottom_end: FieldCell::get_editable_instance(),
            sample_top_start: FieldCell::get_editable_instance(),
            sample_top_end: FieldCell::get_editable_instance(),
            slices_in_sample: FieldCell::get_ineditable_instance(),
            current_chunk: CurrentChunk::default(),
            reference_section: FieldCell::get_ineditable_instance(),
            current_section: FieldCell::get_ineditable_instance(),
            setup_final_start: FieldCell::get_editable_instance(),
            setup_final_end: FieldCell::get_editable_instance(),
            join_final_start: FieldCell::get_editable_instance(),
            join_final_end: FieldCell::get_editable_instance(),
            rotation_angle_x: FieldCell::get_editable_instance(),
            rotation_angle_y: FieldCell::get_editable_instance(),
            rotation_angle_z: FieldCell::get_editable_instance(),
            data: SectionTableRowData::default(),
            highlighter_button: HighlighterButton::get_instance(0, Some(0)),
            imod_index: -1,
            imod_rot_index: -1,
            section_expanded: false,
            valid: true,
            inverted: false,
            mode: SETUP_MODE,
            displayed: false,
            displayed_tab: None,
            imod_opened: None,
            imod_removed: false,
            angles_retrieved: false,
            setup_to_join_synchronized: false,
            join_to_setup_synchronized: false,
            warnings: Vec::new(),
        }
    }
}

impl SectionTableRow {
    /// Java constructor `(JoinManager, SectionTablePanel, int, File, boolean)`.
    pub fn new(row_number: usize, section: impl Into<PathBuf>, section_expanded: bool) -> Self {
        let mut row = Self {
            data: SectionTableRowData {
                row_index: row_number.saturating_sub(1),
                setup_section: section.into(),
                valid: true,
                ..Default::default()
            },
            section_expanded,
            ..Default::default()
        };
        row.display_data(row_number);
        row.set_tool_tip_text();
        row
    }

    /// Java constructor `(JoinManager, SectionTablePanel, SectionTableRowData, boolean)`.
    pub fn from_data(data: SectionTableRowData, section_expanded: bool) -> Self {
        let row_number = data.row_index + 1;
        let mut row = Self {
            data,
            section_expanded,
            ..Default::default()
        };
        row.display_data(row_number);
        row.set_tool_tip_text();
        row
    }

    pub fn to_string(&self) -> String {
        format!("[{}]", self.setup_section)
    }

    pub fn set_names(&mut self) {
        let row = (self.data.row_index + 1).to_string();
        for (field, header) in [
            (&mut self.sample_bottom_start, "Sample Slices"),
            (&mut self.sample_bottom_end, "Sample Slices"),
            (&mut self.sample_top_start, "Sample Slices"),
            (&mut self.sample_top_end, "Sample Slices"),
            (&mut self.rotation_angle_x, "Rotation Angles"),
            (&mut self.rotation_angle_y, "Rotation Angles"),
            (&mut self.rotation_angle_z, "Rotation Angles"),
            (&mut self.join_final_start, "Final"),
            (&mut self.join_final_end, "Final"),
        ] {
            field.set_name_three(Some("Section Table"), Some(&row), Some(header));
        }
    }

    pub fn set_in_use(&mut self) {}
    pub fn set_in_use_for(&mut self, tab: Tab, size: usize) {
        let row = self.data.row_index + 1;
        if tab == Tab::Setup {
            let bottom = row > 1;
            let top = row < size;
            for field in [&mut self.sample_bottom_start, &mut self.sample_bottom_end] {
                field.set_in_use(bottom);
            }
            for field in [&mut self.sample_top_start, &mut self.sample_top_end] {
                field.set_in_use(top);
            }
            self.setup_final_start.set_in_use(false);
            self.setup_final_end.set_in_use(false);
        } else if matches!(tab, Tab::Join | Tab::Rejoin) {
            self.join_final_start.set_in_use(true);
            self.join_final_end.set_in_use(true);
        }
    }

    pub fn is_rotated(&self) -> bool {
        self.data.rotated
    }
    pub fn remove(&mut self) {
        self.displayed = false;
        self.highlighter_button.remove();
        self.current_chunk.displayed = false;
    }
    pub fn remove_imod(&mut self) {
        self.imod_removed = true;
        self.imod_index = -1;
        self.imod_rot_index = -1;
    }
    pub fn set_join_final_start_highlight(&mut self, highlight: bool) {
        if highlight || !self.is_highlighted() {
            self.join_final_start.text_field.foreground = if highlight {
                HIGHLIGHT_BACKGROUND
            } else {
                FOREGROUND
            };
        }
    }
    pub fn set_join_final_end_highlight(&mut self, highlight: bool) {
        if highlight || !self.is_highlighted() {
            self.join_final_end.text_field.foreground = if highlight {
                HIGHLIGHT_BACKGROUND
            } else {
                FOREGROUND
            };
        }
    }
    #[allow(non_snake_case)]
    pub fn setCellHighlight(&mut self, highlight: bool, cell: &mut FieldCell) {
        if !highlight && self.is_highlighted() {
            return;
        }
        cell.text_field.foreground = if highlight {
            HIGHLIGHT_BACKGROUND
        } else {
            FOREGROUND
        };
    }
    #[allow(non_snake_case)]
    pub fn getPrevSampleEnd(previous: Option<&SectionTableRow>) -> i32 {
        previous.map_or(0, |row| row.slices_in_sample.get_end_value())
    }
    #[allow(non_snake_case)]
    pub fn getBottomSampleSlices(&self, previous: Option<&SectionTableRow>) -> i32 {
        previous.map_or(0, |_| {
            self.total_in_range(self.data.sample_bottom_start, self.data.sample_bottom_end)
        })
    }
    #[allow(non_snake_case)]
    pub fn getTopSampleSlices(&self, total_rows: usize) -> i32 {
        if self.data.row_index + 1 == total_rows {
            0
        } else {
            self.total_in_range(self.data.sample_top_start, self.data.sample_top_end)
        }
    }
    #[allow(non_snake_case)]
    pub fn getPrevTopSampleSlices(previous: Option<&SectionTableRow>) -> i32 {
        previous.map_or(0, |row| {
            row.total_in_range(row.data.sample_top_start, row.data.sample_top_end)
        })
    }
    #[allow(non_snake_case)]
    pub fn addSetup(&mut self) {
        self.displayed_tab = Some(Tab::Setup);
    }
    #[allow(non_snake_case)]
    pub fn addAlign(&mut self) {
        self.displayed_tab = Some(Tab::Align);
    }
    #[allow(non_snake_case)]
    pub fn addJoin(&mut self) {
        self.displayed_tab = Some(Tab::Join);
    }
    #[allow(non_snake_case)]
    pub fn addRejoin(&mut self) {
        self.displayed_tab = Some(Tab::Rejoin);
        self.join_final_start.set_editable(false);
        self.join_final_end.set_editable(false);
    }

    pub fn set_mode(&mut self, mode: i32) {
        self.mode = mode;
        let editable = match mode {
            SAMPLE_PRODUCED_MODE => false,
            SETUP_MODE | SAMPLE_NOT_PRODUCED_MODE | CHANGING_SAMPLE_MODE => true,
            _ => panic!("mode={mode}"),
        };
        for field in [
            &mut self.sample_bottom_start,
            &mut self.sample_bottom_end,
            &mut self.sample_top_start,
            &mut self.sample_top_end,
            &mut self.rotation_angle_x,
            &mut self.rotation_angle_y,
            &mut self.rotation_angle_z,
        ] {
            field.set_editable(editable);
        }
    }

    pub fn set_inverted(&mut self, inverted: bool) {
        self.inverted = inverted;
        self.data.inverted = inverted;
        if inverted {
            self.warnings
                .push(format!("This section is inverted.  {INVERTED_WARNING}"));
        }
    }

    pub fn setup_cur_tab(&mut self, previous: Option<&SectionTableRow>, total_rows: usize) {
        let previous_end = previous.map_or(0, |row| row.slices_in_sample.get_end_value());
        let bottom = previous.map_or(0, |_| {
            self.total_in_range(self.data.sample_bottom_start, self.data.sample_bottom_end)
        });
        let top = if self.data.row_index + 1 == total_rows {
            0
        } else {
            self.total_in_range(self.data.sample_top_start, self.data.sample_top_end)
        };
        if previous.is_none() {
            self.current_chunk.text.clear();
            self.current_section.clear();
            self.reference_section.clear();
        } else {
            self.current_chunk.text = (self.data.row_index + 1).to_string();
            self.current_section
                .set_range_value(previous_end + 1, previous_end + bottom);
            let prior_top = previous.map_or(0, |row| {
                row.total_in_range(row.data.sample_top_start, row.data.sample_top_end)
            });
            self.reference_section
                .set_range_value(previous_end - prior_top + 1, previous_end);
        }
        self.slices_in_sample
            .set_range_value(previous_end + 1, previous_end + bottom + top);
    }

    pub fn display(&mut self, index: usize, viewport: &Viewport) {
        self.displayed = viewport.in_viewport(index);
        if self.displayed {
            self.highlighter_button.add();
        } else {
            self.highlighter_button.remove();
        }
    }
    pub fn display_for(&mut self, index: usize, tab: Tab, viewport: &Viewport) {
        self.display(index, viewport);
        if self.displayed {
            self.displayed_tab = Some(tab);
            if tab == Tab::Rejoin {
                self.join_final_start.set_editable(false);
                self.join_final_end.set_editable(false);
            }
        }
    }
    pub fn is_highlighted(&self) -> bool {
        self.highlighter_button.is_highlighted()
    }
    pub fn select_highlight_button(&mut self) {
        let highlight = self.highlighter_button.set_selected(true);
        self.highlight(highlight);
    }
    pub fn highlight(&mut self, highlight: bool) {
        self.highlighter_button.set_selected(highlight);
    }
    pub fn expand_section(&mut self, expand: bool) {
        self.section_expanded = expand;
        self.set_section_text();
    }
    pub fn swap_bottom_top(&mut self) {
        std::mem::swap(
            &mut self.sample_bottom_start.text_field.text,
            &mut self.sample_top_start.text_field.text,
        );
        std::mem::swap(
            &mut self.sample_bottom_end.text_field.text,
            &mut self.sample_top_end.text_field.text,
        );
    }
    pub fn set_row_number(&mut self, row_number: usize) {
        self.data.row_index = row_number.saturating_sub(1);
    }
    pub fn set_rotation_angles(&mut self, x: impl ToString, y: impl ToString, z: impl ToString) {
        self.rotation_angle_x.set_value(&x.to_string());
        self.rotation_angle_y.set_value(&y.to_string());
        self.rotation_angle_z.set_value(&z.to_string());
    }
    pub fn is_valid(&self) -> bool {
        self.valid
    }
    pub fn get_data(&mut self) -> &ConstSectionTableRowData {
        self.retrieve_data(true);
        &self.data
    }
    pub fn get_invalid_reason(&self) -> Option<&str> {
        self.data.invalid_reason.as_deref()
    }
    pub fn get_setup_section_file(&self) -> &Path {
        &self.data.setup_section
    }
    pub fn get_join_section_file(&self) -> Option<&Path> {
        self.data.join_section.as_deref()
    }
    pub fn get_setup_section_text(&self) -> String {
        self.setup_section.get_value().to_owned()
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
    pub fn equals_setup_section(&self, section: &Path) -> bool {
        self.data.setup_section == section
    }
    pub fn equals_join_section(&self, section: &Path) -> bool {
        self.data.join_section.as_deref() == Some(section)
    }
    pub fn equals(&self, data: &ConstSectionTableRowData) -> bool {
        self.data == *data
    }
    pub fn equals_sample(&self, data: &ConstSectionTableRowData) -> bool {
        self.data.setup_section == data.setup_section
            && self.data.sample_bottom_start == data.sample_bottom_start
            && self.data.sample_bottom_end == data.sample_bottom_end
            && self.data.sample_top_start == data.sample_top_start
            && self.data.sample_top_end == data.sample_top_end
    }
    pub fn imod_open_setup_section_file(&mut self, binning: i32) {
        self.imod_index = self.imod_index.max(0);
        self.imod_opened = Some((true, binning));
    }
    pub fn imod_open_join_section_file(&mut self, binning: i32) {
        if self.is_rotated() {
            self.imod_rot_index = self.imod_rot_index.max(0);
        } else {
            self.imod_index = self.imod_index.max(0);
        }
        self.imod_opened = Some((false, binning));
    }
    pub fn imod_get_angles(&mut self) -> bool {
        if self.imod_index < 0 {
            self.warnings
                .push("Open in 3dmod and use the Slicer to change the angles.".into());
            return false;
        }
        self.angles_retrieved = true;
        true
    }
    pub fn synchronize_setup_to_join(&mut self) {
        self.retrieve_data(true);
        self.data.join_section = Some(self.data.setup_section.clone());
        self.setup_to_join_synchronized = true;
        self.display_data_values();
    }
    pub fn synchronize_join_to_setup(&mut self) {
        self.retrieve_data(true);
        if let Some(section) = &self.data.join_section {
            self.data.setup_section = section.clone();
        }
        self.join_to_setup_synchronized = true;
        self.display_data_values();
    }
    pub fn validate_makejoincom(&mut self, max_row: &str) -> bool {
        self.retrieve_data(false);
        self.validate_pair(
            self.data.sample_bottom_start,
            self.data.sample_bottom_end,
            true,
            self.data.row_index == 0,
        ) && self.validate_pair(
            self.data.sample_top_start,
            self.data.sample_top_end,
            true,
            (self.data.row_index + 1).to_string() == max_row,
        )
    }
    pub fn validate_finishjoin(&mut self) -> bool {
        self.retrieve_data(false);
        self.validate_pair(
            Some(self.data.join_final_start),
            Some(self.data.join_final_end),
            false,
            true,
        )
    }

    fn total_in_range(&self, start: Option<i32>, end: Option<i32>) -> i32 {
        match (start, end) {
            (Some(start), Some(end)) => end - start + 1,
            _ => 0,
        }
    }
    fn display_data(&mut self, row_number: usize) {
        self.data.row_index = row_number.saturating_sub(1);
        self.display_data_values();
    }
    fn display_data_values(&mut self) {
        self.set_section_text();
        for (field, value) in [
            (&mut self.sample_bottom_start, self.data.sample_bottom_start),
            (&mut self.sample_bottom_end, self.data.sample_bottom_end),
            (&mut self.sample_top_start, self.data.sample_top_start),
            (&mut self.sample_top_end, self.data.sample_top_end),
            (&mut self.setup_final_start, self.data.setup_final_start),
            (&mut self.setup_final_end, self.data.setup_final_end),
            (&mut self.join_final_start, Some(self.data.join_final_start)),
            (&mut self.join_final_end, Some(self.data.join_final_end)),
        ] {
            field.set_value(&value.map_or_else(String::new, |value| value.to_string()));
        }
        self.rotation_angle_x.set_value(&self.data.rotation_angle_x);
        self.rotation_angle_y.set_value(&self.data.rotation_angle_y);
        self.rotation_angle_z.set_value(&self.data.rotation_angle_z);
        self.set_inverted(self.data.inverted);
    }
    fn retrieve_data(&mut self, _display_error_message: bool) -> bool {
        self.data.inverted = self.inverted;
        self.valid = true;
        for (text, destination) in [
            (
                self.sample_bottom_start.get_value(),
                &mut self.data.sample_bottom_start,
            ),
            (
                self.sample_bottom_end.get_value(),
                &mut self.data.sample_bottom_end,
            ),
            (
                self.sample_top_start.get_value(),
                &mut self.data.sample_top_start,
            ),
            (
                self.sample_top_end.get_value(),
                &mut self.data.sample_top_end,
            ),
            (
                self.setup_final_start.get_value(),
                &mut self.data.setup_final_start,
            ),
            (
                self.setup_final_end.get_value(),
                &mut self.data.setup_final_end,
            ),
        ] {
            match text.trim() {
                "" => *destination = None,
                value => match value.parse() {
                    Ok(value) => *destination = Some(value),
                    Err(_) => {
                        self.valid = false;
                        self.data.invalid_reason = Some(format!(
                            "Invalid number in section {}",
                            self.data.row_index + 1
                        ));
                    }
                },
            }
        }
        for (text, destination) in [
            (
                self.join_final_start.get_value(),
                &mut self.data.join_final_start,
            ),
            (
                self.join_final_end.get_value(),
                &mut self.data.join_final_end,
            ),
        ] {
            match text.trim().parse() {
                Ok(value) => *destination = value,
                Err(_) => {
                    self.valid = false;
                    self.data.invalid_reason = Some(format!(
                        "Invalid number in section {}",
                        self.data.row_index + 1
                    ));
                }
            }
        }
        self.data.rotation_angle_x = self.rotation_angle_x.get_value().to_owned();
        self.data.rotation_angle_y = self.rotation_angle_y.get_value().to_owned();
        self.data.rotation_angle_z = self.rotation_angle_z.get_value().to_owned();
        self.data.valid = self.valid;
        self.valid
    }
    fn validate_pair(
        &mut self,
        start: Option<i32>,
        end: Option<i32>,
        validate_values: bool,
        optional: bool,
    ) -> bool {
        self.valid = match (start, end) {
            (None, Some(_)) | (Some(_), None) => false,
            (None, None) => optional,
            (Some(start), Some(end)) => !validate_values || start <= end,
        };
        if !self.valid {
            self.data.invalid_reason = Some(format!(
                "Invalid numbers in section {}",
                self.data.row_index + 1
            ));
        }
        self.valid
    }
    fn set_section_text(&mut self) {
        let setup = &self.data.setup_section;
        self.setup_section.set_value(&if self.section_expanded {
            setup.display().to_string()
        } else {
            setup
                .file_name()
                .map_or_else(String::new, |name| name.to_string_lossy().into_owned())
        });
        if let Some(join) = &self.data.join_section {
            self.join_section.set_value(&if self.section_expanded {
                join.display().to_string()
            } else {
                join.file_name()
                    .map_or_else(String::new, |name| name.to_string_lossy().into_owned())
            });
        }
    }
    fn set_tool_tip_text(&mut self) {
        self.highlighter_button
            .set_tool_tip_text("Press to select the section.");
        self.current_chunk.tooltip = "The number of the chunk in Midas.".into();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn section_expansion_and_sample_validation_follow_source() {
        let mut row = SectionTableRow::new(2, "/work/section.mrc", false);
        assert_eq!(row.get_setup_section_text(), "section.mrc");
        row.expand_section(true);
        assert_eq!(row.get_setup_section_text(), "/work/section.mrc");
        row.sample_bottom_start.set_value("8");
        row.sample_bottom_end.set_value("4");
        assert!(!row.validate_makejoincom("2"));
        row.sample_bottom_end.set_value("9");
        assert!(row.validate_makejoincom("2"));
    }
    #[test]
    fn align_ranges_swap_and_sync_preserve_row_state() {
        let mut first = SectionTableRow::new(1, "a", false);
        first.data.sample_top_start = Some(2);
        first.data.sample_top_end = Some(3);
        first.slices_in_sample.set_range_value(1, 5);
        let mut second = SectionTableRow::new(2, "b", false);
        second.data.sample_bottom_start = Some(4);
        second.data.sample_bottom_end = Some(6);
        second.setup_cur_tab(Some(&first), 2);
        assert_eq!(second.current_section.get_value(), "6 - 8");
        second.sample_bottom_start.set_value("1");
        second.sample_top_start.set_value("2");
        second.swap_bottom_top();
        assert_eq!(second.sample_bottom_start.get_value(), "2");
        second.synchronize_setup_to_join();
        assert_eq!(second.get_join_section_file(), Some(Path::new("b")));
    }
}
