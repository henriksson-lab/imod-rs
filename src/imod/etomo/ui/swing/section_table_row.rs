//! `IMOD/Etomo/src/etomo/ui/swing/SectionTableRow.java`.
//!
//! One row of the Join dialog's section table: the section file, its sample and
//! final slice ranges, its rotation angles, and the Align tab's display-only chunk
//! columns.  An event dispatch thread object, created as `Rc<Self>`; the table owns
//! it (the row keeps a weak reference to the table).

use std::cell::{Cell, RefCell};
use std::path::{Path, PathBuf};
use std::rc::{Rc, Weak};

use super::cell::CellVirtual;
use super::field_cell::FieldCell;
use super::header_cell::HeaderCell;
use super::highlightable::Highlightable;
use super::highlighter_button::HighlighterButton;
use super::input_cell::InputCell;
use super::section_table_panel::{self, SectionTablePanel};
use super::ui_harness;
use super::ui_parameters::UIParameters;
use super::viewport::Viewport;
use super::join_dialog;
use crate::imod::etomo::join_manager::JoinManager;
use crate::imod::etomo::jdk::{GRID_BAG_REMAINDER, JComponent};
use crate::imod::etomo::process::imod_manager;
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::ConstEtomoNumber;
use crate::imod::etomo::r#type::const_section_table_row_data::ConstSectionTableRowData;
use crate::imod::etomo::r#type::section_table_row_data::SectionTableRowData;
use crate::imod::etomo::r#type::slicer_angles::SlicerAngles;
use crate::imod::etomo::base_manager::BaseManager as _;
use crate::imod::etomo::util::dataset_files;
use crate::imod::etomo::util::utilities;

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java package-private static final `INVERTED_WARNING`.
pub const INVERTED_WARNING: &str = "The handedness of structures will change in inverted sections.";

/// Java package-private `final class SectionTableRow implements Highlightable`.
pub struct SectionTableRow {
    /// Java private final `setupSection = FieldCell.getIneditableInstance()`.
    setup_section: Rc<FieldCell>,
    /// Java private final `joinSection = FieldCell.getIneditableInstance()`.
    join_section: Rc<FieldCell>,
    /// Java private final `sampleBottomStart = FieldCell.getEditableInstance()`.
    sample_bottom_start: Rc<FieldCell>,
    /// Java private final `sampleBottomEnd = FieldCell.getEditableInstance()`.
    sample_bottom_end: Rc<FieldCell>,
    /// Java private final `sampleTopStart = FieldCell.getEditableInstance()`.
    sample_top_start: Rc<FieldCell>,
    /// Java private final `sampleTopEnd = FieldCell.getEditableInstance()`.
    sample_top_end: Rc<FieldCell>,
    /// Java private final `slicesInSample = FieldCell.getIneditableInstance()`.
    slices_in_sample: Rc<FieldCell>,
    /// Java private final `currentChunk = new HeaderCell()`.
    current_chunk: Rc<HeaderCell>,
    /// Java private final `referenceSection = FieldCell.getIneditableInstance()`.
    reference_section: Rc<FieldCell>,
    /// Java private final `currentSection = FieldCell.getIneditableInstance()`.
    current_section: Rc<FieldCell>,
    /// Java private final `setupFinalStart = FieldCell.getEditableInstance()`.
    setup_final_start: Rc<FieldCell>,
    /// Java private final `setupFinalEnd = FieldCell.getEditableInstance()`.
    setup_final_end: Rc<FieldCell>,
    /// Java private final `joinFinalStart = FieldCell.getEditableInstance()`.
    join_final_start: Rc<FieldCell>,
    /// Java private final `joinFinalEnd = FieldCell.getEditableInstance()`.
    join_final_end: Rc<FieldCell>,
    /// Java private final `rotationAngleX = FieldCell.getEditableInstance()`.
    rotation_angle_x: Rc<FieldCell>,
    /// Java private final `rotationAngleY = FieldCell.getEditableInstance()`.
    rotation_angle_y: Rc<FieldCell>,
    /// Java private final `rotationAngleZ = FieldCell.getEditableInstance()`.
    rotation_angle_z: Rc<FieldCell>,

    /// Java private final `manager`.
    manager: &'static JoinManager,
    /// Java private final `table` (the table owns the row).
    table: Weak<SectionTablePanel>,
    /// Java private final `highlighterButton`.
    highlighter_button: Rc<HighlighterButton>,

    /// Java private `data`.
    data: RefCell<SectionTableRowData>,
    /// Java private final `rowNumber = new HeaderCell((int) (30 *
    /// UIParameters.getInstance().getFontSizeAdjustment()))`.
    row_number: Rc<HeaderCell>,
    /// Java private `imodIndex`, initially -1.
    imod_index: Cell<i32>,
    /// Java private `imodRotIndex`, initially -1.
    imod_rot_index: Cell<i32>,
    /// Java private `sectionExpanded`, initially false.
    section_expanded: Cell<bool>,
    /// Java private `valid`, initially true.
    valid: Cell<bool>,
    /// Java private `inverted`, initially false.
    inverted: Cell<bool>,
}

impl SectionTableRow {
    /// Java private `SectionTableRow(JoinManager, SectionTablePanel, boolean)`, with
    /// the field initialisers; `data` is the value the public constructor gives it.
    fn construct(
        manager: &'static JoinManager,
        table: &Rc<SectionTablePanel>,
        section_expanded: bool,
        data: SectionTableRowData,
    ) -> Rc<SectionTableRow> {
        Rc::new_cyclic(|self_ref: &Weak<SectionTableRow>| {
            let parent: Weak<dyn Highlightable> = self_ref.clone();
            let group: Weak<dyn Highlightable> = Rc::downgrade(table) as Weak<dyn Highlightable>;
            SectionTableRow {
                setup_section: FieldCell::get_ineditable_instance(),
                join_section: FieldCell::get_ineditable_instance(),
                sample_bottom_start: FieldCell::get_editable_instance(),
                sample_bottom_end: FieldCell::get_editable_instance(),
                sample_top_start: FieldCell::get_editable_instance(),
                sample_top_end: FieldCell::get_editable_instance(),
                slices_in_sample: FieldCell::get_ineditable_instance(),
                current_chunk: HeaderCell::new_void(),
                reference_section: FieldCell::get_ineditable_instance(),
                current_section: FieldCell::get_ineditable_instance(),
                setup_final_start: FieldCell::get_editable_instance(),
                setup_final_end: FieldCell::get_editable_instance(),
                join_final_start: FieldCell::get_editable_instance(),
                join_final_end: FieldCell::get_editable_instance(),
                rotation_angle_x: FieldCell::get_editable_instance(),
                rotation_angle_y: FieldCell::get_editable_instance(),
                rotation_angle_z: FieldCell::get_editable_instance(),
                manager,
                table: Rc::downgrade(table),
                highlighter_button: HighlighterButton::get_instance(parent, Some(group)),
                data: RefCell::new(data),
                row_number: HeaderCell::new_int(
                    (30.0 * UIParameters::get_instance_void().get_font_size_adjustment()) as i32,
                ),
                imod_index: Cell::new(-1),
                imod_rot_index: Cell::new(-1),
                section_expanded: Cell::new(section_expanded),
                valid: Cell::new(true),
                inverted: Cell::new(false),
            }
        })
    }

    /// Java package-private `SectionTableRow(JoinManager, SectionTablePanel, int,
    /// File, boolean)`.  Create colors, fields, and buttons.  Add the row to the
    /// table.
    pub fn new_tomogram(
        manager: &'static JoinManager,
        table: &Rc<SectionTablePanel>,
        row_number: i32,
        tomogram: &Path,
        section_expanded: bool,
    ) -> Rc<SectionTableRow> {
        let mut data = SectionTableRowData::new(manager, row_number);
        data.set_setup_section(tomogram);
        let this = SectionTableRow::construct(manager, table, section_expanded, data);
        this.display_data_int(row_number);
        this.set_tool_tip_text();
        this
    }

    /// Java package-private `SectionTableRow(JoinManager, SectionTablePanel,
    /// SectionTableRowData, boolean)`.
    pub fn new_data(
        manager: &'static JoinManager,
        table: &Rc<SectionTablePanel>,
        data: &SectionTableRowData,
        section_expanded: bool,
    ) -> Rc<SectionTableRow> {
        let this = SectionTableRow::construct(
            manager,
            table,
            section_expanded,
            SectionTableRowData::new_from(manager, data),
        );
        this.section_expanded.set(section_expanded);
        this.display_data_int(data.get_row_number().get_int());
        this.set_tool_tip_text();
        this
    }

    /// Java field read `table`.
    fn table(&self) -> Rc<SectionTablePanel> {
        self.table.upgrade().expect("the section table owns its rows")
    }

    /// Java package-private `setNames()`.
    pub fn set_names(&self) {
        let table = self.table();
        let sample_header_cell = table.get_sample_header_cell();
        let rotation_header_cell = table.get_rotation_header_cell();
        let join_final_header_cell = table.get_join_final_header_cell();
        let label = Some(section_table_panel::LABEL);
        self.sample_bottom_start
            .set_headers(label, &self.row_number, &sample_header_cell);
        self.sample_bottom_end
            .set_headers(label, &self.row_number, &sample_header_cell);
        self.sample_top_start
            .set_headers(label, &self.row_number, &sample_header_cell);
        self.sample_top_end
            .set_headers(label, &self.row_number, &sample_header_cell);
        self.rotation_angle_x
            .set_headers(label, &self.row_number, &rotation_header_cell);
        self.rotation_angle_y
            .set_headers(label, &self.row_number, &rotation_header_cell);
        self.rotation_angle_z
            .set_headers(label, &self.row_number, &rotation_header_cell);
        self.join_final_start
            .set_headers(label, &self.row_number, &join_final_header_cell);
        self.join_final_end
            .set_headers(label, &self.row_number, &join_final_header_cell);
    }

    /// Java package-private `setInUse()`.
    pub fn set_in_use(&self) {
        let table = self.table();
        if table.is_setup_tab() {
            let row_number = self.data.borrow().get_row_number().get_int();
            let bottom_in_use = row_number > 1;
            let top_in_use = row_number < table.size();
            let final_inuse = table.is_join_tab();
            self.sample_bottom_start.set_in_use(bottom_in_use);
            self.sample_bottom_end.set_in_use(bottom_in_use);
            self.sample_top_start.set_in_use(top_in_use);
            self.sample_top_end.set_in_use(top_in_use);
            self.setup_final_start.set_in_use(final_inuse);
            self.setup_final_end.set_in_use(final_inuse);
        } else if table.is_join_tab() {
            self.join_final_start.set_in_use(true);
            self.join_final_end.set_in_use(true);
        }
    }

    /// Java package-private final `isRotated()`.
    pub fn is_rotated(&self) -> bool {
        // Fixed in translation: a row without a join section makes the source pass
        // null to `isRotatedTomogram` (NullPointerException); it is not rotated here.
        self.data
            .borrow()
            .get_join_section()
            .is_some_and(dataset_files::is_rotated_tomogram)
    }

    /// Java package-private `remove()`.
    pub fn remove(&self) {
        self.row_number.remove();
        self.highlighter_button.remove();
        self.setup_section.remove();
        self.sample_bottom_start.remove();
        self.sample_bottom_end.remove();
        self.sample_top_start.remove();
        self.sample_top_end.remove();
        self.setup_final_start.remove();
        self.setup_final_end.remove();
        self.join_final_start.remove();
        self.join_final_end.remove();
        self.rotation_angle_x.remove();
        self.rotation_angle_y.remove();
        self.rotation_angle_z.remove();
        // align
        self.slices_in_sample.remove();
        self.current_chunk.remove();
        self.reference_section.remove();
        self.current_section.remove();
        // join
        self.join_section.remove();
        self.join_final_start.remove();
        self.join_final_end.remove();
    }

    /// Java package-private final `removeImod()`.
    pub fn remove_imod(&self) {
        self.manager
            .imod_remove(Some(imod_manager::TOMOGRAM_KEY), self.imod_index.get());
        self.manager
            .imod_remove(Some(imod_manager::ROT_TOMOGRAM_KEY), self.imod_rot_index.get());
    }

    /// Java package-private `setJoinFinalStartHighlight(boolean)`.
    pub fn set_join_final_start_highlight(&self, highlight: bool) {
        self.set_cell_highlight(highlight, &self.join_final_start);
    }

    /// Java package-private `setJoinFinalEndHighlight(boolean)`.
    pub fn set_join_final_end_highlight(&self, highlight: bool) {
        self.set_cell_highlight(highlight, &self.join_final_end);
    }

    /// Java private `setCellHighlight(boolean, InputCell)`.
    fn set_cell_highlight(&self, highlight: bool, cell: &InputCell) {
        // avoid turning off highlighting in a highlighted row
        if !highlight && self.highlighter_button.is_highlighted() {
            return;
        }
        cell.set_highlight(highlight);
    }

    /// Java package-private `setMode(int)`.  The source's `IllegalStateException`
    /// for an unknown mode is a panic.
    pub fn set_mode(&self, mode: i32) {
        match mode {
            join_dialog::SAMPLE_PRODUCED_MODE => {
                self.sample_bottom_start.set_editable(false);
                self.sample_bottom_end.set_editable(false);
                self.sample_top_start.set_editable(false);
                self.sample_top_end.set_editable(false);
                self.rotation_angle_x.set_editable(false);
                self.rotation_angle_y.set_editable(false);
                self.rotation_angle_z.set_editable(false);
            }
            join_dialog::SETUP_MODE
            | join_dialog::SAMPLE_NOT_PRODUCED_MODE
            | join_dialog::CHANGING_SAMPLE_MODE => {
                self.sample_bottom_start.set_editable(true);
                self.sample_bottom_end.set_editable(true);
                self.sample_top_start.set_editable(true);
                self.sample_top_end.set_editable(true);
                self.rotation_angle_x.set_editable(true);
                self.rotation_angle_y.set_editable(true);
                self.rotation_angle_z.set_editable(true);
            }
            _ => panic!("java.lang.IllegalStateException: mode={mode}"),
        }
    }

    /// Java package-private `setInverted(ConstEtomoNumber)`.
    pub fn set_inverted(&self, inverted: Option<&ConstEtomoNumber>) {
        let Some(inverted) = inverted else {
            return;
        };
        self.inverted.set(inverted.is());
        let mut tooltip: Option<String> = None;
        if self.inverted.get() {
            tooltip = Some(format!("This section is inverted.  {}", INVERTED_WARNING));
        }
        let inverted = self.inverted.get();
        let tooltip = tooltip.as_deref();
        self.row_number.set_warning_boolean_string(inverted, tooltip);
        self.setup_section
            .set_warning_boolean_string(inverted, tooltip);
        self.join_section
            .set_warning_boolean_string(inverted, tooltip);
        self.sample_bottom_start
            .set_warning_boolean_string(inverted, tooltip);
        self.sample_bottom_end
            .set_warning_boolean_string(inverted, tooltip);
        self.sample_top_start
            .set_warning_boolean_string(inverted, tooltip);
        self.sample_top_end
            .set_warning_boolean_string(inverted, tooltip);
        self.slices_in_sample
            .set_warning_boolean_string(inverted, tooltip);
        self.setup_final_start
            .set_warning_boolean_string(inverted, tooltip);
        self.setup_final_end
            .set_warning_boolean_string(inverted, tooltip);
        self.join_final_start
            .set_warning_boolean_string(inverted, tooltip);
        self.join_final_end
            .set_warning_boolean_string(inverted, tooltip);
        self.rotation_angle_x
            .set_warning_boolean_string(inverted, tooltip);
        self.rotation_angle_y
            .set_warning_boolean_string(inverted, tooltip);
        self.rotation_angle_z
            .set_warning_boolean_string(inverted, tooltip);
    }

    /// Java private `totalInRange(ConstEtomoNumber, ConstEtomoNumber)`.
    fn total_in_range(start: &ConstEtomoNumber, end: &ConstEtomoNumber) -> i32 {
        if start.is_null() || end.is_null() {
            return 0;
        }
        end.get_int().wrapping_sub(start.get_int()).wrapping_add(1)
    }

    /// Java private `getPrevSampleEnd(SectionTableRow)`.
    fn get_prev_sample_end(prev_row: Option<&SectionTableRow>) -> i32 {
        // first row
        match prev_row {
            None => 0,
            Some(prev_row) => prev_row.slices_in_sample.get_end_value(),
        }
    }

    /// Java private `getBottomSampleSlices(SectionTableRow)`.
    fn get_bottom_sample_slices(&self, prev_row: Option<&SectionTableRow>) -> i32 {
        // first row
        if prev_row.is_none() {
            return 0;
        }
        let data = self.data.borrow();
        SectionTableRow::total_in_range(data.get_sample_bottom_start(), data.get_sample_bottom_end())
    }

    /// Java private `getTopSampleSlices(int, ConstEtomoNumber)`.
    fn get_top_sample_slices(&self, total_rows: i32, row_num: &ConstEtomoNumber) -> i32 {
        // last row
        if row_num.equals_int(total_rows) {
            return 0;
        }
        let data = self.data.borrow();
        SectionTableRow::total_in_range(data.get_sample_top_start(), data.get_sample_top_end())
    }

    /// Java private `getPrevTopSampleSlices(SectionTableRow)`.
    fn get_prev_top_sample_slices(prev_row: Option<&SectionTableRow>) -> i32 {
        // first row
        let Some(prev_row) = prev_row else {
            return 0;
        };
        let data = prev_row.data.borrow();
        SectionTableRow::total_in_range(data.get_sample_top_start(), data.get_sample_top_end())
    }

    /// Java package-private `setupCurTab(SectionTableRow, int)`.
    pub fn setup_cur_tab(&self, prev_row: Option<&SectionTableRow>, total_rows: i32) {
        // Set align display only fields
        if self.table().is_align_tab() {
            let row_num = self.data.borrow().get_row_number().clone();
            let prev_sample_end = SectionTableRow::get_prev_sample_end(prev_row);
            let bottom_sample_slices = self.get_bottom_sample_slices(prev_row);
            let top_sample_slices = self.get_top_sample_slices(total_rows, &row_num);
            let prev_top_sample_slices = SectionTableRow::get_prev_top_sample_slices(prev_row);

            self.slices_in_sample.set_range_value(
                prev_sample_end + 1,
                prev_sample_end + bottom_sample_slices + top_sample_slices,
            );
            if prev_row.is_none() {
                self.current_chunk.set_text_string(Some(""));
                self.current_section.set_value_void();
                self.reference_section.set_value_void();
            } else {
                self.current_chunk
                    .set_text_string(Some(&row_num.to_string()));
                self.current_section
                    .set_range_value(prev_sample_end + 1, prev_sample_end + bottom_sample_slices);
                self.reference_section.set_range_value(
                    prev_sample_end - prev_top_sample_slices + 1,
                    prev_sample_end,
                );
            }
        }
    }

    /// Java package-private `display(int, JPanel, Viewport)`.  Remove row from
    /// display.  Display the row if it is inside the viewer.
    pub fn display(&self, index: i32, panel: &Rc<JComponent>, viewport: &Viewport) {
        if !viewport.in_viewport(index) {
            return;
        }
        let table = self.table();
        if table.is_setup_tab() {
            self.add_setup(panel);
        } else if table.is_align_tab() {
            self.add_align(panel);
        } else if table.is_join_tab() {
            self.add_join(panel);
        } else if table.is_rejoin_tab() {
            self.add_rejoin(panel);
        }
    }

    /// `cell.add(panel, layout, constraints)`: the cell's `add` and the
    /// `layout.setConstraints` it makes.
    fn add_cell(&self, table: &SectionTablePanel, cell: &dyn CellVirtual, component: Rc<JComponent>, panel: &Rc<JComponent>) {
        cell.add(panel);
        table
            .get_table_layout()
            .set_constraints(&component, &table.get_table_constraints());
    }

    /// Java private `addSetup(JPanel)`.
    fn add_setup(&self, panel: &Rc<JComponent>) {
        let table = self.table();
        table.with_table_constraints(|constraints| {
            constraints.weightx = 0.0;
            constraints.weighty = 0.1;
            constraints.gridwidth = 1;
        });
        self.add_cell(&table, &*self.row_number, self.row_number.get_component(), panel);
        table.with_table_constraints(|constraints| {
            self.highlighter_button
                .add(panel, &table.get_table_layout(), constraints);
        });
        table.with_table_constraints(|constraints| {
            constraints.weightx = 0.2;
            constraints.gridwidth = 2;
        });
        self.add_cell(&table, &*self.setup_section, self.setup_section.get_component(), panel);
        table.with_table_constraints(|constraints| {
            constraints.weightx = 0.1;
            constraints.gridwidth = 1;
        });
        self.add_cell(&table, &*self.sample_bottom_start, self.sample_bottom_start.get_component(), panel);
        self.add_cell(&table, &*self.sample_bottom_end, self.sample_bottom_end.get_component(), panel);
        self.add_cell(&table, &*self.sample_top_start, self.sample_top_start.get_component(), panel);
        self.add_cell(&table, &*self.sample_top_end, self.sample_top_end.get_component(), panel);
        self.add_cell(&table, &*self.setup_final_start, self.setup_final_start.get_component(), panel);
        self.add_cell(&table, &*self.setup_final_end, self.setup_final_end.get_component(), panel);
        self.add_cell(&table, &*self.rotation_angle_x, self.rotation_angle_x.get_component(), panel);
        self.add_cell(&table, &*self.rotation_angle_y, self.rotation_angle_y.get_component(), panel);
        table.with_table_constraints(|constraints| constraints.gridwidth = GRID_BAG_REMAINDER);
        self.add_cell(&table, &*self.rotation_angle_z, self.rotation_angle_z.get_component(), panel);
    }

    /// Java private `addAlign(JPanel)`.
    fn add_align(&self, panel: &Rc<JComponent>) {
        let table = self.table();
        table.with_table_constraints(|constraints| {
            constraints.weightx = 0.0;
            constraints.weighty = 0.1;
            constraints.gridwidth = 1;
        });
        self.add_cell(&table, &*self.row_number, self.row_number.get_component(), panel);
        table.with_table_constraints(|constraints| {
            constraints.weightx = 0.2;
            constraints.gridwidth = 2;
        });
        self.add_cell(&table, &*self.setup_section, self.setup_section.get_component(), panel);
        table.with_table_constraints(|constraints| {
            constraints.weightx = 0.1;
            constraints.gridwidth = 1;
        });
        self.add_cell(&table, &*self.slices_in_sample, self.slices_in_sample.get_component(), panel);
        self.add_cell(&table, &*self.current_chunk, self.current_chunk.get_component(), panel);
        self.add_cell(&table, &*self.reference_section, self.reference_section.get_component(), panel);
        table.with_table_constraints(|constraints| constraints.gridwidth = GRID_BAG_REMAINDER);
        self.add_cell(&table, &*self.current_section, self.current_section.get_component(), panel);
    }

    /// Java private `addJoin(JPanel)`.
    fn add_join(&self, panel: &Rc<JComponent>) {
        let table = self.table();
        table.with_table_constraints(|constraints| {
            constraints.weightx = 0.0;
            constraints.weighty = 0.1;
            constraints.gridwidth = 1;
        });
        self.add_cell(&table, &*self.row_number, self.row_number.get_component(), panel);
        table.with_table_constraints(|constraints| {
            self.highlighter_button
                .add(panel, &table.get_table_layout(), constraints);
        });
        table.with_table_constraints(|constraints| {
            constraints.weightx = 0.2;
            constraints.gridwidth = 2;
        });
        self.add_cell(&table, &*self.join_section, self.join_section.get_component(), panel);
        table.with_table_constraints(|constraints| {
            constraints.weightx = 0.1;
            constraints.gridwidth = 1;
        });
        self.add_cell(&table, &*self.join_final_start, self.join_final_start.get_component(), panel);
        table.with_table_constraints(|constraints| constraints.gridwidth = GRID_BAG_REMAINDER);
        self.add_cell(&table, &*self.join_final_end, self.join_final_end.get_component(), panel);
        self.join_final_start.set_editable(true);
        self.join_final_end.set_editable(true);
    }

    /// Java private `addRejoin(JPanel)`.
    fn add_rejoin(&self, panel: &Rc<JComponent>) {
        self.add_join(panel);
        self.join_final_start.set_editable(false);
        self.join_final_end.set_editable(false);
    }

    /// Java private `displayData(int)`.
    fn display_data_int(&self, row_number: i32) {
        self.row_number
            .set_text_string(Some(&row_number.to_string()));
        self.display_data();
    }

    /// Java private `displayData()`.  Copy field from data to the screen.  Copy all
    /// fields stored in data that can be displayed on the screen.
    fn display_data(&self) {
        self.set_section_text();
        let data = self.data.borrow();
        self.sample_bottom_start
            .set_value_string(Some(&data.get_sample_bottom_start().to_string()));
        self.sample_bottom_end
            .set_value_string(Some(&data.get_sample_bottom_end().to_string()));
        self.sample_top_start
            .set_value_string(Some(&data.get_sample_top_start().to_string()));
        self.sample_top_end
            .set_value_string(Some(&data.get_sample_top_end().to_string()));
        self.setup_final_start
            .set_value_string(Some(&data.get_setup_final_start().to_string()));
        self.setup_final_end
            .set_value_string(Some(&data.get_setup_final_end().to_string()));
        self.join_final_start
            .set_value_string(Some(&data.get_join_final_start().to_string()));
        self.join_final_end
            .set_value_string(Some(&data.get_join_final_end().to_string()));
        self.rotation_angle_x
            .set_value_string(Some(&data.get_rotation_angle_x().to_string()));
        self.rotation_angle_y
            .set_value_string(Some(&data.get_rotation_angle_y().to_string()));
        self.rotation_angle_z
            .set_value_string(Some(&data.get_rotation_angle_z().to_string()));
        let inverted = data.get_inverted().clone();
        drop(data);
        self.set_inverted(Some(&inverted));
    }

    /// Java private `retrieveData(boolean)`.  Copy data from screen to data.  Copies
    /// all fields that can be modified on the screen and are stored in data.  Checks
    /// for errors.  Prints a error message based on the first error found and
    /// returns false.  Tries to retrieve all values, regardless of errors.
    fn retrieve_data(&self, _display_error_message: bool) -> bool {
        self.data.borrow_mut().set_inverted(self.inverted.get());
        self.valid.set(true);
        let error_info = format!(
            "\nInvalid number in section {}",
            self.row_number.get_text().unwrap_or_else(|| "null".to_string())
        );
        let error_title = "Invalid Number";
        let report = |error_message: Option<String>| {
            if let Some(error_message) = error_message
                && self.valid.get()
            {
                ui_harness::with(|harness| {
                    harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self.manager),
                        &format!("{}{}", error_message, error_info),
                        error_title,
                        Some(AxisID::Only),
                    )
                });
                self.valid.set(false);
            }
        };
        let error_message = self
            .data
            .borrow_mut()
            .set_sample_bottom_start(self.sample_bottom_start.get_value().as_deref())
            .validate(None);
        report(error_message);
        let error_message = self
            .data
            .borrow_mut()
            .set_sample_bottom_end(self.sample_bottom_end.get_value().as_deref())
            .validate(None);
        report(error_message);
        let error_message = self
            .data
            .borrow_mut()
            .set_sample_top_start(self.sample_top_start.get_value().as_deref())
            .validate(None);
        report(error_message);
        let error_message = self
            .data
            .borrow_mut()
            .set_sample_top_end(self.sample_top_end.get_value().as_deref())
            .validate(None);
        report(error_message);
        let error_message = self
            .data
            .borrow_mut()
            .set_setup_final_start(self.setup_final_start.get_value().as_deref())
            .validate(None);
        report(error_message);
        let error_message = self
            .data
            .borrow_mut()
            .set_setup_final_end(self.setup_final_end.get_value().as_deref())
            .validate(None);
        report(error_message);
        let error_message = self
            .data
            .borrow_mut()
            .set_join_final_start(self.join_final_start.get_value().as_deref())
            .validate(None);
        report(error_message);
        let error_message = self
            .data
            .borrow_mut()
            .set_join_final_end(self.join_final_end.get_value().as_deref())
            .validate(None);
        report(error_message);
        let error_message = self
            .data
            .borrow_mut()
            .set_rotation_angle_x(self.rotation_angle_x.get_value().as_deref())
            .validate(None);
        report(error_message);
        let error_message = self
            .data
            .borrow_mut()
            .set_rotation_angle_y(self.rotation_angle_y.get_value().as_deref())
            .validate(None);
        report(error_message);
        let error_message = self
            .data
            .borrow_mut()
            .set_rotation_angle_z(self.rotation_angle_z.get_value().as_deref())
            .validate(None);
        report(error_message);
        self.valid.get()
    }

    /// Java package-private `validateMakejoincom(String)`.
    pub fn validate_makejoincom(&self, max_row: &str) -> bool {
        self.retrieve_data(false);
        let row_number_text = self.row_number.get_text().unwrap_or_default();
        let (bottom_start, bottom_end, top_start, top_end) = {
            let data = self.data.borrow();
            (
                data.get_sample_bottom_start().clone(),
                data.get_sample_bottom_end().clone(),
                data.get_sample_top_start().clone(),
                data.get_sample_top_end().clone(),
            )
        };
        if !self.validate(&bottom_start, &bottom_end, true, row_number_text == "1") {
            return false;
        }
        self.validate(&top_start, &top_end, true, row_number_text == max_row)
    }

    /// Java package-private `validateFinishjoin()`.
    pub fn validate_finishjoin(&self) -> bool {
        self.retrieve_data(false);
        let (start, end) = {
            let data = self.data.borrow();
            (
                data.get_join_final_start().clone(),
                data.get_join_final_end().clone(),
            )
        };
        self.validate(&start, &end, false, true)
    }

    /// Java private `validate(ConstEtomoNumber, ConstEtomoNumber, boolean, boolean)`.
    fn validate(
        &self,
        start: &ConstEtomoNumber,
        end: &ConstEtomoNumber,
        validate_values: bool,
        optional: bool,
    ) -> bool {
        let row_number_text = self.row_number.get_text().unwrap_or_else(|| "null".to_string());
        let open_message = |message: String| {
            ui_harness::with(|harness| {
                harness.open_message_dialog_base_manager_string_string_axis_id(
                    Some(self.manager),
                    &message,
                    "Entry Error",
                    Some(AxisID::Only),
                )
            });
        };
        if start.is_null() && !end.is_null() {
            open_message(format!(
                "{} cannot be empty when {} has been entered.  Invalid numbers in section {}",
                start.get_description(),
                end.get_description(),
                row_number_text
            ));
            self.valid.set(false);
            return self.valid.get();
        }
        if !start.is_null() && end.is_null() {
            open_message(format!(
                "{} cannot be empty when {} has been entered.  Invalid numbers in section {}",
                end.get_description(),
                start.get_description(),
                row_number_text
            ));
            self.valid.set(false);
            return self.valid.get();
        }
        if validate_values {
            if start.is_int() && start.get_int() > end.get_int() {
                open_message(format!(
                    "{} must be less then or equal to {}.",
                    start.get_description(),
                    start.get_description()
                ));
                self.valid.set(false);
                return self.valid.get();
            }
            if start.get_int() > end.get_int() {
                open_message(format!(
                    "{} must be less then or equal to {}.  Invalid numbers in section {}",
                    start.get_description(),
                    start.get_description(),
                    row_number_text
                ));
                self.valid.set(false);
                return self.valid.get();
            }
        }
        if !optional {
            if start.is_null() {
                open_message(format!(
                    "{} is required in section {}",
                    start.get_description(),
                    row_number_text
                ));
                self.valid.set(false);
                return self.valid.get();
            }
            if end.is_null() {
                open_message(format!(
                    "{} is required in section {}",
                    end.get_description(),
                    row_number_text
                ));
                self.valid.set(false);
                return self.valid.get();
            }
        }
        self.valid.set(true);
        self.valid.get()
    }

    /// Java package-private `isValid()`.
    pub fn is_valid(&self) -> bool {
        self.valid.get()
    }

    /// Java package-private `expandSection(boolean)`.  Toggle the setup section
    /// between absolute path when expand is true, and name when expand is false.
    pub fn expand_section(&self, expand: bool) {
        self.section_expanded.set(expand);
        self.set_section_text();
    }

    /// Java private `setSectionText()`.
    fn set_section_text(&self) {
        let data = self.data.borrow();
        if let Some(section) = data.get_setup_section() {
            if self.section_expanded.get() {
                self.setup_section.set_value_string(Some(
                    &utilities::java_io_file_get_absolute_path(&section.to_string_lossy()),
                ));
            } else {
                self.setup_section.set_value_string(Some(
                    &utilities::java_io_file_get_name(&section.to_string_lossy()),
                ));
            }
        }
        let Some(section) = data.get_join_section() else {
            return;
        };
        if self.section_expanded.get() {
            self.join_section.set_value_string(Some(
                &utilities::java_io_file_get_absolute_path(&section.to_string_lossy()),
            ));
        } else {
            self.join_section.set_value_string(Some(&utilities::java_io_file_get_name(
                &section.to_string_lossy(),
            )));
        }
    }

    /// Java package-private `swapBottomTop()`.
    pub fn swap_bottom_top(&self) {
        let bottom = self.sample_bottom_start.get_value();
        let top = self.sample_top_start.get_value();
        self.sample_bottom_start.set_value_string(top.as_deref());
        self.sample_top_start.set_value_string(bottom.as_deref());
        let bottom = self.sample_bottom_end.get_value();
        let top = self.sample_top_end.get_value();
        self.sample_bottom_end.set_value_string(top.as_deref());
        self.sample_top_end.set_value_string(bottom.as_deref());
    }

    /// Java package-private `setRowNumber(int)`.
    pub fn set_row_number(&self, row_number: i32) {
        self.data.borrow_mut().set_row_number(row_number);
        self.row_number
            .set_text_string(Some(&row_number.to_string()));
    }

    /// Java package-private `setRotationAngles(SlicerAngles)`.
    pub fn set_rotation_angles(&self, slicer_angles: &SlicerAngles) {
        self.rotation_angle_x
            .set_value_string(Some(&slicer_angles.get_x().to_string()));
        self.rotation_angle_y
            .set_value_string(Some(&slicer_angles.get_y().to_string()));
        self.rotation_angle_z
            .set_value_string(Some(&slicer_angles.get_z().to_string()));
    }

    /// Java package-private `isHighlighted()`.
    pub fn is_highlighted(&self) -> bool {
        self.highlighter_button.is_highlighted()
    }

    /// Java package-private `selectHighlightButton()`.
    pub fn select_highlight_button(&self) {
        self.highlighter_button.set_selected(true);
    }

    /// Java package-private `getSetupSectionFile()`.
    pub fn get_setup_section_file(&self) -> Option<PathBuf> {
        self.data
            .borrow()
            .get_setup_section()
            .map(Path::to_path_buf)
    }

    /// Java package-private `getJoinSectionFile()`.
    pub fn get_join_section_file(&self) -> Option<PathBuf> {
        self.data.borrow().get_join_section().map(Path::to_path_buf)
    }

    /// Java package-private `getSetupSectionText()`.
    pub fn get_setup_section_text(&self) -> String {
        self.setup_section.get_value().unwrap_or_default()
    }

    /// Java package-private `getXMax()`.
    pub fn get_x_max(&self) -> i32 {
        if self.table().is_join_tab() {
            return self.data.borrow().get_join_x_max();
        }
        self.data.borrow().get_setup_x_max()
    }

    /// Java package-private `getYMax()`.
    pub fn get_y_max(&self) -> i32 {
        if self.table().is_join_tab() {
            return self.data.borrow().get_join_y_max();
        }
        self.data.borrow().get_setup_y_max()
    }

    /// Java package-private `getZMax()`.
    pub fn get_z_max(&self) -> i32 {
        if self.table().is_join_tab() {
            return self.data.borrow().get_join_z_max();
        }
        self.data.borrow().get_setup_z_max()
    }

    /// Java package-private `getData()`: a copy of the row's data after reading the
    /// screen.
    pub fn get_data(&self) -> SectionTableRowData {
        self.retrieve_data(true);
        let data = self.data.borrow();
        SectionTableRowData::new_from(self.manager, &*data)
    }

    /// Java package-private `getInvalidReason()`.
    pub fn get_invalid_reason(&self) -> Option<String> {
        self.data.borrow().get_invalid_reason()
    }

    /// Java package-private `equalsSetupSection(File)`.
    pub fn equals_setup_section(&self, section: &Path) -> bool {
        let data = self.data.borrow();
        // Fixed in translation: a row without a setup section dereferences null in the
        // source; it equals nothing here.
        data.get_setup_section().is_some_and(|setup_section| {
            utilities::java_io_file_get_absolute_path(&setup_section.to_string_lossy())
                == utilities::java_io_file_get_absolute_path(&section.to_string_lossy())
        })
    }

    /// Java package-private `equalsJoinSection(File)`.
    pub fn equals_join_section(&self, section: &Path) -> bool {
        let data = self.data.borrow();
        data.get_join_section().is_some_and(|join_section| {
            utilities::java_io_file_get_absolute_path(&join_section.to_string_lossy())
                == utilities::java_io_file_get_absolute_path(&section.to_string_lossy())
        })
    }

    /// Java `equals(SectionTableRow)`.
    pub fn equals_row(&self, that: &SectionTableRow) -> bool {
        self.retrieve_data(false);
        let data = self.data.borrow();
        let that_data = that.data.borrow();
        data.equals(&*that_data)
    }

    /// Java `equals(ConstSectionTableRowData)`.
    pub fn equals(&self, that_data: &dyn ConstSectionTableRowData) -> bool {
        self.retrieve_data(false);
        self.data.borrow().equals(that_data)
    }

    /// Java package-private `equalsSample(ConstSectionTableRowData)`.
    pub fn equals_sample(&self, that_data: &dyn ConstSectionTableRowData) -> bool {
        self.retrieve_data(false);
        self.data.borrow().equals_sample(that_data)
    }

    /// Java package-private final `imodOpenSetupSectionFile(int,
    /// Run3dmodMenuOptions)`.
    pub fn imod_open_setup_section_file(
        &self,
        binning: i32,
        menu_options: Option<Run3dmodMenuOptions>,
    ) {
        let setup_section = self.get_setup_section_file();
        self.imod_index.set(self.manager.imod_open_with_binning(
            Some(imod_manager::TOMOGRAM_KEY),
            self.imod_index.get(),
            setup_section.as_deref(),
            binning,
            menu_options,
        ));
    }

    /// Java package-private final `imodOpenJoinSectionFile(int,
    /// Run3dmodMenuOptions)`.
    pub fn imod_open_join_section_file(
        &self,
        binning: i32,
        menu_options: Option<Run3dmodMenuOptions>,
    ) {
        let join_section = self.get_join_section_file();
        if join_section
            .as_deref()
            .is_some_and(dataset_files::is_rotated_tomogram)
        {
            self.imod_rot_index.set(self.manager.imod_open_with_binning(
                Some(imod_manager::ROT_TOMOGRAM_KEY),
                self.imod_rot_index.get(),
                join_section.as_deref(),
                binning,
                menu_options,
            ));
        } else {
            self.imod_index.set(self.manager.imod_open_with_binning(
                Some(imod_manager::TOMOGRAM_KEY),
                self.imod_index.get(),
                join_section.as_deref(),
                binning,
                menu_options,
            ));
        }
    }

    /// Java package-private final `imodGetAngles()`.
    pub fn imod_get_angles(&self) -> bool {
        if self.imod_index.get() == -1 {
            ui_harness::with(|harness| {
                harness.open_message_dialog_base_manager_string_string_axis_id(
                    Some(self.manager),
                    "Open in 3dmod and use the Slicer to change the angles.",
                    "Open 3dmod",
                    Some(AxisID::Only),
                )
            });
            return false;
        }
        let slicer_angles = self
            .manager
            .imod_get_slicer_angles(Some(imod_manager::TOMOGRAM_KEY), self.imod_index.get());
        let Some(slicer_angles) = slicer_angles.filter(SlicerAngles::is_complete) else {
            return false;
        };
        self.set_rotation_angles(&slicer_angles);
        true
    }

    /// Java package-private final `synchronizeSetupToJoin()`.
    pub fn synchronize_setup_to_join(&self) {
        self.retrieve_data(true);
        self.data.borrow_mut().synchronize_setup_to_join();
        self.display_data();
    }

    /// Java package-private final `synchronizeJoinToSetup()`.
    pub fn synchronize_join_to_setup(&self) {
        self.retrieve_data(true);
        self.data.borrow_mut().synchronize_join_to_setup();
        self.display_data();
    }

    /// Java private `setToolTipText()`.
    fn set_tool_tip_text(&self) {
        self.highlighter_button
            .set_tool_tip_text(Some("Press to select the section."));
        self.current_chunk
            .set_tool_tip_text(Some("The number of the chunk in Midas."));
    }
}

impl Highlightable for SectionTableRow {
    /// Java `highlight(boolean)`.  Change the foreground and background for all the
    /// fields in the row based on whether the highlighter button is selected.
    fn highlight(&self, highlight: bool) {
        self.setup_section.set_highlight(highlight);
        self.join_section.set_highlight(highlight);
        self.sample_bottom_start.set_highlight(highlight);
        self.sample_bottom_end.set_highlight(highlight);
        self.sample_top_start.set_highlight(highlight);
        self.sample_top_end.set_highlight(highlight);
        self.slices_in_sample.set_highlight(highlight);
        self.setup_final_start.set_highlight(highlight);
        self.setup_final_end.set_highlight(highlight);
        self.join_final_start.set_highlight(highlight);
        self.join_final_end.set_highlight(highlight);
        self.rotation_angle_x.set_highlight(highlight);
        self.rotation_angle_y.set_highlight(highlight);
        self.rotation_angle_z.set_highlight(highlight);
    }
}

/// Java `toString()`: `"[" + setupSection + "]"`.
impl std::fmt::Display for SectionTableRow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}]", self.setup_section.to_string())
    }
}
