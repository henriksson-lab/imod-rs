//! `IMOD/Etomo/src/etomo/ui/swing/BoundaryTable.java`.
//!
//! The Join dialog's boundary table (Model and Rejoin tabs): one `BoundaryRow` per
//! boundary between consecutive sections.  An event dispatch thread object, created
//! as `Rc<Self>`; the dialog owns it.  Cells record their `GridBagConstraints` as
//! they are added (see `section_table_panel.rs`), which is how the Slint window draws
//! the rows.

use std::cell::{Cell, RefCell};
use std::rc::{Rc, Weak};

use super::boundary_row::BoundaryRow;
use super::cell::CellVirtual;
use super::etched_border::EtchedBorder;
use super::etomo_panel::EtomoPanel;
use super::header_cell::HeaderCell;
use super::join_dialog::{JoinDialog, Tab};
use super::viewable::Viewable;
use super::viewport::Viewport;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::jdk::{
    GRID_BAG_BOTH, GRID_BAG_CENTER, GRID_BAG_REMAINDER, GridBagConstraints, GridBagLayout,
    JComponent,
};
use crate::imod::etomo::join_manager::JoinManager;
use crate::imod::etomo::storage::log_file::LogFileError;
use crate::imod::etomo::r#type::const_join_meta_data::ConstJoinMetaData;
use crate::imod::etomo::r#type::join_meta_data::JoinMetaData;
use crate::imod::etomo::r#type::join_screen_state::JoinScreenState;

/// Java package-private static final `TABLE_LABEL`.
pub const TABLE_LABEL: &str = "Boundary Table";

/// Java package-private `final class BoundaryTable implements Viewable`.
pub struct BoundaryTable {
    // header
    // first row
    header1_boundaries: Rc<HeaderCell>,
    header1_sections: Rc<HeaderCell>,
    header1_best_gap: Rc<HeaderCell>,
    header1_error: Rc<HeaderCell>,
    header1_original: Rc<HeaderCell>,
    header1_adjusted: Rc<HeaderCell>,
    // second row
    header2_boundaries: Rc<HeaderCell>,
    header2_sections: Rc<HeaderCell>,
    header2_best_gap: Rc<HeaderCell>,
    header2_mean_error: Rc<HeaderCell>,
    header2_max_error: Rc<HeaderCell>,
    header2_original_end: Rc<HeaderCell>,
    header2_original_start: Rc<HeaderCell>,
    header2_adjusted_end: Rc<HeaderCell>,
    header2_adjusted_start: Rc<HeaderCell>,
    // third row
    header3_sections: Rc<HeaderCell>,
    header3_best_gap: Rc<HeaderCell>,
    header3_original_end: Rc<HeaderCell>,
    header3_original_start: Rc<HeaderCell>,
    header3_adjusted_end: Rc<HeaderCell>,
    header3_adjusted_start: Rc<HeaderCell>,

    /// Java private final `rowList`.
    row_list: RefCell<Vec<Rc<BoundaryRow>>>,
    /// Java private final `rootPanel = new JPanel()`.
    root_panel: Rc<JComponent>,
    /// Java private final `constraints = new GridBagConstraints()`.
    constraints: RefCell<GridBagConstraints>,
    /// Java private final `pnlTable = new JPanel()`.
    pnl_table: Rc<JComponent>,
    /// Java private final `layout = new GridBagLayout()`.
    layout: GridBagLayout,
    /// Java private final `viewport`.
    viewport: Rc<Viewport>,

    /// Java private final `manager`.
    manager: &'static JoinManager,
    /// Java private final `parent` (the dialog owns the table).
    parent: Weak<JoinDialog>,
    /// Java private final `screenState`.
    screen_state: &'static JoinScreenState,
    /// Java private final `metaData`.
    meta_data: &'static JoinMetaData,
    /// Java private final `focusableParents`.
    focusable_parents: Vec<Rc<JComponent>>,

    /// Java private `rowChange`, initially true.
    row_change: Cell<bool>,
    /// Java private `tab`, initially null.
    tab: Cell<Option<Tab>>,
    /// Rust-only: Java `this`.
    self_ref: Weak<BoundaryTable>,
}

impl BoundaryTable {
    /// Java package-private `BoundaryTable(JoinManager, JoinDialog)`.
    pub fn new(manager: &'static JoinManager, join_dialog: &Rc<JoinDialog>) -> Rc<BoundaryTable> {
        let this = Rc::new_cyclic(|self_ref: &Weak<BoundaryTable>| {
            let viewable: Weak<dyn Viewable> = self_ref.clone();
            let join_table_size = etomo_director::INSTANCE
                .with_user_configuration(|user_config| user_config.get_join_table_size().get_int());
            BoundaryTable {
                header1_boundaries: HeaderCell::new_string(Some("Boundaries")),
                header1_sections: HeaderCell::new_string(Some("Sections")),
                header1_best_gap: HeaderCell::new_string(Some("Best")),
                header1_error: HeaderCell::new_string(Some("Error")),
                header1_original: HeaderCell::new_string(Some("Original")),
                header1_adjusted: HeaderCell::new_string(Some("Adjusted")),
                header2_boundaries: HeaderCell::new_void(),
                header2_sections: HeaderCell::new_void(),
                header2_best_gap: HeaderCell::new_string(Some("Gap")),
                header2_mean_error: HeaderCell::new_string(Some("Mean")),
                header2_max_error: HeaderCell::new_string(Some("Max")),
                header2_original_end: HeaderCell::new_void(),
                header2_original_start: HeaderCell::new_void(),
                header2_adjusted_end: HeaderCell::new_void(),
                header2_adjusted_start: HeaderCell::new_void(),
                header3_sections: HeaderCell::new_void(),
                header3_best_gap: HeaderCell::new_void(),
                header3_original_end: HeaderCell::new_string(Some("End")),
                header3_original_start: HeaderCell::new_string(Some("Start")),
                header3_adjusted_end: HeaderCell::new_string(Some("End")),
                header3_adjusted_start: HeaderCell::new_string(Some("Start")),
                row_list: RefCell::new(Vec::new()),
                root_panel: JComponent::new_panel(),
                constraints: RefCell::new(GridBagConstraints::default()),
                pnl_table: JComponent::new_panel(),
                layout: GridBagLayout::new(),
                viewport: Viewport::new(viewable, join_table_size, Some("Boundary")),
                manager,
                parent: Rc::downgrade(join_dialog),
                screen_state: manager.get_screen_state(),
                meta_data: manager.get_join_meta_data(),
                focusable_parents: vec![
                    join_dialog.get_model_tab_j_component(),
                    join_dialog.get_rejoin_tab_j_component(),
                ],
                row_change: Cell::new(true),
                tab: Cell::new(None),
                self_ref: self_ref.clone(),
            }
        });
        // construct panels
        let pnl_border = EtomoPanel::new();
        // init
        this.viewport.init_paging();
        // root panel
        this.root_panel.set_focusable(true);
        // Swing layout: rootPanel BoxLayout Y_AXIS.
        this.root_panel.add(&pnl_border.get_component());
        // Swing layout: Box.createRigidArea(FixedDim.x0_y40), Box.createRigidArea(x0_y20).
        // border pane
        // Swing layout: pnlBorder BoxLayout X_AXIS.
        pnl_border.set_border(&EtchedBorder::new(Some(TABLE_LABEL)).get_border());
        pnl_border.get_component().add(&this.pnl_table);
        if let Some(paging_panel) = this.viewport.get_paging_panel() {
            pnl_border.get_component().add(&paging_panel);
        }
        // table panel
        // Swing painting: pnlTable LineBorder.createBlackLineBorder(); layout.
        this.constraints.borrow_mut().fill = GRID_BAG_BOTH;
        this.header1_best_gap.pad();
        this.header3_original_end.pad();
        this.header3_original_start.pad();
        this.set_tool_tip_text();
        this
    }

    /// Java field read `parent`.
    fn parent(&self) -> Rc<JoinDialog> {
        self.parent.upgrade().expect("the join dialog owns its boundary table")
    }

    /// Java `pnlTable` read by the rows (Java passes it to each row).
    pub fn get_table_panel(&self) -> Rc<JComponent> {
        self.pnl_table.clone()
    }

    /// Java `layout` read by the rows.
    pub fn get_layout(&self) -> &GridBagLayout {
        &self.layout
    }

    /// Java `constraints` read by the rows: a copy.
    pub fn get_constraints(&self) -> GridBagConstraints {
        *self.constraints.borrow()
    }

    /// Java `constraints` mutated by the rows.
    pub fn with_constraints<R>(&self, f: impl FnOnce(&mut GridBagConstraints) -> R) -> R {
        let mut constraints = *self.constraints.borrow();
        let result = f(&mut constraints);
        *self.constraints.borrow_mut() = constraints;
        result
    }

    /// Java package-private `setXfjointomoResult() throws LogFileException,
    /// IOException, LockException`.
    pub fn set_xfjointomo_result(&self) -> Result<(), LogFileError> {
        let rows = self.row_list.borrow().clone();
        for row in rows {
            row.set_xfjointomo_result(self.manager)?;
        }
        Ok(())
    }

    /// Java package-private `display()`.  Updates and displays the table as
    /// necessary.  Does nothing if the tab has not changed and rowChange is false.
    pub fn display(&self) {
        self.display_force(false);
    }

    /// Java package-private `display(boolean)`.  Updates and displays the table as
    /// necessary.  Does nothing if the tab has not changed and rowChange is false.
    /// Always updates if force is true.
    pub fn display_force(&self, force: bool) {
        let old_tab = self.tab.get();
        self.tab.set(Some(self.parent().get_tab()));
        if !force && old_tab == self.tab.get() && !self.row_change.get() {
            return;
        }
        let rows = self.row_list.borrow().clone();
        for row in rows {
            row.remove_display();
        }
        self.pnl_table.remove_all();
        self.add_header(self.tab.get());
        self.add_rows();
        if let Some(main_panel) = self.manager.get_main_panel() {
            main_panel.main_panel().repaint();
        }
    }

    /// Java package-private `msgRowChange()`.  Causes the rows in the table to be
    /// deleted and recreated when the table is displayed.
    pub fn msg_row_change(&self) {
        self.row_change.set(true);
        // when addRows() is called, it will load from screenState and metaData, so
        // they need to be empty if they are out of date.
        BoundaryRow::reset_screen_state(self.screen_state);
        BoundaryRow::reset_meta_data(self.meta_data);
    }

    /// Java package-private `getScreenState()`.
    pub fn get_screen_state(&self) {
        BoundaryRow::reset_screen_state(self.screen_state);
        let rows = self.row_list.borrow().clone();
        for row in rows {
            row.get_screen_state(self.screen_state);
        }
    }

    /// Java package-private `getAdjustedHeaderCell()`.
    pub fn get_adjusted_header_cell(&self) -> Rc<HeaderCell> {
        self.header1_adjusted.clone()
    }

    /// Java package-private `getMetaData()`.
    pub fn get_meta_data(&self) {
        BoundaryRow::reset_meta_data(self.meta_data);
        let rows = self.row_list.borrow().clone();
        for row in rows {
            row.get_meta_data(self.meta_data);
        }
    }

    /// Java package-private `getContainer()`.
    pub fn get_container(&self) -> Rc<JComponent> {
        self.root_panel.clone()
    }

    /// `header.add(pnlTable, layout, constraints)`: the cell's `add` and the
    /// `layout.setConstraints` it makes.
    fn add_header_cell(&self, cell: &Rc<HeaderCell>) {
        CellVirtual::add(&**cell, &self.pnl_table);
        self.layout
            .set_constraints(&cell.get_component(), &self.constraints.borrow());
    }

    /// Java private `addHeader(JoinDialog.Tab)`.
    fn add_header(&self, tab: Option<Tab>) {
        if tab == Some(Tab::Model) {
            self.add_model_header();
        } else if tab == Some(Tab::Rejoin) {
            self.add_rejoin_header();
        }
    }

    /// Java private `addModelHeader()`.
    fn add_model_header(&self) {
        // Header
        // First row
        {
            let mut constraints = self.constraints.borrow_mut();
            constraints.anchor = GRID_BAG_CENTER;
            constraints.weightx = 0.0;
            constraints.weighty = 0.0;
            constraints.gridheight = 1;
            constraints.gridwidth = 1;
        }
        self.add_header_cell(&self.header1_boundaries);
        self.constraints.borrow_mut().weightx = 0.1;
        self.add_header_cell(&self.header1_best_gap);
        self.constraints.borrow_mut().gridwidth = GRID_BAG_REMAINDER;
        self.add_header_cell(&self.header1_error);
        // second row
        {
            let mut constraints = self.constraints.borrow_mut();
            constraints.weightx = 0.0;
            constraints.gridwidth = 1;
        }
        self.add_header_cell(&self.header2_boundaries);
        self.constraints.borrow_mut().weightx = 0.1;
        self.add_header_cell(&self.header2_best_gap);
        self.add_header_cell(&self.header2_mean_error);
        self.constraints.borrow_mut().gridwidth = GRID_BAG_REMAINDER;
        self.add_header_cell(&self.header2_max_error);
    }

    /// Java private `addRejoinHeader()`.
    fn add_rejoin_header(&self) {
        // Header
        // First row
        {
            let mut constraints = self.constraints.borrow_mut();
            constraints.anchor = GRID_BAG_CENTER;
            constraints.weightx = 0.0;
            constraints.weighty = 0.0;
            constraints.gridheight = 1;
            constraints.gridwidth = 1;
        }
        self.add_header_cell(&self.header1_sections);
        {
            let mut constraints = self.constraints.borrow_mut();
            constraints.weightx = 0.1;
            constraints.gridwidth = 2;
        }
        self.add_header_cell(&self.header1_original);
        self.constraints.borrow_mut().gridwidth = 1;
        self.add_header_cell(&self.header1_best_gap);
        self.constraints.borrow_mut().gridwidth = GRID_BAG_REMAINDER;
        self.add_header_cell(&self.header1_adjusted);
        // second row
        {
            let mut constraints = self.constraints.borrow_mut();
            constraints.weightx = 0.0;
            constraints.gridwidth = 1;
        }
        self.add_header_cell(&self.header2_sections);
        self.constraints.borrow_mut().weightx = 0.1;
        self.add_header_cell(&self.header2_original_end);
        self.add_header_cell(&self.header2_original_start);
        self.add_header_cell(&self.header2_best_gap);
        self.add_header_cell(&self.header2_adjusted_end);
        self.constraints.borrow_mut().gridwidth = GRID_BAG_REMAINDER;
        self.add_header_cell(&self.header2_adjusted_start);
        // third row
        {
            let mut constraints = self.constraints.borrow_mut();
            constraints.weightx = 0.0;
            constraints.gridwidth = 1;
        }
        self.add_header_cell(&self.header3_sections);
        self.constraints.borrow_mut().weightx = 0.1;
        self.add_header_cell(&self.header3_original_end);
        self.add_header_cell(&self.header3_original_start);
        self.add_header_cell(&self.header3_best_gap);
        self.add_header_cell(&self.header3_adjusted_end);
        self.constraints.borrow_mut().gridwidth = GRID_BAG_REMAINDER;
        self.add_header_cell(&self.header3_adjusted_start);
    }

    /// Java private `setToolTipText()`.
    fn set_tool_tip_text(&self) {
        let text = "Boundaries between sections.";
        self.header1_boundaries.set_tool_tip_text(Some(text));
        self.header2_boundaries.set_tool_tip_text(Some(text));
        let text = "The pairs of sections which define each boundary.";
        self.header1_sections.set_tool_tip_text(Some(text));
        self.header2_sections.set_tool_tip_text(Some(text));
        self.header3_sections.set_tool_tip_text(Some(text));
        let text = "Describes how the final start and end values will change when the join is \
                    recreated, with a positive gap adding slices and a negative gap removing \
                    slices at the corresponding boundary.";
        self.header1_best_gap.set_tool_tip_text(Some(text));
        self.header2_best_gap.set_tool_tip_text(Some(text));
        self.header3_best_gap.set_tool_tip_text(Some(text));
        self.header1_error.set_tool_tip_text(Some(
            "Deviations between transformed points extrapolated from above and below the \
             corresponding boundary.",
        ));
        self.header2_mean_error
            .set_tool_tip_text(Some("Mean deviations."));
        self.header2_max_error
            .set_tool_tip_text(Some("Maximum deviations."));
        let text = "End and start values used to create the original join.";
        self.header1_original.set_tool_tip_text(Some(text));
        self.header2_original_end.set_tool_tip_text(Some(text));
        self.header2_original_start.set_tool_tip_text(Some(text));
        self.header3_original_end
            .set_tool_tip_text(Some("End values used to create the original join."));
        self.header3_original_start
            .set_tool_tip_text(Some("Start values used to create the original join."));
        let text = "End and start values which will be used to create the new join.";
        self.header1_adjusted.set_tool_tip_text(Some(text));
        self.header2_adjusted_end.set_tool_tip_text(Some(text));
        self.header2_adjusted_start.set_tool_tip_text(Some(text));
        self.header3_adjusted_end
            .set_tool_tip_text(Some("End values which will be used to create the new join."));
        self.header3_adjusted_start.set_tool_tip_text(Some(
            "Start values which will be used to create the new join.",
        ));
    }

    /// Java private `addRows()`.  Displays the rows.  Updates the rows when rowChange
    /// is true.  The number of rows to add is the section table size minus 1.
    fn add_rows(&self) {
        if self.row_change.get() {
            self.row_change.set(false);
            // RowList.clear(Viewport)
            self.row_list.borrow_mut().clear();
            self.parent().get_section_table().get_meta_data(self.meta_data);
            // RowList.add(int, ConstJoinMetaData, JoinScreenState, JPanel,
            // GridBagLayout, GridBagConstraints, Viewport): adds new BoundaryRow
            // instances; the number parameter in the BoundaryRow constructor starts
            // at 1.
            let size = self.parent().get_section_table_size() - 1;
            let this = self.self_ref.upgrade().expect("BoundaryTable");
            for i in 0..size {
                let row = BoundaryRow::new(
                    i + 1,
                    self.meta_data as &dyn ConstJoinMetaData,
                    self.screen_state,
                    &this,
                );
                self.row_list.borrow_mut().push(Rc::clone(&row));
                row.set_names();
            }
        }
        // RowList.display(JoinDialog.Tab, Viewport): BoundaryRow.display() on rows
        // that are in the viewer.
        let rows = self.row_list.borrow().clone();
        for (i, row) in rows.iter().enumerate() {
            row.display(i as i32, &self.viewport, self.tab.get());
        }
    }
}

impl Viewable for BoundaryTable {
    /// Java `getFocusableParents()`.
    fn get_focusable_parents(&self) -> Vec<Rc<JComponent>> {
        self.focusable_parents.clone()
    }

    /// Java `msgViewportPaged()`.
    fn msg_viewport_paged(&self) {
        self.display_force(true);
    }

    /// Java `size()`.
    fn size(&self) -> i32 {
        self.row_list.borrow().len() as i32
    }
}
