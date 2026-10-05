//! `IMOD/Etomo/src/etomo/ui/swing/SectionTablePanel.java`.
//!
//! The Join dialog's section table: the header rows and one `SectionTableRow` per
//! section, drawn differently on the Setup, Align, Join and Rejoin tabs, with the
//! Move Up/Down, Add/Delete Section, Open in 3dmod, Get Angles and Invert Table
//! buttons.  An event dispatch thread object, created as `Rc<Self>` by
//! [`SectionTablePanel::new`]; the dialog owns it (the table keeps a weak reference
//! to the dialog).
//!
//! **Superclass.**  `extends HighlightableTable`: the base state is `base`, the
//! abstract methods are the [`HighlightableTableVirtual`] implementation.
//!
//! **Table layout.**  The cells are added to `pnlTable` in order, and each `add`
//! records the `GridBagConstraints` it was laid out with
//! (`jdk::GridBagLayout::set_constraints`); a cell with `gridwidth == REMAINDER` ends
//! its row.  That is how the Slint window draws the rows (see `slint_bridge.rs`).

use std::cell::{Cell, RefCell};
use std::path::{Path, PathBuf};
use std::rc::{Rc, Weak};

use super::binned_xy_3dmod_button::BinnedXY3dmodButton;
use super::cell::CellVirtual;
use super::context_menu::ContextMenu;
use super::deferred_3dmod_button::Deferred3dmodButton;
use super::etched_border::EtchedBorder;
use super::expand_button::{self, ExpandButton};
use super::expandable::Expandable;
use super::file_chooser::{self, FileChooser};
use super::global_expand_button::GlobalExpandButton;
use super::header_cell::HeaderCell;
use super::highlightable::Highlightable;
use super::highlightable_table::{HighlightableTable, HighlightableTableVirtual};
use super::join_dialog::{self, JoinDialog, Tab};
use super::multi_line_button::MultiLineButton;
use super::run_3dmod_button_container::Run3dmodButtonContainer;
use super::section_table_row::{self, SectionTableRow};
use super::spaced_panel::{self, SpacedPanel};
use super::ui_harness;
use super::ui_parameters::UIParameters;
use super::viewable::Viewable;
use super::viewport::Viewport;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::jdk::{
    ActionEvent, ActionListener, FileFilter, GRID_BAG_BOTH, GRID_BAG_CENTER, GRID_BAG_REMAINDER,
    GridBagConstraints, GridBagLayout, JComponent, MouseEvent,
};
use crate::imod::etomo::join_manager::JoinManager;
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::storage::join_info_file::JoinInfoFile;
use crate::imod::etomo::storage::log_file::LogFileError;
use crate::imod::etomo::storage::tomogram_file_filter::TomogramFileFilter;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::ConstEtomoNumber;
use crate::imod::etomo::r#type::const_join_meta_data::ConstJoinMetaData;
use crate::imod::etomo::r#type::const_section_table_row_data::ConstSectionTableRowData;
use crate::imod::etomo::r#type::join_meta_data::JoinMetaData;
use crate::imod::etomo::r#type::join_state::JoinState;
use crate::imod::etomo::r#type::section_table_row_data::SectionTableRowData;
use crate::imod::etomo::util::mrc_header::MRCHeader;
use crate::imod::etomo::util::utilities;

/// Java private static final `flipWarning`.
const FLIP_WARNING: [&str; 2] = [
    "Tomograms have to be rotated after generation",
    "in order to be in the right orientation for joining serial sections.",
];
/// Java private static final `HEADER1_SECTIONS_LABEL`.
const HEADER1_SECTIONS_LABEL: &str = "Sections";
/// Java package-private static final `LABEL`.
pub const LABEL: &str = "Section Table";
/// Java private static final `UNIQUE_KEY`.
const UNIQUE_KEY: &str = "Section";

/// Java package-private `final class SectionTablePanel extends HighlightableTable
/// implements ContextMenu, Expandable, Run3dmodButtonContainer, Viewable`.
pub struct SectionTablePanel {
    /// The Java superclass part.
    base: HighlightableTable,
    /// Java private final `rootPanel = new JPanel()`.
    root_panel: Rc<JComponent>,
    /// Java private final `pnlBorder = SpacedPanel.getInstance()`.
    pnl_border: Rc<SpacedPanel>,
    /// Java private final `pnlTable = new JPanel()`.
    pnl_table: Rc<JComponent>,
    /// Java private final `pnlButtons = SpacedPanel.getInstance()`.
    pnl_buttons: Rc<SpacedPanel>,
    /// Java private final `pnlButtonsComponent1`.
    pnl_buttons_component1: Rc<SpacedPanel>,
    /// Java private final `pnlButtonsComponent2`.
    pnl_buttons_component2: Rc<SpacedPanel>,
    /// Java private final `pnlButtonsComponent4`.
    pnl_buttons_component4: Rc<SpacedPanel>,
    /// Java private final `btnMoveSectionUp`.
    btn_move_section_up: Rc<MultiLineButton>,
    /// Java private final `btnMoveSectionDown`.
    btn_move_section_down: Rc<MultiLineButton>,
    /// Java private final `btnAddSection`.
    btn_add_section: Rc<MultiLineButton>,
    /// Java private final `btnDeleteSection`.
    btn_delete_section: Rc<MultiLineButton>,
    /// Java private final `btnGetAngles`.
    btn_get_angles: Rc<MultiLineButton>,
    /// Java private final `btnInvertTable`.
    btn_invert_table: Rc<MultiLineButton>,
    // first header row
    header1_z_order: Rc<HeaderCell>,
    header1_setup_sections: Rc<HeaderCell>,
    /// Java private `button1ExpandSections`, initially null.
    button1_expand_sections: RefCell<Option<Rc<ExpandButton>>>,
    header1_join_sections: Rc<HeaderCell>,
    header1_sample: Rc<HeaderCell>,
    header1_slices_in_sample: Rc<HeaderCell>,
    header1_current_chunk: Rc<HeaderCell>,
    header1_reference_section: Rc<HeaderCell>,
    header1_current_section: Rc<HeaderCell>,
    header1_setup_final: Rc<HeaderCell>,
    header1_join_final: Rc<HeaderCell>,
    header1_rotation: Rc<HeaderCell>,
    // second header row
    header2_z_order: Rc<HeaderCell>,
    header2_setup_sections: Rc<HeaderCell>,
    header2_join_sections: Rc<HeaderCell>,
    header2_sample_bottom: Rc<HeaderCell>,
    header2_sample_top: Rc<HeaderCell>,
    header2_slices_in_sample: Rc<HeaderCell>,
    header2_current_chunk: Rc<HeaderCell>,
    header2_reference_section: Rc<HeaderCell>,
    header2_current_section: Rc<HeaderCell>,
    header2_setup_final: Rc<HeaderCell>,
    header2_join_final: Rc<HeaderCell>,
    header2_rotation: Rc<HeaderCell>,
    // third header row
    header3_z_order: Rc<HeaderCell>,
    header3_setup_sections: Rc<HeaderCell>,
    header3_join_sections: Rc<HeaderCell>,
    header3_sample_bottom_start: Rc<HeaderCell>,
    header3_sample_bottom_end: Rc<HeaderCell>,
    header3_sample_top_start: Rc<HeaderCell>,
    header3_sample_top_end: Rc<HeaderCell>,
    header3_setup_final_start: Rc<HeaderCell>,
    header3_setup_final_end: Rc<HeaderCell>,
    header3_join_final_start: Rc<HeaderCell>,
    header3_join_final_end: Rc<HeaderCell>,
    header3_rotation_x: Rc<HeaderCell>,
    header3_rotation_y: Rc<HeaderCell>,
    header3_rotation_z: Rc<HeaderCell>,
    /// Java private final `rowList`.
    row_list: RowList,
    /// Java private final `layout = new GridBagLayout()`.
    layout: GridBagLayout,
    /// Java private final `constraints = new GridBagConstraints()` (mutated by the
    /// rows as they lay themselves out).
    constraints: RefCell<GridBagConstraints>,
    /// Java private final `sectionTableActionListener`.
    section_table_action_listener: ActionListener,
    /// Java private final `pnlViewport = new JPanel()`.
    pnl_viewport: Rc<JComponent>,
    /// Java private final `viewport`.
    viewport: Rc<Viewport>,
    /// Java private final `focusableParents`.
    focusable_parents: RefCell<Vec<Option<Rc<JComponent>>>>,
    /// Java private final `manager`.
    manager: &'static JoinManager,
    /// Java private final `joinDialog` (the dialog owns the table).
    join_dialog: Weak<JoinDialog>,
    /// Java private `b3bOpen3dmod`.
    b3b_open_3dmod: RefCell<Option<Rc<BinnedXY3dmodButton>>>,
    /// Java private `mode`, initially `JoinDialog.SETUP_MODE`.
    mode: Cell<i32>,
    /// Java private `rotating`, initially false.
    rotating: Cell<bool>,
    /// Java private final `state`.
    state: &'static JoinState,
    /// Java private `lastLocation`, initially null.
    last_location: RefCell<Option<PathBuf>>,
    /// Rust-only: Java `this`.
    self_ref: Weak<SectionTablePanel>,
}

impl SectionTablePanel {
    /// Java package-private `SectionTablePanel(JoinDialog, JoinManager, JoinState)`.
    /// Creates the panel and table.
    pub fn new(
        join_dialog: &Rc<JoinDialog>,
        manager: &'static JoinManager,
        state: &'static JoinState,
    ) -> Rc<SectionTablePanel> {
        let this = Rc::new_cyclic(|self_ref: &Weak<SectionTablePanel>| {
            let font_numeric_width = UIParameters::get_instance_void().get_numeric_width();
            let sections_width = UIParameters::get_instance_void().get_sections_width();
            let adaptee = self_ref.clone();
            // Java `new SectionTableActionListener(this)`.
            let section_table_action_listener: ActionListener =
                Rc::new(move |event: &ActionEvent| {
                    if let Some(adaptee) = adaptee.upgrade() {
                        adaptee.action(event.get_action_command().unwrap_or(""), None, None);
                    }
                });
            let viewable: Weak<dyn Viewable> = self_ref.clone();
            let join_table_size = etomo_director::INSTANCE
                .with_user_configuration(|user_config| user_config.get_join_table_size().get_int());
            SectionTablePanel {
                // super(UNIQUE_KEY)
                base: HighlightableTable::new(UNIQUE_KEY),
                root_panel: JComponent::new_panel(),
                pnl_border: SpacedPanel::get_instance_void(),
                pnl_table: JComponent::new_panel(),
                pnl_buttons: SpacedPanel::get_instance_void(),
                pnl_buttons_component1: SpacedPanel::get_instance_void(),
                pnl_buttons_component2: SpacedPanel::get_instance_void(),
                pnl_buttons_component4: SpacedPanel::get_instance_void(),
                btn_move_section_up: MultiLineButton::new_boolean_string(
                    true,
                    Some("Move Section Up"),
                ),
                btn_move_section_down: MultiLineButton::new_boolean_string(
                    true,
                    Some("Move Section Down"),
                ),
                btn_add_section: MultiLineButton::new_boolean_string(true, Some("Add Section")),
                btn_delete_section: MultiLineButton::new_boolean_string(
                    true,
                    Some("Delete Section"),
                ),
                btn_get_angles: MultiLineButton::new_boolean_string(
                    true,
                    Some("Get Angles from Slicer"),
                ),
                btn_invert_table: MultiLineButton::new_boolean_string(true, Some("Invert Table")),
                header1_z_order: HeaderCell::new_string(Some("Z Order")),
                header1_setup_sections: HeaderCell::new_string_int(
                    Some(HEADER1_SECTIONS_LABEL),
                    sections_width,
                ),
                button1_expand_sections: RefCell::new(None),
                header1_join_sections: HeaderCell::new_string_int(
                    Some(HEADER1_SECTIONS_LABEL),
                    sections_width,
                ),
                header1_sample: HeaderCell::new_string(Some("Sample Slices")),
                header1_slices_in_sample: HeaderCell::new_string(Some("Slices in")),
                header1_current_chunk: HeaderCell::new_string(Some("Current")),
                header1_reference_section: HeaderCell::new_string(Some("Reference")),
                header1_current_section: HeaderCell::new_string(Some("Current")),
                header1_setup_final: HeaderCell::new_string(Some("Final")),
                header1_join_final: HeaderCell::new_string(Some("Final")),
                header1_rotation: HeaderCell::new_string(Some("Rotation Angles")),
                header2_z_order: HeaderCell::new_void(),
                header2_setup_sections: HeaderCell::new_void(),
                header2_join_sections: HeaderCell::new_string(Some("In Final")),
                header2_sample_bottom: HeaderCell::new_string(Some("Bottom")),
                header2_sample_top: HeaderCell::new_string(Some("Top")),
                header2_slices_in_sample: HeaderCell::new_string(Some("Sample")),
                header2_current_chunk: HeaderCell::new_string(Some("Chunk")),
                header2_reference_section: HeaderCell::new_string(Some("Section")),
                header2_current_section: HeaderCell::new_string(Some("Section")),
                header2_setup_final: HeaderCell::new_void(),
                header2_join_final: HeaderCell::new_void(),
                header2_rotation: HeaderCell::new_void(),
                header3_z_order: HeaderCell::new_void(),
                header3_setup_sections: HeaderCell::new_void(),
                header3_join_sections: HeaderCell::new_void(),
                header3_sample_bottom_start: HeaderCell::new_string_int(
                    Some("Start"),
                    font_numeric_width,
                ),
                header3_sample_bottom_end: HeaderCell::new_string_int(
                    Some("End"),
                    font_numeric_width,
                ),
                header3_sample_top_start: HeaderCell::new_string_int(
                    Some("Start"),
                    font_numeric_width,
                ),
                header3_sample_top_end: HeaderCell::new_string_int(Some("End"), font_numeric_width),
                header3_setup_final_start: HeaderCell::new_string_int(
                    Some("Start"),
                    font_numeric_width,
                ),
                header3_setup_final_end: HeaderCell::new_string_int(
                    Some("End"),
                    font_numeric_width,
                ),
                header3_join_final_start: HeaderCell::new_string_int(
                    Some("Start"),
                    font_numeric_width,
                ),
                header3_join_final_end: HeaderCell::new_string_int(Some("End"), font_numeric_width),
                header3_rotation_x: HeaderCell::new_string_int(Some("X"), font_numeric_width),
                header3_rotation_y: HeaderCell::new_string_int(Some("Y"), font_numeric_width),
                header3_rotation_z: HeaderCell::new_string_int(Some("Z"), font_numeric_width),
                row_list: RowList::new(manager),
                layout: GridBagLayout::new(),
                constraints: RefCell::new(GridBagConstraints::default()),
                section_table_action_listener,
                pnl_viewport: JComponent::new_panel(),
                viewport: Viewport::new(viewable, join_table_size, Some(UNIQUE_KEY)),
                focusable_parents: RefCell::new(Vec::new()),
                manager,
                join_dialog: Rc::downgrade(join_dialog),
                b3b_open_3dmod: RefCell::new(None),
                mode: Cell::new(join_dialog::SETUP_MODE),
                rotating: Cell::new(false),
                state,
                last_location: RefCell::new(None),
                self_ref: self_ref.clone(),
            }
        });
        // Constructor body.
        *this.focusable_parents.borrow_mut() = vec![
            Some(join_dialog.get_setup_tab_j_component()),
            Some(join_dialog.get_align_tab_j_component()),
            Some(join_dialog.get_join_tab_j_component()),
        ];
        // init
        this.viewport.init_paging();
        let table: Weak<dyn HighlightableTableVirtual> = Rc::downgrade(&this) as _;
        this.base.init_highlight_hotkeys(table);
        // create root panel
        this.pnl_border.set_box_layout(spaced_panel::Y_AXIS);
        this.pnl_border
            .set_border(&EtchedBorder::new(Some(LABEL)).get_border());
        this.root_panel.add(&this.pnl_border.get_container());
        this.pnl_border.add_j_panel(&this.pnl_table);
        // Swing layout: pnlViewport BoxLayout X_AXIS.
        // table
        // Swing painting: pnlTable LineBorder.createBlackLineBorder(); layout.
        this.constraints.borrow_mut().fill = GRID_BAG_BOTH;
        let expandable: Weak<dyn Expandable> = Rc::downgrade(&this) as _;
        let button1_expand_sections = ExpandButton::get_instance_expandable_type(
            Some(expandable),
            Some(&expand_button::Type::MORE),
        );
        button1_expand_sections.set_name(Some(HEADER1_SECTIONS_LABEL));
        *this.button1_expand_sections.borrow_mut() = Some(button1_expand_sections);
        this.add_table_panel_components();
        // buttons
        this.create_buttons_panel();
        this.add_buttons_panel_components();
        this.add_root_panel_components();
        this.set_tool_tip_text();
        this
    }

    /// Java field read `joinDialog`.
    fn join_dialog(&self) -> Rc<JoinDialog> {
        self.join_dialog
            .upgrade()
            .expect("the join dialog owns its section table")
    }

    /// Java field read `button1ExpandSections`.
    fn button1_expand_sections(&self) -> Rc<ExpandButton> {
        self.button1_expand_sections
            .borrow()
            .clone()
            .expect("button1ExpandSections is set by the constructor")
    }

    /// Java field read `b3bOpen3dmod`.
    fn b3b_open_3dmod(&self) -> Rc<BinnedXY3dmodButton> {
        self.b3b_open_3dmod
            .borrow()
            .clone()
            .expect("b3bOpen3dmod is set by createButtonsPanel")
    }

    /// Java package-private `isSetupTab()`.
    pub fn is_setup_tab(&self) -> bool {
        self.join_dialog().is_setup_tab()
    }

    /// Java package-private `isAlignTab()`.
    pub fn is_align_tab(&self) -> bool {
        self.join_dialog().is_align_tab()
    }

    /// Java package-private `isJoinTab()`.
    pub fn is_join_tab(&self) -> bool {
        self.join_dialog().is_join_tab()
    }

    /// Java package-private `isRejoinTab()`.
    pub fn is_rejoin_tab(&self) -> bool {
        self.join_dialog().is_rejoin_tab()
    }

    /// Java private `addRootPanelComponents()`.
    fn add_root_panel_components(&self) {
        // Swing layout: GridLayout(2, 1) on the Join tab, else BoxLayout Y_AXIS.
        let _ = self.is_join_tab();
        self.root_panel.add(&self.pnl_border.get_container());
        self.pnl_border.add_j_panel(&self.pnl_viewport);
        self.pnl_viewport.add(&self.pnl_table);
        if let Some(paging_panel) = self.viewport.get_paging_panel() {
            self.pnl_viewport.add(&paging_panel);
        }
        if !self.is_align_tab() {
            self.add_buttons_panel_components();
            self.pnl_border
                .add_container(&self.pnl_buttons.get_container());
        }
    }

    /// Java private `addTablePanelComponents()`.
    fn add_table_panel_components(&self) {
        // Table constraints
        if self.is_setup_tab() {
            self.add_setup_table_panel_components();
        } else if self.is_align_tab() {
            self.add_align_table_panel_components();
        } else if self.is_join_tab() || self.is_rejoin_tab() {
            self.add_join_table_panel_components();
        }
    }

    /// `cell.add(pnlTable, layout, constraints)`: the cell's `add` and the
    /// `layout.setConstraints` it makes.
    fn add_header(&self, cell: &dyn CellVirtual, component: Rc<JComponent>) {
        cell.add(&self.pnl_table);
        self.layout
            .set_constraints(&component, &self.constraints.borrow());
    }

    /// `header.add(pnlTable, layout, constraints)` for a `HeaderCell`.
    fn add_header_cell(&self, cell: &Rc<HeaderCell>) {
        self.add_header(&**cell, cell.get_component());
    }

    /// `button1ExpandSections.add(pnlTable, layout, constraints)`.
    fn add_expand_button(&self) {
        let button = self.button1_expand_sections();
        button.add(&self.pnl_table);
        self.layout
            .set_constraints(&button.get_component(), &self.constraints.borrow());
    }

    /// Java private `addSetupTablePanelComponents()`.  Creates the panel and table.
    /// Adds the header rows.  Adds SectionTableRows to rows to create each row.
    fn add_setup_table_panel_components(&self) {
        // Header
        // First row
        {
            let mut constraints = self.constraints.borrow_mut();
            constraints.anchor = GRID_BAG_CENTER;
            constraints.weightx = 0.0;
            constraints.weighty = 0.2;
            constraints.gridheight = 1;
            constraints.gridwidth = 2;
        }
        self.add_header_cell(&self.header1_z_order);
        {
            let mut constraints = self.constraints.borrow_mut();
            constraints.gridwidth = 1;
            constraints.weightx = 0.2;
        }
        self.add_header_cell(&self.header1_setup_sections);
        self.constraints.borrow_mut().weightx = 0.0;
        self.add_expand_button();
        {
            let mut constraints = self.constraints.borrow_mut();
            constraints.weightx = 0.1;
            constraints.gridwidth = 4;
        }
        self.add_header_cell(&self.header1_sample);
        self.constraints.borrow_mut().gridwidth = 2;
        self.add_header_cell(&self.header1_setup_final);
        self.constraints.borrow_mut().gridwidth = GRID_BAG_REMAINDER;
        self.add_header_cell(&self.header1_rotation);
        // second row
        {
            let mut constraints = self.constraints.borrow_mut();
            constraints.weightx = 0.0;
            constraints.gridwidth = 2;
        }
        self.add_header_cell(&self.header2_z_order);
        self.constraints.borrow_mut().weightx = 0.2;
        self.add_header_cell(&self.header2_setup_sections);
        self.constraints.borrow_mut().weightx = 0.1;
        self.add_header_cell(&self.header2_sample_bottom);
        self.add_header_cell(&self.header2_sample_top);
        self.add_header_cell(&self.header2_setup_final);
        self.constraints.borrow_mut().gridwidth = GRID_BAG_REMAINDER;
        self.add_header_cell(&self.header2_rotation);
        // Third row
        {
            let mut constraints = self.constraints.borrow_mut();
            constraints.weightx = 0.0;
            constraints.gridwidth = 2;
        }
        self.add_header_cell(&self.header3_z_order);
        self.constraints.borrow_mut().weightx = 0.2;
        self.add_header_cell(&self.header3_setup_sections);
        {
            let mut constraints = self.constraints.borrow_mut();
            constraints.weightx = 0.1;
            constraints.gridwidth = 1;
        }
        self.add_header_cell(&self.header3_sample_bottom_start);
        self.add_header_cell(&self.header3_sample_bottom_end);
        self.add_header_cell(&self.header3_sample_top_start);
        self.add_header_cell(&self.header3_sample_top_end);
        self.add_header_cell(&self.header3_setup_final_start);
        self.add_header_cell(&self.header3_setup_final_end);
        self.add_header_cell(&self.header3_rotation_x);
        self.add_header_cell(&self.header3_rotation_y);
        self.constraints.borrow_mut().gridwidth = GRID_BAG_REMAINDER;
        self.add_header_cell(&self.header3_rotation_z);
    }

    /// Java package-private `getMode()`.
    pub fn get_mode(&self) -> i32 {
        self.mode.get()
    }

    /// Java private `addAlignTablePanelComponents()`.
    fn add_align_table_panel_components(&self) {
        // Header
        // First row
        {
            let mut constraints = self.constraints.borrow_mut();
            constraints.anchor = GRID_BAG_CENTER;
            constraints.weightx = 0.0;
            constraints.weighty = 0.2;
            constraints.gridheight = 1;
            constraints.gridwidth = 1;
        }
        self.add_header_cell(&self.header1_z_order);
        self.constraints.borrow_mut().weightx = 0.2;
        self.add_header_cell(&self.header1_setup_sections);
        {
            let mut constraints = self.constraints.borrow_mut();
            constraints.weightx = 0.0;
            constraints.gridwidth = 1;
        }
        self.add_expand_button();
        {
            let mut constraints = self.constraints.borrow_mut();
            constraints.weightx = 0.1;
            constraints.gridwidth = 1;
        }
        self.add_header_cell(&self.header1_slices_in_sample);
        self.add_header_cell(&self.header1_current_chunk);
        self.add_header_cell(&self.header1_reference_section);
        self.constraints.borrow_mut().gridwidth = GRID_BAG_REMAINDER;
        self.add_header_cell(&self.header1_current_section);
        // second row
        {
            let mut constraints = self.constraints.borrow_mut();
            constraints.weightx = 0.0;
            constraints.gridwidth = 1;
        }
        self.add_header_cell(&self.header2_z_order);
        {
            let mut constraints = self.constraints.borrow_mut();
            constraints.weightx = 0.2;
            constraints.gridwidth = 2;
        }
        self.add_header_cell(&self.header2_setup_sections);
        {
            let mut constraints = self.constraints.borrow_mut();
            constraints.weightx = 0.1;
            constraints.gridwidth = 1;
        }
        self.add_header_cell(&self.header2_slices_in_sample);
        self.add_header_cell(&self.header2_current_chunk);
        self.add_header_cell(&self.header2_reference_section);
        self.constraints.borrow_mut().gridwidth = GRID_BAG_REMAINDER;
        self.add_header_cell(&self.header2_current_section);
    }

    /// Java private `addJoinTablePanelComponents()`.
    fn add_join_table_panel_components(&self) {
        // Header
        // First row
        {
            let mut constraints = self.constraints.borrow_mut();
            constraints.weightx = 0.0;
            constraints.weighty = 0.2;
            constraints.gridheight = 1;
            constraints.gridwidth = 2;
        }
        self.add_header_cell(&self.header1_z_order);
        {
            let mut constraints = self.constraints.borrow_mut();
            constraints.gridwidth = 1;
            constraints.weightx = 0.2;
        }
        self.add_header_cell(&self.header1_join_sections);
        self.constraints.borrow_mut().weightx = 0.0;
        self.add_expand_button();
        {
            let mut constraints = self.constraints.borrow_mut();
            constraints.weightx = 0.1;
            constraints.gridwidth = GRID_BAG_REMAINDER;
        }
        self.add_header_cell(&self.header1_join_final);
        if self.has_rotated_section() {
            // Second row
            {
                let mut constraints = self.constraints.borrow_mut();
                constraints.weightx = 0.0;
                constraints.gridwidth = 2;
            }
            self.add_header_cell(&self.header2_z_order);
            {
                let mut constraints = self.constraints.borrow_mut();
                constraints.weightx = 0.2;
                constraints.gridwidth = 2;
            }
            self.add_header_cell(&self.header2_join_sections);
            {
                let mut constraints = self.constraints.borrow_mut();
                constraints.weightx = 0.1;
                constraints.gridwidth = GRID_BAG_REMAINDER;
            }
            self.add_header_cell(&self.header2_join_final);
            self.header3_join_sections
                .set_text_string(Some("Orientation"));
        }
        // Third row
        {
            let mut constraints = self.constraints.borrow_mut();
            constraints.weightx = 0.0;
            constraints.gridwidth = 2;
        }
        self.add_header_cell(&self.header3_z_order);
        self.constraints.borrow_mut().weightx = 0.2;
        self.add_header_cell(&self.header3_join_sections);
        {
            let mut constraints = self.constraints.borrow_mut();
            constraints.weightx = 0.1;
            constraints.gridwidth = 1;
        }
        self.add_header_cell(&self.header3_join_final_start);
        self.constraints.borrow_mut().gridwidth = GRID_BAG_REMAINDER;
        self.add_header_cell(&self.header3_join_final_end);
    }

    /// Java private `hasRotatedSection()`.  Returns true if at least one row is
    /// rotated.
    fn has_rotated_section(&self) -> bool {
        self.row_list.has_rotated_section()
    }

    /// Java private `createButtonsPanel()`.
    fn create_buttons_panel(&self) {
        self.pnl_buttons.set_box_layout(spaced_panel::X_AXIS);
        // first component
        self.pnl_buttons_component1
            .set_box_layout(spaced_panel::Y_AXIS);
        self.btn_move_section_up
            .add_action_listener(self.section_table_action_listener.clone());
        self.pnl_buttons_component1
            .add_multi_line_button(&self.btn_move_section_up);
        self.btn_add_section
            .add_action_listener(self.section_table_action_listener.clone());
        self.pnl_buttons_component1
            .add_multi_line_button(&self.btn_add_section);
        // Swing layout: UIUtilities.setButtonSizeAll(pnlButtonsComponent1, buttonDimension).
        // second component
        self.pnl_buttons_component2
            .set_box_layout(spaced_panel::Y_AXIS);
        self.btn_move_section_down
            .add_action_listener(self.section_table_action_listener.clone());
        self.pnl_buttons_component2
            .add_multi_line_button(&self.btn_move_section_down);
        self.btn_delete_section
            .add_action_listener(self.section_table_action_listener.clone());
        self.pnl_buttons_component2
            .add_multi_line_button(&self.btn_delete_section);
        // third component
        let container: Weak<dyn Run3dmodButtonContainer> = self.self_ref.clone();
        let b3b_open_3dmod = BinnedXY3dmodButton::new(Some("Open in 3dmod"), Some(container));
        b3b_open_3dmod.add_action_listener(self.section_table_action_listener.clone());
        b3b_open_3dmod
            .set_spinner_tool_tip_text(Some("The binning to use when opening a section in 3dmod."));
        *self.b3b_open_3dmod.borrow_mut() = Some(b3b_open_3dmod);
        // fourth component
        self.pnl_buttons_component4
            .set_box_layout(spaced_panel::Y_AXIS);
        self.btn_get_angles
            .add_action_listener(self.section_table_action_listener.clone());
        self.pnl_buttons_component4
            .add_multi_line_button(&self.btn_get_angles);
        self.btn_invert_table
            .add_action_listener(self.section_table_action_listener.clone());
        self.pnl_buttons_component4
            .add_multi_line_button(&self.btn_invert_table);
    }

    /// Java private `addButtonsPanelComponents()`.
    fn add_buttons_panel_components(&self) {
        if self.is_setup_tab() {
            self.pnl_buttons
                .add_spaced_panel(&self.pnl_buttons_component1);
            self.pnl_buttons
                .add_spaced_panel(&self.pnl_buttons_component2);
        }
        if !self.is_align_tab() {
            self.pnl_buttons
                .add_container(&self.b3b_open_3dmod().get_container());
        }
        if self.is_setup_tab() {
            self.pnl_buttons
                .add_spaced_panel(&self.pnl_buttons_component4);
        }
    }

    /// Java package-private `displayCurTab()`.
    pub fn display_cur_tab(&self) {
        self.root_panel.remove_all();
        self.pnl_buttons.remove_all();
        self.pnl_border.remove_all();
        self.pnl_viewport.remove_all();
        self.add_root_panel_components();
        self.add_buttons_panel_components();
        self.pnl_table.remove_all();
        self.add_table_panel_components();
        // redisplay rows and calculate chunks
        self.row_list
            .display_cur_tab(&self.pnl_table, &self.viewport);
    }

    /// Java `getChunkSizes()`.
    pub fn get_chunk_sizes(&self) -> String {
        let mut chunk_sizes = String::new();
        for i in 0..self.size() {
            let row = self.row_list.get(i).expect("row in range");
            let data = row.get_data();
            let curr_row_chunk_size = data
                .get_join_final_end()
                .get_int()
                .wrapping_sub(data.get_join_final_start().get_int())
                .wrapping_add(1);
            chunk_sizes.push_str(&curr_row_chunk_size.to_string());
            if i < self.size() - 1 {
                chunk_sizes.push(',');
            }
        }
        chunk_sizes
    }

    /// Java package-private `setInverted() throws FileException, IOException`.
    pub fn set_inverted(&self) -> Result<(), LogFileError> {
        let mut join_info_file = JoinInfoFile::get_instance(self.manager)?;
        let inverted_count = self.row_list.set_inverted(&mut join_info_file);
        if inverted_count > self.row_list.size() / 2 {
            ui_harness::with(|harness| {
                harness.open_message_dialog_base_manager_string_string(
                    Some(self.manager),
                    &format!(
                        "Most of the sections in this join will be inverted.  {}  If you don't \
                         want these inversions, push the \"Change Setup\" button and then push \
                         the \"Invert Table\" button.",
                        section_table_row::INVERTED_WARNING
                    ),
                    "Join Warning",
                )
            });
        }
        Ok(())
    }

    /// Java package-private `setMode()`.  Enable buttons made on the current mode
    /// parameter.
    pub fn set_mode_void(&self) {
        self.set_mode(self.mode.get());
    }

    /// Java package-private `setMode(int)`.  Enable buttons based on the mode
    /// parameter.  The source's `IllegalStateException` for an unknown mode is a
    /// panic.
    pub fn set_mode(&self, mode: i32) {
        self.mode.set(mode);
        // enable buttons that are not effected by highlighting
        match mode {
            join_dialog::SAMPLE_PRODUCED_MODE => {
                self.btn_add_section.set_enabled(false);
                self.btn_move_section_up.set_enabled(false);
                self.btn_move_section_down.set_enabled(false);
                self.btn_delete_section.set_enabled(false);
                self.btn_get_angles.set_enabled(false);
                self.btn_invert_table.set_enabled(false);
            }
            join_dialog::SETUP_MODE
            | join_dialog::SAMPLE_NOT_PRODUCED_MODE
            | join_dialog::CHANGING_SAMPLE_MODE => {
                if !self.rotating.get() {
                    self.btn_add_section.set_enabled(true);
                    self.btn_invert_table.set_enabled(true);
                }
            }
            _ => panic!("java.lang.IllegalStateException: mode={mode}"),
        }
        self.enable_row_buttons(self.row_list.get_highlighted_index());
        self.row_list.set_mode(mode);
    }

    /// Java package-private `setJoinFinalStartHighlight(boolean)`.
    pub fn set_join_final_start_highlight(&self, highlight: bool) {
        self.row_list.set_join_final_start_highlight(highlight);
    }

    /// Java package-private `setJoinFinalEndHighlight(boolean)`.
    pub fn set_join_final_end_highlight(&self, highlight: bool) {
        self.row_list.set_join_final_end_highlight(highlight);
    }

    /// Java private `enableRowButtons(int)`.  Enable row level buttons based on the
    /// current highlight.
    fn enable_row_buttons(&self, highlighted_row_index: i32) {
        let rows_size = self.row_list.size();
        let mode = self.mode.get();
        if rows_size == 0 {
            self.b3b_open_3dmod().set_enabled(false);
            self.button1_expand_sections().set_enabled(false);
            if mode != join_dialog::SAMPLE_PRODUCED_MODE {
                self.btn_move_section_up.set_enabled(false);
                self.btn_move_section_down.set_enabled(false);
                self.btn_delete_section.set_enabled(false);
                self.btn_get_angles.set_enabled(false);
            }
            return;
        }
        self.b3b_open_3dmod()
            .set_enabled(highlighted_row_index > -1);
        self.button1_expand_sections().set_enabled(true);
        if mode != join_dialog::SAMPLE_PRODUCED_MODE {
            self.btn_move_section_up
                .set_enabled(highlighted_row_index > 0);
            self.btn_move_section_down
                .set_enabled(highlighted_row_index > -1 && highlighted_row_index < rows_size - 1);
            self.btn_delete_section
                .set_enabled(highlighted_row_index > -1);
            self.btn_get_angles.set_enabled(highlighted_row_index > -1);
        }
    }

    /// Java package-private `getTableLayout()`.
    pub fn get_table_layout(&self) -> &GridBagLayout {
        &self.layout
    }

    /// Java package-private `getTableConstraints()`: a copy of the shared
    /// constraints (rows that change them use [`Self::with_table_constraints`]).
    pub fn get_table_constraints(&self) -> GridBagConstraints {
        *self.constraints.borrow()
    }

    /// The shared `constraints` object the rows mutate through
    /// `getTableConstraints()`.
    pub fn with_table_constraints<R>(&self, f: impl FnOnce(&mut GridBagConstraints) -> R) -> R {
        let mut constraints = *self.constraints.borrow();
        let result = f(&mut constraints);
        *self.constraints.borrow_mut() = constraints;
        result
    }

    /// Java package-private `enableAddSection()`.
    pub fn enable_add_section(&self) {
        self.rotating.set(false);
        self.set_mode_void();
    }

    /// Java `equals(ConstJoinMetaData)`.
    pub fn equals(&self, meta_data: &dyn ConstJoinMetaData) -> bool {
        let Some(array) = meta_data.get_section_table_data() else {
            return false;
        };
        self.row_list.equals(&array)
    }

    /// Java package-private `equalsSample(ConstJoinMetaData)`.
    pub fn equals_sample(&self, meta_data: &dyn ConstJoinMetaData) -> bool {
        let Some(array) = meta_data.get_section_table_data() else {
            return false;
        };
        self.row_list.equals_sample(&array)
    }

    /// Java private `moveSectionUp()`.  Swap the highlighted row with the one above
    /// it.  Move it in the rows ArrayList.  Move it in the table by removing and adding
    /// the two involved rows and everything below them.  Renumber the row numbers in
    /// the table.
    fn move_section_up(&self) {
        let index = self.row_list.get_highlighted_index();
        if index == -1 {
            return;
        }
        if index == 0 {
            ui_harness::with(|harness| {
                harness.open_message_dialog_base_manager_string_string_axis_id(
                    Some(self.manager),
                    "Can't move the row up.  Its at the top.",
                    "Wrong Row",
                    Some(AxisID::Only),
                )
            });
            return;
        }
        // rowList.removeRows(index - 1);
        self.row_list.move_section_up(index);
        self.viewport.adjust_viewport(index - 1);
        self.row_list.remove_rows();
        self.row_list.display_rows(&self.pnl_table, &self.viewport);
        self.row_list.renumber_table(index - 1);
        if let Err(e) = self.state.move_row_up(index) {
            // An uncaught IllegalStateException: Swing's handler prints it.
            eprintln!("java.lang.IllegalStateException: {e}");
            return;
        }
        self.row_list.configure_rows();
        self.enable_row_buttons(index - 1);
        self.join_dialog().msg_row_change();
        if let Some(main_panel) = self.manager.get_main_panel() {
            main_panel.main_panel().repaint();
        }
    }

    /// Java private `moveSectionDown()`.  Swap the highlighted row with the one below
    /// it.
    fn move_section_down(&self) {
        let index = self.row_list.get_highlighted_index();
        if index == -1 {
            return;
        }
        if index == self.row_list.size() - 1 {
            ui_harness::with(|harness| {
                harness.open_message_dialog_base_manager_string_string_axis_id(
                    Some(self.manager),
                    "Can't move the row down.  Its at the bottom.",
                    "Wrong Row",
                    Some(AxisID::Only),
                )
            });
            return;
        }
        // rowList.removeRows(index);
        self.row_list.move_section_down(index);
        self.viewport.adjust_viewport(index + 1);
        self.row_list.remove_rows();
        self.row_list.display_rows(&self.pnl_table, &self.viewport);
        self.row_list.renumber_table(index);
        if let Err(e) = self.state.move_row_down(index) {
            eprintln!("java.lang.IllegalStateException: {e}");
            return;
        }
        self.row_list.configure_rows();
        self.enable_row_buttons(index + 1);
        self.join_dialog().msg_row_change();
        if let Some(main_panel) = self.manager.get_main_panel() {
            main_panel.main_panel().repaint();
        }
    }

    /// Java private `addSection()`.
    fn add_section(&self) {
        let mut invalid_buffer = String::new();
        if !utilities::is_valid_file(
            self.join_dialog().get_working_dir().as_deref(),
            Some(join_dialog::WORKING_DIRECTORY_TEXT),
            &mut invalid_buffer,
            true,
            true,
            true,
            true,
        ) {
            ui_harness::with(|harness| {
                harness.open_message_dialog_base_manager_string_string_axis_id(
                    Some(self.manager),
                    &invalid_buffer,
                    "Unable to Add Section",
                    Some(AxisID::Only),
                )
            });
            return;
        }
        // Open up the file chooser in the working directory
        let last_location = self.last_location.borrow().clone();
        let chooser =
            FileChooser::new_base_manager_file(Some(self.manager), last_location.as_deref());
        chooser.set_dialog_title(Some("Choose a section"));
        let tomogram_filter =
            TomogramFileFilter::get_all_image_filename_style_instance(self.manager);
        chooser.set_file_filter(Some(tomogram_filter as Rc<dyn FileFilter>));
        // Swing layout: chooser.setPreferredSize(FixedDim.fileChooser).
        chooser.set_file_selection_mode(file_chooser::FILES_ONLY);
        let return_val = chooser.show_open_dialog(Some(&self.pnl_border.get_container()));
        if return_val == file_chooser::APPROVE_OPTION {
            let Some(tomogram) = chooser.get_selected_file() else {
                return;
            };
            *self.last_location.borrow_mut() = tomogram.parent().map(Path::to_path_buf);
            if self.is_duplicate(&tomogram) {
                return;
            }
            let header = MRCHeader::get_instance_in_dir(
                self.manager.get_property_user_dir().as_deref(),
                Some(&utilities::java_io_file_get_absolute_path(
                    &tomogram.to_string_lossy(),
                )),
                Some(AxisID::Only),
            );
            let Some(header) = header else {
                return;
            };
            if !self.read_header(&header) {
                return;
            }
            self.rotating.set(true);
            self.btn_add_section.set_enabled(false);
            self.btn_invert_table.set_enabled(false);
            let (n_rows, n_sections) = {
                let header = header.borrow();
                (header.get_n_rows(), header.get_n_sections())
            };
            if n_rows < n_sections {
                // The tomogram may not be flipped
                // Ask user if can rotate the tomogram
                let msg_flipped = [
                    "It looks like you didn't rotate the tomogram in Post Processing".to_string(),
                    "bacause the tomogram is thicker in Z then it is long in Y.".to_string(),
                    FLIP_WARNING[0].to_string(),
                    FLIP_WARNING[1].to_string(),
                    "Should Etomo use the clip rotx command to rotate -90 degrees in X?"
                        .to_string(),
                ];
                if ui_harness::with(|harness| {
                    harness.open_yes_no_dialog_base_manager_string_array_axis_id(
                        Some(self.manager),
                        &msg_flipped,
                        Some(AxisID::Only),
                    )
                }) {
                    self.manager.rotx(
                        Some(&tomogram),
                        self.join_dialog().get_working_dir().as_deref(),
                        None,
                    );
                    return;
                }
            }
            self.add_section_file(&tomogram);
            ui_harness::with(|harness| {
                harness.pack_axis_id_base_manager(Some(AxisID::Only), Some(self.manager))
            });
        }
    }

    /// Java private `readHeader(MRCHeader)`.
    fn read_header(
        &self,
        header: &std::sync::Arc<crate::imod::etomo::util::mrc_header::SharedMRCHeader>,
    ) -> bool {
        let result = header.borrow_mut().read_with_manager(self.manager);
        match result {
            Ok(true) => {}
            Ok(false) => {
                ui_harness::with(|harness| {
                    harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self.manager),
                        "File does not exist",
                        "System Error",
                        Some(AxisID::Only),
                    )
                });
                return false;
            }
            Err(crate::imod::etomo::util::mrc_header::ReadError::InvalidParameter(message)) => {
                eprintln!("etomo.util.InvalidParameterException: {message}");
                let msg_invalid_parameter_exception = [
                    "The header command returned an error (InvalidParameterException).".to_string(),
                    "This file may not contain a tomogram.".to_string(),
                    "Are you sure you want to open this file?".to_string(),
                ];
                if !ui_harness::with(|harness| {
                    harness.open_yes_no_dialog_base_manager_string_array_axis_id(
                        Some(self.manager),
                        &msg_invalid_parameter_exception,
                        Some(AxisID::Only),
                    )
                }) {
                    return false;
                }
            }
            Err(crate::imod::etomo::util::mrc_header::ReadError::Io(_)) => {
                let (n_rows, n_sections) = {
                    let header = header.borrow();
                    (header.get_n_rows(), header.get_n_sections())
                };
                if n_rows == -1 || n_sections == -1 {
                    let msg_io_exception = [
                        "The header command returned an error (IOException).".to_string(),
                        "Unable to tell if the tomogram is flipped.".to_string(),
                        FLIP_WARNING[0].to_string(),
                        FLIP_WARNING[1].to_string(),
                        "Are you sure you want to open this file?".to_string(),
                    ];
                    if !ui_harness::with(|harness| {
                        harness.open_yes_no_dialog_base_manager_string_array_axis_id(
                            Some(self.manager),
                            &msg_io_exception,
                            Some(AxisID::Only),
                        )
                    }) {
                        return false;
                    }
                }
            }
            Err(crate::imod::etomo::util::mrc_header::ReadError::NumberFormat(message)) => {
                eprintln!("java.lang.NumberFormatException: {message}");
                let (n_rows, n_sections) = {
                    let header = header.borrow();
                    (header.get_n_rows(), header.get_n_sections())
                };
                if n_rows == -1 || n_sections == -1 {
                    let msg_number_format_exception = [
                        "The header command returned an error (NumberFormatException).".to_string(),
                        "Unable to tell if the tomogram is flipped.".to_string(),
                        FLIP_WARNING[0].to_string(),
                        FLIP_WARNING[1].to_string(),
                        "Are you sure you want to open this file?".to_string(),
                    ];
                    if !ui_harness::with(|harness| {
                        harness.open_yes_no_dialog_base_manager_string_array_axis_id(
                            Some(self.manager),
                            &msg_number_format_exception,
                            Some(AxisID::Only),
                        )
                    }) {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Java private `isDuplicate(File)`.
    fn is_duplicate(&self, section: &Path) -> bool {
        if self.row_list.is_duplicate(section) {
            let msg_duplicate = format!(
                "The file, {}, is already in the table.",
                utilities::java_io_file_get_absolute_path(&section.to_string_lossy())
            );
            ui_harness::with(|harness| {
                harness.open_message_dialog_base_manager_string_string_axis_id(
                    Some(self.manager),
                    &msg_duplicate,
                    "Add Section Failed",
                    Some(AxisID::Only),
                )
            });
            return true;
        }
        false
    }

    /// Java package-private `addSection(File)`.
    pub fn add_section_file(&self, tomogram: &Path) {
        self.rotating.set(false);
        self.set_mode_void();
        if !tomogram.exists() {
            ui_harness::with(|harness| {
                harness.open_message_dialog_base_manager_string_string_axis_id(
                    Some(self.manager),
                    &format!(
                        "{} does not exist.",
                        utilities::java_io_file_get_absolute_path(&tomogram.to_string_lossy())
                    ),
                    "File Error",
                    Some(AxisID::Only),
                )
            });
            return;
        }
        if !tomogram.is_file() {
            ui_harness::with(|harness| {
                harness.open_message_dialog_base_manager_string_string_axis_id(
                    Some(self.manager),
                    &format!(
                        "{} is not a file.",
                        utilities::java_io_file_get_absolute_path(&tomogram.to_string_lossy())
                    ),
                    "File Error",
                    Some(AxisID::Only),
                )
            });
            return;
        }
        // Sections are only added in the Setup tab, so assume that the join
        // expand button is contracted.
        let this = self.self_ref.upgrade().expect("SectionTablePanel");
        let index = self.row_list.add(
            self.manager,
            &this,
            tomogram,
            self.button1_expand_sections().is_expanded(),
            self.mode.get(),
        );
        self.viewport.adjust_viewport(index);
        self.row_list.remove_rows();
        self.row_list.display_rows(&self.pnl_table, &self.viewport);
        self.row_list.configure_rows();
        self.join_dialog()
            .set_num_sections(self.row_list.size(), false);
        self.join_dialog().msg_row_change();
        if let Some(main_panel) = self.manager.get_main_panel() {
            main_panel.main_panel().repaint();
        }
    }

    /// Java private `deleteSection()`.  Delete the highlighted row.  Remove it in the
    /// rows ArrayList.  Remove it from the table.  Renumber the row numbers in the
    /// table.
    fn delete_section(&self) {
        let index = self.row_list.get_highlighted_index();
        if index == -1 {
            return;
        }
        if !ui_harness::with(|harness| {
            harness.open_yes_no_dialog_base_manager_string_axis_id(
                Some(self.manager),
                &format!(
                    "Really remove {}?",
                    self.row_list.get_setup_section_text(index)
                ),
                Some(AxisID::Only),
            )
        }) {
            return;
        }
        // rowList.removeRows(index);
        self.row_list.delete_section(index);
        self.row_list.remove_rows();
        self.viewport.adjust_viewport(index);
        self.row_list.display_rows(&self.pnl_table, &self.viewport);
        self.row_list.renumber_table(index);
        if let Err(e) = self.state.delete_row(index) {
            eprintln!("java.lang.IllegalStateException: {e}");
            return;
        }
        self.row_list.configure_rows();
        self.join_dialog()
            .set_num_sections(self.row_list.size(), false);
        self.enable_row_buttons(-1);
        self.join_dialog().msg_row_change();
        if let Some(main_panel) = self.manager.get_main_panel() {
            main_panel.main_panel().repaint();
        }
    }

    /// Java package-private `deleteSections()`.
    pub fn delete_sections(&self) {
        self.row_list.delete_sections();
        if let Some(main_panel) = self.manager.get_main_panel() {
            main_panel.main_panel().repaint();
        }
    }

    /// Java private `imodSection(Run3dmodMenuOptions)`.  Opens a section in 3dmod.
    /// May open a .rot file instead of the original section in the join tab.  Keeps
    /// track of the index of the 3dmod so it can close it and retrieve rotation
    /// angles.
    fn imod_section(&self, menu_options: Option<Run3dmodMenuOptions>) {
        let row_index = self.row_list.get_highlighted_index();
        if row_index == -1 {
            return;
        }
        let binning = self.b3b_open_3dmod().get_binning_in_xand_y();
        let Some(row) = self.row_list.get(row_index) else {
            return;
        };
        if self.is_setup_tab() {
            row.imod_open_setup_section_file(binning, menu_options);
        } else {
            row.imod_open_join_section_file(binning, menu_options);
        }
    }

    /// Java private `imodGetAngles()`.
    fn imod_get_angles(&self) {
        let row_index = self.row_list.get_highlighted_index();
        if row_index == -1 {
            return;
        }
        if self
            .row_list
            .get(row_index)
            .is_some_and(|row| row.imod_get_angles())
            && let Some(main_panel) = self.manager.get_main_panel()
        {
            main_panel.main_panel().repaint();
        }
    }

    /// Java private `invertTable()`.
    fn invert_table(&self) {
        self.row_list.invert_table();
        self.row_list.remove_rows();
        self.row_list.display_rows(&self.pnl_table, &self.viewport);
        self.row_list.configure_rows();
        self.enable_row_buttons(self.row_list.get_highlighted_index());
        self.join_dialog().msg_row_change();
        if let Some(main_panel) = self.manager.get_main_panel() {
            main_panel.main_panel().repaint();
        }
    }

    /// Java package-private `getMetaData(JoinMetaData)`.
    pub fn get_meta_data(&self, meta_data: &JoinMetaData) -> bool {
        meta_data.reset_section_table_data();
        self.row_list.get_meta_data(meta_data, self.manager)
    }

    /// Java package-private `validateMakejoincom()`.
    pub fn validate_makejoincom(&self) -> bool {
        self.row_list.validate_makejoincom()
    }

    /// Java package-private `validateFinishjoin()`.
    pub fn validate_finishjoin(&self) -> bool {
        self.row_list.validate_finishjoin()
    }

    /// Java package-private `setMetaData(ConstJoinMetaData)`.
    pub fn set_meta_data(&self, meta_data: &dyn ConstJoinMetaData) {
        let Some(row_data) = meta_data.get_section_table_data() else {
            return;
        };
        let this = self.self_ref.upgrade().expect("SectionTablePanel");
        self.row_list.set_meta_data(
            &row_data,
            self.manager,
            &this,
            self.mode.get(),
            &self.pnl_table,
            &self.viewport,
        );
        self.row_list.configure_rows();
        self.join_dialog()
            .set_num_sections(self.row_list.size(), true);
        if let Some(main_panel) = self.manager.get_main_panel() {
            main_panel.main_panel().repaint();
        }
    }

    /// Java package-private `getSampleHeaderCell()`.
    pub fn get_sample_header_cell(&self) -> Rc<HeaderCell> {
        self.header1_sample.clone()
    }

    /// Java package-private `getRotationHeaderCell()`.
    pub fn get_rotation_header_cell(&self) -> Rc<HeaderCell> {
        self.header1_rotation.clone()
    }

    /// Java package-private `getJoinFinalHeaderCell()`.
    pub fn get_join_final_header_cell(&self) -> Rc<HeaderCell> {
        self.header1_join_final.clone()
    }

    /// Java package-private `getInvalidReason()`.
    pub fn get_invalid_reason(&self) -> Option<String> {
        self.row_list.get_invalid_reason()
    }

    /// Java package-private `getXMax()`.
    pub fn get_x_max(&self) -> i32 {
        self.row_list.get_x_max()
    }

    /// Java package-private `getYMax()`.
    pub fn get_y_max(&self) -> i32 {
        self.row_list.get_y_max()
    }

    /// Java package-private `getZMax()`.
    pub fn get_z_max(&self) -> i32 {
        self.row_list.get_z_max()
    }

    /// Java package-private `removeCell(Component)`.
    pub fn remove_cell(&self, cell: &Rc<JComponent>) {
        self.pnl_table.remove(cell);
    }

    /// Java package-private `getContainer()`.
    pub fn get_container(&self) -> Rc<JComponent> {
        self.root_panel.clone()
    }

    /// Java package-private `getRootPanel()`.
    pub fn get_root_panel(&self) -> Rc<JComponent> {
        self.root_panel.clone()
    }

    /// Java package-private `synchronize(JoinDialog.Tab, JoinDialog.Tab)`.
    /// Synchronizes when entering or leaving the join tab.  This function should be
    /// called when switching tabs and before any save to the .ejf file.
    pub fn synchronize(&self, prev_tab: Option<Tab>, cur_tab: Option<Tab>) {
        if self.row_list.size() == 0 {
            return;
        }
        // synchronize setup columns to join columns when the user gets to the join
        // tab or the model tab
        if cur_tab == Some(Tab::Join) || cur_tab == Some(Tab::Rejoin) || cur_tab == Some(Tab::Model)
        {
            self.row_list.synchronize_setup_to_join();
            // joinDialog.defaultSizeInXY();
        }
        // synchronize join columns to setup columns when the users leaves the join
        // tab
        else if prev_tab == Some(Tab::Join) || prev_tab == Some(Tab::Rejoin) {
            self.row_list.synchronize_join_to_setup();
        }
    }

    /// Java private `setToolTipText()`.
    fn set_tool_tip_text(&self) {
        self.btn_move_section_up
            .set_tool_tip_text(Some("Press to move the selected section up."));
        self.btn_move_section_down
            .set_tool_tip_text(Some("Press to move the selected section down."));
        self.btn_add_section
            .set_tool_tip_text(Some("Press to add a section to the joined tomogram."));
        self.btn_delete_section.set_tool_tip_text(Some(
            "Press to delete the selected section from the joined tomogram.",
        ));
        self.btn_get_angles.set_tool_tip_text(Some(
            "Press to get the X, Y, and Z rotation from the slicer in 3dmod for the selected \
             section.",
        ));
        self.btn_delete_section
            .set_tool_tip_text(Some("The order of the sections in the joined tomogram."));

        let text = "The sections used in the joined tomogram.";
        self.header1_setup_sections.set_tool_tip_text(Some(text));
        self.header2_setup_sections.set_tool_tip_text(Some(text));
        self.header3_setup_sections.set_tool_tip_text(Some(text));

        let text = "The sections, including rotated sections, used in the joined tomogram.";
        self.header1_join_sections.set_tool_tip_text(Some(text));
        self.header2_join_sections.set_tool_tip_text(Some(text));
        self.header3_join_sections.set_tool_tip_text(Some(text));

        self.header1_sample
            .set_tool_tip_text(Some("The slices to be used in the sample."));
        self.header2_sample_bottom.set_tool_tip_text(Some(
            "The bottom slices to be used in the sample.  The bottom slices should be matched \
             against the top slices of the previous section.",
        ));
        self.header3_sample_bottom_start.set_tool_tip_text(Some(
            "The starting bottom slice to be used in the sample.  The bottom slices should be \
             matched against the top slices of the previous section.",
        ));
        self.header3_sample_bottom_end.set_tool_tip_text(Some(
            "The ending bottom slice to be used in the sample.  The bottom slices should be \
             matched against the top slices of the previous section.",
        ));
        self.header2_sample_top.set_tool_tip_text(Some(
            "The top slices to be used in the sample.  The top slices should be matched against \
             the bottom slices of the next section.",
        ));
        self.header3_sample_top_start.set_tool_tip_text(Some(
            "The starting top slice to be used in the sample.  The top slices should be matched \
             against the bottom slices of the next section.",
        ));
        self.header3_sample_top_end.set_tool_tip_text(Some(
            "The ending top slice to be used in the sample.  The top slices should be matched \
             against the bottom slices of the next section.",
        ));

        let text = "Shows where each of the sample slices comes from.";
        self.header1_slices_in_sample.set_tool_tip_text(Some(text));
        self.header2_slices_in_sample.set_tool_tip_text(Some(text));

        let text = "Shows how the sample is divided up in Midas.";
        self.header1_current_chunk.set_tool_tip_text(Some(text));
        self.header2_current_chunk.set_tool_tip_text(Some(text));

        let text = "The reference section slices for each chunk in Midas.";
        self.header1_reference_section.set_tool_tip_text(Some(text));
        self.header2_reference_section.set_tool_tip_text(Some(text));

        let text = "The current section slices for each chunk in Midas.";
        self.header1_current_section.set_tool_tip_text(Some(text));
        self.header2_current_section.set_tool_tip_text(Some(text));

        let text = "Enter to starting and ending Z values to be used to trim each section in the \
                    joined tomogram.";
        self.header1_setup_final.set_tool_tip_text(Some(text));
        self.header2_setup_final.set_tool_tip_text(Some(text));
        self.header3_setup_final_start.set_tool_tip_text(Some(
            "Enter to starting Z value to be used to trim each section in the joined tomogram.",
        ));
        self.header3_setup_final_end.set_tool_tip_text(Some(
            "Enter to ending Z value to be used to trim the section in the joined tomogram.",
        ));

        let text = "Enter to starting and ending Z values to be used to trim each section or \
                    rotated section in the joined tomogram.";
        self.header1_join_final.set_tool_tip_text(Some(text));
        self.header2_join_final.set_tool_tip_text(Some(text));
        self.header3_join_final_start.set_tool_tip_text(Some(
            "Enter to starting Z values to be used to trim each section or rotated section in \
             the joined tomogram.",
        ));
        self.header3_join_final_end.set_tool_tip_text(Some(
            "Enter to ending Z values to be used to trim each section or rotated section in the \
             joined tomogram.",
        ));

        let text = "The rotation in X, Y, and Z of each section.";
        self.header1_rotation.set_tool_tip_text(Some(text));
        self.header2_rotation.set_tool_tip_text(Some(text));
        self.header3_rotation_x
            .set_tool_tip_text(Some("The rotation in X of each section."));
        self.header3_rotation_y
            .set_tool_tip_text(Some("The rotation in Y of each section."));
        self.header3_rotation_z
            .set_tool_tip_text(Some("The rotation in Z of each section."));
        self.b3b_open_3dmod()
            .set_button_tool_tip_text(Some("Press to open a section in 3dmod."));
        let text = "Order of the sections in Z.";
        self.header1_z_order.set_tool_tip_text(Some(text));
        self.header2_z_order.set_tool_tip_text(Some(text));
        self.header3_z_order.set_tool_tip_text(Some(text));
        self.btn_invert_table
            .set_tool_tip_text(Some("Reverse the order of the sections in the table."));
    }
}

impl Highlightable for SectionTablePanel {
    /// Java `highlight(boolean)`.  Respond to highlight request.
    fn highlight(&self, _highlight: bool) {
        self.set_mode_void();
    }
}

impl HighlightableTableVirtual for SectionTablePanel {
    fn highlightable_table(&self) -> &HighlightableTable {
        &self.base
    }

    /// Java `highlightDownActionPerformed()`.
    fn highlight_down_action_performed(&self) {
        // If the highlight is not visible, don't change the viewport
        let adjust_viewport = self
            .viewport
            .in_viewport(self.row_list.get_highlighted_index());
        let index = self.row_list.highlight_down();
        if index != -1 && adjust_viewport && !self.viewport.in_viewport(index) {
            // The highlight is visible - keep it in the viewport
            if index == 0 {
                self.viewport.home_button_action();
            } else {
                self.viewport.down_button_action();
            }
        }
    }

    /// Java `highlightUpActionPerformed()`.
    fn highlight_up_action_performed(&self) {
        // If the highlight is not visible, don't change the viewport
        let adjust_viewport = self
            .viewport
            .in_viewport(self.row_list.get_highlighted_index());
        let index = self.row_list.highlight_up();
        if index != -1 && adjust_viewport && !self.viewport.in_viewport(index) {
            // The highlight is visible - keep it in the viewport
            if index == self.row_list.size() - 1 {
                self.viewport.end_button_action();
            } else {
                self.viewport.up_button_action();
            }
        }
    }

    /// Java `getFocusableParents()`.
    fn get_focusable_parents(&self) -> Vec<Option<Rc<JComponent>>> {
        self.focusable_parents.borrow().clone()
    }
}

impl Viewable for SectionTablePanel {
    /// Java `msgViewportPaged()`.
    fn msg_viewport_paged(&self) {
        self.display_cur_tab();
        if let Some(main_panel) = self.manager.get_main_panel() {
            main_panel.main_panel().repaint();
        }
    }

    /// Java `size()`.
    fn size(&self) -> i32 {
        self.row_list.size()
    }

    /// Java `getFocusableParents()`.
    fn get_focusable_parents(&self) -> Vec<Rc<JComponent>> {
        self.focusable_parents
            .borrow()
            .iter()
            .flatten()
            .cloned()
            .collect()
    }
}

impl SectionTablePanel {
    /// Java `size()` (the `Viewable` member, called by the rows).
    pub fn size(&self) -> i32 {
        self.row_list.size()
    }
}

impl Expandable for SectionTablePanel {
    /// Java `expand(ExpandButton)`.  Implements the Expandable interface.  Matches
    /// the expand button parameter and performs the expand/contract operation.
    /// Expands the section in each row.  The source's `IllegalStateException` for an
    /// unknown button is a panic.
    fn expand_expand_button(&self, expand_button: &Rc<ExpandButton>) {
        let button1_expand_sections = self.button1_expand_sections();
        if expand_button.equals_expand_button(Some(&button1_expand_sections)) {
            self.row_list.expand(button1_expand_sections.is_expanded());
        } else {
            panic!("java.lang.IllegalStateException: Unknown expand button");
        }
    }

    /// Java `expand(GlobalExpandButton)`, empty.
    fn expand_global_expand_button(&self, _button: &Rc<GlobalExpandButton>) {}
}

impl ContextMenu for SectionTablePanel {
    /// Java `popUpContextMenu(MouseEvent)`.  Right mouse button context menu; empty.
    fn pop_up_context_menu(&self, _mouse_event: &MouseEvent) {}
}

impl Run3dmodButtonContainer for SectionTablePanel {
    /// Java `action(String, Deferred3dmodButton, Run3dmodMenuOptions)`.
    fn action(
        &self,
        command: &str,
        _deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    ) {
        let command = Some(command);
        if command == self.btn_move_section_up.get_action_command().as_deref() {
            self.move_section_up();
        } else if command == self.btn_move_section_down.get_action_command().as_deref() {
            self.move_section_down();
        } else if command == self.btn_add_section.get_action_command().as_deref() {
            self.add_section();
        } else if command == self.btn_delete_section.get_action_command().as_deref() {
            self.delete_section();
        } else if command == self.btn_get_angles.get_action_command().as_deref() {
            self.imod_get_angles();
        } else if command == self.btn_invert_table.get_action_command().as_deref() {
            self.invert_table();
        } else if command == self.b3b_open_3dmod().get_action_command().as_deref() {
            self.imod_section(run_3dmod_menu_options);
        }
    }
}

/// Java private static final nested class `RowList`.  A list of SectionTableRow
/// classes.  Has the functionality of an array and can also run SectionTableRow
/// functions on the whole list.
///
/// The list is iterated over a snapshot of its `Rc`s, since a row's methods call
/// back into the table (`size()`), which reads the list.
struct RowList {
    /// Java private `list = new ArrayList()`.
    list: RefCell<Vec<Rc<SectionTableRow>>>,
    /// Java private final `manager`.
    manager: &'static JoinManager,
}

impl RowList {
    /// Java private `RowList(BaseManager)`.
    fn new(manager: &'static JoinManager) -> RowList {
        RowList {
            list: RefCell::new(Vec::new()),
            manager,
        }
    }

    /// The rows, in order.
    fn rows(&self) -> Vec<Rc<SectionTableRow>> {
        self.list.borrow().clone()
    }

    /// Java private `hasRotatedSection()`.  Returns true if at least one row is
    /// rotated.
    fn has_rotated_section(&self) -> bool {
        self.rows().iter().any(|row| row.is_rotated())
    }

    /// Java private `displayCurTab(JPanel, Viewport)`.
    fn display_cur_tab(&self, pnl_table: &Rc<JComponent>, viewport: &Viewport) {
        let rows = self.rows();
        let mut prev_row: Option<Rc<SectionTableRow>> = None;
        // redisplay rows and calculate chunks
        for (i, row) in rows.iter().enumerate() {
            row.setup_cur_tab(prev_row.as_deref(), rows.len() as i32);
            prev_row = Some(Rc::clone(row));
            row.display(i as i32, pnl_table, viewport);
        }
    }

    /// Java `size()`.
    fn size(&self) -> i32 {
        self.list.borrow().len() as i32
    }

    /// Java `getHighlightedIndex()`.
    fn get_highlighted_index(&self) -> i32 {
        for (i, row) in self.rows().iter().enumerate() {
            if row.is_highlighted() {
                return i as i32;
            }
        }
        -1
    }

    /// Java private `highlight(int)`.
    fn highlight(&self, index: i32) {
        if index < 0 || index >= self.size() {
            return;
        }
        if let Some(row) = self.get(index) {
            row.select_highlight_button();
        }
    }

    /// Java private `highlightDown()`.
    fn highlight_down(&self) -> i32 {
        let mut index = self.get_highlighted_index();
        if index < 0 {
            return -1;
        }
        index += 1;
        if index >= self.size() {
            index = 0;
        }
        self.highlight(index);
        index
    }

    /// Java private `highlightUp()`.
    fn highlight_up(&self) -> i32 {
        let mut index = self.get_highlighted_index();
        if index < 0 {
            return -1;
        }
        index -= 1;
        let size = self.size();
        if index < 0 || index >= size {
            index = size - 1;
        }
        self.highlight(index);
        index
    }

    /// Java private `setInverted(JoinInfoFile)`.
    fn set_inverted(&self, join_info_file: &mut JoinInfoFile) -> i32 {
        let mut inverted_count = 0;
        for (i, row) in self.rows().iter().enumerate() {
            let Some(inverted) = join_info_file.get_inverted(self.manager, i) else {
                continue;
            };
            let inverted: &ConstEtomoNumber = &inverted;
            if inverted.is() {
                inverted_count += 1;
            }
            row.set_inverted(Some(inverted));
        }
        inverted_count
    }

    /// Java private `setMode(int)`.
    fn set_mode(&self, mode: i32) {
        for row in self.rows() {
            row.set_mode(mode);
        }
    }

    /// Java private `setJoinFinalStartHighlight(boolean)`.
    fn set_join_final_start_highlight(&self, highlight: bool) {
        for row in self.rows() {
            row.set_join_final_start_highlight(highlight);
        }
    }

    /// Java private `setJoinFinalEndHighlight(boolean)`.
    fn set_join_final_end_highlight(&self, highlight: bool) {
        for row in self.rows() {
            row.set_join_final_end_highlight(highlight);
        }
    }

    /// Java private `expand(boolean)`.  Expands the section in each row.
    fn expand(&self, expand: bool) {
        for row in self.rows() {
            row.expand_section(expand);
        }
    }

    /// Java `equals(ArrayList)`.
    fn equals(&self, array: &[std::sync::Arc<SectionTableRowData>]) -> bool {
        let rows = self.rows();
        if rows.len() != array.len() {
            return false;
        }
        for (i, row) in rows.iter().enumerate() {
            if !row.equals(&*array[i]) {
                return false;
            }
        }
        true
    }

    /// Java private `equalsSample(ArrayList)`.
    fn equals_sample(&self, array: &[std::sync::Arc<SectionTableRowData>]) -> bool {
        let rows = self.rows();
        if rows.len() != array.len() {
            return false;
        }
        for (i, row) in rows.iter().enumerate() {
            if !row.equals_sample(&*array[i]) {
                return false;
            }
        }
        true
    }

    /// Java private `moveSectionUp(int)`.  Swap the highlighted row with the one above
    /// it.
    fn move_section_up(&self, row_index: i32) {
        let mut list = self.list.borrow_mut();
        let row_index = row_index as usize;
        let row_move_up = list.remove(row_index);
        let row_move_down = list.remove(row_index - 1);
        list.insert(row_index - 1, row_move_up);
        list.insert(row_index, row_move_down);
    }

    /// Java private `moveSectionDown(int)`.
    fn move_section_down(&self, row_index: i32) {
        let mut list = self.list.borrow_mut();
        let row_index = row_index as usize;
        let row_move_up = list.remove(row_index + 1);
        let row_move_down = list.remove(row_index);
        list.insert(row_index, row_move_up);
        list.insert(row_index + 1, row_move_down);
    }

    /// Java private `isDuplicate(File)`.
    fn is_duplicate(&self, section: &Path) -> bool {
        self.rows()
            .iter()
            .any(|row| row.equals_setup_section(section))
    }

    /// Java private `add(JoinManager, SectionTablePanel, File, boolean, int)`.
    /// Creates and adds a row.  Returns the index of the new row.
    fn add(
        &self,
        manager: &'static JoinManager,
        table: &Rc<SectionTablePanel>,
        tomogram: &Path,
        expanded: bool,
        mode: i32,
    ) -> i32 {
        let row =
            SectionTableRow::new_tomogram(manager, table, self.size() + 1, tomogram, expanded);
        row.set_mode(mode);
        row.set_names();
        self.list.borrow_mut().push(row);
        self.size() - 1
    }

    /// Java private `get(int)`.
    fn get(&self, index: i32) -> Option<Rc<SectionTableRow>> {
        if index < 0 || index >= self.size() {
            return None;
        }
        Some(Rc::clone(&self.list.borrow()[index as usize]))
    }

    /// Java private `remove(int)`.
    fn remove(&self, index: i32) -> Option<Rc<SectionTableRow>> {
        if index < 0 || index >= self.size() {
            return None;
        }
        Some(self.list.borrow_mut().remove(index as usize))
    }

    /// Java private `deleteSection(int)`.
    fn delete_section(&self, index: i32) {
        if let Some(row) = self.remove(index) {
            row.remove();
            row.remove_imod();
        }
    }

    /// Java private `deleteSections()`.
    fn delete_sections(&self) {
        while self.size() > 0 {
            if let Some(row) = self.remove(0) {
                row.remove();
            }
        }
    }

    /// Java private `renumberTable(int)`.  Renumber the table starting from the row in
    /// the ArrayList at startIndex.
    fn renumber_table(&self, start_index: i32) {
        let rows = self.rows();
        for i in start_index.max(0) as usize..rows.len() {
            rows[i].set_row_number(i as i32 + 1);
        }
    }

    /// Java private `removeRows()`.  Remove the rows from the table.
    fn remove_rows(&self) {
        for row in self.rows() {
            row.remove();
        }
    }

    /// Java private `displayRows(JPanel, Viewport)`.  Display rows in the table.
    fn display_rows(&self, pnl_table: &Rc<JComponent>, viewport: &Viewport) {
        for (i, row) in self.rows().iter().enumerate() {
            row.display(i as i32, pnl_table, viewport);
        }
    }

    /// Java private `invertTable()`.
    fn invert_table(&self) {
        let rows = self.rows();
        let mut new_rows: Vec<Rc<SectionTableRow>> = Vec::new();
        let mut row_number = 0;
        for row in rows.iter().rev() {
            // place the row in its new position in the array and configure it
            row_number += 1;
            row.set_row_number(row_number);
            row.swap_bottom_top();
            new_rows.push(Rc::clone(row));
        }
        *self.list.borrow_mut() = new_rows;
    }

    /// Java private `getMetaData(JoinMetaData, BaseManager)`.
    fn get_meta_data(&self, meta_data: &JoinMetaData, manager: &'static JoinManager) -> bool {
        let mut success = true;
        for row in self.rows() {
            let row_data = row.get_data();
            if !row.is_valid() {
                success = false; // getData() failed
            }
            meta_data.set_section_table_data(SectionTableRowData::new_from(manager, &row_data));
        }
        success
    }

    /// Java private `setMetaData(ArrayList, JoinManager, SectionTablePanel, int,
    /// JPanel, Viewport)`.
    ///
    /// Fixed in translation: `list.add(rowIndex, row)` throws
    /// IndexOutOfBoundsException for a row index past the end of the list (a
    /// corrupted file); such a row is appended here.
    fn set_meta_data(
        &self,
        row_data: &[std::sync::Arc<SectionTableRowData>],
        manager: &'static JoinManager,
        table: &Rc<SectionTablePanel>,
        mode: i32,
        pnl_table: &Rc<JComponent>,
        viewport: &Viewport,
    ) {
        for data in row_data {
            let row = SectionTableRow::new_data(manager, table, data, false);
            let row_index = data.get_row_index();
            row.set_names();
            let mut list = self.list.borrow_mut();
            if row_index >= 0 && (row_index as usize) <= list.len() {
                list.insert(row_index as usize, row);
            } else {
                list.push(row);
            }
        }
        for (i, row) in self.rows().iter().enumerate() {
            row.set_mode(mode);
            row.display(i as i32, pnl_table, viewport);
        }
    }

    /// Java private `validateMakejoincom()`.
    fn validate_makejoincom(&self) -> bool {
        let max_row = self.size().to_string();
        for row in self.rows() {
            if !row.validate_makejoincom(&max_row) {
                return false;
            }
        }
        true
    }

    /// Java private `validateFinishjoin()`.
    fn validate_finishjoin(&self) -> bool {
        for row in self.rows() {
            if !row.validate_finishjoin() {
                return false;
            }
        }
        true
    }

    /// Java private `configureRows()`.
    fn configure_rows(&self) {
        for row in self.rows() {
            row.set_in_use();
        }
    }

    /// Java private `getInvalidReason()`.
    fn get_invalid_reason(&self) -> Option<String> {
        for row in self.rows() {
            let invalid_reason = row.get_invalid_reason();
            if invalid_reason.is_some() {
                return invalid_reason;
            }
        }
        None
    }

    /// Java private `getXMax()`.
    fn get_x_max(&self) -> i32 {
        let mut x_max = 0;
        for row in self.rows() {
            x_max = x_max.max(row.get_x_max());
        }
        x_max
    }

    /// Java private `getYMax()`.
    fn get_y_max(&self) -> i32 {
        let mut y_max = 0;
        for row in self.rows() {
            y_max = y_max.max(row.get_y_max());
        }
        y_max
    }

    /// Java private `getZMax()`.
    fn get_z_max(&self) -> i32 {
        let mut z_max = 0;
        for row in self.rows() {
            z_max = z_max.max(row.get_z_max());
        }
        z_max
    }

    /// Java private `getSetupSectionText(int)`.
    fn get_setup_section_text(&self, index: i32) -> String {
        if index >= 0 && index < self.size() {
            return self
                .get(index)
                .map(|row| row.get_setup_section_text())
                .unwrap_or_default();
        }
        String::new()
    }

    /// Java private `synchronizeSetupToJoin()`.
    fn synchronize_setup_to_join(&self) {
        for row in self.rows() {
            row.synchronize_setup_to_join();
        }
    }

    /// Java private `synchronizeJoinToSetup()`.
    fn synchronize_join_to_setup(&self) {
        for row in self.rows() {
            row.synchronize_join_to_setup();
        }
    }
}
