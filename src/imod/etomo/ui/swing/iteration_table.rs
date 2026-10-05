//! `IMOD/Etomo/src/etomo/ui/swing/IterationTable.java`.
//!
//! The PEET dialog's iteration table: one `IterationRow` per alignment iteration,
//! three header rows, side buttons (Up, Down, Insert, Delete, Dup) and the "Remove
//! duplicates", "Bandpass filtering" and "Strict search limit checking" check boxes.
//! An event dispatch thread object, created as `Rc<Self>` by
//! [`IterationTable::get_instance`]; it keeps a weak reference to its parent.
//! `IterationTable extends HighlightableTable`: the superclass is the `base` field.

use std::cell::{Cell as StdCell, RefCell};
use std::rc::{Rc, Weak};

use super::cell::CellVirtual;
use super::check_box::CheckBox;
use super::etched_border::EtchedBorder;
use super::header_cell::HeaderCell;
use super::highlightable::Highlightable;
use super::highlightable_table::{HighlightableTable, HighlightableTableVirtual};
use super::iteration_parent::IterationParent;
use super::iteration_row::IterationRow;
use super::single_line_button::SingleLineButton;
use super::spaced_panel::{self, SpacedPanel};
use super::ui_harness;
use super::ui_parameters::UIParameters;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::jdk::{
    ActionEvent, ActionListener, GRID_BAG_BOTH, GRID_BAG_CENTER, GRID_BAG_REMAINDER,
    GridBagConstraints, GridBagLayout, JComponent,
};
use crate::imod::etomo::storage::autodoc::autodoc_factory;
use crate::imod::etomo::storage::autodoc::read_only_autodoc::ReadOnlyAutodoc;
use crate::imod::etomo::storage::log_file::LogFileError;
use crate::imod::etomo::storage::matlab_param::{self, MatlabParam};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_peet_meta_data::ConstPeetMetaData;
use crate::imod::etomo::r#type::etomo_autodoc;
use crate::imod::etomo::r#type::peet_meta_data::PeetMetaData;
use crate::imod::etomo::ui::shared_strings;

/// Java `D_PHI_D_THETA_D_PSI_HEADER1`.
pub const D_PHI_D_THETA_D_PSI_HEADER1: &str = "Angular Search Range";
/// Java `INCR_HEADER3`.
pub const INCR_HEADER3: &str = "Step";
/// Java `SEARCH_RADIUS_HEADER1`.
pub const SEARCH_RADIUS_HEADER1: &str = "Search";
/// Java `SEARCH_RADIUS_HEADER2`.
pub const SEARCH_RADIUS_HEADER2: &str = "Distance";
/// Java `LABEL`.
pub const LABEL: &str = "Iteration Table";
/// Java `MAX_HEADER3`.
pub const MAX_HEADER3: &str = "Max";
/// Java `HICUTOFF_HEADER1`.
pub const HICUTOFF_HEADER1: &str = "Low-pass";
/// Java `HICUTOFF_HEADER2`.
pub const HICUTOFF_HEADER2: &str = "Filter";
/// Java `HICUTOFF_CUTOFF_HEADER3`.
pub const HICUTOFF_CUTOFF_HEADER3: &str = "Cutoff";
/// Java `HICUTOFF_SIGMA_HEADER3`.
pub const HICUTOFF_SIGMA_HEADER3: &str = "Sigma";
/// Java `LOWCUTOFF_HEADER1`.
pub const LOWCUTOFF_HEADER1: &str = "High-pass";
/// Java `LOWCUTOFF_HEADER2`.
pub const LOWCUTOFF_HEADER2: &str = "Filter";
/// Java `LOWCUTOFF_CUTOFF_HEADER3`.
pub const LOWCUTOFF_CUTOFF_HEADER3: &str = "Cutoff";
/// Java `LOWCUTOFF_SIGMA_HEADER3`.
pub const LOWCUTOFF_SIGMA_HEADER3: &str = "Sigma";
/// Java `REF_THRESHOLD_HEADER1`.
pub const REF_THRESHOLD_HEADER1: &str = "Ref";
/// Java `REF_THRESHOLD_HEADER2`.
pub const REF_THRESHOLD_HEADER2: &str = "Threshold";
/// Java `DUPLICATE_TOLERANCE_HEADER1`.
pub const DUPLICATE_TOLERANCE_HEADER1: &str = "Duplicate";
/// Java `DUPLICATE_TOLERANCE_HEADER2`.
pub const DUPLICATE_TOLERANCE_HEADER2: &str = "Tolerance";
/// Java `DUPLICATE_SHIFT_TOLERANCE_HEADER3`.
pub const DUPLICATE_SHIFT_TOLERANCE_HEADER3: &str = "Shift";
/// Java `DUPLICATE_ANGULAR_TOLERANCE_HEADER3`.
pub const DUPLICATE_ANGULAR_TOLERANCE_HEADER3: &str = "Angle";

/// Java package-private `final class IterationTable extends HighlightableTable`.
pub struct IterationTable {
    /// Java superclass `HighlightableTable`.
    base: HighlightableTable,
    /// Java private final `rootPanel`.
    root_panel: Rc<JComponent>,
    /// Java private final `pnlTable`.
    pnl_table: Rc<JComponent>,
    /// Java private final `layout`.
    layout: GridBagLayout,
    /// Java private final `rowList`.
    row_list: RowList,
    /// Java private final `constraints`.
    constraints: RefCell<GridBagConstraints>,
    /// Java private final `header1IterationNumber`.
    header1_iteration_number: Rc<HeaderCell>,
    /// Java private final `header2IterationNumber`.
    header2_iteration_number: Rc<HeaderCell>,
    /// Java private final `header3IterationNumber`.
    header3_iteration_number: Rc<HeaderCell>,
    /// Java private final `header1DPhiDThetaDPsi`.
    header1_d_phi_d_theta_d_psi: Rc<HeaderCell>,
    /// Java private final `header2DPhi`.
    header2_d_phi: Rc<HeaderCell>,
    /// Java private final `header2DTheta`.
    header2_d_theta: Rc<HeaderCell>,
    /// Java private final `header2DPsi`.
    header2_d_psi: Rc<HeaderCell>,
    /// Java private final `header3DPhiMax`.
    header3_d_phi_max: Rc<HeaderCell>,
    /// Java private final `header3DPhiIncrement`.
    header3_d_phi_increment: Rc<HeaderCell>,
    /// Java private final `header3DThetaMax`.
    header3_d_theta_max: Rc<HeaderCell>,
    /// Java private final `header3DThetaIncrement`.
    header3_d_theta_increment: Rc<HeaderCell>,
    /// Java private final `header3DPsiMax`.
    header3_d_psi_max: Rc<HeaderCell>,
    /// Java private final `header3DPsiIncrement`.
    header3_d_psi_increment: Rc<HeaderCell>,
    /// Java private final `header1SearchRadius`.
    header1_search_radius: Rc<HeaderCell>,
    /// Java private final `header2SearchRadius`.
    header2_search_radius: Rc<HeaderCell>,
    /// Java private final `header3SearchRadius`.
    header3_search_radius: Rc<HeaderCell>,
    /// Java private final `header1HiCutoff`.
    header1_hi_cutoff: Rc<HeaderCell>,
    /// Java private final `header2HiCutoff`.
    header2_hi_cutoff: Rc<HeaderCell>,
    /// Java private final `header3HiCutoff`.
    header3_hi_cutoff: Rc<HeaderCell>,
    /// Java private final `header3HiCutoffSigma`.
    header3_hi_cutoff_sigma: Rc<HeaderCell>,
    /// Java private final `header1LowCutoff`.
    header1_low_cutoff: Rc<HeaderCell>,
    /// Java private final `header2LowCutoff`.
    header2_low_cutoff: Rc<HeaderCell>,
    /// Java private final `header3LowCutoff`.
    header3_low_cutoff: Rc<HeaderCell>,
    /// Java private final `header3LowCutoffSigma`.
    header3_low_cutoff_sigma: Rc<HeaderCell>,
    /// Java private final `header1RefThreshold`.
    header1_ref_threshold: Rc<HeaderCell>,
    /// Java private final `header2RefThreshold`.
    header2_ref_threshold: Rc<HeaderCell>,
    /// Java private final `header3RefThreshold`.
    header3_ref_threshold: Rc<HeaderCell>,
    /// Java private final `header1DuplicateTolerance`.
    header1_duplicate_tolerance: Rc<HeaderCell>,
    /// Java private final `header2DuplicateTolerance`.
    header2_duplicate_tolerance: Rc<HeaderCell>,
    /// Java private final `header3DuplicateShiftTolerance`.
    header3_duplicate_shift_tolerance: Rc<HeaderCell>,
    /// Java private final `header3DuplicateAngularTolerance`.
    header3_duplicate_angular_tolerance: Rc<HeaderCell>,
    /// Java private final `btnMoveUp`.
    btn_move_up: Rc<SingleLineButton>,
    /// Java private final `btnMoveDown`.
    btn_move_down: Rc<SingleLineButton>,
    /// Java private final `btnAddRow`.
    btn_add_row: Rc<SingleLineButton>,
    /// Java private final `btnDeleteRow`.
    btn_delete_row: Rc<SingleLineButton>,
    /// Java private final `btnCopyRow`.
    btn_copy_row: Rc<SingleLineButton>,
    /// Java private final `cbFlgRemoveDuplicates`.
    cb_flg_remove_duplicates: Rc<CheckBox>,
    /// Java private final `cbLowCutoff`.
    cb_low_cutoff: Rc<CheckBox>,
    /// Java private final `cbFlgStrictSearchLimits`.
    cb_flg_strict_search_limits: Rc<CheckBox>,
    /// Java private final `pnlTableAndCheckbox`.
    pnl_table_and_checkbox: Rc<JComponent>,
    /// Java private final `pnlFlgRemoveDuplicates`.
    pnl_flg_remove_duplicates: Rc<JComponent>,
    /// Java private final `manager`.
    manager: &'static dyn BaseManager,
    /// Java private final `parent`.
    parent: Weak<dyn IterationParent>,
    /// Java private `verticalRigidArea1` (a layout spacer; whether it is in the
    /// panel).
    vertical_rigid_area1: StdCell<bool>,
    /// Java `this`.
    self_ref: Weak<IterationTable>,
}

impl IterationTable {
    /// Java private `IterationTable(BaseManager, IterationParent)`.
    fn new(
        manager: &'static dyn BaseManager,
        parent: Weak<dyn IterationParent>,
    ) -> Rc<IterationTable> {
        let ui_parameters = UIParameters::get_instance_void();
        let numeric_width = ui_parameters.get_numeric_width();
        let wide_numeric_width = ui_parameters.get_wide_numeric_width();
        let this = Rc::new_cyclic(|self_ref: &Weak<IterationTable>| IterationTable {
            // super("Interation")
            base: HighlightableTable::new("Interation"),
            root_panel: JComponent::new_panel(),
            pnl_table: JComponent::new_panel(),
            layout: GridBagLayout::new(),
            row_list: RowList::new(manager, self_ref.clone()),
            constraints: RefCell::new(GridBagConstraints::default()),
            header1_iteration_number: HeaderCell::new_string(Some("Run #")),
            header2_iteration_number: HeaderCell::new_void(),
            header3_iteration_number: HeaderCell::new_void(),
            header1_d_phi_d_theta_d_psi: HeaderCell::new_string(Some(D_PHI_D_THETA_D_PSI_HEADER1)),
            header2_d_phi: HeaderCell::new_string(Some(shared_strings::D_PHI_LABEL)),
            header2_d_theta: HeaderCell::new_string(Some(shared_strings::D_THETA_LABEL)),
            header2_d_psi: HeaderCell::new_string(Some(shared_strings::D_PSI_LABEL)),
            header3_d_phi_max: HeaderCell::new_string_int(Some(MAX_HEADER3), numeric_width),
            header3_d_phi_increment: HeaderCell::new_string_int(Some(INCR_HEADER3), numeric_width),
            header3_d_theta_max: HeaderCell::new_string_int(Some(MAX_HEADER3), numeric_width),
            header3_d_theta_increment: HeaderCell::new_string_int(
                Some(INCR_HEADER3),
                numeric_width,
            ),
            header3_d_psi_max: HeaderCell::new_string_int(Some(MAX_HEADER3), numeric_width),
            header3_d_psi_increment: HeaderCell::new_string_int(Some(INCR_HEADER3), numeric_width),
            header1_search_radius: HeaderCell::new_string(Some(SEARCH_RADIUS_HEADER1)),
            header2_search_radius: HeaderCell::new_string(Some(SEARCH_RADIUS_HEADER2)),
            header3_search_radius: HeaderCell::new_int(ui_parameters.get_integer_triplet_width()),
            header1_hi_cutoff: HeaderCell::new_string(Some(HICUTOFF_HEADER1)),
            header2_hi_cutoff: HeaderCell::new_string(Some(HICUTOFF_HEADER2)),
            header3_hi_cutoff: HeaderCell::new_string_int(
                Some(HICUTOFF_CUTOFF_HEADER3),
                wide_numeric_width,
            ),
            header3_hi_cutoff_sigma: HeaderCell::new_string_int(
                Some(HICUTOFF_SIGMA_HEADER3),
                wide_numeric_width,
            ),
            header1_low_cutoff: HeaderCell::new_string(Some(LOWCUTOFF_HEADER1)),
            header2_low_cutoff: HeaderCell::new_string(Some(LOWCUTOFF_HEADER2)),
            header3_low_cutoff: HeaderCell::new_string_int(
                Some(LOWCUTOFF_CUTOFF_HEADER3),
                wide_numeric_width,
            ),
            header3_low_cutoff_sigma: HeaderCell::new_string_int(
                Some(LOWCUTOFF_SIGMA_HEADER3),
                wide_numeric_width,
            ),
            header1_ref_threshold: HeaderCell::new_string(Some(REF_THRESHOLD_HEADER1)),
            header2_ref_threshold: HeaderCell::new_string(Some(REF_THRESHOLD_HEADER2)),
            header3_ref_threshold: HeaderCell::new_void(),
            header1_duplicate_tolerance: HeaderCell::new_string(Some(DUPLICATE_TOLERANCE_HEADER1)),
            header2_duplicate_tolerance: HeaderCell::new_string(Some(DUPLICATE_TOLERANCE_HEADER2)),
            header3_duplicate_shift_tolerance: HeaderCell::new_string(Some(
                DUPLICATE_SHIFT_TOLERANCE_HEADER3,
            )),
            header3_duplicate_angular_tolerance: HeaderCell::new_string(Some(
                DUPLICATE_ANGULAR_TOLERANCE_HEADER3,
            )),
            btn_move_up: SingleLineButton::get_html_instance(Some("Up")),
            btn_move_down: SingleLineButton::get_html_instance(Some("Down")),
            btn_add_row: SingleLineButton::get_html_instance(Some("Insert")),
            btn_delete_row: SingleLineButton::get_html_instance(Some("Delete")),
            btn_copy_row: SingleLineButton::get_html_instance(Some("Dup")),
            cb_flg_remove_duplicates: CheckBox::new_string(Some(
                shared_strings::FLG_REMOVE_DUPLICATES_LABEL,
            )),
            cb_low_cutoff: CheckBox::new_string(Some(shared_strings::BANDPASS_FILTERING_LABEL)),
            cb_flg_strict_search_limits: CheckBox::new_string(Some(
                shared_strings::FLG_STRICT_SEARCH_LIMITS_LABEL,
            )),
            pnl_table_and_checkbox: JComponent::new_panel(),
            pnl_flg_remove_duplicates: JComponent::new_panel(),
            manager,
            parent,
            vertical_rigid_area1: StdCell::new(false),
            self_ref: self_ref.clone(),
        });
        // Constructor body.
        this.create_table();
        this.row_list.add(
            Rc::downgrade(&this) as Weak<dyn Highlightable>,
            &this.pnl_table,
            this.cb_low_cutoff.is_selected(),
        );
        this.display();
        this.update_display_void();
        this.refresh_vertical_padding();
        this.set_tool_tip_text();
        this
    }

    /// Java static `getInstance(BaseManager, IterationParent)`.
    pub fn get_instance(
        manager: &'static dyn BaseManager,
        parent: Weak<dyn IterationParent>,
    ) -> Rc<IterationTable> {
        let instance = IterationTable::new(manager, parent);
        instance.add_listeners();
        let table: Weak<dyn HighlightableTableVirtual> = Rc::downgrade(&instance) as _;
        instance.base.init_highlight_hotkeys(table);
        instance
    }

    /// Java field read `parent`.
    fn parent(&self) -> Rc<dyn IterationParent> {
        self.parent
            .upgrade()
            .expect("the PEET dialog owns its iteration table")
    }

    /// Java field read `layout` (the rows add their cells with it).
    pub fn get_layout(&self) -> &GridBagLayout {
        &self.layout
    }

    /// Java field read `constraints`: a copy of the shared constraints.
    pub fn get_constraints(&self) -> GridBagConstraints {
        *self.constraints.borrow()
    }

    /// The shared `constraints` object the rows change.
    pub fn with_constraints<R>(&self, f: impl FnOnce(&mut GridBagConstraints) -> R) -> R {
        let mut constraints = *self.constraints.borrow();
        let result = f(&mut constraints);
        *self.constraints.borrow_mut() = constraints;
        result
    }

    /// Java package-private `validateRun()`.
    pub fn validate_run(&self) -> bool {
        self.row_list.validate_run(self.cb_low_cutoff.is_selected())
    }

    /// Java package-private `getContainer()`.
    pub fn get_container(&self) -> Rc<JComponent> {
        self.root_panel.clone()
    }

    /// Java package-private `reset(boolean)`.
    pub fn reset(&self, init: bool) {
        self.row_list.remove();
        self.add_row(init, false);
        self.cb_flg_remove_duplicates.set_selected_boolean(false);
        self.update_display_void();
        ui_harness::with(|harness| harness.pack_base_manager(Some(self.manager)));
    }

    /// Java package-private `getParameters(MatlabParam)`.
    pub fn get_parameters_matlab_param(&self, matlab_param_file: &mut MatlabParam) {
        self.row_list
            .get_parameters_matlab_param(matlab_param_file, self.cb_low_cutoff.is_selected());
        matlab_param_file.set_flg_remove_duplicates(self.cb_flg_remove_duplicates.is_selected());
        matlab_param_file
            .set_flg_strict_search_limits(self.cb_flg_strict_search_limits.is_selected());
    }

    /// Java package-private `getParameters(PeetMetaData)`.
    pub fn get_parameters_meta_data(&self, meta_data: &PeetMetaData) {
        meta_data.set_is_low_cutoff(self.cb_low_cutoff.is_selected());
        self.row_list.get_parameters_meta_data(meta_data);
    }

    /// Java package-private `setParameters(ConstPeetMetaData)`.
    pub fn set_parameters_meta_data(&self, meta_data: &dyn ConstPeetMetaData) {
        let row_list_size = self.row_list.size();
        self.cb_low_cutoff
            .set_selected_boolean(meta_data.is_low_cutoff());
        for i in 0..row_list_size {
            if let Some(row) = self.row_list.get_row(i) {
                row.set_parameters_meta_data(Some(meta_data));
            }
        }
    }

    /// Java package-private `updateDisplay(boolean)`.  Update display in rows.
    pub fn update_display(&self, sample_sphere: bool) {
        self.row_list
            .update_display(sample_sphere, self.cb_flg_remove_duplicates.is_selected());
    }

    /// Java package-private `checkLowCutoffBackwardsCompatibility(MatlabParam)`.
    pub fn check_low_cutoff_backwards_compatibility(&self, matlab_param_file: &mut MatlabParam) {
        let row_list_size = self.row_list.size();
        for i in 0..row_list_size {
            if let Some(row) = self.row_list.get_row(i)
                && !row.check_low_cutoff_backwards_compatibility(matlab_param_file)
            {
                self.cb_low_cutoff.set_selected_boolean(true);
                break;
            }
        }
        if !self.cb_low_cutoff.is_selected() {
            for i in 0..row_list_size {
                if let Some(row) = self.row_list.get_row(i) {
                    row.set_low_cutoff_sigma(Some(matlab_param::LOW_CUTOFF_SIGMA_DEFAULT));
                }
            }
        }
    }

    /// Java package-private `setParameters(MatlabParam)`.
    pub fn set_parameters_matlab_param(&self, matlab_param_file: &mut MatlabParam) {
        // overwrite existing rows
        let row_list_size = self.row_list.size();
        self.set_visible_high_pass_filter_column(self.cb_low_cutoff.is_selected());
        for i in 0..row_list_size {
            if let Some(row) = self.row_list.get_row(i) {
                row.set_parameters_matlab_param(
                    matlab_param_file,
                    self.cb_low_cutoff.is_selected(),
                );
                row.set_visible_low_cutoff_rows(self.cb_low_cutoff.is_selected());
            }
        }
        self.cb_flg_remove_duplicates
            .set_selected_boolean(matlab_param_file.is_flg_remove_duplicates());
        self.cb_flg_strict_search_limits
            .set_selected_boolean(matlab_param_file.is_flg_strict_search_limits());
        self.update_display_void();
        ui_harness::with(|harness| harness.pack_base_manager(Some(self.manager)));
    }

    /// Java package-private `addIterationRows(MatlabParam)`.
    pub fn add_iteration_rows(&self, matlab_param_file: &mut MatlabParam) {
        let row_list_size = self.row_list.size();
        for _ in row_list_size..matlab_param_file.get_iteration_list_size() {
            let row = self.add_row(true, false);
            row.set_parameters_matlab_param(matlab_param_file, self.cb_low_cutoff.is_selected());
        }
    }

    /// Java private `addRow(boolean, boolean)`.
    fn add_row(&self, init: bool, action_insert_btn: bool) -> Rc<IterationRow> {
        let row = self.row_list.add(
            self.self_ref.clone() as Weak<dyn Highlightable>,
            &self.pnl_table,
            self.cb_low_cutoff.is_selected(),
        );
        if action_insert_btn {
            // Leave default cutoff empty and populate sigma values.
            // Cutoffs are left empty so that the user remembers to add cutoffs.
            // Leaving cutoffs default makes it very likely for the filter to be
            // left disabled even though checkbox for bandpass filtering is ON.
            row.set_low_cutoff_sigma(Some(matlab_param::LOW_CUTOFF_SIGMA_DEFAULT));
        }
        row.display();
        self.parent().update_display(init);
        self.refresh_vertical_padding();
        row
    }

    /// Java package-private `size()`.
    pub fn size(&self) -> i32 {
        self.row_list.size()
    }

    /// Java private `setToolTipText()`.
    fn set_tool_tip_text(&self) {
        let autodoc = match unsafe {
            autodoc_factory::get_instance(
                Some(self.manager),
                Some(autodoc_factory::PEET_PRM),
                AxisID::Only,
                false,
            )
        } {
            Ok(autodoc) => autodoc,
            Err(LogFileError::Lock(_)) => std::ptr::null_mut(),
            Err(e) => {
                eprintln!("{e}");
                std::ptr::null_mut()
            }
        };
        let autodoc = unsafe { autodoc.as_ref() }.map(|autodoc| autodoc as &dyn ReadOnlyAutodoc);
        self.header3_duplicate_shift_tolerance.set_tool_tip_text(
            etomo_autodoc::get_tooltip(autodoc, Some(matlab_param::DUPLICATE_SHIFT_TOLERANCE_KEY))
                .as_deref(),
        );
        self.header3_duplicate_angular_tolerance.set_tool_tip_text(
            etomo_autodoc::get_tooltip(
                autodoc,
                Some(matlab_param::DUPLICATE_ANGULAR_TOLERANCE_KEY),
            )
            .as_deref(),
        );
        self.btn_add_row
            .set_tool_tip_text(Some("Add a new iteration row to the table."));
        self.btn_copy_row.set_tool_tip_text(Some(
            "Create a new row that is a duplicate of the highlighted row.",
        ));
        self.btn_move_up
            .set_tool_tip_text(Some("Move highlighted row up in the table."));
        self.btn_move_down
            .set_tool_tip_text(Some("Move highlighted row down in the table"));
        self.btn_delete_row
            .set_tool_tip_text(Some("Remove highlighted row from table."));
        self.cb_flg_remove_duplicates.set_tool_tip_text_string(Some(
            "Remove mulitple references to the same particle aftereach iteration.",
        ));
        self.cb_low_cutoff
            .set_tool_tip_text_string(Some("Show both low-pass and high-pass filtration settings"));
        self.cb_flg_strict_search_limits.set_tool_tip_text_string(Some(
            "When checked, the overall change for any parameter will be limited to the largest change specified at any single iteration.",
        ));
    }

    /// Java private `addListeners()` with `ITActionListener`.
    fn add_listeners(&self) {
        let adaptee = self.self_ref.clone();
        let action_listener: ActionListener = Rc::new(move |event: &ActionEvent| {
            if let Some(iteration_table) = adaptee.upgrade() {
                iteration_table.action(event);
            }
        });
        self.btn_add_row
            .add_action_listener(action_listener.clone());
        self.btn_copy_row
            .add_action_listener(action_listener.clone());
        self.btn_delete_row
            .add_action_listener(action_listener.clone());
        self.btn_move_up
            .add_action_listener(action_listener.clone());
        self.btn_move_down
            .add_action_listener(action_listener.clone());
        self.cb_flg_remove_duplicates
            .add_action_listener(Some(action_listener.clone()));
        self.cb_low_cutoff
            .add_action_listener(Some(action_listener));
    }

    /// Java private `action(ActionEvent)`.
    fn action(&self, event: &ActionEvent) {
        let action_command = event.get_action_command();
        let action_command = action_command.as_deref();
        if action_command == self.btn_add_row.get_action_command().as_deref() {
            self.add_row(false, true);
            ui_harness::with(|harness| harness.pack_base_manager(Some(self.manager)));
        } else if action_command == self.btn_copy_row.get_action_command().as_deref() {
            let Some(row) = self.row_list.get_highlighted_row() else {
                return;
            };
            self.copy_row(&row);
        } else if action_command == self.btn_delete_row.get_action_command().as_deref() {
            let Some(row) = self.row_list.get_highlighted_row() else {
                return;
            };
            self.delete_row(&row);
        } else if action_command == self.btn_move_up.get_action_command().as_deref() {
            self.move_row_up();
        } else if action_command == self.btn_move_down.get_action_command().as_deref() {
            self.move_row_down();
        } else if action_command
            == self
                .cb_flg_remove_duplicates
                .get_action_command()
                .as_deref()
        {
            self.update_display(self.parent().is_sample_sphere());
        } else if action_command == self.cb_low_cutoff.get_action_command().as_deref() {
            self.set_visible_high_pass_filter_column(self.cb_low_cutoff.is_selected());
            ui_harness::with(|harness| harness.pack_base_manager(Some(self.manager)));
        }
    }

    /// Java private `copyRow(IterationRow)`.
    fn copy_row(&self, row: &IterationRow) {
        self.row_list.copy(row, self.cb_low_cutoff.is_selected());
        self.parent().update_display(false);
        self.refresh_vertical_padding();
        ui_harness::with(|harness| harness.pack_base_manager(Some(self.manager)));
    }

    /// Java private `deleteRow(IterationRow)`.
    fn delete_row(&self, row: &IterationRow) {
        self.row_list.remove();
        let index = self.row_list.delete(row);
        self.row_list.highlight(index);
        self.row_list.display();
        self.update_display_void();
        self.refresh_vertical_padding();
        ui_harness::with(|harness| harness.pack_base_manager(Some(self.manager)));
    }

    /// Java private `moveRowUp()`.  Swap the highlighted row with the one above it.
    fn move_row_up(&self) {
        let index = self.row_list.get_highlight_index();
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
        self.row_list.move_row_up(index);
        self.row_list.remove();
        self.row_list.reindex(index - 1);
        self.row_list.display();
        self.update_display_void();
        if let Some(main_panel) = self.manager.get_main_panel() {
            main_panel.main_panel().repaint();
        }
    }

    /// Java private `moveRowDown()`.  Swap the highlighted row with the one below it.
    fn move_row_down(&self) {
        let index = self.row_list.get_highlight_index();
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
        self.row_list.move_row_down(index);
        self.row_list.remove();
        self.row_list.reindex(index);
        self.row_list.display();
        self.update_display_void();
        if let Some(main_panel) = self.manager.get_main_panel() {
            main_panel.main_panel().repaint();
        }
    }

    /// Java private `updateDisplay()`.
    fn update_display_void(&self) {
        let highlight_index = self.row_list.get_highlight_index();
        self.btn_copy_row.set_enabled(highlight_index != -1);
        self.btn_delete_row.set_enabled(highlight_index != -1);
        self.btn_move_up.set_enabled(highlight_index > 0);
        self.btn_move_down
            .set_enabled(highlight_index != -1 && highlight_index < self.row_list.size() - 1);
    }

    /// Java private `createTable()`.
    fn create_table(&self) {
        // init
        self.set_visible_high_pass_filter_column(self.cb_low_cutoff.is_selected());
        // local panels
        let pnl_buttons = JComponent::new_panel();
        // table: GridBagLayout, black line border
        {
            let mut constraints = self.constraints.borrow_mut();
            constraints.fill = GRID_BAG_BOTH;
            constraints.anchor = GRID_BAG_CENTER;
            constraints.gridheight = 1;
        }
        // button panel (BoxLayout Y_AXIS, rigid areas x0_y5 between, vertical glue)
        pnl_buttons.add(&self.btn_move_up.get_component());
        pnl_buttons.add(&self.btn_move_down.get_component());
        pnl_buttons.add(&self.btn_add_row.get_component());
        pnl_buttons.add(&self.btn_delete_row.get_component());
        pnl_buttons.add(&self.btn_copy_row.get_component());
        // border
        let pnl_border = SpacedPanel::get_instance_void();
        pnl_border.set_box_layout(spaced_panel::Y_AXIS);
        pnl_border.add_j_panel(&self.pnl_table);
        // checkbox (rigid areas x40_y0 between)
        self.pnl_flg_remove_duplicates
            .add(&self.cb_flg_remove_duplicates.get_component());
        self.pnl_flg_remove_duplicates
            .add(&self.cb_low_cutoff.get_component());
        self.pnl_flg_remove_duplicates
            .add(&self.cb_flg_strict_search_limits.get_component());
        // table and checkbox (BoxLayout Y_AXIS)
        self.pnl_table_and_checkbox.add(&pnl_border.get_container());
        self.refresh_vertical_padding();
        // root (BoxLayout X_AXIS, rigid areas x3_y0 between)
        self.root_panel.set_border_title(
            EtchedBorder::new(Some(LABEL))
                .get_border()
                .get_title()
                .as_deref(),
        );
        self.root_panel.add(&self.pnl_table_and_checkbox);
        self.root_panel.add(&pnl_buttons);

        ui_harness::with(|harness| harness.pack_base_manager(Some(self.manager)));
    }

    /// Java private `refreshVerticalPadding()`.
    fn refresh_vertical_padding(&self) {
        let size = self.row_list.size();
        let no_padding = 3;
        if self.vertical_rigid_area1.get() {
            self.pnl_table_and_checkbox
                .remove(&self.pnl_flg_remove_duplicates);
        }
        // Swing layout: verticalRigidArea1 is max(0 + (noPadding - size) * 22, 0)
        // high.
        let _height = ((no_padding - size) * 22).max(0);
        self.vertical_rigid_area1.set(true);
        self.pnl_table_and_checkbox
            .add(&self.pnl_flg_remove_duplicates);
    }

    /// `cell.add(pnlTable, layout, constraints)` for a header cell.
    fn add_to_table(&self, cell: &Rc<HeaderCell>) {
        CellVirtual::add(&**cell, &self.pnl_table);
        self.layout
            .set_constraints(&cell.get_component(), &self.constraints.borrow());
    }

    /// Java private `display()`.
    fn display(&self) {
        {
            let mut constraints = self.constraints.borrow_mut();
            constraints.weighty = 0.0;
            // first header row
            constraints.weightx = 0.0;
            constraints.gridwidth = 2;
        }
        self.add_to_table(&self.header1_iteration_number);
        self.constraints.borrow_mut().gridwidth = 6;
        self.add_to_table(&self.header1_d_phi_d_theta_d_psi);
        self.constraints.borrow_mut().gridwidth = 1;
        self.add_to_table(&self.header1_search_radius);
        self.constraints.borrow_mut().gridwidth = 2;
        self.add_to_table(&self.header1_hi_cutoff);
        self.constraints.borrow_mut().gridwidth = 2;
        self.add_to_table(&self.header1_low_cutoff);
        self.constraints.borrow_mut().gridwidth = 1;
        self.add_to_table(&self.header1_ref_threshold);
        self.constraints.borrow_mut().gridwidth = GRID_BAG_REMAINDER;
        self.add_to_table(&self.header1_duplicate_tolerance);

        // Second header row
        {
            let mut constraints = self.constraints.borrow_mut();
            constraints.weightx = 0.0;
            constraints.gridwidth = 2;
        }
        self.add_to_table(&self.header2_iteration_number);
        self.add_to_table(&self.header2_d_phi);
        self.add_to_table(&self.header2_d_theta);
        self.add_to_table(&self.header2_d_psi);
        self.constraints.borrow_mut().gridwidth = 1;
        self.add_to_table(&self.header2_search_radius);
        self.constraints.borrow_mut().gridwidth = 2;
        self.add_to_table(&self.header2_hi_cutoff);
        self.constraints.borrow_mut().gridwidth = 2;
        self.add_to_table(&self.header2_low_cutoff);
        self.constraints.borrow_mut().gridwidth = 1;
        self.add_to_table(&self.header2_ref_threshold);
        self.constraints.borrow_mut().gridwidth = GRID_BAG_REMAINDER;
        self.add_to_table(&self.header2_duplicate_tolerance);

        // Third header row
        {
            let mut constraints = self.constraints.borrow_mut();
            constraints.weightx = 0.0;
            constraints.gridwidth = 2;
        }
        self.add_to_table(&self.header3_iteration_number);
        self.constraints.borrow_mut().gridwidth = 1;
        for cell in [
            &self.header3_d_phi_max,
            &self.header3_d_phi_increment,
            &self.header3_d_theta_max,
            &self.header3_d_theta_increment,
            &self.header3_d_psi_max,
            &self.header3_d_psi_increment,
            &self.header3_search_radius,
            &self.header3_hi_cutoff,
            &self.header3_hi_cutoff_sigma,
            &self.header3_low_cutoff,
            &self.header3_low_cutoff_sigma,
            &self.header3_ref_threshold,
            &self.header3_duplicate_shift_tolerance,
        ] {
            self.add_to_table(cell);
        }
        self.constraints.borrow_mut().gridwidth = GRID_BAG_REMAINDER;
        self.add_to_table(&self.header3_duplicate_angular_tolerance);
        self.row_list.display();
    }

    /// Java package-private `getDPhiDThetaDPsiHeaderCell()`.
    pub fn get_d_phi_d_theta_d_psi_header_cell(&self) -> Rc<HeaderCell> {
        self.header1_d_phi_d_theta_d_psi.clone()
    }

    /// Java package-private `getSearchRadiusHeaderCell()`.
    pub fn get_search_radius_header_cell(&self) -> Rc<HeaderCell> {
        self.header1_search_radius.clone()
    }

    /// Java package-private `getHiCutoffHeaderCell()`.
    pub fn get_hi_cutoff_header_cell(&self) -> Rc<HeaderCell> {
        self.header1_hi_cutoff.clone()
    }

    /// Java package-private `getLowCutoffHeaderCell()`.
    pub fn get_low_cutoff_header_cell(&self) -> Rc<HeaderCell> {
        self.header1_low_cutoff.clone()
    }

    /// Java package-private `getRefThresholdHeaderCell()`.
    pub fn get_ref_threshold_header_cell(&self) -> Rc<HeaderCell> {
        self.header1_ref_threshold.clone()
    }

    /// Java package-private `getDuplicateToleranceHeaderCell()`.
    pub fn get_duplicate_tolerance_header_cell(&self) -> Rc<HeaderCell> {
        self.header1_duplicate_tolerance.clone()
    }

    /// Java package-private `getIterationNumberHeaderCell()`.
    pub fn get_iteration_number_header_cell(&self) -> Rc<HeaderCell> {
        self.header1_iteration_number.clone()
    }

    /// Java private `setVisibleHighPassFilterColumn(boolean)`.
    fn set_visible_high_pass_filter_column(&self, input: bool) {
        self.header1_low_cutoff.get_component().set_visible(input);
        self.header2_low_cutoff.get_component().set_visible(input);
        self.header3_low_cutoff.get_component().set_visible(input);
        self.header3_low_cutoff_sigma
            .get_component()
            .set_visible(input);
        let row_list_size = self.row_list.size();
        for i in 0..row_list_size {
            if let Some(row) = self.row_list.get_row(i) {
                row.set_visible_low_cutoff_rows(input);
            }
        }
    }
}

impl Highlightable for IterationTable {
    /// Java `highlight(boolean)`.
    fn highlight(&self, _highlight: bool) {
        self.update_display_void();
    }
}

impl HighlightableTableVirtual for IterationTable {
    fn highlightable_table(&self) -> &HighlightableTable {
        &self.base
    }

    /// Java `highlightUpActionPerformed()`.
    fn highlight_up_action_performed(&self) {
        self.row_list.highlight_up();
    }

    /// Java `highlightDownActionPerformed()`.
    fn highlight_down_action_performed(&self) {
        self.row_list.highlight_down();
    }

    /// Java `getFocusableParents()`: `{ rootPanel }`.
    fn get_focusable_parents(&self) -> Vec<Option<Rc<JComponent>>> {
        vec![Some(self.root_panel.clone())]
    }
}

/// Java private static final class `RowList`.
struct RowList {
    /// Java private final `list`.
    list: RefCell<Vec<Rc<IterationRow>>>,
    /// Java private final `manager`.
    manager: &'static dyn BaseManager,
    /// Java private final `table`.
    table: Weak<IterationTable>,
}

impl RowList {
    /// Java private `RowList(BaseManager, IterationTable)`.
    fn new(manager: &'static dyn BaseManager, table: Weak<IterationTable>) -> RowList {
        RowList {
            list: RefCell::new(Vec::new()),
            manager,
            table,
        }
    }

    /// The rows, copied out so a row's call may reach the list again.
    fn rows(&self) -> Vec<Rc<IterationRow>> {
        self.list.borrow().clone()
    }

    /// Java private synchronized `add(Highlightable, JPanel, GridBagLayout,
    /// GridBagConstraints, boolean)`.
    fn add(
        &self,
        parent: Weak<dyn Highlightable>,
        panel: &Rc<JComponent>,
        is_low_cutoff: bool,
    ) -> Rc<IterationRow> {
        let index = self.size();
        let row = IterationRow::new(index, parent, panel, self.manager, self.table.clone());
        row.set_visible_low_cutoff_rows(is_low_cutoff);
        row.set_names();
        self.list.borrow_mut().push(row.clone());
        row
    }

    /// Java private `getParameters(MatlabParam, boolean)`.
    fn get_parameters_matlab_param(
        &self,
        matlab_param_file: &mut MatlabParam,
        low_cutoff_active: bool,
    ) {
        matlab_param_file.set_iteration_list_size(self.size());
        for row in self.rows() {
            row.get_parameters_matlab_param(matlab_param_file, low_cutoff_active);
        }
    }

    /// Java private `getParameters(PeetMetaData)`.
    fn get_parameters_meta_data(&self, meta_data: &PeetMetaData) {
        for row in self.rows() {
            row.get_parameters_meta_data(meta_data);
        }
    }

    /// Java private synchronized `delete(IterationRow, Highlightable, JPanel,
    /// GridBagLayout, GridBagConstraints)`.
    fn delete(&self, row: &IterationRow) -> i32 {
        let index = row.get_index();
        self.list.borrow_mut().remove(index as usize);
        for i in index..self.size() {
            if let Some(row) = self.get_row(i) {
                row.set_index(i);
            }
        }
        index
    }

    /// Java private `validateRun(boolean)`.
    fn validate_run(&self, is_low_cutoff: bool) -> bool {
        let rows = self.rows();
        if rows.is_empty() {
            ui_harness::with(|harness| {
                harness.open_message_dialog_base_manager_string_string(
                    Some(self.manager),
                    &format!("Must enter at least one row in {LABEL}"),
                    "Entry Error",
                )
            });
            return false;
        }
        for row in rows {
            if !row.validate_run(is_low_cutoff) {
                return false;
            }
        }
        true
    }

    /// Java private `updateDisplay(boolean, boolean)`.  Update display in all rows.
    fn update_display(&self, sample_sphere: bool, flg_remove_duplicates: bool) {
        for row in self.rows() {
            row.update_display(sample_sphere, flg_remove_duplicates);
        }
    }

    /// Java private `remove()`.
    fn remove(&self) {
        for row in self.rows() {
            row.remove();
        }
    }

    /// Java private synchronized `copy(IterationRow, Highlightable, JPanel,
    /// GridBagLayout, GridBagConstraints, boolean)`.
    fn copy(&self, row: &IterationRow, is_low_cutoff: bool) {
        let index = self.size();
        let copy = IterationRow::new_copy(index, row, self.manager, self.table.clone());
        copy.set_visible_low_cutoff_rows(is_low_cutoff);
        copy.set_names();
        self.list.borrow_mut().push(copy.clone());
        copy.display();
    }

    /// Java private `size()`.
    fn size(&self) -> i32 {
        self.list.borrow().len() as i32
    }

    /// Java private `display()`.
    fn display(&self) {
        for row in self.rows() {
            row.display();
        }
    }

    /// Java private `getRow(int)`.
    fn get_row(&self, index: i32) -> Option<Rc<IterationRow>> {
        if index < 0 || index >= self.size() {
            return None;
        }
        Some(self.list.borrow()[index as usize].clone())
    }

    /// Java private `getHighlightedRow()`.
    fn get_highlighted_row(&self) -> Option<Rc<IterationRow>> {
        for row in self.rows() {
            if row.is_highlighted() {
                return Some(row);
            }
        }
        ui_harness::with(|harness| {
            harness.open_message_dialog_base_manager_string_string(
                Some(self.manager),
                "Please highlight a row.",
                "Entry Error",
            )
        });
        None
    }

    /// Java private `highlightDown()`.
    fn highlight_down(&self) {
        let mut index = self.get_highlight_index();
        if index < 0 {
            return;
        }
        index += 1;
        if index >= self.size() {
            index = 0;
        }
        self.highlight(index);
    }

    /// Java private `highlightUp()`.
    fn highlight_up(&self) {
        let mut index = self.get_highlight_index();
        if index < 0 {
            return;
        }
        index -= 1;
        let size = self.size();
        if index < 0 || index >= size {
            index = size - 1;
        }
        self.highlight(index);
    }

    /// Java private `getHighlightIndex()`.
    fn get_highlight_index(&self) -> i32 {
        for (i, row) in self.rows().iter().enumerate() {
            if row.is_highlighted() {
                return i as i32;
            }
        }
        -1
    }

    /// Java private `highlight(int)`.  Highlight the row in list at rowIndex.
    fn highlight(&self, row_index: i32) {
        if let Some(row) = self.get_row(row_index) {
            row.set_highlighter_selected(true);
        }
    }

    /// Java private `moveRowUp(int)`.  Swap two rows.
    fn move_row_up(&self, row_index: i32) {
        let mut list = self.list.borrow_mut();
        let row_move_up = list.remove(row_index as usize);
        let row_move_down = list.remove(row_index as usize - 1);
        list.insert(row_index as usize - 1, row_move_up);
        list.insert(row_index as usize, row_move_down);
    }

    /// Java private `moveRowDown(int)`.  Swap two rows.
    fn move_row_down(&self, row_index: i32) {
        let mut list = self.list.borrow_mut();
        let row_move_up = list.remove(row_index as usize + 1);
        let row_move_down = list.remove(row_index as usize);
        list.insert(row_index as usize, row_move_up);
        list.insert(row_index as usize + 1, row_move_down);
    }

    /// Java private `reindex(int)`.  Renumber the table starting from the row in the
    /// ArrayList at startIndex.
    fn reindex(&self, start_index: i32) {
        for (i, row) in self
            .rows()
            .iter()
            .enumerate()
            .skip(start_index.max(0) as usize)
        {
            row.set_index(i as i32);
        }
    }
}
