//! `IMOD/Etomo/src/etomo/ui/swing/IterationRow.java`.
//!
//! One row of the PEET dialog's iteration table: the angular search ranges, search
//! distance, low- and high-pass filters, reference threshold and duplicate
//! tolerances of one alignment iteration.  An event dispatch thread object, created
//! as `Rc<Self>`; the table owns it (the row keeps weak references to the table,
//! which is also its highlight group `parent`).  The table's panel, layout and
//! shared constraints are the table's (`IterationTable::with_constraints`).

use std::cell::Cell as StdCell;
use std::rc::{Rc, Weak};

use super::cell::CellVirtual;
use super::field_cell::FieldCell;
use super::header_cell::HeaderCell;
use super::highlightable::Highlightable;
use super::highlighter_button::HighlighterButton;
use super::iteration_table::{self, IterationTable};
use super::ui_harness;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::jdk::{GRID_BAG_REMAINDER, JComponent};
use crate::imod::etomo::logic::converter;
use crate::imod::etomo::storage::autodoc::autodoc_factory;
use crate::imod::etomo::storage::autodoc::read_only_autodoc::ReadOnlyAutodoc;
use crate::imod::etomo::storage::log_file::LogFileError;
use crate::imod::etomo::storage::matlab_param::{self, MatlabParam};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::Type;
use crate::imod::etomo::r#type::const_peet_meta_data::ConstPeetMetaData;
use crate::imod::etomo::r#type::etomo_autodoc;
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::r#type::peet_meta_data::PeetMetaData;
use crate::imod::etomo::ui::shared_strings;

/// Java package-private `final class IterationRow implements Highlightable`.
pub struct IterationRow {
    /// Java private final `number = new HeaderCell()`.
    number: Rc<HeaderCell>,
    /// Java private final `dPhiMax`.
    d_phi_max: Rc<FieldCell>,
    /// Java private final `dPhiIncrement`.
    d_phi_increment: Rc<FieldCell>,
    /// Java private final `dThetaMax`.
    d_theta_max: Rc<FieldCell>,
    /// Java private final `dThetaIncrement`.
    d_theta_increment: Rc<FieldCell>,
    /// Java private final `dPsiMax`.
    d_psi_max: Rc<FieldCell>,
    /// Java private final `dPsiIncrement`.
    d_psi_increment: Rc<FieldCell>,
    /// Java private final `searchRadius`.
    search_radius: Rc<FieldCell>,
    /// Java private final `hiCutoff`.
    hi_cutoff: Rc<FieldCell>,
    /// Java private final `hiCutoffSigma`.
    hi_cutoff_sigma: Rc<FieldCell>,
    /// Java private final `lowCutoff`.
    low_cutoff: Rc<FieldCell>,
    /// Java private final `lowCutoffSigma`.
    low_cutoff_sigma: Rc<FieldCell>,
    /// Java private final `refThreshold`.
    ref_threshold: Rc<FieldCell>,
    /// Java private final `duplicateShiftTolerance`.
    duplicate_shift_tolerance: Rc<FieldCell>,
    /// Java private final `duplicateAngularTolerance`.
    duplicate_angular_tolerance: Rc<FieldCell>,

    /// Java private final `panel` (the table's).
    panel: Rc<JComponent>,
    /// Java private final `btnHighlighter`.
    btn_highlighter: Rc<HighlighterButton>,
    /// Java private final `parent` (the highlight group: the table).
    parent: Weak<dyn Highlightable>,
    /// Java private final `manager`.
    manager: &'static dyn BaseManager,
    /// Java private final `table`.
    table: Weak<IterationTable>,

    /// Java private `index`.
    index: StdCell<i32>,
}

impl IterationRow {
    /// The field initializers and the assignments both Java constructors make.
    fn construct(
        index: i32,
        parent: Weak<dyn Highlightable>,
        panel: &Rc<JComponent>,
        manager: &'static dyn BaseManager,
        table: Weak<IterationTable>,
    ) -> Rc<IterationRow> {
        Rc::new_cyclic(|self_ref: &Weak<IterationRow>| {
            let this: Weak<dyn Highlightable> = self_ref.clone();
            let number = HeaderCell::new_void();
            number.set_text_string(Some(&(index + 1).to_string()));
            IterationRow {
                number,
                d_phi_max: FieldCell::get_editable_matlab_instance(),
                d_phi_increment: FieldCell::get_editable_matlab_instance(),
                d_theta_max: FieldCell::get_editable_matlab_instance(),
                d_theta_increment: FieldCell::get_editable_matlab_instance(),
                d_psi_max: FieldCell::get_editable_matlab_instance(),
                d_psi_increment: FieldCell::get_editable_matlab_instance(),
                search_radius: FieldCell::get_editable_matlab_instance(),
                hi_cutoff: FieldCell::get_editable_matlab_instance(),
                hi_cutoff_sigma: FieldCell::get_editable_matlab_instance(),
                low_cutoff: FieldCell::get_editable_matlab_instance(),
                low_cutoff_sigma: FieldCell::get_editable_matlab_instance(),
                ref_threshold: FieldCell::get_editable_matlab_instance(),
                duplicate_shift_tolerance: FieldCell::get_editable_matlab_instance(),
                duplicate_angular_tolerance: FieldCell::get_editable_matlab_instance(),
                panel: panel.clone(),
                btn_highlighter: HighlighterButton::get_instance(this, Some(parent.clone())),
                parent,
                manager,
                table,
                index: StdCell::new(index),
            }
        })
    }

    /// Java package-private `IterationRow(int, Highlightable, JPanel, GridBagLayout,
    /// GridBagConstraints, BaseManager, IterationTable)`.
    pub fn new(
        index: i32,
        parent: Weak<dyn Highlightable>,
        panel: &Rc<JComponent>,
        manager: &'static dyn BaseManager,
        table: Weak<IterationTable>,
    ) -> Rc<IterationRow> {
        let this = IterationRow::construct(index, parent, panel, manager, table);
        this.set_tooltips();
        this
    }

    /// Java package-private `IterationRow(int, IterationRow, BaseManager,
    /// IterationTable)`.
    pub fn new_copy(
        index: i32,
        iteration_row: &IterationRow,
        manager: &'static dyn BaseManager,
        table: Weak<IterationTable>,
    ) -> Rc<IterationRow> {
        let this = IterationRow::construct(
            index,
            iteration_row.parent.clone(),
            &iteration_row.panel,
            manager,
            table,
        );
        for (to, from) in [
            (&this.d_phi_max, &iteration_row.d_phi_max),
            (&this.d_phi_increment, &iteration_row.d_phi_increment),
            (&this.d_theta_max, &iteration_row.d_theta_max),
            (&this.d_theta_increment, &iteration_row.d_theta_increment),
            (&this.d_psi_max, &iteration_row.d_psi_max),
            (&this.d_psi_increment, &iteration_row.d_psi_increment),
            (&this.search_radius, &iteration_row.search_radius),
            (&this.hi_cutoff, &iteration_row.hi_cutoff),
            (&this.hi_cutoff_sigma, &iteration_row.hi_cutoff_sigma),
            (&this.low_cutoff, &iteration_row.low_cutoff),
            (&this.low_cutoff_sigma, &iteration_row.low_cutoff_sigma),
            (&this.ref_threshold, &iteration_row.ref_threshold),
            (
                &this.duplicate_shift_tolerance,
                &iteration_row.duplicate_shift_tolerance,
            ),
            (
                &this.duplicate_angular_tolerance,
                &iteration_row.duplicate_angular_tolerance,
            ),
        ] {
            to.set_value_string(from.get_value().as_deref());
        }
        this.set_tooltips();
        this
    }

    /// Java field read `table`.
    fn table(&self) -> Rc<IterationTable> {
        self.table
            .upgrade()
            .expect("the iteration table owns its rows")
    }

    /// Java package-private `setNames()`.
    pub fn set_names(&self) {
        let table = self.table();
        let label = Some(iteration_table::LABEL);
        self.btn_highlighter.set_headers(
            label,
            &self.number,
            &table.get_iteration_number_header_cell(),
        );
        let angles = table.get_d_phi_d_theta_d_psi_header_cell();
        self.d_phi_max.set_headers(label, &self.number, &angles);
        self.d_phi_increment
            .set_headers(label, &self.number, &angles);
        self.d_theta_max.set_headers(label, &self.number, &angles);
        self.d_theta_increment
            .set_headers(label, &self.number, &angles);
        self.d_psi_max.set_headers(label, &self.number, &angles);
        self.d_psi_increment
            .set_headers(label, &self.number, &angles);
        self.search_radius
            .set_headers(label, &self.number, &table.get_search_radius_header_cell());
        self.hi_cutoff
            .set_headers(label, &self.number, &table.get_hi_cutoff_header_cell());
        self.hi_cutoff_sigma
            .set_headers(label, &self.number, &table.get_hi_cutoff_header_cell());
        self.low_cutoff
            .set_headers(label, &self.number, &table.get_low_cutoff_header_cell());
        self.low_cutoff_sigma
            .set_headers(label, &self.number, &table.get_low_cutoff_header_cell());
        self.ref_threshold
            .set_headers(label, &self.number, &table.get_ref_threshold_header_cell());
        self.duplicate_shift_tolerance.set_headers(
            label,
            &self.number,
            &table.get_duplicate_tolerance_header_cell(),
        );
        self.duplicate_angular_tolerance.set_headers(
            label,
            &self.number,
            &table.get_duplicate_tolerance_header_cell(),
        );
    }

    /// Java package-private `setHighlighterSelected(boolean)`.
    pub fn set_highlighter_selected(&self, select: bool) {
        self.btn_highlighter.set_selected(select);
    }

    /// Java package-private `updateDisplay(boolean, boolean)`.  In the first row, turn
    /// off theta and psi when sampleSphere is on.  In all rows turn off duplicates
    /// columns when flgRemoveDuplicates is off.
    pub fn update_display(&self, sample_sphere: bool, flg_remove_duplicates: bool) {
        if self.get_index() == 0 {
            self.d_theta_max.set_enabled(!sample_sphere);
            self.d_theta_increment.set_enabled(!sample_sphere);
            self.d_psi_max.set_enabled(!sample_sphere);
            self.d_psi_increment.set_enabled(!sample_sphere);
        }
        self.duplicate_shift_tolerance
            .set_enabled(flg_remove_duplicates);
        self.duplicate_angular_tolerance
            .set_enabled(flg_remove_duplicates);
    }

    /// Java package-private `getParameters(MatlabParam, boolean)`.
    pub fn get_parameters_matlab_param(
        &self,
        matlab_param_file: &mut MatlabParam,
        low_cutoff_active: bool,
    ) {
        let iteration = matlab_param_file.get_iteration(self.index.get());
        if !self.d_phi_max.is_empty() || !self.d_phi_increment.is_empty() {
            iteration.set_d_phi_end(self.d_phi_max.get_value().as_deref());
            iteration.set_d_phi_increment(self.d_phi_increment.get_value().as_deref());
        } else {
            iteration.clear_d_phi();
        }
        if !self.d_theta_max.is_empty() || !self.d_theta_increment.is_empty() {
            iteration.set_d_theta_end(self.d_theta_max.get_value().as_deref());
            iteration.set_d_theta_increment(self.d_theta_increment.get_value().as_deref());
        } else {
            iteration.clear_d_theta();
        }
        if !self.d_psi_max.is_empty() || !self.d_psi_increment.is_empty() {
            iteration.set_d_psi_end(self.d_psi_max.get_value().as_deref());
            iteration.set_d_psi_increment(self.d_psi_increment.get_value().as_deref());
        } else {
            iteration.clear_d_psi();
        }
        iteration.set_search_radius(self.search_radius.get_value().as_deref());
        iteration.set_hi_cutoff_cutoff(self.hi_cutoff.get_value().as_deref());
        iteration.set_hi_cutoff_sigma(self.hi_cutoff_sigma.get_value().as_deref());
        if low_cutoff_active {
            iteration.set_low_cutoff_cutoff(self.low_cutoff.get_value().as_deref());
            iteration.set_low_cutoff_sigma(self.low_cutoff_sigma.get_value().as_deref());
        } else {
            iteration.set_low_cutoff_cutoff(Some(matlab_param::LOW_CUTOFF_DEFAULT));
            iteration.set_low_cutoff_sigma(Some(matlab_param::LOW_CUTOFF_SIGMA_DEFAULT));
        }
        iteration.set_ref_threshold(self.ref_threshold.get_value().as_deref());
        iteration
            .set_duplicate_shift_tolerance(self.duplicate_shift_tolerance.get_value().as_deref());
        iteration.set_duplicate_angular_tolerance(
            self.duplicate_angular_tolerance.get_value().as_deref(),
        );
    }

    /// Java package-private `getParameters(PeetMetaData)`.
    pub fn get_parameters_meta_data(&self, meta_data: &PeetMetaData) {
        let index = self.index.get();
        meta_data.set_low_cutoff_cutoff(self.low_cutoff.get_value().as_deref(), index);
        meta_data.set_low_cutoff_sigma(self.low_cutoff_sigma.get_value().as_deref(), index);
    }

    /// Java package-private `setParameters(ConstPeetMetaData)`.
    pub fn set_parameters_meta_data(&self, meta_data: Option<&dyn ConstPeetMetaData>) {
        let Some(meta_data) = meta_data else {
            return;
        };
        let index = self.index.get();
        self.set_low_cutoff_cutoff(meta_data.get_low_cutoff_cutoff(index).as_deref());
        self.set_low_cutoff_sigma(meta_data.get_low_cutoff_sigma(index).as_deref());
    }

    /// Java package-private `checkLowCutoffBackwardsCompatibility(MatlabParam)`.
    pub fn check_low_cutoff_backwards_compatibility(
        &self,
        matlab_param_file: &mut MatlabParam,
    ) -> bool {
        let iteration = matlab_param_file.get_iteration(self.index.get());
        let low_cutoff = converter::to_double(iteration.get_low_cutoff_cutoff().as_deref());
        let low_cutoff_sigma = converter::to_double(iteration.get_low_cutoff_sigma().as_deref());
        // `Double.equals`: bit-wise equality of the two doubles.
        match low_cutoff {
            Some(low_cutoff)
                if low_cutoff.to_bits() == matlab_param::DOUBLE_LOW_CUTOFF_DEFAULT.to_bits() => {}
            _ => return false,
        }
        match low_cutoff_sigma {
            Some(low_cutoff_sigma)
                if low_cutoff_sigma.to_bits()
                    == matlab_param::DOUBLE_LOW_CUTOFF_SIGMA_DEFAULT.to_bits() => {}
            _ => return false,
        }
        true
    }

    /// Java package-private `setParameters(MatlabParam, boolean)`.
    pub fn set_parameters_matlab_param(
        &self,
        matlab_param_file: &mut MatlabParam,
        is_low_cutoff: bool,
    ) {
        let iteration = matlab_param_file.get_iteration(self.index.get());
        self.d_phi_max
            .set_value_string(iteration.get_d_phi_end().as_deref());
        self.d_phi_increment
            .set_value_string(iteration.get_d_phi_increment().as_deref());
        self.d_theta_max
            .set_value_string(iteration.get_d_theta_end().as_deref());
        self.d_theta_increment
            .set_value_string(iteration.get_d_theta_increment().as_deref());
        self.d_psi_max
            .set_value_string(iteration.get_d_psi_end().as_deref());
        self.d_psi_increment
            .set_value_string(iteration.get_d_psi_increment().as_deref());
        self.search_radius
            .set_value_string(iteration.get_search_radius_string().as_deref());
        self.hi_cutoff
            .set_value_string(iteration.get_hi_cutoff_cutoff().as_deref());
        self.hi_cutoff_sigma
            .set_value_string(iteration.get_hi_cutoff_sigma().as_deref());
        if is_low_cutoff {
            self.low_cutoff
                .set_value_string(iteration.get_low_cutoff_cutoff().as_deref());
            self.low_cutoff_sigma
                .set_value_string(iteration.get_low_cutoff_sigma().as_deref());
        }
        if self.low_cutoff_sigma.is_empty() {
            self.set_low_cutoff_sigma(Some(matlab_param::LOW_CUTOFF_SIGMA_DEFAULT));
        }
        self.ref_threshold
            .set_value_string(iteration.get_ref_threshold_string().as_deref());
        self.duplicate_shift_tolerance
            .set_value_string(iteration.get_duplicate_shift_tolerance_string().as_deref());
        self.duplicate_angular_tolerance.set_value_string(
            iteration
                .get_duplicate_angular_tolerance_string()
                .as_deref(),
        );
    }

    /// Java package-private `isHighlighted()`.
    pub fn is_highlighted(&self) -> bool {
        self.btn_highlighter.is_highlighted()
    }

    /// Java package-private `getIndex()`.
    pub fn get_index(&self) -> i32 {
        self.index.get()
    }

    /// Java package-private `setIndex(int)`.
    pub fn set_index(&self, index: i32) {
        self.index.set(index);
        self.number.set_text_string(Some(&(index + 1).to_string()));
    }

    /// Java package-private `remove()`.
    pub fn remove(&self) {
        self.number.remove();
        self.btn_highlighter.remove();
        self.d_phi_max.remove();
        self.d_phi_increment.remove();
        self.d_theta_max.remove();
        self.d_theta_increment.remove();
        self.d_psi_max.remove();
        self.d_psi_increment.remove();
        self.search_radius.remove();
        self.hi_cutoff.remove();
        self.hi_cutoff_sigma.remove();
        self.low_cutoff.remove();
        self.low_cutoff_sigma.remove();
        self.ref_threshold.remove();
        self.duplicate_shift_tolerance.remove();
        self.duplicate_angular_tolerance.remove();
    }

    /// Java package-private `display()`.
    pub fn display(&self) {
        let table = self.table();
        let panel = &self.panel;
        // `cell.add(panel, layout, constraints)`: the cell's `add` and the
        // `layout.setConstraints` it makes.
        let add = |cell: &dyn CellVirtual, component: Rc<JComponent>| {
            cell.add(panel);
            table
                .get_layout()
                .set_constraints(&component, &table.get_constraints());
        };
        table.with_constraints(|constraints| {
            constraints.weightx = 0.0;
            constraints.weighty = 0.1;
            constraints.gridwidth = 1;
        });
        add(&*self.number, self.number.get_component());
        table.with_constraints(|constraints| {
            self.btn_highlighter
                .add(panel, table.get_layout(), constraints);
        });
        table.with_constraints(|constraints| constraints.weightx = 0.1);
        for cell in [
            &self.d_phi_max,
            &self.d_phi_increment,
            &self.d_theta_max,
            &self.d_theta_increment,
            &self.d_psi_max,
            &self.d_psi_increment,
            &self.search_radius,
            &self.hi_cutoff,
            &self.hi_cutoff_sigma,
            &self.low_cutoff,
            &self.low_cutoff_sigma,
            &self.ref_threshold,
            &self.duplicate_shift_tolerance,
        ] {
            add(&**cell, cell.get_component());
        }
        table.with_constraints(|constraints| constraints.gridwidth = GRID_BAG_REMAINDER);
        add(
            &*self.duplicate_angular_tolerance,
            self.duplicate_angular_tolerance.get_component(),
        );
    }

    /// Java private `buildHeaderDescription(String[])`.
    fn build_header_description(header_array: Option<&[&str]>) -> String {
        let mut header = String::new();
        if let Some(header_array) = header_array {
            for element in header_array {
                header.push_str(&format!(", {element}"));
            }
        }
        header
    }

    /// Java private `validateRun(boolean, EtomoNumber, String[], String)`.  Validates
    /// empty and n.  Empty must always be false.  If n is not null, it must be valid
    /// and not negative.
    fn validate_run_value(
        &self,
        empty: bool,
        n: Option<&EtomoNumber>,
        header_array: &[&str],
        additional_empty_error_message: Option<&str>,
    ) -> bool {
        if empty {
            ui_harness::with(|harness| {
                harness.open_message_dialog_base_manager_string_string(
                    Some(self.manager),
                    &format!(
                        "{}:  In row {}{} must not be empty.{}",
                        iteration_table::LABEL,
                        self.number,
                        Self::build_header_description(Some(header_array)),
                        additional_empty_error_message.unwrap_or("")
                    ),
                    "Entry Error",
                )
            });
            return false;
        }
        if let Some(n) = n {
            if !n.is_valid() {
                ui_harness::with(|harness| {
                    harness.open_message_dialog_base_manager_string_string(
                        Some(self.manager),
                        &format!(
                            "{}:  In row {}{}:   {}",
                            iteration_table::LABEL,
                            self.number,
                            Self::build_header_description(Some(header_array)),
                            n.get_invalid_reason()
                        ),
                        "Entry Error",
                    )
                });
                return false;
            }
            if n.is_negative() {
                ui_harness::with(|harness| {
                    harness.open_message_dialog_base_manager_string_string(
                        Some(self.manager),
                        &format!(
                            "{}:  In row {}{} must not be negative.",
                            iteration_table::LABEL,
                            self.number.get_text().unwrap_or_default(),
                            Self::build_header_description(Some(header_array))
                        ),
                        "Entry Error",
                    )
                });
                return false;
            }
        }
        true
    }

    /// Java package-private `validateRun(boolean)`.
    pub fn validate_run(&self, is_low_cutoff: bool) -> bool {
        // Phi
        let mut n_double = EtomoNumber::new_with_type(Some(Type::Double));
        let mut n_integer = EtomoNumber::new();
        n_double.set_string(self.d_phi_max.get_value().as_deref());
        if !self.validate_run_value(
            self.d_phi_max.is_empty(),
            Some(&n_double),
            &[
                iteration_table::D_PHI_D_THETA_D_PSI_HEADER1,
                shared_strings::D_PHI_LABEL,
                iteration_table::MAX_HEADER3,
            ],
            Some("Use 0 to not search on the angle."),
        ) {
            return false;
        }
        n_double.set_string(self.d_phi_increment.get_value().as_deref());
        if !self.validate_run_value(
            self.d_phi_increment.is_empty(),
            Some(&n_double),
            &[
                iteration_table::D_PHI_D_THETA_D_PSI_HEADER1,
                shared_strings::D_PHI_LABEL,
                iteration_table::INCR_HEADER3,
            ],
            None,
        ) {
            return false;
        }
        // Theta
        if self.d_theta_max.is_enabled() {
            n_double.set_string(self.d_theta_max.get_value().as_deref());
            if !self.validate_run_value(
                self.d_theta_max.is_empty(),
                Some(&n_double),
                &[
                    iteration_table::D_PHI_D_THETA_D_PSI_HEADER1,
                    shared_strings::D_THETA_LABEL,
                    iteration_table::MAX_HEADER3,
                ],
                Some("Use 0 to not search on the angle."),
            ) {
                return false;
            }
        }
        if self.d_theta_increment.is_enabled() {
            n_double.set_string(self.d_theta_increment.get_value().as_deref());
            if !self.validate_run_value(
                self.d_theta_increment.is_empty(),
                Some(&n_double),
                &[
                    iteration_table::D_PHI_D_THETA_D_PSI_HEADER1,
                    shared_strings::D_THETA_LABEL,
                    iteration_table::INCR_HEADER3,
                ],
                None,
            ) {
                return false;
            }
        }
        // Psi
        if self.d_psi_max.is_enabled() {
            n_double.set_string(self.d_psi_max.get_value().as_deref());
            if !self.validate_run_value(
                self.d_psi_max.is_empty(),
                Some(&n_double),
                &[
                    iteration_table::D_PHI_D_THETA_D_PSI_HEADER1,
                    shared_strings::D_PSI_LABEL,
                    iteration_table::MAX_HEADER3,
                ],
                Some("Use 0 to not search on the angle."),
            ) {
                return false;
            }
        }
        if self.d_psi_increment.is_enabled() {
            n_double.set_string(self.d_psi_increment.get_value().as_deref());
            if !self.validate_run_value(
                self.d_psi_increment.is_empty(),
                Some(&n_double),
                &[
                    iteration_table::D_PHI_D_THETA_D_PSI_HEADER1,
                    shared_strings::D_PSI_LABEL,
                    iteration_table::INCR_HEADER3,
                ],
                None,
            ) {
                return false;
            }
        }
        // search radius
        let search_radius_string = self
            .search_radius
            .get_value()
            .unwrap_or_default()
            .trim_matches(|c: char| (c as u32) <= 0x20)
            .to_owned();
        let header_array = [
            iteration_table::SEARCH_RADIUS_HEADER1,
            iteration_table::SEARCH_RADIUS_HEADER2,
        ];
        n_integer.set_string(Some(&search_radius_string));
        if self.search_radius.is_empty() || n_integer.is_valid() {
            if !self.validate_run_value(
                self.search_radius.is_empty(),
                Some(&n_integer),
                &header_array,
                None,
            ) {
                return false;
            }
        } else {
            // If its not a single number then it must be a list of three numbers
            // divided by "," or " ".
            // `split("\\s*,\\s*")` (trailing empty strings removed).
            let mut search_radius_array: Vec<String> = search_radius_string
                .split(',')
                .map(|element| element.trim().to_owned())
                .collect();
            while search_radius_array.len() > 1
                && search_radius_array.last().is_some_and(String::is_empty)
            {
                search_radius_array.pop();
            }
            if search_radius_array.len() != 3 {
                // `split("\\s+")`.
                search_radius_array = search_radius_string
                    .split(char::is_whitespace)
                    .filter(|element| !element.is_empty())
                    .map(str::to_owned)
                    .collect();
                if search_radius_array.len() != 3 {
                    ui_harness::with(|harness| {
                        harness.open_message_dialog_base_manager_string_string(
                            Some(self.manager),
                            &format!(
                                "{}:  In row {}{} must have either 1 or 3 elements.",
                                iteration_table::LABEL,
                                self.number,
                                Self::build_header_description(Some(&header_array))
                            ),
                            "Entry Error",
                        )
                    });
                    return false;
                }
            }
            // Validate each number in the array.
            for element in &search_radius_array {
                n_integer.set_string(Some(element));
                if !self.validate_run_value(false, Some(&n_integer), &header_array, None) {
                    return false;
                }
            }
        }
        // hiCutoff
        if !self.validate_run_value(
            self.hi_cutoff.is_empty(),
            None,
            &[
                iteration_table::HICUTOFF_HEADER1,
                iteration_table::HICUTOFF_HEADER2,
                iteration_table::HICUTOFF_CUTOFF_HEADER3,
            ],
            None,
        ) {
            return false;
        }
        // hiCutoffSigma
        if !self.validate_run_value(
            self.hi_cutoff_sigma.is_empty(),
            None,
            &[
                iteration_table::HICUTOFF_HEADER1,
                iteration_table::HICUTOFF_HEADER2,
                iteration_table::HICUTOFF_SIGMA_HEADER3,
            ],
            None,
        ) {
            return false;
        }
        if is_low_cutoff {
            // lowCutoff
            if !self.validate_run_value(
                self.low_cutoff.is_empty(),
                None,
                &[
                    iteration_table::LOWCUTOFF_HEADER1,
                    iteration_table::LOWCUTOFF_HEADER2,
                    iteration_table::LOWCUTOFF_CUTOFF_HEADER3,
                ],
                Some(" Use 0 to disable filtering for an iteration."),
            ) {
                return false;
            }
            // lowCutoffSigma
            if !self.validate_run_value(
                self.low_cutoff_sigma.is_empty(),
                None,
                &[
                    iteration_table::LOWCUTOFF_HEADER1,
                    iteration_table::LOWCUTOFF_HEADER2,
                    iteration_table::LOWCUTOFF_SIGMA_HEADER3,
                ],
                None,
            ) {
                return false;
            }
        }
        // refThreshold
        if !self.validate_run_value(
            self.ref_threshold.is_empty(),
            None,
            &[
                iteration_table::REF_THRESHOLD_HEADER1,
                iteration_table::REF_THRESHOLD_HEADER2,
            ],
            None,
        ) {
            return false;
        }
        // duplicateShiftTolerance
        if self.duplicate_shift_tolerance.is_enabled() {
            n_integer.set_string(self.duplicate_shift_tolerance.get_value().as_deref());
            if !self.validate_run_value(
                self.duplicate_shift_tolerance.is_empty(),
                Some(&n_integer),
                &[
                    iteration_table::DUPLICATE_TOLERANCE_HEADER1,
                    iteration_table::DUPLICATE_TOLERANCE_HEADER2,
                    iteration_table::DUPLICATE_SHIFT_TOLERANCE_HEADER3,
                ],
                None,
            ) {
                return false;
            }
        }
        // duplicateAngularTolerance
        if self.duplicate_angular_tolerance.is_enabled() {
            n_integer.set_string(self.duplicate_angular_tolerance.get_value().as_deref());
            if !self.validate_run_value(
                self.duplicate_angular_tolerance.is_empty(),
                Some(&n_integer),
                &[
                    iteration_table::DUPLICATE_TOLERANCE_HEADER1,
                    iteration_table::DUPLICATE_TOLERANCE_HEADER2,
                    iteration_table::DUPLICATE_ANGULAR_TOLERANCE_HEADER3,
                ],
                None,
            ) {
                return false;
            }
        }
        true
    }

    /// Java private `setTooltips()`.
    fn set_tooltips(&self) {
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
        self.duplicate_shift_tolerance.set_tool_tip_text(
            etomo_autodoc::get_tooltip(autodoc, Some(matlab_param::DUPLICATE_SHIFT_TOLERANCE_KEY))
                .as_deref(),
        );
        self.duplicate_angular_tolerance.set_tool_tip_text(
            etomo_autodoc::get_tooltip(
                autodoc,
                Some(matlab_param::DUPLICATE_ANGULAR_TOLERANCE_KEY),
            )
            .as_deref(),
        );
        self.number.set_tool_tip_text(Some("Iteration number"));
        self.d_phi_max.set_tool_tip_text(Some(
            "Maximum magnitude of rotation about the particle Y axis in degrees.  Search will range from -(Phi Max) to +(Phi Max) in steps of (Phi Step).",
        ));
        self.d_phi_increment.set_tool_tip_text(Some(
            "Increment between sample points for rotation about Y in degrees.  Search will range from -(Phi Max) to +(Phi Max) in steps of (Phi Step).",
        ));
        self.d_theta_max.set_tool_tip_text(Some(
            "Maximum magnitude of rotation about the particle Z axis in degrees.  Search will range from -(Theta Max) to +(Theta Max) in steps of (Theta Step).",
        ));
        self.d_theta_increment.set_tool_tip_text(Some(
            "Increment between sample points for rotation about Z in degrees.  Search will range from -(Theta Max) to +(Theta Max) in steps of (Theta Step).",
        ));
        self.d_psi_max.set_tool_tip_text(Some(
            "Maximum magnitude of rotation about the particle X axis in degrees.  Search will range from -(Psi Max) to +(Psi Max) in steps of (Psi Step).",
        ));
        self.d_psi_increment.set_tool_tip_text(Some(
            "Increment between sample points for rotation about X in degrees.  Search will range from -(Psi Max) to +(Psi Max) in steps of (Psi Step).",
        ));
        self.search_radius.set_tool_tip_text(Some(
            "The number of pixels to search in the X, Y, and Z directions.  A single, integer number of pixels can be specified, which will be applied to all 3 dimensions, or a vector of 3 integers can be specified, giving the X, Y, and Z search distances individually. E.g. '3' is equivalent to '3 3 3'.",
        ));
        self.hi_cutoff.set_tool_tip_text(Some(
            "The normalized spatial frequency above which high frequencies are attenuated.  0.5 corresponds to the Nyquist frequency, and values of 0.866 or larger disable low-pass filtering.",
        ));
        self.hi_cutoff_sigma.set_tool_tip_text(Some(
            "The width (standard deviation) in normalized frequency units of a Gaussian determining the rate at which attenuation increases above the cutoff.",
        ));
        self.low_cutoff.set_tool_tip_text(Some(
            "The normalized frequency below which low frequencies will be attenuated. Values <= 0 disable high-pass filtering.",
        ));
        self.low_cutoff_sigma.set_tool_tip_text(Some(
            "An optional parameter which defines the transition width of the high-pass filter.",
        ));
        self.ref_threshold.set_tool_tip_text(Some(
            "Determines the number of particles averaged to form the reference for the next alignment iteration. If less than 1, it represents a cross-correlation coefficient threshold, with particles having a larger correlation eligible for inclusion in the reference.  If greater than 1, it is the number of particles to include.",
        ));
    }

    /// Java `setVisibleLowCutoffRows(boolean)`.
    pub fn set_visible_low_cutoff_rows(&self, input: bool) {
        self.low_cutoff.get_component().set_visible(input);
        self.low_cutoff_sigma.get_component().set_visible(input);
    }

    /// Java `setLowCutoffCutoff(String)`.
    pub fn set_low_cutoff_cutoff(&self, input: Option<&str>) {
        let Some(input) = input else {
            return;
        };
        self.low_cutoff.set_value_string(Some(input));
    }

    /// Java `setLowCutoffSigma(String)`.
    pub fn set_low_cutoff_sigma(&self, input: Option<&str>) {
        let Some(input) = input else {
            return;
        };
        self.low_cutoff_sigma.set_value_string(Some(input));
    }
}

impl Highlightable for IterationRow {
    /// Java `highlight(boolean)`.
    fn highlight(&self, highlight: bool) {
        for cell in [
            &self.d_phi_max,
            &self.d_phi_increment,
            &self.d_theta_max,
            &self.d_theta_increment,
            &self.d_psi_max,
            &self.d_psi_increment,
            &self.search_radius,
            &self.hi_cutoff,
            &self.hi_cutoff_sigma,
            &self.low_cutoff,
            &self.low_cutoff_sigma,
            &self.ref_threshold,
            &self.duplicate_shift_tolerance,
            &self.duplicate_angular_tolerance,
        ] {
            cell.set_highlight(highlight);
        }
    }
}
