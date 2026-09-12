//! `IMOD/Etomo/src/etomo/ui/swing/IterationRow.java`.
//!
//! Swing panel/grid placement and the manager's autodoc and message-dialog
//! services remain frontend boundaries.  This module retains every row field,
//! its source validation rules, row highlighting, and parameter transfer.
#![allow(dead_code)]

use super::field_cell::FieldCell;
use super::header_cell::HeaderCell;
use super::highlighter_button::HighlighterButton;
use super::iteration_table::{
    ConstPeetMetaData, D_PHI_D_THETA_D_PSI_HEADER1, DUPLICATE_ANGULAR_TOLERANCE_HEADER3,
    DUPLICATE_SHIFT_TOLERANCE_HEADER3, DUPLICATE_TOLERANCE_HEADER1, DUPLICATE_TOLERANCE_HEADER2,
    HICUTOFF_CUTOFF_HEADER3, HICUTOFF_HEADER1, HICUTOFF_HEADER2, HICUTOFF_SIGMA_HEADER3,
    INCR_HEADER3, Iteration, LABEL, LOW_CUTOFF_DEFAULT, LOW_CUTOFF_SIGMA_DEFAULT,
    LOWCUTOFF_CUTOFF_HEADER3, LOWCUTOFF_HEADER1, LOWCUTOFF_HEADER2, LOWCUTOFF_SIGMA_HEADER3,
    MAX_HEADER3, MatlabParam, PeetMetaData, REF_THRESHOLD_HEADER1, REF_THRESHOLD_HEADER2,
    SEARCH_RADIUS_HEADER1, SEARCH_RADIUS_HEADER2,
};

/// Java package-private final `IterationRow`.
#[derive(Clone, Debug)]
pub struct IterationRow {
    pub number: HeaderCell,
    pub d_phi_max: FieldCell,
    pub d_phi_increment: FieldCell,
    pub d_theta_max: FieldCell,
    pub d_theta_increment: FieldCell,
    pub d_psi_max: FieldCell,
    pub d_psi_increment: FieldCell,
    pub search_radius: FieldCell,
    pub hi_cutoff: FieldCell,
    pub hi_cutoff_sigma: FieldCell,
    pub low_cutoff: FieldCell,
    pub low_cutoff_sigma: FieldCell,
    pub ref_threshold: FieldCell,
    pub duplicate_shift_tolerance: FieldCell,
    pub duplicate_angular_tolerance: FieldCell,
    pub btn_highlighter: HighlighterButton,
    pub index: usize,
    pub displayed: bool,
    pub last_message: Option<(String, String)>,
}

impl IterationRow {
    /// Java `IterationRow(int, Highlightable, JPanel, GridBagLayout,
    /// GridBagConstraints, BaseManager, IterationTable)`.
    pub fn new(index: usize, _is_low_cutoff: bool) -> Self {
        let mut number = HeaderCell::default();
        number.set_text_int((index + 1) as i32);
        let mut value = Self {
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
            btn_highlighter: HighlighterButton::get_instance(index, None),
            index,
            displayed: false,
            last_message: None,
        };
        value.set_tooltips();
        value
    }

    /// Java copy constructor `IterationRow(int, IterationRow, BaseManager,
    /// IterationTable)`.
    pub fn copy(index: usize, iteration_row: &Self) -> Self {
        let mut value = iteration_row.clone();
        value.index = index;
        value.number.set_text_int((index + 1) as i32);
        value.btn_highlighter = HighlighterButton::get_instance(index, None);
        value.displayed = false;
        value.last_message = None;
        value.set_tooltips();
        value
    }

    /// Java `setNames()`; table header identities are supplied by the caller's
    /// canonical header text because the Swing table is a frontend boundary.
    pub fn set_names(&mut self) {
        let row = self.number.to_string().to_owned();
        self.btn_highlighter.set_headers(LABEL, &row, "Run #");
        for (field, column) in [
            (&mut self.d_phi_max, "Angular Search Range, Phi, Max"),
            (&mut self.d_phi_increment, "Angular Search Range, Phi, Step"),
            (&mut self.d_theta_max, "Angular Search Range, Theta, Max"),
            (
                &mut self.d_theta_increment,
                "Angular Search Range, Theta, Step",
            ),
            (&mut self.d_psi_max, "Angular Search Range, Psi, Max"),
            (&mut self.d_psi_increment, "Angular Search Range, Psi, Step"),
            (&mut self.search_radius, "Search, Distance"),
            (&mut self.hi_cutoff, "Low-pass, Filter, Cutoff"),
            (&mut self.hi_cutoff_sigma, "Low-pass, Filter, Sigma"),
            (&mut self.low_cutoff, "High-pass, Filter, Cutoff"),
            (&mut self.low_cutoff_sigma, "High-pass, Filter, Sigma"),
            (&mut self.ref_threshold, "Ref, Threshold"),
            (
                &mut self.duplicate_shift_tolerance,
                "Duplicate, Tolerance, Shift",
            ),
            (
                &mut self.duplicate_angular_tolerance,
                "Duplicate, Tolerance, Angle",
            ),
        ] {
            field.set_name_three(Some(LABEL), Some(&row), Some(column));
        }
    }

    /// Java `highlight(boolean)`.
    pub fn highlight(&mut self, highlight: bool) {
        for field in self.fields_mut() {
            field.set_field_highlight(highlight);
        }
    }

    /// Java `setHighlighterSelected(boolean)`.
    pub fn set_highlighter_selected(&mut self, select: bool) {
        self.btn_highlighter.set_selected(select);
    }

    /// Java `updateDisplay(boolean, boolean)`.
    pub fn update_display(&mut self, sample_sphere: bool, flg_remove_duplicates: bool) {
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

    /// Java `getParameters(MatlabParam, boolean)`.
    pub fn get_parameters_matlab(&self, iteration: &mut Iteration, low_cutoff_active: bool) {
        if !self.d_phi_max.is_empty() || !self.d_phi_increment.is_empty() {
            iteration
                .values
                .insert("dPhiEnd".into(), self.d_phi_max.get_value().into());
            iteration.values.insert(
                "dPhiIncrement".into(),
                self.d_phi_increment.get_value().into(),
            );
        } else {
            iteration.values.remove("dPhiEnd");
            iteration.values.remove("dPhiIncrement");
        }
        if !self.d_theta_max.is_empty() || !self.d_theta_increment.is_empty() {
            iteration
                .values
                .insert("dThetaEnd".into(), self.d_theta_max.get_value().into());
            iteration.values.insert(
                "dThetaIncrement".into(),
                self.d_theta_increment.get_value().into(),
            );
        } else {
            iteration.values.remove("dThetaEnd");
            iteration.values.remove("dThetaIncrement");
        }
        if !self.d_psi_max.is_empty() || !self.d_psi_increment.is_empty() {
            iteration
                .values
                .insert("dPsiEnd".into(), self.d_psi_max.get_value().into());
            iteration.values.insert(
                "dPsiIncrement".into(),
                self.d_psi_increment.get_value().into(),
            );
        } else {
            iteration.values.remove("dPsiEnd");
            iteration.values.remove("dPsiIncrement");
        }
        iteration
            .values
            .insert("searchRadius".into(), self.search_radius.get_value().into());
        iteration
            .values
            .insert("hiCutoffCutoff".into(), self.hi_cutoff.get_value().into());
        iteration.values.insert(
            "hiCutoffSigma".into(),
            self.hi_cutoff_sigma.get_value().into(),
        );
        iteration.values.insert(
            "lowCutoffCutoff".into(),
            if low_cutoff_active {
                self.low_cutoff.get_value()
            } else {
                LOW_CUTOFF_DEFAULT
            }
            .into(),
        );
        iteration.values.insert(
            "lowCutoffSigma".into(),
            if low_cutoff_active {
                self.low_cutoff_sigma.get_value()
            } else {
                LOW_CUTOFF_SIGMA_DEFAULT
            }
            .into(),
        );
        iteration
            .values
            .insert("refThreshold".into(), self.ref_threshold.get_value().into());
        iteration.values.insert(
            "duplicateShiftTolerance".into(),
            self.duplicate_shift_tolerance.get_value().into(),
        );
        iteration.values.insert(
            "duplicateAngularTolerance".into(),
            self.duplicate_angular_tolerance.get_value().into(),
        );
    }

    /// Java `getParameters(PeetMetaData)`.
    pub fn get_parameters_peet(&self, meta_data: &mut PeetMetaData) {
        if meta_data.low_cutoff_values.len() <= self.index {
            meta_data
                .low_cutoff_values
                .resize_with(self.index + 1, Default::default);
        }
        let values = &mut meta_data.low_cutoff_values[self.index];
        values.insert("lowCutoffCutoff".into(), self.low_cutoff.get_value().into());
        values.insert(
            "lowCutoffSigma".into(),
            self.low_cutoff_sigma.get_value().into(),
        );
    }

    /// Java `setParameters(ConstPeetMetaData)`.
    pub fn set_parameters_peet(&mut self, meta_data: Option<&ConstPeetMetaData>) {
        let Some(meta_data) = meta_data else { return };
        let Some(values) = meta_data.low_cutoff_values.get(self.index) else {
            return;
        };
        self.set_low_cutoff_cutoff(values.get("lowCutoffCutoff").map(String::as_str));
        self.set_low_cutoff_sigma(values.get("lowCutoffSigma").map(String::as_str));
    }

    /// Java `checkLowCutoffBackwardsCompatibility(MatlabParam)`.
    pub fn check_low_cutoff_backwards_compatibility(&self, matlab: &MatlabParam) -> bool {
        let Some(iteration) = matlab.iterations.get(self.index) else {
            return false;
        };
        iteration
            .values
            .get("lowCutoffCutoff")
            .is_some_and(|v| v.parse::<f64>().ok() == Some(0.0))
            && iteration
                .values
                .get("lowCutoffSigma")
                .is_some_and(|v| v.parse::<f64>().ok() == Some(0.05))
    }

    /// Java `setParameters(MatlabParam, boolean)`.
    pub fn set_parameters_matlab(&mut self, matlab: &MatlabParam, is_low_cutoff: bool) {
        let Some(iteration) = matlab.iterations.get(self.index) else {
            return;
        };
        Self::set_field_value(&iteration.values, "dPhiEnd", &mut self.d_phi_max);
        Self::set_field_value(
            &iteration.values,
            "dPhiIncrement",
            &mut self.d_phi_increment,
        );
        Self::set_field_value(&iteration.values, "dThetaEnd", &mut self.d_theta_max);
        Self::set_field_value(
            &iteration.values,
            "dThetaIncrement",
            &mut self.d_theta_increment,
        );
        Self::set_field_value(&iteration.values, "dPsiEnd", &mut self.d_psi_max);
        Self::set_field_value(
            &iteration.values,
            "dPsiIncrement",
            &mut self.d_psi_increment,
        );
        Self::set_field_value(&iteration.values, "searchRadius", &mut self.search_radius);
        Self::set_field_value(&iteration.values, "hiCutoffCutoff", &mut self.hi_cutoff);
        Self::set_field_value(
            &iteration.values,
            "hiCutoffSigma",
            &mut self.hi_cutoff_sigma,
        );
        if is_low_cutoff {
            Self::set_field_value(&iteration.values, "lowCutoffCutoff", &mut self.low_cutoff);
            Self::set_field_value(
                &iteration.values,
                "lowCutoffSigma",
                &mut self.low_cutoff_sigma,
            );
        }
        if self.low_cutoff_sigma.is_empty() {
            self.set_low_cutoff_sigma(Some(LOW_CUTOFF_SIGMA_DEFAULT));
        }
        Self::set_field_value(&iteration.values, "refThreshold", &mut self.ref_threshold);
        Self::set_field_value(
            &iteration.values,
            "duplicateShiftTolerance",
            &mut self.duplicate_shift_tolerance,
        );
        Self::set_field_value(
            &iteration.values,
            "duplicateAngularTolerance",
            &mut self.duplicate_angular_tolerance,
        );
    }

    /// Java `isHighlighted()`.
    pub fn is_highlighted(&self) -> bool {
        self.btn_highlighter.is_highlighted()
    }
    /// Java `getIndex()`.
    pub fn get_index(&self) -> usize {
        self.index
    }
    /// Java `setIndex(int)`.
    pub fn set_index(&mut self, index: usize) {
        self.index = index;
        self.number.set_text_int((index + 1) as i32);
    }
    /// Java `remove()`.
    pub fn remove(&mut self) {
        self.number.remove();
        self.btn_highlighter.remove();
        self.displayed = false;
    }
    /// Java `display()`.
    pub fn display(&mut self) {
        self.number.add();
        self.btn_highlighter.add();
        self.displayed = true;
    }

    /// Java private `buildHeaderDescription(String[])`.
    pub fn build_header_description(header_array: &[&str]) -> String {
        header_array
            .iter()
            .map(|header| format!(", {header}"))
            .collect()
    }

    /// Java private `validateRun(boolean, EtomoNumber, String[], String)`.
    pub fn validate_run_number(
        &mut self,
        empty: bool,
        value: &str,
        integer: bool,
        header_array: &[&str],
        additional_empty_error_message: Option<&str>,
    ) -> bool {
        let description = Self::build_header_description(header_array);
        if empty {
            self.last_message = Some((
                format!(
                    "{LABEL}:  In row {}{description} must not be empty.{}",
                    self.number.to_string(),
                    additional_empty_error_message.unwrap_or_default()
                ),
                "Entry Error".into(),
            ));
            return false;
        }
        let parsed: Result<f64, ()> = if integer {
            value.parse::<i64>().map(|n| n as f64).map_err(|_| ())
        } else {
            value.parse::<f64>().map_err(|_| ())
        };
        let Ok(number) = parsed else {
            self.last_message = Some((
                format!(
                    "{LABEL}:  In row {}{description}:   Invalid number",
                    self.number.to_string()
                ),
                "Entry Error".into(),
            ));
            return false;
        };
        if number < 0.0 {
            self.last_message = Some((
                format!(
                    "{LABEL}:  In row {}{description} must not be negative.",
                    self.number.get_text().unwrap_or_default()
                ),
                "Entry Error".into(),
            ));
            return false;
        }
        true
    }

    /// Java `validateRun(boolean)`.
    pub fn validate_run(&mut self, is_low_cutoff: bool) -> bool {
        let d_phi_max = self.d_phi_max.get_value().to_owned();
        let d_phi_increment = self.d_phi_increment.get_value().to_owned();
        let d_theta_max = self.d_theta_max.get_value().to_owned();
        let d_theta_increment = self.d_theta_increment.get_value().to_owned();
        let d_psi_max = self.d_psi_max.get_value().to_owned();
        let d_psi_increment = self.d_psi_increment.get_value().to_owned();
        let search_radius = self.search_radius.get_value().trim().to_owned();
        let hi_cutoff = self.hi_cutoff.get_value().to_owned();
        let hi_cutoff_sigma = self.hi_cutoff_sigma.get_value().to_owned();
        let low_cutoff = self.low_cutoff.get_value().to_owned();
        let low_cutoff_sigma = self.low_cutoff_sigma.get_value().to_owned();
        let ref_threshold = self.ref_threshold.get_value().to_owned();
        let duplicate_shift_tolerance = self.duplicate_shift_tolerance.get_value().to_owned();
        let duplicate_angular_tolerance = self.duplicate_angular_tolerance.get_value().to_owned();
        let theta_max_enabled = self.d_theta_max.is_enabled();
        let theta_increment_enabled = self.d_theta_increment.is_enabled();
        let psi_max_enabled = self.d_psi_max.is_enabled();
        let psi_increment_enabled = self.d_psi_increment.is_enabled();
        let duplicate_shift_enabled = self.duplicate_shift_tolerance.is_enabled();
        let duplicate_angular_enabled = self.duplicate_angular_tolerance.is_enabled();
        let checks = [
            (
                &d_phi_max,
                false,
                [D_PHI_D_THETA_D_PSI_HEADER1, "Phi", MAX_HEADER3].as_slice(),
                Some("Use 0 to not search on the angle."),
            ),
            (
                &d_phi_increment,
                false,
                [D_PHI_D_THETA_D_PSI_HEADER1, "Phi", INCR_HEADER3].as_slice(),
                None,
            ),
        ];
        for (value, integer, headers, extra) in checks {
            if !self.validate_run_number(value.trim().is_empty(), value, integer, headers, extra) {
                return false;
            }
        }
        for (value, enabled, headers, extra) in [
            (
                &d_theta_max,
                theta_max_enabled,
                [D_PHI_D_THETA_D_PSI_HEADER1, "Theta", MAX_HEADER3].as_slice(),
                Some("Use 0 to not search on the angle."),
            ),
            (
                &d_theta_increment,
                theta_increment_enabled,
                [D_PHI_D_THETA_D_PSI_HEADER1, "Theta", INCR_HEADER3].as_slice(),
                None,
            ),
            (
                &d_psi_max,
                psi_max_enabled,
                [D_PHI_D_THETA_D_PSI_HEADER1, "Psi", MAX_HEADER3].as_slice(),
                Some("Use 0 to not search on the angle."),
            ),
            (
                &d_psi_increment,
                psi_increment_enabled,
                [D_PHI_D_THETA_D_PSI_HEADER1, "Psi", INCR_HEADER3].as_slice(),
                None,
            ),
        ] {
            if enabled
                && !self.validate_run_number(value.trim().is_empty(), value, false, headers, extra)
            {
                return false;
            }
        }
        let radius_headers = [SEARCH_RADIUS_HEADER1, SEARCH_RADIUS_HEADER2];
        let radius = search_radius.as_str();
        if radius.parse::<i64>().is_ok() || radius.is_empty() {
            if !self.validate_run_number(radius.is_empty(), radius, true, &radius_headers, None) {
                return false;
            }
        } else {
            let parts: Vec<_> = radius
                .split(',')
                .map(str::trim)
                .filter(|part| !part.is_empty())
                .collect();
            let parts = if parts.len() == 3 {
                parts
            } else {
                radius.split_whitespace().collect()
            };
            if parts.len() != 3 {
                self.last_message = Some((
                    format!(
                        "{LABEL}:  In row {}{} must have either 1 or 3 elements.",
                        self.number.to_string(),
                        Self::build_header_description(&radius_headers)
                    ),
                    "Entry Error".into(),
                ));
                return false;
            }
            for part in parts {
                if !self.validate_run_number(false, part, true, &radius_headers, None) {
                    return false;
                }
            }
        }
        for (value, headers, extra) in [
            (
                &hi_cutoff,
                [HICUTOFF_HEADER1, HICUTOFF_HEADER2, HICUTOFF_CUTOFF_HEADER3].as_slice(),
                None,
            ),
            (
                &hi_cutoff_sigma,
                [HICUTOFF_HEADER1, HICUTOFF_HEADER2, HICUTOFF_SIGMA_HEADER3].as_slice(),
                None,
            ),
            (
                &ref_threshold,
                [REF_THRESHOLD_HEADER1, REF_THRESHOLD_HEADER2].as_slice(),
                None,
            ),
        ] {
            if !self.validate_run_number(value.trim().is_empty(), value, false, headers, extra) {
                return false;
            }
        }
        if is_low_cutoff {
            for (value, headers, extra) in [
                (
                    &low_cutoff,
                    [
                        LOWCUTOFF_HEADER1,
                        LOWCUTOFF_HEADER2,
                        LOWCUTOFF_CUTOFF_HEADER3,
                    ]
                    .as_slice(),
                    Some(" Use 0 to disable filtering for an iteration."),
                ),
                (
                    &low_cutoff_sigma,
                    [
                        LOWCUTOFF_HEADER1,
                        LOWCUTOFF_HEADER2,
                        LOWCUTOFF_SIGMA_HEADER3,
                    ]
                    .as_slice(),
                    None,
                ),
            ] {
                if !self.validate_run_number(value.trim().is_empty(), value, false, headers, extra)
                {
                    return false;
                }
            }
        }
        for (value, enabled, headers) in [
            (
                &duplicate_shift_tolerance,
                duplicate_shift_enabled,
                [
                    DUPLICATE_TOLERANCE_HEADER1,
                    DUPLICATE_TOLERANCE_HEADER2,
                    DUPLICATE_SHIFT_TOLERANCE_HEADER3,
                ]
                .as_slice(),
            ),
            (
                &duplicate_angular_tolerance,
                duplicate_angular_enabled,
                [
                    DUPLICATE_TOLERANCE_HEADER1,
                    DUPLICATE_TOLERANCE_HEADER2,
                    DUPLICATE_ANGULAR_TOLERANCE_HEADER3,
                ]
                .as_slice(),
            ),
        ] {
            if enabled
                && !self.validate_run_number(value.trim().is_empty(), value, true, headers, None)
            {
                return false;
            }
        }
        true
    }

    /// Java private `setTooltips`; autodoc duplicate tolerance text remains the
    /// explicit storage/autodoc boundary, while source-local tooltips are exact.
    pub fn set_tooltips(&mut self) {
        self.number.set_tool_tip_text("Iteration number");
        self.d_phi_max.set_tooltip_text("Maximum magnitude of rotation about the particle Y axis in degrees.  Search will range from -(Phi Max) to +(Phi Max) in steps of (Phi Step).");
        self.d_phi_increment.set_tooltip_text("Increment between sample points for rotation about Y in degrees.  Search will range from -(Phi Max) to +(Phi Max) in steps of (Phi Step).");
        self.d_theta_max.set_tooltip_text("Maximum magnitude of rotation about the particle Z axis in degrees.  Search will range from -(Theta Max) to +(Theta Max) in steps of (Theta Step).");
        self.d_theta_increment.set_tooltip_text("Increment between sample points for rotation about Z in degrees.  Search will range from -(Theta Max) to +(Theta Max) in steps of (Theta Step).");
        self.d_psi_max.set_tooltip_text("Maximum magnitude of rotation about the particle X axis in degrees.  Search will range from -(Psi Max) to +(Psi Max) in steps of (Psi Step).");
        self.d_psi_increment.set_tooltip_text("Increment between sample points for rotation about X in degrees.  Search will range from -(Psi Max) to +(Psi Max) in steps of (Psi Step).");
        self.search_radius.set_tooltip_text("The number of pixels to search in the X, Y, and Z directions.  A single, integer number of pixels can be specified, which will be applied to all 3 dimensions, or a vector of 3 integers can be specified, giving the X, Y, and Z search distances individually. E.g. '3' is equivalent to '3 3 3'.");
        self.hi_cutoff.set_tooltip_text("The normalized spatial frequency above which high frequencies are attenuated.  0.5 corresponds to the Nyquist frequency, and values of 0.866 or larger disable low-pass filtering.");
        self.hi_cutoff_sigma.set_tooltip_text("The width (standard deviation) in normalized frequency units of a Gaussian determining the rate at which attenuation increases above the cutoff.");
        self.low_cutoff.set_tooltip_text("The normalized frequency below which low frequencies will be attenuated. Values <= 0 disable high-pass filtering.");
        self.low_cutoff_sigma.set_tooltip_text(
            "An optional parameter which defines the transition width of the high-pass filter.",
        );
        self.ref_threshold.set_tooltip_text("Determines the number of particles averaged to form the reference for the next alignment iteration. If less than 1, it represents a cross-correlation coefficient threshold, with particles having a larger correlation eligible for inclusion in the reference.  If greater than 1, it is the number of particles to include.");
    }
    /// Java `setVisibleLowCutoffRows(boolean)`.
    pub fn set_visible_low_cutoff_rows(&mut self, input: bool) {
        self.low_cutoff.set_visible(input);
        self.low_cutoff_sigma.set_visible(input);
    }
    /// Java `setLowCutoffCutoff(String)`.
    pub fn set_low_cutoff_cutoff(&mut self, input: Option<&str>) {
        if let Some(input) = input {
            self.low_cutoff.set_value(input);
        }
    }
    /// Java `setLowCutoffSigma(String)`.
    pub fn set_low_cutoff_sigma(&mut self, input: Option<&str>) {
        if let Some(input) = input {
            self.low_cutoff_sigma.set_value(input);
        }
    }

    fn set_field_value(
        values: &std::collections::BTreeMap<String, String>,
        key: &str,
        field: &mut FieldCell,
    ) {
        field.set_value(values.get(key).map(String::as_str).unwrap_or_default());
    }
    fn fields_mut(&mut self) -> [&mut FieldCell; 14] {
        [
            &mut self.d_phi_max,
            &mut self.d_phi_increment,
            &mut self.d_theta_max,
            &mut self.d_theta_increment,
            &mut self.d_psi_max,
            &mut self.d_psi_increment,
            &mut self.search_radius,
            &mut self.hi_cutoff,
            &mut self.hi_cutoff_sigma,
            &mut self.low_cutoff,
            &mut self.low_cutoff_sigma,
            &mut self.ref_threshold,
            &mut self.duplicate_shift_tolerance,
            &mut self.duplicate_angular_tolerance,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn source_validation_obeys_disabled_and_triplet_rules() {
        let mut row = IterationRow::new(0, false);
        for field in [
            &mut row.d_phi_max,
            &mut row.d_phi_increment,
            &mut row.search_radius,
            &mut row.hi_cutoff,
            &mut row.hi_cutoff_sigma,
            &mut row.ref_threshold,
        ] {
            field.set_value("1");
        }
        row.search_radius.set_value("1, 2, 3");
        row.update_display(true, false);
        assert!(row.validate_run(false));
        row.search_radius.set_value("1, 2");
        assert!(!row.validate_run(false));
        assert!(
            row.last_message
                .as_ref()
                .unwrap()
                .0
                .contains("either 1 or 3 elements")
        );
    }
    #[test]
    fn source_parameter_defaults_and_copy_are_retained() {
        let mut row = IterationRow::new(0, false);
        row.d_phi_max.set_value("3");
        row.d_phi_increment.set_value("1");
        let mut iteration = Iteration::default();
        row.get_parameters_matlab(&mut iteration, false);
        assert_eq!(iteration.values["dPhiEnd"], "3");
        assert_eq!(iteration.values["lowCutoffSigma"], LOW_CUTOFF_SIGMA_DEFAULT);
        let copy = IterationRow::copy(1, &row);
        assert_eq!(copy.get_index(), 1);
        assert_eq!(copy.d_phi_max.get_value(), "3");
        assert!(!copy.is_highlighted());
    }
}
