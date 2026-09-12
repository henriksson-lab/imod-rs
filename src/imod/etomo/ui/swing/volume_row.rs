//! `IMOD/Etomo/src/etomo/ui/swing/VolumeRow.java`.
//!
//! The Swing cells, file chooser, `BaseManager`, and 3dmod invocation are native GUI
//! boundaries.  This source unit owns the complete row state and source-shaped
//! operations; `VolumeTable` owns only its `RowList` and table-wide controls.
#![allow(dead_code)]

use std::path::{Path, PathBuf};

use super::volume_table::{
    LABEL, MatlabVolume, PeetMetaData, Run3dmodMenuOptions, TILT_RANGE_HEADER1_LABEL,
};

/// Java final package-private `VolumeRow`.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct VolumeRow {
    /// Java `HeaderCell number` text.
    pub number: String,
    pub index: usize,
    pub imod_index: isize,
    pub highlighted: bool,
    pub fn_volume: String,
    pub fn_mod_particle: String,
    pub init_motl_file: String,
    pub tilt_range_min: String,
    pub tilt_range_max: String,
    pub tilt_range_multi_axes: String,
    pub fn_volume_expanded: bool,
    pub fn_mod_particle_expanded: bool,
    pub init_motl_file_expanded: bool,
    pub tilt_range_multi_axes_expanded: bool,
    pub displayed: bool,
    pub removed: bool,
    pub action_targets_added: bool,
    pub names_set: bool,
    pub tooltips_set: bool,
    pub fn_volume_tooltip: String,
    pub fn_mod_particle_tooltip: String,
    pub init_motl_file_tooltip: String,
    pub tilt_range_min_tooltip: String,
    pub tilt_range_max_tooltip: String,
    pub tilt_range_multi_axes_tooltip: String,
    pub fn_volume_file_filter: String,
    pub fn_mod_particle_file_filter: String,
    pub init_motl_file_filter: String,
    pub tilt_range_multi_axes_file_filter: String,
    pub last_imod_open: Option<(String, String, Run3dmodMenuOptions)>,
    /// File chooser is an explicit boundary. `None` is Java cancel; `Some` is the
    /// selected existing file returned by the boundary.
    pub correct_path: Option<PathBuf>,
    pub table_header: Option<String>,
    pub row_header: Option<String>,
    pub fn_volume_header: Option<String>,
    pub fn_mod_particle_header: Option<String>,
    pub init_motl_file_header: Option<String>,
    pub tilt_range_header: Option<String>,
    pub tilt_range_multi_axes_header: Option<String>,
    /// Java `Column.add` calls remain column-layout boundary registrations.
    pub registered_init_motl_column: bool,
    pub registered_tilt_range_column: bool,
}

impl VolumeRow {
    /// Java `getInstance(BaseManager,int,VolumeTable,JPanel,GridBagLayout,GridBagConstraints,VolumeFileFilter)`.
    pub fn get_instance(index: usize) -> Self {
        let mut instance = Self {
            number: (index + 1).to_string(),
            index,
            imod_index: -1,
            fn_volume_file_filter: "VolumeFileFilter".to_owned(),
            fn_mod_particle_file_filter: "ModelFileFilter".to_owned(),
            init_motl_file_filter: "MotlFileFilter".to_owned(),
            tilt_range_multi_axes_file_filter: "VolumeFileFilter".to_owned(),
            ..Default::default()
        };
        instance.add_action_targets();
        instance.set_tooltips();
        instance
    }

    /// Java file-valued `getInstance` overload.
    pub fn get_instance_with_files(
        fn_volume: Option<&Path>,
        fn_mod_particle: Option<&Path>,
        tilt_range_multi_axes: Option<&Path>,
        index: usize,
    ) -> Self {
        let mut instance = Self::get_instance(index);
        Self::set_value_file(&mut instance.fn_volume, fn_volume);
        Self::set_value_file(&mut instance.fn_mod_particle, fn_mod_particle);
        Self::set_value_file(&mut instance.tilt_range_multi_axes, tilt_range_multi_axes);
        instance
    }

    /// Java string-valued `getInstance` overload.
    pub fn get_instance_with_values(
        fn_volume: Option<&str>,
        fn_mod_particle: Option<&str>,
        tilt_range_multi_axes: Option<&str>,
        index: usize,
    ) -> Self {
        let mut instance = Self::get_instance(index);
        Self::set_value(&mut instance.fn_volume, fn_volume);
        Self::set_value(&mut instance.fn_mod_particle, fn_mod_particle);
        Self::set_value(&mut instance.tilt_range_multi_axes, tilt_range_multi_axes);
        instance
    }

    /// Java copying `getInstance(VolumeRow,int)`.
    pub fn get_instance_copy(volume_row: &Self, index: usize) -> Self {
        let mut instance = volume_row.clone();
        instance.index = index;
        instance.number = (index + 1).to_string();
        instance.highlighted = false;
        instance.displayed = false;
        instance.removed = false;
        instance.imod_index = -1;
        instance.add_action_targets();
        instance.set_tooltips();
        instance
    }

    /// Java private `addActionTargets`.
    pub fn add_action_targets(&mut self) {
        self.action_targets_added = true;
    }

    /// Java `setNames`; header identities are rendered by the Swing boundary.
    pub fn set_names(&mut self) {
        self.names_set = true;
    }

    /// Java `setHeaders(FieldCell,FileButtonCell,HeaderCell)`; actual widget naming
    /// is delegated to their source units, while this preserves the row's header link.
    pub fn set_headers(&mut self, table_header: &str, row_header: &str, column_header: &str) {
        self.table_header = Some(table_header.to_owned());
        self.row_header = Some(row_header.to_owned());
        self.fn_volume_header = Some(column_header.to_owned());
    }

    /// Java `getTextSize`.
    pub fn get_text_size(&self, tilt_range_multi_axes: bool) -> usize {
        self.fn_volume.len().max(6)
            + self.fn_mod_particle.len().max(5)
            + self.init_motl_file.len().max(5)
            + if tilt_range_multi_axes {
                self.tilt_range_multi_axes.len().max(5)
            } else {
                0
            }
    }

    /// Java `setHighlighterSelected`.
    pub fn set_highlighter_selected(&mut self, select: bool) {
        self.highlighted = select;
        self.highlight(select);
    }

    /// Java `Highlightable.highlight`; field highlighting is rendered by Swing.
    pub fn highlight(&mut self, highlight: bool) {
        self.highlighted = highlight;
    }

    /// Java `remove`.
    pub fn remove(&mut self) {
        self.displayed = false;
        self.removed = true;
    }

    /// Java `display(int,Viewport)` after the caller has established viewport membership.
    pub fn display(&mut self, in_viewport: bool, _tilt_range_multi_axes: bool) {
        if in_viewport {
            self.displayed = true;
            self.removed = false;
        }
    }

    pub fn expand_fn_volume(&mut self, expanded: bool) {
        self.fn_volume_expanded = expanded;
    }
    pub fn get_index(&self) -> usize {
        self.index
    }
    pub fn set_index(&mut self, index: usize) {
        self.index = index;
        self.number = (index + 1).to_string();
    }
    pub fn expand_fn_mod_particle(&mut self, expanded: bool) {
        self.fn_mod_particle_expanded = expanded;
    }
    pub fn expand_init_motl_file(&mut self, expanded: bool) {
        self.init_motl_file_expanded = expanded;
    }
    pub fn expand_tilt_range_multi_axes(&mut self, expanded: bool) {
        self.tilt_range_multi_axes_expanded = expanded;
    }

    /// Java `getParameters(PeetMetaData)`.
    pub fn get_parameters(&self, meta_data: &mut PeetMetaData) {
        meta_data.init_motl_file.push(self.init_motl_file.clone());
        meta_data.tilt_range_min.push(self.tilt_range_min.clone());
        meta_data.tilt_range_max.push(self.tilt_range_max.clone());
        meta_data
            .tilt_range_multi_axes_file
            .push(self.tilt_range_multi_axes.clone());
    }

    /// Java `convertCopiedPaths`; the supplied roots are the `FilePath` boundary.
    pub fn convert_copied_paths(&mut self, orig_dataset_dir: &Path, property_user_dir: &Path) {
        for value in [
            &mut self.fn_volume,
            &mut self.fn_mod_particle,
            &mut self.init_motl_file,
            &mut self.tilt_range_multi_axes,
        ] {
            if !value.trim().is_empty() && !Path::new(value.as_str()).is_absolute() {
                let absolute = orig_dataset_dir.join(value.as_str());
                *value = absolute
                    .strip_prefix(property_user_dir)
                    .unwrap_or(&absolute)
                    .to_string_lossy()
                    .into_owned();
            }
        }
    }

    /// Java `isIncorrectPaths`.
    pub fn is_incorrect_paths(&self, root: &Path) -> bool {
        [
            self.fn_volume.as_str(),
            self.fn_mod_particle.as_str(),
            self.init_motl_file.as_str(),
            self.tilt_range_multi_axes.as_str(),
        ]
        .into_iter()
        .any(|value| !value.trim().is_empty() && !root.join(value).exists())
    }

    /// Java `fixIncorrectPaths`; chooser selection remains the native boundary.
    pub fn fix_incorrect_paths(&mut self, root: &Path, choose_path_every_row: bool) -> bool {
        let mut choose_path = choose_path_every_row;
        for (value, expanded) in [
            (&mut self.fn_volume, self.fn_volume_expanded),
            (&mut self.fn_mod_particle, self.fn_mod_particle_expanded),
            (&mut self.init_motl_file, self.init_motl_file_expanded),
            (
                &mut self.tilt_range_multi_axes,
                self.tilt_range_multi_axes_expanded,
            ),
        ] {
            if !value.trim().is_empty() && !root.join(value.as_str()).exists() {
                if !Self::fix_incorrect_path(
                    value,
                    root,
                    &mut self.correct_path,
                    choose_path,
                    expanded,
                ) {
                    return false;
                }
                choose_path = false;
            }
        }
        true
    }

    /// Java private `fixIncorrectPath`; chosen file input is represented by `correct_path`.
    pub fn fix_incorrect_path(
        value: &mut String,
        root: &Path,
        correct_path: &mut Option<PathBuf>,
        choose_path: bool,
        _expand: bool,
    ) -> bool {
        if !choose_path {
            if let Some(correct_path) = correct_path.as_ref() {
                let candidate =
                    correct_path.join(Path::new(value.as_str()).file_name().unwrap_or_default());
                if candidate.exists() {
                    *value = candidate
                        .strip_prefix(root)
                        .unwrap_or(&candidate)
                        .to_string_lossy()
                        .into_owned();
                    return true;
                }
            }
        }
        false
    }

    /// Java `setParameters(ConstPeetMetaData)`.
    pub fn set_parameters(&mut self, meta_data: &PeetMetaData) {
        if let Some(value) = meta_data.init_motl_file.get(self.index) {
            self.set_init_motl_file(Some(value));
        }
        if let Some(value) = meta_data.tilt_range_min.get(self.index) {
            self.set_tilt_range_min(Some(value));
        }
        if let Some(value) = meta_data.tilt_range_max.get(self.index) {
            self.set_tilt_range_max(Some(value));
        }
        if let Some(value) = meta_data.tilt_range_multi_axes_file.get(self.index) {
            Self::set_value(&mut self.tilt_range_multi_axes, Some(value));
        }
    }

    /// Java `getParameters(MatlabParam,boolean)` row portion.
    pub fn get_matlab_parameters(&self, multi_axes: bool) -> MatlabVolume {
        MatlabVolume {
            fn_volume: self.fn_volume.clone(),
            fn_mod_particle: self.fn_mod_particle.clone(),
            init_motl: self.init_motl_file.clone(),
            tilt_range_start: if multi_axes {
                String::new()
            } else {
                self.tilt_range_min.clone()
            },
            tilt_range_end: if multi_axes {
                String::new()
            } else {
                self.tilt_range_max.clone()
            },
            tilt_range_multi_axes: if multi_axes {
                self.tilt_range_multi_axes.clone()
            } else {
                String::new()
            },
        }
    }

    /// Java `setParameters(MatlabParam,boolean,boolean,boolean)` row portion.
    pub fn set_matlab_parameters(
        &mut self,
        volume: &MatlabVolume,
        use_init_motl: bool,
        use_tilt_range: bool,
        multi_axes: bool,
    ) {
        if use_init_motl {
            self.set_init_motl_file(Some(&volume.init_motl));
        }
        if use_tilt_range && !multi_axes {
            self.set_tilt_range_min(Some(&volume.tilt_range_start));
            self.set_tilt_range_max(Some(&volume.tilt_range_end));
        } else if use_tilt_range {
            Self::set_value(
                &mut self.tilt_range_multi_axes,
                Some(&volume.tilt_range_multi_axes),
            );
        }
    }

    pub fn clear_init_motl_file(&mut self) {
        self.init_motl_file.clear();
    }
    /// Java `registerInitMotlFileColumn`.
    pub fn register_init_motl_file_column(&mut self) {
        self.registered_init_motl_column = true;
    }
    /// Java `registerTiltRangeColumn`.
    pub fn register_tilt_range_column(&mut self) {
        self.registered_tilt_range_column = true;
    }
    pub fn imod_volume(&mut self, menu_options: Run3dmodMenuOptions) {
        self.imod_index += 1;
        self.last_imod_open = Some((
            self.fn_volume.clone(),
            self.fn_mod_particle.clone(),
            menu_options,
        ));
    }

    /// Java `validateRun`; Java requires model, not volume.
    pub fn validate_run(
        &self,
        tilt_range_required: bool,
        tilt_range_multi_axes: bool,
    ) -> Option<String> {
        if self.fn_mod_particle.trim().is_empty() {
            return Some(format!(
                "{LABEL}:  In row {}, Model must not be empty.",
                self.number
            ));
        }
        if tilt_range_required
            && !tilt_range_multi_axes
            && (self.tilt_range_min.trim().is_empty() || self.tilt_range_max.trim().is_empty())
        {
            return Some(format!(
                "{LABEL}:  In row {}, {TILT_RANGE_HEADER1_LABEL} is required.",
                self.number
            ));
        }
        if tilt_range_required
            && tilt_range_multi_axes
            && self.tilt_range_multi_axes.trim().is_empty()
        {
            return Some(format!(
                "{LABEL}:  In row {}, Missing Wedge Mask is required.",
                self.number
            ));
        }
        None
    }

    pub fn set_init_motl_file(&mut self, input: Option<&str>) {
        if let Some(input) = input.filter(|value| !value.trim().is_empty()) {
            self.init_motl_file = input.to_owned();
        }
    }
    /// Java overloaded `setInitMotlFile(File)`.
    pub fn set_init_motl_file_file(&mut self, input: Option<&Path>) {
        Self::set_value_file(&mut self.init_motl_file, input);
    }
    pub fn get_expanded_init_motl_file(&self) -> Option<&str> {
        (!self.init_motl_file.trim().is_empty()).then_some(self.init_motl_file.as_str())
    }
    pub fn get_fn_volume_file(&self, root: &Path) -> Option<PathBuf> {
        (!self.fn_volume.trim().is_empty()).then(|| root.join(&self.fn_volume))
    }
    pub fn get_fn_mod_particle_file(&self, root: &Path) -> Option<PathBuf> {
        (!self.fn_mod_particle.trim().is_empty()).then(|| root.join(&self.fn_mod_particle))
    }
    /// Java `setFnModParticle(File)`.
    pub fn set_fn_mod_particle(&mut self, input: Option<&Path>) {
        Self::set_value_file(&mut self.fn_mod_particle, input);
    }
    pub fn set_tilt_range_min(&mut self, input: Option<&str>) {
        if let Some(input) = input {
            self.tilt_range_min = input.to_owned();
        }
    }
    pub fn get_tilt_range_min(&self) -> &str {
        &self.tilt_range_min
    }
    pub fn get_tilt_range_max(&self) -> &str {
        &self.tilt_range_max
    }
    pub fn set_tilt_range_max(&mut self, input: Option<&str>) {
        if let Some(input) = input {
            self.tilt_range_max = input.to_owned();
        }
    }
    pub fn is_highlighted(&self) -> bool {
        self.highlighted
    }

    /// Java private `setValue(FieldCell,String)`.
    fn set_value(field_cell: &mut String, input: Option<&str>) {
        if let Some(input) = input.filter(|value| !value.trim().is_empty()) {
            *field_cell = input.to_owned();
        }
    }
    /// Java private `setValue(FieldCell,File)`.
    fn set_value_file(field_cell: &mut String, input: Option<&Path>) {
        if let Some(input) = input {
            *field_cell = input.to_string_lossy().into_owned();
        }
    }

    /// Java private `setTooltips`.
    pub fn set_tooltips(&mut self) {
        self.tooltips_set = true;
        self.fn_volume_tooltip = "The filename of the tomogram in MRC format.".to_owned();
        self.fn_mod_particle_tooltip =
            "The filename of the IMOD model specifying particle positions in the tomogram."
                .to_owned();
        self.init_motl_file_tooltip = "The name of a .csv file containing an initial motive list with orientations and shifts.".to_owned();
        self.tilt_range_min_tooltip = "The minimum tilt angle (in degrees) used during image acquisition for this tomogram.  Used only if missing wedge compensation is enabled.".to_owned();
        self.tilt_range_max_tooltip = self
            .tilt_range_min_tooltip
            .replacen("minimum", "maximum", 1);
        self.tilt_range_multi_axes_tooltip = "A binary mask file in MRC format with 0's and 1's indicating missing and valid regions in Fourier space, respectively, for this volume.  The mask should be cubical, with an even number of voxels, at least as large as the largest dimension of the reference, along each edge.".to_owned();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_copy_index_validation_and_matlab_paths_are_row_owned() {
        let mut row =
            VolumeRow::get_instance_with_values(Some("a.rec"), Some("a.mod"), Some("mask.rec"), 2);
        row.set_init_motl_file(Some("init.csv"));
        row.set_tilt_range_min(Some("-60"));
        row.set_tilt_range_max(Some("60"));
        assert_eq!(row.number, "3");
        assert!(row.validate_run(true, false).is_none());
        let copy = VolumeRow::get_instance_copy(&row, 3);
        assert_eq!(copy.number, "4");
        assert!(!copy.highlighted);
        assert_eq!(copy.get_matlab_parameters(false).tilt_range_start, "-60");
    }
}
