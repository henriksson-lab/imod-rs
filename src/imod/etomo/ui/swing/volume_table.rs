//! `IMOD/Etomo/src/etomo/ui/swing/VolumeTable.java`.
//!
//! Swing components, file chooser interaction, 3dmod launch, `PeetDialog`, and
//! `VolumeRow.java` remain direct presentation/source-unit boundaries.  This
//! module deliberately retains the table's source-owned row ordering, paging,
//! expansion, enablement, validation, and action dispatch state.
#![allow(dead_code)]

use std::path::{Path, PathBuf};

pub const FN_VOLUME_HEADER1: &str = "Volume";
pub const FN_MOD_PARTICLE_HEADER1: &str = "Model";
pub const INIT_MOTL_FILE_HEADER1: &str = "Initial";
pub const INIT_MOTL_FILE_HEADER2: &str = "MOTL";
pub const LABEL: &str = "Volume Table";
pub const TILT_RANGE_HEADER1_LABEL: &str = "Tilt Range";
pub const TILT_RANGE_MULTI_AXES_HEADER1_LABEL: &str = "Missing Wedge";
pub const TILT_RANGE_MULTI_AXES_HEADER2_LABEL: &str = "Mask";
pub const UNIQUE_KEY: &str = "Volume";

/// Java `Run3dmodMenuOptions`, passed without interpretation to the manager boundary.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Run3dmodMenuOptions;

/// `MatlabParam.Volume` state observed by this source unit.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MatlabVolume {
    pub fn_volume: String,
    pub fn_mod_particle: String,
    pub init_motl: String,
    pub tilt_range_start: String,
    pub tilt_range_end: String,
    pub tilt_range_multi_axes: String,
}

/// `MatlabParam` data boundary used by the two Java `set/getParameters` overloads.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MatlabParam {
    pub flg_vol_names_are_templates: bool,
    pub volumes: Vec<MatlabVolume>,
}

/// `PeetMetaData`/`ConstPeetMetaData` data boundary used by `RowList`.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PeetMetaData {
    pub init_motl_file: Vec<String>,
    pub tilt_range_min: Vec<String>,
    pub tilt_range_max: Vec<String>,
    pub tilt_range_multi_axes_file: Vec<String>,
}

/// Values from `VolumeRow.java` which are directly observed by `VolumeTable`.
/// The full field/widget implementation belongs to that separate source unit.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct VolumeRow {
    pub index: usize,
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
}
impl VolumeRow {
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
    pub fn validate_run(
        &self,
        tilt_range_required: bool,
        tilt_range_multi_axes: bool,
    ) -> Option<String> {
        if self.fn_volume.trim().is_empty() {
            return Some(format!("Volume is required in row {}", self.index + 1));
        }
        if tilt_range_required
            && if tilt_range_multi_axes {
                self.tilt_range_multi_axes.trim().is_empty()
            } else {
                self.tilt_range_min.trim().is_empty() || self.tilt_range_max.trim().is_empty()
            }
        {
            return Some(format!("Tilt range is required in row {}", self.index + 1));
        }
        None
    }
    pub fn is_incorrect_paths(&self, root: &Path) -> bool {
        [
            self.fn_volume.as_str(),
            self.fn_mod_particle.as_str(),
            self.init_motl_file.as_str(),
            self.tilt_range_multi_axes.as_str(),
        ]
        .into_iter()
        .any(|value| !value.is_empty() && !root.join(value).exists())
    }
    pub fn get_parameters(&self, meta_data: &mut PeetMetaData) {
        meta_data.init_motl_file.push(self.init_motl_file.clone());
        meta_data.tilt_range_min.push(self.tilt_range_min.clone());
        meta_data.tilt_range_max.push(self.tilt_range_max.clone());
        meta_data
            .tilt_range_multi_axes_file
            .push(self.tilt_range_multi_axes.clone());
    }
    pub fn set_parameters(&mut self, meta_data: &PeetMetaData) {
        if let Some(value) = meta_data.init_motl_file.get(self.index) {
            self.init_motl_file = value.clone();
        }
        if let Some(value) = meta_data.tilt_range_min.get(self.index) {
            self.tilt_range_min = value.clone();
        }
        if let Some(value) = meta_data.tilt_range_max.get(self.index) {
            self.tilt_range_max = value.clone();
        }
        if let Some(value) = meta_data.tilt_range_multi_axes_file.get(self.index) {
            self.tilt_range_multi_axes = value.clone();
        }
    }
    pub fn get_matlab_parameters(&self, multi_axes: bool) -> MatlabVolume {
        MatlabVolume {
            fn_volume: self.fn_volume.clone(),
            fn_mod_particle: self.fn_mod_particle.clone(),
            init_motl: self.init_motl_file.clone(),
            tilt_range_start: (!multi_axes)
                .then(|| self.tilt_range_min.clone())
                .unwrap_or_default(),
            tilt_range_end: (!multi_axes)
                .then(|| self.tilt_range_max.clone())
                .unwrap_or_default(),
            tilt_range_multi_axes: multi_axes
                .then(|| self.tilt_range_multi_axes.clone())
                .unwrap_or_default(),
        }
    }
    pub fn set_matlab_parameters(
        &mut self,
        volume: &MatlabVolume,
        use_init_motl: bool,
        use_tilt_range: bool,
        multi_axes: bool,
    ) {
        if use_init_motl {
            self.init_motl_file = volume.init_motl.clone();
        }
        if use_tilt_range && !multi_axes {
            self.tilt_range_min = volume.tilt_range_start.clone();
            self.tilt_range_max = volume.tilt_range_end.clone();
        } else if use_tilt_range {
            self.tilt_range_multi_axes = volume.tilt_range_multi_axes.clone();
        }
    }
}

/// Java private final inner `RowList`.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RowList {
    pub list: Vec<VolumeRow>,
    /// `ConstPeetMetaData` is a direct type/source boundary; this preserves the
    /// Java lifetime fact that metadata is retained only while constructing rows.
    pub meta_data_pending: bool,
}
impl RowList {
    pub fn size(&self) -> usize {
        self.list.len()
    }
    pub fn is_empty(&self) -> bool {
        self.list.is_empty()
    }
    pub fn remove(&mut self) {
        for row in &mut self.list {
            row.displayed = false;
        }
    }
    pub fn delete(&mut self, row_index: Option<usize>) -> isize {
        let Some(index) = row_index.filter(|index| *index < self.list.len()) else {
            return -1;
        };
        self.list.remove(index);
        self.reindex(index);
        index as isize
    }
    pub fn add(&mut self) -> &mut VolumeRow {
        let index = self.list.len();
        self.list.push(VolumeRow {
            index,
            ..Default::default()
        });
        self.list.last_mut().unwrap()
    }
    pub fn add_at(&mut self, index: usize) -> &mut VolumeRow {
        let index = index.min(self.list.len());
        self.list.insert(
            index,
            VolumeRow {
                index,
                ..Default::default()
            },
        );
        self.reindex(index);
        &mut self.list[index]
    }
    pub fn add_values(
        &mut self,
        fn_volume: impl Into<String>,
        fn_mod_particle: impl Into<String>,
        tilt_range_multi_axes: impl Into<String>,
    ) -> &mut VolumeRow {
        let row = self.add();
        row.fn_volume = fn_volume.into();
        row.fn_mod_particle = fn_mod_particle.into();
        row.tilt_range_multi_axes = tilt_range_multi_axes.into();
        row
    }
    pub fn add_copy(&mut self, from_index: usize) -> Option<&mut VolumeRow> {
        let row = self.list.get(from_index)?.clone();
        let index = from_index + 1;
        self.list.insert(
            index,
            VolumeRow {
                index,
                highlighted: false,
                ..row
            },
        );
        self.reindex(index);
        Some(&mut self.list[index])
    }
    pub fn move_row_up(&mut self, row_index: usize) {
        self.list.swap(row_index, row_index - 1);
    }
    pub fn move_row_down(&mut self, row_index: usize) {
        self.list.swap(row_index, row_index + 1);
    }
    pub fn highlight(&mut self, row_index: usize) {
        if let Some(row) = self.list.get_mut(row_index) {
            row.highlighted = true;
        }
    }
    pub fn highlight_down(&mut self) -> isize {
        let index = self.get_highlighted_row_index();
        if index < 0 {
            return -1;
        }
        let index = (index as usize + 1) % self.size();
        self.highlight(index);
        index as isize
    }
    pub fn highlight_up(&mut self) -> isize {
        let index = self.get_highlighted_row_index();
        if index < 0 {
            return -1;
        }
        let index = if index == 0 {
            self.size() - 1
        } else {
            index as usize - 1
        };
        self.highlight(index);
        index as isize
    }
    pub fn reindex(&mut self, start_index: usize) {
        for (index, row) in self.list.iter_mut().enumerate().skip(start_index) {
            row.index = index;
        }
    }
    pub fn validate_run(
        &self,
        tilt_range_required: bool,
        tilt_range_multi_axes: bool,
    ) -> Option<String> {
        if self.list.is_empty() {
            return Some(format!("Must enter at least one row in {LABEL}"));
        }
        self.list
            .iter()
            .find_map(|row| row.validate_run(tilt_range_required, tilt_range_multi_axes))
    }
    pub fn get_parameters(&self, meta_data: &mut PeetMetaData) {
        meta_data.init_motl_file.clear();
        meta_data.tilt_range_min.clear();
        meta_data.tilt_range_max.clear();
        meta_data.tilt_range_multi_axes_file.clear();
        for row in &self.list {
            row.get_parameters(meta_data);
        }
    }
    pub fn get_matlab_parameters(&self, matlab_param: &mut MatlabParam, multi_axes: bool) {
        matlab_param.volumes = self
            .list
            .iter()
            .map(|row| row.get_matlab_parameters(multi_axes))
            .collect();
    }
    pub fn display(&mut self, viewport: &Viewport) {
        for (index, row) in self.list.iter_mut().enumerate() {
            row.displayed = viewport.in_viewport(index);
        }
    }
    pub fn expand_fn_volume(&mut self, expanded: bool) {
        for row in &mut self.list {
            row.fn_volume_expanded = expanded;
        }
    }
    pub fn expand_fn_mod_particle(&mut self, expanded: bool) {
        for row in &mut self.list {
            row.fn_mod_particle_expanded = expanded;
        }
    }
    pub fn expand_init_motl(&mut self, expanded: bool) {
        for row in &mut self.list {
            row.init_motl_file_expanded = expanded;
        }
    }
    pub fn expand_tilt_range_multi_axes(&mut self, expanded: bool) {
        for row in &mut self.list {
            row.tilt_range_multi_axes_expanded = expanded;
        }
    }
    pub fn is_highlighted(&self) -> bool {
        self.list.iter().any(|row| row.highlighted)
    }
    pub fn get_highlighted_row_index(&self) -> isize {
        self.list
            .iter()
            .position(|row| row.highlighted)
            .map_or(-1, |index| index as isize)
    }
    pub fn get_max_row_text_size(&self, tilt_range_multi_axes: bool) -> usize {
        self.list
            .iter()
            .map(|row| row.get_text_size(tilt_range_multi_axes))
            .max()
            .unwrap_or(0)
    }
}

/// Java `Viewport`; its paging buttons remain a native GUI boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Viewport {
    pub table_size: usize,
    pub first_row: usize,
}
impl Viewport {
    pub fn new(table_size: usize) -> Self {
        Self {
            table_size,
            first_row: 0,
        }
    }
    pub fn in_viewport(&self, index: usize) -> bool {
        index >= self.first_row && index < self.first_row.saturating_add(self.table_size)
    }
    pub fn adjust_viewport(&mut self, index: usize) {
        if index < self.first_row {
            self.first_row = index;
        } else if self.table_size > 0 && index >= self.first_row + self.table_size {
            self.first_row = index + 1 - self.table_size;
        }
    }
    pub fn home_button_action(&mut self) {
        self.first_row = 0;
    }
    pub fn end_button_action(&mut self, size: usize) {
        self.first_row = size.saturating_sub(self.table_size);
    }
    pub fn down_button_action(&mut self) {
        self.first_row = self.first_row.saturating_add(1);
    }
    pub fn up_button_action(&mut self) {
        self.first_row = self.first_row.saturating_sub(1);
    }
}

/// Source-owned command identities, including `VTActionListener` dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VolumeTableAction {
    Insert,
    ReadTiltFile,
    Delete,
    OpenIn3dmod,
    MoveUp,
    MoveDown,
    Copy,
    VolNamesAreTemplates,
}

/// Java final `VolumeTable` state. Widget construction and manager calls are represented
/// by explicit state/notification boundaries rather than replaced with a different UI.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VolumeTable {
    pub row_list: RowList,
    pub volume_names_are_templates: bool,
    pub fn_volume_expanded: bool,
    pub fn_mod_particle_expanded: bool,
    pub init_motl_file_expanded: bool,
    pub tilt_range_multi_axes_expanded: bool,
    pub use_init_motl_file: bool,
    pub use_tilt_range: bool,
    pub tilt_range_multi_axes: bool,
    pub viewport: Viewport,
    pub horizontal_padding: i32,
    pub vertical_padding: i32,
    pub controls_enabled: VolumeTableControls,
    pub manager_set_param_file: bool,
    pub root_directory: PathBuf,
    pub messages: Vec<(String, String)>,
    pub packed: bool,
    pub repainted: bool,
    pub volume_table_size_changed: Vec<bool>,
    pub templates_changed: Vec<bool>,
    pub last_3dmod_row: Option<usize>,
}
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct VolumeTableControls {
    pub expand_fn_volume: bool,
    pub expand_fn_mod_particle: bool,
    pub expand_init_motl: bool,
    pub expand_tilt_range: bool,
    pub read_tilt_file: bool,
    pub open_in_3dmod: bool,
    pub delete: bool,
    pub move_up: bool,
    pub move_down: bool,
    pub copy: bool,
}
impl VolumeTable {
    /// `getInstance(PeetManager, PeetDialog)` plus construction and `addListeners()`.
    pub fn get_instance(table_size: usize, root_directory: impl Into<PathBuf>) -> Self {
        let mut table = Self {
            row_list: RowList::default(),
            volume_names_are_templates: false,
            fn_volume_expanded: false,
            fn_mod_particle_expanded: false,
            init_motl_file_expanded: false,
            tilt_range_multi_axes_expanded: false,
            use_init_motl_file: true,
            use_tilt_range: true,
            tilt_range_multi_axes: false,
            viewport: Viewport::new(table_size),
            horizontal_padding: 181,
            vertical_padding: 70,
            controls_enabled: VolumeTableControls::default(),
            manager_set_param_file: true,
            root_directory: root_directory.into(),
            messages: vec![],
            packed: false,
            repainted: false,
            volume_table_size_changed: vec![],
            templates_changed: vec![],
            last_3dmod_row: None,
        };
        table.create_panel();
        table.update_display();
        table.set_tool_tip_text();
        table.add_listeners();
        table
    }
    pub fn highlight_down_action_performed(&mut self) {
        let adjust = self
            .viewport
            .in_viewport(self.row_list.get_highlighted_row_index().max(0) as usize);
        let index = self.row_list.highlight_down();
        if index >= 0 && adjust && !self.viewport.in_viewport(index as usize) {
            if index == 0 {
                self.viewport.home_button_action()
            } else {
                self.viewport.down_button_action()
            }
        }
    }
    pub fn highlight_up_action_performed(&mut self) {
        let adjust = self
            .viewport
            .in_viewport(self.row_list.get_highlighted_row_index().max(0) as usize);
        let index = self.row_list.highlight_up();
        if index >= 0 && adjust && !self.viewport.in_viewport(index as usize) {
            if index as usize == self.row_list.size() - 1 {
                self.viewport.end_button_action(self.row_list.size())
            } else {
                self.viewport.up_button_action()
            }
        }
    }
    pub fn is_tilt_range_multi_axes(&self) -> bool {
        self.tilt_range_multi_axes
    }
    pub fn is_fn_volume_expanded(&self) -> bool {
        self.fn_volume_expanded
    }
    pub fn is_fn_mod_particle_expanded(&self) -> bool {
        self.fn_mod_particle_expanded
    }
    pub fn is_flg_vol_names_are_templates(&self) -> bool {
        self.volume_names_are_templates
    }
    pub fn is_init_motl_file_expanded(&self) -> bool {
        self.init_motl_file_expanded
    }
    pub fn get_tilt_range_multi_axes_label() -> String {
        format!("{TILT_RANGE_MULTI_AXES_HEADER1_LABEL} {TILT_RANGE_MULTI_AXES_HEADER2_LABEL}")
    }
    pub fn expand_fn_volume(&mut self, expanded: bool) {
        self.fn_volume_expanded = expanded;
        self.row_list.expand_fn_volume(expanded);
        self.pack();
    }
    pub fn expand_fn_mod_particle(&mut self, expanded: bool) {
        self.fn_mod_particle_expanded = expanded;
        self.row_list.expand_fn_mod_particle(expanded);
        self.pack();
    }
    pub fn expand_init_motl_file(&mut self, expanded: bool) {
        self.init_motl_file_expanded = expanded;
        self.row_list.expand_init_motl(expanded);
        self.pack();
    }
    pub fn expand_tilt_range_multi_axes(&mut self, expanded: bool) {
        self.tilt_range_multi_axes_expanded = expanded;
        self.row_list.expand_tilt_range_multi_axes(expanded);
        self.pack();
    }
    pub fn highlight(&mut self, _highlight: bool) {
        self.update_display();
    }
    pub fn size(&self) -> usize {
        self.row_list.size()
    }
    pub fn is_empty(&self) -> bool {
        self.row_list.is_empty()
    }
    /// Java overloaded `getParameters(PeetMetaData)`.
    pub fn get_parameters(&self, meta_data: &mut PeetMetaData) {
        self.row_list.get_parameters(meta_data);
    }
    /// Java `setParameters(ConstPeetMetaData)`.
    pub fn set_parameters(&mut self, meta_data: &PeetMetaData) {
        for row in &mut self.row_list.list {
            row.set_parameters(meta_data);
        }
    }
    /// Java `setParameters(MatlabParam,boolean,boolean,boolean,File)`.
    pub fn set_matlab_parameters(
        &mut self,
        matlab_param: &MatlabParam,
        use_init_motl: bool,
        use_tilt_range: bool,
        multi_axes: bool,
    ) {
        self.volume_names_are_templates = matlab_param.flg_vol_names_are_templates;
        let expanded = self.init_motl_file_expanded;
        for volume in &matlab_param.volumes {
            self.add_row_values(
                volume.fn_volume.clone(),
                volume.fn_mod_particle.clone(),
                volume.tilt_range_multi_axes.clone(),
            );
            let row = self.row_list.list.last_mut().unwrap();
            row.set_matlab_parameters(volume, use_init_motl, use_tilt_range, multi_axes);
            row.init_motl_file_expanded = expanded;
        }
        self.refresh_vertical_padding();
        self.refresh_horizontal_padding();
        self.done_setting_parameters();
        self.viewport.adjust_viewport(0);
        self.row_list.remove();
        self.row_list.display(&self.viewport);
        self.update_display();
        self.packed = true;
    }
    /// Java overloaded `getParameters(MatlabParam)`.
    pub fn get_matlab_parameters(&self, matlab_param: &mut MatlabParam) {
        matlab_param.flg_vol_names_are_templates = self.volume_names_are_templates;
        self.row_list
            .get_matlab_parameters(matlab_param, self.tilt_range_multi_axes);
    }
    pub fn is_incorrect_paths(&self) -> bool {
        !self.volume_names_are_templates
            && self
                .row_list
                .list
                .iter()
                .any(|row| row.is_incorrect_paths(&self.root_directory))
    }
    pub fn set_parameters_pending_metadata(&mut self) {
        self.row_list.meta_data_pending = true;
    }
    pub fn done_setting_parameters(&mut self) {
        self.row_list.meta_data_pending = false;
    }
    pub fn update_display_with_options(
        &mut self,
        use_init_motl_file: bool,
        use_tilt_range: bool,
        tilt_range_multi_axes: bool,
    ) {
        self.use_init_motl_file = use_init_motl_file;
        self.use_tilt_range = use_tilt_range;
        if use_tilt_range && self.tilt_range_multi_axes != tilt_range_multi_axes {
            self.tilt_range_multi_axes = tilt_range_multi_axes;
            self.display();
            self.row_list.remove();
            self.row_list.display(&self.viewport);
            self.refresh_horizontal_padding();
            self.repainted = true;
        }
        self.update_display();
    }
    pub fn msg_viewport_paged(&mut self) {
        self.row_list.remove();
        self.row_list.display(&self.viewport);
        self.packed = true;
    }
    pub fn action(
        &mut self,
        action: VolumeTableAction,
        _menu_options: Option<Run3dmodMenuOptions>,
    ) {
        match action {
            VolumeTableAction::Insert => self.insert_row(false),
            VolumeTableAction::ReadTiltFile => self.open_tilt_file(),
            VolumeTableAction::Delete => self.delete_row(self.row_list.get_highlighted_row_index()),
            VolumeTableAction::OpenIn3dmod => self.imod_volume(),
            VolumeTableAction::MoveUp => self.move_row_up(),
            VolumeTableAction::MoveDown => self.move_row_down(),
            VolumeTableAction::Copy => self.copy_row(false),
            VolumeTableAction::VolNamesAreTemplates => {
                self.templates_changed.push(self.volume_names_are_templates)
            }
        }
    }
    pub fn validate_run(&self, tilt_range_required: bool) -> Option<String> {
        self.row_list
            .validate_run(tilt_range_required, self.tilt_range_multi_axes)
    }
    pub fn pack(&mut self) {
        self.refresh_horizontal_padding();
        self.packed = true;
    }
    fn create_panel(&mut self) {
        self.build_table();
        self.refresh_vertical_padding();
        self.refresh_horizontal_padding();
    }
    fn build_table(&mut self) {
        self.display();
    }
    fn display(&mut self) {}
    fn refresh_horizontal_padding(&mut self) {
        let size = self
            .row_list
            .get_max_row_text_size(self.tilt_range_multi_axes);
        self.horizontal_padding = if size <= 50 {
            181
        } else {
            (181.0 - (size - 50) as f64 * 6.3).round().max(8.0) as i32
        };
    }
    fn refresh_vertical_padding(&mut self) {
        self.vertical_padding = (10 + (3_i32 - self.row_list.size() as i32) * 20).max(10);
    }
    fn imod_volume(&mut self) {
        let index = self.row_list.get_highlighted_row_index();
        if index < 0 {
            panic!("r3bVolume enabled when no row is highlighted");
        }
        self.last_3dmod_row = Some(index as usize);
    }
    fn delete_row(&mut self, row_index: isize) {
        self.row_list.remove();
        let index = self
            .row_list
            .delete((row_index >= 0).then_some(row_index as usize));
        if index >= 0 {
            self.row_list.highlight(index as usize);
            self.viewport.adjust_viewport(index as usize);
        }
        self.row_list.display(&self.viewport);
        self.refresh_vertical_padding();
        self.refresh_horizontal_padding();
        self.update_display();
        self.packed = true;
    }
    fn insert_row(&mut self, init: bool) {
        if !self.manager_set_param_file {
            self.messages.push((
                "Entry Error".into(),
                "Please set the directory and output fields before adding rows.".into(),
            ));
            return;
        }
        let index = self.row_list.get_highlighted_row_index();
        if index < 0 {
            self.add_row();
        } else {
            self.add_row_at(index as usize + 1);
        }
        self.viewport.adjust_viewport(self.row_list.size() - 1);
        self.row_list.remove();
        self.row_list.display(&self.viewport);
        self.refresh_vertical_padding();
        self.refresh_horizontal_padding();
        self.update_display();
        self.volume_table_size_changed.push(init);
        self.packed = true;
    }
    fn open_tilt_file(&mut self) {
        if self.row_list.get_highlighted_row_index() < 0 {
            self.messages
                .push(("Entry Error".into(), "Please highlight a row.".into()));
        }
    }
    fn add_row(&mut self) {
        let row = self.row_list.add();
        row.fn_volume_expanded = self.fn_volume_expanded;
        row.fn_mod_particle_expanded = self.fn_mod_particle_expanded;
    }
    fn add_row_at(&mut self, index: usize) {
        let row = self.row_list.add_at(index);
        row.fn_volume_expanded = self.fn_volume_expanded;
        row.fn_mod_particle_expanded = self.fn_mod_particle_expanded;
    }
    pub fn add_row_values(
        &mut self,
        fn_volume: impl Into<String>,
        fn_mod_particle: impl Into<String>,
        tilt_range_multi_axes: impl Into<String>,
    ) {
        let row = self
            .row_list
            .add_values(fn_volume, fn_mod_particle, tilt_range_multi_axes);
        row.fn_volume_expanded = self.fn_volume_expanded;
        row.fn_mod_particle_expanded = self.fn_mod_particle_expanded;
        row.tilt_range_multi_axes_expanded = self.tilt_range_multi_axes_expanded;
    }
    fn copy_row(&mut self, init: bool) {
        let index = self.row_list.get_highlighted_row_index();
        if index < 0 {
            return;
        }
        if let Some(row) = self.row_list.add_copy(index as usize) {
            row.fn_volume_expanded = self.fn_volume_expanded;
            row.fn_mod_particle_expanded = self.fn_mod_particle_expanded;
            row.init_motl_file_expanded = self.init_motl_file_expanded;
            row.tilt_range_multi_axes_expanded = self.tilt_range_multi_axes_expanded;
        }
        self.viewport.adjust_viewport(self.row_list.size() - 1);
        self.row_list.remove();
        self.row_list.display(&self.viewport);
        self.update_display();
        self.volume_table_size_changed.push(init);
        self.packed = true;
    }
    fn move_row_up(&mut self) {
        let index = self.row_list.get_highlighted_row_index();
        if index < 0 {
            return;
        }
        if index == 0 {
            self.messages.push((
                "Wrong Row".into(),
                "Can't move the row up.  Its at the top.".into(),
            ));
            return;
        }
        self.row_list.move_row_up(index as usize);
        self.viewport.adjust_viewport(index as usize - 1);
        self.row_list.remove();
        self.row_list.display(&self.viewport);
        self.refresh_vertical_padding();
        self.refresh_horizontal_padding();
        self.row_list.reindex(index as usize - 1);
        self.update_display();
        self.repainted = true;
    }
    fn move_row_down(&mut self) {
        let index = self.row_list.get_highlighted_row_index();
        if index < 0 {
            return;
        }
        if index as usize == self.row_list.size() - 1 {
            self.messages.push((
                "Wrong Row".into(),
                "Can't move the row down.  Its at the bottom.".into(),
            ));
            return;
        }
        self.row_list.move_row_down(index as usize);
        self.viewport.adjust_viewport(index as usize + 1);
        self.row_list.remove();
        self.row_list.display(&self.viewport);
        self.row_list.reindex(index as usize);
        self.update_display();
        self.repainted = true;
    }
    fn update_display(&mut self) {
        let enable = self.row_list.size() > 0;
        let highlighted = self.row_list.is_highlighted();
        self.controls_enabled = VolumeTableControls {
            expand_fn_volume: enable,
            expand_fn_mod_particle: enable,
            expand_init_motl: enable,
            expand_tilt_range: enable && self.use_tilt_range,
            read_tilt_file: enable && highlighted && self.use_tilt_range,
            open_in_3dmod: enable && highlighted,
            delete: enable && highlighted,
            move_up: enable && highlighted && self.row_list.get_highlighted_row_index() > 0,
            move_down: enable
                && highlighted
                && (self.row_list.get_highlighted_row_index() as usize) < self.row_list.size() - 1,
            copy: enable && highlighted,
        };
    }
    fn set_tool_tip_text(&mut self) {}
    fn add_listeners(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn insertion_move_copy_delete_and_paging_match_source_ordering() {
        let mut table = VolumeTable::get_instance(2, ".");
        table.action(VolumeTableAction::Insert, None);
        table.row_list.highlight(0);
        table.action(VolumeTableAction::Insert, None);
        assert_eq!(table.size(), 2);
        table.action(VolumeTableAction::Copy, None);
        assert_eq!(table.size(), 3);
        table.action(VolumeTableAction::MoveDown, None);
        assert_eq!(
            table
                .row_list
                .list
                .iter()
                .map(|row| row.index)
                .collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        table.action(VolumeTableAction::Delete, None);
        assert_eq!(table.size(), 2);
    }
    #[test]
    fn expansion_updates_rows_and_padding() {
        let mut table = VolumeTable::get_instance(3, ".");
        table.add_row_values("a".repeat(100), "model", "mask");
        table.expand_fn_volume(true);
        table.expand_tilt_range_multi_axes(true);
        assert!(table.row_list.list[0].fn_volume_expanded);
        assert!(table.row_list.list[0].tilt_range_multi_axes_expanded);
        assert_eq!(table.horizontal_padding, 8);
    }
    #[test]
    fn validation_and_template_path_exception_match_java() {
        let mut table = VolumeTable::get_instance(3, ".");
        table.add_row_values("missing.rec", "", "");
        assert!(table.is_incorrect_paths());
        table.volume_names_are_templates = true;
        assert!(!table.is_incorrect_paths());
        assert!(table.validate_run(false).is_none());
        assert!(table.validate_run(true).is_some());
    }
}
