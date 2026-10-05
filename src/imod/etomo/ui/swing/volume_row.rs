//! `IMOD/Etomo/src/etomo/ui/swing/VolumeRow.java`.
//!
//! One row of the PEET dialog's volume table: the tomogram, its particle model, its
//! initial motive list, and its tilt range (or missing wedge mask), each with a file
//! button.  An event dispatch thread object, created as `Rc<Self>`; the table owns it
//! (the row keeps a weak reference to the table).  The table's panel, layout and
//! shared constraints are the table's (`VolumeTable::with_constraints`).

use std::cell::{Cell as StdCell, RefCell};
use std::path::{Path, PathBuf};
use std::rc::{Rc, Weak};

use super::action_target::ActionTarget;
use super::cell::CellVirtual;
use super::column::Column;
use super::field_cell::FieldCell;
use super::file_button_cell::FileButtonCell;
use super::header_cell::HeaderCell;
use super::highlightable::Highlightable;
use super::highlighter_button::HighlighterButton;
use super::viewport::Viewport;
use super::volume_table::{self, VolumeTable};
use super::{file_chooser, ui_parameters::UIParameters};
use crate::imod::etomo::base_manager::{BaseManager, ManagerBrowsingDirectory};
use crate::imod::etomo::jdk::{FileFilter, GRID_BAG_REMAINDER, JComponent};
use crate::imod::etomo::process::imod_manager;
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::storage::matlab_param::MatlabParam;
use crate::imod::etomo::storage::model_file_filter::ModelFileFilter;
use crate::imod::etomo::storage::motl_file_filter::MotlFileFilter;
use crate::imod::etomo::storage::volume_file_filter::VolumeFileFilter;
use crate::imod::etomo::r#type::const_etomo_number::java_lang_string_matches_whitespace;
use crate::imod::etomo::r#type::const_peet_meta_data::ConstPeetMetaData;
use crate::imod::etomo::r#type::peet_meta_data::PeetMetaData;
use crate::imod::etomo::util::file_path::FilePath;

/// Java package-private `final class VolumeRow implements Highlightable`.
pub struct VolumeRow {
    /// Java private final `number = new HeaderCell()`.
    number: Rc<HeaderCell>,
    /// Java private final `btnHighlighter`.
    btn_highlighter: Rc<HighlighterButton>,
    /// Java private final `fnVolume`.
    fn_volume: Rc<FieldCell>,
    /// Java private final `fbFnVolume`.
    fb_fn_volume: Rc<FileButtonCell>,
    /// Java private final `fnModParticle`.
    fn_mod_particle: Rc<FieldCell>,
    /// Java private final `fbFnModParticle`.
    fb_fn_mod_particle: Rc<FileButtonCell>,
    /// Java private final `initMotlFile`.
    init_motl_file: Rc<FieldCell>,
    /// Java private final `fbInitMotlFile`.
    fb_init_motl_file: Rc<FileButtonCell>,
    /// Java private final `tiltRangeMin`.
    tilt_range_min: Rc<FieldCell>,
    /// Java private final `tiltRangeMax`.
    tilt_range_max: Rc<FieldCell>,
    /// Java private final `table` (the table owns the row).
    table: Weak<VolumeTable>,
    /// Java private final `panel` (the table's).
    panel: Rc<JComponent>,
    /// Java private final `manager`.
    manager: &'static dyn BaseManager,
    /// Java private final `tiltRangeMultiAxes`.
    tilt_range_multi_axes: Rc<FieldCell>,
    /// Java private final `fbTiltRangeMultiAxes`.
    fb_tilt_range_multi_axes: Rc<FileButtonCell>,
    /// Java private `imodIndex`, initially -1.
    imod_index: StdCell<i32>,
    /// Java private `index`.
    index: StdCell<i32>,
}

impl VolumeRow {
    /// Builds the row from its cells (the fields every Java constructor assigns),
    /// with `number.setText(String.valueOf(index + 1))` and the highlighter button.
    #[allow(clippy::too_many_arguments)]
    fn construct_with(
        manager: &'static dyn BaseManager,
        index: i32,
        table: &Rc<VolumeTable>,
        panel: &Rc<JComponent>,
        fn_volume: Rc<FieldCell>,
        fb_fn_volume: Rc<FileButtonCell>,
        fn_mod_particle: Rc<FieldCell>,
        fb_fn_mod_particle: Rc<FileButtonCell>,
        init_motl_file: Rc<FieldCell>,
        fb_init_motl_file: Rc<FileButtonCell>,
        tilt_range_min: Rc<FieldCell>,
        tilt_range_max: Rc<FieldCell>,
        tilt_range_multi_axes: Rc<FieldCell>,
        fb_tilt_range_multi_axes: Rc<FileButtonCell>,
    ) -> Rc<VolumeRow> {
        Rc::new_cyclic(|self_ref: &Weak<VolumeRow>| {
            let number = HeaderCell::new_void();
            number.set_text_string(Some(&(index + 1).to_string()));
            let parent: Weak<dyn Highlightable> = self_ref.clone();
            let group: Weak<dyn Highlightable> = Rc::downgrade(table) as Weak<dyn Highlightable>;
            VolumeRow {
                number,
                btn_highlighter: HighlighterButton::get_instance(parent, Some(group)),
                fn_volume,
                fb_fn_volume,
                fn_mod_particle,
                fb_fn_mod_particle,
                init_motl_file,
                fb_init_motl_file,
                tilt_range_min,
                tilt_range_max,
                table: Rc::downgrade(table),
                panel: panel.clone(),
                manager,
                tilt_range_multi_axes,
                fb_tilt_range_multi_axes,
                imod_index: StdCell::new(-1),
                index: StdCell::new(index),
            }
        })
    }

    /// Java static `getInstance(BaseManager, int, VolumeTable, JPanel, GridBagLayout,
    /// GridBagConstraints, VolumeFileFilter)`.
    pub fn get_instance(
        manager: &'static dyn BaseManager,
        index: i32,
        table: &Rc<VolumeTable>,
        panel: &Rc<JComponent>,
        volume_file_filter: &Rc<VolumeFileFilter>,
    ) -> Rc<VolumeRow> {
        let instance = VolumeRow::new_index(manager, index, table, panel, volume_file_filter);
        instance.add_action_targets();
        instance.set_tooltips();
        instance
    }

    /// Java static `getInstance(BaseManager, File, File, File, int, VolumeTable,
    /// JPanel, GridBagLayout, GridBagConstraints, VolumeFileFilter)`.
    #[allow(clippy::too_many_arguments)]
    pub fn get_instance_files(
        manager: &'static dyn BaseManager,
        fn_volume: Option<&Path>,
        fn_mod_particle: Option<&Path>,
        tilt_range_multi_axes_file: Option<&Path>,
        index: i32,
        table: &Rc<VolumeTable>,
        panel: &Rc<JComponent>,
        volume_file_filter: &Rc<VolumeFileFilter>,
    ) -> Rc<VolumeRow> {
        let instance = VolumeRow::new_index(manager, index, table, panel, volume_file_filter);
        // The File constructor: setValue(fnVolume, fnVolumeFile) etc. after each cell
        // is built (no other cell reads them).
        Self::set_value_file(&instance.fn_volume, fn_volume);
        Self::set_value_file(&instance.fn_mod_particle, fn_mod_particle);
        Self::set_value_file(&instance.tilt_range_multi_axes, tilt_range_multi_axes_file);
        instance.add_action_targets();
        instance.set_tooltips();
        instance
    }

    /// Java static `getInstance(BaseManager, String, String, String, int, VolumeTable,
    /// JPanel, GridBagLayout, GridBagConstraints, VolumeFileFilter)`.
    #[allow(clippy::too_many_arguments)]
    pub fn get_instance_strings(
        manager: &'static dyn BaseManager,
        fn_volume: Option<&str>,
        fn_mod_particle: Option<&str>,
        tilt_range_multi_axes_file: Option<&str>,
        index: i32,
        table: &Rc<VolumeTable>,
        panel: &Rc<JComponent>,
        volume_file_filter: &Rc<VolumeFileFilter>,
    ) -> Rc<VolumeRow> {
        let instance = VolumeRow::new_index(manager, index, table, panel, volume_file_filter);
        // The String constructor: setValue(fnVolume, fnVolumeFile) etc. after each
        // cell is built (no other cell reads them).
        Self::set_value_string(&instance.fn_volume, fn_volume);
        Self::set_value_string(&instance.fn_mod_particle, fn_mod_particle);
        Self::set_value_string(&instance.tilt_range_multi_axes, tilt_range_multi_axes_file);
        instance.add_action_targets();
        instance.set_tooltips();
        instance
    }

    /// Java static `getInstance(VolumeRow, int)`.
    pub fn get_instance_copy(volume_row: &VolumeRow, index: i32) -> Rc<VolumeRow> {
        let instance = VolumeRow::new_copy(volume_row, index);
        instance.add_action_targets();
        instance.set_tooltips();
        instance
    }

    /// Java private `VolumeRow(BaseManager, int, VolumeTable, JPanel, GridBagLayout,
    /// GridBagConstraints, VolumeFileFilter)`.
    fn new_index(
        manager: &'static dyn BaseManager,
        index: i32,
        table: &Rc<VolumeTable>,
        panel: &Rc<JComponent>,
        volume_file_filter: &Rc<VolumeFileFilter>,
    ) -> Rc<VolumeRow> {
        // `FileButtonCell.getInstance(manager)`, `setBrowsingDirectory(manager)`,
        // `setFileFilter(filter)` for each of the four file buttons.
        let file_button = |filter: Rc<dyn FileFilter>| {
            let file_button = FileButtonCell::get_instance_base_manager(manager);
            file_button.set_browsing_directory(Some(Rc::new(ManagerBrowsingDirectory(manager))));
            file_button.set_file_filter_file_filter(Some(filter));
            file_button
        };
        let root_dir = manager.get_property_user_dir();
        let fn_volume = FieldCell::get_expandable_instance(root_dir.as_deref());
        let fb_fn_volume = file_button(volume_file_filter.clone() as Rc<dyn FileFilter>);
        let fn_mod_particle = FieldCell::get_expandable_instance(root_dir.as_deref());
        let fb_fn_mod_particle = file_button(Rc::new(ModelFileFilter::new()));
        let init_motl_file = FieldCell::get_expandable_instance(root_dir.as_deref());
        let fb_init_motl_file = file_button(Rc::new(MotlFileFilter::new()));
        let tilt_range_min = FieldCell::get_editable_matlab_instance();
        let tilt_range_max = FieldCell::get_editable_matlab_instance();
        let tilt_range_multi_axes = FieldCell::get_expandable_instance(root_dir.as_deref());
        let fb_tilt_range_multi_axes =
            file_button(volume_file_filter.clone() as Rc<dyn FileFilter>);
        VolumeRow::construct_with(
            manager,
            index,
            table,
            panel,
            fn_volume,
            fb_fn_volume,
            fn_mod_particle,
            fb_fn_mod_particle,
            init_motl_file,
            fb_init_motl_file,
            tilt_range_min,
            tilt_range_max,
            tilt_range_multi_axes,
            fb_tilt_range_multi_axes,
        )
    }

    /// Java private `VolumeRow(VolumeRow, int)`.
    fn new_copy(volume_row: &VolumeRow, index: i32) -> Rc<VolumeRow> {
        let table = volume_row.table();
        VolumeRow::construct_with(
            volume_row.manager,
            index,
            &table,
            &volume_row.panel,
            FieldCell::get_instance(&volume_row.fn_volume),
            FileButtonCell::get_instance_file_button_cell(&volume_row.fb_fn_volume),
            FieldCell::get_instance(&volume_row.fn_mod_particle),
            FileButtonCell::get_instance_file_button_cell(&volume_row.fb_fn_mod_particle),
            FieldCell::get_instance(&volume_row.init_motl_file),
            FileButtonCell::get_instance_file_button_cell(&volume_row.fb_init_motl_file),
            FieldCell::get_instance(&volume_row.tilt_range_min),
            FieldCell::get_instance(&volume_row.tilt_range_max),
            FieldCell::get_instance(&volume_row.tilt_range_multi_axes),
            FileButtonCell::get_instance_file_button_cell(&volume_row.fb_tilt_range_multi_axes),
        )
    }

    /// Java field read `table`.
    fn table(&self) -> Rc<VolumeTable> {
        self.table
            .upgrade()
            .expect("the volume table owns its rows")
    }

    /// Java private `addActionTargets()`.
    fn add_action_targets(&self) {
        self.fb_fn_volume
            .set_action_target(Some(self.fn_volume.clone() as Rc<dyn ActionTarget>));
        self.fb_fn_mod_particle
            .set_action_target(Some(self.fn_mod_particle.clone() as Rc<dyn ActionTarget>));
        self.fb_init_motl_file
            .set_action_target(Some(self.init_motl_file.clone() as Rc<dyn ActionTarget>));
        self.fb_tilt_range_multi_axes.set_action_target(Some(
            self.tilt_range_multi_axes.clone() as Rc<dyn ActionTarget>
        ));
    }

    /// Java package-private `setNames()`.
    pub fn set_names(&self) {
        let table = self.table();
        self.btn_highlighter.set_headers(
            Some(volume_table::LABEL),
            &self.number,
            &table.get_volume_number_header_cell(),
        );
        self.set_headers(
            &self.fn_volume,
            &self.fb_fn_volume,
            &table.get_fn_volume_header_cell(),
        );
        self.set_headers(
            &self.fn_mod_particle,
            &self.fb_fn_mod_particle,
            &table.get_fn_mod_particle_header_cell(),
        );
        self.set_headers(
            &self.init_motl_file,
            &self.fb_init_motl_file,
            &table.get_init_motl_file_header_cell(),
        );
        self.set_headers(
            &self.tilt_range_multi_axes,
            &self.fb_tilt_range_multi_axes,
            &table.get_tilt_range_multi_axes_header_cell(),
        );
        self.fb_init_motl_file.set_label(Some(&format!(
            "{} {}",
            volume_table::INIT_MOTL_FILE_HEADER1,
            volume_table::INIT_MOTL_FILE_HEADER2
        )));
        self.tilt_range_min.set_headers(
            Some(volume_table::LABEL),
            &self.number,
            &table.get_tilt_range_header_cell(),
        );
        self.tilt_range_max.set_headers(
            Some(volume_table::LABEL),
            &self.number,
            &table.get_tilt_range_header_cell(),
        );
    }

    /// Java package-private `getTextSize(boolean)`.  Return the text size, or an
    /// estimate or the minimum field text width of the three changeable field
    /// (volume, model, and MOTL).
    pub fn get_text_size(&self, is_tilt_range_multi_axes: bool) -> i32 {
        let length =
            |cell: &FieldCell| cell.get_value().unwrap_or_default().encode_utf16().count() as i32;
        length(&self.fn_volume).max(6)
            + length(&self.fn_mod_particle).max(5)
            + length(&self.init_motl_file).max(5)
            + if is_tilt_range_multi_axes {
                length(&self.tilt_range_multi_axes).max(5)
            } else {
                0
            }
    }

    /// Java package-private `setHeaders(FieldCell, FileButtonCell, HeaderCell)`.
    fn set_headers(
        &self,
        field_cell: &FieldCell,
        file_button_cell: &FileButtonCell,
        header_cell: &Rc<HeaderCell>,
    ) {
        field_cell.set_headers(Some(volume_table::LABEL), &self.number, header_cell);
        file_button_cell.set_headers(Some(volume_table::LABEL), &self.number, header_cell);
    }

    /// Java package-private `setHighlighterSelected(boolean)`.
    pub fn set_highlighter_selected(&self, select: bool) {
        self.btn_highlighter.set_selected(select);
    }

    /// Java package-private `remove()`.
    pub fn remove(&self) {
        self.number.remove();
        self.btn_highlighter.remove();
        self.fn_volume.remove();
        self.fb_fn_volume.remove();
        self.fn_mod_particle.remove();
        self.fb_fn_mod_particle.remove();
        self.init_motl_file.remove();
        self.fb_init_motl_file.remove();
        self.tilt_range_min.remove();
        self.tilt_range_max.remove();
        self.tilt_range_multi_axes.remove();
        self.fb_tilt_range_multi_axes.remove();
    }

    /// Java package-private `display(int, Viewport)`.
    pub fn display(&self, index: i32, viewport: &Viewport) {
        if !viewport.in_viewport(index) {
            return;
        }
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
        table.with_constraints(|constraints| constraints.gridwidth = 1);
        add(&*self.number, self.number.get_component());
        table.with_constraints(|constraints| {
            self.btn_highlighter
                .add(panel, table.get_layout(), constraints);
        });
        add(&*self.fn_volume, self.fn_volume.get_component());
        add(&*self.fb_fn_volume, self.fb_fn_volume.get_component());
        add(&*self.fn_mod_particle, self.fn_mod_particle.get_component());
        add(
            &*self.fb_fn_mod_particle,
            self.fb_fn_mod_particle.get_component(),
        );
        add(&*self.init_motl_file, self.init_motl_file.get_component());
        add(
            &*self.fb_init_motl_file,
            self.fb_init_motl_file.get_component(),
        );
        if !table.is_tilt_range_multi_axes() {
            add(&*self.tilt_range_min, self.tilt_range_min.get_component());
            table.with_constraints(|constraints| constraints.gridwidth = GRID_BAG_REMAINDER);
            add(&*self.tilt_range_max, self.tilt_range_max.get_component());
        } else {
            add(
                &*self.tilt_range_multi_axes,
                self.tilt_range_multi_axes.get_component(),
            );
            table.with_constraints(|constraints| constraints.gridwidth = GRID_BAG_REMAINDER);
            add(
                &*self.fb_tilt_range_multi_axes,
                self.fb_tilt_range_multi_axes.get_component(),
            );
        }
    }

    /// Java package-private `expandFnVolume(boolean)`.
    pub fn expand_fn_volume(&self, expanded: bool) {
        self.fn_volume.expand(expanded);
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

    /// Java package-private `expandFnModParticle(boolean)`.
    pub fn expand_fn_mod_particle(&self, expanded: bool) {
        self.fn_mod_particle.expand(expanded);
    }

    /// Java package-private `expandInitMotlFile(boolean)`.
    pub fn expand_init_motl_file(&self, expanded: bool) {
        self.init_motl_file.expand(expanded);
    }

    /// Java package-private `expandTiltRangeMultiAxes(boolean)`.
    pub fn expand_tilt_range_multi_axes(&self, expanded: bool) {
        self.tilt_range_multi_axes.expand(expanded);
    }

    /// Java package-private `getParameters(PeetMetaData)`.
    pub fn get_parameters_meta_data(&self, meta_data: &PeetMetaData) {
        let index = self.index.get();
        meta_data.set_init_motl_file(self.init_motl_file.get_expanded_value().as_deref(), index);
        meta_data.set_tilt_range_min(self.tilt_range_min.get_value().as_deref(), index);
        meta_data.set_tilt_range_max(self.tilt_range_max.get_value().as_deref(), index);
        meta_data.set_tilt_range_multi_axes_file(
            self.tilt_range_multi_axes.get_expanded_value().as_deref(),
            index,
        );
    }

    /// Java package-private `convertCopiedPaths(String)`.  Make the copied paths
    /// relative to this dataset, preserving the location of the files that the old
    /// dataset was using.  If a file path is absolute, don't change it.
    pub fn convert_copied_paths(&self, orig_dataset_dir: &str) {
        let property_user_dir = self.manager.get_property_user_dir();
        for cell in [
            &self.fn_volume,
            &self.fn_mod_particle,
            &self.init_motl_file,
            &self.tilt_range_multi_axes,
        ] {
            if !cell.is_empty() {
                cell.set_value_string(
                    FilePath::get_rerooted_relative_path(
                        Some(orig_dataset_dir),
                        property_user_dir.as_deref(),
                        cell.get_expanded_value().as_deref(),
                    )
                    .as_deref(),
                );
            }
        }
    }

    /// Java package-private `isIncorrectPaths()`.  Returns true if one or more paths
    /// are incorrect.
    pub fn is_incorrect_paths(&self) -> bool {
        let is_incorrect_path = |cell: &FieldCell| {
            !cell.is_empty()
                && !FilePath::build_absolute_file_string_string(
                    self.manager.get_property_user_dir().as_deref(),
                    &cell.get_expanded_value().unwrap_or_default(),
                )
                .exists()
        };
        is_incorrect_path(&self.fn_volume)
            || is_incorrect_path(&self.fn_mod_particle)
            || is_incorrect_path(&self.init_motl_file)
            || is_incorrect_path(&self.tilt_range_multi_axes)
    }

    /// Java package-private `fixIncorrectPaths(boolean)`.
    pub fn fix_incorrect_paths(&self, choose_path_every_row: bool) -> bool {
        let table = self.table();
        let is_incorrect_path = |cell: &FieldCell| {
            !cell.is_empty()
                && !FilePath::build_absolute_file_string_string(
                    self.manager.get_property_user_dir().as_deref(),
                    &cell.get_expanded_value().unwrap_or_default(),
                )
                .exists()
        };
        if is_incorrect_path(&self.fn_volume)
            && !self.fix_incorrect_path(
                &self.fn_volume,
                choose_path_every_row,
                table.is_fn_volume_expanded(),
                self.fb_fn_volume.get_file_filter(),
            )
        {
            return false;
        }
        if is_incorrect_path(&self.fn_mod_particle)
            && !self.fix_incorrect_path(
                &self.fn_mod_particle,
                false,
                table.is_fn_mod_particle_expanded(),
                self.fb_fn_mod_particle.get_file_filter(),
            )
        {
            return false;
        }
        if is_incorrect_path(&self.init_motl_file)
            && !self.fix_incorrect_path(
                &self.init_motl_file,
                false,
                table.is_init_motl_file_expanded(),
                self.fb_init_motl_file.get_file_filter(),
            )
        {
            return false;
        }
        if is_incorrect_path(&self.tilt_range_multi_axes)
            && !self.fix_incorrect_path(
                &self.tilt_range_multi_axes,
                false,
                table.is_init_motl_file_expanded(),
                self.fb_tilt_range_multi_axes.get_file_filter(),
            )
        {
            return false;
        }
        true
    }

    /// Java private `fixIncorrectPath(FieldCell, boolean, boolean, FileFilter)`.
    /// Returns false if the user cancels the file selector.
    fn fix_incorrect_path(
        &self,
        field_cell: &FieldCell,
        choose_path: bool,
        expand: bool,
        file_filter: Option<Rc<dyn FileFilter>>,
    ) -> bool {
        let table = self.table();
        let mut new_file: Option<PathBuf> = None;
        while new_file.as_ref().is_none_or(|file| !file.exists()) {
            // Have the user choose the location of the file if they haven't chosen
            // before or they want to choose most of the files individuallly, otherwise
            // just use the current correctPath.
            if table.is_correct_path_null()
                || choose_path
                || new_file.as_ref().is_some_and(|file| !file.exists())
            {
                let file_chooser = table.get_file_chooser_instance();
                file_chooser.set_selected_file(Some(&FilePath::build_absolute_file_string_string(
                    self.manager.get_property_user_dir().as_deref(),
                    &field_cell.get_expanded_value().unwrap_or_default(),
                )));
                // Swing layout: fileChooser.setPreferredSize(
                // UIParameters.getInstance().getFileChooserDimension()).
                let _ = UIParameters::get_instance_void().get_file_chooser_dimension();
                file_chooser.set_file_selection_mode(file_chooser::FILES_ONLY);
                file_chooser.set_file_filter(file_filter.clone());
                let return_val = file_chooser.show_open_dialog(Some(&table.get_container()));
                if return_val != file_chooser::APPROVE_OPTION {
                    return false;
                }
                new_file = file_chooser.get_selected_file();
                if let Some(file) = new_file.as_ref().filter(|file| file.exists()) {
                    table.set_correct_path(
                        file.parent()
                            .map(|parent| parent.to_string_lossy().into_owned()),
                    );
                    Self::set_value_file(field_cell, Some(file));
                    field_cell.expand(expand);
                }
            } else if !table.is_correct_path_null() {
                let file = Path::new(&table.get_correct_path().unwrap_or_default())
                    .join(field_cell.get_contracted_value().unwrap_or_default());
                if file.exists() {
                    Self::set_value_file(field_cell, Some(&file));
                    field_cell.expand(expand);
                }
                new_file = Some(file);
            }
        }
        true
    }

    /// Java package-private `setParameters(ConstPeetMetaData)`.  Always set metaData
    /// before the functional data from the prm file, since that may override the
    /// metaData.
    pub fn set_parameters_meta_data(&self, meta_data: Option<&dyn ConstPeetMetaData>) {
        let Some(meta_data) = meta_data else {
            return;
        };
        let index = self.index.get();
        Self::set_value_string(
            &self.init_motl_file,
            meta_data.get_init_motl_file(index).as_deref(),
        );
        self.set_tilt_range_min(meta_data.get_tilt_range_min(index).as_deref());
        self.set_tilt_range_max(meta_data.get_tilt_range_max(index).as_deref());
        Self::set_value_string(
            &self.tilt_range_multi_axes,
            meta_data.get_tilt_range_multi_axes_file(index).as_deref(),
        );
    }

    /// Java package-private `getParameters(MatlabParam, boolean)`.
    pub fn get_parameters_matlab_param(
        &self,
        matlab_param_file: &mut MatlabParam,
        is_tilt_range_multi_axes: bool,
    ) {
        let volume = matlab_param_file.get_volume(self.index.get());
        volume.set_fn_volume(self.fn_volume.get_expanded_value().as_deref());
        volume.set_fn_mod_particle(self.fn_mod_particle.get_expanded_value().as_deref());
        volume.set_init_motl(self.init_motl_file.get_expanded_value().as_deref());
        if !is_tilt_range_multi_axes {
            volume.set_tilt_range_start(self.tilt_range_min.get_value().as_deref());
            volume.set_tilt_range_end(self.tilt_range_max.get_value().as_deref());
        } else {
            volume.set_tilt_range_multi_axes(
                self.tilt_range_multi_axes.get_expanded_value().as_deref(),
            );
        }
    }

    /// Java package-private `setParameters(MatlabParam, boolean, boolean, boolean)`.
    pub fn set_parameters_matlab_param(
        &self,
        matlab_param: &mut MatlabParam,
        use_init_motl_file: bool,
        use_tilt_range: bool,
        is_tilt_range_multi_axes: bool,
    ) {
        let volume = matlab_param.get_volume(self.index.get());
        if use_init_motl_file {
            Self::set_value_string(
                &self.init_motl_file,
                volume.get_init_motl_string().as_deref(),
            );
        }
        if use_tilt_range {
            if !is_tilt_range_multi_axes {
                self.set_tilt_range_min(volume.get_tilt_range_start().as_deref());
                self.set_tilt_range_max(volume.get_tilt_range_end().as_deref());
            } else {
                Self::set_value_string(
                    &self.tilt_range_multi_axes,
                    volume.get_tilt_range_multi_axes_string().as_deref(),
                );
            }
        }
    }

    /// Java package-private `clearInitMotlFile()`.
    pub fn clear_init_motl_file(&self) {
        self.init_motl_file.set_value_void();
    }

    /// Java package-private `registerInitMotlFileColumn(Column)`.
    pub fn register_init_motl_file_column(&self, column: &Column) {
        column.add(self.init_motl_file.clone() as Rc<dyn CellVirtual>);
        column.add(self.fb_init_motl_file.clone() as Rc<dyn CellVirtual>);
    }

    /// Java package-private `registerTiltRangeColumn(Column)`.
    pub fn register_tilt_range_column(&self, column: &Column) {
        column.add(self.tilt_range_min.clone() as Rc<dyn CellVirtual>);
        column.add(self.tilt_range_max.clone() as Rc<dyn CellVirtual>);
        column.add(self.tilt_range_multi_axes.clone() as Rc<dyn CellVirtual>);
        column.add(self.fb_tilt_range_multi_axes.clone() as Rc<dyn CellVirtual>);
    }

    /// Java package-private `imodVolume(Run3dmodMenuOptions)`.
    pub fn imod_volume(&self, menu_options: Option<Run3dmodMenuOptions>) {
        self.imod_index.set(self.manager.imod_open_with_model(
            Some(imod_manager::TOMOGRAM_KEY),
            self.imod_index.get(),
            self.fn_volume.get_expanded_value().as_deref(),
            self.fn_mod_particle.get_expanded_value().as_deref(),
            menu_options,
        ));
    }

    /// Java package-private `validateRun(boolean, boolean)`.  Validate for running.
    /// Returns error message (null if valid).
    pub fn validate_run(
        &self,
        tilt_range_required: bool,
        is_tilt_range_multi_axes: bool,
    ) -> Option<String> {
        let number = self.number.get_text().unwrap_or_default();
        if self.fn_mod_particle.is_empty() {
            return Some(format!(
                "{}:  In row {number}, {} must not be empty.",
                volume_table::LABEL,
                volume_table::FN_MOD_PARTICLE_HEADER1
            ));
        }
        if tilt_range_required {
            if !is_tilt_range_multi_axes {
                if self.tilt_range_min.is_empty() || self.tilt_range_max.is_empty() {
                    return Some(format!(
                        "{}:  In row {number}, {} is required.",
                        volume_table::LABEL,
                        volume_table::TILT_RANGE_HEADER1_LABEL
                    ));
                }
            } else if self.tilt_range_multi_axes.is_empty() {
                return Some(format!(
                    "{}:  In row {number}, {} is required.",
                    volume_table::LABEL,
                    VolumeTable::get_tilt_range_multi_axes_label()
                ));
            }
        }
        None
    }

    /// Java private `setValue(FieldCell, String)`.  Sets the contracted and expanded
    /// values of the fieldCell while preserving the filePath string.
    fn set_value_string(field_cell: &FieldCell, file_path: Option<&str>) {
        // Don't override existing values with null value.
        let Some(file_path) = file_path else {
            return;
        };
        if java_lang_string_matches_whitespace(file_path) {
            return;
        }
        // Preserve the text of the filePath.
        field_cell.set_value_string(Some(file_path));
    }

    /// Java private `setValue(FieldCell, File)`.  Sets the contracted and expanded
    /// values of the fieldCell with the file name and a relative path from
    /// propertyUserDir to the file.
    fn set_value_file(field_cell: &FieldCell, file: Option<&Path>) {
        // Don't override existing values with null value.
        let Some(file) = file else {
            return;
        };
        field_cell.set_value_file(Some(file));
    }

    /// Java package-private `setInitMotlFile(String)`.
    pub fn set_init_motl_file_string(&self, init_motl_file: Option<&str>) {
        Self::set_value_string(&self.init_motl_file, init_motl_file);
    }

    /// Java package-private `setInitMotlFile(File)`.
    pub fn set_init_motl_file_file(&self, init_motl_file: Option<&Path>) {
        Self::set_value_file(&self.init_motl_file, init_motl_file);
    }

    /// Java package-private `getExpandedInitMotlFile()`.
    pub fn get_expanded_init_motl_file(&self) -> Option<String> {
        if self.init_motl_file.is_empty() {
            return None;
        }
        self.init_motl_file.get_expanded_value()
    }

    /// Java package-private `getFnVolumeFile()`.
    pub fn get_fn_volume_file(&self) -> Option<PathBuf> {
        if self.fn_volume.is_empty() {
            return None;
        }
        Some(FilePath::build_absolute_file_string_string(
            self.manager.get_property_user_dir().as_deref(),
            &self.fn_volume.get_expanded_value().unwrap_or_default(),
        ))
    }

    /// Java package-private `getFnModParticleFile()`.
    pub fn get_fn_mod_particle_file(&self) -> Option<PathBuf> {
        if self.fn_mod_particle.is_empty() {
            return None;
        }
        Some(FilePath::build_absolute_file_string_string(
            self.manager.get_property_user_dir().as_deref(),
            &self
                .fn_mod_particle
                .get_expanded_value()
                .unwrap_or_default(),
        ))
    }

    /// Java package-private `setFnModParticle(File)`.
    pub fn set_fn_mod_particle(&self, input: Option<&Path>) {
        Self::set_value_file(&self.fn_mod_particle, input);
    }

    /// Java package-private `setTiltRangeMin(String)`.
    pub fn set_tilt_range_min(&self, input: Option<&str>) {
        let Some(input) = input else {
            return;
        };
        self.tilt_range_min.set_value_string(Some(input));
    }

    /// Java package-private `getTiltRangeMin()`.
    pub fn get_tilt_range_min(&self) -> Option<String> {
        self.tilt_range_min.get_value()
    }

    /// Java package-private `getTiltRangeMax()`.
    pub fn get_tilt_range_max(&self) -> Option<String> {
        self.tilt_range_max.get_value()
    }

    /// Java package-private `setTiltRangeMax(String)`.
    pub fn set_tilt_range_max(&self, input: Option<&str>) {
        let Some(input) = input else {
            return;
        };
        self.tilt_range_max.set_value_string(Some(input));
    }

    /// Java package-private `isHighlighted()`.
    pub fn is_highlighted(&self) -> bool {
        self.btn_highlighter.is_highlighted()
    }

    /// Java private `setTooltips()`.
    fn set_tooltips(&self) {
        self.fn_volume
            .set_tool_tip_text(Some("The filename of the tomogram in MRC format."));
        self.fb_fn_volume
            .set_tool_tip_text(Some("Select a filename of the tomogram in MRC format."));
        self.fn_mod_particle.set_tool_tip_text(Some(
            "The filename of the IMOD model specifying particle positions in the tomogram.",
        ));
        self.fb_fn_mod_particle.set_tool_tip_text(Some(
            "Select a filename of the IMOD model specifying particle positions in the tomogram.",
        ));
        self.init_motl_file.set_tool_tip_text(Some(
            "The name of a .csv file containing an initial motive list with orientations and shifts.",
        ));
        self.fb_init_motl_file.set_tool_tip_text(Some(
            "Select a .csv file with initial orientations and shifts",
        ));
        let tooltip = " tilt angle (in degrees) used during image acquisition for this tomogram.  Used only if missing wedge compensation is enabled.";
        self.tilt_range_min
            .set_tool_tip_text(Some(&format!("The minimum{tooltip}")));
        self.tilt_range_max
            .set_tool_tip_text(Some(&format!("The maximum{tooltip}")));
        self.tilt_range_multi_axes.set_tool_tip_text(Some(
            "A binary mask file in MRC format with 0's and 1's indicating missing and valid regions in Fourier space, respectively, for this volume.  The mask should be cubical, with an even number of voxels, at least as large as the largest dimension of the reference, along each edge.",
        ));
    }
}

impl Highlightable for VolumeRow {
    /// Java `highlight(boolean)`.
    fn highlight(&self, highlight: bool) {
        self.fn_volume.set_highlight(highlight);
        self.fn_mod_particle.set_highlight(highlight);
        self.init_motl_file.set_highlight(highlight);
        self.tilt_range_multi_axes.set_highlight(highlight);
        self.tilt_range_min.set_highlight(highlight);
        self.tilt_range_max.set_highlight(highlight);
        self.tilt_range_multi_axes.set_highlight(highlight);
    }
}
