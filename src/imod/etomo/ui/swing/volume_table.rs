//! `IMOD/Etomo/src/etomo/ui/swing/VolumeTable.java`.
//!
//! The PEET dialog's volume table: one `VolumeRow` per tomogram, two header rows,
//! side buttons (Up, Down, Insert, Delete, Dup) and bottom controls ("File names are
//! templates", "Open in 3dmod", "Read tilt file").  An event dispatch thread object,
//! created as `Rc<Self>` by [`VolumeTable::get_instance`]; it keeps a weak reference
//! to its `PeetDialog`.  `VolumeTable extends HighlightableTable implements Expandable,
//! Run3dmodButtonContainer, Viewable`: the superclass is the `base` field.
//!
//! The rows share the table's panel, `GridBagLayout` and `GridBagConstraints` as in
//! Java; the constraints sit in a `RefCell` the rows change through
//! [`VolumeTable::with_constraints`].

use std::cell::{Cell as StdCell, RefCell};
use std::path::Path;
use std::rc::{Rc, Weak};

use super::cell::CellVirtual;
use super::check_box::CheckBox;
use super::column::Column;
use super::deferred_3dmod_button::Deferred3dmodButton;
use super::etched_border::EtchedBorder;
use super::expand_button::{self, ExpandButton};
use super::expandable::Expandable;
use super::file_chooser::{self, FileChooser};
use super::global_expand_button::GlobalExpandButton;
use super::header_cell::HeaderCell;
use super::highlightable::Highlightable;
use super::highlightable_table::{HighlightableTable, HighlightableTableVirtual};
use super::peet_dialog::PeetDialog;
use super::run_3dmod_button_container::Run3dmodButtonContainer;
use super::run_3dmod_single_line_button::Run3dmodSingleLineButton;
use super::single_line_button::SingleLineButton;
use super::ui_harness;
use super::ui_parameters::UIParameters;
use super::viewable::Viewable;
use super::viewport::Viewport;
use super::volume_row::VolumeRow;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::jdk::{
    ActionEvent, ActionListener, FileFilter, GRID_BAG_BOTH, GRID_BAG_CENTER, GRID_BAG_REMAINDER,
    GridBagConstraints, GridBagLayout, JComponent,
};
use crate::imod::etomo::peet_manager::PeetManager;
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::storage::autodoc::autodoc_factory;
use crate::imod::etomo::storage::log_file::LogFileError;
use crate::imod::etomo::storage::matlab_param::{self, MatlabParam};
use crate::imod::etomo::storage::tilt_file::TiltFile;
use crate::imod::etomo::storage::tilt_file_filter::TiltFileFilter;
use crate::imod::etomo::storage::tilt_log::TiltLog;
use crate::imod::etomo::storage::tilt_log_file_filter::TiltLogFileFilter;
use crate::imod::etomo::storage::volume_file_filter::VolumeFileFilter;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_peet_meta_data::ConstPeetMetaData;
use crate::imod::etomo::r#type::etomo_autodoc;
use crate::imod::etomo::r#type::peet_meta_data::PeetMetaData;
use crate::imod::etomo::ui::shared_strings;
use crate::imod::etomo::util::utilities;

/// Java package-private static final `FN_VOLUME_HEADER1`.
pub const FN_VOLUME_HEADER1: &str = "Volume";
/// Java package-private static final `FN_MOD_PARTICLE_HEADER1`.
pub const FN_MOD_PARTICLE_HEADER1: &str = "Model";
/// Java package-private static final `INIT_MOTL_FILE_HEADER1`.
pub const INIT_MOTL_FILE_HEADER1: &str = "Initial";
/// Java package-private static final `INIT_MOTL_FILE_HEADER2`.
pub const INIT_MOTL_FILE_HEADER2: &str = "MOTL";
/// Java package-private static final `LABEL`.
pub const LABEL: &str = "Volume Table";
/// Java package-private static final `TILT_RANGE_HEADER1_LABEL`.
pub const TILT_RANGE_HEADER1_LABEL: &str = "Tilt Range";
/// Java private static final `TILT_RANGE_MULTI_AXES_HEADER1_LABEL`.
const TILT_RANGE_MULTI_AXES_HEADER1_LABEL: &str = "Missing Wedge";
/// Java private static final `TILT_RANGE_MULTI_AXES_HEADER2_LABEL`.
const TILT_RANGE_MULTI_AXES_HEADER2_LABEL: &str = "Mask";
/// Java private static final `UNIQUE_KEY`.
const UNIQUE_KEY: &str = "Volume";

/// Java package-private `final class VolumeTable extends HighlightableTable implements
/// Expandable, Run3dmodButtonContainer, Viewable`.
pub struct VolumeTable {
    /// Java superclass `HighlightableTable`.
    base: HighlightableTable,
    /// Java private final `rowList`.
    row_list: RowList,
    /// Java private final `rootPanel`.
    root_panel: Rc<JComponent>,
    /// Java private final `cbFlgVolNamesAreTemplates`.
    cb_flg_vol_names_are_templates: Rc<CheckBox>,
    /// Java private final `btnReadTiltFile`.
    btn_read_tilt_file: Rc<SingleLineButton>,
    /// Java private final `r3bVolume`.
    r3b_volume: Rc<Run3dmodSingleLineButton>,
    /// Java private final `header1VolumeNumber`.
    header1_volume_number: Rc<HeaderCell>,
    /// Java private final `header1FnVolume`.
    header1_fn_volume: Rc<HeaderCell>,
    /// Java private final `header1FnModParticle`.
    header1_fn_mod_particle: Rc<HeaderCell>,
    /// Java private final `header1InitMotlFile`.
    header1_init_motl_file: Rc<HeaderCell>,
    /// Java private final `header1TiltRange`.
    header1_tilt_range: Rc<HeaderCell>,
    /// Java private final `header1TiltRangeMultiAxes`.
    header1_tilt_range_multi_axes: Rc<HeaderCell>,
    /// Java private final `header2VolumeNumber`.
    header2_volume_number: Rc<HeaderCell>,
    /// Java private final `header2FnVolume`.
    header2_fn_volume: Rc<HeaderCell>,
    /// Java private final `header2FnModParticle`.
    header2_fn_mod_particle: Rc<HeaderCell>,
    /// Java private final `header2InitMotlFile`.
    header2_init_motl_file: Rc<HeaderCell>,
    /// Java private final `header2TiltRangeStart`.
    header2_tilt_range_start: Rc<HeaderCell>,
    /// Java private final `header2TiltRangeEnd`.
    header2_tilt_range_end: Rc<HeaderCell>,
    /// Java private final `header2TiltRangeMultiAxes`.
    header2_tilt_range_multi_axes: Rc<HeaderCell>,
    /// Java private final `initMotlFileColumn`.
    init_motl_file_column: Column,
    /// Java private final `tiltRangeColumn`.
    tilt_range_column: Column,
    /// Java private final `btnMoveUp`.
    btn_move_up: Rc<SingleLineButton>,
    /// Java private final `btnMoveDown`.
    btn_move_down: Rc<SingleLineButton>,
    /// Java private final `btnInsertRow`.
    btn_insert_row: Rc<SingleLineButton>,
    /// Java private final `btnDeleteRow`.
    btn_delete_row: Rc<SingleLineButton>,
    /// Java private final `btnCopyRow`.
    btn_copy_row: Rc<SingleLineButton>,
    /// Java private final `pnlTableButtons`.
    pnl_table_buttons: Rc<JComponent>,
    /// Java private final `pnlBottomButtons`.
    pnl_bottom_buttons: Rc<JComponent>,
    /// Java private final `pnlSideButtons`.
    pnl_side_buttons: Rc<JComponent>,
    /// Java private final `pnlBorder`.
    pnl_border: Rc<JComponent>,
    /// Java private final `pnlTable`.
    pnl_table: Rc<JComponent>,
    /// Java private final `layout`.
    layout: GridBagLayout,
    /// Java private final `constraints`.
    constraints: RefCell<GridBagConstraints>,
    /// Java private final `viewport`.
    viewport: Rc<Viewport>,

    /// Java private final `volumeFileFilter`.
    volume_file_filter: Rc<VolumeFileFilter>,
    /// Java private final `btnExpandFnVolume`.
    btn_expand_fn_volume: Rc<ExpandButton>,
    /// Java private final `btnExpandFnModParticle`.
    btn_expand_fn_mod_particle: Rc<ExpandButton>,
    /// Java private final `btnExpandInitMotlFile`.
    btn_expand_init_motl_file: Rc<ExpandButton>,
    /// Java private final `btnExpandTiltRangeMultiAxes`.
    btn_expand_tilt_range_multi_axes: Rc<ExpandButton>,
    /// Java private final `manager`.
    manager: &'static PeetManager,
    /// Java private final `parent`.
    parent: Weak<PeetDialog>,
    /// Java private final `focusableParents`.
    focusable_parents: Vec<Rc<JComponent>>,

    /// Java private `useInitMotlFile`, initially true.
    use_init_motl_file: StdCell<bool>,
    /// Java private `useTiltRange`, initially true.
    use_tilt_range: StdCell<bool>,
    /// Java private `verticalRigidArea1` (a layout spacer; whether it is in the panel).
    vertical_rigid_area1: StdCell<bool>,
    /// Java private `horizontalRigidArea1` (a layout spacer; whether it is in the
    /// panel).
    horizontal_rigid_area1: StdCell<bool>,
    /// Java private `tiltRangeMultiAxes`, initially false.
    tilt_range_multi_axes: StdCell<bool>,
    /// Java `this`.
    self_ref: Weak<VolumeTable>,
}

impl VolumeTable {
    /// Java private `VolumeTable(PeetManager, PeetDialog)`.
    fn new(manager: &'static PeetManager, parent: &Rc<PeetDialog>) -> Rc<VolumeTable> {
        let numeric_width = UIParameters::get_instance_void().get_numeric_width();
        let peet_table_size = etomo_director::INSTANCE
            .with_user_configuration(|user_config| user_config.get_peet_table_size().get_int());
        let this = Rc::new_cyclic(|self_ref: &Weak<VolumeTable>| {
            let viewable: Weak<dyn Viewable> = self_ref.clone();
            let expandable: Weak<dyn Expandable> = self_ref.clone();
            let container: Weak<dyn Run3dmodButtonContainer> = self_ref.clone();
            // construction
            let volume_file_filter = Rc::new(VolumeFileFilter::get_instance(Some(
                manager as &'static dyn BaseManager,
            )));
            let btn_expand_fn_volume = ExpandButton::get_instance_expandable_type(
                Some(expandable.clone()),
                Some(&expand_button::Type::MORE),
            );
            btn_expand_fn_volume.set_name(Some(FN_VOLUME_HEADER1));
            let btn_expand_fn_mod_particle = ExpandButton::get_instance_expandable_type(
                Some(expandable.clone()),
                Some(&expand_button::Type::MORE),
            );
            btn_expand_fn_mod_particle.set_name(Some(FN_MOD_PARTICLE_HEADER1));
            let btn_expand_init_motl_file = ExpandButton::get_instance_expandable_type(
                Some(expandable.clone()),
                Some(&expand_button::Type::MORE),
            );
            let btn_expand_tilt_range_multi_axes = ExpandButton::get_instance_expandable_type(
                Some(expandable),
                Some(&expand_button::Type::MORE),
            );
            // (The source names btnExpandFnModParticle a second time here, after the
            // initial MOTL header, instead of btnExpandInitMotlFile.)
            btn_expand_fn_mod_particle.set_name(Some(INIT_MOTL_FILE_HEADER1));
            let r3b_volume = Run3dmodSingleLineButton::get_3dmod_instance(
                Some("Open in 3dmod"),
                Some(container),
            );
            VolumeTable {
                // super(UNIQUE_KEY)
                base: HighlightableTable::new(UNIQUE_KEY),
                row_list: RowList::new(),
                root_panel: JComponent::new_panel(),
                cb_flg_vol_names_are_templates: CheckBox::new_string(Some(
                    shared_strings::FLG_VOL_NAMES_ARE_TEMPLATES_LABEL,
                )),
                btn_read_tilt_file: SingleLineButton::new_string(Some("Read tilt file")),
                r3b_volume,
                header1_volume_number: HeaderCell::new_string(Some("Vol #")),
                header1_fn_volume: HeaderCell::new_string(Some(FN_VOLUME_HEADER1)),
                header1_fn_mod_particle: HeaderCell::new_string(Some(FN_MOD_PARTICLE_HEADER1)),
                header1_init_motl_file: HeaderCell::new_string(Some(INIT_MOTL_FILE_HEADER1)),
                header1_tilt_range: HeaderCell::new_string(Some(TILT_RANGE_HEADER1_LABEL)),
                header1_tilt_range_multi_axes: HeaderCell::new_string(Some(
                    TILT_RANGE_MULTI_AXES_HEADER1_LABEL,
                )),
                header2_volume_number: HeaderCell::new_void(),
                header2_fn_volume: HeaderCell::new_void(),
                header2_fn_mod_particle: HeaderCell::new_void(),
                header2_init_motl_file: HeaderCell::new_string(Some(INIT_MOTL_FILE_HEADER2)),
                header2_tilt_range_start: HeaderCell::new_string_int(Some("Min"), numeric_width),
                header2_tilt_range_end: HeaderCell::new_string_int(Some("Max"), numeric_width),
                header2_tilt_range_multi_axes: HeaderCell::new_string(Some(
                    TILT_RANGE_MULTI_AXES_HEADER2_LABEL,
                )),
                init_motl_file_column: Column::new(),
                tilt_range_column: Column::new(),
                btn_move_up: SingleLineButton::get_html_instance(Some("Up")),
                btn_move_down: SingleLineButton::get_html_instance(Some("Down")),
                btn_insert_row: SingleLineButton::get_html_instance(Some("Insert")),
                btn_delete_row: SingleLineButton::get_html_instance(Some("Delete")),
                btn_copy_row: SingleLineButton::get_html_instance(Some("Dup")),
                pnl_table_buttons: JComponent::new_panel(),
                pnl_bottom_buttons: JComponent::new_panel(),
                pnl_side_buttons: JComponent::new_panel(),
                pnl_border: JComponent::new_panel(),
                pnl_table: JComponent::new_panel(),
                layout: GridBagLayout::new(),
                constraints: RefCell::new(GridBagConstraints::default()),
                viewport: Viewport::new(viewable, peet_table_size, Some(UNIQUE_KEY)),
                volume_file_filter,
                btn_expand_fn_volume,
                btn_expand_fn_mod_particle,
                btn_expand_init_motl_file,
                btn_expand_tilt_range_multi_axes,
                manager,
                parent: Rc::downgrade(parent),
                focusable_parents: vec![parent.get_setup_j_component()],
                use_init_motl_file: StdCell::new(true),
                use_tilt_range: StdCell::new(true),
                vertical_rigid_area1: StdCell::new(false),
                horizontal_rigid_area1: StdCell::new(false),
                tilt_range_multi_axes: StdCell::new(false),
                self_ref: self_ref.clone(),
            }
        });
        this.create_panel();
        this.update_display_void();
        this.set_tool_tip_text();
        this
    }

    /// Java static `getInstance(PeetManager, PeetDialog)`.
    pub fn get_instance(manager: &'static PeetManager, parent: &Rc<PeetDialog>) -> Rc<VolumeTable> {
        let instance = VolumeTable::new(manager, parent);
        instance.add_listeners();
        instance
    }

    /// Java field read `parent`.
    fn parent(&self) -> Rc<PeetDialog> {
        self.parent
            .upgrade()
            .expect("the PEET dialog owns its volume table")
    }

    /// Java `this`.
    fn this(&self) -> Rc<VolumeTable> {
        self.self_ref.upgrade().expect("the volume table is alive")
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

    /// Java package-private `isTiltRangeMultiAxes()`.
    pub fn is_tilt_range_multi_axes(&self) -> bool {
        self.tilt_range_multi_axes.get()
    }

    /// Java package-private `isFnVolumeExpanded()`.
    pub fn is_fn_volume_expanded(&self) -> bool {
        self.btn_expand_fn_volume.is_expanded()
    }

    /// Java package-private `isFnModParticleExpanded()`.
    pub fn is_fn_mod_particle_expanded(&self) -> bool {
        self.btn_expand_fn_mod_particle.is_expanded()
    }

    /// Java package-private `isFlgVolNamesAreTemplates()`.
    pub fn is_flg_vol_names_are_templates(&self) -> bool {
        self.cb_flg_vol_names_are_templates.is_selected()
    }

    /// Java package-private `isInitMotlFileExpanded()`.
    pub fn is_init_motl_file_expanded(&self) -> bool {
        self.btn_expand_init_motl_file.is_expanded()
    }

    /// Java package-private static `getTiltRangeMultiAxesLabel()`.
    pub fn get_tilt_range_multi_axes_label() -> String {
        format!("{TILT_RANGE_MULTI_AXES_HEADER1_LABEL} {TILT_RANGE_MULTI_AXES_HEADER2_LABEL}")
    }

    /// Java package-private `getContainer()`.
    pub fn get_container(&self) -> Rc<JComponent> {
        self.root_panel.clone()
    }

    /// Java package-private `getParameters(PeetMetaData)`.
    pub fn get_parameters_meta_data(&self, meta_data: &PeetMetaData) {
        self.row_list.get_parameters_meta_data(meta_data);
    }

    /// Java package-private `setParameters(ConstPeetMetaData)`.
    pub fn set_parameters_meta_data(&self, meta_data: &'static dyn ConstPeetMetaData) {
        self.row_list.set_parameters(Some(meta_data));
    }

    /// Java package-private `convertCopiedPaths(String)`.
    pub fn convert_copied_paths(&self, orig_dataset_dir: &str) {
        self.row_list.convert_copied_paths(orig_dataset_dir);
    }

    /// Java package-private `isIncorrectPaths()`.
    pub fn is_incorrect_paths(&self) -> bool {
        if self.cb_flg_vol_names_are_templates.is_selected() {
            return false;
        }
        self.row_list.is_incorrect_paths()
    }

    /// Java package-private `fixIncorrectPaths(boolean)`.
    pub fn fix_incorrect_paths(&self, choose_path_every_row: bool) -> bool {
        self.row_list.fix_incorrect_paths(choose_path_every_row)
    }

    /// Java package-private `isCorrectPathNull()`.
    pub fn is_correct_path_null(&self) -> bool {
        self.parent().is_correct_path_null()
    }

    /// Java package-private `getFileChooserInstance()`.
    pub fn get_file_chooser_instance(&self) -> Rc<FileChooser> {
        self.parent().get_file_chooser_instance()
    }

    /// Java package-private `setCorrectPath(String)`.
    pub fn set_correct_path(&self, correct_path: Option<String>) {
        self.parent().set_correct_path(correct_path);
    }

    /// Java package-private `getCorrectPath()`.
    pub fn get_correct_path(&self) -> Option<String> {
        self.parent().get_correct_path()
    }

    /// Java package-private `setParameters(MatlabParam, boolean, boolean, boolean,
    /// File)`.
    pub fn set_parameters_matlab_param(
        &self,
        matlab_param_file: &mut MatlabParam,
        use_init_motl_file: bool,
        use_tilt_range: bool,
        tilt_range_multi_axes: bool,
        import_dir: Option<&Path>,
    ) {
        self.cb_flg_vol_names_are_templates
            .set_selected_boolean(matlab_param_file.is_flg_vol_names_are_templates());
        let init_motl_file_is_expanded = self.btn_expand_init_motl_file.is_expanded();
        let mut user_dir = None;
        if let Some(import_dir) = import_dir {
            // `System.setProperty("user.dir", importDir.getAbsolutePath())`; `PWD` is
            // this translation's `user.dir`.
            user_dir = std::env::var("PWD").ok();
            unsafe {
                std::env::set_var(
                    "PWD",
                    utilities::java_io_file_get_absolute_path(&import_dir.to_string_lossy()),
                )
            };
        }
        for i in 0..matlab_param_file.get_volume_list_size() {
            let row = self.add_row_strings(
                matlab_param_file.get_fn_volume(i).as_deref(),
                matlab_param_file.get_fn_mod_particle(i).as_deref(),
                matlab_param_file.get_tilt_range_multi_axes(i).as_deref(),
            );
            row.set_parameters_matlab_param(
                matlab_param_file,
                use_init_motl_file,
                use_tilt_range,
                tilt_range_multi_axes,
            );
            row.expand_init_motl_file(init_motl_file_is_expanded);
        }
        self.refresh_vertical_padding();
        self.refresh_horizontal_padding();
        if import_dir.is_some() {
            // `System.setProperty("user.dir", userDir)`.
            match user_dir {
                Some(user_dir) => unsafe { std::env::set_var("PWD", user_dir) },
                None => unsafe { std::env::remove_var("PWD") },
            }
        }
        self.row_list.done_setting_parameters();
        self.viewport.adjust_viewport(0);
        self.row_list.remove();
        self.row_list.display(&self.viewport);
        self.update_display_void();
        ui_harness::with(|harness| harness.pack_base_manager(Some(self.manager)));
    }

    /// Java package-private `getParameters(MatlabParam)`.
    pub fn get_parameters_matlab_param(&self, matlab_param_file: &mut MatlabParam) {
        matlab_param_file
            .set_flg_vol_names_are_templates(self.cb_flg_vol_names_are_templates.is_selected());
        self.row_list
            .get_parameters_matlab_param(matlab_param_file, self.tilt_range_multi_axes.get());
    }

    /// Java `size()`.
    pub fn size(&self) -> i32 {
        self.row_list.size()
    }

    /// Java package-private `isEmpty()`.
    pub fn is_empty(&self) -> bool {
        self.row_list.is_empty()
    }

    /// Java package-private `updateDisplay(boolean, boolean, boolean)`.
    pub fn update_display(
        &self,
        use_init_motl_file: bool,
        use_tilt_range: bool,
        tilt_range_multi_axes: bool,
    ) {
        self.use_init_motl_file.set(use_init_motl_file);
        self.use_tilt_range.set(use_tilt_range);
        self.init_motl_file_column.set_enabled(use_init_motl_file);
        self.tilt_range_column.set_enabled(use_tilt_range);
        if use_tilt_range && tilt_range_multi_axes != self.tilt_range_multi_axes.get() {
            self.tilt_range_multi_axes.set(tilt_range_multi_axes);
            self.pnl_table.remove_all();
            self.display();
            self.row_list.remove();
            self.row_list.display(&self.viewport);
            self.refresh_horizontal_padding();
            if let Some(main_panel) = self.manager.get_main_panel() {
                main_panel.main_panel().repaint();
            }
        }
        self.update_display_void();
    }

    /// Java private `createPanel()`.
    fn create_panel(&self) {
        // init
        // Swing layout: r3bVolume.setToPreferredSize(); btnReadTiltFile
        // .setToPreferredSize().
        self.build_table();
        self.viewport.init_paging();
        let table: Weak<dyn HighlightableTableVirtual> = self.self_ref.clone();
        self.base.init_highlight_hotkeys(table);
        // border (BoxLayout X_AXIS)
        self.pnl_border.add(&self.pnl_table);
        if let Some(paging_panel) = self.viewport.get_paging_panel() {
            self.pnl_border.add(&paging_panel);
        }
        // buttons -side (BoxLayout Y_AXIS, rigid areas x0_y5 between, vertical glue)
        self.pnl_side_buttons.add(&self.btn_move_up.get_component());
        self.pnl_side_buttons
            .add(&self.btn_move_down.get_component());
        self.pnl_side_buttons
            .add(&self.btn_insert_row.get_component());
        self.pnl_side_buttons
            .add(&self.btn_delete_row.get_component());
        self.pnl_side_buttons
            .add(&self.btn_copy_row.get_component());
        // buttons - bottom (BoxLayout X_AXIS, horizontal glue between)
        self.pnl_bottom_buttons
            .add(&self.cb_flg_vol_names_are_templates.get_component());
        self.pnl_bottom_buttons
            .add(&self.r3b_volume.get_component());
        self.pnl_bottom_buttons
            .add(&self.btn_read_tilt_file.get_component());
        // Table and side buttons (BoxLayout Y_AXIS)
        self.pnl_table_buttons.add(&self.pnl_border);
        self.refresh_vertical_padding();
        // root (BoxLayout X_AXIS, rigid area x5_y0 first)
        self.root_panel.set_border_title(
            EtchedBorder::new(Some(LABEL))
                .get_border()
                .get_title()
                .as_deref(),
        );
        self.root_panel.add(&self.pnl_table_buttons);
        self.refresh_horizontal_padding();
    }

    /// Java private `buildTable()`.
    fn build_table(&self) {
        // columns
        self.init_motl_file_column
            .add(self.header1_init_motl_file.clone() as Rc<dyn CellVirtual>);
        self.init_motl_file_column
            .add(self.header2_init_motl_file.clone() as Rc<dyn CellVirtual>);
        self.tilt_range_column
            .add(self.header1_tilt_range.clone() as Rc<dyn CellVirtual>);
        self.tilt_range_column
            .add(self.header2_tilt_range_start.clone() as Rc<dyn CellVirtual>);
        self.tilt_range_column
            .add(self.header2_tilt_range_end.clone() as Rc<dyn CellVirtual>);
        self.tilt_range_column
            .add(self.header1_tilt_range_multi_axes.clone() as Rc<dyn CellVirtual>);
        self.tilt_range_column
            .add(self.header2_tilt_range_multi_axes.clone() as Rc<dyn CellVirtual>);
        // table: GridBagLayout, black line border
        self.display();
    }

    /// Java private `refreshHorizontalPadding()`.  Pad the table on the right side
    /// based on the number of character in the three columns that change size.  6.3 is
    /// an estimate of how much the table grows on average with each character.
    fn refresh_horizontal_padding(&self) {
        if self.horizontal_rigid_area1.get() {
            self.root_panel.remove(&self.pnl_side_buttons);
        }
        let max_row_text_size = self
            .row_list
            .get_max_row_text_size(self.tilt_range_multi_axes.get());
        // Swing layout: horizontalRigidArea1 = Box.createRigidArea(FixedDim.x181_y0)
        // when maxRowTextSize <= 50, else max(round(181 - (maxRowTextSize - 50) *
        // 6.3), 8) wide.
        let _ = max_row_text_size;
        self.horizontal_rigid_area1.set(true);
        self.root_panel.add(&self.pnl_side_buttons);
        // Swing layout: horizontalRigidArea2.
    }

    /// Java private `refreshVerticalPadding()`.
    fn refresh_vertical_padding(&self) {
        let size = self.row_list.size();
        let no_padding = 3;
        if self.vertical_rigid_area1.get() {
            self.pnl_table_buttons.remove(&self.pnl_bottom_buttons);
        }
        // Swing layout: verticalRigidArea1 is max(10 + (noPadding - size) * 20, 10)
        // high, then pnlBottomButtons, then verticalRigidArea2.
        let _height = (10 + (no_padding - size) * 20).max(10);
        self.vertical_rigid_area1.set(true);
        self.pnl_table_buttons.add(&self.pnl_bottom_buttons);
    }

    /// Java private `imodVolume(Run3dmodMenuOptions)`.
    fn imod_volume(&self, menu_options: Option<Run3dmodMenuOptions>) {
        let row = self.row_list.get_highlighted_row();
        let Some(row) = row else {
            panic!("java.lang.IllegalStateException: r3bVolume enabled when no row is highlighted");
        };
        row.imod_volume(menu_options);
    }

    /// `cell.add(pnlTable, layout, constraints)` for a header cell or expand button.
    fn add_to_table(&self, cell: &dyn CellVirtual, component: Rc<JComponent>) {
        cell.add(&self.pnl_table);
        self.layout
            .set_constraints(&component, &self.constraints.borrow());
    }

    /// `expandButton.add(pnlTable, layout, constraints)`.
    fn add_expand_button(&self, button: &ExpandButton) {
        let old_weightx = self.constraints.borrow().weightx;
        self.constraints.borrow_mut().weightx = 0.0;
        button.add(&self.pnl_table);
        self.layout
            .set_constraints(&button.get_component(), &self.constraints.borrow());
        self.constraints.borrow_mut().weightx = old_weightx;
    }

    /// Java private `display()`.
    fn display(&self) {
        {
            let mut constraints = self.constraints.borrow_mut();
            constraints.fill = GRID_BAG_BOTH;
            constraints.anchor = GRID_BAG_CENTER;
            constraints.gridheight = 1;
            constraints.weighty = 1.0;
            // First header row
            constraints.weightx = 1.0;
            constraints.gridwidth = 2;
        }
        self.add_to_table(
            &*self.header1_volume_number,
            self.header1_volume_number.get_component(),
        );
        self.constraints.borrow_mut().gridwidth = 1;
        self.add_to_table(
            &*self.header1_fn_volume,
            self.header1_fn_volume.get_component(),
        );
        self.add_expand_button(&self.btn_expand_fn_volume);
        self.add_to_table(
            &*self.header1_fn_mod_particle,
            self.header1_fn_mod_particle.get_component(),
        );
        self.add_expand_button(&self.btn_expand_fn_mod_particle);
        self.add_to_table(
            &*self.header1_init_motl_file,
            self.header1_init_motl_file.get_component(),
        );
        self.add_expand_button(&self.btn_expand_init_motl_file);
        if !self.tilt_range_multi_axes.get() {
            self.constraints.borrow_mut().gridwidth = GRID_BAG_REMAINDER;
            self.add_to_table(
                &*self.header1_tilt_range,
                self.header1_tilt_range.get_component(),
            );
        } else {
            self.add_to_table(
                &*self.header1_tilt_range_multi_axes,
                self.header1_tilt_range_multi_axes.get_component(),
            );
            self.constraints.borrow_mut().gridwidth = GRID_BAG_REMAINDER;
            self.add_expand_button(&self.btn_expand_tilt_range_multi_axes);
        }
        // Second header row
        self.constraints.borrow_mut().gridwidth = 2;
        self.add_to_table(
            &*self.header2_volume_number,
            self.header2_volume_number.get_component(),
        );
        self.add_to_table(
            &*self.header2_fn_volume,
            self.header2_fn_volume.get_component(),
        );
        self.add_to_table(
            &*self.header2_fn_mod_particle,
            self.header2_fn_mod_particle.get_component(),
        );
        self.add_to_table(
            &*self.header2_init_motl_file,
            self.header2_init_motl_file.get_component(),
        );
        if !self.tilt_range_multi_axes.get() {
            self.constraints.borrow_mut().gridwidth = 1;
            self.add_to_table(
                &*self.header2_tilt_range_start,
                self.header2_tilt_range_start.get_component(),
            );
            self.constraints.borrow_mut().gridwidth = GRID_BAG_REMAINDER;
            self.add_to_table(
                &*self.header2_tilt_range_end,
                self.header2_tilt_range_end.get_component(),
            );
        } else {
            self.constraints.borrow_mut().gridwidth = GRID_BAG_REMAINDER;
            self.add_to_table(
                &*self.header2_tilt_range_multi_axes,
                self.header2_tilt_range_multi_axes.get_component(),
            );
        }
    }

    /// Java package-private `getVolumeNumberHeaderCell()`.
    pub fn get_volume_number_header_cell(&self) -> Rc<HeaderCell> {
        self.header1_volume_number.clone()
    }

    /// Java package-private `getFnVolumeHeaderCell()`.
    pub fn get_fn_volume_header_cell(&self) -> Rc<HeaderCell> {
        self.header1_fn_volume.clone()
    }

    /// Java package-private `getFnModParticleHeaderCell()`.
    pub fn get_fn_mod_particle_header_cell(&self) -> Rc<HeaderCell> {
        self.header1_fn_mod_particle.clone()
    }

    /// Java package-private `getTiltRangeMultiAxesHeaderCell()`.
    pub fn get_tilt_range_multi_axes_header_cell(&self) -> Rc<HeaderCell> {
        self.header1_tilt_range_multi_axes.clone()
    }

    /// Java package-private `getInitMotlFileHeaderCell()`.
    pub fn get_init_motl_file_header_cell(&self) -> Rc<HeaderCell> {
        self.header1_init_motl_file.clone()
    }

    /// Java package-private `getTiltRangeHeaderCell()`.
    pub fn get_tilt_range_header_cell(&self) -> Rc<HeaderCell> {
        self.header1_tilt_range.clone()
    }

    /// Java package-private `validateRun(boolean)`.  Validate for running.  Returns
    /// error message (null if valid).
    pub fn validate_run(&self, tilt_range_required: bool) -> Option<String> {
        self.row_list
            .validate_run(tilt_range_required, self.tilt_range_multi_axes.get())
    }

    /// Java private `deleteRow(VolumeRow)`.
    fn delete_row(&self, row: Option<Rc<VolumeRow>>) {
        self.row_list.remove();
        let index = self.row_list.delete(row);
        self.row_list.highlight(index);
        self.viewport.adjust_viewport(index);
        self.row_list.display(&self.viewport);
        self.refresh_vertical_padding();
        self.refresh_horizontal_padding();
        self.update_display_void();
        ui_harness::with(|harness| harness.pack_base_manager(Some(self.manager)));
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
        let autodoc = unsafe { autodoc.as_ref() }.map(|autodoc| {
            autodoc as &dyn crate::imod::etomo::storage::autodoc::read_only_autodoc::ReadOnlyAutodoc
        });
        let _tooltip =
            etomo_autodoc::get_tooltip_autodoc_add_source(autodoc, Some("maskModelPts"), false);
        self.cb_flg_vol_names_are_templates
            .set_tool_tip_text_string(
                etomo_autodoc::get_tooltip_autodoc_add_source(
                    autodoc,
                    Some(matlab_param::FLG_VOL_NAMES_ARE_TEMPLATES_KEY),
                    false,
                )
                .as_deref(),
            );
        self.btn_insert_row
            .set_tool_tip_text(Some("Add a new row to the table."));
        self.btn_read_tilt_file.set_tool_tip_text(Some(
            "Fill in the tilt range for the highlighted row by selecting a file with tilt angles.",
        ));
        self.r3b_volume.set_tool_tip_text(Some(
            "Open the volume and model for the highlighted row in 3dmod.",
        ));
        self.btn_copy_row.set_tool_tip_text(Some(
            "Create a new row that is a duplicate of the highlighted row.",
        ));
        self.btn_move_up
            .set_tool_tip_text(Some("Move highlighted row up in the table."));
        self.btn_move_down
            .set_tool_tip_text(Some("Move highlighted row down in the table."));
        self.btn_delete_row
            .set_tool_tip_text(Some("Remove highlighted row from table."));
    }

    /// Java private `copyRow(boolean)`.  Made a new that contains data copied from the
    /// highlighted row.
    fn copy_row(&self, init: bool) {
        self.add_row_from(self.row_list.get_highlighted_row());
        self.viewport.adjust_viewport(self.row_list.size() - 1);
        self.row_list.remove();
        self.row_list.display(&self.viewport);
        self.update_display_void();
        self.parent().msg_volume_table_size_changed(init);
        ui_harness::with(|harness| harness.pack_base_manager(Some(self.manager)));
    }

    /// Java package-private `pack()`.
    pub fn pack(&self) {
        self.refresh_horizontal_padding();
    }

    /// Java private `insertRow(boolean)`.  Allow the user to choose a tomogram and a
    /// model and add them to the table in a new row.  The tomogram is required.  The
    /// model is optional.
    fn insert_row(&self, init: bool) {
        if !self.manager.set_param_file() {
            ui_harness::with(|harness| {
                harness.open_message_dialog_base_manager_string_string(
                    Some(self.manager),
                    &format!(
                        "Please set the {} and {} fields before adding rows.",
                        super::peet_dialog::DIRECTORY_LABEL,
                        super::peet_dialog::FN_OUTPUT_LABEL
                    ),
                    "Entry Error",
                )
            });
            return;
        }
        let curr_row_index = self.row_list.get_highlighted_row_index();
        if curr_row_index == -1 {
            self.add_row_void();
        } else {
            let new_row_index = curr_row_index + 1;
            self.add_row_index(new_row_index);
        }
        self.viewport.adjust_viewport(self.row_list.size() - 1);
        self.row_list.remove();
        self.row_list.display(&self.viewport);
        self.refresh_vertical_padding();
        self.refresh_horizontal_padding();
        self.update_display_void();
        self.parent().msg_volume_table_size_changed(init);
        ui_harness::with(|harness| harness.pack_base_manager(Some(self.manager)));
    }

    /// Java private `openTiltFile()`.
    fn open_tilt_file(&self) {
        let row = self.row_list.get_highlighted_row();
        let Some(row) = row else {
            ui_harness::with(|harness| {
                harness.open_message_dialog_base_manager_string_string(
                    Some(self.manager),
                    "Please highlight a row.",
                    "Entry Error",
                )
            });
            return;
        };
        let chooser = self.parent().get_file_chooser_instance();
        chooser.add_choosable_file_filter(Rc::new(TiltFileFilter::new()));
        // Add the default file filter (tilt log)
        let tilt_log_file_filter = TiltLogFileFilter::new();
        chooser.add_choosable_file_filter(Rc::new(tilt_log_file_filter));
        // Swing layout: chooser.setPreferredSize(UIParameters.getInstance()
        // .getFileChooserDimension()).
        chooser.set_file_selection_mode(file_chooser::FILES_ONLY);
        let return_val = chooser.show_open_dialog(Some(&self.root_panel));
        if return_val == file_chooser::APPROVE_OPTION {
            let Some(file) = chooser.get_selected_file() else {
                return;
            };
            self.parent()
                .set_last_location(file.parent().map(Path::to_path_buf));
            if tilt_log_file_filter.accept(&file)
                || utilities::java_io_file_get_name(&file.to_string_lossy()).ends_with(".log")
            {
                match TiltLog::get_instance(Some(self.manager), AxisID::Only, &file) {
                    Ok(mut tilt_log) => {
                        tilt_log.read();
                        row.set_tilt_range_min(Some(&tilt_log.get_min_angle()));
                        row.set_tilt_range_max(Some(&tilt_log.get_max_angle()));
                    }
                    Err(e) => {
                        eprintln!("{e}");
                        ui_harness::with(|harness| {
                            harness.open_message_dialog_base_manager_string_string(
                                Some(self.manager),
                                &format!(
                                    "Unable to open tilt log {}\n{}",
                                    utilities::java_io_file_get_absolute_path(
                                        &file.to_string_lossy()
                                    ),
                                    e.get_message()
                                ),
                                "File Open Failure",
                            )
                        });
                    }
                }
            } else {
                let tilt_file = TiltFile::get_instance(Some(self.manager), AxisID::Only, &file);
                row.set_tilt_range_min(Some(&tilt_file.get_min_angle().to_string()));
                row.set_tilt_range_max(Some(&tilt_file.get_max_angle().to_string()));
            }
        }
    }

    /// Java private `addRow()`.
    fn add_row_void(&self) -> Rc<VolumeRow> {
        let row = self.row_list.add(
            self.manager,
            &self.this(),
            &self.pnl_table,
            &self.init_motl_file_column,
            &self.tilt_range_column,
            &self.volume_file_filter,
        );
        row.expand_fn_volume(self.btn_expand_fn_volume.is_expanded());
        row.expand_fn_mod_particle(self.btn_expand_fn_mod_particle.is_expanded());
        row
    }

    /// Java private `addRow(int)`.
    fn add_row_index(&self, new_row_index: i32) -> Rc<VolumeRow> {
        let row = self.row_list.add_index(
            self.manager,
            &self.this(),
            &self.pnl_table,
            &self.init_motl_file_column,
            &self.tilt_range_column,
            &self.volume_file_filter,
            new_row_index,
        );
        row.expand_fn_volume(self.btn_expand_fn_volume.is_expanded());
        row.expand_fn_mod_particle(self.btn_expand_fn_mod_particle.is_expanded());
        row
    }

    /// Java private `addRow(String, String, String)`.
    fn add_row_strings(
        &self,
        fn_volume: Option<&str>,
        fn_mod_particle: Option<&str>,
        tilt_range_multi_axes: Option<&str>,
    ) -> Rc<VolumeRow> {
        let row = self.row_list.add_strings(
            self.manager,
            fn_volume,
            fn_mod_particle,
            tilt_range_multi_axes,
            &self.this(),
            &self.pnl_table,
            &self.init_motl_file_column,
            &self.tilt_range_column,
            &self.volume_file_filter,
        );
        row.expand_fn_volume(self.btn_expand_fn_volume.is_expanded());
        row.expand_fn_mod_particle(self.btn_expand_fn_mod_particle.is_expanded());
        row.expand_tilt_range_multi_axes(self.btn_expand_tilt_range_multi_axes.is_expanded());
        row
    }

    /// Java private `addRow(VolumeRow)`.  Copy volume, model, initial motl, and tilt
    /// range.
    ///
    /// Fixed in translation (VolumeTable.java:671): "Dup" with no highlighted row (the
    /// button is disabled then, but `copyRow` can be reached through the action) makes
    /// Java dereference null; nothing is copied.
    fn add_row_from(&self, from_row: Option<Rc<VolumeRow>>) {
        let Some(from_row) = from_row else {
            return;
        };
        let new_row_index = from_row.get_index() + 1;
        let row = self.row_list.add_copy(
            &from_row,
            &self.init_motl_file_column,
            &self.tilt_range_column,
            new_row_index,
        );
        row.expand_fn_volume(self.btn_expand_fn_volume.is_expanded());
        row.expand_fn_mod_particle(self.btn_expand_fn_mod_particle.is_expanded());
        row.set_init_motl_file_string(from_row.get_expanded_init_motl_file().as_deref());
        row.expand_init_motl_file(self.btn_expand_init_motl_file.is_expanded());
        row.expand_tilt_range_multi_axes(self.btn_expand_tilt_range_multi_axes.is_expanded());
        // TODO are these duplicate functionality from VolumeRow(VolumeRow...)?
        row.set_tilt_range_min(from_row.get_tilt_range_min().as_deref());
        row.set_tilt_range_max(from_row.get_tilt_range_max().as_deref());
    }

    /// Java private `moveRowUp()`.  Swap the highlighted row with the one above it.
    fn move_row_up(&self) {
        let index = self.row_list.get_highlighted_row_index();
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
        self.viewport.adjust_viewport(index - 1);
        self.row_list.remove();
        self.row_list.display(&self.viewport);
        self.refresh_vertical_padding();
        self.refresh_horizontal_padding();
        self.row_list.reindex(index - 1);
        self.update_display_void();
        if let Some(main_panel) = self.manager.get_main_panel() {
            main_panel.main_panel().repaint();
        }
    }

    /// Java private `moveRowDown()`.  Swap the highlighted row with the one below it.
    fn move_row_down(&self) {
        let index = self.row_list.get_highlighted_row_index();
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
        self.viewport.adjust_viewport(index + 1);
        self.row_list.remove();
        self.row_list.display(&self.viewport);
        self.row_list.reindex(index);
        self.update_display_void();
        if let Some(main_panel) = self.manager.get_main_panel() {
            main_panel.main_panel().repaint();
        }
    }

    /// Java private `updateDisplay()`.
    fn update_display_void(&self) {
        let enable = self.row_list.size() > 0;
        let highlighted = self.row_list.is_highlighted();
        self.btn_expand_fn_volume.set_enabled(enable);
        self.btn_expand_fn_mod_particle.set_enabled(enable);
        self.btn_expand_init_motl_file.set_enabled(enable);
        self.btn_expand_tilt_range_multi_axes
            .set_enabled(enable && self.use_tilt_range.get());
        self.btn_read_tilt_file
            .set_enabled(enable && highlighted && self.use_tilt_range.get());
        self.r3b_volume.set_enabled(enable && highlighted);
        self.btn_delete_row.set_enabled(enable && highlighted);
        self.btn_move_up
            .set_enabled(enable && highlighted && self.row_list.get_highlighted_row_index() > 0);
        self.btn_move_down.set_enabled(
            enable
                && highlighted
                && self.row_list.get_highlighted_row_index() < self.row_list.size() - 1,
        );
        self.btn_copy_row.set_enabled(enable && highlighted);
    }

    /// Java private `addListeners()` with `VTActionListener`.
    fn add_listeners(&self) {
        let adaptee = self.self_ref.clone();
        let action_listener: ActionListener = Rc::new(move |event: &ActionEvent| {
            if let Some(volume_table) = adaptee.upgrade() {
                volume_table.action(event.get_action_command().unwrap_or(""), None, None);
            }
        });
        self.btn_insert_row
            .add_action_listener(action_listener.clone());
        self.btn_read_tilt_file
            .add_action_listener(action_listener.clone());
        self.r3b_volume.add_action_listener(action_listener.clone());
        self.btn_delete_row
            .add_action_listener(action_listener.clone());
        self.btn_move_up
            .add_action_listener(action_listener.clone());
        self.btn_move_down
            .add_action_listener(action_listener.clone());
        self.btn_copy_row
            .add_action_listener(action_listener.clone());
        self.cb_flg_vol_names_are_templates
            .add_action_listener(Some(action_listener));
    }
}

impl Highlightable for VolumeTable {
    /// Java `highlight(boolean)`.
    fn highlight(&self, _highlight: bool) {
        self.update_display_void();
    }
}

impl HighlightableTableVirtual for VolumeTable {
    fn highlightable_table(&self) -> &HighlightableTable {
        &self.base
    }

    /// Java `highlightDownActionPerformed()`.
    fn highlight_down_action_performed(&self) {
        // If the highlight is not visible, don't change the viewport
        let adjust_viewport = self
            .viewport
            .in_viewport(self.row_list.get_highlighted_row_index());
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
            .in_viewport(self.row_list.get_highlighted_row_index());
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
        self.focusable_parents.iter().cloned().map(Some).collect()
    }
}

impl Viewable for VolumeTable {
    /// Java `msgViewportPaged()`.
    fn msg_viewport_paged(&self) {
        self.row_list.remove();
        self.row_list.display(&self.viewport);
        ui_harness::with(|harness| harness.pack_base_manager(Some(self.manager)));
    }

    /// Java `size()`.
    fn size(&self) -> i32 {
        self.row_list.size()
    }

    /// Java `getFocusableParents()`.
    fn get_focusable_parents(&self) -> Vec<Rc<JComponent>> {
        self.focusable_parents.clone()
    }
}

impl Expandable for VolumeTable {
    /// Java `expand(ExpandButton)`.
    fn expand_expand_button(&self, button: &Rc<ExpandButton>) {
        if Rc::ptr_eq(button, &self.btn_expand_fn_volume) {
            self.row_list
                .expand_fn_volume(self.btn_expand_fn_volume.is_expanded());
        } else if Rc::ptr_eq(button, &self.btn_expand_fn_mod_particle) {
            self.row_list
                .expand_fn_mod_particle(self.btn_expand_fn_mod_particle.is_expanded());
        } else if Rc::ptr_eq(button, &self.btn_expand_init_motl_file) {
            self.row_list
                .expand_init_motl(self.btn_expand_init_motl_file.is_expanded());
        } else if Rc::ptr_eq(button, &self.btn_expand_tilt_range_multi_axes) {
            self.row_list
                .expand_tilt_range_multi_axes(self.btn_expand_tilt_range_multi_axes.is_expanded());
        }
        self.refresh_horizontal_padding();
        ui_harness::with(|harness| harness.pack_base_manager(Some(self.manager)));
    }

    /// Java `expand(GlobalExpandButton)`: empty.
    fn expand_global_expand_button(&self, _button: &Rc<GlobalExpandButton>) {}
}

impl Run3dmodButtonContainer for VolumeTable {
    /// Java `action(String, Deferred3dmodButton, Run3dmodMenuOptions)`.
    fn action(
        &self,
        command: &str,
        _deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    ) {
        if Some(command) == self.btn_insert_row.get_action_command().as_deref() {
            self.insert_row(false);
        } else if Some(command) == self.btn_read_tilt_file.get_action_command().as_deref() {
            self.open_tilt_file();
        } else if Some(command) == self.btn_delete_row.get_action_command().as_deref() {
            self.delete_row(self.row_list.get_highlighted_row());
        } else if Some(command) == self.r3b_volume.get_action_command().as_deref() {
            self.imod_volume(run_3dmod_menu_options);
        } else if Some(command) == self.btn_move_up.get_action_command().as_deref() {
            self.move_row_up();
        } else if Some(command) == self.btn_move_down.get_action_command().as_deref() {
            self.move_row_down();
        } else if Some(command) == self.btn_copy_row.get_action_command().as_deref() {
            self.copy_row(false);
        } else if Some(command)
            == self
                .cb_flg_vol_names_are_templates
                .get_action_command()
                .as_deref()
        {
            self.parent().msg_flg_vol_names_are_templates(
                false,
                self.cb_flg_vol_names_are_templates.is_selected(),
            );
        }
    }
}

/// Java private static final class `RowList`.
struct RowList {
    /// Java private final `list`.
    list: RefCell<Vec<Rc<VolumeRow>>>,
    /// Java private `metaData`, initially null.  Saved until the rows are created.
    meta_data: RefCell<Option<&'static dyn ConstPeetMetaData>>,
}

impl RowList {
    /// Java private `RowList()`.
    fn new() -> RowList {
        RowList {
            list: RefCell::new(Vec::new()),
            meta_data: RefCell::new(None),
        }
    }

    /// Java private `size()`.
    fn size(&self) -> i32 {
        self.list.borrow().len() as i32
    }

    /// Java private `isEmpty()`.
    fn is_empty(&self) -> bool {
        self.list.borrow().is_empty()
    }

    /// The rows, copied out so a row's call may reach the list again.
    fn rows(&self) -> Vec<Rc<VolumeRow>> {
        self.list.borrow().clone()
    }

    /// Java private `remove()`.
    fn remove(&self) {
        for row in self.rows() {
            row.remove();
        }
    }

    /// Java private synchronized `delete(VolumeRow, Highlightable, JPanel,
    /// GridBagLayout, GridBagConstraints)`.  Returns the index where the row used to
    /// be (points to the next row).
    fn delete(&self, row: Option<Rc<VolumeRow>>) -> i32 {
        let mut index = -1;
        if let Some(row) = row {
            index = row.get_index();
            self.list.borrow_mut().remove(index as usize);
            let rows = self.rows();
            for i in index as usize..rows.len() {
                rows[i].set_index(i as i32);
            }
        }
        index
    }

    /// Java private synchronized `add(VolumeRow, Column, Column, int)`.
    fn add_copy(
        &self,
        volume_row: &VolumeRow,
        init_motl_file_column: &Column,
        tilt_range_column: &Column,
        new_row_index: i32,
    ) -> Rc<VolumeRow> {
        let row = VolumeRow::get_instance_copy(volume_row, self.size());
        self.list
            .borrow_mut()
            .insert(new_row_index as usize, row.clone());
        let rows = self.rows();
        for i in new_row_index as usize..rows.len() {
            rows[i].set_index(i as i32);
        }
        row.register_init_motl_file_column(init_motl_file_column);
        row.register_tilt_range_column(tilt_range_column);
        row.set_names();
        row
    }

    /// Java private synchronized `add(BaseManager, VolumeTable, JPanel, GridBagLayout,
    /// GridBagConstraints, Column, Column, VolumeFileFilter)`.
    fn add(
        &self,
        manager: &'static PeetManager,
        table: &Rc<VolumeTable>,
        panel: &Rc<JComponent>,
        init_motl_file_column: &Column,
        tilt_range_column: &Column,
        volume_file_filter: &Rc<VolumeFileFilter>,
    ) -> Rc<VolumeRow> {
        let row = VolumeRow::get_instance(manager, self.size(), table, panel, volume_file_filter);
        self.list.borrow_mut().push(row.clone());
        row.register_init_motl_file_column(init_motl_file_column);
        row.register_tilt_range_column(tilt_range_column);
        row.set_names();
        // When this function is used to load from the .epe and .prm files,
        // metadata must be set before MatlabParamFile data. Wait until row is
        // added, then set from metadata.
        row.set_parameters_meta_data(*self.meta_data.borrow());
        row
    }

    /// Java private synchronized `add(BaseManager, VolumeTable, JPanel, GridBagLayout,
    /// GridBagConstraints, Column, Column, VolumeFileFilter, int)`.
    #[allow(clippy::too_many_arguments)]
    fn add_index(
        &self,
        manager: &'static PeetManager,
        table: &Rc<VolumeTable>,
        panel: &Rc<JComponent>,
        init_motl_file_column: &Column,
        tilt_range_column: &Column,
        volume_file_filter: &Rc<VolumeFileFilter>,
        new_row_index: i32,
    ) -> Rc<VolumeRow> {
        let row = VolumeRow::get_instance(manager, new_row_index, table, panel, volume_file_filter);
        self.list
            .borrow_mut()
            .insert(new_row_index as usize, row.clone());
        let rows = self.rows();
        for i in new_row_index as usize..rows.len() {
            rows[i].set_index(i as i32);
        }
        row.register_init_motl_file_column(init_motl_file_column);
        row.register_tilt_range_column(tilt_range_column);
        row.set_names();
        // When this function is used to load from the .epe and .prm files,
        // metadata must be set before MatlabParamFile data. Wait until row is
        // added, then set from metadata.
        row.set_parameters_meta_data(*self.meta_data.borrow());
        row
    }

    /// Java private synchronized `add(BaseManager, String, String, String,
    /// VolumeTable, JPanel, GridBagLayout, GridBagConstraints, Column, Column,
    /// VolumeFileFilter)`.
    #[allow(clippy::too_many_arguments)]
    fn add_strings(
        &self,
        manager: &'static PeetManager,
        fn_volume: Option<&str>,
        fn_mod_particle: Option<&str>,
        tilt_range_multi_axes: Option<&str>,
        table: &Rc<VolumeTable>,
        panel: &Rc<JComponent>,
        init_motl_file_column: &Column,
        tilt_range_column: &Column,
        volume_file_filter: &Rc<VolumeFileFilter>,
    ) -> Rc<VolumeRow> {
        let row = VolumeRow::get_instance_strings(
            manager,
            fn_volume,
            fn_mod_particle,
            tilt_range_multi_axes,
            self.size(),
            table,
            panel,
            volume_file_filter,
        );
        self.list.borrow_mut().push(row.clone());
        row.register_init_motl_file_column(init_motl_file_column);
        row.register_tilt_range_column(tilt_range_column);
        row.set_names();
        // When this function is used to load from the .epe and .prm files,
        // metadata must be set before MatlabParamFile data. Wait until row is
        // added, then set from metadata.
        row.set_parameters_meta_data(*self.meta_data.borrow());
        row
    }

    /// Java private `convertCopiedPaths(String)`.
    fn convert_copied_paths(&self, orig_dataset_dir: &str) {
        for row in self.rows() {
            row.convert_copied_paths(orig_dataset_dir);
        }
    }

    /// Java private `isIncorrectPaths()`.
    fn is_incorrect_paths(&self) -> bool {
        for row in self.rows() {
            if row.is_incorrect_paths() {
                return true;
            }
        }
        false
    }

    /// Java private `fixIncorrectPaths(boolean)`.  Returns false if the user canceled
    /// a file chooser.
    fn fix_incorrect_paths(&self, choose_path_every_row: bool) -> bool {
        for row in self.rows() {
            if !row.fix_incorrect_paths(choose_path_every_row) {
                return false;
            }
        }
        true
    }

    /// Java private `moveRowUp(int)`.  Swap two rows.
    fn move_row_up(&self, row_index: i32) {
        let mut list = self.list.borrow_mut();
        let row_move_up = list.remove(row_index as usize);
        let row_move_down = list.remove(row_index as usize - 1);
        list.insert(row_index as usize - 1, row_move_up);
        list.insert(row_index as usize, row_move_down);
    }

    /// Java private `moveRowDown(int)`.
    fn move_row_down(&self, row_index: i32) {
        let mut list = self.list.borrow_mut();
        let row_move_up = list.remove(row_index as usize + 1);
        let row_move_down = list.remove(row_index as usize);
        list.insert(row_index as usize, row_move_up);
        list.insert(row_index as usize + 1, row_move_down);
    }

    /// Java private `highlight(int)`.  Highlight the row in list at rowIndex.
    fn highlight(&self, row_index: i32) {
        let rows = self.rows();
        if row_index >= 0 && (row_index as usize) < rows.len() {
            rows[row_index as usize].set_highlighter_selected(true);
        }
    }

    /// Java private `highlightDown()`.
    fn highlight_down(&self) -> i32 {
        let mut index = self.get_highlighted_row_index();
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
        let mut index = self.get_highlighted_row_index();
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

    /// Java private `reindex(int)`.  Renumber the table starting from the row in the
    /// ArrayList at startIndex.
    fn reindex(&self, start_index: i32) {
        let rows = self.rows();
        for i in start_index.max(0) as usize..rows.len() {
            rows[i].set_index(i as i32);
        }
    }

    /// Java private `validateRun(boolean, boolean)`.  Returns error message (null if
    /// valid).
    fn validate_run(
        &self,
        tilt_range_required: bool,
        tilt_range_multi_axes: bool,
    ) -> Option<String> {
        let rows = self.rows();
        if rows.is_empty() {
            return Some(format!("Must enter at least one row in {LABEL}"));
        }
        for row in rows {
            let error_message = row.validate_run(tilt_range_required, tilt_range_multi_axes);
            if error_message.is_some() {
                return error_message;
            }
        }
        None
    }

    /// Java private `getParameters(PeetMetaData)`.
    fn get_parameters_meta_data(&self, meta_data: &PeetMetaData) {
        meta_data.reset_init_motl_file();
        meta_data.reset_tilt_range_min();
        meta_data.reset_tilt_range_max();
        for row in self.rows() {
            row.get_parameters_meta_data(meta_data);
        }
    }

    /// Java private `getParameters(MatlabParam, boolean)`.
    fn get_parameters_matlab_param(
        &self,
        matlab_param_file: &mut MatlabParam,
        tilt_range_multi_axes: bool,
    ) {
        matlab_param_file.set_volume_list_size(self.size());
        for row in self.rows() {
            row.get_parameters_matlab_param(matlab_param_file, tilt_range_multi_axes);
        }
    }

    /// Java private `setParameters(ConstPeetMetaData)`.  Save metaData until the rows
    /// are created.
    fn set_parameters(&self, meta_data: Option<&'static dyn ConstPeetMetaData>) {
        *self.meta_data.borrow_mut() = meta_data;
    }

    /// Java private `doneSettingParameters()`.
    fn done_setting_parameters(&self) {
        *self.meta_data.borrow_mut() = None;
    }

    /// Java private `display(Viewport)`.
    fn display(&self, viewport: &Viewport) {
        for (i, row) in self.rows().iter().enumerate() {
            row.display(i as i32, viewport);
        }
    }

    /// Java private `expandFnVolume(boolean)`.
    fn expand_fn_volume(&self, expanded: bool) {
        for row in self.rows() {
            row.expand_fn_volume(expanded);
        }
    }

    /// Java private `expandFnModParticle(boolean)`.
    fn expand_fn_mod_particle(&self, expanded: bool) {
        for row in self.rows() {
            row.expand_fn_mod_particle(expanded);
        }
    }

    /// Java private `expandInitMotl(boolean)`.
    fn expand_init_motl(&self, expanded: bool) {
        for row in self.rows() {
            row.expand_init_motl_file(expanded);
        }
    }

    /// Java private `expandTiltRangeMultiAxes(boolean)`.
    fn expand_tilt_range_multi_axes(&self, expanded: bool) {
        for row in self.rows() {
            row.expand_tilt_range_multi_axes(expanded);
        }
    }

    /// Java private `isHighlighted()`.
    fn is_highlighted(&self) -> bool {
        self.rows().iter().any(|row| row.is_highlighted())
    }

    /// Java private `getHighlightedRow()`.
    fn get_highlighted_row(&self) -> Option<Rc<VolumeRow>> {
        self.rows().into_iter().find(|row| row.is_highlighted())
    }

    /// Java private `getMaxRowTextSize(boolean)`.
    fn get_max_row_text_size(&self, tilt_range_multi_axes: bool) -> i32 {
        let mut max_row_text_size = 0;
        for row in self.rows() {
            max_row_text_size = row
                .get_text_size(tilt_range_multi_axes)
                .max(max_row_text_size);
        }
        max_row_text_size
    }

    /// Java private `getHighlightedRowIndex()`.
    fn get_highlighted_row_index(&self) -> i32 {
        for (i, row) in self.rows().iter().enumerate() {
            if row.is_highlighted() {
                return i as i32;
            }
        }
        -1
    }
}
