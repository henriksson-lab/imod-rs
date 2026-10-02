//! `IMOD/Etomo/src/etomo/ui/swing/EtomoMenu.java`.
//!
//! The menu bar of the main frame, the sub frame and manager frames.  The
//! listeners are defined here; the action functions they call are in the frame
//! classes (`AbstractFrame` and its subclasses).
//!
//! Mnemonics and accelerators are key bindings, which `jdk.rs` does not model;
//! those statements are kept as comments.  `JMenuBar`/`JMenu`/separators are
//! `JComponent` nodes so the menu tree can be searched by name.

use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::{Rc, Weak};

use super::abstract_frame::AbstractFrameVirtual;
use super::check_box_menu_item::CheckBoxMenuItem;
use super::constants;
use super::main_frame_about_box::MainFrameAboutBox;
use super::menu::Menu;
use super::menu_item::MenuItem;
use super::tomogram_process_panel;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::jdk::{ActionEvent, ActionListener, JComponent};
use crate::imod::etomo::process::base_process_manager::BaseProcessManager;
// TODO(unit): needs etomo/process/ImodqtassistProcess.java - `INSTANCE.open(manager,
// action, axisID)` starts/feeds the imodqtassist help viewer.
use crate::imod::etomo::process::imodqtassist_process;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::directive_file_type::DirectiveFileType;
use crate::imod::etomo::util::environment_variable::{self, PARTICLE_DIR};

/// Java package-private `static final String RECON_LABEL`.
pub const RECON_LABEL: &str = "Build Tomogram";
/// Java package-private `static final String JOIN_LABEL`.
pub const JOIN_LABEL: &str = "Join Serial Tomograms";
/// Java `public static final String GENERIC_LABEL`.
pub const GENERIC_LABEL: &str = "Generic Parallel Process";
/// Java package-private `static final String NAD_LABEL`.
pub const NAD_LABEL: &str = "Nonlinear Anisotropic Diffusion";
/// Java package-private `static final String BATCH_RUN_TOMO_LABEL`.
pub const BATCH_RUN_TOMO_LABEL: &str = "Batch Tomograms";
/// Java package-private `static final String PEET_LABEL`.
pub const PEET_LABEL: &str = "Subvolume Averaging (PEET)";
/// Java package-private `static final String FLATTEN_VOLUME_LABEL`.
pub const FLATTEN_VOLUME_LABEL: &str = "Flatten Volume";
/// Java package-private `static final String GPU_TILT_TEST_LABEL`.
pub const GPU_TILT_TEST_LABEL: &str = "Test GPU";
/// Java package-private `static final String ALIGN_FRAMES_LABEL`.
pub const ALIGN_FRAMES_LABEL: &str = "Align Frames";
/// Java package-private `static final String SERIAL_SECTIONS_LABEL`.
pub const SERIAL_SECTIONS_LABEL: &str = "Align Serial Sections / Blend Montages";

/// Java `private static final int nMRUFileMax = 10`.
const N_MRU_FILE_MAX: usize = 10;
/// Java `private static final String TOP_ANCHOR = Constants.TOP_ANCHOR`.
const TOP_ANCHOR: &str = constants::TOP_ANCHOR;

// Java `java.awt.event.KeyEvent` virtual key codes used as mnemonics.
const VK_2: i32 = 0x32;
const VK_3: i32 = 0x33;
const VK_A: i32 = 0x41;
const VK_B: i32 = 0x42;
const VK_C: i32 = 0x43;
const VK_D: i32 = 0x44;
const VK_E: i32 = 0x45;
const VK_F: i32 = 0x46;
const VK_G: i32 = 0x47;
const VK_H: i32 = 0x48;
const VK_I: i32 = 0x49;
const VK_J: i32 = 0x4A;
const VK_L: i32 = 0x4C;
const VK_O: i32 = 0x4F;
const VK_P: i32 = 0x50;
const VK_R: i32 = 0x52;
const VK_S: i32 = 0x53;
const VK_T: i32 = 0x54;
const VK_U: i32 = 0x55;
const VK_X: i32 = 0x58;
const VK_Y: i32 = 0x59;

/// Java `etomo.type.ToolType` (`FLATTEN_VOLUME`, `GPU_TILT_TEST`,
/// `ALIGN_FRAMES`).
///
/// Not part of `EtomoMenu.java`: it stays in this module because the live tree
/// (`etomo_director.rs`, `tools_manager.rs`, ...) imports it from here; it
/// belongs in `type/tool_type.rs`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolType {
    FlattenVolume,
    GpuTiltTest,
    AlignFrames,
}

impl ToolType {
    /// Java `ToolType.toString()` (the `string` field).
    pub const fn label(self) -> &'static str {
        match self {
            Self::FlattenVolume => "Flatten Volume",
            Self::GpuTiltTest => "GPU Test",
            Self::AlignFrames => "Align Frames",
        }
    }
}

/// Java `public final class EtomoMenu`.
pub struct EtomoMenu {
    menu_bar: Rc<JComponent>,

    menu_file: Rc<Menu>,
    menu_open: Rc<MenuItem>,
    /// Java `private final JMenuItem menuRecentProjects = new Menu("Recent Projects")`.
    menu_recent_projects: Rc<Menu>,
    menu_save: Rc<MenuItem>,
    menu_save_as: Rc<MenuItem>,
    menu_close: Rc<MenuItem>,
    menu_cancel: Rc<MenuItem>,
    menu_exit: Rc<MenuItem>,
    menu_tomosnapshot: Rc<MenuItem>,
    /// Java `private final JMenuItem[] menuMRUList = new MenuItem[nMRUFileMax]`;
    /// the elements stay null unless `addListeners` runs its dataset branch.
    menu_mru_list: RefCell<Vec<Option<Rc<MenuItem>>>>,
    menu_export_batch: Rc<MenuItem>,
    menu_template: Rc<Menu>,

    menu_new: Rc<Menu>,
    menu_new_tomogram: Rc<MenuItem>,
    menu_new_join: Rc<MenuItem>,
    menu_new_peet: Rc<MenuItem>,
    menu_serial_sections: Rc<MenuItem>,
    menu_new_anisotropic_diffusion: Rc<MenuItem>,
    menu_new_batch_run_tomo: Rc<MenuItem>,
    menu_new_generic_parallel: Rc<MenuItem>,

    menu_save_scope: Rc<MenuItem>,
    menu_save_system: Rc<MenuItem>,
    menu_save_user: Rc<MenuItem>,

    menu_tools: Rc<Menu>,
    menu_flatten_volume: Rc<MenuItem>,
    menu_gpu_tilt_test: Rc<MenuItem>,
    menu_align_frames: Rc<MenuItem>,

    menu_view: Rc<Menu>,
    menu_log_window: Rc<MenuItem>,
    menu_axis_a: Rc<MenuItem>,
    menu_axis_b: Rc<MenuItem>,
    menu_axis_both: Rc<MenuItem>,
    menu_fit_window: Rc<MenuItem>,

    menu_options: Rc<Menu>,
    menu_settings: Rc<MenuItem>,
    menu_3dmod_startup_window: Rc<CheckBoxMenuItem>,
    menu_3dmod_bin_by_2: Rc<CheckBoxMenuItem>,

    menu_help: Rc<Menu>,
    menu_tomo_guide: Rc<MenuItem>,
    menu_imod_guide: Rc<MenuItem>,
    menu_3dmod_guide: Rc<MenuItem>,
    menu_etomo_guide: Rc<MenuItem>,
    menu_join_guide: Rc<MenuItem>,
    menu_peet_guide: Rc<MenuItem>,
    menu_batch_guide: Rc<MenuItem>,
    peet_help_item: Rc<MenuItem>,
    menu_help_about: Rc<MenuItem>,

    peet_available: bool,

    /// Java `private final AbstractFrame frame`.  The frame owns this menu, so
    /// the back reference is weak.
    frame: Weak<dyn AbstractFrameVirtual>,
}

impl EtomoMenu {
    /// Java private `EtomoMenu(AbstractFrame)`, with the field initializers.
    fn new(frame: Weak<dyn AbstractFrameVirtual>) -> EtomoMenu {
        let original_user_dir = etomo_director::INSTANCE.get_original_user_dir();
        EtomoMenu {
            // `new JMenuBar()`: jdk.rs has no menu-bar kind; the bar is a plain
            // container of menus.
            menu_bar: JComponent::new_other(),
            menu_file: Menu::new("File"),
            menu_open: MenuItem::new_string_int("Open...", VK_O),
            menu_recent_projects: Menu::new("Recent Projects"),
            menu_save: MenuItem::new_string_int("Save", VK_S),
            menu_save_as: MenuItem::new_string_int("Save As...", VK_A),
            menu_close: MenuItem::new_string_int("Close", VK_C),
            menu_cancel: MenuItem::new_string("Cancel"),
            menu_exit: MenuItem::new_string_int("Exit", VK_X),
            menu_tomosnapshot: MenuItem::new_string_int("Run Tomosnapshot", VK_T),
            menu_mru_list: RefCell::new(vec![None; N_MRU_FILE_MAX]),
            menu_export_batch: MenuItem::new_string_int("Export Batch Directive File", VK_B),
            menu_template: Menu::new("Templates"),
            menu_new: Menu::new("New"),
            menu_new_tomogram: MenuItem::new_string_int(RECON_LABEL, VK_T),
            menu_new_join: MenuItem::new_string_int(JOIN_LABEL, VK_J),
            menu_new_peet: MenuItem::new_string_int(PEET_LABEL, VK_P),
            menu_serial_sections: MenuItem::new_string_int(SERIAL_SECTIONS_LABEL, VK_R),
            menu_new_anisotropic_diffusion: MenuItem::new_string_int(NAD_LABEL, VK_D),
            menu_new_batch_run_tomo: MenuItem::new_string_int(BATCH_RUN_TOMO_LABEL, VK_H),
            menu_new_generic_parallel: MenuItem::new_string_int(GENERIC_LABEL, VK_G),
            menu_save_scope: MenuItem::new_string_int("Save Scope Template", VK_C),
            menu_save_system: MenuItem::new_string_int("Save System Template", VK_Y),
            menu_save_user: MenuItem::new_string_int("Save User Template", VK_U),
            menu_tools: Menu::new("Tools"),
            menu_flatten_volume: MenuItem::new_string_int(FLATTEN_VOLUME_LABEL, VK_F),
            menu_gpu_tilt_test: MenuItem::new_string_int(GPU_TILT_TEST_LABEL, VK_U),
            menu_align_frames: MenuItem::new_string_int(ALIGN_FRAMES_LABEL, VK_A),
            menu_view: Menu::new("View"),
            menu_log_window: MenuItem::new_string_int("Show/Hide Log Window", VK_L),
            menu_axis_a: MenuItem::new_string_int("Axis A", VK_A),
            menu_axis_b: MenuItem::new_string_int(tomogram_process_panel::AXIS_B_LABEL, VK_B),
            menu_axis_both: MenuItem::new_string_int("Both Axes", VK_2),
            menu_fit_window: MenuItem::new_string_int("Fit Window", VK_F),
            menu_options: Menu::new("Options"),
            menu_settings: MenuItem::new_string_int("Settings", VK_S),
            menu_3dmod_startup_window: CheckBoxMenuItem::new_string(Some(
                "Open 3dmod with Startup Window",
            )),
            menu_3dmod_bin_by_2: CheckBoxMenuItem::new_string(Some("Open 3dmod Binned by 2")),
            menu_help: Menu::new("Help"),
            menu_tomo_guide: MenuItem::new_string_int("Tomography Guide", VK_T),
            menu_imod_guide: MenuItem::new_string_int("Imod Users Guide", VK_I),
            menu_3dmod_guide: MenuItem::new_string_int("3dmod Users Guide", VK_3),
            menu_etomo_guide: MenuItem::new_string_int("Etomo Users Guide", VK_E),
            menu_join_guide: MenuItem::new_string_int("Join Users Guide", VK_J),
            menu_peet_guide: MenuItem::new_string_int("PEET Users Guide", VK_P),
            menu_batch_guide: MenuItem::new_string_int("Batch Interface Guide", VK_B),
            peet_help_item: MenuItem::new_string("PEET Help"),
            menu_help_about: MenuItem::new_string_int("About", VK_A),
            peet_available: environment_variable::INSTANCE.exists(
                None,
                original_user_dir.as_deref(),
                PARTICLE_DIR,
                Some(AxisID::Only),
            ),
            frame,
        }
    }

    /// Java package-private static `getInstance(AbstractFrame)`.
    pub fn get_instance_abstract_frame(frame: Weak<dyn AbstractFrameVirtual>) -> Rc<EtomoMenu> {
        let instance = Rc::new(EtomoMenu::new(frame.clone()));
        let dataset = true;
        let savable = true;
        instance.create_panel(&frame, dataset, savable);
        instance.add_listeners(&frame, dataset, savable);
        instance
    }

    /// Java package-private static `getInstance(ManagerFrame, boolean)`.
    pub fn get_instance_manager_frame_boolean(
        frame: Weak<dyn AbstractFrameVirtual>,
        savable: bool,
    ) -> Rc<EtomoMenu> {
        let instance = Rc::new(EtomoMenu::new(frame.clone()));
        let dataset = false;
        instance.create_panel(&frame, dataset, savable);
        instance.add_listeners(&frame, dataset, savable);
        instance
    }

    /// Java private `createPanel(AbstractFrame, boolean, boolean)`.
    fn create_panel(&self, _frame: &Weak<dyn AbstractFrameVirtual>, dataset: bool, savable: bool) {
        // init
        // Mnemonics for the main menu bar
        // Swing key binding: menuTools/menuView/menuOptions/menuHelp.setMnemonic(
        // VK_T/VK_V/VK_O/VK_H).
        if dataset || savable {
            // Mnomonics for the file menu
            // Swing key binding: menuNew.setMnemonic(VK_N).
        }
        // Accelerators
        // Swing key binding: menuSettings/menuFitWindow.setAccelerator(ctrl-S/ctrl-F).
        if dataset || savable {
            // Mnemonics for the main menu bar
            // Swing key binding: menuFile.setMnemonic(VK_F).
            if dataset {
                // Swing key binding: menuTemplate.setMnemonic(VK_M).
                // Accelerators
                // Swing key binding: menuAxisA/menuAxisB/menuAxisBoth/menuLogWindow
                // .setAccelerator(ctrl-A/ctrl-B/ctrl-2/ctrl-L).
            }
        }
        // Options menu
        self.menu_options.add(&self.menu_settings.get_component());

        if dataset || savable {
            // Construct menu bar
            self.menu_bar.add(&self.menu_file.get_component());

            if dataset {
                // Options menu
                self.menu_options
                    .add(&self.menu_3dmod_startup_window.get_component());
                self.menu_options
                    .add(&self.menu_3dmod_bin_by_2.get_component());
                // New menu
                self.menu_new.add(&self.menu_new_tomogram.get_component());
                self.menu_new.add(&self.menu_new_join.get_component());
                self.menu_new.add(&self.menu_new_peet.get_component());
                self.menu_new
                    .add(&self.menu_serial_sections.get_component());
                self.menu_new
                    .add(&self.menu_new_anisotropic_diffusion.get_component());
                self.menu_new
                    .add(&self.menu_new_generic_parallel.get_component());
                self.menu_new
                    .add(&self.menu_new_batch_run_tomo.get_component());
                // template menu
                self.menu_template
                    .add(&self.menu_save_scope.get_component());
                self.menu_template
                    .add(&self.menu_save_system.get_component());
                self.menu_template.add(&self.menu_save_user.get_component());
                // View menu
                self.menu_view.add(&self.menu_log_window.get_component());
                self.menu_view.add(&self.menu_axis_a.get_component());
                self.menu_view.add(&self.menu_axis_b.get_component());
                self.menu_view.add(&self.menu_axis_both.get_component());
            }
            if dataset {
                // File menu
                self.menu_file.add(&self.menu_new.get_component());
                self.menu_file.add(&self.menu_open.get_component());
                self.menu_file
                    .add(&self.menu_recent_projects.get_component());
                // addSeparator()
                self.menu_file.add(&JComponent::new_other());
                // Adding the same component again moves it, as in AWT.
                self.menu_file
                    .add(&self.menu_recent_projects.get_component());
                self.menu_file.add(&JComponent::new_other());
            }
            self.menu_file.add(&self.menu_close.get_component());
            self.menu_file.add(&self.menu_save.get_component());
            self.menu_file.add(&self.menu_save_as.get_component());
            if !dataset {
                // cancel is for menus which are savable, but have no dataset
                self.menu_file.add(&JComponent::new_other());
                self.menu_file.add(&self.menu_cancel.get_component());
            } else {
                self.menu_file.add(&JComponent::new_other());
                self.menu_file.add(&self.menu_tomosnapshot.get_component());
                self.menu_file.add(&self.menu_export_batch.get_component());
                self.menu_file.add(&JComponent::new_other());
                self.menu_file.add(&self.menu_template.get_component());
                self.menu_file.add(&JComponent::new_other());
                self.menu_file.add(&self.menu_exit.get_component());
            }
        }
        // Construct menu bar
        self.menu_bar.add(&self.menu_tools.get_component());
        self.menu_bar.add(&self.menu_view.get_component());
        self.menu_bar.add(&self.menu_options.get_component());
        self.menu_bar.add(&self.menu_help.get_component());
        // View menu
        self.menu_view.add(&self.menu_fit_window.get_component());
        // Tool menu
        self.menu_tools
            .add(&self.menu_flatten_volume.get_component());
        self.menu_tools
            .add(&self.menu_gpu_tilt_test.get_component());
        self.menu_tools.add(&self.menu_align_frames.get_component());
        // Help menu
        self.menu_help.add(&self.menu_tomo_guide.get_component());
        self.menu_help.add(&self.menu_imod_guide.get_component());
        self.menu_help.add(&self.menu_3dmod_guide.get_component());
        self.menu_help.add(&self.menu_etomo_guide.get_component());
        self.menu_help.add(&self.menu_join_guide.get_component());
        if self.peet_available {
            self.menu_help.add(&self.menu_peet_guide.get_component());
            self.menu_help.add(&self.peet_help_item.get_component());
        }
        self.menu_help.add(&self.menu_batch_guide.get_component());
        self.menu_help.add(&self.menu_help_about.get_component());
    }

    /// Java private `addListeners(AbstractFrame, boolean, boolean)`.
    fn add_listeners(&self, frame: &Weak<dyn AbstractFrameVirtual>, dataset: bool, savable: bool) {
        // Bind the menu items to their listeners
        let tools_action_listener: ActionListener = {
            let listener = ToolsActionListener::new(frame.clone());
            Rc::new(move |event: &ActionEvent| listener.action_performed(event))
        };
        self.menu_flatten_volume
            .add_action_listener(tools_action_listener.clone());
        self.menu_gpu_tilt_test
            .add_action_listener(tools_action_listener.clone());
        self.menu_align_frames
            .add_action_listener(tools_action_listener);

        let view_action_listener: ActionListener = {
            let listener = ViewActionListener::new(frame.clone());
            Rc::new(move |event: &ActionEvent| listener.action_performed(event))
        };
        self.menu_fit_window
            .add_action_listener(view_action_listener.clone());

        let options_action_listener: ActionListener = {
            let listener = OptionsActionListener::new(frame.clone());
            Rc::new(move |event: &ActionEvent| listener.action_performed(event))
        };
        self.menu_settings
            .add_action_listener(options_action_listener.clone());

        let help_action_listener: ActionListener = {
            let listener = HelpActionListener::new(frame.clone());
            Rc::new(move |event: &ActionEvent| listener.action_performed(event))
        };
        self.menu_tomo_guide
            .add_action_listener(help_action_listener.clone());
        self.menu_imod_guide
            .add_action_listener(help_action_listener.clone());
        self.menu_3dmod_guide
            .add_action_listener(help_action_listener.clone());
        self.menu_etomo_guide
            .add_action_listener(help_action_listener.clone());
        self.menu_join_guide
            .add_action_listener(help_action_listener.clone());
        if self.peet_available {
            self.menu_peet_guide
                .add_action_listener(help_action_listener.clone());
            self.peet_help_item
                .add_action_listener(help_action_listener.clone());
        }
        self.menu_batch_guide
            .add_action_listener(help_action_listener.clone());
        self.menu_help_about
            .add_action_listener(help_action_listener);

        if dataset || savable {
            // Bind the menu items to their listeners
            let file_action_listener: ActionListener = {
                let listener = FileActionListener::new(frame.clone());
                Rc::new(move |event: &ActionEvent| listener.action_performed(event))
            };
            self.menu_save
                .add_action_listener(file_action_listener.clone());
            self.menu_save_as
                .add_action_listener(file_action_listener.clone());
            self.menu_close
                .add_action_listener(file_action_listener.clone());
            self.menu_cancel
                .add_action_listener(file_action_listener.clone());
            if dataset {
                self.menu_new_tomogram
                    .add_action_listener(file_action_listener.clone());
                self.menu_new_join
                    .add_action_listener(file_action_listener.clone());
                self.menu_new_generic_parallel
                    .add_action_listener(file_action_listener.clone());
                self.menu_new_anisotropic_diffusion
                    .add_action_listener(file_action_listener.clone());
                self.menu_new_batch_run_tomo
                    .add_action_listener(file_action_listener.clone());
                self.menu_new_peet
                    .add_action_listener(file_action_listener.clone());
                self.menu_serial_sections
                    .add_action_listener(file_action_listener.clone());
                self.menu_open
                    .add_action_listener(file_action_listener.clone());
                self.menu_exit
                    .add_action_listener(file_action_listener.clone());
                self.menu_tomosnapshot
                    .add_action_listener(file_action_listener.clone());
                self.menu_export_batch
                    .add_action_listener(file_action_listener.clone());
                self.menu_save_scope
                    .add_action_listener(file_action_listener.clone());
                self.menu_save_system
                    .add_action_listener(file_action_listener.clone());
                self.menu_save_user
                    .add_action_listener(file_action_listener);

                self.menu_log_window
                    .add_action_listener(view_action_listener.clone());
                self.menu_axis_a
                    .add_action_listener(view_action_listener.clone());
                self.menu_axis_b
                    .add_action_listener(view_action_listener.clone());
                self.menu_axis_both
                    .add_action_listener(view_action_listener);

                self.menu_3dmod_startup_window
                    .get_component()
                    .add_action_listener(options_action_listener.clone());
                self.menu_3dmod_bin_by_2
                    .get_component()
                    .add_action_listener(options_action_listener);
                // Initialize all of the MRU file menu items
                let file_mru_list_action_listener: ActionListener = {
                    let listener = FileMRUListActionListener::new(frame.clone());
                    Rc::new(move |event: &ActionEvent| listener.action_performed(event))
                };
                for i in 0..N_MRU_FILE_MAX {
                    let item = MenuItem::new_void();
                    self.menu_mru_list.borrow_mut()[i] = Some(item.clone());
                    item.add_action_listener(file_mru_list_action_listener.clone());
                    item.set_visible(false);
                    self.menu_recent_projects.add(&item.get_component());
                }
            }
        }
    }

    /// Java package-private `getMenuBar()`.
    pub fn get_menu_bar(&self) -> Rc<JComponent> {
        self.menu_bar.clone()
    }

    /// Java package-private `setEnabled(BaseManager)`.
    ///
    /// Enable/disable menu items based on currentManager.
    pub fn set_enabled_base_manager(&self, current_manager: Option<&'static dyn BaseManager>) {
        match current_manager {
            None => {
                self.menu_save.set_enabled(false);
                self.menu_save_as.set_enabled(false);
                self.menu_close.set_enabled(false);
                self.menu_axis_a.set_enabled(false);
                self.menu_axis_b.set_enabled(false);
                self.menu_axis_both.set_enabled(false);
                self.menu_export_batch.set_enabled(false);
                self.menu_template.set_enabled(false);
                self.menu_save_scope.set_enabled(false);
                self.menu_save_system.set_enabled(false);
                self.menu_save_user.set_enabled(false);
            }
            Some(current_manager) => {
                self.menu_save.set_enabled(current_manager.is_setup_done());
                self.menu_save_as
                    .set_enabled(current_manager.can_change_param_file_name());
                self.menu_close.set_enabled(true);
                // Java dereferences getBaseMetaData() unchecked; a manager without
                // meta data is taken as not dual axis.
                let dual_axis = current_manager
                    .get_base_meta_data()
                    .map(|meta_data| meta_data.base().get_axis_type())
                    == Some(AxisType::DualAxis);
                self.menu_axis_a.set_enabled(dual_axis);
                self.menu_axis_b.set_enabled(dual_axis);
                self.menu_axis_both.set_enabled(dual_axis);
                self.menu_export_batch
                    .set_enabled(current_manager.can_save_directives());
                self.menu_template
                    .set_enabled(current_manager.can_save_directives());
                self.menu_save_scope
                    .set_enabled(current_manager.can_save_directives());
                self.menu_save_system
                    .set_enabled(current_manager.can_save_directives());
                self.menu_save_user
                    .set_enabled(current_manager.can_save_directives());
            }
        }
    }

    /// Java package-private `setEnabled(EtomoMenu)`.
    ///
    /// Enable/disable menu items based on main Frame Menu.  Only used by the
    /// subframe.
    pub fn set_enabled_etomo_menu(&self, main_frame_menu: &EtomoMenu) {
        self.menu_new_tomogram
            .set_enabled(main_frame_menu.menu_new_tomogram.is_enabled());
        self.menu_new_join
            .set_enabled(main_frame_menu.menu_new_join.is_enabled());
        self.menu_new_generic_parallel
            .set_enabled(main_frame_menu.menu_new_generic_parallel.is_enabled());
        self.menu_new_anisotropic_diffusion
            .set_enabled(main_frame_menu.menu_new_anisotropic_diffusion.is_enabled());
        self.menu_new_batch_run_tomo
            .set_enabled(main_frame_menu.menu_new_batch_run_tomo.is_enabled());
        self.menu_new_peet
            .set_enabled(main_frame_menu.menu_new_peet.is_enabled());
        self.menu_serial_sections
            .set_enabled(main_frame_menu.menu_serial_sections.is_enabled());
        self.menu_save_as
            .set_enabled(main_frame_menu.menu_save_as.is_enabled());
        self.menu_axis_a
            .set_enabled(main_frame_menu.menu_axis_a.is_enabled());
        self.menu_axis_b
            .set_enabled(main_frame_menu.menu_axis_b.is_enabled());
        self.menu_axis_both
            .set_enabled(main_frame_menu.menu_axis_both.is_enabled());
    }

    /// Java package-private `setMRUFileLabels(String[])`.
    ///
    /// Set the MRU etomo data file list.  This fills in the MRU menu items on the
    /// File menu.
    pub fn set_mru_file_labels(&self, m_ru_list: &[String]) {
        // Upstream bug fixed (EtomoMenu.java:401-416): the MRU items exist only
        // when addListeners ran its dataset branch, and Java throws
        // NullPointerException on a menu without them.  Missing items are
        // skipped.
        let menu_mru_list = self.menu_mru_list.borrow().clone();
        for i in 0..m_ru_list.len() {
            if i == N_MRU_FILE_MAX {
                return;
            }
            let Some(item) = &menu_mru_list[i] else {
                continue;
            };
            if m_ru_list[i].is_empty() {
                item.set_visible(false);
            } else {
                item.set_text(&m_ru_list[i]);
                item.set_visible(true);
            }
        }
        for i in m_ru_list.len()..N_MRU_FILE_MAX {
            if let Some(item) = &menu_mru_list[i] {
                item.set_visible(false);
            }
        }
    }

    /// Java package-private `menuToolsAction(AxisID, ActionEvent)`.
    pub fn menu_tools_action(&self, _axis_id: AxisID, event: &ActionEvent) {
        if self.equals_flatten_volume(event) {
            let _ = etomo_director::INSTANCE.open_tool(true, ToolType::FlattenVolume);
        } else if self.equals_gpu_tilt_test(event) {
            let _ = etomo_director::INSTANCE.open_tool(true, ToolType::GpuTiltTest);
        } else if self.equals_align_frames(event) {
            let _ = etomo_director::INSTANCE.open_tool(true, ToolType::AlignFrames);
        }
    }

    /// Java package-private `menuFileAction(ActionEvent)`.
    pub fn menu_file_action(&self, event: &ActionEvent) -> bool {
        let Some(frame) = self.frame.upgrade() else {
            return false;
        };
        let axis_id = frame.get_axis_id();
        if self.equals(&self.menu_save.get_component(), event) {
            frame.save(axis_id);
        } else if self.equals(&self.menu_save_as.get_component(), event) {
            frame.save_as();
        } else if self.equals(&self.menu_close.get_component(), event) {
            frame.save(axis_id);
            frame.close();
        } else if self
            .menu_cancel
            .get_action_command()
            .is_some_and(|command| Some(command.as_str()) == event.get_action_command())
        {
            frame.cancel();
        } else {
            return false;
        }
        true
    }

    /// Java package-private `menuHelpAction(BaseManager, AxisID, JFrame,
    /// ActionEvent)`.
    ///
    /// Handle help menu actions.
    pub fn menu_help_action(
        &self,
        manager: Option<&'static dyn BaseManager>,
        axis_id: AxisID,
        frame: &Rc<JComponent>,
        event: &ActionEvent,
    ) {
        // Get the URL to the IMOD html directory
        // `imodURL = getIMODDirectory().toURI().toURL() + "/html/"`; the value is
        // never read afterwards.  A local directory always converts to a URL, so
        // the MalformedURLException arm (print, return) is unreachable.
        // Upstream bug fixed (EtomoMenu.java:461): with no IMOD directory Java
        // throws NullPointerException here and no help item works; the unused
        // URL is skipped instead.
        let imod_directory: Option<PathBuf> = etomo_director::INSTANCE
            .get_imod_directory()
            .map(|directory| directory.to_path_buf());
        let _imod_url =
            imod_directory.map(|directory| format!("file:{}/html/", directory.display()));

        if self.equals_tomo_guide(event) {
            // TODO
            // HTMLPageWindow manpage = new HTMLPageWindow(); manpage.openURL(imodURL +
            // "tomoguide.html"); manpage.setVisible(true);
            imodqtassist_process::INSTANCE.open(
                manager,
                &format!("tomoguide.html{}", TOP_ANCHOR),
                axis_id,
            );
        }

        if self.equals_imod_guide(event) {
            imodqtassist_process::INSTANCE.open(
                manager,
                &format!("guide.html{}", TOP_ANCHOR),
                axis_id,
            );
        }

        if self.equals_3dmod_guide(event) {
            imodqtassist_process::INSTANCE.open(
                manager,
                &format!("3dmodguide.html{}", TOP_ANCHOR),
                axis_id,
            );
        }

        if self.equals_etomo_guide(event) {
            imodqtassist_process::INSTANCE.open(
                manager,
                &format!("UsingEtomo.html{}", TOP_ANCHOR),
                axis_id,
            );
        }

        if self.equals_join_guide(event) {
            imodqtassist_process::INSTANCE.open(
                manager,
                &format!("tomojoin.html{}", TOP_ANCHOR),
                axis_id,
            );
        }

        if self.equals_peet_guide(event) {
            imodqtassist_process::INSTANCE.open(
                manager,
                &format!("PEETmanual.html{}", TOP_ANCHOR),
                axis_id,
            );
        }
        if self.equals_batch_guide(event) {
            imodqtassist_process::INSTANCE.open(
                manager,
                &format!("batchGuide.html{}", TOP_ANCHOR),
                axis_id,
            );
        }
        if self.equals_peet_help(event) {
            // new File(new File(new File($PARTICLE_DIR), "bin"), "PEETHelp")
            // .getAbsolutePath(): a File built on the empty path resolves its
            // child against "/".
            let particle_dir =
                environment_variable::INSTANCE.get_value(None, None, PARTICLE_DIR, None);
            let bin = if particle_dir.is_empty() {
                PathBuf::from("/bin")
            } else {
                PathBuf::from(&particle_dir).join("bin")
            };
            let peet_help = bin.join("PEETHelp");
            let peet_help = std::path::absolute(&peet_help).unwrap_or(peet_help);
            BaseProcessManager::start_system_program_thread(
                vec![peet_help.to_string_lossy().into_owned()],
                axis_id,
                manager,
            );
        }
        if self.equals_help_about(event) {
            let dlg = MainFrameAboutBox::new(frame, axis_id);
            // Swing layout: dlgSize = dlg.getPreferredSize(); frmSize =
            // frame.getSize(); loc = frame.getLocation(); dlg.setLocation(centred
            // on the frame).
            dlg.set_modal(true);
            dlg.set_visible(true);
        }
    }

    /// Java package-private `isMenu3dmodStartupWindow()`.
    pub fn is_menu_3dmod_startup_window(&self) -> bool {
        self.menu_3dmod_startup_window.get_component().is_selected()
    }

    /// Java package-private `isMenu3dmodBinBy2()`.
    pub fn is_menu_3dmod_bin_by_2(&self) -> bool {
        self.menu_3dmod_bin_by_2.get_component().is_selected()
    }

    /// Java package-private `setMenu3dmodStartupWindow(boolean)`.
    pub fn set_menu_3dmod_startup_window(&self, menu_3dmod_startup_window: bool) {
        self.menu_3dmod_startup_window
            .get_component()
            .set_selected(menu_3dmod_startup_window);
    }

    /// Java package-private `setMenu3dmodBinBy2(boolean)`.
    pub fn set_menu_3dmod_bin_by_2(&self, menu_3dmod_bin_by_2: bool) {
        self.menu_3dmod_bin_by_2
            .get_component()
            .set_selected(menu_3dmod_bin_by_2);
    }

    /// Java package-private `doClickFileExit()`.
    pub fn do_click_file_exit(&self) {
        self.menu_exit.do_click();
    }

    /// Java package-private `setEnabledLogWindow(boolean)`.
    pub fn set_enabled_log_window(&self, enable: bool) {
        self.menu_log_window.set_enabled(enable);
    }

    /// Java package-private `setEnabledNewTomogram(boolean)`.
    pub fn set_enabled_new_tomogram(&self, enable: bool) {
        self.menu_new_tomogram.set_enabled(enable);
    }

    /// Java package-private `setEnabledNewJoin(boolean)`.
    pub fn set_enabled_new_join(&self, enable: bool) {
        self.menu_new_join.set_enabled(enable);
    }

    /// Java package-private `setEnabledNewGenericParallel(boolean)`.
    pub fn set_enabled_new_generic_parallel(&self, enable: bool) {
        self.menu_new_generic_parallel.set_enabled(enable);
    }

    /// Java package-private `setEnabledNewAnisotropicDiffusion(boolean)`.
    pub fn set_enabled_new_anisotropic_diffusion(&self, enable: bool) {
        self.menu_new_anisotropic_diffusion.set_enabled(enable);
    }

    /// Java package-private `setEnabledNewBatchRunTomo(boolean)`.
    pub fn set_enabled_new_batch_run_tomo(&self, enable: bool) {
        self.menu_new_batch_run_tomo.set_enabled(enable);
    }

    /// Java package-private `setEnabledNewPeet(boolean)`.
    pub fn set_enabled_new_peet(&self, enable: bool) {
        self.menu_new_peet.set_enabled(enable);
    }

    /// Java package-private `setEnabledNewSerialSections(boolean)`.
    pub fn set_enabled_new_serial_sections(&self, enable: bool) {
        self.menu_serial_sections.set_enabled(enable);
    }

    /// Java package-private `equalsNewTomogram(ActionEvent)`.
    pub fn equals_new_tomogram(&self, event: &ActionEvent) -> bool {
        self.equals(&self.menu_new_tomogram.get_component(), event)
    }

    /// Java package-private `equalsNewJoin(ActionEvent)`.
    pub fn equals_new_join(&self, event: &ActionEvent) -> bool {
        self.equals(&self.menu_new_join.get_component(), event)
    }

    /// Java package-private `equalsNewGenericParallel(ActionEvent)`.
    pub fn equals_new_generic_parallel(&self, event: &ActionEvent) -> bool {
        self.equals(&self.menu_new_generic_parallel.get_component(), event)
    }

    /// Java package-private `equalsNewAnisotropicDiffusion(ActionEvent)`.
    pub fn equals_new_anisotropic_diffusion(&self, event: &ActionEvent) -> bool {
        self.equals(&self.menu_new_anisotropic_diffusion.get_component(), event)
    }

    /// Java package-private `equalsNewBatchRunTomo(ActionEvent)`.
    pub fn equals_new_batch_run_tomo(&self, event: &ActionEvent) -> bool {
        self.equals(&self.menu_new_batch_run_tomo.get_component(), event)
    }

    /// Java package-private `equalsNewPeet(ActionEvent)`.
    pub fn equals_new_peet(&self, event: &ActionEvent) -> bool {
        self.equals(&self.menu_new_peet.get_component(), event)
    }

    /// Java package-private `equalsNewSerialSections(ActionEvent)`.
    pub fn equals_new_serial_sections(&self, event: &ActionEvent) -> bool {
        self.equals(&self.menu_serial_sections.get_component(), event)
    }

    /// Java package-private `equalsOpen(ActionEvent)`.
    pub fn equals_open(&self, event: &ActionEvent) -> bool {
        self.equals(&self.menu_open.get_component(), event)
    }

    /// Java package-private `isMenuSaveEnabled()`.
    pub fn is_menu_save_enabled(&self) -> bool {
        self.menu_save.is_enabled()
    }

    /// Java package-private `equalsExit(ActionEvent)`.
    pub fn equals_exit(&self, event: &ActionEvent) -> bool {
        self.equals(&self.menu_exit.get_component(), event)
    }

    /// Java package-private `equalsTomosnapshot(ActionEvent)`.
    pub fn equals_tomosnapshot(&self, event: &ActionEvent) -> bool {
        self.equals(&self.menu_tomosnapshot.get_component(), event)
    }

    /// Java package-private `equalsFlattenVolume(ActionEvent)`.
    pub fn equals_flatten_volume(&self, event: &ActionEvent) -> bool {
        self.equals(&self.menu_flatten_volume.get_component(), event)
    }

    /// Java package-private `equalsGpuTiltTest(ActionEvent)`.
    pub fn equals_gpu_tilt_test(&self, event: &ActionEvent) -> bool {
        self.equals(&self.menu_gpu_tilt_test.get_component(), event)
    }

    /// Java package-private `equalsAlignFrames(ActionEvent)`.
    pub fn equals_align_frames(&self, event: &ActionEvent) -> bool {
        self.equals(&self.menu_align_frames.get_component(), event)
    }

    /// Java package-private `equalsSettings(ActionEvent)`.
    pub fn equals_settings(&self, event: &ActionEvent) -> bool {
        self.equals(&self.menu_settings.get_component(), event)
    }

    /// Java package-private `equalsAxisA(ActionEvent)`.
    pub fn equals_axis_a(&self, event: &ActionEvent) -> bool {
        self.equals(&self.menu_axis_a.get_component(), event)
    }

    /// Java package-private `equalsAxisB(ActionEvent)`.
    pub fn equals_axis_b(&self, event: &ActionEvent) -> bool {
        self.equals(&self.menu_axis_b.get_component(), event)
    }

    /// Java package-private `equalsAxisBoth(ActionEvent)`.
    pub fn equals_axis_both(&self, event: &ActionEvent) -> bool {
        self.equals(&self.menu_axis_both.get_component(), event)
    }

    /// Java package-private `equals3dmodStartUpWindow(ActionEvent)`.
    pub fn equals_3dmod_start_up_window(&self, event: &ActionEvent) -> bool {
        self.equals(&self.menu_3dmod_startup_window.get_component(), event)
    }

    /// Java package-private `equals3dmodBinBy2(ActionEvent)`.
    pub fn equals_3dmod_bin_by_2(&self, event: &ActionEvent) -> bool {
        self.equals(&self.menu_3dmod_bin_by_2.get_component(), event)
    }

    /// Java package-private `equalsLogWindow(ActionEvent)`.
    pub fn equals_log_window(&self, event: &ActionEvent) -> bool {
        self.equals(&self.menu_log_window.get_component(), event)
    }

    /// Java package-private `equalsFitWindow(ActionEvent)`.
    pub fn equals_fit_window(&self, event: &ActionEvent) -> bool {
        self.equals(&self.menu_fit_window.get_component(), event)
    }

    /// Java package-private `equalsTomoGuide(ActionEvent)`.
    pub fn equals_tomo_guide(&self, event: &ActionEvent) -> bool {
        self.equals(&self.menu_tomo_guide.get_component(), event)
    }

    /// Java package-private `equalsImodGuide(ActionEvent)`.
    pub fn equals_imod_guide(&self, event: &ActionEvent) -> bool {
        self.equals(&self.menu_imod_guide.get_component(), event)
    }

    /// Java package-private `equals3dmodGuide(ActionEvent)`.
    pub fn equals_3dmod_guide(&self, event: &ActionEvent) -> bool {
        self.equals(&self.menu_3dmod_guide.get_component(), event)
    }

    /// Java package-private `equalsEtomoGuide(ActionEvent)`.
    pub fn equals_etomo_guide(&self, event: &ActionEvent) -> bool {
        self.equals(&self.menu_etomo_guide.get_component(), event)
    }

    /// Java package-private `equalsJoinGuide(ActionEvent)`.
    pub fn equals_join_guide(&self, event: &ActionEvent) -> bool {
        self.equals(&self.menu_join_guide.get_component(), event)
    }

    /// Java package-private `equalsPeetGuide(ActionEvent)`.
    pub fn equals_peet_guide(&self, event: &ActionEvent) -> bool {
        self.equals(&self.menu_peet_guide.get_component(), event)
    }

    /// Java package-private `equalsBatchGuide(ActionEvent)`.
    pub fn equals_batch_guide(&self, event: &ActionEvent) -> bool {
        self.equals(&self.menu_batch_guide.get_component(), event)
    }

    /// Java package-private `equalsPeetHelp(ActionEvent)`.
    pub fn equals_peet_help(&self, event: &ActionEvent) -> bool {
        self.equals(&self.peet_help_item.get_component(), event)
    }

    /// Java package-private `equalsHelpAbout(ActionEvent)`.
    pub fn equals_help_about(&self, event: &ActionEvent) -> bool {
        self.equals(&self.menu_help_about.get_component(), event)
    }

    /// Java package-private `equalsDirectiveFileEditor(ActionEvent)`.
    pub fn equals_directive_file_editor(&self, event: &ActionEvent) -> bool {
        self.equals(&self.menu_save_scope.get_component(), event)
            || self.equals(&self.menu_save_system.get_component(), event)
            || self.equals(&self.menu_save_user.get_component(), event)
            || self.equals(&self.menu_export_batch.get_component(), event)
    }

    /// Java package-private `getDirectiveFileType(ActionEvent)`.
    pub fn get_directive_file_type(&self, event: &ActionEvent) -> Option<DirectiveFileType> {
        if self.equals(&self.menu_save_scope.get_component(), event) {
            return Some(DirectiveFileType::Scope);
        }
        if self.equals(&self.menu_save_system.get_component(), event) {
            return Some(DirectiveFileType::System);
        }
        if self.equals(&self.menu_save_user.get_component(), event) {
            return Some(DirectiveFileType::User);
        }
        if self.equals(&self.menu_export_batch.get_component(), event) {
            return Some(DirectiveFileType::Batch);
        }
        None
    }

    /// Java `public boolean equals(JMenuItem, ActionEvent)`:
    /// `menuItem.getActionCommand().equals(event.getActionCommand())`.
    pub fn equals(&self, menu_item: &Rc<JComponent>, event: &ActionEvent) -> bool {
        menu_item
            .get_action_command()
            .is_some_and(|command| Some(command.as_str()) == event.get_action_command())
    }
}

/// Java `private static final class FileActionListener implements ActionListener`.
struct FileActionListener {
    adaptee: Weak<dyn AbstractFrameVirtual>,
}

impl FileActionListener {
    fn new(adaptee: Weak<dyn AbstractFrameVirtual>) -> FileActionListener {
        FileActionListener { adaptee }
    }

    /// Java `actionPerformed(ActionEvent)`.
    fn action_performed(&self, event: &ActionEvent) {
        if let Some(adaptee) = self.adaptee.upgrade() {
            adaptee.menu_file_action(event);
        }
    }
}

/// Java `private static final class ToolsActionListener implements ActionListener`.
struct ToolsActionListener {
    adaptee: Weak<dyn AbstractFrameVirtual>,
}

impl ToolsActionListener {
    fn new(adaptee: Weak<dyn AbstractFrameVirtual>) -> ToolsActionListener {
        ToolsActionListener { adaptee }
    }

    /// Java `actionPerformed(ActionEvent)`.
    fn action_performed(&self, event: &ActionEvent) {
        if let Some(adaptee) = self.adaptee.upgrade() {
            adaptee.menu_tools_action(event);
        }
    }
}

/// Java `private static final class FileMRUListActionListener implements
/// ActionListener`.
struct FileMRUListActionListener {
    adaptee: Weak<dyn AbstractFrameVirtual>,
}

impl FileMRUListActionListener {
    fn new(adaptee: Weak<dyn AbstractFrameVirtual>) -> FileMRUListActionListener {
        FileMRUListActionListener { adaptee }
    }

    /// Java `actionPerformed(ActionEvent)`.
    fn action_performed(&self, event: &ActionEvent) {
        if let Some(adaptee) = self.adaptee.upgrade() {
            adaptee.menu_file_mru_list_action(event);
        }
    }
}

/// Java `private static final class ViewActionListener implements ActionListener`.
struct ViewActionListener {
    adaptee: Weak<dyn AbstractFrameVirtual>,
}

impl ViewActionListener {
    fn new(adaptee: Weak<dyn AbstractFrameVirtual>) -> ViewActionListener {
        ViewActionListener { adaptee }
    }

    /// Java `actionPerformed(ActionEvent)`.
    fn action_performed(&self, event: &ActionEvent) {
        if let Some(adaptee) = self.adaptee.upgrade() {
            adaptee.menu_view_action(event);
        }
    }
}

/// Java `private static final class OptionsActionListener implements
/// ActionListener`.
struct OptionsActionListener {
    adaptee: Weak<dyn AbstractFrameVirtual>,
}

impl OptionsActionListener {
    fn new(adaptee: Weak<dyn AbstractFrameVirtual>) -> OptionsActionListener {
        OptionsActionListener { adaptee }
    }

    /// Java `actionPerformed(ActionEvent)`.
    fn action_performed(&self, event: &ActionEvent) {
        if let Some(adaptee) = self.adaptee.upgrade() {
            adaptee.menu_options_action(event);
        }
    }
}

/// Java `private static final class HelpActionListener implements ActionListener`.
struct HelpActionListener {
    adaptee: Weak<dyn AbstractFrameVirtual>,
}

impl HelpActionListener {
    fn new(adaptee: Weak<dyn AbstractFrameVirtual>) -> HelpActionListener {
        HelpActionListener { adaptee }
    }

    /// Java `actionPerformed(ActionEvent)`.
    fn action_performed(&self, event: &ActionEvent) {
        if let Some(adaptee) = self.adaptee.upgrade() {
            adaptee.menu_help_action(event);
        }
    }
}
