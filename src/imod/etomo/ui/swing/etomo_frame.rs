//! `IMOD/Etomo/src/etomo/ui/swing/EtomoFrame.java`.
//!
//! The part of `MainFrame` and `SubFrame` they share: the menu bar, the menu
//! actions common to both, the message functions routed to the frame showing
//! an axis, and the static registration of the one main frame and the one
//! sub frame (`mainFrame`, `subFrame`).
//!
//! The Java statics hold `EtomoFrame` references that `MainFrame` casts back
//! to `SubFrame`; they are EDT-confined `thread_local!`s holding the concrete
//! types here, and [`get_other_frame`](EtomoFrame::get_other_frame) /
//! `getFrame` hand them out as `Rc<dyn EtomoFrameVirtual>`.
//!
//! Java `etomo.type.FrameType` is defined here as well
//! (`type/user_configuration.rs` imports it from this module).

use std::cell::{Cell, RefCell};
use std::ops::Deref;
use std::path::PathBuf;
use std::rc::{Rc, Weak};

use super::abstract_frame::{AbstractFrame, AbstractFrameVirtual};
use super::etomo_menu::EtomoMenu;
use super::file_chooser::{self, FileChooser};
use super::main_frame::MainFrame;
use super::main_panel::MainPanelVirtual;
use super::sub_frame::SubFrame;
use super::ui_harness;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::jdk::{ActionEvent, FileFilter, JComponent};
use crate::imod::etomo::process::process_messages::ProcessMessages;
use crate::imod::etomo::storage::data_file_filter::DataFileFilter;
use crate::imod::etomo::storage::log_file::LogFileError;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::ui::ui_component::UIComponent;

/// Java `etomo.type.FrameType` (`Main`, `Sub`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FrameType {
    /// Java `FrameType.Main`.
    Main,
    /// Java `FrameType.Sub`.
    Sub,
}

impl std::fmt::Display for FrameType {
    /// Java `toString()`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FrameType::Main => f.write_str("Main"),
            FrameType::Sub => f.write_str("Sub"),
        }
    }
}

thread_local! {
    /// Java package-private static `EtomoFrame mainFrame = null`.
    static MAIN_FRAME: RefCell<Option<Rc<MainFrame>>> = const { RefCell::new(None) };
    /// Java package-private static `EtomoFrame subFrame = null`.
    static SUB_FRAME: RefCell<Option<Rc<SubFrame>>> = const { RefCell::new(None) };
}

/// Java static `mainFrame`.
pub fn main_frame() -> Option<Rc<MainFrame>> {
    MAIN_FRAME.with(|frame| frame.borrow().clone())
}

/// Java `mainFrame = ...`.
pub fn set_main_frame(frame: Option<Rc<MainFrame>>) {
    MAIN_FRAME.with(|slot| *slot.borrow_mut() = frame);
}

/// Java static `subFrame`.
pub fn sub_frame() -> Option<Rc<SubFrame>> {
    SUB_FRAME.with(|frame| frame.borrow().clone())
}

/// Java `subFrame = ...`.
pub fn set_sub_frame(frame: Option<Rc<SubFrame>>) {
    SUB_FRAME.with(|slot| *slot.borrow_mut() = frame);
}

/// The members `EtomoFrame` declares abstract or that a subclass overrides
/// and `EtomoFrame` calls.
pub trait EtomoFrameVirtual: AbstractFrameVirtual {
    /// The `EtomoFrame` part of the object.
    fn etomo_frame(&self) -> &EtomoFrame;

    /// Java abstract package-private `register()`.
    fn register(self: Rc<Self>);

    /// Java package-private `moveSubFrame()` (`SubFrame` overrides it).
    fn move_sub_frame(&self) {
        self.etomo_frame().move_sub_frame_super();
    }
}

/// Java package-private `abstract class EtomoFrame extends AbstractFrame`.
pub struct EtomoFrame {
    /// The `AbstractFrame` superclass part.
    base: AbstractFrame,
    /// Java package-private `boolean main`.
    pub main: Cell<bool>,
    /// Java package-private `EtomoMenu menu`.
    pub menu: RefCell<Option<Rc<EtomoMenu>>>,
    /// Java package-private `JMenuBar menuBar`.
    pub menu_bar: RefCell<Option<Rc<JComponent>>>,
    /// Java package-private `MainPanel mainPanel`.
    pub main_panel: RefCell<Option<Rc<dyn MainPanelVirtual>>>,
    /// Java package-private `BaseManager currentManager`.
    pub current_manager: Cell<Option<&'static dyn BaseManager>>,
    /// Java private final `singleFrame`.
    single_frame: bool,
    /// The subclass object (Java `this`).
    this: RefCell<Option<Weak<dyn EtomoFrameVirtual>>>,
}

impl Deref for EtomoFrame {
    type Target = AbstractFrame;
    fn deref(&self) -> &AbstractFrame {
        &self.base
    }
}

impl EtomoFrame {
    /// Java `EtomoFrame()`.
    pub fn new_void() -> EtomoFrame {
        EtomoFrame {
            base: AbstractFrame::new(),
            main: Cell::new(false),
            menu: RefCell::new(None),
            menu_bar: RefCell::new(None),
            main_panel: RefCell::new(None),
            current_manager: Cell::new(None),
            single_frame: false,
            this: RefCell::new(None),
        }
    }

    /// Java `EtomoFrame(boolean)`.
    pub fn new_boolean(single_frame: bool) -> EtomoFrame {
        EtomoFrame {
            base: AbstractFrame::new(),
            main: Cell::new(false),
            menu: RefCell::new(None),
            menu_bar: RefCell::new(None),
            main_panel: RefCell::new(None),
            current_manager: Cell::new(None),
            single_frame,
            this: RefCell::new(None),
        }
    }

    /// Installs the subclass object for virtual dispatch (Rust-only; also
    /// installs it in the `AbstractFrame` part).
    pub fn set_this(&self, this: Weak<dyn EtomoFrameVirtual>) {
        self.base
            .set_this(this.clone() as Weak<dyn AbstractFrameVirtual>);
        *self.this.borrow_mut() = Some(this);
    }

    /// The subclass object (Java `this`).
    pub fn this(&self) -> Option<Rc<dyn EtomoFrameVirtual>> {
        self.this.borrow().as_ref().and_then(Weak::upgrade)
    }

    /// The `EtomoMenu` (Java `menu`); set by `initialize`.
    fn menu(&self) -> Rc<EtomoMenu> {
        self.menu
            .borrow()
            .clone()
            .expect("EtomoFrame.menu is set by initialize()")
    }

    /// Java package-private `initialize()`.
    pub fn initialize(&self) {
        *self.menu.borrow_mut() = Some(EtomoMenu::get_instance_abstract_frame(
            self.base.this_weak(),
        ));
        // Swing: ImageIcon iconEtomo = new ImageIcon(ClassLoader.getSystemResource(
        // "images/etomo.png")); setIconImage(iconEtomo.getImage()).
        self.get_menus();
    }

    /// Java package-private `saveLocation()`.  Saves the current location of
    /// the frame to UserConfiguration.
    pub fn save_location(&self) {
        let is_ignore_loc = etomo_director::ARGUMENTS.lock().unwrap().is_ignore_loc();
        if !is_ignore_loc {
            let frame_type = self.this().and_then(|this| this.get_frame_type());
            let location = self.get_location();
            if let Some(frame_type) = frame_type {
                etomo_director::INSTANCE.with_user_configuration_mut(|c| {
                    c.set_last_location(frame_type, Some((location.x, location.y)))
                });
            }
        }
    }

    /// Java package-private `moveSubFrame()`, the `EtomoFrame` body.
    pub fn move_sub_frame_super(&self) {
        if self.single_frame {
            return;
        }
        if let Some(sub_frame) = sub_frame() {
            sub_frame.move_sub_frame();
        }
    }

    /// Java package-private `toFront(AxisID)`.
    pub fn to_front_axis_id(&self, axis_id: Option<AxisID>) {
        self.get_frame(axis_id).abstract_frame().to_front();
    }

    /// Java final `isMenu3dmodStartupWindow()`.
    pub fn is_menu_3dmod_startup_window(&self) -> bool {
        self.menu().is_menu_3dmod_startup_window()
    }

    /// Java final `isMenuSaveEnabled()`.
    pub fn is_menu_save_enabled(&self) -> bool {
        self.menu().is_menu_save_enabled()
    }

    /// Java final `isMenu3dmodBinBy2()`.
    pub fn is_menu_3dmod_bin_by_2(&self) -> bool {
        self.menu().is_menu_3dmod_bin_by_2()
    }

    /// Java final `setMenu3dmodStartupWindow(boolean)`.
    pub fn set_menu_3dmod_startup_window(&self, menu_3dmod_startup_window: bool) {
        self.menu()
            .set_menu_3dmod_startup_window(menu_3dmod_startup_window);
    }

    /// Java final `setMenu3dmodBinBy2(boolean)`.
    pub fn set_menu_3dmod_bin_by_2(&self, menu_3dmod_bin_by_2: bool) {
        self.menu().set_menu_3dmod_bin_by_2(menu_3dmod_bin_by_2);
    }

    /// Java `menuToolsAction(ActionEvent)`.
    pub fn menu_tools_action(&self, event: &ActionEvent) {
        // `EtomoMenu.menuToolsAction` takes a non-null AxisID; getAxisID() is
        // null only before a main panel is set, and the menu ignores it.
        self.menu()
            .menu_tools_action(self.get_axis_id().unwrap_or(AxisID::Only), event);
    }

    /// Java `menuFileAction(ActionEvent)`.  Handle File menu actions.
    pub fn menu_file_action(&self, event: &ActionEvent) {
        let menu = self.menu();
        if !menu.menu_file_action(event) {
            let axis_id = self.get_axis_id();
            if menu.equals_new_tomogram(event) {
                let _ = etomo_director::INSTANCE.open_tomogram_boolean_axis_id(true, axis_id);
            } else if menu.equals_new_join(event) {
                let _ = etomo_director::INSTANCE.open_join_boolean_axis_id(true, axis_id);
            } else if menu.equals_new_generic_parallel(event) {
                let _ = etomo_director::INSTANCE.open_generic_parallel(true, axis_id);
            } else if menu.equals_new_anisotropic_diffusion(event) {
                let _ = etomo_director::INSTANCE.open_anisotropic_diffusion(true, axis_id);
            } else if menu.equals_new_batch_run_tomo(event) {
                let _ = etomo_director::INSTANCE.open_batch_run_tomo_boolean_axis_id(true, axis_id);
            } else if menu.equals_new_peet(event) {
                // TODO(unit): needs etomo/PeetManager.java - `if
                // (PeetManager.isInterfaceAvailable())
                // EtomoDirector.INSTANCE.openPeet(true, axisID)`.  Without the
                // availability check the PEET manager is not opened.
            } else if menu.equals_new_serial_sections(event) {
                let _ =
                    etomo_director::INSTANCE.open_serial_sections_boolean_axis_id(true, axis_id);
            } else if menu.equals_open(event) {
                let data_file = self.open_data_file_dialog();
                if let Some(data_file) = data_file {
                    let ui_component: &dyn UIComponent = &self.base;
                    let _ = etomo_director::INSTANCE.open_manager(
                        &data_file,
                        true,
                        axis_id,
                        Some(ui_component),
                    );
                }
            } else if menu.equals_exit(event) {
                ui_harness::with(|harness| harness.exit(axis_id, 0));
            } else if menu.equals_tomosnapshot(event) {
                if let Some(current_manager) = self.current_manager.get() {
                    current_manager.tomosnapshot(axis_id);
                }
            } else if menu.equals_directive_file_editor(event) {
                let directive_file_type = menu.get_directive_file_type(event);
                if let Some(directive_file_type) = directive_file_type {
                    // Upstream bug fixed (EtomoFrame.java:144): Java calls
                    // currentManager.saveAll with no current manager and throws
                    // NullPointerException; nothing is done here instead.
                    let Some(current_manager) = self.current_manager.get() else {
                        return;
                    };
                    let mut errmsg = String::new();
                    let timestamp = current_manager.save_all(&mut errmsg);
                    if let Some(timestamp) = timestamp {
                        etomo_director::INSTANCE.open_directive_editor(
                            Some(directive_file_type),
                            Some(current_manager),
                            Some(&timestamp),
                            Some(&errmsg),
                        );
                    }
                }
            }
        }
    }

    /// Java `close()`.
    pub fn close(&self) {
        etomo_director::INSTANCE.close_current_manager(self.get_axis_id(), false);
    }

    /// Java `cancel()`: empty.
    pub fn cancel(&self) {}

    /// Java `save(AxisID)`.
    pub fn save(&self, axis_id: Option<AxisID>) {
        // Upstream bug fixed (EtomoFrame.java:180): with no current manager the
        // Java throws NullPointerException; nothing is saved instead.
        let Some(current_manager) = self.current_manager.get() else {
            return;
        };
        // try {
        let result: Result<(), LogFileError> = (|| {
            if current_manager.save_param_file()? {
                return Ok(());
            }
            // Don't allow the user to do the equivalent of a Save As if Save As isn't
            // available.
            if !current_manager.can_change_param_file_name() {
                ui_harness::with(|harness| {
                    harness.open_message_dialog_base_manager_string_string(
                        Some(current_manager),
                        "Please set the name of dataset or the join before saving",
                        "Cannot Save",
                    )
                });
                return Ok(());
            }
            // Do a Save As
            if self.get_param_filename() {
                current_manager.save_param_file()?;
            }
            Ok(())
        })();
        match result {
            Ok(()) => {}
            // catch (final LockException e) {}
            Err(LogFileError::Lock(_)) => {}
            // catch (final LogFileException | IOException e)
            Err(e) => ui_harness::with(|harness| {
                harness.open_message_dialog_base_manager_string_string_axis_id(
                    Some(current_manager),
                    &format!("Unable to write parameters.\n{e}"),
                    "Etomo Error",
                    axis_id,
                )
            }),
        }
    }

    /// Java `saveAs()`.
    pub fn save_as(&self) {
        // try {
        if self.get_param_filename() {
            // Upstream bug fixed (EtomoFrame.java:206): with no current manager
            // the Java throws NullPointerException; nothing is saved instead.
            let Some(current_manager) = self.current_manager.get() else {
                return;
            };
            match current_manager.save_param_file() {
                Ok(_) => {}
                // catch (final LockException e) {}
                Err(LogFileError::Lock(_)) => {}
                // catch (final LogFileException | IOException e)
                Err(e) => {
                    let axis_id = self.get_axis_id();
                    ui_harness::with(|harness| {
                        harness.open_message_dialog_base_manager_string_string_axis_id(
                            Some(current_manager),
                            &format!("Unable to save parameters.\n{e}"),
                            "Etomo Error",
                            axis_id,
                        )
                    })
                }
            }
        }
    }

    /// Java `menuFileMRUListAction(ActionEvent)`.  Open the specified MRU EDF
    /// file.
    pub fn menu_file_mru_list_action(&self, event: &ActionEvent) {
        let data_file = PathBuf::from(event.get_action_command().unwrap_or(""));
        let ui_component: &dyn UIComponent = &self.base;
        let _ = etomo_director::INSTANCE.open_manager(
            &data_file,
            true,
            self.get_axis_id(),
            Some(ui_component),
        );
    }

    /// Java `menuHelpAction(ActionEvent)`.  Handle help menu actions.
    pub fn menu_help_action(&self, event: &ActionEvent) {
        let frame = self.get_content_pane();
        self.menu().menu_help_action(
            self.current_manager.get(),
            self.get_axis_id().unwrap_or(AxisID::Only),
            &frame,
            event,
        );
    }

    /// Java `menuViewAction(ActionEvent)`, the `EtomoFrame` body.  Handle some
    /// of the view menu events.  Axis switch events should be handled in the
    /// child classes.
    pub fn menu_view_action(&self, event: &ActionEvent) {
        // Run fitWindow on both frames.
        if self.menu().equals_fit_window(event) {
            let current_manager = self.current_manager.get();
            ui_harness::with(|harness| harness.pack_boolean_base_manager(true, current_manager));
            if self.get_other_frame().is_some() {
                ui_harness::with(|harness| {
                    harness.pack_axis_id_boolean_base_manager(
                        Some(AxisID::Second),
                        true,
                        current_manager,
                    )
                });
            }
        } else {
            // Upstream behaviour kept as Swing reports it: the
            // IllegalStateException thrown on the EDT is printed and the event
            // is dropped.
            eprintln!(
                "Exception in thread \"AWT-EventQueue-0\" java.lang.IllegalStateException: Cannot handled menu command in this class.  command={}",
                event.get_action_command().unwrap_or("null")
            );
        }
    }

    /// Java `menuOptionsAction(ActionEvent)`.  Handle some of the options menu
    /// events.
    pub fn menu_options_action(&self, event: &ActionEvent) {
        let menu = self.menu();
        if menu.equals_settings(event) {
            let _ = etomo_director::INSTANCE.open_settings_dialog();
        } else if menu.equals_3dmod_start_up_window(event) {
            let frame = self.get_other_frame();
            if let Some(frame) = frame {
                frame
                    .etomo_frame()
                    .set_menu_3dmod_startup_window(self.is_menu_3dmod_startup_window());
            }
        } else if menu.equals_3dmod_bin_by_2(event) {
            let frame = self.get_other_frame();
            if let Some(frame) = frame {
                frame
                    .etomo_frame()
                    .set_menu_3dmod_bin_by_2(self.is_menu_3dmod_bin_by_2());
            }
        } else {
            // IllegalStateException on the EDT: printed, event dropped.
            eprintln!(
                "Exception in thread \"AWT-EventQueue-0\" java.lang.IllegalStateException: Cannot handled menu command in this class.  command={}",
                event.get_action_command().unwrap_or("null")
            );
        }
    }

    /// Java package-private `setEnabled(BaseManager)`.  Enable/disable menu
    /// items.  Also run this function on the other frame since it changes the
    /// menu's appearance.
    pub fn set_enabled_base_manager(&self, current_manager: Option<&'static dyn BaseManager>) {
        self.menu().set_enabled_base_manager(current_manager);
        let other_frame = self.get_other_frame();
        if let Some(other_frame) = other_frame {
            other_frame
                .etomo_frame()
                .menu()
                .set_enabled_base_manager(current_manager);
        }
    }

    /// Java package-private `setMRUFileLabels(String[])`.  Set the MRU etomo
    /// data file list.  Also run this function on the other frame.
    pub fn set_mru_file_labels(&self, m_ru_list: &[String]) {
        self.menu().set_mru_file_labels(m_ru_list);
        let other_frame = self.get_other_frame();
        if let Some(other_frame) = other_frame {
            other_frame
                .etomo_frame()
                .menu()
                .set_mru_file_labels(m_ru_list);
        }
    }

    /// Java package-private `setEnabledNewTomogramMenuItem(boolean)`.
    pub fn set_enabled_new_tomogram_menu_item(&self, enable: bool) {
        self.menu().set_enabled_new_tomogram(enable);
        let other_frame = self.get_other_frame();
        if let Some(other_frame) = other_frame {
            other_frame
                .etomo_frame()
                .menu()
                .set_enabled_new_tomogram(enable);
        }
    }

    /// Java package-private `setEnabledLogWindowMenuItem(boolean)`.
    pub fn set_enabled_log_window_menu_item(&self, enable: bool) {
        self.menu().set_enabled_log_window(enable);
        let other_frame = self.get_other_frame();
        if let Some(other_frame) = other_frame {
            other_frame
                .etomo_frame()
                .menu()
                .set_enabled_log_window(enable);
        }
    }

    /// Java package-private `setEnabledNewJoinMenuItem(boolean)`.
    pub fn set_enabled_new_join_menu_item(&self, enable: bool) {
        self.menu().set_enabled_new_join(enable);
        let other_frame = self.get_other_frame();
        if let Some(other_frame) = other_frame {
            other_frame
                .etomo_frame()
                .menu()
                .set_enabled_new_join(enable);
        }
    }

    /// Java package-private `setEnabledNewGenericParallelMenuItem(boolean)`.
    pub fn set_enabled_new_generic_parallel_menu_item(&self, enable: bool) {
        self.menu().set_enabled_new_generic_parallel(enable);
        let other_frame = self.get_other_frame();
        if let Some(other_frame) = other_frame {
            other_frame
                .etomo_frame()
                .menu()
                .set_enabled_new_generic_parallel(enable);
        }
    }

    /// Java package-private `setEnabledNewAnisotropicDiffusionMenuItem(boolean)`.
    pub fn set_enabled_new_anisotropic_diffusion_menu_item(&self, enable: bool) {
        self.menu().set_enabled_new_anisotropic_diffusion(enable);
        let other_frame = self.get_other_frame();
        if let Some(other_frame) = other_frame {
            other_frame
                .etomo_frame()
                .menu()
                .set_enabled_new_anisotropic_diffusion(enable);
        }
    }

    /// Java package-private `setEnabledNewBatchRunTomoMenuItem(boolean)`.
    pub fn set_enabled_new_batch_run_tomo_menu_item(&self, enable: bool) {
        self.menu().set_enabled_new_batch_run_tomo(enable);
        let other_frame = self.get_other_frame();
        if let Some(other_frame) = other_frame {
            other_frame
                .etomo_frame()
                .menu()
                .set_enabled_new_batch_run_tomo(enable);
        }
    }

    /// Java package-private `setEnabledNewPeetMenuItem(boolean)`.
    pub fn set_enabled_new_peet_menu_item(&self, enable: bool) {
        self.menu().set_enabled_new_peet(enable);
        let other_frame = self.get_other_frame();
        if let Some(other_frame) = other_frame {
            other_frame
                .etomo_frame()
                .menu()
                .set_enabled_new_peet(enable);
        }
    }

    /// Java package-private `setEnabledNewSerialSectionsMenuItem(boolean)`.
    pub fn set_enabled_new_serial_sections_menu_item(&self, enable: bool) {
        self.menu().set_enabled_new_serial_sections(enable);
        let other_frame = self.get_other_frame();
        if let Some(other_frame) = other_frame {
            other_frame
                .etomo_frame()
                .menu()
                .set_enabled_new_serial_sections(enable);
        }
    }

    /// Java `repaint(AxisID)` (overrides `AbstractFrame`).
    pub fn repaint(&self, axis_id: Option<AxisID>) {
        let _frame = self.get_frame(axis_id);
        // Swing painting: getFrame(axisID).repaint().
    }

    /// Java `displayMessage(BaseManager, String, String, AxisID)` (override).
    /// Open a message dialog.
    pub fn display_message_base_manager_string_string_axis_id(
        &self,
        manager: Option<&'static dyn BaseManager>,
        message: Option<&str>,
        title: Option<&str>,
        axis_id: Option<AxisID>,
    ) {
        self.get_frame(axis_id)
            .abstract_frame()
            .open_message_dialog_base_manager_axis_id_string_string(
                manager, axis_id, message, title,
            );
    }

    /// Java package-private `displayInfoMessage(BaseManager, String, String,
    /// AxisID)` (an `EtomoFrame` overload; not an override).
    pub fn display_info_message_base_manager_string_string_axis_id(
        &self,
        manager: Option<&'static dyn BaseManager>,
        message: Option<&str>,
        title: Option<&str>,
        axis_id: Option<AxisID>,
    ) {
        self.get_frame(axis_id)
            .abstract_frame()
            .open_info_message_dialog_base_manager_component_axis_id_string_string(
                manager, None, axis_id, message, title,
            );
    }

    /// Java `displayMessage(BaseManager, String, String)` (override).
    pub fn display_message_base_manager_string_string(
        &self,
        manager: Option<&'static dyn BaseManager>,
        message: Option<&str>,
        title: Option<&str>,
    ) {
        self.get_frame(Some(AxisID::Only))
            .abstract_frame()
            .open_message_dialog_base_manager_axis_id_string_string(
                manager,
                Some(AxisID::Only),
                message,
                title,
            );
    }

    /// Java `displayMessage(BaseManager, String[], String, AxisID)` (override).
    pub fn display_message_base_manager_string_array_string_axis_id(
        &self,
        manager: Option<&'static dyn BaseManager>,
        message: &[String],
        title: Option<&str>,
        axis_id: Option<AxisID>,
    ) {
        self.get_frame(axis_id)
            .abstract_frame()
            .open_message_dialog_base_manager_axis_id_string_array_string(
                manager, axis_id, message, title,
            );
    }

    /// Java `displayErrorMessage(BaseManager, ProcessMessages, String, AxisID)`
    /// (override).
    pub fn display_error_message(
        &self,
        manager: Option<&'static dyn BaseManager>,
        process_messages: &ProcessMessages,
        title: Option<&str>,
        axis_id: Option<AxisID>,
    ) {
        self.get_frame(axis_id)
            .abstract_frame()
            .open_error_message_dialog_base_manager_axis_id_process_messages_string(
                manager,
                axis_id,
                process_messages,
                title,
            );
    }

    /// Java `displayWarningMessage(BaseManager, ProcessMessages, String,
    /// AxisID)` (override).
    pub fn display_warning_message_base_manager_process_messages_string_axis_id(
        &self,
        manager: Option<&'static dyn BaseManager>,
        process_messages: &ProcessMessages,
        title: Option<&str>,
        axis_id: Option<AxisID>,
    ) {
        self.get_frame(axis_id)
            .abstract_frame()
            .open_warning_message_dialog_base_manager_axis_id_process_messages_string(
                manager,
                axis_id,
                process_messages,
                title,
            );
    }

    /// Java `displayYesNoCancelMessage(BaseManager, String, AxisID)`
    /// (override).
    pub fn display_yes_no_cancel_message(
        &self,
        manager: Option<&'static dyn BaseManager>,
        message: Option<&str>,
        axis_id: Option<AxisID>,
    ) -> i32 {
        self.get_frame(axis_id)
            .abstract_frame()
            .open_yes_no_cancel_dialog_base_manager_axis_id_string(manager, axis_id, message)
    }

    /// Java `displayYesNoMessage(BaseManager, String, AxisID)` (override).
    pub fn display_yes_no_message_base_manager_string_axis_id(
        &self,
        manager: Option<&'static dyn BaseManager>,
        message: Option<&str>,
        axis_id: Option<AxisID>,
    ) -> bool {
        self.get_frame(axis_id)
            .abstract_frame()
            .open_yes_no_dialog_base_manager_axis_id_string(manager, axis_id, message)
    }

    /// Java `displayDeleteMessage(BaseManager, String[], AxisID)` (override).
    pub fn display_delete_message(
        &self,
        manager: Option<&'static dyn BaseManager>,
        message: &[String],
        axis_id: Option<AxisID>,
    ) -> bool {
        self.get_frame(axis_id)
            .abstract_frame()
            .open_delete_dialog_base_manager_axis_id_string_array(manager, axis_id, message)
    }

    /// Java `displayYesNoWarningDialog(BaseManager, String, AxisID)`
    /// (override).
    pub fn display_yes_no_warning_dialog(
        &self,
        manager: Option<&'static dyn BaseManager>,
        message: Option<&str>,
        axis_id: Option<AxisID>,
    ) -> bool {
        self.get_frame(axis_id)
            .abstract_frame()
            .open_yes_no_warning_dialog_base_manager_axis_id_string(manager, axis_id, message)
    }

    /// Java `displayYesNoMessage(BaseManager, String[], AxisID)` (override).
    pub fn display_yes_no_message_base_manager_string_array_axis_id(
        &self,
        manager: Option<&'static dyn BaseManager>,
        message: &[String],
        axis_id: Option<AxisID>,
    ) -> bool {
        self.get_frame(axis_id)
            .abstract_frame()
            .open_yes_no_dialog_base_manager_axis_id_string_array(manager, axis_id, message)
    }

    /// Java private `getParamFilename()`.  Open a file chooser to get an
    /// .edf, .ejf, .epp, or .epe file.  Returns true if succeeded.
    fn get_param_filename(&self) -> bool {
        let Some(current_manager) = self.current_manager.get() else {
            // Upstream bug fixed (EtomoFrame.java:431): Java throws
            // NullPointerException without a current manager.
            return false;
        };
        // Open up the file chooser in current working directory
        let chooser = FileChooser::new_base_manager(Some(current_manager));
        let file_filter: Option<Rc<dyn FileFilter>> = self
            .main_panel
            .borrow()
            .as_ref()
            .and_then(|main_panel| main_panel.main_panel().get_data_file_filter());
        chooser.set_file_filter(file_filter.clone());
        chooser.set_dialog_title(Some(&format!(
            "Save {}",
            file_filter
                .as_ref()
                .and_then(|filter| filter.get_description())
                .unwrap_or_else(|| "null".to_owned())
        )));
        chooser.set_dialog_type(file_chooser::SAVE_DIALOG);
        // Swing layout: chooser.setPreferredSize(UIParameters.getInstance()
        // .getFileChooserDimension()).
        chooser.set_file_selection_mode(file_chooser::FILES_ONLY);
        let working_dir =
            PathBuf::from(current_manager.get_property_user_dir().unwrap_or_default());
        // File.listFiles(FileFilter).  Upstream bug fixed (EtomoFrame.java:440):
        // listFiles returns null for an unreadable directory and Java throws
        // NullPointerException on edfFiles.length; an unreadable directory
        // counts as having no data files here.
        let edf_files_length = match std::fs::read_dir(&working_dir) {
            Ok(entries) => entries
                .filter_map(Result::ok)
                .filter(|entry| {
                    file_filter
                        .as_ref()
                        .is_none_or(|filter| filter.accept(&entry.path()))
                })
                .count(),
            Err(_) => 0,
        };
        if edf_files_length == 0 {
            let meta_data_file_name = current_manager
                .get_base_meta_data()
                .and_then(|meta_data| meta_data.get_meta_data_file_name());
            if let Some(meta_data_file_name) = meta_data_file_name {
                chooser.set_selected_file(Some(working_dir.join(meta_data_file_name).as_path()));
            }
        }
        let this_component = self.get_content_pane();
        let return_val = chooser.show_save_dialog(Some(&this_component));

        if return_val != file_chooser::APPROVE_OPTION {
            return false;
        }
        // If the file does not already have an extension appended then add the
        // extension
        let Some(mut data_file) = chooser.get_selected_file() else {
            return false;
        };
        let file_name = data_file
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        if !file_name.contains('.') {
            let absolute = std::path::absolute(&data_file).unwrap_or_else(|_| data_file.clone());
            let extension = current_manager
                .get_base_meta_data()
                .and_then(|meta_data| meta_data.base().get_file_extension())
                .unwrap_or_else(|| "null".to_owned());
            data_file = PathBuf::from(format!("{}{}", absolute.display(), extension));
        }
        current_manager.set_param_file_from(Some(data_file.as_path()))
    }

    /// Java `pack(AxisID)` (override).
    pub fn pack_axis_id(&self, axis_id: Option<AxisID>) {
        self.get_frame(axis_id).pack_void();
    }

    /// Java `pack(AxisID, boolean)` (override).
    pub fn pack_axis_id_boolean(&self, axis_id: Option<AxisID>, force: bool) {
        self.get_frame(axis_id).pack_boolean(force);
    }

    /// Java public `pack()` (overrides `Window.pack`).  Increase the bounds by
    /// one pixel before packing.  This preserves the scrollbar when the window
    /// size doesn't change.
    pub fn pack_void(&self) {
        let mut _busy_status_mediator = None;
        let _axis_id = self.get_axis_id();
        if let Some(current_manager) = self.current_manager.get() {
            _busy_status_mediator = Some(current_manager.get_busy_status_mediator());
        }
        if let Some(this) = self.this() {
            this.pack_boolean(false);
        }
    }

    /// Java private `getMenus()`.  Get the Etomo menus.
    fn get_menus(&self) {
        let menu_bar = self.menu().get_menu_bar();
        *self.menu_bar.borrow_mut() = Some(menu_bar.clone());
        self.set_j_menu_bar(Some(menu_bar));
    }

    /// Java private `openDataFileDialog()`.  Open a File Chooser dialog with a
    /// data file filter; if the user selects or names a file return it,
    /// otherwise return null.
    fn open_data_file_dialog(&self) -> Option<PathBuf> {
        // Open up the file chooser in current working directory
        let user_dir = std::env::current_dir()
            .map(|dir| dir.to_string_lossy().into_owned())
            .ok();
        let chooser =
            FileChooser::new_base_manager_string(self.current_manager.get(), user_dir.as_deref());
        let file_filter: Rc<dyn FileFilter> = Rc::new(DataFileFilter::new());
        chooser.set_file_filter(Some(file_filter.clone()));
        chooser.set_dialog_title(Some(&format!(
            "Open {}",
            file_filter
                .get_description()
                .unwrap_or_else(|| "null".to_owned())
        )));
        // Swing layout: chooser.setPreferredSize(UIParameters.getInstance()
        // .getFileChooserDimension()).
        chooser.set_file_selection_mode(file_chooser::FILES_ONLY);
        let this_component = self.get_content_pane();
        let return_val = chooser.show_open_dialog(Some(&this_component));
        if return_val == file_chooser::APPROVE_OPTION {
            return chooser.get_selected_file();
        }
        None
    }

    /// Java `getAxisID()` (overrides `AbstractFrame`).
    pub fn get_axis_id(&self) -> Option<AxisID> {
        if !self.main.get() {
            return Some(AxisID::Second);
        }
        let main_panel = self.main_panel.borrow().clone();
        let Some(main_panel) = main_panel else {
            return None;
        };
        if main_panel.main_panel().get_axis_type() == AxisType::SingleAxis
            || main_panel.main_panel().is_showing_setup()
        {
            return Some(AxisID::Only);
        }
        if !main_panel.main_panel().is_showing_axis_a() {
            return Some(AxisID::Second);
        }
        Some(AxisID::First)
    }

    /// Java package-private `getOtherFrame()`.
    pub fn get_other_frame(&self) -> Option<Rc<dyn EtomoFrameVirtual>> {
        if self.single_frame {
            return self.this();
        }
        if self.main.get() {
            return sub_frame().map(|frame| frame as Rc<dyn EtomoFrameVirtual>);
        }
        main_frame().map(|frame| frame as Rc<dyn EtomoFrameVirtual>)
    }

    /// Java private `getFrame(AxisID)`.  Used by single axis functions.  Gets
    /// the frame containing the axis.
    fn get_frame(&self, axis_id: Option<AxisID>) -> Rc<dyn EtomoFrameVirtual> {
        if self.single_frame {
            if let Some(this) = self.this() {
                return this;
            }
        }
        // Java throws NullPointerException("MainFrame instance was not
        // registered.") here; every EtomoFrame is created after the main frame
        // registers itself, so this cannot happen.
        let main_frame: Rc<dyn EtomoFrameVirtual> =
            main_frame().expect("MainFrame instance was not registered.");
        if axis_id != Some(AxisID::Second) {
            return main_frame;
        }
        let showing_both_axis = self
            .main_panel
            .borrow()
            .as_ref()
            .is_some_and(|main_panel| main_panel.main_panel().is_showing_both_axis());
        if showing_both_axis {
            if let Some(sub_frame) = sub_frame() {
                return sub_frame;
            }
        }
        main_frame
    }
}
