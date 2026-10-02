//! `IMOD/Etomo/src/etomo/FrontPageManager.java`.
//!
//! The manager of the front page, the default window eTomo shows when it is
//! started without a data file.  Its `FrontPageDialog` holds the buttons that
//! open the other interfaces.
//!
//! **Construction.**  As for every manager (see `base_manager.rs`), the Java
//! constructor becomes a constructor function that leaks the allocation and
//! then runs `BaseManager`'s constructor body (`base_manager`), the field
//! initialisers that need `this` (`dialogExpert`), and this class's body.
//! Fields the Java assigns after `super()` (`metaData`, `dialogExpert`) are
//! `OnceLock`s, null until then, as in the Java.

use std::rc::Rc;
use std::sync::OnceLock;

use crate::imod::etomo::base_manager::{BaseManager, BaseManagerBase};
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::local_arguments::LocalArguments;
use crate::imod::etomo::process::base_process_manager::BaseProcessManager;
use crate::imod::etomo::process::front_page_process_manager::FrontPageProcessManager;
use crate::imod::etomo::storage::log_file::LogFileError;
use crate::imod::etomo::storage::storable::Storable;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::base_meta_data::BaseMetaData;
use crate::imod::etomo::r#type::front_page_meta_data::FrontPageMetaData;
use crate::imod::etomo::r#type::image_filename_style::ImageFilenameStyle;
use crate::imod::etomo::r#type::interface_type::InterfaceType;
use crate::imod::etomo::ui::front_page_ui_harness::FrontPageUIHarness;
use crate::imod::etomo::ui::swing::log_interface::LogInterface;
use crate::imod::etomo::ui::swing::log_window::LogWindow;
use crate::imod::etomo::ui::swing::main_front_page_panel::MainFrontPagePanel;
use crate::imod::etomo::ui::swing::main_panel::MainPanelVirtual;
use crate::imod::etomo::ui::swing::ui_harness;
use crate::imod::etomo::util::event_queue::EdtCell;

/// Java private static final `AXIS_ID`.
pub const AXIS_ID: AxisID = AxisID::Only;

/// Java public final class `FrontPageManager extends BaseManager`.
pub struct FrontPageManager {
    /// The Java superclass part.
    base: BaseManagerBase,
    /// This manager at its final address (Java `this`).  Rust-only: set by
    /// the constructor right after the allocation is leaked, because
    /// `createMainPanel` and `getProcessManager` pass `this` on from methods
    /// that take `&self`.
    this_ref: OnceLock<&'static FrontPageManager>,
    /// Java private final `dialogExpert = new FrontPageUIHarness(this,
    /// AXIS_ID)`.  A field initialiser that runs after `super()`.
    dialog_expert: OnceLock<FrontPageUIHarness>,
    /// Java private `mainPanel`.  An event-dispatch-thread object.
    main_panel: EdtCell<Rc<MainFrontPagePanel>>,
    /// Java private final `metaData`, assigned in the constructor body after
    /// `super()`.  Leaked, as the manager is, so `getStorables` can hand it out
    /// as a `&'static dyn Storable`.
    meta_data: OnceLock<&'static FrontPageMetaData>,
    /// Java private `processManager`, initialised to null and created by
    /// `getProcessManager` (synchronized).
    process_manager: OnceLock<&'static FrontPageProcessManager>,
}

/// Owns every `FrontPageManager` this module builds.  Java's owner is the
/// collector, by way of `EtomoDirector.managerList`, which keeps each manager
/// for the run; the translation hands out `&'static Self`, so without a root
/// here the allocation is unreachable the moment the constructor returns.
static INSTANCES: std::sync::Mutex<Vec<&'static FrontPageManager>> =
    std::sync::Mutex::new(Vec::new());

impl FrontPageManager {
    /// Java `FrontPageManager()`: `this(null)`.
    pub fn new() -> &'static Self {
        Self::new_image_filename_style(None)
    }

    /// Java `FrontPageManager(ImageFilenameStyle)`.
    pub fn new_image_filename_style(
        image_filename_style: Option<ImageFilenameStyle>,
    ) -> &'static Self {
        let manager: &'static FrontPageManager = Box::leak(Box::new(FrontPageManager {
            base: BaseManagerBase::initial(),
            this_ref: OnceLock::new(),
            dialog_expert: OnceLock::new(),
            main_panel: EdtCell::new(),
            meta_data: OnceLock::new(),
            process_manager: OnceLock::new(),
        }));
        INSTANCES.lock().unwrap().push(manager);
        let _ = manager.this_ref.set(manager);
        // super()
        manager.base_manager();
        // Field initialiser `dialogExpert = new FrontPageUIHarness(this, AXIS_ID)`.
        let _ = manager
            .dialog_expert
            .set(FrontPageUIHarness::new(manager, AXIS_ID));
        let meta_data: &'static FrontPageMetaData = Box::leak(Box::new(FrontPageMetaData::new(
            Some(manager as &'static dyn BaseManager),
            manager.get_log_properties(),
            image_filename_style,
            true,
        )));
        let _ = manager.meta_data.set(meta_data);
        manager.create_state();
        manager.initialize_ui_parameters_from_name(Some(""), Some(AXIS_ID));
        let headless = etomo_director::ARGUMENTS.lock().unwrap().is_headless();
        if !headless {
            manager.open_processing_panel();
            if let Some(main_panel) = manager.main_panel.get() {
                let param_file = manager.base.param_file.lock().unwrap().clone();
                MainPanelVirtual::set_status_bar_text(
                    &*main_panel,
                    param_file.as_deref(),
                    Some(meta_data as &dyn BaseMetaData),
                    None,
                );
            }
            manager.open_front_page_dialog();
            ui_harness::INSTANCE
                .with(|ui_harness| ui_harness.to_front(Some(manager as &'static dyn BaseManager)));
        }
        manager
    }

    /// Java `getMetaData()`.  Null only while the constructor has not yet
    /// assigned it (during `super()`), as in the Java.
    pub fn get_meta_data(&self) -> Option<&'static FrontPageMetaData> {
        self.meta_data.get().copied()
    }

    /// Java private `createState()`, whose source body is empty.
    fn create_state(&self) {}

    /// Java private `openProcessingPanel()`.
    fn open_processing_panel(&'static self) {
        // Upstream bug fixed in translation (FrontPageManager.java:167): Java
        // dereferences mainPanel unchecked; it is only null when headless, and
        // the constructor calls this only when not headless.
        if let Some(main_panel) = self.main_panel.get() {
            MainPanelVirtual::show_processing_panel(&*main_panel, AxisType::SingleAxis);
        }
        self.set_panel();
    }

    /// Java private `openFrontPageDialog()`.
    fn open_front_page_dialog(&self) {
        if let Some(dialog_expert) = self.dialog_expert.get() {
            dialog_expert.open_dialog();
        }
    }
}

impl BaseManager for FrontPageManager {
    fn base(&self) -> &BaseManagerBase {
        &self.base
    }

    fn this(&'static self) -> &'static dyn BaseManager {
        self
    }

    /// Java `doAutomation(LocalArguments)`.
    fn do_automation(&self, local_arguments: Option<&LocalArguments>) {
        let recon_automation = etomo_director::ARGUMENTS
            .lock()
            .unwrap()
            .is_recon_automation();
        if recon_automation {
            if let Some(dialog_expert) = self.dialog_expert.get() {
                dialog_expert.recon_action_for_automation();
            }
        }
        let directive = etomo_director::ARGUMENTS.lock().unwrap().is_directive();
        if !directive {
            self.do_automation_super(local_arguments);
        }
    }

    /// Java `allowProcessWatching()`.
    fn allow_process_watching(&self) -> bool {
        false
    }

    /// Java `getInterfaceType()`.
    fn get_interface_type(&self) -> Option<InterfaceType> {
        Some(InterfaceType::FrontPage)
    }

    /// Java `getLogInterface()`: null.
    fn get_log_interface(&self) -> Option<Rc<dyn LogInterface>> {
        None
    }

    /// Java package-private `createMainPanel()`.
    fn create_main_panel(&self) {
        if !etomo_director::ARGUMENTS.lock().unwrap().is_headless() {
            // `new MainFrontPagePanel(this)`: the manager at its final address.
            if let Some(this) = self.this_ref.get().copied() {
                self.main_panel.set(Some(MainFrontPagePanel::new(this)));
            }
        }
    }

    /// Java package-private `createLogWindow()`: null.
    fn create_log_window(&'static self) -> Option<Rc<LogWindow>> {
        None
    }

    /// Java `getBaseMetaData()`.
    fn get_base_meta_data(&self) -> Option<&dyn BaseMetaData> {
        self.meta_data
            .get()
            .map(|meta_data| *meta_data as &dyn BaseMetaData)
    }

    /// Java `getMainPanel()`.
    fn get_main_panel(&self) -> Option<Rc<dyn MainPanelVirtual>> {
        self.main_panel
            .get()
            .map(|main_panel| main_panel as Rc<dyn MainPanelVirtual>)
    }

    /// Java package-private `getStorables(int)`.  Java allocates `3 + offset`
    /// slots and fills only the one at `offset`; the rest stay null.
    fn get_storables_with_offset(&self, offset: i32) -> Option<Vec<Option<&'static dyn Storable>>> {
        let mut storables: Vec<Option<&'static dyn Storable>> = vec![None; (3 + offset) as usize];
        let index = offset as usize;
        storables[index] = self
            .meta_data
            .get()
            .map(|meta_data| *meta_data as &'static dyn Storable);
        Some(storables)
    }

    /// Java synchronized `getProcessManager()`.
    fn get_process_manager(&self) -> Option<&'static BaseProcessManager> {
        let this = self.this_ref.get().copied()?;
        let process_manager = self
            .process_manager
            .get_or_init(|| FrontPageProcessManager::new(this));
        Some(&process_manager.base)
    }

    /// Java `kill(AxisID)`, whose source body is empty.
    fn kill(&self, _axis_id: Option<AxisID>) {}

    /// Java `pause(AxisID)`.
    fn pause(&self, _axis_id: Option<AxisID>) -> bool {
        false
    }

    /// Java `save() throws LogFileException, IOException, LockException`.
    fn save(&'static self) -> Result<bool, LogFileError> {
        self.save_super()?;
        // Upstream bug fixed in translation (FrontPageManager.java:140): Java
        // dereferences mainPanel unchecked, which is null when headless; here
        // `done` is skipped then.
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.done();
        }
        Ok(true)
    }

    /// Java `exitProgram(AxisID)`.
    fn exit_program(&'static self, axis_id: Option<AxisID>) -> bool {
        // Java `catch (final Throwable e) { e.printStackTrace(); return true; }`:
        // a panic or a checked exception from saveParamFile ends the same way.
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            if self.exit_program_super(axis_id) {
                self.end_threads();
                if let Err(e) = self.save_param_file() {
                    eprintln!("{e:?}");
                }
                return true;
            }
            false
        }))
        .unwrap_or(true)
    }

    /// Java `getName()`: `metaData.getName()`.
    fn get_name(&self) -> Option<String> {
        self.meta_data
            .get()
            .and_then(|meta_data| meta_data.get_name())
    }
}
