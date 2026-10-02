//! `IMOD/Etomo/src/etomo/ui/swing/AxisProcessPanel.java`.
//!
//! Abstract base of the per-axis process panels (`TomogramProcessPanel`,
//! the join/parallel/tools/... axis panels): the process-select column, the
//! status area (axis progress panel plus the optional parallel panel), and the
//! dialog area into which `MainPanel.showProcess` puts the current dialog.
//!
//! Object model (see `ui.md`): the Java abstract class is the struct
//! [`AxisProcessPanel`], created as an `Rc` (the constructor registers it with
//! the `ProcessingMethodMediator` and `buildParallelPanel` hands it to the
//! `ParallelPanel`, so it needs a handle of its own).  A subclass holds it as
//! `base: Rc<AxisProcessPanel>` and derefs to it.  The methods a subclass
//! overrides and the Java calls virtually are the trait
//! [`AxisProcessPanelVirtual`]; the base stores the subclass as `this`.

use std::cell::{Cell, RefCell};
use std::rc::{Rc, Weak};

use super::axis_progress_panel::AxisProgressPanel;
use super::context_menu::ContextMenu;
use super::context_popup::ContextPopup;
use super::parallel_panel::ParallelPanel;
use super::ui_harness;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::jdk::JComponent;
use crate::imod::etomo::jdk::MouseEvent;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::Type as EtomoNumberType;
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::r#type::interface_type::InterfaceType;
use crate::imod::etomo::ui::field_validation_failed_exception::FieldValidationFailedException;

/// The `AxisProcessPanel` methods a subclass overrides, dispatched
/// virtually.  A subclass implements this trait; its defaults are the
/// `AxisProcessPanel` bodies.
pub trait AxisProcessPanelVirtual {
    /// The embedded `AxisProcessPanel` (the Java superclass part).
    fn axis_process_panel(&self) -> &AxisProcessPanel;

    /// Java `showBothAxis()`; empty in `AxisProcessPanel`.
    fn show_both_axis(&self) {}

    /// Java `createProcessControlPanel()`.
    fn create_process_control_panel(&self) {
        self.axis_process_panel()
            .create_process_control_panel_super();
    }
}

/// Java public abstract class `AxisProcessPanel implements ContextMenu`.
pub struct AxisProcessPanel {
    /// This panel's own handle (Java `this`).
    self_ref: Weak<AxisProcessPanel>,
    /// The subclass object, for virtual dispatch.
    this: RefCell<Weak<dyn AxisProcessPanelVirtual>>,

    /// Java `panelRoot = new JPanel()`.
    panel_root: Rc<JComponent>,
    /// Java `panelProcessInfo = new JPanel()`.
    panel_process_info: Rc<JComponent>,
    /// Java `outerStatusPanel = new JPanel()`.
    outer_status_panel: Rc<JComponent>,
    /// Java `panelDialog = new JPanel()`.
    panel_dialog: Rc<JComponent>,
    /// Java `parallelStatusPanel = new JPanel()`.
    parallel_status_panel: Rc<JComponent>,
    /// Java `lastWidth = new EtomoNumber(EtomoNumber.Type.INTEGER)`.
    last_width: RefCell<EtomoNumber>,
    /// Java package-private `panelProcessSelect = new JPanel()`.
    pub panel_process_select: Rc<JComponent>,

    /// Java package-private `manager`.
    pub manager: &'static dyn BaseManager,
    /// Java package-private `axisID`.
    pub axis_id: AxisID,
    interface_type: InterfaceType,
    popup_chunk_warnings: bool,
    #[allow(dead_code)]
    alt_parallel_loc: bool,
    /// False if parallel processing tables can be displayed, but parallel
    /// processing is not used to run processes for the interface.
    runnable_parallel: bool,
    axis_progress_panel: Rc<AxisProgressPanel>,

    parallel_showing: Cell<bool>,
    /// processingMethodLocked: when on, prevents any changes to visibility of
    /// parallel panel.
    processing_method_locked: Cell<bool>,

    parallel_panel: RefCell<Option<Rc<ParallelPanel>>>,
}

impl AxisProcessPanel {
    /// Java constructor `AxisProcessPanel(AxisID, BaseManager, boolean, boolean,
    /// InterfaceType, boolean, AxisProgressPanel)`.  The subclass must call
    /// [`AxisProcessPanel::set_this`] right after it has been created.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        axis_id: AxisID,
        manager: &'static dyn BaseManager,
        popup_chunk_warnings: bool,
        runnable_parallel: bool,
        interface_type: InterfaceType,
        alt_parallel_loc: bool,
        axis_progress_panel: Rc<AxisProgressPanel>,
    ) -> Rc<AxisProcessPanel> {
        let this = Rc::new_cyclic(|self_ref| AxisProcessPanel {
            self_ref: self_ref.clone(),
            this: RefCell::new(Weak::<NoSubclass>::new() as Weak<dyn AxisProcessPanelVirtual>),
            panel_root: JComponent::new_panel(),
            panel_process_info: JComponent::new_panel(),
            outer_status_panel: JComponent::new_panel(),
            panel_dialog: JComponent::new_panel(),
            parallel_status_panel: JComponent::new_panel(),
            last_width: RefCell::new(EtomoNumber::new_with_type(Some(EtomoNumberType::Integer))),
            panel_process_select: JComponent::new_panel(),
            manager,
            axis_id,
            interface_type,
            popup_chunk_warnings,
            alt_parallel_loc,
            runnable_parallel,
            axis_progress_panel,
            parallel_showing: Cell::new(false),
            processing_method_locked: Cell::new(false),
            parallel_panel: RefCell::new(None),
        });
        // Create the status panel
        // Swing layout: outerStatusPanel.setLayout(new BoxLayout(outerStatusPanel,
        // BoxLayout.Y_AXIS)).
        this.outer_status_panel
            .add(&this.axis_progress_panel.get_component());
        this.parallel_status_panel.set_visible(false);
        if !alt_parallel_loc {
            this.outer_status_panel.add(&this.parallel_status_panel);
        }
        // Java dereferences the mediator unconditionally; it is always set for the
        // managers that build an AxisProcessPanel.
        if let Some(mediator) = manager.get_processing_method_mediator(Some(axis_id)) {
            mediator.register_axis_process_panel(&this);
        }
        this
    }

    /// Installs the subclass object for virtual dispatch (Rust-only; the Java
    /// `this` is the subclass object from the start).
    pub fn set_this(&self, this: Weak<dyn AxisProcessPanelVirtual>) {
        *self.this.borrow_mut() = this;
    }

    /// The subclass object (Java `this` seen through a virtual call).
    fn this(&self) -> Option<Rc<dyn AxisProcessPanelVirtual>> {
        self.this.borrow().upgrade()
    }

    /// Java `setVisible(boolean)`.
    pub fn set_visible(&self, visible: bool) {
        self.panel_root.set_visible(visible);
    }

    // Java `setBackground(Color)`: sets panelRoot, outerStatusPanel,
    // parallelStatusPanel, panelDialog, panelProcessSelect and
    // axisProgressPanel backgrounds - Swing painting, not modelled.

    /// Java `initializePanels()`.
    pub fn initialize_panels(&self) {
        // Swing layout: panelProcessSelect.setAlignmentY(Component.TOP_ALIGNMENT);
        // panelProcessInfo.setAlignmentY(Component.TOP_ALIGNMENT);
        // panelProcessInfo.setLayout(new BorderLayout()).
        if self.manager.get_process_manager().is_some() && self.manager.allow_process_watching() {
            // BorderLayout.NORTH
            self.panel_process_info.add(&self.outer_status_panel);
        }
        // BorderLayout.CENTER
        self.panel_process_info.add(&self.panel_dialog);

        // Swing layout: panelRoot.setLayout(new BoxLayout(panelRoot, BoxLayout.X_AXIS));
        // panelRoot.add(Box.createRigidArea(FixedDim.x5_y0)).
        self.panel_root.add(&self.panel_process_select);
        self.panel_root.add(&self.panel_process_info);
        // Swing layout: panelRoot.add(Box.createRigidArea(FixedDim.x5_y0)).
    }

    /// Java final `hide()`.  Hide the panel if its width is zero because of the
    /// divider.
    pub fn hide(&self) -> bool {
        let mut hide = false;
        if self.get_width() != 0 {
            return hide;
        }
        hide = true;
        self.panel_root.set_visible(false);
        hide
    }

    /// Java `lockProcessingMethod(boolean)`.
    pub fn lock_processing_method(&self, lock: bool) {
        self.processing_method_locked.set(lock);
    }

    /// Java public `showParallelPanel(boolean)`.  Create and show, or hide
    /// parallel panel if necessary.
    pub fn show_parallel_panel_boolean(&self, show: bool) {
        self.show_parallel_panel_boolean_boolean(show, false);
    }

    /// Java `forceShowParallelPanel(boolean)`.  Call showParallelPanel,
    /// ignoring processingMethodLocked.
    pub fn force_show_parallel_panel(&self, show: bool) {
        self.show_parallel_panel_boolean_boolean(show, true);
    }

    /// Java `getCPUsSelectedInt(boolean)`.
    pub fn get_cpus_selected_int(
        &self,
        do_validation: bool,
    ) -> Result<i32, FieldValidationFailedException> {
        let parallel_panel = self.parallel_panel.borrow().clone();
        if let Some(parallel_panel) = parallel_panel {
            return parallel_panel.get_cpus_selected_int(do_validation);
        }

        Ok(0)
    }

    /// Java `buildParallelPanel()`.
    pub fn build_parallel_panel(&self) {
        if self.parallel_panel.borrow().is_none() {
            let parallel_panel = ParallelPanel::get_instance(
                self.manager,
                self.axis_id,
                self.manager
                    .get_base_screen_state(Some(self.axis_id))
                    .unwrap()
                    .get_parallel_header_state(),
                self.self_ref.clone(),
                self.popup_chunk_warnings,
                self.runnable_parallel,
                self.interface_type,
            );
            *self.parallel_panel.borrow_mut() = Some(parallel_panel.clone());
            // Swing layout: parallelStatusPanel.add(Box.createRigidArea(FixedDim.x5_y0)).
            self.parallel_status_panel
                .add(&parallel_panel.get_container());
        }
    }

    /// Java private final `showParallelPanel(boolean, boolean)`.  Create and
    /// show, or hide parallel panel if necessary.
    fn show_parallel_panel_boolean_boolean(&self, show: bool, force: bool) {
        if self.processing_method_locked.get() && !force {
            return;
        }
        if !show {
            // Parallel panel is not in use, hide it if necessary
            if self.parallel_panel.borrow().is_some() && self.parallel_showing.get() {
                self.parallel_showing.set(false);
                self.parallel_status_panel.set_visible(false);
                self.pack_ui_harness();
            }
        } else {
            self.build_parallel_panel();
            if !self.parallel_showing.get() {
                self.parallel_showing.set(true);
                self.parallel_status_panel.set_visible(true);
                self.pack_ui_harness();
            }
        }
    }

    /// Java private final `startParallelPanel()`.  Uncalled in the Java.
    #[allow(dead_code)]
    fn start_parallel_panel(&self) {
        self.parallel_showing.set(true);
        let parallel_panel = self.parallel_panel.borrow().clone();
        // Upstream: dereferences a possibly null parallelPanel; the method is
        // never called.
        if let Some(parallel_panel) = parallel_panel {
            parallel_panel.get_load_display().start_load();
        }
        self.parallel_status_panel.set_visible(true);
        self.pack_ui_harness();
    }

    /// Java private final `stopParallelPanel()`.  Uncalled in the Java.
    #[allow(dead_code)]
    fn stop_parallel_panel(&self) {
        self.parallel_showing.set(false);
        let parallel_panel = self.parallel_panel.borrow().clone();
        if let Some(parallel_panel) = parallel_panel {
            parallel_panel.get_load_display().stop_load();
        }
        self.parallel_status_panel.set_visible(false);
        self.pack_ui_harness();
    }

    /// `UIHarness.INSTANCE.pack(axisID, manager)`, as written at each site.
    fn pack_ui_harness(&self) {
        ui_harness::INSTANCE.with(|harness| {
            harness.pack_axis_id_base_manager(Some(self.axis_id), Some(self.manager))
        });
    }

    /// Java final `getParallelPanel()`.
    pub fn get_parallel_panel(&self) -> Option<Rc<ParallelPanel>> {
        self.parallel_panel.borrow().clone()
    }

    /// Java final `getParallelStatusPanel()`.
    pub fn get_parallel_status_panel(&self) -> Rc<JComponent> {
        self.parallel_status_panel.clone()
    }

    /// Java final `done()`.
    pub fn done(&self) {
        let parallel_panel = self.parallel_panel.borrow().clone();
        if let Some(parallel_panel) = parallel_panel {
            parallel_panel.get_header_state(
                self.manager
                    .get_base_screen_state(Some(self.axis_id))
                    .unwrap()
                    .get_parallel_header_state(),
            );
        }
    }

    /// Java final `show()`.  Make panel visible.
    pub fn show(&self) {
        self.panel_root.set_visible(true);
    }

    /// Java final `saveDisplayState()`.
    pub fn save_display_state(&self) {
        let width = self.get_width();
        self.last_width.borrow_mut().set_int(width);
    }

    /// Java final `getWidth()`.  Get panel width.
    pub fn get_width(&self) -> i32 {
        let mut last_width = self.last_width.borrow_mut();
        if !last_width.is_null() {
            let width = last_width.get_int();
            last_width.reset();
            width
        } else {
            // Swing layout: `Rectangle size = new Rectangle();
            // panelRoot.computeVisibleRect(size); return size.width;`.  Geometry
            // is not modelled; the only reader of a width is hide(), which
            // nothing in the Java calls.
            0
        }
    }

    /// Java final `getContainer()`.
    pub fn get_container(&self) -> Rc<JComponent> {
        self.panel_root.clone()
    }

    /// Java final `replaceDialogPanel(Container)`.
    pub fn replace_dialog_panel(&self, new_dialog: &Rc<JComponent>) {
        self.panel_dialog.remove_all();
        self.panel_dialog.add(new_dialog);
        // Swing layout: panelDialog.revalidate(); panelDialog.repaint().
    }

    /// Java final `eraseDialogPanel()`.  Remove all process information from
    /// the dialog panel.
    pub fn erase_dialog_panel(&self) {
        // Get the current panel size and a new blank panel of the same size
        self.panel_dialog.remove_all();
        // Swing layout: panelDialog.revalidate(); panelDialog.repaint().
    }

    /// Java final `setPauseEnabled(boolean)`.
    pub fn set_pause_enabled(&self, enable_pause: bool) {
        let parallel_panel = self.parallel_panel.borrow().clone();
        if let Some(parallel_panel) = parallel_panel {
            parallel_panel.set_pause_enabled(enable_pause);
        }
    }

    /// The `AxisProcessPanel` body of Java `createProcessControlPanel()`,
    /// reached from an override the way `super.createProcessControlPanel()` is.
    pub fn create_process_control_panel_super(&self) {
        // Swing layout: panelProcessSelect.setLayout(new BoxLayout(panelProcessSelect,
        // BoxLayout.Y_AXIS)).

        if self.axis_id == AxisID::First {
            let axis_label = JComponent::new_label("Axis A:");
            // Swing layout: axisLabel.setAlignmentX(Container.CENTER_ALIGNMENT).
            self.panel_process_select.add(&axis_label);
        }
        if self.axis_id == AxisID::Second {
            let axis_label = JComponent::new_label("Axis B:");
            // Swing layout: axisLabel.setAlignmentX(Container.CENTER_ALIGNMENT).
            self.panel_process_select.add(&axis_label);
        }
    }

    /// Java `createProcessControlPanel()`, dispatched to the subclass.
    pub fn create_process_control_panel(&self) {
        match self.this() {
            Some(this) => this.create_process_control_panel(),
            None => self.create_process_control_panel_super(),
        }
    }

    /// Java `showBothAxis()`, dispatched to the subclass.
    pub fn show_both_axis(&self) {
        if let Some(this) = self.this() {
            this.show_both_axis();
        }
    }
}

impl ContextMenu for AxisProcessPanel {
    /// Java final `popUpContextMenu(MouseEvent)`.  Right mouse button context
    /// menu.
    fn pop_up_context_menu(&self, mouse_event: &MouseEvent) {
        let _context_popup = ContextPopup::new_component_mouse_event_string_base_manager_axis_id(
            &self.panel_root,
            mouse_event,
            Some(""),
            self.manager,
            self.axis_id,
        );
    }
}

/// Placeholder type for the empty `Weak<dyn AxisProcessPanelVirtual>` held
/// before the subclass installs itself.
struct NoSubclass;

impl AxisProcessPanelVirtual for NoSubclass {
    fn axis_process_panel(&self) -> &AxisProcessPanel {
        unreachable!("an empty Weak never upgrades")
    }
}
