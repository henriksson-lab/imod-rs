//! `IMOD/Etomo/src/etomo/ui/swing/ParallelPanel.java`.
//!
//! Java `public final class ParallelPanel implements Expandable, Storable,
//! QueueTableListener`: the parallel processing panel under an axis - the
//! processor tables (CPU, GPU, queue), the nice spinner, pause / resume, and
//! the parameters it hands to processchunks.
//!
//! Object model (`ui.md`): an EDT object created as `Rc<Self>` by
//! [`ParallelPanel::get_instance`]; every method takes `&self`.  The processor
//! tables hold this panel weakly and call back into it while they are being
//! built (`ProcessorTable.createTable` loads the stored table state, which
//! reads [`ParallelPanel::get_version`], and selects the only row, which calls
//! [`ParallelPanel::set_cpus_selected`]), so the panel `Rc` exists before the
//! tables are made: the table fields the Java constructor assigns are
//! `OnceCell`s, set in constructor order.  A Java `ProcessorTable` reference is
//! an `Rc<dyn ProcessorTableVirtual>`; Java `==` between tables is
//! `Rc::ptr_eq`.

use std::cell::{Cell, OnceCell, RefCell};
use std::collections::BTreeMap;
use std::rc::{Rc, Weak};
use std::sync::Arc;

use super::axis_process_panel::AxisProcessPanel;
use super::button_component::ButtonComponent;
use super::check_box::CheckBox;
use super::cpu_table::CpuTable;
use super::etomo_panel::EtomoPanel;
use super::expand_button::ExpandButton;
use super::expandable::Expandable;
use super::global_expand_button::GlobalExpandButton;
use super::gpu_table::GpuTable;
use super::labeled_text_field::LabeledTextField;
use super::load_display::LoadDisplay;
use super::panel_header::PanelHeader;
use super::parallel_progress_display::ParallelProgressDisplay;
use super::processor_table::ProcessorTableVirtual;
use super::queue_table::QueueTable;
use super::single_line_button::SingleLineButton;
use super::spaced_panel::SpacedPanel;
use super::spinner::Spinner;
use super::ui_harness;
use super::ui_parameters::UIParameters;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::alt_tomo_setup_param::AltTomoSetupParam;
use crate::imod::etomo::comscript::batchruntomo_param::BatchruntomoParam;
use crate::imod::etomo::comscript::ctf3d_setup_param::Ctf3dSetupParam;
use crate::imod::etomo::comscript::processchunks_param::{self, ProcesschunksParam};
use crate::imod::etomo::comscript::sirtsetup_param::SirtsetupParam;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::jdk::{ActionEvent, ActionListener};
use crate::imod::etomo::process::reconnect_process::ReconnectProcess;
use crate::imod::etomo::processing_method_mediator::ProcessingMethodMediator;
use crate::imod::etomo::storage::cpu_adoc;
use crate::imod::etomo::storage::network::Network;
use crate::imod::etomo::storage::storable::{Storable, StorableValue};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::java_lang_integer_parse_int;
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::r#type::etomo_version::EtomoVersion;
use crate::imod::etomo::r#type::interface_type::InterfaceType;
use crate::imod::etomo::r#type::panel_header_state::PanelHeaderState;
use crate::imod::etomo::r#type::process_result_display::ProcessResultDisplayHandle;
use crate::imod::etomo::r#type::processing_method::ProcessingMethod;
use crate::imod::etomo::ui::field::Field;
use crate::imod::etomo::ui::field_type::FieldType;
use crate::imod::etomo::ui::field_validation_failed_exception::FieldValidationFailedException;
use crate::imod::etomo::ui::queue_table_event::QueueTableEvent;
use crate::imod::etomo::ui::queue_table_listener::QueueTableListener;

/// Java `public static final String rcsid`.
pub const RCSID: &str = "$Id$";

/// Java private static final `STORE_PREPEND`.
const STORE_PREPEND: &str = "ProcessorTable";

/// Java private static final `TITLE`.
const TITLE: &str = "Parallel Processing";
/// Java private static final `NON_RUNNABLE_TITLE`.
const NON_RUNNABLE_TITLE: &str = "Resources";
/// Java package-private static final `RESUME_LABEL`.
pub const RESUME_LABEL: &str = "Resume";
/// Java package-private static final `PAUSE_LABEL`.
pub const PAUSE_LABEL: &str = "Pause";
/// Java package-private static final `FIELD_LABEL`.
pub const FIELD_LABEL: &str = "Parallel processing";
/// Java package-private static final `MAX_CPUS_STRING`.
pub const MAX_CPUS_STRING: &str = ":  Maximum number of cores recommended is ";
/// Java private final (instance) `CPUS_SELECTED_LABEL`.
const CPUS_SELECTED_LABEL: &str = "Cores: ";
/// Java private final (instance) `GPUS_SELECTED_LABEL`.
const GPUS_SELECTED_LABEL: &str = "GPUs: ";

// Java private static `maxCPUList = null` (HashedArray) and `validAutodoc =
// null` (EtomoBoolean2): declared, never assigned or read.

/// Java `public final class ParallelPanel`.
pub struct ParallelPanel {
    /// Rust-only: this panel's own handle (Java `this`).
    self_ref: Weak<ParallelPanel>,

    /// Java private final `tablePanel`.
    table_panel: Rc<EtomoPanel>,
    /// Java private final `computerTablePanel` (never used after construction).
    #[allow(dead_code)]
    computer_table_panel: Rc<EtomoPanel>,
    /// Java private final `queueTablePanel` (never used after construction).
    #[allow(dead_code)]
    queue_table_panel: Rc<EtomoPanel>,
    /// Java private final `ltfCPUsSelected`.
    ltf_cpus_selected: Rc<LabeledTextField>,
    /// Java private final `ltfSecondaryCPUsSelected`.
    ltf_secondary_cpus_selected: Rc<LabeledTextField>,
    /// Java private final `ltfChunksFinished`.
    ltf_chunks_finished: Rc<LabeledTextField>,
    /// Java private final `btnSaveDefaults`.
    btn_save_defaults: Rc<SingleLineButton>,
    /// Java private final `bodyPanel = SpacedPanel.getInstance()`.
    body_panel: Rc<SpacedPanel>,
    /// Java private final `rootPanel`.
    root_panel: Rc<EtomoPanel>,
    /// Java private final `btnRestartLoad`.
    btn_restart_load: Rc<SingleLineButton>,
    /// Java private final `cbQueues`.
    cb_queues: Rc<CheckBox>,
    /// Java private final `version = EtomoVersion.getDefaultInstance()`.
    version: RefCell<EtomoVersion>,
    /// Java private final `btnResume`.
    btn_resume: Rc<SingleLineButton>,
    /// Java private final `btnPause`.
    btn_pause: Rc<SingleLineButton>,

    /// Java private final `manager`.
    manager: &'static dyn BaseManager,
    /// Java private final `axisID`.
    axis_id: AxisID,
    /// Java private final `cpuTable`.
    cpu_table: OnceCell<Rc<dyn ProcessorTableVirtual>>,
    /// Java private final `queueTable` (null when there are no queues).
    queue_table: OnceCell<Option<Rc<dyn ProcessorTableVirtual>>>,
    /// Java private final `gpuTable` (null when there are no GPUs).
    gpu_table: OnceCell<Option<Rc<dyn ProcessorTableVirtual>>>,
    /// Java private final `sNice`.
    s_nice: OnceCell<Rc<Spinner>>,
    /// Java private final `header`.
    header: Rc<PanelHeader>,
    /// Java private final `parent` (stored, never read by this class).
    #[allow(dead_code)]
    parent: Weak<AxisProcessPanel>,
    /// Java private final `niceFloor`.
    nice_floor: Cell<i32>,
    /// Java private final `mediator`.
    mediator: Rc<ProcessingMethodMediator>,

    // private ParallelProcessPanelHeaderMonitor parallelProcessMonitor = null;
    /// Java private `visible` (assigned only by its initializer).
    #[allow(dead_code)]
    visible: Cell<bool>,
    /// Java private `open`.
    open: Cell<bool>,
    /// Java private `pauseEnabled`.
    pause_enabled: Cell<bool>,
    /// Java private `processchunksParam`.
    processchunks_param: RefCell<Option<Arc<ProcesschunksParam>>>,
    /// Java private `processResultDisplay`.
    process_result_display: RefCell<Option<ProcessResultDisplayHandle>>,
    /// Java private `currentTable`: the visible table - should never be null.
    current_table: RefCell<Option<Rc<dyn ProcessorTableVirtual>>>,
    /// Java private `processingMethodLocked`.
    processing_method_locked: Cell<bool>,
    /// Java private `processingRunning`.
    processing_running: Cell<bool>,
    /// Java private `secondaryTable`.
    secondary_table: RefCell<Option<Rc<dyn ProcessorTableVirtual>>>,
    /// Java private `runnable`.
    runnable: Cell<bool>,
    /// Java private `outsideResumeControl`.
    outside_resume_control: Cell<bool>,
    /// Java private `queueTableListenerArray`.
    queue_table_listener_array: RefCell<Option<Vec<Rc<dyn QueueTableListener>>>>,

    /// Java private final `popupChunkWarnings`.
    popup_chunk_warnings: bool,
}

impl ParallelPanel {
    /// Java static `getInstance(BaseManager, AxisID, PanelHeaderState,
    /// AxisProcessPanel, boolean, boolean, InterfaceType)`.
    pub fn get_instance(
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        state: &PanelHeaderState,
        parent: Weak<AxisProcessPanel>,
        popup_chunk_warnings: bool,
        runnable: bool,
        interface_type: InterfaceType,
    ) -> Rc<ParallelPanel> {
        let instance = ParallelPanel::new(
            manager,
            axis_id,
            state,
            parent,
            popup_chunk_warnings,
            runnable,
            interface_type,
        );
        instance.add_listeners();
        instance
    }

    /// Java private constructor `ParallelPanel(BaseManager, AxisID,
    /// PanelHeaderState, AxisProcessPanel, boolean, boolean, InterfaceType)`.
    /// Runnable: set to true if the table will be used for parallel processing,
    /// false if the table is used as a set of resources.
    fn new(
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        state: &PanelHeaderState,
        parent: Weak<AxisProcessPanel>,
        popup_chunk_warnings: bool,
        runnable: bool,
        interface_type: InterfaceType,
    ) -> Rc<ParallelPanel> {
        let mediator = manager
            .get_processing_method_mediator(Some(axis_id))
            // Built on the event dispatch thread, where the mediator exists.
            .expect("processing method mediator on the event dispatch thread");
        let instance = Rc::new_cyclic(|self_ref: &Weak<ParallelPanel>| {
            // header: `PanelHeader.getMoreLessInstance(runnable ? TITLE :
            // NON_RUNNABLE_TITLE, this, null)`.  (Created here because it holds
            // `this`; it only stores the weak reference.)
            let expandable: Weak<dyn Expandable> = self_ref.clone();
            let header = PanelHeader::get_more_less_instance(
                Some(if runnable { TITLE } else { NON_RUNNABLE_TITLE }),
                Some(expandable),
                None,
            );
            ParallelPanel {
                self_ref: self_ref.clone(),
                // Field initializers, in declaration order.
                table_panel: EtomoPanel::new(),
                computer_table_panel: EtomoPanel::new(),
                queue_table_panel: EtomoPanel::new(),
                ltf_cpus_selected: LabeledTextField::new_field_type_string(
                    FieldType::Integer,
                    Some(CPUS_SELECTED_LABEL),
                ),
                ltf_secondary_cpus_selected: LabeledTextField::new_field_type_string(
                    FieldType::Integer,
                    Some(GPUS_SELECTED_LABEL),
                ),
                ltf_chunks_finished: LabeledTextField::new_field_type_string(
                    FieldType::Integer,
                    Some("Chunks finished: "),
                ),
                btn_save_defaults: SingleLineButton::new_string(Some("Save As Defaults")),
                body_panel: SpacedPanel::get_instance_void(),
                root_panel: EtomoPanel::new(),
                btn_restart_load: SingleLineButton::new_string(Some("Restart Load")),
                cb_queues: CheckBox::new_string(Some("Use a cluster")),
                version: RefCell::new(EtomoVersion::get_default_instance()),
                btn_resume: SingleLineButton::new_string(Some(RESUME_LABEL)),
                btn_pause: SingleLineButton::new_string(Some(PAUSE_LABEL)),
                manager,
                axis_id,
                cpu_table: OnceCell::new(),
                queue_table: OnceCell::new(),
                gpu_table: OnceCell::new(),
                s_nice: OnceCell::new(),
                header,
                parent,
                nice_floor: Cell::new(0),
                mediator,
                visible: Cell::new(true),
                open: Cell::new(true),
                pause_enabled: Cell::new(false),
                processchunks_param: RefCell::new(None),
                process_result_display: RefCell::new(None),
                current_table: RefCell::new(None),
                processing_method_locked: Cell::new(false),
                processing_running: Cell::new(false),
                secondary_table: RefCell::new(None),
                runnable: Cell::new(true),
                outside_resume_control: Cell::new(false),
                queue_table_listener_array: RefCell::new(None),
                popup_chunk_warnings,
            }
        });
        // Constructor body.
        instance.runnable.set(runnable);
        // try {
        // ParameterStore parameterStore = EtomoDirector.INSTANCE.getParameterStore();
        // parameterStore.load(this);
        {
            let mut storable = instance.clone();
            let director = &*etomo_director::INSTANCE;
            if let Some(parameter_store) = director.get_parameter_store().as_ref() {
                parameter_store.load(&mut storable);
            }
        }
        // this.manager, this.axisID, this.parent, mediator,
        // this.popupChunkWarnings and header: assigned above.
        // initialize table
        // Less state is sometimes confusing.
        instance.header.set_save_more_less_state(false);
        let weak_this = Rc::downgrade(&instance);
        let more_less = instance
            .header
            .get_more_less_button()
            .map(|button| button as Rc<dyn crate::imod::etomo::ui::expander::Expander>);
        let cpu_table: Rc<dyn ProcessorTableVirtual> = CpuTable::new(
            manager,
            weak_this.clone(),
            axis_id,
            runnable,
            more_less.clone(),
            interface_type,
        );
        let _ = instance.cpu_table.set(cpu_table.clone());
        cpu_table.processor_table().create_table();
        *instance.current_table.borrow_mut() = Some(cpu_table.clone());
        instance.current_table().processor_table().set_visible(true);
        if Network::has_queues() {
            let queue_table: Rc<dyn ProcessorTableVirtual> = QueueTable::new(
                manager,
                weak_this.clone(),
                axis_id,
                runnable,
                more_less.clone(),
                interface_type,
            );
            let _ = instance.queue_table.set(Some(queue_table.clone()));
            queue_table.processor_table().create_table();
            queue_table.processor_table().set_visible(false);
        } else {
            let _ = instance.queue_table.set(None);
        }
        if Network::get_total_gpus(manager, axis_id, manager.get_property_user_dir().as_deref()) > 0
        {
            let gpu_table: Rc<dyn ProcessorTableVirtual> = GpuTable::new(
                manager,
                weak_this.clone(),
                axis_id,
                runnable,
                more_less.clone(),
                interface_type,
            );
            let _ = instance.gpu_table.set(Some(gpu_table.clone()));
            gpu_table.processor_table().create_table();
            gpu_table.processor_table().set_visible(false);
        } else {
            let _ = instance.gpu_table.set(None);
        }
        instance.ltf_secondary_cpus_selected.set_visible(false);
        // panels
        // Swing layout: rootPanel BoxLayout Y_AXIS with an etched border;
        // tablePanel BoxLayout X_AXIS; bodyPanel.setBoxLayout(Y_AXIS).
        let south_panel = SpacedPanel::get_instance_void();
        // Swing layout: southPanel.setBoxLayout(BoxLayout.X_AXIS).
        // southPanel;
        south_panel.add_labeled_text_field(&instance.ltf_cpus_selected);
        south_panel.add_labeled_text_field(&instance.ltf_secondary_cpus_selected);
        south_panel.add_multi_line_button(&instance.btn_restart_load);
        // sNice
        instance.nice_floor.set(cpu_adoc::INSTANCE.get_min_nice());
        let s_nice = Spinner::get_labeled_instance_string_int_int_int(
            Some("Nice: "),
            manager.get_parallel_processing_default_nice(),
            instance.nice_floor.get(),
            processchunks_param::NICE_CEILING,
        );
        let _ = instance.s_nice.set(s_nice.clone());
        south_panel.add_container(&s_nice.get_container());
        south_panel.add_multi_line_button(&instance.btn_pause);
        south_panel.add_multi_line_button(&instance.btn_resume);
        south_panel.add_multi_line_button(&instance.btn_save_defaults);

        // tablePanel
        instance.build_table_panel();
        // bodyPanel
        instance.body_panel.add_rigid_area_void();
        instance
            .body_panel
            .add_j_panel(&instance.table_panel.get_component());
        instance.body_panel.add_spaced_panel(&south_panel);
        if Network::has_queues()
            && let Some(queue_table) = instance.queue_table()
            && queue_table.processor_table().is_valid()
        {
            let cluster_panel = crate::imod::etomo::jdk::JComponent::new_panel();
            // Swing layout: clusterPanel BoxLayout X_AXIS, CENTER_ALIGNMENT, trailing
            // horizontal glue.
            cluster_panel.add(&instance.cb_queues.get_component());
            instance.body_panel.add_j_panel(&cluster_panel);
            instance.send_queue_table_event(&instance.get_queue_table_displayed_event());
        }
        // rootPanel
        instance.root_panel.add(&instance.header);
        instance
            .root_panel
            .get_component()
            .add(&instance.body_panel.get_container());
        let four_digit_width = UIParameters::get_instance_void().get_four_digit_width() as f64;
        instance
            .ltf_chunks_finished
            .set_text_preferred_width(four_digit_width);
        instance.ltf_chunks_finished.set_editable(false);
        instance
            .ltf_cpus_selected
            .set_text_preferred_width(four_digit_width);
        instance.ltf_cpus_selected.set_editable(false);
        instance
            .ltf_secondary_cpus_selected
            .set_text_preferred_width(four_digit_width);
        instance.ltf_secondary_cpus_selected.set_editable(false);
        // Java `if (btnPause != null)`: btnPause is final and always set.
        instance.btn_pause.set_enabled(false);
        instance.header.set_state(Some(state));

        instance.set_tool_tip_text();
        instance.root_panel.get_component().set_visible(runnable);
        instance.mediator.register_parallel_panel(instance.clone());
        instance
    }

    /// The Java `currentTable` field read.  Java: "the visible table - should
    /// never be null"; it is set in the constructor before any use.
    fn current_table(&self) -> Rc<dyn ProcessorTableVirtual> {
        self.current_table
            .borrow()
            .clone()
            .expect("ParallelPanel: currentTable is set by the constructor")
    }

    /// The Java `cpuTable` field read (set by the constructor).
    fn cpu_table(&self) -> Rc<dyn ProcessorTableVirtual> {
        self.cpu_table
            .get()
            .cloned()
            .expect("ParallelPanel: cpuTable is set by the constructor")
    }

    /// The Java `queueTable` field read (null is `None`).
    fn queue_table(&self) -> Option<Rc<dyn ProcessorTableVirtual>> {
        self.queue_table.get().cloned().flatten()
    }

    /// The Java `gpuTable` field read (null is `None`).
    fn gpu_table(&self) -> Option<Rc<dyn ProcessorTableVirtual>> {
        self.gpu_table.get().cloned().flatten()
    }

    /// The Java `sNice` field read (set by the constructor).
    fn s_nice(&self) -> Rc<Spinner> {
        self.s_nice
            .get()
            .cloned()
            .expect("ParallelPanel: sNice is set by the constructor")
    }

    /// Java private `addListeners()`.
    fn add_listeners(self: &Rc<Self>) {
        // Java `new ParallelPanelActionListener(this)`.
        let adaptee = Rc::downgrade(self);
        let action_listener: ActionListener = Rc::new(move |event: &ActionEvent| {
            let Some(adaptee) = adaptee.upgrade() else {
                return;
            };
            // Java `if (event != null)`: an event is always passed.
            adaptee.action(event.get_action_command());
        });
        self.btn_pause.add_action_listener(action_listener.clone());
        self.btn_resume.add_action_listener(action_listener.clone());
        self.btn_save_defaults
            .add_action_listener(action_listener.clone());
        self.btn_restart_load
            .add_action_listener(action_listener.clone());
        self.cb_queues.add_action_listener(Some(action_listener));
    }

    /// Java `addQueueListener(ButtonComponent[])`.
    pub fn add_queue_listener(&self, gpu_component_array: &[Rc<dyn ButtonComponent>]) {
        if let Some(queue_table) = self.queue_table() {
            for gpu_component in gpu_component_array {
                // `addActionListener(queueTable)`: the queue table is the
                // ActionListener.
                let queue_table = Rc::downgrade(&queue_table);
                gpu_component.add_action_listener(Rc::new(move |event: &ActionEvent| {
                    if let Some(queue_table) = queue_table.upgrade() {
                        queue_table.action_performed_virtual(event);
                    }
                }));
            }
        }
    }

    /// Java `addQueueTableListener(QueueTableListener)`.
    pub fn add_queue_table_listener(&self, listener: Option<Rc<dyn QueueTableListener>>) {
        let Some(listener) = listener else {
            return;
        };
        self.queue_table_listener_array
            .borrow_mut()
            .get_or_insert_with(Vec::new)
            .push(listener.clone());
        // Pass the listener on to the queue table.
        if let Some(queue_table) = self.queue_table() {
            queue_table.add_queue_table_listener(listener.clone());
        }
        // Since listeners can be added at any time, send the status to each new
        // listener.
        listener.queue_table_event_action(&self.get_queue_table_displayed_event());
    }

    /// Java `removeQueueTableListener(QueueTableListener)`.
    pub fn remove_queue_table_listener(&self, listener: Option<&Rc<dyn QueueTableListener>>) {
        let Some(listener) = listener else {
            return;
        };
        if let Some(array) = self.queue_table_listener_array.borrow_mut().as_mut() {
            // ArrayList.remove(Object): the first equal element.
            if let Some(index) = array.iter().position(|item| Rc::ptr_eq(item, listener)) {
                array.remove(index);
            }
        }
        // Also remove listener from the queue table.
        if let Some(queue_table) = self.queue_table() {
            queue_table.remove_queue_table_listener(listener);
        }
    }

    /// Java private `buildTablePanel()`.
    fn build_table_panel(&self) {
        let table_panel = self.table_panel.get_component();
        table_panel.remove_all();
        // Swing layout: leading horizontal glue.
        table_panel.add(&self.cpu_table().processor_table().get_container());
        if let Some(queue_table) = self.queue_table() {
            table_panel.add(&queue_table.processor_table().get_container());
        }
        if let Some(gpu_table) = self.gpu_table() {
            table_panel.add(&gpu_table.processor_table().get_container());
        }
        // Swing layout: trailing horizontal glue.
    }

    /// Java `isRunnable()`.
    pub fn is_runnable(&self) -> bool {
        self.runnable.get()
    }

    /// Java `getParallelProgressDisplay()`.
    pub fn get_parallel_progress_display(&self) -> Rc<dyn ParallelProgressDisplay> {
        self.current_table()
    }

    /// Java `resetResults()`.
    pub fn reset_results(&self) {
        self.current_table().processor_table().reset_results();
        let secondary_table = self.secondary_table.borrow().clone();
        if let Some(secondary_table) = secondary_table {
            secondary_table.processor_table().reset_results();
        }
    }

    /// Java `getLoadDisplay()`.
    pub fn get_load_display(&self) -> Rc<dyn LoadDisplay> {
        self.current_table()
    }

    /// Java `getSecondaryLoadDisplay()`.
    pub fn get_secondary_load_display(&self) -> Option<Rc<dyn LoadDisplay>> {
        self.secondary_table
            .borrow()
            .clone()
            .map(|table| table as Rc<dyn LoadDisplay>)
    }

    /// Java `setPauseEnabled(boolean)`.
    pub fn set_pause_enabled(&self, pause_enabled: bool) {
        self.pause_enabled.set(pause_enabled);
        // Java `if (btnPause != null)`: always set.
        self.btn_pause.set_enabled(pause_enabled);
    }

    /// Java `setCPUsSelected(int)`.
    pub fn set_cpus_selected(&self, cpus_selected: i32) {
        self.ltf_cpus_selected.set_text_int(cpus_selected);
    }

    /// Java `setSecondaryCPUsSelected(int)`.
    pub fn set_secondary_cpus_selected(&self, cpus_selected: i32) {
        self.ltf_secondary_cpus_selected.set_text_int(cpus_selected);
    }

    /// Java `getCPUsSelected(boolean) throws FieldValidationFailedException`.
    pub fn get_cpus_selected(
        &self,
        do_validation: bool,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        self.ltf_cpus_selected.get_text_boolean(do_validation)
    }

    /// Java `getCPUsSelectedInt(boolean) throws FieldValidationFailedException`.
    pub fn get_cpus_selected_int(
        &self,
        do_validation: bool,
    ) -> Result<i32, FieldValidationFailedException> {
        // try { return Integer.parseInt(...); } catch (NumberFormatException) {
        // return 0; }
        // Integer.parseInt(null) also throws NumberFormatException.
        Ok(self
            .get_cpus_selected(do_validation)?
            .as_deref()
            .and_then(|text| java_lang_integer_parse_int(text).ok())
            .unwrap_or(0))
    }

    /// Java `getCPUsSelectedLabel()`.
    pub fn get_cpus_selected_label(&self) -> String {
        self.ltf_cpus_selected.get_label()
    }

    /// Java `getNoCpusSelectedErrorMessage()`.
    pub fn get_no_cpus_selected_error_message(&self) -> Option<String> {
        self.current_table().get_no_cpus_selected_error_message()
    }

    /// Java `getSecondaryNoCpusSelectedErrorMessage()`.
    pub fn get_secondary_no_cpus_selected_error_message(&self) -> Option<String> {
        let secondary_table = self.secondary_table.borrow().clone();
        if let Some(secondary_table) = secondary_table {
            return secondary_table.get_no_cpus_selected_error_message();
        }
        None
    }

    /// Java `getContainer()`.
    pub fn get_container(&self) -> Rc<crate::imod::etomo::jdk::JComponent> {
        self.root_panel.get_component()
    }

    /// Java `getParallelPauseButton()`.
    pub fn get_parallel_pause_button(&self) -> Option<Rc<SingleLineButton>> {
        Some(self.btn_pause.clone())
    }

    /// Java `getParallelResumeButton()`.
    pub fn get_parallel_resume_button(&self) -> Option<Rc<SingleLineButton>> {
        // The enabling/disabling the resume button will need finer control and
        // should be left to the caller of the is function.
        self.outside_resume_control.set(true);
        Some(self.btn_resume.clone())
    }

    /// Java `setProcessInfo(ProcesschunksParam, ProcessResultDisplay)`.
    pub fn set_process_info(
        &self,
        processchunks_param: Option<Arc<ProcesschunksParam>>,
        process_result_display: Option<ProcessResultDisplayHandle>,
    ) {
        *self.processchunks_param.borrow_mut() = processchunks_param.clone();
        *self.process_result_display.borrow_mut() = process_result_display;
        if let Some(processchunks_param) = processchunks_param {
            self.header.set_text_string_string(
                Some(TITLE),
                processchunks_param.get_root_name().as_deref(),
            );
        }
    }

    /// Java private `action(String)`.
    fn action(&self, command: Option<&str>) {
        let Some(command) = command else {
            return;
        };
        if Some(command) == self.btn_resume.get_action_command().as_deref() {
            let processchunks_param = self.processchunks_param.borrow().clone();
            let process_result_display = self.process_result_display.borrow().clone();
            self.manager.resume(
                Some(self.axis_id),
                processchunks_param,
                process_result_display,
                None,
                None,
                self.popup_chunk_warnings,
                self.mediator
                    .get_run_method_for_parallel_panel(self.get_processing_method()),
                false,
                None,
            );
        } else if Some(command) == self.btn_pause.get_action_command().as_deref() {
            self.manager.pause(Some(self.axis_id));
        } else if Some(command) == self.btn_save_defaults.get_action_command().as_deref() {
            let this = self.self_ref.upgrade();
            self.manager.save_preferences(
                Some(self.axis_id),
                this.as_ref().map(|this| this as &dyn Storable),
            );
            let cpu_table = self.cpu_table();
            self.manager
                .save_preferences(Some(self.axis_id), Some(&cpu_table as &dyn Storable));
            let gpu_table = self.gpu_table();
            self.manager.save_preferences(
                Some(self.axis_id),
                gpu_table.as_ref().map(|table| table as &dyn Storable),
            );
            let queue_table = self.queue_table();
            self.manager.save_preferences(
                Some(self.axis_id),
                queue_table.as_ref().map(|table| table as &dyn Storable),
            );
        } else if Some(command) == self.btn_restart_load.get_action_command().as_deref() {
            self.current_table()
                .processor_table()
                .restart_load_monitor();
            let secondary_table = self.secondary_table.borrow().clone();
            if let Some(secondary_table) = secondary_table {
                secondary_table.processor_table().restart_load_monitor();
            }
        } else if Some(command) == self.cb_queues.get_action_command().as_deref() {
            if self.is_queues() {
                self.set_processing_method(Some(ProcessingMethod::Queue));
                if let Some(this) = self.self_ref.upgrade() {
                    self.mediator.set_method_parallel_panel_processing_method(
                        &this,
                        Some(ProcessingMethod::Queue),
                    );
                }
            } else {
                // dialogs can turn on GPU check box
                if let Some(this) = self.self_ref.upgrade() {
                    self.mediator
                        .set_method_parallel_panel_processing_method(&this, None);
                }
                // Need to know whether to use the CPU or GPU table
                self.set_processing_method(self.mediator.get_run_method_for_parallel_panel(None));
                self.current_table()
                    .processor_table()
                    .restart_load_monitor();
                let secondary_table = self.secondary_table.borrow().clone();
                if let Some(secondary_table) = secondary_table {
                    secondary_table.processor_table().restart_load_monitor();
                }
            }
            self.send_queue_table_event(&self.get_queue_table_displayed_event());
        }
    }

    /// Java `queueTableEventAction(QueueTableEvent)` (implements
    /// `QueueTableListener`).
    pub fn queue_table_event_action(&self, event: &QueueTableEvent) {
        if *event == QueueTableEvent::AllowDisplay
            || *event == QueueTableEvent::Display
            || *event == QueueTableEvent::PreventDisplay
        {
            if *event == QueueTableEvent::PreventDisplay {
                self.cb_queues.set_enabled(false);
            } else {
                self.cb_queues.set_enabled(true);
                if *event == QueueTableEvent::Display {
                    self.cb_queues.set_selected_boolean(true);
                }
            }
            let command = self.cb_queues.get_action_command();
            self.action(command.as_deref());
        }
        if let Some(queue_table) = self.queue_table() {
            queue_table.queue_table_event_action(event);
        }
    }

    /// Java private `getQueueTableDisplayedEvent()`.
    fn get_queue_table_displayed_event(&self) -> QueueTableEvent {
        if self.is_queues() {
            QueueTableEvent::Displayed
        } else {
            QueueTableEvent::Hidden
        }
    }

    /// Java private `sendQueueTableEvent(QueueTableEvent)`.
    fn send_queue_table_event(&self, event: &QueueTableEvent) {
        // Iterate over a copy: a listener may add or remove listeners.
        let listeners = self.queue_table_listener_array.borrow().clone();
        let Some(listeners) = listeners else {
            return;
        };
        for listener in listeners {
            listener.queue_table_event_action(event);
        }
    }

    /// Java `load(Properties)` (implements `Storable`).
    pub fn load_properties(&self, props: &BTreeMap<String, String>) {
        self.load_properties_string(props, "");
    }

    /// Java `load(Properties, String)` (implements `Storable`).
    pub fn load_properties_string(&self, props: &BTreeMap<String, String>, prepend: &str) {
        // Java `String group;` - declared, unused.
        // Java `prepend == ""` is an identity test against the interned literal;
        // every caller's empty prepend is that literal.
        let prepend = if prepend.is_empty() {
            STORE_PREPEND.to_string()
        } else {
            format!("{prepend}.{STORE_PREPEND}")
        };
        self.version.borrow_mut().load_with_prepend(props, &prepend);
    }

    /// Java `getVersion()`.  Returns the (const) version; the translation hands
    /// out a copy so no borrow of the field is held.
    pub fn get_version(&self) -> EtomoVersion {
        self.version.borrow().clone()
    }

    /// Java `store(Properties)` (implements `Storable`).
    pub fn store_properties(&self, props: &mut BTreeMap<String, String>) {
        self.store_properties_string(props, "");
    }

    /// Java `store(Properties, String)` (implements `Storable`).
    pub fn store_properties_string(&self, props: &mut BTreeMap<String, String>, prepend: &str) {
        // Java `String group;` - declared, unused.
        let prepend = if prepend.is_empty() {
            STORE_PREPEND.to_string()
        } else {
            format!("{prepend}.{STORE_PREPEND}")
        };
        self.version.borrow_mut().set(Some("1.1"));
        self.version.borrow().store_with_prepend(props, &prepend);
    }

    /// Java `lockProcessingMethod(boolean)`.
    pub fn lock_processing_method(&self, lock: bool) {
        self.processing_method_locked.set(lock);
        self.update_processing_method_lock();
    }

    /// Java private `updateProcessingMethodLock()`.  lock/unlock processing
    /// method by making the queue checkbox editable or ineditable.  The user
    /// won't be able to change it, but it can be distinguished from a disabled
    /// checkbox.
    fn update_processing_method_lock(&self) {
        self.cb_queues
            .set_editable(!self.processing_method_locked.get() && !self.processing_running.get());
    }

    /// Java `setRunnable(boolean)`.  Toggled the primary table between runnable
    /// and not runnable.
    pub fn set_runnable(&self, runnable: bool) {
        self.header
            .set_text_string(Some(if runnable { TITLE } else { NON_RUNNABLE_TITLE }));
        self.current_table()
            .processor_table()
            .set_runnable(runnable);
        self.btn_pause.set_visible(runnable);
        self.btn_resume.set_visible(runnable);
    }

    /// Java `setLimited(boolean)`.  Toggled the tables between limited and not
    /// limited.
    pub fn set_limited(&self, limited: bool) {
        self.cpu_table().processor_table().set_limited(limited);
        if let Some(queue_table) = self.queue_table() {
            queue_table.processor_table().set_limited(limited);
        }
        if let Some(gpu_table) = self.gpu_table() {
            gpu_table.processor_table().set_limited(limited);
        }
    }

    /// Java `setQueue(boolean, ReconnectProcess, ProcessingMethod)`.  The queues
    /// checkbox needs to be set during the reconnect to a process being run on a
    /// queue.
    pub fn set_queue(
        &self,
        registering: bool,
        origin: Option<&ReconnectProcess>,
        method: Option<ProcessingMethod>,
    ) {
        if registering
            && origin.is_some()
            && self.cb_queues.is_enabled()
            && method == Some(ProcessingMethod::Queue)
        {
            self.cb_queues.set_selected_boolean(true);
        }
    }

    /// Java `setProcessingMethod(ProcessingMethod)`.  Set currentTable based on
    /// method.
    pub fn set_processing_method(&self, mut method: Option<ProcessingMethod>) {
        // Handle local method
        if method.is_none_or(|method| method.is_local()) {
            self.current_table().processor_table().stop_load();
            return;
        }
        // Handle parallel method
        // The queue checkbox overrides the dialog's parallel processing settings
        // for the current table, and this class's default.
        if self.is_queues() {
            method = Some(ProcessingMethod::Queue);
            if let Some(queue_table) = self.queue_table() {
                queue_table.enable_gpu_queue_rows();
            }
        }
        // The table load is stopped when the panel is hidden - needs to be
        // started when the panel is shown.
        let new_table = self.get_table(method);
        if let Some(new_table) = new_table {
            let current_table = self.current_table();
            if Rc::ptr_eq(&current_table, &new_table) {
                if current_table.processor_table().is_stopped() {
                    current_table.processor_table().start_load();
                    current_table.processor_table().msg_cpus_selected_changed();
                }
            } else {
                // Stop and hide the current table
                if !current_table.processor_table().is_stopped() {
                    current_table.processor_table().stop_load();
                }
                current_table.processor_table().set_visible(false);
                // Show and start a different table
                *self.current_table.borrow_mut() = Some(new_table.clone());
                let current_table = new_table;
                // If current table has taken the secondary table, reset the
                // secondary table pointer
                let taken = self
                    .secondary_table
                    .borrow()
                    .as_ref()
                    .is_some_and(|secondary_table| Rc::ptr_eq(&current_table, secondary_table));
                if taken {
                    *self.secondary_table.borrow_mut() = None;
                    self.ltf_secondary_cpus_selected.set_visible(false);
                }
                current_table.processor_table().set_secondary(false);
                // CurrentTable may or may not be runnable. Must be set by separate
                // function if it can be changed. Otherwise is defaults to true if
                // construction is runnable.
                current_table.processor_table().set_visible(true);
                current_table.processor_table().start_load();
                current_table.processor_table().msg_cpus_selected_changed();
                self.s_nice().set_enabled(current_table.is_niceable());
                if self
                    .gpu_table()
                    .is_some_and(|gpu_table| Rc::ptr_eq(&current_table, &gpu_table))
                {
                    self.ltf_cpus_selected.set_label(Some(GPUS_SELECTED_LABEL));
                } else {
                    self.ltf_cpus_selected.set_label(Some(CPUS_SELECTED_LABEL));
                }
                ui_harness::with(|harness| {
                    harness.pack_axis_id_base_manager(Some(self.axis_id), Some(self.manager))
                });
            }
        }
    }

    /// Java `setVisible(boolean)`.
    pub fn set_visible(&self, visible: bool) {
        self.root_panel.get_component().set_visible(visible);
        let current_table = self.current_table();
        if current_table.processor_table().is_stopped() {
            if visible {
                current_table.processor_table().start_load();
            }
        } else if !visible {
            current_table.processor_table().stop_load();
        }
    }

    /// Java `setSecondaryProcessingMethod(ProcessingMethod)`.  Update secondary
    /// table if necessary.
    pub fn set_secondary_processing_method(&self, method: Option<ProcessingMethod>) {
        if self.secondary_table.borrow().is_none() && method.is_none() {
            return;
        }
        let new_table = self.get_secondary_table(method);
        let secondary_table = self.secondary_table.borrow().clone();
        let same = match (&secondary_table, &new_table) {
            (None, None) => true,
            (Some(secondary_table), Some(new_table)) => Rc::ptr_eq(secondary_table, new_table),
            _ => false,
        };
        if same {
            return;
        }
        if let Some(secondary_table) = &secondary_table {
            // Done with the current secondary table
            secondary_table.processor_table().set_secondary(false);
            secondary_table.processor_table().set_visible(false);
            self.ltf_secondary_cpus_selected.set_visible(false);
        }
        *self.secondary_table.borrow_mut() = new_table.clone();
        if let Some(secondary_table) = new_table {
            secondary_table.processor_table().set_secondary(true);
            secondary_table.processor_table().set_runnable(false);
            secondary_table.processor_table().set_visible(true);
            self.ltf_secondary_cpus_selected.set_visible(true);
            secondary_table
                .processor_table()
                .msg_cpus_selected_changed();
            if self
                .gpu_table()
                .is_some_and(|gpu_table| Rc::ptr_eq(&secondary_table, &gpu_table))
            {
                self.ltf_secondary_cpus_selected
                    .set_label(Some(GPUS_SELECTED_LABEL));
            } else {
                self.ltf_secondary_cpus_selected
                    .set_label(Some(CPUS_SELECTED_LABEL));
            }
        }
        ui_harness::with(|harness| {
            harness.pack_axis_id_base_manager(Some(self.axis_id), Some(self.manager))
        });
    }

    /// Java `stopTable()`.
    pub fn stop_table(&self) {
        self.current_table().processor_table().stop_load();
    }

    /// Java `endTable()`.
    pub fn end_table(&self) {
        self.current_table().processor_table().end_load();
    }

    /// Java `getTable(ProcessingMethod)`.
    pub fn get_table(
        &self,
        method: Option<ProcessingMethod>,
    ) -> Option<Rc<dyn ProcessorTableVirtual>> {
        if method == Some(ProcessingMethod::PpCpu) {
            return Some(self.cpu_table());
        }
        if method == Some(ProcessingMethod::PpGpu) {
            return self.gpu_table();
        }
        if method == Some(ProcessingMethod::Queue) {
            return self.queue_table();
        }
        Some(self.current_table())
    }

    /// Java private `getSecondaryTable(ProcessingMethod)`.
    fn get_secondary_table(
        &self,
        method: Option<ProcessingMethod>,
    ) -> Option<Rc<dyn ProcessorTableVirtual>> {
        if method == Some(ProcessingMethod::PpCpu) {
            return Some(self.cpu_table());
        }
        if method == Some(ProcessingMethod::PpGpu) {
            return self.gpu_table();
        }
        if method == Some(ProcessingMethod::Queue) {
            return self.queue_table();
        }
        None
    }

    /// Java `getProcessingMethod()`.
    pub fn get_processing_method(&self) -> Option<ProcessingMethod> {
        let current_table = self.current_table();
        if Rc::ptr_eq(&current_table, &self.cpu_table()) {
            return Some(ProcessingMethod::PpCpu);
        }
        if self
            .gpu_table()
            .is_some_and(|gpu_table| Rc::ptr_eq(&current_table, &gpu_table))
        {
            return Some(ProcessingMethod::PpGpu);
        }
        if self
            .queue_table()
            .is_some_and(|queue_table| Rc::ptr_eq(&current_table, &queue_table))
        {
            return Some(ProcessingMethod::Queue);
        }
        None
    }

    /// Java `getSecondaryProcessingMethod()`.
    pub fn get_secondary_processing_method(&self) -> Option<ProcessingMethod> {
        let secondary_table = self.secondary_table.borrow().clone();
        // Java compares a possibly-null secondaryTable with each table; null ==
        // null (no GPU or queue table) matches as well.
        let matches =
            |table: Option<Rc<dyn ProcessorTableVirtual>>| match (&secondary_table, &table) {
                (None, None) => true,
                (Some(secondary_table), Some(table)) => Rc::ptr_eq(secondary_table, table),
                _ => false,
            };
        if matches(Some(self.cpu_table())) {
            return Some(ProcessingMethod::PpCpu);
        }
        if matches(self.gpu_table()) {
            return Some(ProcessingMethod::PpGpu);
        }
        if matches(self.queue_table()) {
            return Some(ProcessingMethod::Queue);
        }
        None
    }

    /// Java `msgEndingProcess()`.
    pub fn msg_ending_process(&self) {
        self.processing_running.set(false);
        self.update_processing_method_lock();
    }

    /// Java `msgKillingProcess()`.
    pub fn msg_killing_process(&self) {
        self.btn_pause.set_enabled(false);
        if !self.outside_resume_control.get() {
            self.btn_resume.set_enabled(false);
        }
    }

    /// Java `msgPausingProcess()`.
    pub fn msg_pausing_process(&self) {
        if !self.outside_resume_control.get() {
            self.btn_resume.set_enabled(true);
        }
    }

    /// Java `msgProcessDone()`.
    pub fn msg_process_done(&self) {
        if !self.outside_resume_control.get() {
            self.btn_resume.set_enabled(true);
        }
    }

    /// Java `msgProcessStarted()`.
    pub fn msg_process_started(&self) {
        if !self.outside_resume_control.get() {
            self.btn_resume.set_enabled(false);
        }
    }

    /// Java `getResumeParameters(ProcesschunksParam, boolean)`.  If getting
    /// parameters, must not allow the user to change the current table until
    /// that parameters have been used.
    pub fn get_resume_parameters(&self, param: &ProcesschunksParam, do_validation: bool) -> bool {
        // try {
        self.processing_running.set(true);
        self.update_processing_method_lock();
        param.set_resume(true);
        param.set_nice(Some(self.s_nice().get_value()));
        let mut cpus_selected = EtomoNumber::new();
        match self.ltf_cpus_selected.get_text_boolean(do_validation) {
            Ok(text) => {
                cpus_selected.set_string(text.as_deref());
            }
            // catch (FieldValidationFailedException e) { return false; }
            Err(_) => return false,
        }
        if cpus_selected.equals_int(0) && do_validation {
            ui_harness::with(|harness| {
                harness.open_message_dialog_base_manager_string_string_axis_id(
                    Some(self.manager),
                    self.get_no_cpus_selected_error_message()
                        .as_deref()
                        .unwrap_or("null"),
                    "Unable to resume",
                    Some(self.axis_id),
                )
            });
            return false;
        }
        param.set_cpu_number_const_etomo_number(Some(&cpus_selected));
        param.reset_machine_name();
        self.current_table()
            .get_parameters_processchunks_param(param);
        // The secondary table is not for running, but if the GPU machine list is
        // in use, it has to be loaded, and can only come from the GPU table.
        let secondary_table = self.secondary_table.borrow().clone();
        if let Some(secondary_table) = secondary_table
            && self.manager.is_add_gpu_machine_to_process_chunks()
            && secondary_table.is_gpu_table()
        {
            secondary_table.get_parameters_processchunks_param(param);
        }
        true
    }

    /// Java `getNumberOfProcessors(boolean)`; null is `None`.
    pub fn get_number_of_processors(&self, do_validation: bool) -> Option<String> {
        // try {
        let mut num_machines = EtomoNumber::new();
        num_machines.set_null_is_valid(false);
        num_machines.set_valid_floor(1);
        match self.ltf_cpus_selected.get_text_boolean(do_validation) {
            Ok(text) => {
                num_machines.set_string(text.as_deref());
            }
            // catch (FieldValidationFailedException e) { return null; }
            Err(_) => return None,
        }
        if !num_machines.is_valid() && do_validation {
            if num_machines.equals_int(0) {
                ui_harness::with(|harness| {
                    harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self.manager),
                        self.get_no_cpus_selected_error_message()
                            .as_deref()
                            .unwrap_or("null"),
                        "Unable to run",
                        Some(self.axis_id),
                    )
                });
                return None;
            } else {
                ui_harness::with(|harness| {
                    harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self.manager),
                        &format!(
                            "{} {}",
                            self.get_cpus_selected_label(),
                            num_machines.get_invalid_reason()
                        ),
                        "Unable to run",
                        Some(self.axis_id),
                    )
                });
                return None;
            }
        }
        Some(num_machines.to_string())
    }

    /// Java `getParameters(SirtsetupParam, boolean)`.
    pub fn get_parameters_sirtsetup_param_boolean(
        &self,
        param: &mut SirtsetupParam,
        do_validation: bool,
    ) -> bool {
        // try {
        let mut num_machines = EtomoNumber::new();
        num_machines.set_null_is_valid(false);
        num_machines.set_valid_floor(1);
        match self.ltf_cpus_selected.get_text_boolean(do_validation) {
            Ok(text) => {
                num_machines.set_string(text.as_deref());
            }
            // catch (FieldValidationFailedException e) { return false; }
            Err(_) => return false,
        }
        if !num_machines.is_valid() && do_validation {
            if num_machines.equals_int(0) {
                ui_harness::with(|harness| {
                    harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self.manager),
                        self.get_no_cpus_selected_error_message()
                            .as_deref()
                            .unwrap_or("null"),
                        "Unable to run splittilt",
                        Some(self.axis_id),
                    )
                });
                return false;
            } else {
                ui_harness::with(|harness| {
                    harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self.manager),
                        &format!(
                            "{} {}",
                            self.get_cpus_selected_label(),
                            num_machines.get_invalid_reason()
                        ),
                        "Unable to run sirtsetup",
                        Some(self.axis_id),
                    )
                });
                return false;
            }
        }
        param.set_number_of_processors(Some(&num_machines.to_string()));
        true
    }

    /// Java static `resetParameters(Ctf3dSetupParam)`.
    pub fn reset_parameters(param: &mut Ctf3dSetupParam) {
        param.reset_number_of_processors();
    }

    /// Java `getParameters(Ctf3dSetupParam, boolean, boolean)`.
    pub fn get_parameters_ctf3d_setup_param_boolean_boolean(
        &self,
        param: &mut Ctf3dSetupParam,
        run_slabs_in_parallel: bool,
        do_validation: bool,
    ) -> bool {
        // try {
        if !run_slabs_in_parallel {
            match self.ltf_cpus_selected.get_text_boolean(do_validation) {
                Ok(text) => param.set_number_of_processors(text.as_deref()),
                // catch (FieldValidationFailedException e) { return false; }
                Err(_) => return false,
            }
        } else {
            param.reset_number_of_processors();
        }
        true
    }

    /// Java `getParameters(AltTomoSetupParam, boolean)`.
    pub fn get_parameters_alt_tomo_setup_param_boolean(
        &self,
        param: &mut AltTomoSetupParam,
        do_validation: bool,
    ) -> bool {
        // try {
        let text = match self.ltf_cpus_selected.get_text_boolean(do_validation) {
            Ok(text) => text,
            // catch (FieldValidationFailedException e) { return false; }
            Err(_) => return false,
        };
        // Upstream bug fixed in translation (ParallelPanel.java:866): the Java
        // `Integer.valueOf(ltfCPUsSelected.getText(doValidation))` throws an
        // uncaught NumberFormatException on an empty or non-integer field (only
        // FieldValidationFailedException is caught), which unwinds out of the
        // caller.  Here an unparsable field fails the call like a validation
        // failure: return false.
        let Some(number_of_processors) = text
            .as_deref()
            .and_then(|text| java_lang_integer_parse_int(text).ok())
        else {
            return false;
        };
        param.set_number_of_processors(number_of_processors);
        true
    }

    /// Java `getParameters(ProcesschunksParam, boolean)`.  If getting
    /// parameters, must not allow the user to change the current table until
    /// that parameters have been used.
    pub fn get_parameters_processchunks_param_boolean(
        &self,
        param: &ProcesschunksParam,
        do_validation: bool,
    ) -> bool {
        // try {
        self.processing_running.set(true);
        // cbQueues.setEditable(!processingMethodLocked && !processingRunning);
        param.set_nice(Some(self.s_nice().get_value()));
        match self.ltf_cpus_selected.get_text_boolean(do_validation) {
            Ok(text) => param.set_cpu_number_string(text.as_deref()),
            // catch (FieldValidationFailedException e) { return false; }
            Err(_) => return false,
        }
        let current_table = self.current_table();
        current_table.get_parameters_processchunks_param(param);
        // The secondary table is not for running, but if the GPU machine list is
        // in use, it has to be loaded, and can only come from the GPU table.
        let secondary_table = self.secondary_table.borrow().clone();
        if let Some(secondary_table) = &secondary_table
            && self.manager.is_add_gpu_machine_to_process_chunks()
            && secondary_table.is_gpu_table()
        {
            secondary_table.get_parameters_processchunks_param(param);
        }
        let error = param.validate();
        match error {
            None => return true,
            Some(error) if do_validation => {
                ui_harness::with(|harness| {
                    harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self.manager),
                        &format!(
                            "{}  {}{}",
                            error,
                            current_table.processor_table().get_help_message(),
                            match &secondary_table {
                                Some(secondary_table) => format!(
                                    "  {}",
                                    secondary_table.processor_table().get_help_message()
                                ),
                                None => String::new(),
                            }
                        ),
                        "Table Error",
                        Some(self.axis_id),
                    )
                });
                return false;
            }
            Some(_) => {}
        }
        true
    }

    /// Java `getParameters(BatchruntomoParam, boolean, boolean)`.  Get
    /// parameters for batchruntomo comfile.  (`validateOnly` is unused in the
    /// Java.)
    pub fn get_parameters_batchruntomo_param_boolean_boolean(
        &self,
        param: &mut BatchruntomoParam,
        do_validation: bool,
        _validate_only: bool,
    ) -> bool {
        param.set_nice_value(Some(self.s_nice().get_value()));
        if !self
            .current_table()
            .get_parameters_processing_method_batchruntomo_param_boolean(
                self.mediator
                    .get_run_method_for_parallel_panel(self.get_processing_method()),
                param,
                do_validation,
            )
        {
            return false;
        }
        let secondary_table = self.secondary_table.borrow().clone();
        if let Some(secondary_table) = secondary_table {
            secondary_table.get_parameters_processing_method_batchruntomo_param_boolean(
                self.mediator.get_secondary_run_method_for_parallel_panel(
                    self.get_secondary_processing_method(),
                ),
                param,
                do_validation,
            );
        }
        true
    }

    /// Java `setParameters(BatchruntomoParam)`.
    pub fn set_parameters(&self, param: &BatchruntomoParam) {
        self.s_nice()
            .set_value_string(Some(&param.get_nice_value()));
        self.current_table().processor_table().set_parameters(param);
        let secondary_table = self.secondary_table.borrow().clone();
        if let Some(secondary_table) = secondary_table {
            secondary_table.processor_table().set_parameters(param);
        }
    }

    /// Java `msgSelectionChanged()`.  Handle changing which CPUs/GPUs/queues are
    /// checked, and which secondary queue is checked in a contracted display.
    pub fn msg_selection_changed(&self) {
        if self.header.is_less() {
            self.set_more_less(true);
            self.set_more_less(false);
        }
    }

    /// Java private `setMoreLess(boolean)`.
    fn set_more_less(&self, more: bool) {
        self.btn_save_defaults.set_visible(more);
        if let Some(queue_table) = self.queue_table() {
            queue_table.processor_table().set_expanded(more);
        }
        self.cpu_table().processor_table().set_expanded(more);
        if let Some(gpu_table) = self.gpu_table() {
            gpu_table.processor_table().set_expanded(more);
        }
        self.build_table_panel();
    }

    /// Java `getHeaderState(PanelHeaderState)`.
    pub fn get_header_state(&self, header_state: &PanelHeaderState) {
        self.header.get_state(Some(header_state));
    }

    /// Java `isCbUseGpu()`.
    pub fn is_cb_use_gpu(&self) -> bool {
        self.mediator.is_use_gpu()
    }

    /// Java private `isQueues()`.
    fn is_queues(&self) -> bool {
        // Java `if (cbQueues == null) return false;`: cbQueues is final and always
        // set.
        self.queue_table().is_some() && self.cb_queues.is_enabled() && self.cb_queues.is_selected()
    }

    /// Java `getUseQueueCheckbox()`.
    pub fn get_use_queue_checkbox(&self) -> Rc<dyn ButtonComponent> {
        self.cb_queues.clone()
    }

    /// Java private `setToolTipText()`.  Initialize the tooltip text for the axis
    /// panel objects.
    fn set_tool_tip_text(&self) {
        self.ltf_cpus_selected
            .set_tool_tip_text(Some("Must be at least 1."));
        self.s_nice().set_tool_tip_text(Some(
            "Lower the value to run the processes at a higher priority.  Raise the value to run at a lower priority.",
        ));
        self.btn_pause.set_tool_tip_text(Some(
            "Finishes the processes that are currently running and then stops.",
        ));
        self.btn_resume.set_tool_tip_text(Some(
            "Starts the process but does not redo the chunks that are already completed.",
        ));
        self.btn_save_defaults
            .set_tool_tip_text(Some("Saves the computers and number of CPUs selected."));
    }
}

impl Expandable for ParallelPanel {
    /// Java `expand(GlobalExpandButton)`: empty.
    fn expand_global_expand_button(&self, _button: &Rc<GlobalExpandButton>) {}

    /// Java `expand(ExpandButton)`.  set the visible boolean based on whether
    /// the panel is visible; the body panel setVisible function was called by
    /// the header panel.
    fn expand_expand_button(&self, button: &Rc<ExpandButton>) {
        if self.header.equals_open_close(button) {
            self.open.set(button.is_expanded());
            self.body_panel.set_visible(self.open.get());
            ui_harness::with(|harness| {
                harness.pack_axis_id_base_manager(Some(self.axis_id), Some(self.manager))
            });
        } else if self.header.equals_more_less(button) {
            self.set_more_less(button.is_expanded());
            ui_harness::with(|harness| {
                harness.pack_axis_id_base_manager(Some(self.axis_id), Some(self.manager))
            });
        }
    }
}

impl QueueTableListener for ParallelPanel {
    /// Java `queueTableEventAction(QueueTableEvent)`.
    fn queue_table_event_action(&self, event: &QueueTableEvent) {
        ParallelPanel::queue_table_event_action(self, event);
    }
}

/// Java `implements Storable`: the panel is held as an `Rc`, so the trait is
/// implemented on the handle (the version is interior-mutable).
impl Storable for Rc<ParallelPanel> {
    fn store(&self, properties: &mut BTreeMap<String, String>) {
        self.store_properties(properties);
    }

    fn store_with_prepend(&self, properties: &mut BTreeMap<String, String>, prepend: &str) {
        self.store_properties_string(properties, prepend);
    }

    fn load(&self, properties: &BTreeMap<String, String>) {
        self.load_properties(properties);
    }

    fn load_with_prepend(&self, properties: &BTreeMap<String, String>, prepend: &str) {
        self.load_properties_string(properties, prepend);
    }
}
