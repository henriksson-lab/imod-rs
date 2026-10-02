//! `IMOD/Etomo/src/etomo/ui/swing/CpuGpuPanel.java`.
//!
//! Java `final class CpuGpuPanel implements ActionListener, ProcessInterface`:
//! the "Parallel processing" / "Use the GPU" check boxes shared by the tilt,
//! tilt_3dfind and CTF-correction panels.  An EDT object created as `Rc<Self>`;
//! every method takes `&self`.  Java passes `this` to the
//! `ProcessingMethodMediator`; that `this` is `self_ref` (upgraded), so the
//! methods that do it need no `Rc` receiver.  The Java `ActionListener`
//! implementation is [`CpuGpuPanel::action_performed`], registered as a
//! closure holding a weak reference.

use std::cell::{Cell, RefCell};
use std::rc::{Rc, Weak};

use super::button_component::ButtonComponent;
use super::check_box::CheckBox;
use super::parallel_panel;
use super::process_interface::ProcessInterface;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::const_ctf_phase_flip_param::ConstCtfPhaseFlipParam;
use crate::imod::etomo::comscript::const_tilt_param::ConstTiltParam;
use crate::imod::etomo::comscript::ctf_phase_flip_param::CtfPhaseFlipParam;
use crate::imod::etomo::comscript::fortran_input_syntax_exception::FortranInputSyntaxException;
use crate::imod::etomo::comscript::tilt_param::TiltParam;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::jdk::{ActionEvent, ActionListener, JComponent};
// TODO(unit): needs etomo/ProcessingMethodMediator.java - the mediator type and
// its register/deregister/setMethod/getRunMethodForProcessInterface/addGpuListener.
use crate::imod::etomo::processing_method_mediator::ProcessingMethodMediator;
use crate::imod::etomo::storage::cpu_adoc;
use crate::imod::etomo::storage::network::Network;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::ConstEtomoNumber;
use crate::imod::etomo::r#type::const_meta_data::ConstMetaData;
use crate::imod::etomo::r#type::etomo_boolean2::EtomoBoolean2;
use crate::imod::etomo::r#type::meta_data::MetaData;
use crate::imod::etomo::r#type::panel_id::PanelId;
use crate::imod::etomo::r#type::processing_method::ProcessingMethod;
use crate::imod::etomo::ui::queue_table_event::QueueTableEvent;
use crate::imod::etomo::ui::queue_table_listener::QueueTableListener;

/// Java `javax.swing.BoxLayout.X_AXIS`.
const BOX_LAYOUT_X_AXIS: i32 = 0;
/// Java `javax.swing.BoxLayout.Y_AXIS`.
const BOX_LAYOUT_Y_AXIS: i32 = 1;

/// Java `final class CpuGpuPanel implements ActionListener, ProcessInterface`.
pub struct CpuGpuPanel {
    /// Rust-only: Java `this` (for the mediator and the listeners).
    self_ref: Weak<CpuGpuPanel>,
    /// Java private final `pnlRoot = new JPanel()`.
    pnl_root: Rc<JComponent>,
    /// Java private final `cbParallelProcess`: call mediator.msgChangedMethod
    /// when cbParallelProcess's value is changed.
    cb_parallel_process: Rc<CheckBox>,
    /// Java private final `manager`.
    manager: &'static dyn BaseManager,
    /// Java private final `axisID`.
    axis_id: AxisID,

    /// Java private final `cbUseGpu`: enable/disable cbUseGpu by changing
    /// gpuEnabled and calling updateDisplay.  Call mediator.msgChangedMethod
    /// when cbUseGpu's value is changed.
    cb_use_gpu: Rc<CheckBox>,
    /// Java private final `mediator`.
    mediator: Rc<ProcessingMethodMediator>,
    /// Java private final `lMaxGpus` (a `JLabel`; null when not displayed).
    l_max_gpus: Option<Rc<JComponent>>,

    /// Java private `alwaysParallel`.
    always_parallel: Cell<bool>,
    /// Java private `processingMethodLocked`.
    processing_method_locked: Cell<bool>,
    /// Java private `gpusAvailable`.
    gpus_available: Cell<bool>,
    /// Java private `nonLocalHostGpusAvailable`.
    non_local_host_gpus_available: Cell<bool>,
    /// Java private `localGpuAvailable`.
    local_gpu_available: Cell<bool>,
    /// Java private `gpuEnabled` (never read in the Java).
    gpu_enabled: Cell<bool>,
    /// Java private `useMaxGpus`.
    use_max_gpus: Cell<bool>,
    /// Java private `gpusForQueueAvailable`.
    gpus_for_queue_available: Cell<bool>,
    /// Java private `maxGpus` (never assigned or read in the Java).
    max_gpus: Cell<i32>,
    /// Java private `useQueueCheckBox`.
    use_queue_check_box: RefCell<Option<Rc<dyn ButtonComponent>>>,
    /// Java private `nonQueueGpuCheckboxStatus`.
    non_queue_gpu_checkbox_status: RefCell<EtomoBoolean2>,
    /// Java private `altStackProcessInterface`.
    alt_stack_process_interface: RefCell<Option<Rc<dyn ProcessInterface>>>,
    /// Rust-only: the registered `ActionListener` (Java `this`).
    action_listener: ActionListener,
}

impl CpuGpuPanel {
    /// Java private constructor `CpuGpuPanel(BaseManager, AxisID, int)`.
    /// `maxGpu`: max recommended GPU - use -1 for no recommendation.
    fn new(manager: &'static dyn BaseManager, axis_id: AxisID, max_gpu: i32) -> Rc<CpuGpuPanel> {
        let instance = Rc::new_cyclic(|self_ref: &Weak<CpuGpuPanel>| {
            // Field initializers.
            let pnl_root = JComponent::new_panel();
            let cb_parallel_process = CheckBox::new_string(Some(parallel_panel::FIELD_LABEL));
            // Java `this` as the ActionListener.
            let adaptee = self_ref.clone();
            let action_listener: ActionListener = Rc::new(move |event: &ActionEvent| {
                if let Some(adaptee) = adaptee.upgrade() {
                    adaptee.action_performed(event);
                }
            });
            // Constructor body.
            let cb_use_gpu = CheckBox::new_string(Some("Use the GPU"));
            let l_max_gpus = if max_gpu == -1 {
                None
            } else if Network::get_total_gpus(
                manager,
                axis_id,
                manager.get_property_user_dir().as_deref(),
            ) > 1
            {
                Some(JComponent::new_label(&format!(
                    ": Maximum number of GPUs recommended is {max_gpu}"
                )))
            } else {
                None
            };
            let mediator = manager
                .get_processing_method_mediator(Some(axis_id))
                // Built on the event dispatch thread, where the mediator exists.
                .expect("processing method mediator on the event dispatch thread");
            CpuGpuPanel {
                self_ref: self_ref.clone(),
                pnl_root,
                cb_parallel_process,
                manager,
                axis_id,
                cb_use_gpu,
                mediator,
                l_max_gpus,
                always_parallel: Cell::new(false),
                processing_method_locked: Cell::new(false),
                gpus_available: Cell::new(false),
                non_local_host_gpus_available: Cell::new(false),
                local_gpu_available: Cell::new(true),
                gpu_enabled: Cell::new(true),
                use_max_gpus: Cell::new(true),
                gpus_for_queue_available: Cell::new(false),
                max_gpus: Cell::new(0),
                use_queue_check_box: RefCell::new(None),
                non_queue_gpu_checkbox_status: RefCell::new(EtomoBoolean2::new()),
                alt_stack_process_interface: RefCell::new(None),
                action_listener,
            }
        });
        // mediator.register(this) (the last statement of the Java constructor).
        instance
            .mediator
            .register_process_interface(instance.clone() as Rc<dyn ProcessInterface>);
        instance
    }

    /// Java `this` as the mediator's `ProcessInterface`.
    fn this(&self) -> Option<Rc<dyn ProcessInterface>> {
        self.self_ref
            .upgrade()
            .map(|this| this as Rc<dyn ProcessInterface>)
    }

    /// Java `mediator.setMethod(this, method)`.
    fn mediator_set_method_this(&self, method: ProcessingMethod) {
        if let Some(this) = self.this() {
            self.mediator
                .set_method_process_interface_processing_method(&this, method);
        }
    }

    /// Java static `getInstance(BaseManager, AxisID, PanelId, int, boolean, int)`.
    /// `maxGpu`: optional, -1 to ignore - displays as recommended max GPUs;
    /// `useMaxCpu`: when true displays recommended max CPUs; `boxLayoutAxis`:
    /// optional, -1 to ignore - BoxLayout.Y_AXIS (default) or .X_AXIS.
    pub fn get_instance(
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        panel_id: PanelId,
        max_gpu: i32,
        use_max_cpu: bool,
        box_layout_axis: i32,
    ) -> Rc<CpuGpuPanel> {
        let instance = CpuGpuPanel::new(manager, axis_id, max_gpu);
        instance.create_panel(box_layout_axis);
        instance.set_tooltips();
        instance.init(panel_id, use_max_cpu);
        instance.add_listeners();
        instance
    }

    /// Java private `createPanel(int)`.  `boxLayoutAxis` - default is Y_AXIS.
    fn create_panel(&self, mut box_layout_axis: i32) {
        // fixup params
        if box_layout_axis != BOX_LAYOUT_X_AXIS {
            box_layout_axis = BOX_LAYOUT_Y_AXIS; // default
        }
        let _ = box_layout_axis;
        // local panels
        let pnl_parallel_process = JComponent::new_panel();
        let pnl_use_gpu = JComponent::new_panel();
        // Root
        // Swing layout: pnlRoot.setLayout(new BoxLayout(pnlRoot, boxLayoutAxis)).
        self.pnl_root.add(&pnl_parallel_process);
        self.pnl_root.add(&pnl_use_gpu);
        // ParallelProcess
        // Swing layout: pnlParallelProcess BoxLayout X_AXIS; horizontal glue after
        // the check box.
        pnl_parallel_process.add(&self.cb_parallel_process.get_component());
        // UseGpu
        // Swing layout: pnlUseGpu BoxLayout X_AXIS; horizontal glue at the end.
        pnl_use_gpu.add(&self.cb_use_gpu.get_component());
        if let Some(l_max_gpus) = &self.l_max_gpus {
            pnl_use_gpu.add(l_max_gpus);
        }
    }

    /// Java private `init(PanelId, boolean)`.
    fn init(&self, panel_id: PanelId, use_max_cpu: bool) {
        if (panel_id == PanelId::Tilt || panel_id == PanelId::Tilt3dFind) && use_max_cpu {
            let max_cpus: ConstEtomoNumber = cpu_adoc::INSTANCE.get_max_tilt();
            if !max_cpus.is_null() {
                self.cb_parallel_process.set_text(Some(&format!(
                    "{}{}{}",
                    parallel_panel::FIELD_LABEL,
                    parallel_panel::MAX_CPUS_STRING,
                    max_cpus
                )));
            }
        }
        // Parallel processing is optional in tomogram reconstruction, so only use it
        // if the user set it up.
        // Use GPU
        self.gpus_available
            .set(Network::is_non_local_only_gpu_processing_enabled());
        self.gpus_for_queue_available
            .set(Network::is_any_queue_gpu());
        self.non_local_host_gpus_available
            .set(Network::is_non_local_host_gpu_processing_enabled(
                self.manager,
                self.axis_id,
                self.manager.get_property_user_dir().as_deref(),
            ));
        self.local_gpu_available
            .set(Network::is_local_host_gpu_processing_enabled(
                self.manager,
                self.axis_id,
                self.manager.get_property_user_dir().as_deref(),
            ));
        self.update_display();
        self.mediator_set_method_this(self.get_processing_method_void());
    }

    /// Java private `isGpusAvailable()`.
    fn is_gpus_available(&self) -> bool {
        // Try using this.queueCheckbox(if exists) instead of mediator.isUseCluster()
        // If doesnt exist, assume Use Cluster is off
        let use_queue_check_box = self.use_queue_check_box.borrow().clone();
        if let Some(use_queue_check_box) = use_queue_check_box {
            if use_queue_check_box.is_selected() {
                return self.gpus_for_queue_available.get();
            }
        }
        self.gpus_available.get()
    }

    /// Java private `isLocalGpuAvailable()`.
    fn is_local_gpu_available(&self) -> bool {
        let use_queue_check_box = self.use_queue_check_box.borrow().clone();
        if let Some(use_queue_check_box) = use_queue_check_box {
            if use_queue_check_box.is_selected() {
                return false;
            }
        }
        self.local_gpu_available.get()
    }

    /// Java `reregisterProcessingMethodMediator()`.
    pub fn reregister_processing_method_mediator(&self) {
        if let Some(this) = self.this() {
            self.mediator.register_process_interface(this);
        }
        self.mediator_set_method_this(self.get_processing_method_void());
    }

    /// Java `isParallelProcess()`.
    pub fn is_parallel_process(&self) -> bool {
        self.cb_parallel_process.is_enabled() && self.cb_parallel_process.is_selected()
    }

    /// Java final `setParameters(ConstMetaData, ConstEtomoNumber)`.
    pub fn set_parameters_const_meta_data_const_etomo_number(
        &self,
        meta_data: &dyn ConstMetaData,
        parallel_process: Option<&ConstEtomoNumber>,
    ) {
        // Parallel processing is optional in tomogram reconstruction, so only use it
        // if the user set it up.
        self.cb_use_gpu
            .set_selected_boolean(meta_data.is_default_gpu_processing());
        // updateUseGpu();
        // Parallel processing
        // cbParallelProcess.setEnabled(parallelProcessingEnabled);
        // If only a local GPU is available and the Use GPU checkbox defaults to on, do not
        // select the parallel processing checkbox.
        if self.non_local_host_gpus_available.get() || !self.cb_use_gpu.is_selected() {
            match parallel_process {
                None => {
                    self.cb_parallel_process
                        .set_selected_boolean(meta_data.is_default_parallel());
                }
                Some(parallel_process) => {
                    self.cb_parallel_process
                        .set_selected_boolean(parallel_process.is());
                }
            }
        }
        self.update_display();
        self.set_method_alt_stack_or_this();
    }

    /// Java `if (altStackProcessInterface != null) mediator.setMethod(
    /// altStackProcessInterface, getProcessingMethod()); else
    /// mediator.setMethod(this, getProcessingMethod());` (repeated inline in
    /// three Java methods).
    fn set_method_alt_stack_or_this(&self) {
        let alt_stack_process_interface = self.alt_stack_process_interface.borrow().clone();
        if let Some(alt_stack_process_interface) = alt_stack_process_interface {
            self.mediator
                .set_method_process_interface_processing_method(
                    &alt_stack_process_interface,
                    self.get_processing_method_void(),
                );
        } else {
            self.mediator_set_method_this(self.get_processing_method_void());
        }
    }

    /// Java `getParameters(PanelId, MetaData) throws FortranInputSyntaxException`.
    pub fn get_parameters_panel_id_meta_data(
        &self,
        panel_id: PanelId,
        meta_data: &MetaData,
    ) -> Result<(), FortranInputSyntaxException> {
        meta_data.set_tilt_parallel(self.axis_id, panel_id, self.is_parallel_process());
        Ok(())
    }

    /// Java `setParameters(ConstTiltParam, boolean)`.  `initialize` - true when
    /// the dialog is first created for the dataset.
    pub fn set_parameters_const_tilt_param_boolean(
        &self,
        tilt_param: &dyn ConstTiltParam,
        initialize: bool,
    ) {
        if !initialize {
            // During initialization the value should coming from setup
            self.cb_use_gpu
                .set_selected_boolean(tilt_param.is_use_gpu());
        }
        self.update_display();
        self.set_method_alt_stack_or_this();
    }

    /// Java `getParameters(TiltParam)`.
    pub fn get_parameters_tilt_param(&self, tilt_param: &mut TiltParam) {
        tilt_param.set_use_gpu(self.cb_use_gpu.is_enabled() && self.cb_use_gpu.is_selected());
    }

    /// Java `getParameters(CtfPhaseFlipParam)`.
    pub fn get_parameters_ctf_phase_flip_param(&self, param: &mut CtfPhaseFlipParam) {
        param.set_use_gpu(self.cb_use_gpu.is_enabled() && self.cb_use_gpu.is_selected());
    }

    /// Java `setParameters(ConstCtfPhaseFlipParam, boolean)`.  `initialize` -
    /// true when the dialog is first created for the dataset.
    pub fn set_parameters_const_ctf_phase_flip_param_boolean(
        &self,
        param: &dyn ConstCtfPhaseFlipParam,
        initialize: bool,
    ) {
        if !initialize {
            // During initialization the value should coming from setup
            self.cb_use_gpu.set_selected_boolean(param.is_use_gpu());
        }
        self.update_display();
        self.mediator_set_method_this(self.get_processing_method_void());
    }

    /// Java `addListeners()`.
    pub fn add_listeners(&self) {
        self.cb_parallel_process
            .add_action_listener(Some(self.action_listener.clone()));
        self.cb_use_gpu
            .add_action_listener(Some(self.action_listener.clone()));
        let gpu_component: Vec<Rc<dyn ButtonComponent>> =
            vec![self.cb_use_gpu.clone() as Rc<dyn ButtonComponent>];
        self.mediator.add_gpu_listener(gpu_component);
    }

    /// Java `getComponent()`.
    pub fn get_component(&self) -> Rc<JComponent> {
        self.pnl_root.clone()
    }

    /// Java `actionPerformed(ActionEvent)` (the panel is its own
    /// `ActionListener`).
    pub fn action_performed(&self, event: &ActionEvent) {
        let action_command = event.get_action_command();
        // If there is a only a local GPU available, turn off parallel processing when Use
        // GPU is selected.
        let local_gpus_int =
            etomo_director::INSTANCE.with_user_configuration(|c| c.get_local_gpus_int());
        if local_gpus_int <= 1 {
            if self.cb_parallel_process.get_action_command().as_deref() == action_command {
                if self.cb_parallel_process.is_selected()
                    && !self.non_local_host_gpus_available.get()
                    && self.cb_use_gpu.is_selected()
                {
                    self.cb_use_gpu.set_selected_boolean(false);
                }
            } else if self.cb_use_gpu.get_action_command().as_deref() == action_command {
                if self.cb_use_gpu.is_selected()
                    && !self.non_local_host_gpus_available.get()
                    && self.cb_parallel_process.is_selected()
                {
                    self.cb_parallel_process.set_selected_boolean(false);
                }
            } else {
                // Upstream bug fixed in translation (CpuGpuPanel.java:311): Java
                // dereferences `useQueueCheckBox` here, which is null until
                // setUseQueueCheckBox is called, so an event from any other source
                // throws a NullPointerException.  A missing queue check box does
                // not match the event here.
                let use_queue_check_box = self.use_queue_check_box.borrow().clone();
                if let Some(use_queue_check_box) = use_queue_check_box {
                    if use_queue_check_box.get_action_command().as_deref() == action_command {
                        // save gpu state for previous state
                        // set gpu state for current state
                        if use_queue_check_box.is_selected() {
                            self.non_queue_gpu_checkbox_status
                                .borrow_mut()
                                .set_boolean(self.cb_use_gpu.is_selected());
                        } else {
                            let status = self.non_queue_gpu_checkbox_status.borrow().is();
                            self.cb_use_gpu.set_selected_boolean(status);
                        }
                    }
                }
            }
        }
        self.update_display();
        self.set_method_alt_stack_or_this();
    }

    /// Java `getRunMethodForProcessInterface()`.
    pub fn get_run_method_for_process_interface(&self) -> ProcessingMethod {
        self.mediator
            .get_run_method_for_process_interface(self.get_processing_method_void())
    }

    /// Java `msgProcessingMethodChanged(boolean, boolean)`.
    pub fn msg_processing_method_changed(&self, always_parallel: bool, use_max_gpus: bool) {
        self.always_parallel.set(always_parallel);
        self.use_max_gpus.set(use_max_gpus);
        self.update_display();
        self.mediator_set_method_this(self.get_processing_method_void());
    }

    /// Java final `done()`.
    pub fn done(&self) {
        if let Some(this) = self.this() {
            self.mediator.deregister_process_interface(&this);
        }
    }

    /// Java `updateDisplay()`.
    pub fn update_display(&self) {
        let always_parallel = self.always_parallel.get();
        let processing_method_locked = self.processing_method_locked.get();
        self.cb_parallel_process.set_visible(!always_parallel);
        self.cb_parallel_process
            .set_enabled(!processing_method_locked);
        // A local GPU installed on this computer means that GPU processing is available
        // with or without parallel processing. Non-local GPU(s) installed in the network mean
        // that GPU processing is available with parallel processing.
        self.cb_use_gpu.set_enabled(
            (((self.is_gpus_available() || self.is_local_gpu_available())
                && ((self.cb_parallel_process.is_enabled()
                    && self.cb_parallel_process.is_selected())
                    || always_parallel))
                || (self.is_local_gpu_available()
                    && !self.cb_parallel_process.is_selected()
                    && !always_parallel))
                && !processing_method_locked,
        );
        if let Some(l_max_gpus) = &self.l_max_gpus {
            l_max_gpus.set_enabled(self.cb_use_gpu.is_enabled());
            l_max_gpus.set_visible(self.use_max_gpus.get());
        }
    }

    /// Java `getProcessingMethod()` (the `ProcessInterface` method, callable
    /// without the trait in scope).
    pub fn get_processing_method_void(&self) -> ProcessingMethod {
        let parallel_process = self.always_parallel.get()
            || (self.cb_parallel_process.is_enabled() && self.cb_parallel_process.is_selected());
        if parallel_process {
            if self.cb_use_gpu.is_enabled() && self.cb_use_gpu.is_selected() {
                return ProcessingMethod::PpGpu;
            }
            return ProcessingMethod::PpCpu;
        }
        if self.local_gpu_available.get()
            && self.cb_use_gpu.is_enabled()
            && self.cb_use_gpu.is_selected()
        {
            return ProcessingMethod::LocalGpu;
        }
        ProcessingMethod::LocalCpu
    }

    /// Java `isProcessingMethodValid()`.
    pub fn is_processing_method_valid(&self) -> bool {
        // If parallel processing isn't available, the method is only valid if it does't have
        // to be parallel.
        !self.always_parallel.get()
    }

    /// Java private `setTooltips()`.
    fn set_tooltips(&self) {
        self.cb_parallel_process.set_tool_tip_text_string(Some(
            "Check to distribute the process across multiple computers.",
        ));
        self.cb_use_gpu.set_tool_tip_text_string(Some(
            "Check to run the process on one or more graphics cards.",
        ));
    }

    /// Java public `registerProcesingMethodMediator()` (sic).
    pub fn register_procesing_method_mediator(&self) {
        if let Some(this) = self.this() {
            self.mediator.register_process_interface(this);
        }
    }

    /// Java public `deregisterProcesingMethodMediator()` (sic).
    pub fn deregister_procesing_method_mediator(&self) {
        if let Some(this) = self.this() {
            self.mediator.deregister_process_interface(&this);
        }
    }

    /// Java public `setAltStackProcessInterface(ProcessInterface)`.
    pub fn set_alt_stack_process_interface(&self, origin: Option<Rc<dyn ProcessInterface>>) {
        *self.alt_stack_process_interface.borrow_mut() = origin;
    }
}

impl QueueTableListener for CpuGpuPanel {
    /// Java `queueTableEventAction(QueueTableEvent)`: empty.
    fn queue_table_event_action(&self, _event: &QueueTableEvent) {}
}

impl ProcessInterface for CpuGpuPanel {
    /// Java `updateGpu(boolean)`.
    fn update_gpu(&self, _disable_gpu: bool) {
        self.update_display();
    }

    /// Java `getProcessingMethod()`.
    fn get_processing_method(&self) -> ProcessingMethod {
        self.get_processing_method_void()
    }

    /// Java `getSecondaryProcessingMethod()`.
    fn get_secondary_processing_method(&self) -> Option<ProcessingMethod> {
        None
    }

    /// Java `lockProcessingMethod(boolean)`.
    fn lock_processing_method(&self, lock: bool) {
        self.processing_method_locked.set(lock);
        self.update_display();
    }

    /// Java `setMethod(ProcessingMethod)`.  (The Java null test on the final
    /// `mediator` field always passes.)
    fn set_method(&self, processing_method: ProcessingMethod) {
        self.mediator_set_method_this(processing_method);
    }

    /// Java `isUseGpu()`.
    fn is_use_gpu(&self) -> bool {
        self.cb_use_gpu.is_enabled() && self.cb_use_gpu.is_selected()
    }

    /// Java `setUseQueueCheckBox(ButtonComponent)`.
    fn set_use_queue_check_box(&self, use_queue_check_box: Option<Rc<dyn ButtonComponent>>) {
        if let Some(use_queue_check_box) = use_queue_check_box {
            if self.use_queue_check_box.borrow().is_none() {
                use_queue_check_box.add_action_listener(self.action_listener.clone());
                *self.use_queue_check_box.borrow_mut() = Some(use_queue_check_box);
            }
        }
    }

    /// Java `addQueueTableListener(QueueTableListener)`: empty.
    fn add_queue_table_listener(&self, _listener: Rc<dyn QueueTableListener>) {}

    /// Java `removeQueueTableListener(QueueTableListener)`: empty.
    fn remove_queue_table_listener(&self, _listener: &Rc<dyn QueueTableListener>) {}
}
