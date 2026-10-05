//! `IMOD/Etomo/src/etomo/ui/swing/SubtomoSetupCpuGpuPanel.java`.
//!
//! Java `public class SubtomoSetupCpuGpuPanel implements ActionListener`: the
//! "Processing resources to use" radio buttons of the Subtomograms panel.  An
//! EDT object created as `Rc<Self>` by
//! [`SubtomoSetupCpuGpuPanel::get_subtomo_setup_instance`]; every method takes
//! `&self`.  The Java `ActionListener` implementation is
//! [`SubtomoSetupCpuGpuPanel::action_performed`], registered as one closure
//! holding a weak reference.
//!
//! The owning `SubtomogramsPanel` and the `ProcessInterface` (the Post
//! Processing dialog) own this panel, so both are held weakly.  They are still
//! being constructed when this panel is built; the Java's
//! `processInterface.setMethod(...)` in `createSubtomoSetupPanel` is then a
//! no-op in the Java as well (`PostProcessingDialog.setMethod` does nothing
//! while its `mediator` is still null), which is what a weak reference that
//! cannot yet be upgraded gives.

use std::cell::{Cell, RefCell};
use std::rc::{Rc, Weak};

use super::button_component::ButtonComponent;
use super::etched_border::EtchedBorder;
use super::process_interface::ProcessInterface;
use super::radio_button::RadioButton;
use super::subtomograms_panel::SubtomogramsPanel;
use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::subtomo_setup_param;
use crate::imod::etomo::jdk::{ActionEvent, ActionListener, ButtonGroup, JComponent};
use crate::imod::etomo::processing_method_mediator::ProcessingMethodMediator;
use crate::imod::etomo::storage::autodoc::autodoc::Autodoc;
use crate::imod::etomo::storage::autodoc::autodoc_factory;
use crate::imod::etomo::storage::autodoc::read_only_autodoc::ReadOnlyAutodoc;
use crate::imod::etomo::storage::autodoc::read_only_section::ReadOnlySection;
use crate::imod::etomo::storage::autodoc::read_only_section_list::ReadOnlySectionList;
use crate::imod::etomo::storage::log_file::LogFileError;
use crate::imod::etomo::storage::network::Network;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::etomo_autodoc;
use crate::imod::etomo::r#type::processing_method::ProcessingMethod;

/// Java private enum `CpuGpuSelection`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CpuGpuSelection {
    /// Java `CPU_ONLY`.
    CpuOnly,
    /// Java `GPU_ONLY`.
    GpuOnly,
    /// Java `MIX_CPU_GPU`.
    MixCpuGpu,
}

/// Java `public class SubtomoSetupCpuGpuPanel implements ActionListener`.
pub struct SubtomoSetupCpuGpuPanel {
    /// Java private final `pnlRoot = new JPanel()`.
    pnl_root: Rc<JComponent>,
    /// Java private final `bgProcessingResourcesToUse`.
    bg_processing_resources_to_use: Rc<ButtonGroup>,
    /// Java private final `rbCpusOnly`.
    rb_cpus_only: Rc<RadioButton>,
    /// Java private final `rbGpuForReconAndCtfCorrect`.
    rb_gpu_for_recon_and_ctf_correct: Rc<RadioButton>,
    /// Java private final `rbCpuForReconAndGpuForCtfCorrect`.
    rb_cpu_for_recon_and_gpu_for_ctf_correct: Rc<RadioButton>,

    /// Java private final `manager`.
    manager: &'static ApplicationManager,
    /// Java private final `axisID`.
    axis_id: AxisID,
    /// Java private final `processInterface` (held weakly; see the module docs).
    process_interface: Weak<dyn ProcessInterface>,
    /// Java private final `mediator`.
    mediator: Rc<ProcessingMethodMediator>,
    /// Java private final `subtomogramsPanel` (held weakly; it owns this panel).
    subtomograms_panel: Weak<SubtomogramsPanel>,
    /// Java private final `localGpuAvailable` (never read in the Java).
    local_gpu_available: bool,

    /// Java private `useQueueCheckBox`.
    use_queue_check_box: RefCell<Option<Rc<dyn ButtonComponent>>>,
    /// Java private `gpusForQueueAvailable` (never read in the Java).
    gpus_for_queue_available: Cell<bool>,
    /// Java private `mixCpuGpuActionEvent`.
    mix_cpu_gpu_action_event: Cell<bool>,
    /// Java private `currCpuGpuSelection`.
    curr_cpu_gpu_selection: Cell<Option<CpuGpuSelection>>,

    /// Rust-only: Java `this` as the `ActionListener` it registers.
    action_listener: ActionListener,
}

impl SubtomoSetupCpuGpuPanel {
    /// Java private constructor `SubtomoSetupCpuGpuPanel(SubtomogramsPanel,
    /// ApplicationManager, AxisID, ProcessInterface)`.
    fn new(
        subtomograms_panel: Weak<SubtomogramsPanel>,
        manager: &'static ApplicationManager,
        axis_id: AxisID,
        process_interface: Weak<dyn ProcessInterface>,
    ) -> Rc<SubtomoSetupCpuGpuPanel> {
        Rc::new_cyclic(|self_ref: &Weak<SubtomoSetupCpuGpuPanel>| {
            // Field initializers.
            let pnl_root = JComponent::new_panel();
            let bg_processing_resources_to_use = ButtonGroup::new();
            let rb_cpus_only = RadioButton::new_string_button_group(
                Some("Use CPU cores only"),
                Some(&bg_processing_resources_to_use),
            );
            let rb_gpu_for_recon_and_ctf_correct = RadioButton::new_string_button_group(
                Some("Use the GPU for reconstruction and CTF correction"),
                Some(&bg_processing_resources_to_use),
            );
            let rb_cpu_for_recon_and_gpu_for_ctf_correct = RadioButton::new_string_button_group(
                Some("Use CPU cores for reconstruction and one GPU for CTF correction"),
                Some(&bg_processing_resources_to_use),
            );
            let adaptee = self_ref.clone();
            let action_listener: ActionListener = Rc::new(move |event: &ActionEvent| {
                if let Some(adaptee) = adaptee.upgrade() {
                    adaptee.action_performed(event);
                }
            });
            // Constructor body.
            let mediator = manager
                .get_processing_method_mediator(Some(axis_id))
                // Built on the event dispatch thread, where the mediator exists.
                .expect("processing method mediator on the event dispatch thread");
            let base_manager: &'static dyn BaseManager = manager;
            let local_gpu_available = Network::is_local_host_gpu_processing_enabled(
                base_manager,
                axis_id,
                manager.get_property_user_dir().as_deref(),
            );
            SubtomoSetupCpuGpuPanel {
                pnl_root,
                bg_processing_resources_to_use,
                rb_cpus_only,
                rb_gpu_for_recon_and_ctf_correct,
                rb_cpu_for_recon_and_gpu_for_ctf_correct,
                manager,
                axis_id,
                process_interface,
                mediator,
                subtomograms_panel,
                local_gpu_available,
                use_queue_check_box: RefCell::new(None),
                gpus_for_queue_available: Cell::new(false),
                mix_cpu_gpu_action_event: Cell::new(false),
                curr_cpu_gpu_selection: Cell::new(None),
                action_listener,
            }
        })
    }

    /// Java static `getSubtomoSetupInstance(SubtomogramsPanel,
    /// ApplicationManager, AxisID, ProcessInterface)`.
    pub fn get_subtomo_setup_instance(
        subtomograms_panel: Weak<SubtomogramsPanel>,
        manager: &'static ApplicationManager,
        axis_id: AxisID,
        process_interface: Weak<dyn ProcessInterface>,
    ) -> Rc<SubtomoSetupCpuGpuPanel> {
        let instance =
            SubtomoSetupCpuGpuPanel::new(subtomograms_panel, manager, axis_id, process_interface);
        instance.create_subtomo_setup_panel();
        instance.set_tooltips();
        instance.add_listeners();
        instance
    }

    /// Java `processInterface.setMethod(getProcessingMethod())`.
    fn process_interface_set_method(&self) {
        if let Some(process_interface) = self.process_interface.upgrade() {
            process_interface.set_method(self.get_processing_method());
        }
    }

    /// Java private `createSubtomoSetupPanel()`.
    fn create_subtomo_setup_panel(&self) {
        let pnl_processing_resources_to_use = JComponent::new_panel();
        let pnl_cpus_only = JComponent::new_panel();
        let pnl_gpu_for_recon_and_ctf_correct = JComponent::new_panel();
        let pnl_cpu_for_recon_and_gpu_for_ctf_correct = JComponent::new_panel();
        // init
        self.rb_cpus_only.set_selected_boolean(true);
        self.process_interface_set_method();
        // Root
        // Swing layout: pnlRoot BoxLayout X_AXIS.
        self.pnl_root.add(&pnl_processing_resources_to_use);
        // Processing Resources To Use panel
        // Swing layout: BoxLayout Y_AXIS.
        pnl_processing_resources_to_use.set_border_title(
            EtchedBorder::new(Some("Processing resources to use"))
                .get_border()
                .get_title()
                .as_deref(),
        );
        pnl_processing_resources_to_use.add(&pnl_cpus_only);
        pnl_processing_resources_to_use.add(&pnl_gpu_for_recon_and_ctf_correct);
        pnl_processing_resources_to_use.add(&pnl_cpu_for_recon_and_gpu_for_ctf_correct);
        // Swing layout: each row BoxLayout X_AXIS with horizontal glue after the
        // radio button.
        pnl_cpus_only.add(&self.rb_cpus_only.get_component());
        pnl_gpu_for_recon_and_ctf_correct
            .add(&self.rb_gpu_for_recon_and_ctf_correct.get_component());
        pnl_cpu_for_recon_and_gpu_for_ctf_correct.add(
            &self
                .rb_cpu_for_recon_and_gpu_for_ctf_correct
                .get_component(),
        );
    }

    /// Java package-private `addListeners()`.
    pub fn add_listeners(&self) {
        self.rb_cpus_only
            .add_action_listener(self.action_listener.clone());
        self.rb_gpu_for_recon_and_ctf_correct
            .add_action_listener(self.action_listener.clone());
        self.rb_cpu_for_recon_and_gpu_for_ctf_correct
            .add_action_listener(self.action_listener.clone());
        let gpu_component: Vec<Rc<dyn ButtonComponent>> = vec![
            self.rb_cpus_only.clone() as Rc<dyn ButtonComponent>,
            self.rb_gpu_for_recon_and_ctf_correct.clone() as Rc<dyn ButtonComponent>,
            self.rb_cpu_for_recon_and_gpu_for_ctf_correct.clone() as Rc<dyn ButtonComponent>,
        ];
        self.mediator.add_gpu_listener(gpu_component);
    }

    /// Java package-private `getComponent()`.
    pub fn get_component(&self) -> Rc<JComponent> {
        self.pnl_root.clone()
    }

    /// Java public `getProcessingMethod()`.  If parallel processing is
    /// required, then this will return a processing method that is not valid
    /// when parallel processing is not set up.  This should be handled by the
    /// function running the process.
    pub fn get_processing_method(&self) -> ProcessingMethod {
        if self.rb_gpu_for_recon_and_ctf_correct.is_selected() {
            return ProcessingMethod::PpGpu;
        }
        ProcessingMethod::PpCpu
    }

    /// Java `actionPerformed(ActionEvent)`.
    pub fn action_performed(&self, event: &ActionEvent) {
        let action_command = event.get_action_command();
        let use_queue_check_box = self.use_queue_check_box.borrow().clone();
        // Java `x.getActionCommand().equals(actionCommand)`.
        if self.rb_cpus_only.get_action_command().is_some()
            && self.rb_cpus_only.get_action_command().as_deref() == action_command
        {
            self.process_interface_set_method();
        } else if self
            .rb_gpu_for_recon_and_ctf_correct
            .get_action_command()
            .is_some()
            && self
                .rb_gpu_for_recon_and_ctf_correct
                .get_action_command()
                .as_deref()
                == action_command
        {
            self.process_interface_set_method();
        } else if self
            .rb_cpu_for_recon_and_gpu_for_ctf_correct
            .get_action_command()
            .is_some()
            && self
                .rb_cpu_for_recon_and_gpu_for_ctf_correct
                .get_action_command()
                .as_deref()
                == action_command
        {
            self.mix_cpu_gpu_action_event.set(true);
            self.process_interface_set_method();
            self.mix_cpu_gpu_action_event.set(false);
        } else if let Some(use_queue_check_box) = use_queue_check_box.filter(|check_box| {
            check_box.get_action_command().is_some()
                && check_box.get_action_command().as_deref() == action_command
        }) {
            if use_queue_check_box.is_selected() {
                self.set_curr_cpu_gpu_selection();
                self.rb_cpu_for_recon_and_gpu_for_ctf_correct
                    .set_enabled(false);
                if Network::is_any_queue_gpu() {
                    self.rb_gpu_for_recon_and_ctf_correct
                        .set_selected_boolean(true);
                    self.process_interface_set_method();
                } else {
                    self.rb_cpus_only.set_selected_boolean(true);
                    self.rb_gpu_for_recon_and_ctf_correct.set_enabled(false);
                }
            } else {
                // Revert to previous non-queue states
                self.rb_gpu_for_recon_and_ctf_correct.set_enabled(true);
                if self.is_extent_of_z_levels_in_nm() {
                    self.rb_cpu_for_recon_and_gpu_for_ctf_correct
                        .set_enabled(true);
                }
                self.revert_to_non_queue_cpu_gpu_selection();
            }
        }
        self.update_display();
    }

    /// Java `subtomogramsPanel.isExtentOfZLevelsInNm()`.  A panel that is gone
    /// answers false (the Java owner outlives this panel).
    fn is_extent_of_z_levels_in_nm(&self) -> bool {
        self.subtomograms_panel
            .upgrade()
            .is_some_and(|panel| panel.is_extent_of_z_levels_in_nm())
    }

    /// Java package-private `setTooltips()`.
    pub fn set_tooltips(&self) {
        // Java `ReadOnlyAutodoc autodoc = null;` then the try/catch.  The Java
        // passes a null AxisID; SUBTOMO_SETUP is not a per-axis autodoc, so
        // `AxisID::Only` stands in for it.
        let mut autodoc: *const dyn ReadOnlyAutodoc = std::ptr::null::<Autodoc>();
        // SAFETY: the factory keeps every autodoc it returns (and its sections)
        // for the life of the process.
        match unsafe {
            autodoc_factory::get_instance(
                Some(self.manager),
                Some(autodoc_factory::SUBTOMO_SETUP),
                AxisID::Only,
                false,
            )
        } {
            Ok(instance) => autodoc = instance as *const Autodoc,
            // catch (final LockException except) {}
            Err(LogFileError::Lock(_)) => {}
            // catch (final LogFileException | IOException except):
            // except.printStackTrace().
            Err(except) => eprintln!("{except}"),
        }
        if !autodoc.is_null() {
            // SAFETY: see above.
            let autodoc: &dyn ReadOnlyAutodoc = unsafe { &*autodoc };
            let autodoc_name = autodoc.get_autodoc_name();
            let section_when_to_use_gpu = unsafe {
                autodoc.get_section(
                    Some(etomo_autodoc::FIELD_SECTION_NAME),
                    Some(subtomo_setup_param::WHEN_TO_USE_GPU),
                )
            };
            // Java `EtomoAutodoc.getTooltip(String, ReadOnlySection, String)`: a
            // null section is caught (NullPointerException) and answers null.
            let tooltip = |value: i32| -> Option<String> {
                if section_when_to_use_gpu.is_null() {
                    return None;
                }
                // SAFETY: see above.
                let section: &dyn ReadOnlySection = unsafe { &*section_when_to_use_gpu };
                etomo_autodoc::get_tooltip_enum_value_name(
                    Some(&autodoc_name),
                    section,
                    Some(&value.to_string()),
                )
            };
            self.rb_cpus_only.set_tool_tip_text_string(
                tooltip(subtomo_setup_param::WHEN_TO_USE_GPU_VAL_0).as_deref(),
            );
            self.rb_gpu_for_recon_and_ctf_correct
                .set_tool_tip_text_string(
                    tooltip(subtomo_setup_param::WHEN_TO_USE_GPU_VAL_1).as_deref(),
                );
            self.rb_cpu_for_recon_and_gpu_for_ctf_correct
                .set_tool_tip_text_string(
                    tooltip(subtomo_setup_param::WHEN_TO_USE_GPU_VAL_2).as_deref(),
                );
        }
    }

    /// Java public `isSelectedCpusOnly()`.
    pub fn is_selected_cpus_only(&self) -> bool {
        self.rb_cpus_only.is_selected()
    }

    /// Java public `setSelectedCpusOnly(boolean)`.
    pub fn set_selected_cpus_only(&self, selected: bool) {
        self.rb_cpus_only.set_selected_boolean(selected);
        self.process_interface_set_method();
    }

    /// Java public `isSelectedGpuForReconAndCtfCorrect()`.
    pub fn is_selected_gpu_for_recon_and_ctf_correct(&self) -> bool {
        self.rb_gpu_for_recon_and_ctf_correct.is_selected()
    }

    /// Java public `setSelectedGpuForReconAndCtfCorrect(boolean)`.
    pub fn set_selected_gpu_for_recon_and_ctf_correct(&self, selected: bool) {
        self.rb_gpu_for_recon_and_ctf_correct
            .set_selected_boolean(selected);
        self.process_interface_set_method();
    }

    /// Java public `setEnabledGpuForReconAndCtfCorrect(boolean)`.
    pub fn set_enabled_gpu_for_recon_and_ctf_correct(&self, enabled: bool) {
        self.rb_gpu_for_recon_and_ctf_correct.set_enabled(enabled);
    }

    /// Java public `isSelectedCpuForReconAndGpuForCtfCorrect()`.
    pub fn is_selected_cpu_for_recon_and_gpu_for_ctf_correct(&self) -> bool {
        self.rb_cpu_for_recon_and_gpu_for_ctf_correct.is_selected()
    }

    /// Java public `setSelectedCpuForReconAndGpuForCtfCorrect(boolean)`.
    pub fn set_selected_cpu_for_recon_and_gpu_for_ctf_correct(&self, selected: bool) {
        self.rb_cpu_for_recon_and_gpu_for_ctf_correct
            .set_selected_boolean(selected);
        self.process_interface_set_method();
    }

    /// Java public `setEnabledCpuForReconAndGpuForCtfCorrect(boolean)`.
    pub fn set_enabled_cpu_for_recon_and_gpu_for_ctf_correct(&self, enabled: bool) {
        let use_queue_check_box = self.use_queue_check_box.borrow().clone();
        if use_queue_check_box.is_none_or(|check_box| !check_box.is_selected()) {
            self.rb_cpu_for_recon_and_gpu_for_ctf_correct
                .set_enabled(enabled);
        }
    }

    /// Java package-private `updateGpu(boolean)`.
    pub fn update_gpu(&self, _disable_gpu: bool) {
        self.update_display();
    }

    /// Java package-private `getSecondaryProcessingMethod()`: null.
    pub fn get_secondary_processing_method(&self) -> Option<ProcessingMethod> {
        None
    }

    /// Java package-private `lockProcessingMethod(boolean)`: empty.
    pub fn lock_processing_method(&self, _lock: bool) {}

    /// Java package-private `setMethod(ProcessingMethod)`: empty.
    pub fn set_method(&self, _processing_method: ProcessingMethod) {}

    /// Java package-private `isUseGpu()`.
    pub fn is_use_gpu(&self) -> bool {
        self.rb_gpu_for_recon_and_ctf_correct.is_enabled()
            && self.rb_gpu_for_recon_and_ctf_correct.is_selected()
    }

    /// Java package-private `setUseQueueCheckBox(ButtonComponent)`.
    pub fn set_use_queue_check_box(&self, use_queue_check_box: Option<Rc<dyn ButtonComponent>>) {
        if let Some(use_queue_check_box) = use_queue_check_box {
            if self.use_queue_check_box.borrow().is_none() {
                use_queue_check_box.add_action_listener(self.action_listener.clone());
                *self.use_queue_check_box.borrow_mut() = Some(use_queue_check_box);
            }
        }
    }

    /// Java package-private `updateDisplay()`.
    pub fn update_display(&self) {
        let use_queue_check_box = self.use_queue_check_box.borrow().clone();
        self.rb_cpu_for_recon_and_gpu_for_ctf_correct.set_enabled(
            use_queue_check_box.is_none_or(|check_box| !check_box.is_selected())
                && self.is_extent_of_z_levels_in_nm(),
        );
    }

    /// Java private `setCurrCpuGpuSelection()`.
    fn set_curr_cpu_gpu_selection(&self) {
        if self.rb_cpus_only.is_selected() {
            self.curr_cpu_gpu_selection
                .set(Some(CpuGpuSelection::CpuOnly));
        } else if self.rb_gpu_for_recon_and_ctf_correct.is_selected() {
            self.curr_cpu_gpu_selection
                .set(Some(CpuGpuSelection::GpuOnly));
        } else {
            self.curr_cpu_gpu_selection
                .set(Some(CpuGpuSelection::MixCpuGpu));
        }
    }

    /// Java private `revertToNonQueueCpuGpuSelection()`.
    fn revert_to_non_queue_cpu_gpu_selection(&self) {
        let curr_cpu_gpu_selection = self.curr_cpu_gpu_selection.get();
        if curr_cpu_gpu_selection == Some(CpuGpuSelection::CpuOnly) {
            self.rb_cpus_only.set_selected_boolean(true);
            self.process_interface_set_method();
        } else if curr_cpu_gpu_selection == Some(CpuGpuSelection::GpuOnly) {
            self.rb_gpu_for_recon_and_ctf_correct
                .set_selected_boolean(true);
            self.process_interface_set_method();
        } else if self.is_extent_of_z_levels_in_nm() {
            self.rb_cpu_for_recon_and_gpu_for_ctf_correct
                .set_selected_boolean(true);
            self.process_interface_set_method();
        }
    }

    /// Java package-private `isMixCpuGpuActionEvent()`.
    pub fn is_mix_cpu_gpu_action_event(&self) -> bool {
        self.mix_cpu_gpu_action_event.get()
    }
}
