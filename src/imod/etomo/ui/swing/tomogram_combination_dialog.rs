//! `IMOD/Etomo/src/etomo/ui/swing/TomogramCombinationDialog.java`.
//!
//! The Tomogram Combination process dialog: three tabs (Setup, Initial Match,
//! Final Match), each a panel of its own (`SetupCombinePanel`,
//! `InitialCombinePanel`, `FinalCombinePanel`).  The Setup tab shares fields
//! with the other two (`InitialCombineFields`, `FinalCombineFields`), and the
//! dialog copies them between tabs ([`TomogramCombinationDialog::synchronize`])
//! when the user leaves a tab or presses a button.
//!
//! Java `public final class TomogramCombinationDialog extends ProcessDialog
//! implements ContextMenu, AbstractParallelDialog, ProcessInterface`: an EDT
//! object created as `Rc<Self>` by [`TomogramCombinationDialog::new`]; every
//! method takes `&self`.  The `ProcessDialog` superclass is the embedded
//! `base` (reached through `Deref`), and the overridden `done()` is
//! `ProcessDialogVirtual::done`.  The inner class `TabChangeListener` is a
//! closure holding a weak reference to the dialog.
//!
//! **Construction order.**  The Java constructor hands `this` to the three
//! panels, and they read the dialog's `parallelProcessCheckBoxText`.  So the
//! dialog is put in its `Rc` first, and the panels are then created in the
//! Java order and stored in `OnceCell`s; after `new` returns they are always
//! set (the Java fields are assigned once, in the constructor).
//!
//! **Naming.**  `synchronize(String, boolean)` is overloaded in the Java with
//! the two private field-copying methods; the public one keeps the plain name
//! `synchronize` because that is what `ApplicationManager` calls, and the
//! private overloads carry the parameter-type suffix.

use std::cell::{Cell, OnceCell};
use std::fmt;
use std::ops::Deref;
use std::rc::{Rc, Weak};

use super::abstract_parallel_dialog::AbstractParallelDialog;
use super::beveled_border::BeveledBorder;
use super::button_component::ButtonComponent;
use super::context_menu::ContextMenu;
use super::context_popup::{self, ContextPopup};
use super::final_combine_fields::FinalCombineFields;
use super::final_combine_panel::FinalCombinePanel;
use super::initial_combine_fields::InitialCombineFields;
use super::initial_combine_panel::InitialCombinePanel;
use super::parallel_panel;
use super::process_dialog::{ProcessDialog, ProcessDialogVirtual};
use super::process_interface::ProcessInterface;
use super::run_3dmod_button::Run3dmodButton;
use super::setup_combine_panel::SetupCombinePanel;
use super::tabbed_pane::TabbedPane;
use super::ui_harness;
use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::combine_params::CombineParams;
use crate::imod::etomo::comscript::const_combine_params::ConstCombineParams;
use crate::imod::etomo::comscript::const_matchorwarp_param::ConstMatchorwarpParam;
use crate::imod::etomo::comscript::const_patchcrawl3d_param::ConstPatchcrawl3DParam;
use crate::imod::etomo::comscript::const_solvematch_param::ConstSolvematchParam;
use crate::imod::etomo::comscript::dualvolmatch_param::DualvolmatchParam;
use crate::imod::etomo::comscript::matchorwarp_param::MatchorwarpParam;
use crate::imod::etomo::comscript::matchvol_param::MatchvolParam;
use crate::imod::etomo::comscript::parallel_param::ParallelParam;
use crate::imod::etomo::comscript::patchcrawl3d_param::Patchcrawl3DParam;
use crate::imod::etomo::comscript::set_param::SetParam;
use crate::imod::etomo::comscript::solvematch_param::SolvematchParam;
use crate::imod::etomo::jdk::{ChangeEvent, ChangeListener, JComponent, MouseEvent};
use crate::imod::etomo::processing_method_mediator::ProcessingMethodMediator;
use crate::imod::etomo::storage::cpu_adoc;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::combine_process_type::CombineProcessType;
use crate::imod::etomo::r#type::const_etomo_number::ConstEtomoNumber;
use crate::imod::etomo::r#type::const_meta_data::ConstMetaData;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::match_mode::MatchMode;
use crate::imod::etomo::r#type::meta_data::MetaData;
use crate::imod::etomo::r#type::processing_method::ProcessingMethod;
use crate::imod::etomo::r#type::recon_screen_state::ReconScreenState;
use crate::imod::etomo::r#type::tomogram_state::TomogramState;
use crate::imod::etomo::ui::queue_table_event::QueueTableEvent;
use crate::imod::etomo::ui::queue_table_listener::QueueTableListener;

/// Java private static final `SETUP_INDEX`.
const SETUP_INDEX: usize = 0;
/// Java private static final `INITIAL_INDEX`.
const INITIAL_INDEX: usize = 1;
/// Java private static final `FINAL_INDEX`.
const FINAL_INDEX: usize = 2;
/// Java public static final `lblSetup`.
pub const LBL_SETUP: &str = "Setup";
/// Java public static final `lblInitial`.
pub const LBL_INITIAL: &str = "Initial Match";
/// Java public static final `lblFinal`.
pub const LBL_FINAL: &str = "Final Match";
/// Java public static final `ALL_FIELDS`.
pub const ALL_FIELDS: i32 = 10;

/// Java `public final class TomogramCombinationDialog extends ProcessDialog
/// implements ContextMenu, AbstractParallelDialog, ProcessInterface`.
pub struct TomogramCombinationDialog {
    /// The `ProcessDialog` superclass.
    base: Rc<ProcessDialog>,

    /// Java private `pnlSetup` (set by the constructor).
    pnl_setup: OnceCell<Rc<SetupCombinePanel>>,
    /// Java private `pnlInitial` (set by the constructor).
    pnl_initial: OnceCell<Rc<InitialCombinePanel>>,
    /// Java private `pnlFinal` (set by the constructor).
    pnl_final: OnceCell<Rc<FinalCombinePanel>>,
    /// Java private `combinePanelEnabled` (never assigned; default false).
    combine_panel_enabled: Cell<bool>,
    /// Java private `parallelPanelContainer = new JPanel()`.
    parallel_panel_container: Rc<JComponent>,
    /// Java private final `mediator`.
    mediator: Rc<ProcessingMethodMediator>,

    /// Java private `tabbedPane = new TabbedPane()`.
    tabbed_pane: Rc<TabbedPane>,
    /// Java package-private final `parallelProcessCheckBoxText`.
    pub parallel_process_check_box_text: String,
    /// Java private `constructed = false`.
    constructed: Cell<bool>,

    /// Java private `idxLastTab`.  This is the index of the last tab to keep
    /// track of what to sync from when switching tabs.
    idx_last_tab: Cell<i32>,

    /// Java `this` (handed to the mediator).
    this: Weak<TomogramCombinationDialog>,
}

impl Deref for TomogramCombinationDialog {
    type Target = ProcessDialog;
    fn deref(&self) -> &ProcessDialog {
        &self.base
    }
}

impl fmt::Display for TomogramCombinationDialog {
    /// Java public override `toString()`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // getClass().getName() + "[" + paramString() + "]\n"
        write!(
            f,
            "etomo.ui.swing.TomogramCombinationDialog[{}]\n",
            self.param_string()
        )
    }
}

impl TomogramCombinationDialog {
    /// Java package-private `paramString()`.  `SetupCombinePanel` and
    /// `InitialCombinePanel` have no `toString`, so Java prints
    /// `Object.toString()` (class name and identity hash); the address stands
    /// in for the hash.
    pub fn param_string(&self) -> String {
        format!(
            "pnlSetup=etomo.ui.swing.SetupCombinePanel@{:x},\n\
             pnlInitial=etomo.ui.swing.InitialCombinePanel@{:x},\npnlFinal={},\n\
             combinePanelEnabled={},\nparallelProcessCheckBoxText={},\nidxLastTab={}",
            Rc::as_ptr(self.pnl_setup()) as usize,
            Rc::as_ptr(self.pnl_initial()) as usize,
            self.pnl_final(),
            self.combine_panel_enabled.get(),
            self.parallel_process_check_box_text,
            self.idx_last_tab.get()
        )
    }

    /// Java public constructor `TomogramCombinationDialog(ApplicationManager)`.
    pub fn new(app_mgr: &'static ApplicationManager) -> Rc<TomogramCombinationDialog> {
        // super(appMgr, AxisID.FIRST, DialogType.TOMOGRAM_COMBINATION)
        let base = ProcessDialog::new_application_manager_axis_id_dialog_type(
            app_mgr,
            AxisID::First,
            DialogType::TomogramCombination,
        );
        // Field initializers.
        let parallel_panel_container = JComponent::new_panel();
        let tabbed_pane = TabbedPane::new();
        // Constructor body.
        let max_cpus: ConstEtomoNumber = cpu_adoc::INSTANCE.get_max_volcombine();
        let mediator = app_mgr
            .get_processing_method_mediator(Some(base.axis_id))
            // Built on the event dispatch thread, where the mediator exists.
            .expect("processing method mediator on the event dispatch thread");
        // `maxCPUs != null && !maxCPUs.isNull()`: the Rust value is never null.
        let parallel_process_check_box_text = if !max_cpus.is_null() {
            format!(
                "{}{}{}",
                parallel_panel::FIELD_LABEL,
                parallel_panel::MAX_CPUS_STRING,
                max_cpus
            )
        } else {
            parallel_panel::FIELD_LABEL.to_string()
        };
        let instance =
            Rc::new_cyclic(
                |this: &Weak<TomogramCombinationDialog>| TomogramCombinationDialog {
                    base,
                    pnl_setup: OnceCell::new(),
                    pnl_initial: OnceCell::new(),
                    pnl_final: OnceCell::new(),
                    combine_panel_enabled: Cell::new(false),
                    parallel_panel_container,
                    mediator,
                    tabbed_pane,
                    parallel_process_check_box_text,
                    constructed: Cell::new(false),
                    idx_last_tab: Cell::new(0),
                    this: this.clone(),
                },
            );
        // Java `this` as the ProcessDialog subclass (for the virtual `done()`).
        let this: Weak<dyn ProcessDialogVirtual> =
            Rc::downgrade(&instance) as Weak<dyn ProcessDialogVirtual>;
        instance.base.set_this(this);
        let application_manager = instance.base.application_manager;
        let dialog_type = instance.base.dialog_type;
        // Instantiate the tab pane contents
        let pnl_setup = SetupCombinePanel::get_instance(
            Rc::downgrade(&instance),
            application_manager,
            dialog_type,
        );
        let _ = instance.pnl_setup.set(pnl_setup);
        let pnl_initial = InitialCombinePanel::get_instance(
            Rc::downgrade(&instance),
            application_manager,
            dialog_type,
            &instance.base.btn_advanced,
        );
        let _ = instance.pnl_initial.set(pnl_initial);
        let pnl_final = FinalCombinePanel::new(
            Rc::downgrade(&instance),
            application_manager,
            dialog_type,
            &instance.base.btn_advanced,
        );
        let _ = instance.pnl_final.set(pnl_final);

        let root_panel = instance.base.root_panel.get_component();
        // Swing layout: rootPanel.setLayout(new BoxLayout(rootPanel, BoxLayout.Y_AXIS)).
        root_panel.add(&instance.parallel_panel_container);
        // Construct the main panel for this dialog panel
        // JTabbedPane.add(String, Component) calls the overridden addTab.
        instance
            .tabbed_pane
            .add_tab_string_component(LBL_SETUP, &instance.pnl_setup().get_container());
        instance
            .tabbed_pane
            .add_tab_string_component(LBL_INITIAL, &instance.pnl_initial().get_container());
        instance
            .tabbed_pane
            .add_tab_string_component(LBL_FINAL, &instance.pnl_final().get_container());

        instance
            .base
            .root_panel
            .set_border(&BeveledBorder::new(Some("Tomogram Combination")).get_border());
        let z_warning =
            JComponent::new_label("For all 3D parameters Z represents the depth domain");
        // Swing layout: zWarning.setAlignmentX(Component.CENTER_ALIGNMENT).
        root_panel.add(&z_warning);
        root_panel.add(&instance.tabbed_pane.get_component());
        instance.base.add_exit_buttons();
        instance.base.btn_execute.set_text(Some("Done"));

        // Java `TabChangeListener tabChangeListener = new TabChangeListener(this)`.
        let adaptee = Rc::downgrade(&instance);
        let tab_change_listener: ChangeListener = Rc::new(move |event: &ChangeEvent| {
            if let Some(adaptee) = adaptee.upgrade() {
                adaptee.tab_state_change(event);
            }
        });
        instance
            .tabbed_pane
            .get_component()
            .add_change_listener(tab_change_listener);

        // Set the default advanced dialog state
        instance.update_advanced();

        instance
            .idx_last_tab
            .set(instance.tabbed_pane.get_component().get_selected_tab());
        instance.set_visible(LBL_SETUP);
        instance.constructed.set(true);
        instance.pnl_setup().set_deferred_3dmod_buttons();
        instance.pnl_initial().set_deferred_3dmod_buttons();
        instance.update_display();
        instance
    }

    /// Rust-only: the Java field `pnlSetup` (set by the constructor).
    fn pnl_setup(&self) -> &Rc<SetupCombinePanel> {
        self.pnl_setup.get().expect("set by the constructor")
    }

    /// Rust-only: the Java field `pnlInitial` (set by the constructor).
    fn pnl_initial(&self) -> &Rc<InitialCombinePanel> {
        self.pnl_initial.get().expect("set by the constructor")
    }

    /// Rust-only: the Java field `pnlFinal` (set by the constructor).
    fn pnl_final(&self) -> &Rc<FinalCombinePanel> {
        self.pnl_final.get().expect("set by the constructor")
    }

    /// Java `this` as a `ProcessInterface` (for the mediator).
    fn as_process_interface(&self) -> Option<Rc<dyn ProcessInterface>> {
        self.this
            .upgrade()
            .map(|this| this as Rc<dyn ProcessInterface>)
    }

    /// Java public `removeListeners()`.
    pub fn remove_listeners(&self) {
        self.pnl_setup().remove_listeners();
        self.pnl_initial().remove_listeners();
        self.pnl_final().remove_listeners();
    }

    /// Java public `setCombineParams(ConstCombineParams, boolean)`.  Set the
    /// setupcombine parameters of the UI from the the ConstCombineParams
    /// object.
    pub fn set_combine_params(&self, combine_params: &dyn ConstCombineParams, init: bool) {
        self.pnl_setup()
            .set_parameters_const_combine_params_boolean(combine_params, init);
        self.pnl_final()
            .set_parameters_const_combine_params(combine_params);
    }

    /// Java public `setZMin(String)`.
    pub fn set_z_min(&self, z_min: &str) {
        FinalCombineFields::set_z_min(&**self.pnl_setup(), Some(z_min));
    }

    /// Java public `setZMax(String)`.
    pub fn set_z_max(&self, z_max: &str) {
        FinalCombineFields::set_z_max(&**self.pnl_setup(), Some(z_max));
    }

    /// Java public `getCombineParams(CombineParams, boolean) throws
    /// NumberFormatException`.  Get the the setupcombine parameters of the UI
    /// returning them in the modified CombineParams object; assumes
    /// synchronize is done.  `Err` is the NumberFormatException's message.
    pub fn get_combine_params(
        &self,
        combine_params: &mut CombineParams,
        do_validation: bool,
    ) -> Result<bool, String> {
        self.pnl_setup()
            .get_parameters_combine_params_boolean(combine_params, do_validation)
    }

    /// Java package-private `getImodCombinedButton()`.
    pub fn get_imod_combined_button(&self) -> Rc<Run3dmodButton> {
        self.pnl_final().get_imod_combined_button()
    }

    /// Java public `setSolvematchParams(ConstSolvematchParam)`.  Set the
    /// solvematch parameters of the UI from the the ConstSolvematchParams
    /// object.
    pub fn set_solvematch_params(&self, solvematch_params: &ConstSolvematchParam) {
        self.pnl_initial().set_solvematch_params(solvematch_params);
    }

    /// Java public `setDualvolmatchParams(DualvolmatchParam)`.
    pub fn set_dualvolmatch_params(&self, param: &DualvolmatchParam) {
        self.pnl_initial().set_dualvolmatch_params(param);
    }

    /// Java public `setParameters(MatchvolParam)`.
    pub fn set_parameters_matchvol_param(&self, param: &MatchvolParam) {
        self.pnl_initial().set_parameters_matchvol_param(param);
    }

    /// Java public `setParameters(ReconScreenState)`.
    pub fn set_parameters_recon_screen_state(&self, screen_state: &ReconScreenState) {
        self.pnl_setup()
            .set_parameters_recon_screen_state(screen_state);
        self.pnl_initial()
            .set_parameters_recon_screen_state(screen_state);
        self.pnl_final()
            .set_parameters_recon_screen_state(screen_state);
    }

    /// Java public `getParameters(ReconScreenState)`.
    pub fn get_parameters_recon_screen_state(&self, screen_state: &ReconScreenState) {
        self.pnl_setup()
            .get_parameters_recon_screen_state(screen_state);
        self.pnl_initial()
            .get_parameters_recon_screen_state(screen_state);
        self.pnl_final()
            .get_parameters_recon_screen_state(screen_state);
    }

    /// Java public `showPane(CombineProcessType)`.  Show the specified tab
    /// pane.  (Java returns on a null type; the Rust value is never null.)
    pub fn show_pane(&self, combine_process_type: CombineProcessType) {
        if combine_process_type == CombineProcessType::SOLVEMATCH
            || combine_process_type == CombineProcessType::DUALVOLMATCH
            || combine_process_type == CombineProcessType::MATCHVOL1
        {
            self.tabbed_pane
                .get_component()
                .set_selected_tab(INITIAL_INDEX as i32);
        } else {
            self.tabbed_pane
                .get_component()
                .set_selected_tab(FINAL_INDEX as i32);
        }
        ui_harness::with(|harness| harness.move_sub_frame());
    }

    /// Java public `getParameters(MetaData)`.
    pub fn get_parameters_meta_data(&self, meta_data: &MetaData) {
        self.synchronize(LBL_SETUP, false);
        self.pnl_setup().get_parameters_meta_data(meta_data);
    }

    /// Java public `show()`.
    pub fn show(&self) {
        if let Some(this) = self.as_process_interface() {
            self.mediator.register_process_interface(this.clone());
            self.mediator
                .set_method_process_interface_processing_method(&this, ProcessingMethod::LocalCpu);
        }
        self.pnl_setup()
            .show(!self.is_changed(self.base.application_manager.get_state()));
        self.base.set_displayed(true);
    }

    /// Java public `setParameters(ConstMetaData)`.
    pub fn set_parameters_const_meta_data(&self, meta_data: &dyn ConstMetaData) {
        self.pnl_setup().set_parameters_const_meta_data(meta_data);
        self.synchronize(LBL_SETUP, true);
    }

    /// Java public `getSolvematchParams(SolvematchParam, boolean) throws
    /// NumberFormatException`.  Get the the solvematch parameters of the UI
    /// returning them in the modified SolvematchParam object.  `Err` is the
    /// NumberFormatException's message.
    pub fn get_solvematch_params(
        &self,
        solvematch_params: &mut SolvematchParam,
        do_validation: bool,
    ) -> Result<bool, String> {
        self.pnl_initial()
            .get_solvematch_params(solvematch_params, do_validation)
    }

    /// Java public `getParameters(DualvolmatchParam, boolean)`.
    pub fn get_parameters_dualvolmatch_param_boolean(
        &self,
        param: &mut DualvolmatchParam,
        do_validation: bool,
    ) -> bool {
        self.pnl_initial()
            .get_parameters_dualvolmatch_param_boolean(param, do_validation)
    }

    /// Java public `getParameters(MatchvolParam, boolean) throws
    /// NumberFormatException` (nothing under it throws one).
    pub fn get_parameters_matchvol_param_boolean(
        &self,
        param: &mut MatchvolParam,
        do_validation: bool,
    ) -> bool {
        self.pnl_initial()
            .get_parameters_matchvol_param_boolean(param, do_validation)
    }

    /// Java public `setPatchcrawl3DParams(ConstPatchcrawl3DParam)`.  Set the
    /// patchcrawl3D parameters of the UI from the the ConstPatchcrawl3DParam
    /// object.
    pub fn set_patchcrawl3_d_params(&self, patchcrawl3_d_params: &ConstPatchcrawl3DParam) {
        self.pnl_setup()
            .set_parameters_const_patchcrawl3d_param(patchcrawl3_d_params);
        self.pnl_final()
            .set_patchcrawl3_d_params(patchcrawl3_d_params);
    }

    /// Java public `setReductionFactorParams(ConstSetParam)`.
    pub fn set_reduction_factor_params(&self, set_param: Option<&SetParam>) {
        self.pnl_final().set_reduction_factor_params(set_param);
    }

    /// Java public `setLowFromBothRadiusParams(ConstSetParam)`.
    pub fn set_low_from_both_radius_params(&self, set_param: Option<&SetParam>) {
        self.pnl_final().set_low_from_both_radius_params(set_param);
    }

    /// Java public `getPatchcrawl3DParams(Patchcrawl3DParam, boolean) throws
    /// NumberFormatException`.  Get the the patchcrawl3d parameters of the UI
    /// returning them in the modified Patchcrawl3DParam object.  `Err` is the
    /// NumberFormatException's message.
    pub fn get_patchcrawl3_d_params(
        &self,
        patchcrawl3_d_params: &mut Patchcrawl3DParam,
        do_validation: bool,
    ) -> Result<bool, String> {
        self.pnl_final()
            .get_patchcrawl3_d_params(patchcrawl3_d_params, do_validation)
    }

    /// Java public `getReductionFactorParam(SetParam, boolean)`.
    pub fn get_reduction_factor_param(
        &self,
        set_param: Option<&mut SetParam>,
        do_validation: bool,
    ) -> bool {
        self.pnl_final()
            .get_reduction_factor_param(set_param, do_validation)
    }

    /// Java public `getLowFromBothRadiusParam(SetParam, boolean)`.
    pub fn get_low_from_both_radius_param(
        &self,
        set_param: Option<&mut SetParam>,
        do_validation: bool,
    ) -> bool {
        self.pnl_final()
            .get_low_from_both_radius_param(set_param, do_validation)
    }

    /// Java public `enableReductionFactor(boolean)`.
    pub fn enable_reduction_factor(&self, enable: bool) {
        self.pnl_final().enable_reduction_factor(enable);
    }

    /// Java public `enableLowFromBothRadius(boolean)`.
    pub fn enable_low_from_both_radius(&self, enable: bool) {
        self.pnl_final().enable_low_from_both_radius(enable);
    }

    /// Java package-private `getMatchMode()`.
    pub fn get_match_mode(&self) -> Option<MatchMode> {
        InitialCombineFields::get_match_mode(&**self.pnl_setup())
    }

    /// Java public `setMatchorwarpParams(ConstMatchorwarpParam)`.  Set the
    /// matchorwarp parameters of the UI from the the ConstMatchorwarp object.
    pub fn set_matchorwarp_params(&self, matchorwarp_params: &dyn ConstMatchorwarpParam) {
        self.pnl_final().set_matchorwarp_params(matchorwarp_params);
    }

    /// Java public `getRunProcessingMethod()`.  This is an unusual situation.
    /// In the combine dialog it is possible to start a parallel process from a
    /// tab where the parallel processing table is not displayed.  So I have to
    /// distinguish between getting the processing method for display from
    /// running from the Initial panel.
    pub fn get_run_processing_method(&self) -> ProcessingMethod {
        let tabbed_pane = self.tabbed_pane.get_component();
        let tab_title = tabbed_pane.get_title_at(tabbed_pane.get_selected_tab() as usize);
        if tab_title.as_deref() == Some(LBL_INITIAL) {
            // Tabs copy their data to other tabs when other tabs is selected, so
            // either setup or final should be correct.
            self.pnl_final().get_processing_method()
        } else {
            self.get_processing_method()
        }
    }

    /// Java public `synchronize(String, boolean)`.  Synchronizes setup panel
    /// to/from initial and final panels.  `copy_from_tab` is true when
    /// synchronizing data from the specified tab to the other tab(s), false
    /// when copying data into the current tab (when running combine on the
    /// setup tab).
    pub fn synchronize(&self, tab_title: &str, copy_from_tab: bool) {
        let pnl_setup: &SetupCombinePanel = self.pnl_setup();
        let pnl_initial: &InitialCombinePanel = self.pnl_initial();
        let pnl_final: &FinalCombinePanel = self.pnl_final();
        if tab_title == LBL_SETUP {
            if copy_from_tab {
                self.synchronize_initial_combine_fields_initial_combine_fields(
                    pnl_setup,
                    pnl_initial,
                );
                self.synchronize_final_combine_fields_final_combine_fields(pnl_setup, pnl_final);
            } else {
                self.synchronize_initial_combine_fields_initial_combine_fields(
                    pnl_initial,
                    pnl_setup,
                );
                self.synchronize_final_combine_fields_final_combine_fields(pnl_final, pnl_setup);
            }
        } else if tab_title == LBL_INITIAL {
            if copy_from_tab {
                self.synchronize_initial_combine_fields_initial_combine_fields(
                    pnl_initial,
                    pnl_setup,
                );
            } else {
                self.synchronize_initial_combine_fields_initial_combine_fields(
                    pnl_setup,
                    pnl_initial,
                );
            }
        } else if tab_title == LBL_FINAL {
            if copy_from_tab {
                self.synchronize_final_combine_fields_final_combine_fields(pnl_final, pnl_setup);
            } else {
                self.synchronize_final_combine_fields_final_combine_fields(pnl_setup, pnl_final);
            }
        }
        self.update_display();
    }

    /// Java private `synchronize(InitialCombineFields, InitialCombineFields)`.
    /// Initial combine fields synchronization method.
    fn synchronize_initial_combine_fields_initial_combine_fields(
        &self,
        from_panel: &dyn InitialCombineFields,
        to_panel: &dyn InitialCombineFields,
    ) {
        if !from_panel.is_enabled() || !to_panel.is_enabled() {
            return;
        }
        to_panel.set_surfaces_or_models(from_panel.get_surfaces_or_models());
        to_panel.set_bin_by2(from_panel.is_bin_by2());
        to_panel.set_fiducial_match_list_a(from_panel.get_fiducial_match_list_a_void().as_deref());
        to_panel.set_fiducial_match_list_b(from_panel.get_fiducial_match_list_b_void().as_deref());
        to_panel.set_use_corresponding_points(from_panel.is_use_corresponding_points());
        to_panel.set_use_list(from_panel.get_use_list_void().as_deref());
        to_panel.set_match_mode(from_panel.get_match_mode());
        to_panel.set_initial_volume_matching(from_panel.is_initial_volume_matching());
    }

    /// Java private `synchronize(FinalCombineFields, FinalCombineFields)`.
    /// Final combine fields synchronization method.
    fn synchronize_final_combine_fields_final_combine_fields(
        &self,
        from_panel: &dyn FinalCombineFields,
        to_panel: &dyn FinalCombineFields,
    ) {
        if !from_panel.is_enabled() || !to_panel.is_enabled() {
            return;
        }
        to_panel.set_use_patch_region_model(from_panel.is_use_patch_region_model());
        to_panel.set_x_min(from_panel.get_x_min().as_deref());
        to_panel.set_x_max(from_panel.get_x_max().as_deref());
        to_panel.set_y_min(from_panel.get_y_min().as_deref());
        to_panel.set_y_max(from_panel.get_y_max().as_deref());
        to_panel.set_z_min(from_panel.get_z_min().as_deref());
        to_panel.set_z_max(from_panel.get_z_max().as_deref());
        to_panel.set_parallel(from_panel.is_parallel());
        to_panel.set_parallel_enabled(from_panel.is_parallel_enabled());
        to_panel.set_no_volcombine(from_panel.is_no_volcombine());
    }

    /// Java public `isChanged(TomogramState)`.
    pub fn is_changed(&self, state: &TomogramState) -> bool {
        self.pnl_setup().is_changed(state)
    }

    /// Java public `updateDisplay()`.
    pub fn update_display(&self) {
        if !self.constructed.get() {
            return;
        }
        let enable_tabs = !self.is_changed(self.base.application_manager.get_state());
        let tabbed_pane = self.tabbed_pane.get_component();
        tabbed_pane.set_enabled_at(INITIAL_INDEX, enable_tabs);
        tabbed_pane.set_enabled_at(FINAL_INDEX, enable_tabs);
        self.pnl_setup().update_display(enable_tabs);
        self.pnl_initial().update_display();
    }

    /// Java public `updatePatchVectorModelDisplay()`.
    pub fn update_patch_vector_model_display(&self) {
        self.pnl_final().update_patch_vector_model_display();
    }

    /// Java public `isRunVolcombine()`.
    pub fn is_run_volcombine(&self) -> bool {
        self.pnl_final().is_run_volcombine()
    }

    /// Java public `setRunVolcombine(boolean)`.
    pub fn set_run_volcombine(&self, run_volcombine: bool) {
        self.pnl_final().set_run_volcombine(run_volcombine);
    }

    /// Java public `isParallel()`.
    pub fn is_parallel(&self) -> bool {
        FinalCombineFields::is_parallel(&**self.pnl_final())
    }

    /// Java public `setBinningWarning(boolean)`.
    pub fn set_binning_warning(&self, binning_warning: bool) {
        self.pnl_setup().set_binning_warning(binning_warning);
    }

    /// Java public `getMatchorwarpParams(MatchorwarpParam, boolean) throws
    /// NumberFormatException`.  Get the the matchorwarp parameters of the UI
    /// returning them in the modified MatchorwarpParam object.  (Nothing under
    /// it throws the declared exception; see
    /// `FinalCombinePanel::get_matchorwarp_params`.)
    pub fn get_matchorwarp_params(
        &self,
        matchorwarp_params: &mut MatchorwarpParam,
        do_validation: bool,
    ) -> bool {
        self.pnl_final()
            .get_matchorwarp_params(matchorwarp_params, do_validation)
    }

    /// Java public `synchronizeFromCurrentTab()`.
    pub fn synchronize_from_current_tab(&self) {
        let title = self
            .tabbed_pane
            .get_component()
            .get_title_at(self.idx_last_tab.get() as usize);
        // A tab title is never null; an out-of-range index (Java
        // IndexOutOfBoundsException) matches no tab.
        self.synchronize(title.as_deref().unwrap_or(""), true);
    }

    /// Java private `updateAdvanced()`.  Update the dialog with the current
    /// advanced state.
    fn update_advanced(&self) {
        self.pnl_initial().update_advanced(self.base.is_advanced());
        self.pnl_final().update_advanced(self.base.is_advanced());
        let axis_id = self.base.axis_id;
        let manager: &'static dyn BaseManager = self.base.application_manager;
        ui_harness::with(|harness| harness.pack_axis_id_base_manager(Some(axis_id), Some(manager)));
    }

    /// Java public `isTabEnabled(String)`.
    ///
    /// Java throws `IllegalArgumentException("tabLabel=" + tabLabel)` for any
    /// other label; every caller passes one of the three constants.
    pub fn is_tab_enabled(&self, tab_label: &str) -> bool {
        let tabbed_pane = self.tabbed_pane.get_component();
        if tab_label == LBL_SETUP {
            return tabbed_pane.is_enabled_at(SETUP_INDEX);
        }
        if tab_label == LBL_INITIAL {
            return tabbed_pane.is_enabled_at(INITIAL_INDEX);
        }
        if tab_label == LBL_FINAL {
            return tabbed_pane.is_enabled_at(FINAL_INDEX);
        }
        panic!("java.lang.IllegalArgumentException: tabLabel={tab_label}");
    }

    /// Java package-private `tabStateChange(ChangeEvent)`.  Handle tab state
    /// changes.
    pub fn tab_state_change(&self, _event: &ChangeEvent) {
        let tabbed_pane = self.tabbed_pane.get_component();
        let idx_new_tab = tabbed_pane.get_selected_tab();
        let last_title = tabbed_pane.get_title_at(self.idx_last_tab.get() as usize);
        self.synchronize(last_title.as_deref().unwrap_or(""), true);
        let new_title = tabbed_pane.get_title_at(idx_new_tab as usize);
        self.set_visible(new_title.as_deref().unwrap_or(""));
        // Set the last tab index to current tab so that we are ready for tab
        // change
        self.idx_last_tab.set(tabbed_pane.get_selected_tab());
        if let Some(this) = self.as_process_interface() {
            self.mediator
                .set_method_process_interface_processing_method(
                    &this,
                    self.get_processing_method(),
                );
        }
    }

    /// Java package-private `setVisible(String)`.
    pub fn set_visible(&self, show_tab_title: &str) {
        if show_tab_title == LBL_SETUP {
            self.pnl_setup().set_visible(true);
            self.pnl_initial().set_visible(false);
            self.pnl_final().set_visible(false);
        } else if show_tab_title == LBL_INITIAL {
            self.pnl_setup().set_visible(false);
            self.pnl_initial().set_visible(true);
            self.pnl_final().set_visible(false);
        } else if show_tab_title == LBL_FINAL {
            self.pnl_setup().set_visible(false);
            self.pnl_initial().set_visible(false);
            self.pnl_final().set_visible(true);
        }
        let manager: &'static dyn BaseManager = self.base.application_manager;
        ui_harness::with(|harness| {
            harness.pack_axis_id_base_manager(Some(AxisID::Only), Some(manager));
            harness.move_sub_frame();
        });
    }
}

impl ProcessDialogVirtual for TomogramCombinationDialog {
    fn process_dialog(&self) -> &ProcessDialog {
        &self.base
    }

    /// Java public override `done()`.
    fn done(&self) {
        self.base
            .application_manager
            .done_tomogram_combination_dialog();
        self.base.set_displayed(false);
        if let Some(this) = self.as_process_interface() {
            if let Some(mediator) = self
                .base
                .application_manager
                .get_processing_method_mediator(Some(self.base.axis_id))
            {
                mediator.deregister_process_interface(&this);
            }
        }
    }
}

impl AbstractParallelDialog for TomogramCombinationDialog {
    /// Java inherited `ProcessDialog.getParameters(ParallelParam)`: empty.
    fn get_parameters(&self, param: &mut dyn ParallelParam) {
        ProcessDialogVirtual::get_parameters(self, param);
    }

    /// Java inherited `ProcessDialog.getDialogType()`.
    fn get_dialog_type(&self) -> DialogType {
        self.base.dialog_type
    }
}

impl ContextMenu for TomogramCombinationDialog {
    /// Java public override `popUpContextMenu(MouseEvent)`.  Right mouse
    /// button context menu.
    fn pop_up_context_menu(&self, mouse_event: &MouseEvent) {
        let man_pagelabel = [
            "Solvematch".to_string(),
            "Matchshifts".to_string(),
            "Patchcrawl3d".to_string(),
            "Matchorwarp".to_string(),
        ];
        let man_page = [
            "solvematch.html".to_string(),
            "matchshifts.html".to_string(),
            "patchcrawl3d.html".to_string(),
            "matchorwarp.html".to_string(),
        ];
        let log_file_label = [
            "Solvematch.log".to_string(),
            "Patchcorr.log".to_string(),
            "Matchorwarp.log".to_string(),
        ];
        let manager: &'static dyn BaseManager = self.base.application_manager;
        // Java passes logFileLabel as both the labels and the file names (kept:
        // odd, but the labels are file names).  The constructor's
        // IllegalArgumentException (mismatched arrays) cannot occur.
        let _context_popup = ContextPopup::new_component_mouse_event_string_string_string_array_string_array_string_array_string_array_base_manager_axis_id(
            &self.base.root_panel.get_component(),
            mouse_event,
            Some("TOMOGRAM COMBINATION"),
            Some(context_popup::TOMO_GUIDE),
            &man_pagelabel,
            &man_page,
            Some(&log_file_label),
            Some(&log_file_label),
            manager,
            self.base.axis_id,
        );
    }
}

impl QueueTableListener for TomogramCombinationDialog {
    /// Java inherited `ProcessDialog.queueTableEventAction(QueueTableEvent)`:
    /// empty.
    fn queue_table_event_action(&self, event: &QueueTableEvent) {
        self.base.queue_table_event_action(event);
    }
}

impl ProcessInterface for TomogramCombinationDialog {
    /// Java public override `updateGpu(boolean)`: empty.
    fn update_gpu(&self, _disable: bool) {}

    /// Java public override `getProcessingMethod()`.
    fn get_processing_method(&self) -> ProcessingMethod {
        let tabbed_pane = self.tabbed_pane.get_component();
        let tab_title = tabbed_pane.get_title_at(tabbed_pane.get_selected_tab() as usize);
        let tab_title = tab_title.as_deref();
        if tab_title == Some(LBL_SETUP) {
            return self.pnl_setup().get_processing_method();
        }
        // Upstream bug fixed in translation (TomogramCombinationDialog.java:689):
        // Java writes `tabTitle.equals(pnlInitial)`, comparing the title with the
        // panel object, which is never true (a slip for lblInitial).  We compare
        // with LBL_INITIAL.  The result is the same either way:
        // InitialCombinePanel.getProcessingMethod() is LOCAL_CPU, which is also
        // what the fall-through returns.
        if tab_title == Some(LBL_INITIAL) {
            return self.pnl_initial().get_processing_method();
        }
        if tab_title == Some(LBL_FINAL) {
            return self.pnl_final().get_processing_method();
        }
        ProcessingMethod::LocalCpu
    }

    /// Java public override `getSecondaryProcessingMethod()`.
    fn get_secondary_processing_method(&self) -> Option<ProcessingMethod> {
        None
    }

    /// Java public override `lockProcessingMethod(boolean)`.
    fn lock_processing_method(&self, lock: bool) {
        self.pnl_setup().lock_processing_method(lock);
        FinalCombineFields::set_parallel_enabled(
            &**self.pnl_final(),
            FinalCombineFields::is_parallel_enabled(&**self.pnl_setup()),
        );
    }

    /// Java public override `setMethod(ProcessingMethod)`.
    fn set_method(&self, processing_method: ProcessingMethod) {
        // `if (mediator != null)`: the mediator is never null here.
        if let Some(this) = self.as_process_interface() {
            self.mediator
                .set_method_process_interface_processing_method(&this, processing_method);
        }
    }

    /// Java public override `isUseGpu()`.
    fn is_use_gpu(&self) -> bool {
        // TODO Auto-generated method stub
        false
    }

    /// Java public override `setUseQueueCheckBox(ButtonComponent)`: empty.
    fn set_use_queue_check_box(&self, _use_queue_check_box: Option<Rc<dyn ButtonComponent>>) {}

    /// Java inherited `ProcessDialog.addQueueTableListener(QueueTableListener)`.
    fn add_queue_table_listener(&self, listener: Rc<dyn QueueTableListener>) {
        self.base.add_queue_table_listener(listener);
    }

    /// Java inherited
    /// `ProcessDialog.removeQueueTableListener(QueueTableListener)`.
    fn remove_queue_table_listener(&self, listener: &Rc<dyn QueueTableListener>) {
        self.base.remove_queue_table_listener(listener);
    }
}
