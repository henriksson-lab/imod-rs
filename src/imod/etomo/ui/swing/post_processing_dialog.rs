//! `IMOD/Etomo/src/etomo/ui/swing/PostProcessingDialog.java`.
//!
//! Java `public final class PostProcessingDialog extends ProcessDialog
//! implements ContextMenu, ProcessInterface`: the Post Processing dialog -
//! a tabbed pane with the Trim vol, Flatten, Reduce/filt vol, Alt Stack and
//! (single axis, non-montage datasets) Subtomograms tabs.  Only the selected
//! tab's root panel holds its panel; `changeTab` moves it.
//!
//! An EDT object (`ui.md`): created as `Rc<Self>` by
//! [`PostProcessingDialog::get_instance`]; every method takes `&self`.  The
//! `ProcessDialog` superclass is the embedded `base` (reached through
//! `Deref`), and the overridden `done()` is `ProcessDialogVirtual::done`.
//! The listener class `TabChangeListener` is a closure holding a weak
//! reference to the dialog; the static inner class `Tab` is the enum [`Tab`].
//!
//! **Construction order.**  The Java constructor hands `this` (as the
//! `ProcessInterface`) to `SubtomogramsPanel` and `AltStackPanel`, and
//! `AltStackPanel`'s constructor registers it with the mediator.  So the
//! dialog is put in its `Rc` first (with the `ProcessDialog` part and the
//! field-initialised members), and the constructor-body members are then
//! created in the Java order and stored in `OnceCell`s; after `get_instance`
//! returns they are always set (Java `final`).

use std::cell::{Cell, OnceCell};
use std::ops::Deref;
use std::rc::{Rc, Weak};

use super::alt_stack_display::AltStackDisplay;
use super::alt_stack_panel::AltStackPanel;
use super::beveled_border::BeveledBorder;
use super::button_component::ButtonComponent;
use super::context_menu::ContextMenu;
use super::flatten_volume_panel::FlattenVolumePanel;
use super::flatten_warp_display::FlattenWarpDisplay;
use super::process_dialog::{ProcessDialog, ProcessDialogVirtual};
use super::process_interface::ProcessInterface;
use super::reduce_filt_vol_display::ReduceFiltVolDisplay;
use super::squeeze_vol_panel::SqueezeVolPanel;
use super::subtomo_setup_display::SubtomoSetupDisplay;
use super::subtomograms_panel::SubtomogramsPanel;
use super::tabbed_pane::TabbedPane;
use super::tool_panel::ToolPanel;
use super::trimvol_display::TrimvolDisplay;
use super::trimvol_panel::TrimvolPanel;
use super::ui_harness;
use super::warp_vol_display::WarpVolDisplay;
use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::const_squeezevol_param::ConstSqueezevolParam;
use crate::imod::etomo::comscript::const_warp_vol_param::ConstWarpVolParam;
use crate::imod::etomo::comscript::makecomfile_param::MakecomfileParam;
use crate::imod::etomo::comscript::reduce_filt_vol_param::ReduceFiltVolParam;
use crate::imod::etomo::comscript::subtomo_setup_param::SubtomoSetupParam;
use crate::imod::etomo::comscript::tilt_param::TiltParam;
use crate::imod::etomo::comscript::trimvol_param::TrimvolParam;
use crate::imod::etomo::comscript::warp_vol_param::WarpVolParam;
use crate::imod::etomo::jdk::{ChangeEvent, JComponent, MouseEvent};
use crate::imod::etomo::logic::trimvol_input_file_state::TrimvolInputFileState;
use crate::imod::etomo::process::base_process_manager::BaseProcessManager;
use crate::imod::etomo::processing_method_mediator::ProcessingMethodMediator;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::const_etomo_number::ConstEtomoNumber;
use crate::imod::etomo::r#type::const_meta_data::ConstMetaData;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::file_type;
use crate::imod::etomo::r#type::meta_data::MetaData;
use crate::imod::etomo::r#type::processing_method::ProcessingMethod;
use crate::imod::etomo::r#type::recon_screen_state::ReconScreenState;
use crate::imod::etomo::r#type::view_type::ViewType;
use crate::imod::etomo::ui::queue_table_event::QueueTableEvent;
use crate::imod::etomo::ui::queue_table_listener::QueueTableListener;
use crate::imod::etomo::util::utilities;

/// Java public static final `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java package-private static final inner class `Tab` (an index into the
/// tabbed pane).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Tab {
    /// Java private static final `TRIM_VOL = new Tab(0)`.
    TrimVol,
    /// Java private static final `FLATTEN = new Tab(1)`.
    Flatten,
    /// Java private static final `SQUEEZE_VOL = new Tab(2)`.
    SqueezeVol,
    /// Java private static final `ALT_STACK = new Tab(3)`.
    AltStack,
    /// Java private static final `SUBTOMOGRAMS = new Tab(4)`.
    Subtomograms,
}

impl Tab {
    /// Java static final `DEFAULT = TRIM_VOL`.
    pub const DEFAULT: Tab = Tab::TrimVol;

    /// Java private final field `index`.
    fn index(self) -> i32 {
        match self {
            Tab::TrimVol => 0,
            Tab::Flatten => 1,
            Tab::SqueezeVol => 2,
            Tab::AltStack => 3,
            Tab::Subtomograms => 4,
        }
    }

    /// Java private static `getInstance(int)`.
    fn get_instance(index: i32) -> Tab {
        if index == Tab::TrimVol.index() {
            return Tab::TrimVol;
        }
        if index == Tab::Flatten.index() {
            return Tab::Flatten;
        }
        if index == Tab::SqueezeVol.index() {
            return Tab::SqueezeVol;
        }
        if index == Tab::Subtomograms.index() {
            return Tab::Subtomograms;
        }
        if index == Tab::AltStack.index() {
            return Tab::AltStack;
        }
        Tab::DEFAULT
    }

    /// Java private `toInt()`.
    fn to_int(self) -> i32 {
        self.index()
    }
}

/// Java `public final class PostProcessingDialog extends ProcessDialog
/// implements ContextMenu, ProcessInterface`.
pub struct PostProcessingDialog {
    /// The `ProcessDialog` superclass.
    base: Rc<ProcessDialog>,
    /// Rust-only: Java `this` (the mediator's ProcessInterface).
    self_ref: Weak<PostProcessingDialog>,

    /// Java private final `trimvolPanel`.
    trimvol_panel: OnceCell<Rc<TrimvolPanel>>,

    /// Java private final `tabbedPane = new TabbedPane()`.
    tabbed_pane: Rc<TabbedPane>,

    /// Java private `curTab = Tab.DEFAULT`.
    cur_tab: Cell<Tab>,
    /// Java private final `flattenVolumePanel`.
    flatten_volume_panel: OnceCell<Rc<FlattenVolumePanel>>,
    /// Java private final `squeezeVolPanel`.
    squeeze_vol_panel: OnceCell<Rc<SqueezeVolPanel>>,
    /// Java private final `subtomogramsPanel`; the inner `None` is Java null
    /// (dual axis or montage datasets).
    subtomograms_panel: OnceCell<Option<Rc<SubtomogramsPanel>>>,
    /// Java private final `altStackPanel`.
    alt_stack_panel: OnceCell<Rc<AltStackPanel>>,
    /// Java private final `mediator` (null until the end of the constructor).
    mediator: OnceCell<Rc<ProcessingMethodMediator>>,
    /// Java package-private `openAltStackFirstTime = true`.
    open_alt_stack_first_time: Cell<bool>,
}

impl Deref for PostProcessingDialog {
    type Target = ProcessDialog;
    fn deref(&self) -> &ProcessDialog {
        &self.base
    }
}

impl PostProcessingDialog {
    /// Java private constructor `PostProcessingDialog(ApplicationManager,
    /// boolean)`.
    fn new(
        app_mgr: &'static ApplicationManager,
        trimvol_input_file_missing: bool,
    ) -> Rc<PostProcessingDialog> {
        // super(appMgr, AxisID.ONLY, DialogType.POST_PROCESSING)
        let base = ProcessDialog::new_application_manager_axis_id_dialog_type(
            app_mgr,
            AxisID::Only,
            DialogType::PostProcessing,
        );
        // Field initializers.
        let instance =
            Rc::new_cyclic(
                |self_ref: &Weak<PostProcessingDialog>| PostProcessingDialog {
                    base,
                    self_ref: self_ref.clone(),
                    trimvol_panel: OnceCell::new(),
                    tabbed_pane: TabbedPane::new(),
                    cur_tab: Cell::new(Tab::DEFAULT),
                    flatten_volume_panel: OnceCell::new(),
                    squeeze_vol_panel: OnceCell::new(),
                    subtomograms_panel: OnceCell::new(),
                    alt_stack_panel: OnceCell::new(),
                    mediator: OnceCell::new(),
                    open_alt_stack_first_time: Cell::new(true),
                },
            );
        // Java `this` as the ProcessDialog subclass (for the virtual `done()`).
        let this: Weak<dyn ProcessDialogVirtual> =
            Rc::downgrade(&instance) as Weak<dyn ProcessDialogVirtual>;
        instance.base.set_this(this);
        let process_interface: Weak<dyn ProcessInterface> =
            Rc::downgrade(&instance) as Weak<dyn ProcessInterface>;
        let axis_id = instance.base.axis_id;
        let dialog_type = instance.base.dialog_type;
        // Constructor body.
        let flatten_volume_panel =
            FlattenVolumePanel::get_post_instance(app_mgr, axis_id, dialog_type);
        let _ = instance.flatten_volume_panel.set(flatten_volume_panel);
        let squeeze_vol_panel = SqueezeVolPanel::get_instance(app_mgr, axis_id, dialog_type);
        let _ = instance.squeeze_vol_panel.set(squeeze_vol_panel);
        if app_mgr.is_dual_axis() || app_mgr.get_view_type() == ViewType::Montage {
            let _ = instance.subtomograms_panel.set(None);
        } else {
            let subtomograms_panel = SubtomogramsPanel::get_instance(
                app_mgr,
                axis_id,
                dialog_type,
                process_interface.clone(),
                &instance.base.btn_advanced,
            );
            subtomograms_panel.update_advanced(instance.base.is_advanced());
            let _ = instance.subtomograms_panel.set(Some(subtomograms_panel));
        }
        let alt_stack_panel =
            AltStackPanel::get_instance(app_mgr, axis_id, dialog_type, process_interface);
        let _ = instance.alt_stack_panel.set(alt_stack_panel);
        // Swing layout: rootPanel.setLayout(new BoxLayout(rootPanel,
        // BoxLayout.Y_AXIS)).
        instance
            .base
            .root_panel
            .set_border(&BeveledBorder::new(Some("Post Processing")).get_border());
        let root_panel = instance.base.root_panel.get_component();
        root_panel.add(&instance.tabbed_pane.get_component());
        let trimvol_panel = TrimvolPanel::new(
            instance.base.application_manager,
            axis_id,
            dialog_type,
            trimvol_input_file_missing,
        );
        let _ = instance.trimvol_panel.set(trimvol_panel);
        let trimvol_root = JComponent::new_panel();
        instance
            .tabbed_pane
            .add_tab_string_component("Trim vol", &trimvol_root);
        trimvol_root.add(&instance.trimvol_panel().get_container());
        let flatten_root = JComponent::new_panel();
        instance
            .tabbed_pane
            .add_tab_string_component("Flatten", &flatten_root);
        let squeeze_root = JComponent::new_panel();
        instance
            .tabbed_pane
            .add_tab_string_component("Reduce/filt vol", &squeeze_root);
        let alt_stack_root = JComponent::new_panel();
        instance
            .tabbed_pane
            .add_tab_string_component("Alt Stack", &alt_stack_root);
        if instance.subtomograms_panel().is_some() {
            let subtomograms_root = JComponent::new_panel();
            instance
                .tabbed_pane
                .add_tab_string_component("Subtomograms", &subtomograms_root);
        }
        instance.base.add_exit_buttons();
        instance.base.btn_execute.set_text(Some("Done"));
        // The mediator exists on the event dispatch thread, where dialogs are built.
        if let Some(mediator) = app_mgr.get_processing_method_mediator(Some(axis_id)) {
            let _ = instance.mediator.set(mediator.clone());
            let this_process_interface: Rc<dyn ProcessInterface> = instance.clone();
            mediator.register_process_interface(this_process_interface.clone());
            mediator.set_method_process_interface_processing_method(
                &this_process_interface,
                ProcessInterface::get_processing_method(&*instance),
            );
        }
        instance
    }

    /// Java public static `getInstance(ApplicationManager, boolean)`.
    pub fn get_instance(
        manager: &'static ApplicationManager,
        trimvol_input_file_missing: bool,
    ) -> Rc<PostProcessingDialog> {
        let instance = PostProcessingDialog::new(manager, trimvol_input_file_missing);
        instance.add_listeners();
        instance
            .tabbed_pane
            .get_component()
            .set_selected_tab(Tab::DEFAULT.to_int());
        instance
    }

    /// Rust-only: the Java `final` field `trimvolPanel`.
    fn trimvol_panel(&self) -> &Rc<TrimvolPanel> {
        self.trimvol_panel.get().expect("set by the constructor")
    }

    /// Rust-only: the Java `final` field `flattenVolumePanel`.
    fn flatten_volume_panel(&self) -> &Rc<FlattenVolumePanel> {
        self.flatten_volume_panel
            .get()
            .expect("set by the constructor")
    }

    /// Rust-only: the Java `final` field `squeezeVolPanel`.
    fn squeeze_vol_panel(&self) -> &Rc<SqueezeVolPanel> {
        self.squeeze_vol_panel
            .get()
            .expect("set by the constructor")
    }

    /// Rust-only: the Java `final` field `subtomogramsPanel`; `None` is Java
    /// null.
    fn subtomograms_panel(&self) -> Option<&Rc<SubtomogramsPanel>> {
        self.subtomograms_panel.get().and_then(Option::as_ref)
    }

    /// Rust-only: the Java `final` field `altStackPanel`.
    fn alt_stack_panel(&self) -> &Rc<AltStackPanel> {
        self.alt_stack_panel.get().expect("set by the constructor")
    }

    /// Java private `addListeners()`.
    fn add_listeners(self: &Rc<Self>) {
        // Mouse adapter for context menu
        // Swing mouse: rootPanel.addMouseListener(mouseAdapter) and
        // tabbedPane.addMouseListener(mouseAdapter) with `new
        // GenericMouseAdapter(this)`.  Mouse events are not modelled; the
        // adapter's only effect is to call popUpContextMenu on a right-button
        // press, which a driver calls directly.
        // Java `new TabChangeListener(this)`.
        let adaptee = Rc::downgrade(self);
        self.tabbed_pane
            .get_component()
            .add_change_listener(Rc::new(move |_event: &ChangeEvent| {
                if let Some(adaptee) = adaptee.upgrade() {
                    adaptee.change_tab_void();
                }
            }));
    }

    /// Java private `changeTab(ConstEtomoNumber)`.
    fn change_tab_const_etomo_number(&self, index: Option<&ConstEtomoNumber>) {
        let Some(index) = index else {
            return;
        };
        if index.is_null() {
            return;
        }
        self.tabbed_pane
            .get_component()
            .set_selected_tab(index.get_int());
        self.change_tab_void();
    }

    /// Java private `changeTab()`.
    fn change_tab_void(&self) {
        let tabbed_pane = self.tabbed_pane.get_component();
        if let Some(cur) = tabbed_pane.get_component_at(self.cur_tab.get().to_int() as usize) {
            cur.remove_all();
        }
        self.cur_tab
            .set(Tab::get_instance(tabbed_pane.get_selected_tab()));
        // `tabbedPane.getSelectedComponent()`.
        let panel = tabbed_pane.get_component_at(tabbed_pane.get_selected_tab() as usize);
        if let Some(panel) = &panel {
            let cur_tab = self.cur_tab.get();
            if cur_tab == Tab::TrimVol {
                panel.add(&self.trimvol_panel().get_container());
            } else if cur_tab == Tab::Flatten {
                panel.add(&ToolPanel::get_component(&**self.flatten_volume_panel()));
            } else if cur_tab == Tab::SqueezeVol {
                panel.add(&self.squeeze_vol_panel().get_component());
            } else if cur_tab == Tab::AltStack {
                if self.open_alt_stack_first_time.get() {
                    self.create_alt_tomo_setup_com_file();
                }
                panel.add(&self.alt_stack_panel().get_component());
                self.alt_stack_panel().check_if_files_exist();
            } else if cur_tab == Tab::Subtomograms {
                // The Subtomograms tab only exists when the panel does.
                if let Some(subtomograms_panel) = self.subtomograms_panel() {
                    panel.add(&subtomograms_panel.get_component());
                }
                if let Some(mediator) = self.mediator.get() {
                    mediator.add_queue_listener_on_switch_dialog();
                }
            }
        }

        if let (Some(mediator), Some(this)) = (self.mediator.get(), self.self_ref.upgrade()) {
            let this: Rc<dyn ProcessInterface> = this;
            mediator.set_method_process_interface_processing_method(
                &this,
                ProcessInterface::get_processing_method(self),
            );
        }
        let manager: &'static dyn BaseManager = self.base.application_manager;
        let axis_id = self.base.axis_id;
        ui_harness::with(|harness| {
            harness.pack_axis_id_base_manager(Some(axis_id), Some(manager));
            harness.move_sub_frame();
        });
    }

    /// Java public `setParameters(ConstSqueezevolParam)`.  Set the panel values
    /// with the specified parameters.
    pub fn set_parameters_const_squeezevol_param(
        &self,
        squeezevol_param: &dyn ConstSqueezevolParam,
    ) {
        self.squeeze_vol_panel()
            .set_parameters_const_squeezevol_param(squeezevol_param);
    }

    /// Java public `setParameters(ReduceFiltVolParam, boolean, boolean)`.
    pub fn set_parameters_reduce_filt_vol_param_boolean_boolean(
        &self,
        reduce_filt_vol_param: &ReduceFiltVolParam,
        dialog_not_exists: bool,
        com_file_exists: bool,
    ) {
        self.squeeze_vol_panel()
            .set_parameters_reduce_filt_vol_param_boolean_boolean(
                reduce_filt_vol_param,
                dialog_not_exists,
                com_file_exists,
            );
    }

    /// Java public `setParameters(ReconScreenState)`.
    pub fn set_parameters_recon_screen_state(&self, screen_state: &ReconScreenState) {
        self.trimvol_panel()
            .set_parameters_recon_screen_state(screen_state);
        self.squeeze_vol_panel()
            .set_parameters_recon_screen_state(screen_state);
    }

    /// Java public `setParameters(SubtomoSetupParam)`.
    pub fn set_parameters_subtomo_setup_param(&self, param: &SubtomoSetupParam) {
        // Upstream bug fixed in translation (PostProcessingDialog.java:178): Java
        // dereferences a null `subtomogramsPanel` (dual axis or montage) and
        // throws NullPointerException.  The manager only calls this for the
        // datasets that have the panel; with no panel the call does nothing.
        if let Some(subtomograms_panel) = self.subtomograms_panel() {
            subtomograms_panel.set_parameters_subtomo_setup_param(param);
        }
    }

    /// Java public `setParameters(TiltParam, boolean)`.
    pub fn set_parameters_tilt_param_boolean(&self, param: &TiltParam, initialize: bool) {
        self.alt_stack_panel()
            .set_parameters_const_tilt_param_boolean(param, initialize);
    }

    /// Java public `initParameters(TrimvolParam)`.  The param is `&mut`
    /// because `RubberbandPanel.initScaleParameters` reads it through
    /// `TrimvolParam.getScaleXYParam()`, which hands out the mutable member.
    pub fn init_parameters(&self, param: &mut TrimvolParam) {
        self.trimvol_panel().init_parameters(param);
    }

    /// Java public `setParameters(ConstMetaData, boolean)`.
    pub fn set_parameters_const_meta_data_boolean(
        &self,
        meta_data: &dyn ConstMetaData,
        dialog_exists: bool,
    ) {
        self.trimvol_panel()
            .set_parameters_const_meta_data_boolean(meta_data, dialog_exists);
        self.flatten_volume_panel()
            .set_parameters_const_meta_data(meta_data);
        self.squeeze_vol_panel()
            .set_parameters_const_meta_data(meta_data);
        if let Some(subtomograms_panel) = self.subtomograms_panel() {
            subtomograms_panel.set_parameters_const_meta_data(meta_data);
        }
        self.alt_stack_panel()
            .set_parameters_const_meta_data(meta_data);
        let post_cur_tab = meta_data.get_post_cur_tab();
        self.change_tab_const_etomo_number(Some(&*post_cur_tab));
    }

    /// Java public `getFlattenWarpDisplay()`.
    pub fn get_flatten_warp_display(&self) -> Rc<dyn FlattenWarpDisplay> {
        self.flatten_volume_panel().get_flatten_warp_display()
    }

    /// Java public `getTrimvolDisplay()`.
    pub fn get_trimvol_display(&self) -> Rc<dyn TrimvolDisplay> {
        self.trimvol_panel().clone()
    }

    /// Java public `getSubtomoSetupDisplay()`: `subtomogramsPanel
    /// .getSubtomoSetupDisplay()`, which is the panel itself.  `None` where
    /// Java would dereference a null `subtomogramsPanel` (dual axis or
    /// montage; the manager does not call it then).
    pub fn get_subtomo_setup_display(&self) -> Option<Rc<dyn SubtomoSetupDisplay>> {
        self.subtomograms_panel()
            .map(|subtomograms_panel| subtomograms_panel.clone() as Rc<dyn SubtomoSetupDisplay>)
    }

    /// Java public `getAltStackDisplay()`.
    pub fn get_alt_stack_display(&self) -> Rc<dyn AltStackDisplay> {
        self.alt_stack_panel().get_alt_stack_display()
    }

    /// Java public `getReduceFiltVolDisplay()`.
    pub fn get_reduce_filt_vol_display(&self) -> Rc<dyn ReduceFiltVolDisplay> {
        self.squeeze_vol_panel().get_reduce_filt_vol_display()
    }

    /// Java public `setStartupWarnings(TrimvolInputFileState)`.
    pub fn set_startup_warnings(&self, input_file_state: &TrimvolInputFileState) -> bool {
        self.trimvol_panel().set_startup_warnings(input_file_state)
    }

    /// Java public `getParameters(MetaData)`.
    pub fn get_parameters_meta_data(&self, meta_data: &MetaData) {
        self.trimvol_panel().get_parameters_meta_data(meta_data);
        self.flatten_volume_panel()
            .get_parameters_meta_data(meta_data);
        self.squeeze_vol_panel().get_parameters_meta_data(meta_data);
        if let Some(subtomograms_panel) = self.subtomograms_panel() {
            subtomograms_panel.get_parameters_meta_data(meta_data);
        }
        if let Err(e) = self.alt_stack_panel().get_parameters_meta_data(meta_data) {
            // TODO Auto-generated catch block
            // e.printStackTrace();
            eprintln!("{e:?}");
        }
        meta_data.set_post_cur_tab(self.cur_tab.get().index());
    }

    /// Java public `getParametersForTrimvol(MetaData)`.
    pub fn get_parameters_for_trimvol(&self, meta_data: &MetaData) {
        self.trimvol_panel().get_parameters_for_trimvol(meta_data);
    }

    /// Java public `getParameters(WarpVolParam, boolean)`.
    pub fn get_parameters_warp_vol_param_boolean(
        &self,
        param: &mut WarpVolParam,
        do_validation: bool,
    ) -> bool {
        WarpVolDisplay::get_parameters(&**self.flatten_volume_panel(), param, do_validation)
    }

    /// Java public `setParameters(ConstWarpVolParam)`.
    pub fn set_parameters_const_warp_vol_param(&self, param: &dyn ConstWarpVolParam) {
        self.flatten_volume_panel()
            .set_parameters_const_warp_vol_param(param);
    }

    /// Java public `getParameters(MakecomfileParam, boolean)`.
    pub fn get_parameters_makecomfile_param_boolean(
        &self,
        make_com_file_param: &mut MakecomfileParam,
        do_validation: bool,
    ) -> bool {
        self.squeeze_vol_panel()
            .get_parameters_makecomfile_param_boolean(make_com_file_param, do_validation)
    }

    /// Java public `getParameters(TrimvolParam, boolean)`.  Get the trimvol
    /// parameter values from the panel.
    pub fn get_parameters_trimvol_param_boolean(
        &self,
        trimvol_param: &mut TrimvolParam,
        do_validation: bool,
    ) -> bool {
        self.trimvol_panel()
            .get_parameters_trimvol_param_boolean(trimvol_param, do_validation)
    }

    /// Java private `createAltTomoSetupComFile()`.
    fn create_alt_tomo_setup_com_file(&self) {
        let application_manager = self.base.application_manager;
        let manager: &'static dyn BaseManager = application_manager;
        let loaded = application_manager
            .get_com_script_manager()
            .load_alt_tomo_setup(AxisID::Only, false);
        if !loaded {
            // `FileType.ALT_TOMO_SETUP_COMSCRIPT.getFile(applicationManager, null,
            // AxisType.SINGLE_AXIS, AxisID.ONLY, propertyUserDir)
            // .getAbsolutePath()`; FileType builds the file for this fixed
            // name, so the result is not null.
            let file = file_type::CLASS
                .alt_tomo_setup_comscript
                .get_file_with_property_user_dir(
                    Some(manager),
                    None,
                    Some(AxisType::SingleAxis),
                    Some(AxisID::Only),
                    application_manager.get_property_user_dir().as_deref(),
                )
                .unwrap_or_default();
            BaseProcessManager::touch(
                &utilities::java_io_file_get_absolute_path(&file.to_string_lossy()),
                Some(manager),
            );
            application_manager
                .get_com_script_manager()
                .load_alt_tomo_setup(AxisID::Only, true);
        }
        let alt_tomo_setup_param = application_manager
            .get_com_script_manager()
            .get_alt_tomo_setup_param(AxisID::Only);
        self.alt_stack_panel()
            .set_parameters_alt_tomo_setup_param(&alt_tomo_setup_param);
        self.open_alt_stack_first_time.set(false);
    }
}

impl ContextMenu for PostProcessingDialog {
    /// Java public override `popUpContextMenu(MouseEvent)`.  Right mouse button
    /// context menu; empty.
    fn pop_up_context_menu(&self, _mouse_event: &MouseEvent) {}
}

impl ProcessDialogVirtual for PostProcessingDialog {
    fn process_dialog(&self) -> &ProcessDialog {
        &self.base
    }

    /// Java package-private override `done()`.
    fn done(&self) {
        self.base.application_manager.done_post_processing();
        self.squeeze_vol_panel().done();
        self.trimvol_panel().done();
        self.flatten_volume_panel().done();
        self.base.set_displayed(false);
    }
}

impl QueueTableListener for PostProcessingDialog {
    /// Java `queueTableEventAction(QueueTableEvent)`, inherited from
    /// `ProcessDialog` (empty).
    fn queue_table_event_action(&self, event: &QueueTableEvent) {
        self.base.queue_table_event_action(event);
    }
}

impl ProcessInterface for PostProcessingDialog {
    /// Java public override `updateGpu(boolean)`.
    fn update_gpu(&self, disable: bool) {
        if self.cur_tab.get() == Tab::Subtomograms {
            if let Some(subtomograms_panel) = self.subtomograms_panel() {
                subtomograms_panel.update_gpu(disable);
            }
        } else if self.cur_tab.get() == Tab::AltStack {
            ProcessInterface::update_gpu(&**self.alt_stack_panel(), disable);
        }
    }

    /// Java public override `getProcessingMethod()`.
    fn get_processing_method(&self) -> ProcessingMethod {
        if self.cur_tab.get() == Tab::Subtomograms {
            // curTab is SUBTOMOGRAMS only when the tab (and so the panel)
            // exists.
            if let Some(subtomograms_panel) = self.subtomograms_panel() {
                return subtomograms_panel.get_processing_method();
            }
        } else if self.cur_tab.get() == Tab::AltStack {
            return ProcessInterface::get_processing_method(&**self.alt_stack_panel());
        }
        ProcessingMethod::LocalCpu
    }

    /// Java public override `getSecondaryProcessingMethod()`; returns null.
    fn get_secondary_processing_method(&self) -> Option<ProcessingMethod> {
        // TODO Auto-generated method stub
        None
    }

    /// Java public override `lockProcessingMethod(boolean)`; empty.
    fn lock_processing_method(&self, _lock: bool) {
        // TODO Auto-generated method stub
    }

    /// Java public override `setMethod(ProcessingMethod)`.
    fn set_method(&self, processing_method: ProcessingMethod) {
        if let (Some(mediator), Some(this)) = (self.mediator.get(), self.self_ref.upgrade()) {
            let this: Rc<dyn ProcessInterface> = this;
            mediator.set_method_process_interface_processing_method(&this, processing_method);
        }
    }

    /// Java public override `isUseGpu()`.
    fn is_use_gpu(&self) -> bool {
        if self.cur_tab.get() == Tab::Subtomograms {
            if let Some(subtomograms_panel) = self.subtomograms_panel() {
                return subtomograms_panel.is_use_gpu();
            }
        }
        false
    }

    /// Java public override `setUseQueueCheckBox(ButtonComponent)`.
    fn set_use_queue_check_box(&self, use_queue_check_box: Option<Rc<dyn ButtonComponent>>) {
        if self.cur_tab.get() == Tab::Subtomograms {
            if let Some(subtomograms_panel) = self.subtomograms_panel() {
                subtomograms_panel.set_use_queue_check_box(use_queue_check_box);
            }
        }
    }

    /// Java `addQueueTableListener(QueueTableListener)`, inherited from
    /// `ProcessDialog` (empty).
    fn add_queue_table_listener(&self, listener: Rc<dyn QueueTableListener>) {
        self.base.add_queue_table_listener(listener);
    }

    /// Java `removeQueueTableListener(QueueTableListener)`, inherited from
    /// `ProcessDialog` (empty).
    fn remove_queue_table_listener(&self, listener: &Rc<dyn QueueTableListener>) {
        self.base.remove_queue_table_listener(listener);
    }
}
