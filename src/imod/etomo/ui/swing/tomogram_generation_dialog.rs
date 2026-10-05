//! `IMOD/Etomo/src/etomo/ui/swing/TomogramGenerationDialog.java`.
//!
//! Java `public class TomogramGenerationDialog extends ProcessDialog
//! implements ContextMenu, TomogramGenerationParent` (Tomogram generation
//! user interface): the method radio buttons (Back Projection, Filter Trials,
//! SIRT, 3D CTF) and the tilt, multifilt, SIRT and 3D CTF panels.
//!
//! An EDT object created as `Rc<Self>` by
//! [`TomogramGenerationDialog::get_instance`]; every method takes `&self`.
//! The `ProcessDialog` superclass is the embedded `base` (reached through
//! `Deref`), and the overridden `done()` is `ProcessDialogVirtual::done`.
//! The listener class `TomogramGenerationActionListener` is a closure holding
//! a weak reference to the dialog.
//!
//! **Construction order.**  The Java constructor hands `this` to the panels
//! it builds, and `Ctf3dPanel`'s constructor calls back into it
//! (`parent.getAdvancedButton()`).  So the dialog is put in its `Rc` first
//! (with the `ProcessDialog` part and the field-initialised radio buttons),
//! and the constructor-body members (`tiltPanel`, `multifiltPanel`,
//! `sirtPanel`, `rbCtf3d`, `ctf3dPanel`, `tiltPanelRoot`) are then created in
//! the Java order and stored in `OnceCell`s; after `get_instance` returns
//! they are always set (Java `final`).
//!
//! The expert is held weakly: the expert owns the dialog (Java field
//! `TomogramGenerationExpert.dialog`), and the Java `expert` field is only
//! used to call back into it.
//!
//! **Method plugin.**  The expert passes its `TomoGenMethodPlugin` (null
//! unless one was loaded; `etomo -plugin` loads the built-in demo).  A plugin
//! with a custom tilt panel supplies a `TiltPanel` subclass, so `tiltPanel` is
//! held as a [`TiltPanelVirtual`]; its panel and radio button are
//! `methodPluginPanel` / `rbMethodPlugin`.

use super::recon_ui_expert::ReconUIExpertVirtual;
use std::cell::OnceCell;
use std::ops::Deref;
use std::rc::{Rc, Weak};

use super::beveled_border::BeveledBorder;
use super::context_menu::ContextMenu;
use super::context_popup::{self, ContextPopup};
use super::ctf3d_panel::Ctf3dPanel;
use super::ctf3d_setup_display::Ctf3dSetupDisplay;
use super::generic_mouse_adapter::GenericMouseAdapter;
use super::global_expand_button::GlobalExpandButton;
use super::multifilt_panel::MultifiltPanel;
use super::multifilt_setup_display::MultifiltSetupDisplay;
use super::process_dialog::{ProcessDialog, ProcessDialogVirtual};
use super::radio_button::RadioButton;
use super::sirt_panel::SirtPanel;
use super::sirtsetup_display::SirtsetupDisplay;
use super::tilt_display::TiltDisplay;
use super::tilt_panel::{TiltPanel, TiltPanelVirtual};
use super::tomogram_generation_expert::TomogramGenerationExpert;
use super::tomogram_generation_parent::TomogramGenerationParent;
use super::trial_tilt_parent::TrialTiltParent;
use super::ui_harness;
use super::ui_utilities;
use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::const_tilt_param::ConstTiltParam;
use crate::imod::etomo::comscript::ctf3d_setup_param::Ctf3dSetupParam;
use crate::imod::etomo::comscript::fortran_input_syntax_exception::FortranInputSyntaxException;
use crate::imod::etomo::comscript::multifilt_setup_param::MultifiltSetupParam;
use crate::imod::etomo::comscript::sirtsetup_param::SirtsetupParam;
use crate::imod::etomo::jdk::{
    ActionEvent, ActionListener, ButtonGroup, FocusListener, JComponent, MouseEvent, MouseListener,
};
use crate::imod::etomo::plugin::plugin_panel::PluginPanel;
use crate::imod::etomo::plugin::tomo_gen_method_plugin::TomoGenMethodPlugin;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_meta_data::ConstMetaData;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::file_type;
use crate::imod::etomo::r#type::meta_data::MetaData;
use crate::imod::etomo::r#type::processing_method::ProcessingMethod;
use crate::imod::etomo::r#type::recon_screen_state::ReconScreenState;
use crate::imod::etomo::r#type::tomogram_state::TomogramState;

/// Java public static final `X_AXIS_TILT_TOOLTIP`.
pub const X_AXIS_TILT_TOOLTIP: &str = "This line allows one to rotate the reconstruction \
     around the X axis, so that a section that appears to be tilted around the X axis can be \
     made flat to fit into a smaller volume.";

/// Java `public class TomogramGenerationDialog extends ProcessDialog
/// implements ContextMenu, TomogramGenerationParent`.
pub struct TomogramGenerationDialog {
    /// The `ProcessDialog` superclass.
    base: Rc<ProcessDialog>,

    /// Java private final `bgMethod = new ButtonGroup()`.
    bg_method: Rc<ButtonGroup>,
    /// Java private final `rbBackProjection`.
    rb_back_projection: Rc<RadioButton>,
    /// Java private final `rbMultifilt`.
    rb_multifilt: Rc<RadioButton>,
    /// Java private final `rbSirt`.
    rb_sirt: Rc<RadioButton>,

    /// Java private final `tiltPanel` (a `TiltPanel` or a plugin's subclass of
    /// it).
    tilt_panel: OnceCell<Rc<dyn TiltPanelVirtual>>,
    /// Java private final `expert` (held weakly; see the module docs).
    expert: Weak<TomogramGenerationExpert>,
    /// Java private final `multifiltPanel`.
    multifilt_panel: OnceCell<Rc<MultifiltPanel>>,
    /// Java private final `sirtPanel`.
    sirt_panel: OnceCell<Rc<SirtPanel>>,
    /// Java private final `tiltPanelRoot` (a `Component`).
    tilt_panel_root: OnceCell<Rc<JComponent>>,
    /// Java private final `rbMethodPlugin`; null without a method plugin panel.
    rb_method_plugin: OnceCell<Option<Rc<RadioButton>>>,
    /// Java private final `methodPluginPanel`; null without a method plugin.
    method_plugin_panel: OnceCell<Option<Rc<dyn PluginPanel>>>,
    /// Java private final `ctf3dPanel`.  Never null in the Java (created
    /// unconditionally), though the Java tests it.
    ctf3d_panel: OnceCell<Rc<Ctf3dPanel>>,
    /// Java private final `rbCtf3d`.  Never null in the Java (created
    /// unconditionally), though the Java tests it.
    rb_ctf3d: OnceCell<Rc<RadioButton>>,
}

impl Deref for TomogramGenerationDialog {
    type Target = ProcessDialog;
    fn deref(&self) -> &ProcessDialog {
        &self.base
    }
}

impl TomogramGenerationDialog {
    /// Java private constructor `TomogramGenerationDialog(ApplicationManager,
    /// TomogramGenerationExpert, AxisID, TomoGenMethodPlugin)`.
    fn new(
        app_mgr: &'static ApplicationManager,
        expert: Weak<TomogramGenerationExpert>,
        axis_id: AxisID,
        method_plugin: Option<&Rc<dyn TomoGenMethodPlugin>>,
    ) -> Rc<TomogramGenerationDialog> {
        // super(appMgr, axisID, DialogType.TOMOGRAM_GENERATION)
        let base = ProcessDialog::new_application_manager_axis_id_dialog_type(
            app_mgr,
            axis_id,
            DialogType::TomogramGeneration,
        );
        // Field initializers, in declaration order.
        let bg_method = ButtonGroup::new();
        let rb_back_projection =
            RadioButton::new_string_button_group(Some("Back Projection"), Some(&bg_method));
        let rb_multifilt =
            RadioButton::new_string_button_group(Some("Filter Trials"), Some(&bg_method));
        let rb_sirt = RadioButton::new_string_button_group(Some("SIRT"), Some(&bg_method));
        let instance = Rc::new(TomogramGenerationDialog {
            base,
            bg_method,
            rb_back_projection,
            rb_multifilt,
            rb_sirt,
            tilt_panel: OnceCell::new(),
            // this.expert = expert;
            expert,
            multifilt_panel: OnceCell::new(),
            sirt_panel: OnceCell::new(),
            tilt_panel_root: OnceCell::new(),
            rb_method_plugin: OnceCell::new(),
            method_plugin_panel: OnceCell::new(),
            ctf3d_panel: OnceCell::new(),
            rb_ctf3d: OnceCell::new(),
        });
        // Java `this` as the ProcessDialog subclass (for the virtual `done()`).
        let this: Weak<dyn ProcessDialogVirtual> =
            Rc::downgrade(&instance) as Weak<dyn ProcessDialogVirtual>;
        instance.base.set_this(this);
        // Constructor body.
        let parent: Weak<dyn TomogramGenerationParent> =
            Rc::downgrade(&instance) as Weak<dyn TomogramGenerationParent>;
        let dialog_type = instance.base.dialog_type;
        let mut tilt_panel: Option<Rc<dyn TiltPanelVirtual>> = None;
        if let Some(method_plugin) = method_plugin
            && method_plugin.has_custom_tilt_panel()
        {
            tilt_panel = method_plugin.get_tilt_panel(parent.clone(), &instance.base.btn_advanced);
        }
        // `methodPlugin == null || !methodPlugin.hasCustomTiltPanel()`.  A plugin
        // that claims a custom tilt panel and returns null leaves the Java field
        // null and every later use throws NullPointerException; fixed in
        // translation: the standard panel is used.
        let tilt_panel = tilt_panel.unwrap_or_else(|| {
            TiltPanel::get_instance(
                app_mgr,
                axis_id,
                dialog_type,
                &instance.base.btn_advanced,
                parent,
            ) as Rc<dyn TiltPanelVirtual>
        });
        let _ = instance.tilt_panel.set(tilt_panel);
        let multifilt_panel =
            MultifiltPanel::get_instance(app_mgr, axis_id, Rc::downgrade(&instance), dialog_type);
        let _ = instance.multifilt_panel.set(multifilt_panel);
        let sirt_panel = SirtPanel::get_instance(
            app_mgr,
            axis_id,
            dialog_type,
            &instance.base.btn_advanced,
            Rc::downgrade(&instance),
        );
        let _ = instance.sirt_panel.set(sirt_panel);
        let rb_ctf3d =
            RadioButton::new_string_button_group(Some("3D CTF"), Some(&instance.bg_method));
        let _ = instance.rb_ctf3d.set(rb_ctf3d);
        let ctf3d_panel = Ctf3dPanel::get_instance(app_mgr, axis_id, Rc::downgrade(&instance));
        let _ = instance.ctf3d_panel.set(ctf3d_panel);
        let method_plugin_panel = match method_plugin {
            Some(method_plugin) => {
                method_plugin.get_panel(Rc::downgrade(&instance), &instance.base.btn_advanced)
            }
            None => None,
        };
        let rb_method_plugin = method_plugin_panel.as_ref().map(|method_plugin_panel| {
            RadioButton::new_string_button_group(
                method_plugin_panel.get_button_title().as_deref(),
                Some(&instance.bg_method),
            )
        });
        let _ = instance.method_plugin_panel.set(method_plugin_panel);
        let _ = instance.rb_method_plugin.set(rb_method_plugin);
        let tilt_panel_root = instance.tilt_panel().get_root();
        let _ = instance.tilt_panel_root.set(tilt_panel_root);
        instance
    }

    /// Java package-private static `getInstance(ApplicationManager,
    /// TomogramGenerationExpert, AxisID, TomoGenMethodPlugin)`.
    pub fn get_instance(
        app_mgr: &'static ApplicationManager,
        expert: Weak<TomogramGenerationExpert>,
        axis_id: AxisID,
        method_plugin: Option<&Rc<dyn TomoGenMethodPlugin>>,
    ) -> Rc<TomogramGenerationDialog> {
        let instance = TomogramGenerationDialog::new(app_mgr, expert, axis_id, method_plugin);
        instance.create_panel();
        instance.update_display();
        instance.add_listeners();
        instance
    }

    /// Rust-only: the Java `final` field `tiltPanel` (set by the constructor),
    /// as the object itself (overridden methods dispatch through it).
    fn tilt_panel_object(&self) -> &Rc<dyn TiltPanelVirtual> {
        self.tilt_panel.get().expect("set by the constructor")
    }

    /// Rust-only: the `TiltPanel` part of the Java `final` field `tiltPanel`.
    fn tilt_panel(&self) -> &TiltPanel {
        self.tilt_panel_object().tilt_panel()
    }

    /// Rust-only: the Java `final` field `rbMethodPlugin` (null before the
    /// constructor sets it, or without a method plugin panel).
    fn rb_method_plugin(&self) -> Option<&Rc<RadioButton>> {
        self.rb_method_plugin.get().and_then(Option::as_ref)
    }

    /// Rust-only: the Java `final` field `methodPluginPanel` (null before the
    /// constructor sets it, or without a method plugin).
    fn method_plugin_panel(&self) -> Option<&Rc<dyn PluginPanel>> {
        self.method_plugin_panel.get().and_then(Option::as_ref)
    }

    /// Rust-only: the Java `final` field `multifiltPanel`.
    fn multifilt_panel(&self) -> &Rc<MultifiltPanel> {
        self.multifilt_panel.get().expect("set by the constructor")
    }

    /// Rust-only: the Java `final` field `sirtPanel`.
    fn sirt_panel(&self) -> &Rc<SirtPanel> {
        self.sirt_panel.get().expect("set by the constructor")
    }

    /// Rust-only: the Java `final` field `ctf3dPanel` (null only before the
    /// constructor sets it).
    fn ctf3d_panel(&self) -> Option<&Rc<Ctf3dPanel>> {
        self.ctf3d_panel.get()
    }

    /// Rust-only: the Java `final` field `rbCtf3d` (null only before the
    /// constructor sets it).
    fn rb_ctf3d(&self) -> Option<&Rc<RadioButton>> {
        self.rb_ctf3d.get()
    }

    /// Java private `createPanel()`.
    fn create_panel(&self) {
        // panels
        let pnl_method = JComponent::new_panel();
        let _pnl_filter_trials = JComponent::new_panel();
        // init
        self.rb_back_projection.set_selected_boolean(true);
        // root panel
        let root_panel = self.base.root_panel.get_component();
        // Swing layout: rootPanel.setLayout(new BoxLayout(rootPanel, BoxLayout.Y_AXIS)).
        self.base
            .root_panel
            .set_border(&BeveledBorder::new(Some("Tomogram Generation")).get_border());
        root_panel.add(&pnl_method);
        if let Some(tilt_panel_root) = self.tilt_panel_root.get() {
            root_panel.add(tilt_panel_root);
        }
        root_panel.add(&self.multifilt_panel().get_component());
        root_panel.add(&self.sirt_panel().get_root());
        if let Some(ctf3d_panel) = self.ctf3d_panel() {
            root_panel.add(&ctf3d_panel.get_component());
        }
        if let Some(method_plugin_panel) = self.method_plugin_panel() {
            root_panel.add(&method_plugin_panel.get_component());
        }
        // method panel
        // Swing layout: pnlMethod.setLayout(new BoxLayout(pnlMethod,
        // BoxLayout.X_AXIS)); horizontal glue before, between and after the radio
        // buttons.
        pnl_method.add(&self.rb_back_projection.get_component());
        pnl_method.add(&self.rb_multifilt.get_component());
        pnl_method.add(&self.rb_sirt.get_component());
        if let Some(rb_ctf3d) = self.rb_ctf3d() {
            pnl_method.add(&rb_ctf3d.get_component());
        }
        if let Some(rb_method_plugin) = self.rb_method_plugin() {
            pnl_method.add(&rb_method_plugin.get_component());
        }
        // buttons
        self.base.btn_execute.set_text(Some("Done"));
        self.base.add_exit_buttons();
        // align
        // Component.LEFT_ALIGNMENT
        ui_utilities::align_components_x(&root_panel, 0.0);
    }

    /// Java private `addListeners()`.
    fn add_listeners(self: &Rc<Self>) {
        let context_menu: Weak<dyn ContextMenu> = Rc::downgrade(self) as Weak<dyn ContextMenu>;
        let mouse_adapter: Rc<dyn MouseListener> = GenericMouseAdapter::new(context_menu);
        self.base
            .root_panel
            .get_component()
            .add_mouse_listener(mouse_adapter);
        // Java `new TomogramGenerationActionListener(this)`.
        let adaptee = Rc::downgrade(self);
        let listener: ActionListener = Rc::new(move |event: &ActionEvent| {
            if let Some(adaptee) = adaptee.upgrade() {
                adaptee.action(event);
            }
        });
        self.rb_back_projection
            .add_action_listener(listener.clone());
        self.rb_multifilt.add_action_listener(listener.clone());
        self.rb_sirt.add_action_listener(listener.clone());
        if let Some(rb_ctf3d) = self.rb_ctf3d() {
            rb_ctf3d.add_action_listener(listener.clone());
        }
        if let Some(rb_method_plugin) = self.rb_method_plugin() {
            rb_method_plugin.add_action_listener(listener.clone());
        }
    }

    /// Java public `getAdvancedButton()`.
    pub fn get_advanced_button(&self) -> Rc<GlobalExpandButton> {
        self.base.btn_advanced.clone()
    }

    /// Java public `msgSirtSucceeded()`.
    pub fn msg_sirt_succeeded(&self) {
        self.sirt_panel().msg_sirt_succeeded();
    }

    /// Java package-private `sirtCheckpoint(ConstTiltParam, TomogramState)`.
    /// The Java does not read `tiltForSirtParam`.
    pub fn sirt_checkpoint(
        &self,
        _tilt_for_sirt_param: &dyn ConstTiltParam,
        state: &TomogramState,
    ) {
        self.sirt_panel().checkpoint(state);
        if let Some(method_plugin_panel) = self.method_plugin_panel() {
            method_plugin_panel.update_display();
        }
    }

    /// Java public `@Deprecated allowTiltComSave()` (8/3/2018 See
    /// TiltDisplay).  This function is called when tilt.com may be updated -
    /// and prevents the update if it returns false.  Tilt.com may be updated
    /// when either tilt, multifiltsetup, or sirtsetup is run.  A resume for
    /// sirt is unusual because the com files are recreated if the user
    /// chooses to resume from an iteration.  In this case nothing else can
    /// change so this function returns false and tilt.com isn't updated at
    /// all.
    ///
    /// If the user has switched to Back Projection or Multifilt this function
    /// returns true and allow tilt.com to be modified.
    pub fn allow_tilt_com_save(&self) -> bool {
        true
    }

    /// Java package-private `getParameters(MetaData) throws
    /// FortranInputSyntaxException`.
    pub fn get_parameters_meta_data(
        &self,
        meta_data: &MetaData,
    ) -> Result<(), FortranInputSyntaxException> {
        meta_data.set_gen_back_projection(self.axis_id, self.rb_back_projection.is_selected());
        meta_data.set_gen_filter_trials(self.axis_id, self.rb_multifilt.is_selected());
        meta_data.set_gen_sirt(self.axis_id, self.rb_sirt.is_selected());
        self.tilt_panel().get_parameters_meta_data(meta_data)?;
        self.multifilt_panel().get_parameters_meta_data(meta_data);
        if let Some(ctf3d_panel) = self.ctf3d_panel() {
            ctf3d_panel.get_parameters_meta_data(meta_data);
        }
        self.sirt_panel().get_parameters_meta_data(meta_data);
        Ok(())
    }

    /// Java package-private `getParameters(ReconScreenState)`.
    pub fn get_parameters_recon_screen_state(&self, screen_state: &ReconScreenState) {
        self.tilt_panel()
            .get_parameters_recon_screen_state(screen_state);
        self.multifilt_panel()
            .get_parameters_recon_screen_state(screen_state);
        if let Some(ctf3d_panel) = self.ctf3d_panel() {
            ctf3d_panel.get_parameters_recon_screen_state(screen_state);
        }
        self.sirt_panel()
            .get_parameters_recon_screen_state(screen_state);
        if let Some(method_plugin_panel) = self.method_plugin_panel() {
            method_plugin_panel.get_parameters(screen_state);
        }
    }

    /// Java package-private `setParameters(ConstMetaData)`.
    pub fn set_parameters_const_meta_data(&self, meta_data: &dyn ConstMetaData) {
        if meta_data.is_gen_back_projection(self.axis_id) {
            self.rb_back_projection.set_selected_boolean(true);
        } else if meta_data.is_gen_filter_trials(self.axis_id) {
            self.rb_multifilt.set_selected_boolean(true);
        } else if meta_data.is_gen_sirt(self.axis_id) {
            self.rb_sirt.set_selected_boolean(true);
        } else if let Some(rb_ctf3d) = self.rb_ctf3d() {
            rb_ctf3d.set_selected_boolean(true);
        }
        self.tilt_panel().set_parameters_const_meta_data(meta_data);
        self.multifilt_panel()
            .set_parameters_const_meta_data(meta_data);
        self.sirt_panel().set_parameters_const_meta_data(meta_data);
        if let Some(ctf3d_panel) = self.ctf3d_panel() {
            ctf3d_panel.set_parameters_const_meta_data(meta_data);
        }
        self.method_changed();
        self.tilt_panel()
            .set_filter_type_action_listener(self.multifilt_panel());
    }

    /// Java package-private `setParameters(ConstTiltParam, boolean)`.  Set the
    /// UI parameters with the specified tiltParam values.  WARNING: be sure
    /// the setNewstParam is called first so the binning value for the stack is
    /// known.  The thickness, first and last slice, width and x,y,z offsets are
    /// scaled so that they are represented to the user in unbinned dimensions.
    pub fn set_parameters_const_tilt_param_boolean(
        &self,
        tilt_param: &dyn ConstTiltParam,
        initialize: bool,
    ) {
        self.tilt_panel()
            .set_parameters_const_tilt_param_boolean(tilt_param, initialize);
    }

    /// Java package-private `setParameters(MultifiltSetupParam)`.
    pub fn set_parameters_multifilt_setup_param(&self, param: &MultifiltSetupParam) {
        self.multifilt_panel()
            .set_parameters_multifilt_setup_param(param);
    }

    /// Java package-private `setParameters(SirtsetupParam)`.
    pub fn set_parameters_sirtsetup_param(&self, param: &SirtsetupParam) {
        self.sirt_panel().set_parameters_sirtsetup_param(param);
    }

    /// Java package-private `setParameters(Ctf3dSetupParam)`.
    pub fn set_parameters_ctf3d_setup_param(&self, param: &Ctf3dSetupParam) {
        if let Some(ctf3d_panel) = self.ctf3d_panel() {
            ctf3d_panel.set_parameters_ctf3d_setup_param(param);
        }
    }

    /// Java package-private final `setParameters(ReconScreenState)`.
    pub fn set_parameters_recon_screen_state(&self, screen_state: &ReconScreenState) {
        self.tilt_panel()
            .set_parameters_recon_screen_state(screen_state);
        self.multifilt_panel()
            .set_parameters_recon_screen_state(screen_state);
        self.sirt_panel()
            .set_parameters_recon_screen_state(screen_state);
        if let Some(ctf3d_panel) = self.ctf3d_panel() {
            ctf3d_panel.set_parameters_recon_screen_state(screen_state);
        }
        if let Some(method_plugin_panel) = self.method_plugin_panel() {
            method_plugin_panel.set_parameters(screen_state);
        }
    }

    /// Java private `updateDisplay()`.  Update the dialog with the current
    /// advanced state.
    fn update_display(&self) {
        self.tilt_panel().update_display();
        self.sirt_panel().update_display();
        if let Some(ctf3d_panel) = self.ctf3d_panel() {
            ctf3d_panel.update_display();
        }
        if let Some(method_plugin_panel) = self.method_plugin_panel() {
            method_plugin_panel.update_display();
        }
        let manager: &'static dyn BaseManager = self.application_manager;
        ui_harness::with(|harness| {
            harness.pack_axis_id_base_manager(Some(self.axis_id), Some(manager))
        });
    }

    /// Java public `setTiltState(TomogramState, ConstMetaData)`.
    pub fn set_tilt_state(&self, state: &TomogramState, meta_data: &dyn ConstMetaData) {
        self.tilt_panel().set_state(state, meta_data);
    }

    /// Java package-private `getTiltDisplay()`.
    pub fn get_tilt_display(&self) -> Rc<dyn TiltDisplay> {
        self.tilt_panel_object().clone() as Rc<dyn TiltDisplay>
    }

    /// Java package-private `getMultifiltSetupDisplay()`.
    pub fn get_multifilt_setup_display(&self) -> Rc<dyn MultifiltSetupDisplay> {
        self.multifilt_panel().clone() as Rc<dyn MultifiltSetupDisplay>
    }

    /// Java package-private `getCtf3dSetupDisplay()`; `None` is Java null.
    pub fn get_ctf3d_setup_display(&self) -> Option<Rc<dyn Ctf3dSetupDisplay>> {
        self.ctf3d_panel()
            .map(|ctf3d_panel| ctf3d_panel.clone() as Rc<dyn Ctf3dSetupDisplay>)
    }

    /// Java package-private `getSirtsetupDisplay()`.
    pub fn get_sirtsetup_display(&self) -> Rc<dyn SirtsetupDisplay> {
        self.sirt_panel().clone() as Rc<dyn SirtsetupDisplay>
    }

    /// Java package-private `addAxisTiltFocusListener(FocusListener)`.
    pub fn add_axis_tilt_focus_listener(&self, listener: FocusListener) {
        self.tilt_panel().add_axis_tilt_focus_listener(listener);
    }

    /// Java package-private `removeAxisTiltFocusListener(FocusListener)`.
    pub fn remove_axis_tilt_focus_listener(&self, listener: &FocusListener) {
        self.tilt_panel().remove_axis_tilt_focus_listener(listener);
    }

    /// Java package-private `addUseLocalAlignmentActionListener(ActionListener)`.
    pub fn add_use_local_alignment_action_listener(&self, listener: ActionListener) {
        self.tilt_panel()
            .add_use_local_alignment_action_listener(listener);
    }

    /// Java package-private
    /// `removeUseLocalAlignmentActionListener(ActionListener)`.
    pub fn remove_use_local_alignment_action_listener(&self, listener: &ActionListener) {
        self.tilt_panel()
            .remove_use_local_alignment_action_listener(listener);
    }

    /// Java package-private `addUseZFactorsActionListener(ActionListener)`.
    pub fn add_use_z_factors_action_listener(&self, listener: ActionListener) {
        self.tilt_panel()
            .add_use_z_factors_action_listener(listener);
    }

    /// Java package-private `removeUseZFactorsActionListener(ActionListener)`.
    pub fn remove_use_z_factors_action_listener(&self, listener: &ActionListener) {
        self.tilt_panel()
            .remove_use_z_factors_action_listener(listener);
    }

    /// Java package-private `getXAxisTilt()`.
    pub fn get_x_axis_tilt(&self) -> Option<String> {
        self.tilt_panel().get_x_axis_tilt()
    }

    /// Java package-private `isUseLocalAlignment()`.
    pub fn is_use_local_alignment(&self) -> bool {
        self.tilt_panel().is_use_local_alignment()
    }

    /// Java package-private `isUseZFactors()`.
    pub fn is_use_z_factors(&self) -> bool {
        self.tilt_panel().is_use_z_factors()
    }

    /// Java package-private `getTomoThickness()`; `None` is Java null.
    pub fn get_tomo_thickness(&self) -> Option<i64> {
        self.tilt_panel().get_tomo_thickness()
    }

    /// Java private `methodChanged()`.
    fn method_changed(&self) {
        self.tilt_panel().msg_method_changed();
        self.multifilt_panel().msg_method_changed();
        self.sirt_panel().msg_method_changed();
        if let Some(ctf3d_panel) = self.ctf3d_panel() {
            ctf3d_panel.msg_method_changed();
        }
        if let Some(method_plugin_panel) = self.method_plugin_panel() {
            method_plugin_panel.msg_visibility_changed(self.is_method_plugin());
        }
        let manager: &'static dyn BaseManager = self.application_manager;
        ui_harness::with(|harness| {
            harness.pack_axis_id_base_manager(Some(self.axis_id), Some(manager));
            harness.move_sub_frame();
        });
    }

    /// Java public `getProcessingMethod()`.
    pub fn get_processing_method(&self) -> ProcessingMethod {
        TrialTiltParent::get_processing_method(&**self.tilt_panel_object())
    }

    /// Java package-private `displayMultifilt()`.
    pub fn display_multifilt(&self) {
        self.rb_multifilt.do_click();
    }

    /// Java private `action(ActionEvent)`.
    fn action(&self, event: &ActionEvent) {
        let action_command = event.get_action_command();
        if action_command == self.rb_back_projection.get_action_command().as_deref()
            || action_command == self.rb_multifilt.get_action_command().as_deref()
            || action_command == self.rb_sirt.get_action_command().as_deref()
            || self
                .rb_ctf3d()
                .is_some_and(|rb_ctf3d| action_command == rb_ctf3d.get_action_command().as_deref())
            || self.rb_method_plugin().is_some_and(|rb_method_plugin| {
                action_command == rb_method_plugin.get_action_command().as_deref()
            })
        {
            self.method_changed();
        }
    }
}

impl ProcessDialogVirtual for TomogramGenerationDialog {
    fn process_dialog(&self) -> &ProcessDialog {
        &self.base
    }

    /// Java package-private override `done()`.
    fn done(&self) {
        if let Some(expert) = self.expert.upgrade() {
            expert.done_dialog_void();
        }
        self.tilt_panel().done();
        self.multifilt_panel().done();
        self.sirt_panel().done();
        if let Some(ctf3d_panel) = self.ctf3d_panel() {
            ctf3d_panel.done();
        }
        if let Some(method_plugin_panel) = self.method_plugin_panel() {
            method_plugin_panel.done();
        }
        self.base.set_displayed(false);
    }
}

impl ContextMenu for TomogramGenerationDialog {
    /// Java override `popUpContextMenu(MouseEvent)`.  Right mouse button
    /// context menu.  Sensitive to the method radio buttons.
    fn pop_up_context_menu(&self, mouse_event: &MouseEvent) {
        let sirt_in_use = self.rb_sirt.is_selected();
        let filter_trials = self.rb_multifilt.is_selected();
        let ctf3d = match self.rb_ctf3d() {
            Some(rb_ctf3d) => rb_ctf3d.is_selected(),
            None => false,
        };
        let mut man_page_label: Vec<String> = Vec::new();
        let mut man_page: Vec<String> = Vec::new();
        let mut log_file_label: Vec<String> = Vec::new();
        let mut log_file: Vec<String> = Vec::new();
        let _i = 0;
        man_page_label.push("Tilt".to_string());
        man_page.push("tilt.html".to_string());
        let manager: &'static dyn BaseManager = self.application_manager;
        if !sirt_in_use {
            log_file_label.push("Tilt".to_string());
            log_file.push(format!("tilt{}.log", self.axis_id.get_extension()));
            if filter_trials {
                man_page_label.push("Multifiltsetup".to_string());
                man_page.push("multifiltsetup.html".to_string());
                log_file_label.push("Multifiltsetup".to_string());
                log_file.push(format!(
                    "multifiltsetup{}.log",
                    self.axis_id.get_extension()
                ));
            } else if ctf3d {
                man_page_label.push("CTF3Dsetup".to_string());
                man_page.push("ctf3dsetup.html".to_string());
                log_file_label.push("CTF3Dsetup".to_string());
                // A null file name (FileType could not build one) is Java's
                // "null" element.
                log_file.push(
                    file_type::CLASS
                        .ctf_3d_setup_log
                        .get_file_name(Some(manager), Some(self.axis_id))
                        .unwrap_or_else(|| "null".to_string()),
                );
                log_file_label.push("Final CTF 3D".to_string());
                log_file.push(
                    file_type::CLASS
                        .ctf_3d_finish_log
                        .get_file_name(Some(manager), Some(self.axis_id))
                        .unwrap_or_else(|| "null".to_string()),
                );
            }
        } else {
            log_file_label.push("Final SIRT".to_string());
            log_file.push(format!(
                "tilt{}_sirt-finish.log",
                self.axis_id.get_extension()
            ));
            man_page_label.push("Sirtsetup".to_string());
            man_page.push("sirtsetup.html".to_string());
        }
        man_page_label.push("3dmod".to_string());
        man_page.push("3dmod.html".to_string());
        // `Converter.toArray(list)` for each list; the Java constructor's
        // IllegalArgumentException (mismatched arrays) cannot occur here: the
        // lists are built in label/value pairs.
        let _context_popup = ContextPopup::new_component_mouse_event_string_string_string_array_string_array_string_array_string_array_base_manager_axis_id(
            &self.base.root_panel.get_component(),
            mouse_event,
            Some("TOMOGRAM GENERATION"),
            Some(context_popup::TOMO_GUIDE),
            &man_page_label,
            &man_page,
            Some(&log_file_label),
            Some(&log_file),
            manager,
            self.axis_id,
        );
    }
}

impl TomogramGenerationParent for TomogramGenerationDialog {
    /// Java public override `isCtf3d()`.
    fn is_ctf3d(&self) -> bool {
        if let Some(rb_ctf3d) = self.rb_ctf3d() {
            return rb_ctf3d.is_selected();
        }
        false
    }

    /// Java public override `isMethodPlugin()`.
    fn is_method_plugin(&self) -> bool {
        if let Some(rb_method_plugin) = self.rb_method_plugin() {
            return rb_method_plugin.is_selected();
        }
        false
    }

    /// Java public override `isMultifilt()`.
    fn is_multifilt(&self) -> bool {
        self.rb_multifilt.is_selected()
    }

    /// Java public override `isBackProjection()`.
    fn is_back_projection(&self) -> bool {
        self.rb_back_projection.is_selected()
    }

    /// Java public override `isSirt()`.
    fn is_sirt(&self) -> bool {
        self.rb_sirt.is_selected()
    }
}
