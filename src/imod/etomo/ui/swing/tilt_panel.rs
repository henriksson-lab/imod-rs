//! `IMOD/Etomo/src/etomo/ui/swing/TiltPanel.java`.
//!
//! Java `public class TiltPanel extends AbstractTiltPanel`: the back
//! projection (tilt.com) panel of the Tomogram Generation dialog.  An EDT
//! object created as `Rc<Self>` by [`TiltPanel::get_instance`]; every method
//! takes `&self`.  The `AbstractTiltPanel` superclass is the embedded `base`
//! (reached through `Deref`); the abstract and overridden members are
//! [`AbstractTiltPanelVirtual`], and the interfaces the superclass implements
//! (`Expandable`, `TrialTiltParent`, `Run3dmodButtonContainer`,
//! `TiltDisplay`, `RadialParent`) are implemented here on the subclass object
//! (Java's `this`), delegating to the superclass bodies or this class's
//! overrides.

use std::ops::Deref;
use std::rc::{Rc, Weak};

use super::abstract_tilt_panel::{AbstractTiltPanel, AbstractTiltPanelVirtual};
use super::context_popup::{self, ContextPopup};
use super::deferred_3dmod_button::Deferred3dmodButton;
use super::expand_button::ExpandButton;
use super::expandable::Expandable;
use super::global_expand_button::GlobalExpandButton;
use super::process_display::ProcessDisplay;
use super::radial_parent::RadialParent;
use super::run_3dmod_button_container::Run3dmodButtonContainer;
use super::tilt_display::{TiltDisplay, TiltDisplayException};
use super::tomogram_generation_parent::TomogramGenerationParent;
use super::trial_tilt_parent::TrialTiltParent;
use super::ui_utilities;
use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::splittilt_param::SplittiltParam;
use crate::imod::etomo::comscript::tilt_param::TiltParam;
use crate::imod::etomo::jdk::{JComponent, MouseEvent};
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::panel_id::PanelId;
use crate::imod::etomo::r#type::process_result_display::ProcessResultDisplayHandle;
use crate::imod::etomo::r#type::processing_method::ProcessingMethod;

/// A `TiltPanel` or an object of a subclass of it (`DemoTiltPanel`), as the
/// `TomogramGenerationDialog` holds its `tiltPanel` field (Java type `TiltPanel`).
/// Overridden methods are reached through the supertraits (`update_display` through
/// `AbstractTiltPanel`'s dispatch, `get_processing_method` through
/// `TrialTiltParent`); everything else through [`TiltPanelVirtual::tilt_panel`].
pub trait TiltPanelVirtual: AbstractTiltPanelVirtual + ProcessDisplay {
    /// The embedded `TiltPanel` (the Java superclass part, or the object itself).
    fn tilt_panel(&self) -> &TiltPanel;
}

impl TiltPanelVirtual for TiltPanel {
    fn tilt_panel(&self) -> &TiltPanel {
        self
    }
}

/// Java `public class TiltPanel extends AbstractTiltPanel /*implements
/// ResumeObserver */`.
pub struct TiltPanel {
    /// The `AbstractTiltPanel` superclass.
    base: AbstractTiltPanel,
    /// Java private final `pnlTiltPanelRoot = new JPanel()`.
    pnl_tilt_panel_root: Rc<JComponent>,
}

impl Deref for TiltPanel {
    type Target = AbstractTiltPanel;
    fn deref(&self) -> &AbstractTiltPanel {
        &self.base
    }
}

impl TiltPanel {
    /// Java protected constructor `TiltPanel(ApplicationManager, AxisID,
    /// DialogType, GlobalExpandButton, PanelId, TomogramGenerationParent)`.
    fn new(
        manager: &'static ApplicationManager,
        axis_id: AxisID,
        dialog_type: DialogType,
        global_advanced_button: &Rc<GlobalExpandButton>,
        panel_id: PanelId,
        parent: Weak<dyn TomogramGenerationParent>,
    ) -> Rc<TiltPanel> {
        Rc::new_cyclic(|this: &Weak<TiltPanel>| {
            TiltPanel::new_base(
                manager,
                axis_id,
                dialog_type,
                global_advanced_button,
                panel_id,
                parent,
                this,
            )
        })
    }

    /// The body of the Java protected constructor, for this class and for a subclass
    /// (`DemoTiltPanel`): `this` is the object being constructed (the subclass object
    /// when there is one), which the superclass dispatches its overridable methods
    /// to.
    pub fn new_base<T: AbstractTiltPanelVirtual + 'static>(
        manager: &'static ApplicationManager,
        axis_id: AxisID,
        dialog_type: DialogType,
        global_advanced_button: &Rc<GlobalExpandButton>,
        panel_id: PanelId,
        parent: Weak<dyn TomogramGenerationParent>,
        this: &Weak<T>,
    ) -> TiltPanel {
        // super(manager, axisID, dialogType, globalAdvancedButton, panelId, false,
        // parent)
        let base = AbstractTiltPanel::new(
            manager,
            axis_id,
            dialog_type,
            Some(global_advanced_button),
            panel_id,
            false,
            parent,
            this,
        );
        // Field initializer.
        let pnl_tilt_panel_root = JComponent::new_panel();
        TiltPanel {
            base,
            pnl_tilt_panel_root,
        }
    }

    /// Java package-private static `getInstance(ApplicationManager, AxisID,
    /// DialogType, GlobalExpandButton, TomogramGenerationParent)`.
    pub fn get_instance(
        manager: &'static ApplicationManager,
        axis_id: AxisID,
        dialog_type: DialogType,
        global_advanced_button: &Rc<GlobalExpandButton>,
        parent: Weak<dyn TomogramGenerationParent>,
    ) -> Rc<TiltPanel> {
        let instance = TiltPanel::new(
            manager,
            axis_id,
            dialog_type,
            global_advanced_button,
            PanelId::Tilt,
            parent,
        );
        instance.create_panel();
        instance.set_tool_tip_text();
        instance.add_listeners();
        instance
    }

    /// Java protected final override `addListeners()`.
    pub fn add_listeners(&self) {
        self.base.add_listeners();
        // If the child class has field listeners, it must set listenForFieldChanges to
        // true. Otherwise changes in these fields will be ignored.
        if self.base.listen_for_field_changes {}
    }

    /// Java protected final override `createPanel()`.
    pub fn create_panel(&self) {
        self.base.create_panel();
        // Swing layout: pnlTiltPanelRoot.setLayout(new BoxLayout(pnlTiltPanelRoot,
        // BoxLayout.Y_AXIS)).
        // `super.getRoot()`: the superclass body.
        ui_utilities::add_with_y_space(&self.pnl_tilt_panel_root, &self.base.get_root());
        // Component.CENTER_ALIGNMENT
        ui_utilities::align_components_x(&self.pnl_tilt_panel_root, 0.5);
    }

    /// Java final override `getRoot()`.
    pub fn get_root(&self) -> Rc<JComponent> {
        self.pnl_tilt_panel_root.clone()
    }

    /// Java public final override `@Deprecated allowTiltComSave()` (8/3/2018
    /// See TiltDisplay).
    pub fn allow_tilt_com_save(&self) -> bool {
        true
    }

    /// Java public final `@Deprecated isResume()` (9/13/2018 See
    /// TiltDisplay).
    pub fn is_resume(&self) -> bool {
        false
    }

    /// Java public final `@Deprecated msgResumeChanged(boolean)` (8/6/18
    /// Saving tilt.com no longer effects SIRT resume - Bug# 2098, comment 10);
    /// empty.
    pub fn msg_resume_changed(&self, _resume: bool) {}

    /// Java protected override `updateDisplay()`.  Z shift is an advanced
    /// field.  This is the body (a subclass's `super.updateDisplay()`); calls on the
    /// object dispatch through `AbstractTiltPanel::update_display` to the most derived
    /// override.
    pub fn update_display_tilt_panel(&self) {
        self.base.update_display_super();
        let advanced = self.base.is_advanced();
        self.base.ltf_z_shift.set_visible(advanced);
    }

    /// Java protected override `setAdvancedFieldDisplayer()`.
    pub fn set_advanced_field_displayer(&self) {
        self.base.set_advanced_field_displayer_super();
        self.base
            .ltf_z_shift
            .set_overridable_field_displayers(None, self.base.advanced_field_displayer.clone());
    }

    /// Java package-private final `popUpContextMenu(String, Component,
    /// MouseEvent)`.
    pub fn pop_up_context_menu(
        &self,
        anchor: Option<&str>,
        root_panel: &Rc<JComponent>,
        mouse_event: &MouseEvent,
    ) {
        let man_pagelabel: Vec<String> = vec!["Tilt".to_string(), "3dmod".to_string()];
        let man_page: Vec<String> = vec!["tilt.html".to_string(), "3dmod.html".to_string()];
        let log_file_label: Vec<String> = vec!["Tilt".to_string()];
        let mut log_file: Vec<String> = vec![String::new(); 1];
        log_file[0] = format!("tilt{}.log", self.base.axis_id.get_extension());
        let manager: &'static dyn BaseManager = self.base.manager;
        // Java `ContextPopup contextPopup = new ContextPopup(...)`; the Java
        // constructor's IllegalArgumentException (mismatched arrays) cannot
        // occur with these literal arrays.
        let _context_popup = ContextPopup::new_component_mouse_event_string_string_string_array_string_array_string_array_string_array_base_manager_axis_id(
            root_panel,
            mouse_event,
            anchor,
            Some(context_popup::TOMO_GUIDE),
            &man_pagelabel,
            &man_page,
            Some(&log_file_label),
            Some(&log_file),
            manager,
            self.base.axis_id,
        );
    }

    /// Java protected final override `setToolTipText()`.
    pub fn set_tool_tip_text(&self) {
        self.base.set_tool_tip_text();
    }
}

impl AbstractTiltPanelVirtual for TiltPanel {
    fn abstract_tilt_panel(&self) -> &AbstractTiltPanel {
        &self.base
    }

    /// Java package-private final override `tiltAction(ProcessResultDisplay,
    /// Deferred3dmodButton, Run3dmodMenuOptions, ProcessingMethod)`.
    fn tilt_action(
        &self,
        process_result_display: Option<ProcessResultDisplayHandle>,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
        tilt_processing_method: ProcessingMethod,
    ) {
        // `manager.tiltAction(processResultDisplay, null, deferred3dmodButton,
        // run3dmodMenuOptions, this, axisID, dialogType, tiltProcessingMethod)`.
        // The manager takes the menu options by value; Java's null (a listener
        // action) is the empty options.
        self.base.manager.tilt_action(
            process_result_display,
            None,
            deferred_3dmod_button,
            run_3dmod_menu_options.unwrap_or_default(),
            Some(self as &dyn TiltDisplay),
            self.base.axis_id,
            self.base.dialog_type,
            tilt_processing_method,
        );
    }

    /// Java package-private final override `imodTomogramAction(
    /// Deferred3dmodButton, Run3dmodMenuOptions)`.
    fn imod_tomogram_action(
        &self,
        _deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    ) {
        self.base.manager.imod_full_volume(
            self.base.axis_id,
            run_3dmod_menu_options.unwrap_or_default(),
        );
    }

    /// Java protected override `updateDisplay()`.
    fn update_display(&self) {
        TiltPanel::update_display_tilt_panel(self);
    }

    /// Java protected override `setAdvancedFieldDisplayer()`.
    fn set_advanced_field_displayer(&self) {
        TiltPanel::set_advanced_field_displayer(self);
    }
}

impl Expandable for TiltPanel {
    /// Java inherited public final `expand(ExpandButton)`.
    fn expand_expand_button(&self, button: &Rc<ExpandButton>) {
        self.base.expand_expand_button(button);
    }

    /// Java inherited public final `expand(GlobalExpandButton)`.
    fn expand_global_expand_button(&self, button: &Rc<GlobalExpandButton>) {
        self.base.expand_global_expand_button(button);
    }
}

impl RadialParent for TiltPanel {
    /// Java inherited public final `isMultifilt()`.
    fn is_multifilt(&self) -> bool {
        self.base.is_multifilt()
    }

    /// Java inherited public final `isCtf3d()`.
    fn is_ctf3d(&self) -> bool {
        self.base.is_ctf3d()
    }

    /// Java inherited `isAdvanced()`.
    fn is_advanced(&self) -> bool {
        self.base.is_advanced()
    }
}

impl TrialTiltParent for TiltPanel {
    /// Java inherited `getParameters(TiltParam, boolean)`.
    fn get_parameters_tilt_param_boolean(
        &self,
        tilt_param: &mut TiltParam,
        do_validation: bool,
    ) -> Result<bool, TiltDisplayException> {
        self.base
            .get_parameters_tilt_param_boolean(tilt_param, do_validation)
    }

    /// Java inherited `getParameters(SplittiltParam, boolean)`.
    fn get_parameters_splittilt_param_boolean(
        &self,
        param: &mut SplittiltParam,
        do_validation: bool,
    ) -> bool {
        self.base
            .get_parameters_splittilt_param_boolean(param, do_validation)
    }

    /// Java inherited `getProcessingMethod()`.
    fn get_processing_method(&self) -> ProcessingMethod {
        self.base.get_processing_method()
    }
}

impl Run3dmodButtonContainer for TiltPanel {
    /// Java inherited public final `action(String, Deferred3dmodButton,
    /// Run3dmodMenuOptions)`.
    fn action(
        &self,
        action_command: &str,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    ) {
        self.base
            .action_string_deferred_3dmod_button_run_3dmod_menu_options(
                action_command,
                deferred_3dmod_button,
                run_3dmod_menu_options,
            );
    }
}

impl ProcessDisplay for TiltPanel {
    fn as_tilt_display(&self) -> Option<&dyn super::tilt_display::TiltDisplay> {
        Some(self)
    }
}

impl TiltDisplay for TiltPanel {
    /// Java inherited `getParameters(TiltParam, boolean)`.
    fn get_parameters(
        &self,
        param: &mut TiltParam,
        do_validation: bool,
    ) -> Result<bool, TiltDisplayException> {
        self.base
            .get_parameters_tilt_param_boolean(param, do_validation)
    }

    /// Java inherited `getParameters(SplittiltParam, boolean)`.
    fn get_parameters_splittilt(&self, param: &mut SplittiltParam, do_validation: bool) -> bool {
        self.base
            .get_parameters_splittilt_param_boolean(param, do_validation)
    }

    /// Java public final override `@Deprecated allowTiltComSave()`.
    fn allow_tilt_com_save(&self) -> bool {
        TiltPanel::allow_tilt_com_save(self)
    }

    /// Java inherited `setDebug(boolean)`.
    fn set_debug(&self, debug: bool) {
        self.base.set_debug(debug);
    }
}
