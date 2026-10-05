//! `IMOD/Etomo/src/etomo/plugin/demo/DemoTiltPanel.java`.
//!
//! Inherit `TiltPanel` to modify field behavior and appearance.
//!
//! `DemoTiltPanel extends TiltPanel`: the superclass is the embedded `base` (reached
//! through `Deref`), built with this object as the `AbstractTiltPanel`'s `this`, so the
//! superclass dispatches `updateDisplay` (and every other overridable call) here.  The
//! two overrides are `updateDisplay` ([`AbstractTiltPanelVirtual::update_display`]) and
//! `getProcessingMethod` ([`TrialTiltParent::get_processing_method`]); the inherited
//! interface methods delegate to `TiltPanel`'s bodies.

use std::ops::Deref;
use std::rc::{Rc, Weak};

use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::comscript::splittilt_param::SplittiltParam;
use crate::imod::etomo::comscript::tilt_param::TiltParam;
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::panel_id::PanelId;
use crate::imod::etomo::r#type::process_result_display::ProcessResultDisplayHandle;
use crate::imod::etomo::r#type::processing_method::ProcessingMethod;
use crate::imod::etomo::ui::swing::abstract_tilt_panel::{
    AbstractTiltPanel, AbstractTiltPanelVirtual,
};
use crate::imod::etomo::ui::swing::deferred_3dmod_button::Deferred3dmodButton;
use crate::imod::etomo::ui::swing::expand_button::ExpandButton;
use crate::imod::etomo::ui::swing::expandable::Expandable;
use crate::imod::etomo::ui::swing::global_expand_button::GlobalExpandButton;
use crate::imod::etomo::ui::swing::process_display::ProcessDisplay;
use crate::imod::etomo::ui::swing::radial_parent::RadialParent;
use crate::imod::etomo::ui::swing::run_3dmod_button_container::Run3dmodButtonContainer;
use crate::imod::etomo::ui::swing::tilt_display::{TiltDisplay, TiltDisplayException};
use crate::imod::etomo::ui::swing::tilt_panel::{TiltPanel, TiltPanelVirtual};
use crate::imod::etomo::ui::swing::tomogram_generation_parent::TomogramGenerationParent;
use crate::imod::etomo::ui::swing::trial_tilt_parent::TrialTiltParent;

/// Java `final class DemoTiltPanel extends TiltPanel`.
pub struct DemoTiltPanel {
    /// The `TiltPanel` superclass.
    base: TiltPanel,
}

impl Deref for DemoTiltPanel {
    type Target = TiltPanel;
    fn deref(&self) -> &TiltPanel {
        &self.base
    }
}

impl DemoTiltPanel {
    /// Java private `DemoTiltPanel(ApplicationManager, AxisID, DialogType,
    /// GlobalExpandButton, PanelId, TomogramGenerationParent)`.
    fn new(
        manager: &'static ApplicationManager,
        axis_id: AxisID,
        dialog_type: DialogType,
        global_advanced_button: &Rc<GlobalExpandButton>,
        panel_id: PanelId,
        parent: Weak<dyn TomogramGenerationParent>,
    ) -> Rc<DemoTiltPanel> {
        Rc::new_cyclic(|this: &Weak<DemoTiltPanel>| DemoTiltPanel {
            // super(manager, axisID, dialogType, globalAdvancedButton, panelId, parent)
            base: TiltPanel::new_base(
                manager,
                axis_id,
                dialog_type,
                global_advanced_button,
                panel_id,
                parent,
                this,
            ),
        })
    }

    /// Java package-private static `getInstance(ApplicationManager, AxisID, DialogType,
    /// GlobalExpandButton, TomogramGenerationParent)`.
    pub fn get_instance(
        manager: &'static ApplicationManager,
        axis_id: AxisID,
        dialog_type: DialogType,
        global_advanced_button: &Rc<GlobalExpandButton>,
        parent: Weak<dyn TomogramGenerationParent>,
    ) -> Rc<DemoTiltPanel> {
        let instance = DemoTiltPanel::new(
            manager,
            axis_id,
            dialog_type,
            global_advanced_button,
            PanelId::Tilt,
            parent,
        );
        instance.init();
        instance.create_panel();
        instance.set_tool_tip_text();
        instance.set_demo_tool_tip_text();
        instance.add_listeners();
        instance
    }

    /// Java private `init()`.
    fn init(&self) {
        self.ctf_log.set_alternate_label(Some(
            "Take logarithm of densities with maximum density value as the offset",
        ));
    }

    /// Java private `setDemoToolTipText()`.
    fn set_demo_tool_tip_text(&self) {
        self.cb_use_z_factors
            .set_alternate_tooltip_text(Some("Demo currently does not support Z factors."));
        self.cb_use_local_alignment
            .set_alternate_tooltip_text(Some("Demo currently does not support local alignments."));
        self.ctf_log.set_alternate_tooltip_text(Some(
            "Take logarithm of densities with maximum density value as an offset.",
        ));
    }
}

impl TiltPanelVirtual for DemoTiltPanel {
    fn tilt_panel(&self) -> &TiltPanel {
        &self.base
    }
}

impl AbstractTiltPanelVirtual for DemoTiltPanel {
    fn abstract_tilt_panel(&self) -> &AbstractTiltPanel {
        &self.base
    }

    /// Java inherited `TiltPanel.tiltAction(...)`: the display handed to the manager is
    /// this object (Java `this`).
    fn tilt_action(
        &self,
        process_result_display: Option<ProcessResultDisplayHandle>,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
        tilt_processing_method: ProcessingMethod,
    ) {
        self.manager.tilt_action(
            process_result_display,
            None,
            deferred_3dmod_button,
            run_3dmod_menu_options.unwrap_or_default(),
            Some(self as &dyn TiltDisplay),
            self.axis_id,
            self.dialog_type,
            tilt_processing_method,
        );
    }

    /// Java inherited `TiltPanel.imodTomogramAction(...)`.
    fn imod_tomogram_action(
        &self,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    ) {
        AbstractTiltPanelVirtual::imod_tomogram_action(
            &self.base,
            deferred_3dmod_button,
            run_3dmod_menu_options,
        );
    }

    /// Java protected override `updateDisplay()`.
    fn update_display(&self) {
        self.base.update_display_tilt_panel();

        // visibility
        if self.is_method_plugin() {
            self.ltf_log_density_scale_offset.set_visible(false);
            self.ltf_log_density_scale_factor.set_visible(false);
            self.ltf_linear_density_scale_offset.set_visible(false);
            self.ltf_linear_density_scale_factor.set_visible(false);
            self.ltf_tilt_angle_offset.set_visible(false);
            self.ltf_z_shift.set_visible(false);
            self.ctf_log.set_text_field_visible(false);
        }

        if self.is_method_plugin() {
            self.ltf_z_shift.set_visible(false);
        }

        let demo = self.is_method_plugin();
        if demo {
            self.cb_use_local_alignment.set_enabled(false);
            self.cb_use_z_factors.set_enabled(false);
        }
        self.cb_use_z_factors.switch_tooltips(demo);
        self.cb_use_local_alignment.switch_tooltips(demo);
        self.ctf_log.switch_labels(demo);
        self.ctf_log.switch_tooltips(demo);
    }

    /// Java inherited `TiltPanel.setAdvancedFieldDisplayer()`.
    fn set_advanced_field_displayer(&self) {
        TiltPanel::set_advanced_field_displayer(&self.base);
    }
}

impl TrialTiltParent for DemoTiltPanel {
    /// Java inherited `getParameters(TiltParam, boolean)`.
    fn get_parameters_tilt_param_boolean(
        &self,
        tilt_param: &mut TiltParam,
        do_validation: bool,
    ) -> Result<bool, TiltDisplayException> {
        TrialTiltParent::get_parameters_tilt_param_boolean(&self.base, tilt_param, do_validation)
    }

    /// Java inherited `getParameters(SplittiltParam, boolean)`.
    fn get_parameters_splittilt_param_boolean(
        &self,
        param: &mut SplittiltParam,
        do_validation: bool,
    ) -> bool {
        TrialTiltParent::get_parameters_splittilt_param_boolean(&self.base, param, do_validation)
    }

    /// Java public override `getProcessingMethod()`.  This demo does do parallel
    /// processing.
    fn get_processing_method(&self) -> ProcessingMethod {
        if self.is_method_plugin() {
            return ProcessingMethod::LocalCpu;
        }
        TrialTiltParent::get_processing_method(&self.base)
    }
}

impl Expandable for DemoTiltPanel {
    /// Java inherited public final `expand(ExpandButton)`.
    fn expand_expand_button(&self, button: &Rc<ExpandButton>) {
        Expandable::expand_expand_button(&self.base, button);
    }

    /// Java inherited public final `expand(GlobalExpandButton)`.
    fn expand_global_expand_button(&self, button: &Rc<GlobalExpandButton>) {
        Expandable::expand_global_expand_button(&self.base, button);
    }
}

impl RadialParent for DemoTiltPanel {
    /// Java inherited public final `isMultifilt()`.
    fn is_multifilt(&self) -> bool {
        RadialParent::is_multifilt(&self.base)
    }

    /// Java inherited public final `isCtf3d()`.
    fn is_ctf3d(&self) -> bool {
        RadialParent::is_ctf3d(&self.base)
    }

    /// Java inherited `isAdvanced()`.
    fn is_advanced(&self) -> bool {
        RadialParent::is_advanced(&self.base)
    }
}

impl Run3dmodButtonContainer for DemoTiltPanel {
    /// Java inherited public final `action(String, Deferred3dmodButton,
    /// Run3dmodMenuOptions)`.
    fn action(
        &self,
        action_command: &str,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    ) {
        Run3dmodButtonContainer::action(
            &self.base,
            action_command,
            deferred_3dmod_button,
            run_3dmod_menu_options,
        );
    }
}

impl ProcessDisplay for DemoTiltPanel {
    fn as_tilt_display(&self) -> Option<&dyn TiltDisplay> {
        Some(self)
    }
}

impl TiltDisplay for DemoTiltPanel {
    /// Java inherited `getParameters(TiltParam, boolean)`.
    fn get_parameters(
        &self,
        param: &mut TiltParam,
        do_validation: bool,
    ) -> Result<bool, TiltDisplayException> {
        TiltDisplay::get_parameters(&self.base, param, do_validation)
    }

    /// Java inherited `getParameters(SplittiltParam, boolean)`.
    fn get_parameters_splittilt(&self, param: &mut SplittiltParam, do_validation: bool) -> bool {
        TiltDisplay::get_parameters_splittilt(&self.base, param, do_validation)
    }

    /// Java inherited public final `@Deprecated allowTiltComSave()`.
    fn allow_tilt_com_save(&self) -> bool {
        TiltDisplay::allow_tilt_com_save(&self.base)
    }

    /// Java inherited `setDebug(boolean)`.
    fn set_debug(&self, debug: bool) {
        TiltDisplay::set_debug(&self.base, debug);
    }
}
