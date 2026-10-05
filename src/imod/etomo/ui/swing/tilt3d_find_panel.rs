//! `IMOD/Etomo/src/etomo/ui/swing/Tilt3dFindPanel.java`.
//!
//! Java `final class Tilt3dFindPanel extends AbstractTiltPanel`: the
//! tilt_3dfind panel of the erase-gold (findbeads3d) tab.  An EDT object
//! created as `Rc<Self>` by [`Tilt3dFindPanel::get_instance`]; every method
//! takes `&self`.  The `AbstractTiltPanel` superclass is the embedded `base`
//! (reached through `Deref`); the abstract and overridden members are
//! [`AbstractTiltPanelVirtual`], and the interfaces the superclass implements
//! are implemented here on the subclass object (Java's `this`), delegating to
//! the superclass bodies or this class's overrides.

use std::ops::Deref;
use std::rc::{Rc, Weak};

use super::abstract_tilt_panel::{AbstractTiltPanel, AbstractTiltPanelVirtual};
use super::deferred_3dmod_button::Deferred3dmodButton;
use super::expand_button::ExpandButton;
use super::expandable::Expandable;
use super::find_beads3d_panel;
use super::global_expand_button::GlobalExpandButton;
use super::labeled_text_field::LabeledTextField;
use super::process_display::ProcessDisplay;
use super::radial_parent::RadialParent;
use super::run_3dmod_button_container::Run3dmodButtonContainer;
use super::spaced_panel::SpacedPanel;
use super::tilt_display::{TiltDisplay, TiltDisplayException};
use super::tilt3d_find_parent::Tilt3dFindParent;
use super::tomogram_generation_parent::TomogramGenerationParent;
use super::trial_tilt_parent::TrialTiltParent;
use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::const_tilt_param::ConstTiltParam;
use crate::imod::etomo::comscript::const_tiltalign_param::ConstTiltalignParam;
use crate::imod::etomo::comscript::fortran_input_syntax_exception::FortranInputSyntaxException;
use crate::imod::etomo::comscript::splittilt_param::SplittiltParam;
use crate::imod::etomo::comscript::tilt_param::{Mode, TiltParam};
use crate::imod::etomo::jdk::JComponent;
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::storage::ta_angles_log::TaAnglesLog;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::ConstEtomoNumber;
use crate::imod::etomo::r#type::const_meta_data::ConstMetaData;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::file_type;
use crate::imod::etomo::r#type::meta_data::MetaData;
use crate::imod::etomo::r#type::panel_id::PanelId;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::process_result_display::ProcessResultDisplayHandle;
use crate::imod::etomo::r#type::processing_method::ProcessingMethod;
use crate::imod::etomo::ui::field::Field;
use crate::imod::etomo::ui::field_type::FieldType;
use crate::imod::etomo::util::utilities;

/// Java `public static final String rcsid`.
pub const RCSID: &str = "$Id$";

/// Java private static final `CENTER_TO_CENTER_THICKNESS_LABEL`.
const CENTER_TO_CENTER_THICKNESS_LABEL: &str = "Center to center thickness";
/// Java private static final `ADDITION_UNBINNED_DIAMETERS_TO_ADD`.
const ADDITION_UNBINNED_DIAMETERS_TO_ADD: &str = "Additional unbinned diameters to add ";
/// Java package-private static final `TILT_3D_FIND_LABEL`.
pub const TILT_3D_FIND_LABEL: &str = "Align and Build Tomogram";
/// Java private static final `PANEL_ID`.
const PANEL_ID: PanelId = PanelId::Tilt3dFind;

/// Java `final class Tilt3dFindPanel extends AbstractTiltPanel`.
pub struct Tilt3dFindPanel {
    /// The `AbstractTiltPanel` superclass.
    base: AbstractTiltPanel,
    /// Java private final `ltfCenterToCenterThickness`.
    ltf_center_to_center_thickness: Rc<LabeledTextField>,
    /// Java private final `ltfAdditionalDiameters`.
    ltf_additional_diameters: Rc<LabeledTextField>,

    /// Java private final `parent`.
    parent: Weak<dyn Tilt3dFindParent>,
    /// Java private final `extraButton` (a `Component`, possibly null).
    extra_button: Option<Rc<JComponent>>,
}

impl Deref for Tilt3dFindPanel {
    type Target = AbstractTiltPanel;
    fn deref(&self) -> &AbstractTiltPanel {
        &self.base
    }
}

impl Tilt3dFindPanel {
    /// Java private constructor `Tilt3dFindPanel(ApplicationManager, AxisID,
    /// DialogType, Tilt3dFindParent, Component)`.
    fn new(
        manager: &'static ApplicationManager,
        axis_id: AxisID,
        dialog_type: DialogType,
        parent: Weak<dyn Tilt3dFindParent>,
        extra_button: Option<Rc<JComponent>>,
    ) -> Rc<Tilt3dFindPanel> {
        Rc::new_cyclic(|this: &Weak<Tilt3dFindPanel>| {
            // super(manager, axisID, dialogType, null, PANEL_ID, false, parent)
            let generation_parent: Weak<dyn TomogramGenerationParent> = parent.clone();
            let base = AbstractTiltPanel::new(
                manager,
                axis_id,
                dialog_type,
                None,
                PANEL_ID,
                false,
                generation_parent,
                this,
            );
            // Field initializers.
            let ltf_center_to_center_thickness = LabeledTextField::new_field_type_string(
                FieldType::FloatingPoint,
                Some(&format!("{CENTER_TO_CENTER_THICKNESS_LABEL}: ")),
            );
            let ltf_additional_diameters = LabeledTextField::new_field_type_string(
                FieldType::Integer,
                Some(&format!("{ADDITION_UNBINNED_DIAMETERS_TO_ADD}: ")),
            );
            // Constructor body.
            // Change some labels.
            base.ltf_z_shift.set_label(Some("Added Z Shift: "));
            base.ltf_tomo_thickness.set_label(Some("Thickness: "));
            Tilt3dFindPanel {
                base,
                ltf_center_to_center_thickness,
                ltf_additional_diameters,
                parent,
                extra_button,
            }
        })
    }

    /// Java static `getInstance(ApplicationManager, AxisID, DialogType,
    /// Tilt3dFindParent, Component)`.
    pub fn get_instance(
        manager: &'static ApplicationManager,
        axis_id: AxisID,
        dialog_type: DialogType,
        parent: Weak<dyn Tilt3dFindParent>,
        extra_button: Option<Rc<JComponent>>,
    ) -> Rc<Tilt3dFindPanel> {
        let instance = Tilt3dFindPanel::new(manager, axis_id, dialog_type, parent, extra_button);
        instance.create_panel();
        instance.set_tool_tip_text();
        instance.base.add_listeners();
        instance
    }

    /// Java protected override `createPanel()`.  Completely different panel.
    pub fn create_panel(&self) {
        // Initialize
        self.base.initialize_panel();
        // Informational fields should not be editable.
        self.ltf_center_to_center_thickness.set_editable(false);
        self.ltf_additional_diameters.set_editable(false);
        // Local panels
        let pnl_a = SpacedPanel::get_instance_void();
        let pnl_buttons = JComponent::new_panel();
        // Root panel
        let pnl_root = self.base.get_root_panel();
        // Swing layout: pnlRoot.setBoxLayout(BoxLayout.Y_AXIS).
        pnl_root.add_component(&self.base.get_cpu_gpu_panel());
        pnl_root.add_labeled_text_field(&self.ltf_center_to_center_thickness);
        pnl_root.add_labeled_text_field(&self.ltf_additional_diameters);
        pnl_root.add_spaced_panel(&pnl_a);
        pnl_root.add_j_panel(&pnl_buttons);
        // Panel A
        // Swing layout: pnlA.setBoxLayout(BoxLayout.X_AXIS).
        pnl_a.add_labeled_text_field(&self.base.ltf_tomo_thickness);
        pnl_a.add_labeled_text_field(&self.base.ltf_z_shift);
        // Buttons panel
        // Java tests each button Component for null; the superclass getters
        // never return null.
        let button = self.base.get_tilt_button();
        pnl_buttons.add(&button);
        if let Some(extra_button) = &self.extra_button {
            pnl_buttons.add(extra_button);
        }
        let button = self.base.get_3dmod_tomogram_button();
        pnl_buttons.add(&button);
    }

    /// Java override `getParameters(TiltParam, boolean) throws
    /// NumberFormatException, InvalidParameterException, IOException`.
    /// Setting the usual parameters, then also setting input file, output
    /// file, and process name.
    pub fn get_parameters_tilt_param_boolean(
        &self,
        param: &mut TiltParam,
        do_validation: bool,
    ) -> Result<bool, TiltDisplayException> {
        // param.setThickness(ltfTomoThickness.getText());
        // param.setZShift(ltfZShift.getText());
        let manager: &'static dyn BaseManager = self.base.manager;
        if self
            .base
            .manager
            .get_state()
            .is_stack_using_newst_or_blend_3d_find_output(self.base.axis_id)
        {
            param.set_input_file(
                file_type::CLASS
                    .newst_or_blend_3d_find_output
                    .get_file_name(Some(manager), Some(self.base.axis_id))
                    .as_deref(),
            );
        } else {
            param.set_input_file(
                file_type::CLASS
                    .aligned_stack
                    .get_file_name(Some(manager), Some(self.base.axis_id))
                    .as_deref(),
            );
        }
        param.set_output_file(
            file_type::CLASS
                .tilt_3d_find_output
                .get_file_name(Some(manager), Some(self.base.axis_id))
                .as_deref(),
        );
        param.set_command_mode(Mode::Tilt3dFind);
        param.set_process_name(ProcessName::TILT_3D_FIND);
        if !self
            .base
            .get_parameters_tilt_param_boolean(param, do_validation)?
        {
            return Ok(false);
        }
        Ok(true)
    }

    /// Java override `setParameters(ConstTiltParam, boolean)`.
    pub fn set_parameters_const_tilt_param_boolean(
        &self,
        param: &dyn ConstTiltParam,
        initialize: bool,
    ) {
        self.base
            .set_parameters_const_tilt_param_boolean(param, initialize);
    }

    /// Java `setParameters(ConstTiltalignParam, boolean)`.  This mainly pulls
    /// data from the align log.
    pub fn set_parameters_const_tiltalign_param_boolean(
        &self,
        _param: &ConstTiltalignParam,
        initialize: bool,
    ) {
        // set center to center thickness and additional diameters
        let ta_angles_log = TaAnglesLog::get_instance(
            self.base.manager.get_property_user_dir().as_deref(),
            self.base.manager,
            self.base.axis_id,
        );
        let mut center_to_center_thickness: Option<ConstEtomoNumber> = None;
        match ta_angles_log.get_center_to_center_thickness() {
            Ok(value) => center_to_center_thickness = Some(value),
            // catch (Exception e) { e.printStackTrace(); }
            Err(e) => eprintln!("{e}"),
        }
        if let Some(center_to_center_thickness) = &center_to_center_thickness
            && center_to_center_thickness.is_valid()
        {
            self.ltf_center_to_center_thickness
                .set_text_const_etomo_number(Some(center_to_center_thickness));
        }
        let additional_diameters: i32 = 5;
        self.ltf_additional_diameters
            .set_text_int(additional_diameters);
        if initialize {
            // The first time the dialog is created tilt_3dfind.com is copied from
            // tilt.com and these values are calculated from align log values.
            if let Some(center_to_center_thickness) = &center_to_center_thickness
                && center_to_center_thickness.is_valid()
                && !center_to_center_thickness.is_null()
            {
                self.base
                    .ltf_tomo_thickness
                    .set_text_long(utilities::java_lang_math_round(
                        center_to_center_thickness.get_double()
                            + self.base.manager.calc_unbinned_bead_diameter_pixels()
                                * additional_diameters as f64,
                    ));
            }
            match ta_angles_log.get_incremental_shift_to_center() {
                Ok(value) => self
                    .base
                    .ltf_z_shift
                    .set_text_const_etomo_number(Some(&value)),
                // catch (Exception e) { e.printStackTrace(); }
                Err(e) => eprintln!("{e}"),
            }
        }
    }

    /// Java `setOverrideParameters(ConstMetaData)`.  Get values from meta data
    /// that override the com script values.
    pub fn set_override_parameters(&self, meta_data: &dyn ConstMetaData) {
        if meta_data.is_stack_3d_find_thickness_set(self.base.axis_id) {
            self.base.ltf_tomo_thickness.set_text_string(Some(
                &meta_data.get_stack_3d_find_thickness(self.base.axis_id),
            ));
        }
    }

    /// Java override `getParameters(MetaData) throws
    /// FortranInputSyntaxException`.
    pub fn get_parameters_meta_data(
        &self,
        meta_data: &MetaData,
    ) -> Result<(), FortranInputSyntaxException> {
        meta_data.set_tilt_parallel(self.base.axis_id, PANEL_ID, self.base.is_parallel_process());
        Ok(())
    }

    /// Java override `getParameters(SplittiltParam, boolean)`.
    pub fn get_parameters_splittilt_param_boolean(
        &self,
        param: &mut SplittiltParam,
        do_validation: bool,
    ) -> bool {
        if !self
            .base
            .get_parameters_splittilt_param_boolean(param, do_validation)
        {
            return false;
        }
        param.set_name(Some(&ProcessName::TILT_3D_FIND.to_string()));
        true
    }

    /// Java `tilt3dFindAction(ProcessResultDisplay, Deferred3dmodButton,
    /// Run3dmodMenuOptions)`.
    pub fn tilt3d_find_action(
        &self,
        process_result_display: Option<ProcessResultDisplayHandle>,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    ) {
        // Java passes a possibly-null Run3dmodMenuOptions; the manager takes the
        // value, so null is the default (no options set).
        self.base.manager.tilt3d_find_action(
            process_result_display,
            None,
            deferred_3dmod_button,
            run_3dmod_menu_options.unwrap_or_default(),
            Some(self as &dyn TiltDisplay),
            self.base.axis_id,
            self.base.dialog_type,
            self.base.get_run_method_for_process_interface(),
        );
    }

    /// Java override `setToolTipText()`.
    pub fn set_tool_tip_text(&self) {
        self.base.set_tool_tip_text();
        self.ltf_center_to_center_thickness.set_tool_tip_text(Some(
            "Used to calculate the thickness of the findbeads3d input tomogram.  From the \
             taAngles log.",
        ));
        self.ltf_additional_diameters
            .set_tool_tip_text(Some("Used to calculate the thickness of the findbeads3d."));
        self.base
            .ltf_tomo_thickness
            .set_tool_tip_text(Some(&format!(
                "Thickness of tomogram in unbinned pixels.  The default is calculated from \"{}\" \
             plus \"{}\" multipled by \"{}\".",
                CENTER_TO_CENTER_THICKNESS_LABEL,
                find_beads3d_panel::BEAD_SIZE_LABEL,
                ADDITION_UNBINNED_DIAMETERS_TO_ADD
            )));
        self.base.ltf_z_shift.set_tool_tip_text(Some(
            "Incremental unbinned shift needed to center range of fiducials in Z.  From the \
             taAngles log.",
        ));
        self.base.set_tilt_button_tooltip(Some(
            "If binning has changed, create a separate full aligned stack.  Then compute a \
             tomogram (tilt_3dfind.com).",
        ));
    }
}

impl AbstractTiltPanelVirtual for Tilt3dFindPanel {
    fn abstract_tilt_panel(&self) -> &AbstractTiltPanel {
        &self.base
    }

    /// Java override `tiltAction(ProcessResultDisplay, Deferred3dmodButton,
    /// Run3dmodMenuOptions, ProcessingMethod)`.
    fn tilt_action(
        &self,
        process_result_display: Option<ProcessResultDisplayHandle>,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
        tilt_processing_method: ProcessingMethod,
    ) {
        if let Some(parent) = self.parent.upgrade() {
            parent.tilt3d_find_action(
                process_result_display,
                deferred_3dmod_button,
                run_3dmod_menu_options,
                tilt_processing_method,
            );
        }
    }

    /// Java override `imodTomogramAction(Deferred3dmodButton,
    /// Run3dmodMenuOptions)`.
    fn imod_tomogram_action(
        &self,
        _deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    ) {
        self.base
            .manager
            .imod_file_type_axis_id_run3dmod_menu_options(
                &file_type::CLASS.tilt_3d_find_output,
                self.base.axis_id,
                run_3dmod_menu_options.unwrap_or_default(),
            );
    }
}

impl Expandable for Tilt3dFindPanel {
    /// Java inherited final `expand(ExpandButton)`.
    fn expand_expand_button(&self, button: &Rc<ExpandButton>) {
        self.base.expand_expand_button(button);
    }

    /// Java inherited final `expand(GlobalExpandButton)`.
    fn expand_global_expand_button(&self, button: &Rc<GlobalExpandButton>) {
        self.base.expand_global_expand_button(button);
    }
}

impl RadialParent for Tilt3dFindPanel {
    /// Java inherited final `isMultifilt()`.
    fn is_multifilt(&self) -> bool {
        self.base.is_multifilt()
    }

    /// Java inherited final `isCtf3d()`.
    fn is_ctf3d(&self) -> bool {
        self.base.is_ctf3d()
    }

    /// Java inherited `isAdvanced()`.
    fn is_advanced(&self) -> bool {
        self.base.is_advanced()
    }
}

impl TrialTiltParent for Tilt3dFindPanel {
    /// Java `getParameters(TiltParam, boolean)` (this class's override).
    fn get_parameters_tilt_param_boolean(
        &self,
        tilt_param: &mut TiltParam,
        do_validation: bool,
    ) -> Result<bool, TiltDisplayException> {
        Tilt3dFindPanel::get_parameters_tilt_param_boolean(self, tilt_param, do_validation)
    }

    /// Java `getParameters(SplittiltParam, boolean)` (this class's override).
    fn get_parameters_splittilt_param_boolean(
        &self,
        param: &mut SplittiltParam,
        do_validation: bool,
    ) -> bool {
        Tilt3dFindPanel::get_parameters_splittilt_param_boolean(self, param, do_validation)
    }

    /// Java inherited `getProcessingMethod()`.
    fn get_processing_method(&self) -> ProcessingMethod {
        self.base.get_processing_method()
    }
}

impl Run3dmodButtonContainer for Tilt3dFindPanel {
    /// Java inherited final `action(String, Deferred3dmodButton,
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

impl ProcessDisplay for Tilt3dFindPanel {
    /// Java cast `(TiltDisplay) display`: this panel is one (the process
    /// series hands it back to `tilt3dFindAction` / the reprojection).
    fn as_tilt_display(&self) -> Option<&dyn super::tilt_display::TiltDisplay> {
        Some(self)
    }
}

impl TiltDisplay for Tilt3dFindPanel {
    /// Java `getParameters(TiltParam, boolean)` (this class's override).
    fn get_parameters(
        &self,
        param: &mut TiltParam,
        do_validation: bool,
    ) -> Result<bool, TiltDisplayException> {
        Tilt3dFindPanel::get_parameters_tilt_param_boolean(self, param, do_validation)
    }

    /// Java `getParameters(SplittiltParam, boolean)` (this class's override).
    fn get_parameters_splittilt(&self, param: &mut SplittiltParam, do_validation: bool) -> bool {
        Tilt3dFindPanel::get_parameters_splittilt_param_boolean(self, param, do_validation)
    }

    /// Java inherited `@Deprecated allowTiltComSave()`.
    fn allow_tilt_com_save(&self) -> bool {
        self.base.allow_tilt_com_save()
    }

    /// Java inherited `setDebug(boolean)`.
    fn set_debug(&self, debug: bool) {
        self.base.set_debug(debug);
    }
}
