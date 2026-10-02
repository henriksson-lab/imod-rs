//! `IMOD/Etomo/src/etomo/ui/swing/AbstractTiltPanel.java`.
//!
//! Java `abstract class AbstractTiltPanel implements Expandable,
//! TrialTiltParent, Run3dmodButtonContainer, TiltDisplay, RadialParent`: the
//! tilt parameters and buttons shared by `TiltPanel` (tomogram generation) and
//! `Tilt3dFindPanel` (erase gold).
//!
//! **Representation** (see `ui.md`).  [`AbstractTiltPanel`] is the superclass
//! part; a subclass embeds it as field `base`, derefs to it, and implements
//! [`AbstractTiltPanelVirtual`] for the two abstract methods and the members
//! it overrides that this class calls on `this` (`updateDisplay()`,
//! `setAdvancedFieldDisplayer()`).  The subclass builds itself with
//! `Rc::new_cyclic` and passes the cyclic weak to [`AbstractTiltPanel::new`],
//! which hands it out where the Java hands out `this` (the panel header's
//! `Expandable`, the radial panel's `RadialParent`, the trial tilt panel's
//! `TrialTiltParent`, the 3dmod buttons' `Run3dmodButtonContainer`).  Because
//! Java's `this` is the subclass object, the interfaces this class implements
//! are implemented by the subclass: its `Expandable`, `RadialParent`,
//! `TrialTiltParent`, `Run3dmodButtonContainer` and `TiltDisplay` impls call
//! the methods of the same names here (or the subclass's override).
//!
//! Java overloads carry the parameter-type suffix (`ui.md`):
//! `getParameters(MetaData)` is [`AbstractTiltPanel::get_parameters_meta_data`],
//! `getParameters(TiltParam, boolean)` is
//! [`AbstractTiltPanel::get_parameters_tilt_param_boolean`], and so on.  A
//! subclass override is an inherent method of the same name on the subclass,
//! which shadows this one when called on the subclass.

use std::cell::Cell;
use std::rc::{Rc, Weak};

use super::check_box::CheckBox;
use super::check_text_field::CheckTextField;
use super::cpu_gpu_panel::CpuGpuPanel;
use super::deferred_3dmod_button::Deferred3dmodButton;
use super::etched_border::EtchedBorder;
use super::expand_button::ExpandButton;
use super::expandable::Expandable;
use super::global_expand_button::GlobalExpandButton;
use super::labeled_text_field::LabeledTextField;
use super::multi_line_button::MultiLineButton;
use super::multifilt_panel::MultifiltPanel;
use super::panel_header::PanelHeader;
use super::process_interface::ProcessInterface;
use super::radial_panel::RadialPanel;
use super::radial_parent::RadialParent;
use super::run_3dmod_button::Run3dmodButton;
use super::run_3dmod_button_container::Run3dmodButtonContainer;
use super::spaced_panel::SpacedPanel;
use super::spinner::Spinner;
use super::tilt_display::{TiltDisplay, TiltDisplayException};
use super::tomogram_generation_dialog;
use super::tomogram_generation_parent::TomogramGenerationParent;
use super::trial_tilt_panel::TrialTiltPanel;
use super::trial_tilt_parent::TrialTiltParent;
use super::ui_harness;
use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::const_tilt_param::ConstTiltParam;
use crate::imod::etomo::comscript::fortran_input_syntax_exception::FortranInputSyntaxException;
use crate::imod::etomo::comscript::splittilt_param::SplittiltParam;
use crate::imod::etomo::comscript::tilt_param::{self, TiltParam};
use crate::imod::etomo::jdk::{ActionEvent, ActionListener, FocusListener, JComponent};
use crate::imod::etomo::logic::converter;
use crate::imod::etomo::logic::tomogram_tool::TomogramTool;
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::storage::autodoc::autodoc_factory;
use crate::imod::etomo::storage::log_file::LogFileError;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::{
    ConstEtomoNumber, Type, java_lang_double_value_of, java_lang_string_matches_whitespace,
};
use crate::imod::etomo::r#type::const_meta_data::ConstMetaData;
use crate::imod::etomo::r#type::const_panel_header_settings::ConstPanelHeaderSettings;
use crate::imod::etomo::r#type::const_panel_header_state::ConstPanelHeaderState;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::r#type::meta_data::MetaData;
use crate::imod::etomo::r#type::panel_id::PanelId;
use crate::imod::etomo::r#type::process_result_display::ProcessResultDisplayHandle;
use crate::imod::etomo::r#type::processing_method::ProcessingMethod;
use crate::imod::etomo::r#type::recon_screen_state::ReconScreenState;
use crate::imod::etomo::r#type::tomogram_state::TomogramState;
use crate::imod::etomo::r#type::view_type::ViewType;
use crate::imod::etomo::ui::field::Field;
use crate::imod::etomo::ui::field_displayer::FieldDisplayer;
use crate::imod::etomo::ui::field_type::FieldType;
use crate::imod::etomo::util::invalid_parameter_exception::InvalidParameterException;
use crate::imod::etomo::util::utilities;

/// Java private static final `BACK_PROJECTION_HEADER`.
const BACK_PROJECTION_HEADER: &str = "Tilt";
/// Java private static final `PARAMETERS_HEADER`.
const PARAMETERS_HEADER: &str = "Tilt Parameters";
/// Java private static final `SUPER_SAMPLE_FACTOR_LABEL`.
const SUPER_SAMPLE_FACTOR_LABEL: &str = "Super-sample by";

/// The members `AbstractTiltPanel` declares abstract or that a subclass
/// overrides and this class calls on `this`.  The supertraits are the
/// interfaces the Java class implements (so the subclass object can be handed
/// out as each of them).  Defaults are the `AbstractTiltPanel` bodies.
pub trait AbstractTiltPanelVirtual:
    Expandable + TrialTiltParent + Run3dmodButtonContainer + TiltDisplay + RadialParent
{
    /// The embedded `AbstractTiltPanel` (the Java superclass part).
    fn abstract_tilt_panel(&self) -> &AbstractTiltPanel;

    /// Java `abstract void tiltAction(ProcessResultDisplay, Deferred3dmodButton,
    /// Run3dmodMenuOptions, ProcessingMethod)`.
    fn tilt_action(
        &self,
        process_result_display: Option<ProcessResultDisplayHandle>,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
        tilt_processing_method: ProcessingMethod,
    );

    /// Java `abstract void imodTomogramAction(Deferred3dmodButton,
    /// Run3dmodMenuOptions)`.
    fn imod_tomogram_action(
        &self,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    );

    /// Java protected `updateDisplay()` (overridden by `TiltPanel`).
    fn update_display(&self) {
        self.abstract_tilt_panel().update_display_super();
    }

    /// Java protected `setAdvancedFieldDisplayer()` (overridden by
    /// `TiltPanel`).
    fn set_advanced_field_displayer(&self) {
        self.abstract_tilt_panel()
            .set_advanced_field_displayer_super();
    }
}

/// Java `abstract class AbstractTiltPanel`: its fields and non-abstract
/// methods.
pub struct AbstractTiltPanel {
    /// Java private final `pnlRoot = SpacedPanel.getInstance()`.
    pnl_root: Rc<SpacedPanel>,
    // Keep components with listeners private.
    /// Java private final `btn3dmodTomogram`.
    btn_3dmod_tomogram: Rc<Run3dmodButton>,
    /// Java private final `actionListener` (`TiltActionListener`).
    action_listener: ActionListener,
    /// Java private final `pnlBody = new JPanel()`.
    pnl_body: Rc<JComponent>,
    /// Java private final `ltfTomoWidth`.
    ltf_tomo_width: Rc<LabeledTextField>,
    /// Java package-private final `ltfTomoThickness`.
    pub ltf_tomo_thickness: Rc<LabeledTextField>,
    /// Java private final `ltfXAxisTilt`.
    ltf_x_axis_tilt: Rc<LabeledTextField>,
    /// Java private final `ltfExtraExcludeList`.
    ltf_extra_exclude_list: Rc<LabeledTextField>,
    /// Java private final `ltfXShift`.
    ltf_x_shift: Rc<LabeledTextField>,
    /// Java private final `ltfTomoHeight`.
    ltf_tomo_height: Rc<LabeledTextField>,
    /// Java private final `ltfYShift`.
    ltf_y_shift: Rc<LabeledTextField>,

    /// Java private final `radialPanel`.
    radial_panel: Rc<RadialPanel>,
    /// Java private final `trialPanel = SpacedPanel.getInstance()`.
    trial_panel: Rc<SpacedPanel>,
    /// Java private final `pnlButton = SpacedPanel.getInstance(true)`.
    pnl_button: Rc<SpacedPanel>,
    /// Java private final `pnlReconWithSuperSampling = new JPanel()`.
    pnl_recon_with_super_sampling: Rc<JComponent>,
    /// Java private final `cbSuperSampleFactor`.
    cb_super_sample_factor: Rc<CheckBox>,
    /// Java private final `spSuperSampleFactor`.
    sp_super_sample_factor: Rc<Spinner>,
    /// Java private final `cbExpandInputLines`.
    cb_expand_input_lines: Rc<CheckBox>,

    // Protected variables can be modified by plugin child classes. Assuming that
    // variables not touched by AbstractTiltPanel and TiltPanel are wrong and fix
    // them in every update function.
    /// Java protected final `ctfLog`.
    pub ctf_log: Rc<CheckTextField>,
    /// Java protected final `ltfTiltAngleOffset`.
    pub ltf_tilt_angle_offset: Rc<LabeledTextField>,
    /// Java protected final `ltfLogDensityScaleFactor`.
    pub ltf_log_density_scale_factor: Rc<LabeledTextField>,
    /// Java protected final `ltfLogDensityScaleOffset`.
    pub ltf_log_density_scale_offset: Rc<LabeledTextField>,
    /// Java protected final `ltfLinearDensityScaleFactor`.
    pub ltf_linear_density_scale_factor: Rc<LabeledTextField>,
    /// Java protected final `ltfLinearDensityScaleOffset`.
    pub ltf_linear_density_scale_offset: Rc<LabeledTextField>,
    /// Java protected final `ltfZShift`.
    pub ltf_z_shift: Rc<LabeledTextField>,
    /// Java protected final `cbUseLocalAlignment`.
    pub cb_use_local_alignment: Rc<CheckBox>,
    /// Java protected final `cbUseZFactors`.
    pub cb_use_z_factors: Rc<CheckBox>,

    /// Java private final `header`.
    header: Rc<PanelHeader>,
    /// Java package-private final `manager`.
    pub manager: &'static ApplicationManager,
    /// Java package-private final `axisID`.
    pub axis_id: AxisID,
    /// Java package-private final `dialogType`.
    pub dialog_type: DialogType,
    /// Java private final `trialTiltPanel`.
    trial_tilt_panel: Rc<TrialTiltPanel>,
    // Keep components with listeners private.
    /// Java private final `btnTilt`.
    btn_tilt: Rc<Run3dmodButton>,
    /// Java private final `btnDeleteStack`.
    btn_delete_stack: Rc<MultiLineButton>,
    /// Java private final `panelId`.
    panel_id: PanelId,
    /// Java package-private final `listenForFieldChanges`.
    pub listen_for_field_changes: bool,
    /// Java protected final `advancedFieldDisplayer`.  advancedFieldDisplayer's
    /// display function puts the dialog in advanced mode. Add to advanced
    /// fields that are validated.
    pub advanced_field_displayer: Option<Rc<dyn FieldDisplayer>>,
    /// Java private final `cpuGpuPanel`.
    cpu_gpu_panel: Rc<CpuGpuPanel>,
    /// Java private final `parent`.
    parent: Weak<dyn TomogramGenerationParent>,

    /// Java private `madeZFactors`.
    made_z_factors: Cell<bool>,
    /// Java private `newstFiducialessAlignment`.
    newst_fiducialess_alignment: Cell<bool>,
    /// Java private `usedLocalAlignments`.
    used_local_alignments: Cell<bool>,

    /// Java `this` (the subclass object), for the virtual calls.
    this: Weak<dyn AbstractTiltPanelVirtual>,
    /// Java `this` as the `Run3dmodButtonContainer` given to `btnTilt`.
    this_container: Weak<dyn Run3dmodButtonContainer>,
}

impl AbstractTiltPanel {
    /// Java package-private constructor `AbstractTiltPanel(ApplicationManager,
    /// AxisID, DialogType, GlobalExpandButton, PanelId, boolean,
    /// TomogramGenerationParent)`.  `this` is the subclass being built
    /// (`Rc::new_cyclic`'s weak).  `globalAdvancedButton` may be null in the
    /// Java (`Tilt3dFindPanel` passes null).
    ///
    /// Backward compatibility functionality - if the metadata binning is
    /// missing get binning from newst.
    #[allow(clippy::too_many_arguments)]
    pub fn new<T: AbstractTiltPanelVirtual + 'static>(
        manager: &'static ApplicationManager,
        axis_id: AxisID,
        dialog_type: DialogType,
        global_advanced_button: Option<&Rc<GlobalExpandButton>>,
        panel_id: PanelId,
        listen_for_field_changes: bool,
        parent: Weak<dyn TomogramGenerationParent>,
        this: &Weak<T>,
    ) -> AbstractTiltPanel {
        let this_virtual: Weak<dyn AbstractTiltPanelVirtual> = this.clone();
        let this_container: Weak<dyn Run3dmodButtonContainer> = this.clone();
        // Field initializers, in declaration order.
        let pnl_root = SpacedPanel::get_instance_void();
        let btn_3dmod_tomogram =
            Run3dmodButton::get_3dmod_instance_string_run_3dmod_button_container(
                Some("View Tomogram In 3dmod"),
                Some(this_container.clone()),
            );
        // Java `new TiltActionListener(this)`: `adaptee.action(...)` is this
        // class's final `action`.
        let adaptee = this_virtual.clone();
        let action_listener: ActionListener = Rc::new(move |event: &ActionEvent| {
            let Some(adaptee) = adaptee.upgrade() else {
                return;
            };
            adaptee
                .abstract_tilt_panel()
                .action_string_deferred_3dmod_button_run_3dmod_menu_options(
                    event.get_action_command().unwrap_or(""),
                    None,
                    None,
                );
        });
        let pnl_body = JComponent::new_panel();
        let ltf_tomo_width = LabeledTextField::new_field_type_string(
            FieldType::Integer,
            Some("Tomogram width in X: "),
        );
        let ltf_tomo_thickness = LabeledTextField::get_numeric_instance_string_type(
            Some("Tomogram thickness in Z: "),
            Type::Integer,
        );
        let ltf_x_axis_tilt =
            LabeledTextField::get_numeric_instance_string_type(Some("X axis tilt: "), Type::Double);
        let ltf_extra_exclude_list = LabeledTextField::new_field_type_string(
            FieldType::IntegerList,
            Some("Extra views to exclude: "),
        );
        let ltf_x_shift =
            LabeledTextField::new_field_type_string(FieldType::FloatingPoint, Some("X shift: "));
        let ltf_tomo_height = LabeledTextField::new_field_type_string(
            FieldType::Integer,
            Some("Tomogram height in Y: "),
        );
        let ltf_y_shift =
            LabeledTextField::new_field_type_string(FieldType::Integer, Some(" Y shift: "));
        let trial_panel = SpacedPanel::get_instance_void();
        let pnl_button = SpacedPanel::get_instance_boolean(true);
        let pnl_recon_with_super_sampling = JComponent::new_panel();
        let cb_super_sample_factor = CheckBox::new_string(Some(SUPER_SAMPLE_FACTOR_LABEL));
        let sp_super_sample_factor =
            Spinner::get_instance_string_int_int_int(Some(SUPER_SAMPLE_FACTOR_LABEL), 2, 2, 8);
        let cb_expand_input_lines = CheckBox::new_string(Some("Super-sample input also"));
        let ctf_log = CheckTextField::get_numeric_instance(
            FieldType::FloatingPoint,
            "Take logarithm of densities with offset: ",
            Some(Type::Double),
        );
        let ltf_tilt_angle_offset = LabeledTextField::get_numeric_instance_string_type(
            Some("Tilt angle offset: "),
            Type::Double,
        );
        let ltf_log_density_scale_factor = LabeledTextField::get_numeric_instance_string_type(
            Some("Logarithm density scaling factor: "),
            Type::Double,
        );
        let ltf_log_density_scale_offset =
            LabeledTextField::get_numeric_instance_string_type(Some(" Offset: "), Type::Double);
        let ltf_linear_density_scale_factor = LabeledTextField::get_numeric_instance_string_type(
            Some("Linear density scaling factor: "),
            Type::Double,
        );
        let ltf_linear_density_scale_offset =
            LabeledTextField::get_numeric_instance_string_type(Some(" Offset: "), Type::Double);
        let ltf_z_shift =
            LabeledTextField::get_numeric_instance_string_type(Some(" Z shift: "), Type::Double);
        let cb_use_local_alignment = CheckBox::new_string(Some("Use local alignments"));
        let cb_use_z_factors = CheckBox::new_string(Some("Use Z factors"));

        // Constructor body.
        let advanced_field_displayer: Option<Rc<dyn FieldDisplayer>> =
            global_advanced_button.map(|button| button.clone() as Rc<dyn FieldDisplayer>);
        let expandable: Weak<dyn Expandable> = this.clone();
        let header =
            PanelHeader::get_advanced_basic_instance_string_expandable_dialog_type_global_expand_button(
                Some(BACK_PROJECTION_HEADER),
                Some(expandable),
                Some(dialog_type),
                global_advanced_button.cloned(),
            );
        let radial_parent: Weak<dyn RadialParent> = this.clone();
        let base_manager: &'static dyn BaseManager = manager;
        let radial_panel =
            RadialPanel::get_instance(base_manager, axis_id, panel_id, radial_parent);
        // mediator = manager.getProcessingMethodMediator(axisID);
        let trial_tilt_parent: Weak<dyn TrialTiltParent> = this.clone();
        let trial_tilt_panel =
            TrialTiltPanel::get_instance(manager, axis_id, dialog_type, trial_tilt_parent);
        let display_factory = manager.get_process_result_display_factory(axis_id);
        // Java casts `(Run3dmodButton) displayFactory.getTilt(dialogType)` and
        // `(MultiLineButton) displayFactory.getDeleteAlignedStack()`; the
        // factory returns the concrete buttons.
        let btn_tilt = display_factory.get_tilt(dialog_type);
        let btn_delete_stack = display_factory.get_delete_aligned_stack();
        let cpu_gpu_panel = CpuGpuPanel::get_instance(base_manager, axis_id, panel_id, 3, true, -1);
        AbstractTiltPanel {
            pnl_root,
            btn_3dmod_tomogram,
            action_listener,
            pnl_body,
            ltf_tomo_width,
            ltf_tomo_thickness,
            ltf_x_axis_tilt,
            ltf_extra_exclude_list,
            ltf_x_shift,
            ltf_tomo_height,
            ltf_y_shift,
            radial_panel,
            trial_panel,
            pnl_button,
            pnl_recon_with_super_sampling,
            cb_super_sample_factor,
            sp_super_sample_factor,
            cb_expand_input_lines,
            ctf_log,
            ltf_tilt_angle_offset,
            ltf_log_density_scale_factor,
            ltf_log_density_scale_offset,
            ltf_linear_density_scale_factor,
            ltf_linear_density_scale_offset,
            ltf_z_shift,
            cb_use_local_alignment,
            cb_use_z_factors,
            header,
            manager,
            axis_id,
            dialog_type,
            trial_tilt_panel,
            btn_tilt,
            btn_delete_stack,
            panel_id,
            listen_for_field_changes,
            advanced_field_displayer,
            cpu_gpu_panel,
            parent,
            made_z_factors: Cell::new(false),
            newst_fiducialess_alignment: Cell::new(false),
            used_local_alignments: Cell::new(false),
            this: this_virtual,
            this_container,
        }
    }

    /// The subclass object (Java `this` seen through a virtual call).
    fn this(&self) -> Option<Rc<dyn AbstractTiltPanelVirtual>> {
        self.this.upgrade()
    }

    // <p>Updates done</p>

    /// Java final `initializePanel()`.
    pub fn initialize_panel(&self) {
        self.btn_tilt
            .set_container(Some(self.this_container.clone()));
        let deferred: Rc<dyn Deferred3dmodButton> = self.btn_3dmod_tomogram.clone();
        self.btn_tilt
            .set_deferred_3dmod_button_deferred_3dmod_button(Some(deferred));
    }

    /// Java `createPanel()` (overridden by both subclasses; `TiltPanel` calls
    /// this as `super.createPanel()`).
    pub fn create_panel(&self) {
        // Initialize
        self.set_advanced_field_displayer();
        self.initialize_panel();
        self.ltf_linear_density_scale_factor
            .set_text_string(Some(tilt_param::LINEAR_SCALE_FACTOR_DEFAULT));
        self.ltf_linear_density_scale_offset
            .set_text_string(Some(tilt_param::LINEAR_SCALE_OFFSET_DEFAULT));
        self.ltf_tomo_width.set_preferred_width(163);
        self.ltf_tomo_height.set_preferred_width(159);
        // local panels
        let pnl_log_density = JComponent::new_panel();
        let pnl_linear_density = JComponent::new_panel();
        let pnl_check_box = JComponent::new_panel();
        let pnl_x = JComponent::new_panel();
        let pnl_y = JComponent::new_panel();
        let pnl_z = JComponent::new_panel();
        let pnl_use_local_alignment = JComponent::new_panel();
        let pnl_use_z_factors = JComponent::new_panel();
        let pnl_supersample_by_spinner = JComponent::new_panel();
        // Root panel
        // Swing layout: pnlRoot.setBoxLayout(BoxLayout.Y_AXIS); untitled etched
        // border.
        self.pnl_root.add_panel_header(&self.header);
        self.pnl_root.add_j_panel(&self.pnl_body);
        // Body panel
        // Swing layout: pnlBody BoxLayout Y_AXIS; rigid areas (x0_y5, x0_y3)
        // between the rows below.
        self.pnl_body.add(&self.cpu_gpu_panel.get_component());
        self.pnl_body.add(&self.ctf_log.get_root_component());
        self.pnl_body.add(&pnl_log_density);
        self.pnl_body.add(&pnl_linear_density);
        self.pnl_body.add(&pnl_x);
        self.pnl_body.add(&pnl_y);
        self.pnl_body.add(&pnl_z);
        self.pnl_body.add(&self.ltf_x_axis_tilt.get_container());
        self.pnl_body
            .add(&self.ltf_tilt_angle_offset.get_container());
        self.pnl_body.add(&self.pnl_recon_with_super_sampling);
        self.pnl_body.add(&self.radial_panel.get_root());
        self.pnl_body
            .add(&self.ltf_extra_exclude_list.get_container());
        self.pnl_body.add(&pnl_check_box);
        self.pnl_body.add(&self.trial_panel.get_container());
        self.pnl_body.add(&self.pnl_button.get_container());
        // Log density panel
        // Swing layout: pnlLogDensity BoxLayout X_AXIS.
        pnl_log_density.add(&self.ltf_log_density_scale_factor.get_container());
        pnl_log_density.add(&self.ltf_log_density_scale_offset.get_container());
        // Linear density panel
        // Swing layout: pnlLinearDensity BoxLayout X_AXIS.
        pnl_linear_density.add(&self.ltf_linear_density_scale_factor.get_container());
        pnl_linear_density.add(&self.ltf_linear_density_scale_offset.get_container());
        // X panel
        // Swing layout: pnlX BoxLayout X_AXIS; rigid area x5_y0 between.
        pnl_x.add(&self.ltf_tomo_width.get_container());
        pnl_x.add(&self.ltf_x_shift.get_container());
        // Y panel
        // Swing layout: pnlY BoxLayout X_AXIS; rigid area x5_y0 between.
        pnl_y.add(&self.ltf_tomo_height.get_container());
        pnl_y.add(&self.ltf_y_shift.get_container());
        // Z panel
        // Swing layout: pnlZ BoxLayout X_AXIS; rigid area x5_y0 between.
        pnl_z.add(&self.ltf_tomo_thickness.get_container());
        pnl_z.add(&self.ltf_z_shift.get_container());
        // TODO
        // Reconstruction with supersampling of pixels panel
        // Swing layout: pnlReconWithSuperSampling BoxLayout Y_AXIS.
        self.pnl_recon_with_super_sampling.set_border_title(
            EtchedBorder::new(Some("Reconstruction with super-sampling of pixels"))
                .get_border()
                .get_title()
                .as_deref(),
        );
        self.pnl_recon_with_super_sampling
            .add(&pnl_supersample_by_spinner);
        // Supersample by spinner panel
        // Swing layout: pnlSupersampleBySpinner BoxLayout X_AXIS; rigid area
        // x5_y0 before cbExpandInputLines; horizontal glue at the end.
        pnl_supersample_by_spinner.add(&self.cb_super_sample_factor.get_component());
        pnl_supersample_by_spinner.add(&self.sp_super_sample_factor.get_component());
        pnl_supersample_by_spinner.add(&self.cb_expand_input_lines.get_component());
        // Check box panel
        // Swing layout: pnlCheckBox BoxLayout Y_AXIS.
        pnl_check_box.add(&pnl_use_local_alignment);
        pnl_check_box.add(&pnl_use_z_factors);
        // UseLocalAlignment
        // Swing layout: pnlUseLocalAlignment BoxLayout X_AXIS, horizontal glue.
        pnl_use_local_alignment.add(&self.cb_use_local_alignment.get_component());
        // UseZFactors
        // Swing layout: pnlUseZFactors BoxLayout X_AXIS, horizontal glue.
        pnl_use_z_factors.add(&self.cb_use_z_factors.get_component());
        // Trial panel
        // Swing layout: trialPanel.setBoxLayout(BoxLayout.X_AXIS).
        self.trial_panel
            .add_component(&self.trial_tilt_panel.get_component());
        // Button panel
        // Swing layout: pnlButton.setBoxLayout(BoxLayout.X_AXIS).
        self.pnl_button.add_multi_line_button(&self.btn_tilt);
        self.pnl_button
            .add_multi_line_button(&self.btn_3dmod_tomogram);
        self.pnl_button
            .add_multi_line_button(&self.btn_delete_stack);

        self.update_display();
    }

    /// Java final `getRootPanel()`.
    pub fn get_root_panel(&self) -> Rc<SpacedPanel> {
        self.pnl_root.clone()
    }

    /// Java private final `getPanelId()`.
    fn get_panel_id(&self) -> PanelId {
        self.panel_id
    }

    /// Java `addAxisTiltFocusListener(FocusListener)`.
    pub fn add_axis_tilt_focus_listener(&self, listener: FocusListener) {
        self.ltf_x_axis_tilt.add_focus_listener(listener);
    }

    /// Java `removeAxisTiltFocusListener(FocusListener)`.
    pub fn remove_axis_tilt_focus_listener(&self, listener: &FocusListener) {
        self.ltf_x_axis_tilt.remove_focus_listener(listener);
    }

    /// Java `addUseLocalAlignmentActionListener(ActionListener)`.
    pub fn add_use_local_alignment_action_listener(&self, listener: ActionListener) {
        self.cb_use_local_alignment
            .add_action_listener(Some(listener));
    }

    /// Java `removeUseLocalAlignmentActionListener(ActionListener)`.
    pub fn remove_use_local_alignment_action_listener(&self, listener: &ActionListener) {
        self.cb_use_local_alignment.remove_action_listener(listener);
    }

    /// Java `addUseZFactorsActionListener(ActionListener)`.
    pub fn add_use_z_factors_action_listener(&self, listener: ActionListener) {
        self.cb_use_z_factors.add_action_listener(Some(listener));
    }

    /// Java `removeUseZFactorsActionListener(ActionListener)`.
    pub fn remove_use_z_factors_action_listener(&self, listener: &ActionListener) {
        self.cb_use_z_factors.remove_action_listener(listener);
    }

    /// Java final `getXAxisTilt()`.
    pub fn get_x_axis_tilt(&self) -> Option<String> {
        self.ltf_x_axis_tilt.get_text_void()
    }

    /// Java final `getTiltButton()`.
    pub fn get_tilt_button(&self) -> Rc<JComponent> {
        self.btn_tilt.get_component()
    }

    /// Java final `get3dmodTomogramButton()`.
    pub fn get_3dmod_tomogram_button(&self) -> Rc<JComponent> {
        self.btn_3dmod_tomogram.get_component()
    }

    /// Java final `getCpuGpuPanel()`.
    pub fn get_cpu_gpu_panel(&self) -> Rc<JComponent> {
        self.cpu_gpu_panel.get_component()
    }

    /// Java final `setTiltButtonTooltip(String)`.
    pub fn set_tilt_button_tooltip(&self, tooltip: Option<&str>) {
        self.btn_tilt.set_tool_tip_text(tooltip);
    }

    /// Java `addListeners()` (`TiltPanel` overrides it and calls super).
    pub fn add_listeners(&self) {
        self.btn_tilt
            .add_action_listener(self.action_listener.clone());
        self.btn_3dmod_tomogram
            .add_action_listener(self.action_listener.clone());
        self.btn_delete_stack
            .add_action_listener(self.action_listener.clone());
        self.ctf_log
            .add_action_listener(self.action_listener.clone());
        self.cb_super_sample_factor
            .add_action_listener(Some(self.action_listener.clone()));
        // If the child class has field listeners, it must set listenForFieldChanges to
        // true. Otherwise changes in these fields will be ignored.
        if self.listen_for_field_changes {
            self.cb_use_local_alignment
                .add_action_listener(Some(self.action_listener.clone()));
            self.cb_use_z_factors
                .add_action_listener(Some(self.action_listener.clone()));
        }
    }

    /// Java `setFilterTypeActionListener(MultifiltPanel)`.  Allow radial panel
    /// to react to the multifilt panel which also contains filter selections.
    pub fn set_filter_type_action_listener(&self, multifilt_panel: &Rc<MultifiltPanel>) {
        // Java `multifiltPanel.addActionListener(radialPanel)`: the radial panel
        // is the ActionListener.
        let radial_panel = Rc::downgrade(&self.radial_panel);
        multifilt_panel.add_action_listener(Rc::new(move |event: &ActionEvent| {
            if let Some(radial_panel) = radial_panel.upgrade() {
                radial_panel.action_performed(event);
            }
        }));
        self.radial_panel.set_multifilt_filter_type(Some(
            multifilt_panel.clone() as Rc<dyn super::filter_type::FilterType>
        ));
    }

    /// Java `getRoot()` (`TiltPanel` overrides it).
    pub fn get_root(&self) -> Rc<JComponent> {
        self.pnl_root.get_container()
    }

    /// Java `@Deprecated allowTiltComSave()` (8/3/2018 See TiltDisplay).
    pub fn allow_tilt_com_save(&self) -> bool {
        true
    }

    /// Java public final `expand(GlobalExpandButton)`; empty.
    pub fn expand_global_expand_button(&self, _button: &Rc<GlobalExpandButton>) {}

    /// Java public final `expand(ExpandButton)`.
    pub fn expand_expand_button(&self, button: &Rc<ExpandButton>) {
        // Java `if (header != null)`: header is final and always set.
        if self.header.equals_open_close(button) {
            self.pnl_body.set_visible(button.is_expanded());
        } else if self.header.equals_advanced_basic(button) {
            self.update_display();
        }
        ui_harness::with(|harness| {
            harness.pack_axis_id_base_manager(Some(self.axis_id), Some(self.manager))
        });
    }

    /// Java `isAdvanced()` (implements `RadialParent`).
    pub fn is_advanced(&self) -> bool {
        self.header.is_advanced()
    }

    /// Java final `msgMethodChanged()`.
    pub fn msg_method_changed(&self) {
        self.update_display();
        self.radial_panel.msg_method_changed();
        self.cpu_gpu_panel
            .msg_processing_method_changed(!self.is_back_projection(), !self.is_ctf3d());
    }

    /// Java `registerProcessingMethodMediator()`.
    pub fn register_processing_method_mediator(&self) {
        self.cpu_gpu_panel.register_procesing_method_mediator();
    }

    /// Java `deregisterProcessingMethodMediator()`.
    pub fn deregister_processing_method_mediator(&self) {
        self.cpu_gpu_panel.deregister_procesing_method_mediator();
    }

    /// Java protected `setAdvancedFieldDisplayer()`, dispatched to the
    /// subclass.
    pub fn set_advanced_field_displayer(&self) {
        match self.this() {
            Some(this) => this.set_advanced_field_displayer(),
            None => self.set_advanced_field_displayer_super(),
        }
    }

    /// The `AbstractTiltPanel` body of Java `setAdvancedFieldDisplayer()`.
    /// Allows advanced fields to be displayed if they get a popup error
    /// message.  Only needed for fields that are validated.
    pub fn set_advanced_field_displayer_super(&self) {
        let displayer = || self.advanced_field_displayer.clone();
        self.ltf_log_density_scale_offset
            .set_overridable_field_displayers(None, displayer());
        self.ltf_log_density_scale_factor
            .set_overridable_field_displayers(None, displayer());
        self.ltf_linear_density_scale_offset
            .set_overridable_field_displayers(None, displayer());
        self.ltf_linear_density_scale_factor
            .set_overridable_field_displayers(None, displayer());
        self.ltf_tomo_width
            .set_overridable_field_displayers(None, displayer());
        self.ltf_tomo_height
            .set_overridable_field_displayers(None, displayer());
        self.ltf_y_shift
            .set_overridable_field_displayers(None, displayer());
        self.ltf_x_shift
            .set_overridable_field_displayers(None, displayer());
        // Z shift is not always an advanced field.
        self.ltf_tilt_angle_offset
            .set_overridable_field_displayers(None, displayer());
        self.ltf_extra_exclude_list
            .set_overridable_field_displayers(None, displayer());
    }

    /// Java private `resetToNonpluginState()`.  Normalize protected fields
    /// when the plugin in not active.  Do this with any protected fields.  Call
    /// this method BEFORE adjusting visibility or enabledness.  It is an
    /// extremely blunt instrument.
    fn reset_to_nonplugin_state(&self) {
        if !self.is_method_plugin() {
            self.ctf_log.set_visible(true);
            self.ctf_log.set_text_field_visible(true);
            self.ctf_log.switch_labels(false);
            self.ctf_log.switch_tooltips(false);
            self.ltf_tilt_angle_offset.set_visible(true);
            self.ltf_log_density_scale_factor.set_visible(true);
            self.ltf_log_density_scale_offset.set_visible(true);
            self.ltf_linear_density_scale_factor.set_visible(true);
            self.ltf_linear_density_scale_offset.set_visible(true);
            self.ltf_z_shift.set_visible(true);
            self.cb_use_local_alignment.set_visible(true);
            self.cb_use_local_alignment.switch_tooltips(false);
            self.cb_use_z_factors.set_visible(true);
            self.cb_use_z_factors.switch_tooltips(false);
        }
    }

    /// Java final `done()`.
    pub fn done(&self) {
        self.btn_tilt.remove_action_listener(&self.action_listener);
        self.btn_delete_stack
            .remove_action_listener(&self.action_listener);
        self.trial_tilt_panel.done();
        self.cpu_gpu_panel.done();
    }

    /// Java final `isBackProjection()`.  A parent that is gone answers false
    /// (Java's parent is never collected while the panel exists).
    pub fn is_back_projection(&self) -> bool {
        self.parent
            .upgrade()
            .is_some_and(|parent| parent.is_back_projection())
    }

    /// Java public final `isMultifilt()` (implements `RadialParent`).
    pub fn is_multifilt(&self) -> bool {
        self.parent
            .upgrade()
            .is_some_and(|parent| parent.is_multifilt())
    }

    /// Java public final `isCtf3d()` (implements `RadialParent`).
    pub fn is_ctf3d(&self) -> bool {
        self.parent
            .upgrade()
            .is_some_and(|parent| parent.is_ctf3d())
    }

    /// Java final `isSirt()`.
    pub fn is_sirt(&self) -> bool {
        self.parent.upgrade().is_some_and(|parent| parent.is_sirt())
    }

    /// Java public `isMethodPlugin()`.
    pub fn is_method_plugin(&self) -> bool {
        self.parent
            .upgrade()
            .is_some_and(|parent| parent.is_method_plugin())
    }

    /// Java protected `updateDisplay()`, dispatched to the subclass.
    pub fn update_display(&self) {
        match self.this() {
            Some(this) => this.update_display(),
            None => self.update_display_super(),
        }
    }

    /// The `AbstractTiltPanel` body of Java `updateDisplay()`.
    pub fn update_display_super(&self) {
        let advanced = self.header.is_advanced();

        // visibility
        self.reset_to_nonplugin_state();
        let back_projection = self.is_back_projection();
        let multifilt = self.is_multifilt();
        let _filter_trials = self.is_multifilt();
        let ctf3d = self.is_ctf3d();
        self.ltf_log_density_scale_offset.set_visible(advanced);
        self.ltf_log_density_scale_factor.set_visible(advanced);
        self.ltf_linear_density_scale_offset.set_visible(advanced);
        self.ltf_linear_density_scale_factor.set_visible(advanced);
        self.ltf_tomo_width
            .set_visible(advanced && (back_projection || ctf3d));
        self.ltf_tomo_height
            .set_visible(advanced && (back_projection || ctf3d));
        self.ltf_tomo_thickness.set_visible(!multifilt);
        self.ltf_y_shift
            .set_visible(advanced && (back_projection || ctf3d));
        self.ltf_x_shift
            .set_visible(advanced && (back_projection || ctf3d));
        self.ltf_z_shift.set_visible(!multifilt);
        let super_sampling = (advanced && back_projection) || ctf3d;
        self.pnl_recon_with_super_sampling
            .set_visible(super_sampling);
        self.cb_super_sample_factor.set_visible(super_sampling);
        self.sp_super_sample_factor.set_visible(super_sampling);
        self.cb_expand_input_lines.set_visible(super_sampling);
        self.ltf_tilt_angle_offset.set_visible(advanced);
        self.radial_panel
            .set_visible(back_projection || ctf3d || multifilt);
        self.ltf_extra_exclude_list.set_visible(advanced);
        self.trial_panel.set_visible(back_projection);
        self.trial_tilt_panel.set_visible(advanced);
        self.pnl_button.set_visible(back_projection);

        self.radial_panel.update_display();
        self.cpu_gpu_panel.update_display();

        self.btn_3dmod_tomogram.set_enabled(true);
        self.ctf_log.set_enabled(true);
        self.ltf_tomo_width.set_enabled(true);
        self.ltf_tomo_thickness.set_enabled(true);
        self.ltf_x_axis_tilt.set_enabled(true);
        self.ltf_tilt_angle_offset.set_enabled(true);
        self.ltf_extra_exclude_list.set_enabled(true);

        let log_is_selected = self.ctf_log.is_selected();
        self.ltf_log_density_scale_factor
            .set_enabled(log_is_selected);
        self.ltf_log_density_scale_offset
            .set_enabled(log_is_selected);
        self.ltf_linear_density_scale_factor
            .set_enabled(!log_is_selected);
        self.ltf_linear_density_scale_offset
            .set_enabled(!log_is_selected);

        self.ltf_tomo_height.set_enabled(true);
        self.ltf_y_shift.set_enabled(true);
        self.ltf_z_shift.set_enabled(true);
        self.ltf_x_shift.set_enabled(true);

        self.cb_super_sample_factor.set_enabled(true);
        let supersample_is_selected = self.cb_super_sample_factor.is_selected();
        self.sp_super_sample_factor
            .set_enabled(supersample_is_selected);
        self.cb_expand_input_lines
            .set_enabled(supersample_is_selected);

        self.radial_panel.set_editable(true);
        self.cb_use_local_alignment.set_enabled(
            self.used_local_alignments.get() && !self.newst_fiducialess_alignment.get(),
        );
        self.cb_use_z_factors
            .set_enabled(self.made_z_factors.get() && !self.newst_fiducialess_alignment.get());
        self.btn_tilt.set_enabled(true);
        self.btn_delete_stack.set_enabled(true);
        self.header
            .set_text_string(Some(if self.is_back_projection() {
                BACK_PROJECTION_HEADER
            } else {
                PARAMETERS_HEADER
            }));
    }

    /// Java protected final `isParallelProcess()`.
    pub fn is_parallel_process(&self) -> bool {
        self.cpu_gpu_panel.is_parallel_process()
    }

    /// Java `getRunMethodForProcessInterface()`.
    pub fn get_run_method_for_process_interface(&self) -> ProcessingMethod {
        self.cpu_gpu_panel.get_run_method_for_process_interface()
    }

    /// Java final `isZShiftSet()`.
    pub fn is_z_shift_set(&self) -> bool {
        let text = self.ltf_z_shift.get_text_void().unwrap_or_default();
        java_lang_string_matches_non_whitespace(&text)
    }

    /// Java final `isUseLocalAlignment()`.
    pub fn is_use_local_alignment(&self) -> bool {
        self.cb_use_local_alignment.is_selected()
    }

    /// Java `setState(TomogramState, ConstMetaData)`.
    pub fn set_state(&self, state: &TomogramState, meta_data: &dyn ConstMetaData) {
        // madeZFactors
        if !state.get_made_z_factors(self.axis_id).is_null() {
            self.made_z_factors
                .set(state.get_made_z_factors(self.axis_id).is());
        } else {
            self.made_z_factors
                .set(state.get_backward_compatible_made_z_factors(self.axis_id));
        }
        // newstFiducialessAlignment
        if !state
            .get_newst_fiducialess_alignment(self.axis_id)
            .is_null()
        {
            self.newst_fiducialess_alignment
                .set(state.get_newst_fiducialess_alignment(self.axis_id).is());
        } else {
            self.newst_fiducialess_alignment
                .set(meta_data.is_fiducialess_alignment(self.axis_id));
        }
        // usedLocalAlignments
        if !state.get_used_local_alignments(self.axis_id).is_null() {
            self.used_local_alignments
                .set(state.get_used_local_alignments(self.axis_id).is());
        } else {
            self.used_local_alignments
                .set(state.get_backward_compatible_used_local_alignments(self.axis_id));
        }
        self.update_display();
    }

    /// Java `isUseZFactors()`.
    pub fn is_use_z_factors(&self) -> bool {
        self.cb_use_z_factors.is_selected()
    }

    /// Java `isCbSuperSampleFactor()`.
    pub fn is_cb_super_sample_factor(&self) -> bool {
        self.cb_super_sample_factor.is_selected()
    }

    /// Java `isCbExpandInputLines()`.
    pub fn is_cb_expand_input_lines(&self) -> bool {
        self.cb_expand_input_lines.is_selected()
    }

    /// Java `getParameters(MetaData) throws FortranInputSyntaxException`
    /// (`Tilt3dFindPanel` overrides it).
    pub fn get_parameters_meta_data(
        &self,
        meta_data: &MetaData,
    ) -> Result<(), FortranInputSyntaxException> {
        self.cpu_gpu_panel
            .get_parameters_panel_id_meta_data(self.panel_id, meta_data)?;
        self.trial_tilt_panel.get_parameters_meta_data(meta_data)?;
        meta_data.set_gen_log(self.axis_id, self.ctf_log.get_text_void().as_deref());
        meta_data.set_gen_scale_factor_log(
            self.axis_id,
            self.ltf_log_density_scale_factor.get_text_void().as_deref(),
        );
        meta_data.set_gen_scale_offset_log(
            self.axis_id,
            self.ltf_log_density_scale_offset.get_text_void().as_deref(),
        );
        meta_data.set_gen_scale_factor_linear(
            self.axis_id,
            self.ltf_linear_density_scale_factor
                .get_text_void()
                .as_deref(),
        );
        meta_data.set_gen_scale_offset_linear(
            self.axis_id,
            self.ltf_linear_density_scale_offset
                .get_text_void()
                .as_deref(),
        );
        meta_data
            .set_super_sample_factor(self.axis_id, Some(self.sp_super_sample_factor.get_value()));
        meta_data.set_expand_input_lines(self.axis_id, self.cb_expand_input_lines.is_selected());
        self.radial_panel.get_parameters_meta_data(meta_data);
        Ok(())
    }

    /// Java final `getParameters(ReconScreenState)`.
    pub fn get_parameters_recon_screen_state(&self, screen_state: &ReconScreenState) {
        self.header
            .get_state(Some(screen_state.get_tomo_gen_tilt_header_state()));
        self.trial_tilt_panel
            .get_parameters_recon_screen_state(screen_state);
    }

    /// Java final `setParameters(ConstMetaData)`.
    pub fn set_parameters_const_meta_data(&self, meta_data: &dyn ConstMetaData) {
        let tilt_parallel = meta_data.get_tilt_parallel(self.axis_id, self.panel_id);
        let parallel_process: Option<&ConstEtomoNumber> = tilt_parallel.as_ref().map(|parallel| {
            let parallel: &ConstEtomoNumber = parallel;
            parallel
        });
        self.cpu_gpu_panel
            .set_parameters_const_meta_data_const_etomo_number(meta_data, parallel_process);
        self.trial_tilt_panel
            .set_parameters_const_meta_data(meta_data);
        self.ctf_log
            .set_text_string(Some(&meta_data.get_gen_log(self.axis_id)));
        self.ltf_log_density_scale_factor
            .set_text_string(Some(&meta_data.get_gen_scale_factor_log(self.axis_id)));
        self.ltf_log_density_scale_offset
            .set_text_string(Some(&meta_data.get_gen_scale_offset_log(self.axis_id)));
        if meta_data.is_gen_scale_factor_linear_set(self.axis_id) {
            self.ltf_linear_density_scale_factor
                .set_text_string(Some(&meta_data.get_gen_scale_factor_linear(self.axis_id)));
        }
        if meta_data.is_gen_scale_offset_linear_set(self.axis_id) {
            self.ltf_linear_density_scale_offset
                .set_text_string(Some(&meta_data.get_gen_scale_offset_linear(self.axis_id)));
        }
        self.sp_super_sample_factor
            .set_value_const_etomo_number(&meta_data.get_gen_super_sample_factor(self.axis_id));
        self.cb_expand_input_lines
            .set_selected_boolean(meta_data.get_gen_expand_input_lines(self.axis_id).is());
        self.radial_panel.set_parameters_const_meta_data(meta_data);
        self.update_display();
    }

    /// Java `@Deprecated disableGpu(boolean)` (2/4/2019 No longer implements
    /// ProcessInterface; see CpuGpuPanel).
    pub fn disable_gpu(&self, disable: bool) {
        self.cpu_gpu_panel.update_gpu(disable);
        self.update_display();
    }

    /// Java `@Deprecated lockProcessingMethod(boolean)` (2/4/2019 No longer
    /// implements ProcessInterface; see CpuGpuPanel).
    pub fn lock_processing_method(&self, lock: bool) {
        self.cpu_gpu_panel.lock_processing_method(lock);
        self.update_display();
    }

    /// Java `getProcessingMethod()` (implements `TrialTiltParent`).
    pub fn get_processing_method(&self) -> ProcessingMethod {
        self.cpu_gpu_panel.get_processing_method()
    }

    /// Java `@Deprecated getSecondaryProcessingMethod()` (2/4/2019 No longer
    /// implements ProcessInterface; see CpuGpuPanel).
    pub fn get_secondary_processing_method(&self) -> Option<ProcessingMethod> {
        None
    }

    /// Java `reregisterProcessingMethodMediator()`.
    pub fn reregister_processing_method_mediator(&self) {
        self.cpu_gpu_panel.reregister_processing_method_mediator();
    }

    /// Java `setParameters(ConstTiltParam, boolean)` (`Tilt3dFindPanel`
    /// overrides it with a call to this).  Set the UI parameters with the
    /// specified tiltParam values.  WARNING: be sure the setNewstParam is
    /// called first so the binning value for the stack is known.  The
    /// thickness, first and last slice, width and x,y,z offsets are scaled so
    /// that they are represented to the user in unbinned dimensions.
    /// `initialize` - true when the dialog is first created for the dataset.
    pub fn set_parameters_const_tilt_param_boolean(
        &self,
        tilt_param: &dyn ConstTiltParam,
        initialize: bool,
    ) {
        if tilt_param.has_width() {
            self.ltf_tomo_width.set_text_int(tilt_param.get_width());
        }
        if tilt_param.has_thickness() {
            self.ltf_tomo_thickness
                .set_text_int(tilt_param.get_thickness());
        }
        if tilt_param.has_x_shift() {
            self.ltf_x_shift.set_text_double(tilt_param.get_x_shift());
        }
        if tilt_param.has_z_shift() {
            self.ltf_z_shift
                .set_text_const_etomo_number(Some(tilt_param.get_z_shift()));
        }
        if tilt_param.has_slice() {
            let y_height_and_shift = TomogramTool::get_y_height_and_shift(
                self.manager,
                self.axis_id,
                tilt_param.get_idx_slice_start(),
                tilt_param.get_idx_slice_stop(),
            );
            // Java: `yHeightAndShift != null && yHeightAndShift.length == 2`.
            if let Some(y_height_and_shift) = y_height_and_shift {
                self.ltf_tomo_height.set_text_int(y_height_and_shift[0]);
                self.ltf_y_shift.set_text_int(y_height_and_shift[1]);
            }
        }
        if tilt_param.has_x_axis_tilt() {
            self.ltf_x_axis_tilt
                .set_text_double(tilt_param.get_x_axis_tilt());
        }
        if tilt_param.has_tilt_angle_offset() {
            self.ltf_tilt_angle_offset
                .set_text_const_etomo_number(Some(tilt_param.get_tilt_angle_offset()));
        }
        self.radial_panel
            .set_parameters_const_tilt_param(tilt_param);
        self.ctf_log
            .set_selected_boolean(tilt_param.has_log_offset());
        let log = tilt_param.has_log_offset();
        // If initialize is true, get defaults from tilt.com
        if log || initialize {
            let text = tilt_param.get_log_shift();
            if !java_lang_string_matches_whitespace(&text) {
                self.ctf_log.set_text_string(Some(&text));
            }
        }
        if !log && initialize {
            let text = self.ctf_log.get_text_void();
            if text
                .as_deref()
                .is_none_or(java_lang_string_matches_whitespace)
            {
                self.ctf_log.set_text_string(Some("0.0"));
            }
        }
        if (log || initialize) && tilt_param.has_scale() {
            if log {
                self.ltf_log_density_scale_offset
                    .set_text_double(tilt_param.get_scale_f_level());
                self.ltf_log_density_scale_factor
                    .set_text_double(tilt_param.get_scale_coeff());
            } else {
                // New combination of parameters: !log and initialize
                self.ltf_log_density_scale_offset
                    .set_text_double(tilt_param.get_scale_f_level());
                self.ltf_log_density_scale_factor.set_text_double(
                    utilities::java_lang_math_round((tilt_param.get_scale_coeff() * 5000.0) / 10.0)
                        as f64
                        * 10.0,
                );
            }
        }
        if !log && tilt_param.has_scale() {
            self.ltf_linear_density_scale_offset
                .set_text_double(tilt_param.get_scale_f_level());
            self.ltf_linear_density_scale_factor
                .set_text_double(tilt_param.get_scale_coeff());
        }
        if initialize && log {
            let mut log_scale = EtomoNumber::new_with_type(Some(Type::Double));
            let text = self.ltf_log_density_scale_factor.get_text_void();
            log_scale.set_string(text.as_deref());
            if !log_scale.is_null() && log_scale.is_valid() {
                self.ltf_linear_density_scale_factor.set_text_double(
                    utilities::java_lang_math_round((log_scale.get_double() / 5000.0) * 1000.0)
                        as f64
                        / 1000.0,
                );
            }
        }
        self.cb_super_sample_factor
            .set_selected_boolean(tilt_param.is_super_sample_factor_set());
        if self.cb_super_sample_factor.is_selected() {
            self.sp_super_sample_factor
                .set_value_string(Some(&tilt_param.get_super_sample_factor()));
            self.cb_expand_input_lines
                .set_selected_boolean(tilt_param.is_expand_input_lines_set());
        }
        self.cpu_gpu_panel
            .set_parameters_const_tilt_param_boolean(tilt_param, initialize);
        if !initialize {
            // During initialization the value should coming from setup
            // cbUseGpu.setSelected(tiltParam.isUseGpu());
            // updateUseGpu();
        }
        let meta_data = self.manager.get_meta_data();
        self.cb_use_local_alignment
            .set_selected_boolean(meta_data.get_use_local_alignments(self.axis_id));
        self.cb_use_z_factors
            .set_selected_boolean(meta_data.get_use_z_factors(self.axis_id).is());
        self.ltf_extra_exclude_list
            .set_text_string(Some(&tilt_param.get_exclude_list2()));
        self.update_display();
    }

    /// Java final `setParameters(ReconScreenState)`.
    pub fn set_parameters_recon_screen_state(&self, screen_state: &ReconScreenState) {
        self.trial_tilt_panel
            .set_parameters_recon_screen_state(screen_state);
        let header_state: &dyn ConstPanelHeaderState =
            screen_state.get_tomo_gen_tilt_header_state();
        self.header.set_state(Some(header_state));
        self.btn_tilt.set_button_state(
            screen_state.get_button_state(self.btn_tilt.get_button_state_key().as_deref()),
        );
        self.btn_delete_stack.set_button_state(
            screen_state.get_button_state(self.btn_delete_stack.get_button_state_key().as_deref()),
        );
        self.update_display();
    }

    /// Java `getParameters(SplittiltParam, boolean)` (implements `TiltDisplay`
    /// and `TrialTiltParent`; `Tilt3dFindPanel` overrides it and calls this).
    pub fn get_parameters_splittilt_param_boolean(
        &self,
        param: &mut SplittiltParam,
        do_validation: bool,
    ) -> bool {
        // try { ... } catch (FieldValidationFailedException e) { return false; }
        // Java dereferences manager.getMainPanel() without a null check; a
        // missing main panel is treated like a missing parallel panel here.
        let parallel_panel = self
            .manager
            .get_main_panel()
            .and_then(|main_panel| main_panel.main_panel().get_parallel_panel(self.axis_id));
        let Some(parallel_panel) = parallel_panel else {
            return false;
        };
        let Ok(cpus_selected) = parallel_panel.get_cpus_selected(do_validation) else {
            return false;
        };
        let num_machines = param.set_num_machines(cpus_selected.as_deref());
        if !num_machines.is_valid() {
            if num_machines.equals_int(0) {
                ui_harness::with(|harness| {
                    harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self.manager),
                        parallel_panel
                            .get_no_cpus_selected_error_message()
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
                            parallel_panel.get_cpus_selected_label(),
                            num_machines.get_invalid_reason()
                        ),
                        "Unable to run splittilt",
                        Some(self.axis_id),
                    )
                });
                return false;
            }
        }
        true
    }

    /// Java `setDebug(boolean)` (implements `TiltDisplay`).
    pub fn set_debug(&self, debug: bool) {
        self.radial_panel.set_debug(debug);
    }

    /// Java public `getTomoThickness()`.
    pub fn get_tomo_thickness(&self) -> Option<i64> {
        converter::to_long(self.ltf_tomo_thickness.get_text_void().as_deref())
    }

    /// Java `getParameters(TiltParam, boolean) throws NumberFormatException,
    /// InvalidParameterException, IOException` (implements `TiltDisplay` and
    /// `TrialTiltParent`; `Tilt3dFindPanel` overrides it and calls this).  Get
    /// the tilt parameters from the requested axis panel.
    pub fn get_parameters_tilt_param_boolean(
        &self,
        tilt_param: &mut TiltParam,
        do_validation: bool,
    ) -> Result<bool, TiltDisplayException> {
        /// What the Java try blocks catch or let through.
        enum Thrown {
            /// `FieldValidationFailedException` (caught by the outer try).
            FieldValidationFailed,
            /// `NumberFormatException` (caught by the inner try, rethrown with
            /// `badParameter`).
            NumberFormat(String),
            /// `InvalidParameterException` (propagates).
            InvalidParameter(InvalidParameterException),
        }
        // try {
        if !self
            .radial_panel
            .get_parameters_tilt_param_boolean(tilt_param, do_validation)
        {
            return Ok(false);
        }
        let mut bad_parameter = String::new();
        let manager = self.manager;
        let axis_id = self.axis_id;
        let result: Result<bool, Thrown> = (|| {
            // try {
            bad_parameter = "IMAGEBINNED".to_string();
            tilt_param.set_image_binned();
            // Do not manage full image size. It is coming from copytomocoms.
            let text_of = |field: &LabeledTextField| field.get_text_void().unwrap_or_default();
            let validated = |field: &LabeledTextField| {
                field
                    .get_text_boolean(do_validation)
                    .map_err(|_| Thrown::FieldValidationFailed)
            };
            let parse_double = |text: Option<String>| {
                java_lang_double_value_of(&text.unwrap_or_default()).map_err(Thrown::NumberFormat)
            };
            if java_lang_string_matches_non_whitespace(&text_of(&self.ltf_tomo_width)) {
                bad_parameter = self.ltf_tomo_width.get_label();
                tilt_param.set_width(validated(&self.ltf_tomo_width)?.as_deref());
            } else {
                tilt_param.reset_width();
            }

            // set Z Shift
            if self.is_z_shift_set() {
                bad_parameter = self.ltf_z_shift.get_label();
                tilt_param.set_z_shift(validated(&self.ltf_z_shift)?.as_deref());
            } else {
                tilt_param.reset_z_shift();
            }
            // set X Shift
            if java_lang_string_matches_non_whitespace(&text_of(&self.ltf_x_shift)) {
                bad_parameter = self.ltf_x_shift.get_label();
                tilt_param.set_x_shift(parse_double(validated(&self.ltf_x_shift)?)?);
            } else if self.is_z_shift_set() {
                tilt_param.set_x_shift(0.0);
                self.ltf_x_shift.set_text_int(0);
            } else {
                tilt_param.reset_x_shift();
            }

            let tomo_height = validated(&self.ltf_tomo_height)?;
            let y_shift = validated(&self.ltf_y_shift)?;
            let starting_slice = TomogramTool::get_y_starting_slice(
                manager,
                axis_id,
                tomo_height.as_deref(),
                y_shift.as_deref(),
                self.ltf_tomo_height.get_quoted_label().as_deref(),
                self.ltf_y_shift.get_quoted_label().as_deref(),
            );
            let Some(starting_slice) = starting_slice else {
                return Ok(false);
            };
            if starting_slice.is_null() {
                tilt_param.reset_idx_slice();
            } else {
                let tomo_height = validated(&self.ltf_tomo_height)?;
                let ending_slice = TomogramTool::get_y_ending_slice(
                    manager,
                    axis_id,
                    &starting_slice,
                    tomo_height.as_deref(),
                    self.ltf_tomo_height.get_quoted_label().as_deref(),
                );
                let Some(ending_slice) = ending_slice else {
                    return Ok(false);
                };
                if ending_slice.is_null() {
                    tilt_param.reset_idx_slice();
                } else {
                    tilt_param.set_idx_slice_start(starting_slice.get_int());
                    tilt_param.set_idx_slice_stop(ending_slice.get_int());
                }
            }

            if java_lang_string_matches_non_whitespace(&text_of(&self.ltf_tomo_thickness)) {
                bad_parameter = self.ltf_tomo_thickness.get_label();
                tilt_param.set_thickness(validated(&self.ltf_tomo_thickness)?.as_deref());
            } else {
                tilt_param.reset_thickness();
            }

            if java_lang_string_matches_non_whitespace(&text_of(&self.ltf_x_axis_tilt)) {
                bad_parameter = self.ltf_x_axis_tilt.get_label();
                tilt_param.set_x_axis_tilt(validated(&self.ltf_x_axis_tilt)?.as_deref());
            } else {
                tilt_param.reset_x_axis_tilt();
            }

            if java_lang_string_matches_non_whitespace(&text_of(&self.ltf_tilt_angle_offset)) {
                bad_parameter = self.ltf_tilt_angle_offset.get_label();
                tilt_param
                    .set_tilt_angle_offset(validated(&self.ltf_tilt_angle_offset)?.as_deref());
            } else {
                tilt_param.reset_tilt_angle_offset();
            }

            if self.ltf_log_density_scale_offset.is_enabled()
                && (java_lang_string_matches_non_whitespace(&text_of(
                    &self.ltf_log_density_scale_offset,
                )) || java_lang_string_matches_non_whitespace(&text_of(
                    &self.ltf_log_density_scale_factor,
                )))
            {
                bad_parameter = self.ltf_log_density_scale_factor.get_label();
                tilt_param.set_scale_coeff(parse_double(validated(
                    &self.ltf_log_density_scale_factor,
                )?)?);
                bad_parameter = self.ltf_log_density_scale_offset.get_label();
                tilt_param.set_scale_f_level(parse_double(validated(
                    &self.ltf_log_density_scale_offset,
                )?)?);
            } else if self.ltf_linear_density_scale_offset.is_enabled()
                && (java_lang_string_matches_non_whitespace(&text_of(
                    &self.ltf_linear_density_scale_offset,
                )) || java_lang_string_matches_non_whitespace(&text_of(
                    &self.ltf_linear_density_scale_factor,
                )))
            {
                bad_parameter = self.ltf_linear_density_scale_factor.get_label();
                tilt_param.set_scale_coeff(parse_double(validated(
                    &self.ltf_linear_density_scale_factor,
                )?)?);
                bad_parameter = self.ltf_linear_density_scale_offset.get_label();
                tilt_param.set_scale_f_level(parse_double(validated(
                    &self.ltf_linear_density_scale_offset,
                )?)?);
            } else {
                tilt_param.reset_scale();
            }

            if self.ctf_log.is_selected()
                && java_lang_string_matches_non_whitespace(
                    &self.ctf_log.get_text_void().unwrap_or_default(),
                )
            {
                bad_parameter = self.ctf_log.get_label().to_string();
                tilt_param.set_log_shift(parse_double(
                    self.ctf_log
                        .get_text_boolean(do_validation)
                        .map_err(|_| Thrown::FieldValidationFailed)?,
                )?);
            } else {
                tilt_param.set_log_shift(f64::NAN);
            }

            let meta_data = manager.get_meta_data();
            if self.is_use_local_alignment() && self.cb_use_local_alignment.is_enabled() {
                tilt_param.set_local_align_file(Some(&format!(
                    "{}{}local.xf",
                    meta_data.get_dataset_name(),
                    axis_id.get_extension()
                )));
            } else {
                tilt_param.set_local_align_file(Some(""));
            }
            meta_data.set_use_local_alignments(axis_id, self.is_use_local_alignment());
            // TiltParam.fiducialess is based on whether final alignment was run
            // fiducialess.
            // newstFiducialessAlignment
            let newst_fiducialess_alignment;
            let state = manager.get_state();
            if !state.get_newst_fiducialess_alignment(axis_id).is_null() {
                newst_fiducialess_alignment = state.get_newst_fiducialess_alignment(axis_id).is();
            } else {
                newst_fiducialess_alignment = meta_data.is_fiducialess_alignment(axis_id);
            }
            tilt_param.set_fiducialess(newst_fiducialess_alignment);

            tilt_param
                .set_use_z_factors(self.is_use_z_factors() && self.cb_use_z_factors.is_enabled());
            meta_data.set_use_z_factors(axis_id, self.is_use_z_factors());

            if self.is_cb_super_sample_factor() {
                // Java `(int) spSuperSampleFactor.getIntValue()`: the spinner
                // always holds a value.
                tilt_param.set_super_sample_factor(
                    self.sp_super_sample_factor
                        .get_int_value()
                        .unwrap_or_default(),
                );
                tilt_param.set_expand_input_lines(self.is_cb_expand_input_lines());
            } else {
                tilt_param.reset_super_sample_factor();
                tilt_param.set_expand_input_lines(false);
            }

            tilt_param.set_exclude_list2(validated(&self.ltf_extra_exclude_list)?.as_deref());
            bad_parameter = tilt_param::SUBSETSTART_KEY.to_string();
            if meta_data.get_view_type() == ViewType::Montage {
                // `setMontageSubsetStart` lets InvalidParameterException through
                // (its header read failures arrive as the same `Err`).
                tilt_param.set_montage_subset_start().map_err(|message| {
                    Thrown::InvalidParameter(InvalidParameterException::new(&message))
                })?;
            } else if !tilt_param.set_subset_start() {
                return Ok(false);
            }
            Ok(true)
        })();
        match result {
            Ok(false) => return Ok(false),
            Ok(true) => {}
            // catch (final NumberFormatException except)
            Err(Thrown::NumberFormat(message)) => {
                let message = format!("{bad_parameter} {message}");
                return Err(TiltDisplayException::NumberFormat(message));
            }
            // catch (final IOException e): nothing in the translated body throws
            // IOException (TomogramTool and TiltParam report through their
            // return values), so the Java rethrow with `badParameter` has no
            // counterpart.
            Err(Thrown::InvalidParameter(except)) => {
                return Err(TiltDisplayException::InvalidParameter(except));
            }
            // } catch (final FieldValidationFailedException e) { return false; }
            Err(Thrown::FieldValidationFailed) => return Ok(false),
        }
        self.cpu_gpu_panel.get_parameters_tilt_param(tilt_param);
        Ok(true)
    }

    /// Java public final `action(Run3dmodButton, Run3dmodMenuOptions)`.
    pub fn action_run_3dmod_button_run_3dmod_menu_options(
        &self,
        button: &Run3dmodButton,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    ) {
        self.action_string_deferred_3dmod_button_run_3dmod_menu_options(
            button.get_action_command().as_deref().unwrap_or(""),
            button.get_deferred_3dmod_button(),
            run_3dmod_menu_options,
        );
    }

    /// Java public final `action(String, Deferred3dmodButton,
    /// Run3dmodMenuOptions)` (implements `Run3dmodButtonContainer`).  Executes
    /// the action associated with command.  Deferred3dmodButton is null if it
    /// comes from the dialog's ActionListener.  Otherwise is comes from a
    /// Run3dmodButton which called action(Run3dmodButton, Run3dmoMenuOptions).
    /// In that case it will be null unless it was set in the Run3dmodButton.
    pub fn action_string_deferred_3dmod_button_run_3dmod_menu_options(
        &self,
        command: &str,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    ) {
        if Some(command) == self.btn_tilt.get_action_command().as_deref() {
            if let Some(this) = self.this() {
                let display: ProcessResultDisplayHandle = self.btn_tilt.clone();
                this.tilt_action(
                    Some(display),
                    deferred_3dmod_button,
                    run_3dmod_menu_options,
                    self.get_run_method_for_process_interface(),
                );
            }
        } else if Some(command) == self.btn_delete_stack.get_action_command().as_deref() {
            let display: ProcessResultDisplayHandle = self.btn_delete_stack.clone();
            self.manager
                .delete_intermediate_image_stacks(self.axis_id, Some(display));
        } else if Some(command) == self.btn_3dmod_tomogram.get_action_command().as_deref() {
            if let Some(this) = self.this() {
                this.imod_tomogram_action(deferred_3dmod_button, run_3dmod_menu_options);
            }
        } else {
            self.update_display();
        }
    }

    /// Java `setToolTipText()` (overridden by both subclasses, which call
    /// super).  Initialize the tooltip text for the axis panel objects.
    pub fn set_tool_tip_text(&self) {
        // Java `ReadOnlyAutodoc autodoc = null;` then the try/catch.  The
        // autodoc is fetched but not read by this method.
        // SAFETY: the factory returns an autodoc it keeps for the life of the
        // process; the pointer is not dereferenced here.
        match unsafe {
            autodoc_factory::get_instance(
                Some(self.manager),
                Some(autodoc_factory::TILT),
                self.axis_id,
                false,
            )
        } {
            Ok(_autodoc) => {}
            // `catch (final LockException except) {}`.
            Err(LogFileError::Lock(_)) => {}
            // `catch (final LogFileException | IOException except)`:
            // `except.printStackTrace()`.
            Err(except) => eprintln!("{}", except),
        }
        self.ltf_tomo_thickness.set_tool_tip_text(Some(
            "Thickness, in unbinned pixels, along the z-axis of the reconstructed volume.",
        ));
        self.ltf_tomo_height.set_tool_tip_text(Some(
            "This entry specifies the Y extent of the output tomogram, in unbinned pixels; \
             the default is the height of the aligned stack.",
        ));
        self.ltf_y_shift.set_tool_tip_text(Some(
            "Amount to shift the reconstructed region in Y, in unbinned pixels.  A positive \
             value will shift the region upward and reconstruct an area lower in Y.",
        ));
        self.ltf_tomo_width.set_tool_tip_text(Some(
            "This entry specifies the width, in unbinned pixels, of the output image; the \
             default is the width of the input image.",
        ));
        self.ltf_x_shift.set_tool_tip_text(Some(
            "Amount, in unbinned pixels, to shift the reconstructed slices in X before \
             output.  A positive value will shift the slice to the right, and the output \
             will contain the left part of the whole potentially reconstructable area.",
        ));
        self.ltf_z_shift.set_tool_tip_text(Some(
            "Amount, in unbinned pixels, to shift the reconstructed slices in Z before \
             output.  A positive value will shift the slice upward.",
        ));
        self.ltf_x_axis_tilt
            .set_tool_tip_text(Some(tomogram_generation_dialog::X_AXIS_TILT_TOOLTIP));
        self.ltf_tilt_angle_offset.set_tool_tip_text(Some(
            "Offset in degrees to apply to the tilt angles; a positive offset will rotate \
             the reconstructed slices counterclockwise.",
        ));
        self.cb_super_sample_factor.set_tool_tip_text_string(Some(
            "Compute slices with sub-pixel resolution and scale the result down to reduce \
             artifacts from stray lines in Fourier space",
        ));
        self.sp_super_sample_factor.set_tool_tip_text(Some(
            "Factor by which to supersample the reconstructed slice; bigger factors take \
             more memory and time",
        ));
        self.cb_expand_input_lines.set_tool_tip_text_string(Some(
            "Expand the input lines in Fourier space by the supersampling factor to reduce \
             loss of information from interpolation",
        ));
        self.ltf_log_density_scale_offset.set_tool_tip_text(Some(
            "Amount to add to reconstructed density values before multiplying by the scale \
             factor and outputting the values.",
        ));
        self.ltf_log_density_scale_factor.set_tool_tip_text(Some(
            "Amount to multiply reconstructed density values by, after adding the offset \
             value.",
        ));
        self.ltf_linear_density_scale_offset.set_tool_tip_text(Some(
            "Amount to add to reconstructed density values before multiplying by the scale \
             factor and outputting the values.",
        ));
        self.ltf_linear_density_scale_factor.set_tool_tip_text(Some(
            "Amount to multiply reconstructed density values by, after adding the offset \
             value.",
        ));
        self.ctf_log.set_tool_tip_text(Some(
            "This parameter allows one to generate a reconstruction using the logarithm of \
             the densities in the input file, with the value specified added before taking \
             the logarithm.  If no parameter is specified the logarithm of the input data is \
             not taken.",
        ));
        self.cb_use_local_alignment.set_tool_tip_text_string(Some(
            "Select this checkbox to use local alignments.  You must have created the local \
             alignments in the Fine Alignment step",
        ));
        self.btn_tilt.set_tool_tip_text(Some(
            "Compute the tomogram from the full aligned stack.  This runs the tilt.com \
             script.",
        ));
        self.btn_3dmod_tomogram
            .set_tool_tip_text(Some("View the reconstructed volume in 3dmod."));
        self.btn_delete_stack.set_tool_tip_text(Some(
            "Delete the aligned stack for this axis.  Once the tomogram is calculated this \
             intermediate file is not used and can be deleted to free up disk space.",
        ));
        self.cb_use_z_factors.set_tool_tip_text_string(Some(
            "Use the file containing factors for adjusting the backprojection position in \
             each image as a function of Z height in the output slice (.zfac file).  These \
             factors are necessary when input images have been transformed to correct for an \
             apparent specimen stretch.  If this box is not checked, Z factors in a local \
             alignment file will not be applied.",
        ));
        self.ltf_extra_exclude_list.set_tool_tip_text(Some(
            "List of views to exclude from the reconstruction, in addition to the ones \
             excluded from fine alignment.",
        ));
    }
}

/// Java `java.lang.String.matches("\\S+")`: one or more characters, none of
/// them Java whitespace (`[ \t\n\x0B\f\r]`).  (A JDK stand-in, not an eTomo
/// member.)
fn java_lang_string_matches_non_whitespace(value: &str) -> bool {
    !value.is_empty()
        && !value
            .chars()
            .any(|c| matches!(c, ' ' | '\t' | '\n' | '\u{0B}' | '\u{0C}' | '\r'))
}
