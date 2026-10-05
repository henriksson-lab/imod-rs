//! `IMOD/Etomo/src/etomo/ui/swing/TomogramPositioningDialog.java`.
//!
//! Java `final class TomogramPositioningDialog extends ProcessDialog
//! implements ContextMenu, FiducialessParams, Run3dmodButtonContainer`
//! (Tomogram Positioning User Interface).  An EDT object: created as
//! `Rc<Self>` by [`TomogramPositioningDialog::get_instance`]; every method
//! takes `&self`; the `ProcessDialog` superclass is the embedded `base`
//! (reached through `Deref`), and the overridden `done()` is
//! `ProcessDialogVirtual::done`.  The inner listener class
//! `LocalActionListener` is a closure holding a weak reference to the dialog;
//! the public static inner class `CalcPanel` is [`CalcPanel`] below.
//!
//! The expert is held weakly: the expert owns the dialog (Java field
//! `TomogramPositioningExpert.dialog`), and the Java `expert` field is only
//! used to call back into it.

use crate::imod::etomo::ui::field::Field;
use std::cell::{Cell, RefCell};
use std::ops::Deref;
use std::rc::{Rc, Weak};

use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::const_tilt_param::ConstTiltParam;
use crate::imod::etomo::comscript::const_tiltalign_param::ConstTiltalignParam;
use crate::imod::etomo::comscript::const_tomopitch_param::ConstTomopitchParam;
use crate::imod::etomo::comscript::cryo_position_param::CryoPositionParam;
use crate::imod::etomo::comscript::makecomfile_param::MakecomfileParam;
use crate::imod::etomo::comscript::tilt_param::TiltParam;
use crate::imod::etomo::comscript::tiltalign_param::TiltalignParam;
use crate::imod::etomo::comscript::tomopitch_param::TomopitchParam;
use crate::imod::etomo::jdk::{ActionListener, JComponent, MouseEvent, MouseListener};
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::storage::network::Network;
use crate::imod::etomo::storage::tomopitch_log::TomopitchLog;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::const_etomo_number::{
    ConstEtomoNumber, Type, java_lang_double_to_string,
};
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::r#type::meta_data::MetaData;
use crate::imod::etomo::r#type::process_result_display::ProcessResultDisplayHandle;
use crate::imod::etomo::r#type::recon_screen_state::ReconScreenState;
use crate::imod::etomo::r#type::sample_type::SampleType;
use crate::imod::etomo::r#type::tomogram_state::TomogramState;
use crate::imod::etomo::r#type::view_type::ViewType;
use crate::imod::etomo::ui::field_type::FieldType;
use crate::imod::etomo::ui::field_validation_failed_exception::FieldValidationFailedException;

use super::beveled_border::BeveledBorder;
use super::check_box::CheckBox;
use super::context_menu::ContextMenu;
use super::context_popup::{self, ContextPopup};
use super::deferred_3dmod_button::Deferred3dmodButton;
use super::etched_border::EtchedBorder;
use super::etomo_panel::EtomoPanel;
use super::fiducialess_params::FiducialessParams;
use super::fixed_dim;
use super::generic_mouse_adapter::GenericMouseAdapter;
use super::labeled_spinner::LabeledSpinner;
use super::labeled_text_field::LabeledTextField;
use super::multi_line_button::MultiLineButton;
use super::process_dialog::{ProcessDialog, ProcessDialogVirtual};
use super::recon_ui_expert::ReconUIExpertVirtual;
use super::run_3dmod_button::Run3dmodButton;
use super::run_3dmod_button_container::Run3dmodButtonContainer;
use super::spaced_panel::SpacedPanel;
use super::text_field::TextField;
use super::tomogram_generation_dialog;
use super::tomogram_positioning_expert::TomogramPositioningExpert;
use super::ui_harness;
use super::ui_parameters::UIParameters;
use super::ui_utilities;

/// Java package-private static final `SAMPLE_TOMOGRAMS_LABEL`.
pub const SAMPLE_TOMOGRAMS_LABEL: &str = "Create Sample Tomograms";
/// Java package-private static final `SAMPLE_TOMOGRAMS_TOOLTIP`.
pub const SAMPLE_TOMOGRAMS_TOOLTIP: &str =
    "Build 3 sample tomograms for finding location and angles of section.";
/// Java private static final `HAS_GOLD_BEADS_LABEL`.
const HAS_GOLD_BEADS_LABEL: &str = "Sample has gold beads of size:";
/// Java private static final `AUTO_WHOLE_TOMOGRAM_TOOLTIP`.
const AUTO_WHOLE_TOMOGRAM_TOOLTIP: &str = "Builds a sample tomogram and creates a boundary model.";
/// Java private static final `CREATE_BOUNDARY_LABEL`.
const CREATE_BOUNDARY_LABEL: &str = "Create Boundary Model";
/// Java private static final `EXTRA_THICKNESS_LABEL`.
const EXTRA_THICKNESS_LABEL: &str = "Added border thickness (unbinned): ";

/// Java `final class TomogramPositioningDialog extends ProcessDialog implements
/// ContextMenu, FiducialessParams, Run3dmodButtonContainer`.
pub struct TomogramPositioningDialog {
    /// The `ProcessDialog` superclass.
    base: Rc<ProcessDialog>,

    /// Java private final `ltfSampleThickness`.
    ltf_sample_thickness: Rc<LabeledTextField>,
    /// Java private final `ltfExtraThickness`.
    ltf_extra_thickness: Rc<LabeledTextField>,
    /// Java private final `ltfThickness`.
    ltf_thickness: Rc<LabeledTextField>,
    /// Java private final `cbFiducialess`.
    cb_fiducialess: Rc<CheckBox>,
    /// Java private final `ltfRotation`.
    ltf_rotation: Rc<LabeledTextField>,
    /// Java private final `spinBinning`.
    spin_binning: Rc<LabeledSpinner>,
    /// Java private final `cbWholeTomogram`.
    cb_whole_tomogram: Rc<CheckBox>,
    /// Java private final `btnCreateBoundary`.
    btn_create_boundary: Rc<Run3dmodButton>,
    /// Java private final `pnlFinalAlign`.
    pnl_final_align: Rc<EtomoPanel>,
    /// Java private final `cpAngleOffset`.
    cp_angle_offset: CalcPanel,
    /// Java private `cpTiltAxisZShift` (never reassigned).
    cp_tilt_axis_z_shift: CalcPanel,
    /// Java private final `cpXAxisTilt`.
    cp_x_axis_tilt: CalcPanel,
    /// Java private final `localActionListener` (`LocalActionListener`).
    local_action_listener: ActionListener,
    /// Java private final `cpTiltAngleOffset`.
    cp_tilt_angle_offset: CalcPanel,
    /// Java private final `cpZShift`.
    cp_z_shift: CalcPanel,
    /// Java private final `cbUseGpu`.
    cb_use_gpu: Rc<CheckBox>,
    /// Java private final `cbSampleTypeAuto`.
    cb_sample_type_auto: Rc<CheckBox>,
    /// Java private final `cbSampleTypeCryo`.
    cb_sample_type_cryo: Rc<CheckBox>,
    /// Java private final `cbHasGoldBeads`.
    cb_has_gold_beads: Rc<CheckBox>,
    /// Java private final `tfBeadSize`.
    tf_bead_size: Rc<TextField>,
    /// Java private final `lBeadSize` (a `JLabel`).
    l_bead_size: Rc<JComponent>,
    /// Java private final `ltfExtraThicknessCryo`.
    ltf_extra_thickness_cryo: Rc<LabeledTextField>,
    /// Java private final `cbNoXAxisTilt`.
    cb_no_x_axis_tilt: Rc<CheckBox>,

    /// Java private final `btnSample`.
    btn_sample: Rc<Run3dmodButton>,
    /// Java private final `btnTomopitch`.
    btn_tomopitch: Rc<MultiLineButton>,
    /// Java private final `btnAlign`.
    btn_align: Rc<MultiLineButton>,
    /// Java private final `expert` (held weakly; see the module docs).
    expert: Weak<TomogramPositioningExpert>,
    /// Java private final `state`.
    state: &'static TomogramState,
    /// Java private final `axisType`.
    axis_type: AxisType,
    /// Java private final `viewType`.
    view_type: ViewType,
}

impl Deref for TomogramPositioningDialog {
    type Target = ProcessDialog;
    fn deref(&self) -> &ProcessDialog {
        &self.base
    }
}

impl TomogramPositioningDialog {
    /// Java private constructor `TomogramPositioningDialog(ApplicationManager,
    /// TomogramPositioningExpert, AxisID, AxisType, ViewType)`
    /// (TomogramPositioningDialog.java:436).
    fn new(
        app_mgr: &'static ApplicationManager,
        expert: Weak<TomogramPositioningExpert>,
        axis_id: AxisID,
        axis_type: AxisType,
        view_type: ViewType,
    ) -> Rc<TomogramPositioningDialog> {
        let instance = Rc::new_cyclic(|this: &Weak<TomogramPositioningDialog>| {
            // super(appMgr, axisID, DialogType.TOMOGRAM_POSITIONING)
            let base = ProcessDialog::new_application_manager_axis_id_dialog_type(
                app_mgr,
                axis_id,
                DialogType::TomogramPositioning,
            );
            // Field initializers, in declaration order.
            let container: Weak<dyn Run3dmodButtonContainer> = this.clone();
            let ltf_sample_thickness = LabeledTextField::new_field_type_string(
                FieldType::Integer,
                Some("Positioning tomogram thickness: "),
            );
            let ltf_extra_thickness = LabeledTextField::new_field_type_string(
                FieldType::FloatingPoint,
                Some(EXTRA_THICKNESS_LABEL),
            );
            let ltf_thickness = LabeledTextField::new_field_type_string(
                FieldType::Integer,
                Some("Final Tomogram Thickness: "),
            );
            let cb_fiducialess = CheckBox::new_string(Some("Coarse alignment only"));
            let ltf_rotation = LabeledTextField::new_field_type_string(
                FieldType::FloatingPoint,
                Some("Tilt axis rotation:"),
            );
            let spin_binning = LabeledSpinner::get_instance_string_int_int_int_int(
                Some("   Binning "),
                3,
                1,
                8,
                1,
            );
            let cb_whole_tomogram = CheckBox::new_string(Some("Use whole tomogram"));
            let btn_create_boundary =
                Run3dmodButton::get_3dmod_instance_string_run_3dmod_button_container(
                    Some(CREATE_BOUNDARY_LABEL),
                    Some(container.clone()),
                );
            let pnl_final_align = EtomoPanel::new();
            let cp_angle_offset = CalcPanel::new("Angle offset");
            let cp_tilt_axis_z_shift = CalcPanel::new("Z shift");
            let cp_x_axis_tilt = CalcPanel::new("X axis tilt");
            // LocalActionListener
            let local_action_listener: ActionListener = {
                let adaptee = this.clone();
                Rc::new(move |event| {
                    if let Some(adaptee) = adaptee.upgrade() {
                        adaptee.action(event.get_action_command().unwrap_or(""), None, None);
                    }
                })
            };
            let cp_tilt_angle_offset = CalcPanel::new("Tilt angle offset");
            let cp_z_shift = CalcPanel::new("Z shift");
            let cb_use_gpu = CheckBox::new_string(Some("Use the GPU"));
            let cb_sample_type_auto =
                CheckBox::new_string(Some("Find boundary model automatically"));
            let cb_sample_type_cryo = CheckBox::new_string(Some("Do positioning for cryo sample"));
            let cb_has_gold_beads = CheckBox::new_string(Some(HAS_GOLD_BEADS_LABEL));
            let tf_bead_size =
                TextField::new(FieldType::FloatingPoint, Some(HAS_GOLD_BEADS_LABEL), None);
            let l_bead_size = JComponent::new_label(" pixels");
            let ltf_extra_thickness_cryo = LabeledTextField::new_field_type_string(
                FieldType::FloatingPoint,
                Some(EXTRA_THICKNESS_LABEL),
            );
            let cb_no_x_axis_tilt = CheckBox::new_string(Some("Keep X-axis tilt at zero"));

            // Constructor body.
            let state = app_mgr.get_state();
            let display_factory = app_mgr.get_process_result_display_factory(axis_id);
            // Java casts `(Run3dmodButton) displayFactory.getSampleTomogram()`;
            // the factory returns the concrete button.
            let btn_sample = display_factory.get_sample_tomogram();
            btn_sample.set_container(Some(container.clone()));
            let deferred: Rc<dyn Deferred3dmodButton> = btn_create_boundary.clone();
            btn_sample.set_deferred_3dmod_button_deferred_3dmod_button(Some(deferred));
            let btn_tomopitch = display_factory.get_compute_pitch();
            let btn_align = display_factory.get_final_alignment();
            // Swing layout: rootPanel BoxLayout Y_AXIS.
            base.btn_execute.set_text(Some("Done"));
            // Construct the binning spinner
            // Swing layout: spinBinning.setTextMaxmimumSize(UIParameters.getInstance()
            // .getSpinnerDimension()) - the size argument is layout only.
            spin_binning.set_text_maxmimum_size();

            // Create the primary panels
            let pnl_whole_tomogram = JComponent::new_panel();
            // Swing layout: pnlWholeTomogram BoxLayout X_AXIS.
            // if (appMgr.getMetaData().getViewType() == ViewType.MONTAGE) {
            // cbWholeTomogram.setEnabled(false);
            // }
            pnl_whole_tomogram.add(&cb_whole_tomogram.get_component());
            pnl_whole_tomogram.add(&spin_binning.get_container());

            let pnl_tomo_params = JComponent::new_panel();
            let pnl_gold_beads = JComponent::new_panel();
            // Swing layout: pnlTomoParams BoxLayout Y_AXIS.
            ui_utilities::add_with_y_space(&pnl_tomo_params, &cb_use_gpu.get_component());
            ui_utilities::add_with_y_space(&pnl_tomo_params, &cb_sample_type_auto.get_component());
            ui_utilities::add_with_y_space(&pnl_tomo_params, &cb_sample_type_cryo.get_component());
            ui_utilities::add_with_y_space(&pnl_tomo_params, &pnl_gold_beads);
            ui_utilities::add_with_y_space(&pnl_tomo_params, &ltf_sample_thickness.get_container());
            ui_utilities::add_with_y_space(&pnl_tomo_params, &cb_fiducialess.get_component());
            ui_utilities::add_with_y_space(&pnl_tomo_params, &ltf_rotation.get_container());
            ui_utilities::add_with_y_space(&pnl_tomo_params, &pnl_whole_tomogram);
            // Component.LEFT_ALIGNMENT
            ui_utilities::align_components_x(&pnl_tomo_params, 0.0);
            // GoldBeads
            // Swing layout: pnlGoldBeads BoxLayout X_AXIS.
            pnl_gold_beads.add(&cb_has_gold_beads.get_component());
            pnl_gold_beads.add(&tf_bead_size.get_component());
            pnl_gold_beads.add(&l_bead_size);
            //
            // Swing layout: pnlFinalAlign BoxLayout Y_AXIS.
            pnl_final_align.set_border(&EtchedBorder::new(Some("Final Alignment")).get_border());
            let pnl_final_align_component = pnl_final_align.get_component();
            ui_utilities::add_with_y_space(
                &pnl_final_align_component,
                &cp_angle_offset.get_container(),
            );
            ui_utilities::add_with_y_space(
                &pnl_final_align_component,
                &cp_tilt_axis_z_shift.get_container(),
            );
            ui_utilities::add_with_space(
                &pnl_final_align_component,
                &btn_align.get_component(),
                fixed_dim::x0_y10,
            );
            // Component.CENTER_ALIGNMENT
            ui_utilities::align_components_x(&pnl_final_align_component, 0.5);

            let pnl_position = JComponent::new_panel();
            pnl_position.set_border_title(
                BeveledBorder::new(Some("Tomogram Positioning"))
                    .get_border()
                    .get_title()
                    .as_deref(),
            );
            // Swing layout: pnlPosition BoxLayout Y_AXIS.

            ui_utilities::add_with_y_space(&pnl_position, &pnl_tomo_params);
            ui_utilities::add_with_space(
                &pnl_position,
                &btn_sample.get_component(),
                fixed_dim::x0_y10,
            );
            ui_utilities::add_with_space(
                &pnl_position,
                &btn_create_boundary.get_component(),
                fixed_dim::x0_y10,
            );
            ui_utilities::add_with_space(
                &pnl_position,
                &ltf_extra_thickness.get_container(),
                fixed_dim::x0_y10,
            );
            ui_utilities::add_with_space(
                &pnl_position,
                &ltf_extra_thickness_cryo.get_container(),
                fixed_dim::x0_y1,
            );
            ui_utilities::add_with_space(
                &pnl_position,
                &cb_no_x_axis_tilt.get_component(),
                fixed_dim::x0_y10,
            );
            ui_utilities::add_with_space(
                &pnl_position,
                &btn_tomopitch.get_component(),
                fixed_dim::x0_y10,
            );
            ui_utilities::add_with_y_space(&pnl_position, &pnl_final_align_component);

            let pnl_tilt_parameters = SpacedPanel::get_instance_void();
            // Swing layout: pnlTiltParameters BoxLayout Y_AXIS.
            pnl_tilt_parameters
                .set_border(&EtchedBorder::new(Some("Tilt Parameters")).get_border());
            pnl_tilt_parameters.add_container(&cp_tilt_angle_offset.get_container());
            pnl_tilt_parameters.add_container(&cp_z_shift.get_container());
            pnl_tilt_parameters.add_container(&cp_x_axis_tilt.get_container());
            pnl_tilt_parameters.add_labeled_text_field(&ltf_thickness);
            pnl_position.add(&pnl_tilt_parameters.get_container());

            // Component.CENTER_ALIGNMENT
            ui_utilities::align_components_x(&pnl_position, 0.5);
            ui_utilities::set_button_size_all(
                &pnl_position,
                UIParameters::get_instance_void().get_button_dimension(),
            );

            // Create dialog content pane
            base.root_panel.get_component().add(&pnl_position);
            base.add_exit_buttons();

            TomogramPositioningDialog {
                base,
                ltf_sample_thickness,
                ltf_extra_thickness,
                ltf_thickness,
                cb_fiducialess,
                ltf_rotation,
                spin_binning,
                cb_whole_tomogram,
                btn_create_boundary,
                pnl_final_align,
                cp_angle_offset,
                cp_tilt_axis_z_shift,
                cp_x_axis_tilt,
                local_action_listener,
                cp_tilt_angle_offset,
                cp_z_shift,
                cb_use_gpu,
                cb_sample_type_auto,
                cb_sample_type_cryo,
                cb_has_gold_beads,
                tf_bead_size,
                l_bead_size,
                ltf_extra_thickness_cryo,
                cb_no_x_axis_tilt,
                btn_sample,
                btn_tomopitch,
                btn_align,
                expert,
                state,
                axis_type,
                view_type,
            }
        });
        // Java `this` as the ProcessDialog subclass (for the virtual `done()`).
        let this: Weak<dyn ProcessDialogVirtual> =
            Rc::downgrade(&instance) as Weak<dyn ProcessDialogVirtual>;
        instance.base.set_this(this);
        // The rest of the Java constructor needs the constructed fields.
        instance.set_tool_tip_text();
        ui_harness::INSTANCE.with(|harness| {
            harness
                .pack_axis_id_base_manager(Some(axis_id), Some(app_mgr as &'static dyn BaseManager))
        });
        instance
    }

    /// Java static `getInstance(ApplicationManager, TomogramPositioningExpert,
    /// AxisID, AxisType, ViewType)` (TomogramPositioningDialog.java:526).
    pub fn get_instance(
        manager: &'static ApplicationManager,
        expert: Weak<TomogramPositioningExpert>,
        axis_id: AxisID,
        axis_type: AxisType,
        view_type: ViewType,
    ) -> Rc<TomogramPositioningDialog> {
        let instance =
            TomogramPositioningDialog::new(manager, expert, axis_id, axis_type, view_type);
        instance.add_listeners();
        instance
    }

    /// Java private `addListeners()` (TomogramPositioningDialog.java:535).
    fn add_listeners(self: &Rc<Self>) {
        self.cb_fiducialess
            .add_action_listener(Some(self.local_action_listener.clone()));
        self.cb_whole_tomogram
            .add_action_listener(Some(self.local_action_listener.clone()));
        self.btn_sample
            .add_action_listener(self.local_action_listener.clone());
        self.btn_create_boundary
            .add_action_listener(self.local_action_listener.clone());
        self.btn_tomopitch
            .add_action_listener(self.local_action_listener.clone());
        self.btn_align
            .add_action_listener(self.local_action_listener.clone());
        self.cb_sample_type_auto
            .add_action_listener(Some(self.local_action_listener.clone()));
        self.cb_sample_type_cryo
            .add_action_listener(Some(self.local_action_listener.clone()));
        self.cb_has_gold_beads
            .add_action_listener(Some(self.local_action_listener.clone()));
        // Mouse adapter for context menu
        let context_menu: Weak<dyn ContextMenu> = Rc::downgrade(self) as Weak<dyn ContextMenu>;
        let mouse_adapter: Rc<dyn MouseListener> = GenericMouseAdapter::new(context_menu);
        self.base
            .root_panel
            .get_component()
            .add_mouse_listener(mouse_adapter);
    }

    /// Java `setImageRotation(String)` (TomogramPositioningDialog.java:554).
    pub fn set_image_rotation(&self, tilt_axis_angle: Option<&str>) {
        self.ltf_rotation.set_text_string(tilt_axis_angle);
    }

    /// Java `getAlignParams(TiltalignParam, MetaData, boolean)`
    /// (TomogramPositioningDialog.java:564).
    pub fn get_align_params(
        &self,
        tiltalign_param: &mut TiltalignParam,
        meta_data: &MetaData,
        do_validation: bool,
    ) -> bool {
        // try { ... } catch (FieldValidationFailedException e) { return false; }
        let Ok(angle_offset) = self.cp_angle_offset.get_total_boolean(do_validation) else {
            return false;
        };
        tiltalign_param.set_angle_offset(Some(&angle_offset));
        let Ok(axis_z_shift) = self.cp_tilt_axis_z_shift.get_total_boolean(do_validation) else {
            return false;
        };
        tiltalign_param.set_axis_z_shift(Some(&axis_z_shift));
        self.update_meta_data(meta_data);
        true
    }

    /// Java `getTomopitchParam(TomopitchParam, MetaData, boolean)`
    /// (TomogramPositioningDialog.java:581): get the tomopitch.com parameters
    /// from the dialog.
    pub fn get_tomopitch_param(
        &self,
        tomopitch_param: &mut TomopitchParam,
        meta_data: &MetaData,
        do_validation: bool,
    ) -> bool {
        // try { ... } catch (FieldValidationFailedException e) { return false; }
        tomopitch_param.set_scale_factor(self.cb_whole_tomogram.is_selected());
        if !self.is_sample_type_cryo() {
            let Ok(extra_thickness) = self.ltf_extra_thickness.get_text_boolean(do_validation)
            else {
                return false;
            };
            tomopitch_param.set_extra_thickness(extra_thickness.as_deref());
        } else {
            let Ok(extra_thickness) = self
                .ltf_extra_thickness_cryo
                .get_text_boolean(do_validation)
            else {
                return false;
            };
            tomopitch_param.set_extra_thickness(extra_thickness.as_deref());
        }
        tomopitch_param.set_no_x_axis_tilt(self.cb_no_x_axis_tilt.is_selected());
        tomopitch_param.set_angle_offset_old_const_etomo_number(Some(
            &*self.state.get_sample_angle_offset(self.axis_id),
        ));
        tomopitch_param.set_z_shift_old_const_etomo_number(Some(
            &*self.state.get_sample_axis_z_shift(self.axis_id),
        ));
        tomopitch_param
            .set_x_axis_tilt_old(Some(&*self.state.get_sample_x_axis_tilt(self.axis_id)));
        self.update_meta_data(meta_data);
        true
    }

    /// Java `getParameters(MakecomfileParam, boolean)`
    /// (TomogramPositioningDialog.java:603).
    pub fn get_parameters_makecomfile_param_boolean(
        &self,
        param: &mut MakecomfileParam,
        do_validation: bool,
    ) -> bool {
        match self.ltf_sample_thickness.get_text_boolean(do_validation) {
            Ok(thickness) => {
                param.set_thickness_to_make(thickness.as_deref());
                true
            }
            Err(_) => false,
        }
    }

    /// Java `getTiltParamsForSample(TiltParam, boolean)`
    /// (TomogramPositioningDialog.java:613).
    pub fn get_tilt_params_for_sample(
        &self,
        tilt_param: &mut TiltParam,
        do_validation: bool,
    ) -> bool {
        match self.ltf_sample_thickness.get_text_boolean(do_validation) {
            Ok(thickness) => {
                tilt_param.set_thickness(thickness.as_deref());
                true
            }
            Err(_) => false,
        }
    }

    /// Java `isWholeTomogram()` (TomogramPositioningDialog.java:623).
    pub fn is_whole_tomogram(&self) -> bool {
        self.cb_whole_tomogram.is_selected()
    }

    /// Java `updateMetaData(MetaData)` (TomogramPositioningDialog.java:627).
    pub fn update_meta_data(&self, meta_data: &MetaData) {
        let whole_tomogram = self.is_whole_tomogram();
        if whole_tomogram != meta_data.is_whole_tomogram_sample(self.axis_id) {
            meta_data.set_whole_tomogram_sample(self.axis_id, whole_tomogram);
        }
        meta_data.set_pos_binning_int(self.axis_id, self.spin_binning.get_value().int_value());
    }

    /// Java `getBinning()` (TomogramPositioningDialog.java:635).
    pub fn get_binning(&self) -> i32 {
        if !self.is_whole_tomogram() {
            return 1;
        }
        self.spin_binning.get_value().int_value()
    }

    /// Java `getTiltParams(TiltParam, MetaData, boolean)`
    /// (TomogramPositioningDialog.java:647): get the tilt.com parameters from
    /// the dialog.
    pub fn get_tilt_params(
        &self,
        tilt_param: &mut TiltParam,
        meta_data: &MetaData,
        do_validation: bool,
    ) -> bool {
        tilt_param.set_use_gpu(self.cb_use_gpu.is_enabled() && self.cb_use_gpu.is_selected());
        let fiducialess = FiducialessParams::is_fiducialess(self);
        tilt_param.set_fiducialess(fiducialess);
        // try { ... } catch (FieldValidationFailedException e) { return false; }
        if fiducialess {
            let Ok(z_shift) = self.cp_z_shift.get_total_boolean(do_validation) else {
                return false;
            };
            tilt_param.set_z_shift(Some(&z_shift));
            let Ok(tilt_angle_offset) = self.cp_tilt_angle_offset.get_total_boolean(do_validation)
            else {
                return false;
            };
            tilt_param.set_tilt_angle_offset(Some(&tilt_angle_offset));
        }
        if self.is_sample_type_cryo() {
            tilt_param.set_x_axis_tilt_double(0.0);
        } else {
            let Ok(x_axis_tilt) = self.cp_x_axis_tilt.get_total_boolean(do_validation) else {
                return false;
            };
            tilt_param.set_x_axis_tilt(Some(&x_axis_tilt));
        }
        tilt_param.set_image_binned_int(self.get_binning());
        let Ok(thickness) = self.ltf_thickness.get_text_boolean(do_validation) else {
            return false;
        };
        tilt_param.set_thickness(thickness.as_deref());
        self.update_meta_data(meta_data);
        true
    }

    /// Java private `getSampleType()` (TomogramPositioningDialog.java:673).
    fn get_sample_type(&self) -> SampleType {
        if !self.cb_sample_type_auto.is_selected() {
            return SampleType::None;
        }
        if self.cb_sample_type_cryo.is_selected() {
            return SampleType::Cryo;
        }
        SampleType::PlasticSection
    }

    /// Java private `setSampleType(SampleType)` (TomogramPositioningDialog.java:683).
    fn set_sample_type(&self, input: Option<SampleType>) {
        self.cb_sample_type_auto.set_selected_boolean(
            input == Some(SampleType::PlasticSection) || input == Some(SampleType::Cryo),
        );
        self.cb_sample_type_cryo
            .set_selected_boolean(input == Some(SampleType::Cryo));
        self.update_display();
    }

    /// Java `updateDisplay()` (TomogramPositioningDialog.java:695).  If
    /// state.sampleFiducialess doesn't match the fiducialess checkbox, don't
    /// allow Tomopitch to be run.  If state.sampleFiducialess is null (sample
    /// was never run) enable Tomopitch.
    pub fn update_display(&self) {
        // If state.sampleFiducialess doesn't match the fiducialess checkbox, don't
        // allow Tomopitch to be run. If state.sampleFiducialess is null (sample was
        // never run) enable Tomopitch.
        let sample_fiducialess = self.state.get_sample_fiducialess(self.axis_id);
        let mut enable = false;
        let fiducialess = FiducialessParams::is_fiducialess(self);
        if sample_fiducialess.is_none()
            || sample_fiducialess
                .as_ref()
                .is_some_and(|sample_fiducialess| sample_fiducialess.is() == fiducialess)
        {
            enable = true;
        }
        self.btn_tomopitch.set_enabled(enable);
        self.cp_angle_offset.set_enabled(enable && !fiducialess);
        self.cp_tilt_axis_z_shift
            .set_enabled(enable && !fiducialess);
        self.btn_align.set_enabled(enable && !fiducialess);
        self.cp_tilt_angle_offset.set_enabled(enable);
        self.cp_z_shift.set_enabled(enable);
        self.cp_x_axis_tilt.set_enabled(enable);
        self.ltf_thickness.set_enabled(enable);
        // Fiducialless
        self.ltf_rotation.set_enabled(fiducialess);
        self.pnl_final_align
            .get_component()
            .set_visible(!fiducialess);
        self.cp_tilt_angle_offset.set_visible(fiducialess);
        self.cp_z_shift.set_visible(fiducialess);
        let sample_type_auto = self.cb_sample_type_auto.is_selected();
        self.cb_sample_type_cryo
            .set_enabled(sample_type_auto && self.view_type != ViewType::Montage);
        let cryo = self.is_sample_type_cryo();
        self.cb_has_gold_beads.set_enabled(cryo);
        self.tf_bead_size.set_enabled(cryo);
        self.l_bead_size.set_enabled(cryo);
        self.ltf_extra_thickness.set_visible(!cryo);
        self.ltf_extra_thickness_cryo.set_visible(cryo);
        // WholeTomogram
        // Cryo is always whole tomogram
        self.cb_whole_tomogram.set_editable(!cryo);
        if cryo {
            self.cb_whole_tomogram.set_selected_boolean(true);
        }
        let whole_tomogram = self.cb_whole_tomogram.is_selected();
        self.spin_binning.set_enabled(!cryo && whole_tomogram);
        // Sample
        if !sample_type_auto {
            self.btn_create_boundary
                .set_text(Some(CREATE_BOUNDARY_LABEL));
            if whole_tomogram {
                self.btn_sample.set_text(Some("Create Whole Tomogram"));
                self.btn_sample.set_tool_tip_text(Some(
                    "Create whole tomogram for drawing positioning model.",
                ));
            } else {
                self.btn_sample.set_text(Some(SAMPLE_TOMOGRAMS_LABEL));
                self.btn_sample
                    .set_tool_tip_text(Some(SAMPLE_TOMOGRAMS_TOOLTIP));
            }
        } else {
            self.btn_create_boundary
                .set_text(Some("View Boundary Model"));
            if !cryo {
                if !whole_tomogram {
                    self.btn_sample
                        .set_text(Some("Create Samples & Boundary Model"));
                    self.btn_sample.set_tool_tip_text(Some(
                        "Builds 3 sample tomograms and creates a boundary model.",
                    ));
                } else {
                    self.btn_sample
                        .set_text(Some("Create Tomogram & Boundary Model"));
                    self.btn_sample
                        .set_tool_tip_text(Some(AUTO_WHOLE_TOMOGRAM_TOOLTIP));
                }
            } else {
                self.btn_sample
                    .set_text(Some("Find Boundary Model for Cryo"));
                self.btn_sample
                    .set_tool_tip_text(Some(AUTO_WHOLE_TOMOGRAM_TOOLTIP));
            }
        }
        let has_gold_beads =
            self.cb_has_gold_beads.is_enabled() && self.cb_has_gold_beads.is_selected();
        self.tf_bead_size.set_enabled(has_gold_beads);
        self.l_bead_size.set_enabled(has_gold_beads);
    }

    /// Java `isSampleTypeAuto()` (TomogramPositioningDialog.java:769).
    pub fn is_sample_type_auto(&self) -> bool {
        self.cb_sample_type_auto.is_selected()
    }

    /// Java `isSampleTypeCryo()` (TomogramPositioningDialog.java:773).
    pub fn is_sample_type_cryo(&self) -> bool {
        self.cb_sample_type_cryo.is_enabled() && self.cb_sample_type_cryo.is_selected()
    }

    /// Java `setParameters(ConstMetaData)` (TomogramPositioningDialog.java:781):
    /// set the metadata parameters in the dialog.
    pub fn set_parameters_const_meta_data(&self, meta_data: &MetaData) {
        let manager: &'static dyn BaseManager = self.application_manager;
        // Use GPU
        self.cb_use_gpu
            .set_enabled(Network::is_local_host_gpu_processing_enabled(
                manager,
                self.axis_id,
                manager.get_property_user_dir().as_deref(),
            ));
        self.cb_use_gpu
            .set_selected_boolean(meta_data.is_default_gpu_processing());
        self.spin_binning
            .set_value_int(meta_data.get_pos_binning(self.axis_id));
        self.ltf_sample_thickness
            .set_text_const_etomo_number(Some(&meta_data.get_sample_thickness(self.axis_id)));
        self.set_sample_type(meta_data.get_sample_type(self.axis_id));
        if meta_data.is_positioning_new_dialog(self.axis_id) {
            // Default bead size
            self.cb_has_gold_beads
                .set_selected_boolean(meta_data.is_fiducial_diameter_available());
            self.tf_bead_size
                .set_text_string(Some(&java_lang_double_to_string(
                    self.application_manager
                        .calc_unbinned_bead_diameter_pixels(),
                )));
        } else {
            // Saved bead size
            self.cb_has_gold_beads
                .set_selected_boolean(meta_data.is_has_gold_beads(self.axis_id));
            if !meta_data.is_positioning_bead_size_null(self.axis_id) {
                self.tf_bead_size
                    .set_text_string(Some(&meta_data.get_positioning_bead_size(self.axis_id)));
            }
        }
        self.ltf_extra_thickness
            .set_text_string(Some(&meta_data.get_extra_thickness(self.axis_id)));
        self.ltf_extra_thickness_cryo
            .set_text_string(Some(&meta_data.get_extra_thickness_cryo(self.axis_id)));
    }

    /// Java `isTomopitchButton()` (TomogramPositioningDialog.java:806).
    pub fn is_tomopitch_button(&self) -> bool {
        self.btn_tomopitch.is_selected()
    }

    /// Java `getParameters(MetaData)` (TomogramPositioningDialog.java:810).
    pub fn get_parameters_meta_data(&self, meta_data: &MetaData) {
        meta_data.set_positioning_new_dialog(self.axis_id, false);
        meta_data.set_sample_type_sample_type(self.axis_id, Some(self.get_sample_type()));
        meta_data.set_sample_thickness(
            self.axis_id,
            self.ltf_sample_thickness.get_text_void().as_deref(),
        );
        meta_data.set_has_gold_beads(self.axis_id, self.cb_has_gold_beads.is_selected());
        meta_data
            .set_positioning_bead_size(self.axis_id, self.tf_bead_size.get_text_void().as_deref());
        meta_data.set_extra_thickness(
            self.axis_id,
            self.ltf_extra_thickness.get_text_void().as_deref(),
        );
        meta_data.set_extra_thickness_cryo(
            self.axis_id,
            self.ltf_extra_thickness_cryo.get_text_void().as_deref(),
        );
    }

    /// Java `setParameters(CryoPositionParam)` (TomogramPositioningDialog.java:820).
    pub fn set_parameters_cryo_position_param(&self, param: &CryoPositionParam) {
        self.cb_has_gold_beads
            .set_selected_boolean(param.is_bead_size_set());
        self.tf_bead_size
            .set_text_string(Some(&param.get_bead_size()));
    }

    /// Java `getParameters(CryoPositionParam, boolean)`
    /// (TomogramPositioningDialog.java:825).
    pub fn get_parameters_cryo_position_param_boolean(
        &self,
        param: &mut CryoPositionParam,
        do_validation: bool,
    ) -> bool {
        // try { ... } catch (FieldValidationFailedException e) { return false; }
        let has_gold_beads = self.cb_has_gold_beads.is_selected();
        param.set_find_beads_in_volume(has_gold_beads);
        if has_gold_beads {
            let Ok(bead_size) = self.tf_bead_size.get_text_boolean(do_validation) else {
                return false;
            };
            param.set_bead_size(bead_size.as_deref());
        } else {
            param.reset_bead_size();
        }
        let Ok(thickness) = self.ltf_sample_thickness.get_text_boolean(do_validation) else {
            return false;
        };
        param.set_thickness_of_tomograms(thickness.as_deref());
        true
    }

    /// Java `isAlignButton()` (TomogramPositioningDialog.java:843).
    pub fn is_align_button(&self) -> bool {
        self.btn_align.is_selected()
    }

    /// Java `isAlignButtonEnabled()` (TomogramPositioningDialog.java:847).
    pub fn is_align_button_enabled(&self) -> bool {
        self.btn_align.is_enabled()
    }

    /// Java `setButtonState(ReconScreenState)` (TomogramPositioningDialog.java:851).
    pub fn set_button_state(&self, screen_state: &ReconScreenState) {
        self.btn_sample.set_button_state(
            screen_state.get_button_state(self.btn_sample.get_button_state_key().as_deref()),
        );
        self.btn_tomopitch.set_button_state(
            screen_state.get_button_state(self.btn_tomopitch.get_button_state_key().as_deref()),
        );
        self.btn_align.set_button_state(
            screen_state.get_button_state(self.btn_align.get_button_state_key().as_deref()),
        );
    }

    /// Java `setTiltParam(ConstTiltParam, boolean)` (TomogramPositioningDialog.java:858).
    pub fn set_tilt_param(&self, tilt_param: &dyn ConstTiltParam, initialize: bool) {
        if !initialize {
            // During initialization the value should coming from setup
            self.cb_use_gpu
                .set_selected_boolean(tilt_param.is_use_gpu());
        }
        self.cp_x_axis_tilt.set_double(tilt_param.get_x_axis_tilt());
        self.ltf_thickness.set_text_int(tilt_param.get_thickness());
        self.cp_tilt_angle_offset
            .set_const_etomo_number(Some(tilt_param.get_tilt_angle_offset()));
        self.cp_z_shift
            .set_const_etomo_number(Some(tilt_param.get_z_shift()));
    }

    /// Java `rollAlignComAngles()` (TomogramPositioningDialog.java:869).
    pub fn roll_align_com_angles(&self) {
        self.cp_angle_offset.update_display(false);
        self.cp_tilt_axis_z_shift.update_display(false);
    }

    /// Java `rollTiltComAngles()` (TomogramPositioningDialog.java:874).
    pub fn roll_tilt_com_angles(&self) {
        self.cp_x_axis_tilt.update_display(false);
    }

    /// Java `setAlignParam(ConstTiltalignParam)` (TomogramPositioningDialog.java:878).
    pub fn set_align_param(&self, tiltalign_param: &ConstTiltalignParam) {
        self.cp_angle_offset
            .set_const_etomo_number(Some(tiltalign_param.get_angle_offset()));
        self.cp_tilt_axis_z_shift
            .set_const_etomo_number(Some(tiltalign_param.get_axis_z_shift()));
    }

    /// Java `setParametersFiducialess(TiltParam, MetaData)`
    /// (TomogramPositioningDialog.java:883).
    pub fn set_parameters_fiducialess(&self, param: &mut TiltParam, meta_data: &MetaData) {
        param.set_fiducialess(meta_data.is_fiducialess(self.axis_id));
        if FiducialessParams::is_fiducialess(self) {
            self.cp_tilt_angle_offset
                .set_const_etomo_number(Some(param.get_tilt_angle_offset()));
            self.cp_z_shift
                .set_const_etomo_number(Some(param.get_z_shift()));
        } else {
            param.set_tilt_angle_offset(Some(&self.cp_tilt_angle_offset.get_total_void()));
            param.set_z_shift(Some(&self.cp_z_shift.get_total_void()));
        }
        param.reset_subset_start();
    }

    /// Java `setFiducialess(ConstMetaData)` (TomogramPositioningDialog.java:896).
    pub fn set_fiducialess(&self, meta_data: &MetaData) {
        self.cb_fiducialess
            .set_selected_boolean(meta_data.is_fiducialess_alignment(self.axis_id));
        self.update_display();
    }

    /// Java `setParameters(TomopitchLog)` (TomogramPositioningDialog.java:901).
    ///
    /// Upstream bug fixed in translation: TomogramPositioningDialog.java:908-913
    /// assigns `missingData = !cp...set(...)` three times, so each result
    /// overwrites the one before and a missing angle offset or Z shift total
    /// is forgotten whenever the X axis tilt total is present (the log is
    /// then not opened for the user).  Here the three results accumulate
    /// (`missing_data |= ...`), as the variable's name and the thickness
    /// check below (`missingData = true` only, never reset) show was meant.
    /// All three panels are still set, in the same order.
    pub fn set_parameters_tomopitch_log(&self, log: &TomopitchLog) -> bool {
        let mut missing_data;
        let angle_offset_original = log.get_angle_offset_original();
        let angle_offset_added = log.get_angle_offset_added();
        let angle_offset_total = log.get_angle_offset_total();
        let axis_z_shift_original = log.get_axis_z_shift_original();
        let axis_z_shift_added = log.get_axis_z_shift_added();
        let axis_z_shift_total = log.get_axis_z_shift_total();
        missing_data = !self
            .cp_angle_offset
            .set_const_etomo_number_const_etomo_number_const_etomo_number(
                Some(&angle_offset_original),
                Some(&angle_offset_added),
                &angle_offset_total,
            );
        missing_data |= !self
            .cp_tilt_axis_z_shift
            .set_const_etomo_number_const_etomo_number_const_etomo_number(
                Some(&axis_z_shift_original),
                Some(&axis_z_shift_added),
                &axis_z_shift_total,
            );
        missing_data |= !self
            .cp_x_axis_tilt
            .set_const_etomo_number_const_etomo_number_const_etomo_number(
                Some(&log.get_x_axis_tilt_original()),
                Some(&log.get_x_axis_tilt_added()),
                &log.get_x_axis_tilt_total(),
            );
        self.cp_tilt_angle_offset
            .set_const_etomo_number_const_etomo_number_const_etomo_number(
                Some(&angle_offset_original),
                Some(&angle_offset_added),
                &angle_offset_total,
            );
        self.cp_z_shift
            .set_const_etomo_number_const_etomo_number_const_etomo_number(
                Some(&axis_z_shift_original),
                Some(&axis_z_shift_added),
                &axis_z_shift_total,
            );
        let thickness = log.get_thickness();
        if thickness.is_null() {
            missing_data = true;
        } else {
            self.ltf_thickness
                .set_text_const_etomo_number(Some(&thickness));
        }
        ui_harness::INSTANCE.with(|harness| {
            harness.pack_axis_id_base_manager(
                Some(self.axis_id),
                Some(self.application_manager as &'static dyn BaseManager),
            )
        });
        !missing_data
    }

    /// Java `setTomopitchParam(ConstTomopitchParam)` (TomogramPositioningDialog.java:928).
    pub fn set_tomopitch_param(&self, tomopitch_param: &ConstTomopitchParam) {
        if !tomopitch_param.is_extra_thickness_null() {
            let extra_thickness = tomopitch_param.get_extra_thickness_string();
            if !self.is_sample_type_cryo() {
                self.ltf_extra_thickness
                    .set_text_string(Some(&extra_thickness));
            } else {
                self.ltf_extra_thickness_cryo
                    .set_text_string(Some(&extra_thickness));
            }
            self.cb_no_x_axis_tilt
                .set_selected_boolean(tomopitch_param.is_no_x_axis_tilt());
        }
    }

    /// Java `setWholeTomogram(boolean)` (TomogramPositioningDialog.java:945):
    /// set the whole tomogram sampling state.
    pub fn set_whole_tomogram(&self, state: bool) {
        self.cb_whole_tomogram.set_selected_boolean(state);
    }

    /// Java private `setToolTipText()` (TomogramPositioningDialog.java:1028):
    /// initialize the tooltip text for the axis panel objects.
    fn set_tool_tip_text(&self) {
        self.ltf_sample_thickness.set_tool_tip_text(Some(
            "Thickness of sample slices, or unbinned thickness of whole tomogram.  Make this \
             much larger than expected section thickness to see borders of section.",
        ));
        self.btn_sample
            .set_tool_tip_text(Some(SAMPLE_TOMOGRAMS_TOOLTIP));
        self.btn_create_boundary.set_tool_tip_text(Some(
            "Open samples in 3dmod to make a model with lines along top and bottom \
             edges of the section in each sample.",
        ));
        self.btn_tomopitch.set_tool_tip_text(Some(
            "Run tomopitch.  This will compute the positioning values and adjust the totals shown here.",
        ));
        self.cp_angle_offset.set_tool_tip_text(Some(
            "The total offset is sum of the original offset and the additional offset from tomopitch.",
        ));
        self.cp_x_axis_tilt
            .set_tool_tip_text(Some(tomogram_generation_dialog::X_AXIS_TILT_TOOLTIP));
        self.cp_tilt_axis_z_shift.set_tool_tip_text(Some(
            "The total shift is the sum of the original shift and theadditional shift from tomopitch.",
        ));
        self.btn_align
            .set_tool_tip_text(Some("Run tiltalign with these final offset parameters."));
        self.cb_whole_tomogram.set_tool_tip_text_string(Some(
            "Generate an entire tomogram instead of 3 samples and draw boundary \
             lines in this tomogram.",
        ));
        self.spin_binning.set_tool_tip_text(Some(
            "Set the binning for the whole tomogram to be used for positioning.  With a \
             binned tomogram, the tomopitch output and entries for offset and thickness will \
             still be in unbinned pixels.",
        ));
        self.cb_fiducialess
            .set_tool_tip_text_string(Some("Use cross-correlation alignment only."));
        self.ltf_rotation.set_tool_tip_text(Some(
            "Rotation angle of tilt axis for generating aligned stack from \
             cross-correlation alignment only.",
        ));
        let tooltip = "Extra thickness to be added to the top and bottom of the final tomogram.";
        self.ltf_extra_thickness.set_tool_tip_text(Some(tooltip));
        self.ltf_extra_thickness_cryo
            .set_tool_tip_text(Some(tooltip));
        self.cb_no_x_axis_tilt.set_tool_tip_text_string(Some(
            "Solve for the positioning parameters with X-axis tilt kept at 0.",
        ));
        self.ltf_thickness
            .set_tool_tip_text(Some("The thickness of the final tomogram."));
        self.cp_tilt_angle_offset.set_tool_tip_text(Some(
            "Tilt parameter:  the spatial frequency at which to switch from the R-weighted \
             radial filter to a Gaussian falloff.  Frequency is in cycles/pixel and ranges \
             from 0-0.5.  Both a cutoff and a falloff must be entered.",
        ));
        self.cp_z_shift.set_tool_tip_text(Some(
            "Tilt parameter:  amount to shift the reconstructed slices in Z before output.  \
             A positive value will shift the slice upward.  Do not use this option if you \
             have fiducials and the tomogram is part of a dual-axis series.",
        ));
        self.cb_use_gpu
            .set_tool_tip_text_string(Some("Check to run the tilt process on the graphics card."));
        self.cb_sample_type_auto.set_tool_tip_text_string(Some(
            "Find surfaces of material automatically using Findsection for plastic section \
             data or Cryposition for a cryo sample.",
        ));
        self.cb_sample_type_cryo.set_tool_tip_text_string(Some(
            "Use Cryoposition to find surfaces of material in low contrast reconstructions.",
        ));
        self.cb_has_gold_beads.set_tool_tip_text_string(Some(
            "Check this if there are any gold beads, even if not used for alignment; \
             Cryoposition needs to take them into account.",
        ));
        self.tf_bead_size
            .set_tool_tip_text(Some("Size of gold beads in unbinned pixels"));
    }
}

impl ProcessDialogVirtual for TomogramPositioningDialog {
    fn process_dialog(&self) -> &ProcessDialog {
        &self.base
    }

    /// Java override `done()` (TomogramPositioningDialog.java:1001).
    fn done(&self) {
        if let Some(expert) = self.expert.upgrade() {
            expert.done_dialog_void();
        }
        self.btn_sample
            .remove_action_listener(&self.local_action_listener);
        self.btn_tomopitch
            .remove_action_listener(&self.local_action_listener);
        self.btn_align
            .remove_action_listener(&self.local_action_listener);
        self.base.set_displayed(false);
    }
}

impl FiducialessParams for TomogramPositioningDialog {
    /// Java override `isFiducialess()` (TomogramPositioningDialog.java:550).
    fn is_fiducialess(&self) -> bool {
        self.cb_fiducialess.is_selected()
    }

    /// Java override `getImageRotation(boolean) throws NumberFormatException,
    /// FieldValidationFailedException` (TomogramPositioningDialog.java:559).
    fn get_image_rotation(
        &self,
        do_validation: bool,
    ) -> Result<String, FieldValidationFailedException> {
        // A JTextField never returns null text.
        self.ltf_rotation
            .get_text_boolean(do_validation)
            .map(|text| text.unwrap_or_default())
    }
}

impl ContextMenu for TomogramPositioningDialog {
    /// Java override `popUpContextMenu(MouseEvent)`
    /// (TomogramPositioningDialog.java:953): right mouse button context menu.
    fn pop_up_context_menu(&self, mouse_event: &MouseEvent) {
        let man_pagelabel: Vec<String> = [
            "Tomopitch",
            "Findsection",
            "Cryoposition",
            "Newstack",
            "3dmod",
            "Tilt",
        ]
        .iter()
        .map(|label| label.to_string())
        .collect();
        let man_page: Vec<String> = [
            "tomopitch.html",
            "findsection.html",
            "cryoposition.html",
            "newstack.html",
            "3dmod.html",
            "tilt.html",
        ]
        .iter()
        .map(|page| page.to_string())
        .collect();
        let log_file_label: Vec<String> = ["Tomopitch", "Sample", "Cryoposition"]
            .iter()
            .map(|label| label.to_string())
            .collect();
        let mut log_file: Vec<String> = vec![String::new(); 3];
        log_file[0] = format!("tomopitch{}.log", self.axis_id.get_extension());
        log_file[1] = format!("sample{}.log", self.axis_id.get_extension());
        log_file[2] = format!("cryoposition{}.log", self.axis_id.get_extension());
        let manager: &'static dyn BaseManager = self.application_manager;
        let _context_popup =
            ContextPopup::new_component_mouse_event_string_string_string_array_string_array_string_array_string_array_base_manager_axis_id(
                &self.base.root_panel.get_component(),
                mouse_event,
                Some("TOMOGRAM POSITIONING"),
                Some(context_popup::TOMO_GUIDE),
                &man_pagelabel,
                &man_page,
                Some(log_file_label.as_slice()),
                Some(log_file.as_slice()),
                manager,
                self.axis_id,
            );
    }
}

impl Run3dmodButtonContainer for TomogramPositioningDialog {
    /// Java override `action(String, Deferred3dmodButton, Run3dmodMenuOptions)`
    /// (TomogramPositioningDialog.java:978).  Executes the action associated
    /// with command.  Deferred3dmodButton is null if it comes from the
    /// dialog's ActionListener.  Otherwise is comes from a Run3dmodButton
    /// which called action(Run3dmodButton, Run3dmoMenuOptions).  In that case
    /// it will be null unless it was set in the Run3dmodButton.
    fn action(
        &self,
        command: &str,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    ) {
        let expert = self.expert.upgrade();
        if Some(command) == self.btn_sample.get_action_command().as_deref() {
            if let Some(expert) = &expert {
                let sample: ProcessResultDisplayHandle = self.btn_sample.clone();
                expert.sample_action(
                    Some(sample),
                    None,
                    deferred_3dmod_button,
                    run_3dmod_menu_options,
                );
            }
        } else if Some(command) == self.btn_tomopitch.get_action_command().as_deref() {
            if let Some(expert) = &expert {
                let display: ProcessResultDisplayHandle = self.btn_tomopitch.clone();
                expert.tomopitch(Some(display), None);
            }
        } else if Some(command) == self.btn_align.get_action_command().as_deref() {
            if let Some(expert) = &expert {
                let display: ProcessResultDisplayHandle = self.btn_align.clone();
                expert.final_align(Some(display), None);
            }
        } else if Some(command) == self.cb_fiducialess.get_action_command().as_deref() {
            if let Some(expert) = &expert {
                expert.fiducialess_action();
            }
        } else if Some(command) == self.btn_create_boundary.get_action_command().as_deref() {
            if let Some(expert) = &expert {
                expert.create_boundary(run_3dmod_menu_options);
            }
        } else {
            self.update_display();
        }
    }
}

/// Java `public static final class CalcPanel` (TomogramPositioningDialog.java:1083).
pub struct CalcPanel {
    /// Java private final `panel`.
    panel: Rc<SpacedPanel>,
    /// Java private final `label` (a `JLabel`).
    label: Rc<JComponent>,
    /// Java private final `ltfOriginal`.
    ltf_original: Rc<LabeledTextField>,
    /// Java private final `ltfAdded`.
    ltf_added: Rc<LabeledTextField>,
    /// Java private final `ltfTotal`.
    ltf_total: Rc<LabeledTextField>,
    /// Java private final `number` (utility field).
    number: RefCell<EtomoNumber>,
    /// Java private `more`.
    more: Cell<bool>,
}

/// Java `CalcPanel.ADDED_KEY`.
pub const ADDED_KEY: &str = "Added";
/// Java `CalcPanel.MAX_DIGITS`.
pub const MAX_DIGITS: i32 = 6;

impl CalcPanel {
    /// Java package-private constructor `CalcPanel(String)`
    /// (TomogramPositioningDialog.java:1097).
    pub fn new(label: &str) -> CalcPanel {
        // Field initializers.
        let panel = SpacedPanel::get_instance_void();
        let ltf_original =
            LabeledTextField::new_field_type_string(FieldType::FloatingPoint, Some("Original:"));
        let ltf_added = LabeledTextField::new_field_type_string(
            FieldType::FloatingPoint,
            Some(&format!("{ADDED_KEY}:")),
        );
        let ltf_total =
            LabeledTextField::new_field_type_string(FieldType::FloatingPoint, Some("Total:"));
        let number = RefCell::new(EtomoNumber::new_with_type(Some(Type::Double)));
        // Constructor body.
        let label = JComponent::new_label(&format!("{label}:"));
        // Swing layout: panel BoxLayout X_AXIS.
        panel.add_j_label(&label);
        // Swing layout: panel.addRigidArea().
        panel.add_labeled_text_field(&ltf_original);
        panel.add_labeled_text_field(&ltf_added);
        panel.add_labeled_text_field(&ltf_total);
        let calc_panel = CalcPanel {
            panel,
            label,
            ltf_original,
            ltf_added,
            ltf_total,
            number,
            more: Cell::new(true),
        };
        calc_panel.ltf_original.set_editable(false);
        calc_panel.ltf_original.set_columns(MAX_DIGITS);
        calc_panel.ltf_added.set_editable(false);
        calc_panel.ltf_added.set_columns(MAX_DIGITS);
        calc_panel.ltf_total.set_columns(MAX_DIGITS);
        calc_panel.ltf_original.set_text_string(Some("0.0"));
        calc_panel.ltf_added.set_text_string(Some("0.0"));
        calc_panel.ltf_total.set_text_string(Some("0.0"));
        calc_panel.update_display(false);
        calc_panel
    }

    /// Java `getContainer()` (TomogramPositioningDialog.java:1116).
    pub fn get_container(&self) -> Rc<JComponent> {
        self.panel.get_container()
    }

    /// Java `setEnabled(boolean)` (TomogramPositioningDialog.java:1120).
    pub fn set_enabled(&self, enabled: bool) {
        self.ltf_total.set_enabled(enabled);
    }

    /// Java `setToolTipText(String)` (TomogramPositioningDialog.java:1124).
    pub fn set_tool_tip_text(&self, text: Option<&str>) {
        self.label.set_tool_tip_text(text);
        self.ltf_original.set_tool_tip_text(text);
        self.ltf_added.set_tool_tip_text(text);
        self.ltf_total.set_tool_tip_text(text);
    }

    /// Java `setVisible(boolean)` (TomogramPositioningDialog.java:1131).
    pub fn set_visible(&self, visible: bool) {
        self.panel.set_visible(visible);
    }

    /// Java `set(ConstEtomoNumber)` (TomogramPositioningDialog.java:1135).
    pub fn set_const_etomo_number(&self, total: Option<&ConstEtomoNumber>) {
        self.update_display(false);
        self.set_const_etomo_number_labeled_text_field(total, &self.ltf_total);
    }

    /// Java `set(double)` (TomogramPositioningDialog.java:1140).
    pub fn set_double(&self, total: f64) {
        self.number.borrow_mut().set_double(total);
        // `set(number)`: a copy, so no borrow of `number` is held across the call.
        let number: ConstEtomoNumber = (**self.number.borrow()).clone();
        self.set_const_etomo_number(Some(&number));
    }

    /// Java `set(ConstEtomoNumber, ConstEtomoNumber, ConstEtomoNumber)`
    /// (TomogramPositioningDialog.java:1145).  `total` is dereferenced
    /// unconditionally in the Java (every caller passes a non-null value).
    pub fn set_const_etomo_number_const_etomo_number_const_etomo_number(
        &self,
        original: Option<&ConstEtomoNumber>,
        added: Option<&ConstEtomoNumber>,
        total: &ConstEtomoNumber,
    ) -> bool {
        if total.is_null() {
            return false;
        }
        self.update_display(true);
        self.set_const_etomo_number_labeled_text_field(original, &self.ltf_original);
        self.set_const_etomo_number_labeled_text_field(added, &self.ltf_added);
        self.set_const_etomo_number_labeled_text_field(Some(total), &self.ltf_total);
        true
    }

    /// Java private `set(ConstEtomoNumber, LabeledTextField)`
    /// (TomogramPositioningDialog.java:1157).
    fn set_const_etomo_number_labeled_text_field(
        &self,
        number: Option<&ConstEtomoNumber>,
        field: &LabeledTextField,
    ) {
        match number {
            Some(number) if !number.is_null() && number.is_valid() => {
                field.set_text_string(Some(&number.to_string()));
            }
            _ => field.set_text_string(Some("0.0")),
        }
    }

    /// Java `updateDisplay(boolean)` (TomogramPositioningDialog.java:1166).
    pub fn update_display(&self, more: bool) {
        if self.more.get() != more {
            self.more.set(more);
            self.ltf_original.set_visible(more);
            self.ltf_added.set_visible(more);
        }
    }

    /// Java `getOriginal(boolean) throws FieldValidationFailedException`
    /// (TomogramPositioningDialog.java:1174).
    pub fn get_original(
        &self,
        do_validation: bool,
    ) -> Result<String, FieldValidationFailedException> {
        // A JTextField never returns null text.
        self.ltf_original
            .get_text_boolean(do_validation)
            .map(|text| text.unwrap_or_default())
    }

    /// Java private `getTotal(boolean) throws FieldValidationFailedException`
    /// (TomogramPositioningDialog.java:1178).  Private to `CalcPanel` in
    /// the Java, but read by the enclosing dialog (Java nested-class access).
    fn get_total_boolean(
        &self,
        do_validation: bool,
    ) -> Result<String, FieldValidationFailedException> {
        self.ltf_total
            .get_text_boolean(do_validation)
            .map(|text| text.unwrap_or_default())
    }

    /// Java private `getTotal()` (TomogramPositioningDialog.java:1183).
    fn get_total_void(&self) -> String {
        self.ltf_total.get_text_void().unwrap_or_default()
    }
}
