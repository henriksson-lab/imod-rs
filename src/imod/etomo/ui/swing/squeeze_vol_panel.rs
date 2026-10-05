//! `IMOD/Etomo/src/etomo/ui/swing/SqueezeVolPanel.java`.
//!
//! Java `final class SqueezeVolPanel implements Run3dmodButtonContainer,
//! ContextMenu, ReduceFiltVolDisplay, FieldDisplayer`: the "Reduce/filt vol"
//! tab of the Post Processing dialog (reducefiltvol: reduce the trimmed or
//! flattened volume by a factor and/or filter it).
//!
//! An EDT object (`ui.md`): created as `Rc<Self>` by
//! [`SqueezeVolPanel::get_instance`]; every method takes `&self`.  The
//! listener class `SqueezeVolPanelActionListener` is a closure holding a weak
//! reference to the panel.  The panel hands itself to its fields as their
//! `FieldDisplayer` (`getText(doValidation, this)`); that `this` is
//! `self_ref`, upgraded.
//!
//! The class's `setParameters`/`getParameters` overloads carry the
//! parameter-type suffix; the `ReduceFiltVolDisplay` `getParameters` is the
//! trait impl.

use std::path::PathBuf;
use std::rc::{Rc, Weak};

use super::beveled_border::BeveledBorder;
use super::check_box::CheckBox;
use super::context_menu::ContextMenu;
use super::generic_mouse_adapter::GenericMouseAdapter;
use super::context_popup::{self, ContextPopup};
use super::deferred_3dmod_button::Deferred3dmodButton;
use super::labeled_text_field::LabeledTextField;
use super::radio_button::RadioButton;
use super::reduce_filt_vol_display::ReduceFiltVolDisplay;
use super::run_3dmod_button::Run3dmodButton;
use super::run_3dmod_button_container::Run3dmodButtonContainer;
use super::spaced_panel::{self, SpacedPanel};
use super::text_field::TextField;
use super::ui_harness;
use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::const_squeezevol_param::ConstSqueezevolParam;
use crate::imod::etomo::comscript::fortran_input_syntax_exception::FortranInputSyntaxException;
use crate::imod::etomo::comscript::makecomfile_param::MakecomfileParam;
use crate::imod::etomo::comscript::reduce_filt_vol_param::{self, ReduceFiltVolParam};
use crate::imod::etomo::comscript::squeezevol_param;
use crate::imod::etomo::jdk::{ActionEvent, ActionListener, ButtonGroup, JComponent, MouseEvent};
use crate::imod::etomo::logic::converter;
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::storage::autodoc::autodoc::Autodoc;
use crate::imod::etomo::storage::autodoc::autodoc_factory;
use crate::imod::etomo::storage::autodoc::read_only_autodoc::ReadOnlyAutodoc;
use crate::imod::etomo::storage::log_file::LogFileError;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::{
    ConstEtomoNumber, Number, java_lang_double_to_string,
};
use crate::imod::etomo::r#type::const_meta_data::ConstMetaData;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::etomo_autodoc;
use crate::imod::etomo::r#type::file_type;
use crate::imod::etomo::r#type::image_file_type::ImageFileType;
use crate::imod::etomo::r#type::meta_data::MetaData;
use crate::imod::etomo::r#type::process_result_display::ProcessResultDisplayHandle;
use crate::imod::etomo::r#type::recon_screen_state::ReconScreenState;
use crate::imod::etomo::ui::field::Field;
use crate::imod::etomo::ui::field_displayer::FieldDisplayer;
use crate::imod::etomo::ui::field_type::FieldType;
use crate::imod::etomo::ui::field_validation_failed_exception::FieldValidationFailedException;
use crate::imod::etomo::ui::ui_component::UIComponent;
use crate::imod::etomo::util::mrc_header::MRCHeader;

/// Java private static final `USE_TRIM_VOL_OUTPUT_LABEL`.
const USE_TRIM_VOL_OUTPUT_LABEL: &str = "Use the trimvol output";
/// Java private static final `USE_FLATTEN_OUTPUT_LABEL`.
const USE_FLATTEN_OUTPUT_LABEL: &str = "Use the flatten output";
/// Java private static final `REDUCE_BY_OVERALL_FACTOR_LABEL`.
const REDUCE_BY_OVERALL_FACTOR_LABEL: &str = "Reduce by overall factor (>=1)";
/// Java private static final `FACTOR_IN_Z_LABEL`.
const FACTOR_IN_Z_LABEL: &str = "Factor in Z ";
/// Java private static final `FILTER_NONE_LABEL`.
const FILTER_NONE_LABEL: &str = "None";
/// Java private static final `FILTER_GAUSSIAN_LOW_PASS_LABEL`.
const FILTER_GAUSSIAN_LOW_PASS_LABEL: &str = "Gaussian low-pass";
/// Java private static final `FILTER_DECONVOLUTION_LABEL`.
const FILTER_DECONVOLUTION_LABEL: &str = "Deconvolution";
/// Java private static final `LOW_PASS_CUTOFF_SIGMA_LABEL`.
const LOW_PASS_CUTOFF_SIGMA_LABEL: &str = "Low pass cutoff and sigma (1/pixel) ";
/// Java private static final `DECONVOLUTION_STRENGTH_LABEL`.
const DECONVOLUTION_STRENGTH_LABEL: &str = "Deconvolution strength ";
/// Java private static final `SNR_FALLOFF_LABEL`.
const SNR_FALLOFF_LABEL: &str = "SNR falloff ";
/// Java private static final `HIGH_PASS_FILTER_CUTOFF_LABEL`.
const HIGH_PASS_FILTER_CUTOFF_LABEL: &str = "High pass filter cutoff (fraction of Nyquist) ";
/// Java private static final `DEOCUS_LABEL` (sic).
const DEOCUS_LABEL: &str = "Defocus (microns) ";
/// Java private static final `PHASE_SHIFT_LABEL`.
const PHASE_SHIFT_LABEL: &str = "Phase shift (degrees) ";
/// Java private static final `DATA_MODE_OF_OUTPUT_LABEL`.
const DATA_MODE_OF_OUTPUT_LABEL: &str = "Data mode of output ";
/// Java private static final `BTN_IMOD_REDUCE_FILT_VOL_LABEL`.
const BTN_IMOD_REDUCE_FILT_VOL_LABEL: &str = "Open Output Volume in 3dmod";

/// Java `final class SqueezeVolPanel`.
pub struct SqueezeVolPanel {
    /// Rust-only: Java `this` (the panel as its fields' FieldDisplayer, and
    /// `getReduceFiltVolDisplay`).
    self_ref: Weak<SqueezeVolPanel>,

    /// Java private final `actionListener = new
    /// SqueezeVolPanelActionListener(this)`.
    action_listener: ActionListener,
    /// Java private final `pnlRoot = SpacedPanel.getInstance()`.
    pnl_root: Rc<SpacedPanel>,
    /// Java private final `bgInputFile = new ButtonGroup()`.
    bg_input_file: Rc<ButtonGroup>,
    /// Java private final `rbInputFileTrimVol`.
    rb_input_file_trim_vol: Rc<RadioButton>,
    /// Java private final `rbInputFileFlattenWarp`.
    rb_input_file_flatten_warp: Rc<RadioButton>,
    /// Java private final `cbReduceByOverallFactor`.
    cb_reduce_by_overall_factor: Rc<CheckBox>,
    /// Java private final `tfReduceByOverallFactor`.
    tf_reduce_by_overall_factor: Rc<TextField>,
    /// Java private final `ltfFactorInZ`.
    ltf_factor_in_z: Rc<LabeledTextField>,
    /// Java private final `bgFiltering = new ButtonGroup()`.
    bg_filtering: Rc<ButtonGroup>,
    /// Java private final `rbFilteringNone`.
    rb_filtering_none: Rc<RadioButton>,
    /// Java private final `rbFilteringGaussian`.
    rb_filtering_gaussian: Rc<RadioButton>,
    /// Java private final `rbFilteringDeconvolution`.
    rb_filtering_deconvolution: Rc<RadioButton>,
    /// Java private final `ltfLowPassCutoffSigma`.
    ltf_low_pass_cutoff_sigma: Rc<LabeledTextField>,
    /// Java private final `ltfDeconvolutionStrength`.
    ltf_deconvolution_strength: Rc<LabeledTextField>,
    /// Java private final `ltfSNRFalloff`.
    ltf_snr_falloff: Rc<LabeledTextField>,
    /// Java private final `ltfHighPassFilterCutoff`.
    ltf_high_pass_filter_cutoff: Rc<LabeledTextField>,
    /// Java private final `ltfDefocus`.
    ltf_defocus: Rc<LabeledTextField>,
    /// Java private final `ltfPhaseShift`.
    ltf_phase_shift: Rc<LabeledTextField>,
    /// Java private final `ltfDataModeOfOutput`.
    ltf_data_mode_of_output: Rc<LabeledTextField>,
    /// Java private final `btnImodReduceFiltVol`.
    btn_imod_reduce_filt_vol: Rc<Run3dmodButton>,

    /// Java private final `btnReduceFiltVol`.
    btn_reduce_filt_vol: Rc<Run3dmodButton>,
    /// Java private final `manager`.
    manager: &'static ApplicationManager,
    /// Java private final `axisID`.
    axis_id: AxisID,
    /// Java private final `dialogType`.
    dialog_type: DialogType,
}

impl SqueezeVolPanel {
    /// Java private constructor `SqueezeVolPanel(ApplicationManager, AxisID,
    /// DialogType)`, with the field initializers.
    fn new(
        manager: &'static ApplicationManager,
        axis_id: AxisID,
        dialog_type: DialogType,
    ) -> Rc<SqueezeVolPanel> {
        let instance = Rc::new_cyclic(|self_ref: &Weak<SqueezeVolPanel>| {
            // Field initializers.
            // Java `new SqueezeVolPanelActionListener(this)`.
            let adaptee = self_ref.clone();
            let action_listener: ActionListener = Rc::new(move |event: &ActionEvent| {
                if let Some(adaptee) = adaptee.upgrade() {
                    adaptee.action(event.get_action_command().unwrap_or(""), None, None);
                }
            });
            let pnl_root = SpacedPanel::get_instance_void();
            let bg_input_file = ButtonGroup::new();
            let rb_input_file_trim_vol = RadioButton::new_string_button_group(
                Some(USE_TRIM_VOL_OUTPUT_LABEL),
                Some(&bg_input_file),
            );
            let rb_input_file_flatten_warp = RadioButton::new_string_button_group(
                Some(USE_FLATTEN_OUTPUT_LABEL),
                Some(&bg_input_file),
            );
            let cb_reduce_by_overall_factor =
                CheckBox::new_string(Some(REDUCE_BY_OVERALL_FACTOR_LABEL));
            let tf_reduce_by_overall_factor = TextField::new(
                FieldType::FloatingPoint,
                Some(REDUCE_BY_OVERALL_FACTOR_LABEL),
                None,
            );
            let ltf_factor_in_z = LabeledTextField::new_field_type_string(
                FieldType::FloatingPoint,
                Some(FACTOR_IN_Z_LABEL),
            );
            let bg_filtering = ButtonGroup::new();
            let rb_filtering_none =
                RadioButton::new_string_button_group(Some(FILTER_NONE_LABEL), Some(&bg_filtering));
            let rb_filtering_gaussian = RadioButton::new_string_button_group(
                Some(FILTER_GAUSSIAN_LOW_PASS_LABEL),
                Some(&bg_filtering),
            );
            let rb_filtering_deconvolution = RadioButton::new_string_button_group(
                Some(FILTER_DECONVOLUTION_LABEL),
                Some(&bg_filtering),
            );
            let ltf_low_pass_cutoff_sigma = LabeledTextField::new_field_type_string(
                FieldType::FloatingPointPair,
                Some(LOW_PASS_CUTOFF_SIGMA_LABEL),
            );
            let ltf_deconvolution_strength = LabeledTextField::new_field_type_string(
                FieldType::FloatingPoint,
                Some(DECONVOLUTION_STRENGTH_LABEL),
            );
            let ltf_snr_falloff = LabeledTextField::new_field_type_string(
                FieldType::FloatingPoint,
                Some(SNR_FALLOFF_LABEL),
            );
            let ltf_high_pass_filter_cutoff = LabeledTextField::new_field_type_string(
                FieldType::FloatingPoint,
                Some(HIGH_PASS_FILTER_CUTOFF_LABEL),
            );
            let ltf_defocus = LabeledTextField::new_field_type_string(
                FieldType::FloatingPoint,
                Some(DEOCUS_LABEL),
            );
            let ltf_phase_shift = LabeledTextField::new_field_type_string(
                FieldType::FloatingPoint,
                Some(PHASE_SHIFT_LABEL),
            );
            let ltf_data_mode_of_output = LabeledTextField::new_field_type_string(
                FieldType::FloatingPoint,
                Some(DATA_MODE_OF_OUTPUT_LABEL),
            );
            let btn_imod_reduce_filt_vol =
                Run3dmodButton::get_3dmod_instance_string_run_3dmod_button_container(
                    Some(BTN_IMOD_REDUCE_FILT_VOL_LABEL),
                    Some(self_ref.clone() as Weak<dyn Run3dmodButtonContainer>),
                );
            // Constructor body.
            let btn_reduce_filt_vol = manager
                .get_process_result_display_factory(axis_id)
                .get_squeeze_volume();
            SqueezeVolPanel {
                self_ref: self_ref.clone(),
                action_listener,
                pnl_root,
                bg_input_file,
                rb_input_file_trim_vol,
                rb_input_file_flatten_warp,
                cb_reduce_by_overall_factor,
                tf_reduce_by_overall_factor,
                ltf_factor_in_z,
                bg_filtering,
                rb_filtering_none,
                rb_filtering_gaussian,
                rb_filtering_deconvolution,
                ltf_low_pass_cutoff_sigma,
                ltf_deconvolution_strength,
                ltf_snr_falloff,
                ltf_high_pass_filter_cutoff,
                ltf_defocus,
                ltf_phase_shift,
                ltf_data_mode_of_output,
                btn_imod_reduce_filt_vol,
                btn_reduce_filt_vol,
                manager,
                axis_id,
                dialog_type,
            }
        });
        // The last statement of the Java constructor (it needs `this`).
        instance.set_field_displayer();
        instance
    }

    /// Java package-private static `getInstance(ApplicationManager, AxisID,
    /// DialogType)`.
    pub fn get_instance(
        manager: &'static ApplicationManager,
        axis_id: AxisID,
        dialog_type: DialogType,
    ) -> Rc<SqueezeVolPanel> {
        let instance = SqueezeVolPanel::new(manager, axis_id, dialog_type);
        instance.create_panel();
        instance.set_tool_tip_text();
        instance.add_listeners();
        instance
    }

    /// Rust-only: Java `this` as a `FieldDisplayer` (`None` only while the
    /// panel is being dropped).
    fn this_field_displayer(&self) -> Option<Rc<dyn FieldDisplayer>> {
        self.self_ref
            .upgrade()
            .map(|this| this as Rc<dyn FieldDisplayer>)
    }

    /// Java private `addListeners()`.
    fn add_listeners(self: &Rc<Self>) {
        let context_menu: Weak<dyn ContextMenu> = Rc::downgrade(self) as Weak<dyn ContextMenu>;
        self.pnl_root.add_mouse_listener(GenericMouseAdapter::new(context_menu));
        self.rb_input_file_trim_vol
            .add_action_listener(self.action_listener.clone());
        self.rb_input_file_flatten_warp
            .add_action_listener(self.action_listener.clone());
        self.cb_reduce_by_overall_factor
            .add_action_listener(Some(self.action_listener.clone()));
        self.ltf_factor_in_z
            .add_action_listener(self.action_listener.clone());
        self.rb_filtering_none
            .add_action_listener(self.action_listener.clone());
        self.rb_filtering_gaussian
            .add_action_listener(self.action_listener.clone());
        self.rb_filtering_deconvolution
            .add_action_listener(self.action_listener.clone());
        self.ltf_low_pass_cutoff_sigma
            .add_action_listener(self.action_listener.clone());
        self.ltf_deconvolution_strength
            .add_action_listener(self.action_listener.clone());
        self.ltf_snr_falloff
            .add_action_listener(self.action_listener.clone());
        self.ltf_high_pass_filter_cutoff
            .add_action_listener(self.action_listener.clone());
        self.ltf_defocus
            .add_action_listener(self.action_listener.clone());
        self.ltf_phase_shift
            .add_action_listener(self.action_listener.clone());
        self.ltf_data_mode_of_output
            .add_action_listener(self.action_listener.clone());
        self.btn_reduce_filt_vol
            .add_action_listener(self.action_listener.clone());
        self.btn_imod_reduce_filt_vol
            .add_action_listener(self.action_listener.clone());
    }

    /// Java private `createPanel()`.
    fn create_panel(&self) {
        self.rb_input_file_trim_vol.set_selected_boolean(true);
        self.rb_filtering_none.set_selected_boolean(true);
        self.tf_reduce_by_overall_factor.set_preferred_width(70);
        self.ltf_factor_in_z.set_preferred_width(70);
        self.ltf_low_pass_cutoff_sigma
            .set_number_must_be_positive(true);
        self.ltf_deconvolution_strength.set_required(true);
        self.ltf_deconvolution_strength
            .set_number_must_be_positive(true);
        self.ltf_snr_falloff.set_number_must_be_positive(true);
        self.ltf_high_pass_filter_cutoff
            .set_number_must_be_positive(true);
        self.ltf_high_pass_filter_cutoff
            .set_minimum(reduce_filt_vol_param::HIGH_PASS_NYQUIST_MIN);
        self.ltf_high_pass_filter_cutoff
            .set_maximum(reduce_filt_vol_param::HIGH_PASS_NYQUIST_MAX);
        self.ltf_defocus.set_number_must_be_positive(true);
        // root panel
        self.pnl_root.set_box_layout(spaced_panel::Y_AXIS);
        self.pnl_root
            .set_border(&BeveledBorder::new(Some("Reduce and/or Filter Volume")).get_border());
        // Choose input file
        let pnl_input_file = JComponent::new_panel();
        // Swing layout: pnlInputFile Y_AXIS BoxLayout, CENTER_ALIGNMENT.
        pnl_input_file.set_border_title(
            BeveledBorder::new(Some("Set Input File"))
                .get_border()
                .get_title()
                .as_deref(),
        );
        pnl_input_file.add(&self.rb_input_file_trim_vol.get_component());
        pnl_input_file.add(&self.rb_input_file_flatten_warp.get_component());
        self.pnl_root.add_j_panel(&pnl_input_file);
        // Reduce by overall factor panel
        let pnl_reduce_by_overall_factor = SpacedPanel::get_instance_void();
        pnl_reduce_by_overall_factor.set_box_layout(spaced_panel::X_AXIS);
        pnl_reduce_by_overall_factor
            .add_component(&self.cb_reduce_by_overall_factor.get_component());
        pnl_reduce_by_overall_factor.add_text_field(&self.tf_reduce_by_overall_factor);
        // Swing layout: pnlReduceByOverallFactor.add(Box.createRigidArea(
        // FixedDim.x10_y0)).
        pnl_reduce_by_overall_factor.add_labeled_text_field(&self.ltf_factor_in_z);
        self.pnl_root
            .add_spaced_panel(&pnl_reduce_by_overall_factor);
        // Filtering
        let pnl_filtering = SpacedPanel::get_instance_void();
        pnl_filtering.set_box_layout(spaced_panel::Y_AXIS);
        pnl_filtering.set_border(&BeveledBorder::new(Some("Filtering")).get_border());
        let pnl_filtering_radio_buttons = SpacedPanel::get_instance_void();
        pnl_filtering_radio_buttons.set_box_layout(spaced_panel::X_AXIS);
        pnl_filtering_radio_buttons.add_component(&self.rb_filtering_none.get_component());
        pnl_filtering_radio_buttons.add_component(&self.rb_filtering_gaussian.get_component());
        pnl_filtering_radio_buttons.add_component(&self.rb_filtering_deconvolution.get_component());
        pnl_filtering.add_spaced_panel(&pnl_filtering_radio_buttons);
        let pnl_low_pass_cutoff = SpacedPanel::get_instance_void();
        pnl_low_pass_cutoff.set_box_layout(spaced_panel::X_AXIS);
        pnl_low_pass_cutoff.add_labeled_text_field(&self.ltf_low_pass_cutoff_sigma);
        pnl_filtering.add_spaced_panel(&pnl_low_pass_cutoff);
        let pnl_deconvolution = SpacedPanel::get_instance_void();
        pnl_deconvolution.set_box_layout(spaced_panel::X_AXIS);
        pnl_deconvolution.add_labeled_text_field(&self.ltf_deconvolution_strength);
        // Swing layout: pnlDeconvolution.add(Box.createRigidArea(FixedDim.x10_y0)).
        pnl_deconvolution.add_labeled_text_field(&self.ltf_snr_falloff);
        pnl_filtering.add_spaced_panel(&pnl_deconvolution);
        let pnl_high_pass_filter = SpacedPanel::get_instance_void();
        pnl_high_pass_filter.set_box_layout(spaced_panel::X_AXIS);
        pnl_high_pass_filter.add_labeled_text_field(&self.ltf_high_pass_filter_cutoff);
        pnl_filtering.add_spaced_panel(&pnl_high_pass_filter);
        let pnl_defocus = SpacedPanel::get_instance_void();
        pnl_defocus.set_box_layout(spaced_panel::X_AXIS);
        pnl_defocus.add_labeled_text_field(&self.ltf_defocus);
        // Swing layout: pnlDefocus.add(Box.createRigidArea(FixedDim.x10_y0)).
        pnl_defocus.add_labeled_text_field(&self.ltf_phase_shift);
        pnl_filtering.add_spaced_panel(&pnl_defocus);
        self.pnl_root.add_spaced_panel(&pnl_filtering);
        // Data mode of output panel
        let pnl_data_mode_of_output = SpacedPanel::get_instance_void();
        pnl_data_mode_of_output.set_box_layout(spaced_panel::X_AXIS);
        pnl_data_mode_of_output.add_labeled_text_field(&self.ltf_data_mode_of_output);
        self.pnl_root.add_spaced_panel(&pnl_data_mode_of_output);
        // third component
        let pnl_buttons = SpacedPanel::get_instance_void();
        pnl_buttons.set_box_layout(spaced_panel::X_AXIS);
        self.btn_reduce_filt_vol.set_container(Some(
            self.self_ref.clone() as Weak<dyn Run3dmodButtonContainer>
        ));
        self.btn_reduce_filt_vol
            .set_deferred_3dmod_button_deferred_3dmod_button(Some(
                self.btn_imod_reduce_filt_vol.clone() as Rc<dyn Deferred3dmodButton>,
            ));
        pnl_buttons.add_multi_line_button(&self.btn_reduce_filt_vol);
        pnl_buttons.add_horizontal_glue();
        pnl_buttons.add_multi_line_button(&self.btn_imod_reduce_filt_vol);
        self.pnl_root.add_spaced_panel(&pnl_buttons);

        self.update_display();
    }

    /// Java private `setToolTipText()`.
    fn set_tool_tip_text(&self) {
        // Java `ReadOnlyAutodoc autodoc = null;` then the try/catch.
        let mut autodoc: *const dyn ReadOnlyAutodoc = std::ptr::null::<Autodoc>();
        // Java passes a null AxisID; `AutodocFactory.getInstance` takes the
        // axis by value here (see NEEDS), and it is only used for the error
        // message's frame.
        let manager: &'static dyn BaseManager = self.manager;
        match unsafe {
            autodoc_factory::get_instance(
                Some(manager),
                Some(autodoc_factory::REDUCE_FILTER_VOLUME),
                AxisID::Only,
                false,
            )
        } {
            Ok(instance) => autodoc = instance as *const Autodoc,
            // `catch (final LockException except) {}`.
            Err(LogFileError::Lock(_)) => {}
            // `catch (final LogFileException | IOException except)`:
            // `except.printStackTrace()`.
            Err(except) => eprintln!("{}", except),
        }
        // SAFETY: `autodoc` is null or an autodoc the factory keeps for the life
        // of the process.
        let autodoc: Option<&dyn ReadOnlyAutodoc> = if autodoc.is_null() {
            None
        } else {
            Some(unsafe { &*autodoc })
        };
        if let Some(read_only_autodoc) = autodoc {
            let _autodoc_name = read_only_autodoc.get_autodoc_name();
            self.rb_input_file_trim_vol.set_tool_tip_text_string(
                etomo_autodoc::get_tooltip(autodoc, Some(reduce_filt_vol_param::INPUT_FILE))
                    .as_deref(),
            );
            self.rb_input_file_flatten_warp.set_tool_tip_text_string(
                etomo_autodoc::get_tooltip(autodoc, Some(reduce_filt_vol_param::INPUT_FILE))
                    .as_deref(),
            );
            self.cb_reduce_by_overall_factor.set_tool_tip_text_string(
                etomo_autodoc::get_tooltip(autodoc, Some(reduce_filt_vol_param::REDUCTION_FACTOR))
                    .as_deref(),
            );
            self.tf_reduce_by_overall_factor.set_tool_tip_text(
                etomo_autodoc::get_tooltip(autodoc, Some(reduce_filt_vol_param::REDUCTION_FACTOR))
                    .as_deref(),
            );
            self.ltf_factor_in_z.set_tool_tip_text(
                etomo_autodoc::get_tooltip(
                    autodoc,
                    Some(reduce_filt_vol_param::Z_REDUCTION_FACTOR),
                )
                .as_deref(),
            );
            self.ltf_low_pass_cutoff_sigma.set_tool_tip_text(
                etomo_autodoc::get_tooltip(
                    autodoc,
                    Some(reduce_filt_vol_param::LOW_PASS_RADIUS_SIGMA),
                )
                .as_deref(),
            );
            self.ltf_deconvolution_strength.set_tool_tip_text(
                etomo_autodoc::get_tooltip(
                    autodoc,
                    Some(reduce_filt_vol_param::DECONVOLUTION_STRENGTH),
                )
                .as_deref(),
            );
            self.ltf_snr_falloff.set_tool_tip_text(
                etomo_autodoc::get_tooltip(autodoc, Some(reduce_filt_vol_param::SNR_FALLOFF))
                    .as_deref(),
            );
            self.ltf_high_pass_filter_cutoff.set_tool_tip_text(
                etomo_autodoc::get_tooltip(autodoc, Some(reduce_filt_vol_param::HIGH_PASS_NYQUIST))
                    .as_deref(),
            );
            self.ltf_defocus.set_tool_tip_text(
                etomo_autodoc::get_tooltip(
                    autodoc,
                    Some(reduce_filt_vol_param::DEFOCUS_IN_MICRONS),
                )
                .as_deref(),
            );
            self.ltf_phase_shift.set_tool_tip_text(
                etomo_autodoc::get_tooltip(autodoc, Some(reduce_filt_vol_param::PHASE_SHIFT))
                    .as_deref(),
            );
            self.ltf_data_mode_of_output.set_tool_tip_text(
                etomo_autodoc::get_tooltip(autodoc, Some(reduce_filt_vol_param::MODE_TO_OUTPUT))
                    .as_deref(),
            );
        }
        self.btn_reduce_filt_vol
            .set_tool_tip_text(Some("Run reducefiltvol on the input volume"));
        self.btn_imod_reduce_filt_vol
            .set_tool_tip_text(Some("View the reduced and/or filtered volume"));
    }

    /// Java package-private `setParameters(ConstSqueezevolParam)`.  Set the
    /// panel values with the specified parameters.
    pub fn set_parameters_const_squeezevol_param(
        &self,
        squeezevol_param: &dyn ConstSqueezevolParam,
    ) {
        // Ideally the Reduction Factor fields should be set from the values that
        // were used to run squeezevol on the input file. But there was no
        // squeezevol log, so it's very difficult to figure out whether this
        // happened. The most practical thing to do is be right as often as
        // possible.
        //
        // The default value (1.25) was always added, even if the original
        // Squeezevol dialog was never opened - so we're ignoring them.
        // Non-default values mean that the user at least opened the original
        // Squeezevol dialog, so we're keeping them.
        let mut reduction_factor: &ConstEtomoNumber = squeezevol_param.get_reduction_factor_x();
        if !reduction_factor.equals_number(Some(Number::Float(
            squeezevol_param::REDUCTION_FACTOR_DEFAULT,
        ))) {
            self.tf_reduce_by_overall_factor
                .set_text_string(Some(&reduction_factor.to_string()));
        }
        if self.manager.is_squeezevol_flipped() {
            reduction_factor = squeezevol_param.get_reduction_factor_z();
        } else {
            reduction_factor = squeezevol_param.get_reduction_factor_y();
        }
        if !reduction_factor.equals_number(Some(Number::Float(
            squeezevol_param::REDUCTION_FACTOR_DEFAULT,
        ))) {
            self.ltf_factor_in_z
                .set_text_const_etomo_number(Some(reduction_factor));
        }
    }

    /// Java package-private `setParameters(ReduceFiltVolParam, boolean,
    /// boolean)`.
    pub fn set_parameters_reduce_filt_vol_param_boolean_boolean(
        &self,
        reduce_filt_vol_param: &ReduceFiltVolParam,
        dialog_not_exists: bool,
        com_file_exists: bool,
    ) {
        let mut file_too_big = false;
        if dialog_not_exists {
            let trim_vol_output_header = MRCHeader::get_instance_from_file_type(
                self.manager,
                Some(self.axis_id),
                &file_type::CLASS.trim_vol_output,
            );
            // `MRCHeader.getInstance` never returns null in the source.
            if let Some(trim_vol_output_header) = trim_vol_output_header {
                // catch (IOException | InvalidParameterException e)
                //   { e.printStackTrace(); }
                let read = trim_vol_output_header
                    .borrow_mut()
                    .read_with_manager(self.manager);
                if let Err(e) = read {
                    eprintln!("{e}");
                }
                let n_rows = trim_vol_output_header.borrow().get_n_rows();
                let n_columns = trim_vol_output_header.borrow().get_n_columns();
                // Java `int * int` wraps; the product widens for the comparison.
                if reduce_filt_vol_param::TRIM_VOL_OUTPUT_MIN_PIXEL_AREA
                    < (n_rows.wrapping_mul(n_columns)) as f64
                {
                    self.cb_reduce_by_overall_factor.set_selected_boolean(true);
                    file_too_big = true;
                }
            }
        }

        let reduction_factor = reduce_filt_vol_param.is_reduction_factor();
        let z_reduction_factor = reduce_filt_vol_param.is_z_reduction_factor();
        if reduction_factor || z_reduction_factor {
            self.cb_reduce_by_overall_factor.set_selected_boolean(true);
            if reduction_factor {
                self.tf_reduce_by_overall_factor
                    .set_text_string(Some(&reduce_filt_vol_param.get_reduction_factor()));
            }
            if z_reduction_factor {
                self.ltf_factor_in_z
                    .set_text_string(Some(&reduce_filt_vol_param.get_z_reduction_factor()));
            }
        } else if com_file_exists {
            self.cb_reduce_by_overall_factor.set_selected_boolean(false);
        }

        if reduce_filt_vol_param.is_low_pass_radius_sigma() {
            self.rb_filtering_gaussian.set_selected_boolean(true);
            self.ltf_low_pass_cutoff_sigma
                .set_text_string(Some(&reduce_filt_vol_param.get_low_pass_radius_sigma()));
        }
        if reduce_filt_vol_param.is_deconvolution_strength() {
            self.rb_filtering_deconvolution.set_selected_boolean(true);
            self.ltf_deconvolution_strength
                .set_text_string(Some(&reduce_filt_vol_param.get_deconvolution_strength()));
        }
        if reduce_filt_vol_param.is_snr_falloff() {
            self.ltf_snr_falloff
                .set_text_string(Some(&reduce_filt_vol_param.get_snr_falloff()));
        }
        if reduce_filt_vol_param.is_high_pass_nyquist() {
            self.ltf_high_pass_filter_cutoff
                .set_text_string(Some(&reduce_filt_vol_param.get_high_pass_nyquist()));
        }
        if reduce_filt_vol_param.is_defocus_in_microns() {
            self.ltf_defocus
                .set_text_string(Some(&reduce_filt_vol_param.get_defocus_in_microns()));
        }
        if reduce_filt_vol_param.is_phase_shift() {
            self.ltf_phase_shift
                .set_text_string(Some(&reduce_filt_vol_param.get_phase_shift()));
        }
        if reduce_filt_vol_param.is_mode_to_output() {
            self.ltf_data_mode_of_output
                .set_text_string(Some(&reduce_filt_vol_param.get_mode_to_output()));
        }

        self.update_display();

        if !com_file_exists && !file_too_big {
            if !self.tf_reduce_by_overall_factor.is_empty()
                || (!self.ltf_factor_in_z.is_empty() && self.ltf_factor_in_z.is_enabled())
            {
                self.cb_reduce_by_overall_factor.set_selected_boolean(true);
            } else {
                self.cb_reduce_by_overall_factor.set_selected_boolean(false);
            }
        }

        self.update_display();
    }

    /// Java package-private `getParameters(MetaData)`.
    pub fn get_parameters_meta_data(&self, meta_data: &MetaData) {
        meta_data.set_post_squeeze_vol_input_trim_vol(self.rb_input_file_trim_vol.is_selected());
        meta_data.set_post_reduce_filt_vol_reduction_factor(
            self.tf_reduce_by_overall_factor.get_text_void().as_deref(),
        );
        meta_data.set_post_reduce_filt_vol_z_reduction_factor(
            self.ltf_factor_in_z.get_text_void().as_deref(),
        );
        meta_data.set_post_reduce_filt_vol_low_pass_radius_sigma(
            self.ltf_low_pass_cutoff_sigma.get_text_void().as_deref(),
        );
        meta_data.set_post_reduce_filt_vol_deconvolution_strength(
            self.ltf_deconvolution_strength.get_text_void().as_deref(),
        );
        meta_data
            .set_post_reduce_filt_vol_snr_falloff(self.ltf_snr_falloff.get_text_void().as_deref());
        meta_data.set_post_reduce_filt_vol_high_pass_nyquist(
            self.ltf_high_pass_filter_cutoff.get_text_void().as_deref(),
        );
        meta_data.set_post_reduce_filt_vol_defocus_in_microns(
            self.ltf_defocus.get_text_void().as_deref(),
        );
        meta_data
            .set_post_reduce_filt_vol_phase_shift(self.ltf_phase_shift.get_text_void().as_deref());
    }

    /// Java package-private `setParameters(ConstMetaData)`.
    pub fn set_parameters_const_meta_data(&self, meta_data: &dyn ConstMetaData) {
        // Backwards compatibility
        self.rb_input_file_trim_vol
            .set_selected_boolean(meta_data.is_post_squeeze_vol_input_trim_vol());
        if !self.rb_input_file_trim_vol.is_selected() {
            self.rb_input_file_flatten_warp.set_selected_boolean(true);
        }
        // Ignore the default values if reducefiltvol hasn't been run yet.
        let reduce_filt_vol_was_run = file_type::CLASS
            .reduce_filt_vol_log
            .exists(Some(self.manager), Some(self.axis_id));
        if meta_data.is_post_reduce_filt_vol_reduction_factor() {
            let reduction_factor =
                meta_data.get_post_reduce_filt_vol_reduction_factor_etomo_number();
            // The source's `if (reduceFiltVolWasRun || !reductionFactor.equals(
            // SqueezevolParam.REDUCTION_FACTOR_DEFAULT)) {}` has an empty body;
            // the condition is still evaluated.
            let _ = reduce_filt_vol_was_run
                || !reduction_factor.equals_number(Some(Number::Float(
                    squeezevol_param::REDUCTION_FACTOR_DEFAULT,
                )));
            self.tf_reduce_by_overall_factor
                .set_text_string(Some(&reduction_factor.to_string()));
        }
        if meta_data.is_post_reduce_filt_vol_z_reduction_factor() {
            let reduction_factor =
                meta_data.get_post_reduce_filt_vol_z_reduction_factor_etomo_number();
            // Empty `if` body in the source; see above.
            let _ = reduce_filt_vol_was_run
                || !reduction_factor.equals_number(Some(Number::Float(
                    squeezevol_param::REDUCTION_FACTOR_DEFAULT,
                )));
            self.ltf_factor_in_z
                .set_text_const_etomo_number(Some(&reduction_factor));
        }
        self.ltf_low_pass_cutoff_sigma.set_text_string(Some(
            &meta_data.get_post_reduce_filt_vol_low_pass_radius_sigma(),
        ));
        self.ltf_deconvolution_strength.set_text_string(Some(
            &meta_data.get_post_reduce_filt_vol_deconvolution_strength(),
        ));
        self.ltf_snr_falloff
            .set_text_string(Some(&meta_data.get_post_reduce_filt_vol_snr_falloff()));
        self.ltf_high_pass_filter_cutoff.set_text_string(Some(
            &meta_data.get_post_reduce_filt_vol_high_pass_nyquist(),
        ));
        self.ltf_defocus.set_text_string(Some(
            &meta_data.get_post_reduce_filt_vol_defocus_in_microns(),
        ));
        self.ltf_phase_shift
            .set_text_string(Some(&meta_data.get_post_reduce_filt_vol_phase_shift()));
    }

    /// Java package-private `setParameters(ReconScreenState)`.
    pub fn set_parameters_recon_screen_state(&self, screen_state: &ReconScreenState) {
        self.btn_reduce_filt_vol.set_button_state(
            screen_state
                .get_button_state(self.btn_reduce_filt_vol.get_button_state_key().as_deref()),
        );
    }

    /// Java package-private `done()`.
    pub fn done(&self) {
        self.btn_reduce_filt_vol
            .remove_action_listener(&self.action_listener);
    }

    /// Java package-private `getComponent()`.
    pub fn get_component(&self) -> Rc<JComponent> {
        self.pnl_root.get_container()
    }

    /// Java public `getParameters(MakecomfileParam, boolean)`.
    pub fn get_parameters_makecomfile_param_boolean(
        &self,
        param: &mut MakecomfileParam,
        do_validation: bool,
    ) -> bool {
        let manager: &'static dyn BaseManager = self.manager;
        // try { ... } catch (FieldValidationFailedException e)
        //   { e.printStackTrace(); }
        let result = (|| -> Result<(), FieldValidationFailedException> {
            if self.rb_input_file_trim_vol.is_selected() {
                param.set_input_file(
                    file_type::CLASS
                        .trim_vol_output
                        .get_file_name(Some(manager), Some(self.axis_id))
                        .as_deref(),
                );
            } else {
                param.set_input_file(
                    file_type::CLASS
                        .flatten_output
                        .get_file_name(Some(manager), Some(self.axis_id))
                        .as_deref(),
                );
            }
            if self.cb_reduce_by_overall_factor.is_selected()
                && !self.tf_reduce_by_overall_factor.is_empty()
            {
                param.set_reduction_factor(
                    self.tf_reduce_by_overall_factor
                        .get_text_boolean(do_validation)?
                        .as_deref(),
                );
            }
            Ok(())
        })();
        if let Err(e) = result {
            eprintln!("{e:?}");
        }

        true
    }

    /// Java private `updateDisplay()`.
    fn update_display(&self) {
        self.tf_reduce_by_overall_factor
            .set_enabled(self.cb_reduce_by_overall_factor.is_selected());
        self.ltf_factor_in_z.set_enabled(
            self.cb_reduce_by_overall_factor.is_selected() && self.is_input_file_flipped(),
        );
        self.ltf_low_pass_cutoff_sigma
            .set_enabled(self.rb_filtering_gaussian.is_selected());
        self.ltf_deconvolution_strength
            .set_enabled(self.rb_filtering_deconvolution.is_selected());
        self.ltf_snr_falloff
            .set_enabled(self.rb_filtering_deconvolution.is_selected());
        self.ltf_high_pass_filter_cutoff
            .set_enabled(self.rb_filtering_deconvolution.is_selected());
        self.ltf_defocus
            .set_enabled(self.rb_filtering_deconvolution.is_selected());
        self.ltf_phase_shift
            .set_enabled(self.rb_filtering_deconvolution.is_selected());
    }

    /// Java private `isInputFileFlipped()`.
    fn is_input_file_flipped(&self) -> bool {
        let mut flipped = true;
        if self.rb_input_file_trim_vol.is_selected() {
            flipped = self.manager.is_trimvol_flipped();
        } else if self.manager.is_result_set_flatten_flipped() {
            if !self.manager.is_flatten_flipped() {
                flipped = false;
            }
        }
        flipped
    }

    /// Java public `getReduceFiltVolDisplay()`: `this`.
    pub fn get_reduce_filt_vol_display(&self) -> Rc<dyn ReduceFiltVolDisplay> {
        self.self_ref
            .upgrade()
            .expect("SqueezeVolPanel used after it was dropped")
    }

    /// Java package-private `getOutputFilename(boolean) throws
    /// FieldValidationFailedException`.  `None` is Java null.
    pub fn get_output_filename(
        &self,
        do_validation: bool,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        let manager: &'static dyn BaseManager = self.manager;
        // FileType.getFileName(BaseManager, AxisID, Number, Number): the Number
        // is passed as Java's `Double.toString` of the value (see NEEDS).
        let mut output_filename = file_type::CLASS
            .reduce_filt_vol_output_file
            .get_file_name_numeric(
                Some(manager),
                Some(self.axis_id),
                Some(&java_lang_double_to_string(
                    reduce_filt_vol_param::DEFAULT_REDUCTION_FACTOR_FOR_OUTPUT_FILE,
                )),
                None,
            );
        let mut db_reduction_factor: Option<f64> = Some(0.0);
        if self.tf_reduce_by_overall_factor.is_enabled()
            && !self.tf_reduce_by_overall_factor.is_empty()
        {
            let str_reduction_factor = self
                .tf_reduce_by_overall_factor
                .get_text_boolean_field_displayer(do_validation, self.this_field_displayer())?;
            db_reduction_factor = converter::to_double(str_reduction_factor.as_deref());
            if let Some(db_reduction_factor) = db_reduction_factor {
                output_filename = file_type::CLASS
                    .reduce_filt_vol_output_file
                    .get_file_name_numeric(
                        Some(manager),
                        Some(self.axis_id),
                        Some(&java_lang_double_to_string(db_reduction_factor)),
                        None,
                    );
            }
        }
        let _ = db_reduction_factor;
        Ok(output_filename)
    }

    /// Java private `setFieldDisplayer()`.
    fn set_field_displayer(&self) {
        let this = self.this_field_displayer();
        self.tf_reduce_by_overall_factor
            .set_overridable_field_displayers(None, this.clone());
        self.ltf_factor_in_z
            .set_overridable_field_displayers(None, this.clone());
        self.ltf_low_pass_cutoff_sigma
            .set_overridable_field_displayers(None, this.clone());
        self.ltf_deconvolution_strength
            .set_overridable_field_displayers(None, this.clone());
        self.ltf_snr_falloff
            .set_overridable_field_displayers(None, this.clone());
        self.ltf_high_pass_filter_cutoff
            .set_overridable_field_displayers(None, this.clone());
        self.ltf_defocus
            .set_overridable_field_displayers(None, this.clone());
        self.ltf_phase_shift
            .set_overridable_field_displayers(None, this.clone());
        self.ltf_data_mode_of_output
            .set_overridable_field_displayers(None, this);
    }
}

impl ContextMenu for SqueezeVolPanel {
    /// Java public `popUpContextMenu(MouseEvent)`.  Right mouse button context
    /// menu.
    fn pop_up_context_menu(&self, mouse_event: &MouseEvent) {
        let man_pagelabel = ["Reducefiltvol".to_string()];
        let man_page = ["reducefiltvol.html".to_string()];

        let log_file_label = ["Reducefiltvol".to_string()];
        let manager: &'static dyn BaseManager = self.manager;
        let log_file = [file_type::CLASS
            .reduce_filt_vol_log
            .get_file_name(Some(manager), Some(self.axis_id))
            .unwrap_or_else(|| "null".to_string())];

        let _context_popup = ContextPopup::new_component_mouse_event_string_string_string_array_string_array_string_array_string_array_base_manager_axis_id(
            &self.pnl_root.get_container(),
            mouse_event,
            Some("Reducefilt"),
            Some(context_popup::TOMO_GUIDE),
            &man_pagelabel,
            &man_page,
            Some(&log_file_label),
            Some(&log_file),
            manager,
            self.axis_id,
        );
    }
}

impl ReduceFiltVolDisplay for SqueezeVolPanel {
    /// Java public `getParameters(ReduceFiltVolParam, boolean) throws
    /// FortranInputSyntaxException`.
    fn get_parameters(
        &self,
        param: &mut ReduceFiltVolParam,
        do_validation: bool,
    ) -> Result<bool, FortranInputSyntaxException> {
        let manager: &'static dyn BaseManager = self.manager;
        // try { ... } catch (final FieldValidationFailedException e)
        //   { return false; }
        let result = (|| -> Result<bool, FieldValidationFailedException> {
            let mut bad_parameter = String::new();
            let is_input_file_flipped = self.is_input_file_flipped();
            if self.rb_input_file_trim_vol.is_selected() {
                param.set_input_file(
                    file_type::CLASS
                        .trim_vol_output
                        .get_file_name(Some(manager), Some(self.axis_id))
                        .as_deref(),
                    is_input_file_flipped,
                );
            } else {
                param.set_input_file(
                    file_type::CLASS
                        .flatten_output
                        .get_file_name(Some(manager), Some(self.axis_id))
                        .as_deref(),
                    is_input_file_flipped,
                );
            }
            if self.cb_reduce_by_overall_factor.is_selected()
                && !self.tf_reduce_by_overall_factor.is_empty()
            {
                param.set_reduction_factor(
                    self.tf_reduce_by_overall_factor
                        .get_text_boolean_field_displayer(
                            do_validation,
                            self.this_field_displayer(),
                        )?
                        .as_deref(),
                );
            } else {
                param.reset_reduction_factor();
            }
            if self.ltf_factor_in_z.is_enabled() && !self.ltf_factor_in_z.is_empty() {
                param.set_z_reduction_factor(
                    self.ltf_factor_in_z
                        .get_text_boolean_field_displayer(
                            do_validation,
                            self.this_field_displayer(),
                        )?
                        .as_deref(),
                );
            } else {
                param.reset_z_reduction_factor();
            }
            if self.ltf_low_pass_cutoff_sigma.is_enabled()
                && !self.ltf_low_pass_cutoff_sigma.is_empty()
            {
                if do_validation {
                    bad_parameter = self
                        .ltf_low_pass_cutoff_sigma
                        .get_quoted_label()
                        .unwrap_or_else(|| "null".to_string());
                }
                let ret_val = param.set_low_pass_radius_sigma(
                    self.ltf_low_pass_cutoff_sigma
                        .get_text_boolean_field_displayer(
                            do_validation,
                            self.this_field_displayer(),
                        )?
                        .as_deref(),
                    do_validation,
                );
                if let Some(ret_val) = ret_val {
                    let message = format!("{bad_parameter} {ret_val}");
                    ui_harness::with(|harness| {
                        harness.open_message_dialog_base_manager_string_string_axis_id(
                            Some(manager),
                            &message,
                            "Syntax Error",
                            Some(self.axis_id),
                        )
                    });
                    return Ok(false);
                }
            } else {
                param.reset_low_pass_radius_sigma();
            }
            if self.ltf_deconvolution_strength.is_enabled() {
                param.set_deconvolution_strength(
                    self.ltf_deconvolution_strength
                        .get_text_boolean_field_displayer(
                            do_validation,
                            self.this_field_displayer(),
                        )?
                        .as_deref(),
                );
            } else {
                param.reset_deconvolution_strength();
            }
            if self.ltf_snr_falloff.is_enabled() {
                param.set_snr_falloff(
                    self.ltf_snr_falloff
                        .get_text_boolean_field_displayer(
                            do_validation,
                            self.this_field_displayer(),
                        )?
                        .as_deref(),
                );
            } else {
                param.reset_snr_falloff();
            }
            if self.ltf_high_pass_filter_cutoff.is_enabled() {
                param.set_high_pass_nyquist(
                    self.ltf_high_pass_filter_cutoff
                        .get_text_boolean_field_displayer(
                            do_validation,
                            self.this_field_displayer(),
                        )?
                        .as_deref(),
                );
            } else {
                param.reset_high_pass_nyquist();
            }
            if self.ltf_defocus.is_enabled() {
                param.set_defocus_in_microns(
                    self.ltf_defocus
                        .get_text_boolean_field_displayer(
                            do_validation,
                            self.this_field_displayer(),
                        )?
                        .as_deref(),
                );
            } else {
                param.reset_defocus_in_microns();
            }
            if self.ltf_phase_shift.is_enabled() {
                param.set_phase_shift(
                    self.ltf_phase_shift
                        .get_text_boolean_field_displayer(
                            do_validation,
                            self.this_field_displayer(),
                        )?
                        .as_deref(),
                );
            } else {
                param.reset_phase_shift();
            }
            param.set_mode_to_output(
                self.ltf_data_mode_of_output
                    .get_text_boolean_field_displayer(do_validation, self.this_field_displayer())?
                    .as_deref(),
            );
            param.set_setup_chunks_if_memory_error(true);
            param.set_output_file(self.get_output_filename(do_validation)?.as_deref());
            Ok(true)
        })();
        Ok(result.unwrap_or(false))
    }
}

impl Run3dmodButtonContainer for SqueezeVolPanel {
    /// Java public `action(String, Deferred3dmodButton, Run3dmodMenuOptions)`.
    fn action(
        &self,
        command: &str,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    ) {
        if Some(command) == self.btn_reduce_filt_vol.get_action_command().as_deref() {
            // The source computes `imageFileType` and does not use it.
            let _image_file_type = if self.rb_input_file_trim_vol.is_selected() {
                ImageFileType::TrimVolOutput
            } else {
                ImageFileType::FlattenOutput
            };
            self.manager.reduce_filt_vol(
                Some(self.btn_reduce_filt_vol.clone() as ProcessResultDisplayHandle),
                None,
                deferred_3dmod_button,
                run_3dmod_menu_options,
                self.dialog_type,
                self,
            );
        } else if Some(command)
            == self
                .btn_imod_reduce_filt_vol
                .get_action_command()
                .as_deref()
        {
            let mut output_file: Option<PathBuf> = None;
            match self.get_output_filename(false) {
                // Upstream bug fixed in translation (SqueezeVolPanel.java:560):
                // `new File(null)` throws NullPointerException when FileType
                // cannot build the output file name; the translation leaves
                // `outputFile` null, so no 3dmod is opened.
                Ok(output_filename) => output_file = output_filename.map(PathBuf::from),
                // catch (FieldValidationFailedException e) { e.printStackTrace(); }
                Err(e) => eprintln!("{e:?}"),
            }
            if let Some(output_file) = &output_file {
                // Java passes a null Run3dmodMenuOptions from the action listener;
                // ImodState.open replaces null with a new Run3dmodMenuOptions()
                // (the default value).
                self.manager.imod_reduced_filtered_volume(
                    run_3dmod_menu_options.unwrap_or_default(),
                    self.axis_id,
                    Some(output_file.as_path()),
                );
            }
        }

        self.update_display();
    }
}

impl FieldDisplayer for SqueezeVolPanel {
    /// Java public `display()`; empty.
    fn display_void(&self) {}

    /// Java public `display(UIComponent)`; empty.
    fn display_ui_component(&self, _ui_component: Option<&dyn UIComponent>) {}
}
