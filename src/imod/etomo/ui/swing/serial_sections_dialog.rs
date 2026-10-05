//! `IMOD/Etomo/src/etomo/ui/swing/SerialSectionsDialog.java`.
//!
//! The Serial Sections dialog: three tabs - "Initial Blend" (preblend: blendmont
//! settings, Make Blended Stack, Fix Edges With Midas), "Align" (the
//! `AutoAlignmentPanel`) and "Make Stack" (xftoxg settings, size, shift, binning,
//! Make Aligned Stack).
//!
//! An event dispatch thread object (`Rc`, `&self` methods).  Java's
//! `FieldValidationFailedException` is the `Err` of the fields' `getText`.

use std::cell::{Cell, RefCell};
use std::rc::{Rc, Weak};
use std::sync::Arc;

use crate::imod::etomo::r#type::process_result_display::ProcessResultDisplay;
use crate::imod::etomo::util::event_queue::EdtRef;

use super::auto_alignment_panel::AutoAlignmentPanel;
use super::beveled_border::BeveledBorder;
use super::button_control_text_efield::ButtonControlTextEfield;
use super::check_box::CheckBox;
use super::check_box_spinner::CheckBoxSpinner;
use super::check_text_field::CheckTextField;
use super::context_menu::ContextMenu;
use super::context_popup::{self, ContextPopup};
use super::deferred_3dmod_button::Deferred3dmodButton;
use super::etched_border::EtchedBorder;
use super::generic_mouse_adapter::GenericMouseAdapter;
use super::labeled_text_field::LabeledTextField;
use super::multi_line_button::MultiLineButton;
use super::radio_button::{RadioButton, RadioButtonModel};
use super::radio_button_interface::EnumeratedTypeRef;
use super::radio_text_field::RadioTextField;
use super::run_3dmod_button::Run3dmodButton;
use super::run_3dmod_button_container::Run3dmodButtonContainer;
use super::select_file_extension::SelectFileExtension;
use super::spinner::Spinner;
use super::tabbed_pane::TabbedPane;
use super::tooltip_formatter;
use super::ui_harness;
use crate::imod::etomo::auto_alignment_controller::AutoAlignmentController;
use crate::imod::etomo::comscript::blendmont_param::{self, BlendmontParam};
use crate::imod::etomo::comscript::const_newst_param::ConstNewstParam;
use crate::imod::etomo::comscript::midas_param::MidasParam;
use crate::imod::etomo::comscript::newst_param::{self, NewstParam, SetSizeToOutputInXandYError};
use crate::imod::etomo::comscript::tomodataplots_param;
use crate::imod::etomo::comscript::xfalign_param::XfalignParam;
use crate::imod::etomo::comscript::xftoxg_param::{self, HybridFits, NumberToFit, XftoxgParam};
use crate::imod::etomo::jdk::{
    ActionEvent, ActionListener, ButtonGroup, ChangeEvent, JComponent, MouseEvent,
};
use crate::imod::etomo::logic::dataset_tool;
use crate::imod::etomo::logic::tomogram_tool::TomogramTool;
use crate::imod::etomo::logic::transforms_tool;
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::serial_sections_manager::SerialSectionsManager;
use crate::imod::etomo::storage::autodoc::autodoc::Autodoc;
use crate::imod::etomo::storage::autodoc::autodoc_factory;
use crate::imod::etomo::storage::autodoc::read_only_autodoc::ReadOnlyAutodoc;
use crate::imod::etomo::storage::log_file::LogFileError;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::java_lang_double_to_string;
use crate::imod::etomo::r#type::const_serial_sections_meta_data::ConstSerialSectionsMetaData;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::etomo_autodoc;
use crate::imod::etomo::r#type::file_key::FileKey;
use crate::imod::etomo::r#type::file_type;
use crate::imod::etomo::r#type::serial_sections_meta_data::SerialSectionsMetaData;
use crate::imod::etomo::r#type::view_type::ViewType;
use crate::imod::etomo::ui::auto_alignment_display::AutoAlignmentDisplay;
use crate::imod::etomo::ui::field::Field;
use crate::imod::etomo::ui::field_displayer::FieldDisplayer;
use crate::imod::etomo::ui::field_type::FieldType;
use crate::imod::etomo::ui::field_validation_failed_exception::FieldValidationFailedException;
use crate::imod::etomo::ui::ui_component::UIComponent;
use crate::imod::etomo::util::mrc_header::MRCHeader;
use crate::imod::etomo::util::utilities;

/// Java `rcsid`.
pub const RCSID: &str = "$Id:$";

/// Java private static final `DIALOG_TYPE`.
const DIALOG_TYPE: DialogType = DialogType::SerialSections;
/// Java private static final `SHIFT_LABEL`.
const SHIFT_LABEL: &str = "Shift in ";
/// Java private static final `SIZE_LABEL`.
const SIZE_LABEL: &str = "Size in ";

/// Java private static final nested class `Tab`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tab {
    /// `INITIAL_BLEND = new Tab(0, "Initial Blend")`.
    InitialBlend,
    /// `ALIGN = new Tab(1, "Align")`.
    Align,
    /// `MAKE_ALIGNED_STACK = new Tab(2, "Make Stack")`.
    MakeAlignedStack,
}

impl Tab {
    /// Java `NUM_TABS`.
    const NUM_TABS: usize = 3;

    /// Java field `index`.
    fn index(self) -> i32 {
        match self {
            Tab::InitialBlend => 0,
            Tab::Align => 1,
            Tab::MakeAlignedStack => 2,
        }
    }

    /// Java field `title`.
    fn title(self) -> &'static str {
        match self {
            Tab::InitialBlend => "Initial Blend",
            Tab::Align => "Align",
            Tab::MakeAlignedStack => "Make Stack",
        }
    }

    /// Java static `getInstance(int)`.
    fn get_instance(index: i32) -> Option<Tab> {
        if index == Tab::InitialBlend.index() {
            return Some(Tab::InitialBlend);
        }
        if index == Tab::Align.index() {
            return Some(Tab::Align);
        }
        if index == Tab::MakeAlignedStack.index() {
            return Some(Tab::MakeAlignedStack);
        }
        None
    }

    /// Java static `getDefaultInstance(ViewType)`.
    fn get_default_instance(view_type: Option<ViewType>) -> Tab {
        if view_type == Some(ViewType::Montage) {
            return Tab::InitialBlend;
        }
        Tab::Align
    }
}

/// Java `toString()`.
impl std::fmt::Display for Tab {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.title())
    }
}

/// Java `public final class SerialSectionsDialog implements ContextMenu,
/// Run3dmodButtonContainer, AutoAlignmentDisplay, FieldDisplayer`.
pub struct SerialSectionsDialog {
    /// Java private final `pnlRoot`.
    pnl_root: Rc<JComponent>,
    /// Java private final `pnlTabArray`.
    pnl_tab_array: Vec<Rc<JComponent>>,
    /// Java private final `pnlTabBodyArray`.
    pnl_tab_body_array: Vec<Rc<JComponent>>,
    /// Java private final `tabPane`.
    tab_pane: Rc<TabbedPane>,
    /// Java private final `cbPreblendVerySloppyMontage`.
    cb_preblend_very_sloppy_montage: Rc<CheckBox>,
    /// Java private final `btnPreblend`.
    btn_preblend: Rc<Run3dmodButton>,
    /// Java private final `btn3dmodPreblend`.
    btn_3dmod_preblend: Rc<Run3dmodButton>,
    /// Java private final `btnFixEdges`.
    btn_fix_edges: Rc<MultiLineButton>,
    /// Java private final `btn3dmodPrealign` (a `Run3dmodButton` declared
    /// `MultiLineButton`).
    btn_3dmod_prealign: Rc<Run3dmodButton>,
    /// Java private final `bgXftoxgAlignment`.
    bg_xftoxg_alignment: Rc<ButtonGroup>,
    /// Java private final `rbNoOptions`.
    rb_no_options: Rc<RadioButton>,
    /// Java private final `rbHybridFitsRotation`.
    rb_hybrid_fits_rotation: Rc<RadioButton>,
    /// Java private final `rbHybridFitsTranslations`.
    rb_hybrid_fits_translations: Rc<RadioButton>,
    /// Java private final `rbHybridFitsTranslationsRotations`.
    rb_hybrid_fits_translations_rotations: Rc<RadioButton>,
    /// Java private final `rbNumberToFitGlobalAlignment`.
    rb_number_to_fit_global_alignment: Rc<RadioButton>,
    /// Java private final `btnAlign`.
    btn_align: Rc<Run3dmodButton>,
    /// Java private final `btn3dmodAlign`.
    btn_3dmod_align: Rc<Run3dmodButton>,
    /// Java private `cbsReferenceSection`.
    cbs_reference_section: Rc<CheckBoxSpinner>,
    /// Java private `ltfSizeX`.
    ltf_size_x: Rc<LabeledTextField>,
    /// Java private `ltfSizeY`.
    ltf_size_y: Rc<LabeledTextField>,
    /// Java private `ltfShiftX`.
    ltf_shift_x: Rc<LabeledTextField>,
    /// Java private `ltfShiftY`.
    ltf_shift_y: Rc<LabeledTextField>,
    /// Java private `spBinByFactor`.
    sp_bin_by_factor: Rc<Spinner>,
    /// Java private `cbFillWithZero`.
    cb_fill_with_zero: Rc<CheckBox>,
    /// Java private `ctfPreblendRobustFitCriterion`.
    ctf_preblend_robust_fit_criterion: Rc<CheckTextField>,
    /// Java private final `spMidasBinning`.
    sp_midas_binning: Rc<Spinner>,
    /// Java private final `btn3dmodRawStack`.
    btn_3dmod_raw_stack: Rc<Run3dmodButton>,
    /// Java private final `cbPreblendReadInXcorrs`.
    cb_preblend_read_in_xcorrs: Rc<CheckBox>,
    /// Java private `bgIntensityCorrection`.
    bg_intensity_correction: Rc<ButtonGroup>,
    /// Java private final `rbNoneIntensityCorrection`.
    rb_none_intensity_correction: Rc<RadioButton>,
    /// Java private final `rbPieceToPieceDifferences`.
    rb_piece_to_piece_differences: Rc<RadioButton>,
    /// Java private final `rbGradientWithinPieces`.
    rb_gradient_within_pieces: Rc<RadioButton>,
    /// Java private `bgInitialGradient`.
    bg_initial_gradient: Rc<ButtonGroup>,
    /// Java private final `rbNoneInitialGradient`.
    rb_none_initial_gradient: Rc<RadioButton>,
    /// Java private final `rbPlanarFit`.
    rb_planar_fit: Rc<RadioButton>,
    /// Java private final `rbGradientFile`.
    rb_gradient_file: Rc<RadioButton>,
    /// Java package-private final `bctfGradientFile`.
    bctf_gradient_file: Rc<ButtonControlTextEfield>,
    /// Java private `cbEMGridMapFilter`.
    cb_e_m_grid_map_filter: Rc<CheckBox>,
    /// Java private `ltfHighFrequencyFilterCutoff`.
    ltf_high_frequency_filter_cutoff: Rc<LabeledTextField>,
    /// Java private `cbWeightForExpectedShifts`.
    cb_weight_for_expected_shifts: Rc<CheckBox>,
    /// Java private final `strDistanceInPixels`.
    str_distance_in_pixels: Rc<JComponent>,
    /// Java private final `bgDistanceInPixels`.
    bg_distance_in_pixels: Rc<ButtonGroup>,
    /// Java private final `rbDefaultDistance`.
    rb_default_distance: Rc<RadioButton>,
    /// Java private final `rtfPixelDistance`.
    rtf_pixel_distance: Rc<RadioTextField>,
    /// Java private final `strPixels`.
    str_pixels: Rc<JComponent>,

    /// Java private final `autoAlignmentPanel`.
    auto_alignment_panel: Rc<AutoAlignmentPanel>,
    /// Java private final `axisID`.
    axis_id: AxisID,
    /// Java private final `manager`.
    manager: &'static SerialSectionsManager,
    /// Java private final `metaData`.
    meta_data: &'static SerialSectionsMetaData,

    /// Java private `curTab`, initially null.
    cur_tab: Cell<Option<Tab>>,
    /// Java private `referenceSectionProblem`, initially false.
    reference_section_problem: Cell<bool>,
    /// `this`, for the listeners.
    this: Weak<SerialSectionsDialog>,
}

impl SerialSectionsDialog {
    /// Java private `SerialSectionsDialog(SerialSectionsManager, AxisID)`.
    fn new(manager: &'static SerialSectionsManager, axis_id: AxisID) -> Rc<SerialSectionsDialog> {
        eprintln!(
            "{}\nDialog: {}",
            utilities::get_date_time_stamp(),
            DIALOG_TYPE
        );
        Rc::new_cyclic(|this: &Weak<SerialSectionsDialog>| {
            let container: Weak<dyn Run3dmodButtonContainer> = this.clone();
            let bg_xftoxg_alignment = ButtonGroup::new();
            let bg_intensity_correction = ButtonGroup::new();
            let bg_initial_gradient = ButtonGroup::new();
            let bg_distance_in_pixels = ButtonGroup::new();
            let btn_3dmod_prealign =
                Run3dmodButton::get_3dmod_instance_string_run_3dmod_button_container(
                    Some("Open Stack"),
                    Some(container.clone()),
                );
            let gradient_file_extension = SelectFileExtension::new();
            gradient_file_extension.set_file_chooser_title(Some("Select gradient file"));
            let bctf_gradient_file =
                ButtonControlTextEfield::get_file_instance_string_select_file_extension_boolean_boolean(
                    Some("Gradient file"),
                    Some(gradient_file_extension),
                    false,
                    false,
                );
            SerialSectionsDialog {
                pnl_root: JComponent::new_panel(),
                pnl_tab_array: (0..Tab::NUM_TABS)
                    .map(|_| JComponent::new_panel())
                    .collect(),
                pnl_tab_body_array: (0..Tab::NUM_TABS)
                    .map(|_| JComponent::new_panel())
                    .collect(),
                tab_pane: TabbedPane::new(),
                cb_preblend_very_sloppy_montage: CheckBox::new_string(Some(
                    "Treat as very sloppy montage",
                )),
                btn_preblend:
                    Run3dmodButton::get_deferred_3dmod_instance_string_run_3dmod_button_container(
                        Some("Make Blended Stack"),
                        Some(container.clone()),
                    ),
                btn_3dmod_preblend:
                    Run3dmodButton::get_3dmod_instance_string_run_3dmod_button_container_file_key(
                        Some("Open Blended Stack"),
                        Some(container.clone()),
                        Some(FileKey::clone(&file_type::CLASS.preblend_output_mrc)),
                    ),
                btn_fix_edges: MultiLineButton::new_string(Some("Fix Edges With Midas")),
                btn_3dmod_prealign,
                rb_no_options: RadioButton::new_string_button_group(
                    Some("Local fitting (retain trends)"),
                    Some(&bg_xftoxg_alignment),
                ),
                rb_hybrid_fits_rotation: RadioButton::new_string_enumerated_type_button_group(
                    Some("Remove trends in rotation"),
                    Some(EnumeratedTypeRef::new(HybridFits::ROTATION)),
                    Some(&bg_xftoxg_alignment),
                ),
                rb_hybrid_fits_translations: RadioButton::new_string_enumerated_type_button_group(
                    Some("Remove trends in translation"),
                    Some(EnumeratedTypeRef::new(HybridFits::TRANSLATIONS)),
                    Some(&bg_xftoxg_alignment),
                ),
                rb_hybrid_fits_translations_rotations:
                    RadioButton::new_string_enumerated_type_button_group(
                        Some("Remove trends in translation & rotation"),
                        Some(EnumeratedTypeRef::new(HybridFits::TRANSLATIONS_ROTATIONS)),
                        Some(&bg_xftoxg_alignment),
                    ),
                rb_number_to_fit_global_alignment:
                    RadioButton::new_string_enumerated_type_button_group(
                        Some("Global alignments (remove all trends)"),
                        Some(EnumeratedTypeRef::new(NumberToFit::GLOBAL_ALIGNMENT)),
                        Some(&bg_xftoxg_alignment),
                    ),
                bg_xftoxg_alignment,
                btn_align:
                    Run3dmodButton::get_deferred_3dmod_instance_string_run_3dmod_button_container(
                        Some("Make Aligned Stack"),
                        Some(container.clone()),
                    ),
                btn_3dmod_align:
                    Run3dmodButton::get_3dmod_instance_string_run_3dmod_button_container_file_key(
                        Some("Open Aligned Stack"),
                        Some(container.clone()),
                        Some(FileKey::clone(&file_type::CLASS.aligned_stack_mrc)),
                    ),
                cbs_reference_section: CheckBoxSpinner::get_instance_string(Some(
                    "Reference section for global alignment: ",
                )),
                ltf_size_x: LabeledTextField::new_field_type_string(
                    FieldType::Integer,
                    Some(&format!("{}X: ", SIZE_LABEL)),
                ),
                ltf_size_y: LabeledTextField::new_field_type_string(
                    FieldType::Integer,
                    Some("Y: "),
                ),
                ltf_shift_x: LabeledTextField::new_field_type_string(
                    FieldType::FloatingPoint,
                    Some(&format!("{}X: ", SHIFT_LABEL)),
                ),
                ltf_shift_y: LabeledTextField::new_field_type_string(
                    FieldType::FloatingPoint,
                    Some("Y: "),
                ),
                sp_bin_by_factor: Spinner::get_labeled_instance_string_int_int_int(
                    Some("Binning: "),
                    1,
                    1,
                    8,
                ),
                cb_fill_with_zero: CheckBox::new_string(Some("Fill empty areas with 0")),
                ctf_preblend_robust_fit_criterion: CheckTextField::get_instance(
                    FieldType::FloatingPoint,
                    "Robust fitting with criterion: ",
                ),
                sp_midas_binning: Spinner::get_labeled_instance_string_int_int_int(
                    Some("Binning: "),
                    1,
                    1,
                    8,
                ),
                btn_3dmod_raw_stack:
                    Run3dmodButton::get_3dmod_instance_string_run_3dmod_button_container(
                        Some("Open Raw Stack"),
                        Some(container),
                    ),
                cb_preblend_read_in_xcorrs: CheckBox::new_string(Some(
                    "Use existing edge displacement file (.ecd)",
                )),
                rb_none_intensity_correction: RadioButton::new_string_button_group(
                    Some("None"),
                    Some(&bg_intensity_correction),
                ),
                rb_piece_to_piece_differences: RadioButton::new_string_button_group(
                    Some("Piece-to-piece differences only"),
                    Some(&bg_intensity_correction),
                ),
                rb_gradient_within_pieces: RadioButton::new_string_button_group(
                    Some("Gradient within pieces also"),
                    Some(&bg_intensity_correction),
                ),
                bg_intensity_correction,
                rb_none_initial_gradient: RadioButton::new_string_button_group(
                    Some("None"),
                    Some(&bg_initial_gradient),
                ),
                rb_planar_fit: RadioButton::new_string_button_group(
                    Some("Planar fit to sum of pieces in this input file"),
                    Some(&bg_initial_gradient),
                ),
                rb_gradient_file: RadioButton::new_string_button_group(
                    Some("Gradient file from other summed images:"),
                    Some(&bg_initial_gradient),
                ),
                bg_initial_gradient,
                bctf_gradient_file,
                cb_e_m_grid_map_filter: CheckBox::new_string(Some(
                    "Treat as low-magnification EM grid map and use peak weighting",
                )),
                ltf_high_frequency_filter_cutoff: LabeledTextField::new_field_type_string(
                    FieldType::FloatingPoint,
                    Some("High-frequency filter cutoff in 1/microns "),
                ),
                cb_weight_for_expected_shifts: CheckBox::new_string(Some(
                    "Just weight correlation peaks by deviations from expected shift",
                )),
                str_distance_in_pixels: JComponent::new_label(
                    "Distance at which weighting falls to 0.5: ",
                ),
                rb_default_distance: RadioButton::new_string_button_group(
                    Some("default"),
                    Some(&bg_distance_in_pixels),
                ),
                rtf_pixel_distance: RadioTextField::get_instance_with_alternate_label(
                    FieldType::FloatingPoint,
                    Some(""),
                    Some(&bg_distance_in_pixels),
                    Some("Distance at which weighting falls to 0.5"),
                ),
                bg_distance_in_pixels,
                str_pixels: JComponent::new_label(" pixels"),
                auto_alignment_panel: AutoAlignmentPanel::get_serial_sections_instance(manager),
                axis_id,
                manager,
                meta_data: manager.get_meta_data(),
                cur_tab: Cell::new(None),
                reference_section_problem: Cell::new(false),
                this: this.clone(),
            }
        })
    }

    /// Java static `getInstance(SerialSectionsManager, AxisID)`.
    pub fn get_instance(
        manager: &'static SerialSectionsManager,
        axis_id: AxisID,
    ) -> Rc<SerialSectionsDialog> {
        let instance = SerialSectionsDialog::new(manager, axis_id);
        instance.create_panel();
        instance.set_tooltips();
        instance.add_listeners();
        instance
    }

    /// Java `setAutoAlignmentController(AutoAlignmentController)`.
    pub fn set_auto_alignment_controller(
        &self,
        auto_alignment_controller: &'static AutoAlignmentController,
    ) {
        self.auto_alignment_panel
            .set_controller(auto_alignment_controller);
    }

    /// Java private `createPanel()`.
    fn create_panel(&self) {
        // panels
        let pnl_initial_blend_buttons = JComponent::new_panel();
        let pnl_preblend_very_sloppy_blend = JComponent::new_panel();
        let pnl_outer_multiple_correlation = JComponent::new_panel();
        let pnl_multiple_correlation = JComponent::new_panel();
        let pnl_preblend_weight_for_expected_shifts = JComponent::new_panel();
        let pnl_preblend_weight_for_expected_shifts2 = JComponent::new_panel();
        let pnl_preblend_e_m_grid_map_filter = JComponent::new_panel();
        let pnl_preblend_high_frequency_filter_cutoff = JComponent::new_panel();
        let pnl_open_aligned_stack = JComponent::new_panel();
        let pnl_xftoxg_alignment = JComponent::new_panel();
        let pnl_xftoxg_alignment_x = JComponent::new_panel();
        let pnl_size = JComponent::new_panel();
        let pnl_shift = JComponent::new_panel();
        let pnl_make_stack_a = JComponent::new_panel();
        let pnl_make_stack_buttons = JComponent::new_panel();
        let pnl_preblend_robust_fit_criterion = JComponent::new_panel();
        let pnl_midas_binning = JComponent::new_panel();
        let pnl_midas = JComponent::new_panel();
        let pnl_midas_x = JComponent::new_panel();
        let pnl_fix_edges = JComponent::new_panel();
        let pnl_preblend_read_in_xcorrs = JComponent::new_panel();
        let pnl_outer_intensity_correction = JComponent::new_panel();
        let pnl_intensity_correction = JComponent::new_panel();
        let pnl_none_intensity_correction = JComponent::new_panel();
        let pnl_piece_to_piece_differences = JComponent::new_panel();
        let pnl_gradient_within_pieces = JComponent::new_panel();
        let pnl_outer_initial_gradient_correction = JComponent::new_panel();
        let pnl_initial_gradient_correction = JComponent::new_panel();
        let pnl_none_initial_gradient_correction = JComponent::new_panel();
        let pnl_planar_fit = JComponent::new_panel();
        let pnl_gradient_file = JComponent::new_panel();
        let pnl_gradient_file_chooser = JComponent::new_panel();
        // init
        for i in 0..Tab::NUM_TABS {
            // pnlTabArray[i] = new JPanel(); pnlTabBodyArray[i] = new JPanel(); (built
            // with the struct)
            self.tab_pane.add_tab_string_component(
                &Tab::get_instance(i as i32)
                    .map(|tab| tab.to_string())
                    .unwrap_or_default(),
                &self.pnl_tab_array[i],
            );
        }
        self.tab_pane.get_component().set_enabled_at(
            Tab::InitialBlend.index() as usize,
            self.meta_data.get_view_type() == Some(ViewType::Montage),
        );
        self.btn_preblend
            .set_deferred_3dmod_button_deferred_3dmod_button(Some(
                self.btn_3dmod_preblend.clone() as Rc<dyn Deferred3dmodButton>
            ));
        // Set the maximum reference section.
        let property_user_dir =
            crate::imod::etomo::base_manager::BaseManager::get_property_user_dir(self.manager);
        let stack = utilities::java_io_file_new(
            property_user_dir.as_deref().unwrap_or("null"),
            &self.meta_data.get_stack(),
        );
        let stack_name = utilities::java_io_file_get_name(&stack);
        let header = MRCHeader::get_instance_in_dir(
            property_user_dir.as_deref(),
            Some(&stack_name),
            Some(self.axis_id),
        );
        // `MRCHeader.getInstance` never answers null in the Java; a missing instance
        // reads as a failed read.
        let read: Result<i32, crate::imod::etomo::util::mrc_header::ReadError> = match &header {
            None => Err(crate::imod::etomo::util::mrc_header::ReadError::Io(
                "null".to_string(),
            )),
            Some(header) => {
                let result = header.borrow_mut().read_with_manager(self.manager);
                result.map(|_| header.borrow().get_n_sections())
            }
        };
        match read {
            Ok(n_sections) => self.cbs_reference_section.set_max(n_sections),
            Err(e) => {
                eprintln!("{e}");
                self.reference_section_problem.set(true);
                self.cbs_reference_section.set_check_box_enabled(false);
                let manager = self.manager;
                ui_harness::with(|harness| {
                    harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(manager),
                        &format!("Unable to read{}\n{}", stack_name, e),
                        "File Read Error",
                        Some(self.axis_id),
                    )
                });
            }
        }
        self.btn_align
            .set_deferred_3dmod_button_deferred_3dmod_button(Some(
                self.btn_3dmod_align.clone() as Rc<dyn Deferred3dmodButton>
            ));
        self.cb_preblend_read_in_xcorrs.set_selected_boolean(
            file_type::CLASS
                .piece_shifts
                .exists(Some(self.manager), Some(self.axis_id)),
        );
        self.rb_none_intensity_correction.set_selected_boolean(true);
        self.rb_none_initial_gradient.set_selected_boolean(true);
        self.bctf_gradient_file.set_preferred_width(250);
        self.bctf_gradient_file.set_editable(true);
        self.bctf_gradient_file
            .set_enabled(self.rb_gradient_file.is_selected());
        self.cb_weight_for_expected_shifts
            .set_selected_boolean(false);
        self.rb_default_distance.set_selected_boolean(true);
        self.cb_e_m_grid_map_filter.set_selected_boolean(false);
        self.ltf_high_frequency_filter_cutoff.set_text_double(0.25);
        // root panel
        // Swing layout: BoxLayout Y_AXIS.
        self.pnl_root.set_border_title(
            BeveledBorder::new(Some("Serial Sections"))
                .get_border()
                .get_title()
                .as_deref(),
        );
        self.pnl_root.add(&self.tab_pane.get_component());
        // initial blend
        // Swing layout: every panel below is a BoxLayout; rigid areas and glue are
        // layout only.
        let index = Tab::InitialBlend.index() as usize;
        let body = &self.pnl_tab_body_array[index];
        body.add(&pnl_preblend_read_in_xcorrs);
        body.add(&pnl_preblend_very_sloppy_blend);
        body.add(&pnl_preblend_robust_fit_criterion);
        body.add(&pnl_outer_multiple_correlation);
        body.add(&pnl_outer_intensity_correction);
        body.add(&pnl_outer_initial_gradient_correction);
        body.add(&pnl_initial_blend_buttons);
        body.add(&pnl_midas_x);
        // PreblendReadInXcorrs
        pnl_preblend_read_in_xcorrs.add(&self.cb_preblend_read_in_xcorrs.get_component());
        // preblend very sloppy blend
        pnl_preblend_very_sloppy_blend.add(&self.cb_preblend_very_sloppy_montage.get_component());
        // PreblendRobustFitCriterion
        pnl_preblend_robust_fit_criterion
            .add(&self.ctf_preblend_robust_fit_criterion.get_root_component());
        // Analysis of multiple correlation peaks
        pnl_outer_multiple_correlation.add(&pnl_multiple_correlation);
        pnl_multiple_correlation.set_border_title(
            EtchedBorder::new(Some("Analysis of Multiple Correlation Peaks"))
                .get_title()
                .as_deref(),
        );
        pnl_multiple_correlation.add(&pnl_preblend_e_m_grid_map_filter);
        pnl_multiple_correlation.add(&pnl_preblend_high_frequency_filter_cutoff);
        pnl_multiple_correlation.add(&pnl_preblend_weight_for_expected_shifts);
        pnl_multiple_correlation.add(&pnl_preblend_weight_for_expected_shifts2);
        // preblend EMGripMapFilter
        pnl_preblend_e_m_grid_map_filter.add(&self.cb_e_m_grid_map_filter.get_component());
        // preblend high frequency filter cutoff
        pnl_preblend_high_frequency_filter_cutoff
            .add(&self.ltf_high_frequency_filter_cutoff.get_component());
        // preblend weight for expected shifts
        pnl_preblend_weight_for_expected_shifts
            .add(&self.cb_weight_for_expected_shifts.get_component());
        // preblend weight for expected shifts line 2
        pnl_preblend_weight_for_expected_shifts2.add(&self.str_distance_in_pixels);
        pnl_preblend_weight_for_expected_shifts2.add(&self.rb_default_distance.get_component());
        pnl_preblend_weight_for_expected_shifts2.add(&self.rtf_pixel_distance.get_container());
        pnl_preblend_weight_for_expected_shifts2.add(&self.str_pixels);
        // Intensity correction from differences in edges
        pnl_outer_intensity_correction.add(&pnl_intensity_correction);
        pnl_intensity_correction.set_border_title(
            EtchedBorder::new(Some("Intensity Correction from Differences in Edges"))
                .get_title()
                .as_deref(),
        );
        pnl_intensity_correction.add(&pnl_none_intensity_correction);
        pnl_intensity_correction.add(&pnl_piece_to_piece_differences);
        pnl_intensity_correction.add(&pnl_gradient_within_pieces);
        pnl_none_intensity_correction.add(&self.rb_none_intensity_correction.get_component());
        pnl_piece_to_piece_differences.add(&self.rb_piece_to_piece_differences.get_component());
        pnl_gradient_within_pieces.add(&self.rb_gradient_within_pieces.get_component());
        // Initial gradient correction from summed images
        pnl_outer_initial_gradient_correction.add(&pnl_initial_gradient_correction);
        pnl_initial_gradient_correction.set_border_title(
            EtchedBorder::new(Some("Initial Gradient Correction from Summed Images"))
                .get_title()
                .as_deref(),
        );
        pnl_initial_gradient_correction.add(&pnl_none_initial_gradient_correction);
        pnl_initial_gradient_correction.add(&pnl_planar_fit);
        pnl_initial_gradient_correction.add(&pnl_gradient_file);
        pnl_initial_gradient_correction.add(&pnl_gradient_file_chooser);
        pnl_none_initial_gradient_correction.add(&self.rb_none_initial_gradient.get_component());
        pnl_planar_fit.add(&self.rb_planar_fit.get_component());
        pnl_gradient_file.add(&self.rb_gradient_file.get_component());
        pnl_gradient_file_chooser.add(&self.bctf_gradient_file.get_component());
        // initial blend buttons
        pnl_initial_blend_buttons.add(&self.btn_3dmod_raw_stack.get_component());
        pnl_initial_blend_buttons.add(&self.btn_preblend.get_component());
        pnl_initial_blend_buttons.add(&self.btn_3dmod_preblend.get_component());
        // midas X direction
        pnl_midas_x.add(&pnl_midas);
        // midas
        pnl_midas.set_border_title(
            EtchedBorder::new(Some("Fix Shifts Between Pieces"))
                .get_title()
                .as_deref(),
        );
        pnl_midas.add(&pnl_midas_binning);
        pnl_midas.add(&pnl_fix_edges);
        pnl_midas.add(&JComponent::new_label(
            "Replaces the edge displacement file (.ecd).",
        ));
        // midas binning
        pnl_midas_binning.add(&self.sp_midas_binning.get_container());
        // FixEdges
        pnl_fix_edges.add(&self.btn_fix_edges.get_component());
        // align
        let index = Tab::Align.index() as usize;
        let body = &self.pnl_tab_body_array[index];
        body.add(&pnl_open_aligned_stack);
        body.add(&self.auto_alignment_panel.get_root_component());
        // open aligned stack
        pnl_open_aligned_stack.add(&self.btn_3dmod_prealign.get_component());
        // make stack
        let index = Tab::MakeAlignedStack.index() as usize;
        let body = &self.pnl_tab_body_array[index];
        body.add(&pnl_xftoxg_alignment_x);
        body.add(&self.cbs_reference_section.get_container());
        body.add(&pnl_size);
        body.add(&pnl_shift);
        body.add(&pnl_make_stack_a);
        body.add(&pnl_make_stack_buttons);
        // xftoxg alignment X panel
        pnl_xftoxg_alignment_x.add(&pnl_xftoxg_alignment);
        // xftoxg alignment
        pnl_xftoxg_alignment.set_border_title(
            EtchedBorder::new(Some("Stack alignment transforms"))
                .get_title()
                .as_deref(),
        );
        pnl_xftoxg_alignment.add(&self.rb_no_options.get_component());
        pnl_xftoxg_alignment.add(&self.rb_hybrid_fits_rotation.get_component());
        pnl_xftoxg_alignment.add(&self.rb_hybrid_fits_translations.get_component());
        pnl_xftoxg_alignment.add(&self.rb_hybrid_fits_translations_rotations.get_component());
        pnl_xftoxg_alignment.add(&self.rb_number_to_fit_global_alignment.get_component());
        let manager = self.manager;
        ui_harness::with(|harness| {
            harness.pack_axis_id_base_manager(Some(self.axis_id), Some(manager))
        });
        // size
        pnl_size.add(&self.ltf_size_x.get_container());
        pnl_size.add(&self.ltf_size_y.get_container());
        // shift
        pnl_shift.add(&self.ltf_shift_x.get_container());
        pnl_shift.add(&self.ltf_shift_y.get_container());
        // make stack subpanel 1
        pnl_make_stack_a.add(&self.sp_bin_by_factor.get_container());
        pnl_make_stack_a.add(&self.cb_fill_with_zero.get_component());
        // make stack buttons
        pnl_make_stack_buttons.add(&self.btn_align.get_component());
        pnl_make_stack_buttons.add(&self.btn_3dmod_align.get_component());
    }

    /// Java `getRootContainer()`.
    pub fn get_root_container(&self) -> Rc<JComponent> {
        self.pnl_root.clone()
    }

    /// Java private `addListeners()`.
    fn add_listeners(&self) {
        let tab_pane = self.tab_pane.get_component();
        tab_pane.add_mouse_listener(GenericMouseAdapter::new(
            self.this.clone() as Weak<dyn ContextMenu>
        ));
        // TabChangeListener
        let this = self.this.clone();
        tab_pane.add_change_listener(Rc::new(move |_event: &ChangeEvent| {
            if let Some(dialog) = this.upgrade() {
                dialog.change_tab();
            }
        }));
        // SerialSectionsActionListener: dialog.action(event.getActionCommand(), null,
        // null)
        let this = self.this.clone();
        let listener: ActionListener = Rc::new(move |event: &ActionEvent| {
            if let Some(dialog) = this.upgrade() {
                Run3dmodButtonContainer::action(
                    &*dialog,
                    event.get_action_command().unwrap_or(""),
                    None,
                    None,
                );
            }
        });
        self.btn_preblend.add_action_listener(listener.clone());
        self.btn_3dmod_preblend
            .add_action_listener(listener.clone());
        self.btn_fix_edges.add_action_listener(listener.clone());
        self.btn_3dmod_prealign
            .add_action_listener(listener.clone());
        self.btn_align.add_action_listener(listener.clone());
        self.btn_3dmod_align.add_action_listener(listener.clone());
        self.btn_3dmod_raw_stack
            .add_action_listener(listener.clone());
        self.rb_no_options.add_action_listener(listener.clone());
        self.rb_hybrid_fits_rotation
            .add_action_listener(listener.clone());
        self.rb_hybrid_fits_translations
            .add_action_listener(listener.clone());
        self.rb_hybrid_fits_translations_rotations
            .add_action_listener(listener.clone());
        self.rb_number_to_fit_global_alignment
            .add_action_listener(listener.clone());
        self.cbs_reference_section
            .add_check_box_action_listener(Some(listener.clone()));
        self.cb_preblend_read_in_xcorrs
            .add_action_listener(Some(listener.clone()));
        self.rb_none_intensity_correction
            .add_action_listener(listener.clone());
        self.rb_piece_to_piece_differences
            .add_action_listener(listener.clone());
        self.rb_gradient_within_pieces
            .add_action_listener(listener.clone());
        self.rb_none_initial_gradient
            .add_action_listener(listener.clone());
        self.rb_planar_fit.add_action_listener(listener.clone());
        self.rb_gradient_file.add_action_listener(listener.clone());
        self.cb_weight_for_expected_shifts
            .add_action_listener(Some(listener.clone()));
        self.rb_default_distance
            .add_action_listener(listener.clone());
        self.rtf_pixel_distance
            .add_action_listener(listener.clone());
        self.cb_e_m_grid_map_filter
            .add_action_listener(Some(listener.clone()));
        self.ltf_high_frequency_filter_cutoff
            .add_action_listener(listener);
    }

    /// Java `getParameters(SerialSectionsMetaData, boolean)`.
    pub fn get_parameters_meta_data(
        &self,
        meta_data: &SerialSectionsMetaData,
        do_validation: bool,
    ) -> bool {
        let result = (|| -> Result<bool, FieldValidationFailedException> {
            meta_data.set_robust_fit_criterion(
                self.ctf_preblend_robust_fit_criterion
                    .get_text_boolean(do_validation)?
                    .as_deref(),
            );
            meta_data.set_midas_binning(Some(self.sp_midas_binning.get_value()));
            if !self.auto_alignment_panel.get_parameters_meta_data(
                &mut meta_data.get_auto_alignment_meta_data().lock().unwrap(),
                do_validation,
            ) {
                return Ok(false);
            }
            meta_data.set_number_to_fit_global_alignment(
                self.rb_number_to_fit_global_alignment.is_selected(),
            );
            meta_data.set_use_reference_section(
                self.cbs_reference_section.is_selected()
                    && self.cbs_reference_section.is_check_box_enabled(),
            );
            meta_data.set_reference_section(Some(self.cbs_reference_section.get_value()));
            meta_data.set_size_x(self.ltf_size_x.get_text_boolean(do_validation)?.as_deref());
            meta_data.set_size_y(self.ltf_size_y.get_text_boolean(do_validation)?.as_deref());
            meta_data.set_shift_x(self.ltf_shift_x.get_text_boolean(do_validation)?.as_deref());
            meta_data.set_shift_y(self.ltf_shift_y.get_text_boolean(do_validation)?.as_deref());
            meta_data.set_preblend_very_sloppy_montage(
                self.cb_preblend_very_sloppy_montage.is_selected(),
            );
            meta_data.set_preblend_weight_for_expected_shifts_boolean(
                self.cb_weight_for_expected_shifts.is_selected(),
            );
            meta_data.set_preblend_weight_for_expected_shifts_string(
                self.rtf_pixel_distance
                    .get_text_boolean(do_validation)?
                    .as_deref(),
            );
            meta_data.set_preblend_e_m_grid_map_filter(self.cb_e_m_grid_map_filter.is_selected());
            meta_data.set_preblend_high_frequency_filter_cutoff(
                self.ltf_high_frequency_filter_cutoff
                    .get_text_boolean(do_validation)?
                    .as_deref(),
            );
            // Upstream bug fixed in translation (SerialSectionsDialog.java:508): Java
            // dereferences a null curTab (no tab shown yet); the tab is not saved then.
            if let Some(cur_tab) = self.cur_tab.get() {
                meta_data.set_tab(cur_tab.index());
            }
            meta_data.set_other_sum_gradient_file(
                self.bctf_gradient_file
                    .get_text_boolean_field_displayer(
                        do_validation,
                        Some(self as &dyn FieldDisplayer),
                    )?
                    .as_deref(),
            );
            Ok(true)
        })();
        result.unwrap_or(false)
    }

    /// Java `setParameters(ConstSerialSectionsMetaData)`.
    pub fn set_parameters_meta_data(&self, meta_data: &dyn ConstSerialSectionsMetaData) {
        self.ctf_preblend_robust_fit_criterion
            .set_text_string(Some(&meta_data.get_robust_fit_criterion()));
        self.sp_midas_binning
            .set_value_const_etomo_number(&meta_data.get_midas_binning());
        self.auto_alignment_panel
            .set_parameters(&meta_data.get_auto_alignment_meta_data().lock().unwrap());
        if !meta_data.is_null_no_options() {
            self.rb_no_options
                .set_selected_boolean(meta_data.is_no_options());
        }
        if !meta_data.is_null_hybrid_fits_translations() {
            self.rb_hybrid_fits_translations
                .set_selected_boolean(meta_data.is_hybrid_fits_translations());
        }
        if !meta_data.is_null_hybrid_fits_translations_rotations() {
            self.rb_hybrid_fits_translations_rotations
                .set_selected_boolean(meta_data.is_hybrid_fits_translations_rotations());
        }
        self.rb_number_to_fit_global_alignment
            .set_selected_boolean(meta_data.is_number_to_fit_global_alignment());
        self.cbs_reference_section
            .set_selected(meta_data.is_use_reference_section());
        self.cbs_reference_section
            .set_value_const_etomo_number(&meta_data.get_reference_section());
        self.ltf_size_x
            .set_text_string(Some(&meta_data.get_size_x()));
        self.ltf_size_y
            .set_text_string(Some(&meta_data.get_size_y()));
        self.ltf_shift_x
            .set_text_string(Some(&meta_data.get_shift_x()));
        self.ltf_shift_y
            .set_text_string(Some(&meta_data.get_shift_y()));
        self.cb_preblend_very_sloppy_montage
            .set_selected_boolean(meta_data.is_preblend_very_sloppy_montage());
        self.cb_weight_for_expected_shifts
            .set_selected_boolean(meta_data.is_bool_preblend_weight_for_expected_shifts());
        if meta_data.is_str_preblend_weight_for_expected_shifts() {
            self.rtf_pixel_distance.set_text_string(Some(
                &meta_data.get_str_preblend_weight_for_expected_shifts(),
            ));
        }
        self.cb_e_m_grid_map_filter
            .set_selected_boolean(meta_data.is_preblend_e_m_grid_map_filter());
        if meta_data.is_preblend_high_frequency_filter_cutoff() {
            self.ltf_high_frequency_filter_cutoff
                .set_text_string(Some(&meta_data.get_preblend_high_frequency_filter_cutoff()));
        }
        self.bctf_gradient_file
            .set_text_string(Some(&meta_data.get_other_sum_gradient_file()));
        let mut tab = Tab::get_default_instance(meta_data.get_view_type()).index();
        if !meta_data.is_tab_empty() {
            tab = meta_data.get_tab();
        }
        self.change_tab_index(tab);
        self.update_display();
    }

    /// Java `isPreblendRobustFitting()`.
    pub fn is_preblend_robust_fitting(&self) -> bool {
        self.ctf_preblend_robust_fit_criterion.is_selected()
    }

    /// Java `getPreblendRobustFitting()`.
    pub fn get_preblend_robust_fitting(&self) -> Option<String> {
        self.ctf_preblend_robust_fit_criterion.get_text_void()
    }

    /// Java `isFixIntensityFromEdges()`.
    pub fn is_fix_intensity_from_edges(&self) -> bool {
        self.rb_none_intensity_correction.is_selected()
            || self.rb_piece_to_piece_differences.is_selected()
            || self.rb_gradient_within_pieces.is_selected()
    }

    /// Java `getFixIntensityFromEdges()`.
    pub fn get_fix_intensity_from_edges(&self) -> Option<i32> {
        if self.rb_piece_to_piece_differences.is_selected() {
            return Some(blendmont_param::PIECE_TO_PIECE_DIFFERENCES_ONLY);
        }
        if self.rb_gradient_within_pieces.is_selected() {
            return Some(blendmont_param::GRADIENT_WITHIN_PIECES_ALSO);
        }
        None
    }

    /// Java `isSumPiecesForGradient()`.
    pub fn is_sum_pieces_for_gradient(&self) -> bool {
        self.rb_none_initial_gradient.is_selected() || self.rb_planar_fit.is_selected()
    }

    /// Java `getSumPiecesForGradient()`.
    pub fn get_sum_pieces_for_gradient(&self) -> Option<i32> {
        if self.rb_planar_fit.is_selected() {
            return Some(blendmont_param::SUM_PIECES_FOR_GRADIENT);
        }
        None
    }

    /// Java `isOtherSumGradientFile()`.
    pub fn is_other_sum_gradient_file(&self) -> bool {
        self.rb_gradient_file.is_selected()
    }

    /// Java `getOtherSumGradientFile()`.
    pub fn get_other_sum_gradient_file(&self) -> Option<String> {
        self.bctf_gradient_file.get_text_void()
    }

    /// Java `getParameters(MidasParam)`.
    pub fn get_parameters_midas(&self, param: &mut MidasParam) {
        param.set_binning(Some(self.sp_midas_binning.get_value()));
    }

    /// Java `setPreblendReadInXcorrs(boolean)`.
    pub fn set_preblend_read_in_xcorrs(&self, read_in_xcorrs: bool) {
        self.cb_preblend_read_in_xcorrs
            .set_selected_boolean(read_in_xcorrs);
        self.update_display();
    }

    /// Java `setPreblendParameters(BlendmontParam)`.
    pub fn set_preblend_parameters(&self, param: &BlendmontParam) {
        self.cb_preblend_very_sloppy_montage
            .set_selected_boolean(param.is_very_sloppy_montage());
        self.ctf_preblend_robust_fit_criterion
            .set_selected_boolean(param.is_robust_fit_criterion());
        if self.ctf_preblend_robust_fit_criterion.is_selected() {
            self.ctf_preblend_robust_fit_criterion
                .set_text_string(Some(&param.get_robust_fit_criterion()));
        }
        self.cb_e_m_grid_map_filter
            .set_selected_boolean(param.is_e_m_grid_map_filter());
        if self.cb_e_m_grid_map_filter.is_selected() {
            self.ltf_high_frequency_filter_cutoff
                .set_text_string(Some(&param.get_e_m_grid_map_filter()));
        }
        let is_weight_for_expected_shifts = param.is_weight_for_expected_shifts();
        self.cb_weight_for_expected_shifts.set_selected_boolean(
            is_weight_for_expected_shifts && !self.cb_e_m_grid_map_filter.is_selected(),
        );
        if is_weight_for_expected_shifts {
            // Double.parseDouble: the com file value is a number (it was parsed into a
            // DOUBLE ScriptParameter); an unparsable one would throw
            // NumberFormatException, read as NaN here (never the default).
            let distance_in_pixels = param
                .get_weight_for_expected_shifts()
                .trim()
                .parse::<f64>()
                .unwrap_or(f64::NAN);
            if blendmont_param::WEIGHT_FOR_EXPECTED_SHIFTS_DEFAULT == distance_in_pixels {
                self.rb_default_distance.set_selected_boolean(true);
            } else {
                self.rtf_pixel_distance.set_selected_boolean(true);
                self.rtf_pixel_distance
                    .set_text_string(Some(&param.get_weight_for_expected_shifts()));
            }
        }
        if self.rb_gradient_file.is_selected() {
            self.bctf_gradient_file
                .set_text_string(Some(&param.get_other_sum_gradient_file()));
        }
        // ReadInXcorrs:
        // When the dialog is opened, set ReadInXcorrs if the .ecd file exists. This function
        // is only called when the dialog is opened so ignore the .com version of
        // ReadInXcorrs. It doesn't make sense (though its probably harmless) to set
        // ReadInXcorrs when there is no .ecd file.
    }

    /// Java `getPreblendParameters(BlendmontParam, boolean)`.
    pub fn get_preblend_parameters(&self, param: &mut BlendmontParam, do_validation: bool) -> bool {
        let result = (|| -> Result<(), FieldValidationFailedException> {
            param.set_very_sloppy_montage(
                self.cb_preblend_very_sloppy_montage.is_enabled()
                    && self.cb_preblend_very_sloppy_montage.is_selected(),
            );
            if self.ctf_preblend_robust_fit_criterion.is_selected() {
                param.set_robust_fit_criterion(
                    self.ctf_preblend_robust_fit_criterion
                        .get_text_boolean(do_validation)?
                        .as_deref(),
                );
            } else {
                param.reset_robust_fit_criterion();
            }
            param.set_read_in_xcorrs(self.cb_preblend_read_in_xcorrs.is_selected());
            if !self.cb_preblend_read_in_xcorrs.is_selected() {
                if self.cb_e_m_grid_map_filter.is_selected() {
                    param.set_e_m_grid_map_filter(
                        self.ltf_high_frequency_filter_cutoff
                            .get_text_boolean(do_validation)?
                            .as_deref(),
                    );
                } else {
                    param.reset_e_m_grid_map_filter();
                }
                if self.cb_e_m_grid_map_filter.is_selected()
                    || self.cb_weight_for_expected_shifts.is_selected()
                {
                    if self.rb_default_distance.is_selected() {
                        param.set_weight_for_expected_shifts(Some(&java_lang_double_to_string(
                            blendmont_param::WEIGHT_FOR_EXPECTED_SHIFTS_DEFAULT,
                        )));
                    } else {
                        param.set_weight_for_expected_shifts(
                            self.rtf_pixel_distance
                                .get_text_boolean(do_validation)?
                                .as_deref(),
                        );
                    }
                } else {
                    param.reset_weight_for_expected_shifts();
                }
            } else {
                param.reset_e_m_grid_map_filter();
                param.reset_weight_for_expected_shifts();
            }
            param.set_fix_intensity_from_edges(self.get_fix_intensity_from_edges());
            param.set_sum_pieces_for_gradient(self.get_sum_pieces_for_gradient());
            if self.rb_gradient_file.is_selected() {
                param.set_other_sum_gradient_file(
                    self.bctf_gradient_file.get_text_void().as_deref(),
                );
            } else {
                param.reset_other_sum_gradient_file();
            }
            Ok(())
        })();
        result.is_ok()
    }

    /// Java `getBlendParameters(BlendmontParam, boolean)`.
    pub fn get_blend_parameters(&self, param: &mut BlendmontParam, do_validation: bool) -> bool {
        let result = (|| -> Result<(), FieldValidationFailedException> {
            if Field::is_empty(&*self.ltf_size_x)
                && Field::is_empty(&*self.ltf_size_y)
                && Field::is_empty(&*self.ltf_shift_x)
                && Field::is_empty(&*self.ltf_shift_y)
            {
                param.reset_starting_and_ending_xand_y();
            } else {
                let size_x = self.ltf_size_x.get_text_boolean(do_validation)?;
                let shift_x = self.ltf_shift_x.get_text_boolean(do_validation)?;
                let size_y = self.ltf_size_y.get_text_boolean(do_validation)?;
                let shift_y = self.ltf_shift_y.get_text_boolean(do_validation)?;
                let pair = TomogramTool::get_starting_and_ending_x_and_y(
                    &file_type::CLASS.preblend_output_mrc,
                    size_x.as_deref(),
                    shift_x.as_deref(),
                    size_y.as_deref(),
                    shift_y.as_deref(),
                    self.manager,
                    self.axis_id,
                    Field::get_quoted_label(&*self.ltf_size_x).as_deref(),
                    Field::get_quoted_label(&*self.ltf_shift_x).as_deref(),
                    utilities::quote_label(Some(&format!(
                        "{}{}",
                        SIZE_LABEL,
                        self.ltf_size_y.get_label()
                    )))
                    .as_deref(),
                    utilities::quote_label(Some(&format!(
                        "{}{}",
                        SHIFT_LABEL,
                        self.ltf_shift_y.get_label()
                    )))
                    .as_deref(),
                    Some("Entry Error"),
                );
                param.set_starting_and_ending_xand_y(pair.as_ref());
            }
            param.set_bin_by_factor(Some(self.sp_bin_by_factor.get_value()));
            if self.cb_fill_with_zero.is_selected() {
                param.set_fill_value(0);
            }
            Ok(())
        })();
        result.is_ok()
    }

    /// Java `setBlendParameters(BlendmontParam)`.
    pub fn set_blend_parameters(&self, param: &BlendmontParam) {
        self.sp_bin_by_factor
            .set_value_const_etomo_number(param.get_bin_by_factor());
        self.cb_fill_with_zero
            .set_selected_boolean(param.fill_value_equals(0));
    }

    /// Java `getParameters(NewstParam, boolean) throws FortranInputSyntaxException,
    /// InvalidParameterException, IOException`.  `Ok(false)` is a field that failed
    /// validation; the `Err` is the declared exceptions.
    pub fn get_parameters_newst(
        &self,
        param: &mut NewstParam,
        do_validation: bool,
    ) -> Result<bool, SetSizeToOutputInXandYError> {
        let result = (|| -> Result<Result<(), SetSizeToOutputInXandYError>, FieldValidationFailedException> {
            let size_x = self.ltf_size_x.get_text_boolean(do_validation)?;
            let size_y = self.ltf_size_y.get_text_boolean(do_validation)?;
            if let Err(e) = param.set_size_to_output_in_xand_y_xy(
                size_x.as_deref().unwrap_or("null"),
                size_y.as_deref().unwrap_or("null"),
                self.sp_bin_by_factor.get_value().int_value(),
                0.0,
                Some("Size"),
            ) {
                return Ok(Err(e));
            }
            let shift_x = self.ltf_shift_x.get_text_boolean(do_validation)?;
            let shift_y = self.ltf_shift_y.get_text_boolean(do_validation)?;
            param.set_offsets_in_xand_y(Some(&TomogramTool::convert_shifts_to_offsets(
                shift_x.as_deref(),
                shift_y.as_deref(),
                true,
            )));
            param.set_bin_by_factor(Some(self.sp_bin_by_factor.get_value()));
            if self.cb_fill_with_zero.is_selected() {
                param.set_fill_value(0);
            }
            Ok(Ok(()))
        })();
        match result {
            Ok(Ok(())) => Ok(true),
            Ok(Err(e)) => Err(e),
            Err(_) => Ok(false),
        }
    }

    /// Java `setParameters(ConstNewstParam)`.
    pub fn set_parameters_newst(&self, param: &dyn ConstNewstParam) {
        self.ltf_shift_x.set_text_string(Some(
            &TomogramTool::convert_offset_to_shift(Some(&param.get_offset_in_x())).to_string(),
        ));
        self.ltf_shift_y.set_text_string(Some(
            &TomogramTool::convert_offset_to_shift(Some(&param.get_offset_in_y())).to_string(),
        ));
        self.sp_bin_by_factor
            .set_value_int(param.get_bin_by_factor());
        self.cb_fill_with_zero
            .set_selected_boolean(param.fill_value_equals(0));
    }

    /// The `EnumeratedType` of the selected `bgXftoxgAlignment` button
    /// (`((RadioButton.RadioButtonModel) bgXftoxgAlignment.getSelection())
    /// .getEnumeratedType()`).
    fn selected_xftoxg_alignment(&self) -> Option<EnumeratedTypeRef> {
        self.bg_xftoxg_alignment
            .get_selection()
            .and_then(|button| button.get_model())
            .and_then(|model| {
                model
                    .as_any()
                    .downcast_ref::<RadioButtonModel>()
                    .and_then(|model| {
                        crate::imod::etomo::ui::swing::abstract_radio_button_model::AbstractRadioButtonModel::get_enumerated_type(model)
                    })
            })
    }

    /// Java `getParameters(XftoxgParam)`.
    pub fn get_parameters_xftoxg(&self, param: &mut XftoxgParam) {
        if self.rb_no_options.is_selected() {
            param.reset_hybrid_fits();
            param.reset_number_to_fit();
        } else if self.rb_hybrid_fits_translations.is_selected()
            || self.rb_hybrid_fits_translations_rotations.is_selected()
            || self.rb_hybrid_fits_rotation.is_selected()
        {
            if let Some(hybrid_fits) = self
                .selected_xftoxg_alignment()
                .as_ref()
                .and_then(|selected| selected.downcast_ref::<HybridFits>().copied())
            {
                param.set_hybrid_fits(hybrid_fits);
            }
            param.reset_number_to_fit();
        } else if self.rb_number_to_fit_global_alignment.is_selected() {
            param.reset_hybrid_fits();
            if let Some(number_to_fit) = self
                .selected_xftoxg_alignment()
                .as_ref()
                .and_then(|selected| selected.downcast_ref::<NumberToFit>().copied())
            {
                param.set_number_to_fit(number_to_fit);
            }
        }
        if self.cbs_reference_section.is_selected()
            && self.cbs_reference_section.is_check_box_enabled()
        {
            param.set_reference_section_number(Some(self.cbs_reference_section.get_value()));
        } else {
            param.reset_reference_section();
        }
    }

    /// Java `setParameters(XftoxgParam)`.
    pub fn set_parameters_xftoxg(&self, param: &XftoxgParam) {
        let hybrid_fits_empty = param.is_hybrid_fits_empty();
        let number_to_fit_empty = param.is_number_to_fit_empty();
        if hybrid_fits_empty && number_to_fit_empty {
            self.rb_no_options.set_selected_boolean(true);
        } else if !hybrid_fits_empty {
            let hybrid_fits = param.get_hybrid_fits();
            if HybridFits::TRANSLATIONS.equals_int(hybrid_fits) {
                self.rb_hybrid_fits_translations.set_selected_boolean(true);
            } else if HybridFits::TRANSLATIONS_ROTATIONS.equals_int(hybrid_fits) {
                self.rb_hybrid_fits_translations_rotations
                    .set_selected_boolean(true);
            } else if HybridFits::ROTATION.equals_int(hybrid_fits) {
                self.rb_hybrid_fits_rotation.set_selected_boolean(true);
            }
        } else if !number_to_fit_empty
            && NumberToFit::GLOBAL_ALIGNMENT.equals_int(param.get_number_to_fit())
        {
            self.rb_number_to_fit_global_alignment
                .set_selected_boolean(true);
        }
        self.cbs_reference_section
            .set_selected(!param.is_reference_section_empty());
        if self.cbs_reference_section.is_selected() {
            self.cbs_reference_section
                .set_value_int(param.get_reference_section());
        }
        self.update_display();
    }

    /// Java private `updateDisplay()`.
    fn update_display(&self) {
        let read_in_xcorrs = self.cb_preblend_read_in_xcorrs.is_selected();
        self.cbs_reference_section.set_check_box_enabled(
            self.rb_number_to_fit_global_alignment.is_selected()
                && !self.reference_section_problem.get(),
        );
        self.cb_preblend_very_sloppy_montage
            .set_enabled(!read_in_xcorrs);
        self.bctf_gradient_file
            .set_enabled(self.rb_gradient_file.is_selected());
        self.cb_e_m_grid_map_filter.set_enabled(!read_in_xcorrs);
        self.ltf_high_frequency_filter_cutoff
            .set_enabled(!read_in_xcorrs && self.cb_e_m_grid_map_filter.is_selected());
        self.cb_weight_for_expected_shifts
            .set_enabled(!read_in_xcorrs && !self.cb_e_m_grid_map_filter.is_selected());
        let weighting = !read_in_xcorrs
            && (self.cb_weight_for_expected_shifts.is_selected()
                || self.cb_e_m_grid_map_filter.is_selected());
        self.str_distance_in_pixels.set_enabled(weighting);
        self.rb_default_distance.set_enabled(weighting);
        self.rtf_pixel_distance.set_enabled(weighting);
        self.str_pixels.set_enabled(
            self.rtf_pixel_distance.is_enabled() && self.rtf_pixel_distance.is_selected(),
        );
    }

    /// Java private `changeTab(int)`.
    fn change_tab_index(&self, new_tab_index: i32) {
        self.tab_pane
            .get_component()
            .set_selected_tab(new_tab_index);
        self.change_tab();
    }

    /// Java private `changeTab()`.
    fn change_tab(&self) {
        let new_tab = Tab::get_instance(self.tab_pane.get_component().get_selected_tab());
        if new_tab == self.cur_tab.get() {
            return;
        }
        if let Some(cur_tab) = self.cur_tab.get() {
            self.tab_change_warning();
            let index = cur_tab.index() as usize;
            self.pnl_tab_array[index].remove(&self.pnl_tab_body_array[index]);
        }
        self.cur_tab.set(new_tab);
        // Upstream bug fixed in translation (SerialSectionsDialog.java:735): a tab
        // index with no Tab (a stored tab out of range) makes Java dereference null;
        // no tab body is shown then.
        if let Some(cur_tab) = new_tab {
            let index = cur_tab.index() as usize;
            self.pnl_tab_array[index].add(&self.pnl_tab_body_array[index]);
        }
        let manager = self.manager;
        ui_harness::with(|harness| {
            harness.pack_axis_id_base_manager(Some(self.axis_id), Some(manager))
        });
        ui_harness::with(|harness| harness.move_sub_frame());
    }

    /// Java private `tabChangeWarning()`.
    fn tab_change_warning(&self) {
        if self.cur_tab.get() == Some(Tab::InitialBlend) {
            transforms_tool::check_up_to_date_edge_functions_file(
                self.manager.get_state().get_invalid_edge_functions().is(),
                self.manager,
                self.axis_id,
                utilities::quote_label(Some(Tab::InitialBlend.title())).as_deref(),
                self.btn_preblend.get_quoted_label().as_deref(),
            );
        }
    }

    /// Java private `setTooltips()`.
    fn set_tooltips(&self) {
        let view_type = self.meta_data.get_view_type();
        // Swing: tabPane.setToolTipTextAt(...) for the three tabs; tab tooltips are
        // the Slint side's.
        let _ = tooltip_formatter::INSTANCE.format(Some(
            "Blends the overlapping edges of the montage images without transformations.  \
             Runs blendmont.",
        ));
        self.btn_preblend.set_tool_tip_text(Some(
            "Use Blendmont to blend each section into a single image.",
        ));
        let text = "Opens the blended stack.";
        self.btn_3dmod_preblend
            .set_tool_tip_text(Some("Open file with single blended images in 3dmod"));
        let text2 = "Opens the raw stack.";
        self.btn_3dmod_raw_stack
            .set_tool_tip_text(Some("Open raw montage file in 3dmod."));
        self.btn_3dmod_prealign
            .set_tool_tip_text(Some(if view_type == Some(ViewType::Montage) {
                text
            } else {
                text2
            }));
        self.sp_midas_binning
            .set_tool_tip_text(Some("Binning to apply when reading into Midas"));
        self.btn_fix_edges.set_tool_tip_text(Some(
            "Run Midas to adjust shifts between pairs of overlapping pieces.",
        ));
        let text = "Applies transformations to the serial sections stack.";
        self.btn_align.set_tool_tip_text(Some(&format!(
            "{}  Runs xftoxg and {}.",
            text,
            if view_type == Some(ViewType::Montage) {
                "blendmont"
            } else {
                "newst"
            }
        )));
        self.rb_no_options.set_tool_tip_text_string(Some(&format!(
            "Align images to nearby ones but retain all progressive trends in position and \
             size (no {} or {} options).",
            xftoxg_param::HYBRID_FITS_KEY,
            xftoxg_param::NUMBER_TO_FIT_KEY
        )));
        self.rb_hybrid_fits_rotation.set_tool_tip_text_string(Some(
            "Eliminate all rotations between images but retain other trends",
        ));
        self.rb_hybrid_fits_translations
            .set_tool_tip_text_string(Some(&format!(
                "Eliminate all shifts between images but retain other trends in the images ({} \
             option {})",
                xftoxg_param::HYBRID_FITS_KEY,
                HybridFits::TRANSLATIONS
            )));
        self.rb_hybrid_fits_translations_rotations
            .set_tool_tip_text_string(Some(&format!(
                "Eliminate shifts and rotations between images, retain trends in size changes \
                 ({} option {})",
                xftoxg_param::HYBRID_FITS_KEY,
                HybridFits::TRANSLATIONS_ROTATIONS
            )));
        self.rb_number_to_fit_global_alignment
            .set_tool_tip_text_string(Some(&format!(
                "Eliminate all trends with a global alignment ({} option {})",
                xftoxg_param::NUMBER_TO_FIT_KEY,
                NumberToFit::GLOBAL_ALIGNMENT
            )));
        self.cbs_reference_section
            .set_tool_tip_text(Some("Reference section number"));
        self.ltf_size_x
            .set_tool_tip_text(Some("Size in X of the aligned stack (unbinned pixels)"));
        self.ltf_size_y
            .set_tool_tip_text(Some("Size in Y of the aligned stack (unbinned pixels)"));
        self.ltf_shift_x.set_tool_tip_text(Some(
            "Amount to shift output images in X, in unbinned pixels (positive shifts to right)",
        ));
        self.ltf_shift_y.set_tool_tip_text(Some(
            "Amount to shift output images in Y, in unbinned pixels (positive shifts up)",
        ));
        self.btn_3dmod_align
            .set_tool_tip_text(Some("Opens the aligned serial sections stack."));
        self.rb_none_intensity_correction
            .set_tool_tip_text_string(Some(
                "No analysis of intensity differences in overlap zones between pieces.",
            ));
        self.rb_piece_to_piece_differences
            .set_tool_tip_text_string(Some(
                "Analyze intensity differences in overlap zones between pieces to determine \
             scaling for each piece that minimizes differences.",
            ));
        self.rb_gradient_within_pieces
            .set_tool_tip_text_string(Some(
                "Correct for intensity gradient within pieces as well as piece-to-piece \
             intensity changes from differences in overlap zones.  Good for small montages, \
             not large ones with progressive intensity changes.  Requires > 1 piece in each \
             direction.",
            ));
        self.rb_none_initial_gradient.set_tool_tip_text_string(Some(
            "No correction for gradient within pieces based on image sums.",
        ));
        self.rb_planar_fit.set_tool_tip_text_string(Some(
            "Fit a plane to sum of all pieces in this montage file and correct each piece for \
             this gradient when it is read in.",
        ));
        self.rb_gradient_file.set_tool_tip_text_string(Some(
            "Use output file from fitting a plane to some other sum of images to correct each \
             piece for a gradient when it is read in.",
        ));
        self.cb_weight_for_expected_shifts
            .set_tool_tip_text_string(Some(
                "Analyze multiple peaks when correlating overlap zones and weight correlation \
             coefficients by the deviations from expected peak positions.",
            ));
        let distance = "Distance from expected peak at which Gaussian weighting function falls \
                        to 0.5, in pixels.  Enter 1 for the default, the width of the correlated \
                        area.";
        self.str_distance_in_pixels
            .set_tool_tip_text(Some(distance));
        self.rb_default_distance
            .set_tool_tip_text_string(Some(distance));
        self.rtf_pixel_distance.set_tool_tip_text(Some(distance));
        self.cb_e_m_grid_map_filter.set_tool_tip_text_string(Some(
            "Set up filters and other parameters for finding piece shifts in a grid map where \
             there may be strong signals from regularly spaced holes within grid squares.",
        ));
        self.ltf_high_frequency_filter_cutoff.set_tool_tip_text(Some(
            "Start of high-frequency filter when treating as a grid map, in reciprocal microns.",
        ));
        let autodoc_name = if view_type == Some(ViewType::Montage) {
            autodoc_factory::BLENDMONT
        } else {
            autodoc_factory::NEWSTACK
        };
        let mut autodoc: *const dyn ReadOnlyAutodoc = std::ptr::null::<Autodoc>();
        let result = (|| -> Result<(), LogFileError> {
            autodoc = unsafe {
                autodoc_factory::get_instance(
                    Some(self.manager),
                    Some(autodoc_name),
                    self.axis_id,
                    false,
                )
            }? as *const Autodoc;
            Ok(())
        })();
        match result {
            Ok(()) => {}
            Err(LogFileError::Lock(_)) => {}
            Err(except) => eprintln!("{}", except),
        }
        // SAFETY: the pointer is null or an autodoc the factory keeps for the life of
        // the process.
        let autodoc: Option<&dyn ReadOnlyAutodoc> = if autodoc.is_null() {
            None
        } else {
            Some(unsafe { &*autodoc })
        };
        if view_type == Some(ViewType::Montage) {
            self.cb_preblend_very_sloppy_montage
                .set_tool_tip_text_string(Some(
                    "Overlaps between images vary substantially; use for montages taken with stage \
                 movement.",
                ));
            self.ctf_preblend_robust_fit_criterion
                .set_check_box_tool_tip_text(Some(
                    "Discount or ignore aberrant shifts between images when solving for overall \
                 shifts",
                ));
            self.ctf_preblend_robust_fit_criterion
                .set_field_tool_tip_text(Some(
                    "Criterion for determining whether a shift is an outlier; lower ignores more \
                 shifts.",
                ));
            self.sp_bin_by_factor.set_tool_tip_text(
                etomo_autodoc::get_tooltip(autodoc, Some(blendmont_param::BIN_BY_FACTOR_KEY))
                    .as_deref(),
            );
            self.cb_fill_with_zero
                .set_tool_tip_text_string(Some(&format!(
                    "Sets {} to zero.  {}",
                    blendmont_param::FILL_VALUE_KEY,
                    etomo_autodoc::get_tooltip(autodoc, Some(blendmont_param::FILL_VALUE_KEY))
                        .unwrap_or_else(|| "null".to_string())
                )));
        } else {
            self.sp_bin_by_factor.set_tool_tip_text(
                etomo_autodoc::get_tooltip(autodoc, Some(newst_param::BIN_BY_FACTOR_KEY))
                    .as_deref(),
            );
            self.cb_fill_with_zero
                .set_tool_tip_text_string(Some(&format!(
                    "Sets {} to zero.  {}",
                    newst_param::FILL_VALUE_KEY,
                    etomo_autodoc::get_tooltip(autodoc, Some(newst_param::FILL_VALUE_KEY))
                        .unwrap_or_else(|| "null".to_string())
                )));
        }
    }
}

impl Run3dmodButtonContainer for SerialSectionsDialog {
    /// Java `action(String, Deferred3dmodButton, Run3dmodMenuOptions)`.
    fn action(
        &self,
        command: &str,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    ) {
        let is = |action_command: Option<String>| action_command.as_deref() == Some(command);
        if is(self.btn_preblend.get_action_command()) {
            self.manager.preblend(
                None,
                Some(Arc::new(EdtRef::new(
                    self.btn_preblend.clone() as Rc<dyn ProcessResultDisplay>
                ))),
                deferred_3dmod_button,
                self.axis_id,
                run_3dmod_menu_options,
                Some(DialogType::SerialSections),
            );
        } else if is(self.btn_fix_edges.get_action_command()) {
            self.manager.midas_fix_edges(self.axis_id, None);
        } else if is(self.btn_align.get_action_command()) {
            self.manager.align(
                self.axis_id,
                Some(Arc::new(EdtRef::new(
                    self.btn_align.clone() as Rc<dyn ProcessResultDisplay>
                ))),
                deferred_3dmod_button,
                run_3dmod_menu_options,
            );
        } else if is(self.btn_3dmod_raw_stack.get_action_command()) {
            self.manager.imod_raw(self.axis_id, run_3dmod_menu_options);
        } else if is(self.btn_3dmod_preblend.get_action_command()) {
            self.manager
                .imod_preblend(self.axis_id, run_3dmod_menu_options);
        } else if is(self.btn_3dmod_prealign.get_action_command()) {
            self.manager
                .imod_prealign(self.axis_id, run_3dmod_menu_options);
        } else if is(self.btn_3dmod_align.get_action_command()) {
            self.manager
                .imod_align(self.axis_id, run_3dmod_menu_options);
        }
        self.update_display();
    }
}

impl AutoAlignmentDisplay for SerialSectionsDialog {
    /// Java `msgProcessEnded()`.  The purpose of this function is to have Midas
    /// button enabled whether or not Initial Auto-Alignment succeeds.
    fn msg_process_ended(&self) {
        self.auto_alignment_panel.msg_process_change(true);
    }

    /// Java `getDialogType()`.
    fn get_dialog_type(&self) -> DialogType {
        DialogType::SerialSections
    }

    /// Java `getAxisID()`.
    fn get_axis_id(&self) -> AxisID {
        self.axis_id
    }

    /// Java `getAutoAlignmentParameters(MidasParam)`.
    fn get_auto_alignment_parameters_midas(&self, param: &mut MidasParam) {
        self.manager
            .get_auto_alignment_parameters_midas(param, self.axis_id);
        self.auto_alignment_panel.get_parameters_midas_param(param);
    }

    /// Java `getAutoAlignmentParameters(XfalignParam, boolean)`.
    fn get_auto_alignment_parameters_xfalign(
        &self,
        param: &mut XfalignParam,
        do_validation: bool,
    ) -> bool {
        self.manager
            .get_auto_alignment_parameters_xfalign(param, self.axis_id);
        self.auto_alignment_panel
            .get_parameters_xfalign_param(param, do_validation)
    }
}

impl ContextMenu for SerialSectionsDialog {
    /// Java `popUpContextMenu(MouseEvent)`.  Right mouse button context menu.
    fn pop_up_context_menu(&self, mouse_event: &MouseEvent) {
        let mut log_file_label: Vec<String> = Vec::new();
        let mut log_file: Vec<String> = Vec::new();
        let mut man_page_label: Vec<String> = Vec::new();
        let mut man_page: Vec<String> = Vec::new();
        let mut anchor: Option<&str> = None;
        let mut graph: Option<Vec<tomodataplots_param::Task>> = None;
        let strings = |values: &[&str]| {
            values
                .iter()
                .map(|value| value.to_string())
                .collect::<Vec<_>>()
        };
        match self.cur_tab.get() {
            Some(Tab::InitialBlend) => {
                anchor = Some("Blending");
                log_file_label = strings(&["Preblend"]);
                log_file = strings(&["preblend.log"]);
                man_page_label = strings(&["Blendmont", "Midas", "3dmod"]);
                man_page = strings(&["blendmont.html", "midas.html", "3dmod.html"]);
                // setup graph
                if let Some(stack) = self.manager.get_stack()
                    && !dataset_tool::is_one_by(
                        crate::imod::etomo::base_manager::BaseManager::get_property_user_dir(
                            self.manager,
                        )
                        .as_deref(),
                        Some(&utilities::java_io_file_get_name(&stack.to_string_lossy())),
                        self.manager,
                        self.axis_id,
                    )
                {
                    graph = Some(vec![tomodataplots_param::Task::SerialSectionsMeanMax]);
                }
            }
            Some(Tab::Align) => {
                anchor = Some("Aligning");
                log_file_label = strings(&["Xfalign"]);
                log_file = strings(&["xfalign.log"]);
                man_page_label = strings(&["Xfalign", "Midas", "3dmod"]);
                man_page = strings(&["xfalign.html", "midas.html", "3dmod.html"]);
            }
            Some(Tab::MakeAlignedStack) => {
                anchor = Some("Images");
                if self.meta_data.get_view_type() == Some(ViewType::Montage) {
                    log_file_label = strings(&["Blend"]);
                    log_file = strings(&["blend.log"]);
                    man_page_label = strings(&["Blendmont", "Xftoxg", "3dmod"]);
                    man_page = strings(&["blendmont.html", "xftoxg.html", "3dmod.html"]);
                } else {
                    log_file_label = strings(&["Newst"]);
                    log_file = strings(&["newst.log"]);
                    man_page_label = strings(&["Colornewst", "Xftoxg", "3dmod"]);
                    man_page = strings(&["colornewst.html", "xftoxg.html", "3dmod.html"]);
                }
            }
            None => {}
        }
        let _context_popup = ContextPopup::new_component_mouse_event_string_string_string_array_string_array_string_array_string_array_task_array_base_manager_axis_id_boolean(
            &self.pnl_root,
            mouse_event,
            anchor,
            Some(context_popup::SERIAL_GUIDE),
            &man_page_label,
            &man_page,
            &log_file_label,
            &log_file,
            graph.as_deref(),
            self.manager,
            self.axis_id,
            true,
        );
    }
}

impl FieldDisplayer for SerialSectionsDialog {
    /// Java `display()`: empty.
    fn display_void(&self) {}

    /// Java `display(UIComponent)`: empty.
    fn display_ui_component(&self, _ui_component: Option<&dyn UIComponent>) {}
}
