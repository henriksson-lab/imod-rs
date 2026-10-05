//! `IMOD/Etomo/src/etomo/ui/swing/AutoAlignmentPanel.java`.
//!
//! The auto-alignment parameters and buttons (Initial Auto Alignment, Midas, Refine
//! with Auto Alignment, the two reverts) of the Join dialog's Align tab and of the
//! Serial Sections dialog.  An event dispatch thread object, created as `Rc<Self>`;
//! the listener class `AutoAlignmentActionListener` is a closure holding a weak
//! reference to the panel.

use std::cell::RefCell;
use std::rc::{Rc, Weak};

use super::check_box::CheckBox;
use super::deferred_3dmod_button::Deferred3dmodButton;
use super::etched_border::EtchedBorder;
use super::labeled_spinner::LabeledSpinner;
use super::labeled_text_field::LabeledTextField;
use super::multi_line_button::MultiLineButton;
use super::run_3dmod_button::Run3dmodButton;
use super::run_3dmod_button_container::Run3dmodButtonContainer;
use super::spaced_panel::{self, SpacedPanel};
use super::spinner::Spinner;
use super::transform_chooser_panel::TransformChooserPanel;
use crate::imod::etomo::auto_alignment_controller::AutoAlignmentController;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::midas_param::MidasParam;
use crate::imod::etomo::comscript::xfalign_param::{self, XfalignParam};
use crate::imod::etomo::jdk::{ActionEvent, ActionListener, JComponent};
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::storage::autodoc::autodoc::Autodoc;
use crate::imod::etomo::storage::autodoc::autodoc_factory;
use crate::imod::etomo::storage::autodoc::read_only_autodoc::ReadOnlyAutodoc;
use crate::imod::etomo::storage::log_file::LogFileError;
use crate::imod::etomo::r#type::auto_alignment_meta_data::AutoAlignmentMetaData;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::etomo_autodoc;
use crate::imod::etomo::ui::field::Field;
use crate::imod::etomo::ui::field_type::FieldType;
use crate::imod::etomo::ui::field_validation_failed_exception::FieldValidationFailedException;
use crate::imod::etomo::util::shared_constants;

/// Java package-private `final class AutoAlignmentPanel implements
/// Run3dmodButtonContainer`.
pub struct AutoAlignmentPanel {
    /// Java private final `pnlRoot = SpacedPanel.getFocusableInstance()`.
    pnl_root: Rc<SpacedPanel>,
    /// Java private final `pnlParameters = SpacedPanel.getInstance()`.
    pnl_parameters: Rc<SpacedPanel>,
    /// Java private final `ltfSigmaLowFrequency`.
    ltf_sigma_low_frequency: Rc<LabeledTextField>,
    /// Java private final `ltfCutoffHighFrequency`.
    ltf_cutoff_high_frequency: Rc<LabeledTextField>,
    /// Java private final `ltfSigmaHighFrequency`.
    ltf_sigma_high_frequency: Rc<LabeledTextField>,
    /// Java private final `pnlButtons = SpacedPanel.getInstance()`.
    pnl_buttons: Rc<SpacedPanel>,
    /// Java private final `btnInitialAutoAlignment`.
    btn_initial_auto_alignment: Rc<MultiLineButton>,
    /// Java private final `btnMidas`.
    btn_midas: Rc<MultiLineButton>,
    /// Java private final `btnRefineAutoAlignment`.
    btn_refine_auto_alignment: Rc<MultiLineButton>,
    /// Java private final `btnRevertToMidas`.
    btn_revert_to_midas: Rc<MultiLineButton>,
    /// Java private final `btnRevertToEmpty`.
    btn_revert_to_empty: Rc<MultiLineButton>,
    /// Java private final `spReduceByBinning`.
    sp_reduce_by_binning: Rc<LabeledSpinner>,
    /// Java private final `ltfSkipSectionsFrom1`.
    ltf_skip_sections_from1: Rc<LabeledTextField>,
    /// Java private final `cbPreCrossCorrelation`.
    cb_pre_cross_correlation: Rc<CheckBox>,
    /// Java private final `ltfEdgeToIgnore`.
    ltf_edge_to_ignore: Rc<LabeledTextField>,
    /// Java private final `spMidasBinning`.
    sp_midas_binning: Rc<Spinner>,

    /// Java private final `manager`.
    manager: &'static dyn BaseManager,
    /// Java private final `joinInterface`.
    join_interface: bool,
    /// Java private final `tcAlign`.
    tc_align: Rc<TransformChooserPanel>,
    /// Java private final `cbFindWarping` (Serial Sections only).
    cb_find_warping: Option<Rc<CheckBox>>,
    /// Java private final `ltfWarpPatchSizeX` (Serial Sections only).
    ltf_warp_patch_size_x: Option<Rc<LabeledTextField>>,
    /// Java private final `ltfWarpPatchSizeY` (Serial Sections only).
    ltf_warp_patch_size_y: Option<Rc<LabeledTextField>>,
    /// Java private final `cbBoundaryModel` (Serial Sections only).
    cb_boundary_model: Option<Rc<CheckBox>>,
    /// Java private final `btnBoundaryModel` (Serial Sections only).
    btn_boundary_model: Option<Rc<Run3dmodButton>>,
    /// Java private final `ltfShiftLimitsForWarpX` (Serial Sections only).
    ltf_shift_limits_for_warp_x: Option<Rc<LabeledTextField>>,
    /// Java private final `ltfShiftLimitsForWarpY` (Serial Sections only).
    ltf_shift_limits_for_warp_y: Option<Rc<LabeledTextField>>,
    /// Java private final `cbSobelFilter` (Serial Sections only).
    cb_sobel_filter: Option<Rc<CheckBox>>,

    /// Java private `controller`, initially null.
    controller: RefCell<Option<&'static AutoAlignmentController>>,
    /// Rust-only: Java `this` for the listener and the 3dmod button's container.
    self_ref: Weak<AutoAlignmentPanel>,
}

impl AutoAlignmentPanel {
    /// Java private `AutoAlignmentPanel(BaseManager, boolean)`.
    fn new(manager: &'static dyn BaseManager, join_interface: bool) -> Rc<AutoAlignmentPanel> {
        Rc::new_cyclic(|self_ref: &Weak<AutoAlignmentPanel>| {
            let container: Weak<dyn Run3dmodButtonContainer> = self_ref.clone();
            let tc_align;
            let cb_find_warping;
            let ltf_warp_patch_size_x;
            let ltf_warp_patch_size_y;
            let cb_boundary_model;
            let btn_boundary_model;
            let ltf_shift_limits_for_warp_x;
            let ltf_shift_limits_for_warp_y;
            let cb_sobel_filter;
            if join_interface {
                tc_align = TransformChooserPanel::get_join_align_instance();
                cb_find_warping = None;
                ltf_warp_patch_size_x = None;
                ltf_warp_patch_size_y = None;
                cb_boundary_model = None;
                btn_boundary_model = None;
                ltf_shift_limits_for_warp_x = None;
                ltf_shift_limits_for_warp_y = None;
                cb_sobel_filter = None;
            } else {
                tc_align = TransformChooserPanel::get_serial_sections_instance();
                cb_find_warping = Some(CheckBox::new_string(Some("Find warping transformations")));
                ltf_warp_patch_size_x = Some(LabeledTextField::new_field_type_string(
                    FieldType::Integer,
                    Some("Correlation patch size in X: "),
                ));
                ltf_warp_patch_size_y = Some(LabeledTextField::new_field_type_string(
                    FieldType::Integer,
                    Some(" Y: "),
                ));
                cb_boundary_model = Some(CheckBox::new_string(Some("Use boundary model:")));
                btn_boundary_model = Some(
                    Run3dmodButton::get_3dmod_instance_string_run_3dmod_button_container(
                        Some("Create/View Boundary Model"),
                        Some(container.clone()),
                    ),
                );
                ltf_shift_limits_for_warp_x = Some(LabeledTextField::new_field_type_string(
                    FieldType::Integer,
                    Some("Limits to shifts in X: "),
                ));
                ltf_shift_limits_for_warp_y = Some(LabeledTextField::new_field_type_string(
                    FieldType::Integer,
                    Some(" Y: "),
                ));
                cb_sobel_filter = Some(CheckBox::new_string(Some("Apply Sobel filter")));
            }
            AutoAlignmentPanel {
                pnl_root: SpacedPanel::get_focusable_instance_void(),
                pnl_parameters: SpacedPanel::get_instance_void(),
                ltf_sigma_low_frequency: LabeledTextField::new_field_type_string(
                    FieldType::FloatingPoint,
                    Some("Sigma for low-frequency filter: "),
                ),
                ltf_cutoff_high_frequency: LabeledTextField::new_field_type_string(
                    FieldType::FloatingPoint,
                    Some("Cutoff for high-frequency filter: "),
                ),
                ltf_sigma_high_frequency: LabeledTextField::new_field_type_string(
                    FieldType::FloatingPoint,
                    Some("Sigma for high-frequency filter: "),
                ),
                pnl_buttons: SpacedPanel::get_instance_void(),
                btn_initial_auto_alignment: MultiLineButton::new_string(Some(
                    "Initial Auto Alignment",
                )),
                btn_midas: MultiLineButton::new_string(Some("Midas")),
                btn_refine_auto_alignment: MultiLineButton::new_string(Some(
                    "Refine with Auto Alignment",
                )),
                btn_revert_to_midas: MultiLineButton::new_string(Some(
                    "Revert Auto Alignment to Midas",
                )),
                btn_revert_to_empty: MultiLineButton::new_string(Some("Revert to No Transforms")),
                sp_reduce_by_binning: LabeledSpinner::get_defaulted_instance(
                    Some("Binning in search: "),
                    2,
                    1,
                    50,
                    1,
                    1,
                ),
                ltf_skip_sections_from1: LabeledTextField::new_field_type_string(
                    FieldType::IntegerList,
                    Some("Sections to skip: "),
                ),
                cb_pre_cross_correlation: CheckBox::new_string(Some(
                    "Find initial shifts with cross-correlation",
                )),
                ltf_edge_to_ignore: LabeledTextField::new_field_type_string(
                    FieldType::FloatingPoint,
                    Some("Fraction to ignore on edges: "),
                ),
                sp_midas_binning: Spinner::get_labeled_instance_string_int_int_int(
                    Some("Binning in Midas: "),
                    1,
                    1,
                    8,
                ),
                manager,
                join_interface,
                tc_align,
                cb_find_warping,
                ltf_warp_patch_size_x,
                ltf_warp_patch_size_y,
                cb_boundary_model,
                btn_boundary_model,
                ltf_shift_limits_for_warp_x,
                ltf_shift_limits_for_warp_y,
                cb_sobel_filter,
                controller: RefCell::new(None),
                self_ref: self_ref.clone(),
            }
        })
    }

    /// Java static package-private `getJoinInstance(BaseManager)`.
    pub fn get_join_instance(manager: &'static dyn BaseManager) -> Rc<AutoAlignmentPanel> {
        let instance = AutoAlignmentPanel::new(manager, true);
        instance.create_panel(true);
        instance.set_tooltips();
        instance
    }

    /// Java static package-private `getSerialSectionsInstance(BaseManager)`.
    pub fn get_serial_sections_instance(
        manager: &'static dyn BaseManager,
    ) -> Rc<AutoAlignmentPanel> {
        let instance = AutoAlignmentPanel::new(manager, false);
        instance.create_panel(false);
        instance.set_tooltips();
        instance
    }

    /// Java private `createPanel(boolean)`.
    fn create_panel(&self, join_configuration: bool) {
        // init
        self.ltf_sigma_low_frequency.set_text_string(Some("0.0"));
        self.ltf_cutoff_high_frequency.set_text_string(Some("0.35"));
        self.ltf_sigma_high_frequency.set_text_string(Some("0.05"));
        self.ltf_edge_to_ignore.set_text_string(Some("0.05"));
        // panels
        let pnl_pre_cross_correlation = JComponent::new_panel();
        let pnl_left_buttons = SpacedPanel::get_instance_void();
        let pnl_right_buttons = SpacedPanel::get_instance_void();
        let mut pnl_warping: Option<Rc<JComponent>> = None;
        let mut pnl_find_warping: Option<Rc<JComponent>> = None;
        let mut pnl_warp_patch_size: Option<Rc<JComponent>> = None;
        let mut pnl_boundary_model: Option<Rc<JComponent>> = None;
        let mut pnl_shift_limits_for_warp: Option<Rc<JComponent>> = None;
        let mut pnl_sobel_filter: Option<Rc<JComponent>> = None;
        if self.cb_sobel_filter.is_some() {
            pnl_sobel_filter = Some(JComponent::new_panel());
        }
        if self.cb_find_warping.is_some() {
            pnl_warping = Some(JComponent::new_panel());
            pnl_find_warping = Some(JComponent::new_panel());
            pnl_warp_patch_size = Some(JComponent::new_panel());
            pnl_boundary_model = Some(JComponent::new_panel());
            pnl_shift_limits_for_warp = Some(JComponent::new_panel());
        }
        // init
        if join_configuration {
            self.sp_reduce_by_binning.set_visible(false);
            self.ltf_skip_sections_from1.set_visible(false);
            self.cb_pre_cross_correlation.set_visible(false);
            self.ltf_edge_to_ignore.set_visible(false);
            self.sp_midas_binning.set_visible(false);
        }
        // root
        self.pnl_root.set_box_layout(spaced_panel::Y_AXIS);
        self.pnl_root
            .add_container(&self.pnl_parameters.get_container());
        self.pnl_root
            .add_container(&self.sp_midas_binning.get_container());
        self.pnl_root.add_container(&self.pnl_buttons.get_container());
        // parameters
        self.pnl_parameters.set_box_layout(spaced_panel::Y_AXIS);
        self.pnl_parameters
            .set_border(&EtchedBorder::new(Some("Auto Alignment Parameters")).get_border());
        self.pnl_parameters
            .add_labeled_text_field(&self.ltf_sigma_low_frequency);
        self.pnl_parameters
            .add_labeled_text_field(&self.ltf_cutoff_high_frequency);
        self.pnl_parameters
            .add_labeled_text_field(&self.ltf_sigma_high_frequency);
        if let Some(pnl_sobel_filter) = &pnl_sobel_filter {
            self.pnl_parameters.add_j_panel(pnl_sobel_filter);
        }
        self.pnl_parameters.add_j_panel(&pnl_pre_cross_correlation);
        self.pnl_parameters
            .add_component(&self.tc_align.get_component());
        if let Some(pnl_warping) = &pnl_warping {
            self.pnl_parameters.add_j_panel(pnl_warping);
        }
        self.pnl_parameters
            .add_container(&self.ltf_skip_sections_from1.get_container());
        self.pnl_parameters
            .add_labeled_text_field(&self.ltf_edge_to_ignore);
        self.pnl_parameters
            .add_container(&self.sp_reduce_by_binning.get_container());
        // SobelFilter panel
        if let (Some(pnl_sobel_filter), Some(cb_sobel_filter)) =
            (&pnl_sobel_filter, &self.cb_sobel_filter)
        {
            // Swing layout: BoxLayout X_AXIS, horizontal glue.
            pnl_sobel_filter.add(&cb_sobel_filter.get_component());
        }
        // warping panel
        if let Some(pnl_warping) = &pnl_warping {
            let pnl_find_warping = pnl_find_warping.as_ref().unwrap();
            let pnl_warp_patch_size = pnl_warp_patch_size.as_ref().unwrap();
            let pnl_boundary_model = pnl_boundary_model.as_ref().unwrap();
            let pnl_shift_limits_for_warp = pnl_shift_limits_for_warp.as_ref().unwrap();
            // Swing layout: BoxLayout Y_AXIS, rigid areas x0_y3 between.
            pnl_warping.set_border_title(EtchedBorder::new(Some("Warping")).get_title().as_deref());
            pnl_warping.add(pnl_find_warping);
            pnl_warping.add(pnl_warp_patch_size);
            pnl_warping.add(pnl_boundary_model);
            pnl_warping.add(pnl_shift_limits_for_warp);
            // FindWarping panel
            pnl_find_warping.add(&self.cb_find_warping.as_ref().unwrap().get_component());
            // WarpPatchSize panel
            pnl_warp_patch_size.add(&self.ltf_warp_patch_size_x.as_ref().unwrap().get_component());
            pnl_warp_patch_size.add(&self.ltf_warp_patch_size_y.as_ref().unwrap().get_component());
            // BoundaryModel panel
            pnl_boundary_model.add(&self.cb_boundary_model.as_ref().unwrap().get_component());
            pnl_boundary_model.add(&self.btn_boundary_model.as_ref().unwrap().get_component());
            // ShiftLimitsForWarp panel
            pnl_shift_limits_for_warp.add(
                &self
                    .ltf_shift_limits_for_warp_x
                    .as_ref()
                    .unwrap()
                    .get_component(),
            );
            pnl_shift_limits_for_warp.add(
                &self
                    .ltf_shift_limits_for_warp_y
                    .as_ref()
                    .unwrap()
                    .get_component(),
            );
        }
        // pre cross correlation
        pnl_pre_cross_correlation.add(&self.cb_pre_cross_correlation.get_component());
        // buttons
        self.pnl_buttons.set_box_layout(spaced_panel::X_AXIS);
        self.pnl_buttons.add_spaced_panel(&pnl_left_buttons);
        self.pnl_buttons.add_spaced_panel(&pnl_right_buttons);
        // left buttons
        pnl_left_buttons.set_box_layout(spaced_panel::Y_AXIS);
        pnl_left_buttons.add_multi_line_button(&self.btn_initial_auto_alignment);
        pnl_left_buttons.add_multi_line_button(&self.btn_midas);
        pnl_left_buttons.add_multi_line_button(&self.btn_refine_auto_alignment);
        // right buttons
        pnl_right_buttons.set_box_layout(spaced_panel::Y_AXIS);
        pnl_right_buttons.add_multi_line_button(&self.btn_revert_to_midas);
        pnl_right_buttons.add_multi_line_button(&self.btn_revert_to_empty);
        // display
        self.update_display();
    }

    /// Java package-private `setController(AutoAlignmentController)`.  Sets the
    /// controller and adds listeners.
    pub fn set_controller(&self, input: &'static AutoAlignmentController) {
        *self.controller.borrow_mut() = Some(input);
        self.add_listeners();
    }

    /// Java private `addListeners()`.
    fn add_listeners(&self) {
        // Java `new AutoAlignmentActionListener(this)`.
        let panel = self.self_ref.clone();
        let listener: ActionListener = Rc::new(move |event: &ActionEvent| {
            if let Some(panel) = panel.upgrade() {
                panel.action(event.get_action_command().unwrap_or(""), None, None);
            }
        });
        if let Some(cb_find_warping) = &self.cb_find_warping {
            cb_find_warping.add_action_listener(Some(listener.clone()));
            if let Some(cb_boundary_model) = &self.cb_boundary_model {
                cb_boundary_model.add_action_listener(Some(listener.clone()));
            }
            if let Some(btn_boundary_model) = &self.btn_boundary_model {
                btn_boundary_model.add_action_listener(listener.clone());
            }
        }
        self.btn_initial_auto_alignment
            .add_action_listener(listener.clone());
        self.btn_midas.add_action_listener(listener.clone());
        self.btn_refine_auto_alignment
            .add_action_listener(listener.clone());
        self.btn_revert_to_midas.add_action_listener(listener.clone());
        self.btn_revert_to_empty.add_action_listener(listener.clone());
        if !self.join_interface {
            self.tc_align.add_search_listener(listener);
        }
    }

    /// Java package-private `getRootComponent()`.
    pub fn get_root_component(&self) -> Rc<JComponent> {
        self.pnl_root.get_container()
    }

    /// Java package-private `getParameters(AutoAlignmentMetaData, boolean)`.
    pub fn get_parameters_meta_data(
        &self,
        meta_data: &mut AutoAlignmentMetaData,
        do_validation: bool,
    ) -> bool {
        let result = (|| -> Result<(), FieldValidationFailedException> {
            meta_data.set_sigma_low_frequency_string(
                self.ltf_sigma_low_frequency
                    .get_text_boolean(do_validation)?
                    .as_deref(),
            );
            meta_data.set_sigma_low_frequency_enabled(self.ltf_sigma_low_frequency.is_enabled());
            meta_data.set_cutoff_high_frequency_string(
                self.ltf_cutoff_high_frequency
                    .get_text_boolean(do_validation)?
                    .as_deref(),
            );
            meta_data
                .set_cutoff_high_frequency_enabled(self.ltf_cutoff_high_frequency.is_enabled());
            meta_data.set_sigma_high_frequency_string(
                self.ltf_sigma_high_frequency
                    .get_text_boolean(do_validation)?
                    .as_deref(),
            );
            meta_data.set_sigma_high_frequency_enabled(self.ltf_sigma_high_frequency.is_enabled());
            meta_data.set_align_transform(self.tc_align.get_transform());
            if let Some(cb_find_warping) = &self.cb_find_warping {
                meta_data.set_find_warping(cb_find_warping.is_selected());
                meta_data.set_warp_patch_size_x(
                    self.ltf_warp_patch_size_x
                        .as_ref()
                        .unwrap()
                        .get_text_void()
                        .as_deref(),
                );
                meta_data.set_warp_patch_size_y(
                    self.ltf_warp_patch_size_y
                        .as_ref()
                        .unwrap()
                        .get_text_void()
                        .as_deref(),
                );
                meta_data.set_boundary_model(self.cb_boundary_model.as_ref().unwrap().is_selected());
                meta_data.set_shift_limits_for_warp_x(
                    self.ltf_shift_limits_for_warp_x
                        .as_ref()
                        .unwrap()
                        .get_text_void()
                        .as_deref(),
                );
                meta_data.set_shift_limits_for_warp_y(
                    self.ltf_shift_limits_for_warp_y
                        .as_ref()
                        .unwrap()
                        .get_text_void()
                        .as_deref(),
                );
            }
            meta_data.set_pre_cross_correlation(self.cb_pre_cross_correlation.is_selected());
            meta_data.set_skip_sections_from1(
                self.ltf_skip_sections_from1
                    .get_text_boolean(do_validation)?
                    .as_deref(),
            );
            meta_data.set_edge_to_ignore(
                self.ltf_edge_to_ignore
                    .get_text_boolean(do_validation)?
                    .as_deref(),
            );
            meta_data.set_reduce_by_binning(Some(self.sp_reduce_by_binning.get_value()));
            meta_data.set_midas_binning(Some(self.sp_midas_binning.get_value()));
            if let Some(cb_sobel_filter) = &self.cb_sobel_filter {
                meta_data.set_sobel_filter(cb_sobel_filter.is_selected());
            }
            Ok(())
        })();
        result.is_ok()
    }

    /// Java package-private `setParameters(AutoAlignmentMetaData)`.
    pub fn set_parameters(&self, meta_data: &AutoAlignmentMetaData) {
        if !meta_data.is_sigma_low_frequency_null() {
            self.ltf_sigma_low_frequency
                .set_text_string(Some(&meta_data.get_sigma_low_frequency().to_string()));
        }
        if !meta_data.is_cutoff_high_frequency_null() {
            self.ltf_cutoff_high_frequency
                .set_text_string(Some(&meta_data.get_cutoff_high_frequency().to_string()));
        }
        if !meta_data.is_sigma_high_frequency_null() {
            self.ltf_sigma_high_frequency
                .set_text_string(Some(&meta_data.get_sigma_high_frequency().to_string()));
        }
        self.tc_align
            .set_transform(Some(meta_data.get_align_transform()));
        if let Some(cb_find_warping) = &self.cb_find_warping {
            cb_find_warping.set_selected_boolean(meta_data.is_find_warping());
            self.ltf_warp_patch_size_x
                .as_ref()
                .unwrap()
                .set_text_string(Some(&meta_data.get_warp_patch_size_x()));
            self.ltf_warp_patch_size_y
                .as_ref()
                .unwrap()
                .set_text_string(Some(&meta_data.get_warp_patch_size_y()));
            self.cb_boundary_model
                .as_ref()
                .unwrap()
                .set_selected_boolean(meta_data.is_boundary_model());
            self.ltf_shift_limits_for_warp_x
                .as_ref()
                .unwrap()
                .set_text_string(Some(&meta_data.get_shift_limits_for_warp_x()));
            self.ltf_shift_limits_for_warp_y
                .as_ref()
                .unwrap()
                .set_text_string(Some(&meta_data.get_shift_limits_for_warp_y()));
        }
        self.cb_pre_cross_correlation
            .set_selected_boolean(meta_data.is_pre_cross_correlation());
        self.ltf_skip_sections_from1
            .set_text_string(Some(&meta_data.get_skip_sections_from1()));
        if !meta_data.is_edge_to_ignore_null() {
            self.ltf_edge_to_ignore
                .set_text_double(meta_data.get_edge_to_ignore());
        }
        if !meta_data.is_reduce_by_binning_null() {
            self.sp_reduce_by_binning
                .set_value_const_etomo_number(meta_data.get_reduce_by_binning());
        }
        if !meta_data.is_midas_binning_null() {
            self.sp_midas_binning
                .set_value_const_etomo_number(meta_data.get_midas_binning());
        }
        if let Some(cb_sobel_filter) = &self.cb_sobel_filter {
            cb_sobel_filter.set_selected_boolean(meta_data.is_sobel_filter());
        }
        self.update_display();
    }

    /// Java package-private `getParameters(XfalignParam, boolean)`.
    pub fn get_parameters_xfalign_param(&self, param: &mut XfalignParam, do_validation: bool) -> bool {
        let result = (|| -> Result<(), FieldValidationFailedException> {
            if self.cb_pre_cross_correlation.is_visible() {
                param.set_pre_cross_correlation(self.cb_pre_cross_correlation.is_selected());
            } else {
                param.set_pre_cross_correlation(false);
            }
            if self.ltf_skip_sections_from1.is_visible() {
                param.set_skip_sections_from1(
                    self.ltf_skip_sections_from1
                        .get_text_boolean(do_validation)?
                        .as_deref(),
                );
            } else {
                param.reset_skip_sections_from1();
            }
            if self.ltf_edge_to_ignore.is_visible() && self.ltf_edge_to_ignore.is_enabled() {
                param.set_edge_to_ignore(
                    self.ltf_edge_to_ignore
                        .get_text_boolean(do_validation)?
                        .as_deref(),
                );
            } else {
                param.reset_edge_to_ignore();
            }
            if self.sp_reduce_by_binning.is_visible() && self.sp_reduce_by_binning.is_enabled() {
                param.set_reduce_by_binning(Some(self.sp_reduce_by_binning.get_value()));
            } else {
                param.reset_reduce_by_binning();
            }
            if let Some(cb_find_warping) = &self.cb_find_warping {
                if cb_find_warping.is_selected() {
                    param.set_warp_patch_size(
                        self.ltf_warp_patch_size_x
                            .as_ref()
                            .unwrap()
                            .get_text_boolean(do_validation)?
                            .as_deref(),
                        self.ltf_warp_patch_size_y
                            .as_ref()
                            .unwrap()
                            .get_text_boolean(do_validation)?
                            .as_deref(),
                    );
                    param.set_boundary_model(self.cb_boundary_model.as_ref().unwrap().is_selected());
                    param.set_shift_limits_for_warp(
                        self.ltf_shift_limits_for_warp_x
                            .as_ref()
                            .unwrap()
                            .get_text_boolean(do_validation)?
                            .as_deref(),
                        self.ltf_shift_limits_for_warp_y
                            .as_ref()
                            .unwrap()
                            .get_text_boolean(do_validation)?
                            .as_deref(),
                    );
                } else {
                    param.reset_warp_patch_size();
                    param.reset_boundary_model();
                    param.reset_shift_limits_for_warp();
                }
            }
            match &self.cb_sobel_filter {
                Some(cb_sobel_filter) if cb_sobel_filter.is_enabled() => {
                    param.set_sobel_filter(cb_sobel_filter.is_selected());
                }
                _ => param.reset_sobel_filter(),
            }
            Ok(())
        })();
        result.is_ok()
    }

    /// Java package-private `getParameters(MidasParam)`.
    pub fn get_parameters_midas_param(&self, param: &mut MidasParam) {
        param.set_binning(Some(self.sp_midas_binning.get_value()));
    }

    /// Java `equals(AutoAlignmentMetaData)`.  Checking if panel is equal to meta data.
    /// Set useDefault to match how useDefault is used in setMetaData().
    pub fn equals(&self, meta_data: &AutoAlignmentMetaData) -> bool {
        if !meta_data
            .get_sigma_low_frequency()
            .equals_string(self.ltf_sigma_low_frequency.get_text_void().as_deref())
        {
            return false;
        }
        if !meta_data
            .get_cutoff_high_frequency()
            .equals_string(self.ltf_cutoff_high_frequency.get_text_void().as_deref())
        {
            return false;
        }
        if !meta_data
            .get_sigma_high_frequency()
            .equals_string(self.ltf_sigma_high_frequency.get_text_void().as_deref())
        {
            return false;
        }
        if self.tc_align.get_transform() != meta_data.get_align_transform() {
            return false;
        }
        if let Some(cb_find_warping) = &self.cb_find_warping {
            if cb_find_warping.is_selected() != meta_data.is_find_warping() {
                return false;
            }
            if Some(meta_data.get_warp_patch_size_x())
                != self.ltf_warp_patch_size_x.as_ref().unwrap().get_text_void()
            {
                return false;
            }
            if Some(meta_data.get_warp_patch_size_y())
                != self.ltf_warp_patch_size_y.as_ref().unwrap().get_text_void()
            {
                return false;
            }
            if self.cb_boundary_model.as_ref().unwrap().is_selected()
                != meta_data.is_boundary_model()
            {
                return false;
            }
            if Some(meta_data.get_shift_limits_for_warp_x())
                != self
                    .ltf_shift_limits_for_warp_x
                    .as_ref()
                    .unwrap()
                    .get_text_void()
            {
                return false;
            }
            if Some(meta_data.get_shift_limits_for_warp_y())
                != self
                    .ltf_shift_limits_for_warp_y
                    .as_ref()
                    .unwrap()
                    .get_text_void()
            {
                return false;
            }
            if let Some(cb_sobel_filter) = &self.cb_sobel_filter
                && cb_sobel_filter.is_selected() != meta_data.is_sobel_filter()
            {
                return false;
            }
        }
        true
    }

    /// Java package-private `msgProcessChange(boolean)`.  The purpose of this function
    /// is to have Midas button enabled whether or not Initial Auto-Alignment succeeds.
    pub fn msg_process_change(&self, process_ended: bool) {
        self.btn_midas.set_enabled(process_ended);
        self.btn_revert_to_midas.set_enabled(process_ended);
        self.btn_revert_to_empty.set_enabled(process_ended);
    }

    /// Java private `updateDisplay()`.
    fn update_display(&self) {
        let search = self.tc_align.is_search();
        self.ltf_sigma_low_frequency.set_enabled(search);
        self.ltf_cutoff_high_frequency.set_enabled(search);
        self.ltf_sigma_high_frequency.set_enabled(search);
        if let Some(cb_sobel_filter) = &self.cb_sobel_filter {
            cb_sobel_filter.set_enabled(search);
        }
        if let Some(cb_find_warping) = &self.cb_find_warping {
            let find_warping = cb_find_warping.is_selected();
            self.ltf_warp_patch_size_x
                .as_ref()
                .unwrap()
                .set_enabled(find_warping);
            self.ltf_warp_patch_size_y
                .as_ref()
                .unwrap()
                .set_enabled(find_warping);
            self.cb_boundary_model
                .as_ref()
                .unwrap()
                .set_enabled(find_warping);
            self.btn_boundary_model.as_ref().unwrap().set_enabled(
                find_warping && self.cb_boundary_model.as_ref().unwrap().is_selected(),
            );
            self.ltf_shift_limits_for_warp_x
                .as_ref()
                .unwrap()
                .set_enabled(find_warping);
            self.ltf_shift_limits_for_warp_y
                .as_ref()
                .unwrap()
                .set_enabled(find_warping);
        }
        self.ltf_edge_to_ignore.set_enabled(search);
        self.sp_reduce_by_binning.set_enabled(search);
    }

    /// Java private `setTooltips()`.
    fn set_tooltips(&self) {
        self.ltf_sigma_low_frequency.set_tool_tip_text(Some(
            "Sigma of an inverted gaussian for filtering out low frequencies before \
             searching for transformation.  Filter is applied to binned image.",
        ));
        self.ltf_cutoff_high_frequency.set_tool_tip_text(Some(
            "Starting radius of a gaussian for filtering out high frequencies before \
             searching for transformation.  Filter is applied to binned image.",
        ));
        self.ltf_sigma_high_frequency.set_tool_tip_text(Some(
            "Sigma of gaussian for filtering out high frequencies before searching for \
             transformation.  Filter is applied to binned image.",
        ));
        self.btn_initial_auto_alignment.set_tool_tip_text(Some(
            "OPTIONAL:  Run xfalign.  Find preliminary translational alignments with \
             tiltxcorr rather then using an existing .xf file.",
        ));
        self.btn_midas.set_tool_tip_text(Some(
            "Open Midas to check the output of the auto alignment and to make \
             transformations by hand.",
        ));
        self.btn_refine_auto_alignment.set_tool_tip_text(Some(
            "OPTIONAL:  Run xfalign using preliminary alignments created by the most recent \
             use of Midas or xfalign.",
        ));
        self.btn_revert_to_midas.set_tool_tip_text(Some(
            "Use to ignore xfalign changes.  Returns transformations to the state created by \
             the most recent save done in Midas.",
        ));
        self.btn_revert_to_empty
            .set_tool_tip_text(Some("Use to remove all transformations."));
        self.sp_midas_binning
            .set_tool_tip_text(Some(shared_constants::MIDAS_BINNING_TOOLTIP));
        if let Some(cb_sobel_filter) = &self.cb_sobel_filter {
            cb_sobel_filter.set_tool_tip_text_string(Some(
                "Apply edge-detecting Sobel filter after image reduction and filtering, if any.",
            ));
        }
        if let Some(cb_find_warping) = &self.cb_find_warping {
            cb_find_warping.set_tool_tip_text_string(Some(
                "Align with non-linear warping by cross-correlating overlapping patches.",
            ));
            let text = "Size of patches to correlate in X and Y, in unbinned pixels.";
            self.ltf_warp_patch_size_x
                .as_ref()
                .unwrap()
                .set_tool_tip_text(Some(text));
            self.ltf_warp_patch_size_y
                .as_ref()
                .unwrap()
                .set_tool_tip_text(Some(text));
            self.cb_boundary_model
                .as_ref()
                .unwrap()
                .set_tool_tip_text_string(Some(
                    "Use model with contours around areas where patches should be correlated.",
                ));
            self.btn_boundary_model
                .as_ref()
                .unwrap()
                .set_tool_tip_text(Some(
                    "Open 3dmod to draw or see contours around areas to use for correlation.",
                ));
            let text = "Maximum pixels of shift for each patch.  If both fields are blank there \
                        are no limits; otherwise there must be a value in both fields.";
            self.ltf_shift_limits_for_warp_x
                .as_ref()
                .unwrap()
                .set_tool_tip_text(Some(text));
            self.ltf_shift_limits_for_warp_y
                .as_ref()
                .unwrap()
                .set_tool_tip_text(Some(text));
        }
        let mut autodoc: *const dyn ReadOnlyAutodoc = std::ptr::null::<Autodoc>();
        let result = (|| -> Result<(), LogFileError> {
            autodoc = unsafe {
                autodoc_factory::get_instance(
                    Some(self.manager),
                    Some(autodoc_factory::XFALIGN),
                    AxisID::Only,
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
        if autodoc.is_some() {
            self.cb_pre_cross_correlation.set_tool_tip_text_string(Some(
                "Use cross-correlation to find initial translations; needed if shifts are \
                 large.  This checkbox has no effect when refining with auto alignment",
            ));
            self.ltf_edge_to_ignore.set_tool_tip_text(
                etomo_autodoc::get_tooltip(autodoc, Some(xfalign_param::EDGE_TO_IGNORE_KEY))
                    .as_deref(),
            );
            self.sp_reduce_by_binning.set_tool_tip_text(
                etomo_autodoc::get_tooltip(autodoc, Some(xfalign_param::REDUCE_BY_BINNING_KEY))
                    .as_deref(),
            );
            self.ltf_skip_sections_from1.set_tool_tip_text(Some(&format!(
                "{}  Also sets the {} option:  {}",
                etomo_autodoc::get_tooltip(autodoc, Some(xfalign_param::SKIP_SECTIONS_KEY))
                    .unwrap_or_else(|| "null".to_string()),
                xfalign_param::SECTIONS_NUMBERED_FROM_ONE_KEY,
                etomo_autodoc::get_tooltip(
                    autodoc,
                    Some(xfalign_param::SECTIONS_NUMBERED_FROM_ONE_KEY)
                )
                .unwrap_or_else(|| "null".to_string())
            )));
        }
    }
}

impl Run3dmodButtonContainer for AutoAlignmentPanel {
    /// Java `action(String, Deferred3dmodButton, Run3dmodMenuOptions)`.
    fn action(
        &self,
        command: &str,
        _deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    ) {
        // Fixed in translation: an action before `setController` (no listener is
        // added before it, so none reaches here) would dereference null.
        let Some(controller) = *self.controller.borrow() else {
            return;
        };
        if Some(command) == self.btn_initial_auto_alignment.get_action_command().as_deref() {
            self.msg_process_change(false);
            controller.xfalign_initial(None, self.join_interface);
        } else if Some(command) == self.btn_midas.get_action_command().as_deref() {
            controller.midas_sample(
                self.btn_midas
                    .get_quoted_label()
                    .as_deref()
                    .unwrap_or("null"),
            );
        } else if Some(command) == self.btn_refine_auto_alignment.get_action_command().as_deref() {
            self.msg_process_change(false);
            controller.xfalign_refine(
                None,
                self.join_interface,
                self.btn_refine_auto_alignment
                    .get_quoted_label()
                    .as_deref()
                    .unwrap_or("null"),
            );
        } else if Some(command) == self.btn_revert_to_midas.get_action_command().as_deref() {
            controller.revert_xf_file_to_midas();
        } else if Some(command) == self.btn_revert_to_empty.get_action_command().as_deref() {
            controller.revert_xf_file_to_empty();
        } else if Some(command) == self.tc_align.get_search_action_command().as_deref() {
            self.update_display();
        } else if let Some(cb_find_warping) = &self.cb_find_warping {
            if Some(command) == cb_find_warping.get_action_command().as_deref() {
                self.update_display();
            } else if Some(command)
                == self
                    .cb_boundary_model
                    .as_ref()
                    .and_then(|cb_boundary_model| cb_boundary_model.get_action_command())
                    .as_deref()
            {
                self.update_display();
            } else if Some(command)
                == self
                    .btn_boundary_model
                    .as_ref()
                    .and_then(|btn_boundary_model| btn_boundary_model.get_action_command())
                    .as_deref()
            {
                controller.imod_boundary_model(run_3dmod_menu_options);
            }
        }
    }
}
