//! `IMOD/Etomo/src/etomo/ui/swing/CcdEraserXRaysPanel.java`.
//!
//! The x-ray and manual replacement half of the pre-processing (CCD eraser)
//! panel.  The module keeps the live tree's name (`ccd_eraser_xrays_panel`) so
//! that `mod.rs` does not change when this file replaces the earlier one.

use crate::imod::etomo::ui::field::Field;
use std::cell::RefCell;
use std::rc::{Rc, Weak};

use super::ccd_eraser_display::CcdEraserDisplay;
use super::check_box::CheckBox;
use super::context_menu::ContextMenu;
use super::context_popup::{self, ContextPopup};
use super::deferred_3dmod_button::Deferred3dmodButton;
use super::etched_border::EtchedBorder;
use super::etomo_panel::EtomoPanel;
use super::expand_button::ExpandButton;
use super::expandable::Expandable;
use super::global_expand_button::GlobalExpandButton;
use super::labeled_text_field::LabeledTextField;
use super::multi_line_button::MultiLineButton;
use super::panel_header::PanelHeader;
use super::process_display::ProcessDisplay;
use super::run_3dmod_button::Run3dmodButton;
use super::run_3dmod_button_container::Run3dmodButtonContainer;
use super::ui_harness;
use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::comscript::ccd_eraser_param::CCDEraserParam;
use crate::imod::etomo::comscript::const_ccd_eraser_param::{self, ConstCCDEraserParam};
use crate::imod::etomo::comscript::makecomfile_param::MakecomfileParam;
use crate::imod::etomo::comscript::tomodataplots_param::Task;
use crate::imod::etomo::jdk::MouseEvent;
use crate::imod::etomo::jdk::{ActionEvent, ActionListener, JComponent};
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::storage::autodoc::autodoc::Autodoc;
use crate::imod::etomo::storage::autodoc::autodoc_factory;
use crate::imod::etomo::storage::autodoc::read_only_autodoc::ReadOnlyAutodoc;
use crate::imod::etomo::storage::log_file::LogFileError;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::base_screen_state::BaseScreenState;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::etomo_autodoc;
use crate::imod::etomo::r#type::file_key::FileKey;
use crate::imod::etomo::r#type::file_type;
use crate::imod::etomo::r#type::process_result_display::ProcessResultDisplayHandle;
use crate::imod::etomo::ui::field_type::FieldType;
use crate::imod::etomo::ui::field_validation_failed_exception::FieldValidationFailedException;

/// Java `static final String ERASE_LABEL`.
pub const ERASE_LABEL: &str = "Create Fixed Stack";
/// Java `static final String USE_FIXED_STACK_LABEL`.
pub const USE_FIXED_STACK_LABEL: &str = "Use Fixed Stack";

/// Java `final class CcdEraserXRaysPanel implements ContextMenu,
/// Run3dmodButtonContainer, CcdEraserDisplay, Expandable`.
pub struct CcdEraserXRaysPanel {
    /// Java `this`, handed to the manager as the `CcdEraserDisplay`.
    this: Weak<CcdEraserXRaysPanel>,
    pnl_ccd_eraser: Rc<JComponent>,
    pnl_manual_replacement: Rc<JComponent>,
    cb_xray_replacement: Rc<CheckBox>,
    ltf_peak_criterion: Rc<LabeledTextField>,
    ltf_diff_criterion: Rc<LabeledTextField>,
    ltf_grow_criterion: Rc<LabeledTextField>,
    ltf_edge_exclusion: Rc<LabeledTextField>,
    ltf_maximum_radius: Rc<LabeledTextField>,
    ltf_annulus_width: Rc<LabeledTextField>,
    ltf_scan_region_size: Rc<LabeledTextField>,
    ltf_scan_criterion: Rc<LabeledTextField>,
    btn_find_x_rays: Rc<Run3dmodButton>,
    btn_view_x_ray_model: Rc<Run3dmodButton>,
    cb_manual_replacement: Rc<CheckBox>,
    ltf_global_replacement_list: Rc<LabeledTextField>,
    ltf_local_replacement_list: Rc<LabeledTextField>,
    ltf_boundary_replacement_list: Rc<LabeledTextField>,
    btn_create_model: Rc<Run3dmodButton>,
    ltf_border_pixels: Rc<LabeledTextField>,
    ltf_polynomial_order: Rc<LabeledTextField>,
    cb_include_adjacent_points: Rc<CheckBox>,
    btn_view_erased: Rc<Run3dmodButton>,
    btn_clip_stats_raw: Rc<MultiLineButton>,
    btn_clip_stats_fixed: Rc<MultiLineButton>,
    ltf_giant_criterion: Rc<LabeledTextField>,
    ltf_big_diff_criterion: Rc<LabeledTextField>,
    ltf_extra_large_radius: Rc<LabeledTextField>,
    /// Java `phManualReplacement`; never null once constructed, but
    /// `expand(ExpandButton)` tests it (CcdEraserXRaysPanel.java:384).
    ph_manual_replacement: Option<Rc<PanelHeader>>,
    pnl_manual_replacement_body: Rc<JComponent>,
    pnl_manual_buttons: Rc<JComponent>,

    application_manager: &'static ApplicationManager,
    axis_id: AxisID,
    btn_erase: Rc<Run3dmodButton>,
    btn_replace_raw_stack: Rc<MultiLineButton>,
    dialog_type: DialogType,
    /// Java `ccdEraserActionListener` (a `CCDEraserXRaysActionListener`).
    /// Kept so that `done()` can remove it by identity.
    ccd_eraser_action_listener: ActionListener,
}

impl CcdEraserXRaysPanel {
    /// Java private constructor `CcdEraserXRaysPanel(ApplicationManager, AxisID,
    /// DialogType, GlobalExpandButton)` (CcdEraserXRaysPanel.java:107-232).
    fn new(
        app_mgr: &'static ApplicationManager,
        id: AxisID,
        dialog_type: DialogType,
        global_advanced_button: &Rc<GlobalExpandButton>,
    ) -> Rc<CcdEraserXRaysPanel> {
        let panel = Rc::new_cyclic(|weak: &Weak<CcdEraserXRaysPanel>| {
            let container: Weak<dyn Run3dmodButtonContainer> = weak.clone();
            // Field initializers (CcdEraserXRaysPanel.java:47-98).
            let pnl_ccd_eraser = JComponent::new_panel();
            let pnl_manual_replacement = JComponent::new_panel();
            let cb_xray_replacement = CheckBox::new_string(Some("Automatic x-ray replacement"));
            let ltf_peak_criterion = LabeledTextField::new_field_type_string(
                FieldType::FloatingPoint,
                Some("Peak criterion:"),
            );
            let ltf_diff_criterion = LabeledTextField::new_field_type_string(
                FieldType::FloatingPoint,
                Some("Difference criterion:"),
            );
            let ltf_grow_criterion = LabeledTextField::new_field_type_string(
                FieldType::FloatingPoint,
                Some("Grow criterion:"),
            );
            let ltf_edge_exclusion = LabeledTextField::new_field_type_string(
                FieldType::Integer,
                Some("Edge exclusion:"),
            );
            let ltf_maximum_radius = LabeledTextField::new_field_type_string(
                FieldType::FloatingPoint,
                Some("Maximum radius:"),
            );
            let ltf_annulus_width = LabeledTextField::new_field_type_string(
                FieldType::FloatingPoint,
                Some("Annulus width:"),
            );
            let ltf_scan_region_size =
                LabeledTextField::new_field_type_string(FieldType::Integer, Some("XY scan size:"));
            let ltf_scan_criterion = LabeledTextField::new_field_type_string(
                FieldType::FloatingPoint,
                Some("Scan criterion:"),
            );
            let btn_view_x_ray_model =
                Run3dmodButton::get_3dmod_instance_string_run_3dmod_button_container(
                    Some("View X-ray Model"),
                    Some(container.clone()),
                );
            let cb_manual_replacement = CheckBox::new_string(Some("Manual replacement"));
            let ltf_global_replacement_list = LabeledTextField::new_field_type_string(
                FieldType::IntegerList,
                Some("All section replacement list: "),
            );
            let ltf_local_replacement_list = LabeledTextField::new_field_type_string(
                FieldType::IntegerList,
                Some("Line replacement list: "),
            );
            let ltf_boundary_replacement_list = LabeledTextField::new_field_type_string(
                FieldType::IntegerList,
                Some("Boundary replacement list: "),
            );
            let btn_create_model =
                Run3dmodButton::get_3dmod_instance_string_run_3dmod_button_container(
                    Some("Create Manual Replacement Model"),
                    Some(container.clone()),
                );
            let ltf_border_pixels = LabeledTextField::new_field_type_string(
                FieldType::Integer,
                Some("Border pixels: "),
            );
            let ltf_polynomial_order = LabeledTextField::new_field_type_string(
                FieldType::Integer,
                Some("Polynomial order: "),
            );
            let cb_include_adjacent_points = CheckBox::new_string(Some("Include adjacent points"));
            // FileType extends FileKey; the FileType passed as the FileKey.
            let btn_view_erased =
                Run3dmodButton::get_3dmod_instance_string_run_3dmod_button_container_file_key(
                    Some("View Fixed Stack"),
                    Some(container.clone()),
                    Some(FileKey::clone(&file_type::CLASS.fixed_xrays_stack)),
                );
            let btn_clip_stats_raw =
                MultiLineButton::new_string(Some("Show Min/Max for Raw Stack"));
            let btn_clip_stats_fixed =
                MultiLineButton::new_string(Some("Show Min/Max for Fixed Stack"));
            let ltf_giant_criterion = LabeledTextField::new_field_type_string(
                FieldType::FloatingPoint,
                Some("Extra-large peak criterion:"),
            );
            let ltf_big_diff_criterion = LabeledTextField::new_field_type_string(
                FieldType::FloatingPoint,
                Some("Extra-large difference criterion:"),
            );
            let ltf_extra_large_radius = LabeledTextField::new_field_type_string(
                FieldType::FloatingPoint,
                Some("Maximum radius of extra-large peak:"),
            );
            let pnl_manual_replacement_body = JComponent::new_panel();
            let pnl_manual_buttons = JComponent::new_panel();

            // Constructor body (CcdEraserXRaysPanel.java:109-124).
            let expandable: Weak<dyn Expandable> = weak.clone();
            global_advanced_button.register_expandable(expandable.clone());
            let ph_manual_replacement = PanelHeader::get_advanced_basic_only_instance(
                Some("Manual Pixel Region Replacement"),
                Some(expandable),
                Some(DialogType::PreProcessing),
                Some(global_advanced_button.clone()),
                true,
            );
            let display_factory = app_mgr.get_process_result_display_factory(id);
            let btn_erase = display_factory.get_create_fixed_stack();
            btn_erase.set_container(Some(container.clone()));
            btn_erase.set_deferred_3dmod_button_deferred_3dmod_button(Some(
                btn_view_erased.clone() as Rc<dyn Deferred3dmodButton>,
            ));
            let btn_find_x_rays = display_factory.get_find_xrays();
            btn_find_x_rays.set_container(Some(container.clone()));
            btn_find_x_rays.set_deferred_3dmod_button_deferred_3dmod_button(Some(
                btn_view_x_ray_model.clone() as Rc<dyn Deferred3dmodButton>,
            ));
            let btn_replace_raw_stack = display_factory.get_use_fixed_stack();

            // Java `ccdEraserActionListener = new CCDEraserXRaysActionListener(this)`
            // (CcdEraserXRaysPanel.java:231); inner class at :564-576.
            let adaptee = weak.clone();
            let ccd_eraser_action_listener: ActionListener = Rc::new(move |event: &ActionEvent| {
                let Some(adaptee) = adaptee.upgrade() else {
                    return;
                };
                adaptee.action(event.get_action_command().unwrap_or(""), None, None);
            });

            CcdEraserXRaysPanel {
                pnl_ccd_eraser,
                pnl_manual_replacement,
                cb_xray_replacement,
                ltf_peak_criterion,
                ltf_diff_criterion,
                ltf_grow_criterion,
                ltf_edge_exclusion,
                ltf_maximum_radius,
                ltf_annulus_width,
                ltf_scan_region_size,
                ltf_scan_criterion,
                btn_find_x_rays,
                btn_view_x_ray_model,
                cb_manual_replacement,
                ltf_global_replacement_list,
                ltf_local_replacement_list,
                ltf_boundary_replacement_list,
                btn_create_model,
                ltf_border_pixels,
                ltf_polynomial_order,
                cb_include_adjacent_points,
                btn_view_erased,
                btn_clip_stats_raw,
                btn_clip_stats_fixed,
                ltf_giant_criterion,
                ltf_big_diff_criterion,
                ltf_extra_large_radius,
                ph_manual_replacement: Some(ph_manual_replacement),
                pnl_manual_replacement_body,
                pnl_manual_buttons,
                application_manager: app_mgr,
                axis_id: id,
                btn_erase,
                btn_replace_raw_stack,
                dialog_type,
                ccd_eraser_action_listener,
                this: weak.clone(),
            }
        });
        // CcdEraserXRaysPanel.java:125.
        panel.set_tool_tip_text();

        let pnl_manual_replacement_check_box = JComponent::new_panel();
        let pnl_x_ray_replacement = EtomoPanel::new();
        // Swing layout: pnlXRayReplacement BoxLayout Y_AXIS.
        pnl_x_ray_replacement
            .set_border(&EtchedBorder::new(Some("Automatic X-ray Replacement")).get_border());

        // Each `UIUtilities.addWithYSpace(panel, c)` is `panel.add(c)` followed
        // by a rigid area (UIUtilities.java:481-484); the rigid areas are layout.
        let x = pnl_x_ray_replacement.get_component();
        x.add(&panel.cb_xray_replacement.get_component());
        x.add(&panel.ltf_peak_criterion.get_container());
        x.add(&panel.ltf_diff_criterion.get_container());
        x.add(&panel.ltf_maximum_radius.get_container());
        x.add(&panel.ltf_giant_criterion.get_container());
        x.add(&panel.ltf_big_diff_criterion.get_container());
        x.add(&panel.ltf_extra_large_radius.get_container());
        x.add(&panel.ltf_grow_criterion.get_container());
        x.add(&panel.ltf_edge_exclusion.get_container());
        x.add(&panel.ltf_annulus_width.get_container());
        x.add(&panel.ltf_scan_region_size.get_container());
        x.add(&panel.ltf_scan_criterion.get_container());

        let pnl_x_ray_buttons = JComponent::new_panel();
        // Swing layout: BoxLayout X_AXIS with horizontal glue between buttons.
        pnl_x_ray_buttons.add(&panel.btn_find_x_rays.get_component());
        pnl_x_ray_buttons.add(&panel.btn_view_x_ray_model.get_component());
        // Swing layout: UIUtilities.setButtonSizeAll(pnlXRayButtons, button dimension).

        x.add(&pnl_x_ray_buttons);

        // Swing layout: pnlManualReplacement BoxLayout Y_AXIS, untitled etched border.
        panel.pnl_manual_replacement.add(
            &panel
                .ph_manual_replacement
                .as_ref()
                .unwrap()
                .get_container(),
        );
        panel
            .pnl_manual_replacement
            .add(&panel.pnl_manual_replacement_body);

        // Swing layout: pnlManualReplacementBody BoxLayout Y_AXIS.
        let body = &panel.pnl_manual_replacement_body;
        body.add(&pnl_manual_replacement_check_box);
        body.add(&panel.ltf_global_replacement_list.get_container());
        body.add(&panel.ltf_local_replacement_list.get_container());
        body.add(&panel.ltf_boundary_replacement_list.get_container());

        // Swing layout: pnlManualButtons BoxLayout X_AXIS with glue.
        panel
            .pnl_manual_buttons
            .add(&panel.btn_create_model.get_component());
        // Swing layout: UIUtilities.setButtonSizeAll(pnlManualButtons, button dimension).
        body.add(&panel.pnl_manual_buttons);

        // Swing layout: pnlManualReplacementCheckBox BoxLayout X_AXIS, glue.
        pnl_manual_replacement_check_box.add(&panel.cb_manual_replacement.get_component());

        // Swing layout: pnlCCDEraser BoxLayout Y_AXIS.
        let root = &panel.pnl_ccd_eraser;
        root.add(&pnl_x_ray_replacement.get_component());
        root.add(&panel.pnl_manual_replacement);
        root.add(&panel.ltf_border_pixels.get_container());
        root.add(&panel.ltf_polynomial_order.get_container());
        root.add(&panel.cb_include_adjacent_points.get_component());

        // Swing layout: rigid area x0_y5.
        let pnl_erase_buttons = JComponent::new_panel();
        // Swing layout: pnlEraseButtons BoxLayout Y_AXIS.
        let pnl_erase = JComponent::new_panel();
        // Swing layout: pnlErase BoxLayout X_AXIS with glue between buttons.
        pnl_erase.add(&panel.btn_erase.get_component());
        pnl_erase.add(&panel.btn_view_erased.get_component());
        pnl_erase.add(&panel.btn_replace_raw_stack.get_component());
        let pnl_clip_stats = JComponent::new_panel();
        // Swing layout: pnlClipStats BoxLayout X_AXIS with glue between buttons.
        pnl_clip_stats.add(&panel.btn_clip_stats_raw.get_component());
        pnl_clip_stats.add(&panel.btn_clip_stats_fixed.get_component());
        pnl_erase_buttons.add(&pnl_erase);
        pnl_erase_buttons.add(&pnl_clip_stats);
        // Swing layout: UIUtilities.setButtonSizeAll on pnlErase and pnlClipStats.

        root.add(&pnl_erase_buttons);

        // Swing layout: left align the components of pnlXRayReplacement and
        // pnlCCDEraser; center align pnlCCDEraser.

        panel.enable_x_ray_replacement();
        panel.enable_manual_replacement();
        panel.update_manual_replacement_advanced(global_advanced_button.is_expanded());
        panel
    }

    /// Java `static getInstance(ApplicationManager, AxisID, DialogType,
    /// GlobalExpandButton)` (CcdEraserXRaysPanel.java:234-240).
    pub fn get_instance(
        app_mgr: &'static ApplicationManager,
        id: AxisID,
        dialog_type: DialogType,
        global_advanced_button: &Rc<GlobalExpandButton>,
    ) -> Rc<CcdEraserXRaysPanel> {
        let instance = CcdEraserXRaysPanel::new(app_mgr, id, dialog_type, global_advanced_button);
        instance.add_listeners();
        instance
    }

    /// Java `addListeners()` (CcdEraserXRaysPanel.java:242-257).
    fn add_listeners(&self) {
        // Mouse adapter for context menu: `pnlCCDEraser.addMouseListener(new
        // GenericMouseAdapter(this))` - mouse events are not modelled.
        let listener = &self.ccd_eraser_action_listener;
        self.btn_find_x_rays.add_action_listener(listener.clone());
        self.btn_view_x_ray_model
            .add_action_listener(listener.clone());
        self.btn_create_model.add_action_listener(listener.clone());
        self.btn_erase.add_action_listener(listener.clone());
        self.btn_view_erased.add_action_listener(listener.clone());
        self.btn_replace_raw_stack
            .add_action_listener(listener.clone());
        self.cb_xray_replacement
            .add_action_listener(Some(listener.clone()));
        self.cb_manual_replacement
            .add_action_listener(Some(listener.clone()));
        self.btn_clip_stats_raw
            .add_action_listener(listener.clone());
        self.btn_clip_stats_fixed
            .add_action_listener(listener.clone());
    }

    /// Java `setParameters(ConstCCDEraserParam)` (CcdEraserXRaysPanel.java:263-285).
    /// Set the fields.
    pub fn set_parameters_const_ccd_eraser_param(&self, ccd_eraser_params: &ConstCCDEraserParam) {
        self.cb_xray_replacement
            .set_selected_boolean(ccd_eraser_params.is_find_peaks());
        self.ltf_peak_criterion
            .set_text_string(ccd_eraser_params.get_peak_criterion().as_deref());
        self.ltf_diff_criterion
            .set_text_string(ccd_eraser_params.get_diff_criterion().as_deref());
        self.ltf_grow_criterion
            .set_text_string(ccd_eraser_params.get_grow_criterion().as_deref());
        self.ltf_scan_criterion
            .set_text_string(ccd_eraser_params.get_scan_criterion().as_deref());
        self.ltf_maximum_radius
            .set_text_string(ccd_eraser_params.get_maximum_radius().as_deref());
        self.ltf_annulus_width
            .set_text_string(ccd_eraser_params.get_annulus_width().as_deref());
        self.ltf_scan_region_size
            .set_text_string(ccd_eraser_params.get_xy_scan_size().as_deref());
        self.ltf_edge_exclusion
            .set_text_string(ccd_eraser_params.get_edge_exclusion().as_deref());
        // Java `!ccdEraserParams.getModelFile().equals("")`, which throws a
        // NullPointerException when the model file is null
        // (CcdEraserXRaysPanel.java:273).  Fixed in translation: a null model
        // file is treated as no model file (manual replacement off).
        self.cb_manual_replacement.set_selected_boolean(
            ccd_eraser_params
                .get_model_file()
                .is_some_and(|model_file| model_file != ""),
        );
        self.ltf_global_replacement_list
            .set_text_string(ccd_eraser_params.get_global_replacement_list().as_deref());
        self.ltf_local_replacement_list
            .set_text_string(ccd_eraser_params.getlocal_replacement_list().as_deref());
        self.ltf_boundary_replacement_list
            .set_text_string(ccd_eraser_params.get_boundary_replacement_list().as_deref());
        self.ltf_border_pixels
            .set_text_string(ccd_eraser_params.get_border_pixels().as_deref());
        self.ltf_polynomial_order
            .set_text_string(ccd_eraser_params.get_polynomial_order().as_deref());
        self.cb_include_adjacent_points
            .set_selected_boolean(ccd_eraser_params.get_include_adjacent_points());
        self.ltf_giant_criterion
            .set_text_string(ccd_eraser_params.get_giant_criterion().as_deref());
        self.ltf_big_diff_criterion
            .set_text_string(ccd_eraser_params.get_big_diff_criterion().as_deref());
        self.ltf_extra_large_radius
            .set_text_string(ccd_eraser_params.get_extra_large_radius().as_deref());
        self.enable_x_ray_replacement();
        self.enable_manual_replacement();
    }

    /// Java `done()` (CcdEraserXRaysPanel.java:287-291).
    pub fn done(&self) {
        let listener = &self.ccd_eraser_action_listener;
        self.btn_find_x_rays.remove_action_listener(listener);
        self.btn_erase.remove_action_listener(listener);
        self.btn_replace_raw_stack.remove_action_listener(listener);
    }

    /// Java `setParameters(BaseScreenState)` (CcdEraserXRaysPanel.java:293-295).
    pub fn set_parameters_base_screen_state(&self, screen_state: &BaseScreenState) {
        self.ph_manual_replacement
            .as_ref()
            .unwrap()
            .set_button_states_base_screen_state(Some(screen_state));
    }

    /// Java `getParameters(BaseScreenState)` (CcdEraserXRaysPanel.java:297-299).
    pub fn get_parameters_base_screen_state(&self, screen_state: &BaseScreenState) {
        self.ph_manual_replacement
            .as_ref()
            .unwrap()
            .get_button_states(Some(screen_state));
    }

    /// Java `getContainer()` (CcdEraserXRaysPanel.java:349-351).  Return the
    /// container of panel.
    pub fn get_container(&self) -> Rc<JComponent> {
        self.pnl_ccd_eraser.clone()
    }

    /// Java `updateAdvanced(boolean)` (CcdEraserXRaysPanel.java:357-369).
    /// Makes the advanced components visible or invisible.
    pub fn update_advanced(&self, state: bool) {
        self.cb_xray_replacement.set_visible(state);
        self.ltf_grow_criterion.set_visible(state);
        self.ltf_edge_exclusion.set_visible(state);
        self.ltf_annulus_width.set_visible(state);
        self.ltf_scan_region_size.set_visible(state);
        self.ltf_scan_criterion.set_visible(state);
        self.ltf_border_pixels.set_visible(state);
        self.ltf_polynomial_order.set_visible(state);
        self.cb_include_adjacent_points.set_visible(state);
        self.ltf_giant_criterion.set_visible(state);
        self.ltf_extra_large_radius.set_visible(state);
    }

    /// Java `updateManualReplacementAdvanced(boolean)` (CcdEraserXRaysPanel.java:371-374).
    pub fn update_manual_replacement_advanced(&self, advanced: bool) {
        self.pnl_manual_replacement_body.set_visible(advanced);
        ui_harness::INSTANCE.with(|harness| {
            harness.pack_axis_id_base_manager(Some(self.axis_id), Some(self.application_manager))
        });
    }

    /// Java private `enableXRayReplacement()` (CcdEraserXRaysPanel.java:461-476).
    fn enable_x_ray_replacement(&self) {
        let state = self.cb_xray_replacement.is_selected();
        self.ltf_peak_criterion.set_enabled(state);
        self.ltf_diff_criterion.set_enabled(state);
        self.ltf_grow_criterion.set_enabled(state);
        self.ltf_edge_exclusion.set_enabled(state);
        self.ltf_maximum_radius.set_enabled(state);
        self.ltf_annulus_width.set_enabled(state);
        self.ltf_scan_region_size.set_enabled(state);
        self.ltf_scan_criterion.set_enabled(state);
        self.btn_find_x_rays.set_enabled(state);
        self.btn_view_x_ray_model.set_enabled(state);
        self.ltf_giant_criterion.set_enabled(state);
        self.ltf_big_diff_criterion.set_enabled(state);
        self.ltf_extra_large_radius.set_enabled(state);
    }

    /// Java private `enableManualReplacement()` (CcdEraserXRaysPanel.java:478-484).
    fn enable_manual_replacement(&self) {
        let state = self.cb_manual_replacement.is_selected();
        self.ltf_global_replacement_list.set_enabled(state);
        self.ltf_local_replacement_list.set_enabled(state);
        self.ltf_boundary_replacement_list.set_enabled(state);
        self.btn_create_model.set_enabled(state);
    }

    /// Java private `setToolTipText()` (CcdEraserXRaysPanel.java:489-561).
    /// Tooltip string initialization.
    fn set_tool_tip_text(&self) {
        // Java `ReadOnlyAutodoc autodoc = null;` then the try/catch.
        let mut autodoc: *const dyn ReadOnlyAutodoc = std::ptr::null::<Autodoc>();
        match unsafe {
            autodoc_factory::get_instance(
                Some(self.application_manager),
                Some(autodoc_factory::CCDERASER),
                self.axis_id,
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
        // SAFETY: `autodoc` is null or an autodoc the factory keeps for the
        // life of the process.
        let autodoc: Option<&dyn ReadOnlyAutodoc> = if autodoc.is_null() {
            None
        } else {
            Some(unsafe { &*autodoc })
        };
        let tooltip = |key: &str| etomo_autodoc::get_tooltip(autodoc, Some(key));
        self.cb_xray_replacement
            .set_tool_tip_text_string(tooltip(const_ccd_eraser_param::FIND_PEAKS_KEY).as_deref());
        self.ltf_peak_criterion
            .set_tool_tip_text(tooltip(const_ccd_eraser_param::PEAK_CRITERION_KEY).as_deref());
        self.ltf_diff_criterion
            .set_tool_tip_text(tooltip(const_ccd_eraser_param::DIFF_CRITERION_KEY).as_deref());
        self.ltf_grow_criterion
            .set_tool_tip_text(tooltip(const_ccd_eraser_param::GROW_CRITERION_KEY).as_deref());
        self.ltf_scan_criterion
            .set_tool_tip_text(tooltip(const_ccd_eraser_param::SCAN_CRITERION_KEY).as_deref());
        self.ltf_maximum_radius
            .set_tool_tip_text(tooltip(const_ccd_eraser_param::MAXIMUM_RADIUS_KEY).as_deref());
        self.ltf_annulus_width
            .set_tool_tip_text(tooltip(const_ccd_eraser_param::ANNULUS_WIDTH_KEY).as_deref());
        self.ltf_scan_region_size
            .set_tool_tip_text(tooltip(const_ccd_eraser_param::X_Y_SCAN_SIZE_KEY).as_deref());
        self.ltf_edge_exclusion.set_tool_tip_text(
            tooltip(const_ccd_eraser_param::EDGE_EXCLUSION_WIDTH_KEY).as_deref(),
        );
        self.ltf_local_replacement_list
            .set_tool_tip_text(tooltip(const_ccd_eraser_param::LINE_OBJECTS_KEY).as_deref());
        self.ltf_boundary_replacement_list
            .set_tool_tip_text(tooltip(const_ccd_eraser_param::BOUNDARY_OBJECTS_KEY).as_deref());
        self.cb_manual_replacement.set_tool_tip_text_string(Some(
            "Use a manually created model to specify regions and lines to replace.",
        ));
        self.ltf_global_replacement_list
            .set_tool_tip_text(tooltip(const_ccd_eraser_param::ALL_SECTION_OBJECTS_KEY).as_deref());
        self.ltf_border_pixels
            .set_tool_tip_text(tooltip(const_ccd_eraser_param::BORDER_SIZE_KEY).as_deref());
        self.ltf_polynomial_order
            .set_tool_tip_text(tooltip(const_ccd_eraser_param::POLYNOMIAL_ORDER_KEY).as_deref());
        self.cb_include_adjacent_points
            .set_tool_tip_text_string(Some(
                &*("Include pixels adjacent to the patch being replaced in the pixels "
                    .to_string()
                    + "being fit."),
            ));
        self.btn_find_x_rays
            .set_tool_tip_text(tooltip(const_ccd_eraser_param::TRIAL_MODE_KEY).as_deref());
        self.ltf_giant_criterion
            .set_tool_tip_text(tooltip(const_ccd_eraser_param::GIANT_CRITERION_KEY).as_deref());
        self.ltf_big_diff_criterion
            .set_tool_tip_text(tooltip(const_ccd_eraser_param::BIG_DIFF_CRITERION_KEY).as_deref());
        self.ltf_extra_large_radius
            .set_tool_tip_text(tooltip(const_ccd_eraser_param::EXTRA_LARGE_RADIUS_KEY).as_deref());
        self.btn_view_x_ray_model
            .set_tool_tip_text(Some("View the x-ray model on the raw stack in 3dmod."));
        self.btn_create_model
            .set_tool_tip_text(Some("Create a manual replacement model using 3dmod."));
        self.btn_erase.set_tool_tip_text(Some(
            &*("Run ccderaser, erasing the raw stack and writing the modified stack ".to_string()
                + "to the output file specified (the default is *_fixed.st).  "
                + "NOTE: subsequent processing uses the "
                + "raw stack filename, therefore for ccderaser to have an effect on "
                + "your data you must commit the raw stack when you are satisfied with"
                + " your ccderaser output stack."),
        ));
        self.btn_view_erased
            .set_tool_tip_text(Some("View the erased stack in 3dmod."));
        self.btn_replace_raw_stack.set_tool_tip_text(Some(
            &*("Use the raw stack with the output from ccderaser.  ".to_string()
                + "NOTE: subsequent processing uses the "
                + "raw stack filename, therefore for ccderaser to have an effect on "
                + "your data you must commit the raw stack when you are satisfied with"
                + " your ccderaser output stack."),
        ));
        self.btn_clip_stats_raw.set_tool_tip_text(Some(
            &*("Run clip stats on the raw stack.  Prints information ".to_string()
                + "about each section"),
        ));
        self.btn_clip_stats_fixed.set_tool_tip_text(Some(
            &*("Run clip stats on the stack created by the ".to_string()
                + ERASE_LABEL
                + " button.  Prints information about each section."),
        ));
    }
}

impl ProcessDisplay for CcdEraserXRaysPanel {
    fn as_ccd_eraser_display(&self) -> Option<&dyn CcdEraserDisplay> {
        Some(self)
    }
}

impl CcdEraserDisplay for CcdEraserXRaysPanel {
    /// Java `getParameters(CCDEraserParam, boolean)` (CcdEraserXRaysPanel.java:301-338).
    fn get_parameters(&self, ccd_eraser_params: &mut CCDEraserParam, do_validation: bool) -> bool {
        // Java `try { ... } catch (FieldValidationFailedException e) { return false; }`.
        let result: Result<(), FieldValidationFailedException> = (|| {
            ccd_eraser_params.set_find_peaks(self.cb_xray_replacement.is_selected());
            ccd_eraser_params.set_peak_criterion(
                self.ltf_peak_criterion
                    .get_text_boolean(do_validation)?
                    .as_deref(),
            );
            ccd_eraser_params.set_diff_criterion(
                self.ltf_diff_criterion
                    .get_text_boolean(do_validation)?
                    .as_deref(),
            );
            ccd_eraser_params.set_grow_criterion(
                self.ltf_grow_criterion
                    .get_text_boolean(do_validation)?
                    .as_deref(),
            );
            ccd_eraser_params.set_scan_criterion(
                self.ltf_scan_criterion
                    .get_text_boolean(do_validation)?
                    .as_deref(),
            );
            ccd_eraser_params.set_maximum_radius(
                self.ltf_maximum_radius
                    .get_text_boolean(do_validation)?
                    .as_deref(),
            );
            ccd_eraser_params.set_annulus_width(
                self.ltf_annulus_width
                    .get_text_boolean(do_validation)?
                    .as_deref(),
            );
            ccd_eraser_params.set_xy_scan_size(
                self.ltf_scan_region_size
                    .get_text_boolean(do_validation)?
                    .as_deref(),
            );
            ccd_eraser_params.set_edge_exclusion(
                self.ltf_edge_exclusion
                    .get_text_boolean(do_validation)?
                    .as_deref(),
            );
            ccd_eraser_params.set_global_replacement_list(
                self.ltf_global_replacement_list
                    .get_text_boolean(do_validation)?
                    .as_deref(),
            );
            ccd_eraser_params.set_local_replacement_list(
                self.ltf_local_replacement_list
                    .get_text_boolean(do_validation)?
                    .as_deref(),
            );
            ccd_eraser_params.set_boundary_replacement_list(
                self.ltf_boundary_replacement_list
                    .get_text_boolean(do_validation)?
                    .as_deref(),
            );
            ccd_eraser_params.set_border_pixels(
                self.ltf_border_pixels
                    .get_text_boolean(do_validation)?
                    .as_deref(),
            );
            ccd_eraser_params.set_polynomial_order(
                self.ltf_polynomial_order
                    .get_text_boolean(do_validation)?
                    .as_deref(),
            );
            ccd_eraser_params
                .set_include_adjacent_points(self.cb_include_adjacent_points.is_selected());
            ccd_eraser_params.set_giant_criterion(
                self.ltf_giant_criterion
                    .get_text_boolean(do_validation)?
                    .as_deref(),
            );
            ccd_eraser_params.set_big_diff_criterion(
                self.ltf_big_diff_criterion
                    .get_text_boolean(do_validation)?
                    .as_deref(),
            );
            ccd_eraser_params.set_extra_large_radius(
                self.ltf_extra_large_radius
                    .get_text_boolean(do_validation)?
                    .as_deref(),
            );
            if self.cb_manual_replacement.is_selected() {
                ccd_eraser_params.set_model_file(
                    file_type::CLASS
                        .manual_replacement_model
                        .get_file_name(Some(self.application_manager), Some(self.axis_id))
                        .as_deref(),
                );
            } else {
                ccd_eraser_params.set_model_file(Some(""));
            }
            Ok(())
        })();
        result.is_ok()
    }

    /// Java `getParameters(MakecomfileParam, boolean)` (CcdEraserXRaysPanel.java:340-343).
    fn get_parameters_makecomfile(
        &self,
        _param: &mut MakecomfileParam,
        _do_validation: bool,
    ) -> bool {
        true
    }
}

impl Expandable for CcdEraserXRaysPanel {
    /// Java `expand(GlobalExpandButton)` (CcdEraserXRaysPanel.java:376-380).
    fn expand_global_expand_button(&self, button: &Rc<GlobalExpandButton>) {
        self.update_advanced(button.is_expanded());
        ui_harness::INSTANCE.with(|harness| {
            harness.pack_axis_id_base_manager(Some(self.axis_id), Some(self.application_manager))
        });
    }

    /// Java `expand(ExpandButton)` (CcdEraserXRaysPanel.java:382-387).
    fn expand_expand_button(&self, button: &Rc<ExpandButton>) {
        if let Some(ph_manual_replacement) = &self.ph_manual_replacement {
            if ph_manual_replacement.equals_advanced_basic(button) {
                self.update_manual_replacement_advanced(button.is_expanded());
            }
        }
    }
}

impl Run3dmodButtonContainer for CcdEraserXRaysPanel {
    /// Java `action(String, Deferred3dmodButton, Run3dmodMenuOptions)`
    /// (CcdEraserXRaysPanel.java:398-435).  Executes the action associated with
    /// command.  Deferred3dmodButton is null if it comes from
    /// CCDEraserActionListener.  Otherwise is comes from a Run3dmodButton which
    /// called action(Run3dmodButton, Run3dmoMenuOptions).  In that case it will
    /// be null unless it was set in the Run3dmodButton.
    fn action(
        &self,
        command: &str,
        deferred3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run3dmod_menu_options: Option<Run3dmodMenuOptions>,
    ) {
        let manager = self.application_manager;
        // The manager takes the options by value; a Java null is the default
        // (no options set).
        let run3dmod_menu_options = run3dmod_menu_options.unwrap_or_default();
        if Some(command) == self.btn_find_x_rays.get_action_command().as_deref() {
            manager.find_xrays(
                self.axis_id,
                Some(self.btn_find_x_rays.clone() as ProcessResultDisplayHandle),
                None,
                deferred3dmod_button,
                run3dmod_menu_options,
                self.dialog_type,
                self,
            );
        } else if Some(command) == self.btn_erase.get_action_command().as_deref() {
            manager.pre_eraser(
                self.axis_id,
                Some(self.btn_erase.clone() as ProcessResultDisplayHandle),
                None,
                deferred3dmod_button,
                run3dmod_menu_options,
                self.dialog_type,
                self.this.upgrade().expect("CcdEraserXRaysPanel dropped")
                    as Rc<dyn CcdEraserDisplay>,
            );
        } else if Some(command) == self.btn_replace_raw_stack.get_action_command().as_deref() {
            manager.replace_raw_stack(
                self.axis_id,
                Some(self.btn_replace_raw_stack.clone() as ProcessResultDisplayHandle),
                self.dialog_type,
            );
        } else if Some(command) == self.cb_xray_replacement.get_action_command().as_deref() {
            self.enable_x_ray_replacement();
        } else if Some(command) == self.cb_manual_replacement.get_action_command().as_deref() {
            self.enable_manual_replacement();
        } else if Some(command) == self.btn_view_x_ray_model.get_action_command().as_deref() {
            manager.imod_xray_model(self.axis_id, run3dmod_menu_options);
        } else if Some(command) == self.btn_create_model.get_action_command().as_deref() {
            manager.imod_manual_erase(self.axis_id, run3dmod_menu_options, self.dialog_type);
        } else if Some(command) == self.btn_view_erased.get_action_command().as_deref() {
            manager.imod_erased_stack(self.axis_id, run3dmod_menu_options);
        } else if Some(command) == self.btn_clip_stats_raw.get_action_command().as_deref() {
            manager.clip_stats(
                self.axis_id,
                &file_type::CLASS.raw_stack,
                None,
                self.dialog_type,
                Task::MinMax,
                None,
            );
        } else if Some(command) == self.btn_clip_stats_fixed.get_action_command().as_deref() {
            manager.clip_stats(
                self.axis_id,
                &file_type::CLASS.fixed_xrays_stack,
                None,
                self.dialog_type,
                Task::FixedMinMax,
                None,
            );
        }
    }
}

impl ContextMenu for CcdEraserXRaysPanel {
    /// Java `popUpContextMenu(MouseEvent)` (CcdEraserXRaysPanel.java:441-459).
    /// Right mouse button context menu.
    fn pop_up_context_menu(&self, mouse_event: &MouseEvent) {
        let label = ["CCDEraser".to_string()];
        let man_page = ["ccderaser.html".to_string()];

        let log_file_label = ["Eraser".to_string()];
        let log_file = ["eraser".to_string() + &self.axis_id.get_extension() + ".log"];

        let graph = [Task::MinMax, Task::FixedMinMax];

        // TEMP alignframes
        let graph_input_file = [
            None,
            file_type::CLASS
                .fixed_stats_log
                .get_file(Some(self.application_manager), Some(self.axis_id)),
        ];
        // TEMP alignframes
        let _context_popup = ContextPopup::new_component_mouse_event_string_string_string_array_string_array_string_array_string_array_task_array_file_array_base_manager_axis_id(
            &self.pnl_ccd_eraser,
            mouse_event,
            Some("PRE-PROCESSING"),
            Some(context_popup::TOMO_GUIDE),
            &label,
            &man_page,
            &log_file_label,
            &log_file,
            Some(&graph[..]),
            Some(&graph_input_file[..]),
            self.application_manager,
            self.axis_id,
        );
    }
}
