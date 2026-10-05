//! `IMOD/Etomo/src/etomo/ui/swing/CoarseAlignDialog.java`.

use crate::imod::etomo::ui::field::Field;
use std::ops::Deref;
use std::rc::{Rc, Weak};

use super::beveled_border::BeveledBorder;
use super::blendmont_display::BlendmontDisplay;
use super::check_box::CheckBox;
use super::coarse_align_display::CoarseAlignDisplay;
use super::context_menu::ContextMenu;
use super::generic_mouse_adapter::GenericMouseAdapter;
use super::context_popup::{self, ContextPopup};
use super::deferred_3dmod_button::Deferred3dmodButton;
use super::etched_border::EtchedBorder;
use super::etomo_panel::EtomoPanel;
use super::fiducialess_params::FiducialessParams;
use super::labeled_text_field::LabeledTextField;
use super::multi_line_button::MultiLineButton;
use super::newstack_display::NewstackDisplay;
use super::prenewst_panel::PrenewstPanel;
use super::process_dialog::{ProcessDialog, ProcessDialogVirtual};
use super::process_display::ProcessDisplay;
use super::run_3dmod_button_container::Run3dmodButtonContainer;
use super::spaced_panel::SpacedPanel;
use super::spinner::Spinner;
use super::tiltxcorr_display::TiltXcorrDisplay;
use super::tiltxcorr_panel::TiltxcorrPanel;
use super::ui_harness;
use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::blendmont_param::BlendmontParam;
use crate::imod::etomo::comscript::const_newst_param::ConstNewstParam;
use crate::imod::etomo::comscript::const_tiltxcorr_param::ConstTiltxcorrParam;
use crate::imod::etomo::comscript::midas_param::MidasParam;
use crate::imod::etomo::comscript::tomodataplots_param::Task;
use crate::imod::etomo::jdk::MouseEvent;
use crate::imod::etomo::jdk::{ActionEvent, ActionListener, JComponent};
use crate::imod::etomo::logic::dataset_tool;
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::base_screen_state::BaseScreenState;
use crate::imod::etomo::r#type::const_meta_data::ConstMetaData;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::file_type;
use crate::imod::etomo::r#type::meta_data::MetaData;
use crate::imod::etomo::r#type::process_result_display::ProcessResultDisplayHandle;
use crate::imod::etomo::r#type::recon_screen_state::ReconScreenState;
use crate::imod::etomo::r#type::view_type::ViewType;
use crate::imod::etomo::ui::field_type::FieldType;
use crate::imod::etomo::ui::field_validation_failed_exception::FieldValidationFailedException;

/// Java `public static final String rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `public final class CoarseAlignDialog extends ProcessDialog implements
/// ContextMenu, FiducialessParams, Run3dmodButtonContainer, CoarseAlignDisplay`.
pub struct CoarseAlignDialog {
    /// The `ProcessDialog` superclass.
    base: Rc<ProcessDialog>,
    pnl_coarse_align: Rc<EtomoPanel>,
    pnl_fiducialess: Rc<JComponent>,
    cb_fiducialess: Rc<CheckBox>,
    ltf_rotation: Rc<LabeledTextField>,

    /// Java `actionListener` (a `CoarseAlignActionListener`).
    action_listener: ActionListener,
    btn_midas: Rc<MultiLineButton>,
    tiltxcorr_panel: Rc<TiltxcorrPanel>,
    pnl_prenewst: Rc<PrenewstPanel>,
    // Montaging
    btn_fix_edges_midas: Rc<MultiLineButton>,
    btn_distortion_corrected_stack: Rc<MultiLineButton>,
    sp_midas_binning: Rc<Spinner>,
}

impl Deref for CoarseAlignDialog {
    type Target = ProcessDialog;
    fn deref(&self) -> &ProcessDialog {
        &self.base
    }
}

impl CoarseAlignDialog {
    /// Java private constructor `CoarseAlignDialog(ApplicationManager, AxisID,
    /// boolean)` (CoarseAlignDialog.java:332-387).
    fn new(
        app_mgr: &'static ApplicationManager,
        axis_id: AxisID,
        mag_changes_mode: bool,
    ) -> Rc<CoarseAlignDialog> {
        let dialog = Rc::new_cyclic(|weak: &Weak<CoarseAlignDialog>| {
            let base = ProcessDialog::new_application_manager_axis_id_dialog_type(
                app_mgr,
                axis_id,
                DialogType::CoarseAlignment,
            );
            let this: Weak<dyn ProcessDialogVirtual> = weak.clone();
            base.set_this(this);
            // Field initializers (CoarseAlignDialog.java:316-330).
            let pnl_coarse_align = EtomoPanel::new();
            let pnl_fiducialess = JComponent::new_panel();
            let cb_fiducialess = CheckBox::new_string(Some("Coarse alignment only"));
            let ltf_rotation = LabeledTextField::new_field_type_string(
                FieldType::FloatingPoint,
                Some("Tilt axis rotation:"),
            );
            let sp_midas_binning = Spinner::get_labeled_instance_string_int_int_int(
                Some("Binning in Midas: "),
                1,
                1,
                8,
            );

            // Constructor body.
            let display_factory = app_mgr.get_process_result_display_factory(axis_id);
            let btn_distortion_corrected_stack = display_factory.get_distortion_corrected_stack();
            let btn_fix_edges_midas = display_factory.get_fix_edges_midas();
            let btn_midas = display_factory.get_midas();
            // `setToolTipText()` (CoarseAlignDialog.java:342) runs after the
            // dialog exists; it only sets these buttons' and fields' tooltips.
            let context_menu: Weak<dyn ContextMenu> = weak.clone();
            let tiltxcorr_panel = TiltxcorrPanel::get_cross_correlation_instance(
                app_mgr,
                axis_id,
                base.dialog_type,
                &base.btn_advanced,
                Some(context_menu),
                mag_changes_mode,
            );
            let pnl_prenewst = PrenewstPanel::new(
                app_mgr,
                axis_id,
                base.dialog_type,
                weak.clone(),
                &base.btn_advanced,
            );
            // Java `actionListener = new CoarseAlignActionListener(this)`
            // (CoarseAlignDialog.java:383); the class is at :584-595.
            let adaptee = weak.clone();
            let action_listener: ActionListener = Rc::new(move |event: &ActionEvent| {
                let Some(adaptee) = adaptee.upgrade() else {
                    return;
                };
                adaptee.action(event.get_action_command().unwrap_or(""), None, None);
            });
            CoarseAlignDialog {
                base,
                pnl_coarse_align,
                pnl_fiducialess,
                cb_fiducialess,
                ltf_rotation,
                action_listener,
                btn_midas,
                tiltxcorr_panel,
                pnl_prenewst,
                btn_fix_edges_midas,
                btn_distortion_corrected_stack,
                sp_midas_binning,
            }
        });
        let meta_data = app_mgr.get_meta_data();
        dialog.set_tool_tip_text();
        dialog.btn_execute.set_text(Some("Done"));

        // Swing layout: pnlFiducialess BoxLayout Y_AXIS.  Each
        // `UIUtilities.addWithYSpace(p, c)` / `addWithSpace(p, c, dim)` is
        // `p.add(c)` plus a rigid area (UIUtilities.java:481-496).
        dialog
            .pnl_fiducialess
            .add(&dialog.cb_fiducialess.get_component());
        dialog
            .pnl_fiducialess
            .add(&dialog.ltf_rotation.get_container());
        // Swing layout: left align pnlFiducialess' components.

        // Swing layout: pnlCoarseAlign BoxLayout Y_AXIS.
        dialog
            .pnl_coarse_align
            .set_border(&BeveledBorder::new(Some("Coarse Alignment")).get_border());
        let coarse_align = dialog.pnl_coarse_align.get_component();
        coarse_align.add(&dialog.tiltxcorr_panel.get_panel());
        if meta_data.get_view_type() == ViewType::Montage {
            let pnl_fix_edges = SpacedPanel::get_instance_void();
            // Swing layout: pnlFixEdges.setBoxLayout(BoxLayout.Y_AXIS).
            pnl_fix_edges.set_border(&EtchedBorder::new(Some("Fix Edges")).get_border());
            pnl_fix_edges.add_multi_line_button(&dialog.btn_distortion_corrected_stack);
            pnl_fix_edges.add_multi_line_button(&dialog.btn_fix_edges_midas);
            coarse_align.add(&pnl_fix_edges.get_container());
            if !meta_data.is_distortion_correction() {
                dialog.btn_distortion_corrected_stack.set_enabled(false);
            }
            dialog.set_enabled_fix_edges_midas_button();
        }
        coarse_align.add(&dialog.pnl_prenewst.get_panel());
        coarse_align.add(&dialog.pnl_fiducialess);
        coarse_align.add(&dialog.sp_midas_binning.get_component());
        coarse_align.add(&dialog.btn_midas.get_component());

        // Set the alignment and size of the UI objects
        // Swing layout: center align pnlCoarseAlign's components;
        // UIUtilities.setButtonSizeAll(pnlCoarseAlign, button dimension).

        // Swing layout: rootPanel BoxLayout Y_AXIS.
        dialog.root_panel.get_component().add(&coarse_align);
        dialog.add_exit_buttons();

        // Set the default advanced state for the window
        dialog.update_advanced();
        dialog
    }

    /// Java `static getInstance(ApplicationManager, AxisID, boolean)`
    /// (CoarseAlignDialog.java:389-394).
    pub fn get_instance(
        app_mgr: &'static ApplicationManager,
        axis_id: AxisID,
        mag_changes_mode: bool,
    ) -> Rc<CoarseAlignDialog> {
        let instance = CoarseAlignDialog::new(app_mgr, axis_id, mag_changes_mode);
        instance.add_listeners();
        instance
    }

    /// Java private `addListeners()` (CoarseAlignDialog.java:396-405).
    fn add_listeners(self: &Rc<Self>) {
        // Action listener assignment for the buttons
        self.btn_midas
            .add_action_listener(self.action_listener.clone());
        self.btn_fix_edges_midas
            .add_action_listener(self.action_listener.clone());
        self.btn_distortion_corrected_stack
            .add_action_listener(self.action_listener.clone());

        // Mouse adapter for context menu
        let context_menu: Weak<dyn ContextMenu> = Rc::downgrade(self) as Weak<dyn ContextMenu>;
        self.pnl_coarse_align
            .get_component()
            .add_mouse_listener(GenericMouseAdapter::new(context_menu));
    }

    /// Java `setEnabledFixEdgesMidasButton()` (CoarseAlignDialog.java:412-421).
    /// Enable Fix Edges with Midas button if there are no distortion correction
    /// files in use or if the .dcst file (distortion corrected stack) has been
    /// created.
    pub fn set_enabled_fix_edges_midas_button(&self) {
        if self.btn_distortion_corrected_stack.is_enabled()
            && !BlendmontParam::get_distortion_corrected_file(
                self.application_manager,
                self.application_manager.get_property_user_dir().as_deref(),
                self.axis_id,
            )
            .exists()
        {
            self.btn_fix_edges_midas.set_enabled(false);
        } else {
            self.btn_fix_edges_midas.set_enabled(true);
        }
    }

    /// Java `setCrossCorrelationParams(ConstTiltxcorrParam)`
    /// (CoarseAlignDialog.java:426-428).  Set the parameters for the cross
    /// correlation panel.
    pub fn set_cross_correlation_params(&self, tilt_xcorr_params: &dyn ConstTiltxcorrParam) {
        self.tiltxcorr_panel
            .set_parameters_const_tiltxcorr_param(tilt_xcorr_params);
    }

    /// Java `getTiltXcorrDisplay()` (CoarseAlignDialog.java:430-432).
    pub fn get_tilt_xcorr_display(&self) -> Rc<dyn TiltXcorrDisplay> {
        self.tiltxcorr_panel.clone()
    }

    /// Java `getNewstackDisplay()` (CoarseAlignDialog.java:434-436).
    pub fn get_newstack_display(&self) -> Rc<dyn NewstackDisplay> {
        self.pnl_prenewst.clone()
    }

    /// Java `setPrenewstParams(ConstNewstParam)` (CoarseAlignDialog.java:442-444).
    /// Set the prenewst params of the prenewst panel.
    pub fn set_prenewst_params(&self, prenewst_param: &dyn ConstNewstParam) {
        NewstackDisplay::set_parameters(&*self.pnl_prenewst, prenewst_param);
    }

    /// Java `setParams(BlendmontParam)` (CoarseAlignDialog.java:450-452).  Set
    /// the blendmont params of the prenewst panel.
    pub fn set_params(&self, blendmont_param: &BlendmontParam) {
        BlendmontDisplay::set_parameters(&*self.pnl_prenewst, blendmont_param);
    }

    /// Java `setParameters(ReconScreenState)` (CoarseAlignDialog.java:454-457).
    pub fn set_parameters_recon_screen_state(&self, screen_state: &ReconScreenState) {
        self.tiltxcorr_panel
            .set_parameters_base_screen_state(screen_state);
        self.pnl_prenewst
            .set_parameters_base_screen_state(screen_state);
    }

    /// Java `setParameters(ConstMetaData)` (CoarseAlignDialog.java:459-461).
    pub fn set_parameters_const_meta_data(&self, meta_data: &dyn ConstMetaData) {
        self.pnl_prenewst.set_parameters_const_meta_data(meta_data);
    }

    /// Java `getParameters(BaseScreenState)` (CoarseAlignDialog.java:463-466).
    pub fn get_parameters_base_screen_state(&self, screen_state: &BaseScreenState) {
        self.tiltxcorr_panel
            .get_parameters_base_screen_state(screen_state);
        self.pnl_prenewst
            .get_parameters_base_screen_state(screen_state);
    }

    /// Java `getParameters(MetaData)` (CoarseAlignDialog.java:468-470).
    pub fn get_parameters_meta_data(&self, meta_data: &MetaData) {
        self.pnl_prenewst.get_parameters_meta_data(meta_data);
    }

    /// Java `setFiducialessAlignment(boolean)` (CoarseAlignDialog.java:472-474).
    pub fn set_fiducialess_alignment(&self, state: bool) {
        self.cb_fiducialess.set_selected_boolean(state);
    }

    /// Java `setImageRotation(String)` (CoarseAlignDialog.java:481-483).
    pub fn set_image_rotation(&self, tilt_axis_angle: Option<&str>) {
        self.ltf_rotation.set_text_string(tilt_axis_angle);
    }

    /// Java `updateAdvanced()` (CoarseAlignDialog.java:491-495).
    pub fn update_advanced(&self) {
        self.tiltxcorr_panel.update_advanced(self.is_advanced());
        self.pnl_prenewst.update_advanced(self.is_advanced());
        ui_harness::INSTANCE.with(|harness| {
            harness.pack_axis_id_base_manager(Some(self.axis_id), Some(self.application_manager))
        });
    }

    /// Java private `setToolTipText()` (CoarseAlignDialog.java:541-552).
    /// Tooltip string initialization.
    fn set_tool_tip_text(&self) {
        self.cb_fiducialess.set_tool_tip_text_string(Some(
            "Enable or disable the processing flow using cross-correlation alignment only.",
        ));
        self.btn_midas
            .set_tool_tip_text(Some("Use Midas to adjust bad alignments."));
        self.ltf_rotation.set_tool_tip_text(Some(
            "Initial rotation angle of tilt axis when viewing images in Midas.",
        ));
        self.btn_distortion_corrected_stack.set_tool_tip_text(Some(
            "Create a stack to use in Midas that incorporates the corrections from the image distortion field file and/or the magnification gradients file.",
        ));
        self.btn_fix_edges_midas.set_tool_tip_text(Some(
            "Use Midas to adjust the alignment of the montage frames.",
        ));
    }
}

impl ProcessDialogVirtual for CoarseAlignDialog {
    fn process_dialog(&self) -> &ProcessDialog {
        &self.base
    }

    /// Java `done()` (CoarseAlignDialog.java:573-582).
    fn done(&self) {
        self.application_manager
            .done_coarse_align_dialog(self.axis_id);
        self.tiltxcorr_panel.done();
        self.pnl_prenewst.done();
        self.btn_distortion_corrected_stack
            .remove_action_listener(&self.action_listener);
        self.btn_fix_edges_midas
            .remove_action_listener(&self.action_listener);
        self.btn_midas.remove_action_listener(&self.action_listener);
        self.set_displayed(false);
    }
}

impl FiducialessParams for CoarseAlignDialog {
    /// Java `isFiducialess()` (CoarseAlignDialog.java:476-479).
    fn is_fiducialess(&self) -> bool {
        self.cb_fiducialess.is_selected()
    }

    /// Java `getImageRotation(boolean) throws FieldValidationFailedException`
    /// (CoarseAlignDialog.java:485-489).
    fn get_image_rotation(
        &self,
        do_validation: bool,
    ) -> Result<String, FieldValidationFailedException> {
        // The trait returns a String; a text field's text is never null.
        Ok(self
            .ltf_rotation
            .get_text_boolean(do_validation)?
            .unwrap_or_default())
    }
}

impl ContextMenu for CoarseAlignDialog {
    /// Java `popUpContextMenu(MouseEvent)` (CoarseAlignDialog.java:500-536).
    /// Right mouse button context menu.
    fn pop_up_context_menu(&self, mouse_event: &MouseEvent) {
        let align_manpage_label;
        let align_manpage;
        let align_logfile_label;
        let align_logfile;
        let mut graph: Option<Vec<Task>> = None;
        let manager = self.application_manager;
        if manager.get_meta_data().get_view_type() == ViewType::Montage {
            align_manpage_label = "Blendmont";
            align_manpage = "blendmont";
            align_logfile_label = "Preblend";
            align_logfile = "preblend";
            if !dataset_tool::is_one_by(
                manager.get_property_user_dir().as_deref(),
                file_type::CLASS
                    .raw_stack
                    .get_file_name(Some(manager), Some(self.axis_id))
                    .as_deref(),
                manager,
                self.axis_id,
            ) && manager.get_state().is_xcorr_blendmont_was_run(self.axis_id)
            {
                graph = Some(vec![Task::CoarseMeanMax]);
            }
        } else {
            align_manpage_label = "Newstack";
            align_manpage = "newstack";
            align_logfile_label = "Prenewst";
            align_logfile = "prenewst";
        }
        let man_pagelabel = [
            "Tiltxcorr".to_string(),
            "Xftoxg".to_string(),
            align_manpage_label.to_string(),
            "3dmod".to_string(),
            "Midas".to_string(),
        ];
        let man_page = [
            "tiltxcorr.html".to_string(),
            "xftoxg.html".to_string(),
            align_manpage.to_string() + ".html",
            "3dmod.html".to_string(),
            "midas.html".to_string(),
        ];
        let log_file_label = ["Xcorr".to_string(), align_logfile_label.to_string()];
        let log_file = [
            "xcorr".to_string() + &self.axis_id.get_extension() + ".log",
            align_logfile.to_string() + &self.axis_id.get_extension() + ".log",
        ];

        let _context_popup = ContextPopup::new_component_mouse_event_string_string_string_array_string_array_string_array_string_array_task_array_file_array_base_manager_axis_id(
            &self.pnl_coarse_align.get_component(),
            mouse_event,
            Some("COARSE ALIGNMENT"),
            Some(context_popup::TOMO_GUIDE),
            &man_pagelabel,
            &man_page,
            &log_file_label,
            &log_file,
            graph.as_deref(),
            None,
            manager,
            self.axis_id,
        );
    }
}

impl Run3dmodButtonContainer for CoarseAlignDialog {
    /// Java `action(String, Deferred3dmodButton, Run3dmodMenuOptions)`
    /// (CoarseAlignDialog.java:558-571).  Action function for process buttons.
    fn action(
        &self,
        command: &str,
        _deferred3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        _menu_options: Option<Run3dmodMenuOptions>,
    ) {
        if Some(command) == self.btn_midas.get_action_command().as_deref() {
            self.application_manager.midas_raw_stack(
                self.axis_id,
                Some(self.btn_midas.clone() as ProcessResultDisplayHandle),
                self,
            );
        } else if Some(command) == self.btn_fix_edges_midas.get_action_command().as_deref() {
            self.application_manager.midas_fix_edges(
                self.axis_id,
                Some(self.btn_fix_edges_midas.clone() as ProcessResultDisplayHandle),
            );
        } else if Some(command)
            == self
                .btn_distortion_corrected_stack
                .get_action_command()
                .as_deref()
        {
            self.application_manager.make_distortion_corrected_stack(
                self.axis_id,
                Some(self.btn_distortion_corrected_stack.clone() as ProcessResultDisplayHandle),
                None,
            );
        }
    }
}

impl ProcessDisplay for CoarseAlignDialog {}

impl CoarseAlignDisplay for CoarseAlignDialog {
    /// Java `getCoarseAlignParameters(MidasParam)` (CoarseAlignDialog.java:597-600).
    fn get_coarse_align_parameters(&self, param: &mut MidasParam) {
        param.set_binning(Some(self.sp_midas_binning.get_value()));
    }
}
