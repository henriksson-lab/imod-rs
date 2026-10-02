//! `IMOD/Etomo/src/etomo/ui/swing/EraseGoldPanel.java`.
//!
//! Java `final class EraseGoldPanel implements ContextMenu`: the "Erase Gold"
//! tab of the final aligned stack dialog.  It chooses the model creation
//! method (existing fiducial model via `XfModelPanel`, or findbeads3d via
//! `Beads3dFindPanel`) and holds the bead eraser (`CcdEraserBeadsPanel`).
//!
//! An EDT object (`Rc<Self>`, `&self` methods).  The inner listener class
//! `EraseGoldPanelActionListener` is a closure holding a weak reference to the
//! panel.  The parent dialog owns this panel, so the Java `parent` field is a
//! `Weak<FinalAlignedStackDialog>`.

use std::rc::{Rc, Weak};

use super::beads3d_find_panel::Beads3dFindPanel;
use super::blendmont_display::BlendmontDisplay;
use super::ccd_eraser_beads_panel::CcdEraserBeadsPanel;
use super::ccd_eraser_display::CcdEraserDisplay;
use super::context_menu::ContextMenu;
use super::context_popup::{self, ContextPopup};
use super::deferred_3dmod_button::Deferred3dmodButton;
use super::etched_border::EtchedBorder;
use super::final_aligned_stack_dialog::FinalAlignedStackDialog;
use super::find_beads3d_display::FindBeads3dDisplay;
use super::global_expand_button::GlobalExpandButton;
use super::newstack_display::NewstackDisplay;
use super::radio_button::RadioButton;
use super::tilt_display::TiltDisplay;
use super::ui_harness;
use super::xf_model_panel::XfModelPanel;
use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::blendmont_param::BlendmontParam;
use crate::imod::etomo::comscript::const_ccd_eraser_param::ConstCCDEraserParam;
use crate::imod::etomo::comscript::const_find_beads3d_param::ConstFindBeads3dParam;
use crate::imod::etomo::comscript::const_tilt_param::ConstTiltParam;
use crate::imod::etomo::comscript::const_tiltalign_param::ConstTiltalignParam;
use crate::imod::etomo::comscript::fortran_input_syntax_exception::FortranInputSyntaxException;
use crate::imod::etomo::comscript::newst_param::NewstParam;
use crate::imod::etomo::jdk::{ActionEvent, ActionListener, ButtonGroup, JComponent, MouseEvent};
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_meta_data::ConstMetaData;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::meta_data::MetaData;
use crate::imod::etomo::r#type::recon_screen_state::ReconScreenState;
use crate::imod::etomo::r#type::tomogram_state::TomogramState;
use crate::imod::etomo::r#type::view_type::ViewType;

/// Java package-private static final `ERASE_GOLD_TAB_LABEL`.
pub const ERASE_GOLD_TAB_LABEL: &str = "Erase Gold";

/// Java `final class EraseGoldPanel implements ContextMenu`.
pub struct EraseGoldPanel {
    /// Java `this`.
    this: Weak<EraseGoldPanel>,
    /// Java private final `pnlRoot = new JPanel()`.
    pnl_root: Rc<JComponent>,
    /// Java private final `actionListener` (`EraseGoldPanelActionListener`).
    action_listener: ActionListener,
    /// Java private final `bgModel = new ButtonGroup()`.
    bg_model: Rc<ButtonGroup>,
    /// Java private final `rbModelUseFid`.
    rb_model_use_fid: Rc<RadioButton>,
    /// Java private final `rbModelUseFindBeads3d`.
    rb_model_use_find_beads3d: Rc<RadioButton>,

    /// Java private final `xfModelPanel`.
    xf_model_panel: Rc<XfModelPanel>,
    /// Java private final `beads3dFindPanel`.
    beads3d_find_panel: Rc<Beads3dFindPanel>,
    /// Java private final `ccdEraserBeadsPanel`.
    ccd_eraser_beads_panel: Rc<CcdEraserBeadsPanel>,
    /// Java private final `manager`.
    manager: &'static ApplicationManager,
    /// Java private final `axisID`.
    axis_id: AxisID,
    /// Java private final `dialogType`.
    dialog_type: DialogType,
    /// Java private final `parent` (the owning dialog; held weakly).
    parent: Weak<FinalAlignedStackDialog>,
}

impl EraseGoldPanel {
    /// Java private constructor `EraseGoldPanel(ApplicationManager,
    /// FinalAlignedStackDialog, AxisID, DialogType, GlobalExpandButton)`.
    fn new(
        manager: &'static ApplicationManager,
        parent: Weak<FinalAlignedStackDialog>,
        axis_id: AxisID,
        dialog_type: DialogType,
        global_advanced_button: &Rc<GlobalExpandButton>,
    ) -> Rc<EraseGoldPanel> {
        Rc::new_cyclic(|this: &Weak<EraseGoldPanel>| {
            // Field initializers, in declaration order.
            let pnl_root = JComponent::new_panel();
            // Java `new EraseGoldPanelActionListener(this)`.
            let adaptee = this.clone();
            let action_listener: ActionListener = Rc::new(move |event: &ActionEvent| {
                let Some(adaptee) = adaptee.upgrade() else {
                    return;
                };
                adaptee.action(event.get_action_command().unwrap_or(""), None, None);
            });
            let bg_model = ButtonGroup::new();
            let rb_model_use_fid = RadioButton::new_string_button_group(
                Some("Use the existing fiducial model"),
                Some(&bg_model),
            );
            let rb_model_use_find_beads3d =
                RadioButton::new_string_button_group(Some("Use findbeads3d"), Some(&bg_model));
            // Constructor body.
            // TODO(unit): needs etomo/ui/swing/XfModelPanel.java - the live
            // xf_model_panel.rs is a generic state model; this assumes the faithful
            // `get_instance(&'static ApplicationManager, AxisID, DialogType) ->
            // Rc<XfModelPanel>`.
            let xf_model_panel = XfModelPanel::get_instance(manager, axis_id, dialog_type);
            let beads3d_find_panel = Beads3dFindPanel::get_instance(
                manager,
                this.clone(),
                axis_id,
                dialog_type,
                global_advanced_button,
            );
            let ccd_eraser_beads_panel =
                CcdEraserBeadsPanel::get_instance(manager, axis_id, dialog_type);
            EraseGoldPanel {
                this: this.clone(),
                pnl_root,
                action_listener,
                bg_model,
                rb_model_use_fid,
                rb_model_use_find_beads3d,
                xf_model_panel,
                beads3d_find_panel,
                ccd_eraser_beads_panel,
                manager,
                axis_id,
                dialog_type,
                parent,
            }
        })
    }

    /// Java static `getInstance(ApplicationManager, FinalAlignedStackDialog,
    /// AxisID, DialogType, GlobalExpandButton)`.
    pub fn get_instance(
        manager: &'static ApplicationManager,
        parent: Weak<FinalAlignedStackDialog>,
        axis_id: AxisID,
        dialog_type: DialogType,
        global_advanced_button: &Rc<GlobalExpandButton>,
    ) -> Rc<EraseGoldPanel> {
        let instance = EraseGoldPanel::new(
            manager,
            parent,
            axis_id,
            dialog_type,
            global_advanced_button,
        );
        instance.create_panel();
        instance.add_listeners();
        instance.set_tool_tip_text();
        instance
    }

    /// Java private `addListeners()`.
    fn add_listeners(&self) {
        // Swing mouse: pnlRoot.addMouseListener(new GenericMouseAdapter(this)) -
        // mouse events are not modelled by jdk.rs; popUpContextMenu is the
        // `ContextMenu` implementation below.
        self.rb_model_use_fid
            .add_action_listener(self.action_listener.clone());
        self.rb_model_use_find_beads3d
            .add_action_listener(self.action_listener.clone());
    }

    /// Java package-private `reregisterProcessingMethodMediator()`.
    pub fn reregister_processing_method_mediator(&self) {
        self.beads3d_find_panel
            .reregister_processing_method_mediator();
    }

    // <p>updates done</p>

    /// Java private `createPanel()`.
    fn create_panel(&self) {
        // Local panels
        let pnl_model = JComponent::new_panel();
        // Root panel
        // Swing layout: pnlRoot BoxLayout Y_AXIS.
        self.pnl_root.set_border_title(
            EtchedBorder::new(Some("Bead Eraser"))
                .get_border()
                .get_title()
                .as_deref(),
        );
        self.pnl_root.add(&pnl_model);
        self.pnl_root.add(&self.xf_model_panel.get_component());
        self.pnl_root.add(&self.beads3d_find_panel.get_component());
        self.pnl_root
            .add(&self.ccd_eraser_beads_panel.get_component());
        // Model panel
        // Swing layout: pnlModel BoxLayout Y_AXIS, Box.CENTER_ALIGNMENT.
        pnl_model.set_border_title(
            EtchedBorder::new(Some("Model Creation Method"))
                .get_border()
                .get_title()
                .as_deref(),
        );
        pnl_model.add(&self.rb_model_use_fid.get_component());
        pnl_model.add(&self.rb_model_use_find_beads3d.get_component());
    }

    /// Java package-private `getComponent()`.
    pub fn get_component(&self) -> Rc<JComponent> {
        self.update_aligned_stack_binning();
        self.pnl_root.clone()
    }

    /// Java package-private `isFiducialess()`.  The parent dialog owns this
    /// panel, so it is alive whenever the panel is used; a dropped parent
    /// answers false.
    pub fn is_fiducialess(&self) -> bool {
        match self.parent.upgrade() {
            Some(parent) => parent.is_fiducialess(),
            None => false,
        }
    }

    /// Java package-private `initializeBeads()`.
    pub fn initialize_beads(&self) {
        self.ccd_eraser_beads_panel.initialize();
    }

    /// Java package-private `updateAdvanced(boolean)`.
    pub fn update_advanced(&self, advanced: bool) {
        self.beads3d_find_panel.update_advanced(advanced);
    }

    /// Java package-private `getBlendmont3dFindDisplay()`.
    pub fn get_blendmont3d_find_display(&self) -> Option<Rc<dyn BlendmontDisplay>> {
        self.beads3d_find_panel.get_blendmont3d_find_display()
    }

    /// Java package-private `getNewstack3dFindDisplay()`.
    pub fn get_newstack3d_find_display(&self) -> Option<Rc<dyn NewstackDisplay>> {
        self.beads3d_find_panel.get_newstack3d_find_display()
    }

    /// Java package-private `getTilt3dFindDisplay()`.
    pub fn get_tilt3d_find_display(&self) -> Option<Rc<dyn TiltDisplay>> {
        self.beads3d_find_panel.get_tilt3d_find_display()
    }

    /// Java package-private `getFindBeads3dDisplay()`.
    pub fn get_find_beads3d_display(&self) -> Option<Rc<dyn FindBeads3dDisplay>> {
        self.beads3d_find_panel.get_find_beads3d_display()
    }

    /// Java package-private `getCcdEraserBeadsDisplay()`.
    pub fn get_ccd_eraser_beads_display(&self) -> Option<Rc<dyn CcdEraserDisplay>> {
        Some(self.ccd_eraser_beads_panel.clone())
    }

    /// Java package-private `done()`.
    pub fn done(&self) {
        self.xf_model_panel.done();
        self.beads3d_find_panel.done();
        self.ccd_eraser_beads_panel.done();
    }

    /// Java package-private `updateAlignedStackBinning()`.
    pub fn update_aligned_stack_binning(&self) {
        self.ccd_eraser_beads_panel.update_aligned_stack_binning();
    }

    /// Java private `updateDisplay()`.
    fn update_display(&self) {
        let model_use_fid_is_selected = self.rb_model_use_fid.is_selected();
        self.xf_model_panel.set_visible(model_use_fid_is_selected);
        self.beads3d_find_panel
            .set_visible(!model_use_fid_is_selected);
        let manager: &'static dyn BaseManager = self.manager;
        ui_harness::INSTANCE
            .with(|harness| harness.pack_axis_id_base_manager(Some(self.axis_id), Some(manager)));
    }

    /// Java package-private `getParameters(MetaData) throws
    /// FortranInputSyntaxException`.
    pub fn get_parameters_meta_data(
        &self,
        meta_data: &MetaData,
    ) -> Result<(), FortranInputSyntaxException> {
        self.ccd_eraser_beads_panel
            .get_parameters_meta_data(meta_data)?;
        meta_data.set_erase_gold_model_use_fid_boolean(
            self.axis_id,
            self.rb_model_use_fid.is_selected(),
        );
        self.beads3d_find_panel
            .get_parameters_meta_data(meta_data)?;
        Ok(())
    }

    /// Java package-private `setParameters(ReconScreenState)`.
    pub fn set_parameters_recon_screen_state(&self, screen_state: &ReconScreenState) {
        self.beads3d_find_panel
            .set_parameters_recon_screen_state(screen_state);
    }

    /// Java package-private `getParameters(ReconScreenState)`.
    pub fn get_parameters_recon_screen_state(&self, screen_state: &ReconScreenState) {
        self.beads3d_find_panel
            .get_parameters_recon_screen_state(screen_state);
    }

    /// Java package-private `setTiltState(TomogramState, ConstMetaData)`.
    pub fn set_tilt_state(&self, state: &TomogramState, meta_data: &dyn ConstMetaData) {
        self.beads3d_find_panel.set_tilt_state(state, meta_data);
    }

    /// Java package-private `setParameters(BlendmontParam)`.
    pub fn set_parameters_blendmont_param(&self, param: &BlendmontParam) {
        self.beads3d_find_panel
            .set_parameters_blendmont_param(param);
    }

    /// Java package-private `setParameters(NewstParam)`.
    pub fn set_parameters_newst_param(&self, param: &NewstParam) {
        self.beads3d_find_panel.set_parameters_newst_param(param);
    }

    /// Java package-private `setParameters(ConstTiltParam, boolean) throws
    /// FileNotFoundException, IOException`.
    pub fn set_parameters_const_tilt_param_boolean(
        &self,
        param: &dyn ConstTiltParam,
        initialize: bool,
    ) -> Result<(), std::io::Error> {
        self.beads3d_find_panel
            .set_parameters_const_tilt_param_boolean(param, initialize)
    }

    /// Java package-private `setParameters(ConstMetaData)`.
    pub fn set_parameters_const_meta_data(&self, meta_data: &dyn ConstMetaData) {
        self.ccd_eraser_beads_panel
            .set_parameters_const_meta_data(meta_data);
        if meta_data.get_erase_gold_model_use_fid(self.axis_id) {
            self.rb_model_use_fid.set_selected_boolean(true);
        } else {
            self.rb_model_use_find_beads3d.set_selected_boolean(true);
        }
        self.beads3d_find_panel
            .set_parameters_const_meta_data(meta_data);
        self.update_display();
    }

    /// Java package-private `setParameters(ConstFindBeads3dParam, boolean)`.
    pub fn set_parameters_const_find_beads3d_param_boolean(
        &self,
        param: &dyn ConstFindBeads3dParam,
        initialize: bool,
    ) {
        self.beads3d_find_panel
            .set_parameters_const_find_beads3d_param_boolean(param, initialize);
    }

    /// Java package-private `initialize()`.
    pub fn initialize(&self) {
        self.beads3d_find_panel.initialize();
    }

    /// Java package-private `setParameters(ConstTiltalignParam, boolean)`.
    pub fn set_parameters_const_tiltalign_param_boolean(
        &self,
        param: &ConstTiltalignParam,
        initialize: bool,
    ) {
        self.beads3d_find_panel
            .set_parameters_const_tiltalign_param_boolean(param, initialize);
    }

    /// Java package-private `setParameters(ConstCCDEraserParam)`.
    pub fn set_parameters_const_ccd_eraser_param(&self, param: &ConstCCDEraserParam) {
        self.ccd_eraser_beads_panel
            .set_parameters_const_ccd_eraser_param(param);
    }

    /// Java package-private `setOverrideParameters(ConstMetaData)`.
    pub fn set_override_parameters(&self, meta_data: &dyn ConstMetaData) {
        self.beads3d_find_panel.set_override_parameters(meta_data);
    }

    /// Java private `action(String, Deferred3dmodButton, Run3dmodMenuOptions)`.
    fn action(
        &self,
        command: &str,
        _deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        _run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    ) {
        if Some(command) == self.rb_model_use_fid.get_action_command().as_deref()
            || Some(command)
                == self
                    .rb_model_use_find_beads3d
                    .get_action_command()
                    .as_deref()
        {
            self.update_display();
        }
    }

    /// Java private `setToolTipText()`.
    fn set_tool_tip_text(&self) {
        self.rb_model_use_fid
            .set_tool_tip_text_string(Some("Erase the fiducials selected in the fiducial model."));
        self.rb_model_use_find_beads3d
            .set_tool_tip_text_string(Some("Find beads in tomogram and project positions."));
    }
}

impl ContextMenu for EraseGoldPanel {
    /// Java `popUpContextMenu(MouseEvent)`: right mouse button context menu.
    fn pop_up_context_menu(&self, mouse_event: &MouseEvent) {
        let align_manpage_label;
        let align_manpage;
        let align_logfile_label;
        let align_logfile;
        if self.manager.get_meta_data().get_view_type() == ViewType::Montage {
            align_manpage_label = "Blendmont";
            align_manpage = "blendmont";
            align_logfile_label = "Blend";
            align_logfile = "blend";
        } else {
            align_manpage_label = "Newstack";
            align_manpage = "newstack";
            align_logfile_label = "Newst";
            align_logfile = "newst";
        }
        let man_pagelabel: Vec<String> = vec![
            align_manpage_label.to_string(),
            "Tilt".to_string(),
            "Findbeads3d".to_string(),
            "CcdEraser".to_string(),
            "3dmod".to_string(),
        ];
        let man_page: Vec<String> = vec![
            format!("{align_manpage}.html"),
            "tilt.html".to_string(),
            "findbeads3d.html".to_string(),
            "ccderaser.html".to_string(),
            "3dmod.html".to_string(),
        ];
        let log_file_label: Vec<String> = vec![
            format!("{align_logfile_label}_3dfind"),
            "Tilt_3dfind".to_string(),
            "Findbeads3d".to_string(),
        ];
        let mut log_file: Vec<String> = vec![String::new(); 3];
        log_file[0] = format!("{align_logfile}_3dfind{}.log", self.axis_id.get_extension());
        log_file[1] = format!("tilt_3dfind{}.log", self.axis_id.get_extension());
        log_file[2] = format!("findbeads3d{}.log", self.axis_id.get_extension());
        let manager: &'static dyn BaseManager = self.manager;
        let _context_popup = ContextPopup::new_component_mouse_event_string_string_string_array_string_array_string_array_string_array_base_manager_axis_id(
            &self.pnl_root,
            mouse_event,
            Some("ErasingGold"),
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
