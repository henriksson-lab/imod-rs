//! `IMOD/Etomo/src/etomo/ui/swing/EraseGoldPanel.java`.
//!
//! Swing construction, the concrete `ApplicationManager`, and the three
//! subordinate panels are direct boundaries.  This unit retains the source
//! panel's controls, ordering, selection/visibility invariant, parameter
//! routing, and context-menu construction without inventing a second process
//! controller.
#![allow(dead_code)]

use std::cell::RefCell;
use std::rc::Rc;

use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::view_type::ViewType;

use super::context_popup::{ContextPopup, MouseEvent, TOMO_GUIDE};
use super::newstack_or_blendmont_panel::{
    BlendmontParam, GlobalExpandButton, NewstParam, ReconScreenState,
};
use super::radio_button::{RadioButton, RadioButtonGroup};
use super::tilt_panel::Deferred3dmodButton;

pub const ERASE_GOLD_TAB_LABEL: &str = "Erase Gold";

/// Boundary data shared by the Java `ConstMetaData` and `MetaData` calls made
/// by this source unit.
#[derive(Clone, Debug, Default)]
pub struct EraseGoldMetaData {
    pub erase_gold_model_use_fid: bool,
}

/// Opaque source parameter boundaries not otherwise owned by this panel.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TiltParam;
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct FindBeads3dParam;
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TiltalignParam;
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CCDEraserParam;
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TomogramState;

/// Display interfaces returned by Java subordinate panels.  They remain
/// explicit boundaries until their owning display classes are translated.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct BlendmontDisplay;
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct NewstackDisplay;
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TiltDisplay;
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct FindBeads3dDisplay;

/// The direct `ApplicationManager` methods called from `EraseGoldPanel.java`.
pub trait EraseGoldPanelApplicationManager {
    fn view_type(&self) -> ViewType;
    fn pack(&mut self, axis_id: AxisID);
}

/// Java `FinalAlignedStackDialog.isFiducialess` boundary.
pub trait FinalAlignedStackDialog {
    fn is_fiducialess(&self) -> bool;
}

/// Source-visible state owned by Java `XfModelPanel` at this panel boundary.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct XfModelPanel {
    pub visible: bool,
    pub done_count: u32,
}

impl XfModelPanel {
    pub fn set_visible(&mut self, visible: bool) {
        self.visible = visible;
    }
    pub fn done(&mut self) {
        self.done_count += 1;
    }
}

/// Source-visible state owned by Java `Beads3dFindPanel` at this panel boundary.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Beads3dFindPanel {
    pub visible: bool,
    pub advanced: bool,
    pub initialized: bool,
    pub done_count: u32,
    pub reregister_processing_method_mediator_count: u32,
    pub parameter_set_count: u32,
    pub parameter_get_count: u32,
    pub override_parameter_count: u32,
    pub blendmont_3d_find_display: BlendmontDisplay,
    pub newstack_3d_find_display: NewstackDisplay,
    pub tilt_3d_find_display: TiltDisplay,
    pub find_beads_3d_display: FindBeads3dDisplay,
}

impl Beads3dFindPanel {
    pub fn reregister_processing_method_mediator(&mut self) {
        self.reregister_processing_method_mediator_count += 1;
    }
    pub fn set_visible(&mut self, visible: bool) {
        self.visible = visible;
    }
    pub fn update_advanced(&mut self, advanced: bool) {
        self.advanced = advanced;
    }
    pub fn done(&mut self) {
        self.done_count += 1;
    }
    pub fn initialize(&mut self) {
        self.initialized = true;
    }
    pub fn get_blendmont_3d_find_display(&self) -> &BlendmontDisplay {
        &self.blendmont_3d_find_display
    }
    pub fn get_newstack_3d_find_display(&self) -> &NewstackDisplay {
        &self.newstack_3d_find_display
    }
    pub fn get_tilt_3d_find_display(&self) -> &TiltDisplay {
        &self.tilt_3d_find_display
    }
    pub fn get_find_beads_3d_display(&self) -> &FindBeads3dDisplay {
        &self.find_beads_3d_display
    }
}

/// Source-visible state owned by Java `CcdEraserBeadsPanel` at this panel boundary.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CcdEraserBeadsPanel {
    pub initialized: bool,
    pub done_count: u32,
    pub aligned_stack_binning_update_count: u32,
    pub parameter_set_count: u32,
    pub parameter_get_count: u32,
}

impl CcdEraserBeadsPanel {
    pub fn initialize(&mut self) {
        self.initialized = true;
    }
    pub fn done(&mut self) {
        self.done_count += 1;
    }
    pub fn update_aligned_stack_binning(&mut self) {
        self.aligned_stack_binning_update_count += 1;
    }
}

/// Java `EraseGoldPanel` source-visible Swing layout state.
#[derive(Clone, Debug, Default)]
pub struct EraseGoldPanelLayout {
    pub root_box_layout_y_axis: bool,
    pub root_border: Option<String>,
    pub root_component_order: Vec<String>,
    pub model_box_layout_y_axis: bool,
    pub model_border: Option<String>,
    pub model_alignment_x: f32,
    pub model_component_order: Vec<String>,
    pub root_mouse_listener_count: usize,
    pub context_popup: Option<ContextPopup>,
}

/// Java final `EraseGoldPanel`.
#[derive(Clone, Debug)]
pub struct EraseGoldPanel {
    pub pnl_root: EraseGoldPanelLayout,
    pub bg_model: Rc<RefCell<RadioButtonGroup>>,
    pub rb_model_use_fid: RadioButton,
    pub rb_model_use_find_beads_3d: RadioButton,
    pub xf_model_panel: XfModelPanel,
    pub beads_3d_find_panel: Beads3dFindPanel,
    pub ccd_eraser_beads_panel: CcdEraserBeadsPanel,
    pub axis_id: AxisID,
    pub dialog_type: DialogType,
}

impl EraseGoldPanel {
    /// Java private constructor `EraseGoldPanel(...)`.
    pub fn new(
        axis_id: AxisID,
        dialog_type: DialogType,
        _global_advanced_button: &GlobalExpandButton,
    ) -> Self {
        let bg_model = Rc::new(RefCell::new(RadioButtonGroup::new()));
        let mut rb_model_use_fid =
            RadioButton::new_in_group("Use the existing fiducial model", bg_model.clone());
        rb_model_use_fid.radio_button.action_command =
            Some("Use the existing fiducial model".into());
        let mut rb_model_use_find_beads_3d =
            RadioButton::new_in_group("Use findbeads3d", bg_model.clone());
        rb_model_use_find_beads_3d.radio_button.action_command = Some("Use findbeads3d".into());
        Self {
            pnl_root: EraseGoldPanelLayout::default(),
            bg_model: bg_model.clone(),
            rb_model_use_fid,
            rb_model_use_find_beads_3d,
            xf_model_panel: XfModelPanel::default(),
            beads_3d_find_panel: Beads3dFindPanel::default(),
            ccd_eraser_beads_panel: CcdEraserBeadsPanel::default(),
            axis_id,
            dialog_type,
        }
    }

    /// Java static `getInstance`.
    pub fn get_instance(
        axis_id: AxisID,
        dialog_type: DialogType,
        global_advanced_button: &GlobalExpandButton,
    ) -> Self {
        let mut instance = Self::new(axis_id, dialog_type, global_advanced_button);
        instance.create_panel();
        instance.add_listeners();
        instance.set_tool_tip_text();
        instance
    }

    /// Java `addListeners`.
    pub fn add_listeners(&mut self) {
        self.pnl_root.root_mouse_listener_count += 1;
        self.rb_model_use_fid.add_action_listener();
        self.rb_model_use_find_beads_3d.add_action_listener();
    }

    /// Java `reregisterProcessingMethodMediator`.
    pub fn reregister_processing_method_mediator(&mut self) {
        self.beads_3d_find_panel
            .reregister_processing_method_mediator();
    }

    /// Java `popUpContextMenu`.
    pub fn pop_up_context_menu<M: EraseGoldPanelApplicationManager>(
        &mut self,
        manager: &M,
        mouse_event: MouseEvent,
    ) {
        let (align_manpage_label, align_manpage, align_logfile_label, align_logfile) =
            if manager.view_type() == ViewType::Montage {
                ("Blendmont", "blendmont", "Blend", "blend")
            } else {
                ("Newstack", "newstack", "Newst", "newst")
            };
        let man_page_label = vec![
            align_manpage_label.into(),
            "Tilt".into(),
            "Findbeads3d".into(),
            "CcdEraser".into(),
            "3dmod".into(),
        ];
        let man_page = vec![
            format!("{align_manpage}.html"),
            "tilt.html".into(),
            "findbeads3d.html".into(),
            "ccderaser.html".into(),
            "3dmod.html".into(),
        ];
        let log_file_label = vec![
            format!("{align_logfile_label}_3dfind"),
            "Tilt_3dfind".into(),
            "Findbeads3d".into(),
        ];
        let extension = self.axis_id.get_extension();
        let log_file = vec![
            format!("{align_logfile}_3dfind{extension}.log"),
            format!("tilt_3dfind{extension}.log"),
            format!("findbeads3d{extension}.log"),
        ];
        self.pnl_root.context_popup = ContextPopup::new_log_files(
            mouse_event,
            Some("ErasingGold"),
            TOMO_GUIDE,
            &man_page_label,
            &man_page,
            &log_file_label,
            &log_file,
            self.axis_id,
            None,
        )
        .ok();
    }

    /// Java `createPanel`.
    pub fn create_panel(&mut self) {
        self.pnl_root.root_box_layout_y_axis = true;
        self.pnl_root.root_border = Some("Bead Eraser".into());
        self.pnl_root.root_component_order = vec![
            "pnlModel".into(),
            "xfModelPanel".into(),
            "beads3dFindPanel".into(),
            "ccdEraserBeadsPanel".into(),
        ];
        self.pnl_root.model_box_layout_y_axis = true;
        self.pnl_root.model_border = Some("Model Creation Method".into());
        self.pnl_root.model_alignment_x = 0.5;
        self.pnl_root.model_component_order =
            vec!["rbModelUseFid".into(), "rbModelUseFindBeads3d".into()];
    }

    /// Java `getComponent`, including its required binning update before return.
    pub fn get_component(&mut self) -> &EraseGoldPanelLayout {
        self.update_aligned_stack_binning();
        &self.pnl_root
    }
    /// Java `isFiducialess`.
    pub fn is_fiducialess<P: FinalAlignedStackDialog>(&self, parent: &P) -> bool {
        parent.is_fiducialess()
    }
    /// Java `initializeBeads`.
    pub fn initialize_beads(&mut self) {
        self.ccd_eraser_beads_panel.initialize();
    }
    /// Java `updateAdvanced`.
    pub fn update_advanced(&mut self, advanced: bool) {
        self.beads_3d_find_panel.update_advanced(advanced);
    }
    /// Java `getBlendmont3dFindDisplay`.
    pub fn get_blendmont_3d_find_display(&self) -> &BlendmontDisplay {
        self.beads_3d_find_panel.get_blendmont_3d_find_display()
    }
    /// Java `getNewstack3dFindDisplay`.
    pub fn get_newstack_3d_find_display(&self) -> &NewstackDisplay {
        self.beads_3d_find_panel.get_newstack_3d_find_display()
    }
    /// Java `getTilt3dFindDisplay`.
    pub fn get_tilt_3d_find_display(&self) -> &TiltDisplay {
        self.beads_3d_find_panel.get_tilt_3d_find_display()
    }
    /// Java `getFindBeads3dDisplay`.
    pub fn get_find_beads_3d_display(&self) -> &FindBeads3dDisplay {
        self.beads_3d_find_panel.get_find_beads_3d_display()
    }
    /// Java `getCcdEraserBeadsDisplay`.
    pub fn get_ccd_eraser_beads_display(&self) -> &CcdEraserBeadsPanel {
        &self.ccd_eraser_beads_panel
    }
    /// Java `done`.
    pub fn done(&mut self) {
        self.xf_model_panel.done();
        self.beads_3d_find_panel.done();
        self.ccd_eraser_beads_panel.done();
    }
    /// Java `updateAlignedStackBinning`.
    pub fn update_aligned_stack_binning(&mut self) {
        self.ccd_eraser_beads_panel.update_aligned_stack_binning();
    }

    /// Java `updateDisplay`.
    pub fn update_display<M: EraseGoldPanelApplicationManager>(&mut self, manager: &mut M) {
        let model_use_fid_is_selected = self.rb_model_use_fid.is_selected();
        self.xf_model_panel.set_visible(model_use_fid_is_selected);
        self.beads_3d_find_panel
            .set_visible(!model_use_fid_is_selected);
        manager.pack(self.axis_id);
    }

    /// Java `getParameters(MetaData)`; validation is a direct subordinate-panel boundary.
    pub fn get_parameters_meta_data(&mut self, meta_data: &mut EraseGoldMetaData) {
        self.ccd_eraser_beads_panel.parameter_get_count += 1;
        meta_data.erase_gold_model_use_fid = self.rb_model_use_fid.is_selected();
        self.beads_3d_find_panel.parameter_get_count += 1;
    }
    /// Java `setParameters(ReconScreenState)`.
    pub fn set_parameters_recon_screen_state(&mut self, _screen_state: &ReconScreenState) {
        self.beads_3d_find_panel.parameter_set_count += 1;
    }
    /// Java `getParameters(ReconScreenState)`.
    pub fn get_parameters_recon_screen_state(&mut self, _screen_state: &mut ReconScreenState) {
        self.beads_3d_find_panel.parameter_get_count += 1;
    }
    /// Java `setTiltState`.
    pub fn set_tilt_state(&mut self, _state: &TomogramState, _meta_data: &EraseGoldMetaData) {
        self.beads_3d_find_panel.parameter_set_count += 1;
    }
    /// Java `setParameters(BlendmontParam)`.
    pub fn set_parameters_blendmont(&mut self, _param: &BlendmontParam) {
        self.beads_3d_find_panel.parameter_set_count += 1;
    }
    /// Java `setParameters(NewstParam)`.
    pub fn set_parameters_newst(&mut self, _param: &NewstParam) {
        self.beads_3d_find_panel.parameter_set_count += 1;
    }
    /// Java `setParameters(ConstTiltParam, boolean)`.
    pub fn set_parameters_tilt(&mut self, _param: &TiltParam, _initialize: bool) {
        self.beads_3d_find_panel.parameter_set_count += 1;
    }
    /// Java `setParameters(ConstMetaData)`.
    pub fn set_parameters_meta_data<M: EraseGoldPanelApplicationManager>(
        &mut self,
        manager: &mut M,
        meta_data: &EraseGoldMetaData,
    ) {
        self.ccd_eraser_beads_panel.parameter_set_count += 1;
        if meta_data.erase_gold_model_use_fid {
            self.rb_model_use_fid.set_selected(true);
        } else {
            self.rb_model_use_find_beads_3d.set_selected(true);
        }
        self.beads_3d_find_panel.parameter_set_count += 1;
        self.update_display(manager);
    }
    /// Java `setParameters(ConstFindBeads3dParam, boolean)`.
    pub fn set_parameters_find_beads_3d(&mut self, _param: &FindBeads3dParam, _initialize: bool) {
        self.beads_3d_find_panel.parameter_set_count += 1;
    }
    /// Java `initialize`.
    pub fn initialize(&mut self) {
        self.beads_3d_find_panel.initialize();
    }
    /// Java `setParameters(ConstTiltalignParam, boolean)`.
    pub fn set_parameters_tiltalign(&mut self, _param: &TiltalignParam, _initialize: bool) {
        self.beads_3d_find_panel.parameter_set_count += 1;
    }
    /// Java `setParameters(ConstCCDEraserParam)`.
    pub fn set_parameters_ccd_eraser(&mut self, _param: &CCDEraserParam) {
        self.ccd_eraser_beads_panel.parameter_set_count += 1;
    }
    /// Java `setOverrideParameters`.
    pub fn set_override_parameters(&mut self, _meta_data: &EraseGoldMetaData) {
        self.beads_3d_find_panel.override_parameter_count += 1;
    }

    /// Java `action(String, Deferred3dmodButton, Run3dmodMenuOptions)`.
    pub fn action<M: EraseGoldPanelApplicationManager>(
        &mut self,
        manager: &mut M,
        command: &str,
        _deferred_3dmod_button: Option<&Deferred3dmodButton>,
        _run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    ) {
        if Some(command) == self.rb_model_use_fid.radio_button.action_command.as_deref()
            || Some(command)
                == self
                    .rb_model_use_find_beads_3d
                    .radio_button
                    .action_command
                    .as_deref()
        {
            self.update_display(manager);
        }
    }
    #[allow(non_snake_case)]
    /// Native-name adapter for the Java listener's `actionPerformed`.
    pub fn actionPerformed<M: EraseGoldPanelApplicationManager>(
        &mut self,
        manager: &mut M,
        command: &str,
    ) {
        self.action(manager, command, None, None);
    }
    /// Java `setToolTipText`.
    pub fn set_tool_tip_text(&mut self) {
        self.rb_model_use_fid
            .set_tool_tip_text(Some("Erase the fiducials selected in the fiducial model."));
        self.rb_model_use_find_beads_3d
            .set_tool_tip_text(Some("Find beads in tomogram and project positions."));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct Manager {
        view_type: Option<ViewType>,
        packed: Vec<AxisID>,
    }
    impl EraseGoldPanelApplicationManager for Manager {
        fn view_type(&self) -> ViewType {
            self.view_type.unwrap_or(ViewType::SingleView)
        }
        fn pack(&mut self, axis_id: AxisID) {
            self.packed.push(axis_id);
        }
    }
    #[test]
    fn source_creation_and_selection_visibility_are_preserved() {
        let mut panel = EraseGoldPanel::get_instance(
            AxisID::Second,
            DialogType::FinalAlignedStack,
            &GlobalExpandButton::get_instance("Advanced", "Basic"),
        );
        assert_eq!(
            panel.pnl_root.root_component_order,
            [
                "pnlModel",
                "xfModelPanel",
                "beads3dFindPanel",
                "ccdEraserBeadsPanel"
            ]
        );
        assert_eq!(
            panel.rb_model_use_fid.get_tooltip(),
            Some("<html>Erase the fiducials selected in the fiducial model.")
        );
        let mut manager = Manager::default();
        panel.set_parameters_meta_data(
            &mut manager,
            &EraseGoldMetaData {
                erase_gold_model_use_fid: true,
            },
        );
        assert!(panel.xf_model_panel.visible);
        assert!(!panel.beads_3d_find_panel.visible);
        assert_eq!(manager.packed, vec![AxisID::Second]);
    }
    #[test]
    fn context_menu_uses_source_montage_names_and_axis_extension() {
        let mut panel = EraseGoldPanel::get_instance(
            AxisID::First,
            DialogType::FinalAlignedStack,
            &GlobalExpandButton::get_instance("Advanced", "Basic"),
        );
        let manager = Manager {
            view_type: Some(ViewType::Montage),
            ..Default::default()
        };
        panel.pop_up_context_menu(&manager, MouseEvent::default());
        let popup = panel.pnl_root.context_popup.as_ref().unwrap();
        assert_eq!(
            popup.man_page_name.as_ref().unwrap()[0],
            "blendmont.html#TOP"
        );
        assert_eq!(
            popup.log_file_name.as_ref().unwrap()[0],
            "blend_3dfinda.log"
        );
    }
}
