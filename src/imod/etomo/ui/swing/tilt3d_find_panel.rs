//! `IMOD/Etomo/src/etomo/ui/swing/Tilt3dFindPanel.java`.
//!
//! This source unit keeps the `ApplicationManager`, `Tilt3dFindParent`,
//! taAngles-log, metadata, and com-script parameter calls at their original
//! ownership boundaries.  It owns the Swing field state, panel ordering, and
//! exact action routing inherited from `AbstractTiltPanel`.
#![allow(dead_code)]

use super::abstract_tilt_panel::{AbstractTiltPanel, AbstractTiltPanelState, TiltTextField};
use super::beads3d_find_panel::{
    Deferred3dmodButton, ProcessResultDisplay, Tilt3dFindPanel as BeadsTilt3dFindPanel,
    TiltParam as BeadsTiltParam, TiltalignParam as BeadsTiltalignParam, TomogramState,
};
use super::newstack_or_blendmont_panel::MetaData as BeadsMetaData;
use super::tilt3d_find_parent::Tilt3dFindParent;
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::processing_method::ProcessingMethod;

/// Java `CENTER_TO_CENTER_THICKNESS_LABEL`.
pub const CENTER_TO_CENTER_THICKNESS_LABEL: &str = "Center to center thickness";
/// Java `ADDITION_UNBINNED_DIAMETERS_TO_ADD`.
pub const ADDITION_UNBINNED_DIAMETERS_TO_ADD: &str = "Additional unbinned diameters to add ";
/// Java `TILT_3D_FIND_LABEL`.
pub const TILT_3D_FIND_LABEL: &str = "Align and Build Tomogram";
/// Java `PANEL_ID`.
pub const PANEL_ID: &str = "Tilt3dFind";

/// Java `TiltParam.Mode.TILT_3D_FIND`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Tilt3dFindMode {
    Tilt3dFind,
}

/// Direct writes to Java `TiltParam` in `getParameters(TiltParam, boolean)`.
pub trait Tilt3dFindTiltParam {
    fn set_input_file(&mut self, file_name: String);
    fn set_output_file(&mut self, file_name: String);
    fn set_command_mode(&mut self, mode: Tilt3dFindMode);
    fn set_process_name(&mut self, process_name: ProcessName);
    fn abstract_tilt_param(&mut self) -> &mut super::abstract_tilt_panel::TiltParamState;
}

/// Java `ConstTiltParam` read by `setParameters`.
pub trait ConstTilt3dFindTiltParam {
    fn abstract_tilt_param(&self) -> &super::abstract_tilt_panel::TiltParamState;
}

/// Direct Java `SplittiltParam` writes from this source unit.
pub trait Tilt3dFindSplittiltParam {
    fn set_name(&mut self, name: String);
}

/// Java `ConstTiltalignParam` is deliberately a boundary: only the align log
/// lookup initiated by this unit supplies the displayed values.
pub trait ConstTilt3dFindTiltalignParam {}

/// Java `TaAnglesLog` values used by `setParameters(ConstTiltalignParam, boolean)`.
pub trait Tilt3dFindTaAnglesLog {
    fn get_center_to_center_thickness(&self) -> Result<Option<f64>, String>;
    fn get_incremental_shift_to_center(&self) -> Result<Option<String>, String>;
}

/// Java `ConstMetaData` call made by `setOverrideParameters`.
pub trait Tilt3dFindMetaData {
    fn is_stack_3d_find_thickness_set(&self, axis_id: AxisID) -> bool;
    fn stack_3d_find_thickness(&self, axis_id: AxisID) -> String;
    fn set_tilt_parallel(&mut self, axis_id: AxisID, panel_id: &str, parallel: bool);
}

/// Direct `ApplicationManager` calls from this Java source unit.
pub trait Tilt3dFindApplicationManager {
    type TaAnglesLog: Tilt3dFindTaAnglesLog;

    fn stack_using_newst_or_blend_3d_find_output(&self, axis_id: AxisID) -> bool;
    fn newst_or_blend_3d_find_output_file_name(&self, axis_id: AxisID) -> String;
    fn aligned_stack_file_name(&self, axis_id: AxisID) -> String;
    fn tilt_3d_find_output_file_name(&self, axis_id: AxisID) -> String;
    fn ta_angles_log(&self, axis_id: AxisID) -> Self::TaAnglesLog;
    fn calc_unbinned_bead_diameter_pixels(&self) -> f64;
    fn tilt_3d_find_action(
        &mut self,
        process_result_display: &ProcessResultDisplay,
        deferred_3dmod_button: Option<&Deferred3dmodButton>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
        axis_id: AxisID,
        dialog_type: DialogType,
        processing_method: ProcessingMethod,
    );
    fn imod_tilt_3d_find_output(
        &mut self,
        axis_id: AxisID,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    );
}

/// Source-visible `SpacedPanel` and `JPanel` hierarchy made by `createPanel`.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Tilt3dFindPanelLayout {
    pub root_box_layout_y_axis: bool,
    pub root_component_order: Vec<String>,
    pub panel_a_box_layout_x_axis: bool,
    pub panel_a_component_order: Vec<String>,
    pub buttons_component_order: Vec<String>,
}

/// Java final `Tilt3dFindPanel`, composed with Java superclass state.
pub struct Tilt3dFindPanel<M, P> {
    pub abstract_tilt_panel: AbstractTiltPanel,
    pub ltf_center_to_center_thickness: TiltTextField,
    pub ltf_additional_diameters: TiltTextField,
    pub parent: P,
    pub extra_button_present: bool,
    pub manager: M,
    pub layout: Tilt3dFindPanelLayout,
}

impl<M: Tilt3dFindApplicationManager, P: Tilt3dFindParent> Tilt3dFindPanel<M, P> {
    /// Java private constructor.
    pub fn new(
        manager: M,
        axis_id: AxisID,
        dialog_type: DialogType,
        parent: P,
        extra_button_present: bool,
    ) -> Self {
        let mut abstract_tilt_panel = AbstractTiltPanel::new(axis_id, dialog_type, PANEL_ID, false);
        abstract_tilt_panel.ltf_z_shift.label = "Added Z Shift: ".into();
        abstract_tilt_panel.ltf_tomo_thickness.label = "Thickness: ".into();
        Self {
            abstract_tilt_panel,
            ltf_center_to_center_thickness: TiltTextField {
                label: format!("{CENTER_TO_CENTER_THICKNESS_LABEL}: "),
                visible: true,
                enabled: false,
                ..Default::default()
            },
            ltf_additional_diameters: TiltTextField {
                label: format!("{ADDITION_UNBINNED_DIAMETERS_TO_ADD}: "),
                visible: true,
                enabled: false,
                ..Default::default()
            },
            parent,
            extra_button_present,
            manager,
            layout: Tilt3dFindPanelLayout::default(),
        }
    }

    /// Java static `getInstance` and its creation order.
    pub fn get_instance(
        manager: M,
        axis_id: AxisID,
        dialog_type: DialogType,
        parent: P,
        extra_button_present: bool,
    ) -> Self {
        let mut instance = Self::new(manager, axis_id, dialog_type, parent, extra_button_present);
        instance.create_panel();
        instance.set_tool_tip_text();
        instance.add_listeners();
        instance
    }

    /// Java override `createPanel`.
    pub fn create_panel(&mut self) {
        self.abstract_tilt_panel.initialize_panel();
        self.ltf_center_to_center_thickness.enabled = false;
        self.ltf_additional_diameters.enabled = false;
        self.layout.root_box_layout_y_axis = true;
        self.layout.root_component_order = vec![
            "cpuGpuPanel".into(),
            "ltfCenterToCenterThickness".into(),
            "ltfAdditionalDiameters".into(),
            "pnlA".into(),
            "pnlButtons".into(),
        ];
        self.layout.panel_a_box_layout_x_axis = true;
        self.layout.panel_a_component_order = vec!["ltfTomoThickness".into(), "ltfZShift".into()];
        self.layout.buttons_component_order = vec!["tiltButton".into()];
        if self.extra_button_present {
            self.layout
                .buttons_component_order
                .push("extraButton".into());
        }
        self.layout
            .buttons_component_order
            .push("3dmodTomogramButton".into());
    }

    /// Java override `getParameters(TiltParam, boolean)`.
    pub fn get_parameters_tilt<Pm: Tilt3dFindTiltParam>(
        &mut self,
        param: &mut Pm,
        do_validation: bool,
        state: &mut AbstractTiltPanelState,
    ) -> Result<bool, String> {
        let axis_id = self.abstract_tilt_panel.axis_id;
        if self
            .manager
            .stack_using_newst_or_blend_3d_find_output(axis_id)
        {
            param.set_input_file(
                self.manager
                    .newst_or_blend_3d_find_output_file_name(axis_id),
            );
        } else {
            param.set_input_file(self.manager.aligned_stack_file_name(axis_id));
        }
        param.set_output_file(self.manager.tilt_3d_find_output_file_name(axis_id));
        param.set_command_mode(Tilt3dFindMode::Tilt3dFind);
        param.set_process_name(ProcessName::TILT_3D_FIND);
        self.abstract_tilt_panel.get_parameters_tilt_param(
            param.abstract_tilt_param(),
            do_validation,
            state,
        )
    }

    /// Java override `setParameters(ConstTiltParam, boolean)`.
    pub fn set_parameters_tilt<Pm: ConstTilt3dFindTiltParam>(
        &mut self,
        param: &Pm,
        initialize: bool,
    ) {
        self.abstract_tilt_panel
            .set_parameters_tilt_param(param.abstract_tilt_param(), initialize);
    }

    /// Java `setParameters(ConstTiltalignParam, boolean)`.
    pub fn set_parameters_tiltalign<Pm: ConstTilt3dFindTiltalignParam>(
        &mut self,
        _param: &Pm,
        initialize: bool,
    ) {
        let log = self.manager.ta_angles_log(self.abstract_tilt_panel.axis_id);
        let center_to_center_thickness = log.get_center_to_center_thickness().ok().flatten();
        if let Some(value) = center_to_center_thickness {
            self.ltf_center_to_center_thickness.text = value.to_string();
        }
        let additional_diameters = 5;
        self.ltf_additional_diameters.text = additional_diameters.to_string();
        if initialize {
            if let Some(value) = center_to_center_thickness {
                self.abstract_tilt_panel.ltf_tomo_thickness.text = (value
                    + self.manager.calc_unbinned_bead_diameter_pixels()
                        * f64::from(additional_diameters))
                .round()
                .to_string();
            }
            if let Ok(Some(value)) = log.get_incremental_shift_to_center() {
                self.abstract_tilt_panel.ltf_z_shift.text = value;
            }
        }
    }

    /// Java `setOverrideParameters(ConstMetaData)`.
    pub fn set_override_parameters<MD: Tilt3dFindMetaData>(&mut self, meta_data: &MD) {
        if meta_data.is_stack_3d_find_thickness_set(self.abstract_tilt_panel.axis_id) {
            self.abstract_tilt_panel.ltf_tomo_thickness.text =
                meta_data.stack_3d_find_thickness(self.abstract_tilt_panel.axis_id);
        }
    }

    /// Java override `getParameters(MetaData)`.
    pub fn get_parameters_meta_data<MD: Tilt3dFindMetaData>(&self, meta_data: &mut MD) {
        meta_data.set_tilt_parallel(
            self.abstract_tilt_panel.axis_id,
            PANEL_ID,
            self.abstract_tilt_panel.is_parallel_process(),
        );
    }

    /// Java override `getParameters(SplittiltParam, boolean)`.
    pub fn get_parameters_splittilt<Pm: Tilt3dFindSplittiltParam>(
        &self,
        param: &mut Pm,
        _do_validation: bool,
        cpus_selected: Result<i32, String>,
    ) -> bool {
        if !self
            .abstract_tilt_panel
            .get_parameters_splittilt(cpus_selected)
        {
            return false;
        }
        param.set_name("tilt_3dfind".into());
        true
    }

    /// Java `tilt3dFindAction(ProcessResultDisplay, Deferred3dmodButton, Run3dmodMenuOptions)`.
    pub fn tilt_3d_find_action(
        &mut self,
        process_result_display: &ProcessResultDisplay,
        deferred_3dmod_button: Option<&Deferred3dmodButton>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    ) {
        self.manager.tilt_3d_find_action(
            process_result_display,
            deferred_3dmod_button,
            run_3dmod_menu_options,
            self.abstract_tilt_panel.axis_id,
            self.abstract_tilt_panel.dialog_type,
            self.abstract_tilt_panel
                .get_run_method_for_process_interface(),
        );
    }

    /// Java override `tiltAction`.
    pub fn tilt_action(
        &mut self,
        process_result_display: &ProcessResultDisplay,
        deferred_3dmod_button: Option<&Deferred3dmodButton>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
        tilt_processing_method: ProcessingMethod,
    ) {
        self.parent.tilt_3d_find_action(
            process_result_display,
            deferred_3dmod_button,
            run_3dmod_menu_options,
            tilt_processing_method,
        );
    }

    /// Java override `imodTomogramAction`.
    pub fn imod_tomogram_action(
        &mut self,
        _deferred_3dmod_button: Option<&Deferred3dmodButton>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    ) {
        self.manager
            .imod_tilt_3d_find_output(self.abstract_tilt_panel.axis_id, run_3dmod_menu_options);
    }

    /// Java override `setToolTipText`.
    pub fn set_tool_tip_text(&mut self) {
        self.abstract_tilt_panel.set_tool_tip_text();
        self.ltf_center_to_center_thickness.tooltip = Some("Used to calculate the thickness of the findbeads3d input tomogram.  From the taAngles log.".into());
        self.ltf_additional_diameters.tooltip =
            Some("Used to calculate the thickness of the findbeads3d.".into());
        self.abstract_tilt_panel.ltf_tomo_thickness.tooltip = Some(format!(
            "Thickness of tomogram in unbinned pixels.  The default is calculated from \"{CENTER_TO_CENTER_THICKNESS_LABEL}\" plus \"Bead size\" multipled by \"{ADDITION_UNBINNED_DIAMETERS_TO_ADD}\"."
        ));
        self.abstract_tilt_panel.ltf_z_shift.tooltip = Some("Incremental unbinned shift needed to center range of fiducials in Z.  From the taAngles log.".into());
        self.abstract_tilt_panel.set_tilt_button_tooltip("If binning has changed, create a separate full aligned stack.  Then compute a tomogram (tilt_3dfind.com).");
    }

    /// Java inherited `addListeners`.
    pub fn add_listeners(&mut self) {
        self.abstract_tilt_panel.add_listeners();
    }
}

impl<M: Tilt3dFindApplicationManager, P: Tilt3dFindParent> BeadsTilt3dFindPanel
    for Tilt3dFindPanel<M, P>
{
    fn reregister_processing_method_mediator(&mut self) {
        self.abstract_tilt_panel
            .reregister_processing_method_mediator();
    }
    fn get_processing_method(&self) -> ProcessingMethod {
        self.abstract_tilt_panel.get_processing_method()
    }
    fn done(&mut self) {
        self.abstract_tilt_panel.done();
    }
    fn set_state(&mut self, _state: &TomogramState, _meta_data: &BeadsMetaData) {
        self.abstract_tilt_panel.update_display();
    }
    fn set_parameters_tilt(&mut self, _param: &BeadsTiltParam, _initialize: bool) {}
    fn set_parameters_tiltalign(&mut self, _param: &BeadsTiltalignParam, initialize: bool) {
        self.set_parameters_tiltalign(_param, initialize);
    }
    fn set_override_parameters(&mut self, _meta_data: &BeadsMetaData) {}
    fn get_parameters(&mut self, _meta_data: &mut BeadsMetaData) {}
    fn set_parameters_meta_data(&mut self, _meta_data: &BeadsMetaData) {}
    fn tilt_3d_find_action(
        &mut self,
        display: &ProcessResultDisplay,
        deferred: Option<&Deferred3dmodButton>,
        options: Option<Run3dmodMenuOptions>,
    ) {
        self.tilt_3d_find_action(display, deferred, options);
    }
}

impl ConstTilt3dFindTiltalignParam for BeadsTiltalignParam {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::etomo::r#type::dialog_type::DialogType;
    use crate::imod::etomo::ui::swing::tomogram_generation_parent::TomogramGenerationParent;

    #[derive(Default)]
    struct Log;
    impl Tilt3dFindTaAnglesLog for Log {
        fn get_center_to_center_thickness(&self) -> Result<Option<f64>, String> {
            Ok(Some(100.4))
        }
        fn get_incremental_shift_to_center(&self) -> Result<Option<String>, String> {
            Ok(Some("-3.5".into()))
        }
    }
    #[derive(Default)]
    struct Manager {
        use_full: bool,
        launch: u32,
        view: u32,
    }
    impl Tilt3dFindApplicationManager for Manager {
        type TaAnglesLog = Log;
        fn stack_using_newst_or_blend_3d_find_output(&self, _: AxisID) -> bool {
            self.use_full
        }
        fn newst_or_blend_3d_find_output_file_name(&self, _: AxisID) -> String {
            "full.ali".into()
        }
        fn aligned_stack_file_name(&self, _: AxisID) -> String {
            "aligned.ali".into()
        }
        fn tilt_3d_find_output_file_name(&self, _: AxisID) -> String {
            "find.rec".into()
        }
        fn ta_angles_log(&self, _: AxisID) -> Log {
            Log
        }
        fn calc_unbinned_bead_diameter_pixels(&self) -> f64 {
            10.0
        }
        fn tilt_3d_find_action(
            &mut self,
            _: &ProcessResultDisplay,
            _: Option<&Deferred3dmodButton>,
            _: Option<Run3dmodMenuOptions>,
            _: AxisID,
            _: DialogType,
            _: ProcessingMethod,
        ) {
            self.launch += 1
        }
        fn imod_tilt_3d_find_output(&mut self, _: AxisID, _: Option<Run3dmodMenuOptions>) {
            self.view += 1
        }
    }
    #[derive(Default)]
    struct Parent {
        calls: u32,
    }
    impl TomogramGenerationParent for Parent {
        fn is_ctf3d(&self) -> bool {
            false
        }
        fn is_method_plugin(&self) -> bool {
            false
        }
        fn is_multifilt(&self) -> bool {
            false
        }
        fn is_back_projection(&self) -> bool {
            true
        }
        fn is_sirt(&self) -> bool {
            false
        }
    }
    impl Tilt3dFindParent for Parent {
        fn tilt_3d_find_action(
            &mut self,
            _: &ProcessResultDisplay,
            _: Option<&Deferred3dmodButton>,
            _: Option<Run3dmodMenuOptions>,
            _: ProcessingMethod,
        ) {
            self.calls += 1
        }
    }
    #[derive(Default)]
    struct Param {
        state: super::super::abstract_tilt_panel::TiltParamState,
        input: String,
        output: String,
        mode: Option<Tilt3dFindMode>,
        name: Option<ProcessName>,
    }
    impl Tilt3dFindTiltParam for Param {
        fn set_input_file(&mut self, v: String) {
            self.input = v
        }
        fn set_output_file(&mut self, v: String) {
            self.output = v
        }
        fn set_command_mode(&mut self, v: Tilt3dFindMode) {
            self.mode = Some(v)
        }
        fn set_process_name(&mut self, v: ProcessName) {
            self.name = Some(v)
        }
        fn abstract_tilt_param(
            &mut self,
        ) -> &mut super::super::abstract_tilt_panel::TiltParamState {
            &mut self.state
        }
    }
    #[test]
    fn creates_source_layout_and_initializes_from_align_log() {
        let mut p = Tilt3dFindPanel::get_instance(
            Manager::default(),
            AxisID::Only,
            DialogType::TomogramGeneration,
            Parent::default(),
            true,
        );
        assert_eq!(
            p.layout.buttons_component_order,
            ["tiltButton", "extraButton", "3dmodTomogramButton"]
        );
        p.set_parameters_tiltalign(&BeadsTiltalignParam, true);
        assert_eq!(p.ltf_center_to_center_thickness.text, "100.4");
        assert_eq!(p.abstract_tilt_panel.ltf_tomo_thickness.text, "150");
        assert_eq!(p.abstract_tilt_panel.ltf_z_shift.text, "-3.5");
    }
    #[test]
    fn writes_tilt_3d_find_files_mode_and_process_name() {
        let mut p = Tilt3dFindPanel::get_instance(
            Manager::default(),
            AxisID::Only,
            DialogType::TomogramGeneration,
            Parent::default(),
            false,
        );
        let mut param = Param::default();
        param.state.subset_start_valid = true;
        p.abstract_tilt_panel.ltf_tomo_thickness.text = "80".into();
        let mut state = AbstractTiltPanelState::default();
        assert!(p.get_parameters_tilt(&mut param, true, &mut state).unwrap());
        assert_eq!(param.input, "aligned.ali");
        assert_eq!(param.output, "find.rec");
        assert_eq!(param.mode, Some(Tilt3dFindMode::Tilt3dFind));
        assert_eq!(param.name, Some(ProcessName::TILT_3D_FIND));
    }
    #[test]
    fn routes_tilt_to_parent_and_view_to_manager() {
        let mut p = Tilt3dFindPanel::get_instance(
            Manager::default(),
            AxisID::Only,
            DialogType::TomogramGeneration,
            Parent::default(),
            false,
        );
        p.tilt_action(
            &ProcessResultDisplay,
            None,
            None,
            ProcessingMethod::LocalCpu,
        );
        p.imod_tomogram_action(None, None);
        assert_eq!(p.parent.calls, 1);
        assert_eq!(p.manager.view, 1);
    }
}
