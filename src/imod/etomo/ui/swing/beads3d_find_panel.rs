//! `IMOD/Etomo/src/etomo/ui/swing/Beads3dFindPanel.java`.
//!
//! The four child Swing panels and `ApplicationManager` are intentionally
//! explicit boundaries.  This module retains the source unit's panel ordering,
//! parameter routing, display selection, and the `tilt3dFindAction` process
//! decision without making a second workflow controller.
#![allow(dead_code)]

use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::processing_method::ProcessingMethod;
use crate::imod::etomo::r#type::view_type::ViewType;

use super::find_beads3d_panel::NewstackOrBlendmont3dFindParent;
pub use super::newstack_or_blendmont_panel::MetaData;
use super::newstack_or_blendmont_panel::{
    BlendmontParam, GlobalExpandButton, NewstParam, ReconScreenState,
};
use super::panel_header::{ExpandButton, Expandable, PanelHeader, PanelHeaderState};

/// Java `ConstTiltParam`, `ConstFindBeads3dParam`, `ConstTiltalignParam`, and
/// `TomogramState` boundaries owned by the still-separate child source units.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TiltParam;
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct FindBeads3dParam;
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TiltalignParam;
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TomogramState;

/// Java `ProcessResultDisplay` and `Deferred3dmodButton` action boundaries.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ProcessResultDisplay;
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Deferred3dmodButton;

/// Java `NewstackDisplay`, `BlendmontDisplay`, and `TiltDisplay` boundaries
/// returned by this source unit.  `FindBeads3dDisplay` is the canonical
/// interface in its own source module; this panel's `F` is its implementation.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct NewstackDisplay;
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct BlendmontDisplay;
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TiltDisplay;

/// State the source receives from `ReconScreenState` beyond its translated
/// common stack fields.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Beads3dFindScreenState {
    pub stack_align_and_tilt_header_state: PanelHeaderState,
    pub find_beads_parameter_set_count: u32,
    pub reproject_model_parameter_set_count: u32,
}

/// Abstract `NewstackOrBlendmont3dFindPanel` methods invoked by this Java source unit.
/// The concrete abstract-base state is in `newstack_or_blendmont_3d_find_panel`.
pub trait NewstackOrBlendmont3dFindPanelDisplay {
    fn get_binning(&self) -> i32;
    fn get_3dmod_button(&self) -> Deferred3dmodButton;
    fn initialize(&mut self);
    fn validate(&self) -> bool;
    fn get_parameters(&mut self, meta_data: &mut MetaData);
    fn set_parameters_meta_data(&mut self, meta_data: &MetaData);
    fn set_parameters_blendmont(&mut self, param: &BlendmontParam);
    fn set_parameters_newst(&mut self, param: &NewstParam);
    fn run_process(
        &mut self,
        process_result_display: &ProcessResultDisplay,
        process_series: &mut ProcessSeries,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    );
}

/// `Tilt3dFindPanel` methods invoked by this Java source unit.
pub trait Tilt3dFindPanel {
    fn reregister_processing_method_mediator(&mut self);
    fn get_processing_method(&self) -> ProcessingMethod;
    fn done(&mut self);
    fn set_state(&mut self, state: &TomogramState, meta_data: &MetaData);
    fn set_parameters_tilt(&mut self, param: &TiltParam, initialize: bool);
    fn set_parameters_tiltalign(&mut self, param: &TiltalignParam, initialize: bool);
    fn set_override_parameters(&mut self, meta_data: &MetaData);
    fn get_parameters(&mut self, meta_data: &mut MetaData);
    fn set_parameters_meta_data(&mut self, meta_data: &MetaData);
    fn tilt_3d_find_action(
        &mut self,
        process_result_display: &ProcessResultDisplay,
        deferred_3dmod_button: Option<&Deferred3dmodButton>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    );
}

/// `FindBeads3dPanel` methods invoked by this Java source unit.
pub trait FindBeads3dPanel {
    fn done(&mut self);
    fn update_advanced(&mut self, advanced: bool);
    fn is_advanced(&self) -> bool;
    fn get_bead_size(&self) -> String;
    fn set_parameters(&mut self, param: &FindBeads3dParam, initialize: bool);
    fn set_parameters_screen_state(&mut self, screen_state: &Beads3dFindScreenState);
    fn get_parameters_screen_state(&mut self, screen_state: &mut Beads3dFindScreenState);
}

/// `ReprojectModelPanel` methods invoked by this Java source unit.
pub trait ReprojectModelPanel {
    fn done(&mut self);
    fn set_parameters_screen_state(&mut self, screen_state: &Beads3dFindScreenState);
}

/// Direct `ApplicationManager` calls made by `Beads3dFindPanel.java`.
pub trait Beads3dFindPanelApplicationManager {
    fn view_type(&self) -> ViewType;
    fn aligned_stack_exists(&self, axis_id: AxisID) -> bool;
    fn equals_binning(&self, axis_id: AxisID, binning: i32) -> bool;
    fn set_stack_using_newst_or_blend_3d_find_output(&mut self, axis_id: AxisID, value: bool);
    fn pack(&mut self, axis_id: AxisID);
    fn open_missing_aligned_stack_message(&mut self, axis_id: AxisID);
}

/// Java `ProcessSeries` state constructed when aligned-stack generation must
/// precede `tilt_3dfind`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProcessSeries {
    pub axis_id: AxisID,
    pub dialog_type: DialogType,
    pub next_process: Option<(String, ProcessingMethod)>,
    pub callback_name: String,
}

impl ProcessSeries {
    /// Java `new ProcessSeries(..., tilt3dFindPanel, "tilt3dFindAction")`.
    pub fn new(axis_id: AxisID, dialog_type: DialogType) -> Self {
        Self {
            axis_id,
            dialog_type,
            next_process: None,
            callback_name: "tilt3dFindAction".into(),
        }
    }

    /// Java `setNextProcess(ProcessName.TILT_3D_FIND.toString(), method)`.
    pub fn set_next_process(&mut self, processing_method: ProcessingMethod) {
        self.next_process = Some(("tilt_3dfind".into(), processing_method));
    }
}

/// Source-visible Swing layout created by `createPanel`.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Beads3dFindPanelLayout {
    pub root_box_layout_y_axis: bool,
    pub root_component_order: Vec<String>,
    pub generate_tomogram_box_layout_y_axis: bool,
    pub generate_tomogram_etched_border: bool,
    pub generate_tomogram_component_order: Vec<String>,
    pub generate_tomogram_body_box_layout_y_axis: bool,
    pub generate_tomogram_body_component_order: Vec<String>,
    pub generate_tomogram_body_visible: bool,
    pub root_visible: bool,
}

/// Java final `Beads3dFindPanel`; child panels are generic so their translated
/// units can replace these direct interfaces without an adapter layer.
pub struct Beads3dFindPanel<N, T, F, R> {
    pub pnl_root: Beads3dFindPanelLayout,
    pub newstack_or_blendmont_3d_find_panel: N,
    pub tilt_3d_find_panel: T,
    pub find_beads_3d_panel: F,
    pub reproject_model_panel: R,
    pub axis_id: AxisID,
    pub dialog_type: DialogType,
    pub header: PanelHeader,
    pub view_type: ViewType,
}

impl<N, T, F, R> Beads3dFindPanel<N, T, F, R>
where
    N: NewstackOrBlendmont3dFindPanelDisplay,
    T: Tilt3dFindPanel,
    F: FindBeads3dPanel,
    R: ReprojectModelPanel,
{
    /// Java constructor.  Child factory selection is an owner boundary; the
    /// selected child object and source `ViewType` are stored unchanged.
    pub fn new(
        newstack_or_blendmont_3d_find_panel: N,
        tilt_3d_find_panel: T,
        find_beads_3d_panel: F,
        reproject_model_panel: R,
        axis_id: AxisID,
        dialog_type: DialogType,
        view_type: ViewType,
        _global_advanced_button: &GlobalExpandButton,
    ) -> Self {
        Self {
            pnl_root: Beads3dFindPanelLayout {
                root_visible: true,
                ..Default::default()
            },
            newstack_or_blendmont_3d_find_panel,
            tilt_3d_find_panel,
            find_beads_3d_panel,
            reproject_model_panel,
            axis_id,
            dialog_type,
            header: PanelHeader::new(
                "Align Stack and Create Tomogram",
                false,
                false,
                dialog_type,
                true,
                false,
                true,
                false,
                true,
            ),
            view_type,
        }
    }

    /// Java static `getInstance` after its caller has selected the source child
    /// factory (`Blendmont3dFindPanel` for MONTAGE, otherwise `Newstack3dFindPanel`).
    pub fn get_instance(
        newstack_or_blendmont_3d_find_panel: N,
        tilt_3d_find_panel: T,
        find_beads_3d_panel: F,
        reproject_model_panel: R,
        axis_id: AxisID,
        dialog_type: DialogType,
        view_type: ViewType,
        global_advanced_button: &GlobalExpandButton,
    ) -> Self {
        let mut instance = Self::new(
            newstack_or_blendmont_3d_find_panel,
            tilt_3d_find_panel,
            find_beads_3d_panel,
            reproject_model_panel,
            axis_id,
            dialog_type,
            view_type,
            global_advanced_button,
        );
        instance.create_panel();
        instance
    }

    /// Java `reregisterProcessingMethodMediator`.
    pub fn reregister_processing_method_mediator(&mut self) {
        self.tilt_3d_find_panel
            .reregister_processing_method_mediator();
    }
    /// Java `getProcessingMethod`.
    pub fn get_processing_method(&self) -> ProcessingMethod {
        self.tilt_3d_find_panel.get_processing_method()
    }
    /// Java `done`.
    pub fn done(&mut self) {
        self.tilt_3d_find_panel.done();
        self.find_beads_3d_panel.done();
        self.reproject_model_panel.done();
    }
    /// Java `updateAdvanced`.
    pub fn update_advanced(&mut self, advanced: bool) {
        self.find_beads_3d_panel.update_advanced(advanced);
    }
    /// Java `expand(GlobalExpandButton)`, intentionally empty.
    pub fn expand_global_button(&mut self) {}
    /// Java `expand(ExpandButton)`.
    pub fn expand_expand_button<M: Beads3dFindPanelApplicationManager>(
        &mut self,
        button: &ExpandButton,
        manager: &mut M,
    ) {
        if self.header.equals_open_close(button) {
            self.pnl_root.generate_tomogram_body_visible = button.is_expanded();
        }
        manager.pack(self.axis_id);
    }
    /// Java `createPanel`.
    pub fn create_panel(&mut self) {
        self.pnl_root.root_box_layout_y_axis = true;
        self.pnl_root.root_component_order = vec![
            "pnlGenerateTomogram".into(),
            "findBeads3dPanel".into(),
            "reprojectModelPanel".into(),
        ];
        self.pnl_root.generate_tomogram_box_layout_y_axis = true;
        self.pnl_root.generate_tomogram_etched_border = true;
        self.pnl_root.generate_tomogram_component_order =
            vec!["header".into(), "pnlGenerateTomogramBody".into()];
        self.pnl_root.generate_tomogram_body_box_layout_y_axis = true;
        self.pnl_root.generate_tomogram_body_component_order = vec![
            "newstackOrBlendmont3dFindPanel".into(),
            "tilt3dFindPanel".into(),
        ];
    }
    /// Java `getComponent`.
    pub fn get_component(&self) -> &Beads3dFindPanelLayout {
        &self.pnl_root
    }
    /// Java `isAdvanced`.
    pub fn is_advanced(&self) -> bool {
        self.find_beads_3d_panel.is_advanced()
    }
    /// Java `isMultifilt`.
    pub fn is_multifilt(&self) -> bool {
        false
    }
    /// Java `isCtf3d`.
    pub fn is_ctf_3d(&self) -> bool {
        false
    }
    /// Java `isBackProjection`.
    pub fn is_back_projection(&self) -> bool {
        true
    }
    /// Java `isMethodPlugin`.
    pub fn is_method_plugin(&self) -> bool {
        false
    }
    /// Java `isSirt`.
    pub fn is_sirt(&self) -> bool {
        false
    }
    /// Java `getNewstack3dFindDisplay`; typed display ownership remains child-panel boundary.
    pub fn get_newstack_3d_find_display(&self) -> Option<&N> {
        (self.view_type != ViewType::Montage).then_some(&self.newstack_or_blendmont_3d_find_panel)
    }
    /// Java `getBlendmont3dFindDisplay`; typed display ownership remains child-panel boundary.
    pub fn get_blendmont_3d_find_display(&self) -> Option<&N> {
        (self.view_type == ViewType::Montage).then_some(&self.newstack_or_blendmont_3d_find_panel)
    }
    /// Java `getTilt3dFindDisplay`.
    pub fn get_tilt_3d_find_display(&self) -> &T {
        &self.tilt_3d_find_panel
    }
    /// Java `getFindBeads3dDisplay`.
    pub fn get_find_beads_3d_display(&self) -> &F {
        &self.find_beads_3d_panel
    }
    /// Java `setTiltState`.
    pub fn set_tilt_state(&mut self, state: &TomogramState, meta_data: &MetaData) {
        self.tilt_3d_find_panel.set_state(state, meta_data);
    }
    /// Java `setParameters(ConstTiltParam, boolean)`.
    pub fn set_parameters_tilt(&mut self, param: &TiltParam, initialize: bool) {
        self.tilt_3d_find_panel
            .set_parameters_tilt(param, initialize);
    }
    /// Java `setParameters(ConstFindBeads3dParam, boolean)`.
    pub fn set_parameters_find_beads_3d(&mut self, param: &FindBeads3dParam, initialize: bool) {
        self.find_beads_3d_panel.set_parameters(param, initialize);
    }
    /// Java `initialize`.
    pub fn initialize(&mut self) {
        self.newstack_or_blendmont_3d_find_panel.initialize();
    }
    /// Java `setParameters(ConstTiltalignParam, boolean)`.
    pub fn set_parameters_tiltalign(&mut self, param: &TiltalignParam, initialize: bool) {
        self.tilt_3d_find_panel
            .set_parameters_tiltalign(param, initialize);
    }
    /// Java `setOverrideParameters`.
    pub fn set_override_parameters(&mut self, meta_data: &MetaData) {
        self.tilt_3d_find_panel.set_override_parameters(meta_data);
    }
    /// Java `setParameters(ReconScreenState)`.
    pub fn set_parameters_screen_state(&mut self, screen_state: &Beads3dFindScreenState) {
        self.header
            .set_state(Some(&screen_state.stack_align_and_tilt_header_state));
        self.find_beads_3d_panel
            .set_parameters_screen_state(screen_state);
        self.reproject_model_panel
            .set_parameters_screen_state(screen_state);
    }
    /// Java `getParameters(ReconScreenState)`.
    pub fn get_parameters_screen_state(&mut self, screen_state: &mut Beads3dFindScreenState) {
        self.header
            .get_state(Some(&mut screen_state.stack_align_and_tilt_header_state));
        self.find_beads_3d_panel
            .get_parameters_screen_state(screen_state);
    }
    /// Java `setParameters(BlendmontParam)`.
    pub fn set_parameters_blendmont(&mut self, param: &BlendmontParam) {
        if self.view_type == ViewType::Montage {
            self.newstack_or_blendmont_3d_find_panel
                .set_parameters_blendmont(param);
        }
    }
    /// Java `setParameters(NewstParam)`.
    pub fn set_parameters_newst(&mut self, param: &NewstParam) {
        if self.view_type != ViewType::Montage {
            self.newstack_or_blendmont_3d_find_panel
                .set_parameters_newst(param);
        }
    }
    /// Java `setVisible`.
    pub fn set_visible(&mut self, visible: bool) {
        self.pnl_root.root_visible = visible;
    }
    /// Java `validate`.
    pub fn validate(&self) -> bool {
        self.newstack_or_blendmont_3d_find_panel.validate()
    }
    /// Java `getParameters(MetaData)`.
    pub fn get_parameters_meta_data(&mut self, meta_data: &mut MetaData) {
        self.newstack_or_blendmont_3d_find_panel
            .get_parameters(meta_data);
        self.tilt_3d_find_panel.get_parameters(meta_data);
    }
    /// Java `setParameters(ConstMetaData)`.
    pub fn set_parameters_meta_data(&mut self, meta_data: &MetaData) {
        self.newstack_or_blendmont_3d_find_panel
            .set_parameters_meta_data(meta_data);
        self.tilt_3d_find_panel.set_parameters_meta_data(meta_data);
    }
    /// Java `getBeadSize`.
    pub fn get_bead_size(&self) -> String {
        self.find_beads_3d_panel.get_bead_size()
    }
    /// Java `isFiducialess` delegated to `EraseGoldPanel`.
    pub fn is_fiducialess<P: NewstackOrBlendmont3dFindParent>(&self, parent: &P) -> bool {
        parent.is_fiducialess()
    }
    /// Java `tilt3dFindAction`.
    pub fn tilt_3d_find_action<
        M: Beads3dFindPanelApplicationManager,
        P: NewstackOrBlendmont3dFindParent,
    >(
        &mut self,
        manager: &mut M,
        parent: &P,
        process_result_display: &ProcessResultDisplay,
        deferred_3dmod_button: Option<&Deferred3dmodButton>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
        tiltp_processing_method: ProcessingMethod,
    ) {
        if !manager.aligned_stack_exists(self.axis_id) && self.is_fiducialess(parent) {
            manager.open_missing_aligned_stack_message(self.axis_id);
            return;
        }
        if !self.validate() {
            return;
        }
        if !manager.aligned_stack_exists(self.axis_id)
            || !manager.equals_binning(
                self.axis_id,
                self.newstack_or_blendmont_3d_find_panel.get_binning(),
            )
        {
            let mut process_series = ProcessSeries::new(self.axis_id, self.dialog_type);
            process_series.set_next_process(tiltp_processing_method);
            self.newstack_or_blendmont_3d_find_panel.run_process(
                process_result_display,
                &mut process_series,
                run_3dmod_menu_options,
            );
        } else {
            manager.set_stack_using_newst_or_blend_3d_find_output(self.axis_id, false);
            self.tilt_3d_find_panel.tilt_3d_find_action(
                process_result_display,
                deferred_3dmod_button,
                run_3dmod_menu_options,
            );
        }
    }
}

impl<N, T, F, R> Expandable for Beads3dFindPanel<N, T, F, R>
where
    N: NewstackOrBlendmont3dFindPanelDisplay,
    T: Tilt3dFindPanel,
    F: FindBeads3dPanel,
    R: ReprojectModelPanel,
{
    fn expand_expand_button(&mut self, _button: &ExpandButton) {}
    fn expand_global_button(&mut self, _: &super::process_dialog::GlobalExpandButton) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct Newstack {
        binning: i32,
        valid: bool,
        runs: Vec<ProcessSeries>,
        initialized: bool,
    }
    impl NewstackOrBlendmont3dFindPanelDisplay for Newstack {
        fn get_binning(&self) -> i32 {
            self.binning
        }
        fn get_3dmod_button(&self) -> Deferred3dmodButton {
            Deferred3dmodButton
        }
        fn initialize(&mut self) {
            self.initialized = true
        }
        fn validate(&self) -> bool {
            self.valid
        }
        fn get_parameters(&mut self, _: &mut MetaData) {}
        fn set_parameters_meta_data(&mut self, _: &MetaData) {}
        fn set_parameters_blendmont(&mut self, _: &BlendmontParam) {}
        fn set_parameters_newst(&mut self, _: &NewstParam) {}
        fn run_process(
            &mut self,
            _: &ProcessResultDisplay,
            p: &mut ProcessSeries,
            _: Option<Run3dmodMenuOptions>,
        ) {
            self.runs.push(p.clone())
        }
    }
    #[derive(Default)]
    struct Tilt {
        runs: u32,
    }
    impl Tilt3dFindPanel for Tilt {
        fn reregister_processing_method_mediator(&mut self) {}
        fn get_processing_method(&self) -> ProcessingMethod {
            ProcessingMethod::LocalCpu
        }
        fn done(&mut self) {}
        fn set_state(&mut self, _: &TomogramState, _: &MetaData) {}
        fn set_parameters_tilt(&mut self, _: &TiltParam, _: bool) {}
        fn set_parameters_tiltalign(&mut self, _: &TiltalignParam, _: bool) {}
        fn set_override_parameters(&mut self, _: &MetaData) {}
        fn get_parameters(&mut self, _: &mut MetaData) {}
        fn set_parameters_meta_data(&mut self, _: &MetaData) {}
        fn tilt_3d_find_action(
            &mut self,
            _: &ProcessResultDisplay,
            _: Option<&Deferred3dmodButton>,
            _: Option<Run3dmodMenuOptions>,
        ) {
            self.runs += 1
        }
    }
    #[derive(Default)]
    struct Find {
        advanced: bool,
    }
    impl FindBeads3dPanel for Find {
        fn done(&mut self) {}
        fn update_advanced(&mut self, a: bool) {
            self.advanced = a
        }
        fn is_advanced(&self) -> bool {
            self.advanced
        }
        fn get_bead_size(&self) -> String {
            "10".into()
        }
        fn set_parameters(&mut self, _: &FindBeads3dParam, _: bool) {}
        fn set_parameters_screen_state(&mut self, _: &Beads3dFindScreenState) {}
        fn get_parameters_screen_state(&mut self, _: &mut Beads3dFindScreenState) {}
    }
    #[derive(Default)]
    struct Reproject;
    impl ReprojectModelPanel for Reproject {
        fn done(&mut self) {}
        fn set_parameters_screen_state(&mut self, _: &Beads3dFindScreenState) {}
    }
    #[derive(Default)]
    struct Manager {
        exists: bool,
        matching: bool,
        missing: u32,
        reset: u32,
    }
    impl Beads3dFindPanelApplicationManager for Manager {
        fn view_type(&self) -> ViewType {
            ViewType::SingleView
        }
        fn aligned_stack_exists(&self, _: AxisID) -> bool {
            self.exists
        }
        fn equals_binning(&self, _: AxisID, _: i32) -> bool {
            self.matching
        }
        fn set_stack_using_newst_or_blend_3d_find_output(&mut self, _: AxisID, _: bool) {
            self.reset += 1
        }
        fn pack(&mut self, _: AxisID) {}
        fn open_missing_aligned_stack_message(&mut self, _: AxisID) {
            self.missing += 1
        }
    }
    struct Parent(bool);
    impl NewstackOrBlendmont3dFindParent for Parent {
        fn get_bead_size(&self) -> String {
            String::new()
        }
        fn is_fiducialess(&self) -> bool {
            self.0
        }
    }
    fn panel(valid: bool) -> Beads3dFindPanel<Newstack, Tilt, Find, Reproject> {
        Beads3dFindPanel::get_instance(
            Newstack {
                binning: 2,
                valid,
                ..Default::default()
            },
            Tilt::default(),
            Find::default(),
            Reproject,
            AxisID::First,
            DialogType::FinalAlignedStack,
            ViewType::SingleView,
            &GlobalExpandButton::get_instance("Advanced", "Basic"),
        )
    }
    #[test]
    fn source_panel_order_and_constants_are_preserved() {
        let panel = panel(true);
        assert_eq!(
            panel.pnl_root.root_component_order,
            [
                "pnlGenerateTomogram",
                "findBeads3dPanel",
                "reprojectModelPanel"
            ]
        );
        assert!(panel.is_back_projection());
        assert!(!panel.is_sirt());
    }
    #[test]
    fn missing_fiducialess_aligned_stack_shows_message_before_validation() {
        let mut panel = panel(false);
        let mut manager = Manager::default();
        panel.tilt_3d_find_action(
            &mut manager,
            &Parent(true),
            &ProcessResultDisplay,
            None,
            None,
            ProcessingMethod::Queue,
        );
        assert_eq!(manager.missing, 1);
        assert!(panel.newstack_or_blendmont_3d_find_panel.runs.is_empty());
    }
    #[test]
    fn action_runs_stack_first_only_when_stack_is_missing_or_binning_differs() {
        let mut panel = panel(true);
        let mut manager = Manager::default();
        panel.tilt_3d_find_action(
            &mut manager,
            &Parent(false),
            &ProcessResultDisplay,
            None,
            None,
            ProcessingMethod::Queue,
        );
        assert_eq!(
            panel.newstack_or_blendmont_3d_find_panel.runs[0].next_process,
            Some(("tilt_3dfind".into(), ProcessingMethod::Queue))
        );
        manager.exists = true;
        manager.matching = true;
        panel.tilt_3d_find_action(
            &mut manager,
            &Parent(false),
            &ProcessResultDisplay,
            None,
            None,
            ProcessingMethod::Queue,
        );
        assert_eq!(manager.reset, 1);
        assert_eq!(panel.tilt_3d_find_panel.runs, 1);
    }
}
