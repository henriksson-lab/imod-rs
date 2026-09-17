//! `IMOD/Etomo/src/etomo/ui/swing/TomogramGenerationDialog.java`.
//!
//! Swing containers, panel implementations, plugins, parameter objects, and
//! `UIHarness` are direct named collaborators.  This unit owns the dialog's
//! method selection, construction order, parameter routing, display updates,
//! context-menu choice, and completion ordering.
#![allow(dead_code)]

use super::tomogram_generation_parent::TomogramGenerationParent;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::processing_method::ProcessingMethod;

pub const X_AXIS_TILT_TOOLTIP: &str = "This line allows one to rotate the reconstruction around the X axis, so that a section that appears to be tilted around the X axis can be made flat to fit into a smaller volume.";

/// Direct `ConstTiltParam`, `MultifiltSetupParam`, `SirtsetupParam`, and
/// `Ctf3dSetupParam` parameter boundaries.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ConstTiltParamBoundary;
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MultifiltSetupParamBoundary;
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SirtsetupParamBoundary;
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Ctf3dSetupParamBoundary;
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TomogramStateBoundary;
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ReconScreenStateBoundary;

/// The `ConstMetaData`/`MetaData` operations this source unit calls.
pub trait TomogramGenerationMetaData {
    fn is_gen_back_projection(&self, axis_id: AxisID) -> bool;
    fn is_gen_filter_trials(&self, axis_id: AxisID) -> bool;
    fn is_gen_sirt(&self, axis_id: AxisID) -> bool;
    fn set_gen_back_projection(&mut self, axis_id: AxisID, value: bool);
    fn set_gen_filter_trials(&mut self, axis_id: AxisID, value: bool);
    fn set_gen_sirt(&mut self, axis_id: AxisID, value: bool);
}

/// Java `RadioButton` state and calls directly made by this dialog.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TomogramGenerationRadioButton {
    pub text: String,
    pub selected: bool,
    pub action_listener_count: usize,
}
impl TomogramGenerationRadioButton {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            selected: false,
            action_listener_count: 0,
        }
    }
    pub fn action_command(&self) -> &str {
        &self.text
    }
    pub fn do_click(&mut self) {
        self.selected = true;
    }
}

/// Java `TiltPanel` calls from this source unit.
pub trait TomogramGenerationTiltPanel {
    fn root_name(&self) -> String;
    fn set_parameters_metadata(&mut self, meta_data: &dyn TomogramGenerationMetaData);
    fn get_parameters_metadata(
        &mut self,
        meta_data: &mut dyn TomogramGenerationMetaData,
    ) -> Result<(), String>;
    fn set_parameters_tilt(&mut self, parameter: &ConstTiltParamBoundary, initialize: bool);
    fn set_parameters_screen_state(&mut self, screen_state: &ReconScreenStateBoundary);
    fn get_parameters_screen_state(&mut self, screen_state: &mut ReconScreenStateBoundary);
    fn update_display(&mut self);
    fn done(&mut self);
    fn checkpoint(&mut self, state: &TomogramStateBoundary);
    fn set_state(
        &mut self,
        state: &TomogramStateBoundary,
        meta_data: &dyn TomogramGenerationMetaData,
    );
    fn msg_method_changed(&mut self);
    fn set_filter_type_action_listener(&mut self);
    fn x_axis_tilt(&self) -> String;
    fn use_local_alignment(&self) -> bool;
    fn use_z_factors(&self) -> bool;
    fn tomo_thickness(&self) -> Option<i64>;
    fn processing_method(&self) -> ProcessingMethod;
    fn focus_listener_added(&mut self, add: bool);
    fn local_alignment_listener_added(&mut self, add: bool);
    fn z_factors_listener_added(&mut self, add: bool);
}
/// Java `MultifiltPanel` calls from this source unit.
pub trait TomogramGenerationMultifiltPanel {
    fn component_name(&self) -> String;
    fn set_parameters_metadata(&mut self, meta_data: &dyn TomogramGenerationMetaData);
    fn get_parameters_metadata(
        &mut self,
        meta_data: &mut dyn TomogramGenerationMetaData,
    ) -> Result<(), String>;
    fn set_parameters(&mut self, parameter: &MultifiltSetupParamBoundary);
    fn set_parameters_screen_state(&mut self, screen_state: &ReconScreenStateBoundary);
    fn get_parameters_screen_state(&mut self, screen_state: &mut ReconScreenStateBoundary);
    fn done(&mut self);
    fn msg_method_changed(&mut self);
}
/// Java `SirtPanel` calls from this source unit.
pub trait TomogramGenerationSirtPanel {
    fn root_name(&self) -> String;
    fn set_parameters_metadata(&mut self, meta_data: &dyn TomogramGenerationMetaData);
    fn get_parameters_metadata(
        &mut self,
        meta_data: &mut dyn TomogramGenerationMetaData,
    ) -> Result<(), String>;
    fn set_parameters(&mut self, parameter: &SirtsetupParamBoundary);
    fn set_parameters_screen_state(&mut self, screen_state: &ReconScreenStateBoundary);
    fn get_parameters_screen_state(&mut self, screen_state: &mut ReconScreenStateBoundary);
    fn update_display(&mut self);
    fn done(&mut self);
    fn checkpoint(&mut self, state: &TomogramStateBoundary);
    fn msg_sirt_succeeded(&mut self);
    fn msg_method_changed(&mut self);
}
/// Java `Ctf3dPanel` calls from this source unit.
pub trait TomogramGenerationCtf3dPanel {
    fn component_name(&self) -> String;
    fn set_parameters_metadata(&mut self, meta_data: &dyn TomogramGenerationMetaData);
    fn get_parameters_metadata(
        &mut self,
        meta_data: &mut dyn TomogramGenerationMetaData,
    ) -> Result<(), String>;
    fn set_parameters(&mut self, parameter: &Ctf3dSetupParamBoundary);
    fn set_parameters_screen_state(&mut self, screen_state: &ReconScreenStateBoundary);
    fn get_parameters_screen_state(&mut self, screen_state: &mut ReconScreenStateBoundary);
    fn update_display(&mut self);
    fn done(&mut self);
    fn msg_method_changed(&mut self);
}
/// Java `PluginPanel` calls made by this source unit.
pub trait TomogramGenerationMethodPluginPanel {
    fn component_name(&self) -> String;
    fn button_title(&self) -> String;
    fn update_display(&mut self);
    fn set_parameters(&mut self, screen_state: &ReconScreenStateBoundary);
    fn get_parameters(&mut self, screen_state: &mut ReconScreenStateBoundary);
    fn done(&mut self);
    fn msg_visibility_changed(&mut self, visible: bool);
}
/// Canonical Java `TomogramGenerationExpert.doneDialog()` boundary.
pub use super::tomogram_generation_expert::TomogramGenerationExpert;
/// Java `UIHarness.INSTANCE.pack/moveSubFrame` boundary.
pub trait TomogramGenerationUiHarness {
    fn pack(&mut self, axis_id: AxisID);
    fn move_sub_frame(&mut self);
}

/// Exact `ContextPopup` construction inputs retained at the GUI boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TomogramGenerationContextPopup {
    pub title: String,
    pub man_page_labels: Vec<String>,
    pub man_pages: Vec<String>,
    pub log_file_labels: Vec<String>,
    pub log_files: Vec<String>,
    pub axis_id: AxisID,
}
/// Source-visible `JPanel` construction state.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TomogramGenerationDialogLayout {
    pub root_order: Vec<String>,
    pub method_order: Vec<String>,
    pub root_y_axis: bool,
    pub method_x_axis: bool,
    pub bevel_title: Option<String>,
    pub aligned_left: bool,
    pub exit_buttons_added: bool,
    pub displayed: bool,
    pub root_mouse_listener_count: usize,
}

/// Java `TomogramGenerationDialog`; generic collaborators retain their source-unit identity.
pub struct TomogramGenerationDialog<T, M, S, C, P, E, H>
where
    T: TomogramGenerationTiltPanel,
    M: TomogramGenerationMultifiltPanel,
    S: TomogramGenerationSirtPanel,
    C: TomogramGenerationCtf3dPanel,
    P: TomogramGenerationMethodPluginPanel,
    E: TomogramGenerationExpert,
    H: TomogramGenerationUiHarness,
{
    pub axis_id: AxisID,
    pub layout: TomogramGenerationDialogLayout,
    pub rb_back_projection: TomogramGenerationRadioButton,
    pub rb_multifilt: TomogramGenerationRadioButton,
    pub rb_sirt: TomogramGenerationRadioButton,
    pub rb_ctf3d: Option<TomogramGenerationRadioButton>,
    pub rb_method_plugin: Option<TomogramGenerationRadioButton>,
    pub tilt_panel: T,
    pub multifilt_panel: M,
    pub sirt_panel: S,
    pub ctf3d_panel: Option<C>,
    pub method_plugin_panel: Option<P>,
    pub expert: E,
    pub ui_harness: H,
    pub btn_execute_text: String,
    pub last_context_popup: Option<TomogramGenerationContextPopup>,
}

impl<T, M, S, C, P, E, H> TomogramGenerationDialog<T, M, S, C, P, E, H>
where
    T: TomogramGenerationTiltPanel,
    M: TomogramGenerationMultifiltPanel,
    S: TomogramGenerationSirtPanel,
    C: TomogramGenerationCtf3dPanel,
    P: TomogramGenerationMethodPluginPanel,
    E: TomogramGenerationExpert,
    H: TomogramGenerationUiHarness,
{
    /// Java private constructor plus static `getInstance` sequence.
    pub fn get_instance(
        axis_id: AxisID,
        tilt_panel: T,
        multifilt_panel: M,
        sirt_panel: S,
        ctf3d_panel: Option<C>,
        method_plugin_panel: Option<P>,
        expert: E,
        ui_harness: H,
    ) -> Self {
        let rb_method_plugin = method_plugin_panel
            .as_ref()
            .map(|panel| TomogramGenerationRadioButton::new(panel.button_title()));
        let mut dialog = Self {
            axis_id,
            layout: TomogramGenerationDialogLayout::default(),
            rb_back_projection: TomogramGenerationRadioButton::new("Back Projection"),
            rb_multifilt: TomogramGenerationRadioButton::new("Filter Trials"),
            rb_sirt: TomogramGenerationRadioButton::new("SIRT"),
            rb_ctf3d: Some(TomogramGenerationRadioButton::new("3D CTF")),
            rb_method_plugin,
            tilt_panel,
            multifilt_panel,
            sirt_panel,
            ctf3d_panel,
            method_plugin_panel,
            expert,
            ui_harness,
            btn_execute_text: String::new(),
            last_context_popup: None,
        };
        dialog.create_panel();
        dialog.update_display();
        dialog.add_listeners();
        dialog
    }
    /// Java private `createPanel()`.
    fn create_panel(&mut self) {
        self.rb_back_projection.selected = true;
        self.layout.root_y_axis = true;
        self.layout.method_x_axis = true;
        self.layout.bevel_title = Some("Tomogram Generation".into());
        self.layout.root_order = vec![
            "pnlMethod".into(),
            self.tilt_panel.root_name(),
            self.multifilt_panel.component_name(),
            self.sirt_panel.root_name(),
        ];
        if let Some(panel) = &self.ctf3d_panel {
            self.layout.root_order.push(panel.component_name());
        }
        if let Some(panel) = &self.method_plugin_panel {
            self.layout.root_order.push(panel.component_name());
        }
        self.layout.method_order = vec![
            "horizontalGlue".into(),
            "Back Projection".into(),
            "horizontalGlue".into(),
            "Filter Trials".into(),
            "horizontalGlue".into(),
            "SIRT".into(),
            "horizontalGlue".into(),
        ];
        if self.rb_ctf3d.is_some() {
            self.layout.method_order.push("3D CTF".into());
        }
        self.layout.method_order.push("horizontalGlue".into());
        if let Some(button) = &self.rb_method_plugin {
            self.layout.method_order.push(button.text.clone());
            self.layout.method_order.push("horizontalGlue".into());
        }
        self.btn_execute_text = "Done".into();
        self.layout.exit_buttons_added = true;
        self.layout.aligned_left = true;
    }
    /// Java private `addListeners()`.
    fn add_listeners(&mut self) {
        self.layout.root_mouse_listener_count += 1;
        self.rb_back_projection.action_listener_count += 1;
        self.rb_multifilt.action_listener_count += 1;
        self.rb_sirt.action_listener_count += 1;
        if let Some(button) = &mut self.rb_ctf3d {
            button.action_listener_count += 1;
        }
        if let Some(button) = &mut self.rb_method_plugin {
            button.action_listener_count += 1;
        }
    }
    pub fn get_advanced_button(&self) -> &'static str {
        "btnAdvanced"
    }
    pub fn msg_sirt_succeeded(&mut self) {
        self.sirt_panel.msg_sirt_succeeded();
    }
    pub fn sirt_checkpoint(
        &mut self,
        _tilt: &ConstTiltParamBoundary,
        state: &TomogramStateBoundary,
    ) {
        self.sirt_panel.checkpoint(state);
        if let Some(panel) = &mut self.method_plugin_panel {
            panel.update_display();
        }
    }
    pub fn allow_tilt_com_save(&self) -> bool {
        true
    }
    pub fn get_parameters(
        &mut self,
        meta_data: &mut dyn TomogramGenerationMetaData,
    ) -> Result<(), String> {
        meta_data.set_gen_back_projection(self.axis_id, self.rb_back_projection.selected);
        meta_data.set_gen_filter_trials(self.axis_id, self.rb_multifilt.selected);
        meta_data.set_gen_sirt(self.axis_id, self.rb_sirt.selected);
        self.tilt_panel.get_parameters_metadata(meta_data)?;
        self.multifilt_panel.get_parameters_metadata(meta_data)?;
        if let Some(panel) = &mut self.ctf3d_panel {
            panel.get_parameters_metadata(meta_data)?;
        }
        self.sirt_panel.get_parameters_metadata(meta_data)
    }
    pub fn get_parameters_screen_state(&mut self, state: &mut ReconScreenStateBoundary) {
        self.tilt_panel.get_parameters_screen_state(state);
        self.multifilt_panel.get_parameters_screen_state(state);
        if let Some(panel) = &mut self.ctf3d_panel {
            panel.get_parameters_screen_state(state);
        }
        self.sirt_panel.get_parameters_screen_state(state);
        if let Some(panel) = &mut self.method_plugin_panel {
            panel.get_parameters(state);
        }
    }
    pub fn set_parameters_metadata(&mut self, meta_data: &dyn TomogramGenerationMetaData) {
        if meta_data.is_gen_back_projection(self.axis_id) {
            self.rb_back_projection.selected = true;
        } else if meta_data.is_gen_filter_trials(self.axis_id) {
            self.rb_multifilt.selected = true;
        } else if meta_data.is_gen_sirt(self.axis_id) {
            self.rb_sirt.selected = true;
        } else if self.rb_ctf3d.is_some() {
            self.rb_ctf3d.as_mut().unwrap().selected = true;
        }
        self.tilt_panel.set_parameters_metadata(meta_data);
        self.multifilt_panel.set_parameters_metadata(meta_data);
        self.sirt_panel.set_parameters_metadata(meta_data);
        if let Some(panel) = &mut self.ctf3d_panel {
            panel.set_parameters_metadata(meta_data);
        }
        self.method_changed();
        self.tilt_panel.set_filter_type_action_listener();
    }
    pub fn set_parameters_tilt(&mut self, parameter: &ConstTiltParamBoundary, initialize: bool) {
        self.tilt_panel.set_parameters_tilt(parameter, initialize);
    }
    pub fn set_parameters_multifilt(&mut self, parameter: &MultifiltSetupParamBoundary) {
        self.multifilt_panel.set_parameters(parameter);
    }
    pub fn set_parameters_sirt(&mut self, parameter: &SirtsetupParamBoundary) {
        self.sirt_panel.set_parameters(parameter);
    }
    pub fn set_parameters_ctf3d(&mut self, parameter: &Ctf3dSetupParamBoundary) {
        if let Some(panel) = &mut self.ctf3d_panel {
            panel.set_parameters(parameter);
        }
    }
    pub fn set_parameters_screen_state(&mut self, state: &ReconScreenStateBoundary) {
        self.tilt_panel.set_parameters_screen_state(state);
        self.multifilt_panel.set_parameters_screen_state(state);
        self.sirt_panel.set_parameters_screen_state(state);
        if let Some(panel) = &mut self.ctf3d_panel {
            panel.set_parameters_screen_state(state);
        }
        if let Some(panel) = &mut self.method_plugin_panel {
            panel.set_parameters(state);
        }
    }
    /// Java private `updateDisplay()`.
    fn update_display(&mut self) {
        self.tilt_panel.update_display();
        self.sirt_panel.update_display();
        if let Some(panel) = &mut self.ctf3d_panel {
            panel.update_display();
        }
        if let Some(panel) = &mut self.method_plugin_panel {
            panel.update_display();
        }
        self.ui_harness.pack(self.axis_id);
    }
    /// Java `popUpContextMenu(MouseEvent)` after event/container boundary handling.
    pub fn pop_up_context_menu(&mut self) {
        let mut labels = vec!["Tilt".into()];
        let mut pages = vec!["tilt.html".into()];
        let mut log_labels = Vec::new();
        let mut logs = Vec::new();
        if !self.rb_sirt.selected {
            log_labels.push("Tilt".into());
            logs.push(format!("tilt{}.log", self.axis_id.get_extension()));
            if self.rb_multifilt.selected {
                labels.push("Multifiltsetup".into());
                pages.push("multifiltsetup.html".into());
                log_labels.push("Multifiltsetup".into());
                logs.push(format!(
                    "multifiltsetup{}.log",
                    self.axis_id.get_extension()
                ));
            } else if self.rb_ctf3d.as_ref().is_some_and(|button| button.selected) {
                labels.push("CTF3Dsetup".into());
                pages.push("ctf3dsetup.html".into());
                log_labels.extend(["CTF3Dsetup".into(), "Final CTF 3D".into()]);
                logs.extend([
                    format!("ctf3dsetup{}.log", self.axis_id.get_extension()),
                    format!("ctf3dfinish{}.log", self.axis_id.get_extension()),
                ]);
            }
        } else {
            log_labels.push("Final SIRT".into());
            logs.push(format!(
                "tilt{}_sirt-finish.log",
                self.axis_id.get_extension()
            ));
            labels.push("Sirtsetup".into());
            pages.push("sirtsetup.html".into());
        }
        labels.push("3dmod".into());
        pages.push("3dmod.html".into());
        self.last_context_popup = Some(TomogramGenerationContextPopup {
            title: "TOMOGRAM GENERATION".into(),
            man_page_labels: labels,
            man_pages: pages,
            log_file_labels: log_labels,
            log_files: logs,
            axis_id: self.axis_id,
        });
    }
    pub fn set_tilt_state(
        &mut self,
        state: &TomogramStateBoundary,
        meta_data: &dyn TomogramGenerationMetaData,
    ) {
        self.tilt_panel.set_state(state, meta_data);
    }
    pub fn done(&mut self) {
        self.expert.done_dialog();
        self.tilt_panel.done();
        self.multifilt_panel.done();
        self.sirt_panel.done();
        if let Some(panel) = &mut self.ctf3d_panel {
            panel.done();
        }
        if let Some(panel) = &mut self.method_plugin_panel {
            panel.done();
        }
        self.layout.displayed = false;
    }
    pub fn add_axis_tilt_focus_listener(&mut self, add: bool) {
        self.tilt_panel.focus_listener_added(add);
    }
    pub fn add_use_local_alignment_action_listener(&mut self, add: bool) {
        self.tilt_panel.local_alignment_listener_added(add);
    }
    pub fn add_use_z_factors_action_listener(&mut self, add: bool) {
        self.tilt_panel.z_factors_listener_added(add);
    }
    pub fn get_x_axis_tilt(&self) -> String {
        self.tilt_panel.x_axis_tilt()
    }
    pub fn is_use_local_alignment(&self) -> bool {
        self.tilt_panel.use_local_alignment()
    }
    pub fn is_use_z_factors(&self) -> bool {
        self.tilt_panel.use_z_factors()
    }
    pub fn get_tomo_thickness(&self) -> Option<i64> {
        self.tilt_panel.tomo_thickness()
    }
    /// Java private `methodChanged()`.
    fn method_changed(&mut self) {
        self.tilt_panel.msg_method_changed();
        self.multifilt_panel.msg_method_changed();
        self.sirt_panel.msg_method_changed();
        if let Some(panel) = &mut self.ctf3d_panel {
            panel.msg_method_changed();
        }
        let plugin = self.is_method_plugin();
        if let Some(panel) = &mut self.method_plugin_panel {
            panel.msg_visibility_changed(plugin);
        }
        self.ui_harness.pack(self.axis_id);
        self.ui_harness.move_sub_frame();
    }
    pub fn get_processing_method(&self) -> ProcessingMethod {
        self.tilt_panel.processing_method()
    }
    pub fn display_multifilt(&mut self) {
        self.rb_multifilt.do_click();
        self.method_changed();
    }
    /// Java private `action(ActionEvent)` with action command substituted for the event boundary.
    pub fn action(&mut self, action_command: &str) {
        if [
            self.rb_back_projection.action_command(),
            self.rb_multifilt.action_command(),
            self.rb_sirt.action_command(),
        ]
        .contains(&action_command)
            || self
                .rb_ctf3d
                .as_ref()
                .is_some_and(|button| button.action_command() == action_command)
            || self
                .rb_method_plugin
                .as_ref()
                .is_some_and(|button| button.action_command() == action_command)
        {
            self.method_changed();
        }
    }
    #[allow(non_snake_case)]
    /// Native-name adapter for the Java listener's `actionPerformed`.
    pub fn actionPerformed(&mut self, action_command: &str) {
        self.action(action_command);
    }
}
impl<T, M, S, C, P, E, H> TomogramGenerationParent for TomogramGenerationDialog<T, M, S, C, P, E, H>
where
    T: TomogramGenerationTiltPanel,
    M: TomogramGenerationMultifiltPanel,
    S: TomogramGenerationSirtPanel,
    C: TomogramGenerationCtf3dPanel,
    P: TomogramGenerationMethodPluginPanel,
    E: TomogramGenerationExpert,
    H: TomogramGenerationUiHarness,
{
    fn is_ctf3d(&self) -> bool {
        self.rb_ctf3d.as_ref().is_some_and(|v| v.selected)
    }
    fn is_method_plugin(&self) -> bool {
        self.rb_method_plugin.as_ref().is_some_and(|v| v.selected)
    }
    fn is_multifilt(&self) -> bool {
        self.rb_multifilt.selected
    }
    fn is_back_projection(&self) -> bool {
        self.rb_back_projection.selected
    }
    fn is_sirt(&self) -> bool {
        self.rb_sirt.selected
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct Meta {
        multi: bool,
    }
    impl TomogramGenerationMetaData for Meta {
        fn is_gen_back_projection(&self, _: AxisID) -> bool {
            false
        }
        fn is_gen_filter_trials(&self, _: AxisID) -> bool {
            self.multi
        }
        fn is_gen_sirt(&self, _: AxisID) -> bool {
            false
        }
        fn set_gen_back_projection(&mut self, _: AxisID, _: bool) {}
        fn set_gen_filter_trials(&mut self, _: AxisID, v: bool) {
            self.multi = v
        }
        fn set_gen_sirt(&mut self, _: AxisID, _: bool) {}
    }
    #[derive(Default)]
    struct T {
        calls: usize,
    }
    impl TomogramGenerationTiltPanel for T {
        fn root_name(&self) -> String {
            "tilt".into()
        }
        fn set_parameters_metadata(&mut self, _: &dyn TomogramGenerationMetaData) {}
        fn get_parameters_metadata(
            &mut self,
            _: &mut dyn TomogramGenerationMetaData,
        ) -> Result<(), String> {
            Ok(())
        }
        fn set_parameters_tilt(&mut self, _: &ConstTiltParamBoundary, _: bool) {}
        fn set_parameters_screen_state(&mut self, _: &ReconScreenStateBoundary) {}
        fn get_parameters_screen_state(&mut self, _: &mut ReconScreenStateBoundary) {}
        fn update_display(&mut self) {
            self.calls += 1
        }
        fn done(&mut self) {
            self.calls += 1
        }
        fn checkpoint(&mut self, _: &TomogramStateBoundary) {}
        fn set_state(&mut self, _: &TomogramStateBoundary, _: &dyn TomogramGenerationMetaData) {}
        fn msg_method_changed(&mut self) {
            self.calls += 1
        }
        fn set_filter_type_action_listener(&mut self) {
            self.calls += 1
        }
        fn x_axis_tilt(&self) -> String {
            "1".into()
        }
        fn use_local_alignment(&self) -> bool {
            true
        }
        fn use_z_factors(&self) -> bool {
            false
        }
        fn tomo_thickness(&self) -> Option<i64> {
            Some(5)
        }
        fn processing_method(&self) -> ProcessingMethod {
            ProcessingMethod::LocalCpu
        }
        fn focus_listener_added(&mut self, _: bool) {}
        fn local_alignment_listener_added(&mut self, _: bool) {}
        fn z_factors_listener_added(&mut self, _: bool) {}
    }
    macro_rules! panel {
        ($n:ident,$trait:ident,$root:expr) => {
            #[derive(Default)]
            struct $n;
            impl $trait for $n {
                fn $root(&self) -> String {
                    stringify!($n).into()
                }
                fn set_parameters_metadata(&mut self, _: &dyn TomogramGenerationMetaData) {}
                fn get_parameters_metadata(
                    &mut self,
                    _: &mut dyn TomogramGenerationMetaData,
                ) -> Result<(), String> {
                    Ok(())
                }
                fn set_parameters_screen_state(&mut self, _: &ReconScreenStateBoundary) {}
                fn get_parameters_screen_state(&mut self, _: &mut ReconScreenStateBoundary) {}
                fn done(&mut self) {}
                fn msg_method_changed(&mut self) {}
            }
        };
    }
    // The three direct panel interfaces differ in one source-specific parameter method.
    #[derive(Default)]
    struct M;
    impl TomogramGenerationMultifiltPanel for M {
        fn component_name(&self) -> String {
            "multi".into()
        }
        fn set_parameters_metadata(&mut self, _: &dyn TomogramGenerationMetaData) {}
        fn get_parameters_metadata(
            &mut self,
            _: &mut dyn TomogramGenerationMetaData,
        ) -> Result<(), String> {
            Ok(())
        }
        fn set_parameters(&mut self, _: &MultifiltSetupParamBoundary) {}
        fn set_parameters_screen_state(&mut self, _: &ReconScreenStateBoundary) {}
        fn get_parameters_screen_state(&mut self, _: &mut ReconScreenStateBoundary) {}
        fn done(&mut self) {}
        fn msg_method_changed(&mut self) {}
    }
    #[derive(Default)]
    struct S;
    impl TomogramGenerationSirtPanel for S {
        fn root_name(&self) -> String {
            "sirt".into()
        }
        fn set_parameters_metadata(&mut self, _: &dyn TomogramGenerationMetaData) {}
        fn get_parameters_metadata(
            &mut self,
            _: &mut dyn TomogramGenerationMetaData,
        ) -> Result<(), String> {
            Ok(())
        }
        fn set_parameters(&mut self, _: &SirtsetupParamBoundary) {}
        fn set_parameters_screen_state(&mut self, _: &ReconScreenStateBoundary) {}
        fn get_parameters_screen_state(&mut self, _: &mut ReconScreenStateBoundary) {}
        fn update_display(&mut self) {}
        fn done(&mut self) {}
        fn checkpoint(&mut self, _: &TomogramStateBoundary) {}
        fn msg_sirt_succeeded(&mut self) {}
        fn msg_method_changed(&mut self) {}
    }
    #[derive(Default)]
    struct C;
    impl TomogramGenerationCtf3dPanel for C {
        fn component_name(&self) -> String {
            "ctf".into()
        }
        fn set_parameters_metadata(&mut self, _: &dyn TomogramGenerationMetaData) {}
        fn get_parameters_metadata(
            &mut self,
            _: &mut dyn TomogramGenerationMetaData,
        ) -> Result<(), String> {
            Ok(())
        }
        fn set_parameters(&mut self, _: &Ctf3dSetupParamBoundary) {}
        fn set_parameters_screen_state(&mut self, _: &ReconScreenStateBoundary) {}
        fn get_parameters_screen_state(&mut self, _: &mut ReconScreenStateBoundary) {}
        fn update_display(&mut self) {}
        fn done(&mut self) {}
        fn msg_method_changed(&mut self) {}
    }
    #[derive(Default)]
    struct P;
    impl TomogramGenerationMethodPluginPanel for P {
        fn component_name(&self) -> String {
            "plugin".into()
        }
        fn button_title(&self) -> String {
            "Plugin".into()
        }
        fn update_display(&mut self) {}
        fn set_parameters(&mut self, _: &ReconScreenStateBoundary) {}
        fn get_parameters(&mut self, _: &mut ReconScreenStateBoundary) {}
        fn done(&mut self) {}
        fn msg_visibility_changed(&mut self, _: bool) {}
    }
    #[derive(Default)]
    struct E;
    impl TomogramGenerationExpert for E {
        fn done_dialog(&mut self) {}
    }
    #[derive(Default)]
    struct H(usize);
    impl TomogramGenerationUiHarness for H {
        fn pack(&mut self, _: AxisID) {
            self.0 += 1
        }
        fn move_sub_frame(&mut self) {
            self.0 += 1
        }
    }
    #[test]
    fn construction_selection_and_context_menu_follow_source() {
        let mut d = TomogramGenerationDialog::get_instance(
            AxisID::Only,
            T::default(),
            M,
            S,
            Some(C),
            Some(P),
            E,
            H::default(),
        );
        assert_eq!(
            d.layout.root_order,
            vec!["pnlMethod", "tilt", "multi", "sirt", "ctf", "plugin"]
        );
        assert_eq!(d.btn_execute_text, "Done");
        let mut m = Meta { multi: true };
        d.set_parameters_metadata(&m);
        assert!(d.is_multifilt());
        d.get_parameters(&mut m).unwrap();
        d.pop_up_context_menu();
        assert_eq!(
            d.last_context_popup.unwrap().man_page_labels,
            vec!["Tilt", "Multifiltsetup", "3dmod"]
        );
    }
}
