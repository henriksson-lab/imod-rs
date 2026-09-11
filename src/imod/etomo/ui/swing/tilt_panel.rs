//! `IMOD/Etomo/src/etomo/ui/swing/TiltPanel.java`.
//!
//! Java inheritance is represented by the owned `abstract_tilt_panel` field.
//! The Swing `JPanel`, `ContextPopup`, and `ApplicationManager` remain direct
//! presentation/application boundaries; this unit preserves every state change
//! and call argument made by `TiltPanel.java` instead of supplying a different
//! reconstruction policy.
#![allow(dead_code)]

use super::abstract_tilt_panel::AbstractTiltPanel;
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::processing_method::ProcessingMethod;

/// Java `PanelId.TILT`.
pub const PANEL_ID_TILT: &str = "Tilt";
/// Java `BoxLayout.Y_AXIS`.
pub const Y_AXIS: i32 = 1;
/// Java `Component.CENTER_ALIGNMENT`.
pub const CENTER_ALIGNMENT: f32 = 0.5;
/// Java `ContextPopup.TOMO_GUIDE`.
pub const TOMO_GUIDE: &str = "TOMO_GUIDE";

/// Direct source dependency `ProcessResultDisplay`.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ProcessResultDisplay;

/// Direct source dependency `Deferred3dmodButton`.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Deferred3dmodButton;

/// Direct source dependency `MouseEvent`; a native event is intentionally not
/// invented here because `TiltPanel` only passes it to `ContextPopup`.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MouseEvent {
    pub x: i32,
    pub y: i32,
    pub popup_trigger: bool,
}

/// Source-visible `JPanel` state of `pnlTiltPanelRoot`.
#[derive(Clone, Debug, PartialEq)]
pub struct TiltPanelRoot {
    pub box_layout_axis: i32,
    pub y_space_before_abstract_root: bool,
    pub abstract_root_added: bool,
    pub alignment_x: f32,
}

impl Default for TiltPanelRoot {
    fn default() -> Self {
        Self {
            box_layout_axis: Y_AXIS,
            y_space_before_abstract_root: false,
            abstract_root_added: false,
            alignment_x: CENTER_ALIGNMENT,
        }
    }
}

/// Exact `ContextPopup` construction inputs retained at its unported GUI boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TiltContextPopup {
    pub anchor: String,
    pub mouse_event: MouseEvent,
    pub guide: &'static str,
    pub man_page_label: [String; 2],
    pub man_page: [String; 2],
    pub log_file_label: [String; 1],
    pub log_file: [String; 1],
    pub axis_id: AxisID,
}

/// The two `ApplicationManager` calls made only by this Java source unit.
pub trait TiltPanelApplicationManager {
    fn tilt_action(
        &mut self,
        process_result_display: &ProcessResultDisplay,
        deferred_3dmod_button: &Deferred3dmodButton,
        run_3dmod_menu_options: Run3dmodMenuOptions,
        axis_id: AxisID,
        dialog_type: DialogType,
        tilt_processing_method: ProcessingMethod,
    );

    fn imod_full_volume(&mut self, axis_id: AxisID, run_3dmod_menu_options: Run3dmodMenuOptions);
}

/// Java `TiltPanel`, composed with its Java superclass.
pub struct TiltPanel {
    pub abstract_tilt_panel: AbstractTiltPanel,
    pub pnl_tilt_panel_root: TiltPanelRoot,
    pub last_context_popup: Option<TiltContextPopup>,
    pub root_tooltip_set: bool,
}

impl TiltPanel {
    /// Java protected `TiltPanel(ApplicationManager, AxisID, DialogType,
    /// GlobalExpandButton, PanelId, TomogramGenerationParent)` constructor.
    /// The manager/global button/parent are direct dependencies already owned by
    /// their caller/source units; the superclass retains their local state.
    pub fn new(axis_id: AxisID, dialog_type: DialogType, panel_id: impl Into<String>) -> Self {
        Self {
            abstract_tilt_panel: AbstractTiltPanel::new(axis_id, dialog_type, panel_id, false),
            pnl_tilt_panel_root: TiltPanelRoot::default(),
            last_context_popup: None,
            root_tooltip_set: false,
        }
    }

    /// Java static `getInstance(...)`.
    pub fn get_instance(axis_id: AxisID, dialog_type: DialogType) -> Self {
        let mut instance = Self::new(axis_id, dialog_type, PANEL_ID_TILT);
        instance.create_panel();
        instance.set_tool_tip_text();
        instance.add_listeners();
        instance
    }

    /// Java override `addListeners`.
    pub fn add_listeners(&mut self) {
        self.abstract_tilt_panel.add_listeners();
        if self.abstract_tilt_panel.listen_for_field_changes {
            // Empty Java branch: subclasses with field listeners set this flag.
        }
    }

    /// Java override `createPanel`.
    pub fn create_panel(&mut self) {
        self.abstract_tilt_panel.create_panel();
        self.pnl_tilt_panel_root.box_layout_axis = Y_AXIS;
        self.pnl_tilt_panel_root.y_space_before_abstract_root = true;
        self.pnl_tilt_panel_root.abstract_root_added = self.abstract_tilt_panel.get_root();
        self.pnl_tilt_panel_root.alignment_x = CENTER_ALIGNMENT;
    }

    /// Java override `getRoot`.
    pub fn get_root(&self) -> &TiltPanelRoot {
        &self.pnl_tilt_panel_root
    }

    /// Java deprecated `allowTiltComSave`.
    pub fn allow_tilt_com_save(&self) -> bool {
        true
    }

    /// Java deprecated `isResume`.
    pub fn is_resume(&self) -> bool {
        false
    }

    /// Java deprecated `msgResumeChanged(boolean)`.
    pub fn msg_resume_changed(&mut self, _resume: bool) {}

    /// Java override `updateDisplay`; Z shift is an advanced field in this subclass.
    pub fn update_display(&mut self) {
        self.abstract_tilt_panel.update_display();
        self.abstract_tilt_panel.ltf_z_shift.visible = self.abstract_tilt_panel.is_advanced();
    }

    /// Java override `setAdvancedFieldDisplayer`.  `null` for the first Java
    /// displayer is retained as `None`; the native control adapter is its direct boundary.
    pub fn set_advanced_field_displayer(&mut self) {
        self.abstract_tilt_panel.set_advanced_field_displayer();
    }

    /// Java override `tiltAction`.
    pub fn tilt_action<M: TiltPanelApplicationManager>(
        &mut self,
        manager: &mut M,
        process_result_display: &ProcessResultDisplay,
        deferred_3dmod_button: &Deferred3dmodButton,
        run_3dmod_menu_options: Run3dmodMenuOptions,
        tilt_processing_method: ProcessingMethod,
    ) {
        manager.tilt_action(
            process_result_display,
            deferred_3dmod_button,
            run_3dmod_menu_options,
            self.abstract_tilt_panel.axis_id,
            self.abstract_tilt_panel.dialog_type,
            tilt_processing_method,
        );
    }

    /// Java override `imodTomogramAction`.
    pub fn imod_tomogram_action<M: TiltPanelApplicationManager>(
        &mut self,
        manager: &mut M,
        _deferred_3dmod_button: &Deferred3dmodButton,
        run_3dmod_menu_options: Run3dmodMenuOptions,
    ) {
        manager.imod_full_volume(self.abstract_tilt_panel.axis_id, run_3dmod_menu_options);
    }

    /// Java `popUpContextMenu(String, Component, MouseEvent)`.
    pub fn pop_up_context_menu(&mut self, anchor: impl Into<String>, mouse_event: MouseEvent) {
        self.last_context_popup = Some(TiltContextPopup {
            anchor: anchor.into(),
            mouse_event,
            guide: TOMO_GUIDE,
            man_page_label: ["Tilt".into(), "3dmod".into()],
            man_page: ["tilt.html".into(), "3dmod.html".into()],
            log_file_label: ["Tilt".into()],
            log_file: [format!(
                "tilt{}.log",
                self.abstract_tilt_panel.axis_id.get_extension()
            )],
            axis_id: self.abstract_tilt_panel.axis_id,
        });
    }

    /// Java override `setToolTipText`.
    pub fn set_tool_tip_text(&mut self) {
        self.root_tooltip_set = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Manager {
        tilt: Option<(AxisID, DialogType, ProcessingMethod)>,
        imod: Option<AxisID>,
    }
    impl TiltPanelApplicationManager for Manager {
        fn tilt_action(
            &mut self,
            _display: &ProcessResultDisplay,
            _button: &Deferred3dmodButton,
            _options: Run3dmodMenuOptions,
            axis_id: AxisID,
            dialog_type: DialogType,
            method: ProcessingMethod,
        ) {
            self.tilt = Some((axis_id, dialog_type, method));
        }
        fn imod_full_volume(&mut self, axis_id: AxisID, _options: Run3dmodMenuOptions) {
            self.imod = Some(axis_id);
        }
    }

    #[test]
    fn get_instance_follows_source_creation_order() {
        let panel = TiltPanel::get_instance(AxisID::First, DialogType::TomogramGeneration);
        assert!(panel.pnl_tilt_panel_root.abstract_root_added);
        assert!(panel.pnl_tilt_panel_root.y_space_before_abstract_root);
        assert!(panel.root_tooltip_set);
        assert_eq!(panel.abstract_tilt_panel.btn_tilt.listener_count, 1);
    }

    #[test]
    fn subclass_z_shift_and_context_popup_follow_source() {
        let mut panel = TiltPanel::get_instance(AxisID::Second, DialogType::TomogramGeneration);
        panel.abstract_tilt_panel.header_advanced = false;
        panel.update_display();
        assert!(!panel.abstract_tilt_panel.ltf_z_shift.visible);
        panel.pop_up_context_menu("tilt", MouseEvent::default());
        assert_eq!(
            panel.last_context_popup.unwrap().log_file,
            ["tiltb.log".to_string()]
        );
    }

    #[test]
    fn manager_actions_keep_the_java_arguments() {
        let mut panel = TiltPanel::get_instance(AxisID::Only, DialogType::TomogramGeneration);
        let mut manager = Manager::default();
        panel.tilt_action(
            &mut manager,
            &ProcessResultDisplay,
            &Deferred3dmodButton,
            Run3dmodMenuOptions::default(),
            ProcessingMethod::LocalCpu,
        );
        panel.imod_tomogram_action(
            &mut manager,
            &Deferred3dmodButton,
            Run3dmodMenuOptions::default(),
        );
        assert_eq!(manager.tilt.unwrap().2, ProcessingMethod::LocalCpu);
        assert_eq!(manager.imod, Some(AxisID::Only));
    }
}
