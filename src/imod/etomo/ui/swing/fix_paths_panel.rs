//! `IMOD/Etomo/src/etomo/ui/swing/FixPathsPanel.java`.
//!
//! Swing component realization, the `FileContainer` implementation, the
//! concrete `BaseManager`, and `UIHarness` packing stay at their source
//! boundaries.  This unit retains the Java panel's component order,
//! visibility transitions, listener registration, and file-path action.
#![allow(dead_code)]

use crate::imod::etomo::r#type::{axis_id::AxisID, dialog_type::DialogType};

use super::{
    check_box::CheckBox,
    file_container::FileContainer,
    multi_line_button::MultiLineButton,
    panel_header::{ExpandButton, Expandable, PanelHeader},
    process_control_panel::COLOR_NOT_STARTED,
};

/// Java `UIHarness.INSTANCE.pack(axisID, manager)` presentation boundary.
/// The actual `BaseManager` object remains owned by eTomo's manager layer.
pub trait FixPathsPanelUiHarness<M> {
    fn pack(&mut self, axis_id: AxisID, manager: &M);
}

/// Source-visible Swing construction state of `pnlRoot`, `pnlMain`,
/// `pnlBody`, and the local button panel.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FixPathsPanelLayout {
    pub root_box_layout_x_axis: bool,
    pub root_visible: bool,
    pub root_component_order: Vec<&'static str>,
    pub main_box_layout_y_axis: bool,
    pub main_etched_border: bool,
    pub main_component_order: Vec<&'static str>,
    pub body_box_layout_y_axis: bool,
    pub body_visible: bool,
    pub body_component_order: Vec<&'static str>,
    pub button_box_layout_x_axis: bool,
    pub button_has_left_horizontal_glue: bool,
    pub button_has_right_horizontal_glue: bool,
    pub main_maximum_size_is_preferred: bool,
}

impl Default for FixPathsPanelLayout {
    fn default() -> Self {
        Self {
            root_box_layout_x_axis: false,
            root_visible: true,
            root_component_order: Vec::new(),
            main_box_layout_y_axis: false,
            main_etched_border: false,
            main_component_order: Vec::new(),
            body_box_layout_y_axis: false,
            body_visible: true,
            body_component_order: Vec::new(),
            button_box_layout_x_axis: false,
            button_has_left_horizontal_glue: false,
            button_has_right_horizontal_glue: false,
            main_maximum_size_is_preferred: false,
        }
    }
}

/// Java `JLabel lblwarning` fields which are owned by this source unit.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FixPathsWarningLabel {
    pub text: &'static str,
    pub foreground: (u8, u8, u8),
    pub visible: bool,
}

/// Java final `FixPathsPanel`.
#[derive(Clone, Debug)]
pub struct FixPathsPanel {
    pub pnl_root: FixPathsPanelLayout,
    pub lbl_warning: FixPathsWarningLabel,
    pub cb_choose_path_every_row: CheckBox,
    pub bn_fix_paths: MultiLineButton,
    pub header: PanelHeader,
    /// Source final `fileContainer`; implementation is passed at the actual
    /// action boundary so this value model has no artificial ownership layer.
    pub file_container_present: bool,
    /// Source final `manager`; manager is supplied to `expand` with UIHarness.
    pub manager_present: bool,
    pub axis_id: AxisID,
    pub dialog_type: DialogType,
}

impl FixPathsPanel {
    /// Java private `FixPathsPanel(FileContainer, BaseManager, AxisID, DialogType)`.
    pub fn new(axis_id: AxisID, dialog_type: DialogType) -> Self {
        Self {
            pnl_root: FixPathsPanelLayout {
                root_box_layout_x_axis: true,
                root_visible: false,
                root_component_order: vec!["pnlMain"],
                main_box_layout_y_axis: true,
                main_etched_border: true,
                main_component_order: vec!["header", "pnlBody"],
                body_box_layout_y_axis: true,
                body_visible: true,
                body_component_order: vec!["lblwarning", "cbChoosePathEveryRow", "pnlButton"],
                button_box_layout_x_axis: true,
                button_has_left_horizontal_glue: true,
                button_has_right_horizontal_glue: true,
                main_maximum_size_is_preferred: true,
            },
            lbl_warning: FixPathsWarningLabel {
                text: "Files cannot be found.  PEET may not run.",
                foreground: COLOR_NOT_STARTED,
                visible: true,
            },
            cb_choose_path_every_row: CheckBox::new_with_text(
                "Files may be in separate directories",
            ),
            bn_fix_paths: MultiLineButton::new_with_label(Some("Fix Incorrect Paths")),
            header: PanelHeader::new(
                "Fix File Paths",
                false,
                false,
                dialog_type,
                true,
                false,
                true,
                false,
                true,
            ),
            file_container_present: true,
            manager_present: true,
            axis_id,
            dialog_type,
        }
    }

    /// Java static `getInstance(...)`.
    pub fn get_instance(axis_id: AxisID, dialog_type: DialogType) -> Self {
        let mut instance = Self::new(axis_id, dialog_type);
        instance.set_tooltips();
        instance.add_listeners();
        instance
    }

    /// Java private `setTooltips()`.
    pub fn set_tooltips(&mut self) {
        self.cb_choose_path_every_row.set_tool_tip_text(Some(
            "Causes a file chooser to be brought up for each row in the Volume Table.  Otherwise the file chooser will only be brought up when a file cannot be found in either the original path or the most recent new path.",
        ));
        self.bn_fix_paths.set_tool_tip_text(Some(
            "Brings up file chooser(s) so that the new location(s) of any files that cannot be found can be specified.",
        ));
    }

    /// Java private `addListeners()`.
    pub fn add_listeners(&mut self) {
        self.bn_fix_paths.add_action_listener();
    }

    /// Java `getRootComponent()` at the Swing component boundary.
    pub fn get_root_component(&self) -> &FixPathsPanelLayout {
        &self.pnl_root
    }

    /// Java `setIncorrectPaths(boolean)`.
    pub fn set_incorrect_paths(&mut self, incorrect_paths: bool) {
        if incorrect_paths {
            self.pnl_root.root_visible = true;
            self.lbl_warning.visible = true;
        } else {
            self.lbl_warning.visible = false;
        }
    }

    /// Java private `action()` at its `FileContainer` boundary.
    pub fn action<F: FileContainer>(&self, file_container: &mut F) {
        file_container.fix_incorrect_paths(self.cb_choose_path_every_row.is_selected());
    }

    /// Java inner `FixPathsPanelListener.actionPerformed(ActionEvent)`.
    pub fn action_performed<F: FileContainer>(&self, file_container: &mut F) {
        self.action(file_container);
    }

    /// Java `expand(ExpandButton)`, with the UI harness and concrete manager
    /// retained as direct integration boundaries.
    pub fn expand<M, H: FixPathsPanelUiHarness<M>>(
        &mut self,
        button: &ExpandButton,
        ui_harness: &mut H,
        manager: &M,
    ) {
        if self.header.equals_open_close(button) {
            self.pnl_root.body_visible = button.is_expanded();
        }
        ui_harness.pack(self.axis_id, manager);
    }

    /// Java `expand(GlobalExpandButton)`, intentionally empty.
    pub fn expand_global_button(&mut self) {}
}

impl Expandable for FixPathsPanel {
    fn expand_expand_button(&mut self, button: &ExpandButton) {
        if self.header.equals_open_close(button) {
            self.pnl_root.body_visible = button.is_expanded();
        }
    }

    fn expand_global_button(&mut self, _: &super::process_dialog::GlobalExpandButton) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Container {
        selected: Option<bool>,
    }
    impl FileContainer for Container {
        fn fix_incorrect_paths(&mut self, choose_path_every_row: bool) {
            self.selected = Some(choose_path_every_row);
        }
    }

    #[derive(Default)]
    struct Harness {
        packed: Vec<AxisID>,
    }
    impl FixPathsPanelUiHarness<()> for Harness {
        fn pack(&mut self, axis_id: AxisID, _: &()) {
            self.packed.push(axis_id);
        }
    }

    #[test]
    fn instance_retains_source_hierarchy_tooltips_and_listener() {
        let panel = FixPathsPanel::get_instance(AxisID::Second, DialogType::Peet);
        assert!(!panel.pnl_root.root_visible);
        assert_eq!(panel.pnl_root.root_component_order, ["pnlMain"]);
        assert_eq!(panel.pnl_root.main_component_order, ["header", "pnlBody"]);
        assert_eq!(
            panel.pnl_root.body_component_order,
            ["lblwarning", "cbChoosePathEveryRow", "pnlButton"]
        );
        assert_eq!(panel.bn_fix_paths.button.action_listener_count, 1);
        assert!(panel.cb_choose_path_every_row.check_box.tooltip.is_some());
        assert!(panel.bn_fix_paths.button.tooltip.is_some());
    }

    #[test]
    fn incorrect_paths_visibility_follows_source_asymmetry() {
        let mut panel = FixPathsPanel::get_instance(AxisID::Only, DialogType::Peet);
        panel.set_incorrect_paths(false);
        assert!(!panel.lbl_warning.visible);
        assert!(!panel.pnl_root.root_visible);
        panel.set_incorrect_paths(true);
        assert!(panel.pnl_root.root_visible);
        assert!(panel.lbl_warning.visible);
        panel.set_incorrect_paths(false);
        assert!(panel.pnl_root.root_visible);
        assert!(!panel.lbl_warning.visible);
    }

    #[test]
    fn listener_routes_checkbox_selection_to_file_container() {
        let mut panel = FixPathsPanel::get_instance(AxisID::First, DialogType::Peet);
        panel.cb_choose_path_every_row.set_selected(true);
        let mut container = Container::default();
        panel.action_performed(&mut container);
        assert_eq!(container.selected, Some(true));
    }

    #[test]
    fn open_close_hides_body_and_packs_with_source_axis() {
        let mut panel = FixPathsPanel::get_instance(AxisID::Second, DialogType::Peet);
        let mut button = panel.header.btn_open_close.clone().unwrap();
        button.update(false);
        let mut harness = Harness::default();
        panel.expand(&button, &mut harness, &());
        assert!(!panel.pnl_root.body_visible);
        assert_eq!(
            panel.pnl_root.body_component_order,
            ["lblwarning", "cbChoosePathEveryRow", "pnlButton"]
        );
        assert_eq!(harness.packed, [AxisID::Second]);
    }
}
