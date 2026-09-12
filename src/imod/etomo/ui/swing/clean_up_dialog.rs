//! `IMOD/Etomo/src/etomo/ui/swing/CleanUpDialog.java`.
//!
//! Swing label/component construction and `UIHarness` packing remain direct GUI
//! boundaries.  This unit owns the source dialog state, archive-label decisions,
//! action-command routing, and context-popup construction.
#![allow(dead_code)]

use super::{
    beveled_border::BeveledBorder,
    cleanup_panel::{CleanupApplicationManager, CleanupPanel},
    context_popup::{ContextPopup, MouseEvent},
    etomo_frame::ActionEvent,
    multi_line_button::MultiLineButton,
    process_dialog::{ProcessDialog, ProcessDialogApplicationManager},
};
use crate::imod::etomo::r#type::{axis_id::AxisID, axis_type::AxisType, dialog_type::DialogType};

/// Source-visible state sent to Java `JLabel` at the Swing boundary.
#[derive(Clone, Debug, PartialEq)]
pub struct ArchiveInfoLabel {
    pub text: String,
    pub visible: bool,
    pub alignment_x: f32,
}

impl Default for ArchiveInfoLabel {
    fn default() -> Self {
        Self {
            text: String::new(),
            visible: true,
            alignment_x: 0.5,
        }
    }
}

/// Direct `ApplicationManager` and `UIHarness.INSTANCE.pack` operations used by
/// `CleanUpDialog`.  The concrete manager and GUI toolkit stay outside this unit.
pub trait CleanUpDialogApplicationManager:
    CleanupApplicationManager + ProcessDialogApplicationManager
{
    fn archive_info(&self, axis_id: AxisID) -> Option<String>;
    fn done_clean_up(&mut self);
    fn archive_original_stack(&mut self, process_series: Option<()>, dialog_type: DialogType);
    fn pack(&mut self, axis_id: AxisID);
}

/// Java `CleanUpDialog` fields and methods.  `process_dialog` is the direct
/// superclass state; `root_border` records its `BeveledBorder` presentation
/// boundary because `EtomoPanel` only has the narrower translated titled-border
/// representation.
pub struct CleanUpDialog<'a> {
    pub process_dialog: ProcessDialog<'a>,
    pub cleanup_panel: CleanupPanel,
    pub btn_archive_stack: Option<MultiLineButton>,
    pub archive_info_a: ArchiveInfoLabel,
    pub archive_info_b: ArchiveInfoLabel,
    pub axis_type: AxisType,
    pub root_panel_box_layout_y_axis: bool,
    pub root_border: BeveledBorder,
    pub root_component_order: Vec<&'static str>,
    pub advanced_button_visible: bool,
    pub mouse_adapter_present: bool,
    pub context_popup: Option<ContextPopup>,
}

impl<'a> CleanUpDialog<'a> {
    /// Java `CleanUpDialog(ApplicationManager)`.
    pub fn new<M: CleanUpDialogApplicationManager>(application_manager: &'a M) -> Self {
        let axis_type = application_manager.axis_type();
        let mut process_dialog = ProcessDialog::new(
            application_manager,
            AxisID::Only,
            DialogType::CleanUp,
            Box::new(|| {}),
        );
        process_dialog.btn_execute.set_text("Done");
        let mut btn_archive_stack = MultiLineButton::new();
        let mut archive_info_b = ArchiveInfoLabel::default();
        if axis_type == AxisType::DualAxis {
            btn_archive_stack.set_text("Archive Original Stacks");
        } else {
            btn_archive_stack.set_text("Archive Original Stack");
            archive_info_b.visible = false;
        }
        btn_archive_stack.add_action_listener();
        btn_archive_stack.set_alignment_x(0.5);
        let mut dialog = Self {
            process_dialog,
            cleanup_panel: CleanupPanel::get_instance(application_manager),
            btn_archive_stack: Some(btn_archive_stack),
            archive_info_a: ArchiveInfoLabel::default(),
            archive_info_b,
            axis_type,
            root_panel_box_layout_y_axis: true,
            root_border: BeveledBorder::new("Clean Up"),
            root_component_order: vec![
                "archive-stack-button",
                "archive-info-a",
                "archive-info-b",
                "cleanup-panel",
                "exit-buttons",
            ],
            advanced_button_visible: false,
            mouse_adapter_present: true,
            context_popup: None,
        };
        dialog.set_archive_fields(application_manager);
        dialog.set_tool_tip_text();
        dialog
    }

    /// Java `updateArchiveDisplay(boolean)`.
    pub fn update_archive_display(&mut self, original_stacks_exist: bool) {
        let Some(btn_archive_stack) = &mut self.btn_archive_stack else {
            return;
        };
        btn_archive_stack.set_enabled(original_stacks_exist);
    }

    /// Java `setArchiveFields()`.
    pub fn set_archive_fields<M: CleanUpDialogApplicationManager>(
        &mut self,
        application_manager: &M,
    ) {
        let archive_info_text = "To restore original stack, run:  archiveorig -r ";
        if self.axis_type == AxisType::DualAxis {
            if let Some(stack_file_name) = application_manager.archive_info(AxisID::First) {
                self.archive_info_a.text = format!("{archive_info_text}{stack_file_name}");
                self.archive_info_a.visible = true;
            } else {
                self.archive_info_a.visible = false;
            }
            if let Some(stack_file_name) = application_manager.archive_info(AxisID::Second) {
                self.archive_info_b.text = format!("{archive_info_text}{stack_file_name}");
                self.archive_info_b.visible = true;
            } else {
                self.archive_info_b.visible = false;
            }
        } else if let Some(stack_file_name) = application_manager.archive_info(AxisID::Only) {
            self.archive_info_a.text =
                format!("To restore original stack run:  archiveorig -r {stack_file_name}");
            self.archive_info_a.visible = true;
        } else {
            self.archive_info_a.visible = false;
        }
    }

    /// Java `popUpContextMenu(MouseEvent)`.
    pub fn pop_up_context_menu(&mut self, mouse_event: MouseEvent) {
        self.context_popup = Some(ContextPopup::new_guide(
            mouse_event,
            Some("Cleaning Up"),
            super::context_popup::TOMO_GUIDE,
            self.process_dialog.axis_id,
        ));
    }

    /// Java override `done()`.
    pub fn done<M: CleanUpDialogApplicationManager>(&mut self, application_manager: &mut M) {
        application_manager.done_clean_up();
        self.process_dialog.set_displayed(false);
        application_manager.pack(self.process_dialog.axis_id);
    }

    /// Java `buttonAction(ActionEvent)`.
    pub fn button_action<M: CleanUpDialogApplicationManager>(
        &mut self,
        event: &ActionEvent,
        application_manager: &mut M,
    ) {
        if self
            .btn_archive_stack
            .as_ref()
            .and_then(MultiLineButton::get_text)
            .is_some_and(|text| event.action_command == text)
        {
            application_manager.archive_original_stack(None, self.process_dialog.dialog_type);
        }
    }

    /// Java private `setToolTipText()`.
    pub fn set_tool_tip_text(&mut self) {
        if let Some(btn_archive_stack) = &mut self.btn_archive_stack {
            btn_archive_stack.set_tool_tip_text(Some(
                "Run archiveorig.  Archiveorig creates a _xray.mrc.gz file, which contains the difference between the .mrc file and the _orig.mrc  file.  If archiveorig succeeds, then you can delete the _orig.mrc file.  To restore _orig.mrc, go to the directory containing the _xray.mrc.gz file and run \"archiveorig -r\" on the .mrc file.",
            ));
        }
    }
}

/// Java private `ButtonActionListener`; frontend event delivery remains a GUI
/// boundary and delegates only to `CleanUpDialog.buttonAction`.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ButtonActionListener;

impl ButtonActionListener {
    /// Java `ButtonActionListener(CleanUpDialog)`.
    pub fn new() -> Self {
        Self
    }

    /// Java `actionPerformed(ActionEvent)`.
    pub fn action_performed<M: CleanUpDialogApplicationManager>(
        &self,
        adaptee: &mut CleanUpDialog<'_>,
        event: &ActionEvent,
        application_manager: &mut M,
    ) {
        adaptee.button_action(event, application_manager);
    }
}

#[cfg(test)]
mod tests {
    use std::cell::{Cell, RefCell};
    use std::path::Path;

    use super::*;
    use crate::imod::etomo::r#type::image_filename_style::ImageFilenameStyle;

    struct Manager {
        axis_type: AxisType,
        archive: [Option<String>; 3],
        done: Cell<usize>,
        archive_calls: Cell<usize>,
        packed: Cell<Option<AxisID>>,
        messages: RefCell<Vec<String>>,
    }
    impl ProcessDialogApplicationManager for Manager {
        fn is_advanced(&self, _: DialogType, _: AxisID) -> bool {
            false
        }
    }
    impl CleanupApplicationManager for Manager {
        fn property_user_dir(&self) -> &Path {
            Path::new("/definitely-not-an-imod-dataset")
        }
        fn dataset_name(&self) -> &str {
            "set"
        }
        fn image_filename_style(&self) -> ImageFilenameStyle {
            ImageFilenameStyle::Mrc
        }
        fn axis_type(&self) -> AxisType {
            self.axis_type
        }
        fn trim_vol_output_file_name(&self, _: AxisID) -> String {
            "trim.mrc".into()
        }
        fn open_message_dialog(&mut self, message: String, _: &str, _: AxisID) {
            self.messages.borrow_mut().push(message);
        }
    }
    impl CleanUpDialogApplicationManager for Manager {
        fn archive_info(&self, axis_id: AxisID) -> Option<String> {
            self.archive[axis_id as usize].clone()
        }
        fn done_clean_up(&mut self) {
            self.done.set(self.done.get() + 1);
        }
        fn archive_original_stack(&mut self, _: Option<()>, _: DialogType) {
            self.archive_calls.set(self.archive_calls.get() + 1);
        }
        fn pack(&mut self, axis_id: AxisID) {
            self.packed.set(Some(axis_id));
        }
    }
    fn manager(axis_type: AxisType) -> Manager {
        Manager {
            axis_type,
            archive: [None, Some("seta.st".into()), Some("setb.st".into())],
            done: Cell::new(0),
            archive_calls: Cell::new(0),
            packed: Cell::new(None),
            messages: RefCell::new(Vec::new()),
        }
    }

    #[test]
    fn dual_axis_constructor_preserves_archive_labels_and_dialog_structure() {
        let manager = manager(AxisType::DualAxis);
        let dialog = CleanUpDialog::new(&manager);
        assert_eq!(
            dialog.btn_archive_stack.as_ref().unwrap().get_text(),
            Some("Archive Original Stacks")
        );
        assert_eq!(
            dialog.archive_info_a.text,
            "To restore original stack, run:  archiveorig -r seta.st"
        );
        assert_eq!(
            dialog.archive_info_b.text,
            "To restore original stack, run:  archiveorig -r setb.st"
        );
        assert!(dialog.archive_info_a.visible && dialog.archive_info_b.visible);
        assert!(!dialog.process_dialog.btn_advanced.expanded);
        assert!(!dialog.advanced_button_visible);
        assert_eq!(
            dialog
                .process_dialog
                .btn_execute
                .multi_line_button
                .get_text(),
            Some("Done")
        );
        assert_eq!(dialog.root_border.get_border().title, "Clean Up");
    }

    #[test]
    fn single_axis_source_wording_and_missing_archive_visibility_are_exact() {
        let manager = manager(AxisType::SingleAxis);
        let dialog = CleanUpDialog::new(&manager);
        assert_eq!(
            dialog.btn_archive_stack.as_ref().unwrap().get_text(),
            Some("Archive Original Stack")
        );
        assert!(!dialog.archive_info_a.visible);
        assert!(!dialog.archive_info_b.visible);
    }

    #[test]
    fn archive_action_done_and_context_popup_follow_source_dispatch() {
        let mut action_manager = manager(AxisType::SingleAxis);
        let manager_ref: &'static Manager = Box::leak(Box::new(manager(AxisType::SingleAxis)));
        let mut dialog = CleanUpDialog::new(manager_ref);
        dialog.update_archive_display(false);
        assert!(!dialog.btn_archive_stack.as_ref().unwrap().is_enabled());
        dialog.button_action(
            &ActionEvent::new("Archive Original Stack"),
            &mut action_manager,
        );
        dialog.button_action(&ActionEvent::new("unrelated"), &mut action_manager);
        assert_eq!(action_manager.archive_calls.get(), 1);
        dialog.done(&mut action_manager);
        assert_eq!(action_manager.done.get(), 1);
        assert!(!dialog.process_dialog.is_displayed());
        assert_eq!(action_manager.packed.get(), Some(AxisID::Only));
        dialog.pop_up_context_menu(MouseEvent { x: 2, y: 3 });
        let popup = dialog.context_popup.unwrap();
        assert_eq!(popup.anchor.as_deref(), Some("Cleaning Up"));
        assert_eq!(
            popup.guide_to_anchor.as_deref(),
            Some(super::super::context_popup::TOMO_GUIDE)
        );
    }
}
