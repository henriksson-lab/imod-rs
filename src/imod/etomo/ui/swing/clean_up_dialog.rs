//! `IMOD/Etomo/src/etomo/ui/swing/CleanUpDialog.java`.
//!
//! Java `public class CleanUpDialog extends ProcessDialog implements
//! ContextMenu`: the Clean Up dialog - archive the original stack(s) and
//! delete intermediate files (`CleanupPanel`).
//!
//! An EDT object (`ui.md`): created as `Rc<Self>` by [`CleanUpDialog::new`];
//! every method takes `&self`.  The `ProcessDialog` superclass is the embedded
//! `base` (reached through `Deref`) and the overridden `done()` is
//! `ProcessDialogVirtual::done`.  The listener class `ButtonActionListener` is
//! a closure holding a weak reference to the dialog.

use std::cell::OnceCell;
use std::ops::Deref;
use std::rc::{Rc, Weak};

use super::beveled_border::BeveledBorder;
use super::cleanup_panel::CleanupPanel;
use super::context_menu::ContextMenu;
use super::context_popup::{self, ContextPopup};
use super::multi_line_button::MultiLineButton;
use super::process_dialog::{ProcessDialog, ProcessDialogVirtual};
use super::ui_harness;
use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::jdk::{ActionEvent, ActionListener, JComponent, MouseEvent};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::const_meta_data::ConstMetaData;
use crate::imod::etomo::r#type::dialog_type::DialogType;

/// Java public static final `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `public class CleanUpDialog extends ProcessDialog implements
/// ContextMenu`.
pub struct CleanUpDialog {
    /// The `ProcessDialog` superclass.
    base: Rc<ProcessDialog>,

    /// Java private `cleanupPanel` (set by the constructor).
    cleanup_panel: OnceCell<Rc<CleanupPanel>>,
    /// Java private `btnArchiveStack = new MultiLineButton()`.
    btn_archive_stack: Rc<MultiLineButton>,
    /// Java private `archiveInfoA = new JLabel()`.
    archive_info_a: Rc<JComponent>,
    /// Java private `archiveInfoB = new JLabel()`.
    archive_info_b: Rc<JComponent>,

    /// Java private `axisType`.
    axis_type: AxisType,
}

impl Deref for CleanUpDialog {
    type Target = ProcessDialog;
    fn deref(&self) -> &ProcessDialog {
        &self.base
    }
}

impl CleanUpDialog {
    /// Java public constructor `CleanUpDialog(ApplicationManager)`.
    pub fn new(app_mgr: &'static ApplicationManager) -> Rc<CleanUpDialog> {
        // super(appMgr, AxisID.ONLY, DialogType.CLEAN_UP)
        let base = ProcessDialog::new_application_manager_axis_id_dialog_type(
            app_mgr,
            AxisID::Only,
            DialogType::CleanUp,
        );
        // Field initializers, then this.axisType =
        // appMgr.getBaseMetaData().getAxisType() (the base meta data of an
        // ApplicationManager is its MetaData).
        let instance = Rc::new(CleanUpDialog {
            base,
            cleanup_panel: OnceCell::new(),
            btn_archive_stack: MultiLineButton::new_void(),
            archive_info_a: JComponent::new_label(""),
            archive_info_b: JComponent::new_label(""),
            axis_type: ConstMetaData::get_axis_type(app_mgr.get_meta_data()),
        });
        // Java `this` as the ProcessDialog subclass (for the virtual `done()`).
        let this: Weak<dyn ProcessDialogVirtual> =
            Rc::downgrade(&instance) as Weak<dyn ProcessDialogVirtual>;
        instance.base.set_this(this);
        // Constructor body.
        // Swing layout: rootPanel.setLayout(new BoxLayout(rootPanel,
        // BoxLayout.Y_AXIS)).
        instance
            .base
            .root_panel
            .set_border(&BeveledBorder::new(Some("Clean Up")).get_border());
        // String stackFileName;  (unused)
        // Add archive original stack button
        if instance.axis_type == AxisType::DualAxis {
            instance
                .btn_archive_stack
                .set_text(Some("Archive Original Stacks"));
        } else {
            instance
                .btn_archive_stack
                .set_text(Some("Archive Original Stack"));
            instance.archive_info_b.set_visible(false);
        }
        instance.set_archive_fields();
        // Java `new ButtonActionListener(this)`.
        let adaptee = Rc::downgrade(&instance);
        let listener: ActionListener = Rc::new(move |event: &ActionEvent| {
            if let Some(adaptee) = adaptee.upgrade() {
                adaptee.button_action(event);
            }
        });
        instance.btn_archive_stack.add_action_listener(listener);
        // Swing layout: btnArchiveStack, archiveInfoA and archiveInfoB
        // setAlignmentX(Component.CENTER_ALIGNMENT).
        let root_panel = instance.base.root_panel.get_component();
        root_panel.add(&instance.btn_archive_stack.get_component());
        root_panel.add(&instance.archive_info_a);
        root_panel.add(&instance.archive_info_b);

        let cleanup_panel = CleanupPanel::get_instance(instance.base.application_manager);
        root_panel.add(&cleanup_panel.get_container());
        let _ = instance.cleanup_panel.set(cleanup_panel);
        instance.base.add_exit_buttons();
        instance.base.btn_advanced.set_visible(false);
        instance.base.btn_execute.set_text(Some("Done"));

        // Mouse adapter for context menu
        // Swing mouse: rootPanel.addMouseListener(new GenericMouseAdapter(this)).
        // Mouse events are not modelled; the adapter's only effect is to call
        // popUpContextMenu on a right-button press, which a driver calls
        // directly.
        instance.set_tool_tip_text();
        instance
    }

    /// Java public final `updateArchiveDisplay(boolean)`.
    pub fn update_archive_display(&self, original_stacks_exist: bool) {
        // `if (btnArchiveStack == null) return;`: never null.
        self.btn_archive_stack.set_enabled(original_stacks_exist);
    }

    /// Java public `setArchiveFields()`.  Set the status of the archive
    /// information labels.
    pub fn set_archive_fields(&self) {
        let archive_info_text = "To restore original stack, run:  archiveorig -r ";
        let mut stack_file_name: Option<String>;
        // if archiveorig has been run, put information about restoring the
        // originals on the screen.
        if self.axis_type == AxisType::DualAxis {
            stack_file_name = self
                .base
                .application_manager
                .get_archive_info(AxisID::First);
            if let Some(name) = &stack_file_name {
                self.archive_info_a
                    .set_text(&format!("{archive_info_text}{name}"));
                self.archive_info_a.set_visible(true);
            } else {
                self.archive_info_a.set_visible(false);
            }
            stack_file_name = self
                .base
                .application_manager
                .get_archive_info(AxisID::Second);
            if let Some(name) = &stack_file_name {
                self.archive_info_b
                    .set_text(&format!("{archive_info_text}{name}"));
                self.archive_info_b.set_visible(true);
            } else {
                self.archive_info_b.set_visible(false);
            }
        } else {
            stack_file_name = self.base.application_manager.get_archive_info(AxisID::Only);
            if let Some(name) = &stack_file_name {
                self.archive_info_a.set_text(&format!(
                    "To restore original stack run:  archiveorig -r {name}"
                ));
                self.archive_info_a.set_visible(true);
            } else {
                self.archive_info_a.set_visible(false);
            }
        }
        let _ = stack_file_name;
    }

    /// Java package-private `buttonAction(ActionEvent)`.
    pub fn button_action(&self, event: &ActionEvent) {
        let command = event.get_action_command();
        // Java `command.equals(btnArchiveStack.getText())`.
        if command.is_some() && command == self.btn_archive_stack.get_text().as_deref() {
            self.base
                .application_manager
                .archive_original_stack_process_series_dialog_type(None, self.base.dialog_type);
        }
    }

    /// Java private `setToolTipText()`.  Initialize the tooltip text.
    fn set_tool_tip_text(&self) {
        self.btn_archive_stack.set_tool_tip_text(Some(
            "Run archiveorig.  Archiveorig creates a _xray.mrc.gz file, which contains the \
             difference between the .mrc file and the _orig.mrc  file.  If archiveorig \
             succeeds, then you can delete the _orig.mrc file.  To restore _orig.mrc, go to \
             the directory containing the _xray.mrc.gz file and run \"archiveorig -r\" on the \
             .mrc file.",
        ));
    }
}

impl ProcessDialogVirtual for CleanUpDialog {
    fn process_dialog(&self) -> &ProcessDialog {
        &self.base
    }

    /// Java package-private override `done()`.
    fn done(&self) {
        self.base.application_manager.done_clean_up();
        self.base.set_displayed(false);
        let manager: &'static dyn BaseManager = self.base.application_manager;
        let axis_id = self.base.axis_id;
        ui_harness::with(|harness| harness.pack_axis_id_base_manager(Some(axis_id), Some(manager)));
    }
}

impl ContextMenu for CleanUpDialog {
    /// Java public `popUpContextMenu(MouseEvent)`.  Right mouse button context
    /// menu.
    fn pop_up_context_menu(&self, mouse_event: &MouseEvent) {
        let manager: &'static dyn BaseManager = self.base.application_manager;
        let _context_popup =
            ContextPopup::new_component_mouse_event_string_string_base_manager_axis_id(
                &self.base.root_panel.get_component(),
                mouse_event,
                Some("Cleaning Up"),
                Some(context_popup::TOMO_GUIDE),
                manager,
                self.base.axis_id,
            );
    }
}
