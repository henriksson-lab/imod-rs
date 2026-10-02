//! `IMOD/Etomo/src/etomo/ui/swing/ReconUIExpert.java`.
//!
//! Java `public abstract class ReconUIExpert implements UIExpert`.  Following
//! the translation's inheritance convention, [`ReconUIExpert`] is the
//! superclass struct (its fields and its concrete/final methods); a concrete
//! expert embeds it as field `base`, derefs to it, and implements
//! [`ReconUIExpertVirtual`] for the three abstract methods
//! (`doneDialog()`, `saveDialog()`, `getDialog()`).  The concrete expert's
//! constructor calls [`ReconUIExpert::set_this`] right after `Rc::new_cyclic` /
//! `Rc::new`, so the final methods that call the abstract ones
//! (`doneDialog(DialogExitState)`, `saveAction()`, `saveDialog(DialogExitState)`)
//! dispatch through it.  The concrete expert's `impl UIExpert` forwards
//! `save_action` / `save_dialog` to the final methods here, as the Java
//! subclasses inherit them.
//!
//! Experts live on the event dispatch thread (`util/event_queue.rs`); every
//! method takes `&self`.

use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::processchunks_param::{OutputImageFileKey, ProcesschunksParam};
use crate::imod::etomo::process::process_state::ProcessState;
use crate::imod::etomo::process_series::ProcessSeriesHandle;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::meta_data::MetaData;
use crate::imod::etomo::r#type::process_end_state::ProcessEndState;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::process_result::ProcessResult;
use crate::imod::etomo::r#type::process_result_display::ProcessResultDisplayHandle;
use crate::imod::etomo::r#type::process_track::ProcessTrack;
use crate::imod::etomo::r#type::processing_method::ProcessingMethod;
use crate::imod::etomo::ui::shared_strings;
use crate::imod::etomo::ui::swing::abstract_parallel_dialog::AbstractParallelDialog;
use crate::imod::etomo::ui::swing::main_tomogram_panel::MainTomogramPanel;
use crate::imod::etomo::ui::swing::parallel_panel::ParallelPanel;
use crate::imod::etomo::ui::swing::process_dialog::{
    DialogExitState, ProcessDialog, ProcessDialogVirtual,
};
use crate::imod::etomo::ui::swing::ui_expert::UIExpert;
use crate::imod::etomo::ui::swing::ui_expert_utilities::UIExpertUtilities;
use crate::imod::etomo::ui::swing::ui_harness;
use std::cell::{Cell, RefCell};
use std::rc::{Rc, Weak};

/// Java `public static final String rcsid`.
pub const RCSID: &str = "$Id$";

/// The abstract methods of Java `ReconUIExpert`, implemented by each concrete
/// expert.
pub trait ReconUIExpertVirtual: UIExpert {
    /// Java abstract package-private `doneDialog()`.
    fn done_dialog_void(&self);

    /// Java abstract package-private `saveDialog()`.
    fn save_dialog_void(&self);

    /// Java abstract package-private `getDialog()`; `None` is Java null.
    fn get_dialog(&self) -> Option<Rc<dyn ProcessDialogVirtual>>;
}

/// Java abstract class `ReconUIExpert`: its fields and non-abstract methods.
pub struct ReconUIExpert {
    /// Java package-private final `manager`.
    pub manager: &'static ApplicationManager,
    /// Java package-private final `metaData`, read once from the manager at
    /// construction as the Java does.
    pub meta_data: &'static MetaData,
    /// Java package-private final `axisID`.
    pub axis_id: AxisID,
    /// Java package-private final `dialogType`.
    pub dialog_type: DialogType,
    /// Java private final `mainPanel`.
    main_panel: Rc<MainTomogramPanel>,
    /// Java private final `processTrack`; the Java tests it for null.
    process_track: Option<&'static ProcessTrack>,
    /// Java private `dialogOutOfDate`.
    dialog_out_of_date: Cell<bool>,
    /// Rust-only: Java's `this`, seen as the subclass, for the abstract calls.
    this: RefCell<Option<Weak<dyn ReconUIExpertVirtual>>>,
}

impl ReconUIExpert {
    /// Java `ReconUIExpert(ApplicationManager, MainTomogramPanel, ProcessTrack,
    /// AxisID, DialogType)`.
    pub fn new(
        manager: &'static ApplicationManager,
        main_panel: Rc<MainTomogramPanel>,
        process_track: Option<&'static ProcessTrack>,
        axis_id: AxisID,
        dialog_type: DialogType,
    ) -> ReconUIExpert {
        let meta_data = manager.get_meta_data();
        ReconUIExpert {
            manager,
            meta_data,
            main_panel,
            process_track,
            axis_id,
            dialog_type,
            dialog_out_of_date: Cell::new(false),
            this: RefCell::new(None),
        }
    }

    /// Rust-only: records the subclass instance (Java `this`) so the final
    /// methods can call the abstract ones.  Called by the concrete expert's
    /// constructor right after its `Rc` is made.
    pub fn set_this(&self, this: Weak<dyn ReconUIExpertVirtual>) {
        *self.this.borrow_mut() = Some(this);
    }

    fn this(&self) -> Option<Rc<dyn ReconUIExpertVirtual>> {
        self.this.borrow().as_ref().and_then(Weak::upgrade)
    }

    /// Java final package-private `canShowDialog()`: false if the com
    /// scripts where never created in Tomogram Setup.
    pub fn can_show_dialog(&self) -> bool {
        // Check to see if the com files are present otherwise pop up a dialog
        // box informing the user to run the setup process
        if !UIExpertUtilities::INSTANCE.are_scripts_created(
            self.manager,
            self.meta_data,
            self.axis_id,
        ) {
            self.main_panel.show_blank_process(self.axis_id);
            return false;
        }
        true
    }

    /// Java final package-private `showDialog(ProcessDialog, String)`: turn on
    /// the button associated with dialog and set the current dialog type.
    /// Display dialog if it already exists and is up to date.  Returns true if
    /// an existing, up to date dialog is shown.
    pub fn show_dialog(
        &self,
        dialog: Option<&ProcessDialog>,
        action_message: Option<&str>,
    ) -> bool {
        self.main_panel
            .select_button(self.axis_id, &self.dialog_type.to_string());
        let dialog = match dialog {
            Some(dialog) if !self.dialog_out_of_date.get() => dialog,
            _ => return false,
        };
        self.main_panel
            .show_process(&dialog.get_container(), self.axis_id);
        if let Some(action_message) = action_message {
            eprintln!("{action_message}");
        }
        true
    }

    /// Java final package-private `openDialog(ProcessDialog, String)`: display
    /// dialog and set parallel dialog if necessary.  This function is used
    /// when a dialog has just been created.
    ///
    /// Named with its parameter types because the class also inherits
    /// `UIExpert.openDialog()`, which a concrete expert implements; a plain
    /// `open_dialog` here would be shadowed by that trait method.
    pub fn open_dialog_process_dialog_string(
        &self,
        dialog: &ProcessDialog,
        action_message: Option<&str>,
    ) {
        self.dialog_out_of_date.set(false);
        self.main_panel
            .show_process(&dialog.get_container(), self.axis_id);
        if let Some(action_message) = action_message {
            eprintln!("{action_message}");
        }
    }

    /// Java final package-private `sendMsgProcessStarting(ProcessResultDisplay)`.
    pub fn send_msg_process_starting(
        &self,
        process_result_display: Option<&ProcessResultDisplayHandle>,
    ) {
        let Some(process_result_display) = process_result_display else {
            return;
        };
        process_result_display.msg_process_starting();
    }

    /// Java final package-private `sendMsg(ProcessResult, ProcessResultDisplay)`.
    pub fn send_msg(
        &self,
        display_state: Option<ProcessResult>,
        process_result_display: Option<&ProcessResultDisplayHandle>,
    ) {
        let (Some(display_state), Some(process_result_display)) =
            (display_state, process_result_display)
        else {
            return;
        };
        process_result_display.msg_process_result(display_state);
    }

    /// Java final package-private `leaveDialog(DialogExitState)`.
    pub fn leave_dialog(&self, exit_state: DialogExitState) {
        if exit_state == DialogExitState::Cancel {
            self.main_panel.show_blank_process(self.axis_id);
        } else if exit_state == DialogExitState::Postpone {
            self.set_dialog_state(ProcessState::InProgress);
            self.main_panel.show_blank_process(self.axis_id);
        } else if exit_state == DialogExitState::Execute {
            self.set_dialog_state(ProcessState::Complete);
            self.manager
                .open_next_dialog(self.axis_id, self.dialog_type);
        }
        self.dialog_out_of_date.set(true);
    }

    /// Java final package-private `setDialogState(ProcessState)`.
    pub fn set_dialog_state(&self, process_state: ProcessState) {
        if let Some(process_track) = self.process_track {
            process_track.set_state_dialog_type(process_state, self.axis_id, self.dialog_type);
        }
        self.main_panel.set_state_process_state_axis_id_dialog_type(
            process_state,
            self.axis_id,
            self.dialog_type,
        );
    }

    /// Java public final `doneDialog(DialogExitState)`.
    pub fn done_dialog_dialog_exit_state(&self, exit_state: DialogExitState) {
        let Some(this) = self.this() else {
            return;
        };
        let Some(dialog) = this.get_dialog() else {
            return;
        };
        dialog.process_dialog().set_exit_state(exit_state);
        this.done_dialog_void();
    }

    /// Java public final `saveAction()` (implements `UIExpert.saveAction`).
    pub fn save_action(&self) {
        let Some(this) = self.this() else {
            return;
        };
        let Some(dialog) = this.get_dialog() else {
            return;
        };
        dialog.process_dialog().save_action();
    }

    /// Java public final `saveDialog(DialogExitState)` (implements
    /// `UIExpert.saveDialog`).
    pub fn save_dialog_dialog_exit_state(&self, exit_state: DialogExitState) {
        let Some(this) = self.this() else {
            return;
        };
        let Some(dialog) = this.get_dialog() else {
            return;
        };
        dialog.process_dialog().set_exit_state(exit_state);
        this.save_dialog_void();
    }

    /// Java final package-private `processchunks(BaseManager,
    /// AbstractParallelDialog, ProcessResultDisplay, ProcessSeries, String,
    /// FileKey, ProcessingMethod, boolean)`: run processchunks.  The Java
    /// parameter `manager` shadows the field.
    #[allow(clippy::too_many_arguments)]
    pub fn processchunks(
        &self,
        manager: &'static dyn BaseManager,
        dialog: Option<&dyn AbstractParallelDialog>,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
        process_name: &str,
        output_image_file_key: Option<OutputImageFileKey>,
        processing_method: Option<ProcessingMethod>,
        multi_line_messages: bool,
    ) -> bool {
        self.send_msg_process_starting(process_result_display.as_ref());
        let Some(dialog) = dialog else {
            self.send_msg(
                Some(ProcessResult::FAILED_TO_START),
                process_result_display.as_ref(),
            );
            if let Some(process_series) = &process_series {
                process_series.borrow().end_series();
            }
            return false;
        };
        let mut param = ProcesschunksParam::get_instance(
            manager,
            self.axis_id,
            Some(process_name),
            output_image_file_key,
        );
        dialog.get_parameters(&mut param);
        // Java `manager.getMainPanel().getParallelPanel(axisID)`, through the
        // Rust-only EDT access path to the main panel (null main panel -> null).
        let mut parallel_panel = None;
        manager.with_main_panel(&mut |main_panel| {
            parallel_panel = main_panel.get_parallel_panel(self.axis_id);
        });
        let Some(parallel_panel) = parallel_panel else {
            ui_harness::open_message_dialog_from_process(
                Some(manager),
                &shared_strings::PARALLEL_PROCESSING_REQUIRED_MESSAGE,
                "Unable to execute command",
                Some(self.axis_id),
            );
            self.send_msg(
                Some(ProcessResult::FAILED_TO_START),
                process_result_display.as_ref(),
            );
            if let Some(process_series) = &process_series {
                process_series.borrow().end_series();
            }
            return false;
        };
        if !parallel_panel.get_parameters_processchunks_param_boolean(&mut param, true) {
            // Java `manager.getMainPanel().stopProgressBar(axisID, ProcessEndState.FAILED)`.
            manager.with_main_panel(&mut |main_panel| {
                main_panel.stop_progress_bar_axis_id_process_end_state(
                    self.axis_id,
                    Some(ProcessEndState::Failed),
                );
            });
            self.send_msg(
                Some(ProcessResult::FAILED_TO_START),
                process_result_display.as_ref(),
            );
            if let Some(process_series) = &process_series {
                process_series.borrow().end_series();
            }
            return false;
        }
        self.set_dialog_state(ProcessState::InProgress);
        // param should never be set to resume
        parallel_panel
            .get_parallel_progress_display()
            .reset_results();
        manager.processchunks(
            Some(self.axis_id),
            Some(std::sync::Arc::new(param)),
            process_result_display,
            process_series,
            true,
            processing_method,
            multi_line_messages,
            Some(self.dialog_type),
            None,
            None,
            None,
        )
    }

    /// Java final package-private `getParallelPanel()`.
    pub fn get_parallel_panel(&self) -> Option<Rc<ParallelPanel>> {
        self.main_panel.get_parallel_panel(self.axis_id)
    }

    /// Java final public `setProgressBar(String, int, AxisID)`.
    pub fn set_progress_bar_string_int_axis_id(&self, label: &str, n_steps: i32, axis_id: AxisID) {
        self.set_progress_bar_string_int_axis_id_process_name(label, n_steps, axis_id, None);
    }

    /// Java final public `setProgressBar(String, int, AxisID, ProcessName)`.
    /// The Java ignores `processName`.
    pub fn set_progress_bar_string_int_axis_id_process_name(
        &self,
        label: &str,
        n_steps: i32,
        axis_id: AxisID,
        _process_name: Option<ProcessName>,
    ) {
        self.main_panel.set_progress_bar_string_int_boolean_axis_id(
            Some(label),
            n_steps,
            false,
            axis_id,
        );
    }

    /// Java final public `startProgressBar(String, AxisID)`.
    pub fn start_progress_bar(&self, label: &str, axis_id: AxisID) {
        self.main_panel
            .start_progress_bar_string_axis_id(Some(label), axis_id);
    }

    /// Java final public `stopProgressBar(AxisID)`.
    pub fn stop_progress_bar_axis_id(&self, axis_id: AxisID) {
        self.main_panel.stop_progress_bar_axis_id(axis_id);
    }

    /// Java final public `stopProgressBar(AxisID, ProcessEndState)`.
    pub fn stop_progress_bar_axis_id_process_end_state(
        &self,
        axis_id: AxisID,
        end_state: ProcessEndState,
    ) {
        self.main_panel
            .stop_progress_bar_axis_id_process_end_state(axis_id, Some(end_state));
    }
}
