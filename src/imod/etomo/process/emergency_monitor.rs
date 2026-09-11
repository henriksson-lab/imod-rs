//! `IMOD/Etomo/src/etomo/process/EmergencyMonitor.java`.
//!
//! The two GUI touchpoints of this unit - `updateProgressBar`'s
//! `etomo/ui/swing/MainPanel.java` and `popupMessage`'s
//! `etomo/ui/swing/UIHarness.java` - are the eTomo Swing boundary and are marked in
//! place; everything else is translated.
#![allow(dead_code)]

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::storage::log_file::{LockException, LockExceptionCause};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::extension::Extension;
use crate::imod::etomo::ui::standard_bar_string::StandardBarString;
use crate::imod::etomo::util::stack_trace::StackTrace;

/// Java `EmergencyMonitor`.
pub struct EmergencyMonitor {
    /// Java field `manager`, an `etomo.BaseManager`.  `LogFile.Handle`'s constructor
    /// supplies Java's `null` here (`new EmergencyMonitor(null, null)`); a manager
    /// reaches it through `BaseManager.getEmergencyMonitor(AxisID)`.
    manager: Option<&'static dyn BaseManager>,
    /// Java field `axisID`.
    axis_id: Option<AxisID>,
    /// The instance's intrinsic monitor.  Java `alert` is `synchronized`, which
    /// serialises calls on one instance; Rust has no per-object monitor, so the lock the
    /// keyword takes is this field.
    monitor: std::sync::Mutex<()>,
}

impl EmergencyMonitor {
    /// Java `EmergencyMonitor(BaseManager, AxisID)`.
    pub fn new(
        manager: Option<&'static dyn BaseManager>,
        axis_id: Option<AxisID>,
    ) -> EmergencyMonitor {
        EmergencyMonitor {
            manager,
            axis_id,
            monitor: std::sync::Mutex::new(()),
        }
    }

    /// Java `alert`.
    pub fn alert(&self, lock_exception: Option<&LockException>, do_popup: bool) {
        let _monitor = self.monitor.lock().unwrap();
        if let Some(lock_exception) = lock_exception {
            let mut stack_trace = StackTrace::new_with_thread(None, None);
            stack_trace.print(Some("alert"), true);
            self.update_progress_bar(Some(lock_exception), &mut stack_trace);
            if do_popup {
                self.popup_message(Some(lock_exception), &mut stack_trace);
            }
        }
    }

    /// Java `updateProgressBar`.
    fn update_progress_bar(
        &self,
        lock_exception: Option<&LockException>,
        stack_trace: &mut StackTrace,
    ) {
        let lock_exception = match lock_exception {
            None => return,
            Some(lock_exception) => lock_exception,
        };
        let mut main_panel: Option<std::convert::Infallible> = None;
        if self.manager.is_some() && !stack_trace.is_starting() {
            // TODO(unit): needs etomo/ui/swing/MainPanel.java - `mainPanel =
            // manager.getMainPanel()`.  `BaseManager.getMainPanel` returns that Swing
            // class, so the translated abstract method can only produce `None`.
            main_panel = self.manager.unwrap().get_main_panel();
        }
        if main_panel.is_some() {
            // TODO(unit): needs etomo/ui/swing/MainPanel.java -
            // `mainPanel.setEmergencyMonitorBarString(lockException.getAction(),
            // lockException.getFileName(), lockException.getToFileName(), false, true,
            // axisID)`.
        } else {
            eprintln!(
                "\n{}\n",
                StandardBarString::build_bar_string_static(
                    lock_exception.get_action(),
                    Some(&lock_exception.get_file_name()),
                    lock_exception.get_to_file_name().as_deref(),
                    false,
                    true,
                )
            );
        }
    }

    /// Java `popupMessage`.  Pops up a message or, if this is the GUI thread, just prints
    /// it.  Always pops up the message during testing.
    fn popup_message(&self, lock_exception: Option<&LockException>, stack_trace: &mut StackTrace) {
        let lock_exception = match lock_exception {
            None => return,
            Some(lock_exception) => lock_exception,
        };
        let action = lock_exception.get_action();
        let mut action_descr = "modify";
        if let Some(action) = action {
            action_descr = action.get_action_descr();
        }
        let file_name = lock_exception.get_file_name();
        let to_file_name = lock_exception.get_to_file_name();
        let cause = lock_exception.get_cause();
        let same_jvm = matches!(cause, Some(LockExceptionCause::OverlappingFileLock(_)));
        let extension = Extension::get_instance(&file_name);
        let mut image_file = false;
        if let Some(extension) = extension {
            image_file = extension.is_image_file();
        }
        let info_request =
            " be extremely helpful if you could send us etomo's error log for this run.";
        // Build the message
        let mut err_msg = String::new();
        err_msg.push_str(&format!("Unable to {} {}", action_descr, file_name));
        let title = err_msg.clone();
        if let Some(to_file_name) = &to_file_name {
            err_msg.push_str(&format!(" to {}", to_file_name));
        }
        err_msg.push_str(".\n\n");
        if cause.is_none() {
            // If there's no cause, then its probably a lock collision, and therefore an
            // etomo problem.
            err_msg.push_str(
                &("Etomo may have had a problem with its internal file handling.  If the problem "
                    .to_string()
                    + "continues, the best course of action is to restart etomo.\n\nIt would also"
                    + info_request),
            );
        } else {
            // For Windows etomo can check for external and internal file locks.
            err_msg.push_str("It appears that this file was already open in ");
            if same_jvm && image_file {
                err_msg.push_str("etomo, 3dmod, or some other ");
            } else if same_jvm && !image_file {
                err_msg.push_str("etomo or another ");
            } else if !same_jvm && image_file {
                err_msg.push_str("3dmod, or some other ");
            } else {
                err_msg.push_str("another ");
            }
            err_msg.push_str(
                &(" application.  Please close the file and try again.  If the file "
                    .to_string()
                    + "does not seem to be open anywhere, restarting etomo may help.\n\nIf you do have "
                    + "to restart etomo, it would"
                    + info_request),
            );
        }
        // Use SwingUtilities.isEventDispatchThread() if it blocks the GUI.
        if !stack_trace.is_starting() && !stack_trace.is_exiting() {
            // TODO(unit): needs etomo/ui/swing/UIHarness.java -
            // `UIHarness.INSTANCE.openMessageDialog(manager, errMsg.toString(), title,
            // axisID)`.
            let _ = (&title, &err_msg, self.axis_id);
        } else {
            eprintln!("\n{}", title);
            eprintln!("{}\n", err_msg);
        }
    }

    /// Java `getManager`.
    pub fn get_manager(&self) -> Option<&'static dyn BaseManager> {
        self.manager
    }

    /// Java `getAxisID`.
    pub fn get_axis_id(&self) -> Option<AxisID> {
        self.axis_id
    }
}
