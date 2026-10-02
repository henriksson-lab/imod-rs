//! `IMOD/Etomo/src/etomo/process/TomosnapshotProcess.java`.
//!
//! Class to run tomosnapshot on a separate thread without a manager or
//! EtomoDirector.  [`TomosnapshotProcess::run`] is the Java `Runnable.run`,
//! the body of the thread `BaseProcessManager.tomosnapshot` starts; its
//! popups go to the event dispatch thread (`ui_harness`'s `*_from_process`
//! helpers), and its one question waits there for the answer, as the Java's
//! modal dialog does.

use super::system_program::SystemProgram;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::command::Command;
use crate::imod::etomo::comscript::tomosnapshot_param::TomosnapshotParam;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::ui::swing::ui_harness;
use crate::imod::etomo::util::event_queue;
use std::sync::Arc;

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// `System.getProperty("user.dir")`: `PWD` is the variable this translation
/// keeps it in (see `BaseManager::make_property_user_dir_local`).
fn user_dir() -> Option<String> {
    std::env::var("PWD").ok()
}

/// Java final `TomosnapshotProcess implements Runnable`.
pub struct TomosnapshotProcess {
    /// Java private final `manager`.
    manager: &'static dyn BaseManager,
    /// Java private final `axisID`.
    axis_id: AxisID,
    /// Java private final `thumbnail`.
    thumbnail: bool,
}

impl TomosnapshotProcess {
    /// Java `TomosnapshotProcess(BaseManager, AxisID, boolean)`.
    pub fn new(
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        thumbnail: bool,
    ) -> TomosnapshotProcess {
        TomosnapshotProcess {
            manager,
            axis_id,
            thumbnail,
        }
    }

    /// Java `run`.
    pub fn run(&self) {
        // Run tomosnapshot.
        let mut param = TomosnapshotParam::new(Some(self.manager), self.axis_id);
        if self.thumbnail {
            // `UIHarness.INSTANCE.openYesNoDialogWithDefaultNo(...)`: a modal
            // question, asked on the event dispatch thread while this thread
            // waits for the answer.
            let manager = self.manager;
            let axis_id = self.axis_id;
            let yes = event_queue::invoke_and_wait(move || {
                ui_harness::with(|harness| {
                    harness.open_yes_no_dialog_with_default_no(
                        Some(manager),
                        "Should the snapshot of this dataset include thumbnails of image data?",
                        "Include Image Data?",
                        Some(axis_id),
                    )
                })
            });
            if yes {
                param.set_thumbnail(true);
            }
        }
        let sys_program = Arc::new(SystemProgram::new_array(
            None,
            user_dir(),
            param.get_command_array(),
            self.axis_id,
        ));
        let thread_program = Arc::clone(&sys_program);
        std::thread::spawn(move || thread_program.run());
        // Wait until tomosnapshot is done.
        while !sys_program.is_done() {
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        // Pop up done message.
        let stdout = sys_program.get_std_output();
        if sys_program.is_done() && sys_program.get_exit_value() == 0 {
            match &stdout {
                Some(stdout) if !stdout.is_empty() => {
                    ui_harness::open_info_message_dialog_from_process(
                        None,
                        &format!(
                            "{}\n\nFor instructions on how to see the contents and to see our privacy practices, run \"imodhelp tomosnapshot\".",
                            stdout[stdout.len() - 1]
                        ),
                        "Snapshot Created",
                        Some(self.axis_id),
                    );
                }
                _ => {
                    ui_harness::open_message_dialog_from_process(
                        None,
                        &format!(
                            "Snapshot file created in {}.  For instructions on how to see the contents and to see our privacy practices, run \"imodhelp tomosnapshot\".",
                            user_dir().unwrap_or_else(|| "null".to_owned())
                        ),
                        "Snapshot Created",
                        Some(self.axis_id),
                    );
                }
            }
            return;
        }
        // Handle error or interrupt
        let mut error_message = String::new();
        let title;
        if !sys_program.is_done() {
            // Handle interrupt.
            title = "Process Incomplete";
            error_message.push_str("Tomosnapshot process did not finish.");
        } else {
            // Handle error.
            title = "Process Failed";
            error_message.push_str(&format!(
                "Unable to run tomosnapshot in {}.  Exit value is {}.  ",
                user_dir().unwrap_or_else(|| "null".to_owned()),
                sys_program.get_exit_value()
            ));
        }
        // Pop up message for error or interrupt.
        if let Some(stdout) = &stdout {
            for line in stdout {
                let trimmed =
                    crate::imod::etomo::r#type::const_etomo_number::java_lang_string_trim(line);
                if trimmed.starts_with("ERROR:") {
                    error_message.push_str(&format!("{line}  "));
                }
            }
        }
        if let Some(stderr) = sys_program.get_std_error() {
            for line in &stderr {
                error_message.push_str(&format!("{line}  "));
            }
        }
        ui_harness::open_message_dialog_from_process(
            None,
            &error_message,
            title,
            Some(self.axis_id),
        );
    }
}
