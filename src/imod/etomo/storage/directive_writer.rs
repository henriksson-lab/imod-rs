//! `IMOD/Etomo/src/etomo/storage/DirectiveWriter.java`.
//!
//! Writes a directive file (a template or a batch directive file) from the directive
//! editor: a title comment, the comment lines, the included directives, and the
//! directives that could not be included, as comments.

use std::path::PathBuf;
use std::sync::Arc;

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::storage::directive::Directive;
use crate::imod::etomo::storage::log_file::{Handle, LogFile, LogFileError, WriterId};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::directive_file_type::DirectiveFileType;
use crate::imod::etomo::ui::swing::ui_harness;
use crate::imod::etomo::util::utilities::java_io_file_get_absolute_path;

/// Java `public final class DirectiveWriter`.
pub struct DirectiveWriter {
    /// Java private final `saveFile`.
    save_file: Option<PathBuf>,
    /// Java private final `manager`.
    manager: Option<&'static dyn BaseManager>,
    /// Java private final `axisID`.
    axis_id: AxisID,
    /// Java private `logFile`, initialised to null.
    log_file: Option<Arc<Handle>>,
    /// Java private `id`, initialised to null.
    id: Option<WriterId>,
}

impl DirectiveWriter {
    /// Java `DirectiveWriter(BaseManager, AxisID, File)`.
    pub fn new(
        manager: Option<&'static dyn BaseManager>,
        axis_id: AxisID,
        save_file: Option<PathBuf>,
    ) -> DirectiveWriter {
        DirectiveWriter {
            save_file,
            manager,
            axis_id,
            log_file: None,
            id: None,
        }
    }

    /// `saveFile.getAbsolutePath()`.
    fn save_file_absolute_path(&self) -> String {
        java_io_file_get_absolute_path(&self.save_file.as_ref().unwrap().to_string_lossy())
    }

    /// Java `UIHarness.INSTANCE.openMessageDialog(manager, message, title)`.
    fn open_message_dialog(&self, message: &str, title: &str) {
        let manager = self.manager;
        ui_harness::with(|harness| {
            harness.open_message_dialog_base_manager_string_string(manager, message, title)
        });
    }

    /// Java `open()`.  Open the file.
    pub fn open(&mut self) -> bool {
        let Some(save_file) = self.save_file.clone() else {
            self.open_message_dialog("No file set.", "Failed to Open Writer");
            return false;
        };
        let result: Result<(), LogFileError> = (|| {
            let log_file = LogFile::get_instance_file(
                Some(&save_file),
                self.manager
                    .map(|manager| manager.get_emergency_monitor(Some(self.axis_id))),
            )?;
            self.log_file = Some(log_file.clone());
            if log_file.exists() {
                log_file.backup()?;
            }
            self.id = Some(log_file.open_writer()?);
            Ok(())
        })();
        if let Err(e) = result {
            // `catch (LogFileException | IOException | LockException e)`: the three
            // arms are identical.  `e.printStackTrace()`; see etomo/util/stack_trace.rs.
            eprintln!("{}", e);
            self.open_message_dialog(
                &format!(
                    "Unable to open file:  {}.  {}",
                    self.save_file_absolute_path(),
                    e.get_message()
                ),
                "Unable to Open File",
            );
            return false;
        }
        true
    }

    /// Java `write(DirectiveFileType, List<String>, List<Directive>, List<String>)`.
    /// Write to the file.  Directives will only be written if at least one of their
    /// include booleans is checked.  Writing is halted if an exception is detected.
    /// `comments` (optional) are written first, `directive_list` (optional) second,
    /// `dropped_directives` (directives that could not be edited) as comments last.
    pub fn write(
        &self,
        r#type: DirectiveFileType,
        comments: Option<&[String]>,
        directive_list: Option<&[Arc<Directive>]>,
        dropped_directives: Option<&[String]>,
    ) {
        let (Some(log_file), Some(id)) = (self.log_file.as_ref(), self.id.as_ref()) else {
            return;
        };
        let result: Result<(), LogFileError> = (|| {
            log_file.write(Some(&format!("# {}", r#type.get_label())), id)?;
            log_file.new_line(id)?;
            if let Some(comments) = comments {
                for comment in comments {
                    log_file.write(Some(&format!("# {}", comment)), id)?;
                    log_file.new_line(id)?;
                }
                log_file.new_line(id)?;
            }
            if let Some(directive_list) = directive_list {
                for directive in directive_list {
                    if directive.is_include() {
                        directive.write(log_file, id)?;
                    }
                }
            }
            if let Some(dropped_directives) = dropped_directives
                && !dropped_directives.is_empty()
            {
                log_file.new_line(id)?;
                log_file.write(Some("# Directives that could not be included:"), id)?;
                log_file.new_line(id)?;
                for dropped in dropped_directives {
                    log_file.write(Some(&format!("# {}", dropped)), id)?;
                    log_file.new_line(id)?;
                }
            }
            Ok(())
        })();
        if let Err(e) = result {
            // `catch (LogFile.UnlockedException e)` and `catch (IOException e)`.
            eprintln!("{}", e);
            self.open_message_dialog(
                &format!(
                    "Unable to write to file:  {}.  {}",
                    self.save_file_absolute_path(),
                    e.get_message()
                ),
                "Unable to Write to File",
            );
        }
    }

    /// Java `close()`.  Close the file.
    pub fn close(&self) {
        if let (Some(log_file), Some(id)) = (self.log_file.as_ref(), self.id.as_ref()) {
            log_file.close_id(Some(&**id));
        }
    }
}
