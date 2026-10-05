//! `IMOD/Etomo/src/etomo/process/MessageReporter.java`.
//!
//! Reads a log file looking for lines that start with "MESSAGE:".  If it finds a
//! matching line, it pops up a message with the line.
//!
//! The reporter is owned by the monitor thread that calls it, so its mutable state
//! (`printedMessageSet`, `id`) takes `&mut self`.

use std::collections::HashSet;
use std::sync::Arc;

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::storage::log_file::{Handle, LogFileError, ReaderId};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::ui::swing::ui_harness;

/// Java private static final `TOKEN`.
const TOKEN: &str = "MESSAGE:"; // "Error returned";

/// Java package-private final `MessageReporter`.
pub struct MessageReporter {
    /// Java private final `printedMessageSet`.
    printed_message_set: HashSet<String>,
    /// Java private final `file`.
    file: Arc<Handle>,
    /// Java private final `axisID` (only handed to the popup, which does not read it).
    #[allow(dead_code)]
    axis_id: AxisID,
    /// Java private `id`.
    id: Option<ReaderId>,
}

impl MessageReporter {
    /// Java `MessageReporter(AxisID, LogFile.Handle)`.
    pub fn new(axis_id: AxisID, file: Arc<Handle>) -> MessageReporter {
        MessageReporter {
            printed_message_set: HashSet::new(),
            file,
            axis_id,
            id: None,
        }
    }

    /// Java `checkForMessages(BaseManager)`.
    pub fn check_for_messages(&mut self, manager: &'static dyn BaseManager) {
        if self.id.is_none() {
            match self.file.open_reader() {
                Ok(id) => self.id = id,
                // catch (final LockException e) {}
                Err(LogFileError::Lock(_)) => {}
                // catch (final LogFileException | IOException e): e.printStackTrace()
                Err(e) => eprintln!("{e:?}"),
            }
        }
        if let Some(id) = &self.id {
            loop {
                let line = match self.file.read_line(id) {
                    Ok(Some(line)) => line,
                    Ok(None) => break,
                    Err(e) => {
                        // catch (LogFileException | IOException e): e.printStackTrace()
                        eprintln!("{e:?}");
                        break;
                    }
                };
                // `line.trim()`: Java's trim strips code units <= ' '.
                if line.trim_matches(|c: char| c <= ' ').starts_with(TOKEN)
                    && !self.printed_message_set.contains(&line)
                {
                    self.printed_message_set.insert(line.clone());
                    // SwingUtilities.invokeLater(new PopupLater(manager, axisID,
                    //   line.substring(line.indexOf(TOKEN) + TOKEN.length())))
                    let index = line.find(TOKEN).unwrap() + TOKEN.len();
                    ui_harness::post_message_dialog(
                        Some(manager),
                        line[index..].to_owned(),
                        "Process Message".to_owned(),
                        None,
                    );
                }
            }
        }
    }

    /// Java `close()`.
    pub fn close(&mut self) {
        let Some(id) = self.id.take() else {
            return;
        };
        self.file.close_id(Some(&*id));
    }
}
