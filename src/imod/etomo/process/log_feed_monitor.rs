//! `IMOD/Etomo/src/etomo/process/LogFeedMonitor.java`: **partial**.
//!
//! Only the static `getFileFromOutput` is translated (ProcessManager's
//! batchruntomo rename check calls it through `BatchRunTomoProcessMonitor`).
//! TODO(unit): the monitor class itself (the batchruntomo log feed).

use super::process_output_strings;

/// Java static `getFileFromOutput`.
pub fn get_file_from_output(
    line: &str,
    msg_id: &str,
    deprecated_tag: Option<&str>,
) -> Option<String> {
    // Attempt to recognize the output line.
    if !line.contains(msg_id) && deprecated_tag.is_none_or(|tag| !line.contains(tag)) {
        return None;
    }
    // Attempt to get the file.
    let index = line.find(process_output_strings::BRT_FILE_LOCATION_TAG)?;
    let mut file = line[index + process_output_strings::BRT_FILE_LOCATION_TAG.len()..]
        .trim()
        .to_owned();
    if file.is_empty() {
        return None;
    }
    // Remove the message ID if it is at the end of the line.
    if let Some(index) = file.find(msg_id) {
        file = file[..index].trim().to_owned();
    }
    if file.is_empty() {
        return None;
    }
    Some(file)
}
