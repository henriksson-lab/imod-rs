//! `IMOD/Etomo/src/etomo/storage/JoinInfoFile.java`.
//!
//! Reads `<root>.info`, which makejoincom writes: its second line says, for each
//! section, whether the section will be inverted.

use std::sync::Arc;

use super::log_file::{Handle, LogFile, LogFileError};
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::java_lang_string_trim;
use crate::imod::etomo::r#type::etomo_boolean2::EtomoBoolean2;
use crate::imod::etomo::ui::swing::ui_harness;
use crate::imod::etomo::util::dataset_files;

/// Java private static final `ERROR_TITLE`.
const ERROR_TITLE: &str = "Etomo Error";
/// Java private static final `ERROR_MESSAGE`.
const ERROR_MESSAGE: &str = "WARNING:  Unable to find out if sections will be inverted.";

/// Java `public class JoinInfoFile`.
pub struct JoinInfoFile {
    /// Java private `joinInfo` (set by the constructor or `getInstance`).
    join_info: Arc<Handle>,
    /// Java private `loaded`, initially false.
    loaded: bool,
    /// Java private `invertedArray`, initially null.
    inverted_array: Option<Vec<EtomoBoolean2>>,
}

impl JoinInfoFile {
    /// Java private `JoinInfoFile(LogFile.Handle)`.
    fn new(log_file: Arc<Handle>) -> JoinInfoFile {
        JoinInfoFile {
            join_info: log_file,
            loaded: false,
            inverted_array: None,
        }
    }

    /// Java static `getInstance(BaseManager) throws FileException, IOException`.
    pub fn get_instance(manager: &'static dyn BaseManager) -> Result<JoinInfoFile, LogFileError> {
        let join_info = LogFile::get_instance_user_dir(
            manager.get_property_user_dir().as_deref().unwrap_or(""),
            &dataset_files::get_join_info_name(manager),
            Some(manager.get_emergency_monitor(Some(AxisID::Only))),
        )?;
        Ok(JoinInfoFile::new(join_info))
    }

    /// Java static package-private `getTestInstance(LogFile.Handle)`.
    pub(crate) fn get_test_instance(log_file: Arc<Handle>) -> JoinInfoFile {
        JoinInfoFile::new(log_file)
    }

    /// Java `getInverted(BaseManager, int)`; null is `None`.
    pub fn get_inverted(
        &mut self,
        manager: &'static dyn BaseManager,
        index: usize,
    ) -> Option<EtomoBoolean2> {
        if !self.loaded && !self.load(manager) {
            return None;
        }
        match self
            .inverted_array
            .as_ref()
            .and_then(|inverted_array| inverted_array.get(index))
        {
            Some(inverted) => Some(inverted.clone()),
            None => {
                // catch (IndexOutOfBoundsException e)
                let size = self.inverted_array.as_ref().map_or(0, Vec::len);
                let exception = format!(
                    "java.lang.IndexOutOfBoundsException: Index: {}, Size: {}",
                    index, size
                );
                eprintln!("{exception}");
                ui_harness::open_message_dialog_from_process(
                    Some(manager),
                    &format!("{}\n{}", ERROR_MESSAGE, exception),
                    ERROR_TITLE,
                    None,
                );
                None
            }
        }
    }

    /// Java private `load(BaseManager)`.
    ///
    /// Fixed in translation (JoinInfoFile.java:107): the source's catch block passes
    /// the title as the message and the message as the title
    /// (`openMessageDialog(manager, "Etomo Error", ERROR_MESSAGE + ...)`); the
    /// arguments are in their places here.
    fn load(&mut self, manager: &'static dyn BaseManager) -> bool {
        self.reset();
        let result = (|| -> Result<bool, LogFileError> {
            let reader_id = self.join_info.open_reader()?;
            let Some(reader_id) = reader_id else {
                return Ok(false);
            };
            self.join_info.read_line(&reader_id)?;
            let line = self.join_info.read_line(&reader_id)?;
            match line {
                None => {
                    ui_harness::open_message_dialog_from_process(
                        Some(manager),
                        ERROR_MESSAGE,
                        ERROR_TITLE,
                        None,
                    );
                    return Ok(false);
                }
                Some(line) => {
                    // `line.trim().split(" +")`
                    let trimmed = java_lang_string_trim(&line);
                    let array: Vec<&str> = if trimmed.is_empty() {
                        vec![""]
                    } else {
                        trimmed.split(' ').filter(|token| !token.is_empty()).collect()
                    };
                    for token in array {
                        let inverted = EtomoBoolean2::new();
                        if inverted.is_valid() {
                            let mut value = EtomoBoolean2::new();
                            value.set_string(Some(token));
                            self.inverted_array.as_mut().unwrap().push(value);
                        }
                    }
                }
            }
            self.join_info.close_id(Some(&*reader_id));
            Ok(true)
        })();
        match result {
            Ok(true) => {}
            Ok(false) => return false,
            Err(LogFileError::Lock(_)) => return false,
            Err(e) => {
                eprintln!("{e:?}");
                ui_harness::open_message_dialog_from_process(
                    Some(manager),
                    &format!("{}\n{:?}", ERROR_MESSAGE, e),
                    ERROR_TITLE,
                    None,
                );
                return false;
            }
        }
        self.loaded = true;
        true
    }

    /// Java private `reset()`.
    fn reset(&mut self) {
        match self.inverted_array.as_mut() {
            None => self.inverted_array = Some(Vec::new()),
            Some(inverted_array) => inverted_array.clear(),
        }
    }
}
