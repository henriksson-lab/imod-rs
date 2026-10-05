//! `IMOD/Etomo/src/etomo/storage/XfjointomoLog.java`.
//!
//! Reads xfjointomo's log (`xfjointomo.log`): for each "At boundary" line, the best
//! gap and the mean and maximum errors, which the boundary table shows and from which
//! `JoinProcessManager` decides whether gaps exist.
//!
//! **Sharing.**  Java keeps one instance per dataset in a static `Hashtable` and reads
//! it from the event dispatch thread (the boundary table) and from process threads
//! (`gapsExist`), so the instance is an `Arc` and its state sits behind one lock (the
//! source's `synchronized` methods).

use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex};

use super::log_file::{Handle, LogFile, LogFileError};
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::{Type, java_lang_string_trim};
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::util::dataset_files;

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java private static final `INSTANCE_LIST`.
static INSTANCE_LIST: LazyLock<Mutex<HashMap<String, Arc<XfjointomoLog>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
/// Java private static final `MAX_ERROR_INDEX`.
const MAX_ERROR_INDEX: usize = 14;

/// The state Java's `synchronized` methods guard.
struct State {
    /// Java private final `rowList` (a `Hashtable` keyed by boundary).
    row_list: HashMap<String, Row>,
    /// Java private final `rowArray` (a `Vector`).
    row_array: Vec<Row>,
    /// Java private `logFile`, initially null.
    log_file: Option<Arc<Handle>>,
}

/// Java `public final class XfjointomoLog`.
pub struct XfjointomoLog {
    /// Java private final `manager`.
    manager: &'static dyn BaseManager,
    /// Java private final `axisID`.
    axis_id: AxisID,
    /// Java private final `dir`.
    dir: Option<String>,
    /// The mutable fields.
    state: Mutex<State>,
}

impl XfjointomoLog {
    /// Java static `getInstance(BaseManager, AxisID)`.
    pub fn get_instance(manager: &'static dyn BaseManager, axis_id: AxisID) -> Arc<XfjointomoLog> {
        let instance = INSTANCE_LIST
            .lock()
            .unwrap()
            .get(&XfjointomoLog::get_unique_key(manager))
            .cloned();
        if let Some(instance) = instance {
            return instance;
        }
        XfjointomoLog::create_instance(manager, axis_id)
    }

    /// Java `reset()`.
    pub fn reset(&self) {
        self.state.lock().unwrap().log_file = None;
    }

    /// Java `rowExists(String) throws LogFileException, IOException, LockException`.
    pub fn row_exists(&self, boundary: &str) -> Result<bool, LogFileError> {
        let mut state = self.state.lock().unwrap();
        self.load(&mut state)?;
        Ok(state.row_list.contains_key(boundary))
    }

    /// Java `getBestGap(String) throws LogFileException, IOException, LockException`.
    pub fn get_best_gap(&self, boundary: &str) -> Result<Option<String>, LogFileError> {
        let mut state = self.state.lock().unwrap();
        self.load(&mut state)?;
        Ok(state.row_list.get(boundary).map(Row::get_best_gap))
    }

    /// Java `getMeanError(String)`.
    pub fn get_mean_error(&self, boundary: &str) -> Option<String> {
        let state = self.state.lock().unwrap();
        state.row_list.get(boundary).map(Row::get_mean_error)
    }

    /// Java `getMaxError(String) throws LogFileException, IOException, LockException`.
    pub fn get_max_error(&self, boundary: &str) -> Result<Option<String>, LogFileError> {
        let mut state = self.state.lock().unwrap();
        self.load(&mut state)?;
        Ok(state.row_list.get(boundary).map(Row::get_max_error))
    }

    /// Java synchronized `gapsExist() throws LogFileException, IOException,
    /// LockException`.
    pub fn gaps_exist(&self) -> Result<bool, LogFileError> {
        let mut state = self.state.lock().unwrap();
        self.load(&mut state)?;
        let mut gaps_exist = false;
        for row in &state.row_array {
            gaps_exist = gaps_exist || row.gap_exists();
        }
        Ok(gaps_exist)
    }

    /// Java private `XfjointomoLog(BaseManager, AxisID)`.
    fn new(manager: &'static dyn BaseManager, axis_id: AxisID) -> XfjointomoLog {
        XfjointomoLog {
            manager,
            axis_id,
            dir: manager.get_property_user_dir(),
            state: Mutex::new(State {
                row_list: HashMap::new(),
                row_array: Vec::new(),
                log_file: None,
            }),
        }
    }

    /// Java private synchronized static `createInstance(BaseManager, AxisID)`.
    fn create_instance(manager: &'static dyn BaseManager, axis_id: AxisID) -> Arc<XfjointomoLog> {
        let mut instance_list = INSTANCE_LIST.lock().unwrap();
        // make sure another thread didn't already run createInstance for this manager
        if let Some(instance) = instance_list.get(&XfjointomoLog::get_unique_key(manager)) {
            return Arc::clone(instance);
        }
        // create instance
        let instance = Arc::new(XfjointomoLog::new(manager, axis_id));
        instance_list.insert(XfjointomoLog::get_unique_key(manager), Arc::clone(&instance));
        instance
    }

    /// Java private static `getUniqueKey(BaseManager)`.  Unlike .edj files, multiple
    /// .ejf files can be placed in one directory.
    fn get_unique_key(manager: &'static dyn BaseManager) -> String {
        format!(
            "{}{}",
            manager
                .get_property_user_dir()
                .unwrap_or_else(|| "null".to_string()),
            manager.get_name().unwrap_or_else(|| "null".to_string())
        )
    }

    /// Java private synchronized `load() throws LogFileException, IOException,
    /// LockException`.  Load data from the log file.  Or reload data, if reset() has
    /// been called.
    ///
    /// Fixed in translation (XfjointomoLog.java:173): the source clears `rowList` but
    /// not `rowArray` before a reload, so after xfjointomo is rerun `gapsExist` also
    /// reads the previous run's rows (and can report gaps the new log does not have).
    /// Both are cleared here.
    fn load(&self, state: &mut State) -> Result<(), LogFileError> {
        if state.log_file.is_some() {
            return Ok(());
        }
        state.row_list.clear();
        state.row_array.clear();
        let log_file = LogFile::get_instance_user_dir(
            self.dir.as_deref().unwrap_or(""),
            dataset_files::XFJOINTOMO_LOG,
            Some(self.manager.get_emergency_monitor(Some(self.axis_id))),
        )?;
        state.log_file = Some(Arc::clone(&log_file));
        let reader_id = log_file.open_reader()?;
        let Some(reader_id) = reader_id else {
            return Ok(());
        };
        while let Some(line) = log_file.read_line(&reader_id)? {
            if line.contains("At boundary") {
                // load the data from the line
                let string_array: Vec<String> = java_lang_string_trim(&line)
                    .split(|c: char| matches!(c, ' ' | '\t' | '\n' | '\x0B' | '\x0C' | '\r'))
                    .filter(|token| !token.is_empty())
                    .map(str::to_owned)
                    .collect();
                if string_array.len() < MAX_ERROR_INDEX + 1 {
                    continue;
                }
                let boundary = XfjointomoLog::parse_boundary(&string_array);
                if boundary.is_empty() {
                    continue;
                }
                let row = Row::new(
                    &XfjointomoLog::parse_best_gap(&string_array),
                    XfjointomoLog::parse_mean_error(&string_array),
                    XfjointomoLog::parse_max_error(&string_array),
                );
                state.row_list.insert(boundary, row.clone());
                state.row_array.push(row);
            }
        }
        log_file.close_id(Some(&*reader_id));
        Ok(())
    }

    /// Java private `parseBoundary(String[])`.
    fn parse_boundary(string_array: &[String]) -> String {
        XfjointomoLog::trim_trailing_comma(java_lang_string_trim(&string_array[2]))
    }

    /// Java private `parseBestGap(String[])`.
    fn parse_best_gap(string_array: &[String]) -> String {
        java_lang_string_trim(&string_array[6]).to_string()
    }

    /// Java private `parseMeanError(String[])`.
    fn parse_mean_error(string_array: &[String]) -> String {
        XfjointomoLog::trim_trailing_comma(java_lang_string_trim(&string_array[11]))
    }

    /// Java private `parseMaxError(String[])`.
    fn parse_max_error(string_array: &[String]) -> String {
        java_lang_string_trim(&string_array[MAX_ERROR_INDEX]).to_string()
    }

    /// Java private `trimTrailingComma(String)`.
    fn trim_trailing_comma(string: &str) -> String {
        if string.ends_with(',') {
            if string.len() == 1 {
                return String::new();
            }
            return string[..string.len() - 1].to_string();
        }
        string.to_string()
    }
}

/// Java private final inner class `Row`.
#[derive(Clone)]
struct Row {
    /// Java private final `bestGap = new EtomoNumber(EtomoNumber.Type.DOUBLE)`.
    best_gap: EtomoNumber,
    /// Java private final `meanError`.
    mean_error: String,
    /// Java private final `maxError`.
    max_error: String,
}

impl Row {
    /// Java `Row(String, String, String)`.
    fn new(best_gap: &str, mean_error: String, max_error: String) -> Row {
        let mut row = Row {
            best_gap: EtomoNumber::new_with_type(Some(Type::Double)),
            mean_error,
            max_error,
        };
        row.best_gap.set_string(Some(best_gap));
        row
    }

    /// Java `getBestGap()`.
    fn get_best_gap(&self) -> String {
        self.best_gap.to_string()
    }

    /// Java `getMeanError()`.
    fn get_mean_error(&self) -> String {
        self.mean_error.clone()
    }

    /// Java `getMaxError()`.
    fn get_max_error(&self) -> String {
        self.max_error.clone()
    }

    /// Java `gapExists()`.
    fn gap_exists(&self) -> bool {
        !self.best_gap.equals_int(0)
    }
}
