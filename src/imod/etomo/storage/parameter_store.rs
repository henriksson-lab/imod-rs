//! `IMOD/Etomo/src/etomo/storage/ParameterStore.java`.
//!
//! The properties of a data file (`.edf`, `.ejf`, the user's `.etomo`, ...),
//! read and written through the file's `LogFile`, which supplies the locking
//! and the once-per-session backup (`backupOnce` for `.etomo`,
//! `doubleBackupOnce` otherwise): the first store of a session backs the file
//! up, or finds nothing to back up, and no later store does.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use super::log_file::{Handle, LogFile, LogFileError};
use super::storable::Storable;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::etomo_director::USER_CONFIG_FILE_EXT;
use crate::imod::etomo::r#type::axis_id::AxisID;

/// Java `public final class ParameterStore`.
pub struct ParameterStore {
    /// Java `private final Properties properties = new Properties()`.
    properties: BTreeMap<String, String>,
    /// Java `private LogFile.Handle dataFile = null`, initialized in
    /// `initialize()`.
    data_file: Option<Arc<Handle>>,
    /// Java `private boolean autoStore = true`.
    auto_store: bool,
    /// Java `private int debug = 0`.
    debug: i32,
}

impl ParameterStore {
    /// Java private `ParameterStore()`.
    fn new() -> Self {
        Self {
            properties: BTreeMap::new(),
            data_file: None,
            auto_store: true,
            debug: 0,
        }
    }

    /// Java static `getInstance(BaseManager, AxisID, File)`.
    pub fn get_instance_manager(
        manager: Option<&'static dyn BaseManager>,
        axis_id: Option<AxisID>,
        param_file: Option<PathBuf>,
    ) -> Result<Option<Self>, LogFileError> {
        let Some(param_file) = param_file else {
            return Ok(None);
        };
        let mut instance = Self::new();
        instance.initialize(manager, axis_id, Some(param_file))?;
        Ok(Some(instance))
    }

    /// Java static `getInstance(BaseManager, AxisID, File)` called with a null
    /// manager (no emergency monitor for the data file).
    pub fn get_instance(param_file: Option<PathBuf>) -> Result<Option<Self>, LogFileError> {
        Self::get_instance_manager(None, None, param_file)
    }

    /// Java static `getFilelessInstance()`.
    pub fn get_fileless_instance() -> Result<Self, LogFileError> {
        let mut instance = Self::new();
        instance.initialize(None, None, None)?;
        Ok(instance)
    }

    /// Java private `initialize(BaseManager, AxisID, File)`.
    fn initialize(
        &mut self,
        manager: Option<&'static dyn BaseManager>,
        axis_id: Option<AxisID>,
        param_file: Option<PathBuf>,
    ) -> Result<(), LogFileError> {
        if let Some(param_file) = &param_file {
            self.data_file = Some(LogFile::get_instance_file(
                Some(param_file.as_path()),
                manager.map(|manager| manager.get_emergency_monitor(axis_id)),
            )?);
        }
        if let Some(data_file) = &self.data_file
            && data_file.exists()
        {
            let mut input_stream_id = None;
            let result = data_file.open_input_stream().and_then(|id| {
                input_stream_id = Some(id.clone());
                data_file.load(&mut self.properties, &id)
            });
            // catch (final LogFileException e) / catch (final IOException e)
            if let Err(e) = result {
                match e {
                    LogFileError::LogFile(_) | LogFileError::Io(_) => {
                        eprintln!("Unable to read {}", data_file.get_absolute_path());
                        eprintln!("{}", e);
                    }
                    // The LockException and FileException are thrown on.
                    other => return Err(other),
                }
            }
            if let Some(input_stream_id) = input_stream_id
                && !input_stream_id.is_empty()
            {
                data_file.close_id(Some(&*input_stream_id));
            }
        }
        Ok(())
    }

    /// Java `storeProperties()`.
    pub fn store_properties(&self) -> Result<(), LogFileError> {
        // If the file has not been set, don't save.
        if let Some(data_file) = &self.data_file {
            // synchronized (dataFile): this store's own lock, held by the caller.
            if !data_file.is_directory() {
                if data_file.get_name().ends_with(USER_CONFIG_FILE_EXT) {
                    data_file.backup_once()?;
                } else {
                    data_file.double_backup_once()?;
                }
            }
            if !data_file.exists() {
                data_file.create()?;
            }
            let output_stream_id = data_file.open_output_stream()?;
            let result = data_file.store(&self.properties, &output_stream_id);
            // Java closes the stream only after a successful store.
            result?;
            data_file.close_id(Some(&*output_stream_id));
        }
        Ok(())
    }

    /// Java `setDebug(boolean)`.
    pub fn set_debug(&mut self, debug: bool) {
        self.debug = i32::from(debug);
    }

    /// Java `setAutoStore(boolean)`.
    pub fn set_auto_store(&mut self, auto_store: bool) {
        self.auto_store = auto_store;
    }

    /// Java `save(Storable)`.
    pub fn save<T: Storable + ?Sized>(&mut self, storable: Option<&T>) -> Result<(), LogFileError> {
        // If the file has not been set, don't save.
        if self.data_file.is_some() {
            // let the storable overwrite its values
            if let Some(storable) = storable {
                storable.store(&mut self.properties);
            }
            if self.auto_store {
                self.store_properties()?;
            }
            if self.debug == 1 {
                eprintln!(
                    "save:JoinState.Join.Version={}",
                    self.properties
                        .get("JoinState.Join.Version")
                        .map(String::as_str)
                        .unwrap_or("null")
                );
            }
        }
        Ok(())
    }

    /// Java `load(Storable)`.  The storable is shared, as Java's is: it loads
    /// itself through `&self` (see `storable.rs`).  Java's
    /// `synchronized (dataFile)` is this store's own lock, which the caller
    /// already holds to reach `&self`.
    pub fn load<T: Storable + ?Sized>(&self, storable: &T) {
        storable.load(&self.properties);
    }

    /// Java `getAbsolutePath()`.
    pub fn get_absolute_path(&self) -> Option<String> {
        self.data_file
            .as_ref()
            .map(|data_file| data_file.get_absolute_path())
    }
}

#[cfg(test)]
mod tests {
    use super::ParameterStore;
    use crate::imod::etomo::storage::storable::Storable;
    use std::cell::RefCell;
    use std::collections::BTreeMap;

    struct Sample {
        value: RefCell<String>,
    }
    impl Storable for Sample {
        fn store(&self, p: &mut BTreeMap<String, String>) {
            p.insert("Value".into(), self.value.borrow().clone());
        }
        fn store_with_prepend(&self, p: &mut BTreeMap<String, String>, prepend: &str) {
            p.insert(format!("{prepend}.Value"), self.value.borrow().clone());
        }
        fn load(&self, p: &BTreeMap<String, String>) {
            *self.value.borrow_mut() = p.get("Value").cloned().unwrap_or_default();
        }
        fn load_with_prepend(&self, p: &BTreeMap<String, String>, prepend: &str) {
            *self.value.borrow_mut() = p
                .get(&format!("{prepend}.Value"))
                .cloned()
                .unwrap_or_default();
        }
    }
    #[test]
    fn fileless_store_load_matches_source() {
        let store = ParameterStore::get_fileless_instance().unwrap();
        let sample = Sample {
            value: RefCell::new(String::new()),
        };
        store.load(&sample);
        assert!(sample.value.borrow().is_empty());
    }
}
