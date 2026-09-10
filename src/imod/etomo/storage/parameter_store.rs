//! `IMOD/Etomo/src/etomo/storage/ParameterStore.java`.
//!
//! Java `LogFile` locking/backup is an external filesystem boundary here.  The
//! source unit's property ownership, fileless behavior, automatic store, and
//! `.etomo` versus normal backup policy are retained.
#![allow(dead_code)]

use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use super::storable::Storable;

pub struct ParameterStore {
    properties: BTreeMap<String, String>,
    data_file: Option<PathBuf>,
    auto_store: bool,
    debug: i32,
}

impl ParameterStore {
    /// Java `ParameterStore()`.
    pub fn new() -> Self {
        Self {
            properties: BTreeMap::new(),
            data_file: None,
            auto_store: true,
            debug: 0,
        }
    }

    /// Java `getInstance(BaseManager, AxisID, File)`; manager/axis only supply
    /// the Java LogFile emergency monitor and are therefore a JVM UI boundary.
    pub fn get_instance(param_file: Option<PathBuf>) -> io::Result<Option<Self>> {
        let Some(param_file) = param_file else {
            return Ok(None);
        };
        let mut instance = Self::new();
        instance.initialize(Some(param_file))?;
        Ok(Some(instance))
    }

    /// Java `getFilelessInstance()`.
    pub fn get_fileless_instance() -> io::Result<Self> {
        let mut instance = Self::new();
        instance.initialize(None)?;
        Ok(instance)
    }

    /// Java `initialize(BaseManager, AxisID, File)`.
    pub fn initialize(&mut self, param_file: Option<PathBuf>) -> io::Result<()> {
        self.data_file = param_file;
        if let Some(data_file) = &self.data_file {
            if data_file.exists() {
                let input = fs::read_to_string(data_file)?;
                for line in input.lines() {
                    if line.starts_with('#') || line.starts_with('!') {
                        continue;
                    }
                    if let Some((key, value)) = line.split_once('=') {
                        self.properties
                            .insert(key.trim().to_owned(), value.trim().to_owned());
                    } else if let Some((key, value)) = line.split_once(':') {
                        self.properties
                            .insert(key.trim().to_owned(), value.trim().to_owned());
                    }
                }
            }
        }
        Ok(())
    }

    /// Java `storeProperties()` including its one/double backup decision.
    pub fn store_properties(&self) -> io::Result<()> {
        let Some(data_file) = &self.data_file else {
            return Ok(());
        };
        if data_file.is_dir() {
            return Ok(());
        }
        if data_file.exists() {
            if data_file.to_string_lossy().ends_with(".etomo") {
                let backup = PathBuf::from(format!("{}~", data_file.display()));
                if !backup.exists() {
                    fs::copy(data_file, backup)?;
                }
            } else {
                let backup = PathBuf::from(format!("{}~", data_file.display()));
                let older_backup = PathBuf::from(format!("{}~~", data_file.display()));
                if backup.exists() {
                    fs::rename(&backup, older_backup)?;
                }
                fs::copy(data_file, backup)?;
            }
        }
        let mut output = String::new();
        for (key, value) in &self.properties {
            output.push_str(key);
            output.push('=');
            output.push_str(value);
            output.push('\n');
        }
        fs::write(data_file, output)
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
    pub fn save<T: Storable>(&mut self, storable: Option<&T>) -> io::Result<()> {
        if self.data_file.is_some() {
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
                        .unwrap_or("")
                );
            }
        }
        Ok(())
    }
    /// Java `load(Storable)`.
    pub fn load<T: Storable>(&self, storable: &mut T) {
        storable.load(&self.properties);
    }
    /// Java `getAbsolutePath()`.
    pub fn get_absolute_path(&self) -> Option<&Path> {
        self.data_file.as_deref()
    }
}

impl Default for ParameterStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::ParameterStore;
    use crate::imod::etomo::storage::storable::Storable;
    use std::collections::BTreeMap;

    struct Sample {
        value: String,
    }
    impl Storable for Sample {
        fn store(&self, p: &mut BTreeMap<String, String>) {
            p.insert("Value".into(), self.value.clone());
        }
        fn store_with_prepend(&self, p: &mut BTreeMap<String, String>, prepend: &str) {
            p.insert(format!("{prepend}.Value"), self.value.clone());
        }
        fn load(&mut self, p: &BTreeMap<String, String>) {
            self.value = p.get("Value").cloned().unwrap_or_default();
        }
        fn load_with_prepend(&mut self, p: &BTreeMap<String, String>, prepend: &str) {
            self.value = p
                .get(&format!("{prepend}.Value"))
                .cloned()
                .unwrap_or_default();
        }
    }
    #[test]
    fn fileless_store_load_matches_source() {
        let store = ParameterStore::get_fileless_instance().unwrap();
        let mut sample = Sample {
            value: String::new(),
        };
        store.load(&mut sample);
        assert!(sample.value.is_empty());
    }
}
