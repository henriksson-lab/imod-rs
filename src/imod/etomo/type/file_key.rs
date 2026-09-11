//! `IMOD/Etomo/src/etomo/type/FileKey.java`.
//!
//! The superclass of `etomo/type/FileType.java`.  Java's inheritance is mirrored by
//! composition: `FileType` holds a `FileKey` and reaches it through `Deref`, the same
//! shape `etomo/type/EtomoNumber.java`'s module uses for `ConstEtomoNumber`.
#![allow(dead_code)]

use std::sync::LazyLock;

use crate::imod::etomo::process::imod_manager;

/// Java `AVERAGED_VOLUMES`.
pub static AVERAGED_VOLUMES: LazyLock<FileKey> =
    LazyLock::new(|| FileKey::new(Some(imod_manager::AVG_VOL_KEY)));
/// Java `NAD_TEST_VARYING_ITERATIONS`.
pub static NAD_TEST_VARYING_ITERATIONS: LazyLock<FileKey> =
    LazyLock::new(|| FileKey::new(Some(imod_manager::VARYING_ITERATION_TEST_KEY)));
/// Java `NAD_TEST_VARYING_K`.
pub static NAD_TEST_VARYING_K: LazyLock<FileKey> =
    LazyLock::new(|| FileKey::new(Some(imod_manager::VARYING_K_TEST_KEY)));
/// Java `POSITIONING_SAMPLE`.
pub static POSITIONING_SAMPLE: LazyLock<FileKey> =
    LazyLock::new(|| FileKey::new(Some(imod_manager::SAMPLE_KEY)));
/// Java `REFERENCE_VOLUMES`.
pub static REFERENCE_VOLUMES: LazyLock<FileKey> =
    LazyLock::new(|| FileKey::new(Some(imod_manager::REF_KEY)));
/// Java `TRIAL_TOMOGRAM`.  This file name is created by the user.
pub static TRIAL_TOMOGRAM: LazyLock<FileKey> =
    LazyLock::new(|| FileKey::new(Some(imod_manager::TRIAL_TOMOGRAM_KEY)));

/// Java `FileKey`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FileKey {
    /// Java field `imodManagerKey`.
    imod_manager_key: Option<String>,
    /// Java field `imodManagerKey2`.
    imod_manager_key2: Option<String>,
    /// Java field `descr`.
    descr: Option<String>,
    /// Java field `debug`.
    debug: bool,
}

impl FileKey {
    /// Java `FileKey(String)`.
    pub fn new(imod_manager_key: Option<&str>) -> FileKey {
        FileKey {
            imod_manager_key: imod_manager_key.map(|key| key.to_string()),
            imod_manager_key2: None,
            descr: None,
            debug: false,
        }
    }

    /// Java `FileKey(String, String, String)`.
    pub fn new_with_descr(
        imod_manager_key: Option<&str>,
        imod_manager_key2: Option<&str>,
        descr: Option<&str>,
    ) -> FileKey {
        FileKey {
            imod_manager_key: imod_manager_key.map(|key| key.to_string()),
            imod_manager_key2: imod_manager_key2.map(|key| key.to_string()),
            descr: descr.map(|descr| descr.to_string()),
            debug: false,
        }
    }

    /// Java `getDescription`.
    pub fn get_description(&self) -> Option<String> {
        if let Some(descr) = &self.descr {
            return Some(descr.clone());
        }
        if let Some(imod_manager_key) = &self.imod_manager_key {
            return Some(imod_manager_key.clone());
        }
        if let Some(imod_manager_key2) = &self.imod_manager_key2 {
            return Some(imod_manager_key2.clone());
        }
        None
    }

    /// Java `getFileName(BaseManager, AxisID)`.  The base class ignores both parameters
    /// and returns the description, so there is nothing here that needs
    /// etomo/BaseManager.java; `FileType` overrides it.
    pub fn get_file_name(&self) -> Option<String> {
        self.get_description()
    }

    /// Java `getImodManagerKey`.
    pub fn get_imod_manager_key(&self) -> Option<&str> {
        self.imod_manager_key.as_deref()
    }

    /// Java `setDebug`.
    pub fn set_debug(&mut self, debug: bool) {
        self.debug = debug;
    }

    /// Java `isDebug`.
    pub fn is_debug(&self) -> bool {
        self.debug
    }

    /// Java `getImodManagerKey2`.
    pub fn get_imod_manager_key2(&self) -> Option<&str> {
        self.imod_manager_key2.as_deref()
    }

    /// Java `getDescr`.
    pub fn get_descr(&self) -> Option<&str> {
        self.descr.as_deref()
    }
}
