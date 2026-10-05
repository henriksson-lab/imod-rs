//! `IMOD/Etomo/src/etomo/type/FileKey.java`.
//!
//! The superclass of `etomo/type/FileType.java`.  Java's inheritance is mirrored by
//! composition: `FileType` holds a `FileKey` and reaches it through `Deref`, the same
//! shape `etomo/type/EtomoNumber.java`'s module uses for `ConstEtomoNumber`.
//!
//! Java's `getFileName(BaseManager, AxisID)` is virtual: on a `FileType` it runs
//! `FileType`'s override.  A `FileKey` embedded in a `FileType` therefore keeps a weak
//! link to that `FileType` (`file_type`), set by `FileType`'s constructors, and
//! `get_file_name` dispatches through it; a clone of that `FileKey` (what the commands'
//! `getOutputImageFileKey` hand out) keeps the link, as the Java reference keeps the
//! object's class.
#![allow(dead_code)]

use std::sync::{LazyLock, Weak};

use super::axis_id::AxisID;
use super::file_type::FileType;
use crate::imod::etomo::base_manager::BaseManager;

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
#[derive(Clone)]
pub struct FileKey {
    /// Java field `imodManagerKey`.
    imod_manager_key: Option<String>,
    /// Java field `imodManagerKey2`.
    imod_manager_key2: Option<String>,
    /// Java field `descr`.
    descr: Option<String>,
    /// Java field `debug`.
    debug: bool,
    /// The `FileType` this key is the superclass part of (an empty `Weak` for a plain
    /// `FileKey`): the run-time class Java's virtual `getFileName` dispatches on.
    file_type: Weak<FileType>,
}

impl std::fmt::Debug for FileKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FileKey")
            .field("imod_manager_key", &self.imod_manager_key)
            .field("imod_manager_key2", &self.imod_manager_key2)
            .field("descr", &self.descr)
            .field("debug", &self.debug)
            .finish()
    }
}

impl PartialEq for FileKey {
    fn eq(&self, other: &FileKey) -> bool {
        self.imod_manager_key == other.imod_manager_key
            && self.imod_manager_key2 == other.imod_manager_key2
            && self.descr == other.descr
            && self.debug == other.debug
            && Weak::ptr_eq(&self.file_type, &other.file_type)
    }
}

impl Eq for FileKey {}

impl FileKey {
    /// Java `FileKey(String)`.
    pub fn new(imod_manager_key: Option<&str>) -> FileKey {
        FileKey {
            imod_manager_key: imod_manager_key.map(|key| key.to_string()),
            imod_manager_key2: None,
            descr: None,
            debug: false,
            file_type: Weak::new(),
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
            file_type: Weak::new(),
        }
    }

    /// Sets the link to the `FileType` this key belongs to (see the module comment).
    /// Called only by `FileType`'s constructors.
    pub(super) fn with_file_type(mut self, file_type: Weak<FileType>) -> FileKey {
        self.file_type = file_type;
        self
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
    /// and returns the description; on a `FileType` it is `FileType`'s override.
    pub fn get_file_name(
        &self,
        manager: Option<&'static dyn BaseManager>,
        axis_id: Option<AxisID>,
    ) -> Option<String> {
        if let Some(file_type) = self.file_type.upgrade() {
            return file_type.get_file_name(manager, axis_id);
        }
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
