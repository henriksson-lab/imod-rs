//! `IMOD/Etomo/src/etomo/process/RenamingStatus.java`.
//!
//! Shared by a monitor's own thread and the process thread that renames its
//! log file, so the `volatile` flags are atomics and the `synchronized`
//! methods take the instance lock, `lock`, which also guards `fileName`.

use crate::imod::etomo::storage::log_file::Handle;
use crate::imod::etomo::util::utilities;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// Java `RenamingStatus`.
pub struct RenamingStatus {
    /// Java `volatile` field `renaming`.
    renaming: AtomicBool,
    /// Java `volatile` field `renamed`.
    renamed: AtomicBool,
    /// Java `volatile` field `failed`.
    failed: AtomicBool,
    /// Java field `fileName`; the mutex is also the instance's intrinsic lock,
    /// which every `synchronized` method takes.
    file_name: Mutex<Option<String>>,
}

impl Default for RenamingStatus {
    fn default() -> RenamingStatus {
        RenamingStatus::new()
    }
}

impl RenamingStatus {
    /// Java `RenamingStatus()`.
    pub fn new() -> RenamingStatus {
        RenamingStatus {
            renaming: AtomicBool::new(false),
            renamed: AtomicBool::new(false),
            failed: AtomicBool::new(false),
            file_name: Mutex::new(None),
        }
    }

    /// Java `setRenaming()`.
    pub fn set_renaming(&self) {
        self.set_renaming_name(None);
    }

    /// Java `setRenaming(LogFile.Handle)`.
    pub fn set_renaming_handle(&self, log_file: Option<&Arc<Handle>>) {
        if let Some(log_file) = log_file {
            self.set_renaming_name(Some(&log_file.get_name()));
        } else {
            self.set_renaming_name(None);
        }
    }

    /// Java `setRenaming(File)`.
    pub fn set_renaming_file(&self, file: Option<&Path>) {
        if let Some(file) = file {
            self.set_renaming_name(Some(&utilities::java_io_file_get_name(
                &file.to_string_lossy(),
            )));
        } else {
            self.set_renaming_name(None);
        }
    }

    /// Java `synchronized setRenaming(String)`.
    pub fn set_renaming_name(&self, file_name: Option<&str>) {
        let mut this_file_name = self.file_name.lock().unwrap();
        self.failed.store(false, Ordering::SeqCst);
        self.renaming.store(true, Ordering::SeqCst);
        self.renamed.store(false, Ordering::SeqCst);
        *this_file_name = file_name.map(str::to_string);
    }

    /// Java `synchronized setFailed`.
    pub fn set_failed(&self) {
        let _lock = self.file_name.lock().unwrap();
        self.failed.store(true, Ordering::SeqCst);
        self.renaming.store(false, Ordering::SeqCst);
        self.renamed.store(false, Ordering::SeqCst);
    }

    /// Java `synchronized renamed`.
    pub fn renamed(&self) {
        let _lock = self.file_name.lock().unwrap();
        self.renaming.store(false, Ordering::SeqCst);
        self.failed.store(false, Ordering::SeqCst);
        self.renamed.store(true, Ordering::SeqCst);
    }

    /// Java `synchronized reset`.
    pub fn reset(&self) {
        let mut file_name = self.file_name.lock().unwrap();
        self.failed.store(false, Ordering::SeqCst);
        self.renaming.store(false, Ordering::SeqCst);
        self.renamed.store(false, Ordering::SeqCst);
        *file_name = None;
    }

    /// Java `isRenamed`.
    pub fn is_renamed(&self) -> bool {
        self.renamed.load(Ordering::SeqCst)
    }

    /// Java `isRenaming`.
    pub fn is_renaming(&self) -> bool {
        self.renaming.load(Ordering::SeqCst)
    }

    /// Java `synchronized getFileName`.
    pub fn get_file_name(&self) -> Option<String> {
        self.file_name.lock().unwrap().clone()
    }

    /// Java `isFailed`.
    pub fn is_failed(&self) -> bool {
        self.failed.load(Ordering::SeqCst)
    }
}
