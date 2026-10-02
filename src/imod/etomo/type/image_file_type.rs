//! `IMOD/Etomo/src/etomo/type/ImageFileType.java`.
//!
//! Description: A type of file associated with a process-level panel.
//!
//! Copyright: Copyright 2008 - 2015 by the Regents of the University of Colorado
//!
//! Organization: Dept. of MCD Biology, University of Colorado

use super::file_type::{self, FileType};
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::process::imod_manager;
use std::path::PathBuf;
use std::sync::Arc;

/// Java `ImageFileType`, a typesafe enum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageFileType {
    /// Java `TRIM_VOL_OUTPUT`.
    TrimVolOutput,
    /// Java `SQUEEZE_VOL_OUTPUT`.
    SqueezeVolOutput,
    /// Java `FLATTEN_OUTPUT`.
    FlattenOutput,
}

impl ImageFileType {
    /// Java `getImodManagerKey`.
    pub fn get_imod_manager_key(&self) -> &'static str {
        match self {
            ImageFileType::TrimVolOutput => imod_manager::TRIMMED_VOLUME_KEY,
            ImageFileType::SqueezeVolOutput => imod_manager::SQUEEZED_VOLUME_KEY,
            ImageFileType::FlattenOutput => imod_manager::FLAT_VOLUME_KEY,
        }
    }

    /// Java `getFileName(BaseManager)`.  (The source's null-manager check has no Rust
    /// counterpart: the manager is a reference.)
    pub fn get_file_name(&self, manager: &'static dyn BaseManager) -> Option<String> {
        let file_type = self.get_file_type()?;
        file_type.get_file_name(Some(manager), None)
    }

    /// Java `getFile(BaseManager)`.
    pub fn get_file(&self, manager: &'static dyn BaseManager) -> Option<PathBuf> {
        let file_type = self.get_file_type()?;
        file_type.get_file(Some(manager), None)
    }

    /// Java `getFileType`.
    pub fn get_file_type(&self) -> Option<Arc<FileType>> {
        match self {
            ImageFileType::TrimVolOutput => Some(Arc::clone(&file_type::CLASS.trim_vol_output)),
            ImageFileType::SqueezeVolOutput => {
                Some(Arc::clone(&file_type::CLASS.squeeze_vol_output))
            }
            ImageFileType::FlattenOutput => Some(Arc::clone(&file_type::CLASS.flatten_output)),
        }
    }
}

/// Java `toString`: the ImodManager key.
impl std::fmt::Display for ImageFileType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.get_imod_manager_key())
    }
}
