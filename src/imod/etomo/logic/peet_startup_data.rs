//! `IMOD/Etomo/src/etomo/logic/PeetStartupData.java`.
//!
//! The data the PEET startup dialog collects: the dataset directory, the base name,
//! and an optional project to copy from.  Paths are made absolute relative to the
//! directory in which etomo was run.

use std::path::{Path, PathBuf};

use crate::imod::etomo::etomo_director;
use crate::imod::etomo::storage::peet_file_filter::PeetFileFilter;
use crate::imod::etomo::r#type::const_etomo_number::java_lang_string_matches_whitespace;
use crate::imod::etomo::r#type::data_file_type::DataFileType;
use crate::imod::etomo::r#type::extension;
use crate::imod::etomo::util::dataset_files;
use crate::imod::etomo::util::file_path::FilePath;
use crate::imod::etomo::util::utilities;

/// Java `public static final String rcsid`.
pub const RCSID: &str = "$Id:$";

/// Java `public final class PeetStartupData`.
#[derive(Clone, Debug, Default)]
pub struct PeetStartupData {
    /// Java private `directory`, initially null.
    directory: Option<PathBuf>,
    /// Java private `copyFrom`, initially null.
    copy_from: Option<PathBuf>,
    /// Java private `baseName`, initially null.
    base_name: Option<String>,
}

impl PeetStartupData {
    /// Java implicit constructor.
    pub fn new() -> PeetStartupData {
        PeetStartupData::default()
    }

    /// Java `setDirectory(String)`.  Sets directory to an absolute path.  If input
    /// isn't an absolute path, it uses the directory in which etomo was run to make an
    /// absolute path.  Null has no effect.
    pub fn set_directory(&mut self, input: Option<&str>) {
        let Some(input) = input else {
            return;
        };
        self.directory = Some(FilePath::build_absolute_file_string_string(
            etomo_director::INSTANCE.get_original_user_dir().as_deref(),
            input,
        ));
    }

    /// Java `setCopyFrom(String)`.  Sets copyFrom to an absolute path, as
    /// `setDirectory`.  Null has no effect.
    pub fn set_copy_from(&mut self, input: Option<&str>) {
        let Some(input) = input else {
            return;
        };
        self.copy_from = Some(FilePath::build_absolute_file_string_string(
            etomo_director::INSTANCE.get_original_user_dir().as_deref(),
            input,
        ));
    }

    /// Java `setBaseName(String)`.
    pub fn set_base_name(&mut self, input: Option<&str>) {
        self.base_name = input.map(str::to_owned);
    }

    /// Java `validate()`.  Returns an error message or null if valid.
    pub fn validate(&self) -> Option<String> {
        let Some(directory) = &self.directory else {
            return Some("Missing required entry:  directory.".to_owned());
        };
        match &self.base_name {
            None => return Some("Missing required entry:  baseName.".to_owned()),
            Some(base_name) if java_lang_string_matches_whitespace(base_name) => {
                return Some("Missing required entry:  baseName.".to_owned());
            }
            Some(_) => {}
        }
        // Only one .epe file per directory
        // OK to use directory if it contains an .epe file of the same name
        let filter = PeetFileFilter::new_with_accept_directories(false);
        // `directory.listFiles(filter)`: null when the directory cannot be listed.
        let param_files: Option<Vec<PathBuf>> = std::fs::read_dir(directory).ok().map(|entries| {
            entries
                .flatten()
                .map(|entry| entry.path())
                .filter(|path| filter.accept(path))
                .collect()
        });
        if let Some(param_files) = &param_files
            && !param_files.is_empty()
            && (param_files.len() > 1
                || dataset_files::get_peet_root_name(&utilities::java_io_file_get_name(
                    &param_files[0].to_string_lossy(),
                )) != *self.base_name.as_deref().unwrap_or(""))
        {
            return Some(format!(
                "The directory {} can contain only one {} file.",
                utilities::java_io_file_get_absolute_path(&directory.to_string_lossy()),
                DataFileType::Peet.extension().unwrap_or("null")
            ));
        }
        if let Some(copy_from) = &self.copy_from
            && copy_from.parent().map(Path::to_path_buf).as_ref() == Some(directory)
            && !utilities::java_io_file_get_name(&copy_from.to_string_lossy()).ends_with(&format!(
                "{}{}",
                extension::EXTENSION_DIVIDER,
                extension::CLASS.prm
            ))
        {
            return Some("Cannot duplicate a project in the same directory.".to_owned());
        }
        None
    }

    /// Java `getBaseName()`.
    pub fn get_base_name(&self) -> Option<String> {
        self.base_name.clone()
    }

    /// Java `getDirectory()`.  Returns the absolute path of directory.
    pub fn get_directory(&self) -> Option<String> {
        self.directory
            .as_ref()
            .map(|directory| utilities::java_io_file_get_absolute_path(&directory.to_string_lossy()))
    }

    /// Java `getCopyFrom()`.  Returns the absolute path of copyFrom.
    pub fn get_copy_from(&self) -> Option<String> {
        self.copy_from
            .as_ref()
            .map(|copy_from| utilities::java_io_file_get_absolute_path(&copy_from.to_string_lossy()))
    }

    /// Java `isCopyFrom()`.
    pub fn is_copy_from(&self) -> bool {
        self.copy_from.is_some()
    }

    /// Java `getParamFile()`.  Builds and returns the param file.
    pub fn get_param_file(&self) -> Option<PathBuf> {
        let (Some(directory), Some(base_name)) = (&self.directory, &self.base_name) else {
            return None;
        };
        Some(directory.join(format!(
            "{base_name}{}",
            DataFileType::Peet.extension().unwrap_or("null")
        )))
    }
}
