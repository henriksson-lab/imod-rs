//! `IMOD/Etomo/src/etomo/comscript/ComscriptState.java`.

use std::sync::Arc;

use crate::imod::etomo::r#type::file_key::FileKey;
use crate::imod::etomo::r#type::file_type::FileType;

/// Java `ComscriptState`.  The deprecated `FileType` getters return the shared
/// singleton the Java returns by reference.
pub trait ComscriptState {
    /// Java `getStartCommand`.
    fn get_start_command(&self) -> i32;
    /// Java `getEndCommand`.
    fn get_end_command(&self) -> i32;
    /// Java `getCommand(int)`.
    fn get_command(&self, command_index: i32) -> Option<String>;
    /// Java `getWatchedFile(int)`.
    fn get_watched_file(&self, command_index: i32) -> Option<String>;
    /// Java `getComscriptName`.
    fn get_comscript_name(&self) -> String;
    /// Java `getComscriptWatchedFile`.
    fn get_comscript_watched_file(&self) -> String;
    /// Java `getOutputImageFileType` (deprecated).
    fn get_output_image_file_type(&self) -> Option<Arc<FileType>>;
    /// Java `getOutputImageFileType2` (deprecated).
    fn get_output_image_file_type2(&self) -> Option<Arc<FileType>>;
    /// Java `getOutputImageFileType3` (deprecated).
    fn get_output_image_file_type3(&self) -> Option<Arc<FileType>>;
    /// Java `getOutputImageFileKey`.
    fn get_output_image_file_key(&self) -> Option<FileKey>;
    /// Java `getOutputImageFileKey2`.
    fn get_output_image_file_key2(&self) -> Option<FileKey>;
    /// Java `getOutputImageFileKey3`.
    fn get_output_image_file_key3(&self) -> Option<FileKey>;
}
