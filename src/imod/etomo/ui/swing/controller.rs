//! `IMOD/Etomo/src/etomo/ui/swing/Controller.java`.

use std::path::PathBuf;

/// Java package-private `interface Controller`.
pub trait Controller {
    /// Java `isControl()`.
    fn is_control(&self) -> bool;

    /// Java `setEditable(boolean)`.
    fn set_editable(&self, editable: bool);

    /// Java `setEnabled(boolean)`.
    fn set_enabled(&self, enabled: bool);

    /// Java `selectFile()`; `null` is `None`.
    fn select_file(&self) -> Option<PathBuf>;

    /// Java `selectMultipleFiles()`; a `null` array is `None`.
    fn select_multiple_files(&self) -> Option<Vec<PathBuf>>;
}
