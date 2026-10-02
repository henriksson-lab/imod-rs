//! `IMOD/Etomo/src/etomo/ui/swing/FileTextFieldInterface.java`.

use std::path::PathBuf;
use std::rc::Rc;

use crate::imod::etomo::jdk::FileFilter;

/// Java `rcsid`.
pub const RCSID: &str = "$Id:$";

/// Java `FileTextFieldInterface`.  A Java `File` is its path.
pub trait FileTextFieldInterface {
    /// Java `getFile()`.
    fn get_file(&self) -> Option<PathBuf>;

    /// Java `setFile(File)`.
    fn set_file(&self, file: Option<PathBuf>);

    /// Java `getFileFilter()`.
    fn get_file_filter(&self) -> Option<Rc<dyn FileFilter>>;
}
