//! `IMOD/Etomo/src/etomo/storage/FileFilterCollection.java`.
//!
//! A file filter that is a container for multiple file filters.
//! `FileFilterCollection extends javax.swing.filechooser.FileFilter`: the two
//! abstract methods are inherent methods and the `jdk::FileFilter`
//! implementation.  Built on the event dispatch thread, so the instance is an
//! `Rc` and the two collections are `RefCell`s.

use std::cell::RefCell;
use std::path::Path;
use std::rc::Rc;

use crate::imod::etomo::jdk::FileFilter;

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `public final class FileFilterCollection extends FileFilter`.
pub struct FileFilterCollection {
    /// Java private final `fileFilterList = new Vector<FileFilter>()`.
    file_filter_list: RefCell<Vec<Rc<dyn FileFilter>>>,
    /// Java private final `descriptionSet = new LinkedHashSet<String>()`: using
    /// a Set to avoid duplicate descriptions.  Insertion order is kept, and a
    /// null description is an element like any other.
    description_set: RefCell<Vec<Option<String>>>,
}

impl FileFilterCollection {
    /// Java's implicit `FileFilterCollection()`.
    pub fn new() -> Rc<FileFilterCollection> {
        Rc::new(FileFilterCollection {
            file_filter_list: RefCell::new(Vec::new()),
            description_set: RefCell::new(Vec::new()),
        })
    }

    /// Java `addFileFilter(FileFilter)`.  Adds a file filter.  The Java null
    /// check (`if (fileFilter == null) return;`) has no counterpart: the
    /// parameter cannot be null here.
    pub fn add_file_filter(&self, file_filter: Rc<dyn FileFilter>) {
        let description = file_filter.get_description();
        self.file_filter_list.borrow_mut().push(file_filter);
        let mut description_set = self.description_set.borrow_mut();
        if !description_set.contains(&description) {
            description_set.push(description);
        }
    }

    /// Java `accept(File)`.  Whether the given file is accepted by any of
    /// these filters.
    pub fn accept(&self, file: &Path) -> bool {
        let file_filter_list = self.file_filter_list.borrow().clone();
        let mut i = file_filter_list.iter();
        while let Some(next) = i.next() {
            if next.accept(file) {
                return true;
            }
        }
        false
    }

    /// Java `getDescription()`.  The description of these filters.  For
    /// example: "JPG and GIF Images, Nethack Save Files".
    pub fn get_description(&self) -> String {
        // `descriptionSet.toString()`: AbstractCollection's "[a, b]", with a
        // null element printed as "null".
        let description = format!(
            "[{}]",
            self.description_set
                .borrow()
                .iter()
                .map(|description| description.clone().unwrap_or_else(|| "null".to_string()))
                .collect::<Vec<String>>()
                .join(", ")
        );
        // Remove []
        description[1..description.len() - 1].to_string()
    }
}

impl FileFilter for FileFilterCollection {
    fn accept(&self, file: &Path) -> bool {
        FileFilterCollection::accept(self, file)
    }

    fn get_description(&self) -> Option<String> {
        Some(FileFilterCollection::get_description(self))
    }
}
