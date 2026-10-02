//! `IMOD/Etomo/src/etomo/storage/TomogramFileFilter.java`.
//!
//! `TomogramFileFilter extends ExtensibleFileFilter implements
//! java.io.FileFilter`: the superclass is embedded as `base` and reached
//! through `Deref`.  The `accept(File)` override is the inherent `accept`, and
//! with `getDescription` it is also the `jdk::FileFilter` implementation, so
//! the filter can be handed to a file chooser as `Rc<dyn FileFilter>`.  Built
//! on the event dispatch thread, so instances are `Rc` and the mutable
//! `allowAll` is a `Cell`.

use std::cell::Cell;
use std::path::Path;
use std::rc::Rc;
use std::sync::Mutex;

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::jdk::FileFilter;
use crate::imod::etomo::storage::extensible_file_filter::{
    ExtensibleFileFilter, ExtensibleFileFilterVirtual,
};
use crate::imod::etomo::r#type::extension;
use crate::imod::etomo::util::utilities;

/// Java private static final `extraExtensionList` (deprecated 5/6/2019 - not
/// used).  A class-wide list, shared by every instance.
static EXTRA_EXTENSION_LIST: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// Java `public class TomogramFileFilter extends ExtensibleFileFilter`.
pub struct TomogramFileFilter {
    /// Java superclass `ExtensibleFileFilter` state.
    pub base: ExtensibleFileFilter,
    /// Java private `allowAll = false` (deprecated 5/6/2019 - not used).
    allow_all: Cell<bool>,
}

impl std::ops::Deref for TomogramFileFilter {
    type Target = ExtensibleFileFilter;

    fn deref(&self) -> &ExtensibleFileFilter {
        &self.base
    }
}

impl TomogramFileFilter {
    /// Java private `TomogramFileFilter(BaseManager, boolean, boolean)`.
    fn new(
        manager: &'static dyn BaseManager,
        use_all_file_styles: bool,
        debug: bool,
    ) -> TomogramFileFilter {
        TomogramFileFilter {
            base: ExtensibleFileFilter::new(Some(manager), true, use_all_file_styles, debug),
            allow_all: Cell::new(false),
        }
    }

    /// Java static `getInstance(BaseManager)`.
    pub fn get_instance(manager: &'static dyn BaseManager) -> Rc<TomogramFileFilter> {
        let mut instance = TomogramFileFilter::new(manager, false, false);
        instance.setup();
        Rc::new(instance)
    }

    /// Java static `getDebugInstance(BaseManager)`.
    pub fn get_debug_instance(manager: &'static dyn BaseManager) -> Rc<TomogramFileFilter> {
        let mut instance = TomogramFileFilter::new(manager, false, true);
        instance.setup();
        Rc::new(instance)
    }

    /// Java static `getAllImageFilenameStyleInstance(BaseManager)`.
    pub fn get_all_image_filename_style_instance(
        manager: &'static dyn BaseManager,
    ) -> Rc<TomogramFileFilter> {
        let mut instance = TomogramFileFilter::new(manager, true, false);
        instance.setup();
        Rc::new(instance)
    }

    /// Java private `setup()`.
    fn setup(&mut self) {
        self.base.base.setup(
            None,
            None,
            None,
            None,
            Some(&[
                &extension::CLASS.mrc,
                &extension::CLASS.hdf,
                &extension::CLASS.rec,
                &extension::CLASS.flip,
                &extension::CLASS.sqz,
                &extension::CLASS.join,
                &extension::CLASS.flat,
            ]),
            None,
        );
    }

    /// Java `accept(File)`.  Accept the if it is a directory or has one of the
    /// defined extensions (.rec, .flip, .sqz, or .join).  Also accept if
    /// allowAll is on, or if its extension is in extraExtensionList.
    pub fn accept(&self, file: &Path) -> bool {
        if self.allow_all.get() {
            return true;
        }
        if self.base.base.accept(Some(file)) {
            return true;
        }
        let file_path = utilities::java_io_file_get_absolute_path(&file.to_string_lossy());
        let extra_extension_list = EXTRA_EXTENSION_LIST.lock().unwrap();
        let mut iterator = extra_extension_list.iter();
        while let Some(next) = iterator.next() {
            if file_path.ends_with(next.as_str()) {
                return true;
            }
        }
        false
    }

    /// Java inherited `ExtensionFileFilter.accept(File, String)`, whose virtual
    /// `accept(File)` call resolves to this class's override.  Its own
    /// directory test repeats the first test `super.accept(File)` makes, so the
    /// result is `accept(new File(dir, name))`.
    pub fn accept_in_dir(&self, dir: Option<&Path>, name: Option<&str>) -> bool {
        let Some(name) = name else {
            return false;
        };
        let file = utilities::java_io_file_new(
            &dir.map(|dir| dir.to_string_lossy().to_string())
                .unwrap_or("null".to_string()),
            name,
        );
        self.accept(Path::new(&file))
    }

    /// Java package-private `getExtraExtensionListSize()`.
    pub fn get_extra_extension_list_size(&self) -> usize {
        EXTRA_EXTENSION_LIST.lock().unwrap().len()
    }

    /// Java `getDescription()`.
    pub fn get_description(&self) -> &'static str {
        "Tomogram file (.mrc, .hdf, .rec, .flip, .sqz, .join, .flat)"
    }
}

impl ExtensibleFileFilterVirtual for TomogramFileFilter {
    /// Java `addExtension(File)` (deprecated 5/6/2019 - not used).  If accept
    /// fails with this file, adds its extension to extraExtensionList.  If the
    /// file has no extension, turn on allowAll.
    fn add_extension(&self, file: &Path) {
        if self.accept(file) {
            return;
        }
        // New extension
        let file_name = utilities::java_io_file_get_name(&file.to_string_lossy());
        match file_name.rfind('.') {
            None => self.allow_all.set(true),
            Some(extension_index) => EXTRA_EXTENSION_LIST
                .lock()
                .unwrap()
                .push(file_name[extension_index..].to_string()),
        }
    }
}

impl FileFilter for TomogramFileFilter {
    fn accept(&self, file: &Path) -> bool {
        TomogramFileFilter::accept(self, file)
    }

    fn get_description(&self) -> Option<String> {
        Some(TomogramFileFilter::get_description(self).to_string())
    }
}
