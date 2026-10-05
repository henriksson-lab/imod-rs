//! `IMOD/Etomo/src/etomo/storage/DatasetFileBuilder.java`.
//!
//! Builds a dataset file from the manager's meta data once the param file is set, or
//! from the root name and location shown by a `DatasetInfoDisplay` before then.  An
//! event dispatch thread object: the display is the batchruntomo dialog.

use std::cell::{Cell, RefCell};
use std::path::PathBuf;
use std::rc::Rc;

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::file_type::FileType;
use crate::imod::etomo::ui::dataset_info_display::DatasetInfoDisplay;

/// Java `public final class DatasetFileBuilder`.
pub struct DatasetFileBuilder {
    /// Java private final `manager`.
    manager: &'static dyn BaseManager,
    /// Java private `datasetInfoDisplay`, initially null.
    dataset_info_display: RefCell<Option<Rc<dyn DatasetInfoDisplay>>>,
    /// Java private `loadedParamFile`, initially false.
    loaded_param_file: Cell<bool>,
}

impl DatasetFileBuilder {
    /// Java `DatasetFileBuilder(BaseManager)`.
    pub fn new(manager: &'static dyn BaseManager) -> DatasetFileBuilder {
        DatasetFileBuilder {
            manager,
            dataset_info_display: RefCell::new(None),
            loaded_param_file: Cell::new(false),
        }
    }

    /// Java `setDatasetInfoDisplay(DatasetInfoDisplay)`.
    pub fn set_dataset_info_display(&self, dataset_info_display: Option<Rc<dyn DatasetInfoDisplay>>) {
        *self.dataset_info_display.borrow_mut() = dataset_info_display;
    }

    /// Java `buildFile(FileType, AxisID)`.
    pub fn build_file(&self, file_type: Option<&FileType>, axis_id: AxisID) -> Option<PathBuf> {
        let file_type = file_type?;
        // Use meta data if the param file is set up, or if there is no display.
        // Once loadedParamFile is set to true, it won't change again.
        let dataset_info_display = self.dataset_info_display.borrow().clone();
        let Some(dataset_info_display) = dataset_info_display else {
            return file_type.get_file(Some(self.manager), Some(axis_id));
        };
        if self.loaded_param_file.get() || {
            self.loaded_param_file
                .set(self.manager.is_loaded_param_file());
            self.loaded_param_file.get()
        } {
            return file_type.get_file(Some(self.manager), Some(axis_id));
        }
        let mut dataset_name = dataset_info_display.get_dataset_name();
        let mut dataset_absolute_path = dataset_info_display.get_dataset_absolute_path();
        if dataset_name.is_none() && dataset_absolute_path.is_none() {
            return file_type.get_file(Some(self.manager), Some(axis_id));
        }
        if dataset_name.is_none() {
            dataset_name = self
                .manager
                .get_base_meta_data()
                .and_then(|meta_data| meta_data.get_name());
        }
        if dataset_absolute_path.is_none() {
            dataset_absolute_path = self.manager.get_property_user_dir();
        }
        // Using data from the interface.
        file_type.get_file_from_root_name(
            dataset_name.as_deref(),
            Some(axis_id),
            dataset_absolute_path.as_deref(),
        )
    }
}
