//! `IMOD/Etomo/src/etomo/ui/DatasetInfoDisplay.java`.

/// Java `public interface DatasetInfoDisplay`.
pub trait DatasetInfoDisplay {
    /// Java `getDatasetName()`: null if empty.
    fn get_dataset_name(&self) -> Option<String>;

    /// Java `getDatasetAbsolutePath()`: null if empty.
    fn get_dataset_absolute_path(&self) -> Option<String>;
}
