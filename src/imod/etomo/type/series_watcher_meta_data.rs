//! `IMOD/Etomo/src/etomo/type/SeriesWatcherMetaData.java`.
//!
//! The rows of a serieswatcher project file, read by the serieswatcher monitor (a
//! process thread) and handed to the dataset table (event dispatch thread); the row
//! map sits behind a lock.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use super::batch_run_tomo_meta_data;
use super::batch_run_tomo_row_meta_data::BatchRunTomoRowMetaData;
use super::const_etomo_number::{java_lang_string_matches_whitespace, java_lang_string_trim};
use super::ordered_hash_map::OrderedHashMap;
use super::table_reference::TableReference;
use crate::imod::etomo::storage::storable::Storable;

/// Java `public class SeriesWatcherMetaData implements Storable`.
pub struct SeriesWatcherMetaData {
    /// Java private final `rowMetaDataMap`.
    row_meta_data_map: Mutex<OrderedHashMap<String, Arc<BatchRunTomoRowMetaData>>>,
    /// Java private final `tableReference`.
    table_reference: Arc<TableReference>,
}

impl SeriesWatcherMetaData {
    /// Java `SeriesWatcherMetaData(TableReference)`.
    pub fn new(table_reference: Arc<TableReference>) -> SeriesWatcherMetaData {
        SeriesWatcherMetaData {
            row_meta_data_map: Mutex::new(OrderedHashMap::new()),
            table_reference,
        }
    }

    /// Java `load(Properties, String)`.
    pub fn load_with_prepend(&self, props: &mut BTreeMap<String, String>, parent_prepend: &str) {
        let mut row_meta_data_map = self.row_meta_data_map.lock().unwrap();
        // reset
        row_meta_data_map.clear();
        // load
        let prepend = self.create_prepend(Some(parent_prepend));
        let _group = format!("{}.", prepend);
        self.table_reference.load(props, Some(&prepend));
        for stack_id in self.table_reference.id_iterator() {
            if !BatchRunTomoRowMetaData::is_row_number_null_in(props, Some(&prepend), &stack_id) {
                let row_meta_data = Arc::new(BatchRunTomoRowMetaData::new(&stack_id));
                row_meta_data.load(props, Some(&prepend));
                row_meta_data_map.put_ordinal(
                    row_meta_data.get_row_number(),
                    stack_id.clone(),
                    row_meta_data,
                );
            }
        }
    }

    /// Java `getRowMetaData(String)`.
    pub fn get_row_meta_data(&self, stack_id: &str) -> Arc<BatchRunTomoRowMetaData> {
        let mut row_meta_data_map = self.row_meta_data_map.lock().unwrap();
        if let Some(row_meta_data) = row_meta_data_map.get(stack_id) {
            return row_meta_data;
        }
        let row_meta_data = Arc::new(BatchRunTomoRowMetaData::new(stack_id));
        row_meta_data_map.put(stack_id.to_owned(), row_meta_data.clone());
        row_meta_data
    }

    /// Java `createPrepend(String)`.
    pub fn create_prepend(&self, prepend: Option<&str>) -> String {
        let Some(prepend) = prepend else {
            return batch_run_tomo_meta_data::GROUP_KEY.to_owned();
        };
        if java_lang_string_matches_whitespace(prepend) {
            return batch_run_tomo_meta_data::GROUP_KEY.to_owned();
        }
        let prepend = java_lang_string_trim(prepend);
        if prepend.ends_with('.') {
            return format!("{}{}", prepend, batch_run_tomo_meta_data::GROUP_KEY);
        }
        format!("{}.{}", prepend, batch_run_tomo_meta_data::GROUP_KEY)
    }
}

impl Storable for SeriesWatcherMetaData {
    /// Java `store(Properties)`: empty.
    fn store(&self, _properties: &mut BTreeMap<String, String>) {}

    /// Java `store(Properties, String)`: empty.
    fn store_with_prepend(&self, _properties: &mut BTreeMap<String, String>, _prepend: &str) {}

    /// Java `load(Properties)`.
    fn load(&self, properties: &mut BTreeMap<String, String>) {
        self.load_with_prepend(properties, "");
    }

    /// Java `load(Properties, String)`.
    fn load_with_prepend(&self, properties: &mut BTreeMap<String, String>, prepend: &str) {
        SeriesWatcherMetaData::load_with_prepend(self, properties, prepend);
    }
}
