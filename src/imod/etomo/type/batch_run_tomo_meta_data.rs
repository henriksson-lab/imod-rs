//! `IMOD/Etomo/src/etomo/type/BatchRunTomoMetaData.java`.
//!
//! The batchruntomo interface's data file (`.ebt`) contents.
//!
//! **Representation.**  `BatchRunTomoMetaData extends BaseMetaData implements
//! DialogCompleteClient`: the superclass state is `base` (`BaseMetaDataBase`) and the
//! abstract methods are the `BaseMetaData` trait.  The object is owned by
//! `BatchRunTomoManager`, written by the dialog on the event dispatch thread and stored
//! (possibly) from process threads, so its fields sit behind one lock and every method
//! takes `&self`.  The lock is never held while calling back into the manager.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use super::axis_type::AxisType;
use super::base_meta_data::{BaseMetaData, BaseMetaDataBase};
use super::batch_run_tomo_dataset_meta_data::BatchRunTomoDatasetMetaData;
use super::batch_run_tomo_row_meta_data::BatchRunTomoRowMetaData;
use super::batch_run_tomo_status::{self, BatchRunTomoStatus};
use super::const_etomo_number::{java_lang_string_matches_whitespace, java_lang_string_trim};
use super::data_file_type::DataFileType;
use super::ending_step::EndingStep;
use super::enumerated_type::EnumeratedType;
use super::etomo_boolean2::EtomoBoolean2;
use super::image_file_meta_data::ImageFileMetaData;
use super::image_filename_style::ImageFilenameStyle;
use super::ordered_hash_map::{OrderedHashMap, ReadOnlyArray};
use super::queue_type::QueueType;
use super::starting_step::StartingStep;
use super::status::Status;
use super::string_property::StringProperty;
use super::table_reference::TableReference;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::batch_run_tomo_manager::BatchRunTomoManager;
use crate::imod::etomo::storage::storable::Storable;
use crate::imod::etomo::ui::dialog_complete_client::DialogCompleteClient;
use crate::imod::etomo::ui::dialog_complete_listener::DialogCompleteListener;
use crate::imod::etomo::ui::log_properties::LogProperties;
use crate::imod::etomo::util::dataset_files;
use crate::imod::etomo::util::utilities;

/// Java package-private static final `GROUP_KEY`.
pub const GROUP_KEY: &str = "meta";
/// Java public static final `NEW_TITLE`.
pub const NEW_TITLE: &str = "Batch Run Tomo";
/// Java private static final `ENDING_STEP_KEY`.
const ENDING_STEP_KEY: &str = "EndingStep";
/// Java private static final `EARLIEST_RUN_KEY`.
const EARLIEST_RUN_KEY: &str = "EarliestRun";
/// Java private static final `STATUS_KEY`.
const STATUS_KEY: &str = "Status";
/// Java private static final `STARTING_STEP_KEY`.
const STARTING_STEP_KEY: &str = "StartingStep";
/// Java private static final `USE_GPU_MACHINE_LIST_PARALLEL_KEY`.
const USE_GPU_MACHINE_LIST_PARALLEL_KEY: &str = "Use.GPU.MachineList.Parallel";

/// The mutable fields of Java `BatchRunTomoMetaData`.
struct Fields {
    /// Java private `rootName`.
    root_name: StringProperty,
    /// Java private `deliverToDirectory`.
    deliver_to_directory: StringProperty,
    /// Java private `inputDirectiveFile`.
    input_directive_file: StringProperty,
    /// Java private `enableStartingStep`.
    enable_starting_step: EtomoBoolean2,
    /// Java private `useEndingStep`.
    use_ending_step: EtomoBoolean2,
    /// Java private `useStartingStep`.
    use_starting_step: EtomoBoolean2,
    /// Java private `browsingDirectory`.
    browsing_directory: StringProperty,
    /// Java private `delivered`.
    delivered: EtomoBoolean2,
    /// Java private `maxGPUsForOneJob`.
    max_gpus_for_one_job: StringProperty,
    /// Java private `splitBatch`.
    split_batch: EtomoBoolean2,
    /// Java private `numberOfJobsToMake`.
    number_of_jobs_to_make: StringProperty,
    /// Java private `queueNumberOfJobsToMake`.
    queue_number_of_jobs_to_make: StringProperty,
    /// Java private `useSecondaryQueue`.
    use_secondary_queue: EtomoBoolean2,
    /// Java private `queueType`.
    queue_type: StringProperty,
    /// Java private `useCPUMachineList`.
    use_cpu_machine_list: EtomoBoolean2,
    /// Java private `useSeriesWatcher`.
    use_series_watcher: EtomoBoolean2,
    /// Java private `watchDirectory`.
    watch_directory: StringProperty,
    /// Java private `dualAxis`.
    dual_axis: EtomoBoolean2,
    /// Java private `twoSurfaces`.
    two_surfaces: EtomoBoolean2,
    /// Java private `mpoeRootName`.
    mpoe_root_name: StringProperty,
    /// Java private `mpoeExt`.
    mpoe_ext: StringProperty,
    /// Java private `mpoeIncludeCombine`.
    mpoe_include_combine: EtomoBoolean2,
    /// Java private `mpoeAOnly`.
    mpoe_a_only: EtomoBoolean2,
    /// Java private `mpoeSeparateB`.
    mpoe_separate_b: EtomoBoolean2,
    /// Java private `minimumTiltRange`.
    minimum_tilt_range: StringProperty,
    /// Java private `minimumNumberOfViews`.
    minimum_number_of_views: StringProperty,
    /// Java private `minimumAgeOfStacks`.
    minimum_age_of_stacks: StringProperty,
    /// Java private `rowMetaDataMap` (key is stackID).
    row_meta_data_map: OrderedHashMap<String, Arc<BatchRunTomoRowMetaData>>,
    /// Java private `earliestRunEndingStep`, initially null.
    earliest_run_ending_step: Option<EndingStep>,
    /// Java private `status = BatchRunTomoStatus.DEFAULT`.
    status: Option<BatchRunTomoStatus>,
    /// Java private `endingStep`, initially null.
    ending_step: Option<EndingStep>,
    /// Java private `startingStep`, initially null.
    starting_step: Option<StartingStep>,
    /// Java private `useGPUMachineListParallel`, initially null.
    use_gpu_machine_list_parallel: Option<EtomoBoolean2>,
}

/// Java `public final class BatchRunTomoMetaData extends BaseMetaData implements
/// DialogCompleteClient`.
pub struct BatchRunTomoMetaData {
    /// Java superclass `BaseMetaData` state.
    base: BaseMetaDataBase,
    /// Java private final `datasetMetaData` (metadata for the global dataset dialog).
    dataset_meta_data: Arc<BatchRunTomoDatasetMetaData>,
    /// Java private final `tableReference`.
    table_reference: Arc<TableReference>,
    /// Java private final `manager`.
    manager: &'static BatchRunTomoManager,
    fields: Mutex<Fields>,
}

// SAFETY: `base` holds `&'static dyn` references to the manager and to the log
// properties, as every `BaseMetaData` implementor here does; they are read only on
// the threads the Java reads them on.
unsafe impl Send for BatchRunTomoMetaData {}
unsafe impl Sync for BatchRunTomoMetaData {}

impl BatchRunTomoMetaData {
    /// Java `BatchRunTomoMetaData(BatchRunTomoManager, LogProperties, TableReference,
    /// boolean)`.
    pub fn new(
        manager: &'static BatchRunTomoManager,
        log_properties: Option<&'static dyn LogProperties>,
        table_reference: Arc<TableReference>,
        new_dataset: bool,
    ) -> BatchRunTomoMetaData {
        // super(manager, logProperties, false, newDataset, false)
        let base = BaseMetaDataBase::new_force_old_style(
            Some(manager as &'static dyn BaseManager),
            log_properties,
            false,
            new_dataset,
            false,
        );
        *base.axis_type.lock().unwrap() = AxisType::SingleAxis;
        *base.file_extension.lock().unwrap() =
            DataFileType::BatchRunTomo.extension().map(str::to_owned);
        BatchRunTomoMetaData {
            base,
            dataset_meta_data: Arc::new(BatchRunTomoDatasetMetaData::new()),
            table_reference,
            manager,
            fields: Mutex::new(Fields {
                root_name: StringProperty::new_with_key(Some("RootName")),
                deliver_to_directory: StringProperty::new_with_key(Some("DeliverToDirectory")),
                input_directive_file: StringProperty::new_with_key(Some("InputDirectiveFile")),
                enable_starting_step: EtomoBoolean2::new_with_name("EnableStartingStep"),
                use_ending_step: EtomoBoolean2::new_with_name(&format!("{}.Use", ENDING_STEP_KEY)),
                use_starting_step: EtomoBoolean2::new_with_name(&format!(
                    "{}.Use",
                    STARTING_STEP_KEY
                )),
                browsing_directory: StringProperty::new_with_key(Some("BrowsingDirectory")),
                delivered: EtomoBoolean2::new_with_name("Delivered"),
                max_gpus_for_one_job: StringProperty::new_with_key(Some("MaxGPUsForOneJob")),
                split_batch: EtomoBoolean2::new_with_name("SplitBatch"),
                number_of_jobs_to_make: StringProperty::new_with_key(Some("NumberOfJobsToMake")),
                queue_number_of_jobs_to_make: StringProperty::new_with_key(Some(
                    "NumberOfJobsToMake.Queue",
                )),
                use_secondary_queue: EtomoBoolean2::new_with_name("Use.SecondaryQueue"),
                queue_type: StringProperty::new_with_key(Some("Queue.Type")),
                use_cpu_machine_list: EtomoBoolean2::new_with_name("Use.CPU.MachineList"),
                use_series_watcher: EtomoBoolean2::new_with_name("SeriesWatcher.Use"),
                watch_directory: StringProperty::new_with_key(Some("SeriesWatcher.WatchDirectory")),
                dual_axis: EtomoBoolean2::new_with_name("SeriesWatcher.DualAxis"),
                two_surfaces: EtomoBoolean2::new_with_name("SeriesWatcher.TwoSurfaces"),
                mpoe_root_name: StringProperty::new_with_key(Some(
                    "SeriesWatcher.MatchPatternOrExt.RootName",
                )),
                mpoe_ext: StringProperty::new_with_key(Some("SeriesWatcher.MatchPatternOrExt.Ext")),
                mpoe_include_combine: EtomoBoolean2::new_with_name(
                    "SeriesWatcher.MatchPatternOrExt.IncludeCombine",
                ),
                mpoe_a_only: EtomoBoolean2::new_with_name("SeriesWatcher.MatchPatternOrExt.AOnly"),
                mpoe_separate_b: EtomoBoolean2::new_with_name(
                    "SeriesWatcher.MatchPatternOrExt.SeparateB",
                ),
                minimum_tilt_range: StringProperty::new_with_key(Some(
                    "SeriesWatcher.MinimumTiltRange",
                )),
                minimum_number_of_views: StringProperty::new_with_key(Some(
                    "SeriesWatcher.MinimumNumberOfViews",
                )),
                minimum_age_of_stacks: StringProperty::new_with_key(Some(
                    "SeriesWatcher.MinimumAgeOfStacks",
                )),
                row_meta_data_map: OrderedHashMap::new(),
                earliest_run_ending_step: None,
                status: Some(batch_run_tomo_status::DEFAULT),
                ending_step: None,
                starting_step: None,
                use_gpu_machine_list_parallel: None,
            }),
        }
    }

    /// Java `setName(String)`.
    pub fn set_name(&self, root_name: Option<&str>) {
        self.fields.lock().unwrap().root_name.set(root_name);
    }

    /// Java `isRootNameNull()`.
    pub fn is_root_name_null(&self) -> bool {
        use super::const_string_property::ConstStringProperty;
        self.fields.lock().unwrap().root_name.is_empty()
    }

    /// Java `getRootName()`.
    pub fn get_root_name(&self) -> String {
        self.fields.lock().unwrap().root_name.to_string()
    }

    /// Java `setRootName(String)`.
    pub fn set_root_name(&self, input: Option<&str>) {
        self.fields.lock().unwrap().root_name.set(input);
    }

    /// Java static `getNewFileTitle()`.
    pub fn get_new_file_title() -> &'static str {
        NEW_TITLE
    }

    /// Java `validate()`.  Returns null if valid, the error message if invalid.
    pub fn validate(&self) -> Option<String> {
        use super::const_string_property::ConstStringProperty;
        if self.fields.lock().unwrap().root_name.is_empty() {
            return Some("Missing root name.".to_owned());
        }
        None
    }

    /// Java `load(Properties, String)`.
    pub fn load_with_prepend(&self, props: &mut BTreeMap<String, String>, parent_prepend: &str) {
        // super.load(props, parentPrepend)
        let created = self.create_prepend(parent_prepend);
        let _ = self
            .base
            .load_with_created_prepend(props, created.as_deref());
        {
            let mut f = self.fields.lock().unwrap();
            // reset
            f.root_name.reset();
            f.deliver_to_directory.reset();
            f.input_directive_file.reset();
            f.earliest_run_ending_step = None;
            f.status = None;
            f.enable_starting_step.reset();
            f.use_ending_step.reset();
            f.row_meta_data_map.clear();
            f.ending_step = None;
            f.starting_step = None;
            f.use_starting_step.reset();
            f.browsing_directory.reset();
            f.delivered.reset();
            f.max_gpus_for_one_job.reset();
            f.split_batch.reset();
            f.number_of_jobs_to_make.reset();
            f.queue_number_of_jobs_to_make.reset();
            f.use_secondary_queue.reset();
            f.use_gpu_machine_list_parallel = None;
            f.queue_type.reset();
            f.use_cpu_machine_list.reset();
            f.use_series_watcher.reset();
            f.watch_directory.reset();
            f.dual_axis.reset();
            f.two_surfaces.reset();
            f.mpoe_root_name.reset();
            f.mpoe_ext.reset();
            f.mpoe_include_combine.reset();
            f.mpoe_a_only.reset();
            f.mpoe_separate_b.reset();
            f.minimum_tilt_range.reset();
            f.minimum_number_of_views.reset();
            f.minimum_number_of_views.reset();
            // load
            let prepend = self
                .create_prepend(parent_prepend)
                .unwrap_or_else(|| "null".to_owned());
            let group = format!("{}.", prepend);
            let p = Some(prepend.as_str());
            // `StringProperty.load` may remove a backward-compatible key from `props`;
            // none of these declares one.
            f.root_name.load_with_prepend(Some(&mut *props), p);
            f.deliver_to_directory
                .load_with_prepend(Some(&mut *props), p);
            f.input_directive_file
                .load_with_prepend(Some(&mut *props), p);
            self.dataset_meta_data.load(props, p);
            self.table_reference.load(props, p);
            f.earliest_run_ending_step = EndingStep::get_instance_from_step_value(
                props
                    .get(&format!(
                        "{}{}.{}",
                        group, ENDING_STEP_KEY, EARLIEST_RUN_KEY
                    ))
                    .map(String::as_str),
            );
            let iterator = self.table_reference.id_iterator();
            f.status = Some(BatchRunTomoStatus::get_instance(
                props
                    .get(&format!("{}{}", group, STATUS_KEY))
                    .map(String::as_str),
            ));
            f.enable_starting_step.load_with_prepend(props, p);
            f.use_ending_step.load_with_prepend(props, p);
            f.ending_step = EndingStep::get_instance_from_step_value(
                props
                    .get(&format!("{}{}", group, ENDING_STEP_KEY))
                    .map(String::as_str),
            );
            f.starting_step = StartingStep::get_instance_from_step_value(
                props
                    .get(&format!("{}{}", group, STARTING_STEP_KEY))
                    .map(String::as_str),
            );
            f.use_starting_step.load_with_prepend(props, p);
            f.browsing_directory.load_with_prepend(Some(&mut *props), p);
            f.delivered.load_with_prepend(props, p);
            f.max_gpus_for_one_job
                .load_with_prepend(Some(&mut *props), p);
            f.split_batch.load_with_prepend(props, p);
            f.number_of_jobs_to_make
                .load_with_prepend(Some(&mut *props), p);
            f.queue_number_of_jobs_to_make
                .load_with_prepend(Some(&mut *props), p);
            f.use_secondary_queue.load_with_prepend(props, p);
            f.use_gpu_machine_list_parallel = EtomoBoolean2::load_instance(
                f.use_gpu_machine_list_parallel.take(),
                USE_GPU_MACHINE_LIST_PARALLEL_KEY,
                props,
                p,
            );
            f.queue_type.load_with_prepend(Some(&mut *props), p);
            f.use_cpu_machine_list.load_with_prepend(props, p);
            f.use_series_watcher.load_with_prepend(props, p);
            f.watch_directory.load_with_prepend(Some(&mut *props), p);
            f.dual_axis.load_with_prepend(props, p);
            f.two_surfaces.load_with_prepend(props, p);
            f.mpoe_root_name.load_with_prepend(Some(&mut *props), p);
            f.mpoe_ext.load_with_prepend(Some(&mut *props), p);
            f.mpoe_include_combine.load_with_prepend(props, p);
            f.mpoe_a_only.load_with_prepend(props, p);
            f.mpoe_separate_b.load_with_prepend(props, p);
            f.minimum_tilt_range.load_with_prepend(Some(&mut *props), p);
            f.minimum_number_of_views
                .load_with_prepend(Some(&mut *props), p);
            f.minimum_age_of_stacks
                .load_with_prepend(Some(&mut *props), p);

            for stack_id in iterator {
                if !BatchRunTomoRowMetaData::is_row_number_null_in(props, p, &stack_id) {
                    let row_meta_data = Arc::new(BatchRunTomoRowMetaData::new(&stack_id));
                    row_meta_data.load(props, p);
                    f.row_meta_data_map.put_ordinal(
                        row_meta_data.get_row_number(),
                        stack_id.clone(),
                        row_meta_data,
                    );
                }
            }
        }
        // Bug# 2403 - complex image filename style correct must be done in child class.
        let prepend = self
            .create_prepend(parent_prepend)
            .unwrap_or_else(|| "null".to_owned());
        self.check_image_filename_style_loaded(&prepend);
    }

    /// Java `repairImageFilenameStyleFromParam(String)`.  If possible uses the
    /// batchruntomo comfile to repair imageFilenameStyle [Bug# 2386].  Repair for all brt
    /// datasets created before the Bug# 2388 fix is needed because BaseMetaData was
    /// never saved.  Before the Bug# 2388 fix, once this dataset was reloaded, the
    /// incorrect image filename style made this dataset incompatible from datasets it
    /// created before it was first reloaded.
    ///
    /// After it was loaded without a saved imageFilenameStyle it always sets the naming
    /// style in the brt comfile to OLD.  So if the naming style is set to OLD, or not
    /// set at all, then it can be repaired via the brt comfile.  Returns true if
    /// filename style didn't need repair or was repaired.
    pub fn repair_image_filename_style_from_param(&self, prepend: &str) -> bool {
        if self.base.was_image_filename_style_loaded() {
            return true;
        }
        // An incorrect image filename style makes this dataset incompatible with the
        // datasets it created in the past.
        let naming_style = self.manager.get_batchruntomo_comfile_naming_style();
        if let Some(naming_style) = &naming_style
            && !naming_style.is_valid()
        {
            return false;
        }
        let image_filename_style;
        let mut repair_type: Option<&str> = None;
        match &naming_style {
            None => {
                // If namingStyle in the brt comfile is missing, then this dataset was
                // created before image filename standardization and never reloaded until
                // now. Its correct image filename style is "OLD".
                image_filename_style = Some(ImageFilenameStyle::Old);
                repair_type = Some("old image file name style\n");
            }
            Some(naming_style) if naming_style.is_null() => {
                image_filename_style = Some(ImageFilenameStyle::Old);
                repair_type = Some("old image file name style\n");
            }
            Some(naming_style) => {
                image_filename_style =
                    ImageFilenameStyle::get_instance_from_naming_style(Some(naming_style));
                if image_filename_style != Some(ImageFilenameStyle::Old) {
                    // If namingStyle in the brt comfile is not 0 (old style), then this
                    // dataset was created after image filename standardization and never
                    // reloaded until now. Its correct image filename style is the
                    // whatever was used when it was originally created.
                    repair_type = Some("image file name style from the batchruntomo\ncomfile ");
                }
            }
        }
        let Some(repair_type) = repair_type else {
            return false;
        };
        let key = self
            .get_image_filename_style_key(prepend)
            .unwrap_or_else(|| "null".to_owned());
        eprintln!(
            "\nINFO: Attempting to repair the {} property (which is missing from the\ndataset file) based on the batchruntomo comfile.  Using the {}to set the {} to {}.  [Bug# 2386]\n",
            key,
            repair_type,
            key,
            image_filename_style.map_or("null".to_owned(), |style| style.to_string())
        );
        self.correct_image_filename_style(prepend, image_filename_style)
    }

    /// Java `store(Properties, String)`.
    pub fn store_with_prepend(&self, props: &mut BTreeMap<String, String>, prepend: &str) {
        // super.store(props, prepend)
        let created = self.create_prepend(prepend);
        self.base
            .store_with_created_prepend(props, created.as_deref());
        let prepend = self
            .create_prepend(prepend)
            .unwrap_or_else(|| "null".to_owned());
        let group = format!("{}.", prepend);
        let p = Some(prepend.as_str());
        let f = self.fields.lock().unwrap();
        f.root_name.store_with_prepend(Some(props), p);
        f.deliver_to_directory.store_with_prepend(Some(props), p);
        f.input_directive_file.store_with_prepend(Some(props), p);
        self.dataset_meta_data.store(props, p);
        self.table_reference.store(props, p);
        match f.earliest_run_ending_step {
            Some(earliest_run_ending_step) => {
                props.insert(
                    format!("{}{}.{}", group, ENDING_STEP_KEY, EARLIEST_RUN_KEY),
                    earliest_run_ending_step.get_value().to_string(),
                );
            }
            None => {
                props.remove(&format!(
                    "{}{}.{}",
                    group, ENDING_STEP_KEY, EARLIEST_RUN_KEY
                ));
            }
        }
        match f.status {
            Some(status) => {
                props.insert(
                    format!("{}{}", group, STATUS_KEY),
                    status.get_text().unwrap_or("null").to_owned(),
                );
            }
            None => {
                props.remove(&format!("{}{}", group, STATUS_KEY));
            }
        }
        f.enable_starting_step.store_with_prepend(props, p);
        f.use_ending_step.store_with_prepend(props, p);
        match f.ending_step {
            Some(ending_step) => {
                props.insert(
                    format!("{}{}", group, ENDING_STEP_KEY),
                    ending_step.get_value().to_string(),
                );
            }
            None => {
                props.remove(&format!("{}{}", group, ENDING_STEP_KEY));
            }
        }
        match f.starting_step {
            Some(starting_step) => {
                props.insert(
                    format!("{}{}", group, STARTING_STEP_KEY),
                    starting_step.get_value().to_string(),
                );
            }
            None => {
                props.remove(&format!("{}{}", group, STARTING_STEP_KEY));
            }
        }
        f.use_starting_step.store_with_prepend(props, p);
        f.browsing_directory.store_with_prepend(Some(props), p);
        f.delivered.store_with_prepend(props, p);
        f.max_gpus_for_one_job.store_with_prepend(Some(props), p);
        f.split_batch.store_with_prepend(props, p);
        f.number_of_jobs_to_make.store_with_prepend(Some(props), p);
        f.queue_number_of_jobs_to_make
            .store_with_prepend(Some(props), p);
        f.use_secondary_queue.store_with_prepend(props, p);
        EtomoBoolean2::store_instance(
            f.use_gpu_machine_list_parallel.as_ref(),
            props,
            p,
            USE_GPU_MACHINE_LIST_PARALLEL_KEY,
        );
        f.queue_type.store_with_prepend(Some(props), p);
        f.use_cpu_machine_list.store_with_prepend(props, p);
        f.use_series_watcher.store_with_prepend(props, p);
        f.watch_directory.store_with_prepend(Some(props), p);
        f.dual_axis.store_with_prepend(props, p);
        f.two_surfaces.store_with_prepend(props, p);
        f.mpoe_root_name.store_with_prepend(Some(props), p);
        f.mpoe_ext.store_with_prepend(Some(props), p);
        f.mpoe_include_combine.store_with_prepend(props, p);
        f.mpoe_a_only.store_with_prepend(props, p);
        f.mpoe_separate_b.store_with_prepend(props, p);
        f.minimum_tilt_range.store_with_prepend(Some(props), p);
        f.minimum_number_of_views.store_with_prepend(Some(props), p);
        f.minimum_age_of_stacks.store_with_prepend(Some(props), p);

        for row_meta_data in f.row_meta_data_map.values() {
            row_meta_data.store(props, p);
        }
    }

    /// Java `setUseStartingStep(boolean)`.
    pub fn set_use_starting_step(&self, input: bool) {
        self.fields
            .lock()
            .unwrap()
            .use_starting_step
            .set_boolean(input);
    }

    /// Java `setUseEndingStep(boolean)`.
    pub fn set_use_ending_step(&self, input: bool) {
        self.fields
            .lock()
            .unwrap()
            .use_ending_step
            .set_boolean(input);
    }

    /// Java `getBrowsingDirectory()`.
    pub fn get_browsing_directory(&self) -> String {
        self.fields.lock().unwrap().browsing_directory.to_string()
    }

    /// Java `getMaxGPUsForOneJob()`.
    pub fn get_max_gpus_for_one_job(&self) -> String {
        self.fields.lock().unwrap().max_gpus_for_one_job.to_string()
    }

    /// Java `isMaxGPUsForOneJobSet()`.
    pub fn is_max_gpus_for_one_job_set(&self) -> bool {
        use super::const_string_property::ConstStringProperty;
        !self.fields.lock().unwrap().max_gpus_for_one_job.is_empty()
    }

    /// Java `isSplitBatch()`.
    pub fn is_split_batch(&self) -> bool {
        self.fields.lock().unwrap().split_batch.is()
    }

    /// Java `isSplitBatchSet()`.
    pub fn is_split_batch_set(&self) -> bool {
        !self.fields.lock().unwrap().split_batch.is_null()
    }

    /// Java `getNumberOfJobsToMake()`.
    pub fn get_number_of_jobs_to_make(&self) -> String {
        self.fields
            .lock()
            .unwrap()
            .number_of_jobs_to_make
            .to_string()
    }

    /// Java `getQueueNumberOfJobsToMake()`.
    pub fn get_queue_number_of_jobs_to_make(&self) -> String {
        self.fields
            .lock()
            .unwrap()
            .queue_number_of_jobs_to_make
            .to_string()
    }

    /// Java `isUseSecondaryQueue()`.
    pub fn is_use_secondary_queue(&self) -> bool {
        self.fields.lock().unwrap().use_secondary_queue.is()
    }

    /// Java `setUseSecondaryQueue(boolean)`.
    pub fn set_use_secondary_queue(&self, input: bool) {
        self.fields
            .lock()
            .unwrap()
            .use_secondary_queue
            .set_boolean(input);
    }

    /// Java `resetUseSecondaryQueue()`.
    pub fn reset_use_secondary_queue(&self) {
        self.fields.lock().unwrap().use_secondary_queue.reset();
    }

    /// Java `getUseGPUMachineListParallel()`.
    pub fn get_use_gpu_machine_list_parallel(&self) -> Option<EtomoBoolean2> {
        self.fields
            .lock()
            .unwrap()
            .use_gpu_machine_list_parallel
            .clone()
    }

    /// Java `setUseGPUMachineListParallelNull()`.
    pub fn set_use_gpu_machine_list_parallel_null(&self) {
        self.fields.lock().unwrap().use_gpu_machine_list_parallel = None;
    }

    /// Java `setUseGPUMachineListParallel(boolean)`.
    pub fn set_use_gpu_machine_list_parallel(&self, input: bool) {
        let mut f = self.fields.lock().unwrap();
        if f.use_gpu_machine_list_parallel.is_none() {
            f.use_gpu_machine_list_parallel = Some(EtomoBoolean2::new_with_name(
                USE_GPU_MACHINE_LIST_PARALLEL_KEY,
            ));
        }
        f.use_gpu_machine_list_parallel
            .as_mut()
            .unwrap()
            .set_boolean(input);
    }

    /// Java `getQueueType()`.
    pub fn get_queue_type(&self) -> Option<QueueType> {
        QueueType::get_instance(Some(&self.fields.lock().unwrap().queue_type.to_string()))
    }

    /// Java `setQueueType(QueueType)`.
    pub fn set_queue_type(&self, input: Option<QueueType>) {
        let mut f = self.fields.lock().unwrap();
        match input {
            Some(input) => f.queue_type.set(Some(input.get_label())),
            None => f.queue_type.reset(),
        }
    }

    /// Java `isUseCPUMachineList()`.
    pub fn is_use_cpu_machine_list(&self) -> bool {
        self.fields.lock().unwrap().use_cpu_machine_list.is()
    }

    /// Java `isUseSeriesWatcher()`.
    pub fn is_use_series_watcher(&self) -> bool {
        self.fields.lock().unwrap().use_series_watcher.is()
    }

    /// Java `getWatchDirectory()`.
    pub fn get_watch_directory(&self) -> String {
        self.fields.lock().unwrap().watch_directory.to_string()
    }

    /// Java `isDualAxisSet()`.  Returns true if EtomoBoolean2.currentValue was set.  Ignores
    /// default and display values.
    pub fn is_dual_axis_set(&self) -> bool {
        self.fields.lock().unwrap().dual_axis.is_set()
    }

    /// Java `isDualAxis()`.
    pub fn is_dual_axis(&self) -> bool {
        self.fields.lock().unwrap().dual_axis.is()
    }

    /// Java `isTwoSurfacesSet()`.  Returns true if EtomoBoolean2.currentValue was set.  Ignores
    /// default and display values.
    pub fn is_two_surfaces_set(&self) -> bool {
        self.fields.lock().unwrap().two_surfaces.is_set()
    }

    /// Java `isTwoSurfaces()`.
    pub fn is_two_surfaces(&self) -> bool {
        self.fields.lock().unwrap().two_surfaces.is()
    }

    /// Java `isMpoeIncludeCombine()`.
    pub fn is_mpoe_include_combine(&self) -> bool {
        self.fields.lock().unwrap().mpoe_include_combine.is()
    }

    /// Java `isMpoeAOnly()`.
    pub fn is_mpoe_a_only(&self) -> bool {
        self.fields.lock().unwrap().mpoe_a_only.is()
    }

    /// Java `isMpoeSeparateB()`.
    pub fn is_mpoe_separate_b(&self) -> bool {
        self.fields.lock().unwrap().mpoe_separate_b.is()
    }

    /// Java `getMpoeRootName()`.
    pub fn get_mpoe_root_name(&self) -> String {
        self.fields.lock().unwrap().mpoe_root_name.to_string()
    }

    /// Java `getMpoeExt()`.
    pub fn get_mpoe_ext(&self) -> String {
        self.fields.lock().unwrap().mpoe_ext.to_string()
    }

    /// Java `isMinimumTiltRangeSet()`.
    pub fn is_minimum_tilt_range_set(&self) -> bool {
        use super::const_string_property::ConstStringProperty;
        !self.fields.lock().unwrap().minimum_tilt_range.is_empty()
    }

    /// Java `getMinimumTiltRange()`.
    pub fn get_minimum_tilt_range(&self) -> String {
        self.fields.lock().unwrap().minimum_tilt_range.to_string()
    }

    /// Java `isMinimumNumberOfViewsSet()`.
    pub fn is_minimum_number_of_views_set(&self) -> bool {
        use super::const_string_property::ConstStringProperty;
        !self
            .fields
            .lock()
            .unwrap()
            .minimum_number_of_views
            .is_empty()
    }

    /// Java `getMinimumNumberOfViews()`.
    pub fn get_minimum_number_of_views(&self) -> String {
        self.fields
            .lock()
            .unwrap()
            .minimum_number_of_views
            .to_string()
    }

    /// Java `isMinimumAgeOfStacksSet()`.
    pub fn is_minimum_age_of_stacks_set(&self) -> bool {
        use super::const_string_property::ConstStringProperty;
        !self.fields.lock().unwrap().minimum_age_of_stacks.is_empty()
    }

    /// Java `getMinimumAgeOfStacks()`.
    pub fn get_minimum_age_of_stacks(&self) -> String {
        self.fields
            .lock()
            .unwrap()
            .minimum_age_of_stacks
            .to_string()
    }

    /// Java `setUseCPUMachineList(boolean)`.
    pub fn set_use_cpu_machine_list(&self, input: bool) {
        self.fields
            .lock()
            .unwrap()
            .use_cpu_machine_list
            .set_boolean(input);
    }

    /// Java `setUseSeriesWatcher(boolean)`.
    pub fn set_use_series_watcher(&self, input: bool) {
        self.fields
            .lock()
            .unwrap()
            .use_series_watcher
            .set_boolean(input);
    }

    /// Java `setWatchDirectory(File)`.
    pub fn set_watch_directory(&self, input: Option<&Path>) {
        let mut f = self.fields.lock().unwrap();
        match input {
            None => f.watch_directory.reset(),
            Some(input) => f
                .watch_directory
                .set(Some(&utilities::java_io_file_get_absolute_path(
                    &input.to_string_lossy(),
                ))),
        }
    }

    /// Java `setDualAxis(boolean)`.
    pub fn set_dual_axis(&self, input: bool) {
        self.fields.lock().unwrap().dual_axis.set_boolean(input);
    }

    /// Java `setTwoSurfaces(boolean)`.
    pub fn set_two_surfaces(&self, input: bool) {
        self.fields.lock().unwrap().two_surfaces.set_boolean(input);
    }

    /// Java `setMpoeIncludeCombine(boolean)`.
    pub fn set_mpoe_include_combine(&self, input: bool) {
        self.fields
            .lock()
            .unwrap()
            .mpoe_include_combine
            .set_boolean(input);
    }

    /// Java `setMpoeAOnly(boolean)`.
    pub fn set_mpoe_a_only(&self, input: bool) {
        self.fields.lock().unwrap().mpoe_a_only.set_boolean(input);
    }

    /// Java `setMpoeSeparateB(boolean)`.
    pub fn set_mpoe_separate_b(&self, input: bool) {
        self.fields
            .lock()
            .unwrap()
            .mpoe_separate_b
            .set_boolean(input);
    }

    /// Java `setMpoeRootName(String)`.
    pub fn set_mpoe_root_name(&self, input: Option<&str>) {
        self.fields.lock().unwrap().mpoe_root_name.set(input);
    }

    /// Java `setMpoeExt(String)`.
    pub fn set_mpoe_ext(&self, input: Option<&str>) {
        self.fields.lock().unwrap().mpoe_ext.set(input);
    }

    /// Java `setMinimumTiltRange(String)`.
    pub fn set_minimum_tilt_range(&self, input: Option<&str>) {
        self.fields.lock().unwrap().minimum_tilt_range.set(input);
    }

    /// Java `setMinimumNumberOfViews(String)`.
    pub fn set_minimum_number_of_views(&self, input: Option<&str>) {
        self.fields
            .lock()
            .unwrap()
            .minimum_number_of_views
            .set(input);
    }

    /// Java `setMinimumAgeOfStacks(String)`.
    pub fn set_minimum_age_of_stacks(&self, input: Option<&str>) {
        self.fields.lock().unwrap().minimum_age_of_stacks.set(input);
    }

    /// Java `isNumberOfJobsToMakeSet()`.
    pub fn is_number_of_jobs_to_make_set(&self) -> bool {
        use super::const_string_property::ConstStringProperty;
        !self
            .fields
            .lock()
            .unwrap()
            .number_of_jobs_to_make
            .is_empty()
    }

    /// Java `isQueueNumberOfJobsToMakeSet()`.
    pub fn is_queue_number_of_jobs_to_make_set(&self) -> bool {
        use super::const_string_property::ConstStringProperty;
        !self
            .fields
            .lock()
            .unwrap()
            .queue_number_of_jobs_to_make
            .is_empty()
    }

    /// Java `isDelivered()`.
    pub fn is_delivered(&self) -> bool {
        self.fields.lock().unwrap().delivered.is()
    }

    /// Java `setDelivered(boolean)`.
    pub fn set_delivered(&self, input: bool) {
        self.fields.lock().unwrap().delivered.set_boolean(input);
    }

    /// Java `setMaxGPUsForOneJob(String)`.
    pub fn set_max_gpus_for_one_job(&self, input: Option<&str>) {
        self.fields.lock().unwrap().max_gpus_for_one_job.set(input);
    }

    /// Java `setNumberOfJobsToMake(String)`.
    pub fn set_number_of_jobs_to_make(&self, input: Option<&str>) {
        self.fields
            .lock()
            .unwrap()
            .number_of_jobs_to_make
            .set(input);
    }

    /// Java `setQueueNumberOfJobsToMake(String)`.
    pub fn set_queue_number_of_jobs_to_make(&self, input: Option<&str>) {
        self.fields
            .lock()
            .unwrap()
            .queue_number_of_jobs_to_make
            .set(input);
    }

    /// Java `setSplitBatch(boolean)`.
    pub fn set_split_batch(&self, input: bool) {
        self.fields.lock().unwrap().split_batch.set_boolean(input);
    }

    /// Java `setBrowsingDirectory(File)`.
    pub fn set_browsing_directory(&self, input: Option<&Path>) {
        let mut f = self.fields.lock().unwrap();
        match input {
            None => f.browsing_directory.reset(),
            Some(input) => {
                f.browsing_directory
                    .set(Some(&utilities::java_io_file_get_absolute_path(
                        &input.to_string_lossy(),
                    )))
            }
        }
    }

    /// Java `setStartingStep(StartingStep)`.
    pub fn set_starting_step(&self, input: Option<StartingStep>) {
        self.fields.lock().unwrap().starting_step = input;
    }

    /// Java `setEndingStep(EndingStep)`.
    pub fn set_ending_step(&self, input: Option<EndingStep>) {
        self.fields.lock().unwrap().ending_step = input;
    }

    /// Java `isUseStartingStep()`.
    pub fn is_use_starting_step(&self) -> bool {
        self.fields.lock().unwrap().use_starting_step.is()
    }

    /// Java `isUseEndingStep()`.
    pub fn is_use_ending_step(&self) -> bool {
        self.fields.lock().unwrap().use_ending_step.is()
    }

    /// Java `getStartingStep()`.
    pub fn get_starting_step(&self) -> Option<StartingStep> {
        self.fields.lock().unwrap().starting_step
    }

    /// Java `getEndingStep()`.
    pub fn get_ending_step(&self) -> Option<EndingStep> {
        self.fields.lock().unwrap().ending_step
    }

    /// Java `setEnableStartingStep(boolean)`.
    pub fn set_enable_starting_step(&self, input: bool) {
        self.fields
            .lock()
            .unwrap()
            .enable_starting_step
            .set_boolean(input);
    }

    /// Java `isEnableStartingStep()`.
    pub fn is_enable_starting_step(&self) -> bool {
        self.fields.lock().unwrap().enable_starting_step.is()
    }

    /// Java `setStatus(BatchRunTomoStatus)`.
    pub fn set_status(&self, input: Option<BatchRunTomoStatus>) {
        self.fields.lock().unwrap().status = input;
    }

    /// Java `setEarliestRunEndingStep(EndingStep)`.
    pub fn set_earliest_run_ending_step(&self, input: Option<EndingStep>) {
        self.fields.lock().unwrap().earliest_run_ending_step = input;
    }

    /// Java `getStatus()`.
    pub fn get_status(&self) -> Option<BatchRunTomoStatus> {
        self.fields.lock().unwrap().status
    }

    /// Java `getEarliestRunEndingStep()`.
    pub fn get_earliest_run_ending_step(&self) -> Option<EndingStep> {
        self.fields.lock().unwrap().earliest_run_ending_step
    }

    /// Java `getOrderedRows()`: the rows in ordinal order (null where an ordinal was not
    /// used), or null if no row was loaded with an ordinal.
    pub fn get_ordered_rows(&self) -> Option<Vec<Option<Arc<BatchRunTomoRowMetaData>>>> {
        let f = self.fields.lock().unwrap();
        let array = f.row_meta_data_map.ordered_values()?;
        Some((0..array.size()).map(|index| array.get(index)).collect())
    }

    /// Java `getRowMetaData(String)`.  Gets the rowMetaData for this stackID.  If it
    /// doesn't exist, create it and add it to the map.  Do not add the ordinal at this
    /// point.  The order of the rows only matters when they are being loaded into the
    /// table.
    pub fn get_row_meta_data(&self, stack_id: &str) -> Arc<BatchRunTomoRowMetaData> {
        let mut f = self.fields.lock().unwrap();
        if let Some(row_meta_data) = f.row_meta_data_map.get(stack_id) {
            return row_meta_data;
        }
        let row_meta_data = Arc::new(BatchRunTomoRowMetaData::new(stack_id));
        f.row_meta_data_map
            .put(stack_id.to_owned(), row_meta_data.clone());
        row_meta_data
    }

    /// Java `getDatasetMetaData()`.
    pub fn get_dataset_meta_data(&self) -> Arc<BatchRunTomoDatasetMetaData> {
        self.dataset_meta_data.clone()
    }

    /// Java `isRowNumberNull(String)`.
    pub fn is_row_number_null(&self, stack_id: &str) -> bool {
        let row_meta_data = self.fields.lock().unwrap().row_meta_data_map.get(stack_id);
        row_meta_data.is_none_or(|row_meta_data| row_meta_data.is_row_number_null())
    }

    /// Java `getDeliverToDirectory()`.
    pub fn get_deliver_to_directory(&self) -> String {
        self.fields.lock().unwrap().deliver_to_directory.to_string()
    }

    /// Java `setDeliverToDirectory(File)`.
    pub fn set_deliver_to_directory(&self, input: Option<&Path>) {
        let mut f = self.fields.lock().unwrap();
        match input {
            Some(input) => {
                f.deliver_to_directory
                    .set(Some(&utilities::java_io_file_get_absolute_path(
                        &input.to_string_lossy(),
                    )))
            }
            None => f.deliver_to_directory.reset(),
        }
    }

    /// Java `getInputDirectiveFile()`.
    pub fn get_input_directive_file(&self) -> String {
        self.fields.lock().unwrap().input_directive_file.to_string()
    }

    /// Java `setInputDirectiveFile(File)`.
    pub fn set_input_directive_file(&self, input: Option<&Path>) {
        let mut f = self.fields.lock().unwrap();
        match input {
            Some(input) => {
                f.input_directive_file
                    .set(Some(&utilities::java_io_file_get_absolute_path(
                        &input.to_string_lossy(),
                    )))
            }
            None => f.input_directive_file.reset(),
        }
    }
}

impl DialogCompleteClient for BatchRunTomoMetaData {
    /// Java `msgDialogComplete(String)`.
    fn msg_dialog_complete(&self, prepend: Option<&str>) {
        self.repair_image_filename_style(prepend.unwrap_or("null"));
    }
}

impl Storable for BatchRunTomoMetaData {
    /// Java `store(Properties)`: `store(props, "")`.
    fn store(&self, properties: &mut BTreeMap<String, String>) {
        self.store_with_prepend(properties, "");
    }

    /// Java `store(Properties, String)`.
    fn store_with_prepend(&self, properties: &mut BTreeMap<String, String>, prepend: &str) {
        BatchRunTomoMetaData::store_with_prepend(self, properties, prepend);
    }

    /// Java `load(Properties)`: `load(props, "")`.
    fn load(&self, properties: &mut BTreeMap<String, String>) {
        self.load_with_prepend(properties, "");
    }

    /// Java `load(Properties, String)`.
    fn load_with_prepend(&self, properties: &mut BTreeMap<String, String>, prepend: &str) {
        BatchRunTomoMetaData::load_with_prepend(self, properties, prepend);
    }
}

impl BaseMetaData for BatchRunTomoMetaData {
    fn base(&self) -> &BaseMetaDataBase {
        &self.base
    }

    /// Java `getDatasetName()`.
    fn get_dataset_name(&self) -> Option<String> {
        Some(self.fields.lock().unwrap().root_name.to_string())
    }

    /// Java `isValid()`.
    fn is_valid(&self) -> bool {
        self.validate().is_none()
    }

    /// Java `getMetaDataFileName()`.
    fn get_meta_data_file_name(&self) -> Option<String> {
        use super::const_string_property::ConstStringProperty;
        let f = self.fields.lock().unwrap();
        if f.root_name.is_empty() {
            return None;
        }
        Some(dataset_files::get_batch_run_tomo_data_file_name(
            &f.root_name.to_string(),
        ))
    }

    /// Java package-private `getGroupKey()`.
    fn get_group_key(&self) -> Option<String> {
        Some(GROUP_KEY.to_owned())
    }

    /// Java `getName()`.
    fn get_name(&self) -> Option<String> {
        use super::const_string_property::ConstStringProperty;
        let f = self.fields.lock().unwrap();
        if f.root_name.is_empty() {
            return Some(NEW_TITLE.to_owned());
        }
        Some(f.root_name.to_string())
    }

    /// Java `createPrepend(String)`.  Better quality createPrepend.  Parent
    /// createPrepend cannot be improved without breaking backwards compatibility.
    fn create_prepend(&self, prepend: &str) -> Option<String> {
        if java_lang_string_matches_whitespace(prepend) {
            return self.get_group_key();
        }
        let prepend = java_lang_string_trim(prepend);
        if prepend.ends_with('.') {
            return Some(format!("{}{}", prepend, GROUP_KEY));
        }
        Some(format!("{}.{}", prepend, GROUP_KEY))
    }

    /// Java package-private `checkImageFilenameStyleLoaded(String)`.
    fn check_image_filename_style_loaded(&self, parent_prepend: &str) {
        if !self.base.was_image_filename_style_loaded()
            && !self.repair_image_filename_style_from_param(parent_prepend)
        {
            self.repair_image_filename_style(parent_prepend);
        }
    }

    /// Java package-private `repairImageFilenameStyle(String)` (Bug# 2403).  Uses the
    /// created datasts or the environment (as a fallback) to repair imageFilenameStyle
    /// [Bug# 2386].  Repair for all brt datasets created before the Bug# 2388 fix is
    /// needed because BaseMetaData was never saved.  Before the Bug# 2388 fix, once this
    /// dataset was reloaded, the incorrect image filename style made this dataset
    /// incompatible from datasets it created before it was first reloaded.
    ///
    /// This repair only looks at one dataset, ignoring the situation where there are
    /// different namingStyles in different datasets.  This is because that situation is
    /// considered to be unlikely.  If this situation did happen, it would be caused by
    /// some of the datasets being created before any reloading, and other being created
    /// after a reload.
    fn repair_image_filename_style(&self, prepend: &str) {
        if self.base.was_image_filename_style_loaded() {
            return;
        }
        if !self.manager.is_setup_done() {
            // This repair cannot be done until the dialog containing the dataset table
            // is complete.
            let client: &'static BatchRunTomoMetaData = self.manager.get_meta_data();
            self.manager
                .add_dialog_complete_listener(Some(DialogCompleteListener::new(
                    Some(client as &'static dyn DialogCompleteClient),
                    Some(prepend),
                )));
            return;
        }
        let repair_type;
        // Get the naming style from one of the datasets.
        let mut image_filename_style = self.manager.get_dataset_image_filename_style();
        if image_filename_style.is_some() {
            repair_type =
                "image file name style from one of the datasets generated by this project";
        } else {
            // Fallback: use the environment's naming style
            image_filename_style =
                Some(ImageFileMetaData::get_temp_instance().get_image_filename_style());
            repair_type = "image file name style from the environment";
        }
        let key = self
            .get_image_filename_style_key(prepend)
            .unwrap_or_else(|| "null".to_owned());
        if let Some(style) = image_filename_style {
            eprintln!(
                "\nINFO: Attempting to repair the {} property, which is missing from the\ndataset file.  Using the {} and\nsetting {} to {}.  [Bug# 2386]\n",
                key, repair_type, key, style
            );
            if self.correct_image_filename_style(prepend, image_filename_style) {
                return;
            }
        }
        eprintln!(
            "\nERROR: Unable to repair the {} property, which is missing from the\ndataset file.  Unable to use the {}.\n[Bug# 2386]\n",
            key, repair_type
        );
    }
}
