//! `IMOD/Etomo/src/etomo/type/UserConfiguration.java`.
//!
//! The user's settings (`$HOME/.etomo`), stored and loaded through `Storable`.
//!
//! **Shape.**  `Storable.store` takes `&self`, but the Java `store` writes two fields
//! as it goes: it sets `revisionNumber` to the current revision, and it walks the MRU
//! list with `CircularBuffer.get()`, which moves the buffer's head.  Those two fields
//! are therefore behind `Mutex`es (the configuration is held by the process-global
//! `EtomoDirector`, so it must be `Send`).  `Storable.load` takes the properties by
//! shared reference while the translated `StringProperty.load` takes
//! `Option<&mut BTreeMap>`; it reads only, so it is handed a copy.  Java `Properties`
//! is `BTreeMap<String, String>` (no null keys or values).  `java.awt.Point` is
//! `(x, y)`, and `File` a path.
//!
//! **Numbers in the settings file.**  `load` parses seven properties with
//! `Integer.parseInt`, which throws `NumberFormatException` on a value that is not an
//! integer and abandons the rest of `load` (and so every setting after it, and
//! `loaded`).  Fixed in translation: an unparsable value leaves that field at the value
//! it had, and loading continues.
// TODO(unit): needs etomo/type/FrameType.java - the class has no module under type/;
// the translated `FrameType` (variants Main and Sub) is in ui/swing/etomo_frame.rs.
#![allow(dead_code)]

use std::collections::{BTreeMap, HashMap};
use std::sync::Mutex;

use crate::imod::etomo::storage::storable::StorableValue;
use crate::imod::etomo::r#type::const_etomo_number::{
    ConstEtomoNumber, INTEGER_NULL_VALUE, Type, java_lang_integer_parse_int,
    java_lang_string_matches_whitespace,
};
use crate::imod::etomo::r#type::const_string_property::ConstStringProperty;
use crate::imod::etomo::r#type::etomo_boolean2::EtomoBoolean2;
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::r#type::etomo_version::EtomoVersion;
use crate::imod::etomo::r#type::string_property::StringProperty;
use crate::imod::etomo::ui::swing::etomo_frame::FrameType;
use crate::imod::etomo::util::circular_buffer::CircularBuffer;
use crate::imod::etomo::util::utilities::java_io_file_get_absolute_path;

/// Java private `COMPACT_DISPLAY_KEY`.
const COMPACT_DISPLAY_KEY: &str = "CompactDisplay";
/// Java private `DEFAULTS_KEY`.
const DEFAULTS_KEY: &str = "Defaults";
/// Java private `CURRENT_REVISION_NUMBER`.
const CURRENT_REVISION_NUMBER: &str = "1.3";
/// Java private `MONTAGE_KEY`, `DEFAULTS_KEY + ".Montage"`.
const MONTAGE_KEY: &str = "Defaults.Montage";
/// Java private `PARALLEL_PROCESSING_KEY`.
const PARALLEL_PROCESSING_KEY: &str = "ParallelProcessing";
/// Java private `SINGLE_AXIS_KEY`, `DEFAULTS_KEY + ".SingleAxis"`.
const SINGLE_AXIS_KEY: &str = "Defaults.SingleAxis";
/// Java private `NO_PARALLEL_PROCESSING_KEY`, `DEFAULTS_KEY + ".NoParallelProcessing"`.
const NO_PARALLEL_PROCESSING_KEY: &str = "Defaults.NoParallelProcessing";
/// Java private `GPU_PROCESSING_DEFAULT_KEY`, `DEFAULTS_KEY + ".GpuProcessingDefault"`.
const GPU_PROCESSING_DEFAULT_KEY: &str = "Defaults.GpuProcessingDefault";
/// Java private `SWAP_Y_AND_Z_KEY`, `DEFAULTS_KEY + ".SwapYAndZ"`.
const SWAP_Y_AND_Z_KEY: &str = "Defaults.SwapYAndZ";
/// Java private `TILT_ANGLES_RAWTLT_FILE_KEY`, `DEFAULTS_KEY + ".TiltAnglesRawtltFile"`.
const TILT_ANGLES_RAWTLT_FILE_KEY: &str = "Defaults.TiltAnglesRawtltFile";
/// Java private `PLUGIN_KEY`.
const PLUGIN_KEY: &str = "Plugin";
/// Java private `DEFAULT_FONT_SIZE`.
const DEFAULT_FONT_SIZE: i32 = 12;

/// `Boolean.valueOf(String).booleanValue()`: true only for "true", ignoring case.
fn java_lang_boolean_value_of(value: &str) -> bool {
    value.eq_ignore_ascii_case("true")
}

/// Java `UserConfiguration implements Storable`.
pub struct UserConfiguration {
    /// Java private final field `revisionNumber`.  See the module header for the
    /// `Mutex`.
    revision_number: Mutex<EtomoVersion>,
    /// Java private final field `gpuProcessing`.
    gpu_processing: EtomoBoolean2,
    /// Java private final field `cpus`.
    cpus: EtomoNumber,
    /// Java private final field `localGPUs`.
    local_gpus: EtomoNumber,
    /// Java private final field `parallelTableSize`.
    parallel_table_size: EtomoNumber,
    /// Java private final field `joinTableSize`.
    join_table_size: EtomoNumber,
    /// Java private final field `peetTableSize`.
    peet_table_size: EtomoNumber,
    /// Java private final field `batchTableSize`.
    batch_table_size: EtomoNumber,
    /// Java private final field `mainLastLocationX`.
    main_last_location_x: EtomoNumber,
    /// Java private final field `mainLastLocationY`.
    main_last_location_y: EtomoNumber,
    /// Java private final field `subLastLocationX`.
    sub_last_location_x: EtomoNumber,
    /// Java private final field `subLastLocationY`.
    sub_last_location_y: EtomoNumber,
    /// Java private final field `setFEIPixelSize`.
    set_fei_pixel_size: EtomoBoolean2,
    /// Java private final field `userTemplateDirAbsPath`.
    user_template_dir_abs_path: StringProperty,
    /// Java private final field `scopeTemplateAbsPath`.
    scope_template_abs_path: StringProperty,
    /// Java private final field `systemTemplateAbsPath`.
    system_template_abs_path: StringProperty,
    /// Java private final field `userTemplateAbsPath`.
    user_template_abs_path: StringProperty,
    /// Java private final field `emailAddress`.
    email_address: StringProperty,
    /// Java private final field `useEmailAddress`.
    use_email_address: EtomoBoolean2,
    /// Java private final field `smtpServer`.
    smtp_server: StringProperty,
    /// Java private final field `removeExcludedViews`.
    remove_excluded_views: EtomoBoolean2,
    /// Java private final field `batchCloseDatasets`.
    batch_close_datasets: EtomoNumber,

    /// Java private field `nativeLookAndFeel`.
    native_look_and_feel: bool,
    /// Java private field `advancedDialogs`.
    advanced_dialogs: bool,
    /// Java private field `compactDisplay`.
    compact_display: bool,
    /// Java private field `toolTipsInitialDelay`.
    tool_tips_initial_delay: i32,
    /// Java private field `toolTipsDismissDelay`.
    tool_tips_dismiss_delay: i32,
    /// Java private field `nMRUFiles`.
    n_mru_files: i32,
    /// Java private field `MRUFileList`.  See the module header for the `Mutex`.
    mru_file_list: Mutex<CircularBuffer<String>>,
    /// Java private field `fontFamily`.
    font_family: Option<String>,
    /// Java private field `fontSize`.
    font_size: i32,
    /// Java private field `mainWindowWidth`.
    main_window_width: i32,
    /// Java private field `mainWindowHeight`.
    main_window_height: i32,
    /// Java private field `autoFit`.
    auto_fit: bool,
    /// Java private field `montage`, initialised to null.
    montage: Option<EtomoBoolean2>,
    /// Java private field `parallelProcessing`, initialised to null.
    parallel_processing: Option<EtomoBoolean2>,
    /// Java private field `singleAxis`, initialised to null.
    single_axis: Option<EtomoBoolean2>,
    /// Java private field `noParallelProcessing`, initialised to null.
    no_parallel_processing: Option<EtomoBoolean2>,
    /// Java private field `gpuProcessingDefault`, initialised to null.
    gpu_processing_default: Option<EtomoBoolean2>,
    /// Java private field `swapYAndZ`, initialised to null.
    swap_y_and_z: Option<EtomoBoolean2>,
    /// Java private field `tiltAnglesRawtltFile`, initialised to null.
    tilt_angles_rawtlt_file: Option<EtomoBoolean2>,
    /// Java private field `pluginMap`, initialised to null.
    plugin_map: Option<HashMap<String, EtomoBoolean2>>,
    /// Java private field `loaded`, initialised to false.
    loaded: bool,
}

impl UserConfiguration {
    /// Java `UserConfiguration()`.
    pub fn new() -> UserConfiguration {
        let n_mru_files = 10;
        let mut instance = UserConfiguration {
            revision_number: Mutex::new(EtomoVersion::get_instance(
                Some("RevisionNumber"),
                Some(CURRENT_REVISION_NUMBER),
            )),
            gpu_processing: EtomoBoolean2::new_with_name("GpuProcessing"),
            cpus: EtomoNumber::new_with_name("Cpus"),
            local_gpus: EtomoNumber::new_with_name("Gpus"),
            parallel_table_size: EtomoNumber::new_with_name("ParallelTableSize"),
            join_table_size: EtomoNumber::new_with_name("JoinTableSize"),
            peet_table_size: EtomoNumber::new_with_name("PeetTableSize"),
            batch_table_size: EtomoNumber::new_with_name("BatchTableSize"),
            main_last_location_x: EtomoNumber::new_with_name("Main.LastLocationX"),
            main_last_location_y: EtomoNumber::new_with_name("Main.LastLocationY"),
            sub_last_location_x: EtomoNumber::new_with_name("Sub.LastLocationX"),
            sub_last_location_y: EtomoNumber::new_with_name("Sub.LastLocationY"),
            set_fei_pixel_size: EtomoBoolean2::new_with_name(&format!(
                "{}.SetFEIPixelSize",
                DEFAULTS_KEY
            )),
            user_template_dir_abs_path: StringProperty::new_with_key_and_return_null_when_empty(
                Some(&format!("{}.UserTemplateDir", DEFAULTS_KEY)),
                true,
            ),
            scope_template_abs_path: StringProperty::new_with_key_and_return_null_when_empty(
                Some(&format!("{}.ScopeTemplate", DEFAULTS_KEY)),
                true,
            ),
            system_template_abs_path: StringProperty::new_with_key_and_return_null_when_empty(
                Some(&format!("{}.SystemTemplate", DEFAULTS_KEY)),
                true,
            ),
            user_template_abs_path: StringProperty::new_with_key_and_return_null_when_empty(
                Some(&format!("{}.UserTemplate", DEFAULTS_KEY)),
                true,
            ),
            email_address: StringProperty::new_with_key_and_return_null_when_empty(
                Some("EmailAddress"),
                true,
            ),
            use_email_address: EtomoBoolean2::new_with_name("EmailAddress.Use"),
            smtp_server: StringProperty::new_with_key_and_return_null_when_empty(
                Some("SmtpServer"),
                true,
            ),
            remove_excluded_views: EtomoBoolean2::new_with_name("RemoveExcludedViews"),
            batch_close_datasets: EtomoNumber::new_with_type_and_name(
                Type::Boolean,
                "Batch.CloseDatasets",
            ),
            native_look_and_feel: false,
            advanced_dialogs: false,
            compact_display: false,
            tool_tips_initial_delay: 2000,
            tool_tips_dismiss_delay: 20000,
            n_mru_files,
            mru_file_list: Mutex::new(CircularBuffer::new(n_mru_files)),
            font_family: Some("Dialog".to_string()),
            font_size: DEFAULT_FONT_SIZE,
            main_window_width: 800,
            main_window_height: 600,
            auto_fit: false,
            montage: None,
            parallel_processing: None,
            single_axis: None,
            no_parallel_processing: None,
            gpu_processing_default: None,
            swap_y_and_z: None,
            tilt_angles_rawtlt_file: None,
            plugin_map: None,
            loaded: false,
        };
        {
            let mut mru_file_list = instance.mru_file_list.lock().unwrap();
            for _ in 0..instance.n_mru_files {
                mru_file_list.put(Some(String::new()));
            }
        }
        instance.parallel_table_size.set_display_value_int(15);
        instance.join_table_size.set_display_value_int(10);
        instance.peet_table_size.set_display_value_int(10);
        instance.batch_table_size.set_display_value_int(20);
        instance.set_fei_pixel_size.set_display_value_boolean(true);
        instance.local_gpus.set_display_value_int(1);
        instance
    }

    /// Java package-private `paramString()`.
    pub(crate) fn param_string(&self) -> String {
        format!(
            "\n,revisionNumber={},\nnativeLookAndFeel={},\nadvancedDialogs={},\ntoolTipsInitialDelay={},\ntoolTipsDismissDelay={},\nnMRUFiles={},\nMRUFileList={},\nfontFamily={},\nfontSize={},\nmainWindowWidth={},\nmainWindowHeight={},\nautoFit={}",
            self.revision_number.lock().unwrap(),
            self.native_look_and_feel,
            self.advanced_dialogs,
            self.tool_tips_initial_delay,
            self.tool_tips_dismiss_delay,
            self.n_mru_files,
            self.mru_file_list.lock().unwrap(),
            self.font_family.as_deref().unwrap_or("null"),
            self.font_size,
            self.main_window_width,
            self.main_window_height,
            self.auto_fit
        )
    }

    /// Java private `getPrepend(String)`.  Modify prepend to include "Setup".
    fn get_prepend(prepend: Option<&str>) -> String {
        match prepend {
            None => String::new(),
            Some(prepend) if java_lang_string_matches_whitespace(prepend) => String::new(),
            Some(prepend) => prepend.to_string() + ".Setup",
        }
    }

    /// Java private `getGroup(String)`.  Run on the output of getPrepend.  (The Java
    /// compares with `==` against the interned "" `getPrepend` returns, which is the
    /// same as comparing contents here.)
    fn get_group(modified_prepend: &str) -> String {
        if modified_prepend.is_empty() {
            return String::new();
        }
        modified_prepend.to_string() + "."
    }

    /// Java `isLoaded()`.
    pub fn is_loaded(&self) -> bool {
        self.loaded
    }

    /// Java `setPlugin(String, boolean)`.  Adds or updates a plugin property.
    pub fn set_plugin(&mut self, key: Option<&str>, value: bool) {
        let key = match key {
            Some(key) if !java_lang_string_matches_whitespace(key) => key,
            _ => return,
        };
        let key = PLUGIN_KEY.to_string() + "." + key;
        let mut property: Option<EtomoBoolean2> = None;
        if let Some(plugin_map) = &self.plugin_map {
            property = plugin_map.get(&key).cloned();
        }
        let mut property = match property {
            None => EtomoBoolean2::new_with_name(&key),
            Some(property) => property,
        };
        property.set_boolean(value);
        if self.plugin_map.is_none() {
            self.plugin_map = Some(HashMap::new());
        }
        self.plugin_map.as_mut().unwrap().insert(key, property);
    }

    /// Java `putDataFile(String)`.  Put a etomo data file onto the MRU list if it is
    /// not already there.
    pub fn put_data_file(&mut self, filename: Option<&str>) {
        let mut mru_file_list = self.mru_file_list.lock().unwrap();
        let filename = filename.map(|filename| filename.to_string());
        if mru_file_list.search(filename.as_ref()) == -1 {
            mru_file_list.put(filename);
        }
    }

    /// Java `hasPlugin(String)`.
    pub fn has_plugin(&self, key: Option<&str>) -> bool {
        let key = match key {
            Some(key) if !java_lang_string_matches_whitespace(key) => key,
            _ => return false,
        };
        let key = PLUGIN_KEY.to_string() + "." + key;
        if let Some(plugin_map) = &self.plugin_map
            && plugin_map.contains_key(&key)
        {
            return true;
        }
        false
    }

    /// Java `isPlugin(String)`.
    ///
    /// UserConfiguration.java:432 calls `pluginMap.get(key).is()` when the map exists,
    /// which throws `NullPointerException` for a plugin the map does not hold.  Fixed in
    /// translation: an unknown plugin is false, as it is when there is no map.
    pub fn is_plugin(&self, key: Option<&str>) -> bool {
        let key = match key {
            Some(key) if !java_lang_string_matches_whitespace(key) => key,
            _ => return false,
        };
        let key = PLUGIN_KEY.to_string() + "." + key;
        if let Some(plugin_map) = &self.plugin_map {
            return plugin_map.get(&key).is_some_and(|property| property.is());
        }
        false
    }

    /// Java `getDataFile()`.  Get the next etomo data file from the MRU list.
    pub fn get_data_file(&mut self) -> Option<String> {
        self.mru_file_list.lock().unwrap().get()
    }

    /// Java `getAdvancedDialogs()`.  Get the advanced dialog state.
    pub fn get_advanced_dialogs(&self) -> bool {
        self.advanced_dialogs
    }

    /// Java `getCompactDisplay()`.
    pub fn get_compact_display(&self) -> bool {
        self.compact_display
    }

    /// Java `setAdvancedDialogs(boolean)`.  Set the advanced dialog state.
    pub fn set_advanced_dialogs(&mut self, state: bool) {
        self.advanced_dialogs = state;
    }

    /// Java `setCompactDisplay(boolean)`.
    pub fn set_compact_display(&mut self, state: bool) {
        self.compact_display = state;
    }

    /// Java `getNativeLookAndFeel()`.  Get the native look and feel state.
    pub fn get_native_look_and_feel(&self) -> bool {
        self.native_look_and_feel
    }

    /// Java `setNativeLookAndFeel(boolean)`.  Set the native look and feel state.
    pub fn set_native_look_and_feel(&mut self, state: bool) {
        self.native_look_and_feel = state;
    }

    /// Java `getToolTipsInitialDelay()`.
    pub fn get_tool_tips_initial_delay(&self) -> i32 {
        self.tool_tips_initial_delay
    }

    /// Java `setToolTipsInitialDelay(int)`.
    pub fn set_tool_tips_initial_delay(&mut self, milli_seconds: i32) {
        self.tool_tips_initial_delay = milli_seconds;
    }

    /// Java `getToolTipsDismissDelay()`.
    pub fn get_tool_tips_dismiss_delay(&self) -> i32 {
        self.tool_tips_dismiss_delay
    }

    /// Java `setToolTipsDismissDelay(int)`.
    pub fn set_tool_tips_dismiss_delay(&mut self, milli_seconds: i32) {
        self.tool_tips_dismiss_delay = milli_seconds;
    }

    /// Java `getNMRUFIles()`.  Get the number of MRU files stored.
    pub fn get_n_mru_files(&self) -> i32 {
        self.n_mru_files
    }

    /// Java `setNMRUFIles(int)`.  Set the number of MRU files stored.
    pub fn set_n_mru_files(&mut self, n_files: i32) {
        self.n_mru_files = n_files;
    }

    /// Java `getMRUFileList()`.  Get the MRU file list at a string array.
    pub fn get_mru_file_list(&mut self) -> Vec<Option<String>> {
        let mut mru_file_list = self.mru_file_list.lock().unwrap();
        let mut list = vec![None; mru_file_list.size() as usize];
        for i in 0..mru_file_list.size() as usize {
            list[i] = mru_file_list.get();
        }
        list
    }

    /// Java `getFontFamily()`.
    pub fn get_font_family(&self) -> Option<String> {
        self.font_family.clone()
    }

    /// Java `getFontSize()`.
    pub fn get_font_size(&self) -> i32 {
        self.font_size
    }

    /// Java `setFontFamily(String)`.
    pub fn set_font_family(&mut self, font_family: Option<&str>) {
        self.font_family = font_family.map(|font_family| font_family.to_string());
    }

    /// Java `setFontSize(int)`.
    pub fn set_font_size(&mut self, font_size: i32) {
        self.font_size = font_size;
    }

    /// Java `getMainWindowHeight()`.
    pub fn get_main_window_height(&self) -> i32 {
        self.main_window_height
    }

    /// Java `getMainWindowWidth()`.
    pub fn get_main_window_width(&self) -> i32 {
        self.main_window_width
    }

    /// Java `setMainWindowHeight(int)`.
    pub fn set_main_window_height(&mut self, main_window_height: i32) {
        self.main_window_height = main_window_height;
    }

    /// Java `setMainWindowWidth(int)`.
    pub fn set_main_window_width(&mut self, main_window_width: i32) {
        self.main_window_width = main_window_width;
    }

    /// Java `isAutoFit()`.
    pub fn is_auto_fit(&self) -> bool {
        self.auto_fit
    }

    /// Java `setAutoFit(boolean)`.
    pub fn set_auto_fit(&mut self, auto_fit: bool) {
        self.auto_fit = auto_fit;
    }

    /// Java `setBatchCloseDatasets(boolean)`.
    pub fn set_batch_close_datasets(&mut self, input: bool) {
        self.batch_close_datasets.set_boolean(input);
    }

    /// Java `isBatchCloseDatasets()`.
    pub fn is_batch_close_datasets(&self) -> bool {
        self.batch_close_datasets.is()
    }

    /// Java `getCpus()`.
    pub fn get_cpus(&self) -> &ConstEtomoNumber {
        &self.cpus
    }

    /// Java `getLocalGPUs()`.
    pub fn get_local_gpus(&self) -> &ConstEtomoNumber {
        &self.local_gpus
    }

    /// Java `getLocalGPUsInt()`.
    pub fn get_local_gpus_int(&self) -> i32 {
        self.local_gpus.get_int()
    }

    /// Java `getParallelTableSize()`.
    pub fn get_parallel_table_size(&self) -> &ConstEtomoNumber {
        &self.parallel_table_size
    }

    /// Java `getJoinTableSize()`.
    pub fn get_join_table_size(&self) -> &ConstEtomoNumber {
        &self.join_table_size
    }

    /// Java `getPeetTableSize()`.
    pub fn get_peet_table_size(&self) -> &ConstEtomoNumber {
        &self.peet_table_size
    }

    /// Java `getBatchTableSize()`.
    pub fn get_batch_table_size(&self) -> &ConstEtomoNumber {
        &self.batch_table_size
    }

    /// Java `isLastLocationSet(FrameType)`.
    pub fn is_last_location_set(&self, frame_type: FrameType) -> bool {
        if frame_type == FrameType::Main {
            return !self.main_last_location_x.is_null() && !self.main_last_location_y.is_null();
        }
        if frame_type == FrameType::Sub {
            return !self.sub_last_location_x.is_null() && !self.sub_last_location_y.is_null();
        }
        false
    }

    /// Java `getLastLocationX(FrameType)`.
    pub fn get_last_location_x(&self, frame_type: FrameType) -> i32 {
        if frame_type == FrameType::Main {
            return self.main_last_location_x.get_int();
        }
        if frame_type == FrameType::Sub {
            return self.sub_last_location_x.get_int();
        }
        INTEGER_NULL_VALUE
    }

    /// Java `getLastLocationY(FrameType)`.
    pub fn get_last_location_y(&self, frame_type: FrameType) -> i32 {
        if frame_type == FrameType::Main {
            return self.main_last_location_y.get_int();
        }
        if frame_type == FrameType::Sub {
            return self.sub_last_location_y.get_int();
        }
        INTEGER_NULL_VALUE
    }

    /// Java `setLastLocation(FrameType, Point)`.  A null point for the main frame
    /// resets the sub frame's location and vice versa; odd, but kept as the source
    /// writes it.
    pub fn set_last_location(&mut self, frame_type: FrameType, point: Option<(i32, i32)>) {
        match point {
            None => {
                if frame_type == FrameType::Main {
                    self.sub_last_location_x.reset();
                    self.sub_last_location_y.reset();
                } else if frame_type == FrameType::Sub {
                    self.main_last_location_x.reset();
                    self.main_last_location_y.reset();
                }
            }
            Some((x, y)) => {
                if frame_type == FrameType::Main {
                    self.main_last_location_x.set_int(x);
                    self.main_last_location_y.set_int(y);
                } else if frame_type == FrameType::Sub {
                    self.sub_last_location_x.set_int(x);
                    self.sub_last_location_y.set_int(y);
                }
            }
        }
    }

    /// Java `isParallelProcessing()`.
    pub fn is_parallel_processing(&self) -> bool {
        match &self.parallel_processing {
            None => false,
            Some(parallel_processing) => parallel_processing.is(),
        }
    }

    /// Java `isGpuProcessing()`.
    pub fn is_gpu_processing(&self) -> bool {
        self.gpu_processing.is()
    }

    /// Java `getNoParallelProcessing()`.
    pub fn get_no_parallel_processing(&self) -> bool {
        match &self.no_parallel_processing {
            None => false,
            Some(no_parallel_processing) => no_parallel_processing.is(),
        }
    }

    /// Java `getGpuProcessingDefault()`.
    pub fn get_gpu_processing_default(&self) -> bool {
        match &self.gpu_processing_default {
            None => false,
            Some(gpu_processing_default) => gpu_processing_default.is(),
        }
    }

    /// Java `getSingleAxis()`.
    pub fn get_single_axis(&self) -> bool {
        match &self.single_axis {
            None => false,
            Some(single_axis) => single_axis.is(),
        }
    }

    /// Java `getMontage()`.
    pub fn get_montage(&self) -> bool {
        match &self.montage {
            None => false,
            Some(montage) => montage.is(),
        }
    }

    /// Java `getSwapYAndZ()`.
    pub fn get_swap_y_and_z(&self) -> bool {
        match &self.swap_y_and_z {
            None => false,
            Some(swap_y_and_z) => swap_y_and_z.is(),
        }
    }

    /// Java `isSetFEIPixelSize()`.
    pub fn is_set_fei_pixel_size(&self) -> bool {
        self.set_fei_pixel_size.is()
    }

    /// Java `isUserTemplateDirSet()`.
    pub fn is_user_template_dir_set(&self) -> bool {
        !self.user_template_dir_abs_path.is_empty()
    }

    /// Java `getUserTemplateDir()`.
    pub fn get_user_template_dir(&self) -> Option<String> {
        self.user_template_dir_abs_path.to_string_option()
    }

    /// Java `getScopeTemplate()`.
    pub fn get_scope_template(&self) -> Option<String> {
        self.scope_template_abs_path.to_string_option()
    }

    /// Java `equalsScopeTemplate(File)`.  Returns true when the template is *not*
    /// the file's absolute path; odd for its name, but kept as the source writes it.
    pub fn equals_scope_template(&self, input: Option<&std::path::Path>) -> bool {
        let mut abs_path: Option<String> = None;
        if let Some(input) = input {
            abs_path = Some(java_io_file_get_absolute_path(&input.to_string_lossy()));
        }
        !self.scope_template_abs_path.equals(abs_path.as_deref())
    }

    /// Java `equalsSystemTemplate(File)`.  See [`Self::equals_scope_template`].
    pub fn equals_system_template(&self, input: Option<&std::path::Path>) -> bool {
        let mut abs_path: Option<String> = None;
        if let Some(input) = input {
            abs_path = Some(java_io_file_get_absolute_path(&input.to_string_lossy()));
        }
        !self.system_template_abs_path.equals(abs_path.as_deref())
    }

    /// Java `equalsUserTemplate(File)`.  See [`Self::equals_scope_template`].
    pub fn equals_user_template(&self, input: Option<&std::path::Path>) -> bool {
        let mut abs_path: Option<String> = None;
        if let Some(input) = input {
            abs_path = Some(java_io_file_get_absolute_path(&input.to_string_lossy()));
        }
        !self.user_template_abs_path.equals(abs_path.as_deref())
    }

    /// Java `isScopeTemplateSet()`.
    pub fn is_scope_template_set(&self) -> bool {
        !self.scope_template_abs_path.is_empty()
    }

    /// Java `isSystemTemplateSet()`.
    pub fn is_system_template_set(&self) -> bool {
        !self.system_template_abs_path.is_empty()
    }

    /// Java `isUserTemplateSet()`.
    pub fn is_user_template_set(&self) -> bool {
        !self.user_template_abs_path.is_empty()
    }

    /// Java `getSystemTemplate()`.
    pub fn get_system_template(&self) -> Option<String> {
        self.system_template_abs_path.to_string_option()
    }

    /// Java `getUserTemplate()`.
    pub fn get_user_template(&self) -> Option<String> {
        self.user_template_abs_path.to_string_option()
    }

    /// Java `isTiltAnglesRawtltFile()`.
    pub fn is_tilt_angles_rawtlt_file(&self) -> bool {
        match &self.tilt_angles_rawtlt_file {
            None => false,
            Some(tilt_angles_rawtlt_file) => tilt_angles_rawtlt_file.is(),
        }
    }

    /// Java `setParallelProcessing(boolean)`.
    pub fn set_parallel_processing(&mut self, input: bool) {
        if self.parallel_processing.is_none() {
            self.parallel_processing = Some(EtomoBoolean2::new_with_name(PARALLEL_PROCESSING_KEY));
        }
        self.parallel_processing
            .as_mut()
            .unwrap()
            .set_boolean(input);
    }

    /// Java `setGpuProcessing(boolean)`.
    pub fn set_gpu_processing(&mut self, input: bool) {
        self.gpu_processing.set_boolean(input);
    }

    /// Java `setCpus(String)`.
    pub fn set_cpus(&mut self, input: Option<&str>) {
        self.cpus.set_string(input);
    }

    /// Java `setLocalGPUs(String)`.
    pub fn set_local_gpus(&mut self, input: Option<&str>) {
        self.local_gpus.set_string(input);
    }

    /// Java `setParallelTableSize(String)`.
    pub fn set_parallel_table_size(&mut self, input: Option<&str>) {
        self.parallel_table_size.set_string(input);
    }

    /// Java `setJoinTableSize(String)`.
    pub fn set_join_table_size(&mut self, input: Option<&str>) {
        self.join_table_size.set_string(input);
    }

    /// Java `setPeetTableSize(String)`.
    pub fn set_peet_table_size(&mut self, input: Option<&str>) {
        self.peet_table_size.set_string(input);
    }

    /// Java `setBatchTableSize(String)`.
    pub fn set_batch_table_size(&mut self, input: Option<&str>) {
        self.batch_table_size.set_string(input);
    }

    /// Java `setNoParallelProcessing(boolean)`.
    pub fn set_no_parallel_processing(&mut self, input: bool) {
        if self.no_parallel_processing.is_none() {
            self.no_parallel_processing =
                Some(EtomoBoolean2::new_with_name(NO_PARALLEL_PROCESSING_KEY));
        }
        self.no_parallel_processing
            .as_mut()
            .unwrap()
            .set_boolean(input);
    }

    /// Java `setGpuProcessingDefault(boolean)`.
    pub fn set_gpu_processing_default(&mut self, input: bool) {
        if self.gpu_processing_default.is_none() {
            self.gpu_processing_default =
                Some(EtomoBoolean2::new_with_name(GPU_PROCESSING_DEFAULT_KEY));
        }
        self.gpu_processing_default
            .as_mut()
            .unwrap()
            .set_boolean(input);
    }

    /// Java `setSingleAxis(boolean)`.
    pub fn set_single_axis(&mut self, input: bool) {
        if self.single_axis.is_none() {
            self.single_axis = Some(EtomoBoolean2::new_with_name(SINGLE_AXIS_KEY));
        }
        self.single_axis.as_mut().unwrap().set_boolean(input);
    }

    /// Java `setMontage(boolean)`.
    pub fn set_montage(&mut self, input: bool) {
        if self.montage.is_none() {
            self.montage = Some(EtomoBoolean2::new_with_name(MONTAGE_KEY));
        }
        self.montage.as_mut().unwrap().set_boolean(input);
    }

    /// Java `setSwapYAndZ(boolean)`.
    pub fn set_swap_y_and_z(&mut self, input: bool) {
        if self.swap_y_and_z.is_none() {
            self.swap_y_and_z = Some(EtomoBoolean2::new_with_name(SWAP_Y_AND_Z_KEY));
        }
        self.swap_y_and_z.as_mut().unwrap().set_boolean(input);
    }

    /// Java `setSetFEIPixelSize(boolean)`.
    pub fn set_set_fei_pixel_size(&mut self, input: bool) {
        self.set_fei_pixel_size.set_boolean(input);
    }

    /// Java `setUserTemplateDir(File)`.
    pub fn set_user_template_dir(&mut self, input: Option<&std::path::Path>) {
        match input {
            Some(input) => {
                self.user_template_dir_abs_path
                    .set(Some(&java_io_file_get_absolute_path(
                        &input.to_string_lossy(),
                    )))
            }
            None => self.user_template_dir_abs_path.reset(),
        }
    }

    /// Java `setUserTemplate(File)`.
    pub fn set_user_template(&mut self, input: Option<&std::path::Path>) {
        match input {
            Some(input) => self
                .user_template_abs_path
                .set(Some(&java_io_file_get_absolute_path(
                    &input.to_string_lossy(),
                ))),
            None => self.user_template_abs_path.reset(),
        }
    }

    /// Java `setSystemTemplate(File)`.
    pub fn set_system_template(&mut self, input: Option<&std::path::Path>) {
        match input {
            Some(input) => {
                self.system_template_abs_path
                    .set(Some(&java_io_file_get_absolute_path(
                        &input.to_string_lossy(),
                    )))
            }
            None => self.system_template_abs_path.reset(),
        }
    }

    /// Java `setScopeTemplate(File)`.
    pub fn set_scope_template(&mut self, input: Option<&std::path::Path>) {
        match input {
            Some(input) => self
                .scope_template_abs_path
                .set(Some(&java_io_file_get_absolute_path(
                    &input.to_string_lossy(),
                ))),
            None => self.scope_template_abs_path.reset(),
        }
    }

    /// Java `setTiltAnglesRawtltFile(boolean)`.
    pub fn set_tilt_angles_rawtlt_file(&mut self, input: bool) {
        if self.tilt_angles_rawtlt_file.is_none() {
            self.tilt_angles_rawtlt_file =
                Some(EtomoBoolean2::new_with_name(TILT_ANGLES_RAWTLT_FILE_KEY));
        }
        self.tilt_angles_rawtlt_file
            .as_mut()
            .unwrap()
            .set_boolean(input);
    }

    /// Java `isUseEmailAddress()`.
    pub fn is_use_email_address(&self) -> bool {
        self.use_email_address.is()
    }

    /// Java `getSmtpServer()`.
    pub fn get_smtp_server(&self) -> Option<String> {
        self.smtp_server.to_string_option()
    }

    /// Java `isRemoveExcludedViews()`.
    pub fn is_remove_excluded_views(&self) -> bool {
        self.remove_excluded_views.is()
    }

    /// Java `setRemoveExcludedViews(boolean)`.
    pub fn set_remove_excluded_views(&mut self, input: bool) {
        self.remove_excluded_views.set_boolean(input);
    }

    /// Java `setSmtpServer(String)`.
    pub fn set_smtp_server(&mut self, input: Option<&str>) {
        self.smtp_server.set(input);
    }

    /// Java `setUseEmailAddress(boolean)`.
    pub fn set_use_email_address(&mut self, input: bool) {
        self.use_email_address.set_boolean(input);
    }

    /// Java `getEmailAddress()`.
    pub fn get_email_address(&self) -> Option<String> {
        self.email_address.to_string_option()
    }

    /// Java `setEmailAddress(String)`.
    pub fn set_email_address(&mut self, input: Option<&str>) {
        self.email_address.set(input);
    }
}

impl Default for UserConfiguration {
    fn default() -> UserConfiguration {
        UserConfiguration::new()
    }
}

/// Java `toString()`.
impl std::fmt::Display for UserConfiguration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "etomo.type.UserConfiguration[{}]", self.param_string())
    }
}

impl StorableValue for UserConfiguration {
    /// Java `store(Properties)`.  Insert the objects attributes into the properties
    /// object.
    fn store(&self, props: &mut BTreeMap<String, String>) {
        self.store_with_prepend(props, "");
    }

    /// Java `store(Properties, String)`.
    fn store_with_prepend(&self, props: &mut BTreeMap<String, String>, prepend: &str) {
        let orig_prepend = prepend;
        let prepend = UserConfiguration::get_prepend(Some(prepend));
        let group = UserConfiguration::get_group(&prepend);

        {
            let mut revision_number = self.revision_number.lock().unwrap();
            revision_number.set(Some(CURRENT_REVISION_NUMBER));
            revision_number.store_with_prepend(props, orig_prepend);
        }
        // props.setProperty(group + "RevisionNumber", revisionNumber);
        props.insert(
            group.clone() + "NativeLookAndFeel",
            self.native_look_and_feel.to_string(),
        );
        props.insert(
            group.clone() + "AdvancedDialogs",
            self.advanced_dialogs.to_string(),
        );
        props.insert(
            group.clone() + COMPACT_DISPLAY_KEY,
            self.compact_display.to_string(),
        );
        props.insert(
            group.clone() + "ToolTipsInitialDelay",
            self.tool_tips_initial_delay.to_string(),
        );
        props.insert(
            group.clone() + "ToolTipsDismissDelay",
            self.tool_tips_dismiss_delay.to_string(),
        );
        // `String.valueOf(fontFamily)` prints a null family as "null".
        props.insert(
            group.clone() + "FontFamily",
            self.font_family
                .clone()
                .unwrap_or_else(|| "null".to_string()),
        );
        props.insert(group.clone() + "FontSize", self.font_size.to_string());
        EtomoBoolean2::store_instance(
            self.single_axis.as_ref(),
            props,
            Some(prepend.as_str()),
            SINGLE_AXIS_KEY,
        );
        EtomoBoolean2::store_instance(
            self.montage.as_ref(),
            props,
            Some(prepend.as_str()),
            MONTAGE_KEY,
        );
        EtomoBoolean2::store_instance(
            self.no_parallel_processing.as_ref(),
            props,
            Some(prepend.as_str()),
            NO_PARALLEL_PROCESSING_KEY,
        );
        EtomoBoolean2::store_instance(
            self.gpu_processing_default.as_ref(),
            props,
            Some(prepend.as_str()),
            GPU_PROCESSING_DEFAULT_KEY,
        );
        EtomoBoolean2::store_instance(
            self.tilt_angles_rawtlt_file.as_ref(),
            props,
            Some(prepend.as_str()),
            TILT_ANGLES_RAWTLT_FILE_KEY,
        );
        EtomoBoolean2::store_instance(
            self.swap_y_and_z.as_ref(),
            props,
            Some(prepend.as_str()),
            SWAP_Y_AND_Z_KEY,
        );
        EtomoBoolean2::store_instance(
            self.parallel_processing.as_ref(),
            props,
            Some(prepend.as_str()),
            PARALLEL_PROCESSING_KEY,
        );
        self.gpu_processing
            .store_with_prepend(props, Some(prepend.as_str()));
        self.cpus.store_with_prepend(props, &prepend);
        self.local_gpus.store_with_prepend(props, &prepend);
        self.parallel_table_size.store_with_prepend(props, &prepend);
        self.join_table_size.store_with_prepend(props, &prepend);
        self.peet_table_size.store_with_prepend(props, &prepend);
        self.batch_table_size.store_with_prepend(props, &prepend);
        self.main_last_location_x
            .store_with_prepend(props, &prepend);
        self.main_last_location_y
            .store_with_prepend(props, &prepend);
        self.sub_last_location_x.store_with_prepend(props, &prepend);
        self.sub_last_location_y.store_with_prepend(props, &prepend);
        self.set_fei_pixel_size
            .store_with_prepend(props, Some(prepend.as_str()));
        self.user_template_dir_abs_path
            .store_with_prepend(Some(&mut *props), Some(prepend.as_str()));
        self.scope_template_abs_path
            .store_with_prepend(Some(&mut *props), Some(prepend.as_str()));
        self.system_template_abs_path
            .store_with_prepend(Some(&mut *props), Some(prepend.as_str()));
        self.user_template_abs_path
            .store_with_prepend(Some(&mut *props), Some(prepend.as_str()));
        self.email_address
            .store_with_prepend(Some(&mut *props), Some(prepend.as_str()));
        self.use_email_address
            .store_with_prepend(props, Some(prepend.as_str()));
        self.smtp_server
            .store_with_prepend(Some(&mut *props), Some(prepend.as_str()));
        self.remove_excluded_views
            .store_with_prepend(props, Some(prepend.as_str()));
        self.batch_close_datasets
            .store_with_prepend(props, &prepend);
        props.insert(
            group.clone() + "MainWindowWidth",
            self.main_window_width.to_string(),
        );
        props.insert(
            group.clone() + "MainWindowHeight",
            self.main_window_height.to_string(),
        );
        props.insert(group.clone() + "NMRUFiles", self.n_mru_files.to_string());
        {
            let mut mru_file_list = self.mru_file_list.lock().unwrap();
            for i in 0..self.n_mru_files {
                // `Properties.setProperty` with a null value throws; the list only ever
                // holds file names (and "" placeholders), except for slots past a
                // shortened buffer, which are stored as "".
                props.insert(
                    group.clone() + "EtomoDataFile" + &i.to_string(),
                    mru_file_list.get().unwrap_or_default(),
                );
            }
        }
        props.insert(group.clone() + "AutoFit", self.auto_fit.to_string());
        if let Some(plugin_map) = &self.plugin_map {
            for property in plugin_map.values() {
                property.store_with_prepend(props, Some(prepend.as_str()));
            }
        }
    }

    /// Java `load(Properties)`.
    fn load(&mut self, props: &BTreeMap<String, String>) {
        self.load_with_prepend(props, "");
    }

    /// Java `load(Properties, String)`.
    ///
    /// The MRU entries are read as `"EtomoDataFile" + i`, without the group that
    /// `store` writes in front of them; with a non-empty prepend they are not found
    /// again.  Odd, but kept as the source writes it.
    fn load_with_prepend(&mut self, props: &BTreeMap<String, String>, prepend: &str) {
        let orig_prepend = prepend;
        let prepend = UserConfiguration::get_prepend(Some(prepend));
        let group = UserConfiguration::get_group(&prepend);
        // The translated `StringProperty.load` takes the map mutably; it only reads.
        let mut props_copy = props.clone();
        //
        // Get the user configuration data from the Properties object
        //
        self.revision_number
            .lock()
            .unwrap()
            .load_with_prepend(props, orig_prepend);
        self.parallel_processing = EtomoBoolean2::load_instance(
            self.parallel_processing.take(),
            PARALLEL_PROCESSING_KEY,
            props,
            Some(prepend.as_str()),
        );
        self.montage = EtomoBoolean2::load_instance(
            self.montage.take(),
            MONTAGE_KEY,
            props,
            Some(prepend.as_str()),
        );
        self.single_axis = EtomoBoolean2::load_instance(
            self.single_axis.take(),
            SINGLE_AXIS_KEY,
            props,
            Some(prepend.as_str()),
        );
        self.no_parallel_processing = EtomoBoolean2::load_instance(
            self.no_parallel_processing.take(),
            NO_PARALLEL_PROCESSING_KEY,
            props,
            Some(prepend.as_str()),
        );
        self.gpu_processing_default = EtomoBoolean2::load_instance(
            self.gpu_processing_default.take(),
            GPU_PROCESSING_DEFAULT_KEY,
            props,
            Some(prepend.as_str()),
        );
        self.parallel_table_size
            .load_with_prepend(props, Some(prepend.as_str()));
        self.swap_y_and_z = EtomoBoolean2::load_instance(
            self.swap_y_and_z.take(),
            SWAP_Y_AND_Z_KEY,
            props,
            Some(prepend.as_str()),
        );
        self.cpus.load_with_prepend(props, Some(prepend.as_str()));
        self.local_gpus
            .load_with_prepend(props, Some(prepend.as_str()));
        self.tilt_angles_rawtlt_file = EtomoBoolean2::load_instance(
            self.tilt_angles_rawtlt_file.take(),
            TILT_ANGLES_RAWTLT_FILE_KEY,
            props,
            Some(prepend.as_str()),
        );
        self.join_table_size
            .load_with_prepend(props, Some(prepend.as_str()));
        self.peet_table_size
            .load_with_prepend(props, Some(prepend.as_str()));
        self.batch_table_size
            .load_with_prepend(props, Some(prepend.as_str()));
        // Backward compatibility
        let old_revision = self.revision_number.lock().unwrap().le(Some(
            &EtomoVersion::get_default_instance_with_version(Some("1.2")),
        ));
        if old_revision {
            // Fields where mistakenly saved with a "." in front of them - use backward
            // compatibility to fix.
            if self.parallel_processing.is_none() {
                self.parallel_processing = EtomoBoolean2::load_instance(
                    self.parallel_processing.take(),
                    PARALLEL_PROCESSING_KEY,
                    props,
                    Some("."),
                );
            }
            if self.montage.is_none() {
                self.montage = EtomoBoolean2::load_instance(
                    self.montage.take(),
                    MONTAGE_KEY,
                    props,
                    Some("."),
                );
            }
            if self.single_axis.is_none() {
                self.single_axis = EtomoBoolean2::load_instance(
                    self.single_axis.take(),
                    SINGLE_AXIS_KEY,
                    props,
                    Some("."),
                );
            }
            if self.no_parallel_processing.is_none() {
                self.no_parallel_processing = EtomoBoolean2::load_instance(
                    self.no_parallel_processing.take(),
                    NO_PARALLEL_PROCESSING_KEY,
                    props,
                    Some("."),
                );
            }
            if self.parallel_table_size.is_null() {
                self.parallel_table_size.load_with_prepend(props, Some("."));
            }
            if self.swap_y_and_z.is_none() {
                self.swap_y_and_z = EtomoBoolean2::load_instance(
                    self.swap_y_and_z.take(),
                    SWAP_Y_AND_Z_KEY,
                    props,
                    Some("."),
                );
            }
            if self.cpus.is_null() {
                self.cpus.load_with_prepend(props, Some("."));
            }
            if self.tilt_angles_rawtlt_file.is_none() {
                self.tilt_angles_rawtlt_file = EtomoBoolean2::load_instance(
                    self.tilt_angles_rawtlt_file.take(),
                    TILT_ANGLES_RAWTLT_FILE_KEY,
                    props,
                    Some("."),
                );
            }
            if self.join_table_size.is_null() {
                self.join_table_size.load_with_prepend(props, Some("."));
            }
            if self.peet_table_size.is_null() {
                self.peet_table_size.load_with_prepend(props, Some("."));
            }
            if self.batch_table_size.is_null() {
                self.batch_table_size.load_with_prepend(props, Some("."));
            }
        }
        // revisionNumber = props.getProperty(group + "RevisionNumber", "1.0");
        self.native_look_and_feel = java_lang_boolean_value_of(
            props
                .get(&(group.clone() + "NativeLookAndFeel"))
                .map(String::as_str)
                .unwrap_or("false"),
        );
        self.advanced_dialogs = java_lang_boolean_value_of(
            props
                .get(&(group.clone() + "AdvancedDialogs"))
                .map(String::as_str)
                .unwrap_or("false"),
        );
        self.compact_display = java_lang_boolean_value_of(
            props
                .get(&(group.clone() + COMPACT_DISPLAY_KEY))
                .map(String::as_str)
                .unwrap_or("false"),
        );
        if let Ok(value) = java_lang_integer_parse_int(
            props
                .get(&(group.clone() + "ToolTipsInitialDelay"))
                .map(String::as_str)
                .unwrap_or("1000"),
        ) {
            self.tool_tips_initial_delay = value;
        }
        if let Ok(value) = java_lang_integer_parse_int(
            props
                .get(&(group.clone() + "ToolTipsDismissDelay"))
                .map(String::as_str)
                .unwrap_or("30000"),
        ) {
            self.tool_tips_dismiss_delay = value;
        }
        self.font_family = Some(
            props
                .get(&(group.clone() + "FontFamily"))
                .cloned()
                .unwrap_or_else(|| "Dialog".to_string()),
        );
        if let Ok(value) = java_lang_integer_parse_int(
            props
                .get(&(group.clone() + "FontSize"))
                .cloned()
                .unwrap_or_else(|| DEFAULT_FONT_SIZE.to_string())
                .as_str(),
        ) {
            self.font_size = value;
        }

        if let Ok(value) = java_lang_integer_parse_int(
            props
                .get(&(group.clone() + "MainWindowWidth"))
                .map(String::as_str)
                .unwrap_or("800"),
        ) {
            self.main_window_width = value;
        }
        if let Ok(value) = java_lang_integer_parse_int(
            props
                .get(&(group.clone() + "MainWindowHeight"))
                .map(String::as_str)
                .unwrap_or("600"),
        ) {
            self.main_window_height = value;
        }

        if let Ok(value) = java_lang_integer_parse_int(
            props
                .get(&(group.clone() + "NMRUFiles"))
                .map(String::as_str)
                .unwrap_or("10"),
        ) {
            self.n_mru_files = value;
        }
        {
            let mut mru_file_list = CircularBuffer::new(self.n_mru_files);
            let mut i = self.n_mru_files - 1;
            while i >= 0 {
                mru_file_list.put(Some(
                    props
                        .get(&("EtomoDataFile".to_string() + &i.to_string()))
                        .cloned()
                        .unwrap_or_default(),
                ));
                i -= 1;
            }
            *self.mru_file_list.lock().unwrap() = mru_file_list;
        }
        self.auto_fit = java_lang_boolean_value_of(
            props
                .get(&(group.clone() + "AutoFit"))
                .map(String::as_str)
                .unwrap_or("false"),
        );
        // TEMP bug# 614
        self.auto_fit = true;
        self.gpu_processing
            .load_with_prepend(props, Some(prepend.as_str()));
        self.main_last_location_x
            .load_with_prepend(props, Some(prepend.as_str()));
        if self.main_last_location_x.is_null() {
            self.main_last_location_x.load_from_other_key(
                Some(props),
                Some(prepend.as_str()),
                "LastLocationX",
            );
        }
        self.main_last_location_y
            .load_with_prepend(props, Some(prepend.as_str()));
        if self.main_last_location_y.is_null() {
            self.main_last_location_y.load_from_other_key(
                Some(props),
                Some(prepend.as_str()),
                "LastLocationY",
            );
        }
        self.sub_last_location_x
            .load_with_prepend(props, Some(prepend.as_str()));
        self.sub_last_location_y
            .load_with_prepend(props, Some(prepend.as_str()));
        self.set_fei_pixel_size
            .load_with_prepend(props, Some(prepend.as_str()));
        self.user_template_dir_abs_path
            .load_with_prepend(Some(&mut props_copy), Some(prepend.as_str()));
        self.scope_template_abs_path
            .load_with_prepend(Some(&mut props_copy), Some(prepend.as_str()));
        self.system_template_abs_path
            .load_with_prepend(Some(&mut props_copy), Some(prepend.as_str()));
        self.user_template_abs_path
            .load_with_prepend(Some(&mut props_copy), Some(prepend.as_str()));
        self.email_address
            .load_with_prepend(Some(&mut props_copy), Some(prepend.as_str()));
        self.use_email_address
            .load_with_prepend(props, Some(prepend.as_str()));
        self.smtp_server
            .load_with_prepend(Some(&mut props_copy), Some(prepend.as_str()));
        self.remove_excluded_views
            .load_with_prepend(props, Some(prepend.as_str()));
        self.batch_close_datasets
            .load_with_prepend(props, Some(prepend.as_str()));
        // Can't use getProperty to find an unknown key - go through the entire collection
        if let Some(plugin_map) = &mut self.plugin_map {
            plugin_map.clear();
        }
        for (prop_key, value) in props {
            if prop_key.starts_with(&(group.clone() + PLUGIN_KEY)) {
                // The key is property name without the group
                let key = prop_key[group.len()..].to_string();
                let mut property = EtomoBoolean2::new_with_name(&key);
                property.set_string(Some(value.as_str()));
                if self.plugin_map.is_none() {
                    self.plugin_map = Some(HashMap::new());
                }
                self.plugin_map.as_mut().unwrap().insert(key, property);
            }
        }
        self.loaded = true;
    }
}
