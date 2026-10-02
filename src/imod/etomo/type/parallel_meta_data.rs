//! `IMOD/Etomo/src/etomo/type/ParallelMetaData.java`.
//!
//! The data file (.epp) of the generic parallel process and nonlinear anisotropic
//! diffusion interfaces.
//!
//! `ParallelMetaData extends BaseMetaData`: the superclass fields are the embedded
//! `BaseMetaDataBase` (see `base_meta_data.rs`), and the abstract/overridden members
//! are the `BaseMetaData` impl.  The object is shared by its manager and the dialogs, so
//! every field sits in a `Mutex` and the setters take `&self`.  Getters of
//! `ConstEtomoNumber` fields return a copy of the number.

use std::collections::BTreeMap;
use std::sync::Mutex;

use super::axis_type::AxisType;
use super::base_meta_data::{BaseMetaData, BaseMetaDataBase};
use super::const_etomo_number::{ConstEtomoNumber, Number, Type};
use super::data_file_type::DataFileType;
use super::dialog_type::DialogType;
use super::enumerated_type::EnumeratedType;
use super::etomo_boolean2::EtomoBoolean2;
use super::etomo_number::EtomoNumber;
use super::image_output_format::ImageOutputFormat;
use super::string_property::StringProperty;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::storage::storable::{Storable, StorableValue};
use crate::imod::etomo::ui::log_properties::LogProperties;
use crate::imod::etomo::util::dataset_files;
use crate::imod::etomo::util::utilities;

/// Java `NEW_GENERIC_PARALLEL_PROCESS_TITLE`.
pub const NEW_GENERIC_PARALLEL_PROCESS_TITLE: &str = "Parallel Processing";
/// Java `NEW_ANISOTROPIC_DIFFUSION_TITLE`.
pub const NEW_ANISOTROPIC_DIFFUSION_TITLE: &str = "Nonlinear Anisotropic Diffusion";
/// Java private static final `REVISION_KEY`.
const REVISION_KEY: &str = "Revision";
/// Java private static final `CURRENT_REVISION`.
const CURRENT_REVISION: &str = "1.0";
/// Java private static final `PARALLEL_GROUP_KEY`.
const PARALLEL_GROUP_KEY: &str = "Parallel";
/// Java private static final `ROOT_NAME_KEY`.
const ROOT_NAME_KEY: &str = "RootName";
/// Java package-private static final `ANISOTROPIC_DIFFUSION_GROUP_KEY`.
pub const ANISOTROPIC_DIFFUSION_GROUP_KEY: &str = "AnisotropicDiffusion";

/// Java `ParallelMetaData`.
pub struct ParallelMetaData {
    /// Java superclass `BaseMetaData` state.
    base: BaseMetaDataBase,
    /// Java private final `loadWithFlipping`.
    load_with_flipping: Mutex<EtomoBoolean2>,
    /// Java private final `volume`.
    volume: Mutex<StringProperty>,
    /// Java private final `xMin`.
    x_min: Mutex<EtomoNumber>,
    /// Java private final `xMax`.
    x_max: Mutex<EtomoNumber>,
    /// Java private final `yMin`.
    y_min: Mutex<EtomoNumber>,
    /// Java private final `yMax`.
    y_max: Mutex<EtomoNumber>,
    /// Java private final `zMin`.
    z_min: Mutex<EtomoNumber>,
    /// Java private final `zMax`.
    z_max: Mutex<EtomoNumber>,
    /// Java private final `testKValueList`.
    test_k_value_list: Mutex<StringProperty>,
    /// Java private final `testIteration`.
    test_iteration: Mutex<EtomoNumber>,
    /// Java private final `testKValue`.
    test_k_value: Mutex<EtomoNumber>,
    /// Java private final `testIterationList`.
    test_iteration_list: Mutex<StringProperty>,
    /// Java private final `kValue`.
    k_value: Mutex<EtomoNumber>,
    /// Java private final `iteration`.
    iteration: Mutex<EtomoNumber>,
    /// Java private final `memoryPerChunk`.
    memory_per_chunk: Mutex<EtomoNumber>,
    /// Java private final `overlapTimesFour`.
    overlap_times_four: Mutex<EtomoBoolean2>,
    /// Java private final `newStyleZ`.
    new_style_z: Mutex<EtomoBoolean2>,
    /// Java private final `useGpus`.
    use_gpus: Mutex<EtomoBoolean2>,
    /// Java private final `oneLineCommandProgram`.
    one_line_command_program: Mutex<StringProperty>,
    /// Java private final `oneLineCommandArguments`.
    one_line_command_arguments: Mutex<StringProperty>,
    /// Java private final `inputImageFile`.
    input_image_file: Mutex<StringProperty>,
    /// Java private final `suffixForOutputName`.
    suffix_for_output_name: Mutex<StringProperty>,
    /// Java private final `formatOfOutputFile`.
    format_of_output_file: Mutex<StringProperty>,
    /// Java private final `overlapPixels`.
    overlap_pixels: Mutex<StringProperty>,
    /// Java private final `megavoxelMaximum`.
    megavoxel_maximum: Mutex<StringProperty>,
    /// Java private `dialogType`, initialised to
    /// `DialogType.getDefault(DataFileType.PARALLEL)`.
    dialog_type: Mutex<Option<DialogType>>,
    /// Java private `revision`, initialised to null.
    revision: Mutex<Option<String>>,
    /// Java private `rootName`, initialised to null.
    root_name: Mutex<Option<String>>,
}

// Safety: every field is a `Mutex` of owned data except the `&'static` references in
// `BaseMetaDataBase`.  `manager` is `Send + Sync` (the `BaseManager` trait requires
// it).  The one member that keeps the auto traits from applying is
// `BaseMetaDataBase`'s `Option<&'static dyn LogProperties>` (the log window, an EDT
// object): `BaseMetaDataBase` never calls through it and nothing in this module reaches
// it, so sharing the reference across threads never touches the object behind it.
// (Same argument as `MetaData`.)
unsafe impl Send for ParallelMetaData {}
unsafe impl Sync for ParallelMetaData {}

impl ParallelMetaData {
    /// Java `ParallelMetaData(BaseManager, LogProperties, boolean, boolean)`, together
    /// with the field initialisers Java runs before its body.
    pub fn new(
        manager: Option<&'static dyn BaseManager>,
        log_properties: Option<&'static dyn LogProperties>,
        force_old_style: bool,
        new_dataset: bool,
    ) -> ParallelMetaData {
        let instance = ParallelMetaData {
            base: BaseMetaDataBase::new_force_old_style(
                manager,
                log_properties,
                force_old_style,
                new_dataset,
                true,
            ),
            load_with_flipping: Mutex::new(EtomoBoolean2::new_with_name("LoadWithFlipping")),
            volume: Mutex::new(StringProperty::new_with_key(Some("Volume"))),
            x_min: Mutex::new(EtomoNumber::new_with_name("XMin")),
            x_max: Mutex::new(EtomoNumber::new_with_name("XMax")),
            y_min: Mutex::new(EtomoNumber::new_with_name("YMin")),
            y_max: Mutex::new(EtomoNumber::new_with_name("YMax")),
            z_min: Mutex::new(EtomoNumber::new_with_name("ZMin")),
            z_max: Mutex::new(EtomoNumber::new_with_name("ZMax")),
            test_k_value_list: Mutex::new(StringProperty::new_with_key(Some("TestKValueList"))),
            test_iteration: Mutex::new(EtomoNumber::new_with_name("TestIteration")),
            test_k_value: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Double,
                "TestKValue",
            )),
            test_iteration_list: Mutex::new(StringProperty::new_with_key(Some(
                "TestIterationList",
            ))),
            k_value: Mutex::new(EtomoNumber::new_with_type_and_name(Type::Double, "KValue")),
            iteration: Mutex::new(EtomoNumber::new_with_name("Iteration")),
            memory_per_chunk: Mutex::new(EtomoNumber::new_with_name("MemoryPerChunk")),
            overlap_times_four: Mutex::new(EtomoBoolean2::new_with_name("OverlapTimesFour")),
            new_style_z: Mutex::new(EtomoBoolean2::new_with_name("NewStyleZ")),
            use_gpus: Mutex::new(EtomoBoolean2::new_with_name("UseGpus")),
            one_line_command_program: Mutex::new(StringProperty::new_with_key(Some(
                "OneLineCommand.Program",
            ))),
            one_line_command_arguments: Mutex::new(StringProperty::new_with_key(Some(
                "OneLineCommand.Arguments",
            ))),
            input_image_file: Mutex::new(StringProperty::new_with_key(Some("InputImageFile"))),
            suffix_for_output_name: Mutex::new(StringProperty::new_with_key(Some(
                "SuffixForOutputName",
            ))),
            format_of_output_file: Mutex::new(StringProperty::new_with_key(Some(
                "FormatOfOutputFile",
            ))),
            overlap_pixels: Mutex::new(StringProperty::new_with_key(Some("OverlapPixels"))),
            megavoxel_maximum: Mutex::new(StringProperty::new_with_key(Some("MegavoxelMaximum"))),
            dialog_type: Mutex::new(DialogType::get_default(Some(DataFileType::Parallel))),
            revision: Mutex::new(None),
            root_name: Mutex::new(None),
        };
        *instance.base.axis_type.lock().unwrap() = AxisType::SingleAxis;
        *instance.base.file_extension.lock().unwrap() = DataFileType::Parallel
            .extension()
            .map(|extension| extension.to_string());
        instance
    }

    /// Java package-private `paramString()`.
    pub fn param_string(&self) -> String {
        format!(
            "revision={},rootName={}",
            self.revision.lock().unwrap().as_deref().unwrap_or("null"),
            self.root_name.lock().unwrap().as_deref().unwrap_or("null")
        )
    }

    /// Java `validate()`.  Returns null if valid, otherwise the error message.
    pub fn validate(&self) -> Option<String> {
        if self.root_name.lock().unwrap().is_none() {
            return Some("Missing root name.".to_string());
        }
        None
    }

    /// Java `setRootName(String)`.
    pub fn set_root_name(&self, root_name: Option<&str>) {
        *self.root_name.lock().unwrap() = root_name.map(|root_name| root_name.to_string());
        let root_name = self.root_name.lock().unwrap().clone();
        utilities::manager_stamp(None, root_name.as_deref());
    }

    /// Java `setLoadWithFlipping(boolean)`.
    pub fn set_load_with_flipping(&self, input: bool) {
        self.load_with_flipping.lock().unwrap().set_boolean(input);
    }

    /// Java `isLoadWithFlipping()`.
    pub fn is_load_with_flipping(&self) -> bool {
        self.load_with_flipping.lock().unwrap().is()
    }

    /// Java `getRootName()`.
    pub fn get_root_name(&self) -> Option<String> {
        self.root_name.lock().unwrap().clone()
    }

    /// Java `setVolume(String)`.
    pub fn set_volume(&self, input: Option<&str>) {
        self.volume.lock().unwrap().set(input);
    }

    /// Java `getVolume()`.
    pub fn get_volume(&self) -> Option<String> {
        self.volume.lock().unwrap().to_string_option()
    }

    /// Java `setXMin(String)`.
    pub fn set_x_min(&self, input: Option<&str>) {
        self.x_min.lock().unwrap().set_string(input);
    }

    /// Java `getXMin()`.
    pub fn get_x_min(&self) -> Option<String> {
        Some(self.x_min.lock().unwrap().to_string())
    }

    /// Java `setXMax(String)`.
    pub fn set_x_max(&self, input: Option<&str>) {
        self.x_max.lock().unwrap().set_string(input);
    }

    /// Java `getXMax()`.
    pub fn get_x_max(&self) -> Option<String> {
        Some(self.x_max.lock().unwrap().to_string())
    }

    /// Java `setYMin(String)`.
    pub fn set_y_min(&self, input: Option<&str>) {
        self.y_min.lock().unwrap().set_string(input);
    }

    /// Java `getYMin()`.
    pub fn get_y_min(&self) -> Option<String> {
        Some(self.y_min.lock().unwrap().to_string())
    }

    /// Java `setYMax(String)`.
    pub fn set_y_max(&self, input: Option<&str>) {
        self.y_max.lock().unwrap().set_string(input);
    }

    /// Java `getYMax()`.
    pub fn get_y_max(&self) -> Option<String> {
        Some(self.y_max.lock().unwrap().to_string())
    }

    /// Java `setZMin(String)`.
    pub fn set_z_min(&self, input: Option<&str>) {
        self.z_min.lock().unwrap().set_string(input);
    }

    /// Java `getZMin()`.
    pub fn get_z_min(&self) -> Option<String> {
        Some(self.z_min.lock().unwrap().to_string())
    }

    /// Java `setZMax(String)`.
    pub fn set_z_max(&self, input: Option<&str>) {
        self.z_max.lock().unwrap().set_string(input);
    }

    /// Java `getZMax()`.
    pub fn get_z_max(&self) -> Option<String> {
        Some(self.z_max.lock().unwrap().to_string())
    }

    /// Java `setTestKValueList(String)`.
    pub fn set_test_k_value_list(&self, input: Option<&str>) {
        self.test_k_value_list.lock().unwrap().set(input);
    }

    /// Java `getTestKValueList()`.
    pub fn get_test_k_value_list(&self) -> Option<String> {
        self.test_k_value_list.lock().unwrap().to_string_option()
    }

    /// Java `setKValue(String)`.
    pub fn set_k_value(&self, input: Option<&str>) {
        self.k_value.lock().unwrap().set_string(input);
    }

    /// Java `getKValue()`.
    pub fn get_k_value(&self) -> Option<String> {
        Some(self.k_value.lock().unwrap().to_string())
    }

    /// Java `setIteration(Number)`.
    pub fn set_iteration(&self, input: Option<Number>) {
        self.iteration.lock().unwrap().set_number(input);
    }

    /// Java `getIteration()`.
    pub fn get_iteration(&self) -> ConstEtomoNumber {
        self.iteration.lock().unwrap().base.clone()
    }

    /// Java `setTestIteration(Number)`.
    pub fn set_test_iteration(&self, input: Option<Number>) {
        self.test_iteration.lock().unwrap().set_number(input);
    }

    /// Java `getTestIteration()`.
    pub fn get_test_iteration(&self) -> ConstEtomoNumber {
        self.test_iteration.lock().unwrap().base.clone()
    }

    /// Java `setMemoryPerChunk(Number)`.
    pub fn set_memory_per_chunk(&self, input: Option<Number>) {
        self.memory_per_chunk.lock().unwrap().set_number(input);
    }

    /// Java `getMemoryPerChunk()`.
    pub fn get_memory_per_chunk(&self) -> ConstEtomoNumber {
        self.memory_per_chunk.lock().unwrap().base.clone()
    }

    /// Java `setOverlapTimesFour(boolean)`.
    pub fn set_overlap_times_four(&self, input: bool) {
        self.overlap_times_four.lock().unwrap().set_boolean(input);
    }

    /// Java `setNewStyleZ(String, String)`.
    pub fn set_new_style_z(&self, ui_z_min: Option<&str>, ui_z_max: Option<&str>) {
        let mut new_style_z = self.new_style_z.lock().unwrap();
        if new_style_z.is_null() || !new_style_z.is() {
            let value = !self.z_min.lock().unwrap().equals_string(ui_z_min)
                || !self.z_max.lock().unwrap().equals_string(ui_z_max);
            new_style_z.set_boolean(value);
        }
    }

    /// Java `isOverlapTimesFour()`.
    pub fn is_overlap_times_four(&self) -> bool {
        self.overlap_times_four.lock().unwrap().is()
    }

    /// Java `isNewStyleZ()`.
    pub fn is_new_style_z(&self) -> bool {
        self.new_style_z.lock().unwrap().is()
    }

    /// Java `setTestKValue(String)`.
    pub fn set_test_k_value(&self, input: Option<&str>) {
        self.test_k_value.lock().unwrap().set_string(input);
    }

    /// Java `getTestKValue()`.
    pub fn get_test_k_value(&self) -> Option<String> {
        Some(self.test_k_value.lock().unwrap().to_string())
    }

    /// Java `setTestIterationList(String)`.
    pub fn set_test_iteration_list(&self, input: Option<&str>) {
        self.test_iteration_list.lock().unwrap().set(input);
    }

    /// Java `getTestIterationList()`.
    pub fn get_test_iteration_list(&self) -> Option<String> {
        self.test_iteration_list.lock().unwrap().to_string_option()
    }

    /// Java `setDialogType(DialogType)`.
    pub fn set_dialog_type(&self, input: Option<DialogType>) {
        *self.dialog_type.lock().unwrap() = input;
    }

    /// Java `setUseGpus(boolean)`.
    pub fn set_use_gpus(&self, input: bool) {
        self.use_gpus.lock().unwrap().set_boolean(input);
    }

    /// Java `setOneLineCommandProgram(String)`.
    pub fn set_one_line_command_program(&self, input: Option<&str>) {
        self.one_line_command_program.lock().unwrap().set(input);
    }

    /// Java `setOneLineCommandArguments(String)`.
    pub fn set_one_line_command_arguments(&self, input: Option<&str>) {
        self.one_line_command_arguments.lock().unwrap().set(input);
    }

    /// Java `setInputImageFile(String)`.
    pub fn set_input_image_file(&self, input: Option<&str>) {
        self.input_image_file.lock().unwrap().set(input);
    }

    /// Java `setSuffixForOutputName(String)`.
    pub fn set_suffix_for_output_name(&self, input: Option<&str>) {
        self.suffix_for_output_name.lock().unwrap().set(input);
    }

    /// Java `setFormatOfOutputFile(EnumeratedType)`.
    pub fn set_format_of_output_file(&self, input: Option<&dyn EnumeratedType>) {
        match input {
            None => {
                self.format_of_output_file.lock().unwrap().reset();
            }
            Some(input) => {
                let label = input.get_label();
                self.format_of_output_file
                    .lock()
                    .unwrap()
                    .set(label.as_deref());
            }
        }
    }

    /// Java `setOverlapPixels(Number)`: `StringProperty.set(Number)` stores the
    /// number's `toString()`.
    pub fn set_overlap_pixels(&self, input: Option<Number>) {
        let input = input.map(|input| input.to_string());
        self.overlap_pixels
            .lock()
            .unwrap()
            .set_number(input.as_deref());
    }

    /// Java `setMegavoxelMaximum(Number)`: `StringProperty.set(Number)` stores the
    /// number's `toString()`.
    pub fn set_megavoxel_maximum(&self, input: Option<Number>) {
        let input = input.map(|input| input.to_string());
        self.megavoxel_maximum
            .lock()
            .unwrap()
            .set_number(input.as_deref());
    }

    /// Java `isUseGpus()`.
    pub fn is_use_gpus(&self) -> bool {
        self.use_gpus.lock().unwrap().is()
    }

    /// Java `getOneLineCommandProgram()`.
    pub fn get_one_line_command_program(&self) -> Option<String> {
        self.one_line_command_program
            .lock()
            .unwrap()
            .to_string_option()
    }

    /// Java `getOneLineCommandArguments()`.
    pub fn get_one_line_command_arguments(&self) -> Option<String> {
        self.one_line_command_arguments
            .lock()
            .unwrap()
            .to_string_option()
    }

    /// Java `getInputImageFile()`.
    pub fn get_input_image_file(&self) -> Option<String> {
        self.input_image_file.lock().unwrap().to_string_option()
    }

    /// Java `getSuffixForOutputName()`.
    pub fn get_suffix_for_output_name(&self) -> Option<String> {
        self.suffix_for_output_name
            .lock()
            .unwrap()
            .to_string_option()
    }

    /// Java `getFormatOfOutputFile()`.
    pub fn get_format_of_output_file(&self) -> ImageOutputFormat {
        let value = self
            .format_of_output_file
            .lock()
            .unwrap()
            .to_string_option();
        ImageOutputFormat::get_instance(value.as_deref())
    }

    /// Java `getOverlapPixels()`.
    pub fn get_overlap_pixels(&self) -> Option<String> {
        self.overlap_pixels.lock().unwrap().to_string_option()
    }

    /// Java `getMegavoxelMaximum()`.
    pub fn get_megavoxel_maximum(&self) -> Option<String> {
        self.megavoxel_maximum.lock().unwrap().to_string_option()
    }

    /// Java `getDialogType()`.
    pub fn get_dialog_type(&self) -> Option<DialogType> {
        *self.dialog_type.lock().unwrap()
    }

    /// Java `load(Properties)`.
    pub fn load(&self, props: &BTreeMap<String, String>) {
        self.load_with_prepend(props, "");
    }

    /// Java `load(Properties, String)`.
    pub fn load_with_prepend(&self, props: &BTreeMap<String, String>, prepend: &str) {
        // `super.load(props, prepend)`.
        if self
            .base
            .load_with_created_prepend(props, self.create_prepend(prepend).as_deref())
        {
            self.check_image_filename_style_loaded(prepend);
        }
        // reset
        *self.revision.lock().unwrap() = None;
        *self.root_name.lock().unwrap() = None;
        self.load_with_flipping.lock().unwrap().reset();
        self.volume.lock().unwrap().reset();
        self.x_min.lock().unwrap().reset();
        self.x_max.lock().unwrap().reset();
        self.y_min.lock().unwrap().reset();
        self.y_max.lock().unwrap().reset();
        self.z_min.lock().unwrap().reset();
        self.z_max.lock().unwrap().reset();
        self.test_k_value_list.lock().unwrap().reset();
        self.test_iteration.lock().unwrap().reset();
        *self.dialog_type.lock().unwrap() = DialogType::get_default(Some(DataFileType::Parallel));
        self.test_k_value.lock().unwrap().reset();
        self.test_iteration_list.lock().unwrap().reset();
        self.k_value.lock().unwrap().reset();
        self.iteration.lock().unwrap().reset();
        self.memory_per_chunk.lock().unwrap().reset();
        self.overlap_times_four.lock().unwrap().reset();
        self.new_style_z.lock().unwrap().reset();
        self.use_gpus.lock().unwrap().reset();
        self.one_line_command_program.lock().unwrap().reset();
        self.one_line_command_arguments.lock().unwrap().reset();
        self.input_image_file.lock().unwrap().reset();
        self.suffix_for_output_name.lock().unwrap().reset();
        self.format_of_output_file.lock().unwrap().reset();
        self.overlap_pixels.lock().unwrap().reset();
        self.megavoxel_maximum.lock().unwrap().reset();
        // load
        *self.dialog_type.lock().unwrap() =
            DialogType::load_with_data_file_type(Some(DataFileType::Parallel), props);
        let prepend = self.create_prepend(prepend).unwrap_or("null".to_string());
        let group = format!("{}.", prepend);
        *self.revision.lock().unwrap() = Some(
            props
                .get(&format!("{}{}", group, REVISION_KEY))
                .cloned()
                .unwrap_or(CURRENT_REVISION.to_string()),
        );
        *self.root_name.lock().unwrap() =
            props.get(&format!("{}{}", group, ROOT_NAME_KEY)).cloned();
        // `StringProperty.load` may remove a backward-compatible key from the Java
        // `Properties`; this translation's `props` is read-only, so the string
        // properties load from a copy.  None of this class's string properties declares
        // such a key.
        let mut props_copy = props.clone();
        let prepend = Some(prepend.as_str());
        self.load_with_flipping
            .lock()
            .unwrap()
            .load_with_prepend(props, prepend);
        self.volume
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), prepend);
        self.x_min.lock().unwrap().load_with_prepend(props, prepend);
        self.x_max.lock().unwrap().load_with_prepend(props, prepend);
        self.y_min.lock().unwrap().load_with_prepend(props, prepend);
        self.y_max.lock().unwrap().load_with_prepend(props, prepend);
        self.z_min.lock().unwrap().load_with_prepend(props, prepend);
        self.z_max.lock().unwrap().load_with_prepend(props, prepend);
        self.test_k_value_list
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), prepend);
        self.test_iteration
            .lock()
            .unwrap()
            .load_with_prepend(props, prepend);
        self.test_k_value
            .lock()
            .unwrap()
            .load_with_prepend(props, prepend);
        self.test_iteration_list
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), prepend);
        self.k_value
            .lock()
            .unwrap()
            .load_with_prepend(props, prepend);
        self.iteration
            .lock()
            .unwrap()
            .load_with_prepend(props, prepend);
        self.memory_per_chunk.lock().unwrap().load_with_default_int(
            props,
            prepend,
            // Java `AnisotropicDiffusionDialog.MEMORY_PER_CHUNK_DEFAULT`, which is
            // `FilterFullVolumePanel.MEMORY_PER_CHUNK_DEFAULT` =
            // `14 * ChunksetupParam.MEMORY_TO_VOXEL`.
            // TODO(unit): read it from the faithful FilterFullVolumePanel.
            14 * crate::imod::etomo::comscript::chunksetup_param::MEMORY_TO_VOXEL,
        );
        self.overlap_times_four
            .lock()
            .unwrap()
            .load_with_prepend(props, prepend);
        self.new_style_z
            .lock()
            .unwrap()
            .load_with_prepend(props, prepend);
        self.use_gpus
            .lock()
            .unwrap()
            .load_with_prepend(props, prepend);
        self.one_line_command_program
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), prepend);
        self.one_line_command_arguments
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), prepend);
        self.input_image_file
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), prepend);
        self.suffix_for_output_name
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), prepend);
        self.format_of_output_file
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), prepend);
        self.overlap_pixels
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), prepend);
        self.megavoxel_maximum
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), prepend);
    }

    /// Java `store(Properties, String)`.
    ///
    /// Upstream bugs fixed (ParallelMetaData.java:423, 425):
    /// - `props.setProperty(group + ROOT_NAME_KEY, rootName)` with a null rootName (a
    ///   dataset stored before its root name is set) throws a NullPointerException from
    ///   `Hashtable.put`.  A null root name is stored as an absent key: the key is
    ///   removed.
    /// - `dialogType.store(props)` throws a NullPointerException when the loaded
    ///   `DialogType` property named no known dialog type (`DialogType.load` returned
    ///   null).  A null dialog type stores nothing.
    pub fn store_with_prepend(&self, props: &mut BTreeMap<String, String>, prepend: &str) {
        // `super.store(props, prepend)`.
        self.base
            .store_with_created_prepend(props, self.create_prepend(prepend).as_deref());
        let prepend = self.create_prepend(prepend).unwrap_or("null".to_string());
        let group = format!("{}.", prepend);
        if let Some(dialog_type) = *self.dialog_type.lock().unwrap() {
            dialog_type.store(props);
        }
        props.insert(
            format!("{}{}", group, REVISION_KEY),
            CURRENT_REVISION.to_string(),
        );
        match self.root_name.lock().unwrap().clone() {
            Some(root_name) => {
                props.insert(format!("{}{}", group, ROOT_NAME_KEY), root_name);
            }
            None => {
                props.remove(&format!("{}{}", group, ROOT_NAME_KEY));
            }
        }
        let prepend = Some(prepend.as_str());
        self.load_with_flipping
            .lock()
            .unwrap()
            .store_with_prepend(props, prepend);
        self.volume
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), prepend);
        self.x_min
            .lock()
            .unwrap()
            .base.store_with_prepend(props, prepend);
        self.x_max
            .lock()
            .unwrap()
            .base.store_with_prepend(props, prepend);
        self.y_min
            .lock()
            .unwrap()
            .base.store_with_prepend(props, prepend);
        self.y_max
            .lock()
            .unwrap()
            .base.store_with_prepend(props, prepend);
        self.z_min
            .lock()
            .unwrap()
            .base.store_with_prepend(props, prepend);
        self.z_max
            .lock()
            .unwrap()
            .base.store_with_prepend(props, prepend);
        self.test_k_value_list
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), prepend);
        self.test_iteration
            .lock()
            .unwrap()
            .base.store_with_prepend(props, prepend);
        self.test_k_value
            .lock()
            .unwrap()
            .base.store_with_prepend(props, prepend);
        self.test_iteration_list
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), prepend);
        self.k_value
            .lock()
            .unwrap()
            .base.store_with_prepend(props, prepend);
        self.iteration
            .lock()
            .unwrap()
            .base.store_with_prepend(props, prepend);
        self.memory_per_chunk
            .lock()
            .unwrap()
            .base.store_with_prepend(props, prepend);
        self.overlap_times_four
            .lock()
            .unwrap()
            .store_with_prepend(props, prepend);
        self.new_style_z
            .lock()
            .unwrap()
            .store_with_prepend(props, prepend);
        self.use_gpus
            .lock()
            .unwrap()
            .store_with_prepend(props, prepend);
        self.one_line_command_program
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), prepend);
        self.one_line_command_arguments
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), prepend);
        self.input_image_file
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), prepend);
        self.suffix_for_output_name
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), prepend);
        self.format_of_output_file
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), prepend);
        self.overlap_pixels
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), prepend);
        self.megavoxel_maximum
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), prepend);
    }
}

/// Java `toString()`.
impl std::fmt::Display for ParallelMetaData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "etomo.type.ParallelMetaData[{}]\n", self.param_string())
    }
}

/// Java `Storable`, implemented through `BaseMetaData`.  `store(Properties)` is
/// `BaseMetaData`'s, which stores with an empty prepend.
impl Storable for ParallelMetaData {
    fn store(&self, properties: &mut BTreeMap<String, String>) {
        ParallelMetaData::store_with_prepend(self, properties, "");
    }

    fn store_with_prepend(&self, properties: &mut BTreeMap<String, String>, prepend: &str) {
        ParallelMetaData::store_with_prepend(self, properties, prepend);
    }

    fn load(&self, properties: &BTreeMap<String, String>) {
        ParallelMetaData::load(self, properties);
    }

    fn load_with_prepend(&self, properties: &BTreeMap<String, String>, prepend: &str) {
        ParallelMetaData::load_with_prepend(self, properties, prepend);
    }
}

impl BaseMetaData for ParallelMetaData {
    fn base(&self) -> &BaseMetaDataBase {
        &self.base
    }

    /// Java `getName()`.
    fn get_name(&self) -> Option<String> {
        if self.root_name.lock().unwrap().is_none() {
            if *self.dialog_type.lock().unwrap() == Some(DialogType::AnisotropicDiffusion) {
                return Some(NEW_ANISOTROPIC_DIFFUSION_TITLE.to_string());
            }
            return Some(NEW_GENERIC_PARALLEL_PROCESS_TITLE.to_string());
        }
        self.root_name.lock().unwrap().clone()
    }

    /// Java `getDatasetName()`.
    fn get_dataset_name(&self) -> Option<String> {
        self.root_name.lock().unwrap().clone()
    }

    /// Java `getMetaDataFileName()`.
    fn get_meta_data_file_name(&self) -> Option<String> {
        let root_name = self.root_name.lock().unwrap().clone()?;
        Some(dataset_files::get_parallel_data_file_name(&root_name))
    }

    /// Java `isValid()`.
    fn is_valid(&self) -> bool {
        self.validate().is_none()
    }

    /// Java package-private `getGroupKey()`.
    fn get_group_key(&self) -> Option<String> {
        if *self.dialog_type.lock().unwrap() == Some(DialogType::AnisotropicDiffusion) {
            return Some(ANISOTROPIC_DIFFUSION_GROUP_KEY.to_string());
        }
        Some(PARALLEL_GROUP_KEY.to_string())
    }
}
