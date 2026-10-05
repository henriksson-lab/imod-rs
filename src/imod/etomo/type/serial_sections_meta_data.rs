//! `IMOD/Etomo/src/etomo/type/SerialSectionsMetaData.java`.
//!
//! The Serial Sections interface's data file (`.ess`) contents: the stack, view type,
//! distortion field, binning, the auto-alignment parameters and the dialog's saved
//! settings.
//!
//! **Representation.**  `SerialSectionsMetaData extends BaseMetaData implements
//! ConstSerialSectionsMetaData`: the superclass state is `base` (`BaseMetaDataBase`),
//! the abstract methods are the `BaseMetaData` trait, and the interface is
//! `ConstSerialSectionsMetaData`.  Owned by `SerialSectionsManager`, read on the event
//! dispatch thread and stored from process threads, so - like `JoinMetaData` - each
//! mutable field carries its own lock and every method takes `&self`.

use std::collections::BTreeMap;
use std::sync::{LazyLock, Mutex};

use super::auto_alignment_meta_data::AutoAlignmentMetaData;
use super::axis_type::AxisType;
use super::base_meta_data::{self, BaseMetaData, BaseMetaDataBase};
use super::const_etomo_number::{ConstEtomoNumber, Number, Type};
use super::const_serial_sections_meta_data::ConstSerialSectionsMetaData;
use super::const_string_property::ConstStringProperty;
use super::data_file_type::DataFileType;
use super::etomo_boolean2::EtomoBoolean2;
use super::etomo_number::EtomoNumber;
use super::etomo_version::EtomoVersion;
use super::string_property::StringProperty;
use super::view_type::ViewType;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::logic::serial_sections_startup_data::SerialSectionsStartupData;
use crate::imod::etomo::storage::storable::{self, Storable};
use crate::imod::etomo::ui::log_properties::LogProperties;
use crate::imod::etomo::util::utilities;

/// Java `rcsid`.
pub const RCSID: &str = "$Id:$";

/// Java `public static final String NEW_TITLE`.
pub const NEW_TITLE: &str = "Serial Sections";

/// Java private static final `CURRENT_VERSION =
/// EtomoVersion.getInstance(BaseMetaData.revisionNumberString, "1.0")`.
static CURRENT_VERSION: LazyLock<EtomoVersion> = LazyLock::new(|| {
    EtomoVersion::get_instance(Some(base_meta_data::REVISION_NUMBER_STRING), Some("1.0"))
});

/// Java `public final class SerialSectionsMetaData extends BaseMetaData implements
/// ConstSerialSectionsMetaData`.
pub struct SerialSectionsMetaData {
    /// Java superclass `BaseMetaData` state.
    base: BaseMetaDataBase,
    /// Java private final `rootName`.
    root_name: Mutex<StringProperty>,
    /// Java private final `autoAlignmentMetaData`.
    auto_alignment_meta_data: Mutex<AutoAlignmentMetaData>,
    /// Java private final `stack`.
    stack: Mutex<StringProperty>,
    /// Java private final `viewType`.
    view_type: Mutex<StringProperty>,
    /// Java private final `extractPlmdocMetadataFile`.
    extract_plmdoc_metadata_file: Mutex<EtomoBoolean2>,
    /// Java private final `distortionField`.
    distortion_field: Mutex<StringProperty>,
    /// Java private final `imagesAreBinned`.
    images_are_binned: Mutex<EtomoNumber>,
    /// Java private final `robustFitCriterion`.
    robust_fit_criterion: Mutex<EtomoNumber>,
    /// Java private final `midasBinning`.
    midas_binning: Mutex<EtomoNumber>,
    /// Java private `useReferenceSection`.
    use_reference_section: Mutex<EtomoBoolean2>,
    /// Java private final `referenceSection`.
    reference_section: Mutex<EtomoNumber>,
    /// Java private final `hybridFitsTranslations`.
    hybrid_fits_translations: Mutex<EtomoBoolean2>,
    /// Java private final `hybridFitsTranslationsRotations`.
    hybrid_fits_translations_rotations: Mutex<EtomoBoolean2>,
    /// Java private final `noOptions`.
    no_options: Mutex<EtomoBoolean2>,
    /// Java private final `numberToFitGlobalAlignment`.
    number_to_fit_global_alignment: Mutex<EtomoBoolean2>,
    /// Java private final `shiftX`.
    shift_x: Mutex<EtomoNumber>,
    /// Java private final `shiftY`.
    shift_y: Mutex<EtomoNumber>,
    /// Java private final `sizeX`.
    size_x: Mutex<EtomoNumber>,
    /// Java private final `sizeY`.
    size_y: Mutex<EtomoNumber>,
    /// Java private final `tab`.
    tab: Mutex<EtomoNumber>,
    /// Java private final `preblendVerySloppyMontage`.
    preblend_very_sloppy_montage: Mutex<EtomoBoolean2>,
    /// Java private final `otherSumGradientFile`.
    other_sum_gradient_file: Mutex<StringProperty>,
    /// Java private final `boolPreblendWeightForExpectedShifts`.
    bool_preblend_weight_for_expected_shifts: Mutex<EtomoBoolean2>,
    /// Java private final `strPreblendWeightForExpectedShifts`.
    str_preblend_weight_for_expected_shifts: Mutex<StringProperty>,
    /// Java private final `preblendEMGridMapFilter`.
    preblend_e_m_grid_map_filter: Mutex<EtomoBoolean2>,
    /// Java private final `preblendHighFrequencyFilterCutoff`.
    preblend_high_frequency_filter_cutoff: Mutex<StringProperty>,
}

// Safety: every field is a `Mutex` of owned data except `BaseMetaDataBase`'s
// `&'static` references (the `Send + Sync` manager and the EDT-only log window, which
// `BaseMetaDataBase` only calls on the event dispatch thread); same argument as
// `join_meta_data.rs`.
unsafe impl Send for SerialSectionsMetaData {}
unsafe impl Sync for SerialSectionsMetaData {}

impl SerialSectionsMetaData {
    /// Java `SerialSectionsMetaData(BaseManager, LogProperties, boolean)`.
    pub fn new(
        manager: &'static dyn BaseManager,
        log_properties: Option<&'static dyn LogProperties>,
        new_dataset: bool,
    ) -> SerialSectionsMetaData {
        // super(manager, logProperties, false, newDataset, true)
        let base = BaseMetaDataBase::new_force_old_style(
            Some(manager),
            log_properties,
            false,
            new_dataset,
            true,
        );
        *base.file_extension.lock().unwrap() =
            DataFileType::SerialSections.extension().map(str::to_owned);
        *base.axis_type.lock().unwrap() = AxisType::SingleAxis;
        let mut no_options = EtomoBoolean2::new_with_name("NoOptions");
        no_options.set_display_value_boolean(true);
        let mut robust_fit_criterion =
            EtomoNumber::new_with_type_and_name(Type::Double, "RobustFitCriterion");
        robust_fit_criterion.set_display_value_int(1);
        SerialSectionsMetaData {
            base,
            root_name: Mutex::new(StringProperty::new_with_key(Some("RootName"))),
            auto_alignment_meta_data: Mutex::new(AutoAlignmentMetaData::new()),
            stack: Mutex::new(StringProperty::new_with_key(Some("Stack"))),
            view_type: Mutex::new(StringProperty::new_with_key(Some("ViewType"))),
            extract_plmdoc_metadata_file: Mutex::new(EtomoBoolean2::new_with_name(
                "MdocMetadataFile",
            )),
            distortion_field: Mutex::new(StringProperty::new_with_key(Some("DistortionField"))),
            images_are_binned: Mutex::new(EtomoNumber::new_with_name("ImagesAreBinned")),
            robust_fit_criterion: Mutex::new(robust_fit_criterion),
            midas_binning: Mutex::new(EtomoNumber::new_with_name("MidasBinning")),
            use_reference_section: Mutex::new(EtomoBoolean2::new_with_name("UseReferenceSection")),
            reference_section: Mutex::new(EtomoNumber::new_with_name("ReferenceSection")),
            hybrid_fits_translations: Mutex::new(EtomoBoolean2::new_with_name(
                "HybridFitsTranslations",
            )),
            hybrid_fits_translations_rotations: Mutex::new(EtomoBoolean2::new_with_name(
                "HybridFitsTranslationsRotations",
            )),
            no_options: Mutex::new(no_options),
            number_to_fit_global_alignment: Mutex::new(EtomoBoolean2::new_with_name(
                "NumberToFitGlobalAlignment",
            )),
            shift_x: Mutex::new(EtomoNumber::new_with_type_and_name(Type::Double, "ShiftX")),
            shift_y: Mutex::new(EtomoNumber::new_with_type_and_name(Type::Double, "ShiftY")),
            size_x: Mutex::new(EtomoNumber::new_with_name("SizeX")),
            size_y: Mutex::new(EtomoNumber::new_with_name("SizeY")),
            tab: Mutex::new(EtomoNumber::new_with_name("Tab")),
            preblend_very_sloppy_montage: Mutex::new(EtomoBoolean2::new_with_name(
                "PreblendVerySloppyMontage",
            )),
            other_sum_gradient_file: Mutex::new(StringProperty::new_with_key(Some(
                "OtherSumGradientFile",
            ))),
            bool_preblend_weight_for_expected_shifts: Mutex::new(EtomoBoolean2::new_with_name(
                "BoolPreblendWeightForExpectedShifts",
            )),
            str_preblend_weight_for_expected_shifts: Mutex::new(StringProperty::new_with_key(
                Some("StrPreblendWeightForExpectedShifts"),
            )),
            preblend_e_m_grid_map_filter: Mutex::new(EtomoBoolean2::new_with_name(
                "PreblendEMGridMapFilter",
            )),
            preblend_high_frequency_filter_cutoff: Mutex::new(StringProperty::new_with_key(Some(
                "PreblendHighFrequencyFilterCutoff",
            ))),
        }
    }

    /// Java `setName(String)`.
    pub fn set_name(&self, root_name: Option<&str>) {
        self.root_name.lock().unwrap().set(root_name);
    }

    /// Java `rootName.toString()` (never null: the property does not return null
    /// when empty).
    fn root_name_string(&self) -> String {
        self.root_name
            .lock()
            .unwrap()
            .to_string_option()
            .unwrap_or_default()
    }

    /// Java `load(Properties, String)`.
    pub fn load_with_prepend(&self, props: &mut BTreeMap<String, String>, prepend: &str) {
        // super.load(props, prepend)
        let created = self.create_prepend(prepend);
        if self
            .base
            .load_with_created_prepend(props, created.as_deref())
        {
            self.check_image_filename_style_loaded(prepend);
        }
        // reset
        self.root_name.lock().unwrap().reset();
        self.stack.lock().unwrap().reset();
        self.view_type.lock().unwrap().reset();
        self.extract_plmdoc_metadata_file.lock().unwrap().reset();
        self.distortion_field.lock().unwrap().reset();
        self.images_are_binned.lock().unwrap().reset();
        self.robust_fit_criterion.lock().unwrap().reset();
        self.midas_binning.lock().unwrap().reset();
        self.use_reference_section.lock().unwrap().reset();
        self.reference_section.lock().unwrap().reset();
        self.hybrid_fits_translations.lock().unwrap().reset();
        self.hybrid_fits_translations_rotations
            .lock()
            .unwrap()
            .reset();
        self.no_options.lock().unwrap().reset();
        self.number_to_fit_global_alignment.lock().unwrap().reset();
        self.shift_x.lock().unwrap().reset();
        self.shift_y.lock().unwrap().reset();
        self.size_x.lock().unwrap().reset();
        self.size_y.lock().unwrap().reset();
        self.tab.lock().unwrap().reset();
        self.preblend_very_sloppy_montage.lock().unwrap().reset();
        self.bool_preblend_weight_for_expected_shifts
            .lock()
            .unwrap()
            .reset();
        self.str_preblend_weight_for_expected_shifts
            .lock()
            .unwrap()
            .reset();
        self.preblend_e_m_grid_map_filter.lock().unwrap().reset();
        self.preblend_high_frequency_filter_cutoff
            .lock()
            .unwrap()
            .reset();
        self.other_sum_gradient_file.lock().unwrap().reset();
        // load
        let prepend = self
            .create_prepend(prepend)
            .unwrap_or_else(|| "null".to_string());
        let prepend_some = Some(prepend.as_str());
        // `StringProperty.load` may remove a backward-compatible key from the Java
        // `Properties`; none of these declares one, so they load from a copy.
        let mut props_copy = props.clone();
        self.auto_alignment_meta_data
            .lock()
            .unwrap()
            .load(props, &prepend);
        self.root_name
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), prepend_some);
        self.stack
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), prepend_some);
        self.view_type
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), prepend_some);
        self.extract_plmdoc_metadata_file
            .lock()
            .unwrap()
            .load_with_prepend(props, prepend_some);
        self.distortion_field
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), prepend_some);
        self.images_are_binned
            .lock()
            .unwrap()
            .load_with_prepend(props, prepend_some);
        self.robust_fit_criterion
            .lock()
            .unwrap()
            .load_with_prepend(props, prepend_some);
        self.midas_binning
            .lock()
            .unwrap()
            .load_with_prepend(props, prepend_some);
        self.use_reference_section
            .lock()
            .unwrap()
            .load_with_prepend(props, prepend_some);
        self.reference_section
            .lock()
            .unwrap()
            .load_with_prepend(props, prepend_some);
        self.hybrid_fits_translations
            .lock()
            .unwrap()
            .load_with_prepend(props, prepend_some);
        self.hybrid_fits_translations_rotations
            .lock()
            .unwrap()
            .load_with_prepend(props, prepend_some);
        self.no_options
            .lock()
            .unwrap()
            .load_with_prepend(props, prepend_some);
        self.number_to_fit_global_alignment
            .lock()
            .unwrap()
            .load_with_prepend(props, prepend_some);
        self.shift_x
            .lock()
            .unwrap()
            .load_with_prepend(props, prepend_some);
        self.shift_y
            .lock()
            .unwrap()
            .load_with_prepend(props, prepend_some);
        self.size_x
            .lock()
            .unwrap()
            .load_with_prepend(props, prepend_some);
        self.size_y
            .lock()
            .unwrap()
            .load_with_prepend(props, prepend_some);
        self.tab
            .lock()
            .unwrap()
            .load_with_prepend(props, prepend_some);
        self.preblend_very_sloppy_montage
            .lock()
            .unwrap()
            .load_with_prepend(props, prepend_some);
        self.bool_preblend_weight_for_expected_shifts
            .lock()
            .unwrap()
            .load_with_prepend(props, prepend_some);
        self.str_preblend_weight_for_expected_shifts
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), prepend_some);
        self.preblend_e_m_grid_map_filter
            .lock()
            .unwrap()
            .load_with_prepend(props, prepend_some);
        self.preblend_high_frequency_filter_cutoff
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), prepend_some);
        self.other_sum_gradient_file
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), prepend_some);
    }

    /// Java `store(Properties, String)`.
    pub fn store_with_prepend(&self, props: &mut BTreeMap<String, String>, prepend: &str) {
        // super.store(props, prepend)
        let created = self.create_prepend(prepend);
        self.base
            .store_with_created_prepend(props, created.as_deref());
        let prepend = self
            .create_prepend(prepend)
            .unwrap_or_else(|| "null".to_string());
        let prepend_some = Some(prepend.as_str());
        storable::StorableValue::store_with_prepend(&*CURRENT_VERSION, props, &prepend);
        self.auto_alignment_meta_data
            .lock()
            .unwrap()
            .store(props, &prepend);
        self.root_name
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), prepend_some);
        self.stack
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), prepend_some);
        self.view_type
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), prepend_some);
        self.extract_plmdoc_metadata_file
            .lock()
            .unwrap()
            .store_with_prepend(props, prepend_some);
        self.distortion_field
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), prepend_some);
        self.images_are_binned
            .lock()
            .unwrap()
            .store_with_prepend(props, prepend_some);
        self.robust_fit_criterion
            .lock()
            .unwrap()
            .store_with_prepend(props, prepend_some);
        self.midas_binning
            .lock()
            .unwrap()
            .store_with_prepend(props, prepend_some);
        self.use_reference_section
            .lock()
            .unwrap()
            .store_with_prepend(props, prepend_some);
        self.reference_section
            .lock()
            .unwrap()
            .store_with_prepend(props, prepend_some);
        self.hybrid_fits_translations
            .lock()
            .unwrap()
            .store_with_prepend(props, prepend_some);
        self.hybrid_fits_translations_rotations
            .lock()
            .unwrap()
            .store_with_prepend(props, prepend_some);
        self.no_options
            .lock()
            .unwrap()
            .store_with_prepend(props, prepend_some);
        self.number_to_fit_global_alignment
            .lock()
            .unwrap()
            .store_with_prepend(props, prepend_some);
        self.shift_x
            .lock()
            .unwrap()
            .store_with_prepend(props, prepend_some);
        self.shift_y
            .lock()
            .unwrap()
            .store_with_prepend(props, prepend_some);
        self.size_x
            .lock()
            .unwrap()
            .store_with_prepend(props, prepend_some);
        self.size_y
            .lock()
            .unwrap()
            .store_with_prepend(props, prepend_some);
        self.tab
            .lock()
            .unwrap()
            .store_with_prepend(props, prepend_some);
        self.preblend_very_sloppy_montage
            .lock()
            .unwrap()
            .store_with_prepend(props, prepend_some);
        self.bool_preblend_weight_for_expected_shifts
            .lock()
            .unwrap()
            .store_with_prepend(props, prepend_some);
        self.str_preblend_weight_for_expected_shifts
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), prepend_some);
        self.preblend_e_m_grid_map_filter
            .lock()
            .unwrap()
            .store_with_prepend(props, prepend_some);
        self.preblend_high_frequency_filter_cutoff
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), prepend_some);
        self.other_sum_gradient_file
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), prepend_some);
    }

    /// Java `setStartupData(SerialSectionsStartupData)`.
    pub fn set_startup_data(&self, startup_data: &SerialSectionsStartupData) {
        self.set_name(startup_data.get_root_name().as_deref());
        // `startupData.getStack().getName()`: the stack is set (the data was
        // validated before it is handed over).
        self.stack.lock().unwrap().set(
            startup_data
                .get_stack()
                .map(|stack| utilities::java_io_file_get_name(&stack.to_string_lossy()))
                .as_deref(),
        );
        self.view_type
            .lock()
            .unwrap()
            .set(startup_data.get_view_type().map(ViewType::get_param_value));
        self.extract_plmdoc_metadata_file
            .lock()
            .unwrap()
            .set_boolean(startup_data.get_mdoc_metadata_file());
        let file = startup_data.get_distortion_field();
        match file {
            None => self.distortion_field.lock().unwrap().reset(),
            Some(file) => self.distortion_field.lock().unwrap().set(Some(
                &utilities::java_io_file_get_absolute_path(&file.to_string_lossy()),
            )),
        }
        self.images_are_binned
            .lock()
            .unwrap()
            .set_number(startup_data.get_images_are_binned());
    }

    /// Java `isextractPlMdocMetadataFile()`.
    pub fn isextract_pl_mdoc_metadata_file(&self) -> bool {
        self.extract_plmdoc_metadata_file.lock().unwrap().is()
    }

    /// Java `getDistortionField()`.
    pub fn get_distortion_field(&self) -> String {
        self.distortion_field
            .lock()
            .unwrap()
            .to_string_option()
            .unwrap_or_default()
    }

    /// Java `getImagesAreBinned()`.
    pub fn get_images_are_binned(&self) -> String {
        self.images_are_binned.lock().unwrap().to_string()
    }

    /// Java `setMidasBinning(Number)`.
    pub fn set_midas_binning(&self, input: Option<Number>) {
        self.midas_binning.lock().unwrap().set_number(input);
    }

    /// Java `setUseReferenceSection(boolean)`.
    pub fn set_use_reference_section(&self, input: bool) {
        self.use_reference_section
            .lock()
            .unwrap()
            .set_boolean(input);
    }

    /// Java `setReferenceSection(Number)`.
    pub fn set_reference_section(&self, input: Option<Number>) {
        self.reference_section.lock().unwrap().set_number(input);
    }

    /// Java `setRobustFitCriterion(String)`.
    pub fn set_robust_fit_criterion(&self, input: Option<&str>) {
        self.robust_fit_criterion.lock().unwrap().set_string(input);
    }

    /// Java `setSizeX(String)`.
    pub fn set_size_x(&self, input: Option<&str>) {
        self.size_x.lock().unwrap().set_string(input);
    }

    /// Java `setSizeY(String)`.
    pub fn set_size_y(&self, input: Option<&str>) {
        self.size_y.lock().unwrap().set_string(input);
    }

    /// Java `setShiftX(String)`.
    pub fn set_shift_x(&self, input: Option<&str>) {
        self.shift_x.lock().unwrap().set_string(input);
    }

    /// Java `setShiftY(String)`.
    pub fn set_shift_y(&self, input: Option<&str>) {
        self.shift_y.lock().unwrap().set_string(input);
    }

    /// Java `setTab(int)`.
    pub fn set_tab(&self, input: i32) {
        self.tab.lock().unwrap().set_int(input);
    }

    /// Java `setHybridFitsTranslations(boolean)`.
    pub fn set_hybrid_fits_translations(&self, input: bool) {
        self.hybrid_fits_translations
            .lock()
            .unwrap()
            .set_boolean(input);
    }

    /// Java `setHybridFitsTranslationsRotations(boolean)`.
    pub fn set_hybrid_fits_translations_rotations(&self, input: bool) {
        self.hybrid_fits_translations_rotations
            .lock()
            .unwrap()
            .set_boolean(input);
    }

    /// Java `setNoOptions(boolean)`.
    pub fn set_no_options(&self, input: bool) {
        self.no_options.lock().unwrap().set_boolean(input);
    }

    /// Java `setNumberToFitGlobalAlignment(boolean)`.
    pub fn set_number_to_fit_global_alignment(&self, input: bool) {
        self.number_to_fit_global_alignment
            .lock()
            .unwrap()
            .set_boolean(input);
    }

    /// Java `setPreblendVerySloppyMontage(boolean)`.
    pub fn set_preblend_very_sloppy_montage(&self, input: bool) {
        self.preblend_very_sloppy_montage
            .lock()
            .unwrap()
            .set_boolean(input);
    }

    /// Java `setPreblendWeightForExpectedShifts(boolean)`.
    pub fn set_preblend_weight_for_expected_shifts_boolean(&self, input: bool) {
        self.bool_preblend_weight_for_expected_shifts
            .lock()
            .unwrap()
            .set_boolean(input);
    }

    /// Java `setPreblendWeightForExpectedShifts(String)`.
    pub fn set_preblend_weight_for_expected_shifts_string(&self, input: Option<&str>) {
        self.str_preblend_weight_for_expected_shifts
            .lock()
            .unwrap()
            .set(input);
    }

    /// Java `setPreblendEMGridMapFilter(boolean)`.
    pub fn set_preblend_e_m_grid_map_filter(&self, input: bool) {
        self.preblend_e_m_grid_map_filter
            .lock()
            .unwrap()
            .set_boolean(input);
    }

    /// Java `setPreblendHighFrequencyFilterCutoff(String)`.
    pub fn set_preblend_high_frequency_filter_cutoff(&self, input: Option<&str>) {
        self.preblend_high_frequency_filter_cutoff
            .lock()
            .unwrap()
            .set(input);
    }

    /// Java `setOtherSumGradientFile(String)`.
    pub fn set_other_sum_gradient_file(&self, input: Option<&str>) {
        self.other_sum_gradient_file.lock().unwrap().set(input);
    }
}

impl ConstSerialSectionsMetaData for SerialSectionsMetaData {
    fn get_stack(&self) -> String {
        self.stack
            .lock()
            .unwrap()
            .to_string_option()
            .unwrap_or_default()
    }

    fn get_view_type(&self) -> Option<ViewType> {
        let view_type = self
            .view_type
            .lock()
            .unwrap()
            .to_string_option()
            .unwrap_or_default();
        ViewType::from_string(&view_type)
    }

    fn get_auto_alignment_meta_data(&self) -> &Mutex<AutoAlignmentMetaData> {
        &self.auto_alignment_meta_data
    }

    fn get_midas_binning(&self) -> ConstEtomoNumber {
        (**self.midas_binning.lock().unwrap()).clone()
    }

    fn get_reference_section(&self) -> ConstEtomoNumber {
        (**self.reference_section.lock().unwrap()).clone()
    }

    fn get_robust_fit_criterion(&self) -> String {
        self.robust_fit_criterion.lock().unwrap().to_string()
    }

    fn get_shift_x(&self) -> String {
        self.shift_x.lock().unwrap().to_string()
    }

    fn get_shift_y(&self) -> String {
        self.shift_y.lock().unwrap().to_string()
    }

    fn get_size_x(&self) -> String {
        self.size_x.lock().unwrap().to_string()
    }

    fn get_size_y(&self) -> String {
        self.size_y.lock().unwrap().to_string()
    }

    fn is_null_hybrid_fits_translations(&self) -> bool {
        self.hybrid_fits_translations.lock().unwrap().is_null()
    }

    fn is_hybrid_fits_translations(&self) -> bool {
        self.hybrid_fits_translations.lock().unwrap().is()
    }

    fn is_null_hybrid_fits_translations_rotations(&self) -> bool {
        self.hybrid_fits_translations_rotations
            .lock()
            .unwrap()
            .is_null()
    }

    fn is_hybrid_fits_translations_rotations(&self) -> bool {
        self.hybrid_fits_translations_rotations.lock().unwrap().is()
    }

    fn is_null_no_options(&self) -> bool {
        self.no_options.lock().unwrap().is_null()
    }

    fn is_no_options(&self) -> bool {
        self.no_options.lock().unwrap().is()
    }

    fn is_number_to_fit_global_alignment(&self) -> bool {
        self.number_to_fit_global_alignment.lock().unwrap().is()
    }

    fn get_tab(&self) -> i32 {
        self.tab.lock().unwrap().get_int()
    }

    fn is_tab_empty(&self) -> bool {
        self.tab.lock().unwrap().is_null()
    }

    fn is_use_reference_section(&self) -> bool {
        self.use_reference_section.lock().unwrap().is()
    }

    fn is_preblend_very_sloppy_montage(&self) -> bool {
        self.preblend_very_sloppy_montage.lock().unwrap().is()
    }

    fn is_bool_preblend_weight_for_expected_shifts(&self) -> bool {
        self.bool_preblend_weight_for_expected_shifts
            .lock()
            .unwrap()
            .is()
    }

    fn is_str_preblend_weight_for_expected_shifts(&self) -> bool {
        !self
            .str_preblend_weight_for_expected_shifts
            .lock()
            .unwrap()
            .is_empty()
    }

    fn get_str_preblend_weight_for_expected_shifts(&self) -> String {
        self.str_preblend_weight_for_expected_shifts
            .lock()
            .unwrap()
            .to_string_option()
            .unwrap_or_default()
    }

    fn is_preblend_e_m_grid_map_filter(&self) -> bool {
        self.preblend_e_m_grid_map_filter.lock().unwrap().is()
    }

    fn is_preblend_high_frequency_filter_cutoff(&self) -> bool {
        !self
            .preblend_high_frequency_filter_cutoff
            .lock()
            .unwrap()
            .is_empty()
    }

    fn get_preblend_high_frequency_filter_cutoff(&self) -> String {
        self.preblend_high_frequency_filter_cutoff
            .lock()
            .unwrap()
            .to_string_option()
            .unwrap_or_default()
    }

    fn get_other_sum_gradient_file(&self) -> String {
        self.other_sum_gradient_file
            .lock()
            .unwrap()
            .to_string_option()
            .unwrap_or_default()
    }
}

/// Java `Storable`, implemented through `BaseMetaData`.
impl Storable for SerialSectionsMetaData {
    /// Java `store(Properties)`: `store(props, "")`.
    fn store(&self, properties: &mut BTreeMap<String, String>) {
        self.store_with_prepend(properties, "");
    }

    /// Java `store(Properties, String)`.
    fn store_with_prepend(&self, properties: &mut BTreeMap<String, String>, prepend: &str) {
        SerialSectionsMetaData::store_with_prepend(self, properties, prepend);
    }

    /// Java `load(Properties)`: `load(props, "")`.
    fn load(&self, properties: &mut BTreeMap<String, String>) {
        self.load_with_prepend(properties, "");
    }

    /// Java `load(Properties, String)`.
    fn load_with_prepend(&self, properties: &mut BTreeMap<String, String>, prepend: &str) {
        SerialSectionsMetaData::load_with_prepend(self, properties, prepend);
    }
}

impl BaseMetaData for SerialSectionsMetaData {
    fn base(&self) -> &BaseMetaDataBase {
        &self.base
    }

    /// Java package-private `getGroupKey()`.
    fn get_group_key(&self) -> Option<String> {
        Some("SerialSections".to_string())
    }

    /// Java `getDatasetName()`.
    fn get_dataset_name(&self) -> Option<String> {
        Some(self.root_name_string())
    }

    /// Java `getMetaDataFileName()`.
    fn get_meta_data_file_name(&self) -> Option<String> {
        if self.root_name.lock().unwrap().equals(Some("")) {
            return None;
        }
        Some(format!(
            "{}{}",
            self.root_name_string(),
            self.base
                .get_file_extension()
                .unwrap_or_else(|| "null".to_string())
        ))
    }

    /// Java `getName()`.
    fn get_name(&self) -> Option<String> {
        let root_name = self.root_name_string();
        // `rootName.toString().matches("\\s*")`
        if root_name
            .chars()
            .all(|c| matches!(c, ' ' | '\t' | '\n' | '\x0B' | '\x0C' | '\r'))
        {
            return Some(NEW_TITLE.to_string());
        }
        Some(root_name)
    }

    /// Java `isValid()`.
    fn is_valid(&self) -> bool {
        true
    }
}
