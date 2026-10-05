//! `IMOD/Etomo/src/etomo/type/BatchRunTomoDatasetMetaData.java`.
//!
//! The dataset values of a batchruntomo dataset dialog (the global one, or a row's).
//!
//! **Representation.**  One instance belongs to the batch meta data and one to each row
//! meta data that has a dataset dialog; the dialogs read and write it on the event
//! dispatch thread and the meta data stores it (possibly from a process thread), so it
//! is shared as an `Arc` and its fields sit behind one lock.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Mutex;

use super::const_etomo_number::{
    ConstEtomoNumber, Number, Type, java_lang_string_matches_whitespace,
};
use super::const_panel_header_settings::ConstPanelHeaderSettings;
use super::etomo_boolean2::EtomoBoolean2;
use super::etomo_number::EtomoNumber;
use super::panel_header_settings::PanelHeaderSettings;
use super::string_property::StringProperty;
use crate::imod::etomo::r#type::const_etomo_number::java_lang_string_trim;

/// Java private static final `GROUP_KEY`.
const GROUP_KEY: &str = "dataset";
/// Java private static final `HEADER_KEY`.
const HEADER_KEY: &str = "header";
/// Java public static final `SCALE_FROM_Z_DEFAULT`.
pub const SCALE_FROM_Z_DEFAULT: &str = "0.33";

/// The fields of Java `BatchRunTomoDatasetMetaData`.
struct Fields {
    /// Java private final `dataset = new EtomoBoolean2()`.
    dataset: EtomoBoolean2,
    /// Java private final `modelFile`.
    model_file: StringProperty,
    /// Java private final `enableStretching`.
    enable_stretching: EtomoNumber,
    /// Java private final `localAlignments`.
    local_alignments: EtomoNumber,
    /// Java private final `gold`.
    gold: EtomoNumber,
    /// Java private final `targetNumberOfBeads`.
    target_number_of_beads: EtomoNumber,
    /// Java private final `numberOfMarkers`.
    number_of_markers: EtomoNumber,
    /// Java private final `localAreaTargetSize`.
    local_area_target_size: StringProperty,
    /// Java private final `sizeOfPatchesXandY`.
    size_of_patches_x_and_y: StringProperty,
    /// Java private final `lengthOfPieces`.
    length_of_pieces: EtomoNumber,
    /// Java private final `scanDefocusRange`.
    scan_defocus_range: StringProperty,
    /// Java private final `defocus`.
    defocus: StringProperty,
    /// Java private final `autoFitRangeAndStep`.
    auto_fit_range_and_step: EtomoNumber,
    /// Java private final `autoFitRange`.
    auto_fit_range: EtomoNumber,
    /// Java private final `fitEveryImage`.
    fit_every_image: EtomoNumber,
    /// Java private final `autoFitStep`.
    auto_fit_step: EtomoNumber,
    /// Java private final `useFakeSIRTiterations`.
    use_fake_sirt_iterations: EtomoNumber,
    /// Java private final `fakeSIRTiterations`.
    fake_sirt_iterations: EtomoNumber,
    /// Java private final `leaveIterations`.
    leave_iterations: StringProperty,
    /// Java private final `scaleToInteger`.
    scale_to_integer: EtomoNumber,
    /// Java private final `thickness`.
    thickness: EtomoNumber,
    /// Java private final `binnedThickness`.
    binned_thickness: EtomoNumber,
    /// Java private final `extraThickness`.
    extra_thickness: EtomoNumber,
    /// Java private final `fallbackThickness`.
    fallback_thickness: EtomoNumber,
    /// Java private final `prenewstBinByFactor`.
    prenewst_bin_by_factor: EtomoNumber,
    /// Java private final `preblendBinByFactor`.
    preblend_bin_by_factor: EtomoNumber,
    /// Java private final `findSecAddThickness`.
    find_sec_add_thickness: EtomoNumber,
    /// Java private final `useFindSecAddThickness`.
    use_find_sec_add_thickness: EtomoNumber,
    /// Java private final `useScaleFromZ`.
    use_scale_from_z: EtomoNumber,
    /// Java private final `scaleFromZ`.
    scale_from_z: EtomoNumber,
    /// Java private final `eraseGoldFid`.
    erase_gold_fid: EtomoNumber,
    /// Java private final `eraseGold3d`.
    erase_gold_3d: EtomoNumber,
    /// Java private final `goldErasingThickness`.
    gold_erasing_thickness: EtomoNumber,
    /// Java private final `sampleTypePlasticSection`.
    sample_type_plastic_section: EtomoNumber,
    /// Java private final `sampleTypeCryo`.
    sample_type_cryo: EtomoNumber,
    /// Java private final `positioningThickness`.
    positioning_thickness: EtomoNumber,
    /// Java private final `hasGoldBeads`.
    has_gold_beads: EtomoNumber,
    /// Java private final `positioningGold`.
    positioning_gold: EtomoNumber,
    /// Java private final `postprocessingHeader = new PanelHeaderSettings("Postprocessing.header")`.
    postprocessing_header: PanelHeaderSettings,
    /// Java private final `tuneFittingAndSampling`.
    tune_fitting_and_sampling: EtomoNumber,
    /// Java private `header`, initially null.
    header: Option<PanelHeaderSettings>,
}

/// Java `public final class BatchRunTomoDatasetMetaData`.
pub struct BatchRunTomoDatasetMetaData {
    fields: Mutex<Fields>,
}

impl BatchRunTomoDatasetMetaData {
    /// Java package-private `BatchRunTomoDatasetMetaData()`.
    pub fn new() -> BatchRunTomoDatasetMetaData {
        let mut dataset = EtomoBoolean2::new();
        dataset.set_boolean(true);
        BatchRunTomoDatasetMetaData {
            fields: Mutex::new(Fields {
                dataset,
                model_file: StringProperty::new_with_key(Some("ModelFile")),
                enable_stretching: EtomoNumber::new_with_type_and_name(
                    Type::Boolean,
                    "enableStretching",
                ),
                local_alignments: EtomoNumber::new_with_type_and_name(
                    Type::Boolean,
                    "LocalAlignments",
                ),
                gold: EtomoNumber::new_with_type_and_name(Type::Double, "gold"),
                target_number_of_beads: EtomoNumber::new_with_type_and_name(
                    Type::Integer,
                    "TargetNumberOfBeads",
                ),
                number_of_markers: EtomoNumber::new_with_type_and_name(
                    Type::Integer,
                    "NumberOfMarkers",
                ),
                local_area_target_size: StringProperty::new_with_key(Some("LocalAreaTargetSize")),
                size_of_patches_x_and_y: StringProperty::new_with_key(Some("SizeOfPatchesXandY")),
                length_of_pieces: EtomoNumber::new_with_type_and_name(
                    Type::Boolean,
                    "LengthOfPieces",
                ),
                scan_defocus_range: StringProperty::new_with_key(Some("scanDefocusRange")),
                defocus: StringProperty::new_with_key(Some("defocus")),
                auto_fit_range_and_step: EtomoNumber::new_with_type_and_name(
                    Type::Boolean,
                    "autoFitRangeAndStep",
                ),
                auto_fit_range: EtomoNumber::new_with_type_and_name(Type::Double, "autoFitRange"),
                fit_every_image: EtomoNumber::new_with_type_and_name(
                    Type::Boolean,
                    "fitEveryImage",
                ),
                auto_fit_step: EtomoNumber::new_with_type_and_name(Type::Double, "autoFitStep"),
                use_fake_sirt_iterations: EtomoNumber::new_with_type_and_name(
                    Type::Boolean,
                    "Use.fakeSIRTiterations",
                ),
                fake_sirt_iterations: EtomoNumber::new_with_type_and_name(
                    Type::Integer,
                    "fakeSIRTiterations",
                ),
                leave_iterations: StringProperty::new_with_key(Some("LeaveIterations")),
                scale_to_integer: EtomoNumber::new_with_type_and_name(
                    Type::Boolean,
                    "ScaleToInteger",
                ),
                thickness: EtomoNumber::new_with_name("THICKNESS"),
                binned_thickness: EtomoNumber::new_with_name("binnedThickness"),
                extra_thickness: EtomoNumber::new_with_name("extraThickness"),
                fallback_thickness: EtomoNumber::new_with_name("fallbackThickness"),
                prenewst_bin_by_factor: EtomoNumber::new_with_name("Prenewst.BinByFactor"),
                preblend_bin_by_factor: EtomoNumber::new_with_name("Preblend.BinByFactor"),
                find_sec_add_thickness: EtomoNumber::new_with_type_and_name(
                    Type::Double,
                    "findSecAddThickness",
                ),
                use_find_sec_add_thickness: EtomoNumber::new_with_type_and_name(
                    Type::Boolean,
                    "Use.findSecAddThickness",
                ),
                use_scale_from_z: EtomoNumber::new_with_type_and_name(
                    Type::Boolean,
                    "Use.fcaleFromZ",
                ),
                scale_from_z: EtomoNumber::new_with_type_and_name(Type::Double, "fcaleFromZ"),
                erase_gold_fid: EtomoNumber::new_with_type_and_name(Type::Boolean, "eraseGold.Fid"),
                erase_gold_3d: EtomoNumber::new_with_type_and_name(Type::Boolean, "eraseGold.3d"),
                gold_erasing_thickness: EtomoNumber::new_with_type_and_name(
                    Type::Integer,
                    "GoldErasing.thickness",
                ),
                sample_type_plastic_section: EtomoNumber::new_with_type_and_name(
                    Type::Boolean,
                    "sampleType.PlasticSection",
                ),
                sample_type_cryo: EtomoNumber::new_with_type_and_name(
                    Type::Boolean,
                    "sampleType.Cryo",
                ),
                positioning_thickness: EtomoNumber::new_with_type_and_name(
                    Type::Integer,
                    "Positioning.thickness",
                ),
                has_gold_beads: EtomoNumber::new_with_type_and_name(Type::Boolean, "hasGoldBeads"),
                positioning_gold: EtomoNumber::new_with_type_and_name(
                    Type::Double,
                    "Positioning.gold",
                ),
                postprocessing_header: PanelHeaderSettings::new("Postprocessing.header"),
                tune_fitting_and_sampling: EtomoNumber::new_with_type_and_name(
                    Type::Boolean,
                    "ctfplotter.TuneFittingAndSampling",
                ),
                header: None,
            }),
        }
    }

    /// Java static `createPrepend(String)`.
    pub fn create_prepend(prepend: Option<&str>) -> String {
        let Some(prepend) = prepend else {
            return GROUP_KEY.to_owned();
        };
        if java_lang_string_matches_whitespace(prepend) {
            return GROUP_KEY.to_owned();
        }
        let prepend = java_lang_string_trim(prepend);
        if prepend.ends_with('.') {
            return format!("{}{}", prepend, GROUP_KEY);
        }
        format!("{}.{}", prepend, GROUP_KEY)
    }

    /// Java package-private static `exists(Properties, String)`.
    pub fn exists(props: &BTreeMap<String, String>, prepend: Option<&str>) -> bool {
        let prepend = Self::create_prepend(prepend);
        let mut bool_ = EtomoBoolean2::new();
        bool_.set_string(props.get(&prepend).map(String::as_str));
        bool_.is()
    }

    /// Java package-private `reset()`.
    pub fn reset(&self) {
        let mut fields = self.fields.lock().unwrap();
        Self::reset_fields(&mut fields);
    }

    fn reset_fields(f: &mut Fields) {
        f.dataset.reset();
        if let Some(header) = &mut f.header {
            header.reset();
        }
        f.postprocessing_header.reset();
        f.model_file.reset();
        f.enable_stretching.reset();
        f.local_alignments.reset();
        f.gold.reset();
        f.target_number_of_beads.reset();
        f.number_of_markers.reset();
        f.local_area_target_size.reset();
        f.size_of_patches_x_and_y.reset();
        f.length_of_pieces.reset();
        f.scan_defocus_range.reset();
        f.defocus.reset();
        f.auto_fit_range_and_step.reset();
        f.auto_fit_range.reset();
        f.fit_every_image.reset();
        f.auto_fit_step.reset();
        f.use_fake_sirt_iterations.reset();
        f.fake_sirt_iterations.reset();
        f.leave_iterations.reset();
        f.scale_to_integer.reset();
        f.thickness.reset();
        f.binned_thickness.reset();
        f.extra_thickness.reset();
        f.fallback_thickness.reset();
        f.prenewst_bin_by_factor.reset();
        f.preblend_bin_by_factor.reset();
        f.find_sec_add_thickness.reset();
        f.use_find_sec_add_thickness.reset();
        f.use_scale_from_z.reset();
        f.scale_from_z.set_string(Some(SCALE_FROM_Z_DEFAULT));
        f.erase_gold_fid.reset();
        f.erase_gold_3d.reset();
        f.gold_erasing_thickness.reset();
        f.sample_type_plastic_section.reset();
        f.sample_type_cryo.reset();
        f.positioning_thickness.reset();
        f.has_gold_beads.reset();
        f.positioning_gold.reset();
        f.tune_fitting_and_sampling.reset();
    }

    /// Java `load(Properties, String)`.
    pub fn load(&self, props: &mut BTreeMap<String, String>, prepend: Option<&str>) {
        let mut f = self.fields.lock().unwrap();
        // reset
        Self::reset_fields(&mut f);
        // load
        let prepend = Self::create_prepend(prepend);
        let p = Some(prepend.as_str());
        // `StringProperty.load` may remove a backward-compatible key from `props`; none
        // of these declares one.
        f.dataset
            .set_string(props.get(&prepend).map(String::as_str));
        if !f.dataset.is() {
            return;
        }
        f.header = PanelHeaderSettings::load_instance(f.header.take(), HEADER_KEY, props, p);
        f.postprocessing_header.load(props, p);
        f.model_file.load_with_prepend(Some(&mut *props), p);
        f.enable_stretching.load_with_prepend(props, p);
        f.local_alignments.load_with_prepend(props, p);
        f.gold.load_with_prepend(props, p);
        f.target_number_of_beads.load_with_prepend(props, p);
        f.number_of_markers.load_with_prepend(props, p);
        f.local_area_target_size
            .load_with_prepend(Some(&mut *props), p);
        f.size_of_patches_x_and_y
            .load_with_prepend(Some(&mut *props), p);
        f.length_of_pieces.load_with_prepend(props, p);
        f.scan_defocus_range.load_with_prepend(Some(&mut *props), p);
        f.defocus.load_with_prepend(Some(&mut *props), p);
        f.auto_fit_range_and_step.load_with_prepend(props, p);
        f.auto_fit_range.load_with_prepend(props, p);
        f.fit_every_image.load_with_prepend(props, p);
        f.auto_fit_step.load_with_prepend(props, p);
        f.use_fake_sirt_iterations.load_with_prepend(props, p);
        f.fake_sirt_iterations.load_with_prepend(props, p);
        f.leave_iterations.load_with_prepend(Some(&mut *props), p);
        f.scale_to_integer.load_with_prepend(props, p);
        f.thickness.load_with_prepend(props, p);
        f.binned_thickness.load_with_prepend(props, p);
        f.extra_thickness.load_with_prepend(props, p);
        f.fallback_thickness.load_with_prepend(props, p);
        f.prenewst_bin_by_factor.load_with_prepend(props, p);
        f.preblend_bin_by_factor.load_with_prepend(props, p);
        f.find_sec_add_thickness.load_with_prepend(props, p);
        f.use_find_sec_add_thickness.load_with_prepend(props, p);
        f.use_scale_from_z.load_with_prepend(props, p);
        f.scale_from_z.load_with_prepend(props, p);
        f.erase_gold_fid.load_with_prepend(props, p);
        f.erase_gold_3d.load_with_prepend(props, p);
        f.gold_erasing_thickness.load_with_prepend(props, p);
        f.sample_type_plastic_section.load_with_prepend(props, p);
        f.sample_type_cryo.load_with_prepend(props, p);
        f.positioning_thickness.load_with_prepend(props, p);
        f.has_gold_beads.load_with_prepend(props, p);
        f.positioning_gold.load_with_prepend(props, p);
        f.tune_fitting_and_sampling.load_with_prepend(props, p);
    }

    /// Java `store(Properties, String)`.
    pub fn store(&self, props: &mut BTreeMap<String, String>, prepend: Option<&str>) {
        let dataset = self.fields.lock().unwrap().dataset.is();
        if !dataset {
            self.remove(props, prepend);
        } else {
            let f = self.fields.lock().unwrap();
            let prepend = Self::create_prepend(prepend);
            let p = Some(prepend.as_str());
            if let Some(header) = &f.header {
                header.store(props, p);
            }
            props.insert(prepend.clone(), f.dataset.to_string());
            f.postprocessing_header.store(props, p);
            f.model_file.store_with_prepend(Some(props), p);
            f.enable_stretching.store_with_prepend(props, p);
            f.local_alignments.store_with_prepend(props, p);
            f.gold.store_with_prepend(props, p);
            f.target_number_of_beads.store_with_prepend(props, p);
            f.number_of_markers.store_with_prepend(props, p);
            f.local_area_target_size.store_with_prepend(Some(props), p);
            f.size_of_patches_x_and_y.store_with_prepend(Some(props), p);
            f.length_of_pieces.store_with_prepend(props, p);
            f.scan_defocus_range.store_with_prepend(Some(props), p);
            f.defocus.store_with_prepend(Some(props), p);
            f.auto_fit_range_and_step.store_with_prepend(props, p);
            f.auto_fit_range.store_with_prepend(props, p);
            f.fit_every_image.store_with_prepend(props, p);
            f.auto_fit_step.store_with_prepend(props, p);
            f.use_fake_sirt_iterations.store_with_prepend(props, p);
            f.fake_sirt_iterations.store_with_prepend(props, p);
            f.leave_iterations.store_with_prepend(Some(props), p);
            f.scale_to_integer.store_with_prepend(props, p);
            f.thickness.store_with_prepend(props, p);
            f.binned_thickness.store_with_prepend(props, p);
            f.extra_thickness.store_with_prepend(props, p);
            f.fallback_thickness.store_with_prepend(props, p);
            f.prenewst_bin_by_factor.store_with_prepend(props, p);
            f.preblend_bin_by_factor.store_with_prepend(props, p);
            f.find_sec_add_thickness.store_with_prepend(props, p);
            f.use_find_sec_add_thickness.store_with_prepend(props, p);
            f.use_scale_from_z.store_with_prepend(props, p);
            f.scale_from_z.store_with_prepend(props, p);
            f.erase_gold_fid.store_with_prepend(props, p);
            f.erase_gold_3d.store_with_prepend(props, p);
            f.gold_erasing_thickness.store_with_prepend(props, p);
            f.sample_type_plastic_section.store_with_prepend(props, p);
            f.sample_type_cryo.store_with_prepend(props, p);
            f.positioning_thickness.store_with_prepend(props, p);
            f.has_gold_beads.store_with_prepend(props, p);
            f.positioning_gold.store_with_prepend(props, p);
            f.tune_fitting_and_sampling.store_with_prepend(props, p);
        }
    }

    /// Java `remove(Properties, String)`.
    pub fn remove(&self, props: &mut BTreeMap<String, String>, prepend: Option<&str>) {
        let f = self.fields.lock().unwrap();
        let prepend = Self::create_prepend(prepend);
        let p = Some(prepend.as_str());
        if let Some(header) = &f.header {
            header.remove(props, p);
        }
        props.remove(&prepend);
        f.postprocessing_header.remove(props, p);
        f.model_file.remove(Some(props), p);
        f.enable_stretching.remove_with_prepend(props, p);
        f.local_alignments.remove_with_prepend(props, p);
        f.gold.remove_with_prepend(props, p);
        f.target_number_of_beads.remove_with_prepend(props, p);
        f.number_of_markers.remove_with_prepend(props, p);
        f.local_area_target_size.remove(Some(props), p);
        f.size_of_patches_x_and_y.remove(Some(props), p);
        f.length_of_pieces.remove_with_prepend(props, p);
        f.scan_defocus_range.remove(Some(props), p);
        f.defocus.remove(Some(props), p);
        f.auto_fit_range_and_step.remove_with_prepend(props, p);
        f.auto_fit_range.remove_with_prepend(props, p);
        f.fit_every_image.remove_with_prepend(props, p);
        f.auto_fit_step.remove_with_prepend(props, p);
        f.use_fake_sirt_iterations.remove_with_prepend(props, p);
        f.fake_sirt_iterations.remove_with_prepend(props, p);
        f.leave_iterations.remove(Some(props), p);
        f.scale_to_integer.remove_with_prepend(props, p);
        f.thickness.remove_with_prepend(props, p);
        f.binned_thickness.remove_with_prepend(props, p);
        f.extra_thickness.remove_with_prepend(props, p);
        f.fallback_thickness.remove_with_prepend(props, p);
        f.prenewst_bin_by_factor.remove_with_prepend(props, p);
        f.preblend_bin_by_factor.remove_with_prepend(props, p);
        f.find_sec_add_thickness.remove_with_prepend(props, p);
        f.use_find_sec_add_thickness.remove_with_prepend(props, p);
        f.use_scale_from_z.remove_with_prepend(props, p);
        f.scale_from_z.remove_with_prepend(props, p);
        f.erase_gold_fid.remove_with_prepend(props, p);
        f.erase_gold_3d.remove_with_prepend(props, p);
        f.gold_erasing_thickness.remove_with_prepend(props, p);
        f.sample_type_plastic_section.remove_with_prepend(props, p);
        f.sample_type_cryo.remove_with_prepend(props, p);
        f.positioning_thickness.remove_with_prepend(props, p);
        f.has_gold_beads.remove_with_prepend(props, p);
        f.positioning_gold.remove_with_prepend(props, p);
        f.tune_fitting_and_sampling.remove_with_prepend(props, p);
    }

    /// Java `getHeader()`.
    pub fn get_header(&self) -> Option<PanelHeaderSettings> {
        self.fields.lock().unwrap().header.clone()
    }

    /// Java `getPostprocessingHeader()`.
    pub fn get_postprocessing_header(&self) -> PanelHeaderSettings {
        self.fields.lock().unwrap().postprocessing_header.clone()
    }

    /// Java `setHeader(ConstPanelHeaderSettings)`.
    pub fn set_header(&self, input: Option<&dyn ConstPanelHeaderSettings>) {
        let Some(input) = input else {
            return;
        };
        let mut f = self.fields.lock().unwrap();
        if f.header.is_none() {
            f.header = Some(PanelHeaderSettings::new(HEADER_KEY));
        }
        f.header.as_mut().unwrap().set(input);
    }

    /// Java `setPostprocessingHeader(ConstPanelHeaderSettings)`.
    pub fn set_postprocessing_header(&self, input: &dyn ConstPanelHeaderSettings) {
        self.fields.lock().unwrap().postprocessing_header.set(input);
    }

    /// Java `setDataset(boolean)`.
    pub fn set_dataset(&self, input: bool) {
        self.fields.lock().unwrap().dataset.set_boolean(input);
    }

    /// Java `setFitEveryImage(boolean)`.
    pub fn set_fit_every_image(&self, input: bool) {
        self.fields
            .lock()
            .unwrap()
            .fit_every_image
            .set_boolean(input);
    }

    /// Java `getFitEveryImage()`: a copy of the field.
    pub fn get_fit_every_image(&self) -> ConstEtomoNumber {
        let f = self.fields.lock().unwrap();
        let number: &ConstEtomoNumber = &f.fit_every_image;
        number.clone()
    }

    /// Java `setLocalAlignments(boolean)`.
    pub fn set_local_alignments(&self, input: bool) {
        self.fields
            .lock()
            .unwrap()
            .local_alignments
            .set_boolean(input);
    }

    /// Java `getLocalAlignments()`: a copy of the field.
    pub fn get_local_alignments(&self) -> ConstEtomoNumber {
        let f = self.fields.lock().unwrap();
        let number: &ConstEtomoNumber = &f.local_alignments;
        number.clone()
    }

    /// Java `setAutoFitRange(String)`.
    pub fn set_auto_fit_range(&self, input: Option<&str>) {
        self.fields.lock().unwrap().auto_fit_range.set_string(input);
    }

    /// Java `getAutoFitRange()`.
    pub fn get_auto_fit_range(&self) -> String {
        self.fields.lock().unwrap().auto_fit_range.to_string()
    }

    /// Java `setAutoFitStep(String)`.
    pub fn set_auto_fit_step(&self, input: Option<&str>) {
        self.fields.lock().unwrap().auto_fit_step.set_string(input);
    }

    /// Java `getAutoFitStep()`.
    pub fn get_auto_fit_step(&self) -> String {
        self.fields.lock().unwrap().auto_fit_step.to_string()
    }

    /// Java `setLengthOfPieces(boolean)`.
    pub fn set_length_of_pieces(&self, input: bool) {
        self.fields
            .lock()
            .unwrap()
            .length_of_pieces
            .set_boolean(input);
    }

    /// Java `getLengthOfPieces()`: a copy of the field.
    pub fn get_length_of_pieces(&self) -> ConstEtomoNumber {
        let f = self.fields.lock().unwrap();
        let number: &ConstEtomoNumber = &f.length_of_pieces;
        number.clone()
    }

    /// Java `setScanDefocusRange(String)`.
    pub fn set_scan_defocus_range(&self, input: Option<&str>) {
        self.fields.lock().unwrap().scan_defocus_range.set(input);
    }

    /// Java `setDefocus(String)`.
    pub fn set_defocus(&self, input: Option<&str>) {
        self.fields.lock().unwrap().defocus.set(input);
    }

    /// Java `getScanDefocusRange()`.
    pub fn get_scan_defocus_range(&self) -> String {
        self.fields.lock().unwrap().scan_defocus_range.to_string()
    }

    /// Java `getDefocus()`.
    pub fn get_defocus(&self) -> String {
        self.fields.lock().unwrap().defocus.to_string()
    }

    /// Java `setThickness(String)`.
    pub fn set_thickness(&self, input: Option<&str>) {
        self.fields.lock().unwrap().thickness.set_string(input);
    }

    /// Java `getThickness()`.
    pub fn get_thickness(&self) -> String {
        self.fields.lock().unwrap().thickness.to_string()
    }

    /// Java `setBinnedThickness(String)`.
    pub fn set_binned_thickness(&self, input: Option<&str>) {
        self.fields
            .lock()
            .unwrap()
            .binned_thickness
            .set_string(input);
    }

    /// Java `getBinnedThickness()`.
    pub fn get_binned_thickness(&self) -> String {
        self.fields.lock().unwrap().binned_thickness.to_string()
    }

    /// Java `setExtraThickness(String)`.
    pub fn set_extra_thickness(&self, input: Option<&str>) {
        self.fields
            .lock()
            .unwrap()
            .extra_thickness
            .set_string(input);
    }

    /// Java `getExtraThickness()`.
    pub fn get_extra_thickness(&self) -> String {
        self.fields.lock().unwrap().extra_thickness.to_string()
    }

    /// Java `setFindSecAddThickness(String)`.
    pub fn set_find_sec_add_thickness(&self, input: Option<&str>) {
        self.fields
            .lock()
            .unwrap()
            .find_sec_add_thickness
            .set_string(input);
    }

    /// Java `setUseFakeSIRTiterations(boolean)`.
    pub fn set_use_fake_sirt_iterations(&self, input: bool) {
        self.fields
            .lock()
            .unwrap()
            .use_fake_sirt_iterations
            .set_boolean(input);
    }

    /// Java `setUseFindSecAddThickness(boolean)`.
    pub fn set_use_find_sec_add_thickness(&self, input: bool) {
        self.fields
            .lock()
            .unwrap()
            .use_find_sec_add_thickness
            .set_boolean(input);
    }

    /// Java `setUseScaleFromZ(boolean)`.
    pub fn set_use_scale_from_z(&self, input: bool) {
        self.fields
            .lock()
            .unwrap()
            .use_scale_from_z
            .set_boolean(input);
    }

    /// Java `setEraseGoldFid(boolean)`.
    pub fn set_erase_gold_fid(&self, input: bool) {
        self.fields
            .lock()
            .unwrap()
            .erase_gold_fid
            .set_boolean(input);
    }

    /// Java `setEraseGold3d(boolean)`.
    pub fn set_erase_gold_3d(&self, input: bool) {
        self.fields.lock().unwrap().erase_gold_3d.set_boolean(input);
    }

    /// Java `setSampleTypePlasticSection(boolean)`.
    pub fn set_sample_type_plastic_section(&self, input: bool) {
        self.fields
            .lock()
            .unwrap()
            .sample_type_plastic_section
            .set_boolean(input);
    }

    /// Java `setSampleTypeCryo(boolean)`.
    pub fn set_sample_type_cryo(&self, input: bool) {
        self.fields
            .lock()
            .unwrap()
            .sample_type_cryo
            .set_boolean(input);
    }

    /// Java `setHasGoldBeads(boolean)`.
    pub fn set_has_gold_beads(&self, input: bool) {
        self.fields
            .lock()
            .unwrap()
            .has_gold_beads
            .set_boolean(input);
    }

    /// Java `setTuneFittingAndSampling(boolean)`.
    pub fn set_tune_fitting_and_sampling(&self, input: bool) {
        self.fields
            .lock()
            .unwrap()
            .tune_fitting_and_sampling
            .set_boolean(input);
    }

    /// Java `setScaleFromZ(String)`.
    pub fn set_scale_from_z(&self, input: Option<&str>) {
        self.fields.lock().unwrap().scale_from_z.set_string(input);
    }

    /// Java `setGoldErasingThickness(String)`.
    pub fn set_gold_erasing_thickness(&self, input: Option<&str>) {
        self.fields
            .lock()
            .unwrap()
            .gold_erasing_thickness
            .set_string(input);
    }

    /// Java `setPositioningThickness(String)`.
    pub fn set_positioning_thickness(&self, input: Option<&str>) {
        self.fields
            .lock()
            .unwrap()
            .positioning_thickness
            .set_string(input);
    }

    /// Java `setPositioningGold(String)`.
    pub fn set_positioning_gold(&self, input: Option<&str>) {
        self.fields
            .lock()
            .unwrap()
            .positioning_gold
            .set_string(input);
    }

    /// Java `setFallbackThickness(String)`.
    pub fn set_fallback_thickness(&self, input: Option<&str>) {
        self.fields
            .lock()
            .unwrap()
            .fallback_thickness
            .set_string(input);
    }

    /// Java `setPrenewstBinByFactor(Number)`.
    pub fn set_prenewst_bin_by_factor(&self, input: Option<Number>) {
        self.fields
            .lock()
            .unwrap()
            .prenewst_bin_by_factor
            .set_number(input);
    }

    /// Java `setPreblendBinByFactor(Number)`.
    pub fn set_preblend_bin_by_factor(&self, input: Option<Number>) {
        self.fields
            .lock()
            .unwrap()
            .preblend_bin_by_factor
            .set_number(input);
    }

    /// Java `getFindSecAddThickness()`.
    pub fn get_find_sec_add_thickness(&self) -> String {
        self.fields
            .lock()
            .unwrap()
            .find_sec_add_thickness
            .to_string()
    }

    /// Java `getUseScaleFromZ()`: a copy of the field.
    pub fn get_use_scale_from_z(&self) -> ConstEtomoNumber {
        let f = self.fields.lock().unwrap();
        let number: &ConstEtomoNumber = &f.use_scale_from_z;
        number.clone()
    }

    /// Java `getEraseGoldFid()`: a copy of the field.
    pub fn get_erase_gold_fid(&self) -> ConstEtomoNumber {
        let f = self.fields.lock().unwrap();
        let number: &ConstEtomoNumber = &f.erase_gold_fid;
        number.clone()
    }

    /// Java `getEraseGold3d()`: a copy of the field.
    pub fn get_erase_gold_3d(&self) -> ConstEtomoNumber {
        let f = self.fields.lock().unwrap();
        let number: &ConstEtomoNumber = &f.erase_gold_3d;
        number.clone()
    }

    /// Java `getSampleTypePlasticSection()`: a copy of the field.
    pub fn get_sample_type_plastic_section(&self) -> ConstEtomoNumber {
        let f = self.fields.lock().unwrap();
        let number: &ConstEtomoNumber = &f.sample_type_plastic_section;
        number.clone()
    }

    /// Java `getSampleTypeCryo()`: a copy of the field.
    pub fn get_sample_type_cryo(&self) -> ConstEtomoNumber {
        let f = self.fields.lock().unwrap();
        let number: &ConstEtomoNumber = &f.sample_type_cryo;
        number.clone()
    }

    /// Java `getHasGoldBeads()`: a copy of the field.
    pub fn get_has_gold_beads(&self) -> ConstEtomoNumber {
        let f = self.fields.lock().unwrap();
        let number: &ConstEtomoNumber = &f.has_gold_beads;
        number.clone()
    }

    /// Java `getTuneFittingAndSampling()`: a copy of the field.
    pub fn get_tune_fitting_and_sampling(&self) -> ConstEtomoNumber {
        let f = self.fields.lock().unwrap();
        let number: &ConstEtomoNumber = &f.tune_fitting_and_sampling;
        number.clone()
    }

    /// Java `getScaleFromZ()`.
    pub fn get_scale_from_z(&self) -> String {
        self.fields.lock().unwrap().scale_from_z.to_string()
    }

    /// Java `getGoldErasingThickness()`.
    pub fn get_gold_erasing_thickness(&self) -> String {
        self.fields
            .lock()
            .unwrap()
            .gold_erasing_thickness
            .to_string()
    }

    /// Java `getPositioningThickness()`.
    pub fn get_positioning_thickness(&self) -> String {
        self.fields
            .lock()
            .unwrap()
            .positioning_thickness
            .to_string()
    }

    /// Java `getPositioningGold()`.
    pub fn get_positioning_gold(&self) -> String {
        self.fields.lock().unwrap().positioning_gold.to_string()
    }

    /// Java `isUseFakeSIRTiterations()`.
    pub fn is_use_fake_sirt_iterations(&self) -> bool {
        self.fields.lock().unwrap().use_fake_sirt_iterations.is()
    }

    /// Java `getUseFindSecAddThickness()`: a copy of the field.
    pub fn get_use_find_sec_add_thickness(&self) -> ConstEtomoNumber {
        let f = self.fields.lock().unwrap();
        let number: &ConstEtomoNumber = &f.use_find_sec_add_thickness;
        number.clone()
    }

    /// Java `getFallbackThickness()`.
    pub fn get_fallback_thickness(&self) -> String {
        self.fields.lock().unwrap().fallback_thickness.to_string()
    }

    /// Java `getPrenewstBinByFactor()`.
    pub fn get_prenewst_bin_by_factor(&self) -> String {
        self.fields
            .lock()
            .unwrap()
            .prenewst_bin_by_factor
            .to_string()
    }

    /// Java `getPreblendBinByFactor()`.
    pub fn get_preblend_bin_by_factor(&self) -> String {
        self.fields
            .lock()
            .unwrap()
            .preblend_bin_by_factor
            .to_string()
    }

    /// Java `setGold(String)`.
    pub fn set_gold(&self, input: Option<&str>) {
        self.fields.lock().unwrap().gold.set_string(input);
    }

    /// Java `getGold()`.
    pub fn get_gold(&self) -> String {
        self.fields.lock().unwrap().gold.to_string()
    }

    /// Java `setFakeSIRTiterations(String)`.
    pub fn set_fake_sirt_iterations(&self, input: Option<&str>) {
        self.fields
            .lock()
            .unwrap()
            .fake_sirt_iterations
            .set_string(input);
    }

    /// Java `setLeaveIterations(String)`.
    pub fn set_leave_iterations(&self, input: Option<&str>) {
        self.fields.lock().unwrap().leave_iterations.set(input);
    }

    /// Java `getFakeSIRTiterations()`.
    pub fn get_fake_sirt_iterations(&self) -> String {
        self.fields.lock().unwrap().fake_sirt_iterations.to_string()
    }

    /// Java `getLeaveIterations()`.
    pub fn get_leave_iterations(&self) -> String {
        self.fields.lock().unwrap().leave_iterations.to_string()
    }

    /// Java `setLocalAreaTargetSize(String)`.
    pub fn set_local_area_target_size(&self, input: Option<&str>) {
        self.fields
            .lock()
            .unwrap()
            .local_area_target_size
            .set(input);
    }

    /// Java `getLocalAreaTargetSize()`.
    pub fn get_local_area_target_size(&self) -> String {
        self.fields
            .lock()
            .unwrap()
            .local_area_target_size
            .to_string()
    }

    /// Java `setModelFile(File)`.
    pub fn set_model_file(&self, input: Option<&Path>) {
        let mut f = self.fields.lock().unwrap();
        match input {
            Some(input) => f.model_file.set(Some(
                &crate::imod::etomo::util::utilities::java_io_file_get_absolute_path(
                    &input.to_string_lossy(),
                ),
            )),
            None => f.model_file.reset(),
        }
    }

    /// Java `getModelFile()`.
    pub fn get_model_file(&self) -> String {
        self.fields.lock().unwrap().model_file.to_string()
    }

    /// Java `setSizeOfPatchesXandY(String)`.
    pub fn set_size_of_patches_x_and_y(&self, input: Option<&str>) {
        self.fields
            .lock()
            .unwrap()
            .size_of_patches_x_and_y
            .set(input);
    }

    /// Java `getSizeOfPatchesXandY()`.
    pub fn get_size_of_patches_x_and_y(&self) -> String {
        self.fields
            .lock()
            .unwrap()
            .size_of_patches_x_and_y
            .to_string()
    }

    /// Java `setTargetNumberOfBeads(String)`.
    pub fn set_target_number_of_beads(&self, input: Option<&str>) {
        self.fields
            .lock()
            .unwrap()
            .target_number_of_beads
            .set_string(input);
    }

    /// Java `setNumberOfMarkers(String)`.
    pub fn set_number_of_markers(&self, input: Option<&str>) {
        self.fields
            .lock()
            .unwrap()
            .number_of_markers
            .set_string(input);
    }

    /// Java `getTargetNumberOfBeads()`.
    pub fn get_target_number_of_beads(&self) -> String {
        self.fields
            .lock()
            .unwrap()
            .target_number_of_beads
            .to_string()
    }

    /// Java `getNumberOfMarkers()`.
    pub fn get_number_of_markers(&self) -> String {
        self.fields.lock().unwrap().number_of_markers.to_string()
    }

    /// Java `setAutoFitRangeAndStep(boolean)`.
    pub fn set_auto_fit_range_and_step(&self, input: bool) {
        self.fields
            .lock()
            .unwrap()
            .auto_fit_range_and_step
            .set_boolean(input);
    }

    /// Java `getAutoFitRangeAndStep()`: a copy of the field.
    pub fn get_auto_fit_range_and_step(&self) -> ConstEtomoNumber {
        let f = self.fields.lock().unwrap();
        let number: &ConstEtomoNumber = &f.auto_fit_range_and_step;
        number.clone()
    }

    /// Java `setEnableStretching(boolean)`.
    pub fn set_enable_stretching(&self, input: bool) {
        self.fields
            .lock()
            .unwrap()
            .enable_stretching
            .set_boolean(input);
    }

    /// Java `getEnableStretching()`: a copy of the field.
    pub fn get_enable_stretching(&self) -> ConstEtomoNumber {
        let f = self.fields.lock().unwrap();
        let number: &ConstEtomoNumber = &f.enable_stretching;
        number.clone()
    }

    /// Java `setScaleToInteger(boolean)`.
    pub fn set_scale_to_integer(&self, input: bool) {
        self.fields
            .lock()
            .unwrap()
            .scale_to_integer
            .set_boolean(input);
    }

    /// Java `getScaleToInteger()`: a copy of the field.
    pub fn get_scale_to_integer(&self) -> ConstEtomoNumber {
        let f = self.fields.lock().unwrap();
        let number: &ConstEtomoNumber = &f.scale_to_integer;
        number.clone()
    }
}

impl Default for BatchRunTomoDatasetMetaData {
    fn default() -> Self {
        Self::new()
    }
}
