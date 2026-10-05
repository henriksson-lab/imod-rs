//! `IMOD/Etomo/src/etomo/type/JoinMetaData.java`.
//!
//! The Join interface's data file (`.ejf`) contents: the root name, the section table
//! rows, the sample and finish-join settings, the boundary table's adjusted values and
//! the auto-alignment parameters.
//!
//! **Representation.**  `JoinMetaData extends BaseMetaData implements
//! ConstJoinMetaData`: the superclass state is `base` (`BaseMetaDataBase`), the
//! abstract methods are the `BaseMetaData` trait, and the interface is
//! `ConstJoinMetaData`.  The object is owned by `JoinManager` and read by the join
//! comscript parameters (on the event dispatch thread) and stored from process
//! threads, so - like `MetaData` - each mutable field carries its own lock and every
//! method takes `&self`.  The section table is a list of shared, immutable rows
//! (`Arc<SectionTableRowData>`): the source only ever replaces rows, never edits one
//! held here.
//!
//! **`Transform`.**  `Transform.load/store/remove` are translated over a `HashMap`
//! (`transform.rs`); the properties are handed to them as one and the result written
//! back, as `auto_alignment_meta_data.rs` does.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::{Arc, LazyLock, Mutex};

use super::auto_alignment_meta_data::AutoAlignmentMetaData;
use super::axis_id::AxisID;
use super::axis_type::AxisType;
use super::base_meta_data::{self, BaseMetaData, BaseMetaDataBase};
use super::const_etomo_number::{ConstEtomoNumber, Number, Type};
use super::const_int_key_list::ConstIntKeyList;
use super::const_join_meta_data::ConstJoinMetaData;
use super::const_join_state::ConstJoinState;
use super::const_section_table_row_data::ConstSectionTableRowData;
use super::data_file_type::DataFileType;
use super::etomo_boolean2::EtomoBoolean2;
use super::etomo_number::EtomoNumber;
use super::etomo_version::EtomoVersion;
use super::image_filename_style::ImageFilenameStyle;
use super::int_key_list::IntKeyList;
use super::join_state::JoinState;
use super::null_required_number_exception::NullRequiredNumberException;
use super::script_parameter::ScriptParameter;
use super::section_table_row_data::SectionTableRowData;
use super::transform::Transform;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::makejoincom_param;
use crate::imod::etomo::storage::storable::{self, Storable};
use crate::imod::etomo::ui::log_properties::LogProperties;
use crate::imod::etomo::ui::swing::join_dialog;
use crate::imod::etomo::ui::swing::ui_harness;
use crate::imod::etomo::util::utilities;

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java private static final `ALIGN_TRANFORM_KEY` (deprecated).
const ALIGN_TRANFORM_KEY: &str = "AlignTransform";
// Version 1.0
/// Java private static final `fullLinearTransformationString` (deprecated).
const FULL_LINEAR_TRANSFORMATION_STRING: &str = "FullLinearTransformation";
/// Java private static final `rotationTranslationMagnificationString` (deprecated).
const ROTATION_TRANSLATION_MAGNIFICATION_STRING: &str = "RotationTranslationMagnification";
/// Java private static final `rotationTranslationString` (deprecated).
const ROTATION_TRANSLATION_STRING: &str = "RotationTranslation";

/// Java private static final `latestRevisionNumber =
/// EtomoVersion.getInstance(revisionNumberString, "1.2")`.
static LATEST_REVISION_NUMBER: LazyLock<EtomoVersion> = LazyLock::new(|| {
    EtomoVersion::get_instance(Some(base_meta_data::REVISION_NUMBER_STRING), Some("1.2"))
});
/// Java private static final `newJoinTitle`.
const NEW_JOIN_TITLE: &str = "New Join";

/// Java private static final `groupString`.
const GROUP_STRING: &str = "Join";
/// Java private static final `sectionTableDataSizeString`.
const SECTION_TABLE_DATA_SIZE_STRING: &str = "SectionTableDataSize";
/// Java private static final `rootNameString`.
const ROOT_NAME_STRING: &str = "RootName";
/// Java private static final `useAlignmentRefSectionString`.
const USE_ALIGNMENT_REF_SECTION_STRING: &str = "UseAlignmentRefSection";
/// Java private static final `REFINING_WITH_TRIAL_KEY` (unused in the source).
const REFINING_WITH_TRIAL_KEY: &str = "RefiningWithTrial";
/// Java private static final `MODEL_TRANFORM_KEY`.
const MODEL_TRANFORM_KEY: &str = "ModelTransform";
/// Java private static final `BOUNDARIES_TO_ANALYZE_KEY`.
const BOUNDARIES_TO_ANALYZE_KEY: &str = "BoundariesToAnalyze";
/// Java private static final `OBJECTS_TO_INCLUDE_KEY`.
const OBJECTS_TO_INCLUDE_KEY: &str = "ObjectsToInclude";
/// Java private static final `BOUNDARY_ROW_KEY` (unused in the source).
const BOUNDARY_ROW_KEY: &str = "BoundaryRow";

/// Java `public final class JoinMetaData extends BaseMetaData implements
/// ConstJoinMetaData`.
pub struct JoinMetaData {
    /// Java superclass `BaseMetaData` state.
    base: BaseMetaDataBase,
    /// Java private `sectionTableData`, initially null.
    section_table_data: Mutex<Option<Vec<Arc<SectionTableRowData>>>>,
    /// Java private `rootName`, initially "".
    root_name: Mutex<String>,
    /// Java private `boundariesToAnalyze`, initially null.
    boundaries_to_analyze: Mutex<Option<String>>,
    /// Java private `objectsToInclude`, initially null.
    objects_to_include: Mutex<Option<String>>,
    /// Java private `densityRefSection`.
    density_ref_section: Mutex<ScriptParameter>,
    /// Java private `sigmaLowFrequency` (deprecated).
    sigma_low_frequency: Mutex<ScriptParameter>,
    /// Java private `cutoffHighFrequency` (deprecated).
    cutoff_high_frequency: Mutex<ScriptParameter>,
    /// Java private `sigmaHighFrequency` (deprecated).
    sigma_high_frequency: Mutex<ScriptParameter>,
    /// Java private `alignTransform` (deprecated), initially `Transform.DEFAULT`.
    align_transform: Mutex<Transform>,
    /// Java private `modelTransform`, initially `Transform.DEFAULT`.
    model_transform: Mutex<Transform>,
    /// Java private `useAlignmentRefSection`, initially false.
    use_alignment_ref_section: Mutex<bool>,
    /// Java private `alignmentRefSection`.
    alignment_ref_section: Mutex<ScriptParameter>,
    /// Java private `sizeInX`.
    size_in_x: Mutex<ScriptParameter>,
    /// Java private `sizeInY`.
    size_in_y: Mutex<ScriptParameter>,
    /// Java private `shiftInX`.
    shift_in_x: Mutex<ScriptParameter>,
    /// Java private `shiftInY`.
    shift_in_y: Mutex<ScriptParameter>,
    /// Java private final `localFits` (FinishJoin -local; check box in the Join tab).
    local_fits: Mutex<EtomoBoolean2>,
    /// Java private `useEveryNSlices`.
    use_every_n_slices: Mutex<EtomoNumber>,
    /// Java private final `trialBinning`.
    trial_binning: Mutex<ScriptParameter>,
    /// Java private final `rejoinTrialBinning`.
    rejoin_trial_binning: Mutex<ScriptParameter>,
    /// Java private final `midasLimit`.
    midas_limit: Mutex<EtomoNumber>,
    /// Java private final `gap`.
    gap: Mutex<EtomoBoolean2>,
    /// Java private final `gapStart`.
    gap_start: Mutex<EtomoNumber>,
    /// Java private final `gapEnd`.
    gap_end: Mutex<EtomoNumber>,
    /// Java private final `gapInc`.
    gap_inc: Mutex<EtomoNumber>,
    /// Java private final `pointsToFitMin`.
    points_to_fit_min: Mutex<EtomoNumber>,
    /// Java private final `pointsToFitMax`.
    points_to_fit_max: Mutex<EtomoNumber>,
    /// Java private final `boundaryRowStartList`.
    boundary_row_start_list: Mutex<IntKeyList>,
    /// Java private final `boundaryRowEndList`.
    boundary_row_end_list: Mutex<IntKeyList>,
    /// Java private `rejoinUseEveryNSlices`.
    rejoin_use_every_n_slices: Mutex<EtomoNumber>,
    /// Java private `autoAlignmentMetaData`, a shared object (`getAutoAlignmentMetaData`
    /// hands it out to the dialog and to `XfalignParam`).
    auto_alignment_meta_data: Mutex<AutoAlignmentMetaData>,
    /// Java private final `manager`.
    manager: &'static dyn BaseManager,
}

// Safety: every field is a `Mutex` of owned data except the `&'static` references.
// `manager` is `Send + Sync` (the `BaseManager` trait requires it).  The one member
// that keeps the auto traits from applying is `BaseMetaDataBase`'s
// `Option<&'static dyn LogProperties>` (the log window, an EDT object), which
// `BaseMetaDataBase` only calls on the event dispatch thread; same argument as
// `meta_data.rs`.
unsafe impl Send for JoinMetaData {}
unsafe impl Sync for JoinMetaData {}

impl JoinMetaData {
    /// Java `JoinMetaData(BaseManager, LogProperties, boolean)`.
    pub fn new(
        manager: &'static dyn BaseManager,
        log_properties: Option<&'static dyn LogProperties>,
        new_dataset: bool,
    ) -> JoinMetaData {
        let mut density_ref_section =
            ScriptParameter::new_with_type_and_name(Type::Integer, "DensityRefSection");
        let mut alignment_ref_section =
            ScriptParameter::new_with_type_and_name(Type::Integer, "AlignmentRefSection");
        let mut trial_binning =
            ScriptParameter::new_with_type_and_name(Type::Integer, "TrialBinning");
        let mut rejoin_trial_binning =
            ScriptParameter::new_with_type_and_name(Type::Integer, "RejoinTrialBinning");
        let mut shift_in_x = ScriptParameter::new_with_type_and_name(Type::Integer, "ShiftInX");
        let mut shift_in_y = ScriptParameter::new_with_type_and_name(Type::Integer, "ShiftInY");
        let mut midas_limit = EtomoNumber::new_with_name("MidasLimit");
        let mut gap_start = EtomoNumber::new_with_name("GapStart");
        let mut gap_end = EtomoNumber::new_with_name("GapEnd");
        let mut gap_inc = EtomoNumber::new_with_name("GapInc");
        let mut gap = EtomoBoolean2::new_with_name("Gap");
        // super(manager, logProperties, false, newDataset, true)
        let base = BaseMetaDataBase::new_force_old_style(
            Some(manager),
            log_properties,
            false,
            new_dataset,
            true,
        );
        *base.axis_type.lock().unwrap() = AxisType::SingleAxis;
        *base.file_extension.lock().unwrap() = DataFileType::Join.extension().map(str::to_owned);
        density_ref_section
            .set_default_int(1)
            .use_default_as_display_value();
        alignment_ref_section
            .set_default_int(1)
            .use_default_as_display_value();
        trial_binning
            .set_default_int(1)
            .use_default_as_display_value();
        rejoin_trial_binning
            .set_default_int(1)
            .use_default_as_display_value();
        shift_in_x.set_default_int(0).use_default_as_display_value();
        shift_in_y.set_default_int(0).use_default_as_display_value();
        midas_limit.set_display_value_int(makejoincom_param::MIDAS_LIMIT_DEFAULT);
        gap_start.set_display_value_int(-4);
        gap_end.set_display_value_int(8);
        gap_inc.set_display_value_int(2);
        gap.set_display_value_boolean(true);
        JoinMetaData {
            base,
            section_table_data: Mutex::new(None),
            root_name: Mutex::new(String::new()),
            boundaries_to_analyze: Mutex::new(None),
            objects_to_include: Mutex::new(None),
            density_ref_section: Mutex::new(density_ref_section),
            sigma_low_frequency: Mutex::new(ScriptParameter::new_with_type_and_name(
                Type::Double,
                "SigmaLowFrequency",
            )),
            cutoff_high_frequency: Mutex::new(ScriptParameter::new_with_type_and_name(
                Type::Double,
                "CutoffHighFrequency",
            )),
            sigma_high_frequency: Mutex::new(ScriptParameter::new_with_type_and_name(
                Type::Double,
                "SigmaHighFrequency",
            )),
            align_transform: Mutex::new(Transform::DEFAULT),
            model_transform: Mutex::new(Transform::DEFAULT),
            use_alignment_ref_section: Mutex::new(false),
            alignment_ref_section: Mutex::new(alignment_ref_section),
            size_in_x: Mutex::new(ScriptParameter::new_with_type_and_name(
                Type::Integer,
                "SizeInX",
            )),
            size_in_y: Mutex::new(ScriptParameter::new_with_type_and_name(
                Type::Integer,
                "SizeInY",
            )),
            shift_in_x: Mutex::new(shift_in_x),
            shift_in_y: Mutex::new(shift_in_y),
            local_fits: Mutex::new(EtomoBoolean2::new_with_name("LocalFits")),
            use_every_n_slices: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Integer,
                "UseEveryNSlices",
            )),
            trial_binning: Mutex::new(trial_binning),
            rejoin_trial_binning: Mutex::new(rejoin_trial_binning),
            midas_limit: Mutex::new(midas_limit),
            gap: Mutex::new(gap),
            gap_start: Mutex::new(gap_start),
            gap_end: Mutex::new(gap_end),
            gap_inc: Mutex::new(gap_inc),
            points_to_fit_min: Mutex::new(EtomoNumber::new_with_name("PointsToFitMin")),
            points_to_fit_max: Mutex::new(EtomoNumber::new_with_name("PointsToFitMax")),
            boundary_row_start_list: Mutex::new(IntKeyList::get_number_instance_with_key(
                &format!("{}.{}", "BoundaryRow", "StartList"),
            )),
            boundary_row_end_list: Mutex::new(IntKeyList::get_number_instance_with_key(&format!(
                "{}.{}",
                "BoundaryRow", "EndList"
            ))),
            rejoin_use_every_n_slices: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Integer,
                "RejoinUseEveryNSlices",
            )),
            auto_alignment_meta_data: Mutex::new(AutoAlignmentMetaData::new()),
            manager,
        }
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
        *self.section_table_data.lock().unwrap() = None;
        self.density_ref_section.lock().unwrap().reset();
        *self.root_name.lock().unwrap() = String::new();
        *self.boundaries_to_analyze.lock().unwrap() = None;
        *self.objects_to_include.lock().unwrap() = None;
        self.gap_start.lock().unwrap().reset();
        self.gap_end.lock().unwrap().reset();
        self.gap_inc.lock().unwrap().reset();
        self.points_to_fit_min.lock().unwrap().reset();
        self.points_to_fit_max.lock().unwrap().reset();
        *self.model_transform.lock().unwrap() = Transform::DEFAULT;
        *self.use_alignment_ref_section.lock().unwrap() = false;
        self.alignment_ref_section.lock().unwrap().reset();
        self.size_in_x.lock().unwrap().reset();
        self.size_in_y.lock().unwrap().reset();
        self.shift_in_x.lock().unwrap().reset();
        self.shift_in_y.lock().unwrap().reset();
        self.local_fits.lock().unwrap().reset();
        self.local_fits.lock().unwrap().reset();
        self.trial_binning.lock().unwrap().reset();
        self.rejoin_trial_binning.lock().unwrap().reset();
        self.gap.lock().unwrap().reset();
        self.boundary_row_start_list.lock().unwrap().reset();
        self.boundary_row_end_list.lock().unwrap().reset();
        // load
        let prepend = self
            .create_prepend(prepend)
            .unwrap_or_else(|| "null".to_string());
        let group = format!("{}.", prepend);
        self.base.revision_number.lock().unwrap().reset();
        storable::StorableValue::load_with_prepend(
            &mut *self.base.revision_number.lock().unwrap(),
            props,
            &prepend,
        );
        self.auto_alignment_meta_data
            .lock()
            .unwrap()
            .load(props, &prepend);
        let hash_props: HashMap<String, String> =
            props.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        if self
            .base
            .revision_number
            .lock()
            .unwrap()
            .le_string(Some("1.1"))
        {
            self.sigma_low_frequency.lock().unwrap().reset();
            self.cutoff_high_frequency.lock().unwrap().reset();
            self.sigma_high_frequency.lock().unwrap().reset();
            *self.align_transform.lock().unwrap() = Transform::DEFAULT;
            self.sigma_low_frequency
                .lock()
                .unwrap()
                .load_with_prepend(props, Some(&prepend));
            {
                let sigma_low_frequency = self.sigma_low_frequency.lock().unwrap();
                self.auto_alignment_meta_data
                    .lock()
                    .unwrap()
                    .set_sigma_low_frequency(Some(&sigma_low_frequency));
            }
            self.cutoff_high_frequency
                .lock()
                .unwrap()
                .load_with_prepend(props, Some(&prepend));
            {
                let cutoff_high_frequency = self.cutoff_high_frequency.lock().unwrap();
                self.auto_alignment_meta_data
                    .lock()
                    .unwrap()
                    .set_cutoff_high_frequency(Some(&cutoff_high_frequency));
            }
            self.sigma_high_frequency
                .lock()
                .unwrap()
                .load_with_prepend(props, Some(&prepend));
            {
                let sigma_high_frequency = self.sigma_high_frequency.lock().unwrap();
                self.auto_alignment_meta_data
                    .lock()
                    .unwrap()
                    .set_sigma_high_frequency(Some(&sigma_high_frequency));
            }
            if self
                .base
                .revision_number
                .lock()
                .unwrap()
                .le_string(Some("1.0"))
            {
                // handling version 1.0
                self.load_version1_0(props, &prepend);
            } else {
                self.auto_alignment_meta_data
                    .lock()
                    .unwrap()
                    .set_align_transform(Transform::load(
                        &hash_props,
                        &prepend,
                        ALIGN_TRANFORM_KEY,
                        Transform::DEFAULT,
                    ));
            }
        }
        *self.model_transform.lock().unwrap() = Transform::load(
            &hash_props,
            &prepend,
            MODEL_TRANFORM_KEY,
            Transform::DEFAULT,
        );
        *self.root_name.lock().unwrap() = props
            .get(&format!("{}{}", group, ROOT_NAME_STRING))
            .cloned()
            .unwrap_or_default();
        *self.boundaries_to_analyze.lock().unwrap() = props
            .get(&format!("{}{}", group, BOUNDARIES_TO_ANALYZE_KEY))
            .cloned();
        *self.objects_to_include.lock().unwrap() = props
            .get(&format!("{}{}", group, OBJECTS_TO_INCLUDE_KEY))
            .cloned();
        self.gap_start
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.gap
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.gap_end
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.gap_inc
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.points_to_fit_min
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.points_to_fit_max
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.density_ref_section
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        // `Boolean.valueOf(String)`: true only for "true", ignoring case.
        *self.use_alignment_ref_section.lock().unwrap() = props
            .get(&format!("{}{}", group, USE_ALIGNMENT_REF_SECTION_STRING))
            .map(String::as_str)
            .unwrap_or("false")
            .eq_ignore_ascii_case("true");
        self.alignment_ref_section
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.size_in_x
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.size_in_y
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.shift_in_x
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.shift_in_y
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.local_fits
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.use_every_n_slices
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.trial_binning
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.rejoin_trial_binning
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.midas_limit
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));

        // `Integer.parseInt(props.getProperty(group + sectionTableDataSizeString,
        // "-1"))`.
        //
        // Fixed in translation: an unparsable size throws NumberFormatException out of
        // the load in the source; it reads as no rows here.
        let section_table_rows_size = props
            .get(&format!("{}{}", group, SECTION_TABLE_DATA_SIZE_STRING))
            .map(String::as_str)
            .unwrap_or("-1")
            .parse::<i32>()
            .unwrap_or(-1);
        if section_table_rows_size < 1 {
            return;
        }
        let mut section_table_data: Vec<Arc<SectionTableRowData>> =
            Vec::with_capacity(section_table_rows_size as usize);
        for i in 0..section_table_rows_size {
            let mut row = SectionTableRowData::new(self.manager, i + 1);
            row.load_with_prepend(props, Some(&prepend));
            let row_index = row.get_row_index();
            if row_index < 0 {
                ui_harness::open_message_dialog_from_process(
                    Some(self.manager),
                    &format!(
                        "Invalid row index: {}.  Corrupted: {} file.",
                        row_index,
                        DataFileType::Join.extension().unwrap_or("null")
                    ),
                    "Corrupted File",
                    Some(AxisID::Only),
                );
            }
            // Fixed in translation: `sectionTableData.add(row.getRowIndex(), row)`
            // throws IndexOutOfBoundsException for a negative index or one past the end
            // of the list (a corrupted file); such a row is appended here.
            let row_index = row.get_row_index();
            if row_index >= 0 && (row_index as usize) <= section_table_data.len() {
                section_table_data.insert(row_index as usize, Arc::new(row));
            } else {
                section_table_data.push(Arc::new(row));
            }
        }
        *self.section_table_data.lock().unwrap() = Some(section_table_data);
        self.boundary_row_start_list
            .lock()
            .unwrap()
            .load(props, &prepend);
        self.boundary_row_end_list
            .lock()
            .unwrap()
            .load(props, &prepend);
        self.rejoin_use_every_n_slices
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
    }

    /// Java private `loadVersion1_0(Properties, String)`.
    fn load_version1_0(&self, props: &BTreeMap<String, String>, prepend: &str) {
        let group = format!("{}.", prepend);
        let property_is_true = |key: &str| {
            props
                .get(&format!("{}{}", group, key))
                .map(String::as_str)
                .unwrap_or("false")
                .eq_ignore_ascii_case("true")
        };
        let align_transform = if property_is_true(FULL_LINEAR_TRANSFORMATION_STRING) {
            Transform::FullLinearTransformation
        } else if property_is_true(ROTATION_TRANSLATION_MAGNIFICATION_STRING) {
            Transform::RotationTranslationMagnification
        } else if property_is_true(ROTATION_TRANSLATION_STRING) {
            Transform::FullLinearTransformation
        } else {
            Transform::DEFAULT
        };
        *self.align_transform.lock().unwrap() = align_transform;
        self.auto_alignment_meta_data
            .lock()
            .unwrap()
            .set_align_transform(align_transform);
    }

    /// Java `setDensityRefSection(Object)` (an `Integer`).
    pub fn set_density_ref_section(&self, density_ref_section: Option<Number>) {
        self.density_ref_section
            .lock()
            .unwrap()
            .set_number(density_ref_section);
    }

    /// Java `setUseEveryNSlices(Object)` (an `Integer`).
    pub fn set_use_every_n_slices(&self, use_every_n_slices: Option<Number>) {
        self.use_every_n_slices
            .lock()
            .unwrap()
            .set_number(use_every_n_slices);
    }

    /// Java `setRejoinUseEveryNSlices(Object)` (an `Integer`).
    pub fn set_rejoin_use_every_n_slices(&self, rejoin_use_every_n_slices: Option<Number>) {
        self.rejoin_use_every_n_slices
            .lock()
            .unwrap()
            .set_number(rejoin_use_every_n_slices);
    }

    /// Java `setTrialBinning(Object)` (an `Integer`).
    pub fn set_trial_binning(&self, trial_binning: Option<Number>) {
        self.trial_binning.lock().unwrap().set_number(trial_binning);
    }

    /// Java `setRejoinTrialBinning(Object)` (an `Integer`).
    pub fn set_rejoin_trial_binning(&self, rejoin_trial_binning: Option<Number>) {
        self.rejoin_trial_binning
            .lock()
            .unwrap()
            .set_number(rejoin_trial_binning);
    }

    /// Java `setGap(boolean)`.
    pub fn set_gap(&self, gap: bool) {
        self.gap.lock().unwrap().set_boolean(gap);
    }

    /// Java `setRootName(String)`.
    pub fn set_root_name(&self, root_name: Option<&str>) {
        // Fixed in translation: a null root name (never passed by the source's
        // callers) would make the later `rootName.equals("")` tests throw; it is
        // stored as "" here.
        *self.root_name.lock().unwrap() = root_name.unwrap_or("").to_owned();
        let root_name = self.root_name.lock().unwrap().clone();
        utilities::manager_stamp(None, Some(&root_name));
    }

    /// Java `setBoundariesToAnalyze(String)`.
    pub fn set_boundaries_to_analyze(&self, boundaries_to_analyze: Option<&str>) {
        *self.boundaries_to_analyze.lock().unwrap() = boundaries_to_analyze.map(str::to_owned);
    }

    /// Java `setObjectsToInclude(String)`.
    pub fn set_objects_to_include(&self, objects_to_include: Option<&str>) {
        *self.objects_to_include.lock().unwrap() = objects_to_include.map(str::to_owned);
    }

    /// Java `setGapStart(String)`.
    pub fn set_gap_start(&self, gap_start: Option<&str>) {
        self.gap_start.lock().unwrap().set_string(gap_start);
    }

    /// Java `setPointsToFitMax(String)`.
    pub fn set_points_to_fit_max(&self, points_to_fit_max: Option<&str>) {
        self.points_to_fit_max
            .lock()
            .unwrap()
            .set_string(points_to_fit_max);
    }

    /// Java `setPointsToFitMin(String)`.
    pub fn set_points_to_fit_min(&self, points_to_fit_min: Option<&str>) {
        self.points_to_fit_min
            .lock()
            .unwrap()
            .set_string(points_to_fit_min);
    }

    /// Java `setLocalFits(boolean)`.
    pub fn set_local_fits(&self, input: bool) {
        self.local_fits.lock().unwrap().set_boolean(input);
    }

    /// Java `setGapEnd(String)`.
    pub fn set_gap_end(&self, gap_end: Option<&str>) {
        self.gap_end.lock().unwrap().set_string(gap_end);
    }

    /// Java `setGapInc(String)`.
    pub fn set_gap_inc(&self, gap_inc: Option<&str>) {
        self.gap_inc.lock().unwrap().set_string(gap_inc);
    }

    /// Java `resetSectionTableData()`.
    pub fn reset_section_table_data(&self) {
        *self.section_table_data.lock().unwrap() = None;
    }

    /// Java `setSectionTableData(ConstSectionTableRowData)`: appends the row.
    pub fn set_section_table_data(&self, row: SectionTableRowData) {
        let mut section_table_data = self.section_table_data.lock().unwrap();
        if section_table_data.is_none() {
            *section_table_data = Some(Vec::new());
        }
        section_table_data.as_mut().unwrap().push(Arc::new(row));
    }

    /// Java `setMidasLimit(String)`.
    pub fn set_midas_limit(&self, midas_limit: Option<&str>) {
        self.midas_limit.lock().unwrap().set_string(midas_limit);
    }

    /// Java `setUseAlignmentRefSection(boolean)`.
    pub fn set_use_alignment_ref_section(&self, use_alignment_ref_section: bool) {
        *self.use_alignment_ref_section.lock().unwrap() = use_alignment_ref_section;
    }

    /// Java `setAlignmentRefSection(Object)` (an `Integer`).
    pub fn set_alignment_ref_section(&self, alignment_ref_section: Option<Number>) {
        self.alignment_ref_section
            .lock()
            .unwrap()
            .set_number(alignment_ref_section);
    }

    /// Java `setModelTransform(Transform)`.
    pub fn set_model_transform(&self, model_transform: Transform) {
        *self.model_transform.lock().unwrap() = model_transform;
    }

    /// Java `setSizeInX(String)`; returns a copy of the field.
    pub fn set_size_in_x(&self, size_in_x: Option<&str>) -> ConstEtomoNumber {
        let mut field = self.size_in_x.lock().unwrap();
        field.set_string(size_in_x);
        let number: &ConstEtomoNumber = &field;
        number.clone()
    }

    /// Java `setSizeInY(String)`.
    pub fn set_size_in_y(&self, size_in_y: Option<&str>) {
        self.size_in_y.lock().unwrap().set_string(size_in_y);
    }

    /// Java `setShiftInX(String)`.
    pub fn set_shift_in_x(&self, shift_in_x: Option<&str>) {
        self.shift_in_x.lock().unwrap().set_string(shift_in_x);
    }

    /// Java `setShiftInY(String)`.
    pub fn set_shift_in_y(&self, shift_in_y: Option<&str>) {
        self.shift_in_y.lock().unwrap().set_string(shift_in_y);
    }

    /// Java private `removeVersion1_0(Properties, String)`.  Remove data not used after
    /// version 1.0 of join meta data.
    fn remove_version1_0(&self, props: &mut BTreeMap<String, String>, prepend: &str) {
        let group = format!("{}.", prepend);
        props.remove(&format!("{}{}", group, FULL_LINEAR_TRANSFORMATION_STRING));
        props.remove(&format!(
            "{}{}",
            group, ROTATION_TRANSLATION_MAGNIFICATION_STRING
        ));
        props.remove(&format!("{}{}", group, ROTATION_TRANSLATION_STRING));
    }

    /// Java `store(Properties, String)`.
    pub fn store_with_prepend(&self, props: &mut BTreeMap<String, String>, prepend: &str) {
        // super.store(props, prepend)
        let created = self.create_prepend(prepend);
        self.base
            .store_with_created_prepend(props, created.as_deref());
        self.remove_section_table_data(props, prepend);
        let prepend = self
            .create_prepend(prepend)
            .unwrap_or_else(|| "null".to_string());
        let group = format!("{}.", prepend);
        // removing data used in old versions of join meta data
        // change this this when there are more then one old version
        if self
            .base
            .revision_number
            .lock()
            .unwrap()
            .le_string(Some("1.1"))
        {
            self.sigma_low_frequency
                .lock()
                .unwrap()
                .remove_with_prepend(props, Some(&prepend));
            self.cutoff_high_frequency
                .lock()
                .unwrap()
                .remove_with_prepend(props, Some(&prepend));
            self.sigma_high_frequency
                .lock()
                .unwrap()
                .remove_with_prepend(props, Some(&prepend));
            if self
                .base
                .revision_number
                .lock()
                .unwrap()
                .le_string(Some("1.0"))
            {
                self.remove_version1_0(props, &prepend);
            } else {
                let mut hash_props: HashMap<String, String> =
                    props.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
                Transform::remove(&mut hash_props, &prepend, ALIGN_TRANFORM_KEY);
                props.retain(|key, _| hash_props.contains_key(key));
            }
        }
        storable::StorableValue::store_with_prepend(&*LATEST_REVISION_NUMBER, props, &prepend);
        props.insert(
            format!("{}{}", group, ROOT_NAME_STRING),
            self.root_name.lock().unwrap().clone(),
        );
        match self.boundaries_to_analyze.lock().unwrap().as_ref() {
            None => {
                props.remove(&format!("{}{}", group, BOUNDARIES_TO_ANALYZE_KEY));
            }
            Some(boundaries_to_analyze) => {
                props.insert(
                    format!("{}{}", group, BOUNDARIES_TO_ANALYZE_KEY),
                    boundaries_to_analyze.clone(),
                );
            }
        }
        match self.objects_to_include.lock().unwrap().as_ref() {
            None => {
                props.remove(&format!("{}{}", group, OBJECTS_TO_INCLUDE_KEY));
            }
            Some(objects_to_include) => {
                props.insert(
                    format!("{}{}", group, OBJECTS_TO_INCLUDE_KEY),
                    objects_to_include.clone(),
                );
            }
        }
        self.gap_start
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.gap_end
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.gap_inc
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.points_to_fit_min
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.points_to_fit_max
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.density_ref_section
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        match self.section_table_data.lock().unwrap().as_ref() {
            None => {
                props.insert(
                    format!("{}{}", group, SECTION_TABLE_DATA_SIZE_STRING),
                    "0".to_string(),
                );
            }
            Some(section_table_data) => {
                props.insert(
                    format!("{}{}", group, SECTION_TABLE_DATA_SIZE_STRING),
                    section_table_data.len().to_string(),
                );
            }
        }
        self.auto_alignment_meta_data
            .lock()
            .unwrap()
            .store(props, &prepend);
        let mut hash_props: HashMap<String, String> = HashMap::new();
        Transform::store(
            Some(*self.model_transform.lock().unwrap()),
            &mut hash_props,
            &prepend,
            MODEL_TRANFORM_KEY,
        );
        for (key, value) in hash_props {
            props.insert(key, value);
        }
        props.insert(
            format!("{}{}", group, USE_ALIGNMENT_REF_SECTION_STRING),
            self.use_alignment_ref_section.lock().unwrap().to_string(),
        );
        self.alignment_ref_section
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.size_in_x
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.size_in_y
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.shift_in_x
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.shift_in_y
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.local_fits
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.use_every_n_slices
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.trial_binning
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.rejoin_trial_binning
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.gap
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.midas_limit
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        if let Some(section_table_data) = self.section_table_data.lock().unwrap().as_ref() {
            for row in section_table_data.iter() {
                ConstSectionTableRowData::store_with_prepend(&**row, props, Some(&prepend));
            }
        }
        self.boundary_row_start_list
            .lock()
            .unwrap()
            .store(props, &prepend);
        self.boundary_row_end_list
            .lock()
            .unwrap()
            .store(props, &prepend);
        self.rejoin_use_every_n_slices
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
    }

    /// Java `removeSectionTableData(Properties, String)`.
    pub fn remove_section_table_data(&self, props: &mut BTreeMap<String, String>, prepend: &str) {
        let prepend = self
            .create_prepend(prepend)
            .unwrap_or_else(|| "null".to_string());
        // Java's unused local `group`.
        let _group = format!("{}.", prepend);
        if let Some(section_table_data) = self.section_table_data.lock().unwrap().as_ref() {
            for row in section_table_data.iter() {
                row.remove(props, Some(&prepend));
            }
        }
    }

    /// Java `isValid(String)`.
    pub fn is_valid_string(&self, working_dir_name: Option<&str>) -> bool {
        let Some(working_dir_name) = working_dir_name.filter(|name| {
            !name.is_empty()
                && !name
                    .chars()
                    .all(|c| matches!(c, ' ' | '\t' | '\n' | '\x0B' | '\x0C' | '\r'))
        }) else {
            *self.base.invalid_reason.lock().unwrap() = "Working directory is not set.".to_string();
            return false;
        };
        self.is_valid_file(Some(Path::new(working_dir_name)))
    }

    /// Java `isValid(File)`.
    pub fn is_valid_file(&self, working_dir: Option<&Path>) -> bool {
        let mut invalid_buffer = String::new();
        if !utilities::is_valid_file(
            working_dir,
            Some(join_dialog::WORKING_DIRECTORY_TEXT),
            &mut invalid_buffer,
            true,
            true,
            true,
            true,
        ) {
            *self.base.invalid_reason.lock().unwrap() = invalid_buffer;
            return false;
        }
        BaseMetaData::is_valid(self)
    }

    /// Java `equals(Object)`.
    ///
    /// Fixed in translation (JoinMetaData.java:532-545): the source dereferences both
    /// section tables when both are null (NullPointerException), and compares each of
    /// its own rows with itself (`sectionTableData.get(i).equals(
    /// sectionTableData.get(i))`) instead of with the other object's row.  Here two
    /// null tables are equal and each row is compared with the other's.
    pub fn equals(&self, that: &JoinMetaData) -> bool {
        if !self.base.equals(&that.base) {
            return false;
        }
        let section_table_data = self.section_table_data.lock().unwrap().clone();
        let that_section_table_data = that.section_table_data.lock().unwrap().clone();
        match (section_table_data, that_section_table_data) {
            (None, None) => true,
            (None, Some(_)) | (Some(_), None) => false,
            (Some(section_table_data), Some(that_section_table_data)) => {
                if section_table_data.len() != that_section_table_data.len() {
                    return false;
                }
                for i in 0..section_table_data.len() {
                    if !section_table_data[i].equals(&*that_section_table_data[i]) {
                        return false;
                    }
                }
                true
            }
        }
    }

    /// Java `isRootNameSet()`.
    pub fn is_root_name_set(&self) -> bool {
        let root_name = self.root_name.lock().unwrap();
        !root_name.is_empty()
            && !root_name
                .chars()
                .any(|c| matches!(c, ' ' | '\t' | '\n' | '\x0B' | '\x0C' | '\r'))
    }

    /// Java `setBoundaryRowStart(int, String)`.
    pub fn set_boundary_row_start(&self, key: i32, start: Option<&str>) {
        self.boundary_row_start_list
            .lock()
            .unwrap()
            .put_string(key, start);
    }

    /// Java `resetBoundaryRowStartList()`.
    pub fn reset_boundary_row_start_list(&self) {
        self.boundary_row_start_list.lock().unwrap().reset();
    }

    /// Java `resetBoundaryRowEndList()`.
    pub fn reset_boundary_row_end_list(&self) {
        self.boundary_row_end_list.lock().unwrap().reset();
    }

    /// Java `setBoundaryRowEnd(int, String)`.
    pub fn set_boundary_row_end(&self, key: i32, end: Option<&str>) {
        self.boundary_row_end_list
            .lock()
            .unwrap()
            .put_string(key, end);
    }

    /// Java static `getNewFileTitle()`.
    pub fn get_new_file_title() -> &'static str {
        NEW_JOIN_TITLE
    }

    /// Java static `getSize(int, int)`.
    pub fn get_size(min: i32, max: i32) -> i32 {
        max.wrapping_sub(min).wrapping_add(1)
    }

    /// Java `getFileExtension()` (the inherited `fileExtension`).
    pub fn get_file_extension(&self) -> Option<String> {
        self.base.get_file_extension()
    }

    /// Java `getInvalidReason()` (inherited).
    pub fn get_invalid_reason(&self) -> String {
        self.base.get_invalid_reason()
    }
}

impl ConstJoinMetaData for JoinMetaData {
    fn get_image_filename_style(&self) -> ImageFilenameStyle {
        self.base.get_image_filename_style()
    }

    fn get_alignment_ref_section(&self) -> ConstEtomoNumber {
        let guard = self.alignment_ref_section.lock().unwrap();
        let number: &ConstEtomoNumber = &guard;
        number.clone()
    }

    fn get_boundaries_to_analyze(&self) -> Option<String> {
        self.boundaries_to_analyze.lock().unwrap().clone()
    }

    /// Java `getCoordinate(ConstEtomoNumber, JoinState) throws
    /// NullRequiredNumberException`.
    fn get_coordinate(
        &self,
        coordinate: &ConstEtomoNumber,
        state: &JoinState,
    ) -> Result<i32, NullRequiredNumberException> {
        let binning = state.get_join_trial_binning();
        if coordinate.is_null() {
            return Err(NullRequiredNumberException::new("Coordinate is null"));
        }
        if binning.is_null() {
            return Err(NullRequiredNumberException::new("Binning is null"));
        }
        Ok(coordinate.get_int().wrapping_mul(binning.get_int()))
    }

    fn get_dataset_name(&self) -> String {
        self.root_name.lock().unwrap().clone()
    }

    fn get_density_ref_section(&self) -> ConstEtomoNumber {
        let guard = self.density_ref_section.lock().unwrap();
        let number: &ConstEtomoNumber = &guard;
        number.clone()
    }

    fn is_use_alignment_ref_section(&self) -> bool {
        *self.use_alignment_ref_section.lock().unwrap()
    }

    fn get_shift_in_x(&self) -> ConstEtomoNumber {
        let guard = self.shift_in_x.lock().unwrap();
        let number: &ConstEtomoNumber = &guard;
        number.clone()
    }

    fn get_size_in_x(&self) -> ConstEtomoNumber {
        let guard = self.size_in_x.lock().unwrap();
        let number: &ConstEtomoNumber = &guard;
        number.clone()
    }

    fn get_shift_in_y(&self) -> ConstEtomoNumber {
        let guard = self.shift_in_y.lock().unwrap();
        let number: &ConstEtomoNumber = &guard;
        number.clone()
    }

    fn get_size_in_y(&self) -> ConstEtomoNumber {
        let guard = self.size_in_y.lock().unwrap();
        let number: &ConstEtomoNumber = &guard;
        number.clone()
    }

    fn is_local_fits(&self) -> bool {
        self.local_fits.lock().unwrap().is()
    }

    fn get_use_every_n_slices(&self) -> ConstEtomoNumber {
        let guard = self.use_every_n_slices.lock().unwrap();
        let number: &ConstEtomoNumber = &guard;
        number.clone()
    }

    fn get_rejoin_use_every_n_slices(&self) -> ConstEtomoNumber {
        let guard = self.rejoin_use_every_n_slices.lock().unwrap();
        let number: &ConstEtomoNumber = &guard;
        number.clone()
    }

    fn get_trial_binning(&self) -> ConstEtomoNumber {
        let guard = self.trial_binning.lock().unwrap();
        let number: &ConstEtomoNumber = &guard;
        number.clone()
    }

    fn get_model_transform(&self) -> Transform {
        *self.model_transform.lock().unwrap()
    }

    fn get_midas_limit(&self) -> ConstEtomoNumber {
        let guard = self.midas_limit.lock().unwrap();
        let number: &ConstEtomoNumber = &guard;
        number.clone()
    }

    fn get_objects_to_include(&self) -> Option<String> {
        self.objects_to_include.lock().unwrap().clone()
    }

    fn get_gap(&self) -> EtomoBoolean2 {
        self.gap.lock().unwrap().clone()
    }

    fn get_gap_start(&self) -> ConstEtomoNumber {
        let guard = self.gap_start.lock().unwrap();
        let number: &ConstEtomoNumber = &guard;
        number.clone()
    }

    fn get_gap_end(&self) -> ConstEtomoNumber {
        let guard = self.gap_end.lock().unwrap();
        let number: &ConstEtomoNumber = &guard;
        number.clone()
    }

    fn get_gap_inc(&self) -> ConstEtomoNumber {
        let guard = self.gap_inc.lock().unwrap();
        let number: &ConstEtomoNumber = &guard;
        number.clone()
    }

    fn get_points_to_fit_min(&self) -> ConstEtomoNumber {
        let guard = self.points_to_fit_min.lock().unwrap();
        let number: &ConstEtomoNumber = &guard;
        number.clone()
    }

    fn get_points_to_fit_max(&self) -> ConstEtomoNumber {
        let guard = self.points_to_fit_max.lock().unwrap();
        let number: &ConstEtomoNumber = &guard;
        number.clone()
    }

    fn get_rejoin_trial_binning(&self) -> ConstEtomoNumber {
        let guard = self.rejoin_trial_binning.lock().unwrap();
        let number: &ConstEtomoNumber = &guard;
        number.clone()
    }

    fn get_boundary_row_end(&self, key: i32) -> Option<ConstEtomoNumber> {
        self.boundary_row_end_list
            .lock()
            .unwrap()
            .get_etomo_number(key)
            .map(|number| number.base)
    }

    fn is_boundary_row_end_list_empty(&self) -> bool {
        ConstIntKeyList::is_empty(&*self.boundary_row_end_list.lock().unwrap())
    }

    fn get_section_table_data(&self) -> Option<Vec<Arc<SectionTableRowData>>> {
        self.section_table_data.lock().unwrap().clone()
    }

    fn get_boundary_row_end_list(&self) -> IntKeyList {
        self.boundary_row_end_list.lock().unwrap().clone()
    }

    fn get_boundary_row_start_list(&self) -> IntKeyList {
        self.boundary_row_start_list.lock().unwrap().clone()
    }

    fn get_size_in_x_parameter(&self) -> ScriptParameter {
        self.size_in_x.lock().unwrap().clone()
    }

    fn get_size_in_y_parameter(&self) -> ScriptParameter {
        self.size_in_y.lock().unwrap().clone()
    }

    fn get_shift_in_x_parameter(&self) -> ScriptParameter {
        self.shift_in_x.lock().unwrap().clone()
    }

    fn get_shift_in_y_parameter(&self) -> ScriptParameter {
        self.shift_in_y.lock().unwrap().clone()
    }

    fn get_rejoin_trial_binning_parameter(&self) -> ScriptParameter {
        self.rejoin_trial_binning.lock().unwrap().clone()
    }

    fn get_trial_binning_parameter(&self) -> ScriptParameter {
        self.trial_binning.lock().unwrap().clone()
    }

    fn get_auto_alignment_meta_data(&self) -> &Mutex<AutoAlignmentMetaData> {
        &self.auto_alignment_meta_data
    }

    fn get_name(&self) -> String {
        let root_name = self.root_name.lock().unwrap();
        if root_name.is_empty() {
            return NEW_JOIN_TITLE.to_string();
        }
        root_name.clone()
    }

    fn get_density_ref_section_parameter(&self) -> ScriptParameter {
        self.density_ref_section.lock().unwrap().clone()
    }
}

impl Storable for JoinMetaData {
    /// Java `store(Properties)`, inherited from `BaseMetaData`.
    fn store(&self, properties: &mut BTreeMap<String, String>) {
        self.store_with_prepend(properties, "");
    }

    /// Java `store(Properties, String)`.
    fn store_with_prepend(&self, properties: &mut BTreeMap<String, String>, prepend: &str) {
        JoinMetaData::store_with_prepend(self, properties, prepend);
    }

    /// Java `load(Properties)`: `load(props, "")`.
    fn load(&self, properties: &mut BTreeMap<String, String>) {
        self.load_with_prepend(properties, "");
    }

    /// Java `load(Properties, String)`.
    fn load_with_prepend(&self, properties: &mut BTreeMap<String, String>, prepend: &str) {
        JoinMetaData::load_with_prepend(self, properties, prepend);
    }
}

impl BaseMetaData for JoinMetaData {
    fn base(&self) -> &BaseMetaDataBase {
        &self.base
    }

    /// Java `getMetaDataFileName()`.
    fn get_meta_data_file_name(&self) -> Option<String> {
        let root_name = self.root_name.lock().unwrap();
        if root_name.is_empty() {
            return None;
        }
        Some(format!(
            "{}{}",
            root_name,
            self.base
                .get_file_extension()
                .unwrap_or_else(|| "null".to_string())
        ))
    }

    /// Java `getName()`.
    fn get_name(&self) -> Option<String> {
        Some(ConstJoinMetaData::get_name(self))
    }

    /// Java `getDatasetName()`.
    fn get_dataset_name(&self) -> Option<String> {
        Some(self.root_name.lock().unwrap().clone())
    }

    /// Java `isValid()`.
    fn is_valid(&self) -> bool {
        // `rootName.matches("\\S+")`
        let valid = {
            let root_name = self.root_name.lock().unwrap();
            !root_name.is_empty()
                && !root_name
                    .chars()
                    .any(|c| matches!(c, ' ' | '\t' | '\n' | '\x0B' | '\x0C' | '\r'))
        };
        if !valid {
            *self.base.invalid_reason.lock().unwrap() = format!("{} is empty.", ROOT_NAME_STRING);
            return false;
        }
        true
    }

    /// Java package-private `getGroupKey()`.
    fn get_group_key(&self) -> Option<String> {
        Some(GROUP_STRING.to_string())
    }
}

/// Java `toString()`.  `super.toString()` is `BaseMetaData`'s.
impl std::fmt::Display for JoinMetaData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[rootName:{},{}]",
            self.root_name.lock().unwrap(),
            self.base
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn size_is_inclusive() {
        assert_eq!(JoinMetaData::get_size(3, 10), 8);
        assert_eq!(JoinMetaData::get_new_file_title(), "New Join");
    }
}
