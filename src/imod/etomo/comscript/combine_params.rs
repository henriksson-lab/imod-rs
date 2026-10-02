//! `IMOD/Etomo/src/etomo/comscript/CombineParams.java`.

use std::collections::BTreeMap;
use std::sync::LazyLock;
use std::sync::Mutex;

use regex::Regex;

use super::const_combine_params::ConstCombineParams;
use super::matchorwarp_param::MatchorwarpParam;
use super::string_list::StringList;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::storage::storable::StorableValue;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::combine_patch_size::CombinePatchSize;
use crate::imod::etomo::r#type::const_etomo_number::{
    ConstEtomoNumber, Type, java_lang_integer_parse_int,
};
use crate::imod::etomo::r#type::const_string_property::ConstStringProperty;
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::r#type::fiducial_match::FiducialMatch;
use crate::imod::etomo::r#type::match_mode::MatchMode;
use crate::imod::etomo::r#type::meta_data;
use crate::imod::etomo::r#type::string_property::StringProperty;
use crate::imod::etomo::util::dataset_files;
use crate::imod::etomo::util::mrc_header::MRCHeader;
use crate::imod::etomo::util::utilities;

/// Java private `MATCH_B_TO_A_KEY`.
const MATCH_B_TO_A_KEY: &str = "MatchBtoA";
/// Java private `MATCH_MODE_KEY`.
const MATCH_MODE_KEY: &str = "MatchMode";
/// Java private `DIALOG_MATCH_MODE_KEY`.
const DIALOG_MATCH_MODE_KEY: &str = "DialogMatchMode";
/// Java `PATCH_Z_MIN_LABEL`.
pub const PATCH_Z_MIN_LABEL: &str = "Z axis min";
/// Java `PATCH_Z_MAX_LABEL`.
pub const PATCH_Z_MAX_LABEL: &str = "Z axis max";
/// Java private `AUTO_PATCH_FINAL_SIZE_KEY`.
const AUTO_PATCH_FINAL_SIZE_KEY: &str = "AutoPatchFinalSize";
/// Java private `XYZ_KEY`.
const XYZ_KEY: &str = "XYZ";
/// Java private `PATCH_SIZE_KEY`.
const PATCH_SIZE_KEY: &str = "PatchSize";
/// Java private `REVISION`.
const REVISION: &str = "1.2";

/// Java private `DEFAULT_PATCH_SIZE = CombinePatchSize.MEDIUM.toString()`.
fn default_patch_size() -> String {
    CombinePatchSize::Medium.to_string()
}

/// Java `Boolean.valueOf(String).booleanValue()`.
fn java_lang_boolean_value_of(value: &str) -> bool {
    value.eq_ignore_ascii_case("true")
}

/// Java `props.getProperty(key, default)`.
fn get_property(props: &BTreeMap<String, String>, key: &str, default: &str) -> String {
    match props.get(key) {
        None => default.to_string(),
        Some(value) => value.clone(),
    }
}

/// Java `Integer.parseInt(props.getProperty(key, String.valueOf(current)))`.
/// CombineParams.java:633-661 throws an uncaught `NumberFormatException` out of
/// `load` when the stored value is not an integer.  Fixed in translation: an
/// unparsable value leaves the current value in place.
fn parse_int_property(props: &BTreeMap<String, String>, key: &str, current: i32) -> i32 {
    match java_lang_integer_parse_int(&get_property(props, key, &current.to_string())) {
        Ok(value) => value,
        Err(_) => current,
    }
}

/// Java `"\\s*,\\s*"`.
static COMMA_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?-u:\s)*,(?-u:\s)*").unwrap());
/// Java `"^\\s+$"`.
static BLANK_PATTERN: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^(?-u:\s)+$").unwrap());
/// Java `"^\\s*$"`.
static EMPTY_PATTERN: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^(?-u:\s)*$").unwrap());

/// Java final `CombineParams implements ConstCombineParams, Storable`.
pub struct CombineParams {
    /// Java `invalidReasons`; filled by `isValid`, which the const interface
    /// declares.
    invalid_reasons: Mutex<Vec<String>>,
    patch_size: StringProperty,
    patch_z_min: EtomoNumber,
    patch_z_max: EtomoNumber,
    patch_size_xyz: StringProperty,
    auto_patch_final_size: StringProperty,
    auto_patch_final_size_xyz: StringProperty,
    extra_residual_targets: StringProperty,
    wedge_reduction_fraction: StringProperty,
    low_from_both_radius: StringProperty,
    initial_volume_matching: EtomoNumber,
    extra_residual_targets_from_batchruntomo: StringProperty,
    fiducial_match_from_batchruntomo: StringProperty,
    auto_patch_final_size_from_batchruntomo: StringProperty,
    match_mode_from_batchruntomo: StringProperty,
    patch_type_or_xyz_from_batchruntomo: StringProperty,
    initial_volume_matching_from_batchruntomo: EtomoNumber,
    /// Java field `manager`; null for a `MetaData` built without a manager.
    manager: Option<&'static dyn BaseManager>,
    use_list: StringList,
    fiducial_match_list_a: StringList,
    fiducial_match_list_b: StringList,
    match_mode: Option<MatchMode>,
    fiducial_match: Option<FiducialMatch>,
    patch_x_min: i32,
    patch_x_max: i32,
    patch_y_min: i32,
    patch_y_max: i32,
    max_patch_z_max: i32,
    patch_region_model: String,
    temp_directory: String,
    manual_cleanup: bool,
    model_based: bool,
    transfer: bool,
    /// Java `revisionNumber`; `store` assigns it.
    revision_number: Mutex<String>,
    new_batchruntomo_settings: bool,
}

impl CombineParams {
    /// Java `CombineParams(BaseManager)`.  Default constructor.
    pub fn new(manager: Option<&'static dyn BaseManager>) -> CombineParams {
        let batchruntomo_prepend =
            format!("{}.{}", meta_data::BATCHRUNTOMO_KEY, meta_data::COMBINE_KEY);
        let mut instance = CombineParams {
            invalid_reasons: Mutex::new(Vec::new()),
            patch_size: StringProperty::new_with_key(Some(PATCH_SIZE_KEY)),
            patch_z_min: EtomoNumber::new_with_name("PatchBoundaryZMin"),
            patch_z_max: EtomoNumber::new_with_name("PatchBoundaryZMax"),
            patch_size_xyz: StringProperty::new_with_key(Some(&format!(
                "{PATCH_SIZE_KEY}.{XYZ_KEY}"
            ))),
            auto_patch_final_size: StringProperty::new_with_key(Some(AUTO_PATCH_FINAL_SIZE_KEY)),
            auto_patch_final_size_xyz: StringProperty::new_with_key(Some(&format!(
                "{AUTO_PATCH_FINAL_SIZE_KEY}.{XYZ_KEY}"
            ))),
            extra_residual_targets: StringProperty::new_with_key(Some("ExtraResidualTargets")),
            wedge_reduction_fraction: StringProperty::new_with_key(Some("WedgeReductionFraction")),
            low_from_both_radius: StringProperty::new_with_key(Some("LowFromBothRadius")),
            initial_volume_matching: EtomoNumber::new_with_type_and_name(
                Type::Boolean,
                "InitialVolumeMatching",
            ),
            extra_residual_targets_from_batchruntomo: StringProperty::new_with_key(Some(&format!(
                "{batchruntomo_prepend}.ExtraResidualTargets"
            ))),
            fiducial_match_from_batchruntomo: StringProperty::new_with_key(Some(&format!(
                "{batchruntomo_prepend}.FiducialMatch"
            ))),
            auto_patch_final_size_from_batchruntomo: StringProperty::new_with_key(Some(&format!(
                "{batchruntomo_prepend}.FinalPatchSize"
            ))),
            match_mode_from_batchruntomo: StringProperty::new_with_key(Some(&format!(
                "{batchruntomo_prepend}.MatchMode"
            ))),
            patch_type_or_xyz_from_batchruntomo: StringProperty::new_with_key(Some(&format!(
                "{batchruntomo_prepend}.PatchSize"
            ))),
            initial_volume_matching_from_batchruntomo: EtomoNumber::new_with_type_and_name(
                Type::Boolean,
                &format!("{batchruntomo_prepend}.InitialVolumeMatching"),
            ),
            manager,
            use_list: StringList::new_with_n_elements(0),
            fiducial_match_list_a: StringList::new_with_n_elements(0),
            fiducial_match_list_b: StringList::new_with_n_elements(0),
            match_mode: None,
            fiducial_match: Some(FiducialMatch::BothSides),
            patch_x_min: 0,
            patch_x_max: 0,
            patch_y_min: 0,
            patch_y_max: 0,
            max_patch_z_max: 0,
            patch_region_model: String::new(),
            temp_directory: String::new(),
            manual_cleanup: false,
            model_based: false,
            transfer: true,
            revision_number: Mutex::new(REVISION.to_string()),
            new_batchruntomo_settings: false,
        };
        instance
            .initial_volume_matching
            .set_display_value_boolean(false);
        instance.reset();
        instance
    }

    /// Java private `reset`.
    fn reset(&mut self) {
        self.invalid_reasons.lock().unwrap().clear();
        self.patch_size.set(Some(&default_patch_size()));
        self.patch_z_min.set_int(0);
        self.patch_z_max.set_int(0);
        self.patch_size_xyz.reset();
        self.auto_patch_final_size.reset();
        self.auto_patch_final_size_xyz.reset();
        self.extra_residual_targets.reset();
        self.wedge_reduction_fraction.reset();
        self.low_from_both_radius.reset();
        self.initial_volume_matching.reset();
        self.use_list.reset();
        self.fiducial_match_list_a.reset();
        self.fiducial_match_list_b.reset();
        self.match_mode = None;
        self.fiducial_match = Some(FiducialMatch::BothSides);
        self.patch_x_min = 0;
        self.patch_x_max = 0;
        self.patch_y_min = 0;
        self.patch_y_max = 0;
        self.max_patch_z_max = 0;
        self.patch_region_model = String::new();
        self.temp_directory = String::new();
        self.manual_cleanup = false;
        self.model_based = false;
        self.transfer = true;
        *self.revision_number.lock().unwrap() = REVISION.to_string();
        self.extra_residual_targets_from_batchruntomo.reset();
        self.fiducial_match_from_batchruntomo.reset();
        self.auto_patch_final_size_from_batchruntomo.reset();
        self.match_mode_from_batchruntomo.reset();
        self.patch_type_or_xyz_from_batchruntomo.reset();
        self.initial_volume_matching_from_batchruntomo.reset();
    }

    /// Java `resetBatchruntomoCombineSettings`.
    pub fn reset_batchruntomo_combine_settings(&mut self) {
        self.new_batchruntomo_settings = false;
    }

    /// Java `isNewBatchruntomoCombineSettings`.
    pub fn is_new_batchruntomo_combine_settings(&self) -> bool {
        self.new_batchruntomo_settings
    }

    /// Java `moveBatchruntomoSettings`.
    pub fn move_batchruntomo_settings(&mut self) {
        if !self.extra_residual_targets_from_batchruntomo.is_empty() {
            self.new_batchruntomo_settings = true;
            let input = self
                .extra_residual_targets_from_batchruntomo
                .to_string_option();
            self.set_extra_residual_targets(input.as_deref());
            self.extra_residual_targets_from_batchruntomo.reset();
        }
        if !self.fiducial_match_from_batchruntomo.is_empty() {
            self.new_batchruntomo_settings = true;
            // The property is lent out for the call and put back.
            let input = std::mem::take(&mut self.fiducial_match_from_batchruntomo);
            self.set_fiducial_match_string_property(Some(&input));
            self.fiducial_match_from_batchruntomo = input;
            self.fiducial_match_from_batchruntomo.reset();
        }
        if !self.auto_patch_final_size_from_batchruntomo.is_empty() {
            self.new_batchruntomo_settings = true;
            // The property is lent out for the call and put back.
            let input = std::mem::take(&mut self.auto_patch_final_size_from_batchruntomo);
            self.set_patch_size_string_property(true, Some(&input));
            self.auto_patch_final_size_from_batchruntomo = input;
            self.auto_patch_final_size_from_batchruntomo.reset();
        }
        if !self.match_mode_from_batchruntomo.is_empty() {
            self.new_batchruntomo_settings = true;
            // The property is lent out for the call and put back.
            let input = std::mem::take(&mut self.match_mode_from_batchruntomo);
            self.set_match_mode_string_property(Some(&input));
            self.match_mode_from_batchruntomo = input;
            self.match_mode_from_batchruntomo.reset();
        }
        if !self.patch_type_or_xyz_from_batchruntomo.is_empty() {
            self.new_batchruntomo_settings = true;
            // The property is lent out for the call and put back.
            let input = std::mem::take(&mut self.patch_type_or_xyz_from_batchruntomo);
            self.set_patch_size_string_property(false, Some(&input));
            self.patch_type_or_xyz_from_batchruntomo = input;
            self.patch_type_or_xyz_from_batchruntomo.reset();
        }
        if !self.initial_volume_matching_from_batchruntomo.is_null() {
            self.new_batchruntomo_settings = true;
            let input = self.initial_volume_matching_from_batchruntomo.is();
            self.set_initial_volume_matching(input);
            self.initial_volume_matching_from_batchruntomo.reset();
        }
    }

    /// Java `setLowFromBothRadius`.
    pub fn set_low_from_both_radius(&mut self, input: Option<&str>) {
        self.low_from_both_radius.set(input);
    }

    /// Java `setMatchMode(boolean)`.
    pub fn set_match_mode_boolean(&mut self, is_bto_a: bool) {
        if is_bto_a {
            self.match_mode = Some(MatchMode::BToA);
        } else {
            self.match_mode = Some(MatchMode::AToB);
        }
    }

    /// Java `setMatchMode(MatchMode)`.
    pub fn set_match_mode(&mut self, match_mode: Option<MatchMode>) {
        self.match_mode = match_mode;
    }

    /// Java `setMatchMode(StringProperty)`.
    pub fn set_match_mode_string_property(&mut self, input: Option<&StringProperty>) {
        if let Some(input) = input {
            self.match_mode = MatchMode::get_instance_string(input.to_string_option().as_deref());
        } else {
            self.match_mode = None;
        }
    }

    /// Java `setFiducialMatch(FiducialMatch)`.
    pub fn set_fiducial_match(&mut self, fiducial_match: Option<FiducialMatch>) {
        self.fiducial_match = fiducial_match;
        if fiducial_match == Some(FiducialMatch::UseModel)
            || fiducial_match == Some(FiducialMatch::UseModelOnly)
        {
            self.model_based = true;
        } else {
            self.model_based = false;
        }
    }

    /// Java `setFiducialMatch(StringProperty)`.
    pub fn set_fiducial_match_string_property(&mut self, input: Option<&StringProperty>) {
        if let Some(input) = input {
            // `FiducialMatch.getInstance(input.toString())`: a null string matches no
            // name.
            self.fiducial_match = match input.to_string_option() {
                None => None,
                Some(string) => FiducialMatch::get_instance(&string),
            };
            if self.fiducial_match.is_none() {
                self.fiducial_match = Some(FiducialMatch::BothSides);
            }
        } else {
            self.fiducial_match = Some(FiducialMatch::BothSides);
        }
    }

    /// Java `setUseList`.
    pub fn set_use_list(&mut self, use_list: Option<&str>) {
        self.use_list.parse_string(use_list);
    }

    /// Java `setFiducialMatchListA`.
    pub fn set_fiducial_match_list_a(&mut self, list: Option<&str>) {
        self.fiducial_match_list_a.parse_string(list);
    }

    /// Java `setFiducialMatchListB`.
    pub fn set_fiducial_match_list_b(&mut self, list: Option<&str>) {
        self.fiducial_match_list_b.parse_string(list);
    }

    /// Java `setWedgeReductionFraction`.
    pub fn set_wedge_reduction_fraction(&mut self, input: Option<&str>) {
        self.wedge_reduction_fraction.set(input);
    }

    /// Java `setPatchSize(boolean, CombinePatchSize)`.
    pub fn set_patch_size(&mut self, auto_final: bool, input: Option<CombinePatchSize>) {
        if let Some(input) = input {
            if !auto_final {
                self.patch_size.set(Some(&input.to_string()));
            } else {
                self.auto_patch_final_size.set(Some(&input.to_string()));
            }
            if input != CombinePatchSize::Custom {
                if !auto_final {
                    self.patch_size_xyz.reset();
                } else {
                    self.auto_patch_final_size_xyz.reset();
                }
            }
        } else if !auto_final {
            self.patch_size.set(Some(&default_patch_size()));
            self.patch_size_xyz.reset();
        } else {
            self.auto_patch_final_size.reset();
            self.auto_patch_final_size_xyz.reset();
        }
    }

    /// Java `setPatchSize(boolean, String)`.
    pub fn set_patch_size_string(&mut self, auto_final: bool, input: Option<&str>) {
        let combine_patch_size = CombinePatchSize::get_instance(input);
        self.set_patch_size(auto_final, combine_patch_size);
        if combine_patch_size == Some(CombinePatchSize::Custom) {
            if !auto_final {
                self.patch_size_xyz.set(input);
            } else {
                self.auto_patch_final_size_xyz.set(input);
            }
        }
    }

    /// Java `setPatchSizeXYZ(boolean, String[])`.
    pub fn set_patch_size_xyz(&mut self, auto_final: bool, xyz: Option<&[&str]>) {
        let xyz = match xyz {
            None => {
                if !auto_final {
                    self.patch_size
                        .set(Some(&CombinePatchSize::Custom.to_string()));
                    self.patch_size_xyz.reset();
                } else {
                    self.auto_patch_final_size
                        .set(Some(&CombinePatchSize::Custom.to_string()));
                    self.auto_patch_final_size_xyz.reset();
                }
                // CombineParams.java:335-345 falls through to
                // `CombinePatchSize.getInstance(null)`, which returns null, and then
                // throws NullPointerException on `combinePatchSize.toString()`.
                // Fixed in translation: a null xyz ends here, with the custom size
                // and empty XYZ the block above has set.
                return;
            }
            Some(xyz) => xyz,
        };
        // `getInstance(String[])` returns non-null for a non-null array.
        let combine_patch_size = CombinePatchSize::get_instance_xyz_strings(Some(xyz)).unwrap();
        if !auto_final {
            self.patch_size.set(Some(&combine_patch_size.to_string()));
            if self.patch_size.is_empty() {
                self.patch_size.set(Some(&default_patch_size()));
            }
        } else {
            self.auto_patch_final_size
                .set(Some(&combine_patch_size.to_string()));
        }
        if combine_patch_size == CombinePatchSize::Custom {
            let mut builder = String::new();
            for i in 0..xyz.len() {
                builder.push_str(xyz[i]);
                if i < xyz.len() - 1 {
                    builder.push_str(", ");
                }
            }
            if !auto_final {
                self.patch_size
                    .set(Some(&CombinePatchSize::Custom.to_string()));
                self.patch_size_xyz.set(Some(&builder));
            } else {
                self.auto_patch_final_size
                    .set(Some(&CombinePatchSize::Custom.to_string()));
                self.auto_patch_final_size_xyz.set(Some(&builder));
            }
        }
    }

    /// Java `setPatchSize(boolean, StringProperty)`.
    pub fn set_patch_size_string_property(
        &mut self,
        auto_final: bool,
        input: Option<&StringProperty>,
    ) {
        if let Some(input) = input {
            self.set_patch_size_string(auto_final, input.to_string_option().as_deref());
        } else if !auto_final {
            self.patch_size.set(Some(&default_patch_size()));
            self.patch_size_xyz.reset();
        } else {
            self.auto_patch_final_size.reset();
            self.auto_patch_final_size_xyz.reset();
        }
    }

    /// Java `setInitialVolumeMatching`.
    pub fn set_initial_volume_matching(&mut self, input: bool) {
        self.initial_volume_matching.set_boolean(input);
    }

    /// Java `setExtraResidualTargets`.
    pub fn set_extra_residual_targets(&mut self, input: Option<&str>) {
        self.extra_residual_targets.set(input);
    }

    /// Java `resetExtraResidualTargets`.
    pub fn reset_extra_residual_targets(&mut self) {
        self.extra_residual_targets.reset();
    }

    /// Java `resetPatchSize`.
    pub fn reset_patch_size(&mut self, auto_final: bool) {
        if !auto_final {
            self.patch_size.set(Some(&default_patch_size()));
        } else {
            self.auto_patch_final_size.reset();
        }
    }

    /// Java `setPatchRegionModel`.
    pub fn set_patch_region_model(&mut self, model_file_name: &str) {
        if BLANK_PATTERN.is_match(model_file_name) {
            self.patch_region_model = String::new();
        } else {
            self.patch_region_model = model_file_name.to_string();
        }
    }

    /// Java `setDefaultPatchRegionModel`.
    pub fn set_default_patch_region_model(&mut self) {
        self.patch_region_model = MatchorwarpParam::get_default_patch_region_model();
    }

    /// Java `setPatchXMax`.
    pub fn set_patch_x_max(&mut self, patch_x_max: i32) {
        self.patch_x_max = patch_x_max;
    }

    /// Java `setPatchXMin`.
    pub fn set_patch_x_min(&mut self, patch_x_min: i32) {
        self.patch_x_min = patch_x_min;
    }

    /// Java `setPatchYMax`.
    pub fn set_patch_y_max(&mut self, patch_y_max: i32) {
        self.patch_y_max = patch_y_max;
    }

    /// Java `setPatchYMin`.
    pub fn set_patch_y_min(&mut self, patch_y_min: i32) {
        self.patch_y_min = patch_y_min;
    }

    /// Java `setPatchZMax(String)`.
    pub fn set_patch_z_max(&mut self, patch_z_max: Option<&str>) {
        self.patch_z_max.set_string(patch_z_max);
    }

    /// Java `setPatchZMin(String)`.
    pub fn set_patch_z_min(&mut self, patch_z_min: Option<&str>) {
        self.patch_z_min.set_string(patch_z_min);
    }

    /// Java `setMaxPatchZMax(String) throws InvalidParameterException,
    /// IOException`.  `Err` carries the exception's message.
    ///
    /// Upstream bug fixed in translation: a null manager throws
    /// NullPointerException; it is returned as `Err` here.
    pub fn set_max_patch_z_max_from_file(&mut self, file_name: &str) -> Result<(), String> {
        let manager = self
            .manager
            .ok_or_else(|| "java.lang.NullPointerException".to_string())?;
        // Get the data size limits from the image stack
        let mrc_header = MRCHeader::get_instance_in_dir(
            manager.get_property_user_dir().as_deref(),
            Some(file_name),
            Some(AxisID::Only),
        )
        .unwrap();
        if !mrc_header.borrow_mut().read_with_manager(manager)? {
            return Err("file does not exist".to_string());
        }
        self.max_patch_z_max = mrc_header.borrow().get_n_rows();
        Ok(())
    }

    /// Java `setMaxPatchZMax(int)`.
    pub fn set_max_patch_z_max(&mut self, max_patch_z_max: i32) {
        self.max_patch_z_max = max_patch_z_max;
    }

    /// Java `setTempDirectory`.
    pub fn set_temp_directory(&mut self, directory_name: &str) {
        if BLANK_PATTERN.is_match(directory_name) {
            self.temp_directory = String::new();
        } else {
            self.temp_directory = directory_name.to_string();
        }
    }

    /// Java `setTransfer`.
    pub fn set_transfer(&mut self, transfer: bool) {
        self.transfer = transfer;
    }

    /// Java `setManualCleanup`.
    pub fn set_manual_cleanup(&mut self, is_manual: bool) {
        self.manual_cleanup = is_manual;
    }

    /// Java `setModelBased`.  Sets the modelBased state; true if a model based
    /// combine is being used.
    pub fn set_model_based(&mut self, model_based: bool) {
        self.model_based = model_based;
        if model_based {
            self.fiducial_match = Some(FiducialMatch::UseModel);
        } else {
            self.fiducial_match = Some(FiducialMatch::BothSides);
        }
    }

    /// Java `setDefaultPatchBoundaries(String) throws InvalidParameterException,
    /// IOException`.  Sets the patch boundaries to the default value that matches
    /// the logic in the setupcombine script.  `Err` carries the exception's
    /// message.
    ///
    /// Upstream bug fixed in translation: a null manager throws
    /// NullPointerException; it is returned as `Err` here.
    pub fn set_default_patch_boundaries(&mut self, file_name: &str) -> Result<(), String> {
        let manager = self
            .manager
            .ok_or_else(|| "java.lang.NullPointerException".to_string())?;
        // Get the data size limits from the image stack
        let mrc_header = MRCHeader::get_instance_in_dir(
            manager.get_property_user_dir().as_deref(),
            Some(file_name),
            Some(AxisID::Only),
        )
        .unwrap();
        if !mrc_header.borrow_mut().read_with_manager(manager)? {
            return Ok(());
        }
        let mrc_header = mrc_header.borrow();
        let xyborder = CombineParams::get_xy_border(&mrc_header);
        self.patch_x_min = xyborder;
        self.patch_x_max = mrc_header.get_n_columns() - xyborder;
        self.patch_y_min = xyborder;
        self.patch_y_max = mrc_header.get_n_sections() - xyborder;
        self.patch_z_min.set_int(1);
        self.patch_z_max.set_int(mrc_header.get_n_rows());
        self.max_patch_z_max = self.patch_z_max.get_int();
        Ok(())
    }

    /// Java static `getXYBorder`.  Gets the border for xy using logic from
    /// setupcombine and the mrcheader.
    pub fn get_xy_border(mrc_header: &MRCHeader) -> i32 {
        // Logic from setupcombine to provide the default border size, the variable
        // names used match those from the setupcombine script
        let xyborders: [i32; 5] = [24, 36, 54, 68, 80];
        let borderinc = 1000;
        // Assume that Y and Z domains are swapped
        let minsize = std::cmp::min(mrc_header.get_n_columns(), mrc_header.get_n_sections());
        let mut borderindex = minsize / borderinc;
        if borderindex > 4 {
            borderindex = 4;
        }
        xyborders[borderindex as usize]
    }

    /// Java `isLowFromBothRadiusSet`.
    pub fn is_low_from_both_radius_set(&self) -> bool {
        !self.low_from_both_radius.is_empty()
    }

    /// Java `isWedgeReductionFractionSet`.
    pub fn is_wedge_reduction_fraction_set(&self) -> bool {
        !self.wedge_reduction_fraction.is_empty()
    }

    /// Java `getPatchRegionModel`.
    pub fn get_patch_region_model(&self) -> String {
        self.patch_region_model.clone()
    }

    /// Java `getPatchSizeXYZ`.
    pub fn get_patch_size_xyz(&self, auto_final: bool) -> Option<String> {
        if !auto_final {
            self.patch_size_xyz.to_string_option()
        } else {
            self.auto_patch_final_size_xyz.to_string_option()
        }
    }

    /// Java `isTempDirectorySet`.
    pub fn is_temp_directory_set(&self) -> bool {
        self.temp_directory != ""
    }
}

impl StorableValue for CombineParams {
    /// Java `store(Properties)`.  Insert the objects attributes into the
    /// properties object.
    fn store(&self, props: &mut BTreeMap<String, String>) {
        self.store_with_prepend(props, "");
    }

    /// Java `store(Properties, String)`.
    fn store_with_prepend(&self, props: &mut BTreeMap<String, String>, prepend: &str) {
        let group;
        let prepend = if prepend == "" {
            "Combine".to_string()
        } else {
            format!("{prepend}Combine")
        };
        group = format!("{prepend}.");
        *self.revision_number.lock().unwrap() = REVISION.to_string();
        props.insert(
            format!("{group}RevisionNumber"),
            self.revision_number.lock().unwrap().clone(),
        );
        // Start backwards compatibility with RevisionNumber = 1.0
        props.remove(&format!("{group}{MATCH_B_TO_A_KEY}"));
        // backwards compatibility with 1.1
        props.remove(&format!("{group}{DIALOG_MATCH_MODE_KEY}"));
        // End backwards compatibility with RevisionNumber = 1.0
        match self.match_mode {
            None => {
                props.remove(&format!("{group}{MATCH_MODE_KEY}"));
            }
            Some(match_mode) => {
                props.insert(
                    format!("{group}{MATCH_MODE_KEY}"),
                    match_mode.to_string().to_string(),
                );
            }
        }
        // CombineParams.java:534 calls `fiducialMatch.toString()`, a
        // NullPointerException when `load` read an unrecognized FiducialMatch name
        // (`FiducialMatch.fromString` returns null).  Fixed in translation: a null
        // fiducial match removes the property, as a null match mode does.
        match self.fiducial_match {
            None => {
                props.remove(&format!("{group}FiducialMatch"));
            }
            Some(fiducial_match) => {
                props.insert(
                    format!("{group}FiducialMatch"),
                    fiducial_match.to_string().to_string(),
                );
            }
        }
        props.insert(format!("{group}UseList"), self.use_list.to_string());
        props.insert(
            format!("{group}FiducialMatchListA"),
            self.fiducial_match_list_a.to_string(),
        );
        props.insert(
            format!("{group}FiducialMatchListB"),
            self.fiducial_match_list_b.to_string(),
        );
        self.patch_size
            .store_with_prepend(Some(props), Some(&prepend));
        props.insert(
            format!("{group}PatchBoundaryXMin"),
            self.patch_x_min.to_string(),
        );
        props.insert(
            format!("{group}PatchBoundaryXMax"),
            self.patch_x_max.to_string(),
        );
        props.insert(
            format!("{group}PatchBoundaryYMin"),
            self.patch_y_min.to_string(),
        );
        props.insert(
            format!("{group}PatchBoundaryYMax"),
            self.patch_y_max.to_string(),
        );
        self.patch_z_min.store_with_prepend(props, &prepend);
        self.patch_z_max.store_with_prepend(props, &prepend);
        props.insert(
            format!("{group}PatchRegionModel"),
            self.patch_region_model.clone(),
        );
        props.insert(format!("{group}TempDirectory"), self.temp_directory.clone());
        props.insert(
            format!("{group}ManualCleanup"),
            self.manual_cleanup.to_string(),
        );
        props.insert(format!("{group}ModelBased"), self.model_based.to_string());
        props.insert(format!("{group}Transfer"), self.transfer.to_string());
        props.insert(
            format!("{group}MaxPatchBoundaryZMax"),
            self.max_patch_z_max.to_string(),
        );
        self.patch_size_xyz
            .store_with_prepend(Some(props), Some(&prepend));
        self.auto_patch_final_size
            .store_with_prepend(Some(props), Some(&prepend));
        self.auto_patch_final_size_xyz
            .store_with_prepend(Some(props), Some(&prepend));
        self.extra_residual_targets
            .store_with_prepend(Some(props), Some(&prepend));
        self.wedge_reduction_fraction
            .store_with_prepend(Some(props), Some(&prepend));
        self.low_from_both_radius
            .store_with_prepend(Some(props), Some(&prepend));
        self.initial_volume_matching
            .store_with_prepend(props, &prepend);
        // batchruntomo settings
        self.extra_residual_targets_from_batchruntomo
            .store(Some(props));
        self.fiducial_match_from_batchruntomo.store(Some(props));
        self.auto_patch_final_size_from_batchruntomo
            .store(Some(props));
        self.match_mode_from_batchruntomo.store(Some(props));
        self.patch_type_or_xyz_from_batchruntomo.store(Some(props));
        self.initial_volume_matching_from_batchruntomo.store(props);
    }

    /// Java `load(Properties)`.  Get the objects attributes from the properties
    /// object.
    fn load(&mut self, props: &BTreeMap<String, String>) {
        self.load_with_prepend(props, "");
    }

    /// Java `load(Properties, String)`.
    fn load_with_prepend(&mut self, props: &BTreeMap<String, String>, prepend: &str) {
        self.reset();
        let group;
        let prepend = if prepend == "" {
            "Combine".to_string()
        } else {
            format!("{prepend}Combine")
        };
        group = format!("{prepend}.");
        // `StringProperty.load` takes a mutable Properties because a backward
        // compatible key is removed from it; none of these properties has one, so
        // loading them from a copy changes nothing.
        let mut props_copy = props.clone();
        // Load the combine values if they are present, don't change the
        // current value if the property is not present
        *self.revision_number.lock().unwrap() =
            get_property(props, &format!("{group}RevisionNumber"), "1.2");
        // Start backwards compatibility with RevisionNumber = 1.0
        // load dialogMatchMode
        // old property was MatchBtoA. MatchBtoA should be deleted in store()
        let dialog_match_mode_string = props.get(&format!("{group}{DIALOG_MATCH_MODE_KEY}"));
        // backwards compatibility with 1.1
        let mut dialog_match_mode: Option<MatchMode> = None;
        match dialog_match_mode_string {
            None => {
                let match_bto_a = props.get(&format!("{group}{MATCH_B_TO_A_KEY}"));
                match match_bto_a {
                    None => {
                        dialog_match_mode = None;
                    }
                    Some(match_bto_a) => {
                        if java_lang_boolean_value_of(match_bto_a) {
                            dialog_match_mode = Some(MatchMode::BToA);
                        } else {
                            dialog_match_mode = Some(MatchMode::AToB);
                        }
                    }
                }
            }
            Some(dialog_match_mode_string) => {
                let load_dialog_match_mode =
                    MatchMode::get_instance_string(Some(dialog_match_mode_string));
                if load_dialog_match_mode.is_some() {
                    dialog_match_mode = load_dialog_match_mode;
                }
            }
        }
        // End backwards compatibility with RevisionNumber = 1.0
        match self.match_mode {
            None => {
                self.match_mode = MatchMode::get_instance_string(
                    props
                        .get(&format!("{group}{MATCH_MODE_KEY}"))
                        .map(|value| value.as_str()),
                );
                if self.match_mode.is_none() {
                    // backwards compatibility with 1.1
                    self.match_mode = dialog_match_mode;
                }
            }
            Some(match_mode) => {
                self.match_mode = MatchMode::get_instance_string(Some(&get_property(
                    props,
                    &format!("{group}{MATCH_MODE_KEY}"),
                    match_mode.to_string(),
                )));
            }
        }
        // `fiducialMatch.toString()`: reset() has just set it to BOTH_SIDES.
        let current_fiducial_match = match self.fiducial_match {
            None => "null",
            Some(fiducial_match) => fiducial_match.to_string(),
        };
        self.fiducial_match = FiducialMatch::from_string(&get_property(
            props,
            &format!("{group}FiducialMatch"),
            current_fiducial_match,
        ));
        let use_list = get_property(
            props,
            &format!("{group}UseList"),
            &self.use_list.to_string(),
        );
        self.use_list.parse_string(Some(&use_list));
        let list_a = get_property(
            props,
            &format!("{group}FiducialMatchListA"),
            &self.fiducial_match_list_a.to_string(),
        );
        self.fiducial_match_list_a.parse_string(Some(&list_a));
        let list_b = get_property(
            props,
            &format!("{group}FiducialMatchListB"),
            &self.fiducial_match_list_b.to_string(),
        );
        self.fiducial_match_list_b.parse_string(Some(&list_b));
        self.patch_size.load_with_default(
            Some(&mut props_copy),
            Some(&prepend),
            Some(&default_patch_size()),
        );
        self.patch_region_model = get_property(
            props,
            &format!("{group}PatchRegionModel"),
            &self.patch_region_model,
        );
        self.patch_x_min = parse_int_property(
            props,
            &format!("{group}PatchBoundaryXMin"),
            self.patch_x_min,
        );
        self.patch_x_max = parse_int_property(
            props,
            &format!("{group}PatchBoundaryXMax"),
            self.patch_x_max,
        );
        self.patch_y_min = parse_int_property(
            props,
            &format!("{group}PatchBoundaryYMin"),
            self.patch_y_min,
        );
        self.patch_y_max = parse_int_property(
            props,
            &format!("{group}PatchBoundaryYMax"),
            self.patch_y_max,
        );
        self.patch_z_min.load_with_prepend(props, Some(&prepend));
        self.patch_z_max.load_with_prepend(props, Some(&prepend));
        self.temp_directory = get_property(
            props,
            &format!("{group}TempDirectory"),
            &self.temp_directory,
        );
        self.manual_cleanup = java_lang_boolean_value_of(&get_property(
            props,
            &format!("{group}ManualCleanup"),
            &self.manual_cleanup.to_string(),
        ));
        self.model_based = java_lang_boolean_value_of(&get_property(
            props,
            &format!("{group}ModelBased"),
            &self.model_based.to_string(),
        ));
        self.transfer = java_lang_boolean_value_of(&get_property(
            props,
            &format!("{group}Transfer"),
            &self.transfer.to_string(),
        ));
        if self.fiducial_match == Some(FiducialMatch::UseModel) {
            self.model_based = true;
        } else {
            self.model_based = false;
        }
        self.max_patch_z_max = parse_int_property(
            props,
            &format!("{group}MaxPatchBoundaryZMax"),
            self.max_patch_z_max,
        );
        self.patch_size_xyz
            .load_with_prepend(Some(&mut props_copy), Some(&prepend));
        self.auto_patch_final_size
            .load_with_prepend(Some(&mut props_copy), Some(&prepend));
        self.auto_patch_final_size_xyz
            .load_with_prepend(Some(&mut props_copy), Some(&prepend));
        self.extra_residual_targets
            .load_with_prepend(Some(&mut props_copy), Some(&prepend));
        self.wedge_reduction_fraction
            .load_with_prepend(Some(&mut props_copy), Some(&prepend));
        self.low_from_both_radius
            .load_with_prepend(Some(&mut props_copy), Some(&prepend));
        self.initial_volume_matching
            .load_with_prepend(props, Some(&prepend));
        // batchruntomo settings
        self.extra_residual_targets_from_batchruntomo
            .load(Some(&mut props_copy));
        self.fiducial_match_from_batchruntomo
            .load(Some(&mut props_copy));
        self.auto_patch_final_size_from_batchruntomo
            .load(Some(&mut props_copy));
        self.match_mode_from_batchruntomo
            .load(Some(&mut props_copy));
        self.patch_type_or_xyz_from_batchruntomo
            .load(Some(&mut props_copy));
        self.initial_volume_matching_from_batchruntomo.load(props);
    }
}

impl ConstCombineParams for CombineParams {
    /// Java `isPatchSizeSet`.
    fn is_patch_size_set(&self, auto_final: bool) -> bool {
        if !auto_final {
            !self.patch_size.is_empty()
        } else {
            !self.auto_patch_final_size.is_empty()
        }
    }

    /// Java `isExtraResidualTargetsSet`.
    fn is_extra_residual_targets_set(&self) -> bool {
        !self.extra_residual_targets.is_empty()
    }

    /// Java `isPatchBoundarySet`.  Returns true if the patch boundary values have
    /// been modified.
    fn is_patch_boundary_set(&self) -> bool {
        if self.patch_x_min == 0
            && self.patch_x_max == 0
            && self.patch_y_min == 0
            && self.patch_y_max == 0
            && self.patch_z_min.equals_int(0)
            && self.patch_z_max.equals_int(0)
        {
            return false;
        }
        true
    }

    /// Java final `isValid`.  Checks the validity of the attribute values.
    fn is_valid(&self, y_and_zflipped: bool) -> bool {
        let mut valid = true;
        // Clear any previous reasons from the list
        let mut invalid_reasons = self.invalid_reasons.lock().unwrap();
        invalid_reasons.clear();
        if self.patch_x_min < 1 {
            valid = false;
            invalid_reasons.push("X min value is less than 1".to_string());
        }
        if self.patch_x_max < 1 {
            valid = false;
            invalid_reasons.push("X max value is less than 1".to_string());
        }
        if self.patch_x_min > self.patch_x_max {
            valid = false;
            invalid_reasons.push("X min value is greater than the X max value".to_string());
        }
        if self.patch_y_min < 1 {
            valid = false;
            invalid_reasons.push("Y min value is less than 1".to_string());
        }
        if self.patch_y_max < 1 {
            valid = false;
            invalid_reasons.push("Y max value is less than 1".to_string());
        }
        if self.patch_y_min > self.patch_y_max {
            valid = false;
            invalid_reasons.push("Y min value is greater than the Y max value".to_string());
        }
        if self.patch_z_min.get_int() < 1 {
            valid = false;
            invalid_reasons.push("Z min value is less than 1".to_string());
        }
        if self.patch_z_max.get_int() < 1 {
            valid = false;
            invalid_reasons.push("ZX max value is less than 1".to_string());
        }
        if self.max_patch_z_max > 0 && self.patch_z_max.gt_int(self.max_patch_z_max) {
            valid = false;
            invalid_reasons.push(format!(
                "Z max value is greater than the maximum Z max value ({})",
                self.max_patch_z_max
            ));
        }
        if self
            .patch_z_min
            .gt_const_etomo_number(Some(&self.patch_z_max))
        {
            valid = false;
            invalid_reasons.push("Z min value is greater than the Z max value".to_string());
        }
        // get the tomogram header to check x, y, and z
        let axis_id;
        if self.match_mode.is_none() || self.match_mode == Some(MatchMode::BToA) {
            axis_id = AxisID::First;
        } else {
            axis_id = AxisID::Second;
        }
        // Upstream bug fixed in translation (CombineParams.java:941): a null manager
        // throws NullPointerException before the `try`; here it returns true, as an
        // unreadable header does.
        let Some(manager) = self.manager else {
            return true;
        };
        let tomogram = dataset_files::get_tomogram(manager, Some(axis_id))
            .map(|tomogram| utilities::java_io_file_get_absolute_path(&tomogram.to_string_lossy()));
        let header = MRCHeader::get_instance_in_dir(
            manager.get_property_user_dir().as_deref(),
            tomogram.as_deref(),
            Some(axis_id),
        )
        .unwrap();
        // Java catches IOException and every other Exception alike.
        match header.borrow_mut().read_with_manager(manager) {
            Ok(false) | Err(_) => return true,
            Ok(true) => {}
        }
        let header = header.borrow();
        let x = header.get_n_columns();
        if x < self.patch_x_min || x < self.patch_x_max {
            valid = false;
            invalid_reasons.push(format!("X values cannot be greater then {x}"));
        }
        let y;
        let z;
        if y_and_zflipped {
            y = header.get_n_sections();
            z = header.get_n_rows();
        } else {
            y = header.get_n_rows();
            z = header.get_n_sections();
        }
        if y < self.patch_y_min || y < self.patch_y_max {
            valid = false;
            invalid_reasons.push(format!("Y values cannot be greater then {y}"));
        }
        if self.patch_z_min.gt_int(z) || self.patch_z_max.gt_int(z) {
            valid = false;
            invalid_reasons.push(format!("Z values cannot be greater then {z}"));
        }
        valid
    }

    /// Java final `getInvalidReasons`.
    fn get_invalid_reasons(&self) -> Vec<String> {
        self.invalid_reasons.lock().unwrap().clone()
    }

    /// Java `getMatchMode`.
    fn get_match_mode(&self) -> Option<MatchMode> {
        self.match_mode
    }

    /// Java `isTransfer`.
    fn is_transfer(&self) -> bool {
        self.transfer
    }

    /// Java `getFiducialMatch`.
    fn get_fiducial_match(&self) -> Option<FiducialMatch> {
        self.fiducial_match
    }

    /// Java `getUseList`.
    fn get_use_list(&self) -> String {
        self.use_list.to_string()
    }

    /// Java `getFiducialMatchListA`.
    fn get_fiducial_match_list_a(&self) -> String {
        self.fiducial_match_list_a.to_string()
    }

    /// Java `getFiducialMatchListB`.
    fn get_fiducial_match_list_b(&self) -> String {
        self.fiducial_match_list_b.to_string()
    }

    /// Java `getPatchSize`.
    fn get_patch_size(&self, auto_final: bool) -> Option<CombinePatchSize> {
        if !auto_final {
            CombinePatchSize::get_instance(self.patch_size.to_string_option().as_deref())
        } else {
            CombinePatchSize::get_instance(self.auto_patch_final_size.to_string_option().as_deref())
        }
    }

    /// Java `getPatchSizeXYZArray`.
    fn get_patch_size_xyz_array(&self, auto_final: bool) -> Option<Vec<String>> {
        let xyz;
        if !auto_final {
            xyz = self.patch_size_xyz.to_string_option();
        } else {
            xyz = self.auto_patch_final_size_xyz.to_string_option();
        }
        let xyz = xyz?;
        Some(utilities::java_lang_string_split(&xyz, &COMMA_PATTERN))
    }

    /// Java `getExtraResidualTargets`.
    fn get_extra_residual_targets(&self) -> Option<String> {
        self.extra_residual_targets.to_string_option()
    }

    /// Java `getTempDirectory`.
    fn get_temp_directory(&self) -> String {
        self.temp_directory.clone()
    }

    /// Java `getManualCleanup`.
    fn get_manual_cleanup(&self) -> bool {
        self.manual_cleanup
    }

    /// Java `getPatchXMax`.
    fn get_patch_x_max(&self) -> i32 {
        self.patch_x_max
    }

    /// Java `getPatchXMin`.
    fn get_patch_x_min(&self) -> i32 {
        self.patch_x_min
    }

    /// Java `getPatchYMax`.
    fn get_patch_y_max(&self) -> i32 {
        self.patch_y_max
    }

    /// Java `getPatchYMin`.
    fn get_patch_y_min(&self) -> i32 {
        self.patch_y_min
    }

    /// Java `getPatchZMax`.
    fn get_patch_z_max(&self) -> &ConstEtomoNumber {
        &self.patch_z_max
    }

    /// Java `getPatchZMin`.
    fn get_patch_z_min(&self) -> &ConstEtomoNumber {
        &self.patch_z_min
    }

    /// Java `getMaxPatchZMax`.
    fn get_max_patch_z_max(&self) -> i32 {
        self.max_patch_z_max
    }

    /// Java `usePatchRegionModel`.  Returns true if a patch region model has been
    /// specified.
    fn use_patch_region_model(&self) -> bool {
        !EMPTY_PATTERN.is_match(&self.patch_region_model)
    }

    /// Java `getWedgeReductionFraction`.
    fn get_wedge_reduction_fraction(&self) -> Option<String> {
        self.wedge_reduction_fraction.to_string_option()
    }

    /// Java `getLowFromBothRadius`.
    fn get_low_from_both_radius(&self) -> Option<String> {
        self.low_from_both_radius.to_string_option()
    }

    /// Java `isInitialVolumeMatching`.
    fn is_initial_volume_matching(&self) -> bool {
        self.initial_volume_matching.is()
    }
}
