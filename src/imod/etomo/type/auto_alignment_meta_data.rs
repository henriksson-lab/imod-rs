//! `IMOD/Etomo/src/etomo/type/AutoAlignmentMetaData.java`.
//!
//! Copyright: Copyright 2012
//!
//! Organization: Boulder Laboratory for 3-Dimensional Electron Microscopy of Cells
//! (BL3DEMC), University of Colorado

use super::base_meta_data;
use super::const_etomo_number::{ConstEtomoNumber, Number, Type};
use super::etomo_boolean2::EtomoBoolean2;
use super::etomo_number::EtomoNumber;
use super::etomo_version::EtomoVersion;
use super::script_parameter::ScriptParameter;
use super::transform::Transform;
use std::collections::{BTreeMap, HashMap};
use std::sync::LazyLock;

/// Java `rcsid`.
pub const RCSID: &str = "$Id:$";

/// Java private `GROUP_KEY`.
const GROUP_KEY: &str = "AutoAlignment";
/// Java private `CURRENT_VERSION`:
/// `EtomoVersion.getInstance(BaseMetaData.revisionNumberString, "1.0")`.
static CURRENT_VERSION: LazyLock<EtomoVersion> = LazyLock::new(|| {
    EtomoVersion::get_instance(Some(base_meta_data::REVISION_NUMBER_STRING), Some("1.0"))
});
/// Java private `ALIGN_TRANFORM_KEY`.
const ALIGN_TRANFORM_KEY: &str = "AlignTransform";

/// Java `AutoAlignmentMetaData`.
#[derive(Clone, Debug)]
pub struct AutoAlignmentMetaData {
    /// Java private final field `sigmaLowFrequency`.
    sigma_low_frequency: ScriptParameter,
    /// Java private final field `cutoffHighFrequency`.
    cutoff_high_frequency: ScriptParameter,
    /// Java private final field `sigmaHighFrequency`.
    sigma_high_frequency: ScriptParameter,
    /// Java private final field `reduceByBinning`.
    reduce_by_binning: EtomoNumber,
    /// Java private final field `edgeToIgnore`.
    edge_to_ignore: EtomoNumber,
    /// Java private final field `midasBinning`.
    midas_binning: EtomoNumber,
    /// Java private final field `skipSectionsFrom1`.
    skip_sections_from1: ScriptParameter,
    /// Java private final field `preCrossCorrelation`.
    pre_cross_correlation: EtomoBoolean2,
    /// Java private final field `shiftLimitsForWarpX`.
    shift_limits_for_warp_x: EtomoNumber,
    /// Java private final field `shiftLimitsForWarpY`.
    shift_limits_for_warp_y: EtomoNumber,
    /// Java private final field `warpPatchSizeX`.
    warp_patch_size_x: EtomoNumber,
    /// Java private final field `warpPatchSizeY`.
    warp_patch_size_y: EtomoNumber,
    /// Java private final field `boundaryModel`.
    boundary_model: EtomoBoolean2,
    /// Java private final field `findWarping`.
    find_warping: EtomoBoolean2,
    /// Java private final field `sobelFilter`.
    sobel_filter: EtomoBoolean2,
    /// Java private final field `sigmaLowFrequencyEnabled`.
    sigma_low_frequency_enabled: EtomoBoolean2,
    /// Java private final field `cutoffHighFrequencyEnabled`.
    cutoff_high_frequency_enabled: EtomoBoolean2,
    /// Java private final field `sigmaHighFrequencyEnabled`.
    sigma_high_frequency_enabled: EtomoBoolean2,
    /// Java private field `alignTransform`, initialised to `Transform.DEFAULT`.
    align_transform: Transform,
}

impl AutoAlignmentMetaData {
    /// Java package-private constructor `AutoAlignmentMetaData()`.
    pub fn new() -> AutoAlignmentMetaData {
        let mut instance = AutoAlignmentMetaData {
            sigma_low_frequency: ScriptParameter::new_with_type_and_name(
                Type::Double,
                "SigmaLowFrequency",
            ),
            cutoff_high_frequency: ScriptParameter::new_with_type_and_name(
                Type::Double,
                "CutoffHighFrequency",
            ),
            sigma_high_frequency: ScriptParameter::new_with_type_and_name(
                Type::Double,
                "SigmaHighFrequency",
            ),
            reduce_by_binning: EtomoNumber::new_with_name("ReduceByBinning"),
            edge_to_ignore: EtomoNumber::new_with_type_and_name(Type::Double, "EdgeToIgnore"),
            midas_binning: EtomoNumber::new_with_name("MidasBinning"),
            skip_sections_from1: ScriptParameter::new_with_name("SkipSectionsFrom1"),
            pre_cross_correlation: EtomoBoolean2::new_with_name("PreCrossCorrelation"),
            shift_limits_for_warp_x: EtomoNumber::new_with_name("ShiftLimitsForWarp.X"),
            shift_limits_for_warp_y: EtomoNumber::new_with_name("ShiftLimitsForWarp.Y"),
            warp_patch_size_x: EtomoNumber::new_with_name("WarpPatchSize.X"),
            warp_patch_size_y: EtomoNumber::new_with_name("WarpPatchSize.Y"),
            boundary_model: EtomoBoolean2::new_with_name("BoundaryModel"),
            find_warping: EtomoBoolean2::new_with_name("FindWarping"),
            sobel_filter: EtomoBoolean2::new_with_name("SobelFilter"),
            sigma_low_frequency_enabled: EtomoBoolean2::new_with_name("SigmaLowFrequency.Enabled"),
            cutoff_high_frequency_enabled: EtomoBoolean2::new_with_name(
                "CutoffHighFrequency.Enabled",
            ),
            sigma_high_frequency_enabled: EtomoBoolean2::new_with_name(
                "SigmaHighFrequency.Enabled",
            ),
            align_transform: Transform::DEFAULT,
        };
        instance.sigma_low_frequency.set_default_int(0);
        instance.cutoff_high_frequency.set_default_int(0);
        instance.sigma_high_frequency.set_default_int(0);
        instance
    }

    /// Java private `createPrepend(String)`.
    fn create_prepend(&self, prepend: &str) -> String {
        if prepend.is_empty() {
            return GROUP_KEY.to_string();
        }
        format!("{}.{}", prepend, GROUP_KEY)
    }

    /// Java package-private `load(Properties, String)`.
    ///
    /// `Transform.load` is translated over a `HashMap`; the properties are handed to it
    /// as one.
    pub fn load(&mut self, props: &BTreeMap<String, String>, prepend: &str) {
        let prepend = self.create_prepend(prepend);
        // reset
        self.sigma_low_frequency.reset();
        self.cutoff_high_frequency.reset();
        self.sigma_high_frequency.reset();
        self.align_transform = Transform::DEFAULT;
        self.reduce_by_binning.reset();
        self.edge_to_ignore.reset();
        self.midas_binning.reset();
        self.skip_sections_from1.reset();
        self.pre_cross_correlation.reset();
        self.shift_limits_for_warp_x.reset();
        self.shift_limits_for_warp_y.reset();
        self.warp_patch_size_x.reset();
        self.warp_patch_size_y.reset();
        self.boundary_model.reset();
        self.find_warping.reset();
        self.sobel_filter.reset();
        self.sigma_low_frequency_enabled.reset();
        self.cutoff_high_frequency_enabled.reset();
        self.sigma_high_frequency_enabled.reset();
        // load
        self.sigma_low_frequency
            .load_with_prepend(props, Some(&prepend));
        self.cutoff_high_frequency
            .load_with_prepend(props, Some(&prepend));
        self.sigma_high_frequency
            .load_with_prepend(props, Some(&prepend));
        let hash_props: HashMap<String, String> =
            props.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        self.align_transform = Transform::load(
            &hash_props,
            &prepend,
            ALIGN_TRANFORM_KEY,
            Transform::DEFAULT,
        );
        self.reduce_by_binning
            .load_with_prepend(props, Some(&prepend));
        self.edge_to_ignore.load_with_prepend(props, Some(&prepend));
        self.midas_binning.load_with_prepend(props, Some(&prepend));
        self.skip_sections_from1
            .load_with_prepend(props, Some(&prepend));
        self.pre_cross_correlation
            .load_with_prepend(props, Some(&prepend));
        self.shift_limits_for_warp_x
            .load_with_prepend(props, Some(&prepend));
        self.shift_limits_for_warp_y
            .load_with_prepend(props, Some(&prepend));
        self.warp_patch_size_x
            .load_with_prepend(props, Some(&prepend));
        self.warp_patch_size_y
            .load_with_prepend(props, Some(&prepend));
        self.boundary_model.load_with_prepend(props, Some(&prepend));
        self.find_warping.load_with_prepend(props, Some(&prepend));
        self.sobel_filter.load_with_prepend(props, Some(&prepend));
        self.sigma_low_frequency_enabled
            .load_with_prepend(props, Some(&prepend));
        self.cutoff_high_frequency_enabled
            .load_with_prepend(props, Some(&prepend));
        self.sigma_high_frequency_enabled
            .load_with_prepend(props, Some(&prepend));
    }

    /// Java package-private `store(Properties, String)`.
    ///
    /// `Transform.store` is translated over a `HashMap`; its one key is written back
    /// into the properties (or removed) the way it wrote or removed it there.
    pub fn store(&self, props: &mut BTreeMap<String, String>, prepend: &str) {
        let prepend = self.create_prepend(prepend);
        // `EtomoVersion.store(Properties, String)`, its `Storable` implementation.  The
        // trait is not imported: `EtomoNumber` implements it too, and its `&str` form would
        // shadow `ConstEtomoNumber.store_with_prepend` in the calls below.
        crate::imod::etomo::storage::storable::StorableValue::store_with_prepend(
            &*CURRENT_VERSION,
            props,
            &prepend,
        );
        self.sigma_low_frequency
            .store_with_prepend(props, Some(&prepend));
        self.cutoff_high_frequency
            .store_with_prepend(props, Some(&prepend));
        self.sigma_high_frequency
            .store_with_prepend(props, Some(&prepend));
        let mut hash_props: HashMap<String, String> = HashMap::new();
        Transform::store(
            Some(self.align_transform),
            &mut hash_props,
            &prepend,
            ALIGN_TRANFORM_KEY,
        );
        for (key, value) in hash_props {
            props.insert(key, value);
        }
        self.reduce_by_binning
            .store_with_prepend(props, Some(&prepend));
        self.edge_to_ignore
            .store_with_prepend(props, Some(&prepend));
        self.midas_binning.store_with_prepend(props, Some(&prepend));
        self.skip_sections_from1
            .store_with_prepend(props, Some(&prepend));
        self.pre_cross_correlation
            .store_with_prepend(props, Some(&prepend));
        self.shift_limits_for_warp_x
            .store_with_prepend(props, Some(&prepend));
        self.shift_limits_for_warp_y
            .store_with_prepend(props, Some(&prepend));
        self.warp_patch_size_x
            .store_with_prepend(props, Some(&prepend));
        self.warp_patch_size_y
            .store_with_prepend(props, Some(&prepend));
        self.boundary_model
            .store_with_prepend(props, Some(&prepend));
        self.find_warping.store_with_prepend(props, Some(&prepend));
        self.sobel_filter.store_with_prepend(props, Some(&prepend));
        self.sigma_low_frequency_enabled
            .store_with_prepend(props, Some(&prepend));
        self.cutoff_high_frequency_enabled
            .store_with_prepend(props, Some(&prepend));
        self.sigma_high_frequency_enabled
            .store_with_prepend(props, Some(&prepend));
    }

    /// Java `isSobelFilter`.
    pub fn is_sobel_filter(&self) -> bool {
        self.sobel_filter.is()
    }

    /// Java `setSobelFilter(boolean)`.
    pub fn set_sobel_filter(&mut self, input: bool) {
        self.sobel_filter.set_boolean(input);
    }

    /// Java `setSigmaLowFrequency(String)`.
    pub fn set_sigma_low_frequency_string(
        &mut self,
        sigma_low_frequency: Option<&str>,
    ) -> ConstEtomoNumber {
        self.sigma_low_frequency.set_string(sigma_low_frequency);
        self.sigma_low_frequency.base.base.clone()
    }

    /// Java `setSigmaLowFrequencyEnabled(boolean)`.
    pub fn set_sigma_low_frequency_enabled(&mut self, input: bool) {
        self.sigma_low_frequency_enabled.set_boolean(input);
    }

    /// Java `isSigmaLowFrequencyEnabled`.
    pub fn is_sigma_low_frequency_enabled(&self) -> bool {
        self.sigma_low_frequency_enabled.is()
    }

    /// Java `isSigmaLowFrequencyNull`.
    pub fn is_sigma_low_frequency_null(&self) -> bool {
        self.sigma_low_frequency.is_null()
    }

    /// Java package-private `setSigmaLowFrequency(ConstEtomoNumber)`.
    pub fn set_sigma_low_frequency(&mut self, input: Option<&ConstEtomoNumber>) {
        self.sigma_low_frequency.set_const_etomo_number(input);
    }

    /// Java `getSigmaLowFrequency`.
    pub fn get_sigma_low_frequency(&self) -> &ConstEtomoNumber {
        &self.sigma_low_frequency.base.base
    }

    /// Java `getSigmaLowFrequencyParameter`.
    pub fn get_sigma_low_frequency_parameter(&self) -> &ScriptParameter {
        &self.sigma_low_frequency
    }

    /// Java `setCutoffHighFrequency(String)`.
    pub fn set_cutoff_high_frequency_string(&mut self, cutoff_high_frequency: Option<&str>) {
        self.cutoff_high_frequency.set_string(cutoff_high_frequency);
    }

    /// Java `setCutoffHighFrequencyEnabled(boolean)`.
    pub fn set_cutoff_high_frequency_enabled(&mut self, input: bool) {
        self.cutoff_high_frequency_enabled.set_boolean(input);
    }

    /// Java `isCutoffHighFrequencyEnabled`.
    pub fn is_cutoff_high_frequency_enabled(&self) -> bool {
        self.cutoff_high_frequency_enabled.is()
    }

    /// Java `isCutoffHighFrequencyNull`.
    pub fn is_cutoff_high_frequency_null(&self) -> bool {
        self.cutoff_high_frequency.is_null()
    }

    /// Java package-private `setCutoffHighFrequency(ConstEtomoNumber)`.
    pub fn set_cutoff_high_frequency(&mut self, input: Option<&ConstEtomoNumber>) {
        self.cutoff_high_frequency.set_const_etomo_number(input);
    }

    /// Java `getReduceByBinning`.
    pub fn get_reduce_by_binning(&self) -> &ConstEtomoNumber {
        &self.reduce_by_binning.base
    }

    /// Java `setReduceByBinning(Number)`.
    pub fn set_reduce_by_binning(&mut self, input: Option<Number>) {
        self.reduce_by_binning.set_number(input);
    }

    /// Java `isReduceByBinningNull`.
    pub fn is_reduce_by_binning_null(&self) -> bool {
        self.reduce_by_binning.is_null()
    }

    /// Java `setShiftLimitsForWarpX(String)`.
    pub fn set_shift_limits_for_warp_x(&mut self, input: Option<&str>) {
        self.shift_limits_for_warp_x.set_string(input);
    }

    /// Java `getShiftLimitsForWarpX`.
    pub fn get_shift_limits_for_warp_x(&self) -> String {
        self.shift_limits_for_warp_x.to_string()
    }

    /// Java `setShiftLimitsForWarpY(String)`.
    pub fn set_shift_limits_for_warp_y(&mut self, input: Option<&str>) {
        self.shift_limits_for_warp_y.set_string(input);
    }

    /// Java `getShiftLimitsForWarpY`.
    pub fn get_shift_limits_for_warp_y(&self) -> String {
        self.shift_limits_for_warp_y.to_string()
    }

    /// Java `setWarpPatchSizeX(String)`.
    pub fn set_warp_patch_size_x(&mut self, input: Option<&str>) {
        self.warp_patch_size_x.set_string(input);
    }

    /// Java `getWarpPatchSizeX`.
    pub fn get_warp_patch_size_x(&self) -> String {
        self.warp_patch_size_x.to_string()
    }

    /// Java `setWarpPatchSizeY(String)`.
    pub fn set_warp_patch_size_y(&mut self, input: Option<&str>) {
        self.warp_patch_size_y.set_string(input);
    }

    /// Java `getWarpPatchSizeY`.
    pub fn get_warp_patch_size_y(&self) -> String {
        self.warp_patch_size_y.to_string()
    }

    /// Java `setBoundaryModel(boolean)`.
    pub fn set_boundary_model(&mut self, input: bool) {
        self.boundary_model.set_boolean(input);
    }

    /// Java `isBoundaryModel`.
    pub fn is_boundary_model(&self) -> bool {
        self.boundary_model.is()
    }

    /// Java `setFindWarping(boolean)`.
    pub fn set_find_warping(&mut self, input: bool) {
        self.find_warping.set_boolean(input);
    }

    /// Java `isFindWarping`.
    pub fn is_find_warping(&self) -> bool {
        self.find_warping.is()
    }

    /// Java `getEdgeToIgnore`.
    pub fn get_edge_to_ignore(&self) -> f64 {
        self.edge_to_ignore.get_double()
    }

    /// Java `setEdgeToIgnore(String)`.
    pub fn set_edge_to_ignore(&mut self, input: Option<&str>) {
        self.edge_to_ignore.set_string(input);
    }

    /// Java `isEdgeToIgnoreNull`.
    pub fn is_edge_to_ignore_null(&self) -> bool {
        self.edge_to_ignore.is_null()
    }

    /// Java `getMidasBinning`.
    pub fn get_midas_binning(&self) -> &ConstEtomoNumber {
        &self.midas_binning.base
    }

    /// Java `setMidasBinning(Number)`.
    pub fn set_midas_binning(&mut self, input: Option<Number>) {
        self.midas_binning.set_number(input);
    }

    /// Java `isMidasBinningNull`.
    pub fn is_midas_binning_null(&self) -> bool {
        self.midas_binning.is_null()
    }

    /// Java `getSkipSectionsFrom1`.
    pub fn get_skip_sections_from1(&self) -> String {
        self.skip_sections_from1.to_string()
    }

    /// Java `setSkipSectionsFrom1(String)`.
    pub fn set_skip_sections_from1(&mut self, input: Option<&str>) {
        self.skip_sections_from1.set_string(input);
    }

    /// Java `isPreCrossCorrelation`.
    pub fn is_pre_cross_correlation(&self) -> bool {
        self.pre_cross_correlation.is()
    }

    /// Java `setPreCrossCorrelation(boolean)`.
    pub fn set_pre_cross_correlation(&mut self, input: bool) {
        self.pre_cross_correlation.set_boolean(input);
    }

    /// Java `getCutoffHighFrequency`.
    pub fn get_cutoff_high_frequency(&self) -> &ConstEtomoNumber {
        &self.cutoff_high_frequency.base.base
    }

    /// Java `getCutoffHighFrequencyParameter`.
    pub fn get_cutoff_high_frequency_parameter(&self) -> &ScriptParameter {
        &self.cutoff_high_frequency
    }

    /// Java `setSigmaHighFrequency(String)`.
    pub fn set_sigma_high_frequency_string(&mut self, sigma_high_frequency: Option<&str>) {
        self.sigma_high_frequency.set_string(sigma_high_frequency);
    }

    /// Java `setSigmaHighFrequencyEnabled(boolean)`.
    pub fn set_sigma_high_frequency_enabled(&mut self, input: bool) {
        self.sigma_high_frequency_enabled.set_boolean(input);
    }

    /// Java `isSigmaHighFrequencyEnabled`.
    pub fn is_sigma_high_frequency_enabled(&self) -> bool {
        self.sigma_high_frequency_enabled.is()
    }

    /// Java `isSigmaHighFrequencyNull`.
    pub fn is_sigma_high_frequency_null(&self) -> bool {
        self.sigma_high_frequency.is_null()
    }

    /// Java package-private `setSigmaHighFrequency(ConstEtomoNumber)`.
    pub fn set_sigma_high_frequency(&mut self, input: Option<&ConstEtomoNumber>) {
        self.sigma_high_frequency.set_const_etomo_number(input);
    }

    /// Java `getSigmaHighFrequency`.
    pub fn get_sigma_high_frequency(&self) -> &ConstEtomoNumber {
        &self.sigma_high_frequency.base.base
    }

    /// Java `getSigmaHighFrequencyParameter`.
    pub fn get_sigma_high_frequency_parameter(&self) -> &ScriptParameter {
        &self.sigma_high_frequency
    }

    /// Java `setAlignTransform(Transform)`.
    pub fn set_align_transform(&mut self, align_transform: Transform) {
        self.align_transform = align_transform;
    }

    /// Java `getAlignTransform`.
    pub fn get_align_transform(&self) -> Transform {
        self.align_transform
    }
}

impl Default for AutoAlignmentMetaData {
    fn default() -> Self {
        AutoAlignmentMetaData::new()
    }
}
