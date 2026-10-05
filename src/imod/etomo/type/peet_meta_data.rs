//! `IMOD/Etomo/src/etomo/type/PeetMetaData.java`.
//!
//! The data file (`.epe`) of the PEET interface.  `PeetMetaData extends BaseMetaData
//! implements ConstPeetMetaData`: the superclass fields are the embedded
//! `BaseMetaDataBase`, and the abstract/overridden members are the `BaseMetaData`
//! impl.  The object is shared by its manager (and process threads), so every field
//! sits in a `Mutex` and the setters take `&self`.  Getters of `ConstEtomoNumber`
//! fields return a copy of the number.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::{LazyLock, Mutex};

use super::axis_type::AxisType;
use super::base_meta_data::{BaseMetaData, BaseMetaDataBase};
use super::const_etomo_number::{ConstEtomoNumber, Number};
use super::const_int_key_list::ConstIntKeyList;
use super::const_peet_meta_data::ConstPeetMetaData;
use super::const_string_property::ConstStringProperty;
use super::data_file_type::DataFileType;
use super::double_key_list::DoubleKeyList;
use super::etomo_boolean2::EtomoBoolean2;
use super::etomo_number::EtomoNumber;
use super::etomo_version::EtomoVersion;
use super::int_key_list::IntKeyList;
use super::string_property::StringProperty;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::logic::multiparticle_reference;
use crate::imod::etomo::storage::storable::{Storable, StorableValue};
use crate::imod::etomo::ui::log_properties::LogProperties;
use crate::imod::etomo::util::dataset_files;
use crate::imod::etomo::util::utilities;

/// Java `NEW_TITLE`.
pub const NEW_TITLE: &str = "PEET";

// do not change these unless backward compatibility work is done
/// Java private static final `TILT_RANGE_KEY`.
const TILT_RANGE_KEY: &str = "TiltRange";
/// Java private static final `GROUP_KEY`.
const GROUP_KEY: &str = "Peet";
/// Java private static final `REFERENCE_KEY`.
const REFERENCE_KEY: &str = "Reference";
/// Java private static final `VERSION_1_1`.
static VERSION_1_1: LazyLock<EtomoVersion> =
    LazyLock::new(|| EtomoVersion::get_default_instance_with_version(Some("1.1")));
/// Java private static final `VERSION_1_2`.
static VERSION_1_2: LazyLock<EtomoVersion> =
    LazyLock::new(|| EtomoVersion::get_default_instance_with_version(Some("1.2")));
/// Java private static final `START_KEY`.
const START_KEY: &str = "Start";
/// Java private static final `END_KEY`.
const END_KEY: &str = "End";
/// Java private static final `MASK_MODEL_PTS_KEY`.
const MASK_MODEL_PTS_KEY: &str = "MaskModelPts";
/// Java private static final `MODEL_NUMBER_KEY`.
const MODEL_NUMBER_KEY: &str = "ModelNumber";
/// Java private static final `PARTICLE_KEY`.
const PARTICLE_KEY: &str = "Particle";
/// Java private static final `VOLUME_KEY`.
const VOLUME_KEY: &str = "Volume";
/// Java private static final `LOW_CUTOFF_KEY`.
const LOW_CUTOFF_KEY: &str = "LowCutoff";
/// Java private static final `CUTOFF_KEY`.
const CUTOFF_KEY: &str = "Cutoff";
/// Java private static final `SIGMA_KEY`.
const SIGMA_KEY: &str = "Sigma";

/// Java private static final `LATEST_VERSION = VERSION_1_2`.
fn latest_version() -> &'static EtomoVersion {
    &VERSION_1_2
}

/// Java `public class PeetMetaData extends BaseMetaData implements
/// ConstPeetMetaData`.
pub struct PeetMetaData {
    /// Java superclass `BaseMetaData` state.
    base: BaseMetaDataBase,
    // do not change the names of these veriables unless backward compatibility work is
    // done
    /// Java private final `rootName`.
    root_name: Mutex<StringProperty>,
    /// Java private final `initMotlFile`.
    init_motl_file: Mutex<IntKeyList>,
    /// Java private final `tiltRangeMultiAxesFile`.
    tilt_range_multi_axes_file: Mutex<IntKeyList>,
    /// Java private final `tiltRangeMin`.
    tilt_range_min: Mutex<IntKeyList>,
    /// Java private final `tiltRangeMax`.
    tilt_range_max: Mutex<IntKeyList>,
    /// Java private final `referenceVolume`.
    reference_volume: Mutex<EtomoNumber>,
    /// Java private final `referenceParticle`.
    reference_particle: Mutex<EtomoNumber>,
    /// Java private final `referenceFile`.
    reference_file: Mutex<StringProperty>,
    /// Java private final `edgeShift`.
    edge_shift: Mutex<EtomoNumber>,
    /// Java private final `flgWedgeWeight`.
    flg_wedge_weight: Mutex<EtomoBoolean2>,
    /// Java private final `lowCutoffCutoff`.
    low_cutoff_cutoff: Mutex<DoubleKeyList>,
    /// Java private final `lowCutoffSigma`.
    low_cutoff_sigma: Mutex<DoubleKeyList>,
    /// Java private final `isLowCutoff`.
    is_low_cutoff: Mutex<EtomoBoolean2>,
    /// Java private `lowCutoffBackwardsCompatibility`, initially false.
    low_cutoff_backwards_compatibility: Mutex<bool>,
    /// Java private final `maskUseReferenceParticle` (deprecated).
    mask_use_reference_particle: Mutex<EtomoBoolean2>,
    /// Java private final `maskModelPtsModelNumber` (deprecated).
    mask_model_pts_model_number: Mutex<EtomoNumber>,
    /// Java private final `maskModelPtsZRotation`.
    mask_model_pts_z_rotation: Mutex<EtomoNumber>,
    /// Java private final `maskModelPtsParticle` (deprecated).
    mask_model_pts_particle: Mutex<EtomoNumber>,
    /// Java private final `maskModelPtsYRotation`.
    mask_model_pts_y_rotation: Mutex<EtomoNumber>,
    /// Java private final `maskTypeVolume`.
    mask_type_volume: Mutex<StringProperty>,
    /// Java private final `nWeightGroup`.
    n_weight_group: Mutex<EtomoNumber>,
    /// Java private final `tiltRange`.
    tilt_range: Mutex<EtomoBoolean2>,
    /// Java private final `manualCylinderOrientation`.
    manual_cylinder_orientation: Mutex<EtomoBoolean2>,
    /// Java private final `referenceMultiparticleLevel`.
    reference_multiparticle_level: Mutex<EtomoNumber>,
    /// Java private final `tiltRangeMultiAxes`.
    tilt_range_multi_axes: Mutex<EtomoBoolean2>,
    /// Java private final `cylinderHeight`.
    cylinder_height: Mutex<EtomoNumber>,
    /// Java private final `maskBlurStdDev`.
    mask_blur_std_dev: Mutex<EtomoNumber>,
    /// Java private final `browsingDirectory`.
    browsing_directory: Mutex<StringProperty>,
    /// Java private final `flgAlignAverages`.
    flg_align_averages: Mutex<EtomoBoolean2>,
    /// Java private final `cNSymmetricAveraging`.
    c_n_symmetric_averaging: Mutex<EtomoNumber>,
}

// Safety: as for `ParallelMetaData`: every field is a `Mutex` of owned data except the
// `&'static` references in `BaseMetaDataBase`, whose log-properties reference is never
// called through off the event dispatch thread.
unsafe impl Send for PeetMetaData {}
unsafe impl Sync for PeetMetaData {}

impl PeetMetaData {
    /// Java `PeetMetaData(BaseManager, LogProperties, boolean)`, with the field
    /// initialisers Java runs before its body.
    pub fn new(
        manager: Option<&'static dyn BaseManager>,
        log_properties: Option<&'static dyn LogProperties>,
        new_dataset: bool,
    ) -> PeetMetaData {
        let instance = PeetMetaData {
            base: BaseMetaDataBase::new_force_old_style(
                manager,
                log_properties,
                true,
                new_dataset,
                true,
            ),
            root_name: Mutex::new(StringProperty::new_with_key(Some("RootName"))),
            init_motl_file: Mutex::new(IntKeyList::get_string_instance_with_key("InitMotlFile")),
            tilt_range_multi_axes_file: Mutex::new(IntKeyList::get_string_instance_with_key(
                "TiltRangeMultiAxesFile",
            )),
            tilt_range_min: Mutex::new(IntKeyList::get_string_instance_with_key(&format!(
                "{TILT_RANGE_KEY}.{START_KEY}"
            ))),
            tilt_range_max: Mutex::new(IntKeyList::get_string_instance_with_key(&format!(
                "{TILT_RANGE_KEY}.{END_KEY}"
            ))),
            reference_volume: Mutex::new(EtomoNumber::new_with_name(&format!(
                "{REFERENCE_KEY}.{VOLUME_KEY}"
            ))),
            reference_particle: Mutex::new(EtomoNumber::new_with_name(&format!(
                "{REFERENCE_KEY}.{PARTICLE_KEY}"
            ))),
            reference_file: Mutex::new(StringProperty::new_with_key(Some(&format!(
                "{REFERENCE_KEY}.File"
            )))),
            edge_shift: Mutex::new(EtomoNumber::new_with_name("EdgeShift")),
            flg_wedge_weight: Mutex::new(EtomoBoolean2::new_with_name("FlgWedgeWeight")),
            low_cutoff_cutoff: Mutex::new(DoubleKeyList::get_string_instance(&format!(
                "{LOW_CUTOFF_KEY}.{CUTOFF_KEY}"
            ))),
            low_cutoff_sigma: Mutex::new(DoubleKeyList::get_string_instance(&format!(
                "{LOW_CUTOFF_KEY}.{SIGMA_KEY}"
            ))),
            is_low_cutoff: Mutex::new(EtomoBoolean2::new_with_name("IsLowCutoff")),
            low_cutoff_backwards_compatibility: Mutex::new(false),
            mask_use_reference_particle: Mutex::new(EtomoBoolean2::new_with_name(
                "Mask.UseReferenceParticle",
            )),
            mask_model_pts_model_number: Mutex::new(EtomoNumber::new_with_name(&format!(
                "{MASK_MODEL_PTS_KEY}.{MODEL_NUMBER_KEY}"
            ))),
            mask_model_pts_z_rotation: Mutex::new(EtomoNumber::new_with_name(&format!(
                "{MASK_MODEL_PTS_KEY}.ZRotation"
            ))),
            mask_model_pts_particle: Mutex::new(EtomoNumber::new_with_name(&format!(
                "{MASK_MODEL_PTS_KEY}.{PARTICLE_KEY}"
            ))),
            mask_model_pts_y_rotation: Mutex::new(EtomoNumber::new_with_name(&format!(
                "{MASK_MODEL_PTS_KEY}.YRotation"
            ))),
            mask_type_volume: Mutex::new(StringProperty::new_with_key(Some(&format!(
                "MastType.{VOLUME_KEY}"
            )))),
            n_weight_group: Mutex::new(EtomoNumber::new_with_name("NWeightGroup")),
            tilt_range: Mutex::new(EtomoBoolean2::new_with_name("TiltRange")),
            manual_cylinder_orientation: Mutex::new(EtomoBoolean2::new_with_name(
                "MaskType.ManualCylinderOrientation",
            )),
            reference_multiparticle_level: Mutex::new(EtomoNumber::new_with_name(&format!(
                "{REFERENCE_KEY}.Multiparticle.level"
            ))),
            tilt_range_multi_axes: Mutex::new(EtomoBoolean2::new_with_name("TiltRangeMultiAxes")),
            cylinder_height: Mutex::new(EtomoNumber::new_with_name("CylinderHeight")),
            mask_blur_std_dev: Mutex::new(EtomoNumber::new_with_name("MaskBlurStdDev")),
            browsing_directory: Mutex::new(
                StringProperty::new_with_key_and_return_null_when_empty(
                    Some("BrowsingDirectory"),
                    true,
                ),
            ),
            flg_align_averages: Mutex::new(EtomoBoolean2::new_with_name("FlgAlignAverages")),
            c_n_symmetric_averaging: Mutex::new(EtomoNumber::new_with_name("CNSymmetricAveraging")),
        };
        *instance.base.file_extension.lock().unwrap() =
            DataFileType::Peet.extension().map(str::to_owned);
        *instance.base.axis_type.lock().unwrap() = AxisType::SingleAxis;
        instance
            .reference_multiparticle_level
            .lock()
            .unwrap()
            .set_default_int(multiparticle_reference::DEFAULT_LEVEL);
        instance
    }

    /// Java `copy(PeetMetaData)`.
    pub fn copy(&self, input: &PeetMetaData) {
        let root_name = input.root_name.lock().unwrap();
        self.root_name
            .lock()
            .unwrap()
            .set_string_property(&root_name);
        let list = input.init_motl_file.lock().unwrap().clone();
        let mut init_motl_file = self.init_motl_file.lock().unwrap();
        init_motl_file.reset();
        init_motl_file.set_int_key_list(Some(&list));
        drop(init_motl_file);
        let list = input.tilt_range_min.lock().unwrap().clone();
        let mut tilt_range_min = self.tilt_range_min.lock().unwrap();
        tilt_range_min.reset();
        tilt_range_min.set_int_key_list(Some(&list));
        drop(tilt_range_min);
        let list = input.tilt_range_max.lock().unwrap().clone();
        let mut tilt_range_max = self.tilt_range_max.lock().unwrap();
        tilt_range_max.reset();
        tilt_range_max.set_int_key_list(Some(&list));
        drop(tilt_range_max);
        copy_number(&self.reference_volume, &input.reference_volume);
        copy_number(&self.reference_particle, &input.reference_particle);
        let value = input.reference_file.lock().unwrap();
        self.reference_file
            .lock()
            .unwrap()
            .set_string_property(&value);
        copy_number(&self.edge_shift, &input.edge_shift);
        copy_boolean(&self.flg_wedge_weight, &input.flg_wedge_weight);
        copy_number(
            &self.mask_model_pts_z_rotation,
            &input.mask_model_pts_z_rotation,
        );
        copy_number(
            &self.mask_model_pts_y_rotation,
            &input.mask_model_pts_y_rotation,
        );
        let value = input.mask_type_volume.lock().unwrap();
        self.mask_type_volume
            .lock()
            .unwrap()
            .set_string_property(&value);
        copy_number(&self.n_weight_group, &input.n_weight_group);
        copy_boolean(&self.tilt_range, &input.tilt_range);
        let revision_number = input.base.revision_number.lock().unwrap().clone();
        self.base
            .revision_number
            .lock()
            .unwrap()
            .set_etomo_version(Some(&revision_number));
        copy_boolean(
            &self.manual_cylinder_orientation,
            &input.manual_cylinder_orientation,
        );
        copy_number(
            &self.reference_multiparticle_level,
            &input.reference_multiparticle_level,
        );
        copy_boolean(&self.tilt_range_multi_axes, &input.tilt_range_multi_axes);
        let list = input.tilt_range_multi_axes_file.lock().unwrap().clone();
        let mut tilt_range_multi_axes_file = self.tilt_range_multi_axes_file.lock().unwrap();
        tilt_range_multi_axes_file.reset();
        tilt_range_multi_axes_file.set_int_key_list(Some(&list));
        drop(tilt_range_multi_axes_file);
        copy_number(&self.cylinder_height, &input.cylinder_height);
        copy_number(&self.mask_blur_std_dev, &input.mask_blur_std_dev);
        let value = input.browsing_directory.lock().unwrap();
        self.browsing_directory
            .lock()
            .unwrap()
            .set_string_property(&value);
        let list = input.low_cutoff_cutoff.lock().unwrap().clone();
        let mut low_cutoff_cutoff = self.low_cutoff_cutoff.lock().unwrap();
        low_cutoff_cutoff.reset();
        low_cutoff_cutoff.set(Some(&list));
        drop(low_cutoff_cutoff);
        let list = input.low_cutoff_sigma.lock().unwrap().clone();
        let mut low_cutoff_sigma = self.low_cutoff_sigma.lock().unwrap();
        low_cutoff_sigma.reset();
        low_cutoff_sigma.set(Some(&list));
        drop(low_cutoff_sigma);
        copy_boolean(&self.is_low_cutoff, &input.is_low_cutoff);
        copy_boolean(&self.flg_align_averages, &input.flg_align_averages);
        copy_number(
            &self.c_n_symmetric_averaging,
            &input.c_n_symmetric_averaging,
        );
    }

    /// Java `toString()`.
    pub fn to_string_java(&self) -> String {
        format!(
            "[rootName:{},{}]",
            self.root_name.lock().unwrap(),
            self.base
        )
    }

    /// Java `setName(String)`.
    pub fn set_name(&self, name: Option<&str>) {
        self.root_name.lock().unwrap().set(name);
        let root_name = self.root_name.lock().unwrap().to_string_option();
        utilities::manager_stamp(None, root_name.as_deref());
    }

    /// Java `validate()`.  returns null if valid.
    pub fn validate(&self) -> Option<String> {
        let root_name = self.root_name.lock().unwrap();
        if root_name.is_empty() {
            return Some("Missing root name.".to_owned());
        }
        let text = root_name.to_string_option().unwrap_or_default();
        // File.pathSeparatorChar and File.separatorChar.
        if text.contains(':') || text.contains('/') {
            return Some(format!("Invalid root name, {root_name}."));
        }
        None
    }

    /// Java `load(Properties)`.
    pub fn load(&self, props: &mut BTreeMap<String, String>) {
        self.load_with_prepend(props, "");
    }

    /// Java `load(Properties, String)`.
    pub fn load_with_prepend(&self, props: &mut BTreeMap<String, String>, prepend: &str) {
        // super.load(props, prepend)
        let created = self.create_prepend(prepend);
        if self
            .base
            .load_with_created_prepend(props, created.as_deref())
        {
            self.check_image_filename_style_loaded(created.as_deref().unwrap_or("null"));
        }
        // reset
        self.init_motl_file.lock().unwrap().reset();
        self.tilt_range_min.lock().unwrap().reset();
        self.tilt_range_max.lock().unwrap().reset();
        self.reference_volume.lock().unwrap().reset();
        self.reference_particle.lock().unwrap().reset();
        self.reference_file.lock().unwrap().reset();
        self.n_weight_group.lock().unwrap().reset();
        self.tilt_range.lock().unwrap().reset();
        self.base.revision_number.lock().unwrap().reset();
        self.reference_multiparticle_level.lock().unwrap().reset();
        self.tilt_range_multi_axes.lock().unwrap().reset();
        self.tilt_range_multi_axes_file.lock().unwrap().reset();
        self.cylinder_height.lock().unwrap().reset();
        self.mask_blur_std_dev.lock().unwrap().reset();
        self.browsing_directory.lock().unwrap().reset();
        // (The source resets lowCutoffCutoff twice and lowCutoffSigma not at all.)
        self.low_cutoff_cutoff.lock().unwrap().reset();
        self.low_cutoff_cutoff.lock().unwrap().reset();
        self.flg_align_averages.lock().unwrap().reset();
        // load
        let prepend = created.unwrap_or_else(|| "null".to_owned());
        let p = Some(prepend.as_str());
        // `StringProperty.load` may remove a backward-compatible key; none is declared
        // here, so the string properties load from a copy.
        let mut props_copy = props.clone();
        self.root_name
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), p);
        self.init_motl_file.lock().unwrap().load(props, &prepend);
        self.tilt_range_min.lock().unwrap().load(props, &prepend);
        self.tilt_range_max.lock().unwrap().load(props, &prepend);
        self.reference_volume
            .lock()
            .unwrap()
            .load_with_prepend(props, p);
        self.reference_particle
            .lock()
            .unwrap()
            .load_with_prepend(props, p);
        self.reference_file
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), p);
        self.edge_shift.lock().unwrap().load_with_prepend(props, p);
        self.flg_wedge_weight
            .lock()
            .unwrap()
            .load_with_prepend(props, p);
        self.mask_model_pts_z_rotation
            .lock()
            .unwrap()
            .load_with_prepend(props, p);
        self.mask_model_pts_y_rotation
            .lock()
            .unwrap()
            .load_with_prepend(props, p);
        self.mask_type_volume
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), p);
        self.n_weight_group
            .lock()
            .unwrap()
            .load_with_prepend(props, p);
        self.tilt_range.lock().unwrap().load_with_prepend(props, p);
        self.manual_cylinder_orientation
            .lock()
            .unwrap()
            .load_with_prepend(props, p);
        self.reference_multiparticle_level
            .lock()
            .unwrap()
            .load_with_prepend(props, p);
        self.tilt_range_multi_axes
            .lock()
            .unwrap()
            .load_with_prepend(props, p);
        self.tilt_range_multi_axes_file
            .lock()
            .unwrap()
            .load(props, &prepend);
        self.cylinder_height
            .lock()
            .unwrap()
            .load_with_prepend(props, p);
        self.mask_blur_std_dev
            .lock()
            .unwrap()
            .load_with_prepend(props, p);
        self.browsing_directory
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), p);
        self.low_cutoff_cutoff.lock().unwrap().load(props, &prepend);
        self.low_cutoff_sigma.lock().unwrap().load(props, &prepend);
        self.is_low_cutoff
            .lock()
            .unwrap()
            .load_with_prepend(props, p);
        self.flg_align_averages
            .lock()
            .unwrap()
            .load_with_prepend(props, p);
        self.c_n_symmetric_averaging
            .lock()
            .unwrap()
            .load_with_prepend(props, p);

        self.base
            .revision_number
            .lock()
            .unwrap()
            .load_with_prepend(props, &prepend);
        let revision_number = self.base.revision_number.lock().unwrap().clone();
        if revision_number.is_null() || revision_number.lt(Some(latest_version())) {
            // backwards compatibility
            if revision_number.is_null() || revision_number.lt(Some(&VERSION_1_1)) {
                self.load1_0(props, &prepend);
            }
            if revision_number.is_null() || revision_number.lt(Some(&VERSION_1_2)) {
                self.load1_1(props, &prepend);
            }
            self.base
                .revision_number
                .lock()
                .unwrap()
                .set_etomo_version(Some(latest_version()));
        }
    }

    /// Java `load1_0(Properties, String)`.  Backwards compatability function for
    /// versions earlier then 1.1.
    ///
    /// The source removes the old keys from the `Properties` it was given; this
    /// translation's `props` is read-only here, so they are removed from a copy that
    /// is dropped.  The keys are rewritten under their new names on the next store,
    /// and the stale ones (`"TILT_RANGE_KEY.Start"` - the source quotes the constant's
    /// name) stay in the file, which is harmless.
    pub fn load1_0(&self, props: &BTreeMap<String, String>, prepend: &str) {
        let mut props_copy = props.clone();
        let key = format!("TILT_RANGE_KEY.{START_KEY}");
        self.tilt_range_min
            .lock()
            .unwrap()
            .load_with_temp_key(props, prepend, &key);
        self.tilt_range_min.lock().unwrap().remove_with_temp_key(
            &mut props_copy,
            prepend,
            Some(&key),
        );
        let key = format!("TILT_RANGE_KEY.{END_KEY}");
        self.tilt_range_max
            .lock()
            .unwrap()
            .load_with_temp_key(props, prepend, &key);
        self.tilt_range_max.lock().unwrap().remove_with_temp_key(
            &mut props_copy,
            prepend,
            Some(&key),
        );
    }

    /// Java `load1_1(Properties, String)`.
    pub fn load1_1(&self, _props: &BTreeMap<String, String>, _prepend: &str) {
        *self.low_cutoff_backwards_compatibility.lock().unwrap() = true;
    }

    /// Java `isLowCutoffBackwardsCompatibility()`.
    pub fn is_low_cutoff_backwards_compatibility(&self) -> bool {
        *self.low_cutoff_backwards_compatibility.lock().unwrap()
    }

    /// Java `store(Properties, String)`.
    pub fn store_with_prepend(&self, props: &mut BTreeMap<String, String>, prepend: &str) {
        // super.store(props, prepend)
        self.base
            .store_with_created_prepend(props, self.create_prepend(prepend).as_deref());
        self.base
            .revision_number
            .lock()
            .unwrap()
            .set_etomo_version(Some(latest_version()));
        let prepend = self
            .create_prepend(prepend)
            .unwrap_or_else(|| "null".to_owned());
        let p = Some(prepend.as_str());
        self.root_name
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), p);
        self.init_motl_file.lock().unwrap().store(props, &prepend);
        self.tilt_range_min.lock().unwrap().store(props, &prepend);
        self.tilt_range_max.lock().unwrap().store(props, &prepend);
        self.reference_volume
            .lock()
            .unwrap()
            .base
            .store_with_prepend(props, p);
        self.reference_particle
            .lock()
            .unwrap()
            .base
            .store_with_prepend(props, p);
        self.reference_file
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), p);
        self.edge_shift
            .lock()
            .unwrap()
            .base
            .store_with_prepend(props, p);
        self.flg_wedge_weight
            .lock()
            .unwrap()
            .store_with_prepend(props, p);
        self.mask_use_reference_particle
            .lock()
            .unwrap()
            .remove_with_prepend(props, p);
        self.mask_model_pts_model_number
            .lock()
            .unwrap()
            .remove_with_prepend(props, p);
        self.mask_model_pts_z_rotation
            .lock()
            .unwrap()
            .base
            .store_with_prepend(props, p);
        self.mask_model_pts_particle
            .lock()
            .unwrap()
            .remove_with_prepend(props, p);
        self.mask_model_pts_y_rotation
            .lock()
            .unwrap()
            .base
            .store_with_prepend(props, p);
        self.mask_type_volume
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), p);
        self.n_weight_group
            .lock()
            .unwrap()
            .base
            .store_with_prepend(props, p);
        self.tilt_range.lock().unwrap().store_with_prepend(props, p);
        self.base
            .revision_number
            .lock()
            .unwrap()
            .store_with_prepend(props, &prepend);
        self.manual_cylinder_orientation
            .lock()
            .unwrap()
            .store_with_prepend(props, p);
        self.reference_multiparticle_level
            .lock()
            .unwrap()
            .base
            .store_with_prepend(props, p);
        self.tilt_range_multi_axes
            .lock()
            .unwrap()
            .store_with_prepend(props, p);
        self.tilt_range_multi_axes_file
            .lock()
            .unwrap()
            .store(props, &prepend);
        self.cylinder_height
            .lock()
            .unwrap()
            .base
            .store_with_prepend(props, p);
        self.mask_blur_std_dev
            .lock()
            .unwrap()
            .base
            .store_with_prepend(props, p);
        self.browsing_directory
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), p);
        self.low_cutoff_cutoff
            .lock()
            .unwrap()
            .store(props, &prepend);
        self.low_cutoff_sigma.lock().unwrap().store(props, &prepend);
        self.is_low_cutoff
            .lock()
            .unwrap()
            .store_with_prepend(props, p);
        self.flg_align_averages
            .lock()
            .unwrap()
            .store_with_prepend(props, p);
        self.c_n_symmetric_averaging
            .lock()
            .unwrap()
            .base
            .store_with_prepend(props, p);
    }

    /// Java `setRootName(String)`.
    pub fn set_root_name(&self, input: Option<&str>) {
        self.root_name.lock().unwrap().set(input);
    }

    /// Java `getRootName()`.
    pub fn get_root_name(&self) -> Option<String> {
        self.root_name.lock().unwrap().to_string_option()
    }

    /// Java `setMaskModelPtsZRotation(String)`.
    pub fn set_mask_model_pts_z_rotation(&self, input: Option<&str>) {
        self.mask_model_pts_z_rotation
            .lock()
            .unwrap()
            .set_string(input);
    }

    /// Java `setMaskModelPtsYRotation(String)`.
    pub fn set_mask_model_pts_y_rotation(&self, input: Option<&str>) {
        self.mask_model_pts_y_rotation
            .lock()
            .unwrap()
            .set_string(input);
    }

    /// Java `setMaskTypeVolume(String)`.
    pub fn set_mask_type_volume(&self, input: Option<&str>) {
        self.mask_type_volume.lock().unwrap().set(input);
    }

    /// Java `setManualCylinderOrientation(boolean)`.
    pub fn set_manual_cylinder_orientation(&self, input: bool) {
        self.manual_cylinder_orientation
            .lock()
            .unwrap()
            .set_boolean(input);
    }

    /// Java `getBrowsingDirectory()`.
    pub fn get_browsing_directory(&self) -> Option<String> {
        self.browsing_directory.lock().unwrap().to_string_option()
    }

    /// Java `setEdgeShift(Number)`.
    pub fn set_edge_shift(&self, edge_shift: Option<Number>) {
        self.edge_shift.lock().unwrap().set_number(edge_shift);
    }

    /// Java `setInitMotlFile(String, int)`.
    pub fn set_init_motl_file(&self, init_motl_file: Option<&str>, key: i32) {
        self.init_motl_file
            .lock()
            .unwrap()
            .put_string(key, init_motl_file);
    }

    /// Java `setTiltRangeMultiAxesFile(String, int)`.
    pub fn set_tilt_range_multi_axes_file(&self, input: Option<&str>, key: i32) {
        self.tilt_range_multi_axes_file
            .lock()
            .unwrap()
            .put_string(key, input);
    }

    /// Java `resetInitMotlFile()`.
    pub fn reset_init_motl_file(&self) {
        self.init_motl_file.lock().unwrap().reset();
    }

    /// Java `setTiltRangeMin(String, int)`.
    pub fn set_tilt_range_min(&self, input: Option<&str>, key: i32) {
        self.tilt_range_min.lock().unwrap().put_string(key, input);
    }

    /// Java `resetTiltRangeMin()`.
    pub fn reset_tilt_range_min(&self) {
        self.tilt_range_min.lock().unwrap().reset();
    }

    /// Java `setTiltRangeMax(String, int)`.
    pub fn set_tilt_range_max(&self, input: Option<&str>, key: i32) {
        self.tilt_range_max.lock().unwrap().put_string(key, input);
    }

    /// Java `resetTiltRangeMax()`.
    pub fn reset_tilt_range_max(&self) {
        self.tilt_range_max.lock().unwrap().reset();
    }

    /// Java `setReferenceMultiparticleLevel(String)`.
    pub fn set_reference_multiparticle_level(&self, input: Option<&str>) {
        self.reference_multiparticle_level
            .lock()
            .unwrap()
            .set_string(input);
    }

    /// Java `setTiltRange(boolean)`.
    pub fn set_tilt_range(&self, input: bool) {
        self.tilt_range.lock().unwrap().set_boolean(input);
    }

    /// Java `setReferenceFile(String)`.
    pub fn set_reference_file(&self, reference_file: Option<&str>) {
        self.reference_file.lock().unwrap().set(reference_file);
    }

    /// Java `setReferenceParticle(String)`.
    pub fn set_reference_particle(&self, reference_particle: Option<&str>) {
        self.reference_particle
            .lock()
            .unwrap()
            .set_string(reference_particle);
    }

    /// Java `setFlgWedgeWeight(boolean)`.
    pub fn set_flg_wedge_weight(&self, input: bool) {
        self.flg_wedge_weight.lock().unwrap().set_boolean(input);
    }

    /// Java `setTiltRangeMultiAxes(boolean)`.
    pub fn set_tilt_range_multi_axes(&self, input: bool) {
        self.tilt_range_multi_axes
            .lock()
            .unwrap()
            .set_boolean(input);
    }

    /// Java `setReferenceVolume(Number)`.
    pub fn set_reference_volume_number(&self, reference_volume: Option<Number>) {
        self.reference_volume
            .lock()
            .unwrap()
            .set_number(reference_volume);
    }

    /// Java `setReferenceVolume(String)`.
    pub fn set_reference_volume_string(&self, reference_volume: Option<&str>) {
        self.reference_volume
            .lock()
            .unwrap()
            .set_string(reference_volume);
    }

    /// Java `setNWeightGroup(Number)`.
    pub fn set_n_weight_group(&self, input: Option<Number>) {
        self.n_weight_group.lock().unwrap().set_number(input);
    }

    /// Java `setCylinderHeight(String)`.
    pub fn set_cylinder_height(&self, input: Option<&str>) {
        self.cylinder_height.lock().unwrap().set_string(input);
    }

    /// Java `setMaskBlurStdDev(String)`.
    pub fn set_mask_blur_std_dev(&self, input: Option<&str>) {
        self.mask_blur_std_dev.lock().unwrap().set_string(input);
    }

    /// Java `setBrowsingDirectory(File)`.
    pub fn set_browsing_directory(&self, input: Option<&Path>) {
        match input {
            None => self.browsing_directory.lock().unwrap().reset(),
            Some(input) => self.browsing_directory.lock().unwrap().set(Some(
                &utilities::java_io_file_get_absolute_path(&input.to_string_lossy()),
            )),
        }
    }

    /// Java `setLowCutoffCutoff(String, int)`.
    pub fn set_low_cutoff_cutoff(&self, input: Option<&str>, key: i32) {
        self.low_cutoff_cutoff.lock().unwrap().put(key, input);
    }

    /// Java `resetLowCutoffCutoff()`.
    pub fn reset_low_cutoff_cutoff(&self) {
        self.low_cutoff_cutoff.lock().unwrap().reset();
    }

    /// Java `setLowCutoffSigma(String, int)`.
    pub fn set_low_cutoff_sigma(&self, input: Option<&str>, key: i32) {
        self.low_cutoff_sigma.lock().unwrap().put(key, input);
    }

    /// Java `resetLowCutoffSigma()`.
    pub fn reset_low_cutoff_sigma(&self) {
        self.low_cutoff_sigma.lock().unwrap().reset();
    }

    /// Java `setIsLowCutoff(boolean)`.
    pub fn set_is_low_cutoff(&self, input: bool) {
        self.is_low_cutoff.lock().unwrap().set_boolean(input);
    }

    /// Java `setFlgAlignAverages(boolean)`.
    pub fn set_flg_align_averages(&self, input: bool) {
        self.flg_align_averages.lock().unwrap().set_boolean(input);
    }

    /// Java `setCNSymmetricAveraging(Number)`.
    pub fn set_cn_symmetric_averaging(&self, input: Option<Number>) {
        self.c_n_symmetric_averaging
            .lock()
            .unwrap()
            .set_number(input);
    }
}

/// `number.set(input.number)` on two `EtomoNumber` fields.
fn copy_number(to: &Mutex<EtomoNumber>, from: &Mutex<EtomoNumber>) {
    let value: ConstEtomoNumber = (**from.lock().unwrap()).clone();
    to.lock().unwrap().set_const_etomo_number(Some(&value));
}

/// `boolean.set(input.boolean)` on two `EtomoBoolean2` fields.
fn copy_boolean(to: &Mutex<EtomoBoolean2>, from: &Mutex<EtomoBoolean2>) {
    let value: ConstEtomoNumber = (****from.lock().unwrap()).clone();
    to.lock().unwrap().set_const_etomo_number(Some(&value));
}

impl ConstPeetMetaData for PeetMetaData {
    fn get_name(&self) -> Option<String> {
        BaseMetaData::get_name(self)
    }

    fn get_init_motl_file(&self, key: i32) -> Option<String> {
        self.init_motl_file.lock().unwrap().get_string(key)
    }

    fn get_tilt_range_multi_axes_file(&self, key: i32) -> Option<String> {
        self.tilt_range_multi_axes_file
            .lock()
            .unwrap()
            .get_string(key)
    }

    fn get_tilt_range_min(&self, key: i32) -> Option<String> {
        self.tilt_range_min.lock().unwrap().get_string(key)
    }

    fn get_tilt_range_max(&self, key: i32) -> Option<String> {
        self.tilt_range_max.lock().unwrap().get_string(key)
    }

    fn get_axis_type(&self) -> AxisType {
        self.base.get_axis_type()
    }

    fn get_reference_file(&self) -> Option<String> {
        self.reference_file.lock().unwrap().to_string_option()
    }

    fn get_reference_particle(&self) -> ConstEtomoNumber {
        (**self.reference_particle.lock().unwrap()).clone()
    }

    fn get_reference_volume(&self) -> ConstEtomoNumber {
        (**self.reference_volume.lock().unwrap()).clone()
    }

    fn get_reference_multiparticle_level(&self) -> i32 {
        self.reference_multiparticle_level
            .lock()
            .unwrap()
            .get_defaulted_int()
    }

    fn get_edge_shift(&self) -> ConstEtomoNumber {
        (**self.edge_shift.lock().unwrap()).clone()
    }

    fn is_flg_wedge_weight(&self) -> bool {
        self.flg_wedge_weight.lock().unwrap().is()
    }

    fn get_mask_model_pts_z_rotation(&self) -> ConstEtomoNumber {
        (**self.mask_model_pts_z_rotation.lock().unwrap()).clone()
    }

    fn get_mask_model_pts_y_rotation(&self) -> Option<String> {
        Some(self.mask_model_pts_y_rotation.lock().unwrap().to_string())
    }

    fn get_mask_type_volume(&self) -> Option<String> {
        self.mask_type_volume.lock().unwrap().to_string_option()
    }

    fn get_n_weight_group(&self) -> ConstEtomoNumber {
        (**self.n_weight_group.lock().unwrap()).clone()
    }

    fn is_tilt_range(&self) -> bool {
        self.tilt_range.lock().unwrap().is()
    }

    fn is_manual_cylinder_orientation(&self) -> bool {
        self.manual_cylinder_orientation.lock().unwrap().is()
    }

    fn is_tilt_range_multi_axes(&self) -> bool {
        self.tilt_range_multi_axes.lock().unwrap().is()
    }

    fn get_cylinder_height(&self) -> Option<String> {
        Some(self.cylinder_height.lock().unwrap().to_string())
    }

    fn get_mask_blur_std_dev(&self) -> Option<String> {
        Some(self.mask_blur_std_dev.lock().unwrap().to_string())
    }

    fn get_low_cutoff_cutoff(&self, key: i32) -> Option<String> {
        self.low_cutoff_cutoff.lock().unwrap().get_string(key)
    }

    fn get_low_cutoff_sigma(&self, key: i32) -> Option<String> {
        self.low_cutoff_sigma.lock().unwrap().get_string(key)
    }

    fn is_low_cutoff(&self) -> bool {
        self.is_low_cutoff.lock().unwrap().is()
    }

    fn is_flg_align_averages(&self) -> bool {
        self.flg_align_averages.lock().unwrap().is()
    }

    fn get_cn_symmetric_averaging(&self) -> ConstEtomoNumber {
        (**self.c_n_symmetric_averaging.lock().unwrap()).clone()
    }

    fn is_cn_symmetric_averaging(&self) -> bool {
        self.c_n_symmetric_averaging.lock().unwrap().is()
    }
}

/// Java `Storable`, through `BaseMetaData`.  `store(Properties)` is `BaseMetaData`'s,
/// which stores with an empty prepend.
impl Storable for PeetMetaData {
    fn store(&self, properties: &mut BTreeMap<String, String>) {
        PeetMetaData::store_with_prepend(self, properties, "");
    }

    fn store_with_prepend(&self, properties: &mut BTreeMap<String, String>, prepend: &str) {
        PeetMetaData::store_with_prepend(self, properties, prepend);
    }

    fn load(&self, properties: &mut BTreeMap<String, String>) {
        PeetMetaData::load(self, properties);
    }

    fn load_with_prepend(&self, properties: &mut BTreeMap<String, String>, prepend: &str) {
        PeetMetaData::load_with_prepend(self, properties, prepend);
    }
}

impl BaseMetaData for PeetMetaData {
    fn base(&self) -> &BaseMetaDataBase {
        &self.base
    }

    /// Java `getMetaDataFileName()`.
    fn get_meta_data_file_name(&self) -> Option<String> {
        let root_name = self.root_name.lock().unwrap();
        if root_name.is_empty() {
            return None;
        }
        Some(dataset_files::get_peet_data_file_name(
            &root_name.to_string_option().unwrap_or_default(),
        ))
    }

    /// Java `getName()`.
    fn get_name(&self) -> Option<String> {
        let root_name = self.root_name.lock().unwrap().to_string_option();
        match root_name {
            None => Some(NEW_TITLE.to_owned()),
            Some(root_name)
                if super::const_etomo_number::java_lang_string_matches_whitespace(&root_name) =>
            {
                Some(NEW_TITLE.to_owned())
            }
            Some(root_name) => Some(root_name),
        }
    }

    /// Java `getDatasetName()`.
    fn get_dataset_name(&self) -> Option<String> {
        self.root_name.lock().unwrap().to_string_option()
    }

    /// Java `isValid()`.
    fn is_valid(&self) -> bool {
        self.validate().is_none()
    }

    /// Java package-private `getGroupKey()`.
    fn get_group_key(&self) -> Option<String> {
        Some(GROUP_KEY.to_owned())
    }
}
