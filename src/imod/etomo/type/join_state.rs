//! `IMOD/Etomo/src/etomo/type/JoinState.java`.
//!
//! What the join has actually done (as opposed to what the join dialog is set to): the
//! chunk limits, shifts, sizes and binning each join and trial join was run with, the
//! refine lists, and the rotation angles of each section row.  Stored in the `.ejf`
//! under the `JoinState` group.
//!
//! **Representation.**  `JoinState extends BaseState implements ConstJoinState`; the
//! abstract `createPrepend` is the `BaseState` trait (`base_state.rs`), whose
//! `store`/`load` bodies only compute and discard a prepend, and the interface is the
//! `ConstJoinState` trait (`const_join_state.rs`).  The object is owned by `JoinManager`
//! and read by the join comscript parameters from process threads, so - like
//! `TomogramState` - each mutable field carries its own lock and every method takes
//! `&self`; a getter that returns a field object returns a copy taken under its lock.
//! The `IntKeyList` fields are `Mutex<IntKeyList>`, which implements `ConstIntKeyList`
//! below by locking for each call, so a `Walker` over one reads the live list one
//! synchronised call at a time, as Java's does.
//!
//! **`prepend == ""`.**  `createPrepend` tests `prepend == ""`, a reference comparison
//! true for the interned literal that `store(Properties)`/`load(Properties)` pass; it
//! is translated as `prepend.is_empty()`.
#![allow(dead_code)]

use std::collections::{BTreeMap, HashMap};
use std::sync::{LazyLock, Mutex};

use super::base_state::BaseState;
use super::const_etomo_number::ConstEtomoNumber;
use super::const_etomo_version::ConstEtomoVersion;
use super::const_int_key_list::ConstIntKeyList;
use super::const_join_meta_data::ConstJoinMetaData;
use super::const_join_state::ConstJoinState;
use super::const_section_table_row_data::{self, ConstSectionTableRowData};
use super::etomo_boolean2::EtomoBoolean2;
use super::etomo_number::EtomoNumber;
use super::etomo_version::EtomoVersion;
use super::int_key_list::{IntKeyList, Walker};
use super::join_meta_data::JoinMetaData;
use super::null_required_number_exception::NullRequiredNumberException;
use super::process_name::ProcessName;
use super::script_parameter::ScriptParameter;
use super::slicer_angles::SlicerAngles;
use crate::imod::etomo::base_manager::BaseManager;

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `ROTATION_ANGLE_X`.
pub const ROTATION_ANGLE_X: &str = "RotationAngleX";
/// Java `ROTATION_ANGLE_Y`.
pub const ROTATION_ANGLE_Y: &str = "RotationAngleY";
/// Java `ROTATION_ANGLE_Z`.
pub const ROTATION_ANGLE_Z: &str = "RotationAngleZ";
/// Java `GAPS_EXIST_KEY`.
pub const GAPS_EXIST_KEY: &str = "GapsExist";
/// Java `JOIN_KEY`.
const JOIN_KEY: &str = "Join";
/// Java `TRIAL_KEY`.
const TRIAL_KEY: &str = "Trial";
/// Java `REFINE_KEY`.
const REFINE_KEY: &str = "Refine";
/// Java `START_LIST_KEY`.
const START_LIST_KEY: &str = "StartList";
/// Java `END_LIST_KEY`.
const END_LIST_KEY: &str = "EndList";
/// Java `ALIGNMENT_REF_SECTION_KEY`.
const ALIGNMENT_REF_SECTION_KEY: &str = "AlignmentRefSection";
/// Java `SHIFT_IN_X_KEY`.
const SHIFT_IN_X_KEY: &str = "ShiftInX";
/// Java `SHIFT_IN_Y_KEY`.
const SHIFT_IN_Y_KEY: &str = "ShiftInY";
/// Java `SIZE_IN_X_KEY`.
const SIZE_IN_X_KEY: &str = "SizeInX";
/// Java `SIZE_IN_Y_KEY`.
const SIZE_IN_Y_KEY: &str = "SizeInY";
/// Java `LOCAL_FITS_KEY`.
const LOCAL_FITS_KEY: &str = "LocalFits";
/// Java `USE_EVERY_N_SLICES_KEY`.
const USE_EVERY_N_SLICES_KEY: &str = "UseEveryNSlices";

/// Java `MIN_REFINE_VERSION`, `EtomoVersion.getDefaultInstance("1.1")`.
pub static MIN_REFINE_VERSION: LazyLock<EtomoVersion> =
    LazyLock::new(|| EtomoVersion::get_default_instance_with_version(Some("1.1")));

/// Java `CURRENT_VERSION`.
const CURRENT_VERSION: &str = "1.0";

/// Java package-private `sampleProducedString`.
pub(crate) const SAMPLE_PRODUCED_STRING: &str = "SampleProduced";
/// Java package-private `defaultSampleProduced`.
pub(crate) const DEFAULT_SAMPLE_PRODUCED: bool = false;

/// Java `groupString`.
const GROUP_STRING: &str = "JoinState";
/// Java `VERSION`.
const VERSION: &str = "1.1";

/// Java `XFMODEL_INPUT_FILE`, `ProcessName.XFMODEL.toString() + "InputFile"`.
static XFMODEL_INPUT_FILE: LazyLock<String> =
    LazyLock::new(|| format!("{}InputFile", ProcessName::XFMODEL));
/// Java `XFMODEL_OUTPUT_FILE`, `ProcessName.XFMODEL.toString() + "OutputFile"`.
static XFMODEL_OUTPUT_FILE: LazyLock<String> =
    LazyLock::new(|| format!("{}OutputFile", ProcessName::XFMODEL));

/// Java `Hashtable` of `SlicerAngles` keyed by row index.
pub type RotationAnglesList = HashMap<i32, SlicerAngles>;

/// Java `JoinState`.
pub struct JoinState {
    /// Java `doneMode`; set on the successful completion of finishjoin.
    done_mode: Mutex<EtomoNumber>,
    /// Java `manager`.
    manager: &'static dyn BaseManager,
    /// Java `rotationAnglesList`, null by default.
    rotation_angles_list: Mutex<Option<RotationAnglesList>>,
    /// Java `revertRotationAnglesList`, null by default.
    revert_rotation_angles_list: Mutex<Option<RotationAnglesList>>,
    /// Java `totalRows`.
    total_rows: Mutex<EtomoNumber>,
    /// Java `revertTotalRows`.
    revert_total_rows: Mutex<EtomoNumber>,
    /// Java package-private `sampleProduced`; state variable for join setup tab.
    sample_produced: Mutex<bool>,
    /// Java `gapsExist`, null by default.
    gaps_exist: Mutex<Option<EtomoBoolean2>>,
    /// Java `refineTrial`.
    refine_trial: Mutex<EtomoBoolean2>,
    /// Java `joinVersion`.
    join_version: Mutex<EtomoVersion>,
    /// Java `joinStartList`.
    join_start_list: Mutex<IntKeyList>,
    /// Java `joinEndList`.
    join_end_list: Mutex<IntKeyList>,
    /// Java `joinAlignmentRefSection`.
    join_alignment_ref_section: Mutex<EtomoNumber>,
    /// Java `joinShiftInX`.
    join_shift_in_x: Mutex<ScriptParameter>,
    /// Java `joinShiftInY`.
    join_shift_in_y: Mutex<ScriptParameter>,
    /// Java `joinSizeInX`.
    join_size_in_x: Mutex<ScriptParameter>,
    /// Java `joinSizeInY`.
    join_size_in_y: Mutex<ScriptParameter>,
    /// Java `joinLocalFits`.
    join_local_fits: Mutex<EtomoBoolean2>,
    /// Java `joinTrialVersion`.
    join_trial_version: Mutex<EtomoVersion>,
    /// Java `joinTrialStartList`.
    join_trial_start_list: Mutex<IntKeyList>,
    /// Java `joinTrialEndList`.
    join_trial_end_list: Mutex<IntKeyList>,
    /// Java `joinTrialAlignmentRefSection`.
    join_trial_alignment_ref_section: Mutex<EtomoNumber>,
    /// Java `joinTrialShiftInX`.
    join_trial_shift_in_x: Mutex<ScriptParameter>,
    /// Java `joinTrialShiftInY`.
    join_trial_shift_in_y: Mutex<ScriptParameter>,
    /// Java `joinTrialSizeInX`.
    join_trial_size_in_x: Mutex<ScriptParameter>,
    /// Java `joinTrialSizeInY`.
    join_trial_size_in_y: Mutex<ScriptParameter>,
    /// Java `joinTrialLocalFits`.
    join_trial_local_fits: Mutex<EtomoBoolean2>,
    /// Java `joinTrialBinning`.
    join_trial_binning: Mutex<EtomoNumber>,
    /// Java `refineStartList`.
    refine_start_list: Mutex<IntKeyList>,
    /// Java `refineEndList`.
    refine_end_list: Mutex<IntKeyList>,
    /// Java `xfModelOutputFile`, null by default.
    xf_model_output_file: Mutex<Option<String>>,
    /// Java `debug`.
    debug: Mutex<bool>,
    /// Java `joinTrialUseEveryNSlices`.
    join_trial_use_every_n_slices: Mutex<EtomoNumber>,
    /// Java `refineTrialUseEveryNSlices`.
    refine_trial_use_every_n_slices: Mutex<EtomoNumber>,
    /// Java `version`, `EtomoVersion.getDefaultInstance()`.
    version: Mutex<EtomoVersion>,
}

impl JoinState {
    /// Java `JoinState(BaseManager)`.
    pub fn new(manager: &'static dyn BaseManager) -> JoinState {
        JoinState {
            done_mode: Mutex::new(EtomoNumber::new_with_name("DoneMode")),
            manager,
            rotation_angles_list: Mutex::new(None),
            revert_rotation_angles_list: Mutex::new(None),
            total_rows: Mutex::new(EtomoNumber::new_with_name("TotalRows")),
            revert_total_rows: Mutex::new(EtomoNumber::new()),
            sample_produced: Mutex::new(false),
            gaps_exist: Mutex::new(None),
            refine_trial: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.{}",
                REFINE_KEY, TRIAL_KEY
            ))),
            join_version: Mutex::new(EtomoVersion::get_empty_instance(Some(&format!(
                "{}.{}",
                JOIN_KEY,
                EtomoVersion::DEFAULT_KEY
            )))),
            join_start_list: Mutex::new(IntKeyList::get_number_instance_with_key(&format!(
                "{}.{}",
                JOIN_KEY, START_LIST_KEY
            ))),
            join_end_list: Mutex::new(IntKeyList::get_number_instance_with_key(&format!(
                "{}.{}",
                JOIN_KEY, END_LIST_KEY
            ))),
            join_alignment_ref_section: Mutex::new(EtomoNumber::new_with_name(&format!(
                "{}.{}",
                JOIN_KEY, ALIGNMENT_REF_SECTION_KEY
            ))),
            join_shift_in_x: Mutex::new(ScriptParameter::new_with_name(&format!(
                "{}.{}",
                JOIN_KEY, SHIFT_IN_X_KEY
            ))),
            join_shift_in_y: Mutex::new(ScriptParameter::new_with_name(&format!(
                "{}.{}",
                JOIN_KEY, SHIFT_IN_Y_KEY
            ))),
            join_size_in_x: Mutex::new(ScriptParameter::new_with_name(&format!(
                "{}.{}",
                JOIN_KEY, SIZE_IN_X_KEY
            ))),
            join_size_in_y: Mutex::new(ScriptParameter::new_with_name(&format!(
                "{}.{}",
                JOIN_KEY, SIZE_IN_Y_KEY
            ))),
            join_local_fits: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.{}",
                JOIN_KEY, LOCAL_FITS_KEY
            ))),
            join_trial_version: Mutex::new(EtomoVersion::get_empty_instance(Some(&format!(
                "{}.{}.{}",
                JOIN_KEY,
                TRIAL_KEY,
                EtomoVersion::DEFAULT_KEY
            )))),
            join_trial_start_list: Mutex::new(IntKeyList::get_number_instance_with_key(&format!(
                "{}.{}.{}",
                JOIN_KEY, TRIAL_KEY, START_LIST_KEY
            ))),
            join_trial_end_list: Mutex::new(IntKeyList::get_number_instance_with_key(&format!(
                "{}.{}.{}",
                JOIN_KEY, TRIAL_KEY, END_LIST_KEY
            ))),
            join_trial_alignment_ref_section: Mutex::new(EtomoNumber::new_with_name(&format!(
                "{}.{}.{}",
                JOIN_KEY, TRIAL_KEY, ALIGNMENT_REF_SECTION_KEY
            ))),
            join_trial_shift_in_x: Mutex::new(ScriptParameter::new_with_name(&format!(
                "{}.{}.{}",
                JOIN_KEY, TRIAL_KEY, SHIFT_IN_X_KEY
            ))),
            join_trial_shift_in_y: Mutex::new(ScriptParameter::new_with_name(&format!(
                "{}.{}.{}",
                JOIN_KEY, TRIAL_KEY, SHIFT_IN_Y_KEY
            ))),
            join_trial_size_in_x: Mutex::new(ScriptParameter::new_with_name(&format!(
                "{}.{}.{}",
                JOIN_KEY, TRIAL_KEY, SIZE_IN_X_KEY
            ))),
            join_trial_size_in_y: Mutex::new(ScriptParameter::new_with_name(&format!(
                "{}.{}.{}",
                JOIN_KEY, TRIAL_KEY, SIZE_IN_Y_KEY
            ))),
            join_trial_local_fits: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.{}.{}",
                JOIN_KEY, TRIAL_KEY, LOCAL_FITS_KEY
            ))),
            join_trial_binning: Mutex::new(EtomoNumber::new_with_name(&format!(
                "{}.{}.{}",
                JOIN_KEY, TRIAL_KEY, "Binning"
            ))),
            refine_start_list: Mutex::new(IntKeyList::get_number_instance_with_key(&format!(
                "{}.{}",
                REFINE_KEY, START_LIST_KEY
            ))),
            refine_end_list: Mutex::new(IntKeyList::get_number_instance_with_key(&format!(
                "{}.{}",
                REFINE_KEY, END_LIST_KEY
            ))),
            xf_model_output_file: Mutex::new(None),
            debug: Mutex::new(false),
            join_trial_use_every_n_slices: Mutex::new(EtomoNumber::new_with_name(&format!(
                "{}.{}.{}",
                JOIN_KEY, TRIAL_KEY, USE_EVERY_N_SLICES_KEY
            ))),
            refine_trial_use_every_n_slices: Mutex::new(EtomoNumber::new_with_name(&format!(
                "{}.{}.{}",
                REFINE_KEY, TRIAL_KEY, USE_EVERY_N_SLICES_KEY
            ))),
            version: Mutex::new(EtomoVersion::get_default_instance()),
        }
    }

    /// Java `store(Properties)`.
    pub fn store(&self, props: &mut BTreeMap<String, String>) {
        self.store_with_prepend(props, "");
    }

    /// Java `store(Properties, String)`.
    pub fn store_with_prepend(&self, props: &mut BTreeMap<String, String>, prepend: &str) {
        // `super.store(props, prepend)`: `BaseState.store` only runs
        // `createPrepend(prepend)` and discards the result.
        let prepend = BaseState::create_prepend(self, prepend);
        let group = format!("{}.", prepend);
        crate::imod::etomo::storage::storable::StorableValue::store_with_prepend(
            &*self.join_version.lock().unwrap(),
            props,
            &prepend,
        );
        crate::imod::etomo::storage::storable::StorableValue::store_with_prepend(
            &*self.join_trial_version.lock().unwrap(),
            props,
            &prepend,
        );
        self.done_mode
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        let sample_produced = *self.sample_produced.lock().unwrap();
        props.insert(
            format!("{}{}", group, SAMPLE_PRODUCED_STRING),
            sample_produced.to_string(),
        );
        self.total_rows
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        // store the rotation angles under their current row number
        {
            let rotation_angles_list = self.rotation_angles_list.lock().unwrap();
            if let Some(rotation_angles_list) = rotation_angles_list.as_ref() {
                let total_rows = self.total_rows.lock().unwrap().get_int();
                let mut i = 0;
                while i < total_rows {
                    let rotation_angles = rotation_angles_list.get(&i);
                    if let Some(rotation_angles) = rotation_angles {
                        let mut row_number = EtomoNumber::new();
                        row_number.set_int(i + 1);
                        rotation_angles.store(
                            props,
                            &const_section_table_row_data::create_prepend_static(
                                Some(&prepend),
                                &row_number,
                            ),
                        );
                    }
                    i += 1;
                }
            }
        }
        EtomoBoolean2::store_instance(
            self.gaps_exist.lock().unwrap().as_ref(),
            props,
            Some(&prepend),
            GAPS_EXIST_KEY,
        );
        self.refine_trial
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.join_alignment_ref_section
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.join_shift_in_x
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.join_shift_in_y
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.join_size_in_x
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.join_size_in_y
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.join_local_fits
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.join_start_list.lock().unwrap().store(props, &prepend);
        self.join_end_list.lock().unwrap().store(props, &prepend);

        self.join_trial_alignment_ref_section
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.join_trial_shift_in_x
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.join_trial_shift_in_y
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.join_trial_size_in_x
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.join_trial_size_in_y
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.join_trial_local_fits
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.join_trial_binning
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.join_trial_start_list
            .lock()
            .unwrap()
            .store(props, &prepend);
        self.join_trial_end_list
            .lock()
            .unwrap()
            .store(props, &prepend);
        self.join_trial_use_every_n_slices
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.refine_trial_use_every_n_slices
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.refine_start_list
            .lock()
            .unwrap()
            .store(props, &prepend);
        self.refine_end_list.lock().unwrap().store(props, &prepend);
        let xf_model_output_file = self.xf_model_output_file.lock().unwrap().clone();
        match xf_model_output_file {
            None => {
                props.remove(&format!("{}.{}", prepend, *XFMODEL_OUTPUT_FILE));
            }
            Some(xf_model_output_file) => {
                props.insert(
                    format!("{}.{}", prepend, *XFMODEL_OUTPUT_FILE),
                    xf_model_output_file,
                );
            }
        }
        let mut version = self.version.lock().unwrap();
        version.set(Some(CURRENT_VERSION));
        crate::imod::etomo::storage::storable::StorableValue::store_with_prepend(
            &*version, props, &prepend,
        );
    }

    /// Java `equals(JoinState)`.
    pub fn equals(&self, that: &JoinState) -> bool {
        if !BaseState::equals(self, that) {
            return false;
        }
        true
    }

    /// Java `load(Properties)`.
    pub fn load(&self, props: &mut BTreeMap<String, String>) {
        self.load_with_prepend(props, "");
    }

    /// Java `load(Properties, String)`.
    pub fn load_with_prepend(&self, props: &mut BTreeMap<String, String>, prepend: &str) {
        // `super.load(props, prepend)`: `BaseState.load` only runs
        // `createPrepend(prepend)` and discards the result.
        let props_copy = props;
        // reset
        self.done_mode.lock().unwrap().reset();
        *self.sample_produced.lock().unwrap() = DEFAULT_SAMPLE_PRODUCED;
        *self.gaps_exist.lock().unwrap() = None;
        self.refine_trial.lock().unwrap().reset();
        self.join_version.lock().unwrap().reset();
        self.join_start_list.lock().unwrap().reset();
        self.join_end_list.lock().unwrap().reset();
        self.join_alignment_ref_section.lock().unwrap().reset();
        self.join_shift_in_x.lock().unwrap().reset();
        self.join_shift_in_y.lock().unwrap().reset();
        self.join_size_in_x.lock().unwrap().reset();
        self.join_size_in_y.lock().unwrap().reset();
        self.join_local_fits.lock().unwrap().reset();
        self.join_trial_version.lock().unwrap().reset();
        self.join_trial_start_list.lock().unwrap().reset();
        self.join_trial_end_list.lock().unwrap().reset();
        self.join_trial_alignment_ref_section
            .lock()
            .unwrap()
            .reset();
        self.join_trial_shift_in_x.lock().unwrap().reset();
        self.join_trial_shift_in_y.lock().unwrap().reset();
        self.join_trial_size_in_x.lock().unwrap().reset();
        self.join_trial_size_in_y.lock().unwrap().reset();
        self.join_trial_local_fits.lock().unwrap().reset();
        self.join_trial_binning.lock().unwrap().reset();
        self.refine_start_list.lock().unwrap().reset();
        self.refine_end_list.lock().unwrap().reset();
        self.join_trial_use_every_n_slices.lock().unwrap().reset();
        self.refine_trial_use_every_n_slices.lock().unwrap().reset();
        self.version.lock().unwrap().reset();
        // load
        let prepend = BaseState::create_prepend(self, prepend);
        let group = format!("{}.", prepend);
        crate::imod::etomo::storage::storable::StorableValue::load_with_prepend(
            &mut *self.join_version.lock().unwrap(),
            &mut *props_copy,
            &prepend,
        );
        crate::imod::etomo::storage::storable::StorableValue::load_with_prepend(
            &mut *self.join_trial_version.lock().unwrap(),
            &mut *props_copy,
            &prepend,
        );
        let join_version_is_null = self.join_version.lock().unwrap().is_null();
        if !join_version_is_null && self.is_join_version_ge(false, Some(&*MIN_REFINE_VERSION)) {
            self.join_alignment_ref_section
                .lock()
                .unwrap()
                .load_with_prepend(&props_copy, Some(&prepend));
            self.join_shift_in_x
                .lock()
                .unwrap()
                .load_with_prepend(&props_copy, Some(&prepend));
            self.join_shift_in_y
                .lock()
                .unwrap()
                .load_with_prepend(&props_copy, Some(&prepend));
            self.join_size_in_x
                .lock()
                .unwrap()
                .load_with_prepend(&props_copy, Some(&prepend));
            self.join_size_in_y
                .lock()
                .unwrap()
                .load_with_prepend(&props_copy, Some(&prepend));
            self.join_start_list
                .lock()
                .unwrap()
                .load(&props_copy, &prepend);
            self.join_end_list
                .lock()
                .unwrap()
                .load(&props_copy, &prepend);
        }
        let join_trial_version_is_null = self.join_trial_version.lock().unwrap().is_null();
        if !join_trial_version_is_null && self.is_join_version_ge(true, Some(&*MIN_REFINE_VERSION))
        {
            self.join_trial_alignment_ref_section
                .lock()
                .unwrap()
                .load_with_prepend(&props_copy, Some(&prepend));
            self.join_trial_shift_in_x
                .lock()
                .unwrap()
                .load_with_prepend(&props_copy, Some(&prepend));
            self.join_trial_shift_in_y
                .lock()
                .unwrap()
                .load_with_prepend(&props_copy, Some(&prepend));
            self.join_trial_size_in_x
                .lock()
                .unwrap()
                .load_with_prepend(&props_copy, Some(&prepend));
            self.join_trial_size_in_y
                .lock()
                .unwrap()
                .load_with_prepend(&props_copy, Some(&prepend));
            self.join_trial_binning
                .lock()
                .unwrap()
                .load_with_prepend(&props_copy, Some(&prepend));
            self.join_trial_start_list
                .lock()
                .unwrap()
                .load(&props_copy, &prepend);
            self.join_trial_end_list
                .lock()
                .unwrap()
                .load(&props_copy, &prepend);
            self.join_trial_use_every_n_slices
                .lock()
                .unwrap()
                .load_with_prepend(&props_copy, Some(&prepend));
        } else {
            self.load_join_trial_version1_0(props_copy, &prepend);
        }
        self.join_local_fits
            .lock()
            .unwrap()
            .load_with_prepend(&props_copy, Some(&prepend));
        self.join_trial_local_fits
            .lock()
            .unwrap()
            .load_with_prepend(&props_copy, Some(&prepend));
        self.done_mode
            .lock()
            .unwrap()
            .load_with_prepend(&props_copy, Some(&prepend));
        // Java `Boolean.valueOf(String)` is `"true".equalsIgnoreCase(s)`.
        let sample_produced = props_copy
            .get(&format!("{}{}", group, SAMPLE_PRODUCED_STRING))
            .cloned()
            .unwrap_or_else(|| DEFAULT_SAMPLE_PRODUCED.to_string());
        *self.sample_produced.lock().unwrap() = sample_produced.eq_ignore_ascii_case("true");
        self.total_rows
            .lock()
            .unwrap()
            .load_with_prepend(&props_copy, Some(&prepend));
        // retrieve the rotation angles by row number
        *self.rotation_angles_list.lock().unwrap() = None;
        let (total_rows_is_null, total_rows) = {
            let total_rows = self.total_rows.lock().unwrap();
            (total_rows.is_null(), total_rows.get_int())
        };
        if !total_rows_is_null {
            let mut i = 0;
            while i < total_rows {
                let mut rotation_angles = SlicerAngles::new();
                let mut row_number = EtomoNumber::new();
                row_number.set_int(i + 1);
                rotation_angles.load(
                    &*props_copy,
                    &const_section_table_row_data::create_prepend_static(
                        Some(&prepend),
                        &row_number,
                    ),
                );
                if !rotation_angles.is_empty() {
                    let mut rotation_angles_list = self.rotation_angles_list.lock().unwrap();
                    if rotation_angles_list.is_none() {
                        *rotation_angles_list = Some(HashMap::new());
                    }
                    rotation_angles_list
                        .as_mut()
                        .unwrap()
                        .insert(i, rotation_angles);
                }
                i += 1;
            }
        }
        {
            let mut gaps_exist = self.gaps_exist.lock().unwrap();
            let instance = gaps_exist.take();
            *gaps_exist =
                EtomoBoolean2::load_instance(instance, GAPS_EXIST_KEY, &props_copy, Some(&prepend));
        }
        self.refine_trial
            .lock()
            .unwrap()
            .load_with_prepend(&props_copy, Some(&prepend));
        self.refine_trial_use_every_n_slices
            .lock()
            .unwrap()
            .load_with_prepend(&props_copy, Some(&prepend));
        self.refine_start_list
            .lock()
            .unwrap()
            .load(&props_copy, &prepend);
        self.refine_end_list
            .lock()
            .unwrap()
            .load(&props_copy, &prepend);
        *self.xf_model_output_file.lock().unwrap() = props_copy
            .get(&format!("{}.{}", prepend, *XFMODEL_OUTPUT_FILE))
            .cloned();
        crate::imod::etomo::storage::storable::StorableValue::load_with_prepend(
            &mut *self.version.lock().unwrap(),
            &mut *props_copy,
            &prepend,
        );
        // Bug# 1165 - fixing previous .ejf files when possible.
        let version_lt = self.version.lock().unwrap().lt(Some(
            &EtomoVersion::get_default_instance_with_version(Some("1.0")),
        ));
        if version_lt {
            let join_trial_binning_is_null = self.join_trial_binning.lock().unwrap().is_null();
            if join_trial_binning_is_null {
                let mut join_trial_alignment_ref_section =
                    self.join_trial_alignment_ref_section.lock().unwrap();
                join_trial_alignment_ref_section.load_with_prepend(&props_copy, Some(&prepend));
                if !join_trial_alignment_ref_section.is_null() {
                    self.join_alignment_ref_section
                        .lock()
                        .unwrap()
                        .set_const_etomo_number(Some(&**join_trial_alignment_ref_section));
                    join_trial_alignment_ref_section.reset();
                }
            }
        }
    }

    /// Java `isJoinVersionGe(boolean, EtomoVersion)`.  Returns true if version is the
    /// minimum version required to do a refine.
    pub fn is_join_version_ge(&self, trial: bool, minimum_version: Option<&EtomoVersion>) -> bool {
        if trial {
            return self.join_trial_version.lock().unwrap().ge(minimum_version);
        }
        self.join_version.lock().unwrap().ge(minimum_version)
    }

    /// Java private `loadJoinTrialVersion1_0(Properties, String)`.  Loads JoinState
    /// version 0.0 data into trialFinishjoin.  There is no version 0.0 data available
    /// for finishjoin.  The `prepend` parameter is overwritten before it is read.
    fn load_join_trial_version1_0(&self, props: &mut BTreeMap<String, String>, prepend: &str) {
        let _ = prepend;
        let prepend = "JoinState.Trial";
        let mut key = format!("{}{}", prepend, "Binning");
        self.join_trial_binning
            .lock()
            .unwrap()
            .set_string(props.get(&key).map(String::as_str));
        props.remove(&key);
        key = format!("{}{}", prepend, "ShiftInX");
        self.join_trial_shift_in_x
            .lock()
            .unwrap()
            .set_string(props.get(&key).map(String::as_str));
        props.remove(&key);
        key = format!("{}{}", prepend, "ShiftInY");
        self.join_trial_shift_in_y
            .lock()
            .unwrap()
            .set_string(props.get(&key).map(String::as_str));
        props.remove(&key);
        key = format!("{}{}", prepend, "SizeInX");
        self.join_trial_size_in_x
            .lock()
            .unwrap()
            .set_string(props.get(&key).map(String::as_str));
        props.remove(&key);
        key = format!("{}{}", prepend, "SizeInY");
        self.join_trial_size_in_y
            .lock()
            .unwrap()
            .set_string(props.get(&key).map(String::as_str));
        props.remove(&key);
    }

    /// Java `setCurrentJoinVersion(boolean)`.
    pub fn set_current_join_version(&self, trial: bool) {
        if trial {
            self.join_trial_version.lock().unwrap().set(Some(VERSION));
        } else {
            self.join_version.lock().unwrap().set(Some(VERSION));
        }
    }

    /// Java `setJoinVersion1_0(boolean, JoinMetaData)`.  Sets refine parameters from
    /// meta data.  Assumes that some of the trial parameters values where already set by
    /// loadJoinTrialVersion1_0().  Sets the refine version or the refine trial version
    /// of to the current version.
    ///
    /// Fixed in translation: JoinState.java:431-432 dereference a null section table
    /// (`dataList.size()`), a NullPointerException; a missing table has size 0 here.
    pub fn set_join_version1_0(&self, trial: bool, meta_data: &JoinMetaData) {
        let data_list = meta_data.get_section_table_data();
        let size = data_list.as_ref().map_or(0, |data_list| data_list.len());
        self.set_current_join_version(trial);
        self.set_join_alignment_ref_section(trial, Some(&meta_data.get_alignment_ref_section()));
        if trial {
            let mut join_trial_start_list = self.join_trial_start_list.lock().unwrap();
            let mut join_trial_end_list = self.join_trial_end_list.lock().unwrap();
            join_trial_start_list.reset();
            join_trial_end_list.reset();
            for i in 0..size {
                let data = &data_list.as_ref().unwrap()[i];
                join_trial_start_list.put_etomo_number(
                    i as i32,
                    &EtomoNumber::new_from_instance(Some(data.get_join_final_start())),
                );
                join_trial_end_list.put_etomo_number(
                    i as i32,
                    &EtomoNumber::new_from_instance(Some(data.get_join_final_end())),
                );
            }
            self.join_trial_use_every_n_slices
                .lock()
                .unwrap()
                .set_const_etomo_number(Some(&meta_data.get_use_every_n_slices()));
        } else {
            let mut join_start_list = self.join_start_list.lock().unwrap();
            let mut join_end_list = self.join_end_list.lock().unwrap();
            join_start_list.reset();
            join_end_list.reset();
            for i in 0..size {
                let data = &data_list.as_ref().unwrap()[i];
                join_start_list.put_etomo_number(
                    i as i32,
                    &EtomoNumber::new_from_instance(Some(data.get_join_final_start())),
                );
                join_end_list.put_etomo_number(
                    i as i32,
                    &EtomoNumber::new_from_instance(Some(data.get_join_final_end())),
                );
            }
            self.join_shift_in_x
                .lock()
                .unwrap()
                .set_const_etomo_number(Some(&meta_data.get_shift_in_x()));
            self.join_shift_in_y
                .lock()
                .unwrap()
                .set_const_etomo_number(Some(&meta_data.get_shift_in_y()));
            self.join_size_in_x
                .lock()
                .unwrap()
                .set_const_etomo_number(Some(&meta_data.get_size_in_x()));
            self.join_size_in_y
                .lock()
                .unwrap()
                .set_const_etomo_number(Some(&meta_data.get_size_in_y()));
        }
    }

    /// Java `setRefineTrial(boolean)`.
    pub fn set_refine_trial(&self, trial: bool) {
        self.refine_trial.lock().unwrap().set_boolean(trial);
    }

    /// Java `getNewShiftInX(int, int)`.
    ///
    /// Fixed in translation: JoinState.java:462 and :465 build the
    /// `NullRequiredNumberException` messages from `joinTrialShiftInY.getName()` and
    /// `joinTrialSizeInY.getName()` while testing the X parameters (a copy of
    /// `getNewShiftInY`); the messages here name the X parameters that were tested.
    pub fn get_new_shift_in_x(
        &self,
        min: i32,
        max: i32,
    ) -> Result<i32, NullRequiredNumberException> {
        let join_trial_shift_in_x = self.join_trial_shift_in_x.lock().unwrap();
        if join_trial_shift_in_x.is_null() {
            return Err(NullRequiredNumberException::new(&format!(
                "{} is null.",
                join_trial_shift_in_x.get_name()
            )));
        }
        let join_trial_size_in_x = self.join_trial_size_in_x.lock().unwrap();
        if join_trial_size_in_x.is_null() {
            return Err(NullRequiredNumberException::new(&format!(
                "{} is null.",
                join_trial_size_in_x.get_name()
            )));
        }
        Ok(join_trial_shift_in_x
            .get_int()
            .wrapping_add(join_trial_size_in_x.get_int().wrapping_add(1) / 2)
            .wrapping_sub(max.wrapping_add(min) / 2))
    }

    /// Java `getRotationAngles(Integer)`.
    pub fn get_rotation_angles(&self, index: i32) -> Option<SlicerAngles> {
        let rotation_angles_list = self.rotation_angles_list.lock().unwrap();
        let rotation_angles_list = rotation_angles_list.as_ref()?;
        rotation_angles_list.get(&index).cloned()
    }

    /// Java `setDebug(boolean)`.
    pub fn set_debug(&self, debug: bool) {
        *self.debug.lock().unwrap() = debug;
    }

    /// Java `isRefineStartListEmpty()`.
    pub fn is_refine_start_list_empty(&self) -> bool {
        self.refine_start_list.lock().unwrap().is_empty()
    }

    /// Java `setJoinSizeInX(boolean, ConstEtomoNumber)`.
    pub fn set_join_size_in_x(&self, trial: bool, size_in_x: Option<&ConstEtomoNumber>) {
        if trial {
            self.join_trial_size_in_x
                .lock()
                .unwrap()
                .set_const_etomo_number(size_in_x);
        } else {
            self.join_size_in_x
                .lock()
                .unwrap()
                .set_const_etomo_number(size_in_x);
        }
    }

    /// Java `setJoinSizeInY(boolean, ConstEtomoNumber)`.
    pub fn set_join_size_in_y(&self, trial: bool, size_in_y: Option<&ConstEtomoNumber>) {
        if trial {
            self.join_trial_size_in_y
                .lock()
                .unwrap()
                .set_const_etomo_number(size_in_y);
        } else {
            self.join_size_in_y
                .lock()
                .unwrap()
                .set_const_etomo_number(size_in_y);
        }
    }

    /// Java `setJoinShiftInX(boolean, ConstEtomoNumber)`.
    pub fn set_join_shift_in_x(&self, trial: bool, shift_in_x: Option<&ConstEtomoNumber>) {
        if trial {
            self.join_trial_shift_in_x
                .lock()
                .unwrap()
                .set_const_etomo_number(shift_in_x);
        } else {
            self.join_shift_in_x
                .lock()
                .unwrap()
                .set_const_etomo_number(shift_in_x);
        }
    }

    /// Java `setJoinShiftInY(boolean, ConstEtomoNumber)`.
    pub fn set_join_shift_in_y(&self, trial: bool, shift_in_y: Option<&ConstEtomoNumber>) {
        if trial {
            self.join_trial_shift_in_y
                .lock()
                .unwrap()
                .set_const_etomo_number(shift_in_y);
        } else {
            self.join_shift_in_y
                .lock()
                .unwrap()
                .set_const_etomo_number(shift_in_y);
        }
    }

    /// Java `setJoinLocalFits(boolean, ConstEtomoNumber)`.
    pub fn set_join_local_fits(&self, trial: bool, local_fits: Option<&ConstEtomoNumber>) {
        if trial {
            self.join_trial_local_fits
                .lock()
                .unwrap()
                .set_const_etomo_number(local_fits);
        } else {
            self.join_local_fits
                .lock()
                .unwrap()
                .set_const_etomo_number(local_fits);
        }
    }

    /// Java `setJoinStartList(boolean, ConstIntKeyList)`.
    pub fn set_join_start_list(&self, trial: bool, start_list: Option<&dyn ConstIntKeyList>) {
        if trial {
            let mut join_trial_start_list = self.join_trial_start_list.lock().unwrap();
            join_trial_start_list.reset();
            join_trial_start_list.set_int_key_list(start_list);
        } else {
            let mut join_start_list = self.join_start_list.lock().unwrap();
            join_start_list.reset();
            join_start_list.set_int_key_list(start_list);
        }
    }

    /// Java `setJoinTrialUseEveryNSlices(ConstEtomoNumber)`.
    pub fn set_join_trial_use_every_n_slices(&self, use_every_n_slices: Option<&ConstEtomoNumber>) {
        self.join_trial_use_every_n_slices
            .lock()
            .unwrap()
            .set_const_etomo_number(use_every_n_slices);
    }

    /// Java `setRefineTrialUseEveryNSlices(ConstEtomoNumber)`.
    pub fn set_refine_trial_use_every_n_slices(
        &self,
        use_every_n_slices: Option<&ConstEtomoNumber>,
    ) {
        self.refine_trial_use_every_n_slices
            .lock()
            .unwrap()
            .set_const_etomo_number(use_every_n_slices);
    }

    /// Java `getJoinTrialUseEveryNSlices()`: a copy of the field (see the module
    /// header).
    pub fn get_join_trial_use_every_n_slices(&self) -> ConstEtomoNumber {
        let guard = self.join_trial_use_every_n_slices.lock().unwrap();
        let number: &ConstEtomoNumber = &guard;
        number.clone()
    }

    /// Java `getRefineTrialUseEveryNSlices()`: a copy of the field (see the module
    /// header).
    pub fn get_refine_trial_use_every_n_slices(&self) -> ConstEtomoNumber {
        let guard = self.refine_trial_use_every_n_slices.lock().unwrap();
        let number: &ConstEtomoNumber = &guard;
        number.clone()
    }

    /// Java `setRefineStartList(ConstIntKeyList)`.
    pub fn set_refine_start_list(&self, start_list: Option<&dyn ConstIntKeyList>) {
        let mut refine_start_list = self.refine_start_list.lock().unwrap();
        refine_start_list.reset();
        refine_start_list.set_int_key_list(start_list);
    }

    /// Java `setRefineEndList(ConstIntKeyList)`.
    pub fn set_refine_end_list(&self, end_list: Option<&dyn ConstIntKeyList>) {
        let mut refine_end_list = self.refine_end_list.lock().unwrap();
        refine_end_list.reset();
        refine_end_list.set_int_key_list(end_list);
    }

    /// Java `setJoinEndList(boolean, ConstIntKeyList)`.
    pub fn set_join_end_list(&self, trial: bool, end_list: Option<&dyn ConstIntKeyList>) {
        if trial {
            let mut join_trial_end_list = self.join_trial_end_list.lock().unwrap();
            join_trial_end_list.reset();
            join_trial_end_list.set_int_key_list(end_list);
        } else {
            let mut join_end_list = self.join_end_list.lock().unwrap();
            join_end_list.reset();
            join_end_list.set_int_key_list(end_list);
        }
    }

    /// Java `setJoinTrialBinning(ConstEtomoNumber)`.
    pub fn set_join_trial_binning(&self, binning: Option<&ConstEtomoNumber>) {
        self.join_trial_binning
            .lock()
            .unwrap()
            .set_const_etomo_number(binning);
    }

    /// Java `setJoinAlignmentRefSection(boolean, ConstEtomoNumber)`.
    pub fn set_join_alignment_ref_section(
        &self,
        trial: bool,
        alignment_ref_section: Option<&ConstEtomoNumber>,
    ) {
        if trial {
            self.join_trial_alignment_ref_section
                .lock()
                .unwrap()
                .set_const_etomo_number(alignment_ref_section);
        } else {
            self.join_alignment_ref_section
                .lock()
                .unwrap()
                .set_const_etomo_number(alignment_ref_section);
        }
    }

    /// Java `getNewShiftInY(int, int)`.  calculate shift in y
    pub fn get_new_shift_in_y(
        &self,
        min: i32,
        max: i32,
    ) -> Result<i32, NullRequiredNumberException> {
        let join_trial_shift_in_y = self.join_trial_shift_in_y.lock().unwrap();
        if join_trial_shift_in_y.is_null() {
            return Err(NullRequiredNumberException::new(&format!(
                "{} is null.",
                join_trial_shift_in_y.get_name()
            )));
        }
        let join_trial_size_in_y = self.join_trial_size_in_y.lock().unwrap();
        if join_trial_size_in_y.is_null() {
            return Err(NullRequiredNumberException::new(&format!(
                "{} is null.",
                join_trial_size_in_y.get_name()
            )));
        }
        Ok(join_trial_shift_in_y
            .get_int()
            .wrapping_add(join_trial_size_in_y.get_int().wrapping_add(1) / 2)
            .wrapping_sub(max.wrapping_add(min) / 2))
    }

    /// Java `isSampleProduced()`.
    pub fn is_sample_produced(&self) -> bool {
        *self.sample_produced.lock().unwrap()
    }

    /// Java `setGapsExist(boolean)`.
    pub fn set_gaps_exist(&self, gaps_exist: bool) {
        let mut field = self.gaps_exist.lock().unwrap();
        let instance = field.take();
        *field = EtomoBoolean2::set_instance_boolean(instance, gaps_exist, GAPS_EXIST_KEY);
    }

    /// Java `isGapsExist()`.
    pub fn is_gaps_exist(&self) -> bool {
        let gaps_exist = self.gaps_exist.lock().unwrap();
        match gaps_exist.as_ref() {
            None => false,
            Some(gaps_exist) => gaps_exist.is(),
        }
    }

    /// Java `setRevertState(boolean)`.
    pub fn set_revert_state(&self, enable_revert: bool) {
        let rotation_angles_list = self.rotation_angles_list.lock().unwrap();
        if enable_revert && rotation_angles_list.is_some() {
            *self.revert_rotation_angles_list.lock().unwrap() = rotation_angles_list.clone();
            let total_rows = self.total_rows.lock().unwrap();
            self.revert_total_rows
                .lock()
                .unwrap()
                .set_const_etomo_number(Some(&**total_rows));
        } else {
            *self.revert_rotation_angles_list.lock().unwrap() = None;
            self.revert_total_rows.lock().unwrap().set_string(Some(""));
        }
    }

    /// Java `revert()`.
    ///
    /// Fixed in translation: JoinState.java:684 assigns the revert table itself
    /// (`rotationAnglesList = revertRotationAnglesList`), so after a revert the two
    /// fields alias one `Hashtable` and a later row edit also rewrites the saved revert
    /// state.  The live list here is a copy of the revert state, which stays as
    /// `setRevertState` saved it.
    pub fn revert(&self) {
        let revert_rotation_angles_list = self.revert_rotation_angles_list.lock().unwrap().clone();
        *self.rotation_angles_list.lock().unwrap() = revert_rotation_angles_list;
        let revert_total_rows = self.revert_total_rows.lock().unwrap().clone();
        self.total_rows
            .lock()
            .unwrap()
            .set_const_etomo_number(Some(&*revert_total_rows));
    }

    /// Java `deleteRow(int)`.  The Java `IllegalStateException` is the `Err` message.
    pub fn delete_row(&self, row_index: i32) -> Result<(), String> {
        let mut rotation_angles_list = self.rotation_angles_list.lock().unwrap();
        let rotation_angles_list = match rotation_angles_list.as_mut() {
            None => return Ok(()),
            Some(rotation_angles_list) => rotation_angles_list,
        };
        let total_rows = self.total_rows.lock().unwrap();
        if row_index >= total_rows.get_int() {
            return Err(format!("rowIndex={},totalRows={}", row_index, *total_rows));
        }
        // delete row
        let mut prev_index = row_index;
        rotation_angles_list.remove(&prev_index);
        // move the other rows up one row
        let mut i = row_index + 1;
        while i < total_rows.get_int() {
            let cur_index = i;
            let rotation_angles = rotation_angles_list.remove(&cur_index);
            if let Some(rotation_angles) = rotation_angles {
                rotation_angles_list.insert(prev_index, rotation_angles);
            }
            prev_index = cur_index;
            i += 1;
        }
        Ok(())
    }

    /// Java `moveRowUp(int)`.  The Java `IllegalStateException` is the `Err` message.
    pub fn move_row_up(&self, row_index: i32) -> Result<(), String> {
        let mut rotation_angles_list = self.rotation_angles_list.lock().unwrap();
        let rotation_angles_list = match rotation_angles_list.as_mut() {
            None => return Ok(()),
            Some(rotation_angles_list) => rotation_angles_list,
        };
        let total_rows = self.total_rows.lock().unwrap();
        if row_index == 0 || row_index >= total_rows.get_int() {
            return Err(format!("rowIndex={},totalRows={}", row_index, *total_rows));
        }
        let cur_index = row_index;
        let prev_index = row_index - 1;
        // swap the current row with the one above it
        let rotation_angles = rotation_angles_list.remove(&cur_index);
        let prev_rotation_angles = match rotation_angles {
            Some(rotation_angles) => rotation_angles_list.insert(prev_index, rotation_angles),
            None => rotation_angles_list.remove(&prev_index),
        };
        if let Some(prev_rotation_angles) = prev_rotation_angles {
            rotation_angles_list.insert(cur_index, prev_rotation_angles);
        }
        Ok(())
    }

    /// Java `moveRowDown(int)`.  The Java `IllegalStateException` is the `Err` message.
    pub fn move_row_down(&self, row_index: i32) -> Result<(), String> {
        let mut rotation_angles_list = self.rotation_angles_list.lock().unwrap();
        let rotation_angles_list = match rotation_angles_list.as_mut() {
            None => return Ok(()),
            Some(rotation_angles_list) => rotation_angles_list,
        };
        let total_rows = self.total_rows.lock().unwrap();
        if row_index >= total_rows.get_int().wrapping_sub(1) {
            return Err(format!("rowIndex={},totalRows={}", row_index, *total_rows));
        }
        let cur_index = row_index;
        let next_index = row_index + 1;
        // swap the current row with the one below it
        let rotation_angles = rotation_angles_list.remove(&cur_index);
        let next_rotation_angles = match rotation_angles {
            Some(rotation_angles) => rotation_angles_list.insert(next_index, rotation_angles),
            None => rotation_angles_list.remove(&next_index),
        };
        if let Some(next_rotation_angles) = next_rotation_angles {
            rotation_angles_list.insert(cur_index, next_rotation_angles);
        }
        Ok(())
    }

    /// Java `setRotationAnglesList(Hashtable)`.
    pub fn set_rotation_angles_list(&self, rotation_angles_list: Option<RotationAnglesList>) {
        *self.rotation_angles_list.lock().unwrap() = rotation_angles_list;
    }

    /// Java `setTotalRows(int)`.
    pub fn set_total_rows(&self, total_rows: i32) {
        self.total_rows.lock().unwrap().set_int(total_rows);
    }

    /// Java `setXfModelOutputFile(String)`.
    pub fn set_xf_model_output_file(&self, xf_model_output_file: Option<&str>) {
        *self.xf_model_output_file.lock().unwrap() = xf_model_output_file.map(str::to_owned);
    }

    /// Java `setSampleProduced(boolean)`.
    pub fn set_sample_produced(&self, sample_produced: bool) {
        *self.sample_produced.lock().unwrap() = sample_produced;
    }

    /// Java final `getDoneMode()`.
    pub fn get_done_mode(&self) -> i32 {
        self.done_mode.lock().unwrap().get_int()
    }

    /// Java final `setDoneMode(int)`.
    pub fn set_done_mode(&self, done_mode: i32) {
        self.done_mode.lock().unwrap().set_int(done_mode);
    }

    /// Java final `clearDoneMode()`.
    pub fn clear_done_mode(&self) {
        self.done_mode.lock().unwrap().reset();
    }
}

/// Java `toString()`.
impl std::fmt::Display for JoinState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let join_version = self.join_version.lock().unwrap().to_string();
        let join_size_in_x = self.join_size_in_x.lock().unwrap().to_string();
        write!(
            f,
            "[joinVersion={},joinSizeInX={}]",
            join_version, join_size_in_x
        )
    }
}

impl BaseState for JoinState {
    /// Java package-private `createPrepend(String)`, implementing `BaseState`.
    fn create_prepend(&self, prepend: &str) -> String {
        if prepend.is_empty() {
            return GROUP_STRING.to_string();
        }
        format!("{}.{}", prepend, GROUP_STRING)
    }
}

/// Java `Storable`, through `BaseState`.
impl crate::imod::etomo::storage::storable::Storable for JoinState {
    fn store(&self, properties: &mut BTreeMap<String, String>) {
        JoinState::store(self, properties);
    }

    fn store_with_prepend(&self, properties: &mut BTreeMap<String, String>, prepend: &str) {
        JoinState::store_with_prepend(self, properties, prepend);
    }

    fn load(&self, properties: &mut BTreeMap<String, String>) {
        JoinState::load(self, properties);
    }

    fn load_with_prepend(&self, properties: &mut BTreeMap<String, String>, prepend: &str) {
        JoinState::load_with_prepend(self, properties, prepend);
    }
}

impl ConstJoinState for JoinState {
    /// Java `getRefineTrial()`: a copy of the `EtomoBoolean2` (see
    /// `const_join_state.rs`).
    fn get_refine_trial(&self) -> EtomoBoolean2 {
        self.refine_trial.lock().unwrap().clone()
    }

    /// Java `getJoinStartListWalker(boolean)`.
    fn get_join_start_list_walker(&self, trial: bool) -> Walker<'_> {
        if trial {
            return self.join_trial_start_list.get_walker();
        }
        self.join_start_list.get_walker()
    }

    /// Java `getJoinEndListWalker(boolean)`.
    fn get_join_end_list_walker(&self, trial: bool) -> Walker<'_> {
        if trial {
            return self.join_trial_end_list.get_walker();
        }
        self.join_end_list.get_walker()
    }

    /// Java `getRefineStartListWalker()`.
    fn get_refine_start_list_walker(&self) -> Walker<'_> {
        self.refine_start_list.get_walker()
    }

    /// Java `getRefineEndListWalker()`.
    fn get_refine_end_list_walker(&self) -> Walker<'_> {
        self.refine_end_list.get_walker()
    }

    /// Java `getJoinSizeInX(boolean)`.
    fn get_join_size_in_x(&self, trial: bool) -> ConstEtomoNumber {
        if trial {
            let guard = self.join_trial_size_in_x.lock().unwrap();
            let number: &ConstEtomoNumber = &guard;
            return number.clone();
        }
        let guard = self.join_size_in_x.lock().unwrap();
        let number: &ConstEtomoNumber = &guard;
        number.clone()
    }

    /// Java `getJoinSizeInXParameter(boolean)`.
    fn get_join_size_in_x_parameter(&self, trial: bool) -> ScriptParameter {
        if trial {
            return self.join_trial_size_in_x.lock().unwrap().clone();
        }
        self.join_size_in_x.lock().unwrap().clone()
    }

    /// Java `getJoinSizeInY(boolean)`.
    fn get_join_size_in_y(&self, trial: bool) -> ConstEtomoNumber {
        if trial {
            let guard = self.join_trial_size_in_y.lock().unwrap();
            let number: &ConstEtomoNumber = &guard;
            return number.clone();
        }
        let guard = self.join_size_in_y.lock().unwrap();
        let number: &ConstEtomoNumber = &guard;
        number.clone()
    }

    /// Java `getJoinSizeInYParameter(boolean)`.
    fn get_join_size_in_y_parameter(&self, trial: bool) -> ScriptParameter {
        if trial {
            return self.join_trial_size_in_y.lock().unwrap().clone();
        }
        self.join_size_in_y.lock().unwrap().clone()
    }

    /// Java `isJoinLocalFits(boolean)`.
    fn is_join_local_fits(&self, trial: bool) -> bool {
        if trial {
            return self.join_trial_local_fits.lock().unwrap().is();
        }
        self.join_local_fits.lock().unwrap().is()
    }

    /// Java `getJoinShiftInX(boolean)`.
    fn get_join_shift_in_x(&self, trial: bool) -> ConstEtomoNumber {
        if trial {
            let guard = self.join_trial_shift_in_x.lock().unwrap();
            let number: &ConstEtomoNumber = &guard;
            return number.clone();
        }
        let guard = self.join_shift_in_x.lock().unwrap();
        let number: &ConstEtomoNumber = &guard;
        number.clone()
    }

    /// Java `getJoinShiftInXParameter(boolean)`.
    fn get_join_shift_in_x_parameter(&self, trial: bool) -> ScriptParameter {
        if trial {
            return self.join_trial_shift_in_x.lock().unwrap().clone();
        }
        self.join_shift_in_x.lock().unwrap().clone()
    }

    /// Java `getJoinShiftInY(boolean)`.
    fn get_join_shift_in_y(&self, trial: bool) -> ConstEtomoNumber {
        if trial {
            let guard = self.join_trial_shift_in_y.lock().unwrap();
            let number: &ConstEtomoNumber = &guard;
            return number.clone();
        }
        let guard = self.join_shift_in_y.lock().unwrap();
        let number: &ConstEtomoNumber = &guard;
        number.clone()
    }

    /// Java `getJoinShiftInYParameter(boolean)`.
    fn get_join_shift_in_y_parameter(&self, trial: bool) -> ScriptParameter {
        if trial {
            return self.join_trial_shift_in_y.lock().unwrap().clone();
        }
        self.join_shift_in_y.lock().unwrap().clone()
    }

    /// Java `getJoinTrialBinning()`.
    fn get_join_trial_binning(&self) -> ConstEtomoNumber {
        let guard = self.join_trial_binning.lock().unwrap();
        let number: &ConstEtomoNumber = &guard;
        number.clone()
    }

    /// Java `getJoinAlignmentRefSection(boolean)`.
    fn get_join_alignment_ref_section(&self, trial: bool) -> ConstEtomoNumber {
        if trial {
            let guard = self.join_trial_alignment_ref_section.lock().unwrap();
            let number: &ConstEtomoNumber = &guard;
            return number.clone();
        }
        let guard = self.join_alignment_ref_section.lock().unwrap();
        let number: &ConstEtomoNumber = &guard;
        number.clone()
    }

    /// Java `getXfModelOutputFile()`.
    fn get_xf_model_output_file(&self) -> Option<String> {
        self.xf_model_output_file.lock().unwrap().clone()
    }
}

/// A `JoinState` list field: Java's `IntKeyList` methods are `synchronized`, so the
/// shared list is read one locked call at a time, which is what lets a `Walker` borrow
/// the field while other threads use the state (see the module header).
impl ConstIntKeyList for Mutex<IntKeyList> {
    fn get_first_key(&self) -> i32 {
        self.lock().unwrap().get_first_key()
    }

    fn get_last_key(&self) -> i32 {
        self.lock().unwrap().get_last_key()
    }

    fn get_string(&self, key: i32) -> Option<String> {
        self.lock().unwrap().get_string(key)
    }

    fn get_etomo_number(&self, key: i32) -> Option<EtomoNumber> {
        self.lock().unwrap().get_etomo_number(key)
    }

    fn contains_key(&self, key: i32) -> bool {
        self.lock().unwrap().contains_key(key)
    }

    fn size(&self) -> i32 {
        self.lock().unwrap().size()
    }

    fn get_walker(&self) -> Walker<'_> {
        Walker::new(self)
    }

    fn is_empty(&self) -> bool {
        ConstIntKeyList::is_empty(&*self.lock().unwrap())
    }
}
