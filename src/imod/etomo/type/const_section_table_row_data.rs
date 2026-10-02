//! `IMOD/Etomo/src/etomo/type/ConstSectionTableRowData.java`.
//!
//! **Shape.**  The Java class is abstract and carries state that its one subclass,
//! `SectionTableRowData`, reads and writes directly.  That state is
//! [`ConstSectionTableRowDataBase`]; the class's concrete public methods are the
//! provided methods of the [`ConstSectionTableRowData`] trait, which reaches the state
//! through `base()`.  Callers holding only the read-only view take
//! `&dyn ConstSectionTableRowData`, as the Java callers take the abstract type.
#![allow(dead_code)]

use std::collections::BTreeMap;
use std::path::PathBuf;

use super::const_etomo_number::{ConstEtomoNumber, INTEGER_NULL_VALUE, Type};
use super::etomo_boolean2::EtomoBoolean2;
use super::etomo_number::EtomoNumber;
use crate::imod::etomo::util::utilities::java_io_file_get_absolute_path;

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java package-private `VERSION`.
pub(crate) const VERSION: &str = "1.1";
/// Java package-private `VERSION_KEY`.
pub(crate) const VERSION_KEY: &str = "SectionTableRowData.Version";
/// Java package-private `groupString`.
pub(crate) const GROUP_STRING: &str = "SectionTableRow";
/// Java package-private `setupSectionString`.
pub(crate) const SETUP_SECTION_STRING: &str = "Section";
// For conversion from version 1.0
/// Java package-private `setupXMaxString`.
pub(crate) const SETUP_X_MAX_STRING: &str = "XMax";
/// Java package-private `setupYMaxString`.
pub(crate) const SETUP_Y_MAX_STRING: &str = "YMax";
/// Java package-private `setupZMaxString`.
pub(crate) const SETUP_Z_MAX_STRING: &str = "ZMax";
/// Java package-private `COS_X_Y_THRESHOLD`.
pub(crate) const COS_X_Y_THRESHOLD: f64 = 0.5;

/// The state of the abstract Java class `ConstSectionTableRowData`.
#[derive(Clone, Debug)]
pub struct ConstSectionTableRowDataBase {
    /// Java field `rowNumber`: key in the .ejf file, not displayed.
    pub(crate) row_number: EtomoNumber,
    /// Java field `sampleBottomStart`.
    pub(crate) sample_bottom_start: EtomoNumber,
    /// Java field `sampleBottomEnd`.
    pub(crate) sample_bottom_end: EtomoNumber,
    /// Java field `sampleTopStart`.
    pub(crate) sample_top_start: EtomoNumber,
    /// Java field `sampleTopEnd`.
    pub(crate) sample_top_end: EtomoNumber,
    /// Java field `setupFinalStart`.
    pub(crate) setup_final_start: EtomoNumber,
    /// Java field `setupFinalEnd`.
    pub(crate) setup_final_end: EtomoNumber,
    /// Java field `joinFinalStart`.
    pub(crate) join_final_start: EtomoNumber,
    /// Java field `joinFinalEnd`.
    pub(crate) join_final_end: EtomoNumber,
    /// Java field `rotationAngleX`.
    pub(crate) rotation_angle_x: EtomoNumber,
    /// Java field `rotationAngleY`.
    pub(crate) rotation_angle_y: EtomoNumber,
    /// Java field `rotationAngleZ`.
    pub(crate) rotation_angle_z: EtomoNumber,

    /// Java field `setupSection`.
    pub(crate) setup_section: Option<PathBuf>,
    /// Java field `joinSection`.
    pub(crate) join_section: Option<PathBuf>,
    /// Java field `setupXMax`.
    pub(crate) setup_x_max: i32,
    /// Java field `joinXMax`.
    pub(crate) join_x_max: i32,
    /// Java field `setupYMax`.
    pub(crate) setup_y_max: i32,
    /// Java field `joinYMax`.
    pub(crate) join_y_max: i32,
    /// Java field `setupZMax`.
    pub(crate) setup_z_max: i32,
    /// Java field `joinZMax`.
    pub(crate) join_z_max: i32,

    // state - these should not be saved to the .ejf file, but they are necessary
    // for remembering the state of a row that is being retrieved from meta data.
    /// Java private field `imodIndex`.
    imod_index: i32,
    /// Java private field `imodRotIndex`.
    imod_rot_index: i32,
    /// Java private field `sectionExpanded`.
    section_expanded: bool,
    /// Java package-private field `invalidReason`.
    pub(crate) invalid_reason: Option<String>,
    /// Java package-private field `inverted`.
    pub(crate) inverted: EtomoBoolean2,
}

impl ConstSectionTableRowDataBase {
    /// Java `ConstSectionTableRowData(int)`.  Construct an instance with a row number.
    pub(crate) fn new(row_number: i32) -> ConstSectionTableRowDataBase {
        // construct
        let mut instance = ConstSectionTableRowDataBase {
            row_number: EtomoNumber::new_with_name("RowNumber"),
            sample_bottom_start: EtomoNumber::new_with_name("SampleBottomStart"),
            sample_bottom_end: EtomoNumber::new_with_name("SampleBottomEnd"),
            sample_top_start: EtomoNumber::new_with_name("SampleTopStart"),
            sample_top_end: EtomoNumber::new_with_name("SampleTopEnd"),
            setup_final_start: EtomoNumber::new_with_name("FinalStart"),
            setup_final_end: EtomoNumber::new_with_name("FinalEnd"),
            join_final_start: EtomoNumber::new(),
            join_final_end: EtomoNumber::new(),
            rotation_angle_x: EtomoNumber::new_with_type_and_name(Type::Double, "RotationAngleX"),
            rotation_angle_y: EtomoNumber::new_with_type_and_name(Type::Double, "RotationAngleY"),
            rotation_angle_z: EtomoNumber::new_with_type_and_name(Type::Double, "RotationAngleZ"),
            setup_section: None,
            join_section: None,
            setup_x_max: INTEGER_NULL_VALUE,
            join_x_max: INTEGER_NULL_VALUE,
            setup_y_max: INTEGER_NULL_VALUE,
            join_y_max: INTEGER_NULL_VALUE,
            setup_z_max: INTEGER_NULL_VALUE,
            join_z_max: INTEGER_NULL_VALUE,
            imod_index: -1,
            imod_rot_index: -1,
            section_expanded: false,
            invalid_reason: None,
            inverted: EtomoBoolean2::new_with_name("Inverted"),
        };
        // configure
        instance
            .sample_bottom_start
            .set_description(Some("Sample Slices, Bottom, Start"));
        instance
            .sample_bottom_end
            .set_description(Some("Sample Slices, Bottom, End"));
        instance
            .sample_top_start
            .set_description(Some("Sample Slices, Top, Start"));
        instance
            .sample_top_end
            .set_description(Some("Sample Slices, Top, End"));
        instance
            .setup_final_start
            .set_description(Some("Final, Start"));
        instance.setup_final_start.set_display_value_int(1);
        instance.setup_final_end.set_description(Some("Final, End"));
        instance
            .join_final_start
            .set_description(Some("Final, Start"));
        instance.join_final_end.set_description(Some("Final, End"));
        instance
            .rotation_angle_x
            .set_description(Some("Rotation Angles, X"));
        instance.rotation_angle_x.set_default_int(0);
        instance
            .rotation_angle_y
            .set_description(Some("Rotation Angles, Y"));
        instance.rotation_angle_y.set_default_int(0);
        instance
            .rotation_angle_z
            .set_description(Some("Rotation Angles, Z"));
        instance.rotation_angle_z.set_default_int(0);
        // initialize
        instance.row_number.set_int(row_number);
        instance
    }

    /// Java `ConstSectionTableRowData(ConstSectionTableRowData)`, the copy constructor.
    /// Does a deep copy.
    ///
    /// The source copies the three rotation angles with `new ScriptParameter(...)` into
    /// fields declared `EtomoNumber`; a `ScriptParameter` built that way stores, loads
    /// and compares as the `EtomoNumber` it copies, so they are copied as `EtomoNumber`s.
    pub(crate) fn new_from(
        const_section_table_row_data: &ConstSectionTableRowDataBase,
    ) -> ConstSectionTableRowDataBase {
        let source = const_section_table_row_data;
        let mut inverted = EtomoBoolean2::new_with_name("Inverted");
        inverted.set_const_etomo_number(Some(&***source.inverted));
        ConstSectionTableRowDataBase {
            // deep copy
            imod_index: source.imod_index,
            imod_rot_index: source.imod_rot_index,
            section_expanded: source.section_expanded,
            row_number: EtomoNumber::new_from_instance(Some(&source.row_number.base)),
            setup_section: source
                .setup_section
                .as_ref()
                .map(|file| PathBuf::from(java_io_file_get_absolute_path(&file.to_string_lossy()))),
            join_section: source
                .join_section
                .as_ref()
                .map(|file| PathBuf::from(java_io_file_get_absolute_path(&file.to_string_lossy()))),
            sample_bottom_start: EtomoNumber::new_from_instance(Some(
                &source.sample_bottom_start.base,
            )),
            sample_bottom_end: EtomoNumber::new_from_instance(Some(&source.sample_bottom_end.base)),
            sample_top_start: EtomoNumber::new_from_instance(Some(&source.sample_top_start.base)),
            sample_top_end: EtomoNumber::new_from_instance(Some(&source.sample_top_end.base)),
            setup_final_start: EtomoNumber::new_from_instance(Some(&source.setup_final_start.base)),
            setup_final_end: EtomoNumber::new_from_instance(Some(&source.setup_final_end.base)),
            join_final_start: EtomoNumber::new_from_instance(Some(&source.join_final_start.base)),
            join_final_end: EtomoNumber::new_from_instance(Some(&source.join_final_end.base)),
            rotation_angle_x: EtomoNumber::new_from_instance(Some(&source.rotation_angle_x.base)),
            rotation_angle_y: EtomoNumber::new_from_instance(Some(&source.rotation_angle_y.base)),
            rotation_angle_z: EtomoNumber::new_from_instance(Some(&source.rotation_angle_z.base)),
            setup_x_max: source.setup_x_max,
            join_x_max: source.join_x_max,
            setup_y_max: source.setup_y_max,
            join_y_max: source.join_y_max,
            setup_z_max: source.setup_z_max,
            join_z_max: source.join_z_max,
            invalid_reason: None,
            inverted,
        }
    }
}

/// Java static `createPrepend(String, ConstEtomoNumber)`.  The source compares
/// `prepend == ""` by reference, which is true for the interned literal every caller
/// passes; it is compared by value here.  A null prepend concatenates as `"null"`.
pub fn create_prepend_static(prepend: Option<&str>, row_number: &ConstEtomoNumber) -> String {
    if prepend == Some("") {
        return format!("{}.{}", GROUP_STRING, row_number);
    }
    format!(
        "{}.{}.{}",
        prepend.unwrap_or("null"),
        GROUP_STRING,
        row_number
    )
}

/// Java private static `convertToString(int)`.
fn convert_to_string_int(value: i32) -> String {
    if value == i32::MIN {
        return String::new();
    }
    value.to_string()
}

/// Java private static `convertToString(double)`.
fn convert_to_string_double(value: f64) -> String {
    if value.is_nan() {
        return String::new();
    }
    crate::imod::etomo::r#type::const_etomo_number::java_lang_double_to_string(value)
}

/// Java abstract class `ConstSectionTableRowData`: its concrete methods.
pub trait ConstSectionTableRowData: std::fmt::Display {
    /// The abstract class's state.
    fn base(&self) -> &ConstSectionTableRowDataBase;

    /// Java package-private `paramString`.
    fn param_string(&self) -> String {
        let base = self.base();
        format!("rowNumber={},inverted={}", base.row_number, base.inverted)
    }

    /// Java `store(Properties)`.
    fn store(&self, props: &mut BTreeMap<String, String>) {
        ConstSectionTableRowData::store_with_prepend(self, props, Some(""));
    }

    /// Java `store(Properties, String)`.
    ///
    /// Fixed in translation: ConstSectionTableRowData.java:217 stores
    /// `setupSection.getAbsolutePath()`, a NullPointerException when no setup section
    /// is set.  Here the `Section` key is not written in that case (`load` then leaves
    /// the section unset, which is what the missing key means to it).
    fn store_with_prepend(&self, props: &mut BTreeMap<String, String>, prepend: Option<&str>) {
        let base = self.base();
        let prepend = self.create_prepend(prepend);
        let group = format!("{}.", prepend);
        props.insert(format!("{}{}", group, VERSION_KEY), VERSION.to_string());
        base.row_number
            .store_with_prepend(props, Some(prepend.as_str()));
        if let Some(setup_section) = &base.setup_section {
            props.insert(
                format!("{}{}", group, SETUP_SECTION_STRING),
                java_io_file_get_absolute_path(&setup_section.to_string_lossy()),
            );
        }
        base.sample_bottom_start
            .store_with_prepend(props, Some(prepend.as_str()));
        base.sample_bottom_end
            .store_with_prepend(props, Some(prepend.as_str()));
        base.sample_top_start
            .store_with_prepend(props, Some(prepend.as_str()));
        base.sample_top_end
            .store_with_prepend(props, Some(prepend.as_str()));
        base.setup_final_start
            .store_with_prepend(props, Some(prepend.as_str()));
        base.setup_final_end
            .store_with_prepend(props, Some(prepend.as_str()));
        base.rotation_angle_x
            .store_with_prepend(props, Some(prepend.as_str()));
        base.rotation_angle_y
            .store_with_prepend(props, Some(prepend.as_str()));
        base.rotation_angle_z
            .store_with_prepend(props, Some(prepend.as_str()));
        base.inverted
            .store_with_prepend(props, Some(prepend.as_str()));
    }

    /// Java package-private `remove(Properties, String)`.
    fn remove(&self, props: &mut BTreeMap<String, String>, prepend: Option<&str>) {
        let base = self.base();
        let prepend = self.create_prepend(prepend);
        let group = format!("{}.", prepend);
        props.remove(&format!("{}{}", group, VERSION_KEY));
        base.row_number
            .remove_with_prepend(props, Some(prepend.as_str()));
        props.remove(&format!("{}{}", group, SETUP_SECTION_STRING));
        base.sample_bottom_start
            .remove_with_prepend(props, Some(prepend.as_str()));
        base.sample_bottom_end
            .remove_with_prepend(props, Some(prepend.as_str()));
        base.sample_top_start
            .remove_with_prepend(props, Some(prepend.as_str()));
        base.sample_top_end
            .remove_with_prepend(props, Some(prepend.as_str()));
        base.setup_final_start
            .remove_with_prepend(props, Some(prepend.as_str()));
        base.setup_final_end
            .remove_with_prepend(props, Some(prepend.as_str()));
        base.rotation_angle_x
            .remove_with_prepend(props, Some(prepend.as_str()));
        base.rotation_angle_y
            .remove_with_prepend(props, Some(prepend.as_str()));
        base.rotation_angle_z
            .remove_with_prepend(props, Some(prepend.as_str()));
        base.inverted
            .remove_with_prepend(props, Some(prepend.as_str()));
    }

    /// Java package-private `createPrepend(String)`.
    fn create_prepend(&self, prepend: Option<&str>) -> String {
        create_prepend_static(prepend, &self.base().row_number)
    }

    /// Java `equals(ConstSectionTableRowData)`.
    ///
    /// Fixed in translation: ConstSectionTableRowData.java:252 calls
    /// `joinSection.equals(...)` after testing only the mixed-null cases, a
    /// NullPointerException when both join sections are null.  Two null join sections
    /// compare equal here.
    fn equals(&self, const_section_table_row_data: &dyn ConstSectionTableRowData) -> bool {
        if !self.equals_sample(const_section_table_row_data) {
            return false;
        }
        let base = self.base();
        let that = const_section_table_row_data.base();
        if base.join_section.is_none() && that.join_section.is_some() {
            return false;
        }
        if base.join_section.is_some() && that.join_section.is_none() {
            return false;
        }
        if base.join_section != that.join_section {
            return false;
        }
        if !base
            .setup_final_start
            .equals_const_etomo_number(Some(&that.setup_final_start.base))
        {
            return false;
        }
        if !base
            .setup_final_end
            .equals_const_etomo_number(Some(&that.setup_final_end.base))
        {
            return false;
        }
        if !base
            .join_final_start
            .equals_const_etomo_number(Some(&that.join_final_start.base))
        {
            return false;
        }
        if !base
            .join_final_end
            .equals_const_etomo_number(Some(&that.join_final_end.base))
        {
            return false;
        }
        true
    }

    /// Java `equalsSample(ConstSectionTableRowData)`.
    ///
    /// Fixed in translation: ConstSectionTableRowData.java:281 calls
    /// `setupSection.equals(...)` after testing only the mixed-null cases, a
    /// NullPointerException when both setup sections are null.  Two null setup sections
    /// compare equal here.
    fn equals_sample(&self, const_section_table_row_data: &dyn ConstSectionTableRowData) -> bool {
        let base = self.base();
        let that = const_section_table_row_data.base();
        if !base
            .row_number
            .equals_const_etomo_number(Some(&that.row_number.base))
        {
            return false;
        }
        if base.setup_section.is_none() && that.setup_section.is_some() {
            return false;
        }
        if base.setup_section.is_some() && that.setup_section.is_none() {
            return false;
        }
        if base.setup_section != that.setup_section {
            return false;
        }
        if !base
            .sample_bottom_start
            .equals_const_etomo_number(Some(&that.sample_bottom_start.base))
        {
            return false;
        }
        if !base
            .sample_bottom_end
            .equals_const_etomo_number(Some(&that.sample_bottom_end.base))
        {
            return false;
        }
        if !base
            .sample_top_start
            .equals_const_etomo_number(Some(&that.sample_top_start.base))
        {
            return false;
        }
        if !base
            .sample_top_end
            .equals_const_etomo_number(Some(&that.sample_top_end.base))
        {
            return false;
        }
        if !base
            .rotation_angle_x
            .equals_const_etomo_number(Some(&that.rotation_angle_x.base))
        {
            return false;
        }
        if !base
            .rotation_angle_y
            .equals_const_etomo_number(Some(&that.rotation_angle_y.base))
        {
            return false;
        }
        if !base
            .rotation_angle_z
            .equals_const_etomo_number(Some(&that.rotation_angle_z.base))
        {
            return false;
        }
        true
    }

    /// Java `getInvalidReason`.
    ///
    /// Fixed in translation: ConstSectionTableRowData.java:318 is
    /// `invalidReason.toString()`, and nothing in the source ever sets `invalidReason`
    /// to a non-null value, so the call is always a NullPointerException.  The null
    /// reason is returned as `None` here.
    fn get_invalid_reason(&self) -> Option<String> {
        self.base().invalid_reason.clone()
    }

    /// Java `getRowNumber`.
    fn get_row_number(&self) -> &ConstEtomoNumber {
        &self.base().row_number
    }

    /// Java `getRowIndex`.
    fn get_row_index(&self) -> i32 {
        let base = self.base();
        if base.row_number.get_int() < 0 {
            return -1;
        }
        base.row_number.get_int() - 1
    }

    /// Java `getSetupSection`.
    fn get_setup_section(&self) -> Option<&std::path::Path> {
        self.base().setup_section.as_deref()
    }

    /// Java `getJoinSection`.
    fn get_join_section(&self) -> Option<&std::path::Path> {
        self.base().join_section.as_deref()
    }

    /// Java `getJoinXMax`.
    fn get_join_x_max(&self) -> i32 {
        self.base().join_x_max
    }

    /// Java `getSetupXMax`.
    fn get_setup_x_max(&self) -> i32 {
        self.base().setup_x_max
    }

    /// Java `getJoinYMax`.
    fn get_join_y_max(&self) -> i32 {
        self.base().join_y_max
    }

    /// Java `getSetupYMax`.
    fn get_setup_y_max(&self) -> i32 {
        self.base().setup_y_max
    }

    /// Java `getJoinZMax`.
    fn get_join_z_max(&self) -> i32 {
        self.base().join_z_max
    }

    /// Java `getSetupZMax`.
    fn get_setup_z_max(&self) -> i32 {
        self.base().setup_z_max
    }

    /// Java `getSampleBottomStart`.
    fn get_sample_bottom_start(&self) -> &ConstEtomoNumber {
        &self.base().sample_bottom_start
    }

    /// Java `getSampleBottomEnd`.
    fn get_sample_bottom_end(&self) -> &ConstEtomoNumber {
        &self.base().sample_bottom_end
    }

    /// Java `getSampleTopStart`.
    fn get_sample_top_start(&self) -> &ConstEtomoNumber {
        &self.base().sample_top_start
    }

    /// Java `getSampleTopEnd`.
    fn get_sample_top_end(&self) -> &ConstEtomoNumber {
        &self.base().sample_top_end
    }

    /// Java `getInverted`.
    fn get_inverted(&self) -> &ConstEtomoNumber {
        &self.base().inverted
    }

    /// Java `getSampleTopNumberSlices(int)`.
    fn get_sample_top_number_slices(&self, table_size: i32) -> i32 {
        let base = self.base();
        if base.row_number.equals_int(table_size) || table_size < 2 {
            return -1;
        }
        let sample_top_end = base.sample_top_end.get_int();
        let sample_top_start = base.sample_top_start.get_int();
        if sample_top_end >= sample_top_start {
            return sample_top_end
                .wrapping_sub(sample_top_start)
                .wrapping_add(1);
        }
        0
    }

    /// Java `getSampleBottomNumberSlices(int)`.
    fn get_sample_bottom_number_slices(&self, table_size: i32) -> i32 {
        let base = self.base();
        if base.row_number.equals_int(1) || table_size < 2 {
            return -1;
        }
        let sample_bottom_end = base.sample_bottom_end.get_int();
        let sample_bottom_start = base.sample_bottom_start.get_int();
        if sample_bottom_end >= sample_bottom_start {
            return sample_bottom_end
                .wrapping_sub(sample_bottom_start)
                .wrapping_add(1);
        }
        0
    }

    /// Java `getSetupFinalStart`.
    fn get_setup_final_start(&self) -> &ConstEtomoNumber {
        &self.base().setup_final_start
    }

    /// Java `getSetupFinalEnd`.
    fn get_setup_final_end(&self) -> &ConstEtomoNumber {
        &self.base().setup_final_end
    }

    /// Java `getJoinFinalStart`.
    fn get_join_final_start(&self) -> &ConstEtomoNumber {
        &self.base().join_final_start
    }

    /// Java `getJoinFinalEnd`.
    fn get_join_final_end(&self) -> &ConstEtomoNumber {
        &self.base().join_final_end
    }

    /// Java `isRotated`.
    fn is_rotated(&self) -> bool {
        let base = self.base();
        !base.rotation_angle_x.is_null()
            || !base.rotation_angle_y.is_null()
            || !base.rotation_angle_z.is_null()
    }

    /// Java `getRotationAngleX`.
    fn get_rotation_angle_x(&self) -> &ConstEtomoNumber {
        &self.base().rotation_angle_x
    }

    /// Java `getRotationAngleY`.
    fn get_rotation_angle_y(&self) -> &ConstEtomoNumber {
        &self.base().rotation_angle_y
    }

    /// Java `getRotationAngleZ`.
    fn get_rotation_angle_z(&self) -> &ConstEtomoNumber {
        &self.base().rotation_angle_z
    }
}
