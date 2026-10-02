//! `IMOD/Etomo/src/etomo/type/SectionTableRowData.java`.
//!
//! Data from SectionTableRow.  Integrated with SectionTableRow.  Can also be stored by
//! JoinMetaData.
//!
//! Version Log:
//! 1.1:  Converted FinalStart and FinalEnd from integers to longs.  The integer null
//! value from version 1.0 will have to be recongnized and changed to a long null value.
//! Stopped saving setupXMax, setupYMax, and setupZMax.  They should come from the header
//! of setupSection.
//!
//! The superclass state is [`ConstSectionTableRowDataBase`], held in `base`; the
//! superclass's concrete methods come from the [`ConstSectionTableRowData`] trait.
#![allow(dead_code)]

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use super::axis_id::AxisID;
use super::const_etomo_number::{ConstEtomoNumber, INTEGER_NULL_VALUE};
use super::const_section_table_row_data::{
    COS_X_Y_THRESHOLD, ConstSectionTableRowData, ConstSectionTableRowDataBase,
    SETUP_SECTION_STRING, SETUP_X_MAX_STRING, SETUP_Y_MAX_STRING, SETUP_Z_MAX_STRING, VERSION,
    VERSION_KEY,
};
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::storage::storable::StorableValue;
use crate::imod::etomo::ui::swing::ui_harness;
use crate::imod::etomo::util::dataset_files;
use crate::imod::etomo::util::mrc_header::MRCHeader;
use crate::imod::etomo::util::utilities::{java_io_file_get_absolute_path, java_lang_math_round};

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `SectionTableRowData extends ConstSectionTableRowData`.
pub struct SectionTableRowData {
    /// Java superclass `ConstSectionTableRowData` state.
    pub base: ConstSectionTableRowDataBase,
    /// Java field `manager`.
    manager: &'static dyn BaseManager,
}

impl ConstSectionTableRowData for SectionTableRowData {
    fn base(&self) -> &ConstSectionTableRowDataBase {
        &self.base
    }
}

/// Java `toString`: `getClass().getName() + "[" + paramString() + "]"`.
impl std::fmt::Display for SectionTableRowData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "etomo.type.SectionTableRowData[{}]",
            ConstSectionTableRowData::param_string(self)
        )
    }
}

impl SectionTableRowData {
    /// Java `SectionTableRowData(BaseManager, int)`.  Construct an empty instance.  Must
    /// be passed a row number.
    pub fn new(manager: &'static dyn BaseManager, row_number: i32) -> SectionTableRowData {
        let mut instance = SectionTableRowData {
            base: ConstSectionTableRowDataBase::new(row_number),
            manager,
        };
        instance.reset();
        instance
    }

    /// Java `SectionTableRowData(BaseManager, ConstSectionTableRowData)`.  Construct an
    /// instance from ConstSectionTableRowData.
    pub fn new_from(
        manager: &'static dyn BaseManager,
        const_section_table_row_data: &dyn ConstSectionTableRowData,
    ) -> SectionTableRowData {
        SectionTableRowData {
            base: ConstSectionTableRowDataBase::new_from(const_section_table_row_data.base()),
            manager,
        }
    }

    /// Java `load(Properties)`.  Get the objects attributes from the properties object.
    pub fn load(&mut self, props: &BTreeMap<String, String>) {
        self.load_with_prepend(props, Some(""));
    }

    /// Java `load(Properties, String)`.
    ///
    /// Fixed in translation: SectionTableRowData.java:88 reads the stored version with
    /// `props.getProperty(VERSION_KEY)`, without the row's group prefix that `store`
    /// writes it under, so the version always read back as null and every load ran the
    /// 1.0 conversion.  The version is read from `group + VERSION_KEY` here.
    pub fn load_with_prepend(&mut self, props: &BTreeMap<String, String>, prepend: Option<&str>) {
        self.reset();
        let prepend = ConstSectionTableRowData::create_prepend(self, prepend);
        let group = format!("{}.", prepend);
        let stored_version = props.get(&format!("{}{}", group, VERSION_KEY)).cloned();
        self.base
            .row_number
            .load_with_prepend(props, Some(prepend.as_str()));
        let section_name = props
            .get(&format!("{}{}", group, SETUP_SECTION_STRING))
            .cloned();
        if let Some(section_name) = section_name {
            self.set_setup_section(&PathBuf::from(section_name));
        }
        self.base
            .sample_bottom_start
            .load_with_prepend(props, Some(prepend.as_str()));
        self.base
            .sample_bottom_end
            .load_with_prepend(props, Some(prepend.as_str()));
        self.base
            .sample_top_start
            .load_with_prepend(props, Some(prepend.as_str()));
        self.base
            .sample_top_end
            .load_with_prepend(props, Some(prepend.as_str()));
        self.base
            .setup_final_start
            .load_with_prepend(props, Some(prepend.as_str()));
        self.base
            .setup_final_end
            .load_with_prepend(props, Some(prepend.as_str()));
        self.base
            .rotation_angle_x
            .load_with_prepend(props, Some(prepend.as_str()));
        self.base
            .rotation_angle_y
            .load_with_prepend(props, Some(prepend.as_str()));
        self.base
            .rotation_angle_z
            .load_with_prepend(props, Some(prepend.as_str()));
        self.base
            .inverted
            .load_with_prepend(props, Some(prepend.as_str()));
        if stored_version.as_deref() != Some(VERSION) {
            self.convert_version(stored_version.as_deref(), props, &prepend);
        }
    }

    /// Java private `convertVersion`.  Convert stored version to current version.
    ///
    /// Deviation: the source removes the obsolete 1.0 `XMax`/`YMax`/`ZMax` keys from the
    /// `Properties` it was loaded from.  `Storable::load` takes the map shared, so those
    /// removals are not made; they only drop keys a 1.1 `store` never writes, and the
    /// next save of the file (which rebuilds it from `store`) drops them anyway.
    fn convert_version(
        &mut self,
        stored_version: Option<&str>,
        props: &BTreeMap<String, String>,
        prepend: &str,
    ) {
        if stored_version.is_none() {
            // convert from version 1.0 to 1.1
            if self.base.setup_final_start.equals_int(INTEGER_NULL_VALUE) {
                self.base.setup_final_start.reset();
            }
            if self.base.setup_final_end.equals_int(INTEGER_NULL_VALUE) {
                self.base.setup_final_end.reset();
            }
            // `props.remove(group + setupXMaxString)` and the Y and Z twins: `load`
            // receives the map shared (`Storable::load`), so these removals of the
            // obsolete 1.0 keys cannot be made here.  See the doc comment.
            let _ = (
                props,
                prepend,
                SETUP_X_MAX_STRING,
                SETUP_Y_MAX_STRING,
                SETUP_Z_MAX_STRING,
            );
        }
    }

    /// Java `synchronizeSetupToJoin`.  Check for a .rot file; if the .rot file exists
    /// then convert the setup data into join data, otherwise copy the setup data to join
    /// data.  Assumes setupSection is set.
    ///
    /// Fixed in translation: SectionTableRowData.java:137 passes `setupSection` to
    /// `DatasetFiles.getRotatedTomogram`, a NullPointerException when it is not set.
    /// With no setup section there is no rotated tomogram, so the setup data is copied.
    pub fn synchronize_setup_to_join(&mut self) {
        let rotated_section =
            self.base.setup_section.clone().map(|setup_section| {
                dataset_files::get_rotated_tomogram(self.manager, &setup_section)
            });
        match rotated_section {
            Some(rotated_section)
                if rotated_section.exists()
                    && (!self.base.rotation_angle_x.is_null()
                        || !self.base.rotation_angle_y.is_null()
                        || !self.base.rotation_angle_z.is_null()) =>
            {
                self.convert_setup_to_join(&rotated_section);
            }
            _ => {
                self.copy_setup_to_join();
            }
        }
        // If inverted: start must be greater then end
        // If not inverted: end must be greater then start
        if (self.base.inverted.is()
            && self
                .base
                .join_final_end
                .gt_const_etomo_number(Some(&self.base.join_final_start.base)))
            || (!self.base.inverted.is()
                && self
                    .base
                    .join_final_start
                    .gt_const_etomo_number(Some(&self.base.join_final_end.base)))
        {
            let temp = self.base.join_final_end.get_int();
            self.base
                .join_final_end
                .set_const_etomo_number(Some(&self.base.join_final_start.base));
            self.base.join_final_start.set_int(temp);
        }
    }

    /// Java `synchronizeJoinToSetup`.  Check whether the join section is a .rot file; if
    /// it is, convert the join data into setup data, if it is not then copy the join data
    /// to setup data.  Assumes joinSection is set.
    ///
    /// Fixed in translation: SectionTableRowData.java:161 passes `joinSection` to
    /// `DatasetFiles.isRotatedTomogram`, a NullPointerException when it is not set.  With
    /// no join section it is not a rotated tomogram, so the join data is copied.
    pub fn synchronize_join_to_setup(&mut self) {
        let is_rotated = match &self.base.join_section {
            None => false,
            Some(join_section) => dataset_files::is_rotated_tomogram(join_section),
        };
        if is_rotated {
            self.convert_join_to_setup();
        } else {
            self.copy_join_to_setup();
        }
    }

    /// Java private `copySetupToJoin`.  Copies Setup tab variables to Join tab
    /// variables.
    fn copy_setup_to_join(&mut self) {
        self.base.join_section = self.base.setup_section.clone();
        self.base
            .join_final_start
            .set_const_etomo_number(Some(&self.base.setup_final_start.base));
        self.base
            .join_final_end
            .set_const_etomo_number(Some(&self.base.setup_final_end.base));
        self.base.join_x_max = self.base.setup_x_max;
        self.base.join_y_max = self.base.setup_y_max;
        self.base.join_z_max = self.base.setup_z_max;
    }

    /// Java private `convertSetupToJoin(File)`.  Attempts to set the Join tab section
    /// and max variables to a rotated tomogram.  Converts final start and end values from
    /// the Setup tab to their equivalents in the rotated tomogram.
    fn convert_setup_to_join(&mut self, rotated_section: &Path) {
        if !self.set_join_section(rotated_section) {
            return;
        }
        let start = self.convert_to_rotated_z(&self.base.setup_final_start.base);
        self.base.join_final_start.set_int(start);
        let end = self.convert_to_rotated_z(&self.base.setup_final_end.base);
        self.base.join_final_end.set_int(end);
    }

    /// Java private `convertJoinToSetup`.  Converts final start and end values from a
    /// rotated tomogram in the Join tab to their equivalents in the original tomogram.
    fn convert_join_to_setup(&mut self) {
        let start = self.convert_from_rotated_z(&self.base.join_final_start.base);
        self.base.setup_final_start.set_int(start);
        let end = self.convert_from_rotated_z(&self.base.join_final_end.base);
        self.base.setup_final_end.set_int(end);
    }

    /// Java private `copyJoinToSetup`.  Copies Join tab final start and end to Setup tab
    /// variables.
    fn copy_join_to_setup(&mut self) {
        self.base
            .setup_final_start
            .set_const_etomo_number(Some(&self.base.join_final_start.base));
        self.base
            .setup_final_end
            .set_const_etomo_number(Some(&self.base.join_final_end.base));
    }

    /// Java private `convertToRotatedZ(ConstEtomoNumber)`.  Converts final start and end
    /// from the original tomogram to the rotated tomogram.  Compute as floating point
    /// then round to nearest integer - i.e. do not do integer arithmetic with
    /// (zsize + 1) / 2.  Allowing for a different Z size for the rec and the rot, the
    /// formula to get from rec slice to rot slice is:
    /// cos(X angle) * cos(Y angle) * (slice - (Zsize_rec + 1) / 2) + (Zsize_rot + 1) / 2
    /// If cos(X angle) * cos(Y angle) is less then COS_X_Y_THRESHOLD, do not convert z.
    fn convert_to_rotated_z(&self, z: &ConstEtomoNumber) -> i32 {
        let cos_xy = self
            .base
            .rotation_angle_x
            .get_defaulted_double()
            .to_radians()
            .cos()
            * self
                .base
                .rotation_angle_y
                .get_defaulted_double()
                .to_radians()
                .cos();
        if cos_xy.abs() <= COS_X_Y_THRESHOLD {
            return z.get_int();
        }
        let z_slice = z.get_double();
        let z_size = self.base.setup_z_max as f64;
        let z_size_rotated = self.base.join_z_max as f64;
        let converted_z = cos_xy * (z_slice - (z_size + 1.) / 2.) + (z_size_rotated + 1.) / 2.;
        java_lang_math_round(converted_z) as i32
    }

    /// Java private `convertFromRotatedZ(ConstEtomoNumber)`.  Converts final start and
    /// end from the rotated tomogram to the original tomogram.  The formula to get from
    /// rot slice to rec slice is:
    /// (slice - (Zsize_rot + 1) / 2) / (cos(X angle) * cos(Y angle)) + (Zsize_rec + 1) / 2
    /// If cos(X angle) * cos(Y angle) is less then COS_X_Y_THRESHOLD, do not convert z.
    fn convert_from_rotated_z(&self, z: &ConstEtomoNumber) -> i32 {
        let cos_xy = self
            .base
            .rotation_angle_x
            .get_defaulted_double()
            .to_radians()
            .cos()
            * self
                .base
                .rotation_angle_y
                .get_defaulted_double()
                .to_radians()
                .cos();
        if cos_xy.abs() <= COS_X_Y_THRESHOLD {
            return z.get_int();
        }
        let z_slice = z.get_double();
        let z_size = self.base.setup_z_max as f64;
        let z_size_rotated = self.base.join_z_max as f64;
        let converted_z = (z_slice - (z_size_rotated + 1.) / 2.) / cos_xy + (z_size + 1.) / 2.;
        java_lang_math_round(converted_z) as i32
    }

    /// Java `setRowNumber(int)`.
    pub fn set_row_number(&mut self, row_number: i32) {
        self.base.row_number.set_int(row_number);
    }

    /// Java private `reset`.  Resets the member variables, except the row number.
    fn reset(&mut self) {
        self.base.invalid_reason = None;
        self.base.setup_section = None;
        self.base.join_section = None;
        self.base.sample_bottom_start.reset();
        self.base.sample_bottom_end.reset();
        self.base.sample_top_start.reset();
        self.base.sample_top_end.reset();
        self.base.setup_final_start.reset();
        self.base.setup_final_end.reset();
        self.base.join_final_start.reset();
        self.base.join_final_end.reset();
        self.base.rotation_angle_x.reset();
        self.base.rotation_angle_y.reset();
        self.base.rotation_angle_z.reset();
        self.base.setup_x_max = INTEGER_NULL_VALUE;
        self.base.setup_y_max = INTEGER_NULL_VALUE;
        self.base.setup_z_max = INTEGER_NULL_VALUE;
        self.base.join_x_max = INTEGER_NULL_VALUE;
        self.base.join_y_max = INTEGER_NULL_VALUE;
        self.base.join_z_max = INTEGER_NULL_VALUE;
    }

    /// Java `setJoinSection(File)`.  Set join section.  Also read the join section
    /// header and sets joinXMax, joinYMax, and joinZMax.
    pub fn set_join_section(&mut self, join_section: &Path) -> bool {
        self.base.join_section = Some(join_section.to_path_buf());
        let header = self.read_header(&java_io_file_get_absolute_path(
            &join_section.to_string_lossy(),
        ));
        let header = match header {
            None => return false,
            Some(header) => header,
        };
        let header = header.borrow();
        self.base.join_x_max = header.get_n_columns();
        self.base.join_y_max = header.get_n_rows();
        self.base.join_z_max = header.get_n_sections();
        true
    }

    /// Java `setSetupSection(File)`.  Set setup section.  Also read the setup section
    /// header and sets setupXMax, setupYMax, and setupZMax.
    pub fn set_setup_section(&mut self, setup_section: &Path) -> bool {
        self.base.setup_section = Some(setup_section.to_path_buf());
        let header = self.read_header(&java_io_file_get_absolute_path(
            &setup_section.to_string_lossy(),
        ));
        let header = match header {
            None => return false,
            Some(header) => header,
        };
        let header = header.borrow();
        self.base.setup_x_max = header.get_n_columns();
        self.base.setup_y_max = header.get_n_rows();
        self.base.setup_z_max = header.get_n_sections();
        self.base
            .setup_final_end
            .set_display_value_int(self.base.setup_z_max);
        true
    }

    /// Java private `readHeader(String)`.  Reads an mrc header and pops up error
    /// messages if there is any kind of failure.
    ///
    /// `MRCHeader.read` throws `InvalidParameterException` or `IOException`, which
    /// [`MRCHeader::read_with_manager`] reports as one message; the source's two catch
    /// blocks differ only in the exception name they print, and the popup here names
    /// neither.
    fn read_header(&self, path: &str) -> Option<Rc<RefCell<MRCHeader>>> {
        let header = MRCHeader::get_instance_in_dir(
            self.manager.get_property_user_dir().as_deref(),
            Some(path),
            Some(AxisID::Only),
        )?;
        let result = header.borrow_mut().read_with_manager(self.manager);
        match result {
            Ok(false) => {
                ui_harness::post_message_dialog(
                    Some(self.manager),
                    format!("Unable to read the header in{}", path),
                    "Setting Max Values Failed".to_string(),
                    None,
                );
                return None;
            }
            Ok(true) => {}
            Err(e) => {
                eprintln!("{}", e);
                ui_harness::post_message_dialog(
                    Some(self.manager),
                    format!("Unable to read the header in{}.\n{}", path, e),
                    "Setting Max Values Failed".to_string(),
                    None,
                );
                return None;
            }
        }
        Some(header)
    }

    /// Java `setInverted(boolean)`.
    pub fn set_inverted(&mut self, inverted: bool) {
        self.base.inverted.set_boolean(inverted);
    }

    /// Java `setSampleBottomStart(String)`.
    pub fn set_sample_bottom_start(
        &mut self,
        sample_bottom_start: Option<&str>,
    ) -> &ConstEtomoNumber {
        self.base
            .sample_bottom_start
            .set_string(sample_bottom_start);
        &self.base.sample_bottom_start
    }

    /// Java `setSampleBottomEnd(String)`.
    pub fn set_sample_bottom_end(&mut self, sample_bottom_end: Option<&str>) -> &ConstEtomoNumber {
        self.base.sample_bottom_end.set_string(sample_bottom_end);
        &self.base.sample_bottom_end
    }

    /// Java `setSampleTopStart(String)`.
    pub fn set_sample_top_start(&mut self, sample_top_start: Option<&str>) -> &ConstEtomoNumber {
        self.base.sample_top_start.set_string(sample_top_start);
        &self.base.sample_top_start
    }

    /// Java `setSampleTopEnd(String)`.
    pub fn set_sample_top_end(&mut self, sample_top_end: Option<&str>) -> &ConstEtomoNumber {
        self.base.sample_top_end.set_string(sample_top_end);
        &self.base.sample_top_end
    }

    /// Java `setSetupFinalStart(String)`.
    pub fn set_setup_final_start(&mut self, setup_final_start: Option<&str>) -> &ConstEtomoNumber {
        self.base.setup_final_start.set_string(setup_final_start);
        &self.base.setup_final_start
    }

    /// Java `setJoinFinalStart(String)`.
    pub fn set_join_final_start(&mut self, join_final_start: Option<&str>) -> &ConstEtomoNumber {
        self.base.join_final_start.set_string(join_final_start);
        &self.base.join_final_start
    }

    /// Java `setSetupFinalEnd(String)`.
    pub fn set_setup_final_end(&mut self, setup_final_end: Option<&str>) -> &ConstEtomoNumber {
        self.base.setup_final_end.set_string(setup_final_end);
        &self.base.setup_final_end
    }

    /// Java `setJoinFinalEnd(String)`.
    pub fn set_join_final_end(&mut self, join_final_end: Option<&str>) -> &ConstEtomoNumber {
        self.base.join_final_end.set_string(join_final_end);
        &self.base.join_final_end
    }

    /// Java `setRotationAngleX(String)`.
    pub fn set_rotation_angle_x(&mut self, rotation_angle_x: Option<&str>) -> &ConstEtomoNumber {
        self.base.rotation_angle_x.set_string(rotation_angle_x);
        &self.base.rotation_angle_x
    }

    /// Java `setRotationAngleY(String)`.
    pub fn set_rotation_angle_y(&mut self, rotation_angle_y: Option<&str>) -> &ConstEtomoNumber {
        self.base.rotation_angle_y.set_string(rotation_angle_y);
        &self.base.rotation_angle_y
    }

    /// Java `setRotationAngleZ(String)`.
    pub fn set_rotation_angle_z(&mut self, rotation_angle_z: Option<&str>) -> &ConstEtomoNumber {
        self.base.rotation_angle_z.set_string(rotation_angle_z);
        &self.base.rotation_angle_z
    }
}

/// Java `ConstSectionTableRowData implements Storable`: `store` comes from the
/// superclass, `load` from this class.
impl StorableValue for SectionTableRowData {
    fn store(&self, properties: &mut BTreeMap<String, String>) {
        ConstSectionTableRowData::store(self, properties);
    }

    fn store_with_prepend(&self, properties: &mut BTreeMap<String, String>, prepend: &str) {
        ConstSectionTableRowData::store_with_prepend(self, properties, Some(prepend));
    }

    fn load(&mut self, properties: &BTreeMap<String, String>) {
        SectionTableRowData::load(self, properties);
    }

    fn load_with_prepend(&mut self, properties: &BTreeMap<String, String>, prepend: &str) {
        SectionTableRowData::load_with_prepend(self, properties, Some(prepend));
    }
}
