//! `IMOD/Etomo/src/etomo/comscript/TransferfidParam.java`.

use std::collections::BTreeMap;

use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::storage::storable::StorableValue;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::{ConstEtomoNumber, Type};
use crate::imod::etomo::r#type::const_meta_data::ConstMetaData;
use crate::imod::etomo::r#type::etomo_boolean2::EtomoBoolean2;
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::r#type::meta_data::MetaData;
use crate::imod::etomo::r#type::mirror_in_x::MirrorInX;
use crate::imod::etomo::r#type::script_parameter::ScriptParameter;
use crate::imod::etomo::r#type::tilt_angle_spec::TiltAngleSpec;
use crate::imod::etomo::r#type::tilt_angle_type::TiltAngleType;
use crate::imod::etomo::util::dataset_files;
use crate::imod::etomo::util::utilities;

/// Java package-private static `group`.
pub(crate) const GROUP: &str = "Transferfid";
/// Java private static `MIRROR_X_AXIS_TRY_BOTH`.
const MIRROR_X_AXIS_TRY_BOTH: i32 = 0;
/// Java private static `MIRROR_X_AXIS_MIRROR`.
#[allow(dead_code)]
const MIRROR_X_AXIS_MIRROR: i32 = 1;

/// Java final `TransferfidParam implements Storable`.
pub struct TransferfidParam {
    input_image_file: Option<String>,
    output_image_file: Option<String>,
    input_model_file: Option<String>,
    output_model_file: Option<String>,
    dataset_name: Option<String>,
    b_to_a: EtomoBoolean2,
    run_midas: EtomoBoolean2,
    /// null => both, -1 => -90, 1=> +90
    search_direction: EtomoNumber,
    center_view_a: EtomoNumber,
    center_view_b: EtomoNumber,
    number_views: ScriptParameter,
    mirror_xaxis: EtomoNumber,

    meta_data: Option<&'static MetaData>,
    create_log: bool,
    group_string: String,
    manager: &'static ApplicationManager,
}

impl TransferfidParam {
    /// Java `TransferfidParam(ApplicationManager, AxisID)`.
    pub fn new(manager: &'static ApplicationManager, axis_id: AxisID) -> TransferfidParam {
        // MetaData always uses FIRST and SECOND to store, so create groupString with
        // FIRST or SECOND
        let mut axis_id = axis_id;
        if axis_id == AxisID::Only {
            axis_id = AxisID::First;
        }
        let mut param = TransferfidParam {
            input_image_file: None,
            output_image_file: None,
            input_model_file: None,
            output_model_file: None,
            dataset_name: None,
            b_to_a: EtomoBoolean2::new_with_name("BToA"),
            run_midas: EtomoBoolean2::new_with_name("RunMidas"),
            search_direction: EtomoNumber::new_with_type_and_name(Type::Integer, "SearchDirection"),
            center_view_a: EtomoNumber::new_with_name("CenterViewA"),
            center_view_b: EtomoNumber::new_with_name("CenterViewB"),
            number_views: ScriptParameter::new_with_type_and_name(Type::Integer, "NumberViews"),
            mirror_xaxis: EtomoNumber::new_with_name("MirrorXaxis"),
            meta_data: None,
            create_log: false,
            group_string: format!("{GROUP}{}", axis_id.get_extension()),
            manager,
        };
        param.search_direction.set_valid_values(Some(&[-1, 1][..]));
        param.number_views.set_display_value_int(5);
        param
            .mirror_xaxis
            .set_display_value_int(MIRROR_X_AXIS_TRY_BOTH);
        param.reset();
        param
    }

    /// Java private `reset`.
    fn reset(&mut self) {
        self.input_image_file = Some(String::new());
        self.output_image_file = Some(String::new());
        self.input_model_file = Some(String::new());
        self.output_model_file = Some(String::new());
        self.dataset_name = Some(String::new());
        self.b_to_a.reset();
        self.reset_storable_fields();
    }

    /// Java private `resetStorableFields`.  reset fields that are loaded and stored
    fn reset_storable_fields(&mut self) {
        self.run_midas.reset();
        self.search_direction.reset();
        self.center_view_a.reset();
        self.center_view_b.reset();
        self.number_views.reset();
        self.mirror_xaxis.reset();
    }

    /// Java `initialize`.
    pub fn initialize(&mut self) {
        self.set_center_view_a_reset_value();
        self.set_center_view_b_reset_value();
        self.center_view_a.reset();
        self.center_view_b.reset();
    }

    /// Java `setMirrorXaxis(ConstEtomoNumber)`.
    pub fn set_mirror_xaxis(&mut self, input: Option<&ConstEtomoNumber>) {
        self.mirror_xaxis.set_const_etomo_number(input);
    }

    /// Java `getMirrorXaxis`.
    pub fn get_mirror_xaxis(&self) -> MirrorInX {
        MirrorInX::get_instance(Some(&*self.mirror_xaxis))
    }

    /// Java private `setCenterViewAResetValue`.
    fn set_center_view_a_reset_value(&mut self) {
        if self.meta_data.is_none() {
            // Java `manager.getConstMetaData()`.
            self.meta_data = Some(self.manager.get_meta_data());
        }
        let tilt_angle_spec = self.meta_data.unwrap().get_tilt_angle_spec_a();
        Self::set_center_view_reset_value(&mut self.center_view_a, &tilt_angle_spec);
    }

    /// Java private `setCenterViewBResetValue`.
    fn set_center_view_b_reset_value(&mut self) {
        if self.meta_data.is_none() {
            // Java `manager.getConstMetaData()`.
            self.meta_data = Some(self.manager.get_meta_data());
        }
        let tilt_angle_spec = self.meta_data.unwrap().get_tilt_angle_spec_b();
        Self::set_center_view_reset_value(&mut self.center_view_b, &tilt_angle_spec);
    }

    /// Java private `setCenterViewResetValue(EtomoNumber, TiltAngleSpec)`.
    fn set_center_view_reset_value(center_view: &mut EtomoNumber, tilt_angle_spec: &TiltAngleSpec) {
        if tilt_angle_spec.get_type() != TiltAngleType::Range {
            return;
        }
        center_view.set_display_value_long(utilities::java_lang_math_round(
            1.0 - tilt_angle_spec.get_range_min() / tilt_angle_spec.get_range_step(),
        ));
    }

    /// Java `getStorableFields(TransferfidParam)`.  get deep copies of fields that are
    /// loaded and stored
    pub fn get_storable_fields(&self, that: &mut TransferfidParam) {
        that.run_midas
            .set_const_etomo_number(Some(&***self.run_midas));
        that.search_direction
            .set_const_etomo_number(Some(&*self.search_direction));
        that.center_view_a
            .set_const_etomo_number(Some(&*self.center_view_a));
        that.center_view_b
            .set_const_etomo_number(Some(&*self.center_view_b));
        that.number_views
            .set_const_etomo_number(Some(&**self.number_views));
        that.mirror_xaxis
            .set_const_etomo_number(Some(&*self.mirror_xaxis));
    }

    /// Java `setStorableFields(TransferfidParam)`.  set deep copies of fields that are
    /// loaded and stored
    pub fn set_storable_fields(&mut self, that: &TransferfidParam) {
        self.run_midas
            .set_const_etomo_number(Some(&***that.run_midas));
        self.search_direction
            .set_const_etomo_number(Some(&*that.search_direction));
        self.center_view_a
            .set_const_etomo_number(Some(&*that.center_view_a));
        self.center_view_b
            .set_const_etomo_number(Some(&*that.center_view_b));
        self.number_views
            .set_const_etomo_number(Some(&**that.number_views));
        self.mirror_xaxis
            .set_const_etomo_number(Some(&*that.mirror_xaxis));
    }

    /// Java package-private `createPrepend(String)`.  Java tests `prepend == ""`, a
    /// reference comparison that is true only for the interned literal `store(Properties)`
    /// and `load(Properties)` pass; any other empty string would produce a prepend
    /// beginning with ".".  Here every empty prepend takes the first branch, which is
    /// the evident intent.
    pub(crate) fn create_prepend(&self, prepend: &str) -> String {
        if prepend.is_empty() {
            return self.group_string.clone();
        }
        format!("{prepend}.{}", self.group_string)
    }

    /// Java `getCommand`.  Get the command string specified by the current state
    pub fn get_command(&self) -> Vec<String> {
        // Do not use the -e flag for tcsh since David's scripts handle the failure
        // of commands and then report appropriately. The exception to this is the
        // com scripts which require the -e flag. RJG: 2003-11-06
        let mut command: Vec<String> = Vec::new();
        command.push("python".to_owned());
        command.push("-u".to_owned());
        // Java string concatenation writes a null path as "null"
        let script_path = etomo_director::INSTANCE
            .get_python_script_path()
            .as_deref()
            .unwrap_or("null")
            .to_owned();
        command.push(format!("{script_path}transferfid"));
        command.push("-PID".to_owned());

        if self.b_to_a.is() {
            command.push("-b".to_owned());
        }
        // A null file name is a NullPointerException in Java (`inputImageFile.equals("")`);
        // here it is treated as empty and the option is left out.
        if let Some(input_image_file) = self.input_image_file.as_deref()
            && input_image_file != ""
        {
            command.push("-ia".to_owned());
            command.push(input_image_file.to_owned());
        }

        if let Some(output_image_file) = self.output_image_file.as_deref()
            && output_image_file != ""
        {
            command.push("-ib".to_owned());
            command.push(output_image_file.to_owned());
        }

        if let Some(input_model_file) = self.input_model_file.as_deref()
            && input_model_file != ""
        {
            command.push("-f".to_owned());
            command.push(input_model_file.to_owned());
        }

        if let Some(output_model_file) = self.output_model_file.as_deref()
            && output_model_file != ""
        {
            command.push("-o".to_owned());
            command.push(output_model_file.to_owned());
        }

        if !self.center_view_a.is_null() {
            command.push("-za".to_owned());
            command.push(self.center_view_a.to_string());
        }

        if !self.center_view_b.is_null() {
            command.push("-zb".to_owned());
            command.push(self.center_view_b.to_string());
        }

        if self.number_views.is_not_null_and_not_default() {
            command.push("-n".to_owned());
            command.push(self.number_views.to_string());
        }

        if self.search_direction.is_positive() {
            command.push("-a".to_owned());
            command.push("90".to_owned());
        }

        if self.search_direction.is_negative() {
            command.push("-a".to_owned());
            command.push("-90".to_owned());
        }

        command.push("-x".to_owned());
        command.push(self.mirror_xaxis.to_string());

        if self.run_midas.is() {
            command.push("-m".to_owned());
        }
        command.push("-c".to_owned());
        command.push(dataset_files::get_transfer_fid_coord_file_name());

        // A null dataset name is a null list element in Java, which `ProcessBuilder`
        // rejects; here it is left out.
        if let Some(dataset_name) = &self.dataset_name {
            command.push(dataset_name.clone());
        }
        command
    }

    /// Java `getInputImageFile`.
    pub fn get_input_image_file(&self) -> Option<&str> {
        self.input_image_file.as_deref()
    }

    /// Java `getInputModelFile`.
    pub fn get_input_model_file(&self) -> Option<&str> {
        self.input_model_file.as_deref()
    }

    /// Java `getOutputImageFile`.
    pub fn get_output_image_file(&self) -> Option<&str> {
        self.output_image_file.as_deref()
    }

    /// Java `getOutputModelFile`.
    pub fn get_output_model_file(&self) -> Option<&str> {
        self.output_model_file.as_deref()
    }

    /// Java `getRunMidas`.
    pub fn get_run_midas(&self) -> &ConstEtomoNumber {
        &self.run_midas
    }

    /// Java `getNumberViews`.
    pub fn get_number_views(&self) -> &ConstEtomoNumber {
        &self.number_views
    }

    /// Java `setNumberViews(String)`.
    pub fn set_number_views(&mut self, number_views: Option<&str>) {
        self.number_views.set_string(number_views);
    }

    /// Java `setInputImageFile(String)`.
    pub fn set_input_image_file(&mut self, input_image_file: Option<&str>) {
        self.input_image_file = input_image_file.map(str::to_owned);
    }

    /// Java `setInputModelFile(String)`.
    pub fn set_input_model_file(&mut self, input_model_file: Option<&str>) {
        self.input_model_file = input_model_file.map(str::to_owned);
    }

    /// Java `setOutputImageFile(String)`.
    pub fn set_output_image_file(&mut self, output_image_file: Option<&str>) {
        self.output_image_file = output_image_file.map(str::to_owned);
    }

    /// Java `setOutputModelFile(String)`.
    pub fn set_output_model_file(&mut self, output_model_file: Option<&str>) {
        self.output_model_file = output_model_file.map(str::to_owned);
    }

    /// Java `setRunMidas(boolean)`.
    pub fn set_run_midas(&mut self, run_midas: bool) {
        self.run_midas.set_boolean(run_midas);
    }

    /// Java `getDatasetName`.
    pub fn get_dataset_name(&self) -> Option<&str> {
        self.dataset_name.as_deref()
    }

    /// Java `setDatasetName(String)`.
    pub fn set_dataset_name(&mut self, dataset_name: Option<&str>) {
        self.dataset_name = dataset_name.map(str::to_owned);
    }

    /// Java `getBToA`.
    pub fn get_b_to_a(&self) -> &ConstEtomoNumber {
        &self.b_to_a
    }

    /// Java `setBToA(boolean)`.
    pub fn set_b_to_a(&mut self, b_to_a: bool) {
        self.b_to_a.set_boolean(b_to_a);
    }

    /// Java `getCenterViewA`.
    pub fn get_center_view_a(&self) -> &EtomoNumber {
        &self.center_view_a
    }

    /// Java `getCenterViewB`.
    pub fn get_center_view_b(&self) -> &EtomoNumber {
        &self.center_view_b
    }

    /// Java `getSearchDirection`.
    pub fn get_search_direction(&self) -> &EtomoNumber {
        &self.search_direction
    }

    /// Java `getSearchDirection()` where the caller changes the returned number.
    pub fn get_search_direction_mut(&mut self) -> &mut EtomoNumber {
        &mut self.search_direction
    }

    /// Java `setCenterViewA(String)`.
    pub fn set_center_view_a(&mut self, center_view_a: Option<&str>) {
        self.center_view_a.set_string(center_view_a);
    }

    /// Java `setCenterViewB(String)`.
    pub fn set_center_view_b(&mut self, center_view_b: Option<&str>) {
        self.center_view_b.set_string(center_view_b);
    }

    /// Java `setSearchDirection(int)`.
    pub fn set_search_direction(&mut self, search_direction: i32) {
        self.search_direction.set_int(search_direction);
    }

    /// Java `isCreateLog`.
    pub fn is_create_log(&self) -> bool {
        self.create_log
    }

    /// Java `setCreateLog(boolean)`.
    pub fn set_create_log(&mut self, create_log: bool) {
        self.create_log = create_log;
    }
}

impl StorableValue for TransferfidParam {
    /// Java `store(Properties)`.
    fn store(&self, props: &mut BTreeMap<String, String>) {
        self.store_with_prepend(props, "");
    }

    /// Java `store(Properties, String)`.
    fn store_with_prepend(&self, props: &mut BTreeMap<String, String>, prepend: &str) {
        let prepend = self.create_prepend(prepend);
        self.run_midas
            .store_with_prepend(props, Some(prepend.as_str()));
        self.search_direction.store_with_prepend(props, &prepend);
        self.center_view_a.store_with_prepend(props, &prepend);
        self.center_view_b.store_with_prepend(props, &prepend);
        self.number_views.store_with_prepend(props, &prepend);
        self.mirror_xaxis.store_with_prepend(props, &prepend);
    }

    /// Java `load(Properties)`.
    fn load(&mut self, props: &mut BTreeMap<String, String>) {
        self.load_with_prepend(props, "");
    }

    /// Java `load(Properties, String)`.
    fn load_with_prepend(&mut self, props: &mut BTreeMap<String, String>, prepend: &str) {
        self.reset_storable_fields();
        let prepend = self.create_prepend(prepend);
        let prepend = Some(prepend.as_str());

        self.run_midas.load_with_prepend(props, prepend);
        self.search_direction.load_with_prepend(props, prepend);
        self.center_view_a.load_with_prepend(props, prepend);
        self.center_view_b.load_with_prepend(props, prepend);
        self.number_views.load_with_prepend(props, prepend);
        self.mirror_xaxis.load_with_prepend(props, prepend);
    }
}
