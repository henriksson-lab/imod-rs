//! `IMOD/Etomo/src/etomo/process/BaseImodManager.java`.
//!
//! Source-shaped 3dmod registry/launch boundary. `ImodState`, `ImodRequestHandler`,
//! 3dmod IPC and Swing notification are intentionally represented only by Java-null
//! equivalents until their source units are translated; this module never fakes a
//! 3dmod process.
#![allow(dead_code)]

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::file_type::FileType;
use std::collections::HashMap;
use std::convert::Infallible;
use std::path::Path;
use std::sync::Mutex;

pub const DEFAULT_BEADFIXER_DIAMETER: i32 = 3;

/// Java abstract `BaseImodManager`.
pub struct BaseImodManager {
    axis_type: Mutex<AxisType>,
    use_map: Mutex<bool>,
    debug: Mutex<bool>,
    /// Java `HashMap<String, Vector<ImodState>>`.
    imod_map: Mutex<HashMap<String, Vec<Option<Infallible>>>>,
    manager: Option<&'static dyn BaseManager>,
    /// Java nullable `ImodRequestHandler`.
    request_handler: Option<Infallible>,
}

impl BaseImodManager {
    /// Java protected constructor.
    pub fn new(manager: Option<&'static dyn BaseManager>) -> Self {
        Self {
            axis_type: Mutex::new(AxisType::SingleAxis),
            use_map: Mutex::new(true),
            debug: Mutex::new(false),
            imod_map: Mutex::new(HashMap::new()),
            manager,
            request_handler: None,
        }
    }
    /// Java protected `newImodState(String,String,AxisID,String,File,String[],String,File[])`.
    // TODO(unit): ImodState.java.
    pub fn new_imod_state(
        &self,
        key: Option<&str>,
        file_extension: Option<&str>,
        axis_id: Option<AxisID>,
        dataset_name: Option<&str>,
        file: Option<&Path>,
        file_name_array: Option<&[Option<&str>]>,
        subdir_name: Option<&str>,
        file_list: Option<&[Option<&Path>]>,
    ) -> Option<Infallible> {
        let _ = (
            key,
            file_extension,
            axis_id,
            dataset_name,
            file,
            file_name_array,
            subdir_name,
            file_list,
        );
        None
    }
    /// Java `getPrivateKey`.
    pub fn get_private_key<'a>(&self, public_key: Option<&'a str>) -> Option<&'a str> {
        public_key
    }
    /// Java `isDualAxisOnly`.
    pub fn is_dual_axis_only(&self, key: Option<&str>) -> bool {
        let _ = key;
        false
    }
    /// Java `isPerAxis`.
    pub fn is_per_axis(&self, key: Option<&str>) -> bool {
        let _ = key;
        true
    }
    pub fn set_axis_type(&self, axis_type: AxisType) {
        *self.axis_type.lock().unwrap() = axis_type;
    }
    pub fn equals_axis_type(&self, input: AxisType) -> bool {
        *self.axis_type.lock().unwrap() == input
    }
    pub fn get_axis_type_string(&self) -> String {
        self.axis_type.lock().unwrap().to_string()
    }

    // Java `newImod` overloads.  Suffixes replace Java overload resolution.
    fn new_imod(&self, key: Option<&str>) -> i32 {
        let _ = key;
        0
    }
    pub fn new_imod_axis(&self, key: Option<&str>, axis_id: Option<AxisID>) -> i32 {
        let _ = (key, axis_id);
        0
    }
    pub fn new_imod_extension_axis(
        &self,
        key: Option<&str>,
        file_extension: Option<&str>,
        axis_id: Option<AxisID>,
    ) -> i32 {
        let _ = (key, file_extension, axis_id);
        0
    }
    pub fn new_imod_axis_dataset(
        &self,
        key: Option<&str>,
        axis_id: Option<AxisID>,
        dataset_name: Option<&str>,
    ) -> i32 {
        let _ = (key, axis_id, dataset_name);
        0
    }
    fn new_imod_extension_axis_dataset(
        &self,
        key: Option<&str>,
        file_extension: Option<&str>,
        axis_id: Option<AxisID>,
        dataset_name: Option<&str>,
    ) -> i32 {
        let _ = (key, file_extension, axis_id, dataset_name);
        0
    }
    pub fn new_imod_axis_file_list(
        &self,
        key: Option<&str>,
        axis_id: Option<AxisID>,
        file_list: Option<&[Option<&Path>]>,
    ) -> i32 {
        let _ = (key, axis_id, file_list);
        0
    }
    pub fn new_imod_file(&self, key: Option<&str>, file: Option<&Path>) -> i32 {
        let _ = (key, file);
        0
    }
    pub fn update_imod(&self, key: Option<&str>, index: i32, file: Option<&Path>) {
        let _ = (key, index, file);
    }
    fn new_imod_axis_file(
        &self,
        key: Option<&str>,
        axis_id: Option<AxisID>,
        file: Option<&Path>,
    ) -> i32 {
        let _ = (key, axis_id, file);
        0
    }
    fn create_imod(
        &self,
        key: Option<&str>,
        axis_id: Option<AxisID>,
        file: Option<&Path>,
    ) -> Option<Infallible> {
        let _ = (key, axis_id, file);
        None
    }
    fn new_imod_file_name_array(
        &self,
        key: Option<&str>,
        file_name_array: Option<&[Option<&str>]>,
    ) -> i32 {
        let _ = (key, file_name_array);
        0
    }
    fn new_imod_file_name_array_subdir(
        &self,
        key: Option<&str>,
        file_name_array: Option<&[Option<&str>]>,
        subdir_name: Option<&str>,
    ) -> i32 {
        let _ = (key, file_name_array, subdir_name);
        0
    }

    // Java `open` overloads.
    pub fn open(&self, key: Option<&str>) {
        let _ = key;
    }
    pub fn open_options(&self, key: Option<&str>, menu_options: Option<Infallible>) {
        let _ = (key, menu_options);
    }
    pub fn open_model_options(
        &self,
        key: Option<&str>,
        model: Option<&str>,
        menu_options: Option<Infallible>,
    ) {
        let _ = (key, model, menu_options);
    }
    pub fn open_axis_options(
        &self,
        key: Option<&str>,
        axis_id: Option<AxisID>,
        menu_options: Option<Infallible>,
    ) {
        let _ = (key, axis_id, menu_options);
    }
    pub fn open_axis_model(&self, key: Option<&str>, axis_id: Option<AxisID>, model: Option<&str>) {
        let _ = (key, axis_id, model);
    }
    pub fn open_axis(&self, key: Option<&str>, axis_id: Option<AxisID>) {
        let _ = (key, axis_id);
    }
    pub fn open_file_options(
        &self,
        key: Option<&str>,
        file: Option<&Path>,
        menu_options: Option<Infallible>,
    ) {
        let _ = (key, file, menu_options);
    }
    pub fn open_axis_file_options(
        &self,
        key: Option<&str>,
        axis_id: Option<AxisID>,
        file: Option<&Path>,
        menu_options: Option<Infallible>,
    ) {
        let _ = (key, axis_id, file, menu_options);
    }
    pub fn open_file_options_swap_yz(
        &self,
        key: Option<&str>,
        file: Option<&Path>,
        menu_options: Option<Infallible>,
        swap_yz: bool,
    ) {
        let _ = (key, file, menu_options, swap_yz);
    }
    pub fn open_axis_model_options(
        &self,
        key: Option<&str>,
        axis_id: Option<AxisID>,
        model: Option<&str>,
        menu_options: Option<Infallible>,
    ) {
        let _ = (key, axis_id, model, menu_options);
    }
    pub fn open_axis_model_list_options(
        &self,
        key: Option<&str>,
        axis_id: Option<AxisID>,
        model_list: Option<&[Option<&str>]>,
        menu_options: Option<Infallible>,
    ) {
        let _ = (key, axis_id, model_list, menu_options);
    }
    pub fn set_open_model_view(&self, key: Option<&str>, axis_id: Option<AxisID>) {
        let _ = (key, axis_id);
    }
    pub fn open_file_name_array_options(
        &self,
        key: Option<&str>,
        file_name_array: Option<&[Option<&str>]>,
        menu_options: Option<Infallible>,
    ) {
        let _ = (key, file_name_array, menu_options);
    }
    pub fn open_file_name_array_options_subdir_swap_yz(
        &self,
        key: Option<&str>,
        file_name_array: Option<&[Option<&str>]>,
        menu_options: Option<Infallible>,
        subdir_name: Option<&str>,
        swap_yz: bool,
    ) {
        let _ = (key, file_name_array, menu_options, subdir_name, swap_yz);
    }
    pub fn open_axis_file_name_array_options_subdir(
        &self,
        axis_id: Option<AxisID>,
        key: Option<&str>,
        file_name_array: Option<&[Option<&str>]>,
        menu_options: Option<Infallible>,
        subdir_name: Option<&str>,
    ) {
        let _ = (axis_id, key, file_name_array, menu_options, subdir_name);
    }
    pub fn open_axis_model_mode_options(
        &self,
        key: Option<&str>,
        axis_id: Option<AxisID>,
        model: Option<&str>,
        model_mode: bool,
        menu_options: Option<Infallible>,
    ) {
        let _ = (key, axis_id, model, model_mode, menu_options);
    }
    pub fn open_axis_file_type_options(
        &self,
        key: Option<&str>,
        axis_id: Option<AxisID>,
        model: Option<&FileType>,
        menu_options: Option<Infallible>,
    ) {
        let _ = (key, axis_id, model, menu_options);
    }
    pub fn open_axis_file_model_mode_options(
        &self,
        key: Option<&str>,
        axis_id: Option<AxisID>,
        file: Option<&Path>,
        model: Option<&str>,
        model_mode: bool,
        menu_options: Option<Infallible>,
    ) {
        let _ = (key, axis_id, file, model, model_mode, menu_options);
    }
    pub fn open_file_model_mode_options(
        &self,
        key: Option<&str>,
        file: Option<&Path>,
        model: Option<&str>,
        model_mode: bool,
        menu_options: Option<Infallible>,
    ) {
        let _ = (key, file, model, model_mode, menu_options);
    }
    pub fn open_file_axis_index_model_mode_options(
        &self,
        key: Option<&str>,
        file: Option<&Path>,
        axis_id: Option<AxisID>,
        vector_index: i32,
        model: Option<&str>,
        model_mode: bool,
        menu_options: Option<Infallible>,
    ) -> i32 {
        let _ = (key, file, axis_id, model, model_mode, menu_options);
        vector_index
    }
    pub fn set_file_file_axis_index(
        &self,
        key: Option<&str>,
        file: Option<&Path>,
        axis_id: Option<AxisID>,
        vector_index: i32,
    ) -> i32 {
        let _ = (key, file, axis_id);
        vector_index
    }
    pub fn open_file_axis_index_model_file_mode_options(
        &self,
        key: Option<&str>,
        file: Option<&Path>,
        axis_id: Option<AxisID>,
        vector_index: i32,
        model_file: Option<&Path>,
        model_mode: bool,
        menu_options: Option<Infallible>,
    ) -> i32 {
        let _ = (key, file, axis_id, model_file, model_mode, menu_options);
        vector_index
    }
    pub fn open_file_index_options(
        &self,
        key: Option<&str>,
        file: Option<&Path>,
        vector_index: i32,
        menu_options: Option<Infallible>,
    ) -> i32 {
        let _ = (key, file, menu_options);
        vector_index
    }
    pub fn open_model(
        &self,
        key: Option<&str>,
        axis_id: Option<AxisID>,
        vector_index: i32,
        model: Option<&str>,
        model_mode: bool,
    ) {
        let _ = (key, axis_id, vector_index, model, model_mode);
    }
    pub fn open_file_axis_index_options(
        &self,
        key: Option<&str>,
        file: Option<&Path>,
        axis_id: Option<AxisID>,
        vector_index: i32,
        menu_options: Option<Infallible>,
    ) -> i32 {
        let _ = (key, file, axis_id, menu_options);
        vector_index
    }
    pub fn open_axis_index_options(
        &self,
        key: Option<&str>,
        axis_id: Option<AxisID>,
        vector_index: i32,
        menu_options: Option<Infallible>,
    ) {
        let _ = (key, axis_id, vector_index, menu_options);
    }
    pub fn open_index_options(
        &self,
        key: Option<&str>,
        vector_index: i32,
        menu_options: Option<Infallible>,
    ) {
        let _ = (key, vector_index, menu_options);
    }
    pub fn open_index_model_mode_options(
        &self,
        key: Option<&str>,
        vector_index: i32,
        model: Option<&str>,
        model_mode: bool,
        menu_options: Option<Infallible>,
    ) {
        let _ = (key, vector_index, model, model_mode, menu_options);
    }
    pub fn delete(&self, key: Option<&str>, vector_index: i32) {
        let _ = (key, vector_index);
    }

    pub fn is_open(&self, key: Option<&str>) -> bool {
        let _ = key;
        false
    }
    pub fn is_open_axis(&self, key: Option<&str>, axis_id: Option<AxisID>) -> bool {
        let _ = (key, axis_id);
        false
    }
    pub fn is_open_axis_dataset(
        &self,
        key: Option<&str>,
        axis_id: Option<AxisID>,
        dataset_name: Option<&str>,
    ) -> bool {
        let _ = (key, axis_id, dataset_name);
        false
    }
    pub fn is_open_axis_file(
        &self,
        key: Option<&str>,
        axis_id: Option<AxisID>,
        file: Option<&Path>,
    ) -> bool {
        let _ = (key, axis_id, file);
        false
    }
    pub fn is_open_any(&self) -> bool {
        false
    }
    pub fn get_model_name(&self, key: Option<&str>, axis_id: Option<AxisID>) -> String {
        let _ = (key, axis_id);
        String::new()
    }
    pub fn get_rubberband_coordinates(&self, key: Option<&str>) -> Option<Vec<Option<Infallible>>> {
        let _ = key;
        None
    }
    pub fn get_slicer_angles(
        &self,
        key: Option<&str>,
        vector_index: i32,
    ) -> Option<Vec<Option<Infallible>>> {
        let _ = (key, vector_index);
        None
    }
    pub fn quit(&self, key: Option<&str>) {
        let _ = key;
    }
    pub fn quit_axis(&self, key: Option<&str>, axis_id: Option<AxisID>) {
        let _ = (key, axis_id);
    }
    pub fn quit_axis_dataset(
        &self,
        key: Option<&str>,
        axis_id: Option<AxisID>,
        dataset_name: Option<&str>,
    ) {
        let _ = (key, axis_id, dataset_name);
    }
    pub fn quit_axis_file(&self, key: Option<&str>, axis_id: Option<AxisID>, file: Option<&Path>) {
        let _ = (key, axis_id, file);
    }
    pub fn quit_all(&self, key: Option<&str>, axis_id: Option<AxisID>) {
        let _ = (key, axis_id);
    }
    pub fn quit_all_imods(&self) {}
    pub fn process_request(&self) {}
    pub fn disconnect(&self) {}

    // Java `set*` configuration boundaries forwarded to ImodState.
    pub fn set_swap_yz(&self, key: Option<&str>, axis_id: Option<AxisID>, swap_yz: bool) {
        let _ = (key, axis_id, swap_yz);
    }
    pub fn set_swap_yz_file(
        &self,
        key: Option<&str>,
        axis_id: Option<AxisID>,
        file: Option<&Path>,
        swap_yz: bool,
    ) {
        let _ = (key, axis_id, file, swap_yz);
    }
    pub fn set_file(&self, key: Option<&str>, axis_id: Option<AxisID>, file: Option<&Path>) {
        let _ = (key, axis_id, file);
    }
    pub fn set_swap_yz_single_file(&self, key: Option<&str>, file: Option<&Path>, swap_yz: bool) {
        let _ = (key, file, swap_yz);
    }
    pub fn set_open_bead_fixer(
        &self,
        key: Option<&str>,
        axis_id: Option<AxisID>,
        open_bead_fixer: bool,
    ) {
        let _ = (key, axis_id, open_bead_fixer);
    }
    pub fn set_open_surf_cont_point(&self, key: Option<&str>, axis_id: Option<AxisID>, open: bool) {
        let _ = (key, axis_id, open);
    }
    pub fn set_auto_center(&self, key: Option<&str>, axis_id: Option<AxisID>, auto_center: bool) {
        let _ = (key, axis_id, auto_center);
    }
    pub fn set_skip_list(
        &self,
        key: Option<&str>,
        axis_id: Option<AxisID>,
        skip_list: Option<&str>,
    ) {
        let _ = (key, axis_id, skip_list);
    }
    pub fn set_delete_all_sections(&self, key: Option<&str>, axis_id: Option<AxisID>, on: bool) {
        let _ = (key, axis_id, on);
    }
    pub fn set_beadfixer_mode(
        &self,
        key: Option<&str>,
        axis_id: Option<AxisID>,
        mode: Option<Infallible>,
    ) {
        let _ = (key, axis_id, mode);
    }
    pub fn set_open_log(
        &self,
        key: Option<&str>,
        axis_id: Option<AxisID>,
        open_log: bool,
        log_name: Option<&str>,
    ) {
        let _ = (key, axis_id, open_log, log_name);
    }
    pub fn reopen_log(&self, key: Option<&str>, axis_id: Option<AxisID>) {
        let _ = (key, axis_id);
    }
    pub fn set_open_log_off(&self, key: Option<&str>, axis_id: Option<AxisID>) {
        let _ = (key, axis_id);
    }
    pub fn set_new_contours(&self, key: Option<&str>, axis_id: Option<AxisID>, new_contours: bool) {
        let _ = (key, axis_id, new_contours);
    }
    pub fn set_binning_axis(&self, key: Option<&str>, axis_id: Option<AxisID>, binning: i32) {
        let _ = (key, axis_id, binning);
    }
    pub fn set_tilt_file(
        &self,
        key: Option<&str>,
        axis_id: Option<AxisID>,
        tilt_file: Option<&str>,
    ) {
        let _ = (key, axis_id, tilt_file);
    }
    pub fn reset_tilt_file(&self, key: Option<&str>, axis_id: Option<AxisID>) {
        let _ = (key, axis_id);
    }
    pub fn set_binning(&self, key: Option<&str>, binning: i32) {
        let _ = (key, binning);
    }
    pub fn set_binning_index(&self, key: Option<&str>, vector_index: i32, binning: i32) {
        let _ = (key, vector_index, binning);
    }
    pub fn set_binning_xy(&self, key: Option<&str>, binning: i32) {
        let _ = (key, binning);
    }
    pub fn set_continuous_listener_target(
        &self,
        key: Option<&str>,
        axis_id: Option<AxisID>,
        target: Option<Infallible>,
    ) {
        let _ = (key, axis_id, target);
    }
    pub fn set_binning_xy_index(&self, key: Option<&str>, vector_index: i32, binning: i32) {
        let _ = (key, vector_index, binning);
    }
    pub fn set_open_contours(
        &self,
        key: Option<&str>,
        axis_id: Option<AxisID>,
        open_contours: bool,
    ) {
        let _ = (key, axis_id, open_contours);
    }
    pub fn set_start_new_contours_at_new_z(
        &self,
        key: Option<&str>,
        axis_id: Option<AxisID>,
        start_new_contours_at_new_z: bool,
    ) {
        let _ = (key, axis_id, start_new_contours_at_new_z);
    }
    pub fn set_point_limit(&self, key: Option<&str>, axis_id: Option<AxisID>, point_limit: i32) {
        let _ = (key, axis_id, point_limit);
    }
    pub fn set_preserve_contrast(
        &self,
        key: Option<&str>,
        axis_id: Option<AxisID>,
        preserve_contrast: bool,
    ) {
        let _ = (key, axis_id, preserve_contrast);
    }
    pub fn set_frames(&self, key: Option<&str>, axis_id: Option<AxisID>, frames: bool) {
        let _ = (key, axis_id, frames);
    }
    pub fn set_piece_list_file_name(
        &self,
        key: Option<&str>,
        axis_id: Option<AxisID>,
        piece_list_file_name: Option<&str>,
    ) {
        let _ = (key, axis_id, piece_list_file_name);
    }
    pub fn set_montage_separation(&self, key: Option<&str>, axis_id: Option<AxisID>) {
        let _ = (key, axis_id);
    }
    pub fn set_interpolation(
        &self,
        key: Option<&str>,
        axis_id: Option<AxisID>,
        interpolation: bool,
    ) {
        let _ = (key, axis_id, interpolation);
    }
    pub fn set_working_directory(
        &self,
        key: Option<&str>,
        axis_id: Option<AxisID>,
        vector_index: i32,
        working_directory: Option<&Path>,
    ) {
        let _ = (key, axis_id, vector_index, working_directory);
    }
    pub fn set_piece_list_file_name_index(
        &self,
        key: Option<&str>,
        axis_id: Option<AxisID>,
        vector_index: i32,
        piece_list_file_name: Option<&str>,
    ) {
        let _ = (key, axis_id, vector_index, piece_list_file_name);
    }
    pub fn stop_request_handler(&self) {}
    pub fn warn_stale_file(&self, key: Option<&str>, axis_id: Option<AxisID>) -> bool {
        let _ = (key, axis_id);
        false
    }

    // Java private `newVector`, `newImodState`, `get`, and `getVector` overloads.
    pub fn new_vector(&self, imod_state: Option<Infallible>) -> Vec<Option<Infallible>> {
        vec![imod_state]
    }
    fn new_vector_extension_axis_dataset(
        &self,
        key: Option<&str>,
        extension: Option<&str>,
        axis: Option<AxisID>,
        dataset: Option<&str>,
    ) -> Vec<Option<Infallible>> {
        let _ = (key, extension, axis, dataset);
        vec![None]
    }
    fn new_vector_file(&self, key: Option<&str>, file: Option<&Path>) -> Vec<Option<Infallible>> {
        let _ = (key, file);
        vec![None]
    }
    fn new_vector_axis_file(
        &self,
        key: Option<&str>,
        axis: Option<AxisID>,
        file: Option<&Path>,
    ) -> Vec<Option<Infallible>> {
        let _ = (key, axis, file);
        vec![None]
    }
    fn new_vector_file_name_array(
        &self,
        key: Option<&str>,
        names: Option<&[Option<&str>]>,
    ) -> Vec<Option<Infallible>> {
        let _ = (key, names);
        vec![None]
    }
    fn new_vector_axis_file_list(
        &self,
        key: Option<&str>,
        axis: Option<AxisID>,
        files: Option<&[Option<&Path>]>,
    ) -> Vec<Option<Infallible>> {
        let _ = (key, axis, files);
        vec![None]
    }
    fn new_vector_file_name_array_subdir(
        &self,
        key: Option<&str>,
        names: Option<&[Option<&str>]>,
        subdir: Option<&str>,
    ) -> Vec<Option<Infallible>> {
        let _ = (key, names, subdir);
        vec![None]
    }
    fn new_imod_state_key(&self, key: Option<&str>) -> Option<Infallible> {
        let _ = key;
        None
    }
    fn new_imod_state_key_axis(
        &self,
        key: Option<&str>,
        axis: Option<AxisID>,
    ) -> Option<Infallible> {
        let _ = (key, axis);
        None
    }
    fn new_imod_state_key_axis_file_list(
        &self,
        key: Option<&str>,
        axis: Option<AxisID>,
        files: Option<&[Option<&Path>]>,
    ) -> Option<Infallible> {
        let _ = (key, axis, files);
        None
    }
    fn new_imod_state_key_extension_axis_dataset(
        &self,
        key: Option<&str>,
        extension: Option<&str>,
        axis: Option<AxisID>,
        dataset: Option<&str>,
    ) -> Option<Infallible> {
        let _ = (key, extension, axis, dataset);
        None
    }
    fn new_imod_state_key_file(
        &self,
        key: Option<&str>,
        file: Option<&Path>,
    ) -> Option<Infallible> {
        let _ = (key, file);
        None
    }
    fn new_imod_state_key_axis_file(
        &self,
        key: Option<&str>,
        axis: Option<AxisID>,
        file: Option<&Path>,
    ) -> Option<Infallible> {
        let _ = (key, axis, file);
        None
    }
    fn new_imod_state_key_file_name_array(
        &self,
        key: Option<&str>,
        names: Option<&[Option<&str>]>,
    ) -> Option<Infallible> {
        let _ = (key, names);
        None
    }
    fn new_imod_state_key_file_name_array_subdir(
        &self,
        key: Option<&str>,
        names: Option<&[Option<&str>]>,
        subdir: Option<&str>,
    ) -> Option<Infallible> {
        let _ = (key, names, subdir);
        None
    }
    pub fn get(&self, key: Option<&str>) -> Option<Infallible> {
        let _ = key;
        None
    }
    pub fn get_axis(&self, key: Option<&str>, axis: Option<AxisID>) -> Option<Infallible> {
        let _ = (key, axis);
        None
    }
    fn get_axis_dataset(
        &self,
        key: Option<&str>,
        axis: Option<AxisID>,
        dataset: Option<&str>,
    ) -> Option<Infallible> {
        let _ = (key, axis, dataset);
        None
    }
    fn get_axis_file(
        &self,
        key: Option<&str>,
        axis: Option<AxisID>,
        file: Option<&Path>,
    ) -> Option<Infallible> {
        let _ = (key, axis, file);
        None
    }
    fn get_axis_index(
        &self,
        key: Option<&str>,
        axis: Option<AxisID>,
        index: i32,
    ) -> Option<Infallible> {
        let _ = (key, axis, index);
        None
    }
    fn get_index(&self, key: Option<&str>, index: i32) -> Option<Infallible> {
        let _ = (key, index);
        None
    }
    fn delete_imod_state(&self, key: Option<&str>, index: i32) {
        let _ = (key, index);
    }
    fn get_vector(&self, key: Option<&str>) -> Option<Vec<Option<Infallible>>> {
        let _ = key;
        None
    }
    fn get_vector_axis_id_in_key(
        &self,
        key: Option<&str>,
        axis_id_in_key: bool,
    ) -> Option<Vec<Option<Infallible>>> {
        let _ = (key, axis_id_in_key);
        None
    }
    fn get_vector_axis(
        &self,
        key: Option<&str>,
        axis: Option<AxisID>,
    ) -> Option<Vec<Option<Infallible>>> {
        let _ = (key, axis);
        None
    }
}
