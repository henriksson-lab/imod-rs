//! `IMOD/Etomo/src/etomo/comscript/CopyTomoComs.java`.
//!
//! Runs `copytomocoms` itself through a `SystemProgram`, constructed as the
//! Java does (`python -u <scripts>/copytomocoms -StandardInput` with the
//! options on standard input); the mapping of that command array onto our own
//! program happens inside `SystemProgram`.
//!
//! **The option map's order.**  Java collects the options in a `HashMap` and
//! writes them in its iteration order; the map here is `JavaHashMap`, which
//! iterates in the same order, so the lines reach copytomocoms's standard input
//! in the Java's order.

use std::sync::{Arc, MutexGuard};

use crate::imod::etomo::util::java_hash_map::JavaHashMap;

use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::process::process_messages::ProcessMessages;
use crate::imod::etomo::process::system_program::{MessagesKind, SystemProgram};
use crate::imod::etomo::storage::directive_def::DirectiveDef;
use crate::imod::etomo::storage::directive_file_collection::DirectiveFileCollection;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::const_etomo_number::{
    ConstEtomoNumber, Type, java_lang_double_to_string,
};
use crate::imod::etomo::r#type::const_meta_data::ConstMetaData;
use crate::imod::etomo::r#type::data_source::DataSource;
use crate::imod::etomo::r#type::directive_file_type::DirectiveFileType;
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::status::Status;
use crate::imod::etomo::r#type::tilt_angle_type::TiltAngleType;
use crate::imod::etomo::r#type::view_type::ViewType;
use crate::imod::etomo::ui::swing::ui_harness;
use crate::imod::etomo::util::utilities;

/// Java private static `CHANGE_PARAMETERS_FILE_TAG`.
const CHANGE_PARAMETERS_FILE_TAG: &str = "change";
/// Java `HALF_FLOAT` (an `Integer`).
pub const HALF_FLOAT: i32 = 2;
/// Java `HALF_FLOAT_IF_FLOAT` (an `Integer`).
pub const HALF_FLOAT_IF_FLOAT: i32 = 1;

/// Java final `CopyTomoComs`.
pub struct CopyTomoComs {
    command: Vec<String>,
    voltage: EtomoNumber,
    spherical_aberration: EtomoNumber,
    ctf_files: EtomoNumber,
    /// Java `manager`.  Java's `metaData` field is `manager.getConstMetaData()`,
    /// the manager's own meta data object, so it is read through the manager.
    manager: &'static ApplicationManager,
    param_automation: bool,
    directive_automation: bool,
    #[allow(dead_code)]
    command_line: Option<String>,
    #[allow(dead_code)]
    exit_value: i32,
    debug: bool,
    copytomocoms: Option<SystemProgram>,
    /// Java `directiveFileCollection`: the setup harness's collection, an event
    /// dispatch thread object.
    directive_file_collection: Option<
        Arc<
            crate::imod::etomo::util::event_queue::EdtRef<
                std::cell::RefCell<DirectiveFileCollection>,
            >,
        >,
    >,
}

impl CopyTomoComs {
    /// Java `CopyTomoComs(ApplicationManager, boolean, boolean)`.
    pub fn new(
        manager: &'static ApplicationManager,
        param_automation: bool,
        directive_automation: bool,
    ) -> CopyTomoComs {
        let debug = etomo_director::ARGUMENTS.lock().unwrap().is_debug();
        CopyTomoComs {
            command: Vec::new(),
            voltage: EtomoNumber::new(),
            spherical_aberration: EtomoNumber::new_with_type(Some(Type::Double)),
            ctf_files: EtomoNumber::new(),
            manager,
            param_automation,
            directive_automation,
            command_line: None,
            exit_value: 0,
            debug,
            copytomocoms: None,
            directive_file_collection: None,
        }
    }

    /// Java `setup`.
    pub fn setup(&mut self) -> bool {
        if !self.gen_options() {
            return false;
        }
        let params: Option<Vec<String>>;
        let size = self.command.len();
        if size == 1 {
            params = Some(vec![self.command[0].clone()]);
        } else {
            params = Some(self.command.clone());
        }
        // Java string concatenation writes a null path as "null"
        let script_path = etomo_director::INSTANCE
            .get_python_script_path()
            .as_deref()
            .unwrap_or("null")
            .to_owned();
        let command = vec![
            "python".to_owned(),
            "-u".to_owned(),
            format!("{script_path}copytomocoms"),
            "-StandardInput".to_owned(),
        ];
        let manager: &'static dyn BaseManager = self.manager;
        let copytomocoms = SystemProgram::new(
            Some(manager),
            manager.get_property_user_dir(),
            Some(command),
            AxisID::Only,
            MessagesKind::MultiLineWarningInfo(true, true, false),
        );
        if params.is_some() {
            copytomocoms.set_std_input(params);
        }
        self.copytomocoms = Some(copytomocoms);
        true
    }

    /// Java `setDirectiveFileCollection(DirectiveFileCollection)`.
    pub fn set_directive_file_collection(
        &mut self,
        input: Option<crate::imod::etomo::ui::setup_recon_interface::DirectiveFileCollectionHandle>,
    ) {
        self.directive_file_collection =
            input.map(|input| Arc::new(crate::imod::etomo::util::event_queue::EdtRef::new(input)));
    }

    /// Java `setVoltage(ConstEtomoNumber)`.
    pub fn set_voltage(&mut self, input: Option<&ConstEtomoNumber>) {
        self.voltage.set_const_etomo_number(input);
    }

    /// Java `setSphericalAberration(ConstEtomoNumber)`.
    pub fn set_spherical_aberration(&mut self, input: Option<&ConstEtomoNumber>) {
        self.spherical_aberration.set_const_etomo_number(input);
    }

    /// Java `setCTFFiles(CtfFilesValue)`.
    pub fn set_ctf_files(&mut self, ctf_files_value: CtfFilesValue) {
        self.ctf_files.set_int(ctf_files_value.get());
    }

    /// Java `getCommandLine`.  Return the current command line string.
    pub fn get_command_line(&self) -> String {
        match self.copytomocoms.as_ref() {
            None => String::new(),
            Some(copytomocoms) => copytomocoms.get_command_line(),
        }
    }

    /// Java private `overrideParameter(Map, String, String)`.
    fn override_parameter(
        command_map: &mut JavaHashMap<String, Option<String>>,
        key: &str,
        value: Option<String>,
    ) {
        if command_map.contains_key(key) {
            command_map.remove(key);
        }
        command_map.insert(key.to_owned(), value);
    }

    /// Java private `genOptions`.
    fn gen_options(&mut self) -> bool {
        // `commandMap.put(key, value)`.
        let put = |command_map: &mut JavaHashMap<String, Option<String>>,
                   key: String,
                   value: Option<String>| {
            command_map.insert(key, value);
        };
        let mut command_map: JavaHashMap<String, Option<String>> = JavaHashMap::new();
        // Add options from the directive file collection. This is the main source of
        // parameters for batch processing. For interactive processing, this fills in the
        // parameters that setup dialog doesn't know about. The parameters that setup dialog
        // does know about are used to override the directive file collection parameters in
        // the next section.
        if let Some(directive_file_collection) = self.directive_file_collection.as_ref() {
            let directive_file_collection = directive_file_collection.get().borrow();
            // Load the setupset.copyarg directives.
            let iterator = directive_file_collection
                .get_copy_arg_entry_set()
                .iterator();
            if let Some(iterator) = iterator {
                for (key, value) in iterator {
                    put(&mut command_map, key, value);
                }
            }
        }
        let meta_data = ApplicationManager::get_meta_data(self.manager);
        if !self.directive_automation {
            // Override the parameters from the directive file collection.
            // Copytomocoms overrides the existing .com files. Make sure that the full
            // functionality is only used during setup.
            if !self.manager.is_new_manager() && self.ctf_files.is_null() {
                ui_harness::open_message_dialog_from_process(
                    Some(self.manager),
                    "ERROR:  Attempting to rebuild .com files when setup is already completed.",
                    "Etomo Error",
                    None,
                );
                return false;
            }
            let mut montage = false;
            let mut gradient = false;
            // Dataset name
            CopyTomoComs::override_parameter(
                &mut command_map,
                &DirectiveDef::NAME.get_name_for_axis(None),
                Some(meta_data.get_dataset_name()),
            );
            // View type: single or montaged
            if meta_data.get_view_type() == ViewType::Montage {
                CopyTomoComs::override_parameter(
                    &mut command_map,
                    &DirectiveDef::MONTAGE.get_name_for_axis(None),
                    Some("1".to_owned()),
                );
                montage = true;
            }
            // Backup directory
            let backup_directory = meta_data.get_backup_directory();
            if backup_directory != "" {
                CopyTomoComs::override_parameter(
                    &mut command_map,
                    "backup",
                    Some(backup_directory),
                );
            }
            // Data source: CCD or film
            if meta_data.get_data_source() == DataSource::Film {
                CopyTomoComs::override_parameter(&mut command_map, "film", Some("1".to_owned()));
            }
            // Pixel size
            CopyTomoComs::override_parameter(
                &mut command_map,
                &DirectiveDef::PIXEL.get_name_for_axis(None),
                Some(java_lang_double_to_string(meta_data.get_pixel_size())),
            );
            // Fiducial diameter
            CopyTomoComs::override_parameter(
                &mut command_map,
                &DirectiveDef::GOLD.get_name_for_axis(None),
                Some(java_lang_double_to_string(
                    meta_data.get_fiducial_diameter(),
                )),
            );
            // Image rotation
            let n = meta_data.get_image_rotation(AxisID::First);
            if !n.is_null() {
                CopyTomoComs::override_parameter(
                    &mut command_map,
                    &DirectiveDef::ROTATION.get_name_for_axis(Some(AxisID::First)),
                    Some(n.to_string()),
                );
            }
            // HalfFloatModeOutput
            let i: Option<i32> = meta_data.get_half_float_mode_output();
            if let Some(i) = i {
                CopyTomoComs::override_parameter(
                    &mut command_map,
                    &DirectiveDef::HALF_FLOAT.get_name_for_axis(None),
                    Some(i.to_string()),
                );
            }
            // A first tilt angle and tilt angle incriment
            if meta_data.get_tilt_angle_spec_a().get_type() == TiltAngleType::Range {
                CopyTomoComs::override_parameter(
                    &mut command_map,
                    &DirectiveDef::FIRST_INC.get_name_for_axis(Some(AxisID::First)),
                    Some(format!(
                        "{},{}",
                        java_lang_double_to_string(
                            meta_data.get_tilt_angle_spec_a().get_range_min()
                        ),
                        java_lang_double_to_string(
                            meta_data.get_tilt_angle_spec_a().get_range_step()
                        )
                    )),
                );
            }
            // Use an existing rawtilt file (this assumes that one is there and has
            // not been deleted by checkTiltAngleFiles()
            else if meta_data.get_tilt_angle_spec_a().get_type() == TiltAngleType::File {
                CopyTomoComs::override_parameter(
                    &mut command_map,
                    &DirectiveDef::USE_RAW_TLT.get_name_for_axis(Some(AxisID::First)),
                    Some("1".to_owned()),
                );
            }
            // Extract the tilt angle data from the stack
            else if meta_data.get_tilt_angle_spec_a().get_type() == TiltAngleType::Extract {
                CopyTomoComs::override_parameter(
                    &mut command_map,
                    &DirectiveDef::EXTRACT.get_name_for_axis(Some(AxisID::First)),
                    Some("1".to_owned()),
                );
            }
            // List of views to exclude from processing
            let mut exclude_projections = meta_data.get_exclude_projections_a();
            if exclude_projections != "" {
                CopyTomoComs::override_parameter(
                    &mut command_map,
                    &DirectiveDef::SKIP.get_name_for_axis(Some(AxisID::First)),
                    Some(exclude_projections),
                );
            }
            if meta_data.is_twodir(AxisID::First) {
                CopyTomoComs::override_parameter(
                    &mut command_map,
                    &DirectiveDef::TWODIR.get_name_for_axis(Some(AxisID::First)),
                    Some(meta_data.get_twodir(AxisID::First)),
                );
            }
            if meta_data.is_dose_sym(AxisID::First) {
                CopyTomoComs::override_parameter(
                    &mut command_map,
                    &DirectiveDef::DOSESYM.get_name_for_axis(Some(AxisID::First)),
                    Some(meta_data.get_dose_sym(AxisID::First)),
                );
            }
            if !self.voltage.is_null() {
                CopyTomoComs::override_parameter(
                    &mut command_map,
                    &DirectiveDef::VOLTAGE.get_name_for_axis(None),
                    Some(self.voltage.to_string()),
                );
            }
            if !self.spherical_aberration.is_null() {
                CopyTomoComs::override_parameter(
                    &mut command_map,
                    &DirectiveDef::CS.get_name_for_axis(None),
                    Some(self.spherical_aberration.to_string()),
                );
            }
            // Only create ctf files.
            if !self.ctf_files.is_null() {
                CopyTomoComs::override_parameter(
                    &mut command_map,
                    "CTFfiles",
                    Some(self.ctf_files.to_string()),
                );
            }
            // Undistort images with the given .idf file
            let distortion_file = meta_data.get_distortion_file();
            if distortion_file != "" {
                CopyTomoComs::override_parameter(
                    &mut command_map,
                    &DirectiveDef::DISTORT.get_name_for_axis(None),
                    Some(distortion_file),
                );
            }
            // Binning of raw stacks (needed to undistort if ambiguous)
            CopyTomoComs::override_parameter(
                &mut command_map,
                &DirectiveDef::BINNING.get_name_for_axis(None),
                Some(meta_data.get_binning()),
            );
            // Mag gradients correction file
            let mag_gradient_file = meta_data.get_mag_gradient_file();
            if mag_gradient_file != "" {
                CopyTomoComs::override_parameter(
                    &mut command_map,
                    &DirectiveDef::GRADIENT.get_name_for_axis(None),
                    Some(mag_gradient_file),
                );
                gradient = true;
                // It is only necessary to know if the focus was adjusted between montages
                // if a mag gradients correction file is being used.
                if montage && meta_data.get_adjusted_focus_a().is() {
                    CopyTomoComs::override_parameter(
                        &mut command_map,
                        &DirectiveDef::FOCUS.get_name_for_axis(Some(AxisID::First)),
                        Some("1".to_owned()),
                    );
                }
            }
            // Axis type: single or dual
            if meta_data.get_axis_type() == AxisType::DualAxis {
                CopyTomoComs::override_parameter(
                    &mut command_map,
                    &DirectiveDef::DUAL.get_name_for_axis(None),
                    Some("1".to_owned()),
                );
                // There is only one image rotation value in setup dialog and it is used to set
                // both image rotation values. Use the directive file B image rotation if it
                // exists.
                let key = DirectiveDef::BROTATION.get_name_for_axis(Some(AxisID::Second));
                if !command_map.contains_key(&key) {
                    let n = meta_data.get_image_rotation(AxisID::Second);
                    if !n.is_null() {
                        put(&mut command_map, key, Some(n.to_string()));
                    }
                }
                // B first tilt angle and tilt angle incriment
                if meta_data.get_tilt_angle_spec_b().get_type() == TiltAngleType::Range {
                    CopyTomoComs::override_parameter(
                        &mut command_map,
                        &DirectiveDef::BFIRST_INC.get_name_for_axis(Some(AxisID::Second)),
                        Some(format!(
                            "{},{}",
                            java_lang_double_to_string(
                                meta_data.get_tilt_angle_spec_b().get_range_min()
                            ),
                            java_lang_double_to_string(
                                meta_data.get_tilt_angle_spec_b().get_range_step()
                            )
                        )),
                    );
                }
                // Take tilt angle from a .rawtlt file - B
                else if meta_data.get_tilt_angle_spec_b().get_type() == TiltAngleType::File {
                    CopyTomoComs::override_parameter(
                        &mut command_map,
                        &DirectiveDef::BUSE_RAW_TLT.get_name_for_axis(Some(AxisID::Second)),
                        Some("1".to_owned()),
                    );
                }
                // Extract the tilt angle data from the stack - B
                else if meta_data.get_tilt_angle_spec_b().get_type() == TiltAngleType::Extract {
                    CopyTomoComs::override_parameter(
                        &mut command_map,
                        &DirectiveDef::BEXTRACT.get_name_for_axis(Some(AxisID::Second)),
                        Some("1".to_owned()),
                    );
                }
                // List of views to exclude from processing - B
                exclude_projections = meta_data.get_exclude_projections_b();
                if exclude_projections != "" {
                    CopyTomoComs::override_parameter(
                        &mut command_map,
                        &DirectiveDef::BSKIP.get_name_for_axis(Some(AxisID::Second)),
                        Some(exclude_projections),
                    );
                }
                if meta_data.is_twodir(AxisID::Second) {
                    CopyTomoComs::override_parameter(
                        &mut command_map,
                        &DirectiveDef::BTWODIR.get_name_for_axis(Some(AxisID::Second)),
                        Some(meta_data.get_twodir(AxisID::Second)),
                    );
                }
                if meta_data.is_dose_sym(AxisID::Second) {
                    CopyTomoComs::override_parameter(
                        &mut command_map,
                        &DirectiveDef::BDOSESYM.get_name_for_axis(Some(AxisID::Second)),
                        Some(meta_data.get_dose_sym(AxisID::Second)),
                    );
                }
                if montage && gradient && meta_data.get_adjusted_focus_b().is() {
                    CopyTomoComs::override_parameter(
                        &mut command_map,
                        &DirectiveDef::BFOCUS.get_name_for_axis(Some(AxisID::Second)),
                        Some("1".to_owned()),
                    );
                }
            }
        } else if let Some(directive_file_collection) = self.directive_file_collection.as_ref() {
            let directive_file_collection = directive_file_collection.get().borrow();
            // For directive-driven automation include command line items that are associated
            // with copyarg directives.
            let command_line_set = directive_file_collection.get_copy_arg_command_line_set();
            if let Some(command_line_set) = command_line_set {
                for (key, value) in command_line_set {
                    put(&mut command_map, key, value);
                }
            }
        }
        // Place the map entries in the command list.
        let mut stack_ext_set = false;
        for (key, value) in command_map.iter() {
            // Java string concatenation writes a null value as "null"
            self.command
                .push(format!("{key} {}", value.as_deref().unwrap_or("null")));
            // May already contain stackext from the copyarg directives.
            if !stack_ext_set && DirectiveDef::STACK_EXT.get_name().starts_with(key.as_str()) {
                stack_ext_set = true;
            }
        }
        if meta_data.is_set_fei_pixel_size() {
            self.command.push("fei".to_owned());
        }
        // StackExtension
        // May already contain stackext from the copyarg directives.
        if !stack_ext_set {
            for pair in self.command.iter() {
                if DirectiveDef::STACK_EXT
                    .get_name()
                    .starts_with(pair.as_str())
                {
                    stack_ext_set = true;
                    break;
                }
            }
        }
        if !stack_ext_set {
            self.command.push(format!(
                "StackExtension {}",
                meta_data
                    .get_raw_image_stack_extension()
                    .map(|extension| extension.to_string())
                    .unwrap_or("null".to_string())
            ));
        }
        self.command.push(format!(
            "NamingStyle {}",
            meta_data.get_image_filename_style()
        ));
        if let Some(directive_file_collection) = self.directive_file_collection.as_ref() {
            let directive_file_collection = directive_file_collection.get().borrow();
            for directive_file_type in [
                DirectiveFileType::BatchDefaults,
                DirectiveFileType::Scope,
                DirectiveFileType::System,
                DirectiveFileType::User,
                DirectiveFileType::Batch,
            ] {
                let directive_file =
                    directive_file_collection.get_directive_file(directive_file_type);
                if let Some(directive_file) = directive_file {
                    let file = directive_file.get_file();
                    if let Some(file) = file {
                        self.command.push(format!(
                            "{CHANGE_PARAMETERS_FILE_TAG} {}",
                            utilities::java_io_file_get_absolute_path(&file.to_string_lossy())
                        ));
                    }
                }
            }
        }
        // Options removed:
        // CCDEraser and local alignment entries
        // Always yes tiltalign relies on local entries to save default values
        // even if they are not used.
        true
    }

    /// Java `run`.  Execute the copytomocoms script.
    pub fn run(&self) -> i32 {
        let copytomocoms = match self.copytomocoms.as_ref() {
            None => return -1,
            Some(copytomocoms) => copytomocoms,
        };
        let exit_value: i32;

        // Delete the rawtilt files if extract raw tilts is selected
        self.check_tilt_angle_files();
        // debug: print complete parameter list
        let std_input = copytomocoms.get_std_input();
        if let Some(std_input) = std_input.as_ref()
            && self.debug
        {
            eprintln!("stdInput:");
            for line in std_input {
                eprintln!("{line}");
            }
        }
        // Execute the script
        copytomocoms.run();
        exit_value = copytomocoms.get_exit_value();
        exit_value
    }

    /// Java `getStdErrorString`.
    pub fn get_std_error_string(&self) -> Option<String> {
        match self.copytomocoms.as_ref() {
            None => Some("ERROR: Copytomocoms is null.".to_owned()),
            Some(copytomocoms) => copytomocoms.get_std_error_string(),
        }
    }

    /// Java `getStdError`.
    pub fn get_std_error(&self) -> Option<Vec<String>> {
        match self.copytomocoms.as_ref() {
            None => Some(vec!["ERROR: Copytomocoms is null.".to_owned()]),
            Some(copytomocoms) => copytomocoms.get_std_error(),
        }
    }

    /// Java `getProcessMessages`.  Returns a String array of warnings - one warning
    /// per element; make sure that warnings get into the error log.
    pub fn get_process_messages(&self) -> Option<MutexGuard<'_, ProcessMessages>> {
        self.copytomocoms
            .as_ref()
            .map(|copytomocoms| copytomocoms.get_process_messages())
    }

    /// Java private `checkTiltAngleFiles`.  The raw tilt files still need to be
    /// deleted because the user might be restarting with a trimmed stack, making
    /// the old raw tilt files invalid.  Only do this during setup.  Don't delete
    /// when using option -CT.
    fn check_tilt_angle_files(&self) {
        // For automation without a GUI, do not delete the raw tilt file and let copytomocoms
        // decide what to do.
        if (!self.manager.is_new_manager()) || !self.ctf_files.is_null() {
            return;
        }
        let manager: &'static dyn BaseManager = self.manager;
        let meta_data = ApplicationManager::get_meta_data(self.manager);
        let working_directory = manager.get_property_user_dir();
        // `new File(workingDirectory, name)`; Java string concatenation writes a null
        // dataset name as "null"
        let raw_tilt_file_for = |suffix: &str| {
            let name = format!("{}{suffix}", meta_data.get_dataset_name());
            match working_directory.as_deref() {
                None => std::path::PathBuf::from(name),
                Some(dir) => std::path::PathBuf::from(utilities::java_io_file_new(dir, &name)),
            }
        };
        if meta_data.get_axis_type() == AxisType::SingleAxis {
            if meta_data.get_tilt_angle_spec_a().get_type() != TiltAngleType::File {
                let raw_tilt_file = raw_tilt_file_for(".rawtlt");
                if raw_tilt_file.exists() && self.can_delete_raw_tilt_file(&raw_tilt_file) {
                    let _ = std::fs::remove_file(&raw_tilt_file);
                }
            }
        } else {
            if meta_data.get_tilt_angle_spec_a().get_type() != TiltAngleType::File {
                let raw_tilt_file = raw_tilt_file_for("a.rawtlt");
                if raw_tilt_file.exists() && self.can_delete_raw_tilt_file(&raw_tilt_file) {
                    let _ = std::fs::remove_file(&raw_tilt_file);
                }
            }
            if meta_data.get_tilt_angle_spec_b().get_type() != TiltAngleType::File {
                let raw_tilt_file = raw_tilt_file_for("b.rawtlt");
                if raw_tilt_file.exists() && self.can_delete_raw_tilt_file(&raw_tilt_file) {
                    let _ = std::fs::remove_file(&raw_tilt_file);
                }
            }
        }
    }

    /// Java private `canDeleteRawTiltFile(File)`.
    fn can_delete_raw_tilt_file(&self, raw_tilt_file: &std::path::Path) -> bool {
        let name = raw_tilt_file
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        let is_head = ui_harness::INSTANCE.with(|ui_harness| ui_harness.is_head());
        if self.param_automation || self.directive_automation || !is_head {
            eprintln!(
                "INFO: Preparing to run {}.  Removing existing raw tilt file {}.",
                ProcessName::COPYTOMOCOMS.get_text().unwrap_or("null"),
                name
            );
            return true;
        }
        ui_harness::open_delete_dialog_from_process(
            Some(self.manager),
            &[format!("Delete raw tilt file {name}?")],
            None,
        )
    }
}

/// Java static final nested class `CopyTomoComs.CtfFilesValue`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CtfFilesValue {
    /// Java `CTF_PLOTTER` (1).
    CtfPlotter,
    /// Java `CTF_CORRECTION` (2).
    CtfCorrection,
}

impl CtfFilesValue {
    /// Java private `get`.
    fn get(self) -> i32 {
        match self {
            CtfFilesValue::CtfPlotter => 1,
            CtfFilesValue::CtfCorrection => 2,
        }
    }
}
