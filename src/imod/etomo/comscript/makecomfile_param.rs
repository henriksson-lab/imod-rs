//! `IMOD/Etomo/src/etomo/comscript/MakecomfileParam.java`.
//!
//! Runs `makecomfile` through a `SystemProgram`, as the Java does; the command
//! array's mapping onto our own program happens inside `SystemProgram`.

use std::sync::{Arc, MutexGuard};

use super::fortran_input_string::FortranInputString;
use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::process::process_messages::ProcessMessages;
use crate::imod::etomo::process::system_program::SystemProgram;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::Type;
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::r#type::file_type::{self, FileType};
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::script_parameter::ScriptParameter;
use crate::imod::etomo::r#type::string_parameter::StringParameter;
use crate::imod::etomo::ui::swing::ui_expert_utilities::UIExpertUtilities;

/// Java `TARGET_AND_MIN_RATIOS_KEY`.
pub const TARGET_AND_MIN_RATIOS_KEY: &str = "TargetAndMinRatios";
/// Java `TARGET_AND_MIN_RATIOS_NPARAMS`.
pub const TARGET_AND_MIN_RATIOS_NPARAMS: i32 = 2;

/// Java `LOW_PASS_RADIUS_SIGMA_NPARAMS`.
pub const LOW_PASS_RADIUS_SIGMA_NPARAMS: i32 = 2;
/// Java `REDUCE_FILT_VOL_REDUCTION_FACTOR`.
pub const REDUCE_FILT_VOL_REDUCTION_FACTOR: &str = "1.0";

/// Java `INPUT_FILE`.
pub const INPUT_FILE: &str = "InputFile";
/// Java `OUTPUT_FILE`.
pub const OUTPUT_FILE: &str = "OutputFile";
/// Java `REDUCTION_FACTOR`.
pub const REDUCTION_FACTOR: &str = "ReductionFactor";
/// Java `Z_REDUCTION_FACTOR`.
pub const Z_REDUCTION_FACTOR: &str = "ZReductionFactor";
/// Java `LOW_PASS_RADIUS_SIGMA`.
pub const LOW_PASS_RADIUS_SIGMA: &str = "LowPassRadiusSigma";
/// Java `DECONVOLUTION_STRENGTH`.
pub const DECONVOLUTION_STRENGTH: &str = "DeconvolutionStrength";
/// Java `SNR_FALLOFF`.
pub const SNR_FALLOFF: &str = "SNRFalloff";
/// Java `HIGH_PASS_NYQUIST`.
pub const HIGH_PASS_NYQUIST: &str = "HighPassNyquist";
/// Java `DEFOCUS_IN_MICRONS`.
pub const DEFOCUS_IN_MICRONS: &str = "DefocusInMicrons";
/// Java `PHASE_SHIFT`.
pub const PHASE_SHIFT: &str = "PhaseShift";
/// Java `MODE_TO_OUTPUT`.
pub const MODE_TO_OUTPUT: &str = "ModeToOutput";
/// Java `SETUP_CHUNKS_IF_MEMORY_ERROR`.
pub const SETUP_CHUNKS_IF_MEMORY_ERROR: &str = "SetupChunksIfMemoryError";

/// Java `MakecomfileParam`.
pub struct MakecomfileParam {
    command: Vec<String>,
    bead_size: EtomoNumber,
    thickness_to_make: EtomoNumber,
    local_align_validation: EtomoNumber,
    target_and_min_ratios: FortranInputString,
    #[allow(dead_code)]
    skip_beam_tilt_with_one_rot: EtomoNumber,
    #[allow(dead_code)]
    input: StringParameter,
    input_file: StringParameter,
    reduction_factor: ScriptParameter,
    // NamingStyle
    // StackExtension
    manager: &'static ApplicationManager,
    axis_id: AxisID,
    file_type: Arc<FileType>,
    #[allow(dead_code)]
    command_line: Option<String>,
    makecomfile: Option<SystemProgram>,
    #[allow(dead_code)]
    exit_value: i32,
}

impl MakecomfileParam {
    /// Java `MakecomfileParam(ApplicationManager, AxisID, FileType)`.
    pub fn new(
        manager: &'static ApplicationManager,
        axis_id: AxisID,
        file_type: Arc<FileType>,
    ) -> MakecomfileParam {
        MakecomfileParam {
            command: Vec::new(),
            bead_size: EtomoNumber::new_with_type(Some(Type::Double)),
            thickness_to_make: EtomoNumber::new(),
            local_align_validation: EtomoNumber::new_with_type(Some(Type::Integer)),
            target_and_min_ratios: FortranInputString::new_with_key(
                Some(TARGET_AND_MIN_RATIOS_KEY),
                TARGET_AND_MIN_RATIOS_NPARAMS,
            ),
            skip_beam_tilt_with_one_rot: EtomoNumber::new_with_type_and_name(
                Type::Boolean,
                "SkipBeamTiltWithOneRot",
            ),
            input: StringParameter::new("Input"),
            input_file: StringParameter::new(INPUT_FILE),
            reduction_factor: ScriptParameter::new_with_type_and_name(
                Type::Double,
                REDUCTION_FACTOR,
            ),
            manager,
            axis_id,
            file_type,
            command_line: None,
            makecomfile: None,
            exit_value: -1,
        }
    }

    /// Java `setup`.
    pub fn setup(&mut self) -> bool {
        let manager: &'static dyn BaseManager = self.manager;
        let axis_id = self.axis_id;
        self.command.push("python".to_owned());
        self.command.push("-u".to_owned());
        // Java string concatenation writes a null path as "null"
        let script_path = etomo_director::INSTANCE
            .get_python_script_path()
            .as_deref()
            .unwrap_or("null")
            .to_owned();
        self.command
            .push(format!("{script_path}{}", ProcessName::MAKECOMFILE));

        let param_index = self.command.len();
        let class = &*file_type::CLASS;
        // A file name the FileType cannot build is a null element in Java; the
        // `(String) null` elements are kept out of the command here.
        if Arc::ptr_eq(&self.file_type, &class.restrict_align_comscript) {
            if !self.local_align_validation.is_null() {
                self.command.push("-local".to_owned());
                self.command.push(self.local_align_validation.to_string());
            }
            if !self.target_and_min_ratios.is_null() {
                self.command.push("-ratios".to_owned());
                self.command.push(self.target_and_min_ratios.to_string());
            }
            let meta_data = ApplicationManager::get_meta_data(self.manager);
            if meta_data.is_skip_beam_tilt_with_one_rot(axis_id) {
                self.command.push("-skipbeam".to_owned());
            }
            self.command.push("-input".to_owned());
            if let Some(name) = class
                .align_comscript
                .get_file_name(Some(manager), Some(axis_id))
            {
                self.command.push(name);
            }
        } else if Arc::ptr_eq(&self.file_type, &class.reduce_filt_vol_comscript) {
            self.command.push("-input".to_owned());
            self.command.push(self.input_file.to_string());
            self.command.push("-binning".to_owned());
            if !self.reduction_factor.is_null() {
                self.command.push(self.reduction_factor.to_string());
            } else {
                self.command
                    .push(REDUCE_FILT_VOL_REDUCTION_FACTOR.to_owned());
            }
        } else {
            if Arc::ptr_eq(&self.file_type, &class.gold_eraser_comscript)
                || Arc::ptr_eq(&self.file_type, &class.patch_tracking_comscript)
                || Arc::ptr_eq(&self.file_type, &class.cryo_position_comscript)
            {
                self.command.push("-root".to_owned());
                // Java string concatenation writes a null name as "null"
                self.command.push(format!(
                    "{}{}",
                    manager.get_name().as_deref().unwrap_or("null"),
                    axis_id.get_extension()
                ));
                if Arc::ptr_eq(&self.file_type, &class.patch_tracking_comscript) {
                    self.command.push("-input".to_owned());
                    if let Some(name) = class
                        .cross_correlation_comscript
                        .get_file_name(Some(manager), Some(axis_id))
                    {
                        self.command.push(name);
                    }
                    self.command.push("-binning".to_owned());
                    self.command.push(
                        UIExpertUtilities::INSTANCE
                            .get_stack_binning_base_manager_axis_id_file_type(
                                manager,
                                axis_id,
                                &class.prealigned_stack,
                            )
                            .to_string(),
                    );
                } else if Arc::ptr_eq(&self.file_type, &class.gold_eraser_comscript) {
                    if self.bead_size.is_null() {
                        return false;
                    }
                    self.command.push("-bead".to_owned());
                    self.command.push(self.bead_size.to_string());
                } else if Arc::ptr_eq(&self.file_type, &class.cryo_position_comscript) {
                    self.command.push("-thickness".to_owned());
                    self.command.push(self.thickness_to_make.to_string());
                }
            } else if !Arc::ptr_eq(&self.file_type, &class.autofidseed_comscript)
                && !Arc::ptr_eq(&self.file_type, &class.sirtsetup_comscript)
            {
                return false;
            }
            self.command.push("-StackExtension".to_owned());
            // The source dereferences `getBaseMetaData()` without a null check; an
            // ApplicationManager always has its meta data.
            let meta_data = manager
                .get_base_meta_data()
                .expect("java.lang.NullPointerException");
            self.command.push(
                meta_data
                    .get_raw_image_stack_extension()
                    .map_or_else(|| "null".to_owned(), |extension| extension.to_string()),
            );
            self.command.push("-NamingStyle".to_owned());
            self.command
                .push(meta_data.base().get_image_filename_style().to_string());
        }
        //
        let command = std::mem::take(&mut self.command);
        self.command = MakecomfileParam::add_standard_options(Some(command), manager, axis_id);
        if let Some(name) = self.file_type.get_file_name(Some(manager), Some(axis_id)) {
            self.command.push(name);
        }

        eprintln!("\nMakecomfile parameters:");
        if self.command.len() > param_index {
            for i in param_index..self.command.len() {
                eprint!("{} ", self.command[i]);
            }
        }
        eprintln!();

        self.makecomfile = Some(SystemProgram::new_array(
            Some(manager),
            manager.get_property_user_dir(),
            Some(self.command.clone()),
            AxisID::Only,
        ));
        true
    }

    /// Java static `addStandardOptions(ArrayList, BaseManager, AxisID)`.  The list is
    /// taken and handed back, as the Java appends to and returns the same list.
    pub fn add_standard_options(
        command: Option<Vec<String>>,
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
    ) -> Vec<String> {
        let mut command = match command {
            None => Vec::new(),
            Some(command) => command,
        };
        let class = &*file_type::CLASS;
        // `file.exists()` on a null file is a NullPointerException in Java; a file the
        // FileType cannot build is treated as absent here.
        let mut file = class
            .local_scope_template
            .get_file(Some(manager), Some(axis_id));
        if let Some(file) = file.as_ref().filter(|file| file.exists()) {
            command.push("-change".to_owned());
            // `file.getName()`
            command.push(
                file.file_name()
                    .map(|name| name.to_string_lossy().into_owned())
                    .unwrap_or_default(),
            );
        }
        file = class
            .local_system_template
            .get_file(Some(manager), Some(axis_id));
        if let Some(file) = file.as_ref().filter(|file| file.exists()) {
            command.push("-change".to_owned());
            // `file.getName()`
            command.push(
                file.file_name()
                    .map(|name| name.to_string_lossy().into_owned())
                    .unwrap_or_default(),
            );
        }
        file = class
            .local_user_template
            .get_file(Some(manager), Some(axis_id));
        if let Some(file) = file.as_ref().filter(|file| file.exists()) {
            command.push("-change".to_owned());
            // `file.getName()`
            command.push(
                file.file_name()
                    .map(|name| name.to_string_lossy().into_owned())
                    .unwrap_or_default(),
            );
        }
        file = class
            .local_batch_directive_file
            .get_file(Some(manager), Some(axis_id));
        if let Some(file) = file.as_ref().filter(|file| file.exists()) {
            command.push("-change".to_owned());
            // `file.getName()`
            command.push(
                file.file_name()
                    .map(|name| name.to_string_lossy().into_owned())
                    .unwrap_or_default(),
            );
        }
        command
    }

    /// Java `setBeadSize(String)`.
    pub fn set_bead_size(&mut self, input: Option<&str>) {
        self.bead_size.set_string(input);
    }

    /// Java `setThicknessToMake(String)`.
    pub fn set_thickness_to_make(&mut self, input: Option<&str>) {
        self.thickness_to_make.set_string(input);
    }

    /// Java `setLocalAlignValidation(int)`.
    pub fn set_local_align_validation(&mut self, input: i32) {
        self.local_align_validation.set_int(input);
    }

    /// Java `resetLocalAlignValidation`.
    pub fn reset_local_align_validation(&mut self) {
        self.local_align_validation.reset();
    }

    /// Java `setTargetAndMinRatios(FortranInputString)`.
    pub fn set_target_and_min_ratios(&mut self, fortran_input_string: &FortranInputString) {
        self.target_and_min_ratios
            .set_fortran_input_string(fortran_input_string);
    }

    /// Java `resetTargetAndMinRatios`.
    pub fn reset_target_and_min_ratios(&mut self) {
        self.target_and_min_ratios.reset();
    }

    /// Java `setInputFile(String)`.
    pub fn set_input_file(&mut self, input: Option<&str>) {
        self.input_file.set(input);
    }

    /// Java `setReductionFactor(String)`.
    pub fn set_reduction_factor(&mut self, input: Option<&str>) {
        self.reduction_factor.set_string(input);
    }

    /// Java `getCommandLine`.  Return the current command line string.
    pub fn get_command_line(&self) -> String {
        match self.makecomfile.as_ref() {
            None => String::new(),
            Some(makecomfile) => makecomfile.get_command_line(),
        }
    }

    /// Java `run`.  Execute the makecomfile script.
    pub fn run(&self) -> i32 {
        let makecomfile = match self.makecomfile.as_ref() {
            None => return -1,
            Some(makecomfile) => makecomfile,
        };
        let exit_value: i32;

        // Execute the script
        makecomfile.run();
        exit_value = makecomfile.get_exit_value();
        exit_value
    }

    /// Java `getStdErrorString`.
    pub fn get_std_error_string(&self) -> Option<String> {
        match self.makecomfile.as_ref() {
            None => Some("ERROR: makecomfile is null.".to_owned()),
            Some(makecomfile) => makecomfile.get_std_error_string(),
        }
    }

    /// Java `getStdError`.
    pub fn get_std_error(&self) -> Option<Vec<String>> {
        match self.makecomfile.as_ref() {
            None => Some(vec!["ERROR: makecomfile is null.".to_owned()]),
            Some(makecomfile) => makecomfile.get_std_error(),
        }
    }

    /// Java `getStdOutputString`.
    pub fn get_std_output_string(&self) -> Option<String> {
        match self.makecomfile.as_ref() {
            None => Some("ERROR: makecomfile is null.".to_owned()),
            Some(makecomfile) => makecomfile.get_std_output_string(),
        }
    }

    /// Java `getProcessMessages`.  Returns a String array of warnings - one warning
    /// per element; make sure that warnings get into the error log.
    pub fn get_process_messages(&self) -> Option<MutexGuard<'_, ProcessMessages>> {
        self.makecomfile
            .as_ref()
            .map(|makecomfile| makecomfile.get_process_messages())
    }
}
