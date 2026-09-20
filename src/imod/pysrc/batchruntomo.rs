//! Statement-for-statement translation of `IMOD/pysrc/batchruntomo`.
//!
//! The Python script is a module: its `def`s close over module-level globals.
//! Those globals become the fields of [`Brt`] and every `def` becomes one
//! method of it, in the source's order, so that the "one Rust function per
//! source function" rule holds while the shared state stays explicit.  The
//! `#### MAIN PROGRAM ####` body is [`batchruntomo`].
//!
//! `etomo`, `copytomocoms`, `makecomfile`, `processchunks` and every other IMOD
//! executable the script names stay process boundaries, run through the same
//! `imodpy`/`prochunks` helpers the script imports.
#![allow(dead_code)]

use std::collections::HashSet;
use std::ffi::{OsStr, OsString};
use std::fs::File;
use std::io::Write;
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};

use super::comchanger::abs_template_path;
use super::imodpy::{
    ImodpyError, MrcInfo, OptionValue, add_imod_bin_ignore_sighup, allowed_raw_stack_extensions,
    cleanup_files, com_extension_from_option, dataset_filename, default_naming_style,
    elapsed_time_components, get_err_strings, get_montage_size, get_mrc, get_mrc_pixel,
    get_mrc_size, get_naming_style, imod_abs_path, is_file_newer, make_backup_file, option_value,
    parse_list, print_pid, prnstr, read_text_file, run_cmd, run_goodframe,
    set_output_format_if_needed, set_root_and_extension, standard_type_extensions, write_text_file,
};
use super::pip::{
    pip_forbid_comments, pip_get_boolean, pip_get_err_no, pip_get_float, pip_get_integer,
    pip_get_non_option_arg, pip_get_string, pip_number_of_entries, pip_open_installed_adoc,
    pip_read_or_parse_options, pip_set_error,
};
use super::prochunks::{check_for_pro_chunks_quit, run_processchunks, transfer_remote_directory};
use super::pysed::{PysedSrc, pysed, sed_del_and_add, sed_modify};

/// `progname` (`IMOD/pysrc/batchruntomo:9`).
const PROGNAME: &str = "batchruntomo";
/// `prefix` (`IMOD/pysrc/batchruntomo:10`).
const PREFIX: &str = "ERROR: batchruntomo - ";

/// `batDictInd` (`IMOD/pysrc/batchruntomo:13`).
const BAT_DICT_IND: usize = 4;
/// `tmplDictInd` (`IMOD/pysrc/batchruntomo:14`).
const TMPL_DICT_IND: usize = 1;

/// `STRING_VALUE, INT_VALUE, FLOAT_VALUE, BOOL_VALUE` (`IMOD/pysrc/imodpy.py:126`).
const STRING_VALUE: i32 = 0;
const INT_VALUE: i32 = 1;
const FLOAT_VALUE: i32 = 2;
const BOOL_VALUE: i32 = 3;

/// The Python object a directive lookup or a converted value can hold.
///
/// `lookupDirective` returns `None`, an `int` (also for booleans), a `float`,
/// or a `str` -- the directive's text for `STRING_VALUE`, and the sentinel
/// `'ERROR'` when a numeric conversion failed.  Callers distinguish the
/// sentinel with `isinstance(val, str)` and test presence with Python
/// truthiness, so both are modelled here rather than collapsed into `Option`.
#[derive(Clone, Debug, PartialEq)]
pub enum PyVal {
    None,
    Int(i64),
    Float(f64),
    Str(String),
}

impl PyVal {
    /// Python truth value: `None`, `0`, `0.0` and `''` are false.
    pub fn truthy(&self) -> bool {
        match self {
            PyVal::None => false,
            PyVal::Int(value) => *value != 0,
            PyVal::Float(value) => *value != 0.0,
            PyVal::Str(value) => !value.is_empty(),
        }
    }

    /// `isinstance(value, str)`.
    pub fn is_str(&self) -> bool {
        matches!(self, PyVal::Str(_))
    }

    /// `value is None`.
    pub fn is_none(&self) -> bool {
        matches!(self, PyVal::None)
    }

    /// `isinstance(value, int)`; a Python `bool` is an `int` too.
    pub fn is_int(&self) -> bool {
        matches!(self, PyVal::Int(_))
    }

    /// The integer of an `int` value, 0 otherwise.
    pub fn int(&self) -> i64 {
        match self {
            PyVal::Int(value) => *value,
            PyVal::Float(value) => *value as i64,
            _ => 0,
        }
    }

    /// The float of an `int` or `float` value, 0. otherwise.
    pub fn float(&self) -> f64 {
        match self {
            PyVal::Int(value) => *value as f64,
            PyVal::Float(value) => *value,
            _ => 0.0,
        }
    }

    /// The text of a `str` value, `''` otherwise.
    pub fn text(&self) -> &str {
        match self {
            PyVal::Str(value) => value.as_str(),
            _ => "",
        }
    }
}

/// `repr(float)` / `str(float)`: the shortest decimal that round-trips, with
/// `.0` forced on an integral value and exponent form outside `1e-4 ..= 1e16`.
///
/// Rust's `{}` prints `1` for `1.0f64` and never uses exponent form, so a
/// string built by the script with `str()` or `'{}'.format()` needs this.
pub fn py_str_float(value: f64) -> String {
    if value.is_nan() {
        return "nan".to_owned();
    }
    if value.is_infinite() {
        return if value < 0.0 {
            "-inf".to_owned()
        } else {
            "inf".to_owned()
        };
    }
    let scientific = format!("{value:e}");
    let exponent: i32 = scientific
        .split_once('e')
        .and_then(|(_, exponent)| exponent.parse().ok())
        .unwrap_or(0);
    if exponent < -4 || exponent >= 16 {
        let (mantissa, _) = scientific
            .split_once('e')
            .unwrap_or((scientific.as_str(), "0"));
        let mantissa = if mantissa.contains('.') {
            mantissa.to_owned()
        } else {
            format!("{mantissa}.0")
        };
        let sign = if exponent < 0 { '-' } else { '+' };
        return format!("{mantissa}e{sign}{:02}", exponent.abs());
    }
    let plain = format!("{value}");
    if plain.contains('.') {
        plain
    } else {
        format!("{plain}.0")
    }
}

/// A Python `dict` with insertion order, which the script's iteration over
/// `allDirectives[ind]` and `validComDict` makes observable in its messages.
#[derive(Clone, Debug)]
pub struct PyDict<V> {
    entries: Vec<(String, V)>,
}

impl<V> Default for PyDict<V> {
    fn default() -> Self {
        PyDict {
            entries: Vec::new(),
        }
    }
}

impl<V> PyDict<V> {
    /// `key in dct`.
    pub fn contains(&self, key: &str) -> bool {
        self.entries.iter().any(|(name, _)| name == key)
    }

    /// `dct[key]`, or `None` when absent.
    pub fn get(&self, key: &str) -> Option<&V> {
        self.entries
            .iter()
            .find(|(name, _)| name == key)
            .map(|(_, value)| value)
    }

    /// `dct[key] = value`, keeping an existing key's position.
    pub fn set(&mut self, key: &str, value: V) {
        if let Some(entry) = self.entries.iter_mut().find(|(name, _)| name == key) {
            entry.1 = value;
        } else {
            self.entries.push((key.to_owned(), value));
        }
    }

    /// `len(dct)`.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// `not dct`.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// `for key in dct`.
    pub fn keys(&self) -> Vec<String> {
        self.entries.iter().map(|(name, _)| name.clone()).collect()
    }
}

/// One entry of the tag lists `printTaggedMessages` takes: the tag itself, the
/// flag sum, and the optional suffix of a three-element tag.
#[derive(Clone, Debug)]
pub struct MessageTag(pub &'static str, pub i32, pub Option<&'static str>);

/// `logSuffixTag` (`IMOD/pysrc/batchruntomo:5098`).
const LOG_SUFFIX_TAG: &str = "    [:LOG]";

/// `direcFileErrorNames` (`IMOD/pysrc/batchruntomo:5151`).
const DIREC_FILE_ERROR_NAMES: [&str; 5] = [
    "batch defaults",
    "scope template",
    "system template",
    "user template",
    "batch directive",
];

/// `handlingMessages` (`IMOD/pysrc/batchruntomo:5155`).
const HANDLING_MESSAGES: [&str; 7] = [
    "autofidseed",
    "transferfid",
    "track",
    "align",
    "solvematch",
    "autopatchfit",
    "dualvolmatch",
];

/// `copyPrefix` .. `combinePrefix` (`IMOD/pysrc/batchruntomo:5081-5094`).
const COPY_PREFIX: &str = "setupset.copyarg.";
const SETUP_PREFIX: &str = "setupset.";
const RUNTIME_PREFIX: &str = "runtime.";
const COM_PREFIX: &str = "comparam.";
const SCOPE_TMPL_TEXT: &str = "setupset.scopeTemplate";
const USER_TMPL_TEXT: &str = "setupset.userTemplate";
const SYS_TMPL_TEXT: &str = "setupset.systemTemplate";
const DATA_DIR_TEXT: &str = "setupset.datasetDirectory";
const ROOT_NAME_TEXT: &str = "setupset.copyarg.name";
const CP_TOMO_EXT_TEXT: &str = "setupset.copyarg.stackext";
const SCAN_HEAD_TEXT: &str = "setupset.scanHeader";
const PATCH_TRACK_TEXT: &str = "runtime.PatchTracking";
const AUTO_SEED_TEXT: &str = "runtime.SeedFinding";
const COMBINE_PREFIX: &str = "runtime.Combine";
const LOCAL_BATCH_COPY: &str = "batchDirective.adoc";

/// Step numbers (`IMOD/pysrc/batchruntomo:5104-5112`).
const CTF_PLOT_STEP_NUM: f64 = 9.;
const DETECT_3D_STEP_NUM: f64 = 10.;
const CTF_CORR_STEP_NUM: f64 = 11.;
const ERASE_STEP_NUM: f64 = 12.;
const TRIM_STEP_NUM: f64 = 20.;
const NAD_STEP_NUM: f64 = 21.;
const RED_FILT_STEP_NUM: f64 = 21.5;
const CLEAN_STEP_NUM: f64 = 22.;
const FINAL_STEP_NUM: f64 = 22.;

/// Defaults (`IMOD/pysrc/batchruntomo:5115-5136`).
const DFLT_DARK_EXCLUDE_FRACTION: f64 = 0.33;
const DFLT_DARK_EXCLUDE_RATIO: f64 = 0.17;
const DFLT_MATCH_ATO_BRATIO: f64 = 0.9;
const DFLT_COMBINE_NUM_SCALES: i64 = 4;
const DFLT_COMBINE_BOX_SIZE: i64 = 32;
const DFLT_SAMPLE_THICKNESS: i64 = 500;
const POSITION_BINNING_TARGET: i64 = 512;
const DFLT_POS_THICKNESSES: [i64; 4] = [250, 400, 500, 600];
const SIZES_FOR_POS_THICKNESSES: [i64; 4] = [0, 512, 1024, 2048];
const DFLT_EXTRA_WARP_TARGETS: &str = "0.4,0.45";
const DFLT_FINAL_PATCH_SIZE: &str = "E";
const SOLVEMATCH_MIN_FIDS: i64 = 8;
const SOLVEMATCH_MIN_EACH_SIDE: i64 = 3;
const SOLVEMATCH_MIN_SIDE_RATIO: f64 = 0.2;
const USE_FALLBACK_RATIO: f64 = 0.4;
const FB3D_OPTIMAL_BINNED_SIZE: f64 = 5.;
const FB3D_MIN_BINNED_SIZE: f64 = 4.;
const CRYO_POS_EXTRA_THICK: i64 = 25;
const DFLT_PT_MAX_TILT_ADJUST: f64 = 20.;
const DFLT_PT_MAX_ADJUSTED_ANGLE: f64 = 78.;

/// `options` (`IMOD/pysrc/batchruntomo:5063-5079`), the autodoc2man fallback.
const OPTIONS: [&str; 39] = [
    "directive:DirectiveFile:FNM:",
    "root:RootName:FNM:",
    "current:CurrentLocation:FNM:",
    "deliver:DeliverToDirectory:FNM:",
    "make:MakeSubDirectory:B:",
    "one:ProcessOneAxis:I:",
    "cpus:CPUMachineList:CH:",
    "single:SingleOnFirstCPU:B:",
    "gpus:GPUMachineList:CH:",
    "parallel:ParallelBatchRootName:CH:",
    "maxGPUs:MaxGPUsInParallelBatch:I:",
    "queue:QueueCommand:CH:",
    "jobs:MaxJobsOnQueue:I:",
    "jcores:CoresPerClusterJob:I:",
    "jgpus:GPUsPerClusterJob:I:",
    "gqueue:GPUQueueCommand:CH:",
    "gjobs:MaxGPUJobsOnQueue:I:",
    "frompath:TranslatePathsFrom:CHM:",
    "topath:TranslatePathsTo:CHM:",
    "remote:RemoteDirectory:FN:",
    "nice:NiceValue:I:",
    "limit:LimitLocalThreads:I:",
    "check:CheckFile:FN:",
    "email:EmailAddress:CH:",
    "SMTP:SMTPserver:CH:",
    "validation:ValidationType:I:",
    "end:EndingStep:F:",
    "start:StartingStep:F:",
    "first:StartForFirstSetOnly:B:",
    "use:UseExistingAlignment:B:",
    "exit:ExitOnError:B:",
    "style:NamingStyle:I:",
    "pcm:MakeComExtensionPcm:I:",
    "ext:StackExtension:CH:",
    "axis:AxisOfExtension:I:",
    "etomo:EtomoDebug:I:",
    "bypass:BypassEtomo:B:",
    ":PID:B:",
    "help:usage:B:",
];

/// The script's module-level state: the globals it declares at
/// `IMOD/pysrc/batchruntomo:17-29` together with the names the main program
/// binds at module level, which its `def`s read as globals too.
pub struct Brt {
    // Globals declared at :17-29
    pub dataset_dir: String,
    pub dual_axis: bool,
    pub set_name: String,
    pub scan_header: bool,
    pub if_montage: bool,
    pub defocus: f64,
    pub pixel_size: f64,
    pub fid_size_nm: Option<f64>,
    pub fid_size_pix: f64,
    pub mont_frame_data: [i64; 10],
    pub num_surfaces: i64,
    pub raw_xsize: i64,
    pub raw_ysize: i64,
    pub zsize: i64,
    pub fiducialless: i64,
    pub coarse_binning: i64,
    pub center_on_gold: bool,
    pub pos_sample_type: i64,
    pub xtilt_needed: f64,
    pub fid_thickness: f64,
    pub fid_inc_shift: f64,
    pub recon_thickness: f64,
    pub did_local_align: i64,
    pub made_zfactors: bool,
    pub ali_binning: i64,
    pub ali_xunbinned: i64,
    pub ali_yunbinned: i64,
    pub patch_track: bool,
    pub total_del_tilt: f64,
    pub latest_messages: Vec<String>,
    pub suppress_abort: bool,
    pub abs_directive_file: String,
    pub user_template_dir: Option<PathBuf>,
    pub summary_message: String,
    pub log_file: Option<File>,
    pub final_ret_val: i32,
    pub correct_ctf: i64,
    pub erase_gold: PyVal,
    pub axis_let: String,
    pub axis_num: usize,
    pub use_vol_match: i32,
    pub nx_rec_a: i64,
    pub ny_rec_a: i64,
    pub finish_set_and_quit: bool,
    pub final_align_resid: f64,
    pub axis_rotation: f64,
    pub transpose_for_ali: bool,
    pub edf_lines: Vec<String>,
    pub edf_changed: bool,
    pub excluded_views_a: String,
    pub excluded_views_b: String,
    pub no_xaxis_tilt: i64,
    pub stack_extension: String,
    pub from_extension: String,
    pub original_stack_ext: String,
    pub autofit_ctf: PyVal,
    pub ctf3d_slab_thick: i64,
    pub do_both_recons: bool,
    pub raw_for_3dctf: i64,
    pub skip_align_stack_mods: bool,
    pub filter_in_2d: i64,
    pub my_parallel_gpu: i64,
    pub my_gpu_list: String,
    pub need_gpu_release: bool,
    pub align_lines: Vec<String>,
    pub make_sym_links: Option<bool>,
    pub last_translate_ind: i32,
    pub dset_dir_translate_ind: i32,
    pub expand_factor: f64,

    // Names the main program binds at module level (:5096-5534)
    pub my_pid: u32,
    pub pro_chunk_check_file: String,
    pub standard_type_exts: Vec<String>,
    pub possible_stack_exts: Vec<String>,
    pub renaming_only: bool,
    pub queue_command: String,
    pub running_on_queue: bool,
    pub max_queue_jobs: i32,
    pub gpu_queue_command: String,
    pub max_gpu_queue_jobs: i32,
    pub cores_per_cluster_job: i32,
    pub gpus_per_cluster_job: i32,
    pub bypass_etomo: i32,
    pub starting_dir: String,
    pub parallel_root: String,
    pub from_parallel_paths: Vec<String>,
    pub to_parallel_paths: Vec<String>,
    pub num_translations: i32,
    pub dir_files: Vec<String>,
    pub direc_translate_inds: Vec<i32>,
    pub current_dirs: Vec<String>,
    pub cur_dir_translate_inds: Vec<i32>,
    pub num_current: i32,
    pub deliver_dirs: Vec<String>,
    pub deliver_translate_inds: Vec<i32>,
    pub num_deliver: i32,
    pub make_sub_dir: i32,
    pub do_delivery: bool,
    pub num_root_opts: i32,
    pub root_by_option: Vec<String>,
    pub num_sets: i32,
    pub cpu_list: String,
    pub local_name: String,
    pub parallel_cpu: i64,
    pub most_cpus: i64,
    pub top_cpu_machine: String,
    pub top_cpu_limit: i64,
    pub local_cpu_limit: i64,
    pub first_cpu_limit: i64,
    pub local_limit: i64,
    pub gpu_list: String,
    pub use_gpu: i64,
    pub parallel_gpu: i64,
    pub max_parallel_gpus: i32,
    pub hostname: String,
    pub niceness: i32,
    pub remote_start_dir: String,
    pub do_one_axis: i32,
    pub starting_step: f64,
    pub ending_step: f64,
    pub first_start: i32,
    pub exit_on_error: i32,
    pub etomo_debug: i32,
    pub use_first_cpu_for_single: i32,
    pub name_style: i32,
    pub type_extension: String,
    pub com_ext: String,
    pub setup_bname: String,
    pub skip_tiltalign: i32,
    pub validation: i32,
    pub remote_data_dir: String,
    pub email_address: String,
    pub smtp_server: String,
    pub check_file: String,
    pub base_com_dict: PyDict<String>,
    pub valid_com_dict: PyDict<(String, bool, bool, bool)>,
    pub valid_run_dict: PyDict<(String, bool, bool, bool)>,
    pub valid_other_dict: PyDict<(String, bool, bool, bool)>,
    pub file_type: String,
    pub defaults_file: String,
    pub validate_file: String,
    pub imod_dir: String,

    // Per-data-set names the main loop binds and the `def`s read (:5536-5926)
    pub dfile_ind: usize,
    pub all_directives: Vec<PyDict<(String, usize)>>,
    pub extIfAlreadySet: String,
    pub deliver_from_dir: String,
    pub del_from_translate_ind: i32,
    pub axis_ind: usize,
    pub axis_edf_let: String,
    pub axis_upper_let: String,
    pub data_name: String,
    pub axis_com: String,

    /// `prochunks.finishSetAndQuit` (`IMOD/pysrc/prochunks.py:27`), which the
    /// translated `check_for_pro_chunks_quit` takes as an out parameter.
    pub prochunks_finish_set_and_quit: bool,
}

impl Brt {
    /// The module-level initialisations at `IMOD/pysrc/batchruntomo:5096-5150`,
    /// plus the `None`/empty starting values of the declared globals.
    pub fn new() -> Brt {
        Brt {
            dataset_dir: String::new(),
            dual_axis: false,
            set_name: String::new(),
            scan_header: false,
            if_montage: false,
            defocus: 0.,
            pixel_size: 0.,
            fid_size_nm: None,
            fid_size_pix: 0.,
            mont_frame_data: [0; 10],
            num_surfaces: 0,
            raw_xsize: 0,
            raw_ysize: 0,
            zsize: 0,
            fiducialless: 0,
            coarse_binning: 1,
            center_on_gold: false,
            pos_sample_type: 0,
            xtilt_needed: 0.,
            fid_thickness: 0.,
            fid_inc_shift: 0.,
            recon_thickness: 0.,
            did_local_align: 0,
            made_zfactors: false,
            ali_binning: 1,
            ali_xunbinned: 0,
            ali_yunbinned: 0,
            patch_track: false,
            total_del_tilt: 0.,
            latest_messages: Vec::new(),
            suppress_abort: false,
            abs_directive_file: String::new(),
            user_template_dir: None,
            summary_message: String::new(),
            log_file: None,
            final_ret_val: 0,
            correct_ctf: 0,
            erase_gold: PyVal::None,
            axis_let: String::new(),
            axis_num: 0,
            use_vol_match: -1,
            nx_rec_a: 0,
            ny_rec_a: 0,
            finish_set_and_quit: false,
            final_align_resid: 0.,
            axis_rotation: 0.,
            transpose_for_ali: false,
            edf_lines: Vec::new(),
            edf_changed: false,
            excluded_views_a: String::new(),
            excluded_views_b: String::new(),
            no_xaxis_tilt: 0,
            stack_extension: String::new(),
            from_extension: String::new(),
            original_stack_ext: String::new(),
            autofit_ctf: PyVal::None,
            ctf3d_slab_thick: 0,
            do_both_recons: false,
            raw_for_3dctf: 0,
            skip_align_stack_mods: false,
            filter_in_2d: 0,
            my_parallel_gpu: 0,
            my_gpu_list: String::new(),
            need_gpu_release: false,
            align_lines: Vec::new(),
            make_sym_links: None,
            last_translate_ind: -1,
            dset_dir_translate_ind: -1,
            expand_factor: 1.,

            my_pid: std::process::id(),
            pro_chunk_check_file: "processchunks.cmds".to_owned(),
            standard_type_exts: standard_type_extensions(),
            possible_stack_exts: allowed_raw_stack_extensions(),
            renaming_only: false,
            queue_command: String::new(),
            running_on_queue: false,
            max_queue_jobs: 0,
            gpu_queue_command: String::new(),
            max_gpu_queue_jobs: 0,
            cores_per_cluster_job: 0,
            gpus_per_cluster_job: 0,
            bypass_etomo: 0,
            starting_dir: String::new(),
            parallel_root: String::new(),
            from_parallel_paths: Vec::new(),
            to_parallel_paths: Vec::new(),
            num_translations: 0,
            dir_files: Vec::new(),
            direc_translate_inds: Vec::new(),
            current_dirs: Vec::new(),
            cur_dir_translate_inds: Vec::new(),
            num_current: 0,
            deliver_dirs: Vec::new(),
            deliver_translate_inds: Vec::new(),
            num_deliver: 0,
            make_sub_dir: 0,
            do_delivery: false,
            num_root_opts: 0,
            root_by_option: Vec::new(),
            num_sets: 0,
            cpu_list: String::new(),
            local_name: String::new(),
            parallel_cpu: 0,
            most_cpus: 0,
            top_cpu_machine: "1".to_owned(),
            top_cpu_limit: 1,
            local_cpu_limit: 1,
            first_cpu_limit: 1,
            local_limit: 0,
            gpu_list: String::new(),
            use_gpu: 0,
            parallel_gpu: 0,
            max_parallel_gpus: 0,
            hostname: String::new(),
            niceness: 15,
            remote_start_dir: String::new(),
            do_one_axis: 0,
            starting_step: 0.,
            ending_step: 10000000.,
            first_start: 0,
            exit_on_error: 0,
            etomo_debug: 0,
            use_first_cpu_for_single: 0,
            name_style: 0,
            type_extension: String::new(),
            com_ext: ".com".to_owned(),
            setup_bname: String::new(),
            skip_tiltalign: 0,
            validation: 0,
            remote_data_dir: String::new(),
            email_address: String::new(),
            smtp_server: "localhost".to_owned(),
            check_file: String::new(),
            base_com_dict: PyDict::default(),
            valid_com_dict: PyDict::default(),
            valid_run_dict: PyDict::default(),
            valid_other_dict: PyDict::default(),
            file_type: "directive".to_owned(),
            defaults_file: String::new(),
            validate_file: String::new(),
            imod_dir: String::new(),

            dfile_ind: 0,
            all_directives: Vec::new(),
            extIfAlreadySet: String::new(),
            deliver_from_dir: String::new(),
            del_from_translate_ind: -1,
            axis_ind: 0,
            axis_edf_let: "a".to_owned(),
            axis_upper_let: "A".to_owned(),
            data_name: String::new(),
            axis_com: String::new(),

            prochunks_finish_set_and_quit: false,
        }
    }

    /// `exitError` (`IMOD/pysrc/pip.py:678`): `PipSetError`, which writes the
    /// exit prefix and message on stdout and exits, then `sys.exit(1)`.
    ///
    /// Not `pysrc::pip::exit_error`, which currently writes on stderr without
    /// the prefix that `PipReadOrParseOptions` installed; that unit is owned
    /// elsewhere in this tree.
    fn exit_error(&self, error_mess: &str) -> ! {
        pip_set_error(error_mess);
        let _ = std::io::stdout().flush();
        std::process::exit(1)
    }
}

impl Brt {
    /// `prnLog` (`IMOD/pysrc/batchruntomo:32`).
    pub fn prn_log(&mut self, string: &str, end: &str, flush: bool) {
        prnstr(string, end, flush);
        if let Some(log_file) = self.log_file.as_mut() {
            let _ = log_file.write_all(string.as_bytes());
            let _ = log_file.write_all(end.as_bytes());
            if flush {
                let _ = log_file.flush();
            }
        }
    }

    /// `closeLogFileWriteEdf` (`IMOD/pysrc/batchruntomo:39`).
    pub fn close_log_file_write_edf(&mut self) {
        if self.log_file.is_some() {
            // Etomo needs the comma
            prnstr(
                &format!(
                    "Batchruntomo finished with data set, {}  [brt12]",
                    ctime_now()
                ),
                "\n",
                false,
            );
            self.log_file = None;
        }

        if self.edf_changed {
            let edf_file = self.set_name.clone() + ".edf";
            make_backup_file(&(edf_file.clone() + "~"));
            make_backup_file(&edf_file);
            let lines = self.edf_lines.clone();
            if write_text_file(&edf_file, &lines, true).is_err() {
                prnstr(&format!("Error writing modified {edf_file}"), "\n", false);
            }
            self.edf_changed = false;
        }

        self.edf_lines = Vec::new();
    }

    /// `warning` (`IMOD/pysrc/batchruntomo:62`).
    pub fn warning(&mut self, strings: &[String], to_log: bool) {
        let mut strings = strings.to_vec();
        strings.push(" ".to_owned());
        strings[0] = "WARNING: ".to_owned() + &strings[0];
        for line in strings {
            if to_log {
                self.prn_log(&line, "\n", true);
            } else {
                prnstr(&line, "\n", true);
            }
        }
    }

    /// `sendEmail` (`IMOD/pysrc/batchruntomo:75`).
    ///
    /// `smtplib`/`email.mime.text` become a direct SMTP conversation; the
    /// network is the same external boundary the Python module reaches.
    pub fn send_email(&mut self, subject: &str, message: &str) {
        if self.email_address.is_empty() {
            return;
        }
        let msg = format!(
            "Content-Type: text/plain; charset=\"us-ascii\"\nMIME-Version: 1.0\n\
             Content-Transfer-Encoding: 7bit\nSubject: {subject}\nFrom: batchruntomo\nTo: {}\n\n{message}",
            self.email_address
        );
        let address = self.email_address.clone();
        let server = self.smtp_server.clone();
        if send_mail(&server, &address, &msg).is_err() {
            if server == "localhost" {
                self.warning(
                    &[
                        "Failed to send email notification; you probably need to specify an SMTP server"
                            .to_owned(),
                    ],
                    true,
                );
            } else {
                self.warning(&["Failed to send email notification".to_owned()], true);
            }
        }
    }

    /// `abortSet` (`IMOD/pysrc/batchruntomo:95`).
    pub fn abort_set(&mut self, err_string: &str) {
        if self.suppress_abort {
            return;
        }
        if self.renaming_only {
            self.exit_error(&format!("renaming only - {err_string}"));
        }

        let abort_str = ["ABORT SET: ", "ABORT AXIS: ", "ABORT AXIS: "];
        let line = format!("{}{err_string}", abort_str[self.axis_num]);
        self.prn_log(&line, "\n", false);
        self.prn_log("", "\n", true);
        let mut message = if self.dual_axis && self.axis_num != 0 {
            format!(
                "Batchruntomo aborted axis {} of",
                self.axis_let.to_uppercase()
            )
        } else if self.dual_axis {
            "Batchruntomo aborted combine of".to_owned()
        } else {
            "Batchruntomo aborted".to_owned()
        };
        message += &format!(" dataset {} after error:\n{err_string}\n", self.set_name);
        let subject = format!("Batchruntomo error on {}", self.set_name);
        self.send_email(&subject, &message);
        self.summary_message += &message;
        if self.axis_num == 0 {
            self.final_ret_val += 1;
        }
        if self.exit_on_error != 0 {
            std::process::exit(1);
        }
    }

    /// `reportImodError` (`IMOD/pysrc/batchruntomo:121`).
    pub fn report_imod_error(&mut self, abort_text: Option<&str>) {
        let err_strings = get_err_strings();
        let num = err_strings.len();
        for ind in 0..num {
            let mut line = err_strings[ind].clone();
            if ind == num - 1 {
                line = "ERROR: ".to_owned() + &line;
            }
            self.prn_log(&line, "", false);
        }
        if let Some(abort_text) = abort_text.filter(|text| !text.is_empty()) {
            self.abort_set(abort_text);
        }
    }

    /// `renameAndAbort` (`IMOD/pysrc/batchruntomo:135`).
    pub fn rename_and_abort(&mut self, from_name: &str, to_name: &str, format_str: &str) -> i32 {
        if std::fs::rename(from_name, to_name).is_err() {
            let message = fmt2(format_str, from_name, to_name);
            self.abort_set(&message);
            return 1;
        }
        0
    }

    /// `testDirectiveValue` (`IMOD/pysrc/batchruntomo:145`).
    pub fn test_directive_value(&mut self, val: &PyVal, directive: &str, dtype: &str) -> i32 {
        if val.is_str() {
            self.abort_set(&format!(
                "An error occurred converting the value of the directive {directive} to a {dtype}"
            ));
            return 1;
        }
        0
    }

    /// `printTaggedMessages` (`IMOD/pysrc/batchruntomo:159`), file-name form.
    pub fn print_tagged_messages_file(&mut self, logfile: &str, tags: &[MessageTag]) {
        let loglines = match read_text_file(logfile, None, true, None) {
            Ok(lines) => lines,
            Err(message) => {
                self.warning(&[format!("Error {message}")], true);
                return;
            }
        };
        self.print_tagged_messages(&loglines, tags);
    }

    /// `printTaggedMessages` (`IMOD/pysrc/batchruntomo:159`), line-list form.
    pub fn print_tagged_messages(&mut self, loglines: &[String], tags: &[MessageTag]) {
        let blank_needed = ["ERROR", "INFO", "WARNING"];
        let mut need_final_blank = false;
        let mut need_blank = String::new();
        self.latest_messages = Vec::new();
        let mut any_error = false;

        let mut reading_multi = false;
        for line in loglines {
            let l = line.clone();
            if reading_multi {
                self.prn_log(&l, "\n", false);
                need_final_blank = true;
                self.latest_messages.push(l.clone());
                if l.trim().is_empty() {
                    reading_multi = false;
                    need_blank = String::new();
                    need_final_blank = false;
                }
            } else {
                for tag in tags {
                    let matched;
                    if let Some(index) = tag.0.find(".*") {
                        if tag.1 & 4 != 0 {
                            matched =
                                l.contains(&tag.0[..index]) && l.contains(&tag.0[index + 2..]);
                        } else {
                            matched =
                                l.starts_with(&tag.0[..index]) && l.contains(&tag.0[index + 2..]);
                        }
                    } else if tag.1 & 4 != 0 {
                        matched = l.contains(tag.0);
                    } else {
                        // Workaround to badly fixed bug 2503 in Etomo: suppress No GUI
                        matched = l.starts_with(tag.0) && !l.contains("No GUI");
                    }

                    if matched {
                        // Suppress uninformative vmstopy output if there is already a message
                        if !any_error
                            || !(l.starts_with("ERROR:") && l.ends_with("exited with status 1"))
                        {
                            if tag.0.starts_with("ERROR") {
                                any_error = true;
                            }

                            // If we needed a blank for a particular kind of message, and this
                            // is not another of that type, put the blank out
                            if !need_blank.is_empty() && !tag.0.starts_with(&need_blank) {
                                self.prn_log("", "\n", false);
                                need_blank = String::new();
                                need_final_blank = false;
                            }

                            // Print message one way or the other
                            self.latest_messages.push(l.clone());
                            let mut l_out = l.clone();
                            if tag.1 & 2 != 0 {
                                l_out = l[tag.0.len()..].trim_start().to_owned();
                            }
                            if let Some(suffix) = tag.2 {
                                l_out += suffix;
                            }
                            self.prn_log(&l_out, "\n", false);
                            if tag.2 == Some(LOG_SUFFIX_TAG) {
                                self.prn_log("", "\n", false);
                            }
                            need_final_blank = true;

                            // See if a blank is needed after contiguous lines of this type
                            if need_blank.is_empty() {
                                for need in blank_needed {
                                    if tag.0.starts_with(need) {
                                        need_blank = need.to_owned();
                                        break;
                                    }
                                }
                            }
                        }

                        if tag.1 & 1 != 0 {
                            reading_multi = true;
                        }
                        break;
                    }
                }
            }
        }

        if need_final_blank {
            self.prn_log(" ", "\n", false);
        }
    }

    /// `findTaggedValue` (`IMOD/pysrc/batchruntomo:242`).
    pub fn find_tagged_value(
        &self,
        lines: &[String],
        tag: &str,
        separator: char,
        val_type: i32,
    ) -> PyVal {
        for l in lines {
            if let Some(ind) = l.find(separator) {
                if ind > 0 && l.contains(tag) && ind < l.len() - 1 {
                    let val_all = l[ind + 1..].trim();
                    if val_type == STRING_VALUE {
                        return PyVal::Str(val_all.to_owned());
                    }
                    let vsplit: Vec<&str> = val_all.split_whitespace().collect();
                    if vsplit.is_empty() {
                        return PyVal::None;
                    }
                    if val_type == INT_VALUE {
                        return match vsplit[0].parse::<i64>() {
                            Ok(value) => PyVal::Int(value),
                            Err(_) => PyVal::None,
                        };
                    }
                    return match vsplit[0].parse::<f64>() {
                        Ok(value) => PyVal::Float(value),
                        Err(_) => PyVal::None,
                    };
                }
            }
        }
        PyVal::None
    }

    /// `writeTextFileReportErr` (`IMOD/pysrc/batchruntomo:264`).
    pub fn write_text_file_report_err(&mut self, filename: &str, lines: &[String]) -> i32 {
        match write_text_file(filename, lines, true) {
            Ok(()) => 0,
            Err(message) => {
                self.abort_set(&format!("Error {message}"));
                1
            }
        }
    }

    /// `readTextFileReportErr` (`IMOD/pysrc/batchruntomo:272`).
    pub fn read_text_file_report_err(
        &mut self,
        filename: &str,
        message: Option<&str>,
    ) -> Vec<String> {
        let lines = match read_text_file(filename, message, true, None) {
            Ok(lines) => lines,
            Err(error) => {
                self.abort_set(&format!("Error {error}"));
                return Vec::new();
            }
        };
        if lines.is_empty() {
            self.abort_set(&format!("File {filename} is empty"));
        }
        lines
    }

    /// `translateParallelPath` (`IMOD/pysrc/batchruntomo:283`).
    pub fn translate_parallel_path(&mut self, dfile: &str) -> String {
        let mut dfile = dfile.to_owned();
        self.last_translate_ind = -1;
        if !self.from_parallel_paths.is_empty() && !self.to_parallel_paths.is_empty() {
            for ind in 0..self.num_translations as usize {
                let from_path = self.from_parallel_paths[ind].clone();
                if dfile.starts_with(&from_path) {
                    dfile = dfile.replace(&from_path, &self.to_parallel_paths[ind]);
                    self.last_translate_ind = ind as i32;
                    break;
                }
            }
        }

        dfile
    }

    /// `reverseTranslatePath` (`IMOD/pysrc/batchruntomo:300`).
    pub fn reverse_translate_path(&self, dfile: &str, trans_ind: i32) -> String {
        if self.from_parallel_paths.is_empty() || trans_ind < 0 {
            return dfile.to_owned();
        }
        let to_path = &self.to_parallel_paths[trans_ind as usize];
        if dfile.starts_with(to_path.as_str()) {
            return dfile.replace(
                to_path.as_str(),
                &self.from_parallel_paths[trans_ind as usize],
            );
        }

        dfile.to_owned()
    }

    /// `findPossibleStacks` (`IMOD/pysrc/batchruntomo:314`).
    pub fn find_possible_stacks(
        &mut self,
        stack_root: &str,
        already_stack_ext: &str,
    ) -> (i32, i32, i64) {
        let num_possible = self.possible_stack_exts.len();
        let mut exists1: i32 = -1;
        let mut exists2: i32 = -1;
        let mut size1: i64 = 0;
        let mut size2: i64 = 0;

        for ind in 0..num_possible {
            let possible = self.possible_stack_exts[ind].clone();
            if !self.from_extension.is_empty()
                && self.from_extension != possible
                && (already_stack_ext.is_empty() || already_stack_ext != possible)
            {
                continue;
            }
            let name = stack_root.to_owned() + &possible;
            if Path::new(&name).exists() {
                match get_mrc_size(&name) {
                    Ok((_nx, _ny, nz)) => {
                        if exists1 < 0 {
                            exists1 = ind as i32;
                            size1 = nz as i64;
                        }
                        // Make sure a file with expected extension is included in the two
                        else if exists2 < 0
                            || (!self.type_extension.is_empty()
                                && !self.stack_extension.is_empty()
                                && self.stack_extension == possible)
                        {
                            exists2 = ind as i32;
                            size2 = nz as i64;
                        }
                    }
                    Err(_) => {
                        self.warning(
                            &[format!(
                                "Could not read header of possible stack file {name}"
                            )],
                            true,
                        );
                    }
                }
            }

            // At end of standard extensions, simply stop if there is one
            if ind == self.standard_type_exts.len() - 1 && exists1 >= 0 {
                break;
            }
        }

        // Ignore two files if either one has a size of 1
        if exists2 >= 0 && size1 == 1 {
            // The source assigns `exist1`, a name it never reads again.
            size1 = size2;
            exists2 = -1;
        }
        if exists2 >= 0 && size2 == 1 {
            exists2 = -1;
        }

        (exists1, exists2, size1)
    }

    /// `checkRenameStack` (`IMOD/pysrc/batchruntomo:361`).
    pub fn check_rename_stack(&mut self, stack: &str) -> i32 {
        let already = self.extIfAlreadySet.clone();
        let (exist_ind1, exist_ind2, _size1) = self.find_possible_stacks(stack, &already);
        let pos_ext = self.possible_stack_exts[exist_ind1.max(0) as usize].clone();

        if exist_ind1 >= 0 && exist_ind2 >= 0 {
            let other = self.possible_stack_exts[exist_ind2 as usize].clone();
            self.abort_set(&format!(
                "There are two possible stack files in the dataset directory : {stack}{pos_ext} and {stack}{other}"
            ));
            return 1;
        }

        if exist_ind1 < 0 {
            if !self.from_extension.is_empty() {
                let from = self.from_extension.clone();
                self.abort_set(&format!("Stack file does not exist: {stack}{from}"));
            } else {
                self.abort_set(&format!(
                    "Stack file does not exist with any allowed extension: {stack}"
                ));
            }
            return 1;
        }

        // Keep track of original stack extension for putting in edf file
        if self.original_stack_ext.is_empty() {
            self.original_stack_ext = pos_ext.clone();
        }

        // If there is a type extension, keep the file as is and record the stack extension
        self.from_extension = String::new();
        if self.stack_extension.is_empty() && (!self.type_extension.is_empty() || exist_ind1 == 0) {
            self.stack_extension = ".".to_owned() + &pos_ext;
            return 0;
        }

        if (exist_ind1 > 0 && self.type_extension.is_empty())
            || (!self.type_extension.is_empty()
                && !self.stack_extension.is_empty()
                && self.stack_extension[1..] != pos_ext)
        {
            // Otherwise rename the stack to st or to the extension established by first axis
            let old_name = stack.to_owned() + &pos_ext;
            let mut new_name = stack.to_owned() + "st";
            if !self.stack_extension.is_empty() {
                new_name = stack.to_owned() + &self.stack_extension[1..];
            }
            if Path::new(&new_name).exists() {
                self.abort_set(&format!(
                    "Cannot rename stack from {old_name} to: {new_name} because a single-image file with that name already exists"
                ));
                return 1;
            }

            // Etomo is looking for "to:" to find new name
            if std::fs::rename(&old_name, &new_name).is_ok() {
                self.prn_log(
                    &format!("[brt9]  Renamed stack from {old_name} to: {new_name}"),
                    "\n",
                    false,
                );
                if self.stack_extension.is_empty() {
                    self.stack_extension = ".st".to_owned();
                }
            } else {
                self.abort_set(&format!(
                    "Error renaming stack from {old_name} to {new_name}"
                ));
                return 1;
            }
        }

        0
    }

    /// `testForSymLink` (`IMOD/pysrc/batchruntomo:416`).
    pub fn test_for_sym_link(&mut self) -> bool {
        if self.make_sym_links.is_none() {
            self.make_sym_links = Some(false);
            if self
                .lookup_directive(SETUP_PREFIX, "makeSymbolicLinks", 0, BOOL_VALUE)
                .truthy()
            {
                if cfg!(windows) {
                    self.prn_log(
                        "Directive to make symbolic links instead of delivering is ignored on Windows",
                        "\n",
                        false,
                    );
                } else {
                    self.make_sym_links = Some(true);
                    self.prn_log(
                        "Making symbolic links instead of delivering files to data set directory",
                        "\n",
                        false,
                    );
                }
            }
        }
        self.make_sym_links.unwrap_or(false)
    }

    /// `deliverAncillary` (`IMOD/pysrc/batchruntomo:432`).
    pub fn deliver_ancillary(
        &mut self,
        source: &str,
        dest: &str,
        _type_name: &str,
        full_ext: &str,
    ) -> i32 {
        if Path::new(source).exists() && !Path::new(dest).exists() {
            let result = if self.test_for_sym_link() {
                std::os::unix::fs::symlink(source, dest)
            } else {
                std::fs::rename(source, dest)
            };
            if result.is_err() {
                // The source's format string has five `{}` and four arguments,
                // so `str.format` raises IndexError here (`:440`).
                let message = format!(
                    "Error moving {} file {}.{} from {} to {}",
                    self.set_name, full_ext, self.deliver_from_dir, self.dataset_dir, ""
                );
                self.abort_set(&message);
                return 1;
            }
        }
        0
    }
}

/// `datetime.datetime.now().ctime()`, the C `asctime` form Python reproduces.
fn ctime_now() -> String {
    let seconds = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    unsafe { libc::localtime_r(&seconds, &mut tm) };
    const DAYS: [&str; 7] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
    const MONTHS: [&str; 12] = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    let weekday = ((tm.tm_wday + 6) % 7) as usize;
    format!(
        "{} {} {:2} {:02}:{:02}:{:02} {}",
        DAYS[weekday],
        MONTHS[tm.tm_mon as usize],
        tm.tm_mday,
        tm.tm_hour,
        tm.tm_min,
        tm.tm_sec,
        1900 + tm.tm_year
    )
}

/// `datetime.datetime.now().strftime('%H:%M:%S')`.
fn hms_now() -> String {
    let seconds = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    unsafe { libc::localtime_r(&seconds, &mut tm) };
    format!("{:02}:{:02}:{:02}", tm.tm_hour, tm.tm_min, tm.tm_sec)
}

/// `time.time()`.
fn py_time() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

/// `'{} ... {}'.format(a, b)` for the two-slot format strings the script
/// passes to `renameAndAbort` and `deliverAncillary`.
fn fmt2(format_str: &str, first: &str, second: &str) -> String {
    let mut result = String::new();
    let mut values = [first, second].into_iter();
    let mut characters = format_str.chars().peekable();
    while let Some(character) = characters.next() {
        if character == '{' && characters.peek() == Some(&'}') {
            characters.next();
            result.push_str(values.next().unwrap_or(""));
        } else {
            result.push(character);
        }
    }
    result
}

/// The `smtplib.SMTP` conversation `sendEmail` performs.
fn send_mail(server: &str, address: &str, message: &str) -> std::io::Result<()> {
    use std::io::{BufRead, BufReader};
    use std::net::TcpStream;
    let stream = TcpStream::connect((server, 25))?;
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut writer = stream;
    let mut line = String::new();
    reader.read_line(&mut line)?;
    for command in [
        format!("HELO {server}\r\n"),
        format!("MAIL FROM:<{address}>\r\n"),
        format!("RCPT TO:<{address}>\r\n"),
        "DATA\r\n".to_owned(),
    ] {
        writer.write_all(command.as_bytes())?;
        line.clear();
        reader.read_line(&mut line)?;
        if line.starts_with('4') || line.starts_with('5') {
            return Err(std::io::Error::other(line));
        }
    }
    writer.write_all(message.replace('\n', "\r\n").as_bytes())?;
    writer.write_all(b"\r\n.\r\n")?;
    line.clear();
    reader.read_line(&mut line)?;
    writer.write_all(b"QUIT\r\n")?;
    Ok(())
}

impl Brt {
    /// `deliverStack` (`IMOD/pysrc/batchruntomo:447`).
    pub fn deliver_stack(&mut self, axis: &str) -> i32 {
        self.test_for_sym_link();
        let stack_root = self.set_name.clone() + axis;
        let stack_root_dot = stack_root.clone() + ".";
        let source_root = Path::new(&self.deliver_from_dir)
            .join(&stack_root_dot)
            .to_string_lossy()
            .into_owned();
        let dest_root = Path::new(&self.dataset_dir)
            .join(&stack_root_dot)
            .to_string_lossy()
            .into_owned();

        // Etomo may change the current dir after delivery, so if they match, skip
        if self.deliver_from_dir == self.dataset_dir {
            return 0;
        }

        // Get existence of files and their Z sizes from both locations
        let (src_ind1, src_ind2, src_size1) = self.find_possible_stacks(&source_root, "");
        let already = self.extIfAlreadySet.clone();
        let (dest_ind1, dest_ind2, dest_size1) = self.find_possible_stacks(&dest_root, &already);
        let src_ext = self.possible_stack_exts[src_ind1.max(0) as usize].clone();
        let dest_ext = self.possible_stack_exts[dest_ind1.max(0) as usize].clone();

        // Handle clear conflict cases
        if dest_ind1 >= 0 && dest_ind2 >= 0 {
            let other = self.possible_stack_exts[dest_ind2 as usize].clone();
            self.abort_set(&format!(
                "There are two possible stack files in the dataset directory : {stack_root_dot}{dest_ext} and {stack_root_dot}{other}"
            ));
            return 1;
        }

        if src_ind1 >= 0 && src_ind2 >= 0 {
            let other = self.possible_stack_exts[src_ind2 as usize].clone();
            let from = self.deliver_from_dir.clone();
            self.abort_set(&format!(
                "There are two possible stack files in {from} : {stack_root_dot}{src_ext} and {stack_root_dot}{other}"
            ));
            return 1;
        }

        // Nothing already there and nothing available
        if src_ind1 < 0 && dest_ind1 < 0 {
            let kind = if axis.is_empty() {
                " for single axis"
            } else {
                " for dual axis"
            };
            let from = self.deliver_from_dir.clone();
            self.abort_set(&format!(
                "Stack file {stack_root}{kind} data set does not exist with any allowed extension in {from}"
            ));
            return 1;
        }

        // Something already there and something available, and not only one is single-image
        if src_ind1 >= 0
            && dest_ind1 >= 0
            && ((src_size1 > 1 && dest_size1 > 1) || (src_size1 == 1 && dest_size1 == 1))
        {
            let mess = format!(
                "A stack file {stack_root_dot}{dest_ext} already exists in {} but there is a possible stack file {stack_root_dot}{src_ext} in {}",
                self.dataset_dir, self.deliver_from_dir
            );
            if self.make_sym_links == Some(true) {
                self.prn_log(&format!("WARNING: {mess}"), "\n", false);
                self.prn_log(
                    "Assuming this is the result of making a symbolic link previously",
                    "\n",
                    false,
                );
            } else {
                self.abort_set(&mess);
                return 1;
            }
        }

        // Nothing available or single-image available but stack already there
        let mut renaming;
        let mut delivering = false;
        let source;
        self.from_extension = String::new();
        if src_ind1 < 0 || (dest_ind1 >= 0 && (src_size1 == 1 || self.make_sym_links == Some(true)))
        {
            self.prn_log(
                "Stack file is already delivered to dataset directory; assuming associated files are too",
                "\n",
                false,
            );

            // The existing extension is OK if: no type extension and it is st, or
            // type extension and either no stack extension yet, or it matches established one
            if (self.type_extension.is_empty() && dest_ind1 == 0)
                || (!self.type_extension.is_empty()
                    && (self.stack_extension.is_empty() || self.stack_extension[1..] == dest_ext))
            {
                if self.stack_extension.is_empty() {
                    self.stack_extension = ".".to_owned() + &dest_ext;
                }
                return 0;
            }
            renaming = true;
            if self.stack_extension.is_empty() {
                self.stack_extension = ".".to_owned() + &dest_ext;
            }
            source = dest_root.clone() + &dest_ext;
        }
        // True delivery
        else {
            delivering = true;
            source = source_root.clone() + &src_ext;
            if self.stack_extension.is_empty() {
                if !self.type_extension.is_empty() {
                    self.stack_extension = ".".to_owned() + &src_ext;
                } else {
                    self.stack_extension = ".st".to_owned();
                }
            }
            renaming = src_ext != self.stack_extension[1..];
        }

        let dest = dest_root.clone() + &self.stack_extension[1..];

        // Test if there is a file there: which should be single-image given tests above
        if Path::new(&dest).exists() {
            let base = Path::new(&dest)
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_default();
            let dataset_dir = self.dataset_dir.clone();
            self.abort_set(&format!(
                "A single-image file named {base} already exists in {dataset_dir}"
            ));
            return 1;
        }
        let result = if delivering && self.make_sym_links == Some(true) {
            std::os::unix::fs::symlink(&source, &dest)
        } else {
            std::fs::rename(&source, &dest)
        };
        if result.is_ok() {
            // Etomo is looking for "to:" to find new name/location
            if delivering {
                let from = self.reverse_translate_path(&source, self.del_from_translate_ind);
                let to = self.reverse_translate_path(&dest, self.dset_dir_translate_ind);
                self.prn_log(
                    &format!("[brt8]  Delivered stack {from} to: {to}"),
                    "\n",
                    false,
                );
            }
            if renaming {
                let base = Path::new(&source)
                    .file_name()
                    .map(|name| name.to_string_lossy().into_owned())
                    .unwrap_or_default();
                let from = self.reverse_translate_path(&base, self.del_from_translate_ind);
                let to = self.reverse_translate_path(&stack_root, self.dset_dir_translate_ind);
                let extension = self.stack_extension.clone();
                self.prn_log(
                    &format!("[brt9]  Renamed stack from {from} to: {to}{extension}"),
                    "\n",
                    false,
                );
            }
        } else {
            self.abort_set(&format!(
                "Error moving/renaming stack file from {source} to {dest}"
            ));
            return 1;
        }
        let _ = &mut renaming;

        // Move .mdoc, .log, .rawtlt files also
        let mdoc_ext = self.stack_extension.clone() + ".mdoc";
        if self.deliver_ancillary(
            &(source.clone() + ".mdoc"),
            &(dest.clone() + ".mdoc"),
            "metadata",
            &mdoc_ext,
        ) != 0
        {
            return 1;
        }
        if self.deliver_ancillary(
            &(source_root.clone() + "log"),
            &(dest_root.clone() + "log"),
            "tilt series log",
            "log",
        ) != 0
        {
            return 1;
        }
        if self.deliver_ancillary(
            &(source_root + "rawtlt"),
            &(dest_root + "rawtlt"),
            "raw tilt angle",
            "rawtlt",
        ) != 0
        {
            return 1;
        }

        0
    }

    /// `getQueueOptions` (`IMOD/pysrc/batchruntomo:573`).
    pub fn get_queue_options(
        &mut self,
        queue_option: &str,
        queue_env_var: &str,
        max_option: &str,
        max_env_var: &str,
        gpu_text: &str,
    ) -> (String, bool, i32) {
        let mut running_on_it = false;
        let mut max_jobs = 0;
        let command = match std::env::var(queue_env_var) {
            Ok(value) if !value.is_empty() => {
                if value == "None" {
                    String::new()
                } else {
                    running_on_it = true;
                    value
                }
            }
            _ => pip_get_string(queue_option, "").unwrap_or_default(),
        };

        if !command.is_empty() {
            if !self.cpu_list.is_empty() && self.cores_per_cluster_job == 0 {
                self.exit_error("You cannot enter a CPU machine list with a queue command");
            }
            if self.parallel_root.is_empty() {
                self.exit_error(
                    "Use of a cluster queue is allowed only when running batch in parallel",
                );
            }

            max_jobs = pip_get_integer(max_option, 0).unwrap_or(0);
            let mut max_entered = 1 - pip_get_err_no();
            if let Ok(value) = std::env::var(max_env_var) {
                if !value.is_empty() {
                    match value.parse::<i32>() {
                        Ok(parsed) => {
                            max_jobs = parsed;
                            max_entered = 1;
                        }
                        Err(_) => {
                            self.exit_error(&format!(
                                "Converting environment variable {max_env_var} to integer"
                            ));
                        }
                    }
                }
            }

            if max_jobs <= 0 || max_entered == 0 {
                self.exit_error(&format!(
                    "Maximum number of {gpu_text}jobs must be entered if a {gpu_text}cluster command is entered"
                ));
            }
        }

        (command, running_on_it, max_jobs)
    }

    /// `getClusterJobOption` (`IMOD/pysrc/batchruntomo:609`).
    pub fn get_cluster_job_option(&mut self, option: &str, env_var: &str) -> i32 {
        let mut entered = 0;
        let per_job;
        match std::env::var(env_var) {
            Ok(envar) => {
                if envar == "None" {
                    per_job = 0;
                } else {
                    match envar.parse::<i32>() {
                        Ok(value) => {
                            per_job = value;
                            entered = 1;
                        }
                        Err(_) => {
                            self.exit_error(&format!(
                                "Environment variable {env_var} must be an integer"
                            ));
                        }
                    }
                }
            }
            Err(_) => {
                per_job = pip_get_integer(option, 0).unwrap_or(0);
                entered = 1 - pip_get_err_no();
            }
        }

        if entered != 0 {
            if per_job <= 0 {
                self.exit_error(&format!(
                    "The value for {option} or {env_var} must be positive"
                ));
            }
            if self.parallel_root.is_empty() {
                self.exit_error(
                    "Cluster node options can be entered only when running batch in parallel",
                );
            }
        }

        per_job
    }

    /// `edfDelAndAdd` (`IMOD/pysrc/batchruntomo:639`).
    pub fn edf_del_and_add(&self, option: &str, value: &str, delim: char) -> Vec<String> {
        let option = "batchruntomo.".to_owned() + option;
        vec![
            format!("{delim}{option}{delim}d"),
            format!("{delim}Setup.DatasetName={delim}a{delim}{option}={value}{delim}"),
        ]
    }

    /// `boolStringForEdf` (`IMOD/pysrc/batchruntomo:646`).
    pub fn bool_string_for_edf(&self, value: bool) -> String {
        if value {
            return "true".to_owned();
        }
        "false".to_owned()
    }

    /// `readEdfFileIfNeeded` (`IMOD/pysrc/batchruntomo:653`).
    pub fn read_edf_file_if_needed(&mut self) -> i32 {
        if !self.edf_lines.is_empty() {
            return 0;
        }
        let name = self.set_name.clone() + ".edf";
        self.edf_lines = self.read_text_file_report_err(&name, None);
        if self.edf_lines.is_empty() {
            if self.bypass_etomo != 0 {
                self.prn_log(
                    "Ignore that error!  Starting a blank edf file.",
                    "\n",
                    false,
                );
                self.edf_lines = vec![format!("Setup.DatasetName={}", self.set_name)];
                return 0;
            }
            return 1;
        }
        0
    }

    /// `modifyEdfLines` (`IMOD/pysrc/batchruntomo:668`).
    pub fn modify_edf_lines(&mut self, sedcom: &[String]) -> i32 {
        if self.read_edf_file_if_needed() != 0 {
            return 1;
        }
        let lines = self.edf_lines.clone();
        match pysed(sedcom, PysedSrc::Lines(&lines), None, false, '/', true) {
            Ok(Some(new_lines)) => {
                self.edf_lines = new_lines;
                self.edf_changed = true;
                0
            }
            _ => {
                self.abort_set("Error modifying lines from .edf file");
                1
            }
        }
    }

    /// `checkExcludedViewsRemoved` (`IMOD/pysrc/batchruntomo:682`).
    pub fn check_excluded_views_removed(&mut self, setroot: &str, exclude_list: &str) -> i32 {
        // Get all possible files and read the last one
        let mut cut_list1: Vec<String> = Vec::new();
        let mut cut_list2: Vec<String> = Vec::new();
        let directory = Path::new(setroot)
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or(Path::new("."))
            .to_owned();
        let base = Path::new(setroot)
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        if let Ok(entries) = std::fs::read_dir(&directory) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().into_owned();
                let Some(rest) = name
                    .strip_prefix(&base)
                    .and_then(|rest| rest.strip_prefix("_cutviews"))
                    .and_then(|rest| rest.strip_suffix(".info"))
                else {
                    continue;
                };
                let full = Path::new(setroot)
                    .parent()
                    .filter(|parent| !parent.as_os_str().is_empty())
                    .map(|parent| parent.join(&name).to_string_lossy().into_owned())
                    .unwrap_or_else(|| name.clone());
                let digits: Vec<char> = rest.chars().collect();
                if digits.len() == 1 && digits[0].is_ascii_digit() {
                    cut_list1.push(full);
                } else if digits.len() == 2
                    && ('1'..='9').contains(&digits[0])
                    && digits[1].is_ascii_digit()
                {
                    cut_list2.push(full);
                }
            }
        }
        cut_list1.sort();
        cut_list2.sort();
        cut_list1.extend(cut_list2);
        if cut_list1.is_empty() {
            return 0;
        }
        let last = cut_list1[cut_list1.len() - 1].clone();
        let cut_lines = self.read_text_file_report_err(&last, None);
        if cut_lines.is_empty() {
            return -1;
        }

        // Get the list from the file and see if it matches the exclude list
        let cut_split: Vec<String> = cut_lines[cut_lines.len() - 1]
            .replace(',', " ")
            .split_whitespace()
            .map(str::to_owned)
            .collect();
        let exclude_split: Vec<String> = exclude_list
            .replace(',', " ")
            .split_whitespace()
            .map(str::to_owned)
            .collect();
        if cut_split.len() != exclude_split.len() {
            return 0;
        }
        for cut in &cut_split {
            if !exclude_split.contains(cut) {
                return 0;
            }
        }
        1
    }

    /// `printDirectiveErrors` (`IMOD/pysrc/batchruntomo:708`).
    pub fn print_directive_errors(&mut self, errors: &[String]) -> i32 {
        if !errors.is_empty() {
            self.prn_log(
                "ERROR: Incorrect directive(s) as listed below:",
                "\n",
                false,
            );
            for l in errors {
                let line = l.clone();
                self.prn_log(&line, "\n", false);
            }
            self.prn_log("", "\n", false);
            self.abort_set("Bad directives");
        }
        errors.len() as i32
    }
}

impl Brt {
    /// `readDirectiveOrTemplate` (`IMOD/pysrc/batchruntomo:720`).
    pub fn read_directive_or_template(
        &mut self,
        filename: &str,
        index: usize,
    ) -> (i32, Vec<String>, (i32, String), bool, String) {
        let err_ret = (1, Vec::new(), (0, String::new()), false, String::new());
        let mut direct_lines =
            match read_text_file(filename, Some("directive/template file"), true, None) {
                Ok(lines) => lines,
                Err(message) => {
                    self.abort_set(&format!("Error {message}"));
                    return err_ret;
                }
            };

        let mut rewrite_batch = false;
        let mut line_num_dict: PyDict<(i32, String)> = PyDict::default();
        let mut datadir = String::new();
        let skip_views_a = COPY_PREFIX.to_owned() + "skip";
        let skip_views_b = COPY_PREFIX.to_owned() + "bskip";

        // For the batch file, find existing lines of various things
        if index == BAT_DICT_IND {
            let nodi = (-1, String::new());
            for key in [
                ROOT_NAME_TEXT,
                DATA_DIR_TEXT,
                SCOPE_TMPL_TEXT,
                SYS_TMPL_TEXT,
                USER_TMPL_TEXT,
                skip_views_a.as_str(),
                skip_views_b.as_str(),
                CP_TOMO_EXT_TEXT,
            ] {
                line_num_dict.set(key, nodi.clone());
            }

            // Add raw boundary model variations if doing delivery
            if self.do_delivery {
                for operation in [PATCH_TRACK_TEXT, AUTO_SEED_TEXT] {
                    for axlet in [".a.", ".b.", ".any."] {
                        let direc = format!("{operation}{axlet}rawBoundaryModel");
                        line_num_dict.set(&direc, nodi.clone());
                    }
                }
            }

            // Find lines
            for ind in 0..direct_lines.len() {
                let line = direct_lines[ind].trim_start().to_owned();
                let lsplit: Vec<&str> = line.split('=').collect();
                if lsplit.len() < 2 {
                    continue;
                }
                let line_direc = lsplit[0].trim().to_owned();
                if line_num_dict.contains(&line_direc) {
                    line_num_dict.set(&line_direc, (ind as i32, lsplit[1].trim().to_owned()));
                }
            }

            let mut rootname = line_num_dict.get(ROOT_NAME_TEXT).unwrap().1.clone();
            datadir = line_num_dict.get(DATA_DIR_TEXT).unwrap().1.clone();

            // Get true root name and fix or add line
            if self.num_root_opts != 0 {
                rootname = self.root_by_option[self.dfile_ind].clone();
                let root_direct = format!("{COPY_PREFIX}name = {rootname}");
                rewrite_batch = true;
                let position = line_num_dict.get(ROOT_NAME_TEXT).unwrap().0;
                if position >= 0 {
                    direct_lines[position as usize] = root_direct;
                } else {
                    direct_lines.push(root_direct);
                }
            }

            // Take care of the data directory now if there are current directory entries
            if self.num_current != 0 && !rootname.is_empty() {
                if !self.do_delivery {
                    datadir = imod_abs_path(&self.current_dirs[self.dfile_ind]);
                    self.dset_dir_translate_ind = self.cur_dir_translate_inds[self.dfile_ind];
                }
                // If delivery, now we can make the directory
                // But we can't deliver file(s) until we know about dual/single
                else {
                    let top_dir;
                    if !self.deliver_dirs.is_empty() {
                        let del_ind = self.dfile_ind.min(self.deliver_dirs.len() - 1);
                        if !Path::new(&self.deliver_dirs[del_ind]).is_dir() {
                            let bad = self.deliver_dirs[del_ind].clone();
                            self.abort_set(&format!(
                                "Cannot make directory for dataset {rootname} because {bad} is not an existing directory"
                            ));
                        }
                        top_dir = self.deliver_dirs[del_ind].clone();
                        self.dset_dir_translate_ind = self.deliver_translate_inds[del_ind];
                    } else {
                        let ind = self.dfile_ind.min(self.current_dirs.len() - 1);
                        top_dir = self.current_dirs[ind].clone();
                        self.dset_dir_translate_ind = self.cur_dir_translate_inds[ind];
                    }

                    datadir = imod_abs_path(&Path::new(&top_dir).join(&rootname).to_string_lossy());

                    // Detect Etomo error when it loses the original stack location and adjust
                    // for it
                    if self.starting_step > 0.
                        && top_dir.ends_with(&rootname)
                        && !(Path::new(&datadir).exists() && Path::new(&datadir).is_dir())
                        && Path::new(&top_dir).join(rootname.clone() + ".edf").exists()
                    {
                        datadir = top_dir.clone();
                        self.warning(
                            &[
                                format!(
                                    "CurrentLocation entry for {rootname} appears to incorrectly be the dataset directory"
                                ),
                                "instead of the original location, so it will be used as the current dataset location"
                                    .to_owned(),
                            ],
                            true,
                        );
                    }

                    if !(Path::new(&datadir).exists() && Path::new(&datadir).is_dir()) {
                        if Path::new(&datadir).exists() {
                            self.abort_set(&format!(
                                "{datadir} already exists and is not a directory"
                            ));
                            return err_ret;
                        }
                        if std::fs::create_dir(&datadir).is_err() {
                            self.abort_set(&format!(
                                "Error making directory for dataset, {datadir}"
                            ));
                            return err_ret;
                        }

                        self.prn_log(&format!("Created dataset directory {datadir}"), "\n", false);
                    }
                }

                let data_direct = format!("{DATA_DIR_TEXT} = {datadir}");
                rewrite_batch = true;
                let position = line_num_dict.get(DATA_DIR_TEXT).unwrap().0;
                if position >= 0 {
                    direct_lines[position as usize] = data_direct;
                } else {
                    direct_lines.push(data_direct);
                }
            }

            // Now check and resolve pathless templates
            for (tmpl_text, ind) in [
                (SCOPE_TMPL_TEXT, TMPL_DICT_IND),
                (SYS_TMPL_TEXT, TMPL_DICT_IND + 1),
                (USER_TMPL_TEXT, TMPL_DICT_IND + 2),
            ] {
                let tmpl_name = line_num_dict.get(tmpl_text).unwrap().1.clone();
                if !tmpl_name.is_empty() {
                    let (abs_tmpl_name, err, user_template_dir, err_mess) = abs_template_path(
                        &tmpl_name,
                        (ind - TMPL_DICT_IND) as i32,
                        self.user_template_dir.clone(),
                        DIREC_FILE_ERROR_NAMES[ind],
                    );
                    self.user_template_dir = user_template_dir;
                    if err < 0 {
                        // The source aborts with `message`, a name it never
                        // binds, so CPython raises NameError here (`:832`).
                        self.abort_set(&err_mess);
                        return err_ret;
                    }
                    if err > 0 {
                        rewrite_batch = true;
                        let position = line_num_dict.get(tmpl_text).unwrap().0;
                        direct_lines[position as usize] = format!(
                            "{tmpl_text} = {}",
                            abs_tmpl_name.unwrap_or_default().to_string_lossy()
                        );
                    }
                }
            }

            // Deliver raw boundary model(s) and adjust directive to local file
            if self.do_delivery {
                for operation in [PATCH_TRACK_TEXT, AUTO_SEED_TEXT] {
                    for axlet in [".a.", ".b.", ".any."] {
                        let direc = format!("{operation}{axlet}rawBoundaryModel");
                        let entry = line_num_dict.get(&direc).unwrap().clone();
                        if entry.0 >= 0 {
                            let src_path = entry.1.clone();
                            let raw_file = Path::new(&src_path)
                                .file_name()
                                .map(|name| name.to_string_lossy().into_owned())
                                .unwrap_or_default();
                            let dest_path = Path::new(&datadir)
                                .join(&raw_file)
                                .to_string_lossy()
                                .into_owned();
                            let src_exists = Path::new(&src_path).exists();
                            let dest_exists = Path::new(&dest_path).exists();
                            if src_exists && !dest_exists {
                                if std::fs::rename(&src_path, &dest_path).is_ok() {
                                    self.prn_log(
                                        &format!("Moved boundary model {src_path} to {datadir}"),
                                        "\n",
                                        false,
                                    );
                                } else {
                                    self.abort_set(&format!(
                                        "Error renaming/moving raw boundary model {src_path} to {dest_path}"
                                    ));
                                    return err_ret;
                                }
                            } else if !(dest_exists && !src_exists) {
                                self.abort_set(&format!(
                                    "Raw boundary model {src_path} does not exist"
                                ));
                                return err_ret;
                            }

                            rewrite_batch = true;
                            direct_lines[entry.0 as usize] = format!("{direc} = {raw_file}");
                        }
                    }
                }
            }

            // Check for excluded views to be removed
            self.excluded_views_a = String::new();
            self.excluded_views_b = String::new();
            let check_root = Path::new(&datadir)
                .join(&rootname)
                .to_string_lossy()
                .into_owned();
            let entry_a = line_num_dict.get(&skip_views_a).unwrap().clone();
            if entry_a.0 >= 0 && !entry_a.1.is_empty() {
                // For first axis, check for no axis letter or a
                let match_single = self.check_excluded_views_removed(&check_root, &entry_a.1);
                if match_single < 0 {
                    return err_ret;
                }
                let match_a =
                    self.check_excluded_views_removed(&(check_root.clone() + "a"), &entry_a.1);
                if match_a < 0 {
                    return err_ret;
                }

                // If either matches, remove the skip entry
                if match_a != 0 || match_single != 0 {
                    rewrite_batch = true;
                    direct_lines[entry_a.0 as usize] = skip_views_a.clone() + " =";
                    self.prn_log(
                        &format!(
                            "Excluded views {} were already removed from A/only axis",
                            entry_a.1
                        ),
                        "\n",
                        false,
                    );
                } else {
                    self.excluded_views_a = entry_a.1.clone();
                }
            }

            // Do the same for B
            let entry_b = line_num_dict.get(&skip_views_b).unwrap().clone();
            if entry_b.0 >= 0 && !entry_b.1.is_empty() {
                let match_b =
                    self.check_excluded_views_removed(&(check_root.clone() + "b"), &entry_b.1);
                if match_b < 0 {
                    return err_ret;
                }
                if match_b != 0 {
                    rewrite_batch = true;
                    direct_lines[entry_b.0 as usize] = skip_views_b.clone() + " =";
                    self.prn_log(
                        &format!(
                            "Excluded views {} were already removed from B axis",
                            entry_b.1
                        ),
                        "\n",
                        false,
                    );
                } else {
                    self.excluded_views_b = entry_b.1.clone();
                }
            }
        }

        // Back to general processing of all kinds of files
        let mut valid_err: Vec<String> = Vec::new();
        for ind in 0..direct_lines.len() {
            let line = direct_lines[ind].trim_start().to_owned();
            if line.starts_with('#') || line.is_empty() {
                continue;
            }
            let lsplit: Vec<&str> = line.split('=').collect();
            if lsplit.len() < 2 {
                valid_err.push(format!(
                    "Directive from {filename} lacks an = separator: {line}"
                ));
                continue;
            }
            if lsplit[0].trim().is_empty() {
                valid_err.push(format!("Directive from {filename} lacks a key: {line}"));
                continue;
            }
            self.all_directives[index].set(lsplit[0].trim(), (lsplit[1].trim().to_owned(), ind));
        }

        let err = self.print_directive_errors(&valid_err);
        if err == 0 && index == BAT_DICT_IND {
            return (
                0,
                direct_lines,
                line_num_dict.get(CP_TOMO_EXT_TEXT).unwrap().clone(),
                rewrite_batch,
                datadir,
            );
        }
        (err, Vec::new(), (0, String::new()), false, String::new())
    }

    /// `lookupDirective` (`IMOD/pysrc/batchruntomo:925`).
    pub fn lookup_directive(
        &self,
        prefix: &str,
        option: &str,
        start_dct: usize,
        val_type: i32,
    ) -> PyVal {
        let mut best_dict: i32 = -1;
        let mut best_ind: usize = 0;
        let mut best_key_ind: usize = 0;
        let keys: [String; 3] = if prefix.starts_with(COM_PREFIX) {
            [
                format!("{prefix}.{option}"),
                format!("{prefix}a.{option}"),
                format!("{prefix}b.{option}"),
            ]
        } else if prefix == SETUP_PREFIX {
            [
                format!("{prefix}{option}"),
                format!("{prefix}{option}"),
                format!("{prefix}{option}"),
            ]
        } else {
            [
                format!("{prefix}.any.{option}"),
                format!("{prefix}.a.{option}"),
                format!("{prefix}.b.{option}"),
            ]
        };
        let key_check: [&[usize]; 3] = [&[0, 1], &[0, 1], &[0, 2]];
        for dct in start_dct..=BAT_DICT_IND {
            for &key_ind in key_check[self.axis_num] {
                if self.all_directives[dct].contains(&keys[key_ind]) {
                    let mut better = true;
                    let new_ind = self.all_directives[dct].get(&keys[key_ind]).unwrap().1;

                    // If two entries are equivalent with regard to axis preference, new one is
                    // better if it comes from later dictionary or was later in file
                    let equiv_better = dct as i32 > best_dict || new_ind > best_ind;
                    if best_dict >= 0 {
                        // For dual axis, new one is better if it matches the current axis and
                        // previous one did not; or if they are for same axis and this one is later
                        if self.dual_axis {
                            better = (key_ind == self.axis_num && best_key_ind != self.axis_num)
                                || (key_ind == best_key_ind && equiv_better);
                        } else {
                            // For single axis, all "any" and "a" entries are equivalent
                            better = equiv_better;
                        }
                    }

                    if better {
                        best_dict = dct as i32;
                        best_ind = new_ind;
                        best_key_ind = key_ind;
                    }
                }
            }
        }

        // If nothing was found, return None, or 0 for a boolean
        if best_dict < 0 {
            if val_type == BOOL_VALUE {
                return PyVal::Int(0);
            }
            return PyVal::None;
        }

        // Otherwise return 1 for a boolean only if it is specifically 1, or return the
        // converted value, or the string
        let value = self.all_directives[best_dict as usize]
            .get(&keys[best_key_ind])
            .unwrap()
            .0
            .clone();
        if val_type == BOOL_VALUE {
            if value == "1" {
                return PyVal::Int(1);
            }
            PyVal::Int(0)
        } else if val_type == INT_VALUE || val_type == FLOAT_VALUE {
            if value.is_empty() {
                return PyVal::None;
            }
            if val_type == INT_VALUE {
                match value.parse::<i64>() {
                    Ok(numval) => PyVal::Int(numval),
                    Err(_) => PyVal::Str("ERROR".to_owned()),
                }
            } else {
                match value.parse::<f64>() {
                    Ok(numval) => PyVal::Float(numval),
                    Err(_) => PyVal::Str("ERROR".to_owned()),
                }
            }
        } else {
            PyVal::Str(value)
        }
    }

    /// `laterComDirectives` (`IMOD/pysrc/batchruntomo:989`).
    pub fn later_com_directives(&self, start_ind: usize) -> Vec<String> {
        let mut lines = Vec::new();
        if start_ind < 1 && !self.all_directives[0].is_empty() {
            lines.push(format!("ChangeParametersFile {}", self.defaults_file));
        }
        if start_ind <= TMPL_DICT_IND && self.all_directives[BAT_DICT_IND].contains(SCOPE_TMPL_TEXT)
        {
            lines.push(format!(
                "ChangeParametersFile {}",
                self.all_directives[BAT_DICT_IND]
                    .get(SCOPE_TMPL_TEXT)
                    .unwrap()
                    .0
            ));
        }
        if start_ind <= TMPL_DICT_IND + 1
            && self.all_directives[BAT_DICT_IND].contains(SYS_TMPL_TEXT)
        {
            lines.push(format!(
                "ChangeParametersFile {}",
                self.all_directives[BAT_DICT_IND]
                    .get(SYS_TMPL_TEXT)
                    .unwrap()
                    .0
            ));
        }
        if start_ind <= TMPL_DICT_IND + 2
            && self.all_directives[BAT_DICT_IND].contains(USER_TMPL_TEXT)
        {
            lines.push(format!(
                "ChangeParametersFile {}",
                self.all_directives[BAT_DICT_IND]
                    .get(USER_TMPL_TEXT)
                    .unwrap()
                    .0
            ));
        }
        lines.push(format!("ChangeParametersFile {}", self.abs_directive_file));
        lines
    }

    /// `useFileAsReplacement` (`IMOD/pysrc/batchruntomo:1005`).
    pub fn use_file_as_replacement(
        &mut self,
        use_file: &str,
        old_file: &str,
        save_orig: bool,
        make_backup: bool,
    ) -> i32 {
        let (base, ext) = match old_file.rfind('.') {
            Some(index) if index > 0 => (&old_file[..index], &old_file[index..]),
            _ => (old_file, ""),
        };
        let origname = format!("{base}_orig{ext}");
        let mut err;
        if save_orig && !Path::new(&origname).exists() {
            err = format!("{old_file} to {origname}");
            if let Err(error) = std::fs::rename(old_file, &origname) {
                self.abort_set(&format!("Error renaming {err} : {error}"));
                return 1;
            }
        } else if make_backup {
            make_backup_file(old_file);
        } else {
            cleanup_files(&[old_file.to_owned()]);
        }
        err = format!("{use_file} to {old_file}");
        if let Err(error) = std::fs::rename(use_file, old_file) {
            self.abort_set(&format!("Error renaming {err} : {error}"));
            return 1;
        }
        0
    }

    /// `transformRawBoundaryModel` (`IMOD/pysrc/batchruntomo:1026`).
    pub fn transform_raw_boundary_model(&mut self, model_in: &str, model_out: &str) -> i32 {
        let imfile = dataset_filename(".preali", None, None);
        let size = match get_mrc_size(&imfile) {
            Ok(size) => size,
            Err(_) => {
                self.report_imod_error(Some("Could not transform boundary model to match stack"));
                return 1;
            }
        };
        let (panx, pany, _panz) = size;
        let comstr = format!(
            "imodtrans -I \"{}{}\" -i \"{}\" -2 \"{}.prexg\" -S {} -tx {} -ty {} \"{}\" \"{}\"",
            self.data_name,
            self.stack_extension,
            imfile,
            self.data_name,
            py_str_float(1. / self.coarse_binning as f64),
            py_str_float(
                (panx as i64 - self.raw_xsize.div_euclid(self.coarse_binning)) as f64 / 2.
            ),
            py_str_float(
                (pany as i64 - self.raw_ysize.div_euclid(self.coarse_binning)) as f64 / 2.
            ),
            model_in,
            model_out
        );
        self.prn_log(
            &format!("Transforming {model_in} to {model_out} with:\n{comstr}"),
            "\n",
            false,
        );
        if run_cmd(&comstr, None, None, None, &[]).is_err() {
            self.report_imod_error(Some("Could not transform boundary model to match stack"));
            return 1;
        }
        0
    }

    /// `processQuitAction` (`IMOD/pysrc/batchruntomo:1044`).
    pub fn process_quit_action(&mut self, action: &str, message: &str) {
        if action.is_empty() {
            return;
        }
        if action == "Q" {
            let mut message = message.to_owned();
            if message.is_empty() {
                message = "RECEIVED SIGNAL TO QUIT, JUST EXITING".to_owned();
            }
            self.prn_log(&format!("{message}   [brt5]"), "\n", false);
            if !self.parallel_root.is_empty() {
                std::process::exit(1);
            } else {
                std::process::exit(0);
            }
        }
        if action == "F" {
            self.finish_set_and_quit = true;
        }
    }

    /// `checkForQuit` (`IMOD/pysrc/batchruntomo:1061`).
    pub fn check_for_quit(&mut self) {
        let check_file = self.check_file.clone();
        let pro_chunk = self.pro_chunk_check_file.clone();
        let mut finish = self.prochunks_finish_set_and_quit;
        let action = check_for_pro_chunks_quit(
            Some(&check_file),
            Some(&pro_chunk),
            false,
            false,
            &mut finish,
        );
        self.prochunks_finish_set_and_quit = finish;
        self.process_quit_action(&action, "");
    }
}

impl Brt {
    /// `runOneProcess` (`IMOD/pysrc/batchruntomo:1067`).
    pub fn run_one_process(
        &mut self,
        comfile: &str,
        single: bool,
        using_gpu: bool,
        message: &str,
        use_most_cpus: bool,
    ) -> i32 {
        let start_time = py_time();
        if (single || (!comfile.contains("sirt") && !comfile.contains("ctf3d")))
            && !Path::new(comfile).exists()
        {
            self.abort_set(&format!("Command file {comfile} does not exist"));
            return 1;
        }

        // Check for quitting then compose the rest of the command array
        self.check_for_quit();
        let mut com_array: Vec<OsString> = vec![
            OsString::from("-n"),
            OsString::from(self.niceness.to_string()),
        ];
        let outfile = format!("processchunks{}.out", self.axis_let);
        if !self.remote_data_dir.is_empty() {
            com_array.push(OsString::from("-w"));
            com_array.push(OsString::from(self.remote_data_dir.clone()));
        }
        let mut machines = self.cpu_list.clone();
        let comroot = match comfile.rfind('.') {
            Some(index) if index > 0 => comfile[..index].to_owned(),
            _ => comfile.to_owned(),
        };
        let mut mess = if !message.is_empty() {
            format!("{message} (running {comfile}")
        } else {
            format!("Running {comfile}")
        };

        // single case no GPU:  regular run if no queue
        //                      queue run if not running on queue already
        //                      regular run if queue but running on queue already
        // Single case GPU:     simple run if no queue with gpu list
        //                      use gpuQueueCommand instead, no G option needed
        // chunk case, no GPU:  cpuList if no queue
        //                      queue run if queue
        // chunk case, GPU:     regular run if no queue with -G and current GPU list
        //                      if gpusPerClusterJob and no gpuQueueCommand, regular run with -G
        //                      and GPU list made of localhost entries
        //                      if gpuQueueCommand, run on that
        let comuse;
        if single {
            // Figure out the machine to use for single and its thread limit if possible
            let mut threads = self.first_cpu_limit;
            if use_most_cpus {
                machines = self.top_cpu_machine.clone();
                threads = self.top_cpu_limit;
            } else if self.use_first_cpu_for_single == 0 {
                machines = "1".to_owned();
                threads = self.local_cpu_limit;
            }
            com_array.push(OsString::from("-s"));
            com_array.push(OsString::from("-e"));
            com_array.push(OsString::from("1"));
            if threads > 0 {
                com_array.push(OsString::from("-O"));
                com_array.push(OsString::from(threads.to_string()));
            }
            comuse = comfile.to_owned();
        } else {
            mess += " in multiple chunks";
            comuse = comroot.clone();
        }

        if !self.queue_command.is_empty() && !using_gpu && !(single && self.running_on_queue) {
            machines = self.queue_command.clone();
            com_array.push(OsString::from("-q"));
            com_array.push(OsString::from(self.max_queue_jobs.to_string()));
        }

        if using_gpu {
            if !self.gpu_queue_command.is_empty() {
                machines = self.gpu_queue_command.clone();
                com_array.push(OsString::from("-q"));
                com_array.push(OsString::from(self.max_gpu_queue_jobs.to_string()));
            } else {
                machines = self.my_gpu_list.clone();
                if machines != "1" {
                    com_array.push(OsString::from("-G"));
                }
            }
            mess += " using GPU";
        }

        if !message.is_empty() {
            mess += ")";
        }
        mess += "   [brt2]";
        if machines.is_empty() {
            machines = "1".to_owned();
        }
        com_array.push(OsString::from(machines));
        com_array.push(OsString::from(comuse));

        // Run the process detached
        self.prn_log(&mess, "\n", true);
        unsafe { std::env::set_var("PIP_PRINT_ENTRIES", "1") };
        let check_file = self.check_file.clone();
        let pro_chunk = self.pro_chunk_check_file.clone();
        let (error, finished, top_quit, _num_done, mut mess) = run_processchunks(
            &com_array,
            &outfile,
            Some(&check_file),
            &pro_chunk,
            false,
            false,
        );
        unsafe { std::env::set_var("PIP_PRINT_ENTRIES", "0") };
        if finished == -1 {
            self.prn_log(&mess, "\n", false);
        }

        self.process_quit_action(&top_quit, &mess);
        if error < 0 {
            self.prn_log(&format!("ERROR: {mess}"), "\n", false);
            self.abort_set(&format!("Cannot start processchunks to run {comfile}"));
            return 1;
        }
        if error != 0 {
            self.abort_set(&mess);
            return 1;
        }

        let mut do_print = true;
        if finished == 1 {
            for comskip in HANDLING_MESSAGES {
                if comroot.contains(comskip) {
                    do_print = false;
                    break;
                }
            }
        }

        // Get error and warnings from logs
        if single && do_print {
            let tags = [
                MessageTag("ERROR:", 0, None),
                MessageTag("WARNING:", 0, None),
            ];
            self.print_tagged_messages_file(&(comroot.clone() + ".log"), &tags);
        } else {
            let tags = [MessageTag("WARNING:", 0, None)];
            self.print_tagged_messages_file(&outfile, &tags);
        }

        // After loop, one last check for quit and some more set aborts
        self.check_for_quit();
        if finished == -1 {
            self.abort_set(&format!("An error occurred running {comfile}"));
        } else if finished == -2 {
            self.abort_set(
                "Strangely, processchunks quit but Q was not detected in the check file",
            );
        } else if finished == 1 {
            let (minutes, seconds, frac) = elapsed_time_components(start_time);
            self.prn_log(
                &format!(
                    "Successfully finished {comfile}   in {minutes:02}:{seconds:02}.{frac}   [brt3]\n"
                ),
                "\n",
                true,
            );
        }
        let _ = &mut mess;
        i32::from(finished < 0)
    }

    /// `manageGPUallocation` (`IMOD/pysrc/batchruntomo:1185`).
    pub fn manage_gpu_allocation(&mut self) -> i32 {
        let check_interval = 5.;
        let warn_interval = 60. * 5.;
        self.my_parallel_gpu = self.parallel_gpu;
        self.my_gpu_list = self.gpu_list.clone();
        if self.use_gpu == 0
            || self.max_parallel_gpus == 0
            || !self.queue_command.is_empty()
            || self.cores_per_cluster_job != 0
        {
            return 0;
        }

        // Loop forever!
        let start_time = py_time();
        let mut last_warn = start_time;
        loop {
            let command = format!(
                "gpuallocator -root {} -full \"{}\" -comm \"{}\" -max {} -con {} -pid {}",
                self.parallel_root,
                self.gpu_list,
                self.starting_dir,
                self.max_parallel_gpus,
                self.hostname,
                self.my_pid
            );
            match run_cmd(&command, None, None, None, &[]) {
                Ok(lines) => {
                    let alloc_lines = lines.unwrap_or_default();
                    self.my_parallel_gpu = 0;
                    self.my_gpu_list = String::new();
                    if alloc_lines.is_empty() {
                        self.abort_set("Gpuallocator gave no output, cannot proceed");
                        return 1;
                    }

                    // Got an allocation: set variables for using them appropriately including
                    // flag that they need to be released
                    self.need_gpu_release = true;
                    for line in &alloc_lines {
                        self.my_parallel_gpu += 1;
                        if !self.my_gpu_list.is_empty() {
                            self.my_gpu_list += ",";
                        }
                        self.my_gpu_list += line.trim_end_matches(['\r', '\n']);
                    }

                    let elapsed = (py_time() - start_time) as i64;
                    let count = alloc_lines.len();
                    let list = self.my_gpu_list.clone();
                    self.prn_log(
                        &format!(
                            "Received an allocation of {count} GPUs in {elapsed} seconds ({list})"
                        ),
                        "\n",
                        false,
                    );
                    return 0;
                }
                Err(_) => {
                    // Got an error: See if it No GPUs and wait, or report error and abort
                    let err_strings = get_err_strings();
                    if err_strings.is_empty() {
                        self.abort_set(
                            "Gpuallocator gave an error with no message, cannot proceed",
                        );
                        return 1;
                    }
                    let mut found = false;
                    for line in &err_strings {
                        if line.contains("[GPA1]") {
                            let nowt = py_time();
                            if nowt - last_warn > warn_interval {
                                let minutes = ((nowt - start_time) / 60.).round() as i64;
                                self.warning(
                                    &[format!(
                                        "Have not been able to get a GPU allocation for {minutes} minutes"
                                    )],
                                    true,
                                );
                                last_warn = nowt;
                            }
                            found = true;
                            break;
                        }
                    }
                    if !found {
                        for line in &err_strings {
                            let line = line.clone();
                            self.prn_log(&line, "\n", false);
                        }
                        self.abort_set("Gpuallocator exited with an error, cannot proceed");
                        return 1;
                    }

                    std::thread::sleep(std::time::Duration::from_secs_f64(check_interval));
                    continue;
                }
            }
        }
    }

    /// `releaseGPUallocation` (`IMOD/pysrc/batchruntomo:1249`).
    pub fn release_gpu_allocation(&mut self) {
        if !self.need_gpu_release {
            return;
        }
        let command = format!(
            "gpuallocator -root {} -comm \"{}\" -con {} -pid {}",
            self.parallel_root, self.starting_dir, self.hostname, self.my_pid
        );
        if run_cmd(&command, None, None, None, &[]).is_err() {
            let err_strings = get_err_strings();
            let mut mess = ": ".to_owned();
            if !err_strings.is_empty() {
                mess += &err_strings[err_strings.len() - 1];
            }
            self.warning(
                &[format!("Error trying to release GPU allocation{mess}")],
                true,
            );
        }
        self.need_gpu_release = false;
    }

    /// `numberOfProcessingUnits` (`IMOD/pysrc/batchruntomo:1266`).
    pub fn number_of_processing_units(&mut self) -> i64 {
        if self.manage_gpu_allocation() != 0 {
            return -1;
        }
        let mut num_proc = 1;
        if self.my_parallel_gpu > 1 {
            num_proc = self.my_parallel_gpu;
        } else if !self.queue_command.is_empty() && self.gpu_queue_command.is_empty() {
            num_proc = self.max_queue_jobs as i64;
        } else if self.parallel_cpu != 0 && self.use_gpu == 0 {
            num_proc = self.parallel_cpu;
        }
        num_proc
    }

    /// `makeAndRunOneCom` (`IMOD/pysrc/batchruntomo:1281`).
    pub fn make_and_run_one_com(
        &mut self,
        comlines: &[String],
        comfile: &str,
        message: &str,
        use_gpu: bool,
        skip_run: bool,
    ) -> i32 {
        let mut comlines = comlines.to_vec();
        comlines.insert(0, format!("OutputFile\t{comfile}"));
        comlines.push(format!("NamingStyle\t{}", self.name_style));
        comlines.push(format!(
            "StackExtension\t{}",
            &self.stack_extension[1.min(self.stack_extension.len())..]
        ));
        if run_cmd(
            "makecomfile -StandardInput",
            Some(&comlines),
            None,
            None,
            &[],
        )
        .is_err()
        {
            self.report_imod_error(Some(&format!("Error making {comfile}")));
            return 1;
        }

        if skip_run {
            return 0;
        }

        if self.run_one_process(comfile, true, use_gpu, message, false) != 0 {
            return 1;
        }
        0
    }

    /// `modifyWriteAndRunCom` (`IMOD/pysrc/batchruntomo:1301`).
    pub fn modify_write_and_run_com(
        &mut self,
        comfile: &str,
        sedcom: &[String],
        in_lines: Option<&[String]>,
        message: &str,
        skip_run: bool,
    ) -> i32 {
        let in_lines = match in_lines.filter(|lines| !lines.is_empty()) {
            Some(lines) => lines.to_vec(),
            None => {
                let lines = self.read_text_file_report_err(comfile, None);
                if lines.is_empty() {
                    return 1;
                }
                lines
            }
        };

        if pysed(
            sedcom,
            PysedSrc::Lines(&in_lines),
            Some(comfile),
            false,
            '/',
            true,
        )
        .is_err()
        {
            self.abort_set(&format!("Error modifying {comfile}"));
            return 1;
        }
        if skip_run {
            return 0;
        }
        if self.run_one_process(comfile, true, false, message, false) != 0 {
            return 1;
        }
        0
    }

    /// `getOneValueAfterToken` (`IMOD/pysrc/batchruntomo:1319`).
    ///
    /// The source's `junk = int('=')` raises `ValueError` when the token is
    /// absent; the caller catches that, so `Err(())` stands for the raise.
    pub fn get_one_value_after_token(
        &self,
        line: &str,
        token: char,
        val_type: i32,
    ) -> Result<f64, ()> {
        let ind = match line.find(token) {
            Some(index) => index + 1,
            None => return Err(()),
        };
        if val_type == INT_VALUE {
            return line[ind..]
                .trim()
                .parse::<i64>()
                .map(|value| value as f64)
                .map_err(|_| ());
        }
        line[ind..].trim().parse::<f64>().map_err(|_| ())
    }

    /// `analyzeAlignLog` (`IMOD/pysrc/batchruntomo:1330`).
    pub fn analyze_align_log(
        &mut self,
        two_surf: bool,
        angle_arr: &mut [f64; 8],
        no_log_ok: bool,
    ) -> i32 {
        // Fraction of shift to subtract from fiducial-based thickness to get reconstruction
        // thickness, and minimum on each side before doing that
        let thick_shift_frac = 1.;
        let min_fid_adjust_thick = 4;
        angle_arr[0..5].fill(0.);
        let log_name = format!("align{}.log", self.axis_let);
        if no_log_ok && !Path::new(&log_name).exists() {
            return 0;
        }

        let loglines = match run_cmd(
            &format!("alignlog -a align{}.log", self.axis_let),
            None,
            None,
            None,
            &[],
        ) {
            Ok(lines) => lines.unwrap_or_default(),
            Err(_) => {
                self.report_imod_error(Some("Extracting angle analysis from align log"));
                return 1;
            }
        };

        let tags = [
            "Total tilt angle change",
            "X axis tilt needed",
            "Unbinned thickness",
            "Incremental unbinned shift",
            "Total unbinned shift",
        ];
        let mut num_bot: i64 = -1;
        let mut num_top: i64 = -1;
        for line in &loglines {
            if line.contains("# of points") {
                // There can be one or three of these entries, so the second one replaces
                // numBot and the third one gives numTop
                let num = match self.get_one_value_after_token(line, '=', INT_VALUE) {
                    Ok(value) => value as i64,
                    Err(()) => {
                        self.abort_set("Error extracting information from align log");
                        return 1;
                    }
                };
                if num_bot < 0 {
                    num_bot = num;
                } else if num_top < 0 {
                    num_bot = num;
                    num_top = 0;
                } else {
                    num_top = num;
                }
            }
            for ind in 0..tags.len() {
                if line.contains(tags[ind]) {
                    match self.get_one_value_after_token(line, '=', FLOAT_VALUE) {
                        Ok(value) => angle_arr[ind] = value,
                        Err(()) => {
                            self.abort_set("Error extracting information from align log");
                            return 1;
                        }
                    }
                }
            }
        }

        // Adjust the thickness by the X-tilt if no tilt is to be used: the spacing itself
        // is increased by cosine of the angle, and the pitch adds tangent times extent in Y
        if self.no_xaxis_tilt != 0 {
            angle_arr[2] = angle_arr[2] / angle_arr[1].to_radians().cos()
                + 0.9 * self.raw_ysize as f64 * angle_arr[1].to_radians().abs().tan();
        }

        // For two surfaces, also compute a reconstruction thickness (thickness are floats here)
        if two_surf {
            angle_arr[5] = angle_arr[2];
            if num_bot >= min_fid_adjust_thick
                && num_top >= min_fid_adjust_thick
                && !self.center_on_gold
            {
                angle_arr[5] -= thick_shift_frac * angle_arr[4].abs();
            }
        }
        angle_arr[6] = num_bot as f64;
        angle_arr[7] = num_top.max(0) as f64;

        0
    }

    /// `montageFrameValues` (`IMOD/pysrc/batchruntomo:1397`).
    pub fn montage_frame_values(
        &self,
        nxmont: i64,
        nymont: i64,
        transpose: bool,
        nxali: i64,
        nyali: i64,
        frame_arr: &mut [i64; 10],
    ) -> i64 {
        let (nx_out, ny_out) = if transpose {
            (nymont, nxmont)
        } else {
            (nxmont, nymont)
        };

        // aligned stacksize
        let (first, second) = run_goodframe(nxali as i32, nyali as i32);
        frame_arr[0] = first as i64;
        frame_arr[1] = second as i64;
        let (first, second) = run_goodframe(nx_out as i32, ny_out as i32);
        frame_arr[6] = first as i64;
        frame_arr[7] = second as i64;
        if frame_arr[0] < 0 || frame_arr[6] < 0 {
            return frame_arr[0].min(frame_arr[6]);
        }

        // starting coordinates X and Y for blend, centered on center in old orientation
        frame_arr[2] = -(frame_arr[0] - nxmont).div_euclid(2);
        frame_arr[3] = -(frame_arr[1] - nymont).div_euclid(2);

        // ending for blend, and SUBSETSTART
        for ind in 0..2 {
            frame_arr[ind + 4] = frame_arr[ind + 2] + frame_arr[ind] - 1;
            frame_arr[ind + 8] = -(frame_arr[ind] - frame_arr[ind + 6]).div_euclid(2);
        }

        0
    }
}

impl Brt {
    /// `getOrXformBoundaryModel` (`IMOD/pysrc/batchruntomo:1425`).
    pub fn get_or_xform_boundary_model(
        &mut self,
        prefix_text: &str,
        dflt_model_suffix: &str,
        do_transfer: bool,
    ) -> (i32, String) {
        // Boundary model has to be transformed if indicated
        // A model on A raw stack has to be transformed to preali
        let a_raw_bound = self.lookup_directive(prefix_text, "rawBoundaryModel", 0, STRING_VALUE);
        let mut a_boundary_mod =
            self.lookup_directive(prefix_text, "prealiBoundaryModel", 0, STRING_VALUE);
        let raw_bound;
        let mut boundary_mod;
        if self.axis_ind == 0 {
            raw_bound = a_raw_bound.clone();
            boundary_mod = a_boundary_mod.text().to_owned();
        } else {
            // If there is an A raw model and not an A preali model, set default name
            // and then set the variable that it actually exists
            boundary_mod = String::new();
            raw_bound = PyVal::Str(String::new());
            if a_boundary_mod.is_none() && !a_raw_bound.is_none() {
                a_boundary_mod = PyVal::Str(format!("{}a{dflt_model_suffix}", self.set_name));
            }
            let have_aaxis_bound =
                !a_boundary_mod.is_none() && Path::new(a_boundary_mod.text()).exists();

            // A raw model for B has to be specific to the B axis
            let key = format!("{prefix_text}.b.rawBoundaryModel");
            let mut raw_bound_b = String::new();
            if self.all_directives[BAT_DICT_IND].contains(&key) {
                raw_bound_b = self.all_directives[BAT_DICT_IND]
                    .get(&key)
                    .unwrap()
                    .0
                    .clone();
            }
            let key = format!("{prefix_text}.b.prealiBoundaryModel");
            if self.all_directives[BAT_DICT_IND].contains(&key) {
                boundary_mod = self.all_directives[BAT_DICT_IND]
                    .get(&key)
                    .unwrap()
                    .0
                    .clone();
            }
            if !boundary_mod.is_empty() {
                self.prn_log("GOT B MODEL", "\n", false);
            }

            // But if there is not a raw model for B, see if there was one for A
            // that can be transferred with the transform
            if raw_bound_b.is_empty() && boundary_mod.is_empty() && have_aaxis_bound && do_transfer
            {
                boundary_mod = format!("{}{dflt_model_suffix}", self.data_name);
                if self.patch_track {
                    // Patch tracking: run transferfid to find best matching views and transform
                    let mut comlines = vec![format!("RootNameOfDataFiles       {}", self.set_name)];
                    comlines.extend(self.later_com_directives(0));
                    comlines.push(format!(
                        "OneParameterChange {COM_PREFIX}transferfid.transferfid.BoundaryModel={}",
                        a_boundary_mod.text()
                    ));
                    comlines.push(format!(
                        "OneParameterChange {COM_PREFIX}transferfid.transferfid.SeedModel={boundary_mod}"
                    ));
                    comlines.push(format!(
                        "OneParameterChange {COM_PREFIX}transferfid.transferfid.CorrespondingCoordFile="
                    ));
                    let comfile = format!("transferfid{}", self.com_ext);
                    if self.make_and_run_one_com(
                        &comlines,
                        &comfile,
                        "Transferring boundary model from A to B axis",
                        false,
                        false,
                    ) != 0
                    {
                        return (1, String::new());
                    }
                } else {
                    // Fiducials: try to get the change in Z from the transfer log
                    let mut delta_z: i64 = 0;
                    if let Ok(loglines) = read_text_file("transferfid.log", None, true, None) {
                        let mut from_view: i64 = 0;
                        let mut to_view: i64 = 0;
                        for line in &loglines {
                            if line.contains("from view") && line.contains("to view") {
                                let lsplit: Vec<&str> = line.split_whitespace().collect();
                                let mut ok = true;
                                for ind in 0..lsplit.len().saturating_sub(2) {
                                    if lsplit[ind] == "from" && lsplit[ind + 1] == "view" {
                                        match lsplit[ind + 2].parse::<i64>() {
                                            Ok(value) => from_view = value,
                                            Err(_) => {
                                                ok = false;
                                                break;
                                            }
                                        }
                                    }
                                    if lsplit[ind] == "to" && lsplit[ind + 1] == "view" {
                                        match lsplit[ind + 2].parse::<i64>() {
                                            Ok(value) => to_view = value,
                                            Err(_) => {
                                                ok = false;
                                                break;
                                            }
                                        }
                                    }
                                }
                                if ok {
                                    delta_z = to_view - from_view;
                                }
                            }
                        }
                    }

                    // Then run imodtrans to transform model
                    let preali = dataset_filename(".preali", None, None);
                    match get_mrc_size(&preali) {
                        Ok((panx, pany, panz)) => {
                            let comstr = format!(
                                "imodtrans -2 \"{}_AtoB.xf\" -l 0 -n {panx},{pany},{panz} -tz {delta_z} \"{}\" \"{boundary_mod}\"",
                                self.set_name,
                                a_boundary_mod.text()
                            );
                            if run_cmd(&comstr, None, None, None, &[]).is_err() {
                                self.report_imod_error(Some(
                                    "Failed to transform boundary model from A to B",
                                ));
                                return (1, String::new());
                            }
                        }
                        Err(_) => {
                            self.report_imod_error(Some(
                                "Failed to transform boundary model from A to B",
                            ));
                            return (1, String::new());
                        }
                    }
                }
            }
            if !raw_bound_b.is_empty() && boundary_mod.is_empty() {
                boundary_mod = format!("{}{dflt_model_suffix}", self.data_name);
                if self.transform_raw_boundary_model(&raw_bound_b.clone(), &boundary_mod.clone())
                    != 0
                {
                    return (1, String::new());
                }
            }
            return (0, boundary_mod);
        }

        // Transform a raw model
        if raw_bound.truthy() && boundary_mod.is_empty() {
            boundary_mod = format!("{}{dflt_model_suffix}", self.data_name);
            let from = raw_bound.text().to_owned();
            let to = boundary_mod.clone();
            if self.transform_raw_boundary_model(&from, &to) != 0 {
                return (1, String::new());
            }
        }
        (0, boundary_mod)
    }

    /// `comAndProcessForAlignedStack` (`IMOD/pysrc/batchruntomo:1514`).
    pub fn com_and_process_for_aligned_stack(&self, prefix: &str) -> (String, String) {
        if self.if_montage {
            return (format!("{prefix}blend"), "blendmont".to_owned());
        }
        (format!("{prefix}newst"), "newstack".to_owned())
    }

    /// `splitAndRunTilt` (`IMOD/pysrc/batchruntomo:1525`).
    pub fn split_and_run_tilt(&mut self, comfile: &str, message: &str, num_proc: i64) -> i32 {
        let mut num_proc = num_proc;
        if num_proc == 0 {
            num_proc = self.number_of_processing_units();
            if num_proc < 1 {
                return 1;
            }
        }

        if num_proc > 1
            && run_cmd(
                &format!("splittilt -n {num_proc} {comfile}"),
                None,
                None,
                None,
                &[],
            )
            .is_err()
        {
            self.report_imod_error(Some(&format!("Error running splittilt on {comfile}")));
            self.release_gpu_allocation();
            return 1;
        }

        let use_gpu = self.use_gpu != 0;
        let err = self.run_one_process(comfile, num_proc < 2, use_gpu, message, false);
        self.release_gpu_allocation();
        err
    }

    /// `runFindSection` (`IMOD/pysrc/batchruntomo:1547`).
    #[allow(clippy::too_many_arguments)]
    pub fn run_find_section(
        &mut self,
        filename: &str,
        num_scales: i64,
        box_size: i64,
        pitch_mod: Option<&str>,
        top_bots: Option<&mut [i64; 6]>,
        block: Option<i64>,
        failure_ok: bool,
        com_root: &str,
    ) -> i32 {
        let limit_keys = ["Median Z values", "autopatchfit combine", "Absolute limits"];
        let mut com_lines = vec![
            "$findsection -StandardInput".to_owned(),
            format!("TomogramFile {filename}"),
            format!("NumberOfDefaultScales {num_scales}"),
            format!("SizeOfBoxesInXYZ {box_size},1,{box_size}"),
        ];
        if let Some(pitch_mod) = pitch_mod {
            com_lines.push(format!("TomoPitchModel {pitch_mod}"));
            com_lines.push("NumberOfSamples 5".to_owned());
            if let Some(block) = block {
                com_lines.push(format!("BlockSize {block}"));
            }
        }
        let com_name = format!("{com_root}{}", self.com_ext);
        if self.write_text_file_report_err(&com_name, &com_lines) != 0 {
            return 1;
        }
        let mess = if pitch_mod.is_some() {
            "Getting a model for tomogram positioning"
        } else {
            "Finding Z limits of the material in the tomogram"
        };
        self.suppress_abort = failure_ok;
        let err = self.run_one_process(&com_name, true, false, mess, false);
        self.suppress_abort = false;
        if err != 0 {
            return -1;
        }
        if let Some(top_bots) = top_bots {
            let loglines = self.read_text_file_report_err(&format!("{com_root}.log"), None);
            if loglines.is_empty() {
                return 1;
            }
            for line in &loglines {
                for ind in 0..3 {
                    if line
                        .to_lowercase()
                        .contains(&limit_keys[ind].to_lowercase())
                    {
                        let lsplit: Vec<&str> = line.split_whitespace().collect();
                        let parsed = lsplit
                            .len()
                            .checked_sub(2)
                            .and_then(|index| lsplit[index].parse::<i64>().ok())
                            .zip(lsplit.last().and_then(|value| value.parse::<i64>().ok()));
                        match parsed {
                            Some((first, second)) => {
                                top_bots[2 * ind] = first;
                                top_bots[2 * ind + 1] = second;
                            }
                            None => {
                                let line = line.clone();
                                self.abort_set(&format!("Error converting Z limits in: {line}"));
                                return 1;
                            }
                        }
                    }
                }
            }
        }

        0
    }

    /// `parseTomopitchLog` (`IMOD/pysrc/batchruntomo:1589`).
    pub fn parse_tomopitch_log(&mut self, angle_arr: &mut [f64; 4]) -> i32 {
        let pitch_file = format!("tomopitch{}.log", self.axis_let);
        let tags = [
            "x-tilted  lines",
            "X axis tilt -",
            "Angle offset -",
            "Z shift -",
        ];
        let mut original = [0f64; 4];
        let mut untilted = [0f64; 4];
        let mut got_all_lines = 0;

        if self.pos_sample_type <= 0 {
            return 1;
        }
        if !Path::new(&pitch_file).exists() {
            return 1;
        }
        let pitch_lines = self.read_text_file_report_err(&pitch_file, None);
        if pitch_lines.is_empty() {
            return -1;
        }

        for line in &pitch_lines {
            if line.contains("ERROR: ") {
                return 2;
            }

            // Look for the standard tagged lines but also get original values
            for ind in 0..4 {
                if line.contains(tags[ind]) {
                    let lsplit: Vec<&str> = line.split_whitespace().collect();
                    let parsed = lsplit.last().and_then(|value| value.parse::<f64>().ok());
                    match parsed {
                        Some(value) => {
                            if ind != 0 {
                                angle_arr[ind] = value;
                                if self.no_xaxis_tilt != 0 {
                                    for word in 0..lsplit.len() {
                                        if lsplit[word].contains("Original:")
                                            && word + 1 < lsplit.len()
                                        {
                                            match lsplit[word + 1].parse::<f64>() {
                                                Ok(value) => original[ind] = value,
                                                Err(_) => {
                                                    let tag = tags[ind].replace(" -", "");
                                                    self.abort_set(&format!(
                                                        "Error converting tomopitch output of {tag} to a number"
                                                    ));
                                                    return -1;
                                                }
                                            }
                                        }
                                    }
                                }
                            } else if lsplit
                                .last()
                                .and_then(|value| value.parse::<i64>().ok())
                                .is_some()
                            {
                                angle_arr[ind] = value.trunc();
                            } else {
                                let tag = tags[ind].replace(" -", "");
                                self.abort_set(&format!(
                                    "Error converting tomopitch output of {tag} to a number"
                                ));
                                return -1;
                            }
                        }
                        None => {
                            let tag = tags[ind].replace(" -", "");
                            self.abort_set(&format!(
                                "Error converting tomopitch output of {tag} to a number"
                            ));
                            return -1;
                        }
                    }
                }
            }

            // But if want no Xtilt, look for the line that starts that output,
            // Then process the specific lines after it with the untilted values
            if self.no_xaxis_tilt != 0 && line.contains("all line pairs") {
                got_all_lines = 1;
            } else if got_all_lines != 0 {
                if got_all_lines == 2 {
                    let ind = line.find("add");
                    let to_ind =
                        ind.and_then(|ind| line[ind + 3..].find("to").map(|at| at + ind + 3));
                    match ind
                        .zip(to_ind)
                        .and_then(|(ind, to_ind)| line[ind + 3..to_ind].trim().parse::<f64>().ok())
                    {
                        Some(value) => untilted[2] = value,
                        None => {
                            self.abort_set(
                                "Error converting tomopitch output for non x-tilted lines  to a number",
                            );
                            return -1;
                        }
                    }
                }
                if got_all_lines == 3 {
                    let lsplit: Vec<&str> = line.split_whitespace().collect();
                    let ind = line.find("shift of");
                    let to_ind = ind.and_then(|ind| line[ind..].find(';').map(|at| at + ind));
                    let values = lsplit
                        .last()
                        .and_then(|value| value.parse::<i64>().ok())
                        .zip(ind.zip(to_ind).and_then(|(ind, to_ind)| {
                            line[ind + 8..to_ind].trim().parse::<f64>().ok()
                        }));
                    match values {
                        Some((first, second)) => {
                            untilted[0] = first as f64;
                            untilted[3] = second;
                        }
                        None => {
                            self.abort_set(
                                "Error converting tomopitch output for non x-tilted lines  to a number",
                            );
                            return -1;
                        }
                    }
                }
                got_all_lines += 1;
            }
        }

        // Add original and no X tilt results (returning original X tilt is consistent with
        // the other returned values)
        if self.no_xaxis_tilt != 0 && got_all_lines != 0 {
            for ind in 0..4 {
                angle_arr[ind] = original[ind] + untilted[ind];
            }
        }

        0
    }

    /// `getReconTypes` (`IMOD/pysrc/batchruntomo:1658`).
    pub fn get_recon_types(&self) -> (bool, bool, bool) {
        let do_sirt = self
            .lookup_directive(
                &format!("{RUNTIME_PREFIX}Reconstruction"),
                "useSirt",
                0,
                BOOL_VALUE,
            )
            .truthy();
        let do_bpalso = self
            .lookup_directive(
                &format!("{RUNTIME_PREFIX}Reconstruction"),
                "doBackprojAlso",
                0,
                BOOL_VALUE,
            )
            .truthy();
        let do_reg_bpalso = self
            .lookup_directive(
                &format!("{RUNTIME_PREFIX}Reconstruction"),
                "doRegularBPalsoIfFakeSIRT",
                0,
                BOOL_VALUE,
            )
            .truthy();
        let do_fake_sirt = self
            .lookup_directive(
                &format!("{COM_PREFIX}tilt"),
                "tilt.FakeSIRTiterations",
                0,
                STRING_VALUE,
            )
            .truthy();
        let do_both = (do_sirt && do_bpalso) || (do_fake_sirt && (do_reg_bpalso || do_bpalso));
        (do_sirt, do_fake_sirt, do_both)
    }

    /// `getSIRTrecName` (`IMOD/pysrc/batchruntomo:1674`).
    pub fn get_sirt_rec_name(
        &self,
        rec_root: &str,
    ) -> (bool, bool, bool, Option<String>, Option<String>) {
        let (do_sirt, do_fake_sirt, do_both) = self.get_recon_types();

        let mut rec_name = None;
        let mut fake_name = None;
        if do_sirt {
            let leave_list = self.lookup_directive(
                &format!("{COM_PREFIX}sirtsetup"),
                "sirtsetup.LeaveIterations",
                0,
                STRING_VALUE,
            );
            if leave_list.truthy() {
                let replaced = leave_list.text().replace('-', ",");
                let lsplit: Vec<&str> = replaced.split(',').collect();
                let mut last_one = lsplit[lsplit.len() - 1].to_owned();
                if last_one.len() < 2 {
                    last_one = "0".to_owned() + &last_one;
                }
                rec_name = Some(dataset_filename(
                    &format!(".srec{last_one}"),
                    Some(rec_root),
                    None,
                ));
            } else {
                // The source returns the integer 1 when the name cannot be made.
                rec_name = Some(String::from("1"));
            }
        }

        if do_fake_sirt {
            fake_name = Some(dataset_filename(".rec", Some(rec_root), None));
            if do_both {
                fake_name = Some(dataset_filename(".slfrec", Some(rec_root), None));
            }
        }

        (do_sirt, do_fake_sirt, do_both, rec_name, fake_name)
    }

    /// `runClipStats` (`IMOD/pysrc/batchruntomo:1700`).
    pub fn run_clip_stats(
        &mut self,
        command: &str,
        stack_suffix: &str,
        descrip: &str,
    ) -> Vec<String> {
        let comfile = format!("{}{stack_suffix}.st.stats{}", self.data_name, self.com_ext);
        if self.write_text_file_report_err(&comfile, &[command.to_owned()]) != 0 {
            return Vec::new();
        }
        let err = self.run_one_process(
            &comfile,
            true,
            false,
            &format!("Getting statistics for {descrip} stack"),
            false,
        );
        cleanup_files(&[comfile.clone()]);
        if err != 0 {
            return Vec::new();
        }

        // Extract the summary lines
        let log = format!("{}{stack_suffix}.st.stats.log", self.data_name);
        self.read_text_file_report_err(&log, None)
    }

    /// `needStep` (`IMOD/pysrc/batchruntomo:1715`).
    pub fn need_step(&self, step: f64) -> bool {
        step >= self.starting_step - 0.005 && step <= self.ending_step + 0.005
    }

    /// `reportReachedStep` (`IMOD/pysrc/batchruntomo:1720`).
    pub fn report_reached_step(&mut self, step: f64) {
        let reach_points = [0., 5., 6., 7., DETECT_3D_STEP_NUM, 13., 14.];
        if reach_points.contains(&step) && self.need_step(step) {
            let text = if step.fract() == 0.0 {
                format!("{}", step as i64)
            } else {
                py_str_float(step)
            };
            self.prn_log(&format!("Reached step {text}"), "\n", false);
            self.prn_log("", "\n", true);
        }
    }

    /// `replaceOrRunAfterStep` (`IMOD/pysrc/batchruntomo:1730`).
    pub fn replace_or_run_after_step(&mut self, step: f64, run_after: bool) -> i32 {
        if !self.need_step(step) {
            return 0;
        }
        let direc = if run_after {
            "unAfterStep"
        } else {
            "eplaceStep"
        };
        let good_ret = if run_after { 0 } else { -1 };
        let step_text = if step.fract() == 0.0 {
            format!("{}", step as i64)
        } else {
            py_str_float(step)
        };
        let comfile = format!("r{direc}{step_text}{}", self.axis_com);
        let command = self.lookup_directive(
            &format!("{RUNTIME_PREFIX}R{direc}"),
            &step_text,
            0,
            STRING_VALUE,
        );
        if !command.truthy() {
            return 0;
        }
        let command = command.text().replace("%{setname}", &self.data_name);
        if self.write_text_file_report_err(&comfile, &[format!("${command}")]) != 0 {
            return 1;
        }
        if self.run_one_process(&comfile, true, false, "", false) != 0 {
            return 1;
        }
        good_ret
    }
}

// SINGLE-CALL FUNCTIONS FOR INITIAL STEPS

impl Brt {
    /// `processValidationFile` (`IMOD/pysrc/batchruntomo:1753`).
    pub fn process_validation_file(&mut self) {
        let com_pref = &COM_PREFIX[..COM_PREFIX.len() - 1];
        let run_pref = &RUNTIME_PREFIX[..RUNTIME_PREFIX.len() - 1];
        let text = match std::fs::read_to_string(&self.validate_file) {
            Ok(text) => text,
            Err(_) => {
                self.exit_error(&format!("Opening {}", self.validate_file));
            }
        };

        // `csv.reader` over the file: comma separated, `"` quoting a field and
        // `""` an embedded quote, and a newline inside quotes continuing a row.
        let mut valid_lines: Vec<Vec<String>> = Vec::new();
        let mut row: Vec<String> = Vec::new();
        let mut field = String::new();
        let mut in_quotes = false;
        let mut characters = text.chars().peekable();
        let mut any = false;
        while let Some(character) = characters.next() {
            any = true;
            if in_quotes {
                if character == '"' {
                    if characters.peek() == Some(&'"') {
                        characters.next();
                        field.push('"');
                    } else {
                        in_quotes = false;
                    }
                } else {
                    field.push(character);
                }
            } else if character == '"' {
                in_quotes = true;
            } else if character == ',' {
                row.push(std::mem::take(&mut field));
            } else if character == '\n' {
                row.push(std::mem::take(&mut field));
                valid_lines.push(std::mem::take(&mut row));
                any = false;
            } else if character != '\r' {
                field.push(character);
            }
        }
        if any || !field.is_empty() {
            row.push(field);
            valid_lines.push(row);
        }

        for line in &valid_lines {
            if line.len() > 1 && !line[1].is_empty() {
                let lsplit: Vec<&str> = line[0].split('.').collect();
                if lsplit.len() < 2 {
                    continue;
                }
                let batch_ok = line.len() > 3 && line[3].trim() == "Y";
                let template_ok = line.len() > 4 && line[4].trim() == "Y";
                let is_bool = line.len() > 2 && line[2].to_lowercase().trim() == "bool";
                if lsplit[0] == com_pref {
                    if lsplit.len() < 4 {
                        continue;
                    }
                    let combase = lsplit[1].to_owned();
                    if !self.base_com_dict.contains(&combase) {
                        self.base_com_dict.set(&combase, combase.clone());
                        self.base_com_dict
                            .set(&(combase.clone() + "a"), combase.clone());
                        self.base_com_dict
                            .set(&(combase.clone() + "b"), combase.clone());
                    }
                    let key = format!("{}.{}.{}", lsplit[1], lsplit[2], lsplit[3]);
                    self.valid_com_dict.set(
                        &key.to_lowercase(),
                        (key.clone(), template_ok, batch_ok, is_bool),
                    );
                } else if lsplit[0] == run_pref {
                    if lsplit.len() < 4 {
                        continue;
                    }
                    let key = format!("{}.{}", lsplit[1], lsplit[3]);
                    self.valid_run_dict.set(
                        &key.to_lowercase(),
                        (key.clone(), template_ok, batch_ok, is_bool),
                    );
                } else {
                    self.valid_other_dict.set(
                        &line[0].to_lowercase(),
                        (line[0].clone(), template_ok, batch_ok, is_bool),
                    );
                }
            }
        }
    }

    /// `checkAllDirectives` (`IMOD/pysrc/batchruntomo:1797`).
    pub fn check_all_directives(&mut self, main_file: &str) -> i32 {
        let mut errors: Vec<String> = Vec::new();
        let mut source: Vec<String> = DIREC_FILE_ERROR_NAMES[0..4]
            .iter()
            .map(|name| (*name).to_owned())
            .collect();
        source.push(main_file.to_owned());
        let types = ["template", "batch directive"];
        let com_pref = &COM_PREFIX[..COM_PREFIX.len() - 1];
        let run_pref = &RUNTIME_PREFIX[..RUNTIME_PREFIX.len() - 1];
        for ind in 0..=BAT_DICT_IND {
            let mut dir_type = ind / BAT_DICT_IND;
            if main_file == "template" {
                dir_type = 0;
            }
            for direc in self.all_directives[ind].keys() {
                let messfrom = format!("irective from {} file", source[ind]);
                let dsplit: Vec<String> = direc.split('.').map(str::to_owned).collect();
                let mut bad_case = false;
                let mut ok_for_file = true;
                let mut empty_bool = false;
                let value = self.all_directives[ind].get(&direc).unwrap().0.clone();
                let val_not_bool = value != "0" && value != "1";
                if dsplit.len() < 2
                    || ((dsplit[0] == com_pref || dsplit[0] == run_pref) && dsplit.len() < 4)
                {
                    errors.push(format!("D{messfrom} too short: {direc}"));
                }
                // comparam first, start by checking the com file is in list
                else if dsplit[0] == com_pref {
                    let comlow = dsplit[1].to_lowercase();
                    if !self.base_com_dict.contains(&comlow) {
                        errors.push(format!(
                            "D{messfrom} does not include a known com file: {direc}"
                        ));
                    } else {
                        // Check for a match other than incorrect case
                        let combase = self.base_com_dict.get(&comlow).unwrap().clone();
                        let key = format!("{}.{}.{}", dsplit[1], dsplit[2], dsplit[3]);
                        if self.valid_com_dict.contains(&key.to_lowercase()) {
                            let entry = self.valid_com_dict.get(&key.to_lowercase()).unwrap();
                            bad_case = key != entry.0;
                            ok_for_file = if dir_type == 0 { entry.1 } else { entry.2 };
                            empty_bool = entry.3 && val_not_bool;
                        } else {
                            // Look for a process match next
                            let proclow = dsplit[2].to_lowercase();
                            let mut found = false;
                            for vkey in self.valid_com_dict.keys() {
                                let vsplit: Vec<String> =
                                    vkey.split('.').map(str::to_owned).collect();
                                if vsplit[0] == combase && vsplit[1].to_lowercase() == proclow {
                                    bad_case = !self.base_com_dict.contains(&dsplit[1])
                                        || vsplit[1] != dsplit[2];

                                    // If the process matches, now try to find and read autodoc
                                    let opt_file = pip_open_installed_adoc(&vsplit[1]);
                                    let mut errmess = String::new();
                                    let mut adoc_lines: Vec<String> = Vec::new();
                                    match opt_file {
                                        // The source passes this open file object to
                                        // `readTextFile`, which calls `open()` on it; the
                                        // handle's own contents are what it means to read.
                                        Some(mut file) => {
                                            use std::io::Read;
                                            let mut text = String::new();
                                            if file.read_to_string(&mut text).is_ok() {
                                                adoc_lines =
                                                    text.lines().map(str::to_owned).collect();
                                            } else {
                                                errmess = "error reading it".to_owned();
                                            }
                                        }
                                        None => {
                                            errmess =
                                                "error opening it at standard location".to_owned();
                                        }
                                    }

                                    // If no autodoc, issue a warning because we just can't tell
                                    if !errmess.is_empty() {
                                        let process = vsplit[1].clone();
                                        self.warning(
                                            &[
                                                format!("Unknown d{messfrom}: {direc}"),
                                                format!(
                                                    "  Could not check {process}.adoc because of {errmess}"
                                                ),
                                            ],
                                            true,
                                        );
                                    } else {
                                        // Otherwise look for a case-insensitive match and report a
                                        // bad case if any, otherwise report error if no match
                                        let target = format!(r"\[ *Field *= *{} *\]", dsplit[3]);
                                        let opt_match =
                                            regex::Regex::new(&format!("^(?:{target})"));
                                        let opt_lc =
                                            regex::RegexBuilder::new(&format!("^(?:{target})"))
                                                .case_insensitive(true)
                                                .build();
                                        let mut matched = false;
                                        if let (Ok(opt_match), Ok(opt_lc)) = (opt_match, opt_lc) {
                                            for line in &adoc_lines {
                                                if opt_lc.is_match(line) {
                                                    if !opt_match.is_match(line) {
                                                        bad_case = true;
                                                    }
                                                    matched = true;
                                                    break;
                                                }
                                            }
                                        }
                                        if !matched {
                                            errors.push(format!("Unknown d{messfrom}: {direc}"));
                                        }
                                    }

                                    found = true;
                                    break;
                                }
                            }
                            if !found {
                                errors.push(format!(
                                    "D{messfrom} does not include a known process: {direc}"
                                ));
                            }
                        }
                    }
                }
                // runtime next, check for a/b/any; all else must match
                else if dsplit[0] == run_pref {
                    if !["any", "a", "b"].contains(&dsplit[2].as_str()) {
                        errors.push(format!("D{messfrom} does not include a/b/any: {direc}"));
                    } else {
                        let key = format!("{}.{}", dsplit[1], dsplit[3]);
                        if self.valid_run_dict.contains(&key.to_lowercase()) {
                            let entry = self.valid_run_dict.get(&key.to_lowercase()).unwrap();
                            bad_case = key != entry.0;
                            ok_for_file = if dir_type == 0 { entry.1 } else { entry.2 };
                            empty_bool = entry.3 && val_not_bool;
                        } else {
                            errors.push(format!("Unknown d{messfrom}: {direc}"));
                        }
                    }
                }
                // Any other directives must match entirely
                else if self.valid_other_dict.contains(&direc.to_lowercase()) {
                    let entry = self.valid_other_dict.get(&direc.to_lowercase()).unwrap();
                    bad_case = direc != entry.0;
                    ok_for_file = if dir_type == 0 { entry.1 } else { entry.2 };
                    empty_bool = entry.3 && val_not_bool;
                } else {
                    errors.push(format!("Unknown d{messfrom}: {direc}"));
                }

                if bad_case {
                    errors.push(format!("D{messfrom} has incorrect case: {direc}"));
                }
                if !ok_for_file {
                    errors.push(format!(
                        "D{messfrom} is not intended for use in a {} file: {direc}",
                        types[dir_type]
                    ));
                }
                if empty_bool {
                    errors.push(format!(
                        "D{messfrom} is a boolean and must be 0 or 1: {direc}"
                    ));
                }
            }
        }

        self.print_directive_errors(&errors)
    }

    /// `scanSetupDirectives` (`IMOD/pysrc/batchruntomo:1913`).
    pub fn scan_setup_directives(&mut self) -> i32 {
        let source = DIREC_FILE_ERROR_NAMES;

        self.if_montage = false;
        self.dual_axis = false;
        self.scan_header = false;
        self.pixel_size = 0.;
        self.fid_size_nm = None;
        self.fid_size_pix = 0.;
        self.dataset_dir = String::new();
        self.set_name = String::new();
        self.defocus = -1000000.;
        if self.all_directives[BAT_DICT_IND].contains(SCOPE_TMPL_TEXT) {
            let name = self.all_directives[BAT_DICT_IND]
                .get(SCOPE_TMPL_TEXT)
                .unwrap()
                .0
                .clone();
            let err = self.read_directive_or_template(&name, TMPL_DICT_IND);
            if err.0 != 0 {
                return 1;
            }
        }
        if self.all_directives[BAT_DICT_IND].contains(SYS_TMPL_TEXT) {
            let name = self.all_directives[BAT_DICT_IND]
                .get(SYS_TMPL_TEXT)
                .unwrap()
                .0
                .clone();
            let err = self.read_directive_or_template(&name, TMPL_DICT_IND + 1);
            if err.0 != 0 {
                return 1;
            }
        }
        if self.all_directives[BAT_DICT_IND].contains(USER_TMPL_TEXT) {
            let name = self.all_directives[BAT_DICT_IND]
                .get(USER_TMPL_TEXT)
                .unwrap()
                .0
                .clone();
            let err = self.read_directive_or_template(&name, TMPL_DICT_IND + 2);
            if err.0 != 0 {
                return 1;
            }
        }

        let name_key = COPY_PREFIX.to_owned() + "name";
        if self.all_directives[BAT_DICT_IND].contains(&name_key) {
            self.set_name = self.all_directives[BAT_DICT_IND]
                .get(&name_key)
                .unwrap()
                .0
                .clone();
        }

        // This was placed into the dictionary in readDirectiveOrTemplate based on entries
        // for currentDir, deliveryDir, and makeSubDir
        if self.all_directives[BAT_DICT_IND].contains(DATA_DIR_TEXT) {
            self.dataset_dir = self.all_directives[BAT_DICT_IND]
                .get(DATA_DIR_TEXT)
                .unwrap()
                .0
                .clone();
        }
        for ind_dir in 0..=BAT_DICT_IND {
            if self.all_directives[ind_dir].contains(SCAN_HEAD_TEXT) {
                self.scan_header =
                    self.all_directives[ind_dir].get(SCAN_HEAD_TEXT).unwrap().0 == "1";
            }
            let key = COPY_PREFIX.to_owned() + "montage";
            if self.all_directives[ind_dir].contains(&key) {
                self.if_montage = self.all_directives[ind_dir].get(&key).unwrap().0 != "0";
            }
            let key = COPY_PREFIX.to_owned() + "dual";
            if self.all_directives[ind_dir].contains(&key) {
                self.dual_axis = self.all_directives[ind_dir].get(&key).unwrap().0 != "0";
            }
            for (suffix, which) in [("pixel", 0), ("gold", 1), ("defocus", 2)] {
                let dir_key = COPY_PREFIX.to_owned() + suffix;
                if self.all_directives[ind_dir].contains(&dir_key) {
                    let ftext = self.all_directives[ind_dir]
                        .get(&dir_key)
                        .unwrap()
                        .0
                        .clone();
                    if !ftext.is_empty() {
                        match ftext.parse::<f64>() {
                            Ok(value) => match which {
                                0 => self.pixel_size = value,
                                1 => self.fid_size_nm = Some(value),
                                _ => self.defocus = value,
                            },
                            Err(_) => {
                                self.abort_set(&format!(
                                    "Error converting the string \"{ftext}\" for directive {dir_key} to float in {} file",
                                    source[ind_dir]
                                ));
                                return 1;
                            }
                        }
                    }
                }
            }
        }

        if self.validation > 0 {
            return 0;
        }

        if self.pixel_size == 0. && !self.scan_header {
            self.abort_set("Pixel size missing from directives, and header is not being scanned");
            return 1;
        }
        if self.set_name.is_empty() || self.dataset_dir.is_empty() {
            self.abort_set("Set name or dataset directory missing from directives");
            return 1;
        }

        0
    }

    /// `checkDefaultsInBatchFile` (`IMOD/pysrc/batchruntomo:1989`).
    pub fn check_defaults_in_batch_file(&mut self) {
        let mut num_duplicate = 0;
        let mut direc_masked: Vec<String> = Vec::new();
        let orig_defaults = [
            ("setupset.scanHeader", "1"),
            ("runtime.Preprocessing.any.archiveOriginal", "1"),
            ("comparam.eraser.ccderaser.LineObjects", "2"),
            ("comparam.eraser.ccderaser.BoundaryObjects", "3"),
            ("comparam.eraser.ccderaser.AllSectionObjects", "1-3"),
            ("comparam.track.beadtrack.RoundsOfTracking", "4"),
            ("runtime.BeadTracking.any.numberOfRuns", "2"),
            ("comparam.align.tiltalign.RobustFitting", "1"),
            ("setupset.copyarg.gold", "0"),
        ];

        for (key, value) in orig_defaults {
            if self.all_directives[BAT_DICT_IND].contains(key)
                && value == self.all_directives[BAT_DICT_IND].get(key).unwrap().0
            {
                num_duplicate += 1;
                for ind in TMPL_DICT_IND..BAT_DICT_IND {
                    if self.all_directives[ind].contains(key)
                        && value != self.all_directives[ind].get(key).unwrap().0
                    {
                        direc_masked.push(format!("    {key} = {value}"));
                    }
                }
            }
        }

        if num_duplicate >= 4 && !direc_masked.is_empty() {
            let mut mess = vec![
                "Incorrect directives in batch file may be masking template values.".to_owned(),
                format!(
                    "{num_duplicate} directives in the batch file match the value of a batch default."
                ),
                "These came from an old starting batch file that contained batch defaults"
                    .to_owned(),
                "(from before IMOD 4.10.15).  Directives overriding a template value are:"
                    .to_owned(),
            ];
            mess.extend(direc_masked);
            mess.push(
                "Check these directives and remove any other unintended directives that".to_owned(),
            );
            mess.push(format!(
                "occur in {} from your starting batch files.",
                self.defaults_file
            ));
            self.warning(&mess, true);
        }
    }

    /// `getAxisInitialParameters` (`IMOD/pysrc/batchruntomo:2022`).
    pub fn get_axis_initial_parameters(&mut self) -> i32 {
        self.fiducialless = self
            .lookup_directive(
                &format!("{RUNTIME_PREFIX}Fiducials"),
                "fiducialless",
                0,
                BOOL_VALUE,
            )
            .int();
        let track_method = self.lookup_directive(
            &format!("{RUNTIME_PREFIX}Fiducials"),
            "trackingMethod",
            0,
            INT_VALUE,
        );
        self.patch_track = track_method.is_int() && track_method.int() == 1;

        self.erase_gold = self.lookup_directive(
            &format!("{RUNTIME_PREFIX}AlignedStack"),
            "eraseGold",
            0,
            INT_VALUE,
        );
        if self.erase_gold.is_str() {
            self.abort_set("Error converting eraseGold entry to integer");
            return 1;
        }

        self.filter_in_2d = self
            .lookup_directive(
                &format!("{RUNTIME_PREFIX}AlignedStack"),
                "filterStack",
                0,
                BOOL_VALUE,
            )
            .int();

        if self.erase_gold.int() == 1 && (self.fiducialless != 0 || self.patch_track) {
            self.abort_set(
                "Cannot erase gold with fiducials after fiducialless processing or patch tracking",
            );
            return 1;
        }

        let (do_sirt, do_fake_sirt, do_both_recons) = self.get_recon_types();
        self.do_both_recons = do_both_recons;

        // Check CTF parameters
        self.correct_ctf = self
            .lookup_directive(
                &format!("{RUNTIME_PREFIX}AlignedStack"),
                "correctCTF",
                0,
                BOOL_VALUE,
            )
            .int();
        if self.correct_ctf != 0 {
            self.autofit_ctf = self.lookup_directive(
                &format!("{RUNTIME_PREFIX}CTFplotting"),
                "autoFitRangeAndStep",
                0,
                STRING_VALUE,
            );
        }
        if self.correct_ctf != 0 && self.defocus < -999999. {
            if self.autofit_ctf.truthy() {
                if !self
                    .lookup_directive(
                        &format!("{COM_PREFIX}ctfplotter"),
                        "ctfplotter.ScanDefocusRange",
                        0,
                        STRING_VALUE,
                    )
                    .truthy()
                {
                    self.abort_set(
                        "Either defocus or a range to scan must be entered to use Ctfplotter",
                    );
                    return 1;
                }
            } else {
                self.abort_set("Defocus must be entered to correct CTF without using Ctfplotter");
                return 1;
            }
        }

        let slab = self.lookup_directive(
            &format!("{COM_PREFIX}ctf3dsetup"),
            "ctf3dsetup.SlabThicknessInNm",
            0,
            INT_VALUE,
        );
        if slab.is_str() {
            self.abort_set("Error converting ctf3dsetup.SlabThicknessInNm entry to integer");
            return 1;
        }
        self.ctf3d_slab_thick = slab.int();
        if self.ctf3d_slab_thick != 0 && self.correct_ctf == 0 {
            self.abort_set(
                "Directive for correcting CTF must also be present to do 3-D CTF-corrected reconstruction",
            );
            return 1;
        }
        if self.ctf3d_slab_thick != 0 && ((do_sirt || do_fake_sirt) && !self.do_both_recons) {
            self.abort_set(
                "You cannot do just a SIRT or SIRT-like reconstruction with 3-D CTF correction",
            );
            return 1;
        }

        self.raw_for_3dctf = self
            .lookup_directive(
                &format!("{COM_PREFIX}ctf3dsetup"),
                "ctf3dsetup.UseUnalignedImages",
                0,
                BOOL_VALUE,
            )
            .int();
        self.skip_align_stack_mods = self.ctf3d_slab_thick != 0 && !self.do_both_recons;
        let super_sample = self.lookup_directive(
            &format!("{COM_PREFIX}tilt"),
            "tilt.SuperSampleFactor",
            0,
            INT_VALUE,
        );
        if !super_sample.is_str()
            && super_sample.truthy()
            && super_sample.int() > 1
            && self
                .lookup_directive(
                    &format!("{COM_PREFIX}tilt"),
                    "tilt.ExpandInputLines",
                    0,
                    BOOL_VALUE,
                )
                .truthy()
        {
            self.abort_set(
                "You cannot expand input lines when doing 3D CTF correction with unaligned images",
            );
            return 1;
        }

        let (com_root, process) = self.com_and_process_for_aligned_stack("pre");
        let binning = self.lookup_directive(
            &format!("{COM_PREFIX}{com_root}"),
            &format!("{process}.BinByFactor"),
            0,
            INT_VALUE,
        );
        if binning.is_str() {
            self.abort_set("Error converting binning value in directive to integer");
            return 1;
        }
        self.coarse_binning = binning.int();
        if self.coarse_binning == 0 {
            self.coarse_binning = 1;
        }

        let stack = self.data_name.clone() + &self.stack_extension;
        let size = if self.if_montage {
            get_montage_size(&stack, Some(&(self.data_name.clone() + ".pl")))
        } else {
            get_mrc_size(&stack)
        };
        match size {
            Ok((nx, ny, nz)) => {
                self.raw_xsize = nx as i64;
                self.raw_ysize = ny as i64;
                self.zsize = nz as i64;
            }
            Err(_) => {
                self.report_imod_error(Some(&format!("Error getting size of {stack}")));
                return 1;
            }
        }

        0
    }

    /// `fixModelHeaderForImage` (`IMOD/pysrc/batchruntomo:2115`).
    pub fn fix_model_header_for_image(&mut self, model: &str, image: &str) -> i32 {
        if !Path::new(image).exists() || !Path::new(model).exists() {
            return 0;
        }
        let tmp_name = model.to_owned() + ".transtemp";
        if run_cmd(
            &format!("imodtrans -I \"{image}\" \"{model}\" \"{tmp_name}\""),
            None,
            None,
            None,
            &[],
        )
        .is_err()
        {
            self.warning(
                &[format!(
                    "Could not fix coordinate information in model {model}"
                )],
                true,
            );
            self.report_imod_error(None);
            return 0;
        }
        make_backup_file(model);
        if std::fs::rename(&tmp_name, model).is_err() {
            self.prn_log(
                &format!("Error renaming fixed model {tmp_name} to {model}"),
                "\n",
                false,
            );
            return 1;
        }
        self.prn_log(
            &format!("Modified model {model} to adjust for change in pixel size of raw stack"),
            "\n",
            false,
        );
        0
    }

    /// `fixAllSuppliedModelHeaders` (`IMOD/pysrc/batchruntomo:2137`).
    pub fn fix_all_supplied_model_headers(&mut self) -> i32 {
        let mut mod_set: Vec<String> = Vec::new();
        let mut b_mod_set: Vec<String> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        let mut b_seen: HashSet<String> = HashSet::new();
        let mut ax_list = vec!["a", "any"];
        let image;
        if self.dual_axis {
            image = format!("{}a{}", self.set_name, self.stack_extension);
            let name = format!("{}a.erase", self.set_name);
            if seen.insert(name.clone()) {
                mod_set.push(name);
            }
            ax_list.push("b");
            let name = format!("{}b.erase", self.set_name);
            if b_seen.insert(name.clone()) {
                b_mod_set.push(name);
            }
        } else {
            image = format!("{}{}", self.set_name, self.stack_extension);
            let name = format!("{}.erase", self.set_name);
            if seen.insert(name.clone()) {
                mod_set.push(name);
            }
        }

        for operation in ["SeedFinding", "PatchTracking"] {
            for ax_let in &ax_list {
                let key = format!("{RUNTIME_PREFIX}{operation}.{ax_let}.rawBoundaryModel");
                if self.all_directives[BAT_DICT_IND].contains(&key) {
                    let raw_bound = self.all_directives[BAT_DICT_IND]
                        .get(&key)
                        .unwrap()
                        .0
                        .clone();
                    if *ax_let == "b" {
                        if b_seen.insert(raw_bound.clone()) {
                            b_mod_set.push(raw_bound);
                        }
                    } else if seen.insert(raw_bound.clone()) {
                        mod_set.push(raw_bound);
                    }
                }
            }
        }

        while let Some(raw_bound) = mod_set.pop() {
            if self.fix_model_header_for_image(&raw_bound, &image) != 0 {
                return 1;
            }
        }

        let b_image = format!("{}b{}", self.set_name, self.stack_extension);
        while let Some(raw_bound) = b_mod_set.pop() {
            if self.fix_model_header_for_image(&raw_bound, &b_image) != 0 {
                return 1;
            }
        }
        0
    }
}

// SINGLE-CALL FUNCTIONS FOR PROCESSING STEPS AND A FEW HELPERS FOR THEM

impl Brt {
    /// `analyzeSDsAdjustExcludes` (`IMOD/pysrc/batchruntomo:2174`).
    pub fn analyze_sds_adjust_excludes(&mut self, clip_lines: &[String]) -> i32 {
        let num_average = 5usize;
        let max_exclude = 3usize;
        let crit = self.lookup_directive(
            &format!("{RUNTIME_PREFIX}Preprocessing"),
            "endExcludeCriterion",
            0,
            FLOAT_VALUE,
        );
        if self.test_directive_value(&crit, "Preprocessing.endExcludeCriterion", "float") != 0 {
            return 1;
        }
        if crit.is_none() || crit.float() <= 0. {
            return 0;
        }
        let crit = crit.float();

        self.prn_log(
            "Analyzing image SDs to see if views should be excluded at ends of series",
            "\n",
            false,
        );

        // If there are no incoming lines from fixed stack, run on raw stack
        let mut clip_lines = clip_lines.to_vec();
        if clip_lines.is_empty() {
            let command = format!("$clip stat {}{}", self.data_name, self.stack_extension);
            clip_lines = self.run_clip_stats(&command, "", "raw");
            if clip_lines.is_empty() {
                return 1;
            }
        }

        // Extract the sds as second to last number
        let mut sds: Vec<f64> = Vec::new();
        for line in &clip_lines {
            if line.contains("mean") || line.contains("-----") {
                continue;
            }
            if line.contains("all") {
                break;
            }
            let lsplit: Vec<&str> = line.split_whitespace().collect();
            match lsplit.last().and_then(|value| value.parse::<f64>().ok()) {
                Some(value) => sds.push(value),
                None => {
                    self.abort_set(
                        "Error converting standard deviation to float in clip stats output",
                    );
                    return 1;
                }
            }
        }

        if sds.len() < num_average + 2 + 2 * max_exclude {
            return 0;
        }

        // Find out if there should be dark region analysis too
        let dark_ratio = self.lookup_directive(
            &format!("{RUNTIME_PREFIX}Preprocessing"),
            "darkExcludeRatio",
            0,
            FLOAT_VALUE,
        );
        let dark_fraction = self.lookup_directive(
            &format!("{RUNTIME_PREFIX}Preprocessing"),
            "darkExcludeFraction",
            0,
            FLOAT_VALUE,
        );
        if self.test_directive_value(&dark_ratio, "Preprocessing.darkExcludeRatio", "float") != 0
            || self.test_directive_value(
                &dark_fraction,
                "Preprocessing.darkExcludeFraction",
                "float",
            ) != 0
        {
            return 1;
        }
        let dark_ratio = if dark_ratio.is_none() {
            DFLT_DARK_EXCLUDE_RATIO
        } else {
            dark_ratio.float()
        };
        let dark_fraction = if !dark_fraction.truthy() {
            DFLT_DARK_EXCLUDE_FRACTION
        } else {
            dark_fraction.float()
        };

        // Run histogram analysis if ratio is positive
        let mut dark_excludes: Vec<i64> = Vec::new();
        if dark_ratio > 0. {
            let num_hist = self.zsize.min(4 * max_exclude as i64);
            let clipcom = format!(
                "$clip hist -2d -s -iz 0-{},{}-{} {}{}",
                py_str_float(num_hist as f64 / 2. - 1.),
                py_str_float(self.zsize as f64 - num_hist as f64 / 2.),
                self.zsize - 1,
                self.data_name,
                self.stack_extension
            );
            let comfile = format!("cliphist{}{}", self.axis_let, self.com_ext);
            if self.write_text_file_report_err(&comfile, &[clipcom]) != 0 {
                return 1;
            }
            if self.run_one_process(
                &comfile,
                true,
                false,
                "Analyzing histograms for large dark areas",
                false,
            ) != 0
            {
                return 1;
            }
            cleanup_files(&[comfile]);
            let hist_lines =
                self.read_text_file_report_err(&format!("cliphist{}.log", self.axis_let), None);
            if hist_lines.is_empty() {
                return 1;
            }

            // Find which ones pass the criteria and add to list
            for line in &hist_lines {
                if line.starts_with("Slice ") && !line.contains("no histogram dip") {
                    let lsplit: Vec<&str> = line.split_whitespace().collect();
                    let values = lsplit
                        .get(1)
                        .map(|value| value.replace(':', ""))
                        .and_then(|value| value.parse::<i64>().ok())
                        .zip(lsplit.get(4).and_then(|value| value.parse::<f64>().ok()))
                        .zip(lsplit.get(6).and_then(|value| value.parse::<f64>().ok()))
                        .zip(lsplit.last().and_then(|value| value.parse::<f64>().ok()));
                    match values {
                        Some((((zval, low_peak), high_peak), fraction)) => {
                            if low_peak < dark_ratio * high_peak && fraction > dark_fraction {
                                dark_excludes.push(zval);
                            }
                        }
                        None => {
                            self.abort_set("Error converting output from clip hist");
                            return 1;
                        }
                    }
                }
            }
        }

        // Look for low images at ends of series
        let mut excludes: Vec<i64> = Vec::new();
        let mut base_ind: i64 = 0;
        let mut base_view: i64 = 1;
        for direc in [1i64, -1i64] {
            let mut num_low = 0usize;
            // use varying indentations before computing the mean; get the mean
            for indent in 2..max_exclude + 2 {
                let mut good_mean = 0.;
                for ind in 0..num_average {
                    let index = base_ind + direc * (ind + indent) as i64;
                    let index = if index < 0 {
                        sds.len() as i64 + index
                    } else {
                        index
                    } as usize;
                    good_mean += sds[index] / num_average as f64;
                }

                // See how many consecutive views are below the criterion; stop at first above
                num_low = 0;
                for ind in 0..indent {
                    let index = base_ind + direc * ind as i64;
                    let index = if index < 0 {
                        sds.len() as i64 + index
                    } else {
                        index
                    } as usize;
                    if sds[index] < crit * good_mean {
                        num_low += 1;
                    } else {
                        break;
                    }
                }

                // Done if found any and there is one not below criterion (to be conservative)
                if num_low != 0 && num_low < indent {
                    for ind in 0..num_low {
                        excludes.push(base_view + direc * ind as i64);
                    }
                    break;
                }
            }

            // If looking for dark regions too, now look from the first non-excluded one for
            // consecutive ones that would be excluded on that basis
            if dark_ratio > 0. {
                for ind in num_low..2 * max_exclude {
                    let view = base_view + direc * ind as i64;
                    if dark_excludes.contains(&(view - 1)) {
                        excludes.push(view);
                    } else {
                        break;
                    }
                }
            }

            // Set up to repeat on ending views
            base_ind = -1;
            base_view = sds.len() as i64;
        }

        if excludes.is_empty() {
            self.prn_log(
                "No views have low SDs or dark regions near end of series",
                "\n",
                false,
            );
            return 0;
        }

        // Get existing skip list
        let skip_let = if self.axis_let == "b" { "b" } else { "" };
        let directive = format!("{COPY_PREFIX}{skip_let}skip");
        excludes.sort_unstable();
        let user_excludes;
        let mut message;
        if self.all_directives[BAT_DICT_IND].contains(&directive) {
            let mut existing = self.all_directives[BAT_DICT_IND]
                .get(&directive)
                .unwrap()
                .0
                .clone();
            message = "Added these views to existing list of views to skip: ".to_owned();
            let mut num_add = 0;
            let user_list = parse_list(&existing).unwrap_or_default();
            for view in &excludes {
                if !user_list.contains(&(*view as i32)) {
                    existing += &format!(",{view}");
                    message += &format!(" {view}");
                    num_add += 1;
                }
            }
            if num_add == 0 {
                self.prn_log(
                    "Views with low SDs were already in the list of views to skip",
                    "\n",
                    false,
                );
                return 0;
            }
            user_excludes = existing;
        } else {
            let mut existing = String::new();
            message = "These views have low SDs or dark regions and will be skipped: ".to_owned();
            let mut comma = "";
            for view in &excludes {
                existing += &format!("{comma}{view}");
                comma = ",";
                message += &format!(" {view}");
            }
            user_excludes = existing;
        }

        self.prn_log(&message, "\n", false);

        // Modify all four com files
        for (com_root, option, after_opt) in [
            ("xcorr", "SkipViews", "RotationAngle"),
            ("track", "SkipViews", "RotationAngle"),
            ("align", "ExcludeList", "RotationAngle"),
            ("tilt", "EXCLUDELIST2", "XTILTFILE"),
        ] {
            let comfile = format!("{com_root}{}", self.axis_com);
            let sedcom = sed_del_and_add(option, &user_excludes, after_opt, '/');
            let in_lines = self.read_text_file_report_err(&comfile, None);
            if in_lines.is_empty() {
                return 1;
            }
            if pysed(
                &sedcom,
                PysedSrc::Lines(&in_lines),
                Some(&comfile),
                false,
                '/',
                true,
            )
            .is_err()
            {
                self.abort_set(&format!("Error modifying {comfile}"));
                return 1;
            }
        }

        0
    }

    /// `fidlessFileOperations` (`IMOD/pysrc/batchruntomo:2347`).
    pub fn fidless_file_operations(&mut self) -> i32 {
        self.did_local_align = 0;
        self.made_zfactors = false;

        let rotfile = format!("rotation{}.xf", self.axis_let);
        let sinrot = self.axis_rotation.to_radians().sin();
        let cosrot = self.axis_rotation.to_radians().cos();
        if self.write_text_file_report_err(
            &rotfile,
            &[format!(
                "{cosrot:.6} {sinrot:.6} {:.6} {cosrot:.6} 0. 0.",
                -sinrot
            )],
        ) != 0
        {
            return 1;
        }

        // Take care of xtilt file
        let pre_lines = self.read_text_file_report_err(&format!("{}.prexf", self.data_name), None);
        if pre_lines.is_empty() {
            return 1;
        }
        let xtilts: Vec<String> = pre_lines.iter().map(|_| "0.".to_owned()).collect();
        if self.write_text_file_report_err(&format!("{}.xtilt", self.data_name), &xtilts) != 0 {
            return 1;
        }

        if run_cmd(
            &format!("xftoxg -nfit 0 {}.prexf", self.data_name),
            None,
            None,
            None,
            &[],
        )
        .is_err()
            || run_cmd(
                &format!(
                    "xfproduct {0}.prexg {rotfile} {0}_nonfid.xf",
                    self.data_name
                ),
                None,
                None,
                None,
                &[],
            )
            .is_err()
        {
            self.report_imod_error(Some("Cannot prepare transformations"));
            return 1;
        }
        if std::fs::copy(
            format!("{}_nonfid.xf", self.data_name),
            format!("{}.xf", self.data_name),
        )
        .is_err()
            || std::fs::copy(
                format!("{}.rawtlt", self.data_name),
                format!("{}.tlt", self.data_name),
            )
            .is_err()
        {
            self.abort_set("Error copying xf or tlt file");
            return 1;
        }
        0
    }

    /// `runPatchTracking` (`IMOD/pysrc/batchruntomo:2384`).
    pub fn run_patch_tracking(&mut self) -> i32 {
        let contour_pieces = self.lookup_directive(PATCH_TRACK_TEXT, "contourPieces", 0, INT_VALUE);
        if contour_pieces.is_str() {
            self.abort_set("Error converting contour pieces in directive to integer");
            return 1;
        }

        if !self
            .lookup_directive(
                &format!("{COM_PREFIX}xcorr_pt"),
                "tiltxcorr.SizeOfPatchesXandY",
                0,
                STRING_VALUE,
            )
            .truthy()
        {
            self.abort_set("Size of patches must be specified to use patch tracking");
            return 1;
        }

        if self
            .lookup_directive(
                &format!("{COM_PREFIX}xcorr_pt"),
                "tiltxcorr.NumberOfPatchesXandY",
                0,
                STRING_VALUE,
            )
            .truthy()
            && self
                .lookup_directive(
                    &format!("{COM_PREFIX}xcorr_pt"),
                    "tiltxcorr.OverlapOfPatchesXandY",
                    0,
                    STRING_VALUE,
                )
                .truthy()
        {
            self.abort_set(
                "You cannot enter both the number of patches to track and the fractional overlap of patches (xcorr_pt.tiltxcorr.NumberOfPatchesXandY and xcorr_pt.tiltxcorr.OverlapOfPatchesXandY)",
            );
            return 1;
        }

        // Start command list for running makecomfile
        let mut comlines = vec![
            format!("InputFile xcorr{}", self.axis_com),
            format!("BinningOfImages {}", self.coarse_binning),
            format!("RootNameOfDataFiles {}", self.data_name),
        ];
        comlines.extend(self.later_com_directives(0));
        if self.use_gpu != 0 {
            comlines.push("UseGPU  0".to_owned());
        }

        // Look for boundary models to transform
        let (error, boundary_mod) =
            self.get_or_xform_boundary_model(PATCH_TRACK_TEXT, "_ptbound.mod", true);
        if error != 0 {
            return 1;
        }
        if !boundary_mod.is_empty() {
            comlines.push(format!(
                "OneParameterChange {COM_PREFIX}xcorr_pt{}.tiltxcorr.BoundaryModel={boundary_mod}",
                self.axis_let
            ));
        }

        // Convert runtime directive for contour pieces into com directive for imodchopconts
        if contour_pieces.truthy() && contour_pieces.int() > 1 {
            let piece_length = self.lookup_directive(
                &format!("{COM_PREFIX}xcorr_pt"),
                "imodchopconts.LengthOfPieces",
                0,
                INT_VALUE,
            );

            // Check if length is entered, allow the default to be overridden
            if piece_length.is_str() {
                self.abort_set("Error converting length of contour pieces in directive to integer");
                return 1;
            }
            if piece_length.truthy() && piece_length.int() > 0 {
                self.abort_set(
                    "Both xcorr_pt.imodchopconts.LengthOfPieces > 0 and PatchTracking.contourPieces were entered; only one is allowed",
                );
                return 1;
            }
            comlines.push(format!(
                "OneParameterChange {COM_PREFIX}xcorr_pt{}.imodchopconts.NumberOfPieces={}",
                self.axis_let,
                contour_pieces.int()
            ));
        }

        let comfile = format!("xcorr_pt{}", self.axis_com);
        if self.make_and_run_one_com(
            &comlines,
            &comfile,
            "Tracking patches to make alignment model",
            false,
            false,
        ) != 0
        {
            return 1;
        }
        0
    }

    /// `OKtoAdjustPatchTrack` (`IMOD/pysrc/batchruntomo:2442`).
    pub fn ok_to_adjust_patch_track(&mut self) -> bool {
        let max_tilt_adjust =
            self.lookup_directive(PATCH_TRACK_TEXT, "maxTiltAdjustment", 0, FLOAT_VALUE);
        let max_tilt_adjust = if max_tilt_adjust.is_str() || !max_tilt_adjust.truthy() {
            DFLT_PT_MAX_TILT_ADJUST
        } else {
            max_tilt_adjust.float()
        };
        let max_adjusted_angle =
            self.lookup_directive(PATCH_TRACK_TEXT, "maxAdjustedAngle", 0, FLOAT_VALUE);
        let max_adjusted_angle = if max_adjusted_angle.is_str() || !max_adjusted_angle.truthy() {
            DFLT_PT_MAX_ADJUSTED_ANGLE
        } else {
            max_adjusted_angle.float()
        };

        if self.total_del_tilt.abs() > max_tilt_adjust {
            let total = self.total_del_tilt;
            self.warning(
                &[format!(
                    "Patch tracking not being iterated: tilt angle adjustment by {total:.1} is above the limit of {max_tilt_adjust:.1}"
                )],
                true,
            );
            return false;
        }

        // Forgive any problems reading the tilt angles
        let name = format!("{}{}.rawtlt", self.set_name, self.axis_let);
        let lines = match read_text_file(&name, None, true, None) {
            Ok(lines) => lines,
            Err(_) => return true,
        };
        let mut max_angle = 0.;
        for line in &lines {
            let lsplit: Vec<&str> = line.split_whitespace().collect();
            match lsplit.first().and_then(|value| value.parse::<f64>().ok()) {
                Some(value) => {
                    max_angle = f64::max(max_angle, (value + self.total_del_tilt).abs());
                }
                None => return true,
            }
        }

        if max_angle > max_adjusted_angle {
            let total = self.total_del_tilt;
            self.warning(
                &[format!(
                    "Patch tracking not being iterated: tilt angle adjustment by {total:.1} would make the highest angle be {max_angle:.1}, above the limit of {max_adjusted_angle:.1}"
                )],
                true,
            );
            return false;
        }

        true
    }

    /// `varyPatchTrack` (`IMOD/pysrc/batchruntomo:2480`).
    pub fn vary_patch_track(&mut self) -> i32 {
        let mut comlines: Vec<String> = Vec::new();
        if self.use_gpu != 0 {
            if self.manage_gpu_allocation() != 0 {
                return 1;
            }
            comlines = vec![format!(
                "OneParameterChange {COM_PREFIX}varypatchtrack.varypatchtrack.UseGPU=0"
            )];
        }

        comlines.extend(self.later_com_directives(0));
        let comfile = format!("varypatchtrack{}", self.axis_com);
        let use_gpu = !self.gpu_list.is_empty();
        if self.make_and_run_one_com(
            &comlines,
            &comfile,
            "Optimizing patch tracking parameters",
            use_gpu,
            false,
        ) != 0
        {
            return 1;
        }

        let vary_lines =
            self.read_text_file_report_err(&format!("varypatchtrack{}.log", self.axis_let), None);
        if vary_lines.is_empty() {
            return 1;
        }
        let mut new_xcorr = String::new();
        let mut new_align = String::new();
        for line in &vary_lines {
            if line.contains("[VPT1]") {
                let lsplit: Vec<&str> = line.trim().split_whitespace().collect();
                new_xcorr = lsplit[lsplit.len() - 2].to_owned();
            }
            if line.contains("[VPT2]") {
                let lsplit: Vec<&str> = line.trim().split_whitespace().collect();
                new_align = lsplit[lsplit.len() - 2].to_owned();
            }
            if line.starts_with("Starting") || line.starts_with("Final") {
                let line = line.clone();
                self.prn_log(&line, "\n", false);
            }
        }

        if new_xcorr.is_empty() {
            self.abort_set(
                "No new Tiltxcorr command file was written after running Varypatchtrack",
            );
            return 1;
        }
        let target = format!("xcorr_pt{}", self.axis_com);
        if self.use_file_as_replacement(&new_xcorr, &target, false, true) != 0 {
            return 1;
        }
        let target = format!("align{}", self.axis_com);
        if !new_align.is_empty()
            && self.use_file_as_replacement(&new_align, &target, false, true) != 0
        {
            return 1;
        }
        if self.run_patch_tracking() != 0 {
            return 1;
        }
        0
    }
}

impl Brt {
    /// `makeSeedAndTrack` (`IMOD/pysrc/batchruntomo:2521`).
    pub fn make_seed_and_track(&mut self) -> i32 {
        let tracking_method = self.lookup_directive(
            &format!("{RUNTIME_PREFIX}Fiducials"),
            "trackingMethod",
            0,
            INT_VALUE,
        );
        let num_btruns = self.lookup_directive(
            &format!("{RUNTIME_PREFIX}BeadTracking"),
            "numberOfRuns",
            0,
            INT_VALUE,
        );
        let seeding_method = self.lookup_directive(
            &format!("{RUNTIME_PREFIX}Fiducials"),
            "seedingMethod",
            0,
            INT_VALUE,
        );

        if tracking_method.is_str() || num_btruns.is_str() || seeding_method.is_str() {
            self.abort_set("Error converting tracking method or number of runs to integer");
            return 1;
        }
        let tracking_method = if !tracking_method.truthy() {
            0
        } else {
            tracking_method.int()
        };
        if !(0..=2).contains(&tracking_method) {
            self.abort_set("trackingMethod must be between 0 and 2");
        }
        if (seeding_method.is_none()
            || seeding_method.int() < 1
            || seeding_method.int() > 3
            || (self.axis_ind == 0 && seeding_method.int() == 2))
            && tracking_method == 0
        {
            self.abort_set(
                "seedingMethod must be between 1 and 3 and not be 2 for first/only axis",
            );
            return 1;
        }
        let seeding_method = seeding_method.int();

        // If runs is not there, assume 0 for RAPTOR and 1 for other
        let mut num_btruns = if num_btruns.is_none() {
            if tracking_method == 2 { 0 } else { 1 }
        } else {
            num_btruns.int()
        };
        if num_btruns <= 0 && tracking_method != 2 {
            self.abort_set("Number of beadtrack runs must be > 0 unless using RAPTOR");
            return 1;
        }

        // Modify track.com with ImageBinned entry
        let track_com = format!("track{}", self.axis_com);
        let btlines = self.read_text_file_report_err(&track_com, None);
        if btlines.is_empty() {
            return 1;
        }
        let sedcom = sed_del_and_add(
            "ImagesAreBinned",
            &self.coarse_binning.to_string(),
            "OutputModel",
            '/',
        );
        if pysed(
            &sedcom,
            PysedSrc::Lines(&btlines),
            Some(&track_com),
            false,
            '/',
            true,
        )
        .is_err()
        {
            self.abort_set(&format!("Error modifying track{}", self.axis_com));
            return 1;
        }
        let light_beads = matches!(
            option_value(&btlines, "LightBeads", BOOL_VALUE, false, 0, None, None),
            Some(OptionValue::Boolean(true))
        );
        let light_int = i32::from(light_beads);
        let edfcom = self.edf_del_and_add(
            &format!("Track.{}.LightBeads", self.axis_upper_let),
            &light_int.to_string(),
            '/',
        );
        if self.modify_edf_lines(&edfcom) != 0 {
            return 1;
        }

        // RAPTOR
        if self.need_step(4.) && tracking_method == 2 {
            let markers = self.lookup_directive(
                &format!("{RUNTIME_PREFIX}RAPTOR"),
                "numberOfMarkers",
                0,
                INT_VALUE,
            );
            if markers.is_none() || markers.is_str() || markers.int() <= 0 {
                self.abort_set(
                    "numberOfMarkers for RAPTOR missing, not positive, or gave conversion error",
                );
                return 1;
            }

            let (rap_stack, beadint) = if self
                .lookup_directive(
                    &format!("{RUNTIME_PREFIX}RAPTOR"),
                    "useAlignedStack",
                    0,
                    BOOL_VALUE,
                )
                .truthy()
            {
                (
                    dataset_filename(".preali", None, None),
                    (self.fid_size_pix / self.coarse_binning as f64).round() as i64,
                )
            } else {
                (
                    self.data_name.clone() + &self.stack_extension,
                    self.fid_size_pix.round() as i64,
                )
            };

            let line = format!(
                "$runraptor -mark {} -diam {beadint} {rap_stack}",
                markers.int()
            );
            let comfile = format!("runraptor{}", self.axis_com);
            if self.write_text_file_report_err(
                &comfile,
                &["# Command file to run raptor".to_owned(), line],
            ) != 0
            {
                return 1;
            }
            if self.run_one_process(
                &comfile,
                true,
                false,
                "Tracking fiducials with RAPTOR",
                false,
            ) != 0
            {
                return 1;
            }
            let from = format!("{}_raptor.fid", self.data_name);
            let to = format!("{}.fid", self.data_name);
            if self.use_file_as_replacement(&from, &to, false, true) != 0 {
                return 1;
            }
        }
        // Seed and track:
        else if self.need_step(4.) {
            // Transferfid for axis b
            let mut skip_auto = false;
            if self.axis_ind > 0 && (seeding_method & 2) != 0 {
                let mut comlines = vec![format!("RootNameOfDataFiles\t{}", self.set_name)];
                comlines.extend(self.later_com_directives(0));
                comlines.push(format!(
                    "OneParameterChange {COM_PREFIX}transferfid.transferfid.LowestTiltTransformFile={}_AtoB.xf",
                    self.set_name
                ));
                let comfile = format!("transferfid{}", self.com_ext);
                if self.make_and_run_one_com(
                    &comlines,
                    &comfile,
                    "Transferring fiducials from A to B axis",
                    false,
                    false,
                ) != 0
                {
                    return 1;
                }
                let tags = [
                    MessageTag("ERROR:", 0, None),
                    MessageTag("WARNING:", 0, None),
                    MessageTag("fiducials that failed", 4, None),
                ];
                self.print_tagged_messages_file("transferfid.log", &tags);
                let messages = self.latest_messages.clone();
                let num_failed =
                    self.find_tagged_value(&messages, "fiducials that failed", ':', INT_VALUE);
                if !num_failed.is_none() && num_failed.int() == 0 {
                    skip_auto = true;
                }
            }

            // Autoseed
            if (seeding_method & 1) != 0 && !skip_auto {
                let two_surf = i32::from(self.num_surfaces > 1);

                // If there is not a specific directive for two surfaces, set it based on
                // tiltalign entry
                let mut comlines = self.later_com_directives(0);
                if self.axis_ind > 0 && (seeding_method & 2) != 0 {
                    comlines.push(format!(
                        "OneParameterChange {COM_PREFIX}autofidseedb.autofidseed.AppendToSeedModel=1"
                    ));
                }
                if self
                    .lookup_directive(
                        &format!("{COM_PREFIX}autofidseed"),
                        "autofidseed.TwoSurfaces",
                        0,
                        STRING_VALUE,
                    )
                    .is_none()
                {
                    comlines.push(format!(
                        "OneParameterChange {COM_PREFIX}autofidseed{}.autofidseed.TwoSurfaces={two_surf}",
                        self.axis_let
                    ));
                }

                // If transferfid was not done, do not append to dummy model
                if self.axis_ind > 0 && (seeding_method & 2) == 0 {
                    comlines.push(format!(
                        "OneParameterChange {COM_PREFIX}autofidseed{}.autofidseed.AppendToSeedModel=0",
                        self.axis_let
                    ));
                }

                let (error, boundary_mod) = self.get_or_xform_boundary_model(
                    AUTO_SEED_TEXT,
                    "_afsbound.mod",
                    (seeding_method & 2) != 0,
                );
                if error != 0 {
                    return 1;
                }

                // Add boundary model to com
                if !boundary_mod.is_empty() {
                    comlines.push(format!(
                        "OneParameterChange {COM_PREFIX}autofidseed{}.autofidseed.BoundaryModel={boundary_mod}",
                        self.axis_let
                    ));
                }

                let comfile = format!("autofidseed{}", self.axis_com);
                if self.make_and_run_one_com(
                    &comlines,
                    &comfile,
                    "Finding seed points for fiducial model",
                    false,
                    false,
                ) != 0
                {
                    return 1;
                }
                let tags = [
                    MessageTag("ERROR:", 0, None),
                    MessageTag("WARNING:", 0, None),
                    MessageTag("candidate points, including", 4, None),
                    MessageTag("Final:   total", 0, None),
                    MessageTag("[AFS1]", 4, None),
                    MessageTag("[AFS3]", 4, None),
                ];
                let log = format!("autofidseed{}.log", self.axis_let);
                self.print_tagged_messages_file(&log, &tags);

                // If track file was adjusted for either reason, use the adjusted file
                for line in self.latest_messages.clone() {
                    if line.contains("[AFS1]") || line.contains("[AFS3]") {
                        let real_track = format!("track{}", self.axis_com);
                        let adj_file = format!("track{}_adjusted{}", self.axis_let, self.com_ext);
                        if !Path::new(&adj_file).exists() {
                            self.abort_set(&format!(
                                "Autofidseed adjusted tracking parameters but {adj_file} does not exist"
                            ));
                            return 1;
                        }

                        let orig_track = format!("track{}_orig{}", self.axis_let, self.com_ext);
                        if Path::new(&orig_track).exists() {
                            make_backup_file(&real_track);
                        } else if self.rename_and_abort(
                            &real_track,
                            &orig_track,
                            "Renaming {} to {} before adopting adjusted file",
                        ) != 0
                        {
                            return 1;
                        }
                        if self.rename_and_abort(
                            &adj_file,
                            &real_track,
                            "Renaming adjusted track command file {} to {}",
                        ) != 0
                        {
                            return 1;
                        }

                        break;
                    }
                }
            }

            let edfcom =
                self.edf_del_and_add(&format!("{}.SeedingDone", self.axis_edf_let), "true", '/');
            if self.modify_edf_lines(&edfcom) != 0 {
                return 1;
            }
        }

        // Run bead tracking indicated number of times
        if !self.need_step(5.) {
            num_btruns = 0;
        }
        for track_ind in 0..num_btruns.max(0) {
            // After first time, save seed as _orig or back it up
            if track_ind != 0 || tracking_method == 2 {
                let from = format!("{}.fid", self.data_name);
                let to = format!("{}.seed", self.data_name);
                if self.use_file_as_replacement(&from, &to, true, true) != 0 {
                    return 1;
                }
            }
            let comfile = format!("track{}", self.axis_com);
            if self.run_one_process(
                &comfile,
                true,
                false,
                &format!("Tracking beads with Beadtrack, run # {}", track_ind + 1),
                false,
            ) != 0
            {
                return 1;
            }
            let tags = [
                MessageTag("ERROR:", 0, None),
                MessageTag("WARNING:", 0, None),
                MessageTag("Total points missing =", 4, None),
            ];
            let log = format!("track{}.log", self.axis_let);
            self.print_tagged_messages_file(&log, &tags);
            let messages = self.latest_messages.clone();
            let missing = self.find_tagged_value(&messages, "Total points missing", '=', INT_VALUE);
            if !missing.is_none() && missing.int() == 0 {
                break;
            }
        }
        0
    }

    /// `postProcessTiltalign` (`IMOD/pysrc/batchruntomo:2700`).
    pub fn post_process_tiltalign(&mut self) -> i32 {
        let lines = self.align_lines.clone();
        let zfacs = option_value(&lines, "XStretchOption", INT_VALUE, false, 1, None, None);
        let local_ali = option_value(&lines, "LocalAlignments", BOOL_VALUE, false, 0, None, None);
        let z_shift = option_value(&lines, "AxisZShift", FLOAT_VALUE, false, 1, None, None);
        let angle = option_value(&lines, "AngleOffset", FLOAT_VALUE, false, 1, None, None);
        let zfacs_positive = match &zfacs {
            Some(OptionValue::Integers(values)) if !values.is_empty() => values[0] > 0,
            _ => false,
        };
        let local_ali = matches!(local_ali, Some(OptionValue::Boolean(true)));
        let z_shift_value = match &z_shift {
            Some(OptionValue::Floats(values)) if !values.is_empty() => values[0] as f64,
            _ => 0.,
        };
        let angle_value = match &angle {
            Some(OptionValue::Floats(values)) if !values.is_empty() => values[0] as f64,
            _ => 0.,
        };
        let flag = self.bool_string_for_edf(zfacs_positive);
        let mut edfcom =
            self.edf_del_and_add(&format!("MadeZFactors{}", self.axis_upper_let), &flag, '/');
        let flag = self.bool_string_for_edf(local_ali);
        edfcom.extend(self.edf_del_and_add(
            &format!("UsedLocalAlignments{}", self.axis_upper_let),
            &flag,
            '/',
        ));
        edfcom.extend(self.edf_del_and_add(
            &format!("{}.align.AxisZShift", self.axis_edf_let),
            &py_str_float(py_round(z_shift_value, 2)),
            '/',
        ));
        edfcom.extend(self.edf_del_and_add(
            &format!("{}.align.AngleOffset", self.axis_edf_let),
            &py_str_float(py_round(angle_value, 3)),
            '/',
        ));
        self.modify_edf_lines(&edfcom)
    }

    /// `modifyRestrictAndRunAlign` (`IMOD/pysrc/batchruntomo:2717`).
    pub fn modify_restrict_and_run_align(
        &mut self,
        comfile: &str,
        sedcom: &[String],
        local_ali: i32,
        message: &str,
        skip_restrict: bool,
    ) -> (i32, i32) {
        let lines = self.align_lines.clone();
        if pysed(
            sedcom,
            PysedSrc::Lines(&lines),
            Some(comfile),
            false,
            '/',
            true,
        )
        .is_err()
        {
            self.abort_set(&format!("Error modifying {comfile}"));
            return (-1, 0);
        }
        match pysed(sedcom, PysedSrc::Lines(&lines), None, false, '/', true) {
            Ok(Some(new_lines)) => self.align_lines = new_lines,
            _ => {
                self.abort_set("Error modifying alignment lines: <class 'str'>");
                return (-1, 0);
            }
        }

        let mut retval = 0;
        let mut no_robust = 0;
        let mut ran_already = false;
        let mut message = message.to_owned();
        if !skip_restrict {
            // Get directives to modify defaults if any when running restrictalign
            let min_ratio = self.lookup_directive(
                &format!("{RUNTIME_PREFIX}RestrictAlign"),
                "minMeasurementRatio",
                0,
                FLOAT_VALUE,
            );
            let min_ratio = if min_ratio.truthy() {
                min_ratio.float()
            } else {
                0.
            };
            let target_ratio = self.lookup_directive(
                &format!("{RUNTIME_PREFIX}RestrictAlign"),
                "targetMeasurementRatio",
                0,
                FLOAT_VALUE,
            );
            let target_ratio = if target_ratio.truthy() {
                target_ratio.float()
            } else {
                0.
            };
            let res_order = self.lookup_directive(
                &format!("{RUNTIME_PREFIX}RestrictAlign"),
                "orderOfRestrictions",
                0,
                STRING_VALUE,
            );
            let skip_bt = self.lookup_directive(
                &format!("{RUNTIME_PREFIX}RestrictAlign"),
                "skipBeamTiltWithOneRot",
                0,
                BOOL_VALUE,
            );
            let mut comlines = vec![
                format!("InputFile {comfile}"),
                format!("OutputFile restrictalign{}", self.axis_com),
            ];

            let mut loc_text = "global ";
            if local_ali != 0 {
                comlines.push("LocalAlignValidation 3".to_owned());
                loc_text = "local ";
            }
            if min_ratio != 0. || target_ratio != 0. {
                comlines.push(format!(
                    "TargetAndMinRatios {},{}",
                    py_str_float(target_ratio),
                    py_str_float(min_ratio)
                ));
            }
            if res_order.truthy() {
                comlines.push(format!(
                    "OneParameterChange {COM_PREFIX}restrictalign{}.restrictalign.OrderOfRestrictions={}",
                    self.axis_let,
                    res_order.text()
                ));
            }
            if skip_bt.truthy() {
                comlines.push("SkipBeamTiltWithOneRot 1".to_owned());
            }

            comlines.extend(self.later_com_directives(0));

            let out_com = format!("restrictalign{}", self.axis_com);
            let mess = format!("Running restrictalign to optimize {loc_text}alignment parameters");
            if self.make_and_run_one_com(&comlines, &out_com, &mess, false, false) != 0 {
                self.suppress_abort = false;
                return (-1, 0);
            }

            let from_log = format!("restrictalign{}.log", self.axis_let);
            let res_lines = self.read_text_file_report_err(&from_log, None);
            if res_lines.is_empty() {
                self.suppress_abort = false;
                return (-1, 0);
            }

            // Copy log to _global if there isn't one, or to _local
            let mut to_log = format!("restrictalign{}_global.log", self.axis_let);
            let mut copy_global = false;
            if local_ali == 0 {
                copy_global = !is_file_newer(&to_log, &format!("origcoms/align{}", self.axis_com));
            }
            if local_ali != 0 || copy_global {
                if local_ali != 0 {
                    to_log = format!("restrictalign{}_local.log", self.axis_let);
                }
                make_backup_file(&to_log);
                if std::fs::copy(&from_log, &to_log).is_err() {
                    self.prn_log(
                        &format!("WARNING: Failed to copy {from_log} to {to_log}"),
                        "\n",
                        false,
                    );
                }
            }

            // Echo all the summary output
            let mut do_output = false;
            retval = 1;
            for line in &res_lines {
                if line.starts_with("restrictalign:") {
                    do_output = true;
                }
                if line.contains("CHUNK DONE")
                    || line.contains("COMPLETED")
                    || line.contains("ERROR:")
                {
                    break;
                }
                if line.contains("No restriction") {
                    retval = 0;
                }
                if line.contains("[rsa2]") {
                    no_robust = 1;
                    if line.contains("fail") {
                        no_robust = 2;
                    }
                }
                if line.contains("[rsa1]") {
                    ran_already = true;
                    do_output = false;
                }
                if do_output {
                    let text = line.trim_matches(['\r', '\n']).to_owned();
                    self.prn_log(&text, "\n", false);
                }
            }

            self.prn_log(" ", "\n", false);
            if no_robust != 0 {
                message = message.replace("with robust", "without robust");
            }
            if retval != 0 {
                self.align_lines = self.read_text_file_report_err(comfile, None);
                if self.align_lines.is_empty() {
                    self.suppress_abort = false;
                    return (-1, 0);
                }
            }
        }

        if ran_already {
            let text = format!("{}     [brt3]", message.replace("Doing", "Did"));
            self.prn_log(&text, "\n", false);
        } else if self.run_one_process(comfile, true, false, &message, false) != 0 {
            return (-2, no_robust);
        }
        if self.post_process_tiltalign() != 0 {
            return (-2, no_robust);
        }
        (retval, no_robust)
    }
}

/// Python's `round(value, digits)`: round half to even at that many decimals.
fn py_round(value: f64, digits: i32) -> f64 {
    let scale = 10f64.powi(digits);
    let scaled = value * scale;
    let rounded = scaled.round();
    let result = if (scaled - scaled.trunc()).abs() == 0.5 && rounded % 2.0 != 0.0 {
        rounded - scaled.signum()
    } else {
        rounded
    };
    result / scale
}

impl Brt {
    /// `runTiltalign` (`IMOD/pysrc/batchruntomo:2826`).
    pub fn run_tiltalign(&mut self) -> i32 {
        let min_tot_glbl_stretch = 12i64;
        let min_surf_glbl_stretch = 4i64;
        let min_ratio_glbl_stretch = 0.125;
        let min_surf_local_stretch = 1.0;
        let min_tot_skip_restrict = 20i64;
        let min_tot_skip_cross_val = 80i64;
        let min_tot_to_set_angle = 4i64;

        // The command file should be configured by various inputs, so let's find out some
        let lines = self.align_lines.clone();
        let patch_size_arr = option_value(
            &lines,
            "TargetPatchSizeXandY",
            INT_VALUE,
            false,
            2,
            None,
            None,
        );
        let min_fids_arr = option_value(
            &lines,
            "MinFidsTotalAndEachSurface",
            INT_VALUE,
            false,
            2,
            None,
            None,
        );
        if patch_size_arr.is_none() || min_fids_arr.is_none() {
            self.abort_set(&format!(
                "Problem finding some options in align{}",
                self.com_ext
            ));
        }
        let (nxpatch, nypatch) = match &patch_size_arr {
            Some(OptionValue::Integers(values)) if values.len() >= 2 => {
                (values[0] as i64, values[1] as i64)
            }
            _ => (0, 0),
        };
        let (min_fids_tot, min_fids_surf) = match &min_fids_arr {
            Some(OptionValue::Integers(values)) if values.len() >= 2 => {
                (values[0] as i64, values[1] as i64)
            }
            _ => (0, 0),
        };
        let align_com = format!("align{}", self.axis_com);
        let align_log = format!("align{}.log", self.axis_let);
        let mut do_robust = self
            .lookup_directive(
                &format!("{COM_PREFIX}align"),
                "tiltalign.RobustFitting",
                0,
                BOOL_VALUE,
            )
            .int();
        let robust_orig = do_robust;

        let mut enable_stretch = self
            .lookup_directive(
                &format!("{RUNTIME_PREFIX}TiltAlignment"),
                "enableStretching",
                0,
                BOOL_VALUE,
            )
            .truthy();
        let enable_skew_only = self.lookup_directive(
            &format!("{RUNTIME_PREFIX}TiltAlignment"),
            "enableSkewOnly",
            0,
            INT_VALUE,
        );
        self.did_local_align = self
            .lookup_directive(
                &format!("{COM_PREFIX}align"),
                "tiltalign.LocalAlignments",
                0,
                BOOL_VALUE,
            )
            .int();

        let cross_val = self.lookup_directive(
            &format!("{COM_PREFIX}restrictalign"),
            "restrictalign.UseCrossValidation",
            0,
            INT_VALUE,
        );
        let cross_val = if cross_val.is_none() {
            1
        } else {
            cross_val.int()
        };

        if self.patch_track {
            enable_stretch = false;
        }

        let mut angle_arr = [0f64; 8];

        // If skipping align, get a few more values from com file, analyze log
        if self.skip_tiltalign != 0 {
            let glb_stretch =
                option_value(&lines, "XStretchOption", INT_VALUE, false, 1, None, None);
            let glb_skew = option_value(&lines, "SkewOption", INT_VALUE, false, 1, None, None);
            let stretch_positive = match &glb_stretch {
                Some(OptionValue::Integers(values)) if !values.is_empty() => values[0] > 0,
                _ => false,
            };
            let skew_positive = match &glb_skew {
                Some(OptionValue::Integers(values)) if !values.is_empty() => values[0] > 0,
                _ => false,
            };
            self.made_zfactors = stretch_positive || skew_positive;
            let two_surf = self.num_surfaces > 1;
            if self.analyze_align_log(two_surf, &mut angle_arr, false) != 0 {
                return 1;
            }
            self.total_del_tilt = angle_arr[0];
            self.xtilt_needed = angle_arr[1];
            self.fid_thickness = angle_arr[2];
            self.fid_inc_shift = angle_arr[3];
            self.recon_thickness = angle_arr[5];
            return 0;
        }

        let mut restricted = 0;
        let mut no_robust;
        let mut message_tags: Vec<MessageTag> = Vec::new();

        // first time, turn off local align and make sure stretch/skew off
        // But loop twice in case have to turn off robust fitting
        for loop_index in [0, 1] {
            let rob_text = if do_robust != 0 { "with" } else { "without" };
            let mess = format!(
                "Doing fine alignment with no distortion or local alignments, {rob_text} robust fitting"
            );
            self.suppress_abort = loop_index == 0 && do_robust != 0;
            let mut sedcom = vec![
                sed_modify("SurfacesToAnalyze", &self.num_surfaces.to_string(), '/'),
                sed_modify("LocalAlignments", "0", '/'),
                sed_modify("XStretchOption", "0", '/'),
                sed_modify("SkewOption", "0", '/'),
            ];
            sedcom.extend(sed_del_and_add(
                "ImagesAreBinned",
                &self.coarse_binning.to_string(),
                "OutputTransformFile",
                '/',
            ));
            sedcom.extend(sed_del_and_add(
                "ScaleShifts",
                &format!("1,{}", self.coarse_binning),
                "InputFile2",
                '/',
            ));
            sedcom.extend(sed_del_and_add(
                "RobustFitting",
                &do_robust.to_string(),
                "OutputTransformFile",
                '/',
            ));

            // Do restrictalign only the first time
            let result =
                self.modify_restrict_and_run_align(&align_com, &sedcom, 0, &mess, loop_index > 0);
            restricted = result.0;
            no_robust = result.1;
            if no_robust != 0 {
                do_robust = 0;
            }
            if restricted < 0 {
                if !self.suppress_abort {
                    return 1;
                }
                self.suppress_abort = false;

                // Look for robust failure: this shouldn't happen with cross-validation
                let mut found = false;
                for line in self.latest_messages.clone() {
                    if line
                        .to_uppercase()
                        .contains("TOO FEW DATA POINTS TO DO ROBUST")
                    {
                        self.prn_log("Trying again without robust fitting", "\n", false);
                        do_robust = 0;
                        found = true;
                        break;
                    }
                }
                if !found {
                    return 1;
                }

                continue;
            }

            message_tags = vec![
                MessageTag("ERROR:", 0, None),
                MessageTag("WARNING:", 0, None),
                MessageTag("Residual error", 4, None),
                MessageTag("leave-out error", 4, None),
                MessageTag("Benefit from", 4, None),
            ];
            let tags = message_tags.clone();
            self.print_tagged_messages_file(&align_log, &tags);
            break;
        }

        self.suppress_abort = false;
        angle_arr = [0f64; 8];
        let two_surf = self.num_surfaces > 1;
        if self.analyze_align_log(two_surf, &mut angle_arr, false) != 0 {
            return 1;
        }
        self.total_del_tilt = angle_arr[0];
        self.xtilt_needed = angle_arr[1];
        self.fid_thickness = angle_arr[2];
        self.fid_inc_shift = angle_arr[3];
        self.recon_thickness = angle_arr[5];
        let mut num_bot = angle_arr[6] as i64;
        let mut num_top = angle_arr[7] as i64;
        let mut total_fid;
        let mut min_on_surf = 0i64;
        if self.num_surfaces > 1 {
            total_fid = num_bot + num_top;
            min_on_surf = num_bot.min(num_top);
        } else {
            total_fid = num_bot;
        }

        let mut local_align = 0;
        let mut glb_stretch = 0;
        let mut glb_skew = 0;
        let mut loc_stretch = 0;
        let mut loc_skew = 0;
        let mut use_min_fids = 0i64;
        let mut numruns = 1;
        self.made_zfactors = false;
        let mut no_robust_outer = 0;
        if self.did_local_align != 0 && (restricted == 0 || cross_val != 0) {
            local_align = 1;
            message_tags = vec![
                MessageTag("ERROR:", 0, None),
                MessageTag("WARNING:", 0, None),
                MessageTag("(Global)", 4, None),
                MessageTag("error local mean:", 4, None),
                MessageTag("Local.*leave-out error", 4, None),
                MessageTag("Benefit from", 4, None),
            ];
        } else {
            self.did_local_align = 0;
        }

        if (restricted == 0 || cross_val != 0)
            && (self.did_local_align != 0 || enable_stretch || enable_skew_only.truthy())
        {
            if self.did_local_align != 0 {
                numruns = 2;

                // Turn on robust again for locals if they were on before and were turned off
                // because of no benefit - failure gives a 2
                if cross_val != 0 && no_robust_outer == 1 {
                    do_robust = robust_orig;
                }
            }

            // are there enough fids for stretch?
            if enable_stretch {
                if total_fid > min_tot_glbl_stretch
                    && (self.num_surfaces == 1
                        || (min_on_surf > min_surf_glbl_stretch
                            && num_bot.min(num_top) as f64 / total_fid as f64
                                > min_ratio_glbl_stretch))
                {
                    glb_stretch = 3;
                    glb_skew = 3;
                    self.made_zfactors = true;
                    numruns = 2;
                    if self.did_local_align != 0 && self.num_surfaces > 1 {
                        let mindens = min_on_surf as f64 / (self.raw_xsize * self.raw_ysize) as f64;
                        let min_in_area = mindens * nxpatch as f64 * nypatch as f64;
                        if min_in_area > min_surf_local_stretch {
                            loc_stretch = 3;
                            use_min_fids = min_fids_surf;
                        } else {
                            self.prn_log(
                                "Too few fiducials on minority surface to enable local stretching solution",
                                "\n",
                                false,
                            );
                        }
                    }
                } else {
                    self.prn_log(
                        "Too few fiducials on minority surface to enable stretching solution",
                        "\n",
                        false,
                    );
                }
            }

            if glb_skew == 0 && enable_skew_only.truthy() {
                if total_fid >= enable_skew_only.int() {
                    glb_skew = 3;
                    self.made_zfactors = true;
                    numruns = 2;
                    if self.did_local_align != 0 {
                        let mindens = total_fid as f64 / (self.raw_xsize * self.raw_ysize) as f64;
                        let min_in_area = mindens * nxpatch as f64 * nypatch as f64;
                        if min_in_area >= enable_skew_only.int() as f64 {
                            loc_skew = 3;
                        } else {
                            self.prn_log(
                                "Too few fiducials to enable local skew-only solution",
                                "\n",
                                false,
                            );
                        }
                    }
                } else {
                    self.prn_log(
                        "Too few fiducials to enable skew-only solution",
                        "\n",
                        false,
                    );
                }
            }
        }

        // Do restrictalign if there are enough fiducials for old ratio-based restricting,
        // or if doing either locals or stretch, as long as there are enough fiducials if
        // adding just stretch.  But skip it if run before on this data set
        let mut skip_restrict = (cross_val == 0 && total_fid >= min_tot_skip_restrict)
            || (cross_val != 0
                && self.did_local_align == 0
                && (glb_stretch == 0 || total_fid >= min_tot_skip_cross_val));
        if is_file_newer(
            &format!("restrictalign{}_local.log", self.axis_let),
            &format!("origcoms/align{}", self.axis_com),
        ) {
            skip_restrict = true;
        }

        let mut reloaded = false;

        // Just run alignment once or twice more.  Again, if robust gives a problem, try it
        // without; and this time if local alignments failed, drop that out too
        for run in 0..numruns {
            for loop_index in [0, 2] {
                let rob_text = if do_robust != 0 { "with" } else { "without" };
                let mut mess = "Doing fine alignment with ".to_owned();
                if glb_skew != 0 && glb_stretch == 0 {
                    mess += "skew only, with";
                } else {
                    if glb_stretch == 0 {
                        mess += "no";
                    }
                    mess += " distortion, with";
                }

                if (restricted != 0 && cross_val == 0) || self.did_local_align == 0 {
                    mess += "out";
                }
                mess += &format!(" local alignments, {rob_text} robust fitting");
                self.suppress_abort = loop_index == 0 && do_robust != 0;
                let mut del_tilt = self.total_del_tilt;
                if total_fid < min_tot_to_set_angle
                    || self
                        .lookup_directive(
                            &format!("{RUNTIME_PREFIX}TiltAlignment"),
                            "noAngleOffset",
                            0,
                            BOOL_VALUE,
                        )
                        .truthy()
                {
                    del_tilt = 0.;
                }
                let mut sedcom = vec![
                    sed_modify("SurfacesToAnalyze", &self.num_surfaces.to_string(), '/'),
                    sed_modify("LocalAlignments", &local_align.to_string(), '/'),
                    sed_modify("AngleOffset", &py_str_float(del_tilt), '/'),
                ];
                sedcom.extend(sed_del_and_add(
                    "ImagesAreBinned",
                    &self.coarse_binning.to_string(),
                    "OutputTransformFile",
                    '/',
                ));
                sedcom.extend(sed_del_and_add(
                    "ScaleShifts",
                    &format!("1,{}", self.coarse_binning),
                    "InputFile2",
                    '/',
                ));
                sedcom.extend(sed_del_and_add(
                    "RobustFitting",
                    &do_robust.to_string(),
                    "OutputTransformFile",
                    '/',
                ));
                if !reloaded {
                    sedcom.push(sed_modify("XStretchOption", &glb_stretch.to_string(), '/'));
                    sedcom.push(sed_modify("SkewOption", &glb_skew.to_string(), '/'));
                    sedcom.push(sed_modify(
                        "LocalStretchOption",
                        &loc_stretch.to_string(),
                        '/',
                    ));
                    sedcom.push(sed_modify("LocalSkewOption", &loc_skew.to_string(), '/'));
                    sedcom.push(sed_modify(
                        "MinFidsTotalAndEachSurface",
                        &format!("{min_fids_tot},{use_min_fids}"),
                        '/',
                    ));
                }

                if self.made_zfactors {
                    sedcom.extend(sed_del_and_add(
                        "OutputZFactorFile",
                        &format!("{}.zfac", self.data_name),
                        "OutputTransformFile",
                        '/',
                    ));
                }

                let result = self.modify_restrict_and_run_align(
                    &align_com,
                    &sedcom,
                    local_align,
                    &mess,
                    skip_restrict || run > 0 || loop_index > 0,
                );
                restricted = result.0;
                let no_robust = result.1;
                no_robust_outer = no_robust;
                if restricted > 0 && cross_val != 0 && run == 0 && loop_index == 0 {
                    reloaded = true;
                }
                if no_robust != 0 {
                    do_robust = 0;
                }

                if restricted < 0 {
                    if !self.suppress_abort {
                        return 1;
                    }
                    self.suppress_abort = false;
                    let mut found = false;
                    for line in self.latest_messages.clone() {
                        if line
                            .to_uppercase()
                            .contains("TOO FEW DATA POINTS TO DO ROBUST")
                        {
                            self.prn_log("Trying again without robust fitting", "\n", false);
                            do_robust = 0;
                            found = true;
                            break;
                        }
                        if line.contains("Minimum numbers of fiducials are too high") {
                            self.prn_log("Trying again without local alignments", "\n", false);
                            self.did_local_align = 0;
                            local_align = 0;
                            found = true;
                            break;
                        }
                    }
                    if !found {
                        return 1;
                    }

                    continue;
                }

                let tags = message_tags.clone();
                self.print_tagged_messages_file(&align_log, &tags);
                break;
            }

            self.suppress_abort = false;
            let two_surf = self.num_surfaces > 1;
            if self.analyze_align_log(two_surf, &mut angle_arr, false) != 0 {
                return 1;
            }
            self.total_del_tilt = angle_arr[0];
            self.xtilt_needed = angle_arr[1];
            self.fid_thickness = angle_arr[2];
            self.fid_inc_shift = angle_arr[3];
            self.recon_thickness = angle_arr[5];
            num_bot = angle_arr[6] as i64;
            num_top = angle_arr[7] as i64;
            let _ = (&mut total_fid, &mut min_on_surf, num_bot, num_top);
        }

        // Find the last residual output message and echo it as the final result, save for eval
        for ind in (0..self.latest_messages.len()).rev() {
            let line = self.latest_messages[ind].clone();
            if let Some(colon_ind) = line.find(':') {
                if colon_ind > 0 && line.contains("error") && line.contains("mean") {
                    let text = format!("Final align - {}{LOG_SUFFIX_TAG}", line.trim());
                    self.prn_log(&text, "\n", false);
                    self.prn_log("", "\n", false);
                    let lsplit: Vec<&str> = line[colon_ind + 1..].split_whitespace().collect();
                    self.final_align_resid = 0.;
                    if let Some(value) = lsplit.first().and_then(|value| value.parse::<f64>().ok())
                    {
                        self.final_align_resid = value;
                    }
                    break;
                }
            }
        }

        // Get the ta...log files
        let opts_names = [
            ('m', "Mappings"),
            ('e', "Error"),
            ('s', "Solution"),
            ('l', "Locals"),
            ('c', "Coordinates"),
            ('a', "Angles"),
            ('b', "Beamtilt"),
            ('r', "Robust"),
        ];
        for opt in opts_names {
            let name = format!("ta{}{}.log", opt.1, self.axis_let);
            match run_cmd(
                &format!("alignlog -{} {align_log}", opt.0),
                None,
                None,
                None,
                &[],
            ) {
                Ok(lines) => {
                    let mut part_lines = lines.unwrap_or_default();
                    for line in part_lines.iter_mut() {
                        *line = line.trim_end_matches(['\r', '\n']).to_owned();
                    }
                    if let Err(err) = write_text_file(&name, &part_lines, true) {
                        self.warning(&[format!("Error {err}")], true);
                    }
                }
                Err(_) => {
                    self.warning(&[format!("Error running alignlog -{}", opt.0)], true);
                }
            }
        }

        0
    }

    /// `makeAlignedStack` (`IMOD/pysrc/batchruntomo:3126`).
    pub fn make_aligned_stack(&mut self, position_binning: i64) -> i32 {
        let alipre = format!("{RUNTIME_PREFIX}AlignedStack");
        let mut outsize_text = PyVal::Str(String::new());
        let mess;
        if position_binning > 0 {
            self.ali_binning = position_binning;
            mess = "Making binned aligned stack for whole tomogram positioning".to_owned();
        } else {
            let binning = self.lookup_directive(&alipre, "binByFactor", 0, INT_VALUE);
            if self.test_directive_value(&binning, "AlignedStack.binByFactor", "integer") != 0 {
                return 1;
            }
            self.ali_binning = binning.int();
            outsize_text = self.lookup_directive(&alipre, "sizeInXandY", 0, STRING_VALUE);
            mess = "Making final aligned stack".to_owned();
        }

        let linear = self
            .lookup_directive(&alipre, "linearInterpolation", 0, BOOL_VALUE)
            .int();
        self.ali_xunbinned = self.raw_xsize;
        self.ali_yunbinned = self.raw_ysize;
        if self.ali_binning == 0 {
            self.ali_binning = 1;
        }

        let mut edfcom = vec![format!(
            "/batchruntomo.Stack.{}.SizeToOutputInXandY/d",
            self.axis_upper_let
        )];
        if outsize_text.truthy() {
            let replaced = outsize_text.text().replace(',', " ");
            let splits: Vec<&str> = replaced.split_whitespace().collect();
            let parsed = splits
                .first()
                .and_then(|value| value.parse::<i64>().ok())
                .zip(splits.get(1).and_then(|value| value.parse::<i64>().ok()));
            match parsed {
                Some((x, y)) => {
                    self.ali_xunbinned = x;
                    self.ali_yunbinned = y;
                    edfcom = self.edf_del_and_add(
                        &format!("Stack.{}.SizeToOutputInXandY", self.axis_upper_let),
                        &format!("{x},{y}"),
                        '/',
                    );
                }
                None => {
                    self.abort_set("Error converting aligned stack output size");
                    return 1;
                }
            }
        }

        if self.transpose_for_ali {
            std::mem::swap(&mut self.ali_xunbinned, &mut self.ali_yunbinned);
        }

        if position_binning < 0 && self.skip_align_stack_mods && self.raw_for_3dctf != 0 {
            return 0;
        }

        let (com_root, _process) = self.com_and_process_for_aligned_stack("");
        let mut sedcom;
        if self.if_montage {
            let mut frame_data = self.mont_frame_data;
            let err = self.montage_frame_values(
                self.raw_xsize,
                self.raw_ysize,
                self.transpose_for_ali,
                self.ali_xunbinned,
                self.ali_yunbinned,
                &mut frame_data,
            );
            self.mont_frame_data = frame_data;
            if err == 1 {
                self.report_imod_error(Some("Error running goodframe on montage sizes"));
                return 1;
            }
            if err != 0 {
                self.abort_set("Error converting output of goodframe to integers");
                return 1;
            }
            sedcom = vec![
                sed_modify(
                    "StartingAndEndingX",
                    &format!("{},{}", self.mont_frame_data[2], self.mont_frame_data[4]),
                    '/',
                ),
                sed_modify(
                    "StartingAndEndingY",
                    &format!("{},{}", self.mont_frame_data[3], self.mont_frame_data[5]),
                    '/',
                ),
                "/^InterpolationOrder/d".to_owned(),
            ];
            if linear != 0 {
                sedcom.push("/^TransformFile/a/InterpolationOrder    1/".to_owned());
            }

            self.ali_xunbinned = self.mont_frame_data[0];
            self.ali_yunbinned = self.mont_frame_data[1];
        } else {
            let expand = self.lookup_directive(
                &format!("{COM_PREFIX}newst"),
                "newstack.ExpandByFactor",
                0,
                FLOAT_VALUE,
            );
            self.expand_factor = if expand.truthy() { expand.float() } else { 1. };
            sedcom = vec![sed_modify(
                "SizeToOutputInXandY",
                &format!(
                    "{},{}",
                    (self.expand_factor * self.ali_xunbinned.div_euclid(self.ali_binning) as f64)
                        as i64,
                    (self.expand_factor * self.ali_yunbinned.div_euclid(self.ali_binning) as f64)
                        as i64
                ),
                '/',
            )];
            sedcom.extend(sed_del_and_add(
                "LinearInterpolation",
                &linear.to_string(),
                "TransformFile",
                '/',
            ));
        }

        sedcom.extend(sed_del_and_add(
            "BinByFactor",
            &self.ali_binning.to_string(),
            "TransformFile",
            '/',
        ));

        // Evaluate whether the aligned stack needs to be rebuilt, if it has already been
        // corrected and correction is to be done
        let mut need_rebuild = false;
        if !(position_binning > 0 || self.need_step(8.))
            && self.need_step(CTF_CORR_STEP_NUM)
            && self.correct_ctf != 0
            && Path::new(&dataset_filename(".ali", None, None)).exists()
        {
            let head_lines = match run_cmd(
                &format!("header {}", dataset_filename(".ali", None, None)),
                None,
                None,
                None,
                &[],
            ) {
                Ok(lines) => lines.unwrap_or_default(),
                Err(_) => {
                    self.report_imod_error(Some("Error reading header of existing aligned stack"));
                    return 1;
                }
            };
            for line in &head_lines {
                if line.contains("ctfPhaseFlip") || line.contains("CTF correct") {
                    need_rebuild = true;
                    self.prn_log(
                        "Remaking aligned stack because it has already been CTF corrected",
                        "\n",
                        false,
                    );
                }
            }
        }

        if position_binning > 0 || self.need_step(8.) || need_rebuild {
            let comfile = format!("{com_root}{}", self.axis_com);
            if self.modify_write_and_run_com(&comfile, &sedcom, None, &mess, false) != 0 {
                return 1;
            }
            if position_binning <= 0 {
                let rotation = py_str_float(self.axis_rotation);
                edfcom.extend(self.edf_del_and_add("ImageRotationForAliStack", &rotation, '/'));
                let flag = self.bool_string_for_edf(linear != 0);
                edfcom.extend(self.edf_del_and_add(
                    &format!("Stack.{}.UseLinearInterpolation", self.axis_upper_let),
                    &flag,
                    '/',
                ));
                if self.modify_edf_lines(&edfcom) != 0 {
                    return 1;
                }
            }
        }

        0
    }

    /// `CTFPlotAlignedStack` (`IMOD/pysrc/batchruntomo:3232`).
    pub fn ctf_plot_aligned_stack(&mut self) -> i32 {
        let replaced = self.replace_or_run_after_step(CTF_PLOT_STEP_NUM, false);
        if replaced != 0 {
            return replaced.max(0);
        }

        // Do autofitting with ctfplotter if enabled
        if self.need_step(CTF_PLOT_STEP_NUM) && self.correct_ctf != 0 && self.autofit_ctf.truthy() {
            let comfile = format!("ctfplotter{}", self.axis_com);
            let ctf_lines = self.read_text_file_report_err(&comfile, None);
            if ctf_lines.is_empty() {
                return 1;
            }
            let sedcom = vec![
                "/^ExpectedDefocus/a/SaveAndExit\t1/".to_owned(),
                format!(
                    "/^ExpectedDefocus/a/AutoFitRangeAndStep\t{}/",
                    self.autofit_ctf.text()
                ),
                "/^AngleRange/d".to_owned(),
            ];

            // Make existing defocus file a backup to avoid error messages
            make_backup_file(&format!("{}.defocus", self.data_name));
            let out_com = format!("ctfplotter_auto{}", self.axis_com);
            if self.modify_write_and_run_com(
                &out_com,
                &sedcom,
                Some(&ctf_lines),
                "Finding defocus for CTF correction with Ctfplotter",
                false,
            ) != 0
            {
                return 1;
            }

            return self.replace_or_run_after_step(CTF_PLOT_STEP_NUM, true);
        }

        0
    }

    /// `CTFCorrectAlignedStack` (`IMOD/pysrc/batchruntomo:3258`).
    pub fn ctf_correct_aligned_stack(&mut self) -> i32 {
        // Write the simple defocus file and use it if there is no defocus file; but if there
        // is, make sure it is in the com file
        let foc_file_name = format!("{}.defocus", self.data_name);
        let simple_name = format!("{}_simple.defocus", self.data_name);
        let mut sedcom;
        if !Path::new(&foc_file_name).exists() {
            let line = format!(
                "{} {} 0. 0. {}",
                self.zsize.div_euclid(2),
                self.zsize.div_euclid(2),
                py_str_float(self.defocus)
            );
            if self.write_text_file_report_err(&simple_name, &[line]) != 0 {
                return 1;
            }
            sedcom = vec![sed_modify("DefocusFile", &simple_name, '/')];
        } else {
            sedcom = vec![sed_modify("DefocusFile", &foc_file_name, '/')];
        }

        sedcom.push(sed_modify(
            "PixelSize",
            &py_str_float(self.ali_binning as f64 * self.pixel_size),
            '/',
        ));
        if self.use_gpu != 0 {
            sedcom.extend(sed_del_and_add("UseGPU", "0", "DefocusFile", '/'));
        }

        // See if X axis tilt is to be corrected; if so try to get it and add option
        let correct_xtilt = self
            .lookup_directive(
                &format!("{RUNTIME_PREFIX}CTFcorrection"),
                "correctForXAxisTilt",
                0,
                BOOL_VALUE,
            )
            .truthy();
        if correct_xtilt && self.no_xaxis_tilt == 0 {
            let mut angle_arr = [0f64; 8];
            let two_surf = self.num_surfaces > 1;
            if self.analyze_align_log(two_surf, &mut angle_arr, true) != 0 {
                return 1;
            }
            let mut xtilt_needed = angle_arr[1];
            let mut pitch_arr = [angle_arr[0], angle_arr[1], angle_arr[2], angle_arr[3]];
            let err = self.parse_tomopitch_log(&mut pitch_arr);
            if err < 0 {
                return 1;
            }
            if err == 0 {
                xtilt_needed = pitch_arr[1];
            }

            if xtilt_needed != 0. {
                sedcom.extend(sed_del_and_add(
                    "XAxisTilt",
                    &py_str_float(xtilt_needed),
                    "DefocusFile",
                    '/',
                ));
            }
        }

        let comfile = format!("ctfcorrection{}", self.axis_com);
        let ctf_lines = self.read_text_file_report_err(&comfile, None);
        if ctf_lines.is_empty() {
            return 1;
        }
        if pysed(
            &sedcom,
            PysedSrc::Lines(&ctf_lines),
            Some(&comfile),
            false,
            '/',
            true,
        )
        .is_err()
        {
            self.abort_set(&format!("Error modifying {comfile}"));
            return 1;
        }

        if self.skip_align_stack_mods {
            return 0;
        }

        let num_proc = self.number_of_processing_units();
        if num_proc < 1 {
            return 1;
        }

        if num_proc > 1 {
            let target = 2 * num_proc;
            let max_slices = (self.zsize + target - 1).div_euclid(target);
            if run_cmd(
                &format!("splitcorrection -m {max_slices} {comfile}"),
                None,
                None,
                None,
                &[],
            )
            .is_err()
            {
                self.report_imod_error(Some("Error trying to run ctfcorrection in parallel"));
                self.release_gpu_allocation();
                return 1;
            }
        }

        let use_gpu = self.use_gpu != 0;
        let err = self.run_one_process(
            &comfile,
            num_proc < 2,
            use_gpu,
            "Correcting for CTF with Ctfphaseflip",
            false,
        );
        self.release_gpu_allocation();
        if err != 0 {
            return 1;
        }

        let from = dataset_filename("_ctfcorr.ali", None, None);
        let to = dataset_filename(".ali", None, None);
        if self.use_file_as_replacement(&from, &to, false, true) != 0 {
            return 1;
        }

        0
    }

    /// `detectGoldIn3D` (`IMOD/pysrc/batchruntomo:3332`).
    pub fn detect_gold_in_3d(&mut self) -> i32 {
        // Get the binning or use the binning that etomo would assign, bead size / 5 rounded
        // to an integer with a minimum binned size of 4
        if self.erase_gold.truthy()
            && self.erase_gold.int() > 1
            && self.need_step(DETECT_3D_STEP_NUM)
        {
            let binning_directive = self.lookup_directive(
                &format!("{RUNTIME_PREFIX}GoldErasing"),
                "binning",
                0,
                INT_VALUE,
            );
            let mut binning = binning_directive.int();
            if !binning_directive.truthy() {
                // desired reduction is expFac * fidSize / optimal
                // actual reduction will be binning / expFac so binning is expFac * desired
                binning = (self.expand_factor.powi(2) * self.fid_size_pix
                    / FB3D_OPTIMAL_BINNED_SIZE)
                    .round() as i64;
                if binning > 1
                    && self.expand_factor.powi(2) * self.fid_size_pix / (binning as f64)
                        < FB3D_MIN_BINNED_SIZE
                {
                    binning -= 1;
                }
            }

            // Get the aligned stack if binning differs
            if binning != self.ali_binning {
                let (com_root, _process) = self.com_and_process_for_aligned_stack("");
                let mut comlines = vec![
                    format!("InputFile {com_root}{}", self.axis_com),
                    format!("BinningOfImages {binning}"),
                    format!("RootNameOfDataFiles {}", self.data_name),
                ];
                if !self.if_montage {
                    comlines.push(format!(
                        "OneParameterChange {COM_PREFIX}newst_3dfind{}.newstack.SizeToOutputInXandY={},{}",
                        self.axis_let,
                        ((self.ali_xunbinned as f64 * self.expand_factor) as i64)
                            .div_euclid(binning),
                        ((self.ali_yunbinned as f64 * self.expand_factor) as i64)
                            .div_euclid(binning)
                    ));
                }
                comlines.extend(self.later_com_directives(0));
                let out_com = format!("{com_root}_3dfind{}", self.axis_com);
                if self.make_and_run_one_com(
                    &comlines,
                    &out_com,
                    "Making aligned stack for erasing gold",
                    false,
                    false,
                ) != 0
                {
                    return 1;
                }
            }

            // Set up to get the reconstruction; get a thickness if any entered
            let thickness_directive = self.lookup_directive(
                &format!("{RUNTIME_PREFIX}GoldErasing"),
                "thickness",
                0,
                INT_VALUE,
            );
            let thickness;

            // use fid alignment if two surfaces, otherwise there needs to be a directive
            if !thickness_directive.truthy() {
                if self.num_surfaces > 1 {
                    let use_size = f64::max(15., self.fid_size_nm.unwrap_or(0.))
                        / (self.pixel_size / self.expand_factor);
                    thickness = 2
                        * (((1.1 * self.fid_thickness + 6. * use_size).round() as i64 + 1)
                            .div_euclid(2));
                } else {
                    self.abort_set(
                        "A GoldErasing.thickness directive must be supplied for 3D gold finding",
                    );
                    return 1;
                }
            } else {
                thickness = thickness_directive.int();
            }

            // Get the com file
            let comfile = format!("tilt_3dfind{}", self.axis_com);
            let mut comlines = vec![
                format!("InputFile tilt{}", self.axis_com),
                format!("OutputFile {comfile}"),
                format!("NamingStyle\t{}", self.name_style),
                format!(
                    "StackExtension\t{}",
                    &self.stack_extension[1.min(self.stack_extension.len())..]
                ),
                format!("BinningOfImages {binning}"),
                format!("RootNameOfDataFiles {}", self.data_name),
                format!("ThicknessToMake {thickness}"),
                format!("ShiftInY {}", py_str_float(self.fid_inc_shift)),
            ];
            if binning != self.ali_binning {
                comlines.push("Use3dfindAliInput 1".to_owned());
            }
            comlines.extend(self.later_com_directives(0));

            if run_cmd(
                "makecomfile -StandardInput",
                Some(&comlines),
                None,
                None,
                &[],
            )
            .is_err()
            {
                self.report_imod_error(Some(&format!("Error making {comfile}")));
                return 1;
            }

            // Make the reconstruction
            if self.split_and_run_tilt(&comfile, "Making tomogram for finding beads", 0) != 0 {
                return 1;
            }

            // Find the beads
            let mut comlines = vec![
                format!("RootNameOfDataFiles {}", self.data_name),
                format!("BinningOfImages {binning}"),
                format!("BeadSize {}", py_str_float(self.fid_size_pix)),
                format!(
                    "OneParameterChange {COM_PREFIX}findbeads3d{}.findbeads3d.StorageThreshold=-1",
                    self.axis_let
                ),
            ];

            let btlines = self.read_text_file_report_err(&format!("track{}", self.axis_com), None);
            if btlines.is_empty() {
                return 1;
            }
            if matches!(
                option_value(&btlines, "LightBeads", BOOL_VALUE, false, 0, None, None),
                Some(OptionValue::Boolean(true))
            ) {
                comlines.push(format!(
                    "OneParameterChange {COM_PREFIX}findbeads3d{}.findbeads3d.LightBeads=1",
                    self.axis_let
                ));
            }
            comlines.extend(self.later_com_directives(0));
            let out_com = format!("findbeads3d{}", self.axis_com);
            if self.make_and_run_one_com(
                &comlines,
                &out_com,
                "Finding beads in tomogram",
                false,
                false,
            ) != 0
            {
                return 1;
            }

            let sedcom = self.edf_del_and_add("AlignedStack.eraseGold", "2", '/');
            self.modify_edf_lines(&sedcom);
        }

        0
    }

    /// `eraseGoldInAlignedStack` (`IMOD/pysrc/batchruntomo:3424`).
    pub fn erase_gold_in_aligned_stack(&mut self) -> i32 {
        if self.erase_gold.int() > 1 {
            // Get the reprojection if using 3D method
            let comlines = vec![
                format!("InputFile tilt_3dfind{}", self.axis_com),
                format!("RootNameOfDataFiles {}", self.data_name),
            ];
            let out_com = format!("tilt_3dfind_reproject{}", self.axis_com);
            if self.make_and_run_one_com(
                &comlines,
                &out_com,
                "Reprojecting model of beads in tomogram onto aligned stack",
                false,
                false,
            ) != 0
            {
                return 1;
            }
        }
        // Or transform fiducial model
        else {
            let mut filled_in = "";
            if Path::new(&format!("{}_nogaps.fid", self.data_name)).exists() {
                filled_in = "_nogaps";
            }
            if run_cmd(
                &format!(
                    "xfmodel -xf {0}.tltxf {0}{filled_in}.fid {0}_erase.fid",
                    self.data_name
                ),
                None,
                None,
                None,
                &[],
            )
            .is_err()
            {
                self.report_imod_error(Some("Could not transform fiducials for erasing gold"));
                return 1;
            }
        }

        // Extend the model
        if self
            .lookup_directive(
                &format!("{RUNTIME_PREFIX}GoldErasing"),
                "extendModel",
                0,
                BOOL_VALUE,
            )
            .truthy()
        {
            let mut comlines = vec![format!("RootNameOfDataFiles {}", self.data_name)];
            if self.erase_gold.int() > 1 {
                comlines.push("ReplaceAboveAngle 0.".to_owned());
            }
            self.suppress_abort = true;
            let out_com = format!("extenderasemod{}", self.axis_com);
            let err = self.make_and_run_one_com(
                &comlines,
                &out_com,
                "Extending model on aligned stack",
                false,
                false,
            );
            self.suppress_abort = false;
            if err != 0 {
                self.warning(
                    &[format!("Error running extenderasemod{}", self.axis_com)],
                    true,
                );
            } else {
                let from = format!("{}_extended.fid", self.data_name);
                let to = format!("{}_erase.fid", self.data_name);
                if self.use_file_as_replacement(&from, &to, false, true) != 0 {
                    return 1;
                }
            }
        }

        // Erase the beads
        let extra_diam = self.lookup_directive(
            &format!("{RUNTIME_PREFIX}GoldErasing"),
            "extraDiameter",
            0,
            FLOAT_VALUE,
        );
        let extra_diam = if extra_diam.truthy() {
            extra_diam.float()
        } else {
            0.
        };
        let mut comlines = vec![
            format!("RootNameOfDataFiles {}", self.data_name),
            format!(
                "BeadSize {}",
                py_str_float(
                    self.expand_factor * self.fid_size_pix / self.ali_binning as f64 + extra_diam
                )
            ),
        ];
        comlines.extend(self.later_com_directives(0));
        if self
            .lookup_directive(
                &format!("{COM_PREFIX}golderaser"),
                "ccderaser.ExpandCircleIterations",
                0,
                STRING_VALUE,
            )
            .is_none()
        {
            comlines.push(format!(
                "OneParameterChange {COM_PREFIX}golderaser{}.ccderaser.ExpandCircleIterations=2",
                self.axis_let
            ));
        }
        let out_com = format!("golderaser{}", self.axis_com);
        let skip = self.skip_align_stack_mods;
        if self.make_and_run_one_com(
            &comlines,
            &out_com,
            "Erasing beads from aligned stack",
            false,
            skip,
        ) != 0
        {
            return 1;
        }
        if self.skip_align_stack_mods {
            return 0;
        }

        let from = dataset_filename("_erase.ali", None, None);
        let to = dataset_filename(".ali", None, None);
        if self.use_file_as_replacement(&from, &to, false, true) != 0 {
            return 1;
        }

        let value = self.erase_gold.int().min(2).to_string();
        let sedcom = self.edf_del_and_add("AlignedStack.eraseGold", &value, '/');
        self.modify_edf_lines(&sedcom);
        0
    }

    /// `filterAlignedStack` (`IMOD/pysrc/batchruntomo:3489`).
    pub fn filter_aligned_stack(&mut self) -> i32 {
        let this_step = 13.;
        let replaced = self.replace_or_run_after_step(this_step, false);
        if replaced != 0 {
            return replaced.max(0);
        }
        if self.filter_in_2d != 0 && self.need_step(this_step) {
            let comfile = format!("mtffilter{}", self.axis_com);
            let filt_lines = self.read_text_file_report_err(&comfile, None);
            if filt_lines.is_empty() {
                return 1;
            }
            let sedcom = vec![sed_modify(
                "PixelSize",
                &py_str_float(self.ali_binning as f64 * self.pixel_size),
                '/',
            )];
            let skip = self.skip_align_stack_mods;
            if self.modify_write_and_run_com(
                &comfile,
                &sedcom,
                Some(&filt_lines),
                "2D filtering the aligned stack with Mtffilter",
                skip,
            ) != 0
            {
                return 1;
            }
            if self.skip_align_stack_mods {
                return 0;
            }
            let from = dataset_filename("_filt.ali", None, None);
            let to = dataset_filename(".ali", None, None);
            if self.use_file_as_replacement(&from, &to, false, true) != 0 {
                return 1;
            }
            return self.replace_or_run_after_step(this_step, true);
        }

        0
    }
}

impl Brt {
    /// `modifyTiltComFile` (`IMOD/pysrc/batchruntomo:3514`).
    pub fn modify_tilt_com_file(&mut self, sample_thickness: i64) -> i32 {
        let comfile = format!("tilt{}", self.axis_com);
        let comlines = self.read_text_file_report_err(&comfile, None);
        if comlines.is_empty() {
            return 1;
        }

        let mut thickness;
        if sample_thickness != 0 {
            thickness = sample_thickness;
        } else {
            // Get different variants on thickness entry and insist on only one
            let thickness_directive =
                self.lookup_directive(&format!("{COM_PREFIX}tilt"), "tilt.THICKNESS", 0, INT_VALUE);
            let binned_thick = self.lookup_directive(
                &format!("{RUNTIME_PREFIX}Reconstruction"),
                "binnedThickness",
                0,
                INT_VALUE,
            );
            let fallback_thick = self.lookup_directive(
                &format!("{RUNTIME_PREFIX}Reconstruction"),
                "fallbackThickness",
                0,
                INT_VALUE,
            );
            if self.test_directive_value(&thickness_directive, "tilt.THICKNESS", "integer") != 0
                || self.test_directive_value(
                    &binned_thick,
                    "Reconstruction.binnedThickness",
                    "integer",
                ) != 0
            {
                return 1;
            }
            if self.test_directive_value(
                &fallback_thick,
                "Reconstruction.fallbackThickness",
                "integer",
            ) != 0
            {
                return 1;
            }

            if thickness_directive.truthy() && binned_thick.truthy() {
                self.abort_set(
                    "Both tilt.THICKNESS and Reconstruction.binnedThickness were entered",
                );
                return 1;
            }
            if thickness_directive.truthy() && fallback_thick.truthy() {
                self.abort_set(
                    "Both tilt.THICKNESS and Reconstruction.fallbackThickness were entered",
                );
                return 1;
            }
            if fallback_thick.truthy() && binned_thick.truthy() {
                self.abort_set(
                    "Both Reconstruction.fallbackThickness and Reconstruction.binnedThickness were entered",
                );
                return 1;
            }

            thickness = thickness_directive.int();
            if binned_thick.truthy() {
                thickness = self.ali_binning * binned_thick.int();
            }

            // If that did not provide a thickness, set it to fallback if there is nothing from
            // the alignment; otherwise take it from the alignment, with possible extra amount
            // but use the fallback if this is too thin
            if thickness == 0 {
                // First look for a thickness from tomopitch
                // Parse this log afresh; results from align log are already in globals
                let mut angle_arr = [0f64; 4];
                let err = self.parse_tomopitch_log(&mut angle_arr);
                if err < 0 {
                    return 1;
                }
                if err == 0 {
                    thickness = angle_arr[0] as i64;
                    self.xtilt_needed = angle_arr[1];
                }
                // Or use the align thickness if no positioning available
                else if self.recon_thickness != 0. {
                    thickness = 2 * ((self.recon_thickness.round() as i64 + 1).div_euclid(2));
                }

                // In either case, add the extra thickness if defined
                if thickness != 0 {
                    let extra = self.lookup_directive(
                        &format!("{RUNTIME_PREFIX}Reconstruction"),
                        "extraThickness",
                        0,
                        INT_VALUE,
                    );
                    if self.test_directive_value(&extra, "Reconstruction.extraThickness", "integer")
                        != 0
                    {
                        return 1;
                    }
                    if extra.truthy() {
                        thickness += extra.int();
                    }
                }

                // In any of these cases, fallback rules if thickness is too low or not set
                if fallback_thick.truthy()
                    && (thickness == 0
                        || (thickness as f64) < fallback_thick.float() * USE_FALLBACK_RATIO)
                {
                    let fallback = fallback_thick.int();
                    self.warning(
                        &[format!(
                            "Using fallback thickness of {fallback} because computed thickness, {thickness}, is less than {} of fallback",
                            py_str_float(USE_FALLBACK_RATIO)
                        )],
                        true,
                    );
                    thickness = fallback;
                }

                if thickness == 0 {
                    self.abort_set(
                        "No thickness was specified and neither fiducial alignment nor positioning gave a thickness to use",
                    );
                    return 1;
                }
            }
        }

        let (fullx, fully, sssx, sssy);
        if self.if_montage {
            fullx = self.mont_frame_data[6];
            fully = self.mont_frame_data[7];
            sssx = self.mont_frame_data[8];
            sssy = self.mont_frame_data[9];
        } else {
            let (mut fx, mut fy) = (self.raw_xsize, self.raw_ysize);
            if self.transpose_for_ali {
                std::mem::swap(&mut fx, &mut fy);
            }
            fullx = fx;
            fully = fy;
            sssx = -(self.ali_xunbinned - fullx).div_euclid(2);
            sssy = -(self.ali_yunbinned - fully).div_euclid(2);
        }

        let mut gpu_val = -1;
        if self.use_gpu != 0 {
            gpu_val = 0;
        }

        // Output 0 X axis tilt if that is desired
        if self.no_xaxis_tilt != 0 {
            self.xtilt_needed = 0.;
        }
        let mut sedcom = vec![
            sed_modify("IMAGEBINNED", &self.ali_binning.to_string(), '/'),
            sed_modify("XAXISTILT", &py_str_float(self.xtilt_needed), '/'),
            sed_modify("FULLIMAGE", &format!("{fullx} {fully}"), '/'),
            sed_modify("SUBSETSTART", &format!("{sssx} {sssy}"), '/'),
            sed_modify("THICKNESS", &thickness.to_string(), '/'),
        ];
        sedcom.extend(sed_del_and_add(
            "UseGPU",
            &gpu_val.to_string(),
            "XTILTFILE",
            '/',
        ));
        if self.axis_num == 0 {
            sedcom.push(sed_modify(
                "OutputFile",
                &dataset_filename("_full.rec", None, None),
                '/',
            ));
        }
        if self.use_gpu != 0 {
            sedcom.extend(sed_del_and_add("ActionIfGPUFails", "1,2", "XTILTFILE", '/'));
        } else {
            sedcom.push("/^ActionIfGPUFails/d".to_owned());
        }
        if self.did_local_align != 0 {
            sedcom.extend(sed_del_and_add(
                "LOCALFILE",
                &format!("{}local.xf", self.data_name),
                "XTILTFILE",
                '/',
            ));
        } else {
            sedcom.push("/^LOCALFILE/d".to_owned());
        }
        if self.made_zfactors {
            sedcom.extend(sed_del_and_add(
                "ZFACTORFILE",
                &format!("{}.zfac", self.data_name),
                "XTILTFILE",
                '/',
            ));
        } else {
            sedcom.push("/^ZFACTORFILE/d".to_owned());
        }

        // Handle change in scaling if they turned off log and did not supply a scale
        let logbase = option_value(&comlines, "LOG", FLOAT_VALUE, false, 0, None, None);
        let logbase_truthy = match &logbase {
            Some(OptionValue::Floats(values)) => !values.is_empty(),
            Some(_) => true,
            None => false,
        };
        if !logbase_truthy
            && !self
                .lookup_directive(&format!("{COM_PREFIX}tilt"), "tilt.SCALE", 0, STRING_VALUE)
                .truthy()
        {
            let scale_arr = option_value(&comlines, "SCALE", FLOAT_VALUE, false, 0, None, None);
            let values = match &scale_arr {
                Some(OptionValue::Floats(values)) => values.clone(),
                _ => Vec::new(),
            };
            if values.len() < 2 {
                self.abort_set(&format!(
                    "Cannot modify SCALE value in tilt{}{} for linear scaling",
                    self.axis_let, self.com_ext
                ));
                return 1;
            }

            // Copytomocoms produces scales from 1000 to 40 depending on X size.
            // Linear requires scale to be reduced by 5000, giving numbers from 0.2 to 0.008
            // 3 is a safe dividing point for deciding whether this has already happened
            if values[1] > 3. {
                sedcom.push(sed_modify(
                    "SCALE",
                    &format!(
                        "{} {:.3}",
                        py_str_float(values[0] as f64),
                        values[1] / 5000.
                    ),
                    '/',
                ));
            }
        }

        if pysed(
            &sedcom,
            PysedSrc::Lines(&comlines),
            Some(&comfile),
            false,
            '/',
            true,
        )
        .is_err()
        {
            self.abort_set(&format!("Error modifying {comfile}"));
            return 1;
        }
        0
    }

    /// `make3dCtfCorrectedTomogram` (`IMOD/pysrc/batchruntomo:3662`).
    pub fn make_3d_ctf_corrected_tomogram(
        &mut self,
        tilt_com: &str,
        num_proc: i64,
        rec_name: &str,
    ) -> i32 {
        if self.do_both_recons && self.raw_for_3dctf == 0 && self.make_aligned_stack(0) != 0 {
            return 1;
        }

        let mut comlines = vec![
            format!("InputFile {tilt_com}"),
            format!("ThicknessToMake {}", self.ctf3d_slab_thick),
        ];
        if !self
            .lookup_directive(
                &format!("{COM_PREFIX}ctf3dsetup"),
                "ctf3dsetup.RunSlabsInParallel",
                0,
                BOOL_VALUE,
            )
            .truthy()
        {
            comlines.push(format!("NumberOfProcessors {num_proc}"));
        }
        comlines.extend(self.later_com_directives(0));
        if self.erase_gold.truthy() {
            comlines.push(format!(
                "OneParameterChange {COM_PREFIX}ctf3dsetup{}.ctf3dsetup.EraseFiducials=1",
                self.axis_let
            ));
        }
        if self
            .lookup_directive(
                &format!("{RUNTIME_PREFIX}AlignedStack"),
                "filterStack",
                0,
                BOOL_VALUE,
            )
            .truthy()
        {
            comlines.push(format!(
                "OneParameterChange {COM_PREFIX}ctf3dsetup{}.ctf3dsetup.FilterIn2D=1",
                self.axis_let
            ));
        }
        let out_com = format!("ctf3dsetup{}", self.axis_com);
        if self.make_and_run_one_com(
            &comlines,
            &out_com,
            "Making command files to make 3D CTF-corrected tomogram",
            false,
            false,
        ) != 0
        {
            return 1;
        }
        let use_gpu = self.use_gpu != 0;
        let com = format!("ctf3d{}", self.com_ext);
        if self.run_one_process(
            &com,
            false,
            use_gpu,
            "Making 3D CTF corrected tomogram",
            false,
        ) != 0
        {
            return 1;
        }

        let ctf_name = dataset_filename("_3dctf.rec", None, None);
        if self.rename_and_abort(
            &ctf_name,
            rec_name,
            "Renaming {} to {} after make CTF corrected reconstruction",
        ) != 0
        {
            return 1;
        }
        0
    }

    /// `generateTomogram` (`IMOD/pysrc/batchruntomo:3692`).
    pub fn generate_tomogram(&mut self) -> i32 {
        let rec_root = if self.dual_axis {
            format!("{}{}", self.set_name, self.axis_let)
        } else {
            format!("{}_full", self.set_name)
        };
        let mut rec_name = dataset_filename(".rec", Some(&rec_root), None);
        let (do_sirt, do_fake_sirt, do_both, sirt_name, fake_name) =
            self.get_sirt_rec_name(&rec_root);
        if do_sirt && do_fake_sirt {
            self.abort_set("You cannot do both SIRT and a SIRT-like filter");
            return 1;
        }

        let tilt_com = format!("tilt{}", self.axis_com);

        // This is needed for both SIRT and 3D CTF
        let num_proc = self.number_of_processing_units();
        if num_proc < 1 {
            return 1;
        }

        if (do_both || (!do_sirt && self.ctf3d_slab_thick == 0))
            && self.split_and_run_tilt(&tilt_com, "Making tomogram by back-projection", num_proc)
                != 0
        {
            self.release_gpu_allocation();
            return 1;
        }
        if do_fake_sirt && do_both {
            let temp_com_name = format!("tilt{}_slf{}", self.axis_let, self.axis_com);
            if self.rename_and_abort(
                &rec_name,
                fake_name.as_deref().unwrap_or(""),
                "Renaming {} to {} before regular back-projection",
            ) != 0
            {
                self.release_gpu_allocation();
                return 1;
            }
            if self.rename_and_abort(&tilt_com, &temp_com_name, "Renaming {} to {}") != 0 {
                self.release_gpu_allocation();
                return 1;
            }
            if pysed(
                &["/FakeSIRTiterations/d".to_owned()],
                PysedSrc::File(&temp_com_name),
                Some(&tilt_com),
                false,
                '/',
                true,
            )
            .is_err()
            {
                self.abort_set(&format!("Error modifying {temp_com_name}"));
                self.release_gpu_allocation();
                return 1;
            }

            if self.ctf3d_slab_thick != 0 {
                let name = rec_name.clone();
                if self.make_3d_ctf_corrected_tomogram(&tilt_com, num_proc, &name) != 0 {
                    self.release_gpu_allocation();
                    return 1;
                }
            } else if self.split_and_run_tilt(
                &tilt_com,
                "Making tomogram by regular back-projection",
                num_proc,
            ) != 0
            {
                self.release_gpu_allocation();
                return 1;
            }

            make_backup_file(&tilt_com);
            if self.rename_and_abort(&temp_com_name, &tilt_com, "Renaming {} to {}") != 0 {
                self.release_gpu_allocation();
                return 1;
            }
        }

        if do_sirt {
            let mut comlines = self.later_com_directives(0);
            comlines.push(format!(
                "OneParameterChange {COM_PREFIX}sirtsetup{}.sirtsetup.NumberOfProcessors={num_proc}",
                self.axis_let
            ));
            let out_com = format!("sirtsetup{}", self.axis_com);
            if self.make_and_run_one_com(
                &comlines,
                &out_com,
                "Making command files to run SIRT",
                false,
                false,
            ) != 0
            {
                self.release_gpu_allocation();
                return 1;
            }
            let use_gpu = self.use_gpu != 0;
            let com = format!("tilt{}_sirt{}", self.axis_let, self.com_ext);
            if self.run_one_process(&com, false, use_gpu, "Making tomogram with SIRT", false) != 0 {
                self.release_gpu_allocation();
                return 1;
            }
        }

        if !(do_fake_sirt && do_both) && self.ctf3d_slab_thick != 0 {
            let name = rec_name.clone();
            if self.make_3d_ctf_corrected_tomogram(&tilt_com, num_proc, &name) != 0 {
                self.release_gpu_allocation();
                return 1;
            }
        }

        // Put the correct size of the reconstruction in the edf file when finished
        self.release_gpu_allocation();
        if do_sirt && !do_both {
            rec_name = sirt_name.unwrap_or_default();
        }
        let (recx, recy, recz) = match get_mrc_size(&rec_name) {
            Ok(size) => size,
            Err(_) => {
                self.report_imod_error(Some(&format!(
                    "Could not get size of reconstruction {rec_name}"
                )));
                return 1;
            }
        };

        let prefix = format!("{}.tomogramSize.", self.axis_upper_let);
        let mut sedcom = self.edf_del_and_add(&format!("{prefix}Columns"), &recx.to_string(), '/');
        sedcom.extend(self.edf_del_and_add(&format!("{prefix}Rows"), &recy.to_string(), '/'));
        sedcom.extend(self.edf_del_and_add(&format!("{prefix}Sections"), &recz.to_string(), '/'));
        if self.modify_edf_lines(&sedcom) != 0 {
            return 1;
        }
        0
    }

    /// `runFinalTiltalign` (`IMOD/pysrc/batchruntomo:3780`).
    pub fn run_final_tiltalign(&mut self, sedcom: &[String], mess: &str) -> i32 {
        let has_skip = std::env::var_os("TILTALIGN_SKIP_CROSS_VAL").is_some();
        if !has_skip {
            unsafe { std::env::set_var("TILTALIGN_SKIP_CROSS_VAL", "1") };
        }
        let comfile = format!("align{}", self.axis_com);
        let lines = self.align_lines.clone();
        let err = self.modify_write_and_run_com(
            &comfile,
            sedcom,
            Some(&lines),
            &format!("Doing final alignment {mess}"),
            false,
        );
        if !has_skip {
            unsafe { std::env::remove_var("TILTALIGN_SKIP_CROSS_VAL") };
        }
        if err != 0 {
            return 1;
        }
        match pysed(sedcom, PysedSrc::Lines(&lines), None, false, '/', true) {
            Ok(Some(new_lines)) => self.align_lines = new_lines,
            _ => {
                self.abort_set("Error modifying alignment lines: <class 'str'>");
                return 1;
            }
        }
        if self.post_process_tiltalign() != 0 {
            return 1;
        }
        0
    }

    /// `positionTomogram` (`IMOD/pysrc/batchruntomo:3800`).
    pub fn position_tomogram(&mut self) -> i32 {
        let num_scales = [4i64, 2, 2, 1];
        let box_sizes = [32i64, 20, 16, 12];
        let block_sizes = [100i64, 60, 48, 36];
        let min_center_fid = 4.;

        // If no positioning and conditions are satisfied to center on gold, get align results
        // and run align if there are enough on each surface
        if self.pos_sample_type <= 0 {
            if !self.center_on_gold || self.num_surfaces < 2 {
                return 0;
            }
            let mut angle_arr = [0f64; 8];
            if self.analyze_align_log(true, &mut angle_arr, true) != 0 {
                return 1;
            }
            if angle_arr[6] < min_center_fid || angle_arr[7] < min_center_fid {
                return 0;
            }
            let sedcom = vec![sed_modify("AxisZShift", &py_str_float(angle_arr[4]), '/')];
            if self.run_final_tiltalign(&sedcom, "to center on gold") != 0 {
                return 1;
            }
        }

        // Get thickness
        let thickness_directive = self.lookup_directive(
            &format!("{RUNTIME_PREFIX}Positioning"),
            "thickness",
            0,
            INT_VALUE,
        );
        if self.test_directive_value(&thickness_directive, "Positioning.thickness", "integer") != 0
        {
            return 1;
        }
        let mut thickness = thickness_directive.int();

        // Cryosample
        if self.pos_sample_type > 1 {
            if thickness == 0 {
                self.abort_set("Thickness must be entered for cryopositioning");
                return 1;
            }
            if self.manage_gpu_allocation() != 0 {
                return 1;
            }

            let mut find_gold = 2;
            if self.patch_track || self.fiducialless != 0 {
                find_gold = 0;
                if self
                    .lookup_directive(
                        &format!("{RUNTIME_PREFIX}Positioning"),
                        "hasGoldBeads",
                        0,
                        BOOL_VALUE,
                    )
                    .truthy()
                {
                    if self.fid_size_pix <= 0. {
                        self.abort_set(
                            "Gold bead size must be entered to find beads in cryopositioning",
                        );
                    }
                    find_gold = 2;
                }
            }

            let mut comlines = vec![
                format!("RootNameOfDataFiles {}", self.data_name),
                format!("ThicknessToMake {thickness}"),
                format!("FindBeadsInVolume {find_gold}"),
            ];
            if !self.gpu_list.is_empty() {
                comlines.push("UseGPU 0".to_owned());
            }

            self.suppress_abort = true;
            let out_com = format!("cryoposition{}", self.axis_com);
            let use_gpu = !self.gpu_list.is_empty();
            let err = self.make_and_run_one_com(
                &comlines,
                &out_com,
                "Positioning tomogram with Cryoposition",
                use_gpu,
                false,
            );
            self.suppress_abort = false;
            let cplines =
                self.read_text_file_report_err(&format!("cryoposition{}.log", self.axis_let), None);
            if cplines.is_empty() {
                return 1;
            }
            if err != 0 {
                self.warning(
                    &["Tomogram positioning did not work; proceeding if possible".to_owned()],
                    true,
                );
                return 0;
            }

            let binning = self.find_tagged_value(&cplines, "stack with binning", '=', INT_VALUE);
            self.ali_binning = binning.int();
            if !binning.truthy() {
                self.abort_set(&format!(
                    "Cannot find binning in cryoposition{}.log",
                    self.axis_let
                ));
                return 1;
            }
        } else {
            // Plastic section: Get binning and make aligned stack
            let size = self.raw_xsize.max(self.raw_ysize);
            let binning = self.lookup_directive(
                &format!("{RUNTIME_PREFIX}Positioning"),
                "binByFactor",
                0,
                INT_VALUE,
            );
            if self.test_directive_value(&binning, "Positioning.binByFactor", "integer") != 0 {
                return 1;
            }
            self.ali_binning = binning.int();
            if !binning.truthy() {
                for bin in [1i64, 2, 3, 4] {
                    if size / bin < POSITION_BINNING_TARGET || bin == 4 {
                        self.ali_binning = bin;
                        let value = self.ali_binning;
                        self.prn_log(
                            &format!("Binning for positioning tomogram set to {value}"),
                            "\n",
                            false,
                        );
                        break;
                    }
                }
            }

            let ali_binning = self.ali_binning;
            if self.make_aligned_stack(ali_binning) != 0 {
                return 1;
            }

            // Set default thickness if needed and make tomogram
            if thickness == 0 {
                for ind in (0..DFLT_POS_THICKNESSES.len()).rev() {
                    if size > SIZES_FOR_POS_THICKNESSES[ind] {
                        thickness = DFLT_POS_THICKNESSES[ind];
                        self.prn_log(
                            &format!("Thickness for positioning tomogram set to {thickness}"),
                            "\n",
                            false,
                        );
                        break;
                    }
                }
            }

            let tilt_com = format!("tilt{}", self.axis_com);
            if self.modify_tilt_com_file(thickness) != 0
                || self.split_and_run_tilt(&tilt_com, "Making tomogram for positioning", 0) != 0
            {
                return 1;
            }

            // Run findsection to get the model
            let filename = if self.axis_num == 0 {
                dataset_filename("_full.rec", None, None)
            } else {
                dataset_filename(".rec", None, None)
            };
            let bin_ind = (self.ali_binning.min(box_sizes.len() as i64) - 1) as usize;
            let pitch_mod = format!("tomopitch{}.mod", self.axis_let);
            let com_root = format!("findsection_pos{}", self.axis_let);
            let err = self.run_find_section(
                &filename,
                num_scales[bin_ind],
                box_sizes[bin_ind],
                Some(&pitch_mod),
                None,
                Some(block_sizes[bin_ind]),
                true,
                &com_root,
            );

            // Distinguish an OK findsection failure from other errors that issued ABORT SET
            if err < 0 {
                self.warning(
                    &[
                        "Finding the section for positioning did not work; proceeding if possible"
                            .to_owned(),
                    ],
                    true,
                );
                return 0;
            }
            if err > 0 {
                return 1;
            }
        }

        // Get existing Z shift, angle offset, and X tilt
        let tilt_lines = self.read_text_file_report_err(&format!("tilt{}", self.axis_com), None);
        if tilt_lines.is_empty() {
            return 1;
        }
        let mut z_shift_orig = 0.;
        let mut angle_offset_orig = 0.;
        let offset;
        if self.fiducialless != 0 {
            let z_shift_arr = option_value(&tilt_lines, "SHIFT", FLOAT_VALUE, false, 2, None, None);
            if let Some(OptionValue::Floats(values)) = &z_shift_arr {
                if values.len() > 1 {
                    z_shift_orig = values[1] as f64;
                }
            }
            offset = option_value(&tilt_lines, "OFFSET", FLOAT_VALUE, false, 1, None, None);
        } else {
            let lines = self.align_lines.clone();
            let z_shift = option_value(&lines, "AxisZShift", FLOAT_VALUE, false, 1, None, None);
            if let Some(OptionValue::Floats(values)) = &z_shift {
                if !values.is_empty() && values[0] != 0. {
                    z_shift_orig = values[0] as f64;
                }
            }
            offset = option_value(&lines, "AngleOffset", FLOAT_VALUE, false, 1, None, None);
        }

        if let Some(OptionValue::Floats(values)) = &offset {
            if !values.is_empty() && values[0] != 0. {
                angle_offset_orig = values[0] as f64;
            }
        }
        let mut xtilt_orig = 0.;
        let xtilt = option_value(&tilt_lines, "XAXISTILT", FLOAT_VALUE, false, 1, None, None);
        let mut xtilt_value = match &xtilt {
            Some(OptionValue::Floats(values)) if !values.is_empty() => values[0] as f64,
            _ => 0.,
        };
        if self.no_xaxis_tilt != 0 {
            xtilt_value = 0.;
        }
        if xtilt_value != 0. && self.pos_sample_type == 1 {
            xtilt_orig = xtilt_value;
        }

        let mut edfcom = self.edf_del_and_add(
            &format!("{}.sample.AngleOffset", self.axis_edf_let),
            &py_str_float(py_round(angle_offset_orig, 3)),
            '/',
        );
        edfcom.extend(self.edf_del_and_add(
            &format!("{}.sample.AxisZShift", self.axis_edf_let),
            &py_str_float(py_round(z_shift_orig, 2)),
            '/',
        ));
        edfcom.extend(self.edf_del_and_add(
            &format!("{}.sample.XAXISTILT", self.axis_edf_let),
            &py_str_float(py_round(xtilt_orig, 3)),
            '/',
        ));
        edfcom.extend(self.edf_del_and_add(
            &format!("{}.positioning.PosSampleType", self.axis_edf_let),
            &self.pos_sample_type.to_string(),
            '/',
        ));
        if self.modify_edf_lines(&edfcom) != 0 {
            return 1;
        }

        // Modify tomopitch and run it
        let mut sedcom = sed_del_and_add(
            "ScaleFactor",
            &self.ali_binning.to_string(),
            "ModelFile",
            '/',
        );
        sedcom.extend(sed_del_and_add(
            "AngleOffsetOld",
            &py_str_float(angle_offset_orig),
            "ModelFile",
            '/',
        ));
        sedcom.extend(sed_del_and_add(
            "ZShiftOld",
            &py_str_float(z_shift_orig),
            "ModelFile",
            '/',
        ));
        sedcom.extend(sed_del_and_add(
            "XAxisTiltOld",
            &py_str_float(xtilt_orig),
            "ModelFile",
            '/',
        ));
        if self.pos_sample_type > 1
            && !self
                .lookup_directive(
                    &format!("{COM_PREFIX}tomopitch"),
                    "tomopitch.ExtraThickness",
                    0,
                    INT_VALUE,
                )
                .truthy()
        {
            sedcom.push(sed_modify(
                "ExtraThickness",
                &CRYO_POS_EXTRA_THICK.to_string(),
                '/',
            ));
        }

        self.suppress_abort = true;
        let comfile = format!("tomopitch{}", self.axis_com);
        let err = self.modify_write_and_run_com(
            &comfile,
            &sedcom,
            None,
            "Finding angles and thickness",
            false,
        );
        self.suppress_abort = false;
        if err != 0 {
            self.warning(
                &["Tomopitch failed; proceeding if possible".to_owned()],
                true,
            );
            return 1;
        }

        // Parse this log
        let mut angle_arr = [0f64; 4];
        let err = self.parse_tomopitch_log(&mut angle_arr);
        if err < 0 {
            return 1;
        }
        let thickness = angle_arr[0] as i64;
        let xtilt = angle_arr[1];
        let offset = angle_arr[2];
        let z_shift = angle_arr[3];
        if err != 0 || thickness == 0 {
            self.prn_log(
                "Tomopitch did not produce a good result; proceeding if possible",
                "\n",
                false,
            );
            return 0;
        }

        // Write tilt.com with the X-axis tilt and shift/angle offset for fidless
        self.prn_log(
            &format!(
                "Positioning gave thickness {thickness}, X-axis tilt {xtilt:.2}, angle offset {offset:.2}, Z shift {z_shift:.1}"
            ),
            "\n",
            false,
        );
        let mut sedcom = vec![
            sed_modify("XAXISTILT", &py_str_float(xtilt), '/'),
            sed_modify("THICKNESS", &thickness.to_string(), '/'),
        ];
        if self.fiducialless != 0 {
            sedcom.extend(sed_del_and_add(
                "OFFSET",
                &py_str_float(offset),
                "OutputFile",
                '/',
            ));
            sedcom.extend(sed_del_and_add(
                "SHIFT",
                &format!("0. {}", py_str_float(z_shift)),
                "OutputFile",
                '/',
            ));
        }
        let comfile = format!("tilt{}", self.axis_com);
        if pysed(
            &sedcom,
            PysedSrc::Lines(&tilt_lines),
            Some(&comfile),
            false,
            '/',
            true,
        )
        .is_err()
        {
            self.abort_set(&format!("Error modifying tilt{}", self.axis_com));
            return 1;
        }

        // Write align.com with Z shift and offset and run it
        if self.fiducialless == 0 {
            let sedcom = vec![
                sed_modify("AxisZShift", &py_str_float(z_shift), '/'),
                sed_modify("AngleOffset", &py_str_float(offset), '/'),
            ];
            if self.run_final_tiltalign(&sedcom, "with new position") != 0 {
                return 1;
            }
        }

        0
    }
}

// COMBINE FUNCTIONS

impl Brt {
    /// `getAutoPatchfitParams` (`IMOD/pysrc/batchruntomo:3999`).
    pub fn get_auto_patchfit_params(&self) -> (String, String) {
        let extra_targets = self.lookup_directive(COMBINE_PREFIX, "extraTargets", 0, STRING_VALUE);
        let final_patch_size =
            self.lookup_directive(COMBINE_PREFIX, "finalPatchSize", 0, STRING_VALUE);
        let extra_targets = if !extra_targets.truthy() {
            DFLT_EXTRA_WARP_TARGETS.to_owned()
        } else {
            extra_targets.text().to_owned()
        };
        let final_patch_size = if !final_patch_size.truthy() {
            DFLT_FINAL_PATCH_SIZE.to_owned()
        } else {
            final_patch_size.text().to_owned()
        };
        (final_patch_size, extra_targets)
    }

    /// `setupCombine` (`IMOD/pysrc/batchruntomo:4010`).
    pub fn setup_combine(&mut self) -> i32 {
        let top_bot_ind = 2usize;
        let mut patch_size = "M".to_owned();
        self.use_vol_match = 0;
        let mut edf_vol = "false";
        let mut top_bot_a = [0i64; 6];
        let mut top_bot_b = [0i64; 6];
        if self.fiducialless != 0 || self.patch_track || !Path::new("transferfid.coord").exists() {
            self.use_vol_match = 1;
            edf_vol = "true";
        }

        // Get the right files in place if SIRT is involved
        let use_sirt_if_both = self.lookup_directive(COMBINE_PREFIX, "doSIRTifBoth", 0, INT_VALUE);
        if self.test_directive_value(&use_sirt_if_both, "Combine.doSIRTifBoth", "integer") != 0 {
            return 1;
        }
        let mut rec_root = self.set_name.clone() + "a";
        for axis_num in [1usize, 2usize] {
            self.axis_num = axis_num;
            let rec_name = dataset_filename(".rec", Some(&rec_root), None);
            let (do_sirt, do_fake_sirt, do_both, sirt_name, fake_name) =
                self.get_sirt_rec_name(&rec_root);
            let use_sirt = (do_sirt && (use_sirt_if_both.truthy() || !do_both))
                || (do_fake_sirt && do_both && use_sirt_if_both.truthy());
            if use_sirt {
                let mut use_sirt_name = sirt_name.clone().unwrap_or_default();
                let mut like_text = "";
                if do_fake_sirt {
                    use_sirt_name = fake_name.clone().unwrap_or_default();
                    like_text = "-like";
                }
                let sirt_exists = Path::new(&use_sirt_name).exists();
                let rec_exists = Path::new(&rec_name).exists();
                let bp_name = dataset_filename("_BP.rec", Some(&rec_root), None);
                if rec_exists && !sirt_exists {
                    self.prn_log(
                        &format!(
                            "The file {rec_name} exists and the SIRT{like_text} reconstruction {use_sirt_name} does not; assuming it was already renamed"
                        ),
                        "\n",
                        false,
                    );
                } else if !sirt_exists {
                    self.abort_set(&format!(
                        "Neither the SIRT{like_text} reconstruction {use_sirt_name} nor the file {rec_name} exist; cannot proceed"
                    ));
                    return 1;
                } else {
                    if rec_exists && do_both {
                        if self.use_file_as_replacement(&rec_name, &bp_name, false, true) != 0 {
                            return 1;
                        }
                        self.prn_log(
                            &format!(
                                "Renamed back-projection reconstruction {rec_name} to: {bp_name}"
                            ),
                            "\n",
                            false,
                        );
                    }
                    if self.use_file_as_replacement(&use_sirt_name, &rec_name, false, true) != 0 {
                        return 1;
                    }
                    self.prn_log(
                        &format!(
                            "Renamed SIRT{like_text} reconstruction {use_sirt_name} to: {rec_name}"
                        ),
                        "\n",
                        false,
                    );
                }
            }

            rec_root = self.set_name.clone() + "b";
        }

        self.axis_num = 0;

        // Get default top and bottom Z limits in case of failure
        let mut z_range_a;
        let mut z_range_b;
        match (
            get_mrc_size(&dataset_filename("a.rec", None, None)),
            get_mrc_size(&dataset_filename("b.rec", None, None)),
        ) {
            (Ok((_nx, ya, _nz)), Ok((_nxb, yb, _nzb))) => {
                z_range_a = ya as i64;
                top_bot_a[top_bot_ind] = 1;
                top_bot_a[top_bot_ind + 1] = z_range_a;
                z_range_b = yb as i64;
                top_bot_b[top_bot_ind] = 1;
                top_bot_b[top_bot_ind + 1] = z_range_b;
            }
            _ => {
                self.report_imod_error(Some("Error running header on reconstruction files"));
                return 1;
            }
        }

        // Get the directives and use the defaults if none
        let match_atob_ratio =
            self.lookup_directive(COMBINE_PREFIX, "matchAtoBThickRatio", 0, FLOAT_VALUE);
        if self.test_directive_value(&match_atob_ratio, "Combine.matchAtoBThickRatio", "float") != 0
        {
            return 1;
        }
        let match_atob_ratio = if !match_atob_ratio.truthy() {
            DFLT_MATCH_ATO_BRATIO
        } else {
            match_atob_ratio.float()
        };
        let num_scales = self.lookup_directive(COMBINE_PREFIX, "findSecNumScales", 0, INT_VALUE);
        if self.test_directive_value(&num_scales, "Combine.findSecNumScales", "integer") != 0 {
            return 1;
        }
        let num_scales = if !num_scales.truthy() {
            DFLT_COMBINE_NUM_SCALES
        } else {
            num_scales.int()
        };
        let box_size = self.lookup_directive(COMBINE_PREFIX, "findSecBoxSize", 0, INT_VALUE);
        if self.test_directive_value(&box_size, "Combine.findSecBoxSize", "integer") != 0 {
            return 1;
        }
        let box_size = if !box_size.truthy() {
            DFLT_COMBINE_BOX_SIZE
        } else {
            box_size.int()
        };

        // Need to find limits in A unless match ratio is very high
        let need_find_a = match_atob_ratio < 4.;
        let mut err_a = 0;
        if need_find_a {
            let name = dataset_filename("a.rec", None, None);
            err_a = self.run_find_section(
                &name,
                num_scales,
                box_size,
                None,
                Some(&mut top_bot_a),
                None,
                true,
                "findsection_zlima",
            );
            if err_a > 0 {
                return 1;
            }
            if err_a == 0 {
                z_range_a = top_bot_a[top_bot_ind + 1] - top_bot_a[top_bot_ind];
                if z_range_a < 4 {
                    self.abort_set("Findsection failed to find a useful Z range in A tomogram");
                }
            }
        }

        // Need to find limits in B unless match ratio is very low
        let need_find_b = match_atob_ratio > 0.25;
        let mut err_b = 0;
        if need_find_b {
            let name = dataset_filename("b.rec", None, None);
            err_b = self.run_find_section(
                &name,
                num_scales,
                box_size,
                None,
                Some(&mut top_bot_b),
                None,
                true,
                "findsection_zlimb",
            );
            if err_b > 0 {
                return 1;
            }
            if err_b == 0 {
                z_range_b = top_bot_b[top_bot_ind + 1] - top_bot_b[top_bot_ind];
                if z_range_b < 4 {
                    self.abort_set("Findsection failed to find a useful Z range in B tomogram");
                }
            }
        }

        // Match A to B if ratio was very high or B is sufficiently smaller
        let match_atob = !need_find_a
            || (need_find_b
                && err_a == 0
                && err_b == 0
                && z_range_b as f64 <= match_atob_ratio * z_range_a as f64);
        let mut edf_match = "B_to_A";
        if match_atob {
            top_bot_a = top_bot_b;
            edf_match = "A_to_B";
            err_a = err_b;
        }

        // Give warning and proceed if the matched to tomogram had error in Z limits
        if err_a != 0 {
            self.warning(
                &["Findsection failed to find Z limits for combine, proceeding anyway".to_owned()],
                true,
            );
        }

        // Determine number of surfaces, make sure 2 is still OK
        // If in doubt, just set to 1 and see what happens
        let mut num_surf = self.num_surfaces;
        if self.use_vol_match != 0 {
            num_surf = 2;
        } else {
            // Do the appropriate axis of align log
            if match_atob {
                self.axis_let = "b".to_owned();
            }
            let mut angle_arr = [0f64; 8];
            let two_surf = self.num_surfaces > 1;
            if self.analyze_align_log(two_surf, &mut angle_arr, true) != 0 {
                return 1;
            }
            self.axis_let = "a".to_owned();
            self.total_del_tilt = angle_arr[0];
            self.xtilt_needed = angle_arr[1];
            self.fid_thickness = angle_arr[2];
            self.fid_inc_shift = angle_arr[3];
            self.recon_thickness = angle_arr[5];
            let num_bot = angle_arr[6] as i64;
            let num_top = angle_arr[7] as i64;
            let mut mess = String::new();
            if num_bot + num_top < SOLVEMATCH_MIN_FIDS {
                self.use_vol_match = 1;
                self.prn_log(
                    "Using Dualvolmatch instead of Solvematch because there are too few fiducials",
                    "\n",
                    false,
                );
            } else if num_surf > 1 {
                if self.fid_thickness
                    < (top_bot_a[top_bot_ind + 1] - top_bot_a[top_bot_ind]) as f64 / 2.
                {
                    num_surf = 1;
                    mess = "because fiducial extent is much smaller than Z range".to_owned();
                } else {
                    let fallback_thick = self.lookup_directive(
                        &format!("{RUNTIME_PREFIX}Reconstruction"),
                        "fallbackThickness",
                        0,
                        INT_VALUE,
                    );
                    if fallback_thick.is_int()
                        && self.recon_thickness < fallback_thick.float() * USE_FALLBACK_RATIO
                    {
                        num_surf = 1;
                        mess = "because fiducial extent is much smaller than fallback thickness"
                            .to_owned();
                    }
                }

                if num_surf > 1
                    && (num_bot < SOLVEMATCH_MIN_EACH_SIDE
                        || num_top < SOLVEMATCH_MIN_EACH_SIDE
                        || f64::min(
                            num_bot as f64 / num_top as f64,
                            num_top as f64 / num_top as f64,
                        ) < SOLVEMATCH_MIN_SIDE_RATIO)
                {
                    num_surf = 1;
                    mess = "because there are too few fiducials on one surface".to_owned();
                }

                if num_surf == 1 {
                    self.prn_log(
                        &format!("Setting number of surfaces for Solvematch to 1 {mess}"),
                        "\n",
                        false,
                    );
                }
            }
        }

        let mut edf_surf = "BothSides";
        if num_surf == 1 {
            edf_surf = "OneSide";
        }

        // Get the rest of the combine setup directives
        let patch_in = self.lookup_directive(COMBINE_PREFIX, "patchSize", 0, STRING_VALUE);
        if patch_in.truthy() {
            patch_size = patch_in.text().to_owned();
        }
        let (final_patch_size, extra_targets) = self.get_auto_patchfit_params();

        let wedge_frac = self.lookup_directive(COMBINE_PREFIX, "wedgeReduction", 0, FLOAT_VALUE);
        let low_radius = self.lookup_directive(COMBINE_PREFIX, "lowFromBothRadius", 0, FLOAT_VALUE);
        if self.test_directive_value(&wedge_frac, "Combine.wedgeReduction", "float") != 0
            || self.test_directive_value(&low_radius, "Combine.lowFromBothRadius", "float") != 0
        {
            return 1;
        }
        let mut com_lines = vec![
            format!("RootName {}", self.set_name),
            "WarningsToStandardOut 1".to_owned(),
            format!("NamingStyle {}", self.name_style),
            format!(
                "StackExtension {}",
                &self.stack_extension[1.min(self.stack_extension.len())..]
            ),
            "TransferPointFile transferfid.coord".to_owned(),
            format!("SurfaceModelType {num_surf}"),
            format!(
                "ZLowerAndUpper {},{}",
                top_bot_a[top_bot_ind],
                top_bot_a[top_bot_ind + 1]
            ),
            format!("PatchTypeOrXYZ {patch_size}"),
            format!("AutoPatchFinalSize {final_patch_size}"),
        ];
        if !extra_targets.is_empty() {
            com_lines.push(format!("ExtraResidualTargets {extra_targets}"));
        }
        if self.use_vol_match != 0 {
            com_lines.push("InitialVolumeMatching 1".to_owned());
        }
        com_lines.extend(self.later_com_directives(0));

        if match_atob {
            com_lines.push("MatchAtoB 1".to_owned());
        }
        if wedge_frac.truthy() {
            com_lines.push(format!(
                "WedgeReductionFraction {}",
                py_str_float(wedge_frac.float())
            ));
        }
        if low_radius.truthy() {
            com_lines.push(format!(
                "LowFromBothRadius {}",
                py_str_float(low_radius.float())
            ));
        }

        if run_cmd(
            "setupcombine -StandardInput",
            Some(&com_lines),
            Some("stdout"),
            None,
            &[],
        )
        .is_err()
        {
            self.report_imod_error(Some("Error running setupcombine"));
            return 1;
        }

        // Now try to modify the edf file.  Z limits are fatal.  First word -> batchruntomo
        let mut sedcom = self.edf_del_and_add("Combine.MatchMode", edf_match, '/');
        sedcom.extend(self.edf_del_and_add("Combine.FiducialMatch", edf_surf, '/'));
        sedcom.extend(self.edf_del_and_add("Combine.InitialVolumeMatching", edf_vol, '/'));
        sedcom.extend(self.edf_del_and_add("Combine.PatchSize", &patch_size, '/'));
        sedcom.extend(self.edf_del_and_add("Combine.FinalPatchSize", &final_patch_size, '/'));
        sedcom.extend(self.edf_del_and_add("Combine.ExtraResidualTargets", &extra_targets, '/'));

        if self.modify_edf_lines(&sedcom) != 0 {
            return 1;
        }

        // Finally, modify matchvol1.com to preserve a larger B volume size
        // The transposed ny,nz match setupcombine usage for better or worse
        let (nza, nzb);
        match (
            get_mrc_size(&dataset_filename("a.rec", None, None)),
            get_mrc_size(&dataset_filename("b.rec", None, None)),
        ) {
            (Ok((nx_a, z_a, y_a)), Ok((_nxb, z_b, _yb))) => {
                self.nx_rec_a = nx_a as i64;
                nza = z_a as i64;
                self.ny_rec_a = y_a as i64;
                nzb = z_b as i64;
            }
            _ => {
                self.report_imod_error(Some("Error getting sizes of axis reconstruction files"));
                return 1;
            }
        }

        let sedcom = vec![sed_modify(
            "OutputSizeXYZ",
            &format!("{} {} {}", self.nx_rec_a, nza.max(nzb), self.ny_rec_a),
            '/',
        )];
        let com_file = format!("matchvol1{}", self.com_ext);
        if pysed(
            &sedcom,
            PysedSrc::File(&com_file),
            Some(&com_file),
            false,
            '/',
            true,
        )
        .is_err()
        {
            self.abort_set(&format!("Error modifying {com_file}"));
            return 1;
        }

        0
    }

    /// `initialCombineMatch` (`IMOD/pysrc/batchruntomo:4250`).
    pub fn initial_combine_match(&mut self) -> i32 {
        // If we are coming in at this step, need to reconstruct whether volume matching
        // is supposed to be used
        if self.use_vol_match < 0 {
            let name = self.set_name.clone() + ".edf";
            let edf_lines = self.read_text_file_report_err(&name, None);
            if edf_lines.is_empty() {
                return 1;
            }
            for line in &edf_lines {
                if line.contains("InitialVolumeMatching") {
                    if line.contains("true") {
                        self.use_vol_match = 1;
                    } else if line.contains("false") {
                        self.use_vol_match = 0;
                    }
                    break;
                }
            }
            if self.use_vol_match < 0 {
                self.abort_set("Cannot determine if initial volume matching was set up to be used");
                return 1;
            }
        }

        // Set up for type of process and whether may need to run it twice
        let (com_root, num_loop, tags): (&str, i32, Vec<MessageTag>) = if self.use_vol_match != 0 {
            (
                "dualvolmatch",
                1,
                vec![
                    MessageTag("unbinned mean residual", 4, None),
                    MessageTag("implies a center", 4, None),
                    MessageTag("Falling back", 4, None),
                ],
            )
        } else {
            (
                "solvematch",
                2,
                vec![
                    MessageTag("Mean residual", 4, None),
                    MessageTag("Scaling along", 0, None),
                    MessageTag("Local fits", 0, None),
                    MessageTag("Average mean", 4, None),
                ],
            )
        };

        for _loop in 0..num_loop {
            // Run the process and process and read log regardless
            self.suppress_abort = true;
            let com = format!("{com_root}{}", self.com_ext);
            let err = self.run_one_process(
                &com,
                true,
                false,
                "Getting initial alignment between volumes",
                false,
            );
            self.suppress_abort = false;

            // Warnings got printed regardless, errors got printed if it failed, so set tags to
            // avoid duplicate messages
            let mut use_tags = vec![MessageTag("ERROR:", 0, None)];
            use_tags.extend(tags.clone());
            if err != 0 {
                use_tags = tags.clone();
            }
            self.print_tagged_messages_file(&format!("{com_root}.log"), &use_tags);
            let log_lines = self.read_text_file_report_err(&format!("{com_root}.log"), None);
            if log_lines.is_empty() {
                return 1;
            }

            // For vol match, see if need to change the matchvol thickness
            if self.use_vol_match != 0 && err == 0 {
                for line in &log_lines {
                    if line.contains("may need to set thickness") {
                        let lsplit: Vec<&str> = line.split_whitespace().collect();
                        let new_thick = match lsplit.last().and_then(|v| v.parse::<i64>().ok()) {
                            Some(value) => value,
                            None => {
                                self.abort_set(
                                    "Error trying to get suggested thickness for matchvol",
                                );
                                0
                            }
                        };

                        let sedcom = vec![sed_modify(
                            "OutputSizeXYZ",
                            &format!("{} {new_thick} {}", self.nx_rec_a, self.ny_rec_a),
                            '/',
                        )];
                        let com_file = format!("matchvol1{}", self.com_ext);
                        if pysed(
                            &sedcom,
                            PysedSrc::File(&com_file),
                            Some(&com_file),
                            false,
                            '/',
                            true,
                        )
                        .is_err()
                        {
                            self.abort_set(&format!("Error modifying{com_file}"));
                            return 1;
                        }
                        self.prn_log(
                            &format!(
                                "Thickness for matched volume set to {new_thick} as suggested"
                            ),
                            "\n",
                            false,
                        );
                        break;
                    }
                }
            }

            // For solvematch look first to see if it suggests using one surface, modify the
            // com and loop for another run
            if self.use_vol_match == 0 {
                let mut redo = false;
                for line in &log_lines {
                    if line.contains("Try specifying") && line.contains("on one surface") {
                        let com_file = format!("solvematch{}", self.com_ext);
                        if pysed(
                            &[sed_modify("SurfacesOrUseModels", "1", '/')],
                            PysedSrc::File(&com_file),
                            Some(&com_file),
                            false,
                            '/',
                            true,
                        )
                        .is_err()
                        {
                            self.abort_set(&format!("Error modifying{com_file}"));
                            return 1;
                        }
                        self.prn_log("Rerunning with one surface as suggested", "\n", false);
                        redo = true;
                        break;
                    }
                }

                if redo {
                    continue;
                }
            }

            // Otherwise done if no error
            if err == 0 {
                return 0;
            }
            let mut init_shift: Vec<i64> = Vec::new();
            let mut shift_limit = 0i64;

            // If there was an error, try to find initial shift as well as center shift
            for line in &log_lines {
                if line.contains("InitialShiftXYZ") && line.contains("needs") {
                    let lsplit: Vec<&str> = line.split_whitespace().collect();
                    let mut values = Vec::new();
                    let mut ok = lsplit.len() >= 3;
                    if ok {
                        for ind in 0..3 {
                            match lsplit[lsplit.len() - 3 + ind].parse::<i64>() {
                                Ok(value) => values.push(value),
                                Err(_) => {
                                    ok = false;
                                    break;
                                }
                            }
                        }
                    }
                    if ok {
                        init_shift = values;
                    } else {
                        let line = line.clone();
                        self.abort_set(&format!("Error trying to get initial shift from :{line}"));
                    }
                }
                if line.contains("CenterShiftLimit") && line.contains("avoid stopping") {
                    let lsplit: Vec<&str> = line.split_whitespace().collect();
                    if let Some(value) = lsplit.last().and_then(|v| v.parse::<i64>().ok()) {
                        shift_limit = value;
                    }
                }

                // If error line is found that says it's OK, modify patchcorr.com
                if line.contains("ERROR:")
                    && ((line.to_uppercase().contains("INITIAL SHIFT")
                        && line.to_uppercase().contains("SOLUTION IS OK"))
                        || line.contains("Initial shift needs"))
                {
                    if !init_shift.is_empty() {
                        self.prn_log(
                            &format!(
                                "Setting initial shift for Corrsearch3d to {} {} {}",
                                init_shift[0], init_shift[2], init_shift[1]
                            ),
                            "\n",
                            false,
                        );
                        let sedcm = sed_del_and_add(
                            "InitialShiftXYZ",
                            &format!("{},{},{}", init_shift[0], init_shift[1], init_shift[2]),
                            "FlipYZMessages",
                            '/',
                        );
                        let com_file = format!("patchcorr{}", self.com_ext);
                        if pysed(
                            &sedcm,
                            PysedSrc::File(&com_file),
                            Some(&com_file),
                            false,
                            '/',
                            true,
                        )
                        .is_err()
                        {
                            self.abort_set(&format!("Error modifying {com_file}"));
                            return 1;
                        }

                        // If that succeeded, modify com file with shift limit
                        if shift_limit != 0 {
                            let com = format!("{com_root}{}", self.com_ext);
                            if pysed(
                                &[sed_modify(
                                    "CenterShiftLimit",
                                    &shift_limit.to_string(),
                                    '/',
                                )],
                                PysedSrc::File(&com),
                                Some(&com),
                                false,
                                '/',
                                true,
                            )
                            .is_err()
                            {
                                self.abort_set(
                                    "Error modifying solvematch.com with new shift limit",
                                );
                                return 1;
                            }
                        }
                        return 0;
                    }

                    self.abort_set(&format!(
                        "Initial shift is required to go on, but was not found in {com_root}.log"
                    ));
                    return 1;
                }
            }
        }

        self.abort_set("Error getting initial alignment between volumes");
        1
    }

    /// `alignAndCombineAxes` (`IMOD/pysrc/batchruntomo:4392`).
    pub fn align_and_combine_axes(&mut self) -> i32 {
        let com = format!("matchvol1{}", self.com_ext);
        if self.need_step(17.)
            && self.run_one_process(&com, true, false, "Making initial matching volume", true) != 0
        {
            return 1;
        }

        if self.need_step(18.) {
            let (final_patch_size, extra_targets) = self.get_auto_patchfit_params();
            let mut com_lines = vec![
                "$autopatchfit -StandardInput".to_owned(),
                format!("FinalPatchTypeOrXYZ {final_patch_size}"),
            ];
            if !extra_targets.is_empty() {
                com_lines.push(format!("ExtraResidualTargets {extra_targets}"));
            }
            let com_file = format!("autopatchfit{}", self.com_ext);
            if self.write_text_file_report_err(&com_file, &com_lines) != 0 {
                return 1;
            }
            let err = self.run_one_process(
                &com_file,
                true,
                false,
                "Doing patch correlation and fitting to local patches",
                true,
            );
            let tags = [
                MessageTag("ERROR:", 0, None),
                MessageTag("WARNING:", 0, None),
                MessageTag("Using ", 4, None),
                MessageTag("Changing ", 4, None),
                MessageTag("Adding ", 4, None),
                MessageTag("FINDWARP -", 0, None),
                MessageTag("found a good", 4, Some(LOG_SUFFIX_TAG)),
            ];
            self.print_tagged_messages_file("autopatchfit.log", &tags);
            if err != 0 {
                self.abort_set("Cannot align the two tomograms");
                return 1;
            }
        }

        if self.need_step(19.) {
            if self.parallel_cpu > 1 && run_cmd("splitcombine", None, None, None, &[]).is_err() {
                self.report_imod_error(Some(
                    "Error trying to run splitcombine to combine in parallel",
                ));
                return 1;
            }

            let com = format!("volcombine{}", self.com_ext);
            if self.run_one_process(
                &com,
                self.parallel_cpu < 2,
                false,
                "Combining the two volumes",
                false,
            ) != 0
            {
                return 1;
            }
        }
        0
    }
}

impl Brt {
    /// `trimVolume` (`IMOD/pysrc/batchruntomo:4429`).
    pub fn trim_volume(&mut self) -> i32 {
        let trim_prefix = format!("{RUNTIME_PREFIX}Trimvol");
        let size_keys = [
            "sizeInX",
            "sizeInY",
            "thickness",
            "scaleFromX",
            "scaleFromY",
            "scaleFromZ",
            "findSecAddThickness",
        ];
        let replaced = self.replace_or_run_after_step(TRIM_STEP_NUM, false);
        if replaced != 0 {
            return replaced.max(0);
        }

        if !self
            .lookup_directive(
                &format!("{RUNTIME_PREFIX}Postprocess"),
                "doTrimvol",
                0,
                BOOL_VALUE,
            )
            .truthy()
        {
            if self
                .lookup_directive(
                    &format!("{RUNTIME_PREFIX}Postprocess"),
                    "doTrimvol",
                    0,
                    STRING_VALUE,
                )
                .truthy()
            {
                return 0;
            }
            if self
                .lookup_directive(&trim_prefix, "reorient", 0, INT_VALUE)
                .is_none()
            {
                let mut found = false;
                for key in size_keys {
                    if !self
                        .lookup_directive(&trim_prefix, key, 0, FLOAT_VALUE)
                        .is_none()
                    {
                        found = true;
                        break;
                    }
                }
                if !found {
                    return 0;
                }
            }
        }

        if self.axis_num > 0
            && !self
                .lookup_directive(&trim_prefix, "doAorBofDualAxis", 0, BOOL_VALUE)
                .truthy()
        {
            return 0;
        }

        let use_find_sec = !self
            .lookup_directive(&trim_prefix, "findSecAddThickness", 0, FLOAT_VALUE)
            .is_none();
        if !self
            .lookup_directive(&trim_prefix, "thickness", 0, FLOAT_VALUE)
            .is_none()
            && use_find_sec
        {
            self.abort_set(
                "Cannot use both Trimvol.findSecAddThickness and Trimvol.thickness directives",
            );
            return 1;
        }

        let reorient_value = self.lookup_directive(&trim_prefix, "reorient", 0, INT_VALUE);
        let reorient = if reorient_value.is_none() {
            2
        } else {
            reorient_value.int()
        };
        if reorient_value.is_str() || !(0..=2).contains(&reorient) {
            self.abort_set("The value for \"reorient\" must be 0, 1, or 2");
            return 1;
        }
        let flag = self.bool_string_for_edf(reorient != 0);
        let mut edfcom = self.edf_del_and_add("TrimvolFlipped", &flag, '/');
        let flag = self.bool_string_for_edf(reorient == 1);
        edfcom.extend(self.edf_del_and_add("Trimvol.SwapYZ", &flag, '/'));
        let flag = self.bool_string_for_edf(reorient == 2);
        edfcom.extend(self.edf_del_and_add("Trimvol.RotateX", &flag, '/'));

        // Figure out what volume to trim and get its size
        let mut rec_root = self.data_name.clone();
        let mut trim_names = vec![dataset_filename(".rec", None, None)];
        let mut fs_name = "findsection_trim".to_owned();
        if self.dual_axis && self.axis_num > 0 {
            trim_names = vec![dataset_filename("_trim.rec", None, None)];
            fs_name += &self.axis_let;
        } else if self.dual_axis {
            rec_root = "sum".to_owned();
        } else {
            rec_root = self.data_name.clone() + "_full";
        }
        let mut rec_names = vec![dataset_filename(".rec", Some(&rec_root), None)];

        let mut trim_both = 0i64;
        if !self.dual_axis || self.axis_num > 0 {
            let (do_sirt, do_fake_sirt, do_both, sirt_name, fake_name) =
                self.get_sirt_rec_name(&rec_root);
            if do_sirt || do_fake_sirt {
                let mut use_sirt = true;
                if do_both {
                    let value = self.lookup_directive(&trim_prefix, "doSIRTifBoth", 0, INT_VALUE);
                    if self.test_directive_value(&value, "Trimvol.doSIRTifBoth", "integer") != 0 {
                        return 1;
                    }
                    trim_both = if value.is_none() { 0 } else { value.int() };
                    use_sirt = trim_both > 0;
                }

                if use_sirt {
                    if do_sirt && sirt_name.is_none() {
                        self.abort_set(
                            "Cannot trim SIRT output, cannot find LeaveIterations directive",
                        );
                        return 1;
                    }

                    let mut use_sirt_name = sirt_name.clone().unwrap_or_default();
                    if do_fake_sirt {
                        use_sirt_name = fake_name.clone().unwrap_or_default();
                    }

                    rec_names = vec![use_sirt_name];
                    if do_both && trim_both > 1 {
                        let bp_root = match trim_names[0].rfind('.') {
                            Some(index) if index > 0 => trim_names[0][..index].to_owned(),
                            _ => trim_names[0].clone(),
                        };
                        rec_names.push(dataset_filename(".rec", Some(&rec_root), None));
                        trim_names.push(dataset_filename("_BP.rec", Some(&bp_root), None));
                    }
                }
            }
        }

        for name in &rec_names {
            if !Path::new(name).exists() {
                let name = name.clone();
                self.abort_set(&format!("The file to be trimmed, {name}, does not exist"));
                return 1;
            }
        }
        let (nxrec, nzrec, nyrec) = match get_mrc_size(&rec_names[0]) {
            Ok((x, y, z)) => (x as i64, y as i64, z as i64),
            Err(_) => {
                self.report_imod_error(Some("Could not get size of volume to trim"));
                return 1;
            }
        };

        edfcom.extend(self.edf_del_and_add("TrimVol.Input.NColumns", &nxrec.to_string(), '/'));
        edfcom.extend(self.edf_del_and_add("TrimVol.Input.NRows", &nzrec.to_string(), '/'));
        edfcom.extend(self.edf_del_and_add("TrimVol.Input.NSections", &nyrec.to_string(), '/'));

        // Determine size parameters, convert fractions, and check them
        let mut sizes_scales: Vec<Option<i64>> = Vec::new();
        let base_vals = [nxrec, nyrec, nzrec, nxrec, nyrec, nzrec, nzrec];
        for key in size_keys {
            let val = self.lookup_directive(&trim_prefix, key, 0, FLOAT_VALUE);
            if self.test_directive_value(&val, &format!("Trimvol.{key}"), "float") != 0 {
                return 1;
            }
            let mut size = None;
            if !val.is_none() {
                let base = base_vals[sizes_scales.len()];
                let mut low_lim = 40;
                if sizes_scales.len() % 3 == 2 {
                    low_lim = 4;
                }
                let value = val.float();
                let mut computed = value.round() as i64;
                if computed <= 1 {
                    computed = (value * base as f64).round() as i64;
                }
                if (sizes_scales.len() < 6
                    && (value < 0.02 || computed < low_lim || computed > base))
                    || (sizes_scales.len() == 6 && (computed < 0 || computed > base))
                {
                    self.abort_set(&format!(
                        "The size specified by the \"Trimvol.{key}\" directive is out of the allowed range"
                    ));
                    return 1;
                }
                size = Some(computed);
            }

            sizes_scales.push(size);
        }

        // Run findsection on the BP file if any
        let mut z_limits = [0i64; 6];
        if use_find_sec {
            let name = rec_names[rec_names.len() - 1].clone();
            if self.run_find_section(
                &name,
                4,
                32,
                None,
                Some(&mut z_limits),
                None,
                false,
                &fs_name,
            ) != 0
            {
                return 1;
            }
            if z_limits[5] - z_limits[4] <= 0 {
                self.abort_set(&format!(
                    "Findsection on {name} did not find limits for the section"
                ));
                return 1;
            }
        }

        // Set up size options in command
        let mut trimcom = "$trimvol -f".to_owned();
        if reorient == 1 {
            trimcom += " -yz";
        } else if reorient == 2 {
            trimcom += " -rx";
        }

        let opt_names = [" -nx ", " -ny ", " -nz "];
        let edf_names = ["X", "Y", "Z"];
        for ind in 0..3 {
            if let Some(value) = sizes_scales[ind].filter(|value| *value != 0) {
                trimcom += &format!("{}{value}", opt_names[ind]);
                let half_trim = (base_vals[ind] - value).div_euclid(2);
                edfcom.extend(self.edf_del_and_add(
                    &format!("Trimvol.{}Min", edf_names[ind]),
                    &(half_trim + 1).to_string(),
                    '/',
                ));
                edfcom.extend(self.edf_del_and_add(
                    &format!("Trimvol.{}Max", edf_names[ind]),
                    &(base_vals[ind] - half_trim).to_string(),
                    '/',
                ));
            }
        }

        if use_find_sec {
            let tzmin = 1.max(z_limits[4] - sizes_scales[6].unwrap_or(0));
            let tzmax = nzrec.min(z_limits[5] + sizes_scales[6].unwrap_or(0));
            trimcom += &format!(" -z {tzmin},{tzmax}");
            edfcom.extend(self.edf_del_and_add("Trimvol.ZMin", &tzmin.to_string(), '/'));
            edfcom.extend(self.edf_del_and_add("Trimvol.ZMax", &tzmax.to_string(), '/'));
        }

        // If any scaling is specified, add options.  Set default for Z, leave others at
        // trimvol defaults
        let scale_mean_sd = self.lookup_directive(&trim_prefix, "scaleToMeanSD", 0, STRING_VALUE);
        if scale_mean_sd.truthy() {
            trimcom += &format!(" -meansd \"{}\"", scale_mean_sd.text());
        }
        let do_scaling = sizes_scales[3].unwrap_or(0) != 0
            || sizes_scales[4].unwrap_or(0) != 0
            || sizes_scales[5].unwrap_or(0) != 0
            || scale_mean_sd.truthy();
        let flag = self.bool_string_for_edf(do_scaling);
        edfcom.extend(self.edf_del_and_add("Trimvol.ConvertToBytes", &flag, '/'));
        if do_scaling {
            if sizes_scales[5].unwrap_or(0) == 0 {
                sizes_scales[5] = Some(nzrec.min(4.max(nzrec.div_euclid(3))));
            }
            for ind in 3..6 {
                let edf_names = ["ScaleXM", "ScaleYM", "ScaleSectionM"];
                if let Some(value) = sizes_scales[ind].filter(|value| *value != 0) {
                    let start = 1 + (base_vals[ind] - value).div_euclid(2);
                    let end = start + value - 1;
                    let letter = char::from(b'x' + (ind - 3) as u8);
                    trimcom += &format!(" -s{letter} {start},{end}");
                    edfcom.extend(self.edf_del_and_add(
                        &format!("Trimvol.{}in", edf_names[ind - 3]),
                        &start.to_string(),
                        '/',
                    ));
                    edfcom.extend(self.edf_del_and_add(
                        &format!("Trimvol.{}ax", edf_names[ind - 3]),
                        &end.to_string(),
                        '/',
                    ));
                }
            }
        }

        // Do one or two volumes
        for ind in 0..rec_names.len() {
            let mess = format!(
                "Running trimvol on {} to create {}",
                rec_names[ind], trim_names[ind]
            );
            let comfile = format!("trimvol{}", self.axis_com);
            let line = format!("{trimcom} \"{}\" \"{}\"", rec_names[ind], trim_names[ind]);
            if self.write_text_file_report_err(&comfile, &[line]) != 0 {
                return 1;
            }
            if self.run_one_process(&comfile, true, false, &mess, false) != 0 {
                return 1;
            }
        }

        if self.modify_edf_lines(&edfcom) != 0 {
            return 1;
        }

        self.replace_or_run_after_step(TRIM_STEP_NUM, true)
    }

    /// `runNAD` (`IMOD/pysrc/batchruntomo:4615`).
    pub fn run_nad(&mut self) -> i32 {
        let rec_file = dataset_filename(".rec", None, None);
        let root = match rec_file.rfind('.') {
            Some(index) if index > 0 => rec_file[..index].to_owned(),
            _ => rec_file.clone(),
        };
        let nad_file = dataset_filename(".nad", Some(&root), None);
        let com_file = format!("autoNAD{}", self.com_ext);

        // Get the directives and make sure both are there, and make sure file exists
        let iterations =
            self.lookup_directive(&format!("{RUNTIME_PREFIX}NAD"), "iterations", 0, INT_VALUE);
        let kvalue =
            self.lookup_directive(&format!("{RUNTIME_PREFIX}NAD"), "Kvalue", 0, FLOAT_VALUE);
        if self.test_directive_value(&iterations, "NAD.iterations", "integer") != 0
            || self.test_directive_value(&kvalue, "NAD.Kvalue", "float") != 0
        {
            return 1;
        }
        if !iterations.truthy() && !kvalue.truthy() {
            return 0;
        }
        if (iterations.truthy() && !kvalue.truthy()) || (kvalue.truthy() && !iterations.truthy()) {
            self.abort_set(
                "Both \"iterations\" and \"Kvalue\" directives need to be present to run NAD",
            );
            return 1;
        }
        if !Path::new(&rec_file).exists() {
            self.abort_set("You must post-process with Trimvol in order to run NAD");
            return 1;
        }

        // Get optional memory entry, compute the chunksize, write the dummy com file
        let memory = self.lookup_directive(
            &format!("{RUNTIME_PREFIX}NAD"),
            "chunkMemoryMB",
            0,
            INT_VALUE,
        );
        if self.test_directive_value(&iterations, "NAD.chunkMemoryMB", "integer") != 0 {
            return 1;
        }
        let memory = if !memory.truthy() { 512 } else { memory.int() };
        let chunk_size = 5.max(memory.div_euclid(36));
        let nadcom = format!(
            "$nad_eed_3d -n {} -k {} INPUTFILE OUTPUTFILE",
            iterations.int(),
            py_str_float(kvalue.float())
        );
        if self.write_text_file_report_err(&com_file, &[nadcom]) != 0 {
            return 1;
        }

        // Make the chunk coms
        let padding = 8.max(iterations.int());
        let chunk_com =
            format!("chunksetup -m {chunk_size} -p {padding} -no {com_file} {rec_file} {nad_file}");
        if run_cmd(&chunk_com, None, None, None, &[]).is_err() {
            self.report_imod_error(Some(&format!("Error running chunksetup on {com_file}")));
            return 1;
        }

        self.run_one_process(
            &com_file,
            false,
            false,
            "Filtering with anisotropic diffusion",
            false,
        )
    }

    /// `runReduceFiltVol` (`IMOD/pysrc/batchruntomo:4660`).
    pub fn run_reduce_filt_vol(&mut self) -> i32 {
        if !self
            .lookup_directive(
                &format!("{RUNTIME_PREFIX}Postprocess"),
                "doReduceFilt",
                0,
                BOOL_VALUE,
            )
            .truthy()
        {
            return 0;
        }
        let binning = self.lookup_directive(
            &format!("{COM_PREFIX}reducefiltvol"),
            "reducefiltvol.ReductionFactor",
            0,
            FLOAT_VALUE,
        );
        if self.test_directive_value(&binning, "reducefiltvol.ReductionFactor", "float") != 0 {
            return 1;
        }
        let binning = if binning.is_none() {
            1.
        } else {
            binning.float()
        };
        let com_file = format!("reducefiltvol{}", self.com_ext);
        let rec_file = dataset_filename(".rec", None, None);
        if !Path::new(&rec_file).exists() {
            self.abort_set("You must post-process with Trimvol in order to run Reducefiltvol");
            return 1;
        }

        let mut comlines = vec![
            format!("RootNameOfDataFiles {}", self.data_name),
            format!("InputFile {rec_file}"),
            format!("BinningOfImages {}", py_str_float(binning)),
        ];
        comlines.push(format!(
            "OneParameterChange {COM_PREFIX}reducefiltvol.reducefiltvol.SetupChunksIfMemoryError=1"
        ));

        comlines.extend(self.later_com_directives(0));

        if self.make_and_run_one_com(
            &comlines,
            &com_file,
            "Reducing and/or filtering final tomogram",
            false,
            false,
        ) != 0
        {
            return 1;
        }

        let rfv_lines = self.read_text_file_report_err("reducefiltvol.log", None);
        if rfv_lines.is_empty() {
            return 1;
        }

        let mut it_ran = true;
        let mut com_name = format!("rfvfilter{}", self.com_ext);
        for line in &rfv_lines {
            if line.contains("[MTF1]") {
                it_ran = false;
            }
            if line.contains("processchunks") {
                let lsplit: Vec<&str> = line.split_whitespace().collect();
                com_name = lsplit[lsplit.len() - 1].to_owned();
            }
        }

        if it_ran {
            return 0;
        }

        self.run_one_process(
            &com_name,
            false,
            false,
            "Filtering volume in chunks to overcome memory limit",
            false,
        )
    }

    /// `runCleanup` (`IMOD/pysrc/batchruntomo:4707`).
    pub fn run_cleanup(&mut self) -> i32 {
        let keep = self
            .lookup_directive(
                &format!("{RUNTIME_PREFIX}Cleanup"),
                "doCleanup",
                0,
                BOOL_VALUE,
            )
            .truthy();
        if !keep {
            return 0;
        }

        let mut cmdstr = "tomocleanup".to_owned();
        if self
            .lookup_directive(
                &format!("{RUNTIME_PREFIX}Cleanup"),
                "keepAligned",
                0,
                BOOL_VALUE,
            )
            .truthy()
        {
            cmdstr += " -aligned";
        }
        if self
            .lookup_directive(
                &format!("{RUNTIME_PREFIX}Cleanup"),
                "keepUntrimmed",
                0,
                BOOL_VALUE,
            )
            .truthy()
        {
            cmdstr += " -untrimmed";
        }
        if self
            .lookup_directive(
                &format!("{RUNTIME_PREFIX}Cleanup"),
                "keepAxis",
                0,
                BOOL_VALUE,
            )
            .truthy()
        {
            cmdstr += " -axis";
        }
        if self
            .lookup_directive(
                &format!("{RUNTIME_PREFIX}Cleanup"),
                "keepSIRT",
                0,
                BOOL_VALUE,
            )
            .truthy()
        {
            cmdstr += " -sirt";
        }

        self.prn_log(&format!("Running {cmdstr} on data set"), "\n", false);
        let clean_lines = match run_cmd(&format!("{cmdstr} ."), None, None, None, &[]) {
            Ok(lines) => lines.unwrap_or_default(),
            Err(_) => {
                self.report_imod_error(Some("Error running tomocleanup on data set"));
                return 1;
            }
        };

        let tags = [MessageTag("WARNING:", 0, None)];
        self.print_tagged_messages(&clean_lines, &tags);
        0
    }
}

impl Brt {
    /// `runOneAxis` (`IMOD/pysrc/batchruntomo:4734`).
    pub fn run_one_axis(&mut self) -> i32 {
        self.xtilt_needed = 0.;
        self.fid_thickness = 0.;
        self.fid_inc_shift = 0.;
        self.recon_thickness = 0.;
        let mut clip_lines: Vec<String> = Vec::new();

        if self.get_axis_initial_parameters() != 0 {
            return 1;
        }

        // Xray removal
        if self
            .lookup_directive(
                &format!("{RUNTIME_PREFIX}Preprocessing"),
                "removeXrays",
                0,
                BOOL_VALUE,
            )
            .truthy()
            && self.need_step(1.)
        {
            let xray_mess = "Removing X-rays with Ccderaser";
            let mut sedcom: Vec<String> = Vec::new();

            // Copy the model file to the standard local name and switch to that unless the
            // file already exists.  But do it with imodtrans to make sure pixel size fits the
            // current stack
            let model = self.lookup_directive(
                &format!("{COM_PREFIX}eraser"),
                "ccderaser.ModelFile",
                0,
                STRING_VALUE,
            );
            if model.truthy() {
                let model = model.text().to_owned();
                let erase_mod = format!("{}{}.erase", self.set_name, self.axis_let);
                let base = Path::new(&model)
                    .file_name()
                    .map(|name| name.to_string_lossy().into_owned())
                    .unwrap_or_default();
                if base != erase_mod && !Path::new(&erase_mod).exists() {
                    if run_cmd(
                        &format!(
                            "imodtrans -I \"{}{}{}\" \"{model}\" \"{erase_mod}\"",
                            self.set_name, self.axis_let, self.stack_extension
                        ),
                        None,
                        None,
                        None,
                        &[],
                    )
                    .is_err()
                    {
                        self.report_imod_error(Some(&format!(
                            "Error transforming manual erasing model {model} to {erase_mod}"
                        )));
                        return 1;
                    }
                    sedcom = vec![sed_modify("ModelFile", &erase_mod, '/')];
                }
            }

            let comfile = format!("eraser{}", self.axis_com);
            if !sedcom.is_empty() {
                if self.modify_write_and_run_com(&comfile, &sedcom, None, xray_mess, false) != 0 {
                    return 1;
                }
            } else if self.run_one_process(&comfile, true, false, xray_mess, false) != 0 {
                return 1;
            }

            // Get clip stats output for fixed stack
            let raw_z = match get_mrc_size(&(self.data_name.clone() + &self.stack_extension)) {
                Ok((_x, _y, z)) => z as i64,
                Err(_) => {
                    self.report_imod_error(Some("Could not get size of raw stack"));
                    return 1;
                }
            };
            let length = 30.min(15.max(raw_z.div_euclid(4)));
            let mut mont_opt = String::new();
            if self.if_montage {
                mont_opt = format!("-P {}.pl -O -10,-10 ", self.data_name);
            }
            let command = format!(
                "$clip stats {mont_opt} -10,-10 -n 2.5 -l {length} {}_fixed{}",
                self.data_name, self.stack_extension
            );

            // Run clip and extract the summary lines
            clip_lines = self.run_clip_stats(&command, "_fixed", "fixed");
            if clip_lines.is_empty() {
                return 1;
            }
            self.prn_log(" ", "\n", false);
            for line in clip_lines.clone() {
                if line.contains("all") || line.contains("extreme") {
                    self.prn_log(&line, "\n", false);
                }
            }
            self.prn_log(" ", "\n", false);

            // Use fixed stack
            let from = format!("{}_fixed{}", self.data_name, self.stack_extension);
            let to = format!("{}{}", self.data_name, self.stack_extension);
            if self.use_file_as_replacement(&from, &to, true, false) != 0 {
                return 1;
            }

            // See if there is already an archive file to rename and find next number for that
            let comp_name = format!("{}_xray{}.gz", self.data_name, self.stack_extension);
            if Path::new(&comp_name).exists() {
                let mut comp_list1: Vec<String> = Vec::new();
                let mut comp_list2: Vec<String> = Vec::new();
                let directory = Path::new(&comp_name)
                    .parent()
                    .filter(|parent| !parent.as_os_str().is_empty())
                    .unwrap_or(Path::new("."))
                    .to_owned();
                let base = Path::new(&comp_name)
                    .file_name()
                    .map(|name| name.to_string_lossy().into_owned())
                    .unwrap_or_default();
                if let Ok(entries) = std::fs::read_dir(&directory) {
                    for entry in entries.flatten() {
                        let name = entry.file_name().to_string_lossy().into_owned();
                        let Some(rest) = name
                            .strip_prefix(&base)
                            .and_then(|rest| rest.strip_prefix('.'))
                        else {
                            continue;
                        };
                        let digits: Vec<char> = rest.chars().collect();
                        if digits.len() == 1 && digits[0].is_ascii_digit() {
                            comp_list1.push(name);
                        } else if digits.len() == 2
                            && digits[0].is_ascii_digit()
                            && digits[1].is_ascii_digit()
                        {
                            comp_list2.push(name);
                        }
                    }
                }
                comp_list1.sort();
                comp_list2.sort();
                comp_list1.extend(comp_list2);
                let mut next_num = 1i64;
                if !comp_list1.is_empty() {
                    let last = &comp_list1[comp_list1.len() - 1];
                    next_num = match last.rfind('.') {
                        Some(index) => last[index + 1..].parse::<i64>().unwrap_or(0) + 1,
                        None => 1,
                    };
                }
                if std::fs::rename(&comp_name, format!("{comp_name}.{next_num}")).is_err() {
                    self.abort_set(&format!(
                        "Error renaming {comp_name} before archiving again"
                    ));
                    return 1;
                }
            }

            // Run archiveorig
            if self
                .lookup_directive(
                    &format!("{RUNTIME_PREFIX}Preprocessing"),
                    "archiveOriginal",
                    0,
                    BOOL_VALUE,
                )
                .truthy()
            {
                let comfile = format!("archiveorig{}", self.axis_com);
                let line = format!("$archiveorig -d {}{}", self.data_name, self.stack_extension);
                if self.write_text_file_report_err(&comfile, &[line]) != 0 {
                    return 1;
                }
                let err = self.run_one_process(
                    &comfile,
                    true,
                    false,
                    "Archiving original stack as compressed difference file",
                    false,
                );
                cleanup_files(&[comfile]);
                if err != 0 {
                    return 1;
                }
            }
        }

        // Possibly a stopgap: look for low-count views at end of series and exclude them
        if !self.if_montage && self.need_step(1.) {
            let lines = clip_lines.clone();
            if self.analyze_sds_adjust_excludes(&lines) != 0 {
                return 1;
            }
        }

        // coarse alignment
        if self.need_step(2.) {
            let comfile = format!("xcorr{}", self.axis_com);
            if self.run_one_process(
                &comfile,
                true,
                false,
                "Finding coarse alignment by cross-correlation with Tiltxcorr",
                false,
            ) != 0
            {
                return 1;
            }
            if self.if_montage {
                let edfcom = self.edf_del_and_add(
                    &format!("xcorr.blendmont.{}.WasRun", self.axis_edf_let),
                    "true",
                    '/',
                );
                if self.modify_edf_lines(&edfcom) != 0 {
                    return 1;
                }
            }
        }

        // Get the tiltalign file regardless, since it will be operated on
        let align_com = format!("align{}", self.axis_com);
        self.align_lines = self.read_text_file_report_err(&align_com, None);
        if self.align_lines.is_empty() {
            return 1;
        }
        let lines = self.align_lines.clone();
        let rotarr = option_value(&lines, "RotationAngle", FLOAT_VALUE, false, 0, None, None);
        let rotarr = match &rotarr {
            Some(OptionValue::Floats(values)) if !values.is_empty() => values.clone(),
            _ => {
                self.abort_set(&format!(
                    "Cannot find RotationAngle in align{}",
                    self.axis_com
                ));
                return 1;
            }
        };
        self.axis_rotation = rotarr[0] as f64;
        while self.axis_rotation > 180. {
            self.axis_rotation -= 360.;
        }
        while self.axis_rotation < -180. {
            self.axis_rotation += 360.;
        }
        self.transpose_for_ali = self.axis_rotation.abs() > 45. && self.axis_rotation.abs() < 135.;

        // Get the number of surfaces from directives in case it changed, fall back to align.com
        let num_surfaces = self.lookup_directive(
            &format!("{COM_PREFIX}align"),
            "tiltalign.SurfacesToAnalyze",
            0,
            INT_VALUE,
        );
        if self.test_directive_value(&num_surfaces, "tiltalign.SurfacesToAnalyze", "integer") != 0 {
            return 1;
        }

        self.num_surfaces = num_surfaces.int();
        if num_surfaces.is_none() {
            let value = option_value(&lines, "SurfacesToAnalyze", INT_VALUE, false, 1, None, None);
            match &value {
                Some(OptionValue::Integers(values)) if !values.is_empty() => {
                    self.num_surfaces = values[0] as i64;
                }
                _ => {
                    // The source aborts and falls through with `numSurfaces`
                    // still None, which the `<= 0` test below then raises on.
                    self.abort_set(&format!(
                        "Failed to find SurfacesToAnalyze in align{}",
                        self.axis_com
                    ));
                    self.num_surfaces = 0;
                }
            }
        }
        if self.num_surfaces <= 0 {
            self.abort_set(&format!(
                "SurfacesToAnalyze is 0 or negative in directives or in align{}",
                self.axis_com
            ));
            return 1;
        }

        if self.fiducialless != 0 || self.patch_track {
            self.num_surfaces = 1;
        }

        // Get whether X axis tilt should be kept at 0
        self.no_xaxis_tilt = self
            .lookup_directive(
                &format!("{RUNTIME_PREFIX}Reconstruction"),
                "noXAxisTilt",
                0,
                BOOL_VALUE,
            )
            .int();

        // Get positioning and set up for centerOnGold if no positioning and 2 surfaces
        let pos_sample_type = self.lookup_directive(
            &format!("{RUNTIME_PREFIX}Positioning"),
            "sampleType",
            0,
            INT_VALUE,
        );
        if self.test_directive_value(&pos_sample_type, "Positioning.sampleType", "integer") != 0 {
            return 1;
        }

        self.pos_sample_type = pos_sample_type.int();
        self.center_on_gold = false;
        if self.pos_sample_type <= 0
            && self.num_surfaces > 1
            && self
                .lookup_directive(
                    &format!("{RUNTIME_PREFIX}Positioning"),
                    "centerOnGold",
                    0,
                    BOOL_VALUE,
                )
                .truthy()
        {
            self.center_on_gold = true;
        }

        // For fiducialless, make up the final xf
        if self.fiducialless != 0 {
            if self.fidless_file_operations() != 0 {
                return 1;
            }
            self.report_reached_step(5.);
        }
        // Otherwise do many things
        else {
            // Make prealigned stack
            let (com_root, _process) = self.com_and_process_for_aligned_stack("pre");
            if self.need_step(3.) {
                let comfile = format!("{com_root}{}", self.axis_com);
                let err = self.run_one_process(
                    &comfile,
                    true,
                    false,
                    "Making coarse aligned stack",
                    false,
                );
                if self.if_montage {
                    let flag = self.bool_string_for_edf(err != 0);
                    let edfcom = self.edf_del_and_add(
                        &format!("InvalidEdgeFunctions{}", self.axis_upper_let),
                        &flag,
                        '/',
                    );
                    if self.modify_edf_lines(&edfcom) != 0 {
                        return 1;
                    }
                }
                if err != 0 {
                    return 1;
                }
            }

            // Patch tracking
            if self.need_step(4.) && self.patch_track {
                if self.run_patch_tracking() != 0 {
                    return 1;
                }
            }
            // Otherwise various ways of getting a fiducial model
            else if self.make_seed_and_track() != 0 {
                return 1;
            }

            // Iterate patch tracking with angle offset if desired, afer running align
            let iterate_patch_track = self
                .lookup_directive(PATCH_TRACK_TEXT, "adjustTiltAngles", 0, BOOL_VALUE)
                .truthy()
                && self.need_step(4.)
                && self.patch_track;
            if iterate_patch_track {
                if self.run_tiltalign() != 0 {
                    return 1;
                }
                if self.ok_to_adjust_patch_track() {
                    let sedcom = sed_del_and_add(
                        "AngleOffset",
                        &py_str_float(self.total_del_tilt),
                        "SizeOfPatchesXandY",
                        '/',
                    );
                    let comfile = format!("xcorr_pt{}", self.axis_com);
                    let mess = format!(
                        "Running patch tracking again with angle offset of {:.1}",
                        self.total_del_tilt
                    );
                    if self.modify_write_and_run_com(&comfile, &sedcom, None, &mess, false) != 0 {
                        return 1;
                    }
                }
            }

            let do_vary_track = self
                .lookup_directive(PATCH_TRACK_TEXT, "varyPatchTrack", 0, BOOL_VALUE)
                .truthy()
                && self.need_step(4.)
                && self.patch_track;
            if do_vary_track && self.vary_patch_track() != 0 {
                return 1;
            }

            // Tilt alignment results needed if aligned stack or reconstruction being made
            self.report_reached_step(5.);
            if self.ending_step >= 6. && self.starting_step <= 14. && self.run_tiltalign() != 0 {
                return 1;
            }
        }

        // Positioning
        self.report_reached_step(6.);
        if self.need_step(7.) && self.position_tomogram() != 0 {
            return 1;
        }

        // Make aligned stack (tests need, sets correctCTF global, remakes if already corrected)
        self.report_reached_step(7.);
        if self.make_aligned_stack(-1) != 0 {
            return 1;
        }

        // Do CTF plotter step (tests need)
        if self.ctf_plot_aligned_stack() != 0 {
            return 1;
        }

        // Set up the tilt com file now that output size is known and montage frame data set
        if self.ending_step >= DETECT_3D_STEP_NUM
            && self.starting_step <= 14.
            && self.modify_tilt_com_file(0) != 0
        {
            return 1;
        }

        // Do 3D gold detection (tests need, sets eraseGold global)
        if self.detect_gold_in_3d() != 0 {
            return 1;
        }

        // Optionally CTF correct the stack
        self.report_reached_step(DETECT_3D_STEP_NUM);
        if self.need_step(CTF_CORR_STEP_NUM)
            && self.correct_ctf != 0
            && self.ctf_correct_aligned_stack() != 0
        {
            return 1;
        }

        // Optionally erase gold
        if self.need_step(ERASE_STEP_NUM)
            && self.erase_gold.truthy()
            && self.erase_gold_in_aligned_stack() != 0
        {
            return 1;
        }

        // Optionally filter
        if self.filter_aligned_stack() != 0 {
            return 1;
        }

        // Make the reconstruction
        self.report_reached_step(13.);
        if self.need_step(14.) && self.generate_tomogram() != 0 {
            return 1;
        }

        // Run trimvol on full data set or at axis level if directive for it
        self.report_reached_step(14.);
        if ((!self.dual_axis && self.need_step(TRIM_STEP_NUM))
            || (self.dual_axis && self.need_step(14.5)))
            && self.trim_volume() != 0
        {
            return 1;
        }

        // Run NAD on full data set only
        if !self.dual_axis && self.need_step(NAD_STEP_NUM) && self.run_nad() != 0 {
            return 1;
        }

        // Run Reducefiltvol on full data set only
        if !self.dual_axis && self.need_step(RED_FILT_STEP_NUM) && self.run_reduce_filt_vol() != 0 {
            return 1;
        }

        // Run Cleanup on full data set only
        if !self.dual_axis && self.need_step(CLEAN_STEP_NUM) && self.run_cleanup() != 0 {
            return 1;
        }

        0
    }

    /// `runCombine` (`IMOD/pysrc/batchruntomo:5010`).
    pub fn run_combine(&mut self) -> i32 {
        if self.need_step(15.) && self.setup_combine() != 0 {
            return 1;
        }

        if self.need_step(16.) && self.initial_combine_match() != 0 {
            return 1;
        }

        if self.align_and_combine_axes() != 0 {
            return 1;
        }

        if self.need_step(TRIM_STEP_NUM) && self.trim_volume() != 0 {
            return 1;
        }

        if self.need_step(NAD_STEP_NUM) && self.run_nad() != 0 {
            return 1;
        }

        if self.need_step(RED_FILT_STEP_NUM) && self.run_reduce_filt_vol() != 0 {
            return 1;
        }

        if self.need_step(CLEAN_STEP_NUM) {
            return self.run_cleanup();
        }

        0
    }
}

/// `#### MAIN PROGRAM ####` (`IMOD/pysrc/batchruntomo:5035-5935`).
pub fn batchruntomo(arguments: &[OsString]) -> i32 {
    let mut brt = Brt::new();

    // Setup runtime environment
    match std::env::var_os("IMOD_DIR") {
        Some(imod_dir) => {
            brt.imod_dir = imod_dir.to_string_lossy().into_owned();
            add_imod_bin_ignore_sighup();
        }
        None => {
            // `sys.stdout.write(prefix + " IMOD_DIR is not defined!\n")`
            print!("{PREFIX} IMOD_DIR is not defined!\n");
            let _ = std::io::stdout().flush();
            std::process::exit(1);
        }
    }

    // This handles the problem with # being lost as a comment in standard input
    pip_forbid_comments("CPUMachineList", "cpus", 0);

    let argv: Vec<String> = arguments
        .iter()
        .map(|argument| argument.to_string_lossy().into_owned())
        .collect();
    let options: Vec<String> = OPTIONS.iter().map(|entry| (*entry).to_owned()).collect();
    let (_opts, nonopts) = pip_read_or_parse_options(&argv, &options, PROGNAME, 1, 1, 0, None);
    unsafe {
        std::env::set_var("PIP_PRINT_ENTRIES", "0");
        std::env::set_var("SKIP_PHYSICAL_DPI", "1");
    }

    let do_pid = pip_get_boolean("PID", 0).unwrap_or(0);
    print_pid(do_pid != 0);
    brt.bypass_etomo = pip_get_boolean("BypassEtomo", 0).unwrap_or(0);

    // Intercept stack extension renaming call and process completely
    brt.from_extension = pip_get_string("StackExtension", "").unwrap_or_default();
    if !brt.from_extension.is_empty() {
        if !brt.possible_stack_exts.contains(&brt.from_extension) {
            let message = format!("{} is not an allowed stack extension", brt.from_extension);
            brt.exit_error(&message);
        }
        brt.renaming_only = true;
        let (name_style, type_extension) = match get_naming_style(None, false, false) {
            Ok(value) => value,
            Err(message) => brt.exit_error(&message),
        };
        brt.type_extension = type_extension.unwrap_or_default();
        if name_style < 0 {
            brt.exit_error("The naming style must be entered for a stack renaming run");
        }
        brt.name_style = name_style;
        if pip_number_of_entries("RootName").unwrap_or(0) != 1 {
            brt.exit_error("One root name must be entered for a stack renaming run");
        }
        brt.set_name = pip_get_string("RootName", "").unwrap_or_default();
        let dual_axis = pip_get_integer("AxisOfExtension", 0).unwrap_or(0);
        if !(0..=2).contains(&dual_axis) {
            brt.exit_error("Axis entry must be 0, 1, or 2");
        }
        brt.dual_axis = dual_axis != 0;
        let mut stack = brt.set_name.clone() + ".";
        if dual_axis != 0 {
            stack = brt.set_name.clone() + "a.";
        }

        brt.stack_extension = String::new();
        brt.extIfAlreadySet = String::new();
        if dual_axis == 2 {
            let name = brt.set_name.clone() + "b.";
            brt.check_rename_stack(&name);
        }
        brt.check_rename_stack(&stack);
        if dual_axis == 1 {
            brt.suppress_abort = true;
            let name = brt.set_name.clone() + "b.";
            brt.check_rename_stack(&name);
        }
        std::process::exit(0);
    }

    // Get current directory
    brt.starting_dir = std::env::current_dir()
        .map(|path| path.to_string_lossy().into_owned())
        .unwrap_or_default();

    // If doing parallel batch, get possible translation paths
    brt.parallel_root = pip_get_string("ParallelBatchRootName", "").unwrap_or_default();
    if !brt.parallel_root.is_empty() {
        brt.pro_chunk_check_file = format!("processchunks-{}.cmds", brt.my_pid);
        brt.num_translations = pip_number_of_entries("TranslatePathsFrom").unwrap_or(0);
        if brt.num_translations != pip_number_of_entries("TranslatePathsTo").unwrap_or(0) {
            brt.exit_error(
                "There are not the same number of entries for TranslatePathsFrom and TranslatePathsTo",
            );
        }
        for _ind in 0..brt.num_translations {
            brt.from_parallel_paths
                .push(pip_get_string("TranslatePathsFrom", "").unwrap_or_default());
            brt.to_parallel_paths
                .push(pip_get_string("TranslatePathsTo", "").unwrap_or_default());
        }
    }

    // Get the directive file names
    let num_by_arg = pip_number_of_entries("DirectiveFile").unwrap_or(0);
    brt.num_sets = nonopts + num_by_arg;
    if brt.num_sets == 0 {
        brt.exit_error("You must enter at least one directive file");
    }
    for ind in 0..brt.num_sets {
        let entered = if ind < num_by_arg {
            pip_get_string("DirectiveFile", "").unwrap_or_default()
        } else {
            pip_get_non_option_arg(ind).unwrap_or_default()
        };
        let dfile = brt.translate_parallel_path(&entered);
        if !Path::new(&dfile).exists() {
            let message = format!("Directive file {dfile} does not exist");
            brt.exit_error(&message);
        }
        brt.dir_files.push(dfile);
        brt.direc_translate_inds.push(brt.last_translate_ind);
    }

    // Get current directory and delivery directory
    brt.num_current = pip_number_of_entries("CurrentLocation").unwrap_or(0);
    for _ind in 0..brt.num_current {
        let entered = pip_get_string("CurrentLocation", "").unwrap_or_default();
        let translated = brt.translate_parallel_path(&entered);
        brt.current_dirs.push(translated);
        brt.cur_dir_translate_inds.push(brt.last_translate_ind);
    }

    brt.num_deliver = pip_number_of_entries("DeliverToDirectory").unwrap_or(0);
    for _ind in 0..brt.num_deliver {
        let entered = pip_get_string("DeliverToDirectory", "").unwrap_or_default();
        let translated = brt.translate_parallel_path(&entered);
        brt.deliver_dirs.push(translated);
        brt.deliver_translate_inds.push(brt.last_translate_ind);
    }

    if brt.num_deliver == 1 && !Path::new(&brt.deliver_dirs[0]).is_dir() {
        if Path::new(&brt.deliver_dirs[0]).exists() {
            let message = format!(
                "The -deliver entry {} must be a directory; there is a file with this name instead",
                brt.deliver_dirs[0]
            );
            brt.exit_error(&message);
        }
        let message = format!(
            "The location in which to create dataset directories ({}) must already exist; it does not",
            brt.deliver_dirs[0]
        );
        brt.exit_error(&message);
    }

    brt.make_sub_dir = pip_get_boolean("MakeSubDirectory", 0).unwrap_or(0);
    if brt.num_deliver != 0 && brt.make_sub_dir != 0 {
        brt.exit_error("You cannot enter both -deliver and -make options");
    }
    brt.do_delivery = brt.num_deliver != 0 || brt.make_sub_dir != 0;
    if brt.num_deliver != 0 && brt.num_deliver != 1 && brt.num_deliver != brt.num_sets {
        brt.exit_error(
            "If the -deliver option is used, it must be entered either once, or once per data set",
        );
    }
    if brt.do_delivery && brt.num_current != 1 && brt.num_current != brt.num_sets {
        brt.exit_error(
            "The -current option must be entered either once, or once per data set, if -deliver or -make is entered",
        );
    }

    // Get the root names if any
    brt.num_root_opts = pip_number_of_entries("RootName").unwrap_or(0);
    for _ind in 0..brt.num_root_opts {
        brt.root_by_option
            .push(pip_get_string("RootName", "").unwrap_or_default());
    }

    // Check validity of these entries and boost the directive file list
    if brt.num_root_opts != 0 && brt.num_sets > 1 && brt.num_sets != brt.num_root_opts {
        brt.exit_error(
            "If you enter root names, you must enter either one directive file or one per root name",
        );
    }
    if brt.num_root_opts != 0 {
        for _ind in brt.num_sets..brt.num_root_opts {
            let first = brt.dir_files[0].clone();
            brt.dir_files.push(first);
            let first = brt.direc_translate_inds[0];
            brt.direc_translate_inds.push(first);
        }
        brt.num_sets = brt.num_root_opts;
    }

    if !brt.do_delivery
        && (brt.num_root_opts != 0 || brt.num_current != 0)
        && brt.num_current != brt.num_sets
    {
        brt.exit_error("The -current option must be entered for each data set");
    }

    // Get other options

    // Get the CPU list from environment or option
    match std::env::var("MULTI_PROC_CPU_LIST") {
        Ok(value) => {
            brt.cpu_list = if value == "None" {
                String::new()
            } else {
                value
            };
        }
        Err(_) => {
            brt.cpu_list = pip_get_string("CPUMachineList", "").unwrap_or_default();
        }
    }

    // Get cluster queue options and alternative of cores when running on a node
    brt.cores_per_cluster_job = 0;
    let (queue_command, running_on_queue, max_queue_jobs) = brt.get_queue_options(
        "QueueCommand",
        "MULTI_PROC_QUEUE_COMMAND",
        "MaxJobsOnQueue",
        "MULTI_PROC_MAX_QUEUE_JOBS",
        "",
    );
    brt.queue_command = queue_command;
    brt.running_on_queue = running_on_queue;
    brt.max_queue_jobs = max_queue_jobs;

    brt.cores_per_cluster_job =
        brt.get_cluster_job_option("CoresPerClusterJob", "MULTI_PROC_JOB_CORES");
    if brt.cores_per_cluster_job != 0 {
        if !brt.queue_command.is_empty() {
            brt.exit_error("You cannot enter CoresPerClusterJob along with QueueCommand");
        }
        if !brt.cpu_list.is_empty() {
            brt.exit_error("You cannot enter CoresPerClusterJob along with a CPU machine list");
        }
        brt.cpu_list = brt.cores_per_cluster_job.to_string();
    }

    // For CPU's, validate the number if it is just a number, then parse the list and add up
    // the values after # if any, set flag for parallel processing = # of cores
    brt.local_name = hostname_node()
        .split('.')
        .next()
        .unwrap_or_default()
        .to_owned();
    brt.parallel_cpu = 0;
    brt.most_cpus = 0;
    brt.top_cpu_machine = "1".to_owned();
    brt.top_cpu_limit = 1;
    brt.local_cpu_limit = 1;
    brt.first_cpu_limit = 1;
    if brt.cpu_list.contains('#') {
        brt.warning(
            &["The # sign should not be used in the -cpus entry; use : instead".to_owned()],
            false,
        );
    }

    if !brt.cpu_list.is_empty() {
        brt.local_cpu_limit = 0;
        brt.first_cpu_limit = 0;
        match brt.cpu_list.trim().parse::<i64>() {
            Ok(parallel_cpu) => {
                brt.parallel_cpu = parallel_cpu;
                brt.local_cpu_limit = parallel_cpu;
                brt.first_cpu_limit = parallel_cpu;
                brt.top_cpu_limit = parallel_cpu;
                if parallel_cpu < 1 || parallel_cpu > 128 {
                    brt.exit_error(
                        "A number of cores entered with -cpus must be between 1 and 128",
                    );
                }
            }
            Err(_) => {
                let cpu_list = brt.cpu_list.clone();
                for machine in cpu_list.split(',') {
                    let replaced = machine.replace('#', ":");
                    let msplit: Vec<&str> = replaced.split(':').collect();
                    if msplit.len() > 2 {
                        brt.exit_error("A machine name cannot be followed by two : or # signs");
                    }
                    let num_cpu;
                    if msplit.len() < 2 {
                        num_cpu = 1;
                    } else {
                        match msplit[1].parse::<i64>() {
                            Ok(value) => {
                                num_cpu = value;
                                if num_cpu < 1 {
                                    let message = format!(
                                        "The value after : or # is less than 1 in {machine}"
                                    );
                                    brt.exit_error(&message);
                                }
                            }
                            Err(_) => {
                                let message = format!(
                                    "Failed to convert value after : or # to integer in {machine}"
                                );
                                brt.exit_error(&message);
                            }
                        }
                    }
                    brt.parallel_cpu += num_cpu;
                    if msplit[0] == "localhost"
                        || msplit[0].split('.').next().unwrap_or_default() == brt.local_name
                    {
                        brt.local_cpu_limit += num_cpu;
                    }
                    if num_cpu > brt.most_cpus {
                        brt.most_cpus = num_cpu;
                        brt.top_cpu_machine = msplit[0].to_owned();
                        brt.top_cpu_limit = num_cpu;
                    }
                    if brt.first_cpu_limit == 0 {
                        brt.first_cpu_limit = num_cpu;
                    }
                }
            }
        }
    }

    // After all that, replace the #'s in the list that will get used, and accept an option
    // to set or revise the limit on local cores, and if it is high enough, then revise the
    // machine with most CPUs
    brt.cpu_list = brt.cpu_list.replace('#', ":");

    // Get thread limit from variable if running in parallel, or from option
    if !brt.queue_command.is_empty() {
        brt.local_limit = 1;
        brt.first_cpu_limit = 1;
        brt.top_cpu_limit = 1;
    } else if brt.cores_per_cluster_job != 0 {
        brt.local_limit = brt.cores_per_cluster_job as i64;
        brt.first_cpu_limit = brt.cores_per_cluster_job as i64;
        brt.top_cpu_limit = brt.cores_per_cluster_job as i64;
    } else if let Ok(temp) = std::env::var("MULTI_PROC_THREAD_LIMIT") {
        match temp.parse::<i64>() {
            Ok(value) => brt.local_limit = value,
            Err(_) => {
                let message =
                    format!("Converting MULTI_PROC_THREAD_LIMIT value of {temp} to an integer");
                brt.exit_error(&message);
            }
        }
    } else {
        brt.local_limit = pip_get_integer("LimitLocalThreads", 0).unwrap_or(0) as i64;
    }
    if brt.local_limit < 0 || brt.local_limit > 128 {
        brt.exit_error("The limit on number of local threads must be between 1 and 128");
    }
    if brt.local_limit != 0 {
        brt.local_cpu_limit = brt.local_limit;
        if brt.local_limit > brt.top_cpu_limit {
            brt.top_cpu_machine = "localhost".to_owned();
            brt.top_cpu_limit = brt.local_limit;
        }
    }

    // For GPU's, set useGPU if entered, insist a number is 1, and set flag for parallel GPU
    // if there is an actual machine list to number of GPUs, adding up entries
    // with : separators.  Here keep parallelGPU = 1 if
    // it is running without splitting on a single remote machine
    brt.gpu_list = pip_get_string("GPUMachineList", "").unwrap_or_default();
    if let Ok(value) = std::env::var("MULTI_PROC_GPU_POOL") {
        if !value.is_empty() {
            brt.gpu_list = if value == "None" {
                String::new()
            } else {
                value
            };
        }
    }
    brt.use_gpu = i64::from(!brt.gpu_list.is_empty());
    brt.parallel_gpu = 0;

    // Determine if a GPU is being done with a separate queue
    let (gpu_queue_command, _nonsense, max_gpu_queue_jobs) = brt.get_queue_options(
        "GPUQueueCommand",
        "MULTI_PROC_GPU_QUEUE",
        "MaxGPUJobsOnQueue",
        "MULTI_PROC_MAX_GPU_JOBS",
        "GPU ",
    );
    brt.gpu_queue_command = gpu_queue_command;
    brt.max_gpu_queue_jobs = max_gpu_queue_jobs;
    if !brt.gpu_queue_command.is_empty() {
        if brt.use_gpu != 0 {
            brt.exit_error("You cannot enter both a GPU queue command and a GPU machine list");
        }
        brt.use_gpu = 1;
        if brt.max_gpu_queue_jobs > 1 {
            brt.parallel_gpu = brt.max_gpu_queue_jobs as i64;
        }
        brt.gpu_list = "1".to_owned();
    }

    // And find out if there are multiple GPUs for the current multicore node
    brt.gpus_per_cluster_job =
        brt.get_cluster_job_option("GPUsPerClusterJob", "MULTI_PROC_JOB_GPUS");
    if brt.gpus_per_cluster_job != 0 {
        if !brt.queue_command.is_empty() || !brt.gpu_queue_command.is_empty() {
            brt.exit_error("You cannot enter GPUs per job with a queue command");
        }
        if brt.cores_per_cluster_job == 0 {
            brt.exit_error("You cannot enter GPUs per job without cores per job");
        }

        brt.use_gpu = brt.gpus_per_cluster_job as i64;
        brt.gpu_list = "1".to_owned();
        if brt.use_gpu > 1 {
            brt.gpu_list = "localhost".to_owned();
            brt.parallel_gpu = brt.use_gpu;
            for ind in 0..brt.use_gpu {
                brt.gpu_list += &format!(":{}", ind + 1);
            }
        }
    }

    // Now process non-queue case for a gpu machine list
    brt.max_parallel_gpus = 0;
    if brt.use_gpu != 0 && !(!brt.gpu_queue_command.is_empty() || brt.gpus_per_cluster_job != 0) {
        if !brt.queue_command.is_empty() {
            brt.exit_error("You cannot enter a GPU machine list with a queue command");
        }
        match brt.gpu_list.trim().parse::<i64>() {
            Ok(value) => {
                brt.use_gpu = value;
                if brt.use_gpu != 1 {
                    brt.exit_error("The entry for a GPU list must be 1 to use just the local GPU");
                }
            }
            Err(_) => {
                if brt.cores_per_cluster_job != 0 {
                    brt.exit_error(
                        "With cores per node specified, an entry for the GPU machine list must be a single positive number",
                    );
                }

                let gpu_list = brt.gpu_list.clone();
                for machine in gpu_list.split(',') {
                    let msplit: Vec<&str> = machine.split(':').collect();
                    brt.parallel_gpu += 1.max(msplit.len() as i64 - 1);
                }
            }
        }
    }

    if !brt.parallel_root.is_empty() && brt.use_gpu != 0 {
        brt.max_parallel_gpus = pip_get_integer("MaxGPUsInParallelBatch", 4).unwrap_or(4);
        if brt.max_parallel_gpus < 1 {
            brt.exit_error("Maximum GPUs for parallel batch runs must be positive");
        }
        if !brt.gpu_queue_command.is_empty() {
            brt.parallel_gpu = brt.parallel_gpu.min(brt.max_parallel_gpus as i64);
            brt.max_gpu_queue_jobs = brt.max_gpu_queue_jobs.min(brt.max_parallel_gpus);
            brt.max_parallel_gpus = 0;
        } else if brt.queue_command.is_empty() && brt.cores_per_cluster_job == 0 {
            brt.hostname = hostname_node()
                .split('.')
                .next()
                .unwrap_or_default()
                .to_owned();
        }
    }

    brt.niceness = pip_get_integer("NiceValue", 15).unwrap_or(15);
    brt.remote_start_dir = pip_get_string("RemoteDirectory", "").unwrap_or_default();
    brt.do_one_axis = pip_get_integer("ProcessOneAxis", 0).unwrap_or(0);
    brt.starting_step = pip_get_float("StartingStep", 0.).unwrap_or(0.) as f64;
    brt.ending_step = pip_get_float("EndingStep", 10000000.).unwrap_or(10000000.) as f64;
    brt.first_start = pip_get_boolean("StartForFirstSetOnly", 0).unwrap_or(0);
    brt.exit_on_error = pip_get_boolean("ExitOnError", 0).unwrap_or(0);
    brt.etomo_debug = pip_get_integer("EtomoDebug", 0).unwrap_or(0);
    brt.use_first_cpu_for_single = pip_get_boolean("SingleOnFirstCPU", 0).unwrap_or(0);
    if brt.first_start == 0 && brt.starting_step > 100. {
        brt.warning(
            &["Entering a starting step above 100 has no effect without -first option".to_owned()],
            false,
        );
        while brt.starting_step > 100. {
            brt.starting_step -= 100.;
        }
    }

    // Get naming style and extension information, and set the output format if necessary
    let (name_style_dflt, type_extension) = default_naming_style();
    let (name_style, type_extension) = match get_naming_style(Some(&type_extension), false, false) {
        Ok(value) => value,
        Err(message) => brt.exit_error(&message),
    };
    brt.name_style = name_style;
    brt.type_extension = type_extension.unwrap_or_default();
    if brt.name_style < 0 {
        brt.name_style = name_style_dflt;
    }
    brt.com_ext = com_extension_from_option(0);

    brt.setup_bname = format!("laterbsetup{}", brt.com_ext);

    set_output_format_if_needed(&brt.type_extension, false);

    // TESTING ONLY: set test variables for running etomo
    if !brt.type_extension.is_empty() && brt.bypass_etomo == 0 {
        unsafe { std::env::set_var("TEST_NAMING_STYLE", brt.name_style.to_string()) };
    }
    if brt.com_ext == ".pcm" && brt.bypass_etomo == 0 {
        unsafe { std::env::set_var("TEST_USE_PCM_FOR_COM", "1") };
    }

    brt.skip_tiltalign = 0;
    if !(brt.starting_step <= 6.005 || (brt.starting_step > 99.995 && brt.starting_step <= 106.005))
    {
        brt.skip_tiltalign = pip_get_boolean("UseExistingAlignment", 1).unwrap_or(1);
    }

    brt.validation = pip_get_integer("ValidationType", 0).unwrap_or(0);
    brt.remote_data_dir = String::new();
    brt.email_address = pip_get_string("EmailAddress", "").unwrap_or_default();
    brt.smtp_server = pip_get_string("SMTPserver", "localhost").unwrap_or_default();

    // Set up check file
    let check_file = pip_get_string("CheckFile", "").unwrap_or_default();
    brt.check_file = if !check_file.is_empty() {
        imod_abs_path(&check_file)
    } else {
        imod_abs_path(
            &Path::new(&brt.starting_dir)
                .join(format!("{PROGNAME}.{}.input", std::process::id()))
                .to_string_lossy(),
        )
    };
    if brt.validation <= 0 {
        let translated = brt.translate_parallel_path(&brt.check_file.clone());
        prnstr(
            &format!("To quit all processing, place a Q in the file: {translated}"),
            "\n",
            false,
        );
        if !brt.parallel_root.is_empty() {
            prnstr(&format!("Running on host {}", hostname_node()), "\n", false);
        }
    }
    if Path::new(&brt.check_file).exists() {
        cleanup_files(&[brt.check_file.clone()]);
    }

    brt.file_type = "directive".to_owned();
    brt.defaults_file = Path::new(&brt.imod_dir)
        .join("com/batchDefaults.adoc")
        .to_string_lossy()
        .into_owned();

    if brt.validation >= 0 {
        brt.validate_file = Path::new(&brt.imod_dir)
            .join("com")
            .join("directives.csv")
            .to_string_lossy()
            .into_owned();
        if !Path::new(&brt.validate_file).exists() {
            let message = format!(
                "Cannot find file for validating directives, {}",
                brt.validate_file
            );
            brt.exit_error(&message);
        }

        brt.process_validation_file();
        if brt.validation > 1 {
            brt.file_type = "template".to_owned();
        }
    }

    // Start looping on the data sets
    for dfile_ind in 0..brt.num_sets as usize {
        brt.dfile_ind = dfile_ind;

        // These need to be initialized for abortSet to work right
        brt.dual_axis = false;
        brt.axis_let = String::new();
        brt.set_name = format!("# {}", dfile_ind + 1);

        brt.close_log_file_write_edf();
        let dfile = brt.dir_files[dfile_ind].clone();
        brt.axis_num = 0;
        brt.dset_dir_translate_ind = -1;

        // Check for an F in the check file before going on
        // No need to remove processchunks check file, it takes care of it itself
        brt.check_for_quit();
        if brt.finish_set_and_quit {
            prnstr(
                "Exiting after finishing dataset as requested   [brt6]",
                "\n",
                false,
            );
            std::process::exit(0);
        }

        let _ = std::env::set_current_dir(&brt.starting_dir);
        brt.all_directives = vec![PyDict::default(); 5];
        brt.abs_directive_file = imod_abs_path(&dfile);
        let (err, mut direct_lines, cpt_ext_directive, mut rewrite_batch, dir_for_rewrite) =
            brt.read_directive_or_template(&dfile, BAT_DICT_IND);
        if err != 0 {
            continue;
        }

        let defaults_file = brt.defaults_file.clone();
        let err = brt.read_directive_or_template(&defaults_file, 0);
        if err.0 != 0 {
            continue;
        }

        let dfile_prn = brt.reverse_translate_path(&dfile, brt.direc_translate_inds[dfile_ind]);

        prnstr(
            &format!(
                "Beginning to process {} file # {} : {dfile_prn}   [brt7]",
                brt.file_type,
                dfile_ind + 1
            ),
            "\n",
            false,
        );
        let start_time = py_time();

        // If validating a single template file, do not process it, just run the checks
        if brt.validation > 1 {
            if brt.check_all_directives("template") == 0 {
                prnstr("Directives all seem OK in that file", "\n", false);
            }
            continue;
        }

        // Get essential setup items from directives
        if brt.scan_setup_directives() != 0 {
            continue;
        }

        if brt.validation >= 0 {
            if brt.check_all_directives("directive") != 0 {
                continue;
            }
            if brt.validation > 0 {
                prnstr("Directives all seem OK in that file", "\n", false);
                continue;
            }
        }

        brt.check_defaults_in_batch_file();

        // If no fiducial size, add directive for 0 now
        if brt.fid_size_nm.is_none() {
            direct_lines.push(format!("{COPY_PREFIX}gold = 0."));
            rewrite_batch = true;
        }

        // Etomo will look for 'set '
        prnstr(
            &format!(
                "Starting data set {}   at {}   [brt1]",
                brt.set_name,
                hms_now()
            ),
            "\n",
            false,
        );
        prnstr("", "\n", true);

        brt.stack_extension = String::new();
        let mut do_bfirst = false;
        brt.from_extension = brt
            .lookup_directive(SETUP_PREFIX, "currentStackExt", 0, STRING_VALUE)
            .text()
            .to_owned();
        if brt.from_extension.is_empty() && brt.dual_axis && brt.do_one_axis != 1 {
            brt.from_extension = brt
                .lookup_directive(SETUP_PREFIX, "currentBStackExt", 0, STRING_VALUE)
                .text()
                .to_owned();
            if !brt.from_extension.is_empty() {
                do_bfirst = true;
            }
        }

        brt.extIfAlreadySet = String::new();
        if !brt.from_extension.is_empty() {
            brt.extIfAlreadySet = "st".to_owned();
            if !brt.type_extension.is_empty() {
                brt.extIfAlreadySet = brt.from_extension.clone();
            }
        }

        // Deliver stacks to dataset directory, from the source current dir or the one for
        // this data set
        if brt.do_delivery {
            let dir_ind = dfile_ind.min(brt.current_dirs.len() - 1);
            brt.deliver_from_dir = brt.current_dirs[dir_ind].clone();
            brt.del_from_translate_ind = brt.cur_dir_translate_inds[dir_ind];
            brt.dset_dir_translate_ind = brt.del_from_translate_ind;
            if !brt.deliver_dirs.is_empty() {
                brt.dset_dir_translate_ind =
                    brt.deliver_translate_inds[dfile_ind.min(brt.deliver_dirs.len() - 1)];
            }
            if do_bfirst && brt.deliver_stack("b") != 0 {
                continue;
            }
            if brt.dual_axis && brt.do_one_axis < 2 && brt.deliver_stack("a") != 0 {
                continue;
            }
            if !do_bfirst && brt.dual_axis && brt.do_one_axis != 1 && brt.deliver_stack("b") != 0 {
                continue;
            }
            if !brt.dual_axis && brt.deliver_stack("") != 0 {
                continue;
            }
        }

        // If there is a remote directory, need to shift it to the dataset dir
        // but translate the path of the starting dir since the dataset dir was also
        if !brt.remote_start_dir.is_empty() || brt.num_translations != 0 {
            let mut dir_to_translate = brt.dataset_dir.clone();
            if brt.remote_start_dir.is_empty() {
                brt.remote_start_dir = brt.translate_parallel_path(&brt.starting_dir.clone());
                dir_to_translate = brt.translate_parallel_path(&brt.dataset_dir.clone());
            }
            let starting = brt.translate_parallel_path(&brt.starting_dir.clone());
            let (remote_data_dir, err_mess) =
                transfer_remote_directory(&brt.remote_start_dir, &starting, &dir_to_translate);
            brt.remote_data_dir = remote_data_dir;
            let mut err_mess = err_mess;

            // If that didn't work, maybe translating the start dir will
            if brt.remote_data_dir.is_empty() {
                let starting = brt.translate_parallel_path(&brt.starting_dir.clone());
                let translated = brt.translate_parallel_path(&dir_to_translate);
                let (remote_data_dir, message) =
                    transfer_remote_directory(&brt.remote_start_dir, &starting, &translated);
                brt.remote_data_dir = remote_data_dir;
                err_mess = message;
            }

            if brt.remote_data_dir.is_empty() {
                brt.abort_set(&err_mess);
                continue;
            }
        }

        if std::env::set_current_dir(&brt.dataset_dir).is_err() {
            let message = format!("Error changing to directory {}", brt.dataset_dir);
            brt.abort_set(&message);
            continue;
        }

        // Open log file now
        match std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open("batchruntomo.log")
        {
            Ok(file) => {
                brt.log_file = Some(file);

                // Etomo needs the comma
                let message = format!(
                    "\nBatchruntomo started on data set {}, {}  [brt11]",
                    brt.set_name,
                    ctime_now()
                );
                brt.prn_log(&message, "\n", false);
                let location = brt.reverse_translate_path(
                    &imod_abs_path(&brt.dataset_dir.clone()),
                    brt.dset_dir_translate_ind,
                );
                brt.prn_log(&format!("In: {location}   [brt13]"), "\n", false);
            }
            Err(_) => {
                brt.warning(
                    &["Failed to open or write to log file in data set directory".to_owned()],
                    false,
                );
                brt.log_file = None;
            }
        }

        // Check existence of file(s)
        let mut stack = brt.set_name.clone() + ".";
        let mut num_axes = 1;
        if brt.dual_axis {
            num_axes = 2;
            stack = brt.set_name.clone() + "a.";
        }
        if do_bfirst {
            let name = brt.set_name.clone() + "b.";
            if brt.check_rename_stack(&name) != 0 {
                continue;
            }
        }
        if brt.check_rename_stack(&stack) != 0 {
            continue;
        }
        if !do_bfirst && brt.dual_axis && brt.do_one_axis != 1 {
            let name = brt.set_name.clone() + "b.";
            if brt.check_rename_stack(&name) != 0 {
                continue;
            }
        }
        stack += &brt.stack_extension[1.min(brt.stack_extension.len())..];

        // Add or correct a stackext entry in the batch file for etomo setup
        let stack_ext = brt.stack_extension[1.min(brt.stack_extension.len())..].to_owned();
        if cpt_ext_directive.0 < 0 || cpt_ext_directive.1 != stack_ext {
            rewrite_batch = true;
            let ext_line = format!("{CP_TOMO_EXT_TEXT} = {stack_ext}");
            if cpt_ext_directive.0 < 0 {
                direct_lines.push(ext_line);
            } else {
                direct_lines[cpt_ext_directive.0 as usize] = ext_line;
            }
        }

        // Rewrite the batch file with any corrections in the data directory
        if rewrite_batch {
            brt.abs_directive_file = Path::new(&dir_for_rewrite)
                .join(LOCAL_BATCH_COPY)
                .to_string_lossy()
                .into_owned();
            let name = brt.abs_directive_file.clone();
            if brt.write_text_file_report_err(&name, &direct_lines) != 0 {
                continue;
            }
        }

        // Get pixel size if scanHeader set
        if brt.scan_header && brt.pixel_size == 0. {
            match get_mrc_pixel(&stack) {
                Ok(px) => brt.pixel_size = px as f64 / 10.,
                Err(_) => {
                    brt.report_imod_error(Some(&format!("Error getting pixel size from {stack}")));
                    continue;
                }
            }
        }

        if brt.starting_step <= 0. && !(brt.dual_axis && brt.do_one_axis > 1) {
            if brt.bypass_etomo != 0 {
                // run copytomocoms directly for testing
                // The adoc will need rotation and brotation
                let mut copyargs = vec![
                    format!("pixel {}", py_str_float(brt.pixel_size)),
                    format!("name {}", brt.set_name),
                    format!("stackext {stack_ext}"),
                ];
                if brt.com_ext == ".pcm" {
                    copyargs.push("pcm 1".to_owned());
                }
                if !brt.type_extension.is_empty() {
                    copyargs.push(format!("style {}", brt.name_style));
                }
                if !brt
                    .lookup_directive(COPY_PREFIX, "userawtlt", 0, BOOL_VALUE)
                    .truthy()
                {
                    copyargs.push("extract 1".to_owned());
                }
                if !brt
                    .lookup_directive(COPY_PREFIX, "buserawtlt", 0, BOOL_VALUE)
                    .truthy()
                {
                    copyargs.push("bextract 1".to_owned());
                }
                if !brt
                    .lookup_directive(COPY_PREFIX, "rotation", 0, FLOAT_VALUE)
                    .truthy()
                {
                    let mut failed = false;
                    match get_mrc(&stack, false, true) {
                        Ok(MrcInfo::AngleLines(values)) => {
                            if let Some(value) = values[0] {
                                copyargs.push(format!("rotation {}", py_str_float(value as f64)));
                            }
                            if brt.dual_axis
                                && !brt
                                    .lookup_directive(COPY_PREFIX, "brotation", 0, FLOAT_VALUE)
                                    .truthy()
                            {
                                let bstack = format!("{}b{}", brt.set_name, brt.stack_extension);
                                match get_mrc(&bstack, false, true) {
                                    Ok(MrcInfo::AngleLines(values)) => {
                                        if let Some(value) = values[0] {
                                            copyargs.push(format!(
                                                "brotation {}",
                                                py_str_float(value as f64)
                                            ));
                                        }
                                    }
                                    Ok(_) => (),
                                    Err(_) => failed = true,
                                }
                            }
                        }
                        Ok(_) => (),
                        Err(_) => failed = true,
                    }
                    if failed {
                        brt.report_imod_error(Some("Error trying to get rotation angle"));
                    }
                }

                for key in brt.all_directives[BAT_DICT_IND].keys() {
                    if key.contains(COPY_PREFIX) {
                        let arg = key[COPY_PREFIX.len()..].to_owned();
                        let val = brt.all_directives[BAT_DICT_IND]
                            .get(&key)
                            .unwrap()
                            .0
                            .clone();
                        if !val.is_empty() {
                            copyargs.push(format!("{arg} {val}"));
                        }
                    }
                }
                for text in [SCOPE_TMPL_TEXT, SYS_TMPL_TEXT, USER_TMPL_TEXT] {
                    if brt.all_directives[BAT_DICT_IND].contains(text) {
                        let value = brt.all_directives[BAT_DICT_IND]
                            .get(text)
                            .unwrap()
                            .0
                            .clone();
                        copyargs.push(format!("change {value}"));
                    }
                }
                copyargs.push(format!("change {}", brt.abs_directive_file));
                if run_cmd(
                    "copytomocoms -StandardInput",
                    Some(&copyargs),
                    Some("stdout"),
                    None,
                    &[],
                )
                .is_err()
                {
                    brt.report_imod_error(Some("Error running copytomocoms"));
                    continue;
                }
            } else {
                // Run etomo for setup and try to report errors
                let mut failed = false;
                let mut errlog = String::new();
                let mut comline = format!(
                    "etomo --fromBRT --directive \"{}\" --namingstyle {}",
                    brt.abs_directive_file, brt.name_style
                );
                if brt.etomo_debug != 0 {
                    comline += &format!(" --debug {}", brt.etomo_debug);
                }
                if brt.parallel_cpu > 1 {
                    comline += &format!(" --cpus \"{}\"", brt.cpu_list);
                }
                if !brt.gpu_list.is_empty() {
                    comline += &format!(" --gpus \"{}\"", brt.gpu_list);
                }

                match run_cmd(&comline, None, None, None, &[]) {
                    Ok(lines) => {
                        for l in lines.unwrap_or_default() {
                            let text = l.trim_end().to_owned();
                            brt.prn_log(&text, "\n", false);
                            if let Some(ind) = l.find("with log in") {
                                if ind > 0 {
                                    errlog = l[ind + 11..].trim().to_owned();
                                }
                            }
                        }
                    }
                    Err(_) => {
                        failed = true;
                        for l in get_err_strings() {
                            if let Some(ind) = l.find("check:") {
                                if ind > 0 {
                                    errlog = l[ind + 7..].trim().to_owned();
                                }
                            }
                        }
                    }
                }

                if !errlog.is_empty() {
                    if errlog.contains('/') || errlog.contains('\\') {
                        let base = Path::new(&errlog)
                            .file_name()
                            .map(|name| name.to_string_lossy().into_owned())
                            .unwrap_or_default();
                        if std::fs::copy(&errlog, &base).is_err() {
                            brt.warning(
                                &[format!("Failed to copy {errlog} to dataset directory")],
                                true,
                            );
                        }
                    }

                    match read_text_file(&errlog, None, true, None) {
                        Err(message) => {
                            brt.warning(&[format!("Error {message}")], true);
                        }
                        Ok(loglines) => {
                            let tags = [MessageTag("INFO:", 1, None), MessageTag("LOG:", 1, None)];
                            brt.print_tagged_messages(&loglines, &tags);
                            let mut err = 0;
                            for line in &loglines {
                                if line.contains("Pixel spacing was set in FEI file")
                                    || line.contains("Pixel spacing was set in file")
                                {
                                    err = brt.fix_all_supplied_model_headers();
                                    break;
                                }
                            }

                            if err != 0 {
                                brt.prn_log("Cannot proceed with this data set", "\n", false);
                                continue;
                            }
                        }
                    }
                } else if failed {
                    brt.prn_log(
                        "Running etomo failed, no etomo error log available",
                        "\n",
                        false,
                    );
                } else {
                    brt.prn_log("Cannot access an error log from running etomo", "\n", false);
                }
                if failed {
                    brt.report_imod_error(Some(
                        "etomo setup failed, cannot proceed with this data set",
                    ));
                    continue;
                }

                // Report on excluded views
                if !brt.excluded_views_a.is_empty()
                    && brt
                        .lookup_directive(
                            &format!("{RUNTIME_PREFIX}Preprocessing"),
                            "removeExcludedViews",
                            0,
                            BOOL_VALUE,
                        )
                        .truthy()
                {
                    let mut mess = format!("Removed excluded views {}", brt.excluded_views_a);
                    if brt.dual_axis {
                        mess += " for axis A";
                    }
                    brt.prn_log(&format!("{mess}  [brt10]"), "\n", false);
                }

                if !brt.excluded_views_b.is_empty() && brt.dual_axis {
                    brt.axis_num = 1;
                    if brt
                        .lookup_directive(
                            &format!("{RUNTIME_PREFIX}Preprocessing"),
                            "removeExcludedViews",
                            0,
                            BOOL_VALUE,
                        )
                        .truthy()
                    {
                        let views = brt.excluded_views_b.clone();
                        brt.prn_log(
                            &format!("Removed excluded views {views} for axis B  [brt10]"),
                            "\n",
                            false,
                        );
                    }
                    brt.axis_num = 0;
                }
            }
        }

        brt.report_reached_step(0.);
        if let Some(fid_size_nm) = brt.fid_size_nm.filter(|value| *value != 0.) {
            brt.fid_size_pix = fid_size_nm / brt.pixel_size;
        }
        let mut axis_failed = false;

        if !brt.original_stack_ext.is_empty() {
            let value = brt.original_stack_ext.clone();
            let edfcom = brt.edf_del_and_add("OrigImageStackExt", &value, '/');
            if brt.modify_edf_lines(&edfcom) != 0 {
                continue;
            }
        }

        // Loop on the axes.  axisInd is 0 or 1, axisNum is 0 for single, 1/2 for a/b
        for axis_ind in 0..num_axes {
            brt.axis_ind = axis_ind;
            if brt.ending_step <= 0.
                || (brt.dual_axis
                    && ((axis_ind != 0 && brt.do_one_axis == 1)
                        || (axis_ind == 0 && brt.do_one_axis > 1)))
            {
                continue;
            }
            brt.axis_let = String::new();
            brt.axis_edf_let = "a".to_owned();
            brt.axis_upper_let = "A".to_owned();
            if brt.dual_axis {
                brt.axis_num = axis_ind + 1;
                brt.axis_let = "a".to_owned();
                let mut add_to_mess = "A".to_owned();
                if axis_ind != 0 {
                    brt.axis_let = "b".to_owned();
                    brt.axis_edf_let = "b".to_owned();
                    brt.axis_upper_let = "B".to_owned();
                    add_to_mess = format!("B{LOG_SUFFIX_TAG}");
                }
                brt.prn_log(&format!("Starting axis {add_to_mess} [brt14]"), "\n", false);
                if axis_ind != 0 {
                    brt.prn_log("", "\n", false);
                }
            }

            if axis_ind != 0 && brt.do_one_axis > 1 && Path::new(&brt.setup_bname).exists() {
                let name = brt.setup_bname.clone();
                if brt.run_one_process(
                    &name,
                    true,
                    false,
                    "Doing setup tasks now that second axis is present",
                    false,
                ) != 0
                {
                    brt.starting_step = 0.;
                    axis_failed = true;
                    continue;
                }
            }

            brt.data_name = brt.set_name.clone() + &brt.axis_let;
            brt.axis_com = brt.axis_let.clone() + &brt.com_ext;
            set_root_and_extension(&brt.data_name.clone(), &brt.type_extension.clone());
            while brt.starting_step > 100. {
                brt.starting_step -= 100.;
            }
            if brt.run_one_axis() != 0 {
                // When an axis fails, mark failure to prevent going on to combine
                axis_failed = true;
                if brt.first_start != 0 {
                    brt.starting_step = 0.;
                    brt.skip_tiltalign = 0;
                }
                continue;
            }
            if brt.first_start != 0 {
                brt.starting_step = 0.;
                brt.skip_tiltalign = 0;
            }
            let mut message = if brt.dual_axis {
                format!(
                    "Completed axis {} of dataset {}",
                    brt.axis_let.to_uppercase(),
                    brt.set_name
                )
            } else {
                format!("Completed dataset {}", brt.set_name)
            };
            if brt.ending_step < FINAL_STEP_NUM {
                let end_print = brt.ending_step.round() as i64;
                if (end_print as f64 - brt.ending_step).abs() > 0.005 {
                    message += &format!(" through step {}", py_str_float(brt.ending_step));
                } else {
                    message += &format!(" through step {end_print}");
                }
            }
            if !brt.dual_axis {
                let (minutes, seconds, frac) = elapsed_time_components(start_time);
                message += &format!("  in {minutes:02}:{seconds:02}.{frac}   [brt4]");
            }
            brt.prn_log(&message, "\n", false);
            brt.summary_message += &(message + "\n");
        }

        // Try to combine dual axis dataset
        if brt.dual_axis && brt.do_one_axis != 1 {
            brt.axis_num = 0;
            brt.axis_let = "a".to_owned();
            brt.axis_ind = 2;
            brt.axis_com = brt.com_ext.clone();
            brt.data_name = brt.set_name.clone();
            set_root_and_extension(&brt.data_name.clone(), &brt.type_extension.clone());
            if axis_failed {
                brt.abort_set("One of the axes failed");
            } else if brt.run_combine() == 0 {
                let (minutes, seconds, frac) = elapsed_time_components(start_time);
                let message = format!(
                    "Completed dataset {}  in {minutes:02}:{seconds:02}.{frac}   [brt4]",
                    brt.set_name
                );
                brt.prn_log(&message, "\n", false);
            }
        }

        brt.close_log_file_write_edf();
        brt.prn_log("", "\n", true);
    }

    let summary = brt.summary_message.clone();
    brt.send_email("Batchruntomo finished all data sets", &summary);
    let mut mess = "Batch run finished; ".to_owned();
    if brt.final_ret_val != 0 {
        mess += &format!("failures occurred for {} datasets", brt.final_ret_val);
    } else {
        mess += "no failures occurred";
    }
    prnstr(&mess, "\n", false);
    let _ = std::io::stdout().flush();
    std::process::exit(0);
}

/// `platform.node()` / `socket.gethostname()`.
fn hostname_node() -> String {
    let mut buffer = [0u8; 256];
    if unsafe { libc::gethostname(buffer.as_mut_ptr().cast(), buffer.len()) } != 0 {
        return String::new();
    }
    let end = buffer
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(buffer.len());
    OsStr::from_bytes(&buffer[..end])
        .to_string_lossy()
        .into_owned()
}
