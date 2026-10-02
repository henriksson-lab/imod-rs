//! Translation of `IMOD/raptor/main.cpp`, the `RAPTOR` program: automatic
//! fiducial marker tracking (and, without `-track`, alignment and
//! reconstruction through IMOD's programs).
//!
//! The file-scope globals `diameter`, `zerotilt`, `maxMarkersPrevFrame` and
//! `maxMarkersNextFrame` are thread-local here, so a run in process starts
//! with its own.  `cout`/`cerr`/`clog` go through `cxx_stream`, which the
//! program redirects to its log file as the source does with `rdbuf`.
//!
//! The source runs IMOD programs with `system("cmd > <basename>_IMOD.log")`.
//! Those programs are translated in this crate, so [`system`] runs them as
//! our own: in process when the command table allows it, else as a child of
//! our `imod` binary.  `mkdir`, `cp`, `mv` and `rm -rf` are done with the
//! file system calls they stand for.
//!
//! Upstream defects fixed here (`BUGS.md`, RAPTOR): `exit(error)` after a
//! failed `system()` passed the *wait status* (256 times the exit code) to
//! `exit`, which keeps only its low 8 bits, so RAPTOR reported the failure in
//! its log and then exited with status 0; it exits with the command's own
//! status here.  The `_preali` suffix was removed with
//! `find_last_of("_preali")`, which finds the last of any of those seven
//! characters, so a base name of seven or more characters ending in `_`,
//! `p`, `r`, `e`, `a`, `l` or `i` lost its last seven characters; only the
//! suffix `_preali` is removed here.  An input file name with no `.` made
//! `substr(npos)` throw (the process aborted); its extension is empty here.

use crate::imod::cxx_stream::{IStream, cout, ostream_double, redirect_standard_streams};
use crate::imod::libcfshr::b3dutil::{RAND_MAX, exit, program_args, rand, srand};
use crate::imod::libcfshr::parse_params::{
    exit_error, pip_get_boolean, pip_get_in_out_file, pip_get_integer, pip_get_integer_array,
    pip_get_string, pip_print_help, pip_read_or_parse_options,
};
use crate::imod::raptor::correspondence::correspondence::correspondence_regions;
use crate::imod::raptor::fill_contours::fill_contours::{fill_contours, join_similar_contours};
use crate::imod::raptor::main_classes::constants::{
    MAX_JUMPS, MAX_TARGETS_NEXT_FRAME, PERCENTILE, get_date,
};
use crate::imod::raptor::main_classes::frame::Frame;
use crate::imod::raptor::main_classes::io_mrc_vol::IoMrc;
use crate::imod::raptor::main_classes::paircorrespondence::PairCorrespondence;
use crate::imod::raptor::main_classes::point2d::Point2D;
use crate::imod::raptor::optimization::contour::Contour;
use crate::imod::raptor::optimization::sfm_data::SfmData;
use crate::imod::raptor::optimization::sfm_estimation_with_ba::{
    decide_alpha_option, decide_tilt_align_options, resid_analysis, sfm_estimation_with_ba,
    write_imod_fid_model_reproj, write_imod_fid_model_sfm,
};
use crate::imod::raptor::template::template::{
    compute_ncc, create_synthetic_template, estimate_number_of_markers, find_all_peaks,
};
use crate::imod::raptor::trajectory::trajectory::{
    find_trajectories, free_trajectory_vector, get_marker_type_from_trajectory,
    write_imod_fid_model, write_imod_tilt_script, write_imod_tiltalign,
};
use std::cell::{Cell, RefCell};
use std::io::Write as _;

thread_local! {
    /// `int* diameter` (`main.cpp:40`).
    pub static DIAMETER: RefCell<Vec<i32>> = const { RefCell::new(Vec::new()) };
    /// `int zerotilt` (`main.cpp:41`).
    pub static ZEROTILT: Cell<i32> = const { Cell::new(0) };
    /// `unsigned int maxMarkersPrevFrame` (`main.cpp:42`).
    pub static MAX_MARKERS_PREV_FRAME: Cell<u32> = const { Cell::new(0) };
    /// `unsigned int maxMarkersNextFrame` (`main.cpp:42`).
    pub static MAX_MARKERS_NEXT_FRAME: Cell<u32> = const { Cell::new(0) };
}

/// C `system(command)` for the command lines RAPTOR builds: `<program>
/// <arguments...> > <log>`.  The program is one of this crate's commands; it
/// runs in process when the command table allows (its standard output
/// captured into `log`), else as a child of our `imod` binary with standard
/// output sent to `log`, and, when this process is not the `imod` launcher,
/// through `/bin/sh` as the source's `system` does.  Returns the program's
/// exit status (the source's `system` returns the wait status; see the module
/// documentation for why the difference matters).
fn system(command: &str, log: &str) -> i32 {
    let words: Vec<&str> = command.split_whitespace().collect();
    let name = words.first().copied().unwrap_or_default();
    let exe = std::env::current_exe().ok();
    let mut argv: Vec<std::ffi::OsString> = Vec::with_capacity(words.len());
    argv.push(match &exe {
        Some(path) => path.with_file_name(name).into_os_string(),
        None => name.into(),
    });
    argv.extend(words.iter().skip(1).map(std::ffi::OsString::from));
    if let Some(entry) = crate::imod::commands::find(name)
        && entry.in_process
    {
        let (status, output) = crate::imod::commands::run_in_process(entry, argv, None, true)
            .unwrap_or((1, Vec::new()));
        let _ = std::fs::write(log, output);
        return status;
    }
    let out = match std::fs::File::create(log) {
        Ok(file) => file,
        Err(_) => return 1,
    };
    let child = match &exe {
        Some(path)
            if crate::imod::commands::find(name).is_some()
                && path.file_name().is_some_and(|base| base == "imod") =>
        {
            std::process::Command::new(path)
                .arg(name)
                .args(&words[1..])
                .stdout(out)
                .status()
        }
        _ => std::process::Command::new("/bin/sh")
            .arg("-c")
            .arg(command)
            .stdout(out)
            .status(),
    };
    match child {
        Ok(status) => status.code().unwrap_or(1),
        Err(_) => 127,
    }
}

/// Writes `text` to `path` as an `ofstream` the source opens without checking
/// does: nothing happens when the file cannot be created.
fn write_unchecked(path: &str, text: &[u8]) {
    let _ = std::fs::write(path, text);
}

/// `main(int argc, char* argv[])` (`main.cpp:45`).  Returns the exit status.
pub fn main() -> i32 {
    let argv_strings = program_args();
    let argv: Vec<Vec<u8>> = argv_strings.iter().map(|a| a.as_bytes().to_vec()).collect();
    let argc = argv.len() as i32;
    let mut error = 0i32;
    let mut cmd: String;
    let mut dmax = 0i32;
    let mut max_pw_table = 0i32;
    let mut min_pw_table = 0i32;
    let mut num_diff_marker_size = 0i32; // I need to initialize this to avoid IMOD crashing

    let mut num_opt_args = 0i32;
    let mut num_non_opt_args = 0i32;
    let num_options = 18i32;
    let options: [&[u8]; 18] = [
        b"execPath:RaptorExecPath:CH:",
        b"path:InputPath:CH:",
        b"input:InputFile:FN:",
        b"output:OutputPath:CH:",
        b"diameter:Diameter:IA:",
        b"markers:MarkersPerImage:I:",
        b"angles:AnglesInHeader:B:",
        b"bin:Binning:I:",
        b"rec:Reconstruction:I:",
        b"thickness:Thickness:I:",
        b"maxDist:MaxDistanceCandidate:I:",
        b"minNeigh:MinNeighborsMRF:I:",
        b"rollOff:RollOffMRF:I:",
        b"verb:Verbose:I:",
        b"white:WhiteMarkers:B:",
        b"tracking:TrackingOnly:B:",
        b"xray:xRay:B:",
        b"seed:Seed:I:",
    ];
    pip_read_or_parse_options(
        argc,
        &argv,
        &options,
        num_options,
        b"raptor",
        1,
        0,
        0,
        &mut num_opt_args,
        &mut num_non_opt_args,
        None,
    );
    if pip_get_boolean(b"usage", &mut error) == 0 {
        pip_print_help(b"RAPTOR", 0, 0, 0);
        exit(0);
    }
    let mut input_: Vec<u8> = Vec::new();
    let mut path_: Vec<u8> = Vec::new();
    let mut output_: Vec<u8> = Vec::new();
    let mut bin_path_: Vec<u8> = Vec::new();
    let mut targets = 0i32;
    let mut white_ = 0i32;
    let mut tracking_only_ = 0i32;
    // amount of output desired: 0->minimial output (just rec, and log
    // file), 1->normal mode of operation: some output and align stack is not
    // deleted, 2->debug mode: lots of information recorded to debug code
    let mut verbose = 0i32;
    let mut angles_header_ = 0i32;
    let mut binning = 0i32;
    let mut thickness = 0i32;
    let mut rec = 0i32;
    let mut x_ray_ = 0i32;

    // initialize random seed
    let mut seed = 0i32;
    if pip_get_integer(b"Seed", &mut seed) != 0 {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        srand(now as u32);
    } else {
        srand(seed as u32);
    }

    if pip_get_string(b"RaptorExecPath", &mut bin_path_) != 0 {
        exit_error(
            b"No binary path specified to execute RAPTOR. Please specify where RAPTOR binary is located\n",
        );
    }
    if pip_get_string(b"InputPath", &mut path_) != 0 {
        exit_error(b"No input path specified exist\n");
    }
    if pip_get_string(b"OutputPath", &mut output_) != 0 {
        exit_error(b"No output path specified exist\n");
    }
    if pip_get_in_out_file(b"InputFile", 0, &mut input_) != 0 {
        exit_error(b"The file does not exist\n");
    }

    let mut diameter_ = vec![0i32; 100]; // to make sure we never run out of space

    if pip_get_integer_array(b"Diameter", &mut diameter_, &mut num_diff_marker_size, 100) != 0 {
        exit_error(b"No diameter of fiducial markers specified\n");
    }

    let mut diameter = vec![0i32; num_diff_marker_size.max(0) as usize];
    let mut templ_size = vec![0i32; num_diff_marker_size.max(0) as usize];
    for kk in 0..num_diff_marker_size as usize {
        diameter[kk] = diameter_[kk];
        templ_size[kk] = 2 * diameter[kk] + 1;
    }
    drop(diameter_);
    DIAMETER.with(|d| *d.borrow_mut() = diameter.clone());

    if pip_get_integer(b"MarkersPerImage", &mut targets) != 0 {
        targets = -1;
    }
    if pip_get_boolean(b"AnglesInHeader", &mut angles_header_) != 0 {
        angles_header_ = 0;
    }
    if pip_get_integer(b"Binning", &mut binning) != 0 {
        binning = 1;
    }
    if pip_get_integer(b"MaxDistanceCandidate", &mut max_pw_table) != 0 {
        max_pw_table = -10;
    }
    if pip_get_integer(b"MinNeighborsMRF", &mut min_pw_table) != 0 {
        min_pw_table = -10;
    }
    if pip_get_integer(b"RollOffMRF", &mut dmax) != 0 {
        dmax = -10;
    }
    if pip_get_integer(b"Thickness", &mut thickness) != 0 {
        thickness = 0;
    }
    if pip_get_integer(b"Reconstruction", &mut rec) != 0 {
        rec = -10000;
    }
    if pip_get_integer(b"Verbose", &mut verbose) != 0 {
        verbose = 1;
    }
    if pip_get_boolean(b"WhiteMarkers", &mut white_) != 0 {
        white_ = 0; // indicates if markers are white (instead of black)
    }
    if pip_get_boolean(b"xRay", &mut x_ray_) != 0 {
        x_ray_ = 0;
    }

    if pip_get_boolean(b"TrackingOnly", &mut tracking_only_) != 0 {
        // indicates if we don't want aligned file. Only fiducial markers model
        tracking_only_ = 0;
    }

    let tracking_only = tracking_only_ == 1;
    let white = white_ == 1;
    let x_ray = x_ray_ == 1;

    if binning < 1 {
        cout(&format!(
            "ERROR: binning factor is {binning}.It should be above 1\n"
        ));
        exit(-1);
    }

    let debug_mode = verbose >= 2;
    let angles_header = angles_header_ != 0;

    let mut input_dir = String::from_utf8_lossy(&path_).into_owned();
    let mut output_dir = String::from_utf8_lossy(&output_).into_owned();
    let input_filename = String::from_utf8_lossy(&input_).into_owned();
    let mut bin_path = String::from_utf8_lossy(&bin_path_).into_owned();
    // generate basename
    let mut basename = match input_filename.rfind('.') {
        Some(pos_dot) => input_filename[..pos_dot].to_string(),
        None => input_filename.clone(),
    };
    if let Some(stripped) = basename.strip_suffix("_preali") {
        basename = stripped.to_string();
    }

    // make sure all paths have / at the end (`find_last_of("/") !=
    // size()-1`, which for an empty string compares npos with npos)
    for path in [&mut input_dir, &mut output_dir, &mut bin_path] {
        let last = path.rfind('/').unwrap_or(usize::MAX);
        if last != path.len().wrapping_sub(1) {
            path.push('/');
        }
    }

    // create necessary folders
    for dir in [
        output_dir.clone(),
        format!("{output_dir}align"),
        format!("{output_dir}IMOD"),
        format!("{output_dir}temp"),
    ] {
        if std::fs::metadata(&dir).is_err() {
            // check if folder exists
            let _ = std::fs::create_dir(&dir);
        }
    }
    if debug_mode {
        let dir = format!("{output_dir}debug");
        if std::fs::metadata(&dir).is_err() {
            let _ = std::fs::create_dir(&dir);
        }
    }

    // setup things so all the outputs are written to a log file
    let log_name = format!("{output_dir}align/{basename}_RAPTOR.log");
    let log_file = match std::fs::File::create(&log_name) {
        Ok(file) => file,
        Err(_) => {
            cout(&format!(
                "ERROR:impossible to open file {output_dir}align/{basename}_RAPTOR.log as log file\n"
            ));
            exit(-1);
        }
    };

    let sbuf = redirect_standard_streams(Some(log_file));
    let error_output_log = format!("{output_dir}align/{basename}_IMOD.log");

    let ini_time_stamp = get_date();
    cout(&format!(
        "Starting RAPTOR on dataset {input_dir}{input_filename} at {ini_time_stamp}\n"
    ));
    cout("RAPTOR called with the following command:\n");
    for a in &argv_strings {
        cout(&format!("{a} "));
    }
    cout("\n");
    cout(&format!("InputPath={input_dir}\n"));
    cout(&format!("Filename={input_filename}\n"));
    cout(&format!("OutputPath={output_dir}\n"));
    cout(&format!("NumberOfMarkers={targets}\n"));
    cout("MarkerDiameter (in pixels)=");
    for d in &diameter {
        cout(&format!("{d},"));
    }
    cout("\n");
    cout(&format!("Binning factor={binning}\n"));
    cout(&format!("Angles in header={}\n", angles_header as i32));
    cout(&format!(
        "Perform reconstruction after alignment={rec} (negative number means no reconstruction)\n"
    ));
    cout("Advanced options:\n");
    cout(&format!(
        "Maximum distance to consider a candidate={max_pw_table} (negative number means set to default)\n"
    ));
    cout(&format!(
        "Minimum amount of neighbors per candidate={min_pw_table} (negative number means set to default)\n"
    ));
    cout(&format!(
        "Roll-off in exponential when considering neighbors={dmax} (negative number means set to default)\n"
    ));

    let mut frames: Vec<Frame> = Vec::new();
    let mut points: Vec<Point2D> = Vec::new();
    let mut vol = IoMrc::new();

    if !vol.read_mrc_file(&format!("{input_dir}{input_filename}")) {
        cout(&format!(
            "ERROR:reading mrc file {input_dir}{input_filename}\n"
        ));
        cout(
            "Most possible source of error: file does not exist and there is not enough memory to load all the stack\n",
        );
        exit(-1);
    }

    let max_side = if vol.header_get_nx() > vol.header_get_ny() {
        vol.header_get_nx()
    } else {
        vol.header_get_ny()
    };
    let file_extension = match input_filename.rfind('.') {
        Some(found) => input_filename[found..].to_string(),
        None => String::new(),
    };
    let prealigned = file_extension == ".preali" || file_extension == ".ali";
    if dmax < 0 {
        dmax = if prealigned {
            max_side / 8
        } else {
            max_side / 4
        };
    }
    if max_pw_table < 0 {
        max_pw_table = if prealigned {
            if max_side / 10 > 50 {
                max_side / 10
            } else {
                50
            }
        } else if max_side / 6 > 50 {
            max_side / 6
        } else {
            50
        };
    }

    for i in 0..vol.header_get_nz() {
        frames.push(Frame::new(i, vol.header_get_nx(), vol.header_get_ny()));
    }

    let mut templ: Vec<Vec<f32>> = Vec::with_capacity(num_diff_marker_size.max(0) as usize);
    let zerotilt = vol.header_get_nz() / 2 + vol.header_get_nz() % 2 - 1;
    ZEROTILT.set(zerotilt);
    for kk in 0..num_diff_marker_size {
        cout(&format!(
            "Creating synthetic template number {kk} at {}\n",
            get_date()
        ));
        // Create the template
        let frame_number =
            (zerotilt as f64 + 10.0 * ((rand() as f64 / RAND_MAX as f64) - 0.5)) as i32;
        templ.push(create_synthetic_template(
            &frames,
            diameter[kk as usize],
            frame_number,
            &vol,
            white,
            kk,
        ));

        let ts = templ_size[kk as usize] as usize;
        let ascii = |t: &[f32]| -> Vec<u8> {
            let mut text = String::new();
            for i in 0..ts {
                for j in 0..ts {
                    text.push_str(&ostream_double(t[i * ts + j] as f64));
                    text.push(' ');
                }
                text.push('\n');
            }
            text.into_bytes()
        };
        if debug_mode {
            // debugging purposes to check template
            write_unchecked(
                &format!("{output_dir}debug/{basename}_syntheticTemplateASCII_{kk}.txt"),
                &ascii(&templ[kk as usize]),
            );
        }

        compute_ncc(&frames, &mut templ[kk as usize], &vol, kk);

        if debug_mode {
            write_unchecked(
                &format!("{output_dir}debug/{basename}_templateASCII_{kk}.txt"),
                &ascii(&templ[kk as usize]),
            );
        }
    }
    // estimate number of markers per image
    if targets == -1 {
        // we need to estimate number of markers
        targets = estimate_number_of_markers(&frames, &templ, &vol, num_diff_marker_size);
        cout(&format!(
            "Estimating number of markers automatically. Number of markers= {targets}\n"
        ));
    }
    let max_markers_prev_frame = targets as u32;
    let mut max_markers_next_frame = (targets * 4).min(MAX_TARGETS_NEXT_FRAME) as u32;
    if max_markers_next_frame < max_markers_prev_frame {
        max_markers_next_frame = max_markers_prev_frame;
    }
    MAX_MARKERS_PREV_FRAME.set(max_markers_prev_frame);
    MAX_MARKERS_NEXT_FRAME.set(max_markers_next_frame);

    // set parameters for pairwise correspondence
    let max_n_cliques = if 10u32.wrapping_mul(max_markers_prev_frame) > 500 {
        10u32.wrapping_mul(max_markers_prev_frame) as i32
    } else {
        500
    };
    if min_pw_table < 0 {
        min_pw_table = if max_markers_prev_frame < 20 {
            max_markers_prev_frame as i32
        } else {
            20
        };
    }

    cout(&format!(
        "Finding gold beads in different projections at {}\n",
        get_date()
    ));
    // Find peaks in all frames with the template
    find_all_peaks(
        &mut frames,
        &templ,
        &vol,
        num_diff_marker_size,
        x_ray,
        &mut points,
    );

    cout(&format!(
        "Computing pairwise correspondences at {}\n",
        get_date()
    ));
    // Compute local correspondences
    let mut correspondences: Vec<Vec<PairCorrespondence>> = Vec::new();
    let mut prev_pair = false;
    let mut flag_not_discard = true;
    let lr = format!("{basename}_LR");
    let mut i = zerotilt;
    while i > 0 {
        let mut ith_correspondence: Vec<PairCorrespondence> = Vec::new();
        for j in 1..=MAX_JUMPS as i32 {
            if i - j >= 0 {
                flag_not_discard = correspondence_regions(
                    &mut frames,
                    i as usize,
                    (i - j) as usize,
                    &mut points,
                    &mut ith_correspondence,
                    1,
                    max_markers_prev_frame,
                    max_markers_next_frame,
                    max_n_cliques,
                    dmax,
                    1,
                    1,
                    max_pw_table,
                    min_pw_table,
                    &lr,
                    10,
                    &vol,
                    prev_pair,
                    &output_dir,
                    &bin_path,
                );
            }
            if !flag_not_discard {
                break;
            }
        }
        if !flag_not_discard {
            // indicate that we can not use these frames
            for ll in (0..i).rev() {
                frames[ll as usize].discard = true;
            }
            break;
        }
        correspondences.push(ith_correspondence);
        prev_pair = MAX_JUMPS as usize <= correspondences.len();
        i -= 1;
    }
    prev_pair = false;
    let mut i = zerotilt as u32;
    while (i as usize) < frames.len() - 1 {
        let mut ith_correspondence: Vec<PairCorrespondence> = Vec::new();
        for j in 1..=MAX_JUMPS {
            if ((i + j) as usize) < frames.len() {
                flag_not_discard = correspondence_regions(
                    &mut frames,
                    i as usize,
                    (i + j) as usize,
                    &mut points,
                    &mut ith_correspondence,
                    1,
                    max_markers_prev_frame,
                    max_markers_next_frame,
                    max_n_cliques,
                    dmax,
                    1,
                    1,
                    max_pw_table,
                    min_pw_table,
                    &lr,
                    10,
                    &vol,
                    prev_pair,
                    &output_dir,
                    &bin_path,
                );
            }
            if !flag_not_discard {
                break;
            }
        }
        if !flag_not_discard {
            for ll in (i as usize + 1)..frames.len() {
                frames[ll].discard = true;
            }
            break;
        }
        correspondences.push(ith_correspondence);
        prev_pair = MAX_JUMPS as usize <= correspondences.len();
        i += 1;
    }

    cout(&format!(
        "Building trajectories using pairwise correspondence at {}\n",
        get_date()
    ));
    // Build the global trajectories with the local correspondences
    let mut t = find_trajectories(
        &frames,
        &correspondences,
        vol.header_get_nz(),
        max_markers_prev_frame,
        &mut points,
    );

    if debug_mode {
        cout(&format!(
            "Recovered {} provisional trajectories before optimization\n",
            t.len()
        ));
        let mut out: Vec<u8> = Vec::new();
        write_imod_fid_model(
            &t,
            vol.header_get_nx(),
            vol.header_get_ny(),
            vol.header_get_nz(),
            &basename,
            &mut out,
            &points,
        );
        write_unchecked(
            &format!("{output_dir}debug/{basename}_trajectoryBeforeOptimization.fid.txt"),
            &out,
        );
    }

    cout("\n-------------------------------------\n");

    let contour_x = Contour::from_trajectories(&t, 1, vol.header_get_nz(), &frames, &points);
    let contour_y = Contour::from_trajectories(&t, 2, vol.header_get_nz(), &frames, &points);

    let m_type = get_marker_type_from_trajectory(&t, &points);
    let mut sfm = SfmData::new(&contour_x, &contour_y, &m_type);
    drop(contour_x);
    drop(contour_y);
    drop(m_type);

    // once we do conversion from trajectories (T) to contour_x contour_y
    // structure we don't need T anymore
    free_trajectory_vector(&t, &mut points);
    t.clear();

    cout(&format!(
        "Reading tilt angles information at {}\n",
        get_date()
    ));

    // copy rawTilt file from origin if it was not in header
    let output_dir_imod = format!("{output_dir}IMOD/");

    if angles_header {
        // extract angles from header
        cmd = format!(
            "extracttilts -tilts -input {input_dir}{input_filename} -output {output_dir_imod}{basename}.rawtlt"
        );
        error = system(&cmd, &error_output_log);
        if error != 0 {
            cout(&format!(
                "ERROR: error executing the command {cmd} in function RAPTOPR::Main\n"
            ));
            exit(error);
        }
    } else {
        // copy angles from inputDir: `cp <inputDir><basename>.rawtlt
        // <outputDirIMOD> > <errorOutputLog>`
        cmd = format!("cp {input_dir}{basename}.rawtlt {output_dir_imod}");
        let _ = std::fs::File::create(&error_output_log);
        let source = format!("{input_dir}{basename}.rawtlt");
        error = match std::fs::copy(&source, format!("{output_dir_imod}{basename}.rawtlt")) {
            Ok(_) => 0,
            Err(err) => {
                eprintln!("cp: cannot copy '{source}': {err}");
                1
            }
        };
        if error != 0 {
            cout(&format!(
                "ERROR: error executing the command {cmd} in function RAPTOPR::Main\n"
            ));
            exit(error);
        }
    }

    let mut tilt_angles: Vec<f64> = Vec::new();
    let Some(mut in2) = IStream::open(&format!("{output_dir_imod}{basename}.rawtlt")) else {
        cout(&format!(
            "ERROR: RAPTOR can not find file {input_dir}{basename}.rawtlt containing tilt angles from goniometer\n"
        ));
        exit(-1);
    };
    let mut tilt_angle = 0f64;
    while in2.good() {
        in2.read_f64(&mut tilt_angle);
        tilt_angles.push(tilt_angle);
    }
    drop(in2);

    // necessary in case C++ reads the last line twice
    tilt_angles.resize(vol.header_get_nz() as usize, 0.0);
    // remove tilt angles that correspond to frames which are discarded
    for kk in (0..tilt_angles.len()).rev() {
        if frames[kk].discard {
            tilt_angles.remove(kk);
        }
    }

    cout("\nStarting optimization loop\n");

    let w = vol.header_get_nx();
    let h = vol.header_get_ny();
    let mut alpha_final = 0f64;
    let mut option_alpha = 0i32;
    let mut iter = 0i32;
    // computes number of markers overall to avoid doing more iterations if
    // there is no progress.
    let mut nnz = sfm.contour_x.calculate_nnz();
    let mut nnz_old: i32;
    let output_dir_debug = format!("{output_dir}debug/");
    while iter < 2 {
        cout(&format!(
            "Starting iteration {iter} of optimization procedure at {}\n",
            get_date()
        ));
        // sfm is freed inside decideAlphaOption
        let sfm_ = decide_alpha_option(sfm, &mut option_alpha, &mut frames);
        cout(&format!("Using alpha option={option_alpha}\n"));

        // _sfm is deleted inside the method
        let thesfm = sfm_estimation_with_ba(
            sfm_,
            &tilt_angles,
            w,
            h,
            PERCENTILE,
            &mut alpha_final,
            option_alpha,
            debug_mode,
        );

        if option_alpha == 0 {
            option_alpha = 2;
        } else if option_alpha == 2 {
            // we change option so next time alpha won't be calculated
            option_alpha = 1;
        }

        // thesfm is freed inside resid analysis
        let mut newsfm = resid_analysis(thesfm, debug_mode);

        if debug_mode {
            let mut out: Vec<u8> = Vec::new();
            write_imod_fid_model_sfm(
                &newsfm,
                vol.header_get_nx(),
                vol.header_get_ny(),
                vol.header_get_nz(),
                &basename,
                &mut out,
                &frames,
                &mut points,
            );
            write_unchecked(
                &format!(
                    "{output_dir_debug}{basename}_trajectoryBeforeFillContours_iter{iter}.fid.txt"
                ),
                &out,
            );
            let mut out: Vec<u8> = Vec::new();
            write_imod_fid_model_reproj(
                &newsfm,
                vol.header_get_nx(),
                vol.header_get_ny(),
                vol.header_get_nz(),
                &basename,
                &mut out,
                &frames,
                &mut points,
            );
            write_unchecked(
                &format!(
                    "{output_dir_debug}{basename}_trajectoryBeforeFillContoursReproj_iter{iter}.fid.txt"
                ),
                &out,
            );
        }

        fill_contours(&mut newsfm, &vol, &templ, &templ_size, debug_mode);

        if debug_mode {
            let mut out: Vec<u8> = Vec::new();
            write_imod_fid_model_sfm(
                &newsfm,
                vol.header_get_nx(),
                vol.header_get_ny(),
                vol.header_get_nz(),
                &basename,
                &mut out,
                &frames,
                &mut points,
            );
            write_unchecked(
                &format!(
                    "{output_dir_debug}{basename}_trajectoryAfterFillContours_iter{iter}.fid.txt"
                ),
                &out,
            );
            let mut out: Vec<u8> = Vec::new();
            write_imod_fid_model_reproj(
                &newsfm,
                vol.header_get_nx(),
                vol.header_get_ny(),
                vol.header_get_nz(),
                &basename,
                &mut out,
                &frames,
                &mut points,
            );
            write_unchecked(
                &format!(
                    "{output_dir_debug}{basename}_trajectoryAfterFillContoursReproj_iter{iter}.fid.txt"
                ),
                &out,
            );
        }

        sfm = join_similar_contours(newsfm);

        // update all the variables for teh while loop
        nnz_old = nnz;
        let _ = nnz_old;
        nnz = sfm.contour_x.calculate_nnz();
        iter += 1;
    }

    if option_alpha == 2 {
        option_alpha = 1;
    }

    // final pass to remove all the outliers without filling contours
    nnz_old = nnz + 10;
    while nnz_old > nnz {
        cout(&format!(
            "Starting iteration {iter} of optimization procedure at {}\n",
            get_date()
        ));
        let sfm_ = decide_alpha_option(sfm, &mut option_alpha, &mut frames);
        cout(&format!("Using alpha option={option_alpha}\n"));
        let thesfm = sfm_estimation_with_ba(
            sfm_,
            &tilt_angles,
            w,
            h,
            PERCENTILE,
            &mut alpha_final,
            option_alpha,
            debug_mode,
        );

        sfm = resid_analysis(thesfm, debug_mode);

        nnz_old = nnz;
        nnz = sfm.contour_x.calculate_nnz();
        iter += 1;
    }

    // Write everything to IMOD
    let mut tilt_option = 0i32;
    let mut mag_option = 0i32;
    let mut rot_option = 0i32;
    decide_tilt_align_options(&sfm, &mut tilt_option, &mut rot_option, &mut mag_option);

    t = sfm
        .contour_x
        .contour2trajectory(&sfm.contour_x, &sfm.contour_y, &frames, &mut points);
    drop(sfm);

    let fid_name = format!("{output_dir_imod}{basename}.fid.txt");
    let mut out: Vec<u8> = Vec::new();
    write_imod_fid_model(&t, w, h, vol.header_get_nz(), &basename, &mut out, &points);
    if std::fs::write(&fid_name, &out).is_err() {
        cout(&format!(
            "ERROR:file {fid_name} can no be created to write results\n"
        ));
        exit(-1);
    }

    let output_dir_align = format!("{output_dir}align/");

    if !tracking_only {
        // if trackingOnly==true the program ends here pretty much
        cout("------------------------------------------------\n");
        cout(&format!("Generating aligned stack at {}\n", get_date()));

        let script_name = format!("{output_dir_imod}{basename}_tiltalignScript.txt");
        // we need to convert alphaFinal to IMOD coordinates
        let mut alpha_imod = alpha_final - 90.0;
        if alpha_imod <= -90.0 {
            // to make sure there is no mirroring of the aligned stack versus
            // the original stack
            alpha_imod += 180.0;
        }
        let mut script: Vec<u8> = Vec::new();
        write_imod_tiltalign(
            alpha_imod,
            &output_dir_imod,
            &basename,
            &mut script,
            &frames,
            tilt_option,
            rot_option,
            mag_option,
        );
        if std::fs::write(&script_name, &script).is_err() {
            cout(&format!(
                "ERROR:file {script_name} can no be created to write results\n"
            ));
            exit(-1);
        }

        // execute tiltalign to compute transformations
        cmd = format!("tiltalign -param {output_dir_imod}{basename}_tiltalignScript.txt");
        error = system(&cmd, &error_output_log);
        if error != 0 {
            cout(&format!(
                "ERROR: error executing the command {cmd} in function RAPTOPR::Main\n"
            ));
            exit(error);
        }
        // create aligned stack
        cmd = format!(
            "newstack  -input {input_dir}{input_filename} -output {output_dir_align}{basename}.ali -offset 0,0 -xform {output_dir_imod}{basename}.xf -secs "
        );
        let mut flag_first = true;
        for (kk, frame) in frames.iter().enumerate() {
            if !frame.discard {
                if flag_first {
                    cmd.push_str(&kk.to_string());
                    flag_first = false;
                } else {
                    cmd.push_str(&format!(",{kk}"));
                }
            }
        }

        error = system(&cmd, &error_output_log);
        if error != 0 {
            cout(&format!(
                "ERROR: error executing the command {cmd} in function RAPTOR::Main\n"
            ));
            exit(error);
        }
        // bin aligned stack if necessary
        if binning != 1 {
            cmd = format!(
                "newstack -input {output_dir_align}{basename}.ali -output {output_dir_align}{basename}Bin.ali -bin {binning}"
            );
            error = system(&cmd, &error_output_log);
            if error != 0 {
                cout("ERROR: generating binned stack from original file\n");
                exit(-1);
            }
            // remove align stack with no binning: `mv <..>Bin.ali <..>.ali
            // > <errorOutputLog>`
            let _ = std::fs::File::create(&error_output_log);
            if std::fs::rename(
                format!("{output_dir_align}{basename}Bin.ali"),
                format!("{output_dir_align}{basename}.ali"),
            )
            .is_err()
            {
                cout("ERROR: copying binned align stack\n");
                exit(-1);
            }
        }
    }

    cout("------------------------------------\n");
    cout("Releasing memory\n");
    vol.clear();
    // free memory correctly
    free_trajectory_vector(&t, &mut points);
    t.clear();
    drop(diameter);
    drop(templ);

    let mut ending_rec = "_full.rec".to_string();
    for frame in &frames {
        if frame.discard {
            ending_rec = "_part.rec".to_string();
        }
    }

    // PERFORM RECONSTRUCTION USING IMOD IF USER HAS REQUESTED SO
    if rec != -10000 {
        if !(0..=2).contains(&rec) {
            cout(&format!(
                "ERROR: reconstruction option mode is {rec}.It should be 0,1 or 2 RAPTOR is skipping reconstruction.\n"
            ));
        } else {
            if thickness == 0 {
                // in case they have not setup the thickness
                thickness = 800;
            }
            cout(&format!("Starting reconstruction at {}\n", get_date()));
            write_imod_tilt_script(
                w / binning,
                h / binning,
                &output_dir_imod,
                &output_dir_align,
                &basename,
                thickness,
                &frames,
                &ending_rec,
            );
            cmd = format!("submfg {output_dir_align}tilt.com");
            error = system(&cmd, &error_output_log);
            if error != 0 {
                cout(&format!(
                    "ERROR: error executing the command {cmd} in function RAPTOR::Main\n"
                ));
                exit(error);
            }
            // rescale output if necessary
            if rec == 0 || rec == 1 {
                let range = if rec == 0 { "-c 0,255" } else { "-mm 0,32767" };
                cmd = format!(
                    "trimvol {range} {output_dir_align}{basename}{ending_rec} {output_dir_align}{basename}_fullTemp.rec"
                );
                error = system(&cmd, &error_output_log);
                if error != 0 {
                    cout(&format!(
                        "ERROR: error executing the command {cmd} in function RAPTOR::Main\n"
                    ));
                    exit(error);
                }
                // `rm -f` and `mv` of the rescaled volume over the original
                let _ = std::fs::remove_file(format!("{output_dir_align}{basename}{ending_rec}"));
                let _ = std::fs::rename(
                    format!("{output_dir_align}{basename}_fullTemp.rec"),
                    format!("{output_dir_align}{basename}{ending_rec}"),
                );
            }
        }
    }

    for (kk, frame) in frames.iter().enumerate() {
        if frame.discard {
            cout(&format!(
                "WARNING: projection {kk} was discarded during the alignment process. Not enough markers where found.\n"
            ));
        }
    }

    cout(&format!(
        "RAPTOR finished succesfully on dataset {input_dir}{input_filename} at {}\n",
        get_date()
    ));
    cout(&format!(
        "You can check aligned stack at {output_dir}align/\n"
    ));
    cout(&format!(
        "You can check log files at {output_dir}align/*.log\n"
    ));
    cout(&format!(
        "You can check fiducial marker model at {output_dir}IMOD/*.fid.txt\n"
    ));

    // remove temp folder
    let _ = std::fs::remove_dir_all(format!("{output_dir}temp/"));

    let output_dir_rec = format!("{output_dir}reconstruction/");

    // minimalistic mode
    if verbose == 0 {
        let _ = std::fs::create_dir(&output_dir_rec);
        let _ = std::fs::rename(
            format!("{output_dir_align}{basename}{ending_rec}"),
            format!("{output_dir_rec}{basename}{ending_rec}"),
        );
        // create minimal log file
        let out_log_name = format!("{output_dir_rec}{basename}_RAPTOR.log");
        let mut out_log = match std::fs::File::create(&out_log_name) {
            Ok(file) => file,
            Err(_) => {
                cout(&format!(
                    "ERROR: could not open file {out_log_name} to write final log file\n"
                ));
                exit(-1);
            }
        };

        let mut text = format!("RAPTOR started at {ini_time_stamp}\n");
        text.push_str("RAPTOR called with the following command:\n");
        for a in &argv_strings {
            text.push_str(&format!("{a} "));
        }
        text.push('\n');
        for (kk, frame) in frames.iter().enumerate() {
            if frame.discard {
                text.push_str(&format!(
                    "WARNING: projection {kk} was discarded during the alignment process. Not enough markers where found.\n"
                ));
            }
        }
        text.push_str(&format!("RAPTOR ended succesfully at {}\n", get_date()));
        let _ = out_log.write_all(text.as_bytes());
    }

    // restore the standard streams and close the log file
    drop(redirect_standard_streams(sbuf));
    frames.clear();

    // remove folders if minimal output is desired
    if verbose == 0 {
        let _ = std::fs::remove_dir_all(&output_dir_align);
        let _ = std::fs::remove_dir_all(&output_dir_imod);
        // we don't need to delete debug because if verbatim==0 then we are
        // not in debug mode; rename reconstruction as align
        let _ = std::fs::rename(
            format!("{output_dir}reconstruction"),
            format!("{output_dir}align"),
        );
    }

    0
}
