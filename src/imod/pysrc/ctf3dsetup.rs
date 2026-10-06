//! Translation of `IMOD/pysrc/ctf3dsetup`: sets up command files for
//! reconstruction with 3D CTF correction, by correcting and reconstructing
//! slabs at different heights and assembling them.
//!
//! A Python command script; its one function is [`next_com_name`] and its
//! top level is [`ctf3dsetup`].  The shared functions come from
//! `tomocoords.py` ([`super::tomocoords`]).  `splitcorrection` and
//! `splittilt` are Python-script translations, which `imodpy::run_cmd` runs
//! as children; `header` (through `getmrc`/`getmrcsize` and `runcmd`),
//! `imodinfo` and `xfmodel` are our own programs and run in process.
//! Python values keep their types: pixel sizes, shifts, offsets and the
//! erasing radius are floats written with `str()` ([`py_str_float`]);
//! thicknesses, binnings and sizes are ints.

use super::imodpy::{
    BOOL_VALUE, FLOAT_VALUE, INT_VALUE, OptionValue, clean_chunk_files, cleanup_files,
    complete_and_check_com_file, exit_from_imod_error, find_root_axis_and_extensions, fmtstr,
    get_mrc_size, glob_glob, option_value, os_path_splitext, parallel_boundary_size, prnstr,
    py_fixed, py_float, py_int_floordiv, py_int_mod, py_round, py_round_ndigits, py_slice_end,
    py_str_float, read_text_file, run_cmd, write_finish_and_message, write_text_file,
};
use super::pip::{
    exit_error, pip_get_boolean, pip_get_err_no, pip_get_in_out_file, pip_get_integer,
    pip_get_string, pip_read_or_parse_options,
};
use super::pysed::{PysedSrc, pysed, sed_del_and_add, sed_modify};
use super::tomocoords::{
    back_transform_erase_model, check_for_distortion, check_xtilt_ctf_vs_rec,
    find_split_com_number, get_axis_angle_and_transpose, get_ctf_options_check_if_corrected,
    get_essential_raw_options, get_fallback_raw_pixel, get_or_derive_com_file,
};
use std::ffi::OsString;
use std::io::Write as _;
use std::path::Path;

/// Matches `nextComName` (`ctf3dsetup:12`): the next command file name, a
/// sync file unless `sync` is false; the number is incremented after use.
pub fn next_com_name(com_num: &mut i32, out_root: &str, sync: bool) -> String {
    let com_name = if sync {
        format!("{out_root}-{:03}-sync.com", *com_num)
    } else {
        format!("{out_root}-{:03}.com", *com_num)
    };
    *com_num += 1;
    com_name
}

/// The script's top level (`ctf3dsetup:22-563`).  Returns the status of its
/// `sys.exit`; error paths exit the process as `exitError` does.
pub fn ctf3dsetup(arguments: &[OsString]) -> i32 {
    let progname = "ctf3dsetup";
    let prefix = format!("ERROR: {progname} - ");
    let argv: Vec<String> = arguments
        .iter()
        .map(|argument| argument.to_string_lossy().into_owned())
        .collect();
    let done = |status: i32| {
        let _ = std::io::stdout().flush();
        status
    };
    let lines_of = |result: Result<Option<Vec<String>>, String>| -> Vec<String> {
        result.ok().flatten().unwrap_or_default()
    };

    //
    // Setup runtime environment
    if std::env::var_os("IMOD_DIR").is_some() {
        super::imodpy::add_imod_bin_ignore_sighup();
    } else {
        print!("{prefix} IMOD_DIR is not defined!\n");
        return done(1);
    }

    let mut bound_pixels = parallel_boundary_size(2048);
    let mut out_root = "ctf3d".to_owned();
    let warn_xtilt_crit = 0.3;
    let mut raw_stack_name = String::new();
    let mut split_corr_size = String::new();
    let mut temp_stacks = String::new();

    // Fallbacks from ../manpages/autodoc2man 3 1 ctf3dsetup
    let options: Vec<String> = [
        "reccom:TiltCommandFile:FN:",
        "ctfcom:CorrectionComFile:FN:",
        "slabs:NumberOfSlabs:I:",
        "thickness:SlabThicknessInNm:I:",
        "invert:InvertSlabZOffsets:I:",
        "adjust:AdjustForAlignZShift:B:",
        "parallel:RunSlabsInParallel:B:",
        "procs:NumberOfProcessors:I:",
        "perproc:ChunksPerProcessor:I:",
        "erase:EraseFiducials:B:",
        "goldcom:GoldEraserComFile:FN:",
        "filter:FilterIn2D:B:",
        "2dcom:2DFilterComFile:FN:",
        "unaligned:UseUnalignedImages:B:",
        "reduce:FourierReduceByFactor:I:",
        "raw:RawStackFile:FN:",
        "pixel:RawPixelSize:F:",
        "xform:AlignTransformFile:FN:",
        "axis:AxisAngle:F:",
        "vertical:VerticalSlices:B:",
        "oldstyle:OldStyleXtilting:B:",
        "tempdir:TemporaryDirectory:CH:",
        "leave:LeaveTempFiles:B:",
        "boundary:BoundaryPixels:I:",
        "help:usage:B:",
    ]
    .iter()
    .map(|option| (*option).to_owned())
    .collect();

    let (_num_opts, _num_non_opts) = pip_read_or_parse_options(&argv, &options, progname, 1, 1, 0);

    // Get the com file name, derive a root name and new com file name, check exists
    let tilt_com_file = pip_get_in_out_file("TiltCommandFile", 0)
        .ok()
        .flatten()
        .unwrap_or_default();
    let (tilt_com_file, tilt_rootname) = complete_and_check_com_file(&tilt_com_file);

    // get or derive ctf correction file and eraser file
    let ctf_com_file = get_or_derive_com_file(
        "CorrectionComFile",
        "ctfcorrection",
        &tilt_com_file,
        "CTF correction",
    );

    let do_erase = pip_get_boolean("EraseFiducials", 0).unwrap_or(0) != 0;
    let mut erase_com_file = String::new();
    if do_erase {
        erase_com_file = get_or_derive_com_file(
            "GoldEraserComFile",
            "golderaser",
            &tilt_com_file,
            "erasing gold",
        );
    }

    // Determine filtering
    let do_filter = pip_get_boolean("FilterIn2D", 0).unwrap_or(0) != 0;
    let mut mtf_com_file = String::new();
    if do_filter {
        mtf_com_file = get_or_derive_com_file(
            "2DFilterComFile",
            "mtffilter",
            &tilt_com_file,
            "2D filtering",
        );
    }

    let adjust_for_z_shift = pip_get_boolean("AdjustForAlignZShift", 0).unwrap_or(0) != 0;

    // See if using raw input
    let use_raw_input = pip_get_boolean("UseUnalignedImages", 0).unwrap_or(0) != 0;

    let mut com_ext = String::new();
    let mut raw_root = String::new();
    let mut axis_let: Option<String> = None;
    if use_raw_input || adjust_for_z_shift {
        // Figure out the axis and extension if possible regardless
        let (com_ext_found, if_dual, raw_root_found, _type_ext, raw_ext) =
            find_root_axis_and_extensions(0, None);
        com_ext = com_ext_found;
        raw_root = raw_root_found;
        if if_dual == 0 {
            axis_let = Some(String::new());
        } else if if_dual == 2 {
            if tilt_com_file.contains("tilta") {
                axis_let = Some("a".to_owned());
            } else if tilt_com_file.contains("tiltb") {
                axis_let = Some("b".to_owned());
            }
        }

        if use_raw_input {
            // Get raw stack name
            raw_stack_name = pip_get_string("RawStackFile", "").unwrap_or_default();
            if raw_stack_name.is_empty() {
                if !raw_ext.is_empty() && !raw_root.is_empty() && axis_let.is_some() {
                    raw_root += axis_let.as_deref().unwrap_or("");
                    raw_stack_name = format!("{raw_root}.{raw_ext}");
                } else {
                    exit_error("The raw stack name must be entered, it cannot be deduced");
                }
            }
        }
    }

    let mut ali_xform_file = String::new();
    let mut raw_binning = 1;
    let mut raw_pix_size = 0.0_f64;
    let mut axis_angle = 0.0_f64;
    let mut transpose_xy = false;
    if use_raw_input {
        if !Path::new(&raw_stack_name).exists() {
            exit_error(&format!(
                "The raw stack file does not exist: {raw_stack_name}"
            ));
        }

        (ali_xform_file, raw_binning, raw_pix_size) = get_essential_raw_options(&raw_root);

        let (nx_corr, ny_corr, nz_corr) = match get_mrc_size(&raw_stack_name) {
            Ok(size) => size,
            Err(_) => exit_from_imod_error(progname),
        };

        split_corr_size = fmtstr(
            "-size {},{},{}",
            &[
                py_int_floordiv(nx_corr as i64, raw_binning as i64).to_string(),
                py_int_floordiv(ny_corr as i64, raw_binning as i64).to_string(),
                nz_corr.to_string(),
            ],
        );

        // Get the axis angle and whether X/Y are transposed
        (axis_angle, transpose_xy) = get_axis_angle_and_transpose(axis_let.as_deref());
    }

    // get other options
    let mut num_slabs = pip_get_integer("NumberOfSlabs", 0).unwrap_or(0);
    let max_slab_thick = pip_get_integer("SlabThicknessInNm", 0).unwrap_or(0);
    if num_slabs <= 0 && max_slab_thick <= 0 {
        exit_error("You must enter either the number of slabs or a slab thickness in nm");
    }
    if num_slabs > 0 && max_slab_thick > 0 {
        exit_error("You cannot enter both the number of slabs and a slab thickness");
    }
    if num_slabs > 0 && num_slabs < 3 {
        exit_error(
            "The number of slabs to compute at different heights must be entered and must be at least 3",
        );
    }

    let slabs_in_parallel = pip_get_boolean("RunSlabsInParallel", 0).unwrap_or(0) != 0;
    let mut num_procs = pip_get_integer("NumberOfProcessors", 8).unwrap_or(8);
    let procs_entered = pip_get_err_no() == 0;
    if slabs_in_parallel && procs_entered {
        exit_error("You cannot enter a number of processors when running slabs in parallel");
    }
    let chunks_per_proc = pip_get_integer("ChunksPerProcessor", 3).unwrap_or(3);
    if slabs_in_parallel && pip_get_err_no() == 0 {
        exit_error("You cannot enter chunks per processor when running slabs in parallel");
    }

    let mut invert_zoffsets = pip_get_integer("InvertSlabZOffsets", -1).unwrap_or(-1);
    bound_pixels = pip_get_integer("BoundaryPixels", bound_pixels).unwrap_or(bound_pixels);
    let leave_temp = pip_get_boolean("LeaveTempFiles", 0).unwrap_or(0) != 0;
    let mut temp_dir = pip_get_string("TemporaryDirectory", ".").unwrap_or_else(|_| ".".to_owned());
    // `os.path.isdir(tempDir)` and `os.access(tempDir, os.W_OK)`: the POSIX
    // `access` call itself
    let writable = Path::new(&temp_dir).is_dir()
        && std::ffi::CString::new(temp_dir.as_str())
            .map(|path| unsafe { libc::access(path.as_ptr(), libc::W_OK) } == 0)
            .unwrap_or(false);
    if !writable {
        exit_error(&format!(
            "Cannot write to {temp_dir} as a temporary directory"
        ));
    }
    temp_dir = temp_dir.replace('\\', "/");
    let do_vert = pip_get_boolean("VerticalSlices", 0).unwrap_or(0) != 0;
    let old_style = pip_get_boolean("OldStyleXtilting", 0).unwrap_or(0) != 0;
    if do_vert && old_style {
        exit_error("You cannot enter options for both vertical slices and old-style X tilting");
    }

    let mut align_zshift = 0.0_f64;
    if adjust_for_z_shift {
        let Some(axis) = axis_let.as_deref() else {
            exit_error(
                "Cannot adjust for Z shift; the axis could not be determined from files in the directory",
            );
        };
        let align_com_file = format!("align{axis}.com");
        let align_lines =
            read_text_file(&align_com_file, Some("Tiltalign command file"), false, None)
                .unwrap_or_default();
        // `BUGS.md` (ctf3dsetup/subtomosetup `AxisZShift`), fixed in
        // translation: with no `AxisZShift` entry the source adds `None` to a
        // number and dies with a TypeError; a missing entry is tiltalign's
        // default of 0 here.
        align_zshift = match option_value(
            &align_lines,
            "AxisZShift",
            FLOAT_VALUE,
            false,
            1,
            None,
            None,
        ) {
            Some(OptionValue::Floats(values)) => values[0],
            _ => 0.,
        };
    }

    if tilt_rootname == "tilta" || tilt_rootname == "tiltb" {
        out_root.push_str(&tilt_rootname[4..5]);
    }

    let tilt_lines =
        read_text_file(&tilt_com_file, Some("Tilt command file"), false, None).unwrap_or_default();
    let ctf_lines = read_text_file(
        &ctf_com_file,
        Some("CTF correction command file"),
        false,
        None,
    )
    .unwrap_or_default();
    let mut mtf_lines = Vec::new();
    if do_filter {
        mtf_lines = read_text_file(
            &mtf_com_file,
            Some("2D filtering command file"),
            false,
            None,
        )
        .unwrap_or_default();
    }
    let mut erase_lines = Vec::new();
    if do_erase {
        erase_lines = read_text_file(
            &erase_com_file,
            Some("Gold erasing command file"),
            false,
            None,
        )
        .unwrap_or_default();
    }

    let int1 = |lines: &[String], option: &str, nocase: bool| match option_value(
        lines, option, INT_VALUE, nocase, 1, None, None,
    ) {
        Some(OptionValue::Integers(values)) => Some(values[0]),
        _ => None,
    };
    let float1 = |lines: &[String], option: &str| match option_value(
        lines,
        option,
        FLOAT_VALUE,
        false,
        1,
        None,
        None,
    ) {
        Some(OptionValue::Floats(values)) => Some(values[0]),
        _ => None,
    };
    let string_of = |lines: &[String], option: &str| match option_value(
        lines, option, 0, true, 0, None, None,
    ) {
        Some(OptionValue::String(value)) => Some(value),
        _ => None,
    };

    let sirt_iter = int1(&tilt_lines, "sirtiterations", true);
    if sirt_iter.is_some_and(|value| value != 0) {
        exit_error(
            "The Tilt command file has a SIRTiterations entry; this cannot be usedwith 3D CTF correction",
        );
    }
    let ub_thickness = int1(&tilt_lines, "thickness", true);
    let mut binning = int1(&tilt_lines, "imagebinned", true);
    let recfile = string_of(&tilt_lines, "outputfile");
    let alifile = string_of(&tilt_lines, "inputproj");
    let tilt_mode = int1(&tilt_lines, "mode", true);
    let shift_arr = match option_value(&tilt_lines, "shift", FLOAT_VALUE, true, 0, None, None) {
        Some(OptionValue::Floats(values)) => Some(values),
        _ => None,
    };
    let mut pixel_size = float1(&ctf_lines, "PixelSize");
    let defocus_tol_entry = int1(&ctf_lines, "DefocusTol", false);
    let expand_fac = float1(&ctf_lines, "ExpandedByFactor");
    if let (Some(pixel), Some(expand)) = (pixel_size, expand_fac)
        && pixel != 0.
        && expand != 0.
    {
        pixel_size = Some(pixel / expand);
    }

    let mut offset_sign = 1.0_f64;
    let invert_opt = option_value(
        &ctf_lines,
        "InvertTiltAngles",
        BOOL_VALUE,
        false,
        0,
        None,
        None,
    );
    if invert_opt == Some(OptionValue::Boolean(true)) && invert_zoffsets < 0 {
        invert_zoffsets = 1;
    }
    if invert_zoffsets > 0 {
        offset_sign = -1.;
    }

    // Get other parameters, make sure aligned stack is not corrected and get its size
    let ctf_options = get_ctf_options_check_if_corrected(
        progname,
        &tilt_lines,
        &ctf_lines,
        &raw_stack_name,
        do_filter,
    );
    let (mut nx, mut ny, nz) = (ctf_options.nx, ctf_options.ny, ctf_options.nz);
    let xaxis_tilt = ctf_options.xaxis_tilt;
    let use_gpu = ctf_options.tilt_use_gpu;
    let ctf_xtilt = ctf_options.ctf_xtilt;
    let mut ali_pix_size = ctf_options.ctf_pix_size;
    let mut ctf_input = ctf_options.ctf_input.unwrap_or_default();
    let ctf_output = ctf_options.ctf_output.unwrap_or_default();

    let Some(recfile) = recfile.filter(|name| !name.is_empty()) else {
        exit_error(&format!("Cannot find output file name in {tilt_com_file}"));
    };

    if binning.is_none_or(|value| value == 0) {
        binning = Some(1);
    }
    let mut binning = binning.unwrap_or(1);
    let ub_thickness = match ub_thickness {
        Some(value) if value != 0 => value,
        _ => exit_error(&format!("Cannot find thickness in {tilt_com_file}")),
    };

    let cannot_find = || -> ! {
        exit_error(&format!(
            "Cannot find needed information in {ctf_com_file} (PixelSize, input or output file)"
        ))
    };

    // Modify for raw input: aliPixSize was the pixel size of the raw stack
    let mut ali_binning = 0.0_f64;
    let (mut nx_full, mut ny_full) = (0, 0);
    if use_raw_input {
        check_for_distortion(axis_let.as_deref(), 1, progname);
        if expand_fac.is_some_and(|value| value != 0.) {
            exit_error("You cannot reconstruct from raw images when applying an expansion factor");
        }
        binning = raw_binning;
        let header_pix = ali_pix_size.unwrap_or(0.);
        if raw_pix_size == 0. {
            raw_pix_size = get_fallback_raw_pixel(header_pix, axis_let.as_deref(), &com_ext);
        }
        // `BUGS.md` (ctf3dsetup raw `PixelSize`), fixed in translation:
        // `pixelSize / rawPixSize` with no `PixelSize` in the CTF command
        // file divides `None` and dies with a TypeError before the source's
        // own check below; that check's error is given here instead.
        let Some(ctf_pixel) = pixel_size.filter(|value| *value != 0.) else {
            cannot_find();
        };
        ali_binning = ctf_pixel / raw_pix_size;
        pixel_size = Some(raw_pix_size * raw_binning as f64);
        ali_pix_size = Some(header_pix * raw_binning as f64);

        // For raw input:
        // binning = rawBinning is the amount it is going to be binned
        // rawPixSize could have been entered as option in nm, otherwise
        //  it is size in nm from header if != 1., otherwise from track.com
        // aliPixSize was originally the raw header pixel size in A and is now the pixel size
        //  in A that will be used for raw input
        // pixelSize is the rawPixelSize in nm times raw binning, so size in nm
        // aliBinning is the pixel size from ctfcorrection divided by rawPixSize

        let (nx_raw, ny_raw) = (nx, ny);
        (nx_full, ny_full) = (nx, ny);
        if transpose_xy {
            nx_full = ny_raw;
            ny_full = nx_raw;
        }
        nx = py_int_floordiv(nx_full as i64, binning as i64) as i32;
        ny = py_int_floordiv(ny_full as i64, binning as i64) as i32;
    }

    let pixel_size = match pixel_size {
        Some(value) if value != 0. && !ctf_input.is_empty() && !ctf_output.is_empty() => value,
        _ => cannot_find(),
    };
    let ali_pix_size = ali_pix_size.unwrap_or(0.);

    // Take care of back transforming the eraser model
    let mut raw_erase_fid = String::new();
    if use_raw_input && do_erase {
        raw_erase_fid = back_transform_erase_model(
            &erase_lines,
            &ali_xform_file,
            raw_pix_size * 10.,
            &erase_com_file,
            progname,
        );
        temp_stacks = raw_erase_fid.clone();

        // The radius to erase must be modified by ratio of stack binning at which it
        // was set/tested and the raw stack binning
        if let Some(mut radius) = float1(&erase_lines, "BetterRadius").filter(|value| *value != 0.)
        {
            let mut need_rad_fix = true;

            // But try to fix it from info in the golderaser log
            let (erase_root, _ext) = os_path_splitext(&erase_com_file);
            let erase_log = format!("{erase_root}.log");
            if Path::new(&erase_log).exists() {
                let log_lines =
                    read_text_file(&erase_log, Some("Log file from gold erasing"), false, None)
                        .unwrap_or_default();
                for line in &log_lines {
                    if line.contains("[CCE1]") {
                        let lsplit: Vec<&str> = line.split_whitespace().collect();
                        let (test_radius, test_pixel) = match (
                            lsplit.get(1).and_then(|text| py_float(text)),
                            lsplit.get(2).and_then(|text| py_float(text)),
                        ) {
                            (Some(r), Some(p)) => (r, p),
                            _ => exit_error(
                                "[CCE1] line in gold erasing log file does not have the correct form",
                            ),
                        };

                        if test_pixel == 0. {
                            exit_error(&format!(
                                "Cannot scale the bead erasing diameter because of bad output in the golderaser log file; rerun {erase_com_file}"
                            ));
                        }
                        need_rad_fix = false;
                        let _was_rad = radius * ali_binning / raw_binning as f64;
                        radius = test_radius * test_pixel / ali_pix_size;
                    }
                }
            }

            if need_rad_fix {
                // `BUGS.md` (ctf3dsetup `None` paths): `os.path.exists(None)`
                // raises a TypeError when tilt.com has no input file; a
                // missing name counts as a file that does not exist.
                if !alifile
                    .as_deref()
                    .is_some_and(|name| Path::new(name).exists())
                {
                    let mut bin_or_unbin = "unbinned pixels".to_owned();
                    if ali_binning > 1.01 {
                        bin_or_unbin = format!("pixels binned by {}", py_fixed(ali_binning, 0, 0));
                    }
                    prnstr(
                        &format!(
                            "WARNING: Ctf3dsetup - Assuming that the entered diameter of {} for bead erasing applies to {bin_or_unbin}",
                            py_fixed(radius * 2., 0, 1)
                        ),
                        "\n",
                        false,
                    );
                }
                radius *= ali_binning / raw_binning as f64;
            }
            let _ = radius;
        }
    }

    // `startingZshift` is the int 0 or a float from the SHIFT entry; it is
    // only ever added to floats.
    let mut starting_zshift = 0.0_f64;
    let mut x_shift = 0.0_f64;
    if let Some(shift) = shift_arr.as_ref().filter(|values| !values.is_empty()) {
        x_shift = shift[0];
        if shift.len() > 1 {
            starting_zshift = shift[1];
        }
    }

    let mut z_shift_adjustment = 0.0_f64;
    if adjust_for_z_shift {
        z_shift_adjustment = (align_zshift + starting_zshift) / binning as f64;
    }

    // Figure out slab sizes and number of slabs
    let pixels_thickness = py_int_floordiv(ub_thickness as i64, binning as i64);
    let nm_thickness = py_round(pixels_thickness as f64 * pixel_size) as i64;
    if max_slab_thick > 0 {
        num_slabs = py_int_floordiv(
            nm_thickness + max_slab_thick as i64 - 1,
            max_slab_thick as i64,
        ) as i32;
        if num_slabs < 3 {
            exit_error(&fmtstr(
                "The entered slab thickness would give only {} slabs; you must enter a value less than {}",
                &[
                    num_slabs.to_string(),
                    py_int_floordiv(nm_thickness, 2).to_string(),
                ],
            ));
        }
    }
    let pixels_per_slab = py_int_floordiv(pixels_thickness, num_slabs as i64);
    let nm_per_slab = pixels_per_slab as f64 * pixel_size;
    prnstr(
        &format!(
            "{num_slabs} slabs of thickness {} nm will be computed",
            py_fixed(nm_per_slab, 0, 0)
        ),
        "\n",
        false,
    );
    let slab_nm_int = py_round(nm_per_slab) as i64;
    let defocus_tol = match defocus_tol_entry {
        Some(value) if value != 0 => (value as i64).min(slab_nm_int),
        _ => slab_nm_int,
    };

    // set up the slab limits
    let remainder = py_int_mod(pixels_thickness, num_slabs as i64);
    let mut shifts_pixels: Vec<f64> = Vec::new();
    let mut offsets_pix: Vec<f64> = Vec::new();
    let mut thicknesses: Vec<i64> = Vec::new();
    let mut slab_start: i64 = 0;
    for ind in 0..num_slabs as i64 {
        let mut slab_end = slab_start + pixels_per_slab;
        if ind < remainder {
            slab_end += 1;
        }

        // Positive tilt shift for slabs at the bottom
        // But also positive Z offset for lower slabs, because negative Y is really positive Z
        // If adjusting for Z shift, the slab shift that is in the opposite direction from
        // that Z shift is is the slab that started out at 0, so ADD the Z shift
        let slab_shift = (pixels_thickness - (slab_end + slab_start)) as f64 / 2.;
        shifts_pixels.push(slab_shift * binning as f64 + starting_zshift);
        offsets_pix.push((slab_shift + z_shift_adjustment) * offset_sign);
        thicknesses.push((slab_end - slab_start) * binning as i64);
        slab_start = slab_end;
    }

    // Test for X tilt consistency and if it matters
    if !use_raw_input {
        check_xtilt_ctf_vs_rec(
            xaxis_tilt,
            ctf_xtilt,
            warn_xtilt_crit,
            ny as f64 * pixel_size,
            nm_per_slab,
            "slab",
            progname,
        );
    }

    // figure out maximum views entry to splitcorrection
    let mut max_slices: i64 = 0;
    if !slabs_in_parallel {
        if use_gpu >= 0 && !procs_entered {
            num_procs = 1;
        }
        if num_procs > 1 {
            let num_chunks = num_procs as i64 * chunks_per_proc as i64;
            max_slices = 1.max(py_int_floordiv(nz as i64 + num_chunks - 1, num_chunks));
        }
    }

    clean_chunk_files(&out_root, false);
    let bound_list = glob_glob(&format!("{out_root}-bound-*.info"));
    if !bound_list.is_empty() {
        cleanup_files(&bound_list);
    }

    let (mut set_name, _ali_ext) = os_path_splitext(&ctf_input);
    if set_name.ends_with("_ali") && _ali_ext != ".ali" {
        set_name = py_slice_end(&set_name, 4);
    }
    let (ali_root, ali_ext) = os_path_splitext(&ctf_output);
    let (_rec_root, mut rec_ext) = os_path_splitext(&recfile);
    if rec_ext != ".rec" {
        rec_ext = format!("_rec{rec_ext}");
    }
    let ali_root = format!("{temp_dir}/{ali_root}");

    // If using raw input, set up input name and optional binning com file
    let mut com_num = 1;
    if use_raw_input {
        ctf_input = raw_stack_name.clone();
        if binning > 1 {
            ctf_input = format!("{ali_root}_red{binning}_tmp{ali_ext}");
            temp_stacks.push_str(&format!(" \"{ctf_input}\""));
            let newst_lines = vec![format!(
                "$newstack -ftreduce {binning} {raw_stack_name} \"{ctf_input}\""
            )];
            let _ = write_text_file(
                &next_com_name(&mut com_num, &out_root, true),
                &newst_lines,
                false,
            );
        }
    }

    // If filtering, do it in an initial command file
    if do_filter {
        let filt_input = ctf_input.clone();
        ctf_input = format!("{ali_root}_filt_tmp{ali_ext}");
        temp_stacks.push_str(&format!(" \"{ctf_input}\""));
        let mtfsed = vec![
            sed_modify("InputFile", &filt_input, '|'),
            sed_modify("OutputFile", &ctf_input, '|'),
            sed_modify("PixelSize", &py_str_float(pixel_size), '|'),
        ];
        let mtf_mod = lines_of(pysed(
            &mtfsed,
            PysedSrc::Lines(&mtf_lines),
            None,
            false,
            '|',
            false,
        ));
        let _ = write_text_file(
            &next_com_name(&mut com_num, &out_root, true),
            &mtf_mod,
            false,
        );
        if !use_raw_input {
            split_corr_size = format!("-size {nx},{ny},{nz}");
        }
    }

    // Work on slabs now
    let mut rec_name_lines: Vec<String> = Vec::new();
    let mut ctf_com_out = String::new();
    let mut tilt_com_out = String::new();
    for slab in 0..num_slabs as usize {
        let (ali_name, erase_name);
        if slabs_in_parallel {
            ali_name = format!("{ali_root}_{slab:02}{ali_ext}");
            erase_name = format!("{ali_root}_erase_{slab:02}{ali_ext}");
        } else {
            ali_name = format!("{ali_root}_tmp{ali_ext}");
            erase_name = format!("{ali_root}_erase_tmp{ali_ext}");
        }

        let rec_name = format!("{ali_root}_{slab:02}{rec_ext}");
        rec_name_lines.push(format!("InputFile {rec_name}"));
        let mut tiltsed = vec![
            sed_modify("InputProjections", &ali_name, '|'),
            sed_modify("OutputFile", &rec_name, '|'),
            sed_modify("THICKNESS", &thicknesses[slab].to_string(), '|'),
        ];
        tiltsed.extend(sed_del_and_add(
            "SHIFT",
            &format!(
                "{} {}",
                py_str_float(py_round_ndigits(x_shift, 2)),
                py_str_float(py_round_ndigits(shifts_pixels[slab], 2))
            ),
            "THICKNESS",
            '|',
        ));
        if use_raw_input {
            tiltsed.extend(sed_del_and_add("UseUnalignedImages", "1", "THICKNESS", '|'));
            tiltsed.extend(sed_del_and_add(
                "AlignTransformFile",
                &ali_xform_file,
                "THICKNESS",
                '|',
            ));
            tiltsed.extend([
                sed_modify("FULLIMAGE", &format!("{nx_full} {ny_full}"), '|'),
                sed_modify("SUBSETSTART", "0 0", '|'),
                sed_modify("IMAGEBINNED", &binning.to_string(), '|'),
            ]);
        }

        let mut tilt_mod = lines_of(pysed(
            &tiltsed,
            PysedSrc::Lines(&tilt_lines),
            None,
            true,
            '|',
            false,
        ));

        let mut ctfsed = vec![sed_modify("OutputFileName", &ali_name, '|')];
        ctfsed.extend(sed_del_and_add(
            "OffsetInZ",
            &py_str_float(offsets_pix[slab]),
            "DefocusFile",
            '|',
        ));
        ctfsed.extend(sed_del_and_add(
            "UseGPU",
            &use_gpu.to_string(),
            "DefocusFile",
            '|',
        ));
        ctfsed.extend(sed_del_and_add(
            "DefocusTol",
            &defocus_tol.to_string(),
            "DefocusFile",
            '|',
        ));
        if !temp_stacks.is_empty() || use_raw_input {
            ctfsed.push(sed_modify("InputStack", &ctf_input, '|'));
        }
        if use_raw_input {
            ctfsed.extend([
                "|TransformFile|d".to_owned(),
                sed_modify("PixelSize", &py_str_float(pixel_size), '|'),
            ]);
            ctfsed.extend(sed_del_and_add(
                "XAxisTilt",
                &py_str_float(xaxis_tilt),
                "DefocusFile",
                '|',
            ));
            ctfsed.extend(sed_del_and_add(
                "AxisAngle",
                &py_str_float(axis_angle),
                "DefocusFile",
                '|',
            ));
        }

        let ctf_mod = lines_of(pysed(
            &ctfsed,
            PysedSrc::Lines(&ctf_lines),
            None,
            false,
            '|',
            false,
        ));

        let mut erase_mod: Vec<String> = Vec::new();
        if do_erase {
            let mut erase_sed = vec![
                sed_modify("InputFile", &ali_name, '|'),
                sed_modify("OutputFile", &erase_name, '|'),
            ];
            if use_raw_input {
                erase_sed.push(sed_modify("ModelFile", &raw_erase_fid, '|'));

                // The radius to erase must be modified by ratio of stack binning at which it
                // was set/tested and the raw stack binning
                if let Some(mut radius) =
                    float1(&erase_lines, "BetterRadius").filter(|value| *value != 0.)
                {
                    let mut need_rad_fix = true;
                    let (erase_root, _ext) = os_path_splitext(&erase_com_file);
                    let erase_log = format!("{erase_root}.log");
                    if Path::new(&erase_log).exists() {
                        let log_lines = read_text_file(
                            &erase_log,
                            Some("Log file from gold erasing"),
                            false,
                            None,
                        )
                        .unwrap_or_default();
                        for line in &log_lines {
                            if line.contains("[CCE1]") {
                                let lsplit: Vec<&str> = line.split_whitespace().collect();
                                let (test_radius, test_pixel) = match (
                                    lsplit.get(1).and_then(|text| py_float(text)),
                                    lsplit.get(2).and_then(|text| py_float(text)),
                                ) {
                                    (Some(r), Some(p)) => (r, p),
                                    _ => exit_error(
                                        "[CCE1] line in gold erasing log file does not have the correct form",
                                    ),
                                };

                                need_rad_fix = false;
                                let _was_rad = radius * ali_binning / raw_binning as f64;
                                radius = test_radius * test_pixel / ali_pix_size;
                            }
                        }
                    }

                    if need_rad_fix {
                        radius *= ali_binning / raw_binning as f64;
                    }
                    erase_sed.push(sed_modify("BetterRadius", &py_str_float(radius), '|'));
                }
            }

            erase_mod = lines_of(pysed(
                &erase_sed,
                PysedSrc::Lines(&erase_lines),
                None,
                false,
                '|',
                false,
            ));
            erase_mod.push(format!("$b3drename \"{erase_name}\" \"{ali_name}\""));
        }

        // For parallel slabs, combine the two command into one file and output it
        if slabs_in_parallel {
            let mut all_lines = ctf_mod.clone();
            if do_erase {
                all_lines.extend(erase_mod.iter().cloned());
            }
            all_lines.extend(tilt_mod.iter().cloned());
            all_lines.push(format!("$b3dremove \"{ali_name}\""));
            let _ = write_text_file(
                &next_com_name(&mut com_num, &out_root, false),
                &all_lines,
                false,
            );
        } else {
            // For sequential slabs, write each com either to numbered sync (1 proc) or to
            // temp files
            let mut erase_com_out = String::new();
            if num_procs > 1 {
                ctf_com_out = format!("{temp_dir}/ctfcorrection.tmp.com");
                tilt_com_out = format!("{temp_dir}/tilt.tmp.com");
            } else {
                ctf_com_out = next_com_name(&mut com_num, &out_root, true);
                if do_erase {
                    erase_com_out = next_com_name(&mut com_num, &out_root, true);
                }
                tilt_com_out = next_com_name(&mut com_num, &out_root, true);
                if !leave_temp {
                    tilt_mod.push(format!("$b3dremove {ali_name}"));
                }
            }
            let _ = write_text_file(&ctf_com_out, &ctf_mod, false);
            let _ = write_text_file(&tilt_com_out, &tilt_mod, false);
            if num_procs == 1 && do_erase {
                let _ = write_text_file(&erase_com_out, &erase_mod, false);
            }

            // Now if multiple processors/GPUs, split each
            if num_procs > 1 {
                let split_lines = match run_cmd(
                    &fmtstr(
                        "splitcorrection -i {} -o -m {} -b {} -uni {} -r {} \"{}\"",
                        &[
                            com_num.to_string(),
                            max_slices.to_string(),
                            bound_pixels.to_string(),
                            split_corr_size.clone(),
                            out_root.clone(),
                            ctf_com_out.clone(),
                        ],
                    ),
                    None,
                    None,
                    None,
                    &[],
                ) {
                    Ok(lines) => lines.unwrap_or_default(),
                    Err(_) => exit_from_imod_error(progname),
                };
                let num_added = find_split_com_number(&split_lines, "output of splitcorrection");
                com_num += num_added;

                if do_erase {
                    let _ = write_text_file(
                        &next_com_name(&mut com_num, &out_root, true),
                        &erase_mod,
                        false,
                    );
                }

                let mut com_lines = vec![
                    format!("CommandFile  {tilt_com_out}"),
                    format!("RootNameOfOutput  {out_root}"),
                    format!("ProcessorNumber  {num_procs}"),
                    format!(
                        "TargetChunks  {}",
                        num_procs as i64 * chunks_per_proc as i64
                    ),
                    format!("BoundaryPixels  {bound_pixels}"),
                    format!("InitialComNumber  {com_num}"),
                    "OpenForMoreComs  1".to_owned(),
                    "UniqueInfoFile  1".to_owned(),
                    format!("DimensionsOfStack  {nx},{ny}"),
                ];
                if do_vert {
                    com_lines.push("VerticalSlices  1".to_owned());
                }
                if old_style {
                    com_lines.push("OldStyleXtiltPenalty  0.5".to_owned());
                }

                let split_lines = match run_cmd(
                    "splittilt -StandardInput",
                    Some(&com_lines),
                    None,
                    None,
                    &[],
                ) {
                    Ok(lines) => lines.unwrap_or_default(),
                    Err(_) => exit_from_imod_error(progname),
                };
                let num_added = find_split_com_number(&split_lines, "output of splittilt");
                com_num += num_added;
                if !leave_temp {
                    let last_sync = format!("{out_root}-{:03}-sync.com", com_num - 1);
                    let mut last_lines =
                        read_text_file(&last_sync, None, false, None).unwrap_or_default();
                    last_lines.push(format!("$b3dremove \"{ali_name}\""));
                    let _ = write_text_file(&last_sync, &last_lines, false);
                }
            }
        }
    }

    // Time to assemble the slabs and clean up
    if !slabs_in_parallel && num_procs > 1 {
        cleanup_files(&[ctf_com_out.clone(), tilt_com_out.clone()]);
    }

    let mut assemble_lines: Vec<String> = Vec::new();
    if tilt_mode == Some(12) {
        assemble_lines = vec!["$setenv IMOD_WRITE_FLOATS_16BIT 1".to_owned()];
    }
    assemble_lines.extend([
        "$assemblevol -StandardInput".to_owned(),
        format!("NumberOfFilesInY {num_slabs}"),
        format!("OutputFile {set_name}_3dctf{rec_ext}"),
    ]);
    assemble_lines.extend(rec_name_lines);
    if !leave_temp {
        assemble_lines.push(format!("$b3dremove -g \"{ali_root}_[0-9][0-9]{rec_ext}\""));
        if slabs_in_parallel {
            assemble_lines.push(format!("$b3dremove -g \"{ali_root}_[0-9][0-9]{ali_ext}\""));
        }
        if !temp_stacks.is_empty() {
            assemble_lines.push(format!("$b3dremove {temp_stacks}"));
        }
    }

    write_finish_and_message(
        Some(&mut assemble_lines),
        &out_root,
        com_num,
        !slabs_in_parallel && num_procs == 1,
        ".com",
    );

    done(0)
}
