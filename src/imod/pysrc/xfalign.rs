//! Translation of `IMOD/pysrc/xfalign`: finds alignment transforms between
//! serial sections with `xfsimplex`, optionally after an initial
//! cross-correlation and followed by warping alignment with `tiltxcorr`.
//!
//! The script's top level is [`xfalign`]; its functions are [`cleanup`],
//! [`make_gray_file`] and [`make_tiltxcorr_com`], which read the script's
//! module-level settings from [`Globals`].  `clip`, `tiltxcorr`,
//! `xfsimplex` and `newstack` are our own programs and run in process
//! through `imodpy::run_cmd`.  The `-diff` report is still read from
//! `xfsimplex`'s printed "FINAL VALUES" block, as the script reads it.
//!
//! Numbers the script writes with `str()` keep their Python type: the
//! default `limits` and `xcfilter` entries are partly ints (`0`, `4`), and
//! entered values are floats, so each value carries its printed text.

use super::imodpy::{
    MrcInfo, add_imod_bin_ignore_sighup, cleanup_files, exit_from_imod_error, fmtstr, get_mrc,
    imod_temp_dir, make_backup_file, parse_list, pass_on_key_interrupt, print_pid, prnstr,
    py_float, py_round, py_str_float, read_text_file, run_cmd, write_text_file,
};
use super::pip::{
    exit_error, pip_get_boolean, pip_get_err_no, pip_get_float, pip_get_float_array,
    pip_get_in_out_file, pip_get_integer, pip_get_string, pip_get_two_integers,
    pip_read_or_parse_options, python_uncaught,
};
use std::ffi::OsString;
use std::io::Write as _;
use std::path::Path;

/// A Python number as the script holds it: its value and its `str()`.
#[derive(Clone, Debug)]
struct PyNumber {
    value: f64,
    text: String,
}

impl PyNumber {
    fn int(value: i64) -> Self {
        PyNumber {
            value: value as f64,
            text: value.to_string(),
        }
    }
    fn float(value: f64) -> Self {
        PyNumber {
            value,
            text: py_str_float(value),
        }
    }
}

/// The module-level names [`cleanup`], [`make_gray_file`] and
/// [`make_tiltxcorr_com`] read.
struct Globals {
    tmpxf: String,
    tmp_raw_ali: String,
    infile: String,
    xcfilter: Vec<PyNumber>,
    warp_patch_x: i32,
    warp_patch_y: i32,
    bound_model: String,
    seed_model: String,
    warp_limits_x: i32,
    warp_limits_y: i32,
    warp_binning: i32,
    limits: Vec<PyNumber>,
    pre_binning: i32,
    newsize: [i64; 2],
    ifsize: i32,
    xmin: i64,
    xmax: i64,
    ymin: i64,
    ymax: i64,
    skip_list: Vec<i32>,
    break_list: Vec<i32>,
}

/// `def cleanup()` (`xfalign:13`): cleanup files.
fn cleanup(g: &Globals) {
    cleanup_files(&[
        g.tmpxf.clone(),
        format!("{}~", g.tmpxf),
        g.tmp_raw_ali.clone(),
        format!("{}~", g.tmp_raw_ali),
    ]);
}

/// `def makeGrayFile(filename)` (`xfalign:17`): make a gray scale file from
/// color file if necessary and return the name to use.
fn make_gray_file(filename: &str) -> String {
    let (root, ext) = super::imodpy::os_path_splitext(filename);
    let newname = format!("{root}_gray{ext}");
    let mtime = |name: &str| {
        std::fs::metadata(name)
            .and_then(|meta| meta.modified())
            .ok()
    };
    if Path::new(&newname).exists() && mtime(&newname) > mtime(filename) {
        return newname;
    }
    if Path::new(&newname).exists() {
        prnstr(
            &format!("Remaking gray scale file {newname} because it is older than {filename}"),
            "\n",
            false,
        );
    } else {
        prnstr(
            &format!("Making gray scale file for computing alignments: {newname}"),
            "\n",
            false,
        );
    }
    if run_cmd(
        &fmtstr(
            "clip resize -m 0 \"{}\" \"{}\"",
            &[filename.to_owned(), newname.clone()],
        ),
        None,
        None,
        None,
        &[],
    )
    .is_err()
    {
        // Fixed in translation (BUGS.md, `xfalign`): the source calls
        // `exitFromImodError()` without the program name it requires.
        exit_from_imod_error("xfalign");
    }
    newname
}

/// `def makeTiltxcorrCom(outputFile, warping)` (`xfalign:35`): make or start
/// a command list for running tiltxcorr.
fn make_tiltxcorr_com(g: &Globals, output_file: &str, warping: i32) -> Vec<String> {
    let mut tiltcom = vec![
        format!("OutputFile {output_file}"),
        format!("FilterSigma1 {}", g.xcfilter[0].text),
        format!("FilterSigma2 {}", g.xcfilter[1].text),
        format!("FilterRadius1 {}", g.xcfilter[2].text),
        format!("FilterRadius2 {}", g.xcfilter[3].text),
    ];
    let mut binning: i64 = 0;
    if warping > 1 {
        tiltcom.push(format!("InputFile {}", g.tmp_raw_ali));
    } else {
        tiltcom.push(format!("InputFile {}", g.infile));
    }
    let size_for_bin: i64;
    if warping != 0 {
        tiltcom.extend([
            format!("SizeOfPatchesXandY {},{}", g.warp_patch_x, g.warp_patch_y),
            "FindWarp 1".to_owned(),
        ]);
        if !g.bound_model.is_empty() {
            tiltcom.push(format!("BoundaryModel {}", g.bound_model));
        }
        if !g.seed_model.is_empty() {
            tiltcom.push(format!("SeedModel {}", g.seed_model));
        }
        if g.warp_limits_x != 0 || g.warp_limits_y != 0 {
            tiltcom.push(format!(
                "ShiftLimitsXandY {},{}",
                g.warp_limits_x, g.warp_limits_y
            ));
        }

        // Set binning if specified, provide size to compute it from
        if g.warp_binning > 0 {
            binning = g.warp_binning as i64;
        }
        size_for_bin = g.warp_patch_x.min(g.warp_patch_y) as i64;
    } else {
        tiltcom.extend([
            "RotationAngle 0".to_owned(),
            "FirstTiltAngle 0.".to_owned(),
            "TiltIncrement 0.".to_owned(),
        ]);
        if g.limits[0].value > 0. && g.limits[1].value > 0. {
            tiltcom.push(format!(
                "ShiftLimitsXandY {},{}",
                py_round(g.limits[0].value) as i64,
                py_round(g.limits[1].value) as i64
            ));
        }

        // Set binning if specified, provide size to compute it from
        if g.pre_binning > 0 {
            binning = g.pre_binning as i64;
        }
        size_for_bin = g.newsize[0].min(g.newsize[1]);
    }

    if g.ifsize != 0 {
        tiltcom.extend([
            format!("XMinAndMax {},{}", g.xmin, g.xmax),
            format!("YMinAndMax {},{}", g.ymin, g.ymax),
        ]);
    }

    // Add the specified binning or predict the default
    if binning > 0 {
        tiltcom.push(format!("BinningToApply {binning}"));
    } else {
        binning = 1.max((size_for_bin.div_euclid(1250) as f64 + 0.99) as i64);
    }

    // Turn on antialiasing for binning >= 4
    if binning >= 4 {
        tiltcom.push("AntialiasFilter 5".to_owned());
    }

    // Add the skip or break list, adding 1 to get from sections to views
    if (!g.skip_list.is_empty() || !g.break_list.is_empty()) && warping < 2 {
        let (nlist, mut opt) = if !g.skip_list.is_empty() {
            (&g.skip_list, "SkipViews".to_owned())
        } else {
            (&g.break_list, "BreakAtViews".to_owned())
        };
        for num in nlist {
            opt += &format!(" {}", num + 1);
        }
        tiltcom.push(opt);
    }

    tiltcom
}

/// The script's top level (`xfalign:102-535`).  Returns the status of its
/// `sys.exit`; error paths exit the process as `exitError` does.
pub fn xfalign(arguments: &[OsString]) -> i32 {
    let progname = "xfalign";
    let prefix = format!("ERROR: {progname} - ");
    let argv: Vec<String> = arguments
        .iter()
        .map(|argument| argument.to_string_lossy().into_owned())
        .collect();
    let done = |status: i32| {
        let _ = std::io::stdout().flush();
        status
    };

    //
    // Setup runtime environment
    if std::env::var_os("IMOD_DIR").is_some() {
        add_imod_bin_ignore_sighup();
    } else {
        print!("{prefix} IMOD_DIR is not defined!\n");
        return done(1);
    }

    // Initializations
    let tmpdir = imod_temp_dir();
    let mut g = Globals {
        tmpxf: format!("{tmpdir}/{progname}.xf.{}", std::process::id()),
        tmp_raw_ali: format!("{tmpdir}/{progname}.st.{}", std::process::id()),
        infile: String::new(),
        // Default in adoc
        xcfilter: vec![
            PyNumber::float(0.01),
            PyNumber::float(0.05),
            PyNumber::int(0),
            PyNumber::float(0.25),
        ],
        warp_patch_x: 0,
        warp_patch_y: 0,
        bound_model: String::new(),
        seed_model: String::new(),
        warp_limits_x: 0,
        warp_limits_y: 0,
        warp_binning: 0,
        // Default in adoc
        limits: vec![
            PyNumber::int(0),
            PyNumber::int(0),
            PyNumber::int(0),
            PyNumber::int(0),
            PyNumber::float(0.1),
            PyNumber::int(4),
        ],
        pre_binning: 0,
        newsize: [0, 0],
        ifsize: 0,
        xmin: 0,
        xmax: 0,
        ymin: 0,
        ymax: 0,
        skip_list: Vec::new(),
        break_list: Vec::new(),
    };

    // Fallbacks from ../manpages/autodoc2man 3 1 xfalign
    let options: Vec<String> = [
        ":InputImageFile:FN:",
        ":OutputTransformFile:FN:",
        "size:SizeToAnalyze:IP:",
        "offset:OffsetToSubarea:IP:",
        "matt:EdgeToIgnore:F:",
        "reduce:ReduceByBinning:I:",
        "filter:FilterParameters:FA:",
        "sobel:SobelFilter:B:",
        "params:ParametersToSearch:I:",
        "limits:LimitsOnSearch:FA:",
        "bilinear:BilinearInterpolation:B:",
        "ccc:CorrelationCoefficient:B:",
        "local:LocalPatchSize:I:",
        "reference:ReferenceFile:FN:",
        "prexcorr:PreCrossCorrelation:B:",
        "xcfilter:XcorrFilter:FA:",
        "xcreduce:XcorrReduction:I:",
        "initial:InitialTransforms:FN:",
        "warp:WarpPatchSizeXandY:IP:",
        "boundary:BoundaryModel:FN:",
        "seed:WarpSeedModel:FN:",
        "shift:ShiftLimitsForWarp:IP:",
        "wreduce:WarpReduction:I:",
        "skip:SkipSections:LI:",
        "break:BreakAtSections:LI:",
        "bpair:PairedImages:B:",
        "tomo:TomogramAverages:B:",
        "diff:DifferenceOutput:B:",
        "one:SectionsNumberedFromOne:B:",
        ":PID:B:",
        "help:usage:B:",
    ]
    .iter()
    .map(|option| (*option).to_owned())
    .collect();

    let (_opts, _nonopts) = pip_read_or_parse_options(&argv, &options, progname, 2, 1, 1);
    pass_on_key_interrupt(true);

    let do_pid = pip_get_boolean("PID", 0).unwrap_or(0);
    print_pid(do_pid != 0);

    g.infile = pip_get_in_out_file("InputImageFile", 0)
        .ok()
        .flatten()
        .unwrap_or_default();
    if g.infile.is_empty() {
        exit_error("Input image file must be entered");
    }
    let mut xflistfile = pip_get_in_out_file("OutputTransformFile", 1)
        .ok()
        .flatten()
        .unwrap_or_default();
    if xflistfile.is_empty() {
        exit_error("Output file for transforms must be entered");
    }

    let newsize_in = pip_get_two_integers("SizeToAnalyze", (0, 0)).unwrap_or((0, 0));
    g.newsize = [newsize_in.0 as i64, newsize_in.1 as i64];
    g.ifsize = 1 - pip_get_err_no();
    let newcen = pip_get_two_integers("OffsetToSubarea", (0, 0)).unwrap_or((0, 0));
    let filterparam = pip_get_float_array("FilterParameters", 4);
    // Default in adoc
    let nreduce = pip_get_integer("ReduceByBinning", 2).unwrap_or(2);
    let mut ifskip = pip_get_string("SkipSections", "").unwrap_or_default();
    if !ifskip.is_empty() {
        g.skip_list = parse_list(&ifskip).unwrap_or_default();
        if g.skip_list.is_empty() {
            exit_error("Parsing the skip list");
        }
    }
    let mut ifbreak = pip_get_string("BreakAtSections", "").unwrap_or_default();
    if !ifbreak.is_empty() {
        g.break_list = parse_list(&ifbreak).unwrap_or_default();
        if g.break_list.is_empty() {
            exit_error("Parsing the break list");
        }
    }

    let ifpair = pip_get_boolean("PairedImages", 0).unwrap_or(0);
    // Default in adoc
    let fracmatt = pip_get_float("EdgeToIgnore", 0.05).unwrap_or(0.05);
    // Default in adoc
    let nparam = pip_get_integer("ParametersToSearch", 0).unwrap_or(0);
    if nparam != 0 && (nparam < -1 || (nparam > 0 && (nparam + 3).div_euclid(5) != 1)) {
        exit_error("Number of parameters to search must be -1, 0, or 2-6");
    }
    let ifbilinear = pip_get_boolean("BilinearInterpolation", 0).unwrap_or(0);
    let mut reffile = pip_get_string("ReferenceFile", "").unwrap_or_default();
    let prexcorr = pip_get_boolean("PreCrossCorrelation", 0).unwrap_or(0);
    (g.warp_patch_x, g.warp_patch_y) =
        pip_get_two_integers("WarpPatchSizeXandY", (0, 0)).unwrap_or((0, 0));
    g.bound_model = pip_get_string("BoundaryModel", "").unwrap_or_default();
    (g.warp_limits_x, g.warp_limits_y) =
        pip_get_two_integers("ShiftLimitsForWarp", (0, 0)).unwrap_or((0, 0));
    if (g.warp_patch_x != 0 && g.warp_patch_y <= 0) || (g.warp_patch_y != 0 && g.warp_patch_x <= 0)
    {
        exit_error("Patch sizes must be positive in both X and Y");
    }
    if nparam == -1 && prexcorr == 0 && g.warp_patch_x == 0 {
        exit_error("-param entry can be -1 only when doing initial cross-correlation or warping");
    }
    if let Some(xcfin) = pip_get_float_array("XcorrFilter", 4).filter(|v| !v.is_empty()) {
        g.xcfilter = xcfin.into_iter().map(PyNumber::float).collect();
    }
    let mut prexffile = pip_get_string("InitialTransforms", "").unwrap_or_default();
    let iftomo = pip_get_boolean("TomogramAverages", 0).unwrap_or(0);
    let diffout = pip_get_boolean("DifferenceOutput", 0).unwrap_or(0);
    let do_ccc = pip_get_boolean("CorrelationCoefficient", 0).unwrap_or(0);
    let sobel = pip_get_boolean("SobelFilter", 0).unwrap_or(0);
    let local = pip_get_integer("LocalPatchSize", 0).unwrap_or(0);
    let from_one = pip_get_boolean("SectionsNumberedFromOne", 0).unwrap_or(0);
    if let Some(limits_in) = pip_get_float_array("LimitsOnSearch", 0).filter(|v| !v.is_empty()) {
        for i in 0..6.min(limits_in.len()) {
            g.limits[i] = PyNumber::float(limits_in[i]);
        }
    }

    g.seed_model = pip_get_string("WarpSeedModel", "").unwrap_or_default();
    g.warp_binning = pip_get_integer("WarpReduction", 0).unwrap_or(0);
    g.pre_binning = pip_get_integer("XcorrReduction", 0).unwrap_or(0);

    // Error checks
    if prexcorr != 0 && !reffile.is_empty() {
        exit_error("You cannot use initial cross-correlation with alignment to one reference");
    }
    if prexcorr != 0 && !prexffile.is_empty() {
        exit_error("You cannot use initial cross-correlation with initial transforms");
    }
    if !g.skip_list.is_empty() && !g.break_list.is_empty() {
        exit_error("You cannot use both a break list and skip list");
    }
    if !g.break_list.is_empty() && !reffile.is_empty() {
        exit_error("It is meaningless to use a break list with alignment to a reference");
    }
    if g.warp_patch_x != 0 && !reffile.is_empty() {
        exit_error("You cannot do warping alignment to a reference");
    }
    if !Path::new(&g.infile).exists() {
        exit_error(&format!("Input image file {} does not exist", g.infile));
    }
    if !prexffile.is_empty() && !Path::new(&prexffile).exists() {
        exit_error(&format!(
            "Initial transform file {prexffile} does not exist"
        ));
    }
    if !reffile.is_empty() && !Path::new(&reffile).exists() {
        exit_error(&format!("Reference image file {reffile} does not exist"));
    }

    // Check if initial file is warping
    if !prexffile.is_empty() {
        let pre_lines = read_text_file(&prexffile, None, false, Some(1)).unwrap_or_default();
        if !pre_lines.is_empty() && pre_lines[0].split_whitespace().count() == 1 {
            exit_error("You cannot use warping transforms as initial transforms");
        }
    }

    let (nx, ny, numsec, mode) = match get_mrc(&g.infile, false, false) {
        Ok(MrcInfo::Basic(x, y, z, m, ..)) => (x as i64, y as i64, z as i64, m),
        _ => exit_from_imod_error(progname),
    };
    let mut refmode = 0;
    if !reffile.is_empty() {
        match get_mrc(&reffile, false, false) {
            Ok(MrcInfo::Basic(_, _, _, m, ..)) => refmode = m,
            _ => exit_from_imod_error(progname),
        }
    }

    // If sections numbered from 1, shift the entered lists down
    if from_one != 0 && !g.skip_list.is_empty() {
        for value in g.skip_list.iter_mut() {
            *value -= 1;
        }
    }
    if from_one != 0 && !g.break_list.is_empty() {
        // Fixed in translation (BUGS.md, `xfalign`): the source decrements
        // `skipList[i]` here, an IndexError (the two lists exclude each
        // other) that leaves the break list unshifted.
        for value in g.break_list.iter_mut() {
            *value -= 1;
        }
    }

    // If serial tomograms or paired images, check other options
    if iftomo != 0 || ifpair != 0 {
        if !ifskip.is_empty() || !ifbreak.is_empty() {
            exit_error("You cannot enter a break or a skip list in tomogram or pair mode");
        }
        if !reffile.is_empty() {
            exit_error("You cannot use alignment to one reference with tomograms or pairs");
        }

        // set up break list as even sections
        g.break_list = (2..numsec).step_by(2).map(|sec| sec as i32).collect();
        ifbreak = "True".to_owned();
    }

    // Given new size and center, compute min/max of ranges
    if g.ifsize != 0 {
        g.xmin = (nx - g.newsize[0]).div_euclid(2) + newcen.0 as i64;
        g.xmax = (nx + g.newsize[0]).div_euclid(2) + newcen.0 as i64 - 1;
        g.ymin = (ny - g.newsize[1]).div_euclid(2) + newcen.1 as i64;
        g.ymax = (ny + g.newsize[1]).div_euclid(2) + newcen.1 as i64 - 1;
        if g.xmin < 0
            || g.xmax >= nx
            || g.xmin >= g.xmax
            || g.ymin < 0
            || g.ymax >= ny
            || g.ymin >= g.ymax
        {
            exit_error("New center or subarea offset gives an area outside range of image");
        }
    } else {
        g.newsize = [nx, ny];
    }

    // Check matt entry
    let mut xtrim = fracmatt as i64;
    let mut ytrim = xtrim;
    if fracmatt < 1. {
        xtrim = (g.newsize[0] as f64 * fracmatt) as i64;
        ytrim = (g.newsize[1] as f64 * fracmatt) as i64;
    }
    if g.newsize[0] - 2 * xtrim < 10 || g.newsize[1] - 2 * ytrim < 10 {
        exit_error("Entry for new size or for edge to ignore leaves too small an area");
    }

    if mode == 16 {
        g.infile = make_gray_file(&g.infile);
    }
    if !reffile.is_empty() && refmode == 16 {
        reffile = make_gray_file(&reffile);
    }

    // Warp patch tracking can be run on the whole stack if there is no prealignment, no
    // initial transforms, and no search
    if g.warp_patch_x != 0 && nparam == -1 && prexcorr == 0 && prexffile.is_empty() {
        let tiltcom = make_tiltxcorr_com(&g, &xflistfile, 1);
        prnstr(
            "RUNNING TILTXCORR WITH PATCH TRACKING TO FIND ALL WARPING ALIGNMENTS...",
            "\n",
            true,
        );

        if run_cmd("tiltxcorr -StandardInput", Some(&tiltcom), None, None, &[]).is_err() {
            exit_from_imod_error(progname);
        }

        prnstr("DONE", "\n", false);
        return done(0);
    }

    // If doing warps afterwards, adjust the output filename for the preliminaries
    let (preroot, _ext) = super::imodpy::os_path_splitext(&xflistfile);
    let mut warp_out_file = String::new();
    if g.warp_patch_x != 0 {
        warp_out_file = xflistfile.clone();
        xflistfile = format!("{preroot}.linxf");
    }

    // Set up initial cross-correlation
    if prexcorr != 0 {
        prexffile = format!("{preroot}.xcxf");
        let tiltcom = make_tiltxcorr_com(&g, &prexffile, 0);
        prnstr(
            "RUNNING TILTXCORR FOR INITIAL CROSS-CORRELATION ALIGNMENTS...",
            "\n",
            true,
        );

        if run_cmd("tiltxcorr -StandardInput", Some(&tiltcom), None, None, &[]).is_err() {
            exit_from_imod_error(progname);
        }

        let mut prexflist = read_text_file(&prexffile, None, false, None).unwrap_or_default();

        // trim the transform list for tomos
        if iftomo != 0 {
            let mut idel = 2;
            while idel < prexflist.len() {
                prexflist.remove(idel);
                idel += 1;
            }
            let _ = write_text_file(&prexffile, &prexflist, false);
        }

        prnstr("X, Y SHIFTS FOUND:", "\n", false);
        for (ind, line) in prexflist.iter().enumerate() {
            if ifpair == 0 || ind % 2 != 0 {
                let lsplit: Vec<&str> = line.split_whitespace().collect();
                match (
                    lsplit.get(4).and_then(|text| py_float(text)),
                    lsplit.get(5).and_then(|text| py_float(text)),
                ) {
                    (Some(dx), Some(dy)) => prnstr(&format!("{dx:9.2}  {dy:9.2}"), "\n", false),
                    _ => exit_error(&format!("Getting shifts from {prexffile}")),
                }
            }
        }
        let _ = std::io::stdout().flush();

        if nparam < 0 {
            let _ = write_text_file(&xflistfile, &prexflist, false);
            if g.warp_patch_x == 0 {
                return done(0);
            }
        }
    }

    // Loop on all the sections in file
    let mut next_ref: i64 = -1;
    let mut xfout_list: Vec<String> = Vec::new();
    if nparam >= 0 {
        prnstr("TRANSFORMS FOUND BY XFSIMPLEX:", "\n", false);
    }
    let in_skip = |sec: i64| g.skip_list.iter().any(|&value| value as i64 == sec);
    let in_break = |sec: i64| g.break_list.iter().any(|&value| value as i64 == sec);
    for sec in 0..numsec {
        if nparam < 0 {
            continue;
        }

        let mut fileref = g.infile.clone();
        let mut refsec = next_ref;
        if !reffile.is_empty() {
            refsec = 0;
            fileref = reffile.clone();
        }

        let mut doskip = false;
        let mut dobreak = false;
        if (!ifskip.is_empty() && in_skip(sec)) || (!ifbreak.is_empty() && in_break(sec)) {
            doskip = !ifskip.is_empty();
            dobreak = !ifbreak.is_empty();
            if !ifbreak.is_empty() {
                next_ref = sec;
            }
        } else {
            next_ref = sec;
        }

        if doskip || dobreak || refsec < 0 {
            if sec == 0 || iftomo == 0 {
                xfout_list.push(
                    "   1.0000000   0.0000000   0.0000000   1.0000000       0.000       0.000"
                        .to_owned(),
                );
            }
        } else {
            let mut simpcom = vec![
                format!("AImageFile {fileref}"),
                format!("BImageFile {}", g.infile),
                format!("OutputFile {}", g.tmpxf),
                format!("SectionsToUse {refsec},{sec}"),
                format!("VariablesToSearch {nparam}"),
                format!("BinningToApply {nreduce}"),
                format!("LinearInterpolation {ifbilinear}"),
                format!("EdgeToIgnore {}", py_str_float(fracmatt)),
                format!("CorrelationCoefficient {do_ccc}"),
                format!("SobelFilter {sobel}"),
                "FloatOption 1".to_owned(),
                "AntialiasFilter -1".to_owned(),
            ];

            if local != 0 {
                simpcom.push(format!("LocalPatchSize {local}"));
            }

            // Set up initial transform
            if !prexffile.is_empty() {
                let mut presec = sec;
                if iftomo != 0 {
                    presec = (sec + 1).div_euclid(2);
                }
                simpcom.extend([
                    format!("InitialTransformFile {prexffile}"),
                    format!("UseTransformLine {presec}"),
                ]);
            }

            // Add filter
            if let Some(filter) = filterparam.as_ref().filter(|v| !v.is_empty()) {
                let item = |index: usize| -> String {
                    match filter.get(index) {
                        Some(value) => py_str_float(*value),
                        None => python_uncaught("IndexError: list index out of range"),
                    }
                };
                simpcom.extend([
                    format!("FilterSigma1 {}", item(0)),
                    format!("FilterSigma2 {}", item(1)),
                    format!("FilterRadius1 {}", item(2)),
                    format!("FilterRadius2 {}", item(3)),
                ]);
            }

            // Handle subarea
            if g.ifsize != 0 {
                simpcom.extend([
                    format!("XMinAndMax {},{}", g.xmin + xtrim, g.xmax - xtrim),
                    format!("YMinAndMax {},{}", g.ymin + ytrim, g.ymax - ytrim),
                ]);
            }

            if nparam != 0 {
                simpcom.push(format!(
                    "LimitsOnSearch {},{},{},{},{},{}",
                    g.limits[0].text,
                    g.limits[1].text,
                    g.limits[2].text,
                    g.limits[3].text,
                    g.limits[4].text,
                    g.limits[5].text
                ));
            } else {
                simpcom.push(format!(
                    "LimitsOnSearch {},{}",
                    g.limits[0].text, g.limits[1].text
                ));
            }

            let simplines =
                match run_cmd("xfsimplex -StandardInput", Some(&simpcom), None, None, &[]) {
                    Ok(lines) => lines.unwrap_or_default(),
                    Err(_) => {
                        cleanup(&g);
                        exit_from_imod_error(progname);
                    }
                };

            if diffout != 0 {
                for i in 0..simplines.len() {
                    if simplines[i].contains("FINAL VALUES") && i < simplines.len() - 1 {
                        let lsplit: Vec<&str> = simplines[i + 1].split_whitespace().collect();
                        if lsplit.len() > 1 {
                            let mut label = "Difference:  ";
                            if do_ccc != 0 {
                                label = "CCC:  ";
                            }
                            prnstr(&format!("{label}{}", lsplit[1]), "\n", false);
                        }
                    }
                }
            }

            let xfone = read_text_file(&g.tmpxf, None, false, None).unwrap_or_default();
            let Some(first) = xfone.first() else {
                python_uncaught("IndexError: list index out of range")
            };
            prnstr(&format!("{sec} : {first}"), "\n", true);
            xfout_list.push(first.clone());
        }
    }

    if nparam >= 0 {
        make_backup_file(&xflistfile);
        let _ = write_text_file(&xflistfile, &xfout_list, false);
        cleanup(&g);
    }

    if g.warp_patch_x == 0 {
        return done(0);
    }

    let mut next_ref: i64 = -1;
    let mut any_done = false;

    // Set file for prealign transforms if any and get them in
    let mut xf_for_warp = prexffile.clone();
    if prexcorr != 0 || nparam >= 0 {
        xf_for_warp = xflistfile.clone();
    }

    prnstr(
        "RUNNING TILTXCORR TO FIND WARPING ALIGNMENTS ONE SECTION AT A TIME...",
        "\n",
        true,
    );

    // Loop on sections as above
    for sec in 0..numsec {
        let refsec = next_ref;
        let mut doskip = false;
        let mut dobreak = false;
        if (!ifskip.is_empty() && in_skip(sec)) || (!ifbreak.is_empty() && in_break(sec)) {
            doskip = !ifskip.is_empty();
            dobreak = !ifbreak.is_empty();
            if !ifbreak.is_empty() {
                next_ref = sec;
            }
        } else {
            next_ref = sec;
        }

        if doskip || dobreak || refsec < 0 {
            // If skipping or breaking and file has been started, append unit
            // transform and a leading 0 for no control points.  Both break and skip require
            // a unit transform here regardless of what is in the initial file
            if any_done {
                let mut action = "Opening";
                let appended: std::io::Result<()> = (|| {
                    let mut warpfile = std::fs::OpenOptions::new()
                        .append(true)
                        .create(true)
                        .open(&warp_out_file)?;
                    action = "Appending to";
                    warpfile.write_all(
                        b"0\n   1.0000000   0.0000000   0.0000000   1.0000000       0.000       0.000\n",
                    )?;
                    Ok(())
                })();
                if let Err(error) = appended {
                    exit_error(&format!("{action} warp transform file: {error}"));
                }
            }
        } else {
            // Newstack the pair of sections
            let mut newstcom = vec![
                format!("InputFile {}", g.infile),
                format!("OutputFile {}", g.tmp_raw_ali),
                format!("SectionsToRead {refsec},{sec}"),
            ];
            if !xf_for_warp.is_empty() {
                newstcom.extend([
                    format!("TransformFile {xf_for_warp}"),
                    format!("UseTransformLines 0,{sec}"),
                ]);
            }
            if run_cmd("newstack -StandardInput", Some(&newstcom), None, None, &[]).is_err() {
                cleanup(&g);
                exit_from_imod_error(progname);
            }

            // Make the tiltxcorr command
            let mut tiltcom = make_tiltxcorr_com(&g, &warp_out_file, 2);
            tiltcom.push(format!("RawAndAlignedPair {},{numsec}", sec + 1));
            if !xf_for_warp.is_empty() {
                tiltcom.push(format!("PrealignmentTransformFile {xf_for_warp}"));
            }
            if any_done {
                tiltcom.push("AppendToWarpFile".to_owned());
            }

            if run_cmd("tiltxcorr -StandardInput", Some(&tiltcom), None, None, &[]).is_err() {
                cleanup(&g);
                exit_from_imod_error(progname);
            }

            any_done = true;
            prnstr(".", "", true);
        }
    }

    prnstr("\nDONE", "\n", false);

    cleanup(&g);
    done(0)
}
