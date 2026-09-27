//! `ctfplotter` in batch mode: a **reimplementation**, not a translation.
//!
//! Owner decision, 2026-09-27: *"later I want ctfplotter reimplemented. but
//! the output need not be precisely the same. given that the code seems to
//! be a mess, happy to write completely new code that does the same thing,
//! using rust native libraries"*, later clarified that parts may be ported
//! verbatim where that gets results close to native faster.  So this module
//! is an explicit exception to the project's line-faithful rule: it is
//! structured freely, uses `rustfft` and `rayon`, and is validated against
//! native `ctfplotter` within stated defocus tolerances rather than byte
//! identity (`TODO.md`, "ctfplotter reimplementation").
//!
//! What it does, as native `ctfplotter -SaveAndExit` does for
//! `ctfplotter.com` and Batchruntomo's `ctfplotter_auto.com`:
//!
//! 1. divide each view into overlapping tiles, group them into strips of
//!    nearly constant defocus parallel to the tilt axis, and sum the tile
//!    power spectra per strip at 20x the final resolution
//!    ([`spectra`]);
//! 2. for a range of views, divide each strip spectrum by the noise
//!    spectrum for its intensity (from `ConfigFile` noise images, when
//!    given), stretch its frequency axis so its CTF zeros match those at the
//!    reference defocus, and average ([`spectrum`]);
//! 3. subtract a baseline from the log spectrum and fit a CTF-like curve by
//!    simplex search for defocus, optionally phase shift and cut-on
//!    ([`fit`]), and astigmatism from fits to 90-degree wedges of sectors
//!    ([`analyzer`]);
//! 4. autofit over the series in steps, iterating each range with the
//!    current defocus, optionally after a scan for the starting defocus; and
//!    write the `.defocus` file in the version 2 or 3 formats `ctfphaseflip`
//!    reads.
//!
//! Not implemented (a message says so when they are entered): the GUI and
//! everything only it does (PNG/TIFF snapshots), focal-pair processing,
//! `TestInversionAndExit`, `AssessTiltAngleOffset`, fitting to zeros for
//! phase (`FitZerosForPhase`), the cropping/resolution changes of
//! autotuning (only the fitting-range extension is done), and saving
//! `ctfplotter.info` settings.

pub mod analyzer;
pub mod ctf;
pub mod fit;
pub mod spectra;
pub mod spectrum;

use crate::imod::ctfplotter::ctfutils::{read_defocus_file, read_tilt_angles};
use crate::imod::libcfshr::b3dutil::{
    exit, imod_prog_name, program_args, standard_memory_limit_mb,
};
use crate::imod::libcfshr::parse_params::{
    exit_error, pip_done, pip_get_boolean, pip_get_float, pip_get_integer, pip_get_string,
    pip_get_three_integers, pip_get_two_floats, pip_get_two_integers, pip_print_help,
    pip_read_or_parse_options,
};
use crate::imod::libcfshr::parselist::parselist;
use crate::imod::libiimod::mrcfiles::{MrcHeader, mrc_get_scale, mrc_head_read};

use analyzer::{Analyzer, RunSettings, say, tile_params};
use ctf::{CtfModel, RADIANS_PER_DEGREE};
use spectra::{
    Binning, ImageStack, SpectrumCache, TileTransformer, compute_view_spectra, view_geometry,
};
use spectrum::NoiseModel;

const OPTIONS: [&[u8]; 65] = [
    b"input:InputStack:FN:",
    b"offset:OffsetToAdd:F:",
    b"config:ConfigFile:FN:",
    b"angleFn:AngleFile:FN:",
    b"single:SingleTiltAngle:F:",
    b"invert:InvertTiltAngles:B:",
    b"taOffset:TiltAngleOffset:F:",
    b"aAngle:AxisAngle:F:",
    b"defFn:DefocusFile:FN:",
    b"pixelSize:PixelSize:F:",
    b"crop:CropToPixelSize:F:",
    b"fitZeros:FitZerosForPhase:B:",
    b"volt:Voltage:I:",
    b"cs:SphericalAberration:F:",
    b"ampContrast:AmplitudeContrast:F:",
    b"degPhase:PhaseShiftInDegrees:F:",
    b"phase:PhasePlateShift:F:",
    b"phRange:PhaseRangeToSearch:F:",
    b"cuton:CutOnFrequency:F:",
    b"maxCuton:MaxCutOnToSearch:F:",
    b"skip:ViewsToSkip:LI:",
    b"bidir:BidirectionalNumViews:I:",
    b"onlyAP:SkipOnlyForAstigPhase:B:",
    b"expDef:ExpectedDefocus:F:",
    b"scan:ScanDefocusRange:FP:",
    b"tune:TuneFittingAndSampling:B:",
    b"assess:AssessTiltAngleOffset:B:",
    b"testInv:TestInversionAndExit:B:",
    b"fpOffset:FocalPairDefocusOffset:F:",
    b"range:AngleRange:FP:",
    b"autoFit:AutoFitRangeAndStep:FP:",
    b"more:FitMoreViewsAboveAngle:IP:",
    b"useDef:UseExpectedDefForAuto:B:",
    b"frequency:FrequencyRangeToFit:FP:",
    b"extra:ExtraZerosToFit:F:",
    b"truncate:AutoTruncateAndWeight:I:",
    b"vary:VaryExponentInFit:B:",
    b"baseline:BaselineFittingOrder:I:",
    b"find:FindAstigPhaseCuton:IT:",
    b"sAstig:SearchAstigmatism:B:",
    b"sPhase:SearchPhaseShift:B:",
    b"sCuton:SearchCutonFrequency:B:",
    b"minViews:MinViewsAstigAndPhase:IP:",
    b"save:SaveAndExit:B:",
    b"png:SavePNGsOfGraphs:I:",
    b"tiff:SaveTIFFsOfGraphs:I:",
    b"snap:SnapshotFilename:FN:",
    b"psRes:PSResolution:I:",
    b"tileSize:TileSize:I:",
    b"defTol:DefocusTol:I:",
    b"leftTol:LeftDefTol:F:",
    b"rightTol:RightDefTol:F:",
    b"cache:MaxCacheSize :I:",
    b"hyper:HyperResolutionFactor:I:",
    b"sectors:NumberOfSectors:I:",
    b"wedge:WedgeRangeAndInterval:FP:",
    b"astigMax:MaximumAstigmatism:F:",
    b"ignore:IgnoreInfoFile:B:",
    b"hideTest:HideAngleTestAndOffset:B:",
    b"colors:ChangeColors:B:",
    b"legacy:ShowLegacyFitting:B:",
    b"dump:DumpWedgeSpectra:LI:",
    b"debug:DebugLevel:I:",
    b"param:ParameterFile:PF:",
    b"help:usage:B:",
];

fn string_opt(name: &[u8]) -> Option<String> {
    let mut v = Vec::new();
    (pip_get_string(name, &mut v) == 0).then(|| String::from_utf8_lossy(&v).into_owned())
}

fn float_opt(name: &[u8]) -> Option<f32> {
    let mut v = 0f32;
    (pip_get_float(name, &mut v) == 0).then_some(v)
}

fn int_opt(name: &[u8]) -> Option<i32> {
    let mut v = 0i32;
    (pip_get_integer(name, &mut v) == 0).then_some(v)
}

fn bool_opt(name: &[u8]) -> Option<bool> {
    let mut v = 0i32;
    (pip_get_boolean(name, &mut v) == 0).then_some(v != 0)
}

fn two_floats(name: &[u8]) -> Option<(f32, f32)> {
    let (mut a, mut b) = (0f32, 0f32);
    (pip_get_two_floats(name, &mut a, &mut b) == 0).then_some((a, b))
}

fn two_ints(name: &[u8]) -> Option<(i32, i32)> {
    let (mut a, mut b) = (0i32, 0i32);
    (pip_get_two_integers(name, &mut a, &mut b) == 0).then_some((a, b))
}

fn fail(msg: &str) -> ! {
    exit_error(msg.as_bytes())
}

/// The `ctfplotter` command.
pub fn ctfplotter() {
    let args = program_args();
    let progname = imod_prog_name(args.first().map_or("ctfplotter", String::as_str));
    let argv: Vec<Vec<u8>> = args.iter().map(|a| a.as_bytes().to_vec()).collect();
    let (mut nopt, mut nnon) = (0, 0);
    pip_read_or_parse_options(
        argv.len() as i32,
        &argv,
        &OPTIONS,
        OPTIONS.len() as i32,
        progname.as_bytes(),
        1,
        0,
        0,
        &mut nopt,
        &mut nnon,
        None,
    );
    if bool_opt(b"usage") == Some(true) {
        pip_print_help(progname.as_bytes(), 0, 0, 0);
        exit(0);
    }
    if let Err(e) = run() {
        fail(&e);
    }
    exit(0);
}

fn run() -> Result<(), String> {
    let debug = int_opt(b"DebugLevel").unwrap_or(1);
    let hyper = int_opt(b"HyperResolutionFactor").unwrap_or(20);
    if !(2..=40).contains(&hyper) {
        return Err("Hyperresolution factor must be between 2 and 40".into());
    }
    let num_sectors = int_opt(b"NumberOfSectors").unwrap_or(36);
    if !(12..=45).contains(&num_sectors) {
        return Err("Number of sectors must be between 12 and 45".into());
    }
    let (wedge_range, wedge_interval_in) =
        two_floats(b"WedgeRangeAndInterval").unwrap_or((90.0, 0.0));
    if wedge_range > 120.0 {
        return Err("Wedge angle range must be no more than 120 degrees".into());
    }
    let config = string_opt(b"ConfigFile");
    let stack_fn = string_opt(b"InputStack").ok_or("No stack specified")?;
    let single = float_opt(b"SingleTiltAngle");
    let angle_fn = string_opt(b"AngleFile");
    if angle_fn.is_some() && single.is_some() {
        return Err("You cannot enter both -single and -angleFn".into());
    }
    let single_angle = single.unwrap_or(0.0) as f64;
    let def_fn = string_opt(b"DefocusFile").ok_or("output defocus file is not specified ")?;
    let cache_mb = int_opt(b"MaxCacheSize")
        .map(|v| v as f64)
        .unwrap_or_else(|| {
            let v = standard_memory_limit_mb(30000);
            if v > 0.0 { v } else { 1000.0 }
        });
    let volt = int_opt(b"Voltage").ok_or("Voltage is not specified")?;
    let cs = float_opt(b"SphericalAberration")
        .ok_or("Spherical Aberration is not specified")?
        .max(0.01) as f64;
    let amp = float_opt(b"AmplitudeContrast").unwrap_or(0.07) as f64;
    let dim = int_opt(b"PSResolution").unwrap_or(101).clamp(101, 501) as usize;
    let tile = int_opt(b"TileSize").unwrap_or(256) as usize;
    if tile < 32 || tile % 2 != 0 {
        return Err("Tile size must be even and at least 32".into());
    }

    let mut stack = ImageStack::open(&stack_fn)?;
    let pixel = match float_opt(b"PixelSize") {
        Some(p) => p as f64,
        None => {
            let (px, py, _) = mrc_get_scale(&stack.header);
            if (px - py).abs() > 1.0e-3 {
                return Err("No PixelSize was specified and the stack file header pixel size differs between X and Y".into());
            }
            let p = px as f64 / 10.0;
            say(&format!(
                "No PixelSize was specified: using size {:.3} nm from stack file header\n",
                p
            ));
            p
        }
    };

    let rad_phase = float_opt(b"PhasePlateShift");
    let mut phase_shift = rad_phase.unwrap_or(0.0) as f64;
    if let Some(deg) = float_opt(b"PhaseShiftInDegrees") {
        if rad_phase.is_some() {
            return Err("You cannot enter phase shift in both degrees and radians".into());
        }
        phase_shift = deg as f64 * RADIANS_PER_DEGREE;
    }
    let phase_range = float_opt(b"PhaseRangeToSearch").unwrap_or(120.0) as f64;
    let cut_on = float_opt(b"CutOnFrequency").unwrap_or(0.0) as f64;
    let max_cuton_in = float_opt(b"MaxCutOnToSearch").unwrap_or(0.0) as f64;
    let max_astig = float_opt(b"MaximumAstigmatism").unwrap_or(1.2) as f64;
    let num_bidir = int_opt(b"BidirectionalNumViews").unwrap_or(0);
    let axis = float_opt(b"AxisAngle");
    if axis.is_none() && (angle_fn.is_some() || single_angle != 0.0) {
        return Err(
            "No AxisAngle specified; it is required with an angle file or non-zero angle entry"
                .into(),
        );
    }
    let axis_angle = axis.unwrap_or(0.0) as f64;
    let tilt_offset = float_opt(b"TiltAngleOffset").unwrap_or(0.0) as f64;
    let mut scan = two_floats(b"ScanDefocusRange").map(|(a, b)| (a.min(b) as f64, a.max(b) as f64));
    if let Some((s, _)) = scan {
        if s < 100.0 {
            return Err("Starting defocus to scan must be at least 100 nm".into());
        }
    }
    let exp_def = match float_opt(b"ExpectedDefocus") {
        Some(v) => v as f64,
        None => match scan {
            Some((a, b)) => 0.5 * (a + b),
            None => return Err("Neither expected defocus nor a range to scan is specified".into()),
        },
    };
    scan = scan.map(|(a, b)| (a / 1000.0, b / 1000.0));
    let weighting = int_opt(b"AutoTruncateAndWeight").unwrap_or(0);
    let tune = bool_opt(b"TuneFittingAndSampling").unwrap_or(false);
    if bool_opt(b"AssessTiltAngleOffset") == Some(true) {
        say(
            "WARNING: AssessTiltAngleOffset is not implemented in this reimplementation; skipping it\n",
        );
    }
    let def_tol = float_opt(b"DefocusTol").unwrap_or(50.0) as f64;
    let left_tol = float_opt(b"LeftDefTol").unwrap_or(2000.0) as f64;
    let right_tol = float_opt(b"RightDefTol").unwrap_or(2000.0) as f64;

    // Batch operation always autofits and saves (as the no-Qt build does)
    let (auto_range, auto_step) = match two_floats(b"AutoFitRangeAndStep") {
        Some((r, s)) => (r as f64, s as f64),
        None => (0.0, 0.0),
    };
    let angle_range = two_floats(b"AngleRange").map(|(a, b)| (a as f64, b as f64));
    let auto_with_expected = bool_opt(b"UseExpectedDefForAuto").unwrap_or(false);
    let more = two_ints(b"FitMoreViewsAboveAngle");
    let (mut find_astig, mut find_phase, mut find_cuton) = (false, false, false);
    let (mut fa, mut fp, mut fc) = (0, 0, 0);
    let find_entered =
        pip_get_three_integers(b"FindAstigPhaseCuton", &mut fa, &mut fp, &mut fc) == 0;
    if find_entered {
        find_astig = fa != 0;
        find_phase = fp != 0;
        find_cuton = fc != 0;
    }
    for (name, flag, what) in [
        (&b"SearchAstigmatism"[..], &mut find_astig, "-sAstig"),
        (&b"SearchPhaseShift"[..], &mut find_phase, "-sPhase"),
        (&b"SearchCutonFrequency"[..], &mut find_cuton, "-sCuton"),
    ] {
        if let Some(v) = bool_opt(name) {
            if find_entered {
                return Err(format!("You cannot enter -find with {what}"));
            }
            *flag = v;
        }
    }
    let (min_astig_views, min_phase_views) = two_ints(b"MinViewsAstigAndPhase").unwrap_or((5, 1));
    if bool_opt(b"FitZerosForPhase") == Some(true) {
        say(
            "WARNING: FitZerosForPhase is not implemented in this reimplementation; fitting to the spectrum instead\n",
        );
    }
    let skip_only = bool_opt(b"SkipOnlyForAstigPhase").unwrap_or(false);
    let skip_list = match string_opt(b"ViewsToSkip") {
        Some(s) => match parselist(&s) {
            Ok(l) if !l.is_empty() => l,
            _ => return Err("Parsing list of views to skip".into()),
        },
        None => Vec::new(),
    };
    let dump_final = match string_opt(b"DumpWedgeSpectra") {
        Some(d) => match parselist(&d) {
            Ok(l) if l == [-2] => true,
            Ok(l) if !l.is_empty() => {
                say(
                    "WARNING: Only -2 (final spectra of autofit ranges) is implemented for DumpWedgeSpectra\n",
                );
                false
            }
            _ => return Err("Parsing list of wedge spectra to dump".into()),
        },
        None => false,
    };
    let min_pixel = 0.105;
    let crop_entered = float_opt(b"CropToPixelSize");
    let mut crop_pixel = crop_entered.unwrap_or(0.0) as f64;
    if crop_pixel == 0.0 && pixel < min_pixel {
        crop_pixel = min_pixel;
    }
    let invert = bool_opt(b"InvertTiltAngles").unwrap_or(false);
    let offset_in = float_opt(b"OffsetToAdd");
    if float_opt(b"TestInversionAndExit").is_some()
        || bool_opt(b"TestInversionAndExit") == Some(true)
    {
        return Err(
            "TestInversionAndExit is not implemented in this reimplementation of ctfplotter".into(),
        );
    }
    let mut freq = two_floats(b"FrequencyRangeToFit").map(|(a, b)| (a as f64, b as f64));
    if freq.is_some_and(|(a, b)| a <= 0.0 && b <= 0.0) {
        freq = None;
    }
    let freq_in = freq;
    if let Some((mut a, mut b)) = freq {
        if a > 1.1 {
            a = 10.0 * pixel / a;
        }
        if b > 1.1 {
            b = 10.0 * pixel / b;
        }
        if (a >= 0.0 && a < 0.01) || b > 0.48 || (b >= 0.0 && b - a < 0.03) {
            return Err(
                "Fitting range values are too extreme, out of order, or too close together".into(),
            );
        }
        freq = Some((a, b));
    }
    if string_opt(b"SnapshotFilename").is_some()
        || int_opt(b"SavePNGsOfGraphs").is_some()
        || int_opt(b"SaveTIFFsOfGraphs").is_some()
    {
        say("WARNING: Graph snapshots are not produced by this reimplementation (no GUI)\n");
    }
    let vary = bool_opt(b"VaryExponentInFit").unwrap_or(false);
    if float_opt(b"FocalPairDefocusOffset").is_some() {
        return Err(
            "Focal pair processing is not implemented in this reimplementation of ctfplotter"
                .into(),
        );
    }
    let base_entered = int_opt(b"BaselineFittingOrder");
    let base_order = base_entered.unwrap_or(4).clamp(0, 4) as usize;
    let extra = float_opt(b"ExtraZerosToFit");
    if extra.is_some() && freq.is_some() {
        return Err("You cannot specify both a frequency range and extra zeros to fit".into());
    }
    let extra_zeros = extra.unwrap_or(1.5) as f64;
    if extra_zeros < -0.5 {
        return Err("Extra zeros to fit must be no less than -0.5".into());
    }
    pip_done();

    // Defocus model
    let mut model = CtfModel::new(volt, pixel, amp, cs, exp_def);
    model.plate_phase = phase_shift;
    model.cut_on = cut_on;
    let mut max_cuton = model.zero(exp_def / 1000.0, 1, None, None) / (2.0 * pixel);
    if max_cuton_in > 0.0 {
        max_cuton = max_cuton_in;
    }

    // Tilt angles
    let nz = stack.nz;
    let angles: Vec<f32> = match &angle_fn {
        Some(f) => {
            let (mut lo, mut hi) = (0f32, 0f32);
            read_tilt_angles(
                f.as_bytes(),
                nz as i32,
                if invert { -1.0 } else { 1.0 },
                &mut lo,
                &mut hi,
            )
        }
        None => {
            if single.is_none() && debug >= 1 {
                say("No angle file is specified, tilt angle is assumed to be 0.0\n");
            }
            (0..nz)
                .map(|i| (single_angle + i as f64 * 0.01) as f32)
                .collect()
        }
    };

    let sector = 180.0 / num_sectors as f64;
    let multiple = |v: f64| (v / sector + 0.4).floor() * sector;
    let wedge_interval = if wedge_interval_in > 0.0 {
        let w = multiple(wedge_interval_in as f64);
        if w == 0.0 {
            return Err(format!(
                "The wedge interval must be at least half as big as the sector width ({:.1} deg)",
                sector
            ));
        }
        w
    } else {
        sector
    };
    let wedge_range = sector.max(multiple(wedge_range as f64));

    let settings = RunSettings {
        dim,
        hyper: hyper as usize,
        num_sectors: num_sectors as usize,
        tile,
        pixel_size: pixel,
        def_tol,
        left_tol,
        right_tol,
        axis_angle,
        tilt_offset,
        base_order,
        two_line: config.is_none(),
        vary_power: vary,
        weighting,
        num_zeros_to_fit: 2.0 + extra_zeros,
        find_astig,
        find_phase,
        find_cuton,
        min_views_astig: (min_astig_views.clamp(1, nz as i32)) as usize,
        min_views_phase: (min_phase_views.clamp(1, nz as i32)) as usize,
        wedge_range,
        wedge_interval,
        max_astig,
        phase_range: phase_range * RADIANS_PER_DEGREE,
        max_cuton,
        skip_list,
        skip_only_astig_phase: skip_only,
        bidir_view: if num_bidir > 0 {
            num_bidir.clamp(2, nz as i32 - 1) as usize
        } else {
            0
        },
        fit_more_views: more.is_some(),
        num_more_views: more.map_or(3, |m| m.0.max(1) as usize),
        more_views_angle: more.map_or(45.0, |m| m.1 as f64),
        debug,
        defocus_file: def_fn.clone(),
        no_angles: angle_fn.is_none(),
        dump_final,
    };

    // Noise spectra
    let noise = match &config {
        Some(c) => Some(load_noise(c, &settings, &model)?),
        None => None,
    };

    // Data offset
    let header: &MrcHeader = &stack.header;
    let mut data_offset = offset_in.map(|v| v as f64);
    if data_offset.is_none() {
        let label0 = String::from_utf8_lossy(&header.labels[0]).into_owned();
        if label0.contains("Fei Company") {
            if header.amin < 0.0 {
                data_offset = Some(32768.0);
            } else {
                say(
                    "WARNING: Data Offset May Be Needed\nThe image stack originated from FEI software,\nbut the usual offset of 32768 is not being added\nbecause the minimum of the file is positive.\n",
                );
            }
        } else if header.amin < -20000.0 && header.amean < -1000.0 {
            data_offset = Some(32768.0);
            say(
                "WARNING: Data Offset Assumed\nThe image stack contains negative values\nunder -20000 and has a negative mean,\nso an offset of 32768 is being added.\n",
            );
        } else if header.amean < 0.0 {
            return Err("The mean of the input stack is negative.  You need to specify an offset to add to make values be proportional to recorded electrons".into());
        }
    }
    stack.data_offset = data_offset.unwrap_or(0.0) as f32;

    if model.exp_defocus <= 0.0 {
        return Err("Invalid expected defocus, it must be >0".into());
    }

    // Existing defocus file entries are kept outside the autofit range
    let saved = if std::path::Path::new(&def_fn).exists() {
        let (mut ver, mut flags) = (0, 0);
        read_defocus_file(def_fn.as_bytes(), &mut ver, &mut flags)
    } else {
        Vec::new()
    };

    let geom_angles = angle_fn.as_ref().map(|_| angles.clone());
    let params = tile_params(&settings, tile);
    let bin = Binning::new(dim, hyper as usize, num_sectors as usize, tile, tile);
    let cache = SpectrumCache::new(stack, params, bin, geom_angles, cache_mb);
    let mut an = Analyzer::new(settings, model, cache, noise, angles, saved);

    // Autofit setup: a zero step fits single views
    let auto_step_angle = (if angle_fn.is_some() { 0.1f64 } else { 0.002 }).max(auto_step);
    let auto_step_range = if auto_step == 0.0 { 0.0 } else { auto_range };
    an.low_angle = if angle_fn.is_some() {
        -3.6
    } else {
        single_angle
    };
    an.high_angle = if angle_fn.is_some() {
        3.6
    } else {
        single_angle
    };
    an.setup_ranges(auto_step_angle, auto_step_range);
    if let Some((a, b)) = angle_range {
        an.auto_from = a;
        an.auto_to = b;
    }
    if !auto_with_expected && crop_pixel <= pixel && scan.is_none() {
        an.use_cur_defocus = true;
        an.use_cur_phase = true;
        an.model.defocus = an.model.exp_defocus;
    }

    let (first_zero, ..) = an.fit_range_from_defocus(false);
    if first_zero > 0.49 && base_entered.is_none() {
        an.s.base_order = 1;
    }
    let low = freq.and_then(|(a, _)| (a > 0.0).then_some(a));
    let high = freq.and_then(|(_, b)| (b > 0.0).then_some(b));
    an.set_initial_fit_range(low, high);

    // Cropping
    if crop_pixel > pixel {
        let cp = crop_pixel.min(5.0 * min_pixel.max(pixel));
        an.crop = true;
        an.set_crop_pixel(cp);
        if crop_entered.is_none() {
            an.set_initial_fit_range(low, high);
        } else {
            let dimf = dim as f64 - 1.0;
            if let (Some(_), Some((a_in, _))) = (low, freq_in) {
                let a = if a_in > 1.1 { 10.0 * cp / a_in } else { a_in };
                an.x1 = (2.0 * a * dimf).round() as usize;
            }
            if let (Some(_), Some((_, b_in))) = (high, freq_in) {
                let b = if b_in > 1.1 { 10.0 * cp / b_in } else { b_in };
                an.x2 = ((2.0 * b * dimf).round() as usize).min(dim - 1);
            }
        }
    }

    // Initial spectrum and fit
    an.compute_ps(false)?;
    an.fit_spectrum();

    if scan.is_some() || tune {
        let (s0, s1) = scan.unwrap_or((0.0, 0.0));
        an.scan_defocus(
            s0,
            s1,
            tune,
            low.is_some(),
            high.is_some(),
            crop_entered.is_some(),
        )?;
    }
    if !((scan.is_some() || tune) && nz == 1) {
        if crop_pixel > pixel || scan.is_some() {
            if !auto_with_expected {
                an.use_cur_defocus = true;
                an.use_cur_phase = true;
                an.model.defocus = an.model.exp_defocus;
            }
            if !an.multiple_spectra_and_fits()? {
                an.compute_ps(false)?;
                an.fit_spectrum();
            }
        } else if find_astig || find_phase {
            an.multiple_spectra_and_fits()?;
        }
    }

    // Autofit
    let mut num_iter = 3;
    if single.is_some() && (single_angle + tilt_offset).abs() < 4.0 {
        num_iter = 2;
    }
    if single.is_some() && (single_angle + tilt_offset).abs() < 1.0 {
        num_iter = 1;
    }
    let (from, to, step) = (an.auto_from, an.auto_to, an.view_range_step);
    an.autofit(from, to, step, num_iter)?;
    an.write_defocus_file()?;
    Ok(())
}

/// Reads the noise images named in a configuration file (or a stack of
/// them) and measures their spectra.
fn load_noise(config: &str, s: &RunSettings, model: &CtfModel) -> Result<NoiseModel, String> {
    let mut files: Vec<(String, Option<usize>)> = Vec::new();
    if !std::path::Path::new(config).is_file() {
        return Err(format!(" could not open config file {config}"));
    }
    let is_stack = {
        let mut header = MrcHeader::default();
        crate::imod::libcfshr::b3dutil::b3d_set_store_error(1);
        let ok = match crate::imod::libiimod::iimage::ii_fopen(config.as_bytes(), "rb") {
            Some(mut fp) => mrc_head_read(&mut fp, &mut header) == 0 && header.nz > 0,
            None => false,
        };
        crate::imod::libcfshr::b3dutil::b3d_set_store_error(0);
        ok
    };
    if is_stack {
        let st = ImageStack::open(config)?;
        for z in 0..st.nz {
            files.push((config.to_string(), Some(z)));
        }
    } else {
        let text = std::fs::read_to_string(config)
            .map_err(|_| format!(" could not reopen config file {config}"))?;
        let dir = match config.replace('\\', "/").rfind('/') {
            Some(i) => config[..=i].to_string(),
            None => "./".to_string(),
        };
        for line in text.lines() {
            let l = line.trim_end_matches(['\r', '\n']).replace('\\', "/");
            if l.is_empty() {
                continue;
            }
            let path = if l.starts_with('/') {
                l
            } else {
                format!("{dir}{l}")
            };
            files.push((path, None));
        }
    }
    if files.len() < 2 {
        return Err("There must be at least two noise images".into());
    }
    let mut entries: Vec<(f64, Vec<f64>)> = Vec::new();
    let params = tile_params(s, s.tile);
    let bin = Binning::new(s.dim, s.hyper, s.num_sectors, s.tile, s.tile);
    let transformer = TileTransformer::new(s.tile);
    let mut open: Option<(String, ImageStack)> = None;
    for (i, (path, z)) in files.iter().enumerate() {
        if open.as_ref().is_none_or(|(p, _)| p != path) {
            open = Some((path.clone(), ImageStack::open(path)?));
        }
        let st = &mut open.as_mut().unwrap().1;
        let data = st.read_view(z.unwrap_or(0))?;
        let geom = view_geometry(st.nx, st.ny, None, &params);
        let spec = compute_view_spectra(&data, st.nx, &geom, &bin, &transformer, false);
        let (ps, mean) = Analyzer::noise_spectrum(&spec, &bin, s.dim, model);
        if mean < 0.0 {
            return Err(format!(
                "The mean of noise file {} is negative.  All noise files must have positive means, with 0 mean corresponding to 0 exposure",
                i + 1
            ));
        }
        if s.debug >= 1 {
            let mp: f64 = ps.iter().skip(1).sum::<f64>() / (s.dim as f64 - 1.0);
            say(&format!(
                "noiseMean[{}]={:.6}   mean power={}\n",
                i, mean, mp
            ));
        }
        entries.push((mean, ps));
    }
    entries.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    Ok(NoiseModel {
        dim: s.dim,
        means: entries.iter().map(|e| e.0).collect(),
        spectra: entries.into_iter().map(|e| e.1).collect(),
    })
}
