//! Translation of `IMOD/mrc/measuredrift.cpp`.
//!
//! Five source functions: `main`, `measureDrift`, `cleanupDrift`,
//! `interpolateCurve` and `areaFunc`.  `areaFunc` is the `dualAmoeba` error
//! callback, so the six file statics the C declares above it
//! (`sSpectrum`, `sFreqs`, `sAllSum`, `sNumFit`, `sNumRings`, `sMinFreq`,
//! `sMaxFreq`) stay file-level state here rather than becoming parameters.

use crate::imod::libcfshr::amoeba::dual_amoeba;
use crate::imod::libcfshr::b3dutil::{
    CArg, ImodFile, b3d_get_error, c_format_bytes, data_size_for_mode, imod_backup_file,
    imod_prog_name, imod_usage_header,
};
use crate::imod::libcfshr::circlefit::fit_centered_ellipse;
use crate::imod::libcfshr::filtxcorr::nice_frame;
use crate::imod::libcfshr::islice::slice_mode_if_real;
use crate::imod::libcfshr::parse_params::{
    exit_error, pip_get_float, pip_get_in_out_file, pip_get_integer, pip_get_string,
    pip_get_two_floats, pip_read_or_parse_options,
};
use crate::imod::libcfshr::parselist::parselist;
use crate::imod::libcfshr::taperpad::{PadIn, slice_taper_in_pad};
use crate::imod::libfft::nice_fft_limit;
use crate::imod::libfft::todfft::todfft_c;
use crate::imod::libiimod::iimage::ii_fopen;
use crate::imod::libiimod::mrcfiles::{MrcHeader, mrc_head_read, mrc_read_slice};
use std::cell::{Cell, RefCell};
use std::io::Write as _;

/// `b3dutil.h:68`: `#define RADIANS_PER_DEGREE 0.01745329252` -- a *double*.
const RADIANS_PER_DEGREE: f64 = 0.01745329252;

/// The `imodUsageHeader` callback in the shape `PipReadOrParseOptions` takes.
fn imod_usage_header_for_pip(prog_name: &[u8]) {
    imod_usage_header(Some(&String::from_utf8_lossy(prog_name)));
    let _ = ImodFile::Stdout.flush();
}

/// C `main` in `measuredrift.cpp:27`.
pub fn measuredrift(arguments: &[String]) -> i32 {
    let mut num_opt_args: i32 = 0;
    let mut num_non_opt_args: i32 = 0;
    let num_zlist: i32;
    let mut ind: i32 = 0;
    let mut data_size: i32 = 0;
    let slice_mode: i32;
    let mut err: i32;
    let mut num_wedge: i32 = 8;
    let mut ring_width: f32 = 0.01;
    let mut fit_start: f32 = 0.025;
    let mut fit_end: f32 = 0.25;
    let mut amp_end_freq: f32 = 0.;
    let mut filename: Vec<u8> = Vec::new();
    let mut out_fp: Option<ImodFile> = None;
    let mut in_head = MrcHeader::default();
    let sec_list: Vec<i32>;

    let progname_owned = imod_prog_name(arguments.first().map_or("", String::as_str));
    let progname = progname_owned.as_bytes();

    // Fallbacks from    ../manpages/autodoc2man 2 1 measuredrift
    let num_options: i32 = 9;
    let options: [&[u8]; 9] = [
        b"input:InputFile:FN:",
        b"output:OutputFile:FN:",
        b"sections:SectionsToDo:LI:",
        b"wedges:NumberOfWedges:I:",
        b"ring:FrequencyRingWidth:F:",
        b"fit:FrequencyRangeToFit:FP:",
        b"oscil:OscillationEndFreq:F:",
        b"param:ParameterFile:PF:",
        b"help:usage:B:",
    ];

    // Startup with fallback
    let argv = arguments
        .iter()
        .map(|value| value.as_bytes().to_vec())
        .collect::<Vec<_>>();
    pip_read_or_parse_options(
        argv.len() as i32,
        &argv,
        &options,
        num_options,
        progname,
        2,
        1,
        1,
        &mut num_opt_args,
        &mut num_non_opt_args,
        Some(imod_usage_header_for_pip),
    );
    if pip_get_in_out_file(b"InputFile", 0, &mut filename) != 0 {
        exit_error(b"No input image file specified");
    }
    let Some(mut in_fp) = ii_fopen(&filename, "rb") else {
        exit_error(&c_format_bytes(
            "Opening input image file %s",
            &[CArg::Bytes(&filename)],
        ));
    };
    // `measuredrift.cpp:55` frees `filename` here and `:57` then passes the
    // freed pointer to `exitError`; the owned Rust name simply stays alive.
    if mrc_head_read(&mut in_fp, &mut in_head) != 0 {
        exit_error(&c_format_bytes(
            "Reading header of image file %s",
            &[CArg::Bytes(&filename)],
        ));
    }

    if pip_get_in_out_file(b"OutputFile", 1, &mut filename) == 0 {
        imod_backup_file(&String::from_utf8_lossy(&filename));
        let out_name = String::from_utf8_lossy(&filename).into_owned();
        out_fp = ImodFile::open(&out_name, "w");
        if out_fp.is_none() {
            exit_error(&c_format_bytes(
                "Failed to open file for spectra, %s",
                &[CArg::Bytes(&filename)],
            ));
        }
    }

    if pip_get_string(b"SectionsToDo", &mut filename) == 0 {
        let Ok(list) = parselist(&String::from_utf8_lossy(&filename)) else {
            exit_error(b"Bad entry in list of sections to do");
        };
        num_zlist = list.len() as i32;
        sec_list = list;
    } else {
        let mut list = vec![0i32; in_head.nz as usize];
        ind = 0;
        while ind < in_head.nz {
            list[ind as usize] = ind;
            ind += 1;
        }
        num_zlist = in_head.nz;
        sec_list = list;
    }

    pip_get_integer(b"NumberOfWedges", &mut num_wedge);
    pip_get_float(b"FrequencyRingWidth", &mut ring_width);
    pip_get_two_floats(b"FrequencyRangeToFit", &mut fit_start, &mut fit_end);
    if pip_get_float(b"OscillationEndFreq", &mut amp_end_freq) == 0 && amp_end_freq < fit_end {
        exit_error(
            b"The ending frequency for measuring oscillations cannot be less than \
the ending frequency for fitting",
        );
    }
    slice_mode = slice_mode_if_real(in_head.mode);
    if slice_mode < 0 {
        exit_error(&c_format_bytes(
            "File mode is %d; only byte, short, float allowed",
            &[CArg::Int(in_head.mode as i64)],
        ));
    }
    data_size_for_mode(in_head.mode, &mut data_size, &mut ind);
    let mut array = vec![0u8; (data_size * in_head.nx * in_head.ny) as usize];

    ind = 0;
    while ind < num_zlist {
        let err_read = mrc_read_slice(
            &mut array,
            &mut in_fp,
            &mut in_head,
            sec_list[ind as usize],
            b'Z',
        );
        if err_read != 0 {
            exit_error(&c_format_bytes(
                "Reading slice %d, return code %d: %s\n",
                &[
                    CArg::Int(sec_list[ind as usize] as i64),
                    CArg::Int(err_read as i64),
                    CArg::Str(&b3d_get_error()),
                ],
            ));
        }
        err = measure_drift(
            &array,
            in_head.mode,
            in_head.nx,
            in_head.ny,
            num_wedge,
            ring_width,
            fit_start,
            fit_end,
            amp_end_freq,
            out_fp.as_mut(),
            sec_list[ind as usize],
        );
        if err != 0 {
            if err > 1 {
                exit_error(b"Parameter out of range in call to measureDrift");
            } else {
                exit_error(b"Allocating memory in measureDrift");
            }
        }
        ind += 1;
    }
    // `if (outFP) fclose(outFP);`
    drop(out_fp);
    0
}

const MAX_RINGS: usize = 200;

thread_local! {
    /// C `static float *sSpectrum;`
    static S_SPECTRUM: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    /// C `static float *sFreqs;`
    static S_FREQS: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    /// C `static float *sAllSum;`
    static S_ALL_SUM: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    /// C `static int sNumFit, sNumRings;`
    static S_NUM_FIT: Cell<i32> = const { Cell::new(0) };
    static S_NUM_RINGS: Cell<i32> = const { Cell::new(0) };
    /// C `static float sMinFreq, sMaxFreq;`
    static S_MIN_FREQ: Cell<f32> = const { Cell::new(0.) };
    static S_MAX_FREQ: Cell<f32> = const { Cell::new(0.) };
}

/// C `measureDrift` (`measuredrift.cpp:119`).
///
/// `void *array` is the raw section buffer, so it arrives here as bytes plus
/// the MRC `type` the caller read the header with, exactly as in the C.
///
/// Three source array bounds are unchecked and reachable from the command
/// line; each is an upstream out-of-bounds access rather than behaviour that
/// can be reproduced:
///   * `numRings` is `(int)(0.5 / ringDelFreq)` with no upper test, so
///     `-ring 0.001` indexes the `float xfit[MAX_RINGS]` stack arrays past
///     200 (`measuredrift.cpp:136-139`);
///   * `ampEndRing` is `B3DNINT(ampEndFreq / ringDelFreq)` with no test
///     against `numRings`, so `-oscil 0.9` reads `spectra[indUse][indRing]`
///     past the `numRings` allocation (`:311`);
///   * `extrema[MAX_RINGS / 2]` holds at most 100 extrema (`:265`).
/// Rust panics on each instead of reading adjacent memory.
#[allow(clippy::too_many_arguments)]
pub fn measure_drift(
    array: &[u8],
    typ: i32,
    nx: i32,
    ny: i32,
    num_wedge: i32,
    ring_del_freq: f32,
    start_freq: f32,
    end_freq: f32,
    mut amp_end_freq: f32,
    mut out_fp: Option<&mut ImodFile>,
    iz_type: i32,
) -> i32 {
    let nx_pad: i32;
    let ny_pad: i32;
    let num_rings: i32;
    let mut num_iter: i32 = 0;
    let start_ring: i32;
    let mut ind_ring: i32;
    let mut ind: i32 = 0;
    let mut ind_wedge: i32;
    let nx_taper: i32;
    let ny_taper: i32;
    let nx_div2p1: i32;
    let mut ix: i32;
    let mut iy: i32;
    let mut index: i32;
    let mut indp1: i32;
    let mut type_num: i32;
    let mut num_extrema: i32;
    let mut cen_iter: i32;
    let delx: f32;
    let dely: f32;
    let theta_base: f32;
    let del_wedge: f32;
    let mut fy: f32;
    let mut fx: f32;
    let mut ysq: f32;
    let mut rad: f32;
    let mut theta: f32 = 0.;
    let mut min_power: f32;
    let mut max_power: f32;
    let mut log_base: f32 = 0.;
    let mut xzero: f32 = 0.;
    let mut slope: f32 = 0.;
    let mut intcp: f32 = 0.;
    let mut err_min: f32 = 0.;
    let mut crossing: f32;
    let mut rms_err: f32 = 0.;
    let mut xrad: f32 = 0.;
    let mut yrad: f32 = 0.;
    let mut afit: f32 = 0.;
    let mut bfit: f32 = 0.;
    let mut cfit: f32 = 0.;
    let mut cen_offset: f32;
    let mut ratio_avg: f32 = 0.;
    let mut ratio_sd: f32 = 0.;
    let mut error_avg: f32 = 0.;
    let mut error_sd: f32 = 0.;
    let amp_end_ring: f32;
    let mut ang_diff: f32;
    let mut max_ang_diff: f32;
    let mut oscil_avg: f32 = 0.;
    let mut oscil_sd: f32 = 0.;
    let mut next_wedge: i32;
    let mut fit_start: i32;
    let mut fit_end: i32;
    let mut fit_range: i32;
    let mut jnd: i32;
    let mut diff_sum: i32;
    let num_cen_iter: i32 = 3;
    let mut break_ring: i32 = 0;
    let mut ind_use: i32;
    let mut ratios = [0.0f32; 10];
    let mut errors = [0.0f32; 10];
    let mut oscil_ratio = [0.0f32; 10];
    let taper_frac: f32 = 0.02;
    let pi_val: f32 = (180. * RADIANS_PER_DEGREE) as f32;
    const MAX_WEDGE: usize = 36;
    let mut smooth_diffs = [0.0f32; MAX_WEDGE + 1];
    let mut xfit = [0.0f32; MAX_RINGS];
    let mut yfit = [0.0f32; MAX_RINGS];
    let mut smoothed = [0.0f32; MAX_RINGS];
    let mut freqs = [0.0f32; MAX_RINGS];
    let mut work = [0.0f32; 4 * MAX_WEDGE + 25];
    let mut extrema = [0.0f32; MAX_RINGS / 2];

    // Values and arrays for simplex fit
    let ptol_facs: [f32; 2] = [5.0e-4, 1.0e-5];
    let ftol_facs: [f32; 2] = [5.0e-4, 1.0e-5];
    let delfac: f32 = 2.;
    let da: [f32; 5] = [0.1, 0.1, 0., 0., 0.];
    let mut yy = [0.0f32; 6];
    let mut aa = [0.0f32; 5];

    num_rings = (0.5 / ring_del_freq as f64) as i32;
    if num_rings < 5 || num_wedge < 2 || num_wedge >= MAX_WEDGE as i32 {
        return 2;
    }
    if start_freq < ring_del_freq || end_freq > 0.5 || start_freq > end_freq - 0.005 {
        return 2;
    }
    start_ring = (start_freq as f64 / ring_del_freq as f64 + 0.5).floor() as i32;
    if amp_end_freq <= 0. {
        amp_end_freq = end_freq;
    }
    amp_end_ring = ((amp_end_freq as f64 / ring_del_freq as f64 + 0.5).floor() as i32) as f32;

    // Get the padding and allocate work array
    nx_pad = nice_frame(nx, 2, nice_fft_limit());
    ny_pad = nice_frame(ny, 2, nice_fft_limit());
    let mut work_arr = vec![0.0f32; ((nx_pad + 2) * ny_pad) as usize];

    // Allocate arrays for wedge spectra.  The C allocates `numWedge + 1` of
    // each with `B3DMALLOC` and `cleanupDrift`s them; owned vectors drop.
    let mut freq_counts = vec![vec![0i32; num_rings as usize]; num_wedge as usize + 1];
    let mut spectra = vec![vec![0.0f32; num_rings as usize]; num_wedge as usize + 1];
    let mut sub_spec = vec![vec![0.0f32; num_rings as usize]; num_wedge as usize + 1];
    // `allSum = spectra[numWedge];` -- an alias, kept as the same element.
    let all_sum_ind = num_wedge as usize;

    // Get the taper and taper/pad and take FFT
    nx_taper = if 8. > taper_frac * nx_pad as f32 {
        8
    } else {
        (taper_frac * nx_pad as f32) as i32
    };
    ny_taper = if 8. > taper_frac * ny_pad as f32 {
        8
    } else {
        (taper_frac * ny_pad as f32) as i32
    };
    let short_view: Vec<i16>;
    let ushort_view: Vec<u16>;
    let float_view: Vec<f32>;
    let pad_in = match typ {
        1 => {
            short_view = array
                .chunks_exact(2)
                .map(|c| i16::from_ne_bytes([c[0], c[1]]))
                .collect();
            PadIn::Short(&short_view)
        }
        6 => {
            ushort_view = array
                .chunks_exact(2)
                .map(|c| u16::from_ne_bytes([c[0], c[1]]))
                .collect();
            PadIn::UShort(&ushort_view)
        }
        2 => {
            float_view = array
                .chunks_exact(4)
                .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            PadIn::Float(&float_view)
        }
        _ => PadIn::Byte(array),
    };
    slice_taper_in_pad(
        pad_in,
        typ,
        nx,
        0,
        nx - 1,
        0,
        ny - 1,
        &mut work_arr,
        nx_pad + 2,
        nx_pad,
        ny_pad,
        nx_taper,
        ny_taper,
    );
    todfft_c(&mut work_arr, nx_pad, ny_pad, 0);

    // Get frequency increments in image, and
    nx_div2p1 = nx_pad / 2 + 1;
    delx = (1.0 / nx_pad as f64) as f32;
    dely = (1.0 / ny_pad as f64) as f32;
    theta_base = -((90.01 * RADIANS_PER_DEGREE) as f32);
    del_wedge = (180.02 * RADIANS_PER_DEGREE / num_wedge as f64) as f32;

    cen_iter = 0;
    while cen_iter < num_cen_iter {
        ind = 0;
        while ind <= num_wedge {
            freq_counts[ind as usize].iter_mut().for_each(|v| *v = 0);
            sub_spec[ind as usize].iter_mut().for_each(|v| *v = 0.);
            ind += 1;
        }
        cen_offset = cen_iter as f32 * del_wedge / num_cen_iter as f32;

        // Loop and add pixels into power sums in wedge rings
        iy = 0;
        while iy < ny_pad {
            fy = iy as f32 * dely;
            index = iy * nx_div2p1;
            if fy > 0.5 {
                fy -= 1.0;
            }
            ysq = fy * fy;
            ix = 0;
            while ix < nx_div2p1 {
                if ix == 0 && iy == 0 {
                    ix += 1;
                    continue;
                }
                fx = ix as f32 * delx;
                ind = 2 * (index + ix);
                indp1 = ind + 1;
                rad = (fx * fx + ysq).sqrt();
                ind_ring = (rad / ring_del_freq) as i32;
                if ind_ring < num_rings {
                    theta = fy.atan2(fx) - theta_base - cen_offset;
                    if theta < 0. {
                        theta += pi_val;
                    }
                    ind_wedge = (theta / del_wedge) as i32;
                    freq_counts[ind_wedge as usize][ind_ring as usize] += 1;
                    sub_spec[ind_wedge as usize][ind_ring as usize] += work_arr[ind as usize]
                        * work_arr[ind as usize]
                        + work_arr[indp1 as usize] * work_arr[indp1 as usize];
                }
                ix += 1;
            }
            iy += 1;
        }

        // Compute mean power in each ring and get mins and maxes outside of first
        // ring/circle
        max_power = -1.0e10;
        min_power = 1.0e30;
        spectra[all_sum_ind].iter_mut().for_each(|v| *v = 0.);
        ind_wedge = 0;
        while ind_wedge < num_wedge {
            next_wedge = (ind_wedge + 1) % num_wedge;
            ind_ring = 0;
            while ind_ring < num_rings {
                ind = freq_counts[ind_wedge as usize][ind_ring as usize]
                    + freq_counts[next_wedge as usize][ind_ring as usize];
                if ind != 0 {
                    spectra[ind_wedge as usize][ind_ring as usize] = (sub_spec[ind_wedge as usize]
                        [ind_ring as usize]
                        + sub_spec[next_wedge as usize][ind_ring as usize])
                        / ind as f32;
                    if ind_ring != 0 {
                        let v = spectra[ind_wedge as usize][ind_ring as usize];
                        if min_power >= v {
                            min_power = v;
                        }
                        if max_power <= v {
                            max_power = v;
                        }
                    }
                }
                ind_ring += 1;
            }
            ind_wedge += 1;
        }

        // Take log with a sensible base
        log_base = (0.1 * min_power as f64) as f32;
        max_power = -1.0e10;
        min_power = 1.0e30;
        ind_ring = 0;
        while ind_ring < num_rings {
            ind_wedge = 0;
            while ind_wedge < num_wedge {
                xzero = spectra[ind_wedge as usize][ind_ring as usize];
                spectra[ind_wedge as usize][ind_ring as usize] =
                    (spectra[ind_wedge as usize][ind_ring as usize] + log_base).ln();

                // Add in to the average of these spectra: this average is bound to be
                // comparable in scaling to the spectra, unlike true full spectrum
                let v = spectra[ind_wedge as usize][ind_ring as usize];
                spectra[all_sum_ind][ind_ring as usize] += v / num_wedge as f32;
                ind_wedge += 1;
            }
            if ind_ring != 0 {
                let v = spectra[all_sum_ind][ind_ring as usize];
                if min_power >= v {
                    min_power = v;
                }
                if max_power <= v {
                    max_power = v;
                }
            }
            freqs[ind_ring as usize] = (ind_ring as f64 + 0.5) as f32 * ring_del_freq;
            ind_ring += 1;
        }

        // Find extrema in grand sum: at least two points monotonically less/more on
        // each side
        num_extrema = 0;
        diff_sum = 0;
        ind = 2;
        while ind < num_rings - 2 {
            let a = &spectra[all_sum_ind];
            if (a[ind as usize] > a[ind as usize - 1]
                && a[ind as usize - 1] > a[ind as usize - 2]
                && a[ind as usize] > a[ind as usize + 1]
                && a[ind as usize + 1] > a[ind as usize + 2])
                || (a[ind as usize] < a[ind as usize - 1]
                    && a[ind as usize - 1] < a[ind as usize - 2]
                    && a[ind as usize] < a[ind as usize + 1]
                    && a[ind as usize + 1] < a[ind as usize + 2])
            {
                if num_extrema != 0 {
                    diff_sum += (ind as f32 - extrema[num_extrema as usize - 1]) as i32;
                }
                extrema[num_extrema as usize] = ind as f32;
                num_extrema += 1;
            }
            ind += 1;
        }

        fit_range = 7;
        if num_extrema > 1 && extrema[0] < (num_rings / 2) as f32 {
            fit_range = 4 * diff_sum / (num_extrema - 1);
            // B3DCLAMP(fitRange, 7, numRings / 3)
            fit_range = if 7
                > (if num_rings / 3 < fit_range {
                    num_rings / 3
                } else {
                    fit_range
                }) {
                7
            } else if num_rings / 3 < fit_range {
                num_rings / 3
            } else {
                fit_range
            };
        }
        // `PRINT3` (`cppdefs.h:23`) writes through C++ `cout`.
        let _ = ImodFile::Stdout.write_all(&c_format_bytes(
            "numExtrema = %d,  diffSum = %d,  fitRange = %d\n",
            &[
                CArg::Int(num_extrema as i64),
                CArg::Int(diff_sum as i64),
                CArg::Int(fit_range as i64),
            ],
        ));
        let _ = ImodFile::Stdout.flush();

        ind_wedge = -1;
        while ind_wedge < num_wedge {
            ind_use = if ind_wedge < 0 { num_wedge } else { ind_wedge };
            smoothed[0] = spectra[ind_use as usize][0];

            // Fit parabola to each point for smoothing
            ind_ring = 1;
            while ind_ring < num_rings {
                fit_start = if 1 > ind_ring - fit_range / 2 {
                    1
                } else {
                    ind_ring - fit_range / 2
                };
                fit_end = if num_rings < fit_start + fit_range {
                    num_rings
                } else {
                    fit_start + fit_range
                };
                fit_start = fit_end - fit_range;
                jnd = 0;
                ind = fit_start;
                while ind < fit_end {
                    xfit[jnd as usize] = freqs[ind as usize] - freqs[ind_ring as usize];
                    yfit[jnd as usize] = xfit[jnd as usize] * xfit[jnd as usize];
                    jnd += 1;
                    ind += 1;
                }
                crate::imod::libcfshr::simplestat::ls_fit2_pred(
                    &xfit,
                    &yfit,
                    &spectra[ind_use as usize][fit_start as usize..],
                    jnd,
                    &mut afit,
                    &mut bfit,
                    Some(&mut cfit),
                    0.,
                    0.,
                    &mut smoothed[ind_ring as usize],
                    &mut xzero,
                );
                ind_ring += 1;
            }

            // Try to find breakpoint in grand sum to start amplitude SD measurement
            if ind_wedge < 0 {
                break_ring = start_ring;
                max_ang_diff = -1.0e10;
                ind_ring = start_ring + 2;
                while (ind_ring as f32) < amp_end_ring - 2. {
                    fit_start = if start_ring > ind_ring - 4 {
                        start_ring
                    } else {
                        ind_ring - 4
                    };
                    ind = fit_start;
                    while ind < ind_ring {
                        yfit[(ind - fit_start) as usize] =
                            (spectra[all_sum_ind][ind as usize] as f64 * 0.5
                                / (max_power - min_power) as f64)
                                as f32;
                        ind += 1;
                    }
                    // `lsFit(&freqs[startRing], ...)` -- the source really does
                    // index `freqs` from `startRing` rather than from `fitStart`.
                    crate::imod::libcfshr::simplestat::ls_fit(
                        &freqs[start_ring as usize..],
                        &yfit,
                        ind_ring - fit_start,
                        &mut slope,
                        &mut intcp,
                        &mut xrad,
                    );
                    theta = (-slope as f64).atan() as f32;
                    fit_end = if ((ind_ring + 6) as f32) < amp_end_ring {
                        ind_ring + 6
                    } else {
                        amp_end_ring as i32
                    };
                    ind = ind_ring + 1;
                    while ind <= fit_end {
                        yfit[(ind - (ind_ring + 1)) as usize] =
                            (spectra[all_sum_ind][ind as usize] as f64 * 0.5
                                / (max_power - min_power) as f64)
                                as f32;
                        ind += 1;
                    }
                    crate::imod::libcfshr::simplestat::ls_fit(
                        &freqs[(ind_ring + 1) as usize..],
                        &yfit,
                        fit_end - ind_ring,
                        &mut slope,
                        &mut intcp,
                        &mut xrad,
                    );
                    ang_diff = (theta as f64 - (-slope as f64).atan()) as f32;
                    if ang_diff > max_ang_diff {
                        max_ang_diff = ang_diff;
                        break_ring = ind_ring;
                    }
                    ind_ring += 1;
                }
            }

            smooth_diffs[ind_use as usize] = 0.;
            ind_ring = break_ring;
            while (ind_ring as f32) <= amp_end_ring {
                smooth_diffs[ind_use as usize] += ((spectra[ind_use as usize][ind_ring as usize]
                    - smoothed[ind_ring as usize])
                    as f64)
                    .abs() as f32;
                ind_ring += 1;
            }

            // Output raw spectrum if asked
            if out_fp.is_some() && cen_iter == 0 {
                type_num = ind_use + 1000 * iz_type;
                ind = 0;
                while ind < num_rings {
                    let line = c_format_bytes(
                        "%4d %.3f %8.5f\n",
                        &[
                            CArg::Int(type_num as i64),
                            CArg::Dbl(freqs[ind as usize] as f64),
                            CArg::Dbl(spectra[ind_use as usize][ind as usize] as f64),
                        ],
                    );
                    let _ = out_fp.as_mut().unwrap().write_all(&line);
                    ind += 1;
                }
            }

            // Copy smoothed in and output it
            ind = 0;
            while ind < num_rings {
                spectra[ind_use as usize][ind as usize] = smoothed[ind as usize];
                type_num = ind_use + 100 + 1000 * iz_type;
                if out_fp.is_some() && cen_iter == 0 {
                    let line = c_format_bytes(
                        "%4d %.3f %8.5f\n",
                        &[
                            CArg::Int(type_num as i64),
                            CArg::Dbl(freqs[ind as usize] as f64),
                            CArg::Dbl(spectra[ind_use as usize][ind as usize] as f64),
                        ],
                    );
                    let _ = out_fp.as_mut().unwrap().write_all(&line);
                }
                ind += 1;
            }
            ind_wedge += 1;
        }

        // Oscillation amount fit to ellipse
        ind_wedge = 0;
        while ind_wedge < num_wedge {
            theta = ((ind_wedge as f64 + 0.5) * del_wedge as f64 + theta_base as f64) as f32;
            xfit[ind_wedge as usize] =
                smooth_diffs[ind_wedge as usize] * theta.cos() / smooth_diffs[num_wedge as usize];
            yfit[ind_wedge as usize] =
                smooth_diffs[ind_wedge as usize] * theta.sin() / smooth_diffs[num_wedge as usize];
            ind_wedge += 1;
        }
        fit_centered_ellipse(
            &xfit,
            &yfit,
            num_wedge,
            &mut xrad,
            &mut yrad,
            &mut theta,
            &mut rms_err,
            &mut work,
        );
        theta = (theta as f64 + cen_offset as f64 / RADIANS_PER_DEGREE) as f32;
        oscil_ratio[cen_iter as usize] = if xrad / yrad > yrad / xrad {
            xrad / yrad
        } else {
            yrad / xrad
        };
        let _ = ImodFile::Stdout.write_all(&c_format_bytes(
            "Modulation ratio %.3f  axis %.1f\n",
            &[
                CArg::Dbl(oscil_ratio[cen_iter as usize] as f64),
                CArg::Dbl(theta as f64),
            ],
        ));

        S_ALL_SUM.with_borrow_mut(|v| {
            v.clear();
            v.extend_from_slice(&spectra[all_sum_ind]);
        });
        S_FREQS.with_borrow_mut(|v| {
            v.clear();
            v.extend_from_slice(&freqs);
        });
        S_NUM_RINGS.with(|c| c.set(num_rings));
        S_MIN_FREQ.with(|c| c.set(start_freq));
        S_MAX_FREQ.with(|c| c.set(end_freq));
        S_NUM_FIT.with(|c| c.set(100));

        // Main fit of scaling to an ellipse
        ind_wedge = 0;
        while ind_wedge < num_wedge {
            S_SPECTRUM.with_borrow_mut(|v| {
                v.clear();
                v.extend_from_slice(&spectra[ind_wedge as usize]);
            });
            aa[0] = 1.;
            aa[1] = 0.;
            dual_amoeba(
                &mut yy,
                2,
                delfac,
                &ptol_facs,
                &ftol_facs,
                &mut aa,
                &da,
                &mut |values: &[f32]| {
                    let mut error = 0.0f32;
                    area_func(values, &mut error);
                    error
                },
                &mut num_iter,
            );
            type_num = iz_type * 100 + ind_wedge;
            let _ = type_num;
            area_func(&aa, &mut err_min);
            theta = ((ind_wedge as f64 + 0.5) * del_wedge as f64 + theta_base as f64) as f32;
            xfit[ind_wedge as usize] = aa[0] * theta.cos();
            yfit[ind_wedge as usize] = aa[0] * theta.sin();
            ind_wedge += 1;
        }
        fit_centered_ellipse(
            &xfit,
            &yfit,
            num_wedge,
            &mut xrad,
            &mut yrad,
            &mut theta,
            &mut rms_err,
            &mut work,
        );
        theta = (theta as f64 + cen_offset as f64 / RADIANS_PER_DEGREE) as f32;
        ratios[cen_iter as usize] = if xrad / yrad > yrad / xrad {
            xrad / yrad
        } else {
            yrad / xrad
        };
        errors[cen_iter as usize] = rms_err;
        let _ = ImodFile::Stdout.write_all(&c_format_bytes(
            "xrad %.3f yrad %.3f theta %.1f ratio %.3f err %f\n",
            &[
                CArg::Dbl(xrad as f64),
                CArg::Dbl(yrad as f64),
                CArg::Dbl(theta as f64),
                CArg::Dbl(ratios[cen_iter as usize] as f64),
                CArg::Dbl(rms_err as f64),
            ],
        ));
        cen_iter += 1;
    }
    crate::imod::libcfshr::simplestat::avg_sd(
        &ratios,
        num_cen_iter,
        &mut ratio_avg,
        &mut ratio_sd,
        &mut xrad,
    );
    crate::imod::libcfshr::simplestat::avg_sd(
        &oscil_ratio,
        num_cen_iter,
        &mut oscil_avg,
        &mut oscil_sd,
        &mut theta,
    );
    let _ = error_avg;
    let _ = error_sd;
    let _ = crossing;
    let _ = ImodFile::Stdout.write_all(&c_format_bytes(
        "%d mean&SD ellipse ratio for fall: %.3f %.3f   oscillations: %.3f %.3f\n",
        &[
            CArg::Int(iz_type as i64),
            CArg::Dbl(ratio_avg as f64),
            CArg::Dbl(ratio_sd as f64),
            CArg::Dbl(oscil_avg as f64),
            CArg::Dbl(oscil_sd as f64),
        ],
    ));
    let _ = ImodFile::Stdout.flush();

    cleanup_drift(
        &mut work_arr,
        &mut spectra,
        &mut sub_spec,
        &mut freq_counts,
        num_wedge,
    );
    0
}

/// C `cleanupDrift` (`measuredrift.cpp:446`).
///
/// The C `free`s the work array and the `maxInd + 1` per-wedge allocations;
/// the owned vectors here are released the same way, by index.
pub fn cleanup_drift(
    work_arr: &mut Vec<f32>,
    spectra: &mut [Vec<f32>],
    sub_spec: &mut [Vec<f32>],
    freq_counts: &mut [Vec<i32>],
    max_ind: i32,
) {
    let mut ix: i32;
    *work_arr = Vec::new();
    ix = 0;
    while ix <= max_ind {
        freq_counts[ix as usize] = Vec::new();
        spectra[ix as usize] = Vec::new();
        sub_spec[ix as usize] = Vec::new();
        ix += 1;
    }
}

/// C `interpolateCurve` (`measuredrift.cpp:458`).
pub fn interpolate_curve(
    xx: &[f32],
    yy: &[f32],
    num_pts: i32,
    xval: f32,
    crossing: &mut f32,
) -> i32 {
    let mut ind: i32 = 0;
    while ind < num_pts - 1 {
        if (xx[ind as usize] <= xval && xx[ind as usize + 1] > xval)
            || (xx[ind as usize] > xval && xx[ind as usize + 1] <= xval)
        {
            *crossing = yy[ind as usize]
                + (xval - xx[ind as usize]) * (yy[ind as usize + 1] - yy[ind as usize])
                    / (xx[ind as usize + 1] - xx[ind as usize]);
            return 0;
        }
        ind += 1;
    }
    1
}

/// C `areaFunc` (`measuredrift.cpp:472`), the `dualAmoeba` error callback.
///
/// `allVal` and `specVal` are uninitialised stack floats in the C and stay so
/// when `interpolateCurve` finds no crossing (its return value is discarded);
/// the zero initialisation here is the deterministic stand-in.
pub fn area_func(aa: &[f32], error: &mut f32) {
    let mut asum: f64 = 0.;
    let freq_scale: f32 = aa[0];
    let spec_add: f32 = aa[1];
    let mut all_freq: f32 = 0.;
    let mut spec_freq: f32 = 0.;
    let mut all_val: f32 = 0.;
    let mut spec_val: f32 = 0.;
    let del_freq: f32;
    let mut ind: i32 = 0;
    let num_fit = S_NUM_FIT.with(|c| c.get());
    let num_rings = S_NUM_RINGS.with(|c| c.get());
    let min_freq = S_MIN_FREQ.with(|c| c.get());
    let max_freq = S_MAX_FREQ.with(|c| c.get());
    del_freq = (max_freq - min_freq) / (num_fit - 1) as f32;
    S_FREQS.with_borrow(|freqs| {
        S_ALL_SUM.with_borrow(|all_sum| {
            S_SPECTRUM.with_borrow(|spectrum| {
                ind = 0;
                while ind < num_fit {
                    all_freq = min_freq + ind as f32 * del_freq;
                    spec_freq = all_freq / freq_scale;
                    interpolate_curve(freqs, all_sum, num_rings, all_freq, &mut all_val);
                    interpolate_curve(freqs, spectrum, num_rings, spec_freq, &mut spec_val);
                    spec_val += spec_add;
                    asum += ((all_val - spec_val) as f64).abs();
                    ind += 1;
                }
            })
        })
    });
    *error = (asum * del_freq as f64) as f32;
}
