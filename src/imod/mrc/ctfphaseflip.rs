//! Translation of `IMOD/mrc/ctfphaseflip.cpp` — CTF correction of tilted
//! images.
//!
//! One Rust function per source function: the one big `main` plus the three
//! file statics `firstZeroShift`, `getAnyOneZero` and `interpolateTable`.
//! The GPU entry points come from the no-CUDA `nogpuctf.cpp` stub, which is
//! what the reference build links.

use std::io::Write as _;

use crate::imod::clip::clip::{ScanArg, sscanf};
use crate::imod::ctfplotter::ctfutils::{
    DEF_FILE_HAS_ASTIG, DEF_FILE_HAS_CUT_ON, DEF_FILE_HAS_PHASE, FREQ_FOR_PHASE, SavedDefocus,
    check_and_fix_defocus_list, read_defocus_file, read_tilt_angles,
};
use crate::imod::libcfshr::amat_to_rotmagstr::amat_to_rotmagstr;
use crate::imod::libcfshr::b3dutil::{
    CArg, ImodFile, angle_within_limits, b3d_lock_file, b3d_output_file_type, b3d_unlock_file,
    c_format_bytes, fgetline, get_standard_gpu_options, imod_backup_file, imod_prog_name,
    set_or_clear_flags,
};
use crate::imod::libcfshr::coresprocsthreads::wall_time;
use crate::imod::libcfshr::filtxcorr::nice_frame;
use crate::imod::libcfshr::islice::{slice_create, slice_mode_if_real};
use crate::imod::libcfshr::parse_params::{
    exit_error, pip_get_boolean, pip_get_float, pip_get_integer, pip_get_string,
    pip_get_two_integers, pip_print_help, pip_read_or_parse_options,
};
use crate::imod::libcfshr::taperpad::{PadIn, slice_taper_in_pad};
use crate::imod::libfft::odfft::nice_fft_limit;
use crate::imod::libfft::rustfft_backend::todfftc;
use crate::imod::libiimod::iihdf::hdf_write_dummy_section;
use crate::imod::libiimod::iihdf::ii_test_if_hdf;
use crate::imod::libiimod::iimage::{
    ii_delete, ii_fclose, ii_fopen, ii_lookup_file_from_fp, ii_sync_from_mrc_header,
};
use crate::imod::libiimod::mrcfiles::{
    MRC_MODE_FLOAT, MrcHeader, mrc_head_label, mrc_head_read, mrc_head_write,
    mrc_init_output_header, mrc_read_slice,
};
use crate::imod::libiimod::mrcslice::{slice_mmm, slice_new_mode};
use crate::imod::libiimod::parallelwrite::{
    par_wrt_close, par_wrt_flush_buffers, par_wrt_initialize, par_wrt_properties,
    par_wrt_reclose_hdf, parallel_write_slice,
};
use crate::imod::libiimod::unit_fileio::IIFILE_HDF;
use crate::imod::mrc::nogpuctf::{
    gpu_available, gpu_copy_columns, gpu_copy_diagonals, gpu_correct_ctf,
    gpu_extract_and_transform, gpu_get_times, gpu_initialize_slice, gpu_interp_diagonals,
    gpu_interpolate_columns, gpu_return_image,
};

/// tilt Angle less than this in degrees is treated as 0.0
const MIN_ANGLE: f64 = 0.001;
const MY_PI: f64 = 3.1415926;
const UNUSED_DEFOCUS: f32 = -2000000.0;
const MAXLINE: i32 = 160;
/// `b3dutil.h`'s `RADIANS_PER_DEGREE` macro.
const RADIANS_PER_DEGREE: f64 = 0.01745329252;
/// `b3dutil.h:52`.
const MRC_FLAGS_CTF_CORRECTED: u32 = 32;
const OUTPUT_TYPE_TIFF: i32 = 1;
const OUTPUT_TYPE_HDF: i32 = 5;
const SLICE_MODE_FLOAT: i32 = 2;

/// C `main` in `ctfphaseflip.cpp` (`ctfphaseflip.cpp:132`).
#[allow(clippy::needless_range_loop)]
pub fn ctfphaseflip(arguments: &[String]) -> i32 {
    let mut num_opt_args = 0i32;
    let mut num_non_opt_args = 0i32;

    // Fallbacks from   ../manpages/autodoc2man 2 1 ctfphaseflip
    let num_options: i32 = 32;
    let options: [&[u8]; 32] = [
        b"input:InputStack:FN:",
        b"output:OutputFileName:FN:",
        b"angleFn:AngleFile:FN:",
        b"invert:InvertTiltAngles:B:",
        b"axis:AxisAngle:F:",
        b"xtilt:XAxisTilt:F:",
        b"defFn:DefocusFile:FN:",
        b"zoff:OffsetInZ:F:",
        b"xform:TransformFile:FN:",
        b"defTol:DefocusTol:I:",
        b"maxWidth:MaximumStripWidth:I:",
        b"iWidth:InterpolationWidth:I:",
        b"zero:MinimumZeroSpacing:F:",
        b"pixelSize:PixelSize:F:",
        b"unbinned:UnbinnedPixelSize:F:",
        b"expanded:ExpandedByFactor:F:",
        b"volt:Voltage:I:",
        b"cs:SphericalAberration:F:",
        b"ampContrast:AmplitudeContrast:F:",
        b"degPhase:PhaseShiftInDegrees:F:",
        b"phase:PhasePlateShift:F:",
        b"cuton:CutOnFrequency:F:",
        b"scale:ScaleByCtfPower:F:",
        b"gpu:UseGPU:I:",
        b"action:ActionIfGPUFails:IP:",
        b"views:StartingEndingViews:IP:",
        b"totalViews:TotalViews:IP:",
        b"boundary:BoundaryInfoFile:FN:",
        b"skipFlag:SkipCorrectedFlag:I:",
        b"debug:DebugOutput:I:",
        b"param:Parameter:PF:",
        b"help:usage:B:",
    ];

    let mut stack_fn: Vec<u8> = Vec::new();
    let mut angle_fn: Vec<u8> = Vec::new();
    let mut have_angle_fn: bool;
    let mut out_fn: Vec<u8> = Vec::new();
    let mut def_fn: Vec<u8> = Vec::new();
    let mut xform_fn: Vec<u8> = Vec::new();
    let mut have_xform = false;
    let mut bound_fn: Vec<u8> = Vec::new();
    let mut volt = 0i32;
    let mut i_width = 0i32;
    let mut defocus_tol = 0i32;
    let mut ii: i32;
    let mut ierr: i32;
    let mut def_flags = 0i32;
    let mut max_width: i32;
    let mut min_width: i32;
    let mut pixel_size = 0.0f32;
    let mut unbinned_pixel: f32;
    let mut cs = 0.0f32;
    let mut amp_contrast = 0.0f32;
    let mut strip_defocus: f32;
    let mut offset_in_z = 0.0f32;
    let mut starting_view: i32;
    let mut ending_view: i32;
    let mut starting_total = 0i32;
    let mut ending_total = 0i32;
    let mut def_version = 0i32;
    let mut is_single_run = false;
    let mut invert_angles = 0i32;
    let binning: i32;
    let mut max_strip_width = 0i32;
    let min_strip_width = 128i32;
    let min_width_to_scale_interp = 256i32; // The former maximum strip width
    let min_zero_shift = 0.6f32;
    let freq_for_inter_zero = 0.8f32; // Nyquist units at which to assess interzero distance
    let mut min_inter_zero_pixels = 8.0f32;
    let atten_start_frac = 0.81f32;
    let mut tilt_axis_angle = 0.0f32;
    let mut expand_factor = 1.0f32;
    let mut x_axis_tilt = 0.0f32;
    let mut use_gpu;
    let mut skip_done_flag = 0i32;
    let mut if_gpu_by_env = 0i32;
    let mut act_gpu_fail_option = 0i32;
    let mut act_gpu_fail_environ = 0i32;
    let mut debug_mode = 0i32;
    let mut hush_defoci = 0i32;
    let angle_sign: f64;
    let mut zero_shift: f64;
    let max_was_entered: bool;
    let scale_interp: bool;
    let mut do_full_images: bool;
    let do_diagonals: bool;
    let mut angle_is_zero: bool;
    let mut min_angle = 0.0f32;
    let mut max_angle = 0.0f32;
    let mut scale_by_power = 0.0f32;
    let mut phase_plate_shift = 0.0f32;
    let mut cut_on_entered = 0.0f32;
    let mut phase_deg = 0.0f32;
    let mut gpu_memory = 0.0f32;
    let mut tilt_angles: Vec<f32> = Vec::new();
    let mut have_tilt_angles = false;
    let mut x_shifts: Vec<f32>;
    let mut rotations: Vec<f32>;
    let mut defocus_list: Vec<SavedDefocus>;
    let mut line = [0u8; MAXLINE as usize];
    let progname_owned = imod_prog_name(arguments.first().map_or("", String::as_str));
    let progname = progname_owned.as_bytes();

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
        1,
        0,
        0,
        &mut num_opt_args,
        &mut num_non_opt_args,
        None,
    );

    ierr = 0;
    if pip_get_boolean(b"usage", &mut ierr) == 0 {
        pip_print_help(progname, 0, 0, 0);
        let _ = ImodFile::Stdout.flush();
        std::process::exit(0);
    }
    pip_get_integer(b"DebugOutput", &mut debug_mode);
    if debug_mode >= 10 {
        hush_defoci = 1;
        debug_mode -= 10;
    }
    if pip_get_string(b"InputStack", &mut stack_fn) != 0 {
        exit_error(b"No stack specified");
    }
    if pip_get_string(b"AngleFile", &mut angle_fn) != 0 {
        have_angle_fn = false;
        angle_fn.clear();
        let _ = ImodFile::Stdout
            .write_all(b"No angle file is specified, tilt angle is assumed to be 0.0\n");
    } else {
        have_angle_fn = true;
    }
    if pip_get_string(b"DefocusFile", &mut def_fn) != 0 {
        exit_error(b"No defocus file is specified");
    }

    // Get axis angle and X axis tilt and decide if doing diagonals
    pip_get_float(b"AxisAngle", &mut tilt_axis_angle);
    pip_get_float(b"XAxisTilt", &mut x_axis_tilt);
    do_diagonals = ((tilt_axis_angle as f64).abs() > MIN_ANGLE && have_angle_fn)
        || (x_axis_tilt as f64).abs() > 0.1;
    tilt_axis_angle = (tilt_axis_angle as f64 * RADIANS_PER_DEGREE) as f32;
    x_axis_tilt = (x_axis_tilt as f64 * RADIANS_PER_DEGREE) as f32;

    if pip_get_integer(b"DefocusTol", &mut defocus_tol) != 0 {
        exit_error(b"No DefocusTol specified");
    }
    if pip_get_integer(b"InterpolationWidth", &mut i_width) != 0 || i_width == 0 {
        exit_error(b"No InterpolationWidth specified, or 0 value entered");
    }
    scale_interp = i_width > 0;
    if !scale_interp {
        i_width = -i_width;
    }
    if pip_get_float(b"MinimumZeroSpacing", &mut min_inter_zero_pixels) == 0
        && (min_inter_zero_pixels < 3. || min_inter_zero_pixels > 30.)
    {
        exit_error(b"MinimumZeroSpacing must be between 3 and 30");
    }
    if pip_get_float(b"PixelSize", &mut pixel_size) != 0 {
        exit_error(b"No PixelSize specified");
    }
    unbinned_pixel = pixel_size;
    pip_get_float(b"UnbinnedPixelSize", &mut unbinned_pixel);
    binning = 1.max(((pixel_size / unbinned_pixel) as f64 + 0.5).floor() as i32);
    // `ctfphaseflip.cpp:231-232`: the division is the body of the `if`,
    // despite the source's indentation.
    if pip_get_float(b"ExpandedByFactor", &mut expand_factor) == 0 {
        pixel_size /= expand_factor;
    }
    if pip_get_integer(b"Voltage", &mut volt) != 0 {
        exit_error(b"Voltage is not specified");
    }
    if pip_get_float(b"SphericalAberration", &mut cs) != 0 {
        exit_error(b"SphericalAberration is not specified");
    }
    cs = if 0.01 > cs { 0.01 } else { cs };
    if pip_get_float(b"AmplitudeContrast", &mut amp_contrast) != 0 {
        exit_error(b"No AmplitudeContrast is specified");
    }
    ierr = pip_get_float(b"PhasePlateShift", &mut phase_plate_shift);
    if pip_get_float(b"PhaseShiftInDegrees", &mut phase_deg) == 0 {
        if ierr == 0 {
            exit_error(b"You cannot enter phase shift in both degrees and radians");
        }
        phase_plate_shift = (phase_deg as f64 * RADIANS_PER_DEGREE) as f32;
    }
    pip_get_float(b"CutOnFrequency", &mut cut_on_entered);
    pip_get_float(b"ScaleByCtfPower", &mut scale_by_power);
    pip_get_float(b"OffsetInZ", &mut offset_in_z);
    if pip_get_two_integers(b"TotalViews", &mut starting_total, &mut ending_total) != 0 {
        is_single_run = true; // TotalViews is not specified;
    }
    if pip_get_string(b"OutputFileName", &mut out_fn) != 0 {
        exit_error(b"OutputFileName is not specified");
    }
    let have_bound_fn = pip_get_string(b"BoundaryInfoFile", &mut bound_fn) == 0;
    pip_get_integer(b"SkipCorrectedFlag", &mut skip_done_flag);
    if pip_get_string(b"TransformFile", &mut xform_fn) == 0 {
        have_xform = true;
    }
    pip_get_boolean(b"InvertTiltAngles", &mut invert_angles);
    angle_sign = if invert_angles != 0 { -1. } else { 1. };

    // Get GPU specification and flag its source
    use_gpu = get_standard_gpu_options(
        &mut if_gpu_by_env,
        Some(&mut act_gpu_fail_option),
        Some(&mut act_gpu_fail_environ),
    );

    let _ = ImodFile::Stdout.write_all(&c_format_bytes(
        "stackFn = %s, angleFn=%s,  invertAngles=%d\n",
        &[
            CArg::Bytes(&stack_fn),
            CArg::Bytes(&angle_fn),
            CArg::Int(invert_angles.into()),
        ],
    ));
    let _ = ImodFile::Stdout.write_all(&c_format_bytes(
        "volt=%d Kv, interpolationWidth=%d pixels, defocusTol=%d nm \n",
        &[
            CArg::Int(volt.into()),
            CArg::Int(i_width.into()),
            CArg::Int(defocus_tol.into()),
        ],
    ));
    let _ = ImodFile::Stdout.write_all(&c_format_bytes(
        "pixelSize=%f nm, cs=%f mm, ampContrast=%f \n",
        &[
            CArg::Dbl(pixel_size as f64),
            CArg::Dbl(cs as f64),
            CArg::Dbl(amp_contrast as f64),
        ],
    ));

    let Some(mut fp_stack) = ii_fopen(&stack_fn, "rb") else {
        exit_error(&c_format_bytes(
            "Could not open input file %s",
            &[CArg::Bytes(&stack_fn)],
        ));
    };
    defocus_list = read_defocus_file(&def_fn, &mut def_version, &mut def_flags);
    if defocus_list.is_empty() {
        exit_error(&c_format_bytes(
            "The defocus file %s is non-existent or empty - did you save in ctfplotter?",
            &[CArg::Bytes(&def_fn)],
        ));
    }

    let mut foutput: Option<ImodFile> = None;
    let mut header = MrcHeader::default();
    let mut out_header: MrcHeader;
    let slice_mode: i32;
    let mut ii_file: *mut crate::imod::libiimod::iimage::ImodImageFile = std::ptr::null_mut();
    let mut parallel_hdf = false;

    /* read header */
    if mrc_head_read(&mut fp_stack, &mut header) != 0 {
        exit_error(&c_format_bytes(
            "Reading header of input file %s",
            &[CArg::Bytes(&stack_fn)],
        ));
    }
    if (skip_done_flag & 1) == 0 && (header.imod_flags as u32 & MRC_FLAGS_CTF_CORRECTED) != 0 {
        exit_error(b"The input file stack header has the flag set that it has been CTF corrected");
    }
    out_header = header.clone();
    max_was_entered = pip_get_integer(b"MaximumStripWidth", &mut max_strip_width) == 0;
    if !max_was_entered {
        max_strip_width = header.nx / 2;
    }
    do_full_images = max_strip_width >= header.nx;
    if do_diagonals && max_was_entered && !do_full_images {
        exit_error(
            b"You cannot enter a maximum strip width less than the image size with a \
              non-zero X-axis tilt or tilt axis angle",
        );
    }
    if do_diagonals {
        do_full_images = true;
    }
    if !do_full_images {
        max_strip_width = 2 * (max_strip_width / 2);
    }
    if use_gpu >= 0 && gpu_available(use_gpu, &mut gpu_memory, debug_mode) == 0 {
        ierr = if if_gpu_by_env != 0 {
            act_gpu_fail_environ
        } else {
            act_gpu_fail_option
        };
        if ierr > 1 {
            exit_error(b"Use of a GPU was requested but none is available");
        }
        let _ = ImodFile::Stdout.write_all(&c_format_bytes(
            "%sUse of a GPU was requested but none is available; using the CPU\n",
            &[CArg::Str(if ierr != 0 {
                "MESSAGE: Ctfphaseflip - "
            } else {
                ""
            })],
        ));
        use_gpu = -1;
    }

    // Allocate arrays for shifts and rotations and set to 0
    x_shifts = vec![0.0f32; header.nz.max(0) as usize];
    rotations = vec![0.0f32; header.nz.max(0) as usize];
    for ii in 0..header.nz {
        x_shifts[ii as usize] = 0.;
        rotations[ii as usize] = 0.;
    }

    // If transforms, read them and store shifts and rotations
    if have_xform {
        let mut dum1 = 0.0f32;
        let mut dum2 = 0.0f32;
        let mut dum3 = 0.0f32;
        let mut dum4 = 0.0f32;
        let Some(mut fp_xf) = ImodFile::open(&*String::from_utf8_lossy(&xform_fn), "r") else {
            exit_error(&c_format_bytes(
                "Opening transform file %s",
                &[CArg::Bytes(&xform_fn)],
            ));
        };
        ii = 0;
        while ii < header.nz {
            ierr = fgetline(&mut fp_xf, &mut line, MAXLINE);
            if ierr == -1 {
                exit_error(&c_format_bytes(
                    "Reading transform file %s",
                    &[CArg::Bytes(&xform_fn)],
                ));
            }
            if ierr == -2 {
                exit_error(&c_format_bytes(
                    "End of file after reading %d transforms; not enough transforms in file",
                    &[CArg::Int(ii.into())],
                ));
            }
            let end = line.iter().position(|&b| b == 0).unwrap_or(line.len());
            let mut shift = x_shifts[ii as usize];
            ierr = sscanf(
                &String::from_utf8_lossy(&line[..end]),
                "%f %f %f %f %f",
                &mut [
                    ScanArg::Flt(&mut dum1),
                    ScanArg::Flt(&mut dum2),
                    ScanArg::Flt(&mut dum3),
                    ScanArg::Flt(&mut dum4),
                    ScanArg::Flt(&mut shift),
                ],
            );
            x_shifts[ii as usize] = shift;
            if ierr <= 0 {
                // `ii--; continue;` inside the source's `for (... ii++)`, so
                // the same index is retried on the next line.
                continue;
            }
            x_shifts[ii as usize] /= binning as f32;
            let (rot, _mag, _str, _phi) = amat_to_rotmagstr(dum1, dum2, dum3, dum4);
            rotations[ii as usize] = rot;
            ii += 1;
        }
    }

    starting_view = 1;
    ending_view = header.nz;
    pip_get_two_integers(b"StartingEndingViews", &mut starting_view, &mut ending_view);
    slice_mode = slice_mode_if_real(header.mode);
    if slice_mode < 0 {
        exit_error(&c_format_bytes(
            "File mode is %d; only byte, short integer, or real allowed",
            &[CArg::Int(header.mode.into())],
        ));
    }

    //The number of slices this run deals with;
    let curr_nz = ending_view - starting_view + 1;
    if is_single_run {
        out_header.nz = curr_nz;
        out_header.mz = curr_nz;
    } else {
        out_header.nz = ending_total - starting_total + 1;
        out_header.mz = ending_total - starting_total + 1;

        // Determine if doing parallel HDF by output type setting for setup run, or
        // by simple test of file for real runs
        if starting_view == -1 && ending_view == -1 {
            parallel_hdf = b3d_output_file_type() == OUTPUT_TYPE_HDF;
        } else {
            parallel_hdf = have_bound_fn && ii_test_if_hdf(&out_fn) > 0;
        }
    }
    out_header.zlen = header.zlen * out_header.mz as f32 / header.mz as f32;
    mrc_init_output_header(&mut out_header);
    mrc_head_label(
        &mut out_header,
        b"ctfPhaseFlip: CTF correction with phase flipping only",
    );
    {
        let mut flags = out_header.imod_flags as u32;
        set_or_clear_flags(
            &mut flags,
            MRC_FLAGS_CTF_CORRECTED,
            if (skip_done_flag & 2) != 0 { 0 } else { 1 },
        );
        out_header.imod_flags = flags as i32;
    }

    if (starting_view == -1 && ending_view == -1) || is_single_run {
        if !is_single_run && b3d_output_file_type() == OUTPUT_TYPE_TIFF {
            exit_error(b"Cannot do parallel writing to a TIFF output file");
        }
        imod_backup_file(&String::from_utf8_lossy(&out_fn));
        foutput = ii_fopen(&out_fn, "wb");
    } else if !parallel_hdf {
        foutput = ii_fopen(&out_fn, "r+b");
        if foutput.is_none() {
            exit_error(&c_format_bytes(
                "fopen() failed to open %s",
                &[CArg::Bytes(&out_fn)],
            ));
        }
    }

    // Starting run of parallel run: write header and exit
    if starting_view == -1 && ending_view == -1 && !is_single_run {
        if parallel_hdf {
            let mut fout = foutput.clone().expect("opened above");
            ii_file = ii_lookup_file_from_fp(&fout).unwrap_or(std::ptr::null_mut());
            if !ii_file.is_null() {
                unsafe { ii_sync_from_mrc_header(&mut *ii_file, &mut out_header) };
            }
            if ii_file.is_null() || unsafe { (*ii_file).file } != IIFILE_HDF {
                exit_error(&c_format_bytes(
                    "Expected an HDF file but new file %s is not HDF",
                    &[CArg::Bytes(&out_fn)],
                ));
            }
            if unsafe { hdf_write_dummy_section(ii_file, bound_fn.as_mut_ptr(), 0) } != 0 {
                exit_error(b"Writing dummy dataset for first section to HDF file");
            }
            let _ = &mut fout;
        }
        let mut fout = foutput.expect("opened above");
        if mrc_head_write(&mut fout, &mut out_header) != 0 {
            exit_error(b"Error when write out header");
        }
        ii_fclose(&mut fp_stack);
        ii_fclose(&mut fout);
        let _ = ImodFile::Stdout.flush();
        return 0;
    }

    let err = par_wrt_initialize(
        &String::from_utf8_lossy(&bound_fn),
        header.nx * if parallel_hdf { -1 } else { 1 },
        header.ny,
    );
    if err != 0 {
        exit_error(&c_format_bytes(
            "Initializing parallel writing with boundary info file %s (error %d)",
            &[CArg::Bytes(&bound_fn), CArg::Int(err.into())],
        ));
    }
    if parallel_hdf {
        let mut lock_index = 0i32;
        let mut scratch = 0i32;
        let mut scratch2 = 0i32;
        par_wrt_properties(&mut lock_index, &mut scratch, &mut scratch2);
        if b3d_lock_file(lock_index) != 0 {
            exit_error(b"Failed to obtain initial lock on HDF file");
        }
        foutput = ii_fopen(&out_fn, "r+b");
        if foutput.is_none() {
            b3d_unlock_file(lock_index);
            exit_error(&c_format_bytes(
                "iiFOpen() failed to open %s",
                &[CArg::Bytes(&out_fn)],
            ));
        }
        ii_file = ii_lookup_file_from_fp(foutput.as_ref().expect("opened above"))
            .unwrap_or(std::ptr::null_mut());
        if ii_file.is_null() || unsafe { (*ii_file).file } != IIFILE_HDF {
            exit_error(&c_format_bytes(
                "Expected an HDF file but existing file %s is not HDF",
                &[CArg::Bytes(&out_fn)],
            ));
        }
        if unsafe { par_wrt_reclose_hdf(ii_file, std::ptr::null_mut()) } != 0 {
            exit_error(b"Closing or unlocking HDF file\n");
        }
        foutput = Some(ImodFile::Token(ii_file as usize));
    }

    let nx = header.nx;
    let nyfile = header.ny;
    let nz = header.nz;
    let mut curr_angle: f32;
    let mut strip_pixel_num = 0i32;
    let mut inter_pixel_num: i32;
    let mut k: i32 = 0;
    let mut view: i32;
    let mut nx_pad: i32;
    let mut full_xdim = 0i32;
    let mut width_for_scaling: i32;
    let mut max_copy_pixels: i32;
    let mut strip_ind: i32;
    let mut ny: i32;
    let mut yoff: i32;
    let mut xoff: i32;
    let mut cur_offset: i32 = 0;
    let mut last_offset: i32 = 0;
    let wl: f32;
    let c1: f32;
    let c2: f32;
    let mut f2: f32;
    let mut ctf: f32;
    let mut freq_scalex: f32;
    let mut freq_scaley: f32;
    let mut fy_component: f32;
    let mut freq_scale_xsq: f32;
    let mut wave_aberration: f32;
    let mut amp_angle: f32;
    let mut atten_frac: f32;
    let mut first_zero_freq_sq: f32;
    let mut min_atten_freq_sq: f32;
    let mut phase_shift: f32;
    let mut denom: f32;
    let mut cos_sum: f32;
    let mut gx: f32;
    let mut gy: f32;
    let mut point_defocus: f32;
    let mut focus_sum: f32;
    let mut focus_diff: f32;
    let mut cuton_to_use: f32;
    let mut sin_astig: f32;
    let mut cos_astig: f32;
    let mut strip_focus2: f32;
    let mut phase_frac: f32;
    let mut phase_frac_factor = 0.0f32;
    let mut cuton_angstroms = 0.0f32;
    let mut inter_zero = 0.0f32;
    let mut last_zero = 0.0f32;
    let mut cur_zero: f32;
    let do_scale_by_power = scale_by_power > 0.;
    let power_is_half = (scale_by_power as f64 - 0.5).abs() < 1.0e-5;
    let general_power = !power_is_half && (scale_by_power as f64 - 1.).abs() > 1.0e-5;
    let have_astig = (def_flags & DEF_FILE_HAS_ASTIG) != 0;
    let have_phase = (def_flags & DEF_FILE_HAS_PHASE) != 0;
    let have_cuton = (def_flags & DEF_FILE_HAS_CUT_ON) != 0;

    //Get the tilt angles and detected defocus for each slice;
    let mut defocus = vec![0.0f32; nz.max(0) as usize];
    let mut defocus2 = vec![0.0f32; nz.max(0) as usize];
    let mut astig_angle = vec![0.0f32; nz.max(0) as usize];
    let mut plate_phase = vec![0.0f32; nz.max(0) as usize];
    let mut cuton_freq = vec![0.0f32; nz.max(0) as usize];
    let mut cur_frac = vec![0.0f32; nx.max(0) as usize];
    let mut last_frac = vec![0.0f32; nx.max(0) as usize];
    let mut wall_prep = 0.0f64;
    let mut wall_fft = 0.0f64;
    let mut wall_corr = 0.0f64;
    let mut wall_interp = 0.0f64;
    let mut wall_start = 0.0f64;
    let mut cur_time;

    if have_angle_fn {
        tilt_angles = read_tilt_angles(
            &angle_fn,
            nz,
            angle_sign as f32,
            &mut min_angle,
            &mut max_angle,
        );
        have_tilt_angles = true;
    }

    // Check the defocus list if there is more than one value
    if defocus_list.len() > 1
        && check_and_fix_defocus_list(
            &mut defocus_list,
            if have_tilt_angles {
                Some(&tilt_angles)
            } else {
                None
            },
            nz,
            def_version,
        ) != 0
    {
        let _ = ImodFile::Stdout.write_all(
            b"WARNING: ctfphaseflip - View numbers in defocus file are not all consistent with the angular ranges\n",
        );
    }

    //sets to UNUSED_DEFOCUS since ctfplotter saves defocus with that value
    // when defocus is not computed.
    // Also detect if all angles are 0 and
    ierr = 1;
    for k in 0..nz {
        defocus[k as usize] = UNUSED_DEFOCUS;
        defocus2[k as usize] = UNUSED_DEFOCUS;
        astig_angle[k as usize] = UNUSED_DEFOCUS;
        plate_phase[k as usize] = UNUSED_DEFOCUS;
        cuton_freq[k as usize] = UNUSED_DEFOCUS;
        if have_angle_fn && (tilt_angles[k as usize] as f64).abs() > MIN_ANGLE {
            ierr = 0;
        }
    }
    if ierr != 0 {
        do_full_images = true;
        max_strip_width = nx;
        let _ = ImodFile::Stdout
            .write_all(b"All tilt angles are 0, doing single correction of each full image\n");
    }

    // Process the defocus values; slice numbers are now numbered from 0
    for i in 0..defocus_list.len() {
        let item = defocus_list[i];
        k = (item.starting_slice + item.ending_slice) / 2;
        if k < 0 || k >= nz {
            exit_error(b"View numbers in defocus file are out of range");
        }
        // They are already in microns
        defocus[k as usize] = item.defocus as f32;
        if item.defocus2 != 0. {
            defocus2[k as usize] = item.defocus2 as f32;
            astig_angle[k as usize] = item.astig_angle as f32;
        }
        if item.plate_phase != 0. {
            plate_phase[k as usize] = item.plate_phase.abs() as f32;
        }
        if item.cut_on_freq > 0. {
            cuton_freq[k as usize] = item.cut_on_freq as f32;
        }
    }

    // interpolation for defocus, astig angle, and phase shifts can use the function
    interpolate_table(&mut defocus, nz, false);
    if have_astig {
        interpolate_table(&mut astig_angle, nz, true);
    }
    if have_phase {
        interpolate_table(&mut plate_phase, nz, false);
    }
    if have_cuton {
        interpolate_table(&mut cuton_freq, nz, false);
    }

    // But if there are any astigmatism entries, we need to fill in defocus values to
    // interpolate the astigmatism amplitude between existing information for that
    if have_astig {
        let mut first: i32 = -1;
        let mut second: i32 = 0;
        for k in 0..nz {
            if astig_angle[k as usize] != UNUSED_DEFOCUS {
                astig_angle[k as usize] += rotations[k as usize];
                if astig_angle[k as usize] < -90. {
                    astig_angle[k as usize] += 180.;
                } else if astig_angle[k as usize] > 90. {
                    astig_angle[k as usize] = (astig_angle[k as usize] as f64 - 180.) as f32;
                }
            }
            if defocus2[k as usize] == UNUSED_DEFOCUS {
                continue;
            }
            second = k;
            if first == -1 {
                for row in 0..second {
                    defocus2[row as usize] = defocus[row as usize]
                        * (defocus2[second as usize] / defocus[second as usize]);
                }
            } else {
                for row in first + 1..second {
                    defocus2[row as usize] = defocus[row as usize]
                        * ((row - first) as f32
                            * (defocus2[second as usize] / defocus[second as usize])
                            + (second - row) as f32
                                * (defocus2[first as usize] / defocus[first as usize]))
                        / (second - first) as f32;
                }
            }
            first = second;
        }

        for k in (0..nz).rev() {
            if defocus2[k as usize] == UNUSED_DEFOCUS {
                defocus2[k as usize] =
                    defocus[k as usize] * (defocus2[second as usize] / defocus[second as usize]);
            } else {
                break;
            }
        }
    }

    // Report the angles, after applying the offset if any
    offset_in_z = (offset_in_z as f64 * 0.001 * pixel_size as f64) as f32;
    focus_diff = offset_in_z;
    for k in 0..nz {
        if offset_in_z != 0. {
            // The rationale for the sign is that the right side of an aligned image is both
            // lower in the scope and at a higher underfocus according to the man page on
            // invertAngles, and lower in the scope should be lower in tomogram.  Thus
            // negative Z offset in the tomogram should be higher underfocus
            // The vertical (z) distance between tilted planes is the hypotenuse of a triangle
            // with the side perpendicular to the planes (the entered offset) adjacent to
            // the angle, hence the cosine here.
            if have_tilt_angles {
                focus_diff = (offset_in_z as f64
                    / (RADIANS_PER_DEGREE * tilt_angles[k as usize] as f64).cos())
                    as f32;
            }
            defocus[k as usize] -= focus_diff;
            if have_astig && astig_angle[k as usize] != UNUSED_DEFOCUS {
                defocus2[k as usize] -= focus_diff;
            }
            if hush_defoci == 0 {
                let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                    "adjusted defocus[%d] = %f microns",
                    &[CArg::Int(k.into()), CArg::Dbl(defocus[k as usize] as f64)],
                ));
            }
        } else if hush_defoci == 0 {
            let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                "defocus[%d] = %f microns",
                &[CArg::Int(k.into()), CArg::Dbl(defocus[k as usize] as f64)],
            ));
        }
        if have_astig && hush_defoci == 0 && astig_angle[k as usize] != UNUSED_DEFOCUS {
            let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                "   defocus2 = %f   angle = %.2f",
                &[
                    CArg::Dbl(defocus2[k as usize] as f64),
                    CArg::Dbl(astig_angle[k as usize] as f64),
                ],
            ));
        }
        if have_phase && hush_defoci == 0 {
            let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                "   phase shift = %f",
                &[CArg::Dbl(plate_phase[k as usize] as f64)],
            ));
        }
        if hush_defoci == 0 {
            let _ = ImodFile::Stdout.write_all(b"\n");
        }
    }

    wl = (12.3984 / (volt as f64 * (volt as f64 + 1022.0)).sqrt()) as f32; //wavelength in Angstroms;
    c1 = (MY_PI * wl as f64) as f32;
    c2 = (c1 as f64 * cs as f64 * 1.0e7 * wl as f64 * wl as f64 / 2.0) as f32; // All in Angstroms

    let mut strip_dist = [0i32; 2];
    let mut strip_limit: i32;
    let mut full_image: Vec<f32> = Vec::new();
    let mut full_copy: [Vec<f32>; 2] = [Vec::new(), Vec::new()];
    let mut have_full_arrays = false;
    let mut mean_sum = 0.0f64;

    let mut amin = 0.1 * f64::MAX;
    let mut amax = -amin;
    let mut nice_limit = nice_fft_limit();
    // if (useGPU >= 0)
    nice_limit = 5;
    let _ = nice_limit;

    // Pad the extent in Y if necessary
    ny = nice_frame(nyfile, 2, nice_limit);
    yoff = (ny - nyfile) / 2;
    xoff = 0;
    let mut curr_k = 0i32;
    if !is_single_run {
        curr_k = starting_view - starting_total;
    }
    nx_pad = nice_frame(nx, 2, nice_limit);
    if do_full_images {
        full_xdim = nx_pad + 2;
        xoff = (nx_pad - nx) / 2;
    }
    let _ = ImodFile::Stdout.flush();

    view = starting_view;
    while view <= ending_view {
        if have_tilt_angles {
            curr_angle = tilt_angles[(view - 1) as usize];
            let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                "Slice %d, tilt angle is %.2f degrees. \n",
                &[CArg::Int(view.into()), CArg::Dbl(curr_angle as f64)],
            ));
        } else {
            curr_angle = 0.0;
            let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                "Slice %d, no angle is specified, set to 0.0\n",
                &[CArg::Int(view.into())],
            ));
        }

        if defocus[(view - 1) as usize] == UNUSED_DEFOCUS {
            exit_error(&c_format_bytes(
                "specified defocus is wrong for slice %d",
                &[CArg::Int(view.into())],
            ));
        }
        amp_angle = (amp_contrast as f64 / (1. - (amp_contrast * amp_contrast) as f64).sqrt())
            .atan() as f32;

        // Assign phase shift from table or entered value
        if have_phase {
            phase_shift = plate_phase[(view - 1) as usize];
        } else {
            phase_shift = phase_plate_shift;
        }
        if phase_shift == UNUSED_DEFOCUS {
            exit_error(&c_format_bytes(
                "specified phase shift is wrong for slice %d",
                &[CArg::Int(view.into())],
            ));
        }

        // Assign cut-on frequency from table or entered value
        if have_cuton {
            cuton_to_use = cuton_freq[(view - 1) as usize];
        } else {
            cuton_to_use = cut_on_entered;
        }
        if cuton_to_use == UNUSED_DEFOCUS {
            exit_error(&c_format_bytes(
                "specified cut-on frequency is wrong for slice %d",
                &[CArg::Int(view.into())],
            ));
        }
        phase_frac = 1.;
        if cuton_to_use > 0. {
            cuton_angstroms = (cuton_to_use as f64 / 10.) as f32;
            phase_frac_factor = (1. / (1. - (-FREQ_FOR_PHASE / cuton_to_use as f64).exp())) as f32;
            let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                "phaseFracFactor %f  coxpx2 %f\n",
                &[
                    CArg::Dbl(phase_frac_factor as f64),
                    CArg::Dbl(cuton_angstroms as f64),
                ],
            ));
        }

        let Some(mut curr_slice) = slice_create(nx, nyfile, slice_mode) else {
            exit_error(b"creating outslice or currSlice");
        };
        let Some(mut out_slice) = slice_create(nx, nyfile, SLICE_MODE_FLOAT) else {
            exit_error(b"creating outslice or currSlice");
        };

        //startingView starts at 1, the API starts 0;
        if mrc_read_slice(
            curr_slice.data.bytes_mut(),
            &mut fp_stack,
            &mut header,
            view - 1,
            b'Z',
        ) != 0
        {
            exit_error(b"reading slice");
        }

        //convert slice to floats
        if slice_mode != SLICE_MODE_FLOAT && slice_new_mode(&mut curr_slice, SLICE_MODE_FLOAT) < 0 {
            exit_error(b"converting slice to float");
        }

        angle_is_zero =
            (curr_angle as f64).abs() <= MIN_ANGLE && (!max_was_entered || do_full_images);
        curr_angle = (curr_angle as f64 * MY_PI / 180.0) as f32;
        if !angle_is_zero {
            let denominator = {
                let a = (curr_angle as f64).abs();
                let b = MIN_ANGLE * MY_PI / 180.;
                if a > b { a } else { b }
            };
            strip_pixel_num =
                2 * ((defocus_tol as f64 / denominator.tan() / pixel_size as f64) as i32 / 2);
            max_copy_pixels = strip_pixel_num / 2;
        } else {
            strip_pixel_num = nx;
            max_copy_pixels = nx;
        }
        if strip_pixel_num > nx_pad {
            strip_pixel_num = nx_pad;
        }

        // Evaluate a minimum strip width that allows some criterion pixels between zeros
        for i in 1..100 {
            cur_zero = get_any_one_zero(
                defocus[(view - 1) as usize] as f64,
                phase_shift as f64,
                i,
                amp_contrast as f64,
                cs as f64,
                pixel_size as f64,
                volt as f64,
            ) as f32;
            if i > 1 {
                inter_zero = cur_zero - last_zero;
                if cur_zero > freq_for_inter_zero {
                    break;
                }
            }
            last_zero = cur_zero;
        }
        min_width = (min_inter_zero_pixels as f64 * 2. / inter_zero as f64) as i32;
        // `B3DCLAMP(v, lo, hi)` is `MAX(lo, MIN(hi, v))`.
        min_width = min_strip_width.max(nx.min(min_width));

        // If the defocus tolerance allows a width bigger than the basic amount, and no
        // specific maximum width was entered, find the first maximum width that gives
        // sufficient zero shift across the whole width of a strip to eliminate rings at the
        // zeros
        max_width = max_strip_width;
        if angle_is_zero {
            max_width = nx;
        } else if !max_was_entered && strip_pixel_num > min_width_to_scale_interp {
            max_width = min_width_to_scale_interp;
            zero_shift = 0.5
                * max_width as f64
                * first_zero_shift(
                    defocus[(view - 1) as usize] as f64,
                    phase_shift as f64,
                    curr_angle as f64,
                    max_width,
                    amp_contrast as f64,
                    cs as f64,
                    pixel_size as f64,
                    volt as f64,
                );
            while max_width < max_strip_width
                && zero_shift < (1. - (curr_angle as f64).tan().abs()) * min_zero_shift as f64
            {
                max_width += 2;
                zero_shift = 0.5
                    * max_width as f64
                    * first_zero_shift(
                        defocus[(view - 1) as usize] as f64,
                        phase_shift as f64,
                        curr_angle as f64,
                        max_width,
                        amp_contrast as f64,
                        cs as f64,
                        pixel_size as f64,
                        volt as f64,
                    );
            }
        }

        // Scale the spacing between strips if this makes width be bigger than basic amount
        width_for_scaling = if strip_pixel_num < max_width {
            strip_pixel_num
        } else {
            max_width
        };
        inter_pixel_num = i_width;
        if angle_is_zero {
            inter_pixel_num = nx;
        } else if i_width > 1 && scale_interp && width_for_scaling > min_width_to_scale_interp {
            let a = (width_for_scaling * i_width) / min_width_to_scale_interp;
            let b = strip_pixel_num / 2 - 1;
            inter_pixel_num = if a < b { a } else { b };
        }

        // Now adjust either the min if max entered, or the max if it was not
        if max_was_entered {
            min_width = if min_width < max_width {
                min_width
            } else {
                max_width
            };
        } else {
            max_width = if min_width > max_width {
                min_width
            } else {
                max_width
            };
        }

        // Limit the strip width then get it to a nice size
        strip_pixel_num = min_width.max(max_width.min(strip_pixel_num));
        strip_pixel_num = nice_frame(strip_pixel_num, 2, nice_limit);

        // If that makes it too big, try stepping down with nice limit, and if it is then
        // enough smaller than the minimum, step down and allow much bigger prime factors
        strip_limit = if do_full_images || angle_is_zero {
            nx_pad
        } else {
            nx
        };
        if strip_pixel_num > strip_limit {
            strip_pixel_num = nice_frame(strip_limit, -2, nice_limit);
            if (strip_pixel_num as f64) < 0.9 * min_width as f64 {
                strip_pixel_num = nice_frame(strip_limit, -2, 19);
            }
        }
        zero_shift = 0.5
            * strip_pixel_num as f64
            * first_zero_shift(
                defocus[(view - 1) as usize] as f64,
                phase_shift as f64,
                curr_angle as f64,
                strip_pixel_num,
                amp_contrast as f64,
                cs as f64,
                pixel_size as f64,
                volt as f64,
            );

        //interPixelNum must be less than stripPixelNum/2;
        if inter_pixel_num >= strip_pixel_num / 2 && !angle_is_zero {
            exit_error(&c_format_bytes(
                "Interpolation width is too high, must be less than %d",
                &[CArg::Int((min_strip_width / 2).into())],
            ));
        }

        // All of the above was needed to get dynamic interpolation width, now set for full
        if do_full_images {
            strip_pixel_num = nx_pad;
        }
        let _ = ImodFile::Stdout.write_all(&c_format_bytes(
            "stripPixelNum=%d interPixelNum=%d zeroShift=%f inter-strip shift=%f\n",
            &[
                CArg::Int(strip_pixel_num.into()),
                CArg::Int(inter_pixel_num.into()),
                CArg::Dbl(zero_shift),
                CArg::Dbl(first_zero_shift(
                    defocus[(view - 1) as usize] as f64,
                    phase_shift as f64,
                    curr_angle as f64,
                    inter_pixel_num,
                    amp_contrast as f64,
                    cs as f64,
                    pixel_size as f64,
                    volt as f64,
                )),
            ],
        ));

        let strip_xdim = strip_pixel_num + 2;
        let mut finished = false;
        let mut do_right_shifted_mid = false;
        let mut in_right_shifted_mid = false;
        let view_has_astig = have_astig && astig_angle[(view - 1) as usize] != UNUSED_DEFOCUS;
        let mut strip: Vec<f32> = Vec::new();
        let mut strip_begin = 0i32;
        let mut full_first_mid = 0i32;
        let mut intervals;
        let mut num_left_shifted_mid = 0i32;
        let mut strip_end = 0i32;
        let mut strip_stride;
        let mut half_strip;
        let mut strip_mid = 0i32;
        let mut effective_nx: i32;
        let mut effective_xcen: f32;
        let mut x_pix_center = 0.0f32;
        let mut y_pix_center = 0.0f32;
        let mut view_axis_angle;
        let cornerdist1: f32;
        let cornerdist2: f32;
        let mut sin_view_axis = 0.0f32;
        let mut cos_view_axis = 0.0f32;
        let mut low_lim: f32;
        let mut high_lim: f32;
        let mut axis_dist: f32;
        let mut cur_axis_dist: f32;
        let mut last_axis_dist: f32;
        let mut cur_ax_frac: f32;
        let mut delta_z: f32;
        let mut constant_zaxis = 0.0f32;

        effective_nx = nx;
        effective_xcen = (nx / 2) as f32;

        // Do initial operations on GPU so that if this one fails, it can fall back
        if use_gpu >= 0
            && gpu_initialize_slice(
                curr_slice.data.f(),
                nx,
                nyfile,
                strip_xdim,
                strip_pixel_num,
                ny,
                do_full_images,
            ) != 0
        {
            ierr = if if_gpu_by_env != 0 {
                act_gpu_fail_environ
            } else {
                act_gpu_fail_option
            };
            if ierr > 1 {
                exit_error(&c_format_bytes(
                    "Failure in initial call to GPU for view %d",
                    &[CArg::Int(view.into())],
                ));
            }
            let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                "%sFailure in initial call to GPU for view %d; falling back to CPU\n",
                &[
                    CArg::Str(if ierr != 0 {
                        "MESSAGE: Ctfphaseflip - "
                    } else {
                        ""
                    }),
                    CArg::Int(view.into()),
                ],
            ));
            use_gpu = -1;
        }

        //Allocate 2 strips, even and odd strip;
        if !do_full_images {
            xoff = 0;
            if use_gpu < 0 {
                strip = vec![0.0f32; (2 * ny * strip_xdim) as usize];
            }
            if !angle_is_zero {
                num_left_shifted_mid = ((strip_pixel_num / 2 - max_copy_pixels) + inter_pixel_num
                    - 1)
                    / inter_pixel_num;
                // `ACCUM_MAX(a, b)` is `a = a > b ? a : b`.
                num_left_shifted_mid = if num_left_shifted_mid > 0 {
                    num_left_shifted_mid
                } else {
                    0
                };
                while strip_pixel_num / 2 - num_left_shifted_mid * inter_pixel_num < 2 {
                    num_left_shifted_mid -= 1;
                }
            }
        } else {
            // Or taper and pad full image and take its FFT
            if use_gpu < 0 {
                // First time this is encountered, allocate the arrays - do it this way
                // in case started on GPU and fell back
                if !have_full_arrays {
                    full_image = vec![0.0f32; (ny * full_xdim) as usize];
                    full_copy[0] = vec![0.0f32; (ny * full_xdim) as usize];
                    full_copy[1] = vec![0.0f32; (ny * full_xdim) as usize];
                    have_full_arrays = true;
                }
                if debug_mode != 0 {
                    wall_start = wall_time();
                }
                slice_taper_in_pad(
                    PadIn::Float(curr_slice.data.f()),
                    SLICE_MODE_FLOAT,
                    nx,
                    0,
                    nx - 1,
                    0,
                    nyfile - 1,
                    &mut full_image,
                    full_xdim,
                    nx_pad,
                    ny,
                    9,
                    9,
                );
                if debug_mode != 0 {
                    cur_time = wall_time();
                    wall_prep += cur_time - wall_start;
                    wall_start = cur_time;
                }
                let _ = todfftc(&mut full_image, nx_pad, ny, 0);
                if debug_mode != 0 {
                    cur_time = wall_time();
                    wall_fft += cur_time - wall_start;
                    wall_start = cur_time;
                }
            }

            // Determine how to do full image along diagonals
            // Angles are already radians
            if do_diagonals {
                x_pix_center = (nx as f64 / 2. - 0.5) as f32;
                y_pix_center = (nyfile as f64 / 2. - 0.5) as f32;

                // Get the axis of constant Z before rotation by the tilt axis angle
                // Keep it in the 1st and 2nd quadrant, on both sides of tilt axis
                if x_axis_tilt != 0. {
                    constant_zaxis =
                        ((curr_angle as f64).sin() / (x_axis_tilt as f64).tan()).atan() as f32;
                    if constant_zaxis < 0. {
                        constant_zaxis = (constant_zaxis as f64 + 180. * RADIANS_PER_DEGREE) as f32;
                    }
                } else {
                    constant_zaxis = (90. * RADIANS_PER_DEGREE) as f32;
                }

                // Add tilt axis angle to get actual axis in image
                view_axis_angle = constant_zaxis + tilt_axis_angle;
                sin_view_axis = (view_axis_angle as f64).sin() as f32;
                cos_view_axis = (view_axis_angle as f64).cos() as f32;
                cornerdist1 = (-sin_view_axis as f64 * nx as f64 / 2.
                    - cos_view_axis as f64 * nyfile as f64 / 2.)
                    .abs() as f32;
                cornerdist2 = (-sin_view_axis as f64 * nx as f64 / 2.
                    + cos_view_axis as f64 * nyfile as f64 / 2.)
                    .abs() as f32;
                effective_nx = 2
                    * (if cornerdist1 > cornerdist2 {
                        cornerdist1
                    } else {
                        cornerdist2
                    } as f64)
                        .ceil() as i32;
                effective_xcen = (effective_nx as f64 / 2. - 0.5) as f32;
                let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                    "Axis of constant Z: %.2f\n",
                    &[CArg::Dbl(view_axis_angle as f64 / RADIANS_PER_DEGREE)],
                ));
            }

            intervals = (effective_nx - 4) / inter_pixel_num;
            full_first_mid = (effective_nx - intervals * inter_pixel_num) / 2;
            if angle_is_zero && !do_diagonals {
                full_first_mid = nx / 2;
            }
            let _ = intervals;
        }

        // convert pixelSize to Angstroms from nm to scale frequency to 1/A
        freq_scalex = (1.0 / (pixel_size as f64 * 10.0 * strip_pixel_num as f64)) as f32;
        freq_scale_xsq = freq_scalex * freq_scalex;
        freq_scaley = (1.0 / (pixel_size as f64 * 10.0 * ny as f64)) as f32;
        strip_ind = 0;
        while !finished {
            /*
             * Set up the strip variables based on the current index
             */
            if do_full_images {
                // Full image, shifting stripMid
                strip_mid = strip_ind * inter_pixel_num + full_first_mid;
                finished = strip_mid + inter_pixel_num > effective_nx - 4;
                strip_stride = inter_pixel_num;
                half_strip = strip_mid + if strip_ind == 0 { 1 } else { 0 };
            } else {
                // Also full image, just one operation
                if angle_is_zero {
                    strip_begin = 0;
                    strip_stride = inter_pixel_num;
                    strip_end = nx - 1;
                    finished = true;
                    xoff = (strip_pixel_num - nx) / 2;
                    strip_mid = (strip_begin + strip_end) / 2;
                    half_strip = strip_pixel_num / 2;

                // Left side matching strips with different middles, offset is zero
                } else if strip_ind <= num_left_shifted_mid {
                    strip_begin = 0;
                    strip_end = if nx - 1 < strip_pixel_num - 1 {
                        nx - 1
                    } else {
                        strip_pixel_num - 1
                    };
                    strip_stride = inter_pixel_num;
                    strip_mid =
                        strip_pixel_num / 2 - (num_left_shifted_mid - strip_ind) * inter_pixel_num;
                    half_strip = strip_mid + if strip_ind == 0 { 1 } else { 0 };

                // Classic region with stripMid in middle of strip that is shifted from last
                } else if (strip_ind - num_left_shifted_mid) * inter_pixel_num + strip_pixel_num - 1
                    < nx
                {
                    let candidate = (strip_ind - num_left_shifted_mid) * inter_pixel_num;
                    strip_begin = if 0 > candidate { 0 } else { candidate };
                    strip_end = if nx - 1 < strip_begin + strip_pixel_num - 1 {
                        nx - 1
                    } else {
                        strip_begin + strip_pixel_num - 1
                    };
                    strip_stride = inter_pixel_num;
                    strip_mid = (strip_begin + strip_end) / 2;
                    half_strip = strip_pixel_num / 2;
                    if strip_end == nx - 1 {
                        do_right_shifted_mid = true;
                        finished = nx - strip_mid <= max_copy_pixels
                            || strip_mid + inter_pixel_num > nx - 2;
                    }

                // In right side matching strips with different middles
                } else if do_right_shifted_mid {
                    strip_stride = inter_pixel_num;
                    strip_mid += inter_pixel_num;
                    half_strip = strip_mid;
                    in_right_shifted_mid = true;
                    xoff = -strip_begin;
                    finished =
                        nx - strip_mid <= max_copy_pixels || strip_mid + inter_pixel_num > nx - 2;

                // Or set up last possible strip with unique stride from last
                } else {
                    strip_stride = (nx - strip_pixel_num) - strip_begin;
                    strip_begin = if 0 > nx - strip_pixel_num {
                        0
                    } else {
                        nx - strip_pixel_num
                    };
                    strip_end = nx - 1;
                    strip_mid = (strip_begin + strip_end) / 2;
                    half_strip = strip_pixel_num / 2;
                    do_right_shifted_mid = true;
                    finished =
                        nx - strip_mid <= max_copy_pixels || strip_mid + inter_pixel_num > nx - 2;
                }
                if debug_mode > 1 {
                    let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                        "stripInd=%d stripBegin=%d stripEnd=%d   ",
                        &[
                            CArg::Int(strip_ind.into()),
                            CArg::Int(strip_begin.into()),
                            CArg::Int(strip_end.into()),
                        ],
                    ));
                }
            }

            // Get a change in Z height and a defocus in nm;
            if do_diagonals {
                // For diagonals, the axis distance is based on the strip coordinate;
                // handle the special case where the constant Z axis is at 0, and apply
                // formula for axis distance to get change in Z height
                axis_dist = (strip_mid as f32 - (effective_xcen + x_shifts[(view - 1) as usize]))
                    * pixel_size;
                if x_axis_tilt != 0.
                    && ((curr_angle as f64).abs() <= MIN_ANGLE * RADIANS_PER_DEGREE
                        || (constant_zaxis as f64).abs() < 1.0e-6)
                {
                    delta_z = (axis_dist as f64 * (x_axis_tilt as f64).tan()) as f32;
                } else {
                    delta_z = (axis_dist as f64 * (curr_angle as f64).tan()
                        / (constant_zaxis as f64).sin()) as f32;
                }
            } else {
                // For regular tilt, the axis distance is negative x at positive tilt
                delta_z = ((((nx / 2) as f32 + x_shifts[(view - 1) as usize] - strip_mid as f32)
                    * pixel_size) as f64
                    * (curr_angle as f64).tan()) as f32;
            }

            // The defocus is higher with negative deltaZ, so subtract it
            strip_defocus = (defocus[(view - 1) as usize] as f64 * 1000.0 - delta_z as f64) as f32;

            //convert defocus to Angstroms
            strip_defocus = (strip_defocus as f64 * 10.) as f32;
            point_defocus = strip_defocus;

            // For astigmatism, get the other defocus and the sine and cosine of angle
            cos_astig = 0.;
            sin_astig = 0.;
            focus_sum = 0.;
            focus_diff = 0.;
            if view_has_astig {
                strip_focus2 =
                    ((defocus2[(view - 1) as usize] as f64 * 1000.0 - delta_z as f64) * 10.) as f32;
                sin_astig =
                    (astig_angle[(view - 1) as usize] as f64 * RADIANS_PER_DEGREE).sin() as f32;
                cos_astig =
                    (astig_angle[(view - 1) as usize] as f64 * RADIANS_PER_DEGREE).cos() as f32;
                focus_sum = (0.5 * (strip_defocus as f64 + strip_focus2 as f64)) as f32;
                focus_diff = (0.5 * (strip_defocus as f64 - strip_focus2 as f64)) as f32;
            }

            // Get the first zero and frequency to start attenuation at (if any) as a square
            first_zero_freq_sq = (get_any_one_zero(
                strip_defocus as f64 / 10000.,
                phase_shift as f64,
                1,
                amp_contrast as f64,
                cs as f64,
                pixel_size as f64,
                volt as f64,
            ) / (20. * pixel_size as f64))
                .powf(2.) as f32;
            min_atten_freq_sq = atten_start_frac * first_zero_freq_sq;

            if use_gpu < 0 {
                // Copy full image transform or taper-pad strip to curStrip and transform it
                {
                    let (first, second) = full_copy.split_at_mut(1);
                    let (cur_full, _last_full) = if strip_ind % 2 != 0 {
                        (&mut second[0], &mut first[0])
                    } else {
                        (&mut first[0], &mut second[0])
                    };
                    if do_full_images {
                        cur_full.copy_from_slice(&full_image[..(full_xdim * ny) as usize]);
                    }
                }
                if !do_full_images {
                    if debug_mode != 0 {
                        wall_start = wall_time();
                    }
                    let half = (ny * strip_xdim) as usize;
                    let (first, second) = strip.split_at_mut(half);
                    let cur_strip = if strip_ind % 2 != 0 { second } else { first };
                    slice_taper_in_pad(
                        PadIn::Float(curr_slice.data.f()),
                        SLICE_MODE_FLOAT,
                        nx,
                        strip_begin,
                        strip_end,
                        0,
                        nyfile - 1,
                        cur_strip,
                        strip_pixel_num + 2,
                        strip_pixel_num,
                        ny,
                        9,
                        9,
                    );
                    if debug_mode != 0 {
                        cur_time = wall_time();
                        wall_prep += cur_time - wall_start;
                        wall_start = cur_time;
                    }
                    let _ = todfftc(cur_strip, strip_pixel_num, ny, 0);
                    if debug_mode != 0 {
                        cur_time = wall_time();
                        wall_fft += cur_time - wall_start;
                        wall_start = cur_time;
                    }
                }

                // Correct the CTF
                {
                    let half = (ny * strip_xdim) as usize;
                    let cur_strip: &mut [f32] = if do_full_images {
                        let (first, second) = full_copy.split_at_mut(1);
                        if strip_ind % 2 != 0 {
                            &mut second[0]
                        } else {
                            &mut first[0]
                        }
                    } else {
                        let (first, second) = strip.split_at_mut(half);
                        if strip_ind % 2 != 0 { second } else { first }
                    };
                    for fy in 0..ny {
                        let mut fyy = fy;
                        if fy > ny / 2 {
                            fyy -= ny;
                        }
                        gy = fyy as f32 * freq_scaley;
                        fy_component = gy * gy;
                        for fx in 0..strip_xdim / 2 {
                            f2 = (fx * fx) as f32 * freq_scale_xsq + fy_component;
                            if view_has_astig && (fx != 0 || fy != 0) {
                                // The equation here is from Rohou and Grigorieff, 2015
                                gx = fx as f32 * freq_scalex;
                                denom = ((gx * gx + fy_component) as f64).sqrt() as f32;
                                cos_sum = (cos_astig * gx + sin_astig * gy) / denom;
                                point_defocus = (focus_sum as f64
                                    + focus_diff as f64
                                        * (2. * cos_sum as f64 * cos_sum as f64 - 1.))
                                    as f32;
                            }
                            if cuton_to_use > 0. {
                                phase_frac = (phase_frac_factor as f64
                                    * (1. - ((-f2.sqrt() / cuton_angstroms) as f64).exp()))
                                    as f32;
                            }
                            wave_aberration =
                                (c2 * f2 - c1 * point_defocus) * f2 - phase_frac * phase_shift;

                            // Produce a positive ctf for consistency and so it can be used
                            // for scaling
                            ctf = -((wave_aberration - amp_angle) as f64).sin() as f32;
                            if do_scale_by_power && f2 > min_atten_freq_sq {
                                if power_is_half {
                                    ctf = ((ctf as f64).abs().sqrt()
                                        * if ctf >= 0. { 1. } else { -1. })
                                        as f32;
                                } else if general_power {
                                    ctf = ((ctf as f64).abs().powf(scale_by_power as f64)
                                        * if ctf >= 0. { 1. } else { -1. })
                                        as f32;
                                }
                                if f2 < first_zero_freq_sq {
                                    atten_frac = ((first_zero_freq_sq - f2) as f64
                                        / ((1. - atten_start_frac as f64)
                                            * first_zero_freq_sq as f64))
                                        as f32;
                                    ctf = (atten_frac as f64
                                        + (1. - atten_frac as f64) * ctf as f64)
                                        as f32;
                                }
                                cur_strip[(fy * strip_xdim + 2 * fx) as usize] *= ctf;
                                cur_strip[(fy * strip_xdim + 2 * fx + 1) as usize] *= ctf;
                            } else if ctf < 0. {
                                cur_strip[(fy * strip_xdim + 2 * fx) as usize] *= -1.;
                                cur_strip[(fy * strip_xdim + 2 * fx + 1) as usize] *= -1.;
                            }
                        }
                    }
                    if debug_mode != 0 {
                        cur_time = wall_time();
                        wall_corr += cur_time - wall_start;
                        wall_start = cur_time;
                    }

                    //inverse FFT;
                    let _ = todfftc(cur_strip, strip_pixel_num, ny, 1);
                    if debug_mode != 0 {
                        cur_time = wall_time();
                        wall_fft += cur_time - wall_start;
                        wall_start = cur_time;
                    }
                }
            } else {
                // On GPU, do extraction and transform or full copy, then correct the CTF
                if gpu_extract_and_transform(strip_ind, strip_begin, strip_end, 9, 9) != 0 {
                    exit_error(&c_format_bytes(
                        "Calling gpuExtractAndTransform for view %d",
                        &[CArg::Int(view.into())],
                    ));
                }
                if gpu_correct_ctf(
                    strip_ind,
                    freq_scalex,
                    freq_scaley,
                    point_defocus,
                    cos_astig,
                    sin_astig,
                    focus_sum,
                    focus_diff,
                    cuton_angstroms,
                    phase_frac_factor,
                    phase_shift,
                    amp_angle,
                    c1,
                    c2,
                    scale_by_power,
                    power_is_half,
                    general_power,
                    first_zero_freq_sq,
                    atten_start_frac,
                    min_atten_freq_sq,
                ) != 0
                {
                    exit_error(&c_format_bytes(
                        "Calling gpuApplyCTF for view %d",
                        &[CArg::Int(view.into())],
                    ));
                }
            }

            /*
             * Put corrected data column(s) into the restored array
             */
            // The starting strip requires a copy of columns
            let half = (ny * strip_xdim) as usize;
            if strip_ind == 0 {
                if do_diagonals {
                    // Diagonals
                    low_lim = (-0.5 - effective_xcen as f64) as f32;
                    high_lim = (half_strip as f64 - 0.5 - effective_xcen as f64) as f32;
                    if use_gpu < 0 {
                        let cur_strip: &[f32] = if do_full_images {
                            &full_copy[(strip_ind % 2) as usize]
                        } else {
                            &strip[(strip_ind % 2) as usize * half..]
                        };
                        for row in 0..nyfile {
                            for column in 0..nx {
                                axis_dist = -sin_view_axis * (column as f32 - x_pix_center)
                                    + cos_view_axis * (row as f32 - y_pix_center);
                                if axis_dist >= low_lim && axis_dist <= high_lim {
                                    out_slice.data.f_mut()[(row * nx + column) as usize] =
                                        cur_strip
                                            [((row + yoff) * strip_xdim + column + xoff) as usize];
                                }
                            }
                        }
                    } else if gpu_copy_diagonals(
                        strip_ind,
                        xoff,
                        yoff,
                        sin_view_axis,
                        cos_view_axis,
                        low_lim,
                        high_lim,
                    ) != 0
                    {
                        exit_error(&c_format_bytes(
                            "Calling gpuCopyDiagonals for view %d",
                            &[CArg::Int(view.into())],
                        ));
                    }
                } else {
                    // Regular columns
                    if use_gpu < 0 {
                        let cur_strip: &[f32] = if do_full_images {
                            &full_copy[(strip_ind % 2) as usize]
                        } else {
                            &strip[(strip_ind % 2) as usize * half..]
                        };
                        for row in 0..nyfile {
                            for column in 0..half_strip {
                                out_slice.data.f_mut()[(row * nx + column) as usize] =
                                    cur_strip[((row + yoff) * strip_xdim + column + xoff) as usize];
                            }
                        }
                    } else if gpu_copy_columns(strip_ind, xoff, yoff, 0, half_strip - 1) != 0 {
                        exit_error(&c_format_bytes(
                            "Calling gpuCopyColumns for columns %d to %d of view %d",
                            &[
                                CArg::Int(0),
                                CArg::Int((half_strip - 1).into()),
                                CArg::Int(view.into()),
                            ],
                        ));
                    }
                }
                if debug_mode > 1 {
                    let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                        "column=1 ... %d stripMid %d  xoff %d\n",
                        &[
                            CArg::Int(half_strip.into()),
                            CArg::Int(strip_mid.into()),
                            CArg::Int(xoff.into()),
                        ],
                    ));
                }
                finished = finished || (strip_end == effective_nx - 1 && !do_right_shifted_mid);
            } else {
                // Single pixel from interpolation width 1 is just a copy
                if inter_pixel_num == 1 {
                    if do_diagonals {
                        // Diagonals
                        low_lim = (strip_mid as f64 - 0.5 - effective_xcen as f64) as f32;
                        high_lim = (strip_mid as f64 + 0.5 - effective_xcen as f64) as f32;
                        if use_gpu < 0 {
                            let cur_strip: &[f32] = if do_full_images {
                                &full_copy[(strip_ind % 2) as usize]
                            } else {
                                &strip[(strip_ind % 2) as usize * half..]
                            };
                            for row in 0..nyfile {
                                for column in 0..nx {
                                    axis_dist = -sin_view_axis * (column as f32 - x_pix_center)
                                        + cos_view_axis * (row as f32 - y_pix_center);
                                    if axis_dist >= low_lim && axis_dist <= high_lim {
                                        out_slice.data.f_mut()[(row * nx + column) as usize] =
                                            cur_strip[((row + yoff) * strip_xdim + column + xoff)
                                                as usize];
                                    }
                                }
                            }
                        } else if gpu_copy_diagonals(
                            strip_ind,
                            xoff,
                            yoff,
                            sin_view_axis,
                            cos_view_axis,
                            low_lim,
                            high_lim,
                        ) != 0
                        {
                            exit_error(&c_format_bytes(
                                "Calling gpuCopyDiagonals for view %d",
                                &[CArg::Int(view.into())],
                            ));
                        }
                    } else {
                        // Regular columns
                        if use_gpu < 0 {
                            let last_strip: &[f32] = if do_full_images {
                                &full_copy[((strip_ind + 1) % 2) as usize]
                            } else {
                                &strip[((strip_ind + 1) % 2) as usize * half..]
                            };
                            for row in 0..nyfile {
                                out_slice.data.f_mut()[(row * nx + strip_mid) as usize] =
                                    last_strip
                                        [((row + yoff) * strip_xdim + half_strip + xoff) as usize];
                            }
                        } else {
                            if do_full_images
                                || strip_ind <= num_left_shifted_mid
                                || in_right_shifted_mid
                            {
                                cur_offset = xoff;
                            } else {
                                cur_offset = half_strip - strip_mid;
                            }
                            if gpu_copy_columns(
                                strip_ind + 1,
                                cur_offset,
                                yoff,
                                strip_mid,
                                strip_mid,
                            ) != 0
                            {
                                exit_error(&c_format_bytes(
                                    "Calling gpuCopyColumns for column %d of view %d",
                                    &[CArg::Int(strip_mid.into()), CArg::Int(k.into())],
                                ));
                            }
                        }
                    }

                // Otherwise do the interpolation
                } else if do_diagonals {
                    // Diagonals
                    last_axis_dist = (strip_mid - strip_stride) as f32 - effective_xcen;
                    cur_axis_dist = strip_mid as f32 - effective_xcen;
                    if use_gpu < 0 {
                        low_lim = last_axis_dist + 0.5;
                        high_lim = cur_axis_dist + 0.5;
                        let (cur_strip, last_strip): (&[f32], &[f32]) = if do_full_images {
                            (
                                &full_copy[(strip_ind % 2) as usize],
                                &full_copy[((strip_ind + 1) % 2) as usize],
                            )
                        } else {
                            let (first, second) = strip.split_at(half);
                            if strip_ind % 2 != 0 {
                                (second, first)
                            } else {
                                (first, second)
                            }
                        };
                        for row in 0..nyfile {
                            for column in 0..nx {
                                axis_dist = -sin_view_axis * (column as f32 - x_pix_center)
                                    + cos_view_axis * (row as f32 - y_pix_center);
                                if axis_dist >= low_lim && axis_dist <= high_lim {
                                    axis_dist = last_axis_dist.max(cur_axis_dist.min(axis_dist));
                                    cur_ax_frac = (if axis_dist < cur_axis_dist {
                                        axis_dist
                                    } else {
                                        cur_axis_dist
                                    } - last_axis_dist)
                                        / strip_stride as f32;
                                    out_slice.data.f_mut()[(row * nx + column) as usize] =
                                        (cur_ax_frac as f64
                                            * cur_strip[((row + yoff) * strip_xdim + xoff + column)
                                                as usize]
                                                as f64
                                            + (1. - cur_ax_frac as f64)
                                                * last_strip[((row + yoff) * strip_xdim
                                                    + xoff
                                                    + column)
                                                    as usize]
                                                    as f64)
                                            as f32;
                                }
                            }
                        }
                    } else if gpu_interp_diagonals(
                        strip_ind,
                        xoff,
                        yoff,
                        strip_stride,
                        sin_view_axis,
                        cos_view_axis,
                        last_axis_dist,
                        cur_axis_dist,
                    ) != 0
                    {
                        exit_error(&c_format_bytes(
                            "Calling gpuInterpDiagonals for view %d",
                            &[CArg::Int(view.into())],
                        ));
                    }
                } else {
                    // Regular columns: set up arrays of fractions and set offsets
                    for column in strip_mid - strip_stride + 1..strip_mid + 1 {
                        strip_dist[0] = column - strip_mid + strip_stride - 1;
                        strip_dist[1] = strip_mid + 1 - column;
                        cur_frac[column as usize] = strip_dist[0] as f32 / strip_stride as f32;
                        last_frac[column as usize] = strip_dist[1] as f32 / strip_stride as f32;
                    }
                    if do_full_images || strip_ind <= num_left_shifted_mid || in_right_shifted_mid {
                        cur_offset = xoff;
                        last_offset = xoff;
                    } else {
                        cur_offset = half_strip - (strip_mid + 1);
                        last_offset = half_strip + strip_stride - (strip_mid + 1);
                    }

                    // Do the interpolation
                    if use_gpu < 0 {
                        let (cur_strip, last_strip): (&[f32], &[f32]) = if do_full_images {
                            (
                                &full_copy[(strip_ind % 2) as usize],
                                &full_copy[((strip_ind + 1) % 2) as usize],
                            )
                        } else {
                            let (first, second) = strip.split_at(half);
                            if strip_ind % 2 != 0 {
                                (second, first)
                            } else {
                                (first, second)
                            }
                        };
                        for row in 0..nyfile {
                            for column in strip_mid - strip_stride + 1..strip_mid + 1 {
                                out_slice.data.f_mut()[(row * nx + column) as usize] = cur_frac
                                    [column as usize]
                                    * cur_strip[((row + yoff) * strip_xdim + cur_offset + column)
                                        as usize]
                                    + last_frac[column as usize]
                                        * last_strip[((row + yoff) * strip_xdim
                                            + last_offset
                                            + column)
                                            as usize];
                            }
                        }
                    } else if gpu_interpolate_columns(
                        strip_ind,
                        yoff,
                        strip_stride,
                        strip_mid,
                        half_strip,
                        cur_offset,
                        last_offset,
                    ) != 0
                    {
                        exit_error(&c_format_bytes(
                            "Calling gpuInterpolateColumns for columns %d to %d of view %d",
                            &[
                                CArg::Int((strip_mid - strip_stride + 1).into()),
                                CArg::Int(strip_mid.into()),
                                CArg::Int(k.into()),
                            ],
                        ));
                    }
                }
                if debug_mode > 1 {
                    let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                        "column=%d ... %d",
                        &[
                            CArg::Int((strip_mid - strip_stride + 1 + 1).into()),
                            CArg::Int((strip_mid + 1).into()),
                        ],
                    ));
                    if inter_pixel_num == 1 {
                        let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                            "  offset %d\n",
                            &[CArg::Int(if use_gpu < 1 {
                                (half_strip + xoff).into()
                            } else {
                                cur_offset.into()
                            })],
                        ));
                    } else {
                        let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                            " curOff %d  lastOff %d\n",
                            &[CArg::Int(cur_offset.into()), CArg::Int(last_offset.into())],
                        ));
                    }
                }
                if use_gpu < 0 && debug_mode != 0 {
                    cur_time = wall_time();
                    wall_interp += cur_time - wall_start;
                    wall_start = cur_time;
                }
            }

            // Do a copy from the last strip when finished is set
            if finished {
                if do_diagonals {
                    // Diagonals
                    low_lim = ((strip_mid + 1) as f64 - 0.5 - effective_xcen as f64) as f32;
                    high_lim = (effective_nx as f64 - 0.5 - effective_xcen as f64) as f32;
                    if use_gpu < 0 {
                        let cur_strip: &[f32] = if do_full_images {
                            &full_copy[(strip_ind % 2) as usize]
                        } else {
                            &strip[(strip_ind % 2) as usize * half..]
                        };
                        for row in 0..nyfile {
                            for column in 0..nx {
                                axis_dist = -sin_view_axis * (column as f32 - x_pix_center)
                                    + cos_view_axis * (row as f32 - y_pix_center);
                                if axis_dist >= low_lim && axis_dist <= high_lim {
                                    out_slice.data.f_mut()[(row * nx + column) as usize] =
                                        cur_strip
                                            [((row + yoff) * strip_xdim + column + xoff) as usize];
                                }
                            }
                        }
                    } else if gpu_copy_diagonals(
                        strip_ind,
                        xoff,
                        yoff,
                        sin_view_axis,
                        cos_view_axis,
                        low_lim,
                        high_lim,
                    ) != 0
                    {
                        exit_error(&c_format_bytes(
                            "Calling gpuCopyDiagonals for view %d",
                            &[CArg::Int(view.into())],
                        ));
                    }
                } else {
                    // Regular columns
                    if do_full_images || in_right_shifted_mid {
                        cur_offset = xoff;
                    } else {
                        cur_offset = half_strip - strip_mid - 1;
                    }
                    if use_gpu < 0 {
                        let cur_strip: &[f32] = if do_full_images {
                            &full_copy[(strip_ind % 2) as usize]
                        } else {
                            &strip[(strip_ind % 2) as usize * half..]
                        };
                        for row in 0..nyfile {
                            for column in strip_mid + 1..nx {
                                out_slice.data.f_mut()[(row * nx + column) as usize] = cur_strip
                                    [((row + yoff) * strip_xdim + cur_offset + column) as usize];
                            }
                        }
                    } else if gpu_copy_columns(strip_ind, cur_offset, yoff, strip_mid + 1, nx - 1)
                        != 0
                    {
                        exit_error(&c_format_bytes(
                            "Calling gpuCopyColumns for columns %d to %d of view %d",
                            &[
                                CArg::Int((strip_mid + 1).into()),
                                CArg::Int((nx - 1).into()),
                                CArg::Int(k.into()),
                            ],
                        ));
                    }
                }
                if debug_mode > 1 {
                    let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                        "column=%d ... %d  curOff %d\n",
                        &[
                            CArg::Int((strip_mid + 1 + 1).into()),
                            CArg::Int(nx.into()),
                            CArg::Int(cur_offset.into()),
                        ],
                    ));
                }
            }

            strip_ind += 1;
        } //while strip loop

        // Get result from GPU if any, process and write
        if use_gpu >= 0 && gpu_return_image(out_slice.data.f_mut()) != 0 {
            exit_error(&c_format_bytes(
                "Calling gpuReturnImage for view %d",
                &[CArg::Int(k.into())],
            ));
        }

        if !do_full_images && use_gpu < 0 {
            drop(std::mem::take(&mut strip));
        }
        slice_mmm(&mut out_slice);
        if (out_slice.min as f64) < amin {
            amin = out_slice.min as f64;
        }
        if out_slice.max as f64 > amax {
            amax = out_slice.max as f64;
        }
        mean_sum += out_slice.mean as f64;
        if slice_mode != SLICE_MODE_FLOAT && slice_new_mode(&mut out_slice, slice_mode) < 0 {
            exit_error(b"converting slice to original mode");
        }

        {
            let mut fout = foutput.clone().expect("output opened");
            let rc = unsafe {
                parallel_write_slice(
                    out_slice.data.bytes_mut().as_mut_ptr().cast(),
                    &mut fout,
                    &mut out_header,
                    curr_k,
                )
            };
            if rc != 0 {
                exit_error(&c_format_bytes(
                    "Writing slice %d error",
                    &[CArg::Int(curr_k.into())],
                ));
            }
        }
        curr_k += 1;
        let _ = ImodFile::Stdout.flush();
        view += 1;
    } //k slice

    if is_single_run {
        out_header.amin = amin as f32;
        out_header.amax = amax as f32;
        out_header.amean = (mean_sum / curr_nz as f64) as f32;
        let mut fout = foutput.clone().expect("output opened");
        if mrc_head_write(&mut fout, &mut out_header) != 0 {
            exit_error(b"Writing slice header error");
        }
    } else {
        //for collectmmm
        let _ = ImodFile::Stdout.write_all(&c_format_bytes(
            "min, max, mean, # pixels= %f  %f  %f %d \n",
            &[
                CArg::Dbl(amin),
                CArg::Dbl(amax),
                CArg::Dbl(mean_sum / curr_nz as f64),
                CArg::Int((nx * ny * curr_nz).into()),
            ],
        ));
    }
    if debug_mode != 0 {
        if use_gpu >= 0 {
            let mut copy = 0.0f64;
            gpu_get_times(
                &mut copy,
                &mut wall_prep,
                &mut wall_fft,
                &mut wall_corr,
                &mut wall_interp,
            );
            let _ = ImodFile::Stdout.write_all(&c_format_bytes("Copy %.3f  ", &[CArg::Dbl(copy)]));
        }
        let _ = ImodFile::Stdout.write_all(&c_format_bytes(
            "Prep %.3f  FFT %.3f  correct %.3f  Interp %.3f\n",
            &[
                CArg::Dbl(wall_prep),
                CArg::Dbl(wall_fft),
                CArg::Dbl(wall_corr),
                CArg::Dbl(wall_interp),
            ],
        ));
    }
    if parallel_hdf {
        if unsafe { par_wrt_flush_buffers(ii_file, &mut out_header) } != 0 {
            exit_error(b"Doing final write of buffer to output HDF file");
        }
        unsafe { ii_delete(ii_file) };
        par_wrt_close();
    } else if let Some(mut fout) = foutput.clone() {
        ii_fclose(&mut fout);
    }
    ii_fclose(&mut fp_stack);
    let _ = ImodFile::Stdout.flush();
    0
}

/// C `firstZeroShift` (`ctfphaseflip.cpp:1300`).
///
/// Computes the shift in the first zero across an extent in pixels, given
/// focus in microns, tilt angle in radians and the basic parameters of the
/// CTF.  The shift is in Nyquist units.
#[allow(clippy::too_many_arguments)]
pub fn first_zero_shift(
    defocus: f64,
    phase: f64,
    angle: f64,
    extent: i32,
    amp_contrast: f64,
    cs: f64,
    pixel_size: f64,
    voltage: f64,
) -> f64 {
    let first_zero: f64;
    let mut focus: f64;
    focus = defocus - 0.5 * extent as f64 * angle.tan() * pixel_size / 1000.;
    first_zero = get_any_one_zero(focus, phase, 1, amp_contrast, cs, pixel_size, voltage);
    focus = defocus + 0.5 * extent as f64 * angle.tan() * pixel_size / 1000.;
    (first_zero - get_any_one_zero(focus, phase, 1, amp_contrast, cs, pixel_size, voltage)).abs()
}

/// C `getAnyOneZero` (`ctfphaseflip.cpp:1316`).
///
/// Computes the given zero position in Nyquist units given focus in microns
/// and phase shift in radians, and the basic parameters of the CTF.
pub fn get_any_one_zero(
    defocus: f64,
    phase: f64,
    zero_num: i32,
    amp_contrast: f64,
    cs: f64,
    pixel_size: f64,
    voltage: f64,
) -> f64 {
    let delz: f64;
    let theta: f64;
    let wavelength = 1.23984 / (voltage * (voltage + 1022.0)).sqrt();
    let cs_one = (cs * wavelength).sqrt(); // deltaZ=-deltaZ'/mCs1;  In microns
    let cs_two = (1000000.0 * cs / wavelength).sqrt().sqrt(); //theta=theta'*mCs2;
    let amp_angle = 2. * (amp_contrast / (1. - amp_contrast * amp_contrast).sqrt()).atan() / MY_PI;
    delz = defocus / cs_one;
    // `B3DMAX(0., x)` is `0. < x ? ... ` -- `a < b ? a : b` with a = 0.
    let inner = delz * delz + amp_angle + 2. * phase / MY_PI - 2. * zero_num as f64;
    theta = (delz - (if 0. > inner { 0. } else { inner }).sqrt()).sqrt();
    theta * pixel_size * 2.0 / (wavelength * cs_two)
}

/// C `interpolateTable` (`ctfphaseflip.cpp:1334`).
///
/// Interpolates or extends values from one item in the table to fill a whole
/// array.
pub fn interpolate_table(defocus: &mut [f32], nz: i32, if_angles: bool) {
    let mut first: i32 = -1;
    let mut second: i32 = 0;
    let mut diff: f32;
    let mut frac: f32;

    // Skip to the next measured value
    for k in 0..nz {
        if defocus[k as usize] == UNUSED_DEFOCUS {
            continue;
        }
        second = k;

        // Then copy into start of table or interpolate between last and this one
        if first == -1 {
            for row in 0..second {
                defocus[row as usize] = defocus[second as usize];
            }
        } else {
            for row in first + 1..second {
                if if_angles {
                    diff = angle_within_limits(
                        defocus[second as usize] - defocus[first as usize],
                        -90.,
                        90.,
                    ) as f32;
                    frac = (row - first) as f32 / (second - first) as f32;
                    defocus[row as usize] =
                        angle_within_limits(defocus[first as usize] + frac * diff, -90., 90.)
                            as f32;
                } else {
                    defocus[row as usize] = ((row - first) as f32 * defocus[second as usize]
                        + (second - row) as f32 * defocus[first as usize])
                        / (second - first) as f32;
                }
            }
        }
        first = second;
    }

    // Then copy last into end of table
    for k in (0..nz).rev() {
        if defocus[k as usize] == UNUSED_DEFOCUS {
            defocus[k as usize] = defocus[second as usize];
        } else {
            break;
        }
    }
}
