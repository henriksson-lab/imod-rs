//! Translation of `IMOD/librgctf/testctffind.cpp`, the test program the
//! library's Makefile builds and installs beside `libctffind`.

use crate::imod::libcfshr::b3dutil::{CArg, ImodFile, b3d_get_error, c_format, wall_time};
use crate::imod::libcfshr::filtxcorr::nice_frame;
use crate::imod::libcfshr::islice::{Islice, MrcData, slice_init};
use crate::imod::libcfshr::parse_params::{
    exit_error, pip_get_boolean, pip_get_float, pip_get_integer, pip_get_non_option_arg,
    pip_get_two_integers, pip_print_help, pip_read_or_parse_options,
};
use crate::imod::libcfshr::reduce_by_binning::extract_and_bin_into_array;
use crate::imod::libcfshr::spectrumscaled::{SpectrumInput, SpectrumOutput, spectrum_scaled};
use crate::imod::libfft::odfft::nice_fft_limit;
use crate::imod::libfft::todfft::todfft;
use crate::imod::libiimod::mrcfiles::{
    MRC_MODE_FLOAT, MrcHeader, mrc_get_scale, mrc_head_read, mrc_mread_slice,
};
use crate::imod::libiimod::mrcslice::slice_write_mrcfile;

use super::ctffind::{CtffindParams, ctffind, ctffind_set_slice_write_func};

/// C++ `RADIANS_PER_DEGREE` from `b3dutil.h`.
const RADIANS_PER_DEGREE: f64 = 0.017453293;

/// C++ `static int writeSlice(const char *, float *, int, int)`
/// (`testctffind.cpp:7`).
///
/// `sliceInit` borrows the caller's float pointer; the Rust `Islice` owns its
/// floats, so the array is copied into the temporary slice before writing.
pub fn write_slice(filename: &str, data: &[f32], xsize: i32, ysize: i32) -> i32 {
    let mut slice = Islice {
        data: MrcData::default(),
        xsize: 0,
        ysize: 0,
        mode: 0,
        csize: 0,
        dsize: 0,
        min: 0.,
        max: 0.,
        mean: 0.,
        index: 0,
        cval: [0.; 4],
    };
    if slice_init(
        &mut slice,
        xsize,
        ysize,
        MRC_MODE_FLOAT,
        MrcData::F(data.to_vec()),
    ) != 0
    {
        return -1;
    }
    slice_write_mrcfile(filename, &mut slice)
}

/// C++ `main` of `testctffind.cpp` (`testctffind.cpp:14`).
pub fn main(argv: &[Vec<u8>]) -> i32 {
    let mut params = CtffindParams::default();
    // Fallbacks from   ../manpages/autodoc2man 2 1 testctffind
    let num_options = 22;
    let options: [&[u8]; 22] = [
        b"volt:Voltage:F:",
        b"sph:SphericalAberration:F:",
        b"box:BoxSize:I:",
        b"rmin:MinimumResolution:F:",
        b"rmax:MaximumResolution:F:",
        b"dmin:MinDefocus:F:",
        b"dmax:MaxDefocus:F:",
        b"dstep:DefocusStepSize:F:",
        b"fast:FastSearch:B:",
        b"atol:AstigmatismTolerance:F:",
        b"known:AstigmatismIsKnown:B:",
        b"astig:KnownAstigmatism:F:",
        b"angle::F:",
        b"noex::B:",
        b"phase:FindPhaseShift:B:",
        b"phmin:MinimumPhaseShift:F:",
        b"phmax:MaximumPhaseShift:F:",
        b"phstep:PhaseStepSize:F:",
        b"resamp:ResampleSpectrum:F:",
        b"sec:SectionRangeToDo:IP:",
        b"param:ParameterFile:PF:",
        b"help:usage:B:",
    ];
    let mut num_opt_arg = 0i32;
    let mut num_non_opt_arg = 0i32;
    let mut val: i32;
    let mut num_points = 0i32;
    let mut pad_size: i32;
    let mut start: i32;
    let mut end: i32;
    let mut use_box: i32;
    let mut z_start: i32;
    let mut z_end: i32;
    let mut hdata = MrcHeader::default();
    let mut results_array = [0.0f32; 7];
    let mut last_bin_freq = 0.0f32;
    let mut resamp_res = 2.8f32;
    let wall_start = wall_time();

    ctffind_set_slice_write_func(Some(write_slice));

    pip_read_or_parse_options(
        argv.len() as i32,
        argv,
        &options,
        num_options,
        b"testctffind",
        1,
        0,
        0,
        &mut num_opt_arg,
        &mut num_non_opt_arg,
        None,
    );
    if num_non_opt_arg == 0 {
        pip_print_help(b"testctffind", 0, 1, 0);
        std::process::exit(0);
    }
    params.acceleration_voltage = 200.;
    params.spherical_aberration = 2.;
    params.amplitude_contrast = 0.07;
    params.box_size = 256;
    params.minimum_resolution = 50.;
    params.maximum_resolution = 10.;
    params.minimum_defocus = 5000.;
    params.maximum_defocus = 80000.;
    params.defocus_search_step = 500.;
    params.astigmatism_tolerance = -100.;
    params.additional_phase_shift_search_step = 0.1;
    params.known_astigmatism = 0.;
    params.known_astigmatism_angle = 0.;
    params.minimum_additional_phase_shift = 0.;
    params.maximum_additional_phase_shift = 0.;
    params.noisy_input_image = false;

    let mut filename: Vec<u8> = Vec::new();
    pip_get_non_option_arg(0, &mut filename);
    pip_get_float(b"volt", &mut params.acceleration_voltage);
    pip_get_float(b"sph", &mut params.spherical_aberration);
    pip_get_integer(b"box", &mut params.box_size);
    pip_get_float(b"rmin", &mut params.minimum_resolution);
    pip_get_float(b"rmax", &mut params.maximum_resolution);
    pip_get_float(b"dmin", &mut params.minimum_defocus);
    pip_get_float(b"dmax", &mut params.maximum_defocus);
    pip_get_float(b"dstep", &mut params.defocus_search_step);
    val = 0;
    pip_get_boolean(b"lowdose", &mut val);
    params.noisy_input_image = val != 0;
    pip_get_float(b"atol", &mut params.astigmatism_tolerance);
    val = 0;
    pip_get_boolean(b"fast", &mut val);
    params.slower_search = val == 0;
    val = 0;
    pip_get_boolean(b"known", &mut val);
    params.astigmatism_is_known = val != 0;
    val = 0;
    pip_get_boolean(b"phase", &mut val);
    params.find_additional_phase_shift = val != 0;
    val = 0;
    pip_get_boolean(b"noex", &mut val);
    params.compute_extra_stats = val == 0;
    if params.astigmatism_is_known {
        pip_get_float(b"astig", &mut params.known_astigmatism);
    }
    pip_get_float(b"angle", &mut params.known_astigmatism_angle);
    if params.find_additional_phase_shift {
        pip_get_float(b"phmin", &mut params.minimum_additional_phase_shift);
        pip_get_float(b"phmax", &mut params.maximum_additional_phase_shift);
        pip_get_float(b"phstep", &mut params.additional_phase_shift_search_step);
    }
    params.box_size = 2 * ((params.box_size + 1) / 2);
    pip_get_float(b"resamp", &mut resamp_res);

    let path = String::from_utf8_lossy(&filename).into_owned();
    let Some(mut fp) = ImodFile::open(&path, "rb") else {
        exit_error(c_format("Opening file %s", &[CArg::Bytes(&filename)]).as_bytes());
    };
    if mrc_head_read(&mut fp, &mut hdata) != 0 {
        exit_error(
            c_format(
                "Reading header of file %s - %s",
                &[CArg::Bytes(&filename), CArg::Str(&b3d_get_error())],
            )
            .as_bytes(),
        );
    }
    z_start = 0;
    z_end = hdata.nz - 1;
    pip_get_two_integers(b"SectionRangeToDo", &mut z_start, &mut z_end);
    if z_start < 0 || z_end >= hdata.nz || z_start > z_end {
        exit_error(b"Section range is out of range or out of order");
    }

    let (xscale, _yscale, _zscale) = mrc_get_scale(&hdata);
    params.pixel_size_of_input_image = xscale;

    use_box = params.box_size;
    if resamp_res > params.pixel_size_of_input_image * 2. {
        use_box = 2
            * (((1.
                + 0.5 * f64::from(params.box_size) * f64::from(resamp_res)
                    / f64::from(params.pixel_size_of_input_image))
                + 0.5)
                .floor() as i32
                / 2);
    }

    let mut spectrum = vec![0.0f32; (use_box * (use_box + 2)) as usize];

    for iz in z_start..=z_end {
        let Some(image_array) = mrc_mread_slice(&mut fp, &mut hdata, iz, b'Z') else {
            exit_error(
                c_format("Reading section 0 - %s", &[CArg::Str(&b3d_get_error())]).as_bytes(),
            );
        };

        pad_size = hdata.nx.max(hdata.ny);
        pad_size = nice_frame(pad_size, 2, nice_fft_limit());
        let input = match &image_array {
            MrcData::B(values) => SpectrumInput::Byte(values),
            MrcData::S(values) => SpectrumInput::Short(values),
            MrcData::Us(values) => SpectrumInput::UShort(values),
            MrcData::F(values) => SpectrumInput::Float(values),
        };
        let val = spectrum_scaled(
            input,
            hdata.nx,
            hdata.ny,
            SpectrumOutput::Float(&mut spectrum),
            -pad_size,
            use_box,
            0,
            0.,
            -1,
            todfft,
        );
        if val != 0 {
            exit_error(
                c_format(
                    "Making reduced spectrum, error code %d",
                    &[CArg::Int(i64::from(val))],
                )
                .as_bytes(),
            );
        }
        if use_box > params.box_size {
            start = (use_box - params.box_size) / 2;
            end = start + params.box_size - 1;
            let source: Vec<u8> = spectrum
                .iter()
                .flat_map(|value| value.to_ne_bytes())
                .collect();
            let mut destination: Vec<u8> = vec![0; source.len()];
            let mut nxr = 0i32;
            let mut nyr = 0i32;
            if extract_and_bin_into_array(
                &source,
                MRC_MODE_FLOAT,
                use_box + 2,
                start,
                end,
                start,
                end,
                1,
                &mut destination,
                params.box_size + 2,
                0,
                0,
                0,
                &mut nxr,
                &mut nyr,
            ) != 0
            {
                exit_error(b"Extracting reduced spectrum from larger box to smaller");
            }
            for (index, chunk) in destination.chunks_exact(4).enumerate() {
                spectrum[index] = f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
            params.pixel_size_of_input_image *= (use_box as f32) / (params.box_size as f32);
        }

        let mut rotational_avg: Option<Vec<f32>> = None;
        let mut normalized_avg: Option<Vec<f32>> = None;
        let mut fit_curve: Option<Vec<f32>> = None;
        if !ctffind(
            &params,
            &spectrum,
            params.box_size + 2,
            &mut results_array,
            Some(&mut rotational_avg),
            Some(&mut normalized_avg),
            Some(&mut fit_curve),
            &mut num_points,
            &mut last_bin_freq,
        ) {
            exit_error(b"An error occurred in ctffind");
        }

        if hdata.nz > 1 {
            print!(
                "{}",
                c_format(
                    "%1.f  %.1f  %.2f  %.4f  %.4f  %.1f  %.1f\n",
                    &[
                        CArg::Dbl(f64::from(results_array[0])),
                        CArg::Dbl(f64::from(results_array[1])),
                        CArg::Dbl(f64::from(results_array[2])),
                        CArg::Dbl(f64::from(results_array[3])),
                        CArg::Dbl(f64::from(results_array[4])),
                        CArg::Dbl(f64::from(results_array[5])),
                        CArg::Dbl(f64::from(results_array[6])),
                    ],
                )
            );
        } else {
            print!(
                "{}",
                c_format(
                    "Defocus 1 %.1f  2 %.1f  (astig. %.1f) angle %.2f",
                    &[
                        CArg::Dbl(f64::from(results_array[0])),
                        CArg::Dbl(f64::from(results_array[1])),
                        CArg::Dbl(f64::from(results_array[0] - results_array[1])),
                        CArg::Dbl(f64::from(results_array[2])),
                    ],
                )
            );
            if params.find_additional_phase_shift {
                print!(
                    "{}",
                    c_format(
                        "    Phase shift %.4f rad (%.2f deg)",
                        &[
                            CArg::Dbl(f64::from(results_array[3])),
                            CArg::Dbl(f64::from(results_array[3]) / RADIANS_PER_DEGREE),
                        ],
                    )
                );
            }
            print!(
                "{}",
                c_format("\nScore %.4f\n", &[CArg::Dbl(f64::from(results_array[4]))],)
            );
            if params.compute_extra_stats {
                print!(
                    "{}",
                    c_format(
                        "Thon rings well fit to %.1f\n",
                        &[CArg::Dbl(f64::from(results_array[5]))],
                    )
                );
                if results_array[6] != 0.0 {
                    print!(
                        "{}",
                        c_format(
                            "CTF aliasing detected at %.1f\n",
                            &[CArg::Dbl(f64::from(results_array[6]))],
                        )
                    );
                }
            }
        }
        let _ = (
            rotational_avg,
            normalized_avg,
            fit_curve,
            num_points,
            last_bin_freq,
        );
    }
    print!(
        "{}",
        c_format(
            "ctffind time %.3f\n",
            &[CArg::Dbl(wall_time() - wall_start)],
        )
    );
    0
}
