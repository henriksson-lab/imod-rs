//! Translation of `IMOD/mrc/mtffilter.cpp`.
//!
//! One Rust function per source function: `main` plus the six file statics
//! `fftFilter3D`, `fftFilter1D`, `amplifier`, `adjustedFakeIter`,
//! `deconvFilter` and `convertFreqUnit`.

use std::io::{BufReader, Write as _};

use crate::imod::libcfshr::autodoc::{
    adoc_get_image_meta_info, adoc_open_image_metadata, adoc_set_current,
};
use crate::imod::libcfshr::b3dutil::{
    CArg, ImodFile, c_format_bytes, imod_prog_name, imod_usage_header,
    set_float_output_for_entered_mode, standard_memory_limit_mb,
};
use crate::imod::libcfshr::extraheader::{
    get_metadata_weighting_doses, prior_doses_from_image_doses,
};
use crate::imod::libcfshr::filtxcorr::{
    FilterIn, dose_weight_filter, nice_frame, xcorr_filter_part, xcorr_set_ctf_no_scl,
};
use crate::imod::libcfshr::islice::MrcData;
use crate::imod::libcfshr::parse_params::{
    exit_error, pip_done, pip_get_boolean, pip_get_float, pip_get_in_out_file, pip_get_integer,
    pip_get_string, pip_get_three_floats, pip_get_two_floats, pip_get_two_integers,
    pip_read_or_parse_options,
};
use crate::imod::libcfshr::parselist::parselist;
use crate::imod::libcfshr::readlinevalues::{
    RLFV_SEPARATE_LINES, ReadValueArray, exit_from_value_read_error, read_lines_for_values,
};
use crate::imod::libcfshr::reduce_by_binning::repack_float_image;
use crate::imod::libcfshr::simplestat::array_min_max_mean;
use crate::imod::libcfshr::taperpad::{PadIn, slice_noise_taper_pad, slice_taper_out_pad};
use crate::imod::libfft::odfft::nice_fft_limit;
use crate::imod::libfft::rustfft_backend::{odfftc, todfftc};
use crate::imod::libfft::thrdfft::thrdfftc;
use crate::imod::libiimod::mrcfiles::{MRC_LABEL_SIZE, MRC_MODE_FLOAT, mrc_fill_label_string};
use crate::imod::libiimod::mrcslice::full_array_min_max_mean;
use crate::imod::libiimod::unit_fileio::{
    iiu_close, iiu_file_type, iiu_open, iiu_read_section, iiu_ret_adoc_index, iiu_set_position,
    iiu_write_section,
};
use crate::imod::libiimod::unit_header::{
    iiu_alt_cell, iiu_alt_mode, iiu_alt_sample, iiu_alt_size, iiu_print_header, iiu_ret_basic_head,
    iiu_ret_cell, iiu_ret_delta, iiu_trans_header, iiu_write_header_str,
};

const MRC_MODE_COMPLEX_FLOAT: i32 = 4;

/// The `imodUsageHeader` callback in the shape `PipReadOrParseOptions` takes.
fn imod_usage_header_for_pip(prog_name: &[u8]) {
    imod_usage_header(Some(&String::from_utf8_lossy(prog_name)));
    let _ = ImodFile::Stdout.flush();
}

/// C `main` in `mtffilter.cpp` (`mtffilter.cpp:30`).
#[allow(clippy::needless_range_loop)]
pub fn mtffilter(arguments: &[String]) -> i32 {
    const LIMMTF: i32 = 8193;
    const MAX_LINE: i32 = 120;
    const LIMSTOCKCURVES: i32 = 1;
    const LIMSTOCKPOINTS: usize = 36;
    //
    let progname_owned = imod_prog_name(arguments.first().map_or("", String::as_str));
    let progname = progname_owned.as_bytes();
    let (nx, ny, mut nz): (i32, i32, i32);
    let mut nxyz = [0i32; 3];
    let mut mxyz = [0i32; 3];
    let nxyzst = [0i32, 0, 0];
    let mut ctfa = [0.0f32; 8193];
    let mut xmtf = [0.0f32; LIMMTF as usize];
    let mut ymtf = [0.0f32; LIMMTF as usize];
    let mut cell = [0.0f32; 6];
    let mut pixel = [0.0f32; 3];
    let mut ctfb = [0.0f32; 8193];
    let mut array: Vec<f32>;
    let mut sec_doses: Vec<f32> = Vec::new();
    let mut prior_doses: Vec<f32> = Vec::new();
    let mut iz_piece: Vec<i32>;
    let mut list_fake_sirt_iter: Vec<i32> = Vec::new();
    let mut line = [0u8; MRC_LABEL_SIZE];
    let num_pt_stock: [i32; LIMSTOCKCURVES as usize] = [36];
    let stock: [f32; 2 * LIMSTOCKPOINTS] = [
        0.0085, 0.98585, 0.0221, 0.94238, 0.0357, 0.89398, 0.0493, 0.83569, 0.0629, 0.76320,
        0.0765, 0.69735, 0.0901, 0.63647, 0.1037, 0.56575, 0.1173, 0.49876, 0.1310, 0.43843,
        0.1446, 0.38424, 0.1582, 0.34210, 0.1718, 0.30289, 0.1854, 0.26933, 0.1990, 0.23836,
        0.2126, 0.21318, 0.2262, 0.18644, 0.2398, 0.15756, 0.2534, 0.14863, 0.2670, 0.12485,
        0.2806, 0.11436, 0.2942, 0.09183, 0.3078, 0.08277, 0.3214, 0.07021, 0.3350, 0.05714,
        0.3486, 0.04388, 0.3622, 0.03955, 0.3759, 0.03367, 0.3895, 0.02844, 0.4031, 0.02107,
        0.4167, 0.02031, 0.4303, 0.01796, 0.4439, 0.00999, 0.4575, 0.01103, 0.4711, 0.00910,
        0.4898, 0.00741,
    ];

    //
    let mut in_file: Vec<u8> = Vec::new();
    let mut out_file: Vec<u8> = Vec::new();
    let mut mtf_file: Vec<u8> = Vec::new();
    let mut dose_file: Vec<u8> = Vec::new();
    let mut list_string: Vec<u8> = Vec::new();
    let mut title_str: String;
    let mut full_out: String;
    let in_file_str: String;
    let mut dose_file_str: String;
    let mode_map: [i32; 17] = [1, 2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 4, 0, 0, 0, 1];
    let mode_min: [i32; 4] = [0, -32768, 0, -65504];
    let mode_max: [i32; 4] = [256, 32768, 65536, 65504];
    let mut mode = 0i32;
    let mut num_pad: i32;
    let mut nx_pad: i32;
    let mut ny_pad: i32;
    let mut im_unit_out: i32;
    let mut iz_start: i32;
    let mut iz_end: i32;
    let num_zdo: i32;
    let mut nsize2 = 0i32;
    let mut iz_out_base: i32;
    let mut nsize: i32;
    let mut ind_stock = 0i32;
    let mut num_mtf: i32;
    let mut i: i32;
    let mut im: i32;
    let mut kk: i32;
    let mut ix_start: i32;
    let mut iy_start: i32;
    let mod_print: i32;
    let mut dmin_in = 0.0f32;
    let mut dmax_in = 0.0f32;
    let mut dmean: f32;
    let mut dmin: f32;
    let mut dmax: f32;
    let array_size: f32;
    let mut xfac = 1.0f32;
    let mut scale_fac = 1.0f32;
    let mut delta: f32;
    let mut ctf_inv_max = 4.0f32;
    let mut radius1 = 0.12f32;
    let mut sigma1 = 0.05f32;
    let mut radius2 = 1.0f32;
    let mut sigma2 = 0.0f32;
    let mut s: f32;
    let mut dmean_sum: f32;
    let mut dmean_in = 0.0f32;
    let mut dmin2 = 0.0f32;
    let mut dmax2 = 0.0f32;
    let mut dmean2 = 0.0f32;
    let mut atten: f32;
    let mut beta1: f32;
    let mut delta_rad = 0.0f32;
    let mut delx: f32;
    let mut dely: f32;
    let mut delz: f32;
    let mut xa: f32;
    let mut ya: f32;
    let mut za: f32;
    let mut za_sq: f32;
    let mut ya_sq: f32;
    let mut sigma1b = 0.0f32;
    let mut radius1b = 0.0f32;
    let mut base: f32;
    let mut pix_size_delta = [0.0f32; 3];
    let mut rad_scale: f32;
    let mut voxel_lim = 2147483000.0f32;
    let mut ampfac = 0.0f32;
    let mut amp_power = 0.0f32;
    let mut pixel_size: f32;
    let mut hole_size = 0.0f32;
    let mut voltage = 0.0f32;
    let mut focal_length = 0.0f32;
    let mut amp_radius = 0.0f32;
    let wavelength: f32;
    let mut fake_iter_use = 0.0f32;
    let fake_match_add = 0.3f32;
    let fake_alpha = 0.00195f32;
    let mut dose_per_image = 0.0f32;
    let physical_mem: f32;
    let mut dose_afac: f32;
    let mut dose_bfac: f32;
    let mut dose_cfac: f32;
    let mut dose_scale: f32;
    let mut dose_initial: f32;
    let crit_dose_scale200_kv = 0.8f32;
    let mut deconv_strength = 0.0f32;
    let mut defocus = 0.0f32;
    let mut snr_falloff = 0.7f32;
    let mut high_pass_nyq = 0.02f32;
    let mut dc_phase_shift = 0.0f32;
    let mut spherical_abs = 2.7f32;
    let mut diff_lim: f32;
    let mut expand_factor: f32;
    let mut ind: i32;
    let mut j: i32;
    let mut ierr: i32;
    let mut indf: i32;
    let mut ix: i32;
    let mut iy: i32;
    let mut iz: i32;
    let mut izLow: i32;
    let mut izHigh: i32;
    let mut izRead: i32;
    let mut dc_df_voltage = 300i32;
    let nx_dim: i32;
    let nz_pad: i32;
    let mut nyz_max: i32;
    let if_cut: i32;
    let if_phase: i32;
    let mut mode_out: i32;
    let mut mode_try: i32;
    let mut num_fake_sirt_iter = 0i32;
    let mut iter_num: i32;
    let mut num_iter_dec = 0i32;
    let mut invert_for_bidir: i32;
    let mut idose_file_type = -1i32;
    let mut ind_adoc = 0i32;
    let if_image_dose: i32;
    let mut montage = 0i32;
    let mut num_sect = 0i32;
    let mut i_type_adoc = 0i32;
    let mut i_verbose = 0i32;
    let mut inv_freq_unit = 0i32;
    let mut ind_work: i64 = 0;
    let idim: i64;
    let mut ibase: i64;
    let mut ix_base: i64;
    let mut i8: i64;
    let fft_input: bool;
    let convert_sigma: bool;
    let mut one_dfilter = 0i32;
    let mut r_weight = 0i32;
    let mut noise_pad = 0i32;
    let mut bidir_reversed: i32;
    let mut filter3d = 0i32;
    //
    let mut num_opt_args = 0i32;
    let mut num_non_opt_args = 0i32;
    //
    // fallbacks from ../../manpages/autodoc2man -3 2  mtffilter
    //
    let num_options: i32 = 42;
    let options: [&[u8]; 1] = [concat!(
        "input:InputFile:FN:@output:OutputFile:FN:@zrange:StartingAndEndingZ:IP:@",
        "mode:ModeToOutput:I:@3dfilter:FilterIn3D:B:@1dfilter:OneDimensionalFilter:B:@",
        "units:UnitsForFrequency:I:@lowpass:LowPassRadiusSigma:FP:@",
        "highpass:HighPassSigma:F:@radius1:FilterRadius1:F:@mtf:MtfFile:FN:@",
        "stock:StockCurve:I:@maxinv:MaximumInverse:F:@",
        "invrolloff:InverseRolloffRadiusSigma:FP:@xscale:XScaleFactor:F:@",
        "noise:NoisePadding:B:@denscale:DensityScaleFactor:F:@",
        "rweight:RWeightedFilter:B:@fake:FakeSIRTiterations:LI:@pixel:PixelSize:F:@",
        "expanded:ExpandedByFactor:F:@volt:Voltage:I:@dtype:TypeOfDoseFile:I:@",
        "dfile:DoseWeightingFile:FN:@dfixed:FixedImageDose:F:@initial:InitialDose:F:@",
        "bidir:BidirectionalNumViews:I:@reversed:ReversedBidirectional:B:@",
        "optimal:OptimalDoseScaling:F:@critical:CriticalDoseFactors:FT:@",
        "verbose:VerboseOutput:I:@deconv:DeconvolutionStrength:F:@snr:SNRFalloff:F:@",
        "dchigh:HighPassNyquist:F:@defocus:DefocusInMicrons:F:@dcphase:PhaseShift:F:@",
        "cs:SphericalAberration:F:@amplifier:AmplifierFactorAndPower:FP:@",
        "cutoff:CutoffForAmplifier:F:@phase:PhasePlateParameters:FT:@",
        "param:ParameterFile:PF:@help:usage:B:"
    )
    .as_bytes()];

    //
    // Pip startup: set error, parse options, do help output
    //
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
    //
    // Open image files.
    //
    if pip_get_in_out_file(b"InputFile", 0, &mut in_file) != 0 {
        exit_error(b"No input file specified");
    }

    ierr = pip_get_in_out_file(b"OutputFile", 1, &mut out_file);
    let have_out_file = ierr == 0;
    //
    unsafe {
        iiu_open(
            1,
            &String::from_utf8_lossy(&in_file),
            if ierr != 0 { "OLD" } else { "RO" },
        );
        iiu_print_header(1, std::ptr::null());
        iiu_ret_basic_head(
            1,
            nxyz.as_mut_ptr(),
            mxyz.as_mut_ptr(),
            &mut mode,
            &mut dmin_in,
            &mut dmax_in,
            &mut dmean_in,
        );
    }
    nx = nxyz[0];
    ny = nxyz[1];
    nz = nxyz[2];
    fft_input = mode == 3 || mode == 4;
    if !fft_input {
        pip_get_boolean(b"FilterIn3D", &mut filter3d);
    }

    // Get physical memory and use same strategy as ctfplotter/newstack for setting a limit
    // to what can be used; do this in bytes like the others and convert to words
    physical_mem = standard_memory_limit_mb(30000) as f32;
    if physical_mem > 0. && filter3d != 0 {
        voxel_lim = ((1024. * 1024.) * physical_mem as f64 / 4.) as f32;
    }

    pip_get_integer(b"VerboseOutput", &mut i_verbose);
    pip_get_boolean(b"NoisePadding", &mut noise_pad);
    pip_get_boolean(b"OneDimensionalFilter", &mut one_dfilter);
    pip_get_boolean(b"RWeightedFilter", &mut r_weight);
    pip_get_float(b"DeconvolutionStrength", &mut deconv_strength);
    //
    // Read in and check fake SIRT iterations
    if pip_get_string(b"FakeSIRTiterations", &mut list_string) == 0 {
        match parselist(&String::from_utf8_lossy(&list_string)) {
            Ok(values) => {
                num_fake_sirt_iter = values.len() as i32;
                list_fake_sirt_iter = values;
            }
            Err(_) => exit_error(b"Parsing list of fake SIRT iterations"),
        }
        num_iter_dec = 2;
        ix = 1;
        while ix <= num_fake_sirt_iter {
            if list_fake_sirt_iter[(ix - 1) as usize] > 99 {
                num_iter_dec = 3;
            }
            if list_fake_sirt_iter[(ix - 1) as usize] < 1
                || list_fake_sirt_iter[(ix - 1) as usize] > 999
            {
                exit_error(b"Fake SIRT iterations must be between 1 and 999");
            }
            ix += 1;
        }
        if num_fake_sirt_iter > 1 && !have_out_file {
            exit_error(
                b"You cannot use multiple fake SIRT iterations when rewriting the input file",
            );
        }
    }
    if r_weight != 0 {
        one_dfilter = 1;
    }
    if (one_dfilter != 0 || num_fake_sirt_iter > 0) && fft_input {
        exit_error(b"You cannot do 1-D or R-weighted or fake SIRT filtering with FFT input data");
    }
    if (one_dfilter != 0 || num_fake_sirt_iter > 0) && filter3d != 0 {
        exit_error(
            b"You cannot specify 3-D filtering with 1-D, R-weighted, or fake SIRT filtering",
        );
    }
    //
    iz_start = 1;
    iz_end = nz;
    ierr = pip_get_two_integers(b"StartingAndEndingZ", &mut iz_start, &mut iz_end);
    if fft_input && ierr == 0 {
        exit_error(b"FFT input file is assumed to be 3D FFT, no Z range allowed");
    }

    if iz_start > iz_end || iz_start < 1 || iz_end > nz {
        exit_error(b"Illegal starting or ending Z values to filter");
    }
    num_zdo = iz_end + 1 - iz_start;
    iz_out_base = 1;
    //
    // get padded size and make sure it will fit
    //
    if fft_input {
        nx_pad = (nx - 1) * 2;
        ny_pad = ny;
        nz_pad = nz;
    } else {
        num_pad = 40.max(2 * (nx / 40));
        nx_pad = nice_frame(nx + num_pad, 2, nice_fft_limit());
        num_pad = 40.max(2 * (ny / 40));
        ny_pad = nice_frame(ny + num_pad, 2, nice_fft_limit());
        num_pad = 40.max(2 * (nz / 40));
        nz_pad = nice_frame(num_zdo + num_pad, 2, nice_fft_limit());
    }
    nx_dim = nx_pad + 2;

    if filter3d != 0 {
        ind_work = nx_dim as i64 * ny_pad as i64 * nz_pad as i64;
        idim = ind_work + nx_dim as i64 * nz_pad as i64 + 16;
    } else {
        let mut idim_tmp = (nx_dim * ny_pad) as i64 + 16;
        if noise_pad != 0 {
            ind_work = idim_tmp;
            idim_tmp += (2 * nx.max(ny) + (nx_pad - nx) + (ny_pad - ny) + 16) as i64;
        }
        idim = idim_tmp;
    }

    if idim as f32 > voxel_lim {
        exit_error(b"Padded volume is bigger than the memory limit   [MTF1]");
    }
    array = Vec::new();
    if array.try_reserve_exact(idim as usize).is_err() {
        exit_error(b"Failed to allocate memory for image data   [MTF1]");
    }
    array.resize(idim as usize, 0.);

    nyz_max = ny_pad;
    if filter3d != 0 || fft_input {
        nyz_max = ny_pad.max(nz_pad);
    }
    if one_dfilter != 0 {
        nyz_max = nx_pad;
    }
    //
    im_unit_out = 1;
    if have_out_file {
        im_unit_out = 3;
        if num_fake_sirt_iter < 2 {
            unsafe {
                iiu_open(3, &String::from_utf8_lossy(&out_file), "NEW");
                iiu_trans_header(3, 1);
            }
        }
    }
    //
    // set up the ctf scaling as usual: but delta needs to be maximum
    // frequency over ctf array size
    //
    nsize = 8192.min(1024.max(2 * nx_pad.max(nyz_max)));
    array_size = nsize as f32;
    nsize += 1;
    delta = (0.71 / array_size as f64) as f32;
    if fft_input || filter3d != 0 {
        delta = (0.87 / array_size as f64) as f32;
    }
    if one_dfilter != 0 {
        delta = (0.505 / array_size as f64) as f32;
    }
    //
    // set default null mtf curve
    //
    ind_stock = 0;
    mtf_file.clear();
    let mut have_mtf_file = false;
    num_mtf = 3;
    i = 1;
    while i <= 3 {
        xmtf[(i - 1) as usize] = ((i - 1) as f64 * 0.25) as f32;
        ymtf[(i - 1) as usize] = 1.;
        i += 1;
    }
    if pip_get_string(b"MtfFile", &mut mtf_file) == 0 {
        have_mtf_file = true;
    }
    if pip_get_integer(b"StockCurve", &mut ind_stock) == 0 {
        if ind_stock <= 0 || ind_stock > LIMSTOCKCURVES {
            exit_error(b"Illegal number entered for stock curve");
        }
        if have_mtf_file {
            exit_error(b"You cannot enter both an MTF file and a stock curve #");
        }

        //
        // if stock curve requested, find index and copy values
        //
        ind = 0;
        for i in 0..ind_stock - 1 {
            ind += num_pt_stock[i as usize];
        }
        for i in 0..num_pt_stock[(ind_stock - 1) as usize] {
            xmtf[i as usize] = stock[(ind * 2) as usize];
            ymtf[i as usize] = stock[(ind * 2 + 1) as usize];
            ind += 1;
        }
        num_mtf = num_pt_stock[(ind_stock - 1) as usize];
    }
    //
    if have_mtf_file {
        //
        // or read from file if one provided
        //
        let name = String::from_utf8_lossy(&mtf_file).into_owned();
        let Ok(file) = std::fs::File::open(&name) else {
            exit_error(&c_format_bytes(
                "Opening file with MTF curve, %s",
                &[CArg::Bytes(&mtf_file)],
            ));
        };
        let mut reader = BufReader::new(file);
        num_mtf = 0;
        ierr = read_lines_for_values(
            &mut reader,
            &mut num_mtf,
            LIMMTF as usize,
            RLFV_SEPARATE_LINES,
            "ff",
            &mut [
                ReadValueArray::Floats(&mut xmtf),
                ReadValueArray::Floats(&mut ymtf),
            ],
        );
        if ierr != 0 {
            if let Err(message) = exit_from_value_read_error(ierr, "MTF values") {
                exit_error(message.as_bytes());
            }
        }
    }
    //
    pip_get_float(b"XScaleFactor", &mut xfac);
    for i in 0..num_mtf {
        xmtf[i as usize] *= xfac;
        // print *,xmtf(i), ymtf(i)
    }
    //
    ctfa[0] = 1.;
    im = 0;
    s = 0.;
    j = 1;
    while j < nsize {
        s += delta;
        if im < num_mtf - 2 && s > xmtf[(im + 1).min(num_mtf - 1) as usize] {
            im += 1;
        }
        ctfa[j as usize] = (0.0f64).max(
            (ymtf[im as usize]
                + (ymtf[(im + 1) as usize] - ymtf[im as usize]) * (s - xmtf[im as usize])
                    / (xmtf[(im + 1) as usize] - xmtf[im as usize])) as f64,
        ) as f32;
        j += 1;
    }
    if (ind_stock != 0 || have_mtf_file) && r_weight != 0 {
        exit_error(b"You cannot do MTF filtering with an R-weighted filter");
    }

    // write (*,'(10f7.4)') (ctfa(j), j=1, nsize)

    // Handle output mode entry
    mode_out = mode;
    if mode == 16 {
        mode_out = 0;
        let _ = ImodFile::Stdout
            .write_all(b"WARNING: MTFFILTER - RGB input data are being converted to gray scale\n");
    }
    if pip_get_integer(b"ModeToOutput", &mut mode_out) == 0 {
        if im_unit_out == 1 {
            exit_error(b"You cannot enter -mode when rewriting to the input file");
        }
        if fft_input {
            exit_error(b"You cannot enter -mode with an FFT input file");
        }
        set_float_output_for_entered_mode(mode_out);
        if !((0..=2).contains(&mode_out) || mode_out == 6 || mode_out == 12) {
            exit_error(b"Output mode must be 0, 1, 2, 6, or 12");
        }
    }

    iiu_ret_delta(1, &mut pix_size_delta);
    pixel_size = (pix_size_delta[0] as f64 / 10.) as f32;
    if pip_get_float(b"PixelSize", &mut pixel_size) == 0 {
        expand_factor = 1.;
        pip_get_float(b"ExpandedByFactor", &mut expand_factor);
        pixel_size /= expand_factor;
        let _ = ImodFile::Stdout.write_all(&c_format_bytes(
            "Adjusted pixel size by expansion factor to %.3f nm\n",
            &[CArg::Dbl(pixel_size as f64)],
        ));
    }
    pip_get_float(b"DensityScaleFactor", &mut scale_fac);
    pip_get_integer(b"Voltage", &mut dc_df_voltage);
    pip_get_integer(b"UnitsForFrequency", &mut inv_freq_unit);
    convert_sigma = inv_freq_unit.abs() > 2;

    if_image_dose = 1 - pip_get_float(b"FixedImageDose", &mut dose_per_image);
    pip_get_integer(b"TypeOfDoseFile", &mut idose_file_type);
    if if_image_dose > 0 && idose_file_type > 0 {
        exit_error(b"You cannot enter both an image dose and type of dose file for dose weighting");
    }
    //
    // When dose weighting, skip options that might be in standard com file for 2D filtering
    //
    ix = 1;
    if if_image_dose == 0 && idose_file_type <= 0 {
        pip_get_float(b"MaximumInverse", &mut ctf_inv_max);
        ierr = pip_get_two_floats(b"InverseRolloffRadiusSigma", &mut radius1, &mut sigma1);
        if ierr == 0 {
            convert_freq_unit(inv_freq_unit, pixel_size, &mut radius1);
            if convert_sigma {
                convert_freq_unit(inv_freq_unit, pixel_size, &mut sigma1);
            }
        }
        ix = pip_get_two_floats(b"LowPassRadiusSigma", &mut radius2, &mut sigma2);
        if ix == 0 {
            convert_freq_unit(inv_freq_unit, pixel_size, &mut radius2);
            if convert_sigma {
                convert_freq_unit(inv_freq_unit, pixel_size, &mut sigma2);
            }
        }
    }
    iy = pip_get_float(b"FilterRadius1", &mut radius1b);
    if iy == 0 {
        convert_freq_unit(inv_freq_unit, pixel_size, &mut radius1b);
    }
    iz = pip_get_float(b"HighPassSigma", &mut sigma1b);
    if iz == 0 && convert_sigma {
        convert_freq_unit(inv_freq_unit, pixel_size, &mut sigma1b);
    }
    ierr = pip_get_two_floats(b"AmplifierFactorAndPower", &mut ampfac, &mut amp_power);
    if_cut = 1 - pip_get_float(b"CutoffForAmplifier", &mut amp_radius);
    if if_cut > 0 {
        convert_freq_unit(inv_freq_unit, pixel_size, &mut amp_radius);
    }
    if_phase = 1 - pip_get_three_floats(
        b"PhasePlateParameters",
        &mut hole_size,
        &mut voltage,
        &mut focal_length,
    );
    //
    // Check and handle amplifier setting or regular radius/sigma based filter
    if ierr == 0 || if_cut != 0 || if_phase != 0 {
        if if_image_dose > 0 || idose_file_type > 0 {
            exit_error(b"You cannot use dose weighting with an amplifier filter");
        }
        if ix == 0 || iy == 0 || iz == 0 {
            exit_error(b"You cannot enter parameters for both a regular and an amplifier filter");
        }
        if ierr != 0 || (if_cut == 0 && if_phase == 0) {
            exit_error(
                b"To use the amplifier filter you must enter -amplifier and either -phase or -cutoff",
            );
        }
        if if_cut != 0 && if_phase != 0 {
            exit_error(b"You cannot enter both -cutoff and -phase for amplifier filter");
        }
        if deconv_strength != 0. {
            exit_error(
                b"You cannot enter parameters for both a deconvolution and an amplifier filter",
            );
        }
        if if_phase != 0 {
            wavelength = (1.226
                / (1000. * voltage as f64 * (1.0 + 0.9788e-3 * voltage as f64)).sqrt())
                as f32;
            amp_radius = (pixel_size as f64 * (hole_size as f64 / 2.0)
                / (wavelength as f64 * focal_length as f64 * 1e6)) as f32;
        }
        amplifier(
            amp_radius,
            ampfac,
            amp_power,
            &mut ctfb,
            nx_pad,
            nyz_max,
            &mut delta_rad,
            &mut nsize2,
        );
    } else if deconv_strength != 0. {
        if ix == 0 || iy == 0 || iz == 0 {
            exit_error(
                b"You cannot enter parameters for both a regular and a deconvolution filter",
            );
        }
        if if_image_dose > 0 || idose_file_type > 0 {
            exit_error(b"You cannot do dose weighting with a deconvolution filter");
        }
        if pip_get_float(b"DefocusInMicrons", &mut defocus) != 0 {
            exit_error(b"You must enter -defocus for a deconvolution filter");
        }
        pip_get_float(b"SNRFalloff", &mut snr_falloff);
        pip_get_float(b"PhaseShift", &mut dc_phase_shift);
        pip_get_float(b"SphericalAberration", &mut spherical_abs);
        pip_get_float(b"HighPassNyquist", &mut high_pass_nyq);
        deconv_filter(
            deconv_strength,
            snr_falloff,
            high_pass_nyq,
            (1000. * dc_df_voltage as f64) as f32,
            (spherical_abs as f64 * 1.0e-3) as f32,
            (-defocus as f64 * 1.0e-6) as f32,
            (dc_phase_shift as f64 * 3.14159 / 180.) as f32,
            (pixel_size as f64 * 10.) as f32,
            &mut ctfb,
            delta,
            nsize,
        );
        delta_rad = delta;
    } else if if_image_dose == 0 && idose_file_type <= 0 {
        xcorr_set_ctf_no_scl(
            sigma1b,
            sigma2,
            radius1b,
            radius2,
            &mut ctfb,
            nx_pad,
            nyz_max,
            &mut delta_rad,
            &mut nsize2,
        );
    }
    //
    // For dose weighting, first initialize and check for conflicts
    invert_for_bidir = 0;
    dose_afac = 0.;
    dose_bfac = 0.;
    dose_cfac = 0.;
    dose_scale = 1.;
    dose_initial = 0.;
    if if_image_dose > 0 || idose_file_type > 0 {
        invert_for_bidir = 0;
        bidir_reversed = 0;
        dose_afac = 0.;
        dose_bfac = 0.;
        dose_cfac = 0.;
        dose_scale = 1.;
        dose_initial = 0.;
        if filter3d != 0
            || one_dfilter != 0
            || r_weight != 0
            || ind_stock != 0
            || have_mtf_file
            || fft_input
            || num_fake_sirt_iter > 1
        {
            exit_error(
                b"You cannot use dose weighting with inverse, 1-D, 3-D, R-weighted, or fake SIRT filtering",
            );
        }
        if ix == 0 || iy == 0 || iz == 0 {
            exit_error(b"You cannot enter parameters for both a regular filter and dose weighting");
        }
        //
        // Get the parameters, allocate arrays
        pip_get_integer(b"BidirectionalNumViews", &mut invert_for_bidir);
        pip_get_boolean(b"ReversedBidirectional", &mut bidir_reversed);
        if bidir_reversed != 0 {
            invert_for_bidir = -invert_for_bidir;
        }
        pip_get_three_floats(
            b"CriticalDoseFactors",
            &mut dose_afac,
            &mut dose_bfac,
            &mut dose_cfac,
        );
        pip_get_float(b"OptimalDoseScaling", &mut dose_scale);
        if dc_df_voltage == 200 {
            dose_scale *= crit_dose_scale200_kv;
        }
        if dc_df_voltage != 200 && dc_df_voltage != 300 {
            exit_error(b"The voltage entry must be either 200 or 300");
        }

        pip_get_float(b"InitialDose", &mut dose_initial);
        sec_doses = vec![0.0f32; nz as usize];
        prior_doses = vec![0.0f32; nz as usize];
        iz_piece = vec![0i32; nz as usize];
        if if_image_dose > 0 {
            idose_file_type = 0;
            if dose_per_image <= 0. {
                exit_error(b"Dose per image must be positive");
            }
            for ind in 0..nz {
                sec_doses[ind as usize] = dose_per_image;
            }
        } else {
            // If no dose file, then it is error unless HDF with autodoc type given; get index
            if pip_get_string(b"DoseWeightingFile", &mut dose_file) != 0 {
                if idose_file_type < 4 || unsafe { iiu_file_type(1) } != 5 {
                    exit_error(b"You must enter a dose weighting file");
                }
                ind_adoc = unsafe { iiu_ret_adoc_index(1, 0, 0) };
                if ind_adoc <= 0 {
                    exit_error(b"Getting index for accessing metadata in HDF file");
                }
                if adoc_set_current(ind_adoc) < 0 {
                    exit_error(b"Setting metadata in HDF file as current autodoc");
                }
                if adoc_get_image_meta_info(&mut montage, &mut num_sect, &mut i_type_adoc) < 0 {
                    exit_error(b"Metadata in HDF file is not of the appropriate type");
                }
            } else {
                // Read in plain numeric dose file of 3 types
                if idose_file_type < 4 {
                    let name = String::from_utf8_lossy(&dose_file).into_owned();
                    let Ok(file) = std::fs::File::open(&name) else {
                        exit_error(&c_format_bytes(
                            "Opening dose file %s",
                            &[CArg::Bytes(&dose_file)],
                        ));
                    };
                    let mut reader = BufReader::new(file);

                    ind = nz;
                    if idose_file_type < 2 {
                        ierr = read_lines_for_values(
                            &mut reader,
                            &mut ind,
                            nz as usize,
                            RLFV_SEPARATE_LINES,
                            "f",
                            &mut [ReadValueArray::Floats(&mut sec_doses)],
                        );
                    } else {
                        let mut pair = [
                            ReadValueArray::Floats(&mut prior_doses),
                            ReadValueArray::Floats(&mut sec_doses),
                        ];
                        ierr = read_lines_for_values(
                            &mut reader,
                            &mut ind,
                            nz as usize,
                            RLFV_SEPARATE_LINES,
                            "ff",
                            &mut pair,
                        );
                    }
                    if ierr != 0 {
                        if let Err(message) = exit_from_value_read_error(ierr, "dose file") {
                            exit_error(message.as_bytes());
                        }
                    }
                    if idose_file_type == 3 {
                        for iz in 0..nz {
                            sec_doses[iz as usize] -= prior_doses[iz as usize];
                        }
                    }
                } else {
                    // If given an apparent extension of original file, strip extension from input
                    // file and add this plus .mdoc
                    in_file_str = String::from_utf8_lossy(&in_file).into_owned();
                    dose_file_str = String::from_utf8_lossy(&dose_file).into_owned();
                    iz = in_file_str.rfind('.').map_or(-1, |p| p as i32);
                    iy = dose_file_str.find('.').map_or(-1, |p| p as i32);
                    j = -1;
                    if iy >= 0 {
                        full_out = dose_file_str.clone();
                        full_out.truncate(iy as usize);
                        j = in_file_str.rfind(&full_out).map_or(-1, |p| p as i32);
                    }
                    ix = dose_file_str.len() as i32;
                    if ix <= 5 && iy == 0 && iz > 0 {
                        full_out = in_file_str.clone();
                        full_out.truncate(iz as usize);
                        dose_file_str = full_out + &String::from_utf8_lossy(&dose_file) + ".mdoc";
                    } else if dose_file_str.find('_') == Some(0) && j > 0 && iz == j + iy {
                        full_out = in_file_str.clone();
                        full_out.truncate(j as usize);
                        dose_file_str =
                            full_out + &dose_file_str[iy as usize..ix as usize] + ".mdoc";
                    }
                    ind_adoc = adoc_open_image_metadata(
                        dose_file_str.as_bytes(),
                        0,
                        &mut montage,
                        &mut num_sect,
                        &mut i_type_adoc,
                    );
                    if ind_adoc < 0 {
                        let mut full_out =
                            format!("trying to access {dose_file_str} as an mdoc file");
                        if ind_adoc == -1 {
                            full_out += "; error opening or reading file";
                        }
                        if ind_adoc == -2 {
                            full_out += "; file does not exist";
                        }
                        if ind_adoc == -3 {
                            full_out += "; inappropriate type of metadata file";
                        }
                        exit_error(full_out.as_bytes());
                    }
                }
            }
            //
            // Now access the dose information from the metadata
            if idose_file_type >= 4 {
                for iz in 0..nz {
                    iz_piece[iz as usize] = iz;
                }
                ierr = get_metadata_weighting_doses(
                    ind_adoc,
                    i_type_adoc,
                    nz,
                    &iz_piece,
                    invert_for_bidir,
                    &mut prior_doses,
                    &mut sec_doses,
                );
                if ierr > 0 {
                    exit_error(b"Getting doses from metadata");
                }
                if ierr < 0 {
                    let _ = ImodFile::Stdout.write_all(
                        b"WARNING: MTFFILTER - Assuming the images were acquired in order \
                          because the metadata has no PriorRecordDose or DateTime entries \
                          and -bidir was not entered\n",
                    );
                }
            }
        }
        //
        // Need to compute cumulative doses if just image doses entered
        if idose_file_type < 2 {
            prior_doses_from_image_doses(&sec_doses.clone(), invert_for_bidir, &mut prior_doses);
        }
        for iz in 0..nz {
            prior_doses[iz as usize] += dose_initial;
        }
        if i_verbose > 0 {
            let _ = ImodFile::Stdout.write_all(b"View   Prior dose  image dose\n");
            for iz in 0..nz {
                let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                    "(%4d%12.3f%12.3f\n",
                    &[
                        CArg::Int((iz + 1).into()),
                        CArg::Dbl(prior_doses[iz as usize] as f64),
                        CArg::Dbl(sec_doses[iz as usize] as f64),
                    ],
                ));
            }
        }
    }
    pip_done();

    //
    // Compute the inverse filter and multiply by the other fliter in ctfb
    beta1 = 0.0;
    if sigma1 > 1.0e-6 {
        beta1 = (-0.5 / (sigma1 * sigma1) as f64) as f32;
    }

    mod_print = (0.0252 / delta as f64) as i32;
    s = 0.;
    rad_scale = (delta as f64 / 0.5) as f32;
    if radius2 > 0. {
        rad_scale = (delta as f64 / (0.5f64).min(radius2 as f64)) as f32;
    }
    if num_fake_sirt_iter == 1 {
        fake_iter_use = adjusted_fake_iter(list_fake_sirt_iter[0]);
    }
    if num_fake_sirt_iter < 2 && idose_file_type < 0 {
        let _ = ImodFile::Stdout
            .write_all(b"\nThe overall filter being applied is:\nradius  multiplier\n");
    }
    for j in 0..nsize {
        atten = 1.;
        if s > radius1 {
            atten = ((beta1 * (s - radius1) * (s - radius1)) as f64).exp() as f32;
        }
        if r_weight != 0 {
            ctfa[j as usize] = j as f32 * rad_scale;
            if j == 0 {
                ctfa[j as usize] = (0.2 * rad_scale as f64) as f32;
            }
        } else if ctfa[j as usize] < 0.01 {
            ctfa[j as usize] = 1.;
        } else {
            ctfa[j as usize] = (1.
                + atten as f64 * ((ctf_inv_max as f64).min(1. / ctfa[j as usize] as f64) - 1.))
                as f32;
        }
        if num_fake_sirt_iter == 1 && s >= fake_alpha {
            ctfa[j as usize] = (ctfa[j as usize] as f64
                * (1.
                    - (1. - fake_alpha as f64 / s as f64)
                        .powf((fake_iter_use + fake_match_add) as f64)))
                as f32;
        }
        if delta_rad != 0. {
            indf = (s / delta_rad) as i32;
            xa = (s / delta_rad - indf as f32) as f32;
            ctfa[j as usize] = (ctfa[j as usize] as f64
                * ((1. - xa as f64) * ctfb[indf as usize] as f64
                    + xa as f64 * ctfb[(indf + 1) as usize] as f64))
                as f32;
        }
        if (j % mod_print) == 0 && s <= 0.5 && num_fake_sirt_iter < 2 && idose_file_type < 0 {
            let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                "%8.4f%8.4f\n",
                &[CArg::Dbl(s as f64), CArg::Dbl(ctfa[j as usize] as f64)],
            ));
        }
        s += delta;
        ctfa[j as usize] = scale_fac * ctfa[j as usize];
    }
    //
    // Save base filter in ctfb for setting  up different fake iterations
    for j in 0..nsize {
        ctfb[j as usize] = ctfa[j as usize];
    }

    title_str = "MTFFILTER: Filtered by inverse of MTF".to_string();
    if ind_stock == 0 && !have_mtf_file {
        title_str = "MTFFILTER: Frequency filtered".to_string();
    }
    if if_image_dose > 0 || idose_file_type > 0 {
        title_str = "MTFFILTER: Dose weight filtered".to_string();
    }
    //
    iter_num = 0;
    while iter_num < 1.max(num_fake_sirt_iter) {
        dmean_sum = 0.;
        dmax = -1.0e10;
        dmin = 1.0e10;
        ix_start = (nx_pad - nx) / 2;
        iy_start = (ny_pad - ny) / 2;
        //
        // When doing multiple iteration numbers, compute final filter function and open file
        if num_fake_sirt_iter > 1 {
            fake_iter_use = adjusted_fake_iter(list_fake_sirt_iter[iter_num as usize]);
            s = 0.;
            for j in 0..nsize {
                ctfa[j as usize] = ctfb[j as usize];
                if s >= fake_alpha {
                    ctfa[j as usize] = (ctfa[j as usize] as f64
                        * (1.
                            - (1. - fake_alpha as f64 / s as f64)
                                .powf((fake_iter_use + fake_match_add) as f64)))
                        as f32;
                }
                s += delta;
            }
            let suffix = if num_iter_dec == 2 {
                c_format_bytes(
                    "%02d",
                    &[CArg::Int(list_fake_sirt_iter[iter_num as usize].into())],
                )
            } else {
                c_format_bytes(
                    "%03d",
                    &[CArg::Int(list_fake_sirt_iter[iter_num as usize].into())],
                )
            };
            let mut full = out_file.clone();
            full.extend_from_slice(&suffix);
            unsafe {
                iiu_open(3, &String::from_utf8_lossy(&full), "NEW");
                iiu_trans_header(3, 1);
            }
        }
        //
        // take care of header if writing a new file
        //
        if im_unit_out == 3 {
            iz_out_base = iz_start;
            iiu_ret_cell(1, &mut cell);
            iiu_ret_delta(1, &mut pixel);
            //
            // change mxyz if it matches existing nz; set cell size to keep
            // pixel spacing the same
            //
            if mxyz[2] == nz {
                mxyz[2] = num_zdo;
            }
            cell[2] = mxyz[2] as f32 * pixel[2];
            nz = num_zdo;
            nxyz[2] = nz;
            iiu_alt_size(3, &nxyz, &nxyzst);
            iiu_alt_sample(3, &mxyz);
            iiu_alt_cell(3, &cell);
            iiu_alt_mode(im_unit_out, mode_out);
        }
        //
        // Proceed to filter volume
        if filter3d == 0 {
            delx = (0.5 / (nx as f64 - 1.)) as f32;
            dely = (1. / ny as f64) as f32;
            delz = (1. / nz as f64) as f32;
            kk = iz_start - 1;
            while kk <= iz_end - 1 {
                //
                // Compute dose weighting filter for this section
                if idose_file_type >= 0 {
                    dose_weight_filter(
                        prior_doses[kk as usize],
                        prior_doses[kk as usize] + sec_doses[kk as usize],
                        (10. * pixel_size as f64) as f32,
                        dose_afac,
                        dose_bfac,
                        dose_cfac,
                        dose_scale,
                        &mut ctfa,
                        LIMMTF,
                        0.71,
                        &mut delta,
                    );
                    let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                        "View%4d  prior dose%9.2f\n",
                        &[
                            CArg::Int((kk + 1).into()),
                            CArg::Dbl(prior_doses[kk as usize] as f64),
                        ],
                    ));
                    if i_verbose > 0 {
                        let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                            "Dose weight filter for view %d\n",
                            &[CArg::Int((kk + 1).into())],
                        ));
                        for iy in 0..=40 {
                            s = ((0.5 * iy as f64) / 40.) as f32;
                            indf = (s / delta + 0.5) as i32;
                            let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                                "%7.3f%9.4f)\n",
                                &[CArg::Dbl(s as f64), CArg::Dbl(ctfa[indf as usize] as f64)],
                            ));
                            if ctfa[indf as usize] == 0. {
                                break;
                            }
                        }
                    }
                }
                //
                // print *,'reading section', kk
                unsafe {
                    iiu_set_position(1, kk, 0);
                    if iiu_read_section(1, array.as_mut_ptr().cast()) != 0 {
                        exit_error(b"Reading image file");
                    }
                }
                if fft_input {
                    //
                    // Filter the 3D fft - assuming the usual ordering
                    //
                    ind = 0;
                    za = (delz * kk as f32 - 0.5) as f32;
                    za_sq = za * za;
                    for iy in 0..ny {
                        ya = dely * iy as f32 - 0.5;
                        ya_sq = ya * ya;
                        for ix in 0..nx {
                            xa = delx * ix as f32;
                            s = ((xa * xa + ya_sq + za_sq) as f64).sqrt() as f32;
                            indf = (s / delta + 0.5) as i32;
                            array[ind as usize] *= ctfa[indf as usize];
                            array[(ind + 1) as usize] *= ctfa[indf as usize];
                            ind += 2;
                        }
                    }

                    let mut data = MrcData::F(std::mem::take(&mut array));
                    let stats = full_array_min_max_mean(&mut data, MRC_MODE_COMPLEX_FLOAT, nx, ny);
                    let MrcData::F(back) = data else {
                        unreachable!()
                    };
                    array = back;
                    if let Some((a, b, c)) = stats {
                        dmin2 = a;
                        dmax2 = b;
                        dmean2 = c;
                    }
                } else {
                    //
                    // Do ordinary 2D filter with padding/tapering, FFT, filter, repack
                    //
                    if noise_pad != 0 {
                        let (head, tail) = array.split_at_mut(ind_work as usize);
                        slice_noise_taper_pad(
                            PadIn::InPlace,
                            MRC_MODE_FLOAT,
                            nx,
                            ny,
                            head,
                            nx_pad + 2,
                            nx_pad,
                            ny_pad,
                            80,
                            5,
                            tail,
                        );
                    } else {
                        slice_taper_out_pad(
                            PadIn::InPlace,
                            MRC_MODE_FLOAT,
                            nx,
                            ny,
                            &mut array,
                            nx_pad + 2,
                            nx_pad,
                            ny_pad,
                            0,
                            0.,
                        );
                    }
                    //
                    if one_dfilter != 0 {
                        let _ = odfftc(&mut array, nx_pad, ny_pad, 0);
                        fft_filter_1d(&mut array, nx_dim / 2, ny_pad, &ctfa, delta);
                        let _ = odfftc(&mut array, nx_pad, ny_pad, 1);
                    } else {
                        // print *,'taking fft'
                        let _ = todfftc(&mut array, nx_pad, ny_pad, 0);
                        xcorr_filter_part(
                            FilterIn::InPlace,
                            &mut array,
                            nx_pad,
                            ny_pad,
                            &ctfa,
                            delta,
                        );
                        //
                        // print *,'taking back fft'
                        let _ = todfftc(&mut array, nx_pad, ny_pad, 1);
                    }

                    // print *,'repack, set density, write'
                    // `repackFloatImage(array, array, ...)` repacks in place;
                    // the translated routine cannot alias its two arguments,
                    // so the source region is copied first.  With `nbin` 1 the
                    // copy is a pure move of the selected rows, so the result
                    // is the same bytes (`mtffilter.cpp:758`).
                    let source: Vec<u8> =
                        array.iter().flat_map(|value| value.to_ne_bytes()).collect();
                    let mut dest = vec![0u8; source.len()];
                    repack_float_image(
                        &mut dest,
                        &source,
                        nx_dim,
                        ix_start,
                        ix_start + nx - 1,
                        iy_start,
                        iy_start + ny - 1,
                    );
                    for (value, chunk) in array.iter_mut().zip(dest.chunks_exact(4)) {
                        *value = f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    }
                    array_min_max_mean(
                        &array,
                        nx,
                        ny,
                        0,
                        nx - 1,
                        0,
                        ny - 1,
                        &mut dmin2,
                        &mut dmax2,
                        &mut dmean2,
                    );
                }
                //
                // Write out lattice.
                //
                unsafe {
                    iiu_set_position(im_unit_out, kk + 1 - iz_out_base, 0);
                    iiu_write_section(im_unit_out, array.as_mut_ptr().cast());
                }
                //
                dmax = if dmax > dmax2 { dmax } else { dmax2 };
                dmin = if dmin < dmin2 { dmin } else { dmin2 };
                dmean_sum += dmean2;
                kk += 1;
            }
        } else {
            //
            // Taking 3d fft and filtering
            // recast some variables to steal code from taperoutvol
            izLow = iz_start - 1;
            izHigh = iz_end - 1;
            iz_start = -((nz_pad - nz) / 2);
            iz_end = iz_start + nz_pad - 1;
            iz = iz_start;
            while iz <= iz_end {
                izRead = izLow.max(izHigh.min(iz));
                ibase = nx_dim as i64 * ny_pad as i64 * (iz - iz_start) as i64;
                unsafe {
                    iiu_set_position(1, izRead, 0);
                    if iiu_read_section(1, array[ibase as usize..].as_mut_ptr().cast()) != 0 {
                        exit_error(b"Reading image file");
                    }
                }
                slice_taper_out_pad(
                    PadIn::InPlace,
                    MRC_MODE_FLOAT,
                    nx,
                    ny,
                    &mut array[ibase as usize..],
                    nx_dim,
                    nx_pad,
                    ny_pad,
                    1,
                    dmean_in,
                );
                if iz < izLow || iz > izHigh {
                    if iz < izLow {
                        atten = (iz - iz_start) as f32 / (izLow - iz_start) as f32;
                    } else {
                        atten = (iz_end - iz) as f32 / (iz_end - izHigh) as f32;
                    }
                    base = ((1. - atten as f64) * dmean_in as f64) as f32;
                    for iy in 0..ny_pad {
                        ix_base = ibase + iy as i64 * nx_dim as i64;
                        i8 = ix_base;
                        while i8 < ix_base + nx_pad as i64 {
                            array[i8 as usize] = base + atten * array[i8 as usize];
                            i8 += 1;
                        }
                    }
                }
                iz += 1;
            }
            //
            // Take fft and filter and inverse fft
            {
                let (head, tail) = array.split_at_mut(ind_work as usize);
                thrdfftc(head, tail, nx_pad, ny_pad, nz_pad, 0);
            }
            fft_filter_3d(&mut array, nx_dim / 2, ny_pad, nz_pad, &ctfa, delta);
            {
                let (head, tail) = array.split_at_mut(ind_work as usize);
                thrdfftc(head, tail, nx_pad, ny_pad, nz_pad, -1);
            }
            //
            // repack and write
            iz = izLow;
            while iz <= izHigh {
                unsafe { iiu_set_position(im_unit_out, iz, 0) };
                ibase = nx_dim as i64 * ny_pad as i64 * (iz - iz_start) as i64;
                let source: Vec<u8> = array[ibase as usize..]
                    .iter()
                    .flat_map(|value| value.to_ne_bytes())
                    .collect();
                let mut dest = vec![0u8; source.len()];
                repack_float_image(
                    &mut dest,
                    &source,
                    nx_dim,
                    ix_start,
                    ix_start + nx - 1,
                    iy_start,
                    iy_start + ny - 1,
                );
                for (value, chunk) in array[ibase as usize..].iter_mut().zip(dest.chunks_exact(4)) {
                    *value = f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                }
                array_min_max_mean(
                    &array[ibase as usize..],
                    nx,
                    ny,
                    0,
                    nx - 1,
                    0,
                    ny - 1,
                    &mut dmin2,
                    &mut dmax2,
                    &mut dmean2,
                );
                unsafe {
                    iiu_write_section(im_unit_out, array[ibase as usize..].as_mut_ptr().cast())
                };
                dmax = if dmax > dmax2 { dmax } else { dmax2 };
                dmin = if dmin < dmin2 { dmin } else { dmin2 };
                dmean_sum += dmean2;
                iz += 1;
            }
        }
        //
        ind = mode_map[(mode_out + 1) as usize] - 1;
        if ind >= 0 {
            diff_lim = (0.004 * (mode_max[ind as usize] - mode_min[ind as usize]) as f64) as f32;
            if (dmin as f64) < mode_min[ind as usize] as f64 - diff_lim as f64
                || dmax as f64 >= mode_max[ind as usize] as f64 + diff_lim as f64
            {
                mode_try = 2;
                if dmin >= -32900. && dmax < 32900. {
                    mode_try = 1;
                }
                let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                    "%s%.0f%s%.0f%s%d%s%d%s %d\n",
                    &[
                        CArg::Str("WARNING: MTFFILTER - The min or max of the filtered data ("),
                        CArg::Dbl(dmin as f64),
                        CArg::Str(" - "),
                        CArg::Dbl(dmax as f64),
                        CArg::Str(") is outside the range for the output data mode ("),
                        CArg::Int(mode_min[ind as usize].into()),
                        CArg::Str(" - "),
                        CArg::Int(mode_max[ind as usize].into()),
                        CArg::Str("): use -ModeToOutput"),
                        CArg::Int(mode_try.into()),
                    ],
                ));
            }
        }
        dmean = dmean_sum / num_zdo as f32;
        mrc_fill_label_string(title_str.as_bytes(), &mut line);
        let label_end = line.iter().position(|&b| b == 0).unwrap_or(MRC_LABEL_SIZE);
        let label = String::from_utf8_lossy(&line[..label_end]).into_owned();
        if im_unit_out == 1 && num_zdo < nz {
            //
            // if writing a subset back to same file, use the extreme of existing
            // and new data, and just use old mean
            //
            dmin = if dmin < dmin_in { dmin } else { dmin_in };
            dmax = if dmax > dmax_in { dmax } else { dmax_in };
            iiu_write_header_str(1, &label, 1, dmin, dmax, dmean_in);
        } else {
            iiu_write_header_str(im_unit_out, &label, 1, dmin, dmax, dmean);
        }
        if im_unit_out == 3 {
            unsafe { iiu_close(im_unit_out) };
        }
        iter_num += 1;
    }
    unsafe { iiu_close(1) };
    //
    let _ = ImodFile::Stdout.write_all(b" PROGRAM EXECUTED TO END.\n");
    let _ = ImodFile::Stdout.flush();
    std::process::exit(0);
}

/// C `fftFilter3D` (`mtffilter.cpp:855`).
///
/// Applies a filter to a 3D FFT that has been computed with thrdfft.
/// ARRAY is a complex array with dimensions NXDIM, NY, NZ so NXDIM
/// has to be half of (real space size + 2)
pub fn fft_filter_3d(array: &mut [f32], nx_dim: i32, ny: i32, nz: i32, ctf: &[f32], delta: f32) {
    let mut indf: i32;
    let mut xbase: i32;
    let delx: f32;
    let dely: f32;
    let delz: f32;
    let mut xa: f32;
    let mut ya: f32;
    let mut za: f32;
    let mut s: f32;
    //
    delx = (0.5 / (nx_dim as f64 - 1.)) as f32;
    dely = (1. / ny as f64) as f32;
    delz = (1. / nz as f64) as f32;
    for jz in 0..nz {
        za = jz as f32 * delz;
        if za > 0.5 {
            za = (1. - za as f64) as f32;
        }
        for jy in 0..ny {
            ya = jy as f32 * dely;
            if ya > 0.5 {
                ya = (1. - ya as f64) as f32;
            }
            xbase = (jz * ny + jy) * nx_dim * 2;
            for jx in 0..nx_dim {
                xa = jx as f32 * delx;
                s = ((xa * xa + ya * ya + za * za) as f64).sqrt() as f32;
                indf = (s / delta + 0.5) as i32;
                array[(xbase + 2 * jx) as usize] *= ctf[indf as usize];
                array[(xbase + 2 * jx + 1) as usize] *= ctf[indf as usize];
            }
        }
    }
}

/// C `fftFilter1D` (`mtffilter.cpp:890`).
///
/// Applies a filter to a 2-D array in the X direction only
pub fn fft_filter_1d(array: &mut [f32], nx_dim: i32, ny: i32, ctf: &[f32], delta: f32) {
    let mut indf: i32;
    let delx: f32;
    let mut s: f32;
    //
    delx = (0.5 / (nx_dim as f64 - 1.)) as f32;
    for jy in 0..ny {
        for jx in 0..nx_dim {
            s = jx as f32 * delx;
            indf = (s / delta + 0.5) as i32;
            array[(jy * 2 * nx_dim + jx * 2) as usize] *= ctf[indf as usize];
            array[(jy * 2 * nx_dim + jx * 2 + 1) as usize] *= ctf[indf as usize];
        }
    }
}

/// C `amplifier` (`mtffilter.cpp:909`).
///
/// Set up the filter for phase-plate fringe correction
pub fn amplifier(
    cutoff: f32,
    ampfac: f32,
    power: f32,
    ctf: &mut [f32],
    nx: i32,
    ny: i32,
    delta: &mut f32,
    nsize: &mut i32,
) {
    let mut s: f32;
    *nsize = 8192.min(1024.max((2 * nx).max(2 * ny)));
    *delta = (1. / (0.71 * *nsize as f64)) as f32;

    // Functional form: (1 + (amp-1)*exp( -(r/cutoff) ^power)) / a"""
    s = 0.;
    for i in 0..*nsize {
        ctf[i as usize] = ((1. + (ampfac - 1.) as f64 * (-(s / cutoff).powf(power) as f64).exp())
            / ampfac as f64) as f32;
        if ctf[i as usize] < 1.0e-6 {
            ctf[i as usize] = 0.;
        }
        s += *delta;
    }
}

/// C `adjustedFakeIter` (`mtffilter.cpp:929`).
///
/// This matches the iteration adjustment done in Tilt.  The source's local is
/// an `int`, so each branch truncates before the `float` return.
pub fn adjusted_fake_iter(nominal: i32) -> f32 {
    let mut adjusted_fake_iter: i32 = nominal;
    if nominal > 15 {
        adjusted_fake_iter = (15. + 0.8 * (nominal - 15) as f64) as i32;
    }
    if nominal > 30 {
        adjusted_fake_iter = (27. + 0.6 * (nominal - 30) as f64) as i32;
    }
    adjusted_fake_iter as f32
}

/// C `deconvFilter` (`mtffilter.cpp:947`).
///
/// Set up the filter for deconvolution.  Variable names and their units mostly
/// match those in Warp.  highPassNyq as fraction of Nyquist; voltage in V;
/// cs in m, phaseShift in radians, angPixSize in angstroms, delta is interval
/// in 1/pixel between ctf values; nsize is extent from 0 to maximum frequency.
#[allow(clippy::too_many_arguments)]
pub fn deconv_filter(
    deconv_strength: f32,
    snr_falloff: f32,
    high_pass_nyq: f32,
    voltage: f32,
    cs: f32,
    defocus: f32,
    phase_shift: f32,
    ang_pix_size: f32,
    ctfb: &mut [f32],
    delta: f32,
    nsize: i32,
) {
    let pix_size_m: f32;
    let pi: f32 = 3.141593;
    let amplitude: f32 = 0.07;
    let eps: f32 = 1.0e-6;
    let mut frac_nyq: f32;
    let mut high_pass: f32;
    let mut snr: f32;
    let mut lambda: f32;
    let mut lambda2: f32;
    let mut k: f32;
    let mut k2: f32;
    let mut term1: f32;
    let mut w: f32;
    let mut ctf: f32;

    pix_size_m = (ang_pix_size as f64 * 1.0e-10) as f32;
    high_pass = 2.;
    ctfb[0] = 1.;
    for ind in 1..nsize {
        frac_nyq = (2. * ind as f64 * delta as f64) as f32;
        if high_pass_nyq > 0. {
            high_pass =
                (1. - ((1.0f64).min((frac_nyq / high_pass_nyq) as f64) * pi as f64).cos()) as f32;
        }
        snr = (((-frac_nyq * snr_falloff) as f64 * 100. / ang_pix_size as f64).exp()
            * (10.0f64).powf(3. * deconv_strength as f64)
            * high_pass as f64
            + eps as f64) as f32;

        lambda = (12.2643247 / (voltage as f64 * (1.0 + voltage as f64 * 0.978466e-6)).sqrt()
            * 1e-10) as f32;
        lambda2 = lambda * 2.;

        k = (frac_nyq as f64 / (2. * pix_size_m as f64)) as f32;
        k2 = k * k;
        term1 = lambda * lambda * lambda * cs * k2 * k2;
        w = ((pi as f64 / 2.) * (term1 + lambda2 * defocus * k2) as f64 - phase_shift as f64)
            as f32;
        ctf = ((w as f64).cos() * amplitude as f64
            - (1. - (amplitude * amplitude) as f64).sqrt() * (w as f64).sin()) as f32;
        ctfb[ind as usize] = (if ctf >= 0. { ctf } else { -ctf } as f64
            / ((ctf * ctf) as f64 + 1. / snr as f64)) as f32;
    }
}

/// C `convertFreqUnit` (`mtffilter.cpp:983`).
///
/// Convert a frequency from nm, A, 1/nm, or 1/A to 1/pixel
pub fn convert_freq_unit(inv_freq_unit: i32, pixel_size: f32, freq_val: &mut f32) {
    if inv_freq_unit == 0 {
        return;
    }
    if inv_freq_unit > 0 && *freq_val > 0. {
        *freq_val = (1. / *freq_val as f64) as f32;
    }
    *freq_val *= pixel_size;
    if (inv_freq_unit.abs() % 2) == 0 {
        *freq_val = (*freq_val as f64 * 10.) as f32;
    }
}
