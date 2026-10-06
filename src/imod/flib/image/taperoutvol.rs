//! Translation of `IMOD/flib/image/taperoutvol.f90`.
//!
//! TAPEROUTVOL will cut a subset out of an image volume, pad it into a larger
//! volume, and taper the intensity down to the mean value of the volume over
//! the extent of the padding region, i.e., from the edge of the actual
//! excised pixels to the edge of the new volume.  None of the original
//! excised pixels are attenuated by this method.
//!
//! The main program maps to [`taperoutvol`]; the subvolume parameters come
//! from the shared `taperPrep` (`taperprep.rs`).  `taperoutpad` and
//! `slicenoisetaperpad` are the Fortran wrappers in `taperpad.c`, called in
//! place (`array` is both input and output), and `iclden` is the Fortran
//! wrapper of `arrayMinMaxMean`.

use crate::imod::flib::image::taperprep::taper_prep;
use crate::imod::flib::subrs::compat::datetime::time;
use crate::imod::flib::subrs::compat::gfortran_rt::{maxss, minss};
use crate::imod::flib::subrs::hvem::b3ddate::b3d_date;
use crate::imod::flib::subrs::hvem::parse_input_params::{
    exit_error, pip_get_in_out_file, pip_get_logical, pip_read_or_parse_options,
};
use crate::imod::flib::subrs::imsubs::irdhdr::irdhdr;
use crate::imod::flib::subrs::imsubs::wrap_iiunit::{imopen, irdpas};
use crate::imod::libcfshr::b3dutil::exit;
use crate::imod::libcfshr::simplestat::array_min_max_mean_fortran;
use crate::imod::libcfshr::taperpad::{PadIn, slice_noise_taper_pad, taperoutpad};
use crate::imod::libiimod::mrcfiles::{MRC_LABEL_SIZE, MRC_NLABELS};
use crate::imod::libiimod::mrcslice::SLICE_MODE_FLOAT;
use crate::imod::libiimod::unit_fileio::{iiu_close, iiu_set_position, iiu_write_section};
use crate::imod::libiimod::unit_header::{
    iiu_alt_cell, iiu_alt_origin, iiu_alt_tilt, iiu_alt_tilt_orig, iiu_create_header,
    iiu_ret_tilt, iiu_ret_tilt_orig, iiu_trans_labels, iiu_write_header,
};

/// `parameter (numOptions = 10)` (`taperoutvol.f90:44`).
const TAPEROUTVOL_NUM_OPTIONS: i32 = 10;
/// Fallback PIP table, the `options(1)` string (`taperoutvol.f90:46-49`).
const TAPEROUTVOL_OPTIONS: &str = "input:InputFile:FN:@output:OutputFile:FN:@xminmax:XMinAndMax:IP:@\
yminmax:YMinAndMax:IP:@zminmax:ZMinAndMax:IP:@taper:TaperPadsInXYZ:IT:@\
nofft:NoFFTSizes:B:@noise:NoisePadding:B:@param:ParameterFile:PF:@help:usage:B:";

/// Original program `taperoutvol` (`taperoutvol.f90:15`).
pub fn taperoutvol() {
    let mut nxyz = [0_i32; 3];
    let mut mxyz = [0_i32; 3];
    let mut nxyz_out = [0_i32; 3];
    let mut mxyz_out = [0_i32; 3];
    let mut cell_out = [0.0_f32; 6];
    let mut input_file = String::new();
    let mut output_file = String::new();
    let mut date_str = [b' '; 9];
    let mut time_str = [b' '; 8];
    let (mut ix_low, mut iy_low, mut iz_low) = (0_i32, 0_i32, 0_i32);
    let (mut ix_high, mut iy_high, mut iz_high) = (0_i32, 0_i32, 0_i32);
    let (mut nx_box, mut ny_box, mut nz_box) = (0_i32, 0_i32, 0_i32);
    let (mut nx_out, mut ny_out, mut nz_out) = (0_i32, 0_i32, 0_i32);
    let mut mode = 0_i32;
    let (mut dmin2, mut dmax2, mut dmean2) = (0.0_f32, 0.0_f32, 0.0_f32);
    let (mut tmp_min, mut tmp_max, mut tmp_mean) = (0.0_f32, 0.0_f32, 0.0_f32);
    let (mut origin_x, mut origin_y, mut origin_z) = (0.0_f32, 0.0_f32, 0.0_f32);
    let (mut num_opt_arg, mut num_non_opt_arg) = (0_i32, 0_i32);
    //
    let mut no_fft = false;
    let mut noise_pad = false;
    //
    // Pip startup: set error, parse options, check help, set flag if used
    //
    pip_read_or_parse_options(
        &[TAPEROUTVOL_OPTIONS],
        TAPEROUTVOL_NUM_OPTIONS,
        "taperoutvol",
        "ERROR: TAPEROUTVOL - ",
        true,
        2,
        1,
        1,
        &mut num_opt_arg,
        &mut num_non_opt_arg,
    );
    let pip_input = num_opt_arg + num_non_opt_arg > 0;

    if pip_get_in_out_file(
        "InputFile",
        1,
        "Name of image input file",
        &mut input_file,
        320,
    ) != 0
    {
        exit_error("No input file specified");
    }
    //
    imopen(1, &input_file, "RO");
    unsafe {
        irdhdr(
            1,
            nxyz.as_mut_ptr(),
            mxyz.as_mut_ptr(),
            &mut mode,
            &mut dmin2,
            &mut dmax2,
            &mut dmean2,
        );
    }
    //
    if pip_get_in_out_file("OutputFile", 2, "Name of output file", &mut output_file, 320) != 0 {
        exit_error("No output file specified");
    }
    //
    if pip_input {
        let _ = pip_get_logical("NoFFTSizes", &mut no_fft);
    }
    if pip_input {
        let _ = pip_get_logical("NoisePadding", &mut noise_pad);
    }
    let nxyz_copy = nxyz;
    taper_prep(
        pip_input,
        no_fft,
        &nxyz_copy,
        &mut ix_low,
        &mut ix_high,
        &mut iy_low,
        &mut iy_high,
        &mut iz_low,
        &mut iz_high,
        &mut nx_box,
        &mut ny_box,
        &mut nz_box,
        &mut nx_out,
        &mut ny_out,
        &mut nz_out,
        &mut nxyz_out,
        &mut mxyz_out,
        &mut cell_out,
        &mut origin_x,
        &mut origin_y,
        &mut origin_z,
    );
    //
    let mut iz: i32 = 1;
    if noise_pad {
        iz = 2 * nx_box.max(ny_box) + (nx_out - nx_box) + (ny_out - ny_box);
    }
    let mut array: Vec<f32> = Vec::new();
    let mut pad_work: Vec<f32> = Vec::new();
    // A negative extent allocates a zero-size array in Fortran.
    if array.try_reserve_exact((nx_out * ny_out).max(0) as usize).is_err()
        || pad_work.try_reserve_exact(iz.max(0) as usize).is_err()
    {
        exit_error("Allocating array for image plane");
    }
    array.resize((nx_out * ny_out).max(0) as usize, 0.0);
    pad_work.resize(iz.max(0) as usize, 0.0);
    //
    imopen(2, &output_file, "NEW");
    time(&mut time_str);
    b3d_date(&mut date_str);
    // `301 format('TAPEROUTVOL: Taper outside of excised volume',t57,a9,2x,a8)`
    let mut title_ch = [b' '; 80];
    let head = b"TAPEROUTVOL: Taper outside of excised volume";
    title_ch[..head.len()].copy_from_slice(head);
    title_ch[56..65].copy_from_slice(&date_str);
    title_ch[67..75].copy_from_slice(&time_str);
    let title: [u8; MRC_LABEL_SIZE] = title_ch;
    let mut labels = [[0_u8; MRC_LABEL_SIZE]; MRC_NLABELS];
    labels[0] = title;
    iiu_create_header(2, &nxyz_out, &mxyz_out, mode, &labels, 0);
    iiu_trans_labels(2, 1);
    iiu_alt_cell(2, &cell_out);
    iiu_alt_origin(2, &[origin_x, origin_y, origin_z]);
    let tilt = iiu_ret_tilt(1);
    let orig_tilt = iiu_ret_tilt_orig(1);
    iiu_alt_tilt(2, &tilt);
    iiu_alt_tilt_orig(2, &orig_tilt);
    let mut dmin = 1.0e30_f32;
    let mut dmax = -1.0e30_f32;
    let mut sum_mean = 0.0_f32;
    //
    let iz_start = iz_low - (nz_out - nz_box) / 2;
    let iz_end = iz_start + nz_out - 1;
    for iz in iz_start..=iz_end {
        let iz_read = iz_low.max(iz_high.min(iz));
        unsafe {
            iiu_set_position(1, iz_read, 0);
            if irdpas(
                1, &mut array, nx_box, ny_box, ix_low, ix_high, iy_low, iy_high,
            )
            .is_err()
            {
                // 99 call exitError('Reading file')
                exit_error("Reading file");
            }
        }
        if noise_pad {
            slice_noise_taper_pad(
                PadIn::InPlace,
                SLICE_MODE_FLOAT,
                nx_box,
                ny_box,
                &mut array,
                nx_out,
                nx_out,
                ny_out,
                20_i32.max(120_i32.min(nx_box.max(ny_box) / 50)),
                4,
                &mut pad_work,
            );
        } else {
            taperoutpad(
                PadIn::InPlace,
                &nx_box,
                &ny_box,
                &mut array,
                &nx_out,
                &nx_out,
                &ny_out,
                &1,
                &dmean2,
            );
        }
        if iz < iz_low || iz > iz_high {
            let atten: f32 = if iz < iz_low {
                (iz - iz_start) as f32 / (iz_low - iz_start) as f32
            } else {
                (iz_end - iz) as f32 / (iz_end - iz_high) as f32
            };
            let base = (1. - atten) * dmean2;
            for value in array[..(nx_out * ny_out) as usize].iter_mut() {
                *value = base + atten * *value;
            }
        }
        array_min_max_mean_fortran(
            &array,
            &nx_out,
            &ny_out,
            &1,
            &nx_out,
            &1,
            &ny_out,
            &mut tmp_min,
            &mut tmp_max,
            &mut tmp_mean,
        );
        // `minss dmin, tmpMin` / `maxss dmax, tmpMax` (reference object).
        dmin = minss(dmin, tmp_min);
        dmax = maxss(dmax, tmp_max);
        sum_mean += tmp_mean;
        unsafe {
            iiu_write_section(2, array.as_mut_ptr().cast());
        }
    }
    let dmean = sum_mean / nz_out as f32;
    iiu_write_header(2, &title, 1, dmin, dmax, dmean);
    unsafe {
        iiu_close(2);
    }
    exit(0);
}
