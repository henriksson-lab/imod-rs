//! Translation of `IMOD/flib/image/xcorrstack.f`.
//!
//! The whole unit is the main program, [`xcorrstack`].  The source's
//! `common /bigarr/ array, brray` are two `complex(idim/2)` arrays, held here
//! as `f32` vectors of `idim` elements (real and imaginary parts
//! interleaved, as the Fortran storage is).  Library calls go to the Fortran
//! wrappers' C bodies with their argument conventions inlined
//! (`setctfwsr` -> `XCorrSetCTF`, `filterpart` -> `XCorrFilterPart` in place,
//! `isetdn` -> `scaleArrayForMode` with 0-based limits).  The in-place
//! `taperoutpad` and `irepak` calls pass one buffer twice in the source;
//! `taperoutpad` has its `InPlace` form, and `irepak` (a forward copy that
//! never overtakes its reads) is given a copy of the source rows.

use crate::imod::flib::subrs::hvem::b3ddate::b3d_date;
use crate::imod::flib::subrs::hvem::parse_input_params::{
    exit_error, pip_get_in_out_file, pip_read_or_parse_options,
};
use crate::imod::flib::subrs::imsubs::irdhdr::irdhdr;
use crate::imod::flib::subrs::imsubs::wrap_iiunit::{imopen, irdsec};
use crate::imod::libcfshr::b3dutil::{exit, set_float_output_for_entered_mode};
use crate::imod::libcfshr::filtxcorr::{FilterIn, nice_frame, xcorr_filter_part, xcorr_set_ctf};
use crate::imod::libcfshr::parse_params::{
    pip_done, pip_get_boolean, pip_get_float, pip_get_integer, pip_get_two_integers,
};
use crate::imod::libcfshr::reduce_by_binning::irepak;
use crate::imod::libcfshr::simplestat::{array_min_max_mean_fortran, scale_array_for_mode};
use crate::imod::libcfshr::taperpad::{PadIn, splitfill, taperoutpad};
use crate::imod::libfft::todfft;
use crate::imod::libiimod::mrcfiles::MRC_LABEL_SIZE;
use crate::imod::libiimod::unit_fileio::{iiu_close, iiu_set_position, iiu_write_section};
use crate::imod::libiimod::unit_header::{
    iiu_alt_cell, iiu_alt_mode, iiu_alt_origin, iiu_alt_sample, iiu_alt_size, iiu_ret_cell,
    iiu_ret_delta, iiu_ret_origin, iiu_trans_header, iiu_write_header_str,
};
use chrono::{Local, Timelike};

/// `parameter (idim=6000*6000)` (`xcorrstack.f:12`).
const IDIM: i32 = 6000 * 6000;
/// `parameter (numOptions = 12)` (`xcorrstack.f:41`).
const NUM_OPTIONS: i32 = 12;
/// Fallback PIP table `options(1)` (`xcorrstack.f:43-49`).
const OPTIONS: &str = "stack:StackInputFile:FN:@single:SingleInputFile:FN:@\
output:OutputFile:FN:@sections:StartingEndingSections:IP:@\
radius1:FilterRadius1:F:@radius2:FilterRadius2:F:@\
sigma1:FilterSigma1:F:@sigma2:FilterSigma2:F:@\
split:SplitIntoCorners:B:@fill:FillValue:F:@\
param:ParameterFile:PF:@help:usage:B:";

/// Original program `xcorrstack` (`xcorrstack.f:1`).
///
/// XCORRSTACK cross-correlates each of the sections in one image stack with
/// the first section in a second image file.  A subset of sections may be
/// done.
pub fn xcorrstack() {
    // `COMMON //NX,NY,NZ,nxs,nys,nzs` with `EQUIVALENCE (NX,NXYZ),(nxs,nxyzs)`
    let mut nxyz = [0_i32; 3];
    let mut mxyz = [0_i32; 3];
    let mut nxyzs = [0_i32; 3];
    let mut mxyzs = [0_i32; 3];
    let mut ctfb = [0.0_f32; 8193];
    // `complex array(idim/2), brray(idim/2)`
    let mut array: Vec<f32> = vec![0.0; IDIM as usize];
    let mut brray: Vec<f32> = vec![0.0; IDIM as usize];
    //
    let mut filbig = String::new();
    let mut filsmall = String::new();
    let mut filout = String::new();
    let mut dat = [b' '; 9];
    let mut mode = 0_i32;
    let nxpad: i32;
    let nypad: i32;
    let nxdim: i32;
    let mut izst: i32;
    let mut iznd: i32;
    let mut ierr: i32;
    let iffill: i32;
    let mut modeout: i32;
    let mut if_split: i32;
    let mut sigma1: f32;
    let mut sigma2: f32;
    let mut radius1: f32;
    let mut radius2: f32;
    let (mut dmin, mut dmax, mut dmean) = (0.0_f32, 0.0_f32, 0.0_f32);
    // Uninitialised in the source unless `-fill` is entered (then unused).
    let mut fill = 0.0_f32;
    let mut deltab = 0.0_f32;
    let delta: [f32; 3];
    let mut cell: [f32; 6];
    let mut dmsum: f32;
    let (mut dmax2, mut dmean2, mut dmin2) = (0.0_f32, 0.0_f32, 0.0_f32);
    let mut ixlo: i32;
    let mut iylo: i32;
    let mut modesm = 0_i32;
    let nxyzst = [0_i32; 3];
    let (mut dmins, mut dmaxs, mut dmeans) = (0.0_f32, 0.0_f32, 0.0_f32);
    //
    let (mut num_opt_arg, mut num_non_opt_arg) = (0_i32, 0_i32);

    sigma1 = 0.;
    sigma2 = 0.;
    radius1 = 0.;
    radius2 = 0.;
    if_split = 0;
    //
    // PIP startup
    pip_read_or_parse_options(
        &[OPTIONS],
        NUM_OPTIONS,
        "xcorrstack",
        "ERROR: XCORRSTACK - ",
        false,
        3,
        2,
        1,
        &mut num_opt_arg,
        &mut num_non_opt_arg,
    );
    //
    // Get input/output files
    if pip_get_in_out_file("StackInputFile", 1, " ", &mut filbig, 160) != 0 {
        exit_error("NO INPUT IMAGE STACK SPECIFIED");
    }
    if pip_get_in_out_file("SingleInputFile", 2, " ", &mut filsmall, 160) != 0 {
        exit_error("NO SINGLE IMAGE FILE SPECIFIED");
    }
    if pip_get_in_out_file("OutputFile", 3, " ", &mut filout, 160) != 0 {
        exit_error("NO OUTPUT FILE SPECIFIED");
    }
    //
    // Open image files.
    //
    imopen(1, &filbig, "RO");
    unsafe {
        irdhdr(
            1,
            nxyz.as_mut_ptr(),
            mxyz.as_mut_ptr(),
            &mut mode,
            &mut dmin,
            &mut dmax,
            &mut dmean,
        );
    }
    let (nx, ny, nz) = (nxyz[0], nxyz[1], nxyz[2]);
    nxpad = nice_frame(nx, 2, 19);
    nypad = nice_frame(ny, 2, 19);
    nxdim = nxpad + 2;
    //
    if nxdim.wrapping_mul(nypad) > IDIM {
        exit_error("IMAGES TOO LARGE FOR ARRAYS");
    }
    //
    imopen(2, &filsmall, "RO");
    unsafe {
        irdhdr(
            2,
            nxyzs.as_mut_ptr(),
            mxyzs.as_mut_ptr(),
            &mut modesm,
            &mut dmins,
            &mut dmaxs,
            &mut dmeans,
        );
    }
    //
    if nxyzs[0] > nx || nxyzs[1] > ny {
        exit_error("SINGLE IMAGE FILE MUST NOT HAVE X OR Y SIZE BIGGER THAN STACK");
    }
    //
    imopen(3, &filout, "NEW");
    //
    iiu_trans_header(3, 1);
    //
    izst = 0;
    iznd = nz - 1;
    ierr = pip_get_two_integers(b"StartingEndingSections", &mut izst, &mut iznd);
    if izst < 0 || iznd > nz - 1 || izst > iznd {
        exit_error("STARTING AND ENDING SECTIONS ARE OUT OF RANGE OR OUT OF ORDER");
    }
    ierr = pip_get_float(b"FilterRadius1", &mut radius1);
    ierr = pip_get_float(b"FilterRadius2", &mut radius2);
    ierr = pip_get_float(b"FilterSigma1", &mut sigma1);
    ierr = pip_get_float(b"FilterSigma2", &mut sigma2);
    iffill = 1 - pip_get_float(b"FillValue", &mut fill);
    ierr = pip_get_boolean(b"SplitIntoCorners", &mut if_split);
    modeout = mode;
    ierr = pip_get_integer(b"ModeOfOutput", &mut modeout);
    if modeout < 0 || (modeout / 3 != 0 && modeout != 6 && modeout != 12) {
        exit_error("IMPROPER MODE VALUE");
    }
    if ierr == 0 {
        set_float_output_for_entered_mode(modeout);
    }
    iiu_alt_mode(3, modeout);
    pip_done();

    xcorr_set_ctf(
        sigma1,
        sigma2,
        radius1,
        radius2,
        &mut ctfb,
        nxpad,
        nypad,
        &mut deltab,
    );
    //
    // if single image is same size or if not splitting, use taperoutpad
    // otherwise, split it up in brray
    //
    if unsafe { irdsec(2, &mut array) }.is_err() {
        exit_error("READING IMAGE FILE");
    }
    let (nxs, nys) = (nxyzs[0], nxyzs[1]);
    if (nxs == nx && nys == ny) && if_split == 0 {
        taperoutpad(
            PadIn::Float(&array),
            &nxs,
            &nys,
            &mut brray,
            &nxdim,
            &nxpad,
            &nypad,
            &iffill,
            &fill,
        );
    } else {
        splitfill(
            &array, &nxs, &nys, &mut brray, &nxdim, &nxpad, &nypad, &iffill, &fill,
        );
    }
    //
    // get fft of single image, apply filter if desired
    //
    todfft(&mut brray, nxpad, nypad, 0);
    if deltab != 0. {
        xcorr_filter_part(FilterIn::InPlace, &mut brray, nxpad, nypad, &ctfb, deltab);
    }
    //
    // manage header pixel size and origin: REUSING NXYZS
    nxyzs[0] = nx;
    nxyzs[1] = ny;
    nxyzs[2] = iznd + 1 - izst;
    let nzs = nxyzs[2];
    delta = iiu_ret_delta(1);
    let [origx, origy, mut origz] = iiu_ret_origin(1);
    cell = iiu_ret_cell(1);
    origz = origz - delta[2] * izst as f32;
    cell[2] = nzs as f32 * cell[2] / mxyz[2] as f32;
    iiu_alt_size(3, &nxyzs, &nxyzst);
    iiu_alt_sample(3, &nxyzs);
    iiu_alt_cell(3, &cell);
    iiu_alt_origin(3, &[origx, origy, origz]);
    //
    // Loop over the images
    dmsum = 0.;
    dmax = -1.0e10;
    dmin = 1.0e10;
    for kk in izst..=iznd {
        //
        println!(" Working on section{:>12}", kk);
        unsafe {
            iiu_set_position(1, kk, 0);
        }
        if unsafe { irdsec(1, &mut array) }.is_err() {
            exit_error("READING IMAGE FILE");
        }
        taperoutpad(
            PadIn::InPlace,
            &nx,
            &ny,
            &mut array,
            &nxdim,
            &nxpad,
            &nypad,
            &iffill,
            &fill,
        );
        //
        // print *,'taking fft'
        todfft(&mut array, nxpad, nypad, 0);
        //
        // multiply array by complex conjugate of brray, put back in array
        //
        // print *,'multiplying arrays'
        for jx in 1..=(nypad * nxdim / 2) as usize {
            // `array(jx)=array(jx)*conjg(brray(jx))`: a complex*8 product,
            // (a + bi)(c - di) = (ac - b(-d)) + (a(-d) + bc)i.
            let (a, b) = (array[2 * jx - 2], array[2 * jx - 1]);
            let (c, d) = (brray[2 * jx - 2], -brray[2 * jx - 1]);
            array[2 * jx - 2] = a * c - b * d;
            array[2 * jx - 1] = a * d + b * c;
        }
        //
        // print *,'taking back fft'
        todfft(&mut array, nxpad, nypad, 1);

        // print *,'repack, set density, write'
        // TODO: check that extra is on right
        ixlo = (nxpad - nx) / 2;
        iylo = (nypad - ny) / 2;
        {
            let count = (nxdim * nypad) as usize;
            let source: Vec<u8> = array[..count]
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect();
            // SAFETY: the `f32` buffer viewed as its bytes, as the source's
            // `irepak(array, array, ...)` passes the same storage.
            let destination = unsafe {
                core::slice::from_raw_parts_mut(array.as_mut_ptr().cast::<u8>(), array.len() * 4)
            };
            irepak(
                destination,
                &source,
                &nxdim,
                &nypad,
                &ixlo,
                &(ixlo + nx - 1),
                &iylo,
                &(iylo + ny - 1),
            );
        }
        if modeout == 2 {
            array_min_max_mean_fortran(
                &array,
                &nx,
                &ny,
                &1,
                &nx,
                &1,
                &ny,
                &mut dmin2,
                &mut dmax2,
                &mut dmean2,
            );
        } else {
            // `isetdn(array, NX, NY, MODEout, 1, NX, 1, NY, ...)`
            scale_array_for_mode(
                &mut array,
                nx,
                modeout,
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
            iiu_write_section(3, array.as_mut_ptr().cast());
        }
        //
        // `xcorrstack.f:185-186`: `maxss dmax, dmax2` / `minss dmin, dmin2` in
        // the reference object.
        dmax = if dmax > dmax2 { dmax } else { dmax2 };
        dmin = if dmin < dmin2 { dmin } else { dmin2 };
        dmsum = dmsum + dmean2;
    }
    //
    dmean = dmsum / nzs as f32;
    //
    b3d_date(&mut dat);
    // `CALL TIME(TIM)`
    let local = Local::now();
    let tim = format!(
        "{:02}:{:02}:{:02}",
        local.hour(),
        local.minute(),
        local.second()
    );
    //
    // `FORMAT('XCORR: Stack correlated with single image',t57,A9,2X,A8)`
    let mut titlech = [b' '; MRC_LABEL_SIZE];
    let head = b"XCORR: Stack correlated with single image";
    titlech[..head.len()].copy_from_slice(head);
    titlech[56..65].copy_from_slice(&dat);
    titlech[67..75].copy_from_slice(tim.as_bytes());
    iiu_write_header_str(
        3,
        std::str::from_utf8(&titlech).unwrap_or_default(),
        1,
        dmin,
        dmax,
        dmean,
    );
    unsafe {
        iiu_close(3);
        iiu_close(2);
        iiu_close(1);
    }
    //
    exit(0);
}
