//! Translation of `IMOD/flib/distort/xf2rotmagstr.f90`.
//!
//! Library calls go to the C entry points with the Fortran wrappers' `iz - 1`
//! and `rows = 2` inlined (`warpwrapfort.c:128-189`), and
//! `amat_to_rotmagstr`'s wrapper order `amat[0], amat[2], amat[1], amat[3]`
//! (`amat_to_rotmagstr.c:172`).  Transforms are `[f32; 6]` in Fortran
//! `(2,3)` storage order.

use crate::imod::flib::subrs::compat::gfortran_rt::format_f;
use crate::imod::flib::subrs::hvem::dopen::dopen;
use crate::imod::flib::subrs::hvem::getinout::getinout;
use crate::imod::flib::subrs::hvem::parse_input_params::{exit_error, set_exit_prefix};
use crate::imod::flib::subrs::xfsubs::xfrdall::xfrdall2;
use crate::imod::libcfshr::amat_to_rotmagstr::amat_to_rotmagstr;
use crate::imod::libcfshr::b3dutil::exit;
use crate::imod::libwarp::warpfiles::{
    get_linear_transform, get_num_warp_points, get_warp_grid_size, separate_linear_transform,
};
use crate::imod::libwarp::warputils::read_check_warp_file;
use std::io::BufReader;

/// `parameter (maxSect = 10000)` (`xf2rotmagstr.f90:8`).
const MAX_SECT: i32 = 10000;

/// Rust-only: one transform's rotation, magnification, stretch and stretch
/// axis as `amat_to_rotmagstr` returns them, and the mean magnification the
/// program derives (`xf2rotmagstr.f90:52-54`).
#[derive(Clone, Copy, Debug, Default)]
pub struct RotMagStr {
    pub theta: f32,
    pub smag: f32,
    pub str_: f32,
    pub phi: f32,
    pub smag_mean: f32,
}

/// Rust-only: what `xf2rotmagstr` computes, as the program's direct-call
/// result (`CLAUDE.md`, "Wherever we control both sides, use a direct
/// function call now"): one [`RotMagStr`] per transform, and whether the file
/// was a warping file (the program then first prints " Reading warping
/// file ...").
#[derive(Clone, Debug, Default)]
pub struct Xf2rotmagstrResult {
    pub from_warp_file: bool,
    pub transforms: Vec<RotMagStr>,
}

/// Original program `xf2rotmagstr` (`xf2rotmagstr.f90:4`): the file name,
/// then [`xf2rotmagstr_compute`], then one line per transform.
///
/// A simple program to convert transforms to rotation-mag-stretch.
pub fn xf2rotmagstr() {
    //
    set_exit_prefix("ERROR: XF2ROTMAGSTR -");
    // `character*320 strFile`
    let str_file: String = match getinout(1) {
        Ok((input, _)) => input,
        Err(_) => exit(2),
    };
    let result = xf2rotmagstr_compute(str_file.trim_end_matches(' '));
    for (i, transform) in result.transforms.iter().enumerate() {
        println!("{}", xf2rotmagstr_line(i as i32 + 1, transform));
    }
    exit(0);
}

/// Rust-only: the program's line for transform `i` (1-based)
/// (`xf2rotmagstr.f90:55-56`), without its ending.
pub fn xf2rotmagstr_line(i: i32, transform: &RotMagStr) -> String {
    let fmt_i5 = format!("{i}");
    let fmt_i5 = if fmt_i5.len() > 5 {
        "*****".to_owned()
    } else {
        format!("{fmt_i5:>5}")
    };
    format!(
        "{}: rot={}, mag={}, str={} on{} axis, Mean mag={}",
        fmt_i5,
        format_f(transform.theta as f64, 8, 2),
        format_f(transform.smag as f64, 7, 4),
        format_f(transform.str_ as f64, 7, 4),
        format_f(transform.phi as f64, 7, 1),
        format_f(transform.smag_mean as f64, 7, 4)
    )
}

/// The body of program `xf2rotmagstr` (`xf2rotmagstr.f90:17-53`) after the
/// file name is read: reads the transforms (or the linear parts of a warping
/// file) and converts each.  Errors end through `exitError`.
pub fn xf2rotmagstr_compute(str_file: &str) -> Xf2rotmagstrResult {
    // `character*320 errString`
    let mut err_string = String::new();
    let mut xform = vec![[0.0_f32; 6]; MAX_SECT as usize];
    let mut pixel_size = 0.0_f32;
    let mut nxf_read = 0_i32;
    let mut ierr: i32;
    let (mut nx, mut ny, mut ibin, mut iflags) = (0_i32, 0_i32, 0_i32, 0_i32);
    let ind_warp_file: i32;
    let mut num_control: i32;
    let mut smag_mean: f32;
    let mut result = Xf2rotmagstrResult::default();
    ind_warp_file = read_check_warp_file(
        str_file,
        0,
        1,
        &mut nx,
        &mut ny,
        &mut nxf_read,
        &mut ibin,
        &mut pixel_size,
        &mut iflags,
        &mut err_string,
    );
    if ind_warp_file < -1 {
        exit_error(&err_string);
    }
    if ind_warp_file >= 0 {
        println!(" Reading warping file and extracting linear transformations");
        result.from_warp_file = true;
        ierr = 1;
        if nxf_read <= MAX_SECT {
            for i in 1..=nxf_read {
                num_control = 0;
                if (iflags / 2) % 2 > 0 {
                    ierr = get_num_warp_points(i - 1, &mut num_control);
                } else {
                    ierr = get_warp_grid_size(i - 1, &mut nx, &mut ny, &mut num_control);
                }
                if num_control > 2 && separate_linear_transform(i - 1) != 0 {
                    exit_error("TRYING TO EXTRACT LINEAR TRANSFORM FROM WARPING TRANSFORM");
                }
                ierr = get_linear_transform(i - 1, &mut xform[i as usize - 1], 2);
            }
            ierr = 0;
        }
    } else {
        let unit1 = dopen(1, str_file, "ro", "f");
        let mut list: Vec<[f32; 6]> = Vec::new();
        ierr = xfrdall2(&mut BufReader::new(unit1), &mut list, MAX_SECT);
        // `xfrdall2` stops storing at the limit; the count it reports is one
        // past it only on that error, which exits below.
        nxf_read = list.len() as i32;
        xform[..list.len()].copy_from_slice(&list);
        if ierr == 2 {
            exit_error("READING TRANSFORM FILE");
        }
    }
    if ierr == 1 {
        exit_error("TOO MANY TRANSFORMS IN FILE FOR ARRAY");
    }

    for i in 1..=nxf_read {
        let f = &xform[i as usize - 1];
        let (theta, smag, str_, phi) = amat_to_rotmagstr(f[0], f[2], f[1], f[3]);
        smag_mean = smag * str_.abs().sqrt();
        // The line is written by the caller (`xf2rotmagstr_line`).
        result.transforms.push(RotMagStr {
            theta,
            smag,
            str_,
            phi,
            smag_mean,
        });
    }
    result
}
