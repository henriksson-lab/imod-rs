//! Translation of `IMOD/flib/image/taperprep.f90`.
//!
//! TAPERPREP does common work of getting subvolume parameters and computing
//! the real image box size for taperoutvol and combinefft.

use crate::imod::flib::subrs::hvem::frefor::{ListItem, ListReadError, list_read};
use crate::imod::flib::subrs::hvem::parse_input_params::exit_error;
use crate::imod::libcfshr::b3dutil::exit;
use crate::imod::libcfshr::filtxcorr::nice_frame;
use crate::imod::libcfshr::parse_params::{pip_get_three_integers, pip_get_two_integers};
use crate::imod::libfft::nice_fft_limit;
use crate::imod::libiimod::unit_header::{iiu_ret_delta, iiu_ret_origin};
use std::io::Write;

/// Original `taperPrep` (`taperprep.f90:17`).
///
/// PIPINPUT indicates whether PIP input is being used; `no_fft` is a flag
/// that sizes are not to be increased for FFTs; NXYZ(3) has the input file
/// sizes; IXLOW, IXHIGH, IYLOW, IYHIGH, IZLOW, IZHIGH are returned with the
/// index coordinates of the subvolume in the file; NXBOX, NYBOX, NZBOX are
/// returned with the size of the box being read in; NXOUT, NYOUT, NZOUT are
/// returned with the size of the padded volume; NXYZOUT, MXYZOUT are filled
/// with the values of nxOUT, nyOUT, nzOUT; CELLOUT is returned with the
/// correct cell values; ORIGINX, ORIGINY, ORIGINZ are returned with the
/// subvolume origin values.
#[allow(clippy::too_many_arguments)]
pub fn taper_prep(
    pip_input: bool,
    no_fft: bool,
    nxyz: &[i32; 3],
    ix_low: &mut i32,
    ix_high: &mut i32,
    iy_low: &mut i32,
    iy_high: &mut i32,
    iz_low: &mut i32,
    iz_high: &mut i32,
    nx_box: &mut i32,
    ny_box: &mut i32,
    nz_box: &mut i32,
    nx_out: &mut i32,
    ny_out: &mut i32,
    nz_out: &mut i32,
    nxyz_out: &mut [i32; 3],
    mxyz_out: &mut [i32; 3],
    cell_out: &mut [f32; 6],
    origin_x: &mut f32,
    origin_y: &mut f32,
    origin_z: &mut f32,
) {
    *ix_low = 0;
    *iy_low = 0;
    *iz_low = 0;
    *ix_high = nxyz[0] - 1;
    *iy_high = nxyz[1] - 1;
    *iz_high = nxyz[2] - 1;
    let mut num_pad_x = 0_i32;
    let mut mum_pad_y = 0_i32;
    let mut num_pad_z = 0_i32;
    if pip_input {
        let _ = pip_get_two_integers(b"XMinAndMax", ix_low, ix_high);
        let _ = pip_get_two_integers(b"YMinAndMax", iy_low, iy_high);
        let _ = pip_get_two_integers(b"ZMinAndMax", iz_low, iz_high);
        let _ = pip_get_three_integers(
            b"TaperPadsInXYZ",
            &mut num_pad_x,
            &mut mum_pad_y,
            &mut num_pad_z,
        );
    } else {
        // `read(5,*)` with no `END=`/`ERR=`: a failed read is the gfortran
        // runtime error, status 2.
        let read_abort = |err: ListReadError| -> ! {
            let _ = std::io::stdout().flush();
            match err {
                ListReadError::End => eprintln!("Fortran runtime error: End of file"),
                ListReadError::Error => eprintln!("Fortran runtime error: Bad value during read"),
            }
            exit(2);
        };
        // `write(*,'(1x,a,/,a,$)')`
        print!(
            " Starting and ending X, then Y, then Z index coordinates to extract\n (/ for whole volume): "
        );
        let _ = std::io::stdout().flush();
        let stdin = std::io::stdin();
        let mut stdin = stdin.lock();
        if let Err(err) = list_read(
            &mut stdin,
            &mut [
                ListItem::Integer(ix_low),
                ListItem::Integer(ix_high),
                ListItem::Integer(iy_low),
                ListItem::Integer(iy_high),
                ListItem::Integer(iz_low),
                ListItem::Integer(iz_high),
            ],
        ) {
            read_abort(err);
        }
        // `write(*,'(1x,a,$)')`
        print!(" Width of pad/taper borders in X, Y, and Z: ");
        let _ = std::io::stdout().flush();
        if let Err(err) = list_read(
            &mut stdin,
            &mut [
                ListItem::Integer(&mut num_pad_x),
                ListItem::Integer(&mut mum_pad_y),
                ListItem::Integer(&mut num_pad_z),
            ],
        ) {
            read_abort(err);
        }
    }
    //
    if *ix_low < 0
        || *ix_high >= nxyz[0]
        || *iy_low < 0
        || *iy_high >= nxyz[1]
        || *iz_low < 0
        || *iz_high >= nxyz[2]
    {
        exit_error("Block not all inside volume");
    }
    *nx_box = *ix_high + 1 - *ix_low;
    *ny_box = *iy_high + 1 - *iy_low;
    *nz_box = *iz_high + 1 - *iz_low;
    //
    if no_fft {
        *nx_out = *nx_box + 2 * num_pad_x;
        *ny_out = *ny_box + 2 * mum_pad_y;
        *nz_out = *nz_box + 2 * num_pad_z;
    } else {
        *nx_out = nice_frame(2 * ((*nx_box + 1) / 2 + num_pad_x), 2, nice_fft_limit());
        *ny_out = nice_frame(2 * ((*ny_box + 1) / 2 + mum_pad_y), 2, nice_fft_limit());
        *nz_out = *nz_box;
        if *nz_out > 1 || num_pad_z > 0 {
            *nz_out = nice_frame(2 * ((*nz_box + 1) / 2 + num_pad_z), 2, nice_fft_limit());
        }
    }

    let delta = iiu_ret_delta(1);
    [*origin_x, *origin_y, *origin_z] = iiu_ret_origin(1);
    mxyz_out[0] = *nx_out;
    mxyz_out[1] = *ny_out;
    mxyz_out[2] = *nz_out;
    *nxyz_out = *mxyz_out;
    // `cellOut(1:3) = mxyzOut * delta`: integer times real, in real*4.
    for i in 0..3 {
        cell_out[i] = mxyz_out[i] as f32 * delta[i];
    }
    for value in &mut cell_out[3..6] {
        *value = 90.;
    }
    *origin_x -= delta[0] * (*ix_low - (*nx_out - *nx_box) / 2) as f32;
    *origin_y -= delta[1] * (*iy_low - (*ny_out - *ny_box) / 2) as f32;
    *origin_z -= delta[2] * (*iz_low - (*nz_out - *nz_box) / 2) as f32;
}
