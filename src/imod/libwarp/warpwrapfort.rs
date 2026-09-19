//! `IMOD/libwarp/warpwrapfort.c` core Fortran wrapper surface.
//!
//! These wrappers retain the C unit's two externally visible adaptations:
//! Fortran section numbers are one-based, and fixed-width character arguments
//! are trimmed only at their trailing NUL/space padding.

use super::{maggradfield, warpfiles, warpinterp, warputils};

pub fn newwarpfile(nx: i32, ny: i32, binning: i32, pixel_size: f32, flags: i32) -> i32 {
    warpfiles::new_warp_file(nx, ny, binning, pixel_size, flags)
}
pub fn setcurrentwarpfile(index: i32) -> i32 {
    warpfiles::set_current_warp_file(index)
}
pub fn clearwarpfile(index: i32) -> i32 {
    warpfiles::clear_warp_file(index)
}
pub fn warpfilesdone() {
    warpfiles::warp_files_done()
}
pub fn setlineartransform(iz: i32, xform: &[f32]) -> i32 {
    warpfiles::set_linear_transform(iz - 1, xform, 2)
}
#[allow(clippy::too_many_arguments)]
pub fn setwarpgrid(
    iz: i32,
    nx: i32,
    ny: i32,
    xs: f32,
    ys: f32,
    xi: f32,
    yi: f32,
    dx: &[f32],
    dy: &[f32],
    xdim: i32,
) -> i32 {
    warpfiles::set_warp_grid(iz - 1, nx, ny, xs, ys, xi, yi, dx, dy, xdim)
}
pub fn setwarppoints(iz: i32, n: i32, xc: &[f32], yc: &[f32], xv: &[f32], yv: &[f32]) -> i32 {
    warpfiles::set_warp_points(iz - 1, n, xc, yc, xv, yv)
}
pub fn getlineartransform(iz: i32, xform: &mut [f32]) -> i32 {
    warpfiles::get_linear_transform(iz - 1, xform, 2)
}
pub fn getnumwarppoints(iz: i32, n: &mut i32) -> i32 {
    warpfiles::get_num_warp_points(iz - 1, n)
}
pub fn getwarppoints(
    iz: i32,
    xc: &mut [f32],
    yc: &mut [f32],
    xv: &mut [f32],
    yv: &mut [f32],
) -> i32 {
    warpfiles::get_warp_points(iz - 1, xc, yc, xv, yv)
}
pub fn getwarpgridsize(iz: i32, nx: &mut i32, ny: &mut i32, prod: &mut i32) -> i32 {
    warpfiles::get_warp_grid_size(iz - 1, nx, ny, prod)
}
#[allow(clippy::too_many_arguments)]
pub fn setgridsizetomake(iz: i32, nx: i32, ny: i32, xs: f32, ys: f32, xi: f32, yi: f32) -> i32 {
    warpfiles::set_grid_size_to_make(iz - 1, nx, ny, xs, ys, xi, yi)
}
pub fn controlpointrange(
    iz: i32,
    xmin: &mut f32,
    xmax: &mut f32,
    ymin: &mut f32,
    ymax: &mut f32,
) -> i32 {
    warpfiles::control_point_range(iz - 1, xmin, xmax, ymin, ymax)
}
pub fn controlpointspacing(iz: i32, percentile: f32, spacing: &mut f32) -> i32 {
    warpfiles::control_point_spacing(iz - 1, percentile, spacing)
}
pub fn gridsizefromspacing(iz: i32, percentile: f32, factor: f32, full_extent: i32) -> i32 {
    warpfiles::grid_size_from_spacing(iz - 1, percentile, factor, full_extent)
}
#[allow(clippy::too_many_arguments)]
pub fn getgridparameters(
    iz: i32,
    nx: &mut i32,
    ny: &mut i32,
    xs: &mut f32,
    ys: &mut f32,
    xi: &mut f32,
    yi: &mut f32,
) -> i32 {
    warpfiles::get_grid_parameters(iz - 1, nx, ny, xs, ys, xi, yi)
}
pub fn separatelineartransform(iz: i32) -> i32 {
    warpfiles::separate_linear_transform(iz - 1)
}
#[allow(clippy::too_many_arguments)]
pub fn readwarpfile(
    filename: &[u8],
    nx: &mut i32,
    ny: &mut i32,
    nz: &mut i32,
    binning: &mut i32,
    pixel_size: &mut f32,
    version: &mut i32,
    flags: &mut i32,
) -> i32 {
    warpfiles::read_warp_file(
        &fortran_filename(filename),
        nx,
        ny,
        nz,
        binning,
        pixel_size,
        version,
        flags,
    )
}
pub fn writewarpfile(filename: &[u8], skip_backup: i32) -> i32 {
    warpfiles::write_warp_file(&fortran_filename(filename), skip_backup)
}
#[allow(clippy::too_many_arguments)]
pub fn interpolategrid(
    x: f32,
    y: f32,
    dxg: &[f32],
    dyg: &[f32],
    dim: i32,
    nx: i32,
    ny: i32,
    xs: f32,
    ys: f32,
    xi: f32,
    yi: f32,
    dx: &mut f32,
    dy: &mut f32,
) {
    warputils::interpolate_grid(x, y, dxg, dyg, dim, nx, ny, xs, ys, xi, yi, dx, dy)
}
#[allow(clippy::too_many_arguments)]
pub fn findinversepoint(
    x: f32,
    y: f32,
    dxg: &[f32],
    dyg: &[f32],
    dim: i32,
    nx: i32,
    ny: i32,
    xs: f32,
    ys: f32,
    xi: f32,
    yi: f32,
    xnew: &mut f32,
    ynew: &mut f32,
    dx: &mut f32,
    dy: &mut f32,
) {
    warputils::find_inverse_point(
        x, y, dxg, dyg, dim, nx, ny, xs, ys, xi, yi, xnew, ynew, dx, dy,
    )
}
#[allow(clippy::too_many_arguments)]
pub fn invertwarpgrid(
    dxg: &[f32],
    dyg: &[f32],
    dim: i32,
    nx: i32,
    ny: i32,
    xs: f32,
    ys: f32,
    xi: f32,
    yi: f32,
    xform: &[f32],
    xcen: f32,
    ycen: f32,
    dx_inv: &mut [f32],
    dy_inv: &mut [f32],
    xf_inv: &mut [f32],
) {
    warputils::invert_warp_grid(
        dxg, dyg, dim, nx, ny, xs, ys, xi, yi, xform, xcen, ycen, dx_inv, dy_inv, xf_inv, 2,
    )
}
#[allow(clippy::too_many_arguments)]
pub fn multiplywarpings(
    dx1: &[f32],
    dy1: &[f32],
    dim1: i32,
    nx1: i32,
    ny1: i32,
    xs1: f32,
    xi1: f32,
    ys1: f32,
    yi1: f32,
    xf1: &[f32],
    xcen: f32,
    ycen: f32,
    dx2: &[f32],
    dy2: &[f32],
    dim2: i32,
    nx2: i32,
    ny2: i32,
    xs2: f32,
    xi2: f32,
    ys2: f32,
    yi2: f32,
    xf2: &[f32],
    dx_out: &mut [f32],
    dy_out: &mut [f32],
    xf_out: &mut [f32],
    use_second: i32,
) -> i32 {
    warputils::multiply_warpings(
        dx1, dy1, dim1, nx1, ny1, xs1, ys1, xi1, yi1, xf1, xcen, ycen, dx2, dy2, dim2, nx2, ny2,
        xs2, ys2, xi2, yi2, xf2, dx_out, dy_out, xf_out, use_second, 2,
    )
}
#[allow(clippy::too_many_arguments)]
pub fn expandandextrapgrid(
    dx_grid: &mut [f32],
    dy_grid: &mut [f32],
    xdim: i32,
    ydim: i32,
    nx_grid: &mut i32,
    ny_grid: &mut i32,
    x_start: &mut f32,
    y_start: &mut f32,
    x_interval: f32,
    y_interval: f32,
    x_big_str: f32,
    y_big_str: f32,
    x_big_end: f32,
    y_big_end: f32,
    ixmin: i32,
    ixmax: i32,
    iymin: i32,
    iymax: i32,
) -> i32 {
    warputils::expand_and_extrap_grid(
        dx_grid, dy_grid, xdim, ydim, nx_grid, ny_grid, x_start, y_start, x_interval, y_interval,
        x_big_str, y_big_str, x_big_end, y_big_end, ixmin, ixmax, iymin, iymax,
    )
}
#[allow(clippy::too_many_arguments)]
pub fn warpinterp(
    array: &[f32],
    bray: &mut [f32],
    nxa: &i32,
    nya: &i32,
    nxb: &i32,
    nyb: &i32,
    amat: &[f32],
    xc: &f32,
    yc: &f32,
    xt: &f32,
    yt: &f32,
    scale: &f32,
    dmean: &f32,
    linear: &i32,
    lin_first: &i32,
    dx_grid: &[f32],
    dy_grid: &[f32],
    ixg_dim: &i32,
    nx_grid: &i32,
    ny_grid: &i32,
    x_grid_strt: &f32,
    y_grid_strt: &f32,
    x_grid_intrv: &f32,
    y_grid_intrv: &f32,
) {
    warpinterp::warpinterp_fortran(
        array,
        bray,
        nxa,
        nya,
        nxb,
        nyb,
        amat,
        xc,
        yc,
        xt,
        yt,
        scale,
        dmean,
        linear,
        lin_first,
        dx_grid,
        dy_grid,
        ixg_dim,
        nx_grid,
        ny_grid,
        x_grid_strt,
        y_grid_strt,
        x_grid_intrv,
        y_grid_intrv,
    )
}
#[allow(clippy::too_many_arguments)]
pub fn readcheckwarpfile(
    filename: &[u8],
    need_dist: i32,
    need_inv: i32,
    nx: &mut i32,
    ny: &mut i32,
    nz: &mut i32,
    ibinning: &mut i32,
    pixel_size: &mut f32,
    iflags: &mut i32,
    err_string: &mut [u8],
) -> i32 {
    let mut error = String::new();
    let status = warputils::read_check_warp_file(
        &fortran_filename(filename),
        need_dist,
        need_inv,
        nx,
        ny,
        nz,
        ibinning,
        pixel_size,
        iflags,
        &mut error,
    );
    if status < -1 {
        pad_error(err_string, &error);
    }
    status
}
#[allow(clippy::too_many_arguments)]
pub fn findmaxgridsize(
    xmin: f32,
    xmax: f32,
    ymin: f32,
    ymax: f32,
    n_control: &mut [i32],
    max_nxg: &mut i32,
    max_nyg: &mut i32,
    err_string: &mut [u8],
) -> i32 {
    let mut error = String::new();
    let status = warputils::find_max_grid_size(
        xmin, xmax, ymin, ymax, n_control, max_nxg, max_nyg, &mut error,
    );
    if status != 0 {
        pad_error(err_string, &error);
    }
    status
}
#[allow(clippy::too_many_arguments)]
pub fn getsizeadjustedgrid(
    iz: i32,
    xnbig: f32,
    ynbig: f32,
    x_offset: f32,
    y_offset: f32,
    adjust_start: i32,
    warp_scale: f32,
    i_binning: i32,
    nx_grid: &mut i32,
    ny_grid: &mut i32,
    x_grid_strt: &mut f32,
    y_grid_strt: &mut f32,
    x_grid_intrv: &mut f32,
    y_grid_intrv: &mut f32,
    field_dx: &mut [f32],
    field_dy: &mut [f32],
    ixgdim: i32,
    iygdim: i32,
    err_string: &mut [u8],
) -> i32 {
    let mut error = String::new();
    let status = warputils::get_size_adjusted_grid(
        iz - 1,
        xnbig,
        ynbig,
        x_offset,
        y_offset,
        adjust_start,
        warp_scale,
        i_binning,
        nx_grid,
        ny_grid,
        x_grid_strt,
        y_grid_strt,
        x_grid_intrv,
        y_grid_intrv,
        field_dx,
        field_dy,
        ixgdim,
        iygdim,
        &mut error,
    );
    if status != 0 {
        pad_error(err_string, &error);
    }
    status
}
#[allow(clippy::too_many_arguments)]
pub fn maggradientshift(
    x: f32,
    y: f32,
    image_nx: i32,
    image_ny: i32,
    xcen: f32,
    ycen: f32,
    pixel_size: f32,
    axis_rot: f32,
    tilt: f32,
    dmag: f32,
    rot: f32,
    dx: &mut f32,
    dy: &mut f32,
) {
    maggradfield::mag_gradient_shift(
        x, y, image_nx, image_ny, xcen, ycen, pixel_size, axis_rot, tilt, dmag, rot, dx, dy,
    )
}
#[allow(clippy::too_many_arguments)]
pub fn addmaggradfield(
    idf_dx: &[f32],
    idf_dy: &[f32],
    grad_dx: &mut [f32],
    grad_dy: &mut [f32],
    lm_grid: i32,
    image_nx: i32,
    image_ny: i32,
    nx_grid: i32,
    ny_grid: i32,
    x_grid_strt: f32,
    y_grid_strt: f32,
    x_grid_intrv: f32,
    y_grid_intrv: f32,
    xcen: f32,
    ycen: f32,
    pixel_size: f32,
    axis_rot: f32,
    tilt: f32,
    dmag_per_um: f32,
    rot_per_um: f32,
) {
    maggradfield::add_mag_grad_field(
        idf_dx,
        idf_dy,
        grad_dx,
        grad_dy,
        lm_grid,
        image_nx,
        image_ny,
        nx_grid,
        ny_grid,
        x_grid_strt,
        y_grid_strt,
        x_grid_intrv,
        y_grid_intrv,
        xcen,
        ycen,
        pixel_size,
        axis_rot,
        tilt,
        dmag_per_um,
        rot_per_um,
    )
}
#[allow(clippy::too_many_arguments)]
pub fn makemaggradfield(
    idf_dx: &mut [f32],
    idf_dy: &mut [f32],
    grad_dx: &mut [f32],
    grad_dy: &mut [f32],
    lm_grid: i32,
    image_nx: i32,
    image_ny: i32,
    nx_grid: &mut i32,
    ny_grid: &mut i32,
    x_grid_strt: &mut f32,
    y_grid_strt: &mut f32,
    x_grid_intrv: &mut f32,
    y_grid_intrv: &mut f32,
    xcen: f32,
    ycen: f32,
    pixel_size: f32,
    axis_rot: f32,
    tilt: f32,
    dmag_per_um: f32,
    rot_per_um: f32,
) {
    maggradfield::make_mag_grad_field(
        idf_dx,
        idf_dy,
        grad_dx,
        grad_dy,
        lm_grid,
        image_nx,
        image_ny,
        nx_grid,
        ny_grid,
        x_grid_strt,
        y_grid_strt,
        x_grid_intrv,
        y_grid_intrv,
        xcen,
        ycen,
        pixel_size,
        axis_rot,
        tilt,
        dmag_per_um,
        rot_per_um,
    )
}
/// C `f2cString` equivalent used by `readwarpfile`/`writewarpfile`.
pub fn fortran_filename(bytes: &[u8]) -> String {
    let end = bytes
        .iter()
        .position(|&byte| byte == 0)
        .unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end])
        .trim_end_matches(' ')
        .to_owned()
}
/// Copy a C error string into Fortran's fixed-width character storage.
fn pad_error(destination: &mut [u8], message: &str) {
    destination.fill(0);
    let count = message.len().min(destination.len().saturating_sub(1));
    destination[..count].copy_from_slice(&message.as_bytes()[..count]);
    for byte in destination.iter_mut().rev() {
        if *byte == 0 {
            *byte = b' ';
        } else {
            break;
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn wrappers_preserve_one_based_section_and_fortran_padding() {
        warpfilesdone();
        assert_eq!(newwarpfile(10, 10, 1, 1., 0), 0);
        assert_eq!(setlineartransform(1, &[1., 0., 0., 1., 3., 4.]), 0);
        let mut xf = [0.; 6];
        assert_eq!(getlineartransform(1, &mut xf), 0);
        assert_eq!(xf, [1., 0., 0., 1., 3., 4.]);
        assert_eq!(fortran_filename(b"name.warp   \0ignored"), "name.warp");
        warpfilesdone();
    }
    #[test]
    fn inversion_wrapper_uses_source_two_row_transform_layout() {
        let grid = [0.; 4];
        let xf = [1., 0., 0., 1., 0., 0.];
        let mut dx = [9.; 4];
        let mut dy = [9.; 4];
        let mut inv = [0.; 6];
        invertwarpgrid(
            &grid, &grid, 2, 2, 2, 0., 0., 1., 1., &xf, 0., 0., &mut dx, &mut dy, &mut inv,
        );
        assert_eq!(dx, [0.; 4]);
        assert_eq!(dy, [0.; 4]);
        assert_eq!(inv, xf);
    }
    #[test]
    fn composition_wrapper_uses_fortran_argument_order_and_two_rows() {
        let grid = [0.; 4];
        let xf = [1., 0., 0., 1., 0., 0.];
        let mut dx = [7.; 4];
        let mut dy = [7.; 4];
        let mut out = [0.; 6];
        assert_eq!(
            multiplywarpings(
                &grid, &grid, 2, 2, 2, 0., 1., 0., 1., &xf, 0., 0., &grid, &grid, 2, 2, 2, 0., 1.,
                0., 1., &xf, &mut dx, &mut dy, &mut out, 0,
            ),
            0
        );
        assert_eq!(dx, [0.; 4]);
        assert_eq!(dy, [0.; 4]);
        assert_eq!(out, xf);
    }
    #[test]
    fn write_wrapper_trims_fortran_filename_padding() {
        warpfilesdone();
        assert_eq!(newwarpfile(4, 5, 1, 1., 0), 0);
        let path =
            std::env::temp_dir().join(format!("imod-rs-warpwrapfort-{}.xf", std::process::id()));
        let mut padded = path.to_string_lossy().into_owned().into_bytes();
        padded.extend_from_slice(b"   \0ignored");
        assert_eq!(writewarpfile(&padded, 1), 0);
        assert!(path.exists());
        std::fs::remove_file(path).unwrap();
        warpfilesdone();
    }
    #[test]
    fn read_check_error_is_padded_for_a_fortran_character_buffer() {
        warpfilesdone();
        let (mut nx, mut ny, mut nz, mut binning, mut flags) = (0, 0, 0, 0, 0);
        let mut pixel_size = 0.;
        let mut error = [0_u8; 64];
        assert_eq!(
            readcheckwarpfile(
                b"missing.warp  ",
                1,
                0,
                &mut nx,
                &mut ny,
                &mut nz,
                &mut binning,
                &mut pixel_size,
                &mut flags,
                &mut error,
            ),
            -2
        );
        assert!(error.starts_with(b"OPENING OR READING DISTORTION FILE"));
        assert_eq!(error[63], b' ');
    }
}
