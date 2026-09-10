//! Translation of `IMOD/libwarp/warpfiles.c`.
#![allow(dead_code)]
#![allow(static_mut_refs)]

use crate::imod::libcfshr::linearxforms::{xf_copy, xf_mult, xf_unit};
use crate::imod::libwarp::delaunay::{delaunay_build, delaunay_destroy};
use crate::imod::libwarp::nn::Point;
use crate::imod::libwarp::nnai::{nnai_build, nnai_destroy, nnai_interpolate, nnai_setwmin};
use crate::imod::libwarp::warputils::extract_linear_xform;
use crate::imod::libwarp::warputils::{extrapolate_done, extrapolate_grid};
use std::ffi::{CStr, CString};
use std::fs;

pub const WARP_INVERSE: i32 = 1;
pub const WARP_CONTROL_PTS: i32 = 2;

#[derive(Clone)]
pub struct Warping {
    pub xform: [f32; 6],
    pub nx_grid: i32,
    pub ny_grid: i32,
    pub x_start: f32,
    pub y_start: f32,
    pub x_interval: f32,
    pub y_interval: f32,
    pub x_vector: Vec<f32>,
    pub y_vector: Vec<f32>,
    pub n_control: i32,
    pub x_control: Vec<f32>,
    pub y_control: Vec<f32>,
    pub max_vectors: i32,
}
pub struct WarpFile {
    pub nx: i32,
    pub ny: i32,
    pub num_frames: i32,
    pub binning: i32,
    pub pixel_size: f32,
    pub flags: i32,
    pub warpings: Vec<Warping>,
    pub in_use: i32,
}
static mut S_WARP_FILES: Option<Vec<WarpFile>> = None;
static mut S_CUR_FILE_IND: i32 = -1;
// Source `warpfiles.c` control-grid cache.  These retain the interpolation geometry,
// whereas vector values are deliberately supplied afresh on every call.
static mut S_GRID_X: *mut f64 = core::ptr::null_mut();
static mut S_GRID_Y: *mut f64 = core::ptr::null_mut();
static mut S_SOLVED: *mut i8 = core::ptr::null_mut();
static mut S_DPOINTS: *mut Point = core::ptr::null_mut();
static mut S_DELAUNAY: *mut crate::imod::libwarp::delaunay::Delaunay = core::ptr::null_mut();
static mut S_NN_INTERP: *mut crate::imod::libwarp::nnai::Nnai = core::ptr::null_mut();
static mut S_LAST_NUM_CONT: i32 = 0;
static mut S_LAST_NX_GRID: i32 = 0;
static mut S_LAST_NY_GRID: i32 = 0;
static mut S_LAST_X_START: f32 = -1.;
static mut S_LAST_Y_START: f32 = -1.;
static mut S_LAST_X_INTERVAL: f32 = 0.;
static mut S_LAST_Y_INTERVAL: f32 = 0.;
static mut S_LAST_XDIM: i32 = 0;

/// Original static `freeStaticArrays` (`warpfiles.c:960`).
unsafe fn free_static_arrays() {
    unsafe {
        libc::free(S_DPOINTS.cast());
        libc::free(S_GRID_X.cast());
        libc::free(S_GRID_Y.cast());
        libc::free(S_SOLVED.cast());
        if !S_DELAUNAY.is_null() {
            delaunay_destroy(S_DELAUNAY);
        }
        if !S_NN_INTERP.is_null() {
            nnai_destroy(S_NN_INTERP);
        }
        S_DPOINTS = core::ptr::null_mut();
        S_GRID_X = core::ptr::null_mut();
        S_GRID_Y = core::ptr::null_mut();
        S_SOLVED = core::ptr::null_mut();
        S_DELAUNAY = core::ptr::null_mut();
        S_NN_INTERP = core::ptr::null_mut();
        S_LAST_NUM_CONT = 0;
        extrapolate_done();
    }
}

/// Original static `pointsMatchStatArray` (`warpfiles.c:977`).
unsafe fn points_match_stat_array(warp: &Warping) -> i32 {
    unsafe {
        if S_DPOINTS.is_null() || S_LAST_NUM_CONT != warp.n_control {
            return 0;
        }
        for index in 0..warp.n_control as usize {
            if ((*S_DPOINTS.add(index)).x - warp.x_control[index] as f64).abs() > 1.0e-3
                || ((*S_DPOINTS.add(index)).y - warp.y_control[index] as f64).abs() > 1.0e-3
            {
                return 0;
            }
        }
        1
    }
}

/// Original static `initWarping` (`warpfiles.c:95`).
pub fn init_warping() -> Warping {
    let mut xform = [0.; 6];
    xf_unit(&mut xform, 1., 2);
    Warping {
        xform,
        nx_grid: 0,
        ny_grid: 0,
        x_start: 0.,
        y_start: 0.,
        x_interval: 0.,
        y_interval: 0.,
        x_vector: Vec::new(),
        y_vector: Vec::new(),
        n_control: 0,
        x_control: Vec::new(),
        y_control: Vec::new(),
        max_vectors: 0,
    }
}
/// Original static `addWarpFile` (`warpfiles.c:104`).
pub unsafe fn add_warp_file() -> i32 {
    unsafe {
        let files = S_WARP_FILES.get_or_insert_with(Vec::new);
        if let Some(ind) = files.iter().position(|file| file.in_use == 0) {
            let file = &mut files[ind];
            file.num_frames = 0;
            file.warpings.clear();
            file.in_use = 1;
            return ind as i32;
        }
        files.push(WarpFile {
            nx: 0,
            ny: 0,
            num_frames: 0,
            binning: 0,
            pixel_size: 0.,
            flags: 0,
            warpings: Vec::new(),
            in_use: 1,
        });
        (files.len() - 1) as i32
    }
}
/// Original `newWarpFile` (`warpfiles.c:144`).
pub unsafe fn new_warp_file(nx: i32, ny: i32, binning: i32, pixel_size: f32, flags: i32) -> i32 {
    unsafe {
        let err = add_warp_file();
        if err < 0 {
            return err;
        }
        set_current_warp_file(err);
        let file = &mut S_WARP_FILES.as_mut().unwrap()[err as usize];
        file.nx = nx;
        file.ny = ny;
        file.binning = binning;
        file.pixel_size = pixel_size;
        file.flags = flags;
        err
    }
}
/// Original static `addWarpingsIfNeeded` (`warpfiles.c:400`).
pub unsafe fn add_warpings_if_needed(iz: i32) -> i32 {
    unsafe {
        if S_CUR_FILE_IND < 0 || iz < 0 {
            return 1;
        }
        let file = &mut S_WARP_FILES.as_mut().unwrap()[S_CUR_FILE_IND as usize];
        if iz < file.num_frames {
            return 0;
        }
        for _ in file.num_frames..=iz {
            file.warpings.push(init_warping());
        }
        file.num_frames = iz + 1;
        0
    }
}
/// Original `setCurrentWarpFile` (`warpfiles.c:419`).
pub unsafe fn set_current_warp_file(index: i32) -> i32 {
    unsafe {
        S_CUR_FILE_IND = -1;
        if index < 0
            || S_WARP_FILES
                .as_ref()
                .is_none_or(|files| index as usize >= files.len())
        {
            return -1;
        }
        S_CUR_FILE_IND = index;
        0
    }
}
/// Original static `deleteWarpFile` (`warpfiles.c:468`).
pub fn delete_warp_file(warp_file: &mut WarpFile) {
    warp_file.num_frames = 0;
    warp_file.in_use = 0;
    warp_file.warpings.clear();
}
/// Original `clearWarpFile` (`warpfiles.c:434`).
pub unsafe fn clear_warp_file(index: i32) -> i32 {
    unsafe {
        let Some(files) = S_WARP_FILES.as_mut() else {
            return 1;
        };
        if index < 0 || index as usize >= files.len() {
            return 1;
        }
        if files[index as usize].in_use == 0 {
            return 2;
        }
        delete_warp_file(&mut files[index as usize]);
        if S_CUR_FILE_IND == index {
            S_CUR_FILE_IND = -1;
        }
        0
    }
}
/// Original `warpFilesDone` (`warpfiles.c:452`).
pub unsafe fn warp_files_done() {
    unsafe {
        free_static_arrays();
        S_WARP_FILES = None;
        S_CUR_FILE_IND = -1;
    }
}
/// Original `getWarpFileSize` (`warpfiles.c:489`).
pub unsafe fn get_warp_file_size(
    nx: *mut i32,
    ny: *mut i32,
    nz: *mut i32,
    if_control: *mut i32,
) -> i32 {
    unsafe {
        if S_CUR_FILE_IND < 0 {
            return 1;
        }
        let file = &S_WARP_FILES.as_ref().unwrap()[S_CUR_FILE_IND as usize];
        *nx = file.nx;
        *ny = file.ny;
        *nz = file.num_frames;
        *if_control = if file.flags & WARP_CONTROL_PTS != 0 {
            1
        } else {
            0
        };
        0
    }
}
/// Original `setLinearTransform` (`warpfiles.c:505`).
pub unsafe fn set_linear_transform(iz: i32, xform: *mut f32, rows: i32) -> i32 {
    unsafe {
        if add_warpings_if_needed(iz) != 0 {
            return 1;
        }
        let warp =
            &mut S_WARP_FILES.as_mut().unwrap()[S_CUR_FILE_IND as usize].warpings[iz as usize];
        xf_copy(
            core::slice::from_raw_parts(xform, (3 * rows) as usize),
            rows as usize,
            &mut warp.xform,
            2,
        );
        0
    }
}
/// Original `getLinearTransform` (`warpfiles.c:680`).
pub unsafe fn get_linear_transform(iz: i32, xform: *mut f32, rows: i32) -> i32 {
    unsafe {
        if S_CUR_FILE_IND < 0 {
            return 1;
        }
        let file = &S_WARP_FILES.as_ref().unwrap()[S_CUR_FILE_IND as usize];
        if iz < 0 || iz >= file.num_frames {
            return 1;
        }
        xf_copy(
            &file.warpings[iz as usize].xform,
            2,
            core::slice::from_raw_parts_mut(xform, (3 * rows) as usize),
            rows as usize,
        );
        0
    }
}
/// Original `setWarpGrid` (`warpfiles.c:522`).
pub unsafe fn set_warp_grid(
    iz: i32,
    nx_grid: i32,
    ny_grid: i32,
    x_start: f32,
    y_start: f32,
    x_interval: f32,
    y_interval: f32,
    dx_grid: *mut f32,
    dy_grid: *mut f32,
    xdim: i32,
) -> i32 {
    unsafe {
        if nx_grid <= 0 || ny_grid <= 0 || add_warpings_if_needed(iz) != 0 {
            return 1;
        }
        let warp =
            &mut S_WARP_FILES.as_mut().unwrap()[S_CUR_FILE_IND as usize].warpings[iz as usize];
        warp.nx_grid = nx_grid;
        warp.ny_grid = ny_grid;
        warp.x_start = x_start;
        warp.y_start = y_start;
        warp.x_interval = x_interval;
        warp.y_interval = y_interval;
        let n = (nx_grid * ny_grid) as usize;
        warp.x_vector.resize(n, 0.);
        warp.y_vector.resize(n, 0.);
        warp.max_vectors = nx_grid * ny_grid;
        for iy in 0..ny_grid {
            for ix in 0..nx_grid {
                warp.x_vector[(ix + iy * nx_grid) as usize] =
                    *dx_grid.add((ix + iy * xdim) as usize);
                warp.y_vector[(ix + iy * nx_grid) as usize] =
                    *dy_grid.add((ix + iy * xdim) as usize);
            }
        }
        0
    }
}
/// Original `setWarpPoints` (`warpfiles.c:566`).
pub unsafe fn set_warp_points(
    iz: i32,
    n_control: i32,
    x_control: *mut f32,
    y_control: *mut f32,
    x_vector: *mut f32,
    y_vector: *mut f32,
) -> i32 {
    unsafe {
        if n_control < 0 || add_warpings_if_needed(iz) != 0 {
            return 1;
        }
        let warp =
            &mut S_WARP_FILES.as_mut().unwrap()[S_CUR_FILE_IND as usize].warpings[iz as usize];
        let n = n_control as usize;
        warp.n_control = n_control;
        warp.x_control = core::slice::from_raw_parts(x_control, n).to_vec();
        warp.y_control = core::slice::from_raw_parts(y_control, n).to_vec();
        warp.x_vector = core::slice::from_raw_parts(x_vector, n).to_vec();
        warp.y_vector = core::slice::from_raw_parts(y_vector, n).to_vec();
        warp.max_vectors = n_control;
        0
    }
}
/// Original `addWarpPoint` (`warpfiles.c:612`).
pub unsafe fn add_warp_point(
    iz: i32,
    x_control: f32,
    y_control: f32,
    x_vector: f32,
    y_vector: f32,
) -> i32 {
    unsafe {
        if add_warpings_if_needed(iz) != 0 {
            return -1;
        }
        let warp =
            &mut S_WARP_FILES.as_mut().unwrap()[S_CUR_FILE_IND as usize].warpings[iz as usize];
        warp.x_control.push(x_control);
        warp.y_control.push(y_control);
        warp.x_vector.push(x_vector);
        warp.y_vector.push(y_vector);
        warp.n_control += 1;
        warp.max_vectors = warp.max_vectors.max(warp.n_control);
        warp.n_control
    }
}
/// Original `removeWarpPoint` (`warpfiles.c:655`).
pub unsafe fn remove_warp_point(iz: i32, index: i32) -> i32 {
    unsafe {
        if S_CUR_FILE_IND < 0 {
            return 1;
        }
        let file = &mut S_WARP_FILES.as_mut().unwrap()[S_CUR_FILE_IND as usize];
        if iz < 0 || iz >= file.num_frames {
            return 1;
        }
        let warp = &mut file.warpings[iz as usize];
        if index < 0 || index >= warp.n_control {
            return 1;
        }
        let i = index as usize;
        warp.x_control.remove(i);
        warp.y_control.remove(i);
        warp.x_vector.remove(i);
        warp.y_vector.remove(i);
        warp.n_control -= 1;
        0
    }
}
/// Original `getNumWarpPoints` (`warpfiles.c:695`).
pub unsafe fn get_num_warp_points(iz: i32, n_control: *mut i32) -> i32 {
    unsafe {
        if S_CUR_FILE_IND < 0 {
            return 1;
        }
        let file = &S_WARP_FILES.as_ref().unwrap()[S_CUR_FILE_IND as usize];
        if iz >= file.num_frames {
            return 1;
        }
        *n_control = if iz < 0 {
            file.warpings
                .iter()
                .map(|warp| warp.n_control)
                .max()
                .unwrap_or(0)
        } else {
            file.warpings[iz as usize].n_control
        };
        0
    }
}
/// Original `getWarpPoints` (`warpfiles.c:719`).
pub unsafe fn get_warp_points(
    iz: i32,
    x_control: *mut f32,
    y_control: *mut f32,
    x_vector: *mut f32,
    y_vector: *mut f32,
) -> i32 {
    unsafe {
        if S_CUR_FILE_IND < 0 {
            return 1;
        }
        let file = &S_WARP_FILES.as_ref().unwrap()[S_CUR_FILE_IND as usize];
        if iz < 0 || iz >= file.num_frames {
            return 1;
        }
        let warp = &file.warpings[iz as usize];
        for i in 0..warp.n_control as usize {
            *x_control.add(i) = warp.x_control[i];
            *y_control.add(i) = warp.y_control[i];
            *x_vector.add(i) = warp.x_vector[i];
            *y_vector.add(i) = warp.y_vector[i];
        }
        0
    }
}
/// Original `getWarpPointArrays` (`warpfiles.c:742`).
pub unsafe fn get_warp_point_arrays(
    iz: i32,
    x_control: *mut *mut f32,
    y_control: *mut *mut f32,
    x_vector: *mut *mut f32,
    y_vector: *mut *mut f32,
) -> i32 {
    unsafe {
        if S_CUR_FILE_IND < 0 {
            return 1;
        }
        let file = &mut S_WARP_FILES.as_mut().unwrap()[S_CUR_FILE_IND as usize];
        if iz < 0 || iz >= file.num_frames {
            return 1;
        }
        let warp = &mut file.warpings[iz as usize];
        *x_control = warp.x_control.as_mut_ptr();
        *y_control = warp.y_control.as_mut_ptr();
        *x_vector = warp.x_vector.as_mut_ptr();
        *y_vector = warp.y_vector.as_mut_ptr();
        0
    }
}
/// Original `getWarpGridSize` (`warpfiles.c:763`).
pub unsafe fn get_warp_grid_size(
    iz: i32,
    nx_max: *mut i32,
    ny_max: *mut i32,
    prod_max: *mut i32,
) -> i32 {
    unsafe {
        if S_CUR_FILE_IND < 0 {
            return 1;
        }
        let file = &S_WARP_FILES.as_ref().unwrap()[S_CUR_FILE_IND as usize];
        if iz >= file.num_frames {
            return 1;
        }
        let warps: Box<dyn Iterator<Item = &Warping>> = if iz < 0 {
            Box::new(file.warpings.iter())
        } else {
            Box::new(core::iter::once(&file.warpings[iz as usize]))
        };
        *nx_max = 0;
        *ny_max = 0;
        *prod_max = 0;
        for warp in warps {
            *nx_max = (*nx_max).max(warp.nx_grid);
            *ny_max = (*ny_max).max(warp.ny_grid);
            *prod_max = (*prod_max).max(warp.nx_grid * warp.ny_grid);
        }
        0
    }
}
/// Original `getGridParameters` (`warpfiles.c:790`).
pub unsafe fn get_grid_parameters(
    iz: i32,
    nx_grid: *mut i32,
    ny_grid: *mut i32,
    x_start: *mut f32,
    y_start: *mut f32,
    x_interval: *mut f32,
    y_interval: *mut f32,
) -> i32 {
    unsafe {
        if S_CUR_FILE_IND < 0 {
            return 1;
        }
        let file = &S_WARP_FILES.as_ref().unwrap()[S_CUR_FILE_IND as usize];
        if iz < 0 || iz >= file.num_frames {
            return 1;
        }
        let warp = &file.warpings[iz as usize];
        *nx_grid = warp.nx_grid;
        *ny_grid = warp.ny_grid;
        *x_start = warp.x_start;
        *y_start = warp.y_start;
        *x_interval = warp.x_interval;
        *y_interval = warp.y_interval;
        0
    }
}
/// Original `setGridSizeToMake` (`warpfiles.c:813`).
pub unsafe fn set_grid_size_to_make(
    iz: i32,
    nx_grid: i32,
    ny_grid: i32,
    x_start: f32,
    y_start: f32,
    x_interval: f32,
    y_interval: f32,
) -> i32 {
    unsafe {
        if S_CUR_FILE_IND < 0 {
            return 1;
        }
        let file = &mut S_WARP_FILES.as_mut().unwrap()[S_CUR_FILE_IND as usize];
        if file.flags & WARP_CONTROL_PTS == 0 || iz < 0 || iz >= file.num_frames {
            return 1;
        }
        let warp = &mut file.warpings[iz as usize];
        warp.nx_grid = nx_grid;
        warp.ny_grid = ny_grid;
        warp.x_start = x_start;
        warp.y_start = y_start;
        warp.x_interval = x_interval;
        warp.y_interval = y_interval;
        0
    }
}
/// Original `controlPointRange` (`warpfiles.c:836`).
pub unsafe fn control_point_range(
    iz: i32,
    xmin: *mut f32,
    xmax: *mut f32,
    ymin: *mut f32,
    ymax: *mut f32,
) -> i32 {
    unsafe {
        if S_CUR_FILE_IND < 0 {
            return 1;
        }
        let file = &S_WARP_FILES.as_ref().unwrap()[S_CUR_FILE_IND as usize];
        if file.flags & WARP_CONTROL_PTS == 0 || iz < 0 || iz >= file.num_frames {
            return 1;
        }
        let warp = &file.warpings[iz as usize];
        if warp.n_control == 0 {
            return 1;
        }
        *xmin = warp.x_control[0];
        *xmax = *xmin;
        *ymin = warp.y_control[0];
        *ymax = *ymin;
        for i in 1..warp.n_control as usize {
            *xmin = (*xmin).min(warp.x_control[i]);
            *xmax = (*xmax).max(warp.x_control[i]);
            *ymin = (*ymin).min(warp.y_control[i]);
            *ymax = (*ymax).max(warp.y_control[i]);
        }
        0
    }
}
/// Original `controlPointSpacing` (`warpfiles.c:866`).
pub unsafe fn control_point_spacing(iz: i32, percentile: f32, spacing: *mut f32) -> i32 {
    unsafe {
        let percentile = if percentile < 0. { 0.5 } else { percentile };
        if S_CUR_FILE_IND < 0 {
            return 1;
        }
        let file = &S_WARP_FILES.as_ref().unwrap()[S_CUR_FILE_IND as usize];
        if file.flags & WARP_CONTROL_PTS == 0 || iz < 0 || iz >= file.num_frames || percentile > 1.
        {
            return 1;
        }
        let warp = &file.warpings[iz as usize];
        if warp.n_control < 2 {
            return 1;
        }
        let mut nearest = Vec::with_capacity(warp.n_control as usize);
        for i in 0..warp.n_control as usize {
            let mut min_sq = 1.0e30_f32;
            for j in 0..warp.n_control as usize {
                if i != j {
                    let dx = warp.x_control[i] - warp.x_control[j];
                    let dxsq = dx * dx;
                    if dxsq < min_sq {
                        let dy = warp.y_control[i] - warp.y_control[j];
                        let square = dxsq + dy * dy;
                        if square < min_sq {
                            min_sq = square;
                        }
                    }
                }
            }
            nearest.push(min_sq);
        }
        nearest.sort_by(|left, right| left.partial_cmp(right).unwrap());
        let sel = ((percentile * warp.n_control as f32).round() as i32).min(warp.n_control - 1);
        *spacing = nearest[sel as usize].sqrt();
        0
    }
}
/// Original `gridSizeFromSpacing` (`warpfiles.c:932`).
pub unsafe fn grid_size_from_spacing(
    iz: i32,
    percentile: f32,
    factor: f32,
    full_extent: i32,
) -> i32 {
    unsafe {
        let factor = if factor < 0. { 0.15 } else { factor };
        let mut spacing = 0.;
        if control_point_spacing(iz, percentile, &mut spacing) != 0 || factor > 2. || factor < 0.05
        {
            return 1;
        }
        let file = &mut S_WARP_FILES.as_mut().unwrap()[S_CUR_FILE_IND as usize];
        let warp = &mut file.warpings[iz as usize];
        spacing *= factor;
        let (xmin, xmax, ymin, ymax) = if full_extent != 0 {
            (
                spacing / 10.,
                file.nx as f32 - spacing / 10.,
                spacing / 10.,
                file.ny as f32 - spacing / 10.,
            )
        } else {
            let (mut xmin, mut xmax, mut ymin, mut ymax) = (0., 0., 0., 0.);
            if control_point_range(iz, &mut xmin, &mut xmax, &mut ymin, &mut ymax) != 0 {
                return 1;
            }
            (xmin, xmax, ymin, ymax)
        };
        warp.nx_grid = (1. + (xmax - xmin) / spacing).ceil() as i32;
        warp.nx_grid = warp.nx_grid.max(2);
        warp.x_interval = (xmax - xmin) / (warp.nx_grid - 1) as f32;
        warp.ny_grid = (1. + (ymax - ymin) / spacing).ceil() as i32;
        warp.ny_grid = warp.ny_grid.max(2);
        warp.y_interval = (ymax - ymin) / (warp.ny_grid - 1) as f32;
        warp.x_start = xmin;
        warp.y_start = ymin;
        0
    }
}
/// Original `getWarpGrid` (`warpfiles.c:1002`), direct grid-file branch.
pub unsafe fn get_warp_grid(
    iz: i32,
    nx_grid: *mut i32,
    ny_grid: *mut i32,
    x_start: *mut f32,
    y_start: *mut f32,
    x_interval: *mut f32,
    y_interval: *mut f32,
    dx_grid: *mut f32,
    dy_grid: *mut f32,
    mut xdim: i32,
) -> i32 {
    unsafe {
        if S_CUR_FILE_IND < 0 {
            return 1;
        }
        let file = &S_WARP_FILES.as_ref().unwrap()[S_CUR_FILE_IND as usize];
        if iz < 0 || iz >= file.num_frames {
            return 1;
        }
        let warp = &file.warpings[iz as usize];
        if warp.nx_grid == 0 || warp.ny_grid == 0 {
            return 1;
        }
        if file.flags & WARP_CONTROL_PTS != 0 {
            if warp.n_control < 3 {
                return 1;
            }
            let ngrid = xdim.max(warp.nx_grid) * warp.ny_grid;
            if xdim <= 0 {
                xdim = warp.nx_grid;
            }
            let mut xmin = warp.x_control[0];
            let mut xmax = xmin;
            let mut ymin = warp.y_control[0];
            let mut ymax = ymin;
            for index in 1..warp.n_control as usize {
                xmin = xmin.min(warp.x_control[index]);
                xmax = xmax.max(warp.x_control[index]);
                ymin = ymin.min(warp.y_control[index]);
                ymax = ymax.max(warp.y_control[index]);
            }
            let off_x_solve =
                (((xmin - 0.1 - warp.x_start) / warp.x_interval).ceil() as i32).max(0);
            let last_x = (((xmax + 0.1 - warp.x_start) / warp.x_interval).floor() as i32)
                .min(warp.nx_grid - 1);
            let nx_solve = last_x + 1 - off_x_solve;
            let off_y_solve =
                (((ymin - 0.1 - warp.y_start) / warp.y_interval).ceil() as i32).max(0);
            let last_y = (((ymax + 0.1 - warp.y_start) / warp.y_interval).floor() as i32)
                .min(warp.ny_grid - 1);
            let ny_solve = last_y + 1 - off_y_solve;
            let nsolve = nx_solve * ny_solve;
            if nsolve <= 0 {
                return 1;
            }
            let mut need_new = S_DPOINTS.is_null()
                || S_GRID_X.is_null()
                || S_GRID_Y.is_null()
                || S_SOLVED.is_null()
                || S_DELAUNAY.is_null()
                || S_NN_INTERP.is_null()
                || S_LAST_NUM_CONT != warp.n_control
                || S_LAST_NX_GRID != warp.nx_grid
                || S_LAST_NY_GRID != warp.ny_grid
                || S_LAST_X_START != warp.x_start
                || S_LAST_Y_START != warp.y_start
                || S_LAST_X_INTERVAL != warp.x_interval
                || S_LAST_Y_INTERVAL != warp.y_interval
                || S_LAST_XDIM != xdim;
            if !need_new && points_match_stat_array(warp) == 0 {
                need_new = true;
            }
            if need_new {
                free_static_arrays();
                S_DPOINTS =
                    libc::malloc(warp.n_control as usize * core::mem::size_of::<Point>()).cast();
                S_GRID_X = libc::malloc(nsolve as usize * core::mem::size_of::<f64>()).cast();
                S_GRID_Y = libc::malloc(nsolve as usize * core::mem::size_of::<f64>()).cast();
                S_SOLVED = libc::malloc((6 * ngrid) as usize).cast();
                if S_DPOINTS.is_null()
                    || S_GRID_X.is_null()
                    || S_GRID_Y.is_null()
                    || S_SOLVED.is_null()
                {
                    free_static_arrays();
                    return 1;
                }
                for index in 0..warp.n_control as usize {
                    *S_DPOINTS.add(index) = Point {
                        x: warp.x_control[index] as f64,
                        y: warp.y_control[index] as f64,
                        z: 0.0,
                    };
                }
                for iy in 0..ny_solve {
                    for ix in 0..nx_solve {
                        let index = (ix + iy * nx_solve) as usize;
                        *S_GRID_X.add(index) =
                            (warp.x_start + (ix + off_x_solve) as f32 * warp.x_interval) as f64;
                        *S_GRID_Y.add(index) =
                            (warp.y_start + (iy + off_y_solve) as f32 * warp.y_interval) as f64;
                    }
                }
                let prune_criteria = [0.2, 0.1];
                S_DELAUNAY = delaunay_build(
                    warp.n_control,
                    S_DPOINTS,
                    0,
                    core::ptr::null_mut(),
                    2,
                    prune_criteria.as_ptr() as *mut f64,
                );
                if !S_DELAUNAY.is_null() {
                    S_NN_INTERP = nnai_build(S_DELAUNAY, nsolve, S_GRID_X, S_GRID_Y);
                }
                if S_NN_INTERP.is_null() {
                    free_static_arrays();
                    return 1;
                }
                S_LAST_NUM_CONT = warp.n_control;
                S_LAST_NX_GRID = warp.nx_grid;
                S_LAST_NY_GRID = warp.ny_grid;
                S_LAST_X_START = warp.x_start;
                S_LAST_Y_START = warp.y_start;
                S_LAST_X_INTERVAL = warp.x_interval;
                S_LAST_Y_INTERVAL = warp.y_interval;
                S_LAST_XDIM = xdim;
            }
            if !need_new {
                for index in 0..ngrid as usize {
                    *dx_grid.add(index) = 0.;
                    *dy_grid.add(index) = 0.;
                    *S_SOLVED.add(index) = 0;
                }
                nnai_setwmin(S_NN_INTERP, 0.0);
                let mut input: Vec<f64> = warp.x_vector[..warp.n_control as usize]
                    .iter()
                    .map(|value| *value as f64)
                    .collect();
                let mut output = vec![0.; nsolve as usize];
                nnai_interpolate(S_NN_INTERP, input.as_mut_ptr(), output.as_mut_ptr());
                for iy in 0..ny_solve {
                    for ix in 0..nx_solve {
                        let source = (ix + iy * nx_solve) as usize;
                        if !output[source].is_nan() {
                            let destination =
                                (ix + off_x_solve + (iy + off_y_solve) * xdim) as usize;
                            *dx_grid.add(destination) = output[source] as f32;
                            *S_SOLVED.add(destination) = 1;
                        }
                    }
                }
                input.clear();
                input.extend(
                    warp.y_vector[..warp.n_control as usize]
                        .iter()
                        .map(|value| *value as f64),
                );
                nnai_interpolate(S_NN_INTERP, input.as_mut_ptr(), output.as_mut_ptr());
                for iy in 0..ny_solve {
                    for ix in 0..nx_solve {
                        let destination = (ix + off_x_solve + (iy + off_y_solve) * xdim) as usize;
                        if *S_SOLVED.add(destination) != 0 {
                            *dy_grid.add(destination) =
                                output[(ix + iy * nx_solve) as usize] as f32;
                        }
                    }
                }
                if extrapolate_grid(
                    dx_grid,
                    dy_grid,
                    S_SOLVED,
                    xdim,
                    warp.nx_grid,
                    warp.ny_grid,
                    warp.x_interval,
                    warp.y_interval,
                    1,
                ) != 0
                {
                    S_LAST_NUM_CONT = 0;
                    return 1;
                }
                *nx_grid = warp.nx_grid;
                *ny_grid = warp.ny_grid;
                *x_start = warp.x_start;
                *y_start = warp.y_start;
                *x_interval = warp.x_interval;
                *y_interval = warp.y_interval;
                return 0;
            }
            for index in 0..ngrid {
                *dx_grid.add(index as usize) = 0.0;
                *dy_grid.add(index as usize) = 0.0;
            }
            let points = libc::malloc(warp.n_control as usize * core::mem::size_of::<Point>())
                .cast::<Point>();
            for index in 0..warp.n_control as usize {
                *points.add(index) = Point {
                    x: warp.x_control[index] as f64,
                    y: warp.y_control[index] as f64,
                    z: 0.0,
                };
            }
            let mut grid_x = Vec::with_capacity(nsolve as usize);
            let mut grid_y = Vec::with_capacity(nsolve as usize);
            for iy in 0..ny_solve {
                for ix in 0..nx_solve {
                    grid_x
                        .push((warp.x_start + (ix + off_x_solve) as f32 * warp.x_interval) as f64);
                    grid_y
                        .push((warp.y_start + (iy + off_y_solve) as f32 * warp.y_interval) as f64);
                }
            }
            let prune_criteria = [0.2, 0.1];
            let delaunay = delaunay_build(
                warp.n_control,
                points,
                0,
                core::ptr::null_mut(),
                2,
                prune_criteria.as_ptr() as *mut f64,
            );
            if delaunay.is_null() {
                return 1;
            }
            let interpolator =
                nnai_build(delaunay, nsolve, grid_x.as_mut_ptr(), grid_y.as_mut_ptr());
            nnai_setwmin(interpolator, 0.0);
            let mut input: Vec<f64> = warp.x_vector[..warp.n_control as usize]
                .iter()
                .map(|value| *value as f64)
                .collect();
            let mut output = vec![0.0; nsolve as usize];
            nnai_interpolate(interpolator, input.as_mut_ptr(), output.as_mut_ptr());
            for index in 0..ngrid as usize {
                *S_SOLVED.add(index) = 0;
            }
            for iy in 0..ny_solve {
                for ix in 0..nx_solve {
                    let source = (ix + iy * nx_solve) as usize;
                    if !output[source].is_nan() {
                        let destination = (ix + off_x_solve + (iy + off_y_solve) * xdim) as usize;
                        *dx_grid.add(destination) = output[source] as f32;
                        *S_SOLVED.add(destination) = 1;
                    }
                }
            }
            input.clear();
            input.extend(
                warp.y_vector[..warp.n_control as usize]
                    .iter()
                    .map(|value| *value as f64),
            );
            nnai_interpolate(interpolator, input.as_mut_ptr(), output.as_mut_ptr());
            for iy in 0..ny_solve {
                for ix in 0..nx_solve {
                    let source = (ix + iy * nx_solve) as usize;
                    let destination = (ix + off_x_solve + (iy + off_y_solve) * xdim) as usize;
                    if *S_SOLVED.add(destination) != 0 {
                        *dy_grid.add(destination) = output[source] as f32;
                    }
                }
            }
            let error = extrapolate_grid(
                dx_grid,
                dy_grid,
                S_SOLVED,
                xdim,
                warp.nx_grid,
                warp.ny_grid,
                warp.x_interval,
                warp.y_interval,
                0,
            );
            nnai_destroy(interpolator);
            delaunay_destroy(delaunay);
            if error != 0 {
                return 1;
            }
            *nx_grid = warp.nx_grid;
            *ny_grid = warp.ny_grid;
            *x_start = warp.x_start;
            *y_start = warp.y_start;
            *x_interval = warp.x_interval;
            *y_interval = warp.y_interval;
            return 0;
        }
        *nx_grid = warp.nx_grid;
        *ny_grid = warp.ny_grid;
        *x_start = warp.x_start;
        *y_start = warp.y_start;
        *x_interval = warp.x_interval;
        *y_interval = warp.y_interval;
        if xdim <= 0 {
            xdim = warp.nx_grid;
        }
        for iy in 0..warp.ny_grid {
            for ix in 0..warp.nx_grid {
                *dx_grid.add((ix + iy * xdim) as usize) =
                    warp.x_vector[(ix + iy * warp.nx_grid) as usize];
                *dy_grid.add((ix + iy * xdim) as usize) =
                    warp.y_vector[(ix + iy * warp.nx_grid) as usize];
            }
        }
        0
    }
}

/// Original `separateLinearTransform` (`warpfiles.c:1261`).
pub unsafe fn separate_linear_transform(iz: i32) -> i32 {
    unsafe {
        if S_CUR_FILE_IND < 0 {
            return 1;
        }
        let file = &mut S_WARP_FILES.as_mut().unwrap()[S_CUR_FILE_IND as usize];
        if iz < 0 || iz >= file.num_frames || file.flags & WARP_INVERSE == 0 {
            return 1;
        }
        let control = file.flags & WARP_CONTROL_PTS != 0;
        let (mut x_point, mut y_point) = {
            let warp = &file.warpings[iz as usize];
            if control {
                if warp.n_control < 3 {
                    return 1;
                }
                (warp.x_control.clone(), warp.y_control.clone())
            } else {
                let n = warp.nx_grid * warp.ny_grid;
                if n < 3 {
                    return 1;
                }
                let mut xp = Vec::with_capacity(n as usize);
                let mut yp = Vec::with_capacity(n as usize);
                for iy in 0..warp.ny_grid {
                    for ix in 0..warp.nx_grid {
                        xp.push(warp.x_start + ix as f32 * warp.x_interval);
                        yp.push(warp.y_start + iy as f32 * warp.y_interval);
                    }
                }
                (xp, yp)
            }
        };
        let warp = &mut file.warpings[iz as usize];
        let mut xfinv = [0.; 6];
        let err = extract_linear_xform(
            x_point.as_mut_ptr(),
            y_point.as_mut_ptr(),
            warp.x_vector.as_mut_ptr(),
            warp.y_vector.as_mut_ptr(),
            x_point.len() as i32,
            file.nx as f32 / 2.,
            file.ny as f32 / 2.,
            warp.x_vector.as_mut_ptr(),
            warp.y_vector.as_mut_ptr(),
            xfinv.as_mut_ptr(),
            2,
        );
        if err == 0 {
            let mut product = [0.; 6];
            xf_mult(&warp.xform, &xfinv, &mut product, 2);
            warp.xform = product;
        }
        err
    }
}

/// Original static `readLineOfValues` (`warpfiles.c:1330`).
pub fn read_line_of_values(line: &str, values: &mut [f32], num_to_get: &mut i32) -> i32 {
    let parsed: Result<Vec<f32>, _> = line
        .split(|c: char| c == ',' || c.is_whitespace())
        .filter(|s| !s.is_empty())
        .map(str::parse)
        .collect();
    let Ok(parsed) = parsed else {
        return -4;
    };
    if parsed.is_empty() {
        return -2;
    }
    if parsed.len() > values.len() {
        return -3;
    }
    if *num_to_get == 0 {
        *num_to_get = parsed.len() as i32;
    }
    if parsed.len() < *num_to_get as usize {
        return -5;
    }
    values[..parsed.len()].copy_from_slice(&parsed);
    0
}

/// Original `readWarpFile` (`warpfiles.c:172`).
pub unsafe fn read_warp_file(
    filename: *mut i8,
    nx: *mut i32,
    ny: *mut i32,
    nz: *mut i32,
    binning: *mut i32,
    pixel_size: *mut f32,
    version: *mut i32,
    flags: *mut i32,
) -> i32 {
    unsafe {
        *version = -1;
        let path = CStr::from_ptr(filename).to_string_lossy();
        let Ok(contents) = fs::read_to_string(path.as_ref()) else {
            return -1;
        };
        let mut lines = contents.lines().filter(|line| !line.trim().is_empty());
        let Some(first) = lines.next() else {
            *version = 0;
            return -3;
        };
        let mut first_values = [0.; 50];
        let mut count = 0;
        let first_error = read_line_of_values(first, &mut first_values, &mut count);
        if first_error != 0 {
            return -2;
        }
        if count == 6 {
            *version = 0;
        }
        if count != 1 {
            return -3;
        }
        *version = first_values[0].round() as i32;
        if *version < 1 || *version > 3 {
            return -4;
        }
        let Some(header) = lines.next() else {
            return -4;
        };
        let mut values = [0.; 50];
        count = 0;
        if read_line_of_values(header, &mut values, &mut count) != 0 || count != *version + 3 {
            return -4;
        }
        let mut at = 0;
        *nx = values[at].round() as i32;
        at += 1;
        *ny = values[at].round() as i32;
        at += 1;
        *nz = if *version > 1 {
            let value = values[at].round() as i32;
            at += 1;
            value
        } else {
            1
        };
        *binning = values[at].round() as i32;
        at += 1;
        *pixel_size = values[at];
        at += 1;
        *flags = if *version == 3 {
            values[at].round() as i32
        } else {
            WARP_INVERSE
        };
        if *nx <= 0 || *ny <= 0 || *nz <= 0 || *binning <= 0 {
            return -5;
        }
        let file_ind = new_warp_file(*nx, *ny, *binning, *pixel_size, *flags);
        if file_ind < 0 {
            return -6;
        }
        for iz in 0..*nz {
            if add_warpings_if_needed(iz) != 0 {
                return -6;
            }
            let Some(line) = lines.next() else {
                clear_warp_file(file_ind);
                return -7;
            };
            count = 0;
            if read_line_of_values(line, &mut values, &mut count) != 0 {
                clear_warp_file(file_ind);
                return -7;
            }
            let warp = &mut S_WARP_FILES.as_mut().unwrap()[file_ind as usize].warpings[iz as usize];
            if *flags & WARP_CONTROL_PTS == 0 {
                if count < 6 {
                    return -7;
                }
                warp.x_start = values[0];
                warp.x_interval = values[1];
                warp.nx_grid = values[2].round() as i32;
                warp.y_start = values[3];
                warp.y_interval = values[4];
                warp.ny_grid = values[5].round() as i32;
                warp.x_vector = Vec::with_capacity((warp.nx_grid * warp.ny_grid) as usize);
                warp.y_vector = Vec::with_capacity((warp.nx_grid * warp.ny_grid) as usize);
            } else {
                warp.n_control = values[0].round() as i32;
                warp.x_vector = Vec::with_capacity(warp.n_control as usize);
                warp.y_vector = Vec::with_capacity(warp.n_control as usize);
                warp.x_control = Vec::with_capacity(warp.n_control as usize);
                warp.y_control = Vec::with_capacity(warp.n_control as usize);
            }
            if *version > 2 {
                let Some(line) = lines.next() else {
                    return -7;
                };
                count = 0;
                if read_line_of_values(line, &mut values, &mut count) != 0 || count != 6 {
                    return -7;
                }
                warp.xform = [
                    values[0], values[2], values[1], values[3], values[4], values[5],
                ];
            }
            if *flags & WARP_CONTROL_PTS == 0 {
                while warp.x_vector.len() < (warp.nx_grid * warp.ny_grid) as usize {
                    let Some(line) = lines.next() else {
                        return -7;
                    };
                    count = 0;
                    if read_line_of_values(line, &mut values, &mut count) != 0 || count % 2 != 0 {
                        return -7;
                    }
                    for i in (0..count as usize).step_by(2) {
                        warp.x_vector.push(values[i]);
                        warp.y_vector.push(values[i + 1]);
                    }
                }
            } else {
                for _ in 0..warp.n_control {
                    let Some(line) = lines.next() else {
                        return -7;
                    };
                    count = 4;
                    if read_line_of_values(line, &mut values, &mut count) != 0 {
                        return -7;
                    }
                    warp.x_control.push(values[0]);
                    warp.y_control.push(values[1]);
                    warp.x_vector.push(values[2]);
                    warp.y_vector.push(values[3]);
                }
            }
            warp.max_vectors = warp.x_vector.len() as i32;
        }
        file_ind
    }
}

/// Original `writeWarpFile` (`warpfiles.c:353`).
pub unsafe fn write_warp_file(filename: *const i8, _skip_backup: i32) -> i32 {
    unsafe {
        if S_CUR_FILE_IND < 0 {
            return -1;
        }
        let path = CStr::from_ptr(filename).to_string_lossy();
        let file = &S_WARP_FILES.as_ref().unwrap()[S_CUR_FILE_IND as usize];
        let mut out = format!(
            "3\n{} {} {} {} {} {}\n",
            file.nx, file.ny, file.num_frames, file.binning, file.pixel_size, file.flags
        );
        for warp in &file.warpings {
            if file.flags & WARP_CONTROL_PTS != 0 {
                out.push_str(&format!("{}\n", warp.n_control));
            } else {
                out.push_str(&format!(
                    "{}  {}  {}  {}  {}  {}\n",
                    warp.x_start,
                    warp.x_interval,
                    warp.nx_grid,
                    warp.y_start,
                    warp.y_interval,
                    warp.ny_grid
                ));
            }
            out.push_str(&format!(
                "{:.6}  {:.6}  {:.6}  {:.6}  {:.3} {:.3}\n",
                warp.xform[0],
                warp.xform[2],
                warp.xform[1],
                warp.xform[3],
                warp.xform[4],
                warp.xform[5]
            ));
            if file.flags & WARP_CONTROL_PTS != 0 {
                for i in 0..warp.n_control as usize {
                    out.push_str(&format!(
                        "{:.3} {:.3} {:.3} {:.3}\n",
                        warp.x_control[i], warp.y_control[i], warp.x_vector[i], warp.y_vector[i]
                    ));
                }
            } else {
                for j in 0..warp.ny_grid {
                    for i in 0..warp.nx_grid {
                        out.push_str(&format!(
                            "  {:.3}  {:.3}",
                            warp.x_vector[(i + j * warp.nx_grid) as usize],
                            warp.y_vector[(i + j * warp.nx_grid) as usize]
                        ));
                        if i % 4 == 3 || i == warp.nx_grid - 1 {
                            out.push('\n');
                        }
                    }
                }
            }
        }
        if fs::write(path.as_ref(), out).is_err() {
            -1
        } else {
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_file_lifecycle_preserves_packed_source_vectors() {
        unsafe {
            warp_files_done();
            let index = new_warp_file(8, 6, 2, 1.5, WARP_INVERSE);
            assert_eq!(index, 0);
            let mut dx = [1., 2., 3., 4.];
            let mut dy = [-1., -2., -3., -4.];
            assert_eq!(
                set_warp_grid(
                    0,
                    2,
                    2,
                    0.5,
                    1.5,
                    4.,
                    3.,
                    dx.as_mut_ptr(),
                    dy.as_mut_ptr(),
                    2
                ),
                0
            );
            let (mut nx, mut ny, mut nz, mut control) = (0, 0, 0, 0);
            assert_eq!(
                get_warp_file_size(&mut nx, &mut ny, &mut nz, &mut control),
                0
            );
            assert_eq!((nx, ny, nz, control), (8, 6, 1, 0));
            let (mut xs, mut ys, mut xi, mut yi) = (0., 0., 0., 0.);
            let mut outx = [0.; 4];
            let mut outy = [0.; 4];
            assert_eq!(
                get_warp_grid(
                    0,
                    &mut nx,
                    &mut ny,
                    &mut xs,
                    &mut ys,
                    &mut xi,
                    &mut yi,
                    outx.as_mut_ptr(),
                    outy.as_mut_ptr(),
                    2
                ),
                0
            );
            assert_eq!((nx, ny, xs, ys, xi, yi), (2, 2, 0.5, 1.5, 4., 3.));
            assert_eq!((outx, outy), (dx, dy));
            warp_files_done();
        }
    }

    #[test]
    fn version_three_warp_file_round_trips_source_text_format() {
        unsafe {
            warp_files_done();
            new_warp_file(8, 6, 2, 1.5, WARP_INVERSE);
            let mut dx = [1., 2., 3., 4.];
            let mut dy = [-1., -2., -3., -4.];
            set_warp_grid(
                0,
                2,
                2,
                0.5,
                1.5,
                4.,
                3.,
                dx.as_mut_ptr(),
                dy.as_mut_ptr(),
                2,
            );
            let path = std::env::temp_dir().join("imod_rs_warpfiles_roundtrip.xf");
            let name = CString::new(path.to_string_lossy().as_bytes()).unwrap();
            assert_eq!(write_warp_file(name.as_ptr(), 1), 0);
            warp_files_done();
            let (mut nx, mut ny, mut nz, mut bin, mut pixel, mut version, mut flags) =
                (0, 0, 0, 0, 0., 0, 0);
            assert_eq!(
                read_warp_file(
                    name.as_ptr() as *mut i8,
                    &mut nx,
                    &mut ny,
                    &mut nz,
                    &mut bin,
                    &mut pixel,
                    &mut version,
                    &mut flags
                ),
                0
            );
            assert_eq!(
                (nx, ny, nz, bin, pixel, version, flags),
                (8, 6, 1, 2, 1.5, 3, WARP_INVERSE)
            );
            let (mut xs, mut ys, mut xi, mut yi) = (0., 0., 0., 0.);
            let (mut ox, mut oy) = ([0.; 4], [0.; 4]);
            assert_eq!(
                get_warp_grid(
                    0,
                    &mut nx,
                    &mut ny,
                    &mut xs,
                    &mut ys,
                    &mut xi,
                    &mut yi,
                    ox.as_mut_ptr(),
                    oy.as_mut_ptr(),
                    2
                ),
                0
            );
            assert_eq!((ox, oy), (dx, dy));
            let _ = std::fs::remove_file(path);
            warp_files_done();
        }
    }

    #[test]
    fn control_points_produce_a_natural_neighbour_warp_grid() {
        unsafe {
            warp_files_done();
            new_warp_file(2, 2, 1, 1.0, WARP_CONTROL_PTS);
            let mut xc = [0.0, 1.0, 0.0];
            let mut yc = [0.0, 0.0, 1.0];
            let mut dx = [0.0, 1.0, 2.0];
            let mut dy = [3.0, 3.0, 3.0];
            assert_eq!(
                set_warp_points(
                    0,
                    3,
                    xc.as_mut_ptr(),
                    yc.as_mut_ptr(),
                    dx.as_mut_ptr(),
                    dy.as_mut_ptr()
                ),
                0
            );
            assert_eq!(set_grid_size_to_make(0, 2, 2, 0.0, 0.0, 1.0, 1.0), 0);
            let (mut nx, mut ny) = (0, 0);
            let (mut xs, mut ys, mut xi, mut yi) = (0.0, 0.0, 0.0, 0.0);
            let mut outx = [f32::NAN; 4];
            let mut outy = [f32::NAN; 4];
            assert_eq!(
                get_warp_grid(
                    0,
                    &mut nx,
                    &mut ny,
                    &mut xs,
                    &mut ys,
                    &mut xi,
                    &mut yi,
                    outx.as_mut_ptr(),
                    outy.as_mut_ptr(),
                    2
                ),
                0
            );
            assert_eq!((nx, ny, xs, ys, xi, yi), (2, 2, 0.0, 0.0, 1.0, 1.0));
            assert_eq!(&outx[..3], &[0.0, 1.0, 2.0]);
            assert_eq!(&outy[..3], &[3.0, 3.0, 3.0]);
            let first = outx;
            assert_eq!(
                get_warp_grid(
                    0,
                    &mut nx,
                    &mut ny,
                    &mut xs,
                    &mut ys,
                    &mut xi,
                    &mut yi,
                    outx.as_mut_ptr(),
                    outy.as_mut_ptr(),
                    2
                ),
                0
            );
            assert_eq!(outx, first);
            xc[1] = 1.2;
            assert_eq!(
                set_warp_points(
                    0,
                    3,
                    xc.as_mut_ptr(),
                    yc.as_mut_ptr(),
                    dx.as_mut_ptr(),
                    dy.as_mut_ptr()
                ),
                0
            );
            assert_eq!(
                points_match_stat_array(
                    &S_WARP_FILES.as_ref().unwrap()[S_CUR_FILE_IND as usize].warpings[0]
                ),
                0
            );
            assert_eq!(
                get_warp_grid(
                    0,
                    &mut nx,
                    &mut ny,
                    &mut xs,
                    &mut ys,
                    &mut xi,
                    &mut yi,
                    outx.as_mut_ptr(),
                    outy.as_mut_ptr(),
                    2
                ),
                0
            );
            warp_files_done();
        }
    }
}
