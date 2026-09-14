//! Translation of `IMOD/libiimod/sliceproc.c` and `include/sliceproc.h`.
#![allow(dead_code, unused_variables, static_mut_refs)]

use crate::imod::libcfshr::islice::{
    Islice, Istack, slice_create, slice_get_val, slice_put_val, slice_scale_and_free,
};
use crate::imod::libcfshr::percentile::{percentile_float, percentile_int};
use crate::imod::libiimod::diffusion::update_matrix;
use crate::imod::libiimod::mrcslice::slice_new_mode;

pub const ANISO_CLEAR_AT_END: i32 = 0;
pub const ANISO_CLEAR_ONLY: i32 = 1;
pub const ANISO_LEAVE_OPEN: i32 = 2;
const SLICE_MODE_BYTE: i32 = 0;
const SLICE_MODE_SHORT: i32 = 1;
const SLICE_MODE_FLOAT: i32 = 2;
const SLICE_MODE_USHORT: i32 = 6;

pub unsafe fn slice_byte_add(sin: *mut Islice, in_val: i32) -> i32 {
    unsafe {
        let imax = (*sin).xsize * (*sin).ysize;
        let image = (*sin).data.b;
        for i in 0..imax {
            let mut aval = *image.add(i as usize) as i32 + in_val;
            if aval > 255 {
                aval = 255;
            }
            if aval < 0 {
                aval = 0;
            }
            *image.add(i as usize) = aval as u8;
        }
    }
    0
}

pub unsafe fn slice_byte_edge_two(sin: *mut Islice, center: i32) -> i32 {
    unsafe {
        let sout = slice_create((*sin).xsize, (*sin).ysize, SLICE_MODE_FLOAT);
        let imax = (*sin).xsize - 1;
        let jmax = (*sin).ysize - 1;
        for i in 1..imax {
            for j in 1..jmax {
                let get =
                    |x: i32, y: i32| *(*sin).data.b.add((x + y * (*sin).xsize) as usize) as f32;
                let mut sr = get(i - 1, j - 1) + center as f32 * get(i, j - 1) + get(i + 1, j - 1)
                    - get(i - 1, j + 1)
                    - center as f32 * get(i, j + 1)
                    - get(i + 1, j + 1);
                let mut sc = get(i + 1, j + 1) + center as f32 * get(i + 1, j) + get(i + 1, j - 1)
                    - get(i - 1, j + 1)
                    - center as f32 * get(i - 1, j)
                    - get(i - 1, j - 1);
                sr *= sr;
                sc *= sc;
                *(*sout).data.f.add((i + j * (*sout).xsize) as usize) = (sr + sc).sqrt();
            }
        }
        for j in 1..jmax {
            *(*sout).data.f.add((j * (*sout).xsize) as usize) =
                *(*sout).data.f.add((1 + j * (*sout).xsize) as usize);
            *(*sout).data.f.add((imax + j * (*sout).xsize) as usize) =
                *(*sout).data.f.add((imax - 1 + j * (*sout).xsize) as usize);
        }
        for i in 0..=imax {
            *(*sout).data.f.add(i as usize) = *(*sout).data.f.add((i + (*sout).xsize) as usize);
            *(*sout).data.f.add((i + jmax * (*sout).xsize) as usize) = *(*sout)
                .data
                .f
                .add((i + (jmax - 1) * (*sout).xsize) as usize);
        }
        slice_scale_and_free(sout, sin);
    }
    0
}

pub unsafe fn slice_byte_edge_sobel(sin: *mut Islice) -> i32 {
    unsafe { slice_byte_edge_two(sin, 2) }
}
pub unsafe fn slice_byte_edge_prewitt(sin: *mut Islice) -> i32 {
    unsafe { slice_byte_edge_two(sin, 1) }
}

unsafe fn nay8(sin: *mut Islice, i: i32, j: i32, val: i32) -> i32 {
    unsafe {
        if *(*sin).data.b.add((i + j * (*sin).xsize) as usize) as i32 != val {
            return 0;
        }
        let mut k = 0;
        for n in -1..=1 {
            for m in -1..=1 {
                let y = n + j;
                let x = m + i;
                if x > 0
                    && y > 0
                    && x < (*sin).xsize
                    && y < (*sin).ysize
                    && *(*sin).data.b.add((x + y * (*sin).xsize) as usize) as i32 == val
                {
                    k += 1;
                }
            }
        }
        k - 1
    }
}

pub unsafe fn slice_byte_threshold(sin: *mut Islice, val: i32) -> i32 {
    unsafe {
        let image = (*sin).data.b;
        let pmin = (*sin).min as i32;
        let pmax = (*sin).max as i32;
        let thresh = pmin + val;
        for i in 0..(*sin).xsize * (*sin).ysize {
            *image.add(i as usize) = if (*image.add(i as usize) as i32) < thresh {
                pmax as u8
            } else {
                pmin as u8
            };
        }
    }
    0
}

pub unsafe fn slice_byte_grow(sin: *mut Islice, val: i32) -> i32 {
    unsafe {
        for j in 0..(*sin).ysize {
            for i in 0..(*sin).xsize {
                if *(*sin).data.b.add((i + j * (*sin).xsize) as usize) as i32 != val {
                    continue;
                }
                for m in -1..=1 {
                    let y = j + m;
                    if y < 0 || y >= (*sin).ysize {
                        continue;
                    }
                    for n in -1..=1 {
                        let x = i + n;
                        if x == i && y == j || x < 0 || x >= (*sin).xsize {
                            continue;
                        }
                        let at = (*sin).data.b.add((x + y * (*sin).xsize) as usize);
                        if *at as f32 == (*sin).min {
                            *at = ((*sin).max - 1.) as u8;
                        }
                    }
                }
            }
        }
        for j in 0..(*sin).ysize {
            for i in 0..(*sin).xsize {
                let at = (*sin).data.b.add((i + j * (*sin).xsize) as usize);
                if *at as f32 == (*sin).max - 1. {
                    *at = val as u8;
                }
            }
        }
    }
    0
}

pub unsafe fn slice_byte_shrink(sin: *mut Islice, _val: i32) -> i32 {
    unsafe {
        let sout = slice_create((*sin).xsize, (*sin).ysize, SLICE_MODE_BYTE);
        let pmin = (*sin).min as u8;
        let pmax = (*sin).max as i32;
        for j in 0..(*sin).ysize {
            for i in 0..(*sin).xsize {
                *(*sout).data.b.add((i + j * (*sout).xsize) as usize) = 0;
            }
        }
        for j in 0..(*sin).ysize {
            for i in 0..(*sin).xsize {
                if nay8(sin, i, j, pmax) < 7 {
                    *(*sout).data.b.add((i + j * (*sout).xsize) as usize) = 1;
                }
            }
        }
        for j in 0..(*sin).ysize {
            for i in 0..(*sin).xsize {
                if *(*sout).data.b.add((i + j * (*sout).xsize) as usize) != 0 {
                    *(*sin).data.b.add((i + j * (*sin).xsize) as usize) = pmin;
                }
            }
        }
        // The C source leaks `sout`; retain that behavior.
    }
    0
}

pub unsafe fn slice_byte_graham(sin: *mut Islice) -> i32 {
    unsafe {
        let sout = slice_create((*sin).xsize, (*sin).ysize, SLICE_MODE_FLOAT);
        let imax = (*sin).xsize - 1;
        let jmax = (*sin).ysize - 1;
        let ld = 1. / 6.;
        let hd = 1. / 3.;
        let delta = 5.;
        for i in 1..imax {
            for j in 1..jmax {
                let g = |x: i32, y: i32| *(*sin).data.b.add((x + y * (*sin).xsize) as usize) as f32;
                let ixx = g(i + 1, j + 1) * ld - g(i, j + 1) * hd
                    + g(i - 1, j + 1) * ld
                    + g(i + 1, j) * ld
                    - g(i, j) * hd
                    + g(i - 1, j) * ld
                    + g(i + 1, j - 1) * ld
                    - g(i, j - 1) * hd
                    + g(i - 1, j - 1) * ld;
                let iyy = g(i + 1, j + 1) * ld + g(i, j + 1) * ld + g(i - 1, j + 1) * ld
                    - g(i + 1, j) * hd
                    - g(i, j) * hd
                    - g(i - 1, j) * hd
                    + g(i + 1, j - 1) * ld
                    + g(i, j - 1) * ld
                    + g(i - 1, j - 1) * ld;
                let out = if ixx < delta {
                    if iyy < delta {
                        (g(i + 1, j + 1)
                            + g(i, j + 1)
                            + g(i - 1, j + 1)
                            + g(i + 1, j)
                            + g(i, j)
                            + g(i - 1, j)
                            + g(i + 1, j - 1)
                            + g(i, j - 1)
                            + g(i - 1, j - 1))
                            / 9.
                    } else {
                        (g(i + 1, j) + g(i, j) + g(i - 1, j)) / 3.
                    }
                } else if iyy < delta {
                    (g(i, j + 1) + g(i, j) + g(1, j - 1)) / 3.
                } else {
                    g(i, j)
                };
                *(*sout).data.f.add((i + j * (*sout).xsize) as usize) = out;
            }
        }
        for j in 1..jmax {
            *(*sout).data.f.add((j * (*sout).xsize) as usize) =
                *(*sout).data.f.add((1 + j * (*sout).xsize) as usize);
            *(*sout).data.f.add((imax + j * (*sout).xsize) as usize) =
                *(*sout).data.f.add((imax - 1 + j * (*sout).xsize) as usize);
        }
        for i in 0..=imax {
            *(*sout).data.f.add(i as usize) = *(*sout).data.f.add((i + (*sout).xsize) as usize);
            *(*sout).data.f.add((i + jmax * (*sout).xsize) as usize) = *(*sout)
                .data
                .f
                .add((i + (jmax - 1) * (*sout).xsize) as usize);
        }
        slice_scale_and_free(sout, sin);
    }
    0
}

unsafe fn opt_med9(p: *mut f32) -> f32 {
    unsafe {
        macro_rules! pix_sort {
            ($a:expr, $b:expr) => {
                if *p.add($a) > *p.add($b) {
                    let ftemp = *p.add($a);
                    *p.add($a) = *p.add($b);
                    *p.add($b) = ftemp;
                }
            };
        }
        pix_sort!(1, 2);
        pix_sort!(4, 5);
        pix_sort!(7, 8);
        pix_sort!(0, 1);
        pix_sort!(3, 4);
        pix_sort!(6, 7);
        pix_sort!(1, 2);
        pix_sort!(4, 5);
        pix_sort!(7, 8);
        pix_sort!(0, 3);
        pix_sort!(5, 8);
        pix_sort!(4, 7);
        pix_sort!(3, 6);
        pix_sort!(1, 4);
        pix_sort!(2, 5);
        pix_sort!(4, 7);
        pix_sort!(4, 2);
        pix_sort!(6, 4);
        pix_sort!(4, 2);
        *p.add(4)
    }
}

pub unsafe fn slice_median_filter(sl_out: *mut Islice, stack: *mut Istack, size: i32) -> i32 {
    unsafe {
        let sl_in = *(*stack).vol;
        let block_size = (*stack).zsize * size * size;
        if (*stack).zsize == 1
            && size == 3
            && (*sl_in).mode == SLICE_MODE_FLOAT
            && (*sl_out).mode == SLICE_MODE_FLOAT
        {
            // This is the source's optimized 2-D float route.  The C OpenMP
            // scheduling does not affect the independent output rows, so one
            // source-shaped worker preserves its data and buffer semantics.
            let line_vals =
                libc::malloc((size * (*sl_in).xsize) as usize * core::mem::size_of::<f32>())
                    .cast::<f32>();
            let f_vals =
                libc::malloc(block_size as usize * core::mem::size_of::<f32>()).cast::<f32>();
            if line_vals.is_null() || f_vals.is_null() {
                libc::free(line_vals.cast());
                libc::free(f_vals.cast());
                return -1;
            }
            let mut initial_loaded = 0;
            let mut offset = 0;
            for oy in 0..(*sl_in).ysize {
                if initial_loaded == 0 {
                    let line1 = (*sl_in)
                        .data
                        .f
                        .add((oy - 1).max(0) as usize * (*sl_in).xsize as usize);
                    let line2 = (*sl_in).data.f.add(oy as usize * (*sl_in).xsize as usize);
                    let line3 = (*sl_in)
                        .data
                        .f
                        .add((oy + 1).min((*sl_in).ysize - 1) as usize * (*sl_in).xsize as usize);
                    for ox in 0..(*sl_in).xsize {
                        *line_vals.add((3 * ox) as usize) = *line1.add(ox as usize);
                        *line_vals.add((3 * ox + 1) as usize) = *line2.add(ox as usize);
                        *line_vals.add((3 * ox + 2) as usize) = *line3.add(ox as usize);
                    }
                    initial_loaded = 1;
                } else {
                    let line3 = (*sl_in)
                        .data
                        .f
                        .add((oy + 1).min((*sl_in).ysize - 1) as usize * (*sl_in).xsize as usize);
                    for ox in 0..(*sl_in).xsize {
                        *line_vals.add((offset + 3 * ox) as usize) = *line3.add(ox as usize);
                    }
                    offset = (offset + 1) % 3;
                }
                for ox in 1..(*sl_in).xsize - 1 {
                    core::ptr::copy_nonoverlapping(
                        line_vals.add((3 * (ox - 1)) as usize),
                        f_vals,
                        9,
                    );
                    *(*sl_out).data.f.add((ox + oy * (*sl_out).xsize) as usize) = opt_med9(f_vals);
                }
                core::ptr::copy_nonoverlapping(line_vals, f_vals, 9);
                *(*sl_out).data.f.add((oy * (*sl_out).xsize) as usize) = opt_med9(f_vals);
                core::ptr::copy_nonoverlapping(
                    line_vals.add((3 * ((*sl_in).xsize - 3)) as usize),
                    f_vals,
                    9,
                );
                *(*sl_out)
                    .data
                    .f
                    .add(((*sl_in).xsize - 1 + oy * (*sl_out).xsize) as usize) = opt_med9(f_vals);
            }
            libc::free(line_vals.cast());
            libc::free(f_vals.cast());
            return 0;
        }

        let mut f_vals = core::ptr::null_mut::<f32>();
        let mut i_vals = core::ptr::null_mut::<i32>();
        let is_float = if (*sl_in).mode == SLICE_MODE_FLOAT {
            f_vals = libc::malloc(block_size as usize * core::mem::size_of::<f32>()).cast();
            if f_vals.is_null() {
                return -1;
            }
            true
        } else if matches!(
            (*sl_in).mode,
            SLICE_MODE_BYTE | SLICE_MODE_SHORT | SLICE_MODE_USHORT
        ) {
            i_vals = libc::malloc(block_size as usize * core::mem::size_of::<i32>()).cast();
            if i_vals.is_null() {
                return -1;
            }
            false
        } else {
            return -2;
        };
        let del_minus = size / 2;
        let del_plus = (size + 1) / 2;
        for oy in 0..(*sl_in).ysize {
            let y_start = (oy - del_minus).max(0);
            let y_end = (oy + del_plus).min((*sl_in).ysize);
            for ox in 0..(*sl_in).xsize {
                let x_start = (ox - del_minus).max(0);
                let x_end = (ox + del_plus).min((*sl_in).xsize);
                let num_vals = (*stack).zsize * (x_end - x_start) * (y_end - y_start);
                let select = (num_vals + 1) / 2;
                let mut value = 0.0;
                for oz in 0..(*stack).zsize {
                    let sl = *(*stack).vol.add(oz as usize);
                    for iy in y_start..y_end {
                        for ix in x_start..x_end {
                            let index =
                                (ix - x_start + (iy - y_start) * (x_end - x_start)) as usize;
                            let index =
                                index + (oz * (x_end - x_start) * (y_end - y_start)) as usize;
                            if is_float {
                                *f_vals.add(index) = *(*sl)
                                    .data
                                    .f
                                    .add((x_start + (ix - x_start) + iy * (*sl_in).xsize) as usize);
                            } else if (*sl_in).mode == SLICE_MODE_SHORT {
                                *i_vals.add(index) =
                                    *(*sl).data.s.add(
                                        (x_start + (ix - x_start) + iy * (*sl_in).xsize) as usize,
                                    ) as i32;
                            } else if (*sl_in).mode == SLICE_MODE_USHORT {
                                *i_vals.add(index) =
                                    *(*sl).data.us.add(
                                        (x_start + (ix - x_start) + iy * (*sl_in).xsize) as usize,
                                    ) as i32;
                            } else {
                                *i_vals.add(index) =
                                    *(*sl).data.b.add(
                                        (x_start + (ix - x_start) + iy * (*sl_in).xsize) as usize,
                                    ) as i32;
                            }
                        }
                    }
                }
                if is_float {
                    let f_vals = core::slice::from_raw_parts_mut(f_vals, num_vals.max(0) as usize);
                    value = percentile_float(select, f_vals, num_vals);
                    if num_vals % 2 == 0 {
                        value = 0.5 * (value + percentile_float(select + 1, f_vals, num_vals));
                    }
                } else {
                    let i_vals = core::slice::from_raw_parts_mut(i_vals, num_vals.max(0) as usize);
                    value = percentile_int(select, i_vals, num_vals) as f32;
                    if num_vals % 2 == 0 {
                        value = 0.5 * (value + percentile_int(select + 1, i_vals, num_vals) as f32);
                    }
                }
                slice_put_val(sl_out, ox, oy, [value, 0., 0., 0.]);
            }
        }
        libc::free(if is_float {
            f_vals.cast()
        } else {
            i_vals.cast()
        });
    }
    0
}

static mut ANISO_IMAGE: *mut *mut f32 = core::ptr::null_mut();
static mut ANISO_IMAGE2: *mut *mut f32 = core::ptr::null_mut();
static mut ANISO_IMOUT: *mut *mut f32 = core::ptr::null_mut();
static mut ANISO_ITER_DONE: i32 = 0;

pub unsafe fn slice_aniso_diff(
    sl: *mut Islice,
    out_mode: i32,
    cc: i32,
    k: f64,
    lambda: f64,
    iterations: i32,
    clear_flag: i32,
) -> i32 {
    unsafe {
        if clear_flag == ANISO_CLEAR_ONLY {
            if ANISO_ITER_DONE != 0 {
                libc::free((*ANISO_IMAGE).cast());
                libc::free(ANISO_IMAGE.cast());
                libc::free((*ANISO_IMAGE2).cast());
                libc::free(ANISO_IMAGE2.cast());
                ANISO_ITER_DONE = 0;
            }
            return 0;
        }
        if (*sl).mode != SLICE_MODE_FLOAT && slice_new_mode(sl, SLICE_MODE_FLOAT) < 0 {
            return -1;
        }
        let n = (*sl).xsize;
        let m = (*sl).ysize;
        if ANISO_ITER_DONE == 0 {
            ANISO_IMAGE = allocate_2d_float(m + 2, n + 2);
            if ANISO_IMAGE.is_null() {
                return -1;
            }
            ANISO_IMAGE2 = allocate_2d_float(m + 2, n + 2);
            if ANISO_IMAGE2.is_null() {
                libc::free((*ANISO_IMAGE).cast());
                libc::free(ANISO_IMAGE.cast());
                return -1;
            }
            for j in 0..m {
                for i in 0..n {
                    *(*ANISO_IMAGE.add((j + 1) as usize)).add((i + 1) as usize) =
                        *(*sl).data.f.add((i + j * (*sl).xsize) as usize);
                }
            }
        }
        for _ in 0..iterations {
            if ANISO_ITER_DONE % 2 == 0 {
                update_matrix(ANISO_IMAGE2, ANISO_IMAGE, m, n, cc, k, lambda);
                ANISO_IMOUT = ANISO_IMAGE2;
            } else {
                update_matrix(ANISO_IMAGE, ANISO_IMAGE2, m, n, cc, k, lambda);
                ANISO_IMOUT = ANISO_IMAGE;
            }
            ANISO_ITER_DONE += 1;
        }
        for j in 0..m {
            for i in 0..n {
                *(*sl).data.f.add((i + j * (*sl).xsize) as usize) =
                    *(*ANISO_IMOUT.add((j + 1) as usize)).add((i + 1) as usize);
            }
        }
        if clear_flag == ANISO_CLEAR_AT_END {
            libc::free((*ANISO_IMAGE).cast());
            libc::free(ANISO_IMAGE.cast());
            libc::free((*ANISO_IMAGE2).cast());
            libc::free(ANISO_IMAGE2.cast());
            ANISO_ITER_DONE = 0;
        }
        if out_mode != SLICE_MODE_FLOAT && slice_new_mode(sl, out_mode) < 0 {
            return -1;
        }
    }
    0
}

pub unsafe fn slice_byte_aniso_diff(
    sl: *mut Islice,
    image: *mut *mut f32,
    image2: *mut *mut f32,
    cc: i32,
    k: f64,
    lambda: f64,
    iterations: i32,
    iter_done: *mut i32,
) {
    unsafe {
        let n = (*sl).xsize;
        let m = (*sl).ysize;
        if *iter_done == 0 {
            for j in 0..m {
                for i in 0..n {
                    *(*image.add((j + 1) as usize)).add((i + 1) as usize) =
                        *(*sl).data.b.add((i + j * (*sl).xsize) as usize) as f32;
                }
            }
        }
        let mut out = image;
        for _ in 0..iterations {
            if *iter_done % 2 == 0 {
                update_matrix(image2, image, m, n, cc, k, lambda);
                out = image2;
            } else {
                update_matrix(image, image2, m, n, cc, k, lambda);
                out = image;
            }
            *iter_done += 1;
        }
        for j in 0..m {
            for i in 0..n {
                let v = *(*out.add((j + 1) as usize)).add((i + 1) as usize) as i32;
                *(*sl).data.b.add((i + j * (*sl).xsize) as usize) = v.clamp(0, 255) as u8;
            }
        }
    }
}

pub unsafe fn allocate_2d_float(m: i32, n: i32) -> *mut *mut f32 {
    unsafe {
        let a = libc::malloc(core::mem::size_of::<*mut f32>() * m as usize).cast::<*mut f32>();
        if a.is_null() {
            return a;
        }
        let fake = libc::malloc((m * n) as usize * core::mem::size_of::<f32>()).cast::<f32>();
        if fake.is_null() {
            libc::free(a.cast());
            return core::ptr::null_mut();
        }
        for i in 0..m {
            *a.add(i as usize) = fake.add((i * n) as usize);
        }
        a
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::libcfshr::islice::slice_free;

    #[test]
    fn optimized_float_three_by_three_path_uses_clamped_source_neighborhoods() {
        unsafe {
            let input = slice_create(3, 3, SLICE_MODE_FLOAT);
            let output = slice_create(3, 3, SLICE_MODE_FLOAT);
            for (index, value) in (1..=9).enumerate() {
                *(*input).data.f.add(index) = value as f32;
            }
            let mut volumes = [input];
            let mut stack = Istack {
                vol: volumes.as_mut_ptr(),
                zsize: 1,
            };
            assert_eq!(slice_median_filter(output, &mut stack, 3), 0);
            assert_eq!(*(*output).data.f.add(4), 5.0);
            assert_eq!(*(*output).data.f.add(0), 3.0);
            assert_eq!(*(*output).data.f.add(8), 7.0);
            slice_free(input);
            slice_free(output);
        }
    }

    #[test]
    fn general_integer_and_three_dimensional_median_paths_keep_source_selection() {
        unsafe {
            let first = slice_create(3, 3, SLICE_MODE_BYTE);
            let second = slice_create(3, 3, SLICE_MODE_BYTE);
            let output = slice_create(3, 3, SLICE_MODE_FLOAT);
            for index in 0..9 {
                *(*first).data.b.add(index) = index as u8;
                *(*second).data.b.add(index) = (index + 10) as u8;
            }
            let mut volumes = [first, second];
            let mut stack = Istack {
                vol: volumes.as_mut_ptr(),
                zsize: 2,
            };
            assert_eq!(slice_median_filter(output, &mut stack, 3), 0);
            // At the center there are 18 values; source selection averages
            // the ninth and tenth one-indexed percentile values.
            assert_eq!(*(*output).data.f.add(4), 9.0);
            slice_free(first);
            slice_free(second);
            slice_free(output);
        }
    }

    #[test]
    fn median_filter_rejects_a_nonreal_input_mode_before_output_writes() {
        unsafe {
            let input = slice_create(2, 2, 3);
            let output = slice_create(2, 2, SLICE_MODE_FLOAT);
            let mut volumes = [input];
            let mut stack = Istack {
                vol: volumes.as_mut_ptr(),
                zsize: 1,
            };
            assert_eq!(slice_median_filter(output, &mut stack, 3), -2);
            slice_free(input);
            slice_free(output);
        }
    }
}
