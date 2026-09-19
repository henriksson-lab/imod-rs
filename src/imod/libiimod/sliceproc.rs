//! Translation of `IMOD/libiimod/sliceproc.c` and `include/sliceproc.h`.
#![allow(dead_code, unused_variables)]

use crate::imod::libcfshr::islice::{Islice, slice_create, slice_put_val, slice_scale_and_free};
use crate::imod::libcfshr::percentile::{percentile_float, percentile_int};
use crate::imod::libiimod::diffusion::update_matrix;
use crate::imod::libiimod::mrcslice::slice_new_mode;
use std::sync::Mutex;

pub const ANISO_CLEAR_AT_END: i32 = 0;
pub const ANISO_CLEAR_ONLY: i32 = 1;
pub const ANISO_LEAVE_OPEN: i32 = 2;
const SLICE_MODE_BYTE: i32 = 0;
const SLICE_MODE_SHORT: i32 = 1;
const SLICE_MODE_FLOAT: i32 = 2;
const SLICE_MODE_USHORT: i32 = 6;

pub fn slice_byte_add(sin: &mut Islice, in_val: i32) -> i32 {
    let imax = sin.xsize * sin.ysize;
    for pixel in &mut sin.data[..imax as usize] {
        let mut aval = *pixel as i32 + in_val;
        if aval > 255 {
            aval = 255;
        }
        if aval < 0 {
            aval = 0;
        }
        *pixel = aval as u8;
    }
    0
}

pub fn slice_byte_edge_two(sin: &mut Islice, center: i32) -> i32 {
    let mut sout = slice_create(sin.xsize, sin.ysize, SLICE_MODE_FLOAT)
        .expect("dimensions were validated by the input slice");
    let imax = sin.xsize - 1;
    let jmax = sin.ysize - 1;
    for i in 1..imax {
        for j in 1..jmax {
            let get = |x: i32, y: i32| sin.data[(x + y * sin.xsize) as usize] as f32;
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
            let offset = (i + j * sout.xsize) as usize * 4;
            sout.data[offset..offset + 4].copy_from_slice(&(sr + sc).sqrt().to_ne_bytes());
        }
    }
    for j in 1..jmax {
        let left = (j * sout.xsize) as usize * 4;
        let left_source = left + 4;
        let right = (imax + j * sout.xsize) as usize * 4;
        let right_source = right - 4;
        let left_value: [u8; 4] = sout.data[left_source..left_source + 4].try_into().unwrap();
        let right_value: [u8; 4] = sout.data[right_source..right_source + 4]
            .try_into()
            .unwrap();
        sout.data[left..left + 4].copy_from_slice(&left_value);
        sout.data[right..right + 4].copy_from_slice(&right_value);
    }
    for i in 0..=imax {
        let top = i as usize * 4;
        let top_source = top + sin.xsize as usize * 4;
        let bottom = (i + jmax * sout.xsize) as usize * 4;
        let bottom_source = bottom - sin.xsize as usize * 4;
        let top_value: [u8; 4] = sout.data[top_source..top_source + 4].try_into().unwrap();
        let bottom_value: [u8; 4] = sout.data[bottom_source..bottom_source + 4]
            .try_into()
            .unwrap();
        sout.data[top..top + 4].copy_from_slice(&top_value);
        sout.data[bottom..bottom + 4].copy_from_slice(&bottom_value);
    }
    slice_scale_and_free(sout.as_mut(), sin);
    0
}

pub fn slice_byte_edge_sobel(sin: &mut Islice) -> i32 {
    slice_byte_edge_two(sin, 2)
}
pub fn slice_byte_edge_prewitt(sin: &mut Islice) -> i32 {
    slice_byte_edge_two(sin, 1)
}

fn nay8(sin: &Islice, i: i32, j: i32, val: i32) -> i32 {
    if sin.data[(i + j * sin.xsize) as usize] as i32 != val {
        return 0;
    }
    let mut k = 0;
    for n in -1..=1 {
        for m in -1..=1 {
            let y = n + j;
            let x = m + i;
            if x > 0
                && y > 0
                && x < sin.xsize
                && y < sin.ysize
                && sin.data[(x + y * sin.xsize) as usize] as i32 == val
            {
                k += 1;
            }
        }
    }
    k - 1
}

pub fn slice_byte_threshold(sin: &mut Islice, val: i32) -> i32 {
    let pmin = sin.min as i32;
    let pmax = sin.max as i32;
    let thresh = pmin + val;
    for pixel in &mut sin.data {
        *pixel = if (*pixel as i32) < thresh {
            pmax as u8
        } else {
            pmin as u8
        };
    }
    0
}

pub fn slice_byte_grow(sin: &mut Islice, val: i32) -> i32 {
    for j in 0..sin.ysize {
        for i in 0..sin.xsize {
            if sin.data[(i + j * sin.xsize) as usize] as i32 != val {
                continue;
            }
            for m in -1..=1 {
                let y = j + m;
                if y < 0 || y >= sin.ysize {
                    continue;
                }
                for n in -1..=1 {
                    let x = i + n;
                    if x == i && y == j || x < 0 || x >= sin.xsize {
                        continue;
                    }
                    let at = (x + y * sin.xsize) as usize;
                    if sin.data[at] as f32 == sin.min {
                        sin.data[at] = (sin.max - 1.) as u8;
                    }
                }
            }
        }
    }
    for pixel in &mut sin.data {
        if *pixel as f32 == sin.max - 1. {
            *pixel = val as u8;
        }
    }
    0
}

pub fn slice_byte_shrink(sin: &mut Islice, _val: i32) -> i32 {
    let mut sout = slice_create(sin.xsize, sin.ysize, SLICE_MODE_BYTE)
        .expect("dimensions were validated by the input slice");
    let pmin = sin.min as u8;
    let pmax = sin.max as i32;
    for j in 0..sin.ysize {
        for i in 0..sin.xsize {
            if nay8(sin, i, j, pmax) < 7 {
                sout.data[(i + j * sout.xsize) as usize] = 1;
            }
        }
    }
    for j in 0..sin.ysize {
        for i in 0..sin.xsize {
            if sout.data[(i + j * sout.xsize) as usize] != 0 {
                sin.data[(i + j * sin.xsize) as usize] = pmin;
            }
        }
    }
    0
}

pub fn slice_byte_graham(sin: &mut Islice) -> i32 {
    let mut sout = slice_create(sin.xsize, sin.ysize, SLICE_MODE_FLOAT)
        .expect("dimensions were validated by the input slice");
    let imax = sin.xsize - 1;
    let jmax = sin.ysize - 1;
    let ld = 1. / 6.;
    let hd = 1. / 3.;
    let delta = 5.;
    for i in 1..imax {
        for j in 1..jmax {
            let g = |x: i32, y: i32| sin.data[(x + y * sin.xsize) as usize] as f32;
            let ixx =
                g(i + 1, j + 1) * ld - g(i, j + 1) * hd + g(i - 1, j + 1) * ld + g(i + 1, j) * ld
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
            let offset = (i + j * sout.xsize) as usize * 4;
            sout.data[offset..offset + 4].copy_from_slice(&out.to_ne_bytes());
        }
    }
    for j in 1..jmax {
        let left = (j * sout.xsize) as usize * 4;
        let left_source = left + 4;
        let right = (imax + j * sout.xsize) as usize * 4;
        let right_source = right - 4;
        let left_value: [u8; 4] = sout.data[left_source..left_source + 4].try_into().unwrap();
        let right_value: [u8; 4] = sout.data[right_source..right_source + 4]
            .try_into()
            .unwrap();
        sout.data[left..left + 4].copy_from_slice(&left_value);
        sout.data[right..right + 4].copy_from_slice(&right_value);
    }
    for i in 0..=imax {
        let top = i as usize * 4;
        let top_source = top + sin.xsize as usize * 4;
        let bottom = (i + jmax * sout.xsize) as usize * 4;
        let bottom_source = bottom - sin.xsize as usize * 4;
        let top_value: [u8; 4] = sout.data[top_source..top_source + 4].try_into().unwrap();
        let bottom_value: [u8; 4] = sout.data[bottom_source..bottom_source + 4]
            .try_into()
            .unwrap();
        sout.data[top..top + 4].copy_from_slice(&top_value);
        sout.data[bottom..bottom + 4].copy_from_slice(&bottom_value);
    }
    slice_scale_and_free(sout.as_mut(), sin);
    0
}

fn opt_med9(p: &mut [f32]) -> f32 {
    macro_rules! pix_sort {
        ($a:expr, $b:expr) => {
            if p[$a] > p[$b] {
                p.swap($a, $b);
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
    p[4]
}

pub fn slice_median_filter(sl_out: &mut Islice, stack: &[Islice], size: i32) -> i32 {
    let Some(sl_in) = stack.first() else {
        return -1;
    };
    if size <= 0
        || !matches!(
            sl_in.mode,
            SLICE_MODE_BYTE | SLICE_MODE_SHORT | SLICE_MODE_FLOAT | SLICE_MODE_USHORT
        )
    {
        return -2;
    }
    if stack.iter().any(|slice| {
        slice.xsize != sl_in.xsize || slice.ysize != sl_in.ysize || slice.mode != sl_in.mode
    }) {
        return -1;
    }

    /* `sliceproc.c:369-439`.  A 2-D 3x3 filter from float to float takes a
    separate, highly optimised path, and it is not merely faster: it produces
    *different* values at the two end columns.  The interior loop runs
    `ox = 1 .. xsize - 1` and the two ends are then filled from the **first** and
    **last complete** nine-value windows, so native's `x = 0` output equals its
    `x = 1` output and `x = xsize - 1` equals `x = xsize - 2`, where the general
    path below builds a clipped six-value window at each end.  `clip median
    -3 -n 3` on a 50x40 float image differed from native in 176 pixels across
    all 40 rows with this path missing.

    The source parallelises the row loop with OpenMP, keeping the three
    interleaved lines and the rotating `offset` private to each thread.  The
    result does not depend on the thread count, so one sequential pass
    reproduces it exactly: a thread's first row rebuilds the buffer from rows
    `oy - 1`, `oy`, `oy + 1`, every later row overwrites one of the three in
    rotation, and `opt_med9` sorts its copy — so each row always sees the same
    *set* of nine values however they are ordered within the triple. */
    if stack.len() == 1
        && size == 3
        && sl_in.mode == SLICE_MODE_FLOAT
        && sl_out.mode == SLICE_MODE_FLOAT
        // `lineVals` holds `3 * xsize` floats, so for a width under 3 both
        // endpoint copies run outside it: `memcpy(fVals, lineVals, 9 * 4)`
        // reads three floats past the end, and `lineVals + 3 * (xsize - 3)` is
        // `lineVals - 3` for width 2, reading three floats *before* it.  Native
        // is stable across runs there — the reads land on adjacent heap — but
        // reproducing them would mean matching glibc's allocator layout, so
        // fall through to the general path instead.  This is the documented
        // source-level-UB deviation, not a defect; `clip median -3 -n 3` on a
        // 2-pixel-wide float image is the only input that reaches it.
        && sl_in.xsize >= 3
    {
        let in_nx = sl_in.xsize as usize;
        let ny = sl_in.ysize as usize;
        let out_nx = sl_out.xsize as usize;
        let mut line_vals = vec![0f32; 3 * in_nx];
        let mut f_vals = [0f32; 9];
        let mut offset = 0usize;
        let mut initial_loaded = false;
        for oy in 0..ny {
            if !initial_loaded {
                /* Set up pointers and copy the three lines interleaved the
                first time */
                let line1 = if oy == 0 { 0 } else { oy - 1 };
                let line2 = oy;
                let line3 = (ny - 1).min(oy + 1);
                for ox in 0..in_nx {
                    for (slot, row) in [line1, line2, line3].into_iter().enumerate() {
                        let at = (ox + row * in_nx) * 4;
                        line_vals[3 * ox + slot] =
                            f32::from_ne_bytes(sl_in.data[at..at + 4].try_into().unwrap());
                    }
                }
                offset = 0;
                initial_loaded = true;
            } else {
                /* Thereafter just copy the third line into the free slot */
                let line3 = (ny - 1).min(oy + 1);
                for ox in 0..in_nx {
                    let at = (ox + line3 * in_nx) * 4;
                    line_vals[3 * ox + offset] =
                        f32::from_ne_bytes(sl_in.data[at..at + 4].try_into().unwrap());
                }
                offset = (offset + 1) % 3;
            }

            /* Step across, copying the chunk of values as needed and taking the
            median */
            for ox in 1..in_nx - 1 {
                f_vals.copy_from_slice(&line_vals[3 * (ox - 1)..3 * (ox - 1) + 9]);
                let at = (ox + oy * out_nx) * 4;
                sl_out.data[at..at + 4].copy_from_slice(&opt_med9(&mut f_vals).to_ne_bytes());
            }

            /* Handle the endpoints.  Note the source indexes the far end with
            the *input* width and the row with the output width. */
            f_vals.copy_from_slice(&line_vals[..9]);
            let at = (oy * out_nx) * 4;
            sl_out.data[at..at + 4].copy_from_slice(&opt_med9(&mut f_vals).to_ne_bytes());
            f_vals.copy_from_slice(&line_vals[3 * (in_nx - 3)..3 * (in_nx - 3) + 9]);
            let at = (in_nx + oy * out_nx - 1) * 4;
            sl_out.data[at..at + 4].copy_from_slice(&opt_med9(&mut f_vals).to_ne_bytes());
        }
        return 0;
    }

    let block_size = stack.len() * size as usize * size as usize;
    let mut f_vals = vec![0.0; block_size];
    let mut i_vals = vec![0; block_size];
    let del_minus = size / 2;
    let del_plus = (size + 1) / 2;
    for oy in 0..sl_in.ysize {
        let y_start = (oy - del_minus).max(0);
        let y_end = (oy + del_plus).min(sl_in.ysize);
        for ox in 0..sl_in.xsize {
            let x_start = (ox - del_minus).max(0);
            let x_end = (ox + del_plus).min(sl_in.xsize);
            let num_vals = stack.len() * (x_end - x_start) as usize * (y_end - y_start) as usize;
            let select = (num_vals as i32 + 1) / 2;
            for (oz, slice) in stack.iter().enumerate() {
                for iy in y_start..y_end {
                    for ix in x_start..x_end {
                        let local = (ix - x_start + (iy - y_start) * (x_end - x_start)) as usize;
                        let index =
                            local + oz * (x_end - x_start) as usize * (y_end - y_start) as usize;
                        let pixel = (ix + iy * sl_in.xsize) as usize;
                        match sl_in.mode {
                            SLICE_MODE_FLOAT => {
                                let offset = pixel * 4;
                                f_vals[index] = f32::from_ne_bytes(
                                    slice.data[offset..offset + 4].try_into().unwrap(),
                                );
                            }
                            SLICE_MODE_SHORT => {
                                let offset = pixel * 2;
                                i_vals[index] = i16::from_ne_bytes(
                                    slice.data[offset..offset + 2].try_into().unwrap(),
                                ) as i32;
                            }
                            SLICE_MODE_USHORT => {
                                let offset = pixel * 2;
                                i_vals[index] = u16::from_ne_bytes(
                                    slice.data[offset..offset + 2].try_into().unwrap(),
                                ) as i32;
                            }
                            SLICE_MODE_BYTE => i_vals[index] = slice.data[pixel] as i32,
                            _ => unreachable!(),
                        }
                    }
                }
            }
            let value = if sl_in.mode == SLICE_MODE_FLOAT {
                let low = percentile_float(select, &mut f_vals[..num_vals], num_vals as i32);
                if num_vals % 2 == 0 {
                    0.5 * (low
                        + percentile_float(select + 1, &mut f_vals[..num_vals], num_vals as i32))
                } else {
                    low
                }
            } else {
                let low = percentile_int(select, &mut i_vals[..num_vals], num_vals as i32) as f32;
                if num_vals % 2 == 0 {
                    0.5 * (low
                        + percentile_int(select + 1, &mut i_vals[..num_vals], num_vals as i32)
                            as f32)
                } else {
                    low
                }
            };
            slice_put_val(sl_out, ox, oy, [value, 0., 0., 0.]);
        }
    }
    0
}

struct AnisoState {
    // `sliceproc.c:646`'s `allocate2D_float` exists solely to manufacture a
    // row-pointer view over one malloc allocation.  These contiguous owned
    // vectors retain the same `(m + 2) * (n + 2)` layout without exposing row
    // pointers or a separate allocation/release API.
    image: Vec<f32>,
    image2: Vec<f32>,
    iter_done: i32,
}

/// Owned equivalent of the row-pointer-plus-contiguous-block allocation made
/// by `allocate2D_float`.  `row` exposes the same logical `a[row][column]`
/// layout without returning pointers whose lifetime can outlive the storage.
#[derive(Clone, Debug, PartialEq)]
pub struct FloatMatrix2D {
    rows: usize,
    columns: usize,
    data: Vec<f32>,
}

impl FloatMatrix2D {
    pub fn row(&self, row: usize) -> Option<&[f32]> {
        let start = row.checked_mul(self.columns)?;
        self.data.get(start..start.checked_add(self.columns)?)
    }
    pub fn row_mut(&mut self, row: usize) -> Option<&mut [f32]> {
        let start = row.checked_mul(self.columns)?;
        self.data.get_mut(start..start.checked_add(self.columns)?)
    }
    pub fn into_data(self) -> Vec<f32> {
        self.data
    }
}

/// `allocate2D_float` (`sliceproc.c:646`).  C returns a row-pointer table
/// backed by one contiguous allocation; this returns the checked, owned form
/// of exactly that representation.  `None` is the source allocation failure.
pub fn allocate_2d_float(rows: i32, columns: i32) -> Option<FloatMatrix2D> {
    let (rows, columns) = (usize::try_from(rows).ok()?, usize::try_from(columns).ok()?);
    let length = rows.checked_mul(columns)?;
    let mut data = Vec::new();
    data.try_reserve_exact(length).ok()?;
    data.resize(length, 0.);
    Some(FloatMatrix2D {
        rows,
        columns,
        data,
    })
}

static ANISO_STATE: Mutex<AnisoState> = Mutex::new(AnisoState {
    image: Vec::new(),
    image2: Vec::new(),
    iter_done: 0,
});

pub fn slice_aniso_diff(
    sl: &mut Islice,
    out_mode: i32,
    cc: i32,
    k: f64,
    lambda: f64,
    iterations: i32,
    clear_flag: i32,
) -> i32 {
    let mut state = ANISO_STATE.lock().unwrap();
    if clear_flag == ANISO_CLEAR_ONLY {
        if state.iter_done != 0 {
            state.image.clear();
            state.image2.clear();
            state.iter_done = 0;
        }
        return 0;
    }
    if sl.mode != SLICE_MODE_FLOAT && slice_new_mode(sl, SLICE_MODE_FLOAT) < 0 {
        return -1;
    }
    let n = sl.xsize;
    let m = sl.ysize;
    if state.iter_done == 0 {
        let stride = (n + 2) as usize;
        let Some(image) = allocate_2d_float(m + 2, n + 2) else {
            return -1;
        };
        let Some(image2) = allocate_2d_float(m + 2, n + 2) else {
            return -1;
        };
        state.image = image.into_data();
        state.image2 = image2.into_data();
        for j in 0..m {
            for i in 0..n {
                let offset = (i + j * sl.xsize) as usize * 4;
                state.image[(j + 1) as usize * stride + (i + 1) as usize] =
                    f32::from_ne_bytes(sl.data[offset..offset + 4].try_into().unwrap());
            }
        }
    }
    for _ in 0..iterations {
        if state.iter_done % 2 == 0 {
            let AnisoState { image, image2, .. } = &mut *state;
            let _ = update_matrix(image2, image, m as usize, n as usize, cc, k, lambda);
        } else {
            let AnisoState { image, image2, .. } = &mut *state;
            let _ = update_matrix(image, image2, m as usize, n as usize, cc, k, lambda);
        }
        state.iter_done += 1;
    }
    for j in 0..m {
        for i in 0..n {
            let value = if state.iter_done % 2 == 0 {
                state.image[(j + 1) as usize * (n + 2) as usize + (i + 1) as usize]
            } else {
                state.image2[(j + 1) as usize * (n + 2) as usize + (i + 1) as usize]
            };
            let offset = (i + j * sl.xsize) as usize * 4;
            sl.data[offset..offset + 4].copy_from_slice(&value.to_ne_bytes());
        }
    }
    if clear_flag == ANISO_CLEAR_AT_END {
        state.image.clear();
        state.image2.clear();
        state.iter_done = 0;
    }
    if out_mode != SLICE_MODE_FLOAT && slice_new_mode(sl, out_mode) < 0 {
        return -1;
    }
    0
}

pub fn slice_byte_aniso_diff(
    sl: &mut Islice,
    image: &mut [f32],
    image2: &mut [f32],
    cc: i32,
    k: f64,
    lambda: f64,
    iterations: i32,
    iter_done: &mut i32,
) {
    let n = sl.xsize;
    let m = sl.ysize;
    let stride = (n + 2) as usize;
    if image.len() != (m + 2) as usize * stride || image2.len() != image.len() {
        return;
    }
    if *iter_done == 0 {
        for j in 0..m {
            for i in 0..n {
                image[(j + 1) as usize * stride + (i + 1) as usize] =
                    sl.data[(i + j * sl.xsize) as usize] as f32;
            }
        }
    }
    let mut use_image2 = false;
    for _ in 0..iterations {
        if *iter_done % 2 == 0 {
            let _ = update_matrix(image2, image, m as usize, n as usize, cc, k, lambda);
            use_image2 = true;
        } else {
            let _ = update_matrix(image, image2, m as usize, n as usize, cc, k, lambda);
            use_image2 = false;
        }
        *iter_done += 1;
    }
    for j in 0..m {
        for i in 0..n {
            let v = if use_image2 {
                image2[(j + 1) as usize * stride + (i + 1) as usize]
            } else {
                image[(j + 1) as usize * stride + (i + 1) as usize]
            } as i32;
            sl.data[(i + j * sl.xsize) as usize] = v.clamp(0, 255) as u8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optimized_float_three_by_three_path_uses_clamped_source_neighborhoods() {
        let mut input = slice_create(3, 3, SLICE_MODE_FLOAT).unwrap();
        let mut output = slice_create(3, 3, SLICE_MODE_FLOAT).unwrap();
        for (index, value) in (1..=9).enumerate() {
            input.data[index * 4..index * 4 + 4].copy_from_slice(&(value as f32).to_ne_bytes());
        }
        let stack = vec![input];
        assert_eq!(slice_median_filter(&mut output, &stack, 3), 0);
        assert_eq!(
            f32::from_ne_bytes(output.data[16..20].try_into().unwrap()),
            5.0
        );
        assert_eq!(
            f32::from_ne_bytes(output.data[..4].try_into().unwrap()),
            3.0
        );
        assert_eq!(
            f32::from_ne_bytes(output.data[32..36].try_into().unwrap()),
            7.0
        );
    }

    #[test]
    fn general_integer_and_three_dimensional_median_paths_keep_source_selection() {
        let mut first = slice_create(3, 3, SLICE_MODE_BYTE).unwrap();
        let mut second = slice_create(3, 3, SLICE_MODE_BYTE).unwrap();
        let mut output = slice_create(3, 3, SLICE_MODE_FLOAT).unwrap();
        for index in 0..9 {
            first.data[index] = index as u8;
            second.data[index] = (index + 10) as u8;
        }
        let stack = vec![first, second];
        assert_eq!(slice_median_filter(&mut output, &stack, 3), 0);
        // At the center there are 18 values; source selection averages
        // the ninth and tenth one-indexed percentile values.
        assert_eq!(
            f32::from_ne_bytes(output.data[16..20].try_into().unwrap()),
            9.0
        );
    }

    #[test]
    fn median_filter_rejects_a_nonreal_input_mode_before_output_writes() {
        let input = slice_create(2, 2, 3).unwrap();
        let mut output = slice_create(2, 2, SLICE_MODE_FLOAT).unwrap();
        let stack = vec![input];
        assert_eq!(slice_median_filter(&mut output, &stack, 3), -2);
    }

    #[test]
    fn byte_threshold_and_diffusion_use_owned_slice_storage() {
        let mut slice = slice_create(2, 2, SLICE_MODE_BYTE).unwrap();
        slice.data.copy_from_slice(&[1, 4, 6, 9]);
        slice.min = 1.;
        slice.max = 9.;
        assert_eq!(slice_byte_threshold(&mut slice, 5), 0);
        assert_eq!(slice.data, [9, 9, 1, 1]);

        let mut image = vec![0.; 16];
        let mut image2 = vec![0.; 16];
        let mut iter_done = 0;
        slice_byte_aniso_diff(
            &mut slice,
            &mut image,
            &mut image2,
            2,
            1.,
            0.25,
            0,
            &mut iter_done,
        );
        assert_eq!(slice.data, [9, 9, 1, 1]);
        assert_eq!(image[1 * 4 + 1], 9.);
        assert_eq!(image[2 * 4 + 2], 1.);
    }

    #[test]
    fn byte_add_has_the_source_saturating_range() {
        let mut slice = slice_create(3, 1, SLICE_MODE_BYTE).unwrap();
        slice.data.copy_from_slice(&[0, 20, 250]);
        assert_eq!(slice_byte_add(&mut slice, 10), 0);
        assert_eq!(slice.data, [10, 30, 255]);
        assert_eq!(slice_byte_add(&mut slice, -40), 0);
        assert_eq!(slice.data, [0, 0, 215]);
    }

    #[test]
    fn float_matrix_keeps_contiguous_source_row_layout() {
        let mut matrix = allocate_2d_float(3, 4).unwrap();
        matrix.row_mut(1).unwrap()[2] = 9.;
        assert_eq!(matrix.row(1).unwrap(), [0., 0., 9., 0.]);
        assert_eq!(
            matrix.into_data(),
            [0., 0., 0., 0., 0., 0., 9., 0., 0., 0., 0., 0.]
        );
        assert!(allocate_2d_float(-1, 2).is_none());
    }
}
