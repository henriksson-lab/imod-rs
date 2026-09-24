//! Translation of `IMOD/libiimod/sliceproc.c` and `include/sliceproc.h`.
#![allow(unused_variables)]

use crate::imod::libcfshr::b3dutil::num_omp_threads;
use crate::imod::libcfshr::islice::{Islice, slice_create, slice_put_val, slice_scale_and_free};
use crate::imod::libcfshr::percentile::{percentile_float, percentile_int};
use crate::imod::libiimod::diffusion::update_matrix;
use crate::imod::libiimod::mrcslice::slice_new_mode;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use rayon::slice::ParallelSliceMut;
use std::sync::Mutex;

pub const ANISO_CLEAR_AT_END: i32 = 0;
pub const ANISO_CLEAR_ONLY: i32 = 1;
pub const ANISO_LEAVE_OPEN: i32 = 2;
const SLICE_MODE_BYTE: i32 = 0;
const SLICE_MODE_SHORT: i32 = 1;
const SLICE_MODE_FLOAT: i32 = 2;
const SLICE_MODE_USHORT: i32 = 6;
/// `sliceproc.c:298`.
const MEDIAN_MAX_THREADS: i32 = 6;

pub fn slice_byte_edge_two(sin: &mut Islice, center: i32) -> i32 {
    let mut sout = slice_create(sin.xsize, sin.ysize, SLICE_MODE_FLOAT)
        .expect("dimensions were validated by the input slice");
    let imax = sin.xsize - 1;
    let jmax = sin.ysize - 1;
    let src = sin.data.b();
    let out = sout.data.f_mut();
    for i in 1..imax {
        for j in 1..jmax {
            let get = |x: i32, y: i32| src[(x + y * sin.xsize) as usize] as f32;
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
            out[(i + j * sin.xsize) as usize] = (sr + sc).sqrt();
        }
    }
    for j in 1..jmax {
        let left = (j * sin.xsize) as usize;
        let right = (imax + j * sin.xsize) as usize;
        out[left] = out[left + 1];
        out[right] = out[right - 1];
    }
    for i in 0..=imax {
        let top = i as usize;
        let bottom = (i + jmax * sin.xsize) as usize;
        out[top] = out[top + sin.xsize as usize];
        out[bottom] = out[bottom - sin.xsize as usize];
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
    let d = sin.data.b();
    if d[(i + j * sin.xsize) as usize] as i32 != val {
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
                && d[(x + y * sin.xsize) as usize] as i32 == val
            {
                k += 1;
            }
        }
    }
    k - 1
}

pub fn slice_byte_grow(sin: &mut Islice, val: i32) -> i32 {
    let (xsize, ysize, min, max) = (sin.xsize, sin.ysize, sin.min, sin.max);
    let d = sin.data.b_mut();
    for j in 0..ysize {
        for i in 0..xsize {
            if d[(i + j * xsize) as usize] as i32 != val {
                continue;
            }
            for m in -1..=1 {
                let y = j + m;
                if y < 0 || y >= ysize {
                    continue;
                }
                for n in -1..=1 {
                    let x = i + n;
                    if x == i && y == j || x < 0 || x >= xsize {
                        continue;
                    }
                    let at = (x + y * xsize) as usize;
                    if d[at] as f32 == min {
                        d[at] = (max - 1.) as u8;
                    }
                }
            }
        }
    }
    for pixel in d.iter_mut() {
        if *pixel as f32 == max - 1. {
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
                sout.data.b_mut()[(i + j * sout.xsize) as usize] = 1;
            }
        }
    }
    for j in 0..sin.ysize {
        for i in 0..sin.xsize {
            if sout.data.b()[(i + j * sout.xsize) as usize] != 0 {
                sin.data.b_mut()[(i + j * sin.xsize) as usize] = pmin;
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
    let src = sin.data.b();
    let out = sout.data.f_mut();
    for i in 1..imax {
        for j in 1..jmax {
            let g = |x: i32, y: i32| src[(x + y * sin.xsize) as usize] as f32;
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
            let value = if ixx < delta {
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
            out[(i + j * sin.xsize) as usize] = value;
        }
    }
    for j in 1..jmax {
        let left = (j * sin.xsize) as usize;
        let right = (imax + j * sin.xsize) as usize;
        out[left] = out[left + 1];
        out[right] = out[right - 1];
    }
    for i in 0..=imax {
        let top = i as usize;
        let bottom = (i + jmax * sin.xsize) as usize;
        out[top] = out[top + sin.xsize as usize];
        out[bottom] = out[bottom - sin.xsize as usize];
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

    The source parallelises the row loop with OpenMP (`sliceproc.c:388`),
    keeping the three interleaved lines, the rotating `offset` and the
    `initialLoaded` flag private to each thread.  That per-thread state is part
    of the result, not just bookkeeping: a thread's *first* row builds the
    buffer as rows `oy - 1, oy, oy + 1` in slots 0, 1, 2, and every later row
    overwrites one slot in rotation, so the order of the nine values handed to
    `opt_med9` depends on where the thread's block of rows began.  For ordinary
    data that order is invisible — the network returns the same median *value*
    for any permutation — but `opt_med9` swaps only on `>`, so when the nine
    values contain `+0.0` and `-0.0` together, or a NaN, which of the tied bit
    patterns lands in slot 4 depends on the permutation.  Native's output on
    such data therefore depends on its thread count.  To match it at every
    count the rows are partitioned exactly as the static schedule GCC emits for
    a `parallel for` with no `schedule` clause does: with
    `q = ny / numThreads` and `r = ny % numThreads`, thread `t` takes
    `q + (t < r)` consecutive rows starting at `t * q + min(t, r)`, and each
    group starts with its buffer unloaded.  With one thread this is the single
    group the previous sequential code ran.

    (a) Each group writes only output rows `row0 .. row1` — `ox + oy * outNx`
    for `ox < inNx <= outNx` — so the groups' output slices are disjoint.
    (b) There is no cross-row reduction: each output pixel is `opt_med9` of a
    private copy of nine input values, and each group owns its own
    `lineVals` (allocated once per group, as the source allocates one
    `fThreadLine` per thread) and its own nine-value `fVals`. */
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
        // `sliceproc.c:371-373`: `B3DNINT` is `(int)floor(x + 0.5)`.
        let mut num_threads = MEDIAN_MAX_THREADS
            .min((0.002 * (sl_in.xsize as f64 * sl_in.ysize as f64).sqrt() + 0.5).floor() as i32);
        num_threads = num_omp_threads(1.max(num_threads));
        let src = sl_in.data.f();
        // One group: `(row0, row1, output slice, row the slice starts at)`.
        let run_group = |(row0, row1, out, out_row0): (usize, usize, &mut [f32], usize)| {
            let mut line_vals = vec![0f32; 3 * in_nx];
            let mut f_vals = [0f32; 9];
            let mut offset = 0usize;
            let mut initial_loaded = false;
            for oy in row0..row1 {
                if !initial_loaded {
                    /* Set up pointers and copy the three lines interleaved the
                    first time */
                    let line1 = if oy == 0 { 0 } else { oy - 1 };
                    let line2 = oy;
                    let line3 = (ny - 1).min(oy + 1);
                    for ox in 0..in_nx {
                        for (slot, row) in [line1, line2, line3].into_iter().enumerate() {
                            line_vals[3 * ox + slot] = src[ox + row * in_nx];
                        }
                    }
                    offset = 0;
                    initial_loaded = true;
                } else {
                    /* Thereafter just copy the third line into the free slot */
                    let line3 = (ny - 1).min(oy + 1);
                    for ox in 0..in_nx {
                        line_vals[3 * ox + offset] = src[ox + line3 * in_nx];
                    }
                    offset = (offset + 1) % 3;
                }

                /* Step across, copying the chunk of values as needed and taking
                the median */
                let orow = (oy - out_row0) * out_nx;
                for ox in 1..in_nx - 1 {
                    f_vals.copy_from_slice(&line_vals[3 * (ox - 1)..3 * (ox - 1) + 9]);
                    out[ox + orow] = opt_med9(&mut f_vals);
                }

                /* Handle the endpoints.  Note the source indexes the far end
                with the *input* width and the row with the output width. */
                f_vals.copy_from_slice(&line_vals[..9]);
                out[orow] = opt_med9(&mut f_vals);
                f_vals.copy_from_slice(&line_vals[3 * (in_nx - 3)..3 * (in_nx - 3) + 9]);
                out[in_nx + orow - 1] = opt_med9(&mut f_vals);
            }
        };
        let nthr = num_threads as usize;
        let q = ny / nthr;
        let r = ny % nthr;
        let out = sl_out.data.f_mut();
        if in_nx > out_nx {
            // Not reachable from `clip`, whose output slice has the input's
            // size: the far endpoint of row `oy` then lands in row `oy + 1`,
            // so rows are not disjoint.  Run the same partition, in order, on
            // the whole array — which is what a single native thread would do
            // for each block in turn.
            for t in 0..nthr {
                let row0 = t * q + t.min(r);
                let row1 = row0 + q + usize::from(t < r);
                run_group((row0, row1, &mut out[..], 0));
            }
            return 0;
        }
        let mut groups = Vec::with_capacity(nthr);
        let mut rest = &mut out[..ny * out_nx];
        for t in 0..nthr {
            let row0 = t * q + t.min(r);
            let rows = q + usize::from(t < r);
            let (head, tail) = std::mem::take(&mut rest).split_at_mut(rows * out_nx);
            rest = tail;
            if rows > 0 {
                groups.push((row0, row0 + rows, head, row0));
            }
        }
        if num_threads > 1 {
            // Same pool sizing as `reduce_by_binning.rs`: bounded to the
            // count `numOMPthreads` can ever return.
            let _ = rayon::ThreadPoolBuilder::new()
                .num_threads(num_omp_threads(i32::MAX) as usize)
                .build_global();
            groups.into_par_iter().for_each(run_group);
        } else {
            groups.into_iter().for_each(run_group);
        }
        return 0;
    }

    /* Back to general case: get array for data (`sliceproc.c:441-452`).  The
    source allocates only the array the mode reads through; its `return (-2)`
    for an illegal mode is taken by the mode check at entry, which the 3x3
    float path cannot pre-empt since it requires a legal (float) mode.  Each
    array is indexed only in its own arm below, so the empty one is never
    touched.

    **Deliberate deviation — parallelism the source does not have.** The source
    runs this loop on one thread (`sliceproc.c:363-367`: "the standard
    percentile routine does not run well in parallel").  The owner asked for
    it to be parallelised (2026-09-24) on condition that output is unchanged,
    and it cannot change: every pixel reloads its whole window into a scratch
    array starting at index 0 before `percentileFloat`/`percentileInt` permute
    it, so the value depends only on the input window, never on what the
    scratch array held before or on which pixels ran first.  Each worker owns
    its own scratch arrays (allocated once per worker, `block_size` long as in
    the source), each output row of a band is written by exactly one worker,
    and nothing is reduced across pixels.  The values then go into `slOut`
    through the same `slicePutVal` call, in row-major order, one band at a
    time, so the extra memory is one band of floats rather than a slice.  With
    one thread the loop below is the source's loop. */
    let block_size = stack.len() * size as usize * size as usize;
    let is_float = sl_in.mode == SLICE_MODE_FLOAT;
    let new_vals = || -> (Vec<f32>, Vec<i32>) {
        if is_float {
            (vec![0.0; block_size], Vec::new())
        } else {
            (Vec::new(), vec![0; block_size])
        }
    };
    let del_minus = size / 2;
    let del_plus = (size + 1) / 2;
    let median_at = |ox: i32, oy: i32, f_vals: &mut [f32], i_vals: &mut [i32]| -> f32 {
        let y_start = (oy - del_minus).max(0);
        let y_end = (oy + del_plus).min(sl_in.ysize);
        let x_start = (ox - del_minus).max(0);
        let x_end = (ox + del_plus).min(sl_in.xsize);
        let num_vals = stack.len() * (x_end - x_start) as usize * (y_end - y_start) as usize;
        let select = (num_vals as i32 + 1) / 2;
        /* Loop on slices and subareas to load arrays: `sliceproc.c:446-470`
        and the `FILTER_INTS` macro walk each row's span with a running
        output pointer. */
        let row_len = (x_end - x_start) as usize;
        let mut out = 0;
        if is_float {
            for slice in stack {
                let d = slice.data.f();
                for iy in y_start..y_end {
                    let start = (x_start + iy * sl_in.xsize) as usize;
                    f_vals[out..out + row_len].copy_from_slice(&d[start..start + row_len]);
                    out += row_len;
                }
            }
            let low = percentile_float(select, &mut f_vals[..num_vals], num_vals as i32);
            if num_vals % 2 == 0 {
                0.5 * (low + percentile_float(select + 1, &mut f_vals[..num_vals], num_vals as i32))
            } else {
                low
            }
        } else {
            match sl_in.mode {
                SLICE_MODE_SHORT => {
                    for slice in stack {
                        let d = slice.data.s();
                        for iy in y_start..y_end {
                            let start = (x_start + iy * sl_in.xsize) as usize;
                            for &v in &d[start..start + row_len] {
                                i_vals[out] = v as i32;
                                out += 1;
                            }
                        }
                    }
                }
                SLICE_MODE_USHORT => {
                    for slice in stack {
                        let d = slice.data.us();
                        for iy in y_start..y_end {
                            let start = (x_start + iy * sl_in.xsize) as usize;
                            for &v in &d[start..start + row_len] {
                                i_vals[out] = v as i32;
                                out += 1;
                            }
                        }
                    }
                }
                _ => {
                    for slice in stack {
                        let d = slice.data.b();
                        for iy in y_start..y_end {
                            let start = (x_start + iy * sl_in.xsize) as usize;
                            for &v in &d[start..start + row_len] {
                                i_vals[out] = v as i32;
                                out += 1;
                            }
                        }
                    }
                }
            }
            let low = percentile_int(select, &mut i_vals[..num_vals], num_vals as i32) as f32;
            if num_vals % 2 == 0 {
                0.5 * (low
                    + percentile_int(select + 1, &mut i_vals[..num_vals], num_vals as i32) as f32)
            } else {
                low
            }
        }
    };
    let nx = sl_in.xsize as usize;
    let ny = sl_in.ysize;
    let num_threads = num_omp_threads(ny.max(1));
    // `par_chunks_mut(0)` would panic on a zero-width slice, which has no
    // pixels to compute anyway.
    if num_threads <= 1 || nx == 0 {
        let (mut f_vals, mut i_vals) = new_vals();
        for oy in 0..ny {
            for ox in 0..sl_in.xsize {
                let value = median_at(ox, oy, &mut f_vals, &mut i_vals);
                slice_put_val(sl_out, ox, oy, [value, 0., 0., 0.]);
            }
        }
        return 0;
    }
    // Same pool sizing as `reduce_by_binning.rs`.
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(num_omp_threads(i32::MAX) as usize)
        .build_global();
    let band_rows = (8 * num_threads).min(ny) as usize;
    let mut band = vec![0f32; band_rows * nx];
    let mut row0 = 0;
    while row0 < ny {
        let rows = (band_rows as i32).min(ny - row0) as usize;
        band[..rows * nx]
            .par_chunks_mut(nx)
            .enumerate()
            .for_each_init(new_vals, |(f_vals, i_vals), (k, row)| {
                let oy = row0 + k as i32;
                for (ox, out) in row.iter_mut().enumerate() {
                    *out = median_at(ox as i32, oy, f_vals, i_vals);
                }
            });
        for k in 0..rows {
            for ox in 0..nx {
                slice_put_val(
                    sl_out,
                    ox as i32,
                    row0 + k as i32,
                    [band[ox + k * nx], 0., 0., 0.],
                );
            }
        }
        row0 += rows as i32;
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
        let src = sl.data.f();
        for j in 0..m {
            for i in 0..n {
                state.image[(j + 1) as usize * stride + (i + 1) as usize] =
                    src[(i + j * sl.xsize) as usize];
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
            sl.data.f_mut()[(i + j * sl.xsize) as usize] = value;
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
                    sl.data.b()[(i + j * sl.xsize) as usize] as f32;
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
            sl.data.b_mut()[(i + j * sl.xsize) as usize] = v.clamp(0, 255) as u8;
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
            input.data.f_mut()[index] = value as f32;
        }
        let stack = vec![input];
        assert_eq!(slice_median_filter(&mut output, &stack, 3), 0);
        assert_eq!(output.data.f()[4], 5.0);
        assert_eq!(output.data.f()[0], 3.0);
        assert_eq!(output.data.f()[8], 7.0);
    }

    #[test]
    fn general_integer_and_three_dimensional_median_paths_keep_source_selection() {
        let mut first = slice_create(3, 3, SLICE_MODE_BYTE).unwrap();
        let mut second = slice_create(3, 3, SLICE_MODE_BYTE).unwrap();
        let mut output = slice_create(3, 3, SLICE_MODE_FLOAT).unwrap();
        for index in 0..9 {
            first.data.b_mut()[index] = index as u8;
            second.data.b_mut()[index] = (index + 10) as u8;
        }
        let stack = vec![first, second];
        assert_eq!(slice_median_filter(&mut output, &stack, 3), 0);
        // At the center there are 18 values; source selection averages
        // the ninth and tenth one-indexed percentile values.
        assert_eq!(output.data.f()[4], 9.0);
    }

    #[test]
    fn median_filter_rejects_a_nonreal_input_mode_before_output_writes() {
        let input = slice_create(2, 2, 3).unwrap();
        let mut output = slice_create(2, 2, SLICE_MODE_FLOAT).unwrap();
        let stack = vec![input];
        assert_eq!(slice_median_filter(&mut output, &stack, 3), -2);
    }

    #[test]
    fn byte_diffusion_uses_owned_slice_storage() {
        let mut slice = slice_create(2, 2, SLICE_MODE_BYTE).unwrap();
        slice.data.b_mut().copy_from_slice(&[9, 9, 1, 1]);
        slice.min = 1.;
        slice.max = 9.;

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
        assert_eq!(slice.data.b(), [9, 9, 1, 1]);
        assert_eq!(image[1 * 4 + 1], 9.);
        assert_eq!(image[2 * 4 + 2], 1.);
    }

    #[test]
    fn float_matrix_keeps_contiguous_source_row_layout() {
        let matrix = allocate_2d_float(3, 4).unwrap();
        assert_eq!(matrix.row(1).unwrap(), [0., 0., 0., 0.]);
        assert!(matrix.row(3).is_none());
        assert_eq!(matrix.into_data().len(), 12);
        assert!(allocate_2d_float(-1, 2).is_none());
    }
}
