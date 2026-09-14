//! Direct, function-for-function translation of `IMOD/libcfshr/zoomdown.c`,
//! which reduces images with a selectable interpolation filter.
//!
//! The source passes every image buffer as `unsigned char **slines` /
//! `void *outData` and casts it to the pixel type named by the run-time
//! `dtype` argument.  Rust cannot reinterpret a slice's element type without
//! `unsafe`, so the cast the C defers into the callee is made by the caller
//! here: [`ZoomLines`] and [`ZoomOut`] name the pixel type, and the routines
//! select the arm that `dtype` calls for.  The same treatment is applied to
//! the three internal buffers the source addresses through a union or a
//! reinterpreting cast: [`Weighttab`]'s `union {short *s; float *f;}`
//! (`zoomdown.c:79`) becomes an offset into a [`WeightBuf`], `accumBuf`
//! (`zoomdown.c:566`) becomes an [`AccumBuf`], and `filtBuf`
//! (`zoomdown.c:314`) becomes a [`FiltBuf`].
#![allow(dead_code)]

use crate::imod::libcfshr::b3dutil::{b3d_omp_thread_num, num_omp_threads};
use core::cell::Cell;

pub type B3dInt16 = i16;
pub type B3dUInt16 = u16;
pub type B3dInt32 = i32;
pub type B3dUInt32 = u32;
pub type B3dFloat = f32;

/// Original `fn_proc` (`zoomdown.c:70`).
pub type FnProc = fn(f64) -> f64;

/// Original `Filt`, "A 1-D FILTER" (`zoomdown.c:72`).
#[derive(Copy, Clone)]
pub struct Filt {
    /// filter function
    pub func: FnProc,
    /// radius of nonzero portion
    pub supp: f64,
}

/// Original `Weighttab`, "SAMPLED FILTER WEIGHT TABLE" (`zoomdown.c:77`).
///
/// The source's `union { short *s; float *f; } weight` points into one shared
/// `malloc` block; here `weight` is the offset of this table's slice within
/// the [`WeightBuf`] that stands in for that block.
#[derive(Copy, Clone, Default)]
pub struct Weighttab {
    /// range of samples is [i0..i1-1]
    pub i0: i32,
    pub i1: i32,
    /// `weight[i]` goes with pixel at `i0 + i`
    pub weight: usize,
}

/// The source's `xweightSbuf` / `yweight[].weight` block (`zoomdown.c:296`,
/// `:313`), one `malloc` of `n * psizeWgt` bytes addressed as shorts when the
/// weights are quantised and as floats when they are not.
pub enum WeightBuf {
    Short(Vec<i16>),
    Float(Vec<f32>),
}

/// The source's `accumBuf` (`zoomdown.c:312`), one `malloc` addressed as
/// `b3dInt32 *` for the byte and RGB paths and as `b3dFloat *accumFbuf` for
/// the float, short and ushort paths (`zoomdown.c:566`).
pub enum AccumBuf {
    Int(Vec<i32>),
    Float(Vec<f32>),
}

/// The source's `filtBuf` (`zoomdown.c:314`), the mapping intermediate, typed
/// as `scanline_filter` writes it and `scanline_remap` reads it.
pub enum FiltBuf {
    Byte(Vec<u8>),
    Short(Vec<i16>),
    UShort(Vec<u16>),
}

/// One input scan line: the source's `unsigned char *lineb` after
/// `scanline_accum` casts it to the type `dtype` names (`zoomdown.c:571`).
#[derive(Copy, Clone)]
pub enum ZoomLine<'a> {
    Byte(&'a [u8]),
    Short(&'a [i16]),
    UShort(&'a [u16]),
    Float(&'a [f32]),
}

/// The source's `unsigned char **slines` (`zoomdown.c:246`).
#[derive(Copy, Clone)]
pub enum ZoomLines<'a> {
    Byte(&'a [&'a [u8]]),
    Short(&'a [&'a [i16]]),
    UShort(&'a [&'a [u16]]),
    Float(&'a [&'a [f32]]),
}

/// The source's `void *outData` (`zoomdown.c:246`) and the `obufb` it hands to
/// `scanline_filter`/`scanline_remap`.  `Index` is the mapped RGBA output,
/// which the documentation requires to be unsigned integers.
pub enum ZoomOut<'a> {
    Byte(&'a mut [u8]),
    Short(&'a mut [i16]),
    UShort(&'a mut [u16]),
    Float(&'a mut [f32]),
    Index(&'a mut [u32]),
}

pub const SLICE_MODE_BYTE: i32 = 0;
pub const SLICE_MODE_SHORT: i32 = 1;
pub const SLICE_MODE_FLOAT: i32 = 2;
pub const SLICE_MODE_USHORT: i32 = 6;
pub const SLICE_MODE_RGB: i32 = 16;

/// Original `PI` (`zoomdown.c:86`).  The literal the source wrote, not
/// `f64::consts::PI`.
pub const PI: f64 = 3.14159265358979323846264338;
/// Original `CHANBITS` (`zoomdown.c:88`).
pub const CHANBITS: i32 = 8;
/// Original `WEIGHTBITS`, "# bits in filter coefficients" (`zoomdown.c:89`).
pub const WEIGHTBITS: i32 = 14;
/// Original `FINALSHIFT`, "shift after x&y filter passes" (`zoomdown.c:90`).
pub const FINALSHIFT: i32 = 2 * WEIGHTBITS - CHANBITS;
/// Original `WEIGHTONE`, "filter weight of one" (`zoomdown.c:91`).
pub const WEIGHTONE: i32 = 1 << WEIGHTBITS;
/// Original `MAX_THREADS` (`zoomdown.c:110`).
pub const MAX_THREADS: i32 = 16;
/// Original `NUM_FILT` (`zoomdown.c:111`).
pub const NUM_FILT: i32 = 6;

/// Original `filters` (`zoomdown.c:113`).
static FILTERS: [Filt; NUM_FILT as usize] = [
    Filt {
        func: filt_binning,
        supp: 0.5,
    },
    Filt {
        func: filt_blackman,
        supp: 1.,
    },
    Filt {
        func: filt_triangle,
        supp: 1.,
    },
    Filt {
        func: filt_mitchell,
        supp: 2.,
    },
    Filt {
        func: filt_lanczos2,
        supp: 2.,
    },
    Filt {
        func: filt_lanczos3,
        supp: 3.,
    },
];

thread_local! {
    /// Original `zoom_debug`, "debug level: 0=none, 2=filters"
    /// (`zoomdown.c:122`).
    static ZOOM_DEBUG: Cell<i32> = const { Cell::new(0) };
    /// Original `sXsupport`, `sYsupport`, "scaled filter support radius"
    /// (`zoomdown.c:125`).
    static S_X_SUPPORT: Cell<f64> = const { Cell::new(0.) };
    static S_Y_SUPPORT: Cell<f64> = const { Cell::new(0.) };
    /// Original `sXscale`, `sYscale`, "filter scale (spacing between centers
    /// in a space)" (`zoomdown.c:126`).
    static S_X_SCALE: Cell<f64> = const { Cell::new(0.) };
    static S_Y_SCALE: Cell<f64> = const { Cell::new(0.) };
    /// Original `sXwidth`, `sYwidth`, "filter width: max number of nonzero
    /// samples" (`zoomdown.c:127`).
    static S_X_WIDTH: Cell<i32> = const { Cell::new(0) };
    static S_Y_WIDTH: Cell<i32> = const { Cell::new(0) };
    /// Original `sValueScaling`, "Scaling factor for image values"
    /// (`zoomdown.c:128`).
    static S_VALUE_SCALING: Cell<f32> = const { Cell::new(1.0) };
    /// Original `sFilt_func`, "The selected filter function"
    /// (`zoomdown.c:131`).
    static S_FILT_FUNC: Cell<Option<FnProc>> = const { Cell::new(None) };
    /// Original `sMitchP0` .. `sMitchQ3` (`zoomdown.c:132`).
    static S_MITCH_P0: Cell<f64> = const { Cell::new(0.) };
    static S_MITCH_P2: Cell<f64> = const { Cell::new(0.) };
    static S_MITCH_P3: Cell<f64> = const { Cell::new(0.) };
    static S_MITCH_Q0: Cell<f64> = const { Cell::new(0.) };
    static S_MITCH_Q1: Cell<f64> = const { Cell::new(0.) };
    static S_MITCH_Q2: Cell<f64> = const { Cell::new(0.) };
    static S_MITCH_Q3: Cell<f64> = const { Cell::new(0.) };
}

/// Original `selectZoomFilter` (`zoomdown.c:143`).
pub fn select_zoom_filter(type_0: i32, zoom: f64, out_width: &mut i32) -> i32 {
    S_FILT_FUNC.with(|c| c.set(None));
    if type_0 < 0 || type_0 >= NUM_FILT {
        return 1;
    }
    if zoom >= 1. || zoom <= 0. {
        return 2;
    }
    S_FILT_FUNC.with(|c| c.set(Some(FILTERS[type_0 as usize].func)));
    let (mut scale, mut support, mut width) = (
        S_X_SCALE.with(|c| c.get()),
        S_X_SUPPORT.with(|c| c.get()),
        S_X_WIDTH.with(|c| c.get()),
    );
    set_filter_statics(
        type_0,
        zoom,
        &mut scale,
        &mut support,
        &mut width,
        out_width,
    );
    S_X_SCALE.with(|c| c.set(scale));
    S_X_SUPPORT.with(|c| c.set(support));
    S_X_WIDTH.with(|c| c.set(width));
    let (mut scale, mut support, mut width) = (
        S_Y_SCALE.with(|c| c.get()),
        S_Y_SUPPORT.with(|c| c.get()),
        S_Y_WIDTH.with(|c| c.get()),
    );
    set_filter_statics(
        type_0,
        zoom,
        &mut scale,
        &mut support,
        &mut width,
        out_width,
    );
    S_Y_SCALE.with(|c| c.set(scale));
    S_Y_SUPPORT.with(|c| c.set(support));
    S_Y_WIDTH.with(|c| c.set(width));
    if type_0 == 3 {
        mitchell_init(1. / 3., 1. / 3.);
    }
    0
}

/// Original `setFilterStatics` (`zoomdown.c:161`), "For the filter type and
/// zoom, set the variables describing the filter".
fn set_filter_statics(
    type_0: i32,
    zoom: f64,
    scale: &mut f64,
    support: &mut f64,
    width: &mut i32,
    ret_width: &mut i32,
) {
    *scale = 1. / zoom;
    *support = FILTERS[type_0 as usize].supp * *scale;
    *width = (2. * *support).ceil() as i32;
    *ret_width = *width;
}

/// Original `selectzoomfilter` (`zoomdown.c:172`).
pub fn selectzoomfilter(type_0: &i32, zoom: &f32, out_width: &mut i32) -> i32 {
    select_zoom_filter(*type_0, *zoom as f64, out_width)
}

/// Original `selectZoomFilterXY` (`zoomdown.c:181`).
pub fn select_zoom_filter_xy(
    type_0: i32,
    xzoom: f64,
    yzoom: f64,
    out_width_x: &mut i32,
    out_width_y: &mut i32,
) -> i32 {
    /* Set it up for the lower zoom then compute the variables for the higher zoom axis */
    let err = select_zoom_filter(
        type_0,
        if xzoom < yzoom { xzoom } else { yzoom },
        out_width_x,
    );
    if err != 0 {
        return err;
    }
    if yzoom >= xzoom {
        let (mut scale, mut support, mut width) = (
            S_Y_SCALE.with(|c| c.get()),
            S_Y_SUPPORT.with(|c| c.get()),
            S_Y_WIDTH.with(|c| c.get()),
        );
        set_filter_statics(
            type_0,
            yzoom,
            &mut scale,
            &mut support,
            &mut width,
            out_width_y,
        );
        S_Y_SCALE.with(|c| c.set(scale));
        S_Y_SUPPORT.with(|c| c.set(support));
        S_Y_WIDTH.with(|c| c.set(width));
    } else {
        *out_width_y = *out_width_x;
        let (mut scale, mut support, mut width) = (
            S_X_SCALE.with(|c| c.get()),
            S_X_SUPPORT.with(|c| c.get()),
            S_X_WIDTH.with(|c| c.get()),
        );
        set_filter_statics(
            type_0,
            xzoom,
            &mut scale,
            &mut support,
            &mut width,
            out_width_x,
        );
        S_X_SCALE.with(|c| c.set(scale));
        S_X_SUPPORT.with(|c| c.set(support));
        S_X_WIDTH.with(|c| c.set(width));
    }
    0
}

/// Original `selectzoomfilterXY` (`zoomdown.c:203`).
pub fn selectzoomfilter_xy(
    type_0: &i32,
    xzoom: &f32,
    yzoom: &f32,
    out_width_x: &mut i32,
    out_width_y: &mut i32,
) -> i32 {
    select_zoom_filter_xy(
        *type_0,
        *xzoom as f64,
        *yzoom as f64,
        out_width_x,
        out_width_y,
    )
}

/// Original `setZoomValueScaling` (`zoomdown.c:214`).
pub fn set_zoom_value_scaling(factor: f32) {
    S_VALUE_SCALING.with(|c| c.set(factor));
}

/// Original `zoomWithFilter` (`zoomdown.c:246`).
///
/// The source's five `malloc` failure returns (error 5) are unreachable here:
/// `Vec` allocation aborts rather than returning null.
pub fn zoom_with_filter(
    slines: ZoomLines,
    a_xsize: i32,
    a_ysize: i32,
    a_xoff: f32,
    a_yoff: f32,
    b_xsize: i32,
    b_ysize: i32,
    b_xdim: i32,
    b_xoff: i32,
    dtype: i32,
    out_data: &mut ZoomOut,
    cindex: Option<&[u32]>,
    bindex: Option<&[u8]>,
) -> i32 {
    let mut yweight: [Weighttab; MAX_THREADS as usize] = [Weighttab::default(); 16];
    let mut mapping = 0;
    let psize_filt: i32;
    let mut fweight: f32 = 0.;
    let mut sweight: i16 = 0;

    if S_FILT_FUNC.with(|c| c.get()).is_none() {
        return 1;
    }

    let mut psize_accum = 4;
    let mut psize_wgt = 2;
    match dtype {
        SLICE_MODE_BYTE => {
            if cindex.is_some() {
                mapping = 1;
            }
            psize_filt = 1;
        }
        SLICE_MODE_RGB => {
            if bindex.is_some() {
                mapping = 1;
            }
            psize_accum = 12;
            psize_filt = 3;
        }
        SLICE_MODE_FLOAT => {
            if cindex.is_some() || bindex.is_some() {
                return 3;
            }
            psize_filt = 4;
            psize_wgt = 4;
        }
        SLICE_MODE_SHORT | SLICE_MODE_USHORT => {
            if cindex.is_some() {
                mapping = 1;
            }
            psize_filt = 2;
            psize_wgt = 4;
        }
        x if x == -(SLICE_MODE_RGB + 2) || x == -SLICE_MODE_RGB => {
            if cindex.is_some() || bindex.is_some() {
                return 3;
            }
            psize_accum = 12;
            psize_filt = 4;
        }
        x if x == -(SLICE_MODE_RGB + 3) || x == -(SLICE_MODE_RGB + 1) => {
            if cindex.is_some() || bindex.is_some() {
                return 3;
            }
            psize_filt = 4;
        }
        _ => return 2,
    }

    if dtype < 0 && mapping != 0 {
        return 3;
    }

    let s_x_scale = S_X_SCALE.with(|c| c.get());
    let s_y_scale = S_Y_SCALE.with(|c| c.get());
    let s_x_support = S_X_SUPPORT.with(|c| c.get());
    let s_y_support = S_Y_SUPPORT.with(|c| c.get());
    let s_x_width = S_X_WIDTH.with(|c| c.get());
    let s_y_width = S_Y_WIDTH.with(|c| c.get());
    let s_value_scaling = S_VALUE_SCALING.with(|c| c.get());

    if (a_xoff as f64) < 0.
        || (a_yoff as f64) < 0.
        || (a_xoff as f64 + b_xsize as f64 * s_x_scale) as i32 > a_xsize
        || (a_yoff as f64 + b_ysize as f64 * s_y_scale) as i32 > a_ysize
    {
        /* printf("axo %f bxs %d sc %f xe %f axs %d   ayo %f bys %d ye %f ays %d\n", aXoff,
        bXsize, sXscale, aXoff + bXsize * sXscale, aXsize, aYoff, bYsize,
        aYoff + bYsize * sYscale, aYsize); */
        return 4;
    }

    let mut num_threads = (0.04 * (a_xsize as f64 * a_ysize as f64).sqrt() + 0.5).floor() as i32;
    num_threads = num_omp_threads(if num_threads < MAX_THREADS {
        num_threads
    } else {
        MAX_THREADS
    });
    num_threads = if num_threads < MAX_THREADS {
        num_threads
    } else {
        MAX_THREADS
    };

    /* Allocate accumulation, filter output, and weight buffers */
    let mut xweights: Vec<Weighttab> = vec![Weighttab::default(); b_xsize as usize];
    let mut xweight_buf = if psize_wgt == 4 {
        WeightBuf::Float(vec![0.; (b_xsize * s_x_width) as usize])
    } else {
        WeightBuf::Short(vec![0; (b_xsize * s_x_width) as usize])
    };

    let mut filt_buf: Vec<Option<FiltBuf>> = Vec::new();
    let mut accum_buf: Vec<AccumBuf> = Vec::new();
    let mut yweight_buf: Vec<WeightBuf> = Vec::new();
    for _i in 0..num_threads {
        accum_buf.push(if psize_wgt == 4 {
            AccumBuf::Float(vec![0.; (a_xsize * psize_accum / 4) as usize])
        } else {
            AccumBuf::Int(vec![0; (a_xsize * psize_accum / 4) as usize])
        });
        yweight_buf.push(if psize_wgt == 4 {
            WeightBuf::Float(vec![0.; s_y_width as usize])
        } else {
            WeightBuf::Short(vec![0; s_y_width as usize])
        });
        filt_buf.push(if mapping != 0 {
            Some(match dtype {
                SLICE_MODE_SHORT => FiltBuf::Short(vec![0; b_xsize as usize]),
                SLICE_MODE_USHORT => FiltBuf::UShort(vec![0; b_xsize as usize]),
                _ => FiltBuf::Byte(vec![0; (b_xsize * psize_filt) as usize]),
            })
        } else {
            None
        });
    }

    /*
     * prepare a weighttab (a sampled filter for source pixels) for
     * each dest x position
     */
    for bx in 0..b_xsize {
        xweights[bx as usize].weight = (bx * s_x_width) as usize;
        make_weighttab(
            bx,
            a_xoff as f64 + (bx as f64 + 0.5) * s_x_scale,
            a_xsize,
            s_x_scale,
            s_x_support,
            dtype,
            &mut xweights[bx as usize],
            &mut xweight_buf,
        );
    }

    for by in 0..b_ysize {
        let thr = b3d_omp_thread_num();

        /* prepare a weighttab for dest y position by */
        yweight[thr as usize].weight = 0;
        make_weighttab(
            by,
            a_yoff as f64 + (by as f64 + 0.5) * s_y_scale,
            a_ysize,
            s_y_scale,
            s_y_support,
            dtype,
            &mut yweight[thr as usize],
            &mut yweight_buf[thr as usize],
        );

        /* Zero the scanline accum buffer */
        for i in 0..(a_xsize * psize_accum / 4) as usize {
            match &mut accum_buf[thr as usize] {
                AccumBuf::Int(v) => v[i] = 0,
                AccumBuf::Float(v) => v[i] = 0.,
            }
        }

        /* loop over source scanlines that influence this dest scanline */
        for ayf in yweight[thr as usize].i0..yweight[thr as usize].i1 {
            let k = (ayf - yweight[thr as usize].i0) as usize;
            match &yweight_buf[thr as usize] {
                WeightBuf::Short(v) => {
                    if psize_wgt == 2 {
                        sweight = v[k];
                    }
                }
                WeightBuf::Float(v) => {
                    if psize_wgt != 2 {
                        fweight = v[k] * s_value_scaling;
                    }
                }
            }

            let lineb = match slines {
                ZoomLines::Byte(l) => ZoomLine::Byte(l[ayf as usize]),
                ZoomLines::Short(l) => ZoomLine::Short(l[ayf as usize]),
                ZoomLines::UShort(l) => ZoomLine::UShort(l[ayf as usize]),
                ZoomLines::Float(l) => ZoomLine::Float(l[ayf as usize]),
            };
            /* add weighted tbuf into accum (these do yfilt) */
            scanline_accum(
                lineb,
                dtype,
                a_xsize,
                &mut accum_buf[thr as usize],
                sweight,
                fweight,
            );
        }

        /* and filter it into the appropriate line of output or into filtBuf */
        let line_index = (by * b_xdim + b_xoff) as usize;
        if mapping != 0 {
            let mut obufb = match filt_buf[thr as usize].as_mut().unwrap() {
                FiltBuf::Byte(v) => ZoomOut::Byte(v),
                FiltBuf::Short(v) => ZoomOut::Short(v),
                FiltBuf::UShort(v) => ZoomOut::UShort(v),
            };
            scanline_filter(
                &accum_buf[thr as usize],
                dtype,
                a_xsize,
                &mut obufb,
                b_xsize,
                &xweights,
                &xweight_buf,
                FINALSHIFT,
            );

            /* Map to RGBA output, always 4 bytes, if index tables provided */
            scanline_remap(
                filt_buf[thr as usize].as_ref().unwrap(),
                dtype,
                b_xsize,
                out_data,
                line_index,
                cindex,
                bindex,
            );
        } else {
            let mut obufb = match out_data {
                ZoomOut::Byte(v) => ZoomOut::Byte(&mut v[psize_filt as usize * line_index..]),
                ZoomOut::Short(v) => ZoomOut::Short(&mut v[line_index..]),
                ZoomOut::UShort(v) => ZoomOut::UShort(&mut v[line_index..]),
                ZoomOut::Float(v) => ZoomOut::Float(&mut v[line_index..]),
                ZoomOut::Index(v) => ZoomOut::Index(&mut v[line_index..]),
            };
            /*for (i = 30; i < 45; i++)
            printf("%.1f ", *((float *)accumBuf[thr] + i));
            printf("\n"); */
            scanline_filter(
                &accum_buf[thr as usize],
                dtype,
                a_xsize,
                &mut obufb,
                b_xsize,
                &xweights,
                &xweight_buf,
                FINALSHIFT,
            );
        }
    }
    0
}

/// Original `zoomwithfilter` (`zoomdown.c:428`).
pub fn zoomwithfilter(
    array: &[f32],
    a_xsize: &i32,
    a_ysize: &i32,
    a_xoff: &f32,
    a_yoff: &f32,
    b_xsize: &i32,
    b_ysize: &i32,
    b_xdim: &i32,
    b_xoff: &i32,
    out_data: &mut [f32],
) -> i32 {
    // `makeLinePointers(array, *aXsize, *aYsize, 4)` (`b3dutil.c:1269`) hands
    // back raw line pointers; a vector of line slices is the same thing and
    // cannot fail, so the source's `if (!linePtrs) return 5` is unreachable.
    let line_ptrs: Vec<&[f32]> = (0..*a_ysize as usize)
        .map(|i| &array[i * *a_xsize as usize..])
        .collect();
    zoom_with_filter(
        ZoomLines::Float(&line_ptrs),
        *a_xsize,
        *a_ysize,
        *a_xoff,
        *a_yoff,
        *b_xsize,
        *b_ysize,
        *b_xdim,
        *b_xoff,
        SLICE_MODE_FLOAT,
        &mut ZoomOut::Float(out_data),
        None,
        None,
    )
}

/// Original `zoomFiltInterp` (`zoomdown.c:456`).
pub fn zoom_filt_interp(
    array: &[f32],
    bray: &mut [f32],
    nxa: i32,
    nya: i32,
    nxb: i32,
    nyb: i32,
    xc: f32,
    yc: f32,
    xt: f32,
    yt: f32,
    dmean: f32,
) -> i32 {
    let (mut a_xoff, mut a_yoff) = (0., 0.);
    let (mut b_xsize, mut b_ysize, mut b_xoff, mut b_yoff) = (0, 0, 0, 0);
    // See `zoomwithfilter` above on `makeLinePointers`.
    let line_ptrs: Vec<&[f32]> = (0..nya as usize)
        .map(|i| &array[i * nxa as usize..])
        .collect();
    interp_limits(
        nxa,
        nxb,
        xc,
        xt,
        S_X_SCALE.with(|c| c.get()),
        &mut a_xoff,
        &mut b_xsize,
        &mut b_xoff,
    );
    interp_limits(
        nya,
        nyb,
        yc,
        yt,
        S_Y_SCALE.with(|c| c.get()),
        &mut a_yoff,
        &mut b_ysize,
        &mut b_yoff,
    );
    /* printf("%f %d %d  %f %d %d\n", aXoff, bXsize, bXoff, aYoff, bYsize, bYoff); */
    if b_xsize > 0 && b_ysize > 0 {
        let i = zoom_with_filter(
            ZoomLines::Float(&line_ptrs),
            nxa,
            nya,
            a_xoff,
            a_yoff,
            b_xsize,
            b_ysize,
            nxb,
            b_xoff,
            SLICE_MODE_FLOAT,
            &mut ZoomOut::Float(&mut bray[(nxb * b_yoff) as usize..]),
            None,
            None,
        );
        if i != 0 {
            /*printf("ERROR FROM zoomWithFilter: %d\n", i);*/
            return i;
        }
    }

    /* Fill lines as needed: whole lines or sides */
    for iy in 0..nyb {
        if iy < b_yoff || iy >= b_ysize + b_yoff {
            for ix in 0..nxb {
                bray[(ix + iy * nxb) as usize] = dmean;
            }
        } else {
            for ix in 0..b_xoff {
                bray[(ix + iy * nxb) as usize] = dmean;
            }
            for ix in b_xsize + b_xoff..nxb {
                bray[(ix + iy * nxb) as usize] = dmean;
            }
        }
    }
    0
}

/// Original `zoomfiltinterp` (`zoomdown.c:498`).
#[allow(clippy::too_many_arguments)]
pub fn zoomfiltinterp(
    array: &[f32],
    bray: &mut [f32],
    nxa: &i32,
    nya: &i32,
    nxb: &i32,
    nyb: &i32,
    xc: &f32,
    yc: &f32,
    xt: &f32,
    yt: &f32,
    dmean: &f32,
) -> i32 {
    zoom_filt_interp(
        array, bray, *nxa, *nya, *nxb, *nyb, *xc, *yc, *xt, *yt, *dmean,
    )
}

/// Original `zoomFiltValue` (`zoomdown.c:508`).
pub fn zoom_filt_value(radius: f32) -> f64 {
    let mut den = 0.;
    let filt = match S_FILT_FUNC.with(|c| c.get()) {
        None => return 0.,
        Some(f) => f,
    };
    let s_x_scale = S_X_SCALE.with(|c| c.get());
    let lim = (S_X_WIDTH.with(|c| c.get()) + 1) / 2;
    for i in -lim..=lim {
        den += filt(i as f64 / s_x_scale);
    }
    filt(radius as f64 / s_x_scale) / den
}

/// Original `zoomfiltvalue` (`zoomdown.c:524`).
pub fn zoomfiltvalue(radius: &f32) -> f64 {
    zoom_filt_value(*radius)
}

/// Original `zoomRawFiltValue` (`zoomdown.c:533`).
///
/// The source dereferences `sFilt_func` without a null check; `unwrap` is the
/// same contract.
pub fn zoom_raw_filt_value(radius: f32) -> f64 {
    let filt = S_FILT_FUNC.with(|c| c.get()).unwrap();
    filt(radius as f64)
}

/// Original `interpLimits` (`zoomdown.c:540`), "Compute the limits of usable
/// data in one dimension for calling from the interp function".
fn interp_limits(
    na: i32,
    nb: i32,
    cen: f32,
    trans: f32,
    scale: f64,
    a_off: &mut f32,
    b_size: &mut i32,
    b_off: &mut i32,
) {
    /* Get starting and ending continuous coordinates in b from corners of a and limit
    them to edges of first and last pixel */
    let mut bc_start: f32 = (-(cen as f64) / scale + nb as f64 / 2. + trans as f64) as f32;
    let mut bc_end: f32 = ((na as f64 - cen as f64) / scale + nb as f64 / 2. + trans as f64) as f32;
    bc_start = if 0. > bc_start as f64 { 0. } else { bc_start };
    bc_end = if (nb as f32) < bc_end {
        nb as f32
    } else {
        bc_end
    };

    /* Get discrete coordinates of the nearest whole pixel by rounding up on the start and
    down on the end, then get the offset and size from them */
    let bd_start = (bc_start as f64 - 0.001).ceil() as i32;
    let bd_end = (bc_end as f64 + 0.001).floor() as i32 - 1;
    *b_off = bd_start;
    *a_off = (scale * (bd_start as f64 - nb as f64 / 2. - trans as f64) + cen as f64) as f32;

    /* Make sure this is not negative; roundoff errors can give negative values */
    *a_off = if 0. > *a_off as f64 { 0. } else { *a_off };
    *b_size = bd_end + 1 - bd_start;

    /* Make sure the output size will pass the test below too */
    if *b_size as f64 * scale + *a_off as f64 > na as f64 {
        *b_size -= 1;
    }
}

/// Original `scanline_accum` (`zoomdown.c:572`), "Accumulate lineb from the
/// input image into accumBuf with the given weighting".
fn scanline_accum(
    lineb: ZoomLine,
    dtype: i32,
    a_xsize: i32,
    accum_buf: &mut AccumBuf,
    sweight: i16,
    fweight: f32,
) {
    match dtype {
        SLICE_MODE_BYTE => {
            if let (ZoomLine::Byte(lb), AccumBuf::Int(ab)) = (lineb, &mut *accum_buf) {
                for i in 0..a_xsize as usize {
                    ab[i] = ab[i].wrapping_add((sweight as i32).wrapping_mul(lb[i] as i32));
                }
            }
        }

        SLICE_MODE_RGB => {
            if let (ZoomLine::Byte(lb), AccumBuf::Int(ab)) = (lineb, &mut *accum_buf) {
                for i in 0..a_xsize as usize {
                    for c in 0..3 {
                        ab[3 * i + c] = ab[3 * i + c]
                            .wrapping_add((sweight as i32).wrapping_mul(lb[3 * i + c] as i32));
                    }
                }
            }
        }

        SLICE_MODE_FLOAT => {
            if let (ZoomLine::Float(linef), AccumBuf::Float(accum_fbuf)) = (lineb, &mut *accum_buf)
            {
                for i in 0..a_xsize as usize {
                    accum_fbuf[i] += fweight * linef[i];
                }
            }
        }

        SLICE_MODE_SHORT => {
            if let (ZoomLine::Short(lines), AccumBuf::Float(accum_fbuf)) = (lineb, &mut *accum_buf)
            {
                for i in 0..a_xsize as usize {
                    accum_fbuf[i] += fweight * lines[i] as f32;
                }
            }
        }

        SLICE_MODE_USHORT => {
            if let (ZoomLine::UShort(lineus), AccumBuf::Float(accum_fbuf)) =
                (lineb, &mut *accum_buf)
            {
                for i in 0..a_xsize as usize {
                    accum_fbuf[i] += fweight * lineus[i] as f32;
                }
            }
        }

        /* RGBA passed in with independent channels: add in 3, skip one byte */
        x if x == -(SLICE_MODE_RGB + 2) || x == -SLICE_MODE_RGB => {
            if let (ZoomLine::Byte(lb), AccumBuf::Int(ab)) = (lineb, &mut *accum_buf) {
                for i in 0..a_xsize as usize {
                    for c in 0..3 {
                        ab[3 * i + c] = ab[3 * i + c]
                            .wrapping_add((sweight as i32).wrapping_mul(lb[4 * i + c] as i32));
                    }
                }
            }
        }

        /* Gray-scale RGBA passed in: add in one byte, skip 3 */
        x if x == -(SLICE_MODE_RGB + 3) || x == -(SLICE_MODE_RGB + 1) => {
            if let (ZoomLine::Byte(lb), AccumBuf::Int(ab)) = (lineb, &mut *accum_buf) {
                for i in 0..a_xsize as usize {
                    ab[i] = ab[i].wrapping_add((sweight as i32).wrapping_mul(lb[4 * i] as i32));
                }
            }
        }
        _ => {}
    }
}

/// Original `scanline_filter` (`zoomdown.c:637`): "Applies the set of weight
/// tables in wtab to the accumulated line in lineb; applies the shift to scale
/// the output values appropriately when using short weights".
#[allow(clippy::too_many_arguments)]
fn scanline_filter(
    lineb: &AccumBuf,
    dtype: i32,
    _a_xsize: i32,
    obufb: &mut ZoomOut,
    b_xsize: i32,
    wtab: &[Weighttab],
    wgt: &WeightBuf,
    shift: i32,
) {
    let mut alpha: i32 = 0;

    match dtype {
        SLICE_MODE_BYTE => {
            if let (AccumBuf::Int(lb), WeightBuf::Short(wb), ZoomOut::Byte(ob)) =
                (lineb, wgt, &mut *obufb)
            {
                for b in 0..b_xsize as usize {
                    /* start sum at 1<<shift-1 for rounding */
                    /* Shift the accumulated values by 8 bits to avoid overflow */
                    let mut sum: i32 = 1 << (shift - 1);
                    let base = wtab[b].i0 as usize;
                    for af in 0..(wtab[b].i1 - wtab[b].i0) as usize {
                        sum = sum.wrapping_add(
                            (wb[wtab[b].weight + af] as i32)
                                .wrapping_mul(((lb[base + af] >> CHANBITS) as i16) as i32),
                        );
                    }
                    let t = sum >> shift;
                    ob[b] = (if t < 0 {
                        0
                    } else if t > 255 {
                        255
                    } else {
                        t
                    }) as u8;
                }
            }
        }

        SLICE_MODE_RGB => {
            if let (AccumBuf::Int(lb), WeightBuf::Short(wb), ZoomOut::Byte(ob)) =
                (lineb, wgt, &mut *obufb)
            {
                for b in 0..b_xsize as usize {
                    let mut sumr: i32 = 1 << (shift - 1);
                    let mut sumg: i32 = sumr;
                    let mut sumb: i32 = sumr;
                    let base = 3 * wtab[b].i0 as usize;
                    for af in 0..(wtab[b].i1 - wtab[b].i0) as usize {
                        let w = wb[wtab[b].weight + af] as i32;
                        sumr = sumr.wrapping_add(
                            w.wrapping_mul(((lb[base + 3 * af] >> CHANBITS) as i16) as i32),
                        );
                        sumg = sumg.wrapping_add(
                            w.wrapping_mul(((lb[base + 3 * af + 1] >> CHANBITS) as i16) as i32),
                        );
                        sumb = sumb.wrapping_add(
                            w.wrapping_mul(((lb[base + 3 * af + 2] >> CHANBITS) as i16) as i32),
                        );
                    }
                    for (c, sum) in [sumr, sumg, sumb].into_iter().enumerate() {
                        let t = sum >> shift;
                        ob[3 * b + c] = (if t < 0 {
                            0
                        } else if t > 255 {
                            255
                        } else {
                            t
                        }) as u8;
                    }
                }
            }
        }

        SLICE_MODE_FLOAT => {
            if let (AccumBuf::Float(linef), WeightBuf::Float(wb), ZoomOut::Float(obuff)) =
                (lineb, wgt, &mut *obufb)
            {
                for b in 0..b_xsize as usize {
                    let mut rsum: f32 = 0.;
                    let base = wtab[b].i0 as usize;
                    for af in 0..(wtab[b].i1 - wtab[b].i0) as usize {
                        rsum += wb[wtab[b].weight + af] * linef[base + af];
                    }
                    obuff[b] = rsum;
                }
            }
        }

        SLICE_MODE_SHORT => {
            if let (AccumBuf::Float(linef), WeightBuf::Float(wb), ZoomOut::Short(obufs)) =
                (lineb, wgt, &mut *obufb)
            {
                for b in 0..b_xsize as usize {
                    let mut rsum: f32 = 0.5;
                    let base = wtab[b].i0 as usize;
                    for af in 0..(wtab[b].i1 - wtab[b].i0) as usize {
                        rsum += wb[wtab[b].weight + af] * linef[base + af];
                    }
                    // `B3DMIN(32767., B3DMAX(-32767., rsum))` -- both limits are
                    // double literals, so the clamp is evaluated in double.
                    let lo = if -32767.0f64 > rsum as f64 {
                        -32767.0f64
                    } else {
                        rsum as f64
                    };
                    obufs[b] = (if 32767.0f64 < lo { 32767.0f64 } else { lo }) as i16;
                }
            }
        }

        SLICE_MODE_USHORT => {
            if let (AccumBuf::Float(linef), WeightBuf::Float(wb), ZoomOut::UShort(obufus)) =
                (lineb, wgt, &mut *obufb)
            {
                for b in 0..b_xsize as usize {
                    let mut rsum: f32 = 0.5;
                    let base = wtab[b].i0 as usize;
                    for af in 0..(wtab[b].i1 - wtab[b].i0) as usize {
                        rsum += wb[wtab[b].weight + af] * linef[base + af];
                    }
                    let lo = if 0.0f64 > rsum as f64 {
                        0.0f64
                    } else {
                        rsum as f64
                    };
                    obufus[b] = (if 65535.0f64 < lo { 65535.0f64 } else { lo }) as u16;
                }
            }
        }

        x if x == -(SLICE_MODE_RGB + 2) || x == -SLICE_MODE_RGB => {
            if x == -(SLICE_MODE_RGB + 2) {
                // The source falls through from `case -(SLICE_MODE_RGB + 2)`,
                // which sets `alpha = 255`, into `case -SLICE_MODE_RGB`.
                alpha = 255;
            }
            if let (AccumBuf::Int(lb), WeightBuf::Short(wb), ZoomOut::Byte(ob)) =
                (lineb, wgt, &mut *obufb)
            {
                for b in 0..b_xsize as usize {
                    let mut sumr: i32 = 1 << (shift - 1);
                    let mut sumg: i32 = sumr;
                    let mut sumb: i32 = sumr;
                    let base = 3 * wtab[b].i0 as usize;
                    for af in 0..(wtab[b].i1 - wtab[b].i0) as usize {
                        let w = wb[wtab[b].weight + af] as i32;
                        sumr = sumr.wrapping_add(
                            w.wrapping_mul(((lb[base + 3 * af] >> CHANBITS) as i16) as i32),
                        );
                        sumg = sumg.wrapping_add(
                            w.wrapping_mul(((lb[base + 3 * af + 1] >> CHANBITS) as i16) as i32),
                        );
                        sumb = sumb.wrapping_add(
                            w.wrapping_mul(((lb[base + 3 * af + 2] >> CHANBITS) as i16) as i32),
                        );
                    }
                    for (c, sum) in [sumr, sumg, sumb].into_iter().enumerate() {
                        let t = sum >> shift;
                        ob[4 * b + c] = (if t < 0 {
                            0
                        } else if t > 255 {
                            255
                        } else {
                            t
                        }) as u8;
                    }
                    ob[4 * b + 3] = alpha as u8;
                }
            }
        }

        x if x == -(SLICE_MODE_RGB + 3) || x == -(SLICE_MODE_RGB + 1) => {
            if x == -(SLICE_MODE_RGB + 3) {
                // Fall-through from `case -(SLICE_MODE_RGB + 3)`.
                alpha = 255;
            }
            if let (AccumBuf::Int(lb), WeightBuf::Short(wb), ZoomOut::Byte(ob)) =
                (lineb, wgt, &mut *obufb)
            {
                for b in 0..b_xsize as usize {
                    /* start sum at 1<<shift-1 for rounding */
                    /* Shift the accumulated values by 8 bits to avoid overflow */
                    let mut sum: i32 = 1 << (shift - 1);
                    let base = wtab[b].i0 as usize;
                    for af in 0..(wtab[b].i1 - wtab[b].i0) as usize {
                        sum = sum.wrapping_add(
                            (wb[wtab[b].weight + af] as i32)
                                .wrapping_mul(((lb[base + af] >> CHANBITS) as i16) as i32),
                        );
                    }
                    let t = sum >> shift;
                    let tempb = (if t < 0 {
                        0
                    } else if t > 255 {
                        255
                    } else {
                        t
                    }) as u8;
                    ob[4 * b] = tempb;
                    ob[4 * b + 1] = tempb;
                    ob[4 * b + 2] = tempb;
                    ob[4 * b + 3] = alpha as u8;
                }
            }
        }
        _ => {}
    }
}

/// Original `scanline_remap` (`zoomdown.c:748`): "fills a line in the output
/// buffer by mapping from the filter buffer to the color index values in
/// cindex or bindex depending on the data type".
///
/// The source writes the RGB case as four `unsigned char` stores into the
/// four bytes of an output word; the same bytes are written here as one
/// native-endian `u32`, because the documented output type for mapping is
/// unsigned integers.
fn scanline_remap(
    filt_buf: &FiltBuf,
    dtype: i32,
    b_xsize: i32,
    obufb: &mut ZoomOut,
    line_index: usize,
    cindex: Option<&[u32]>,
    bindex: Option<&[u8]>,
) {
    let ZoomOut::Index(obufi) = obufb else {
        return;
    };
    match dtype {
        SLICE_MODE_BYTE => {
            if let (FiltBuf::Byte(fb), Some(ci)) = (filt_buf, cindex) {
                for i in 0..b_xsize as usize {
                    obufi[line_index + i] = ci[fb[i] as usize];
                }
            }
        }
        SLICE_MODE_RGB => {
            if let (FiltBuf::Byte(fb), Some(bi)) = (filt_buf, bindex) {
                for i in 0..b_xsize as usize {
                    obufi[line_index + i] = u32::from_ne_bytes([
                        bi[fb[3 * i] as usize],
                        bi[fb[3 * i + 1] as usize],
                        bi[fb[3 * i + 2] as usize],
                        0,
                    ]);
                }
            }
        }
        SLICE_MODE_SHORT => {
            if let (FiltBuf::Short(fs), Some(ci)) = (filt_buf, cindex) {
                for i in 0..b_xsize as usize {
                    obufi[line_index + i] = ci[fs[i] as usize];
                }
            }
        }
        SLICE_MODE_USHORT => {
            if let (FiltBuf::UShort(fus), Some(ci)) = (filt_buf, cindex) {
                for i in 0..b_xsize as usize {
                    obufi[line_index + i] = ci[fus[i] as usize];
                }
            }
        }
        _ => {}
    }
}

/// Original `make_weighttab` (`zoomdown.c:789`): "sample the continuous
/// filter, scaled by scale and positioned at continuous source coordinate cen,
/// for source coordinates in the range \[0..len-1\], writing the weights into
/// wtab.  For byte and RGB data types as given in dtype, scale the weights so
/// they sum to WEIGHTONE, store as shorts, and trim leading and trailing zeros
/// for.  b is the dest coordinate (for diagnostics)."
#[allow(clippy::too_many_arguments)]
fn make_weighttab(
    b: i32,
    cen: f64,
    len: i32,
    scale: f64,
    support: f64,
    dtype: i32,
    wtab: &mut Weighttab,
    wgt: &mut WeightBuf,
) {
    let mut lastnonzero: i32 = 0;
    let short_wgts = i32::from(dtype == SLICE_MODE_BYTE || dtype == SLICE_MODE_RGB || dtype < 0);
    let off = wtab.weight;
    let filt = S_FILT_FUNC.with(|c| c.get()).unwrap();
    let zoom_debug = ZOOM_DEBUG.with(|c| c.get());

    /* find the source coord range of this positioned filter: [i0..i1-1] */
    let mut i0 = (cen - support + 0.5) as i32;
    let mut i1 = (cen + support + 0.5) as i32;
    if i0 < 0 {
        i0 = 0;
    }
    if i1 > len {
        i1 = len;
    }
    /*if (i0 >= i1) {
    fprintf(stderr, "make_weighttab: null filter at %d\n", b);
    exit(1);
    }*/

    /* the range of source samples to buffer: */
    wtab.i0 = i0;
    wtab.i1 = i1;

    /* find scale factor sc to normalize the filter */
    let mut den = 0.;
    let mut i = i0;
    while i < i1 {
        den += filt((i as f64 + 0.5 - cen) / scale);
        i += 1;
    }

    /* set sc so that sum of sc*func() is approximately WEIGHTONE */
    let sc = if short_wgts != 0 {
        if den == 0. {
            WEIGHTONE as f64
        } else {
            WEIGHTONE as f64 / den
        }
    } else if den == 0. {
        1.
    } else {
        1. / den
    };
    if zoom_debug > 1 {
        eprint!("    b={b} cen={cen} scale={scale} [{i0}..{i1}) sc={sc}:  ");
    }

    /* compute the discrete, sampled filter coefficients */
    let mut stillzero = short_wgts;
    let mut rsum = 0.;
    let mut sum: i32 = 0;
    let mut wp = 0usize;
    let mut i = i0;
    while i < i1 {
        /* evaluate the filter function: */
        let tr = sc * filt((i as f64 + 0.5 - cen) / scale);
        rsum += tr;

        /* if (tr<MINSHORT || tr>MAXSHORT) {
        fprintf(stderr, "tr=%g at %d\n", tr, b);
        exit(1);
        } */
        if short_wgts != 0 {
            let t = (tr + 0.5).floor() as i32;
            if stillzero != 0 && t == 0 {
                i0 += 1; /* find first nonzero */
            } else {
                stillzero = 0;
                if let WeightBuf::Short(s) = &mut *wgt {
                    s[off + wp] = t as i16; /* add weight to table */
                }
                wp += 1;
                sum = sum.wrapping_add(t);
                if t != 0 {
                    lastnonzero = i; /* find last nonzero */
                }
            }
        } else if let WeightBuf::Float(f) = &mut *wgt {
            f[off + (i - i0) as usize] = tr as f32;
        }
        i += 1;
    }

    if (short_wgts != 0 && sum == 0) || rsum == 0. {
        /* fprintf(stderr, "sum=0 at %d\n", b); */
        wtab.i0 = (wtab.i0 + wtab.i1) >> 1;
        wtab.i1 = wtab.i0 + 1;
        match &mut *wgt {
            WeightBuf::Short(s) => {
                if short_wgts != 0 {
                    s[off] = WEIGHTONE as i16;
                }
            }
            WeightBuf::Float(f) => {
                if short_wgts == 0 {
                    f[off] = 1.;
                }
            }
        }
    } else if short_wgts != 0 {
        /* skip leading and trailing zeros */
        /* set wtab->i0 and ->i1 to the nonzero support of the filter */
        wtab.i0 = i0;
        i1 = lastnonzero + 1;
        wtab.i1 = i1;
        if sum != WEIGHTONE {
            /*
             * Fudge the center slightly to make sum=WEIGHTONE exactly.
             * Is this the best way to normalize a discretely sampled
             * continuous filter?
             */
            let mut i = (cen + 0.5) as i32;
            if i < i0 {
                i = i0;
            } else if i >= i1 {
                i = i1 - 1;
            }
            let t = WEIGHTONE - sum;
            if zoom_debug > 1 {
                eprint!("[{i}]+={t} ");
            }
            if let WeightBuf::Short(s) = &mut *wgt {
                let idx = off + (i - i0) as usize;
                s[idx] = (s[idx] as i32).wrapping_add(t) as i16; /* fudge center sample */
            }
        }
    }
    if zoom_debug > 1 {
        eprint!("\t");
        match &*wgt {
            WeightBuf::Short(s) => {
                if short_wgts != 0 {
                    let mut i = i0;
                    let mut wp = 0usize;
                    while i < i1 {
                        eprint!("{:>5} ", s[off + wp]);
                        i += 1;
                        wp += 1;
                    }
                }
            }
            WeightBuf::Float(f) => {
                if short_wgts == 0 {
                    let mut i = i0;
                    while i < i1 {
                        eprint!("{:.4} ", f[off + (i - i0) as usize]);
                        i += 1;
                    }
                }
            }
        }
        eprintln!();
    }
}

/*
 * The filters
 */

/// Original `filt_binning` (`zoomdown.c:894`).
fn filt_binning(x: f64) -> f64 {
    if x >= -0.5 && x < 0.5 { 1. } else { 0. }
}

/// Original `mitchell_init` (`zoomdown.c:899`).
fn mitchell_init(b: f64, c: f64) {
    S_MITCH_P0.with(|s| s.set((6. - 2. * b) / 6.));
    S_MITCH_P2.with(|s| s.set((-18. + 12. * b + 6. * c) / 6.));
    S_MITCH_P3.with(|s| s.set((12. - 9. * b - 6. * c) / 6.));
    S_MITCH_Q0.with(|s| s.set((8. * b + 24. * c) / 6.));
    S_MITCH_Q1.with(|s| s.set((-12. * b - 48. * c) / 6.));
    S_MITCH_Q2.with(|s| s.set((6. * b + 30. * c) / 6.));
    S_MITCH_Q3.with(|s| s.set((-b - 6. * c) / 6.));
}

/// Original `filt_mitchell` (`zoomdown.c:910`).
fn filt_mitchell(x: f64) -> f64 {
    /*
     * see Mitchell&Netravali, "Reconstruction Filters in Computer Graphics",
     * SIGGRAPH 88
     */
    let p0 = S_MITCH_P0.with(|s| s.get());
    let p2 = S_MITCH_P2.with(|s| s.get());
    let p3 = S_MITCH_P3.with(|s| s.get());
    let q0 = S_MITCH_Q0.with(|s| s.get());
    let q1 = S_MITCH_Q1.with(|s| s.get());
    let q2 = S_MITCH_Q2.with(|s| s.get());
    let q3 = S_MITCH_Q3.with(|s| s.get());
    if x < -2. {
        return 0.;
    }
    if x < -1. {
        return q0 - x * (q1 - x * (q2 - x * q3));
    }
    if x < 0. {
        return p0 + x * x * (p2 - x * p3);
    }
    if x < 1. {
        return p0 + x * x * (p2 + x * p3);
    }
    if x < 2. {
        return q0 + x * (q1 + x * (q2 + x * q3));
    }
    0.
}

/// Original `filt_blackman` (`zoomdown.c:929`), the Blackman window.
fn filt_blackman(x: f64) -> f64 {
    0.42 + 0.50 * (PI * x).cos() + 0.08 * (2. * PI * x).cos()
}

/// Original `filt_triangle` (`zoomdown.c:934`).
fn filt_triangle(x: f64) -> f64 {
    if x <= -1. || x >= 1. {
        return 0.;
    }
    1. - x.abs()
}

/// Original `filt_lanczos2` (`zoomdown.c:941`).
fn filt_lanczos2(x: f64) -> f64 {
    let a = 2.;
    if x < -a || x > a {
        return 0.;
    }
    if x < 1.0e-6 && x > -1.0e-6 {
        return 1.;
    }
    (a * (PI * x).sin() * (PI * x / a).sin()) / (PI * PI * x * x)
}

/// Original `filt_lanczos3` (`zoomdown.c:951`).
fn filt_lanczos3(x: f64) -> f64 {
    let a = 3.;
    if x < -a || x > a {
        return 0.;
    }
    if x < 1.0e-6 && x > -1.0e-6 {
        return 1.;
    }
    (a * (PI * x).sin() * (PI * x / a).sin()) / (PI * PI * x * x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selected_binning_filter_is_normalized() {
        let mut width = 0;
        assert_eq!(select_zoom_filter(0, 0.5, &mut width), 0);
        assert_eq!(width, 2);
        assert!((zoom_filt_value(0.0) - 0.5).abs() < 1.0e-12);
    }

    #[test]
    fn box_filter_reduces_byte_and_float_images() {
        let mut width = 0;
        assert_eq!(select_zoom_filter(0, 0.5, &mut width), 0);

        let byte_input = [
            10_u8, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160,
        ];
        let byte_lines: Vec<&[u8]> = (0..4).map(|i| &byte_input[i * 4..]).collect();
        let mut byte_out = [0_u8; 4];
        assert_eq!(
            zoom_with_filter(
                ZoomLines::Byte(&byte_lines),
                4,
                4,
                0.,
                0.,
                2,
                2,
                2,
                0,
                SLICE_MODE_BYTE,
                &mut ZoomOut::Byte(&mut byte_out),
                None,
                None,
            ),
            0
        );
        assert_eq!(byte_out, [35, 55, 115, 135]);

        let float_input = [
            1_f32, 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
        ];
        let float_lines: Vec<&[f32]> = (0..4).map(|i| &float_input[i * 4..]).collect();
        let mut float_out = [0_f32; 4];
        assert_eq!(
            zoom_with_filter(
                ZoomLines::Float(&float_lines),
                4,
                4,
                0.,
                0.,
                2,
                2,
                2,
                0,
                SLICE_MODE_FLOAT,
                &mut ZoomOut::Float(&mut float_out),
                None,
                None,
            ),
            0
        );
        assert_eq!(float_out, [3.5, 5.5, 11.5, 13.5]);
    }

    #[test]
    fn contiguous_float_slice_entry_point_reduces_image() {
        let mut width = 0;
        assert_eq!(select_zoom_filter(0, 0.5, &mut width), 0);
        let input = [
            1_f32, 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
        ];
        let mut output = [0_f32; 4];

        assert_eq!(
            zoomwithfilter(&input, &4, &4, &0., &0., &2, &2, &2, &0, &mut output,),
            0
        );
        assert_eq!(output, [3.5, 5.5, 11.5, 13.5]);
    }
}
