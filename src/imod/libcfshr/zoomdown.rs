//! Direct, function-for-function translation of `IMOD/libcfshr/zoomdown.c`.
//!
//! The public routines retain the source's C-compatible image-buffer interface because
//! all supported pixel formats are selected at runtime by `dtype`.
#![allow(
    non_snake_case,
    non_camel_case_types,
    non_upper_case_globals,
    dead_code,
    unused_variables,
    static_mut_refs
)]
#[repr(C)]
pub struct _IO_wide_data {
    _private: [u8; 0],
}
#[repr(C)]
pub struct _IO_codecvt {
    _private: [u8; 0],
}
#[repr(C)]
pub struct _IO_marker {
    _private: [u8; 0],
}
unsafe extern "C" {
    static mut stderr: *mut FILE;
    fn fprintf(
        __stream: *mut FILE,
        __format: *const ::core::ffi::c_char,
        ...
    ) -> ::core::ffi::c_int;
    fn cos(__x: ::core::ffi::c_double) -> ::core::ffi::c_double;
    fn sin(__x: ::core::ffi::c_double) -> ::core::ffi::c_double;
    fn sqrt(__x: ::core::ffi::c_double) -> ::core::ffi::c_double;
    fn ceil(__x: ::core::ffi::c_double) -> ::core::ffi::c_double;
    fn fabs(__x: ::core::ffi::c_double) -> ::core::ffi::c_double;
    fn floor(__x: ::core::ffi::c_double) -> ::core::ffi::c_double;
    fn malloc(__size: size_t) -> *mut ::core::ffi::c_void;
    fn free(__ptr: *mut ::core::ffi::c_void);
    fn num_omp_threads(optimalThreads: ::core::ffi::c_int) -> ::core::ffi::c_int;
    fn b3d_omp_thread_num() -> ::core::ffi::c_int;
}
pub type size_t = usize;
pub type __uint16_t = u16;
pub type __uint32_t = u32;
pub type __uint64_t = u64;
pub type __off_t = ::core::ffi::c_long;
pub type __off64_t = ::core::ffi::c_long;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _IO_FILE {
    pub _flags: ::core::ffi::c_int,
    pub _IO_read_ptr: *mut ::core::ffi::c_char,
    pub _IO_read_end: *mut ::core::ffi::c_char,
    pub _IO_read_base: *mut ::core::ffi::c_char,
    pub _IO_write_base: *mut ::core::ffi::c_char,
    pub _IO_write_ptr: *mut ::core::ffi::c_char,
    pub _IO_write_end: *mut ::core::ffi::c_char,
    pub _IO_buf_base: *mut ::core::ffi::c_char,
    pub _IO_buf_end: *mut ::core::ffi::c_char,
    pub _IO_save_base: *mut ::core::ffi::c_char,
    pub _IO_backup_base: *mut ::core::ffi::c_char,
    pub _IO_save_end: *mut ::core::ffi::c_char,
    pub _markers: *mut _IO_marker,
    pub _chain: *mut _IO_FILE,
    pub _fileno: ::core::ffi::c_int,
    pub _flags2: ::core::ffi::c_int,
    pub _old_offset: __off_t,
    pub _cur_column: ::core::ffi::c_ushort,
    pub _vtable_offset: ::core::ffi::c_schar,
    pub _shortbuf: [::core::ffi::c_char; 1],
    pub _lock: *mut ::core::ffi::c_void,
    pub _offset: __off64_t,
    pub _codecvt: *mut _IO_codecvt,
    pub _wide_data: *mut _IO_wide_data,
    pub _freeres_list: *mut _IO_FILE,
    pub _freeres_buf: *mut ::core::ffi::c_void,
    pub __pad5: size_t,
    pub _mode: ::core::ffi::c_int,
    pub _unused2: [::core::ffi::c_char; 20],
}
pub type _IO_lock_t = ();
pub type FILE = _IO_FILE;
pub type b3dInt16 = ::core::ffi::c_short;
pub type b3dUInt16 = ::core::ffi::c_ushort;
pub type b3dInt32 = ::core::ffi::c_int;
pub type b3dUInt32 = ::core::ffi::c_uint;
pub type b3dFloat = ::core::ffi::c_float;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct Filt {
    pub func: fn_proc,
    pub supp: ::core::ffi::c_double,
}
pub type fn_proc = Option<unsafe extern "C" fn(::core::ffi::c_double) -> ::core::ffi::c_double>;
#[derive(Copy, Clone)]
#[repr(C)]
pub union C2RustUnnamed {
    pub s: *mut ::core::ffi::c_short,
    pub f: *mut ::core::ffi::c_float,
}
#[derive(Copy, Clone)]
#[repr(C)]
pub struct Weighttab {
    pub i0: ::core::ffi::c_int,
    pub i1: ::core::ffi::c_int,
    pub weight: C2RustUnnamed,
}
pub const NULL: *mut ::core::ffi::c_void = ::core::ptr::null_mut::<::core::ffi::c_void>();
#[inline]
unsafe extern "C" fn __bswap_16(mut __bsx: __uint16_t) -> __uint16_t {
    return (__bsx as ::core::ffi::c_int >> 8 as ::core::ffi::c_int & 0xff as ::core::ffi::c_int
        | (__bsx as ::core::ffi::c_int & 0xff as ::core::ffi::c_int) << 8 as ::core::ffi::c_int)
        as __uint16_t;
}
#[inline]
unsafe extern "C" fn __bswap_32(mut __bsx: __uint32_t) -> __uint32_t {
    return (__bsx & 0xff000000 as __uint32_t) >> 24 as ::core::ffi::c_int
        | (__bsx & 0xff0000 as __uint32_t) >> 8 as ::core::ffi::c_int
        | (__bsx & 0xff00 as __uint32_t) << 8 as ::core::ffi::c_int
        | (__bsx & 0xff as __uint32_t) << 24 as ::core::ffi::c_int;
}
#[inline]
unsafe extern "C" fn __bswap_64(mut __bsx: __uint64_t) -> __uint64_t {
    return ((__bsx as ::core::ffi::c_ulonglong & 0xff00000000000000 as ::core::ffi::c_ulonglong)
        >> 56 as ::core::ffi::c_int
        | (__bsx as ::core::ffi::c_ulonglong & 0xff000000000000 as ::core::ffi::c_ulonglong)
            >> 40 as ::core::ffi::c_int
        | (__bsx as ::core::ffi::c_ulonglong & 0xff0000000000 as ::core::ffi::c_ulonglong)
            >> 24 as ::core::ffi::c_int
        | (__bsx as ::core::ffi::c_ulonglong & 0xff00000000 as ::core::ffi::c_ulonglong)
            >> 8 as ::core::ffi::c_int
        | (__bsx as ::core::ffi::c_ulonglong & 0xff000000 as ::core::ffi::c_ulonglong)
            << 8 as ::core::ffi::c_int
        | (__bsx as ::core::ffi::c_ulonglong & 0xff0000 as ::core::ffi::c_ulonglong)
            << 24 as ::core::ffi::c_int
        | (__bsx as ::core::ffi::c_ulonglong & 0xff00 as ::core::ffi::c_ulonglong)
            << 40 as ::core::ffi::c_int
        | (__bsx as ::core::ffi::c_ulonglong & 0xff as ::core::ffi::c_ulonglong)
            << 56 as ::core::ffi::c_int) as __uint64_t;
}
#[inline]
unsafe extern "C" fn __uint16_identity(mut __x: __uint16_t) -> __uint16_t {
    return __x;
}
#[inline]
unsafe extern "C" fn __uint32_identity(mut __x: __uint32_t) -> __uint32_t {
    return __x;
}
#[inline]
unsafe extern "C" fn __uint64_identity(mut __x: __uint64_t) -> __uint64_t {
    return __x;
}
pub const SLICE_MODE_BYTE: ::core::ffi::c_int = 0;
pub const SLICE_MODE_SHORT: ::core::ffi::c_int = 1;
pub const SLICE_MODE_FLOAT: ::core::ffi::c_int = 2;
pub const SLICE_MODE_USHORT: ::core::ffi::c_int = 6;
pub const SLICE_MODE_RGB: ::core::ffi::c_int = 16;
pub const PI: ::core::ffi::c_double = 3.14159265358979323846264338f64;
pub const CHANBITS: ::core::ffi::c_int = 8 as ::core::ffi::c_int;
pub const WEIGHTBITS: ::core::ffi::c_int = 14 as ::core::ffi::c_int;
pub const FINALSHIFT: ::core::ffi::c_int = 2 as ::core::ffi::c_int * WEIGHTBITS - CHANBITS;
pub const WEIGHTONE: ::core::ffi::c_int = (1 as ::core::ffi::c_int) << WEIGHTBITS;
pub const NUM_FILT: ::core::ffi::c_int = 6 as ::core::ffi::c_int;
static mut filters: [Filt; 6] = unsafe {
    [
        Filt {
            func: Some(
                filt_binning
                    as unsafe extern "C" fn(::core::ffi::c_double) -> ::core::ffi::c_double,
            ),
            supp: 0.5f64,
        },
        Filt {
            func: Some(
                filt_blackman
                    as unsafe extern "C" fn(::core::ffi::c_double) -> ::core::ffi::c_double,
            ),
            supp: 1.0f64,
        },
        Filt {
            func: Some(
                filt_triangle
                    as unsafe extern "C" fn(::core::ffi::c_double) -> ::core::ffi::c_double,
            ),
            supp: 1.0f64,
        },
        Filt {
            func: Some(
                filt_mitchell
                    as unsafe extern "C" fn(::core::ffi::c_double) -> ::core::ffi::c_double,
            ),
            supp: 2.0f64,
        },
        Filt {
            func: Some(
                filt_lanczos2
                    as unsafe extern "C" fn(::core::ffi::c_double) -> ::core::ffi::c_double,
            ),
            supp: 2.0f64,
        },
        Filt {
            func: Some(
                filt_lanczos3
                    as unsafe extern "C" fn(::core::ffi::c_double) -> ::core::ffi::c_double,
            ),
            supp: 3.0f64,
        },
    ]
};
static mut zoom_debug: ::core::ffi::c_int = 0 as ::core::ffi::c_int;
static mut sXsupport: ::core::ffi::c_double = 0.;
static mut sYsupport: ::core::ffi::c_double = 0.;
static mut sXscale: ::core::ffi::c_double = 0.;
static mut sYscale: ::core::ffi::c_double = 0.;
static mut sYwidth: ::core::ffi::c_int = 0;
static mut sXwidth: ::core::ffi::c_int = 0;
static mut sValueScaling: ::core::ffi::c_float = 1.0f32;
static mut sFilt_func: fn_proc = None;
static mut sMitchP2: ::core::ffi::c_double = 0.;
static mut sMitchP0: ::core::ffi::c_double = 0.;
static mut sMitchQ0: ::core::ffi::c_double = 0.;
static mut sMitchP3: ::core::ffi::c_double = 0.;
static mut sMitchQ2: ::core::ffi::c_double = 0.;
static mut sMitchQ1: ::core::ffi::c_double = 0.;
static mut sMitchQ3: ::core::ffi::c_double = 0.;
#[unsafe(export_name = "selectZoomFilter")]
pub unsafe extern "C" fn select_zoom_filter(
    mut type_0: ::core::ffi::c_int,
    mut zoom: ::core::ffi::c_double,
    mut outWidth: *mut ::core::ffi::c_int,
) -> ::core::ffi::c_int {
    sFilt_func = None;
    if type_0 < 0 as ::core::ffi::c_int || type_0 >= NUM_FILT {
        return 1 as ::core::ffi::c_int;
    }
    if zoom >= 1.0f64 || zoom <= 0.0f64 {
        return 2 as ::core::ffi::c_int;
    }
    sFilt_func = filters[type_0 as usize].func;
    set_filter_statics(
        type_0,
        zoom,
        &raw mut sXscale,
        &raw mut sXsupport,
        &raw mut sXwidth,
        outWidth,
    );
    set_filter_statics(
        type_0,
        zoom,
        &raw mut sYscale,
        &raw mut sYsupport,
        &raw mut sYwidth,
        outWidth,
    );
    if type_0 == 3 as ::core::ffi::c_int {
        mitchell_init(1.0f64 / 3.0f64, 1.0f64 / 3.0f64);
    }
    return 0 as ::core::ffi::c_int;
}
unsafe extern "C" fn set_filter_statics(
    mut type_0: ::core::ffi::c_int,
    mut zoom: ::core::ffi::c_double,
    mut scale: *mut ::core::ffi::c_double,
    mut support: *mut ::core::ffi::c_double,
    mut width: *mut ::core::ffi::c_int,
    mut retWidth: *mut ::core::ffi::c_int,
) {
    *scale = 1.0f64 / zoom;
    *support = filters[type_0 as usize].supp * *scale;
    *width = ceil(2.0f64 * *support) as ::core::ffi::c_int;
    *retWidth = *width;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn selectzoomfilter_(
    mut type_0: *mut ::core::ffi::c_int,
    mut zoom: *mut ::core::ffi::c_float,
    mut outWidth: *mut ::core::ffi::c_int,
) -> ::core::ffi::c_int {
    return select_zoom_filter(*type_0, *zoom as ::core::ffi::c_double, outWidth);
}
#[unsafe(export_name = "selectZoomFilterXY")]
pub unsafe extern "C" fn select_zoom_filter_xy(
    mut type_0: ::core::ffi::c_int,
    mut xzoom: ::core::ffi::c_double,
    mut yzoom: ::core::ffi::c_double,
    mut outWidthX: *mut ::core::ffi::c_int,
    mut outWidthY: *mut ::core::ffi::c_int,
) -> ::core::ffi::c_int {
    let mut err: ::core::ffi::c_int =
        select_zoom_filter(type_0, if xzoom < yzoom { xzoom } else { yzoom }, outWidthX);
    if err != 0 {
        return err;
    }
    if yzoom >= xzoom {
        set_filter_statics(
            type_0,
            yzoom,
            &raw mut sYscale,
            &raw mut sYsupport,
            &raw mut sYwidth,
            outWidthY,
        );
    } else {
        *outWidthY = *outWidthX;
        set_filter_statics(
            type_0,
            xzoom,
            &raw mut sXscale,
            &raw mut sXsupport,
            &raw mut sXwidth,
            outWidthX,
        );
    }
    return 0 as ::core::ffi::c_int;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn selectzoomfilter_xy(
    mut type_0: *mut ::core::ffi::c_int,
    mut xzoom: *mut ::core::ffi::c_float,
    mut yzoom: *mut ::core::ffi::c_float,
    mut outWidthX: *mut ::core::ffi::c_int,
    mut outWidthY: *mut ::core::ffi::c_int,
) -> ::core::ffi::c_int {
    return select_zoom_filter_xy(
        *type_0,
        *xzoom as ::core::ffi::c_double,
        *yzoom as ::core::ffi::c_double,
        outWidthX,
        outWidthY,
    );
}
#[unsafe(export_name = "setZoomValueScaling")]
pub unsafe extern "C" fn set_zoom_value_scaling(mut factor: ::core::ffi::c_float) {
    sValueScaling = factor;
}
#[unsafe(export_name = "zoomWithFilter")]
pub unsafe extern "C" fn zoom_with_filter(
    mut slines: *mut *mut ::core::ffi::c_uchar,
    mut aXsize: ::core::ffi::c_int,
    mut aYsize: ::core::ffi::c_int,
    mut aXoff: ::core::ffi::c_float,
    mut aYoff: ::core::ffi::c_float,
    mut bXsize: ::core::ffi::c_int,
    mut bYsize: ::core::ffi::c_int,
    mut bXdim: ::core::ffi::c_int,
    mut bXoff: ::core::ffi::c_int,
    mut dtype: ::core::ffi::c_int,
    mut outData: *mut ::core::ffi::c_void,
    mut cindex: *mut b3dUInt32,
    mut bindex: *mut ::core::ffi::c_uchar,
) -> ::core::ffi::c_int {
    let mut xweights: *mut Weighttab = ::core::ptr::null_mut::<Weighttab>();
    let mut xweightSbuf: *mut b3dInt16 = ::core::ptr::null_mut::<b3dInt16>();
    let mut xwp: *mut b3dInt16 = ::core::ptr::null_mut::<b3dInt16>();
    let mut yweight: [Weighttab; 16] = [Weighttab {
        i0: 0,
        i1: 0,
        weight: C2RustUnnamed {
            s: ::core::ptr::null_mut::<::core::ffi::c_short>(),
        },
    }; 16];
    let mut xweightFbuf: *mut b3dFloat = ::core::ptr::null_mut::<b3dFloat>();
    let mut xwfp: *mut b3dFloat = ::core::ptr::null_mut::<b3dFloat>();
    let mut accumBuf: [*mut b3dInt32; 16] = [::core::ptr::null_mut::<b3dInt32>(); 16];
    let mut filtBuf: [*mut ::core::ffi::c_uchar; 16] =
        [::core::ptr::null_mut::<::core::ffi::c_uchar>(); 16];
    let mut lineb: *mut ::core::ffi::c_uchar = ::core::ptr::null_mut::<::core::ffi::c_uchar>();
    let mut obufb: *mut ::core::ffi::c_uchar = ::core::ptr::null_mut::<::core::ffi::c_uchar>();
    let mut mapping: ::core::ffi::c_int = 0 as ::core::ffi::c_int;
    let mut numThreads: ::core::ffi::c_int = 0;
    let mut psizeAccum: ::core::ffi::c_int = 0;
    let mut psizeWgt: ::core::ffi::c_int = 0;
    let mut psizeFilt: ::core::ffi::c_int = 0;
    let mut bx: ::core::ffi::c_int = 0;
    let mut by: ::core::ffi::c_int = 0;
    let mut i: ::core::ffi::c_int = 0;
    let mut ayf: ::core::ffi::c_int = 0;
    let mut thr: ::core::ffi::c_int = 0;
    let mut fweight: ::core::ffi::c_float = 0.0f32;
    let mut sweight: b3dInt16 = 0 as b3dInt16;
    if sFilt_func.is_none() {
        return 1 as ::core::ffi::c_int;
    }
    psizeAccum = 4 as ::core::ffi::c_int;
    psizeWgt = 2 as ::core::ffi::c_int;
    match dtype {
        SLICE_MODE_BYTE => {
            if !cindex.is_null() {
                mapping = 1 as ::core::ffi::c_int;
            }
            psizeFilt = 1 as ::core::ffi::c_int;
        }
        SLICE_MODE_RGB => {
            if !bindex.is_null() {
                mapping = 1 as ::core::ffi::c_int;
            }
            psizeAccum = 12 as ::core::ffi::c_int;
            psizeFilt = 3 as ::core::ffi::c_int;
        }
        SLICE_MODE_FLOAT => {
            if !cindex.is_null() || !bindex.is_null() {
                return 3 as ::core::ffi::c_int;
            }
            psizeFilt = 4 as ::core::ffi::c_int;
            psizeWgt = 4 as ::core::ffi::c_int;
        }
        SLICE_MODE_SHORT | SLICE_MODE_USHORT => {
            if !cindex.is_null() {
                mapping = 1 as ::core::ffi::c_int;
            }
            psizeFilt = 2 as ::core::ffi::c_int;
            psizeWgt = 4 as ::core::ffi::c_int;
        }
        -18 | -16 => {
            if !cindex.is_null() || !bindex.is_null() {
                return 3 as ::core::ffi::c_int;
            }
            psizeAccum = 12 as ::core::ffi::c_int;
            psizeFilt = 4 as ::core::ffi::c_int;
        }
        -19 | -17 => {
            if !cindex.is_null() || !bindex.is_null() {
                return 3 as ::core::ffi::c_int;
            }
            psizeFilt = 4 as ::core::ffi::c_int;
        }
        _ => return 2 as ::core::ffi::c_int,
    }
    if dtype < 0 as ::core::ffi::c_int && mapping != 0 {
        return 3 as ::core::ffi::c_int;
    }
    if (aXoff as ::core::ffi::c_double) < 0.0f64
        || (aYoff as ::core::ffi::c_double) < 0.0f64
        || (aXoff as ::core::ffi::c_double + bXsize as ::core::ffi::c_double * sXscale)
            as ::core::ffi::c_int
            > aXsize
        || (aYoff as ::core::ffi::c_double + bYsize as ::core::ffi::c_double * sYscale)
            as ::core::ffi::c_int
            > aYsize
    {
        return 4 as ::core::ffi::c_int;
    }
    numThreads = floor(
        0.04f64 * sqrt(aXsize as ::core::ffi::c_double * aYsize as ::core::ffi::c_double) + 0.5f64,
    ) as ::core::ffi::c_int;
    numThreads = num_omp_threads(if numThreads < 16 as ::core::ffi::c_int {
        numThreads
    } else {
        16 as ::core::ffi::c_int
    });
    numThreads = if numThreads < 16 as ::core::ffi::c_int {
        numThreads
    } else {
        16 as ::core::ffi::c_int
    };
    xweights =
        malloc((bXsize as size_t).wrapping_mul(::core::mem::size_of::<Weighttab>() as size_t))
            as *mut Weighttab;
    xweightSbuf = malloc((bXsize * sXwidth * psizeWgt) as size_t) as *mut b3dInt16;
    if xweightSbuf.is_null() || xweights.is_null() {
        free(xweights as *mut ::core::ffi::c_void);
        xweights = ::core::ptr::null_mut::<Weighttab>();
        free(xweightSbuf as *mut ::core::ffi::c_void);
        xweightSbuf = ::core::ptr::null_mut::<b3dInt16>();
        return 5 as ::core::ffi::c_int;
    }
    i = 0 as ::core::ffi::c_int;
    while i < numThreads {
        filtBuf[i as usize] = ::core::ptr::null_mut::<::core::ffi::c_uchar>();
        accumBuf[i as usize] = malloc((aXsize * psizeAccum) as size_t) as *mut b3dInt32;
        yweight[i as usize].weight.s =
            malloc((sYwidth * psizeWgt) as size_t) as *mut b3dInt16 as *mut ::core::ffi::c_short;
        if mapping != 0 {
            filtBuf[i as usize] =
                malloc((bXsize * psizeFilt) as size_t) as *mut ::core::ffi::c_uchar;
        }
        if accumBuf[i as usize].is_null()
            || yweight[i as usize].weight.s.is_null()
            || mapping != 0 && filtBuf[i as usize].is_null()
        {
            by = 0 as ::core::ffi::c_int;
            while by <= i {
                free(yweight[by as usize].weight.s as *mut ::core::ffi::c_void);
                yweight[by as usize].weight.s = ::core::ptr::null_mut::<::core::ffi::c_short>();
                free(accumBuf[by as usize] as *mut ::core::ffi::c_void);
                accumBuf[by as usize] = ::core::ptr::null_mut::<b3dInt32>();
                free(filtBuf[by as usize] as *mut ::core::ffi::c_void);
                filtBuf[by as usize] = ::core::ptr::null_mut::<::core::ffi::c_uchar>();
                by += 1;
            }
            return 5 as ::core::ffi::c_int;
        }
        i += 1;
    }
    xweightFbuf = xweightSbuf as *mut b3dFloat;
    xwfp = xweightFbuf;
    xwp = xweightSbuf;
    bx = 0 as ::core::ffi::c_int;
    while bx < bXsize {
        if psizeWgt == 4 as ::core::ffi::c_int {
            let ref mut fresh0 = (*xweights.offset(bx as isize)).weight.f;
            *fresh0 = xwfp as *mut ::core::ffi::c_float;
        } else {
            let ref mut fresh1 = (*xweights.offset(bx as isize)).weight.s;
            *fresh1 = xwp as *mut ::core::ffi::c_short;
        }
        make_weighttab(
            bx,
            aXoff as ::core::ffi::c_double + (bx as ::core::ffi::c_double + 0.5f64) * sXscale,
            aXsize,
            sXscale,
            sXsupport,
            dtype,
            xweights.offset(bx as isize) as *mut Weighttab,
        );
        bx += 1;
        xwp = xwp.offset(sXwidth as isize);
        xwfp = xwfp.offset(sXwidth as isize);
    }
    by = 0 as ::core::ffi::c_int;
    while by < bYsize {
        thr = b3d_omp_thread_num();
        make_weighttab(
            by,
            aYoff as ::core::ffi::c_double + (by as ::core::ffi::c_double + 0.5f64) * sYscale,
            aYsize,
            sYscale,
            sYsupport,
            dtype,
            (&raw mut yweight as *mut Weighttab).offset(thr as isize) as *mut Weighttab,
        );
        i = 0 as ::core::ffi::c_int;
        while i < aXsize * psizeAccum / 4 as ::core::ffi::c_int {
            *accumBuf[thr as usize].offset(i as isize) = 0 as ::core::ffi::c_int as b3dInt32;
            i += 1;
        }
        ayf = yweight[thr as usize].i0;
        while ayf < yweight[thr as usize].i1 {
            if psizeWgt == 2 as ::core::ffi::c_int {
                sweight = *yweight[thr as usize]
                    .weight
                    .s
                    .offset((ayf - yweight[thr as usize].i0) as isize)
                    as b3dInt16;
            } else {
                fweight = *yweight[thr as usize]
                    .weight
                    .f
                    .offset((ayf - yweight[thr as usize].i0) as isize)
                    * sValueScaling;
            }
            lineb = *slines.offset(ayf as isize);
            scanline_accum(
                lineb,
                dtype,
                aXsize,
                accumBuf[thr as usize],
                sweight,
                fweight,
            );
            ayf += 1;
        }
        obufb = (outData as *mut ::core::ffi::c_uchar)
            .offset((psizeFilt * (by * bXdim + bXoff)) as isize);
        if mapping != 0 {
            obufb = filtBuf[thr as usize];
        }
        scanline_filter(
            accumBuf[thr as usize],
            dtype,
            aXsize,
            obufb,
            bXsize,
            xweights,
            FINALSHIFT,
        );
        if mapping != 0 {
            obufb = (outData as *mut ::core::ffi::c_uchar)
                .offset((4 as ::core::ffi::c_int * (by * bXdim + bXoff)) as isize);
            scanline_remap(filtBuf[thr as usize], dtype, bXsize, obufb, cindex, bindex);
        }
        by += 1;
    }
    free(xweights as *mut ::core::ffi::c_void);
    xweights = ::core::ptr::null_mut::<Weighttab>();
    free(xweightSbuf as *mut ::core::ffi::c_void);
    xweightSbuf = ::core::ptr::null_mut::<b3dInt16>();
    by = 0 as ::core::ffi::c_int;
    while by < numThreads {
        free(yweight[by as usize].weight.s as *mut ::core::ffi::c_void);
        yweight[by as usize].weight.s = ::core::ptr::null_mut::<::core::ffi::c_short>();
        free(accumBuf[by as usize] as *mut ::core::ffi::c_void);
        accumBuf[by as usize] = ::core::ptr::null_mut::<b3dInt32>();
        free(filtBuf[by as usize] as *mut ::core::ffi::c_void);
        filtBuf[by as usize] = ::core::ptr::null_mut::<::core::ffi::c_uchar>();
        by += 1;
    }
    return 0 as ::core::ffi::c_int;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zoomwithfilter_(
    mut array: *mut ::core::ffi::c_float,
    mut aXsize: *mut ::core::ffi::c_int,
    mut aYsize: *mut ::core::ffi::c_int,
    mut aXoff: *mut ::core::ffi::c_float,
    mut aYoff: *mut ::core::ffi::c_float,
    mut bXsize: *mut ::core::ffi::c_int,
    mut bYsize: *mut ::core::ffi::c_int,
    mut bXdim: *mut ::core::ffi::c_int,
    mut bXoff: *mut ::core::ffi::c_int,
    mut outData: *mut ::core::ffi::c_float,
) -> ::core::ffi::c_int {
    let mut i: ::core::ffi::c_int = 0;
    let mut linePtrs: *mut *mut ::core::ffi::c_uchar =
        crate::imod::libcfshr::b3dutil::make_line_pointers(
            array as *mut ::core::ffi::c_void,
            *aXsize,
            *aYsize,
            4 as ::core::ffi::c_int,
        );
    if linePtrs.is_null() {
        return 5 as ::core::ffi::c_int;
    }
    i = zoom_with_filter(
        linePtrs,
        *aXsize,
        *aYsize,
        *aXoff,
        *aYoff,
        *bXsize,
        *bYsize,
        *bXdim,
        *bXoff,
        SLICE_MODE_FLOAT,
        outData as *mut ::core::ffi::c_void,
        ::core::ptr::null_mut::<b3dUInt32>(),
        ::core::ptr::null_mut::<::core::ffi::c_uchar>(),
    );
    free(linePtrs as *mut ::core::ffi::c_void);
    return i;
}
#[unsafe(export_name = "zoomFiltInterp")]
pub unsafe extern "C" fn zoom_filt_interp(
    mut array: *mut ::core::ffi::c_float,
    mut bray: *mut ::core::ffi::c_float,
    mut nxa: ::core::ffi::c_int,
    mut nya: ::core::ffi::c_int,
    mut nxb: ::core::ffi::c_int,
    mut nyb: ::core::ffi::c_int,
    mut xc: ::core::ffi::c_float,
    mut yc: ::core::ffi::c_float,
    mut xt: ::core::ffi::c_float,
    mut yt: ::core::ffi::c_float,
    mut dmean: ::core::ffi::c_float,
) -> ::core::ffi::c_int {
    let mut i: ::core::ffi::c_int = 0;
    let mut bXsize: ::core::ffi::c_int = 0;
    let mut bYsize: ::core::ffi::c_int = 0;
    let mut bXoff: ::core::ffi::c_int = 0;
    let mut bYoff: ::core::ffi::c_int = 0;
    let mut ix: ::core::ffi::c_int = 0;
    let mut iy: ::core::ffi::c_int = 0;
    let mut aXoff: ::core::ffi::c_float = 0.;
    let mut aYoff: ::core::ffi::c_float = 0.;
    let mut linePtrs: *mut *mut ::core::ffi::c_uchar =
        crate::imod::libcfshr::b3dutil::make_line_pointers(
            array as *mut ::core::ffi::c_void,
            nxa,
            nya,
            4 as ::core::ffi::c_int,
        );
    if linePtrs.is_null() {
        return 5 as ::core::ffi::c_int;
    }
    interp_limits(
        nxa,
        nxb,
        xc,
        xt,
        sXscale,
        &raw mut aXoff,
        &raw mut bXsize,
        &raw mut bXoff,
    );
    interp_limits(
        nya,
        nyb,
        yc,
        yt,
        sYscale,
        &raw mut aYoff,
        &raw mut bYsize,
        &raw mut bYoff,
    );
    if bXsize > 0 as ::core::ffi::c_int && bYsize > 0 as ::core::ffi::c_int {
        i = zoom_with_filter(
            linePtrs,
            nxa,
            nya,
            aXoff,
            aYoff,
            bXsize,
            bYsize,
            nxb,
            bXoff,
            SLICE_MODE_FLOAT,
            bray.offset((nxb * bYoff) as isize) as *mut ::core::ffi::c_float
                as *mut ::core::ffi::c_void,
            ::core::ptr::null_mut::<b3dUInt32>(),
            ::core::ptr::null_mut::<::core::ffi::c_uchar>(),
        );
        if i != 0 {
            free(linePtrs as *mut ::core::ffi::c_void);
            return i;
        }
    }
    iy = 0 as ::core::ffi::c_int;
    while iy < nyb {
        if iy < bYoff || iy >= bYsize + bYoff {
            ix = 0 as ::core::ffi::c_int;
            while ix < nxb {
                *bray.offset((ix + iy * nxb) as isize) = dmean;
                ix += 1;
            }
        } else {
            ix = 0 as ::core::ffi::c_int;
            while ix < bXoff {
                *bray.offset((ix + iy * nxb) as isize) = dmean;
                ix += 1;
            }
            ix = bXsize + bXoff;
            while ix < nxb {
                *bray.offset((ix + iy * nxb) as isize) = dmean;
                ix += 1;
            }
        }
        iy += 1;
    }
    free(linePtrs as *mut ::core::ffi::c_void);
    return 0 as ::core::ffi::c_int;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zoomfiltinterp_(
    mut array: *mut ::core::ffi::c_float,
    mut bray: *mut ::core::ffi::c_float,
    mut nxa: *mut ::core::ffi::c_int,
    mut nya: *mut ::core::ffi::c_int,
    mut nxb: *mut ::core::ffi::c_int,
    mut nyb: *mut ::core::ffi::c_int,
    mut xc: *mut ::core::ffi::c_float,
    mut yc: *mut ::core::ffi::c_float,
    mut xt: *mut ::core::ffi::c_float,
    mut yt: *mut ::core::ffi::c_float,
    mut dmean: *mut ::core::ffi::c_float,
) -> ::core::ffi::c_int {
    return zoom_filt_interp(
        array, bray, *nxa, *nya, *nxb, *nyb, *xc, *yc, *xt, *yt, *dmean,
    );
}
#[unsafe(export_name = "zoomFiltValue")]
pub unsafe extern "C" fn zoom_filt_value(
    mut radius: ::core::ffi::c_float,
) -> ::core::ffi::c_double {
    let mut den: ::core::ffi::c_double = 0.0f64;
    let mut i: ::core::ffi::c_int = 0;
    let mut lim: ::core::ffi::c_int = 0;
    if sFilt_func.is_none() {
        return 0.0f64;
    }
    lim = (sXwidth + 1 as ::core::ffi::c_int) / 2 as ::core::ffi::c_int;
    i = -lim;
    while i <= lim {
        den += sFilt_func.expect("non-null function pointer")(i as ::core::ffi::c_double / sXscale);
        i += 1;
    }
    return sFilt_func.expect("non-null function pointer")(
        radius as ::core::ffi::c_double / sXscale,
    ) / den;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zoomfiltvalue_(
    mut radius: *mut ::core::ffi::c_float,
) -> ::core::ffi::c_double {
    return zoom_filt_value(*radius);
}
#[unsafe(export_name = "zoomRawFiltValue")]
pub unsafe extern "C" fn zoom_raw_filt_value(
    mut radius: ::core::ffi::c_float,
) -> ::core::ffi::c_double {
    return sFilt_func.expect("non-null function pointer")(radius as ::core::ffi::c_double);
}
unsafe extern "C" fn interp_limits(
    mut na: ::core::ffi::c_int,
    mut nb: ::core::ffi::c_int,
    mut cen: ::core::ffi::c_float,
    mut trans: ::core::ffi::c_float,
    mut scale: ::core::ffi::c_double,
    mut aOff: *mut ::core::ffi::c_float,
    mut bSize: *mut ::core::ffi::c_int,
    mut bOff: *mut ::core::ffi::c_int,
) {
    let mut bcStart: ::core::ffi::c_float = 0.;
    let mut bcEnd: ::core::ffi::c_float = 0.;
    let mut bdStart: ::core::ffi::c_int = 0;
    let mut bdEnd: ::core::ffi::c_int = 0;
    bcStart = (-cen as ::core::ffi::c_double / scale
        + nb as ::core::ffi::c_double / 2.0f64
        + trans as ::core::ffi::c_double) as ::core::ffi::c_float;
    bcEnd = ((na as ::core::ffi::c_float - cen) as ::core::ffi::c_double / scale
        + nb as ::core::ffi::c_double / 2.0f64
        + trans as ::core::ffi::c_double) as ::core::ffi::c_float;
    bcStart = (if 0.0f64 > bcStart as ::core::ffi::c_double {
        0.0f64
    } else {
        bcStart as ::core::ffi::c_double
    }) as ::core::ffi::c_float;
    bcEnd = if (nb as ::core::ffi::c_float) < bcEnd {
        nb as ::core::ffi::c_float
    } else {
        bcEnd
    };
    bdStart = ceil(bcStart as ::core::ffi::c_double - 0.001f64) as ::core::ffi::c_int;
    bdEnd = floor(bcEnd as ::core::ffi::c_double + 0.001f64) as ::core::ffi::c_int
        - 1 as ::core::ffi::c_int;
    *bOff = bdStart;
    *aOff = (scale
        * (bdStart as ::core::ffi::c_double
            - nb as ::core::ffi::c_double / 2.0f64
            - trans as ::core::ffi::c_double)
        + cen as ::core::ffi::c_double) as ::core::ffi::c_float;
    *aOff = (if 0.0f64 > *aOff as ::core::ffi::c_double {
        0.0f64
    } else {
        *aOff as ::core::ffi::c_double
    }) as ::core::ffi::c_float;
    *bSize = bdEnd + 1 as ::core::ffi::c_int - bdStart;
    if *bSize as ::core::ffi::c_double * scale + *aOff as ::core::ffi::c_double
        > na as ::core::ffi::c_double
    {
        *bSize -= 1;
    }
}
unsafe extern "C" fn scanline_accum(
    mut lineb: *mut ::core::ffi::c_uchar,
    mut dtype: ::core::ffi::c_int,
    mut aXsize: ::core::ffi::c_int,
    mut accumBuf: *mut b3dInt32,
    mut sweight: b3dInt16,
    mut fweight: ::core::ffi::c_float,
) {
    let mut i: ::core::ffi::c_int = 0;
    let mut linef: *mut b3dFloat = ::core::ptr::null_mut::<b3dFloat>();
    let mut accumFbuf: *mut b3dFloat = accumBuf as *mut b3dFloat;
    let mut lines: *mut b3dInt16 = ::core::ptr::null_mut::<b3dInt16>();
    let mut lineus: *mut b3dUInt16 = ::core::ptr::null_mut::<b3dUInt16>();
    match dtype {
        SLICE_MODE_BYTE => {
            i = 0 as ::core::ffi::c_int;
            while i < aXsize {
                let fresh46 = lineb;
                lineb = lineb.offset(1);
                let fresh47 = accumBuf;
                accumBuf = accumBuf.offset(1);
                *fresh47 += sweight as ::core::ffi::c_int * *fresh46 as ::core::ffi::c_int;
                i += 1;
            }
        }
        SLICE_MODE_RGB => {
            i = 0 as ::core::ffi::c_int;
            while i < aXsize {
                let fresh48 = lineb;
                lineb = lineb.offset(1);
                let fresh49 = accumBuf;
                accumBuf = accumBuf.offset(1);
                *fresh49 += sweight as ::core::ffi::c_int * *fresh48 as ::core::ffi::c_int;
                let fresh50 = lineb;
                lineb = lineb.offset(1);
                let fresh51 = accumBuf;
                accumBuf = accumBuf.offset(1);
                *fresh51 += sweight as ::core::ffi::c_int * *fresh50 as ::core::ffi::c_int;
                let fresh52 = lineb;
                lineb = lineb.offset(1);
                let fresh53 = accumBuf;
                accumBuf = accumBuf.offset(1);
                *fresh53 += sweight as ::core::ffi::c_int * *fresh52 as ::core::ffi::c_int;
                i += 1;
            }
        }
        SLICE_MODE_FLOAT => {
            linef = lineb as *mut b3dFloat;
            i = 0 as ::core::ffi::c_int;
            while i < aXsize {
                let fresh54 = linef;
                linef = linef.offset(1);
                let ref mut fresh55 = *accumFbuf.offset(i as isize);
                *fresh55 += (fweight as b3dFloat * *fresh54) as ::core::ffi::c_float;
                i += 1;
            }
        }
        SLICE_MODE_SHORT => {
            lines = lineb as *mut b3dInt16;
            i = 0 as ::core::ffi::c_int;
            while i < aXsize {
                let fresh56 = lines;
                lines = lines.offset(1);
                let ref mut fresh57 = *accumFbuf.offset(i as isize);
                *fresh57 += fweight * *fresh56 as ::core::ffi::c_int as ::core::ffi::c_float;
                i += 1;
            }
        }
        SLICE_MODE_USHORT => {
            lineus = lineb as *mut b3dUInt16;
            i = 0 as ::core::ffi::c_int;
            while i < aXsize {
                let fresh58 = lineus;
                lineus = lineus.offset(1);
                let ref mut fresh59 = *accumFbuf.offset(i as isize);
                *fresh59 += fweight * *fresh58 as ::core::ffi::c_int as ::core::ffi::c_float;
                i += 1;
            }
        }
        -18 | -16 => {
            i = 0 as ::core::ffi::c_int;
            while i < aXsize {
                let fresh60 = lineb;
                lineb = lineb.offset(1);
                let fresh61 = accumBuf;
                accumBuf = accumBuf.offset(1);
                *fresh61 += sweight as ::core::ffi::c_int * *fresh60 as ::core::ffi::c_int;
                let fresh62 = lineb;
                lineb = lineb.offset(1);
                let fresh63 = accumBuf;
                accumBuf = accumBuf.offset(1);
                *fresh63 += sweight as ::core::ffi::c_int * *fresh62 as ::core::ffi::c_int;
                let fresh64 = lineb;
                lineb = lineb.offset(1);
                let fresh65 = accumBuf;
                accumBuf = accumBuf.offset(1);
                *fresh65 += sweight as ::core::ffi::c_int * *fresh64 as ::core::ffi::c_int;
                lineb = lineb.offset(1);
                i += 1;
            }
        }
        -19 | -17 => {
            i = 0 as ::core::ffi::c_int;
            while i < aXsize {
                let fresh66 = lineb;
                lineb = lineb.offset(1);
                let fresh67 = accumBuf;
                accumBuf = accumBuf.offset(1);
                *fresh67 += sweight as ::core::ffi::c_int * *fresh66 as ::core::ffi::c_int;
                lineb = lineb.offset(3 as ::core::ffi::c_int as isize);
                i += 1;
            }
        }
        _ => {}
    };
}
unsafe extern "C" fn scanline_filter(
    mut lineb: *mut b3dInt32,
    mut dtype: ::core::ffi::c_int,
    mut aXsize: ::core::ffi::c_int,
    mut obufb: *mut ::core::ffi::c_uchar,
    mut bXsize: ::core::ffi::c_int,
    mut wtab: *mut Weighttab,
    mut shift: ::core::ffi::c_int,
) {
    let mut b: ::core::ffi::c_int = 0;
    let mut af: ::core::ffi::c_int = 0;
    let mut sum: ::core::ffi::c_int = 0;
    let mut sumr: ::core::ffi::c_int = 0;
    let mut sumb: ::core::ffi::c_int = 0;
    let mut sumg: ::core::ffi::c_int = 0;
    let mut t: ::core::ffi::c_int = 0;
    let mut alpha: ::core::ffi::c_int = 0 as ::core::ffi::c_int;
    let mut linef: *mut b3dFloat = ::core::ptr::null_mut::<b3dFloat>();
    let mut obuff: *mut b3dFloat = ::core::ptr::null_mut::<b3dFloat>();
    let mut wfp: *mut b3dFloat = ::core::ptr::null_mut::<b3dFloat>();
    let mut afp: *mut b3dFloat = ::core::ptr::null_mut::<b3dFloat>();
    let mut obufs: *mut b3dInt16 = ::core::ptr::null_mut::<b3dInt16>();
    let mut wp: *mut b3dInt16 = ::core::ptr::null_mut::<b3dInt16>();
    let mut obufus: *mut b3dUInt16 = ::core::ptr::null_mut::<b3dUInt16>();
    let mut ap: *mut b3dInt32 = ::core::ptr::null_mut::<b3dInt32>();
    let mut rsum: ::core::ffi::c_float = 0.;
    let mut tempb: ::core::ffi::c_uchar = 0;
    linef = lineb as *mut b3dFloat;
    let mut current_block_71: u64;
    match dtype {
        SLICE_MODE_BYTE => {
            b = 0 as ::core::ffi::c_int;
            while b < bXsize {
                sum = (1 as ::core::ffi::c_int) << shift - 1 as ::core::ffi::c_int;
                wp = (*wtab).weight.s as *mut b3dInt16;
                ap = lineb.offset((*wtab).i0 as isize) as *mut b3dInt32;
                af = (*wtab).i1 - (*wtab).i0;
                while af > 0 as ::core::ffi::c_int {
                    let fresh15 = wp;
                    wp = wp.offset(1);
                    let fresh16 = ap;
                    ap = ap.offset(1);
                    sum += *fresh15 as ::core::ffi::c_int
                        * (*fresh16 >> CHANBITS) as ::core::ffi::c_short as ::core::ffi::c_int;
                    af -= 1;
                }
                t = sum >> shift;
                let fresh17 = obufb;
                obufb = obufb.offset(1);
                *fresh17 = (if t < 0 as ::core::ffi::c_int {
                    0 as ::core::ffi::c_int
                } else {
                    (if t > 255 as ::core::ffi::c_int {
                        255 as ::core::ffi::c_int
                    } else {
                        t
                    })
                }) as ::core::ffi::c_uchar;
                b += 1;
                wtab = wtab.offset(1);
            }
            current_block_71 = 4216521074440650966;
        }
        SLICE_MODE_RGB => {
            b = 0 as ::core::ffi::c_int;
            while b < bXsize {
                sumb = (1 as ::core::ffi::c_int) << shift - 1 as ::core::ffi::c_int;
                sumg = sumb;
                sumr = sumg;
                wp = (*wtab).weight.s as *mut b3dInt16;
                ap = lineb.offset((3 as ::core::ffi::c_int * (*wtab).i0) as isize) as *mut b3dInt32;
                af = (*wtab).i1 - (*wtab).i0;
                while af > 0 as ::core::ffi::c_int {
                    let fresh18 = ap;
                    ap = ap.offset(1);
                    sumr += *wp as ::core::ffi::c_int
                        * (*fresh18 >> CHANBITS) as ::core::ffi::c_short as ::core::ffi::c_int;
                    let fresh19 = ap;
                    ap = ap.offset(1);
                    sumg += *wp as ::core::ffi::c_int
                        * (*fresh19 >> CHANBITS) as ::core::ffi::c_short as ::core::ffi::c_int;
                    let fresh20 = ap;
                    ap = ap.offset(1);
                    sumb += *wp as ::core::ffi::c_int
                        * (*fresh20 >> CHANBITS) as ::core::ffi::c_short as ::core::ffi::c_int;
                    af -= 1;
                    wp = wp.offset(1);
                }
                t = sumr >> shift;
                let fresh21 = obufb;
                obufb = obufb.offset(1);
                *fresh21 = (if t < 0 as ::core::ffi::c_int {
                    0 as ::core::ffi::c_int
                } else {
                    (if t > 255 as ::core::ffi::c_int {
                        255 as ::core::ffi::c_int
                    } else {
                        t
                    })
                }) as ::core::ffi::c_uchar;
                t = sumg >> shift;
                let fresh22 = obufb;
                obufb = obufb.offset(1);
                *fresh22 = (if t < 0 as ::core::ffi::c_int {
                    0 as ::core::ffi::c_int
                } else {
                    (if t > 255 as ::core::ffi::c_int {
                        255 as ::core::ffi::c_int
                    } else {
                        t
                    })
                }) as ::core::ffi::c_uchar;
                t = sumb >> shift;
                let fresh23 = obufb;
                obufb = obufb.offset(1);
                *fresh23 = (if t < 0 as ::core::ffi::c_int {
                    0 as ::core::ffi::c_int
                } else {
                    (if t > 255 as ::core::ffi::c_int {
                        255 as ::core::ffi::c_int
                    } else {
                        t
                    })
                }) as ::core::ffi::c_uchar;
                b += 1;
                wtab = wtab.offset(1);
            }
            current_block_71 = 4216521074440650966;
        }
        SLICE_MODE_FLOAT => {
            obuff = obufb as *mut b3dFloat;
            b = 0 as ::core::ffi::c_int;
            while b < bXsize {
                rsum = 0.0f32;
                wfp = (*wtab).weight.f as *mut b3dFloat;
                afp = linef.offset((*wtab).i0 as isize) as *mut b3dFloat;
                af = (*wtab).i1 - (*wtab).i0;
                while af > 0 as ::core::ffi::c_int {
                    let fresh24 = wfp;
                    wfp = wfp.offset(1);
                    let fresh25 = afp;
                    afp = afp.offset(1);
                    rsum += (*fresh24 * *fresh25) as ::core::ffi::c_float;
                    af -= 1;
                }
                let fresh26 = obuff;
                obuff = obuff.offset(1);
                *fresh26 = rsum as b3dFloat;
                b += 1;
                wtab = wtab.offset(1);
            }
            current_block_71 = 4216521074440650966;
        }
        SLICE_MODE_SHORT => {
            obufs = obufb as *mut b3dInt16;
            b = 0 as ::core::ffi::c_int;
            while b < bXsize {
                rsum = 0.5f32;
                wfp = (*wtab).weight.f as *mut b3dFloat;
                afp = linef.offset((*wtab).i0 as isize) as *mut b3dFloat;
                af = (*wtab).i1 - (*wtab).i0;
                while af > 0 as ::core::ffi::c_int {
                    let fresh27 = wfp;
                    wfp = wfp.offset(1);
                    let fresh28 = afp;
                    afp = afp.offset(1);
                    rsum += (*fresh27 * *fresh28) as ::core::ffi::c_float;
                    af -= 1;
                }
                let fresh29 = obufs;
                obufs = obufs.offset(1);
                *fresh29 = (if 32767.0f64
                    < (if -32767.0f64 > rsum as ::core::ffi::c_double {
                        -32767.0f64
                    } else {
                        rsum as ::core::ffi::c_double
                    }) {
                    32767.0f64
                } else if -32767.0f64 > rsum as ::core::ffi::c_double {
                    -32767.0f64
                } else {
                    rsum as ::core::ffi::c_double
                }) as b3dInt16;
                b += 1;
                wtab = wtab.offset(1);
            }
            current_block_71 = 4216521074440650966;
        }
        SLICE_MODE_USHORT => {
            obufus = obufb as *mut b3dUInt16;
            b = 0 as ::core::ffi::c_int;
            while b < bXsize {
                rsum = 0.5f32;
                wfp = (*wtab).weight.f as *mut b3dFloat;
                afp = linef.offset((*wtab).i0 as isize) as *mut b3dFloat;
                af = (*wtab).i1 - (*wtab).i0;
                while af > 0 as ::core::ffi::c_int {
                    let fresh30 = wfp;
                    wfp = wfp.offset(1);
                    let fresh31 = afp;
                    afp = afp.offset(1);
                    rsum += (*fresh30 * *fresh31) as ::core::ffi::c_float;
                    af -= 1;
                }
                let fresh32 = obufus;
                obufus = obufus.offset(1);
                *fresh32 = (if 65535.0f64
                    < (if 0.0f64 > rsum as ::core::ffi::c_double {
                        0.0f64
                    } else {
                        rsum as ::core::ffi::c_double
                    }) {
                    65535.0f64
                } else if 0.0f64 > rsum as ::core::ffi::c_double {
                    0.0f64
                } else {
                    rsum as ::core::ffi::c_double
                }) as b3dUInt16;
                b += 1;
                wtab = wtab.offset(1);
            }
            current_block_71 = 4216521074440650966;
        }
        -18 => {
            alpha = 255 as ::core::ffi::c_int;
            current_block_71 = 14615920641664762699;
        }
        -16 => {
            current_block_71 = 14615920641664762699;
        }
        -19 => {
            alpha = 255 as ::core::ffi::c_int;
            current_block_71 = 17395913555154045066;
        }
        -17 => {
            current_block_71 = 17395913555154045066;
        }
        _ => {
            current_block_71 = 4216521074440650966;
        }
    }
    match current_block_71 {
        14615920641664762699 => {
            b = 0 as ::core::ffi::c_int;
            while b < bXsize {
                sumb = (1 as ::core::ffi::c_int) << shift - 1 as ::core::ffi::c_int;
                sumg = sumb;
                sumr = sumg;
                wp = (*wtab).weight.s as *mut b3dInt16;
                ap = lineb.offset((3 as ::core::ffi::c_int * (*wtab).i0) as isize) as *mut b3dInt32;
                af = (*wtab).i1 - (*wtab).i0;
                while af > 0 as ::core::ffi::c_int {
                    let fresh33 = ap;
                    ap = ap.offset(1);
                    sumr += *wp as ::core::ffi::c_int
                        * (*fresh33 >> CHANBITS) as ::core::ffi::c_short as ::core::ffi::c_int;
                    let fresh34 = ap;
                    ap = ap.offset(1);
                    sumg += *wp as ::core::ffi::c_int
                        * (*fresh34 >> CHANBITS) as ::core::ffi::c_short as ::core::ffi::c_int;
                    let fresh35 = ap;
                    ap = ap.offset(1);
                    sumb += *wp as ::core::ffi::c_int
                        * (*fresh35 >> CHANBITS) as ::core::ffi::c_short as ::core::ffi::c_int;
                    af -= 1;
                    wp = wp.offset(1);
                }
                t = sumr >> shift;
                let fresh36 = obufb;
                obufb = obufb.offset(1);
                *fresh36 = (if t < 0 as ::core::ffi::c_int {
                    0 as ::core::ffi::c_int
                } else {
                    (if t > 255 as ::core::ffi::c_int {
                        255 as ::core::ffi::c_int
                    } else {
                        t
                    })
                }) as ::core::ffi::c_uchar;
                t = sumg >> shift;
                let fresh37 = obufb;
                obufb = obufb.offset(1);
                *fresh37 = (if t < 0 as ::core::ffi::c_int {
                    0 as ::core::ffi::c_int
                } else {
                    (if t > 255 as ::core::ffi::c_int {
                        255 as ::core::ffi::c_int
                    } else {
                        t
                    })
                }) as ::core::ffi::c_uchar;
                t = sumb >> shift;
                let fresh38 = obufb;
                obufb = obufb.offset(1);
                *fresh38 = (if t < 0 as ::core::ffi::c_int {
                    0 as ::core::ffi::c_int
                } else {
                    (if t > 255 as ::core::ffi::c_int {
                        255 as ::core::ffi::c_int
                    } else {
                        t
                    })
                }) as ::core::ffi::c_uchar;
                let fresh39 = obufb;
                obufb = obufb.offset(1);
                *fresh39 = alpha as ::core::ffi::c_uchar;
                b += 1;
                wtab = wtab.offset(1);
            }
        }
        17395913555154045066 => {
            b = 0 as ::core::ffi::c_int;
            while b < bXsize {
                sum = (1 as ::core::ffi::c_int) << shift - 1 as ::core::ffi::c_int;
                wp = (*wtab).weight.s as *mut b3dInt16;
                ap = lineb.offset((*wtab).i0 as isize) as *mut b3dInt32;
                af = (*wtab).i1 - (*wtab).i0;
                while af > 0 as ::core::ffi::c_int {
                    let fresh40 = wp;
                    wp = wp.offset(1);
                    let fresh41 = ap;
                    ap = ap.offset(1);
                    sum += *fresh40 as ::core::ffi::c_int
                        * (*fresh41 >> CHANBITS) as ::core::ffi::c_short as ::core::ffi::c_int;
                    af -= 1;
                }
                t = sum >> shift;
                tempb = (if t < 0 as ::core::ffi::c_int {
                    0 as ::core::ffi::c_int
                } else {
                    (if t > 255 as ::core::ffi::c_int {
                        255 as ::core::ffi::c_int
                    } else {
                        t
                    })
                }) as ::core::ffi::c_uchar;
                let fresh42 = obufb;
                obufb = obufb.offset(1);
                *fresh42 = tempb;
                let fresh43 = obufb;
                obufb = obufb.offset(1);
                *fresh43 = tempb;
                let fresh44 = obufb;
                obufb = obufb.offset(1);
                *fresh44 = tempb;
                let fresh45 = obufb;
                obufb = obufb.offset(1);
                *fresh45 = alpha as ::core::ffi::c_uchar;
                b += 1;
                wtab = wtab.offset(1);
            }
        }
        _ => {}
    };
}
unsafe extern "C" fn scanline_remap(
    mut filtBuf: *mut ::core::ffi::c_uchar,
    mut dtype: ::core::ffi::c_int,
    mut bXsize: ::core::ffi::c_int,
    mut obufb: *mut ::core::ffi::c_uchar,
    mut cindex: *mut b3dUInt32,
    mut bindex: *mut ::core::ffi::c_uchar,
) {
    let mut i: ::core::ffi::c_int = 0;
    let mut obufi: *mut ::core::ffi::c_int = obufb as *mut ::core::ffi::c_int;
    let mut filts: *mut b3dInt16 = filtBuf as *mut b3dInt16;
    let mut filtus: *mut b3dUInt16 = filtBuf as *mut b3dUInt16;
    match dtype {
        SLICE_MODE_BYTE => {
            i = 0 as ::core::ffi::c_int;
            while i < bXsize {
                let fresh2 = filtBuf;
                filtBuf = filtBuf.offset(1);
                let fresh3 = obufi;
                obufi = obufi.offset(1);
                *fresh3 = *cindex.offset(*fresh2 as isize) as ::core::ffi::c_int;
                i += 1;
            }
        }
        SLICE_MODE_RGB => {
            i = 0 as ::core::ffi::c_int;
            while i < bXsize {
                let fresh4 = filtBuf;
                filtBuf = filtBuf.offset(1);
                let fresh5 = obufb;
                obufb = obufb.offset(1);
                *fresh5 = *bindex.offset(*fresh4 as isize);
                let fresh6 = filtBuf;
                filtBuf = filtBuf.offset(1);
                let fresh7 = obufb;
                obufb = obufb.offset(1);
                *fresh7 = *bindex.offset(*fresh6 as isize);
                let fresh8 = filtBuf;
                filtBuf = filtBuf.offset(1);
                let fresh9 = obufb;
                obufb = obufb.offset(1);
                *fresh9 = *bindex.offset(*fresh8 as isize);
                let fresh10 = obufb;
                obufb = obufb.offset(1);
                *fresh10 = 0 as ::core::ffi::c_uchar;
                i += 1;
            }
        }
        SLICE_MODE_SHORT => {
            i = 0 as ::core::ffi::c_int;
            while i < bXsize {
                let fresh11 = filts;
                filts = filts.offset(1);
                let fresh12 = obufi;
                obufi = obufi.offset(1);
                *fresh12 = *cindex.offset(*fresh11 as isize) as ::core::ffi::c_int;
                i += 1;
            }
        }
        SLICE_MODE_USHORT => {
            i = 0 as ::core::ffi::c_int;
            while i < bXsize {
                let fresh13 = filtus;
                filtus = filtus.offset(1);
                let fresh14 = obufi;
                obufi = obufi.offset(1);
                *fresh14 = *cindex.offset(*fresh13 as isize) as ::core::ffi::c_int;
                i += 1;
            }
        }
        _ => {}
    };
}
unsafe extern "C" fn make_weighttab(
    mut b: ::core::ffi::c_int,
    mut cen: ::core::ffi::c_double,
    mut len: ::core::ffi::c_int,
    mut scale: ::core::ffi::c_double,
    mut support: ::core::ffi::c_double,
    mut dtype: ::core::ffi::c_int,
    mut wtab: *mut Weighttab,
) {
    let mut i0: ::core::ffi::c_int = 0;
    let mut i1: ::core::ffi::c_int = 0;
    let mut i: ::core::ffi::c_int = 0;
    let mut sum: ::core::ffi::c_int = 0;
    let mut t: ::core::ffi::c_int = 0;
    let mut stillzero: ::core::ffi::c_int = 0;
    let mut lastnonzero: ::core::ffi::c_int = 0;
    let mut wp: *mut ::core::ffi::c_short = ::core::ptr::null_mut::<::core::ffi::c_short>();
    let mut den: ::core::ffi::c_double = 0.;
    let mut sc: ::core::ffi::c_double = 0.;
    let mut tr: ::core::ffi::c_double = 0.;
    let mut rsum: ::core::ffi::c_double = 0.;
    let mut shortWgts: ::core::ffi::c_int =
        if dtype == SLICE_MODE_BYTE || dtype == SLICE_MODE_RGB || dtype < 0 as ::core::ffi::c_int {
            1 as ::core::ffi::c_int
        } else {
            0 as ::core::ffi::c_int
        };
    i0 = (cen - support + 0.5f64) as ::core::ffi::c_int;
    i1 = (cen + support + 0.5f64) as ::core::ffi::c_int;
    if i0 < 0 as ::core::ffi::c_int {
        i0 = 0 as ::core::ffi::c_int;
    }
    if i1 > len {
        i1 = len;
    }
    (*wtab).i0 = i0;
    (*wtab).i1 = i1;
    den = 0 as ::core::ffi::c_int as ::core::ffi::c_double;
    i = i0;
    while i < i1 {
        den += sFilt_func.expect("non-null function pointer")(
            (i as ::core::ffi::c_double + 0.5f64 - cen) / scale,
        );
        i += 1;
    }
    if shortWgts != 0 {
        sc = if den == 0.0f64 {
            WEIGHTONE as ::core::ffi::c_double
        } else {
            WEIGHTONE as ::core::ffi::c_double / den
        };
    } else {
        sc = if den == 0.0f64 { 1.0f64 } else { 1.0f64 / den };
    }
    if zoom_debug > 1 as ::core::ffi::c_int {
        fprintf(
            stderr,
            b"    b=%d cen=%g scale=%g [%d..%d) sc=%g:  \0" as *const u8
                as *const ::core::ffi::c_char,
            b,
            cen,
            scale,
            i0,
            i1,
            sc,
        );
    }
    stillzero = shortWgts;
    rsum = 0.0f64;
    sum = 0 as ::core::ffi::c_int;
    wp = (*wtab).weight.s;
    i = i0;
    while i < i1 {
        tr = sc
            * sFilt_func.expect("non-null function pointer")(
                (i as ::core::ffi::c_double + 0.5f64 - cen) / scale,
            );
        rsum += tr;
        if shortWgts != 0 {
            t = floor(tr + 0.5f64) as ::core::ffi::c_int;
            if stillzero != 0 && t == 0 as ::core::ffi::c_int {
                i0 += 1;
            } else {
                stillzero = 0 as ::core::ffi::c_int;
                let fresh68 = wp;
                wp = wp.offset(1);
                *fresh68 = t as ::core::ffi::c_short;
                sum += t;
                if t != 0 as ::core::ffi::c_int {
                    lastnonzero = i;
                }
            }
        } else {
            *(*wtab).weight.f.offset((i - i0) as isize) = tr as ::core::ffi::c_float;
        }
        i += 1;
    }
    if shortWgts != 0 && sum == 0 as ::core::ffi::c_int || rsum == 0.0f64 {
        (*wtab).i0 = (*wtab).i0 + (*wtab).i1 >> 1 as ::core::ffi::c_int;
        (*wtab).i1 = (*wtab).i0 + 1 as ::core::ffi::c_int;
        if shortWgts != 0 {
            *(*wtab).weight.s.offset(0 as ::core::ffi::c_int as isize) =
                WEIGHTONE as ::core::ffi::c_short;
        } else {
            *(*wtab).weight.f.offset(0 as ::core::ffi::c_int as isize) = 1.0f32;
        }
    } else if shortWgts != 0 {
        (*wtab).i0 = i0;
        i1 = lastnonzero + 1 as ::core::ffi::c_int;
        (*wtab).i1 = i1;
        if sum != WEIGHTONE {
            i = (cen + 0.5f64) as ::core::ffi::c_int;
            if i < i0 {
                i = i0;
            } else if i >= i1 {
                i = i1 - 1 as ::core::ffi::c_int;
            }
            t = WEIGHTONE - sum;
            if zoom_debug > 1 as ::core::ffi::c_int {
                fprintf(
                    stderr,
                    b"[%d]+=%d \0" as *const u8 as *const ::core::ffi::c_char,
                    i,
                    t,
                );
            }
            let ref mut fresh69 = *(*wtab).weight.s.offset((i - i0) as isize);
            *fresh69 = (*fresh69 as ::core::ffi::c_int + t) as ::core::ffi::c_short;
        }
    }
    if zoom_debug > 1 as ::core::ffi::c_int {
        fprintf(stderr, b"\t\0" as *const u8 as *const ::core::ffi::c_char);
        if shortWgts != 0 {
            wp = (*wtab).weight.s;
            i = i0;
            while i < i1 {
                fprintf(
                    stderr,
                    b"%5d \0" as *const u8 as *const ::core::ffi::c_char,
                    *wp as ::core::ffi::c_int,
                );
                i += 1;
                wp = wp.offset(1);
            }
        } else {
            i = i0;
            while i < i1 {
                fprintf(
                    stderr,
                    b"%.4f \0" as *const u8 as *const ::core::ffi::c_char,
                    *(*wtab).weight.f.offset((i - i0) as isize) as ::core::ffi::c_double,
                );
                i += 1;
            }
        }
        fprintf(stderr, b"\n\0" as *const u8 as *const ::core::ffi::c_char);
    }
}
unsafe extern "C" fn filt_binning(mut x: ::core::ffi::c_double) -> ::core::ffi::c_double {
    return if x >= -0.5f64 && x < 0.5f64 {
        1.0f64
    } else {
        0.0f64
    };
}
unsafe extern "C" fn mitchell_init(mut b: ::core::ffi::c_double, mut c: ::core::ffi::c_double) {
    sMitchP0 = (6.0f64 - 2.0f64 * b) / 6.0f64;
    sMitchP2 = (-18.0f64 + 12.0f64 * b + 6.0f64 * c) / 6.0f64;
    sMitchP3 = (12.0f64 - 9.0f64 * b - 6.0f64 * c) / 6.0f64;
    sMitchQ0 = (8.0f64 * b + 24.0f64 * c) / 6.0f64;
    sMitchQ1 = (-12.0f64 * b - 48.0f64 * c) / 6.0f64;
    sMitchQ2 = (6.0f64 * b + 30.0f64 * c) / 6.0f64;
    sMitchQ3 = (-b - 6.0f64 * c) / 6.0f64;
}
unsafe extern "C" fn filt_mitchell(mut x: ::core::ffi::c_double) -> ::core::ffi::c_double {
    if x < -2.0f64 {
        return 0.0f64;
    }
    if x < -1.0f64 {
        return sMitchQ0 - x * (sMitchQ1 - x * (sMitchQ2 - x * sMitchQ3));
    }
    if x < 0.0f64 {
        return sMitchP0 + x * x * (sMitchP2 - x * sMitchP3);
    }
    if x < 1.0f64 {
        return sMitchP0 + x * x * (sMitchP2 + x * sMitchP3);
    }
    if x < 2.0f64 {
        return sMitchQ0 + x * (sMitchQ1 + x * (sMitchQ2 + x * sMitchQ3));
    }
    return 0.0f64;
}
unsafe extern "C" fn filt_blackman(mut x: ::core::ffi::c_double) -> ::core::ffi::c_double {
    return 0.42f64 + 0.50f64 * cos(PI * x) + 0.08f64 * cos(2.0f64 * PI * x);
}
unsafe extern "C" fn filt_triangle(mut x: ::core::ffi::c_double) -> ::core::ffi::c_double {
    if x <= -1.0f64 || x >= 1.0f64 {
        return 0.0f64;
    }
    return 1.0f64 - fabs(x);
}
unsafe extern "C" fn filt_lanczos2(mut x: ::core::ffi::c_double) -> ::core::ffi::c_double {
    let mut a: ::core::ffi::c_double = 2.0f64;
    if x < -a || x > a {
        return 0.0f64;
    }
    if x < 1.0e-6f64 && x > -1.0e-6f64 {
        return 1.0f64;
    }
    return a * sin(PI * x) * sin(PI * x / a) / (PI * PI * x * x);
}
unsafe extern "C" fn filt_lanczos3(mut x: ::core::ffi::c_double) -> ::core::ffi::c_double {
    let mut a: ::core::ffi::c_double = 3.0f64;
    if x < -a || x > a {
        return 0.0f64;
    }
    if x < 1.0e-6f64 && x > -1.0e-6f64 {
        return 1.0f64;
    }
    return a * sin(PI * x) * sin(PI * x / a) / (PI * PI * x * x);
}

#[cfg(test)]
mod tests {
    use super::{
        SLICE_MODE_BYTE, SLICE_MODE_FLOAT, select_zoom_filter, zoom_filt_value, zoom_with_filter,
    };

    #[test]
    fn selected_binning_filter_is_normalized() {
        let mut width = 0;
        unsafe {
            assert_eq!(select_zoom_filter(0, 0.5, &raw mut width), 0);
            assert_eq!(width, 2);
            assert!((zoom_filt_value(0.0) - 0.5).abs() < 1.0e-12);
        }
    }

    #[test]
    fn box_filter_reduces_byte_and_float_images() {
        unsafe {
            let mut width = 0;
            assert_eq!(select_zoom_filter(0, 0.5, &raw mut width), 0);

            let mut byte_input = [
                10_u8, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160,
            ];
            let mut byte_lines = [
                byte_input.as_mut_ptr(),
                byte_input.as_mut_ptr().add(4),
                byte_input.as_mut_ptr().add(8),
                byte_input.as_mut_ptr().add(12),
            ];
            let mut byte_out = [0_u8; 4];
            assert_eq!(
                zoom_with_filter(
                    byte_lines.as_mut_ptr(),
                    4,
                    4,
                    0.,
                    0.,
                    2,
                    2,
                    2,
                    0,
                    SLICE_MODE_BYTE,
                    byte_out.as_mut_ptr().cast(),
                    core::ptr::null_mut(),
                    core::ptr::null_mut()
                ),
                0
            );
            assert_eq!(byte_out, [35, 55, 115, 135]);

            let mut float_input = [
                1_f32, 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
            ];
            let mut float_lines = [
                float_input.as_mut_ptr().cast(),
                float_input.as_mut_ptr().add(4).cast(),
                float_input.as_mut_ptr().add(8).cast(),
                float_input.as_mut_ptr().add(12).cast(),
            ];
            let mut float_out = [0_f32; 4];
            assert_eq!(
                zoom_with_filter(
                    float_lines.as_mut_ptr(),
                    4,
                    4,
                    0.,
                    0.,
                    2,
                    2,
                    2,
                    0,
                    SLICE_MODE_FLOAT,
                    float_out.as_mut_ptr().cast(),
                    core::ptr::null_mut(),
                    core::ptr::null_mut()
                ),
                0
            );
            assert_eq!(float_out, [3.5, 5.5, 11.5, 13.5]);
        }
    }
}
