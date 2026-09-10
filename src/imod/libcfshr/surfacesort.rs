#![allow(
    dead_code,
    non_snake_case,
    non_camel_case_types,
    unused_mut,
    unused_assignments,
    unsafe_op_in_unsafe_fn
)]
//! Mechanical C2Rust baseline of `IMOD/libcfshr/surfacesort.c`, wired to direct translated dependencies.

use super::robuststat::{rs_mad_median_outliers, rs_median, rs_sort_indexed_floats};
use super::simplestat::{ls_fit2, ls_fit3, sums_to_avg_sd};

pub enum _IO_wide_data {}
pub enum _IO_codecvt {}
pub enum _IO_marker {}
unsafe extern "C" {
    static mut stdout: *mut FILE;
    fn fflush(__stream: *mut FILE) -> ::core::ffi::c_int;
    fn printf(__format: *const ::core::ffi::c_char, ...) -> ::core::ffi::c_int;
    fn atan(__x: ::core::ffi::c_double) -> ::core::ffi::c_double;
    fn atan2(__y: ::core::ffi::c_double, __x: ::core::ffi::c_double) -> ::core::ffi::c_double;
    fn cos(__x: ::core::ffi::c_double) -> ::core::ffi::c_double;
    fn sin(__x: ::core::ffi::c_double) -> ::core::ffi::c_double;
    fn sqrt(__x: ::core::ffi::c_double) -> ::core::ffi::c_double;
    fn fabs(__x: ::core::ffi::c_double) -> ::core::ffi::c_double;
    fn floor(__x: ::core::ffi::c_double) -> ::core::ffi::c_double;
    fn malloc(__size: size_t) -> *mut ::core::ffi::c_void;
    fn free(__ptr: *mut ::core::ffi::c_void);
    fn c_sums_to_avg_sd(
        sx: ::core::ffi::c_float,
        sxsq: ::core::ffi::c_float,
        n: ::core::ffi::c_int,
        avg: *mut ::core::ffi::c_float,
        sd: *mut ::core::ffi::c_float,
    );
    fn c_ls_fit2(
        x1: *mut ::core::ffi::c_float,
        x2: *mut ::core::ffi::c_float,
        y: *mut ::core::ffi::c_float,
        n: ::core::ffi::c_int,
        a: *mut ::core::ffi::c_float,
        b: *mut ::core::ffi::c_float,
        c: *mut ::core::ffi::c_float,
    );
    fn c_ls_fit3(
        x1: *mut ::core::ffi::c_float,
        x2: *mut ::core::ffi::c_float,
        x3: *mut ::core::ffi::c_float,
        y: *mut ::core::ffi::c_float,
        n: ::core::ffi::c_int,
        a1: *mut ::core::ffi::c_float,
        a2: *mut ::core::ffi::c_float,
        a3: *mut ::core::ffi::c_float,
        c: *mut ::core::ffi::c_float,
    );
    fn c_rs_sort_indexed_floats(
        x: *mut ::core::ffi::c_float,
        index: *mut ::core::ffi::c_int,
        n: ::core::ffi::c_int,
    );
    fn c_rs_median(
        x: *mut ::core::ffi::c_float,
        n: ::core::ffi::c_int,
        tmp: *mut ::core::ffi::c_float,
        median: *mut ::core::ffi::c_float,
    );
    fn c_rs_mad_median_outliers(
        x: *mut ::core::ffi::c_float,
        n: ::core::ffi::c_int,
        kcrit: ::core::ffi::c_float,
        out: *mut ::core::ffi::c_float,
    );
}
pub type size_t = usize;
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
pub const DTOR: ::core::ffi::c_double = 0.017453293f64;
static mut sGridSpacing: ::core::ffi::c_float = 50.0f32;
static mut sMaxAngleNeigh: ::core::ffi::c_int = 50 as ::core::ffi::c_int;
static mut sSteepestRatio: ::core::ffi::c_float = 0.5f32;
static mut sNumMinForAmax: ::core::ffi::c_int = 10 as ::core::ffi::c_int;
static mut sAngleMax: ::core::ffi::c_float = 20.0f32;
static mut sNumMinForArelax: ::core::ffi::c_int = 3 as ::core::ffi::c_int;
static mut sAngleRelax: ::core::ffi::c_float = 5.0f32;
static mut sOutlierCrit: ::core::ffi::c_float = 3.0f32;
static mut sMaxFitDist: ::core::ffi::c_float = 2048.0f32;
static mut sMaxNumFit: ::core::ffi::c_int = 15 as ::core::ffi::c_int;
static mut sBiplaneMinFit: ::core::ffi::c_int = 5 as ::core::ffi::c_int;
static mut sPlaneMinFit: ::core::ffi::c_int = 4 as ::core::ffi::c_int;
static mut sMaxRound1Dist: ::core::ffi::c_float = 100.0f32;
static mut sDebugLevel: ::core::ffi::c_int = 0 as ::core::ffi::c_int;
#[unsafe(no_mangle)]
pub unsafe extern "C" fn set_surf_sort_param(
    mut index: ::core::ffi::c_int,
    mut value: ::core::ffi::c_float,
) -> ::core::ffi::c_int {
    match index {
        0 => {
            sGridSpacing = value;
        }
        1 => {
            sMaxAngleNeigh = floor(value as ::core::ffi::c_double + 0.5f64) as ::core::ffi::c_int;
        }
        2 => {
            sSteepestRatio = value;
        }
        3 => {
            sNumMinForAmax = floor(value as ::core::ffi::c_double + 0.5f64) as ::core::ffi::c_int;
        }
        4 => {
            sAngleMax = value;
        }
        5 => {
            sNumMinForArelax = floor(value as ::core::ffi::c_double + 0.5f64) as ::core::ffi::c_int;
        }
        6 => {
            sAngleRelax = value;
        }
        7 => {
            sOutlierCrit = value;
        }
        8 => {
            sMaxFitDist = value;
        }
        9 => {
            sMaxNumFit = floor(value as ::core::ffi::c_double + 0.5f64) as ::core::ffi::c_int;
        }
        10 => {
            sBiplaneMinFit = floor(value as ::core::ffi::c_double + 0.5f64) as ::core::ffi::c_int;
        }
        11 => {
            sPlaneMinFit = floor(value as ::core::ffi::c_double + 0.5f64) as ::core::ffi::c_int;
        }
        12 => {
            sMaxRound1Dist = value;
        }
        13 => {
            sDebugLevel = floor(value as ::core::ffi::c_double + 0.5f64) as ::core::ffi::c_int;
        }
        _ => return 1 as ::core::ffi::c_int,
    }
    return 0 as ::core::ffi::c_int;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn set_surf_sort_param_fortran(
    mut index: *mut ::core::ffi::c_int,
    mut value: *mut ::core::ffi::c_float,
) -> ::core::ffi::c_int {
    return set_surf_sort_param(*index, *value);
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn surface_sort(
    mut xyz: *mut ::core::ffi::c_float,
    mut numPts: ::core::ffi::c_int,
    mut markersInGroup: ::core::ffi::c_int,
    mut group: *mut ::core::ffi::c_int,
) -> ::core::ffi::c_int {
    let mut i: ::core::ffi::c_int = 0;
    let mut numGridX: ::core::ffi::c_int = 0;
    let mut numGridY: ::core::ffi::c_int = 0;
    let mut numSquares: ::core::ffi::c_int = 0;
    let mut numRings: ::core::ffi::c_int = 0;
    let mut ind: ::core::ffi::c_int = 0;
    let mut ring: ::core::ffi::c_int = 0;
    let mut dx: ::core::ffi::c_int = 0;
    let mut dy: ::core::ffi::c_int = 0;
    let mut sx: ::core::ffi::c_int = 0;
    let mut sy: ::core::ffi::c_int = 0;
    let mut afit: ::core::ffi::c_float = 0.;
    let mut bfit: ::core::ffi::c_float = 0.;
    let mut cfit: ::core::ffi::c_float = 0.;
    let mut zp: ::core::ffi::c_float = 0.;
    let mut xmin: ::core::ffi::c_float = 0.;
    let mut xmax: ::core::ffi::c_float = 0.;
    let mut ymin: ::core::ffi::c_float = 0.;
    let mut ymax: ::core::ffi::c_float = 0.;
    let mut diagonal: ::core::ffi::c_float = 0.;
    let mut pdx: ::core::ffi::c_float = 0.;
    let mut pdy: ::core::ffi::c_float = 0.;
    let mut alpha: ::core::ffi::c_double = 0.;
    let mut cosal: ::core::ffi::c_double = 0.;
    let mut sinal: ::core::ffi::c_double = 0.;
    let mut theta: ::core::ffi::c_double = 0.;
    let mut sinth: ::core::ffi::c_double = 0.;
    let mut costh: ::core::ffi::c_double = 0.;
    let mut slope: ::core::ffi::c_double = 0.;
    let mut xrot: *mut ::core::ffi::c_float = ::core::ptr::null_mut::<::core::ffi::c_float>();
    let mut yrot: *mut ::core::ffi::c_float = ::core::ptr::null_mut::<::core::ffi::c_float>();
    let mut zrot: *mut ::core::ffi::c_float = ::core::ptr::null_mut::<::core::ffi::c_float>();
    let mut idx: *mut ::core::ffi::c_short = ::core::ptr::null_mut::<::core::ffi::c_short>();
    let mut idy: *mut ::core::ffi::c_short = ::core::ptr::null_mut::<::core::ffi::c_short>();
    let mut ringStart: *mut ::core::ffi::c_int = ::core::ptr::null_mut::<::core::ffi::c_int>();
    let mut numInSquare: *mut ::core::ffi::c_int = ::core::ptr::null_mut::<::core::ffi::c_int>();
    let mut squareInd: *mut ::core::ffi::c_int = ::core::ptr::null_mut::<::core::ffi::c_int>();
    let mut pointLists: *mut ::core::ffi::c_int = ::core::ptr::null_mut::<::core::ffi::c_int>();
    let mut steepNeigh: *mut ::core::ffi::c_int = ::core::ptr::null_mut::<::core::ffi::c_int>();
    let mut sortInd: *mut ::core::ffi::c_int = ::core::ptr::null_mut::<::core::ffi::c_int>();
    let mut squareDone: *mut ::core::ffi::c_uchar = ::core::ptr::null_mut::<::core::ffi::c_uchar>();
    let mut steepAngle: *mut ::core::ffi::c_float = ::core::ptr::null_mut::<::core::ffi::c_float>();
    let mut xfit: *mut ::core::ffi::c_float = ::core::ptr::null_mut::<::core::ffi::c_float>();
    let mut yfit: *mut ::core::ffi::c_float = ::core::ptr::null_mut::<::core::ffi::c_float>();
    let mut zfit: *mut ::core::ffi::c_float = ::core::ptr::null_mut::<::core::ffi::c_float>();
    let mut grpfit: *mut ::core::ffi::c_float = ::core::ptr::null_mut::<::core::ffi::c_float>();
    let mut delZ: *mut ::core::ffi::c_float = ::core::ptr::null_mut::<::core::ffi::c_float>();
    let mut outlie: *mut ::core::ffi::c_float = ::core::ptr::null_mut::<::core::ffi::c_float>();
    let mut cluster: *mut ::core::ffi::c_int = ::core::ptr::null_mut::<::core::ffi::c_int>();
    let mut clusterSX: *mut ::core::ffi::c_int = ::core::ptr::null_mut::<::core::ffi::c_int>();
    let mut clusterSY: *mut ::core::ffi::c_int = ::core::ptr::null_mut::<::core::ffi::c_int>();
    let mut isq: ::core::ffi::c_int = 0;
    let mut ipt: ::core::ffi::c_int = 0;
    let mut ix: ::core::ffi::c_int = 0;
    let mut iy: ::core::ffi::c_int = 0;
    let mut jnd: ::core::ffi::c_int = 0;
    let mut jsq: ::core::ffi::c_int = 0;
    let mut jpt: ::core::ffi::c_int = 0;
    let mut numNeigh: ::core::ffi::c_int = 0;
    let mut firstSteep: ::core::ffi::c_int = 0;
    let mut numSteep: ::core::ffi::c_int = 0;
    let mut ifdup: ::core::ffi::c_int = 0;
    let mut angle: ::core::ffi::c_float = 0.;
    let mut verySteepest: ::core::ffi::c_float = 0.;
    let mut zbot: ::core::ffi::c_float = 0.;
    let mut ztop: ::core::ffi::c_float = 0.;
    let mut a1: ::core::ffi::c_float = 0.;
    let mut a2: ::core::ffi::c_float = 0.;
    let mut dzfit: ::core::ffi::c_float = 0.;
    let mut con: ::core::ffi::c_float = 0.;
    let mut medianDelZ: ::core::ffi::c_float = 0.;
    let mut j: ::core::ffi::c_int = 0;
    let mut numDone: ::core::ffi::c_int = 0;
    let mut jdxy: ::core::ffi::c_int = 0;
    let mut kdxy: ::core::ffi::c_int = 0;
    let mut jring: ::core::ffi::c_int = 0;
    let mut tx: ::core::ffi::c_int = 0;
    let mut ty: ::core::ffi::c_int = 0;
    let mut maxRings: ::core::ffi::c_int = 0;
    let mut grpsum: ::core::ffi::c_int = 0;
    let mut keepGroup: ::core::ffi::c_int = 0;
    let mut ntop: ::core::ffi::c_int = 0;
    let mut nbot: ::core::ffi::c_int = 0;
    let mut nfit: ::core::ffi::c_int = 0;
    let mut jx: ::core::ffi::c_int = 0;
    let mut jy: ::core::ffi::c_int = 0;
    let mut knd: ::core::ffi::c_int = 0;
    let mut kpt: ::core::ffi::c_int = 0;
    let mut numCluster: ::core::ffi::c_int = 0;
    let mut numInClust: ::core::ffi::c_int = 0;
    let mut checkInd: ::core::ffi::c_int = 0;
    let mut oldpt: ::core::ffi::c_int = 0;
    let mut newpt: ::core::ffi::c_int = 0;
    let mut numErr: ::core::ffi::c_int = 0;
    let mut round: ::core::ffi::c_int = 0;
    let mut xsum: ::core::ffi::c_float = 0.;
    let mut ysum: ::core::ffi::c_float = 0.;
    let mut errsum: ::core::ffi::c_float = 0.;
    let mut errsq: ::core::ffi::c_float = 0.;
    let mut errmax: ::core::ffi::c_float = 0.;
    if numPts < 3 as ::core::ffi::c_int {
        *group.offset(0 as ::core::ffi::c_int as isize) = 1 as ::core::ffi::c_int;
        if numPts > 1 as ::core::ffi::c_int {
            *group.offset(1 as ::core::ffi::c_int as isize) = 2 as ::core::ffi::c_int;
        }
        return 0 as ::core::ffi::c_int;
    }
    xrot = malloc(
        (numPts as size_t).wrapping_mul(::core::mem::size_of::<::core::ffi::c_float>() as size_t),
    ) as *mut ::core::ffi::c_float;
    yrot = malloc(
        (numPts as size_t).wrapping_mul(::core::mem::size_of::<::core::ffi::c_float>() as size_t),
    ) as *mut ::core::ffi::c_float;
    zrot = malloc(
        (numPts as size_t).wrapping_mul(::core::mem::size_of::<::core::ffi::c_float>() as size_t),
    ) as *mut ::core::ffi::c_float;
    if xrot.is_null() || yrot.is_null() || zrot.is_null() {
        return 1 as ::core::ffi::c_int;
    }
    i = 0 as ::core::ffi::c_int;
    while i < numPts {
        *xrot.offset(i as isize) = *xyz.offset((3 as ::core::ffi::c_int * i) as isize);
        *yrot.offset(i as isize) =
            *xyz.offset((3 as ::core::ffi::c_int * i + 1 as ::core::ffi::c_int) as isize);
        *zrot.offset(i as isize) =
            *xyz.offset((3 as ::core::ffi::c_int * i + 2 as ::core::ffi::c_int) as isize);
        i += 1;
    }
    ls_fit2(
        xrot,
        yrot,
        zrot,
        numPts,
        &raw mut afit,
        &raw mut bfit,
        &raw mut cfit,
    );
    alpha = atan(bfit as ::core::ffi::c_double);
    cosal = cos(alpha);
    sinal = sin(alpha);
    slope = afit as ::core::ffi::c_double / (cosal - bfit as ::core::ffi::c_double * sinal);
    theta = -atan(slope);
    costh = cos(theta);
    sinth = sin(theta);
    ymin = 1.0e30f32;
    xmin = ymin;
    ymax = -1.0e30f64 as ::core::ffi::c_float;
    xmax = ymax;
    i = 0 as ::core::ffi::c_int;
    while i < numPts {
        *yrot.offset(i as isize) = (*xyz
            .offset((3 as ::core::ffi::c_int * i + 1 as ::core::ffi::c_int) as isize)
            as ::core::ffi::c_double
            * cosal
            + *xyz.offset((3 as ::core::ffi::c_int * i + 2 as ::core::ffi::c_int) as isize)
                as ::core::ffi::c_double
                * sinal) as ::core::ffi::c_float;
        zp = (-*xyz.offset((3 as ::core::ffi::c_int * i + 1 as ::core::ffi::c_int) as isize)
            as ::core::ffi::c_double
            * sinal
            + *xyz.offset((3 as ::core::ffi::c_int * i + 2 as ::core::ffi::c_int) as isize)
                as ::core::ffi::c_double
                * cosal) as ::core::ffi::c_float;
        *xrot.offset(i as isize) =
            (*xyz.offset((3 as ::core::ffi::c_int * i) as isize) as ::core::ffi::c_double * costh
                - zp as ::core::ffi::c_double * sinth) as ::core::ffi::c_float;
        *zrot.offset(i as isize) =
            (*xyz.offset((3 as ::core::ffi::c_int * i) as isize) as ::core::ffi::c_double * sinth
                + zp as ::core::ffi::c_double * costh) as ::core::ffi::c_float;
        xmin = if xmin < *xrot.offset(i as isize) {
            xmin
        } else {
            *xrot.offset(i as isize)
        };
        xmax = if xmax > *xrot.offset(i as isize) {
            xmax
        } else {
            *xrot.offset(i as isize)
        };
        ymin = if ymin < *yrot.offset(i as isize) {
            ymin
        } else {
            *yrot.offset(i as isize)
        };
        ymax = if ymax > *yrot.offset(i as isize) {
            ymax
        } else {
            *yrot.offset(i as isize)
        };
        i += 1;
    }
    numGridX =
        (((xmax - xmin) / sGridSpacing) as ::core::ffi::c_double + 1.0f64) as ::core::ffi::c_int;
    numGridY =
        (((ymax - ymin) / sGridSpacing) as ::core::ffi::c_double + 1.0f64) as ::core::ffi::c_int;
    numSquares = numGridX * numGridY;
    diagonal = sqrt(
        (numGridX as ::core::ffi::c_double - 1.0f64) * (numGridX as ::core::ffi::c_double - 1.0f64)
            + (numGridY as ::core::ffi::c_double - 1.0f64)
                * (numGridY as ::core::ffi::c_double - 1.0f64),
    ) as ::core::ffi::c_float;
    numRings = floor(diagonal as ::core::ffi::c_double + 0.5f64) as ::core::ffi::c_int
        + 1 as ::core::ffi::c_int;
    idx = malloc(
        ((4 as ::core::ffi::c_int * numSquares) as size_t)
            .wrapping_mul(::core::mem::size_of::<::core::ffi::c_short>() as size_t),
    ) as *mut ::core::ffi::c_short;
    idy = malloc(
        ((4 as ::core::ffi::c_int * numSquares) as size_t)
            .wrapping_mul(::core::mem::size_of::<::core::ffi::c_short>() as size_t),
    ) as *mut ::core::ffi::c_short;
    ringStart = malloc(
        ((numRings + 2 as ::core::ffi::c_int) as size_t)
            .wrapping_mul(::core::mem::size_of::<::core::ffi::c_int>() as size_t),
    ) as *mut ::core::ffi::c_int;
    numInSquare = malloc(
        (numSquares as size_t).wrapping_mul(::core::mem::size_of::<::core::ffi::c_int>() as size_t),
    ) as *mut ::core::ffi::c_int;
    squareInd = malloc(
        (numSquares as size_t).wrapping_mul(::core::mem::size_of::<::core::ffi::c_int>() as size_t),
    ) as *mut ::core::ffi::c_int;
    pointLists = malloc(
        (numPts as size_t).wrapping_mul(::core::mem::size_of::<::core::ffi::c_int>() as size_t),
    ) as *mut ::core::ffi::c_int;
    squareDone = malloc(
        (numSquares as size_t)
            .wrapping_mul(::core::mem::size_of::<::core::ffi::c_uchar>() as size_t),
    ) as *mut ::core::ffi::c_uchar;
    steepAngle = malloc(
        (numPts as size_t).wrapping_mul(::core::mem::size_of::<::core::ffi::c_float>() as size_t),
    ) as *mut ::core::ffi::c_float;
    steepNeigh = malloc(
        (numPts as size_t).wrapping_mul(::core::mem::size_of::<::core::ffi::c_int>() as size_t),
    ) as *mut ::core::ffi::c_int;
    sortInd = malloc(
        (numPts as size_t).wrapping_mul(::core::mem::size_of::<::core::ffi::c_int>() as size_t),
    ) as *mut ::core::ffi::c_int;
    xfit = malloc(
        (sMaxNumFit as size_t)
            .wrapping_mul(::core::mem::size_of::<::core::ffi::c_float>() as size_t),
    ) as *mut ::core::ffi::c_float;
    yfit = malloc(
        (sMaxNumFit as size_t)
            .wrapping_mul(::core::mem::size_of::<::core::ffi::c_float>() as size_t),
    ) as *mut ::core::ffi::c_float;
    zfit = malloc(
        (sMaxNumFit as size_t)
            .wrapping_mul(::core::mem::size_of::<::core::ffi::c_float>() as size_t),
    ) as *mut ::core::ffi::c_float;
    grpfit = malloc(
        (sMaxNumFit as size_t)
            .wrapping_mul(::core::mem::size_of::<::core::ffi::c_float>() as size_t),
    ) as *mut ::core::ffi::c_float;
    if idx.is_null()
        || idy.is_null()
        || numInSquare.is_null()
        || squareInd.is_null()
        || squareDone.is_null()
        || ringStart.is_null()
        || pointLists.is_null()
        || steepAngle.is_null()
        || steepNeigh.is_null()
        || sortInd.is_null()
        || xfit.is_null()
        || yfit.is_null()
        || zfit.is_null()
        || grpfit.is_null()
    {
        return 1 as ::core::ffi::c_int;
    }
    ind = 0 as ::core::ffi::c_int;
    ring = 0 as ::core::ffi::c_int;
    while ring < numRings {
        *ringStart.offset(ring as isize) = ind;
        dy = -(numGridY - 1 as ::core::ffi::c_int);
        while dy < numGridY {
            dx = -(numGridX - 1 as ::core::ffi::c_int);
            while dx < numGridX {
                if floor(sqrt((dx * dx + dy * dy) as ::core::ffi::c_double) + 0.5f64)
                    as ::core::ffi::c_int
                    == ring
                {
                    *idx.offset(ind as isize) = dx as ::core::ffi::c_short;
                    let fresh0 = ind;
                    ind = ind + 1;
                    *idy.offset(fresh0 as isize) = dy as ::core::ffi::c_short;
                }
                dx += 1;
            }
            dy += 1;
        }
        ring += 1;
    }
    *ringStart.offset(numRings as isize) = ind;
    i = 0 as ::core::ffi::c_int;
    while i < numSquares {
        *numInSquare.offset(i as isize) = 0 as ::core::ffi::c_int;
        i += 1;
    }
    i = 0 as ::core::ffi::c_int;
    while i < numPts {
        sx = ((*xrot.offset(i as isize) - xmin) / sGridSpacing) as ::core::ffi::c_int;
        sy = ((*yrot.offset(i as isize) - ymin) / sGridSpacing) as ::core::ffi::c_int;
        let ref mut fresh1 = *numInSquare.offset((sx + sy * numGridX) as isize);
        *fresh1 += 1;
        i += 1;
    }
    ind = 0 as ::core::ffi::c_int;
    i = 0 as ::core::ffi::c_int;
    while i < numSquares {
        *squareInd.offset(i as isize) = ind;
        ind += *numInSquare.offset(i as isize);
        *numInSquare.offset(i as isize) = 0 as ::core::ffi::c_int;
        *squareDone.offset(i as isize) = 0 as ::core::ffi::c_uchar;
        i += 1;
    }
    i = 0 as ::core::ffi::c_int;
    while i < numPts {
        sx = ((*xrot.offset(i as isize) - xmin) / sGridSpacing) as ::core::ffi::c_int;
        sy = ((*yrot.offset(i as isize) - ymin) / sGridSpacing) as ::core::ffi::c_int;
        ind = sx + sy * numGridX;
        *pointLists.offset(
            (*squareInd.offset(ind as isize) + *numInSquare.offset(ind as isize)) as isize,
        ) = i;
        let ref mut fresh2 = *numInSquare.offset(ind as isize);
        *fresh2 += 1;
        i += 1;
    }
    ind = 0 as ::core::ffi::c_int;
    while ind < numSquares {
        sx = ind % numGridX;
        sy = ind / numGridX;
        isq = 0 as ::core::ffi::c_int;
        while isq < *numInSquare.offset(ind as isize) {
            ipt = *pointLists.offset((*squareInd.offset(ind as isize) + isq) as isize);
            *steepAngle.offset(ipt as isize) = -1.0f64 as ::core::ffi::c_float;
            *steepNeigh.offset(ipt as isize) = -(1 as ::core::ffi::c_int);
            *sortInd.offset(ipt as isize) = ipt;
            if markersInGroup != 0 && *group.offset(ipt as isize) < 0 as ::core::ffi::c_int {
                if sDebugLevel > 1 as ::core::ffi::c_int {
                    printf(
                        b"Skipping search for %d\n\0" as *const u8 as *const ::core::ffi::c_char,
                        ipt,
                    );
                }
            } else {
                numNeigh = 0 as ::core::ffi::c_int;
                jdxy = 0 as ::core::ffi::c_int;
                while jdxy < *ringStart.offset(numRings as isize) && numNeigh < sMaxAngleNeigh {
                    ix = sx + *idx.offset(jdxy as isize) as ::core::ffi::c_int;
                    iy = sy + *idy.offset(jdxy as isize) as ::core::ffi::c_int;
                    if !(ix < 0 as ::core::ffi::c_int
                        || ix >= numGridX
                        || iy < 0 as ::core::ffi::c_int
                        || iy >= numGridY)
                    {
                        jnd = ix + iy * numGridX;
                        jsq = 0 as ::core::ffi::c_int;
                        while jsq < *numInSquare.offset(jnd as isize) && numNeigh < sMaxAngleNeigh {
                            jpt = *pointLists
                                .offset((*squareInd.offset(jnd as isize) + jsq) as isize);
                            if !(ipt == jpt
                                || markersInGroup != 0
                                    && *group.offset(jpt as isize) < 0 as ::core::ffi::c_int)
                            {
                                pdx = *xrot.offset(ipt as isize) - *xrot.offset(jpt as isize);
                                pdy = *yrot.offset(ipt as isize) - *yrot.offset(jpt as isize);
                                angle = (atan2(
                                    fabs(
                                        (*zrot.offset(ipt as isize) - *zrot.offset(jpt as isize))
                                            as ::core::ffi::c_double,
                                    ),
                                    sqrt((pdx * pdx + pdy * pdy) as ::core::ffi::c_double),
                                ) / DTOR)
                                    as ::core::ffi::c_float;
                                if angle > *steepAngle.offset(ipt as isize) {
                                    *steepAngle.offset(ipt as isize) = angle;
                                    *steepNeigh.offset(ipt as isize) = jpt;
                                }
                                numNeigh += 1;
                            }
                            jsq += 1;
                        }
                    }
                    jdxy += 1;
                }
            }
            isq += 1;
        }
        ind += 1;
    }
    i = 0 as ::core::ffi::c_int;
    while i < numPts {
        *group.offset(i as isize) = 0 as ::core::ffi::c_int;
        i += 1;
    }
    rs_sort_indexed_floats(steepAngle, sortInd, numPts);
    if sDebugLevel > 1 as ::core::ffi::c_int {
        i = 0 as ::core::ffi::c_int;
        while i < numPts {
            ipt = *sortInd.offset(i as isize);
            printf(
                b"pt %d  neigh %d  angle %f\n\0" as *const u8 as *const ::core::ffi::c_char,
                ipt,
                *steepNeigh.offset(ipt as isize),
                *steepAngle.offset(ipt as isize) as ::core::ffi::c_double,
            );
            i += 1;
        }
    }
    verySteepest =
        *steepAngle.offset(*sortInd.offset((numPts - 1 as ::core::ffi::c_int) as isize) as isize);
    firstSteep = numPts - 1 as ::core::ffi::c_int;
    numSteep = 1 as ::core::ffi::c_int;
    ind = numPts - 2 as ::core::ffi::c_int;
    while ind >= 0 as ::core::ffi::c_int {
        angle = *steepAngle.offset(*sortInd.offset(ind as isize) as isize);
        if angle < sSteepestRatio * verySteepest && numSteep >= sNumMinForAmax
            || angle < sAngleMax && numSteep >= sNumMinForArelax
            || angle < sAngleRelax
        {
            break;
        }
        ifdup = 0 as ::core::ffi::c_int;
        i = ind + 1 as ::core::ffi::c_int;
        while i < numPts {
            if *steepAngle.offset(*sortInd.offset(i as isize) as isize) as ::core::ffi::c_double
                > angle as ::core::ffi::c_double + 1.0e-5f64
            {
                break;
            }
            if *sortInd.offset(i as isize)
                == *steepNeigh.offset(*sortInd.offset(ind as isize) as isize)
                && *sortInd.offset(ind as isize)
                    == *steepNeigh.offset(*sortInd.offset(i as isize) as isize)
            {
                ifdup = 1 as ::core::ffi::c_int;
                if sDebugLevel > 1 as ::core::ffi::c_int {
                    printf(
                        b"Duplicate  %d  %d  %f\n\0" as *const u8 as *const ::core::ffi::c_char,
                        *sortInd.offset(ind as isize),
                        *sortInd.offset(i as isize),
                        *steepAngle.offset(*sortInd.offset(ind as isize) as isize)
                            as ::core::ffi::c_double,
                    );
                }
                *steepAngle.offset(*sortInd.offset(ind as isize) as isize) =
                    -1.0f64 as ::core::ffi::c_float;
                break;
            } else {
                i += 1;
            }
        }
        if !(ifdup != 0) {
            numSteep += 1;
            firstSteep = ind;
        }
        ind -= 1;
    }
    delZ = malloc(
        ((numPts - firstSteep) as size_t)
            .wrapping_mul(::core::mem::size_of::<::core::ffi::c_float>() as size_t),
    ) as *mut ::core::ffi::c_float;
    outlie = malloc(
        ((numPts - firstSteep) as size_t)
            .wrapping_mul(::core::mem::size_of::<::core::ffi::c_float>() as size_t),
    ) as *mut ::core::ffi::c_float;
    cluster = malloc(
        ((numPts - firstSteep) as size_t)
            .wrapping_mul(::core::mem::size_of::<::core::ffi::c_int>() as size_t),
    ) as *mut ::core::ffi::c_int;
    clusterSX = malloc(
        ((numPts - firstSteep) as size_t)
            .wrapping_mul(::core::mem::size_of::<::core::ffi::c_int>() as size_t),
    ) as *mut ::core::ffi::c_int;
    clusterSY = malloc(
        ((numPts - firstSteep) as size_t)
            .wrapping_mul(::core::mem::size_of::<::core::ffi::c_int>() as size_t),
    ) as *mut ::core::ffi::c_int;
    if delZ.is_null()
        || outlie.is_null()
        || cluster.is_null()
        || clusterSX.is_null()
        || clusterSY.is_null()
    {
        return 1 as ::core::ffi::c_int;
    }
    i = 0 as ::core::ffi::c_int;
    ind = firstSteep;
    while ind < numPts {
        if *steepAngle.offset(*sortInd.offset(ind as isize) as isize)
            > 0 as ::core::ffi::c_int as ::core::ffi::c_float
        {
            let fresh3 = i;
            i = i + 1;
            *delZ.offset(fresh3 as isize) =
                fabs(
                    (*zrot.offset(*sortInd.offset(ind as isize) as isize)
                        - *zrot.offset(
                            *steepNeigh.offset(*sortInd.offset(ind as isize) as isize) as isize
                        )) as ::core::ffi::c_double,
                ) as ::core::ffi::c_float;
        }
        ind += 1;
    }
    rs_median(delZ, numSteep, outlie, &raw mut medianDelZ);
    if sDebugLevel != 0 {
        printf(
            b"Steep pairs: n = %d  median delz = %f\n\0" as *const u8 as *const ::core::ffi::c_char,
            i,
            medianDelZ as ::core::ffi::c_double,
        );
    }
    if numSteep > 2 as ::core::ffi::c_int {
        rs_mad_median_outliers(delZ, numSteep, sOutlierCrit, outlie);
        i = 0 as ::core::ffi::c_int;
        ind = firstSteep;
        while ind < numPts {
            if *steepAngle.offset(*sortInd.offset(ind as isize) as isize)
                > 0 as ::core::ffi::c_int as ::core::ffi::c_float
            {
                if *outlie.offset(i as isize) as ::core::ffi::c_double != 0.0f64 {
                    *steepAngle.offset(*sortInd.offset(ind as isize) as isize) =
                        -1.0f64 as ::core::ffi::c_float;
                    numSteep -= 1;
                }
                i += 1;
            }
            ind += 1;
        }
    }
    numCluster = 0 as ::core::ffi::c_int;
    numDone = 0 as ::core::ffi::c_int;
    loop {
        ind = numPts - 1 as ::core::ffi::c_int;
        while ind >= firstSteep {
            if *steepAngle.offset(*sortInd.offset(ind as isize) as isize)
                > 0 as ::core::ffi::c_int as ::core::ffi::c_float
            {
                break;
            }
            ind -= 1;
        }
        if ind < firstSteep {
            break;
        }
        numInClust = 2 as ::core::ffi::c_int;
        ipt = *sortInd.offset(ind as isize);
        jpt = *steepNeigh.offset(ipt as isize);
        *cluster.offset(0 as ::core::ffi::c_int as isize) = ipt;
        *cluster.offset(1 as ::core::ffi::c_int as isize) = jpt;
        *group.offset(ipt as isize) = if *zrot.offset(ipt as isize) < *zrot.offset(jpt as isize) {
            1 as ::core::ffi::c_int
        } else {
            2 as ::core::ffi::c_int
        };
        *group.offset(jpt as isize) = 3 as ::core::ffi::c_int - *group.offset(ipt as isize);
        checkInd = 0 as ::core::ffi::c_int;
        *steepAngle.offset(ipt as isize) = -1.0f64 as ::core::ffi::c_float;
        while checkInd < numInClust {
            jnd = ind - 1 as ::core::ffi::c_int;
            while jnd >= firstSteep {
                ipt = *sortInd.offset(jnd as isize);
                jpt = *steepNeigh.offset(ipt as isize);
                if *steepAngle.offset(ipt as isize)
                    > 0 as ::core::ffi::c_int as ::core::ffi::c_float
                    && (ipt == *cluster.offset(checkInd as isize)
                        || jpt == *cluster.offset(checkInd as isize))
                {
                    newpt = ipt;
                    oldpt = jpt;
                    if ipt == *cluster.offset(checkInd as isize) {
                        oldpt = ipt;
                        newpt = jpt;
                    }
                    knd = if *zrot.offset(oldpt as isize) < *zrot.offset(newpt as isize) {
                        1 as ::core::ffi::c_int
                    } else {
                        2 as ::core::ffi::c_int
                    };
                    if knd != *group.offset(oldpt as isize) {
                        printf(
                            b"INCONSISTENCY IN INITIAL STEEP PAIRS IN SURFACE SORT.\n\0"
                                as *const u8
                                as *const ::core::ffi::c_char,
                        );
                    } else {
                        ifdup = 0 as ::core::ffi::c_int;
                        knd = if *zrot.offset(oldpt as isize) >= *zrot.offset(newpt as isize) {
                            1 as ::core::ffi::c_int
                        } else {
                            2 as ::core::ffi::c_int
                        };
                        j = 0 as ::core::ffi::c_int;
                        while j < numInClust {
                            if newpt == *cluster.offset(j as isize) {
                                ifdup = 1 as ::core::ffi::c_int;
                                if knd != *group.offset(newpt as isize) {
                                    printf(
                                        b"INCONSISTENCY IN INITIAL STEEP PAIRS IN SURFACE SORT.\n\0"
                                            as *const u8
                                            as *const ::core::ffi::c_char,
                                    );
                                }
                                break;
                            } else {
                                j += 1;
                            }
                        }
                        if ifdup == 0 {
                            let fresh4 = numInClust;
                            numInClust = numInClust + 1;
                            *cluster.offset(fresh4 as isize) = newpt;
                            *group.offset(newpt as isize) = knd;
                        }
                    }
                    *steepAngle.offset(ipt as isize) = -1.0f64 as ::core::ffi::c_float;
                }
                jnd -= 1;
            }
            checkInd += 1;
        }
        nbot = 0 as ::core::ffi::c_int;
        ntop = 0 as ::core::ffi::c_int;
        zbot = 0.0f32;
        ztop = 0.0f32;
        xsum = 0.0f32;
        ysum = 0.0f32;
        i = 0 as ::core::ffi::c_int;
        while i < numInClust {
            ipt = *cluster.offset(i as isize);
            if *group.offset(ipt as isize) == 2 as ::core::ffi::c_int {
                ntop += 1;
                ztop += *zrot.offset(ipt as isize);
            } else {
                nbot += 1;
                zbot += *zrot.offset(ipt as isize);
            }
            xsum += *xrot.offset(ipt as isize);
            ysum += *yrot.offset(ipt as isize);
            i += 1;
        }
        *delZ.offset(numCluster as isize) =
            ztop / ntop as ::core::ffi::c_float - zbot / nbot as ::core::ffi::c_float;
        *clusterSX.offset(numCluster as isize) = ((xsum / numInClust as ::core::ffi::c_float
            - xmin)
            / sGridSpacing) as ::core::ffi::c_int;
        let fresh5 = numCluster;
        numCluster = numCluster + 1;
        *clusterSY.offset(fresh5 as isize) = ((ysum / numInClust as ::core::ffi::c_float - ymin)
            / sGridSpacing) as ::core::ffi::c_int;
        numDone += numInClust;
        if sDebugLevel != 0 {
            printf(
                b"cluster %d  num  %d  delz  %f  sx, sy %d %d, done %d\n\0" as *const u8
                    as *const ::core::ffi::c_char,
                numCluster,
                numInClust,
                *delZ.offset((numCluster - 1 as ::core::ffi::c_int) as isize)
                    as ::core::ffi::c_double,
                *clusterSX.offset((numCluster - 1 as ::core::ffi::c_int) as isize),
                *clusterSY.offset((numCluster - 1 as ::core::ffi::c_int) as isize),
                numDone,
            );
        }
        if sDebugLevel > 1 as ::core::ffi::c_int {
            i = 0 as ::core::ffi::c_int;
            while i < numInClust {
                printf(
                    b"%d  %.1f  %.1f  %.1f  %d\n\0" as *const u8 as *const ::core::ffi::c_char,
                    *cluster.offset(i as isize),
                    *xrot.offset(*cluster.offset(i as isize) as isize) as ::core::ffi::c_double,
                    *yrot.offset(*cluster.offset(i as isize) as isize) as ::core::ffi::c_double,
                    *zrot.offset(*cluster.offset(i as isize) as isize) as ::core::ffi::c_double,
                    *group.offset(*cluster.offset(i as isize) as isize),
                );
                i += 1;
            }
        }
    }
    numErr = 0 as ::core::ffi::c_int;
    errmax = -1000.0f64 as ::core::ffi::c_float;
    errsq = 0.0f32;
    errsum = errsq;
    round = 0 as ::core::ffi::c_int;
    while round < 2 as ::core::ffi::c_int {
        ring = 0 as ::core::ffi::c_int;
        while ring < numRings && numDone < numPts {
            ind = 0 as ::core::ffi::c_int;
            while ind < numCluster && numDone < numPts {
                sx = *clusterSX.offset(ind as isize);
                sy = *clusterSY.offset(ind as isize);
                jdxy = *ringStart.offset(ring as isize);
                while jdxy < *ringStart.offset((ring + 1 as ::core::ffi::c_int) as isize)
                    && numDone < numPts
                {
                    ix = sx + *idx.offset(jdxy as isize) as ::core::ffi::c_int;
                    iy = sy + *idy.offset(jdxy as isize) as ::core::ffi::c_int;
                    if !(ix < 0 as ::core::ffi::c_int
                        || ix >= numGridX
                        || iy < 0 as ::core::ffi::c_int
                        || iy >= numGridY)
                    {
                        jnd = ix + iy * numGridX;
                        if !(*squareDone.offset(jnd as isize) != 0) {
                            *squareDone.offset(jnd as isize) = 1 as ::core::ffi::c_uchar;
                            jsq = 0 as ::core::ffi::c_int;
                            while jsq < *numInSquare.offset(jnd as isize) && numDone < numPts {
                                jpt = *pointLists
                                    .offset((*squareInd.offset(jnd as isize) + jsq) as isize);
                                if *group.offset(jpt as isize) == 0 {
                                    *squareDone.offset(jnd as isize) = 0 as ::core::ffi::c_uchar;
                                    maxRings = floor(
                                        (sMaxFitDist / sGridSpacing) as ::core::ffi::c_double
                                            + 0.5f64,
                                    )
                                        as ::core::ffi::c_int;
                                    maxRings = if maxRings < numRings {
                                        maxRings
                                    } else {
                                        numRings
                                    };
                                    nfit = 0 as ::core::ffi::c_int;
                                    grpsum = 0 as ::core::ffi::c_int;
                                    tx = ((*xrot.offset(jpt as isize) - xmin) / sGridSpacing)
                                        as ::core::ffi::c_int;
                                    ty = ((*yrot.offset(jpt as isize) - ymin) / sGridSpacing)
                                        as ::core::ffi::c_int;
                                    jring = 0 as ::core::ffi::c_int;
                                    while (jring < maxRings || nfit < 2 as ::core::ffi::c_int)
                                        && nfit < sMaxNumFit
                                    {
                                        kdxy = *ringStart.offset(jring as isize);
                                        while kdxy
                                            < *ringStart
                                                .offset((jring + 1 as ::core::ffi::c_int) as isize)
                                            && nfit < sMaxNumFit
                                        {
                                            jx = tx
                                                + *idx.offset(kdxy as isize) as ::core::ffi::c_int;
                                            jy = ty
                                                + *idy.offset(kdxy as isize) as ::core::ffi::c_int;
                                            if !(jx < 0 as ::core::ffi::c_int
                                                || jx >= numGridX
                                                || jy < 0 as ::core::ffi::c_int
                                                || jy >= numGridY)
                                            {
                                                knd = jx + jy * numGridX;
                                                isq = 0 as ::core::ffi::c_int;
                                                while isq < *numInSquare.offset(knd as isize)
                                                    && nfit < sMaxNumFit
                                                {
                                                    kpt = *pointLists.offset(
                                                        (*squareInd.offset(knd as isize) + isq)
                                                            as isize,
                                                    );
                                                    if *group.offset(kpt as isize) != 0 {
                                                        *xfit.offset(nfit as isize) =
                                                            *xrot.offset(kpt as isize);
                                                        *yfit.offset(nfit as isize) =
                                                            *yrot.offset(kpt as isize);
                                                        *zfit.offset(nfit as isize) =
                                                            *zrot.offset(kpt as isize);
                                                        let fresh6 = nfit;
                                                        nfit = nfit + 1;
                                                        *grpfit.offset(fresh6 as isize) = (*group
                                                            .offset(kpt as isize)
                                                            - 1 as ::core::ffi::c_int)
                                                            as ::core::ffi::c_float;
                                                        grpsum += *group.offset(kpt as isize)
                                                            - 1 as ::core::ffi::c_int;
                                                    }
                                                    isq += 1;
                                                }
                                            }
                                            kdxy += 1;
                                        }
                                        jring += 1;
                                    }
                                    if sDebugLevel > 1 as ::core::ffi::c_int {
                                        printf(
                                            b"For %d  %.1f %.1f  %.1f  nfit %d  ntop %d\n\0"
                                                as *const u8
                                                as *const ::core::ffi::c_char,
                                            jpt,
                                            *xrot.offset(jpt as isize) as ::core::ffi::c_double,
                                            *yrot.offset(jpt as isize) as ::core::ffi::c_double,
                                            *zrot.offset(jpt as isize) as ::core::ffi::c_double,
                                            nfit,
                                            grpsum,
                                        );
                                    }
                                    if nfit >= sBiplaneMinFit && grpsum != 0 && grpsum != nfit {
                                        ls_fit3(
                                            xfit,
                                            yfit,
                                            grpfit,
                                            zfit,
                                            nfit,
                                            &raw mut a1,
                                            &raw mut a2,
                                            &raw mut dzfit,
                                            &raw mut con,
                                        );
                                        zbot = a1 * *xrot.offset(jpt as isize)
                                            + a2 * *yrot.offset(jpt as isize)
                                            + con;
                                        ztop = zbot + dzfit;
                                        if sDebugLevel > 1 as ::core::ffi::c_int {
                                            printf(
                                                b"fit3 %f  %f %f %f\n\0" as *const u8
                                                    as *const ::core::ffi::c_char,
                                                a1 as ::core::ffi::c_double,
                                                a2 as ::core::ffi::c_double,
                                                dzfit as ::core::ffi::c_double,
                                                con as ::core::ffi::c_double,
                                            );
                                        }
                                        if grpsum > 1 as ::core::ffi::c_int
                                            && nfit - grpsum > 1 as ::core::ffi::c_int
                                        {
                                            *delZ.offset(ind as isize) = dzfit;
                                        }
                                    } else {
                                        keepGroup = 1 as ::core::ffi::c_int;
                                        if grpsum <= nfit / 2 as ::core::ffi::c_int {
                                            keepGroup = 0 as ::core::ffi::c_int;
                                        }
                                        if keepGroup != 0 && grpsum >= sPlaneMinFit
                                            || keepGroup == 0 && nfit - grpsum >= sPlaneMinFit
                                        {
                                            if grpsum != 0 && grpsum != nfit {
                                                j = 0 as ::core::ffi::c_int;
                                                i = 0 as ::core::ffi::c_int;
                                                while i < nfit {
                                                    if *grpfit.offset(i as isize)
                                                        == keepGroup as ::core::ffi::c_float
                                                    {
                                                        *xfit.offset(j as isize) =
                                                            *xfit.offset(i as isize);
                                                        *yfit.offset(j as isize) =
                                                            *yfit.offset(i as isize);
                                                        *zfit.offset(j as isize) =
                                                            *zfit.offset(i as isize);
                                                        let fresh7 = j;
                                                        j = j + 1;
                                                        *grpfit.offset(fresh7 as isize) =
                                                            *grpfit.offset(i as isize);
                                                    }
                                                    i += 1;
                                                }
                                                nfit = j;
                                                grpsum = j * keepGroup;
                                            }
                                            ls_fit2(
                                                xfit,
                                                yfit,
                                                zfit,
                                                nfit,
                                                &raw mut a1,
                                                &raw mut a2,
                                                &raw mut con,
                                            );
                                            ztop = a1 * *xrot.offset(jpt as isize)
                                                + a2 * *yrot.offset(jpt as isize)
                                                + con;
                                            zbot = ztop;
                                            if keepGroup != 0 {
                                                zbot -= *delZ.offset(ind as isize);
                                            } else {
                                                ztop += *delZ.offset(ind as isize);
                                            }
                                            if sDebugLevel > 1 as ::core::ffi::c_int {
                                                printf(
                                                    b"fit2  %f  %f  %f  zbot %.1f  ztop  %.1f\n\0"
                                                        as *const u8
                                                        as *const ::core::ffi::c_char,
                                                    a1 as ::core::ffi::c_double,
                                                    a2 as ::core::ffi::c_double,
                                                    con as ::core::ffi::c_double,
                                                    zbot as ::core::ffi::c_double,
                                                    ztop as ::core::ffi::c_double,
                                                );
                                            }
                                        } else {
                                            nbot = 0 as ::core::ffi::c_int;
                                            ntop = 0 as ::core::ffi::c_int;
                                            zbot = 0.0f32;
                                            ztop = 0.0f32;
                                            i = 0 as ::core::ffi::c_int;
                                            while i < nfit {
                                                if *grpfit.offset(i as isize) != 0. {
                                                    ntop += 1;
                                                    ztop += *zfit.offset(i as isize);
                                                } else {
                                                    nbot += 1;
                                                    zbot += *zfit.offset(i as isize);
                                                }
                                                i += 1;
                                            }
                                            if nbot != 0 {
                                                zbot /= nbot as ::core::ffi::c_float;
                                            }
                                            if ntop != 0 {
                                                ztop /= ntop as ::core::ffi::c_float;
                                            }
                                            if nbot != 0 && ntop == 0 {
                                                ztop = zbot + *delZ.offset(ind as isize);
                                            }
                                            if nbot == 0 && ntop != 0 {
                                                zbot = ztop - *delZ.offset(ind as isize);
                                            }
                                            if sDebugLevel > 1 as ::core::ffi::c_int {
                                                printf(
                                                    b"means  %d  %d  zbot %.1f  ztop  %.1f\n\0"
                                                        as *const u8
                                                        as *const ::core::ffi::c_char,
                                                    nbot,
                                                    ntop,
                                                    zbot as ::core::ffi::c_double,
                                                    ztop as ::core::ffi::c_double,
                                                );
                                            }
                                        }
                                    }
                                    if fabs(
                                        (*zrot.offset(jpt as isize) - zbot)
                                            as ::core::ffi::c_double,
                                    ) < fabs(
                                        (*zrot.offset(jpt as isize) - ztop)
                                            as ::core::ffi::c_double,
                                    ) {
                                        *group.offset(jpt as isize) = 1 as ::core::ffi::c_int;
                                        xsum = (100.0f64
                                            * (*zrot.offset(jpt as isize) - zbot)
                                                as ::core::ffi::c_double
                                            / (ztop - zbot) as ::core::ffi::c_double)
                                            as ::core::ffi::c_float;
                                    } else {
                                        *group.offset(jpt as isize) = 2 as ::core::ffi::c_int;
                                        xsum = (100.0f64
                                            * (ztop - *zrot.offset(jpt as isize))
                                                as ::core::ffi::c_double
                                            / (ztop - zbot) as ::core::ffi::c_double)
                                            as ::core::ffi::c_float;
                                    }
                                    if round != 0 || xsum < sMaxRound1Dist {
                                        if sDebugLevel > 1 as ::core::ffi::c_int {
                                            printf(
                                                b"Assign to %d   distance %.1f%%\n\0" as *const u8
                                                    as *const ::core::ffi::c_char,
                                                *group.offset(jpt as isize),
                                                xsum as ::core::ffi::c_double,
                                            );
                                        }
                                        numDone += 1;
                                        errsum += xsum;
                                        errmax = if errmax > xsum { errmax } else { xsum };
                                        errsq += xsum * xsum;
                                        numErr += 1;
                                    } else {
                                        if sDebugLevel > 1 as ::core::ffi::c_int {
                                            printf(
                                                b"Defer %d because distance is %.1f%%\n\0"
                                                    as *const u8
                                                    as *const ::core::ffi::c_char,
                                                jpt,
                                                xsum as ::core::ffi::c_double,
                                            );
                                        }
                                        *group.offset(jpt as isize) = 0 as ::core::ffi::c_int;
                                    }
                                }
                                jsq += 1;
                            }
                        }
                    }
                    jdxy += 1;
                }
                ind += 1;
            }
            ring += 1;
        }
        round += 1;
    }
    sums_to_avg_sd(errsum, errsq, numErr, &raw mut xsum, &raw mut ysum);
    if sDebugLevel != 0 {
        printf(
            b"Distance from nearest plane for %d points: mean %.1f%%  SD %.1f%%  max %.1f%%\n\0"
                as *const u8 as *const ::core::ffi::c_char,
            numErr,
            xsum as ::core::ffi::c_double,
            ysum as ::core::ffi::c_double,
            errmax as ::core::ffi::c_double,
        );
        fflush(stdout);
    }
    free(xrot as *mut ::core::ffi::c_void);
    free(yrot as *mut ::core::ffi::c_void);
    free(zrot as *mut ::core::ffi::c_void);
    free(idx as *mut ::core::ffi::c_void);
    free(idy as *mut ::core::ffi::c_void);
    free(ringStart as *mut ::core::ffi::c_void);
    free(numInSquare as *mut ::core::ffi::c_void);
    free(squareInd as *mut ::core::ffi::c_void);
    free(pointLists as *mut ::core::ffi::c_void);
    free(steepNeigh as *mut ::core::ffi::c_void);
    free(sortInd as *mut ::core::ffi::c_void);
    free(squareDone as *mut ::core::ffi::c_void);
    free(steepAngle as *mut ::core::ffi::c_void);
    free(xfit as *mut ::core::ffi::c_void);
    free(yfit as *mut ::core::ffi::c_void);
    free(zfit as *mut ::core::ffi::c_void);
    free(grpfit as *mut ::core::ffi::c_void);
    free(delZ as *mut ::core::ffi::c_void);
    free(outlie as *mut ::core::ffi::c_void);
    free(cluster as *mut ::core::ffi::c_void);
    free(clusterSY as *mut ::core::ffi::c_void);
    free(clusterSX as *mut ::core::ffi::c_void);
    fflush(stdout);
    return 0 as ::core::ffi::c_int;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn surface_sort_fortran(
    mut xyz: *mut ::core::ffi::c_float,
    mut numPts: *mut ::core::ffi::c_int,
    mut markersInGroup: *mut ::core::ffi::c_int,
    mut group: *mut ::core::ffi::c_int,
) -> ::core::ffi::c_int {
    return surface_sort(xyz, *numPts, *markersInGroup, group);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parameter_dispatch_and_small_point_source_path() {
        unsafe {
            assert_eq!(set_surf_sort_param(0, 25.), 0);
            assert_eq!(set_surf_sort_param(14, 0.), 1);
            let mut xyz = [0_f32; 6];
            let mut group = [0_i32; 2];
            assert_eq!(surface_sort(xyz.as_mut_ptr(), 2, 0, group.as_mut_ptr()), 0);
            assert_eq!(group, [1, 2]);
        }
    }

    #[test]
    fn fortran_parameter_wrapper_preserves_rounding() {
        unsafe {
            let mut index = 1;
            let mut value = 7.6;
            assert_eq!(set_surf_sort_param_fortran(&mut index, &mut value), 0);
            assert_eq!(core::ptr::addr_of!(sMaxAngleNeigh).read(), 8);
            let mut index = 1;
            let mut value = 50.;
            set_surf_sort_param_fortran(&mut index, &mut value);
        }
    }
}
