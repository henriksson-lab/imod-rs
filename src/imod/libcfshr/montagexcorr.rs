unsafe extern "C" {
    static mut stdout: *mut FILE;
    fn fflush(__stream: *mut FILE) -> ::core::ffi::c_int;
    fn printf(__format: *const ::core::ffi::c_char, ...) -> ::core::ffi::c_int;
    fn sprintf(
        __s: *mut ::core::ffi::c_char,
        __format: *const ::core::ffi::c_char,
        ...
    ) -> ::core::ffi::c_int;
    fn exp(__x: ::core::ffi::c_double) -> ::core::ffi::c_double;
    fn pow(__x: ::core::ffi::c_double, __y: ::core::ffi::c_double) -> ::core::ffi::c_double;
    fn sqrt(__x: ::core::ffi::c_double) -> ::core::ffi::c_double;
    fn fabs(__x: ::core::ffi::c_double) -> ::core::ffi::c_double;
    fn floor(__x: ::core::ffi::c_double) -> ::core::ffi::c_double;
    fn malloc(__size: size_t) -> *mut ::core::ffi::c_void;
    fn free(__ptr: *mut ::core::ffi::c_void);
    fn memcpy(
        __dest: *mut ::core::ffi::c_void,
        __src: *const ::core::ffi::c_void,
        __n: size_t,
    ) -> *mut ::core::ffi::c_void;
    fn strchr(__s: *const ::core::ffi::c_char, __c: ::core::ffi::c_int)
    -> *mut ::core::ffi::c_char;
    fn strlen(__s: *const ::core::ffi::c_char) -> size_t;
    fn get_sample_of_array(
        image: *mut ::core::ffi::c_void,
        mode: ::core::ffi::c_int,
        nx: ::core::ffi::c_int,
        ny: ::core::ffi::c_int,
        sampleFrac: ::core::ffi::c_float,
        ixStart: ::core::ffi::c_int,
        iyStart: ::core::ffi::c_int,
        nxUse: ::core::ffi::c_int,
        nyUse: ::core::ffi::c_int,
        filltoExclude: ::core::ffi::c_float,
        samples: *mut ::core::ffi::c_float,
        maxSamples: ::core::ffi::c_int,
        numSamples: *mut ::core::ffi::c_int,
    ) -> ::core::ffi::c_int;
    fn nice_frame(
        num: ::core::ffi::c_int,
        idnum: ::core::ffi::c_int,
        limit: ::core::ffi::c_int,
    ) -> ::core::ffi::c_int;
    fn xcorr_set_ctf(
        sigma1: ::core::ffi::c_float,
        sigma2: ::core::ffi::c_float,
        radius1: ::core::ffi::c_float,
        radius2: ::core::ffi::c_float,
        ctf: *mut ::core::ffi::c_float,
        nx: ::core::ffi::c_int,
        ny: ::core::ffi::c_int,
        delta: *mut ::core::ffi::c_float,
    );
    fn xcorr_filter_part(
        fft: *mut ::core::ffi::c_float,
        array: *mut ::core::ffi::c_float,
        nx: ::core::ffi::c_int,
        ny: ::core::ffi::c_int,
        ctf: *mut ::core::ffi::c_float,
        delta: ::core::ffi::c_float,
    );
    fn xcorr_mean_zero(
        array: *mut ::core::ffi::c_float,
        nxdim: ::core::ffi::c_int,
        nx: ::core::ffi::c_int,
        ny: ::core::ffi::c_int,
    );
    fn xcorr_peak_find(
        array: *mut ::core::ffi::c_float,
        nxdim: ::core::ffi::c_int,
        ny: ::core::ffi::c_int,
        xpeak: *mut ::core::ffi::c_float,
        ypeak: *mut ::core::ffi::c_float,
        peak: *mut ::core::ffi::c_float,
        maxpeaks: ::core::ffi::c_int,
    );
    fn set_peak_find_limits(
        limXlo: ::core::ffi::c_int,
        limXhi: ::core::ffi::c_int,
        limYlo: ::core::ffi::c_int,
        limYhi: ::core::ffi::c_int,
        useEllipse: ::core::ffi::c_int,
    );
    fn parabolic_fit_position(
        y1: ::core::ffi::c_float,
        y2: ::core::ffi::c_float,
        y3: ::core::ffi::c_float,
    ) -> ::core::ffi::c_double;
    fn conjugate_product(
        array: *mut ::core::ffi::c_float,
        brray: *mut ::core::ffi::c_float,
        nx: ::core::ffi::c_int,
        ny: ::core::ffi::c_int,
    );
    fn weighted_corr_from_sums(
        aSum: ::core::ffi::c_double,
        aSumSq: ::core::ffi::c_double,
        bsum: ::core::ffi::c_double,
        bSumSq: ::core::ffi::c_double,
        abSum: ::core::ffi::c_double,
        wSum: ::core::ffi::c_double,
        sumArray: *mut ::core::ffi::c_double,
        descrip: *const ::core::ffi::c_char,
    ) -> ::core::ffi::c_double;
    fn slice_taper_out_pad(
        array: *mut ::core::ffi::c_void,
        type_0: ::core::ffi::c_int,
        nxbox: ::core::ffi::c_int,
        nybox: ::core::ffi::c_int,
        brray: *mut ::core::ffi::c_float,
        nxdim: ::core::ffi::c_int,
        nx: ::core::ffi::c_int,
        ny: ::core::ffi::c_int,
        ifmean: ::core::ffi::c_int,
        dmeanin: ::core::ffi::c_float,
    );
    fn slice_smooth_out_pad(
        array: *mut ::core::ffi::c_void,
        type_0: ::core::ffi::c_int,
        nxbox: ::core::ffi::c_int,
        nybox: ::core::ffi::c_int,
        brray: *mut ::core::ffi::c_float,
        nxdim: ::core::ffi::c_int,
        nx: ::core::ffi::c_int,
        ny: ::core::ffi::c_int,
    );
    fn rs_sort_indexed_floats(
        x: *mut ::core::ffi::c_float,
        index: *mut ::core::ffi::c_int,
        n: ::core::ffi::c_int,
    );
    fn percentile_float(
        s: ::core::ffi::c_int,
        r: *mut ::core::ffi::c_float,
        num: ::core::ffi::c_int,
    ) -> ::core::ffi::c_float;
    fn make_standard_dev_map(
        array: *mut ::core::ffi::c_float,
        nxDim: ::core::ffi::c_int,
        ixStart: ::core::ffi::c_int,
        ixEnd: ::core::ffi::c_int,
        iyStart: ::core::ffi::c_int,
        iyEnd: ::core::ffi::c_int,
        binning: ::core::ffi::c_int,
        boxSize: ::core::ffi::c_int,
        sdArr: *mut ::core::ffi::c_float,
        sumArr: *mut ::core::ffi::c_float,
        sqrArr: *mut ::core::ffi::c_float,
        xOffset: *mut ::core::ffi::c_int,
        yOffset: *mut ::core::ffi::c_int,
    );
    fn b3dIMax(narg: ::core::ffi::c_int, ...) -> ::core::ffi::c_int;
    fn wall_time() -> ::core::ffi::c_double;
    fn num_omp_threads(optimalThreads: ::core::ffi::c_int) -> ::core::ffi::c_int;
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
    pub _markers: *mut core::ffi::c_void,
    pub _chain: *mut _IO_FILE,
    pub _fileno: ::core::ffi::c_int,
    pub _flags2: ::core::ffi::c_int,
    pub _old_offset: __off_t,
    pub _cur_column: ::core::ffi::c_ushort,
    pub _vtable_offset: ::core::ffi::c_schar,
    pub _shortbuf: [::core::ffi::c_char; 1],
    pub _lock: *mut ::core::ffi::c_void,
    pub _offset: __off64_t,
    pub _codecvt: *mut core::ffi::c_void,
    pub _wide_data: *mut core::ffi::c_void,
    pub _freeres_list: *mut _IO_FILE,
    pub _freeres_buf: *mut ::core::ffi::c_void,
    pub __pad5: size_t,
    pub _mode: ::core::ffi::c_int,
    pub _unused2: [::core::ffi::c_char; 20],
}
pub type _IO_lock_t = ();
pub type FILE = _IO_FILE;
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
pub const MONTXC_MAX_PEAKS: ::core::ffi::c_int = 100 as ::core::ffi::c_int;
pub const MONTXC_MAX_DEBUG_LINE: ::core::ffi::c_int = 90 as ::core::ffi::c_int;
pub const MAX_RUNNERS_UP: ::core::ffi::c_int = 2 as ::core::ffi::c_int;
static mut sDistWeightHalfFall: ::core::ffi::c_float = 0.0f32;
static mut sLastTrimmedMaxSD: ::core::ffi::c_float = -1.0f64 as ::core::ffi::c_float;
static mut sLastRunnersUp: [::core::ffi::c_float; 4] = [0.; 4];
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mont_xc_basic_sizes(
    mut ixy: ::core::ffi::c_int,
    mut nbin: ::core::ffi::c_int,
    mut indentXC: ::core::ffi::c_int,
    mut nxyPiece: *mut ::core::ffi::c_int,
    mut nxyOverlap: *mut ::core::ffi::c_int,
    mut aspectMax: ::core::ffi::c_float,
    mut extraWidth: ::core::ffi::c_float,
    mut padFrac: ::core::ffi::c_float,
    mut niceLimit: ::core::ffi::c_int,
    mut indentUse: *mut ::core::ffi::c_int,
    mut nxyBox: *mut ::core::ffi::c_int,
    mut numExtra: *mut ::core::ffi::c_int,
    mut nxPad: *mut ::core::ffi::c_int,
    mut nyPad: *mut ::core::ffi::c_int,
    mut maxLongShift: *mut ::core::ffi::c_int,
) {
    let mut iyx: ::core::ffi::c_int = 0;
    let mut nxyBorder: [::core::ffi::c_int; 2] = [0; 2];
    let mut shiftInOverlap: ::core::ffi::c_int = 0 as ::core::ffi::c_int;
    if ixy > 1 as ::core::ffi::c_int {
        ixy = ixy % 2 as ::core::ffi::c_int;
        shiftInOverlap = if *nxyOverlap.offset((1 as ::core::ffi::c_int - ixy) as isize)
            >= 0 as ::core::ffi::c_int
        {
            *nxyOverlap.offset((1 as ::core::ffi::c_int - ixy) as isize)
        } else {
            -*nxyOverlap.offset((1 as ::core::ffi::c_int - ixy) as isize)
        };
    }
    iyx = 1 as ::core::ffi::c_int - ixy;
    *indentUse = if indentXC
        < (*nxyOverlap.offset(ixy as isize) - 8 as ::core::ffi::c_int) / 2 as ::core::ffi::c_int
    {
        indentXC
    } else {
        (*nxyOverlap.offset(ixy as isize) - 8 as ::core::ffi::c_int) / 2 as ::core::ffi::c_int
    };
    *nxyBox.offset(ixy as isize) =
        (*nxyOverlap.offset(ixy as isize) - *indentUse * 2 as ::core::ffi::c_int) / nbin;
    *nxyBox.offset(iyx as isize) = (if *nxyPiece.offset(iyx as isize)
        - shiftInOverlap
        - (if 2 as ::core::ffi::c_int * nbin
            > *nxyPiece.offset(ixy as isize) / 20 as ::core::ffi::c_int
        {
            2 as ::core::ffi::c_int * nbin
        } else {
            *nxyPiece.offset(ixy as isize) / 20 as ::core::ffi::c_int
        })
        < (aspectMax * *nxyOverlap.offset(ixy as isize) as ::core::ffi::c_float)
            as ::core::ffi::c_int
    {
        *nxyPiece.offset(iyx as isize)
            - shiftInOverlap
            - (if 2 as ::core::ffi::c_int * nbin
                > *nxyPiece.offset(ixy as isize) / 20 as ::core::ffi::c_int
            {
                2 as ::core::ffi::c_int * nbin
            } else {
                *nxyPiece.offset(ixy as isize) / 20 as ::core::ffi::c_int
            })
    } else {
        (aspectMax * *nxyOverlap.offset(ixy as isize) as ::core::ffi::c_float) as ::core::ffi::c_int
    }) / nbin;
    *numExtra.offset(iyx as isize) = 0 as ::core::ffi::c_int;
    *numExtra.offset(ixy as isize) = if 2 as ::core::ffi::c_int
        * (floor(
            (extraWidth * *nxyBox.offset(ixy as isize) as ::core::ffi::c_float)
                as ::core::ffi::c_double
                + 0.5f64,
        ) as ::core::ffi::c_int
            / 2 as ::core::ffi::c_int)
        < (*nxyPiece.offset(ixy as isize)
            - (3 as ::core::ffi::c_int)
                .max(nbin)
                .max(*indentUse)
                .max(*nxyPiece.offset(ixy as isize) / 20 as ::core::ffi::c_int)
                * 2 as ::core::ffi::c_int)
            / nbin
            - *nxyBox.offset(ixy as isize)
    {
        2 as ::core::ffi::c_int
            * (floor(
                (extraWidth * *nxyBox.offset(ixy as isize) as ::core::ffi::c_float)
                    as ::core::ffi::c_double
                    + 0.5f64,
            ) as ::core::ffi::c_int
                / 2 as ::core::ffi::c_int)
    } else {
        (*nxyPiece.offset(ixy as isize)
            - (3 as ::core::ffi::c_int)
                .max(nbin)
                .max(*indentUse)
                .max(*nxyPiece.offset(ixy as isize) / 20 as ::core::ffi::c_int)
                * 2 as ::core::ffi::c_int)
            / nbin
            - *nxyBox.offset(ixy as isize)
    };
    *nxyBox.offset(ixy as isize) = *nxyBox.offset(ixy as isize) + *numExtra.offset(ixy as isize);
    *maxLongShift = floor(
        (if 1.9f64 * *nxyOverlap.offset(ixy as isize) as ::core::ffi::c_double
            / nbin as ::core::ffi::c_double
            > 1.5f64 * *nxyBox.offset(ixy as isize) as ::core::ffi::c_double
        {
            1.9f64 * *nxyOverlap.offset(ixy as isize) as ::core::ffi::c_double
                / nbin as ::core::ffi::c_double
        } else {
            1.5f64 * *nxyBox.offset(ixy as isize) as ::core::ffi::c_double
        }) + 0.5f64,
    ) as ::core::ffi::c_int;
    nxyBorder[ixy as usize] = if 5 as ::core::ffi::c_int
        > floor(
            (padFrac * *nxyBox.offset(ixy as isize) as ::core::ffi::c_float)
                as ::core::ffi::c_double
                + 0.5f64,
        ) as ::core::ffi::c_int
    {
        5 as ::core::ffi::c_int
    } else {
        floor(
            (padFrac * *nxyBox.offset(ixy as isize) as ::core::ffi::c_float)
                as ::core::ffi::c_double
                + 0.5f64,
        ) as ::core::ffi::c_int
    };
    nxyBorder[iyx as usize] = if (if 5 as ::core::ffi::c_int
        > floor(
            (padFrac * *nxyBox.offset(iyx as isize) as ::core::ffi::c_float)
                as ::core::ffi::c_double
                + 0.5f64,
        ) as ::core::ffi::c_int
    {
        5 as ::core::ffi::c_int
    } else {
        floor(
            (padFrac * *nxyBox.offset(iyx as isize) as ::core::ffi::c_float)
                as ::core::ffi::c_double
                + 0.5f64,
        ) as ::core::ffi::c_int
    }) < (if 5 as ::core::ffi::c_int
        > floor(0.45f64 * *maxLongShift as ::core::ffi::c_double + 0.5f64) as ::core::ffi::c_int
    {
        5 as ::core::ffi::c_int
    } else {
        floor(0.45f64 * *maxLongShift as ::core::ffi::c_double + 0.5f64) as ::core::ffi::c_int
    }) {
        if 5 as ::core::ffi::c_int
            > floor(
                (padFrac * *nxyBox.offset(iyx as isize) as ::core::ffi::c_float)
                    as ::core::ffi::c_double
                    + 0.5f64,
            ) as ::core::ffi::c_int
        {
            5 as ::core::ffi::c_int
        } else {
            floor(
                (padFrac * *nxyBox.offset(iyx as isize) as ::core::ffi::c_float)
                    as ::core::ffi::c_double
                    + 0.5f64,
            ) as ::core::ffi::c_int
        }
    } else if 5 as ::core::ffi::c_int
        > floor(0.45f64 * *maxLongShift as ::core::ffi::c_double + 0.5f64) as ::core::ffi::c_int
    {
        5 as ::core::ffi::c_int
    } else {
        floor(0.45f64 * *maxLongShift as ::core::ffi::c_double + 0.5f64) as ::core::ffi::c_int
    };
    *nxPad = crate::imod::libcfshr::filtxcorr::nice_frame(
        *nxyBox.offset(0 as ::core::ffi::c_int as isize)
            + 2 as ::core::ffi::c_int * nxyBorder[0 as ::core::ffi::c_int as usize],
        2 as ::core::ffi::c_int,
        niceLimit,
    );
    *nyPad = crate::imod::libcfshr::filtxcorr::nice_frame(
        *nxyBox.offset(1 as ::core::ffi::c_int as isize)
            + 2 as ::core::ffi::c_int * nxyBorder[1 as ::core::ffi::c_int as usize],
        2 as ::core::ffi::c_int,
        niceLimit,
    );
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn montxcbasicsizes_(
    mut ixy: *mut ::core::ffi::c_int,
    mut nbin: *mut ::core::ffi::c_int,
    mut indentXC: *mut ::core::ffi::c_int,
    mut nxyPiece: *mut ::core::ffi::c_int,
    mut nxyOverlap: *mut ::core::ffi::c_int,
    mut aspectMax: *mut ::core::ffi::c_float,
    mut extraWidth: *mut ::core::ffi::c_float,
    mut padFrac: *mut ::core::ffi::c_float,
    mut niceLimit: *mut ::core::ffi::c_int,
    mut indentUse: *mut ::core::ffi::c_int,
    mut nxyBox: *mut ::core::ffi::c_int,
    mut numExtra: *mut ::core::ffi::c_int,
    mut nxPad: *mut ::core::ffi::c_int,
    mut nyPad: *mut ::core::ffi::c_int,
    mut maxLongShift: *mut ::core::ffi::c_int,
) {
    mont_xc_basic_sizes(
        *ixy - 1 as ::core::ffi::c_int,
        *nbin,
        *indentXC,
        nxyPiece,
        nxyOverlap,
        *aspectMax,
        *extraWidth,
        *padFrac,
        *niceLimit,
        indentUse,
        nxyBox,
        numExtra,
        nxPad,
        nyPad,
        maxLongShift,
    );
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mont_xc_inds_and_ctf(
    mut ixy: ::core::ffi::c_int,
    mut nxyPiece: *mut ::core::ffi::c_int,
    mut nxyOverlap: *mut ::core::ffi::c_int,
    mut nxyBox: *mut ::core::ffi::c_int,
    mut nbin: ::core::ffi::c_int,
    mut indentUse: ::core::ffi::c_int,
    mut numExtra: *mut ::core::ffi::c_int,
    mut nxPad: ::core::ffi::c_int,
    mut nyPad: ::core::ffi::c_int,
    mut numSmooth: ::core::ffi::c_int,
    mut sigma1: ::core::ffi::c_float,
    mut sigma2: ::core::ffi::c_float,
    mut radius1: ::core::ffi::c_float,
    mut radius2: ::core::ffi::c_float,
    mut evalCCC: ::core::ffi::c_int,
    mut ind0Lower: *mut ::core::ffi::c_int,
    mut ind1Lower: *mut ::core::ffi::c_int,
    mut ind0Upper: *mut ::core::ffi::c_int,
    mut ind1Upper: *mut ::core::ffi::c_int,
    mut nxSmooth: *mut ::core::ffi::c_int,
    mut nySmooth: *mut ::core::ffi::c_int,
    mut ctf: *mut ::core::ffi::c_float,
    mut delta: *mut ::core::ffi::c_float,
) {
    let mut iyx: ::core::ffi::c_int = 0;
    let mut shiftInOverlap: ::core::ffi::c_int = 0 as ::core::ffi::c_int;
    let mut longShift: ::core::ffi::c_int = 0 as ::core::ffi::c_int;
    if ixy > 1 as ::core::ffi::c_int {
        ixy = ixy % 2 as ::core::ffi::c_int;
        longShift = 1 as ::core::ffi::c_int;
        shiftInOverlap = -*nxyOverlap.offset((1 as ::core::ffi::c_int - ixy) as isize);
    }
    iyx = 1 as ::core::ffi::c_int - ixy;
    *ind0Lower.offset(iyx as isize) = *nxyPiece.offset(iyx as isize) / 2 as ::core::ffi::c_int
        - shiftInOverlap
        - nbin * *nxyBox.offset(iyx as isize) / 2 as ::core::ffi::c_int;
    *ind1Lower.offset(iyx as isize) = *ind0Lower.offset(iyx as isize)
        + nbin * *nxyBox.offset(iyx as isize)
        - 1 as ::core::ffi::c_int;
    *ind0Lower.offset(ixy as isize) =
        *nxyPiece.offset(ixy as isize) - *nxyOverlap.offset(ixy as isize) + indentUse
            - nbin * *numExtra.offset(ixy as isize);
    *ind1Lower.offset(ixy as isize) = *ind0Lower.offset(ixy as isize)
        + nbin * *nxyBox.offset(ixy as isize)
        - 1 as ::core::ffi::c_int;
    *ind0Upper = indentUse;
    *ind1Upper = indentUse + nbin * *nxyBox.offset(ixy as isize) - 1 as ::core::ffi::c_int;
    if longShift != 0 {
        if ixy != 0 {
            *ind0Upper.offset(1 as ::core::ffi::c_int as isize) =
                *ind0Upper.offset(0 as ::core::ffi::c_int as isize);
            *ind1Upper.offset(1 as ::core::ffi::c_int as isize) =
                *ind1Upper.offset(0 as ::core::ffi::c_int as isize);
        }
        *ind0Upper.offset(iyx as isize) = *ind0Lower.offset(iyx as isize) + shiftInOverlap;
        *ind1Upper.offset(iyx as isize) = *ind1Lower.offset(iyx as isize) + shiftInOverlap;
    }
    *nxSmooth = *nxyBox.offset(0 as ::core::ffi::c_int as isize)
        + (if 2 as ::core::ffi::c_int * numSmooth
            < (nxPad - *nxyBox.offset(0 as ::core::ffi::c_int as isize)) / 2 as ::core::ffi::c_int
        {
            2 as ::core::ffi::c_int * numSmooth
        } else {
            (nxPad - *nxyBox.offset(0 as ::core::ffi::c_int as isize)) / 2 as ::core::ffi::c_int
        });
    *nySmooth = *nxyBox.offset(1 as ::core::ffi::c_int as isize)
        + (if 2 as ::core::ffi::c_int * numSmooth
            < (nyPad - *nxyBox.offset(1 as ::core::ffi::c_int as isize)) / 2 as ::core::ffi::c_int
        {
            2 as ::core::ffi::c_int * numSmooth
        } else {
            (nyPad - *nxyBox.offset(1 as ::core::ffi::c_int as isize)) / 2 as ::core::ffi::c_int
        });
    crate::imod::libcfshr::filtxcorr::xcorr_set_ctf(
        sigma1,
        nbin as ::core::ffi::c_float * sigma2,
        radius1,
        nbin as ::core::ffi::c_float * radius2,
        ctf,
        nxPad,
        nyPad,
        delta,
    );
    if evalCCC != 0 {
        iyx = 0 as ::core::ffi::c_int;
        while iyx < 8193 as ::core::ffi::c_int {
            *ctf.offset(iyx as isize) =
                sqrt(*ctf.offset(iyx as isize) as ::core::ffi::c_double) as ::core::ffi::c_float;
            iyx += 1;
        }
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn montxcindsandctf_(
    mut ixy: *mut ::core::ffi::c_int,
    mut nxyPiece: *mut ::core::ffi::c_int,
    mut nxyOverlap: *mut ::core::ffi::c_int,
    mut nxyBox: *mut ::core::ffi::c_int,
    mut nbin: *mut ::core::ffi::c_int,
    mut indentUse: *mut ::core::ffi::c_int,
    mut numExtra: *mut ::core::ffi::c_int,
    mut nxPad: *mut ::core::ffi::c_int,
    mut nyPad: *mut ::core::ffi::c_int,
    mut numSmooth: *mut ::core::ffi::c_int,
    mut sigma1: *mut ::core::ffi::c_float,
    mut sigma2: *mut ::core::ffi::c_float,
    mut radius1: *mut ::core::ffi::c_float,
    mut radius2: *mut ::core::ffi::c_float,
    mut evalCCC: *mut ::core::ffi::c_int,
    mut ind0Lower: *mut ::core::ffi::c_int,
    mut ind1Lower: *mut ::core::ffi::c_int,
    mut ind0Upper: *mut ::core::ffi::c_int,
    mut ind1Upper: *mut ::core::ffi::c_int,
    mut nxSmooth: *mut ::core::ffi::c_int,
    mut nySmooth: *mut ::core::ffi::c_int,
    mut ctf: *mut ::core::ffi::c_float,
    mut delta: *mut ::core::ffi::c_float,
) {
    mont_xc_inds_and_ctf(
        *ixy - 1 as ::core::ffi::c_int,
        nxyPiece,
        nxyOverlap,
        nxyBox,
        *nbin,
        *indentUse,
        numExtra,
        *nxPad,
        *nyPad,
        *numSmooth,
        *sigma1,
        *sigma2,
        *radius1,
        *radius2,
        *evalCCC,
        ind0Lower,
        ind1Lower,
        ind0Upper,
        ind1Upper,
        nxSmooth,
        nySmooth,
        ctf,
        delta,
    );
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mont_xc_find_binning(
    mut maxBin: ::core::ffi::c_int,
    mut targetSize: ::core::ffi::c_int,
    mut indentXC: ::core::ffi::c_int,
    mut nxyPiece: *mut ::core::ffi::c_int,
    mut nxyOverlap: *mut ::core::ffi::c_int,
    mut aspectMax: ::core::ffi::c_float,
    mut extraWidth: ::core::ffi::c_float,
    mut padFrac: ::core::ffi::c_float,
    mut niceLimit: ::core::ffi::c_int,
    mut numPaddedPix: *mut ::core::ffi::c_int,
    mut numBoxedPix: *mut ::core::ffi::c_int,
) -> ::core::ffi::c_int {
    let mut nxPad: ::core::ffi::c_int = 0;
    let mut nyPad: ::core::ffi::c_int = 0;
    let mut indentUse: ::core::ffi::c_int = 0;
    let mut nxyBox: [::core::ffi::c_int; 2] = [0; 2];
    let mut numExtra: [::core::ffi::c_int; 2] = [0; 2];
    let mut maxLongShift: ::core::ffi::c_int = 0;
    let mut ixy: ::core::ffi::c_int = 0;
    let mut nbin: ::core::ffi::c_int = 0;
    nbin = 1 as ::core::ffi::c_int;
    while nbin <= maxBin {
        *numPaddedPix = 0 as ::core::ffi::c_int;
        *numBoxedPix = 0 as ::core::ffi::c_int;
        ixy = 0 as ::core::ffi::c_int;
        while ixy < 2 as ::core::ffi::c_int {
            mont_xc_basic_sizes(
                ixy,
                nbin,
                indentXC,
                nxyPiece,
                nxyOverlap,
                aspectMax,
                extraWidth,
                padFrac,
                niceLimit,
                &raw mut indentUse,
                (&raw mut nxyBox as *mut ::core::ffi::c_int)
                    .offset(0 as ::core::ffi::c_int as isize)
                    as *mut ::core::ffi::c_int,
                (&raw mut numExtra as *mut ::core::ffi::c_int)
                    .offset(0 as ::core::ffi::c_int as isize)
                    as *mut ::core::ffi::c_int,
                &raw mut nxPad,
                &raw mut nyPad,
                &raw mut maxLongShift,
            );
            *numPaddedPix = if *numPaddedPix
                > (nxPad + 8 as ::core::ffi::c_int) * (nyPad + 8 as ::core::ffi::c_int)
            {
                *numPaddedPix
            } else {
                (nxPad + 8 as ::core::ffi::c_int) * (nyPad + 8 as ::core::ffi::c_int)
            };
            *numBoxedPix = if *numBoxedPix
                > (nxyBox[0 as ::core::ffi::c_int as usize] + 4 as ::core::ffi::c_int)
                    * (nxyBox[1 as ::core::ffi::c_int as usize] + 4 as ::core::ffi::c_int)
            {
                *numBoxedPix
            } else {
                (nxyBox[0 as ::core::ffi::c_int as usize] + 4 as ::core::ffi::c_int)
                    * (nxyBox[1 as ::core::ffi::c_int as usize] + 4 as ::core::ffi::c_int)
            };
            ixy += 1;
        }
        if *numBoxedPix <= targetSize * targetSize {
            return nbin;
        }
        nbin += 1;
    }
    return maxBin;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn montxcfindbinning_(
    mut maxBin: *mut ::core::ffi::c_int,
    mut targetSize: *mut ::core::ffi::c_int,
    mut indentXC: *mut ::core::ffi::c_int,
    mut nxyPiece: *mut ::core::ffi::c_int,
    mut nxyOverlap: *mut ::core::ffi::c_int,
    mut aspectMax: *mut ::core::ffi::c_float,
    mut extraWidth: *mut ::core::ffi::c_float,
    mut padFrac: *mut ::core::ffi::c_float,
    mut niceLimit: *mut ::core::ffi::c_int,
    mut numPaddedPix: *mut ::core::ffi::c_int,
    mut numBoxedPix: *mut ::core::ffi::c_int,
) -> ::core::ffi::c_int {
    return mont_xc_find_binning(
        *maxBin,
        *targetSize,
        *indentXC,
        nxyPiece,
        nxyOverlap,
        *aspectMax,
        *extraWidth,
        *padFrac,
        *niceLimit,
        numPaddedPix,
        numBoxedPix,
    );
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mont_xc_find_binning2(
    mut maxBin: ::core::ffi::c_int,
    mut targetSize: ::core::ffi::c_int,
    mut indentXC: ::core::ffi::c_int,
    mut ixy: ::core::ffi::c_int,
    mut nxyPiece: *mut ::core::ffi::c_int,
    mut nxyOverlap: *mut ::core::ffi::c_int,
    mut expectedShift: *mut ::core::ffi::c_int,
    mut aspectMax: ::core::ffi::c_float,
    mut extraWidth: ::core::ffi::c_float,
    mut padFrac: ::core::ffi::c_float,
    mut niceLimit: ::core::ffi::c_int,
    mut numPaddedPix: *mut ::core::ffi::c_int,
    mut numBoxedPix: *mut ::core::ffi::c_int,
) -> ::core::ffi::c_int {
    let mut nxPad: ::core::ffi::c_int = 0;
    let mut nyPad: ::core::ffi::c_int = 0;
    let mut indentUse: ::core::ffi::c_int = 0;
    let mut nxyBox: [::core::ffi::c_int; 2] = [0; 2];
    let mut numExtra: [::core::ffi::c_int; 2] = [0; 2];
    let mut maxLongShift: ::core::ffi::c_int = 0;
    let mut nbin: ::core::ffi::c_int = 0;
    let mut overlapUse: [::core::ffi::c_int; 2] = [0; 2];
    overlapUse[ixy as usize] = *nxyOverlap.offset(ixy as isize)
        + (if 0 as ::core::ffi::c_int > -*expectedShift.offset(ixy as isize) {
            0 as ::core::ffi::c_int
        } else {
            -*expectedShift.offset(ixy as isize)
        });
    overlapUse[(1 as ::core::ffi::c_int - ixy) as usize] =
        *expectedShift.offset((1 as ::core::ffi::c_int - ixy) as isize);
    nbin = 1 as ::core::ffi::c_int;
    while nbin <= maxBin {
        mont_xc_basic_sizes(
            ixy + 2 as ::core::ffi::c_int,
            nbin,
            indentXC,
            nxyPiece,
            &raw mut overlapUse as *mut ::core::ffi::c_int,
            aspectMax,
            extraWidth,
            padFrac,
            niceLimit,
            &raw mut indentUse,
            (&raw mut nxyBox as *mut ::core::ffi::c_int).offset(0 as ::core::ffi::c_int as isize)
                as *mut ::core::ffi::c_int,
            (&raw mut numExtra as *mut ::core::ffi::c_int).offset(0 as ::core::ffi::c_int as isize)
                as *mut ::core::ffi::c_int,
            &raw mut nxPad,
            &raw mut nyPad,
            &raw mut maxLongShift,
        );
        *numPaddedPix = (nxPad + 8 as ::core::ffi::c_int) * (nyPad + 8 as ::core::ffi::c_int);
        *numBoxedPix = (nxyBox[0 as ::core::ffi::c_int as usize] + 4 as ::core::ffi::c_int)
            * (nxyBox[1 as ::core::ffi::c_int as usize] + 4 as ::core::ffi::c_int);
        if *numBoxedPix <= targetSize * targetSize {
            return nbin;
        }
        nbin += 1;
    }
    return maxBin;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn montxcfindbinning2_(
    mut maxBin: *mut ::core::ffi::c_int,
    mut targetSize: *mut ::core::ffi::c_int,
    mut indentXC: *mut ::core::ffi::c_int,
    mut ixy: *mut ::core::ffi::c_int,
    mut nxyPiece: *mut ::core::ffi::c_int,
    mut nxyOverlap: *mut ::core::ffi::c_int,
    mut expectedShift: *mut ::core::ffi::c_int,
    mut aspectMax: *mut ::core::ffi::c_float,
    mut extraWidth: *mut ::core::ffi::c_float,
    mut padFrac: *mut ::core::ffi::c_float,
    mut niceLimit: *mut ::core::ffi::c_int,
    mut numPaddedPix: *mut ::core::ffi::c_int,
    mut numBoxedPix: *mut ::core::ffi::c_int,
) -> ::core::ffi::c_int {
    return mont_xc_find_binning2(
        *maxBin,
        *targetSize,
        *indentXC,
        *ixy - 1 as ::core::ffi::c_int,
        nxyPiece,
        nxyOverlap,
        expectedShift,
        *aspectMax,
        *extraWidth,
        *padFrac,
        *niceLimit,
        numPaddedPix,
        numBoxedPix,
    );
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mont_xcorr_edge(
    mut lowerIn: *mut ::core::ffi::c_float,
    mut upperIn: *mut ::core::ffi::c_float,
    mut nxyBox: *mut ::core::ffi::c_int,
    mut nxyPiece: *mut ::core::ffi::c_int,
    mut nxyOverlap: *mut ::core::ffi::c_int,
    mut nxSmooth: ::core::ffi::c_int,
    mut nySmooth: ::core::ffi::c_int,
    mut nxPad: ::core::ffi::c_int,
    mut nyPad: ::core::ffi::c_int,
    mut lowerPad: *mut ::core::ffi::c_float,
    mut upperPad: *mut ::core::ffi::c_float,
    mut lowerCopy: *mut ::core::ffi::c_float,
    mut numXcorrPeaks: ::core::ffi::c_int,
    mut legacy: ::core::ffi::c_int,
    mut ctf: *mut ::core::ffi::c_float,
    mut delta: ::core::ffi::c_float,
    mut inExtra: *mut ::core::ffi::c_int,
    mut nbin: ::core::ffi::c_int,
    mut ixy: ::core::ffi::c_int,
    mut maxLongShift: ::core::ffi::c_int,
    mut weightCCC: ::core::ffi::c_int,
    mut xDisplace: *mut ::core::ffi::c_float,
    mut yDisplace: *mut ::core::ffi::c_float,
    mut CCC: *mut ::core::ffi::c_float,
    mut twoDfft: Option<
        unsafe extern "C" fn(
            *mut ::core::ffi::c_float,
            *mut ::core::ffi::c_int,
            *mut ::core::ffi::c_int,
            *mut ::core::ffi::c_int,
        ) -> (),
    >,
    mut dumpEdge: Option<
        unsafe extern "C" fn(
            *mut ::core::ffi::c_float,
            *mut ::core::ffi::c_int,
            *mut ::core::ffi::c_int,
            *mut ::core::ffi::c_int,
            *mut ::core::ffi::c_int,
            *mut ::core::ffi::c_int,
        ) -> (),
    >,
    mut debugStr: *mut ::core::ffi::c_char,
    mut debugLen: ::core::ffi::c_int,
    mut debugLevel: ::core::ffi::c_int,
) {
    let mut ind: ::core::ffi::c_int = 0;
    let mut i: ::core::ffi::c_int = 0;
    let mut nxTrim: ::core::ffi::c_int = 0;
    let mut nyTrim: ::core::ffi::c_int = 0;
    let mut numPixel: ::core::ffi::c_int = 0;
    let mut indPeak: ::core::ffi::c_int = 0;
    let mut indSecond: ::core::ffi::c_int = 0;
    let mut indThird: ::core::ffi::c_int = 0;
    let mut curDebugLen: ::core::ffi::c_int = 0 as ::core::ffi::c_int;
    let mut nxPadDim: ::core::ffi::c_int = nxPad + 2 as ::core::ffi::c_int;
    let mut arrayIn: *mut ::core::ffi::c_float = lowerIn;
    let mut arrayOut: *mut ::core::ffi::c_float = lowerPad;
    let mut xpeak: [::core::ffi::c_float; 100] = [0.; 100];
    let mut ypeak: [::core::ffi::c_float; 100] = [0.; 100];
    let mut peak: [::core::ffi::c_float; 100] = [0.; 100];
    let mut wgtOrderInds: [::core::ffi::c_int; 100] = [0; 100];
    let mut wgtPeaks: [::core::ffi::c_float; 100] = [0.; 100];
    let mut gaussPeakProbs: [::core::ffi::c_float; 100] = [0.; 100];
    let mut sumArray: [::core::ffi::c_double; 7] = [0.; 7];
    let mut grandSums: [::core::ffi::c_double; 7] = [0.; 7];
    let mut cccSecond: ::core::ffi::c_double = 0.;
    let mut cccThird: ::core::ffi::c_double = 0.;
    let mut xTemp: ::core::ffi::c_float = 0.;
    let mut yTemp: ::core::ffi::c_float = 0.;
    let mut newCCC: ::core::ffi::c_float = 0.;
    let mut zero: ::core::ffi::c_int = 0 as ::core::ffi::c_int;
    let mut one: ::core::ffi::c_int = 1 as ::core::ffi::c_int;
    let mut jxy: ::core::ffi::c_int = 0;
    let mut ixyP1: ::core::ffi::c_int = 0;
    let mut numInSum: ::core::ffi::c_int = 0;
    let mut weights: *mut ::core::ffi::c_float = ::core::ptr::null_mut::<::core::ffi::c_float>();
    let mut aWeights: *mut ::core::ffi::c_float = ::core::ptr::null_mut::<::core::ffi::c_float>();
    let mut bWeights: *mut ::core::ffi::c_float = ::core::ptr::null_mut::<::core::ffi::c_float>();
    let mut nxWgt: ::core::ffi::c_int = 0;
    let mut nyWgt: ::core::ffi::c_int = 0;
    let mut numSamp: ::core::ffi::c_int = 0;
    let mut binWgt: ::core::ffi::c_int = 2 as ::core::ffi::c_int;
    let mut wgtXoffset: ::core::ffi::c_int = 0;
    let mut wgtYoffset: ::core::ffi::c_int = 0;
    let mut wgtBox: ::core::ffi::c_int = 10 as ::core::ffi::c_int;
    let mut evalCCC: ::core::ffi::c_int = if numXcorrPeaks > 1 as ::core::ffi::c_int && legacy == 0
    {
        1 as ::core::ffi::c_int
    } else {
        0 as ::core::ffi::c_int
    };
    let mut ccc: ::core::ffi::c_double = 0.;
    let mut cccMax: ::core::ffi::c_double = 0.;
    let mut wgtCCC: ::core::ffi::c_double = 0.;
    let mut fracArea: ::core::ffi::c_double = 0.;
    let mut sigma: ::core::ffi::c_double = 0.;
    let mut gaussProb: ::core::ffi::c_double = 0.;
    let mut expectDist: [::core::ffi::c_double; 2] = [0.; 2];
    let mut delx: ::core::ffi::c_int = 0;
    let mut dely: ::core::ffi::c_int = 0;
    let mut xStart: ::core::ffi::c_int = 0;
    let mut xEnd: ::core::ffi::c_int = 0;
    let mut yStart: ::core::ffi::c_int = 0;
    let mut yEnd: ::core::ffi::c_int = 0;
    let mut fullPixel: ::core::ffi::c_int = 0;
    let mut wgtTrim: ::core::ffi::c_int = 0;
    let mut nyLocal: ::core::ffi::c_int = 0;
    let mut nxLocal: ::core::ffi::c_int = 0;
    let mut numLocalX: ::core::ffi::c_int = 0;
    let mut localXoverlap: ::core::ffi::c_int = 0;
    let mut numLocalY: ::core::ffi::c_int = 0;
    let mut localYoverlap: ::core::ffi::c_int = 0;
    let mut lyStart: ::core::ffi::c_int = 0;
    let mut lyEnd: ::core::ffi::c_int = 0;
    let mut lxStart: ::core::ffi::c_int = 0;
    let mut lxEnd: ::core::ffi::c_int = 0;
    let mut localX: ::core::ffi::c_int = 0;
    let mut localY: ::core::ffi::c_int = 0;
    let mut loc: ::core::ffi::c_int = 0;
    let mut indOrd: ::core::ffi::c_int = 0;
    let mut localXseq: [::core::ffi::c_int; 100] = [0; 100];
    let mut localYseq: [::core::ffi::c_int; 100] = [0; 100];
    let mut localAspect: ::core::ffi::c_float = 0.;
    let mut maxLocalAspect: ::core::ffi::c_float = 2.0f32;
    let mut maxWsum: ::core::ffi::c_float = 0.0f32;
    let mut distLimit: ::core::ffi::c_float = 0.;
    let mut expectedXpeak: ::core::ffi::c_float = 0.;
    let mut expectedYpeak: ::core::ffi::c_float = 0.;
    let mut wsumAtMax: ::core::ffi::c_float = 0.;
    let mut wsum: ::core::ffi::c_float = 0.;
    let mut delExtent: ::core::ffi::c_float = 0.;
    let mut wgtThresh: ::core::ffi::c_float = 0.;
    let mut fracDiffCrit: ::core::ffi::c_float = 0.95f32;
    let mut minWsumRatio: ::core::ffi::c_float = 0.33f32;
    let mut runnerUpThreshFac: ::core::ffi::c_float = 0.8f32;
    let mut longShiftToAdd: [::core::ffi::c_int; 2] =
        [0 as ::core::ffi::c_int, 0 as ::core::ffi::c_int];
    let mut extraFromExpected: [::core::ffi::c_int; 2] =
        [0 as ::core::ffi::c_int, 0 as ::core::ffi::c_int];
    let mut expectedLeft: [::core::ffi::c_float; 2] = [
        0.0f64 as ::core::ffi::c_float,
        0.0f64 as ::core::ffi::c_float,
    ];
    let mut numExtra: [::core::ffi::c_int; 2] = [0; 2];
    let mut edgeDisplace: ::core::ffi::c_float = if ixy != 0 { *yDisplace } else { *xDisplace };
    let mut longDisplace: ::core::ffi::c_float = if ixy != 0 { *xDisplace } else { *yDisplace };
    let mut overlapPow: ::core::ffi::c_double = 0.166667f64;
    static mut first: ::core::ffi::c_int = 1 as ::core::ffi::c_int;
    let mut wallStart: ::core::ffi::c_double = 0.;
    numExtra[0 as ::core::ffi::c_int as usize] = *inExtra.offset(0 as ::core::ffi::c_int as isize);
    numExtra[1 as ::core::ffi::c_int as usize] =
        if *inExtra.offset(1 as ::core::ffi::c_int as isize) >= 0 as ::core::ffi::c_int {
            *inExtra.offset(1 as ::core::ffi::c_int as isize)
        } else {
            -*inExtra.offset(1 as ::core::ffi::c_int as isize)
        };
    if weightCCC > 0 as ::core::ffi::c_int {
        extraFromExpected[ixy as usize] = floor(
            (if 0.0f64 > -edgeDisplace as ::core::ffi::c_double {
                0.0f64
            } else {
                -edgeDisplace as ::core::ffi::c_double
            }) + 0.5f64,
        ) as ::core::ffi::c_int;
        expectedLeft[ixy as usize] = (if 0.0f64 > edgeDisplace as ::core::ffi::c_double {
            0.0f64
        } else {
            edgeDisplace as ::core::ffi::c_double
        }) as ::core::ffi::c_float;
        longShiftToAdd[(1 as ::core::ffi::c_int - ixy) as usize] =
            floor(longDisplace as ::core::ffi::c_double + 0.5f64) as ::core::ffi::c_int;
        sigma = 2.0f64
            * (if sDistWeightHalfFall as ::core::ffi::c_double > 0.0f64 {
                sDistWeightHalfFall / nbin as ::core::ffi::c_float
            } else {
                (*nxyOverlap.offset(ixy as isize) / nbin + numExtra[ixy as usize])
                    as ::core::ffi::c_float
            }) as ::core::ffi::c_double
            / 2.355f64;
        distLimit = (2.03f64 * sigma) as ::core::ffi::c_float;
        expectedXpeak = expectedLeft[0 as ::core::ffi::c_int as usize]
            / nbin as ::core::ffi::c_float
            + numExtra[0 as ::core::ffi::c_int as usize] as ::core::ffi::c_float;
        expectedYpeak = expectedLeft[1 as ::core::ffi::c_int as usize]
            / nbin as ::core::ffi::c_float
            + *inExtra.offset(1 as ::core::ffi::c_int as isize) as ::core::ffi::c_float;
        if debugLevel > 1 as ::core::ffi::c_int && debugLen > curDebugLen + MONTXC_MAX_DEBUG_LINE {
            sprintf(
                debugStr.offset(curDebugLen as isize) as *mut ::core::ffi::c_char,
                b"sigma %.2f limit %.1f  expected %.1f %.1f\n\0" as *const u8
                    as *const ::core::ffi::c_char,
                sigma,
                distLimit as ::core::ffi::c_double,
                expectedXpeak as ::core::ffi::c_double,
                expectedYpeak as ::core::ffi::c_double,
            );
            curDebugLen = (curDebugLen as ::core::ffi::c_ulong)
                .wrapping_add(strlen(
                    debugStr.offset(curDebugLen as isize) as *mut ::core::ffi::c_char
                ) as ::core::ffi::c_ulong) as ::core::ffi::c_int
                as ::core::ffi::c_int;
        }
    }
    ixyP1 = ixy + 1 as ::core::ffi::c_int;
    i = 0 as ::core::ffi::c_int;
    while i < 4 as ::core::ffi::c_int {
        sLastRunnersUp[i as usize] = -1.0e30f64 as ::core::ffi::c_float;
        i += 1;
    }
    xStart = (nxPad - *nxyBox.offset(0 as ::core::ffi::c_int as isize)) / 2 as ::core::ffi::c_int;
    xEnd = nxPad - xStart;
    yStart = (nyPad - *nxyBox.offset(1 as ::core::ffi::c_int as isize)) / 2 as ::core::ffi::c_int;
    yEnd = nyPad - yStart;
    nxWgt = (xEnd - xStart + binWgt - 1 as ::core::ffi::c_int) / binWgt;
    xEnd = xStart + binWgt * nxWgt - 1 as ::core::ffi::c_int;
    nyWgt = (yEnd - yStart + binWgt - 1 as ::core::ffi::c_int) / binWgt;
    yEnd = yStart + binWgt * nyWgt - 1 as ::core::ffi::c_int;
    wgtTrim = if (5 as ::core::ffi::c_int)
        < (if nxWgt < nyWgt { nxWgt } else { nyWgt }) / 20 as ::core::ffi::c_int
    {
        5 as ::core::ffi::c_int
    } else {
        (if nxWgt < nyWgt { nxWgt } else { nyWgt }) / 20 as ::core::ffi::c_int
    };
    if !lowerCopy.is_null() {
        if numXcorrPeaks > 1 as ::core::ffi::c_int {
            aWeights = malloc(
                ((nxWgt * nyWgt) as size_t)
                    .wrapping_mul(::core::mem::size_of::<::core::ffi::c_float>() as size_t),
            ) as *mut ::core::ffi::c_float;
        }
        bWeights = malloc(
            ((nxWgt * nyWgt) as size_t)
                .wrapping_mul(::core::mem::size_of::<::core::ffi::c_float>() as size_t),
        ) as *mut ::core::ffi::c_float;
        if weightCCC != 0 && aWeights.is_null() || bWeights.is_null() {
            free(aWeights as *mut ::core::ffi::c_void);
            aWeights = ::core::ptr::null_mut::<::core::ffi::c_float>();
            free(bWeights as *mut ::core::ffi::c_void);
            bWeights = ::core::ptr::null_mut::<::core::ffi::c_float>();
        }
    }
    weights = aWeights;
    sLastTrimmedMaxSD = -1.0f64 as ::core::ffi::c_float;
    ind = 0 as ::core::ffi::c_int;
    while ind < 2 as ::core::ffi::c_int {
        if nxSmooth > *nxyBox.offset(0 as ::core::ffi::c_int as isize)
            && nySmooth > *nxyBox.offset(1 as ::core::ffi::c_int as isize)
        {
            crate::imod::libcfshr::taperpad::slice_smooth_out_pad(
                arrayIn as *mut ::core::ffi::c_void,
                SLICE_MODE_FLOAT,
                *nxyBox.offset(0 as ::core::ffi::c_int as isize),
                *nxyBox.offset(1 as ::core::ffi::c_int as isize),
                arrayOut,
                nxSmooth,
                nxSmooth,
                nySmooth,
            );
            crate::imod::libcfshr::taperpad::slice_taper_out_pad(
                arrayOut as *mut ::core::ffi::c_void,
                SLICE_MODE_FLOAT,
                nxSmooth,
                nySmooth,
                arrayOut,
                nxPadDim,
                nxPad,
                nyPad,
                0 as ::core::ffi::c_int,
                0.0f32,
            );
        } else {
            crate::imod::libcfshr::taperpad::slice_taper_out_pad(
                arrayIn as *mut ::core::ffi::c_void,
                SLICE_MODE_FLOAT,
                *nxyBox.offset(0 as ::core::ffi::c_int as isize),
                *nxyBox.offset(1 as ::core::ffi::c_int as isize),
                arrayOut,
                nxPadDim,
                nxPad,
                nyPad,
                0 as ::core::ffi::c_int,
                0.0f32,
            );
        }
        crate::imod::libcfshr::filtxcorr::xcorr_mean_zero(arrayOut, nxPadDim, nxPad, nyPad);
        if dumpEdge.is_some() {
            dumpEdge.expect("non-null function pointer")(
                arrayOut,
                &raw mut nxPadDim,
                &raw mut nxPad,
                &raw mut nyPad,
                &raw mut ixyP1,
                &raw mut zero,
            );
        }
        if !weights.is_null() {
            crate::imod::libcfshr::multibinstat::make_standard_dev_map(
                arrayOut,
                nxPadDim,
                xStart,
                xEnd,
                yStart,
                yEnd,
                binWgt,
                wgtBox,
                weights,
                lowerCopy,
                lowerCopy.offset((nxWgt * nyWgt) as isize),
                &raw mut wgtXoffset,
                &raw mut wgtYoffset,
            );
            crate::imod::libcfshr::samplemeansd::get_sample_of_array(
                weights as *mut ::core::ffi::c_void,
                2 as ::core::ffi::c_int,
                nxWgt,
                nyWgt,
                1.0f32,
                wgtTrim,
                wgtTrim,
                nxWgt - 2 as ::core::ffi::c_int * wgtTrim,
                nyWgt - 2 as ::core::ffi::c_int * wgtTrim,
                -1.0f64 as ::core::ffi::c_float,
                lowerCopy,
                if (10000 as ::core::ffi::c_int) < nxPad * nyPad {
                    10000 as ::core::ffi::c_int
                } else {
                    nxPad * nyPad
                },
                &raw mut numSamp,
            );
            if numSamp > 0 as ::core::ffi::c_int {
                if numSamp <= 20 as ::core::ffi::c_int {
                    sLastTrimmedMaxSD =
                        *lowerCopy.offset((numSamp - 1 as ::core::ffi::c_int) as isize);
                } else {
                    sLastTrimmedMaxSD = crate::imod::libcfshr::percentile::percentile_float(
                        (0.95f64 * numSamp as ::core::ffi::c_double) as ::core::ffi::c_int,
                        lowerCopy,
                        numSamp,
                    );
                }
            }
        }
        twoDfft.expect("non-null function pointer")(
            arrayOut,
            &raw mut nxPad,
            &raw mut nyPad,
            &raw mut zero,
        );
        if delta as ::core::ffi::c_double > 0.0f64 && (ind == 0 || evalCCC != 0) {
            crate::imod::libcfshr::filtxcorr::xcorr_filter_part(
                arrayOut, arrayOut, nxPad, nyPad, ctf, delta,
            );
        }
        arrayIn = upperIn;
        arrayOut = upperPad;
        weights = bWeights;
        ind += 1;
    }
    first = 0 as ::core::ffi::c_int;
    if delta as ::core::ffi::c_double > 0.0f64 && evalCCC != 0 {
        memcpy(
            lowerCopy as *mut ::core::ffi::c_void,
            lowerPad as *const ::core::ffi::c_void,
            ((nxPadDim * nyPad) as size_t)
                .wrapping_mul(::core::mem::size_of::<::core::ffi::c_float>() as size_t),
        );
    }
    crate::imod::libcfshr::filtxcorr::conjugate_product(lowerPad, upperPad, nxPad, nyPad);
    twoDfft.expect("non-null function pointer")(
        lowerPad,
        &raw mut nxPad,
        &raw mut nyPad,
        &raw mut one,
    );
    if weightCCC != 0 {
        crate::imod::libcfshr::filtxcorr::set_peak_find_limits(
            (expectedXpeak - distLimit) as ::core::ffi::c_int,
            (expectedXpeak + distLimit) as ::core::ffi::c_int,
            (expectedYpeak - distLimit) as ::core::ffi::c_int,
            (expectedYpeak + distLimit) as ::core::ffi::c_int,
            1 as ::core::ffi::c_int,
        );
    }
    crate::imod::libcfshr::filtxcorr::xcorr_peak_find(
        lowerPad,
        nxPadDim,
        nyPad,
        &raw mut xpeak as *mut ::core::ffi::c_float,
        &raw mut ypeak as *mut ::core::ffi::c_float,
        &raw mut peak as *mut ::core::ffi::c_float,
        if 16 as ::core::ffi::c_int > numXcorrPeaks {
            16 as ::core::ffi::c_int
        } else {
            numXcorrPeaks
        },
    );
    indThird = -(1 as ::core::ffi::c_int);
    indSecond = indThird;
    indPeak = indSecond;
    i = 0 as ::core::ffi::c_int;
    while i
        < (if 16 as ::core::ffi::c_int > numXcorrPeaks {
            16 as ::core::ffi::c_int
        } else {
            numXcorrPeaks
        })
    {
        if ixy == 0 as ::core::ffi::c_int
            && fabs(ypeak[i as usize] as ::core::ffi::c_double)
                > maxLongShift as ::core::ffi::c_double
            || ixy == 1 as ::core::ffi::c_int
                && fabs(xpeak[i as usize] as ::core::ffi::c_double)
                    > maxLongShift as ::core::ffi::c_double
        {
            peak[i as usize] = -1.0e30f64 as ::core::ffi::c_float;
            if debugLevel > 2 as ::core::ffi::c_int
                && debugLen > curDebugLen + MONTXC_MAX_DEBUG_LINE
            {
                sprintf(
                    debugStr.offset(curDebugLen as isize) as *mut ::core::ffi::c_char,
                    b"Eliminated peak %d at %.1f %.1f\n\0" as *const u8
                        as *const ::core::ffi::c_char,
                    i,
                    xpeak[i as usize] as ::core::ffi::c_double,
                    ypeak[i as usize] as ::core::ffi::c_double,
                );
                curDebugLen = (curDebugLen as ::core::ffi::c_ulong)
                    .wrapping_add(strlen(
                        debugStr.offset(curDebugLen as isize) as *mut ::core::ffi::c_char
                    ) as ::core::ffi::c_ulong) as ::core::ffi::c_int
                    as ::core::ffi::c_int;
            }
        } else if indPeak == -(1 as ::core::ffi::c_int)
            && peak[i as usize] as ::core::ffi::c_double > -1.0e29f64
        {
            indPeak = i;
        }
        i += 1;
    }
    if indPeak == -(1 as ::core::ffi::c_int) {
        indPeak = 0 as ::core::ffi::c_int;
        xpeak[0 as ::core::ffi::c_int as usize] =
            numExtra[0 as ::core::ffi::c_int as usize] as ::core::ffi::c_float;
        ypeak[0 as ::core::ffi::c_int as usize] =
            *inExtra.offset(1 as ::core::ffi::c_int as isize) as ::core::ffi::c_float;
    }
    *CCC = -1.5f32;
    if evalCCC != 0 {
        if delta == 0 as ::core::ffi::c_int as ::core::ffi::c_float {
            crate::imod::libcfshr::taperpad::slice_taper_out_pad(
                lowerIn as *mut ::core::ffi::c_void,
                SLICE_MODE_FLOAT,
                *nxyBox.offset(0 as ::core::ffi::c_int as isize),
                *nxyBox.offset(1 as ::core::ffi::c_int as isize),
                lowerCopy,
                nxPadDim,
                nxPad,
                nyPad,
                0 as ::core::ffi::c_int,
                0.0f32,
            );
            crate::imod::libcfshr::taperpad::slice_taper_out_pad(
                upperIn as *mut ::core::ffi::c_void,
                SLICE_MODE_FLOAT,
                *nxyBox.offset(0 as ::core::ffi::c_int as isize),
                *nxyBox.offset(1 as ::core::ffi::c_int as isize),
                upperPad,
                nxPadDim,
                nxPad,
                nyPad,
                0 as ::core::ffi::c_int,
                0.0f32,
            );
        } else {
            twoDfft.expect("non-null function pointer")(
                lowerCopy,
                &raw mut nxPad,
                &raw mut nyPad,
                &raw mut one,
            );
            twoDfft.expect("non-null function pointer")(
                upperPad,
                &raw mut nxPad,
                &raw mut nyPad,
                &raw mut one,
            );
            if dumpEdge.is_some() {
                dumpEdge.expect("non-null function pointer")(
                    lowerCopy,
                    &raw mut nxPadDim,
                    &raw mut nxPad,
                    &raw mut nyPad,
                    &raw mut ixyP1,
                    &raw mut zero,
                );
                dumpEdge.expect("non-null function pointer")(
                    upperPad,
                    &raw mut nxPadDim,
                    &raw mut nxPad,
                    &raw mut nyPad,
                    &raw mut ixyP1,
                    &raw mut zero,
                );
            }
        }
        cccThird = -1.5f64;
        cccSecond = cccThird;
        cccMax = cccSecond;
        wsumAtMax = 0.0f32;
        nxTrim = (if (4 as ::core::ffi::c_int)
            < *nxyBox.offset(0 as ::core::ffi::c_int as isize) / 8 as ::core::ffi::c_int
        {
            4 as ::core::ffi::c_int
        } else {
            *nxyBox.offset(0 as ::core::ffi::c_int as isize) / 8 as ::core::ffi::c_int
        }) + (nxPad - *nxyBox.offset(0 as ::core::ffi::c_int as isize))
            / 2 as ::core::ffi::c_int;
        nyTrim = (if (4 as ::core::ffi::c_int)
            < *nxyBox.offset(1 as ::core::ffi::c_int as isize) / 8 as ::core::ffi::c_int
        {
            4 as ::core::ffi::c_int
        } else {
            *nxyBox.offset(1 as ::core::ffi::c_int as isize) / 8 as ::core::ffi::c_int
        }) + (nyPad - *nxyBox.offset(1 as ::core::ffi::c_int as isize))
            / 2 as ::core::ffi::c_int;
        fullPixel =
            (nxPad - 2 as ::core::ffi::c_int * nxTrim) * (nyPad - 2 as ::core::ffi::c_int * nyTrim);
        localAspect = ((*nxyBox.offset((1 as ::core::ffi::c_int - ixy) as isize)
            as ::core::ffi::c_float
            / *nxyBox.offset(ixy as isize) as ::core::ffi::c_float)
            as ::core::ffi::c_double
            / 1.4f64) as ::core::ffi::c_float;
        localAspect = if localAspect < maxLocalAspect {
            localAspect
        } else {
            maxLocalAspect
        };
        if ixy > 0 as ::core::ffi::c_int {
            nyLocal = (nyPad - 2 as ::core::ffi::c_int * nyTrim) / 2 as ::core::ffi::c_int;
            nxLocal = (localAspect * nyLocal as ::core::ffi::c_float) as ::core::ffi::c_int;
        } else {
            nxLocal = (nxPad - 2 as ::core::ffi::c_int * nxTrim) / 2 as ::core::ffi::c_int;
            nyLocal = (localAspect * nxLocal as ::core::ffi::c_float) as ::core::ffi::c_int;
        }
        i = 0 as ::core::ffi::c_int;
        while i < numXcorrPeaks {
            wgtOrderInds[i as usize] = i;
            gaussPeakProbs[i as usize] = 1.0f32;
            i += 1;
        }
        if weightCCC != 0 {
            i = 0 as ::core::ffi::c_int;
            while i < numXcorrPeaks {
                expectDist[0 as ::core::ffi::c_int as usize] =
                    (xpeak[i as usize] - expectedXpeak) as ::core::ffi::c_double;
                expectDist[1 as ::core::ffi::c_int as usize] =
                    (ypeak[i as usize] - expectedYpeak) as ::core::ffi::c_double;
                gaussPeakProbs[i as usize] = exp(-0.5f64
                    * (pow(expectDist[0 as ::core::ffi::c_int as usize] / sigma, 2.0f64)
                        + pow(expectDist[1 as ::core::ffi::c_int as usize] / sigma, 2.0f64)))
                    as ::core::ffi::c_float;
                if debugLevel > 2 as ::core::ffi::c_int
                    && debugLen > curDebugLen + MONTXC_MAX_DEBUG_LINE
                {
                    sprintf(
                        debugStr.offset(curDebugLen as isize) as *mut ::core::ffi::c_char,
                        b"%d: expected dist %.1f %.1f  prob %.3f\n\0" as *const u8
                            as *const ::core::ffi::c_char,
                        i,
                        expectDist[0 as ::core::ffi::c_int as usize],
                        expectDist[1 as ::core::ffi::c_int as usize],
                        gaussPeakProbs[i as usize] as ::core::ffi::c_double,
                    );
                    curDebugLen = (curDebugLen as ::core::ffi::c_ulong)
                        .wrapping_add(strlen(
                            debugStr.offset(curDebugLen as isize) as *mut ::core::ffi::c_char
                        ) as ::core::ffi::c_ulong)
                        as ::core::ffi::c_int
                        as ::core::ffi::c_int;
                }
                wgtPeaks[i as usize] = -gaussPeakProbs[i as usize] * peak[i as usize];
                i += 1;
            }
            crate::imod::libcfshr::robuststat::rs_sort_indexed_floats(
                &raw mut wgtPeaks as *mut ::core::ffi::c_float,
                &raw mut wgtOrderInds as *mut ::core::ffi::c_int,
                numXcorrPeaks,
            );
        }
        wallStart = crate::imod::libcfshr::b3dutil::wall_time();
        indOrd = 0 as ::core::ffi::c_int;
        while indOrd < numXcorrPeaks {
            i = wgtOrderInds[indOrd as usize];
            if !(peak[i as usize] as ::core::ffi::c_double <= -1.0e29f64) {
                gaussProb = gaussPeakProbs[i as usize] as ::core::ffi::c_double;
                if !(i != 0 && gaussProb < 1.01f64 * cccMax) {
                    if !(ixy == 0 as ::core::ffi::c_int
                        && fabs(
                            (*nxyPiece.offset(0 as ::core::ffi::c_int as isize)
                                as ::core::ffi::c_float
                                + nbin as ::core::ffi::c_float
                                    * (xpeak[i as usize]
                                        - numExtra[0 as ::core::ffi::c_int as usize]
                                            as ::core::ffi::c_float)
                                - extraFromExpected[0 as ::core::ffi::c_int as usize]
                                    as ::core::ffi::c_float
                                - *nxyOverlap.offset(0 as ::core::ffi::c_int as isize)
                                    as ::core::ffi::c_float)
                                as ::core::ffi::c_double,
                        ) <= 3.0f64
                        || ixy == 1 as ::core::ffi::c_int
                            && fabs(
                                (*nxyPiece.offset(1 as ::core::ffi::c_int as isize)
                                    as ::core::ffi::c_float
                                    + nbin as ::core::ffi::c_float
                                        * (ypeak[i as usize]
                                            - *inExtra.offset(1 as ::core::ffi::c_int as isize)
                                                as ::core::ffi::c_float)
                                    - extraFromExpected[1 as ::core::ffi::c_int as usize]
                                        as ::core::ffi::c_float
                                    - *nxyOverlap.offset(1 as ::core::ffi::c_int as isize)
                                        as ::core::ffi::c_float)
                                    as ::core::ffi::c_double,
                            ) <= 3.0f64)
                    {
                        delx = floor(xpeak[i as usize] as ::core::ffi::c_double + 0.5f64)
                            as ::core::ffi::c_int;
                        xStart = if nxTrim > nxTrim + delx {
                            nxTrim
                        } else {
                            nxTrim + delx
                        };
                        xEnd = if nxPad - nxTrim < nxPad - nxTrim + delx {
                            nxPad - nxTrim
                        } else {
                            nxPad - nxTrim + delx
                        };
                        dely = floor(ypeak[i as usize] as ::core::ffi::c_double + 0.5f64)
                            as ::core::ffi::c_int;
                        yStart = if nyTrim > nyTrim + dely {
                            nyTrim
                        } else {
                            nyTrim + dely
                        };
                        yEnd = if nyPad - nyTrim < nyPad - nyTrim + dely {
                            nyPad - nyTrim
                        } else {
                            nyPad - nyTrim + dely
                        };
                        numPixel = (yEnd - yStart) * (xEnd - xStart);
                        fracArea =
                            numPixel as ::core::ffi::c_double / fullPixel as ::core::ffi::c_double;
                        if !(i != 0 && fracArea < 0.125f64) {
                            local_num_and_overlap(
                                xEnd - xStart,
                                nxLocal,
                                &raw mut numLocalX,
                                &raw mut localXoverlap,
                            );
                            local_num_and_overlap(
                                yEnd - yStart,
                                nyLocal,
                                &raw mut numLocalY,
                                &raw mut localYoverlap,
                            );
                            if ixy > 0 as ::core::ffi::c_int {
                                setup_local_sequence(
                                    numLocalX,
                                    numLocalY,
                                    &raw mut localXseq as *mut ::core::ffi::c_int,
                                    &raw mut localYseq as *mut ::core::ffi::c_int,
                                );
                            } else {
                                setup_local_sequence(
                                    numLocalY,
                                    numLocalX,
                                    &raw mut localYseq as *mut ::core::ffi::c_int,
                                    &raw mut localXseq as *mut ::core::ffi::c_int,
                                );
                            }
                            ind = 0 as ::core::ffi::c_int;
                            while ind < 7 as ::core::ffi::c_int {
                                grandSums[ind as usize] = 0.0f64;
                                ind += 1;
                            }
                            numInSum = 0 as ::core::ffi::c_int;
                            loc = 0 as ::core::ffi::c_int;
                            while loc < numLocalX * numLocalY {
                                localX = localXseq[loc as usize];
                                localY = localYseq[loc as usize];
                                if numLocalY == 1 as ::core::ffi::c_int {
                                    lyStart = yStart
                                        + (if 0 as ::core::ffi::c_int
                                            > (yEnd - yStart - nyLocal) / 2 as ::core::ffi::c_int
                                        {
                                            0 as ::core::ffi::c_int
                                        } else {
                                            (yEnd - yStart - nyLocal) / 2 as ::core::ffi::c_int
                                        });
                                    lyEnd = if yEnd < yStart + nyLocal {
                                        yEnd
                                    } else {
                                        yStart + nyLocal
                                    };
                                } else {
                                    lyStart = yStart + localY * (nyLocal - localYoverlap);
                                    lyEnd = if yEnd < lyStart + nyLocal {
                                        yEnd
                                    } else {
                                        lyStart + nyLocal
                                    };
                                }
                                if numLocalX == 1 as ::core::ffi::c_int {
                                    lxStart = xStart
                                        + (if 0 as ::core::ffi::c_int
                                            > (xEnd - xStart - nxLocal) / 2 as ::core::ffi::c_int
                                        {
                                            0 as ::core::ffi::c_int
                                        } else {
                                            (xEnd - xStart - nxLocal) / 2 as ::core::ffi::c_int
                                        });
                                    lxEnd = if xEnd < xStart + nxLocal {
                                        xEnd
                                    } else {
                                        xStart + nxLocal
                                    };
                                } else {
                                    lxStart = xStart + localX * (nxLocal - localXoverlap);
                                    lxEnd = if xEnd < lxStart + nxLocal - 1 as ::core::ffi::c_int {
                                        xEnd
                                    } else {
                                        lxStart + nxLocal - 1 as ::core::ffi::c_int
                                    };
                                }
                                xTemp = xpeak[i as usize];
                                yTemp = ypeak[i as usize];
                                mont_xc_find_best_corr(
                                    lowerCopy,
                                    upperPad,
                                    nxPadDim,
                                    nxPad,
                                    nyPad,
                                    nxTrim,
                                    nyTrim,
                                    lxStart,
                                    lxEnd - 1 as ::core::ffi::c_int,
                                    lyStart,
                                    lyEnd - 1 as ::core::ffi::c_int,
                                    &raw mut xTemp,
                                    &raw mut yTemp,
                                    &raw mut newCCC,
                                    10.0f32,
                                    aWeights,
                                    bWeights,
                                    nxWgt,
                                    binWgt,
                                    wgtXoffset,
                                    wgtYoffset,
                                    (if 0.02f64 < cccMax / 10.0f64 {
                                        0.02f64
                                    } else {
                                        cccMax / 10.0f64
                                    }) as ::core::ffi::c_float,
                                    &raw mut sumArray as *mut ::core::ffi::c_double,
                                );
                                if newCCC != 0 as ::core::ffi::c_int as ::core::ffi::c_float {
                                    numInSum += 1;
                                    ind = 0 as ::core::ffi::c_int;
                                    while ind < 6 as ::core::ffi::c_int {
                                        grandSums[ind as usize] += sumArray[ind as usize];
                                        ind += 1;
                                    }
                                    wsum = grandSums[5 as ::core::ffi::c_int as usize]
                                        as ::core::ffi::c_float;
                                    ccc = crate::imod::libcfshr::filtxcorr::weighted_corr_from_sums(
                                        grandSums[0 as ::core::ffi::c_int as usize],
                                        grandSums[1 as ::core::ffi::c_int as usize],
                                        grandSums[2 as ::core::ffi::c_int as usize],
                                        grandSums[3 as ::core::ffi::c_int as usize],
                                        grandSums[4 as ::core::ffi::c_int as usize],
                                        wsum as ::core::ffi::c_double,
                                        ::core::ptr::null_mut::<::core::ffi::c_double>(),
                                        b"\0" as *const u8 as *const ::core::ffi::c_char,
                                    );
                                    if indOrd > 0 as ::core::ffi::c_int
                                        && (wsum as ::core::ffi::c_double
                                            > maxWsum as ::core::ffi::c_double / 10.0f64
                                            && gaussProb * ccc < 0.5f64 * cccMax
                                            || wsum as ::core::ffi::c_double
                                                > maxWsum as ::core::ffi::c_double / 5.0f64
                                                && gaussProb * ccc < 0.75f64 * cccMax
                                            || wsum as ::core::ffi::c_double
                                                > maxWsum as ::core::ffi::c_double / 3.0f64
                                                && gaussProb * ccc < 0.85f64 * cccMax
                                            || wsum as ::core::ffi::c_double
                                                > maxWsum as ::core::ffi::c_double / 2.0f64
                                                && gaussProb * ccc < 0.9f64 * cccMax)
                                    {
                                        break;
                                    }
                                }
                                loc += 1;
                            }
                            maxWsum = if maxWsum > wsum { maxWsum } else { wsum };
                            if wsum > minWsumRatio * wsumAtMax {
                                if gaussProb * ccc > cccMax {
                                    if cccMax > -(1 as ::core::ffi::c_int) as ::core::ffi::c_double
                                    {
                                        indThird = indSecond;
                                        indSecond = indPeak;
                                        cccThird = cccSecond;
                                        cccSecond = cccMax;
                                    }
                                    cccMax = gaussProb * ccc;
                                    indPeak = i;
                                    wsumAtMax = wsum;
                                } else if gaussProb * ccc > cccSecond {
                                    cccThird = cccSecond;
                                    cccSecond = gaussProb * ccc;
                                    indThird = indSecond;
                                    indSecond = i;
                                } else if gaussProb * ccc > cccThird {
                                    cccThird = gaussProb * ccc;
                                    indThird = i;
                                }
                            }
                            if debugLevel > 1 as ::core::ffi::c_int
                                && debugLen > curDebugLen + MONTXC_MAX_DEBUG_LINE
                            {
                                sprintf(
                                    debugStr.offset(curDebugLen as isize)
                                        as *mut ::core::ffi::c_char,
                                    b"%2d: at %7.1f %7.1f peak %14.7e  frac %.3f CCC %.5f%s wgt %.5f%s %g\n\0"
                                        as *const u8 as *const ::core::ffi::c_char,
                                    i,
                                    xpeak[i as usize] as ::core::ffi::c_double,
                                    ypeak[i as usize] as ::core::ffi::c_double,
                                    peak[i as usize] as ::core::ffi::c_double,
                                    fracArea,
                                    ccc,
                                    if weightCCC == 0 && indPeak == i {
                                        b"*\0" as *const u8 as *const ::core::ffi::c_char
                                    } else {
                                        b" \0" as *const u8 as *const ::core::ffi::c_char
                                    },
                                    gaussProb * ccc,
                                    if weightCCC != 0 && indPeak == i {
                                        b"*\0" as *const u8 as *const ::core::ffi::c_char
                                    } else {
                                        b" \0" as *const u8 as *const ::core::ffi::c_char
                                    },
                                    wsum as ::core::ffi::c_double,
                                );
                                curDebugLen = (curDebugLen as ::core::ffi::c_ulong).wrapping_add(
                                    strlen(debugStr.offset(curDebugLen as isize)
                                        as *mut ::core::ffi::c_char)
                                        as ::core::ffi::c_ulong,
                                )
                                    as ::core::ffi::c_int
                                    as ::core::ffi::c_int;
                            }
                        }
                    }
                }
            }
            indOrd += 1;
        }
        i = indPeak;
        if debugLevel == 1 as ::core::ffi::c_int && debugLen > curDebugLen + MONTXC_MAX_DEBUG_LINE {
            sprintf(
                debugStr.offset(curDebugLen as isize) as *mut ::core::ffi::c_char,
                b"Peak %d at %7.1f %7.1f  peak = %14.7g  CCC = %.5f\n\0" as *const u8
                    as *const ::core::ffi::c_char,
                i,
                xpeak[i as usize] as ::core::ffi::c_double,
                ypeak[i as usize] as ::core::ffi::c_double,
                peak[i as usize] as ::core::ffi::c_double,
                cccMax,
            );
            curDebugLen = (curDebugLen as ::core::ffi::c_ulong)
                .wrapping_add(strlen(
                    debugStr.offset(curDebugLen as isize) as *mut ::core::ffi::c_char
                ) as ::core::ffi::c_ulong) as ::core::ffi::c_int
                as ::core::ffi::c_int;
        }
        *CCC = cccMax as ::core::ffi::c_float;
        if indSecond >= 0 as ::core::ffi::c_int
            && cccSecond > runnerUpThreshFac as ::core::ffi::c_double * cccMax
        {
            sLastRunnersUp[0 as ::core::ffi::c_int as usize] = nbin as ::core::ffi::c_float
                * (xpeak[indSecond as usize]
                    - numExtra[0 as ::core::ffi::c_int as usize] as ::core::ffi::c_float)
                + longShiftToAdd[0 as ::core::ffi::c_int as usize] as ::core::ffi::c_float
                - extraFromExpected[0 as ::core::ffi::c_int as usize] as ::core::ffi::c_float;
            sLastRunnersUp[1 as ::core::ffi::c_int as usize] = nbin as ::core::ffi::c_float
                * (ypeak[indSecond as usize]
                    - *inExtra.offset(1 as ::core::ffi::c_int as isize) as ::core::ffi::c_float)
                + longShiftToAdd[1 as ::core::ffi::c_int as usize] as ::core::ffi::c_float
                - extraFromExpected[1 as ::core::ffi::c_int as usize] as ::core::ffi::c_float;
        }
        if indThird >= 0 as ::core::ffi::c_int
            && cccThird > runnerUpThreshFac as ::core::ffi::c_double * cccMax
        {
            sLastRunnersUp[2 as ::core::ffi::c_int as usize] = nbin as ::core::ffi::c_float
                * (xpeak[indThird as usize]
                    - numExtra[0 as ::core::ffi::c_int as usize] as ::core::ffi::c_float)
                + longShiftToAdd[0 as ::core::ffi::c_int as usize] as ::core::ffi::c_float
                - extraFromExpected[0 as ::core::ffi::c_int as usize] as ::core::ffi::c_float;
            sLastRunnersUp[3 as ::core::ffi::c_int as usize] = nbin as ::core::ffi::c_float
                * (ypeak[indThird as usize]
                    - *inExtra.offset(1 as ::core::ffi::c_int as isize) as ::core::ffi::c_float)
                + longShiftToAdd[1 as ::core::ffi::c_int as usize] as ::core::ffi::c_float
                - extraFromExpected[1 as ::core::ffi::c_int as usize] as ::core::ffi::c_float;
        }
    }
    if dumpEdge.is_some() {
        dumpEdge.expect("non-null function pointer")(
            lowerPad,
            &raw mut nxPadDim,
            &raw mut nxPad,
            &raw mut nyPad,
            &raw mut ixyP1,
            &raw mut one,
        );
    }
    *xDisplace = nbin as ::core::ffi::c_float
        * (xpeak[indPeak as usize]
            - numExtra[0 as ::core::ffi::c_int as usize] as ::core::ffi::c_float)
        + longShiftToAdd[0 as ::core::ffi::c_int as usize] as ::core::ffi::c_float
        - extraFromExpected[0 as ::core::ffi::c_int as usize] as ::core::ffi::c_float;
    *yDisplace = nbin as ::core::ffi::c_float
        * (ypeak[indPeak as usize]
            - *inExtra.offset(1 as ::core::ffi::c_int as isize) as ::core::ffi::c_float)
        + longShiftToAdd[1 as ::core::ffi::c_int as usize] as ::core::ffi::c_float
        - extraFromExpected[1 as ::core::ffi::c_int as usize] as ::core::ffi::c_float;
    if debugLevel != 0 && debugLen > curDebugLen + MONTXC_MAX_DEBUG_LINE {
        sprintf(
            debugStr.offset(curDebugLen as isize) as *mut ::core::ffi::c_char,
            b"Peak at %8.2f %8.2f  Displacement %8.2f %8.2f\n\0" as *const u8
                as *const ::core::ffi::c_char,
            xpeak[indPeak as usize] as ::core::ffi::c_double,
            ypeak[indPeak as usize] as ::core::ffi::c_double,
            *xDisplace as ::core::ffi::c_double,
            *yDisplace as ::core::ffi::c_double,
        );
    }
    free(aWeights as *mut ::core::ffi::c_void);
    aWeights = ::core::ptr::null_mut::<::core::ffi::c_float>();
    free(bWeights as *mut ::core::ffi::c_void);
    bWeights = ::core::ptr::null_mut::<::core::ffi::c_float>();
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn montxcorredge_(
    mut lowerIn: *mut ::core::ffi::c_float,
    mut upperIn: *mut ::core::ffi::c_float,
    mut nxyBox: *mut ::core::ffi::c_int,
    mut nxyPiece: *mut ::core::ffi::c_int,
    mut nxyOverlap: *mut ::core::ffi::c_int,
    mut nxSmooth: *mut ::core::ffi::c_int,
    mut nySmooth: *mut ::core::ffi::c_int,
    mut nxPad: *mut ::core::ffi::c_int,
    mut nyPad: *mut ::core::ffi::c_int,
    mut lowerPad: *mut ::core::ffi::c_float,
    mut upperPad: *mut ::core::ffi::c_float,
    mut lowerCopy: *mut ::core::ffi::c_float,
    mut numXcorrPeaks: *mut ::core::ffi::c_int,
    mut legacy: *mut ::core::ffi::c_int,
    mut ctf: *mut ::core::ffi::c_float,
    mut delta: *mut ::core::ffi::c_float,
    mut numExtra: *mut ::core::ffi::c_int,
    mut nbin: *mut ::core::ffi::c_int,
    mut ixy: *mut ::core::ffi::c_int,
    mut maxLongShift: *mut ::core::ffi::c_int,
    mut weightCCC: *mut ::core::ffi::c_int,
    mut xDisplace: *mut ::core::ffi::c_float,
    mut yDisplace: *mut ::core::ffi::c_float,
    mut CCC: *mut ::core::ffi::c_float,
    mut twoDfft: Option<
        unsafe extern "C" fn(
            *mut ::core::ffi::c_float,
            *mut ::core::ffi::c_int,
            *mut ::core::ffi::c_int,
            *mut ::core::ffi::c_int,
        ) -> (),
    >,
    mut dumpEdge: Option<
        unsafe extern "C" fn(
            *mut ::core::ffi::c_float,
            *mut ::core::ffi::c_int,
            *mut ::core::ffi::c_int,
            *mut ::core::ffi::c_int,
            *mut ::core::ffi::c_int,
            *mut ::core::ffi::c_int,
        ) -> (),
    >,
    mut debugLevel: *mut ::core::ffi::c_int,
) {
    let mut debugLen: ::core::ffi::c_int = MONTXC_MAX_PEAKS * MONTXC_MAX_DEBUG_LINE;
    let mut debugStr: [::core::ffi::c_char; 9000] = [0; 9000];
    let mut curDebug: *mut ::core::ffi::c_char = (&raw mut debugStr as *mut ::core::ffi::c_char)
        .offset(0 as ::core::ffi::c_int as isize)
        as *mut ::core::ffi::c_char;
    let mut lineEnd: *mut ::core::ffi::c_char = ::core::ptr::null_mut::<::core::ffi::c_char>();
    mont_xcorr_edge(
        lowerIn,
        upperIn,
        nxyBox,
        nxyPiece,
        nxyOverlap,
        *nxSmooth,
        *nySmooth,
        *nxPad,
        *nyPad,
        lowerPad,
        upperPad,
        lowerCopy,
        *numXcorrPeaks,
        *legacy,
        ctf,
        *delta,
        numExtra,
        *nbin,
        *ixy - 1 as ::core::ffi::c_int,
        *maxLongShift,
        *weightCCC,
        xDisplace,
        yDisplace,
        CCC,
        twoDfft,
        dumpEdge,
        &raw mut debugStr as *mut ::core::ffi::c_char,
        debugLen,
        *debugLevel,
    );
    if *debugLevel != 0 {
        loop {
            lineEnd = strchr(curDebug, '\n' as i32);
            if lineEnd.is_null() {
                break;
            }
            *lineEnd = 0 as ::core::ffi::c_char;
            printf(
                b"%s\n\0" as *const u8 as *const ::core::ffi::c_char,
                curDebug,
            );
            curDebug = lineEnd.offset(1 as ::core::ffi::c_int as isize);
        }
        fflush(stdout);
    }
}
unsafe extern "C" fn local_num_and_overlap(
    mut extent: ::core::ffi::c_int,
    mut nxLocal: ::core::ffi::c_int,
    mut numLocalXp: *mut ::core::ffi::c_int,
    mut nxOverlap: *mut ::core::ffi::c_int,
) {
    let mut targetOverlap: ::core::ffi::c_float = 0.35f32;
    let mut minOverlap: ::core::ffi::c_float = 0.2f32;
    let mut maxOverlap: ::core::ffi::c_float = 0.5f32;
    let mut numLocalX: ::core::ffi::c_int = 0;
    numLocalX = floor(
        (extent - nxLocal) as ::core::ffi::c_double
            / (nxLocal as ::core::ffi::c_double
                * (1.0f64 - targetOverlap as ::core::ffi::c_double))
            + 0.5f64,
    ) as ::core::ffi::c_int
        + 1 as ::core::ffi::c_int;
    numLocalX = if numLocalX > 1 as ::core::ffi::c_int {
        numLocalX
    } else {
        1 as ::core::ffi::c_int
    };
    while numLocalX > 1 as ::core::ffi::c_int
        && (nxLocal as ::core::ffi::c_double
            - (extent - nxLocal) as ::core::ffi::c_double
                / (numLocalX as ::core::ffi::c_double - 1.0f64))
            / nxLocal as ::core::ffi::c_double
            > maxOverlap as ::core::ffi::c_double
    {
        numLocalX -= 1;
    }
    while numLocalX > 1 as ::core::ffi::c_int
        && ((nxLocal as ::core::ffi::c_double
            - (extent - nxLocal) as ::core::ffi::c_double
                / (numLocalX as ::core::ffi::c_double - 1.0f64))
            / nxLocal as ::core::ffi::c_double)
            < minOverlap as ::core::ffi::c_double
    {
        numLocalX += 1;
    }
    *nxOverlap = nxLocal
        - (extent - nxLocal)
            / (if 1 as ::core::ffi::c_int > numLocalX - 1 as ::core::ffi::c_int {
                1 as ::core::ffi::c_int
            } else {
                numLocalX - 1 as ::core::ffi::c_int
            });
    *numLocalXp = numLocalX;
}
unsafe extern "C" fn setup_local_sequence(
    mut numLocalX: ::core::ffi::c_int,
    mut numLocalY: ::core::ffi::c_int,
    mut localXseq: *mut ::core::ffi::c_int,
    mut localYseq: *mut ::core::ffi::c_int,
) {
    let mut numLocalSeq: ::core::ffi::c_int = 0 as ::core::ffi::c_int;
    let mut ind: ::core::ffi::c_int = 0;
    let mut localX: ::core::ffi::c_int = 0;
    let mut localY: ::core::ffi::c_int = 0;
    let mut dir: ::core::ffi::c_int = -(1 as ::core::ffi::c_int);
    ind = 0 as ::core::ffi::c_int;
    while ind < numLocalX + 2 as ::core::ffi::c_int {
        if ind != 0 {
            localX = numLocalX / 2 as ::core::ffi::c_int
                + dir * ((ind + 1 as ::core::ffi::c_int) / 2 as ::core::ffi::c_int);
            dir = -dir;
        } else {
            localX = numLocalX / 2 as ::core::ffi::c_int;
        }
        if localX >= 0 as ::core::ffi::c_int && localX < numLocalX {
            localY = 0 as ::core::ffi::c_int;
            while localY < numLocalY {
                *localXseq.offset(numLocalSeq as isize) = localX;
                let fresh0 = numLocalSeq;
                numLocalSeq = numLocalSeq + 1;
                *localYseq.offset(fresh0 as isize) = localY;
                localY += 1;
            }
        }
        ind += 1;
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn montxcorrgetmaxes_(
    mut maxPeak: *mut ::core::ffi::c_int,
    mut maxLines: *mut ::core::ffi::c_int,
) {
    *maxPeak = MONTXC_MAX_PEAKS;
    *maxLines = MONTXC_MAX_DEBUG_LINE;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mont_xc_set_dist_weight_half_fall(mut inVal: *mut ::core::ffi::c_float) {
    sDistWeightHalfFall = *inVal;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mont_xc_get_last_trimmed_max_sd() -> ::core::ffi::c_double {
    return sLastTrimmedMaxSD as ::core::ffi::c_double;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mont_xc_get_last_runners_up(
    mut disps: *mut ::core::ffi::c_float,
    mut maxPairs: ::core::ffi::c_int,
) {
    let mut i: ::core::ffi::c_int = 0;
    i = 0 as ::core::ffi::c_int;
    while i < 2 as ::core::ffi::c_int
        * (if (2 as ::core::ffi::c_int) < maxPairs {
            2 as ::core::ffi::c_int
        } else {
            maxPairs
        })
    {
        *disps.offset(i as isize) = sLastRunnersUp[i as usize];
        i += 1;
    }
    i = 2 as ::core::ffi::c_int * MAX_RUNNERS_UP;
    while i < 2 as ::core::ffi::c_int * maxPairs {
        *disps.offset(i as isize) = -1.0e30f64 as ::core::ffi::c_float;
        i += 1;
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn montxcgetlastrunnersup_(
    mut disps: *mut ::core::ffi::c_float,
    mut maxPairs: *mut ::core::ffi::c_int,
) {
    mont_xc_get_last_runners_up(disps, *maxPairs);
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn row_of_three_corrs(
    mut array: *mut ::core::ffi::c_float,
    mut brray: *mut ::core::ffi::c_float,
    mut nxDim: ::core::ffi::c_int,
    mut ix0: ::core::ffi::c_int,
    mut ix1: ::core::ffi::c_int,
    mut iy0: ::core::ffi::c_int,
    mut iy1: ::core::ffi::c_int,
    mut delX: ::core::ffi::c_int,
    mut delY: ::core::ffi::c_int,
    mut aWeights: *mut ::core::ffi::c_float,
    mut bWeights: *mut ::core::ffi::c_float,
    mut nxWgt: ::core::ffi::c_int,
    mut binning: ::core::ffi::c_int,
    mut wgtXoffset: ::core::ffi::c_int,
    mut wgtYoffset: ::core::ffi::c_int,
    mut corr1: *mut ::core::ffi::c_float,
    mut corr2: *mut ::core::ffi::c_float,
    mut corr3: *mut ::core::ffi::c_float,
    mut sumArr1: *mut ::core::ffi::c_double,
    mut sumArr2: *mut ::core::ffi::c_double,
    mut sumArr3: *mut ::core::ffi::c_double,
) {
    let mut ix: ::core::ffi::c_int = 0;
    let mut iy: ::core::ffi::c_int = 0;
    let mut aBase: ::core::ffi::c_int = 0;
    let mut nsum: ::core::ffi::c_int = 0;
    let mut end: ::core::ffi::c_int = 0;
    let mut bBase: ::core::ffi::c_int = 0;
    let mut aWgtBase: ::core::ffi::c_int = 0;
    let mut bWgtBase: ::core::ffi::c_int = 0;
    let mut abSum1: ::core::ffi::c_double = 0.;
    let mut abSum2: ::core::ffi::c_double = 0.;
    let mut abSum3: ::core::ffi::c_double = 0.;
    let mut aSumSq1: ::core::ffi::c_double = 0.;
    let mut aSumSq2: ::core::ffi::c_double = 0.;
    let mut aSumSq3: ::core::ffi::c_double = 0.;
    let mut aSum1: ::core::ffi::c_double = 0.;
    let mut aSum2: ::core::ffi::c_double = 0.;
    let mut aSum3: ::core::ffi::c_double = 0.;
    let mut bSumSq1: ::core::ffi::c_double = 0.;
    let mut bSumSq2: ::core::ffi::c_double = 0.;
    let mut bSumSq3: ::core::ffi::c_double = 0.;
    let mut bSumSq: ::core::ffi::c_double = 0.;
    let mut denom: ::core::ffi::c_double = 0.;
    let mut amean: ::core::ffi::c_double = 0.;
    let mut bmean: ::core::ffi::c_double = 0.;
    let mut bSum1: ::core::ffi::c_double = 0.;
    let mut bSum2: ::core::ffi::c_double = 0.;
    let mut bSum3: ::core::ffi::c_double = 0.;
    let mut bSum: ::core::ffi::c_double = 0.;
    let mut wSum2: ::core::ffi::c_double = 0.;
    let mut wSum1: ::core::ffi::c_double = 0.;
    let mut wSum3: ::core::ffi::c_double = 0.;
    let mut abTmp1: ::core::ffi::c_double = 0.;
    let mut abTmp2: ::core::ffi::c_double = 0.;
    let mut abTmp3: ::core::ffi::c_double = 0.;
    let mut aTmp1: ::core::ffi::c_double = 0.;
    let mut aTmpSq1: ::core::ffi::c_double = 0.;
    let mut aTmp2: ::core::ffi::c_double = 0.;
    let mut aTmpSq2: ::core::ffi::c_double = 0.;
    let mut aTmp3: ::core::ffi::c_double = 0.;
    let mut aTmpSq3: ::core::ffi::c_double = 0.;
    let mut bTmp1: ::core::ffi::c_double = 0.;
    let mut bTmp2: ::core::ffi::c_double = 0.;
    let mut bTmp3: ::core::ffi::c_double = 0.;
    let mut bTmpSq1: ::core::ffi::c_double = 0.;
    let mut bTmpSq2: ::core::ffi::c_double = 0.;
    let mut bTmpSq3: ::core::ffi::c_double = 0.;
    let mut wTmp2: ::core::ffi::c_double = 0.;
    let mut wTmp1: ::core::ffi::c_double = 0.;
    let mut wTmp3: ::core::ffi::c_double = 0.;
    let mut aval: ::core::ffi::c_float = 0.;
    let mut bval1: ::core::ffi::c_float = 0.;
    let mut bval2: ::core::ffi::c_float = 0.;
    let mut bval3: ::core::ffi::c_float = 0.;
    let mut bval: ::core::ffi::c_float = 0.;
    let mut wgt: ::core::ffi::c_float = 0.;
    let mut wgt1: ::core::ffi::c_float = 0.;
    let mut wgt3: ::core::ffi::c_float = 0.;
    let mut awgt: ::core::ffi::c_float = 0.;
    let mut numThreads: ::core::ffi::c_int = 0;
    let mut maxThreads: ::core::ffi::c_int = 8 as ::core::ffi::c_int;
    numThreads = (floor(
        sqrt((ix1 - ix0) as ::core::ffi::c_double * (iy1 - iy0) as ::core::ffi::c_double) / 80.0f64
            + 0.5f64,
    ) as ::core::ffi::c_int as ::core::ffi::c_double
        * (if !aWeights.is_null() { 2.0f64 } else { 1.0f64 }))
        as ::core::ffi::c_int;
    numThreads = if 1 as ::core::ffi::c_int
        > (if maxThreads < numThreads {
            maxThreads
        } else {
            numThreads
        }) {
        1 as ::core::ffi::c_int
    } else if maxThreads < numThreads {
        maxThreads
    } else {
        numThreads
    };
    numThreads = crate::imod::libcfshr::b3dutil::num_omp_threads(numThreads);
    binning = if 1 as ::core::ffi::c_int > binning {
        1 as ::core::ffi::c_int
    } else {
        binning
    };
    aSumSq3 = 0.0f64;
    aSum3 = aSumSq3;
    aSumSq2 = aSum3;
    aSum1 = aSumSq2;
    aSumSq1 = aSum1;
    aSum2 = aSumSq1;
    abSum3 = aSum2;
    abSum2 = abSum3;
    abSum1 = abSum2;
    bSumSq3 = 0.0f64;
    bSumSq2 = bSumSq3;
    bSumSq1 = bSumSq2;
    wSum3 = 0.0f64;
    wSum2 = wSum3;
    wSum1 = wSum2;
    bSum3 = wSum1;
    bSum2 = bSum3;
    bSum1 = bSum2;
    iy = iy0;
    while iy <= iy1 {
        aBase = iy * nxDim;
        bBase = (iy - delY) * nxDim - delX;
        aWgtBase = (iy / binning + wgtYoffset) * nxWgt + wgtXoffset;
        bWgtBase = ((iy - delY) / binning + wgtYoffset) * nxWgt + wgtXoffset;
        bSum = 0.0f64;
        bSumSq = bSum;
        wTmp3 = 0.0f64;
        wTmp1 = wTmp3;
        wTmp2 = wTmp1;
        abTmp3 = wTmp2;
        abTmp2 = abTmp3;
        abTmp1 = abTmp2;
        aTmpSq3 = 0.0f64;
        aTmp3 = aTmpSq3;
        aTmpSq2 = aTmp3;
        aTmp2 = aTmpSq2;
        aTmpSq1 = aTmp2;
        aTmp1 = aTmpSq1;
        bTmpSq3 = 0.0f64;
        bTmpSq2 = bTmpSq3;
        bTmpSq1 = bTmpSq2;
        bTmp3 = bTmpSq1;
        bTmp2 = bTmp3;
        bTmp1 = bTmp2;
        if !aWeights.is_null() {
            ix = ix0;
            while ix <= ix1 {
                awgt = *aWeights.offset((ix / binning + aWgtBase) as isize);
                wgt1 = awgt
                    * *bWeights.offset(
                        ((ix + 1 as ::core::ffi::c_int - delX) / binning + bWgtBase) as isize,
                    );
                wgt = awgt * *bWeights.offset(((ix - delX) / binning + bWgtBase) as isize);
                wgt3 = awgt
                    * *bWeights.offset(
                        ((ix - 1 as ::core::ffi::c_int - delX) / binning + bWgtBase) as isize,
                    );
                wTmp1 += wgt1 as ::core::ffi::c_double;
                wTmp2 += wgt as ::core::ffi::c_double;
                wTmp3 += wgt3 as ::core::ffi::c_double;
                aval = *array.offset((ix + aBase) as isize);
                bval = *brray.offset((ix + bBase) as isize);
                bval1 = *brray.offset((ix + bBase + 1 as ::core::ffi::c_int) as isize);
                bval3 = *brray.offset((ix + bBase - 1 as ::core::ffi::c_int) as isize);
                aTmp1 += (aval * wgt1) as ::core::ffi::c_double;
                aTmp2 += (aval * wgt) as ::core::ffi::c_double;
                aTmp3 += (aval * wgt3) as ::core::ffi::c_double;
                bTmp1 += (bval1 * wgt1) as ::core::ffi::c_double;
                bTmp2 += (bval * wgt) as ::core::ffi::c_double;
                bTmp3 += (bval3 * wgt3) as ::core::ffi::c_double;
                aTmpSq1 += (aval * aval * wgt1) as ::core::ffi::c_double;
                aTmpSq2 += (aval * aval * wgt) as ::core::ffi::c_double;
                aTmpSq3 += (aval * aval * wgt3) as ::core::ffi::c_double;
                bTmpSq1 += (bval1 * bval1 * wgt1) as ::core::ffi::c_double;
                bTmpSq2 += (bval * bval * wgt) as ::core::ffi::c_double;
                bTmpSq3 += (bval3 * bval3 * wgt3) as ::core::ffi::c_double;
                abTmp1 += (aval * bval1 * wgt1) as ::core::ffi::c_double;
                abTmp2 += (aval * bval * wgt) as ::core::ffi::c_double;
                abTmp3 += (aval * bval3 * wgt3) as ::core::ffi::c_double;
                ix += 1;
            }
        } else {
            aval = *array.offset((ix0 + aBase) as isize);
            bval1 = *brray.offset((ix0 + bBase + 1 as ::core::ffi::c_int) as isize);
            bval2 = *brray.offset((ix0 + bBase) as isize);
            bval3 = *brray.offset((ix0 + bBase - 1 as ::core::ffi::c_int) as isize);
            aTmp2 += aval as ::core::ffi::c_double;
            aTmpSq2 += (aval * aval) as ::core::ffi::c_double;
            abTmp1 += (aval * bval1) as ::core::ffi::c_double;
            abTmp2 += (aval * bval2) as ::core::ffi::c_double;
            abTmp3 += (aval * bval3) as ::core::ffi::c_double;
            bTmp2 += bval2 as ::core::ffi::c_double;
            bTmp3 += (bval3 + bval2) as ::core::ffi::c_double;
            bTmpSq2 += (bval2 * bval2) as ::core::ffi::c_double;
            bTmpSq3 += (bval3 * bval3 + bval2 * bval2) as ::core::ffi::c_double;
            ix = ix0 + 1 as ::core::ffi::c_int;
            while ix < ix1 {
                aval = *array.offset((ix + aBase) as isize);
                bval = *brray.offset((ix + bBase) as isize);
                aTmp2 += aval as ::core::ffi::c_double;
                bSum += bval as ::core::ffi::c_double;
                aTmpSq2 += (aval * aval) as ::core::ffi::c_double;
                bSumSq += (bval * bval) as ::core::ffi::c_double;
                abTmp1 += (aval * *brray.offset((ix + bBase + 1 as ::core::ffi::c_int) as isize))
                    as ::core::ffi::c_double;
                abTmp2 += (aval * bval) as ::core::ffi::c_double;
                abTmp3 += (aval * *brray.offset((ix + bBase - 1 as ::core::ffi::c_int) as isize))
                    as ::core::ffi::c_double;
                ix += 1;
            }
            bTmp1 += bSum;
            bTmp2 += bSum;
            bTmp3 += bSum;
            bTmpSq1 += bSumSq;
            bTmpSq2 += bSumSq;
            bTmpSq3 += bSumSq;
            aval = *array.offset((ix1 + aBase) as isize);
            bval1 = *brray.offset((ix1 + bBase + 1 as ::core::ffi::c_int) as isize);
            bval2 = *brray.offset((ix1 + bBase) as isize);
            bval3 = *brray.offset((ix1 + bBase - 1 as ::core::ffi::c_int) as isize);
            aTmp2 += aval as ::core::ffi::c_double;
            aTmpSq2 += (aval * aval) as ::core::ffi::c_double;
            abTmp1 += (aval * bval1) as ::core::ffi::c_double;
            abTmp2 += (aval * bval2) as ::core::ffi::c_double;
            abTmp3 += (aval * bval3) as ::core::ffi::c_double;
            bTmp1 += (bval1 + bval2) as ::core::ffi::c_double;
            bTmp2 += bval2 as ::core::ffi::c_double;
            bTmpSq1 += (bval1 * bval1 + bval2 * bval2) as ::core::ffi::c_double;
            bTmpSq2 += (bval2 * bval2) as ::core::ffi::c_double;
            wTmp1 += (ix1 + 1 as ::core::ffi::c_int - ix0) as ::core::ffi::c_double;
            wTmp2 += (ix1 + 1 as ::core::ffi::c_int - ix0) as ::core::ffi::c_double;
            wTmp3 += (ix1 + 1 as ::core::ffi::c_int - ix0) as ::core::ffi::c_double;
            aTmp3 = aTmp2;
            aTmp1 = aTmp3;
            aTmpSq3 = aTmpSq2;
            aTmpSq1 = aTmpSq3;
        }
        wSum1 += wTmp1;
        wSum2 += wTmp2;
        wSum3 += wTmp3;
        aSum1 += aTmp1;
        aSum2 += aTmp2;
        aSum3 += aTmp3;
        aSumSq1 += aTmpSq1;
        aSumSq2 += aTmpSq2;
        aSumSq3 += aTmpSq3;
        abSum1 += abTmp1;
        abSum2 += abTmp2;
        abSum3 += abTmp3;
        bSum1 += bTmp1;
        bSum2 += bTmp2;
        bSum3 += bTmp3;
        bSumSq1 += bTmpSq1;
        bSumSq2 += bTmpSq2;
        bSumSq3 += bTmpSq3;
        iy += 1;
    }
    *corr1 = crate::imod::libcfshr::filtxcorr::weighted_corr_from_sums(
        aSum1,
        aSumSq1,
        bSum1,
        bSumSq1,
        abSum1,
        wSum1,
        sumArr1,
        b"\0" as *const u8 as *const ::core::ffi::c_char,
    ) as ::core::ffi::c_float;
    *corr2 = crate::imod::libcfshr::filtxcorr::weighted_corr_from_sums(
        aSum2,
        aSumSq2,
        bSum2,
        bSumSq2,
        abSum2,
        wSum2,
        sumArr2,
        b"\0" as *const u8 as *const ::core::ffi::c_char,
    ) as ::core::ffi::c_float;
    *corr3 = crate::imod::libcfshr::filtxcorr::weighted_corr_from_sums(
        aSum3,
        aSumSq3,
        bSum3,
        bSumSq3,
        abSum3,
        wSum3,
        sumArr3,
        b"\0" as *const u8 as *const ::core::ffi::c_char,
    ) as ::core::ffi::c_float;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn column_of_three_corrs(
    mut array: *mut ::core::ffi::c_float,
    mut brray: *mut ::core::ffi::c_float,
    mut nxDim: ::core::ffi::c_int,
    mut ix0: ::core::ffi::c_int,
    mut ix1: ::core::ffi::c_int,
    mut iy0: ::core::ffi::c_int,
    mut iy1: ::core::ffi::c_int,
    mut delX: ::core::ffi::c_int,
    mut delY: ::core::ffi::c_int,
    mut aWeights: *mut ::core::ffi::c_float,
    mut bWeights: *mut ::core::ffi::c_float,
    mut nxWgt: ::core::ffi::c_int,
    mut binning: ::core::ffi::c_int,
    mut wgtXoffset: ::core::ffi::c_int,
    mut wgtYoffset: ::core::ffi::c_int,
    mut corr1: *mut ::core::ffi::c_float,
    mut corr2: *mut ::core::ffi::c_float,
    mut corr3: *mut ::core::ffi::c_float,
    mut sumArr1: *mut ::core::ffi::c_double,
    mut sumArr2: *mut ::core::ffi::c_double,
    mut sumArr3: *mut ::core::ffi::c_double,
) {
    let mut ix: ::core::ffi::c_int = 0;
    let mut iy: ::core::ffi::c_int = 0;
    let mut aBase: ::core::ffi::c_int = 0;
    let mut nsum: ::core::ffi::c_int = 0;
    let mut end: ::core::ffi::c_int = 0;
    let mut bBase: ::core::ffi::c_int = 0;
    let mut aWgtBase: ::core::ffi::c_int = 0;
    let mut bWgtBase: ::core::ffi::c_int = 0;
    let mut bWgtBase1: ::core::ffi::c_int = 0;
    let mut bWgtBase3: ::core::ffi::c_int = 0;
    let mut wInd: ::core::ffi::c_int = 0;
    let mut abSum1: ::core::ffi::c_double = 0.;
    let mut abSum2: ::core::ffi::c_double = 0.;
    let mut abSum3: ::core::ffi::c_double = 0.;
    let mut aSumSq1: ::core::ffi::c_double = 0.;
    let mut aSumSq2: ::core::ffi::c_double = 0.;
    let mut aSumSq3: ::core::ffi::c_double = 0.;
    let mut aSum1: ::core::ffi::c_double = 0.;
    let mut aSum2: ::core::ffi::c_double = 0.;
    let mut aSum3: ::core::ffi::c_double = 0.;
    let mut bSumSq1: ::core::ffi::c_double = 0.;
    let mut bSumSq2: ::core::ffi::c_double = 0.;
    let mut bSumSq3: ::core::ffi::c_double = 0.;
    let mut bSumSq: ::core::ffi::c_double = 0.;
    let mut denom: ::core::ffi::c_double = 0.;
    let mut amean: ::core::ffi::c_double = 0.;
    let mut bmean: ::core::ffi::c_double = 0.;
    let mut bSum1: ::core::ffi::c_double = 0.;
    let mut bSum2: ::core::ffi::c_double = 0.;
    let mut bSum3: ::core::ffi::c_double = 0.;
    let mut bSum: ::core::ffi::c_double = 0.;
    let mut wSum2: ::core::ffi::c_double = 0.;
    let mut wSum1: ::core::ffi::c_double = 0.;
    let mut wSum3: ::core::ffi::c_double = 0.;
    let mut abTmp1: ::core::ffi::c_double = 0.;
    let mut abTmp2: ::core::ffi::c_double = 0.;
    let mut abTmp3: ::core::ffi::c_double = 0.;
    let mut aTmp1: ::core::ffi::c_double = 0.;
    let mut aTmpSq1: ::core::ffi::c_double = 0.;
    let mut aTmp2: ::core::ffi::c_double = 0.;
    let mut aTmpSq2: ::core::ffi::c_double = 0.;
    let mut aTmp3: ::core::ffi::c_double = 0.;
    let mut aTmpSq3: ::core::ffi::c_double = 0.;
    let mut aTmpSq: ::core::ffi::c_double = 0.;
    let mut bTmp1: ::core::ffi::c_double = 0.;
    let mut bTmp2: ::core::ffi::c_double = 0.;
    let mut bTmp3: ::core::ffi::c_double = 0.;
    let mut bTmpSq1: ::core::ffi::c_double = 0.;
    let mut bTmpSq2: ::core::ffi::c_double = 0.;
    let mut bTmpSq3: ::core::ffi::c_double = 0.;
    let mut wTmp2: ::core::ffi::c_double = 0.;
    let mut wTmp1: ::core::ffi::c_double = 0.;
    let mut wTmp3: ::core::ffi::c_double = 0.;
    let mut aTmp: ::core::ffi::c_double = 0.;
    let mut aval: ::core::ffi::c_float = 0.;
    let mut bval1: ::core::ffi::c_float = 0.;
    let mut bval2: ::core::ffi::c_float = 0.;
    let mut bval3: ::core::ffi::c_float = 0.;
    let mut bval: ::core::ffi::c_float = 0.;
    let mut wgt: ::core::ffi::c_float = 0.;
    let mut wgt1: ::core::ffi::c_float = 0.;
    let mut wgt3: ::core::ffi::c_float = 0.;
    let mut awgt: ::core::ffi::c_float = 0.;
    let mut numThreads: ::core::ffi::c_int = 0;
    let mut maxThreads: ::core::ffi::c_int = 8 as ::core::ffi::c_int;
    numThreads = (floor(
        sqrt((ix1 - ix0) as ::core::ffi::c_double * (iy1 - iy0) as ::core::ffi::c_double) / 80.0f64
            + 0.5f64,
    ) as ::core::ffi::c_int as ::core::ffi::c_double
        * (if !aWeights.is_null() { 2.0f64 } else { 1.0f64 }))
        as ::core::ffi::c_int;
    numThreads = if 1 as ::core::ffi::c_int
        > (if maxThreads < numThreads {
            maxThreads
        } else {
            numThreads
        }) {
        1 as ::core::ffi::c_int
    } else if maxThreads < numThreads {
        maxThreads
    } else {
        numThreads
    };
    numThreads = crate::imod::libcfshr::b3dutil::num_omp_threads(numThreads);
    aSumSq3 = 0.0f64;
    aSum3 = aSumSq3;
    aSumSq2 = aSum3;
    aSum1 = aSumSq2;
    aSumSq1 = aSum1;
    aSum2 = aSumSq1;
    abSum3 = aSum2;
    abSum2 = abSum3;
    abSum1 = abSum2;
    bSumSq3 = 0.0f64;
    bSumSq2 = bSumSq3;
    bSumSq1 = bSumSq2;
    wSum3 = 0.0f64;
    wSum2 = wSum3;
    wSum1 = wSum2;
    bSum3 = wSum1;
    bSum2 = bSum3;
    bSum1 = bSum2;
    iy = iy0
        + (if !aWeights.is_null() {
            0 as ::core::ffi::c_int
        } else {
            1 as ::core::ffi::c_int
        });
    while iy
        <= iy1
            - (if !aWeights.is_null() {
                0 as ::core::ffi::c_int
            } else {
                1 as ::core::ffi::c_int
            })
    {
        aBase = iy * nxDim;
        bBase = (iy - delY) * nxDim - delX;
        bSum = 0.0f64;
        bSumSq = bSum;
        wTmp3 = 0.0f64;
        wTmp1 = wTmp3;
        wTmp2 = wTmp1;
        abTmp3 = wTmp2;
        abTmp2 = abTmp3;
        abTmp1 = abTmp2;
        aTmpSq3 = 0.0f64;
        aTmp3 = aTmpSq3;
        aTmpSq2 = aTmp3;
        aTmp2 = aTmpSq2;
        aTmpSq1 = aTmp2;
        aTmp1 = aTmpSq1;
        bTmpSq3 = 0.0f64;
        bTmpSq2 = bTmpSq3;
        bTmpSq1 = bTmpSq2;
        bTmp3 = bTmpSq1;
        bTmp2 = bTmp3;
        bTmp1 = bTmp2;
        if !aWeights.is_null() {
            aWgtBase = (iy / binning + wgtYoffset) * nxWgt + wgtXoffset;
            bWgtBase1 =
                ((iy + 1 as ::core::ffi::c_int - delY) / binning + wgtYoffset) * nxWgt + wgtXoffset;
            bWgtBase = ((iy - delY) / binning + wgtYoffset) * nxWgt + wgtXoffset;
            bWgtBase3 =
                ((iy - 1 as ::core::ffi::c_int - delY) / binning + wgtYoffset) * nxWgt + wgtXoffset;
            ix = ix0;
            while ix <= ix1 {
                awgt = *aWeights.offset((ix / binning + aWgtBase) as isize);
                wInd = (ix - delX) / binning;
                wgt1 = awgt * *bWeights.offset((wInd + bWgtBase1) as isize);
                wgt = awgt * *bWeights.offset((wInd + bWgtBase) as isize);
                wgt3 = awgt * *bWeights.offset((wInd + bWgtBase3) as isize);
                wTmp1 += wgt1 as ::core::ffi::c_double;
                wTmp2 += wgt as ::core::ffi::c_double;
                wTmp3 += wgt3 as ::core::ffi::c_double;
                aval = *array.offset((ix + aBase) as isize);
                bval = *brray.offset((ix + bBase) as isize);
                bval1 = *brray.offset((ix + bBase + nxDim) as isize);
                bval3 = *brray.offset((ix + bBase - nxDim) as isize);
                aTmp1 += (aval * wgt1) as ::core::ffi::c_double;
                aTmp2 += (aval * wgt) as ::core::ffi::c_double;
                aTmp3 += (aval * wgt3) as ::core::ffi::c_double;
                bTmp1 += (bval1 * wgt1) as ::core::ffi::c_double;
                bTmp2 += (bval * wgt) as ::core::ffi::c_double;
                bTmp3 += (bval3 * wgt3) as ::core::ffi::c_double;
                aTmpSq1 += (aval * aval * wgt1) as ::core::ffi::c_double;
                aTmpSq2 += (aval * aval * wgt) as ::core::ffi::c_double;
                aTmpSq3 += (aval * aval * wgt3) as ::core::ffi::c_double;
                bTmpSq1 += (bval1 * bval1 * wgt1) as ::core::ffi::c_double;
                bTmpSq2 += (bval * bval * wgt) as ::core::ffi::c_double;
                bTmpSq3 += (bval3 * bval3 * wgt3) as ::core::ffi::c_double;
                abTmp1 += (aval * bval1 * wgt1) as ::core::ffi::c_double;
                abTmp2 += (aval * bval * wgt) as ::core::ffi::c_double;
                abTmp3 += (aval * bval3 * wgt3) as ::core::ffi::c_double;
                ix += 1;
            }
        } else {
            aBase = iy * nxDim;
            bBase = (iy - delY) * nxDim - delX;
            bSum = 0.0f64;
            bSumSq = bSum;
            aTmpSq = 0 as ::core::ffi::c_int as ::core::ffi::c_double;
            aTmp = aTmpSq;
            abTmp3 = aTmp;
            abTmp2 = abTmp3;
            abTmp1 = abTmp2;
            ix = ix0;
            while ix <= ix1 {
                aval = *array.offset((ix + aBase) as isize);
                aTmp += aval as ::core::ffi::c_double;
                bSum += *brray.offset((ix + bBase) as isize) as ::core::ffi::c_double;
                aTmpSq += (aval * aval) as ::core::ffi::c_double;
                bSumSq += (*brray.offset((ix + bBase) as isize)
                    * *brray.offset((ix + bBase) as isize))
                    as ::core::ffi::c_double;
                abTmp1 +=
                    (aval * *brray.offset((ix + bBase + nxDim) as isize)) as ::core::ffi::c_double;
                abTmp2 += (aval * *brray.offset((ix + bBase) as isize)) as ::core::ffi::c_double;
                abTmp3 +=
                    (aval * *brray.offset((ix + bBase - nxDim) as isize)) as ::core::ffi::c_double;
                ix += 1;
            }
            wTmp1 += (ix1 + 1 as ::core::ffi::c_int - ix0) as ::core::ffi::c_double;
            wTmp2 += (ix1 + 1 as ::core::ffi::c_int - ix0) as ::core::ffi::c_double;
            wTmp3 += (ix1 + 1 as ::core::ffi::c_int - ix0) as ::core::ffi::c_double;
            bTmp3 = bSum;
            bTmp2 = bTmp3;
            bTmp1 = bTmp2;
            bTmpSq3 = bSumSq;
            bTmpSq2 = bTmpSq3;
            bTmpSq1 = bTmpSq2;
            aTmp3 = aTmp;
            aTmp2 = aTmp3;
            aTmp1 = aTmp2;
            aTmpSq3 = aTmpSq;
            aTmpSq2 = aTmpSq3;
            aTmpSq1 = aTmpSq2;
        }
        wSum1 += wTmp1;
        wSum2 += wTmp2;
        wSum3 += wTmp3;
        aSum1 += aTmp1;
        aSum2 += aTmp2;
        aSum3 += aTmp3;
        aSumSq1 += aTmpSq1;
        aSumSq2 += aTmpSq2;
        aSumSq3 += aTmpSq3;
        abSum1 += abTmp1;
        abSum2 += abTmp2;
        abSum3 += abTmp3;
        bSum1 += bTmp1;
        bSum2 += bTmp2;
        bSum3 += bTmp3;
        bSumSq1 += bTmpSq1;
        bSumSq2 += bTmpSq2;
        bSumSq3 += bTmpSq3;
        iy += 1;
    }
    if aWeights.is_null() {
        aBase = iy0 * nxDim;
        bBase = (iy0 - delY) * nxDim - delX;
        bTmpSq3 = 0 as ::core::ffi::c_int as ::core::ffi::c_double;
        bTmpSq2 = bTmpSq3;
        bTmpSq1 = bTmpSq2;
        bTmp3 = bTmpSq1;
        bTmp2 = bTmp3;
        bTmp1 = bTmp2;
        aTmpSq = 0 as ::core::ffi::c_int as ::core::ffi::c_double;
        aTmp = aTmpSq;
        abTmp3 = aTmp;
        abTmp2 = abTmp3;
        abTmp1 = abTmp2;
        ix = ix0;
        while ix <= ix1 {
            aval = *array.offset((ix + aBase) as isize);
            bval1 = *brray.offset((ix + bBase + nxDim) as isize);
            bval2 = *brray.offset((ix + bBase) as isize);
            bval3 = *brray.offset((ix + bBase - nxDim) as isize);
            aTmp += aval as ::core::ffi::c_double;
            aTmpSq += (aval * aval) as ::core::ffi::c_double;
            abTmp1 += (aval * bval1) as ::core::ffi::c_double;
            abTmp2 += (aval * bval2) as ::core::ffi::c_double;
            abTmp3 += (aval * bval3) as ::core::ffi::c_double;
            bTmp2 += bval2 as ::core::ffi::c_double;
            bTmp3 += (bval3 + bval2) as ::core::ffi::c_double;
            bTmpSq2 += (bval2 * bval2) as ::core::ffi::c_double;
            bTmpSq3 += (bval3 * bval3 + bval2 * bval2) as ::core::ffi::c_double;
            ix += 1;
        }
        aBase = iy1 * nxDim;
        bBase = (iy1 - delY) * nxDim - delX;
        ix = ix0;
        while ix <= ix1 {
            aval = *array.offset((ix + aBase) as isize);
            bval1 = *brray.offset((ix + bBase + nxDim) as isize);
            bval2 = *brray.offset((ix + bBase) as isize);
            bval3 = *brray.offset((ix + bBase - nxDim) as isize);
            aTmp += aval as ::core::ffi::c_double;
            aTmpSq += (aval * aval) as ::core::ffi::c_double;
            abTmp1 += (aval * bval1) as ::core::ffi::c_double;
            abTmp2 += (aval * bval2) as ::core::ffi::c_double;
            abTmp3 += (aval * bval3) as ::core::ffi::c_double;
            bTmp1 += (bval1 + bval2) as ::core::ffi::c_double;
            bTmp2 += bval2 as ::core::ffi::c_double;
            bTmpSq1 += (bval1 * bval1 + bval2 * bval2) as ::core::ffi::c_double;
            bTmpSq2 += (bval2 * bval2) as ::core::ffi::c_double;
            ix += 1;
        }
        wSum1 += (2 as ::core::ffi::c_int * (ix1 + 1 as ::core::ffi::c_int - ix0))
            as ::core::ffi::c_double;
        wSum2 += (2 as ::core::ffi::c_int * (ix1 + 1 as ::core::ffi::c_int - ix0))
            as ::core::ffi::c_double;
        wSum3 += (2 as ::core::ffi::c_int * (ix1 + 1 as ::core::ffi::c_int - ix0))
            as ::core::ffi::c_double;
        aSum1 += aTmp;
        aSum2 += aTmp;
        aSum3 += aTmp;
        aSumSq1 += aTmpSq;
        aSumSq2 += aTmpSq;
        aSumSq3 += aTmpSq;
        abSum1 += abTmp1;
        abSum2 += abTmp2;
        abSum3 += abTmp3;
        bSum1 += bTmp1;
        bSum2 += bTmp2;
        bSum3 += bTmp3;
        bSumSq1 += bTmpSq1;
        bSumSq2 += bTmpSq2;
        bSumSq3 += bTmpSq3;
    }
    *corr1 = crate::imod::libcfshr::filtxcorr::weighted_corr_from_sums(
        aSum1,
        aSumSq1,
        bSum1,
        bSumSq1,
        abSum1,
        wSum1,
        sumArr1,
        b"\0" as *const u8 as *const ::core::ffi::c_char,
    ) as ::core::ffi::c_float;
    *corr2 = crate::imod::libcfshr::filtxcorr::weighted_corr_from_sums(
        aSum2,
        aSumSq2,
        bSum2,
        bSumSq2,
        abSum2,
        wSum2,
        sumArr2,
        b"\0" as *const u8 as *const ::core::ffi::c_char,
    ) as ::core::ffi::c_float;
    *corr3 = crate::imod::libcfshr::filtxcorr::weighted_corr_from_sums(
        aSum3,
        aSumSq3,
        bSum3,
        bSumSq3,
        abSum3,
        wSum3,
        sumArr3,
        b"\0" as *const u8 as *const ::core::ffi::c_char,
    ) as ::core::ffi::c_float;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mont_xc_find_best_corr(
    mut array: *mut ::core::ffi::c_float,
    mut brray: *mut ::core::ffi::c_float,
    mut nxDim: ::core::ffi::c_int,
    mut nx: ::core::ffi::c_int,
    mut ny: ::core::ffi::c_int,
    mut nxTrim: ::core::ffi::c_int,
    mut nyTrim: ::core::ffi::c_int,
    mut ixStart: ::core::ffi::c_int,
    mut ixEnd: ::core::ffi::c_int,
    mut iyStart: ::core::ffi::c_int,
    mut iyEnd: ::core::ffi::c_int,
    mut delX: *mut ::core::ffi::c_float,
    mut delY: *mut ::core::ffi::c_float,
    mut corr: *mut ::core::ffi::c_float,
    mut maxDist: ::core::ffi::c_float,
    mut aWeights: *mut ::core::ffi::c_float,
    mut bWeights: *mut ::core::ffi::c_float,
    mut nxWgt: ::core::ffi::c_int,
    mut binning: ::core::ffi::c_int,
    mut wgtXoffset: ::core::ffi::c_int,
    mut wgtYoffset: ::core::ffi::c_int,
    mut threshCCC: ::core::ffi::c_float,
    mut bestSumArr: *mut ::core::ffi::c_double,
) {
    let mut corrs: [[::core::ffi::c_float; 3]; 3] = [[0.; 3]; 3];
    let mut corrTmp: [[::core::ffi::c_float; 3]; 3] = [[0.; 3]; 3];
    let mut cccMax: ::core::ffi::c_float = 0.;
    let mut sumArrs: [[[::core::ffi::c_double; 6]; 3]; 3] = [[[0.; 6]; 3]; 3];
    let mut sumTmp: [[[::core::ffi::c_double; 6]; 3]; 3] = [[[0.; 6]; 3]; 3];
    let mut done: [[::core::ffi::c_int; 3]; 3] = [[0; 3]; 3];
    let mut first: ::core::ffi::c_int = 1 as ::core::ffi::c_int;
    let mut curDelX: ::core::ffi::c_int =
        floor(*delX as ::core::ffi::c_double + 0.5f64) as ::core::ffi::c_int;
    let mut curDelY: ::core::ffi::c_int =
        floor(*delY as ::core::ffi::c_double + 0.5f64) as ::core::ffi::c_int;
    let mut ix: ::core::ffi::c_int = 0;
    let mut iy: ::core::ffi::c_int = 0;
    let mut ixMax: ::core::ffi::c_int = 0;
    let mut iyMax: ::core::ffi::c_int = 0;
    let mut ix0: ::core::ffi::c_int = 0;
    let mut ix1: ::core::ffi::c_int = 0;
    let mut iy0: ::core::ffi::c_int = 0;
    let mut iy1: ::core::ffi::c_int = 0;
    let mut nc: ::core::ffi::c_int = 0;
    let mut ind: ::core::ffi::c_int = 0;
    let mut needCol: ::core::ffi::c_int = -(1 as ::core::ffi::c_int);
    ix0 = if ixStart > nxTrim + curDelX {
        ixStart
    } else {
        nxTrim + curDelX
    };
    ix1 = if ixEnd < nx + curDelX - nxTrim {
        ixEnd
    } else {
        nx + curDelX - nxTrim
    };
    iy0 = if iyStart > nyTrim + curDelY {
        iyStart
    } else {
        nyTrim + curDelY
    };
    iy1 = if iyEnd < ny + curDelY - nyTrim {
        iyEnd
    } else {
        ny + curDelY - nyTrim
    };
    fflush(stdout);
    iy = 0 as ::core::ffi::c_int;
    while iy < 3 as ::core::ffi::c_int {
        done[iy as usize][2 as ::core::ffi::c_int as usize] = 0 as ::core::ffi::c_int;
        done[iy as usize][1 as ::core::ffi::c_int as usize] =
            done[iy as usize][2 as ::core::ffi::c_int as usize];
        done[iy as usize][0 as ::core::ffi::c_int as usize] =
            done[iy as usize][1 as ::core::ffi::c_int as usize];
        iy += 1;
    }
    while pow(
        (curDelX as ::core::ffi::c_float - *delX) as ::core::ffi::c_double,
        2.0f64,
    ) + pow(
        (curDelY as ::core::ffi::c_float - *delY) as ::core::ffi::c_double,
        2.0f64,
    ) < (maxDist * maxDist) as ::core::ffi::c_double
    {
        ix0 = if ixStart > nxTrim + curDelX - 1 as ::core::ffi::c_int {
            ixStart
        } else {
            nxTrim + curDelX - 1 as ::core::ffi::c_int
        };
        ix1 = if ixEnd < nx + curDelX + 1 as ::core::ffi::c_int - nxTrim {
            ixEnd
        } else {
            nx + curDelX + 1 as ::core::ffi::c_int - nxTrim
        };
        iy0 = if iyStart > nyTrim + curDelY - 1 as ::core::ffi::c_int {
            iyStart
        } else {
            nyTrim + curDelY - 1 as ::core::ffi::c_int
        };
        iy1 = if iyEnd < ny + curDelY + 1 as ::core::ffi::c_int - nyTrim {
            iyEnd
        } else {
            ny + curDelY + 1 as ::core::ffi::c_int - nyTrim
        };
        if needCol >= 0 as ::core::ffi::c_int {
            nc = needCol;
            column_of_three_corrs(
                array,
                brray,
                nxDim,
                ix0,
                ix1,
                iy0,
                iy1,
                curDelX + nc - 1 as ::core::ffi::c_int,
                curDelY,
                aWeights,
                bWeights,
                nxWgt,
                binning,
                wgtXoffset,
                wgtYoffset,
                (&raw mut *(&raw mut corrs as *mut [::core::ffi::c_float; 3])
                    .offset(0 as ::core::ffi::c_int as isize)
                    as *mut ::core::ffi::c_float)
                    .offset(nc as isize) as *mut ::core::ffi::c_float,
                (&raw mut *(&raw mut corrs as *mut [::core::ffi::c_float; 3])
                    .offset(1 as ::core::ffi::c_int as isize)
                    as *mut ::core::ffi::c_float)
                    .offset(nc as isize) as *mut ::core::ffi::c_float,
                (&raw mut *(&raw mut corrs as *mut [::core::ffi::c_float; 3])
                    .offset(2 as ::core::ffi::c_int as isize)
                    as *mut ::core::ffi::c_float)
                    .offset(nc as isize) as *mut ::core::ffi::c_float,
                (&raw mut *(&raw mut *(&raw mut sumArrs as *mut [[::core::ffi::c_double; 6]; 3])
                    .offset(0 as ::core::ffi::c_int as isize)
                    as *mut [::core::ffi::c_double; 6])
                    .offset(nc as isize) as *mut ::core::ffi::c_double)
                    .offset(0 as ::core::ffi::c_int as isize)
                    as *mut ::core::ffi::c_double,
                (&raw mut *(&raw mut *(&raw mut sumArrs as *mut [[::core::ffi::c_double; 6]; 3])
                    .offset(1 as ::core::ffi::c_int as isize)
                    as *mut [::core::ffi::c_double; 6])
                    .offset(nc as isize) as *mut ::core::ffi::c_double)
                    .offset(0 as ::core::ffi::c_int as isize)
                    as *mut ::core::ffi::c_double,
                (&raw mut *(&raw mut *(&raw mut sumArrs as *mut [[::core::ffi::c_double; 6]; 3])
                    .offset(2 as ::core::ffi::c_int as isize)
                    as *mut [::core::ffi::c_double; 6])
                    .offset(nc as isize) as *mut ::core::ffi::c_double)
                    .offset(0 as ::core::ffi::c_int as isize)
                    as *mut ::core::ffi::c_double,
            );
            done[2 as ::core::ffi::c_int as usize][nc as usize] = 1 as ::core::ffi::c_int;
            done[1 as ::core::ffi::c_int as usize][nc as usize] =
                done[2 as ::core::ffi::c_int as usize][nc as usize];
            done[0 as ::core::ffi::c_int as usize][nc as usize] =
                done[1 as ::core::ffi::c_int as usize][nc as usize];
        }
        iy = 0 as ::core::ffi::c_int;
        while iy < 3 as ::core::ffi::c_int {
            if !(done[iy as usize][0 as ::core::ffi::c_int as usize] != 0
                && done[iy as usize][1 as ::core::ffi::c_int as usize] != 0
                && done[iy as usize][2 as ::core::ffi::c_int as usize] != 0)
            {
                row_of_three_corrs(
                    array,
                    brray,
                    nxDim,
                    ix0,
                    ix1,
                    iy0,
                    iy1,
                    curDelX,
                    curDelY + iy - 1 as ::core::ffi::c_int,
                    aWeights,
                    bWeights,
                    nxWgt,
                    binning,
                    wgtXoffset,
                    wgtYoffset,
                    (&raw mut *(&raw mut corrs as *mut [::core::ffi::c_float; 3])
                        .offset(iy as isize) as *mut ::core::ffi::c_float)
                        .offset(0 as ::core::ffi::c_int as isize)
                        as *mut ::core::ffi::c_float,
                    (&raw mut *(&raw mut corrs as *mut [::core::ffi::c_float; 3])
                        .offset(iy as isize) as *mut ::core::ffi::c_float)
                        .offset(1 as ::core::ffi::c_int as isize)
                        as *mut ::core::ffi::c_float,
                    (&raw mut *(&raw mut corrs as *mut [::core::ffi::c_float; 3])
                        .offset(iy as isize) as *mut ::core::ffi::c_float)
                        .offset(2 as ::core::ffi::c_int as isize)
                        as *mut ::core::ffi::c_float,
                    (&raw mut *(&raw mut *(&raw mut sumArrs as *mut [[::core::ffi::c_double; 6]; 3])
                        .offset(iy as isize)
                        as *mut [::core::ffi::c_double; 6])
                        .offset(0 as ::core::ffi::c_int as isize)
                        as *mut ::core::ffi::c_double)
                        .offset(0 as ::core::ffi::c_int as isize)
                        as *mut ::core::ffi::c_double,
                    (&raw mut *(&raw mut *(&raw mut sumArrs as *mut [[::core::ffi::c_double; 6]; 3])
                        .offset(iy as isize)
                        as *mut [::core::ffi::c_double; 6])
                        .offset(1 as ::core::ffi::c_int as isize)
                        as *mut ::core::ffi::c_double)
                        .offset(0 as ::core::ffi::c_int as isize)
                        as *mut ::core::ffi::c_double,
                    (&raw mut *(&raw mut *(&raw mut sumArrs as *mut [[::core::ffi::c_double; 6]; 3])
                        .offset(iy as isize)
                        as *mut [::core::ffi::c_double; 6])
                        .offset(2 as ::core::ffi::c_int as isize)
                        as *mut ::core::ffi::c_double)
                        .offset(0 as ::core::ffi::c_int as isize)
                        as *mut ::core::ffi::c_double,
                );
                done[iy as usize][2 as ::core::ffi::c_int as usize] = 1 as ::core::ffi::c_int;
                done[iy as usize][1 as ::core::ffi::c_int as usize] =
                    done[iy as usize][2 as ::core::ffi::c_int as usize];
                done[iy as usize][0 as ::core::ffi::c_int as usize] =
                    done[iy as usize][1 as ::core::ffi::c_int as usize];
            }
            iy += 1;
        }
        iyMax = 1 as ::core::ffi::c_int;
        ixMax = iyMax;
        cccMax = corrs[1 as ::core::ffi::c_int as usize][1 as ::core::ffi::c_int as usize];
        iy = 0 as ::core::ffi::c_int;
        while iy < 3 as ::core::ffi::c_int {
            ix = 0 as ::core::ffi::c_int;
            while ix < 3 as ::core::ffi::c_int {
                corrTmp[iy as usize][ix as usize] = corrs[iy as usize][ix as usize];
                done[iy as usize][ix as usize] = 0 as ::core::ffi::c_int;
                ind = 0 as ::core::ffi::c_int;
                while ind < 6 as ::core::ffi::c_int {
                    sumTmp[iy as usize][ix as usize][ind as usize] =
                        sumArrs[iy as usize][ix as usize][ind as usize];
                    ind += 1;
                }
                if corrs[iy as usize][ix as usize] > cccMax
                    || corrs[iy as usize][ix as usize] == cccMax
                        && (ixMax != 1 as ::core::ffi::c_int || iyMax != 1 as ::core::ffi::c_int)
                {
                    ixMax = ix;
                    iyMax = iy;
                    cccMax = corrs[iy as usize][ix as usize];
                }
                ix += 1;
            }
            iy += 1;
        }
        curDelX += ixMax - 1 as ::core::ffi::c_int;
        curDelY += iyMax - 1 as ::core::ffi::c_int;
        if ixMax == 1 as ::core::ffi::c_int && iyMax == 1 as ::core::ffi::c_int
            || cccMax < threshCCC
        {
            *corr = cccMax;
            *delX = (curDelX as ::core::ffi::c_double
                + (if ixMax == 1 as ::core::ffi::c_int && iyMax == 1 as ::core::ffi::c_int {
                    0.0f64
                } else {
                    crate::imod::libcfshr::filtxcorr::parabolic_fit_position(
                        corrs[1 as ::core::ffi::c_int as usize][0 as ::core::ffi::c_int as usize],
                        corrs[1 as ::core::ffi::c_int as usize][1 as ::core::ffi::c_int as usize],
                        corrs[1 as ::core::ffi::c_int as usize][2 as ::core::ffi::c_int as usize],
                    )
                })) as ::core::ffi::c_float;
            *delY = (curDelY as ::core::ffi::c_double
                + (if ixMax == 1 as ::core::ffi::c_int && iyMax == 1 as ::core::ffi::c_int {
                    0.0f64
                } else {
                    crate::imod::libcfshr::filtxcorr::parabolic_fit_position(
                        corrs[0 as ::core::ffi::c_int as usize][1 as ::core::ffi::c_int as usize],
                        corrs[1 as ::core::ffi::c_int as usize][1 as ::core::ffi::c_int as usize],
                        corrs[2 as ::core::ffi::c_int as usize][1 as ::core::ffi::c_int as usize],
                    )
                })) as ::core::ffi::c_float;
            if !aWeights.is_null() {
                ind = 0 as ::core::ffi::c_int;
                while ind < 6 as ::core::ffi::c_int {
                    *bestSumArr.offset(ind as isize) =
                        sumArrs[ixMax as usize][iyMax as usize][ind as usize];
                    ind += 1;
                }
            }
            return;
        }
        if ixMax != 1 as ::core::ffi::c_int {
            needCol = ixMax;
        }
        iy = if 0 as ::core::ffi::c_int > iyMax - 1 as ::core::ffi::c_int {
            0 as ::core::ffi::c_int
        } else {
            iyMax - 1 as ::core::ffi::c_int
        };
        while iy
            <= (if (2 as ::core::ffi::c_int) < iyMax + 1 as ::core::ffi::c_int {
                2 as ::core::ffi::c_int
            } else {
                iyMax + 1 as ::core::ffi::c_int
            })
        {
            ix = if 0 as ::core::ffi::c_int > ixMax - 1 as ::core::ffi::c_int {
                0 as ::core::ffi::c_int
            } else {
                ixMax - 1 as ::core::ffi::c_int
            };
            while ix
                <= (if (2 as ::core::ffi::c_int) < ixMax + 1 as ::core::ffi::c_int {
                    2 as ::core::ffi::c_int
                } else {
                    ixMax + 1 as ::core::ffi::c_int
                })
            {
                done[(iy + 1 as ::core::ffi::c_int - iyMax) as usize]
                    [(ix + 1 as ::core::ffi::c_int - ixMax) as usize] = 1 as ::core::ffi::c_int;
                corrs[(iy + 1 as ::core::ffi::c_int - iyMax) as usize]
                    [(ix + 1 as ::core::ffi::c_int - ixMax) as usize] =
                    corrTmp[iy as usize][ix as usize];
                ind = 0 as ::core::ffi::c_int;
                while ind < 6 as ::core::ffi::c_int {
                    sumArrs[(iy + 1 as ::core::ffi::c_int - iyMax) as usize]
                        [(ix + 1 as ::core::ffi::c_int - ixMax) as usize][ind as usize] =
                        sumTmp[iy as usize][ix as usize][ind as usize];
                    ind += 1;
                }
                ix += 1;
            }
            iy += 1;
        }
    }
    *corr = 0 as ::core::ffi::c_int as ::core::ffi::c_float;
}
pub const SLICE_MODE_FLOAT: ::core::ffi::c_int = 2 as ::core::ffi::c_int;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_sizes_and_binning_follow_source_edge_geometry() {
        unsafe {
            let mut pieces = [100, 120];
            let mut overlap = [20, 25];
            let mut use_indent = 0;
            let mut boxed = [0; 2];
            let mut extra = [0; 2];
            let mut xpad = 0;
            let mut ypad = 0;
            let mut maximum = 0;
            mont_xc_basic_sizes(
                0,
                1,
                4,
                pieces.as_mut_ptr(),
                overlap.as_mut_ptr(),
                2.0,
                0.2,
                0.1,
                5,
                &mut use_indent,
                boxed.as_mut_ptr(),
                extra.as_mut_ptr(),
                &mut xpad,
                &mut ypad,
                &mut maximum,
            );
            assert_eq!(use_indent, 4);
            assert!(boxed[0] >= 12 && boxed[1] > 0 && xpad >= boxed[0] && ypad >= boxed[1]);
            let mut padded = 0;
            let mut area = 0;
            assert!(
                mont_xc_find_binning(
                    4,
                    200,
                    4,
                    pieces.as_mut_ptr(),
                    overlap.as_mut_ptr(),
                    2.,
                    0.2,
                    0.1,
                    5,
                    &mut padded,
                    &mut area
                ) >= 1
            );
        }
    }
}
