#![allow(
    dead_code,
    non_snake_case,
    non_camel_case_types,
    unused_mut,
    unused_assignments,
    unsafe_op_in_unsafe_fn
)]
//! Mechanical C2Rust baseline of `IMOD/libcfshr/gcvspl.c`; source-faithful normalization follows.

pub type Integer = ::core::ffi::c_int;
pub type DoubleReal = ::core::ffi::c_double;
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gcvspl(
    mut x: *mut ::core::ffi::c_double,
    mut y: *mut ::core::ffi::c_double,
    mut yDim: ::core::ffi::c_int,
    mut wgtx: *mut ::core::ffi::c_double,
    mut wgty: *mut ::core::ffi::c_double,
    mut mOrder: ::core::ffi::c_int,
    mut numVal: ::core::ffi::c_int,
    mut numYcol: ::core::ffi::c_int,
    mut mode: ::core::ffi::c_int,
    mut val: ::core::ffi::c_double,
    mut coeff: *mut ::core::ffi::c_double,
    mut coeffDim: ::core::ffi::c_int,
    mut work: *mut ::core::ffi::c_double,
    mut ier: *mut ::core::ffi::c_int,
) -> ::core::ffi::c_int {
    return gcvspl_(
        x as *mut DoubleReal,
        y as *mut DoubleReal,
        &raw mut yDim,
        wgtx as *mut DoubleReal,
        wgty as *mut DoubleReal,
        &raw mut mOrder,
        &raw mut numVal,
        &raw mut numYcol,
        &raw mut mode,
        &raw mut val,
        coeff as *mut DoubleReal,
        &raw mut coeffDim,
        work as *mut DoubleReal,
        ier as *mut Integer,
    );
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn splder(
    mut derivOrder: ::core::ffi::c_int,
    mut mOrder: ::core::ffi::c_int,
    mut numVal: ::core::ffi::c_int,
    mut tVal: ::core::ffi::c_double,
    mut x: *mut ::core::ffi::c_double,
    mut coeff: *mut ::core::ffi::c_double,
    mut nearInd: *mut ::core::ffi::c_int,
    mut work: *mut ::core::ffi::c_double,
) -> ::core::ffi::c_double {
    return splder_(
        &raw mut derivOrder,
        &raw mut mOrder,
        &raw mut numVal,
        &raw mut tVal,
        x as *mut DoubleReal,
        coeff as *mut DoubleReal,
        nearInd as *mut Integer,
        work as *mut DoubleReal,
    );
}
static mut c_b6: DoubleReal = 1e-15f64;
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gcvspl_(
    mut x: *mut DoubleReal,
    mut y: *mut DoubleReal,
    mut ny: *mut Integer,
    mut wx: *mut DoubleReal,
    mut wy: *mut DoubleReal,
    mut m: *mut Integer,
    mut n: *mut Integer,
    mut k: *mut Integer,
    mut md: *mut Integer,
    mut val: *mut DoubleReal,
    mut c__: *mut DoubleReal,
    mut nc: *mut Integer,
    mut wk: *mut DoubleReal,
    mut ier: *mut Integer,
) -> ::core::ffi::c_int {
    let mut current_block: u64;
    static mut m2: Integer = 0 as Integer;
    static mut nm1: Integer = 0 as Integer;
    static mut el: DoubleReal = 0.0f64;
    let mut y_dim1: Integer = 0;
    let mut y_offset: Integer = 0;
    let mut c_dim1: Integer = 0;
    let mut c_offset: Integer = 0;
    let mut i__1: Integer = 0;
    static mut i__: Integer = 0;
    static mut j: Integer = 0;
    static mut r1: DoubleReal = 0.;
    static mut r2: DoubleReal = 0.;
    static mut r3: DoubleReal = 0.;
    static mut r4: DoubleReal = 0.;
    static mut ib: Integer = 0;
    static mut gf1: DoubleReal = 0.;
    static mut gf2: DoubleReal = 0.;
    static mut gf3: DoubleReal = 0.;
    static mut gf4: DoubleReal = 0.;
    static mut iwe: Integer = 0;
    static mut err: DoubleReal = 0.;
    static mut nm2m1: Integer = 0;
    static mut nm2p1: Integer = 0;
    #[unsafe(export_name = "splc_")]
    pub unsafe extern "C" fn splc(
        mut m_0: *mut Integer,
        mut n_0: *mut Integer,
        mut k_0: *mut Integer,
        mut y_0: *mut DoubleReal,
        mut ny_0: *mut Integer,
        mut wx_0: *mut DoubleReal,
        mut wy_0: *mut DoubleReal,
        mut mode: *mut Integer,
        mut val_0: *mut DoubleReal,
        mut p: *mut DoubleReal,
        mut eps: *mut DoubleReal,
        mut c___0: *mut DoubleReal,
        mut nc_0: *mut Integer,
        mut stat: *mut DoubleReal,
        mut b: *mut DoubleReal,
        mut we: *mut DoubleReal,
        mut el_0: *mut DoubleReal,
        mut bwe: *mut DoubleReal,
    ) -> DoubleReal {
        let mut y_dim1_0: Integer = 0;
        let mut y_offset_0: Integer = 0;
        let mut c_dim1_0: Integer = 0;
        let mut c_offset_0: Integer = 0;
        let mut b_dim1: Integer = 0;
        let mut b_offset: Integer = 0;
        let mut we_dim1: Integer = 0;
        let mut we_offset: Integer = 0;
        let mut bwe_dim1: Integer = 0;
        let mut bwe_offset: Integer = 0;
        let mut i__1_0: Integer = 0;
        let mut i__2: Integer = 0;
        let mut i__3: Integer = 0;
        let mut i__4: Integer = 0;
        let mut ret_val: DoubleReal = 0.;
        let mut d__1: DoubleReal = 0.;
        static mut i___0: Integer = 0;
        static mut j_0: Integer = 0;
        static mut l: Integer = 0;
        static mut dp: DoubleReal = 0.;
        static mut km: Integer = 0;
        static mut dt: DoubleReal = 0.;
        static mut kp: Integer = 0;
        static mut pel: DoubleReal = 0.;
        static mut esn: DoubleReal = 0.;
        static mut trn: DoubleReal = 0.;
        #[unsafe(export_name = "trinv_")]
        pub unsafe extern "C" fn trinv(
            mut b_0: *mut DoubleReal,
            mut e: *mut DoubleReal,
            mut m_1: *mut Integer,
            mut n_1: *mut Integer,
        ) -> DoubleReal {
            let mut b_dim1_0: Integer = 0;
            let mut b_offset_0: Integer = 0;
            let mut e_dim1: Integer = 0;
            let mut e_offset: Integer = 0;
            let mut i__1_1: Integer = 0;
            let mut i__2_0: Integer = 0;
            let mut i__3_0: Integer = 0;
            let mut ret_val_0: DoubleReal = 0.;
            static mut i___1: Integer = 0;
            static mut j_1: Integer = 0;
            static mut k_1: Integer = 0;
            static mut dd: DoubleReal = 0.;
            static mut dl: DoubleReal = 0.;
            static mut mi: Integer = 0;
            static mut du: DoubleReal = 0.;
            static mut mn: Integer = 0;
            static mut mp: Integer = 0;
            e_dim1 = (*m_1 - -*m_1 + 1 as ::core::ffi::c_int) as Integer;
            e_offset = -*m_1 + e_dim1;
            e = e.offset(-(e_offset as isize));
            b_dim1_0 = (*m_1 - -*m_1 + 1 as ::core::ffi::c_int) as Integer;
            b_offset_0 = -*m_1 + b_dim1_0;
            b_0 = b_0.offset(-(b_offset_0 as isize));
            *e.offset((*n_1 * e_dim1) as isize) = 1.0f64 / *e.offset((*n_1 * e_dim1) as isize);
            i___1 = (*n_1 - 1 as ::core::ffi::c_int) as Integer;
            while i___1 >= 1 as ::core::ffi::c_int {
                i__1_1 = *m_1;
                i__2_0 = *n_1 - i___1;
                mi = (if i__1_1 <= i__2_0 {
                    i__1_1 as ::core::ffi::c_int
                } else {
                    i__2_0 as ::core::ffi::c_int
                }) as Integer;
                dd = 1.0f64 / *e.offset((i___1 * e_dim1) as isize);
                i__1_1 = mi;
                k_1 = 1 as ::core::ffi::c_int as Integer;
                while k_1 <= i__1_1 {
                    *e.offset((k_1 + *n_1 * e_dim1) as isize) =
                        *e.offset((k_1 + i___1 * e_dim1) as isize) * dd;
                    *e.offset((-k_1 + e_dim1) as isize) =
                        *e.offset((-k_1 + (k_1 + i___1) * e_dim1) as isize);
                    k_1 += 1;
                }
                dd += dd as ::core::ffi::c_double;
                j_1 = mi;
                while j_1 >= 1 as ::core::ffi::c_int {
                    du = 0.0f64 as DoubleReal;
                    dl = 0.0f64 as DoubleReal;
                    i__1_1 = mi;
                    k_1 = 1 as ::core::ffi::c_int as Integer;
                    while k_1 <= i__1_1 {
                        du -= (*e.offset((k_1 + *n_1 * e_dim1) as isize)
                            * *e.offset((j_1 - k_1 + (i___1 + k_1) * e_dim1) as isize))
                            as ::core::ffi::c_double;
                        dl -= (*e.offset((-k_1 + e_dim1) as isize)
                            * *e.offset((k_1 - j_1 + (i___1 + j_1) * e_dim1) as isize))
                            as ::core::ffi::c_double;
                        k_1 += 1;
                    }
                    *e.offset((j_1 + i___1 * e_dim1) as isize) = du;
                    *e.offset((-j_1 + (j_1 + i___1) * e_dim1) as isize) = dl;
                    dd -= (*e.offset((j_1 + *n_1 * e_dim1) as isize) * dl
                        + *e.offset((-j_1 + e_dim1) as isize) * du)
                        as ::core::ffi::c_double;
                    j_1 -= 1;
                }
                *e.offset((i___1 * e_dim1) as isize) =
                    (dd as ::core::ffi::c_double * 0.5f64) as DoubleReal;
                i___1 -= 1;
            }
            dd = 0.0f64 as DoubleReal;
            i__1_1 = *n_1;
            i___1 = 1 as ::core::ffi::c_int as Integer;
            while i___1 <= i__1_1 {
                i__2_0 = *m_1;
                i__3_0 = (i___1 as ::core::ffi::c_int - 1 as ::core::ffi::c_int) as Integer;
                mn = -if i__2_0 <= i__3_0 {
                    i__2_0 as ::core::ffi::c_int
                } else {
                    i__3_0 as ::core::ffi::c_int
                } as Integer;
                i__2_0 = *m_1;
                i__3_0 = *n_1 - i___1;
                mp = (if i__2_0 <= i__3_0 {
                    i__2_0 as ::core::ffi::c_int
                } else {
                    i__3_0 as ::core::ffi::c_int
                }) as Integer;
                i__2_0 = mp;
                k_1 = mn;
                while k_1 <= i__2_0 {
                    dd += (*b_0.offset((k_1 + i___1 * b_dim1_0) as isize)
                        * *e.offset((-k_1 + (k_1 + i___1) * e_dim1) as isize))
                        as ::core::ffi::c_double;
                    k_1 += 1;
                }
                i___1 += 1;
            }
            ret_val_0 = dd;
            i__1_1 = *m_1;
            k_1 = 1 as ::core::ffi::c_int as Integer;
            while k_1 <= i__1_1 {
                *e.offset((k_1 + *n_1 * e_dim1) as isize) = 0.0f64 as DoubleReal;
                *e.offset((-k_1 + e_dim1) as isize) = 0.0f64 as DoubleReal;
                k_1 += 1;
            }
            return ret_val_0;
        }
        #[unsafe(export_name = "bandet_")]
        pub unsafe extern "C" fn bandet(
            mut e: *mut DoubleReal,
            mut m_1: *mut Integer,
            mut n_1: *mut Integer,
        ) -> ::core::ffi::c_int {
            let mut e_dim1: Integer = 0;
            let mut e_offset: Integer = 0;
            let mut i__1_1: Integer = 0;
            let mut i__2_0: Integer = 0;
            let mut i__3_0: Integer = 0;
            let mut i__4_0: Integer = 0;
            static mut i___1: Integer = 0;
            static mut k_1: Integer = 0;
            static mut l_0: Integer = 0;
            static mut di: DoubleReal = 0.;
            static mut dl: DoubleReal = 0.;
            static mut mi: Integer = 0;
            static mut km_0: Integer = 0;
            static mut lm: Integer = 0;
            static mut du: DoubleReal = 0.;
            e_dim1 = (*m_1 - -*m_1 + 1 as ::core::ffi::c_int) as Integer;
            e_offset = -*m_1 + e_dim1;
            e = e.offset(-(e_offset as isize));
            if *m_1 <= 0 as ::core::ffi::c_int {
                return 0 as ::core::ffi::c_int;
            }
            i__1_1 = *n_1;
            i___1 = 1 as ::core::ffi::c_int as Integer;
            while i___1 <= i__1_1 {
                di = *e.offset((i___1 * e_dim1) as isize);
                i__2_0 = *m_1;
                i__3_0 = (i___1 as ::core::ffi::c_int - 1 as ::core::ffi::c_int) as Integer;
                mi = (if i__2_0 <= i__3_0 {
                    i__2_0 as ::core::ffi::c_int
                } else {
                    i__3_0 as ::core::ffi::c_int
                }) as Integer;
                if mi >= 1 as ::core::ffi::c_int {
                    i__2_0 = mi;
                    k_1 = 1 as ::core::ffi::c_int as Integer;
                    while k_1 <= i__2_0 {
                        di -= (*e.offset((-k_1 + i___1 * e_dim1) as isize)
                            * *e.offset((k_1 + (i___1 - k_1) * e_dim1) as isize))
                            as ::core::ffi::c_double;
                        k_1 += 1;
                    }
                    *e.offset((i___1 * e_dim1) as isize) = di;
                }
                i__2_0 = *m_1;
                i__3_0 = *n_1 - i___1;
                lm = (if i__2_0 <= i__3_0 {
                    i__2_0 as ::core::ffi::c_int
                } else {
                    i__3_0 as ::core::ffi::c_int
                }) as Integer;
                if lm >= 1 as ::core::ffi::c_int {
                    i__2_0 = lm;
                    l_0 = 1 as ::core::ffi::c_int as Integer;
                    while l_0 <= i__2_0 {
                        dl = *e.offset((-l_0 + (i___1 + l_0) * e_dim1) as isize);
                        i__3_0 = *m_1 - l_0;
                        i__4_0 = (i___1 as ::core::ffi::c_int - 1 as ::core::ffi::c_int) as Integer;
                        km_0 = (if i__3_0 <= i__4_0 {
                            i__3_0 as ::core::ffi::c_int
                        } else {
                            i__4_0 as ::core::ffi::c_int
                        }) as Integer;
                        if km_0 >= 1 as ::core::ffi::c_int {
                            du = *e.offset((l_0 + i___1 * e_dim1) as isize);
                            i__3_0 = km_0;
                            k_1 = 1 as ::core::ffi::c_int as Integer;
                            while k_1 <= i__3_0 {
                                du -= (*e.offset((-k_1 + i___1 * e_dim1) as isize)
                                    * *e.offset((l_0 + k_1 + (i___1 - k_1) * e_dim1) as isize))
                                    as ::core::ffi::c_double;
                                dl -= (*e.offset((-l_0 - k_1 + (l_0 + i___1) * e_dim1) as isize)
                                    * *e.offset((k_1 + (i___1 - k_1) * e_dim1) as isize))
                                    as ::core::ffi::c_double;
                                k_1 += 1;
                            }
                            *e.offset((l_0 + i___1 * e_dim1) as isize) = du;
                        }
                        *e.offset((-l_0 + (i___1 + l_0) * e_dim1) as isize) = dl / di;
                        l_0 += 1;
                    }
                }
                i___1 += 1;
            }
            return 0 as ::core::ffi::c_int;
        }
        #[unsafe(export_name = "bansol_")]
        pub unsafe extern "C" fn bansol(
            mut e: *mut DoubleReal,
            mut y_1: *mut DoubleReal,
            mut ny_1: *mut Integer,
            mut c___1: *mut DoubleReal,
            mut nc_1: *mut Integer,
            mut m_1: *mut Integer,
            mut n_1: *mut Integer,
            mut k_1: *mut Integer,
        ) -> ::core::ffi::c_int {
            let mut e_dim1: Integer = 0;
            let mut e_offset: Integer = 0;
            let mut y_dim1_1: Integer = 0;
            let mut y_offset_1: Integer = 0;
            let mut c_dim1_1: Integer = 0;
            let mut c_offset_1: Integer = 0;
            let mut i__1_1: Integer = 0;
            let mut i__2_0: Integer = 0;
            let mut i__3_0: Integer = 0;
            let mut i__4_0: Integer = 0;
            static mut d__: DoubleReal = 0.;
            static mut i___1: Integer = 0;
            static mut j_1: Integer = 0;
            static mut l_0: Integer = 0;
            static mut mi: Integer = 0;
            static mut nm1_0: Integer = 0;
            e_dim1 = (*m_1 - -*m_1 + 1 as ::core::ffi::c_int) as Integer;
            e_offset = -*m_1 + e_dim1;
            e = e.offset(-(e_offset as isize));
            c_dim1_1 = *nc_1;
            c_offset_1 = 1 as Integer + c_dim1_1;
            c___1 = c___1.offset(-(c_offset_1 as isize));
            y_dim1_1 = *ny_1;
            y_offset_1 = 1 as Integer + y_dim1_1;
            y_1 = y_1.offset(-(y_offset_1 as isize));
            nm1_0 = (*n_1 - 1 as ::core::ffi::c_int) as Integer;
            i__1_1 = (*m_1 - 1 as ::core::ffi::c_int) as Integer;
            if i__1_1 < 0 as ::core::ffi::c_int {
                i__1_1 = *n_1;
                i___1 = 1 as ::core::ffi::c_int as Integer;
                while i___1 <= i__1_1 {
                    i__2_0 = *k_1;
                    j_1 = 1 as ::core::ffi::c_int as Integer;
                    while j_1 <= i__2_0 {
                        *c___1.offset((i___1 + j_1 * c_dim1_1) as isize) = *y_1
                            .offset((i___1 + j_1 * y_dim1_1) as isize)
                            / *e.offset((i___1 * e_dim1) as isize);
                        j_1 += 1;
                    }
                    i___1 += 1;
                }
                return 0 as ::core::ffi::c_int;
            } else if i__1_1 == 0 as ::core::ffi::c_int {
                i__1_1 = *k_1;
                j_1 = 1 as ::core::ffi::c_int as Integer;
                while j_1 <= i__1_1 {
                    *c___1.offset(
                        (j_1 as ::core::ffi::c_int * c_dim1_1 as ::core::ffi::c_int
                            + 1 as ::core::ffi::c_int) as isize,
                    ) = *y_1.offset(
                        (j_1 as ::core::ffi::c_int * y_dim1_1 as ::core::ffi::c_int
                            + 1 as ::core::ffi::c_int) as isize,
                    );
                    i__2_0 = *n_1;
                    i___1 = 2 as ::core::ffi::c_int as Integer;
                    while i___1 <= i__2_0 {
                        *c___1.offset((i___1 + j_1 * c_dim1_1) as isize) = *y_1
                            .offset((i___1 + j_1 * y_dim1_1) as isize)
                            - *e.offset(
                                (i___1 as ::core::ffi::c_int * e_dim1 as ::core::ffi::c_int
                                    - 1 as ::core::ffi::c_int)
                                    as isize,
                            ) * *c___1.offset((i___1 - 1 as Integer + j_1 * c_dim1_1) as isize);
                        i___1 += 1;
                    }
                    let ref mut fresh0 = *c___1.offset((*n_1 + j_1 * c_dim1_1) as isize);
                    *fresh0 /= *e.offset((*n_1 * e_dim1) as isize) as ::core::ffi::c_double;
                    i___1 = nm1_0;
                    while i___1 >= 1 as ::core::ffi::c_int {
                        *c___1.offset((i___1 + j_1 * c_dim1_1) as isize) = (*c___1
                            .offset((i___1 + j_1 * c_dim1_1) as isize)
                            - *e.offset(
                                (i___1 as ::core::ffi::c_int * e_dim1 as ::core::ffi::c_int
                                    + 1 as ::core::ffi::c_int)
                                    as isize,
                            ) * *c___1.offset((i___1 + 1 as Integer + j_1 * c_dim1_1) as isize))
                            / *e.offset((i___1 * e_dim1) as isize);
                        i___1 -= 1;
                    }
                    j_1 += 1;
                }
                return 0 as ::core::ffi::c_int;
            } else {
                i__1_1 = *k_1;
                j_1 = 1 as ::core::ffi::c_int as Integer;
                while j_1 <= i__1_1 {
                    *c___1.offset(
                        (j_1 as ::core::ffi::c_int * c_dim1_1 as ::core::ffi::c_int
                            + 1 as ::core::ffi::c_int) as isize,
                    ) = *y_1.offset(
                        (j_1 as ::core::ffi::c_int * y_dim1_1 as ::core::ffi::c_int
                            + 1 as ::core::ffi::c_int) as isize,
                    );
                    i__2_0 = *n_1;
                    i___1 = 2 as ::core::ffi::c_int as Integer;
                    while i___1 <= i__2_0 {
                        i__3_0 = *m_1;
                        i__4_0 = (i___1 as ::core::ffi::c_int - 1 as ::core::ffi::c_int) as Integer;
                        mi = (if i__3_0 <= i__4_0 {
                            i__3_0 as ::core::ffi::c_int
                        } else {
                            i__4_0 as ::core::ffi::c_int
                        }) as Integer;
                        d__ = *y_1.offset((i___1 + j_1 * y_dim1_1) as isize);
                        i__3_0 = mi;
                        l_0 = 1 as ::core::ffi::c_int as Integer;
                        while l_0 <= i__3_0 {
                            d__ -= (*e.offset((-l_0 + i___1 * e_dim1) as isize)
                                * *c___1.offset((i___1 - l_0 + j_1 * c_dim1_1) as isize))
                                as ::core::ffi::c_double;
                            l_0 += 1;
                        }
                        *c___1.offset((i___1 + j_1 * c_dim1_1) as isize) = d__;
                        i___1 += 1;
                    }
                    let ref mut fresh1 = *c___1.offset((*n_1 + j_1 * c_dim1_1) as isize);
                    *fresh1 /= *e.offset((*n_1 * e_dim1) as isize) as ::core::ffi::c_double;
                    i___1 = nm1_0;
                    while i___1 >= 1 as ::core::ffi::c_int {
                        i__2_0 = *m_1;
                        i__3_0 = *n_1 - i___1;
                        mi = (if i__2_0 <= i__3_0 {
                            i__2_0 as ::core::ffi::c_int
                        } else {
                            i__3_0 as ::core::ffi::c_int
                        }) as Integer;
                        d__ = *c___1.offset((i___1 + j_1 * c_dim1_1) as isize);
                        i__2_0 = mi;
                        l_0 = 1 as ::core::ffi::c_int as Integer;
                        while l_0 <= i__2_0 {
                            d__ -= (*e.offset((l_0 + i___1 * e_dim1) as isize)
                                * *c___1.offset((i___1 + l_0 + j_1 * c_dim1_1) as isize))
                                as ::core::ffi::c_double;
                            l_0 += 1;
                        }
                        *c___1.offset((i___1 + j_1 * c_dim1_1) as isize) =
                            d__ / *e.offset((i___1 * e_dim1) as isize);
                        i___1 -= 1;
                    }
                    j_1 += 1;
                }
                return 0 as ::core::ffi::c_int;
            };
        }
        bwe_dim1 = (*m_0 - -*m_0 + 1 as ::core::ffi::c_int) as Integer;
        bwe_offset = -*m_0 + bwe_dim1;
        bwe = bwe.offset(-(bwe_offset as isize));
        we_dim1 = (*m_0 - -*m_0 + 1 as ::core::ffi::c_int) as Integer;
        we_offset = -*m_0 + we_dim1;
        we = we.offset(-(we_offset as isize));
        b_dim1 = (*m_0 - 1 as ::core::ffi::c_int - (1 as ::core::ffi::c_int - *m_0)
            + 1 as ::core::ffi::c_int) as Integer;
        b_offset = 1 as Integer - *m_0 + b_dim1;
        b = b.offset(-(b_offset as isize));
        wx_0 = wx_0.offset(-1);
        wy_0 = wy_0.offset(-1);
        y_dim1_0 = *ny_0;
        y_offset_0 = 1 as Integer + y_dim1_0;
        y_0 = y_0.offset(-(y_offset_0 as isize));
        c_dim1_0 = *nc_0;
        c_offset_0 = 1 as Integer + c_dim1_0;
        c___0 = c___0.offset(-(c_offset_0 as isize));
        stat = stat.offset(-1);
        dp = *p;
        *stat.offset(4 as ::core::ffi::c_int as isize) = *p;
        pel = *p * *el_0;
        if pel < *eps {
            dp = *eps / *el_0;
            *stat.offset(4 as ::core::ffi::c_int as isize) = 0.0f64 as DoubleReal;
        }
        if pel * *eps > 1.0f64 {
            dp = 1.0f64 / (*el_0 * *eps);
            *stat.offset(4 as ::core::ffi::c_int as isize) = dp;
        }
        i__1_0 = *n_0;
        i___0 = 1 as ::core::ffi::c_int as Integer;
        while i___0 <= i__1_0 {
            i__2 = *m_0;
            i__3 = (i___0 as ::core::ffi::c_int - 1 as ::core::ffi::c_int) as Integer;
            km = -if i__2 <= i__3 {
                i__2 as ::core::ffi::c_int
            } else {
                i__3 as ::core::ffi::c_int
            } as Integer;
            i__2 = *m_0;
            i__3 = *n_0 - i___0;
            kp = (if i__2 <= i__3 {
                i__2 as ::core::ffi::c_int
            } else {
                i__3 as ::core::ffi::c_int
            }) as Integer;
            i__2 = kp;
            l = km;
            while l <= i__2 {
                if (if l >= 0 as ::core::ffi::c_int {
                    l as ::core::ffi::c_int
                } else {
                    -(l as ::core::ffi::c_int)
                }) == *m_0
                {
                    *bwe.offset((l + i___0 * bwe_dim1) as isize) =
                        dp * *we.offset((l + i___0 * we_dim1) as isize);
                } else {
                    *bwe.offset((l + i___0 * bwe_dim1) as isize) = *b
                        .offset((l + i___0 * b_dim1) as isize)
                        + dp * *we.offset((l + i___0 * we_dim1) as isize);
                }
                l += 1;
            }
            i___0 += 1;
        }
        bandet(bwe.offset(bwe_offset as isize) as *mut DoubleReal, m_0, n_0);
        bansol(
            bwe.offset(bwe_offset as isize) as *mut DoubleReal,
            y_0.offset(y_offset_0 as isize) as *mut DoubleReal,
            ny_0,
            c___0.offset(c_offset_0 as isize) as *mut DoubleReal,
            nc_0,
            m_0,
            n_0,
            k_0,
        );
        *stat.offset(3 as ::core::ffi::c_int as isize) = trinv(
            we.offset(we_offset as isize) as *mut DoubleReal,
            bwe.offset(bwe_offset as isize) as *mut DoubleReal,
            m_0,
            n_0,
        ) * dp;
        trn = (*stat.offset(3 as ::core::ffi::c_int as isize) as ::core::ffi::c_double
            / *n_0 as ::core::ffi::c_double) as DoubleReal;
        esn = 0.0f64 as DoubleReal;
        i__1_0 = *k_0;
        j_0 = 1 as ::core::ffi::c_int as Integer;
        while j_0 <= i__1_0 {
            i__2 = *n_0;
            i___0 = 1 as ::core::ffi::c_int as Integer;
            while i___0 <= i__2 {
                dt = -*y_0.offset((i___0 + j_0 * y_dim1_0) as isize);
                i__3 = (*m_0 - 1 as ::core::ffi::c_int) as Integer;
                i__4 = (i___0 as ::core::ffi::c_int - 1 as ::core::ffi::c_int) as Integer;
                km = -if i__3 <= i__4 {
                    i__3 as ::core::ffi::c_int
                } else {
                    i__4 as ::core::ffi::c_int
                } as Integer;
                i__3 = (*m_0 - 1 as ::core::ffi::c_int) as Integer;
                i__4 = *n_0 - i___0;
                kp = (if i__3 <= i__4 {
                    i__3 as ::core::ffi::c_int
                } else {
                    i__4 as ::core::ffi::c_int
                }) as Integer;
                i__3 = kp;
                l = km;
                while l <= i__3 {
                    dt += (*b.offset((l + i___0 * b_dim1) as isize)
                        * *c___0.offset((i___0 + l + j_0 * c_dim1_0) as isize))
                        as ::core::ffi::c_double;
                    l += 1;
                }
                esn += (dt * dt * *wx_0.offset(i___0 as isize) * *wy_0.offset(j_0 as isize))
                    as ::core::ffi::c_double;
                i___0 += 1;
            }
            j_0 += 1;
        }
        esn /= (*n_0 * *k_0) as ::core::ffi::c_double;
        *stat.offset(6 as ::core::ffi::c_int as isize) = esn / trn;
        *stat.offset(1 as ::core::ffi::c_int as isize) =
            *stat.offset(6 as ::core::ffi::c_int as isize) / trn;
        *stat.offset(2 as ::core::ffi::c_int as isize) = esn;
        if (if *mode >= 0 as ::core::ffi::c_int {
            *mode
        } else {
            -*mode
        }) != 3 as ::core::ffi::c_int
        {
            *stat.offset(5 as ::core::ffi::c_int as isize) =
                *stat.offset(6 as ::core::ffi::c_int as isize) - esn;
            if (if *mode >= 0 as ::core::ffi::c_int {
                *mode
            } else {
                -*mode
            }) == 1 as ::core::ffi::c_int
            {
                ret_val = 0.0f64 as DoubleReal;
            }
            if (if *mode >= 0 as ::core::ffi::c_int {
                *mode
            } else {
                -*mode
            }) == 2 as ::core::ffi::c_int
            {
                ret_val = *stat.offset(1 as ::core::ffi::c_int as isize);
            }
            if (if *mode >= 0 as ::core::ffi::c_int {
                *mode
            } else {
                -*mode
            }) == 4 as ::core::ffi::c_int
            {
                d__1 = *stat.offset(3 as ::core::ffi::c_int as isize) - *val_0;
                ret_val = (if d__1 >= 0 as ::core::ffi::c_int as ::core::ffi::c_double {
                    d__1 as ::core::ffi::c_double
                } else {
                    -(d__1 as ::core::ffi::c_double)
                }) as DoubleReal;
            }
        } else {
            *stat.offset(5 as ::core::ffi::c_int as isize) = (esn as ::core::ffi::c_double
                - *val_0 * (trn as ::core::ffi::c_double * 2.0f64 - 1.0f64))
                as DoubleReal;
            ret_val = *stat.offset(5 as ::core::ffi::c_int as isize);
        }
        return ret_val;
    }
    #[unsafe(export_name = "prep_")]
    pub unsafe extern "C" fn prep(
        mut m_0: *mut Integer,
        mut n_0: *mut Integer,
        mut x_0: *mut DoubleReal,
        mut w: *mut DoubleReal,
        mut we: *mut DoubleReal,
        mut el_0: *mut DoubleReal,
    ) -> ::core::ffi::c_int {
        let mut i__1_0: Integer = 0;
        let mut i__2: Integer = 0;
        let mut i__3: Integer = 0;
        let mut d__1: DoubleReal = 0.;
        static mut f: DoubleReal = 0.;
        static mut i___0: Integer = 0;
        static mut j_0: Integer = 0;
        static mut k_0: Integer = 0;
        static mut l: Integer = 0;
        static mut y_0: DoubleReal = 0.;
        static mut f1: DoubleReal = 0.;
        static mut i1: Integer = 0;
        static mut i2: Integer = 0;
        static mut m2_0: Integer = 0;
        static mut ff: DoubleReal = 0.;
        static mut jj: Integer = 0;
        static mut jm: Integer = 0;
        static mut kl: Integer = 0;
        static mut nm: Integer = 0;
        static mut ku: Integer = 0;
        static mut wi: DoubleReal = 0.;
        static mut n2m: Integer = 0;
        static mut mp1: Integer = 0;
        static mut i2m1: Integer = 0;
        static mut inc: Integer = 0;
        static mut i1p1: Integer = 0;
        static mut m2m1: Integer = 0;
        static mut m2p1: Integer = 0;
        we = we.offset(-1);
        w = w.offset(-1);
        x_0 = x_0.offset(-1);
        m2_0 = *m_0 << 1 as ::core::ffi::c_int;
        mp1 = (*m_0 + 1 as ::core::ffi::c_int) as Integer;
        m2m1 = (m2_0 as ::core::ffi::c_int - 1 as ::core::ffi::c_int) as Integer;
        m2p1 = (m2_0 as ::core::ffi::c_int + 1 as ::core::ffi::c_int) as Integer;
        nm = *n_0 - *m_0;
        f1 = -1.0f64 as DoubleReal;
        if *m_0 != 1 as ::core::ffi::c_int {
            i__1_0 = *m_0;
            i___0 = 2 as ::core::ffi::c_int as Integer;
            while i___0 <= i__1_0 {
                f1 =
                    (-(f1 as ::core::ffi::c_double) * i___0 as ::core::ffi::c_double) as DoubleReal;
                i___0 += 1;
            }
            i__1_0 = m2m1;
            i___0 = mp1;
            while i___0 <= i__1_0 {
                f1 *= i___0 as ::core::ffi::c_double;
                i___0 += 1;
            }
        }
        i1 = 1 as ::core::ffi::c_int as Integer;
        i2 = *m_0;
        jm = mp1;
        i__1_0 = *n_0;
        j_0 = 1 as ::core::ffi::c_int as Integer;
        while j_0 <= i__1_0 {
            inc = m2p1;
            if j_0 > nm {
                f1 = -f1;
                f = f1;
            } else if j_0 < mp1 {
                inc = 1 as ::core::ffi::c_int as Integer;
                f = f1;
            } else {
                f = f1 * (*x_0.offset((j_0 + *m_0) as isize) - *x_0.offset((j_0 - *m_0) as isize));
            }
            if j_0 > mp1 {
                i1 += 1;
            }
            if i2 < *n_0 {
                i2 += 1;
            }
            jj = jm;
            ff = f;
            y_0 = *x_0.offset(i1 as isize);
            i1p1 = (i1 as ::core::ffi::c_int + 1 as ::core::ffi::c_int) as Integer;
            i__2 = i2;
            i___0 = i1p1;
            while i___0 <= i__2 {
                ff /= (y_0 - *x_0.offset(i___0 as isize)) as ::core::ffi::c_double;
                i___0 += 1;
            }
            *we.offset(jj as isize) = ff;
            jj += m2_0 as ::core::ffi::c_int;
            i2m1 = (i2 as ::core::ffi::c_int - 1 as ::core::ffi::c_int) as Integer;
            if i1p1 <= i2m1 {
                i__2 = i2m1;
                l = i1p1;
                while l <= i__2 {
                    ff = f;
                    y_0 = *x_0.offset(l as isize);
                    i__3 = (l as ::core::ffi::c_int - 1 as ::core::ffi::c_int) as Integer;
                    i___0 = i1;
                    while i___0 <= i__3 {
                        ff /= (y_0 - *x_0.offset(i___0 as isize)) as ::core::ffi::c_double;
                        i___0 += 1;
                    }
                    i__3 = i2;
                    i___0 = (l as ::core::ffi::c_int + 1 as ::core::ffi::c_int) as Integer;
                    while i___0 <= i__3 {
                        ff /= (y_0 - *x_0.offset(i___0 as isize)) as ::core::ffi::c_double;
                        i___0 += 1;
                    }
                    *we.offset(jj as isize) = ff;
                    jj += m2_0 as ::core::ffi::c_int;
                    l += 1;
                }
            }
            ff = f;
            y_0 = *x_0.offset(i2 as isize);
            i__2 = i2m1;
            i___0 = i1;
            while i___0 <= i__2 {
                ff /= (y_0 - *x_0.offset(i___0 as isize)) as ::core::ffi::c_double;
                i___0 += 1;
            }
            *we.offset(jj as isize) = ff;
            jj += m2_0 as ::core::ffi::c_int;
            jm += inc as ::core::ffi::c_int;
            j_0 += 1;
        }
        kl = 1 as ::core::ffi::c_int as Integer;
        n2m = (m2p1 as ::core::ffi::c_int * *n_0 + 1 as ::core::ffi::c_int) as Integer;
        i__1_0 = *m_0;
        i___0 = 1 as ::core::ffi::c_int as Integer;
        while i___0 <= i__1_0 {
            ku = kl + *m_0 - i___0;
            i__2 = ku;
            k_0 = kl;
            while k_0 <= i__2 {
                *we.offset(k_0 as isize) = 0.0f64 as DoubleReal;
                *we.offset((n2m - k_0) as isize) = 0.0f64 as DoubleReal;
                k_0 += 1;
            }
            kl += m2p1 as ::core::ffi::c_int;
            i___0 += 1;
        }
        jj = 0 as ::core::ffi::c_int as Integer;
        *el_0 = 0.0f64 as DoubleReal;
        i__1_0 = *n_0;
        i___0 = 1 as ::core::ffi::c_int as Integer;
        while i___0 <= i__1_0 {
            wi = *w.offset(i___0 as isize);
            i__2 = m2p1;
            j_0 = 1 as ::core::ffi::c_int as Integer;
            while j_0 <= i__2 {
                jj += 1;
                let ref mut fresh2 = *we.offset(jj as isize);
                *fresh2 /= wi as ::core::ffi::c_double;
                d__1 = *we.offset(jj as isize);
                *el_0 += (if d__1 >= 0 as ::core::ffi::c_int as ::core::ffi::c_double {
                    d__1 as ::core::ffi::c_double
                } else {
                    -(d__1 as ::core::ffi::c_double)
                });
                j_0 += 1;
            }
            i___0 += 1;
        }
        *el_0 /= *n_0 as ::core::ffi::c_double;
        return 0 as ::core::ffi::c_int;
    }
    static mut alpha: DoubleReal = 0.;
    #[unsafe(export_name = "basis_")]
    pub unsafe extern "C" fn basis(
        mut m_0: *mut Integer,
        mut n_0: *mut Integer,
        mut x_0: *mut DoubleReal,
        mut b: *mut DoubleReal,
        mut bl: *mut DoubleReal,
        mut q: *mut DoubleReal,
    ) -> ::core::ffi::c_int {
        let mut b_dim1: Integer = 0;
        let mut b_offset: Integer = 0;
        let mut q_offset: Integer = 0;
        let mut i__1_0: Integer = 0;
        let mut i__2: Integer = 0;
        let mut i__3: Integer = 0;
        let mut i__4: Integer = 0;
        let mut d__1: DoubleReal = 0.;
        static mut i___0: Integer = 0;
        static mut j_0: Integer = 0;
        static mut k_0: Integer = 0;
        static mut l: Integer = 0;
        static mut u: DoubleReal = 0.;
        static mut v: DoubleReal = 0.;
        static mut y_0: DoubleReal = 0.;
        static mut j1: Integer = 0;
        static mut j2: Integer = 0;
        static mut m2_0: Integer = 0;
        static mut ir: Integer = 0;
        static mut mm1: Integer = 0;
        static mut mp1: Integer = 0;
        static mut arg: DoubleReal = 0.;
        static mut nmip1: Integer = 0;
        q_offset = 1 as Integer - *m_0;
        q = q.offset(-(q_offset as isize));
        b_dim1 = (*m_0 - 1 as ::core::ffi::c_int - (1 as ::core::ffi::c_int - *m_0)
            + 1 as ::core::ffi::c_int) as Integer;
        b_offset = 1 as Integer - *m_0 + b_dim1;
        b = b.offset(-(b_offset as isize));
        x_0 = x_0.offset(-1);
        if *m_0 == 1 as ::core::ffi::c_int {
            i__1_0 = *n_0;
            i___0 = 1 as ::core::ffi::c_int as Integer;
            while i___0 <= i__1_0 {
                *b.offset((i___0 * b_dim1) as isize) = 1.0f64 as DoubleReal;
                i___0 += 1;
            }
            *bl = 1.0f64 as DoubleReal;
            return 0 as ::core::ffi::c_int;
        }
        mm1 = (*m_0 - 1 as ::core::ffi::c_int) as Integer;
        mp1 = (*m_0 + 1 as ::core::ffi::c_int) as Integer;
        m2_0 = *m_0 << 1 as ::core::ffi::c_int;
        i__1_0 = *n_0;
        l = 1 as ::core::ffi::c_int as Integer;
        while l <= i__1_0 {
            i__2 = *m_0;
            j_0 = -mm1;
            while j_0 <= i__2 {
                *q.offset(j_0 as isize) = 0.0f64 as DoubleReal;
                j_0 += 1;
            }
            *q.offset(mm1 as isize) = 1.0f64 as DoubleReal;
            if l != 1 as ::core::ffi::c_int && l != *n_0 {
                *q.offset(mm1 as isize) = 1.0f64
                    / (*x_0.offset((l as ::core::ffi::c_int + 1 as ::core::ffi::c_int) as isize)
                        - *x_0
                            .offset((l as ::core::ffi::c_int - 1 as ::core::ffi::c_int) as isize));
            }
            arg = *x_0.offset(l as isize);
            i__2 = m2_0;
            i___0 = 3 as ::core::ffi::c_int as Integer;
            while i___0 <= i__2 {
                ir = mp1 - i___0;
                v = *q.offset(ir as isize);
                if l < i___0 {
                    i__3 = i___0;
                    j_0 = (l as ::core::ffi::c_int + 1 as ::core::ffi::c_int) as Integer;
                    while j_0 <= i__3 {
                        u = v;
                        v = *q
                            .offset((ir as ::core::ffi::c_int + 1 as ::core::ffi::c_int) as isize);
                        *q.offset(ir as isize) = u + (*x_0.offset(j_0 as isize) - arg) * v;
                        ir += 1;
                        j_0 += 1;
                    }
                }
                i__3 = (l as ::core::ffi::c_int - i___0 as ::core::ffi::c_int
                    + 1 as ::core::ffi::c_int) as Integer;
                j1 = (if i__3 >= 1 as ::core::ffi::c_int {
                    i__3 as ::core::ffi::c_int
                } else {
                    1 as ::core::ffi::c_int
                }) as Integer;
                i__3 = (l as ::core::ffi::c_int - 1 as ::core::ffi::c_int) as Integer;
                i__4 = *n_0 - i___0;
                j2 = (if i__3 <= i__4 {
                    i__3 as ::core::ffi::c_int
                } else {
                    i__4 as ::core::ffi::c_int
                }) as Integer;
                if j1 <= j2 {
                    if i___0 < m2_0 {
                        i__3 = j2;
                        j_0 = j1;
                        while j_0 <= i__3 {
                            y_0 = *x_0.offset((i___0 + j_0) as isize);
                            u = v;
                            v = *q.offset(
                                (ir as ::core::ffi::c_int + 1 as ::core::ffi::c_int) as isize,
                            );
                            *q.offset(ir as isize) =
                                u + (v - u) * (y_0 - arg) / (y_0 - *x_0.offset(j_0 as isize));
                            ir += 1;
                            j_0 += 1;
                        }
                    } else {
                        i__3 = j2;
                        j_0 = j1;
                        while j_0 <= i__3 {
                            u = v;
                            v = *q.offset(
                                (ir as ::core::ffi::c_int + 1 as ::core::ffi::c_int) as isize,
                            );
                            *q.offset(ir as isize) = (arg - *x_0.offset(j_0 as isize)) * u
                                + (*x_0.offset((i___0 + j_0) as isize) - arg) * v;
                            ir += 1;
                            j_0 += 1;
                        }
                    }
                }
                nmip1 = (*n_0 - i___0 as ::core::ffi::c_int + 1 as ::core::ffi::c_int) as Integer;
                if nmip1 < l {
                    i__3 = (l as ::core::ffi::c_int - 1 as ::core::ffi::c_int) as Integer;
                    j_0 = nmip1;
                    while j_0 <= i__3 {
                        u = v;
                        v = *q
                            .offset((ir as ::core::ffi::c_int + 1 as ::core::ffi::c_int) as isize);
                        *q.offset(ir as isize) = (arg - *x_0.offset(j_0 as isize)) * u + v;
                        ir += 1;
                        j_0 += 1;
                    }
                }
                i___0 += 1;
            }
            i__2 = mm1;
            j_0 = -mm1;
            while j_0 <= i__2 {
                *b.offset((j_0 + l * b_dim1) as isize) = *q.offset(j_0 as isize);
                j_0 += 1;
            }
            l += 1;
        }
        i__1_0 = mm1;
        i___0 = 1 as ::core::ffi::c_int as Integer;
        while i___0 <= i__1_0 {
            i__2 = mm1;
            k_0 = i___0;
            while k_0 <= i__2 {
                *b.offset((-k_0 + i___0 * b_dim1) as isize) = 0.0f64 as DoubleReal;
                *b.offset((k_0 + (*n_0 + 1 as Integer - i___0) * b_dim1) as isize) =
                    0.0f64 as DoubleReal;
                k_0 += 1;
            }
            i___0 += 1;
        }
        *bl = 0.0f64 as DoubleReal;
        i__1_0 = *n_0;
        i___0 = 1 as ::core::ffi::c_int as Integer;
        while i___0 <= i__1_0 {
            i__2 = mm1;
            k_0 = -mm1;
            while k_0 <= i__2 {
                d__1 = *b.offset((k_0 + i___0 * b_dim1) as isize);
                *bl += (if d__1 >= 0 as ::core::ffi::c_int as ::core::ffi::c_double {
                    d__1 as ::core::ffi::c_double
                } else {
                    -(d__1 as ::core::ffi::c_double)
                });
                k_0 += 1;
            }
            i___0 += 1;
        }
        *bl /= *n_0 as ::core::ffi::c_double;
        return 0 as ::core::ffi::c_int;
    }
    wk = wk.offset(-1);
    wx = wx.offset(-1);
    x = x.offset(-1);
    wy = wy.offset(-1);
    y_dim1 = *ny;
    y_offset = 1 as Integer + y_dim1;
    y = y.offset(-(y_offset as isize));
    c_dim1 = *nc;
    c_offset = 1 as Integer + c_dim1;
    c__ = c__.offset(-(c_offset as isize));
    *ier = 0 as ::core::ffi::c_int as Integer;
    if (if *md >= 0 as ::core::ffi::c_int {
        *md
    } else {
        -*md
    }) > 4 as ::core::ffi::c_int
        || *md == 0 as ::core::ffi::c_int
        || (if *md >= 0 as ::core::ffi::c_int {
            *md
        } else {
            -*md
        }) == 1 as ::core::ffi::c_int
            && *val < 0.0f64
        || (if *md >= 0 as ::core::ffi::c_int {
            *md
        } else {
            -*md
        }) == 3 as ::core::ffi::c_int
            && *val < 0.0f64
        || (if *md >= 0 as ::core::ffi::c_int {
            *md
        } else {
            -*md
        }) == 4 as ::core::ffi::c_int
            && (*val < 0.0f64 || *val > (*n - *m) as DoubleReal)
    {
        *ier = 3 as ::core::ffi::c_int as Integer;
        return 0 as ::core::ffi::c_int;
    }
    if *md > 0 as ::core::ffi::c_int {
        m2 = *m << 1 as ::core::ffi::c_int;
        nm1 = (*n - 1 as ::core::ffi::c_int) as Integer;
    } else if m2 != *m << 1 as ::core::ffi::c_int || nm1 != *n - 1 as ::core::ffi::c_int {
        *ier = 3 as ::core::ffi::c_int as Integer;
        return 0 as ::core::ffi::c_int;
    }
    if *m <= 0 as ::core::ffi::c_int || *n < m2 {
        *ier = 1 as ::core::ffi::c_int as Integer;
        return 0 as ::core::ffi::c_int;
    }
    if *wx.offset(1 as ::core::ffi::c_int as isize) <= 0.0f64 {
        *ier = 2 as ::core::ffi::c_int as Integer;
    }
    i__1 = *n;
    i__ = 2 as ::core::ffi::c_int as Integer;
    while i__ <= i__1 {
        if *wx.offset(i__ as isize) <= 0.0f64
            || *x.offset((i__ as ::core::ffi::c_int - 1 as ::core::ffi::c_int) as isize)
                >= *x.offset(i__ as isize)
        {
            *ier = 2 as ::core::ffi::c_int as Integer;
        }
        if *ier != 0 as ::core::ffi::c_int {
            return 0 as ::core::ffi::c_int;
        }
        i__ += 1;
    }
    i__1 = *k;
    j = 1 as ::core::ffi::c_int as Integer;
    while j <= i__1 {
        if *wy.offset(j as isize) <= 0.0f64 {
            *ier = 2 as ::core::ffi::c_int as Integer;
        }
        if *ier != 0 as ::core::ffi::c_int {
            return 0 as ::core::ffi::c_int;
        }
        j += 1;
    }
    nm2p1 = (*n * (m2 as ::core::ffi::c_int + 1 as ::core::ffi::c_int)) as Integer;
    nm2m1 = (*n * (m2 as ::core::ffi::c_int - 1 as ::core::ffi::c_int)) as Integer;
    ib = (nm2p1 as ::core::ffi::c_int + 7 as ::core::ffi::c_int) as Integer;
    iwe = ib + nm2m1;
    if *md > 0 as ::core::ffi::c_int {
        basis(
            m,
            n,
            x.offset(1 as ::core::ffi::c_int as isize) as *mut DoubleReal,
            wk.offset(ib as isize) as *mut DoubleReal,
            &raw mut r1,
            wk.offset(7 as ::core::ffi::c_int as isize) as *mut DoubleReal,
        );
        prep(
            m,
            n,
            x.offset(1 as ::core::ffi::c_int as isize) as *mut DoubleReal,
            wx.offset(1 as ::core::ffi::c_int as isize) as *mut DoubleReal,
            wk.offset(iwe as isize) as *mut DoubleReal,
            &raw mut el,
        );
        el /= r1 as ::core::ffi::c_double;
    }
    if (if *md >= 0 as ::core::ffi::c_int {
        *md
    } else {
        -*md
    }) != 1 as ::core::ffi::c_int
    {
        if *md < -(1 as ::core::ffi::c_int) {
            r1 = *wk.offset(4 as ::core::ffi::c_int as isize);
        } else {
            r1 = 1.0f64 / el;
        }
        r2 = (r1 as ::core::ffi::c_double * 2.0f64) as DoubleReal;
        gf2 = splc(
            m,
            n,
            k,
            y.offset(y_offset as isize) as *mut DoubleReal,
            ny,
            wx.offset(1 as ::core::ffi::c_int as isize) as *mut DoubleReal,
            wy.offset(1 as ::core::ffi::c_int as isize) as *mut DoubleReal,
            md,
            val,
            &raw mut r2,
            &raw mut c_b6,
            c__.offset(c_offset as isize) as *mut DoubleReal,
            nc,
            wk.offset(1 as ::core::ffi::c_int as isize) as *mut DoubleReal,
            wk.offset(ib as isize) as *mut DoubleReal,
            wk.offset(iwe as isize) as *mut DoubleReal,
            &raw mut el,
            wk.offset(7 as ::core::ffi::c_int as isize) as *mut DoubleReal,
        );
        loop {
            gf1 = splc(
                m,
                n,
                k,
                y.offset(y_offset as isize) as *mut DoubleReal,
                ny,
                wx.offset(1 as ::core::ffi::c_int as isize) as *mut DoubleReal,
                wy.offset(1 as ::core::ffi::c_int as isize) as *mut DoubleReal,
                md,
                val,
                &raw mut r1,
                &raw mut c_b6,
                c__.offset(c_offset as isize) as *mut DoubleReal,
                nc,
                wk.offset(1 as ::core::ffi::c_int as isize) as *mut DoubleReal,
                wk.offset(ib as isize) as *mut DoubleReal,
                wk.offset(iwe as isize) as *mut DoubleReal,
                &raw mut el,
                wk.offset(7 as ::core::ffi::c_int as isize) as *mut DoubleReal,
            );
            if gf1 > gf2 {
                r3 = (r2 as ::core::ffi::c_double * 2.0f64) as DoubleReal;
                current_block = 8394302012487765337;
                break;
            } else {
                if *wk.offset(4 as ::core::ffi::c_int as isize) <= 0.0f64 {
                    current_block = 3190149595005962170;
                    break;
                }
                r2 = r1;
                gf2 = gf1;
                r1 /= 2.0f64;
            }
        }
        match current_block {
            3190149595005962170 => {}
            _ => {
                loop {
                    gf3 = splc(
                        m,
                        n,
                        k,
                        y.offset(y_offset as isize) as *mut DoubleReal,
                        ny,
                        wx.offset(1 as ::core::ffi::c_int as isize) as *mut DoubleReal,
                        wy.offset(1 as ::core::ffi::c_int as isize) as *mut DoubleReal,
                        md,
                        val,
                        &raw mut r3,
                        &raw mut c_b6,
                        c__.offset(c_offset as isize) as *mut DoubleReal,
                        nc,
                        wk.offset(1 as ::core::ffi::c_int as isize) as *mut DoubleReal,
                        wk.offset(ib as isize) as *mut DoubleReal,
                        wk.offset(iwe as isize) as *mut DoubleReal,
                        &raw mut el,
                        wk.offset(7 as ::core::ffi::c_int as isize) as *mut DoubleReal,
                    );
                    if gf3 > gf2 {
                        r2 = r3;
                        gf2 = gf3;
                        alpha = ((r2 as ::core::ffi::c_double - r1 as ::core::ffi::c_double)
                            / 1.618033983f64) as DoubleReal;
                        r4 = r1 + alpha;
                        r3 = r2 - alpha;
                        gf3 = splc(
                            m,
                            n,
                            k,
                            y.offset(y_offset as isize) as *mut DoubleReal,
                            ny,
                            wx.offset(1 as ::core::ffi::c_int as isize) as *mut DoubleReal,
                            wy.offset(1 as ::core::ffi::c_int as isize) as *mut DoubleReal,
                            md,
                            val,
                            &raw mut r3,
                            &raw mut c_b6,
                            c__.offset(c_offset as isize) as *mut DoubleReal,
                            nc,
                            wk.offset(1 as ::core::ffi::c_int as isize) as *mut DoubleReal,
                            wk.offset(ib as isize) as *mut DoubleReal,
                            wk.offset(iwe as isize) as *mut DoubleReal,
                            &raw mut el,
                            wk.offset(7 as ::core::ffi::c_int as isize) as *mut DoubleReal,
                        );
                        gf4 = splc(
                            m,
                            n,
                            k,
                            y.offset(y_offset as isize) as *mut DoubleReal,
                            ny,
                            wx.offset(1 as ::core::ffi::c_int as isize) as *mut DoubleReal,
                            wy.offset(1 as ::core::ffi::c_int as isize) as *mut DoubleReal,
                            md,
                            val,
                            &raw mut r4,
                            &raw mut c_b6,
                            c__.offset(c_offset as isize) as *mut DoubleReal,
                            nc,
                            wk.offset(1 as ::core::ffi::c_int as isize) as *mut DoubleReal,
                            wk.offset(ib as isize) as *mut DoubleReal,
                            wk.offset(iwe as isize) as *mut DoubleReal,
                            &raw mut el,
                            wk.offset(7 as ::core::ffi::c_int as isize) as *mut DoubleReal,
                        );
                        current_block = 5432794640721371522;
                        break;
                    } else {
                        if *wk.offset(4 as ::core::ffi::c_int as isize) >= 999999999999999.88f64 {
                            current_block = 3190149595005962170;
                            break;
                        }
                        r2 = r3;
                        gf2 = gf3;
                        r3 *= 2.0f64;
                    }
                }
                match current_block {
                    3190149595005962170 => {}
                    _ => {
                        loop {
                            if gf3 <= gf4 {
                                r2 = r4;
                                gf2 = gf4;
                                err = (r2 - r1) / (r1 + r2);
                                if err as ::core::ffi::c_double * err as ::core::ffi::c_double
                                    + 1.0f64
                                    == 1.0f64
                                    || err <= 1e-6f64
                                {
                                    break;
                                }
                                r4 = r3;
                                gf4 = gf3;
                                alpha /= 1.618033983f64;
                                r3 = r2 - alpha;
                                gf3 = splc(
                                    m,
                                    n,
                                    k,
                                    y.offset(y_offset as isize) as *mut DoubleReal,
                                    ny,
                                    wx.offset(1 as ::core::ffi::c_int as isize) as *mut DoubleReal,
                                    wy.offset(1 as ::core::ffi::c_int as isize) as *mut DoubleReal,
                                    md,
                                    val,
                                    &raw mut r3,
                                    &raw mut c_b6,
                                    c__.offset(c_offset as isize) as *mut DoubleReal,
                                    nc,
                                    wk.offset(1 as ::core::ffi::c_int as isize) as *mut DoubleReal,
                                    wk.offset(ib as isize) as *mut DoubleReal,
                                    wk.offset(iwe as isize) as *mut DoubleReal,
                                    &raw mut el,
                                    wk.offset(7 as ::core::ffi::c_int as isize) as *mut DoubleReal,
                                );
                            } else {
                                r1 = r3;
                                gf1 = gf3;
                                err = (r2 - r1) / (r1 + r2);
                                if err as ::core::ffi::c_double * err as ::core::ffi::c_double
                                    + 1.0f64
                                    == 1.0f64
                                    || err <= 1e-6f64
                                {
                                    break;
                                }
                                r3 = r4;
                                gf3 = gf4;
                                alpha /= 1.618033983f64;
                                r4 = r1 + alpha;
                                gf4 = splc(
                                    m,
                                    n,
                                    k,
                                    y.offset(y_offset as isize) as *mut DoubleReal,
                                    ny,
                                    wx.offset(1 as ::core::ffi::c_int as isize) as *mut DoubleReal,
                                    wy.offset(1 as ::core::ffi::c_int as isize) as *mut DoubleReal,
                                    md,
                                    val,
                                    &raw mut r4,
                                    &raw mut c_b6,
                                    c__.offset(c_offset as isize) as *mut DoubleReal,
                                    nc,
                                    wk.offset(1 as ::core::ffi::c_int as isize) as *mut DoubleReal,
                                    wk.offset(ib as isize) as *mut DoubleReal,
                                    wk.offset(iwe as isize) as *mut DoubleReal,
                                    &raw mut el,
                                    wk.offset(7 as ::core::ffi::c_int as isize) as *mut DoubleReal,
                                );
                            }
                        }
                        r1 = ((r1 as ::core::ffi::c_double + r2 as ::core::ffi::c_double) * 0.5f64)
                            as DoubleReal;
                    }
                }
            }
        }
    } else {
        r1 = *val;
    }
    gf1 = splc(
        m,
        n,
        k,
        y.offset(y_offset as isize) as *mut DoubleReal,
        ny,
        wx.offset(1 as ::core::ffi::c_int as isize) as *mut DoubleReal,
        wy.offset(1 as ::core::ffi::c_int as isize) as *mut DoubleReal,
        md,
        val,
        &raw mut r1,
        &raw mut c_b6,
        c__.offset(c_offset as isize) as *mut DoubleReal,
        nc,
        wk.offset(1 as ::core::ffi::c_int as isize) as *mut DoubleReal,
        wk.offset(ib as isize) as *mut DoubleReal,
        wk.offset(iwe as isize) as *mut DoubleReal,
        &raw mut el,
        wk.offset(7 as ::core::ffi::c_int as isize) as *mut DoubleReal,
    );
    return 0 as ::core::ffi::c_int;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn splder_(
    mut ider: *mut Integer,
    mut m: *mut Integer,
    mut n: *mut Integer,
    mut t: *mut DoubleReal,
    mut x: *mut DoubleReal,
    mut c__: *mut DoubleReal,
    mut l: *mut Integer,
    mut q: *mut DoubleReal,
) -> ::core::ffi::c_double {
    let mut i__1: Integer = 0;
    let mut i__2: Integer = 0;
    let mut ret_val: DoubleReal = 0.;
    static mut i__: Integer = 0;
    static mut j: Integer = 0;
    static mut k: Integer = 0;
    static mut z__: DoubleReal = 0.;
    static mut i1: Integer = 0;
    static mut j1: Integer = 0;
    static mut k1: Integer = 0;
    static mut j2: Integer = 0;
    static mut m2: Integer = 0;
    static mut ii: Integer = 0;
    static mut jj: Integer = 0;
    static mut ki: Integer = 0;
    static mut jl: Integer = 0;
    static mut lk: Integer = 0;
    static mut mi: Integer = 0;
    static mut nk: Integer = 0;
    static mut lm: Integer = 0;
    static mut ml: Integer = 0;
    static mut jm: Integer = 0;
    static mut ir: Integer = 0;
    static mut ju: Integer = 0;
    static mut tt: DoubleReal = 0.;
    static mut lk1: Integer = 0;
    static mut mp1: Integer = 0;
    static mut m2m1: Integer = 0;
    static mut jin: Integer = 0;
    static mut nki: Integer = 0;
    static mut npm: Integer = 0;
    static mut lk1i: Integer = 0;
    static mut nki1: Integer = 0;
    static mut lk1i1: Integer = 0;
    static mut xjki: DoubleReal = 0.;
    #[unsafe(export_name = "search_")]
    pub unsafe extern "C" fn search(
        mut n_0: *mut Integer,
        mut x_0: *mut DoubleReal,
        mut t_0: *mut DoubleReal,
        mut l_0: *mut Integer,
    ) -> ::core::ffi::c_int {
        let mut current_block: u64;
        static mut il: Integer = 0;
        static mut iu: Integer = 0;
        x_0 = x_0.offset(-1);
        if *t_0 < *x_0.offset(1 as ::core::ffi::c_int as isize) {
            *l_0 = 0 as ::core::ffi::c_int as Integer;
            return 0 as ::core::ffi::c_int;
        }
        if *t_0 >= *x_0.offset(*n_0 as isize) {
            *l_0 = *n_0;
            return 0 as ::core::ffi::c_int;
        }
        *l_0 = (if *l_0 >= 1 as ::core::ffi::c_int {
            *l_0
        } else {
            1 as ::core::ffi::c_int
        }) as Integer;
        if *l_0 >= *n_0 {
            *l_0 = (*n_0 - 1 as ::core::ffi::c_int) as Integer;
        }
        if *t_0 >= *x_0.offset(*l_0 as isize) {
            if *t_0 < *x_0.offset((*l_0 + 1 as ::core::ffi::c_int) as isize) {
                return 0 as ::core::ffi::c_int;
            }
            *l_0 += 1;
            if *t_0 < *x_0.offset((*l_0 + 1 as ::core::ffi::c_int) as isize) {
                return 0 as ::core::ffi::c_int;
            }
            il = (*l_0 + 1 as ::core::ffi::c_int) as Integer;
            iu = *n_0;
            current_block = 3202994239531656598;
        } else {
            *l_0 -= 1;
            if *t_0 >= *x_0.offset(*l_0 as isize) {
                return 0 as ::core::ffi::c_int;
            }
            il = 1 as ::core::ffi::c_int as Integer;
            current_block = 13523560774170872681;
        }
        loop {
            match current_block {
                13523560774170872681 => {
                    iu = *l_0;
                    current_block = 3202994239531656598;
                }
                _ => {
                    *l_0 = ((il as ::core::ffi::c_int + iu as ::core::ffi::c_int)
                        / 2 as ::core::ffi::c_int) as Integer;
                    if iu - il <= 1 as ::core::ffi::c_int {
                        return 0 as ::core::ffi::c_int;
                    }
                    if *t_0 < *x_0.offset(*l_0 as isize) {
                        current_block = 13523560774170872681;
                        continue;
                    }
                    il = *l_0;
                    current_block = 3202994239531656598;
                }
            }
        }
    }
    q = q.offset(-1);
    c__ = c__.offset(-1);
    x = x.offset(-1);
    m2 = *m << 1 as ::core::ffi::c_int;
    k = m2 - *ider;
    if k < 1 as ::core::ffi::c_int {
        ret_val = 0.0f64 as DoubleReal;
        return ret_val as ::core::ffi::c_double;
    }
    search(
        n,
        x.offset(1 as ::core::ffi::c_int as isize) as *mut DoubleReal,
        t,
        l,
    );
    tt = *t;
    mp1 = (*m + 1 as ::core::ffi::c_int) as Integer;
    npm = *n + *m;
    m2m1 = (m2 as ::core::ffi::c_int - 1 as ::core::ffi::c_int) as Integer;
    k1 = (k as ::core::ffi::c_int - 1 as ::core::ffi::c_int) as Integer;
    nk = *n - k;
    lk = *l - k;
    lk1 = (lk as ::core::ffi::c_int + 1 as ::core::ffi::c_int) as Integer;
    lm = *l - *m;
    jl = (*l + 1 as ::core::ffi::c_int) as Integer;
    ju = *l + m2;
    ii = *n - m2;
    ml = -*l;
    i__1 = ju;
    j = jl;
    while j <= i__1 {
        if j >= mp1 && j <= npm {
            *q.offset((j + ml) as isize) = *c__.offset((j - *m) as isize);
        } else {
            *q.offset((j + ml) as isize) = 0.0f64 as DoubleReal;
        }
        j += 1;
    }
    if *ider > 0 as ::core::ffi::c_int {
        jl -= m2 as ::core::ffi::c_int;
        ml += m2 as ::core::ffi::c_int;
        i__1 = *ider;
        i__ = 1 as ::core::ffi::c_int as Integer;
        while i__ <= i__1 {
            jl += 1;
            ii += 1;
            j1 = (if 1 as ::core::ffi::c_int >= jl {
                1 as ::core::ffi::c_int
            } else {
                jl as ::core::ffi::c_int
            }) as Integer;
            j2 = (if *l <= ii {
                *l
            } else {
                ii as ::core::ffi::c_int
            }) as Integer;
            mi = m2 - i__;
            j = (j2 as ::core::ffi::c_int + 1 as ::core::ffi::c_int) as Integer;
            if j1 <= j2 {
                i__2 = j2;
                jin = j1;
                while jin <= i__2 {
                    j -= 1;
                    jm = ml + j;
                    *q.offset(jm as isize) = (*q.offset(jm as isize)
                        - *q.offset((jm as ::core::ffi::c_int - 1 as ::core::ffi::c_int) as isize))
                        / (*x.offset((j + mi) as isize) - *x.offset(j as isize));
                    jin += 1;
                }
            }
            if !(jl >= 1 as ::core::ffi::c_int) {
                i1 = (i__ as ::core::ffi::c_int + 1 as ::core::ffi::c_int) as Integer;
                j = (ml as ::core::ffi::c_int + 1 as ::core::ffi::c_int) as Integer;
                if i1 <= ml {
                    i__2 = ml;
                    jin = i1;
                    while jin <= i__2 {
                        j -= 1;
                        *q.offset(j as isize) = -*q
                            .offset((j as ::core::ffi::c_int - 1 as ::core::ffi::c_int) as isize);
                        jin += 1;
                    }
                }
            }
            i__ += 1;
        }
        i__1 = k;
        j = 1 as ::core::ffi::c_int as Integer;
        while j <= i__1 {
            *q.offset(j as isize) = *q.offset((j + *ider) as isize);
            j += 1;
        }
    }
    if k1 >= 1 as ::core::ffi::c_int {
        i__1 = k1;
        i__ = 1 as ::core::ffi::c_int as Integer;
        while i__ <= i__1 {
            nki = nk + i__;
            ir = k;
            jj = *l;
            ki = k - i__;
            nki1 = (nki as ::core::ffi::c_int + 1 as ::core::ffi::c_int) as Integer;
            if *l >= nki1 {
                i__2 = *l;
                j = nki1;
                while j <= i__2 {
                    *q.offset(ir as isize) = *q
                        .offset((ir as ::core::ffi::c_int - 1 as ::core::ffi::c_int) as isize)
                        + (tt - *x.offset(jj as isize)) * *q.offset(ir as isize);
                    jj -= 1;
                    ir -= 1;
                    j += 1;
                }
            }
            lk1i = lk1 + i__;
            j1 = (if 1 as ::core::ffi::c_int >= lk1i {
                1 as ::core::ffi::c_int
            } else {
                lk1i as ::core::ffi::c_int
            }) as Integer;
            j2 = (if *l <= nki {
                *l
            } else {
                nki as ::core::ffi::c_int
            }) as Integer;
            if j1 <= j2 {
                i__2 = j2;
                j = j1;
                while j <= i__2 {
                    xjki = *x.offset((jj + ki) as isize);
                    z__ = *q.offset(ir as isize);
                    *q.offset(ir as isize) = z__
                        + (xjki - tt)
                            * (*q.offset(
                                (ir as ::core::ffi::c_int - 1 as ::core::ffi::c_int) as isize,
                            ) - z__)
                            / (xjki - *x.offset(jj as isize));
                    ir -= 1;
                    jj -= 1;
                    j += 1;
                }
            }
            if lk1i <= 0 as ::core::ffi::c_int {
                jj = ki;
                lk1i1 = 1 as Integer - lk1i;
                i__2 = lk1i1;
                j = 1 as ::core::ffi::c_int as Integer;
                while j <= i__2 {
                    let ref mut fresh3 = *q.offset(ir as isize);
                    *fresh3 += ((*x.offset(jj as isize) - tt)
                        * *q.offset((ir as ::core::ffi::c_int - 1 as ::core::ffi::c_int) as isize))
                        as ::core::ffi::c_double;
                    jj -= 1;
                    ir -= 1;
                    j += 1;
                }
            }
            i__ += 1;
        }
    }
    z__ = *q.offset(k as isize);
    if *ider > 0 as ::core::ffi::c_int {
        i__1 = m2m1;
        j = k;
        while j <= i__1 {
            z__ *= j as ::core::ffi::c_double;
            j += 1;
        }
    }
    ret_val = z__;
    return ret_val as ::core::ffi::c_double;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_order_one_interpolation_round_trips_through_splder() {
        let mut x = [0_f64, 1.];
        let mut y = [2_f64, 4.];
        let mut wx = [1_f64, 1.];
        let mut wy = [1_f64];
        let mut coefficients = [0_f64; 2];
        let mut work = [0_f64; 20];
        let mut error = -1;
        unsafe {
            assert_eq!(
                gcvspl(
                    x.as_mut_ptr(),
                    y.as_mut_ptr(),
                    2,
                    wx.as_mut_ptr(),
                    wy.as_mut_ptr(),
                    1,
                    2,
                    1,
                    1,
                    0.,
                    coefficients.as_mut_ptr(),
                    2,
                    work.as_mut_ptr(),
                    &mut error
                ),
                0
            );
        }
        assert_eq!(error, 0);
        let mut near = 0;
        let mut derivative_work = [0_f64; 2];
        let value = unsafe {
            splder(
                0,
                1,
                2,
                0.25,
                x.as_mut_ptr(),
                coefficients.as_mut_ptr(),
                &mut near,
                derivative_work.as_mut_ptr(),
            )
        };
        assert!(
            (value - 2.5).abs() < 1.0e-12,
            "value {value}, coefficients {coefficients:?}"
        );
    }

    #[test]
    fn invalid_order_sets_source_error_one() {
        let mut x = [0_f64, 1.];
        let mut y = [0_f64; 2];
        let mut wx = [1_f64; 2];
        let mut wy = [1_f64];
        let mut c = [0_f64; 2];
        let mut work = [0_f64; 20];
        let mut error = 0;
        unsafe {
            gcvspl(
                x.as_mut_ptr(),
                y.as_mut_ptr(),
                2,
                wx.as_mut_ptr(),
                wy.as_mut_ptr(),
                0,
                2,
                1,
                1,
                0.,
                c.as_mut_ptr(),
                2,
                work.as_mut_ptr(),
                &mut error,
            );
        }
        assert_eq!(error, 1);
    }
}
