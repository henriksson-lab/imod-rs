//! Translation of `IMOD/libwarp/hull-io.c` ("io.c : input-output").
//!
//! Two arms of the source are not selected on this platform: `off_out` exists
//! only under `_WIN32` or `OFF_OUT`, neither of which this build defines, and
//! `epopen`'s body is compiled out under `_WIN32` only (so it is present here).
#![allow(dead_code)]

use core::ffi::{c_int, c_void};

use crate::imod::libwarp::hull::{
    Basis, CDIM, Coord, Neighbor, RDIM, SITE_NUM, Simplex, Site, visit_triang, visit_triang_gen,
};
use crate::imod::libwarp::hull_ch::{HULL_INFINITY, alph_test};
use crate::imod::libwarp::pointops::{PDIM, print_point, print_point_int};

/// `typedef void print_neighbor_f(FILE*, neighbor*)` (`hull.h:217`).
pub type PrintNeighborF = unsafe fn(*mut libc::FILE, *mut Neighbor);
/// `typedef void out_func(point *, int, FILE*, int)` (`hull.h:137`).
pub type OutFunc = unsafe fn(*mut Site, c_int, *mut libc::FILE, c_int);

/// `DFILE` (`hull.h:41`), set to `stderr` by `hullwrap.c:42`.
pub static mut DFILE: *mut libc::FILE = core::ptr::null_mut();

/// `mult_up` (`hull-io.c:30`).
pub static mut MULT_UP: f64 = 1.0;
/// `mins` (`hull-io.c:31`).
pub static mut MINS: [Coord; 8] = [f64::MAX; 8];
/// `maxs` (`hull-io.c:33`).
pub static mut MAXS: [Coord; 8] = [-f64::MAX; 8];
/// `tmpfilenam` (`hull-io.c:48`).
pub static mut TMPFILENAM: [u8; 20] = [0; 20];

/// Original `panic` (`hull-io.c:37`).
///
/// The source is variadic; every call site passes one format and its
/// arguments, so the formatted text is handed in already rendered.
pub unsafe fn panic(text: &std::ffi::CStr) -> ! {
    unsafe {
        libc::fprintf(DFILE, c"%s".as_ptr(), text.as_ptr());
        libc::fflush(DFILE);
        libc::exit(1)
    }
}

/// Original `efopen` (`hull-io.c:51`).
pub unsafe fn efopen(
    file: *const core::ffi::c_char,
    mode: *const core::ffi::c_char,
) -> *mut libc::FILE {
    unsafe {
        let fp = libc::fopen(file, mode);
        if !fp.is_null() {
            return fp;
        }
        libc::fprintf(
            DFILE,
            c"couldn't open file %s mode %s\n".as_ptr(),
            file,
            mode,
        );
        libc::exit(1)
    }
}

/// Original `epopen` (`hull-io.c:61`).
pub unsafe fn epopen(
    com: *const core::ffi::c_char,
    mode: *const core::ffi::c_char,
) -> *mut libc::FILE {
    unsafe {
        let fp = libc::popen(com, mode);
        if !fp.is_null() {
            return fp;
        }
        libc::fprintf(
            *(&raw const STDERR),
            c"couldn't open stream %s mode %s\n".as_ptr(),
            com,
            mode,
        );
        libc::exit(1)
    }
}

unsafe extern "C" {
    #[link_name = "stderr"]
    static STDERR: *mut libc::FILE;
}

/// Original `print_neighbor_snum` (`hull-io.c:74`).
pub unsafe fn print_neighbor_snum(f: *mut libc::FILE, n: *mut Neighbor) {
    unsafe {
        if !(*n).vert.is_null() {
            libc::fprintf(f, c"%d ".as_ptr(), SITE_NUM.unwrap()((*n).vert));
        } else {
            libc::fprintf(f, c"NULL vert ".as_ptr());
        }
        libc::fflush(*(&raw const STDOUT));
    }
}

unsafe extern "C" {
    #[link_name = "stdout"]
    static STDOUT: *mut libc::FILE;
}

/// Original `print_basis` (`hull-io.c:84`).
pub unsafe fn print_basis(f: *mut libc::FILE, b: *mut Basis) {
    unsafe {
        if b.is_null() {
            libc::fprintf(f, c"NULL basis ".as_ptr());
            libc::fflush(*(&raw const STDOUT));
            return;
        }
        if (*b).lscale < 0 {
            libc::fprintf(f, c"\nbasis computed".as_ptr());
            return;
        }
        libc::fprintf(
            f,
            c"\n%p  %d \n b=".as_ptr(),
            b.cast::<c_void>(),
            (*b).lscale,
        );
        print_point(f, RDIM, (*b).vecs.as_mut_ptr());
        libc::fprintf(f, c"\n a= ".as_ptr());
        print_point_int(f, RDIM, (*b).vecs.as_mut_ptr().add(RDIM as usize));
        libc::fprintf(f, c"   ".as_ptr());
        libc::fflush(f);
    }
}

/// Original static `print_simplex_num` (`hull-io.c:95`).
unsafe fn print_simplex_num(f: *mut libc::FILE, s: *mut Simplex) {
    unsafe {
        libc::fprintf(f, c"simplex ".as_ptr());
        if s.is_null() {
            libc::fprintf(f, c"NULL ".as_ptr());
        } else {
            libc::fprintf(f, c"%p  ".as_ptr(), s.cast::<c_void>());
        }
    }
}

/// Original `print_neighbor_full` (`hull-io.c:101`).
pub unsafe fn print_neighbor_full(f: *mut libc::FILE, n: *mut Neighbor) {
    unsafe {
        if n.is_null() {
            libc::fprintf(f, c"null neighbor\n".as_ptr());
            return;
        }

        print_simplex_num(f, (*n).simp);
        print_neighbor_snum(f, n);
        libc::fprintf(f, c":  ".as_ptr());
        libc::fflush(f);
        if !(*n).vert.is_null() {
            print_point(f, PDIM, (*n).vert);
            libc::fflush(f);
        }
        print_basis(f, (*n).basis);
        libc::fflush(f);
        libc::fprintf(f, c"\n".as_ptr());
    }
}

/// Original `print_facet` (`hull-io.c:116`).
pub unsafe fn print_facet(
    f: *mut libc::FILE,
    s: *mut Simplex,
    pnfin: Option<PrintNeighborF>,
) -> *mut c_void {
    unsafe {
        let mut sn = (*s).neigh.as_mut_ptr();
        for _ in 0..CDIM {
            pnfin.unwrap()(f, sn);
            sn = sn.add(1);
        }
        libc::fprintf(f, c"\n".as_ptr());
        libc::fflush(f);
        core::ptr::null_mut()
    }
}

/// Original `print_simplex_f` (`hull-io.c:130`).
///
/// `pnf` is a function static in the source: a non-null `pnfin` installs it
/// for every later call, and a null `s` with a non-null `pnfin` only installs.
pub unsafe fn print_simplex_f(
    s: *mut Simplex,
    f: *mut libc::FILE,
    pnfin: Option<PrintNeighborF>,
) -> *mut c_void {
    unsafe {
        static mut PNF: Option<PrintNeighborF> = None;

        if pnfin.is_some() {
            PNF = pnfin;
            if s.is_null() {
                return core::ptr::null_mut();
            }
        }

        print_simplex_num(f, s);
        libc::fprintf(f, c"\n".as_ptr());
        if s.is_null() {
            return core::ptr::null_mut();
        }
        libc::fprintf(f, c"normal =".as_ptr());
        print_basis(f, (*s).normal);
        libc::fprintf(f, c"\n".as_ptr());
        libc::fprintf(f, c"peak =".as_ptr());
        PNF.unwrap()(f, core::ptr::addr_of_mut!((*s).peak));
        libc::fprintf(f, c"facet =\n".as_ptr());
        libc::fflush(f);
        print_facet(f, s, PNF)
    }
}

/// Original `print_simplex` (`hull-io.c:145`).
pub unsafe fn print_simplex(s: *mut Simplex, fin: *mut c_void) -> *mut c_void {
    unsafe {
        static mut F: *mut libc::FILE = core::ptr::null_mut();

        if !fin.is_null() {
            F = fin.cast::<libc::FILE>();
            if s.is_null() {
                return core::ptr::null_mut();
            }
        }

        print_simplex_f(s, *(&raw const F), None)
    }
}

/// Original `print_triang` (`hull-io.c:155`).
pub unsafe fn print_triang(root: *mut Simplex, f: *mut libc::FILE, pnf: Option<PrintNeighborF>) {
    unsafe {
        print_simplex(core::ptr::null_mut(), f.cast::<c_void>());
        print_simplex_f(core::ptr::null_mut(), core::ptr::null_mut(), pnf);
        visit_triang(root, print_simplex);
    }
}

/// Original static `p_peak_test` (`hull-io.c:161`).
unsafe fn p_peak_test(s: *mut Simplex) -> *mut c_void {
    unsafe {
        if (*s).peak.vert == crate::imod::libwarp::hull::P {
            s.cast::<c_void>()
        } else {
            core::ptr::null_mut()
        }
    }
}

/// Original `check_simplex` (`hull-io.c:164`).
pub unsafe fn check_simplex(s: *mut Simplex, _dum: *mut c_void) -> *mut c_void {
    unsafe {
        let mut sn = (*s).neigh.as_mut_ptr().offset(-1);
        let mut i = -1;
        while i < CDIM {
            let sns = (*sn).simp;
            if sns.is_null() {
                libc::fprintf(DFILE, c"check_triang; bad simplex\n".as_ptr());
                /* DNM: fix format %G -> %d */
                print_simplex_f(s, DFILE, Some(print_neighbor_full));
                libc::fprintf(
                    DFILE,
                    c"site_num(p)=%d\n".as_ptr(),
                    SITE_NUM.unwrap()(crate::imod::libwarp::hull::P),
                );
                return s.cast::<c_void>();
            }
            if (*s).peak.vert.is_null() && !(*sns).peak.vert.is_null() && i != -1 {
                libc::fprintf(DFILE, c"huh?\n".as_ptr());
                print_simplex_f(s, DFILE, Some(print_neighbor_full));
                print_simplex_f(sns, DFILE, Some(print_neighbor_full));
                libc::exit(1);
            }
            let mut j = -1;
            let mut snn = (*sns).neigh.as_mut_ptr().offset(-1);
            while j < CDIM && (*snn).simp != s {
                j += 1;
                snn = snn.add(1);
            }
            if j == CDIM {
                libc::fprintf(DFILE, c"adjacency failure:\n".as_ptr());
                /* DEBEXP(-1, site_num(p)) -- DEBUG (-7) > -1 is false */
                print_simplex_f(sns, DFILE, Some(print_neighbor_full));
                print_simplex_f(s, DFILE, Some(print_neighbor_full));
                libc::exit(1);
            }
            let mut k = -1;
            let mut snn = (*sns).neigh.as_mut_ptr().offset(-1);
            while k < CDIM {
                let vn = (*snn).vert;
                if k != j {
                    let mut l = -1;
                    let mut sn2 = (*s).neigh.as_mut_ptr().offset(-1);
                    while l < CDIM && (*sn2).vert != vn {
                        l += 1;
                        sn2 = sn2.add(1);
                    }
                    if l == CDIM {
                        libc::fprintf(DFILE, c"cdim=%d\n".as_ptr(), CDIM);
                        libc::fprintf(
                            DFILE,
                            c"error: neighboring simplices with incompatible vertices:\n".as_ptr(),
                        );
                        print_simplex_f(sns, DFILE, Some(print_neighbor_full));
                        print_simplex_f(s, DFILE, Some(print_neighbor_full));
                        libc::exit(1);
                    }
                }
                k += 1;
                snn = snn.add(1);
            }
            i += 1;
            sn = sn.add(1);
        }
        core::ptr::null_mut()
    }
}

/// Original static `p_neight` (`hull-io.c:213`).
unsafe fn p_neight(s: *mut Simplex, i: c_int, _dum: *mut c_void) -> c_int {
    unsafe {
        ((*s).neigh.as_mut_ptr().add(i as usize).read().vert != crate::imod::libwarp::hull::P)
            as c_int
    }
}

/// Original `check_triang` (`hull-io.c:215`).
pub unsafe fn check_triang(root: *mut Simplex) {
    unsafe {
        visit_triang(root, check_simplex);
    }
}

/// Original `check_new_triangs` (`hull-io.c:217`).
pub unsafe fn check_new_triangs(s: *mut Simplex) {
    unsafe {
        visit_triang_gen(s, check_simplex, p_neight);
    }
}

/* outfuncs: given a list of points, output in a given format */

/// Original `vlist_out` (`hull-io.c:226`).
pub unsafe fn vlist_out(v: *mut Site, vdim: c_int, fin: *mut libc::FILE, _amble: c_int) {
    unsafe {
        static mut F: *mut libc::FILE = core::ptr::null_mut();

        if !fin.is_null() {
            F = fin;
            if v.is_null() {
                return;
            }
        }

        for j in 0..vdim {
            libc::fprintf(
                *(&raw const F),
                c"%d ".as_ptr(),
                SITE_NUM.unwrap()(*v.add(j as usize)),
            );
        }
        libc::fprintf(*(&raw const F), c"\n".as_ptr());
    }
}

/// Original `mp_out` (`hull-io.c:300`).
pub unsafe fn mp_out(v: *mut Site, mut vdim: c_int, fin: *mut libc::FILE, amble: c_int) {
    unsafe {
        static mut FIGNO: c_int = 1;
        static mut F: *mut libc::FILE = core::ptr::null_mut();

        if !fin.is_null() {
            F = fin;
        }

        if PDIM != 2 {
            /* warning(-10, mp for planar points only) */
            static mut MESSCOUNT: c_int = 0;
            MESSCOUNT += 1;
            if MESSCOUNT <= 10 {
                libc::fprintf(DFILE, c"mp for planar points only\n".as_ptr());
                libc::fflush(DFILE);
                libc::fprintf(DFILE, c"hull-io.c line %d \n".as_ptr(), 309);
                libc::fflush(DFILE);
            }
            if MESSCOUNT == 10 {
                libc::fprintf(DFILE, c"consider yourself warned\n".as_ptr());
                libc::fflush(DFILE);
            }
            return;
        }
        if amble == 0 {
            if v.is_null() {
                return;
            }
            for i in 0..vdim {
                if *v.add(i as usize) == core::ptr::addr_of_mut!(HULL_INFINITY).cast::<Coord>() {
                    let t = *v.add(i as usize);
                    *v.add(i as usize) = *v.add(vdim as usize - 1);
                    *v.add(vdim as usize - 1) = t;
                    vdim -= 1;
                    break;
                }
            }
            libc::fprintf(*(&raw const F), c"draw ".as_ptr());
            for i in 0..vdim {
                libc::fprintf(
                    *(&raw const F),
                    if i + 1 < vdim {
                        c"(%Gu,%Gu)--".as_ptr()
                    } else {
                        c"(%Gu,%Gu);\n".as_ptr()
                    },
                    *(*v.add(i as usize)) / MULT_UP,
                    *(*v.add(i as usize)).add(1) / MULT_UP,
                );
            }
        } else if amble == -1 {
            if FIGNO == 1 {
                libc::fprintf(*(&raw const F), c"u=1pt;\n".as_ptr());
            }
            libc::fprintf(*(&raw const F), c"beginfig(%d);\n".as_ptr(), FIGNO);
            FIGNO += 1;
        } else if amble == 1 {
            libc::fprintf(*(&raw const F), c"endfig;\n".as_ptr());
        }
    }
}

/// Original `ps_out` (`hull-io.c:336`).
pub unsafe fn ps_out(v: *mut Site, mut vdim: c_int, fin: *mut libc::FILE, amble: c_int) {
    unsafe {
        static mut F: *mut libc::FILE = core::ptr::null_mut();
        static mut SCALER: f64 = 0.;

        if !fin.is_null() {
            F = fin;
        }

        if PDIM != 2 {
            /* warning(-10, ps for planar points only) */
            static mut MESSCOUNT: c_int = 0;
            MESSCOUNT += 1;
            if MESSCOUNT <= 10 {
                libc::fprintf(DFILE, c"ps for planar points only\n".as_ptr());
                libc::fflush(DFILE);
                libc::fprintf(DFILE, c"hull-io.c line %d \n".as_ptr(), 343);
                libc::fflush(DFILE);
            }
            if MESSCOUNT == 10 {
                libc::fprintf(DFILE, c"consider yourself warned\n".as_ptr());
                libc::fflush(DFILE);
            }
            return;
        }

        if amble == 0 {
            if v.is_null() {
                return;
            }
            for i in 0..vdim {
                if *v.add(i as usize) == core::ptr::addr_of_mut!(HULL_INFINITY).cast::<Coord>() {
                    let t = *v.add(i as usize);
                    *v.add(i as usize) = *v.add(vdim as usize - 1);
                    *v.add(vdim as usize - 1) = t;
                    vdim -= 1;
                    break;
                }
            }
            libc::fprintf(
                *(&raw const F),
                c"newpath %G %G moveto\n".as_ptr(),
                *(*v) * SCALER,
                *(*v).add(1) * SCALER,
            );
            for i in 1..vdim {
                libc::fprintf(
                    *(&raw const F),
                    c"%G %G lineto\n".as_ptr(),
                    *(*v.add(i as usize)) * SCALER,
                    *(*v.add(i as usize)).add(1) * SCALER,
                );
            }
            libc::fprintf(*(&raw const F), c"stroke\n".as_ptr());
        } else if amble == -1 {
            let mut len = [0f32; 2];
            libc::fprintf(*(&raw const F), c"%!PS\n".as_ptr());
            len[0] = (MAXS[0] - MINS[0]) as f32;
            len[1] = (MAXS[1] - MINS[1]) as f32;
            let maxlen = if len[0] > len[1] { len[0] } else { len[1] };
            SCALER = 216. / maxlen as f64;

            libc::fprintf(
                *(&raw const F),
                c"%%BoundingBox: %G %G %G %G \n".as_ptr(),
                MINS[0] * SCALER,
                MINS[1] * SCALER,
                MAXS[0] * SCALER,
                MAXS[1] * SCALER,
            );
            libc::fprintf(*(&raw const F), c"%%Creator: hull program\n".as_ptr());
            libc::fprintf(*(&raw const F), c"%%Pages: 1\n".as_ptr());
            libc::fprintf(*(&raw const F), c"%%EndProlog\n".as_ptr());
            libc::fprintf(*(&raw const F), c"%%Page: 1 1\n".as_ptr());
            libc::fprintf(
                *(&raw const F),
                c" 0.5 setlinewidth [] 0 setdash\n".as_ptr(),
            );
            libc::fprintf(
                *(&raw const F),
                c" 1 setlinecap 1 setlinejoin 10 setmiterlimit\n".as_ptr(),
            );
        } else if amble == 1 {
            libc::fprintf(*(&raw const F), c"showpage\n %%EOF\n".as_ptr());
        }
    }
}

/// Original `cpr_out` (`hull-io.c:387`).
pub unsafe fn cpr_out(v: *mut Site, vdim: c_int, fin: *mut libc::FILE, _amble: c_int) {
    unsafe {
        static mut F: *mut libc::FILE = core::ptr::null_mut();

        if !fin.is_null() {
            F = fin;
            if v.is_null() {
                return;
            }
        }

        if PDIM != 3 {
            /* warning(-10, cpr for 3d points only) */
            static mut MESSCOUNT: c_int = 0;
            MESSCOUNT += 1;
            if MESSCOUNT <= 10 {
                libc::fprintf(DFILE, c"cpr for 3d points only\n".as_ptr());
                libc::fflush(DFILE);
                libc::fprintf(DFILE, c"hull-io.c line %d \n".as_ptr(), 394);
                libc::fflush(DFILE);
            }
            if MESSCOUNT == 10 {
                libc::fprintf(DFILE, c"consider yourself warned\n".as_ptr());
                libc::fflush(DFILE);
            }
            return;
        }

        for i in 0..vdim {
            if *v.add(i as usize) == core::ptr::addr_of_mut!(HULL_INFINITY).cast::<Coord>() {
                return;
            }
        }

        libc::fprintf(
            *(&raw const F),
            c"t %G %G %G %G %G %G %G %G %G 3 128\n".as_ptr(),
            *(*v) / MULT_UP,
            *(*v).add(1) / MULT_UP,
            *(*v).add(2) / MULT_UP,
            *(*v.add(1)) / MULT_UP,
            *(*v.add(1)).add(1) / MULT_UP,
            *(*v.add(1)).add(2) / MULT_UP,
            *(*v.add(2)) / MULT_UP,
            *(*v.add(2)).add(1) / MULT_UP,
            *(*v.add(2)).add(2) / MULT_UP,
        );
    }
}

/* vist_funcs for different kinds of output: facets, alpha shapes, etc. */

/// Original `facets_print` (`hull-io.c:413`).
///
/// DNM: Fix argument type and avoid use of p for arg since it is global.
pub unsafe fn facets_print(s: *mut Simplex, funcp: Option<OutFunc>) -> *mut c_void {
    unsafe {
        static mut OUT_FUNC_HERE: Option<OutFunc> = None;
        let mut v: [Site; 8] = [core::ptr::null_mut(); 8];

        if funcp.is_some() {
            OUT_FUNC_HERE = funcp;
            if s.is_null() {
                return core::ptr::null_mut();
            }
        }

        for j in 0..CDIM {
            v[j as usize] = (*s).neigh.as_mut_ptr().add(j as usize).read().vert;
        }

        OUT_FUNC_HERE.unwrap()(v.as_mut_ptr(), CDIM, core::ptr::null_mut(), 0);

        core::ptr::null_mut()
    }
}

/// Original `ridges_print` (`hull-io.c:429`).
pub unsafe fn ridges_print(s: *mut Simplex, funcp: Option<OutFunc>) -> *mut c_void {
    unsafe {
        static mut OUT_FUNC_HERE: Option<OutFunc> = None;
        let mut v: [Site; 8] = [core::ptr::null_mut(); 8];

        if funcp.is_some() {
            OUT_FUNC_HERE = funcp;
            if s.is_null() {
                return core::ptr::null_mut();
            }
        }

        for j in 0..CDIM {
            let mut vnum = 0;
            for k in 0..CDIM {
                if k == j {
                    continue;
                }
                v[vnum] = (*s).neigh.as_mut_ptr().add(k as usize).read().vert;
                vnum += 1;
            }
            OUT_FUNC_HERE.unwrap()(v.as_mut_ptr(), CDIM - 1, core::ptr::null_mut(), 0);
        }
        core::ptr::null_mut()
    }
}

/// Original `afacets_print` (`hull-io.c:450`).
pub unsafe fn afacets_print(s: *mut Simplex, funcp: Option<OutFunc>) -> *mut c_void {
    unsafe {
        static mut OUT_FUNC_HERE: Option<OutFunc> = None;
        let mut v: [Site; 8] = [core::ptr::null_mut(); 8];

        if funcp.is_some() {
            OUT_FUNC_HERE = funcp;
            if s.is_null() {
                return core::ptr::null_mut();
            }
        }

        for j in 0..CDIM {
            /* check for ashape consistency */
            let mut k = 0;
            while k < CDIM {
                let neighbour = (*s).neigh.as_mut_ptr().add(j as usize).read().simp;
                if (*neighbour).neigh.as_mut_ptr().add(k as usize).read().simp == s {
                    break;
                }
                k += 1;
            }
            let other = (*s).neigh.as_mut_ptr().add(j as usize).read().simp;
            if alph_test(s, j, core::ptr::null_mut()) != alph_test(other, k, core::ptr::null_mut())
            {
                /* DEB(-10, alpha-shape not consistent) DEBTR(-10) */
                libc::fprintf(DFILE, c"alpha-shape not consistent\n".as_ptr());
                libc::fflush(DFILE);
                libc::fprintf(DFILE, c"hull-io.c line %d \n".as_ptr(), 461);
                libc::fflush(DFILE);
                print_simplex_f(s, DFILE, Some(print_neighbor_full));
                print_simplex_f(other, DFILE, Some(print_neighbor_full));
                libc::fflush(DFILE);
                libc::exit(1);
            }
        }
        for j in 0..CDIM {
            let mut vnum = 0;
            if alph_test(s, j, core::ptr::null_mut()) != 0 {
                continue;
            }
            for k in 0..CDIM {
                if k == j {
                    continue;
                }
                v[vnum] = (*s).neigh.as_mut_ptr().add(k as usize).read().vert;
                vnum += 1;
            }
            OUT_FUNC_HERE.unwrap()(v.as_mut_ptr(), CDIM - 1, core::ptr::null_mut(), 0);
        }
        core::ptr::null_mut()
    }
}
