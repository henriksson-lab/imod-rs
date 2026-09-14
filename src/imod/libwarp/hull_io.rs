//! Translation of `IMOD/libwarp/hull-io.c` ("io.c : input-output").
//!
//! Two of the source's arms are not compiled on this platform.  `off_out` is
//! inside `#ifdef _WIN32 ... #else #ifdef OFF_OUT`, and the vendored build
//! defines neither, so the function has **no definition at all** in
//! `libiwarp` on Linux and none is written here; `hull.h:245` still declares
//! it.  `epopen`'s body is compiled out only under `_WIN32`, so it is present
//! — but its only caller is `off_out`, so it is unreachable.
//!
//! Everything the source writes goes to `DFILE`, which `hullwrap.c:29`
//! declares and `hullwrap.c:42` sets to `stderr`; nothing else ever assigns
//! it, so the sink is spelled [`ImodFile::Stderr`] at the point of use rather
//! than kept in a global.
//!
//! Several routines here use the source's "pass a non-NULL argument to latch
//! my `static FILE *` / `static out_func *`, then pass NULL for the real
//! calls" convention.  A captured writer or a passed function does the same
//! job without the latch, so `print_simplex`, `vlist_out`, `mp_out`, `ps_out`,
//! `cpr_out`, `facets_print`, `ridges_print` and `afacets_print` take theirs
//! directly.  `print_simplex_f`'s latch survives, because it holds a plain
//! function pointer that `print_triang` really does install for later calls.
#![allow(dead_code)]

use std::cell::{Cell, RefCell};
use std::io::Write;

use crate::imod::libcfshr::b3dutil::{CArg, ImodFile, c_format};
use crate::imod::libwarp::hull::{
    CDIM, Coord, HULL_INFINITY, HullStorage, MAXDIM, Neighbor, P, RDIM, SITE_NUM, Site,
    visit_triang, visit_triang_gen,
};
use crate::imod::libwarp::hull_ch::alph_test;
use crate::imod::libwarp::pointops::{PDIM, print_point, print_point_int};

/// C `print_neighbor_f` (`hull.h:217`).
pub type PrintNeighborF = fn(&mut HullStorage, &mut dyn Write, Option<Neighbor>);
/// C `out_func` (`hull.h:136`).  The source's `FILE *` and `amble` arguments
/// stay; the writer is captured instead of latched.
pub type OutFunc<'a> = &'a mut dyn FnMut(&mut HullStorage, &mut [Site], i32, i32);

thread_local! {
    /// C `double mult_up` (`hull-io.c:31`).
    pub static MULT_UP: Cell<f64> = const { Cell::new(1.0) };
    /// C `Coord mins[MAXDIM]` (`hull-io.c:32`).
    pub static MINS: Cell<[Coord; MAXDIM]> = const { Cell::new([f64::MAX; MAXDIM]) };
    /// C `Coord maxs[MAXDIM]` (`hull-io.c:34`).
    pub static MAXS: Cell<[Coord; MAXDIM]> = const { Cell::new([-f64::MAX; MAXDIM]) };
    /// C `char tmpfilenam[L_tmpnam]` (`hull-io.c:49`); only `off_out`, which
    /// this platform does not compile, ever names it.
    pub static TMPFILENAM: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };

    /// `print_simplex_f`'s `static print_neighbor_f *pnf` (`hull-io.c:132`).
    static PNF: Cell<Option<PrintNeighborF>> = const { Cell::new(None) };
    /// The `static int messcount` of `warning(-10, ...)` at `hull-io.c:310`.
    static MP_MESSCOUNT: Cell<i32> = const { Cell::new(0) };
    /// The `static int messcount` of `warning(-10, ...)` at `hull-io.c:343`.
    static PS_MESSCOUNT: Cell<i32> = const { Cell::new(0) };
    /// The `static int messcount` of `warning(-10, ...)` at `hull-io.c:394`.
    static CPR_MESSCOUNT: Cell<i32> = const { Cell::new(0) };
    /// `mp_out`'s `static int figno = 1` (`hull-io.c:303`).
    static FIGNO: Cell<i32> = const { Cell::new(1) };
    /// `ps_out`'s `static double scaler` (`hull-io.c:339`).
    static SCALER: Cell<f64> = const { Cell::new(0.) };
}

/// Original `panic` (`hull-io.c:37`).
///
/// The source is `panic(char *fmt, ...)` and `vfprintf`s to `DFILE`; the two
/// call sites format with [`c_format`] and hand the text in.
pub fn panic(text: &str) -> ! {
    let mut dfile = ImodFile::Stderr;
    let _ = dfile.write_all(text.as_bytes());
    let _ = dfile.flush();
    std::process::exit(1)
}

/// Original `efopen` (`hull-io.c:51`).
pub fn efopen(file: &str, mode: &str) -> ImodFile {
    /* DNM: add suggested parens */
    if let Some(fp) = ImodFile::open(file, mode) {
        return fp;
    }
    let mut dfile = ImodFile::Stderr;
    let _ = dfile.write_all(
        c_format(
            "couldn't open file %s mode %s\n",
            &[CArg::Str(file), CArg::Str(mode)],
        )
        .as_bytes(),
    );
    std::process::exit(1)
}

/// Original `epopen` (`hull-io.c:61`).
///
/// A `popen` is a process boundary, so it is a `std::process::Command` here.
/// Its only caller is `off_out`, which this platform does not compile.
pub fn epopen(com: &str, mode: &str) -> Option<std::process::Child> {
    /* DNM: add suggested parens */
    let child = std::process::Command::new("sh")
        .arg("-c")
        .arg(com)
        .stdin(if mode.starts_with('w') {
            std::process::Stdio::piped()
        } else {
            std::process::Stdio::inherit()
        })
        .stdout(if mode.starts_with('r') {
            std::process::Stdio::piped()
        } else {
            std::process::Stdio::inherit()
        })
        .spawn();
    if let Ok(fp) = child {
        return Some(fp);
    }
    let mut err = ImodFile::Stderr;
    let _ = err.write_all(
        c_format(
            "couldn't open stream %s mode %s\n",
            &[CArg::Str(com), CArg::Str(mode)],
        )
        .as_bytes(),
    );
    std::process::exit(1)
}

/// Original `print_neighbor_snum` (`hull-io.c:74`).
pub fn print_neighbor_snum(_st: &mut HullStorage, f: &mut dyn Write, n: Option<Neighbor>) {
    assert!(SITE_NUM.get().is_some());
    let n = n.unwrap();
    if n.vert != 0 {
        let _ = f.write_all(
            c_format("%d ", &[CArg::Int(SITE_NUM.get().unwrap()(n.vert) as i64)]).as_bytes(),
        );
    } else {
        let _ = f.write_all(b"NULL vert ");
    }
    let _ = ImodFile::Stdout.flush();
}

/// Original `print_basis` (`hull-io.c:84`).
///
/// **Deviation.** The source prints the basis's *address* with `%p`; a slot in
/// the arena has an index rather than an address, so the index is what `%p`
/// receives.  This line is only reachable from `reduce_inner`'s failure report
/// (`hull-ch.c:322`) and from `print_neighbor_full`, i.e. from the
/// adjacency-failure paths that end in `exit(1)`.
pub fn print_basis(st: &mut HullStorage, f: &mut dyn Write, b: usize) {
    if b == 0 {
        let _ = f.write_all(b"NULL basis ");
        let _ = ImodFile::Stdout.flush();
        return;
    }
    if st.basis[b].lscale < 0 {
        let _ = f.write_all(b"\nbasis computed");
        return;
    }
    let _ = f.write_all(
        c_format(
            "\n%p  %d \n b=",
            &[CArg::Ptr(b), CArg::Int(st.basis[b].lscale as i64)],
        )
        .as_bytes(),
    );
    let rdim = RDIM.get();
    let vecs = st.basis[b].vecs;
    print_point(f, rdim, Some(&vecs[..]));
    let _ = f.write_all(b"\n a= ");
    print_point_int(f, rdim, Some(&vecs[rdim as usize..]));
    let _ = f.write_all(b"   ");
    let _ = f.flush();
}

/// Original static `print_simplex_num` (`hull-io.c:95`).
///
/// Same `%p` deviation as [`print_basis`].
fn print_simplex_num(_st: &mut HullStorage, f: &mut dyn Write, s: usize) {
    let _ = f.write_all(b"simplex ");
    if s == 0 {
        let _ = f.write_all(b"NULL ");
    } else {
        let _ = f.write_all(c_format("%p  ", &[CArg::Ptr(s)]).as_bytes());
    }
}

/// Original `print_neighbor_full` (`hull-io.c:101`).
pub fn print_neighbor_full(st: &mut HullStorage, f: &mut dyn Write, n: Option<Neighbor>) {
    if n.is_none() {
        let _ = f.write_all(b"null neighbor\n");
        return;
    }
    let n = n.unwrap();

    print_simplex_num(st, f, n.simp);
    print_neighbor_snum(st, f, Some(n));
    let _ = f.write_all(b":  ");
    let _ = f.flush();
    if n.vert != 0 {
        /* if (n->basis && n->basis->lscale <0) fprintf(F, "trans "); else */
        let pdim = PDIM.get();
        let slice: Vec<Coord> = st.sites[n.vert..n.vert + pdim as usize].to_vec();
        print_point(f, pdim, Some(&slice[..]));
        let _ = f.flush();
    }
    print_basis(st, f, n.basis);
    let _ = f.flush();
    let _ = f.write_all(b"\n");
}

/// Original `print_facet` (`hull-io.c:116`).
pub fn print_facet(
    st: &mut HullStorage,
    f: &mut dyn Write,
    s: usize,
    pnfin: PrintNeighborF,
) -> usize {
    /* fprintf(F, "%d ", s->mark); */
    for i in 0..CDIM.get() {
        let sn = st.simplex[s].neigh[i as usize];
        pnfin(st, f, Some(sn));
    }
    let _ = f.write_all(b"\n");
    let _ = f.flush();
    0
}

/// Original `print_simplex_f` (`hull-io.c:130`).
pub fn print_simplex_f(
    st: &mut HullStorage,
    s: usize,
    f: &mut dyn Write,
    pnfin: Option<PrintNeighborF>,
) -> usize {
    if let Some(pnfin) = pnfin {
        PNF.set(Some(pnfin));
        if s == 0 {
            return 0;
        }
    }

    print_simplex_num(st, f, s);
    let _ = f.write_all(b"\n");
    if s == 0 {
        return 0;
    }
    let _ = f.write_all(b"normal =");
    let normal = st.simplex[s].normal;
    print_basis(st, f, normal);
    let _ = f.write_all(b"\n");
    let _ = f.write_all(b"peak =");
    let pnf = PNF.get().unwrap();
    let peak = st.simplex[s].peak;
    pnf(st, f, Some(peak));
    let _ = f.write_all(b"facet =\n");
    let _ = f.flush();
    print_facet(st, f, s, pnf)
}

/// Original `print_simplex` (`hull-io.c:145`).
///
/// The source's `static FILE *F` latch is gone; the writer is passed.
pub fn print_simplex(st: &mut HullStorage, s: usize, f: &mut dyn Write) -> usize {
    print_simplex_f(st, s, f, None)
}

/// Original `print_triang` (`hull-io.c:155`).
pub fn print_triang(st: &mut HullStorage, root: usize, f: &mut dyn Write, pnf: PrintNeighborF) {
    /* print_simplex(0, F) only latched the writer, which is now passed. */
    print_simplex_f(st, 0, f, Some(pnf));
    visit_triang(st, root, &mut |st, s| print_simplex(st, s, f));
}

/// Original static `p_peak_test` (`hull-io.c:161`).  Nothing calls it.
fn p_peak_test(st: &mut HullStorage, s: usize) -> usize {
    if st.simplex[s].peak.vert == P.get() {
        s
    } else {
        0
    }
}

/// Original `check_simplex` (`hull-io.c:164`).
pub fn check_simplex(st: &mut HullStorage, s: usize) -> usize {
    let cdim = CDIM.get();
    let mut dfile = ImodFile::Stderr;

    for i in -1..cdim {
        let sn = if i < 0 {
            st.simplex[s].peak
        } else {
            st.simplex[s].neigh[i as usize]
        };
        let sns = sn.simp;
        if sns == 0 {
            let _ = dfile.write_all(b"check_triang; bad simplex\n");
            print_simplex_f(st, s, &mut dfile, Some(print_neighbor_full));
            /* DNM: fix format %G -> %d */
            let _ = dfile.write_all(
                c_format(
                    "site_num(p)=%d\n",
                    &[CArg::Int(SITE_NUM.get().unwrap()(P.get()) as i64)],
                )
                .as_bytes(),
            );
            return s;
        }
        if st.simplex[s].peak.vert == 0 && st.simplex[sns].peak.vert != 0 && i != -1 {
            let _ = dfile.write_all(b"huh?\n");
            print_simplex_f(st, s, &mut dfile, Some(print_neighbor_full));
            print_simplex_f(st, sns, &mut dfile, Some(print_neighbor_full));
            std::process::exit(1);
        }
        let mut j = -1;
        while j < cdim
            && (if j < 0 {
                st.simplex[sns].peak.simp
            } else {
                st.simplex[sns].neigh[j as usize].simp
            }) != s
        {
            j += 1;
        }
        if j == cdim {
            let _ = dfile.write_all(b"adjacency failure:\n");
            /* DEBEXP(-1, site_num(p)) -- DEBUG (-7) > -1 is false. */
            print_simplex_f(st, sns, &mut dfile, Some(print_neighbor_full));
            print_simplex_f(st, s, &mut dfile, Some(print_neighbor_full));
            std::process::exit(1);
        }
        for k in -1..cdim {
            let vn = if k < 0 {
                st.simplex[sns].peak.vert
            } else {
                st.simplex[sns].neigh[k as usize].vert
            };
            if k != j {
                let mut l = -1;
                while l < cdim
                    && (if l < 0 {
                        st.simplex[s].peak.vert
                    } else {
                        st.simplex[s].neigh[l as usize].vert
                    }) != vn
                {
                    l += 1;
                }
                if l == cdim {
                    let _ = dfile
                        .write_all(c_format("cdim=%d\n", &[CArg::Int(cdim as i64)]).as_bytes());
                    let _ = dfile
                        .write_all(b"error: neighboring simplices with incompatible vertices:\n");
                    print_simplex_f(st, sns, &mut dfile, Some(print_neighbor_full));
                    print_simplex_f(st, s, &mut dfile, Some(print_neighbor_full));
                    std::process::exit(1);
                }
            }
        }
    }
    0
}

/// Original static `p_neight` (`hull-io.c:213`).
fn p_neight(st: &mut HullStorage, s: usize, i: i32) -> i32 {
    (st.simplex[s].neigh[i as usize].vert != P.get()) as i32
}

/// Original `check_triang` (`hull-io.c:215`).
pub fn check_triang(st: &mut HullStorage, root: usize) {
    visit_triang(st, root, &mut |st, s| check_simplex(st, s));
}

/// Original `check_new_triangs` (`hull-io.c:217`).
pub fn check_new_triangs(st: &mut HullStorage, s: usize) {
    visit_triang_gen(st, s, &mut |st, s| check_simplex(st, s), &mut |st, s, i| {
        p_neight(st, s, i)
    });
}

/* outfuncs: given a list of points, output in a given format */

/// Original `vlist_out` (`hull-io.c:226`).
pub fn vlist_out(_st: &mut HullStorage, v: &mut [Site], vdim: i32, f: &mut dyn Write, _amble: i32) {
    for j in 0..vdim {
        let _ = f.write_all(
            c_format(
                "%d ",
                &[CArg::Int(SITE_NUM.get().unwrap()(v[j as usize]) as i64)],
            )
            .as_bytes(),
        );
    }
    let _ = f.write_all(b"\n");
}

/// Original `mp_out` (`hull-io.c:300`).
pub fn mp_out(st: &mut HullStorage, v: &mut [Site], mut vdim: i32, f: &mut dyn Write, amble: i32) {
    /* should fix scaling */
    if PDIM.get() != 2 {
        /* warning(-10, mp for planar points only) (`hull-io.c:310`) */
        MP_MESSCOUNT.set(MP_MESSCOUNT.get() + 1);
        let count = MP_MESSCOUNT.get();
        let mut d = ImodFile::Stderr;
        if count <= 10 {
            let _ = d.write_all(b"mp for planar points only\n");
            let _ = d.flush();
            let _ = d.write_all(c_format("hull-io.c line %d \n", &[CArg::Int(310)]).as_bytes());
            let _ = d.flush();
        }
        if count == 10 {
            let _ = d.write_all(b"consider yourself warned\n");
            let _ = d.flush();
        }
        return;
    }
    if amble == 0 {
        if v.is_empty() {
            return;
        }
        for i in 0..vdim {
            if v[i as usize] == HULL_INFINITY {
                let t = v[i as usize];
                v[i as usize] = v[(vdim - 1) as usize];
                v[(vdim - 1) as usize] = t;
                vdim -= 1;
                break;
            }
        }
        let _ = f.write_all(b"draw ");
        for i in 0..vdim {
            let p = v[i as usize];
            let _ = f.write_all(
                c_format(
                    if i + 1 < vdim {
                        "(%Gu,%Gu)--"
                    } else {
                        "(%Gu,%Gu);\n"
                    },
                    &[
                        CArg::Dbl(st.sites[p] / MULT_UP.get()),
                        CArg::Dbl(st.sites[p + 1] / MULT_UP.get()),
                    ],
                )
                .as_bytes(),
            );
        }
    } else if amble == -1 {
        if FIGNO.get() == 1 {
            let _ = f.write_all(b"u=1pt;\n");
        }
        let _ =
            f.write_all(c_format("beginfig(%d);\n", &[CArg::Int(FIGNO.get() as i64)]).as_bytes());
        FIGNO.set(FIGNO.get() + 1);
    } else if amble == 1 {
        let _ = f.write_all(b"endfig;\n");
    }
}

/// Original `ps_out` (`hull-io.c:336`).
pub fn ps_out(st: &mut HullStorage, v: &mut [Site], mut vdim: i32, f: &mut dyn Write, amble: i32) {
    if PDIM.get() != 2 {
        /* warning(-10, ps for planar points only) (`hull-io.c:343`) */
        PS_MESSCOUNT.set(PS_MESSCOUNT.get() + 1);
        let count = PS_MESSCOUNT.get();
        let mut d = ImodFile::Stderr;
        if count <= 10 {
            let _ = d.write_all(b"ps for planar points only\n");
            let _ = d.flush();
            let _ = d.write_all(c_format("hull-io.c line %d \n", &[CArg::Int(343)]).as_bytes());
            let _ = d.flush();
        }
        if count == 10 {
            let _ = d.write_all(b"consider yourself warned\n");
            let _ = d.flush();
        }
        return;
    }

    if amble == 0 {
        if v.is_empty() {
            return;
        }
        for i in 0..vdim {
            if v[i as usize] == HULL_INFINITY {
                let t = v[i as usize];
                v[i as usize] = v[(vdim - 1) as usize];
                v[(vdim - 1) as usize] = t;
                vdim -= 1;
                break;
            }
        }
        let _ = f.write_all(
            c_format(
                "newpath %G %G moveto\n",
                &[
                    CArg::Dbl(st.sites[v[0]] * SCALER.get()),
                    CArg::Dbl(st.sites[v[0] + 1] * SCALER.get()),
                ],
            )
            .as_bytes(),
        );
        for i in 1..vdim {
            let p = v[i as usize];
            let _ = f.write_all(
                c_format(
                    "%G %G lineto\n",
                    &[
                        CArg::Dbl(st.sites[p] * SCALER.get()),
                        CArg::Dbl(st.sites[p + 1] * SCALER.get()),
                    ],
                )
                .as_bytes(),
            );
        }
        let _ = f.write_all(b"stroke\n");
    } else if amble == -1 {
        let mut len = [0f32; 2];
        let _ = f.write_all(c_format("%%!PS\n", &[]).as_bytes());
        let mins = MINS.get();
        let maxs = MAXS.get();
        len[0] = (maxs[0] - mins[0]) as f32;
        len[1] = (maxs[1] - mins[1]) as f32;
        let maxlen = if len[0] > len[1] { len[0] } else { len[1] };
        SCALER.set(216. / maxlen as f64);

        let _ = f.write_all(
            c_format(
                "%%%%BoundingBox: %G %G %G %G \n",
                &[
                    CArg::Dbl(mins[0] * SCALER.get()),
                    CArg::Dbl(mins[1] * SCALER.get()),
                    CArg::Dbl(maxs[0] * SCALER.get()),
                    CArg::Dbl(maxs[1] * SCALER.get()),
                ],
            )
            .as_bytes(),
        );
        let _ = f.write_all(c_format("%%%%Creator: hull program\n", &[]).as_bytes());
        let _ = f.write_all(c_format("%%%%Pages: 1\n", &[]).as_bytes());
        let _ = f.write_all(c_format("%%%%EndProlog\n", &[]).as_bytes());
        let _ = f.write_all(c_format("%%%%Page: 1 1\n", &[]).as_bytes());
        let _ = f.write_all(b" 0.5 setlinewidth [] 0 setdash\n");
        let _ = f.write_all(b" 1 setlinecap 1 setlinejoin 10 setmiterlimit\n");
    } else if amble == 1 {
        let _ = f.write_all(c_format("showpage\n %%%%EOF\n", &[]).as_bytes());
    }
}

/// Original `cpr_out` (`hull-io.c:387`).
pub fn cpr_out(st: &mut HullStorage, v: &mut [Site], vdim: i32, f: &mut dyn Write, _amble: i32) {
    if PDIM.get() != 3 {
        /* warning(-10, cpr for 3d points only) (`hull-io.c:394`) */
        CPR_MESSCOUNT.set(CPR_MESSCOUNT.get() + 1);
        let count = CPR_MESSCOUNT.get();
        let mut d = ImodFile::Stderr;
        if count <= 10 {
            let _ = d.write_all(b"cpr for 3d points only\n");
            let _ = d.flush();
            let _ = d.write_all(c_format("hull-io.c line %d \n", &[CArg::Int(394)]).as_bytes());
            let _ = d.flush();
        }
        if count == 10 {
            let _ = d.write_all(b"consider yourself warned\n");
            let _ = d.flush();
        }
        return;
    }

    for i in 0..vdim {
        if v[i as usize] == HULL_INFINITY {
            return;
        }
    }

    let m = MULT_UP.get();
    let _ = f.write_all(
        c_format(
            "t %G %G %G %G %G %G %G %G %G 3 128\n",
            &[
                CArg::Dbl(st.sites[v[0]] / m),
                CArg::Dbl(st.sites[v[0] + 1] / m),
                CArg::Dbl(st.sites[v[0] + 2] / m),
                CArg::Dbl(st.sites[v[1]] / m),
                CArg::Dbl(st.sites[v[1] + 1] / m),
                CArg::Dbl(st.sites[v[1] + 2] / m),
                CArg::Dbl(st.sites[v[2]] / m),
                CArg::Dbl(st.sites[v[2] + 1] / m),
                CArg::Dbl(st.sites[v[2] + 2] / m),
            ],
        )
        .as_bytes(),
    );
}

/* vist_funcs for different kinds of output: facets, alpha shapes, etc. */

/// Original `facets_print` (`hull-io.c:413`).
///
/// DNM: Fix argument type and avoid use of `p` for arg since it is global.
pub fn facets_print(st: &mut HullStorage, s: usize, out_func_here: OutFunc) -> usize {
    let mut v: [Site; MAXDIM] = [0; MAXDIM];
    let cdim = CDIM.get();

    for j in 0..cdim {
        v[j as usize] = st.simplex[s].neigh[j as usize].vert;
    }

    out_func_here(st, &mut v, cdim, 0);

    0
}

/// Original `ridges_print` (`hull-io.c:429`).
pub fn ridges_print(st: &mut HullStorage, s: usize, out_func_here: OutFunc) -> usize {
    let mut v: [Site; MAXDIM] = [0; MAXDIM];
    let cdim = CDIM.get();

    for j in 0..cdim {
        let mut vnum = 0;
        for k in 0..cdim {
            if k == j {
                continue;
            }
            v[vnum] = st.simplex[s].neigh[k as usize].vert;
            vnum += 1;
        }
        out_func_here(st, &mut v, cdim - 1, 0);
    }
    0
}

/// Original `afacets_print` (`hull-io.c:450`).
pub fn afacets_print(st: &mut HullStorage, s: usize, out_func_here: OutFunc) -> usize {
    let mut v: [Site; MAXDIM] = [0; MAXDIM];
    let cdim = CDIM.get();

    for j in 0..cdim {
        /* check for ashape consistency */
        let other = st.simplex[s].neigh[j as usize].simp;
        let mut k = 0;
        while k < cdim {
            if st.simplex[other].neigh[k as usize].simp == s {
                break;
            }
            k += 1;
        }
        if alph_test(st, s, j, None) != alph_test(st, other, k, None) {
            /* DEB(-10, alpha-shape not consistent) DEBTR(-10) -- both live. */
            let mut d = ImodFile::Stderr;
            let _ = d.write_all(b"alpha-shape not consistent\n");
            let _ = d.flush();
            let _ = d.write_all(c_format("hull-io.c line %d \n", &[CArg::Int(462)]).as_bytes());
            let _ = d.flush();
            print_simplex_f(st, s, &mut d, Some(print_neighbor_full));
            print_simplex_f(st, other, &mut d, Some(print_neighbor_full));
            let _ = d.flush();
            std::process::exit(1);
        }
    }
    for j in 0..cdim {
        let mut vnum = 0;
        if alph_test(st, s, j, None) != 0 {
            continue;
        }
        for k in 0..cdim {
            if k == j {
                continue;
            }
            v[vnum] = st.simplex[s].neigh[k as usize].vert;
            vnum += 1;
        }
        out_func_here(st, &mut v, cdim - 1, 0);
    }
    0
}
