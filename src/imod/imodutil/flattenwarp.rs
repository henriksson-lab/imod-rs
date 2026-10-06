//! Translation of `IMOD/imodutil/flattenwarp.c` -- computes warping
//! transforms to flatten a volume.
//!
//! The program maps to [`flattenwarp`]; each static function of the unit has
//! its own function.  Ownership notes, forced by the C's pointers:
//!
//! * `ContData` points at model contours and later at new resampled ones;
//!   here it owns its contours ([`ContData::cont`], `cont2`, `cdup`), copied
//!   from the model after the in-place sorting and merging the source applies
//!   to them, which the model never reads again.
//! * The sparse matrix for `lsqr` is the C's single `int` work array
//!   (`libcfshr/sparselsqr.rs`); `ia`, `ja` and the float values are its three
//!   disjoint parts.
//! * `interpolateCont2`'s static arrays are per-call vectors (only their
//!   allocation is shared in C).
//! * The thin plate spline solve is LAPACK `dsysv` (`flib/subrs/lapack/
//!   dsysv.rs`, `faer`); results that pass through it carry the CLAUDE.md
//!   LAPACK tolerance.

use crate::imod::flib::subrs::lapack::dsysv::dsysv;
use crate::imod::libcfshr::b3dutil::{
    CArg, ImodFile, c_format, c_format_bytes, exit, imod_backup_file, imod_prog_name,
    imod_usage_header, program_args,
};
use crate::imod::libcfshr::convexbound::convex_bound;
use crate::imod::libcfshr::lsqr::lsqr;
use crate::imod::libcfshr::parse_params::{
    exit_error, pip_get_boolean, pip_get_float, pip_get_float_array, pip_get_in_out_file,
    pip_get_integer, pip_get_string, pip_get_two_floats, pip_get_two_integers, pip_print_help,
    pip_read_or_parse_options,
};
use crate::imod::libcfshr::robuststat::{rs_madn, rs_median};
use crate::imod::libcfshr::simplestat::ls_fit2;
use crate::imod::libcfshr::sparselsqr::{
    add_row_to_matrix, add_value_to_row, normalize_columns, sparse_prod,
};
use crate::imod::libimod::icont::{
    imod_contour_area, imod_contour_dup, imod_contour_get_bbox, imod_contour_new,
    imodel_contour_scan, imodel_contour_sortx,
};
use crate::imod::libimod::imat::{
    imod_mat_id, imod_mat_new, imod_mat_rotate_vector, imod_mat_transform3d,
};
use crate::imod::libimod::imesh::{IMESH_MK_SKIP, imesh_params_new};
use crate::imod::libimod::imodel::{
    IMODF_FLIPYZ, Icont, Imod, Iobj, Ipoint, imod_flip_yz, imod_new, imod_new_object,
};
use crate::imod::libimod::imodel_files::{imod_read, imod_write};
use crate::imod::libimod::iobj::{
    IMOD_OBJFLAG_FILL, IMOD_OBJFLAG_MESH, IMOD_OBJFLAG_NOLINE, IMOD_OBJFLAG_TWO_SIDE,
    IOBJ_FLAG_CLOSED, imod_object_add_contour, imod_object_copy, imod_object_dup,
    imod_object_get_bbox, imod_object_set_value, iobj_scat,
};
use crate::imod::libimod::imodel::IMOD_OBJFLAG_OFF;
use crate::imod::libimod::ipoint::{imod_point_append, imod_point_append_xyz, imod_point_delete};
use crate::imod::libmesh::objprep::analyze_prep_skin_obj;
use std::io::Write;

/// Structure for storing contour data (`flattenwarp.c:31`).
#[derive(Clone, Default)]
struct ContData {
    cont: Icont,
    yval: i32,
    cont2: Option<Icont>,
    xmin: f32,
    xmax: f32,
    cdup: Option<Icont>,
}

/// Structure for storing warping data (`flattenwarp.c:41`).
#[derive(Clone, Copy, Default)]
struct WarpData {
    xpos: f32,
    ypos: f32,
    aa: f32,
    bb: f32,
    dx: f32,
    dy: f32,
    dz: f32,
}

/// `#define APPEND_ERROR` (`flattenwarp.c:82`).
const APPEND_ERROR: &[u8] = b"Adding point to contour for output model";
/// `#define ADDCONT_ERROR`.
const ADDCONT_ERROR: &[u8] = b"Error adding new contour to model for middle contours";
/// `#define MAX_LAMBDAS 100`.
const MAX_LAMBDAS: usize = 100;

/// `exitError` with the source's variadic format.
macro_rules! exit_error_fmt {
    ($fmt:expr $(, $arg:expr)* $(,)?) => {
        exit_error(&c_format_bytes($fmt, &[$($arg),*]))
    };
}

/// `printf` with the source's format, to stdout.
macro_rules! printf {
    ($fmt:expr $(, $arg:expr)* $(,)?) => {{
        let _ = ImodFile::Stdout.write_all(c_format($fmt, &[$($arg),*]).as_bytes());
    }};
}

/// `B3DNINT(a)`: `(int)floor((a) + 0.5)`.
macro_rules! b3dnint {
    ($a:expr) => {
        (($a) as f64 + 0.5).floor() as i32
    };
}

/// `B3DMAX(a,b)`: `((a) > (b) ? (a) : (b))`.
macro_rules! b3dmax {
    ($a:expr, $b:expr) => {{
        let (a, b) = ($a, $b);
        if a > b { a } else { b }
    }};
}

/// `B3DMIN(a,b)`: `((a) < (b) ? (a) : (b))`.
macro_rules! b3dmin {
    ($a:expr, $b:expr) => {{
        let (a, b) = ($a, $b);
        if a < b { a } else { b }
    }};
}

/// `NEWCONTOUR(a)`.
fn new_contour() -> Icont {
    match imod_contour_new() {
        Some(cont) => cont,
        None => exit_error(b"Allocating new contour"),
    }
}

/// The `PipReadOrParseOptions` header callback, `imodUsageHeader`.
fn imod_usage_header_for_pip(prog_name: &[u8]) {
    imod_usage_header(Some(&String::from_utf8_lossy(prog_name)));
    let _ = ImodFile::Stdout.flush();
}

/// Original `main` (`flattenwarp.c:91`).
pub fn flattenwarp() {
    let argv = program_args();
    let argv_bytes: Vec<Vec<u8>> = argv.iter().map(|a| a.as_bytes().to_vec()).collect();
    let progname = imod_prog_name(argv.first().map(String::as_str).unwrap_or(""));
    let mut filename: Vec<u8> = Vec::new();
    let (mut num_opt_args, mut num_non_opt_args) = (0_i32, 0_i32);

    /* Fallbacks from    ../manpages/autodoc2man 2 1 flattenwarp  */
    let num_options = 12;
    let options: [&[u8]; 12] = [
        b"input:InputFile:FN:",
        b"output:OutputFile:FN:",
        b"patch:PatchOutputFile:FN:",
        b"middle:MiddleContourFile:FN:",
        b"binning:BinningOfTomogram:IP:",
        b"one:OneSurface:B:",
        b"flip:FlipOption:I:",
        b"spacing:WarpSpacingXandY:FP:",
        b"lambda:LambdaForSmoothing:FA:",
        b"show:ShowContours:B:",
        b"restore:RestoreOrientation:B:",
        b":PID:B:",
    ];

    /* Maximum locations to output.  It could be 200000, but limit it to keep
    linear equation system from getting truly enormous */
    let max_locations = 50000_i32;
    let mut x_spacing = 0.0_f32;
    let mut y_spacing = 0.0_f32;
    let (x_space_fac, y_space_fac, resample_xy_fac) = (4.5_f32, 3.0_f32, 0.67_f32);
    let resamp_window_fac = 1.5_f32;
    let (mut xy_binning, mut z_binning) = (1_i32, 1_i32);
    let mut one_surface = 0_i32;
    let mut flipyz = -1_i32;
    let mut flipped = 0_i32;
    let mut restore_orientation = 0_i32;
    let mut pid = 0_i32;
    let mut scattered = 0_i32;
    let mut showcont = 0_i32;
    let drop_outliers: i32;
    let mut crit_madn = 2.5_f32;
    let min_num_scat = 6_i32;
    let scat_space_fac = 0.33_f32;
    let frac_omit = 0.0_f32;
    let mut clist: Vec<ContData> = Vec::with_capacity(1000);
    let mut patchfile: Option<Vec<u8>> = None;
    let mut midfile: Option<Vec<u8>> = None;
    let mut name_prefix: [&str; 3] = [""; 3];
    let top_prefix = "Top, ";
    let bot_prefix = "Bottom, ";
    let mid_prefix = "Middle, ";
    let empty_string = "";
    let mut warps: Vec<WarpData>;
    let mut ind_warp: Vec<i32>;
    let mut ptadd = Ipoint::default();
    let mut axisvec: Ipoint;
    let scale = Ipoint {
        x: 1.,
        y: 1.,
        z: 10.,
    };
    let Some(mut mat) = imod_mat_new(3) else {
        exit_error(b"Allocating matrix")
    };
    let mut xp = [0.0_f32; 9];
    let mut yp = [0.0_f32; 9];
    let mut zp = [0.0_f32; 9];
    let mut val_row = [0.0_f32; 5];
    let mut icol_row = [0_i32; 5];
    let mut lambda = [0.0_f32; MAX_LAMBDAS];
    let mut num_lambdas = 0_i32;
    let mut num_scat = [0_i32; 2];
    let mut num_scat_fit = [0_i32; 2];
    let mut scat_pts: [Vec<Ipoint>; 2] = [Vec::new(), Vec::new()];
    let mut scat_vec: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
    let mut scat_scl = [Ipoint::default(); 2];
    let mut scan_cont: [Option<Icont>; 2] = [None, None];
    let mut mean_zscat = [0.0_f32; 2];
    let mut fit_pts: Vec<Ipoint> = Vec::new();
    let mut yvec: Vec<f64> = Vec::new();
    let mut tps_scl = Ipoint::default();
    let (mut xmin, mut xmax, mut ymin, mut ymax): (f32, f32, f32, f32);
    let (mut zval, mut zval2): (f32, f32) = (0., 0.);
    let (mut xcen, mut ycen, mut zcen) = (0.0_f32, 0.0_f32, 0.0_f32);
    let (mut num_xloc, mut num_yloc) = (0_i32, 0_i32);
    let mut num_loc: i32;
    let mut zsum: f64 = 0.;
    let mut found: i32;
    let mut num_points: i32;
    let mut err: i32;
    let mut i: i32;
    let mut model: Imod;
    let mut midmod: Option<Imod> = None;

    /* Startup with fallback */
    pip_read_or_parse_options(
        argv_bytes.len() as i32,
        &argv_bytes,
        &options,
        num_options,
        progname.as_bytes(),
        2,
        1,
        1,
        &mut num_opt_args,
        &mut num_non_opt_args,
        Some(imod_usage_header_for_pip),
    );
    let mut usage = 0;
    if pip_get_boolean(b"usage", &mut usage) == 0 {
        pip_print_help(progname.as_bytes(), 0, 1, 1);
        exit(0);
    }

    /* Get input and output files */
    if pip_get_in_out_file(b"InputFile", 0, &mut filename) != 0 {
        exit_error(b"No input file specified");
    }

    {
        let name = String::from_utf8_lossy(&filename).into_owned();
        model = match imod_read(&name) {
            Ok(model) => model,
            Err(_) => exit_error_fmt!("Reading model %s", CArg::Str(&name)),
        };
    }

    let _ = pip_get_float_array(
        b"LambdaForSmoothing",
        &mut lambda,
        &mut num_lambdas,
        MAX_LAMBDAS as i32,
    );
    filename.clear();
    if pip_get_in_out_file(b"OutputFile", 1, &mut filename) != 0 && num_lambdas < 2 {
        exit_error(b"No output file specified");
    }
    let filename = String::from_utf8_lossy(&filename).into_owned();

    /* Process options */
    let _ = pip_get_two_floats(b"WarpSpacingXandY", &mut x_spacing, &mut y_spacing);
    let _ = pip_get_two_integers(b"BinningOfTomogram", &mut xy_binning, &mut z_binning);
    let _ = pip_get_boolean(b"OneSurface", &mut one_surface);
    let _ = pip_get_integer(b"FlipOption", &mut flipyz);
    {
        let mut name: Vec<u8> = Vec::new();
        if pip_get_string(b"PatchOutputFile", &mut name) == 0 {
            patchfile = Some(name);
        }
        let mut name: Vec<u8> = Vec::new();
        if pip_get_string(b"MiddleContourFile", &mut name) == 0 {
            midfile = Some(name);
        }
    }
    let _ = pip_get_boolean(b"ShowContour", &mut showcont);
    let _ = pip_get_boolean(b"RestoreOrientation", &mut restore_orientation);
    drop_outliers = 1 - pip_get_float(b"CriterionForOutliers", &mut crit_madn);
    let _ = pip_get_boolean(b"PID", &mut pid);
    if pid != 0 {
        eprintln!("Shell PID: {}", std::process::id());
        let _ = std::io::stderr().flush();
    }

    if num_lambdas > 1 && midfile.is_none() {
        exit_error(
            b"You must enter an output model file for smoothed contours if entering multiple lambdas",
        );
    }

    /* Start output model now before splitting out on type of model */
    if midfile.is_some() {
        midmod = imod_new();
        if midmod.is_none() {
            exit_error(b"Error creating new model for middle contours");
        }
    }
    let new_object = |midmod: &mut Option<Imod>| {
        if imod_new_object(midmod.as_mut().unwrap()) != 0 {
            exit_error(b"Error creating new object in middle contour model");
        }
    };

    /* Determine if scattered point objects are present */
    if iobj_scat(model.obj[0].flags) != 0 {
        /* THE MODEL HAS SCATTERED POINTS */

        scattered = 1;
        let num_obj = model.obj.len() as i32;
        if num_obj > 2 {
            exit_error(b"A model with scattered points must have either one or two objects");
        }
        if num_obj > 1 && iobj_scat(model.obj[1].flags) == 0 {
            exit_error(b"The first object is scattered points but the second is not");
        }
        if num_lambdas == 0 {
            exit_error(b"You must specify a lambda for smoothing with scattered points");
        }

        /* Manage the flip/rotation state of model: first flip to native coords */
        if model.flags & IMODF_FLIPYZ != 0 {
            imod_flip_yz(&mut model);
        }

        /* Then rotate if necessary unless user directs otherwise */
        if (flipyz < 0 && model.zmax > model.ymax) || flipyz > 0 {
            if flipyz == 1 {
                imod_flip_yz(&mut model);
                flipped = 1;
            } else {
                rotate_model(&mut model, -1);
                flipped = 2;
            }
        }

        /* Count points in each object and get bounding box for min/max */
        num_points = 0;
        let mut max_points = 0_i32;
        xmin = -1.0e20;
        ymin = -1.0e20;
        xmax = -ymin;
        ymax = -ymin;
        for ob in 0..num_obj as usize {
            let (mut minpt, mut maxpt) = (Ipoint::default(), Ipoint::default());
            imod_object_get_bbox(&model.obj[ob], &mut minpt, &mut maxpt);
            xmin = b3dmax!(xmin, minpt.x);
            xmax = b3dmin!(xmax, maxpt.x);
            ymin = b3dmax!(ymin, minpt.y);
            ymax = b3dmin!(ymax, maxpt.y);
            num_scat[ob] = 0;
            for co in 0..model.obj[ob].cont.len() {
                num_scat[ob] += model.obj[ob].cont[co].pts.len() as i32;
            }
            num_points += num_scat[ob];
            max_points = b3dmax!(max_points, num_scat[ob]);
            if num_scat[ob] < min_num_scat {
                exit_error_fmt!(
                    "There must be at least %d points in each scattered point object",
                    CArg::Int(min_num_scat as i64)
                );
            }
        }

        /* Copy points into arrays for convex bound */
        let mut bound_area = 0.0_f32;
        let mut bx = vec![0.0_f32; max_points as usize];
        let mut by = vec![0.0_f32; max_points as usize];
        for ob in 0..num_obj as usize {
            let mut scx = vec![0.0_f32; num_scat[ob] as usize];
            let mut scy = vec![0.0_f32; num_scat[ob] as usize];
            let mut i = 0;
            mean_zscat[ob] = 0.;
            for cont in &model.obj[ob].cont {
                for pt in &cont.pts {
                    scx[i] = pt.x;
                    scy[i] = pt.y;
                    i += 1;
                    mean_zscat[ob] += pt.z / num_scat[ob] as f32;
                }
            }

            /* Get convex boundary and convert to a scan contour */
            let pad = (0.5
                * (((xmax - xmin) as f64) * (ymax - ymin) as f64 / num_scat[ob] as f64).sqrt())
                as f32;
            let mut num_bound = 0_i32;
            let (mut xcenpts, mut ycenpts) = (0.0_f32, 0.0_f32);
            convex_bound(
                &scx,
                &scy,
                frac_omit,
                pad,
                &mut bx[..num_scat[ob] as usize],
                &mut by[..num_scat[ob] as usize],
                &mut num_bound,
                &mut xcenpts,
                &mut ycenpts,
            );
            let mut cont = new_contour();
            cont.pts = (0..num_bound as usize)
                .map(|i| Ipoint {
                    x: bx[i],
                    y: by[i],
                    z: 0.,
                })
                .collect();
            scan_cont[ob] = imodel_contour_scan(Some(&cont));
            if scan_cont[ob].is_none() {
                exit_error(b"Getting scan contour from boundary contour");
            }
            bound_area += imod_contour_area(Some(&cont));
            /* Leave bx, by for outlier detection */
        }

        if x_spacing != 0. && y_spacing != 0. {
            if (1. + ((ymax - ymin) / y_spacing) as f64) * (1. + ((xmax - xmin) / x_spacing) as f64)
                >= max_locations as f64
            {
                exit_error(b"Your spacings between grid points are too small for warpvol");
            }
            found = 0;
        } else {
            /* Set spacing if not specified */
            x_spacing = (scat_space_fac as f64 * (bound_area as f64 / num_points as f64).sqrt())
                as f32;
            y_spacing = x_spacing;
            found = 1;
            if (1. + ((ymax - ymin) / y_spacing) as f64) * (1. + ((xmax - xmin) / x_spacing) as f64)
                >= max_locations as f64
            {
                // `(ymax + 2. * ySpacing - ymin)` is double, `(xmax + ySpacing
                // - xmin)` float.
                x_spacing = (((ymax as f64 + 2. * y_spacing as f64 - ymin as f64)
                    * (xmax + y_spacing - xmin) as f64)
                    / max_locations as f64)
                    .sqrt() as f32;
                y_spacing = x_spacing;
                printf!(
                    "Setting grid spacing to %.1f based on maximum allowed grid points\n",
                    CArg::Dbl(x_spacing as f64)
                );
            } else {
                printf!(
                    "Setting grid spacing to %.1f based on mean distance between points\n",
                    CArg::Dbl(x_spacing as f64)
                );
            }
        }

        (warps, ind_warp) = setup_warp_grid(
            xmin,
            xmax,
            ymin,
            ymax,
            xy_binning,
            z_binning,
            &model,
            &mut x_spacing,
            &mut y_spacing,
            &mut num_xloc,
            &mut num_yloc,
            &mut xcen,
            &mut ycen,
            &mut zcen,
        );
        if num_xloc < 3 || num_yloc < 3 {
            exit_error_fmt!(
                "The number of grid positions is too small; %s",
                CArg::Str(if found != 0 {
                    "there are too few data points"
                } else {
                    "try reducing the grid spacing"
                })
            );
        }

        /* Set up the warp positions and determine whether each is feasible */

        num_loc = 0;
        for j in 0..num_yloc {
            let yloc = ymin + j as f32 * y_spacing;
            let mut xleft = -1.0e20_f32;
            let mut xright = 1.0e20_f32;
            let iyval = b3dnint!(yloc);
            for ob in 0..num_obj as usize {
                /* Find extent of scan contours at this Y value */
                let mut scmin = 1.0e20_f32;
                let mut scmax = -1.0e20_f32;
                for pt in &scan_cont[ob].as_ref().unwrap().pts {
                    if b3dnint!(pt.y) == iyval {
                        scmin = b3dmin!(scmin, pt.x);
                        scmax = b3dmax!(scmax, pt.x);
                    }
                }

                /* AND the scan contour extents if there are two */
                xleft = b3dmax!(xleft, scmin);
                xright = b3dmin!(xright, scmax);
            }
            for i in 0..num_xloc {
                let xloc = xmin + i as f32 * x_spacing;
                if xloc >= xleft && xloc <= xright {
                    let w = &mut warps[num_loc as usize];
                    w.xpos = xloc * xy_binning as f32 - xcen;
                    w.ypos = yloc * xy_binning as f32 - ycen;

                    w.aa = -999.;
                    ind_warp[(i + j * num_xloc) as usize] = num_loc;
                    num_loc += 1;
                } else {
                    ind_warp[(i + j * num_xloc) as usize] = -1;
                }
            }
        }

        /* If doing output model, duplicate the point objects */
        if midfile.is_some() {
            for ob in 0..num_obj as usize {
                new_object(&mut midmod);
                let Some(mut obj) = imod_object_dup(&model.obj[ob]) else {
                    exit_error(b"Duplicating scattered point object for output model")
                };
                obj.flags |= IMOD_OBJFLAG_MESH | IMOD_OBJFLAG_NOLINE | IMOD_OBJFLAG_FILL;
                let name = if num_obj == 1 {
                    "Bead positions".to_string()
                } else {
                    format!(
                        "{} bead positions",
                        if mean_zscat[ob] > mean_zscat[1 - ob] {
                            "Top"
                        } else {
                            "Bottom"
                        }
                    )
                };
                obj.name.fill(0);
                obj.name[..name.len()].copy_from_slice(name.as_bytes());
                imod_object_copy(&obj, &mut midmod.as_mut().unwrap().obj[ob]);
            }
            name_prefix[0] = if num_obj > 1 { mid_prefix } else { empty_string };
            if num_obj > 1 {
                name_prefix[1] = if mean_zscat[0] > mean_zscat[1] {
                    top_prefix
                } else {
                    bot_prefix
                };
                name_prefix[2] = if mean_zscat[0] > mean_zscat[1] {
                    bot_prefix
                } else {
                    top_prefix
                };
            }

            /* And make all the other objects */
            for _ in 0..num_lambdas * (1 + 2 * (num_obj - 1)) {
                new_object(&mut midmod);
            }
            midmod.as_mut().unwrap().cindex.object = num_obj;
        }

        /* Now loop on TPS fits */
        for indc in 0..num_lambdas as usize {
            for ob in 0..num_obj as usize {
                /* Get the arrays for this number of points */
                let (mut lmat, mut ipiv, mut work, lwork);
                (scat_pts[ob], scat_vec[ob], lmat, ipiv, work, lwork) =
                    prepare_tps(num_scat[ob]);

                /* Load the points into fit point array */
                let mut j = 0;
                for cont in &model.obj[ob].cont {
                    for pt in &cont.pts {
                        scat_pts[ob][j] = *pt;
                        j += 1;
                    }
                }
                set_tps_scaling(&scat_pts[ob], num_scat[ob], &mut scat_scl[ob]);

                printf!(
                    "Fitting thin plate spline to obj %d, log lambda = %.2f:",
                    CArg::Int(ob as i64 + 1),
                    CArg::Dbl(lambda[indc] as f64)
                );

                /* If doing outliers, loop twice, eliminate outliers from array the
                first time */
                num_scat_fit[ob] = num_scat[ob];
                for lp in 0..=drop_outliers {
                    fit_tps(
                        &scat_pts[ob],
                        num_scat_fit[ob],
                        lambda[indc],
                        &scat_scl[ob],
                        &mut scat_vec[ob],
                        &mut lmat,
                        &mut ipiv,
                        &mut work,
                        lwork,
                    );
                    let mut alpha = 0.0_f32;
                    for j in 0..num_scat[ob] as usize {
                        zval = evaluate_tps(
                            scat_pts[ob][j].x,
                            scat_pts[ob][j].y,
                            &scat_pts[ob],
                            num_scat_fit[ob],
                            &scat_scl[ob],
                            &scat_vec[ob],
                        );
                        bx[j] = zval - scat_pts[ob][j].z;
                        alpha = (alpha as f64 + ((zval - scat_pts[ob][j].z) as f64).abs()) as f32;
                    }
                    printf!(
                        " Mean deviation = %.3f\n",
                        CArg::Dbl((alpha / num_scat[ob] as f32) as f64)
                    );

                    let mut median_dev = 0.0_f32;
                    let mut madn = 0.0_f32;
                    rs_median(&bx, num_scat_fit[ob], &mut by, &mut median_dev);
                    rs_madn(&bx, num_scat_fit[ob], median_dev, &mut by, &mut madn);
                    if lp == 0 && drop_outliers != 0 {
                        let mut j = num_scat[ob] - 1;
                        while j >= 0 {
                            if ((bx[j as usize] - median_dev) as f64).abs() / madn as f64
                                > crit_madn as f64
                            {
                                for i in (j + 1) as usize..num_scat_fit[ob] as usize {
                                    scat_pts[ob][i - 1] = scat_pts[ob][i];
                                }
                                num_scat_fit[ob] -= 1;
                            }
                            j -= 1;
                        }
                        printf!(
                            "             Dropped %4d of %5d points as outliers:",
                            CArg::Int((num_scat[ob] - num_scat_fit[ob]) as i64),
                            CArg::Int(num_scat[ob] as i64)
                        );
                    }
                }
            }

            /* Get the Z value at the warp positions */
            zsum = 0.;
            for j in 0..num_yloc {
                let yloc = ymin + j as f32 * y_spacing;
                let mut cont: Option<Icont> = None;
                let mut cont1: Option<Icont> = None;
                let mut cont2: Option<Icont> = None;
                if midfile.is_some() {
                    cont = Some(new_contour());
                    if num_obj > 1 {
                        cont1 = Some(new_contour());
                        cont2 = Some(new_contour());
                    }
                }
                for i in 0..num_xloc {
                    let xloc = xmin + i as f32 * x_spacing;
                    let ind = ind_warp[(i + j * num_xloc) as usize];
                    if ind >= 0 {
                        zval = 0.;
                        for ob in 0..num_obj as usize {
                            zval2 = evaluate_tps(
                                xloc,
                                yloc,
                                &scat_pts[ob],
                                num_scat_fit[ob],
                                &scat_scl[ob],
                                &scat_vec[ob],
                            );
                            zval += zval2 / num_obj as f32;
                            if ob == 0 {
                                if let Some(c) = cont1.as_mut() {
                                    if imod_point_append_xyz(c, xloc, yloc, zval2) == 0 {
                                        exit_error(APPEND_ERROR);
                                    }
                                }
                            }
                            if ob != 0 {
                                if let Some(c) = cont2.as_mut() {
                                    if imod_point_append_xyz(c, xloc, yloc, zval2) == 0 {
                                        exit_error(APPEND_ERROR);
                                    }
                                }
                            }
                        }
                        warps[ind as usize].dz = z_binning as f32 * zval;
                        zsum += warps[ind as usize].dz as f64;
                        if let Some(c) = cont.as_mut() {
                            if imod_point_append_xyz(c, xloc, yloc, zval) == 0 {
                                exit_error(APPEND_ERROR);
                            }
                        }
                    }
                }

                /* Add contours to objects */
                if let Some(mm) = midmod.as_mut() {
                    let c = cont.take().unwrap();
                    if !c.pts.is_empty()
                        && imod_object_add_contour(&mut mm.obj[indc + num_obj as usize], c) < 0
                    {
                        exit_error(ADDCONT_ERROR);
                    }
                    if num_obj > 1 {
                        let c1 = cont1.take().unwrap();
                        if !c1.pts.is_empty()
                            && imod_object_add_contour(
                                &mut mm.obj[num_obj as usize + indc + num_lambdas as usize],
                                c1,
                            ) < 0
                        {
                            exit_error(ADDCONT_ERROR);
                        }
                        let c2 = cont2.take().unwrap();
                        if !c2.pts.is_empty()
                            && imod_object_add_contour(
                                &mut mm.obj[num_obj as usize + indc + 2 * num_lambdas as usize],
                                c2,
                            ) < 0
                        {
                            exit_error(ADDCONT_ERROR);
                        }
                    }
                }
            }
            if let Some(mm) = midmod.as_mut() {
                for ob in 0..(1 + 2 * (num_obj - 1)) as usize {
                    let obj = &mut mm.obj[num_obj as usize + indc + ob * num_lambdas as usize];
                    adjust_and_mesh_obj(obj, lambda[indc], &scale, showcont, name_prefix[ob]);
                    if indc == 0 {
                        obj.flags &= !IMOD_OBJFLAG_OFF;
                    }
                }
            }
        }

        /* Close up the output model file now and exit if multiple lambdas */
        finish_output_model(
            midfile.as_deref(),
            midmod.as_mut(),
            &mut model,
            &scale,
            flipped,
            num_lambdas,
            restore_orientation,
        );
    } else {
        /* THE MODEL HAS BOUNDARY CONTOURS */

        /* Flip model so that z is depth unless user directs otherwise */
        if (flipyz < 0 && model.zmax > model.ymax) || flipyz > 0 {
            imod_flip_yz(&mut model);
            flipped = 1;
        }

        ymin = 1.0e20;
        ymax = -ymin;
        /* Make list of contours to be used */
        for ob in 0..model.obj.len() {
            for co in 0..model.obj[ob].cont.len() {
                let cont = &mut model.obj[ob].cont[co];
                if cont.pts.len() < 2 {
                    continue;
                }

                /* Make sure contour is planar and get x min and max */
                let iyval = b3dnint!(cont.pts[0].y);
                let mut planar = 1;
                let mut cxmin = cont.pts[0].x;
                let mut cxmax = cxmin;
                for pt in 1..cont.pts.len() {
                    if b3dnint!(cont.pts[pt].y) != iyval {
                        planar = 0;
                    }
                    cxmin = b3dmin!(cxmin, cont.pts[pt].x);
                    cxmax = b3dmax!(cxmax, cont.pts[pt].x);
                }
                if planar == 0 {
                    printf!(
                        "\nWARNING: Obj %d, cont %d, at Y = %d, is not planar and is being ignored\n\n",
                        CArg::Int(ob as i64 + 1),
                        CArg::Int(co as i64 + 1),
                        CArg::Int(iyval as i64 + 1)
                    );
                    continue;
                }

                /* Sort in X and remove points at same X */
                let last = cont.pts.len() as i32 - 1;
                let _ = imodel_contour_sortx(cont, 0, last);
                let mut i = 0usize;
                while i + 1 < cont.pts.len() {
                    if cont.pts[i + 1].x - cont.pts[i].x < 1.0e-3 {
                        cont.pts[i].z = ((cont.pts[i].z + cont.pts[i + 1].z) as f64 / 2.) as f32;
                        let _ = imod_point_delete(cont, i as i32 + 1);
                    } else {
                        i += 1;
                    }
                }

                /* Check for other contour at this Y value */
                found = 0;
                for cdptr in clist.iter_mut() {
                    if cdptr.yval == iyval {
                        if cdptr.cont2.is_some() {
                            exit_error_fmt!(
                                "There seem to be 3 contours at Y = %d",
                                CArg::Int(iyval as i64 + 1)
                            );
                        }
                        if one_surface != 0 {
                            exit_error_fmt!(
                                "You specified one surface and there seem to be 2 contours at Y = %d",
                                CArg::Int(iyval as i64 + 1)
                            );
                        }
                        if cxmin >= cdptr.xmax || cxmax <= cdptr.xmin {
                            exit_error_fmt!(
                                "Two contours at Y = %d do not overlap enough in X",
                                CArg::Int(iyval as i64 + 1)
                            );
                        }
                        found = 1;
                        cdptr.xmin = b3dmax!(cxmin, cdptr.xmin);
                        cdptr.xmax = b3dmin!(cxmax, cdptr.xmax);
                        cdptr.cont2 = Some(cont.clone());
                        break;
                    }
                }

                if found == 0 {
                    clist.push(ContData {
                        xmin: cxmin,
                        xmax: cxmax,
                        cont: cont.clone(),
                        cont2: None,
                        yval: iyval,
                        cdup: None,
                    });
                    ymin = b3dmin!(ymin, iyval as f32);
                    ymax = b3dmax!(ymax, iyval as f32);
                }
            }
        }

        if clist.len() < 2 {
            exit_error(b"You must enter contours at more than one Y level");
        }

        /* Sort contours in Y */
        for i in 0..clist.len() - 1 {
            for j in i + 1..clist.len() {
                if clist[i].yval > clist[j].yval {
                    clist.swap(i, j);
                }
            }
        }

        /* Resample the contours at something close to the local Y spacing and
        reduce two contours to one */
        xmin = 1.0e20;
        xmax = -xmin;
        num_points = 0;
        let nlist = clist.len() as i32;
        for i in 0..nlist {
            xmin = b3dmin!(xmin, clist[i as usize].xmin);
            xmax = b3dmax!(xmax, clist[i as usize].xmax);

            /* Get Y spacing from 4 adjacent contours, set X sample & window size */
            let mut iy = b3dmax!(i - 2, 0);
            let j = b3dmin!(iy + 4, nlist - 1);
            iy = b3dmax!(j - 4, 0);
            let mut local_yspace = clist[j as usize].yval as f32;
            local_yspace = (local_yspace - clist[iy as usize].yval as f32) / (j - iy) as f32;
            let cdptr = &mut clist[i as usize];
            let num_bound = (((cdptr.xmax - cdptr.xmin) / (resample_xy_fac * local_yspace)) as f64)
                .ceil() as i32
                + 1;
            let resamp_x = (cdptr.xmax - cdptr.xmin) / (num_bound - 1) as f32;
            let window_x = resamp_x * resamp_window_fac;

            /* Make a new contour */
            let mut ncont = new_contour();

            for j in 0..num_bound {
                ptadd.x = b3dmin!(cdptr.xmin + j as f32 * resamp_x, cdptr.xmax);
                ptadd.y = cdptr.yval as f32;
                interpolate_cont2(&cdptr.cont, ptadd.x, window_x, &mut ptadd.z);
                if let Some(c2) = cdptr.cont2.as_ref() {
                    interpolate_cont2(c2, ptadd.x, window_x, &mut zval);
                    ptadd.z = ((ptadd.z + zval) as f64 / 2.) as f32;
                }
                if imod_point_append(&mut ncont, ptadd) == 0 {
                    exit_error(b"Adding point to average contour");
                }
            }
            cdptr.cont = ncont;
            num_points += num_bound;
        }

        /* Output middle contour file if desired */
        if let Some(mm) = midmod.as_mut() {
            if imod_new_object(mm) != 0 {
                exit_error(b"Error creating new object in middle contour model");
            }
            for cdptr in &clist {
                let Some(cdup) = imod_contour_dup(&cdptr.cont) else {
                    exit_error(b"Duplicating contour")
                };
                if imod_object_add_contour(&mut mm.obj[0], cdup) < 0 {
                    exit_error(ADDCONT_ERROR);
                }
            }
            adjust_and_mesh_obj(&mut mm.obj[0], -999., &scale, showcont, "Original positions");
        }

        /* Now do smoothing if there are lambdas */
        if num_lambdas != 0 {
            /* Get the arrays for this number of points */
            let (mut lmat, mut ipiv, mut work, lwork);
            (fit_pts, yvec, lmat, ipiv, work, lwork) = prepare_tps(num_points);

            /* Load the points from contours into fit point array */
            let mut j = 0;
            for cdptr in &clist {
                for pt in &cdptr.cont.pts {
                    fit_pts[j] = *pt;
                    j += 1;
                }
            }
            set_tps_scaling(&fit_pts, num_points, &mut tps_scl);
            for indc in 0..num_lambdas as usize {
                printf!(
                    "Fitting thin plate spline, n = %d, log lambda = %.2f:",
                    CArg::Int(num_points as i64),
                    CArg::Dbl(lambda[indc] as f64)
                );
                fit_tps(
                    &fit_pts,
                    num_points,
                    lambda[indc],
                    &tps_scl,
                    &mut yvec,
                    &mut lmat,
                    &mut ipiv,
                    &mut work,
                    lwork,
                );
                if let Some(mm) = midmod.as_mut() {
                    if imod_new_object(mm) != 0 {
                        exit_error(b"Error creating new object in middle contour model");
                    }
                }

                /* Make duplicate contours with predicted data */
                let mut alpha = 0.0_f32;
                for cdptr in clist.iter_mut() {
                    cdptr.cdup = imod_contour_dup(&cdptr.cont);
                    if cdptr.cdup.is_none() {
                        exit_error(b"Duplicating contour");
                    }
                    for pt in 0..cdptr.cont.pts.len() {
                        let ptp = cdptr.cont.pts[pt];
                        zval = evaluate_tps(ptp.x, ptp.y, &fit_pts, num_points, &tps_scl, &yvec);
                        cdptr.cdup.as_mut().unwrap().pts[pt].z = zval;
                        alpha = (alpha as f64 + ((zval - ptp.z) as f64).abs()) as f32;
                        /* Last time, replace Z value in original data too */
                        if indc == num_lambdas as usize - 1 {
                            cdptr.cont.pts[pt].z = zval;
                        }
                    }

                    /* Add contour to model if writing model */
                    if let Some(mm) = midmod.as_mut() {
                        if imod_object_add_contour(
                            &mut mm.obj[indc + 1],
                            cdptr.cdup.clone().unwrap(),
                        ) < 0
                        {
                            exit_error(ADDCONT_ERROR);
                        }
                    }
                }
                printf!(
                    " Mean deviation = %.3f\n",
                    CArg::Dbl((alpha / num_points as f32) as f64)
                );
                if let Some(mm) = midmod.as_mut() {
                    adjust_and_mesh_obj(&mut mm.obj[indc + 1], lambda[indc], &scale, showcont, "");
                    if indc == 0 {
                        mm.obj[indc + 1].flags &= !IMOD_OBJFLAG_OFF;
                    }
                    mm.cindex.object = 1;
                }
            }
        }

        /* Close up the output model file now and exit if multiple lambdas */
        finish_output_model(
            midfile.as_deref(),
            midmod.as_mut(),
            &mut model,
            &scale,
            flipped,
            num_lambdas,
            restore_orientation,
        );

        if x_spacing != 0. && y_spacing != 0. {
            if (1. + ((ymax - ymin) / y_spacing) as f64) * (1. + ((xmax - xmin) / x_spacing) as f64)
                >= max_locations as f64
            {
                exit_error(b"Your spacings between grid points are too small for warpvol");
            }
        } else {
            /* Find minimum interval in Y */
            let mut minint = 10000000_i32;
            for i in 0..clist.len() - 1 {
                let iyval = clist[i + 1].yval - clist[i].yval;
                minint = b3dmin!(minint, iyval);
            }

            /* Start at "ideal" spacing factors and work down to factors of 1 */
            i = 20;
            while i >= 0 {
                x_spacing =
                    (minint as f64 / (1. + i as f64 * (x_space_fac as f64 - 1.) / 20.)) as f32;
                y_spacing =
                    (minint as f64 / (1. + i as f64 * (y_space_fac as f64 - 1.) / 20.)) as f32;
                if (1. + ((ymax - ymin) / y_spacing) as f64)
                    * (1. + ((xmax - xmin) / x_spacing) as f64)
                    < max_locations as f64
                {
                    break;
                }
                i -= 1;
            }

            if i < 0 {
                exit_error_fmt!(
                    "The minimum spacing between contours (%d) in Y is too small",
                    CArg::Int(minint as i64)
                );
            }
            printf!(
                "Minimum spacing between contours is %d\nSetting target spacings in X and Y to %.1f and %1.f\n",
                CArg::Int(minint as i64),
                CArg::Dbl(x_spacing as f64),
                CArg::Dbl(y_spacing as f64)
            );
        }

        (warps, ind_warp) = setup_warp_grid(
            xmin,
            xmax,
            ymin,
            ymax,
            xy_binning,
            z_binning,
            &model,
            &mut x_spacing,
            &mut y_spacing,
            &mut num_xloc,
            &mut num_yloc,
            &mut xcen,
            &mut ycen,
            &mut zcen,
        );
        if num_xloc < 3 || num_yloc < 3 {
            exit_error(
                b"The number of positions is too small; reduce spacing or increase range of contours",
            );
        }
        let mut cdptr = 0usize;
        let mut cdptr2 = 1usize;
        let mut indc = 1usize;
        zsum = 0.;
        num_loc = 0;
        for j in 0..num_yloc {
            let yloc = ymin + j as f32 * y_spacing;

            /* Advance to next contour pair if necessary */
            while yloc > clist[cdptr2].yval as f32 && indc < clist.len() {
                cdptr = cdptr2;
                cdptr2 = indc;
                indc += 1;
            }

            /* Compute inverse transform, so keep sign of z */
            for i in 0..num_xloc {
                let xloc = xmin + i as f32 * x_spacing;
                if interpolate_cont(&clist[cdptr].cont, xloc, &mut zval) >= 0
                    && interpolate_cont(&clist[cdptr2].cont, xloc, &mut zval2) >= 0
                {
                    let w = &mut warps[num_loc as usize];
                    w.xpos = xloc * xy_binning as f32 - xcen;
                    w.ypos = yloc * xy_binning as f32 - ycen;

                    /* Get the value from the TPS if it exists */
                    if num_lambdas != 0 {
                        w.dz = z_binning as f32
                            * evaluate_tps(xloc, yloc, &fit_pts, num_points, &tps_scl, &yvec);
                    } else {
                        let frac = (yloc - clist[cdptr].yval as f32)
                            / (clist[cdptr2].yval - clist[cdptr].yval) as f32;
                        w.dz = (z_binning as f64
                            * ((1. - frac as f64) * zval as f64 + (frac * zval2) as f64))
                            as f32;
                    }
                    zsum += w.dz as f64;
                    w.aa = -999.;
                    ind_warp[(i + j * num_xloc) as usize] = num_loc;
                    num_loc += 1;
                } else {
                    ind_warp[(i + j * num_xloc) as usize] = -1;
                }
            }
        }
    }

    /* DONE WITH BOTH KINDS OF MODELS - PROCESS WARP ARRAY FOR WARPING */

    zsum /= num_loc as f64;
    printf!("Mean Z height is %.1f\n", CArg::Dbl(zsum / z_binning as f64));
    let zmid: f32 = if one_surface != 0 || scattered != 0 {
        zsum as f32
    } else {
        zcen
    };
    for i in 0..num_loc as usize {
        warps[i].dz -= zmid;
    }
    let indwarp = |ind_warp: &[i32], a: i32, b: i32| ind_in_array(ind_warp, num_xloc, num_yloc, a, b);

    /* At each position, fit a plane to up to 9 points to get normal */
    for j in 0..num_yloc {
        for i in 0..num_xloc {
            let ind = ind_warp[(i + j * num_xloc) as usize];
            if ind < 0 {
                continue;
            }
            let mut ndat = 0usize;
            for iy in j - 1..=j + 1 {
                for ix in i - 1..=i + 1 {
                    let ind2 = indwarp(&ind_warp, ix, iy);
                    if ind2 >= 0 {
                        xp[ndat] = warps[ind2 as usize].xpos;
                        yp[ndat] = warps[ind2 as usize].ypos;
                        zp[ndat] = warps[ind2 as usize].dz;
                        ndat += 1;
                    }
                }
            }
            /* Fit a plane to the data and save the normal vector */
            if ndat >= 5 {
                let (mut aa, mut bb, mut cc) = (0.0_f32, 0.0_f32, 0.0_f32);
                ls_fit2(&xp, &yp, &zp, ndat as i32, &mut aa, &mut bb, Some(&mut cc));
                warps[ind as usize].aa = -aa;
                warps[ind as usize].bb = -bb;
            }
        }
    }

    /* Fill in normals from closest transformation */
    for j in 0..num_yloc {
        for i in 0..num_xloc {
            let ind = ind_warp[(i + j * num_xloc) as usize];
            if ind < 0 || warps[ind as usize].aa > -998. {
                continue;
            }
            let mut distmin = 1.0e30_f32;
            let mut indmin = -1_i32;
            let mut delind = (2. + b3dmax!(y_spacing / x_spacing, x_spacing / y_spacing)) as i32;
            while indmin < 0 {
                for iy in j - delind..=j + delind {
                    for ix in i - delind..=i + delind {
                        let ind2 = indwarp(&ind_warp, ix, iy);
                        if ind2 < 0 || warps[ind2 as usize].aa < -998. {
                            continue;
                        }
                        let dx = warps[ind as usize].xpos - warps[ind2 as usize].xpos;
                        let dy = warps[ind as usize].ypos - warps[ind2 as usize].ypos;
                        let dist = dx * dx + dy * dy;
                        if dist < distmin {
                            distmin = dist;
                            indmin = ind2;
                        }
                    }
                }
                // Fixed in translation (BUGS.md, `flattenwarp`): when no
                // position has a normal (none had 5 neighbours for a plane
                // fit), the source widens the search forever.  Defined: once
                // the search covers the whole grid, the normal is vertical.
                if indmin < 0 && delind > num_xloc + num_yloc {
                    break;
                }
                delind *= 2;
            }
            if indmin < 0 {
                warps[ind as usize].aa = 0.;
                warps[ind as usize].bb = 0.;
                continue;
            }
            warps[ind as usize].aa = warps[indmin as usize].aa;
            warps[ind as usize].bb = warps[indmin as usize].bb;
        }
    }

    /* Get array for lsqr and pointers to subarrays */
    let max_rows = 3 * num_loc + 20;
    let max_vals = max_rows * 4;
    let mut iwrk = vec![0_i32; (2 + max_rows + 2 * max_vals) as usize];
    let mut uu1 = vec![0.0_f64; max_rows as usize];
    let mut uu2 = vec![0.0_f64; max_rows as usize];
    let mut vv = vec![0.0_f64; max_rows as usize];
    let mut ww = vec![0.0_f64; max_rows as usize];
    let mut xx = vec![0.0_f64; (num_loc + 20) as usize];
    let mut sum_entries = vec![0.0_f32; (num_loc + 20) as usize];
    iwrk[0] = max_rows + 2;
    iwrk[1] = 2 + max_rows + max_vals;
    let mut num_rows = 0_i32;
    let mut num_tied = 0_i32;
    {
        // `ia = iwrk + 2; ja = iwrk + iwrk[0]; rwrk = (float *)iwrk + iwrk[1];`
        let (_, rest) = iwrk.split_at_mut(2);
        let (ia, rest) = rest.split_at_mut(max_rows as usize);
        let (ja, rwrk) = rest.split_at_mut(max_vals as usize);
        ia[0] = 1;

        /* Set up and solve equations for dx, dy */
        let dx_scale = (1. / (xy_binning as f64 * x_spacing as f64)) as f32;
        let dy_scale = (1. / (xy_binning as f64 * y_spacing as f64)) as f32;
        let dxy_scale =
            (1. / (xy_binning as f64 * ((x_spacing * y_spacing) as f64).sqrt())) as f32;
        for j in 0..num_yloc {
            for i in 0..num_xloc {
                let ind11 = ind_warp[(i + j * num_xloc) as usize];
                if ind11 < 0 {
                    continue;
                }
                let mut num_in_row = 0_i32;
                let ind21 = indwarp(&ind_warp, i + 1, j);
                let ind12 = indwarp(&ind_warp, i, j + 1);
                let ind22 = indwarp(&ind_warp, i + 1, j + 1);
                err = 0;
                if ind12 >= 0 && ind21 >= 0 && ind22 >= 0 {
                    /* If this is the lower left corner of a rectangle, get the average
                    normal and derive a stretch transformation from rotating normal
                    to vertical */
                    let w = |k: i32| warps[k as usize];
                    let aa = ((w(ind11).aa + w(ind21).aa + w(ind12).aa + w(ind22).aa) as f64 / 4.)
                        as f32;
                    let bb = ((w(ind11).bb + w(ind21).bb + w(ind12).bb + w(ind22).bb) as f64 / 4.)
                        as f32;
                    let stretch = (1. + (aa * aa) as f64 + (bb * bb) as f64).sqrt();
                    let mut axis = 0.0_f64;
                    if aa != 0. || bb != 0. {
                        axis = (bb as f64).atan2(aa as f64);
                    }
                    let cosphi = axis.cos();
                    let sinphi = axis.sin();
                    let cosphisq = cosphi * cosphi;
                    let sinphisq = sinphi * sinphi;
                    let a11 = (stretch * cosphisq + sinphisq) as f32;
                    let a12 = ((stretch - 1.) * cosphi * sinphi) as f32;
                    let a21 = a12;
                    let a22 = (stretch * sinphisq + cosphisq) as f32;

                    add_value_to_row(-dx_scale, ind11 + 1, &mut val_row, &mut icol_row, &mut num_in_row);
                    add_value_to_row(dx_scale, ind21 + 1, &mut val_row, &mut icol_row, &mut num_in_row);
                    uu1[num_rows as usize] = a11 as f64 - 1.;
                    uu2[num_rows as usize] = a21 as f64;
                    err += add_row_to_matrix(
                        &val_row, &icol_row, num_in_row, rwrk, ia, ja, &mut num_rows, max_rows,
                        max_vals,
                    );
                    num_in_row = 0;
                    add_value_to_row(-dy_scale, ind11 + 1, &mut val_row, &mut icol_row, &mut num_in_row);
                    add_value_to_row(dy_scale, ind12 + 1, &mut val_row, &mut icol_row, &mut num_in_row);
                    uu1[num_rows as usize] = a12 as f64;
                    uu2[num_rows as usize] = a22 as f64 - 1.;
                    err += add_row_to_matrix(
                        &val_row, &icol_row, num_in_row, rwrk, ia, ja, &mut num_rows, max_rows,
                        max_vals,
                    );
                    num_in_row = 0;
                    add_value_to_row(-dxy_scale, ind11 + 1, &mut val_row, &mut icol_row, &mut num_in_row);
                    add_value_to_row(dxy_scale, ind22 + 1, &mut val_row, &mut icol_row, &mut num_in_row);
                    uu1[num_rows as usize] = ((a11 as f64 - 1.) * x_spacing as f64
                        + (a12 * y_spacing) as f64)
                        * dxy_scale as f64
                        * xy_binning as f64;
                    uu2[num_rows as usize] = ((a22 as f64 - 1.) * y_spacing as f64
                        + (a21 * x_spacing) as f64)
                        * dxy_scale as f64
                        * xy_binning as f64;
                    err += add_row_to_matrix(
                        &val_row, &icol_row, num_in_row, rwrk, ia, ja, &mut num_rows, max_rows,
                        max_vals,
                    );
                } else {
                    let ind02 = indwarp(&ind_warp, i - 1, j + 1);
                    let ind01 = indwarp(&ind_warp, i - 1, j);
                    let ind00 = indwarp(&ind_warp, i - 1, j - 1);
                    let ind10 = indwarp(&ind_warp, i, j - 1);
                    let ind20 = indwarp(&ind_warp, i + 1, j - 1);

                    if !(ind01 >= 0 && ind02 >= 0 && ind12 >= 0)
                        && !(ind01 >= 0 && ind00 >= 0 && ind10 >= 0)
                        && !(ind10 >= 0 && ind20 >= 0 && ind21 >= 0)
                    {
                        /* If this is not part of any full rectangle then need to tie to
                        adjacent point(s) */
                        let on = |k: i32| if k >= 0 { 1.0_f64 } else { 0.0 };
                        let coef = (1. / (on(ind01) + on(ind10) + on(ind21) + on(ind12))) as f32;
                        add_value_to_row(-1., ind11 + 1, &mut val_row, &mut icol_row, &mut num_in_row);
                        for k in [ind01, ind10, ind21, ind12] {
                            if k >= 0 {
                                add_value_to_row(coef, k + 1, &mut val_row, &mut icol_row, &mut num_in_row);
                            }
                        }
                        uu1[num_rows as usize] = 0.;
                        uu2[num_rows as usize] = 0.;
                        err += add_row_to_matrix(
                            &val_row, &icol_row, num_in_row, rwrk, ia, ja, &mut num_rows,
                            max_rows, max_vals,
                        );
                        num_tied += 1;
                    }
                }
                if err != 0 {
                    exit_error_fmt!(
                        "Failed to make arrays big enough (numVals %d maxVals %d numRows %d maxRows %d)",
                        CArg::Int(ia[num_rows as usize] as i64),
                        CArg::Int(max_vals as i64),
                        CArg::Int(num_rows as i64),
                        CArg::Int(max_rows as i64)
                    );
                }
            }
        }

        /* Set the center point zero */
        let mut ind = ind_warp[(num_xloc / 2 + (num_yloc / 2) * num_xloc) as usize];
        if ind < 0 {
            let mut delind = 1;
            while delind <= b3dmax!(num_xloc / 2, num_yloc / 2) && ind < 0 {
                let mut j = num_yloc / 2 - delind;
                while j <= num_yloc / 2 + delind && ind < 0 {
                    let mut i = num_xloc / 2 - delind;
                    while i <= num_xloc / 2 + delind && ind < 0 {
                        ind = indwarp(&ind_warp, i, j);
                        i += 1;
                    }
                    j += 1;
                }
                delind += 1;
            }
        }

        let mut num_in_row = 0_i32;
        add_value_to_row(1., ind + 1, &mut val_row, &mut icol_row, &mut num_in_row);
        uu1[num_rows as usize] = 0.;
        uu2[num_rows as usize] = 0.;
        if add_row_to_matrix(
            &val_row, &icol_row, num_in_row, rwrk, ia, ja, &mut num_rows, max_rows, max_vals,
        ) != 0
        {
            exit_error(b"Failed to make arrays big enough");
        }

        printf!(
            "%d rows of data, %d variables (%d tied to neighbors)\n",
            CArg::Int(num_rows as i64),
            CArg::Int(num_loc as i64),
            CArg::Int(num_tied as i64)
        );

        /* Normalize columns, then solve for dx and dy */
        normalize_columns(rwrk, ia, ja, num_loc, num_rows, &mut sum_entries);
    }
    let itnlim = num_loc;
    let atol = 0.;
    let btol = 0.;
    let conlim = 1.0e7;
    let (mut istop, mut itndone) = (0_i32, 0_i32);
    let (mut anorm, mut acond, mut rnorm, mut arnorm, mut xnorm) = (0., 0., 0., 0., 0.);
    let (m, n) = (num_rows as usize, num_loc as usize);
    lsqr(
        m,
        n,
        |mode, x, y| sparse_prod(mode, num_rows, num_loc, x, y, &iwrk),
        0.,
        &mut uu1[..m],
        &mut vv[..n],
        &mut ww[..n],
        &mut xx[..n],
        None,
        atol,
        btol,
        conlim,
        itnlim,
        None,
        &mut istop,
        &mut itndone,
        &mut anorm,
        &mut acond,
        &mut rnorm,
        &mut arnorm,
        &mut xnorm,
    );
    report_lsqr("dx", istop, itndone, acond);

    /* Store the negative for an inverse transform */
    for i in 0..num_loc as usize {
        warps[i].dx = (-xx[i] / sum_entries[i] as f64) as f32;
    }
    lsqr(
        m,
        n,
        |mode, x, y| sparse_prod(mode, num_rows, num_loc, x, y, &iwrk),
        0.,
        &mut uu2[..m],
        &mut vv[..n],
        &mut ww[..n],
        &mut xx[..n],
        None,
        atol,
        btol,
        conlim,
        itnlim,
        None,
        &mut istop,
        &mut itndone,
        &mut anorm,
        &mut acond,
        &mut rnorm,
        &mut arnorm,
        &mut xnorm,
    );
    report_lsqr("dy", istop, itndone, acond);
    for i in 0..num_loc as usize {
        warps[i].dy = (-xx[i] / sum_entries[i] as f64) as f32;
    }

    let _ = imod_backup_file(&filename);
    let Some(mut fp) = ImodFile::open(&filename, "w") else {
        exit_error_fmt!("Opening output file %s", CArg::Str(&filename))
    };
    let mut text = c_format(
        "%d %d 1 %.2f %.2f 0. %.4f %.4f 1.\n",
        &[
            CArg::Int(num_xloc as i64),
            CArg::Int(num_yloc as i64),
            CArg::Dbl((xmin * xy_binning as f32 - xcen) as f64),
            CArg::Dbl((ymin * xy_binning as f32 - ycen) as f64),
            CArg::Dbl((x_spacing * xy_binning as f32) as f64),
            CArg::Dbl((y_spacing * xy_binning as f32) as f64),
        ],
    );
    for i in 0..num_loc as usize {
        let w = warps[i];
        /* Get the matrix that rotates the normal to vertical */
        let angle = 180.
            * (1. / (1. + (w.aa * w.aa) as f64 + (w.bb * w.bb) as f64).sqrt()).acos()
            / 3.14159;
        axisvec = Ipoint {
            x: w.bb,
            y: -w.aa,
            z: 0.,
        };
        imod_mat_id(&mut mat);
        let _ = imod_mat_rotate_vector(&mut mat, -angle, &axisvec);

        /* Then apply this inverse transform to the center position and get
        the actual shift needed to get to the displaced point */
        axisvec = Ipoint {
            x: w.xpos,
            y: w.ypos,
            z: 0.,
        };
        imod_mat_transform3d(&mat, &axisvec, &mut ptadd);
        let mdata = &mat.data;

        text.push_str(&c_format(
            "%.2f %.2f 0.\n%.5f %.5f %.5f %.2f\n%.5f %.5f %.5f %.2f\n%.5f %.5f %.5f %.2f\n",
            &[
                CArg::Dbl(w.xpos as f64),
                CArg::Dbl(w.ypos as f64),
                CArg::Dbl(mdata[0] as f64),
                CArg::Dbl(mdata[4] as f64),
                CArg::Dbl(mdata[8] as f64),
                CArg::Dbl((w.xpos + w.dx - ptadd.x) as f64),
                CArg::Dbl(mdata[1] as f64),
                CArg::Dbl(mdata[5] as f64),
                CArg::Dbl(mdata[9] as f64),
                CArg::Dbl((w.ypos + w.dy - ptadd.y) as f64),
                CArg::Dbl(mdata[2] as f64),
                CArg::Dbl(mdata[6] as f64),
                CArg::Dbl(mdata[10] as f64),
                CArg::Dbl((w.dz - ptadd.z) as f64),
            ],
        ));
    }
    let _ = fp.write_all(text.as_bytes());
    drop(fp);
    printf!(
        "%d warping transformations written\n",
        CArg::Int(num_loc as i64)
    );

    if let Some(patchfile) = patchfile {
        let patchfile = String::from_utf8_lossy(&patchfile).into_owned();
        let _ = imod_backup_file(&patchfile);
        let Some(mut fp) = ImodFile::open(&patchfile, "w") else {
            exit_error_fmt!("Opening patch output file %s", CArg::Str(&patchfile))
        };
        let mut text = c_format("%d positions\n", &[CArg::Int(num_loc as i64)]);
        for w in &warps[..num_loc as usize] {
            text.push_str(&c_format(
                "%d %d %d %.2f %.2f %.2f\n",
                &[
                    CArg::Int(b3dnint!(w.xpos + xcen) as i64),
                    CArg::Int(b3dnint!(w.ypos + ycen) as i64),
                    CArg::Int(b3dnint!(zcen) as i64),
                    CArg::Dbl(w.dx as f64),
                    CArg::Dbl(w.dy as f64),
                    CArg::Dbl(w.dz as f64),
                ],
            ));
        }
        let _ = fp.write_all(text.as_bytes());
    }
    let _ = ImodFile::Stdout.flush();
    exit(0);
}

/// Original `interpolateCont` (`flattenwarp.c:1121`, static).
///
/// The original contour interpolation routine interpolates between the
/// appropriate pair of points or returns -1 for X out of range.  It is still
/// used to get a zval from the smoothed contour data.
fn interpolate_cont(cont: &Icont, xval: f32, zval: &mut f32) -> i32 {
    let psize = cont.pts.len();
    if xval < cont.pts[0].x || xval > cont.pts[psize - 1].x {
        return -1;
    }
    let mut pt = 1usize;
    while pt + 1 < psize {
        if cont.pts[pt].x > xval {
            break;
        }
        pt += 1;
    }
    let frac = (xval - cont.pts[pt - 1].x) / (cont.pts[pt].x - cont.pts[pt - 1].x);
    // `frac * z` is a float product; `(1. - frac) * z` is double.
    *zval = ((frac * cont.pts[pt].z) as f64 + (1. - frac as f64) * cont.pts[pt - 1].z as f64)
        as f32;
    pt as i32 - 1
}

/// Original `interpolateCont2` (`flattenwarp.c:1139`, static).
///
/// New contour interpolation method simply interpolates if there are only a
/// few values within the window around x, or finds a smoothed value by
/// fitting a quadratic to nearby points.
fn interpolate_cont2(cont: &Icont, xval: f32, window: f32, zval: &mut f32) -> i32 {
    let psize = cont.pts.len();
    if xval < cont.pts[0].x || xval > cont.pts[psize - 1].x {
        return -1;
    }
    let mut num_pts = 0usize;
    let mut pt_after: i32 = -1;

    /* Count points in the window around xval, keep track of first one after */
    let mut pt = 0usize;
    while pt < psize {
        if pt_after < 0 && cont.pts[pt].x > xval {
            pt_after = pt as i32;
        }
        if cont.pts[pt].x as f64 > xval as f64 + window as f64 / 2. {
            break;
        }
        if cont.pts[pt].x as f64 >= xval as f64 - window as f64 / 2. {
            num_pts += 1;
        }
        pt += 1;
    }

    /* If there are fewer than 4 points in the window, just interpolate */
    if num_pts <= 3 {
        if pt_after < 0 {
            pt_after = psize as i32 - 1;
        }
        let pt = pt_after as usize;
        let frac = (xval - cont.pts[pt - 1].x) / (cont.pts[pt].x - cont.pts[pt - 1].x);
        *zval = ((frac * cont.pts[pt].z) as f64 + (1. - frac as f64) * cont.pts[pt - 1].z as f64)
            as f32;
        return pt as i32 - 1;
    }

    /* Allocate arrays if needed */
    let first_pt = pt - num_pts;
    let mut xx = vec![0.0_f32; num_pts];
    let mut yy = vec![0.0_f32; num_pts];
    let mut zz = vec![0.0_f32; num_pts];

    /* Load the arrays */
    for i in 0..num_pts {
        let pt = i + first_pt;
        xx[i] = cont.pts[pt].x - xval;
        yy[i] = xx[i] * xx[i];
        zz[i] = cont.pts[pt].z;
    }

    /* Quadratic fit with xval at origin gives z as the constant term */
    let (mut aa, mut bb) = (0.0_f32, 0.0_f32);
    ls_fit2(&xx, &yy, &zz, num_pts as i32, &mut aa, &mut bb, Some(zval));
    pt_after - 1
}

/// Original `indInArray` (`flattenwarp.c:1192`, static).
///
/// Return the index in the warp array of the given position, or -1 if the
/// position is outside the range.
fn ind_in_array(ind_warp: &[i32], num_xloc: i32, num_yloc: i32, i: i32, j: i32) -> i32 {
    if i < 0 || i >= num_xloc || j < 0 || j >= num_yloc {
        return -1;
    }
    ind_warp[(i + j * num_xloc) as usize]
}

/// Original `reportLsqr` (`flattenwarp.c:1199`, static).
fn report_lsqr(dd: &str, istop: i32, itndone: i32, acond: f64) {
    printf!(
        "Solution for %s: condition # %.4f, %d iterations\n",
        CArg::Str(dd),
        CArg::Dbl(acond),
        CArg::Int(itndone as i64)
    );
    if istop == 4 {
        printf!("The system appears to be ill conditioned\n");
    }
    if istop == 5 {
        printf!("The iteration limit was reached\n");
    }
}

/// Original `setupWarpGrid` (`flattenwarp.c:1210`, static).
///
/// Get number of positions in warp grid, refine spacing, allocate array.
/// The arrays are returned instead of through pointer arguments.
#[allow(clippy::too_many_arguments)]
fn setup_warp_grid(
    xmin: f32,
    xmax: f32,
    ymin: f32,
    ymax: f32,
    xy_binning: i32,
    z_binning: i32,
    model: &Imod,
    x_spacing: &mut f32,
    y_spacing: &mut f32,
    num_xloc: &mut i32,
    num_yloc: &mut i32,
    xcen: &mut f32,
    ycen: &mut f32,
    zcen: &mut f32,
) -> (Vec<WarpData>, Vec<i32>) {
    *num_xloc = (1. + ((xmax - xmin) / *x_spacing) as f64) as i32;
    *num_yloc = (1. + ((ymax - ymin) / *y_spacing) as f64) as i32;
    *x_spacing = (xmax - xmin) / (*num_xloc - 1) as f32;
    *y_spacing = (ymax - ymin) / (*num_yloc - 1) as f32;

    *xcen = ((xy_binning * model.xmax) as f64 / 2.) as f32;
    *ycen = ((xy_binning * model.ymax) as f64 / 2.) as f32;
    *zcen = ((z_binning * model.zmax) as f64 / 2.) as f32;

    let size = (*num_xloc * *num_yloc).max(0) as usize;
    let mut warps: Vec<WarpData> = Vec::new();
    let mut ind_warp: Vec<i32> = Vec::new();
    if warps.try_reserve_exact(size).is_err() || ind_warp.try_reserve_exact(size).is_err() {
        exit_error(b"Getting memory for warping data");
    }
    warps.resize(size, WarpData::default());
    ind_warp.resize(size, 0);
    (warps, ind_warp)
}

/// Original `prepareTPS` (`flattenwarp.c:1232`, static).
///
/// Allocate arrays for thin plate spline solution and return size of work.
/// Returns the point array, `yvec`, `lmat`, `ipiv`, `work` and `lwork`.
#[allow(clippy::type_complexity)]
fn prepare_tps(num_points: i32) -> (Vec<Ipoint>, Vec<f64>, Vec<f64>, Vec<i32>, Vec<f64>, i32) {
    let uplo = "U";
    let mut query = [0.0_f64; 1];
    let mut info = 0;
    let one = 1;
    let pp3 = num_points + 3;
    let fit_pts = vec![Ipoint::default(); num_points.max(0) as usize];
    let mut lmat = vec![0.0_f64; (pp3 * pp3) as usize];
    let mut yvec = vec![0.0_f64; pp3 as usize];
    let mut ipiv = vec![0_i32; pp3 as usize];
    dsysv(
        uplo, pp3, one, &mut lmat, pp3, &mut ipiv, &mut yvec, pp3, &mut query, -1, &mut info,
    );
    if info != 0 {
        exit_error_fmt!(
            "Error %d from dsysv workspace query",
            CArg::Int(info as i64)
        );
    }
    let lwork = b3dnint!(query[0]);
    let work = vec![0.0_f64; lwork.max(1) as usize];
    (fit_pts, yvec, lmat, ipiv, work, lwork)
}

/// Original `fitTPS` (`flattenwarp.c:1256`, static).
///
/// Solve for the thin plate spline.
#[allow(clippy::too_many_arguments)]
fn fit_tps(
    fit_pts: &[Ipoint],
    num_points: i32,
    lambda: f32,
    tps_scl: &Ipoint,
    yvec: &mut [f64],
    lmat: &mut [f64],
    ipiv: &mut [i32],
    work: &mut [f64],
    lwork: i32,
) {
    let one = 1;
    let pp3 = num_points + 3;
    let p = pp3 as usize;
    let np = num_points as usize;
    let mut alpha = 0.0_f64;
    let uplo = "U";
    let lambda = 10.0_f64.powf(lambda as f64) as f32;
    for row in 0..np {
        let ptp = fit_pts[row];

        /* Load the row at the far end of the matrix */
        lmat[row * p + p - 3] = 1.;
        lmat[row * p + p - 2] = (ptp.x * tps_scl.x) as f64;
        lmat[row * p + p - 1] = (ptp.y * tps_scl.y) as f64;
        lmat[p * (p - 3) + row] = 1.;
        lmat[p * (p - 2) + row] = (ptp.x * tps_scl.x) as f64;
        lmat[p * (p - 1) + row] = (ptp.y * tps_scl.y) as f64;
        yvec[row] = (ptp.z * tps_scl.z) as f64;

        for col in row + 1..np {
            /* Get r12, add to alpha, get U and put in matrix */
            let pt2p = fit_pts[col];
            let dx = ((ptp.x - pt2p.x) * tps_scl.x) as f64;
            let dy = ((ptp.y - pt2p.y) * tps_scl.y) as f64;
            let rr = (dx * dx + dy * dy).sqrt();
            alpha += rr;
            let mut uval = 0.;
            if rr > 0. {
                uval = rr * rr * rr.ln();
            }
            lmat[row * p + col] = uval;
            lmat[col * p + row] = uval;
        }
    }

    /* Fill zeros and diagonal */
    for row in p - 3..p {
        yvec[row] = 0.;
        for col in p - 3..p {
            lmat[row * p + col] = 0.;
        }
    }
    alpha *= 2. / (num_points * num_points) as f64;

    for i in 0..np {
        lmat[i * p + i] = lambda as f64 * alpha * alpha / num_points as f64;
    }

    /* Solve it */
    let mut info = 0;
    dsysv(
        uplo, pp3, one, lmat, pp3, ipiv, yvec, pp3, work, lwork, &mut info,
    );
    if info != 0 {
        exit_error_fmt!("Error %d from dsysv", CArg::Int(info as i64));
    }
}

/// Original `evaluateTPS` (`flattenwarp.c:1320`, static).
///
/// Evaluate the spline at the given position.
fn evaluate_tps(
    x: f32,
    y: f32,
    fit_pts: &[Ipoint],
    num_points: i32,
    tps_scl: &Ipoint,
    yvec: &[f64],
) -> f32 {
    let np = num_points as usize;
    let mut spsum = yvec[np]
        + yvec[np + 1] * x as f64 * tps_scl.x as f64
        + yvec[np + 2] * y as f64 * tps_scl.y as f64;
    for row in 0..np {
        let ptp = fit_pts[row];
        let dx = ((x - ptp.x) * tps_scl.x) as f64;
        let dy = ((y - ptp.y) * tps_scl.y) as f64;
        let rr = (dx * dx + dy * dy).sqrt();
        let mut uval = 0.;
        if rr > 0. {
            uval = rr * rr * rr.ln();
        }
        spsum += yvec[row] * uval;
    }
    spsum as f32 / tps_scl.z
}

/// Original `setTPSscaling` (`flattenwarp.c:1342`, static).
///
/// Find the bounds of the points and set the scaling factors to scale data to
/// a range of 1 on all axes.
fn set_tps_scaling(fit_pts: &[Ipoint], num_points: i32, tps_scl: &mut Ipoint) {
    let mut cont = new_contour();
    cont.pts = fit_pts[..num_points as usize].to_vec();
    let (minpt, maxpt) = imod_contour_get_bbox(Some(&cont)).unwrap_or_default();
    tps_scl.x = (1. / (maxpt.x - minpt.x) as f64) as f32;
    tps_scl.y = (1. / (maxpt.y - minpt.y) as f64) as f32;
    tps_scl.z = (1. / (maxpt.z - minpt.z) as f64) as f32;
}

/// Original `rotateModel` (`flattenwarp.c:1359`, static).
///
/// Rotate a model and associated meshes by -90 or 90 depending on the sign of
/// dir.
fn rotate_model(imod: &mut Imod, dir: i32) {
    let (ymax, zmax) = (imod.ymax, imod.zmax);
    for obj in imod.obj.iter_mut() {
        for cont in obj.cont.iter_mut() {
            for pt in cont.pts.iter_mut() {
                let tmp = pt.y;
                if dir < 0 {
                    pt.y = pt.z;
                    pt.z = ((ymax as f32 - tmp) as f64 - 1.) as f32;
                } else {
                    pt.y = ((zmax as f32 - pt.z) as f64 - 1.) as f32;
                    pt.z = tmp;
                }
            }
        }
        for mesh in obj.mesh.iter_mut() {
            for vert in mesh.vert.iter_mut() {
                let tmp = vert.y;
                if dir < 0 {
                    vert.y = vert.z;
                    vert.z = ((ymax as f32 - tmp) as f64 - 1.) as f32;
                } else {
                    vert.y = ((zmax as f32 - vert.z) as f64 - 1.) as f32;
                    vert.z = tmp;
                }
            }
        }
    }

    // `tmp = imod->ymax; imod->ymax = imod->zmax; imod->zmax = tmp;` through
    // the float `tmp`.
    let tmp = imod.ymax as f32;
    imod.ymax = imod.zmax;
    imod.zmax = tmp as i32;
}

/// Original `adjustAndMeshObj` (`flattenwarp.c:1402`, static).
///
/// Set object properties and mesh the contours.
fn adjust_and_mesh_obj(obj: &mut Iobj, lambda: f32, scale: &Ipoint, showcont: i32, prefix: &str) {
    let name = if lambda > -998. {
        c_format(
            "%sLog lambda %.2f",
            &[CArg::Str(prefix), CArg::Dbl(lambda as f64)],
        )
    } else {
        prefix.to_string()
    };
    obj.name.fill(0);
    let count = name.len().min(obj.name.len() - 1);
    obj.name[..count].copy_from_slice(&name.as_bytes()[..count]);

    imod_object_set_value(obj, IOBJ_FLAG_CLOSED, 0);
    obj.flags |= IMOD_OBJFLAG_TWO_SIDE;
    obj.mesh_param = imesh_params_new();
    if let Some(params) = obj.mesh_param.as_mut() {
        params.flags |= IMESH_MK_SKIP;
        analyze_prep_skin_obj(obj, 0, scale, None);
        if lambda > -999. && showcont == 0 {
            obj.flags |=
                IMOD_OBJFLAG_MESH | IMOD_OBJFLAG_NOLINE | IMOD_OBJFLAG_FILL | IMOD_OBJFLAG_OFF;
        }

        /* Tone down the lighting to allow bumps to show up better */
        obj.ambient = 116;
        obj.diffuse = 130;
        obj.specular = 68;
    }
}

/// Original `finishOutputModel` (`flattenwarp.c:1425`, static).
///
/// Set properties of output model and save it.
fn finish_output_model(
    midfile: Option<&[u8]>,
    midmod: Option<&mut Imod>,
    model: &mut Imod,
    scale: &Ipoint,
    flipped: i32,
    num_lambdas: i32,
    restore: i32,
) {
    let (Some(midfile), Some(midmod)) = (midfile, midmod) else {
        return;
    };
    let midfile = String::from_utf8_lossy(midfile).into_owned();
    midmod.xmax = model.xmax;
    midmod.ymax = model.ymax;
    midmod.zmax = model.zmax;
    midmod.flags = model.flags;
    midmod.ref_image = model.ref_image.clone();
    midmod.zscale = scale.z;
    let _ = imod_backup_file(&midfile);

    /* Restore to flip/rotation state of input model if requested */
    if restore != 0 {
        if flipped == 1 {
            imod_flip_yz(midmod);
        } else if flipped == 2 {
            rotate_model(midmod, 1);
        }
        if iobj_scat(model.obj[0].flags) != 0 && (model.flags & IMODF_FLIPYZ) != 0 {
            imod_flip_yz(midmod);
        }
    } else {
        /* Otherwise just toggle the flipped flag if appropriate */
        // Kept native (BUGS.md, `flattenwarp`): this toggles the input
        // model's flag, after `midmod->flags` was copied from it.
        if flipped == 1
            || (flipped != 1
                && iobj_scat(model.obj[0].flags) != 0
                && (model.flags & IMODF_FLIPYZ) != 0)
        {
            if model.flags & IMODF_FLIPYZ != 0 {
                model.flags &= !IMODF_FLIPYZ;
            } else {
                model.flags |= IMODF_FLIPYZ;
            }
        }
    }

    // Fixed in translation (BUGS.md, `flattenwarp`): the source passes the
    // model pointer `midmod` to "%s"; the file name is what it meant.
    let Some(mut fout) = ImodFile::open(&midfile, "wb") else {
        exit_error_fmt!(
            "Error opening file for middle contour model: %s",
            CArg::Str(&midfile)
        )
    };
    if imod_write(midmod, &mut fout).is_err() {
        exit_error(b"Error writing middle contour model");
    }
    drop(fout);
    if num_lambdas > 1 {
        let _ = ImodFile::Stdout.flush();
        exit(0);
    }
}
