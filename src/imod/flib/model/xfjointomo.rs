//! Translation of `IMOD/flib/model/xfjointomo.f`.
//!
//! XFJOINTOMO computes transforms for aligning tomograms of serial sections
//! from modeled features.  The model can have two kinds of features: contours
//! following trajectories on either side of a boundary, and contours with one
//! point on either side of a boundary.  The data from trajectories allows the
//! optimal spacing between the tomograms to be determined.  Written by David
//! Mastronarde, 10/24/06.
//!
//! The main program maps to [`xfjointomo`]; the `fortmodel` module arrays are
//! the [`FortModel`] that `readw_or_imod` fills.  The `real*4 (2,3,*)`
//! transforms are `[f32; 6]` in Fortran order (`a11 a21 a12 a22 dx dy`), and
//! `xr(19, idim)` is column major: `xr(i,j)` is `xr[(i-1) + 19*(j-1)]`.
//! `findxf` is the `findtransform.c` Fortran wrapper, whose allocation
//! failure prints its message and exits; `lsfit` is `lsFit`'s.

use crate::imod::flib::subrs::compat::gfortran_rt::{format_f, maxss, minss};
use crate::imod::flib::subrs::hvem::dopen::dopen;
use crate::imod::flib::subrs::hvem::get_nxyz::get_nxyz;
use crate::imod::flib::subrs::hvem::objtocont::objtocont;
use crate::imod::flib::subrs::hvem::parse_input_params::{
    exit_error, pip_get_in_out_file, pip_get_logical, pip_read_or_parse_options,
};
use crate::imod::flib::subrs::hvem::rdlist::parselist;
use crate::imod::flib::subrs::model::fortmodel::FortModel;
use crate::imod::flib::subrs::model::readw_or_imod::readw_or_imod;
use crate::imod::flib::subrs::model::scale_model::scale_model;
use crate::imod::flib::subrs::xfsubs::xflincom::xflincom;
use crate::imod::flib::subrs::xfsubs::xfrdall::xfrdall;
use crate::imod::flib::subrs::xfsubs::xfwrite::xfwrite;
use crate::imod::libcfshr::amat_to_rotmagstr::{amat_to_rotmag, rotmag_to_amat};
use crate::imod::libcfshr::b3dutil::{exit, fortran_string};
use crate::imod::libcfshr::findtransform::find_transform;
use crate::imod::libcfshr::linearxforms::{xfcopy, xfinvert, xfmult, xfunit};
use crate::imod::libcfshr::parse_params::{
    pip_get_boolean, pip_get_integer, pip_get_integer_array, pip_get_three_floats,
    pip_get_two_integers,
};
use crate::imod::libcfshr::pip_fwrap::pipgetstring_;
use crate::imod::libcfshr::simplestat::ls_fit;
use crate::imod::libimod::imodel_fwrap::getimodmaxes;
use std::io::{BufReader, Write};

/// `parameter (limsec=500, ...)` (`xfjointomo.f:19`).
const LIMSEC: i32 = 500;
/// `limslc=10000`.
const LIMSLC: i32 = 10000;
/// `idim=1000`.
const IDIM: i32 = 1000;
/// `msizexr=19`.
const MSIZEXR: i32 = 19;
/// `parameter (numOptions = 20)` (`xfjointomo.f:51`).
const XFJOINTOMO_NUM_OPTIONS: i32 = 20;
/// Fallback PIP table, the `options(1)` string (`xfjointomo.f:53-63`).
const XFJOINTOMO_OPTIONS: &str = "input:InputFile:FN:@foutput:FOutputFile:FN:@\
goutput:GOutputFile:FN:@edit:EditExistingFile:B:@\
slice:SliceTransforms:B:@join:JoinFileOrSizeXYZ:FN:@\
sizes:SizesOfSections:IA:@zvalues:ZValuesOfBoundaries:IA:@\
offset:OffsetOfJoin:IP:@binning:BinningOfJoin:I:@\
refsec:ReferenceSection:I:@transonly:TranslationOnly:B:@\
rottrans:RotationTranslation:B:@magrot:MagRotTrans:B:@\
boundaries:BoundariesToAnalyze:LI:@points:PointsToFit:IP:@\
gap:GapStartEndInc:FT:@objects:ObjectsToInclude:LI:@\
param:ParameterFile:PF:@help:usage:B:";

/// Original program `xfjointomo` (`xfjointomo.f:14`).
pub fn xfjointomo() {
    let mut fm = FortModel::default();
    let mut f = vec![[0.0_f32; 6]; LIMSLC as usize];
    let mut gtmp = [0.0_f32; 6];
    let mut g = vec![[0.0_f32; 6]; LIMSLC as usize];
    let mut xnat = vec![[0.0_f32; 6]; LIMSLC as usize];
    let mut xnatav = [0.0_f32; 6];
    let mut ginv = [0.0_f32; 6];
    let mut gav = [0.0_f32; 6];
    let mut nxyz = [0_i32; 3];
    let mut iobj_use = vec![0_i32; LIMSEC as usize];
    let mut isec_do = vec![0_i32; LIMSEC as usize];
    let mut iz_bound = vec![0_i32; LIMSEC as usize];
    let mut nz_sizes = vec![0_i32; LIMSEC as usize];
    let mut xx = vec![0.0_f32; LIMSEC as usize];
    let mut yy = vec![0.0_f32; LIMSEC as usize];
    let mut zz = vec![0.0_f32; LIMSEC as usize];
    let mut xr = vec![0.0_f32; (MSIZEXR * IDIM) as usize];
    let mut errmin = vec![0.0_f32; LIMSEC as usize];
    let mut errmaxmin = vec![0.0_f32; LIMSEC as usize];
    let mut gapmin = vec![0.0_f32; LIMSEC as usize];
    let mut modelfile = String::new();
    let mut xffile = String::new();
    let mut xgfile = String::new();
    let mut list_string = [b' '; 1024];
    let (mut imodobj, mut imodcont) = (0_i32, 0_i32);
    let mut ipntmax = 0_i32;
    let (mut devavg, mut devmax, mut devsd) = (0.0_f32, 0.0_f32, 0.0_f32);
    let mut slice_out: bool;
    let mut edit_old: bool;
    let mut limpnts: i32;
    let mut if_trans: i32;
    let mut if_ro_trans: i32;
    let mut if_mag_rot: i32;
    let mut join_bin: i32;
    let mut join_offset_x: i32;
    let mut join_offset_y: i32;
    let mut nfit: i32;
    let mut minfit: i32;
    let mut nobj_use: i32;
    let mut ierr: i32;
    let mut ierr2: i32;
    let mut num_boundaries: i32;
    let mut nz_sec_sum: i32;
    let mut gapstr: f32;
    let mut gapend: f32;
    let mut delgap: f32;
    let mut nf_write: i32;
    let mut num_sec_do: i32;
    let mut num_sizes: i32;
    let if_full_rpt: i32;
    let mut iref_sec: i32;
    let (mut num_opt_arg, mut num_non_opt_arg) = (0_i32, 0_i32);
    // `xr(i,j)`
    let xri = |i: i32, j: i32| ((i - 1) + MSIZEXR * (j - 1)) as usize;
    // Fortran `Iw` and `Fw.d` edit descriptors.
    let fmt_i = |value: i32, w: usize| -> String {
        let text = format!("{:>w$}", value);
        if text.len() > w { "*".repeat(w) } else { text }
    };
    let fmt_f = |value: f32, w: usize, d: usize| -> String { format_f(value as f64, w, d) };
    //
    //       defaults
    //
    limpnts = 4;
    if_trans = 0;
    if_ro_trans = 0;
    if_mag_rot = 0;
    join_bin = 1;
    join_offset_x = 0;
    join_offset_y = 0;
    nfit = 5;
    minfit = 2;
    gapstr = 0.;
    gapend = 0.;
    delgap = 0.;
    nobj_use = 0;
    edit_old = false;
    slice_out = false;
    if_full_rpt = 0;
    iref_sec = 0;
    fm.fm_mod_size_type = 2;
    //
    //       Pip startup: set error, parse options, check help
    //
    pip_read_or_parse_options(
        &[XFJOINTOMO_OPTIONS],
        XFJOINTOMO_NUM_OPTIONS,
        "xfjointomo",
        "ERROR: XFJOINTOMO - ",
        false,
        2,
        1,
        2,
        &mut num_opt_arg,
        &mut num_non_opt_arg,
    );

    let _ = pip_get_logical("SliceTransforms", &mut slice_out);
    if pip_get_in_out_file("InputFile", 1, " ", &mut modelfile, 320) != 0 {
        exit_error("NO INPUT MODEL FILE SPECIFIED");
    }
    if pip_get_in_out_file("FOutputFile", 2, " ", &mut xffile, 320) != 0 {
        exit_error("NO OUTPUT FILE SPECIFIED FOR F TRANSFORMS");
    }
    ierr = pip_get_in_out_file("GOutputFile", 3, " ", &mut xgfile, 320);
    if ierr != 0 && !slice_out {
        exit_error("NO OUTPUT FILE SPECIFIED FOR G TRANSFORMS TO BE USED IN FINISHJOIN");
    }
    if ierr == 0 && slice_out {
        exit_error("NO SECOND OUTPUT FILE ALLOWED WITH TRANSFORM OUTPUT FOR EVERY SLICE");
    }

    let exist = readw_or_imod(modelfile.trim_end_matches(' '), &mut fm);
    if !exist {
        exit_error("READING MODEL FILE");
    }
    //
    //       Get boundaries or Z sizes; if the latter, fill boundary array and
    //       compute a total Z size from section sizes
    //
    num_boundaries = 0;
    ierr = pip_get_integer_array(b"ZValuesOfBoundaries", &mut iz_bound, &mut num_boundaries, LIMSEC);
    num_sizes = 0;
    ierr2 = pip_get_integer_array(b"SizesOfSections", &mut nz_sizes, &mut num_sizes, LIMSEC);
    if ierr + ierr2 == 0 {
        exit_error("YOU CANNOT ENTER BOTH BOUNDARY Z VALUES AND SIZES OF SECTIONS");
    }
    if ierr + ierr2 == 2 {
        exit_error("YOU MUST ENTER EITHER BOUNDARY Z VALUES OR SIZES OF SECTIONS");
    }
    nz_sec_sum = 0;
    if ierr2 == 0 {
        num_boundaries = num_sizes - 1;
        nz_sec_sum = nz_sizes[0];
        for i in 1..=num_boundaries {
            iz_bound[(i - 1) as usize] = nz_sec_sum;
            nz_sec_sum += nz_sizes[i as usize];
        }
    }

    //
    //       Get the image size from the file, or from the model
    //       Z size must come from file if they want slice output, because we
    //       cannot trust that model was loaded on whole image
    //       Z size from file must match sum of section sizes if any
    //
    nxyz[0] = 0;
    get_nxyz(true, "JoinFileOrSizeXYZ", " ", 1, &mut nxyz);
    if nxyz[0] == 0 {
        if num_sizes == 0 && slice_out {
            exit_error("YOU MUST ENTER EITHER -join OR -sizes TO GET TRANSFORMS FOR EVERY SLICE");
        }
        let [nx, ny, nz] = &mut nxyz;
        let _ = getimodmaxes(nx, ny, nz);
    } else if nz_sec_sum > 0 && nz_sec_sum != nxyz[2] {
        exit_error("SUM OF SECTION SIZES DOES NOT EQUAL Z SIZE OF JOIN FILE");
    }
    let xcen = nxyz[0] as f32 / 2.;
    let ycen = nxyz[1] as f32 / 2.;
    //
    //       Figure out number of transforms to output and initialize transforms
    //
    nf_write = num_boundaries + 1;
    if slice_out {
        nf_write = nxyz[2];
    }
    if nf_write > LIMSLC {
        exit_error("TOO MANY TRANSFORMS FOR ARRAYS");
    }
    for i in 0..nf_write.max(0) as usize {
        xfunit(&mut f[i], 1.);
    }
    let _ = pip_get_logical("EditExistingFile", &mut edit_old);
    if edit_old {
        // open(1, file=xffile, status='old', form='formatted', err=21)
        if let Ok(file) = std::fs::File::open(xffile.trim_end_matches(' ')) {
            let mut unit1 = BufReader::new(file);
            let mut flist: Vec<[f32; 6]> = Vec::new();
            if xfrdall(&mut unit1, &mut flist).is_err() {
                // 99 call exitError('READING EXISTING TRANSFORM FILE')
                exit_error("READING EXISTING TRANSFORM FILE");
            }
            let nf_read = flist.len() as i32;
            for (i, transform) in flist.iter().enumerate().take(LIMSLC as usize) {
                f[i] = *transform;
            }
            if nf_write != nf_read {
                exit_error("NUMBER OF TRANSFORMS IN FILE DOES NOT MATCH NUMBER TO BE WRITTEN");
            }
        }
    }
    //
    //       Jump to here if the old file doesn't exist this time
    //       Get entries for adjusting transforms back to original data, and reject
    //       them if asked for slice output.
    //
    // 21
    ierr = pip_get_integer(b"ReferenceSection", &mut iref_sec);
    if iref_sec > num_boundaries + 1 {
        exit_error("REFERENCE SECTION NUMBER TOO LARGE");
    }
    ierr = pip_get_integer(b"BinningOfJoin", &mut join_bin);
    ierr2 = pip_get_two_integers(b"OffsetOfJoin", &mut join_offset_x, &mut join_offset_y);
    if slice_out && (ierr + ierr2 != 2 || iref_sec > 0) {
        exit_error(
            "IT MAKES NO SENSE TO ENTER -binning, -offset, OR -refsec WITH TRANSFORMS FOR EVERY SLICE",
        );
    }
    //
    let _ = pip_get_boolean(b"TranslationOnly", &mut if_trans);
    let _ = pip_get_boolean(b"RotationTranslation", &mut if_ro_trans);
    let _ = pip_get_boolean(b"MagRotTrans", &mut if_mag_rot);
    if if_trans + if_ro_trans + if_mag_rot > 1 {
        exit_error("ONLY ONE OF -transonly, -rottrans, AND -magrot CAN BE ENTERED");
    }
    if if_mag_rot != 0 {
        if_ro_trans = 2;
    }
    if if_ro_trans + if_trans > 0 {
        limpnts = if_ro_trans + 1;
    }
    //
    let _ = pip_get_two_integers(b"PointsToFit", &mut nfit, &mut minfit);
    if nfit < 2 || minfit < 2 {
        exit_error("NUMBER OF POINTS TO FIT MUST BE AT LEAST 2");
    }
    let _ = pip_get_three_floats(b"GapStartEndInc", &mut gapstr, &mut gapend, &mut delgap);
    let mut ngaps = 1_i32;
    if gapstr > gapend {
        exit_error("STARTING GAP SIZE MUST NOT BE BIGGER THAN ENDING SIZE");
    }
    if delgap > 0. {
        ngaps = ((gapend - gapstr) / delgap).round() as i32 + 1;
    }
    //
    if pipgetstring_(b"ObjectsToInclude", &mut list_string) == 0 {
        let _ = parselist(&fortran_string(&list_string), &mut iobj_use, &mut nobj_use);
        if nobj_use > LIMSEC {
            exit_error("OBJECT LIST TOO LARGE FOR ARRAYS");
        }
    }
    //
    num_sec_do = num_boundaries;
    for i in 1..=num_sec_do {
        isec_do[(i - 1) as usize] = i;
    }
    if pipgetstring_(b"BoundariesToAnalyze", &mut list_string) == 0 {
        let _ = parselist(&fortran_string(&list_string), &mut isec_do, &mut num_sec_do);
        if num_sec_do > LIMSEC {
            exit_error("BOUNDARRY LIST TOO LARGE FOR ARRAYS");
        }
    }
    for i in 1..=num_sec_do {
        let izsec = isec_do[(i - 1) as usize];
        if izsec < 1 || izsec > num_boundaries {
            exit_error("SECTION NUMBER OUT OF RANGE");
        }
    }
    //
    let mut unit1 = dopen(1, &xffile, "new", "f");
    scale_model(0, &mut fm);
    //
    //       Loop on the sections
    //
    // `122 format(25x,'Deviations ...',/,26x,'from above ...',/,30x,'Mean ...')`
    println!(
        "{:25}Deviations between transformed points extrapolated\n{:26}from above boundary and points extrapolated from below\n{:30}Mean     Max  @obj cont point     X-Y Position",
        "", "", ""
    );
    //
    // p_coord(k, object(index)) -- `object` holds a point number.
    let pz = |fm: &FortModel, index: i32| -> f32 {
        fm.p_coord[(fm.object[(index - 1) as usize] - 1) as usize][2]
    };
    for isec in 1..=num_sec_do {
        let izsec = isec_do[(isec - 1) as usize];
        let gapz = iz_bound[(izsec - 1) as usize] as f32 - 0.5;
        errmin[(isec - 1) as usize] = 1.0e10;
        if ngaps > 1 {
            // `123 format(/,'For boundary #',i4,':')`
            println!("\nFor boundary #{}:", fmt_i(isec_do[(isec - 1) as usize], 4));
        }
        //
        for igap in 1..=ngaps {
            let gapinc = gapstr + (igap - 1) as f32 * delgap;
            let mut npnts = 0_i32;
            for iobject in 1..=fm.max_mod_obj {
                objtocont(iobject, &fm.obj_color, &mut imodobj, &mut imodcont);
                let ninobj = fm.npt_in_obj[(iobject - 1) as usize];
                let ibase = fm.ibase_obj[(iobject - 1) as usize];
                //
                //             See if contour is in included object
                //
                let mut ifonlist = 0;
                for i in 0..nobj_use.max(0) as usize {
                    if imodobj == iobj_use[i] {
                        ifonlist = 1;
                    }
                }
                if ninobj > 1 && (ifonlist == 1 || nobj_use == 0) {
                    //
                    //               Set up indexes of limiting points in this section pair
                    //
                    let mut ipol = 1_i32;
                    let mut i_above_gap = 1_i32;
                    let mut last_in_sec = ninobj;
                    if pz(&fm, ninobj + ibase) < pz(&fm, 1 + ibase) {
                        ipol = -1;
                        i_above_gap = ninobj;
                        last_in_sec = 1;
                    }
                    let mut i_first_in_prev = i_above_gap;
                    //
                    //               find index of point above the gap, first point in previous
                    //               section and last point in current section
                    //
                    while i_above_gap <= ninobj && i_above_gap >= 1 {
                        if pz(&fm, i_above_gap + ibase) >= gapz {
                            break;
                        }
                        i_above_gap += ipol;
                    }
                    while izsec > 1 && i_first_in_prev <= ninobj && i_first_in_prev >= 1 {
                        if pz(&fm, i_first_in_prev + ibase)
                            >= iz_bound[(izsec - 2) as usize] as f32 - 0.5
                        {
                            break;
                        }
                        i_first_in_prev += ipol;
                    }
                    while izsec < num_boundaries && last_in_sec > 1 && last_in_sec < ninobj - 1 {
                        if pz(&fm, last_in_sec + ibase + ipol)
                            <= iz_bound[izsec as usize] as f32 - 0.5
                        {
                            break;
                        }
                        last_in_sec += ipol;
                    }
                    //
                    let num_above = ipol * (last_in_sec - i_above_gap) + 1;
                    let num_below = ipol * (i_above_gap - i_first_in_prev);
                    if num_above >= minfit && num_below >= minfit {
                        //
                        //                 If there are enough points, fit lines and extrapolate
                        //
                        npnts += 1;
                        xr[xri(6, npnts)] = iobject as f32;
                        xr[xri(7, npnts)] = i_above_gap as f32;
                        let mut ipt = i_above_gap - ipol;
                        let mut icol = 4_i32;
                        let mut coplanar = false;
                        for idir in [-1_i32, 1] {
                            let mut mfit = 0_i32;
                            let mut zmin = 1.0e30_f32;
                            let mut zmax = -zmin;
                            while mfit < nfit
                                && ipol * (ipt - i_first_in_prev) >= 0
                                && ipol * (last_in_sec - ipt) >= 0
                            {
                                let ipnt = fm.object[(ipt + ibase - 1) as usize].abs();
                                ipt += idir * ipol;
                                mfit += 1;
                                let point = fm.p_coord[(ipnt - 1) as usize];
                                xx[(mfit - 1) as usize] = point[0];
                                yy[(mfit - 1) as usize] = point[1];
                                zz[(mfit - 1) as usize] = point[2];
                                zmin = minss(zmin, zz[(mfit - 1) as usize]);
                                zmax = maxss(zmax, zz[(mfit - 1) as usize]);
                            }
                            //
                            //                   Do fits only if there is enough range in Z
                            if zmax - zmin > 0.1 {
                                let (mut slopex, mut bintx, mut ro) = (0.0_f32, 0.0_f32, 0.0_f32);
                                let (mut slopey, mut binty) = (0.0_f32, 0.0_f32);
                                ls_fit(&zz, &xx, mfit, &mut slopex, &mut bintx, &mut ro);
                                ls_fit(&zz, &yy, mfit, &mut slopey, &mut binty, &mut ro);
                                let extraz = gapz - idir as f32 * gapinc / 2.;
                                xr[xri(icol, npnts)] = slopex * extraz + bintx - xcen;
                                xr[xri(icol + 1, npnts)] = slopey * extraz + binty - ycen;
                            } else {
                                coplanar = true;
                            }
                            ipt = i_above_gap;
                            icol = 1;
                        }
                        //
                        //                 If either side was coplanar, drop the point-pair
                        if coplanar {
                            npnts -= 1;
                        }
                    } else if num_above == 1 && num_below == 1 {
                        //
                        //                 If one point on either side, use them alone
                        //
                        npnts += 1;
                        xr[xri(6, npnts)] = iobject as f32;
                        xr[xri(7, npnts)] = i_above_gap as f32;
                        let mut ipnt = fm.object[(i_above_gap + ibase - 1) as usize].abs();
                        xr[xri(1, npnts)] = fm.p_coord[(ipnt - 1) as usize][0] - xcen;
                        xr[xri(2, npnts)] = fm.p_coord[(ipnt - 1) as usize][1] - ycen;
                        ipnt = fm.object[(i_above_gap + ibase - ipol - 1) as usize].abs();
                        xr[xri(4, npnts)] = fm.p_coord[(ipnt - 1) as usize][0] - xcen;
                        xr[xri(5, npnts)] = fm.p_coord[(ipnt - 1) as usize][1] - ycen;
                    }
                }
                if npnts >= IDIM {
                    exit_error("TOO MANY POINTS BEING FIT FOR DATA MATRIX");
                }
            }
            //
            if npnts >= limpnts {
                //
                //             now if there are at least limpnts points, do regressions
                //
                if find_transform(
                    &mut xr,
                    MSIZEXR,
                    4,
                    npnts,
                    xcen,
                    ycen,
                    if_trans,
                    if_ro_trans,
                    2,
                    &mut gtmp,
                    &mut devavg,
                    &mut devsd,
                    &mut devmax,
                    &mut ipntmax,
                ) != 0
                {
                    println!("ERROR: Findxf function - Allocating array for matrices");
                    let _ = std::io::stdout().flush();
                    exit(1);
                }
                //
                //             save transform at minimum error, adjust for binning
                //
                if devavg < errmin[(isec - 1) as usize] {
                    errmin[(isec - 1) as usize] = devavg;
                    errmaxmin[(isec - 1) as usize] = devmax;
                    gapmin[(isec - 1) as usize] = gapinc;
                    let mut indf = izsec + 1;
                    if slice_out {
                        indf = iz_bound[(izsec - 1) as usize] + 1;
                    }
                    let fi = &mut f[(indf - 1) as usize];
                    xfcopy(&gtmp, fi);
                    fi[4] = join_bin as f32 * gtmp[4];
                    fi[5] = join_bin as f32 * gtmp[5];
                }
                //
                objtocont(
                    xr[xri(6, ipntmax)].round() as i32,
                    &fm.obj_color,
                    &mut imodobj,
                    &mut imodcont,
                );
                if ngaps > 1 {
                    // `121 format(i4, ' points, gap ',f6.1,2x,2f9.2,2i4,i6,2x,2f9.2)`
                    println!(
                        "{} points, gap {}  {}{}{}{}{}  {}{}",
                        fmt_i(npnts, 4),
                        fmt_f(gapinc, 6, 1),
                        fmt_f(devavg, 9, 2),
                        fmt_f(devmax, 9, 2),
                        fmt_i(imodobj, 4),
                        fmt_i(imodcont, 4),
                        fmt_i(xr[xri(7, ipntmax)].round() as i32, 6),
                        fmt_f(xr[xri(8, ipntmax)], 9, 2),
                        fmt_f(xr[xri(9, ipntmax)], 9, 2)
                    );
                } else {
                    // `125 format('boundary',i4, ',',i4,' points ',2f9.2,2i4,i6,2x,2f9.2)`
                    println!(
                        "boundary{},{} points {}{}{}{}{}  {}{}",
                        fmt_i(izsec, 4),
                        fmt_i(npnts, 4),
                        fmt_f(devavg, 9, 2),
                        fmt_f(devmax, 9, 2),
                        fmt_i(imodobj, 4),
                        fmt_i(imodcont, 4),
                        fmt_i(xr[xri(7, ipntmax)].round() as i32, 6),
                        fmt_f(xr[xri(8, ipntmax)], 9, 2),
                        fmt_f(xr[xri(9, ipntmax)], 9, 2)
                    );
                }
                if if_full_rpt != 0 {
                    // `124 format(' Obj Cont Point        position        ', ...)`
                    println!(
                        " Obj Cont Point        position        deviation vector   angle   magnitude"
                    );
                    for j in 1..=npnts {
                        objtocont(
                            xr[xri(6, j)].round() as i32,
                            &fm.obj_color,
                            &mut imodobj,
                            &mut imodcont,
                        );
                        // `128 format(2i4,f6.0,4f10.2,f9.0,f10.2)`
                        let mut line = format!("{}{}", fmt_i(imodobj, 4), fmt_i(imodcont, 4));
                        line.push_str(&fmt_f(xr[xri(7, j)], 6, 0));
                        for i in 8..=11 {
                            line.push_str(&fmt_f(xr[xri(i, j)], 10, 2));
                        }
                        line.push_str(&fmt_f(xr[xri(12, j)], 9, 0));
                        line.push_str(&fmt_f(xr[xri(13, j)], 10, 2));
                        println!("{line}");
                    }
                }
            } else if igap == 1 {
                // `130 format('WARNING: There are only',i2,' points across boundary',i4, ...)`
                println!(
                    "WARNING: There are only{} points across boundary{}; {} are needed to solve for the selected parameters",
                    fmt_i(npnts, 2),
                    fmt_i(isec, 4),
                    fmt_i(limpnts, 2)
                );
            }
        }
    }
    println!();
    for isec in 1..=num_sec_do {
        let k = (isec - 1) as usize;
        if errmin[k] < 1.0e9 {
            // `126 format('At boundary',i4,', best gap =',f6.1,' with error mean =',f9.2,
            //      ', max =',f9.2)`
            println!(
                "At boundary{}, best gap ={} with error mean ={}, max ={}",
                fmt_i(isec_do[k], 4),
                fmt_f(gapmin[k], 6, 1),
                fmt_f(errmin[k], 9, 2),
                fmt_f(errmaxmin[k], 9, 2)
            );
        }
    }
    //
    //       Write the F transforms
    //
    for i in 0..nf_write.max(0) as usize {
        if xfwrite(&mut unit1, &f[i]).is_err() {
            // 94 call exitError('WRITING OUT TRANSFORM FILE')
            exit_error("WRITING OUT TRANSFORM FILE");
        }
    }
    if !slice_out {
        //
        //         Now need to replicate xftoxg treatment of these transforms.  This
        //         a copy fo xftoxg with no angular majority vote analysis
        //         Start by computing cumulative transform
        //
        let mut unit2 = dopen(2, &xgfile, "new", "f");
        xfcopy(&f[0], &mut g[0]);
        for i in 1..nf_write.max(0) as usize {
            let previous = g[i - 1];
            xfmult(&f[i], &previous, &mut g[i]);
        }
        if iref_sec > 0 {
            //
            //           If doing reference section, just invert its transform
            //
            xfinvert(&g[(iref_sec - 1) as usize], &mut ginv);
        } else {
            //
            //           Otherwise compute cumulative transform, then convert to natural
            //           transforms
            //
            for i in 0..nf_write.max(0) as usize {
                // `amat_to_rotmag(g(1,1,i), xnat(1,1,i), xnat(1,2,i), xnat(2,1,i),
                // xnat(2,2,i))`: theta, ydtheta, smag, ydmag into elements 0, 2,
                // 1, 3, from `amat[0], amat[2], amat[1], amat[3]`.
                let (theta, ydtheta, smag, ydmag) =
                    amat_to_rotmag(g[i][0], g[i][2], g[i][1], g[i][3]);
                xnat[i][0] = theta;
                xnat[i][2] = ydtheta;
                xnat[i][1] = smag;
                xnat[i][3] = ydmag;
                xnat[i][4] = g[i][4];
                xnat[i][5] = g[i][5];
            }
            //
            //           average natural g's, convert back to xform, take inverse of average
            //
            xfunit(&mut xnatav, 0.);
            for i in 0..nf_write.max(0) as usize {
                let previous = xnatav;
                xflincom(&previous, 1., &xnat[i], 1. / nf_write as f32, &mut xnatav);
            }
            // `rotmag_to_amat(xnatav(1,1), xnatav(1,2), xnatav(2,1), xnatav(2,2), gav)`
            gav[..4].copy_from_slice(&rotmag_to_amat(
                xnatav[0], xnatav[2], xnatav[1], xnatav[3],
            ));
            gav[4] = xnatav[4];
            gav[5] = xnatav[5];
            xfinvert(&gav, &mut ginv);
        }
        //
        //         Compute the g transforms by multiplying by the inverse and adjust
        //         them for the offset when the join was done
        //
        for i in 0..nf_write.max(0) as usize {
            xfmult(&g[i], &ginv, &mut gtmp);
            gtmp[4] = gtmp[4] + (1. - gtmp[0]) * join_offset_x as f32
                - gtmp[2] * join_offset_y as f32;
            gtmp[5] = gtmp[5] - gtmp[1] * join_offset_x as f32
                + (1. - gtmp[3]) * join_offset_y as f32;
            if xfwrite(&mut unit2, &gtmp).is_err() {
                exit_error("WRITING OUT TRANSFORM FILE");
            }
        }
        drop(unit2);
    }
    drop(unit1);
    let _ = std::io::stdout().flush();
    exit(0);
}
